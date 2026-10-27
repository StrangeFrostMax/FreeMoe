/*
 * net.cpp - portable socket helpers for the virtual NPU cluster protocol.
 */
#include "protocol.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>

#include <mutex>

namespace vnpu {
namespace net {
namespace {

std::once_flag g_once;
bool g_wsa_ok = false;

void ensure_init()
{
    std::call_once(g_once, []() {
#if defined(_WIN32)
        WSADATA wsa;
        g_wsa_ok = (WSAStartup(MAKEWORD(2, 2), &wsa) == 0);
#else
        g_wsa_ok = true;
#endif
    });
}

int last_error()
{
#if defined(_WIN32)
    return WSAGetLastError();
#else
    return errno;
#endif
}

bool would_block()
{
    int e = last_error();
#if defined(_WIN32)
    return e == WSAEWOULDBLOCK || e == WSAEINPROGRESS;
#else
    return e == EINPROGRESS || e == EWOULDBLOCK || e == EAGAIN;
#endif
}

} /* anonymous namespace */

void global_init() { ensure_init(); }

void global_cleanup()
{
#if defined(_WIN32)
    if (g_wsa_ok) WSACleanup();
#endif
}

std::string error_string(int err)
{
    char buf[256];
#if defined(_WIN32)
    std::snprintf(buf, sizeof(buf), "winsock error %d", err);
#else
    const char *s = std::strerror(err);
    std::snprintf(buf, sizeof(buf), "%s (%d)", s ? s : "socket error", err);
#endif
    return std::string(buf);
}

void close_socket(socket_t &fd)
{
    if (fd == VNPU_INVALID_SOCKET) return;
#if defined(_WIN32)
    closesocket(fd);
#else
    ::close(fd);
#endif
    fd = VNPU_INVALID_SOCKET;
}

bool set_timeouts(socket_t fd, uint32_t timeout_ms)
{
    if (fd == VNPU_INVALID_SOCKET || timeout_ms == 0) return false;
#if defined(_WIN32)
    DWORD ms = static_cast<DWORD>(timeout_ms);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO,
               reinterpret_cast<const char *>(&ms), sizeof(ms));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO,
               reinterpret_cast<const char *>(&ms), sizeof(ms));
#else
    struct timeval tv;
    tv.tv_sec = timeout_ms / 1000;
    tv.tv_usec = static_cast<long>((timeout_ms % 1000) * 1000);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
    return true;
}

intptr_t connect_tcp(const char *host, uint16_t port, uint32_t timeout_ms)
{
    ensure_init();
    if (!host || !host[0]) host = "127.0.0.1";
    char service[16];
    std::snprintf(service, sizeof(service), "%u", static_cast<unsigned>(port));
    struct addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    struct addrinfo *result = nullptr;
    int rc = getaddrinfo(host, service, &hints, &result);
    if (rc != 0 || !result) return -1;
    socket_t fd = VNPU_INVALID_SOCKET;
    for (auto *ai = result; ai; ai = ai->ai_next) {
        fd = socket(ai->ai_family, ai->ai_socktype, ai->ai_protocol);
        if (fd == VNPU_INVALID_SOCKET) continue;
        set_timeouts(fd, timeout_ms);
        if (::connect(fd, ai->ai_addr, static_cast<int>(ai->ai_addrlen)) == 0)
            break;
        int e = last_error();
        close_socket(fd);
        (void)e;
    }
    freeaddrinfo(result);
    if (fd == VNPU_INVALID_SOCKET) return -1;
#if !defined(_WIN32)
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY,
               reinterpret_cast<const char *>(&one), sizeof(one));
#endif
    return static_cast<intptr_t>(fd);
}

intptr_t listen_tcp(const char *bind_address, uint16_t port, int backlog)
{
    ensure_init();
    if (!bind_address || !bind_address[0]) bind_address = "0.0.0.0";
    char service[16];
    std::snprintf(service, sizeof(service), "%u", static_cast<unsigned>(port));
    struct addrinfo hints{};
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    hints.ai_protocol = IPPROTO_TCP;
    hints.ai_flags = AI_PASSIVE;
    struct addrinfo *result = nullptr;
    if (getaddrinfo(bind_address, service, &hints, &result) != 0 || !result)
        return -1;
    socket_t fd = socket(result->ai_family, result->ai_socktype,
                         result->ai_protocol);
    if (fd == VNPU_INVALID_SOCKET) {
        freeaddrinfo(result);
        return -1;
    }
    int one = 1;
#if defined(_WIN32)
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char *>(&one), sizeof(one));
#else
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    if (::bind(fd, result->ai_addr, static_cast<int>(result->ai_addrlen)) != 0 ||
        ::listen(fd, backlog) != 0) {
        close_socket(fd);
        freeaddrinfo(result);
        return -1;
    }
    freeaddrinfo(result);
    return static_cast<intptr_t>(fd);
}

bool send_all(socket_t fd, const void *data, size_t bytes)
{
    const char *p = static_cast<const char *>(data);
    while (bytes) {
#if defined(_WIN32)
        int n = ::send(fd, p, static_cast<int>(std::min<size_t>(bytes, 1 << 20)), 0);
#else
        ssize_t n = ::send(fd, p, std::min<size_t>(bytes, 1 << 20), MSG_NOSIGNAL);
#endif
        if (n > 0) {
            p += n;
            bytes -= static_cast<size_t>(n);
            continue;
        }
        if (n < 0 && (would_block() || last_error() == EINTR)) continue;
        return false;
    }
    return true;
}

bool recv_all(socket_t fd, void *data, size_t bytes)
{
    char *p = static_cast<char *>(data);
    while (bytes) {
#if defined(_WIN32)
        int n = ::recv(fd, p, static_cast<int>(std::min<size_t>(bytes, 1 << 20)), 0);
#else
        ssize_t n = ::recv(fd, p, std::min<size_t>(bytes, 1 << 20), 0);
#endif
        if (n > 0) {
            p += n;
            bytes -= static_cast<size_t>(n);
            continue;
        }
        if (n == 0) return false;
        if (n < 0 && (would_block() || last_error() == EINTR)) continue;
        return false;
    }
    return true;
}

vnpu_status_t send_message(socket_t fd, uint32_t op, uint32_t request_id,
                           const void *payload, size_t payload_size,
                           uint32_t status, uint32_t flags)
{
    if (fd == VNPU_INVALID_SOCKET || payload_size > kMaxPayload)
        return VNPU_ERR_NETWORK;
    Header h{};
    h.magic = kMagic;
    h.version = kVersion;
    h.op = op;
    h.request_id = request_id;
    h.payload_size = static_cast<uint32_t>(payload_size);
    h.status = status;
    h.flags = flags;
    h.reserved = 0;
    if (!send_all(fd, &h, sizeof(h))) return VNPU_ERR_NETWORK;
    if (payload_size && payload) {
        if (!send_all(fd, payload, payload_size)) return VNPU_ERR_NETWORK;
    }
    return VNPU_OK;
}

vnpu_status_t recv_header(socket_t fd, Header *header)
{
    if (!header || fd == VNPU_INVALID_SOCKET) return VNPU_ERR_NETWORK;
    if (!recv_all(fd, header, sizeof(*header))) return VNPU_ERR_NETWORK;
    if (header->magic != kMagic || header->version != kVersion)
        return VNPU_ERR_PROTOCOL;
    if (header->payload_size > kMaxPayload) return VNPU_ERR_PROTOCOL;
    return VNPU_OK;
}

vnpu_status_t recv_payload(socket_t fd, uint32_t payload_size,
                           std::vector<uint8_t> *payload)
{
    if (!payload) return VNPU_ERR_INVALID_ARGUMENT;
    payload->resize(payload_size);
    if (!payload_size) return VNPU_OK;
    if (!recv_all(fd, payload->data(), payload_size))
        return VNPU_ERR_NETWORK;
    return VNPU_OK;
}

uint64_t fnv1a64(const std::string &s)
{
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ull;
    }
    return h;
}

} /* namespace net */
} /* namespace vnpu */
