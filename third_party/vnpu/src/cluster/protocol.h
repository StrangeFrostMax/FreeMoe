/*
 * protocol.h - compact binary protocol for vnpu-node and cluster clients.
 *
 * The wire format is intentionally simple and dependency-free.  All integers
 * are native endian in v1; run nodes on machines with the same architecture or
 * put an explicit little-endian converter here in v2.  Transport security is
 * expected to come from WireGuard, Tailscale, an SSH tunnel or mTLS proxy.
 */
#ifndef VNPU_PROTOCOL_H
#define VNPU_PROTOCOL_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "vnpu.h"

#if defined(_WIN32)
#  ifndef NOMINMAX
#    define NOMINMAX
#  endif
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  define VNPU_INVALID_SOCKET INVALID_SOCKET
#else
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <unistd.h>
#  define VNPU_INVALID_SOCKET (-1)
#endif

namespace vnpu {
namespace net {

#if defined(_WIN32)
using socket_t = SOCKET;
#else
using socket_t = int;
#endif

static constexpr uint32_t kMagic = 0x314e5056u; /* "VPN1" */
static constexpr uint32_t kVersion = 1u;
static constexpr uint64_t kMaxPayload = 1ull << 30;

enum Op : uint32_t {
    OP_HELLO          = 1,
    OP_CAPS           = 2,
    OP_PING           = 3,
    OP_STORE_WEIGHTS  = 4,
    OP_MATMUL_F32     = 5,
    OP_GET_STATS      = 6,
    OP_SHUTDOWN       = 7,
};

#pragma pack(push, 1)
struct Header {
    uint32_t magic;
    uint32_t version;
    uint32_t op;
    uint32_t request_id;
    uint32_t payload_size;
    uint32_t status;      /* 0 = OK, otherwise -vnpu_status_t */
    uint32_t flags;
    uint32_t reserved;
};

struct Hello {
    char token[64];
    char client_name[64];
    uint32_t abi_version;
    uint32_t flags;
};

struct Caps {
    uint32_t abi_version;
    uint32_t num_threads;
    uint32_t features_lo;
    uint32_t features_hi;
    char backend[64];
    char node_name[64];
};

struct StoreWeights {
    uint64_t weight_id;
    uint32_t K;
    uint32_t N;
    uint32_t trans_b;
    uint32_t reserved;
    char key[64];
    /* followed by K*N float values in canonical row-major KxN order */
};

struct MatmulF32 {
    uint64_t weight_id;
    uint32_t M;
    uint32_t K;
    uint32_t N;
    uint32_t unit;
    uint32_t flags;
    uint32_t reserved;
    /* followed by M*K float values */
};

struct Stats {
    uint64_t commands_completed;
    uint64_t kernel_time_ns;
    uint64_t flops;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t tiles_dispatched;
    uint64_t tiles_culled;
};
#pragma pack(pop)

static_assert(sizeof(Header) == 32, "protocol header must be 32 bytes");

void global_init();
void global_cleanup();
void close_socket(socket_t &fd);
intptr_t connect_tcp(const char *host, uint16_t port, uint32_t timeout_ms);
intptr_t listen_tcp(const char *bind_address, uint16_t port, int backlog);
bool set_timeouts(socket_t fd, uint32_t timeout_ms);
bool send_all(socket_t fd, const void *data, size_t bytes);
bool recv_all(socket_t fd, void *data, size_t bytes);
vnpu_status_t send_message(socket_t fd, uint32_t op, uint32_t request_id,
                           const void *payload, size_t payload_size,
                           uint32_t status = 0, uint32_t flags = 0);
vnpu_status_t recv_header(socket_t fd, Header *header);
vnpu_status_t recv_payload(socket_t fd, uint32_t payload_size,
                           std::vector<uint8_t> *payload);
std::string error_string(int err);
uint64_t fnv1a64(const std::string &s);

} /* namespace net */
} /* namespace vnpu */

#endif /* VNPU_PROTOCOL_H */
