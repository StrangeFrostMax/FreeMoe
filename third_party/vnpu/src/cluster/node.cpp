/*
 * node.cpp - standalone vnpu-node server.
 *
 * Weights are stored once per node and reused, so token-generation traffic
 * only carries activations and results.  This is important for speculative
 * decoding across machines: weights never cross the network per token.
 */
#include "../internal.h"
#include "protocol.h"

#include <csignal>
#include <thread>
#include <unordered_map>

namespace vnpu {
namespace {

using net::socket_t;

struct WeightEntry {
    std::vector<float> B; /* canonical KxN */
    uint32_t K = 0;
    uint32_t N = 0;
};

struct NodeState {
    vnpu_device_t *device = nullptr;
    std::mutex weights_mutex;
    std::unordered_map<uint64_t, WeightEntry> weights;
    std::atomic<uint64_t> requests{0};
    std::atomic<uint64_t> errors{0};
    std::string node_name;
};

std::atomic<bool> g_node_stop{false};
std::mutex g_server_mutex;
std::vector<intptr_t> g_listen_fds;
std::vector<intptr_t> g_client_fds;

void register_listen(intptr_t fd)
{
    std::lock_guard<std::mutex> lock(g_server_mutex);
    g_listen_fds.push_back(fd);
}

void unregister_listen(intptr_t fd)
{
    std::lock_guard<std::mutex> lock(g_server_mutex);
    auto &v = g_listen_fds;
    v.erase(std::remove(v.begin(), v.end(), fd), v.end());
}

void register_client(intptr_t fd)
{
    std::lock_guard<std::mutex> lock(g_server_mutex);
    g_client_fds.push_back(fd);
}

void unregister_client(intptr_t fd)
{
    std::lock_guard<std::mutex> lock(g_server_mutex);
    auto &v = g_client_fds;
    v.erase(std::remove(v.begin(), v.end(), fd), v.end());
}

vnpu_status_t send_status(socket_t fd, uint32_t op, uint32_t request_id,
                          vnpu_status_t status)
{
    uint32_t encoded = status == VNPU_OK ? 0u :
                       static_cast<uint32_t>(-static_cast<int>(status));
    return net::send_message(fd, op, request_id, nullptr, 0, encoded);
}

bool shape_ok(uint64_t a, uint64_t b)
{
    if (a == 0 || b == 0) return false;
    if (a > std::numeric_limits<uint64_t>::max() / b) return false;
    return a * b <= net::kMaxPayload / sizeof(float);
}

void handle_client(intptr_t fd_raw, std::shared_ptr<NodeState> state,
                   vnpu_node_config_t config)
{
    socket_t fd = static_cast<socket_t>(fd_raw);
    vnpu_stream_t *stream = nullptr;
    if (vnpu_stream_create(state->device, 0, &stream) != VNPU_OK) {
        net::close_socket(fd);
        return;
    }
    bool authenticated = false;
    std::string expected_token = config.auth_token ? config.auth_token : "";
    for (;;) {
        net::Header h{};
        if (net::recv_header(fd, &h) != VNPU_OK) break;
        std::vector<uint8_t> payload;
        if (net::recv_payload(fd, h.payload_size, &payload) != VNPU_OK) break;
        state->requests.fetch_add(1);

        if (h.op == net::OP_HELLO) {
            net::Hello hello{};
            if (payload.size() < sizeof(hello)) {
                send_status(fd, h.op, h.request_id, VNPU_ERR_PROTOCOL);
                break;
            }
            std::memcpy(&hello, payload.data(), sizeof(hello));
            hello.token[sizeof(hello.token) - 1] = '\0';
            if (!expected_token.empty() &&
                expected_token != std::string(hello.token)) {
                send_status(fd, h.op, h.request_id, VNPU_ERR_AUTH);
                state->errors.fetch_add(1);
                break;
            }
            authenticated = true;
            send_status(fd, h.op, h.request_id, VNPU_OK);
            continue;
        }
        if (!authenticated) {
            send_status(fd, h.op, h.request_id, VNPU_ERR_AUTH);
            break;
        }

        switch (h.op) {
        case net::OP_PING:
            send_status(fd, h.op, h.request_id, VNPU_OK);
            break;
        case net::OP_CAPS: {
            net::Caps caps{};
            vnpu_device_info_t info{};
            vnpu_device_get_info(state->device, &info);
            caps.abi_version = VNPU_ABI_VERSION;
            caps.num_threads = info.num_threads;
            caps.features_lo = static_cast<uint32_t>(info.features);
            caps.features_hi = static_cast<uint32_t>(info.features >> 32);
            std::snprintf(caps.backend, sizeof(caps.backend), "%s",
                          info.backend);
            std::snprintf(caps.node_name, sizeof(caps.node_name), "%s",
                          state->node_name.c_str());
            net::send_message(fd, h.op, h.request_id, &caps, sizeof(caps));
            break;
        }
        case net::OP_STORE_WEIGHTS: {
            if (payload.size() < sizeof(net::StoreWeights)) {
                send_status(fd, h.op, h.request_id, VNPU_ERR_PROTOCOL);
                state->errors.fetch_add(1);
                break;
            }
            net::StoreWeights req{};
            std::memcpy(&req, payload.data(), sizeof(req));
            if (!shape_ok(req.K, req.N)) {
                send_status(fd, h.op, h.request_id, VNPU_ERR_INVALID_ARGUMENT);
                break;
            }
            size_t expected = sizeof(req) +
                              static_cast<size_t>(req.K) * req.N *
                              sizeof(float);
            if (payload.size() < expected) {
                send_status(fd, h.op, h.request_id, VNPU_ERR_PROTOCOL);
                break;
            }
            WeightEntry entry;
            entry.K = req.K;
            entry.N = req.N;
            entry.B.resize(static_cast<size_t>(req.K) * req.N);
            const float *src = reinterpret_cast<const float *>(
                payload.data() + sizeof(req));
            if (req.trans_b) {
                /* Incoming NxK -> canonical KxN. */
                for (uint32_t k = 0; k < req.K; ++k)
                    for (uint32_t n = 0; n < req.N; ++n)
                        entry.B[static_cast<size_t>(k) * req.N + n] =
                            src[static_cast<size_t>(n) * req.K + k];
            } else {
                std::memcpy(entry.B.data(), src,
                            entry.B.size() * sizeof(float));
            }
            req.key[sizeof(req.key) - 1] = '\0';
            {
                std::lock_guard<std::mutex> lock(state->weights_mutex);
                state->weights[req.weight_id] = std::move(entry);
            }
            send_status(fd, h.op, h.request_id, VNPU_OK);
            break;
        }
        case net::OP_MATMUL_F32: {
            if (payload.size() < sizeof(net::MatmulF32)) {
                send_status(fd, h.op, h.request_id, VNPU_ERR_PROTOCOL);
                break;
            }
            net::MatmulF32 req{};
            std::memcpy(&req, payload.data(), sizeof(req));
            if (!shape_ok(req.M, req.K) || !shape_ok(req.K, req.N) ||
                !shape_ok(req.M, req.N)) {
                send_status(fd, h.op, h.request_id, VNPU_ERR_INVALID_ARGUMENT);
                break;
            }
            size_t expected = sizeof(req) +
                              static_cast<size_t>(req.M) * req.K *
                              sizeof(float);
            if (payload.size() < expected) {
                send_status(fd, h.op, h.request_id, VNPU_ERR_PROTOCOL);
                break;
            }
            WeightEntry entry;
            bool found = false;
            {
                std::lock_guard<std::mutex> lock(state->weights_mutex);
                auto it = state->weights.find(req.weight_id);
                if (it != state->weights.end()) {
                    entry = it->second;
                    found = true;
                }
            }
            if (!found || entry.K != req.K || entry.N != req.N) {
                send_status(fd, h.op, h.request_id, VNPU_ERR_NOT_FOUND);
                break;
            }
            std::vector<float> C(static_cast<size_t>(req.M) * req.N, 0.0f);
            vnpu_matmul_f32_args_t args{};
            args.A = reinterpret_cast<const float *>(
                payload.data() + sizeof(req));
            args.B = entry.B.data();
            args.C = C.data();
            args.M = req.M;
            args.K = req.K;
            args.N = req.N;
            args.alpha = 1.0f;
            args.beta = 0.0f;
            args.unit = req.unit;
            vnpu_status_t rc = vnpu_matmul_f32(stream, &args);
            if (rc != VNPU_OK) {
                send_status(fd, h.op, h.request_id, rc);
                state->errors.fetch_add(1);
                break;
            }
            net::send_message(fd, h.op, h.request_id, C.data(),
                              C.size() * sizeof(float));
            break;
        }
        case net::OP_GET_STATS: {
            net::Stats s{};
            vnpu_device_status_t status{};
            vnpu_device_get_status(state->device, &status);
            s.commands_completed = status.commands_completed;
            s.kernel_time_ns = status.kernel_time_ns;
            s.flops = status.flops;
            s.bytes_read = status.bytes_read;
            s.bytes_written = status.bytes_written;
            s.tiles_dispatched = status.tiles_dispatched;
            s.tiles_culled = status.tiles_culled;
            (void)state;
            net::send_message(fd, h.op, h.request_id, &s, sizeof(s));
            break;
        }
        case net::OP_SHUTDOWN:
            send_status(fd, h.op, h.request_id, VNPU_OK);
            vnpu_node_stop();
            break;
        default:
            send_status(fd, h.op, h.request_id, VNPU_ERR_UNSUPPORTED);
            break;
        }
    }
    vnpu_stream_destroy(stream);
    unregister_client(fd_raw);
    net::close_socket(fd);
}

} /* anonymous namespace */
} /* namespace vnpu */

extern "C" {

VNPU_API void VNPU_CALL vnpu_node_stop(void)
{
    vnpu::g_node_stop.store(true);
    std::vector<intptr_t> fds;
    {
        std::lock_guard<std::mutex> lock(vnpu::g_server_mutex);
        fds.insert(fds.end(), vnpu::g_listen_fds.begin(),
                   vnpu::g_listen_fds.end());
        fds.insert(fds.end(), vnpu::g_client_fds.begin(),
                   vnpu::g_client_fds.end());
    }
    /* shutdown() interrupts accept()/recv() without risking a double close;
     * each server/client owns its own close at thread exit. */
    for (intptr_t raw : fds) {
        if (raw < 0) continue;
        vnpu::net::socket_t fd = static_cast<vnpu::net::socket_t>(raw);
        ::shutdown(fd, 2); /* SHUT_RDWR / SD_BOTH */
    }
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_node_run(const vnpu_node_config_t *config)
{
    if (!config) return VNPU_ERR_INVALID_ARGUMENT;
    return vnpu::guard([&]() -> vnpu_status_t {
        vnpu::net::global_init();
#if !defined(_WIN32)
        std::signal(SIGPIPE, SIG_IGN);
#endif
        {
            std::lock_guard<std::mutex> lock(vnpu::g_server_mutex);
            if (vnpu::g_listen_fds.empty())
                vnpu::g_node_stop.store(false);
        }
        vnpu_config_t devcfg;
        vnpu_config_init(&devcfg);
        devcfg.num_threads = config->num_threads;
        devcfg.node_name = config->node_name ? config->node_name : "vnpu-node";
        vnpu_device_t *device = nullptr;
        vnpu_status_t rc = vnpu_device_create(&devcfg, &device);
        if (rc != VNPU_OK) return rc;
        auto state = std::make_shared<vnpu::NodeState>();
        state->device = device;
        state->node_name = devcfg.node_name ? devcfg.node_name : "vnpu-node";
        intptr_t listen_raw = vnpu::net::listen_tcp(
            config->bind_address, config->port, 64);
        if (listen_raw < 0) {
            vnpu_device_destroy(device);
            return VNPU_ERR_NETWORK;
        }
        vnpu::register_listen(listen_raw);
        vnpu::net::socket_t listen_fd =
            static_cast<vnpu::net::socket_t>(listen_raw);
        vnpu_device_info_t info{};
        vnpu_device_get_info(device, &info);
        std::fprintf(stdout,
                     "vnpu-node %s listening on %s:%u | %s | %u threads | auth=%s\n",
                     state->node_name.c_str(),
                     config->bind_address ? config->bind_address : "0.0.0.0",
                     static_cast<unsigned>(config->port),
                     info.backend, info.num_threads,
                     config->auth_token && config->auth_token[0] ?
                         "token" : "open");
        std::fflush(stdout);

        std::vector<std::thread> clients;
        while (!vnpu::g_node_stop.load()) {
            struct sockaddr_storage addr{};
            socklen_t addrlen = sizeof(addr);
            vnpu::net::socket_t client = ::accept(
                listen_fd, reinterpret_cast<struct sockaddr *>(&addr),
                &addrlen);
            if (client == VNPU_INVALID_SOCKET) {
                if (vnpu::g_node_stop.load()) break;
                continue;
            }
            vnpu::net::set_timeouts(client, 30000);
#if !defined(_WIN32)
            int one = 1;
            setsockopt(client, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#endif
            intptr_t client_raw = static_cast<intptr_t>(client);
            vnpu::register_client(client_raw);
            clients.emplace_back(vnpu::handle_client, client_raw, state,
                                 *config);
        }
        vnpu::unregister_listen(listen_raw);
        vnpu::net::close_socket(listen_fd);
        for (auto &t : clients)
            if (t.joinable()) t.join();
        vnpu_device_destroy(device);
        vnpu::net::global_cleanup();
        return VNPU_OK;
    });
}

} /* extern "C" */
