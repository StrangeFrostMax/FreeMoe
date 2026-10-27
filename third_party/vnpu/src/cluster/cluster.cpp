/*
 * cluster.cpp - local multi-NPU cluster, remote vnpu-node workers and
 * DFlash/MTP-style speculative decoding coordinator.
 */
#include "../internal.h"
#include "protocol.h"

#include <thread>

namespace vnpu {
namespace {

std::atomic<uint32_t> g_request_id{1};

net::socket_t conn_fd(RemoteConn *conn)
{
    return conn ? static_cast<net::socket_t>(conn->fd) :
                  static_cast<net::socket_t>(-1);
}

vnpu_status_t status_from_header(const net::Header &h)
{
    if (h.status == 0) return VNPU_OK;
    if (h.status > 0 && h.status < 1000)
        return static_cast<vnpu_status_t>(-static_cast<int>(h.status));
    return VNPU_ERR_PROTOCOL;
}

vnpu_status_t hello_on_conn(RemoteConn *conn,
                            const vnpu_remote_node_t &remote,
                            const vnpu_cluster_t &cluster)
{
    net::Hello hello{};
    std::string token = remote.auth_token.empty() ?
                        cluster.auth_token : remote.auth_token;
    std::snprintf(hello.token, sizeof(hello.token), "%s", token.c_str());
    std::snprintf(hello.client_name, sizeof(hello.client_name), "%s",
                  cluster.name.c_str());
    hello.abi_version = VNPU_ABI_VERSION;
    hello.flags = cluster.config.flags;
    uint32_t id = g_request_id.fetch_add(1);
    vnpu_status_t rc = net::send_message(conn_fd(conn), net::OP_HELLO, id,
                                         &hello, sizeof(hello));
    if (rc != VNPU_OK) return rc;
    net::Header h{};
    rc = net::recv_header(conn_fd(conn), &h);
    if (rc != VNPU_OK) return rc;
    std::vector<uint8_t> payload;
    rc = net::recv_payload(conn_fd(conn), h.payload_size, &payload);
    if (rc != VNPU_OK) return rc;
    if (h.op != net::OP_HELLO || h.request_id != id) return VNPU_ERR_PROTOCOL;
    return status_from_header(h);
}

vnpu_status_t connect_remote_conn(vnpu_cluster_t *cluster,
                                  vnpu_remote_node_t *remote,
                                  RemoteConn *conn)
{
    intptr_t raw = net::connect_tcp(remote->host.c_str(), remote->port,
                                     cluster->config.request_timeout_ms);
    if (raw < 0) return VNPU_ERR_NETWORK;
    conn->fd = raw;
    net::socket_t fd = static_cast<net::socket_t>(raw);
    net::set_timeouts(fd, cluster->config.request_timeout_ms);
#if !defined(_WIN32)
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
#endif
    return hello_on_conn(conn, *remote, *cluster);
}

vnpu_status_t ensure_conn(vnpu_cluster_t *cluster,
                          vnpu_remote_node_t *remote,
                          RemoteConn *conn)
{
    if (conn->fd >= 0 && conn->healthy) return VNPU_OK;
    net::socket_t old = conn_fd(conn);
    if (old != static_cast<net::socket_t>(-1)) net::close_socket(old);
    conn->fd = -1;
    vnpu_status_t rc = connect_remote_conn(cluster, remote, conn);
    conn->healthy = (rc == VNPU_OK);
    return rc;
}

vnpu_status_t remote_store_weights(vnpu_cluster_t *cluster,
                                   vnpu_remote_node_t *remote,
                                   RemoteConn *conn,
                                   vnpu_weight_handle_t *weights,
                                   uint64_t *sent)
{
    std::lock_guard<std::mutex> lock(conn->mutex);
    vnpu_status_t rc = ensure_conn(cluster, remote, conn);
    if (rc != VNPU_OK) return rc;
    net::StoreWeights req{};
    req.weight_id = weights->id;
    req.K = weights->K;
    req.N = weights->N;
    req.trans_b = 0; /* canonical KxN */
    req.reserved = 0;
    std::snprintf(req.key, sizeof(req.key), "%s", weights->key.c_str());
    size_t data_bytes = weights->B.size() * sizeof(float);
    size_t total = sizeof(req) + data_bytes;
    if (total > net::kMaxPayload) return VNPU_ERR_INVALID_ARGUMENT;
    std::vector<uint8_t> payload(total);
    std::memcpy(payload.data(), &req, sizeof(req));
    std::memcpy(payload.data() + sizeof(req), weights->B.data(), data_bytes);
    uint32_t id = g_request_id.fetch_add(1);
    rc = net::send_message(conn_fd(conn), net::OP_STORE_WEIGHTS, id,
                           payload.data(), payload.size());
    if (rc != VNPU_OK) { conn->healthy = false; return rc; }
    net::Header h{};
    rc = net::recv_header(conn_fd(conn), &h);
    if (rc != VNPU_OK) { conn->healthy = false; return rc; }
    std::vector<uint8_t> response;
    rc = net::recv_payload(conn_fd(conn), h.payload_size, &response);
    if (rc != VNPU_OK) { conn->healthy = false; return rc; }
    *sent += total + sizeof(net::Header);
    cluster->bytes_received.fetch_add(sizeof(net::Header) + h.payload_size);
    if (h.op != net::OP_STORE_WEIGHTS || h.request_id != id)
        return VNPU_ERR_PROTOCOL;
    return status_from_header(h);
}

vnpu_status_t remote_matmul(vnpu_cluster_t *cluster,
                            vnpu_remote_node_t *remote,
                            RemoteConn *conn,
                            vnpu_weight_handle_t *weights,
                            const float *A, float *C,
                            uint32_t M, uint32_t K, uint32_t N,
                            uint32_t unit,
                            uint64_t *sent, uint64_t *received)
{
    std::lock_guard<std::mutex> lock(conn->mutex);
    vnpu_status_t rc = ensure_conn(cluster, remote, conn);
    if (rc != VNPU_OK) return rc;
    net::MatmulF32 req{};
    req.weight_id = weights->id;
    req.M = M; req.K = K; req.N = N;
    req.unit = unit;
    req.flags = 0;
    req.reserved = 0;
    size_t a_bytes = static_cast<size_t>(M) * K * sizeof(float);
    size_t total = sizeof(req) + a_bytes;
    if (total > net::kMaxPayload) return VNPU_ERR_INVALID_ARGUMENT;
    std::vector<uint8_t> payload(total);
    std::memcpy(payload.data(), &req, sizeof(req));
    std::memcpy(payload.data() + sizeof(req), A, a_bytes);
    uint32_t id = g_request_id.fetch_add(1);
    rc = net::send_message(conn_fd(conn), net::OP_MATMUL_F32, id,
                           payload.data(), payload.size());
    if (rc != VNPU_OK) { conn->healthy = false; return rc; }
    net::Header h{};
    rc = net::recv_header(conn_fd(conn), &h);
    if (rc != VNPU_OK) { conn->healthy = false; return rc; }
    std::vector<uint8_t> response;
    rc = net::recv_payload(conn_fd(conn), h.payload_size, &response);
    if (rc != VNPU_OK) { conn->healthy = false; return rc; }
    size_t expected = static_cast<size_t>(M) * N * sizeof(float);
    *sent += total + sizeof(net::Header);
    *received += h.payload_size + sizeof(net::Header);
    cluster->bytes_received.fetch_add(h.payload_size + sizeof(net::Header));
    if (h.op != net::OP_MATMUL_F32 || h.request_id != id)
        return VNPU_ERR_PROTOCOL;
    vnpu_status_t status = status_from_header(h);
    if (status != VNPU_OK) return status;
    if (response.size() < expected) return VNPU_ERR_PROTOCOL;
    std::memcpy(C, response.data(), expected);
    return VNPU_OK;
}

uint64_t weight_id_for(const std::string &key, uint32_t K, uint32_t N)
{
    std::string s = key + ":" + std::to_string(K) + "x" +
                    std::to_string(N);
    return net::fnv1a64(s);
}

/* Speculative sampling helpers. */
uint32_t argmax_logits(const float *logits, uint32_t vocab)
{
    uint32_t best = 0;
    float best_v = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < vocab; ++i) {
        float v = logits[i];
        if (!std::isfinite(v)) continue;
        if (v > best_v) { best_v = v; best = i; }
    }
    return best;
}

uint64_t xorshift64(uint64_t &state)
{
    uint64_t x = state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    state = x;
    return x;
}

int32_t sample_token(const float *logits, uint32_t vocab, float temperature,
                     uint64_t &rng)
{
    if (temperature <= 0.0f)
        return static_cast<int32_t>(argmax_logits(logits, vocab));
    float maxv = -std::numeric_limits<float>::infinity();
    for (uint32_t i = 0; i < vocab; ++i)
        if (std::isfinite(logits[i])) maxv = std::max(maxv, logits[i]);
    float inv_t = 1.0f / temperature;
    double sum = 0.0;
    std::vector<double> cdf(vocab);
    for (uint32_t i = 0; i < vocab; ++i) {
        float e = std::isfinite(logits[i]) ?
                  fast_expf((logits[i] - maxv) * inv_t) : 0.0f;
        sum += e;
        cdf[i] = sum;
    }
    if (sum <= 0.0) return static_cast<int32_t>(argmax_logits(logits, vocab));
    double u = static_cast<double>(xorshift64(rng) >> 11) /
               static_cast<double>(1ull << 53);
    double target = u * sum;
    uint32_t lo = 0, hi = vocab;
    while (lo < hi) {
        uint32_t mid = (lo + hi) / 2;
        if (cdf[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return static_cast<int32_t>(std::min<uint32_t>(lo, vocab - 1));
}

} /* anonymous namespace */

void cluster_register_builtins() { /* reserved for plugin registration */ }

} /* namespace vnpu */

/* ------------------------------------------------------------------------- */
/* Cluster C API                                                             */
/* ------------------------------------------------------------------------- */

extern "C" {

VNPU_API vnpu_status_t VNPU_CALL vnpu_cluster_create(
    const vnpu_cluster_config_t *config,
    vnpu_cluster_t **out_cluster)
{
    if (!out_cluster) return VNPU_ERR_INVALID_ARGUMENT;
    *out_cluster = nullptr;
    return vnpu::guard([&]() -> vnpu_status_t {
        vnpu::net::global_init();
        auto *cluster = new vnpu_cluster_t();
        if (config) cluster->config = *config;
        else vnpu_cluster_config_init(&cluster->config);
        if (cluster->config.abi_version == 0)
            cluster->config.abi_version = VNPU_ABI_VERSION;
        if (cluster->config.num_streams == 0) cluster->config.num_streams = 1;
        if (cluster->config.request_timeout_ms == 0)
            cluster->config.request_timeout_ms = 30000;
        cluster->name = cluster->config.cluster_name ?
                        cluster->config.cluster_name : "vnpu-cluster";
        cluster->auth_token = cluster->config.auth_token ?
                              cluster->config.auth_token : "";
        *out_cluster = cluster;
        return VNPU_OK;
    });
}

VNPU_API void VNPU_CALL vnpu_cluster_destroy(vnpu_cluster_t *cluster)
{
    if (!cluster) return;
    for (auto *stream : cluster->owned_streams) vnpu_stream_destroy(stream);
    for (auto *event : cluster->owned_events) vnpu_event_destroy(event);
    for (auto &remote : cluster->remotes) {
        for (auto &conn : remote->conns) {
            vnpu::net::socket_t fd =
                static_cast<vnpu::net::socket_t>(conn->fd);
            vnpu::net::close_socket(fd);
            conn->fd = -1;
        }
    }
    delete cluster;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_cluster_add_device(
    vnpu_cluster_t *cluster,
    vnpu_device_t *device,
    uint32_t weight)
{
    if (!cluster || !device) return VNPU_ERR_INVALID_ARGUMENT;
    return vnpu::guard([&]() -> vnpu_status_t {
        uint32_t streams = std::max<uint32_t>(1, cluster->config.num_streams);
        for (uint32_t i = 0; i < streams; ++i) {
            vnpu_stream_t *stream = nullptr;
            vnpu_status_t rc = vnpu_stream_create(device, 0, &stream);
            if (rc != VNPU_OK) return rc;
            cluster->owned_streams.push_back(stream);
            auto worker = std::make_unique<ClusterWorker>();
            worker->kind = ClusterWorker::LOCAL;
            worker->device = device;
            worker->stream = stream;
            worker->weight = weight ? weight : 1;
            cluster->workers.push_back(std::move(worker));
        }
        return VNPU_OK;
    });
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_cluster_add_remote(
    vnpu_cluster_t *cluster,
    const vnpu_remote_node_config_t *config,
    vnpu_remote_node_t **out_node)
{
    if (!cluster || !config || !config->host || !config->port)
        return VNPU_ERR_INVALID_ARGUMENT;
    if (out_node) *out_node = nullptr;
    return vnpu::guard([&]() -> vnpu_status_t {
        auto remote_owner = std::make_unique<vnpu_remote_node_t>();
        auto *remote = remote_owner.get();
        remote->host = config->host;
        remote->port = config->port;
        remote->auth_token = config->auth_token ? config->auth_token :
                             cluster->auth_token;
        remote->weight = config->weight ? config->weight : 1;
        remote->flags = config->flags;
        uint32_t connections = std::max<uint32_t>(1,
            cluster->config.num_streams);
        uint32_t succeeded = 0;
        for (uint32_t i = 0; i < connections; ++i) {
            auto conn = std::make_unique<RemoteConn>();
            vnpu_status_t rc = vnpu::connect_remote_conn(cluster, remote,
                                                         conn.get());
            if (rc != VNPU_OK) {
                vnpu::net::socket_t fd =
                    static_cast<vnpu::net::socket_t>(conn->fd);
                vnpu::net::close_socket(fd);
                conn->fd = -1;
                conn->healthy = false;
                continue;
            }
            auto worker = std::make_unique<ClusterWorker>();
            worker->kind = ClusterWorker::REMOTE;
            worker->remote = remote;
            worker->conn = conn.get();
            worker->connection_index = succeeded;
            worker->weight = remote->weight;
            remote->conns.push_back(std::move(conn));
            cluster->workers.push_back(std::move(worker));
            succeeded++;
        }
        if (!succeeded) return VNPU_ERR_NETWORK;
        remote->connected = true;
        cluster->remotes.push_back(std::move(remote_owner));
        if (out_node) *out_node = remote;
        return VNPU_OK;
    });
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_cluster_get_stats(
    const vnpu_cluster_t *cluster,
    vnpu_cluster_stats_t *stats)
{
    if (!cluster || !stats) return VNPU_ERR_INVALID_ARGUMENT;
    std::memset(stats, 0, sizeof(*stats));
    stats->local_jobs = cluster->local_jobs.load();
    stats->remote_jobs = cluster->remote_jobs.load();
    stats->total_jobs = cluster->total_jobs.load();
    stats->network_bytes_sent = cluster->bytes_sent.load();
    stats->network_bytes_received = cluster->bytes_received.load();
    stats->failed_jobs = cluster->failed_jobs.load();
    stats->culled_tiles = cluster->culled_tiles.load();
    stats->local_time_ns = static_cast<double>(cluster->local_time_ns.load());
    stats->remote_time_ns = static_cast<double>(cluster->remote_time_ns.load());
    stats->speedup_estimate =
        static_cast<double>(cluster->last_speedup_x1000.load()) / 1000.0;
    return VNPU_OK;
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_cluster_cache_weights_f32(
    vnpu_cluster_t *cluster,
    const char *key,
    const float *B,
    uint32_t K,
    uint32_t N,
    uint32_t trans_b,
    vnpu_weight_handle_t **out_handle)
{
    if (!cluster || !B || !K || !N || !out_handle || trans_b > 1)
        return VNPU_ERR_INVALID_ARGUMENT;
    *out_handle = nullptr;
    return vnpu::guard([&]() -> vnpu_status_t {
        std::string key_s = key && key[0] ? key :
            "weights-" + std::to_string(cluster->weights.size());
        auto *handle = new vnpu_weight_handle_t();
        handle->key = key_s;
        handle->id = vnpu::weight_id_for(key_s, K, N);
        handle->K = K;
        handle->N = N;
        handle->trans_b = trans_b;
        handle->B.resize(static_cast<size_t>(K) * N);
        if (trans_b) {
            for (uint32_t k = 0; k < K; ++k)
                for (uint32_t n = 0; n < N; ++n)
                    handle->B[static_cast<size_t>(k) * N + n] =
                        B[static_cast<size_t>(n) * K + k];
        } else {
            std::memcpy(handle->B.data(), B,
                        handle->B.size() * sizeof(float));
        }
        /* One STORE_WEIGHTS per remote node, not per connection. */
        std::vector<vnpu_remote_node_t *> unique;
        for (const auto &w : cluster->workers) {
            if (w->kind != ClusterWorker::REMOTE || !w->remote) continue;
            if (std::find(unique.begin(), unique.end(), w->remote) ==
                unique.end())
                unique.push_back(w->remote);
        }
        uint64_t sent = 0;
        for (auto *remote : unique) {
            if (remote->conns.empty()) continue;
            vnpu_status_t rc = vnpu::remote_store_weights(
                cluster, remote, remote->conns[0].get(), handle, &sent);
            if (rc != VNPU_OK) {
                cluster->failed_jobs.fetch_add(1);
                delete handle;
                cluster->bytes_sent.fetch_add(sent);
                return rc;
            }
        }
        cluster->bytes_sent.fetch_add(sent);
        cluster->weights.emplace_back(
            std::unique_ptr<vnpu_weight_handle_t>(handle));
        *out_handle = handle;
        return VNPU_OK;
    });
}

VNPU_API vnpu_status_t VNPU_CALL vnpu_cluster_matmul_f32(
    vnpu_cluster_t *cluster,
    vnpu_weight_handle_t *weights,
    const float *A,
    float *C,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint32_t unit)
{
    if (!cluster || !weights || !A || !C || !M || !K || !N)
        return VNPU_ERR_INVALID_ARGUMENT;
    if (weights->K != K || weights->N != N)
        return VNPU_ERR_INVALID_ARGUMENT;
    if (cluster->workers.empty()) return VNPU_ERR_DEVICE_ERROR;
    return vnpu::guard([&]() -> vnpu_status_t {
        size_t worker_count = cluster->workers.size();
        uint64_t total_weight = 0;
        for (const auto &w : cluster->workers)
            total_weight += std::max<uint32_t>(1, w->weight);
        if (total_weight == 0) total_weight = 1;
        struct Range { uint32_t start, rows; };
        std::vector<Range> ranges(worker_count);
        uint32_t cursor = 0;
        for (size_t i = 0; i < worker_count; ++i) {
            uint64_t w = std::max<uint32_t>(1,
                cluster->workers[i]->weight);
            uint32_t rows = static_cast<uint32_t>(
                (static_cast<uint64_t>(M) * w) / total_weight);
            if (i + 1 == worker_count) rows = M - cursor;
            if (cursor + rows > M) rows = M - cursor;
            ranges[i] = Range{cursor, rows};
            cursor += rows;
        }
        std::vector<vnpu_status_t> results(worker_count, VNPU_OK);
        std::vector<uint64_t> times(worker_count, 0);
        std::vector<std::thread> threads;
        uint64_t wall0 = vnpu::now_ns();
        for (size_t i = 0; i < worker_count; ++i) {
            if (ranges[i].rows == 0) continue;
            threads.emplace_back([&, i]() {
                ClusterWorker *w = cluster->workers[i].get();
                uint64_t t0 = vnpu::now_ns();
                vnpu_status_t rc;
                if (w->kind == ClusterWorker::LOCAL) {
                    vnpu_matmul_f32_args_t args{};
                    args.A = A + static_cast<size_t>(ranges[i].start) * K;
                    args.B = weights->B.data();
                    args.C = C + static_cast<size_t>(ranges[i].start) * N;
                    args.M = ranges[i].rows;
                    args.K = K;
                    args.N = N;
                    args.alpha = 1.0f;
                    args.beta = 0.0f;
                    args.unit = unit;
                    rc = vnpu_matmul_f32(w->stream, &args);
                    cluster->local_jobs.fetch_add(1);
                    cluster->local_time_ns.fetch_add(vnpu::now_ns() - t0);
                } else {
                    uint64_t sent = 0, received = 0;
                    rc = vnpu::remote_matmul(cluster, w->remote, w->conn,
                                             weights,
                                             A + static_cast<size_t>(
                                                 ranges[i].start) * K,
                                             C + static_cast<size_t>(
                                                 ranges[i].start) * N,
                                             ranges[i].rows, K, N, unit,
                                             &sent, &received);
                    cluster->bytes_sent.fetch_add(sent);
                    cluster->remote_jobs.fetch_add(1);
                    cluster->remote_time_ns.fetch_add(vnpu::now_ns() - t0);
                    {
                        std::lock_guard<std::mutex> lock(w->remote->mutex);
                        w->remote->jobs++;
                        w->remote->time_ns +=
                            static_cast<double>(vnpu::now_ns() - t0);
                        if (rc != VNPU_OK) w->remote->failures++;
                    }
                }
                times[i] = vnpu::now_ns() - t0;
                results[i] = rc;
                {
                    std::lock_guard<std::mutex> lock(w->mutex);
                    w->jobs++;
                    w->time_ns += static_cast<double>(times[i]);
                    if (rc != VNPU_OK) w->failures++;
                }
                if (rc != VNPU_OK) cluster->failed_jobs.fetch_add(1);
            });
        }
        for (auto &t : threads) if (t.joinable()) t.join();
        uint64_t wall = vnpu::now_ns() - wall0;
        cluster->last_wall_time_ns.store(wall);
        cluster->total_jobs.fetch_add(1);
        uint64_t sum_time = 0, max_time = 0;
        for (uint64_t t : times) {
            sum_time += t;
            max_time = std::max(max_time, t);
        }
        if (max_time > 0) {
            double speedup = static_cast<double>(sum_time) /
                             static_cast<double>(max_time);
            cluster->last_speedup_x1000.store(
                static_cast<uint64_t>(speedup * 1000.0));
        }
        for (auto rc : results)
            if (rc != VNPU_OK) return rc;
        return VNPU_OK;
    });
}

/* ------------------------------------------------------------------------- */
/* DFlash/MTP-style speculative decoding                                     */
/* ------------------------------------------------------------------------- */

VNPU_API vnpu_status_t VNPU_CALL vnpu_speculative_step(
    vnpu_cluster_t *cluster,
    const vnpu_speculative_config_t *config,
    const int32_t *context,
    uint32_t context_len,
    vnpu_logits_fn_t draft_fn,
    void *draft_user,
    vnpu_logits_fn_t verify_fn,
    void *verify_user,
    vnpu_speculative_result_t *result)
{
    if (!config || (!context && context_len) || !draft_fn || !verify_fn ||
        !result || !config->vocab)
        return VNPU_ERR_INVALID_ARGUMENT;
    if (config->gamma == 0 || config->gamma >= VNPU_MAX_TOKENS)
        return VNPU_ERR_INVALID_ARGUMENT;
    return vnpu::guard([&]() -> vnpu_status_t {
        std::memset(result, 0, sizeof(*result));
        const uint32_t gamma = config->gamma;
        const uint32_t vocab = config->vocab;
        std::vector<int32_t> sequence;
        sequence.reserve(context_len + gamma + 1);
        if (context) sequence.assign(context, context + context_len);
        std::vector<int32_t> drafts(gamma, 0);
        std::vector<float> draft_logits(vocab);
        uint64_t rng = static_cast<uint64_t>(config->seed ? config->seed : 1) ^
                       0x9e3779b97f4a7c15ull;
        uint64_t total0 = vnpu::now_ns();
        uint64_t draft_begin = total0;
        for (uint32_t j = 0; j < gamma; ++j) {
            vnpu_status_t rc = draft_fn(draft_user, sequence.data(),
                                        static_cast<uint32_t>(sequence.size()),
                                        context_len + j, 1,
                                        draft_logits.data(), vocab);
            if (rc != VNPU_OK) return rc;
            drafts[j] = vnpu::sample_token(draft_logits.data(), vocab,
                                           config->temperature, rng);
            sequence.push_back(drafts[j]);
        }
        double draft_ms = static_cast<double>(vnpu::now_ns() - draft_begin) /
                          1e6;
        std::vector<float> verify_logits(
            static_cast<size_t>(gamma + 1) * vocab, 0.0f);
        uint64_t verify_begin = vnpu::now_ns();
        vnpu_status_t rc = verify_fn(verify_user, sequence.data(),
                                     static_cast<uint32_t>(sequence.size()),
                                     context_len, gamma + 1,
                                     verify_logits.data(), vocab);
        if (rc != VNPU_OK) return rc;
        double verify_ms = static_cast<double>(vnpu::now_ns() -
                                               verify_begin) / 1e6;
        uint32_t accepted = 0;
        for (uint32_t j = 0; j < gamma; ++j) {
            /* v1 uses deterministic greedy acceptance.  Temperature affects
             * draft and bonus sampling; an exact stochastic acceptance test
             * would also need draft-model probabilities. */
            int32_t verifier_token = static_cast<int32_t>(
                vnpu::argmax_logits(
                    verify_logits.data() + static_cast<size_t>(j) * vocab,
                    vocab));
            if (verifier_token != drafts[j]) break;
            accepted++;
        }
        int32_t bonus = vnpu::sample_token(
            verify_logits.data() + static_cast<size_t>(accepted) * vocab,
            vocab, config->temperature, rng);
        uint32_t produced = accepted + 1;
        uint32_t limit = config->max_tokens ?
                         std::min<uint32_t>(config->max_tokens, produced) :
                         produced;
        for (uint32_t j = 0; j < limit; ++j)
            result->tokens[j] = (j < accepted) ? drafts[j] : bonus;
        result->accepted = std::min<uint32_t>(accepted, limit);
        result->produced = limit;
        for (uint32_t j = 0; j < limit; ++j) {
            if (result->tokens[j] == static_cast<int32_t>(config->eos_token)) {
                result->produced = j + 1;
                result->accepted = std::min<uint32_t>(result->accepted, j + 1);
                result->eos = 1;
                break;
            }
        }
        double total_ms = static_cast<double>(vnpu::now_ns() - total0) / 1e6;
        result->draft_time_ms = draft_ms;
        result->verify_time_ms = verify_ms;
        result->total_time_ms = total_ms;
        result->tokens_per_second = total_ms > 0.0 ?
            static_cast<double>(result->produced) * 1000.0 / total_ms : 0.0;
        double draft_step = draft_ms / static_cast<double>(gamma);
        double verify_cost = std::max(verify_ms, 1e-6);
        result->theoretical_speedup =
            static_cast<double>(result->produced) /
            (1.0 + static_cast<double>(gamma) * draft_step / verify_cost);
        if (cluster) cluster->total_jobs.fetch_add(2);
        return VNPU_OK;
    });
}

} /* extern "C" */
