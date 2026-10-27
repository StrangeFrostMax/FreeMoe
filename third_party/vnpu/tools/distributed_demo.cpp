/*
 * distributed_demo.cpp - in-process two-node cluster and speculative decoding.
 *
 * The demo starts two vnpu-node servers on loopback, keeps a weight matrix
 * resident on each, then either splits one GEMM across both nodes or runs a
 * DFlash/MTP-style draft/verifier pipeline with each side on a different node.
 */
#include "vnpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

namespace {

struct Options {
    uint16_t port1 = 9450;
    uint16_t port2 = 9451;
    std::string token = "vnpu-demo";
    uint32_t K = 256;
    uint32_t N = 256;
    uint32_t M = 256;
    uint32_t iterations = 3;
    uint32_t unit = VNPU_UNIT_AUTO;
    uint32_t vocab = 256;
    uint32_t hidden = 128;
    uint32_t gamma = 4;
    uint32_t steps = 6;
    float temperature = 0.0f;
};

Options parse_options(int argc, char **argv)
{
    Options o;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };
        if (a == "--port1") o.port1 = static_cast<uint16_t>(std::atoi(next()));
        else if (a == "--port2") o.port2 = static_cast<uint16_t>(std::atoi(next()));
        else if (a == "--token") o.token = next();
        else if (a == "--shape") {
            unsigned m = 0, k = 0, n = 0;
            char c = 0;
            if (std::sscanf(next(), "%u%c%u%c%u", &m, &c, &k, &c, &n) == 5) {
                o.M = m; o.K = k; o.N = n;
            }
        } else if (a == "--iterations") o.iterations = std::atoi(next());
        else if (a == "--vocab") o.vocab = std::atoi(next());
        else if (a == "--hidden") o.hidden = std::atoi(next());
        else if (a == "--gamma") o.gamma = std::atoi(next());
        else if (a == "--steps") o.steps = std::atoi(next());
        else if (a == "--temperature") o.temperature = std::atof(next());
        else if (a == "--unit") {
            std::string u = next();
            if (u == "voodoo") o.unit = VNPU_UNIT_VOODOO;
            else if (u == "wildcat") o.unit = VNPU_UNIT_WILDCAT;
            else if (u == "cell") o.unit = VNPU_UNIT_CELL;
            else o.unit = VNPU_UNIT_AUTO;
        }
    }
    o.vocab = std::min<uint32_t>(std::max<uint32_t>(o.vocab, 16), 4096);
    o.hidden = std::min<uint32_t>(std::max<uint32_t>(o.hidden, 16), 2048);
    o.gamma = std::min<uint32_t>(std::max<uint32_t>(o.gamma, 1), 16);
    o.steps = std::min<uint32_t>(std::max<uint32_t>(o.steps, 1), 64);
    o.iterations = std::min<uint32_t>(std::max<uint32_t>(o.iterations, 1), 20);
    return o;
}

struct DistributedContext {
    Options opt;
    std::thread nodes[2];
    vnpu_cluster_t *cluster_all = nullptr;
    vnpu_cluster_t *cluster[2] = {nullptr, nullptr};
    vnpu_weight_handle_t *weight_all = nullptr;
    vnpu_weight_handle_t *weight[2] = {nullptr, nullptr};
    std::vector<float> W;
    bool running = false;

    void destroy_clusters()
    {
        if (cluster_all) { vnpu_cluster_destroy(cluster_all); cluster_all = nullptr; }
        for (int i = 0; i < 2; ++i) {
            if (cluster[i]) { vnpu_cluster_destroy(cluster[i]); cluster[i] = nullptr; }
        }
        weight_all = nullptr;
        weight[0] = weight[1] = nullptr;
    }

    bool create_clusters()
    {
        vnpu_cluster_config_t cc;
        vnpu_cluster_config_init(&cc);
        cc.num_streams = 1;
        cc.auth_token = opt.token.c_str();
        cc.cluster_name = "cluster-all";
        if (vnpu_cluster_create(&cc, &cluster_all) != VNPU_OK) return false;
        vnpu_remote_node_config_t r0{};
        r0.host = "127.0.0.1"; r0.port = opt.port1;
        r0.auth_token = opt.token.c_str(); r0.weight = 1;
        vnpu_remote_node_config_t r1 = r0;
        r1.port = opt.port2;
        if (vnpu_cluster_add_remote(cluster_all, &r0, nullptr) != VNPU_OK ||
            vnpu_cluster_add_remote(cluster_all, &r1, nullptr) != VNPU_OK)
            return false;
        for (int i = 0; i < 2; ++i) {
            cc.cluster_name = i == 0 ? "cluster-draft" : "cluster-verify";
            if (vnpu_cluster_create(&cc, &cluster[i]) != VNPU_OK) return false;
            vnpu_remote_node_config_t r = i == 0 ? r0 : r1;
            if (vnpu_cluster_add_remote(cluster[i], &r, nullptr) != VNPU_OK)
                return false;
        }
        return true;
    }

    bool cache_weights(uint32_t K, uint32_t N)
    {
        W.resize(static_cast<size_t>(K) * N);
        for (uint32_t k = 0; k < K; ++k) {
            for (uint32_t n = 0; n < N; ++n) {
                uint32_t x = (k * 1103515245u + n * 12345u + 7u);
                W[static_cast<size_t>(k) * N + n] =
                    static_cast<float>(static_cast<int>(x % 2001) - 1000) /
                    1000.0f;
            }
        }
        /* Add a deterministic diagonal signal so the toy logits head produces
         * a stable argmax instead of pure noise. */
        for (uint32_t n = 0; n < N; ++n)
            W[static_cast<size_t>(n % K) * N + n] += 2.0f;
        if (vnpu_cluster_cache_weights_f32(cluster_all, "all", W.data(),
                                           K, N, 0, &weight_all) != VNPU_OK)
            return false;
        if (vnpu_cluster_cache_weights_f32(cluster[0], "draft", W.data(),
                                           K, N, 0, &weight[0]) != VNPU_OK)
            return false;
        if (vnpu_cluster_cache_weights_f32(cluster[1], "verify", W.data(),
                                           K, N, 0, &weight[1]) != VNPU_OK)
            return false;
        return true;
    }

    bool start(const Options &o, uint32_t K, uint32_t N)
    {
        opt = o;
        opt.K = K;
        opt.N = N;
        vnpu_node_config_t nc[2];
        vnpu_node_config_init(&nc[0]);
        vnpu_node_config_init(&nc[1]);
        nc[0].port = opt.port1;
        nc[1].port = opt.port2;
        nc[0].bind_address = nc[1].bind_address = "127.0.0.1";
        nc[0].auth_token = nc[1].auth_token = opt.token.c_str();
        nc[0].node_name = "draft-node";
        nc[1].node_name = "verify-node";
        nc[0].num_threads = nc[1].num_threads = 0;
        nodes[0] = std::thread([nc0 = nc[0]]() mutable {
            vnpu_node_run(&nc0);
        });
        nodes[1] = std::thread([nc1 = nc[1]]() mutable {
            vnpu_node_run(&nc1);
        });
        running = true;
        bool ready = false;
        for (int attempt = 0; attempt < 40; ++attempt) {
            destroy_clusters();
            if (create_clusters()) { ready = true; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
        if (!ready) { stop(); return false; }
        if (!cache_weights(K, N)) { stop(); return false; }
        return true;
    }

    void stop()
    {
        if (!running) return;
        destroy_clusters();
        vnpu_node_stop();
        for (auto &t : nodes)
            if (t.joinable()) t.join();
        running = false;
    }

    ~DistributedContext() { stop(); }
};

struct LogitsUser {
    vnpu_cluster_t *cluster;
    vnpu_weight_handle_t *weights;
    uint32_t hidden;
    uint32_t vocab;
};

vnpu_status_t remote_logits(void *user,
                            const int32_t *tokens,
                            uint32_t num_tokens,
                            uint32_t first_position,
                            uint32_t num_positions,
                            float *logits,
                            uint32_t vocab)
{
    auto *u = static_cast<LogitsUser *>(user);
    if (!u || !u->cluster || !u->weights || !tokens || !num_tokens ||
        !num_positions || !logits || vocab != u->vocab)
        return VNPU_ERR_INVALID_ARGUMENT;
    std::vector<float> A(static_cast<size_t>(num_positions) * u->hidden,
                         0.0f);
    for (uint32_t p = 0; p < num_positions; ++p) {
        /* Logits at sequence position q are a function of the previous token.
         * first_position is the first q being predicted in this callback. */
        size_t token_index = first_position > 0
            ? std::min<size_t>(num_tokens - 1,
                               static_cast<size_t>(first_position) - 1 + p)
            : std::min<size_t>(num_tokens - 1, p);
        int32_t t = tokens[token_index];
        uint32_t idx = static_cast<uint32_t>(t < 0 ? -t : t) % u->hidden;
        A[static_cast<size_t>(p) * u->hidden + idx] = 1.0f;
    }
    return vnpu_cluster_matmul_f32(u->cluster, u->weights, A.data(), logits,
                                   num_positions, u->hidden, u->vocab,
                                   VNPU_UNIT_AUTO);
}

uint32_t rng = 0xdecafbadu;
uint32_t xs()
{
    uint32_t x = rng;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng = x;
    return x;
}
float rndf()
{
    return static_cast<float>(static_cast<int>(xs() % 2000) - 1000) /
           1000.0f;
}

double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

void print_cluster_stats(const char *label, vnpu_cluster_t *cluster)
{
    vnpu_cluster_stats_t s{};
    vnpu_cluster_get_stats(cluster, &s);
    std::printf("%s stats: jobs=%llu local=%llu remote=%llu "
                "sent=%llu B recv=%llu B failed=%llu speedup=%.2fx\n",
                label,
                static_cast<unsigned long long>(s.total_jobs),
                static_cast<unsigned long long>(s.local_jobs),
                static_cast<unsigned long long>(s.remote_jobs),
                static_cast<unsigned long long>(s.network_bytes_sent),
                static_cast<unsigned long long>(s.network_bytes_received),
                static_cast<unsigned long long>(s.failed_jobs),
                s.speedup_estimate);
}

} /* namespace */

int vnpu_run_cluster_demo(int argc, char **argv)
{
    Options opt = parse_options(argc, argv);
    std::printf("=== Two-node virtual NPU cluster demo ===\n");
    std::printf("shape=%ux%ux%u ports=%u,%u token=%s\n",
                opt.M, opt.K, opt.N, opt.port1, opt.port2,
                opt.token.c_str());
    DistributedContext ctx;
    if (!ctx.start(opt, opt.K, opt.N)) {
        std::fprintf(stderr, "failed to start demo nodes\n");
        return 1;
    }
    const uint32_t M = opt.M, K = opt.K, N = opt.N;
    std::vector<float> A(static_cast<size_t>(M) * K);
    std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
    std::vector<float> Ref(static_cast<size_t>(M) * N, 0.0f);
    for (auto &x : A) x = rndf();
    /* Reference, kept small enough for a demo. */
    for (uint32_t m = 0; m < M; ++m)
        for (uint32_t n = 0; n < N; ++n) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < K; ++k)
                acc += A[static_cast<size_t>(m) * K + k] *
                       ctx.W[static_cast<size_t>(k) * N + n];
            Ref[static_cast<size_t>(m) * N + n] = acc;
        }
    /* Local single-device timing for comparison. */
    double local_ms = 0.0;
    {
        vnpu_config_t cfg;
        vnpu_config_init(&cfg);
        vnpu_device_t *device = nullptr;
        vnpu_stream_t *stream = nullptr;
        if (vnpu_device_create(&cfg, &device) == VNPU_OK &&
            vnpu_stream_create(device, 0, &stream) == VNPU_OK) {
            for (uint32_t i = 0; i < 2; ++i) {
                vnpu_matmul_f32_args_t a{};
                a.A = A.data(); a.B = ctx.W.data(); a.C = C.data();
                a.M = M; a.K = K; a.N = N;
                a.alpha = 1.0f; a.beta = 0.0f; a.unit = opt.unit;
                vnpu_matmul_f32(stream, &a);
            }
            double t0 = now_ms();
            for (uint32_t i = 0; i < opt.iterations; ++i) {
                vnpu_matmul_f32_args_t a{};
                a.A = A.data(); a.B = ctx.W.data(); a.C = C.data();
                a.M = M; a.K = K; a.N = N;
                a.alpha = 1.0f; a.beta = 0.0f; a.unit = opt.unit;
                vnpu_matmul_f32(stream, &a);
            }
            local_ms = (now_ms() - t0) /
                       static_cast<double>(opt.iterations);
            vnpu_stream_destroy(stream);
            vnpu_device_destroy(device);
        }
    }
    double cluster_ms = 0.0;
    vnpu_status_t rc = VNPU_OK;
    for (uint32_t i = 0; i < 2; ++i)
        rc = vnpu_cluster_matmul_f32(ctx.cluster_all, ctx.weight_all,
                                     A.data(), C.data(), M, K, N, opt.unit);
    if (rc == VNPU_OK) {
        double t0 = now_ms();
        for (uint32_t i = 0; i < opt.iterations; ++i) {
            rc = vnpu_cluster_matmul_f32(ctx.cluster_all, ctx.weight_all,
                                         A.data(), C.data(), M, K, N,
                                         opt.unit);
            if (rc != VNPU_OK) break;
        }
        cluster_ms = (now_ms() - t0) /
                     static_cast<double>(opt.iterations);
    }
    bool correct = false;
    if (rc == VNPU_OK) {
        correct = true;
        for (size_t i = 0; i < C.size(); ++i) {
            if (std::fabs(C[i] - Ref[i]) >
                5e-3f + 5e-3f * std::max(std::fabs(C[i]),
                                         std::fabs(Ref[i]))) {
                correct = false;
                break;
            }
        }
    }
    uint64_t flops = 2ull * M * K * N;
    std::printf("local  : %8.4f ms  %8.3f GFLOP/s\n", local_ms,
                local_ms > 0 ? static_cast<double>(flops) /
                                   (local_ms * 1e6) : 0.0);
    std::printf("cluster: %8.4f ms  %8.3f GFLOP/s  speedup=%.2fx\n",
                cluster_ms,
                cluster_ms > 0 ? static_cast<double>(flops) /
                                     (cluster_ms * 1e6) : 0.0,
                cluster_ms > 0 ? local_ms / cluster_ms : 0.0);
    std::printf("correct: %s\n", correct ? "YES" : "NO");
    print_cluster_stats("cluster-all", ctx.cluster_all);
    ctx.stop();
    return rc == VNPU_OK && correct ? 0 : 1;
}

int vnpu_run_specdemo(int argc, char **argv)
{
    Options opt = parse_options(argc, argv);
    std::printf("=== Distributed DFlash/MTP-style speculative demo ===\n");
    std::printf("hidden=%u vocab=%u gamma=%u steps=%u ports=%u,%u\n",
                opt.hidden, opt.vocab, opt.gamma, opt.steps,
                opt.port1, opt.port2);
    DistributedContext ctx;
    if (!ctx.start(opt, opt.hidden, opt.vocab)) {
        std::fprintf(stderr, "failed to start demo nodes\n");
        return 1;
    }
    LogitsUser draft{ctx.cluster[0], ctx.weight[0], opt.hidden, opt.vocab};
    LogitsUser verify{ctx.cluster[1], ctx.weight[1], opt.hidden, opt.vocab};
    vnpu_speculative_config_t cfg;
    vnpu_speculative_config_init(&cfg);
    cfg.gamma = opt.gamma;
    cfg.vocab = opt.vocab;
    cfg.max_tokens = opt.gamma + 1;
    cfg.eos_token = static_cast<uint32_t>(-1);
    cfg.temperature = opt.temperature;
    cfg.seed = 42;
    std::vector<int32_t> context = {1, 7, 3, 9, 2, 8};
    double total_tokens = 0.0;
    double total_ms = 0.0;
    uint32_t total_accepted = 0;
    for (uint32_t step = 0; step < opt.steps; ++step) {
        vnpu_speculative_result_t r{};
        vnpu_status_t rc = vnpu_speculative_step(
            ctx.cluster_all, &cfg, context.data(),
            static_cast<uint32_t>(context.size()),
            remote_logits, &draft, remote_logits, &verify, &r);
        if (rc != VNPU_OK) {
            std::fprintf(stderr, "speculative step failed: %s\n",
                         vnpu_status_string(rc));
            ctx.stop();
            return 1;
        }
        std::printf("step %2u: produced=%u accepted=%u draft=%.3f ms "
                    "verify=%.3f ms %.1f tok/s theoretical=%.2fx\n",
                    step, r.produced, r.accepted, r.draft_time_ms,
                    r.verify_time_ms, r.tokens_per_second,
                    r.theoretical_speedup);
        for (uint32_t i = 0; i < r.produced; ++i)
            context.push_back(r.tokens[i]);
        total_tokens += r.produced;
        total_ms += r.total_time_ms;
        total_accepted += r.accepted;
        if (r.eos) break;
    }
    std::printf("summary: %.0f tokens in %.3f ms = %.1f tok/s, "
                "accepted=%u/%u\n",
                total_tokens, total_ms,
                total_ms > 0.0 ? total_tokens * 1000.0 / total_ms : 0.0,
                total_accepted, total_accepted + opt.steps);
    print_cluster_stats("draft", ctx.cluster[0]);
    print_cluster_stats("verify", ctx.cluster[1]);
    print_cluster_stats("cluster-all", ctx.cluster_all);
    ctx.stop();
    return 0;
}
