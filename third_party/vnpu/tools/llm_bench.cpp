/*
 * llm_bench.cpp - synthetic transformer block benchmark.
 *
 * This is not a model runner or tokenizer.  It exercises the exact operator
 * mix that dominates transformer inference - RMSNorm, RoPE, attention, four
 * large GEMMs and SwiGLU - with realistic dimensions, so prefill and decode
 * can be compared on CPU-only virtual machines.
 */
#include "vnpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Options {
    uint32_t hidden = 1024;
    uint32_t intermediate = 2816;
    uint32_t heads = 16;
    uint32_t layers = 2;
    uint32_t prefill = 32;
    uint32_t decode = 1;
    uint32_t iterations = 3;
    uint32_t threads = 0;
    uint32_t unit = VNPU_UNIT_AUTO;
    std::string dtype = "f32";
    std::string phase = "both";
    std::string trace_path;
    std::string json_path;
};

Options parse_options(int argc, char **argv)
{
    Options o;
    for (int i = 0; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> const char * {
            return (i + 1 < argc) ? argv[++i] : nullptr;
        };
        if (a == "--hidden") o.hidden = std::atoi(next());
        else if (a == "--intermediate") o.intermediate = std::atoi(next());
        else if (a == "--heads") o.heads = std::atoi(next());
        else if (a == "--layers") o.layers = std::atoi(next());
        else if (a == "--prefill") o.prefill = std::atoi(next());
        else if (a == "--decode") o.decode = std::atoi(next());
        else if (a == "--iterations") o.iterations = std::atoi(next());
        else if (a == "--threads") o.threads = std::atoi(next());
        else if (a == "--dtype") o.dtype = next();
        else if (a == "--phase") o.phase = next();
        else if (a == "--trace") o.trace_path = next();
        else if (a == "--json") o.json_path = next();
        else if (a == "--unit") {
            std::string u = next();
            if (u == "voodoo") o.unit = VNPU_UNIT_VOODOO;
            else if (u == "wildcat") o.unit = VNPU_UNIT_WILDCAT;
            else if (u == "cell") o.unit = VNPU_UNIT_CELL;
            else o.unit = VNPU_UNIT_AUTO;
        } else if (a == "--help") {
            std::printf("Usage: vnpu-cli llm-bench [--hidden N] "
                        "[--intermediate N] [--heads N] [--layers N] "
                        "[--prefill N] [--decode N] [--iterations N] "
                        "[--dtype f32|i8|both] [--phase prefill|decode|both] "
                        "[--trace path] [--json path]\n");
            std::exit(0);
        }
    }
    o.hidden = std::min<uint32_t>(std::max<uint32_t>(o.hidden, 64), 8192);
    o.intermediate = std::min<uint32_t>(
        std::max<uint32_t>(o.intermediate, o.hidden), 32768);
    o.heads = std::min<uint32_t>(std::max<uint32_t>(o.heads, 1), 128);
    if (o.hidden % o.heads) o.heads = 1;
    o.layers = std::min<uint32_t>(std::max<uint32_t>(o.layers, 1), 64);
    o.prefill = std::min<uint32_t>(std::max<uint32_t>(o.prefill, 1), 4096);
    o.decode = std::min<uint32_t>(std::max<uint32_t>(o.decode, 1), 64);
    o.iterations = std::min<uint32_t>(std::max<uint32_t>(o.iterations, 1), 20);
    return o;
}

uint32_t rng = 0x5eed1234u;
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
           5000.0f;
}

double now_ms()
{
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double, std::milli>(
        clock::now().time_since_epoch()).count();
}

struct GemmWeights {
    uint32_t K = 0, N = 0;
    std::vector<float> B;
    std::vector<int8_t> B8;
    std::vector<float> scales;
    vnpu_packed_weights_t *packed = nullptr;
    bool quantized = false;

    bool init(vnpu_device_t *device, vnpu_stream_t *stream,
              uint32_t k, uint32_t n, bool quantize)
    {
        K = k; N = n; quantized = quantize;
        B.resize(static_cast<size_t>(K) * N);
        for (auto &x : B) x = rndf();
        if (!quantize) return true;
        B8.resize(B.size());
        scales.resize(N, 1.0f);
        vnpu_quantize_i8_args_t q{};
        q.X = B.data(); q.Y = B8.data(); q.scales = scales.data();
        q.rows = K; q.cols = N; q.axis = VNPU_QUANT_PER_COLUMN;
        if (vnpu_quantize_i8(stream, &q) != VNPU_OK) return false;
        return vnpu_pack_weights_i8(device, B8.data(), scales.data(),
                                    K, N, 0, 0, &packed) == VNPU_OK;
    }
    void destroy()
    {
        if (packed) { vnpu_packed_weights_destroy(packed); packed = nullptr; }
    }
};

struct Bundle {
    GemmWeights qkv, attn_out, gate_up, down;
    bool init(vnpu_device_t *device, vnpu_stream_t *stream,
              const Options &o, bool quantize)
    {
        if (!qkv.init(device, stream, o.hidden, 3 * o.hidden, quantize))
            return false;
        if (!attn_out.init(device, stream, o.hidden, o.hidden, quantize))
            return false;
        if (!gate_up.init(device, stream, o.hidden,
                          2 * o.intermediate, quantize))
            return false;
        if (!down.init(device, stream, o.intermediate, o.hidden, quantize))
            return false;
        return true;
    }
    void destroy()
    {
        qkv.destroy(); attn_out.destroy();
        gate_up.destroy(); down.destroy();
    }
};

struct Workspace {
    uint32_t tokens, hidden, intermediate, heads, head_dim;
    std::vector<float> x, xn, residual, norm_w;
    std::vector<float> qkv, q, k, v, attn, attn_major, attn_proj, tmp;
    std::vector<float> gate_up, gate, up, ff, down;
    std::vector<int8_t> a8;
    std::vector<float> scale_a;

    Workspace(uint32_t t, const Options &o)
        : tokens(t), hidden(o.hidden), intermediate(o.intermediate),
          heads(o.heads), head_dim(o.hidden / o.heads)
    {
        size_t hs = static_cast<size_t>(tokens) * hidden;
        size_t is = static_cast<size_t>(tokens) * intermediate;
        x.resize(hs); xn.resize(hs); residual.resize(hs);
        norm_w.resize(hidden, 1.0f);
        qkv.resize(static_cast<size_t>(tokens) * 3 * hidden);
        q.resize(hs); k.resize(hs); v.resize(hs);
        attn.resize(hs); attn_major.resize(hs); attn_proj.resize(hs);
        tmp.resize(hs);
        gate_up.resize(2 * is); gate.resize(is); up.resize(is);
        ff.resize(is); down.resize(hs);
        a8.resize(std::max<size_t>(hs, is));
        scale_a.resize(tokens, 1.0f);
        for (auto &val : x) val = rndf();
        residual = x;
    }
};

void to_head_major(const float *src, float *dst, uint32_t tokens,
                   uint32_t heads, uint32_t head_dim)
{
    uint32_t hidden = heads * head_dim;
    for (uint32_t h = 0; h < heads; ++h)
        for (uint32_t t = 0; t < tokens; ++t)
            for (uint32_t d = 0; d < head_dim; ++d)
                dst[(static_cast<size_t>(h) * tokens + t) * head_dim + d] =
                    src[static_cast<size_t>(t) * hidden +
                        static_cast<size_t>(h) * head_dim + d];
}

void from_head_major(const float *src, float *dst, uint32_t tokens,
                     uint32_t heads, uint32_t head_dim)
{
    uint32_t hidden = heads * head_dim;
    for (uint32_t h = 0; h < heads; ++h)
        for (uint32_t t = 0; t < tokens; ++t)
            for (uint32_t d = 0; d < head_dim; ++d)
                dst[static_cast<size_t>(t) * hidden +
                    static_cast<size_t>(h) * head_dim + d] =
                    src[(static_cast<size_t>(h) * tokens + t) * head_dim + d];
}

vnpu_status_t run_gemm(vnpu_stream_t *stream, GemmWeights &w,
                       Workspace &ws, const float *A, float *C,
                       uint32_t M, uint32_t unit)
{
    if (!w.quantized) {
        vnpu_matmul_f32_args_t args{};
        args.A = A; args.B = w.B.data(); args.C = C;
        args.M = M; args.K = w.K; args.N = w.N;
        args.alpha = 1.0f; args.beta = 0.0f; args.unit = unit;
        return vnpu_matmul_f32(stream, &args);
    }
    size_t count = static_cast<size_t>(M) * w.K;
    if (ws.a8.size() < count) ws.a8.resize(count);
    if (ws.scale_a.size() < M) ws.scale_a.resize(M, 1.0f);
    vnpu_quantize_i8_args_t q{};
    q.X = A; q.Y = ws.a8.data(); q.scales = ws.scale_a.data();
    q.rows = M; q.cols = w.K; q.axis = VNPU_QUANT_PER_ROW;
    vnpu_status_t rc = vnpu_quantize_i8(stream, &q);
    if (rc != VNPU_OK) return rc;
    vnpu_matmul_i8_args_t args{};
    args.A = ws.a8.data(); args.B = w.packed;
    args.scale_a = ws.scale_a.data(); args.C = C;
    args.M = M; args.K = w.K; args.N = w.N; args.unit = unit;
    return vnpu_matmul_i8(stream, &args);
}

vnpu_status_t run_block(vnpu_stream_t *stream, Bundle &b, Workspace &ws,
                        const Options &o)
{
    const uint32_t T = ws.tokens;
    const uint32_t H = ws.hidden;
    const uint32_t I = ws.intermediate;
    const uint32_t HD = ws.head_dim;
    /* pre-attention RMSNorm */
    vnpu_rmsnorm_f32_args_t rn{};
    rn.X = ws.x.data(); rn.W = ws.norm_w.data(); rn.Y = ws.xn.data();
    rn.rows = T; rn.cols = H; rn.eps = 1e-5f;
    vnpu_status_t rc = vnpu_rmsnorm_f32(stream, &rn);
    if (rc != VNPU_OK) return rc;
    /* QKV projection */
    rc = run_gemm(stream, b.qkv, ws, ws.xn.data(), ws.qkv.data(), T, o.unit);
    if (rc != VNPU_OK) return rc;
    /* Split QKV and transpose to head-major for RoPE + attention. */
    for (uint32_t t = 0; t < T; ++t) {
        const float *row = ws.qkv.data() + static_cast<size_t>(t) * 3 * H;
        std::copy(row, row + H, ws.q.begin() + static_cast<size_t>(t) * H);
        std::copy(row + H, row + 2 * H,
                  ws.k.begin() + static_cast<size_t>(t) * H);
        std::copy(row + 2 * H, row + 3 * H,
                  ws.v.begin() + static_cast<size_t>(t) * H);
    }
    to_head_major(ws.q.data(), ws.tmp.data(), T, ws.heads, HD);
    std::copy(ws.tmp.begin(), ws.tmp.end(), ws.q.begin());
    to_head_major(ws.k.data(), ws.tmp.data(), T, ws.heads, HD);
    std::copy(ws.tmp.begin(), ws.tmp.end(), ws.k.begin());
    to_head_major(ws.v.data(), ws.tmp.data(), T, ws.heads, HD);
    std::copy(ws.tmp.begin(), ws.tmp.end(), ws.v.begin());
    vnpu_rope_f32_args_t rope{};
    rope.Q = ws.q.data(); rope.K = ws.k.data();
    rope.batch = 1; rope.heads = ws.heads; rope.q_seq = T; rope.k_seq = T;
    rope.head_dim = HD; rope.theta_base = 10000.0f; rope.scaling = 1.0f;
    rc = vnpu_rope_f32(stream, &rope);
    if (rc != VNPU_OK) return rc;
    vnpu_attention_f32_args_t att{};
    att.Q = ws.q.data(); att.K = ws.k.data(); att.V = ws.v.data();
    att.O = ws.attn_major.data(); att.batch = 1; att.heads = ws.heads;
    att.seq_q = T; att.seq_k = T; att.head_dim = HD; att.causal = 1;
    att.scale = 1.0f / std::sqrt(static_cast<float>(HD));
    att.unit = o.unit;
    rc = vnpu_attention_f32(stream, &att);
    if (rc != VNPU_OK) return rc;
    from_head_major(ws.attn_major.data(), ws.attn.data(), T, ws.heads, HD);
    /* Output projection + residual. */
    rc = run_gemm(stream, b.attn_out, ws, ws.attn.data(),
                  ws.attn_proj.data(), T, o.unit);
    if (rc != VNPU_OK) return rc;
    vnpu_binary_f32_args_t add{};
    add.A = ws.residual.data(); add.B = ws.attn_proj.data();
    add.Y = ws.x.data(); add.count = static_cast<size_t>(T) * H;
    add.alpha = 1.0f; add.beta = 1.0f;
    rc = vnpu_add_f32(stream, &add);
    if (rc != VNPU_OK) return rc;
    ws.residual = ws.x;
    /* FFN RMSNorm. */
    rn.X = ws.x.data(); rn.Y = ws.xn.data();
    rc = vnpu_rmsnorm_f32(stream, &rn);
    if (rc != VNPU_OK) return rc;
    /* Gate/up projection. */
    rc = run_gemm(stream, b.gate_up, ws, ws.xn.data(),
                  ws.gate_up.data(), T, o.unit);
    if (rc != VNPU_OK) return rc;
    for (uint32_t t = 0; t < T; ++t) {
        const float *row = ws.gate_up.data() +
                           static_cast<size_t>(t) * 2 * I;
        std::copy(row, row + I,
                  ws.gate.begin() + static_cast<size_t>(t) * I);
        std::copy(row + I, row + 2 * I,
                  ws.up.begin() + static_cast<size_t>(t) * I);
    }
    vnpu_swiglu_f32_args_t sg{};
    sg.gate = ws.gate.data(); sg.up = ws.up.data(); sg.Y = ws.ff.data();
    sg.count = static_cast<size_t>(T) * I; sg.alpha = 1.0f;
    rc = vnpu_swiglu_f32(stream, &sg);
    if (rc != VNPU_OK) return rc;
    /* Down projection + residual. */
    rc = run_gemm(stream, b.down, ws, ws.ff.data(), ws.down.data(),
                  T, o.unit);
    if (rc != VNPU_OK) return rc;
    add.A = ws.residual.data(); add.B = ws.down.data(); add.Y = ws.x.data();
    rc = vnpu_add_f32(stream, &add);
    if (rc != VNPU_OK) return rc;
    ws.residual = ws.x;
    return VNPU_OK;
}

struct PhaseResult {
    std::string phase;
    std::string dtype;
    uint32_t tokens;
    uint32_t layers;
    uint32_t iterations;
    double ms_per_iteration;
    double tokens_per_second;
    double gflops;
    uint64_t flops_per_iteration;
};

PhaseResult run_phase(vnpu_device_t *device, vnpu_stream_t *stream,
                      Bundle &b, const Options &o,
                      const std::string &phase, const std::string &dtype,
                      uint32_t tokens)
{
    Workspace ws(tokens, o);
    /* Warmup. */
    for (uint32_t l = 0; l < o.layers; ++l)
        run_block(stream, b, ws, o);
    double t0 = now_ms();
    for (uint32_t it = 0; it < o.iterations; ++it) {
        for (uint32_t l = 0; l < o.layers; ++l)
            run_block(stream, b, ws, o);
    }
    double ms = (now_ms() - t0) / static_cast<double>(o.iterations);
    uint64_t flops_per_layer =
        2ull * tokens * o.hidden * 3ull * o.hidden +       /* QKV */
        4ull * tokens * tokens * o.hidden +               /* attention */
        2ull * tokens * o.hidden * o.hidden +            /* O proj */
        2ull * tokens * o.hidden * 2ull * o.intermediate + /* gate/up */
        2ull * tokens * o.intermediate * o.hidden;        /* down */
    uint64_t flops = flops_per_layer * o.layers;
    PhaseResult r;
    r.phase = phase;
    r.dtype = dtype;
    r.tokens = tokens;
    r.layers = o.layers;
    r.iterations = o.iterations;
    r.ms_per_iteration = ms;
    r.tokens_per_second = ms > 0.0 ?
        static_cast<double>(tokens) * 1000.0 / ms : 0.0;
    r.gflops = ms > 0.0 ? static_cast<double>(flops) / (ms * 1e6) : 0.0;
    r.flops_per_iteration = flops;
    (void)device;
    return r;
}

void print_result(const PhaseResult &r)
{
    std::printf("%-8s %-4s tokens=%-5u layers=%-2u %9.3f ms "
                "%9.1f tok/s %9.2f GFLOP/s\n",
                r.phase.c_str(), r.dtype.c_str(), r.tokens, r.layers,
                r.ms_per_iteration, r.tokens_per_second, r.gflops);
}

std::string json_results(const std::vector<PhaseResult> &results,
                         const Options &o)
{
    std::string s = "{\n  \"hidden\": " + std::to_string(o.hidden) +
                    ",\n  \"intermediate\": " +
                    std::to_string(o.intermediate) +
                    ",\n  \"heads\": " + std::to_string(o.heads) +
                    ",\n  \"layers\": " + std::to_string(o.layers) +
                    ",\n  \"results\": [\n";
    for (size_t i = 0; i < results.size(); ++i) {
        const auto &r = results[i];
        s += "    {\"phase\":\"" + r.phase + "\",\"dtype\":\"" + r.dtype +
             "\",\"tokens\":" + std::to_string(r.tokens) +
             ",\"ms\":" + std::to_string(r.ms_per_iteration) +
             ",\"tokens_per_second\":" +
             std::to_string(r.tokens_per_second) +
             ",\"gflops\":" + std::to_string(r.gflops) +
             ",\"flops\":" + std::to_string(r.flops_per_iteration) + "}";
        if (i + 1 < results.size()) s += ',';
        s += '\n';
    }
    s += "  ]\n}\n";
    return s;
}

} /* namespace */

int vnpu_run_llm_bench(int argc, char **argv)
{
    Options opt = parse_options(argc, argv);
    std::printf("=== Synthetic transformer block benchmark ===\n");
    std::printf("hidden=%u intermediate=%u heads=%u head_dim=%u layers=%u\n",
                opt.hidden, opt.intermediate, opt.heads,
                opt.hidden / opt.heads, opt.layers);
    std::printf("prefill=%u decode=%u iterations=%u dtype=%s\n",
                opt.prefill, opt.decode, opt.iterations, opt.dtype.c_str());
    vnpu_config_t cfg;
    vnpu_config_init(&cfg);
    cfg.num_threads = opt.threads;
    if (!opt.trace_path.empty()) cfg.flags |= VNPU_CONFIG_TRACE;
    vnpu_device_t *device = nullptr;
    vnpu_stream_t *stream = nullptr;
    if (vnpu_device_create(&cfg, &device) != VNPU_OK ||
        vnpu_stream_create(device, 0, &stream) != VNPU_OK) {
        std::fprintf(stderr, "failed to create runtime\n");
        return 1;
    }
    vnpu_device_info_t info{};
    vnpu_device_get_info(device, &info);
    std::printf("backend=%s threads=%u\n", info.backend, info.num_threads);
    std::vector<std::string> dtypes;
    if (opt.dtype == "both") dtypes = {"f32", "i8"};
    else if (opt.dtype == "i8") dtypes = {"i8"};
    else dtypes = {"f32"};
    std::vector<PhaseResult> results;
    for (const auto &dt : dtypes) {
        Bundle bundle;
        if (!bundle.init(device, stream, opt, dt == "i8")) {
            std::fprintf(stderr, "failed to initialize %s weights\n",
                         dt.c_str());
            bundle.destroy();
            vnpu_stream_destroy(stream);
            vnpu_device_destroy(device);
            return 1;
        }
        if (opt.phase == "both" || opt.phase == "prefill")
            results.push_back(run_phase(device, stream, bundle, opt,
                                        "prefill", dt, opt.prefill));
        if (opt.phase == "both" || opt.phase == "decode")
            results.push_back(run_phase(device, stream, bundle, opt,
                                        "decode", dt, opt.decode));
        bundle.destroy();
    }
    std::printf("------------------------------------------------------------------\n");
    for (const auto &r : results) print_result(r);
    if (!opt.trace_path.empty()) {
        vnpu_trace_dump(device, nullptr);
        vnpu_trace_write_chrome(device, opt.trace_path.c_str());
        std::printf("Chrome trace: %s\n", opt.trace_path.c_str());
    }
    if (!opt.json_path.empty()) {
        std::ofstream out(opt.json_path);
        if (out) out << json_results(results, opt);
    }
    vnpu_stream_destroy(stream);
    vnpu_device_destroy(device);
    return 0;
}
