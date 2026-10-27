/*
 * autotune.cpp - empirical tile/unit selection driven by Sherlock timings.
 */
#include "../internal.h"

namespace vnpu {
namespace {

struct Candidate {
    uint32_t unit;
    uint32_t tm, tk, tn;
};

static bool shape_overflow(uint64_t a, uint64_t b, uint64_t c)
{
    if (a == 0 || b == 0 || c == 0) return false;
    uint64_t ab = a * b;
    if (b != 0 && ab / b != a) return true;
    return c != 0 && ab > std::numeric_limits<uint64_t>::max() / c;
}

double elapsed_ms(uint64_t start_ns)
{
    return static_cast<double>(now_ns() - start_ns) / 1e6;
}

void fill_f32(float *p, size_t n, uint32_t seed)
{
    for (size_t i = 0; i < n; ++i)
        p[i] = static_cast<float>(
            (static_cast<int>((i * 1103515245ull + seed) % 199) - 99)) *
            0.0037f;
}

vnpu_status_t benchmark_f32(vnpu_device_t *device, vnpu_stream_t *stream,
                            const std::vector<float> &A,
                            const std::vector<float> &B,
                            std::vector<float> &C,
                            uint32_t M, uint32_t K, uint32_t N,
                            const Candidate &c, uint32_t iterations,
                            double *best_ms)
{
    vnpu_device_set_tiles(device, c.tm, c.tk, c.tn);
    vnpu_matmul_f32_args_t args{};
    args.A = A.data();
    args.B = B.data();
    args.C = C.data();
    args.M = M; args.K = K; args.N = N;
    args.alpha = 1.0f;
    args.beta = 0.0f;
    args.unit = c.unit;
    for (uint32_t i = 0; i < 2; ++i) {
        vnpu_status_t rc = vnpu_matmul_f32(stream, &args);
        if (rc != VNPU_OK) return rc;
    }
    uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iterations; ++i) {
        vnpu_status_t rc = vnpu_matmul_f32(stream, &args);
        if (rc != VNPU_OK) return rc;
    }
    *best_ms = elapsed_ms(t0) / static_cast<double>(iterations);
    return VNPU_OK;
}

vnpu_status_t benchmark_i8(vnpu_device_t *device, vnpu_stream_t *stream,
                           const std::vector<int8_t> &A,
                           vnpu_packed_weights_t *W,
                           const std::vector<float> &scale_a,
                           std::vector<float> &C,
                           uint32_t M, uint32_t K, uint32_t N,
                           const Candidate &c, uint32_t iterations,
                           double *best_ms)
{
    vnpu_device_set_tiles(device, c.tm, c.tk, c.tn);
    vnpu_matmul_i8_args_t args{};
    args.A = A.data();
    args.B = W;
    args.scale_a = scale_a.data();
    args.C = C.data();
    args.M = M; args.K = K; args.N = N;
    args.unit = c.unit;
    for (uint32_t i = 0; i < 2; ++i) {
        vnpu_status_t rc = vnpu_matmul_i8(stream, &args);
        if (rc != VNPU_OK) return rc;
    }
    uint64_t t0 = now_ns();
    for (uint32_t i = 0; i < iterations; ++i) {
        vnpu_status_t rc = vnpu_matmul_i8(stream, &args);
        if (rc != VNPU_OK) return rc;
    }
    *best_ms = elapsed_ms(t0) / static_cast<double>(iterations);
    return VNPU_OK;
}

double scalar_baseline_f32(const std::vector<float> &A,
                           const std::vector<float> &B,
                           std::vector<float> &C,
                           uint32_t M, uint32_t K, uint32_t N)
{
    GemmF32Params p;
    p.A = A.data(); p.B = B.data(); p.C = C.data();
    p.M = M; p.K = K; p.N = N;
    p.lda = K; p.ldb = N; p.ldc = N;
    p.alpha = 1.0f; p.beta = 0.0f;
    p.isa = ISA_SCALAR;
    p.unit = VNPU_UNIT_VOODOO;
    std::vector<TileJob> tiles;
    TileJob t{};
    t.m0 = 0; t.n0 = 0; t.tm = M; t.tn = N;
    t.tile_m = M; t.tile_n = N; t.tile_k = K;
    tiles.push_back(t);
    p.tiles = &tiles;
    uint64_t t0 = now_ns();
    kernel_gemm_f32_range(p, 0, 1, 0);
    return elapsed_ms(t0);
}

double scalar_baseline_i8(const std::vector<int8_t> &A,
                          vnpu_packed_weights_t *W,
                          const std::vector<float> &scale_a,
                          std::vector<float> &C,
                          uint32_t M, uint32_t K, uint32_t N)
{
    GemmI8Params p;
    p.A = A.data(); p.C = C.data(); p.scale_a = scale_a.data();
    p.M = M; p.K = K; p.N = N;
    p.lda = K; p.ldc = N;
    p.isa = ISA_SCALAR;
    p.has_vnni = false;
    p.unit = VNPU_UNIT_VOODOO;
    p.B.packed = W->packed.data();
    p.B.original = W->original.data();
    p.B.scales = W->scales.data();
    p.B.column_sums = W->column_sums.data();
    p.B.K = W->K; p.B.N = W->N;
    p.B.Kpad = W->Kpad; p.B.Npad = W->Npad;
    std::vector<TileJob> tiles;
    TileJob t{};
    t.m0 = 0; t.n0 = 0; t.tm = M; t.tn = N;
    t.tile_m = M; t.tile_n = N; t.tile_k = K;
    tiles.push_back(t);
    p.tiles = &tiles;
    uint64_t t0 = now_ns();
    kernel_gemm_i8_range(p, 0, 1, 0);
    return elapsed_ms(t0);
}

} /* anonymous namespace */
} /* namespace vnpu */

extern "C" {

VNPU_API vnpu_status_t VNPU_CALL vnpu_autotune_gemm(
    vnpu_device_t *device,
    uint32_t M,
    uint32_t K,
    uint32_t N,
    uint32_t iterations,
    vnpu_dtype_t dtype,
    vnpu_tune_result_t *result)
{
    if (!device || !result || !M || !K || !N)
        return VNPU_ERR_INVALID_ARGUMENT;
    if (dtype != VNPU_DTYPE_F32 && dtype != VNPU_DTYPE_I8)
        return VNPU_ERR_UNSUPPORTED;
    if (vnpu::shape_overflow(M, K, N)) return VNPU_ERR_INVALID_ARGUMENT;
    if (iterations == 0) iterations = 3;
    if (iterations > 20) iterations = 20;
    return vnpu::guard([&]() -> vnpu_status_t {
        std::memset(result, 0, sizeof(*result));
        uint32_t orig_tm = device->tile_m.load();
        uint32_t orig_tk = device->tile_k.load();
        uint32_t orig_tn = device->tile_n.load();
        bool trace_was_on = device->trace.enabled.load();
        device->trace.enabled.store(false);
        vnpu_stream_t *stream = nullptr;
        vnpu_status_t rc = vnpu_stream_create(device, 0, &stream);
        if (rc != VNPU_OK) {
            device->trace.enabled.store(trace_was_on);
            return rc;
        }
        std::vector<vnpu::Candidate> candidates;
        const vnpu::Candidate voodoo[] = {
            {VNPU_UNIT_VOODOO, 32, 128, 64},
            {VNPU_UNIT_VOODOO, 64, 128, 64},
            {VNPU_UNIT_VOODOO, 64, 256, 64},
            {VNPU_UNIT_VOODOO, 128, 256, 128},
        };
        const vnpu::Candidate wildcat[] = {
            {VNPU_UNIT_WILDCAT, 16, 64, 32},
            {VNPU_UNIT_WILDCAT, 32, 64, 64},
            {VNPU_UNIT_WILDCAT, 32, 128, 64},
        };
        const vnpu::Candidate cell[] = {
            {VNPU_UNIT_CELL, 8, 32, 32},
            {VNPU_UNIT_CELL, 16, 64, 64},
            {VNPU_UNIT_CELL, 16, 128, 64},
        };
        auto add = [&](const vnpu::Candidate &c) {
            vnpu::Candidate x = c;
            x.tm = std::min<uint32_t>(x.tm, M);
            x.tn = std::min<uint32_t>(x.tn, N);
            x.tk = std::min<uint32_t>(x.tk, std::max<uint32_t>(1, K));
            candidates.push_back(x);
        };
        for (const auto &c : voodoo) add(c);
        for (const auto &c : wildcat) add(c);
        for (const auto &c : cell) add(c);

        double baseline_ms = 0.0;
        double best_ms = std::numeric_limits<double>::infinity();
        vnpu::Candidate best{};
        std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
        vnpu_packed_weights_t *W = nullptr;

        if (dtype == VNPU_DTYPE_F32) {
            std::vector<float> A(static_cast<size_t>(M) * K);
            std::vector<float> B(static_cast<size_t>(K) * N);
            vnpu::fill_f32(A.data(), A.size(), 17);
            vnpu::fill_f32(B.data(), B.size(), 71);
            baseline_ms = vnpu::scalar_baseline_f32(A, B, C, M, K, N);
            for (const auto &c : candidates) {
                double ms = 0.0;
                rc = vnpu::benchmark_f32(device, stream, A, B, C,
                                         M, K, N, c, iterations, &ms);
                if (rc == VNPU_OK && ms > 0.0 && ms < best_ms) {
                    best_ms = ms;
                    best = c;
                }
            }
        } else {
            std::vector<int8_t> A(static_cast<size_t>(M) * K);
            std::vector<int8_t> B(static_cast<size_t>(K) * N);
            std::vector<float> scale_a(M, 1.0f), scale_b(N, 1.0f);
            for (size_t i = 0; i < A.size(); ++i)
                A[i] = static_cast<int8_t>((i * 37 + 11) % 61 - 30);
            for (size_t i = 0; i < B.size(); ++i)
                B[i] = static_cast<int8_t>((i * 17 + 5) % 51 - 25);
            rc = vnpu_pack_weights_i8(device, B.data(), scale_b.data(),
                                      K, N, 0, 0, &W);
            if (rc != VNPU_OK) {
                vnpu_stream_destroy(stream);
                device->trace.enabled.store(trace_was_on);
                return rc;
            }
            baseline_ms = vnpu::scalar_baseline_i8(A, W, scale_a, C,
                                                   M, K, N);
            for (const auto &c : candidates) {
                double ms = 0.0;
                rc = vnpu::benchmark_i8(device, stream, A, W, scale_a, C,
                                        M, K, N, c, iterations, &ms);
                if (rc == VNPU_OK && ms > 0.0 && ms < best_ms) {
                    best_ms = ms;
                    best = c;
                }
            }
            vnpu_packed_weights_destroy(W);
        }
        vnpu_stream_destroy(stream);
        device->trace.enabled.store(trace_was_on);
        if (!std::isfinite(best_ms)) {
            vnpu_device_set_tiles(device, orig_tm, orig_tk, orig_tn);
            return VNPU_ERR_INTERNAL;
        }
        vnpu_device_set_tiles(device, best.tm, best.tk, best.tn);
        uint64_t flops = 2ull * M * K * N;
        result->tile_m = best.tm;
        result->tile_k = best.tk;
        result->tile_n = best.tn;
        result->unit = best.unit;
        result->threads = device->info.num_threads;
        result->best_ms = best_ms;
        result->baseline_ms = baseline_ms;
        result->speedup = baseline_ms > 0.0 ? baseline_ms / best_ms : 0.0;
        result->gflops = static_cast<double>(flops) / (best_ms * 1e6);
        result->bytes_read = static_cast<uint64_t>(M) * K * sizeof(float) +
                             static_cast<uint64_t>(K) * N * sizeof(float);
        result->bytes_written = static_cast<uint64_t>(M) * N * sizeof(float);
        return VNPU_OK;
    });
}

} /* extern "C" */
