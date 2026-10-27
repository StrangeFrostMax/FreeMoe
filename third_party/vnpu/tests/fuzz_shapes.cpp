/*
 * fuzz_shapes.cpp - randomized shape/unit/culling correctness fuzzer.
 */
#include "vnpu.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

namespace {

uint64_t rng_state = 0x123456789abcdefull;
uint32_t xs()
{
    uint64_t x = rng_state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    rng_state = x;
    return static_cast<uint32_t>(x >> 16);
}

float rnd()
{
    return static_cast<float>(static_cast<int>(xs() % 2001) - 1000) /
           1000.0f;
}

float ref_dot(const std::vector<float> &A,
              const std::vector<float> &B,
              uint32_t M, uint32_t K, uint32_t N,
              uint32_t m, uint32_t n, bool trans_b)
{
    float acc = 0.0f;
    for (uint32_t k = 0; k < K; ++k) {
        float b = trans_b ? B[static_cast<size_t>(n) * K + k] :
                            B[static_cast<size_t>(k) * N + n];
        acc += A[static_cast<size_t>(m) * K + k] * b;
    }
    return acc;
}

bool close(float a, float b, float atol, float rtol)
{
    return std::fabs(a - b) <=
           atol + rtol * std::max(std::fabs(a), std::fabs(b));
}

} /* namespace */

int main(int argc, char **argv)
{
    int iterations = argc > 1 ? std::atoi(argv[1]) : 80;
    if (iterations < 1) iterations = 1;
    if (iterations > 5000) iterations = 5000;
    vnpu_config_t cfg;
    vnpu_config_init(&cfg);
    cfg.node_name = "fuzz";
    vnpu_device_t *device = nullptr;
    vnpu_stream_t *stream = nullptr;
    if (vnpu_device_create(&cfg, &device) != VNPU_OK ||
        vnpu_stream_create(device, 0, &stream) != VNPU_OK) {
        std::fprintf(stderr, "failed to create runtime\n");
        return 1;
    }
    int failures = 0;
    const uint32_t units[] = {VNPU_UNIT_AUTO, VNPU_UNIT_VOODOO,
                              VNPU_UNIT_WILDCAT, VNPU_UNIT_CELL};
    for (int it = 0; it < iterations; ++it) {
        uint32_t M = 1 + xs() % 73;
        uint32_t K = 1 + xs() % 73;
        uint32_t N = 1 + xs() % 73;
        bool trans_b = (xs() & 1u) != 0;
        uint32_t unit = units[xs() % 4];
        std::vector<float> A(static_cast<size_t>(M) * K);
        std::vector<float> B(static_cast<size_t>(K) * N);
        std::vector<float> C(static_cast<size_t>(M) * N, 0.0f);
        for (auto &x : A) x = rnd();
        for (auto &x : B) x = rnd();

        vnpu_matmul_f32_args_t args{};
        args.A = A.data();
        args.B = B.data();
        args.C = C.data();
        args.M = M; args.K = K; args.N = N;
        args.trans_b = trans_b ? 1 : 0;
        args.alpha = 1.0f; args.beta = 0.0f;
        args.unit = unit;
        vnpu_status_t rc = vnpu_matmul_f32(stream, &args);
        if (rc != VNPU_OK) {
            std::fprintf(stderr, "iter %d: matmul_f32 failed: %s\n", it,
                         vnpu_status_string(rc));
            failures++;
            continue;
        }
        float max_err = 0.0f;
        for (uint32_t m = 0; m < M; ++m)
            for (uint32_t n = 0; n < N; ++n) {
                float expected = ref_dot(A, B, M, K, N, m, n, trans_b);
                max_err = std::max(max_err,
                    std::fabs(C[static_cast<size_t>(m) * N + n] -
                              expected));
            }
        if (max_err > 5e-3f) {
            std::fprintf(stderr,
                "iter %d F32 mismatch shape=%ux%ux%u unit=%u transB=%d "
                "max_err=%g\n",
                it, M, K, N, unit, trans_b ? 1 : 0, max_err);
            failures++;
        }

        /* Every third iteration also exercises INT8 packing + VNNI. */
        if (it % 3 == 0 && !trans_b) {
            std::vector<int8_t> A8(A.size()), B8(B.size());
            std::vector<float> sa(M, 1.0f), sb(N, 1.0f);
            vnpu_quantize_i8_args_t qa{};
            qa.X = A.data(); qa.Y = A8.data(); qa.scales = sa.data();
            qa.rows = M; qa.cols = K; qa.axis = VNPU_QUANT_PER_ROW;
            vnpu_quantize_i8(stream, &qa);
            vnpu_quantize_i8_args_t qb{};
            qb.X = B.data(); qb.Y = B8.data(); qb.scales = sb.data();
            qb.rows = K; qb.cols = N; qb.axis = VNPU_QUANT_PER_COLUMN;
            vnpu_quantize_i8(stream, &qb);
            vnpu_packed_weights_t *W = nullptr;
            if (vnpu_pack_weights_i8(device, B8.data(), sb.data(), K, N,
                                     0, 0, &W) == VNPU_OK) {
                std::fill(C.begin(), C.end(), 0.0f);
                vnpu_matmul_i8_args_t mi{};
                mi.A = A8.data(); mi.B = W; mi.scale_a = sa.data();
                mi.C = C.data(); mi.M = M; mi.K = K; mi.N = N;
                mi.unit = unit;
                rc = vnpu_matmul_i8(stream, &mi);
                if (rc != VNPU_OK) {
                    std::fprintf(stderr, "iter %d: matmul_i8 failed: %s\n",
                                 it, vnpu_status_string(rc));
                    failures++;
                } else {
                    float scale = 0.0f;
                    for (uint32_t m = 0; m < M; ++m)
                        for (uint32_t n = 0; n < N; ++n)
                            scale = std::max(scale,
                                std::fabs(ref_dot(A, B, M, K, N, m, n,
                                                  false)));
                    float tol = std::max(0.2f, 0.06f * scale);
                    bool i8_bad = false;
                    for (uint32_t m = 0; m < M && !i8_bad; ++m)
                        for (uint32_t n = 0; n < N; ++n) {
                            float expected = ref_dot(A, B, M, K, N, m, n,
                                                      false);
                            if (!close(C[static_cast<size_t>(m) * N + n],
                                       expected, tol, 0.08f)) {
                                i8_bad = true;
                                break;
                            }
                        }
                    if (i8_bad) {
                        std::fprintf(stderr,
                            "iter %d I8 mismatch shape=%ux%ux%u unit=%u\n",
                            it, M, K, N, unit);
                        failures++;
                    }
                }
                vnpu_packed_weights_destroy(W);
            }
        }
        if ((it + 1) % 20 == 0)
            std::printf("  %d/%d shapes checked\n", it + 1, iterations);
    }
    vnpu_stream_destroy(stream);
    vnpu_device_destroy(device);
    std::printf("fuzz: %d iterations, %d failures\n", iterations, failures);
    return failures ? 1 : 0;
}
