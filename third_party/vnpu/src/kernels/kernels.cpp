/*
 * kernels.cpp - SIMD GEMM microkernels.
 *
 * Every ISA-specific routine is runtime-dispatched.  The translation unit is
 * compiled with conservative baseline flags; GCC/Clang target attributes
 * enable AVX2/AVX512 code only inside the marked functions.
 */
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif

#include "../internal.h"

#if VNPU_X86 && (defined(__GNUC__) || defined(__clang__))
#  define VNPU_TARGET_AVX2   __attribute__((target("avx2,fma")))
#  define VNPU_TARGET_AVX512 __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni,fma")))
#  define VNPU_TARGET_VNNI   __attribute__((target("avx512f,avx512bw,avx512dq,avx512vl,avx512vnni")))
#else
#  define VNPU_TARGET_AVX2
#  define VNPU_TARGET_AVX512
#  define VNPU_TARGET_VNNI
#endif

namespace vnpu {
namespace {

inline const float *a_ptr(const GemmF32Params &p, uint32_t m, uint32_t k)
{
    if (p.trans_a) return p.A + static_cast<size_t>(k) * p.lda + m;
    return p.A + static_cast<size_t>(m) * p.lda + k;
}

inline const float *b_ptr(const GemmF32Params &p, uint32_t k, uint32_t n)
{
    if (p.trans_b) return p.B + static_cast<size_t>(n) * p.ldb + k;
    return p.B + static_cast<size_t>(k) * p.ldb + n;
}

inline float load_a(const GemmF32Params &p, uint32_t m, uint32_t k)
{
    if (p.trans_a) return p.A[static_cast<size_t>(k) * p.lda + m];
    return p.A[static_cast<size_t>(m) * p.lda + k];
}

inline float load_b(const GemmF32Params &p, uint32_t k, uint32_t n)
{
    if (p.trans_b) return p.B[static_cast<size_t>(n) * p.ldb + k];
    return p.B[static_cast<size_t>(k) * p.ldb + n];
}

inline void store_c(const GemmF32Params &p, uint32_t m, uint32_t n, float value)
{
    float *dst = p.C + static_cast<size_t>(m) * p.ldc + n;
    float out = p.alpha * value;
    if (p.bias) out += p.bias[n];
    if (p.beta != 0.0f) out += p.beta * *dst;
    *dst = apply_epilogue(out, p.epilogue);
}

void zero_c_tile(const GemmF32Params &p, const TileJob &tile)
{
    for (uint32_t m = tile.m0; m < tile.m0 + tile.tm; ++m) {
        float *row = p.C + static_cast<size_t>(m) * p.ldc + tile.n0;
        for (uint32_t n = 0; n < tile.tn; ++n) {
            float out = p.bias ? p.bias[tile.n0 + n] : 0.0f;
            if (p.beta != 0.0f) out += p.beta * row[n];
            row[n] = apply_epilogue(out, p.epilogue);
        }
    }
}

/* Portable scalar path - also the correctness oracle. */
void gemm_f32_scalar_tile(const GemmF32Params &p, const TileJob &tile)
{
    for (uint32_t m = tile.m0; m < tile.m0 + tile.tm; ++m) {
        for (uint32_t n = tile.n0; n < tile.n0 + tile.tn; ++n) {
            float acc = 0.0f;
            for (uint32_t k = 0; k < p.K; ++k)
                acc += load_a(p, m, k) * load_b(p, k, n);
            store_c(p, m, n, acc);
        }
    }
}

#if VNPU_X86
VNPU_TARGET_AVX2
static inline float hsum256_ps(__m256 v)
{
    __m128 lo = _mm256_castps256_ps128(v);
    __m128 hi = _mm256_extractf128_ps(v, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    return _mm_cvtss_f32(sum);
}

/* B is row-major KxN: accumulate 4 rows x 8 columns in registers. */
VNPU_TARGET_AVX2
static void gemm_f32_avx2_nn_tile(const GemmF32Params &p, const TileJob &tile)
{
    constexpr uint32_t MR = 4;
    constexpr uint32_t NR = 8;
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;
    uint32_t m = tile.m0;
    for (; m + MR <= mend; m += MR) {
        uint32_t n = tile.n0;
        for (; n + NR <= nend; n += NR) {
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            const float *a0 = a_ptr(p, m + 0, 0);
            const float *a1 = a_ptr(p, m + 1, 0);
            const float *a2 = a_ptr(p, m + 2, 0);
            const float *a3 = a_ptr(p, m + 3, 0);
            if (p.trans_a) {
                gemm_f32_scalar_tile(p, TileJob{m, n, MR, NR,
                                                tile.tile_m, tile.tile_n,
                                                tile.tile_k, 0, {0, 0, 0}});
                continue;
            }
            for (uint32_t k = 0; k < p.K; ++k) {
                __m256 bv = _mm256_loadu_ps(p.B +
                    static_cast<size_t>(k) * p.ldb + n);
                acc0 = _mm256_fmadd_ps(_mm256_set1_ps(a0[k]), bv, acc0);
                acc1 = _mm256_fmadd_ps(_mm256_set1_ps(a1[k]), bv, acc1);
                acc2 = _mm256_fmadd_ps(_mm256_set1_ps(a2[k]), bv, acc2);
                acc3 = _mm256_fmadd_ps(_mm256_set1_ps(a3[k]), bv, acc3);
            }
            float tmp[MR][NR];
            _mm256_storeu_ps(tmp[0], acc0);
            _mm256_storeu_ps(tmp[1], acc1);
            _mm256_storeu_ps(tmp[2], acc2);
            _mm256_storeu_ps(tmp[3], acc3);
            for (uint32_t i = 0; i < MR; ++i)
                for (uint32_t j = 0; j < NR; ++j)
                    store_c(p, m + i, n + j, tmp[i][j]);
        }
        if (n < nend) {
            TileJob tail{m, n, MR, nend - n, tile.tile_m, tile.tile_n,
                         tile.tile_k, 0, {0, 0, 0}};
            gemm_f32_scalar_tile(p, tail);
        }
    }
    if (m < mend) {
        TileJob tail{m, tile.n0, mend - m, tile.tn, tile.tile_m,
                     tile.tile_n, tile.tile_k, 0, {0, 0, 0}};
        gemm_f32_scalar_tile(p, tail);
    }
}

/* B is NxK: vectorized dot product over K for each output element. */
VNPU_TARGET_AVX2
static void gemm_f32_avx2_nt_tile(const GemmF32Params &p, const TileJob &tile)
{
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;
    for (uint32_t m = tile.m0; m < mend; ++m) {
        const float *arow = a_ptr(p, m, 0);
        for (uint32_t n = tile.n0; n < nend; ++n) {
            const float *brow = p.B + static_cast<size_t>(n) * p.ldb;
            __m256 acc = _mm256_setzero_ps();
            uint32_t k = 0;
            for (; k + 8 <= p.K; k += 8) {
                __m256 av = _mm256_loadu_ps(arow + k);
                __m256 bv = _mm256_loadu_ps(brow + k);
                acc = _mm256_fmadd_ps(av, bv, acc);
            }
            float dot = hsum256_ps(acc);
            for (; k < p.K; ++k) dot += arow[k] * brow[k];
            store_c(p, m, n, dot);
        }
    }
}

/* Two independent 256-bit accumulator chains expose Zen 3's dual FMA pipes.
 * The path uses only AVX2/FMA3 instructions, so it is safe on any AVX2 CPU;
 * it is selected automatically on Zen 3 or explicitly via VNPU_ISA_FORCE_AVX2_DUAL. */
VNPU_TARGET_AVX2
static void gemm_f32_avx2_dual_nt_tile(const GemmF32Params &p, const TileJob &tile)
{
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;
    for (uint32_t m = tile.m0; m < mend; ++m) {
        const float *arow = a_ptr(p, m, 0);
        for (uint32_t n = tile.n0; n < nend; ++n) {
            const float *brow = p.B + static_cast<size_t>(n) * p.ldb;
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            uint32_t k = 0;
            for (; k + 16 <= p.K; k += 16) {
                __m256 av0 = _mm256_loadu_ps(arow + k);
                __m256 av1 = _mm256_loadu_ps(arow + k + 8);
                __m256 bv0 = _mm256_loadu_ps(brow + k);
                __m256 bv1 = _mm256_loadu_ps(brow + k + 8);
                acc0 = _mm256_fmadd_ps(av0, bv0, acc0);
                acc1 = _mm256_fmadd_ps(av1, bv1, acc1);
            }
            __m256 acc = _mm256_add_ps(acc0, acc1);
            if (k + 8 <= p.K) {
                acc = _mm256_fmadd_ps(_mm256_loadu_ps(arow + k),
                                      _mm256_loadu_ps(brow + k), acc);
                k += 8;
            }
            float dot = hsum256_ps(acc);
            for (; k < p.K; ++k) dot += arow[k] * brow[k];
            store_c(p, m, n, dot);
        }
    }
}

/* Dual-chain NN path: four rows by eight columns, with two independent K
 * accumulators per row. M/N tails use the scalar correctness oracle. */
VNPU_TARGET_AVX2
static void gemm_f32_avx2_dual_nn_tile(const GemmF32Params &p, const TileJob &tile)
{
    constexpr uint32_t MR = 4;
    constexpr uint32_t NR = 8;
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;
    uint32_t m = tile.m0;
    for (; m + MR <= mend; m += MR) {
        uint32_t n = tile.n0;
        for (; n + NR <= nend; n += NR) {
            if (p.trans_a) {
                gemm_f32_scalar_tile(p, TileJob{m, n, MR, NR,
                                                tile.tile_m, tile.tile_n,
                                                tile.tile_k, 0, {0, 0, 0}});
                continue;
            }
            __m256 acc0[MR], acc1[MR];
            for (uint32_t r = 0; r < MR; ++r) {
                acc0[r] = _mm256_setzero_ps();
                acc1[r] = _mm256_setzero_ps();
            }
            const float *arow[MR] = {
                a_ptr(p, m + 0, 0), a_ptr(p, m + 1, 0),
                a_ptr(p, m + 2, 0), a_ptr(p, m + 3, 0)
            };
            uint32_t k = 0;
            for (; k + 2 <= p.K; k += 2) {
                __m256 bv0 = _mm256_loadu_ps(p.B + static_cast<size_t>(k) * p.ldb + n);
                __m256 bv1 = _mm256_loadu_ps(p.B + static_cast<size_t>(k + 1) * p.ldb + n);
                for (uint32_t r = 0; r < MR; ++r) {
                    acc0[r] = _mm256_fmadd_ps(_mm256_set1_ps(arow[r][k]), bv0, acc0[r]);
                    acc1[r] = _mm256_fmadd_ps(_mm256_set1_ps(arow[r][k + 1]), bv1, acc1[r]);
                }
            }
            if (k < p.K) {
                __m256 bv = _mm256_loadu_ps(p.B + static_cast<size_t>(k) * p.ldb + n);
                for (uint32_t r = 0; r < MR; ++r)
                    acc0[r] = _mm256_fmadd_ps(_mm256_set1_ps(arow[r][k]), bv, acc0[r]);
            }
            float tmp[MR][NR];
            for (uint32_t r = 0; r < MR; ++r)
                _mm256_storeu_ps(tmp[r], _mm256_add_ps(acc0[r], acc1[r]));
            for (uint32_t i = 0; i < MR; ++i)
                for (uint32_t j = 0; j < NR; ++j)
                    store_c(p, m + i, n + j, tmp[i][j]);
        }
        if (n < nend) {
            TileJob tail{m, n, MR, nend - n, tile.tile_m, tile.tile_n,
                         tile.tile_k, 0, {0, 0, 0}};
            gemm_f32_scalar_tile(p, tail);
        }
    }
    if (m < mend) {
        TileJob tail{m, tile.n0, mend - m, tile.tn, tile.tile_m,
                     tile.tile_n, tile.tile_k, 0, {0, 0, 0}};
        gemm_f32_scalar_tile(p, tail);
    }
}

VNPU_TARGET_AVX512
static inline float hsum512_ps(__m512 v)
{
    float tmp[16];
    /* memcpy avoids a GCC 14 + ASan bug with AVX512 stores to stack arrays. */
    std::memcpy(tmp, &v, sizeof(v));
    float sum = 0.0f;
    for (float x : tmp) sum += x;
    return sum;
}

VNPU_TARGET_AVX512
static void gemm_f32_avx512_nn_tile(const GemmF32Params &p,
                                    const TileJob &tile)
{
    constexpr uint32_t MR = 4;
    constexpr uint32_t NR = 16;
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;
    uint32_t m = tile.m0;
    for (; m + MR <= mend; m += MR) {
        uint32_t n = tile.n0;
        for (; n + NR <= nend; n += NR) {
            __m512 acc0 = _mm512_setzero_ps();
            __m512 acc1 = _mm512_setzero_ps();
            __m512 acc2 = _mm512_setzero_ps();
            __m512 acc3 = _mm512_setzero_ps();
            const float *a0 = a_ptr(p, m + 0, 0);
            const float *a1 = a_ptr(p, m + 1, 0);
            const float *a2 = a_ptr(p, m + 2, 0);
            const float *a3 = a_ptr(p, m + 3, 0);
            if (p.trans_a) {
                gemm_f32_scalar_tile(p, TileJob{m, n, MR, NR,
                                                tile.tile_m, tile.tile_n,
                                                tile.tile_k, 0, {0, 0, 0}});
                continue;
            }
            for (uint32_t k = 0; k < p.K; ++k) {
                __m512 bv = _mm512_loadu_ps(p.B +
                    static_cast<size_t>(k) * p.ldb + n);
                acc0 = _mm512_fmadd_ps(_mm512_set1_ps(a0[k]), bv, acc0);
                acc1 = _mm512_fmadd_ps(_mm512_set1_ps(a1[k]), bv, acc1);
                acc2 = _mm512_fmadd_ps(_mm512_set1_ps(a2[k]), bv, acc2);
                acc3 = _mm512_fmadd_ps(_mm512_set1_ps(a3[k]), bv, acc3);
            }
            float tmp[MR][NR];
            std::memcpy(tmp[0], &acc0, sizeof(acc0));
            std::memcpy(tmp[1], &acc1, sizeof(acc1));
            std::memcpy(tmp[2], &acc2, sizeof(acc2));
            std::memcpy(tmp[3], &acc3, sizeof(acc3));
            for (uint32_t i = 0; i < MR; ++i)
                for (uint32_t j = 0; j < NR; ++j)
                    store_c(p, m + i, n + j, tmp[i][j]);
        }
        if (n < nend) {
            TileJob tail{m, n, MR, nend - n, tile.tile_m, tile.tile_n,
                         tile.tile_k, 0, {0, 0, 0}};
            gemm_f32_scalar_tile(p, tail);
        }
    }
    if (m < mend) {
        TileJob tail{m, tile.n0, mend - m, tile.tn, tile.tile_m,
                     tile.tile_n, tile.tile_k, 0, {0, 0, 0}};
        gemm_f32_scalar_tile(p, tail);
    }
}

VNPU_TARGET_AVX512
static void gemm_f32_avx512_nt_tile(const GemmF32Params &p,
                                    const TileJob &tile)
{
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;
    for (uint32_t m = tile.m0; m < mend; ++m) {
        const float *arow = a_ptr(p, m, 0);
        for (uint32_t n = tile.n0; n < nend; ++n) {
            const float *brow = p.B + static_cast<size_t>(n) * p.ldb;
            __m512 acc = _mm512_setzero_ps();
            uint32_t k = 0;
            for (; k + 16 <= p.K; k += 16) {
                __m512 av = _mm512_loadu_ps(arow + k);
                __m512 bv = _mm512_loadu_ps(brow + k);
                acc = _mm512_fmadd_ps(av, bv, acc);
            }
            float dot = hsum512_ps(acc);
            for (; k < p.K; ++k) dot += arow[k] * brow[k];
            store_c(p, m, n, dot);
        }
    }
}
#endif /* VNPU_X86 */

#if VNPU_NEON
static void gemm_f32_neon_nn_tile(const GemmF32Params &p,
                                  const TileJob &tile)
{
    constexpr uint32_t MR = 4;
    constexpr uint32_t NR = 4;
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;
    uint32_t m = tile.m0;
    for (; m + MR <= mend; m += MR) {
        uint32_t n = tile.n0;
        for (; n + NR <= nend; n += NR) {
            float32x4_t acc0 = vdupq_n_f32(0.0f);
            float32x4_t acc1 = vdupq_n_f32(0.0f);
            float32x4_t acc2 = vdupq_n_f32(0.0f);
            float32x4_t acc3 = vdupq_n_f32(0.0f);
            const float *a0 = a_ptr(p, m + 0, 0);
            const float *a1 = a_ptr(p, m + 1, 0);
            const float *a2 = a_ptr(p, m + 2, 0);
            const float *a3 = a_ptr(p, m + 3, 0);
            for (uint32_t k = 0; k < p.K; ++k) {
                float32x4_t bv = vld1q_f32(p.B +
                    static_cast<size_t>(k) * p.ldb + n);
                acc0 = vfmaq_n_f32(acc0, bv, a0[k]);
                acc1 = vfmaq_n_f32(acc1, bv, a1[k]);
                acc2 = vfmaq_n_f32(acc2, bv, a2[k]);
                acc3 = vfmaq_n_f32(acc3, bv, a3[k]);
            }
            float tmp[MR][NR];
            vst1q_f32(tmp[0], acc0);
            vst1q_f32(tmp[1], acc1);
            vst1q_f32(tmp[2], acc2);
            vst1q_f32(tmp[3], acc3);
            for (uint32_t i = 0; i < MR; ++i)
                for (uint32_t j = 0; j < NR; ++j)
                    store_c(p, m + i, n + j, tmp[i][j]);
        }
        if (n < nend) {
            TileJob tail{m, n, MR, nend - n, tile.tile_m, tile.tile_n,
                         tile.tile_k, 0, {0, 0, 0}};
            gemm_f32_scalar_tile(p, tail);
        }
    }
    if (m < mend) {
        TileJob tail{m, tile.n0, mend - m, tile.tn, tile.tile_m,
                     tile.tile_n, tile.tile_k, 0, {0, 0, 0}};
        gemm_f32_scalar_tile(p, tail);
    }
}
#endif

/* Cell SPE-style path: explicit DMA into an aligned local store followed by
 * register-blocked accumulation.  It deliberately uses a smaller K window so
 * the same code can run on a remote worker with a tiny working set. */
static void gemm_f32_cell_tile(const GemmF32Params &p, const TileJob &tile,
                               uint8_t *scratch, uint64_t scratch_bytes)
{
    uint32_t MR = 4;
    uint32_t NR = (p.isa == ISA_AVX512) ? 16 : 8;
    uint32_t BK = tile.tile_k ? tile.tile_k : 64;
    BK = std::min<uint32_t>(BK, 256);
    const uint64_t a_bytes = static_cast<uint64_t>(MR) * BK * sizeof(float);
    const uint64_t b_bytes = static_cast<uint64_t>(BK) * NR * sizeof(float);
    if (!scratch || a_bytes + b_bytes + 128 > scratch_bytes) {
        gemm_f32_scalar_tile(p, tile);
        return;
    }
    uintptr_t base = reinterpret_cast<uintptr_t>(scratch);
    base = align_up(base, VNPU_CACHELINE_BYTES);
    float *A_s = reinterpret_cast<float *>(base);
    float *B_s = reinterpret_cast<float *>(base + a_bytes);
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;

    for (uint32_t m0 = tile.m0; m0 < mend; m0 += MR) {
        uint32_t rm = std::min<uint32_t>(MR, mend - m0);
        for (uint32_t n0 = tile.n0; n0 < nend; n0 += NR) {
            uint32_t rn = std::min<uint32_t>(NR, nend - n0);
            float acc[4][16] = {};
            for (uint32_t k0 = 0; k0 < p.K; k0 += BK) {
                uint32_t rk = std::min<uint32_t>(BK, p.K - k0);
                /* DMA A panel. */
                for (uint32_t i = 0; i < rm; ++i)
                    for (uint32_t k = 0; k < rk; ++k)
                        A_s[i * BK + k] = load_a(p, m0 + i, k0 + k);
                /* DMA B panel. */
                for (uint32_t k = 0; k < rk; ++k)
                    for (uint32_t j = 0; j < rn; ++j)
                        B_s[k * NR + j] = load_b(p, k0 + k, n0 + j);
                /* SPU local compute. */
                for (uint32_t i = 0; i < rm; ++i) {
                    const float *arow = A_s + i * BK;
                    for (uint32_t k = 0; k < rk; ++k) {
                        float av = arow[k];
                        const float *brow = B_s + k * NR;
                        for (uint32_t j = 0; j < rn; ++j)
                            acc[i][j] += av * brow[j];
                    }
                }
            }
            for (uint32_t i = 0; i < rm; ++i)
                for (uint32_t j = 0; j < rn; ++j)
                    store_c(p, m0 + i, n0 + j, acc[i][j]);
        }
    }
}

static void gemm_f32_dispatch_tile(const GemmF32Params &p,
                                   const TileJob &tile,
                                   uint32_t worker_id)
{
    if (tile.cull) {
        zero_c_tile(p, tile);
        return;
    }
    if (p.unit == VNPU_UNIT_CELL) {
        uint8_t *scratch = nullptr;
        uint64_t bytes = 0;
        if (p.local_stores && worker_id < p.local_stores->size()) {
            scratch = (*p.local_stores)[worker_id].data();
            bytes = (*p.local_stores)[worker_id].size();
        }
        gemm_f32_cell_tile(p, tile, scratch, bytes);
        return;
    }
    if (p.isa == ISA_AVX512 && !p.trans_a) {
        if (!p.trans_b) gemm_f32_avx512_nn_tile(p, tile);
        else gemm_f32_avx512_nt_tile(p, tile);
        return;
    }
    if (p.isa == ISA_AVX2_DUAL && !p.trans_a) {
        if (!p.trans_b) gemm_f32_avx2_dual_nn_tile(p, tile);
        else gemm_f32_avx2_dual_nt_tile(p, tile);
        return;
    }
    if (p.isa == ISA_AVX2 && !p.trans_a) {
        if (!p.trans_b) gemm_f32_avx2_nn_tile(p, tile);
        else gemm_f32_avx2_nt_tile(p, tile);
        return;
    }
#if VNPU_NEON
    if (p.isa == ISA_NEON && !p.trans_a && !p.trans_b) {
        gemm_f32_neon_nn_tile(p, tile);
        return;
    }
#endif
    gemm_f32_scalar_tile(p, tile);
}

/* ------------------------------------------------------------------------- */
/* INT8 kernels                                                              */
/* ------------------------------------------------------------------------- */

inline void store_i8_c(const GemmI8Params &p, uint32_t m, uint32_t n,
                       int32_t dot)
{
    float scale_a = p.scale_a ? p.scale_a[m] : 1.0f;
    float scale_b = p.B.scales ? p.B.scales[n] : 1.0f;
    float out = static_cast<float>(dot) * scale_a * scale_b;
    if (p.bias) out += p.bias[n];
    p.C[static_cast<size_t>(m) * p.ldc + n] =
        apply_epilogue(out, p.epilogue);
}

static void gemm_i8_scalar_tile(const GemmI8Params &p, const TileJob &tile)
{
    for (uint32_t m = tile.m0; m < tile.m0 + tile.tm; ++m) {
        const int8_t *arow = p.A + static_cast<size_t>(m) * p.lda;
        for (uint32_t n = tile.n0; n < tile.n0 + tile.tn; ++n) {
            int32_t acc = 0;
            for (uint32_t k = 0; k < p.K; ++k)
                acc += static_cast<int32_t>(arow[k]) *
                       static_cast<int32_t>(p.B.original[
                           static_cast<size_t>(k) * p.B.N + n]);
            store_i8_c(p, m, n, acc);
        }
    }
}

#if VNPU_X86
VNPU_TARGET_AVX2
static inline int32_t hsum256_epi32(__m256i v)
{
    int32_t tmp[8];
    _mm256_storeu_si256(reinterpret_cast<__m256i *>(tmp), v);
    int32_t sum = 0;
    for (int32_t x : tmp) sum += x;
    return sum;
}

VNPU_TARGET_AVX2
static void gemm_i8_avx2_tile(const GemmI8Params &p, const TileJob &tile)
{
    for (uint32_t m = tile.m0; m < tile.m0 + tile.tm; ++m) {
        const int8_t *arow = p.A + static_cast<size_t>(m) * p.lda;
        for (uint32_t n = tile.n0; n < tile.n0 + tile.tn; ++n) {
            __m256i acc = _mm256_setzero_si256();
            uint32_t k = 0;
            for (; k + 8 <= p.K; k += 8) {
                uint64_t av = 0, bv = 0;
                std::memcpy(&av, arow + k, sizeof(av));
                for (uint32_t j = 0; j < 8; ++j)
                    reinterpret_cast<uint8_t *>(&bv)[j] =
                        static_cast<uint8_t>(p.B.original[
                            static_cast<size_t>(k + j) * p.B.N + n]);
                __m128i a8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(&av));
                __m128i b8 = _mm_loadl_epi64(
                    reinterpret_cast<const __m128i *>(&bv));
                __m256i a32 = _mm256_cvtepi8_epi32(a8);
                __m256i b32 = _mm256_cvtepi8_epi32(b8);
                acc = _mm256_add_epi32(acc, _mm256_mullo_epi32(a32, b32));
            }
            int32_t dot = hsum256_epi32(acc);
            for (; k < p.K; ++k)
                dot += static_cast<int32_t>(arow[k]) *
                       static_cast<int32_t>(p.B.original[
                           static_cast<size_t>(k) * p.B.N + n]);
            store_i8_c(p, m, n, dot);
        }
    }
}

/* AVX512-VNNI: A is int8 + 128 (u8), B is signed int8 in [16][4] blocks.
 * The extra 128*sum(B) term is removed with precomputed column sums. */
VNPU_TARGET_VNNI
static inline __m512i make_vnni_avec(const int8_t *row, uint32_t K,
                                     uint32_t kbase)
{
    uint8_t ab[4];
    for (uint32_t j = 0; j < 4; ++j) {
        int16_t v = (kbase + j < K)
            ? static_cast<int16_t>(row[kbase + j]) + 128
            : 128;
        ab[j] = static_cast<uint8_t>(v);
    }
    uint32_t aword;
    std::memcpy(&aword, ab, sizeof(aword));
    return _mm512_set1_epi32(static_cast<int>(aword));
}

VNPU_TARGET_VNNI
static void gemm_i8_vnni_tile(const GemmI8Params &p, const TileJob &tile,
                              uint8_t *scratch, uint64_t scratch_bytes)
{
    constexpr uint32_t MR = 4;
    constexpr uint32_t NB = 4; /* process 64 output columns per tile pass */
    if (tile.n0 & 15u) {
        gemm_i8_scalar_tile(p, tile);
        return;
    }
    const uint32_t k4_groups = p.B.Kpad / 4;
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = std::min<uint32_t>(tile.n0 + tile.tn, p.N);
    const uint64_t avec_bytes = static_cast<uint64_t>(MR) *
                                k4_groups * 64;
    /* Precomputing broadcast A vectors pays off only for large GEMMs; small
     * decode-shaped calls are faster with on-the-fly construction. */
    bool precompute = scratch && scratch_bytes >= avec_bytes + 64 &&
                      p.K >= 512 && (nend - tile.n0) >= 256;
    __m512i *Avec = nullptr;
    if (precompute) {
        uintptr_t aligned = align_up(
            reinterpret_cast<uintptr_t>(scratch), VNPU_CACHELINE_BYTES);
        Avec = reinterpret_cast<__m512i *>(aligned);
    }
    for (uint32_t m0 = tile.m0; m0 < mend; m0 += MR) {
        uint32_t rm = std::min<uint32_t>(MR, mend - m0);
        const int8_t *arow[MR] = {};
        for (uint32_t i = 0; i < rm; ++i)
            arow[i] = p.A + static_cast<size_t>(m0 + i) * p.lda;
        if (precompute) {
            for (uint32_t i = 0; i < rm; ++i) {
                for (uint32_t g = 0; g < k4_groups; ++g) {
                    __m512i v = make_vnni_avec(arow[i], p.K, g * 4);
                    std::memcpy(Avec + static_cast<size_t>(i) *
                                          k4_groups + g,
                                &v, sizeof(v));
                }
            }
        }
        for (uint32_t n0 = tile.n0; n0 < nend; n0 += 16 * NB) {
            uint32_t nb = std::min<uint32_t>(
                NB, (nend - n0 + 15u) / 16u);
            const uint8_t *bp[NB] = {};
            uint32_t counts[NB] = {};
            for (uint32_t b = 0; b < nb; ++b) {
                uint32_t col = n0 + 16 * b;
                bp[b] = p.B.packed +
                    (static_cast<size_t>(col / 16) * k4_groups) * 64;
                counts[b] = std::min<uint32_t>(16u, nend - col);
            }
            __m512i acc[MR][NB];
            for (uint32_t i = 0; i < MR; ++i)
                for (uint32_t b = 0; b < NB; ++b)
                    acc[i][b] = _mm512_setzero_si512();
            for (uint32_t g = 0; g < k4_groups; ++g) {
                __m512i bvec[NB];
                for (uint32_t b = 0; b < nb; ++b)
                    bvec[b] = _mm512_loadu_si512(
                        reinterpret_cast<const void *>(bp[b] + g * 64));
                uint32_t kbase = g * 4;
                for (uint32_t i = 0; i < rm; ++i) {
                    __m512i avec;
                    if (precompute) {
                        avec = _mm512_loadu_si512(
                            reinterpret_cast<const void *>(
                                Avec + static_cast<size_t>(i) *
                                           k4_groups + g));
                    } else {
                        avec = make_vnni_avec(arow[i], p.K, kbase);
                    }
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC unroll 4
#endif
                    for (uint32_t b = 0; b < nb; ++b)
                        acc[i][b] = _mm512_dpbusd_epi32(
                            acc[i][b], avec, bvec[b]);
                }
            }
            float tmp[16];
            for (uint32_t b = 0; b < nb; ++b) {
                uint32_t col = n0 + 16 * b;
                __m512i sums = _mm512_loadu_si512(
                    reinterpret_cast<const void *>(p.B.column_sums + col));
                __m512i corr = _mm512_slli_epi32(sums, 7);
                __m512 scalesb = _mm512_loadu_ps(p.B.scales + col);
                for (uint32_t i = 0; i < rm; ++i) {
                    acc[i][b] = _mm512_sub_epi32(acc[i][b], corr);
                    __m512 dotsf = _mm512_cvtepi32_ps(acc[i][b]);
                    __m512 scalea = _mm512_set1_ps(
                        p.scale_a ? p.scale_a[m0 + i] : 1.0f);
                    __m512 out = _mm512_mul_ps(
                        _mm512_mul_ps(dotsf, scalesb), scalea);
                    std::memcpy(tmp, &out, sizeof(out));
                    for (uint32_t j = 0; j < counts[b]; ++j) {
                        float value = tmp[j];
                        if (p.bias) value += p.bias[col + j];
                        p.C[static_cast<size_t>(m0 + i) * p.ldc + col + j] =
                            apply_epilogue(value, p.epilogue);
                    }
                }
            }
        }
    }
}
#endif /* VNPU_X86 */

static void gemm_i8_cell_tile(const GemmI8Params &p, const TileJob &tile,
                              uint8_t *scratch, uint64_t scratch_bytes)
{
    uint32_t MR = 4;
    uint32_t NR = 16;
    uint32_t BK = std::min<uint32_t>(tile.tile_k ? tile.tile_k : 64, 256);
    const uint64_t a_bytes = static_cast<uint64_t>(MR) * BK;
    const uint64_t b_bytes = static_cast<uint64_t>(BK) * NR;
    if (!scratch || a_bytes + b_bytes + 128 > scratch_bytes) {
        gemm_i8_scalar_tile(p, tile);
        return;
    }
    uintptr_t base = align_up(reinterpret_cast<uintptr_t>(scratch),
                              VNPU_CACHELINE_BYTES);
    int8_t *A_s = reinterpret_cast<int8_t *>(base);
    int8_t *B_s = reinterpret_cast<int8_t *>(base + a_bytes);
    const uint32_t mend = tile.m0 + tile.tm;
    const uint32_t nend = tile.n0 + tile.tn;
    for (uint32_t m0 = tile.m0; m0 < mend; m0 += MR) {
        uint32_t rm = std::min<uint32_t>(MR, mend - m0);
        for (uint32_t n0 = tile.n0; n0 < nend; n0 += NR) {
            uint32_t rn = std::min<uint32_t>(NR, nend - n0);
            int32_t acc[4][16] = {};
            for (uint32_t k0 = 0; k0 < p.K; k0 += BK) {
                uint32_t rk = std::min<uint32_t>(BK, p.K - k0);
                for (uint32_t i = 0; i < rm; ++i)
                    std::memcpy(A_s + i * BK,
                                p.A + static_cast<size_t>(m0 + i) *
                                          p.lda + k0,
                                rk);
                for (uint32_t k = 0; k < rk; ++k)
                    for (uint32_t j = 0; j < rn; ++j)
                        B_s[k * NR + j] = p.B.original[
                            static_cast<size_t>(k0 + k) * p.B.N + n0 + j];
                for (uint32_t i = 0; i < rm; ++i)
                    for (uint32_t k = 0; k < rk; ++k) {
                        int32_t av = A_s[i * BK + k];
                        for (uint32_t j = 0; j < rn; ++j)
                            acc[i][j] += av *
                                static_cast<int32_t>(B_s[k * NR + j]);
                    }
            }
            for (uint32_t i = 0; i < rm; ++i)
                for (uint32_t j = 0; j < rn; ++j)
                    store_i8_c(p, m0 + i, n0 + j, acc[i][j]);
        }
    }
}

static void gemm_i8_dispatch_tile(const GemmI8Params &p,
                                  const TileJob &tile,
                                  uint32_t worker_id)
{
    if (tile.cull) {
        for (uint32_t m = tile.m0; m < tile.m0 + tile.tm; ++m)
            for (uint32_t n = tile.n0; n < tile.n0 + tile.tn; ++n)
                store_i8_c(p, m, n, 0);
        return;
    }
    if (p.unit == VNPU_UNIT_CELL) {
        uint8_t *scratch = nullptr;
        uint64_t bytes = 0;
        if (p.local_stores && worker_id < p.local_stores->size()) {
            scratch = (*p.local_stores)[worker_id].data();
            bytes = (*p.local_stores)[worker_id].size();
        }
        gemm_i8_cell_tile(p, tile, scratch, bytes);
        return;
    }
#if VNPU_X86
    if (p.has_vnni && p.isa == ISA_AVX512) {
        uint8_t *scratch = nullptr;
        uint64_t bytes = 0;
        if (p.local_stores && worker_id < p.local_stores->size()) {
            scratch = (*p.local_stores)[worker_id].data();
            bytes = (*p.local_stores)[worker_id].size();
        }
        gemm_i8_vnni_tile(p, tile, scratch, bytes);
        return;
    }
    if (isa_has_avx2(p.isa)) {
        gemm_i8_avx2_tile(p, tile);
        return;
    }
#endif
    gemm_i8_scalar_tile(p, tile);
}

} /* anonymous namespace */

void kernel_gemm_f32_range(const GemmF32Params &p, size_t begin, size_t end,
                           uint32_t worker_id)
{
    if (!p.tiles) return;
    for (size_t i = begin; i < end; ++i)
        gemm_f32_dispatch_tile(p, (*p.tiles)[i], worker_id);
}

void kernel_gemm_i8_range(const GemmI8Params &p, size_t begin, size_t end,
                          uint32_t worker_id)
{
    if (!p.tiles) return;
    for (size_t i = begin; i < end; ++i)
        gemm_i8_dispatch_tile(p, (*p.tiles)[i], worker_id);
}

} /* namespace vnpu */
