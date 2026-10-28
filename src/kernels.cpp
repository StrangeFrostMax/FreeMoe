#include "fm/kernels.hpp"

#include "fm/backend.hpp"
#include "fm/quant.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(__i386__)
#define FM_X86 1
#include <immintrin.h>
#endif

namespace fm {
namespace {

// Below this many multiply-adds a threaded dispatch costs more than it saves.
constexpr uint64_t THREAD_MIN_WORK = 1ull << 16;

// std::thread::hardware_concurrency() is not free (it may consult sysfs), so it
// must never be called on a per-GEMV path. Cached once per process.
uint32_t hardware_threads() {
    static const uint32_t cached = []() {
        unsigned n = std::thread::hardware_concurrency();
        if (n == 0) n = 1;
        return std::min<uint32_t>(n, MAX_KERNEL_THREADS);
    }();
    return cached;
}

bool kernel_trace_enabled() {
    static const bool enabled = std::getenv("FM_KERNEL_TRACE") != nullptr;
    return enabled;
}

} // namespace

bool host_has_avx2_fma() {
#if defined(FM_X86) && (defined(__GNUC__) || defined(__clang__))
    // __builtin_cpu_supports requires a string literal, so probe directly.
    static const bool ok = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx2") != 0 && __builtin_cpu_supports("fma") != 0;
    }();
    return ok;
#else
    return false;
#endif
}

bool host_has_avx512() {
#if defined(FM_X86) && (defined(__GNUC__) || defined(__clang__))
    static const bool ok = []() {
        __builtin_cpu_init();
        return __builtin_cpu_supports("avx512f") != 0;
    }();
    return ok;
#else
    return false;
#endif
}

const char *kernel_isa_name(KernelIsa isa) {
    switch (isa) {
    case KernelIsa::Scalar: return "scalar";
    case KernelIsa::Avx2Fma: return "avx2_fma";
    case KernelIsa::Avx512: return "avx512";
    case KernelIsa::Auto: return "auto";
    }
    return "unknown";
}

KernelIsa parse_kernel_isa(const std::string &name) {
    if (name == "auto") return KernelIsa::Auto;
    if (name == "scalar") return KernelIsa::Scalar;
    if (name == "avx2" || name == "avx2_fma") return KernelIsa::Avx2Fma;
    if (name == "avx512") return KernelIsa::Avx512;
    fail(ErrorCode::INVALID_ARGUMENT, "unknown kernel ISA profile: " + name);
}

KernelIsa resolve_kernel_isa(KernelIsa requested) {
    if (requested == KernelIsa::Scalar) return KernelIsa::Scalar;
    // Zen 3 policy: never schedule AVX-512, use dual 256-bit AVX2/FMA3.
    const bool zen3 = host_is_zen3();
    if (requested == KernelIsa::Avx512) {
        if (host_has_avx512() && !zen3) return KernelIsa::Avx512;
        if (host_has_avx2_fma()) return KernelIsa::Avx2Fma;
        return KernelIsa::Scalar;
    }
    if (requested == KernelIsa::Avx2Fma) {
        return host_has_avx2_fma() ? KernelIsa::Avx2Fma : KernelIsa::Scalar;
    }
    // Auto
    if (zen3) return host_has_avx2_fma() ? KernelIsa::Avx2Fma : KernelIsa::Scalar;
    if (host_has_avx512()) return KernelIsa::Avx512;
    if (host_has_avx2_fma()) return KernelIsa::Avx2Fma;
    return KernelIsa::Scalar;
}

KernelIsa resolved_isa(const KernelConfig &config) {
    return resolve_kernel_isa(config.requested);
}

uint32_t resolved_threads(const KernelConfig &config) {
    if (config.threads == 0) return hardware_threads();
    return std::min<uint32_t>(config.threads, MAX_KERNEL_THREADS);
}

void gemv_scalar(const float *weights, const float *x, float *y,
                 uint32_t out_dim, uint32_t in_dim) {
    for (uint32_t o = 0; o < out_dim; ++o) {
        const float *row = weights + static_cast<size_t>(o) * in_dim;
        float acc = 0.0f;
        for (uint32_t k = 0; k < in_dim; ++k) acc += row[k] * x[k];
        y[o] = acc;
    }
}

#if defined(FM_X86) && (defined(__GNUC__) || defined(__clang__))

// Two independent 256-bit accumulator chains, matching the dual-pipe strategy
// used in the vendored VNPU kernels. Each row is summed in 8-lane vectors and
// horizontally reduced once at the end of the row.
__attribute__((target("avx2,fma")))
void gemv_avx2_row(const float *row, const float *x, uint32_t in_dim, float *out) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t k = 0;
    for (; k + 16 <= in_dim; k += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + k), _mm256_loadu_ps(x + k), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(row + k + 8), _mm256_loadu_ps(x + k + 8), acc1);
    }
    for (; k + 8 <= in_dim; k += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(row + k), _mm256_loadu_ps(x + k), acc0);
    __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 lo = _mm256_castps256_ps128(acc);
    __m128 hi = _mm256_extractf128_ps(acc, 1);
    __m128 sum = _mm_add_ps(lo, hi);
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float total = _mm_cvtss_f32(sum);
    for (; k < in_dim; ++k) total += row[k] * x[k];
    *out = total;
}

__attribute__((target("avx2,fma")))
void gemv_avx2_range(const float *weights, const float *x, float *y,
                     uint32_t begin, uint32_t end, uint32_t in_dim) {
    for (uint32_t o = begin; o < end; ++o)
        gemv_avx2_row(weights + static_cast<size_t>(o) * in_dim, x, in_dim, y + o);
}

// GCC 12 emits a false-positive -Wmaybe-uninitialized from the AVX-512
// intrinsics themselves; the same suppression is used in the vendored VNPU.
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#endif
__attribute__((target("avx512f")))
void gemv_avx512_range(const float *weights, const float *x, float *y,
                       uint32_t begin, uint32_t end, uint32_t in_dim) {
    for (uint32_t o = begin; o < end; ++o) {
        const float *row = weights + static_cast<size_t>(o) * in_dim;
        __m512 acc0 = _mm512_setzero_ps();
        __m512 acc1 = _mm512_setzero_ps();
        uint32_t k = 0;
        for (; k + 32 <= in_dim; k += 32) {
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(row + k), _mm512_loadu_ps(x + k), acc0);
            acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(row + k + 16), _mm512_loadu_ps(x + k + 16), acc1);
        }
        for (; k + 16 <= in_dim; k += 16)
            acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(row + k), _mm512_loadu_ps(x + k), acc0);
        float total = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
        for (; k < in_dim; ++k) total += row[k] * x[k];
        y[o] = total;
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

__attribute__((target("avx2,fma")))
float dot_avx2(const float *a, const float *b, uint32_t n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    uint32_t k = 0;
    for (; k + 16 <= n; k += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8), _mm256_loadu_ps(b + k + 8), acc1);
    }
    for (; k + 8 <= n; k += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k), _mm256_loadu_ps(b + k), acc0);
    __m256 acc = _mm256_add_ps(acc0, acc1);
    __m128 sum = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    sum = _mm_hadd_ps(sum, sum);
    sum = _mm_hadd_ps(sum, sum);
    float total = _mm_cvtss_f32(sum);
    for (; k < n; ++k) total += a[k] * b[k];
    return total;
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
#pragma GCC diagnostic ignored "-Wuninitialized"
#endif
__attribute__((target("avx512f")))
float dot_avx512(const float *a, const float *b, uint32_t n) {
    __m512 acc0 = _mm512_setzero_ps();
    __m512 acc1 = _mm512_setzero_ps();
    uint32_t k = 0;
    for (; k + 32 <= n; k += 32) {
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k), acc0);
        acc1 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k + 16), _mm512_loadu_ps(b + k + 16), acc1);
    }
    for (; k + 16 <= n; k += 16)
        acc0 = _mm512_fmadd_ps(_mm512_loadu_ps(a + k), _mm512_loadu_ps(b + k), acc0);
    float total = _mm512_reduce_add_ps(_mm512_add_ps(acc0, acc1));
    for (; k < n; ++k) total += a[k] * b[k];
    return total;
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#endif

float dot_isa(KernelIsa isa, const float *a, const float *b, uint32_t n) {
#if defined(FM_X86) && (defined(__GNUC__) || defined(__clang__))
    if (isa == KernelIsa::Avx512 && host_has_avx512()) return dot_avx512(a, b, n);
    if (isa == KernelIsa::Avx2Fma && host_has_avx2_fma()) return dot_avx2(a, b, n);
#endif
    (void)isa;
    float total = 0.0f;
    for (uint32_t k = 0; k < n; ++k) total += a[k] * b[k];
    return total;
}

void gemv_isa(KernelIsa isa, const float *weights, const float *x, float *y,
              uint32_t out_dim, uint32_t in_dim) {
    if (out_dim == 0 || in_dim == 0) return;
#if defined(FM_X86) && (defined(__GNUC__) || defined(__clang__))
    if (isa == KernelIsa::Avx512 && host_has_avx512()) {
        gemv_avx512_range(weights, x, y, 0, out_dim, in_dim);
        return;
    }
    if (isa == KernelIsa::Avx2Fma && host_has_avx2_fma()) {
        gemv_avx2_range(weights, x, y, 0, out_dim, in_dim);
        return;
    }
#endif
    (void)isa;
    gemv_scalar(weights, x, y, out_dim, in_dim);
}

// ---------------------------------------------------------------------------
// Persistent worker pool
// ---------------------------------------------------------------------------

struct WorkerPool::Impl {
    std::thread thread;
    std::mutex mutex;
    std::condition_variable wake;
    std::condition_variable done;
    const std::function<void(uint32_t, uint32_t)> *body = nullptr;
    uint32_t chunk_begin = 0;
    uint32_t chunk_end = 0;
    uint64_t generation = 0;
    uint64_t seen = 0;
    uint32_t pending = 0;
    bool stop = false;
};

WorkerPool::WorkerPool() {
    const uint32_t workers = hardware_threads() > 1 ? hardware_threads() - 1 : 0;
    for (uint32_t i = 0; i < workers; ++i) {
        auto *impl = new Impl();
        Impl *ptr = impl;
        impl->thread = std::thread([ptr]() {
            uint64_t seen = 0;
            for (;;) {
                const std::function<void(uint32_t, uint32_t)> *body = nullptr;
                uint32_t begin = 0;
                uint32_t end = 0;
                {
                    std::unique_lock<std::mutex> lock(ptr->mutex);
                    ptr->wake.wait(lock, [ptr, seen]() {
                        return ptr->stop || ptr->generation != seen;
                    });
                    if (ptr->stop) return;
                    seen = ptr->generation;
                    body = ptr->body;
                    begin = ptr->chunk_begin;
                    end = ptr->chunk_end;
                }
                if (body && begin < end) (*body)(begin, end);
                {
                    std::lock_guard<std::mutex> lock(ptr->mutex);
                    if (ptr->pending > 0) --ptr->pending;
                }
                ptr->done.notify_all();
            }
        });
        workers_.push_back(impl);
    }
}

WorkerPool::~WorkerPool() {
    for (Impl *impl : workers_) {
        {
            std::lock_guard<std::mutex> lock(impl->mutex);
            impl->stop = true;
        }
        impl->wake.notify_all();
        if (impl->thread.joinable()) impl->thread.join();
        delete impl;
    }
}

WorkerPool &WorkerPool::instance() {
    static WorkerPool pool;
    return pool;
}

void WorkerPool::parallel_for(uint32_t total, uint32_t max_threads,
                              const std::function<void(uint32_t, uint32_t)> &body) {
    if (total == 0) return;
    uint32_t n = std::min<uint32_t>(max_threads, total);
    n = std::min<uint32_t>(n, static_cast<uint32_t>(workers_.size()) + 1);
    if (n <= 1) {
        body(0, total);
        return;
    }
    const uint32_t chunk = (total + n - 1) / n;
    std::vector<std::pair<uint32_t, uint32_t>> ranges;
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t begin = i * chunk;
        uint32_t end = std::min<uint32_t>(total, begin + chunk);
        if (begin < end) ranges.emplace_back(begin, end);
    }
    if (ranges.size() <= 1) {
        body(0, total);
        return;
    }
    {
        std::lock_guard<std::mutex> lock(workers_[0]->mutex);
        // One shared task description in worker 0's slot; every worker reads its
        // own slot, so publish per-worker chunks.
    }
    // Publish the task to every worker.
    for (size_t i = 0; i < workers_.size(); ++i) {
        Impl *impl = workers_[i];
        std::lock_guard<std::mutex> lock(impl->mutex);
        impl->body = &body;
        if (i + 1 < ranges.size()) {
            impl->chunk_begin = ranges[i + 1].first;
            impl->chunk_end = ranges[i + 1].second;
        } else {
            impl->chunk_begin = 0;
            impl->chunk_end = 0;
        }
        impl->pending = static_cast<uint32_t>(ranges.size() - 1);
        ++impl->generation;
    }
    for (Impl *impl : workers_) impl->wake.notify_all();

    // Main thread executes the first chunk.
    body(ranges[0].first, ranges[0].second);

    std::unique_lock<std::mutex> lock(workers_[0]->mutex);
    workers_[0]->done.wait(lock, [this]() { return workers_[0]->pending == 0; });
    for (Impl *impl : workers_) impl->body = nullptr;
}

void gemv(const KernelConfig &config, const float *weights, const float *x, float *y,
          uint32_t out_dim, uint32_t in_dim) {
    if (out_dim == 0 || in_dim == 0) return;
    const KernelIsa isa = resolved_isa(config);
    const uint32_t threads = resolved_threads(config);
    const uint64_t work = static_cast<uint64_t>(out_dim) * in_dim;
    if (kernel_trace_enabled())
        std::fprintf(stderr, "gemv out=%u in=%u work=%llu threads=%u isa=%s\n", out_dim, in_dim,
                     static_cast<unsigned long long>(work), threads, kernel_isa_name(isa));
    if (threads <= 1 || work < THREAD_MIN_WORK) {
        gemv_isa(isa, weights, x, y, out_dim, in_dim);
        return;
    }
    if (isa == KernelIsa::Scalar) {
        WorkerPool::instance().parallel_for(out_dim, threads, [&](uint32_t begin, uint32_t end) {
            gemv_scalar(weights + static_cast<size_t>(begin) * in_dim, x, y + begin,
                        end - begin, in_dim);
        });
        return;
    }
#if defined(FM_X86) && (defined(__GNUC__) || defined(__clang__))
    if (isa == KernelIsa::Avx512) {
        WorkerPool::instance().parallel_for(out_dim, threads, [&](uint32_t begin, uint32_t end) {
            gemv_avx512_range(weights, x, y, begin, end, in_dim);
        });
        return;
    }
    if (isa == KernelIsa::Avx2Fma) {
        WorkerPool::instance().parallel_for(out_dim, threads, [&](uint32_t begin, uint32_t end) {
            gemv_avx2_range(weights, x, y, begin, end, in_dim);
        });
        return;
    }
#endif
    WorkerPool::instance().parallel_for(out_dim, threads, [&](uint32_t begin, uint32_t end) {
        gemv_scalar(weights + static_cast<size_t>(begin) * in_dim, x, y + begin,
                    end - begin, in_dim);
    });
}

bool kernel_supports_type(uint32_t type_id) {
    // Derive support from the decoder table instead of keeping a second list
    // that can drift. If a type decodes, the fused scalar path can execute it;
    // the AVX2 kernels are a throughput refinement, not a support requirement.
    return tensor_capability(type_id).execute;
}

namespace {

inline uint16_t read_u16_le(const uint8_t *p) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline float read_f32_le(const uint8_t *p) {
    float v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// Decodes a single dense element (block_size == 1 types).
inline float dequantize_one(uint32_t type_id, const uint8_t *src) {
    float out = 0.0f;
    dequantize_block(type_id, src, &out);
    return out;
}

// Six-bit scale/min unpacking, identical to the reference decoder.
inline void k_scale_min(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = static_cast<uint8_t>(q[j] & 63);
        *m = static_cast<uint8_t>(q[j + 4] & 63);
    } else {
        *d = static_cast<uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

// ---------------------------------------------------------------------------
// Quantised row kernels
//
// Two layers:
//   * a fused scalar path that decodes one block into a stack buffer and dots
//     it immediately (valid for every supported type, the oracle for the rest);
//   * fused AVX2 block-dot kernels for the formats that dominate real
//     checkpoints (Q8_0, Q4_0, Q4_K) plus F16/BF16 rows.
//
// The AVX2 form is the one that matters on the Zen 3 target, where AVX-512 is
// deliberately not used. Decoding into a scratch row and then summing costs two
// passes over the row and measured ~2 GFLOPS; fusing keeps the decoded lanes in
// registers.
// ---------------------------------------------------------------------------

struct RowDotCtx {
    KernelIsa isa = KernelIsa::Scalar;
    uint32_t type_id = 0;
    uint32_t in_dim = 0;
    uint32_t block_size = 1;
    uint32_t block_bytes = 4;
};

using RowDotFn = float (*)(const RowDotCtx &, const uint8_t *row, const float *x);

float row_dot_fused_scalar(const RowDotCtx &ctx, const uint8_t *row, const float *x) {
    const uint32_t blocks = ctx.in_dim / ctx.block_size;
    float total = 0.0f;
    // 256 values is the largest block in any supported format.
    float decoded[256];
    for (uint32_t b = 0; b < blocks; ++b) {
        const float *xb = x + b * ctx.block_size;
        if (ctx.block_size == 1) {
            // Dense element types (F16/BF16): no block structure to exploit.
            total += dequantize_one(ctx.type_id, row + b * ctx.block_bytes) * xb[0];
            continue;
        }
        dequantize_block(ctx.type_id, row + b * ctx.block_bytes, decoded);
        total += dot_isa(ctx.isa, decoded, xb, ctx.block_size);
    }
    return total;
}

#if defined(FM_X86) && (defined(__GNUC__) || defined(__clang__))

#define FM_AVX2_TARGET __attribute__((target("avx2,fma,f16c")))

FM_AVX2_TARGET inline float hsum256(__m256 v) {
    __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
    s = _mm_hadd_ps(s, s);
    s = _mm_hadd_ps(s, s);
    return _mm_cvtss_f32(s);
}

// 8 consecutive int8 values -> 8 floats, fused multiply-add into acc.
FM_AVX2_TARGET inline __m256 fma_i8x8(__m256 acc, const uint8_t *q, const float *x) {
    const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(q));
    const __m256 qf = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(raw));
    return _mm256_fmadd_ps(qf, _mm256_loadu_ps(x), acc);
}

// 16 packed bytes -> 32 nibbles. Bytes 0..7 hold elements 0..7 (low nibble) and
// 16..23 (high nibble); bytes 8..15 hold elements 8..15 and 24..31. Each 8-byte
// half needs its own conversion because a 256-bit lane only holds 8 floats, so
// all four accumulators must be summed or half the block is silently dropped.
struct NibbleAcc {
    __m256 lo0, lo1, hi0, hi1;
};

FM_AVX2_TARGET inline NibbleAcc fma_nibbles16(const uint8_t *q, const float *x) {
    const __m128i mask = _mm_set1_epi8(0x0F);
    NibbleAcc acc;
    for (int half = 0; half < 2; ++half) {
        const __m128i raw = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(q + half * 8));
        const __m256 lo = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_and_si128(raw, mask)));
        const __m256 hi = _mm256_cvtepi32_ps(
            _mm256_cvtepi8_epi32(_mm_and_si128(_mm_srli_epi16(raw, 4), mask)));
        if (half == 0) {
            acc.lo0 = _mm256_mul_ps(lo, _mm256_loadu_ps(x));
            acc.hi0 = _mm256_mul_ps(hi, _mm256_loadu_ps(x + 16));
        } else {
            acc.lo1 = _mm256_mul_ps(lo, _mm256_loadu_ps(x + 8));
            acc.hi1 = _mm256_mul_ps(hi, _mm256_loadu_ps(x + 24));
        }
    }
    return acc;
}

// Q8_0: half d, 32 int8. dot = d * sum(q * x).
FM_AVX2_TARGET float row_dot_q8_0_avx2(const RowDotCtx &ctx, const uint8_t *row,
                                       const float *x) {
    const uint32_t blocks = ctx.in_dim / 32;
    __m256 total = _mm256_setzero_ps();
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *block = row + b * 34;
        const float d = fp16_to_f32(read_u16_le(block));
        __m256 acc = _mm256_setzero_ps();
        const uint8_t *q = block + 2;
        const float *xb = x + b * 32;
        for (int k = 0; k < 32; k += 8) acc = fma_i8x8(acc, q + k, xb + k);
        total = _mm256_fmadd_ps(acc, _mm256_set1_ps(d), total);
    }
    return hsum256(total);
}

// Q8_1: float d, float s, 32 int8. dot = d * sum(q * x).
FM_AVX2_TARGET float row_dot_q8_1_avx2(const RowDotCtx &ctx, const uint8_t *row,
                                       const float *x) {
    const uint32_t blocks = ctx.in_dim / 32;
    __m256 total = _mm256_setzero_ps();
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *block = row + b * 40;
        const float d = read_f32_le(block);
        __m256 acc = _mm256_setzero_ps();
        const uint8_t *q = block + 8;
        const float *xb = x + b * 32;
        for (int k = 0; k < 32; k += 8) acc = fma_i8x8(acc, q + k, xb + k);
        total = _mm256_fmadd_ps(acc, _mm256_set1_ps(d), total);
    }
    return hsum256(total);
}

// Q4_0: half d, 16 bytes of nibbles. Values are (q - 8) * d.
FM_AVX2_TARGET float row_dot_q4_0_avx2(const RowDotCtx &ctx, const uint8_t *row,
                                       const float *x) {
    const uint32_t blocks = ctx.in_dim / 32;
    const __m256 eight = _mm256_set1_ps(8.0f);
    __m256 total = _mm256_setzero_ps();
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *block = row + b * 18;
        const float d = fp16_to_f32(read_u16_le(block));
        const float *xb = x + b * 32;
        const NibbleAcc nib = fma_nibbles16(block + 2, xb);
        // (q - 8) * x summed == sum(q * x) - 8 * sum(x)
        const __m256 lo = _mm256_add_ps(nib.lo0, nib.lo1);
        const __m256 hi = _mm256_add_ps(nib.hi0, nib.hi1);
        const __m256 centred = _mm256_sub_ps(
            _mm256_add_ps(lo, hi),
            _mm256_mul_ps(eight, _mm256_add_ps(
                _mm256_add_ps(_mm256_loadu_ps(xb), _mm256_loadu_ps(xb + 8)),
                _mm256_add_ps(_mm256_loadu_ps(xb + 16), _mm256_loadu_ps(xb + 24)))));
        total = _mm256_fmadd_ps(centred, _mm256_set1_ps(d), total);
    }
    return hsum256(total);
}

// Q4_1: half d, half m, 16 bytes of nibbles. Values are q * d + m.
FM_AVX2_TARGET float row_dot_q4_1_avx2(const RowDotCtx &ctx, const uint8_t *row,
                                       const float *x) {
    const uint32_t blocks = ctx.in_dim / 32;
    float total = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *block = row + b * 20;
        const float d = fp16_to_f32(read_u16_le(block));
        const float m = fp16_to_f32(read_u16_le(block + 2));
        const float *xb = x + b * 32;
        const NibbleAcc nib = fma_nibbles16(block + 4, xb);
        const __m256 sx = _mm256_add_ps(
            _mm256_add_ps(_mm256_loadu_ps(xb), _mm256_loadu_ps(xb + 8)),
            _mm256_add_ps(_mm256_loadu_ps(xb + 16), _mm256_loadu_ps(xb + 24)));
        const float qx = hsum256(_mm256_add_ps(_mm256_add_ps(nib.lo0, nib.lo1),
                                               _mm256_add_ps(nib.hi0, nib.hi1)));
        total += d * qx + m * hsum256(sx);
    }
    return total;
}

// Q4_K: half d, half min, 12 bytes of 6-bit scales, 128 bytes of nibbles.
// Per 64-value group the low nibbles use (d*sc, min*m) and the high nibbles the
// next pair, so each 32-value sub-block is summed separately.
FM_AVX2_TARGET float row_dot_q4_k_avx2(const RowDotCtx &ctx, const uint8_t *row,
                                       const float *x) {
    const uint32_t blocks = ctx.in_dim / 256;
    float total = 0.0f;
    for (uint32_t b = 0; b < blocks; ++b) {
        const uint8_t *block = row + b * 144;
        const float d = fp16_to_f32(read_u16_le(block));
        const float dmin = fp16_to_f32(read_u16_le(block + 2));
        const uint8_t *scales = block + 4;
        const uint8_t *qs = block + 16;
        const float *xb = x + b * 256;
        int is = 0;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc1, m1, sc2, m2;
            k_scale_min(is + 0, scales, &sc1, &m1);
            k_scale_min(is + 1, scales, &sc2, &m2);
            const uint8_t *q = qs;
            const float *group = xb + j;
            __m256 acc_lo = _mm256_setzero_ps();
            __m256 acc_hi = _mm256_setzero_ps();
            for (int k = 0; k < 32; k += 8) {
                const __m256i raw = _mm256_cvtepu8_epi32(
                    _mm_loadl_epi64(reinterpret_cast<const __m128i *>(q + k)));
                const __m256i lo = _mm256_and_si256(raw, _mm256_set1_epi32(0x0F));
                const __m256i hi = _mm256_and_si256(_mm256_srli_epi32(raw, 4),
                                                    _mm256_set1_epi32(0x0F));
                acc_lo = _mm256_fmadd_ps(_mm256_cvtepi32_ps(lo),
                                         _mm256_loadu_ps(group + k), acc_lo);
                acc_hi = _mm256_fmadd_ps(_mm256_cvtepi32_ps(hi),
                                         _mm256_loadu_ps(group + 32 + k), acc_hi);
            }
            const __m256 sx_lo = _mm256_add_ps(_mm256_loadu_ps(group + 0),
                                               _mm256_loadu_ps(group + 8));
            const __m256 sx_lo2 = _mm256_add_ps(_mm256_loadu_ps(group + 16),
                                                _mm256_loadu_ps(group + 24));
            const __m256 sx_hi = _mm256_add_ps(_mm256_loadu_ps(group + 32),
                                               _mm256_loadu_ps(group + 40));
            const __m256 sx_hi2 = _mm256_add_ps(_mm256_loadu_ps(group + 48),
                                                _mm256_loadu_ps(group + 56));
            const float sum_x_lo = hsum256(_mm256_add_ps(sx_lo, sx_lo2));
            const float sum_x_hi = hsum256(_mm256_add_ps(sx_hi, sx_hi2));
            const float part_lo =
                d * static_cast<float>(sc1) * hsum256(acc_lo) - dmin * static_cast<float>(m1) * sum_x_lo;
            const float part_hi =
                d * static_cast<float>(sc2) * hsum256(acc_hi) - dmin * static_cast<float>(m2) * sum_x_hi;
            total += part_lo + part_hi;
            qs += 32;
            is += 2;
        }
    }
    return total;
}

// F16 / BF16 rows: no block structure, so convert 8 lanes at a time.
FM_AVX2_TARGET float row_dot_f16_avx2(const RowDotCtx &ctx, const uint8_t *row,
                                      const float *x) {
    const uint32_t n = ctx.in_dim;
    __m256 total = _mm256_setzero_ps();
    uint32_t k = 0;
    for (; k + 8 <= n; k += 8) {
        const __m128i packed =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + k * 2));
        total = _mm256_fmadd_ps(_mm256_cvtph_ps(packed), _mm256_loadu_ps(x + k), total);
    }
    float tail = hsum256(total);
    for (; k < n; ++k) tail += fp16_to_f32(read_u16_le(row + k * 2)) * x[k];
    return tail;
}

FM_AVX2_TARGET float row_dot_bf16_avx2(const RowDotCtx &ctx, const uint8_t *row,
                                       const float *x) {
    const uint32_t n = ctx.in_dim;
    __m256 total = _mm256_setzero_ps();
    uint32_t k = 0;
    for (; k + 8 <= n; k += 8) {
        const __m128i packed =
            _mm_loadu_si128(reinterpret_cast<const __m128i *>(row + k * 2));
        // BF16 is the top half of an F32, so widening is a shift.
        const __m256i wide = _mm256_slli_epi32(_mm256_cvtepu16_epi32(packed), 16);
        total = _mm256_fmadd_ps(_mm256_castsi256_ps(wide), _mm256_loadu_ps(x + k), total);
    }
    float tail = hsum256(total);
    for (; k < n; ++k) tail += bf16_to_f32(read_u16_le(row + k * 2)) * x[k];
    return tail;
}

#endif // FM_X86

RowDotFn select_row_dot(const RowDotCtx &ctx) {
    // An explicit scalar request must stay scalar: the benchmark columns and the
    // `--kernels scalar` escape hatch both rely on it.
    if (ctx.isa == KernelIsa::Scalar) return row_dot_fused_scalar;
#if defined(FM_X86) && (defined(__GNUC__) || defined(__clang__))
    // The quantised kernels are AVX2/FMA3 only, which is deliberate: AVX2 is the
    // Zen 3 target and is available on every x86-64 host this project supports.
    if (host_has_avx2_fma()) {
        switch (ctx.type_id) {
        case 1: return row_dot_f16_avx2;
        case 30: return row_dot_bf16_avx2;
        case 2: return row_dot_q4_0_avx2;
        case 3: return row_dot_q4_1_avx2;
        case 8: return row_dot_q8_0_avx2;
        case 9: return row_dot_q8_1_avx2;
        case 12: return row_dot_q4_k_avx2;
        default: break;
        }
    }
#else
    (void)ctx;
#endif
    return row_dot_fused_scalar;
}

void gemv_quant_rows(const RowDotCtx &ctx, const uint8_t *weights, const float *x, float *y,
                     uint32_t begin, uint32_t end) {
    const RowDotFn fn = select_row_dot(ctx);
    const uint64_t row_bytes =
        ctx.block_size == 1
            ? static_cast<uint64_t>(ctx.in_dim) * ctx.block_bytes
            : static_cast<uint64_t>(ctx.in_dim / ctx.block_size) * ctx.block_bytes;
    for (uint32_t o = begin; o < end; ++o)
        y[o] = fn(ctx, weights + static_cast<uint64_t>(o) * row_bytes, x);
}

} // namespace

void gemv_quant(const KernelConfig &config, uint32_t type_id, const uint8_t *weights,
                const float *x, float *y, uint32_t out_dim, uint32_t in_dim) {
    if (out_dim == 0 || in_dim == 0) return;
    if (!weights || !x || !y)
        fail(ErrorCode::INVALID_ARGUMENT, "invalid quantised GEMV arguments");
    if (!kernel_supports_type(type_id))
        fail(ErrorCode::UNSUPPORTED_QUANTISATION,
             "no GEMV kernel for tensor type " + std::to_string(type_id));
    const QuantGeometry &geo = quant_geometry(type_id);
    if (in_dim % geo.block_size != 0)
        fail(ErrorCode::SHAPE_MISMATCH,
             "quantised row length is not a whole number of blocks");
    if (type_id == 0) {
        gemv(config, reinterpret_cast<const float *>(weights), x, y, out_dim, in_dim);
        return;
    }
    const KernelIsa isa = resolved_isa(config);
    const uint32_t threads = resolved_threads(config);
    const uint64_t work = static_cast<uint64_t>(out_dim) * in_dim;
    RowDotCtx ctx;
    ctx.isa = isa;
    ctx.type_id = type_id;
    ctx.in_dim = in_dim;
    ctx.block_size = geo.block_size;
    ctx.block_bytes = geo.block_bytes;
    if (threads <= 1 || work < THREAD_MIN_WORK) {
        gemv_quant_rows(ctx, weights, x, y, 0, out_dim);
        return;
    }
    WorkerPool::instance().parallel_for(out_dim, threads, [&](uint32_t begin, uint32_t end) {
        gemv_quant_rows(ctx, weights, x, y, begin, end);
    });
}

GemvBenchmark benchmark_gemv_quant(const KernelConfig &config, uint32_t type_id,
                                   uint32_t out_dim, uint32_t in_dim, uint32_t iterations) {
    const QuantGeometry &geo = quant_geometry(type_id);
    if (in_dim % geo.block_size != 0)
        fail(ErrorCode::INVALID_ARGUMENT, "benchmark width must be block aligned");
    const uint64_t row_bytes = (in_dim / geo.block_size) * static_cast<uint64_t>(geo.block_bytes);
    std::vector<uint8_t> weights(row_bytes * out_dim);
    std::vector<float> x(in_dim, 1.0f);
    std::vector<float> y(out_dim, 0.0f);
    uint32_t state = 0x51ED270Bu;
    for (auto &w : weights) {
        state = state * 1664525u + 1013904223u;
        w = static_cast<uint8_t>((state >> 9) & 0xFF);
    }
    // Keep scales finite: zero every block's scale field.
    for (uint32_t r = 0; r < out_dim; ++r) {
        uint8_t *row = weights.data() + r * row_bytes;
        for (uint64_t b = 0; b < in_dim / geo.block_size; ++b) {
            uint8_t *block = row + b * geo.block_bytes;
            for (uint64_t i = 0; i < geo.block_bytes; ++i) block[i] = 1;
            const float scale = 1e-3f;
            const uint16_t half = 0x1400; // 2^-11
            if (type_id == 9 || type_id == 15) std::memcpy(block, &scale, 4);
            else if (type_id == 10) { std::memcpy(block + 80, &half, 2); std::memcpy(block + 82, &half, 2); }
            else if (type_id == 11) std::memcpy(block + 108, &half, 2);
            else if (type_id == 12 || type_id == 13) { std::memcpy(block, &half, 2); std::memcpy(block + 2, &half, 2); }
            else if (type_id == 14) std::memcpy(block + 208, &half, 2);
            else if (type_id != 0) std::memcpy(block, &half, 2);
        }
    }
    GemvBenchmark result;
    result.isa = kernel_isa_name(resolved_isa(config));
    result.threads = resolved_threads(config);
    result.out_dim = out_dim;
    result.in_dim = in_dim;
    result.iterations = std::max<uint32_t>(1, iterations);
    double best = 1e300;
    for (uint32_t i = 0; i < result.iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        gemv_quant(config, type_id, weights.data(), x.data(), y.data(), out_dim, in_dim);
        const auto stop = std::chrono::steady_clock::now();
        best = std::min(best, std::chrono::duration<double, std::milli>(stop - start).count());
    }
    volatile float sink = 0.0f;
    for (float v : y) sink = sink + v;
    (void)sink;
    result.best_ms = best;
    const double flops = 2.0 * static_cast<double>(out_dim) * in_dim;
    const double bytes = static_cast<double>(row_bytes * out_dim) +
                         static_cast<double>(in_dim) * sizeof(float) +
                         static_cast<double>(out_dim) * sizeof(float);
    result.gflops = best > 0.0 ? flops / (best * 1e6) : 0.0;
    result.gbytes_per_s = best > 0.0 ? bytes / (best * 1e6) : 0.0;
    return result;
}

GemvBenchmark benchmark_gemv(const KernelConfig &config, uint32_t out_dim,
                             uint32_t in_dim, uint32_t iterations) {
    std::vector<float> weights(static_cast<size_t>(out_dim) * in_dim);
    std::vector<float> x(in_dim, 1.0f);
    std::vector<float> y(out_dim, 0.0f);
    uint32_t state = 0x9E3779B9u;
    for (auto &w : weights) {
        state = state * 1664525u + 1013904223u;
        w = static_cast<float>((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
    }
    GemvBenchmark result;
    result.isa = kernel_isa_name(resolved_isa(config));
    result.threads = resolved_threads(config);
    result.out_dim = out_dim;
    result.in_dim = in_dim;
    result.iterations = std::max<uint32_t>(1, iterations);

    double best = 1e300;
    for (uint32_t i = 0; i < result.iterations; ++i) {
        const auto start = std::chrono::steady_clock::now();
        gemv(config, weights.data(), x.data(), y.data(), out_dim, in_dim);
        const auto stop = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(stop - start).count();
        best = std::min(best, ms);
    }
    // Keep the compiler from eliminating the buffers.
    volatile float sink = 0.0f;
    for (float v : y) sink = sink + v;
    (void)sink;

    result.best_ms = best;
    const double flops = 2.0 * static_cast<double>(out_dim) * in_dim;
    const double bytes = static_cast<double>(out_dim) * in_dim * sizeof(float) +
                         static_cast<double>(in_dim) * sizeof(float) +
                         static_cast<double>(out_dim) * sizeof(float);
    result.gflops = best > 0.0 ? flops / (best * 1e6) : 0.0;
    result.gbytes_per_s = best > 0.0 ? bytes / (best * 1e6) : 0.0;
    return result;
}

double gemv_max_abs_error(KernelIsa isa, uint32_t out_dim, uint32_t in_dim, uint32_t seed) {
    std::vector<float> weights(static_cast<size_t>(out_dim) * in_dim);
    std::vector<float> x(in_dim);
    std::vector<float> ref(out_dim, 0.0f);
    std::vector<float> got(out_dim, 0.0f);
    uint32_t state = seed ? seed : 1u;
    for (auto &w : weights) {
        state = state * 1664525u + 1013904223u;
        w = static_cast<float>((state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
    }
    for (auto &v : x) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>((state >> 8) & 0xFFFF) / 32768.0f - 1.0f;
    }
    gemv_scalar(weights.data(), x.data(), ref.data(), out_dim, in_dim);
    gemv_isa(isa, weights.data(), x.data(), got.data(), out_dim, in_dim);
    double worst = 0.0;
    for (uint32_t i = 0; i < out_dim; ++i)
        worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - ref[i]));
    return worst;
}

} // namespace fm
