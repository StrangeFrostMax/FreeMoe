#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace fm {

// Kernel-level ISA selection. This is deliberately separate from the VNPU ISA
// policy: the FreeMoe CPU backend owns its own dispatch, and Zen 3 must still
// prefer dual-issue 256-bit AVX2/FMA3 over AVX-512 even when AVX-512 exists.
enum class KernelIsa { Auto = 0, Scalar, Avx2Fma, Avx512 };

const char *kernel_isa_name(KernelIsa isa);
KernelIsa parse_kernel_isa(const std::string &name);

// Resolved ISA for this host given a request. `Auto` never selects AVX-512 on a
// host detected as Zen 3, matching the repository-wide Zen 3 policy.
KernelIsa resolve_kernel_isa(KernelIsa requested);

// CPU feature probes. Safe to call on any host.
bool host_has_avx2_fma();
bool host_has_avx512();

struct KernelConfig {
    KernelIsa requested = KernelIsa::Auto;
    // 0 means "use the hardware concurrency, capped at MAX_KERNEL_THREADS".
    uint32_t threads = 0;
};

constexpr uint32_t MAX_KERNEL_THREADS = 64;

KernelIsa resolved_isa(const KernelConfig &config);
uint32_t resolved_threads(const KernelConfig &config);

// Deterministic scalar reference: out = W * x with W[row, k] at row*in_dim + k,
// accumulated in ascending k order in float. This is the correctness oracle for
// every faster path.
void gemv_scalar(const float *weights, const float *x, float *y,
                 uint32_t out_dim, uint32_t in_dim);

// Vectorised GEMV. Falls back to the scalar loop when the requested ISA is not
// available on this host, so callers never need to check capabilities.
void gemv_isa(KernelIsa isa, const float *weights, const float *x, float *y,
              uint32_t out_dim, uint32_t in_dim);

// Vectorised and/or threaded GEMV using a persistent worker pool.
void gemv(const KernelConfig &config, const float *weights, const float *x, float *y,
          uint32_t out_dim, uint32_t in_dim);

// Dot product of two F32 rows using the best available ISA. Exposed so the
// quantised path can dequantise into a scratch row and reuse the vector units.
float dot_isa(KernelIsa isa, const float *a, const float *b, uint32_t n);

// Quantised GEMV: out = W*x where W is stored in `type_id` block format with
// `in_dim` values per row. Rows are decoded block-by-block into a small stack
// buffer, so a quantised tensor never has to be materialised as F32.
// Requires in_dim to be a whole number of quantisation blocks.
void gemv_quant(const KernelConfig &config, uint32_t type_id, const uint8_t *weights,
                const float *x, float *y, uint32_t out_dim, uint32_t in_dim);

// True when a GEMV kernel exists for this tensor type.
bool kernel_supports_type(uint32_t type_id);

// Small persistent thread pool. One process-wide instance is created lazily.
class WorkerPool {
public:
    static WorkerPool &instance();
    ~WorkerPool();

    // Runs `body(begin, end)` over [0, total) split into contiguous chunks.
    // Executes inline when the pool has no workers or `total` is 1.
    void parallel_for(uint32_t total, uint32_t max_threads,
                      const std::function<void(uint32_t, uint32_t)> &body);

    uint32_t size() const { return static_cast<uint32_t>(workers_.size()); }

private:
    WorkerPool();
    WorkerPool(const WorkerPool &) = delete;
    WorkerPool &operator=(const WorkerPool &) = delete;
    struct Impl;
    std::vector<Impl *> workers_;
};

struct GemvBenchmark {
    std::string isa;
    uint32_t threads = 1;
    uint32_t out_dim = 0;
    uint32_t in_dim = 0;
    uint32_t iterations = 0;
    double best_ms = 0.0;
    double gflops = 0.0;
    double gbytes_per_s = 0.0;
};

// Runs a deterministic benchmark for one configuration. Weights are read-only,
// so the reported bandwidth is a lower bound on the memory traffic of a GEMV.
GemvBenchmark benchmark_gemv(const KernelConfig &config, uint32_t out_dim,
                             uint32_t in_dim, uint32_t iterations);
GemvBenchmark benchmark_gemv_quant(const KernelConfig &config, uint32_t type_id,
                                   uint32_t out_dim, uint32_t in_dim, uint32_t iterations);

// Self-check used by unit tests: compares a candidate GEMV against gemv_scalar.
double gemv_max_abs_error(KernelIsa isa, uint32_t out_dim, uint32_t in_dim, uint32_t seed);

} // namespace fm
