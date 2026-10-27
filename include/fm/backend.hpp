#pragma once

#include "fm/common.hpp"

#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace fm {

enum class BackendKind { ScalarCPU, VirtualNPU, PhysicalGPU };
const char *backend_kind_name(BackendKind kind);
BackendKind parse_backend_kind(const std::string &name);

enum class AllocationTag { Weights, ExpertCache, KV, Workspace, Staging, Other };
const char *allocation_tag_name(AllocationTag tag);

struct DeviceInfo {
    std::string name;
    std::string backend;
    std::string memory_type = "host_ram";
    std::string api_version = "none";
    std::string driver_version = "none";
    std::string notes;
    uint64_t total_memory_bytes = 0;
    uint64_t free_memory_bytes = 0;
    uint64_t dedicated_vram_bytes = 0;
    uint32_t simd_bits = 0;
    uint32_t threads = 0;

    Json to_json() const;
};

struct AllocationRecord {
    uint64_t id = 0;
    AllocationTag tag = AllocationTag::Other;
    uint64_t bytes = 0;
};

struct AllocationSnapshot {
    uint64_t live_bytes = 0;
    uint64_t peak_bytes = 0;
    uint64_t live_allocations = 0;
    uint64_t total_allocations = 0;
    uint64_t total_frees = 0;
    std::map<AllocationTag, uint64_t> by_tag;
    Json to_json() const;
};

class AllocationTracker {
public:
    uint64_t allocate(AllocationTag tag, uint64_t bytes);
    void free(uint64_t id);
    AllocationSnapshot snapshot() const;
private:
    mutable std::mutex mutex_;
    uint64_t next_id_ = 1;
    uint64_t live_bytes_ = 0;
    uint64_t peak_bytes_ = 0;
    uint64_t total_allocations_ = 0;
    uint64_t total_frees_ = 0;
    std::map<uint64_t, AllocationRecord> live_;
    std::map<AllocationTag, uint64_t> by_tag_;
};

struct DeviceBuffer {
    void *data = nullptr;
    uint64_t bytes = 0;
    AllocationTag tag = AllocationTag::Other;
    uint64_t allocation_id = 0;
    std::function<void()> cleanup;

    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
    DeviceBuffer(DeviceBuffer &&other) noexcept { move_from(std::move(other)); }
    DeviceBuffer &operator=(DeviceBuffer &&other) noexcept {
        if (this != &other) {
            reset();
            move_from(std::move(other));
        }
        return *this;
    }
    ~DeviceBuffer() { reset(); }
    void reset() {
        if (cleanup) cleanup();
        cleanup = nullptr;
        data = nullptr;
        bytes = 0;
        allocation_id = 0;
    }
    float *f32() { return static_cast<float *>(data); }
    const float *f32() const { return static_cast<const float *>(data); }
private:
    void move_from(DeviceBuffer &&other) {
        data = other.data;
        bytes = other.bytes;
        tag = other.tag;
        allocation_id = other.allocation_id;
        cleanup = std::move(other.cleanup);
        other.cleanup = nullptr;
        other.data = nullptr;
        other.bytes = 0;
        other.allocation_id = 0;
    }
};

struct Fence {
    uint64_t id = 0;
    uint64_t ready_at_ms = 0;
    bool deferred = false;
};

enum class BackendFault { None, TransferFailure, FenceDeferred, FenceTimeout, DeviceLoss };

struct OpRecord {
    std::string name;
    std::string backend;
    uint64_t input_bytes = 0;
    uint64_t output_bytes = 0;
};

struct BackendConfig {
    BackendKind kind = BackendKind::ScalarCPU;
    bool allow_cpu_fallback = false;
    uint32_t vnpu_threads = 0;
    std::string vnpu_isa = "avx2_dual";
    // Test-only: forces the detection branch on non-Zen-3 CI hosts.
    // Never set by the CLI or normal engine configuration.
    bool test_simulate_zen3 = false;
};

class Backend {
public:
    explicit Backend(BackendConfig config, TraceWriter *trace = nullptr);
    virtual ~Backend() = default;
    static std::unique_ptr<Backend> create(const BackendConfig &config, TraceWriter *trace = nullptr);

    virtual std::string name() const = 0;
    virtual DeviceInfo info() const = 0;
    virtual std::unique_ptr<DeviceBuffer> allocate(uint64_t bytes, AllocationTag tag) = 0;
    virtual Fence upload(DeviceBuffer &buffer, const void *src, uint64_t bytes) = 0;
    virtual bool fence_complete(const Fence &fence) const;
    virtual void wait_fence(const Fence &fence, uint32_t timeout_ms);
    virtual void advance_fences() {}
    virtual void matvec(const float *weights, const float *x, float *y,
                        uint32_t out_dim, uint32_t in_dim, const std::string &op_name) = 0;
    virtual void inject_fault(BackendFault fault) { (void)fault; }
    virtual void clear_faults() {}

    void record_control(const std::string &name, uint64_t input_bytes, uint64_t output_bytes,
                        const std::string &backend = "scalar-control");
    void record_operation(const OpRecord &record);
    std::vector<OpRecord> operations() const;
    std::shared_ptr<AllocationTracker> tracker() const { return tracker_; }
    Json metrics_json() const;
    BackendConfig config() const { return config_; }

protected:
    BackendConfig config_;
    TraceWriter *trace_ = nullptr;
    std::shared_ptr<AllocationTracker> tracker_ = std::make_shared<AllocationTracker>();
    mutable std::mutex op_mutex_;
    std::vector<OpRecord> operations_;
};

std::string host_cpu_name();
bool host_is_zen3();
uint64_t host_total_memory_bytes();
uint64_t host_available_memory_bytes();

struct VnpuIsaAttempt {
    uint32_t forced_isa = 0;
    std::string label;
};

// Pure decision helper: testable on non-Zen-3 hosts by passing true.
// On Zen 3 it never schedules AVX-512, including for explicit AVX-512 requests.
std::vector<VnpuIsaAttempt> plan_vnpu_isa_attempts(const std::string &profile, bool zen3_detected);

} // namespace fm
