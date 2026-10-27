#include "fm/backend.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <set>
#include <thread>
#include <unistd.h>

#include "vnpu.h"

namespace fm {

const char *backend_kind_name(BackendKind kind) {
    switch (kind) {
    case BackendKind::ScalarCPU: return "cpu";
    case BackendKind::VirtualNPU: return "vnpu";
    case BackendKind::PhysicalGPU: return "gpu";
    }
    return "unknown";
}

BackendKind parse_backend_kind(const std::string &name) {
    if (name == "cpu" || name == "scalar") return BackendKind::ScalarCPU;
    if (name == "vnpu" || name == "virtual-npu") return BackendKind::VirtualNPU;
    if (name == "gpu" || name == "physical-gpu") return BackendKind::PhysicalGPU;
    fail(ErrorCode::INVALID_ARGUMENT, "unknown backend: " + name);
}

const char *allocation_tag_name(AllocationTag tag) {
    switch (tag) {
    case AllocationTag::Weights: return "weights";
    case AllocationTag::ExpertCache: return "expert_cache";
    case AllocationTag::KV: return "kv";
    case AllocationTag::Workspace: return "workspace";
    case AllocationTag::Staging: return "staging";
    case AllocationTag::Other: return "other";
    }
    return "other";
}

Json DeviceInfo::to_json() const {
    Json j = Json::object();
    j.set("name", name);
    j.set("backend", backend);
    j.set("memory_type", memory_type);
    j.set("api_version", api_version);
    j.set("driver_version", driver_version);
    j.set("notes", notes);
    j.set("total_memory_bytes", total_memory_bytes);
    j.set("free_memory_bytes", free_memory_bytes);
    j.set("dedicated_vram_bytes", dedicated_vram_bytes);
    j.set("simd_bits", static_cast<int64_t>(simd_bits));
    j.set("threads", static_cast<int64_t>(threads));
    return j;
}

Json AllocationSnapshot::to_json() const {
    Json j = Json::object();
    j.set("live_bytes", live_bytes);
    j.set("peak_bytes", peak_bytes);
    j.set("live_allocations", live_allocations);
    j.set("total_allocations", total_allocations);
    j.set("total_frees", total_frees);
    Json by = Json::object();
    for (const auto &entry : by_tag) by.set(allocation_tag_name(entry.first), entry.second);
    j.set("by_tag", std::move(by));
    return j;
}

uint64_t AllocationTracker::allocate(AllocationTag tag, uint64_t bytes) {
    if (bytes == 0) fail(ErrorCode::INVALID_ARGUMENT, "zero-byte allocation");
    std::lock_guard<std::mutex> lock(mutex_);
    AllocationRecord rec;
    rec.id = next_id_++;
    rec.tag = tag;
    rec.bytes = bytes;
    live_.emplace(rec.id, rec);
    live_bytes_ += bytes;
    peak_bytes_ = std::max(peak_bytes_, live_bytes_);
    by_tag_[tag] += bytes;
    ++total_allocations_;
    return rec.id;
}

void AllocationTracker::free(uint64_t id) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = live_.find(id);
    if (it == live_.end()) return;
    uint64_t bytes = it->second.bytes;
    AllocationTag tag = it->second.tag;
    live_.erase(it);
    live_bytes_ -= bytes;
    auto tag_it = by_tag_.find(tag);
    if (tag_it != by_tag_.end()) {
        if (tag_it->second >= bytes) tag_it->second -= bytes;
        else tag_it->second = 0;
    }
    ++total_frees_;
}

AllocationSnapshot AllocationTracker::snapshot() const {
    std::lock_guard<std::mutex> lock(mutex_);
    AllocationSnapshot s;
    s.live_bytes = live_bytes_;
    s.peak_bytes = peak_bytes_;
    s.live_allocations = live_.size();
    s.total_allocations = total_allocations_;
    s.total_frees = total_frees_;
    for (auto tag : {AllocationTag::Weights, AllocationTag::ExpertCache,
                     AllocationTag::KV, AllocationTag::Workspace,
                     AllocationTag::Staging, AllocationTag::Other}) {
        auto it = by_tag_.find(tag);
        s.by_tag[tag] = it == by_tag_.end() ? 0 : it->second;
    }
    return s;
}

Backend::Backend(BackendConfig config, TraceWriter *trace)
    : config_(std::move(config)), trace_(trace) {}

bool Backend::fence_complete(const Fence &fence) const {
    return !fence.deferred || now_ms() >= fence.ready_at_ms;
}

void Backend::wait_fence(const Fence &fence, uint32_t timeout_ms) {
    Timer timer;
    while (!fence_complete(fence)) {
        if (timer.elapsed_ns() / 1000000ull >= timeout_ms)
            fail(ErrorCode::FENCE_TIMEOUT, "fence " + std::to_string(fence.id) + " timed out");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
}

void Backend::record_control(const std::string &name, uint64_t input_bytes,
                             uint64_t output_bytes, const std::string &backend) {
    record_operation({name, backend, input_bytes, output_bytes});
}

void Backend::record_operation(const OpRecord &record) {
    std::lock_guard<std::mutex> lock(op_mutex_);
    operations_.push_back(record);
}

std::vector<OpRecord> Backend::operations() const {
    std::lock_guard<std::mutex> lock(op_mutex_);
    return operations_;
}

Json Backend::metrics_json() const {
    auto ops = operations();
    auto alloc = tracker_->snapshot();
    uint64_t input_bytes = 0, output_bytes = 0;
    Json op_array = Json::array();
    for (const auto &op : ops) {
        input_bytes += op.input_bytes;
        output_bytes += op.output_bytes;
        Json j = Json::object();
        j.set("name", op.name);
        j.set("backend", op.backend);
        j.set("input_bytes", op.input_bytes);
        j.set("output_bytes", op.output_bytes);
        op_array.push(std::move(j));
    }
    Json j = Json::object();
    j.set("name", name());
    j.set("operation_count", static_cast<uint64_t>(ops.size()));
    j.set("operation_input_bytes", input_bytes);
    j.set("operation_output_bytes", output_bytes);
    j.set("allocations", alloc.to_json());
    j.set("operations", std::move(op_array));
    return j;
}

std::string host_cpu_name() {
#ifdef __linux__
    std::ifstream in("/proc/cpuinfo");
    std::string line;
    while (std::getline(in, line)) {
        if (line.rfind("model name", 0) == 0) {
            auto pos = line.find(':');
            if (pos != std::string::npos) return trim(line.substr(pos + 1));
        }
    }
#endif
    return "unknown CPU";
}

bool host_is_zen3() {
#if (defined(__x86_64__) || defined(__i386__)) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
    return __builtin_cpu_is("znver3");
#else
    return false;
#endif
}

uint64_t host_total_memory_bytes() {
    long pages = ::sysconf(_SC_PHYS_PAGES);
    long page_size = ::sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0)
        return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
    return 0;
}

uint64_t host_available_memory_bytes() {
    long pages = ::sysconf(_SC_AVPHYS_PAGES);
    long page_size = ::sysconf(_SC_PAGESIZE);
    if (pages > 0 && page_size > 0)
        return static_cast<uint64_t>(pages) * static_cast<uint64_t>(page_size);
    return 0;
}

namespace {

DeviceInfo cpu_device_info() {
    DeviceInfo info;
    info.name = "FreeMoe scalar CPU reference";
    info.backend = host_cpu_name();
    info.memory_type = "host_ram";
    info.api_version = "C++17 scalar";
    info.driver_version = "none";
    info.total_memory_bytes = host_total_memory_bytes();
    info.free_memory_bytes = host_available_memory_bytes();
    info.simd_bits = 0;
    info.threads = 1;
    info.notes = "Correctness oracle; no vendor acceleration.";
    return info;
}

class CpuBackend final : public Backend {
public:
    CpuBackend(BackendConfig config, TraceWriter *trace) : Backend(std::move(config), trace) {}
    std::string name() const override { return "cpu"; }
    DeviceInfo info() const override { return cpu_device_info(); }

    std::unique_ptr<DeviceBuffer> allocate(uint64_t bytes, AllocationTag tag) override {
        void *ptr = nullptr;
        if (::posix_memalign(&ptr, 64, static_cast<size_t>(bytes)) != 0)
            fail(ErrorCode::ALLOCATION_FAILED, "CPU buffer allocation failed");
        std::memset(ptr, 0, static_cast<size_t>(bytes));
        auto buffer = std::make_unique<DeviceBuffer>();
        buffer->data = ptr;
        buffer->bytes = bytes;
        buffer->tag = tag;
        auto tracker = tracker_;
        buffer->allocation_id = tracker->allocate(tag, bytes);
        buffer->cleanup = [tracker, id = buffer->allocation_id, ptr]() {
            tracker->free(id);
            std::free(ptr);
        };
        return buffer;
    }

    Fence upload(DeviceBuffer &buffer, const void *src, uint64_t bytes) override {
        if (fault_ == BackendFault::TransferFailure)
            fail(ErrorCode::TRANSFER_FAILED, "injected CPU transfer failure");
        if (fault_ == BackendFault::DeviceLoss)
            fail(ErrorCode::DEVICE_LOSS, "injected device loss");
        if (!buffer.data || bytes > buffer.bytes)
            fail(ErrorCode::INVALID_ARGUMENT, "bad CPU upload");
        std::memcpy(buffer.data, src, static_cast<size_t>(bytes));
        Fence f;
        f.id = next_fence_++;
        f.ready_at_ms = now_ms();
        return f;
    }

    void wait_fence(const Fence &fence, uint32_t timeout_ms) override {
        if (fault_ == BackendFault::FenceTimeout) {
            fault_ = BackendFault::None;
            fail(ErrorCode::FENCE_TIMEOUT, "injected CPU fence timeout");
        }
        Backend::wait_fence(fence, timeout_ms);
    }

    void matvec(const float *weights, const float *x, float *y,
                uint32_t out_dim, uint32_t in_dim, const std::string &op_name) override {
        if (fault_ == BackendFault::DeviceLoss)
            fail(ErrorCode::DEVICE_LOSS, "injected device loss");
        if (!weights || !x || !y || out_dim == 0 || in_dim == 0)
            fail(ErrorCode::INVALID_ARGUMENT, "invalid matvec arguments");
        for (uint32_t o = 0; o < out_dim; ++o) {
            const float *row = weights + static_cast<size_t>(o) * in_dim;
            float acc = 0.0f;
            for (uint32_t k = 0; k < in_dim; ++k) acc += row[k] * x[k];
            y[o] = acc;
        }
        record_operation({op_name, "scalar-cpu",
                          (static_cast<uint64_t>(out_dim) * in_dim + in_dim) * sizeof(float),
                          static_cast<uint64_t>(out_dim) * sizeof(float)});
    }

    void inject_fault(BackendFault fault) override { fault_ = fault; }
    void clear_faults() override { fault_ = BackendFault::None; }
private:
    BackendFault fault_ = BackendFault::None;
    uint64_t next_fence_ = 1;
};

struct VnpuRuntime {
    vnpu_device_t *device = nullptr;
    vnpu_stream_t *stream = nullptr;
    vnpu_device_info_t info{};
    std::string notes;
    ~VnpuRuntime() {
        if (stream) vnpu_stream_destroy(stream);
        if (device) vnpu_device_destroy(device);
    }
};

std::vector<VnpuIsaAttempt> plan_vnpu_isa_attempts_impl(const std::string &profile, bool zen3_detected) {
    using A = VnpuIsaAttempt;
    // Zen 3's two 256-bit FMA pipes are the target path. Detection overrides
    // both `auto` and an explicit AVX-512 request, avoiding 512-bit dispatch
    // entirely rather than waiting for an unsupported-device error.
    const std::vector<A> zen3_fallback = {
        {VNPU_ISA_FORCE_AVX2_DUAL, "Zen 3 AVX2 dual-pipe fallback"},
        {VNPU_ISA_FORCE_AVX2, "AVX2"},
        {VNPU_ISA_FORCE_SCALAR, "scalar"}};
    const std::vector<A> generic_auto = {
        {VNPU_ISA_AUTO, "runtime auto dispatch"},
        {VNPU_ISA_FORCE_AVX2_DUAL, "AVX2 dual-pipe fallback"},
        {VNPU_ISA_FORCE_AVX2, "AVX2"},
        {VNPU_ISA_FORCE_SCALAR, "scalar"}};

    if (profile == "zen3-auto" || profile == "zen3_auto")
        return zen3_detected ? zen3_fallback : generic_auto;
    if (profile == "auto")
        return zen3_detected ? zen3_fallback : generic_auto;
    if (profile == "avx512") {
        if (zen3_detected) return zen3_fallback;
        return {{VNPU_ISA_FORCE_AVX512, "AVX-512"},
                {VNPU_ISA_FORCE_AVX2_DUAL, "AVX2 dual-pipe fallback"},
                {VNPU_ISA_FORCE_AVX2, "AVX2"},
                {VNPU_ISA_FORCE_SCALAR, "scalar"}};
    }
    if (profile == "avx2_dual" || profile == "zen3")
        return {{VNPU_ISA_FORCE_AVX2_DUAL, "AVX2 dual-pipe"},
                {VNPU_ISA_FORCE_AVX2, "AVX2"},
                {VNPU_ISA_FORCE_SCALAR, "scalar"}};
    if (profile == "avx2")
        return {{VNPU_ISA_FORCE_AVX2, "AVX2"},
                {VNPU_ISA_FORCE_SCALAR, "scalar"}};
    if (profile == "scalar")
        return {{VNPU_ISA_FORCE_SCALAR, "scalar"}};
    fail(ErrorCode::INVALID_ARGUMENT, "unknown VNPU ISA profile: " + profile);
}

std::shared_ptr<VnpuRuntime> create_vnpu_runtime(const BackendConfig &config, TraceWriter *trace) {
    auto runtime = std::make_shared<VnpuRuntime>();
    bool real_zen3 = host_is_zen3();
    bool simulated_zen3 = !real_zen3 && config.test_simulate_zen3;
    bool zen3 = real_zen3 || config.test_simulate_zen3;
    std::vector<VnpuIsaAttempt> attempts = plan_vnpu_isa_attempts(config.vnpu_isa, zen3);
    vnpu_status_t last = VNPU_OK;
    size_t selected = 0;
    for (size_t i = 0; i < attempts.size(); ++i) {
        vnpu_config_t cfg;
        vnpu_config_init(&cfg);
        cfg.forced_isa = attempts[i].forced_isa;
        cfg.num_threads = config.vnpu_threads;
        cfg.node_name = "freemoe";
        last = vnpu_device_create(&cfg, &runtime->device);
        if (last == VNPU_OK) {
            selected = i;
            if (i > 0) runtime->notes += "Explicit ISA fallback to " + attempts[i].label + "; ";
            break;
        }
        if (last != VNPU_ERR_UNSUPPORTED)
            fail(ErrorCode::DEVICE_LOSS, std::string("VNPU device creation failed: ") + vnpu_status_string(last));
    }
    if (!runtime->device)
        fail(ErrorCode::DEVICE_LOSS, std::string("no supported VNPU ISA: ") + vnpu_status_string(last));
    if (vnpu_stream_create(runtime->device, 0, &runtime->stream) != VNPU_OK)
        fail(ErrorCode::DEVICE_LOSS, "VNPU stream creation failed");
    if (vnpu_device_get_info(runtime->device, &runtime->info) != VNPU_OK)
        fail(ErrorCode::DEVICE_LOSS, "VNPU device query failed");
    runtime->notes += "Virtual NPU is a CPU software device; no dedicated VRAM. ";
    runtime->notes += "Selected VNPU dispatch: " + attempts[selected].label + ". ";
    if (zen3) {
        runtime->notes += simulated_zen3 ? "Zen 3 detected (simulated for test); "
                                         : "Zen 3 detected; ";
        runtime->notes += "AVX-512 dispatch disabled in favor of dual 256-bit AVX2/FMA3. ";
    } else {
        runtime->notes += "Zen 3 not detected; using requested VNPU dispatch. ";
    }
    if (trace) {
        Json fields = Json::object();
        fields.set("backend", runtime->info.backend);
        fields.set("simd_bits", static_cast<int64_t>(runtime->info.simd_bits));
        fields.set("requested_profile", config.vnpu_isa);
        fields.set("selected_attempt", attempts[selected].label);
        fields.set("fallback_used", selected > 0);
        fields.set("zen3_detected", zen3);
        fields.set("zen3_detection_simulated", simulated_zen3);
        fields.set("zen3_fallback_selected",
                   zen3 && attempts[selected].forced_isa == VNPU_ISA_FORCE_AVX2_DUAL);
        fields.set("notes", runtime->notes);
        trace->event("backend_init", std::move(fields));
    }
    return runtime;
}

class VnpuBackend final : public Backend {
public:
    VnpuBackend(BackendConfig config, TraceWriter *trace)
        : Backend(config, trace),
          runtime_(create_vnpu_runtime(config_, trace)) {}

    std::string name() const override { return "vnpu"; }
    DeviceInfo info() const override {
        DeviceInfo info;
        info.name = "Vendored Virtual NPU 1.0 (patched)";
        info.backend = runtime_->info.backend;
        info.memory_type = "host_ram (virtual device buffers)";
        info.api_version = "VNPU ABI " + std::to_string(runtime_->info.abi_version);
        info.driver_version = "vendored C++ runtime";
        info.total_memory_bytes = runtime_->info.total_memory_bytes;
        info.free_memory_bytes = host_available_memory_bytes();
        info.dedicated_vram_bytes = 0;
        info.simd_bits = runtime_->info.simd_bits;
        info.threads = runtime_->info.num_threads;
        info.notes = runtime_->notes;
        if (runtime_->info.features & VNPU_FEATURE_AVX2_DUAL_FMA)
            info.notes += "Dual independent 256-bit AVX2/FMA3 accumulators enabled. ";
        return info;
    }

    std::unique_ptr<DeviceBuffer> allocate(uint64_t bytes, AllocationTag tag) override {
        vnpu_buffer_t *handle = nullptr;
        vnpu_status_t rc = vnpu_buffer_alloc(runtime_->device, bytes, &handle);
        if (rc != VNPU_OK || !handle)
            fail(ErrorCode::ALLOCATION_FAILED, std::string("VNPU buffer allocation failed: ") + vnpu_status_string(rc));
        auto buffer = std::make_unique<DeviceBuffer>();
        buffer->data = vnpu_buffer_host_ptr(handle);
        buffer->bytes = vnpu_buffer_size(handle);
        buffer->tag = tag;
        auto tracker = tracker_;
        buffer->allocation_id = tracker->allocate(tag, bytes);
        auto runtime = runtime_;
        buffer->cleanup = [tracker, runtime, id = buffer->allocation_id, handle]() {
            (void)runtime; // keep device alive until after buffer destruction
            vnpu_buffer_free(handle);
            tracker->free(id);
        };
        return buffer;
    }

    Fence upload(DeviceBuffer &buffer, const void *src, uint64_t bytes) override {
        if (fault_ == BackendFault::TransferFailure)
            fail(ErrorCode::TRANSFER_FAILED, "injected VNPU transfer failure");
        if (fault_ == BackendFault::DeviceLoss)
            fail(ErrorCode::DEVICE_LOSS, "injected VNPU device loss");
        if (!buffer.data || !src || bytes > buffer.bytes)
            fail(ErrorCode::INVALID_ARGUMENT, "bad VNPU upload");
        vnpu_status_t rc = vnpu_buffer_copy(runtime_->stream, buffer.data, src, bytes,
                                             VNPU_COPY_HOST_TO_DEVICE);
        if (rc != VNPU_OK)
            fail(ErrorCode::TRANSFER_FAILED, std::string("VNPU H2D copy failed: ") + vnpu_status_string(rc));
        Fence f;
        f.id = next_fence_++;
        f.ready_at_ms = now_ms();
        if (fault_ == BackendFault::FenceDeferred) {
            f.deferred = true;
            f.ready_at_ms = now_ms() + 3600000ull;
            deferred_fences_.insert(f.id);
            fault_ = BackendFault::None;
        }
        return f;
    }

    bool fence_complete(const Fence &fence) const override {
        if (!fence.deferred) return true;
        if (ready_fences_.count(fence.id)) return true;
        return now_ms() >= fence.ready_at_ms;
    }

    void wait_fence(const Fence &fence, uint32_t timeout_ms) override {
        if (fault_ == BackendFault::FenceTimeout) {
            fault_ = BackendFault::None;
            fail(ErrorCode::FENCE_TIMEOUT, "injected VNPU fence timeout");
        }
        Backend::wait_fence(fence, timeout_ms);
    }

    void advance_fences() override {
        for (uint64_t id : deferred_fences_) ready_fences_.insert(id);
    }

    void matvec(const float *weights, const float *x, float *y,
                uint32_t out_dim, uint32_t in_dim, const std::string &op_name) override {
        if (fault_ == BackendFault::DeviceLoss)
            fail(ErrorCode::DEVICE_LOSS, "injected VNPU device loss");
        vnpu_matmul_f32_args_t args{};
        args.A = x;
        args.B = weights;
        args.C = y;
        args.M = 1;
        args.K = in_dim;
        args.N = out_dim;
        args.alpha = 1.0f;
        args.beta = 0.0f;
        args.trans_b = 1; // weights are [out_dim,in_dim]
        args.unit = VNPU_UNIT_CELL;
        vnpu_status_t rc = vnpu_matmul_f32(runtime_->stream, &args);
        if (rc != VNPU_OK)
            fail(ErrorCode::DEVICE_LOSS, std::string("VNPU matvec failed: ") + vnpu_status_string(rc));
        record_operation({op_name, "vnpu",
                          (static_cast<uint64_t>(out_dim) * in_dim + in_dim) * sizeof(float),
                          static_cast<uint64_t>(out_dim) * sizeof(float)});
    }

    void inject_fault(BackendFault fault) override { fault_ = fault; }
    void clear_faults() override {
        fault_ = BackendFault::None;
        ready_fences_.clear();
        deferred_fences_.clear();
    }
private:
    std::shared_ptr<VnpuRuntime> runtime_;
    BackendFault fault_ = BackendFault::None;
    uint64_t next_fence_ = 1;
    std::set<uint64_t> deferred_fences_;
    std::set<uint64_t> ready_fences_;
};

class GpuBackend final : public Backend {
public:
    GpuBackend(BackendConfig config, TraceWriter *trace)
        : Backend(config, trace), fallback_(config, trace) {}
    std::string name() const override { return "gpu-placeholder"; }
    DeviceInfo info() const override {
        DeviceInfo info;
        info.name = "Physical GPU placeholder";
        info.backend = "not implemented";
        info.memory_type = "none";
        info.api_version = "none";
        info.driver_version = "none";
        info.notes = "No CUDA/HIP/Vulkan backend exists. GPU requests fail loudly unless explicit CPU fallback is enabled.";
        return info;
    }
    std::unique_ptr<DeviceBuffer> allocate(uint64_t bytes, AllocationTag tag) override {
        if (!config_.allow_cpu_fallback)
            fail(ErrorCode::GPU_BACKEND_UNIMPLEMENTED, "physical GPU backend is not implemented");
        return fallback_.allocate(bytes, tag);
    }
    Fence upload(DeviceBuffer &buffer, const void *src, uint64_t bytes) override {
        if (!config_.allow_cpu_fallback)
            fail(ErrorCode::GPU_BACKEND_UNIMPLEMENTED, "physical GPU backend is not implemented");
        return fallback_.upload(buffer, src, bytes);
    }
    void matvec(const float *weights, const float *x, float *y,
                uint32_t out_dim, uint32_t in_dim, const std::string &op_name) override {
        if (!config_.allow_cpu_fallback)
            fail(ErrorCode::GPU_BACKEND_UNIMPLEMENTED, "physical GPU backend is not implemented: " + op_name);
        std::fprintf(stderr, "[fallback] %s executed on scalar CPU (explicit --allow-cpu-fallback)\n", op_name.c_str());
        fallback_.matvec(weights, x, y, out_dim, in_dim, op_name);
        // Copy the fallback operation into this backend's log with an explicit marker.
        auto ops = fallback_.operations();
        if (!ops.empty()) {
            OpRecord record = ops.back();
            record.backend = "cpu-fallback";
            record_operation(record);
        }
    }
    void inject_fault(BackendFault fault) override { fallback_.inject_fault(fault); }
    void clear_faults() override { fallback_.clear_faults(); }
private:
    CpuBackend fallback_;
};

} // namespace

std::vector<VnpuIsaAttempt> plan_vnpu_isa_attempts(const std::string &profile, bool zen3_detected) {
    return plan_vnpu_isa_attempts_impl(profile, zen3_detected);
}

std::unique_ptr<Backend> Backend::create(const BackendConfig &config, TraceWriter *trace) {
    switch (config.kind) {
    case BackendKind::ScalarCPU: return std::make_unique<CpuBackend>(config, trace);
    case BackendKind::VirtualNPU: return std::make_unique<VnpuBackend>(config, trace);
    case BackendKind::PhysicalGPU: return std::make_unique<GpuBackend>(config, trace);
    }
    fail(ErrorCode::INVALID_ARGUMENT, "unknown backend kind");
}

} // namespace fm
