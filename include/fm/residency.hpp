#pragma once

#include "fm/backend.hpp"
#include "fm/gguf.hpp"
#include "fm/io.hpp"

#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace fm {

enum class SlotState {
    ABSENT,
    IO_QUEUED,
    HOST_READY,
    UPLOAD_QUEUED,
    GPU_READY,
    IN_USE,
    EVICTABLE,
    ERROR,
    CANCELLED,
};

enum class SlotEvent {
    DemandAccepted,
    PrefetchAccepted,
    IOComplete,
    CancelPrefetchOnly,
    CancelWithDemand,
    IOError,
    ValidationError,
    UploadQueued,
    FenceComplete,
    GenerationMismatch,
    UploadFailed,
    Acquire,
    ReleaseLast,
    DeviceLoss,
    EvictSelected,
    FencesDone,
    CallbacksRetired,
    Shutdown,
};

const char *slot_state_name(SlotState state);
const char *slot_event_name(SlotEvent event);

struct StateTransition {
    bool valid = false;
    SlotState next = SlotState::ERROR;
    const char *action = "reject";
};

StateTransition transition_for(SlotState state, SlotEvent event);
bool apply_state_transition(SlotState &state, SlotEvent event, std::string &action);

enum class ReplacementPolicy { LRU, Clock };
const char *replacement_policy_name(ReplacementPolicy policy);
ReplacementPolicy parse_replacement_policy(const std::string &name);

struct BudgetConfig {
    uint64_t total_hot_bytes = 64ull * 1024 * 1024;
    uint64_t workspace_bytes = 1ull * 1024 * 1024;
    uint64_t staging_bytes = 1ull * 1024 * 1024;
    uint64_t margin_bytes = 4ull * 1024 * 1024;
    uint64_t pinned_bytes = 0;
    uint32_t requested_slots = 0;
};

struct BudgetResult {
    std::string memory_type = "host_ram";
    uint64_t total_hot_bytes = 0;
    uint64_t fixed_weights_bytes = 0;
    uint64_t kv_bytes = 0;
    uint64_t workspace_bytes = 0;
    uint64_t staging_bytes = 0;
    uint64_t margin_bytes = 0;
    uint64_t pinned_bytes = 0;
    uint64_t available_expert_bytes = 0;
    uint64_t slot_bytes = 0;
    uint32_t slot_count = 0;
    uint64_t expert_cache_bytes = 0;
    uint64_t unused_expert_bytes = 0;
    uint64_t dedicated_vram_bytes = 0;

    std::string print() const;
    Json to_json() const;
};

BudgetResult compute_budget(const ModelManifest &manifest, const BudgetConfig &config,
                            const DeviceInfo &device);

struct ResidencyMetrics {
    uint64_t cache_hits = 0;
    uint64_t cache_misses = 0;
    uint64_t demand_misses = 0;
    uint64_t prefetch_hits = 0;
    uint64_t evictions = 0;
    uint64_t invalid_transitions = 0;
    uint64_t stale_completions = 0;
    uint64_t checksum_checks = 0;
    uint64_t checksum_failures = 0;
    uint64_t checksum_unavailable = 0;
    uint64_t transfer_failures = 0;
    uint64_t fence_timeouts = 0;
    uint64_t device_loss_events = 0;
    uint64_t slots_busy_events = 0;
    uint64_t selected_experts = 0;
    uint64_t experts_executed = 0;
    uint64_t h2d_bytes = 0;
    uint64_t resident_expert_bytes = 0;
    uint64_t host_cache_bytes = 0;
    uint64_t prefetch_issued = 0;
    uint64_t prefetch_useful_bytes = 0;
    uint64_t prefetch_wasted_bytes = 0;
    uint64_t slot_wait_ms = 0;
    uint64_t error_events = 0;

    Json to_json() const;
};

struct ResidencyConfig {
    ReplacementPolicy replacement = ReplacementPolicy::LRU;
    bool resident_mode = false;
    bool enable_prefetch = true;
    uint32_t max_prefetch = 2;
    uint32_t host_cache_records = 4;
    uint32_t t_wait_ms = 100;
};

struct Slot {
    size_t index = 0;
    SlotState state = SlotState::ABSENT;
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint64_t key = 0;
    uint64_t generation = 0;
    uint64_t payload_bytes = 0;
    uint64_t checksum = 0;
    uint64_t last_used = 0;
    uint64_t demand_use_count = 0;
    uint64_t prefetch_use_count = 0;
    uint32_t refs = 0;
    uint32_t demand_waiters = 0;
    uint8_t clock_bit = 0;
    Fence fence{};
    std::unique_ptr<DeviceBuffer> buffer;
    std::string error_detail;
};

class ResidencyManager;

class SlotHandle {
public:
    SlotHandle() = default;
    SlotHandle(ResidencyManager *manager, Slot *slot)
        : manager_(manager), slot_(slot),
          data_(slot && slot->buffer ? static_cast<const float *>(slot->buffer->data) : nullptr) {}
    SlotHandle(const SlotHandle &) = delete;
    SlotHandle &operator=(const SlotHandle &) = delete;
    SlotHandle(SlotHandle &&other) noexcept { move_from(std::move(other)); }
    SlotHandle &operator=(SlotHandle &&other) noexcept {
        if (this != &other) { release(); move_from(std::move(other)); }
        return *this;
    }
    ~SlotHandle() { release(); }
    explicit operator bool() const { return slot_ != nullptr && data_ != nullptr; }
    const float *data() const { return data_; }
    Slot *slot() const { return slot_; }
    void release();
private:
    void move_from(SlotHandle &&other) {
        manager_ = other.manager_;
        slot_ = other.slot_;
        data_ = other.data_;
        other.manager_ = nullptr;
        other.slot_ = nullptr;
        other.data_ = nullptr;
    }
    ResidencyManager *manager_ = nullptr;
    Slot *slot_ = nullptr;
    const float *data_ = nullptr;
};

class ResidencyManager {
public:
    ResidencyManager(const ModelManifest &manifest, Backend &backend, ReadScheduler &scheduler,
                     TraceWriter &trace, BudgetResult budget, ResidencyConfig config);
    ~ResidencyManager();

    void initialize();
    void preload_resident_experts();
    SlotHandle acquire(uint32_t layer, uint32_t expert, uint64_t sequence, bool demand = true);
    void release(Slot *slot);

    template <typename Fn>
    void with_expert(uint32_t layer, uint32_t expert, uint64_t sequence, Fn &&fn) {
        SlotHandle handle = acquire(layer, expert, sequence, true);
        fn(handle.data());
        ++metrics_.experts_executed;
        {
            Json fields = Json::object();
            fields.set("layer", static_cast<int64_t>(layer));
            fields.set("expert", static_cast<int64_t>(expert));
            fields.set("sequence", sequence);
            fields.set("slot", static_cast<uint64_t>(handle.slot()->index));
            trace_.event("expert_executed", std::move(fields));
        }
    }

    void note_selected(uint32_t layer, const std::vector<uint32_t> &experts, uint64_t sequence);
    void assert_selected_executed();
    void issue_prefetch(uint32_t layer, uint32_t expert, uint64_t sequence);
    void poll_prefetches();
    void cancel_stale_prefetches(uint64_t current_sequence);
    void shutdown();

    // Fault injection and state-machine test hooks.
    bool complete_upload(size_t slot_index, uint64_t generation);
    bool inject_stale_completion(size_t slot_index, uint64_t wrong_generation);
    bool try_evict(size_t slot_index);
    void pin_slot_for_test(size_t slot_index);
    void unpin_slot_for_test(size_t slot_index);
    void set_clock_bits_for_test(const std::vector<uint8_t> &bits);
    Slot *select_victim_for_test(uint64_t sequence) { return select_victim(sequence); }
    void inject_device_loss();
    void reset_errors_for_test();

    const std::vector<Slot> &slots() const { return slots_; }
    std::vector<Slot> &slots() { return slots_; }
    const ResidencyMetrics &metrics() const { return metrics_; }
    const BudgetResult &budget() const { return budget_; }
    Json metrics_json() const;
    Json slots_json() const;
    void emit_budget_trace();

private:
    struct PendingPrefetch {
        uint64_t id = 0;
        uint64_t key = 0;
        uint64_t bytes = 0;
        uint64_t sequence = 0;
    };
    struct HostCacheEntry {
        uint64_t key = 0;
        uint64_t bytes = 0;
        uint64_t sequence = 0;
        std::vector<uint8_t> payload;
    };

    bool transition(Slot &slot, SlotEvent event, const std::string &where);
    Slot *find_resident(uint64_t key);
    Slot *select_victim(uint64_t sequence);
    std::vector<Extent> record_extents(const ExpertRecord &record) const;
    void evict_slot(Slot &slot, uint64_t sequence);
    void load_slot(Slot &slot, const ExpertRecord &record, uint64_t sequence);
    bool consume_host_cache(uint64_t key, std::vector<uint8_t> &payload);
    void insert_host_cache(uint64_t key, uint64_t sequence, std::vector<uint8_t> payload);
    void count_wasted_host_cache(uint64_t bytes, const std::string &reason);
    void mark_error(Slot &slot, ErrorCode code, const std::string &message);

    const ModelManifest &manifest_;
    Backend &backend_;
    ReadScheduler &scheduler_;
    TraceWriter &trace_;
    BudgetResult budget_;
    ResidencyConfig config_;
    std::vector<Slot> slots_;
    ResidencyMetrics metrics_;
    std::map<uint64_t, PendingPrefetch> pending_prefetch_;
    std::map<uint64_t, HostCacheEntry> host_cache_;
    std::map<uint64_t, std::vector<uint8_t>> resident_payloads_;
    size_t clock_hand_ = 0;
    uint64_t access_counter_ = 0;
    bool initialized_ = false;
    bool shutdown_ = false;
};

} // namespace fm
