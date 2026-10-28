#include "fm/residency.hpp"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <sstream>
#include <thread>

namespace fm {

const char *slot_state_name(SlotState state) {
    switch (state) {
    case SlotState::ABSENT: return "ABSENT";
    case SlotState::IO_QUEUED: return "IO_QUEUED";
    case SlotState::HOST_READY: return "HOST_READY";
    case SlotState::UPLOAD_QUEUED: return "UPLOAD_QUEUED";
    case SlotState::GPU_READY: return "GPU_READY";
    case SlotState::IN_USE: return "IN_USE";
    case SlotState::EVICTABLE: return "EVICTABLE";
    case SlotState::ERROR: return "ERROR";
    case SlotState::CANCELLED: return "CANCELLED";
    }
    return "UNKNOWN";
}

const char *slot_event_name(SlotEvent event) {
    switch (event) {
    case SlotEvent::DemandAccepted: return "DemandAccepted";
    case SlotEvent::PrefetchAccepted: return "PrefetchAccepted";
    case SlotEvent::IOComplete: return "IOComplete";
    case SlotEvent::CancelPrefetchOnly: return "CancelPrefetchOnly";
    case SlotEvent::CancelWithDemand: return "CancelWithDemand";
    case SlotEvent::IOError: return "IOError";
    case SlotEvent::ValidationError: return "ValidationError";
    case SlotEvent::UploadQueued: return "UploadQueued";
    case SlotEvent::FenceComplete: return "FenceComplete";
    case SlotEvent::GenerationMismatch: return "GenerationMismatch";
    case SlotEvent::UploadFailed: return "UploadFailed";
    case SlotEvent::Acquire: return "Acquire";
    case SlotEvent::ReleaseLast: return "ReleaseLast";
    case SlotEvent::DeviceLoss: return "DeviceLoss";
    case SlotEvent::EvictSelected: return "EvictSelected";
    case SlotEvent::FencesDone: return "FencesDone";
    case SlotEvent::CallbacksRetired: return "CallbacksRetired";
    case SlotEvent::Shutdown: return "Shutdown";
    }
    return "UNKNOWN";
}

StateTransition transition_for(SlotState state, SlotEvent event) {
    using S = SlotState;
    using E = SlotEvent;
    auto ok = [](S next, const char *action) {
        return StateTransition{true, next, action};
    };
    switch (state) {
    case S::ABSENT:
        if (event == E::DemandAccepted || event == E::PrefetchAccepted)
            return ok(S::IO_QUEUED, "reserve slot; record key/generation/subscriber");
        if (event == E::Shutdown) return ok(S::CANCELLED, "stop new work");
        break;
    case S::IO_QUEUED:
        if (event == E::IOComplete) return ok(S::HOST_READY, "publish validated host copy");
        if (event == E::CancelPrefetchOnly) return ok(S::CANCELLED, "cancel prefetch-only I/O");
        if (event == E::CancelWithDemand) return ok(S::IO_QUEUED, "keep load alive for demand");
        if (event == E::IOError || event == E::ValidationError)
            return ok(S::ERROR, "notify demand subscribers with typed error");
        if (event == E::Shutdown) return ok(S::CANCELLED, "stop new work");
        break;
    case S::HOST_READY:
        if (event == E::UploadQueued) return ok(S::UPLOAD_QUEUED, "record H2D fence and generation");
        if (event == E::IOError || event == E::ValidationError)
            return ok(S::ERROR, "notify demand subscribers with typed error");
        if (event == E::Shutdown) return ok(S::CANCELLED, "stop new work");
        break;
    case S::UPLOAD_QUEUED:
        if (event == E::FenceComplete) return ok(S::GPU_READY, "publish to slot after fence/generation check");
        if (event == E::UploadFailed || event == E::IOError || event == E::ValidationError)
            return ok(S::ERROR, "retire fence before slot reuse");
        if (event == E::DeviceLoss) return ok(S::ERROR, "invalidate slots on lost device");
        if (event == E::Shutdown) return ok(S::CANCELLED, "stop new work");
        break;
    case S::GPU_READY:
        if (event == E::Acquire) return ok(S::IN_USE, "increment reference count");
        if (event == E::EvictSelected) return ok(S::EVICTABLE, "prevent new references");
        if (event == E::DeviceLoss) return ok(S::ERROR, "invalidate slots on lost device");
        if (event == E::Shutdown) return ok(S::CANCELLED, "stop new work");
        break;
    case S::IN_USE:
        if (event == E::ReleaseLast) return ok(S::GPU_READY, "decrement last reference");
        if (event == E::DeviceLoss) return ok(S::ERROR, "invalidate slots on lost device");
        if (event == E::Shutdown) return ok(S::CANCELLED, "stop new work");
        break;
    case S::EVICTABLE:
        if (event == E::FencesDone) return ok(S::ABSENT, "release slot; clear key");
        if (event == E::Shutdown) return ok(S::CANCELLED, "stop new work");
        break;
    case S::ERROR:
        if (event == E::CallbacksRetired) return ok(S::ABSENT, "log metrics; release after callbacks retire");
        break;
    case S::CANCELLED:
        if (event == E::CallbacksRetired) return ok(S::ABSENT, "release after I/O and upload retire");
        break;
    }
    return StateTransition{};
}

bool apply_state_transition(SlotState &state, SlotEvent event, std::string &action) {
    StateTransition t = transition_for(state, event);
    if (!t.valid) {
        action = "reject";
        return false;
    }
    state = t.next;
    action = t.action;
    return true;
}

const char *replacement_policy_name(ReplacementPolicy policy) {
    return policy == ReplacementPolicy::Clock ? "clock" : "lru";
}

ReplacementPolicy parse_replacement_policy(const std::string &name) {
    if (name == "lru") return ReplacementPolicy::LRU;
    if (name == "clock") return ReplacementPolicy::Clock;
    fail(ErrorCode::INVALID_ARGUMENT, "unknown replacement policy: " + name);
}

std::string BudgetResult::print() const {
    std::ostringstream os;
    os << "memory_type=" << memory_type << "\n"
       << "total_hot_bytes=" << total_hot_bytes << "\n"
       << "reserve.fixed_weights=" << fixed_weights_bytes << "\n"
       << "reserve.kv=" << kv_bytes << "\n"
       << "reserve.workspace=" << workspace_bytes << "\n"
       << "reserve.staging=" << staging_bytes << "\n"
       << "reserve.margin=" << margin_bytes << "\n"
       << "reserve.pinned=" << pinned_bytes << "\n"
       << "available_for_expert_cache=" << available_expert_bytes << "\n"
       << "slot_bytes=" << slot_bytes << "\n"
       << "slot_count=" << slot_count << "\n"
       << "expert_cache_bytes=" << expert_cache_bytes << "\n"
       << "unused_expert_bytes=" << unused_expert_bytes << "\n"
       << "dedicated_vram_bytes=" << dedicated_vram_bytes << "\n";
    return os.str();
}

Json BudgetResult::to_json() const {
    Json j = Json::object();
    j.set("memory_type", memory_type);
    j.set("total_hot_bytes", total_hot_bytes);
    j.set("fixed_weights_bytes", fixed_weights_bytes);
    j.set("kv_bytes", kv_bytes);
    j.set("workspace_bytes", workspace_bytes);
    j.set("staging_bytes", staging_bytes);
    j.set("margin_bytes", margin_bytes);
    j.set("pinned_bytes", pinned_bytes);
    j.set("available_for_expert_cache", available_expert_bytes);
    j.set("slot_bytes", slot_bytes);
    j.set("slot_count", static_cast<int64_t>(slot_count));
    j.set("expert_cache_bytes", expert_cache_bytes);
    j.set("unused_expert_bytes", unused_expert_bytes);
    j.set("dedicated_vram_bytes", dedicated_vram_bytes);
    return j;
}

BudgetResult compute_budget(const ModelManifest &manifest, const BudgetConfig &config,
                            const DeviceInfo &device) {
    BudgetResult b;
    b.memory_type = device.memory_type;
    b.total_hot_bytes = config.total_hot_bytes;
    b.fixed_weights_bytes = manifest.fixed_weight_bytes;
    const ModelConfig &cfg = manifest.config;
    // Under grouped-query attention the cache holds kv_heads * head_dim, not
    // the full hidden width, so budgeting by `hidden` over-reserves.
    const uint64_t kv_dim = static_cast<uint64_t>(cfg.kv_heads) * cfg.head_dim;
    b.kv_bytes = checked_mul(static_cast<uint64_t>(cfg.layers) * cfg.context,
                             kv_dim * 2ull * sizeof(float), "KV cache");
    b.workspace_bytes = config.workspace_bytes;
    b.staging_bytes = config.staging_bytes;
    b.margin_bytes = config.margin_bytes;
    b.pinned_bytes = config.pinned_bytes;
    b.slot_bytes = manifest.slot_bytes;
    b.dedicated_vram_bytes = device.dedicated_vram_bytes;

    uint64_t reserved = 0;
    for (uint64_t value : {b.fixed_weights_bytes, b.kv_bytes, b.workspace_bytes,
                           b.staging_bytes, b.margin_bytes, b.pinned_bytes}) {
        if (reserved > std::numeric_limits<uint64_t>::max() - value)
            fail(ErrorCode::BUDGET_TOO_SMALL, "budget reservation overflow");
        reserved += value;
    }
    if (reserved >= b.total_hot_bytes) {
        BudgetResult error = b;
        error.available_expert_bytes = 0;
        fail(ErrorCode::BUDGET_TOO_SMALL, "unsafe memory budget:\n" + error.print());
    }
    b.available_expert_bytes = b.total_hot_bytes - reserved;
    if (b.slot_bytes == 0) {
        b.unused_expert_bytes = b.available_expert_bytes;
        return b;
    }
    if (b.available_expert_bytes < b.slot_bytes)
        fail(ErrorCode::BUDGET_TOO_SMALL, "expert cache is smaller than one slot:\n" + b.print());
    uint32_t max_slots = static_cast<uint32_t>(std::min<uint64_t>(
        b.available_expert_bytes / b.slot_bytes, std::numeric_limits<uint32_t>::max()));
    if (config.requested_slots && config.requested_slots > max_slots)
        fail(ErrorCode::BUDGET_TOO_SMALL,
             "requested slot count " + std::to_string(config.requested_slots) +
             " exceeds budget of " + std::to_string(max_slots) + ":\n" + b.print());
    b.slot_count = config.requested_slots ? config.requested_slots : max_slots;
    b.expert_cache_bytes = static_cast<uint64_t>(b.slot_count) * b.slot_bytes;
    b.unused_expert_bytes = b.available_expert_bytes - b.expert_cache_bytes;
    return b;
}

Json ResidencyMetrics::to_json() const {
    Json j = Json::object();
    j.set("cache_hits", cache_hits);
    j.set("cache_misses", cache_misses);
    j.set("demand_misses", demand_misses);
    j.set("prefetch_hits", prefetch_hits);
    j.set("evictions", evictions);
    j.set("invalid_transitions", invalid_transitions);
    j.set("stale_completions", stale_completions);
    j.set("checksum_checks", checksum_checks);
    j.set("checksum_failures", checksum_failures);
    j.set("checksum_unavailable", checksum_unavailable);
    j.set("transfer_failures", transfer_failures);
    j.set("fence_timeouts", fence_timeouts);
    j.set("device_loss_events", device_loss_events);
    j.set("slots_busy_events", slots_busy_events);
    j.set("selected_experts", selected_experts);
    j.set("experts_executed", experts_executed);
    j.set("h2d_bytes", h2d_bytes);
    j.set("resident_expert_bytes", resident_expert_bytes);
    j.set("host_cache_bytes", host_cache_bytes);
    j.set("prefetch_issued", prefetch_issued);
    j.set("prefetch_useful_bytes", prefetch_useful_bytes);
    j.set("prefetch_wasted_bytes", prefetch_wasted_bytes);
    j.set("slot_wait_ms", slot_wait_ms);
    j.set("error_events", error_events);
    uint64_t accesses = cache_hits + cache_misses;
    double hit_rate = accesses ? static_cast<double>(cache_hits) / static_cast<double>(accesses) : 0.0;
    j.set("hit_rate", fixed_double(hit_rate, 6));
    return j;
}

void SlotHandle::release() {
    if (manager_ && slot_) manager_->release(slot_);
    manager_ = nullptr;
    slot_ = nullptr;
    data_ = nullptr;
}

ResidencyManager::ResidencyManager(const ModelManifest &manifest, Backend &backend,
                                   ReadScheduler &scheduler, TraceWriter &trace,
                                   BudgetResult budget, ResidencyConfig config)
    : manifest_(manifest), backend_(backend), scheduler_(scheduler), trace_(trace),
      budget_(std::move(budget)), config_(std::move(config)) {}

ResidencyManager::~ResidencyManager() { shutdown(); }

void ResidencyManager::emit_budget_trace() {
    Json fields = Json::object();
    fields.set("memory_type", budget_.memory_type);
    fields.set("slot_count", static_cast<int64_t>(budget_.slot_count));
    fields.set("slot_bytes", budget_.slot_bytes);
    fields.set("expert_cache_bytes", budget_.expert_cache_bytes);
    fields.set("replacement", replacement_policy_name(config_.replacement));
    trace_.event("budget", std::move(fields));
}

void ResidencyManager::initialize() {
    if (initialized_) return;
    slots_.resize(budget_.slot_count);
    for (size_t i = 0; i < slots_.size(); ++i) {
        Slot &slot = slots_[i];
        slot.index = i;
        slot.state = SlotState::ABSENT;
        slot.key = std::numeric_limits<uint64_t>::max();
        slot.buffer = backend_.allocate(budget_.slot_bytes, AllocationTag::ExpertCache);
        if (!slot.buffer || !slot.buffer->data)
            fail(ErrorCode::ALLOCATION_FAILED, "expert slot allocation failed");
    }
    initialized_ = true;
    emit_budget_trace();
}

std::vector<Extent> ResidencyManager::record_extents(const ExpertRecord &record) const {
    std::vector<Extent> extents;
    for (const auto &e : record.extents) extents.push_back({e.offset, e.bytes});
    return extents;
}

void ResidencyManager::preload_resident_experts() {
    if (!config_.resident_mode || !resident_payloads_.empty()) return;
    for (const auto &record : manifest_.experts) {
        uint64_t key = ModelManifest::expert_key(record.layer, record.expert);
        std::vector<uint8_t> payload = scheduler_.read_record(record_extents(record), ReadPriority::Demand);
        if (record.checksum_known) {
            ++metrics_.checksum_checks;
            if (fnv1a64(payload) != record.checksum) {
                ++metrics_.checksum_failures;
                fail(ErrorCode::EXPERT_CHECKSUM_MISMATCH,
                     "resident expert checksum mismatch at layer " + std::to_string(record.layer) +
                     " expert " + std::to_string(record.expert));
            }
        } else {
            ++metrics_.checksum_unavailable;
        }
        metrics_.resident_expert_bytes += payload.size();
        resident_payloads_[key] = std::move(payload);
    }
    Json fields = Json::object();
    fields.set("bytes", metrics_.resident_expert_bytes);
    fields.set("records", static_cast<uint64_t>(resident_payloads_.size()));
    trace_.event("resident_experts_loaded", std::move(fields));
}

bool ResidencyManager::transition(Slot &slot, SlotEvent event, const std::string &where) {
    std::string action;
    SlotState before = slot.state;
    bool ok = apply_state_transition(slot.state, event, action);
    if (!ok) {
        ++metrics_.invalid_transitions;
        Json fields = Json::object();
        fields.set("slot", static_cast<uint64_t>(slot.index));
        fields.set("state", slot_state_name(before));
        fields.set("event", slot_event_name(event));
        fields.set("where", where);
        trace_.event("invalid_transition", std::move(fields));
        return false;
    }
    (void)action;
    return true;
}

void ResidencyManager::mark_error(Slot &slot, ErrorCode code, const std::string &message) {
    ++metrics_.error_events;
    slot.error_detail = message;
    if (slot.state == SlotState::IO_QUEUED || slot.state == SlotState::HOST_READY)
        transition(slot, SlotEvent::ValidationError, "mark_error");
    else if (slot.state == SlotState::UPLOAD_QUEUED)
        transition(slot, SlotEvent::UploadFailed, "mark_error");
    else if (slot.state != SlotState::ERROR && slot.state != SlotState::CANCELLED)
        slot.state = SlotState::ERROR;
    Json fields = Json::object();
    fields.set("slot", static_cast<uint64_t>(slot.index));
    fields.set("code", error_name(code));
    fields.set("message", message);
    trace_.event("error", std::move(fields));
}

Slot *ResidencyManager::find_resident(uint64_t key) {
    for (auto &slot : slots_) {
        if (slot.key == key && (slot.state == SlotState::GPU_READY ||
                                slot.state == SlotState::IN_USE ||
                                slot.state == SlotState::UPLOAD_QUEUED))
            return &slot;
    }
    return nullptr;
}

Slot *ResidencyManager::select_victim(uint64_t sequence) {
    (void)sequence;
    auto eligible = [&](Slot &slot) {
        return slot.state == SlotState::GPU_READY && slot.refs == 0 &&
               slot.demand_waiters == 0 && backend_.fence_complete(slot.fence);
    };
    auto scan = [&]() -> Slot * {
        for (auto &slot : slots_) if (slot.state == SlotState::ABSENT) return &slot;
        if (config_.replacement == ReplacementPolicy::LRU) {
            Slot *best = nullptr;
            for (auto &slot : slots_) {
                if (!eligible(slot)) continue;
                if (!best || slot.last_used < best->last_used) best = &slot;
            }
            return best;
        }
        if (slots_.empty()) return nullptr;
        for (size_t i = 0; i < slots_.size() * 2; ++i) {
            size_t index = (clock_hand_ + i) % slots_.size();
            Slot &slot = slots_[index];
            if (!eligible(slot)) continue;
            if (slot.clock_bit == 0) {
                clock_hand_ = (index + 1) % slots_.size();
                return &slot;
            }
            slot.clock_bit = 0;
        }
        return nullptr;
    };
    Slot *victim = scan();
    if (victim) return victim;
    Timer timer;
    metrics_.slot_wait_ms += 0;
    while (timer.elapsed_ns() / 1000000ull < config_.t_wait_ms) {
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        victim = scan();
        if (victim) {
            metrics_.slot_wait_ms += timer.elapsed_ns() / 1000000ull;
            return victim;
        }
    }
    ++metrics_.slots_busy_events;
    return nullptr;
}

bool ResidencyManager::try_evict(size_t slot_index) {
    if (slot_index >= slots_.size()) return false;
    Slot &slot = slots_[slot_index];
    if (slot.state != SlotState::GPU_READY || slot.refs != 0 ||
        slot.demand_waiters != 0 || !backend_.fence_complete(slot.fence))
        return false;
    uint64_t old_key = slot.key;
    if (!transition(slot, SlotEvent::EvictSelected, "try_evict")) return false;
    if (!transition(slot, SlotEvent::FencesDone, "try_evict")) return false;
    slot.key = std::numeric_limits<uint64_t>::max();
    slot.layer = 0;
    slot.expert = 0;
    slot.checksum = 0;
    slot.clock_bit = 0;
    ++metrics_.evictions;
    Json fields = Json::object();
    fields.set("slot", static_cast<uint64_t>(slot.index));
    if (old_key == std::numeric_limits<uint64_t>::max()) fields.set("old_key", "none");
    else fields.set("old_key", old_key);
    fields.set("generation", slot.generation);
    trace_.event("evict", std::move(fields));
    return true;
}

void ResidencyManager::evict_slot(Slot &slot, uint64_t sequence) {
    (void)sequence;
    if (slot.state == SlotState::ABSENT) return;
    if (!try_evict(slot.index))
        fail(ErrorCode::SLOTS_BUSY, "cannot safely evict slot " + std::to_string(slot.index));
}

bool ResidencyManager::consume_host_cache(uint64_t key, std::vector<uint8_t> &payload) {
    auto it = host_cache_.find(key);
    if (it == host_cache_.end()) return false;
    uint64_t bytes = it->second.bytes;
    payload = std::move(it->second.payload);
    host_cache_.erase(it);
    metrics_.host_cache_bytes = (metrics_.host_cache_bytes >= bytes) ? metrics_.host_cache_bytes - bytes : 0;
    ++metrics_.prefetch_hits;
    metrics_.prefetch_useful_bytes += bytes;
    scheduler_.note_prefetch_useful(bytes);
    Json fields = Json::object();
    fields.set("key", key);
    fields.set("bytes", bytes);
    trace_.event("prefetch_useful", std::move(fields));
    return true;
}

void ResidencyManager::count_wasted_host_cache(uint64_t bytes, const std::string &reason) {
    metrics_.prefetch_wasted_bytes += bytes;
    scheduler_.note_prefetch_wasted(bytes);
    Json fields = Json::object();
    fields.set("bytes", bytes);
    fields.set("reason", reason);
    trace_.event("prefetch_wasted", std::move(fields));
}

void ResidencyManager::insert_host_cache(uint64_t key, uint64_t sequence,
                                         std::vector<uint8_t> payload) {
    while (!host_cache_.empty() && host_cache_.size() >= config_.host_cache_records) {
        auto oldest = host_cache_.begin();
        for (auto it = host_cache_.begin(); it != host_cache_.end(); ++it)
            if (it->second.sequence < oldest->second.sequence) oldest = it;
        uint64_t bytes = oldest->second.bytes;
        metrics_.host_cache_bytes = (metrics_.host_cache_bytes >= bytes) ? metrics_.host_cache_bytes - bytes : 0;
        host_cache_.erase(oldest);
        count_wasted_host_cache(bytes, "host_cache_eviction");
    }
    HostCacheEntry entry;
    entry.key = key;
    entry.bytes = payload.size();
    entry.sequence = sequence;
    entry.payload = std::move(payload);
    metrics_.host_cache_bytes += entry.bytes;
    host_cache_[key] = std::move(entry);
}

void ResidencyManager::issue_prefetch(uint32_t layer, uint32_t expert, uint64_t sequence) {
    if (!config_.enable_prefetch || config_.resident_mode || shutdown_) return;
    const ExpertRecord *record = manifest_.expert(layer, expert);
    if (!record) return;
    uint64_t key = ModelManifest::expert_key(layer, expert);
    if (find_resident(key) || host_cache_.count(key) || pending_prefetch_.count(key)) return;
    while (pending_prefetch_.size() >= config_.max_prefetch) {
        auto oldest = pending_prefetch_.begin();
        for (auto it = pending_prefetch_.begin(); it != pending_prefetch_.end(); ++it)
            if (it->second.sequence < oldest->second.sequence) oldest = it;
        scheduler_.cancel_prefetch(oldest->second.id);
        metrics_.prefetch_wasted_bytes += oldest->second.bytes;
        Json fields = Json::object();
        fields.set("key", oldest->first);
        fields.set("bytes", oldest->second.bytes);
        fields.set("reason", "prefetch_queue_pressure");
        trace_.event("prefetch_wasted", std::move(fields));
        pending_prefetch_.erase(oldest);
    }
    std::vector<Extent> extents = record_extents(*record);
    uint64_t id = scheduler_.submit_prefetch(extents, key, sequence);
    PendingPrefetch pending{id, key, record->payload_bytes, sequence};
    pending_prefetch_[key] = pending;
    ++metrics_.prefetch_issued;
}

void ResidencyManager::poll_prefetches() {
    if (shutdown_) return;
    for (auto it = pending_prefetch_.begin(); it != pending_prefetch_.end();) {
        std::vector<uint8_t> payload;
        PrefetchStatus status = scheduler_.poll_prefetch(it->second.id, payload);
        if (status == PrefetchStatus::Pending) {
            ++it;
            continue;
        }
        uint64_t key = it->first;
        uint64_t bytes = it->second.bytes;
        uint64_t sequence = it->second.sequence;
        if (status == PrefetchStatus::Ready && !payload.empty()) {
            insert_host_cache(key, sequence, std::move(payload));
            Json fields = Json::object();
            fields.set("key", key);
            fields.set("bytes", bytes);
            trace_.event("prefetch_host_ready", std::move(fields));
        } else {
            // Cancelled or reaped by demand priority: the scheduler already
            // accounted for the wasted I/O bytes.
            metrics_.prefetch_wasted_bytes += bytes;
            Json fields = Json::object();
            fields.set("key", key);
            fields.set("bytes", bytes);
            fields.set("reason", "cancelled_or_reaped");
            trace_.event("prefetch_wasted", std::move(fields));
        }
        it = pending_prefetch_.erase(it);
    }
}

void ResidencyManager::cancel_stale_prefetches(uint64_t current_sequence) {
    for (auto it = pending_prefetch_.begin(); it != pending_prefetch_.end();) {
        if (it->second.sequence + 1 < current_sequence) {
            scheduler_.cancel_prefetch(it->second.id);
            metrics_.prefetch_wasted_bytes += it->second.bytes;
            Json fields = Json::object();
            fields.set("key", it->first);
            fields.set("bytes", it->second.bytes);
            fields.set("reason", "stale_prediction");
            trace_.event("prefetch_wasted", std::move(fields));
            it = pending_prefetch_.erase(it);
        } else {
            ++it;
        }
    }
}

bool ResidencyManager::complete_upload(size_t slot_index, uint64_t generation) {
    if (slot_index >= slots_.size()) return false;
    Slot &slot = slots_[slot_index];
    if (slot.generation != generation) {
        ++metrics_.stale_completions;
        Json fields = Json::object();
        fields.set("slot", static_cast<uint64_t>(slot_index));
        fields.set("expected", slot.generation);
        fields.set("received", generation);
        trace_.event("stale_completion", std::move(fields));
        return false;
    }
    if (slot.state != SlotState::UPLOAD_QUEUED || !backend_.fence_complete(slot.fence))
        return false;
    if (!transition(slot, SlotEvent::FenceComplete, "complete_upload")) return false;
    Json fields = Json::object();
    fields.set("slot", static_cast<uint64_t>(slot_index));
    fields.set("generation", generation);
    trace_.event("upload_complete", std::move(fields));
    return true;
}

bool ResidencyManager::inject_stale_completion(size_t slot_index, uint64_t wrong_generation) {
    if (slot_index >= slots_.size()) return false;
    Slot &slot = slots_[slot_index];
    if (slot.generation == wrong_generation) return false;
    ++metrics_.stale_completions;
    Json fields = Json::object();
    fields.set("slot", static_cast<uint64_t>(slot_index));
    fields.set("expected", slot.generation);
    fields.set("received", wrong_generation);
    trace_.event("stale_completion", std::move(fields));
    return true;
}

void ResidencyManager::load_slot(Slot &slot, const ExpertRecord &record, uint64_t sequence) {
    uint64_t key = ModelManifest::expert_key(record.layer, record.expert);
    slot.key = key;
    slot.layer = record.layer;
    slot.expert = record.expert;
    slot.payload_bytes = record.payload_bytes;
    slot.checksum = record.checksum;
    slot.error_detail.clear();
    ++slot.generation;
    slot.fence = Fence{};
    slot.clock_bit = 0;
    if (!transition(slot, SlotEvent::DemandAccepted, "load_slot"))
        fail(ErrorCode::INTERNAL_ERROR, "cannot start expert load in state " + std::string(slot_state_name(slot.state)));
    Json io_fields = Json::object();
    io_fields.set("slot", static_cast<uint64_t>(slot.index));
    io_fields.set("key", key);
    io_fields.set("generation", slot.generation);
    io_fields.set("priority", "demand");
    trace_.event("io_queued", std::move(io_fields));

    std::vector<uint8_t> payload;
    auto resident = resident_payloads_.find(key);
    if (resident != resident_payloads_.end()) {
        payload = resident->second;
    } else if (!consume_host_cache(key, payload)) {
        try {
            payload = scheduler_.read_record(record_extents(record), ReadPriority::Demand);
        } catch (const Error &e) {
            mark_error(slot, e.code, e.what());
            throw;
        }
    }
    if (payload.size() != record.payload_bytes) {
        mark_error(slot, ErrorCode::IO_ERROR, "expert payload has wrong length");
        fail(ErrorCode::IO_ERROR, "expert payload has wrong length");
    }
    if (!record.checksum_known) {
        // Vendor checkpoints carry no checksum manifest. Integrity against the
        // source file is still checked by the offset/extent bounds, but the
        // value check is honestly reported as unavailable rather than skipped.
        ++metrics_.checksum_unavailable;
    } else {
        ++metrics_.checksum_checks;
        if (fnv1a64(payload) != record.checksum) {
            ++metrics_.checksum_failures;
            mark_error(slot, ErrorCode::EXPERT_CHECKSUM_MISMATCH, "source expert checksum mismatch");
            fail(ErrorCode::EXPERT_CHECKSUM_MISMATCH,
                 "source expert checksum mismatch at layer " + std::to_string(record.layer) +
                 " expert " + std::to_string(record.expert));
        }
    }
    if (!transition(slot, SlotEvent::IOComplete, "load_slot"))
        fail(ErrorCode::INTERNAL_ERROR, "invalid IOComplete transition");
    trace_.event("host_ready", Json::object());
    if (!transition(slot, SlotEvent::UploadQueued, "load_slot"))
        fail(ErrorCode::INTERNAL_ERROR, "invalid UploadQueued transition");
    {
        Json fields = Json::object();
        fields.set("slot", static_cast<uint64_t>(slot.index));
        fields.set("bytes", payload.size());
        fields.set("generation", slot.generation);
        trace_.event("upload_queued", std::move(fields));
    }

    try {
        slot.fence = backend_.upload(*slot.buffer, payload.data(), payload.size());
    } catch (const Error &e) {
        if (e.code == ErrorCode::TRANSFER_FAILED) ++metrics_.transfer_failures;
        if (e.code == ErrorCode::DEVICE_LOSS) ++metrics_.device_loss_events;
        mark_error(slot, e.code, e.what());
        throw;
    }
    metrics_.h2d_bytes += payload.size();
    try {
        backend_.wait_fence(slot.fence, config_.t_wait_ms);
    } catch (const Error &e) {
        if (e.code == ErrorCode::FENCE_TIMEOUT) ++metrics_.fence_timeouts;
        mark_error(slot, e.code, e.what());
        throw;
    }
    if (!complete_upload(slot.index, slot.generation))
        fail(ErrorCode::STALE_COMPLETION, "upload completion did not publish slot");

    if (!record.checksum_known) {
        // No source checksum to compare against; the slot copy is still checked
        // against the host payload so a staging fault cannot pass unnoticed.
        std::vector<uint8_t> slot_copy(payload.size());
        std::memcpy(slot_copy.data(), slot.buffer->data, payload.size());
        if (fnv1a64(slot_copy) != fnv1a64(payload)) {
            ++metrics_.checksum_failures;
            mark_error(slot, ErrorCode::EXPERT_CHECKSUM_MISMATCH, "slot payload checksum mismatch");
            fail(ErrorCode::EXPERT_CHECKSUM_MISMATCH, "slot payload checksum mismatch");
        }
        ++metrics_.checksum_unavailable;
    } else {
        ++metrics_.checksum_checks;
        std::vector<uint8_t> slot_copy(payload.size());
        std::memcpy(slot_copy.data(), slot.buffer->data, payload.size());
        if (fnv1a64(slot_copy) != record.checksum) {
            ++metrics_.checksum_failures;
            mark_error(slot, ErrorCode::EXPERT_CHECKSUM_MISMATCH, "slot payload checksum mismatch");
            fail(ErrorCode::EXPERT_CHECKSUM_MISMATCH, "slot payload checksum mismatch");
        }
    }
    ++slot.demand_use_count;
    slot.clock_bit = 1;
}

SlotHandle ResidencyManager::acquire(uint32_t layer, uint32_t expert, uint64_t sequence, bool demand) {
    if (!initialized_) fail(ErrorCode::INTERNAL_ERROR, "residency manager not initialized");
    const ExpertRecord *record = manifest_.expert(layer, expert);
    if (!record) fail(ErrorCode::INVALID_EXPERT_ID,
                      "invalid expert " + std::to_string(expert) + " at layer " + std::to_string(layer));
    uint64_t key = ModelManifest::expert_key(layer, expert);
    // A monotonic access ordinal, not the token position, is required for LRU:
    // multiple experts are demanded at the same token position.
    uint64_t access = ++access_counter_;
    Slot *slot = find_resident(key);
    if (slot && slot->state == SlotState::GPU_READY) {
        ++metrics_.cache_hits;
        slot->last_used = access;
        slot->clock_bit = 1;
        if (demand) ++slot->demand_use_count; else ++slot->prefetch_use_count;
        if (!transition(*slot, SlotEvent::Acquire, "acquire_hit"))
            fail(ErrorCode::INTERNAL_ERROR, "invalid cache-hit acquire transition");
        ++slot->refs;
        Json fields = Json::object();
        fields.set("slot", static_cast<uint64_t>(slot->index));
        fields.set("key", key);
        fields.set("sequence", sequence);
        trace_.event("cache_hit", std::move(fields));
        return SlotHandle(this, slot);
    }
    ++metrics_.cache_misses;
    if (demand) ++metrics_.demand_misses;
    {
        Json fields = Json::object();
        fields.set("key", key);
        fields.set("sequence", sequence);
        fields.set("priority", demand ? "demand" : "prefetch");
        trace_.event("cache_miss", std::move(fields));
    }
    slot = select_victim(sequence);
    if (!slot) fail(ErrorCode::SLOTS_BUSY, "all expert slots are busy after wait deadline");
    evict_slot(*slot, sequence);
    load_slot(*slot, *record, sequence);
    slot->last_used = access;
    if (!transition(*slot, SlotEvent::Acquire, "acquire_miss"))
        fail(ErrorCode::INTERNAL_ERROR, "invalid post-load acquire transition");
    ++slot->refs;
    {
        Json fields = Json::object();
        fields.set("slot", static_cast<uint64_t>(slot->index));
        fields.set("key", key);
        fields.set("sequence", sequence);
        trace_.event("acquire", std::move(fields));
    }
    return SlotHandle(this, slot);
}

void ResidencyManager::release(Slot *slot) {
    if (!slot) return;
    if (slot->refs > 0) --slot->refs;
    slot->clock_bit = 1;
    if (slot->refs == 0 && slot->state == SlotState::IN_USE) {
        transition(*slot, SlotEvent::ReleaseLast, "release");
        Json fields = Json::object();
        fields.set("slot", static_cast<uint64_t>(slot->index));
        fields.set("key", slot->key);
        trace_.event("release", std::move(fields));
    }
    if (slot->refs == 0 && (slot->state == SlotState::ERROR || slot->state == SlotState::CANCELLED))
        transition(*slot, SlotEvent::CallbacksRetired, "release");
}

void ResidencyManager::note_selected(uint32_t layer, const std::vector<uint32_t> &experts, uint64_t sequence) {
    metrics_.selected_experts += experts.size();
    Json fields = Json::object();
    fields.set("layer", static_cast<int64_t>(layer));
    fields.set("sequence", sequence);
    Json ids = Json::array();
    for (uint32_t id : experts) ids.push(Json::integer(id));
    fields.set("experts", std::move(ids));
    trace_.event("expert_selected", std::move(fields));
}

void ResidencyManager::assert_selected_executed() {
    if (metrics_.selected_experts != metrics_.experts_executed)
        fail(ErrorCode::EXPERT_EXECUTION_MISMATCH,
             "selected experts " + std::to_string(metrics_.selected_experts) +
             " != executed experts " + std::to_string(metrics_.experts_executed));
}

void ResidencyManager::pin_slot_for_test(size_t slot_index) {
    if (slot_index >= slots_.size()) return;
    Slot &slot = slots_[slot_index];
    slot.state = SlotState::IN_USE;
    slot.refs = 1;
    slot.clock_bit = 1;
}

void ResidencyManager::unpin_slot_for_test(size_t slot_index) {
    if (slot_index >= slots_.size()) return;
    Slot &slot = slots_[slot_index];
    slot.refs = 0;
    slot.state = SlotState::GPU_READY;
}

void ResidencyManager::set_clock_bits_for_test(const std::vector<uint8_t> &bits) {
    for (size_t i = 0; i < slots_.size() && i < bits.size(); ++i) slots_[i].clock_bit = bits[i] ? 1 : 0;
    clock_hand_ = 0;
}

void ResidencyManager::inject_device_loss() {
    backend_.inject_fault(BackendFault::DeviceLoss);
    for (auto &slot : slots_) {
        if (slot.state == SlotState::IN_USE || slot.state == SlotState::GPU_READY ||
            slot.state == SlotState::UPLOAD_QUEUED) {
            transition(slot, SlotEvent::DeviceLoss, "inject_device_loss");
            slot.error_detail = "injected device loss";
        }
    }
    ++metrics_.device_loss_events;
    ++metrics_.error_events;
    trace_.event("device_loss", Json::object());
}

void ResidencyManager::reset_errors_for_test() {
    backend_.clear_faults();
    for (auto &slot : slots_) {
        if (slot.state == SlotState::ERROR || slot.state == SlotState::CANCELLED)
            transition(slot, SlotEvent::CallbacksRetired, "reset_errors_for_test");
        slot.refs = 0;
        slot.demand_waiters = 0;
        slot.error_detail.clear();
    }
}

void ResidencyManager::shutdown() {
    if (shutdown_) return;
    shutdown_ = true;
    for (auto &entry : pending_prefetch_) {
        scheduler_.cancel_prefetch(entry.second.id);
        metrics_.prefetch_wasted_bytes += entry.second.bytes;
    }
    pending_prefetch_.clear();
    for (auto &entry : host_cache_) {
        uint64_t bytes = entry.second.bytes;
        metrics_.host_cache_bytes = (metrics_.host_cache_bytes >= bytes) ? metrics_.host_cache_bytes - bytes : 0;
        count_wasted_host_cache(bytes, "shutdown");
    }
    host_cache_.clear();
    for (auto &slot : slots_) {
        if (slot.state != SlotState::ERROR && slot.state != SlotState::CANCELLED &&
            slot.state != SlotState::ABSENT) {
            if (transition(slot, SlotEvent::Shutdown, "shutdown"))
                transition(slot, SlotEvent::CallbacksRetired, "shutdown");
        } else if (slot.state == SlotState::ERROR || slot.state == SlotState::CANCELLED) {
            transition(slot, SlotEvent::CallbacksRetired, "shutdown");
        }
    }
    trace_.event("shutdown", Json::object());
}

Json ResidencyManager::slots_json() const {
    Json array = Json::array();
    for (const auto &slot : slots_) {
        Json j = Json::object();
        j.set("index", static_cast<uint64_t>(slot.index));
        j.set("state", slot_state_name(slot.state));
        if (slot.key == std::numeric_limits<uint64_t>::max()) j.set("key", "none");
        else j.set("key", slot.key);
        j.set("layer", static_cast<int64_t>(slot.layer));
        j.set("expert", static_cast<int64_t>(slot.expert));
        j.set("generation", slot.generation);
        j.set("refs", static_cast<int64_t>(slot.refs));
        j.set("last_used", slot.last_used);
        j.set("clock_bit", static_cast<int64_t>(slot.clock_bit));
        j.set("error", slot.error_detail);
        array.push(std::move(j));
    }
    return array;
}

Json ResidencyManager::metrics_json() const {
    Json j = Json::object();
    j.set("cache", metrics_.to_json());
    j.set("budget", budget_.to_json());
    j.set("replacement", replacement_policy_name(config_.replacement));
    j.set("resident_mode", config_.resident_mode);
    j.set("prefetch_enabled", config_.enable_prefetch);
    j.set("slots", slots_json());
    j.set("io", scheduler_.metrics().to_json());
    return j;
}

} // namespace fm
