#pragma once

#include "fm/common.hpp"

#include <atomic>
#include <condition_variable>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fm {

class MappedFile;

enum class ReaderMode { Resident, Buffered, Mmap, Direct };
const char *reader_mode_name(ReaderMode mode);
ReaderMode parse_reader_mode(const std::string &name);

struct Extent {
    uint64_t offset = 0;
    uint64_t size = 0;
};

enum class ReadPriority { Demand = 0, Prefetch = 1 };
enum class SubscriberType { Demand, Prefetch };

struct ReadRequest {
    uint64_t id = 0;
    ReadPriority priority = ReadPriority::Demand;
    uint32_t demand_subscribers = 0;
    uint32_t prefetch_subscribers = 0;
    bool cancelled = false;

    bool has_subscribers() const { return demand_subscribers || prefetch_subscribers; }
    bool cancel(SubscriberType type) {
        if (type == SubscriberType::Demand && demand_subscribers) --demand_subscribers;
        if (type == SubscriberType::Prefetch && prefetch_subscribers) --prefetch_subscribers;
        if (!has_subscribers()) cancelled = true;
        return cancelled;
    }
};

struct IOMetrics {
    uint64_t demand_records = 0;
    uint64_t prefetch_records = 0;
    uint64_t read_requests = 0;
    uint64_t physical_bytes = 0;
    uint64_t useful_bytes = 0;
    uint64_t coalesced_requests = 0;
    uint64_t direct_fallbacks = 0;
    uint64_t source_verifications = 0;
    uint64_t source_changed_events = 0;
    uint64_t prefetch_issued = 0;
    uint64_t prefetch_cancelled = 0;
    uint64_t prefetch_useful_bytes = 0;
    uint64_t prefetch_wasted_bytes = 0;
    uint64_t max_queue_depth = 0;
    uint64_t total_latency_ns = 0;
    uint64_t max_latency_ns = 0;

    Json to_json() const;
};

struct CoalescedGroup {
    uint64_t offset = 0;
    uint64_t end = 0;
    std::vector<size_t> request_indices;
};

std::vector<CoalescedGroup> coalesce_extents(const std::vector<Extent> &extents, uint64_t max_gap);
uint64_t extents_total_bytes(const std::vector<Extent> &extents);

struct ReadResult {
    std::vector<uint8_t> bytes;
    uint64_t physical_bytes = 0;
    bool fallback = false;
};

class PositionalReader {
public:
    virtual ~PositionalReader() = default;
    virtual ReadResult read_span(uint64_t offset, uint64_t size) = 0;
    virtual std::string name() const = 0;
};

std::unique_ptr<PositionalReader> make_reader(ReaderMode mode, const std::string &path,
                                              const MappedFile *mapping,
                                              bool force_direct_fallback);

enum class PrefetchStatus { Pending, Ready, Cancelled, NotFound };

struct PrefetchResult {
    std::vector<uint8_t> bytes;
    bool cancelled = false;
    ErrorCode error = ErrorCode::OK;
    std::string message;
};

struct SchedulerConfig {
    ReaderMode mode = ReaderMode::Buffered;
    std::string path;
    FileIdentity expected_identity;
    uint32_t queue_depth = 2;
    uint64_t coalesce_gap = 4096;
    bool force_direct_fallback = false;
};

class ReadScheduler {
public:
    ReadScheduler(SchedulerConfig config, std::unique_ptr<PositionalReader> reader,
                  TraceWriter *trace = nullptr);
    ~ReadScheduler();

    std::vector<uint8_t> read_record(const std::vector<Extent> &extents, ReadPriority priority);
    uint64_t submit_prefetch(const std::vector<Extent> &extents, uint64_t key, uint64_t sequence);
    PrefetchStatus poll_prefetch(uint64_t id, std::vector<uint8_t> &payload);
    void cancel_prefetch(uint64_t id);
    void cancel_all_prefetch(const std::string &reason);
    void wait_idle();
    void note_prefetch_useful(uint64_t bytes);
    void note_prefetch_wasted(uint64_t bytes);
    void verify_source();
    IOMetrics metrics() const;
    ReaderMode mode() const { return config_.mode; }
    const std::string &path() const { return config_.path; }
    uint64_t next_id() const { return next_id_.load(); }

private:
    struct PrefetchState {
        std::atomic<bool> cancelled{false};
    };
    struct PrefetchTask {
        uint64_t id = 0;
        uint64_t key = 0;
        uint64_t bytes = 0;
        uint64_t sequence = 0;
        std::shared_ptr<PrefetchState> state;
        std::shared_future<PrefetchResult> future;
        bool reaped = false;
    };

    PrefetchResult read_internal(const std::vector<Extent> &extents, ReadPriority priority,
                                 const std::shared_ptr<PrefetchState> &state);
    void update_metrics_after_read(const PrefetchResult &result, const std::vector<Extent> &extents,
                                   ReadPriority priority, uint64_t latency_ns,
                                   uint64_t physical_bytes, uint64_t request_count,
                                   uint64_t coalesced, uint64_t fallbacks);
    void reap_cancelled_locked();

    SchedulerConfig config_;
    std::unique_ptr<PositionalReader> reader_;
    TraceWriter *trace_ = nullptr;
    mutable std::mutex mutex_;
    IOMetrics metrics_;
    std::atomic<uint64_t> next_id_{1};
    std::map<uint64_t, PrefetchTask> prefetch_tasks_;
};

} // namespace fm
