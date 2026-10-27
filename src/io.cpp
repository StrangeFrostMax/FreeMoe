#include "fm/io.hpp"
#include "fm/gguf.hpp"

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fcntl.h>
#include <future>
#include <sys/statvfs.h>
#include <unistd.h>

namespace fm {

const char *reader_mode_name(ReaderMode mode) {
    switch (mode) {
    case ReaderMode::Resident: return "resident";
    case ReaderMode::Buffered: return "buffered";
    case ReaderMode::Mmap: return "mmap";
    case ReaderMode::Direct: return "direct";
    }
    return "unknown";
}

ReaderMode parse_reader_mode(const std::string &name) {
    if (name == "resident") return ReaderMode::Resident;
    if (name == "buffered") return ReaderMode::Buffered;
    if (name == "mmap") return ReaderMode::Mmap;
    if (name == "direct") return ReaderMode::Direct;
    fail(ErrorCode::INVALID_ARGUMENT, "unknown reader mode: " + name);
}

uint64_t extents_total_bytes(const std::vector<Extent> &extents) {
    uint64_t total = 0;
    for (const auto &e : extents) {
        if (e.size == 0 || e.offset > std::numeric_limits<uint64_t>::max() - e.size)
            fail(ErrorCode::INVALID_ARGUMENT, "invalid extent");
        if (total > std::numeric_limits<uint64_t>::max() - e.size)
            fail(ErrorCode::INVALID_ARGUMENT, "extent total overflow");
        total += e.size;
    }
    return total;
}

std::vector<CoalescedGroup> coalesce_extents(const std::vector<Extent> &extents, uint64_t max_gap) {
    std::vector<size_t> order(extents.size());
    for (size_t i = 0; i < order.size(); ++i) order[i] = i;
    std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
        if (extents[a].offset != extents[b].offset) return extents[a].offset < extents[b].offset;
        return extents[a].size < extents[b].size;
    });
    std::vector<CoalescedGroup> groups;
    for (size_t idx : order) {
        const Extent &e = extents[idx];
        if (e.size == 0 || e.offset > std::numeric_limits<uint64_t>::max() - e.size)
            fail(ErrorCode::INVALID_ARGUMENT, "invalid extent");
        uint64_t end = e.offset + e.size;
        bool coalesce = false;
        if (!groups.empty()) {
            CoalescedGroup &g = groups.back();
            bool gap_ok = max_gap <= std::numeric_limits<uint64_t>::max() - g.end;
            coalesce = gap_ok && e.offset <= g.end + max_gap;
            if (coalesce) {
                g.end = std::max(g.end, end);
                g.request_indices.push_back(idx);
                continue;
            }
        }
        CoalescedGroup g;
        g.offset = e.offset;
        g.end = end;
        g.request_indices.push_back(idx);
        groups.push_back(std::move(g));
    }
    return groups;
}

Json IOMetrics::to_json() const {
    Json j = Json::object();
    j.set("demand_records", demand_records);
    j.set("prefetch_records", prefetch_records);
    j.set("read_requests", read_requests);
    j.set("physical_bytes", physical_bytes);
    j.set("useful_bytes", useful_bytes);
    j.set("coalesced_requests", coalesced_requests);
    j.set("direct_fallbacks", direct_fallbacks);
    j.set("source_verifications", source_verifications);
    j.set("source_changed_events", source_changed_events);
    j.set("prefetch_issued", prefetch_issued);
    j.set("prefetch_cancelled", prefetch_cancelled);
    j.set("prefetch_useful_bytes", prefetch_useful_bytes);
    j.set("prefetch_wasted_bytes", prefetch_wasted_bytes);
    j.set("max_queue_depth", max_queue_depth);
    j.set("total_latency_ns", total_latency_ns);
    j.set("max_latency_ns", max_latency_ns);
    double amplification = useful_bytes ? static_cast<double>(physical_bytes) / static_cast<double>(useful_bytes) : 0.0;
    j.set("read_amplification", fixed_double(amplification, 6));
    return j;
}

namespace {

class BufferedReader final : public PositionalReader {
public:
    explicit BufferedReader(const std::string &path) : path_(path) {
        fd_ = ::open(path.c_str(), O_RDONLY);
        if (fd_ < 0) fail(ErrorCode::IO_ERROR, "open failed for " + path + ": " + std::strerror(errno));
        struct stat st {};
        if (::fstat(fd_, &st) != 0) fail(ErrorCode::IO_ERROR, "fstat failed for " + path);
        file_size_ = static_cast<uint64_t>(st.st_size);
    }
    ~BufferedReader() override { if (fd_ >= 0) ::close(fd_); }
    std::string name() const override { return "buffered"; }
    ReadResult read_span(uint64_t offset, uint64_t size) override {
        if (size > file_size_ || offset > file_size_ - size)
            fail(ErrorCode::SHORT_READ, "read beyond EOF: " + path_);
        ReadResult result;
        result.bytes.resize(static_cast<size_t>(size));
        uint64_t done = 0;
        while (done < size) {
            ssize_t n = ::pread(fd_, result.bytes.data() + done,
                                static_cast<size_t>(size - done),
                                static_cast<off_t>(offset + done));
            if (n < 0) {
                if (errno == EINTR) continue;
                fail(ErrorCode::IO_ERROR, "pread failed for " + path_ + ": " + std::strerror(errno));
            }
            if (n == 0) fail(ErrorCode::SHORT_READ, "short read from " + path_);
            done += static_cast<uint64_t>(n);
        }
        result.physical_bytes = size;
        return result;
    }
private:
    std::string path_;
    int fd_ = -1;
    uint64_t file_size_ = 0;
};

class MmapReader final : public PositionalReader {
public:
    explicit MmapReader(const MappedFile *mapping) : mapping_(mapping) {
        if (!mapping_ || !mapping_->data()) fail(ErrorCode::INVALID_ARGUMENT, "mmap reader needs a mapping");
    }
    std::string name() const override { return "mmap"; }
    ReadResult read_span(uint64_t offset, uint64_t size) override {
        uint64_t file_size = mapping_->size();
        if (size > file_size || offset > file_size - size)
            fail(ErrorCode::SHORT_READ, "mmap read beyond EOF");
        ReadResult result;
        result.bytes.assign(mapping_->data() + offset, mapping_->data() + offset + size);
        // mmap has no syscall request boundary; physical bytes are logical bytes.
        result.physical_bytes = size;
        return result;
    }
private:
    const MappedFile *mapping_;
};

class DirectReader final : public PositionalReader {
public:
    DirectReader(const std::string &path, bool force_fallback)
        : path_(path), force_fallback_(force_fallback) {
        fallback_ = std::make_unique<BufferedReader>(path);
#ifdef O_DIRECT
        direct_fd_ = ::open(path.c_str(), O_RDONLY | O_DIRECT);
        if (direct_fd_ >= 0) {
            struct statvfs vfs {};
            uint64_t candidate = 4096;
            if (::fstatvfs(direct_fd_, &vfs) == 0 && vfs.f_bsize >= 512)
                candidate = static_cast<uint64_t>(vfs.f_bsize);
            if (candidate && (candidate & (candidate - 1)) == 0 && candidate <= (1u << 20))
                alignment_ = candidate;
        }
#endif
    }
    ~DirectReader() override { if (direct_fd_ >= 0) ::close(direct_fd_); }
    std::string name() const override { return "direct"; }
    ReadResult read_span(uint64_t offset, uint64_t size) override {
        if (force_fallback_ || direct_fd_ < 0 || size == 0 ||
            offset > std::numeric_limits<uint64_t>::max() - size)
            return fallback_read(offset, size);
        uint64_t end = offset + size;
        uint64_t aligned_start = offset & ~(alignment_ - 1);
        uint64_t prefix = offset - aligned_start;
        uint64_t aligned_end = align_up(end, alignment_);
        uint64_t read_len = aligned_end - aligned_start;
        void *buffer = nullptr;
        if (::posix_memalign(&buffer, static_cast<size_t>(alignment_),
                             static_cast<size_t>(read_len)) != 0)
            fail(ErrorCode::ALLOCATION_FAILED, "aligned direct-I/O bounce buffer allocation failed");
        ssize_t n = ::pread(direct_fd_, buffer, static_cast<size_t>(read_len),
                            static_cast<off_t>(aligned_start));
        int saved_errno = errno;
        if (n >= 0 && static_cast<uint64_t>(n) >= prefix + size) {
            ReadResult result;
            result.bytes.resize(static_cast<size_t>(size));
            std::memcpy(result.bytes.data(), static_cast<uint8_t *>(buffer) + prefix,
                        static_cast<size_t>(size));
            result.physical_bytes = static_cast<uint64_t>(n);
            std::free(buffer);
            return result;
        }
        std::free(buffer);
        (void)saved_errno;
        return fallback_read(offset, size);
    }
private:
    ReadResult fallback_read(uint64_t offset, uint64_t size) {
        ReadResult result = fallback_->read_span(offset, size);
        result.fallback = true;
        return result;
    }
    std::string path_;
    bool force_fallback_ = false;
    int direct_fd_ = -1;
    uint64_t alignment_ = 4096;
    std::unique_ptr<BufferedReader> fallback_;
};

} // namespace

std::unique_ptr<PositionalReader> make_reader(ReaderMode mode, const std::string &path,
                                              const MappedFile *mapping,
                                              bool force_direct_fallback) {
    switch (mode) {
    case ReaderMode::Resident:
    case ReaderMode::Buffered:
        return std::make_unique<BufferedReader>(path);
    case ReaderMode::Mmap:
        return std::make_unique<MmapReader>(mapping);
    case ReaderMode::Direct:
        return std::make_unique<DirectReader>(path, force_direct_fallback);
    }
    fail(ErrorCode::INVALID_ARGUMENT, "unknown reader mode");
}

// ---------------------------------------------------------------------------
// ReadScheduler
// ---------------------------------------------------------------------------

ReadScheduler::ReadScheduler(SchedulerConfig config, std::unique_ptr<PositionalReader> reader,
                             TraceWriter *trace)
    : config_(std::move(config)), reader_(std::move(reader)), trace_(trace) {
    if (!reader_) fail(ErrorCode::INVALID_ARGUMENT, "ReadScheduler needs a reader");
    if (config_.queue_depth == 0) fail(ErrorCode::INVALID_ARGUMENT, "queue depth must be positive");
}

ReadScheduler::~ReadScheduler() {
    cancel_all_prefetch("shutdown");
    std::vector<std::shared_future<PrefetchResult>> futures;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &entry : prefetch_tasks_) futures.push_back(entry.second.future);
    }
    for (auto &future : futures) {
        if (future.valid()) future.wait();
    }
}

void ReadScheduler::verify_source() {
    FileIdentity current = current_identity(config_.path);
    bool ok = current == config_.expected_identity;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        ++metrics_.source_verifications;
        if (!ok) ++metrics_.source_changed_events;
    }
    if (trace_) {
        Json fields = Json::object();
        fields.set("ok", ok);
        fields.set("reader", reader_->name());
        trace_->event("source_verify", std::move(fields));
    }
    if (!ok) fail(ErrorCode::SOURCE_CHANGED,
                  "model source changed after index creation: " + config_.path);
}

void ReadScheduler::update_metrics_after_read(const PrefetchResult &result,
                                              const std::vector<Extent> &extents,
                                              ReadPriority priority, uint64_t latency_ns,
                                              uint64_t physical_bytes, uint64_t request_count,
                                              uint64_t coalesced, uint64_t fallbacks) {
    (void)result;
    uint64_t useful = extents_total_bytes(extents);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        if (priority == ReadPriority::Demand) ++metrics_.demand_records;
        else ++metrics_.prefetch_records;
        metrics_.read_requests += request_count;
        metrics_.physical_bytes += physical_bytes;
        metrics_.useful_bytes += useful;
        metrics_.coalesced_requests += coalesced;
        metrics_.direct_fallbacks += fallbacks;
        metrics_.total_latency_ns += latency_ns;
        metrics_.max_latency_ns = std::max(metrics_.max_latency_ns, latency_ns);
    }
    if (trace_) {
        Json fields = Json::object();
        fields.set("priority", priority == ReadPriority::Demand ? "demand" : "prefetch");
        fields.set("reader", reader_->name());
        fields.set("physical_bytes", physical_bytes);
        fields.set("useful_bytes", useful);
        fields.set("requests", request_count);
        fields.set("coalesced", coalesced);
        fields.set("fallbacks", fallbacks);
        fields.set("latency_ns", latency_ns);
        trace_->event("io_complete", std::move(fields));
    }
}

PrefetchResult ReadScheduler::read_internal(const std::vector<Extent> &extents,
                                            ReadPriority priority,
                                            const std::shared_ptr<PrefetchState> &state) {
    PrefetchResult result;
    Timer timer;
    uint64_t physical_bytes = 0;
    uint64_t request_count = 0;
    uint64_t coalesced = 0;
    uint64_t fallbacks = 0;
    auto is_cancelled = [&]() { return state && state->cancelled.load(); };
    try {
        if (is_cancelled()) {
            result.cancelled = true;
            update_metrics_after_read(result, extents, priority, timer.elapsed_ns(), 0, 0, 0, 0);
            return result;
        }
        verify_source();
        if (is_cancelled()) {
            result.cancelled = true;
            update_metrics_after_read(result, extents, priority, timer.elapsed_ns(), 0, 0, 0, 0);
            return result;
        }
        uint64_t payload_bytes = extents_total_bytes(extents);
        std::vector<uint8_t> payload(static_cast<size_t>(payload_bytes));
        std::vector<uint64_t> output_offsets(extents.size());
        uint64_t cursor = 0;
        for (size_t i = 0; i < extents.size(); ++i) {
            output_offsets[i] = cursor;
            cursor += extents[i].size;
        }
        std::vector<CoalescedGroup> groups = coalesce_extents(extents, config_.coalesce_gap);
        for (const auto &group : groups) {
            if (is_cancelled()) break;
            uint64_t span = group.end - group.offset;
            ReadResult rr = reader_->read_span(group.offset, span);
            physical_bytes += rr.physical_bytes;
            ++request_count;
            if (rr.fallback) ++fallbacks;
            if (group.request_indices.size() > 1)
                coalesced += group.request_indices.size() - 1;
            for (size_t idx : group.request_indices) {
                const Extent &e = extents[idx];
                if (e.offset < group.offset || e.offset + e.size > group.end ||
                    rr.bytes.size() < static_cast<size_t>((e.offset - group.offset) + e.size))
                    fail(ErrorCode::INTERNAL_ERROR, "coalesced read result is incomplete");
                std::memcpy(payload.data() + output_offsets[idx],
                            rr.bytes.data() + (e.offset - group.offset),
                            static_cast<size_t>(e.size));
            }
        }
        if (is_cancelled()) {
            result.cancelled = true;
        } else {
            result.bytes = std::move(payload);
        }
    } catch (const Error &e) {
        result.error = e.code;
        result.message = e.what();
        result.bytes.clear();
    } catch (const std::exception &e) {
        result.error = ErrorCode::INTERNAL_ERROR;
        result.message = e.what();
        result.bytes.clear();
    }
    if (fallbacks && trace_) {
        Json fields = Json::object();
        fields.set("reader", reader_->name());
        fields.set("count", fallbacks);
        trace_->event("direct_io_fallback", std::move(fields));
    }
    update_metrics_after_read(result, extents, priority, timer.elapsed_ns(),
                              physical_bytes, request_count, coalesced, fallbacks);
    return result;
}

std::vector<uint8_t> ReadScheduler::read_record(const std::vector<Extent> &extents,
                                                ReadPriority priority) {
    // Demand reads execute synchronously on this thread and therefore take
    // priority over background prefetch work. We do not cancel every prefetch:
    // a completed prediction may already be useful to the next demand miss.
    // Cancellation is explicit (stale prediction, queue pressure, shutdown).
    PrefetchResult result = read_internal(extents, priority, nullptr);
    if (result.error != ErrorCode::OK)
        throw Error(result.error, result.message);
    if (result.cancelled)
        fail(ErrorCode::INTERNAL_ERROR, "demand read was unexpectedly cancelled");
    return std::move(result.bytes);
}

uint64_t ReadScheduler::submit_prefetch(const std::vector<Extent> &extents, uint64_t key,
                                        uint64_t sequence) {
    uint64_t id = next_id_.fetch_add(1);
    auto state = std::make_shared<PrefetchState>();
    auto future = std::async(std::launch::async, [this, extents, state]() {
        return read_internal(extents, ReadPriority::Prefetch, state);
    }).share();
    uint64_t bytes = extents_total_bytes(extents);
    {
        std::lock_guard<std::mutex> lock(mutex_);
        reap_cancelled_locked();
        uint64_t active = 0;
        for (const auto &entry : prefetch_tasks_) if (!entry.second.reaped) ++active;
        if (active >= config_.queue_depth) {
            for (auto &entry : prefetch_tasks_) {
                if (!entry.second.reaped && !entry.second.state->cancelled.exchange(true)) {
                    ++metrics_.prefetch_cancelled;
                    metrics_.prefetch_wasted_bytes += entry.second.bytes;
                    entry.second.reaped = true;
                    break;
                }
            }
        }
        PrefetchTask task;
        task.id = id;
        task.key = key;
        task.bytes = bytes;
        task.sequence = sequence;
        task.state = state;
        task.future = future;
        prefetch_tasks_[id] = std::move(task);
        active = 0;
        for (const auto &entry : prefetch_tasks_) if (!entry.second.reaped) ++active;
        metrics_.max_queue_depth = std::max(metrics_.max_queue_depth, active);
        ++metrics_.prefetch_issued;
    }
    if (trace_) {
        Json fields = Json::object();
        fields.set("id", id);
        fields.set("key", key);
        fields.set("bytes", bytes);
        fields.set("sequence", sequence);
        trace_->event("prefetch_issue", std::move(fields));
    }
    return id;
}

void ReadScheduler::reap_cancelled_locked() {
    // Keep completed/cancelled entries until shutdown. Erasing a shared_future
    // under the mutex could block on the worker that is trying to update the
    // same metrics mutex. Marking them reaped is sufficient for queue bounds.
}

PrefetchStatus ReadScheduler::poll_prefetch(uint64_t id, std::vector<uint8_t> &payload) {
    std::shared_future<PrefetchResult> future;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = prefetch_tasks_.find(id);
        if (it == prefetch_tasks_.end() || it->second.reaped) return PrefetchStatus::NotFound;
        if (it->second.future.wait_for(std::chrono::seconds(0)) != std::future_status::ready)
            return PrefetchStatus::Pending;
        future = it->second.future;
        it->second.reaped = true;
    }
    PrefetchResult result = future.get();  // copy from shared state
    if (result.error != ErrorCode::OK) throw Error(result.error, result.message);
    if (result.cancelled) return PrefetchStatus::Cancelled;
    payload = std::move(result.bytes);
    return PrefetchStatus::Ready;
}

void ReadScheduler::cancel_prefetch(uint64_t id) {
    bool changed = false;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = prefetch_tasks_.find(id);
        if (it != prefetch_tasks_.end() && !it->second.reaped &&
            !it->second.state->cancelled.exchange(true)) {
            ++metrics_.prefetch_cancelled;
            metrics_.prefetch_wasted_bytes += it->second.bytes;
            it->second.reaped = true;
            changed = true;
        }
    }
    if (changed && trace_) {
        Json fields = Json::object();
        fields.set("id", id);
        fields.set("reason", "stale");
        trace_->event("prefetch_cancel", std::move(fields));
    }
}

void ReadScheduler::cancel_all_prefetch(const std::string &reason) {
    std::vector<uint64_t> cancelled;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &entry : prefetch_tasks_) {
            if (!entry.second.reaped && !entry.second.state->cancelled.exchange(true)) {
                ++metrics_.prefetch_cancelled;
                metrics_.prefetch_wasted_bytes += entry.second.bytes;
                entry.second.reaped = true;
                cancelled.push_back(entry.first);
            }
        }
    }
    if (trace_) {
        for (uint64_t id : cancelled) {
            Json fields = Json::object();
            fields.set("id", id);
            fields.set("reason", reason);
            trace_->event("prefetch_cancel", std::move(fields));
        }
    }
}

void ReadScheduler::wait_idle() {
    cancel_all_prefetch("wait_idle");
    std::vector<std::shared_future<PrefetchResult>> futures;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto &entry : prefetch_tasks_) {
            if (entry.second.future.valid()) futures.push_back(entry.second.future);
            entry.second.reaped = true;
        }
    }
    for (auto &future : futures) future.wait();
}

void ReadScheduler::note_prefetch_useful(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_.prefetch_useful_bytes += bytes;
}

void ReadScheduler::note_prefetch_wasted(uint64_t bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    metrics_.prefetch_wasted_bytes += bytes;
}

IOMetrics ReadScheduler::metrics() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return metrics_;
}

} // namespace fm
