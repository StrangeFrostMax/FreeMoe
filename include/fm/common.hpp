#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/resource.h>
#include <sys/stat.h>
#include <utility>
#include <vector>

namespace fm {

enum class ErrorCode {
    OK = 0,
    MALFORMED_GGUF,
    UNSUPPORTED_ARCHITECTURE,
    UNSUPPORTED_QUANTISATION,
    INVALID_TOKENIZER,
    MISSING_REQUIRED_TENSOR,
    UNKNOWN_TENSOR,
    DUPLICATE_TENSOR,
    SHAPE_MISMATCH,
    METADATA_MISMATCH,
    SOURCE_CHANGED,
    IO_ERROR,
    SHORT_READ,
    DIRECT_IO_FALLBACK,
    ALLOCATION_FAILED,
    BUDGET_TOO_SMALL,
    INVALID_EXPERT_ID,
    EXPERT_CHECKSUM_MISMATCH,
    EXPERT_EXECUTION_MISMATCH,
    SLOTS_BUSY,
    STALE_COMPLETION,
    TRANSFER_FAILED,
    FENCE_TIMEOUT,
    DEVICE_LOSS,
    GPU_BACKEND_UNIMPLEMENTED,
    CPU_FALLBACK_DISABLED,
    INVALID_ARGUMENT,
    NOT_IMPLEMENTED,
    INTERNAL_ERROR,
};

const char *error_name(ErrorCode code);
int error_exit_code(ErrorCode code);

struct Error : std::runtime_error {
    ErrorCode code;
    std::string detail;
    Error(ErrorCode code_, const std::string &message)
        : std::runtime_error(message), code(code_), detail(message) {}
};

[[noreturn]] inline void fail(ErrorCode code, const std::string &message) {
    throw Error(code, message);
}

inline uint64_t fnv1a64(const void *data, size_t size) {
    const auto *p = static_cast<const uint8_t *>(data);
    uint64_t h = 0xCBF29CE484222325ull;
    for (size_t i = 0; i < size; ++i) {
        h ^= p[i];
        h *= 0x100000001B3ull;
    }
    return h;
}

inline uint64_t fnv1a64(const std::vector<uint8_t> &data) {
    return fnv1a64(data.data(), data.size());
}

inline std::string hex64(uint64_t value) {
    std::ostringstream os;
    os << std::hex << std::setw(16) << std::setfill('0') << value;
    return os.str();
}

inline bool add_overflow_u64(uint64_t a, uint64_t b, uint64_t &out) {
    if (a > std::numeric_limits<uint64_t>::max() - b) return true;
    out = a + b;
    return false;
}

inline bool mul_overflow_u64(uint64_t a, uint64_t b, uint64_t &out) {
    if (a && b > std::numeric_limits<uint64_t>::max() / a) return true;
    out = a * b;
    return false;
}

inline uint64_t checked_mul(uint64_t a, uint64_t b, const std::string &where) {
    uint64_t out = 0;
    if (mul_overflow_u64(a, b, out)) fail(ErrorCode::MALFORMED_GGUF, "integer overflow in " + where);
    return out;
}

inline uint64_t checked_add(uint64_t a, uint64_t b, const std::string &where) {
    uint64_t out = 0;
    if (add_overflow_u64(a, b, out)) fail(ErrorCode::MALFORMED_GGUF, "integer overflow in " + where);
    return out;
}

inline uint64_t align_up(uint64_t value, uint64_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0)
        fail(ErrorCode::MALFORMED_GGUF, "alignment is not a power of two");
    if (value > std::numeric_limits<uint64_t>::max() - (alignment - 1))
        fail(ErrorCode::MALFORMED_GGUF, "alignment overflow");
    return (value + alignment - 1) & ~(alignment - 1);
}

inline uint64_t now_ns() {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}

inline uint64_t now_ms() {
    return now_ns() / 1000000ull;
}

class Timer {
public:
    Timer() : start_(now_ns()) {}
    void reset() { start_ = now_ns(); }
    uint64_t elapsed_ns() const { return now_ns() - start_; }
    double elapsed_ms() const { return static_cast<double>(elapsed_ns()) / 1e6; }
private:
    uint64_t start_;
};

struct FileIdentity {
    uint64_t size = 0;
    uint64_t dev = 0;
    uint64_t ino = 0;
    int64_t mtime_ns = 0;
    int64_t ctime_ns = 0;
    // Sparse content probe (first/middle/last 64 bytes). It is not a
    // cryptographic hash, but catches payload edits on coarse-timestamp
    // filesystems without reading the whole model before every batch.
    uint64_t content_probe = 0;

    bool operator==(const FileIdentity &o) const {
        return size == o.size && dev == o.dev && ino == o.ino &&
               mtime_ns == o.mtime_ns && ctime_ns == o.ctime_ns &&
               content_probe == o.content_probe;
    }
};

FileIdentity current_identity(const std::string &path);

struct RssSnapshot {
    uint64_t max_rss_kib = 0;
    uint64_t minor_faults = 0;
    uint64_t major_faults = 0;
    double user_cpu_s = 0.0;
    double system_cpu_s = 0.0;
};

RssSnapshot process_snapshot();

std::vector<uint8_t> read_binary_file(const std::string &path, uint64_t max_bytes = 1ull << 30);
void write_text_file(const std::string &path, const std::string &text);
std::string read_text_file(const std::string &path);
std::string trim(const std::string &s);
// Remove only CR/LF line endings. Prompt whitespace is semantically significant.
std::string strip_eol(const std::string &s);

class Json {
public:
    enum class Type { Null, Bool, Int, String, Array, Object };

    Json() : type_(Type::Null) {}
    static Json null() { return Json(); }
    static Json boolean(bool v) { Json j; j.type_ = Type::Bool; j.bool_ = v; return j; }
    static Json integer(int64_t v) { Json j; j.type_ = Type::Int; j.int_ = v; return j; }
    static Json string(const std::string &v) { Json j; j.type_ = Type::String; j.string_ = v; return j; }
    static Json array() { Json j; j.type_ = Type::Array; return j; }
    static Json object() { Json j; j.type_ = Type::Object; return j; }

    Json &push(Json value) {
        if (type_ != Type::Array) fail(ErrorCode::INTERNAL_ERROR, "Json::push on non-array");
        array_.push_back(std::move(value));
        return *this;
    }
    Json &set(const std::string &key, Json value) {
        if (type_ != Type::Object) fail(ErrorCode::INTERNAL_ERROR, "Json::set on non-object");
        object_.push_back({key, std::move(value)});
        return *this;
    }
    Json &set(const std::string &key, const std::string &value) { return set(key, Json::string(value)); }
    Json &set(const std::string &key, const char *value) { return set(key, Json::string(value)); }
    Json &set(const std::string &key, int64_t value) { return set(key, Json::integer(value)); }
    Json &set(const std::string &key, uint64_t value) {
        if (value > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
            fail(ErrorCode::INTERNAL_ERROR, "JSON integer overflow");
        return set(key, Json::integer(static_cast<int64_t>(value)));
    }
    Json &set(const std::string &key, int value) { return set(key, Json::integer(value)); }
    Json &set(const std::string &key, bool value) { return set(key, Json::boolean(value)); }

    std::string dump() const {
        std::ostringstream os;
        dump_into(os);
        return os.str();
    }
    Type type() const { return type_; }
    const std::vector<Json> &items() const { return array_; }
    const std::vector<std::pair<std::string, Json>> &members() const { return object_; }
    const std::string &str() const { return string_; }
    int64_t integer_value() const { return int_; }
    bool bool_value() const { return bool_; }

private:
    void dump_into(std::ostringstream &os) const {
        switch (type_) {
        case Type::Null: os << "null"; break;
        case Type::Bool: os << (bool_ ? "true" : "false"); break;
        case Type::Int: os << int_; break;
        case Type::String: dump_string(os, string_); break;
        case Type::Array: {
            os << '[';
            for (size_t i = 0; i < array_.size(); ++i) {
                if (i) os << ',';
                array_[i].dump_into(os);
            }
            os << ']';
            break;
        }
        case Type::Object: {
            os << '{';
            for (size_t i = 0; i < object_.size(); ++i) {
                if (i) os << ',';
                dump_string(os, object_[i].first);
                os << ':';
                object_[i].second.dump_into(os);
            }
            os << '}';
            break;
        }
        }
    }
    static void dump_string(std::ostringstream &os, const std::string &s) {
        static const char *hex = "0123456789abcdef";
        os << '"';
        for (unsigned char c : s) {
            switch (c) {
            case '"': os << "\\\""; break;
            case '\\': os << "\\\\"; break;
            case '\b': os << "\\b"; break;
            case '\f': os << "\\f"; break;
            case '\n': os << "\\n"; break;
            case '\r': os << "\\r"; break;
            case '\t': os << "\\t"; break;
            default:
                if (c < 0x20) {
                    os << "\\u00" << hex[(c >> 4) & 0xf] << hex[c & 0xf];
                } else {
                    os << static_cast<char>(c);
                }
            }
        }
        os << '"';
    }

    Type type_;
    bool bool_ = false;
    int64_t int_ = 0;
    std::string string_;
    std::vector<Json> array_;
    std::vector<std::pair<std::string, Json>> object_;
};

class TraceWriter {
public:
    TraceWriter() = default;
    explicit TraceWriter(std::string path) : path_(std::move(path)) {
        if (path_.empty()) return;
        std::filesystem::create_directories(std::filesystem::path(path_).parent_path());
        out_.open(path_, std::ios::trunc);
        if (!out_) fail(ErrorCode::IO_ERROR, "cannot open trace file: " + path_);
    }
    void event(const std::string &type, Json fields = Json::object()) {
        if (!out_.is_open()) return;
        std::lock_guard<std::mutex> lock(mutex_);
        Json line = Json::object();
        line.set("ts_ns", static_cast<uint64_t>(now_ns()));
        line.set("event", type);
        if (fields.type() == Json::Type::Object) {
            for (const auto &kv : fields.members()) line.set(kv.first, kv.second);
        }
        out_ << line.dump() << '\n';
        out_.flush();
        event_types_.insert(type);
    }
    const std::set<std::string> &event_types() const { return event_types_; }
    const std::string &path() const { return path_; }
private:
    std::string path_;
    std::ofstream out_;
    std::mutex mutex_;
    std::set<std::string> event_types_;
};

inline std::string fixed_double(double value, int precision = 6) {
    std::ostringstream os;
    os << std::fixed << std::setprecision(precision) << value;
    return os.str();
}

inline double median(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t n = values.size();
    return (n % 2) ? values[n / 2] : (values[n / 2 - 1] + values[n / 2]) / 2.0;
}

inline double percentile95(std::vector<double> values) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    size_t index = static_cast<size_t>(std::ceil(0.95 * values.size())) - 1;
    return values[std::min(index, values.size() - 1)];
}

} // namespace fm
