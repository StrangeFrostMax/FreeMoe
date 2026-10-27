#include "fm/common.hpp"

#include <cerrno>
#include <cctype>
#include <cstdio>

namespace fm {

const char *error_name(ErrorCode code) {
    switch (code) {
    case ErrorCode::OK: return "OK";
    case ErrorCode::MALFORMED_GGUF: return "MALFORMED_GGUF";
    case ErrorCode::UNSUPPORTED_ARCHITECTURE: return "UNSUPPORTED_ARCHITECTURE";
    case ErrorCode::UNSUPPORTED_QUANTISATION: return "UNSUPPORTED_QUANTISATION";
    case ErrorCode::INVALID_TOKENIZER: return "INVALID_TOKENIZER";
    case ErrorCode::MISSING_REQUIRED_TENSOR: return "MISSING_REQUIRED_TENSOR";
    case ErrorCode::UNKNOWN_TENSOR: return "UNKNOWN_TENSOR";
    case ErrorCode::DUPLICATE_TENSOR: return "DUPLICATE_TENSOR";
    case ErrorCode::SHAPE_MISMATCH: return "SHAPE_MISMATCH";
    case ErrorCode::METADATA_MISMATCH: return "METADATA_MISMATCH";
    case ErrorCode::SOURCE_CHANGED: return "SOURCE_CHANGED";
    case ErrorCode::IO_ERROR: return "IO_ERROR";
    case ErrorCode::SHORT_READ: return "SHORT_READ";
    case ErrorCode::DIRECT_IO_FALLBACK: return "DIRECT_IO_FALLBACK";
    case ErrorCode::ALLOCATION_FAILED: return "ALLOCATION_FAILED";
    case ErrorCode::BUDGET_TOO_SMALL: return "BUDGET_TOO_SMALL";
    case ErrorCode::INVALID_EXPERT_ID: return "INVALID_EXPERT_ID";
    case ErrorCode::EXPERT_CHECKSUM_MISMATCH: return "EXPERT_CHECKSUM_MISMATCH";
    case ErrorCode::EXPERT_EXECUTION_MISMATCH: return "EXPERT_EXECUTION_MISMATCH";
    case ErrorCode::SLOTS_BUSY: return "SLOTS_BUSY";
    case ErrorCode::STALE_COMPLETION: return "STALE_COMPLETION";
    case ErrorCode::TRANSFER_FAILED: return "TRANSFER_FAILED";
    case ErrorCode::FENCE_TIMEOUT: return "FENCE_TIMEOUT";
    case ErrorCode::DEVICE_LOSS: return "DEVICE_LOSS";
    case ErrorCode::GPU_BACKEND_UNIMPLEMENTED: return "GPU_BACKEND_UNIMPLEMENTED";
    case ErrorCode::CPU_FALLBACK_DISABLED: return "CPU_FALLBACK_DISABLED";
    case ErrorCode::INVALID_ARGUMENT: return "INVALID_ARGUMENT";
    case ErrorCode::NOT_IMPLEMENTED: return "NOT_IMPLEMENTED";
    case ErrorCode::INTERNAL_ERROR: return "INTERNAL_ERROR";
    }
    return "UNKNOWN";
}

int error_exit_code(ErrorCode code) {
    switch (code) {
    case ErrorCode::OK: return 0;
    case ErrorCode::MALFORMED_GGUF: return 10;
    case ErrorCode::UNSUPPORTED_ARCHITECTURE:
    case ErrorCode::UNSUPPORTED_QUANTISATION:
    case ErrorCode::INVALID_TOKENIZER:
    case ErrorCode::METADATA_MISMATCH: return 11;
    case ErrorCode::MISSING_REQUIRED_TENSOR:
    case ErrorCode::UNKNOWN_TENSOR:
    case ErrorCode::DUPLICATE_TENSOR:
    case ErrorCode::SHAPE_MISMATCH: return 12;
    case ErrorCode::SOURCE_CHANGED: return 13;
    case ErrorCode::IO_ERROR:
    case ErrorCode::SHORT_READ:
    case ErrorCode::DIRECT_IO_FALLBACK: return 14;
    case ErrorCode::ALLOCATION_FAILED:
    case ErrorCode::BUDGET_TOO_SMALL: return 15;
    case ErrorCode::INVALID_EXPERT_ID:
    case ErrorCode::EXPERT_CHECKSUM_MISMATCH:
    case ErrorCode::EXPERT_EXECUTION_MISMATCH: return 16;
    case ErrorCode::SLOTS_BUSY: return 17;
    case ErrorCode::STALE_COMPLETION: return 18;
    case ErrorCode::TRANSFER_FAILED:
    case ErrorCode::FENCE_TIMEOUT:
    case ErrorCode::DEVICE_LOSS: return 19;
    case ErrorCode::GPU_BACKEND_UNIMPLEMENTED:
    case ErrorCode::CPU_FALLBACK_DISABLED: return 20;
    case ErrorCode::INVALID_ARGUMENT: return 2;
    case ErrorCode::NOT_IMPLEMENTED: return 21;
    case ErrorCode::INTERNAL_ERROR: return 70;
    }
    return 70;
}

FileIdentity current_identity(const std::string &path) {
    struct stat st {};
    if (::stat(path.c_str(), &st) != 0)
        fail(ErrorCode::IO_ERROR, "stat failed for " + path + ": " + std::strerror(errno));
    FileIdentity id;
    id.size = static_cast<uint64_t>(st.st_size);
    id.dev = static_cast<uint64_t>(st.st_dev);
    id.ino = static_cast<uint64_t>(st.st_ino);
#if defined(__linux__)
    id.mtime_ns = static_cast<int64_t>(st.st_mtim.tv_sec) * 1000000000ll + st.st_mtim.tv_nsec;
    id.ctime_ns = static_cast<int64_t>(st.st_ctim.tv_sec) * 1000000000ll + st.st_ctim.tv_nsec;
#else
    id.mtime_ns = static_cast<int64_t>(st.st_mtime) * 1000000000ll;
    id.ctime_ns = static_cast<int64_t>(st.st_ctime) * 1000000000ll;
#endif
    std::ifstream in(path, std::ios::binary);
    if (in && id.size > 0) {
        uint64_t probe = 0xCBF29CE484222325ull;
        auto mix = [&](const uint8_t *data, size_t count) {
            for (size_t i = 0; i < count; ++i) {
                probe ^= data[i];
                probe *= 0x100000001B3ull;
            }
        };
        uint64_t len = std::min<uint64_t>(64, id.size);
        uint64_t middle = id.size > 64 ? id.size / 2 - std::min<uint64_t>(32, id.size / 2) : 0;
        uint64_t last = id.size > len ? id.size - len : 0;
        std::vector<uint8_t> buffer(static_cast<size_t>(len));
        const uint64_t offsets[3] = {0, middle, last};
        for (uint64_t offset : offsets) {
            in.clear();
            in.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
            if (!in) break;
            in.read(reinterpret_cast<char *>(buffer.data()), static_cast<std::streamsize>(len));
            std::streamsize got = in.gcount();
            if (got > 0) mix(buffer.data(), static_cast<size_t>(got));
        }
        id.content_probe = probe;
    }
    return id;
}

RssSnapshot process_snapshot() {
    rusage usage {};
    if (getrusage(RUSAGE_SELF, &usage) != 0) return {};
    RssSnapshot s;
    s.max_rss_kib = static_cast<uint64_t>(usage.ru_maxrss);
    s.minor_faults = static_cast<uint64_t>(usage.ru_minflt);
    s.major_faults = static_cast<uint64_t>(usage.ru_majflt);
    s.user_cpu_s = usage.ru_utime.tv_sec + usage.ru_utime.tv_usec / 1e6;
    s.system_cpu_s = usage.ru_stime.tv_sec + usage.ru_stime.tv_usec / 1e6;
    return s;
}

std::vector<uint8_t> read_binary_file(const std::string &path, uint64_t max_bytes) {
    FileIdentity id = current_identity(path);
    if (id.size > max_bytes) fail(ErrorCode::INVALID_ARGUMENT, "file exceeds parser cap: " + path);
    std::ifstream in(path, std::ios::binary);
    if (!in) fail(ErrorCode::IO_ERROR, "cannot open " + path);
    std::vector<uint8_t> data(static_cast<size_t>(id.size));
    if (!data.empty()) in.read(reinterpret_cast<char *>(data.data()), static_cast<std::streamsize>(data.size()));
    if (in.gcount() != static_cast<std::streamsize>(data.size()))
        fail(ErrorCode::IO_ERROR, "short read from " + path);
    return data;
}

void write_text_file(const std::string &path, const std::string &text) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) fail(ErrorCode::IO_ERROR, "cannot write " + path);
    out.write(text.data(), static_cast<std::streamsize>(text.size()));
    if (!out) fail(ErrorCode::IO_ERROR, "write failed for " + path);
}

std::string read_text_file(const std::string &path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) fail(ErrorCode::IO_ERROR, "cannot open " + path);
    std::ostringstream ss;
    ss << in.rdbuf();
    return ss.str();
}

std::string strip_eol(const std::string &s) {
    size_t end = s.size();
    while (end > 0 && (s[end - 1] == '\n' || s[end - 1] == '\r')) --end;
    return s.substr(0, end);
}

std::string trim(const std::string &s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

} // namespace fm
