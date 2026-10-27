#include "fm/gguf.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <set>
#include <sstream>
#include <sys/mman.h>
#include <unistd.h>

namespace fm {

// ---------------------------------------------------------------------------
// MappedFile
// ---------------------------------------------------------------------------

MappedFile::MappedFile(const std::string &path) : path_(path) {
    identity_ = current_identity(path);
    if (identity_.size == 0) fail(ErrorCode::MALFORMED_GGUF, "empty GGUF file: " + path);
    if (identity_.size > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
        fail(ErrorCode::MALFORMED_GGUF, "file too large to map: " + path);
#ifdef _WIN32
    fallback_ = read_binary_file(path);
    data_ = fallback_.data();
    size_ = fallback_.size();
#else
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) fail(ErrorCode::IO_ERROR, "cannot open " + path + ": " + std::strerror(errno));
    void *p = ::mmap(nullptr, static_cast<size_t>(identity_.size), PROT_READ, MAP_PRIVATE, fd, 0);
    int saved = errno;
    ::close(fd);
    if (p == MAP_FAILED) fail(ErrorCode::IO_ERROR, "mmap failed for " + path + ": " + std::strerror(saved));
    mapping_ = p;
    data_ = static_cast<const uint8_t *>(p);
    size_ = static_cast<size_t>(identity_.size);
#ifdef MADV_RANDOM
    ::madvise(p, size_, MADV_RANDOM);
#endif
#endif
}

MappedFile::~MappedFile() {
#ifndef _WIN32
    if (mapping_) ::munmap(mapping_, size_);
#endif
}

// ---------------------------------------------------------------------------
// Quantisation geometry
// ---------------------------------------------------------------------------

const std::map<uint32_t, QuantGeometry> &quantisation_table() {
    static const std::map<uint32_t, QuantGeometry> table = {
        {0,  {1, 4, "F32"}},
        {1,  {1, 2, "F16"}},
        {2,  {32, 18, "Q4_0"}},
        {3,  {32, 20, "Q4_1"}},
        {6,  {32, 22, "Q5_0"}},
        {7,  {32, 24, "Q5_1"}},
        {8,  {32, 34, "Q8_0"}},
        {9,  {32, 40, "Q8_1"}},
        {10, {256, 84, "Q2_K"}},
        {11, {256, 110, "Q3_K"}},
        {12, {256, 144, "Q4_K"}},
        {13, {256, 176, "Q5_K"}},
        {14, {256, 210, "Q6_K"}},
        {15, {256, 292, "Q8_K"}},
        {30, {1, 2, "BF16"}},
    };
    return table;
}

const char *tensor_role_name(TensorRole role) {
    switch (role) {
    case TensorRole::UNKNOWN: return "UNKNOWN";
    case TensorRole::TOKEN_EMBD: return "TOKEN_EMBD";
    case TensorRole::OUTPUT: return "OUTPUT";
    case TensorRole::OUTPUT_NORM: return "OUTPUT_NORM";
    case TensorRole::ATTN_NORM: return "ATTN_NORM";
    case TensorRole::ATTN_Q: return "ATTN_Q";
    case TensorRole::ATTN_K: return "ATTN_K";
    case TensorRole::ATTN_V: return "ATTN_V";
    case TensorRole::ATTN_OUTPUT: return "ATTN_OUTPUT";
    case TensorRole::FFN_NORM: return "FFN_NORM";
    case TensorRole::FFN_GATE: return "FFN_GATE";
    case TensorRole::FFN_UP: return "FFN_UP";
    case TensorRole::FFN_DOWN: return "FFN_DOWN";
    case TensorRole::FFN_ROUTER: return "FFN_ROUTER";
    case TensorRole::FFN_GATE_EXPS: return "FFN_GATE_EXPS";
    case TensorRole::FFN_UP_EXPS: return "FFN_UP_EXPS";
    case TensorRole::FFN_DOWN_EXPS: return "FFN_DOWN_EXPS";
    }
    return "UNKNOWN";
}

// ---------------------------------------------------------------------------
// Metadata values
// ---------------------------------------------------------------------------

uint64_t MetadataValue::as_u64() const {
    switch (type) {
    case GGUFValueType::UINT8:
    case GGUFValueType::UINT16:
    case GGUFValueType::UINT32:
    case GGUFValueType::UINT64:
    case GGUFValueType::BOOL: return u;
    case GGUFValueType::INT8:
    case GGUFValueType::INT16:
    case GGUFValueType::INT32:
    case GGUFValueType::INT64:
        if (i < 0) fail(ErrorCode::MALFORMED_GGUF, "negative metadata value used as unsigned");
        return static_cast<uint64_t>(i);
    default:
        fail(ErrorCode::MALFORMED_GGUF, "metadata value is not an integer");
    }
}

int64_t MetadataValue::as_i64() const {
    if (type == GGUFValueType::INT8 || type == GGUFValueType::INT16 ||
        type == GGUFValueType::INT32 || type == GGUFValueType::INT64) return i;
    return static_cast<int64_t>(as_u64());
}

double MetadataValue::as_double() const {
    switch (type) {
    case GGUFValueType::FLOAT32:
    case GGUFValueType::FLOAT64: return f;
    case GGUFValueType::UINT8:
    case GGUFValueType::UINT16:
    case GGUFValueType::UINT32:
    case GGUFValueType::UINT64: return static_cast<double>(u);
    case GGUFValueType::INT8:
    case GGUFValueType::INT16:
    case GGUFValueType::INT32:
    case GGUFValueType::INT64: return static_cast<double>(i);
    default: fail(ErrorCode::MALFORMED_GGUF, "metadata value is not numeric");
    }
}

bool MetadataValue::as_bool() const {
    if (type == GGUFValueType::BOOL) return b;
    if (type == GGUFValueType::UINT8) return u != 0;
    fail(ErrorCode::MALFORMED_GGUF, "metadata value is not a bool");
}

const std::string &MetadataValue::as_string() const {
    if (type != GGUFValueType::STRING)
        fail(ErrorCode::MALFORMED_GGUF, "metadata value is not a string");
    return s;
}

std::string MetadataValue::display() const {
    std::ostringstream os;
    switch (type) {
    case GGUFValueType::UINT8:
    case GGUFValueType::UINT16:
    case GGUFValueType::UINT32:
    case GGUFValueType::UINT64: os << u; break;
    case GGUFValueType::INT8:
    case GGUFValueType::INT16:
    case GGUFValueType::INT32:
    case GGUFValueType::INT64: os << i; break;
    case GGUFValueType::FLOAT32:
    case GGUFValueType::FLOAT64: os << std::setprecision(6) << f; break;
    case GGUFValueType::BOOL: os << (b ? "true" : "false"); break;
    case GGUFValueType::STRING: os << s; break;
    case GGUFValueType::ARRAY: os << "array[" << array.size() << "]"; break;
    }
    return os.str();
}

// ---------------------------------------------------------------------------
// Bounded GGUF parser
// ---------------------------------------------------------------------------

namespace {

struct Cursor {
    const uint8_t *data;
    size_t size;
    size_t pos = 0;

    size_t remaining() const { return size - pos; }
    void require(uint64_t n, const std::string &what) const {
        if (n > remaining())
            fail(ErrorCode::MALFORMED_GGUF, "truncated " + what + " at byte " + std::to_string(pos));
    }
    uint8_t u8(const std::string &what) {
        require(1, what); return data[pos++];
    }
    uint16_t u16(const std::string &what) {
        require(2, what);
        uint16_t v = static_cast<uint16_t>(data[pos]) |
                     (static_cast<uint16_t>(data[pos + 1]) << 8);
        pos += 2; return v;
    }
    uint32_t u32(const std::string &what) {
        require(4, what);
        uint32_t v = 0;
        for (int i = 0; i < 4; ++i) v |= static_cast<uint32_t>(data[pos + i]) << (8 * i);
        pos += 4; return v;
    }
    uint64_t u64(const std::string &what) {
        require(8, what);
        uint64_t v = 0;
        for (int i = 0; i < 8; ++i) v |= static_cast<uint64_t>(data[pos + i]) << (8 * i);
        pos += 8; return v;
    }
    float f32(const std::string &what) {
        uint32_t bits = u32(what);
        float v;
        std::memcpy(&v, &bits, 4);
        return v;
    }
    double f64(const std::string &what) {
        uint64_t bits = u64(what);
        double v;
        std::memcpy(&v, &bits, 8);
        return v;
    }
    std::string str(const std::string &what) {
        uint64_t n = u64(what + " length");
        if (n > remaining())
            fail(ErrorCode::MALFORMED_GGUF, "oversized " + what + " length " + std::to_string(n));
        std::string out(reinterpret_cast<const char *>(data + pos), static_cast<size_t>(n));
        pos += static_cast<size_t>(n);
        return out;
    }
};

uint64_t value_min_size(uint32_t type) {
    switch (static_cast<GGUFValueType>(type)) {
    case GGUFValueType::UINT8:
    case GGUFValueType::INT8:
    case GGUFValueType::BOOL: return 1;
    case GGUFValueType::UINT16:
    case GGUFValueType::INT16: return 2;
    case GGUFValueType::UINT32:
    case GGUFValueType::INT32:
    case GGUFValueType::FLOAT32: return 4;
    case GGUFValueType::UINT64:
    case GGUFValueType::INT64:
    case GGUFValueType::FLOAT64: return 8;
    case GGUFValueType::STRING: return 8;
    case GGUFValueType::ARRAY: return 12;
    default: return 0;
    }
}

MetadataValue parse_value(Cursor &c, uint32_t raw_type, uint32_t depth) {
    if (depth > 4) fail(ErrorCode::MALFORMED_GGUF, "GGUF array nesting exceeds 4");
    MetadataValue v;
    v.type = static_cast<GGUFValueType>(raw_type);
    switch (v.type) {
    case GGUFValueType::UINT8: v.u = c.u8("uint8"); break;
    case GGUFValueType::INT8: v.i = static_cast<int8_t>(c.u8("int8")); break;
    case GGUFValueType::UINT16: v.u = c.u16("uint16"); break;
    case GGUFValueType::INT16: v.i = static_cast<int16_t>(c.u16("int16")); break;
    case GGUFValueType::UINT32: v.u = c.u32("uint32"); break;
    case GGUFValueType::INT32: v.i = static_cast<int32_t>(c.u32("int32")); break;
    case GGUFValueType::FLOAT32: v.f = c.f32("float32"); break;
    case GGUFValueType::BOOL: {
        uint8_t raw = c.u8("bool");
        v.b = raw != 0; v.u = raw; break;
    }
    case GGUFValueType::STRING: v.s = c.str("string"); break;
    case GGUFValueType::UINT64: v.u = c.u64("uint64"); break;
    case GGUFValueType::INT64: v.i = static_cast<int64_t>(c.u64("int64")); break;
    case GGUFValueType::FLOAT64: v.f = c.f64("float64"); break;
    case GGUFValueType::ARRAY: {
        uint32_t element = c.u32("array element type");
        uint64_t count = c.u64("array count");
        uint64_t min_size = value_min_size(element);
        if (min_size == 0)
            fail(ErrorCode::MALFORMED_GGUF, "invalid GGUF array element type " + std::to_string(element));
        if (count > (1u << 20) || count > c.remaining() / min_size)
            fail(ErrorCode::MALFORMED_GGUF, "oversized GGUF array count " + std::to_string(count));
        v.element_type = static_cast<GGUFValueType>(element);
        v.array.reserve(static_cast<size_t>(count));
        for (uint64_t i = 0; i < count; ++i)
            v.array.push_back(parse_value(c, element, depth + 1));
        break;
    }
    default:
        fail(ErrorCode::MALFORMED_GGUF, "invalid GGUF value type " + std::to_string(raw_type));
    }
    return v;
}

bool parse_layer_suffix(const std::string &name, const std::string &suffix, uint32_t &layer) {
    const std::string prefix = "blk.";
    if (name.rfind(prefix, 0) != 0) return false;
    size_t dot = name.find('.', prefix.size());
    if (dot == std::string::npos) return false;
    std::string digits = name.substr(prefix.size(), dot - prefix.size());
    if (digits.empty() || digits.size() > 9) return false;
    for (char ch : digits) if (ch < '0' || ch > '9') return false;
    if (name.substr(dot + 1) != suffix) return false;
    layer = static_cast<uint32_t>(std::stoul(digits));
    return true;
}

TensorRole role_from_name(const std::string &name, uint32_t &layer) {
    layer = 0;
    if (name == "token_embd.weight") return TensorRole::TOKEN_EMBD;
    if (name == "output.weight") return TensorRole::OUTPUT;
    if (name == "output_norm.weight") return TensorRole::OUTPUT_NORM;
    static const std::vector<std::pair<std::string, TensorRole>> suffixes = {
        {"attn_norm.weight", TensorRole::ATTN_NORM},
        {"attn_q.weight", TensorRole::ATTN_Q},
        {"attn_k.weight", TensorRole::ATTN_K},
        {"attn_v.weight", TensorRole::ATTN_V},
        {"attn_output.weight", TensorRole::ATTN_OUTPUT},
        {"ffn_norm.weight", TensorRole::FFN_NORM},
        {"ffn_gate.weight", TensorRole::FFN_GATE},
        {"ffn_up.weight", TensorRole::FFN_UP},
        {"ffn_down.weight", TensorRole::FFN_DOWN},
        {"ffn_router.weight", TensorRole::FFN_ROUTER},
        {"ffn_gate_exps.weight", TensorRole::FFN_GATE_EXPS},
        {"ffn_up_exps.weight", TensorRole::FFN_UP_EXPS},
        {"ffn_down_exps.weight", TensorRole::FFN_DOWN_EXPS},
    };
    for (const auto &entry : suffixes)
        if (parse_layer_suffix(name, entry.first, layer)) return entry.second;
    return TensorRole::UNKNOWN;
}

} // namespace

GGUFDocument parse_gguf(const uint8_t *data, size_t size) {
    if (!data || size < 24)
        fail(ErrorCode::MALFORMED_GGUF, "truncated GGUF header");
    if (std::memcmp(data, "GGUF", 4) != 0)
        fail(ErrorCode::MALFORMED_GGUF, "bad GGUF magic");
    Cursor c{data, size, 0};
    c.pos = 4;
    GGUFDocument doc;
    doc.version = c.u32("GGUF version");
    if (doc.version != 3)
        fail(ErrorCode::MALFORMED_GGUF, "unsupported GGUF version " + std::to_string(doc.version));
    uint64_t tensor_count = c.u64("tensor count");
    uint64_t metadata_count = c.u64("metadata count");
    if (metadata_count > size)
        fail(ErrorCode::MALFORMED_GGUF, "unreasonable metadata count " + std::to_string(metadata_count));
    if (tensor_count > size / 32 + 1)
        fail(ErrorCode::MALFORMED_GGUF, "unreasonable tensor count " + std::to_string(tensor_count));
    doc.tensor_count = tensor_count;
    doc.metadata_count = metadata_count;
    doc.file_size = size;

    for (uint64_t i = 0; i < metadata_count; ++i) {
        std::string key = c.str("metadata key");
        uint32_t type = c.u32("metadata value type");
        MetadataValue value = parse_value(c, type, 0);
        if (std::getenv("FM_GGUF_DEBUG"))
            std::fprintf(stderr, "meta[%llu] %s type=%u pos=%zu\n",
                         static_cast<unsigned long long>(i), key.c_str(), type, c.pos);
        if (doc.metadata.count(key))
            fail(ErrorCode::MALFORMED_GGUF, "duplicate metadata key " + key);
        doc.metadata.emplace(std::move(key), std::move(value));
    }

    auto alignment_it = doc.metadata.find("general.alignment");
    uint64_t alignment = alignment_it == doc.metadata.end() ? 32 : alignment_it->second.as_u64();
    if (alignment == 0 || alignment > (1u << 20) || (alignment & (alignment - 1)) != 0)
        fail(ErrorCode::MALFORMED_GGUF, "invalid GGUF alignment " + std::to_string(alignment));
    doc.alignment = static_cast<uint32_t>(alignment);
    // The data section begins after BOTH metadata and the tensor directory.
    // Absolute tensor offsets are resolved after the directory is parsed.

    const auto &qtable = quantisation_table();
    std::vector<std::pair<uint64_t, uint64_t>> extents;
    for (uint64_t i = 0; i < tensor_count; ++i) {
        TensorInfo t;
        t.name = c.str("tensor name");
        uint32_t n_dims = c.u32("tensor dimension count");
        if (n_dims == 0 || n_dims > 4)
            fail(ErrorCode::MALFORMED_GGUF, "invalid dimension count for " + t.name);
        t.dims.reserve(n_dims);
        uint64_t elements = 1;
        for (uint32_t d = 0; d < n_dims; ++d) {
            uint64_t dim = c.u64("tensor dimension");
            if (dim == 0 || dim > (1ull << 40))
                fail(ErrorCode::MALFORMED_GGUF, "invalid dimension in " + t.name);
            elements = checked_mul(elements, dim, "tensor " + t.name + " element count");
            if (elements > (1ull << 62))
                fail(ErrorCode::MALFORMED_GGUF, "dimension overflow in " + t.name);
            t.dims.push_back(dim);
        }
        t.type_id = c.u32("tensor quantisation type");
        auto qit = qtable.find(t.type_id);
        if (qit == qtable.end())
            fail(ErrorCode::MALFORMED_GGUF, "unknown quantisation type " + std::to_string(t.type_id) + " in " + t.name);
        t.block_size = qit->second.block_size;
        t.block_bytes = qit->second.block_bytes;
        t.type_name = qit->second.name;
        if (t.dims[0] % t.block_size != 0)
            fail(ErrorCode::MALFORMED_GGUF, "ne0 is not a whole number of blocks in " + t.name);
        uint64_t blocks = elements / t.block_size;
        t.byte_size = checked_mul(blocks, t.block_bytes, "tensor " + t.name + " byte count");
        t.relative_offset = c.u64("tensor offset");
        if (t.relative_offset % alignment != 0)
            fail(ErrorCode::MALFORMED_GGUF, "misaligned tensor offset for " + t.name);
        if (doc.tensor_by_name.count(t.name))
            fail(ErrorCode::MALFORMED_GGUF, "duplicate tensor name " + t.name);
        if (std::getenv("FM_GGUF_DEBUG"))
            std::fprintf(stderr, "tensor[%llu] %s pos=%zu\n",
                         static_cast<unsigned long long>(i), t.name.c_str(), c.pos);
        doc.tensor_by_name.emplace(t.name, doc.tensors.size());
        doc.tensors.push_back(std::move(t));
    }
    doc.data_offset = align_up(c.pos, alignment);
    if (doc.data_offset > size)
        fail(ErrorCode::MALFORMED_GGUF, "data section starts beyond file end");
    for (auto &tensor : doc.tensors) {
        tensor.absolute_offset = checked_add(doc.data_offset, tensor.relative_offset,
                                             "tensor " + tensor.name + " offset");
        uint64_t end = checked_add(tensor.absolute_offset, tensor.byte_size,
                                   "tensor " + tensor.name + " end");
        if (end > size)
            fail(ErrorCode::MALFORMED_GGUF, "tensor out of bounds: " + tensor.name);
        extents.push_back({tensor.absolute_offset, end});
    }
    std::sort(extents.begin(), extents.end());
    for (size_t i = 1; i < extents.size(); ++i) {
        if (extents[i].first < extents[i - 1].second)
            fail(ErrorCode::MALFORMED_GGUF, "overlapping tensor extents");
    }
    doc.source_fingerprint = fnv1a64(data, doc.data_offset);
    return doc;
}

GGUFDocument parse_gguf_file(const std::string &path) {
    MappedFile file(path);
    return parse_gguf(file.data(), file.size());
}

// ---------------------------------------------------------------------------
// Architecture adapter and canonical model map
// ---------------------------------------------------------------------------

namespace {

const MetadataValue &require_metadata(const GGUFDocument &doc, const std::string &key) {
    auto it = doc.metadata.find(key);
    if (it == doc.metadata.end())
        fail(ErrorCode::METADATA_MISMATCH, "missing required metadata key " + key);
    return it->second;
}

std::string require_string(const GGUFDocument &doc, const std::string &key) {
    return require_metadata(doc, key).as_string();
}

uint64_t require_u64(const GGUFDocument &doc, const std::string &key) {
    return require_metadata(doc, key).as_u64();
}

double require_double(const GGUFDocument &doc, const std::string &key) {
    return require_metadata(doc, key).as_double();
}

bool require_bool(const GGUFDocument &doc, const std::string &key) {
    return require_metadata(doc, key).as_bool();
}

void expect_dims(const TensorInfo &t, std::initializer_list<uint64_t> dims) {
    if (t.dims.size() != dims.size())
        fail(ErrorCode::SHAPE_MISMATCH, "tensor " + t.name + " has wrong rank");
    size_t i = 0;
    for (uint64_t d : dims) {
        if (t.dims[i++] != d)
            fail(ErrorCode::SHAPE_MISMATCH, "tensor " + t.name + " has wrong shape");
    }
}

std::string layer_name(uint32_t layer, const std::string &suffix) {
    return "blk." + std::to_string(layer) + "." + suffix;
}

} // namespace

ModelManifest build_manifest(const GGUFDocument &doc, const std::string &source_path) {
    ModelManifest m;
    m.doc = doc;
    m.source_path = source_path;
    ModelConfig &cfg = m.config;
    cfg.architecture = require_string(doc, "general.architecture");
    cfg.adapter = require_string(doc, "freemoe.adapter");
    cfg.model_kind = require_string(doc, "freemoe.model_kind");
    cfg.rope_mode = require_string(doc, "freemoe.rope_mode");
    cfg.tokenizer_model = require_string(doc, "freemoe.tokenizer_model");
    cfg.chat_template = require_string(doc, "tokenizer.chat_template");
    cfg.router_operation = require_string(doc, "freemoe.router_operation");
    cfg.expert_layout = require_string(doc, "freemoe.expert_layout");
    cfg.hidden = static_cast<uint32_t>(require_u64(doc, "freemoe.hidden_size"));
    cfg.intermediate = static_cast<uint32_t>(require_u64(doc, "freemoe.intermediate_size"));
    cfg.layers = static_cast<uint32_t>(require_u64(doc, "freemoe.num_hidden_layers"));
    cfg.heads = static_cast<uint32_t>(require_u64(doc, "freemoe.num_attention_heads"));
    cfg.kv_heads = static_cast<uint32_t>(require_u64(doc, "freemoe.num_key_value_heads"));
    cfg.head_dim = static_cast<uint32_t>(require_u64(doc, "freemoe.head_dim"));
    cfg.vocab = static_cast<uint32_t>(require_u64(doc, "freemoe.vocab_size"));
    cfg.context = static_cast<uint32_t>(require_u64(doc, "freemoe.max_context_length"));
    cfg.expert_count = static_cast<uint32_t>(require_u64(doc, "freemoe.expert_count"));
    cfg.expert_top_k = static_cast<uint32_t>(require_u64(doc, "freemoe.expert_top_k"));
    cfg.bos_id = static_cast<uint32_t>(require_u64(doc, "tokenizer.ggml.bos_token_id"));
    cfg.eos_id = static_cast<uint32_t>(require_u64(doc, "tokenizer.ggml.eos_token_id"));
    cfg.rms_eps = require_double(doc, "freemoe.rms_norm_epsilon");
    cfg.rope_theta = require_double(doc, "freemoe.rope_theta");
    cfg.shared_expert = require_bool(doc, "freemoe.shared_expert");

    if (cfg.architecture != "freemoe-tiny" || cfg.adapter != "freemoe-tiny-v1")
        fail(ErrorCode::UNSUPPORTED_ARCHITECTURE, "unsupported architecture/adapter: " + cfg.architecture + "/" + cfg.adapter);
    if (cfg.model_kind != "dense" && cfg.model_kind != "moe")
        fail(ErrorCode::METADATA_MISMATCH, "unknown model kind " + cfg.model_kind);
    if (cfg.rope_mode != "half" || cfg.chat_template != "raw-v1" ||
        cfg.tokenizer_model != "freemoe.byte-v1" ||
        require_string(doc, "tokenizer.ggml.model") != "freemoe.byte-v1")
        fail(ErrorCode::INVALID_TOKENIZER, "unsupported tokenizer/chat template metadata");
    if (cfg.router_operation != "linear-topk-softmax-v1" || cfg.shared_expert)
        fail(ErrorCode::METADATA_MISMATCH, "unsupported router/shared-expert semantics");
    if (cfg.heads == 0 || cfg.head_dim == 0 ||
        static_cast<uint64_t>(cfg.heads) * cfg.head_dim != cfg.hidden ||
        cfg.kv_heads != cfg.heads || (cfg.head_dim % 2) != 0 ||
        cfg.hidden == 0 || cfg.intermediate == 0 || cfg.layers == 0 ||
        cfg.vocab == 0 || cfg.context == 0 || cfg.rms_eps <= 0.0 || cfg.rope_theta <= 0.0)
        fail(ErrorCode::SHAPE_MISMATCH, "invalid model dimensions");
    // Bound every value that drives a loop or allocation. Fuzzed metadata can
    // otherwise request billions of layers/experts and turn validation into an
    // apparent hang before any tensor is read.
    if (cfg.hidden > 65536 || cfg.intermediate > (1u << 20) || cfg.layers > 1024 ||
        cfg.heads > 4096 || cfg.head_dim > 1024 || cfg.vocab > (1u << 20) ||
        cfg.context > (1u << 20) || cfg.expert_count > 4096 ||
        cfg.expert_top_k > cfg.expert_count || cfg.bos_id >= cfg.vocab ||
        cfg.eos_id >= cfg.vocab)
        fail(ErrorCode::SHAPE_MISMATCH, "model dimension exceeds adapter safety bounds");

    const MetadataValue &tokens = require_metadata(doc, "tokenizer.ggml.tokens");
    if (!tokens.is_array() || tokens.array.size() != cfg.vocab ||
        tokens.element_type != GGUFValueType::STRING)
        fail(ErrorCode::INVALID_TOKENIZER, "tokenizer token table has wrong shape");
    for (const auto &token : tokens.array) {
        if (!token.is_string()) fail(ErrorCode::INVALID_TOKENIZER, "non-string tokenizer token");
        m.tokenizer_tokens.push_back(token.s);
    }

    const MetadataValue &checksums = require_metadata(doc, "freemoe.expert_checksums");
    if (!checksums.is_array() || checksums.element_type != GGUFValueType::UINT64)
        fail(ErrorCode::METADATA_MISMATCH, "expert checksum array has wrong type");

    bool moe = cfg.model_kind == "moe";
    if (!moe && (cfg.expert_count != 0 || cfg.expert_top_k != 0 || !checksums.array.empty() || cfg.expert_layout != "none"))
        fail(ErrorCode::METADATA_MISMATCH, "dense model declares MoE metadata");
    if (moe && (cfg.expert_count == 0 || cfg.expert_top_k == 0 ||
                cfg.expert_top_k > cfg.expert_count || cfg.expert_layout != "stacked-f32"))
        fail(ErrorCode::METADATA_MISMATCH, "MoE model has inconsistent expert metadata");
    if (moe && checksums.array.size() != static_cast<size_t>(cfg.layers) * cfg.expert_count)
        fail(ErrorCode::METADATA_MISMATCH, "expert checksum count mismatch");

    std::set<std::string> expected = {"token_embd.weight", "output.weight", "output_norm.weight"};
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        for (const char *suffix : {"attn_norm.weight", "attn_q.weight", "attn_k.weight",
                                   "attn_v.weight", "attn_output.weight", "ffn_norm.weight"})
            expected.insert(layer_name(l, suffix));
        if (moe) {
            for (const char *suffix : {"ffn_router.weight", "ffn_gate_exps.weight",
                                       "ffn_up_exps.weight", "ffn_down_exps.weight"})
                expected.insert(layer_name(l, suffix));
        } else {
            for (const char *suffix : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"})
                expected.insert(layer_name(l, suffix));
        }
    }

    for (auto &tensor : m.doc.tensors) {
        if (!expected.count(tensor.name))
            fail(ErrorCode::UNKNOWN_TENSOR, "unknown tensor " + tensor.name);
        uint32_t layer = 0;
        tensor.role = role_from_name(tensor.name, layer);
        bool global_role = tensor.role == TensorRole::TOKEN_EMBD ||
                           tensor.role == TensorRole::OUTPUT ||
                           tensor.role == TensorRole::OUTPUT_NORM;
        tensor.layer = global_role ? -1 : static_cast<int>(layer);
        if (tensor.role == TensorRole::UNKNOWN)
            fail(ErrorCode::UNKNOWN_TENSOR, "cannot map tensor role for " + tensor.name);
        if (tensor.type_id != 0)
            fail(ErrorCode::UNSUPPORTED_QUANTISATION, "adapter supports F32 only: " + tensor.name);
        tensor.expert_tensor = tensor.role == TensorRole::FFN_GATE_EXPS ||
                               tensor.role == TensorRole::FFN_UP_EXPS ||
                               tensor.role == TensorRole::FFN_DOWN_EXPS;
        if (tensor.expert_tensor) m.expert_tensor_bytes += tensor.byte_size;
        else m.fixed_weight_bytes += tensor.byte_size;
    }
    for (const auto &name : expected) {
        auto it = m.doc.tensor_by_name.find(name);
        if (it == m.doc.tensor_by_name.end())
            fail(ErrorCode::MISSING_REQUIRED_TENSOR, "missing required tensor " + name);
    }

    auto t = [&](const std::string &name) -> const TensorInfo & {
        return m.doc.tensors[m.doc.tensor_by_name.at(name)];
    };
    expect_dims(t("token_embd.weight"), {cfg.hidden, cfg.vocab});
    expect_dims(t("output.weight"), {cfg.hidden, cfg.vocab});
    expect_dims(t("output_norm.weight"), {cfg.hidden});
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        expect_dims(t(layer_name(l, "attn_norm.weight")), {cfg.hidden});
        expect_dims(t(layer_name(l, "ffn_norm.weight")), {cfg.hidden});
        for (const char *suffix : {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_output.weight"})
            expect_dims(t(layer_name(l, suffix)), {cfg.hidden, cfg.hidden});
        if (moe) {
            expect_dims(t(layer_name(l, "ffn_router.weight")), {cfg.hidden, cfg.expert_count});
            expect_dims(t(layer_name(l, "ffn_gate_exps.weight")), {cfg.hidden, cfg.intermediate, cfg.expert_count});
            expect_dims(t(layer_name(l, "ffn_up_exps.weight")), {cfg.hidden, cfg.intermediate, cfg.expert_count});
            expect_dims(t(layer_name(l, "ffn_down_exps.weight")), {cfg.intermediate, cfg.hidden, cfg.expert_count});
        } else {
            expect_dims(t(layer_name(l, "ffn_gate.weight")), {cfg.hidden, cfg.intermediate});
            expect_dims(t(layer_name(l, "ffn_up.weight")), {cfg.hidden, cfg.intermediate});
            expect_dims(t(layer_name(l, "ffn_down.weight")), {cfg.intermediate, cfg.hidden});
        }
    }

    if (moe) {
        static const TensorRole extent_roles[3] = {
            TensorRole::FFN_GATE_EXPS, TensorRole::FFN_UP_EXPS, TensorRole::FFN_DOWN_EXPS};
        static const char *extent_suffixes[3] = {
            "ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"};
        for (uint32_t l = 0; l < cfg.layers; ++l) {
            uint64_t expert_stride = 0;
            for (int r = 0; r < 3; ++r) {
                const TensorInfo &tensor = t(layer_name(l, extent_suffixes[r]));
                uint64_t stride = checked_mul(tensor.dims[0], tensor.dims[1], "expert stride");
                stride = checked_mul(stride, tensor.block_bytes / tensor.block_size, "expert bytes");
                if (stride % tensor.block_bytes != 0)
                    fail(ErrorCode::SHAPE_MISMATCH, "expert slice is not whole quantisation blocks");
                if (r == 0) expert_stride = stride;
                else if (stride != expert_stride)
                    fail(ErrorCode::SHAPE_MISMATCH, "expert projections have different byte strides");
            }
            for (uint32_t e = 0; e < cfg.expert_count; ++e) {
                ExpertRecord rec;
                rec.layer = l;
                rec.expert = e;
                rec.payload_bytes = checked_mul(3, expert_stride, "expert payload");
                size_t checksum_index = static_cast<size_t>(l) * cfg.expert_count + e;
                rec.checksum = checksums.array[checksum_index].as_u64();
                for (int r = 0; r < 3; ++r) {
                    const TensorInfo &tensor = t(layer_name(l, extent_suffixes[r]));
                    uint64_t rel = checked_mul(e, expert_stride, "expert relative offset");
                    uint64_t end = checked_add(rel, expert_stride, "expert extent end");
                    if (end > tensor.byte_size)
                        fail(ErrorCode::SHAPE_MISMATCH, "expert extent exceeds stacked tensor");
                    rec.extents[r] = {extent_roles[r], tensor.absolute_offset + rel, expert_stride};
                }
                m.expert_index.emplace(ModelManifest::expert_key(l, e), m.experts.size());
                m.experts.push_back(rec);
                m.slot_bytes = std::max(m.slot_bytes, rec.payload_bytes);
            }
        }
    }
    return m;
}

Json metadata_summary_json(const ModelManifest &manifest) {
    static const std::vector<std::string> keys = {
        "general.architecture", "general.name", "general.alignment",
        "freemoe.adapter", "freemoe.model_kind", "freemoe.hidden_size",
        "freemoe.intermediate_size", "freemoe.num_hidden_layers",
        "freemoe.num_attention_heads", "freemoe.num_key_value_heads",
        "freemoe.head_dim", "freemoe.vocab_size", "freemoe.max_context_length",
        "freemoe.rope_mode", "freemoe.tokenizer_model",
        "freemoe.router_operation", "freemoe.shared_expert",
        "freemoe.expert_layout", "freemoe.expert_count",
        "freemoe.expert_top_k", "freemoe.seed", "tokenizer.chat_template",
    };
    Json out = Json::object();
    for (const auto &key : keys) {
        auto it = manifest.doc.metadata.find(key);
        if (it == manifest.doc.metadata.end())
            fail(ErrorCode::METADATA_MISMATCH, "missing summary metadata " + key);
        out.set(key, it->second.display());
    }
    return out;
}

Json inspect_json(const ModelManifest &manifest, const std::string &source_path) {
    const GGUFDocument &doc = manifest.doc;
    const ModelConfig &cfg = manifest.config;
    Json out = Json::object();
    out.set("schema", "freemoe.inspect.v1");
    out.set("status", "validated");
    out.set("source_path", source_path);
    out.set("file_size", doc.file_size);
    out.set("data_offset", doc.data_offset);
    out.set("gguf_version", static_cast<int64_t>(doc.version));
    out.set("alignment", static_cast<int64_t>(doc.alignment));
    out.set("metadata_count", doc.metadata_count);
    out.set("tensor_count", doc.tensor_count);

    uint64_t tensor_bytes = 0;
    std::set<std::string> quant_types;
    std::vector<const TensorInfo *> sorted;
    for (const auto &tensor : doc.tensors) {
        tensor_bytes = checked_add(tensor_bytes, tensor.byte_size, "total tensor bytes");
        quant_types.insert(tensor.type_name);
        sorted.push_back(&tensor);
    }
    std::sort(sorted.begin(), sorted.end(), [](const TensorInfo *a, const TensorInfo *b) {
        return a->name < b->name;
    });
    out.set("tensor_bytes", tensor_bytes);
    out.set("source_fingerprint", hex64(doc.source_fingerprint));

    Json adapter = Json::object();
    adapter.set("id", cfg.adapter);
    adapter.set("version", 1);
    out.set("architecture_adapter", std::move(adapter));
    out.set("model_kind", cfg.model_kind);
    Json quant = Json::array();
    for (const auto &q : quant_types) quant.push(Json::string(q));
    out.set("quantization_types", std::move(quant));

    Json tokenizer = Json::object();
    tokenizer.set("model", cfg.tokenizer_model);
    tokenizer.set("bos_token_id", static_cast<int64_t>(cfg.bos_id));
    tokenizer.set("eos_token_id", static_cast<int64_t>(cfg.eos_id));
    tokenizer.set("chat_template", cfg.chat_template);
    out.set("tokenizer", std::move(tokenizer));
    out.set("metadata_summary", metadata_summary_json(manifest));
    out.set("expert_count", static_cast<int64_t>(cfg.expert_count));
    out.set("expert_top_k", static_cast<int64_t>(cfg.expert_top_k));
    out.set("expert_layout", cfg.expert_layout);

    Json roles = Json::array();
    for (const TensorInfo *tensor : sorted) {
        Json role = Json::object();
        role.set("name", tensor->name);
        role.set("role", tensor_role_name(tensor->role));
        role.set("layer", static_cast<int64_t>(tensor->layer));
        role.set("expert", static_cast<int64_t>(-1));
        role.set("type", tensor->type_name);
        Json dims = Json::array();
        for (uint64_t d : tensor->dims) dims.push(Json::integer(static_cast<int64_t>(d)));
        role.set("dims", std::move(dims));
        role.set("offset", tensor->absolute_offset);
        role.set("bytes", tensor->byte_size);
        roles.push(std::move(role));
    }
    out.set("roles", std::move(roles));
    out.set("unresolved_roles", Json::array());

    Json experts = Json::array();
    for (const auto &rec : manifest.experts) {
        Json e = Json::object();
        e.set("layer", static_cast<int64_t>(rec.layer));
        e.set("expert", static_cast<int64_t>(rec.expert));
        e.set("bytes", rec.payload_bytes);
        e.set("checksum", hex64(rec.checksum));
        Json extents = Json::array();
        for (const auto &extent : rec.extents) {
            Json x = Json::object();
            x.set("role", tensor_role_name(extent.role));
            x.set("offset", extent.offset);
            x.set("bytes", extent.bytes);
            extents.push(std::move(x));
        }
        e.set("extents", std::move(extents));
        experts.push(std::move(e));
    }
    out.set("expert_records", std::move(experts));

    Json validation = Json::object();
    validation.set("errors", Json::array());
    validation.set("warnings", Json::array());
    out.set("validation", std::move(validation));
    return out;
}

} // namespace fm
