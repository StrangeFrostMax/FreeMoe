#include "fm/gguf.hpp"
#include "fm/adapter.hpp"
#include "fm/quant.hpp"
#include "fm/rope.hpp"

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
        {16, {256, 66, "IQ2_XXS"}},
        {17, {256, 74, "IQ2_XS"}},
        {18, {256, 98, "IQ3_XXS"}},
        {19, {256, 50, "IQ1_S"}},
        {20, {32, 18, "IQ4_NL"}},
        {21, {256, 110, "IQ3_S"}},
        {22, {256, 82, "IQ2_S"}},
        {23, {256, 136, "IQ4_XS"}},
        {29, {256, 56, "IQ1_M"}},
        {39, {32, 17, "MXFP4"}},
        {40, {64, 36, "NVFP4"}},
        {24, {1, 1, "I8"}},
        {25, {1, 2, "I16"}},
        {26, {1, 4, "I32"}},
        {27, {1, 8, "I64"}},
        {28, {1, 8, "F64"}},
        {30, {1, 2, "BF16"}},
        // Ternary family. Sizes follow the published block layouts:
        //   34 TQ1_0  2 + 256/64 + (256 - 4*256/64)/5 = 54 bytes / 256
        //   35 TQ2_0  2 + 256/4                      = 66 bytes / 256
        //   41 Q1_0   2 + 128/8                      = 18 bytes / 128
        //   42 Q2_0   2 +  64/4                      = 18 bytes /  64
        //  142 PQ2_0  2 + 128/4                      = 34 bytes / 128
        //  143 PTQ1_0 2 + 128/64 + (128 - 4*128/64)/5 = 28 bytes / 128
        {34, {256, 54, "TQ1_0"}},
        {35, {256, 66, "TQ2_0"}},
        {41, {128, 18, "Q1_0"}},
        {42, {64, 18, "Q2_0"}},
        {142, {128, 34, "PQ2_0"}},
        {143, {128, 28, "PTQ1_0"}},
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
    case TensorRole::FFN_GATE_UP_EXPS: return "FFN_GATE_UP_EXPS";
    case TensorRole::FFN_UP_EXPS: return "FFN_UP_EXPS";
    case TensorRole::FFN_DOWN_EXPS: return "FFN_DOWN_EXPS";
    case TensorRole::ATTN_Q_BIAS: return "ATTN_Q_BIAS";
    case TensorRole::ATTN_K_BIAS: return "ATTN_K_BIAS";
    case TensorRole::ATTN_V_BIAS: return "ATTN_V_BIAS";
    case TensorRole::ATTN_OUTPUT_BIAS: return "ATTN_OUTPUT_BIAS";
    case TensorRole::ATTN_SUB_NORM: return "ATTN_SUB_NORM";
    case TensorRole::FFN_SUB_NORM: return "FFN_SUB_NORM";
    case TensorRole::PROJECTION_SCALE: return "PROJECTION_SCALE";
    case TensorRole::ATTN_Q_NORM: return "ATTN_Q_NORM";
    case TensorRole::ATTN_SINKS: return "ATTN_SINKS";
    case TensorRole::ROPE_FREQS: return "ROPE_FREQS";
    case TensorRole::ATTN_K_NORM: return "ATTN_K_NORM";
    case TensorRole::ATTN_POST_NORM: return "ATTN_POST_NORM";
    case TensorRole::FFN_POST_NORM: return "FFN_POST_NORM";
    case TensorRole::ATTN_QKV: return "ATTN_QKV";
    case TensorRole::ATTN_QKV_BIAS: return "ATTN_QKV_BIAS";
    case TensorRole::FFN_GATE_INP_SHEXP: return "FFN_GATE_INP_SHEXP";
    case TensorRole::FFN_GATE_SHEXP: return "FFN_GATE_SHEXP";
    case TensorRole::FFN_UP_SHEXP: return "FFN_UP_SHEXP";
    case TensorRole::FFN_DOWN_SHEXP: return "FFN_DOWN_SHEXP";
    case TensorRole::ATTN_NORM_BIAS: return "ATTN_NORM_BIAS";
    case TensorRole::FFN_NORM_BIAS: return "FFN_NORM_BIAS";
    case TensorRole::OUTPUT_NORM_BIAS: return "OUTPUT_NORM_BIAS";
    case TensorRole::FFN_UP_BIAS: return "FFN_UP_BIAS";
    case TensorRole::FFN_DOWN_BIAS: return "FFN_DOWN_BIAS";
    case TensorRole::FFN_GATE_BIAS: return "FFN_GATE_BIAS";
    case TensorRole::OUTPUT_BIAS: return "OUTPUT_BIAS";
    case TensorRole::ATTN_Q_A: return "ATTN_Q_A";
    case TensorRole::ATTN_Q_A_NORM: return "ATTN_Q_A_NORM";
    case TensorRole::ATTN_Q_B: return "ATTN_Q_B";
    case TensorRole::ATTN_KV_A_MQA: return "ATTN_KV_A_MQA";
    case TensorRole::ATTN_KV_A_NORM: return "ATTN_KV_A_NORM";
    case TensorRole::ATTN_K_B: return "ATTN_K_B";
    case TensorRole::ATTN_V_B: return "ATTN_V_B";
    case TensorRole::FFN_EXP_PROBS_B: return "FFN_EXP_PROBS_B";
    case TensorRole::FFN_ROUTER_BIAS: return "FFN_ROUTER_BIAS";
    case TensorRole::FFN_GATE_EXPS_BIAS: return "FFN_GATE_EXPS_BIAS";
    case TensorRole::FFN_UP_EXPS_BIAS: return "FFN_UP_EXPS_BIAS";
    case TensorRole::FFN_DOWN_EXPS_BIAS: return "FFN_DOWN_EXPS_BIAS";
    case TensorRole::FFN_NORM_ADA_DOWN: return "FFN_NORM_ADA_DOWN";
    case TensorRole::FFN_NORM_ADA_UP: return "FFN_NORM_ADA_UP";
    case TensorRole::SSM_IN: return "SSM_IN";
    case TensorRole::SSM_CONV1D: return "SSM_CONV1D";
    case TensorRole::SSM_CONV1D_BIAS: return "SSM_CONV1D_BIAS";
    case TensorRole::SSM_X: return "SSM_X";
    case TensorRole::SSM_DT: return "SSM_DT";
    case TensorRole::SSM_DT_BIAS: return "SSM_DT_BIAS";
    case TensorRole::SSM_A: return "SSM_A";
    case TensorRole::SSM_D: return "SSM_D";
    case TensorRole::SSM_OUT: return "SSM_OUT";
    case TensorRole::SSM_NORM: return "SSM_NORM";
    case TensorRole::SSM_DT_NORM: return "SSM_DT_NORM";
    case TensorRole::SSM_B_NORM: return "SSM_B_NORM";
    case TensorRole::SSM_C_NORM: return "SSM_C_NORM";
    case TensorRole::IGNORED: return "IGNORED";
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

// A tensor the file may legally carry that the main forward pass does not use.
// Returning a reason (rather than true/false) is deliberate: the probe prints
// it, so "supported" never quietly means "part of this was skipped".
const char *ignorable_reason(const std::string &name) {
    static const std::pair<const char *, const char *> prefixes[] = {
        {"nextn.", "next-token-prediction head (used only for speculative decoding)"},
        {"mtp.", "multi-token-prediction head (used only for speculative decoding)"},
    };
    // These appear inside a layer name (blk.N.nextn.eh_proj) or as a top-level
    // prefix, so the test is on the substring after the layer index.
    for (const auto &entry : prefixes) {
        if (name.find(std::string(".", 1) + entry.first) != std::string::npos ||
            name.rfind(entry.first, 0) == 0)
            return entry.second;
    }
    return nullptr;
}

// llama.cpp renamed the sandwich-normalisation tensors. Current converters write
// `post_attention_norm` / `post_ffw_norm`; older exports (and every fixture this
// repo built before the rename) say `attn_post_norm` / `ffn_post_norm`. They are
// the same tensors in the same place in the block, so the name is normalised at
// parse time and everything downstream keeps one spelling.
std::string canonical_layer_name(std::string name) {
    static const std::map<std::string, std::string> renamed = {
        {"post_attention_norm", "attn_post_norm"},
        {"post_ffw_norm", "ffn_post_norm"},
    };
    const size_t dot = name.rfind('.');
    if (dot == std::string::npos) return name;
    std::string stem = name.substr(0, dot);
    const std::string suffix = name.substr(dot + 1);
    for (const auto &entry : renamed) {
        const size_t at = stem.rfind("." + entry.first);
        if (at != std::string::npos && at + 1 + entry.first.size() == stem.size()) {
            stem.replace(at + 1, entry.first.size(), entry.second);
            return stem + "." + suffix;
        }
    }
    return name;
}

TensorRole role_from_name(const std::string &name, uint32_t &layer) {
    layer = 0;
    if (name == "token_embd.weight") return TensorRole::TOKEN_EMBD;
    if (name == "output.weight") return TensorRole::OUTPUT;
    if (name == "output.bias") return TensorRole::OUTPUT_BIAS;
    if (name == "output_norm.weight") return TensorRole::OUTPUT_NORM;
    if (name == "output_norm.bias") return TensorRole::OUTPUT_NORM_BIAS;
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
        {"ffn_gate_up_exps.weight", TensorRole::FFN_GATE_UP_EXPS},
        {"ffn_up_exps.weight", TensorRole::FFN_UP_EXPS},
        {"ffn_down_exps.weight", TensorRole::FFN_DOWN_EXPS},
        {"attn_q.bias", TensorRole::ATTN_Q_BIAS},
        {"attn_k.bias", TensorRole::ATTN_K_BIAS},
        {"attn_v.bias", TensorRole::ATTN_V_BIAS},
        {"attn_output.bias", TensorRole::ATTN_OUTPUT_BIAS},
        // BitNet: sub-norms and the optional per-projection scalars. `.scale`
        // names are one-element tensors, so they are matched before the weight
        // table would reject them.
        {"attn_sub_norm.weight", TensorRole::ATTN_SUB_NORM},
        {"ffn_sub_norm.weight", TensorRole::FFN_SUB_NORM},
        {"attn_q.scale", TensorRole::PROJECTION_SCALE},
        {"attn_k.scale", TensorRole::PROJECTION_SCALE},
        {"attn_v.scale", TensorRole::PROJECTION_SCALE},
        {"attn_output.scale", TensorRole::PROJECTION_SCALE},
        {"ffn_gate.scale", TensorRole::PROJECTION_SCALE},
        {"ffn_up.scale", TensorRole::PROJECTION_SCALE},
        {"ffn_down.scale", TensorRole::PROJECTION_SCALE},
        {"attn_q_norm.weight", TensorRole::ATTN_Q_NORM},
        {"attn_sinks.weight", TensorRole::ATTN_SINKS},
        {"rope_freqs.weight", TensorRole::ROPE_FREQS},
        {"attn_k_norm.weight", TensorRole::ATTN_K_NORM},
        {"attn_post_norm.weight", TensorRole::ATTN_POST_NORM},
        {"ffn_post_norm.weight", TensorRole::FFN_POST_NORM},
        {"attn_qkv.weight", TensorRole::ATTN_QKV},
        {"attn_qkv.bias", TensorRole::ATTN_QKV_BIAS},
        // The shared expert keeps its own gate/projection names; `_shexp` is
        // distinct from `_exps`, so these cannot collide with routed experts.
        {"ffn_gate_inp_shexp.weight", TensorRole::FFN_GATE_INP_SHEXP},
        {"ffn_gate_shexp.weight", TensorRole::FFN_GATE_SHEXP},
        {"ffn_up_shexp.weight", TensorRole::FFN_UP_SHEXP},
        {"ffn_down_shexp.weight", TensorRole::FFN_DOWN_SHEXP},
        {"attn_norm.bias", TensorRole::ATTN_NORM_BIAS},
        {"ffn_norm.bias", TensorRole::FFN_NORM_BIAS},
        {"ffn_up.bias", TensorRole::FFN_UP_BIAS},
        {"ffn_down.bias", TensorRole::FFN_DOWN_BIAS},
        {"ffn_gate.bias", TensorRole::FFN_GATE_BIAS},
        {"attn_q_a.weight", TensorRole::ATTN_Q_A},
        {"attn_q_a_norm.weight", TensorRole::ATTN_Q_A_NORM},
        {"attn_q_b.weight", TensorRole::ATTN_Q_B},
        {"attn_kv_a_mqa.weight", TensorRole::ATTN_KV_A_MQA},
        {"attn_kv_a_norm.weight", TensorRole::ATTN_KV_A_NORM},
        {"attn_k_b.weight", TensorRole::ATTN_K_B},
        {"attn_v_b.weight", TensorRole::ATTN_V_B},
        {"ffn_exp_probs_b.bias", TensorRole::FFN_EXP_PROBS_B},
        {"ffn_gate_inp.bias", TensorRole::FFN_ROUTER_BIAS},
        {"ffn_gate_exps.bias", TensorRole::FFN_GATE_EXPS_BIAS},
        {"ffn_up_exps.bias", TensorRole::FFN_UP_EXPS_BIAS},
        {"ffn_down_exps.bias", TensorRole::FFN_DOWN_EXPS_BIAS},
        {"ffn_norm_ada_down.weight", TensorRole::FFN_NORM_ADA_DOWN},
        {"ffn_norm_ada_up.weight", TensorRole::FFN_NORM_ADA_UP},
        // llama.cpp's mamba tensors. `ssm_a` and `ssm_d` carry no `.weight`
        // suffix upstream, so they are matched by their bare names.
        {"ssm_in.weight", TensorRole::SSM_IN},
        {"ssm_conv1d.weight", TensorRole::SSM_CONV1D},
        {"ssm_conv1d.bias", TensorRole::SSM_CONV1D_BIAS},
        {"ssm_x.weight", TensorRole::SSM_X},
        {"ssm_dt.weight", TensorRole::SSM_DT},
        {"ssm_dt.bias", TensorRole::SSM_DT_BIAS},
        {"ssm_a", TensorRole::SSM_A},
        {"ssm_d", TensorRole::SSM_D},
        {"ssm_out.weight", TensorRole::SSM_OUT},
        {"ssm_norm.weight", TensorRole::SSM_NORM},
        {"ssm_dt_norm.weight", TensorRole::SSM_DT_NORM},
        {"ssm_b_norm.weight", TensorRole::SSM_B_NORM},
        {"ssm_c_norm.weight", TensorRole::SSM_C_NORM},
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
        doc.metadata_order.push_back(key);
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
        if (qit == qtable.end()) {
            // An unknown type is reported, not fatal. The container stays
            // inspectable so `freemoe probe` can name the exact blocker; the
            // byte range is recovered from the neighbouring offsets instead of
            // being guessed from a geometry we do not have.
            t.block_size = 0;
            t.block_bytes = 0;
            t.type_name = "TYPE_" + std::to_string(t.type_id);
            t.type_known = false;
        } else {
            t.block_size = qit->second.block_size;
            t.block_bytes = qit->second.block_bytes;
            t.type_name = qit->second.name;
            if (t.dims[0] % t.block_size != 0)
                fail(ErrorCode::MALFORMED_GGUF, "ne0 is not a whole number of blocks in " + t.name);
            uint64_t blocks = elements / t.block_size;
            t.byte_size = checked_mul(blocks, t.block_bytes, "tensor " + t.name + " byte count");
            t.type_known = true;
        }
        t.relative_offset = c.u64("tensor offset");
        if (t.relative_offset % alignment != 0)
            fail(ErrorCode::MALFORMED_GGUF, "misaligned tensor offset for " + t.name);
        t.name = canonical_layer_name(t.name);
        if (doc.tensor_by_name.count(t.name))
            fail(ErrorCode::MALFORMED_GGUF, "duplicate tensor name " + t.name);
        if (std::getenv("FM_GGUF_DEBUG"))
            std::fprintf(stderr, "tensor[%llu] %s pos=%zu\n",
                         static_cast<unsigned long long>(i), t.name.c_str(), c.pos);
        doc.tensor_by_name.emplace(t.name, doc.tensors.size());
        doc.tensors.push_back(std::move(t));
    }
    doc.data_offset = align_up(c.pos, alignment);
    if (!doc.tensors.empty() && doc.data_offset > size)
        fail(ErrorCode::MALFORMED_GGUF, "data section starts beyond file end");
    // A tokenizer-only file has no tensors, so its (empty) data section may be
    // padding past the end of the file. Nothing can read from it, and llama.cpp
    // ships its vocabularies in exactly this form.
    if (doc.tensors.empty() && doc.data_offset > size) doc.data_offset = size;

    // Recover byte ranges for tensors whose type has no geometry. Offsets are
    // the only trustworthy source: they are authoritative in the container, so
    // a gap between neighbours is this tensor's extent. A geometry we do not
    // implement must never cause a guessed size.
    {
        std::vector<size_t> order;
        order.reserve(doc.tensors.size());
        for (size_t i = 0; i < doc.tensors.size(); ++i) order.push_back(i);
        std::sort(order.begin(), order.end(), [&](size_t a, size_t b) {
            return doc.tensors[a].relative_offset < doc.tensors[b].relative_offset;
        });
        for (size_t k = 0; k < order.size(); ++k) {
            TensorInfo &tensor = doc.tensors[order[k]];
            if (tensor.type_known) continue;
            const uint64_t begin = tensor.relative_offset;
            const uint64_t end = (k + 1 < order.size())
                                     ? doc.tensors[order[k + 1]].relative_offset
                                     : (size > doc.data_offset ? size - doc.data_offset : 0);
            if (end <= begin)
                fail(ErrorCode::MALFORMED_GGUF,
                     "cannot bound unknown-type tensor " + tensor.name);
            tensor.byte_size = end - begin;
            tensor.byte_size_derived = true;
        }
    }

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

// Optional metadata lookup: returns false instead of failing when absent.
bool opt_metadata(const GGUFDocument &doc, const std::string &key, MetadataValue *out) {
    auto it = doc.metadata.find(key);
    if (it == doc.metadata.end()) return false;
    if (out) *out = it->second;
    return true;
}

// Reads an integer under the GGUF convention `<arch>.<name>`, then the bare
// `name`. Vendor files use the prefixed form; our own fixtures use `freemoe.`.
bool opt_u64(const GGUFDocument &doc, const std::set<std::string> &keys, uint64_t *out) {
    for (const auto &key : keys) {
        MetadataValue value;
        if (!opt_metadata(doc, key, &value)) continue;
        switch (value.type) {
        case GGUFValueType::UINT8: case GGUFValueType::UINT16:
        case GGUFValueType::UINT32: case GGUFValueType::UINT64:
            *out = value.u;
            return true;
        case GGUFValueType::INT8: case GGUFValueType::INT16:
        case GGUFValueType::INT32: case GGUFValueType::INT64:
            if (value.i < 0) return false;
            *out = static_cast<uint64_t>(value.i);
            return true;
        case GGUFValueType::FLOAT32: case GGUFValueType::FLOAT64:
            *out = static_cast<uint64_t>(value.as_double());
            return true;
        default:
            return false;
        }
    }
    return false;
}

bool opt_bool(const GGUFDocument &doc, const std::set<std::string> &keys, bool *out) {
    for (const auto &key : keys) {
        auto it = doc.metadata.find(key);
        if (it == doc.metadata.end()) continue;
        if (it->second.type != GGUFValueType::BOOL) continue;
        *out = it->second.b;
        return true;
    }
    return false;
}

bool opt_double(const GGUFDocument &doc, const std::set<std::string> &keys, double *out) {
    for (const auto &key : keys) {
        MetadataValue value;
        if (!opt_metadata(doc, key, &value)) continue;
        if (value.type == GGUFValueType::FLOAT32 || value.type == GGUFValueType::FLOAT64) {
            *out = value.as_double();
            return true;
        }
        if (value.type == GGUFValueType::UINT32 || value.type == GGUFValueType::UINT64) {
            *out = static_cast<double>(value.u);
            return true;
        }
        if (value.type == GGUFValueType::INT32 || value.type == GGUFValueType::INT64) {
            *out = static_cast<double>(value.i);
            return true;
        }
    }
    return false;
}

bool opt_string(const GGUFDocument &doc, const std::set<std::string> &keys, std::string *out) {
    for (const auto &key : keys) {
        MetadataValue value;
        if (!opt_metadata(doc, key, &value)) continue;
        if (value.type != GGUFValueType::STRING) continue;
        *out = value.s;
        return true;
    }
    return false;
}

// Same lookup, returning a pointer instead of a reference.
//
// `require_metadata` returns a reference *into `doc`*, but a call with a literal
// key also constructs a temporary `std::string` for the parameter, and GCC 13's
// `-Wdangling-reference` (part of `-Wextra`, so fatal under this project's
// `-Werror`) then assumes the returned reference may point at that temporary and
// refuses to compile the file. It is a false positive, and it is real: the
// project did not build on current Ubuntu/Debian toolchains until this became a
// pointer, which cannot dangle.
const MetadataValue *require_metadata_ptr(const GGUFDocument &doc, const std::string &key) {
    auto it = doc.metadata.find(key);
    if (it == doc.metadata.end())
        fail(ErrorCode::METADATA_MISMATCH, "missing required metadata key " + key);
    return &it->second;
}

const MetadataValue &require_metadata(const GGUFDocument &doc, const std::string &key) {
    return *require_metadata_ptr(doc, key);
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

// The activations this build implements, in the vocabulary `ModelConfig::
// activation` uses. One list, used by the manifest builder (which refuses a
// name it does not know) and by the engine (which would otherwise fall back to
// silu through `pointwise_activation`'s default). A new activation has to be
// added here and in include/fm/activation.hpp, which is the point.
bool is_known_activation(const std::string &kind) {
    static const std::set<std::string> known = {
        "silu", "gelu", "gelu_erf", "relu", "relu_sq", "softplus",
        "swiglu-oai", "silu-clamp",
    };
    return known.count(kind) != 0;
}

// Which architectures this build can map. Everything here shares the same
// pre-norm block structure and llama.cpp tensor naming; the differences are
// handled by per-architecture flags rather than by separate code paths. This
// lives outside the anonymous namespace because the tests bind it to the adapter
// table and the probe's table, which are the other two lists that decide it.
bool is_llama_family(const std::string &arch) {
    static const std::set<std::string> family = {
        "llama", "qwen2", "qwen3", "smollm", "smollm2", "tinyllama",
        "gemma", "gemma2", "gemma3", "phi3", "minicpm", "qwen3moe", "glm4",
        "qwen2moe", "phi2", "deepseek2", "internlm2", "orion", "exaone", "gptneox",
        "stablelm", "smollm3", "bitnet",
        // Gate-free FFN families whose only difference from the llama block is
        // the activation: relu^2 instead of silu (llama.cpp models/arcee.cpp,
        // models/nemotron.cpp, models/jais2.cpp, all LLM_FFN_RELU_SQR with
        // LLM_FFN_SEQ, i.e. a single up projection).
        "arcee", "nemotron", "jais2",
        // gpt-oss: the llama block with attention sinks, a post-attention norm
        // that feeds the MoE, `swiglu_oai` and per-expert projection biases.
        "gpt-oss",
        // plm: multi-head latent attention with a one-step q projection, a
        // gate-free relu^2 FFN and a tied output head.
        "plm",
        // Text paths of the mRoPE families and the two clamped deepseek
        // revisions; their graph is the block below, the differences are in the
        // rotary sections and the clamp bounds, both read from the file.
        "qwen3vl", "qwen3vlmoe", "deepseek4", "deepseek41",
        // mistral3: the llama block with optional MLP biases, an optional
        // adaptive RMSNorm driven by a sinusoidal delay embedding (whose MLP is
        // a gelu_erf), and llama.cpp's per-position attention temperature.
        "mistral3",
        // mamba/mamba2: no attention at all. Each layer is a 1D convolution and
        // a state-space scan over a rolling state, and a pure file has no FFN.
        "mamba", "mamba2",
        // jamba: the same recurrent block interleaved with attention layers in
        // one stack, and no rotary position on those attention layers at all.
        "jamba",
        // Batch A: verified llama-block families (see tools/triage_upstream_archs.py).
        "codeshell",
        "cohere2",
        "deci",
        "hunyuan-dense",
        "llama-embed",
        "maincoder",
        "nanbeige",
        "paddleocr",
        "pockettts",
        "qwen",
        "starcoder2",
        "xverse",
    };
    return family.count(arch) != 0;
}

// Maps a llama.cpp-family checkpoint onto the engine's canonical tensor names.
//
// The block structure is the same as the reference adapter (pre-norm RMSNorm,
// RoPE, SwiGLU, optional top-k MoE), so the forward pass is shared. What differs
// is the metadata vocabulary (`llama.block_count` vs `freemoe.num_hidden_layers`),
// the router tensor name, and the fact that vendor files carry no expert
// checksums. Each of those is handled explicitly rather than by guessing.
ModelManifest build_manifest_llama(const GGUFDocument &doc, const std::string &source_path,
                                   const std::string &arch) {
    ModelManifest m;
    m.doc = doc;
    m.source_path = source_path;
    ModelConfig &cfg = m.config;

    auto keys = [&arch](const std::string &name) {
        return std::set<std::string>{arch + "." + name, "freemoe." + name, name};
    };

    cfg.architecture = arch;
    cfg.adapter = "llama-family-v1";
    cfg.model_kind = "dense";
    cfg.rope_mode = "half";
    cfg.router_operation = "linear-topk-softmax-v1";
    cfg.expert_layout = "stacked-f32";

    uint64_t value = 0;
    auto require_key = [&](const std::string &name, const std::string &what) -> uint64_t {
        if (!opt_u64(doc, keys(name), &value))
            fail(ErrorCode::METADATA_MISMATCH,
                 "missing required " + what + " metadata (" + arch + "." + name + ")");
        return value;
    };

    cfg.layers = static_cast<uint32_t>(require_key("block_count", "layer count"));
    cfg.hidden = static_cast<uint32_t>(require_key("embedding_length", "embedding length"));

    // --- recurrent (state-space) layers -------------------------------------
    // A layer with `ssm_in` has no attention: llama.cpp's mamba block is a 1D
    // convolution followed by a state-space scan. A *pure* mamba model is then
    // all-recurrent and has no feed-forward either, while a hybrid stack
    // interleaves recurrent and attention layers. jamba is the only such family
    // mapped here; nemotron-h and falcon-h1 interleave differently and are not.
    // The tensors are the evidence, so the mask starts from them; metadata can
    // only confirm it.
    std::vector<uint8_t> recurrence(cfg.layers, 0);
    for (uint32_t l = 0; l < cfg.layers; ++l)
        recurrence[l] = doc.tensor_by_name.count(layer_name(l, "ssm_in.weight")) != 0 ? 1 : 0;
    {
        // llama.cpp reads the split from `attention.recurrent_layers` (a bool
        // array: minimax-01, qwen3next) or from a per-layer
        // `attention.head_count_kv` whose zero entries mean the layer has no
        // attention at all (jamba, nemotron-h). Whichever is present is only a
        // claim; the tensors decide, and a claim that disagrees with them is an
        // error rather than a preference.
        auto apply_mask = [&](const std::vector<uint8_t> &claimed, const char *key) {
            if (claimed.size() != cfg.layers)
                fail(ErrorCode::METADATA_MISMATCH,
                     std::string(key) + " has " + std::to_string(claimed.size()) +
                         " entries for " + std::to_string(cfg.layers) + " layers");
            for (uint32_t l = 0; l < cfg.layers; ++l) {
                if (claimed[l] != recurrence[l])
                    fail(ErrorCode::METADATA_MISMATCH,
                         std::string(key) + " says layer " + std::to_string(l) + " is " +
                             (claimed[l] ? "recurrent" : "an attention layer") + " but " +
                             (recurrence[l] ? "a" : "no") + " blk." + std::to_string(l) +
                             ".ssm_in.weight is present");
            }
        };
        if (auto it = doc.metadata.find(arch + ".attention.recurrent_layers");
            it != doc.metadata.end() && it->second.is_array()) {
            std::vector<uint8_t> claimed;
            for (const auto &entry : it->second.array) claimed.push_back(entry.as_bool() ? 1 : 0);
            apply_mask(claimed, (arch + ".attention.recurrent_layers").c_str());
        } else if (auto it = doc.metadata.find(arch + ".attention.head_count_kv");
                   it != doc.metadata.end() && it->second.is_array()) {
            std::vector<uint8_t> claimed;
            for (const auto &entry : it->second.array)
                claimed.push_back(entry.as_u64() == 0 ? 1 : 0);
            apply_mask(claimed, (arch + ".attention.head_count_kv").c_str());
        }
    }
    uint32_t first_recurrent = cfg.layers;
    uint32_t first_attention = cfg.layers;
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        if (recurrence[l]) {
            if (first_recurrent == cfg.layers) first_recurrent = l;
        } else if (first_attention == cfg.layers) {
            first_attention = l;
        }
    }
    const bool has_ssm = first_recurrent != cfg.layers;
    const bool has_attn = first_attention != cfg.layers;
    // Block facts are read from a layer that runs the block in question. In a
    // hybrid the first such layer is not layer 0, so probing layer 0 would
    // describe the wrong one.
    const uint32_t probe = has_attn ? first_attention : 0;
    cfg.recurrent_layers = recurrence;
    if (has_ssm) {
        cfg.ssm_conv = static_cast<uint32_t>(require_key("ssm.conv_kernel", "SSM convolution width"));
        cfg.ssm_inner = static_cast<uint32_t>(require_key("ssm.inner_size", "SSM inner size"));
        cfg.ssm_state = static_cast<uint32_t>(require_key("ssm.state_size", "SSM state size"));
        cfg.ssm_dt_rank =
            static_cast<uint32_t>(require_key("ssm.time_step_rank", "SSM time-step rank"));
        uint64_t groups = 1;
        if (opt_u64(doc, keys("ssm.group_count"), &groups) && groups > 1)
            cfg.ssm_groups = static_cast<uint32_t>(groups);
        opt_bool(doc, keys("ssm.dt_b_c_rms"), &cfg.ssm_dt_b_c_rms);
        if (cfg.ssm_conv < 2 || cfg.ssm_inner == 0 || cfg.ssm_state == 0 || cfg.ssm_dt_rank == 0)
            fail(ErrorCode::METADATA_MISMATCH,
                 "an SSM layer needs a convolution width of at least 2 and non-zero sizes");
        // mamba2 says so with a scalar decay: `ssm_a` is {1, n_head}, while
        // mamba's is one row of `d_state` per channel.
        const auto a_it = doc.tensor_by_name.find(layer_name(first_recurrent, "ssm_a"));
        const bool scalar_a =
            a_it != doc.tensor_by_name.end() && !doc.tensors[a_it->second].dims.empty() &&
            doc.tensors[a_it->second].dims[0] == 1;
        cfg.ssm_kind = scalar_a ? "mamba2" : "mamba";
        // mamba2 channels the grouped B and C through the same convolution as
        // the state, and normalises the scan's output per group (`ssm_norm`).
        // It never normalises dt, B or C on the way in: upstream's
        // `build_mamba2_layer` has no dt/B/C norm, and `ssm.dt_b_c_rms` is a
        // mamba-1 key read only by models/mamba.cpp. A file that carries those
        // weights anyway describes a norm the reference does not apply.
        if (cfg.ssm_kind == "mamba2") {
            if (doc.tensor_by_name.count(layer_name(first_recurrent, "ssm_norm.weight")) == 0)
                fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                     "mamba2 needs its grouped norm (blk." + std::to_string(first_recurrent) +
                         ".ssm_norm.weight)");
            if (cfg.ssm_inner % cfg.ssm_groups != 0 || cfg.ssm_inner % cfg.ssm_dt_rank != 0)
                fail(ErrorCode::SHAPE_MISMATCH,
                     "the SSM inner size must divide into groups and heads");
        } else {
            // mamba-1: upstream normalises dt, B and C before the scan when the
            // file sets `ssm.dt_b_c_rms` (models/mamba.cpp reads it) or carries
            // all three weights (models/mamba-base.cpp:99). A weight that is
            // absent is not "no norm": `build_norm` multiplies only when the
            // weight tensor is present, so that norm degenerates to unit gain.
            const char *const dt_suffix = "ssm_dt_norm.weight";
            const char *const b_suffix = "ssm_b_norm.weight";
            const char *const c_suffix = "ssm_c_norm.weight";
            const auto carries = [&](uint32_t layer, const char *suffix) {
                return doc.tensor_by_name.count(layer_name(layer, suffix)) != 0;
            };
            cfg.ssm_dt_norm_present = carries(first_recurrent, dt_suffix);
            cfg.ssm_b_norm_present = carries(first_recurrent, b_suffix);
            cfg.ssm_c_norm_present = carries(first_recurrent, c_suffix);
            // One rule is applied to every recurrent layer, so a stack whose
            // layers disagree would silently get the wrong arithmetic on some
            // of them. Refuse rather than pick one.
            for (uint32_t l = 0; l < cfg.layers; ++l) {
                if (!recurrence[l]) continue;
                if (carries(l, dt_suffix) != cfg.ssm_dt_norm_present ||
                    carries(l, b_suffix) != cfg.ssm_b_norm_present ||
                    carries(l, c_suffix) != cfg.ssm_c_norm_present)
                    fail(ErrorCode::METADATA_MISMATCH,
                         "recurrent layers disagree about the dt/B/C norms: blk." +
                             std::to_string(l) + " differs from blk." +
                             std::to_string(first_recurrent));
            }
            const bool all_norms = cfg.ssm_dt_norm_present && cfg.ssm_b_norm_present &&
                                   cfg.ssm_c_norm_present;
            cfg.ssm_vector_norms = cfg.ssm_dt_b_c_rms || all_norms;
            // jamba creates the three norms as required tensors
            // (models/jamba.cpp:65-71), so a jamba file without them is
            // malformed rather than plain.
            if (arch == "jamba" && !all_norms)
                fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                     "jamba's recurrent layers require ssm_dt_norm.weight, "
                     "ssm_b_norm.weight and ssm_c_norm.weight (models/jamba.cpp)");
            // mamba-1's scan is one head per channel.
            if (cfg.ssm_inner != 2 * cfg.hidden)
                fail(ErrorCode::SHAPE_MISMATCH,
                     "mamba expands the hidden width by exactly 2 (inner_size must be 2*hidden)");
        }
    }

    // --- feed-forward presence, and how many layers route to experts --------
    // Both are per-layer facts. A pure state-space model has no FFN at all; a
    // hybrid puts one behind each layer; and jamba mixes dense and MoE blocks in
    // one stack, which upstream expresses by making `ffn_gate_inp` optional per
    // layer. The tensors are the evidence in every case.
    cfg.ffn_layers.assign(cfg.layers, 0);
    cfg.moe_layers.assign(cfg.layers, 0);
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        const bool dense_ffn = doc.tensor_by_name.count(layer_name(l, "ffn_up.weight")) != 0 ||
                               doc.tensor_by_name.count(layer_name(l, "ffn_gate_up.weight")) != 0;
        const bool routed =
            doc.tensor_by_name.count(layer_name(l, "ffn_up_exps.weight")) != 0 ||
            doc.tensor_by_name.count(layer_name(l, "ffn_gate_exps.weight")) != 0 ||
            doc.tensor_by_name.count(layer_name(l, "ffn_gate_up_exps.weight")) != 0 ||
            doc.tensor_by_name.count(layer_name(l, "ffn_down_exps.weight")) != 0;
        // The router has two spellings in the adapter's own table: llama.cpp's
        // `ffn_gate_inp` and the reference adapter's `ffn_router`, which the
        // loader renames to the same role. Both mean the same thing here.
        const bool router = doc.tensor_by_name.count(layer_name(l, "ffn_gate_inp.weight")) != 0 ||
                            doc.tensor_by_name.count(layer_name(l, "ffn_router.weight")) != 0;
        cfg.ffn_layers[l] = (dense_ffn || routed) ? 1 : 0;
        cfg.moe_layers[l] = router ? 1 : 0;
        if (router && !routed)
            fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                 "blk." + std::to_string(l) + " has a router but no routed experts");
        if (routed && !router)
            fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                 "blk." + std::to_string(l) + " has routed experts but no router");
    }
    cfg.has_ffn = false;
    for (uint32_t l = 0; l < cfg.layers; ++l)
        if (cfg.ffn_layers[l]) cfg.has_ffn = true;

    // `feed_forward_length` is the dense width. It is required wherever a dense
    // FFN exists -- which is every layer of a dense model, an attention layer of a
    // hybrid, and even a recurrent layer whose stack has no attention at all.
    {
        bool dense_ffn = false;
        for (uint32_t l = 0; l < cfg.layers; ++l)
            if (cfg.ffn_layers[l] && !cfg.moe_layers[l]) dense_ffn = true;
        uint64_t intermediate = 0;
        if (opt_u64(doc, keys("feed_forward_length"), &intermediate))
            cfg.intermediate = static_cast<uint32_t>(intermediate);
        else if (dense_ffn)
            fail(ErrorCode::METADATA_MISMATCH, "missing required metadata: feed_forward_length");
        if (dense_ffn && cfg.intermediate == 0)
            fail(ErrorCode::METADATA_MISMATCH, "feed_forward_length must be positive");
    }
    if (has_attn) {
        // `attention.head_count` is scalar-or-array upstream
        // (`get_key_or_arr`, llama-model.cpp:1480). The array form states one
        // count per layer, with zero entries marking the recurrent layers of a
        // hybrid. This engine keeps one head count for the stack, so the array is
        // accepted when every attention layer agrees -- and a zero is checked
        // against the tensor-derived mask, so a file cannot claim a recurrence
        // split the tensors do not show.
        const std::string heads_key = arch + ".attention.head_count";
        auto heads_it = doc.metadata.find(heads_key);
        if (heads_it != doc.metadata.end() && heads_it->second.is_array()) {
            const auto &entries = heads_it->second.array;
            if (entries.size() != cfg.layers)
                fail(ErrorCode::METADATA_MISMATCH,
                     heads_key + " has " + std::to_string(entries.size()) + " entries for " +
                         std::to_string(cfg.layers) + " layers");
            uint64_t agreed = 0;
            bool set = false;
            for (uint32_t l = 0; l < cfg.layers; ++l) {
                const uint64_t count = entries[l].as_u64();
                if (count == 0) {
                    if (!recurrence[l])
                        fail(ErrorCode::METADATA_MISMATCH,
                             heads_key + " marks layer " + std::to_string(l) +
                                 " as recurrent but blk." + std::to_string(l) +
                                 ".ssm_in.weight is absent");
                    continue;
                }
                if (recurrence[l])
                    fail(ErrorCode::METADATA_MISMATCH,
                         heads_key + " gives layer " + std::to_string(l) +
                             " a head count but blk." + std::to_string(l) +
                             ".ssm_in.weight makes it recurrent");
                if (set && count != agreed)
                    fail(ErrorCode::SHAPE_MISMATCH,
                         "per-layer attention.head_count values differ among attention layers");
                agreed = count;
                set = true;
            }
            if (!set)
                fail(ErrorCode::METADATA_MISMATCH, "no attention layer states a head count");
            cfg.heads = static_cast<uint32_t>(agreed);
        } else {
            cfg.heads =
                static_cast<uint32_t>(require_key("attention.head_count", "attention head count"));
        }
        // Per-head kv counts can be a per-layer array, whose zero entries are the
        // recurrence mask handled above. Every attention layer must then agree on
        // the count: a cache sized per kv head cannot vary inside one stack.
        uint64_t kv_heads = 0;
        if (auto it = doc.metadata.find(arch + ".attention.head_count_kv");
            it != doc.metadata.end() && it->second.is_array()) {
            bool set = false;
            for (const auto &entry : it->second.array) {
                const uint64_t value = entry.as_u64();
                if (value == 0) continue;
                if (!set) {
                    kv_heads = value;
                    set = true;
                } else if (value != kv_heads) {
                    fail(ErrorCode::SHAPE_MISMATCH,
                         "per-layer attention.head_count_kv values differ among attention layers");
                }
            }
            if (!set) fail(ErrorCode::METADATA_MISMATCH, "no attention layer states a kv head count");
        } else if (!opt_u64(doc, keys("attention.head_count_kv"), &kv_heads)) {
            kv_heads = cfg.heads;
        }
        cfg.kv_heads = static_cast<uint32_t>(kv_heads);
        uint64_t head_dim = 0;
        if (!opt_u64(doc, keys("attention.key_length"), &head_dim)) {
            if (cfg.heads == 0 || cfg.hidden % cfg.heads != 0)
                fail(ErrorCode::SHAPE_MISMATCH,
                     "cannot derive head dimension from embedding length");
            head_dim = cfg.hidden / cfg.heads;
        }
        cfg.head_dim = static_cast<uint32_t>(head_dim);
    }
    // --- multi-head latent attention ---------------------------------------
    // Detected from the tensors: only MLA files carry `attn_kv_a_mqa`, which is
    // the projection into the latent. `head_dim` is then the *key* width
    // (nope + rope) and `attention.value_length_mla` the value width.
    if (has_attn && doc.tensor_by_name.count(layer_name(probe, "attn_kv_a_mqa.weight")) != 0) {
        cfg.mla = true;
        uint64_t rank = 0;
        if (opt_u64(doc, keys("attention.q_lora_rank"), &rank)) cfg.q_lora_rank = static_cast<uint32_t>(rank);
        if (!opt_u64(doc, keys("attention.kv_lora_rank"), &rank))
            fail(ErrorCode::METADATA_MISMATCH, "MLA needs attention.kv_lora_rank");
        cfg.kv_lora_rank = static_cast<uint32_t>(rank);
        uint64_t value_dim = 0;
        if (opt_u64(doc, keys("attention.value_length_mla"), &value_dim))
            cfg.head_v = static_cast<uint32_t>(value_dim);
        else
            cfg.head_v = cfg.head_dim;
        if (cfg.q_lora_rank == 0) {
            // The "lite" variant projects q in one step. That is a layout
            // difference only, so it is recorded and both are supported.
            if (doc.tensor_by_name.count(layer_name(probe, "attn_q_a.weight")) != 0)
                fail(ErrorCode::METADATA_MISMATCH,
                     "q_lora_rank is zero but a low-rank q projection is present");
        }
        // RoPE applies to the trailing part of the key only.
        uint64_t mla_rope = 0;
        if (opt_u64(doc, keys("rope.dimension_count"), &mla_rope)) cfg.rope_dim = static_cast<uint32_t>(mla_rope);
        if (cfg.rope_dim == 0 || cfg.rope_dim >= cfg.head_dim)
            fail(ErrorCode::SHAPE_MISMATCH,
                 "MLA needs a rotary width strictly smaller than the key width");
    }

    uint64_t rope_dim = 0;
    if (has_attn && opt_u64(doc, keys("rope.dimension_count"), &rope_dim))
        cfg.rope_dim = static_cast<uint32_t>(rope_dim);
    // jamba's attention layers carry no rotary position: upstream's block never
    // calls the rope, whatever `rope.dimension_count` says (converted files do
    // carry the key, because the converter writes the head width there). So the
    // architecture decides, and the key is read but not applied.
    if (arch == "jamba") cfg.rope_disabled = true;
    uint64_t vocab = 0;
    if (!opt_u64(doc, keys("vocab_size"), &vocab)) {
        MetadataValue tokens;
        if (!opt_metadata(doc, "tokenizer.ggml.tokens", &tokens) || !tokens.is_array())
            fail(ErrorCode::INVALID_TOKENIZER, "no vocabulary size and no token table");
        vocab = tokens.array.size();
    }
    cfg.vocab = static_cast<uint32_t>(vocab);
    cfg.context = static_cast<uint32_t>(require_key("context_length", "context length"));
    if (!opt_u64(doc, keys("expert_count"), &value)) value = 0;
    cfg.expert_count = static_cast<uint32_t>(value);
    uint64_t top_k = 0;
    if (!opt_u64(doc, keys("expert_used_count"), &top_k)) top_k = 0;
    cfg.expert_top_k = static_cast<uint32_t>(top_k);
    if (cfg.expert_count != 0) {
        opt_bool(doc, keys("expert_weights_norm"), &cfg.expert_weights_norm);
        opt_double(doc, keys("expert_weights_scale"), &cfg.expert_weights_scale);
        if (!(cfg.expert_weights_scale > 0.0))
            fail(ErrorCode::METADATA_MISMATCH, "expert_weights_scale must be positive");
        // Which function turns router logits into probabilities is a per-family
        // fact, not an MoE fact: V2 softmaxes, V3 and later use a sigmoid.
        // GGUF stores it as an enum, so the numbers are read as numbers.
        uint64_t gating = 0;
        if (opt_u64(doc, keys("expert_gating_func"), &gating)) {
            switch (gating) {
            case 0: case 1: cfg.expert_gating = "softmax"; break;
            case 2: cfg.expert_gating = "sigmoid"; break;
            case 3: cfg.expert_gating = "softmax-weight"; break;
            case 4: cfg.expert_gating = "sqrt-softplus"; break;
            default:
                fail(ErrorCode::METADATA_MISMATCH,
                     "expert_gating_func " + std::to_string(gating) + " has no known meaning");
            }
        }
        uint64_t groups = 0, groups_used = 0;
        if (opt_u64(doc, keys("expert_group_count"), &groups) && groups > 1) {
            cfg.expert_groups = static_cast<uint32_t>(groups);
            if (!opt_u64(doc, keys("expert_group_used_count"), &groups_used))
                fail(ErrorCode::METADATA_MISMATCH,
                     "group-limited routing needs expert_group_used_count");
            cfg.expert_groups_used = static_cast<uint32_t>(groups_used);
            if (cfg.expert_groups_used == 0 || cfg.expert_groups_used > cfg.expert_groups)
                fail(ErrorCode::METADATA_MISMATCH, "expert_group_used_count is out of range");
            if (cfg.expert_count % cfg.expert_groups != 0)
                fail(ErrorCode::SHAPE_MISMATCH,
                     "expert_count is not a multiple of expert_group_count");
        }
    }
    uint64_t dense_lead = 0;
    if (opt_u64(doc, keys("leading_dense_block_count"), &dense_lead))
        cfg.dense_lead = static_cast<uint32_t>(dense_lead);
    // Routed experts and the qwen2moe shared expert each have their own width.
    // Both default to the dense width, which is what every other MoE family
    // does; a checkpoint that narrows either one says so explicitly.
    if (opt_u64(doc, keys("expert_feed_forward_length"), &value)) {
        cfg.expert_intermediate = static_cast<uint32_t>(value);
        if (cfg.expert_intermediate == 0)
            fail(ErrorCode::METADATA_MISMATCH, "expert feed-forward length must be positive");
    }
    // The shared expert is detected from its tensors rather than from
    // `expert_shared_count`: the tensors are what the forward pass reads, so
    // they are the evidence. A file that declares a count without the tensors
    // is caught by the missing-tensor check below.
    {
        // These features belong to the MoE layers, and the first MoE layer is
        // not necessarily layer 0: deepseek2 has plain-FFN blocks in front. So
        // the test scans every layer instead of assuming blk.0.
        auto has_any_layer = [&](const char *suffix) {
            const std::string tail = std::string(".") + suffix;
            for (const auto &tensor : doc.tensors) {
                if (tensor.name.size() < tail.size()) continue;
                if (tensor.name.compare(tensor.name.size() - tail.size(), tail.size(), tail) == 0)
                    return true;
            }
            return false;
        };
        const bool shexp_gate = has_any_layer("ffn_gate_shexp.weight");
        const bool shexp_up = has_any_layer("ffn_up_shexp.weight");
        const bool shexp_down = has_any_layer("ffn_down_shexp.weight");
        const bool shexp_inp = has_any_layer("ffn_gate_inp_shexp.weight");
        cfg.expert_selection_bias = has_any_layer("ffn_exp_probs_b.bias");
        // gpt-oss adds a bias to the router logits (before gating) and one to
        // each expert projection. Both are tensor facts, and they are separate
        // from the selection bias above, which is applied after gating.
        cfg.expert_router_bias = has_any_layer("ffn_gate_inp.bias");
        cfg.expert_projection_bias = has_any_layer("ffn_gate_exps.bias");
        {
            const bool up_b = has_any_layer("ffn_up_exps.bias");
            const bool down_b = has_any_layer("ffn_down_exps.bias");
            if (up_b != cfg.expert_projection_bias || down_b != cfg.expert_projection_bias)
                fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                     "expert projection biases must be present on gate, up and down, or on none");
        }
        if (shexp_gate || shexp_up || shexp_down || shexp_inp) {
            // The three projections are the branch; the per-token gate on top of
            // them is a qwen2moe addition, so it is optional and its presence is
            // what switches the scaling on.
            if (!(shexp_gate && shexp_up && shexp_down))
                fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                     "shared expert is only partly present: its gate, up and down projections "
                     "must all be there or none");
            cfg.shared_expert = true;
            cfg.shared_expert_gate = shexp_inp;
            cfg.shared_expert_intermediate = 0;
            if (opt_u64(doc, keys("expert_shared_feed_forward_length"), &value))
                cfg.shared_expert_intermediate = static_cast<uint32_t>(value);
            if (cfg.shared_expert_intermediate == 0) cfg.shared_expert_intermediate = cfg.intermediate;
        }
    }
    cfg.model_kind = cfg.expert_count > 0 ? "moe" : "dense";

    // --- family semantics ---------------------------------------------------
    // These change the arithmetic, not just the vocabulary, so each is stated
    // explicitly and validated rather than inferred from whatever tensors
    // happen to be present.
    if (arch == "gemma" || arch == "gemma2" || arch == "gemma3") {
        // gemma scales embeddings by sqrt(n_embd), stores norm scales
        // zero-centred, and uses GeGLU instead of SwiGLU.
        cfg.embedding_scale = std::sqrt(static_cast<double>(cfg.hidden));
        cfg.norm_add_one = true;
        cfg.activation = "gelu";
    }
    double number = 0.0;
    if (arch == "minicpm") {
        // MiniCPM scales the token embedding before the first block. Older GGUF
        // exports omit the key, in which case the original model used 12.0;
        // newer exports write it explicitly.
        cfg.embedding_scale = 12.0;
        if (opt_double(doc, keys("embedding_scale"), &number) && number > 0.0)
            cfg.embedding_scale = number;
        // These two are recorded, not applied. llama.cpp reads and logs them and
        // never uses them in the graph, so a runtime that applied them would
        // disagree with the reference on every logit.
        if (opt_double(doc, keys("residual_scale"), &number)) cfg.residual_scale = number;
        if (opt_double(doc, keys("logit_scale"), &number)) cfg.logit_scale = number;
    }

    if (arch == "gemma2" || arch == "gemma3") {
        cfg.post_norms = true;
        if (opt_double(doc, keys("attn_logit_softcapping"), &number)) {
            if (!(number > 0.0))
                fail(ErrorCode::METADATA_MISMATCH, "attn_logit_softcapping must be positive");
            cfg.attn_logit_softcap = number;
        }
        if (opt_double(doc, keys("final_logit_softcapping"), &number)) {
            if (!(number > 0.0))
                fail(ErrorCode::METADATA_MISMATCH, "final_logit_softcapping must be positive");
            cfg.final_logit_softcap = number;
        }
    }

    // Sliding window. Any family may declare one; the window is a property of
    // the file, not of one architecture, so it is read for every family rather
    // than only for the ones that used to be the only users of it.
    {
        uint64_t window = 0;
        if (opt_u64(doc, keys("attention.sliding_window"), &window))
            cfg.sliding_window = static_cast<uint32_t>(window);
        if (cfg.sliding_window != 0 && cfg.sliding_window >= cfg.context)
            // A window at least as wide as the context cannot change anything,
            // so treating it as "no window" is exact rather than approximate.
            cfg.sliding_window = 0;
    }

    // Per-layer window selection. `attention.sliding_window_pattern` is either an
    // array of per-layer flags or an integer period; llama.cpp's period form
    // marks every layer whose index is not a multiple of the period as windowed
    // (dense_first), which is what gpt-oss relies on.
    if (cfg.sliding_window != 0) {
        if (auto it = doc.metadata.find(arch + ".attention.sliding_window_pattern");
            it != doc.metadata.end()) {
            if (it->second.is_array()) {
                const auto &entries = it->second.array;
                if (!entries.empty() && entries.size() != cfg.layers)
                    fail(ErrorCode::METADATA_MISMATCH,
                         "attention.sliding_window_pattern has " +
                             std::to_string(entries.size()) + " entries for " +
                             std::to_string(cfg.layers) + " layers");
                for (const auto &entry : entries)
                    cfg.swa_layers.push_back(entry.as_bool() ? 1 : 0);
            } else {
                const uint64_t period = it->second.as_u64();
                if (period == 0)
                    fail(ErrorCode::METADATA_MISMATCH,
                         "attention.sliding_window_pattern must not be zero");
                for (uint32_t layer = 0; layer < cfg.layers; ++layer)
                    cfg.swa_layers.push_back((layer % period) != 0 ? 1 : 0);
            }
        }
    }

    if (opt_double(doc, keys("attention.layer_norm_rms_epsilon"), &number)) cfg.rms_eps = number;
    else cfg.rms_eps = 1e-5;

    // Norm type. RMSNorm files declare `attention.layer_norm_rms_epsilon`;
    // LayerNorm files declare `attention.layer_norm_epsilon`. That key is the
    // evidence, so the norm is chosen from it rather than from the arch name.
    {
        const bool has_ln = opt_metadata(doc, arch + ".attention.layer_norm_epsilon", nullptr);
        const bool has_rms = opt_metadata(doc, arch + ".attention.layer_norm_rms_epsilon", nullptr);
        if (has_ln && has_rms)
            fail(ErrorCode::METADATA_MISMATCH,
                 "the file declares both a LayerNorm and an RMSNorm epsilon; the norm type is "
                 "ambiguous");
        if (has_ln) {
            cfg.layer_norm = true;
            if (opt_double(doc, keys("attention.layer_norm_epsilon"), &number)) cfg.rms_eps = number;
        }
    }
    // Biased norms, and the biases phi2 puts on the FFN, the attention output
    // and the logits. All of them are read from the tensors.
    {
        const bool attn_nb = doc.tensor_by_name.count(layer_name(probe, "attn_norm.bias")) != 0;
        const bool out_nb = doc.tensor_by_name.count("output_norm.bias") != 0;
        const bool ffn_present =
            doc.tensor_by_name.count(layer_name(probe, "ffn_norm.weight")) != 0;
        const bool ffn_nb = doc.tensor_by_name.count(layer_name(probe, "ffn_norm.bias")) != 0;
        if ((attn_nb || out_nb || ffn_nb) && !(attn_nb && out_nb && (!ffn_present || ffn_nb)))
            fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                 "norm biases must be present on every norm, or on none");
        if (attn_nb != out_nb)
            fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                 "the attention norm and the final norm disagree about carrying a bias");
        cfg.norm_bias = attn_nb;
        const bool up_b = doc.tensor_by_name.count(layer_name(probe, "ffn_up.bias")) != 0;
        const bool down_b = doc.tensor_by_name.count(layer_name(probe, "ffn_down.bias")) != 0;
        if (up_b != down_b)
            fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                 "ffn_up and ffn_down must both carry a bias or neither");
        cfg.ffn_bias = up_b;
        cfg.ffn_gate_bias = doc.tensor_by_name.count(layer_name(probe, "ffn_gate.bias")) != 0;
        cfg.attn_output_bias = doc.tensor_by_name.count(layer_name(probe, "attn_output.bias")) != 0;
        cfg.logit_bias = doc.tensor_by_name.count("output.bias") != 0;
        // A block with no FFN norm can only work if the FFN reuses the
        // attention's normalised input: that is phi2's parallel residual. A
        // file that has a post-attention norm instead (gpt-oss) does not take
        // this branch -- it normalises the residual, so it is an ordinary
        // pre-norm block and the "parallel" reading would double-count.
        const bool attn_post_present =
            doc.tensor_by_name.count(layer_name(probe, "attn_post_norm.weight")) != 0;
        // A recurrent layer has no FFN to normalise, so the absence of an
        // ffn_norm there is not phi2's shared residual.
        if (!ffn_present && !attn_post_present && has_attn) {
            if (doc.tensor_by_name.count(layer_name(probe, "attn_norm.weight")) == 0)
                fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                     "no ffn_norm and no attn_norm: nothing normalises the FFN input");
            cfg.parallel_residual = true;
        }
        // A block with up/down but no gate has no gated activation at all. The
        // activation itself is not representable in GGUF metadata, so it comes
        // from the architecture table -- the only such decision in this file,
        // and the reason each such architecture needs a parity fixture.
        const bool has_gate = doc.tensor_by_name.count(layer_name(probe, "ffn_gate.weight")) != 0;
        // A [hidden, 2*intermediate] up tensor is glm4's fused gate+up, not a
        // gate-free FFN, so the width decides between the two.
        const auto up_it = doc.tensor_by_name.find(layer_name(probe, "ffn_up.weight"));
        const bool up_present = up_it != doc.tensor_by_name.end();
        const bool fused_width =
            up_present && doc.tensors[up_it->second].dims.size() == 2 &&
            doc.tensors[up_it->second].dims[1] == static_cast<uint64_t>(cfg.intermediate) * 2;
        if (!has_gate && up_present && cfg.expert_count == 0 && !fused_width)
            cfg.ffn_no_gate = true;
    }
    // Attention sinks are a tensor fact. Every layer must carry them or none:
    // a file that has them on some layers only would silently mix two attention
    // definitions, which is a mapping error rather than a variant worth guessing.
    if (doc.tensor_by_name.count(layer_name(probe, "attn_sinks.weight")) != 0) {
        for (uint32_t layer = 0; layer < cfg.layers; ++layer)
            if (doc.tensor_by_name.count(layer_name(layer, "attn_sinks.weight")) == 0)
                fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                     "attention sinks are present on some layers only; every layer "
                     "must define them");
        cfg.attn_sinks = true;
    }
    // SmolLM3 interleaves NoPE layers: llama.cpp sets `n_no_rope_layer_step = 4`
    // for that architecture alone, so the rule is the architecture's, not the
    // file's. Every fourth layer (1-based) keeps its keys unrotated.
    if (arch == "smollm3") cfg.no_rope_layer_step = 4;

    // BitNet keeps its normalisation inside the block: a second RMSNorm on the
    // attention output before `attn_output`, and another on the gated FFN output
    // before `ffn_down` (llama.cpp's models/bitnet.cpp). Its output head is tied
    // to the token table, which the loader already handles by absence.
    // Presence is the evidence -- the tensors are what the forward pass reads --
    // but bitnet *requires* them, and a partially written file is a mapping
    // error rather than a variant. Checking both keeps a wrong graph from being
    // run silently either way.
    const bool has_attn_sub_norm =
        doc.tensor_by_name.count(layer_name(probe, "attn_sub_norm.weight")) != 0;
    const bool has_ffn_sub_norm =
        doc.tensor_by_name.count(layer_name(probe, "ffn_sub_norm.weight")) != 0;
    if (has_attn_sub_norm != has_ffn_sub_norm)
        fail(ErrorCode::MISSING_REQUIRED_TENSOR,
             "sub-norms must exist on both the attention and the FFN path; this "
             "file defines only one of them");
    if (arch == "bitnet" && !has_attn_sub_norm)
        fail(ErrorCode::MISSING_REQUIRED_TENSOR,
             "bitnet requires attn_sub_norm.weight and ffn_sub_norm.weight");
    cfg.sub_norms = has_attn_sub_norm;
    if (cfg.sub_norms) {
        for (uint32_t layer = 0; layer < cfg.layers; ++layer)
            for (const char *suffix : {"attn_sub_norm.weight", "ffn_sub_norm.weight"})
                if (doc.tensor_by_name.count(layer_name(layer, suffix)) == 0)
                    fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                         "sub-norms are present on some layers only: " +
                             layer_name(layer, suffix));
    }

    // gpt-oss routes with `LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT`: the
    // router logits are the probabilities and the softmax happens over the
    // selected experts only, which is `softmax-weight` here. It comes from the
    // architecture because llama.cpp hard-codes it per model file rather than
    // reading a key.
    if (arch == "gpt-oss") cfg.expert_gating = "softmax-weight";

    // mistral3's adaptive RMSNorm. The tensors are the evidence: the norm's MLP
    // only exists in realtime checkpoints, and it is defined on a delay
    // embedding whose length the file states.
    {
        const bool ada = doc.tensor_by_name.count(layer_name(probe, "ffn_norm_ada_down.weight")) != 0;
        const bool ada_up = doc.tensor_by_name.count(layer_name(probe, "ffn_norm_ada_up.weight")) != 0;
        if (ada != ada_up)
            fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                 "the adaptive norm needs both of its projections (ffn_norm_ada_down/-up)");
        cfg.ffn_norm_ada = ada;
        if (ada) {
            uint64_t delay = 0;
            if (opt_u64(doc, keys("realtime_delay_tokens"), &delay))
                cfg.realtime_delay = static_cast<uint32_t>(delay);
            // The bottleneck width is a property of the file, not a key: it is
            // read from the tensor so no metadata has to agree with it.
            auto down = doc.tensor_by_name.find(layer_name(probe, "ffn_norm_ada_down.weight"));
            const auto &dims = doc.tensors[down->second].dims;
            if (dims.size() != 2 || dims[0] != cfg.hidden || dims[1] == 0)
                fail(ErrorCode::SHAPE_MISMATCH,
                     "ffn_norm_ada_down must be [hidden, bottleneck] with a positive bottleneck");
        }
    }
    // The per-position attention temperature. Upstream reads the scale and then
    // requires a floor; a file that sets one without the other cannot be run,
    // and silently using 1.0 instead would change every logit.
    {
        double temp = 0.0;
        if (opt_double(doc, keys("attention.temperature_scale"), &temp) && temp != 0.0) {
            cfg.attn_temp_scale = temp;
            double floor = 0.0;
            if (!opt_double(doc, keys("rope.scaling.original_context_length"), &floor) || floor <= 0.0)
                fail(ErrorCode::METADATA_MISMATCH,
                     "attention.temperature_scale needs rope.scaling.original_context_length as "
                     "the ramp floor");
            cfg.attn_temp_floor = floor;
        }
    }

    // Router renormalisation denominator. llama.cpp clamps it to the smallest F16
    // normal for every architecture except deepseek4.1, which adds 1e-20; keeping
    // the rule here means the two are not conflated.
    if (arch == "deepseek41") {
        cfg.expert_weights_epsilon = 1e-20;
        cfg.expert_weights_epsilon_additive = true;
    }

    // Clamped gated activations. Two families, two variants:
    //   * gpt-oss / minimax-m3 use `swiglu_oai` (alpha 1.702, limit 7.0), the
    //     constants llama.cpp hard-codes for LLM_FFN_SWIGLU_OAI_MOE;
    //   * deepseek4/4.1, glm5-next and dflash keep plain silu but clamp gate and
    //     up per layer, with the bound in `swiglu_clamp_exp` / `swiglu_clamp_shexp`.
    {
        if (arch == "gpt-oss" || arch == "minimax-m3") {
            cfg.activation = "swiglu-oai";
            cfg.activation_alpha = 1.702;
            cfg.activation_limit = 7.0;
        } else if (auto clamp_it = doc.metadata.find(arch + ".swiglu_clamp_exp");
                   clamp_it != doc.metadata.end() && clamp_it->second.is_array()) {
            for (const auto &entry : clamp_it->second.array)
                cfg.swiglu_clamp.push_back(entry.as_double());
            if (auto shexp_it = doc.metadata.find(arch + ".swiglu_clamp_shexp");
                shexp_it != doc.metadata.end() && shexp_it->second.is_array()) {
                for (const auto &entry : shexp_it->second.array)
                    cfg.swiglu_clamp_shexp.push_back(entry.as_double());
            }
            // Any bound above upstream's 1e-6 threshold selects the clamped
            // variant; the per-layer decision is then made at compute time.
            const double worst =
                *std::max_element(cfg.swiglu_clamp.begin(), cfg.swiglu_clamp.end());
            if (worst > 1e-6) {
                cfg.activation = "silu-clamp";
                cfg.activation_limit = worst;
                // Which of the two upstream forms this architecture uses
                // (llama-graph.cpp:2449). DFLASH only takes the fused branch when
                // it carries a dsv4 compression factor; without one it is
                // unfused like everything else.
                static const std::set<std::string> fused_clamp = {
                    "maple", "deepseek4", "deepseek41", "glm5-next", "hy-v4",
                };
                cfg.swiglu_clamp_fused = fused_clamp.count(arch) != 0;
                if (arch == "dflash" && cfg.hyper_connection_count > 0.0)
                    cfg.swiglu_clamp_fused = true;
            }
        }
    }

    // Fused gate+up expert tensors. The two projections live in one tensor, gate
    // block first (ggml_swiglu's split order), so presence of the fused name and
    // absence of the separate ones is the evidence. A file carrying both is
    // ambiguous and refused rather than guessed.
    {
        const bool fused = doc.tensor_by_name.count(layer_name(probe, "ffn_gate_up_exps.weight")) != 0;
        const bool separate = doc.tensor_by_name.count(layer_name(probe, "ffn_gate_exps.weight")) != 0;
        if (fused && separate)
            fail(ErrorCode::METADATA_MISMATCH,
                 "both ffn_gate_up_exps and ffn_gate_exps are present; the expert layout is ambiguous");
        cfg.fused_gate_up_exps = fused;
    }

    // Per-pair rotary frequency divisors. Layer 0 carries them; later layers may
    // repeat them, so a file with the vector on layer 0 only is complete.
    if (doc.tensor_by_name.count(layer_name(probe, "rope_freqs.weight")) != 0)
        cfg.rope_freqs = true;
    if (cfg.ffn_no_gate) {
        // A gate-free FFN cannot be told from its tensors which activation it
        // means, so the architecture has to say. The table is read from
        // llama.cpp's own model files (`LLM_FFN_SEQ` with the listed
        // activation); anything else is still refused rather than guessed.
        static const std::map<std::string, const char *> seq_activation = {
            {"phi2", "gelu"},       // src/models/phi2.cpp: LLM_FFN_GELU, LLM_FFN_SEQ
            {"gptneox", "gelu"},    // src/models/gptneox.cpp: LLM_FFN_GELU, LLM_FFN_SEQ
            {"arcee", "relu_sq"},   // src/models/arcee.cpp: LLM_FFN_RELU_SQR, SEQ
            {"nemotron", "relu_sq"},// src/models/nemotron.cpp: LLM_FFN_RELU_SQR, SEQ
            {"jais2", "relu_sq"},   // src/models/jais2.cpp: LLM_FFN_RELU_SQR, SEQ
            {"plm", "relu_sq"},     // src/models/plm.cpp: LLM_FFN_RELU_SQR, SEQ
            // GeGLU with the up projection as the whole branch: no gate tensor.
            {"codeshell", "gelu"},  // src/models/codeshell.cpp: LLM_FFN_GELU, SEQ
            {"starcoder2", "gelu"}, // src/models/starcoder2.cpp: LLM_FFN_GELU, SEQ
            {"pockettts", "gelu"},  // src/models/pockettts.cpp: LLM_FFN_GELU, SEQ
        };
        auto it = seq_activation.find(arch);
        if (it == seq_activation.end())
            fail(ErrorCode::NOT_IMPLEMENTED,
                 "a gate-free feed-forward needs its activation named by the architecture, and " +
                     arch + " is not in the table; refusing rather than guessing");
        cfg.activation = it->second;
    }

    // The activation drives the arithmetic, so an unknown name is refused rather
    // than silently becoming silu. This runs after every branch that can set it
    // (the family defaults, the clamped variants, the gate-free table).
    if (!is_known_activation(cfg.activation))
        fail(ErrorCode::METADATA_MISMATCH,
             "unsupported activation " + cfg.activation + " for architecture " + arch);
    if (opt_double(doc, keys("hyper_connection.count"), &number))
        cfg.hyper_connection_count = number;
    if (opt_double(doc, keys("rope.freq_base"), &number)) cfg.rope_theta = number;
    else if (opt_double(doc, keys("rope_theta"), &number)) cfg.rope_theta = number;
    else cfg.rope_theta = 10000.0;

    // mRoPE sections. The key holds four counts of rotary *pairs*; llama.cpp
    // turns mRoPE on only when the first two are non-zero, which is what keeps a
    // text-only checkpoint that carries the key in ordinary rotary mode.
    if (auto it = doc.metadata.find(arch + ".rope.dimension_sections");
        it != doc.metadata.end() && it->second.is_array()) {
        const auto &entries = it->second.array;
        if (entries.size() != cfg.rope_sections.size())
            fail(ErrorCode::METADATA_MISMATCH,
                 arch + ".rope.dimension_sections must have four entries, got " +
                     std::to_string(entries.size()));
        for (size_t i = 0; i < entries.size(); ++i)
            cfg.rope_sections[i] = static_cast<uint32_t>(entries[i].as_u64());
        // The interleaved layout is an architecture rule, not a file flag:
        // upstream selects it in llama_rope_type() for these families.
        static const std::set<std::string> interleaved = {
            "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe", "clef",
        };
        cfg.mrope_interleaved = interleaved.count(arch) != 0;
        cfg.mrope = cfg.rope_sections[0] > 0 && cfg.rope_sections[1] > 0;
        const uint32_t total = cfg.rope_sections[0] + cfg.rope_sections[1] +
                               cfg.rope_sections[2] + cfg.rope_sections[3];
        if (cfg.mrope && total > (cfg.rope_dim ? cfg.rope_dim : cfg.head_dim) / 2)
            fail(ErrorCode::METADATA_MISMATCH,
                 arch + ".rope.dimension_sections cover " + std::to_string(total) +
                     " pairs, more than the rotary head width holds");
    }

    // The mRoPE families cannot be run without their section split: upstream
    // reads `rope.dimension_sections` as a *required* key for them, so a file
    // that omits it is not a smaller model but a different one.
    {
        static const std::set<std::string> mrope_required = {
            "qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe",
        };
        if (mrope_required.count(arch) != 0 &&
            doc.metadata.find(arch + ".rope.dimension_sections") == doc.metadata.end())
            fail(ErrorCode::METADATA_MISMATCH,
                 arch + ".rope.dimension_sections is required for this architecture");
    }

    // Context extension. The keys are read verbatim and the arithmetic is
    // derived in src/rope.cpp, so the derived values cannot drift from the file.
    std::string scaling_type;
    if (opt_string(doc, keys("rope.scaling.type"), &scaling_type) && !scaling_type.empty())
        cfg.rope_scaling_type = scaling_type;
    double scaling_number = 0.0;
    if (opt_double(doc, keys("rope.scaling.factor"), &scaling_number))
        cfg.rope_scaling_factor = scaling_number;
    else if (opt_double(doc, keys("rope.scale_linear"), &scaling_number))
        cfg.rope_scaling_factor = scaling_number;  // pre-2024 key name
    if (opt_double(doc, keys("rope.scaling.original_context_length"), &scaling_number))
        cfg.rope_n_ctx_orig = scaling_number;
    else
        cfg.rope_n_ctx_orig = static_cast<double>(cfg.context);
    if (opt_double(doc, keys("rope.scaling.attn_factor"), &scaling_number))
        cfg.rope_scaling_attn_factor = scaling_number;
    if (opt_double(doc, keys("rope.scaling.yarn_log_multiplier"), &scaling_number))
        cfg.rope_yarn_log_mul = scaling_number;
    if (opt_double(doc, keys("rope.scaling.yarn_beta_fast"), &scaling_number))
        cfg.rope_yarn_beta_fast = scaling_number;
    if (opt_double(doc, keys("rope.scaling.yarn_beta_slow"), &scaling_number))
        cfg.rope_yarn_beta_slow = scaling_number;
    resolve_rope_scaling(cfg);

    cfg.bos_id = 1;
    cfg.eos_id = 2;
    if (opt_u64(doc, {"tokenizer.ggml.bos_token_id"}, &value))
        cfg.bos_id = static_cast<uint32_t>(value);
    if (opt_u64(doc, {"tokenizer.ggml.eos_token_id"}, &value))
        cfg.eos_id = static_cast<uint32_t>(value);

    std::string tokenizer;
    if (!opt_string(doc, {"tokenizer.ggml.model"}, &tokenizer) || tokenizer.empty())
        fail(ErrorCode::INVALID_TOKENIZER, "checkpoint carries no tokenizer model");
    cfg.tokenizer_model = tokenizer;
    std::string chat_template;
    if (opt_string(doc, {"tokenizer.chat_template"}, &chat_template))
        cfg.chat_template = chat_template;

    // --- semantic checks that must fail closed rather than be approximated ---
    // A recurrent model has no attention, so it states no head counts at all.
    if (cfg.ssm_kind.empty() &&
        (cfg.heads == 0 || cfg.kv_heads == 0 || cfg.head_dim == 0 ||
         cfg.heads % cfg.kv_heads != 0 || cfg.kv_heads > cfg.heads))
        fail(ErrorCode::SHAPE_MISMATCH, "grouped-query head counts are inconsistent");
    // MLA projects q through a bottleneck, so the per-head key width has no
    // reason to fill the embedding. Every other family keeps the identity.
    if (cfg.ssm_kind.empty() && !cfg.mla &&
        static_cast<uint64_t>(cfg.heads) * cfg.head_dim != cfg.hidden)
        fail(ErrorCode::SHAPE_MISMATCH,
             "attention head count times head dimension does not equal embedding length");
    if (cfg.rope_dim != 0 && (cfg.rope_dim % 2 != 0 || cfg.rope_dim > cfg.head_dim))
        fail(ErrorCode::SHAPE_MISMATCH, "rotary dimension is invalid");
    if (cfg.shared_expert && cfg.expert_count == 0)
        fail(ErrorCode::METADATA_MISMATCH,
             "a shared expert without routed experts has no defined semantics");
    auto reject = [&](const char *metadata_key, const char *what) {
        MetadataValue v;
        if (opt_metadata(doc, arch + "." + metadata_key, &v) && v.as_bool())
            fail(ErrorCode::NOT_IMPLEMENTED, std::string(what) + " is not implemented");
    };
    // Post-attention / post-FFN norms: present in gemma2/3 and glm4. The tensors
    // are the evidence, so this is read rather than assumed per architecture.
    {
        const bool attn_post = doc.tensor_by_name.count(layer_name(probe, "attn_post_norm.weight")) != 0;
        const bool ffn_post = doc.tensor_by_name.count(layer_name(probe, "ffn_post_norm.weight")) != 0;
        const bool ffn_norm = doc.tensor_by_name.count(layer_name(probe, "ffn_norm.weight")) != 0;
        if (attn_post && ffn_post) {
            // gemma2/3 and glm4: both norms live inside the block, on the branch
            // outputs, before the residual adds.
            cfg.post_norms = true;
        } else if (attn_post && !ffn_post && !ffn_norm) {
            // gpt-oss: one post-attention norm, applied after the attention
            // residual, and no separate FFN norm. Running the FFN on it is the
            // ordinary pre-norm block with a different tensor name.
            cfg.ffn_norm_is_post = true;
        } else if (attn_post != ffn_post) {
            fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                 "post-attention and post-FFN norms must both be present or both absent");
        }
    }

    // phi3 stores one fused q/k/v projection instead of three. Detect it from
    // the tensors, because the metadata does not say either way.
    if (has_attn && !cfg.mla) {
        // MLA builds q from a low-rank pair, so it has neither shape. A
        // recurrent layer has no attention projection at all.
        const bool fused = doc.tensor_by_name.count(layer_name(probe, "attn_qkv.weight")) != 0;
        const bool split = doc.tensor_by_name.count(layer_name(probe, "attn_q.weight")) != 0;
        if (fused && split)
            fail(ErrorCode::METADATA_MISMATCH,
                 "checkpoint carries both a fused attn_qkv and split q/k/v projections");
        if (!fused && !split)
            fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                 "no attention projection found (neither attn_qkv nor attn_q)");
        cfg.fused_qkv = fused;
    }

    // qwen2 and its descendants add a bias to the query/key/value projections.
    // The checkpoint says so with `attention.qkv_bias`, and the bias tensors
    // must then exist: a declared-but-absent bias would silently change the
    // maths, so the two are cross-checked rather than trusted.
    {
        MetadataValue declared;
        bool wants_bias = false;
        if (opt_metadata(doc, arch + ".attention.qkv_bias", &declared)) wants_bias = declared.as_bool();
        const bool has_bias_tensors =
            cfg.fused_qkv ? doc.tensor_by_name.count(layer_name(probe, "attn_qkv.bias")) != 0
                          : doc.tensor_by_name.count(layer_name(probe, "attn_q.bias")) != 0;
        if (wants_bias && !has_bias_tensors)
            fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                 arch + ".attention.qkv_bias is set but blk.0.attn_q.bias is absent");
        if (has_bias_tensors && !wants_bias && doc.metadata.count(arch + ".attention.qkv_bias"))
            fail(ErrorCode::METADATA_MISMATCH,
                 "attention bias tensors are present but " + arch +
                     ".attention.qkv_bias is false");
        cfg.attn_qkv_bias = has_bias_tensors;
    }
    reject("attention.o_bias", "attention output bias");
    // qwen3 normalises q and k per head before RoPE. The tensors are the only
    // evidence a file uses it, so presence drives the decision and the shape is
    // then validated: the scale is one head's worth, shared by every head.
    {
        const bool has_q_norm = doc.tensor_by_name.count(layer_name(probe, "attn_q_norm.weight")) != 0;
        const bool has_k_norm = doc.tensor_by_name.count(layer_name(probe, "attn_k_norm.weight")) != 0;
        if (has_q_norm != has_k_norm)
            fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                 "q/k normalisation must be present on both q and k, not just one");
        cfg.qk_norm = has_q_norm;
    }

    if (cfg.hidden > 65536 || cfg.layers > 1024 || cfg.heads > 4096 || cfg.head_dim > 1024 ||
        cfg.vocab > (1u << 20) || cfg.context > (1u << 20) || cfg.expert_count > 4096 ||
        (cfg.expert_count && cfg.expert_top_k > cfg.expert_count))
        fail(ErrorCode::SHAPE_MISMATCH, "model dimension exceeds adapter safety bounds");

    // glm4 keeps the FFN gate and up halves in a single tensor. Detected from
    // the shapes rather than the architecture name, because the same adapter
    // serves both layouts.
    if (cfg.expert_count == 0 && !cfg.ffn_no_gate && cfg.has_ffn) {
        const TensorInfo *up = m.tensor(layer_name(probe, "ffn_up.weight"));
        const bool has_gate = doc.tensor_by_name.count(layer_name(probe, "ffn_gate.weight")) != 0;
        if (!up)
            fail(ErrorCode::MISSING_REQUIRED_TENSOR, "missing required tensor blk.0.ffn_up.weight");
        if (!has_gate) {
            if (up->dims.size() != 2 || up->dims[1] != static_cast<uint64_t>(cfg.intermediate) * 2)
                fail(ErrorCode::SHAPE_MISMATCH,
                     "ffn_gate is absent, so ffn_up must be [hidden, 2*intermediate] holding "
                     "both halves");
            cfg.fused_gate_up = true;
        }
    }

    // --- canonical tensor map ------------------------------------------------
    // `name` keeps the bytes' real identity; `canonical` is what the engine
    // addresses. Only the router is renamed, because llama.cpp stores the
    // gate in `ffn_gate_inp` while the reference adapter calls it `ffn_router`.
    std::set<std::string> expected;
    auto expect = [&](const std::string &name) { expected.insert(name); };
    // BitNet's optional projection scales: one element each, multiplied into the
    // projection output. They are claimed here rather than in the per-layer list
    // because a file may carry any subset of them, and an unclaimed tensor would
    // be reported as an adapter gap.
    cfg.projection_scales = false;
    for (const char *suffix : {"attn_q.scale", "attn_k.scale", "attn_v.scale",
                               "attn_output.scale", "ffn_gate.scale", "ffn_up.scale",
                               "ffn_down.scale"}) {
        for (uint32_t l = 0; l < cfg.layers; ++l) {
            const std::string name = layer_name(l, suffix);
            if (!m.tensor(name)) continue;
            const TensorInfo *scale = m.tensor(name);
            if (scale->element_count() != 1)
                fail(ErrorCode::SHAPE_MISMATCH,
                     name + " must hold exactly one element");
            expect(name);
            cfg.projection_scales = true;
        }
    }
    expect("token_embd.weight");
    expect("output_norm.weight");
    // A checkpoint without `output.weight` ties the output projection to the
    // token table. That is a supported layout, not a missing tensor: the maths
    // is identical, so it is aliased rather than duplicated.
    const bool tied_output = m.doc.tensor_by_name.count("output.weight") == 0;
    if (tied_output) m.alias["output.weight"] = "token_embd.weight";
    else expect("output.weight");
    if (cfg.norm_bias) expect("output_norm.bias");
    if (cfg.logit_bias) expect("output.bias");
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        // Every layer opens with the block's entry norm, whatever kind of mixer
        // follows it, so this is claimed before the two mixer branches split.
        expect(layer_name(l, "attn_norm.weight"));
        if (cfg.norm_bias) expect(layer_name(l, "attn_norm.bias"));
        if (cfg.layer_is_recurrent(l)) {
            // The recurrent block, in llama.cpp's tensor names. There is no
            // attention projection and no rotary on such a layer.
            for (const char *suffix : {"ssm_in.weight", "ssm_conv1d.weight",
                                       "ssm_conv1d.bias", "ssm_dt.bias", "ssm_a", "ssm_d",
                                       "ssm_out.weight"})
                expect(layer_name(l, suffix));
            // mamba-1 projects dt, B and C separately; mamba2 folds dt into
            // `ssm_in`, so it has no `ssm_x` and no dt weight.
            if (cfg.ssm_kind != "mamba2") {
                expect(layer_name(l, "ssm_x.weight"));
                expect(layer_name(l, "ssm_dt.weight"));
            }
            if (cfg.ssm_kind == "mamba2") expect(layer_name(l, "ssm_norm.weight"));
            // The dt/B/C norms are claimed when the file carries them, not
            // required: only jamba creates them upstream, and a mamba file that
            // sets `ssm.dt_b_c_rms` without them gets unit-gain norms rather than
            // a missing-tensor error (models/mamba-base.cpp:99 with null
            // weights). Claiming by presence also stops such a tensor from being
            // reported as an adapter gap.
            for (const char *suffix : {"ssm_dt_norm.weight", "ssm_b_norm.weight",
                                       "ssm_c_norm.weight"})
                if (m.tensor(layer_name(l, suffix)) != nullptr) expect(layer_name(l, suffix));
        } else {
            expect(layer_name(l, "attn_output.weight"));
            if (cfg.attn_output_bias) expect(layer_name(l, "attn_output.bias"));
            if (cfg.mla) {
                // Keys and values never exist as weights: each head expands the
                // per-token latent through its own slice of k_b / v_b.
                if (cfg.q_lora_rank != 0)
                    for (const char *suffix :
                         {"attn_q_a.weight", "attn_q_a_norm.weight", "attn_q_b.weight"})
                        expect(layer_name(l, suffix));
                else
                    expect(layer_name(l, "attn_q.weight"));
                for (const char *suffix :
                     {"attn_kv_a_mqa.weight", "attn_kv_a_norm.weight", "attn_k_b.weight",
                      "attn_v_b.weight"})
                    expect(layer_name(l, suffix));
            } else if (cfg.fused_qkv) {
                expect(layer_name(l, "attn_qkv.weight"));
            } else {
                for (const char *suffix : {"attn_q.weight", "attn_k.weight", "attn_v.weight"})
                    expect(layer_name(l, suffix));
            }
            if (cfg.attn_qkv_bias) {
                if (cfg.fused_qkv) {
                    expect(layer_name(l, "attn_qkv.bias"));
                } else {
                    for (const char *suffix : {"attn_q.bias", "attn_k.bias", "attn_v.bias"})
                        expect(layer_name(l, suffix));
                }
            }
            if (cfg.qk_norm)
                for (const char *suffix : {"attn_q_norm.weight", "attn_k_norm.weight"})
                    expect(layer_name(l, suffix));
            if (cfg.attn_sinks) expect(layer_name(l, "attn_sinks.weight"));
            if (cfg.post_norms)
                for (const char *suffix : {"attn_post_norm.weight", "ffn_post_norm.weight"})
                    expect(layer_name(l, suffix));
            if (cfg.sub_norms)
                for (const char *suffix : {"attn_sub_norm.weight", "ffn_sub_norm.weight"})
                    expect(layer_name(l, suffix));
            if (cfg.rope_freqs) expect(layer_name(l, "rope_freqs.weight"));
        }
        // The feed-forward is a per-layer fact too: a pure state-space file has
        // none, a hybrid has one behind both kinds of layer, and jamba mixes
        // dense blocks with routed ones in a single stack.
        if (!cfg.layer_has_ffn(l)) continue;
        if (!cfg.parallel_residual && cfg.ffn_norm_is_post)
            expect(layer_name(l, "attn_post_norm.weight"));
        else if (!cfg.parallel_residual) expect(layer_name(l, "ffn_norm.weight"));
        if (cfg.norm_bias && !cfg.parallel_residual) expect(layer_name(l, "ffn_norm.bias"));
        if (cfg.ffn_bias) {
            expect(layer_name(l, "ffn_up.bias"));
            expect(layer_name(l, "ffn_down.bias"));
        }
        if (cfg.ffn_gate_bias) expect(layer_name(l, "ffn_gate.bias"));
        if (cfg.ffn_norm_ada)
            for (const char *suffix : {"ffn_norm_ada_down.weight", "ffn_norm_ada_up.weight"})
                expect(layer_name(l, suffix));
        if (cfg.layer_is_moe(l)) {
            if (cfg.fused_gate_up_exps) {
                for (const char *suffix : {"ffn_gate_inp.weight", "ffn_gate_up_exps.weight",
                                           "ffn_down_exps.weight"})
                    expect(layer_name(l, suffix));
            } else {
                for (const char *suffix : {"ffn_gate_inp.weight", "ffn_gate_exps.weight",
                                           "ffn_up_exps.weight", "ffn_down_exps.weight"})
                    expect(layer_name(l, suffix));
            }
            if (cfg.shared_expert) {
                for (const char *suffix : {"ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                                           "ffn_down_shexp.weight"})
                    expect(layer_name(l, suffix));
                if (cfg.shared_expert_gate) expect(layer_name(l, "ffn_gate_inp_shexp.weight"));
            }
            // The bias only steers selection, so it is required on MoE layers
            // independently of whether a shared expert exists.
            if (cfg.expert_selection_bias) expect(layer_name(l, "ffn_exp_probs_b.bias"));
            if (cfg.expert_router_bias) expect(layer_name(l, "ffn_gate_inp.bias"));
            if (cfg.expert_projection_bias)
                for (const char *suffix : {"ffn_gate_exps.bias", "ffn_up_exps.bias",
                                           "ffn_down_exps.bias"})
                    expect(layer_name(l, suffix));
        } else if (cfg.fused_gate_up) {
            for (const char *suffix : {"ffn_up.weight", "ffn_down.weight"})
                expect(layer_name(l, suffix));
        } else if (cfg.ffn_no_gate) {
            for (const char *suffix : {"ffn_up.weight", "ffn_down.weight"})
                expect(layer_name(l, suffix));
        } else {
            for (const char *suffix : {"ffn_gate.weight", "ffn_up.weight", "ffn_down.weight"})
                expect(layer_name(l, suffix));
        }
    }
    for (auto &tensor : m.doc.tensors) {
        if (!expected.count(tensor.name)) {
            const char *reason = ignorable_reason(tensor.name);
            if (reason == nullptr)
                fail(ErrorCode::UNKNOWN_TENSOR,
                     "tensor " + tensor.name + " has no mapping in the " + arch +
                         " adapter; no adapter tag claims this name (see `freemoe adapters`)");
            tensor.role = TensorRole::IGNORED;
            tensor.layer = -1;
            tensor.canonical = tensor.name;
            m.ignored_tensors.emplace_back(tensor.name, reason);
            continue;
        }
        if (!tensor.type_known)
            fail(ErrorCode::UNSUPPORTED_QUANTISATION,
                 "tensor " + tensor.name + " uses type " + std::to_string(tensor.type_id) +
                     ", which this build cannot decode");
        if (!tensor_capability(tensor.type_id).execute)
            fail(ErrorCode::UNSUPPORTED_QUANTISATION,
                 "tensor " + tensor.name + " uses type " + tensor.type_name +
                     ", which has no execution kernel");
        uint32_t layer = 0;
        TensorRole role = role_from_name(tensor.name, layer);
        if (role == TensorRole::UNKNOWN && parse_layer_suffix(tensor.name, "ffn_gate_inp.weight", layer))
            role = TensorRole::FFN_ROUTER;
        if (role == TensorRole::UNKNOWN)
            fail(ErrorCode::UNKNOWN_TENSOR, "cannot map tensor role for " + tensor.name);
        tensor.role = role;
        tensor.tag = adapter_tensor_tag(role);
        tensor.op = adapter_tensor_op(role);
        const bool global_role = role == TensorRole::TOKEN_EMBD || role == TensorRole::OUTPUT ||
                                 role == TensorRole::OUTPUT_NORM ||
                                 role == TensorRole::OUTPUT_BIAS ||
                                 role == TensorRole::OUTPUT_NORM_BIAS;
        tensor.layer = global_role ? -1 : static_cast<int>(layer);
        tensor.canonical = role == TensorRole::FFN_ROUTER
                               ? layer_name(layer, "ffn_router.weight")
                               : tensor.name;
    }

    // Every expected tensor must be present: a partially mapped model would
    // silently change the forward pass.
    for (const auto &name : expected) {
        if (!m.doc.tensor_by_name.count(name))
            fail(ErrorCode::MISSING_REQUIRED_TENSOR, "missing required tensor " + name);
    }
    if (tied_output) {
        // The token table must have the output projection's shape.
        const TensorInfo &embd = *m.tensor("token_embd.weight");
        if (embd.dims.size() != 2 || embd.dims[0] != cfg.hidden)
            fail(ErrorCode::SHAPE_MISMATCH,
                 "tied output requires token_embd.weight to be [hidden, vocab]");
    }

    // --- shape validation against the declared hyperparameters ---------------
    auto dims_of = [&](const std::string &name) -> const std::vector<uint64_t> & {
        return m.doc.tensors[m.doc.tensor_by_name.at(name)].dims;
    };
    const uint64_t kv_dim = static_cast<uint64_t>(cfg.kv_heads) * cfg.head_dim;
    expect_dims(*m.tensor("token_embd.weight"), {cfg.hidden, cfg.vocab});
    if (const TensorInfo *t = m.tensor("output.weight"))
        expect_dims(*t, {cfg.hidden, cfg.vocab});
    expect_dims(*m.tensor("output_norm.weight"), {cfg.hidden});
    if (cfg.norm_bias) expect_dims(*m.tensor("output_norm.bias"), {cfg.hidden});
    if (cfg.logit_bias) expect_dims(*m.tensor("output.bias"), {cfg.vocab});
    for (uint32_t l = 0; l < cfg.layers; ++l) {
        expect_dims(*m.tensor(layer_name(l, "attn_norm.weight")), {cfg.hidden});
        if (cfg.layer_is_recurrent(l)) {
            // The shapes the state layout and both scans are derived from. Each
            // is checked against the metadata, so a file whose tensors and keys
            // disagree is refused rather than run with one of the two.
            const uint64_t conv_channels = cfg.ssm_conv_channels();
            expect_dims(*m.tensor(layer_name(l, "ssm_in.weight")),
                        {cfg.hidden, cfg.ssm_kind == "mamba2"
                                         ? static_cast<uint64_t>(cfg.ssm_inner) + conv_channels +
                                               cfg.ssm_dt_rank
                                         : static_cast<uint64_t>(2) * cfg.ssm_inner});
            expect_dims(*m.tensor(layer_name(l, "ssm_conv1d.weight")),
                        {cfg.ssm_conv, conv_channels});
            expect_dims(*m.tensor(layer_name(l, "ssm_conv1d.bias")), {conv_channels});
            expect_dims(*m.tensor(layer_name(l, "ssm_dt.bias")),
                        {cfg.ssm_kind == "mamba2" ? cfg.ssm_dt_rank : cfg.ssm_inner});
            if (cfg.ssm_kind == "mamba2")
                expect_dims(*m.tensor(layer_name(l, "ssm_a")), {1, cfg.ssm_dt_rank});
            else
                expect_dims(*m.tensor(layer_name(l, "ssm_a")), {cfg.ssm_state, cfg.ssm_inner});
            expect_dims(*m.tensor(layer_name(l, "ssm_out.weight")),
                        {cfg.ssm_inner, cfg.hidden});
            if (cfg.ssm_kind == "mamba2") {
                expect_dims(*m.tensor(layer_name(l, "ssm_d")), {1, cfg.ssm_dt_rank});
                expect_dims(*m.tensor(layer_name(l, "ssm_norm.weight")),
                            {cfg.ssm_inner / cfg.ssm_groups, cfg.ssm_groups});
            } else {
                expect_dims(*m.tensor(layer_name(l, "ssm_x.weight")),
                            {cfg.ssm_inner,
                             static_cast<uint64_t>(cfg.ssm_dt_rank) + 2 * cfg.ssm_state});
                expect_dims(*m.tensor(layer_name(l, "ssm_dt.weight")),
                            {cfg.ssm_dt_rank, cfg.ssm_inner});
                expect_dims(*m.tensor(layer_name(l, "ssm_d")), {cfg.ssm_inner});
            }
        } else {
        if (cfg.attn_output_bias)
            expect_dims(*m.tensor(layer_name(l, "attn_output.bias")), {cfg.hidden});
        if (cfg.ffn_bias) {
            expect_dims(*m.tensor(layer_name(l, "ffn_up.bias")), {cfg.intermediate});
            expect_dims(*m.tensor(layer_name(l, "ffn_down.bias")), {cfg.hidden});
        }
        if (cfg.ffn_gate_bias)
            expect_dims(*m.tensor(layer_name(l, "ffn_gate.bias")), {cfg.intermediate});
        if (cfg.ffn_norm_ada) {
            const TensorInfo *down = m.tensor(layer_name(l, "ffn_norm_ada_down.weight"));
            expect_dims(*down, {cfg.hidden, down->dims.size() > 1 ? down->dims[1] : 0});
            expect_dims(*m.tensor(layer_name(l, "ffn_norm_ada_up.weight")),
                        {down->dims[1], cfg.hidden});
        }
        if (cfg.norm_bias && !cfg.parallel_residual)
            expect_dims(*m.tensor(layer_name(l, "ffn_norm.bias")), {cfg.hidden});
        if (cfg.mla) {
            // k_b is [nope, kv_lora_rank, heads] and v_b is
            // [kv_lora_rank, head_v, heads]: one slice per head.
            const uint64_t nope = cfg.head_dim - cfg.rope_dim;
            if (cfg.q_lora_rank != 0) {
                expect_dims(*m.tensor(layer_name(l, "attn_q_a.weight")),
                            {cfg.hidden, cfg.q_lora_rank});
                expect_dims(*m.tensor(layer_name(l, "attn_q_a_norm.weight")), {cfg.q_lora_rank});
                expect_dims(*m.tensor(layer_name(l, "attn_q_b.weight")),
                            {cfg.q_lora_rank, static_cast<uint64_t>(cfg.heads) * cfg.head_dim});
            } else {
                expect_dims(*m.tensor(layer_name(l, "attn_q.weight")),
                            {cfg.hidden, static_cast<uint64_t>(cfg.heads) * cfg.head_dim});
            }
            expect_dims(*m.tensor(layer_name(l, "attn_kv_a_mqa.weight")),
                        {cfg.hidden, static_cast<uint64_t>(cfg.kv_lora_rank) + cfg.rope_dim});
            expect_dims(*m.tensor(layer_name(l, "attn_kv_a_norm.weight")), {cfg.kv_lora_rank});
            expect_dims(*m.tensor(layer_name(l, "attn_k_b.weight")),
                        {nope, cfg.kv_lora_rank, cfg.heads});
            expect_dims(*m.tensor(layer_name(l, "attn_v_b.weight")),
                        {cfg.kv_lora_rank, cfg.head_v_dim(), cfg.heads});
            // The output projection maps the per-head value widths, not `hidden`.
            expect_dims(*m.tensor(layer_name(l, "attn_output.weight")),
                        {static_cast<uint64_t>(cfg.heads) * cfg.head_v_dim(), cfg.hidden});
        } else {
            if (cfg.fused_qkv) {
                // The fused projection holds q, then k, then v along its output axis.
                expect_dims(*m.tensor(layer_name(l, "attn_qkv.weight")),
                            {cfg.hidden, cfg.hidden + 2 * kv_dim});
            } else {
                expect_dims(*m.tensor(layer_name(l, "attn_q.weight")), {cfg.hidden, cfg.hidden});
                expect_dims(*m.tensor(layer_name(l, "attn_k.weight")), {cfg.hidden, kv_dim});
                expect_dims(*m.tensor(layer_name(l, "attn_v.weight")), {cfg.hidden, kv_dim});
            }
            expect_dims(*m.tensor(layer_name(l, "attn_output.weight")), {cfg.hidden, cfg.hidden});
        }
        if (cfg.post_norms) {
            expect_dims(*m.tensor(layer_name(l, "attn_post_norm.weight")), {cfg.hidden});
            expect_dims(*m.tensor(layer_name(l, "ffn_post_norm.weight")), {cfg.hidden});
        }
        if (cfg.qk_norm) {
            expect_dims(*m.tensor(layer_name(l, "attn_q_norm.weight")), {cfg.head_dim});
            expect_dims(*m.tensor(layer_name(l, "attn_k_norm.weight")), {cfg.head_dim});
        }
        // One sink logit per query head -- not per kv head: the sink is applied
        // after the query head's own softmax.
        if (cfg.attn_sinks)
            expect_dims(*m.tensor(layer_name(l, "attn_sinks.weight")), {cfg.heads});
        // One divisor per rotating pair.
        if (cfg.rope_freqs &&
            m.doc.tensor_by_name.count(layer_name(l, "rope_freqs.weight")) != 0)
            expect_dims(*m.tensor(layer_name(l, "rope_freqs.weight")),
                        {static_cast<uint64_t>(std::max<uint32_t>(cfg.rope_dim, cfg.head_dim)) / 2});
        if (cfg.attn_qkv_bias) {
            if (cfg.fused_qkv) {
                expect_dims(*m.tensor(layer_name(l, "attn_qkv.bias")),
                            {cfg.hidden + 2 * kv_dim});
            } else {
                expect_dims(*m.tensor(layer_name(l, "attn_q.bias")), {cfg.hidden});
                expect_dims(*m.tensor(layer_name(l, "attn_k.bias")), {kv_dim});
                expect_dims(*m.tensor(layer_name(l, "attn_v.bias")), {kv_dim});
            }
        }
        }  // end of the attention shapes
        // The entry norm's bias is on every layer; the feed-forward half is only
        // on the layers that carry one (all of them, except in a pure mamba file).
        if (cfg.norm_bias) expect_dims(*m.tensor(layer_name(l, "attn_norm.bias")), {cfg.hidden});
        if (!cfg.layer_has_ffn(l)) continue;
        if (!cfg.parallel_residual)
            expect_dims(*m.tensor(layer_name(l, cfg.ffn_norm_is_post ? "attn_post_norm.weight"
                                                                   : "ffn_norm.weight")),
                        {cfg.hidden});
        if (cfg.ffn_bias) {
            expect_dims(*m.tensor(layer_name(l, "ffn_up.bias")), {cfg.intermediate});
            expect_dims(*m.tensor(layer_name(l, "ffn_down.bias")), {cfg.hidden});
        }
        if (cfg.ffn_gate_bias)
            expect_dims(*m.tensor(layer_name(l, "ffn_gate.bias")), {cfg.intermediate});
        if (cfg.ffn_norm_ada) {
            const TensorInfo *down = m.tensor(layer_name(l, "ffn_norm_ada_down.weight"));
            expect_dims(*down, {cfg.hidden, down->dims.size() > 1 ? down->dims[1] : 0});
            expect_dims(*m.tensor(layer_name(l, "ffn_norm_ada_up.weight")),
                        {down->dims[1], cfg.hidden});
        }
        if (cfg.norm_bias && !cfg.parallel_residual)
            expect_dims(*m.tensor(layer_name(l, "ffn_norm.bias")), {cfg.hidden});
        const bool moe_layer = cfg.layer_is_moe(l);
        if (moe_layer) {
            const uint32_t ewidth = cfg.expert_width();
            expect_dims(*m.tensor(layer_name(l, "ffn_gate_inp.weight")),
                        {cfg.hidden, cfg.expert_count});
            if (cfg.expert_selection_bias)
                expect_dims(*m.tensor(layer_name(l, "ffn_exp_probs_b.bias")),
                            {cfg.expert_count});
            if (cfg.expert_router_bias)
                expect_dims(*m.tensor(layer_name(l, "ffn_gate_inp.bias")), {cfg.expert_count});
            if (cfg.expert_projection_bias) {
                // One column per expert, matching the projection it biases.
                expect_dims(*m.tensor(layer_name(l, "ffn_gate_exps.bias")),
                            {ewidth, cfg.expert_count});
                expect_dims(*m.tensor(layer_name(l, "ffn_up_exps.bias")),
                            {ewidth, cfg.expert_count});
                expect_dims(*m.tensor(layer_name(l, "ffn_down_exps.bias")),
                            {cfg.hidden, cfg.expert_count});
            }
            if (cfg.fused_gate_up_exps) {
                // [hidden, 2*width, experts]: gate block then up block per expert.
                expect_dims(*m.tensor(layer_name(l, "ffn_gate_up_exps.weight")),
                            {cfg.hidden, static_cast<uint64_t>(ewidth) * 2, cfg.expert_count});
            } else {
                for (const char *suffix : {"ffn_gate_exps.weight", "ffn_up_exps.weight"})
                    expect_dims(*m.tensor(layer_name(l, suffix)),
                                {cfg.hidden, ewidth, cfg.expert_count});
            }
            expect_dims(*m.tensor(layer_name(l, "ffn_down_exps.weight")),
                        {ewidth, cfg.hidden, cfg.expert_count});
            if (cfg.shared_expert) {
                const uint32_t swidth = cfg.shared_expert_intermediate;
                // One scalar per token, so the gate is a hidden-vector.
                if (cfg.shared_expert_gate)
                    expect_dims(*m.tensor(layer_name(l, "ffn_gate_inp_shexp.weight")),
                                {cfg.hidden});
                for (const char *suffix : {"ffn_gate_shexp.weight", "ffn_up_shexp.weight"})
                    expect_dims(*m.tensor(layer_name(l, suffix)), {cfg.hidden, swidth});
                expect_dims(*m.tensor(layer_name(l, "ffn_down_shexp.weight")),
                            {swidth, cfg.hidden});
            }
        } else if (cfg.ffn_no_gate) {
            expect_dims(*m.tensor(layer_name(l, "ffn_up.weight")), {cfg.hidden, cfg.intermediate});
            expect_dims(*m.tensor(layer_name(l, "ffn_down.weight")),
                        {cfg.intermediate, cfg.hidden});
        } else if (cfg.fused_gate_up) {
            expect_dims(*m.tensor(layer_name(l, "ffn_up.weight")),
                        {cfg.hidden, static_cast<uint64_t>(cfg.intermediate) * 2});
        } else {
            for (const char *suffix : {"ffn_gate.weight", "ffn_up.weight"})
                expect_dims(*m.tensor(layer_name(l, suffix)), {cfg.hidden, cfg.intermediate});
            expect_dims(*m.tensor(layer_name(l, "ffn_down.weight")),
                        {cfg.intermediate, cfg.hidden});
        }
    }
    (void)dims_of;

    // --- token table ---------------------------------------------------------
    MetadataValue tokens;
    if (!opt_metadata(doc, "tokenizer.ggml.tokens", &tokens) || !tokens.is_array())
        fail(ErrorCode::INVALID_TOKENIZER, "checkpoint carries no token table");
    if (tokens.array.size() != cfg.vocab)
        fail(ErrorCode::INVALID_TOKENIZER, "token table size does not match vocabulary size");
    for (const auto &token : tokens.array) {
        if (!token.is_string()) fail(ErrorCode::INVALID_TOKENIZER, "non-string tokenizer token");
        m.tokenizer_tokens.push_back(token.s);
    }

    // --- residency layout ----------------------------------------------------
    for (const auto &tensor : m.doc.tensors)
        if (!tensor.expert_tensor) m.fixed_weight_bytes += tensor.byte_size;

    if (cfg.expert_count) {
        // Each expert's slice of a stacked tensor must be a whole number of
        // quantisation blocks, otherwise a slice would start or end mid-block.
        const uint32_t expert_count = cfg.expert_count;
        // The record always carries three extents; the fused layout splits its
        // single gate_up tensor into two of them, so everything downstream (the
        // slot layout, the expert reader) is unchanged.
        const std::array<TensorRole, 3> roles = {TensorRole::FFN_GATE_EXPS,
                                                 TensorRole::FFN_UP_EXPS,
                                                 TensorRole::FFN_DOWN_EXPS};
        const std::array<std::string, 3> suffixes =
            cfg.fused_gate_up_exps
                ? std::array<std::string, 3>{"ffn_gate_up_exps.weight",
                                             "ffn_gate_up_exps.weight",
                                             "ffn_down_exps.weight"}
                : std::array<std::string, 3>{"ffn_gate_exps.weight",
                                             "ffn_up_exps.weight",
                                             "ffn_down_exps.weight"};
        // Which layers carry stacked experts is per layer: the leading blocks of
        // a deepseek checkpoint are plain FFNs, and a hybrid can place its
        // experts on one kind of layer only.
        for (uint32_t l = 0; l < cfg.layers; ++l) {
            if (!cfg.layer_is_moe(l)) continue;
            // Each projection may use a different quantisation type (Q4_K gate
            // and up, Q6_K down is common), so their per-expert strides differ
            // legitimately. The record carries three independent extents and the
            // loader concatenates them in role order.
            std::array<uint64_t, 3> strides{};
            for (int r = 0; r < 3; ++r) {
                const TensorInfo &tensor = *m.tensor(layer_name(l, suffixes[r]));
                if (tensor.byte_size % expert_count != 0)
                    fail(ErrorCode::SHAPE_MISMATCH, "stacked expert tensor is not evenly divisible");
                uint64_t slice = tensor.byte_size / expert_count;
                if (cfg.fused_gate_up_exps && r < 2) slice /= 2;  // gate half, up half
                strides[r] = slice;
                const uint64_t rows = cfg.fused_gate_up_exps && r < 2
                                          ? tensor.dims[0] * tensor.dims[1] / (2 * expert_count)
                                          : tensor.dims[0] * tensor.dims[1] / expert_count;
                if (tensor.block_size > 1 && rows % tensor.block_size != 0)
                    fail(ErrorCode::SHAPE_MISMATCH,
                         "expert slice is not a whole number of quantisation blocks");
            }
            for (uint32_t e = 0; e < expert_count; ++e) {
                ExpertRecord rec;
                rec.layer = l;
                rec.expert = e;
                rec.payload_bytes = strides[0] + strides[1] + strides[2];
                rec.checksum_known = false; // vendor files publish no checksums
                for (int r = 0; r < 3; ++r) {
                    const TensorInfo &tensor = *m.tensor(layer_name(l, suffixes[r]));
                    // A fused tensor's expert slice starts at e * (2*half); its
                    // gate half is first and its up half follows immediately.
                    const uint64_t base =
                        tensor.absolute_offset + e * (cfg.fused_gate_up_exps && r < 2
                                                          ? strides[0] * 2
                                                          : strides[r]);
                    const uint64_t offset = r == 1 && cfg.fused_gate_up_exps ? base + strides[0] : base;
                    rec.extents[r] = {roles[r], offset, strides[r]};
                }
                m.expert_index.emplace(ModelManifest::expert_key(l, e), m.experts.size());
                m.experts.push_back(rec);
                m.slot_bytes = std::max(m.slot_bytes, rec.payload_bytes);
            }
        }
        for (auto &tensor : m.doc.tensors) {
            bool expert = false;
            for (uint32_t l = 0; l < cfg.layers && !expert; ++l) {
                if (!cfg.layer_is_moe(l)) continue;
                for (const auto &suffix : suffixes)
                    if (tensor.name == layer_name(l, suffix)) expert = true;
            }
            tensor.expert_tensor = expert;
        }
        uint64_t expert_bytes = 0;
        for (const auto &tensor : m.doc.tensors)
            if (tensor.expert_tensor) expert_bytes += tensor.byte_size;
        m.expert_tensor_bytes = expert_bytes;
        // The manifest above summed every tensor; rebuild it without experts.
        m.fixed_weight_bytes = 0;
        for (const auto &tensor : m.doc.tensors)
            if (!tensor.expert_tensor) m.fixed_weight_bytes += tensor.byte_size;
    }
    return m;
}

ModelManifest build_manifest(const GGUFDocument &doc, const std::string &source_path) {
    {
        const std::string arch = require_string(doc, "general.architecture");
        if (arch != "freemoe-tiny") {
            if (!is_llama_family(arch))
                fail(ErrorCode::UNSUPPORTED_ARCHITECTURE,
                     "no adapter for architecture '" + arch + "'; run `freemoe probe` for the "
                     "exact blockers");
            return build_manifest_llama(doc, source_path, arch);
        }
    }
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
    // The reference fixture is defined as normalising its router weights over
    // the selected experts. Stating it here keeps the fixture's semantics a
    // property of the adapter instead of an implicit engine default, which is
    // what the real models need (they do not renormalise).
    cfg.expert_weights_norm = true;
    cfg.expert_weights_scale = 1.0;
    cfg.bos_id = static_cast<uint32_t>(require_u64(doc, "tokenizer.ggml.bos_token_id"));
    cfg.eos_id = static_cast<uint32_t>(require_u64(doc, "tokenizer.ggml.eos_token_id"));
    cfg.rms_eps = require_double(doc, "freemoe.rms_norm_epsilon");
    cfg.rope_theta = require_double(doc, "freemoe.rope_theta");
    cfg.shared_expert = require_bool(doc, "freemoe.shared_expert");
    // The reference fixture predates `rope.scaling.*` and defines unscaled
    // rotation; resolving keeps the derived fields defined rather than defaulted
    // by luck.
    cfg.rope_n_ctx_orig = static_cast<double>(cfg.context);
    resolve_rope_scaling(cfg);

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

    const MetadataValue *tokens = require_metadata_ptr(doc, "tokenizer.ggml.tokens");
    if (!tokens->is_array() || tokens->array.size() != cfg.vocab ||
        tokens->element_type != GGUFValueType::STRING)
        fail(ErrorCode::INVALID_TOKENIZER, "tokenizer token table has wrong shape");
    for (const auto &token : tokens->array) {
        if (!token.is_string()) fail(ErrorCode::INVALID_TOKENIZER, "non-string tokenizer token");
        m.tokenizer_tokens.push_back(token.s);
    }

    const MetadataValue *checksums = require_metadata_ptr(doc, "freemoe.expert_checksums");
    if (!checksums->is_array() || checksums->element_type != GGUFValueType::UINT64)
        fail(ErrorCode::METADATA_MISMATCH, "expert checksum array has wrong type");

    bool moe = cfg.model_kind == "moe";
    if (!moe && (cfg.expert_count != 0 || cfg.expert_top_k != 0 || !checksums->array.empty() ||
                 cfg.expert_layout != "none"))
        fail(ErrorCode::METADATA_MISMATCH, "dense model declares MoE metadata");
    if (moe && (cfg.expert_count == 0 || cfg.expert_top_k == 0 ||
                cfg.expert_top_k > cfg.expert_count || cfg.expert_layout != "stacked-f32"))
        fail(ErrorCode::METADATA_MISMATCH, "MoE model has inconsistent expert metadata");
    if (moe && checksums->array.size() != static_cast<size_t>(cfg.layers) * cfg.expert_count)
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
        if (tensor.canonical.empty()) tensor.canonical = tensor.name;
        uint32_t layer = 0;
        tensor.role = role_from_name(tensor.name, layer);
        tensor.tag = adapter_tensor_tag(tensor.role);
        tensor.op = adapter_tensor_op(tensor.role);
        bool global_role = tensor.role == TensorRole::TOKEN_EMBD ||
                           tensor.role == TensorRole::OUTPUT ||
                           tensor.role == TensorRole::OUTPUT_NORM;
        tensor.layer = global_role ? -1 : static_cast<int>(layer);
        if (tensor.role == TensorRole::UNKNOWN)
            fail(ErrorCode::UNKNOWN_TENSOR, "cannot map tensor role for " + tensor.name);
        // Execution requires a kernel for this type. Parsing alone is never
        // enough to run a model, so this fails closed with the type named.
        {
            const auto &qtable = quantisation_table();
            auto qit = qtable.find(tensor.type_id);
            const bool known = qit != qtable.end();
            const bool k_block = known && quant_is_k_block(tensor.type_id);
            const bool implemented =
                known && (tensor.type_id == 0 || tensor.type_id == 1 || tensor.type_id == 30 ||
                          tensor.type_id == 2 || tensor.type_id == 3 || tensor.type_id == 6 ||
                          tensor.type_id == 7 || tensor.type_id == 8 || tensor.type_id == 9 ||
                          k_block);
            if (!implemented)
                fail(ErrorCode::UNSUPPORTED_QUANTISATION,
                     "no execution kernel for tensor type " +
                         (known ? qit->second.name : std::to_string(tensor.type_id)) +
                         " in " + tensor.name);
        }
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
                uint64_t elements = checked_mul(tensor.dims[0], tensor.dims[1], "expert stride");
                if (elements % tensor.block_size != 0)
                    fail(ErrorCode::SHAPE_MISMATCH, "expert slice is not whole quantisation blocks");
                uint64_t stride =
                    checked_mul(elements / tensor.block_size, tensor.block_bytes, "expert bytes");
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
                rec.checksum = checksums->array[checksum_index].as_u64();
                rec.checksum_known = true;
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
    // `inspect` must describe any checkpoint, including the many that omit
    // optional keys. Every key that exists is reported; the rest are listed in
    // `absent` rather than silently dropped or turned into a hard failure.
    static const std::vector<std::string> keys = {
        "general.architecture", "general.name", "general.alignment",
        "general.file_type", "general.quantization_version", "general.basename",
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
    Json absent = Json::array();
    for (const auto &key : keys) {
        auto it = manifest.doc.metadata.find(key);
        if (it == manifest.doc.metadata.end()) {
            if (key == "general.alignment") continue; // optional, defaults to 32
            absent.push(Json::string(key));
            continue;
        }
        out.set(key, it->second.display());
    }
    // general.alignment is optional in GGUF and defaults to 32.
    if (!manifest.doc.metadata.count("general.alignment")) out.set("general.alignment", "32");
    out.set("absent_keys", std::move(absent));

    // Architecture-prefixed dims drive everything real: expose them under stable
    // names so callers do not have to know the architecture.
    const std::string arch = [&] {
        auto it = manifest.doc.metadata.find("general.architecture");
        return it == manifest.doc.metadata.end() ? std::string() : it->second.as_string();
    }();
    if (!arch.empty()) {
        const std::pair<const char *, const char *> dims[] = {
            {"block_count", "layer_count"},
            {"embedding_length", "hidden_size"},
            {"feed_forward_length", "intermediate_size"},
            {"attention.head_count", "head_count"},
            {"attention.head_count_kv", "head_count_kv"},
            {"attention.key_length", "key_length"},
            {"attention.value_length", "value_length"},
            {"rope.dimension_count", "rope_dimension_count"},
            {"rope.freq_base", "rope_freq_base"},
            {"rope.scaling.type", "rope_scaling"},
            {"rope.scaling.factor", "rope_scaling_factor"},
            {"rope.scaling.original_context_length", "rope_original_context_length"},
            {"attention.layer_norm_rms_epsilon", "rms_epsilon"},
            {"context_length", "context_length"},
            {"expert_count", "expert_count"},
            {"expert_used_count", "expert_used_count"},
            {"vocab_size", "vocab_size"},
        };
        Json derived = Json::object();
        for (const auto &[gguf_key, out_key] : dims) {
            auto it = manifest.doc.metadata.find(arch + "." + gguf_key);
            if (it != manifest.doc.metadata.end()) derived.set(out_key, it->second.display());
        }
        auto tok = manifest.doc.metadata.find("tokenizer.ggml.model");
        if (tok != manifest.doc.metadata.end()) derived.set("tokenizer_model", tok->second.display());
        out.set("architecture_fields", std::move(derived));
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

    // Section 6.3 canonical model map: the internal representation that joins
    // the GGUF source to the engine's logical roles. One entry per source
    // tensor, carrying everything needed to page it and to prove the mapping.
    Json model_map = Json::array();
    for (const TensorInfo *tensor : sorted) {
        Json entry = Json::object();
        entry.set("source_name", tensor->name);
        // The engine's address for this tensor. For most files it equals the
        // source name; adapters rename where an architecture uses a different
        // convention (ffn_gate_inp -> ffn_router).
        entry.set("canonical_name",
                  tensor->canonical.empty() ? tensor->name : tensor->canonical);
        entry.set("role", tensor_role_name(tensor->role));
        // The adapter tag and the operation it feeds, in llama.cpp's vocabulary.
        // Two adapters can map the same source name to different roles, so the
        // tag is the thing to compare in a bug report.
        entry.set("tag", tensor->tag);
        entry.set("op", tensor->op);
        entry.set("type", tensor->type_name);
        // Whether the type byte was recognised and whether `bytes` is the
        // decoder's own figure or was recovered from the neighbouring tensor.
        entry.set("type_known", tensor->type_known);
        entry.set("byte_size_derived", tensor->byte_size_derived);

        Json dims = Json::array();
        for (uint64_t d : tensor->dims) dims.push(Json::integer(static_cast<int64_t>(d)));
        entry.set("dims", std::move(dims));
        entry.set("element_count", tensor->element_count());
        Json geometry = Json::object();
        geometry.set("block_size", static_cast<int64_t>(tensor->block_size));
        geometry.set("block_bytes", static_cast<int64_t>(tensor->block_bytes));
        const uint64_t blocks =
            tensor->block_size ? tensor->element_count() / tensor->block_size : 0;
        geometry.set("block_count", blocks);
        entry.set("quant_geometry", std::move(geometry));

        // Exact source byte range.
        entry.set("source_offset", tensor->absolute_offset);
        entry.set("source_end", tensor->absolute_offset + tensor->byte_size);
        entry.set("bytes", tensor->byte_size);

        // Layer is only reported when the adapter verified it.
        entry.set("layer", tensor->layer < 0 ? Json::null()
                                            : Json::integer(tensor->layer));
        if (tensor->expert_tensor) {
            // Stacked expert tensors hold many experts; the expert index is not
            // parsed from the name, it comes from the adapter's stacking rule.
            entry.set("expert_count", static_cast<int64_t>(cfg.expert_count));
            const uint64_t stride =
                cfg.expert_count ? tensor->byte_size / cfg.expert_count : 0;
            entry.set("expert_stride_bytes", stride);
            // Each expert slice is pageable on its own only when the stride is a
            // whole number of quantisation blocks, so a read never cuts a block.
            entry.set("independently_pageable",
                      tensor->block_bytes != 0 && stride % tensor->block_bytes == 0);
        } else {
            entry.set("expert", Json::null());
            entry.set("expert_count", Json::integer(0));
            entry.set("independently_pageable", true);
        }
        entry.set("adapter_id", cfg.adapter);
        entry.set("adapter_version", Json::integer(1));
        model_map.push(std::move(entry));
    }
    out.set("model_map", std::move(model_map));
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

// ---------------------------------------------------------------------------
// Tensor type capabilities, dequantisation and generic GGUF inspection
// ---------------------------------------------------------------------------

namespace {
const char *value_type_name(GGUFValueType type) {
    switch (type) {
    case GGUFValueType::UINT8: return "u8";
    case GGUFValueType::INT8: return "i8";
    case GGUFValueType::UINT16: return "u16";
    case GGUFValueType::INT16: return "i16";
    case GGUFValueType::UINT32: return "u32";
    case GGUFValueType::INT32: return "i32";
    case GGUFValueType::FLOAT32: return "f32";
    case GGUFValueType::BOOL: return "bool";
    case GGUFValueType::STRING: return "string";
    case GGUFValueType::ARRAY: return "array";
    case GGUFValueType::UINT64: return "u64";
    case GGUFValueType::INT64: return "i64";
    case GGUFValueType::FLOAT64: return "f64";
    }
    return "unknown";
}
} // namespace

TensorCapability tensor_capability(uint32_t type_id) {
    TensorCapability cap;
    auto it = quantisation_table().find(type_id);
    if (it == quantisation_table().end()) {
        cap.note = "unknown tensor type";
        return cap;
    }
    // Any type whose geometry is known can be bounded and skipped safely.
    cap.parse = true;
    switch (type_id) {
    case 0: // F32
        cap.dequantize = true;
        cap.execute = true;
        cap.note = "native F32 kernel";
        break;
    case 1: case 30: // F16, BF16
    case 2: case 3: case 6: case 7:  // Q4_0, Q4_1, Q5_0, Q5_1
    case 8: case 9:                  // Q8_0, Q8_1
    case 10: case 11: case 12: case 13: case 14: case 15: // Q2_K..Q8_K
        cap.dequantize = true;
        cap.execute = true;
        cap.note = "block decoder plus quantised GEMV kernel";
        break;
    case 16: case 17: case 18: case 19: case 21: case 22: case 23: case 29:
        // The IQ family: grid-indexed codecs with sign bits and packed scales.
        cap.dequantize = true;
        cap.execute = true;
        cap.note = "grid-indexed IQ block decoder plus quantised GEMV kernel";
        break;
    case 20: // IQ4_NL: 4-bit non-linear lookup, decode + execute
        cap.dequantize = true;
        cap.execute = true;
        cap.note = "non-linear 4-bit block decoder plus quantised GEMV kernel";
        break;
    case 39: case 40: // MXFP4, NVFP4: FP4 with a per-block exponent scale
        cap.dequantize = true;
        cap.execute = true;
        cap.note = "FP4 block decoder (E8M0/UE4M3 scale) plus quantised GEMV kernel";
        break;
    case 24: case 25: case 26: case 27: case 28: // I8..F64
        cap.dequantize = true;
        cap.execute = true;
        cap.note = "integer/float element type, widened on read";
        break;
    case 34: case 35: case 41: case 42: case 142: // TQ1_0, TQ2_0, Q1_0, Q2_0, PQ2_0
        cap.dequantize = true;
        cap.execute = true;
        cap.note = "ternary/low-bit block decoder plus quantised GEMV kernel";
        break;
    case 143: // PTQ1_0
        // The layout is parsed so the file can be inspected and its expert
        // ranges computed, but the element order is not decoded. PrismML's CPU
        // reference and their Vulkan shader disagree about how many elements a
        // 128-wide block holds, so any decoder written today would be a guess.
        cap.dequantize = false;
        cap.execute = false;
        cap.note = "geometry only: the element layout is not verifiable from the "
                   "published sources, so it is not decoded";
        break;
    default:
        cap.note = "parsed geometry only; block decoder not implemented";
        break;
    }
    return cap;
}

void dequantize_row(uint32_t type_id, const uint8_t *src, float *dst, uint64_t count) {
    dequantize_tensor_region(type_id, src, dst, count);
}

Json generic_inspect_json(const GGUFDocument &doc, const std::string &source_path) {
    Json out = Json::object();
    out.set("schema", "freemoe.generic-inspect.v1");
    out.set("status", "parsed");
    out.set("source_path", source_path);
    out.set("file_size", doc.file_size);
    out.set("gguf_version", static_cast<int64_t>(doc.version));
    out.set("alignment", static_cast<int64_t>(doc.alignment));
    out.set("metadata_count", doc.metadata_count);
    out.set("tensor_count", doc.tensor_count);
    out.set("data_offset", doc.data_offset);
    out.set("source_fingerprint", hex64(doc.source_fingerprint));

    // Architecture is reported when present, but its absence is not an error:
    // this view never requires an adapter.
    auto arch = doc.metadata.find("general.architecture");
    out.set("architecture",
            arch == doc.metadata.end() ? std::string("<none>") : arch->second.display());
    auto name = doc.metadata.find("general.name");
    out.set("name", name == doc.metadata.end() ? std::string("<none>") : name->second.display());

    Json metadata = Json::array();
    for (const std::string &key : doc.metadata_order) {
        auto it = doc.metadata.find(key);
        if (it == doc.metadata.end()) continue;
        Json row = Json::object();
        row.set("key", key);
        row.set("type", value_type_name(it->second.type));
        row.set("value", it->second.display());
        metadata.push(std::move(row));
    }
    out.set("metadata", std::move(metadata));

    Json tensors = Json::array();
    std::map<std::string, uint64_t> bytes_by_type;
    uint64_t total_bytes = 0;
    for (const TensorInfo &tensor : doc.tensors) {
        Json row = Json::object();
        row.set("name", tensor.name);
        Json dims = Json::array();
        for (uint64_t d : tensor.dims) dims.push(Json::integer(static_cast<int64_t>(d)));
        row.set("dims", std::move(dims));
        row.set("type", tensor.type_name);
        row.set("type_id", static_cast<int64_t>(tensor.type_id));
        row.set("block_size", static_cast<int64_t>(tensor.block_size));
        row.set("block_bytes", static_cast<int64_t>(tensor.block_bytes));
        row.set("byte_size", tensor.byte_size);
        row.set("absolute_offset", tensor.absolute_offset);
        TensorCapability cap = tensor_capability(tensor.type_id);
        row.set("parse", cap.parse);
        row.set("dequantize", cap.dequantize);
        row.set("execute", cap.execute);
        tensors.push(std::move(row));
        bytes_by_type[tensor.type_name] += tensor.byte_size;
        total_bytes += tensor.byte_size;
    }
    out.set("tensors", std::move(tensors));
    out.set("tensor_bytes", total_bytes);

    Json by_type = Json::array();
    for (const auto &entry : bytes_by_type) {
        Json row = Json::object();
        row.set("type", entry.first);
        row.set("bytes", entry.second);
        by_type.push(std::move(row));
    }
    out.set("bytes_by_type", std::move(by_type));

    // Support classification is deliberately conservative: parsing a container
    // is never reported as architecture or execution support.
    bool all_executable = !doc.tensors.empty();
    bool all_dequantizable = !doc.tensors.empty();
    for (const TensorInfo &tensor : doc.tensors) {
        TensorCapability cap = tensor_capability(tensor.type_id);
        all_executable = all_executable && cap.execute;
        all_dequantizable = all_dequantizable && cap.dequantize;
    }
    std::string classification = "inspect-only";
    if (arch != doc.metadata.end() && arch->second.display() == "freemoe-tiny" && all_executable)
        classification = "execute-cpu";
    else if (!doc.tensors.empty() && all_dequantizable && !all_executable)
        classification = "dequantize-only";
    out.set("container_support", "parse");
    out.set("tensor_support", all_executable ? "execute" : (all_dequantizable ? "dequantize" : "parse"));
    out.set("classification", classification);
    out.set("adapter_available", arch != doc.metadata.end() &&
                                    arch->second.display() == "freemoe-tiny");
    return out;
}

namespace {

void put_u32(std::vector<uint8_t> &out, uint32_t v) {
    for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

void put_u64(std::vector<uint8_t> &out, uint64_t v) {
    for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>((v >> (8 * i)) & 0xFF));
}

void put_string(std::vector<uint8_t> &out, const std::string &s) {
    put_u64(out, s.size());
    out.insert(out.end(), s.begin(), s.end());
}

void put_value(std::vector<uint8_t> &out, const MetadataValue &value);

void put_array(std::vector<uint8_t> &out, const std::vector<MetadataValue> &items,
               GGUFValueType element_type) {
    put_u32(out, static_cast<uint32_t>(element_type));
    put_u64(out, items.size());
    for (const MetadataValue &item : items) put_value(out, item);
}

void put_value(std::vector<uint8_t> &out, const MetadataValue &value) {
    switch (value.type) {
    case GGUFValueType::UINT8: out.push_back(static_cast<uint8_t>(value.u)); break;
    case GGUFValueType::INT8: out.push_back(static_cast<uint8_t>(value.i)); break;
    case GGUFValueType::UINT16: {
        uint16_t v = static_cast<uint16_t>(value.u);
        out.push_back(static_cast<uint8_t>(v & 0xFF));
        out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
        break;
    }
    case GGUFValueType::INT16: {
        uint16_t v = static_cast<uint16_t>(value.i);
        out.push_back(static_cast<uint8_t>(v & 0xFF));
        out.push_back(static_cast<uint8_t>((v >> 8) & 0xFF));
        break;
    }
    case GGUFValueType::UINT32: put_u32(out, static_cast<uint32_t>(value.u)); break;
    case GGUFValueType::INT32: put_u32(out, static_cast<uint32_t>(value.i)); break;
    case GGUFValueType::FLOAT32: {
        float v = static_cast<float>(value.f);
        uint32_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        put_u32(out, bits);
        break;
    }
    case GGUFValueType::BOOL: out.push_back(value.b ? 1 : 0); break;
    case GGUFValueType::STRING: put_string(out, value.s); break;
    case GGUFValueType::ARRAY: put_array(out, value.array, value.element_type); break;
    case GGUFValueType::UINT64: put_u64(out, value.u); break;
    case GGUFValueType::INT64: put_u64(out, static_cast<uint64_t>(value.i)); break;
    case GGUFValueType::FLOAT64: {
        double v = value.f;
        uint64_t bits;
        std::memcpy(&bits, &v, sizeof(bits));
        put_u64(out, bits);
        break;
    }
    }
}

} // namespace

std::vector<uint8_t> write_gguf(const GGUFDocument &doc, const uint8_t *source,
                                size_t source_size) {
    if (!source) fail(ErrorCode::INVALID_ARGUMENT, "write_gguf requires the source buffer");
    const uint64_t alignment = doc.alignment ? doc.alignment : 32;
    if ((alignment & (alignment - 1)) != 0)
        fail(ErrorCode::INVALID_ARGUMENT, "write_gguf alignment must be a power of two");

    std::vector<uint8_t> out;
    out.reserve(static_cast<size_t>(doc.file_size) + alignment);
    out.insert(out.end(), {'G', 'G', 'U', 'F'});
    put_u32(out, doc.version);
    put_u64(out, doc.tensor_count);
    put_u64(out, doc.metadata_count);

    for (const std::string &key : doc.metadata_order) {
        auto it = doc.metadata.find(key);
        if (it == doc.metadata.end())
            fail(ErrorCode::INTERNAL_ERROR, "metadata order references a missing key");
        put_string(out, key);
        put_u32(out, static_cast<uint32_t>(it->second.type));
        put_value(out, it->second);
    }
    if (doc.metadata_order.size() != doc.metadata.size())
        fail(ErrorCode::INTERNAL_ERROR, "metadata order is incomplete");

    // Tensor payloads are laid out from the data section, in directory order.
    std::vector<uint64_t> relative;
    uint64_t cursor = 0;
    for (const TensorInfo &tensor : doc.tensors) {
        cursor = align_up(cursor, alignment);
        relative.push_back(cursor);
        cursor += tensor.byte_size;
    }
    for (size_t i = 0; i < doc.tensors.size(); ++i) {
        const TensorInfo &tensor = doc.tensors[i];
        put_string(out, tensor.name);
        put_u32(out, static_cast<uint32_t>(tensor.dims.size()));
        for (uint64_t d : tensor.dims) put_u64(out, d);
        put_u32(out, tensor.type_id);
        put_u64(out, relative[i]);
    }

    const uint64_t data_offset = align_up(out.size(), alignment);
    out.resize(static_cast<size_t>(data_offset), 0);
    for (size_t i = 0; i < doc.tensors.size(); ++i) {
        const TensorInfo &tensor = doc.tensors[i];
        if (tensor.absolute_offset + tensor.byte_size > source_size)
            fail(ErrorCode::SHAPE_MISMATCH, "tensor payload outside the source buffer: " + tensor.name);
        const size_t at = static_cast<size_t>(data_offset + relative[i]);
        if (out.size() < at) out.resize(at, 0);
        out.insert(out.end(), source + tensor.absolute_offset,
                   source + tensor.absolute_offset + tensor.byte_size);
    }
    return out;
}

} // namespace fm
