#pragma once

#include "fm/common.hpp"

#include <array>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace fm {

class MappedFile {
public:
    MappedFile() = default;
    explicit MappedFile(const std::string &path);
    ~MappedFile();
    MappedFile(const MappedFile &) = delete;
    MappedFile &operator=(const MappedFile &) = delete;
    const uint8_t *data() const { return data_; }
    size_t size() const { return size_; }
    const std::string &path() const { return path_; }
    const FileIdentity &identity() const { return identity_; }
private:
    std::string path_;
    const uint8_t *data_ = nullptr;
    size_t size_ = 0;
    FileIdentity identity_{};
    void *mapping_ = nullptr;
#ifdef _WIN32
    std::vector<uint8_t> fallback_;
#endif
};

enum class GGUFValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3,
    UINT32 = 4, INT32 = 5, FLOAT32 = 6, BOOL = 7,
    STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12,
};

struct MetadataValue {
    GGUFValueType type = GGUFValueType::UINT32;
    uint64_t u = 0;
    int64_t i = 0;
    double f = 0.0;
    bool b = false;
    std::string s;
    std::vector<MetadataValue> array;
    GGUFValueType element_type = GGUFValueType::UINT32;

    bool is_array() const { return type == GGUFValueType::ARRAY; }
    bool is_string() const { return type == GGUFValueType::STRING; }
    uint64_t as_u64() const;
    int64_t as_i64() const;
    double as_double() const;
    bool as_bool() const;
    const std::string &as_string() const;
    std::string display() const;
};

enum class TensorRole : uint32_t {
    UNKNOWN = 0,
    TOKEN_EMBD,
    OUTPUT,
    OUTPUT_NORM,
    ATTN_NORM,
    ATTN_Q,
    ATTN_K,
    ATTN_V,
    ATTN_OUTPUT,
    FFN_NORM,
    FFN_GATE,
    FFN_UP,
    FFN_DOWN,
    FFN_ROUTER,
    FFN_GATE_EXPS,
    FFN_UP_EXPS,
    FFN_DOWN_EXPS,
    // Optional per-projection biases. qwen2 and several descendants carry them.
    ATTN_Q_BIAS,
    ATTN_K_BIAS,
    ATTN_V_BIAS,
    ATTN_OUTPUT_BIAS,
    // Per-head RMSNorm on q/k, which qwen3 applies before RoPE.
    ATTN_Q_NORM,
    ATTN_K_NORM,
    ATTN_POST_NORM,
    FFN_POST_NORM,
    // Fused query/key/value projection (phi3).
    ATTN_QKV,
    ATTN_QKV_BIAS,
    // The dense "shared expert" that qwen2moe adds to the routed experts, and
    // the per-token scalar that gates it.
    FFN_GATE_INP_SHEXP,
    FFN_GATE_SHEXP,
    FFN_UP_SHEXP,
    FFN_DOWN_SHEXP,
    // LayerNorm carries a bias, and phi2 also biases the FFN, the attention
    // output and the logits.
    ATTN_NORM_BIAS,
    FFN_NORM_BIAS,
    OUTPUT_NORM_BIAS,
    FFN_UP_BIAS,
    FFN_DOWN_BIAS,
    OUTPUT_BIAS,
};

const char *tensor_role_name(TensorRole role);

struct TensorInfo {
    std::string name;
    std::vector<uint64_t> dims;
    uint32_t type_id = 0;
    std::string type_name;
    uint64_t relative_offset = 0;
    uint64_t absolute_offset = 0;
    uint64_t byte_size = 0;
    // Canonical name the engine addresses this tensor by. Equals `name` for
    // files that already use canonical names; adapters may map a vendor name
    // onto it while `name` keeps the original.
    std::string canonical;
    uint32_t block_size = 1;
    uint32_t block_bytes = 4;
    // False when the container declares a type this build has no geometry for.
    // The tensor is still bounded (from offsets) but cannot be decoded.
    bool type_known = true;
    bool byte_size_derived = false;
    TensorRole role = TensorRole::UNKNOWN;
    int layer = -1;
    bool expert_tensor = false;

    // Number of scalar elements the tensor holds.
    uint64_t element_count() const {
        uint64_t n = 1;
        for (uint64_t d : dims) n *= d;
        return n;
    }
};

struct GGUFDocument {
    uint32_t version = 0;
    uint32_t alignment = 32;
    uint64_t metadata_count = 0;
    uint64_t tensor_count = 0;
    uint64_t data_offset = 0;
    uint64_t file_size = 0;
    uint64_t source_fingerprint = 0;
    std::map<std::string, MetadataValue> metadata;
    // Insertion order as it appeared in the file. Needed for byte-exact
    // re-serialisation, because GGUF metadata is an ordered list, not a map.
    std::vector<std::string> metadata_order;
    std::vector<TensorInfo> tensors;
    std::map<std::string, size_t> tensor_by_name;
};

struct ModelConfig {
    std::string architecture;
    std::string adapter;
    std::string model_kind;
    std::string rope_mode;
    std::string tokenizer_model;
    std::string chat_template;
    std::string router_operation;
    std::string expert_layout;
    uint32_t hidden = 0;
    uint32_t intermediate = 0;
    uint32_t layers = 0;
    uint32_t heads = 0;
    uint32_t kv_heads = 0;
    uint32_t head_dim = 0;
    // Rotary dimensions actually rotated; 0 means "whole head".
    uint32_t rope_dim = 0;
    uint32_t vocab = 0;
    uint32_t context = 0;
    uint32_t expert_count = 0;
    uint32_t expert_top_k = 0;
    // Routing semantics, straight from the checkpoint. Renormalising over the
    // selected experts is only equivalent to a full softmax when every expert is
    // selected, so these cannot be assumed.
    bool expert_weights_norm = false;
    double expert_weights_scale = 1.0;
    // Attention projections carry biases (qwen2 family).
    bool attn_qkv_bias = false;
    // Q and K are normalised per head before RoPE (qwen3 family).
    bool qk_norm = false;
    // Multiplier applied to the token embedding before the first block.
    // gemma uses sqrt(hidden); minicpm uses a fixed 12.0 unless the file says
    // otherwise. 1.0 means "no scaling", so this is off for llama-family.
    double embedding_scale = 1.0;
    // minicpm carries two further scales. llama.cpp parses and logs them but
    // never applies them to the computation, so neither do we: applying them
    // would disagree with the reference. They are recorded so the probe can
    // report them rather than silently dropping information.
    double residual_scale = 0.0;
    double logit_scale = 0.0;
    // gemma stores RMSNorm scales zero-centred: the effective scale is 1 + w.
    bool norm_add_one = false;
    // "silu" (SwiGLU) or "gelu" (GeGLU). The two are not interchangeable.
    std::string activation = "silu";
    // gemma2/3 clamp attention scores and/or final logits through tanh.
    double attn_logit_softcap = 0.0;
    double final_logit_softcap = 0.0;
    // gemma2/3 apply an extra RMSNorm after attention and after the FFN.
    bool post_norms = false;
    // gemma2/3 restrict most layers to a sliding attention window.
    uint32_t sliding_window = 0;
    // phi3 fuses q, k and v into one projection (and one optional bias).
    bool fused_qkv = false;
    // glm4 stores the FFN gate and up halves in one tensor of
    // [hidden, 2*intermediate]: gate first, then up (ggml_swiglu's convention).
    bool fused_gate_up = false;
    uint32_t bos_id = 0;
    uint32_t eos_id = 0;
    double rms_eps = 0.0;
    double rope_theta = 0.0;
    // qwen2moe adds a dense feed-forward branch to the routed experts. Its
    // width is its own hyperparameter (`expert_shared_feed_forward_length`),
    // not the dense width, so it is stored separately. 0 means "no shared
    // expert", which keeps every other architecture's path unchanged.
    bool shared_expert = false;
    uint32_t shared_expert_intermediate = 0;
    // Routed experts may be narrower than the dense FFN (`expert_feed_forward_
    // length`). 0 means "same as intermediate", the usual case.
    uint32_t expert_intermediate = 0;

    uint32_t expert_width() const { return expert_intermediate ? expert_intermediate : intermediate; }

    // phi2 is built differently enough to need four separate switches, and each
    // is evidence-driven: the norm type comes from which epsilon key the file
    // declares, the biases from the tensors, and the parallel residual from the
    // absence of a separate FFN norm.
    bool layer_norm = false;        // mean-centred LayerNorm instead of RMSNorm
    bool norm_bias = false;         // norms are scale + bias, not just scale
    bool parallel_residual = false;  // FFN runs on the attention's normalised input
    bool ffn_no_gate = false;       // up -> gelu -> down, with no gate projection
    bool ffn_bias = false;          // biases on ffn_up / ffn_down
    bool attn_output_bias = false;  // bias on attn_output
    bool logit_bias = false;        // bias on the output projection
};

struct ExpertExtent {
    TensorRole role = TensorRole::UNKNOWN;
    uint64_t offset = 0;
    uint64_t bytes = 0;
};

struct ExpertRecord {
    uint32_t layer = 0;
    uint32_t expert = 0;
    uint64_t payload_bytes = 0;
    uint64_t checksum = 0;
    // False for checkpoints that carry no checksum manifest (every vendor file).
    // The loader then verifies only that the bytes it read are the bytes it
    // asked for, and reports that source checksums were unavailable.
    bool checksum_known = false;
    std::array<ExpertExtent, 3> extents{};
};

struct ModelManifest {
    GGUFDocument doc;
    ModelConfig config;
    std::string source_path;
    std::vector<std::string> tokenizer_tokens;
    std::vector<ExpertRecord> experts;
    // Canonical name -> real canonical name, for tensors the model shares.
    // Tied embeddings are the common case: `output.weight` is the token table.
    std::map<std::string, std::string> alias;
    std::map<uint64_t, size_t> expert_index;
    uint64_t fixed_weight_bytes = 0;
    uint64_t expert_tensor_bytes = 0;
    uint64_t slot_bytes = 0;

    // Looks up by real name first, then by canonical name. Adapters may rename
    // a vendor tensor (ffn_gate_inp -> ffn_router) while `doc` keeps the
    // original name, so both must be addressable.
    const TensorInfo *tensor(const std::string &name) const {
        auto it = doc.tensor_by_name.find(name);
        if (it != doc.tensor_by_name.end()) return &doc.tensors[it->second];
        for (const auto &candidate : doc.tensors)
            if (candidate.canonical == name) return &candidate;
        return nullptr;
    }
    const ExpertRecord *expert(uint32_t layer, uint32_t id) const {
        auto it = expert_index.find(expert_key(layer, id));
        return it == expert_index.end() ? nullptr : &experts[it->second];
    }
    static uint64_t expert_key(uint32_t layer, uint32_t id) {
        return (static_cast<uint64_t>(layer) << 32) | id;
    }
};

struct QuantGeometry {
    uint32_t block_size;
    uint32_t block_bytes;
    const char *name;
};

const std::map<uint32_t, QuantGeometry> &quantisation_table();

// Per-tensor-type capability. "Parsed" never implies "executable": a type must
// pass block-exact dequantisation tests before it is advertised as executable.
struct TensorCapability {
    bool parse = false;
    bool dequantize = false;
    bool execute = false;
    std::string note;
};

TensorCapability tensor_capability(uint32_t type_id);

// Decodes `count` elements of `type_id` from `src` into F32 `dst`. Throws a
// typed error for types whose decode is not implemented rather than guessing.
void dequantize_row(uint32_t type_id, const uint8_t *src, float *dst, uint64_t count);

// Container-level inspection that works for any GGUF file, independent of any
// architecture adapter. Never executes the model and never assumes an adapter.
Json generic_inspect_json(const GGUFDocument &doc, const std::string &source_path);

// Re-serialises a parsed document. Payload bytes are copied from `source`, which
// must be the exact buffer the document was parsed from. Used to prove that the
// reader and writer agree byte-for-byte.
std::vector<uint8_t> write_gguf(const GGUFDocument &doc, const uint8_t *source,
                                size_t source_size);
GGUFDocument parse_gguf(const uint8_t *data, size_t size);
GGUFDocument parse_gguf_file(const std::string &path);
ModelManifest build_manifest(const GGUFDocument &doc, const std::string &source_path);
Json inspect_json(const ModelManifest &manifest, const std::string &source_path);
Json metadata_summary_json(const ModelManifest &manifest);

} // namespace fm
