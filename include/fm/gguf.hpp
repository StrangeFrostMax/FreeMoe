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
    uint32_t block_size = 1;
    uint32_t block_bytes = 4;
    TensorRole role = TensorRole::UNKNOWN;
    int layer = -1;
    bool expert_tensor = false;
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
    uint32_t vocab = 0;
    uint32_t context = 0;
    uint32_t expert_count = 0;
    uint32_t expert_top_k = 0;
    uint32_t bos_id = 0;
    uint32_t eos_id = 0;
    double rms_eps = 0.0;
    double rope_theta = 0.0;
    bool shared_expert = false;
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
    std::array<ExpertExtent, 3> extents{};
};

struct ModelManifest {
    GGUFDocument doc;
    ModelConfig config;
    std::string source_path;
    std::vector<std::string> tokenizer_tokens;
    std::vector<ExpertRecord> experts;
    std::map<uint64_t, size_t> expert_index;
    uint64_t fixed_weight_bytes = 0;
    uint64_t expert_tensor_bytes = 0;
    uint64_t slot_bytes = 0;

    const TensorInfo *tensor(const std::string &name) const {
        auto it = doc.tensor_by_name.find(name);
        return it == doc.tensor_by_name.end() ? nullptr : &doc.tensors[it->second];
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
GGUFDocument parse_gguf(const uint8_t *data, size_t size);
GGUFDocument parse_gguf_file(const std::string &path);
ModelManifest build_manifest(const GGUFDocument &doc, const std::string &source_path);
Json inspect_json(const ModelManifest &manifest, const std::string &source_path);
Json metadata_summary_json(const ModelManifest &manifest);

} // namespace fm
