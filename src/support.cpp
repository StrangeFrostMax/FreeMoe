// Architecture-agnostic support probe.
//
// The point of this file is honesty at scale. FreeMoe has one adapter
// (`freemoe-tiny`) plus a generic quantisation layer, so most published models
// do not run yet. Rather than a prose list that goes stale, `freemoe probe`
// reads an actual file and reports, in machine-readable form:
//
//   * what architecture and hyperparameters the file declares,
//   * which tensor types it uses and whether each one can be decoded/executed,
//   * what tokenizer it carries,
//   * every concrete blocker that would stop `freemoe run` today,
//   * which of those blockers are shared with models already known to work.
//
// A missing metadata key is reported as missing, never defaulted silently: the
// difference between "we assume 32 heads" and "the file says 32 heads" is
// exactly the difference between a working loader and a wrong one.

#include "fm/support.hpp"
#include "fm/quant.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fm {
namespace {

struct ArchFacts {
    // Families whose block structure FreeMoe already understands: pre-norm
    // RMSNorm + RoPE + SwiGLU with the standard tensor names.
    bool llama_family = false;
    bool needs_adapter = true;
};

const std::map<std::string, ArchFacts> &known_architectures() {
    static const std::map<std::string, ArchFacts> table = {
        {"freemoe-tiny", {true, false}},
        {"llama", {true, true}},
        {"qwen2", {true, true}},
        {"qwen3", {true, true}},
        {"smollm", {true, true}},
        {"gemma", {false, true}},
        {"gemma2", {false, true}},
        {"gemma3", {false, true}},
        {"phi3", {false, true}},
        {"glm4", {false, true}},
        {"glm4moe", {false, true}},
        {"deepseek2", {false, true}},
        {"qwen3moe", {false, true}},
        {"minicpm", {false, true}},
        {"multimodal", {false, true}},
    };
    return table;
}

const MetadataValue *find(const GGUFDocument &doc, const std::string &key) {
    auto it = doc.metadata.find(key);
    return it == doc.metadata.end() ? nullptr : &it->second;
}

// Reads an integer hyperparameter, following the `<arch>.<name>` convention that
// GGUF uses and falling back to the unprefixed key some converters emit.
bool read_u64(const GGUFDocument &doc, const std::string &arch, const std::string &name,
              uint64_t *out) {
    // Architecture-prefixed keys are the GGUF convention. `freemoe.` and the
    // unprefixed form are accepted so this build's own fixtures and a few older
    // converters still report their real hyperparameters instead of blanks.
    std::vector<std::string> candidates = {arch + "." + name, "freemoe." + name, name};
    static const std::map<std::string, std::string> aliases = {
        {"embedding_length", "hidden_size"},
        {"feed_forward_length", "intermediate_size"},
        {"attention.head_count", "num_attention_heads"},
        {"attention.head_count_kv", "num_key_value_heads"},
        {"attention.key_length", "head_dim"},
        {"block_count", "num_hidden_layers"},
        {"expert_used_count", "expert_top_k"},
    };
    auto alias = aliases.find(name);
    if (alias != aliases.end()) {
        candidates.push_back(arch + "." + alias->second);
        candidates.push_back("freemoe." + alias->second);
        candidates.push_back(alias->second);
    }
    for (const std::string &candidate : candidates) {
        const MetadataValue *value = find(doc, candidate);
        if (!value) continue;
        switch (value->type) {
        case GGUFValueType::UINT8:
        case GGUFValueType::UINT16:
        case GGUFValueType::UINT32:
        case GGUFValueType::UINT64:
            *out = value->u;
            return true;
        case GGUFValueType::INT8:
        case GGUFValueType::INT16:
        case GGUFValueType::INT32:
        case GGUFValueType::INT64:
            *out = static_cast<uint64_t>(value->i);
            return true;
        default:
            return false;
        }
    }
    return false;
}

std::string read_string(const GGUFDocument &doc, const std::string &key) {
    const MetadataValue *value = find(doc, key);
    if (!value || value->type != GGUFValueType::STRING) return std::string();
    return value->s;
}

} // namespace

Json probe_model(const GGUFDocument &doc, const std::string &source_path) {
    Json out = Json::object();
    out.set("schema", "freemoe.model-probe.v1");
    out.set("source", source_path);
    out.set("gguf_version", static_cast<int64_t>(doc.version));
    out.set("tensor_count", static_cast<int64_t>(doc.tensors.size()));
    out.set("file_bytes", static_cast<uint64_t>(doc.file_size));

    const std::string arch = read_string(doc, "general.architecture");
    out.set("architecture", arch);
    out.set("name", read_string(doc, "general.name"));

    auto known = known_architectures().find(arch);
    const bool arch_known = known != known_architectures().end();
    const bool has_adapter = arch_known && !known->second.needs_adapter;
    out.set("architecture_known", arch_known);
    out.set("adapter_available", has_adapter);
    if (arch_known) out.set("llama_family", known->second.llama_family);

    // --- hyperparameters -----------------------------------------------------
    Json hyper = Json::object();
    for (const auto &entry : std::vector<std::pair<const char *, const char *>>{
             {"block_count", "num_hidden_layers"},
             {"embedding_length", "hidden_size"},
             {"feed_forward_length", "intermediate_size"},
             {"attention.head_count", "num_attention_heads"},
             {"attention.head_count_kv", "num_key_value_heads"},
             {"attention.key_length", "head_dim"},
             {"expert_count", "expert_count"},
             {"expert_used_count", "expert_top_k"}}) {
        uint64_t value = 0;
        if (read_u64(doc, arch, entry.first, &value))
            hyper.set(entry.second, static_cast<uint64_t>(value));
    }
    uint64_t vocab = 0;
    if (read_u64(doc, arch, "vocab_size", &vocab))
        hyper.set("vocab_size", vocab);
    out.set("hyperparameters", std::move(hyper));

    // --- tensor types --------------------------------------------------------
    std::map<uint32_t, uint64_t> bytes_by_type;
    std::set<uint32_t> unsupported_types;
    bool all_executable = !doc.tensors.empty();
    bool all_dequantizable = !doc.tensors.empty();
    for (const auto &tensor : doc.tensors) {
        bytes_by_type[tensor.type_id] += tensor.byte_size;
        const TensorCapability cap = tensor_capability(tensor.type_id);
        all_executable = all_executable && cap.execute;
        all_dequantizable = all_dequantizable && cap.dequantize;
        if (!cap.dequantize) unsupported_types.insert(tensor.type_id);
    }
    Json types = Json::array();
    for (const auto &entry : bytes_by_type) {
        Json row = Json::object();
        const TensorCapability cap = tensor_capability(entry.first);
        row.set("type_id", static_cast<int64_t>(entry.first));
        row.set("type", quant_geometry_or_name(entry.first));
        row.set("bytes", entry.second);
        row.set("dequantize", cap.dequantize);
        row.set("execute", cap.execute);
        types.push(std::move(row));
    }
    out.set("tensor_types", std::move(types));
    // Which tensor names the engine would recognise. Anything else is work an
    // adapter has to do, so it is reported rather than quietly ignored.
    std::map<std::string, uint64_t> families;
    for (const auto &tensor : doc.tensors) {
        std::string family = tensor.name;
        const size_t blk = family.find("blk.");
        if (blk != std::string::npos) {
            const size_t dot = family.find('.', blk + 4);
            if (dot != std::string::npos) family = family.substr(dot + 1);
        }
        families[family] += tensor.byte_size;
    }
    Json family_rows = Json::array();
    for (const auto &entry : families) {
        Json row = Json::object();
        row.set("name", entry.first);
        row.set("bytes", entry.second);
        family_rows.push(std::move(row));
    }
    out.set("tensor_families", std::move(family_rows));
    out.set("all_tensors_executable", all_executable);
    out.set("all_tensors_dequantizable", all_dequantizable);

    // --- tokenizer -----------------------------------------------------------
    const std::string tokenizer_model = read_string(doc, "tokenizer.ggml.model");
    out.set("tokenizer_model", tokenizer_model);
    const MetadataValue *tokens = find(doc, "tokenizer.ggml.tokens");
    const MetadataValue *merges = find(doc, "tokenizer.ggml.merges");
    const MetadataValue *scores = find(doc, "tokenizer.ggml.scores");
    out.set("tokenizer_tokens", tokens ? static_cast<int64_t>(tokens->array.size()) : 0);
    out.set("tokenizer_merges", merges ? static_cast<int64_t>(merges->array.size()) : 0);
    out.set("tokenizer_scores", scores ? static_cast<int64_t>(scores->array.size()) : 0);
    out.set("chat_template_present",
            find(doc, "tokenizer.chat_template") != nullptr);

    // --- blockers ------------------------------------------------------------
    // Each blocker names one thing that would have to exist before `freemoe run`
    // could produce a correct token for this file.
    Json blockers = Json::array();
    int64_t blockers_count = 0;
    auto block = [&blockers, &blockers_count](const std::string &code,
                                              const std::string &detail) {
        Json row = Json::object();
        row.set("code", code);
        row.set("detail", detail);
        blockers.push(std::move(row));
        ++blockers_count;
    };

    if (!arch_known) {
        block("UNKNOWN_ARCHITECTURE",
              "no adapter knows architecture '" + arch + "'; its layer order, norms and "
              "tensor naming would have to be transcribed before anything can run");
    } else if (!has_adapter) {
        block("MISSING_ADAPTER",
              "architecture '" + arch + "' is recognised but its forward pass is not "
              "implemented; only the llama-family block structure is understood today");
    }
    for (uint32_t type_id : unsupported_types) {
        block("UNSUPPORTED_TENSOR_TYPE",
              "tensor type " + std::to_string(type_id) +
                  " has no decoder, so its weights cannot even be read");
    }
    if (tokenizer_model.empty()) {
        block("NO_TOKENIZER", "the file carries no tokenizer metadata");
    } else if (tokenizer_model != "freemoe.byte-v1") {
        block("TOKENIZER_NOT_IMPLEMENTED",
              "tokenizer '" + tokenizer_model +
                  "' needs a real BPE/sentencepiece implementation; only the raw byte "
                  "tokenizer is implemented");
    }
    if (find(doc, arch + ".attention.head_count_kv") != nullptr) {
        uint64_t heads = 0, kv = 0;
        if (read_u64(doc, arch, "attention.head_count", &heads) &&
            read_u64(doc, arch, "attention.head_count_kv", &kv) && kv != 0 && kv != heads) {
            block("GROUPED_QUERY_ATTENTION",
              "key/value head count (" + std::to_string(kv) + ") differs from query head "
              "count (" + std::to_string(heads) + "); the engine only implements "
              "multi-head attention with kv == heads");
        }
    }
    if (find(doc, arch + ".expert_count") != nullptr || find(doc, "expert_count") != nullptr) {
        block("MOE_LAYER_LAYOUT", "the file declares experts; the engine's MoE path expects "
                                  "the freemoe stacked-expert tensor layout");
    }
    bool has_expert_tensors = false;
    for (const auto &tensor : doc.tensors)
        has_expert_tensors = has_expert_tensors || tensor.name.find("_exps") != std::string::npos;
    if (has_expert_tensors && !has_adapter)
        block("EXPERT_TENSORS", "expert tensors are present but their layout is not mapped");
    for (const auto &tensor : doc.tensors) {
        if (tensor.name.find("attn_q_norm") != std::string::npos ||
            tensor.name.find("attn_k_norm") != std::string::npos) {
            block("QK_NORM", "per-head Q/K normalisation is present and not implemented");
            break;
        }
    }
    const MetadataValue *tied = find(doc, arch + ".tie_word_embeddings");
    if (tied && tied->b && doc.tensor_by_name.count("output.weight") == 0) {
        block("TIED_OUTPUT_WEIGHT",
              "output weight is tied to the embedding table; the engine requires a "
              "separate output tensor");
    }
    const MetadataValue *bias = find(doc, arch + ".attention.qkv_bias");
    if (bias && bias->b) block("QKV_BIAS", "attention carries qkv biases; the engine's matvec "
                                          "has no bias add");

    const int64_t blocker_count = static_cast<int64_t>(blockers_count);
    out.set("blockers", std::move(blockers));
    out.set("blocker_count", blocker_count);

    const bool runnable = has_adapter && unsupported_types.empty() &&
                          tokenizer_model == "freemoe.byte-v1" && blocker_count == 0;
    out.set("runnable", runnable);
    out.set("verdict", runnable ? "runnable"
                                : (all_dequantizable ? "inspectable-and-decodable"
                                                     : "inspect-only"));
    return out;
}

} // namespace fm
