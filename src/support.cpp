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
#include "fm/adapter.hpp"
#include "fm/quant.hpp"
#include "fm/tokenizer.hpp"

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace fm {

// `needs_adapter` is false when this build can actually run the architecture;
// everything else is inspected and reported, never silently attempted.
const std::map<std::string, ArchFacts> &known_architectures() {
    static const std::map<std::string, ArchFacts> table = {
        // Implemented: pre-norm RMSNorm + RoPE + SwiGLU + optional top-k MoE,
        // with grouped-query attention and tied output embeddings.
        {"freemoe-tiny", {true, false}},
        {"llama", {true, false}},
        {"qwen2", {true, false}},
        {"qwen3", {true, false}},
        {"smollm", {true, false}},
        {"smollm2", {true, false}},
        {"tinyllama", {true, false}},
        // gemma: scaled embeddings, zero-centred norm scales, GeGLU, tied
        // output. gemma2 adds post-norms, tanh softcapping and a sliding window.
        {"gemma", {true, false}},
        {"gemma2", {true, false}},
        {"gemma3", {true, false}},
        // phi3: one fused q/k/v projection (with fused bias) and a rotary width
        // narrower than the head.
        {"phi3", {true, false}},
        // minicpm: the token embedding is scaled before the first block.
        {"minicpm", {true, false}},
        // qwen3moe: qwen3 semantics with a stacked-expert feed-forward.
        {"qwen3moe", {true, false}},
        // glm4: fused gate+up FFN with post-attention and post-FFN norms.
        {"glm4", {true, false}},
        // qwen2moe: routed experts plus a dense shared expert whose output is
        // gated by one scalar per token.
        {"qwen2moe", {true, false}},
        // phi2: LayerNorm instead of RMSNorm, a gate-free GELU FFN, and one
        // shared residual for the attention and FFN branches.
        {"phi2", {true, false}},
        // internlm2: the plain block. It exists here to prove the block is
        // driven by the checkpoint rather than by one family's code path.
        {"internlm2", {true, false}},
        // orion: LayerNorm with biases on every norm, sequential residual.
        {"orion", {true, false}},
        // exaone: RMSNorm, SwiGLU, tied output, optional per-layer rope factors.
        {"exaone", {true, false}},
        // gptneox: fused q/k/v with a fused bias, LayerNorm with biases, and a
        // gate-free GELU feed-forward.
        {"gptneox", {true, false}},
        // stablelm: LayerNorm with biases and a rotary width narrower than the
        // head; the FFN norm is optional, and its absence means the FFN runs on
        // the attention's normalised input.
        {"stablelm", {true, false}},
        // deepseek2: multi-head latent attention, dense-lead blocks, and a
        // router whose gating function, selection bias and group masking all
        // change which experts run.
        {"deepseek2", {true, false}},
        // smollm3: the llama block with every fourth layer left unrotated.
        {"smollm3", {true, false}},
        // bitnet: the llama block plus SUBLN sub-norms and optional one-element
        // projection scales.
        {"bitnet", {true, false}},
        // Gate-free relu^2 feed-forward families: the same llama block, one
        // projection, and an activation GGUF metadata cannot state, so the
        // architecture name is the evidence (see build_manifest_llama).
        {"arcee", {true, false}},
        {"nemotron", {true, false}},
        {"jais2", {true, false}},
        // gpt-oss: attention sinks, a post-attention norm that feeds the MoE,
        // a biased router, per-expert projection biases, swiglu_oai and
        // alternating full/windowed layers.
        {"gpt-oss", {true, false}},
        // plm: multi-head latent attention, a gate-free relu^2 FFN and a tied
        // output head.
        {"plm", {true, false}},
        // The mRoPE text paths and the two clamped deepseek revisions share the
        // llama block; their differences are metadata the adapter reads.
        {"qwen3vl", {true, false}},
        {"qwen3vlmoe", {true, false}},
        {"deepseek4", {true, false}},
        {"deepseek41", {true, false}},
        // mistral3: the llama block plus optional MLP biases and an adaptive
        // RMSNorm whose MLP is a gelu_erf.
        {"mistral3", {true, false}},
        // mamba/mamba2: pure state-space models, with no attention and no FFN.
        // The state rolls forward in place rather than through a kv cache.
        {"mamba", {true, false}},
        {"mamba2", {true, false}},
        // jamba: a hybrid stack, so it needs both a rolling state and a
        // per-position cache, and its attention layers carry no rotary.
        {"jamba", {true, false}},
        {"gpt2", {true, false}},
        {"starcoder", {true, false}},
        {"falcon", {true, false}},
        {"mpt", {true, false}},
        {"seed-oss", {true, false}},
        {"plamo", {true, false}},
        {"smallthinker", {true, false}},
        {"command-r", {true, false}},
        {"olmo", {true, false}},
        {"phimoe", {true, false}},
        {"baichuan", {true, false}},
        {"chatglm", {true, false}},
        {"olmo2", {true, false}},
        {"olmoe", {true, false}},
        {"granite", {true, false}},
        {"granitemoe", {true, false}},
        {"bloom", {true, false}},
        {"jais", {true, false}},
        {"codeshell", {true, false}},
        {"cohere2", {true, false}},
        {"deci", {true, false}},
        {"hunyuan-dense", {true, false}},
        {"llama-embed", {true, false}},
        {"maincoder", {true, false}},
        {"nanbeige", {true, false}},
        {"paddleocr", {true, false}},
        {"pockettts", {true, false}},
        {"qwen", {true, false}},
        {"starcoder2", {true, false}},
        {"xverse", {true, false}},
        // bailingmoe: softmax routing, an ungated shared expert, and (v2) a fused
        // qkv, a per-head q/k norm and dense leading blocks.
        {"bailingmoe", {true, false}},
        {"bailingmoe2", {true, false}},
        // Census batch 1, slice B. dbrx is a LayerNorm block whose only
        // post-attention norm feeds the MoE; deepseek and dots1 add a dense lead
        // and an ungated shared expert; ernie4-5-moe interleaves MoE layers;
        // glm4-moe routes by sigmoid and normalises after the attention residual;
        // minimax-m2 adds a router selection bias.
        {"dbrx", {true, false}},
        {"deepseek", {true, false}},
        {"dots1", {true, false}},
        {"ernie4-5-moe", {true, false}},
        {"glm4-moe", {true, false}},
        {"minimax-m2", {true, false}},
        // Slice C: per-layer rotary (file pattern, or the window pattern) with an
        // optional per-layer base, per-layer expert widths, and hunyuan-moe's
        // dense-plus-MoE layer.
        {"exaone-moe", {true, false}},
        {"flint", {true, false}},
        {"hunyuan-moe", {true, false}},
        {"maple", {true, false}},
        {"glm4moe", {false, true}},
        {"multimodal", {false, true}},
    };
    return table;
}

namespace {

// Tokenizers this build can decode and encode with.
bool tokenizer_implemented(const std::string &model) {
    return model == "freemoe.byte-v1" || model == "llama" || model == "spm" || model == "gpt2" ||
           model == "bpe";
}

const MetadataValue *find(const GGUFDocument &doc, const std::string &key) {
    auto it = doc.metadata.find(key);
    return it == doc.metadata.end() ? nullptr : &it->second;
}

// Reads an integer hyperparameter, following the `<arch>.<name>` convention that
// GGUF uses and falling back to the unprefixed key some converters emit.
// Reads a real-valued hyperparameter, for keys stored as F32/F64
// (`hyper_connection.count` is one). Integers are accepted too, because a
// converter that writes 2 instead of 2.0 means the same thing.
bool read_double(const GGUFDocument &doc, const std::string &arch, const std::string &name,
                 double *out) {
    for (const std::string &key : {arch + "." + name, "freemoe." + name, name}) {
        const MetadataValue *value = find(doc, key);
        if (value == nullptr) continue;
        if (value->type == GGUFValueType::FLOAT32 || value->type == GGUFValueType::FLOAT64) {
            *out = value->as_double();
            return true;
        }
        if (value->type == GGUFValueType::UINT32 || value->type == GGUFValueType::UINT64 ||
            value->type == GGUFValueType::INT32 || value->type == GGUFValueType::INT64) {
            *out = static_cast<double>(value->as_i64());
            return true;
        }
    }
    return false;
}

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
        {"expert_feed_forward_length", "expert_intermediate_size"},
        {"expert_shared_feed_forward_length", "shared_expert_size"},
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

// `blk.L.ffn_gate.E.weight`: one tensor per routed expert. The probe cannot use
// the adapter's records (it runs before the manifest exists), so it settles the
// layout from the names, the same evidence the loader uses.
bool has_per_expert_tensors(const GGUFDocument &doc) {
    const std::string marker = ".ffn_gate.";
    for (const auto &tensor : doc.tensors) {
        const size_t pos = tensor.name.find(marker);
        if (pos == std::string::npos) continue;
        const std::string tail = tensor.name.substr(pos + marker.size());
        const std::string suffix = ".weight";
        if (tail.size() <= suffix.size()) continue;
        if (tail.compare(tail.size() - suffix.size(), suffix.size(), suffix) != 0) continue;
        bool digits = true;
        for (size_t i = 0; i + suffix.size() < tail.size(); ++i)
            if (tail[i] < '0' || tail[i] > '9') digits = false;
        if (digits) return true;
    }
    return false;
}

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
             {"expert_used_count", "expert_top_k"},
             {"expert_feed_forward_length", "expert_intermediate_size"},
             {"expert_shared_feed_forward_length", "shared_expert_size"}}) {
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
    // Structures the loader refuses, reported here for the same reason: `probe` is
    // only useful if it gives the answer a run would give. Both change the forward
    // pass without changing the tensor set, so no missing-tensor check catches
    // them.
    {
        double hc_count = 0.0;
        if (read_double(doc, arch, "hyper_connection.count", &hc_count) && hc_count > 1.0) {
            block("HYPER_CONNECTIONS_UNMODELLED",
                  "hyper_connection.count = " + std::to_string(static_cast<int>(hc_count)) +
                      ": hyper-connections widen the residual stream and are a block this "
                      "build does not have");
        }
        const MetadataValue *deepstack = find(doc, arch + ".deepstack_mapping");
        if (deepstack != nullptr && deepstack->is_array()) {
            for (const MetadataValue &entry : deepstack->array) {
                if (entry.as_i64() >= 0) {
                    block("DEEPSTACK_UNMODELLED",
                          "granite deepstack_mapping injects a projector stream into the layers "
                          "it names, and that stream is not modelled here");
                    break;
                }
            }
        }
        for (const char *indexer_key : {"attention.indexer.head_count",
                                        "attention.indexer.key_length",
                                        "attention.indexer.top_k"}) {
            if (find(doc, arch + "." + indexer_key) != nullptr ||
                find(doc, std::string("freemoe.") + indexer_key) != nullptr) {
                block("INDEXER_UNMODELLED",
                      "the DSA indexer (" + arch + "." + indexer_key +
                          ") selects which positions attention reads and is not modelled here");
                break;
            }
        }
    }
    for (uint32_t type_id : unsupported_types) {
        block("UNSUPPORTED_TENSOR_TYPE",
              "tensor type " + std::to_string(type_id) +
                  " has no decoder, so its weights cannot even be read");
    }
    if (tokenizer_model.empty()) {
        block("NO_TOKENIZER", "the file carries no tokenizer metadata");
    } else if (!tokenizer_implemented(tokenizer_model)) {
        block("TOKENIZER_NOT_IMPLEMENTED",
              "tokenizer '" + tokenizer_model + "' is not implemented");
    } else if ((tokenizer_model == "gpt2" || tokenizer_model == "bpe") && merges == nullptr) {
        block("TOKENIZER_NO_MERGES",
              "a BPE tokenizer without its merge list cannot segment text");
    } else if (tokenizer_model == "gpt2" || tokenizer_model == "bpe") {
        // Without a pre-tokeniser the word boundaries are unknown, and guessing
        // them produces different token ids on ordinary text.
        const MetadataValue *pre = find(doc, "tokenizer.ggml.pre");
        const std::string pre_name =
            (pre && pre->type == GGUFValueType::STRING) ? pre->s : std::string();
        const Tokenizer::PreTokenizer family = Tokenizer::pre_tokenizer_from_name(pre_name);
        if (family == Tokenizer::PreTokenizer::Unspecified) {
            block("TOKENIZER_NO_PRE",
                  "the file carries no tokenizer.ggml.pre, so its word-splitting rule is unknown");
        } else if (family == Tokenizer::PreTokenizer::Unsupported) {
            block("TOKENIZER_PRE_NOT_IMPLEMENTED",
                  "pre-tokenizer '" + pre_name +
                      "' is not implemented; implemented families are gpt2, gpt2-digits, "
                      "llama3 and qwen2");
        }
    }
    // Grouped-query attention is implemented, so it is reported as a capability
    // rather than as a blocker.
    {
        uint64_t heads = 0, kv = 0;
        if (read_u64(doc, arch, "attention.head_count", &heads) &&
            read_u64(doc, arch, "attention.head_count_kv", &kv) && kv != 0) {
            out.set("grouped_query_attention", kv != heads);
            if (heads % kv != 0)
                block("INCONSISTENT_GQA",
                      "key/value head count (" + std::to_string(kv) +
                          ") does not divide the query head count (" + std::to_string(heads) + ")");
        }
    }
    // The standard stacked-expert layout (ffn_gate_exps/ffn_up_exps/ffn_down_exps
    // with ffn_gate_inp routing) is implemented, so it is only a blocker for
    // architectures that have no adapter at all.
    const bool declares_experts =
        find(doc, arch + ".expert_count") != nullptr || find(doc, "expert_count") != nullptr;
    if (declares_experts && known_architectures().count(arch) && !has_adapter) {
        block("MOE_LAYER_LAYOUT", "this architecture's MoE layout is not mapped");
    }
    bool has_expert_tensors = false;
    for (const auto &tensor : doc.tensors)
        has_expert_tensors = has_expert_tensors || tensor.name.find("_exps") != std::string::npos;
    if (has_expert_tensors && !has_adapter)
        block("EXPERT_TENSORS", "expert tensors are present but their layout is not mapped");
    // minicpm declares three scales. Only the embedding scale is applied by the
    // reference implementation; the other two are parsed and logged there and
    // never used in the graph, so they are reported rather than silently
    // dropped or silently applied.
    {
        const MetadataValue *emb = find(doc, arch + ".embedding_scale");
        const MetadataValue *res = find(doc, arch + ".residual_scale");
        const MetadataValue *logit = find(doc, arch + ".logit_scale");
        if (emb || res || logit) {
            Json scales = Json::object();
            // Json has no double overload, so the values are rendered as strings
            // with enough precision to be checkable.
            auto fmt = [](double v) {
                char buf[64];
                std::snprintf(buf, sizeof(buf), "%.9g", v);
                return std::string(buf);
            };
            scales.set("embedding_scale", fmt(emb ? emb->as_double() : 0.0));
            scales.set("residual_scale", fmt(res ? res->as_double() : 0.0));
            scales.set("logit_scale", fmt(logit ? logit->as_double() : 0.0));
            scales.set("applied", Json::array());
            out.set("declared_scales", std::move(scales));
            if (res || logit)
                out.set("scale_note",
                        "residual_scale and logit_scale are parsed but not applied: "
                        "llama.cpp reads and logs them without using them in the graph, "
                        "so applying them here would disagree with the reference");
        }
    }

    // --- ternary family -----------------------------------------------------
    // PrismML's Bonsai 2 checkpoint stores its weights in a *rotated basis*: each
    // matrix is transformed by a blockwise Hadamard rotation before the ternary
    // assignment, and the runtime must apply the matching transform to
    // activations. Without it the weights decode to plausible-looking numbers
    // and the model emits garbage, so this is a hard blocker rather than a
    // warning. The checkpoint declares the rotation, which makes it checkable.
    {
        bool rotated = false;
        bool linear_attention = false;
        bool ternary = false;
        for (const auto &key : doc.metadata)
            if (key.first.rfind("prism.hadamard.", 0) == 0) rotated = true;
        for (const auto &tensor : doc.tensors) {
            const std::string &name = tensor.name;
            if (name.rfind("ssm_", 0) == 0 || name.find(".ssm_") != std::string::npos ||
                name.find("conv1d") != std::string::npos)
                linear_attention = true;
            if (tensor.type_id == 34 || tensor.type_id == 35 || tensor.type_id == 41 ||
                tensor.type_id == 42 || tensor.type_id == 142 || tensor.type_id == 143)
                ternary = true;
        }
        out.set("rotated_weight_basis", rotated);
        out.set("linear_attention", linear_attention);
        out.set("ternary_weights", ternary);
        if (rotated)
            block("ROTATED_WEIGHT_BASIS",
                  "weights are stored in a Hadamard-rotated basis and the file declares "
                  "prism.hadamard.* metadata; the runtime must apply the matching "
                  "transform to activations, which is not implemented, so decoding the "
                  "weights directly would produce garbage");
        if (linear_attention)
            block("LINEAR_ATTENTION",
                  "the checkpoint contains linear-attention state (ssm_/conv) alongside "
                  "full attention; only full attention is implemented");
    }

    // Per-head q/k normalisation (qwen3) is implemented; report it as a
    // capability, and only complain when the pair is incomplete.
    {
        bool q_norm = false, k_norm = false;
        for (const auto &tensor : doc.tensors) {
            const bool per_layer = tensor.name.rfind("blk.", 0) == 0;
            if (per_layer && tensor.name.find("attn_q_norm") != std::string::npos) q_norm = true;
            if (per_layer && tensor.name.find("attn_k_norm") != std::string::npos) k_norm = true;
        }
        out.set("per_head_qk_norm", q_norm || k_norm);
        // Multi-head latent attention, the dense lead-in blocks and the router's
        // gating function are all decided by the checkpoint, so the probe says
        // which of them it found rather than leaving the reader to guess.
        bool mla = false, dense_lead = false, selection_bias = false;
        for (const auto &tensor : doc.tensors) {
            if (tensor.name.find("attn_kv_a_mqa") != std::string::npos) mla = true;
            if (tensor.name.find("ffn_exp_probs_b.bias") != std::string::npos) selection_bias = true;
            if (tensor.name.find("ffn_gate_shexp") != std::string::npos) dense_lead = true;
        }
        out.set("multi_head_latent_attention", mla);
        out.set("expert_selection_bias", selection_bias);
        out.set("shared_expert", dense_lead);
        uint64_t lead = 0;
        if (read_u64(doc, arch, "leading_dense_block_count", &lead) && lead != 0)
            out.set("leading_dense_blocks", lead);
        uint64_t gating = 0;
        if (read_u64(doc, arch, "expert_gating_func", &gating) && gating != 0)
            out.set("expert_gating_func", gating);
        if (q_norm != k_norm)
            block("QK_NORM_INCOMPLETE", "q/k normalisation is present on only one of q and k");
    }
    // Tied output embeddings are supported (the output projection reads the
    // token table), so only a *missing* both-ways case is a blocker.
    if (doc.tensor_by_name.count("output.weight") == 0 &&
        doc.tensor_by_name.count("token_embd.weight") == 0) {
        block("NO_OUTPUT_WEIGHT", "neither output.weight nor a token table is present");
    }
    // qwen2-style projection biases are implemented; only an inconsistent file
    // (declared bias with no bias tensor, or vice versa) is a blocker.
    const MetadataValue *bias = find(doc, arch + ".attention.qkv_bias");
    const bool declares_bias = bias && bias->b;
    const bool has_bias_tensor = doc.tensor_by_name.count("blk.0.attn_q.bias") != 0 ||
                                 doc.tensor_by_name.count("blk.0.attn_qkv.bias") != 0;
    if (bias && declares_bias && !has_bias_tensor)
        block("QKV_BIAS_MISSING_TENSOR",
              "attention.qkv_bias is set but no attention bias tensor is present");
    if (bias && !declares_bias && has_bias_tensor)
        block("QKV_BIAS_INCONSISTENT",
              "attention bias tensors are present but attention.qkv_bias is false");
    out.set("attention_qkv_bias", declares_bias || has_bias_tensor);
    out.set("attn_sinks", doc.tensor_by_name.count("blk.0.attn_sinks.weight") != 0);
    // bitnet's SUBLN pair and its optional one-element projection scales. Both
    // change the arithmetic inside the block, so a reader needs to see them.
    const bool has_attn_sub_norm = doc.tensor_by_name.count("blk.0.attn_sub_norm.weight") != 0;
    const bool has_ffn_sub_norm = doc.tensor_by_name.count("blk.0.ffn_sub_norm.weight") != 0;
    out.set("sub_norms", has_attn_sub_norm && has_ffn_sub_norm);
    if (has_attn_sub_norm != has_ffn_sub_norm)
        block("SUB_NORM_INCOMPLETE",
              "attn_sub_norm and ffn_sub_norm must both be present or both absent");
    if (arch == "bitnet" && !has_attn_sub_norm)
        block("SUB_NORM_MISSING",
              "bitnet requires attn_sub_norm.weight and ffn_sub_norm.weight");
    bool has_projection_scale = false;
    for (const char *suffix : {"attn_q.scale", "attn_k.scale", "attn_v.scale",
                               "attn_output.scale", "ffn_gate.scale", "ffn_up.scale",
                               "ffn_down.scale"})
        if (doc.tensor_by_name.count("blk.0." + std::string(suffix)) != 0)
            has_projection_scale = true;
    out.set("projection_scales", has_projection_scale);
    // Context extension. `none`, `linear` and `yarn` are implemented; anything
    // else changes the rotation in a way this build cannot reproduce, so it is a
    // blocker rather than an approximation. Reported even when it is none, so a
    // reader can see the file was actually checked.
    {
        const MetadataValue *scaling = find(doc, arch + ".rope.scaling.type");
        std::string kind;
        if (scaling && scaling->type == GGUFValueType::STRING) kind = scaling->s;
        if (kind.empty()) kind = "none";
        out.set("rope_scaling", kind);
        if (kind != "none" && kind != "linear" && kind != "yarn")
            block("ROPE_SCALING_NOT_IMPLEMENTED",
                  "rope.scaling.type '" + kind +
                      "' is not implemented; 'none', 'linear' and 'yarn' are");
        // YaRN's ramp needs the context the model was trained on. Without it the
        // model would still run, at the wrong frequencies, so this is a blocker.
        if ((kind == "yarn" || kind == "linear") &&
            find(doc, arch + ".rope.scaling.original_context_length") == nullptr &&
            find(doc, arch + ".context_length") == nullptr)
            block("ROPE_SCALING_NO_ANCHOR",
                  "rope.scaling is set but neither original_context_length nor "
                  "context_length anchors the correction ramp");
    }
    // The block structure that the RMSNorm/SwiGLU assumption does not cover.
    // Each is read from the tensors or from the epsilon key, never guessed.
    {
        const bool ln = find(doc, arch + ".attention.layer_norm_epsilon") != nullptr;
        const bool rms = find(doc, arch + ".attention.layer_norm_rms_epsilon") != nullptr;
        if (ln && rms)
            block("AMBIGUOUS_NORM_TYPE",
                  "both a LayerNorm and an RMSNorm epsilon are declared");
        out.set("norm_type", ln ? "layernorm" : "rmsnorm");
        const bool ffn_norm = doc.tensor_by_name.count("blk.0.ffn_norm.weight") != 0;
        const bool gate = doc.tensor_by_name.count("blk.0.ffn_gate.weight") != 0;
        if (!ffn_norm) {
            if (doc.tensor_by_name.count("blk.0.attn_norm.weight") == 0)
                block("NO_NORM_ON_FFN_INPUT",
                      "no ffn_norm and no attn_norm: nothing normalises the FFN input");
            out.set("parallel_residual", true);
        }
        if (!gate && doc.tensor_by_name.count("blk.0.ffn_up.weight") != 0)
            out.set("ffn_without_gate", true);
        if (doc.tensor_by_name.count("output.bias") != 0) out.set("output_bias", true);
        if (doc.tensor_by_name.count("blk.0.ffn_up.bias") != 0) out.set("ffn_bias", true);
    }
    // qwen2moe's shared expert is a dense FFN added to the routed output. The
    // tensors are the evidence, and a half-present branch is an error rather
    // than something to skip.
    {
        const bool inp = doc.tensor_by_name.count("blk.0.ffn_gate_inp_shexp.weight") != 0;
        const bool gate = doc.tensor_by_name.count("blk.0.ffn_gate_shexp.weight") != 0;
        const bool up = doc.tensor_by_name.count("blk.0.ffn_up_shexp.weight") != 0;
        const bool down = doc.tensor_by_name.count("blk.0.ffn_down_shexp.weight") != 0;
        if (inp || gate || up || down) {
            if (!(inp && gate && up && down))
                block("SHARED_EXPERT_PARTIAL",
                      "the shared expert is only partly present; all four tensors are required");
            out.set("shared_expert", true);
        }
    }
    const MetadataValue *tied_flag = find(doc, arch + ".tie_word_embeddings");
    if (tied_flag && tied_flag->b && doc.tensor_by_name.count("output.weight") != 0) {
        block("AMBIGUOUS_TIED_OUTPUT",
              "the file declares tied embeddings but also ships a separate output tensor");
    }

    // Adapter row + tags. The probe is what a user runs before `run`, so it has
    // to name the row and the branch decisions in the same vocabulary the
    // engine's own errors use.
    {
        Json adapter = Json::object();
        adapter.set("id", "llama-family-v1");
        if (const AdapterRow *row = adapter_row(arch)) {
            adapter.set("arch", row->arch);
            adapter.set("upstream_enum", row->upstream_enum ? row->upstream_enum : "");
            adapter.set("graph", row->graph ? row->graph : "");
            adapter.set("block", row->block ? row->block : "");
        } else {
            adapter.set("arch", arch);
            adapter.set("upstream_enum", "");
        }
        // Branch tags need a ModelConfig, which does not exist for an unmapped
        // architecture. Report the facts the probe already established instead
        // of pretending to have resolved a block.
        Json tags = Json::array();
        tags.push(Json::string("arch=" + arch));
        tags.push(Json::string(has_adapter ? "block=llama-family" : "block=unmapped"));
        if (doc.tensor_by_name.count("blk.0.attn_sinks.weight") != 0) tags.push(Json::string("sinks=yes"));
        if (doc.tensor_by_name.count("blk.0.rope_freqs.weight") != 0) tags.push(Json::string("rope_freqs=yes"));
        if (doc.tensor_by_name.count("blk.0.attn_q_norm.weight") != 0)
            tags.push(Json::string("qk_norm=per-head"));
        if (has_attn_sub_norm && has_ffn_sub_norm) tags.push(Json::string("subln=yes"));
        if (has_projection_scale) tags.push(Json::string("proj_scale=yes"));
        if (doc.tensor_by_name.count("blk.0.ffn_gate_exps.weight") != 0) {
            tags.push(Json::string("moe=stacked"));
        } else if (has_per_expert_tensors(doc)) {
            // The layout is a storage fact the probe can settle before any
            // adapter runs: the expert index is in the name instead of in an
            // axis, which is the only difference a reader has to know about.
            tags.push(Json::string("moe=per-expert"));
        }
        adapter.set("tags", std::move(tags));
        out.set("adapter", std::move(adapter));
    }

    const int64_t blocker_count = static_cast<int64_t>(blockers_count);
    out.set("blockers", std::move(blockers));
    out.set("blocker_count", blocker_count);

    const bool runnable = has_adapter && unsupported_types.empty() &&
                          tokenizer_implemented(tokenizer_model) && blocker_count == 0;
    out.set("runnable", runnable);
    out.set("verdict", runnable ? "runnable"
                                : (all_dequantizable ? "inspectable-and-decodable"
                                                     : "inspect-only"));
    return out;
}

} // namespace fm
