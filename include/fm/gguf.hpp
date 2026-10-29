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
    // Gate and up fused into one expert tensor ([hidden, 2*width] per expert,
    // gate block first). cohere2moe, dots3note, gemma4 and deepseek3.2 use it.
    FFN_GATE_UP_EXPS,
    FFN_UP_EXPS,
    FFN_DOWN_EXPS,
    // Optional per-projection biases. qwen2 and several descendants carry them.
    ATTN_Q_BIAS,
    ATTN_K_BIAS,
    ATTN_V_BIAS,
    ATTN_OUTPUT_BIAS,
    // Per-head RMSNorm on q/k, which qwen3 applies before RoPE.
    // BitNet's SUBLN: RMSNorm applied to the attention output *before* the
    // output projection, and to the gated FFN output *before* ffn_down.
    ATTN_SUB_NORM,
    FFN_SUB_NORM,
    // bitnet.cpp also writes optional one-element `scale` tensors per
    // projection (llama.cpp's `TENSOR_NOT_REQUIRED` q/k/v/o/gate/up/down
    // scales). A file may carry none of them.
    PROJECTION_SCALE,
    ATTN_Q_NORM,
    ATTN_SINKS,
    ROPE_FREQS,
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
    FFN_GATE_BIAS,
    OUTPUT_BIAS,
    // MLA: q through a low-rank bottleneck, and the kv latent that each head
    // expands with its own slice of k_b / v_b.
    ATTN_Q_A,
    ATTN_Q_A_NORM,
    ATTN_Q_B,
    ATTN_KV_A_MQA,
    ATTN_KV_A_NORM,
    ATTN_K_B,
    ATTN_V_B,
    // ffn_exp_probs_b: added to the router output for *selection* only.
    FFN_EXP_PROBS_B,
    // ffn_gate_inp.bias: added to the router *logits*, before the gating
    // function, so it moves the probabilities of every expert (gpt-oss,
    // minimax). Distinct from FFN_EXP_PROBS_B, which never reaches a weight.
    FFN_ROUTER_BIAS,
    // Per-expert bias columns on the routed projections. Each holds one bias
    // vector per expert, so its shape is [width, n_expert]; this is what
    // llama.cpp adds with `ggml_add_id(..., selected_experts)`.
    FFN_GATE_EXPS_BIAS,
    FFN_UP_EXPS_BIAS,
    FFN_DOWN_EXPS_BIAS,
    // mistral3's adaptive RMSNorm MLP: a two-projection gelu_erf over a
    // sinusoidal delay embedding, applied to the normalised FFN input.
    FFN_NORM_ADA_DOWN,
    FFN_NORM_ADA_UP,
    // Recurrent (state-space) layers: llama.cpp's mamba/mamba2 block. The state
    // lives in the activation buffer, not in a kv cache, so these tensors carry
    // the shapes the state layout is derived from.
    SSM_IN,
    SSM_CONV1D,
    SSM_CONV1D_BIAS,
    SSM_X,
    SSM_DT,
    SSM_DT_BIAS,
    SSM_A,
    SSM_D,
    SSM_OUT,
    SSM_NORM,
    SSM_DT_NORM,
    SSM_B_NORM,
    SSM_C_NORM,
    // Tensors that are valid in the file but are not part of the main forward
    // pass (nextn/MTP heads, vision towers). They are named in the probe rather
    // than silently dropped.
    IGNORED,
};

const char *tensor_role_name(TensorRole role);

// True when this build has a forward pass for `arch` (the llama-family block).
// One list decides it, and `adapter_binding` in the tests requires it to agree
// with the adapter table both ways, so an architecture cannot be added to one
// and forgotten in the other.
bool is_llama_family(const std::string &arch);

// True when `kind` names an activation this build implements. Both the manifest
// builder and the engine consult this one list, so an activation cannot be
// half-added (reachable in one place, silently defaulted in the other).
bool is_known_activation(const std::string &kind);

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
    // Adapter tag for this tensor, in llama.cpp's vocabulary (`ATTN_Q`,
    // `FFN_GATE_EXPS`, …). Populated when the adapter maps the tensor, so a
    // failure names the same thing an upstream log would name.
    std::string tag;
    // The ggml operation this tensor feeds (`matmul`, `norm`, …), for reports.
    std::string op;
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
    // Attention sink: one extra learnable logit per head that the softmax
    // denominator includes and the output never reads. ggml calls it
    // `soft_max_ext(..., sinks)`. Presence of `blk.N.attn_sinks.weight` decides,
    // because it is a per-tensor fact rather than an architecture rule.
    bool attn_sinks = false;
    // Per-pair rotary frequency divisors (`blk.N.rope_freqs.weight`, n_rot/2
    // entries). llama.cpp only stores real data on layer 0 and duplicates it, so
    // one vector serves every layer.
    bool rope_freqs = false;
    // Every n-th layer skips rotary position entirely (SmolLM3's NoPE layers:
    // `n_no_rope_layer_step = 4`, i.e. layer 3, 7, …). 0 means "every layer
    // rotates", which is what all but that family do.
    uint32_t no_rope_layer_step = 0;
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
    // "silu" (SwiGLU), "gelu" (GeGLU), "swiglu-oai" (gpt-oss / minimax-m3: a
    // clamped gate times a clamped-up-plus-one) or "silu-clamp" (deepseek4/4.1,
    // glm5-next: silu(gate) clamped from above times a clamped up).
    std::string activation = "silu";
    // Clamp bounds for the two clamped variants. 0 means "not clamped", which
    // keeps every other family on the plain path.
    double activation_limit = 0.0;
    // `swiglu-oai` scales the sigmoid argument; 1.702 is the constant
    // llama.cpp uses for gpt-oss and minimax-m3.
    double activation_alpha = 0.0;
    // Per-layer clamp for `silu-clamp`; empty means "use activation_limit".
    std::vector<double> swiglu_clamp;
    // The clamp has two upstream forms and they disagree by design:
    //   fused      silu(min(gate, limit)) * clamp(up, +-limit)
    //   non-fused  min(silu(gate), limit) * clamp(up, +-limit)
    // llama.cpp picks per architecture (llama-graph.cpp:2449), so the choice is
    // a checkpoint-derived fact, not a style. A layer whose bound is <= 1e-6 is
    // unclamped in both.
    bool swiglu_clamp_fused = false;
    // Shared-expert bound, from `swiglu_clamp_shexp`; empty means "same as the
    // routed experts", which is what most files do.
    std::vector<double> swiglu_clamp_shexp;
    // BitNet's SUBLN: RMSNorm over the attention output before the output
    // projection, and over the gated FFN output before the down projection.
    // Both are inside the block, not after it, so they cannot be expressed with
    // the post-norm flag.
    bool sub_norms = false;
    // Optional one-element `*.scale` tensors multiply a projection's output.
    // bitnet.cpp writes them for its ternary weights; a file with none keeps
    // scale 1.0.
    bool projection_scales = false;
    // `hyper_connection.count` (llama.cpp's dsv4_hc_mult). dflash only selects
    // the fused clamp when it also carries compression connections, so the two
    // facts have to be kept together.
    double hyper_connection_count = 0.0;
    // Routed experts store gate and up in one tensor, gate block first.
    bool fused_gate_up_exps = false;
    // gemma2/3 clamp attention scores and/or final logits through tanh.
    double attn_logit_softcap = 0.0;
    double final_logit_softcap = 0.0;
    // gemma2/3 apply an extra RMSNorm after attention and after the FFN.
    bool post_norms = false;
    // --- recurrent (state-space) layers ---------------------------------------
    // `ssm_kind` is "mamba" (one decay per channel, an element-wise A) or
    // "mamba2" (a scalar decay per head and a grouped B/C). Empty means the file
    // has no recurrent layers at all, which is every other family.
    std::string ssm_kind;
    uint32_t ssm_conv = 0;     // d_conv, the 1D convolution width
    uint32_t ssm_inner = 0;    // d_inner, the expanded width the state lives in
    uint32_t ssm_state = 0;    // d_state
    uint32_t ssm_dt_rank = 0;  // dt_rank (mamba), n_head (mamba2)
    uint32_t ssm_groups = 1;   // n_group, the B/C groups
    bool ssm_dt_b_c_rms = false;
    // falcon-mamba and jamba normalise dt, B and C before the state update. The
    // tensors are the evidence, so this is read rather than assumed.
    bool ssm_vector_norms = false;
    // Per-layer recurrence. An empty list means every layer is recurrent, which
    // is what a pure mamba file says by simply having no attention tensors. A
    // hybrid stack (jamba, nemotron-h) states the split explicitly: in llama.cpp
    // it is `attention.recurrent_layers` or a per-layer `attention.head_count_kv`
    // whose zero entries mean "this layer has no attention at all".
    std::vector<uint8_t> recurrent_layers;
    // Per-layer feed-forward presence. A pure state-space model has none at all;
    // a hybrid may put one behind either kind of layer, so this is a mask rather
    // than a flag. Empty means every layer has one.
    std::vector<uint8_t> ffn_layers;
    bool has_ffn = true;
    // Per-layer routed-expert presence. Upstream marks `ffn_gate_inp` as
    // optional per layer, which is how a family can mix dense and MoE blocks
    // (jamba does). Empty means "follow model_kind and dense_lead", which is what
    // a uniform file says.
    std::vector<uint8_t> moe_layers;
    // jamba's attention layers carry no rotary position at all; upstream says so
    // by not calling the rope in its block, and the file says so by carrying no
    // `rope.dimension_count`.
    bool rope_disabled = false;

    bool layer_is_recurrent(uint32_t layer) const {
        if (ssm_kind.empty()) return false;
        if (recurrent_layers.empty()) return true;
        return layer < recurrent_layers.size() && recurrent_layers[layer] != 0;
    }
    bool layer_has_ffn(uint32_t layer) const {
        if (!has_ffn) return false;
        if (ffn_layers.empty()) return true;
        return layer < ffn_layers.size() && ffn_layers[layer] != 0;
    }
    bool layer_is_moe(uint32_t layer) const {
        // A file with no per-layer mask (the native reference family) uses the
        // model-wide rule: experts on every layer past the dense lead.
        if (moe_layers.empty()) return model_kind == "moe" && layer >= dense_lead;
        return layer < moe_layers.size() && moe_layers[layer] != 0;
    }
    // How many layers of each kind a stack has. The cache is laid out as one
    // region per kind, so an attention layer indexes the attention region by its
    // position among attention layers rather than by its own layer index.
    uint32_t attention_layer_count() const {
        uint32_t count = 0;
        for (uint32_t l = 0; l < layers; ++l)
            if (!layer_is_recurrent(l)) ++count;
        return count;
    }
    uint32_t recurrent_layer_count() const {
        uint32_t count = 0;
        for (uint32_t l = 0; l < layers; ++l)
            if (layer_is_recurrent(l)) ++count;
        return count;
    }
    uint32_t attention_slot(uint32_t layer) const {
        uint32_t slot = 0;
        for (uint32_t l = 0; l < layer && l < layers; ++l)
            if (!layer_is_recurrent(l)) ++slot;
        return slot;
    }
    uint32_t recurrent_slot(uint32_t layer) const {
        uint32_t slot = 0;
        for (uint32_t l = 0; l < layer && l < layers; ++l)
            if (layer_is_recurrent(l)) ++slot;
        return slot;
    }
    // Floats of per-position cache the attention layers need, at `context`
    // positions: one (key, value) pair per kv head per layer. The recurrent
    // layers contribute nothing here; their rolling state is fixed size.
    uint64_t attention_cache_floats(uint64_t context) const {
        const uint64_t kv_dim = static_cast<uint64_t>(kv_heads) * head_dim;
        return static_cast<uint64_t>(attention_layer_count()) * context * (kv_dim + v_dim());
    }
    uint64_t recurrent_state_floats() const {
        return static_cast<uint64_t>(recurrent_layer_count()) * ssm_state_floats();
    }
    // The channels the 1D convolution runs over: the state width for mamba, and
    // the state plus the grouped B and C for mamba2.
    uint32_t ssm_conv_channels() const {
        return ssm_kind == "mamba2" ? ssm_inner + 2 * ssm_groups * ssm_state : ssm_inner;
    }
    // Floats of recurrent state one recurrent layer needs: the convolution
    // history (d_conv - 1 vectors) plus one state vector of d_state per channel.
    uint64_t ssm_state_floats() const {
        if (ssm_kind.empty()) return 0;
        return static_cast<uint64_t>(ssm_conv - 1) * ssm_conv_channels() +
               static_cast<uint64_t>(ssm_state) * ssm_inner;
    }

    // gpt-oss normalises *after* the attention residual and feeds that to the
    // FFN, so its FFN norm is the tensor named `attn_post_norm` rather than
    // `ffn_norm`. The arithmetic is the ordinary pre-norm block; only the tensor
    // name differs, which is why this is a flag and not a second code path.
    bool ffn_norm_is_post = false;
    // gemma2/3 restrict most layers to a sliding attention window.
    uint32_t sliding_window = 0;
    // Per-layer window selection (`attention.sliding_window_pattern`). gpt-oss
    // alternates full and windowed layers, so a single model-wide window would
    // be wrong for half the block. Empty means "every layer uses the window",
    // which is what the families with no pattern key do.
    std::vector<uint8_t> swa_layers;
    bool layer_uses_window(uint32_t layer) const {
        if (sliding_window == 0) return false;
        if (swa_layers.empty()) return true;
        return layer < swa_layers.size() && swa_layers[layer] != 0;
    }
    // phi3 fuses q, k and v into one projection (and one optional bias).
    bool fused_qkv = false;
    // glm4 stores the FFN gate and up halves in one tensor of
    // [hidden, 2*intermediate]: gate first, then up (ggml_swiglu's convention).
    bool fused_gate_up = false;
    uint32_t bos_id = 0;
    uint32_t eos_id = 0;
    double rms_eps = 0.0;
    double rope_theta = 0.0;
    // Context extension, straight from `rope.scaling.*`. "none" is the identity
    // and is what every checkpoint without those keys gets.
    std::string rope_scaling_type = "none";
    // `rope.scaling.factor`: the position divisor. 0 means "unspecified", which
    // llama.cpp treats as 1.0 (no stretching).
    double rope_scaling_factor = 0.0;
    // `rope.scaling.original_context_length`: the trained context, which anchors
    // the YaRN correction ramp. Defaults to `context_length`.
    double rope_n_ctx_orig = 0.0;
    // `rope.scaling.attn_factor`: a magnitude multiplier the checkpoint itself
    // asks for, applied on top of whatever the scaling mode derives.
    double rope_scaling_attn_factor = 1.0;
    // `rope.scaling.yarn_log_multiplier`: YaRN's `mscale_all_dim`.
    double rope_yarn_log_mul = 0.0;
    // YaRN ramp knees. llama.cpp hard-codes these two as its defaults and never
    // reads the keys, so the defaults are what a real checkpoint runs with.
    double rope_yarn_beta_fast = 32.0;
    double rope_yarn_beta_slow = 1.0;
    // mRoPE: `{arch}.rope.dimension_sections`, four counts of rotary *pairs*
    // assigned to the time, height, width and extra axes. llama.cpp enables
    // mRoPE only when the first two are non-zero, so a file that carries the key
    // with zeros still rotates normally.
    std::array<uint32_t, 4> rope_sections = {0, 0, 0, 0};
    // Two section layouts exist. The plain one assigns runs of pairs in order
    // (MROPE); the interleaved one cycles t/t/y/x in triplets (IMROPE, qwen3-vl
    // and qwen3.5). They differ only in which axis each pair reads its position
    // from, so both are one switch here.
    bool mrope = false;
    bool mrope_interleaved = false;
    // Derived by `resolve_rope_scaling()` in src/rope.cpp; never read from the
    // file. `rope_freq_scale` keeps llama.cpp's direction (1/factor) so the two
    // implementations can be compared line by line.
    double rope_freq_scale = 1.0;
    double rope_ext_factor = 0.0;
    double rope_mscale = 1.0;
    // qwen2moe adds a dense feed-forward branch to the routed experts. Its
    // width is its own hyperparameter (`expert_shared_feed_forward_length`),
    // not the dense width, so it is stored separately. 0 means "no shared
    // expert", which keeps every other architecture's path unchanged.
    bool shared_expert = false;
    // qwen2moe scales the shared expert with one scalar per token; the deepseek
    // family adds it unscaled. The gate tensor's presence decides.
    bool shared_expert_gate = false;
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
    // A bias on the gate projection is its own flag: mistral3 carries one and
    // not the others, and adding it to the wrong projection changes every logit.
    bool ffn_gate_bias = false;
    bool attn_output_bias = false;  // bias on attn_output
    // mistral3's adaptive RMSNorm (voxtral realtime): the normalised FFN input
    // is scaled by (1 + mlp(t_cond)), where t_cond is a sinusoidal embedding of
    // the configured output delay and the mlp is a two-projection gelu_erf.
    bool ffn_norm_ada = false;
    uint32_t realtime_delay = 0;
    // `attention.temperature_scale`: llama.cpp's per-position logit temperature.
    // 0 disables it, which is what every file without the key gets.
    double attn_temp_scale = 0.0;
    // The floor of the temperature ramp, `realtime`/`temperature` files spell it
    // as `rope.scaling.original_context_length`; it must be non-zero when the
    // scale is set (upstream throws otherwise).
    double attn_temp_floor = 0.0;
    bool logit_bias = false;        // bias on the output projection

    // --- multi-head latent attention (the deepseek family) ------------------
    // The decoder never forms full keys and values: it projects into a
    // low-rank latent per token, and each head expands it. `head_dim` is the
    // *key* width (nope + rope); `head_v` is the value width, which differs.
    bool mla = false;
    uint32_t q_lora_rank = 0;   // 0 for the "lite" variant: q projected in one step
    uint32_t kv_lora_rank = 0;
    uint32_t head_v = 0;
    // Leading dense blocks: the first N layers use a plain FFN, the rest MoE.
    uint32_t dense_lead = 0;

    // MoE biases. Two different places, and conflating them changes the router:
    //   * `ffn_gate_inp.bias` is added to the router *logits*, before the gating
    //     function (llama.cpp build_moe_ffn: gate_inp_b);
    //   * `ffn_exp_probs_b.bias` is added to the probabilities *after* gating and
    //     only steers which experts are selected (deepseek V3).
    bool expert_router_bias = false;
    // Per-expert projection biases (`ffn_{gate,up,down}_exps.bias`, one column
    // per expert). gpt-oss carries all three.
    bool expert_projection_bias = false;
    // Router semantics. The gating function is not implied by "MoE": deepseek
    // V2 softmaxes, V3 and later use a sigmoid, and some families use
    // sqrt(softplus). A selection bias and group-limited routing change which
    // experts are picked without changing their weights.
    std::string expert_gating = "softmax";
    bool expert_selection_bias = false;
    // Floor added to the renormalisation denominator. llama.cpp clamps it to the
    // smallest F16 normal (6.103515625e-5) except for deepseek4.1, which adds
    // 1e-20 instead; the difference only shows when every selected weight is
    // tiny, which is exactly the case a comparison would otherwise miss.
    double expert_weights_epsilon = 6.103515625e-5;
    // Rearm the epsilon above the sum rather than comparing with `> 0`: a
    // denominator of zero is a router bug, not a reason to emit NaN.
    bool expert_weights_epsilon_additive = false;
    uint32_t expert_groups = 0;
    uint32_t expert_groups_used = 0;

    // Value width per head; everywhere outside MLA this equals head_dim.
    uint32_t head_v_dim() const { return head_v ? head_v : head_dim; }
    uint32_t v_dim() const { return kv_heads * head_v_dim(); }
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
    // Tensors that are valid in the file but outside the main forward pass,
    // with the reason. Reported rather than silently dropped.
    std::vector<std::pair<std::string, std::string>> ignored_tensors;

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
