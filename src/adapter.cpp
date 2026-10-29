#include "fm/adapter.hpp"

#include <algorithm>

namespace fm {
namespace {

// One row per architecture this build actually maps, with the upstream enum it
// corresponds to. Keeping the upstream name here is what makes an error message
// actionable: "deepseek2 (LLM_ARCH_DEEPSEEK2)" points at one file in llama.cpp.
const AdapterRow kRows[] = {
    {"freemoe-tiny", "n/a (native)", "freemoe", "native reference block"},
    {"llama", "LLM_ARCH_LLAMA", "llm_build_llama", "pre-norm RMSNorm, RoPE, SwiGLU, GQA"},
    {"tinyllama", "LLM_ARCH_LLAMA", "llm_build_llama", "the llama block under its own name"},
    {"smollm", "LLM_ARCH_LLAMA", "llm_build_llama", "the llama block under its own name"},
    {"smollm2", "LLM_ARCH_LLAMA", "llm_build_llama", "the llama block under its own name"},
    {"smollm3", "LLM_ARCH_SMOLLM3", "llm_build_smollm3", "llama block, every 4th layer skips RoPE"},
    {"qwen2", "LLM_ARCH_QWEN2", "llm_build_qwen2", "GQA plus q/k/v projection biases"},
    {"qwen3", "LLM_ARCH_QWEN3", "llm_build_qwen3", "per-head RMSNorm on q and k before RoPE"},
    {"qwen2moe", "LLM_ARCH_QWEN2MOE", "llm_build_qwen2moe", "routed experts plus a gated dense shared expert"},
    {"qwen3moe", "LLM_ARCH_QWEN3MOE", "llm_build_qwen3moe", "qwen3 attention with stacked-expert FFN"},
    {"gemma", "LLM_ARCH_GEMMA", "llm_build_gemma", "scaled embeddings, (1+w) norm, GeGLU, tied output"},
    {"gemma2", "LLM_ARCH_GEMMA2", "llm_build_gemma2", "gemma plus post-norms, softcap, sliding window"},
    {"gemma3", "LLM_ARCH_GEMMA3", "llm_build_gemma3", "gemma2 text path with per-layer windows"},
    {"phi2", "LLM_ARCH_PHI2", "llm_build_phi2", "LayerNorm with biases, gate-free GELU, shared residual"},
    {"phi3", "LLM_ARCH_PHI3", "llm_build_phi3", "fused q/k/v projection with bias, partial rotary"},
    {"glm4", "LLM_ARCH_GLM4", "llm_build_glm4", "fused gate+up FFN, post-attention and post-FFN norms"},
    {"minicpm", "LLM_ARCH_MINICPM", "llm_build_minicpm", "scaled embedding, tied output, ignored scales"},
    {"deepseek2", "LLM_ARCH_DEEPSEEK2", "llm_build_deepseek2", "MLA, dense lead, sigmoid/group routing, shared expert"},
    {"internlm2", "LLM_ARCH_INTERNLM2", "llm_build_internlm2", "llama block with its own norm biases"},
    {"exaone", "LLM_ARCH_EXAONE", "llm_build_exaone", "llama block with its own head layout"},
    {"orion", "LLM_ARCH_ORION", "llm_build_orion", "llama block with its own head layout"},
    {"gptneox", "LLM_ARCH_GPTNEOX", "llm_build_gptneox", "LayerNorm, gate-free GELU, parallel attention/FFN"},
    {"stablelm", "LLM_ARCH_STABLELM", "llm_build_stablelm", "llama block with its own rotary width"},
    {"bitnet", "LLM_ARCH_BITNET", "llm_build_bitnet", "llama block plus SUBLN sub-norms and optional projection scales"},
    {"arcee", "LLM_ARCH_ARCEE", "llm_build_arcee", "RMSNorm llama block, gate-free FFN with relu^2"},
    {"nemotron", "LLM_ARCH_NEMOTRON", "llm_build_nemotron", "LayerNorm llama block with biases, gate-free relu^2 FFN"},
    {"jais2", "LLM_ARCH_JAIS2", "llm_build_jais2", "LayerNorm llama block, biased projections, gate-free relu^2 FFN"},
    {"gpt-oss", "LLM_ARCH_OPENAI_MOE", "llama_model_openai_moe::build_arch_graph",
     "sinks, post-attention norm as FFN norm, biased MoE, swiglu_oai, alternating windows"},
    {"plm", "LLM_ARCH_PLM", "llama_model_plm::build_arch_graph",
     "MLA with one-step q, gate-free relu^2 FFN, tied output"},
    {"qwen3vl", "LLM_ARCH_QWEN3VL", "llama_model_qwen3vl::build_arch_graph",
     "qwen3 text path with mRoPE sections"},
    {"qwen3vlmoe", "LLM_ARCH_QWEN3VLMOE", "llama_model_qwen3vlmoe::build_arch_graph",
     "qwen3moe text path with mRoPE sections"},
    {"deepseek4", "LLM_ARCH_DEEPSEEK4", "llama_model_deepseek4::build_arch_graph",
     "MLA plus per-layer swiglu clamp bounds"},
    {"deepseek41", "LLM_ARCH_DEEPSEEK41", "llama_model_deepseek41::build_arch_graph",
     "deepseek4 graph with the renormalised router"},
    {"mistral3", "LLM_ARCH_MISTRAL3", "llama_model_mistral3::build_arch_graph",
     "llama block, optional MLP biases, gelu_erf adaptive norm, attention temperature"},
};

}  // namespace

const AdapterRow *adapter_row(const std::string &arch) {
    for (const AdapterRow &row : kRows)
        if (arch == row.arch) return &row;
    return nullptr;
}

std::vector<AdapterRow> adapter_rows() {
    return std::vector<AdapterRow>(std::begin(kRows), std::end(kRows));
}

const char *adapter_tensor_tag(TensorRole role) {
    // Upstream names a handful of roles differently from this build's internal
    // spelling. Mapping them here keeps the report comparable with a llama.cpp
    // log instead of merely recognisable.
    switch (role) {
    case TensorRole::ATTN_OUTPUT: return "ATTN_OUT";
    case TensorRole::FFN_ROUTER: return "FFN_GATE_INP";
    case TensorRole::ATTN_QKV: return "ATTN_QKV";
    case TensorRole::ATTN_Q_A: return "ATTN_Q_A";       // q_lora, deepseek2
    case TensorRole::ATTN_KV_A_MQA: return "ATTN_KV_A_MQA";
    case TensorRole::ATTN_K_B: return "ATTN_K_B";
    case TensorRole::ATTN_V_B: return "ATTN_V_B";
    case TensorRole::FFN_EXP_PROBS_B: return "FFN_EXP_PROBS_B";
    case TensorRole::FFN_ROUTER_BIAS: return "FFN_GATE_INP";       // + ".bias"
    case TensorRole::FFN_GATE_BIAS: return "FFN_GATE_BIAS";
    case TensorRole::FFN_NORM_ADA_DOWN: return "FFN_NORM_ADA_DOWN";
    case TensorRole::FFN_NORM_ADA_UP: return "FFN_NORM_ADA_UP";
    case TensorRole::FFN_GATE_EXPS_BIAS: return "FFN_GATE_EXPS";   // + ".bias"
    case TensorRole::FFN_UP_EXPS_BIAS: return "FFN_UP_EXPS";       // + ".bias"
    case TensorRole::FFN_DOWN_EXPS_BIAS: return "FFN_DOWN_EXPS";   // + ".bias"
    case TensorRole::OUTPUT: return "OUTPUT";
    case TensorRole::OUTPUT_NORM: return "OUTPUT_NORM";
    case TensorRole::TOKEN_EMBD: return "TOKEN_EMBD";
    default: return tensor_role_name(role);
    }
}

const char *adapter_tensor_op(TensorRole role) {
    switch (role) {
    case TensorRole::TOKEN_EMBD: return "get_rows";
    case TensorRole::OUTPUT: return "matmul";
    case TensorRole::OUTPUT_NORM:
    case TensorRole::ATTN_NORM:
    case TensorRole::FFN_NORM:
    case TensorRole::ATTN_POST_NORM:
    case TensorRole::FFN_POST_NORM:
    case TensorRole::ATTN_Q_NORM:
    case TensorRole::ATTN_K_NORM:
    case TensorRole::ATTN_Q_A_NORM:
    case TensorRole::ATTN_KV_A_NORM:
    case TensorRole::ATTN_SUB_NORM:
    case TensorRole::FFN_SUB_NORM: return "norm";
    case TensorRole::PROJECTION_SCALE: return "mul";
    case TensorRole::ATTN_Q:
    case TensorRole::ATTN_K:
    case TensorRole::ATTN_V:
    case TensorRole::ATTN_QKV:
    case TensorRole::ATTN_OUTPUT:
    case TensorRole::ATTN_Q_A:
    case TensorRole::ATTN_Q_B:
    case TensorRole::ATTN_KV_A_MQA:
    case TensorRole::ATTN_K_B:
    case TensorRole::ATTN_V_B:
    case TensorRole::FFN_GATE:
    case TensorRole::FFN_UP:
    case TensorRole::FFN_DOWN:
    case TensorRole::FFN_GATE_SHEXP:
    case TensorRole::FFN_UP_SHEXP:
    case TensorRole::FFN_DOWN_SHEXP: return "matmul";
    case TensorRole::FFN_ROUTER:
    case TensorRole::FFN_GATE_INP_SHEXP: return "matmul+softmax";
    case TensorRole::FFN_GATE_EXPS:
    case TensorRole::FFN_UP_EXPS:
    case TensorRole::FFN_DOWN_EXPS: return "matmul (per selected expert)";
    case TensorRole::FFN_ROUTER_BIAS: return "add (router logits)";
    case TensorRole::FFN_GATE_EXPS_BIAS:
    case TensorRole::FFN_UP_EXPS_BIAS:
    case TensorRole::FFN_DOWN_EXPS_BIAS: return "add (per selected expert)";
    case TensorRole::ATTN_SINKS: return "softmax denominator";
    case TensorRole::ROPE_FREQS: return "rope divisor";
    default: return "add";
    }
}

std::vector<std::string> adapter_config_tags(const ModelConfig &cfg) {
    std::vector<std::string> tags;
    tags.push_back("arch=" + cfg.architecture);
    if (const AdapterRow *row = adapter_row(cfg.architecture)) {
        if (row->upstream_enum) tags.push_back(std::string("upstream=") + row->upstream_enum);
        if (row->graph) tags.push_back(std::string("graph=") + row->graph);
    }
    tags.push_back("adapter=" + cfg.adapter);
    tags.push_back("kind=" + cfg.model_kind);

    // Attention shape.
    if (cfg.kv_heads == cfg.heads) tags.push_back("attn=mha");
    else if (cfg.kv_heads == 1) tags.push_back("attn=mqa");
    else tags.push_back("attn=gqa");
    if (cfg.mla) tags.push_back("attn=mla");
    if (cfg.qk_norm) tags.push_back("qk_norm=per-head");
    if (cfg.attn_sinks) tags.push_back("sinks=yes");
    if (cfg.ffn_norm_is_post) tags.push_back("ffn_norm=post_attention");
    if (cfg.expert_router_bias) tags.push_back("router_bias=yes");
    if (cfg.expert_projection_bias) tags.push_back("expert_bias=per_expert");
    if (!cfg.swa_layers.empty()) tags.push_back("window=per_layer");
    if (cfg.ffn_norm_ada) tags.push_back("ffn_norm=adaptive");
    if (cfg.attn_temp_scale != 0.0) tags.push_back("attn_temp=on");
    if (cfg.ffn_gate_bias) tags.push_back("bias=ffn_gate");
    if (cfg.attn_qkv_bias) tags.push_back("bias=qkv");
    if (cfg.fused_qkv) tags.push_back("qkv=fused");
    if (cfg.sliding_window) tags.push_back("window=" + std::to_string(cfg.sliding_window));
    if (cfg.attn_logit_softcap > 0.0) tags.push_back("attn_softcap=on");

    // Rotary.
    tags.push_back("rope=" + (cfg.rope_dim && cfg.rope_dim != cfg.head_dim ? std::string("partial")
                                                                          : std::string("full")));
    if (cfg.rope_scaling_type != "none")
        tags.push_back("rope_scaling=" + cfg.rope_scaling_type);
    if (cfg.rope_freqs) tags.push_back("rope_freqs=yes");
    if (cfg.no_rope_layer_step) tags.push_back("nope_step=" + std::to_string(cfg.no_rope_layer_step));
    if (cfg.mrope) {
        // The section split is the part a failure report needs: a wrong split
        // rotates the wrong pairs and no other symptom points at it.
        tags.push_back(std::string("mrope=") +
                       (cfg.mrope_interleaved ? "interleaved" : "sections") + "[" +
                       std::to_string(cfg.rope_sections[0]) + "," +
                       std::to_string(cfg.rope_sections[1]) + "," +
                       std::to_string(cfg.rope_sections[2]) + "," +
                       std::to_string(cfg.rope_sections[3]) + "]");
    }

    // Normalisation and activation.
    tags.push_back(std::string("norm=") + (cfg.layer_norm ? "layernorm" : "rmsnorm"));
    if (cfg.norm_add_one) tags.push_back("norm=zero-centred");
    if (cfg.post_norms) tags.push_back("post_norms=yes");
    if (cfg.sub_norms) tags.push_back("subln=yes");
    if (cfg.projection_scales) tags.push_back("proj_scale=yes");
    tags.push_back("ffn=" + cfg.activation);
    if (cfg.activation_limit > 0.0)
        tags.push_back("ffn_clamp=" + std::to_string(cfg.activation_limit));
    if (cfg.activation_alpha > 0.0)
        tags.push_back("ffn_alpha=" + std::to_string(cfg.activation_alpha));
    if (cfg.fused_gate_up_exps) tags.push_back("ffn_exps=fused-gate-up");

    // Feed-forward shape.
    if (cfg.fused_gate_up) tags.push_back("ffn=fused-gate-up");
    if (cfg.ffn_no_gate) tags.push_back("ffn=no-gate");
    if (cfg.expert_count) {
        tags.push_back("moe=stacked");
        tags.push_back("experts=" + std::to_string(cfg.expert_count) + "x" +
                       std::to_string(cfg.expert_top_k));
        tags.push_back("gating=" + cfg.expert_gating);
        if (cfg.expert_groups) tags.push_back("groups=" + std::to_string(cfg.expert_groups) + "/" +
                                              std::to_string(cfg.expert_groups_used));
        if (cfg.expert_selection_bias) tags.push_back("selection_bias=yes");
        if (cfg.shared_expert) tags.push_back("shared_expert=yes");
    }

    // Output.
    if (cfg.embedding_scale != 1.0) tags.push_back("embedding_scale=on");
    if (cfg.final_logit_softcap > 0.0) tags.push_back("logit_softcap=on");
    tags.push_back("tokenizer=" + cfg.tokenizer_model);
    return tags;
}

}  // namespace fm
