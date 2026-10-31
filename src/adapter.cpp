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
    {"gpt2", "LLM_ARCH_GPT2", "llama_model_gpt2::build_arch_graph",
     "absolute position table, gate-free GELU, no rotary"},
    {"starcoder", "LLM_ARCH_STARCODER", "llama_model_starcoder::build_arch_graph",
     "absolute position table, gate-free GELU, no rotary"},
    {"falcon", "LLM_ARCH_FALCON", "llama_model_falcon::build_arch_graph",
     "second pre-attention norm (falcon-40b)"},
    {"mpt", "LLM_ARCH_MPT", "llama_model_mpt::build_arch_graph",
     "optional absolute position table, gate-free GELU, no rotary"},
    {"seed-oss", "LLM_ARCH_SEED_OSS", "llama_model_seed_oss::build_arch_graph",
     "post-attention norm as the FFN norm"},
    {"plamo", "LLM_ARCH_PLAMO", "llama_model_plamo::build_arch_graph",
     "shared attention/FFN norm, both residuals added separately"},
    {"smallthinker", "LLM_ARCH_SMALLTHINKER", "llama_model_smallthinker::build_arch_graph",
     "ReLU-routed experts with renormalised weights"},
    {"command-r", "LLM_ARCH_COMMAND_R", "llama_model_command_r::build_arch_graph",
     "per-head q/k LayerNorm, logit scale"},
    {"olmo", "LLM_ARCH_OLMO", "llama_model_olmo::build_arch_graph",
     "weightless norms"},
    {"phimoe", "LLM_ARCH_PHIMOE", "llama_model_phi3::graph",
     "phi3's block with routed experts"},
    {"baichuan", "LLM_ARCH_BAICHUAN", "llama_model_baichuan::graph",
     "llama block; the 40-layer (13B) file uses ALiBi and no rope"},
    {"chatglm", "LLM_ARCH_CHATGLM", "llama_model_chatglm::graph",
     "RMSNorm block with a fused gate/up FFN"},
    {"olmo2", "LLM_ARCH_OLMO2", "llama_model_olmo2::build_arch_graph",
     "post-norms only, whole-vector q/k norm"},
    {"olmoe", "LLM_ARCH_OLMOE", "llama_model_olmoe::build_arch_graph",
     "whole-vector q/k norm, routed experts"},
    {"granite", "LLM_ARCH_GRANITE", "llama_model_granite::graph",
     "scaled residual branches, logits divided by logit_scale"},
    {"granitemoe", "LLM_ARCH_GRANITE_MOE", "llama_model_granite::graph",
     "granite's block with routed and shared experts"},
    // bailingmoe (src/models/bailingmoe.cpp): the llama block with routed
    // experts, a sigmoid-free softmax router that may renormalise and rescale its
    // weights, and a shared expert that is *added* rather than gated -- there is
    // no `ffn_gate_inp_shexp` tensor in the file at all.
    {"bailingmoe", "LLM_ARCH_BAILINGMOE", "llama_model_bailingmoe::build_arch_graph",
     "softmax router with optional weight norm/scale plus an ungated shared expert"},
    // bailingmoe2 (src/models/bailingmoe2.cpp): the same feed-forward with a fused
    // q/k/v projection, a per-head RMSNorm on q and k before the rotary, dense
    // leading blocks, an optional selection bias on the router and a real MTP head.
    {"bailingmoe2", "LLM_ARCH_BAILINGMOE2", "llama_model_bailingmoe2::build_arch_graph",
     "fused qkv, per-head q/k norm, dense lead, selection bias, MTP head"},
    // dbrx (src/models/dbrx.cpp): LayerNorm instead of RMSNorm, a fused q/k/v
    // projection, and *one* post-attention norm that normalises the feed-forward
    // input (the file calls it `attn_out_norm`).
    {"dbrx", "LLM_ARCH_DBRX", "llama_model_dbrx::build_arch_graph",
     "LayerNorm block, fused qkv, post-attention norm as the FFN norm"},
    // deepseek (src/models/deepseek.cpp): the pre-norm block with a dense lead, a
    // softmax router whose weights are *not* renormalised, a scale, and an ungated
    // shared expert. Tied output is optional.
    {"deepseek", "LLM_ARCH_DEEPSEEK", "llama_model_deepseek::build_arch_graph",
     "dense lead, unnormalised softmax router, ungated shared expert"},
    // dots1 (src/models/dots1.cpp): per-head q/k norm before the rope, dense lead,
    // router with an optional selection bias and metadata-driven gating, ungated
    // shared expert.
    {"dots1", "LLM_ARCH_DOTS1", "llama_model_dots1::build_arch_graph",
     "per-head q/k norm, dense lead, selection bias, ungated shared expert"},
    // ernie4-5-moe (src/models/ernie4-5-moe.cpp): every Nth layer is MoE rather
    // than every layer past a dense lead; the router renormalises as a constant.
    {"ernie4-5-moe", "LLM_ARCH_ERNIE4_5_MOE", "llama_model_ernie4_5_moe::build_arch_graph",
     "interleaved MoE layers, constant weight norm, optional shared expert"},
    // glm4-moe (src/models/glm4-moe.cpp): post-attention norm as the FFN norm,
    // sigmoid routing by default, optional per-head q/k norm, shared expert.
    {"glm4-moe", "LLM_ARCH_GLM4_MOE", "llama_model_glm4_moe::build_arch_graph",
     "sigmoid routing, post-attention FFN norm, shared expert, optional q/k norm"},
    // minimax-m2 (src/models/minimax-m2.cpp): per-head q/k norm, router selection
    // bias, constant weight norm, no shared expert.
    {"minimax-m2", "LLM_ARCH_MINIMAX_M2", "llama_model_minimax_m2::build_arch_graph",
     "per-head q/k norm, selection bias, constant router weight norm"},

    {"bloom", "LLM_ARCH_BLOOM", "llama_model_bloom::build_arch_graph",
     "ALiBi attention, biased LayerNorm, embedding norm"},
    {"jais", "LLM_ARCH_JAIS", "llama_model_jais::build_arch_graph",
     "ALiBi attention, no rotary, biased LayerNorm"},
    {"codeshell", "LLM_ARCH_CODESHELL", "llama_model_codeshell::build_arch_graph",
     "llama block with its own metadata"},
    {"cohere2", "LLM_ARCH_COHERE2", "llama_model_cohere2::build_arch_graph",
     "llama block with its own metadata"},
    {"deci", "LLM_ARCH_DECI", "llama_model_deci::build_arch_graph",
     "llama block with its own metadata"},
    {"hunyuan-dense", "LLM_ARCH_HUNYUAN_DENSE", "llama_model_hunyuan-dense::build_arch_graph",
     "llama block with its own metadata"},
    {"llama-embed", "LLM_ARCH_LLAMA_EMBED", "llama_model_llama-embed::build_arch_graph",
     "llama block with its own metadata"},
    {"maincoder", "LLM_ARCH_MAINCODER", "llama_model_maincoder::build_arch_graph",
     "llama block with its own metadata"},
    {"nanbeige", "LLM_ARCH_NANBEIGE", "llama_model_nanbeige::build_arch_graph",
     "llama block with its own metadata"},
    {"paddleocr", "LLM_ARCH_PADDLEOCR", "llama_model_paddleocr::build_arch_graph",
     "llama block with its own metadata"},
    {"pockettts", "LLM_ARCH_POCKETTTS", "llama_model_pockettts::build_arch_graph",
     "llama block with its own metadata"},
    {"qwen", "LLM_ARCH_QWEN", "llama_model_qwen::build_arch_graph",
     "llama block with its own metadata"},
    {"starcoder2", "LLM_ARCH_STARCODER2", "llama_model_starcoder2::build_arch_graph",
     "llama block with its own metadata"},
    {"xverse", "LLM_ARCH_XVERSE", "llama_model_xverse::build_arch_graph",
     "llama block with its own metadata"},
    {"mamba", "LLM_ARCH_MAMBA", "llm_build_mamba_base::build_mamba_layer",
     "1D convolution plus a state-space scan, element-wise decay, no attention"},
    {"mamba2", "LLM_ARCH_MAMBA2", "llm_build_mamba_base::build_mamba2_layer",
     "grouped scan with a scalar decay and a grouped norm, no attention"},
    {"jamba", "LLM_ARCH_JAMBA", "llama_model_jamba::build_arch_graph",
     "mamba layers interleaved with attention layers that carry no rotary"},
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
    // The absolute position table is read a row at a time, like the token table.
    case TensorRole::POSITION_EMBD: return "get_rows";
    case TensorRole::OUTPUT: return "matmul";
    case TensorRole::OUTPUT_NORM:
    case TensorRole::TOKEN_EMBD_NORM:
    case TensorRole::TOKEN_EMBD_NORM_BIAS:
    case TensorRole::ATTN_NORM_2:
    case TensorRole::ATTN_NORM_2_BIAS:
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

    // Attention shape. A pure recurrent file has no attention to describe; a
    // hybrid has both, so both sets of tags are emitted and the reader can see
    // which layers run which.
    const bool attention_layers = cfg.attention_layer_count() != 0;
    if (attention_layers) {
        if (cfg.kv_heads == cfg.heads) tags.push_back("attn=mha");
        else if (cfg.kv_heads == 1) tags.push_back("attn=mqa");
        else tags.push_back("attn=gqa");
        if (cfg.mla) tags.push_back("attn=mla");
    }
    if (cfg.qk_norm) tags.push_back("qk_norm=per-head");
    if (cfg.attn_sinks) tags.push_back("sinks=yes");
    if (cfg.ffn_norm_is_post) tags.push_back("ffn_norm=post_attention");
    if (cfg.expert_router_bias) tags.push_back("router_bias=yes");
    if (cfg.expert_projection_bias) tags.push_back("expert_bias=per_expert");
    if (!cfg.swa_layers.empty()) tags.push_back("window=per_layer");
    if (cfg.ffn_norm_ada) tags.push_back("ffn_norm=adaptive");
    if (!cfg.ssm_kind.empty()) {
        // A recurrent block has no attention at all, so this is the tag a reader
        // needs to see instead of the attention shape.
        tags.push_back("recurrent=" + cfg.ssm_kind);
        tags.push_back("ssm=conv" + std::to_string(cfg.ssm_conv) + "/state" +
                       std::to_string(cfg.ssm_state) + "/inner" + std::to_string(cfg.ssm_inner));
        if (cfg.ssm_groups > 1) tags.push_back("ssm_groups=" + std::to_string(cfg.ssm_groups));
        if (cfg.ssm_vector_norms) tags.push_back("ssm_dt_b_c_rms=yes");
        if (!cfg.has_ffn) tags.push_back("ffn=none");
        if (cfg.recurrent_layer_count() != cfg.layers)
            tags.push_back("recurrent_layers=" + std::to_string(cfg.recurrent_layer_count()) +
                           "/" + std::to_string(cfg.layers));
    }
    if (cfg.attn_temp_scale != 0.0) tags.push_back("attn_temp=on");
    if (cfg.ffn_gate_bias) tags.push_back("bias=ffn_gate");
    if (cfg.attn_qkv_bias) tags.push_back("bias=qkv");
    if (cfg.fused_qkv) tags.push_back("qkv=fused");
    if (cfg.sliding_window) tags.push_back("window=" + std::to_string(cfg.sliding_window));
    if (cfg.attn_logit_softcap > 0.0) tags.push_back("attn_softcap=on");

    // Rotary. A recurrent layer has none, so the tag is omitted rather than
    // claiming a full-width rotation that never runs; jamba has attention layers
    // that rotate nothing, which is a different fact again and says so.
    if (cfg.rope_disabled) tags.push_back("rope=none");
    else if (attention_layers)
        tags.push_back("rope=" + (cfg.rope_dim && cfg.rope_dim != cfg.head_dim
                                      ? std::string("partial")
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
    // A pure recurrent block has no feed-forward, so the activation is not a
    // fact about it; `ffn=none` above already says so.
    if (cfg.has_ffn) tags.push_back("ffn=" + cfg.activation);
    if (cfg.has_ffn && cfg.activation_limit > 0.0)
        tags.push_back("ffn_clamp=" + std::to_string(cfg.activation_limit));
    if (cfg.has_ffn && cfg.activation_alpha > 0.0)
        tags.push_back("ffn_alpha=" + std::to_string(cfg.activation_alpha));
    if (cfg.fused_gate_up_exps) tags.push_back("ffn_exps=fused-gate-up");

    // Feed-forward shape.
    if (cfg.fused_gate_up) tags.push_back("ffn=fused-gate-up");
    if (cfg.ffn_no_gate) tags.push_back("ffn=no-gate");
    if (cfg.expert_count) {
        // The tag names the storage layout, because it is what a reader has to
        // know to find an expert's bytes: one stacked tensor with an expert axis,
        // or one tensor per expert.
        tags.push_back(cfg.any_per_expert_layout() ? "moe=per-expert" : "moe=stacked");
        // A hybrid can place its experts on one kind of layer only (jamba mixes
        // dense and MoE blocks), so the count of routed layers is a fact about
        // the file that a per-model tag would hide.
        uint32_t routed = 0;
        for (uint32_t l = 0; l < cfg.layers; ++l)
            if (cfg.layer_is_moe(l)) ++routed;
        if (routed != cfg.layers) tags.push_back("moe_layers=" + std::to_string(routed) + "/" +
                                                 std::to_string(cfg.layers));
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
