#!/usr/bin/env python3
"""Independent reference implementation of the llama-family forward pass.

This is deliberately a *second* implementation. It shares no code with
`src/engine.cpp`, so a disagreement between the two is a real disagreement
about the architecture, not a shared typo. The C++ engine is checked against the
goldens this file produces.

It exists because the acceptance rule for a new architecture is parity with an
independent implementation, not "it produced words that look plausible".

Supported here, matching llama.cpp's definitions:

* pre-norm RMSNorm blocks;
* RoPE applied to each head independently ("half" rotation), with the rotary
  width taken from `rope.dimension_count` when present;
* grouped-query attention (`attention.head_count_kv` < `attention.head_count`);
* optional query/key/value projection biases (`attention.qkv_bias`), which is
  the qwen2 family;
* SwiGLU feed-forward.

Weights are read as F32 so the arithmetic is unambiguous; the quantised decoders
are verified separately by `tools/quant_reference.py`.
"""
from __future__ import annotations

import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_format import parse_gguf  # noqa: E402


class ReferenceError(RuntimeError):
    pass


def _f32_values(data: bytes) -> list[float]:
    count = len(data) // 4
    return list(struct.unpack(f"<{count}f", data[: count * 4]))


def _window_layers(md: dict[str, Any], arch: str, n_layer: int, window: int) -> list[bool]:
    """Per-layer sliding-window flags, mirroring the engine's adapter.

    `attention.sliding_window_pattern` is either a per-layer array or an integer
    period; llama.cpp's period form marks every layer whose index is not a
    multiple of the period as windowed. An empty list means "every layer".
    """
    value = md.get(f"{arch}.attention.sliding_window_pattern")
    if window == 0 or value is None:
        return []
    if isinstance(value, list):
        return [bool(v) for v in value]
    period = int(value)
    return [(layer % period) != 0 for layer in range(n_layer)]


def _rope_layers(md: dict[str, Any], arch: str, n_layer: int) -> list[bool]:
    """Per-layer rotary on/off, from `attention.rope_pattern` (flint).

    Empty means the file states no per-layer pattern, in which case the periodic
    `no_rope_step` rule applies instead.
    """
    value = md.get(f"{arch}.attention.rope_pattern")
    if value is None:
        return []
    return [bool(v) for v in value]


def _rope_theta_layers(md: dict[str, Any], arch: str, n_layer: int, window: int,
                       window_layers: list[bool], theta: float) -> list[float]:
    """Per-layer rotary bases, from `rope.freq_base_swa`.

    A windowed layer rotates at the second base and the rest at the file's single
    one. Empty means the file states no second base.
    """
    swa = md.get(f"{arch}.rope.freq_base_swa")
    if swa is None or float(swa) <= 0.0 or float(swa) == theta:
        return []
    layers = []
    for layer in range(n_layer):
        windowed = (window_layers[layer] if window_layers else True) if window else False
        layers.append(float(swa) if windowed else theta)
    return layers


# The names llama.cpp has used for the two sandwich-normalisation tensors, plus
# dbrx's `attn_out_norm`, which is the same slot: the norm written after the
# attention residual that normalises the feed-forward input. The rename is applied
# when the checkpoint is read, exactly as the loader does, so every
# tensor-presence test below sees one spelling. Renaming only at lookup time is a
# real bug: a dbrx block then looks like phi2's parallel residual, because it has
# no tensor called `ffn_norm`.
RENAMED_TENSORS = (
    ("post_attention_norm", "attn_post_norm"),
    ("post_ffw_norm", "ffn_post_norm"),
    ("attn_out_norm", "attn_post_norm"),
)


def canonical_name(name: str) -> str:
    """Normalise a post-norm tensor name to this module's spelling."""
    stem, _, suffix = name.rpartition(".")
    for new, old in RENAMED_TENSORS:
        if stem.endswith("." + new):
            return stem[: -len(new)] + old + "." + suffix
    return name


def load_model(path: Path) -> dict[str, Any]:
    """Reads a standard llama.cpp-style checkpoint (architecture-prefixed keys)."""
    doc = parse_gguf(Path(path).read_bytes())
    md = doc["metadata"]
    if "general.architecture" not in md:
        raise ReferenceError("missing general.architecture")
    arch = md["general.architecture"]

    def key(name: str, default: Any = None) -> Any:
        for candidate in (f"{arch}.{name}", f"freemoe.{name}", name):
            if candidate in md:
                return md[candidate]
        if default is None:
            raise ReferenceError(f"missing metadata {arch}.{name}")
        return default

    tensors: dict[str, tuple[list[int], list[float]]] = {}
    for t in doc["tensors"]:
        if t["type"] != 0:
            raise ReferenceError(f"this reference reads F32 only, got {t['type']}")
        tensors[canonical_name(t["name"])] = (list(t["dims"]), _f32_values(t["data"]))

    hidden = int(key("embedding_length"))
    # A NextN / MTP head appends its layers to `block_count`: llama.cpp reads that
    # key into `n_layer_all` and runs `n_layer_all - n_layer_nextn` trunk layers,
    # building the head only when the loader asks for it. The reference runs the
    # trunk for the same reason the engine does, and reads the same key, so a
    # file with a head cannot be checked by an oracle that counts every block.
    n_layer = int(key("block_count")) - int(key("nextn_predict_layers", 0))
    # Which layers are recurrent, read from the tensors exactly as the engine
    # does: a layer with `ssm_in` has no attention, and the layers without one
    # are an ordinary attention stack. A pure mamba file is all-recurrent; a
    # hybrid (jamba) mixes the two, so every block fact below is read from a
    # layer that actually runs that block.
    recurrent = [f"blk.{l}.ssm_in.weight" in tensors for l in range(n_layer)]
    has_ssm = any(recurrent)
    has_attn = not all(recurrent)
    probe = next(l for l in range(n_layer) if not recurrent[l]) if has_attn else 0
    first_recr = next(l for l in range(n_layer) if recurrent[l]) if has_ssm else 0

    def blk(name: str) -> str:
        return f"blk.{probe}.{name}"

    def has(name: str) -> bool:
        return blk(name) in tensors

    def blk_rec(name: str) -> str:
        return f"blk.{first_recr}.{name}"

    def moe_layer(layer: int) -> bool:
        # Upstream makes `ffn_gate_inp` optional per layer, which is how one stack
        # mixes dense and MoE blocks.
        return f"blk.{layer}.ffn_gate_inp.weight" in tensors

    def ffn_layer(layer: int) -> bool:
        for suffix in ("ffn_up.weight", "ffn_gate_up.weight", "ffn_up_exps.weight",
                       "ffn_gate_up_exps.weight", "ffn_down_exps.weight",
                       "ffn_up.0.weight", "ffn_down.0.weight"):
            if f"blk.{layer}.{suffix}" in tensors:
                return True
        return False

    heads = int(key("attention.head_count")) if has_attn else 0
    hd = int(key("attention.key_length", hidden // heads)) if has_attn else 0
    kv_head_key = key("attention.head_count_kv", heads) if has_attn else 0
    if isinstance(kv_head_key, list):
        # Per-layer array form: zero entries are the recurrent layers.
        values = {int(v) for v in kv_head_key if int(v) != 0}
        if len(values) != 1:
            raise ReferenceError("per-layer attention.head_count_kv values differ")
        kv_head_key = values.pop()
    # The dt/B/C norms are mamba-1 only. Upstream's gate is
    # `ssm.dt_b_c_rms || (the three weight tensors)` (models/mamba-base.cpp:99)
    # and `build_norm` multiplies only when the weight is present, so a norm
    # without its weight degenerates to unit gain rather than disappearing.
    ssm_a_rec = blk_rec("ssm_a")
    ssm_kind = ("mamba2" if (ssm_a_rec in tensors and tensors[ssm_a_rec][0][0] == 1)
                else "mamba" if has_ssm else "")
    ssm_norms_present = tuple(blk_rec(s) in tensors
                              for s in ("ssm_dt_norm.weight", "ssm_b_norm.weight",
                                        "ssm_c_norm.weight"))
    if ssm_kind == "mamba":
        ssm_vector_norms = bool(key("ssm.dt_b_c_rms", False)) or all(ssm_norms_present)
    else:
        ssm_vector_norms = False

    cfg = {
        "arch": arch,
        "H": hidden,
        "I": int(key("feed_forward_length")) if has_attn else 0,
        # The trunk only: a head layer is in `block_count` but never runs.
        "L": n_layer,
        "heads": heads,
        "kv_heads": int(kv_head_key) if has_attn else 0,
        "hd": hd,
        "V": int(key("vocab_size", len(md.get("tokenizer.ggml.tokens", [])))),
        "ctx": int(key("context_length")),
        "eps": float(key("attention.layer_norm_epsilon", 1e-5)
                     if f"{arch}.attention.layer_norm_epsilon" in md
                     else key("attention.layer_norm_rms_epsilon", 1e-5)),
        "theta": float(key("rope.freq_base", 10000.0)),
        "rope_dim": int(key("rope.dimension_count", hd)),
        # Context extension, read from the same `rope.scaling.*` keys llama.cpp
        # uses. Derived here rather than in the rotation loop so the formulas can
        # be checked on their own.
        "rope_scale": rope_scaling(md, arch, int(key("context_length"))),
        "qkv_bias": bool(key("attention.qkv_bias", False)),
        # The prompt is tokenized with the checkpoint's own BOS id, exactly as
        # the engine does; hard-coding 1 would hide a real disagreement.
        "bos": int(md.get("tokenizer.ggml.bos_token_id", 1)),
        "eos": int(md.get("tokenizer.ggml.eos_token_id", 2)),
        # Presence of the per-head q/k scales is the evidence qwen3 uses them.
        "qk_norm": has("attn_q_norm.weight"),
        # One row per head ({head_dim, heads}) instead of a single shared vector
        # (command-r, chameleon), and an optional bias on the same layout.
        "qk_norm_matrix": (has("attn_q_norm.weight") and
                           len(tensors[blk("attn_q_norm.weight")][0]) == 2),
        # A one-dimensional scale as wide as the whole q projection is olmoe's
        # whole-vector norm; one head's worth is qwen3's per-head norm.
        "qk_norm_whole_vector": (has("attn_q_norm.weight") and
                                 len(tensors[blk("attn_q_norm.weight")][0]) == 1 and
                                 int(tensors[blk("attn_q_norm.weight")][0][0]) ==
                                 int(key("attention.head_count")) * hd),
        # The norm's *mode* is not in the file: metadata says a tensor of
        # head_dim entries exists and nothing else. Same three facts the engine
        # reads from the architecture, quoted from llama.cpp's model files.
        "qk_norm_layernorm": arch == "command-r",
        # olmo2 has post-norms and no pre-norms: both branch inputs are raw.
        "no_branch_norms": (not has("attn_norm.weight") and has("attn_post_norm.weight")
                            and has("ffn_post_norm.weight")),
        # granite scales each branch output by `residual_scale` before the add.
        "residual_scale_applied": arch in ("granite", "granitemoe") and
                                  float(key("residual_scale", 0.0)) != 0.0,
        "residual_scale": float(key("residual_scale", 0.0)),
        # granite may override the softmax scale outright.
        "attention_scale": (float(key("attention.scale", 0.0))
                            if arch in ("granite", "granitemoe") else 0.0),
        # granite divides the logits by `logit_scale` after the head.
        "logit_scale_inverse": ((arch in ("granite", "granitemoe")) and
                                float(key("logit_scale", 0.0)) != 0.0),
        "logit_scale_inverse_value": float(key("logit_scale", 0.0)),
        # talkie normalises q/k after the rope; so does hunyuan-moe
        # (src/models/hunyuan-moe.cpp rotates first, then calls build_norm).
        "qk_norm_after_rope": arch in ("talkie", "hunyuan-moe"),
        "qk_norm_eps": None,  # None -> the file's own epsilon
        # A block that normalises without scaling carries no norm tensor at all.
        "weightless_norm": (not has("attn_norm.weight") and
                            not (has("attn_post_norm.weight") and has("ffn_post_norm.weight"))),
        # `*.logit_scale` is applied only where the graph applies it: command-r
        # multiplies the logits, granite records the key and never uses it.
        "logit_scale": (key("logit_scale", 0.0) if arch == "command-r" else 0.0),
        # phi3 fuses q/k/v into one projection (and one optional bias).
        "fused_qkv": has("attn_qkv.weight"),
        # glm4 keeps gate and up in one tensor: gate first, then up. The width is
        # the evidence -- phi2 also has no gate, but its up is [hidden, I].
        "fused_gate_up": (has_attn
                          and not has("ffn_gate.weight")
                          and has("ffn_up.weight")
                          and not has("ffn_gate_exps.weight")
                          and tensors[blk("ffn_up.weight")][0][1] == 2 * int(key("feed_forward_length"))),
        # phi2's block: LayerNorm, biases on the norms/FFN/logits, a gate-free
        # GELU feed-forward fed by the attention norm, and one shared residual.
        # Every one of these is read from the tensors or the epsilon key.
        "layer_norm": f"{arch}.attention.layer_norm_epsilon" in md,
        "norm_bias": has("attn_norm.bias"),
        # A block with neither an FFN norm nor a post-attention norm reuses the
        # attention's normalised input (phi2). gpt-oss has the post-attention
        # norm, so it is an ordinary pre-norm block.
        # A weightless block (olmo, talkie) has no ffn_norm *tensor* but still has
        # a feed-forward norm, so it is not phi2's shared residual.
        "parallel_residual": (has("attn_norm.weight")
                              and not has("ffn_norm.weight")
                              and not has("attn_post_norm.weight")
                              # a recurrent layer has no FFN to normalise
                              and not has_ssm),
        "ffn_bias": has("ffn_up.bias"),
        "ffn_gate_bias": has("ffn_gate.bias"),
        # mistral3's adaptive RMSNorm and its delay embedding.
        "ffn_norm_ada": has("ffn_norm_ada_down.weight"),
        "realtime_delay": int(key("realtime_delay_tokens", 0)),
        # llama.cpp's per-position attention temperature.
        "attn_temp_scale": float(key("attention.temperature_scale", 0.0)),
        "attn_temp_floor": float(key("rope.scaling.original_context_length", 0.0)),
        "attn_output_bias": has("attn_output.bias"),
        "logit_bias": "output.bias" in tensors,
        # gemma family: scaled embeddings, zero-centred norm scales, GeGLU.
        # A multiplier, not a flag: gemma uses sqrt(hidden), minicpm 12.0.
        "embedding_scale": (math.sqrt(hidden) if arch in ("gemma", "gemma2", "gemma3")
                            else float(key("embedding_scale", 12.0))
                            if arch == "minicpm" else 1.0),
        "norm_add_one": arch in ("gemma", "gemma2", "gemma3"),
        "activation": ("swiglu-oai" if arch in ("gpt-oss", "minimax-m3")
                       else "gelu" if arch in ("gemma", "gemma2", "gemma3") else "silu"),
        "activation_limit": 7.0 if arch in ("gpt-oss", "minimax-m3") else 0.0,
        "activation_alpha": 1.702 if arch in ("gpt-oss", "minimax-m3") else 0.0,
        # Per-layer clamp for `silu-clamp`, straight from the metadata array.
        # A file that declares it selects the clamped variant, exactly as the
        # engine does; the mode is never inferred from the architecture name.
        "swiglu_clamp": [float(v) for v in key("swiglu_clamp_exp", [])],
        # Which of the two upstream clamp forms; mirrors llama-graph.cpp:2449.
        "swiglu_clamp_fused": arch in ("maple", "deepseek4", "deepseek41",
                                       "glm5-next", "hy-v4"),
        # Detected from the tensors, exactly as the engine's adapter does:
        # keying this off the architecture name silently skipped glm4's norms.
        # Both post-norms or neither: gpt-oss has the attention one alone,
        # which is not a post-norm block at all but a re-named FFN norm.
        "post_norms": ((has("attn_post_norm.weight") or has("post_attention_norm.weight"))
                        and (has("ffn_post_norm.weight") or has("post_ffw_norm.weight"))),
        # BitNet's SUBLN: an RMSNorm on the attention output before the output
        # projection, and one on the gated FFN output before the down
        # projection. Tensor-presence evidence, like every other structure here.
        "sub_norms": has("attn_sub_norm.weight"),
        # Optional one-element `*.scale` tensors that multiply a projection's
        # output; llama.cpp folds them into `build_lora_mm`, before any bias.
        "proj_scales": has("attn_q.scale"),
        "attn_softcap": float(key("attn_logit_softcapping", 0.0)),
        "final_softcap": float(key("final_logit_softcapping", 0.0)),
        "window": int(key("attention.sliding_window", 0)),
        # Per-layer window selection, from `attention.sliding_window_pattern`.
        # gpt-oss alternates full and windowed layers, so one model-wide window
        # would describe half the block wrongly. Empty means "every layer".
        "window_layers": _window_layers(md, arch, n_layer, int(key("attention.sliding_window", 0))),
        # flint states which layers rotate at all, per layer. exaone-moe and
        # maple have no such key: they rotate exactly where `is_swa(il)` holds,
        # so the window pattern *is* the rotary pattern.
        "rope_layers": (_rope_layers(md, arch, n_layer)
                        or ([(layer % 4) != 0 for layer in range(n_layer)]
                            if arch == "exaone-moe"
                               and md.get(f"{arch}.attention.sliding_window_pattern") is None
                               and int(key("attention.sliding_window", 0)) != 0
                            else [])
                        or (_window_layers(md, arch, n_layer,
                                           int(key("attention.sliding_window", 0)))
                            if arch in ("exaone-moe", "maple") else [])),
        # A windowed layer may rotate at its own base (`rope.freq_base_swa`).
        "theta_layers": _rope_theta_layers(md, arch, n_layer,
                                           int(key("attention.sliding_window", 0)),
                                           _window_layers(md, arch, n_layer,
                                                          int(key("attention.sliding_window", 0))),
                                           float(key("rope.freq_base", 10000.0))),
        # gpt-oss feeds the FFN from its post-attention norm and has no ffn_norm
        # tensor of its own.
        "ffn_norm_is_post": (has("attn_post_norm.weight") and not has("ffn_norm.weight")),
        # Two distinct MoE biases, in two distinct places.
        "router_bias": any(f"blk.{l}.ffn_gate_inp.bias" in tensors for l in range(n_layer)),
        "expert_proj_bias": any(f"blk.{l}.ffn_gate_exps.bias" in tensors
                                for l in range(n_layer)),
        # --- recurrent (state-space) layers -----------------------------------
        # A file with `ssm_in` has no attention; a pure mamba file has no FFN
        # either. Both facts are read from the tensors, as they are in the
        # adapter.
        #
        "ssm_kind": ssm_kind,
        "ssm_conv": int(key("ssm.conv_kernel", 0)),
        "ssm_inner": int(key("ssm.inner_size", 0)),
        "ssm_state": int(key("ssm.state_size", 0)),
        "ssm_dt_rank": int(key("ssm.time_step_rank", 0)),
        "ssm_groups": max(1, int(key("ssm.group_count", 1))),
        "ssm_vector_norms": ssm_vector_norms,
        "ssm_norms_present": ssm_norms_present,
        "recurrent": recurrent,
        "moe_layers": [moe_layer(l) for l in range(n_layer)],
        "ffn_layers": [ffn_layer(l) for l in range(n_layer)],
        "has_ffn": any(ffn_layer(l) for l in range(n_layer)),
        "tensors": tensors,
        # Routing semantics straight from the checkpoint. Renormalising over the
        # selected experts is only equivalent to a full softmax when every expert
        # is selected, so it cannot be assumed.
        "experts": int(key("expert_count", 0)),
        "top_k": int(key("expert_used_count", 0)),
        # Routed experts and the shared expert can each be narrower than the
        # dense FFN, so the three widths are separate.
        # The width may be one number or one per layer (`get_key_or_arr`:
        # flint, exaone-moe and maple all read `n_ff_exp_arr`).
        "I_exp": (0 if not has_attn else
                  int((lambda v: v[0] if isinstance(v, list) else v)(
                      key("expert_feed_forward_length", key("feed_forward_length"))))),
        "I_exp_layers": ([int(v) for v in md[f"{arch}.expert_feed_forward_length"]]
                         if isinstance(md.get(f"{arch}.expert_feed_forward_length"), list)
                         else []),
        "I_shexp": 0 if not has_attn else int(key("expert_shared_feed_forward_length",
                                             key("feed_forward_length"))),
        # Evidence, not architecture name: the tensors are what the forward pass
        # reads, so their presence is what decides whether the branch runs. Every
        # one of these is scanned over *all* layers: a deepseek checkpoint's
        # leading dense blocks carry none of the MoE tensors, so a blk.0 probe
        # silently reports "no shared expert" for a model that has one.
        "shared_expert": any(f"blk.{l}.ffn_gate_shexp.weight" in tensors
                             for l in range(n_layer)),
        "shared_expert_gate": any(f"blk.{l}.ffn_gate_inp_shexp.weight" in tensors
                                  for l in range(n_layer)),
        # Multi-head latent attention: the file carries the projection into the
        # latent and the per-head expansions, and no keys or values.
        # Attention sinks: a per-head logit that the softmax denominator includes
        # and the value sum never reads. Detected from the tensor, like every
        # other structural fact.
        "attn_sinks": has("attn_sinks.weight"),
        # SmolLM3 keeps every fourth layer unrotated.
        "no_rope_step": 4 if arch == "smollm3" else 0,
        # Per-pair rotary frequency divisors; layer 0 is authoritative.
        "rope_freqs": (has("rope_freqs.weight") and tensors[blk("rope_freqs.weight")][1]),
        "mla": has("attn_kv_a_mqa.weight"),
        "q_lora_rank": int(key("attention.q_lora_rank", 0)),
        "kv_lora_rank": int(key("attention.kv_lora_rank", 0)),
        "head_v": int(key("attention.value_length_mla", hd)),
        "dense_lead": int(key("leading_dense_block_count", 0)),
        # Router semantics: the gating function, the selection bias and
        # group-limited routing all change which experts run.
        "gating": ("softmax" if arch == "maple" else
                   "softmax-weight" if arch == "gpt-oss" else
                   {1: "softmax", 2: "sigmoid", 3: "softmax-weight",
                    4: "sqrt-softplus"}.get(
                        int(key("expert_gating_func", 2 if arch == "glm4-moe" else 1)),
                        "softmax")),
        "selection_bias": any(f"blk.{l}.ffn_exp_probs_b.bias" in tensors
                              for l in range(n_layer)),
        "expert_groups": int(key("expert_group_count", 0)),
        "groups_used": int(key("expert_group_used_count", 0)),
        # smallthinker passes both as constants to build_moe_ffn(..., LLM_FFN_RELU,
        # /*norm_w=*/true, ...), so neither is in the file.
        # dbrx, ernie4-5-moe and minimax-m2 pass `norm_w = true` as a constant to
        # build_moe_ffn, and deepseek passes `false`, so for those four the file
        # carries no key and the architecture is the evidence -- the same rule the
        # engine applies.
        "expert_norm": (False if arch == "deepseek" else
                        bool(key("expert_weights_norm", False)) or
                        arch in ("smallthinker", "dbrx", "ernie4-5-moe", "maple",
                                 "minimax-m2")),
        # maple passes both the scale and the gating function as constants too,
        # so a file that carries either key must not change the arithmetic.
        "expert_scale": (1.0 if arch == "maple"
                         else float(key("expert_weights_scale", 1.0))),
    }
    # A block with up/down but no gate has no gated activation. GGUF cannot say
    # which activation, so phi2's GELU is applied and anything else is refused --
    # the same rule the engine's adapter enforces.
    cfg["ffn_no_gate"] = (cfg["experts"] == 0 and not has("ffn_gate.weight")
                          and has("ffn_up.weight") and not cfg["fused_gate_up"])
    # jamba applies no rotation at all: upstream's block never calls the rope, so
    # a file of that architecture is not rotated at head width by default.
    cfg["rope_disabled"] = arch in ("jamba", "jais")
    # baichuan picks its block by layer count: 32 layers is the 7B file and
    # rotates; anything else is the 13B file, which calls no rope at all and, at
    # 40 layers, biases attention with ALiBi (both hardcoded upstream).
    if arch == "baichuan":
        if cfg["L"] != 32:
            cfg["rope_disabled"] = True

    # ALiBi: `attention.max_alibi_bias`. bloom hardcodes 8.0 upstream, jais reads
    # the key, and everything with no bias keeps 0.0.
    cfg["alibi_max_bias"] = (8.0 if (arch == "bloom" or (arch == "baichuan" and cfg["L"] == 40))
                             else float(key("attention.max_alibi_bias", 0.0)))
    # bloom normalises the embedding once before block 0. The tensor has no
    # `blk.N.` prefix, so `has()` (which prefixes) cannot see it.
    cfg["token_embd_norm"] = "token_embd_norm.weight" in tensors
    # An absolute learned position table (gpt2, starcoder): the position is a
    # vector added to the token embedding, and neither rotates anything.
    cfg["position_embd"] = "position_embd.weight" in tensors
    # falcon-40b's second pre-attention norm.
    cfg["attn_norm_2"] = "blk.%d.attn_norm_2.weight" % probe in tensors
    if cfg["position_embd"]:
        cfg["rope_disabled"] = True
    if cfg["ffn_no_gate"]:
        # A gate-free FFN cannot say which activation it means, so the
        # architecture has to. Read from llama.cpp's model files: both phi2 and
        # gpt-neox use `LLM_FFN_GELU` with `LLM_FFN_SEQ`.
        # Read from llama.cpp's per-architecture files: the gate-free MLPs are
        # GeGLU in the two code models and in PocketTTS, and ReLU-squared in the
        # four named below.
        seq_activation = {"phi2": "gelu", "gptneox": "gelu",
                          "codeshell": "gelu", "starcoder2": "gelu",
                          "pockettts": "gelu", "bloom": "gelu",
                          "gpt2": "gelu", "starcoder": "gelu", "falcon": "gelu", "mpt": "gelu",
                          "arcee": "relu_sq", "nemotron": "relu_sq",
                          "jais2": "relu_sq", "plm": "relu_sq"}
        if arch not in seq_activation:
            raise ReferenceError("gate-free feed-forward is only defined for phi2 and gptneox")
        cfg["activation"] = seq_activation[arch]
    if arch == "smallthinker":
        # build_moe_ffn(..., LLM_FFN_RELU, /*norm_w=*/true, ...): a plain ReLU on
        # the gate, and the expert weights renormalised after top-k.
        cfg["activation"] = "relu"
        cfg["expert_norm"] = True
    # llama.cpp reads the section split as a required key for these families.
    # On a single text sequence every axis carries the same position, so the
    # split does not change a number here; what it does change is that a file
    # without it is a different model and must not load.
    if arch in ("qwen3vl", "qwen3vlmoe", "qwen35", "qwen35moe") and \
            f"{arch}.rope.dimension_sections" not in md:
        raise ReferenceError(f"{arch} requires rope.dimension_sections")
    if has_attn:
        if cfg["heads"] * cfg["hd"] != cfg["H"]:
            raise ReferenceError("heads * head_dim != hidden")
        if cfg["kv_heads"] > cfg["heads"] or cfg["heads"] % cfg["kv_heads"]:
            raise ReferenceError("inconsistent grouped-query head counts")
        if cfg["rope_dim"] % 2 or cfg["rope_dim"] > cfg["hd"]:
            raise ReferenceError("invalid rotary dimension")
    if has_ssm and (cfg["ssm_conv"] < 2 or not (cfg["ssm_inner"] and cfg["ssm_state"] and
                                                cfg["ssm_dt_rank"])):
        raise ReferenceError("an SSM layer needs a convolution width of at least 2 "
                             "and non-zero sizes")
    if cfg["activation"] == "silu" and any(v > 1e-6 for v in cfg["swiglu_clamp"]):
        cfg["activation"] = "silu-clamp"
        cfg["activation_limit"] = max(cfg["swiglu_clamp"])
    return cfg


def matvec(w: list[float], x: list[float], out_dim: int, in_dim: int) -> list[float]:
    """y[o] = dot(w[o*in_dim:(o+1)*in_dim], x), accumulated in double."""
    out = [0.0] * out_dim
    for o in range(out_dim):
        base = o * in_dim
        acc = 0.0
        for i in range(in_dim):
            acc += w[base + i] * x[i]
        out[o] = acc
    return out


def rmsnorm(x: list[float], w: list[float] | None, eps: float,
            add_one: bool = False) -> list[float]:
    """RMS norm. A `None` weight is unit gain, matching `build_norm`, which
    multiplies only when the weight tensor is present."""
    mean_sq = sum(v * v for v in x) / len(x)
    scale = 1.0 / math.sqrt(mean_sq + eps)
    if w is None:
        return [v * scale for v in x]
    bias = 1.0 if add_one else 0.0
    return [x[i] * scale * (bias + w[i]) for i in range(len(x))]


def layernorm(x: list[float], w: list[float], b: list[float], eps: float) -> list[float]:
    """Mean-centred normalisation with a bias, which phi2 uses instead of RMSNorm."""
    n = len(x)
    mean = sum(x) / n
    var = sum((v - mean) * (v - mean) for v in x) / n
    inv = 1.0 / math.sqrt(var + eps)
    return [(x[i] - mean) * inv * w[i] + b[i] for i in range(n)]


def gelu(x: float) -> float:
    """ggml's tanh approximation of GELU."""
    return 0.5 * x * (1.0 + math.tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)))


def head_rmsnorm(vec: list[float], heads: int, scale: list[float], eps: float) -> list[float]:
    """Normalises each head of `vec` on its own, using one shared scale vector."""
    hd = len(scale)
    out = list(vec)
    for h in range(heads):
        base = h * hd
        mean_sq = sum(vec[base + j] ** 2 for j in range(hd)) / hd
        s = 1.0 / math.sqrt(mean_sq + eps)
        for j in range(hd):
            out[base + j] = vec[base + j] * s * scale[j]
    return out


def rope_scaling(md: dict[str, Any], arch: str, n_ctx_train: int) -> dict[str, Any]:
    """llama.cpp's derived rope-scaling parameters, from the raw keys.

    `freq_scale` is 1/factor (llama.cpp's direction), `ext_factor` is 1 only for
    YaRN, and `mscale` is the net magnitude multiplier. llama.cpp's kernel folds
    a `1 + 0.1*log(1/freq_scale)` factor into YaRN's magnitude and its context
    setup divides it straight back out; the net value is what is returned here.
    """
    def key(name: str, default: Any) -> Any:
        for candidate in (f"{arch}.{name}", f"freemoe.{name}", name):
            if candidate in md:
                return md[candidate]
        return default

    kind = key("rope.scaling.type", "none")
    factor = float(key("rope.scaling.factor", key("rope.scale_linear", 0.0)) or 1.0)
    n_ctx_orig = float(key("rope.scaling.original_context_length", n_ctx_train))
    attn_factor = float(key("rope.scaling.attn_factor", 1.0))
    log_mul = float(key("rope.scaling.yarn_log_multiplier", 0.0))
    beta_fast = float(key("rope.scaling.yarn_beta_fast", 32.0))
    beta_slow = float(key("rope.scaling.yarn_beta_slow", 1.0))
    if kind == "none":
        return {"kind": kind, "freq_scale": 1.0, "ext_factor": 0.0,
                "mscale": attn_factor, "n_ctx_orig": n_ctx_orig, "theta": None}
    if kind not in ("linear", "yarn"):
        raise ReferenceError(f"rope.scaling.type {kind!r} is not implemented")
    if n_ctx_orig <= 0:
        raise ReferenceError("rope.scaling without an original context length")

    def mscale_of(scale: float, m: float) -> float:
        return 1.0 if scale <= 1.0 else 0.1 * m * math.log(scale) + 1.0

    if kind == "yarn":
        if log_mul != 0.0:
            mscale_dim = log_mul if (arch == "deepseek2" and log_mul != 1.0) else 1.0
            mscale = mscale_of(factor, mscale_dim) / mscale_of(factor, log_mul)
        else:
            mscale = mscale_of(factor, 1.0)
    else:
        mscale = 1.0
    return {"kind": kind, "freq_scale": 1.0 / factor, "ext_factor": 1.0 if kind == "yarn" else 0.0,
            "mscale": mscale * attn_factor, "n_ctx_orig": n_ctx_orig,
            "beta_fast": beta_fast, "beta_slow": beta_slow}


def yarn_corr_dims(n_dims: int, n_ctx_orig: float, freq_base: float,
                   beta_fast: float, beta_slow: float) -> tuple[float, float]:
    """ggml's `ggml_rope_yarn_corr_dims`, in pair indices."""
    def corr(n_rot: float) -> float:
        return n_dims * math.log(n_ctx_orig / (n_rot * 2.0 * math.pi)) / (2.0 * math.log(freq_base))

    start = math.floor(corr(beta_fast))
    end = math.ceil(corr(beta_slow))
    return max(0.0, start), min(float(n_dims - 1), end)


def rope_angles(scale: dict[str, Any], n_dims: int, pair: int, pos: int,
                theta: float, freq_factors: list[float] | None = None) -> tuple[float, float]:
    """cos/sin for one rotary pair, with linear/YaRN scaling applied."""
    extrap = theta ** (-2.0 * pair / n_dims)
    if freq_factors:
        extrap /= freq_factors[pair]
    angle_freq = scale["freq_scale"] * extrap
    if scale["ext_factor"] != 0.0:
        lo, hi = yarn_corr_dims(n_dims, scale["n_ctx_orig"], theta,
                                scale["beta_fast"], scale["beta_slow"])
        ramp = 1.0 - min(1.0, max(0.0, (pair - lo) / max(0.001, hi - lo)))
        angle_freq = angle_freq * (1.0 - ramp * scale["ext_factor"]) + extrap * ramp * scale["ext_factor"]
    angle = pos * angle_freq
    return math.cos(angle) * scale["mscale"], math.sin(angle) * scale["mscale"]


def rope_half(vec: list[float], head_dim: int, heads: int, pos: int,
              rope_dim: int, theta: float,
              scale: dict[str, Any] | None = None,
              cfg_freqs: list[float] | None = None) -> None:
    """Rotates each head's vector in place over `rope_dim` dimensions.

    This is ggml's default (`GGML_ROPE_TYPE_NORM`) layout, the one llama.cpp
    uses for llama-family checkpoints: the rotating pairs are *adjacent*
    elements (2i, 2i+1). The `NEOX` layout pairs (i, i + rope_dim/2) instead and
    belongs to different architectures; using it here silently rotates the wrong
    coordinates, which looks like a plausible model that talks nonsense.

    Verified against a real checkpoint: on tinyllama-15M the adjacent-pair
    layout produces fluent text and the half-pair layout produces word salad.
    """
    if scale is None:
        scale = {"kind": "none", "freq_scale": 1.0, "ext_factor": 0.0, "mscale": 1.0}
    for head in range(heads):
        base = head * head_dim
        for j in range(rope_dim // 2):
            c, s = rope_angles(scale, rope_dim, j, pos, theta, cfg_freqs)
            a = vec[base + 2 * j]
            b = vec[base + 2 * j + 1]
            vec[base + 2 * j] = a * c - b * s
            vec[base + 2 * j + 1] = a * s + b * c


def rope_mla(vec: list[float], heads: int, stride: int, offset: int, width: int,
             pos: int, theta: float, scale: dict[str, Any] | None = None,
             cfg_freqs: list[float] | None = None) -> list[float]:
    """Rotary position applied to one slice per head.

    MLA rotates only the rope part of each query head, and the shared k_pe
    vector on its own, so the slice geometry has to be explicit.
    """
    if scale is None:
        scale = {"kind": "none", "freq_scale": 1.0, "ext_factor": 0.0, "mscale": 1.0}
    out = list(vec)
    for head in range(heads):
        base = head * stride + offset
        for i in range(width // 2):
            cs, sn = rope_angles(scale, width, i, pos, theta, cfg_freqs)
            a0 = out[base + 2 * i]
            a1 = out[base + 2 * i + 1]
            out[base + 2 * i] = a0 * cs - a1 * sn
            out[base + 2 * i + 1] = a0 * sn + a1 * cs
    return out


def expert_output(cfg: dict[str, Any], layer: int, expert: int,
                  x: list[float], prefix: str) -> list[float]:
    """One expert's SwiGLU/GeGLU output.

    The stacked tensors hold every expert back to back, so expert `e` occupies a
    contiguous slice. The slice geometry comes from the declared shapes, never
    from an assumption about float sizes.
    """
    H, I = cfg["H"], (cfg["I_exp_layers"][layer] if cfg["I_exp_layers"] else cfg["I_exp"])
    tensors = cfg["tensors"]
    if prefix + "ffn_gate.0.weight" in tensors:
        # Per-expert layout: one two-dimensional tensor per (expert, projection),
        # so the expert index selects the tensor rather than a slice of it. The
        # numbers are the same ones a stacked file of the same model holds.
        gate_vals = matvec(tensors[prefix + f"ffn_gate.{expert}.weight"][1], x, I, H)
        up_vals = matvec(tensors[prefix + f"ffn_up.{expert}.weight"][1], x, I, H)
    elif prefix + "ffn_gate_up_exps.weight" in tensors:
        # Fused layout: one tensor per layer of shape [H, 2*I, experts], gate
        # block first. Each expert owns a contiguous 2*I*H slice, so the halves
        # are a slice of that slice rather than of the whole tensor.
        dims, data = tensors[prefix + "ffn_gate_up_exps.weight"]
        per_expert = 2 * I * H
        base = expert * per_expert
        gate_vals = matvec(data[base:base + I * H], x, I, H)
        up_vals = matvec(data[base + I * H:base + 2 * I * H], x, I, H)
    else:
        out = []
        for name, out_dim, in_dim in (
                (prefix + "ffn_gate_exps.weight", I, H),
                (prefix + "ffn_up_exps.weight", I, H)):
            dims, data = tensors[name]
            per_expert = out_dim * in_dim
            base = expert * per_expert
            out.append(matvec(data[base:base + per_expert], x, out_dim, in_dim))
        gate_vals, up_vals = out
    if cfg["expert_proj_bias"]:
        # Per-expert bias columns, added after each matmul and before the
        # activation — llama.cpp's `ggml_add_id(..., selected_experts)`.
        for name, values in ((prefix + "ffn_gate_exps.bias", gate_vals),
                             (prefix + "ffn_up_exps.bias", up_vals)):
            _, bias = tensors[name]
            for i in range(I):
                values[i] += bias[expert * I + i]
    act = [gated(cfg, gate_vals[i], up_vals[i], layer) for i in range(I)]
    if prefix + "ffn_down.0.weight" in tensors:
        return matvec(tensors[prefix + f"ffn_down.{expert}.weight"][1], act, H, I)
    dims, data = tensors[prefix + "ffn_down_exps.weight"]
    per_expert = H * I
    base = expert * per_expert
    out = matvec(data[base:base + per_expert], act, H, I)
    if cfg["expert_proj_bias"]:
        _, bias = tensors[prefix + "ffn_down_exps.bias"]
        for i in range(H):
            out[i] += bias[expert * H + i]
    return out


def pointwise(cfg: dict[str, Any], x: float) -> float:
    """The un-gated activation, mirroring include/fm/activation.hpp.

    One implementation for the gate-free feed-forward and for the gate branch of
    the gated one, so the reference cannot disagree with the engine about which
    activation a family uses.
    """
    kind = cfg["activation"]
    if kind == "gelu":
        return gelu(x)
    if kind == "gelu_erf":
        return 0.5 * x * (1.0 + math.erf(x / math.sqrt(2.0)))
    if kind == "relu":
        return max(0.0, x)
    if kind == "relu_sq":
        r = max(0.0, x)
        return r * r
    if kind == "softplus":
        return math.log1p(math.exp(x))
    return x / (1.0 + math.exp(-x))


def silu(v: float) -> float:
    return v / (1.0 + math.exp(-v))


def softplus(v: float) -> float:
    """`ggml_compute_softplus_f32`: the identity above 20, where exp overflows."""
    return v if v > 20.0 else math.log1p(math.exp(v))


def gated(cfg: dict[str, Any], gate: float, up: float, layer: int) -> float:
    """The gated activation, including the two clamped variants.

    `swiglu-oai` is gpt-oss's `silu(alpha*gate) * (up + 1)` with both branches
    clamped to +/-limit; `silu-clamp` is plain silu with a per-layer bound. Both
    come from ggml's `swiglu_oai` / `swiglu_clamp` kernels.
    """
    kind = cfg["activation"]
    limit = cfg.get("activation_limit") or 0.0
    if kind == "swiglu-oai":
        x = min(gate, limit)
        y = max(-limit, min(limit, up))
        return (x / (1.0 + math.exp(-cfg["activation_alpha"] * x))) * (y + 1.0)
    if kind == "silu-clamp":
        per_layer = cfg.get("swiglu_clamp") or []
        bound = per_layer[layer] if layer < len(per_layer) else limit
        # 1e-6, not 0: that is upstream's threshold for "clamped at all".
        if bound <= 1e-6:
            return gate / (1.0 + math.exp(-gate)) * up
        u = max(-bound, min(bound, up))
        if cfg.get("swiglu_clamp_fused"):
            # ggml_swiglu_clamp: the gate is clamped before silu.
            g = min(gate, bound)
            return (g / (1.0 + math.exp(-g))) * u
        # The graph fallback: silu first, then clamp its output.
        g = min(gate / (1.0 + math.exp(-gate)), bound)
        return g * u
    return pointwise(cfg, gate) * up


def softmax(values: list[float]) -> list[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


# Debug hook: set to a dict to record per-layer intermediate vectors.
TRACE: dict[str, Any] | None = None


class Session:
    """One forward pass per token, with a KV cache laid out per layer."""

    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.pos = 0
        kv_dim = cfg["kv_heads"] * cfg["hd"]
        self.cache: list[list[tuple[list[float], list[float]]]] = [
            [] for _ in range(cfg["L"])
        ]
        self.kv_dim = kv_dim
        # Normalise upstream's current post-norm names once, at load, so every
        # later lookup (and every tensor-presence test) sees one spelling.
        self.tensors = {self.canonical(k): v for k, v in cfg["tensors"].items()}
        # One rolling recurrent state per layer: the convolution history
        # (d_conv - 1 columns) followed by one state vector per channel.
        self.ssm: list[dict[str, Any]] = [{} for _ in range(cfg["L"])]

    def canonical(self, name: str) -> str:
        """Map upstream's current post-norm names onto this module's spelling.

        llama.cpp renamed `attn_post_norm` -> `post_attention_norm` and
        `ffn_post_norm` -> `post_ffw_norm`. The tensors did not move, so a file
        of either vintage describes the same block.
        """
        return canonical_name(name)

    def tensor(self, name: str) -> list[float]:
        name = self.canonical(name)
        if name not in self.tensors:
            raise ReferenceError(f"missing tensor {name}")
        return self.tensors[name][1]

    def ssm_block(self, layer: int, p: str, h: list[float]) -> list[float]:
        """One llama.cpp mamba/mamba2 block, for a single token.

        Mirrors the engine: the projection splits into the gate half and the
        convolved half, the convolution slides a window of `d_conv` columns (the
        history followed by this token), and the scan updates one state vector
        per channel. mamba-1 decays each state element by its own A row; mamba2
        decays the whole head by a scalar and normalises B, C and dt per group.
        """
        cfg = self.cfg
        H = cfg["H"]
        inner, d_state = cfg["ssm_inner"], cfg["ssm_state"]
        d_conv, groups = cfg["ssm_conv"], cfg["ssm_groups"]
        kind = cfg["ssm_kind"]
        mamba2 = kind == "mamba2"
        conv_channels = inner + 2 * groups * d_state if mamba2 else inner
        in_dim = inner + conv_channels + cfg["ssm_dt_rank"] if mamba2 else 2 * inner
        st = self.ssm[layer]
        if not st:
            st["conv"] = [[0.0] * conv_channels for _ in range(d_conv - 1)]
            st["state"] = [[0.0] * d_state for _ in range(inner)]

        projected = matvec(self.tensor(p + "ssm_in.weight"), h, in_dim, H)
        z = projected[:inner]
        if mamba2:
            conv_in = projected[inner:inner + conv_channels]
            # mamba2 projects dt directly from ssm_in and stores only its bias,
            # so there is no ssm_dt weight here (llama.cpp models/mamba2.cpp).
            dt = projected[inner + conv_channels:]
        else:
            conv_in = projected[inner:]
            dt = None

        # ggml_ssm_conv: each output channel dots the sliding window with its
        # own d_conv kernel column.
        kernel = self.tensor(p + "ssm_conv1d.weight")
        bias = self.tensor(p + "ssm_conv1d.bias")
        history = st["conv"]
        convolved = [0.0] * conv_channels
        for i in range(conv_channels):
            acc = 0.0
            for k in range(d_conv):
                value = history[k][i] if k < d_conv - 1 else conv_in[i]
                acc += value * kernel[k + d_conv * i]
            convolved[i] = silu(acc + bias[i])
        if d_conv > 1:
            st["conv"] = [[conv_in[i] for i in range(conv_channels)]] if d_conv == 2 else \
                history[1:] + [[conv_in[i] for i in range(conv_channels)]]

        state = st["state"]
        A = self.tensor(p + "ssm_a")
        D = self.tensor(p + "ssm_d")
        y = [0.0] * inner
        if mamba2:
            dt_bias = self.tensor(p + "ssm_dt.bias")
            # No dt norm: upstream's build_mamba2_layer has none. The grouped
            # `ssm_norm` after the scan is the only norm in that block.
            dt = [dt[i] + dt_bias[i] for i in range(cfg["ssm_dt_rank"])]
            head_dim = inner // cfg["ssm_dt_rank"]
            per_group = cfg["ssm_dt_rank"] // groups
            B = convolved[inner:inner + groups * d_state]
            C = convolved[inner + groups * d_state:inner + 2 * groups * d_state]
            for head in range(cfg["ssm_dt_rank"]):
                dt_sp = softplus(dt[head])
                dA = math.exp(dt_sp * A[head])
                g = head // per_group
                for i in range(head_dim):
                    ii = i + head * head_dim
                    x_dt = convolved[ii] * dt_sp
                    acc = 0.0
                    for k in range(d_state):
                        value = state[ii][k] * dA + B[k + g * d_state] * x_dt
                        state[ii][k] = value
                        acc += value * C[k + g * d_state]
                    y[ii] = acc + convolved[ii] * D[head]
            y = [silu(z[i]) * y[i] for i in range(inner)]
            # The grouped RMSNorm after the scan.
            norm_w = self.tensor(p + "ssm_norm.weight")
            width = inner // groups
            out = list(y)
            for g in range(groups):
                base = g * width
                scale = 1.0 / math.sqrt(sum(v * v for v in out[base:base + width]) / width
                                        + cfg["eps"])
                for i in range(width):
                    out[base + i] = out[base + i] * scale * norm_w[i + width * g]
            y = out
        else:
            dt_rank = cfg["ssm_dt_rank"]
            dbc = matvec(self.tensor(p + "ssm_x.weight"), convolved, dt_rank + 2 * d_state, inner)
            if cfg["ssm_vector_norms"]:
                # falcon-mamba and jamba: RMS norm on dt, B and C before the scan.
                # A weight the file does not carry is unit gain (build_norm with
                # a null weight), not a skipped norm.
                gain = cfg["ssm_norms_present"]

                def norm_weight(idx: int, name: str):
                    return self.tensor(p + name) if gain[idx] else None

                dbc[:dt_rank] = rmsnorm(dbc[:dt_rank], norm_weight(0, "ssm_dt_norm.weight"),
                                        cfg["eps"])
                dbc[dt_rank:dt_rank + d_state] = rmsnorm(
                    dbc[dt_rank:dt_rank + d_state], norm_weight(1, "ssm_b_norm.weight"), cfg["eps"])
                dbc[dt_rank + d_state:] = rmsnorm(
                    dbc[dt_rank + d_state:], norm_weight(2, "ssm_c_norm.weight"), cfg["eps"])
            B = dbc[dt_rank:dt_rank + d_state]
            C = dbc[dt_rank + d_state:dt_rank + 2 * d_state]
            dt = matvec(self.tensor(p + "ssm_dt.weight"), dbc[:dt_rank], inner, dt_rank)
            dt_bias = self.tensor(p + "ssm_dt.bias")
            for channel in range(inner):
                dt_sp = softplus(dt[channel] + dt_bias[channel])
                x_dt = convolved[channel] * dt_sp
                acc = 0.0
                for k in range(d_state):
                    dA = math.exp(dt_sp * A[channel * d_state + k])
                    value = state[channel][k] * dA + B[k] * x_dt
                    state[channel][k] = value
                    acc += value * C[k]
                y[channel] = acc + convolved[channel] * D[channel]
            y = [silu(z[i]) * y[i] for i in range(inner)]
        return matvec(self.tensor(p + "ssm_out.weight"), y, H, inner)

    def step(self, token: int) -> list[float]:
        cfg = self.cfg
        H, hd, heads, kv_heads = cfg["H"], cfg["hd"], cfg["heads"], cfg["kv_heads"]
        kv_dim = kv_heads * hd
        group = heads // kv_heads if kv_heads else 1
        embd = self.tensor("token_embd.weight")
        x = [embd[token * H + i] for i in range(H)]
        if cfg["embedding_scale"] != 1.0:
            scale = cfg["embedding_scale"]
            x = [v * scale for v in x]

        def norm_of(vec: list[float], prefix: str) -> list[float]:
            """The one place both norm types are chosen, mirroring the engine."""
            if cfg["weightless_norm"]:
                # upstream's build_norm(x, NULL, NULL, ...): normalise, do not scale
                if cfg["layer_norm"]:
                    return layernorm(vec, [1.0] * len(vec), [0.0] * len(vec), cfg["eps"])
                return rmsnorm(vec, [1.0] * len(vec), cfg["eps"], cfg["norm_add_one"])
            w = self.tensor(prefix + ".weight")
            if cfg["layer_norm"]:
                # Upstream passes the bias pointer straight through, so a file
                # without `*.bias` is a bias-free LayerNorm, not a broken one
                # (cohere2's decoder passes NULL there).
                bias = (self.tensor(prefix + ".bias")
                        if prefix + ".bias" in self.tensors else [0.0] * len(w))
                return layernorm(vec, w, bias, cfg["eps"])
            return rmsnorm(vec, w, cfg["eps"], cfg["norm_add_one"])

        if cfg["token_embd_norm"]:
            # bloom's pre-block embedding norm (models/bloom.cpp).
            x = norm_of(x, "token_embd_norm")
        if cfg["position_embd"]:
            table = self.tensor("position_embd.weight")
            if self.pos >= len(table) // H:
                raise ReferenceError("position outside the checkpoint's position table")
            base = self.pos * H
            x = [x[i] + table[base + i] for i in range(H)]

        for layer in range(cfg["L"]):
            p = f"blk.{layer}."
            # olmo2 has no pre-norms: the branch inputs are the residual stream.
            h = list(x) if cfg["no_branch_norms"] else norm_of(x, p + "attn_norm")
            # falcon-40b normalises the attention input a second time; the FFN
            # and the residual keep reading `attn_norm`.
            attn_h = (norm_of(x, p + "attn_norm_2") if (cfg["attn_norm_2"] and
                                                       p + "attn_norm_2.weight" in self.tensors)
                      else h)
            if TRACE is not None:
                TRACE[f"h.L{layer}"] = list(h)
            attn_normed = list(h) if cfg["parallel_residual"] else None
            if cfg["recurrent"][layer]:
                delta = self.ssm_block(layer, p, h)
                if TRACE is not None:
                    TRACE[f"ssm.L{layer}"] = (sum(h), sum(delta))
                x = [x[i] + delta[i] for i in range(H)]
                if not cfg["ffn_layers"][layer]:
                    continue
                h = norm_of(x, p + "ffn_norm")
            else:
                if cfg["mla"]:
                    # Multi-head latent attention, expanded per head. See the engine's
                    # comment: k_b / v_b are linear, so expanding here is the same
                    # arithmetic as llama.cpp's absorbed form.
                    latent = cfg["kv_lora_rank"]
                    nope = hd - cfg["rope_dim"]
                    head_v = cfg["head_v"]
                    if cfg["q_lora_rank"]:
                        qa = matvec(self.tensor(p + "attn_q_a.weight"), attn_h, cfg["q_lora_rank"], H)
                        qa = rmsnorm(qa, self.tensor(p + "attn_q_a_norm.weight"), cfg["eps"])
                        q = matvec(self.tensor(p + "attn_q_b.weight"), qa, heads * hd,
                                   cfg["q_lora_rank"])
                    else:
                        q = matvec(self.tensor(p + "attn_q.weight"), attn_h, heads * hd, H)
                    kvpe = matvec(self.tensor(p + "attn_kv_a_mqa.weight"), attn_h,
                                  latent + cfg["rope_dim"], H)
                    ckv = rmsnorm(kvpe[:latent], self.tensor(p + "attn_kv_a_norm.weight"), cfg["eps"])
                    kpe = kvpe[latent:]
                    # Rotary on the trailing rope slice of each query head, and on
                    # the shared k_pe.
                    q = rope_mla(q, heads, hd, hd - cfg["rope_dim"], cfg["rope_dim"], self.pos,
                                 cfg["theta"], cfg["rope_scale"], cfg["rope_freqs"])
                    kpe = rope_mla(kpe, 1, cfg["rope_dim"], 0, cfg["rope_dim"], self.pos,
                                   cfg["theta"], cfg["rope_scale"], cfg["rope_freqs"])
                    kb = self.tensor(p + "attn_k_b.weight")
                    vb = self.tensor(p + "attn_v_b.weight")
                    k_vec: list[float] = []
                    v_vec: list[float] = []
                    for head in range(heads):
                        k_head = list(kpe)
                        kb_head = head * nope * latent
                        for o in range(nope):
                            # ggml stores k_b as [nope, kv_lora_rank, heads] with the
                            # first axis fastest, so (o, i, head) is at
                            # o + nope*(i + latent*head).
                            k_head.insert(o, sum(kb[kb_head + i * nope + o] * ckv[i]
                                                 for i in range(latent)))
                        k_vec.extend(k_head)
                        for o in range(head_v):
                            base_o = (head * head_v + o) * latent
                            v_vec.append(sum(vb[base_o + i] * ckv[i] for i in range(latent)))
                    k = k_vec
                    v = v_vec
                elif cfg["fused_qkv"]:
                    fused = matvec(self.tensor(p + "attn_qkv.weight"), attn_h, H + 2 * kv_dim, H)
                    if cfg["qkv_bias"]:
                        bias = self.tensor(p + "attn_qkv.bias")
                        fused = [fused[i] + bias[i] for i in range(H + 2 * kv_dim)]
                    q = fused[:H]
                    k = fused[H:H + kv_dim]
                    v = fused[H + kv_dim:]
                else:
                    q = matvec(self.tensor(p + "attn_q.weight"), attn_h, H, H)
                    k = matvec(self.tensor(p + "attn_k.weight"), attn_h, kv_dim, H)
                    v = matvec(self.tensor(p + "attn_v.weight"), attn_h, kv_dim, H)
                    if cfg["proj_scales"]:
                        # Before the bias and before RoPE, exactly as build_lora_mm
                        # folds the scale into the projection.
                        sq = self.tensor(p + "attn_q.scale")[0]
                        sk = self.tensor(p + "attn_k.scale")[0]
                        sv = self.tensor(p + "attn_v.scale")[0]
                        q = [v * sq for v in q]
                        k = [v * sk for v in k]
                        v = [v * sv for v in v]
                    if cfg["qkv_bias"]:
                        bq = self.tensor(p + "attn_q.bias")
                        bk = self.tensor(p + "attn_k.bias")
                        bv = self.tensor(p + "attn_v.bias")
                        q = [q[i] + bq[i] for i in range(H)]
                        k = [k[i] + bk[i] for i in range(kv_dim)]
                        v = [v[i] + bv[i] for i in range(kv_dim)]
                def qk_whole_norm(vec: list[float], prefix: str) -> list[float]:
                    """olmoe: one RMS over the concatenated heads, one wide scale."""
                    w = self.tensor(prefix + ".weight")
                    eps = cfg["qk_norm_eps"] or cfg["eps"]
                    mean_sq = sum(v * v for v in vec) / len(vec)
                    s = 1.0 / math.sqrt(mean_sq + eps)
                    return [vec[i] * s * w[i] for i in range(len(vec))]

                def qk_head_norm(vec: list[float], n_heads: int, prefix: str) -> list[float]:
                    """Per-head normalisation: shared or per-head row, RMS or LN."""
                    weight = self.tensor(prefix + ".weight")
                    eps = cfg["qk_norm_eps"] or cfg["eps"]
                    out = list(vec)
                    for h in range(n_heads):
                        base = h * hd
                        row = (weight[h * hd:(h + 1) * hd] if cfg["qk_norm_matrix"] else weight)
                        if cfg["qk_norm_layernorm"]:
                            head = layernorm(vec[base:base + hd], row, [0.0] * hd, eps)
                        else:
                            head = rmsnorm(vec[base:base + hd], row, eps, 0.0)
                        out[base:base + hd] = head
                    return out

                if not cfg["mla"]:
                    if cfg["qk_norm"] and not cfg["qk_norm_after_rope"]:
                        if cfg["qk_norm_whole_vector"]:
                            q = qk_whole_norm(q, p + "attn_q_norm")
                            k = qk_whole_norm(k, p + "attn_k_norm")
                        else:
                            q = qk_head_norm(q, heads, p + "attn_q_norm")
                            k = qk_head_norm(k, kv_heads, p + "attn_k_norm")
                    step = cfg["no_rope_step"]
                    # A per-layer pattern (`attention.rope_pattern`) is the only
                    # evidence that names the layers, so it wins over the periodic
                    # rule; jamba rotates nothing at all.
                    pattern = cfg["rope_layers"]
                    has_rope = (pattern[layer] if pattern
                                else (step == 0 or (layer + 1) % step != 0))
                    theta = (cfg["theta_layers"][layer]
                             if cfg["theta_layers"] else cfg["theta"])
                    if not cfg["rope_disabled"] and has_rope:
                        rope_half(q, hd, heads, self.pos, cfg["rope_dim"], theta,
                                  cfg["rope_scale"], cfg["rope_freqs"])
                        rope_half(k, hd, kv_heads, self.pos, cfg["rope_dim"], theta,
                                  cfg["rope_scale"], cfg["rope_freqs"])
                    # talkie normalises q and k after the rotation.
                    if cfg["qk_norm"] and cfg["qk_norm_after_rope"]:
                        q = qk_head_norm(q, heads, p + "attn_q_norm")
                        k = qk_head_norm(k, kv_heads, p + "attn_k_norm")
                if cfg["attn_temp_scale"] != 0.0:
                    # llama.cpp's per-position attention temperature: a multiplier on
                    # every query component, applied after the rotation.
                    temp = (math.log(math.floor(self.pos / cfg["attn_temp_floor"]) + 1.0) *
                            cfg["attn_temp_scale"] + 1.0)
                    q = [v * temp for v in q]
                if TRACE is not None:
                    TRACE["pos"] = self.pos
                    TRACE[f"q.L{layer}"] = list(q)
                    TRACE[f"k.L{layer}"] = list(k)
                    TRACE[f"v.L{layer}"] = list(v)
                self.cache[layer].append((k, v))

                head_v = cfg["head_v"] if cfg["mla"] else hd
                # ALiBi slopes, as ggml derives them inside soft_max_ext.
                slopes = []
                if cfg["alibi_max_bias"] > 0.0:
                    log2_heads = 1 << int(math.floor(math.log2(heads)))
                    m0 = 2.0 ** (-cfg["alibi_max_bias"] / log2_heads)
                    m1 = 2.0 ** (-(cfg["alibi_max_bias"] / 2.0) / log2_heads)
                    slopes = [m0 ** (h + 1) if h < log2_heads
                              else m1 ** (2 * (h - log2_heads) + 1) for h in range(heads)]
                attention = [0.0] * (heads * head_v)
                scale = (cfg["attention_scale"] if cfg["attention_scale"] > 0.0
                         else 1.0 / math.sqrt(hd))
                for head in range(heads):
                    qb = head * hd
                    kvb = (head // group) * hd
                    vb = (head // group) * head_v
                    cache = self.cache[layer]
                    window = cfg["window"]
                    layers = cfg["window_layers"]
                    windowed = (layers[layer] if layers else True) if window else False
                    first = (0 if (not windowed or self.pos < window)
                             else self.pos + 1 - window)
                    scores = []
                    for t_index in range(first, len(cache)):
                        kt = cache[t_index][0]
                        dot = sum(q[qb + j] * kt[kvb + j] for j in range(hd)) * scale
                        if cfg["attn_softcap"] > 0.0:
                            dot = cfg["attn_softcap"] * math.tanh(dot / cfg["attn_softcap"])
                        if slopes:
                            # The mask value upstream writes is -(query - key), and
                            # soft_max_ext multiplies it by the head's slope.
                            dot += slopes[head] * -(self.pos - t_index)
                        scores.append(dot)
                    if cfg["attn_sinks"]:
                        # Append the sink logit, then ignore its probability.
                        scores = scores + [self.tensor(p + "attn_sinks.weight")[head]]
                    probs = softmax(scores)
                    acc = [0.0] * head_v
                    for offset, t_index in enumerate(range(first, len(cache))):
                        vt = cache[t_index][1]
                        p_i = probs[offset]
                        for j in range(head_v):
                            acc[j] += p_i * vt[vb + j]
                    ab = head * head_v
                    attention[ab:ab + head_v] = acc
                if TRACE is not None:
                    TRACE[f"attn.L{layer}"] = list(attention)
                if cfg["sub_norms"]:
                    # Inside the block: the head mix is normalised before the output
                    # projection, which is not what a post-norm does.
                    attention = rmsnorm(attention, self.tensor(p + "attn_sub_norm.weight"),
                                        cfg["eps"], cfg["norm_add_one"])
                attn_out = matvec(self.tensor(p + "attn_output.weight"), attention, H,
                                  heads * head_v)
                if cfg["proj_scales"]:
                    s = self.tensor(p + "attn_output.scale")[0]
                    attn_out = [v * s for v in attn_out]
                if cfg["attn_output_bias"]:
                    bias = self.tensor(p + "attn_output.bias")
                    attn_out = [attn_out[i] + bias[i] for i in range(H)]
                if cfg["post_norms"]:
                    attn_out = rmsnorm(attn_out, self.tensor(p + "attn_post_norm.weight"),
                                       cfg["eps"], cfg["norm_add_one"])
                if cfg["residual_scale_applied"]:
                    rs = cfg["residual_scale"]
                    attn_out = [v * rs for v in attn_out]
                x = [x[i] + attn_out[i] for i in range(H)]

                if TRACE is not None:
                    TRACE[f"attn_out.L{layer}"] = list(attn_out)
                    TRACE[f"xpost.L{layer}"] = list(x)
                h = (attn_normed if cfg["parallel_residual"]
                     else list(x) if cfg["no_branch_norms"]
                     else norm_of(x, p + ("attn_post_norm" if cfg["ffn_norm_is_post"]
                                          else "ffn_norm")))
                if cfg["ffn_norm_ada"]:
                    # mistral3's adaptive RMSNorm: h *= 1 + up(gelu_erf(down · t)),
                    # with t the sinusoidal embedding of the configured delay -- a
                    # session constant, not a per-token value.
                    half = H // 2
                    t_cond = [0.0] * H
                    for i in range(half):
                        freq = math.exp(-math.log(10000.0) * i / half) * cfg["realtime_delay"]
                        t_cond[i] = math.cos(freq)
                        t_cond[half + i] = math.sin(freq)
                    dims, data = self.tensors[p + "ffn_norm_ada_down.weight"]
                    bottleneck = dims[1]
                    ada = matvec(data, t_cond, bottleneck, H)
                    ada = [0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0))) for v in ada]
                    dims, data = self.tensors[p + "ffn_norm_ada_up.weight"]
                    scale = matvec(data, ada, H, bottleneck)
                    h = [h[j] * (1.0 + scale[j]) for j in range(H)]
            if TRACE is not None:
                TRACE[f"x.L{layer}"] = list(x)
            if cfg["ffn_no_gate"]:
                # phi2: up -> gelu -> down, with a bias on both projections.
                up = matvec(self.tensor(p + "ffn_up.weight"), h, cfg["I"], H)
                if cfg["ffn_bias"]:
                    bias = self.tensor(p + "ffn_up.bias")
                    up = [up[i] + bias[i] for i in range(cfg["I"])]
                act = [pointwise(cfg, v) for v in up]
                down = matvec(self.tensor(p + "ffn_down.weight"), act, H, cfg["I"])
                if cfg["ffn_bias"]:
                    bias = self.tensor(p + "ffn_down.bias")
                    down = [down[i] + bias[i] for i in range(H)]
            elif cfg["moe_layers"][layer]:
                # Router -> gating function -> optional selection bias -> optional
                # group mask -> top-k -> combine. Which function turns logits into
                # probabilities is a per-family fact, so the checkpoint says.
                logits = matvec(self.tensor(p + "ffn_gate_inp.weight"), h,
                                cfg["experts"], H)
                if cfg["router_bias"]:
                    bias = self.tensor(p + "ffn_gate_inp.bias")
                    logits = [logits[i] + bias[i] for i in range(cfg["experts"])]
                gating = cfg["gating"]
                if gating == "sigmoid":
                    probs = [1.0 / (1.0 + math.exp(-v)) for v in logits]
                elif gating == "sqrt-softplus":
                    probs = [math.sqrt(math.log1p(math.exp(v))) for v in logits]
                elif gating == "softmax-weight":
                    probs = list(logits)
                else:
                    probs = softmax(logits)
                selection = list(probs)
                if cfg["selection_bias"]:
                    bias = self.tensor(p + "ffn_exp_probs_b.bias")
                    selection = [selection[i] + bias[i] for i in range(cfg["experts"])]
                if cfg["expert_groups"] > 1:
                    per_group = cfg["experts"] // cfg["expert_groups"]
                    group_scores = []
                    for g in range(cfg["expert_groups"]):
                        vals = sorted(selection[g * per_group:(g + 1) * per_group], reverse=True)
                        group_scores.append((sum(vals[:2]), g))
                    group_scores.sort(key=lambda x: (-x[0], x[1]))
                    keep = {g for _, g in group_scores[:cfg["groups_used"]]}
                    selection = [selection[i] if (i // per_group) in keep else -math.inf
                                 for i in range(cfg["experts"])]
                order = sorted(range(cfg["experts"]), key=lambda e: (-selection[e], e))
                chosen = order[:cfg["top_k"]]
                weights = [probs[e] for e in chosen]
                if gating == "softmax-weight":
                    weights = softmax(weights)
                if cfg["expert_norm"]:
                    total = sum(weights)
                    if total > 0.0:
                        weights = [w / total for w in weights]
                weights = [w * cfg["expert_scale"] for w in weights]
                down = [0.0] * H
                for weight, expert in zip(weights, chosen):
                    down = [down[i] + weight * v for i, v in enumerate(
                        expert_output(cfg, layer, expert, h, p))]
            else:
                if cfg["fused_gate_up"]:
                    both = matvec(self.tensor(p + "ffn_up.weight"), h, 2 * cfg["I"], H)
                    gate = both[:cfg["I"]]
                    up = both[cfg["I"]:]
                else:
                    gate = matvec(self.tensor(p + "ffn_gate.weight"), h, cfg["I"], H)
                    up = matvec(self.tensor(p + "ffn_up.weight"), h, cfg["I"], H)
                if cfg["ffn_gate_bias"]:
                    bias = self.tensor(p + "ffn_gate.bias")
                    gate = [gate[i] + bias[i] for i in range(cfg["I"])]
                if cfg["ffn_bias"]:
                    bias = self.tensor(p + "ffn_up.bias")
                    up = [up[i] + bias[i] for i in range(cfg["I"])]
                if cfg["proj_scales"]:
                    sg = self.tensor(p + "ffn_gate.scale")[0]
                    su = self.tensor(p + "ffn_up.scale")[0]
                    gate = [v * sg for v in gate]
                    up = [v * su for v in up]
                if cfg["activation"] == "gelu":
                    act = [gelu(gate[i]) * up[i] for i in range(cfg["I"])]
                else:
                    act = [gated(cfg, gate[i], up[i], layer) for i in range(cfg["I"])]
                if cfg["sub_norms"]:
                    act = rmsnorm(act, self.tensor(p + "ffn_sub_norm.weight"),
                                  cfg["eps"], cfg["norm_add_one"])
                down = matvec(self.tensor(p + "ffn_down.weight"), act, H, cfg["I"])
                if cfg["ffn_bias"]:
                    bias = self.tensor(p + "ffn_down.bias")
                    down = [down[i] + bias[i] for i in range(H)]
                if cfg["proj_scales"]:
                    sd = self.tensor(p + "ffn_down.scale")[0]
                    down = [v * sd for v in down]
            # The shared expert rides with the routed experts, and which layers
            # have those is per layer (`ffn_gate_inp` is optional upstream), not
            # `layer >= dense_lead`: ernie4-5-moe interleaves them.
            if cfg["shared_expert"] and cfg["moe_layers"][layer]:
                # qwen2moe's shared expert: a dense FFN beside the routed ones,
                # scaled by sigmoid(dot(gate_inp, h)). The reference writes that
                # gate as silu(s)/s, which is the same number except at s == 0
                # (0/0); the limit 0.5 is used there.
                if cfg["shared_expert_gate"]:
                    s = matvec(self.tensor(p + "ffn_gate_inp_shexp.weight"), h, 1, H)[0]
                    gate_scalar = (s / (1.0 + math.exp(-s)) / s) if s != 0.0 else 0.5
                else:
                    gate_scalar = 1.0
                W = cfg["I_shexp"]
                s_gate = matvec(self.tensor(p + "ffn_gate_shexp.weight"), h, W, H)
                s_up = matvec(self.tensor(p + "ffn_up_shexp.weight"), h, W, H)
                if cfg["activation"] == "gelu":
                    s_act = [gelu(s_gate[i]) * s_up[i] for i in range(W)]
                else:
                    s_act = [gated(cfg, s_gate[i], s_up[i], layer) for i in range(W)]
                s_down = matvec(self.tensor(p + "ffn_down_shexp.weight"), s_act, H, W)
                down = [down[i] + gate_scalar * s_down[i] for i in range(H)]
            if cfg["post_norms"]:
                down = rmsnorm(down, self.tensor(p + "ffn_post_norm.weight"),
                               cfg["eps"], cfg["norm_add_one"])
            if cfg["residual_scale_applied"]:
                rs = cfg["residual_scale"]
                down = [v * rs for v in down]
            if TRACE is not None:
                TRACE[f"ffn.L{layer}"] = list(down)
            x = [x[i] + down[i] for i in range(H)]
            if TRACE is not None:
                TRACE[f"end.L{layer}"] = list(x)

        final = norm_of(x, "output_norm")
        out_name = "output.weight" if "output.weight" in self.tensors else "token_embd.weight"
        self.pos += 1
        logits = matvec(self.tensor(out_name), final, cfg["V"], H)
        if cfg["logit_bias"]:
            bias = self.tensor("output.bias")
            logits = [logits[i] + bias[i] for i in range(cfg["V"])]
        if cfg["logit_scale"]:
            logits = [v * cfg["logit_scale"] for v in logits]
        if cfg["logit_scale_inverse"]:
            inv = 1.0 / cfg["logit_scale_inverse_value"]
            logits = [v * inv for v in logits]
        if cfg["final_softcap"] > 0.0:
            cap = cfg["final_softcap"]
            logits = [cap * math.tanh(v / cap) for v in logits]
        return logits


def tokenize_bytes(prompt: str, bos: int = 1) -> list[int]:
    """The fixtures carry the byte tokenizer, so ids are BOS plus raw bytes."""
    return [bos] + [b for b in prompt.encode("utf-8")]


def prefill(cfg: dict[str, Any], prompt: str) -> dict[str, Any]:
    session = Session(cfg)
    ids = tokenize_bytes(prompt, cfg["bos"])
    logits: list[float] = []
    for token in ids:
        logits = session.step(token)
    return {"input_ids": ids, "prefill_logits": logits}


def greedy(cfg: dict[str, Any], prompt: str, n: int) -> dict[str, Any]:
    session = Session(cfg)
    ids = tokenize_bytes(prompt, cfg["bos"])
    logits: list[float] = []
    for token in ids:
        logits = session.step(token)
    generated = []
    for _ in range(n):
        token = max(range(len(logits)), key=lambda i: (logits[i], -i))
        generated.append(token)
        # The checkpoint's own end-of-sequence id. Hard-coding 2 here made the
        # reference stop on an ordinary token whenever a fixture numbered its
        # vocabulary differently, which is exactly what happened to deepseek2.
        if token == cfg["eos"]:
            break
        logits = session.step(token)
    return {"input_ids": ids, "generated_ids": generated, "prefill_logits": logits}


def write_golden(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")


if __name__ == "__main__":
    model = load_model(Path(sys.argv[1]))
    print(json.dumps(greedy(model, "the", 4), indent=1)[:400])
