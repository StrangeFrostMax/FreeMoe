#!/usr/bin/env python3
"""Generate architecture fixtures shaped like real llama.cpp checkpoints.

The `freemoe-tiny` fixtures prove the reference adapter. These fixtures prove the
*standard* adapter: standard architecture names (`qwen2`, `qwen3`, …),
architecture-prefixed metadata (`qwen2.block_count`), llama.cpp tensor names
(`blk.N.attn_q.weight`, …) and the layout quirks each family brings.

Each fixture is paired with goldens produced by `tools/llama_oracle.py`, an
independent implementation of the forward pass. The engine is required to match
them, so "it looks like English" is never the acceptance criterion.

Usage:
    python3 tools/make_arch_fixtures.py          # write fixtures and goldens
    python3 tools/make_arch_fixtures.py --check  # verify they are up to date
"""
from __future__ import annotations

import hashlib
import json
import random
import struct
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_format import TensorSpec, write_gguf  # noqa: E402
import llama_oracle as oracle  # noqa: E402

SEED = 0xA5C4E1
ALIGN = 32
V = 258  # byte tokenizer: 256 bytes plus BOS/EOS
BOS, EOS = 256, 257
PROMPT = "the cat"

# Small but not degenerate: enough heads for GQA to mean something, and a
# context long enough for the prompt plus generated tokens.
SHAPE = {
    "hidden": 32,
    "intermediate": 64,
    "layers": 2,
    "heads": 4,
    "head_dim": 8,
    "context": 64,
}


def f32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def values_for(name: str, count: int, scale: float, *, norm: bool = False) -> list[float]:
    """Deterministic pseudo-random weights, keyed by tensor name."""
    digest = hashlib.sha256(f"FreeMoe-arch:{SEED}:{name}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "little"))
    raw = [(rng.random() * 2.0 - 1.0) * scale for _ in range(count)]
    if norm:
        return [1.0 + x * 0.1 for x in raw]
    return raw


def tensor(name: str, dims: list[int], scale: float, *, norm: bool = False,
           values: list[float] | None = None) -> TensorSpec:
    count = 1
    for d in dims:
        count *= d
    if values is None:
        values = values_for(name, count, scale, norm=norm)
    if len(values) != count:
        raise ValueError(f"{name}: {len(values)} values for {count} elements")
    return TensorSpec(name, list(dims), f32_bytes(values), 0)


def tokenizer_tokens() -> list[str]:
    tokens = [f"<{i}>" for i in range(256)]
    tokens.append("<s>")
    tokens.append("</s>")
    return tokens


def post_norm_name(spec: dict[str, Any], which: str) -> str:
    """The post-norm tensor name: this repo's spelling, or upstream's current one.

    llama.cpp renamed these tensors `post_attention_norm` / `post_ffw_norm`.
    Either spelling names the same tensor in the same place in the block, so one
    fixture of each vintage proves the loader accepts both.
    """
    if spec.get("upstream_post_norms"):
        return ("post_attention_norm.weight" if which == "attn"
                else "post_ffw_norm.weight")
    return ("attn_post_norm.weight" if which == "attn" else "ffn_post_norm.weight")


def arch_tensors(spec: dict[str, Any]) -> list[TensorSpec]:
    h, i, layers = spec["hidden"], spec["intermediate"], spec["layers"]
    heads, hd = spec["heads"], spec["head_dim"]
    kv_heads = spec.get("kv_heads", heads)
    kv_dim = kv_heads * hd
    out = [tensor("token_embd.weight", [h, V], 0.35)]
    if spec.get("position_embd"):
        # One learned row per context position (gpt2/starcoder). Deliberately not
        # constant across positions: a runtime that ignores the table, or adds
        # row 0 for every token, cannot match the golden.
        rows = spec["context"]
        out += [tensor("position_embd.weight", [h, rows], 0.0,
                       values=[0.25 * ((i % 7) - 3) for i in range(h * rows)])]
    if spec.get("token_embd_norm"):
        # bloom normalises the embedding once before block 0; both halves are
        # required, as upstream's `create_tensor` calls demand.
        out += [
            tensor("token_embd_norm.weight", [h], 0.0, norm=True),
            tensor("token_embd_norm.bias", [h], 0.05),
        ]
    for layer in range(layers):
        p = f"blk.{layer}."
        out += [tensor(p + "attn_norm.weight", [h], 0.0, norm=True)]
        # A hybrid stack (jamba) interleaves the two block kinds, so the mask
        # decides what this layer carries. A pure file leaves it unset and every
        # layer gets the same shape.
        recr_here = bool(spec.get("ssm")) and layer in spec.get("ssm_layers", range(layers))
        if recr_here:
            # llama.cpp's mamba block: no attention, and a pure file has no
            # feed-forward either. The A and D tensors carry no ".weight" suffix
            # upstream, so they are written bare here too.
            inner = spec["ssm_inner"]
            d_state = spec["ssm_state"]
            d_conv = spec["ssm_conv"]
            groups = spec.get("ssm_groups", 1)
            mamba2 = spec["ssm"] == "mamba2"
            channels = inner + 2 * groups * d_state if mamba2 else inner
            dt_rank = spec["ssm_dt_rank"]
            in_dim = inner + channels + dt_rank if mamba2 else 2 * inner
            out += [
                tensor(p + "ssm_in.weight", [h, in_dim], 0.16),
                tensor(p + "ssm_conv1d.weight", [d_conv, channels], 0.30),
                tensor(p + "ssm_conv1d.bias", [channels], 0.05),
                # A is negative in a real checkpoint (it is a decay), which the
                # scan multiplies by softplus(dt); a positive one would diverge.
                tensor(p + "ssm_a", [1, dt_rank] if mamba2 else [d_state, inner], -0.20),
                tensor(p + "ssm_d", [1, dt_rank] if mamba2 else [inner], 0.10),
                tensor(p + "ssm_out.weight", [inner, h], 0.16),
            ]
            if mamba2:
                out += [
                    tensor(p + "ssm_dt.bias", [dt_rank], 0.05),
                    tensor(p + "ssm_norm.weight", [inner // groups, groups], 0.0, norm=True),
                ]
            else:
                out += [
                    tensor(p + "ssm_x.weight", [inner, dt_rank + 2 * d_state], 0.16),
                    tensor(p + "ssm_dt.weight", [dt_rank, inner], 0.16),
                    tensor(p + "ssm_dt.bias", [inner], 0.05),
                ]
            if spec.get("ssm_vector_norms"):
                # falcon-mamba / jamba: RMS norms on dt, B and C before the scan.
                for suffix, width in (("ssm_dt_norm", dt_rank), ("ssm_b_norm", d_state),
                                      ("ssm_c_norm", d_state)):
                    out += [tensor(p + suffix + ".weight", [width], 0.0, norm=True)]
            if spec.get("ssm_ffn"):
                out += [
                    tensor(p + "ffn_norm.weight", [h], 0.0, norm=True),
                    tensor(p + "ffn_gate.weight", [h, i], 0.16),
                    tensor(p + "ffn_up.weight", [h, i], 0.16),
                    tensor(p + "ffn_down.weight", [i, h], 0.16),
                ]
            continue
        if spec.get("layernorm") and spec.get("norm_bias", True):
            out += [tensor(p + "attn_norm.bias", [h], 0.05)]
        if spec.get("attn_norm_2"):
            # falcon-40b's second pre-attention norm: its own weight, plus a bias
            # when the file's norms carry one.
            out += [tensor(p + "attn_norm_2.weight", [h], 0.0, norm=True)]
            if spec.get("norm_bias", True):
                out += [tensor(p + "attn_norm_2.bias", [h], 0.05)]
        if spec.get("parallel_residual"):
            # No ffn_norm at all: the FFN consumes the attention norm.
            pass
        if spec.get("mla"):
            # No keys or values: q through its low-rank bottleneck, and the kv
            # latent that each head expands.
            latent = spec["kv_lora_rank"]
            nope = hd - spec["rope_dim"]
            if spec.get("q_lora_rank", 0):
                out += [
                    tensor(p + "attn_q_a.weight", [h, spec["q_lora_rank"]], 0.16),
                    tensor(p + "attn_q_a_norm.weight", [spec["q_lora_rank"]], 0.0, norm=True),
                    tensor(p + "attn_q_b.weight", [spec["q_lora_rank"], heads * hd], 0.16),
                ]
            else:
                out += [tensor(p + "attn_q.weight", [h, heads * hd], 0.16)]
            out += [
                tensor(p + "attn_kv_a_mqa.weight", [h, latent + spec["rope_dim"]], 0.16),
                tensor(p + "attn_kv_a_norm.weight", [latent], 0.0, norm=True),
                tensor(p + "attn_k_b.weight", [nope, latent, heads], 0.20),
                tensor(p + "attn_v_b.weight", [latent, spec["head_v"], heads], 0.20),
            ]
        elif spec.get("fused_qkv"):
            # q, then k, then v along the output axis of one projection.
            out += [tensor(p + "attn_qkv.weight", [h, h + 2 * kv_dim], 0.16)]
            if spec.get("qkv_bias"):
                out += [tensor(p + "attn_qkv.bias", [h + 2 * kv_dim], 0.05)]
        else:
            out += [
                tensor(p + "attn_q.weight", [h, h], 0.16),
                tensor(p + "attn_k.weight", [h, kv_dim], 0.16),
                tensor(p + "attn_v.weight", [h, kv_dim], 0.16),
            ]
        if spec.get("rope_freqs"):
            # One divisor per rotating pair. Values above 1 stretch those pairs'
            # positions; the pattern is uneven so a runtime that applies a single
            # scalar (or none) cannot match the golden.
            pairs = spec["rope_dim"] // 2
            out += [tensor(p + "rope_freqs.weight", [pairs], 0.0,
                           values=[1.0 + 0.5 * (i % 3) for i in range(pairs)])]
        if spec.get("attn_sinks"):
            # One logit per query head. Deliberately non-zero and asymmetric, so a
            # runtime that ignores the sink or reads the wrong head's value cannot
            # match the golden.
            out += [tensor(p + "attn_sinks.weight", [heads], 0.75)]
        if spec.get("qk_norm"):
            # One head's worth of scale, shared by every head.
            out += [
                tensor(p + "attn_q_norm.weight", [hd], 0.0, norm=True),
                tensor(p + "attn_k_norm.weight", [hd], 0.0, norm=True),
            ]
        if spec.get("qkv_bias") and not spec.get("fused_qkv"):
            # A bias of modest size: large enough to matter if the engine forgets
            # it, small enough not to swamp the rest of the arithmetic.
            out += [
                tensor(p + "attn_q.bias", [h], 0.05),
                tensor(p + "attn_k.bias", [kv_dim], 0.05),
                tensor(p + "attn_v.bias", [kv_dim], 0.05),
            ]
        attn_out_width = heads * spec["head_v"] if spec.get("mla") else h
        out += [tensor(p + "attn_output.weight", [attn_out_width, h], 0.16)]
        if spec.get("attn_output_bias"):
            out += [tensor(p + "attn_output.bias", [h], 0.05)]
        if spec.get("post_norms"):
            out += [tensor(p + post_norm_name(spec, "attn"), [h], 0.0, norm=True)]
        if spec.get("ffn_norm_ada"):
            # mistral3's adaptive RMSNorm MLP: one down and one up projection
            # around a gelu_erf, applied to the delay embedding.
            ada_width = spec.get("ada_width", 10)
            out += [
                tensor(p + "ffn_norm_ada_down.weight", [h, ada_width], 0.20),
                tensor(p + "ffn_norm_ada_up.weight", [ada_width, h], 0.20),
            ]
        if spec.get("ffn_norm_from_post"):
            # gpt-oss normalises after the attention residual and calls that
            # tensor attn_post_norm; there is no ffn_norm of its own.
            out += [tensor(p + "attn_post_norm.weight", [h], 0.0, norm=True)]
        elif not spec.get("parallel_residual"):
            out += [tensor(p + "ffn_norm.weight", [h], 0.0, norm=True)]
            if spec.get("layernorm"):
                if spec.get("norm_bias", True):
                    out += [tensor(p + "ffn_norm.bias", [h], 0.05)]
        moe_here = (spec.get("experts") and layer >= spec.get("dense_lead", 0)
                    and layer in spec.get("moe_layers", range(layers)))
        if spec.get("experts") and not moe_here:
            out += [
                tensor(p + "ffn_gate.weight", [h, i], 0.16),
                tensor(p + "ffn_up.weight", [h, i], 0.16),
                tensor(p + "ffn_down.weight", [i, h], 0.16),
            ]
        if spec.get("ffn_no_gate"):
            out += [
                tensor(p + "ffn_up.weight", [h, i], 0.16),
                tensor(p + "ffn_down.weight", [i, h], 0.16),
            ]
            if spec.get("ffn_bias"):
                out += [
                    tensor(p + "ffn_up.bias", [i], 0.05),
                    tensor(p + "ffn_down.bias", [h], 0.05),
                ]
        elif moe_here:
            # Stacked experts: one tensor holds every expert's slice, and the
            # router projects hidden -> expert_count. Routed experts may be
            # narrower than the dense FFN, so the width is its own setting.
            experts = spec["experts"]
            ew = spec.get("expert_width", i)
            out += [
                tensor(p + "ffn_gate_inp.weight", [h, experts], 0.12),
                # Fused layout: gate and up in one tensor, gate block first, which
                # is the order ggml_swiglu splits in.
                *([tensor(p + "ffn_gate_up_exps.weight", [h, 2 * ew, experts], 0.16)]
                  if spec.get("fused_exps") else
                  [tensor(p + "ffn_gate_exps.weight", [h, ew, experts], 0.16),
                   tensor(p + "ffn_up_exps.weight", [h, ew, experts], 0.16)]),
                tensor(p + "ffn_down_exps.weight", [ew, h, experts], 0.16),
            ]
            if spec.get("expert_bias"):
                out += [tensor(p + "ffn_exp_probs_b.bias", [experts], 0.10)]
            if spec.get("router_bias"):
                # gpt-oss biases the router logits themselves; this is a
                # different tensor from the selection bias above, and applying
                # one in place of the other routes a real file wrong.
                out += [tensor(p + "ffn_gate_inp.bias", [experts], 0.10)]
            if spec.get("expert_proj_bias"):
                # One bias column per expert on each projection.
                out += [
                    tensor(p + "ffn_gate_exps.bias", [ew, experts], 0.05),
                    tensor(p + "ffn_up_exps.bias", [ew, experts], 0.05),
                    tensor(p + "ffn_down_exps.bias", [h, experts], 0.05),
                ]
            if spec.get("shared_expert"):
                # A dense FFN beside the routed experts, plus the one-vector
                # projection that produces its per-token scalar gate.
                sw = spec.get("shared_width", i)
                if spec.get("shared_gate", True):
                    out += [tensor(p + "ffn_gate_inp_shexp.weight", [h], 0.30)]
                out += [
                    tensor(p + "ffn_gate_shexp.weight", [h, sw], 0.16),
                    tensor(p + "ffn_up_shexp.weight", [h, sw], 0.16),
                    tensor(p + "ffn_down_shexp.weight", [sw, h], 0.16),
                ]
        elif spec.get("fused_gate_up"):
            # One tensor holds both halves: gate first, then up.
            out += [
                tensor(p + "ffn_up.weight", [h, 2 * i], 0.16),
                tensor(p + "ffn_down.weight", [i, h], 0.16),
            ]
        elif not spec.get("experts"):
            # Plain dense model. A dense-lead layer of an MoE model was already
            # written above, so it must not be written twice.
            out += [
                tensor(p + "ffn_gate.weight", [h, i], 0.16),
                tensor(p + "ffn_up.weight", [h, i], 0.16),
                tensor(p + "ffn_down.weight", [i, h], 0.16),
            ]
            if spec.get("ffn_mlp_bias"):
                out += [
                    tensor(p + "ffn_gate.bias", [i], 0.05),
                    tensor(p + "ffn_up.bias", [i], 0.05),
                    tensor(p + "ffn_down.bias", [h], 0.05),
                ]
        if spec.get("post_norms"):
            out += [tensor(p + post_norm_name(spec, "ffn"), [h], 0.0, norm=True)]
        if spec.get("sub_norms"):
            # BitNet's SUBLN: one norm on the attention output (hidden wide) and
            # one on the gated FFN output (intermediate wide). Both are inside
            # the block, so a post-norm cannot stand in for them.
            out += [
                tensor(p + "attn_sub_norm.weight", [h], 0.0, norm=True),
                tensor(p + "ffn_sub_norm.weight", [i], 0.0, norm=True),
            ]
        if spec.get("projection_scales"):
            # One-element scales, deliberately not 1.0 and different from each
            # other: a runtime that ignores them, or applies one to the wrong
            # projection, cannot match the golden.
            for name, value in (("attn_q", 1.5), ("attn_k", 0.5), ("attn_v", 2.0),
                                ("attn_output", 0.75), ("ffn_gate", 1.25),
                                ("ffn_up", 0.5), ("ffn_down", 2.0)):
                out += [tensor(p + name + ".scale", [1], 0.0, values=[value])]
    out += [tensor("output_norm.weight", [h], 0.0, norm=True)]
    if spec.get("layernorm") and spec.get("norm_bias", True):
        out += [tensor("output_norm.bias", [h], 0.05)]
    if not spec.get("tied_output"):
        out += [tensor("output.weight", [h, V], 0.25)]
    if spec.get("logit_bias"):
        out += [tensor("output.bias", [V], 0.05)]
    return out


def arch_metadata(spec: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    arch = spec["arch"]
    h = spec["hidden"]
    md: dict[str, tuple[Any, ...]] = {
        "general.architecture": ("string", arch),
        "general.name": ("string", spec["name"]),
        "general.alignment": ("u32", ALIGN),
        "general.file_type": ("u32", 0),
        "general.quantization_version": ("u32", 1),
        f"{arch}.block_count": ("u32", spec["layers"]),
        f"{arch}.embedding_length": ("u32", h),
        # A pure state-space file states none of these: llama.cpp's mamba
        # loader reads no head count, no feed-forward width and no rotary width,
        # and writing them here would be a fixture that no real file resembles.
        # A hybrid (jamba) states all of them, and states the split itself as a
        # per-layer head_count_kv whose zero entries are the recurrent layers --
        # which is exactly how llama.cpp reads it.
        **({} if spec.get("ssm") and not spec.get("hybrid") else {
            f"{arch}.feed_forward_length": ("u32", spec["intermediate"]),
            f"{arch}.attention.head_count": ("u32", spec["heads"]),
            **({f"{arch}.attention.head_count_kv": (
                "array_i32", [0 if l in spec["ssm_layers"] else spec["kv_heads"]
                              for l in range(spec["layers"])])}
               if spec.get("hybrid") else
               {f"{arch}.attention.head_count_kv": ("u32", spec["kv_heads"])}),
            f"{arch}.attention.key_length": ("u32", spec["head_dim"]),
            f"{arch}.attention.value_length": ("u32", spec["head_dim"]),
        }),
        # LayerNorm files declare a different epsilon key, and that key is what
        # the adapter uses as evidence for the norm type.
        (f"{arch}.attention.layer_norm_epsilon" if spec.get("layernorm")
         else f"{arch}.attention.layer_norm_rms_epsilon"): ("f32", 1e-5),
        # jamba rotates nothing: upstream's attention layer never calls the rope,
        # so the file carries no rotary keys at all.
        **({} if spec.get("ssm") and not spec.get("hybrid") else
           ({} if spec.get("no_rope") or spec.get("alibi_only")
              or spec.get("position_embd") else {
               f"{arch}.rope.dimension_count": ("u32", spec["rope_dim"]),
               f"{arch}.rope.freq_base": ("f32", 10000.0),
           })),
        f"{arch}.context_length": ("u32", spec["context"]),
        f"{arch}.vocab_size": ("u32", V),
        "tokenizer.ggml.model": ("string", "freemoe.byte-v1"),
        "tokenizer.ggml.bos_token_id": ("u32", BOS),
        "tokenizer.ggml.eos_token_id": ("u32", EOS),
        "tokenizer.ggml.tokens": ("array_string", tokenizer_tokens()),
    }
    if spec.get("chat_template"):
        md["tokenizer.chat_template"] = ("string", spec["chat_template"])
    if spec.get("experts"):
        md[f"{arch}.expert_count"] = ("u32", spec["experts"])
        md[f"{arch}.expert_used_count"] = ("u32", spec.get("expert_used", 2))
    if spec.get("qkv_bias"):
        md[f"{arch}.attention.qkv_bias"] = ("bool", True)
    if spec.get("mla"):
        md[f"{arch}.attention.q_lora_rank"] = ("u32", spec.get("q_lora_rank", 0))
        md[f"{arch}.attention.kv_lora_rank"] = ("u32", spec["kv_lora_rank"])
        md[f"{arch}.attention.value_length_mla"] = ("u32", spec["head_v"])
    if spec.get("dense_lead"):
        md[f"{arch}.leading_dense_block_count"] = ("u32", spec["dense_lead"])
    if spec.get("gating"):
        md[f"{arch}.expert_gating_func"] = ("u32", spec["gating"])
    if spec.get("expert_groups"):
        md[f"{arch}.expert_group_count"] = ("u32", spec["expert_groups"])
        md[f"{arch}.expert_group_used_count"] = ("u32", spec["group_used"])
    if spec.get("expert_width"):
        md[f"{arch}.expert_feed_forward_length"] = ("u32", spec["expert_width"])
    if spec.get("shared_expert"):
        md[f"{arch}.expert_shared_feed_forward_length"] = (
            "u32", spec.get("shared_width", spec["intermediate"]))
    if spec.get("embedding_scale") is not None:
        md[f"{arch}.embedding_scale"] = ("f32", spec["embedding_scale"])
    if spec.get("residual_scale") is not None:
        md[f"{arch}.residual_scale"] = ("f32", spec["residual_scale"])
    if spec.get("logit_scale") is not None:
        md[f"{arch}.logit_scale"] = ("f32", spec["logit_scale"])
    if spec.get("rope_scaling"):
        # Exactly the keys llama.cpp reads; the values are what a YaRN-extended
        # checkpoint actually carries (see e.g. DeepSeek-V2-Lite's config).
        scaling = spec["rope_scaling"]
        md[f"{arch}.rope.scaling.type"] = ("string", scaling["type"])
        md[f"{arch}.rope.scaling.factor"] = ("f32", scaling["factor"])
        md[f"{arch}.rope.scaling.original_context_length"] = (
            "u32", scaling["original_context_length"])
        if scaling.get("attn_factor") is not None:
            md[f"{arch}.rope.scaling.attn_factor"] = ("f32", scaling["attn_factor"])
        if scaling.get("yarn_log_multiplier") is not None:
            md[f"{arch}.rope.scaling.yarn_log_multiplier"] = (
                "f32", scaling["yarn_log_multiplier"])
    if spec.get("rope_sections"):
        # mRoPE section split, four counts of rotary pairs. llama.cpp only turns
        # mRoPE on when the first two are non-zero.
        md[f"{arch}.rope.dimension_sections"] = ("array_i32", list(spec["rope_sections"]))
    if spec.get("ssm"):
        md[f"{arch}.ssm.conv_kernel"] = ("u32", spec["ssm_conv"])
        if spec.get("ssm_vector_norms") or spec.get("ssm_flag_only"):
            md[f"{arch}.ssm.dt_b_c_rms"] = ("bool", True)
        md[f"{arch}.ssm.inner_size"] = ("u32", spec["ssm_inner"])
        md[f"{arch}.ssm.state_size"] = ("u32", spec["ssm_state"])
        md[f"{arch}.ssm.time_step_rank"] = ("u32", spec["ssm_dt_rank"])
        if spec.get("ssm_groups", 1) > 1:
            md[f"{arch}.ssm.group_count"] = ("u32", spec["ssm_groups"])
    if spec.get("realtime_delay") is not None:
        md[f"{arch}.realtime_delay_tokens"] = ("u32", spec["realtime_delay"])
    if spec.get("attn_temp_scale") is not None:
        md[f"{arch}.attention.temperature_scale"] = ("f32", spec["attn_temp_scale"])
    if spec.get("swiglu_clamp"):
        # Per-layer clamp bounds, the shape llama.cpp reads for deepseek4/4.1.
        rows = spec["swiglu_clamp"]
        md[f"{arch}.swiglu_clamp_exp"] = ("array_f32", [float(v) for v in rows])
    if spec.get("alibi_max_bias") is not None:
        # The ALiBi bias knob (`attention.max_alibi_bias`), as bloom (hardcoded
        # upstream) and jais (read from the file) select it.
        md[f"{arch}.attention.max_alibi_bias"] = ("f32", spec["alibi_max_bias"])
    if spec.get("attn_softcap"):
        md[f"{arch}.attn_logit_softcapping"] = ("f32", spec["attn_softcap"])
    if spec.get("final_softcap"):
        md[f"{arch}.final_logit_softcapping"] = ("f32", spec["final_softcap"])
    if spec.get("sliding_window"):
        md[f"{arch}.attention.sliding_window"] = ("u32", spec["sliding_window"])
    if spec.get("sliding_window_pattern") is not None:
        # The period form llama.cpp reads: every layer whose index is not a
        # multiple of the period is windowed.
        md[f"{arch}.attention.sliding_window_pattern"] = (
            "u32", spec["sliding_window_pattern"])
    return md


SPECS: list[dict[str, Any]] = [
    {
        # gpt-oss (llama.cpp models/openai-moe.cpp): attention sinks, a
        # post-attention norm that feeds the MoE (there is no ffn_norm), a biased
        # router, per-expert projection biases, alternating full/windowed layers,
        # and `swiglu_oai` rather than SwiGLU. Five separate facts, and every one
        # of them changes a number.
        "slug": "gpt-oss",
        "arch": "gpt-oss",
        "name": "FreeMoe synthetic gpt-oss with sinks, biased MoE and swiglu_oai",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "attn_sinks": True,
        "attn_output_bias": True,
        "ffn_norm_from_post": True,
        "experts": 4,
        "expert_used": 2,
        "expert_width": 48,
        "router_bias": True,
        "expert_proj_bias": True,
        "sliding_window": 12,
        "sliding_window_pattern": 2,
        "layers": 4,
    },
    {
        # qwen3vl (llama.cpp models/qwen3vl.cpp): the qwen3 text block with
        # multi-resolution rotary sections, in the interleaved layout llama.cpp
        # selects by architecture rather than by file flag.
        "slug": "qwen3vl",
        "arch": "qwen3vl",
        "name": "FreeMoe synthetic qwen3vl text path with mRoPE sections",
        "kv_heads": 2,
        "rope_dim": 4,
        "rope_sections": [1, 1, 0, 0],
        "qk_norm": True,
    },
    {
        # qwen3vlmoe: the same mRoPE text path with stacked experts.
        "slug": "qwen3vlmoe",
        "arch": "qwen3vlmoe",
        "name": "FreeMoe synthetic qwen3vlmoe text path with mRoPE sections",
        "kv_heads": 2,
        "rope_dim": 4,
        "rope_sections": [1, 1, 0, 0],
        "qk_norm": True,
        "experts": 4,
        "expert_used": 2,
        "expert_width": 48,
        "fused_exps": True,
    },
    {
        # mamba (llama.cpp models/mamba.cpp): no attention at all. Each layer is
        # a 1D convolution and a state-space scan over a rolling state, and a
        # pure mamba file has no feed-forward either.
        "slug": "mamba",
        "arch": "mamba",
        "name": "FreeMoe synthetic mamba (recurrent block, no attention)",
        "ssm": "mamba",
        "ssm_inner": 64,      # 2 * hidden, the only expansion mamba supports
        "ssm_state": 4,
        "ssm_conv": 4,
        "ssm_dt_rank": 3,
        "layers": 2,
    },
    {
        # falcon-mamba's variant of the mamba block: the same scan with an RMS
        # norm on dt, B and C first (llama.cpp's `ssm.dt_b_c_rms`). The norms
        # change the scan's inputs, so a runtime that ignores them diverges.
        "slug": "mamba-dtrms",
        "arch": "mamba",
        "name": "FreeMoe synthetic mamba with dt/B/C RMS norms",
        "ssm": "mamba",
        "ssm_inner": 64,
        "ssm_state": 4,
        "ssm_conv": 3,
        "ssm_dt_rank": 3,
        "ssm_vector_norms": True,
        "layers": 2,
    },
    {
        # The same mamba block with `ssm.dt_b_c_rms` set and none of the three
        # weights. Upstream creates those tensors only for jamba, so a mamba
        # file that sets the flag has null weights and `build_norm` degenerates
        # to unit gain -- the norm still runs, it just does not scale. A runtime
        # that treats the missing weights as "no norm" diverges from the
        # reference on this file, and only this file can show it.
        "slug": "mamba-flagonly",
        "arch": "mamba",
        "name": "FreeMoe synthetic mamba with dt/B/C norms and no weights",
        "ssm": "mamba",
        "ssm_inner": 64,
        "ssm_state": 4,
        "ssm_conv": 4,
        "ssm_dt_rank": 3,
        "ssm_flag_only": True,
        "layers": 2,
    },
    {
        # gpt2 (llama.cpp src/models/gpt2.cpp): biased LayerNorm, one fused qkv
        # with its bias, a gate-free GELU FFN with biases on both projections,
        # and a learned position table added to the embedding. No rotary at all.
        "slug": "gpt2",
        "arch": "gpt2",
        "name": "FreeMoe synthetic gpt2 with absolute position embeddings",
        "layernorm": True,
        "position_embd": True,
        "no_rope": True,
        "fused_qkv": True,
        "qkv_bias": True,
        "ffn_no_gate": True,
        "ffn_bias": True,
        "attn_output_bias": True,
    },
    {
        # starcoder (llama.cpp src/models/starcoder.cpp): the same block as gpt2,
        # with its own position table and a tied output head left out.
        "slug": "starcoder",
        "arch": "starcoder",
        "name": "FreeMoe synthetic starcoder with absolute position embeddings",
        "layernorm": True,
        "position_embd": True,
        "no_rope": True,
        "fused_qkv": True,
        "qkv_bias": True,
        "ffn_no_gate": True,
        "ffn_bias": True,
        "attn_output_bias": True,
    },
    {
        # falcon-40b (llama.cpp src/models/falcon.cpp): a second pre-attention
        # norm feeding the qkv projection, while the FFN keeps reading the first.
        # Rotary for the attention, biased LayerNorm, a gate-free GELU FFN.
        "slug": "falcon",
        "arch": "falcon",
        "name": "FreeMoe synthetic falcon-40b with a second pre-attention norm",
        "layernorm": True,
        "attn_norm_2": True,
        "fused_qkv": True,
        "ffn_no_gate": True,
    },
    {
        # mpt (llama.cpp src/models/mpt.cpp): biased LayerNorm on both block norms,
        # a fused qkv with its bias, a gate-free GELU FFN with biases on both
        # projections, and an *optional* learned position table added to the
        # embedding. No rotary anywhere.
        "slug": "mpt",
        "arch": "mpt",
        "name": "FreeMoe synthetic mpt with an absolute position table",
        "layernorm": True,
        "position_embd": True,
        "no_rope": True,
        "fused_qkv": True,
        "qkv_bias": True,
        "ffn_no_gate": True,
        "ffn_bias": True,
        "attn_output_bias": True,
    },
    {
        # bloom (llama.cpp src/models/bloom.cpp): bias-carrying LayerNorm on the
        # embedding, on both block norms and on the final norm; one fused qkv
        # with its bias; a gate-free GELU MLP; and ALiBi instead of any rotation.
        # Every one of those is a number the golden can tell apart.
        "slug": "bloom",
        "arch": "bloom",
        "name": "FreeMoe synthetic bloom with ALiBi and an embedding norm",
        "layernorm": True,
        "token_embd_norm": True,
        "fused_qkv": True,
        "qkv_bias": True,
        "ffn_no_gate": True,
        "ffn_bias": True,
        "attn_output_bias": True,
        "alibi_only": True,
    },
    {
        # jais (llama.cpp src/models/jais.cpp): the same bias-carrying LayerNorm
        # and fused qkv, a gated SwiGLU FFN, no rotation at all, and ALiBi from
        # the metadata key rather than a hardcoded constant.
        "slug": "jais",
        "arch": "jais",
        "name": "FreeMoe synthetic jais with ALiBi",
        "layernorm": True,
        "fused_qkv": True,
        "qkv_bias": True,
        "attn_output_bias": True,
        "alibi_only": True,
        "alibi_max_bias": 8.0,
    },
    {
        # deci (llama.cpp src/models/deci.cpp): RMSNorm, SwiGLU, separate q/k/v and optional YaRN rope factors.
        # The block itself is llama's.
        "slug": "deci",
        "arch": "deci",
        "name": "FreeMoe synthetic deci",
    },
    {
        # nanbeige (llama.cpp src/models/nanbeige.cpp): RMSNorm and SwiGLU with a plain rope, plus optional rope_freqs.
        "slug": "nanbeige",
        "arch": "nanbeige",
        "name": "FreeMoe synthetic nanbeige",
    },
    {
        # xverse (llama.cpp src/models/xverse.cpp): RMSNorm, SwiGLU, no biases anywhere; the llama block verbatim.
        "slug": "xverse",
        "arch": "xverse",
        "name": "FreeMoe synthetic xverse",
    },
    {
        # maincoder (llama.cpp src/models/maincoder.cpp): RMSNorm on q and k inside the block, then RMSNorm + SwiGLU.
        "slug": "maincoder",
        "arch": "maincoder",
        "name": "FreeMoe synthetic maincoder",
        "qk_norm": True,
    },
    {
        # qwen (llama.cpp src/models/qwen.cpp): one fused qkv projection with a bias, not three separate ones.
        "slug": "qwen",
        "arch": "qwen",
        "name": "FreeMoe synthetic qwen",
        "fused_qkv": True, "qkv_bias": True,
    },
    {
        # codeshell (llama.cpp src/models/codeshell.cpp): LayerNorm with bias, GeGLU applied to a single up projection.
        "slug": "codeshell",
        "arch": "codeshell",
        "name": "FreeMoe synthetic codeshell",
        "layernorm": True, "ffn_no_gate": True,
    },
    {
        # starcoder2 (llama.cpp src/models/starcoder2.cpp): LayerNorm with bias, a gate-free GeGLU FFN, and rope.
        "slug": "starcoder2",
        "arch": "starcoder2",
        "name": "FreeMoe synthetic starcoder2",
        "layernorm": True, "ffn_no_gate": True,
    },
    {
        # pockettts (llama.cpp src/models/pockettts.cpp): LayerNorm with bias, gate-free GeGLU, output projection absent.
        "slug": "pockettts",
        "arch": "pockettts",
        "name": "FreeMoe synthetic pockettts",
        "layernorm": True, "ffn_no_gate": True,
    },
    {
        # cohere2 (llama.cpp src/models/cohere2.cpp): bias-free LayerNorm and an SWA pattern: every layer whose index is
        # not a multiple of four is windowed.
        "slug": "cohere2",
        "arch": "cohere2",
        "name": "FreeMoe synthetic cohere2",
        "layernorm": True, "norm_bias": False, "sliding_window": 8,
        "sliding_window_pattern": 4,
    },
    {
        # paddleocr (llama.cpp src/models/paddleocr.cpp): RMSNorm + SwiGLU language decoder, plain rope.
        "slug": "paddleocr",
        "arch": "paddleocr",
        "name": "FreeMoe synthetic paddleocr",
    },
    {
        # hunyuan-dense (llama.cpp src/models/hunyuan-dense.cpp): reuses the hunyuan-vl decoder: q/k RMSNorm then the llama block.
        "slug": "hunyuan-dense",
        "arch": "hunyuan-dense",
        "name": "FreeMoe synthetic hunyuan-dense",
        "qk_norm": True,
    },
    {
        # llama-embed (llama.cpp src/models/llama-embed.cpp): llama's graph with an embedding head: same block, same tensors.
        "slug": "llama-embed",
        "arch": "llama-embed",
        "name": "FreeMoe synthetic llama-embed",
    },
    {
        # mamba2 (llama.cpp models/mamba2.cpp): a scalar decay per head and
        # grouped B/C, so the same convolution covers the state and the groups.
        "slug": "mamba2",
        "arch": "mamba2",
        "name": "FreeMoe synthetic mamba2 (grouped scan, scalar decay)",
        "ssm": "mamba2",
        "ssm_inner": 64,
        "ssm_state": 4,
        "ssm_conv": 4,
        "ssm_dt_rank": 4,
        "ssm_groups": 2,
        "layers": 2,
    },
    {
        # The same mamba2 block carrying the dt/B/C norm weights a falcon-mamba
        # style file has, plus `ssm.dt_b_c_rms`. Upstream's mamba2 layer applies
        # none of them -- `build_mamba2_layer` has no dt/B/C norm, and
        # `ssm.dt_b_c_rms` is a key only models/mamba.cpp reads -- so a runtime
        # that applies them computes different numbers. Because every tensor is
        # generated from its own name, the shared weights are the same sequence
        # as `mamba2`, which makes the pair a differential test: the norms must
        # change nothing.
        "slug": "mamba2-dtnorm",
        "arch": "mamba2",
        "name": "FreeMoe synthetic mamba2 carrying unused dt/B/C norm weights",
        "ssm": "mamba2",
        "ssm_inner": 64,
        "ssm_state": 4,
        "ssm_conv": 4,
        "ssm_dt_rank": 4,
        "ssm_groups": 2,
        "ssm_vector_norms": True,
        "layers": 2,
    },
    {
        # jamba (llama.cpp models/jamba.cpp): a hybrid stack, half mamba and half
        # attention, with a feed-forward behind both kinds of layer. Three things
        # in it are per-layer rather than model-wide: the recurrence mask, the
        # feed-forward presence and the router (jamba mixes dense and MoE blocks).
        # Its attention layers also carry no rotary position at all -- upstream's
        # block never calls the rope -- so the file states no rotary keys.
        "slug": "jamba",
        "arch": "jamba",
        "name": "FreeMoe synthetic jamba (mamba and attention layers interleaved)",
        "ssm": "mamba",
        "hybrid": True,
        "ssm_layers": (0, 2),
        "ssm_inner": 64,      # mamba-1 expansion: 2 * hidden
        "ssm_state": 4,
        "ssm_conv": 4,
        "ssm_dt_rank": 3,
        "ssm_vector_norms": True,
        "ssm_ffn": True,
        "kv_heads": 2,
        "no_rope": True,
        "experts": 4,
        "expert_width": 16,
        "moe_layers": (1, 3),
        "layers": 4,
    },
    {
        # mistral3 (llama.cpp models/mistral3.cpp): the llama block plus three
        # optional pieces that all change numbers — biases on all three FFN
        # projections, an adaptive RMSNorm whose MLP is a gelu_erf over a
        # sinusoidal delay embedding, and llama.cpp's per-position attention
        # temperature. This is the file that makes `gelu_erf` reachable.
        "slug": "mistral3",
        "arch": "mistral3",
        "name": "FreeMoe synthetic mistral3 with an adaptive norm and FFN biases",
        "kv_heads": 2,
        "rope_dim": 4,
        "ffn_mlp_bias": True,
        "ffn_norm_ada": True,
        "ada_width": 10,
        "realtime_delay": 7,
        "attn_temp_scale": 0.25,
        "rope_scaling": {"type": "yarn", "factor": 2.0,
                         "original_context_length": 32, "attn_factor": 1.0},
        "layers": 2,
    },
    {
        # plm (llama.cpp models/plm.cpp): multi-head latent attention with a
        # one-step q projection (q_lora_rank 0), a gate-free relu^2 FFN and the
        # output head tied to the token table. This is what makes the plain
        # `relu_sq` primitive reachable end to end.
        "slug": "plm",
        "arch": "plm",
        "name": "FreeMoe synthetic plm with MLA-lite and a gate-free relu^2 FFN",
        "kv_heads": SHAPE["heads"],
        "rope_dim": 4,
        "mla": True,
        "q_lora_rank": 0,
        "kv_lora_rank": 16,
        "head_v": 10,
        "ffn_no_gate": True,
        "tied_output": True,
    },
    {
        # arcee (llama.cpp models/arcee.cpp): the llama block with a gate-free
        # FFN and relu^2 in place of SwiGLU. GGUF metadata cannot state an
        # activation, so the architecture name is the evidence and this fixture
        # is the check that the engine applies the same one.
        "slug": "arcee",
        "arch": "arcee",
        "name": "FreeMoe synthetic arcee with a gate-free relu^2 FFN",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "ffn_no_gate": True,
    },
    {
        # nemotron: LayerNorm (biased) on all three norms and a gate-free relu^2
        # FFN with biases on both projections. Distinct from arcee because the
        # norm type and the FFN biases are separate facts.
        "slug": "nemotron",
        "arch": "nemotron",
        "name": "FreeMoe synthetic nemotron with LayerNorm biases and relu^2 FFN",
        "kv_heads": SHAPE["heads"],
        "rope_dim": SHAPE["head_dim"],
        "layernorm": True,
        "ffn_no_gate": True,
        "ffn_bias": True,
    },
    {
        # jais2: LayerNorm, a biased attention output projection, the same
        # gate-free relu^2 FFN, and the output head tied to the token table
        # (llama.cpp marks it TENSOR_NOT_REQUIRED and falls back).
        "slug": "jais2",
        "arch": "jais2",
        "name": "FreeMoe synthetic jais2 with biased projections and relu^2 FFN",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "layernorm": True,
        "ffn_no_gate": True,
        "ffn_bias": True,
        "attn_output_bias": True,
        "tied_output": True,
    },
    {
        # bitnet (llama.cpp's models/bitnet.cpp): the plain pre-norm block plus
        # two SUBLNs -- an RMSNorm on the attention output before `attn_output`
        # and one on the gated FFN output before `ffn_down` -- and the optional
        # one-element `*.scale` on every projection. The output head is tied to
        # the token table, as upstream's FIXME comment describes.
        "slug": "bitnet",
        "arch": "bitnet",
        "name": "FreeMoe synthetic bitnet with SUBLN sub-norms and projection scales",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "tied_output": True,
        "sub_norms": True,
        "projection_scales": True,
    },
    {
        # glm4: fused gate+up FFN (gate first, then up), and post-attention /
        # post-FFN norms applied before each residual add. No separate gate.
        "slug": "glm4",
        "arch": "glm4",
        "name": "FreeMoe synthetic glm4 with fused gate-up and post norms",
        "kv_heads": SHAPE["heads"],
        "rope_dim": SHAPE["head_dim"],
        "fused_gate_up": True,
        "post_norms": True,
    },
    {
        # minicpm: the token embedding is scaled before the first block. The file
        # also declares residual_scale and logit_scale, which llama.cpp parses
        # and never applies -- so the fixture below is the check that we do not
        # apply them either.
        "slug": "minicpm",
        "arch": "minicpm",
        "name": "FreeMoe synthetic minicpm",
        "kv_heads": SHAPE["heads"],
        "rope_dim": SHAPE["head_dim"],
        "tied_output": True,
        "embedding_scale": 12.0,
        "residual_scale": 0.0989,
        "logit_scale": 8.0,
    },
    {
        # qwen3moe: qwen3 semantics (per-head q/k norm) with stacked-expert FFN
        # and a router. Tests the combination rather than either half alone.
        "slug": "qwen3moe",
        "arch": "qwen3moe",
        "name": "FreeMoe synthetic qwen3moe",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "qk_norm": True,
        "experts": 4,
        "expert_used": 2,
    },
    {
        # deepseek2: multi-head latent attention — q through a low-rank
        # bottleneck, and a per-token kv latent that every head expands. One
        # leading dense block, sigmoid gating with a selection bias, and
        # group-limited routing, which together are the V3 layout. Every width
        # is distinct (key 12 = 8 nope + 4 rope, value 10, latent 16) so a
        # runtime that confuses them cannot match the golden.
        "slug": "deepseek2-mla",
        "arch": "deepseek2",
        "name": "FreeMoe synthetic deepseek2 with MLA and group routing",
        "kv_heads": SHAPE["heads"],
        "rope_dim": 4,
        "mla": True,
        "q_lora_rank": 12,
        "kv_lora_rank": 16,
        "head_v": 10,
        "layers": 3,
        "dense_lead": 1,
        "experts": 4,
        "expert_used": 2,
        "expert_width": 48,
        "gating": 2,          # sigmoid
        "expert_bias": True,
        "expert_groups": 2,
        "group_used": 1,
        "shared_expert": True,
        "shared_gate": False,
        "shared_width": 40,
    },
    {
        # internlm2: the plain block with nothing added. It exists to prove the
        # block is driven by the checkpoint, not by one family's code path: the
        # only difference from `llama-gqa` is the architecture string.
        "slug": "internlm2",
        "arch": "internlm2",
        "name": "FreeMoe synthetic internlm2 (plain block, different arch string)",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
    },
    {
        # exaone: RMSNorm and SwiGLU with a tied output projection, which is the
        # same block again with a different vocabulary size.
        "slug": "exaone",
        "arch": "exaone",
        "name": "FreeMoe synthetic exaone with a tied output",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "tied_output": True,
    },
    {
        # orion: LayerNorm *with biases on every norm*, sequential residual (the
        # FFN has its own norm, unlike phi2).
        "slug": "orion",
        "arch": "orion",
        "name": "FreeMoe synthetic orion with LayerNorm biases",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "layernorm": True,
    },
    {
        # gptneox: one fused q/k/v projection with a fused bias, a bias on the
        # attention output, LayerNorm with biases, and a gate-free GELU FFN with
        # biases on both projections. Four separate biases, so forgetting any of
        # them changes the logits.
        "slug": "gptneox",
        "arch": "gptneox",
        "name": "FreeMoe synthetic gpt-neox (fused qkv, gate-free GELU)",
        "kv_heads": SHAPE["heads"],
        "rope_dim": SHAPE["head_dim"],
        "layernorm": True,
        "fused_qkv": True,
        "qkv_bias": True,
        "attn_output_bias": True,
        "ffn_no_gate": True,
        "ffn_bias": True,
    },
    {
        # stablelm: LayerNorm with biases and a rotary width narrower than the
        # head, so the unrotated tail of every key and query must survive.
        "slug": "stablelm",
        "arch": "stablelm",
        "name": "FreeMoe synthetic stablelm (partial rotary, LayerNorm)",
        "kv_heads": 2,
        "rope_dim": 4,
        "layernorm": True,
    },
    {
        # phi2: LayerNorm with a bias instead of RMSNorm, no ffn_norm at all (the
        # FFN runs on the attention's normalised input and both branches are
        # added to one residual), a gate-free GELU FFN with biases, and biases on
        # the attention output and the logits.
        "slug": "phi2-layernorm",
        "arch": "phi2",
        "name": "FreeMoe synthetic phi2 with LayerNorm and a parallel residual",
        "kv_heads": SHAPE["heads"],
        "rope_dim": SHAPE["head_dim"],
        "layernorm": True,
        "parallel_residual": True,
        "ffn_no_gate": True,
        "ffn_bias": True,
        "attn_output_bias": True,
        "logit_bias": True,
    },
    {
        # qwen2moe: routed experts plus a dense shared expert, whose output is
        # scaled by one scalar per token. The three feed-forward widths are all
        # different (64 dense, 48 per routed expert, 40 shared) so a runtime that
        # reuses one width for another cannot match the golden.
        "slug": "qwen2moe",
        "arch": "qwen2moe",
        "name": "FreeMoe synthetic qwen2moe with a shared expert",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "experts": 4,
        "expert_used": 2,
        "expert_width": 48,
        "shared_expert": True,
        "shared_width": 40,
    },
    {
        # phi3: one fused q/k/v projection with a fused bias, and a rotary width
        # narrower than the head (half the head rotates, the rest passes through).
        "slug": "phi3-fused",
        "arch": "phi3",
        "name": "FreeMoe synthetic phi3 with fused qkv and partial rotary",
        "kv_heads": SHAPE["heads"],
        "rope_dim": SHAPE["head_dim"] // 2,
        "fused_qkv": True,
        "qkv_bias": True,
    },
    {
        # gemma v1: scaled embeddings, zero-centred norm scales, GeGLU, and no
        # separate output tensor (the token table is the output projection).
        "slug": "gemma",
        "arch": "gemma",
        "name": "FreeMoe synthetic gemma v1",
        "kv_heads": SHAPE["heads"],
        "rope_dim": SHAPE["head_dim"],
        "tied_output": True,
    },
    {
        # gemma2: adds post-attention/post-FFN norms, tanh softcapping on both
        # attention scores and final logits, and a sliding window.
        "slug": "gemma2",
        "arch": "gemma2",
        "name": "FreeMoe synthetic gemma2",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "post_norms": True,
        "attn_softcap": 50.0,
        "final_softcap": 30.0,
        "sliding_window": 4,
        "tied_output": False,
    },
    {
        # gemma2, written with the post-norm names current llama.cpp converters
        # emit (`post_attention_norm` / `post_ffw_norm`). Identical arithmetic.
        # original: adds post-attention/post-FFN norms, tanh softcapping on both
        # attention scores and final logits, and a sliding window.
        "slug": "gemma2-renamed",
        "arch": "gemma2",
        "name": "FreeMoe synthetic gemma2 (current post-norm names)",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "post_norms": True,
        "attn_softcap": 50.0,
        "final_softcap": 30.0,
        "sliding_window": 4,
        "tied_output": False,
        "upstream_post_norms": True,
    },
    {
        # Grouped-query attention with an exact head ratio, no bias. This is the
        # shape most current small checkpoints use.
        "slug": "llama-gqa",
        "arch": "llama",
        "name": "FreeMoe synthetic llama with grouped-query attention",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        # A real ChatML template, so the chat path is exercised end to end
        # against a checkpoint that carries one rather than only against the
        # built-in table.
        "chat_template": ("{% for message in messages %}"
                          "{{ '<|im_start|>' + message['role'] + '\n' + message['content']"
                          " + '<|im_end|>\n' }}"
                          "{% endfor %}"
                          "{% if add_generation_prompt %}"
                          "{{ '<|im_start|>assistant\n' }}{% endif %}"),
    },
    {
        # qwen3: grouped-query attention plus per-head RMSNorm on q and k.
        "slug": "qwen3-qknorm",
        "arch": "qwen3",
        "name": "FreeMoe synthetic qwen3 with per-head q/k normalisation",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "qk_norm": True,
    },
    {
        # qwen2: grouped-query attention plus biases on q/k/v.
        "slug": "qwen2-bias",
        "arch": "qwen2",
        "name": "FreeMoe synthetic qwen2 with projection biases",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "qkv_bias": True,
    },
    {
        # Fused gate+up expert tensors: one tensor per layer instead of two, gate
        # block first. The layout cohere2moe, dots3note, gemma4 and deepseek3.2
        # use; the block is the existing stacked-expert MoE, so this fixture pins
        # the residency/extent split rather than a family.
        "slug": "qwen3moe-fused-up",
        "arch": "qwen3moe",
        "name": "FreeMoe synthetic qwen3moe with fused gate-up expert tensors",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "qk_norm": True,
        "experts": 4,
        "expert_used": 2,
        "fused_exps": True,
    },
    {
        # Clamped SwiGLU with a per-layer bound (`swiglu_clamp_exp`): deepseek4/4.1
        # and glm5-next square their activation this way. The fixture's bounds
        # differ per layer and are small enough that the clamp actually binds.
        "slug": "qwen3moe-clamp",
        "arch": "qwen3moe",
        "name": "FreeMoe synthetic qwen3moe with per-layer clamped SwiGLU",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "qk_norm": True,
        "experts": 4,
        "expert_used": 2,
        "layers": 3,
        "swiglu_clamp": [0.7, 0.0, 1.3],
    },
    {
        # mRoPE: the same glm4 block plus a section split. On a text prompt all
        # four positions coincide, so the token stream must be unchanged; the
        # fixture pins that the metadata is read, and the unit test pins the
        # section assignment itself.
        "slug": "glm4-mrope",
        "arch": "glm4",
        "name": "FreeMoe synthetic glm4 with mRoPE sections",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "rope_sections": [2, 1, 1, 0],
    },
    {
        # SmolLM3: the standard llama block with every fourth layer's rotary step
        # removed (NoPE). Four layers are used so the skip actually happens at
        # layer 3 -- a two-layer fixture would never exercise it.
        "slug": "smollm3-nope",
        "arch": "smollm3",
        "name": "FreeMoe synthetic SmolLM3 with NoPE layers",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "layers": 4,
    },
    {
        # Per-pair rotary frequency divisors (`rope_freqs.weight`): the pairs that
        # carry them advance slower, which is how granite/llama4-style checkpoints
        # stretch context without YaRN.
        "slug": "llama-ropfreqs",
        "arch": "llama",
        "name": "FreeMoe synthetic llama with per-pair rotary frequency factors",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "rope_freqs": True,
    },
    {
        # Attention sinks (ggml `soft_max_ext(..., sinks)`): every head's softmax
        # denominator carries one extra learnable logit. Families that need it
        # include deepseek4/4.1, mimo2 and minimax-m2.
        "slug": "llama-sinks",
        "arch": "llama",
        "name": "FreeMoe synthetic llama with attention sinks",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "attn_sinks": True,
    },
    {
        # YaRN context extension: the checkpoint was trained on 64 positions and
        # stretches them by 4x. Both halves of the algorithm matter here -- the
        # per-dimension ramp (high frequencies keep their exact angles, low ones
        # interpolate) and the magnitude scale (1.0 + 0.1*log(4)). A fixture with
        # only a factor would pass even if the ramp were ignored.
        "slug": "llama-yarn",
        "arch": "llama",
        "name": "FreeMoe synthetic llama with YaRN context extension",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "rope_scaling": {
            "type": "yarn",
            "factor": 4.0,
            "original_context_length": SHAPE["context"],
        },
    },
    {
        # Linear (position-interpolation) extension with a magnitude multiplier
        # from the file. Its rotation is the same `position/factor` every
        # long-context qwen2 checkpoint uses, and it checks that `attn_factor` is
        # applied on top of the mode's own scaling rather than instead of it.
        "slug": "qwen2-linear",
        "arch": "qwen2",
        "name": "FreeMoe synthetic qwen2 with linear position interpolation",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "rope_scaling": {
            "type": "linear",
            "factor": 2.0,
            "original_context_length": SHAPE["context"],
            "attn_factor": 1.25,
        },
    },
    {
        # deepseek2-style YaRN, which stores a mscale_all_dim other than 1 in
        # `yarn_log_multiplier` and expects it in the numerator. llama.cpp
        # special-cases exactly this, so the fixture pins the behaviour.
        "slug": "llama-yarn-mscale",
        "arch": "llama",
        "name": "FreeMoe synthetic llama with YaRN mscale_all_dim",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
        "rope_scaling": {
            "type": "yarn",
            "factor": 8.0,
            "original_context_length": SHAPE["context"],
            "yarn_log_multiplier": 1.5,
        },
    },
]


def raw_tensor(name: str, dims: list[int], type_id: int) -> TensorSpec:
    """A tensor whose payload is deliberately opaque.

    The probe must judge these files from their metadata and type ids, not from
    whether the bytes happen to decode to something, so deterministic filler is
    the honest choice here.
    """
    from gguf_format import tensor_byte_size

    size = tensor_byte_size(dims, type_id)
    seed = int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little")
    rng = random.Random(seed)
    return TensorSpec(name, list(dims), bytes(rng.getrandbits(8) for _ in range(size)), type_id)


def build_ternary_fixture() -> dict[str, Any]:
    """PrismML Bonsai 2 shape: PTQ1_0 weights in a rotated basis, hybrid attention."""
    h, layers = 128, 1
    tensors = [
        raw_tensor("token_embd.weight", [h, h], 143),
        raw_tensor("blk.0.attn_norm.weight", [h], 0),
        raw_tensor("blk.0.attn_qkv.weight", [h, 3 * h], 143),
        raw_tensor("blk.0.attn_gate.weight", [h, h], 143),
        raw_tensor("blk.0.ssm_alpha.weight", [h, h], 0),
        raw_tensor("blk.0.ssm_conv1d.weight", [h, 4], 0),
        raw_tensor("blk.0.ffn_norm.weight", [h], 0),
        raw_tensor("blk.0.ffn_gate.weight", [h, h], 143),
        raw_tensor("blk.0.ffn_up.weight", [h, h], 143),
        raw_tensor("blk.0.ffn_down.weight", [h, h], 143),
        raw_tensor("output_norm.weight", [h], 0),
    ]
    md: dict[str, tuple[Any, ...]] = {
        "general.architecture": ("string", "qwen35"),
        "general.name": ("string", "FreeMoe synthetic Bonsai-2-shaped ternary checkpoint"),
        "general.alignment": ("u32", ALIGN),
        "general.file_type": ("u32", 143),
        "qwen35.block_count": ("u32", layers),
        "qwen35.embedding_length": ("u32", h),
        "qwen35.feed_forward_length": ("u32", h),
        "qwen35.attention.head_count": ("u32", 4),
        "qwen35.attention.head_count_kv": ("u32", 4),
        "qwen35.attention.key_length": ("u32", 32),
        "qwen35.attention.layer_norm_rms_epsilon": ("f32", 1e-6),
        "qwen35.rope.dimension_count": ("u32", 32),
        "qwen35.context_length": ("u32", 64),
        "qwen35.vocab_size": ("u32", V),
        "tokenizer.ggml.model": ("string", "gpt2"),
        "tokenizer.ggml.bos_token_id": ("u32", BOS),
        "tokenizer.ggml.eos_token_id": ("u32", EOS),
        "tokenizer.ggml.tokens": ("array_string", tokenizer_tokens()),
        # The rotation is what makes the weights unreadable without the matching
        # activation transform.
        "prism.hadamard.version": ("u32", 1),
        "prism.hadamard.block_size": ("u32", 1024),
        "prism.hadamard.transform": ("string", "fwht"),
        "prism.hadamard.sign_mode": ("string", "fixed"),
    }
    path = ROOT / "tests" / "fixtures" / "ternary-bonsai-shape.gguf"
    write_gguf(path, md, tensors)
    return {"slug": "ternary-bonsai", "model": str(path), "tensors": len(tensors)}


def fixture_path(slug: str) -> Path:
    return ROOT / "tests" / "fixtures" / f"{slug}-shape.gguf"


def golden_path(slug: str) -> Path:
    return ROOT / "tests" / "golden" / f"{slug}_reference.txt"


def render_golden(spec: dict[str, Any], prefill: dict[str, Any],
                  greedy: dict[str, Any]) -> str:
    """Line-oriented golden.

    Deliberately not JSON: the test binary has no JSON reader, and a format the
    test cannot parse would be documentation pretending to be a test.
    """
    def floats(values: list[float]) -> str:
        return " ".join(f"{v:.6f}" for v in values)

    lines = [
        "# FreeMoe architecture reference, produced by tools/llama_oracle.py",
        f"# fixture: {spec['slug']}-shape.gguf",
        "# An independent implementation of the forward pass; the engine must match.",
        f"adapter {spec['arch']}",
        f"prompt {PROMPT}",
        f"bos {prefill['input_ids'][0]}",
        f"input_ids {len(prefill['input_ids'])} {' '.join(str(i) for i in prefill['input_ids'])}",
        f"logits {len(prefill['prefill_logits'])} {floats(prefill['prefill_logits'])}",
        f"greedy {len(greedy['generated_ids'])} "
        f"{' '.join(str(i) for i in greedy['generated_ids'])}",
    ]
    return "\n".join(lines) + "\n"


def build(spec: dict[str, Any]) -> dict[str, Any]:
    resolved = {**SHAPE, **{k: v for k, v in spec.items() if k in SHAPE}}
    resolved.update({k: spec[k] for k in ("arch", "name")})
    # A state-space file states no rotary width and no head count.
    resolved.setdefault("rope_dim", resolved.get("head_dim"))
    resolved.setdefault("qkv_bias", False)
    resolved["kv_heads"] = spec.get("kv_heads", resolved["heads"])

    model_path = fixture_path(spec["slug"])
    corpus = {**SHAPE, **spec}
    # The defaults `resolved` computed above are what `--check` validates, so
    # they have to reach the writer too: a spec that omits them would otherwise
    # serialise a file the checker could never reproduce.
    corpus.setdefault("rope_dim", spec.get("head_dim", SHAPE["head_dim"]))
    corpus["kv_heads"] = spec.get("kv_heads", SHAPE["heads"])
    write_gguf(model_path, arch_metadata(corpus), arch_tensors(corpus))

    cfg = oracle.load_model(model_path)
    prefill = oracle.prefill(cfg, PROMPT)
    greedy = oracle.greedy(cfg, PROMPT, 8)
    path = golden_path(spec["slug"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_golden(spec, prefill, greedy))
    return {"slug": spec["slug"], "tensors": len(arch_tensors(corpus)),
            "model": str(model_path), "golden": str(path)}


def main() -> int:
    if "--check" in sys.argv:
        stale = []
        for spec in SPECS:
            resolved = {**SHAPE, **spec}
            resolved.setdefault("rope_dim", resolved.get("head_dim"))
            resolved.setdefault("qkv_bias", False)
            resolved["kv_heads"] = spec.get("kv_heads", resolved["heads"])
            expected_meta = arch_metadata(resolved)
            expected_tensors = arch_tensors(resolved)
            model_path = fixture_path(spec["slug"])
            if not model_path.exists():
                stale.append(f"{spec['slug']}: missing fixture")
                continue
            blob = model_path.read_bytes()
            rebuilt, _ = _render(expected_meta, expected_tensors)
            if rebuilt != blob:
                stale.append(f"{spec['slug']}: fixture is out of date")
            cfg = oracle.load_model(model_path)
            prefill = oracle.prefill(cfg, PROMPT)
            greedy = oracle.greedy(cfg, PROMPT, 8)
            if render_golden(spec, prefill, greedy) != golden_path(spec["slug"]).read_text():
                stale.append(f"{spec['slug']}: golden is out of date")
        if stale:
            for item in stale:
                print(item)
            return 1
        print("architecture fixtures are up to date")
        return 0

    for spec in SPECS:
        report = build(spec)
        print(json.dumps(report))
    print(json.dumps(build_ternary_fixture()))
    return 0


def _render(metadata: dict[str, tuple[Any, ...]], tensors: list[TensorSpec]) -> tuple[bytes, Any]:
    """Serialise without touching disk, for the --check comparison."""
    from gguf_format import GGUFWriter

    writer = GGUFWriter(metadata, tensors)
    return writer.build(), writer.layout


if __name__ == "__main__":
    raise SystemExit(main())
