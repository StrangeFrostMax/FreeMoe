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


def tensor(name: str, dims: list[int], scale: float, *, norm: bool = False) -> TensorSpec:
    count = 1
    for d in dims:
        count *= d
    return TensorSpec(name, list(dims), f32_bytes(values_for(name, count, scale, norm=norm)), 0)


def tokenizer_tokens() -> list[str]:
    tokens = [f"<{i}>" for i in range(256)]
    tokens.append("<s>")
    tokens.append("</s>")
    return tokens


def arch_tensors(spec: dict[str, Any]) -> list[TensorSpec]:
    h, i, layers = spec["hidden"], spec["intermediate"], spec["layers"]
    heads, hd = spec["heads"], spec["head_dim"]
    kv_heads = spec["kv_heads"]
    kv_dim = kv_heads * hd
    out = [tensor("token_embd.weight", [h, V], 0.35)]
    for layer in range(layers):
        p = f"blk.{layer}."
        out += [tensor(p + "attn_norm.weight", [h], 0.0, norm=True)]
        if spec.get("layernorm"):
            out += [tensor(p + "attn_norm.bias", [h], 0.05)]
        if spec.get("parallel_residual"):
            # No ffn_norm at all: the FFN consumes the attention norm.
            pass
        if spec.get("fused_qkv"):
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
        out += [tensor(p + "attn_output.weight", [h, h], 0.16)]
        if spec.get("attn_output_bias"):
            out += [tensor(p + "attn_output.bias", [h], 0.05)]
        if spec.get("post_norms"):
            out += [tensor(p + "attn_post_norm.weight", [h], 0.0, norm=True)]
        if not spec.get("parallel_residual"):
            out += [tensor(p + "ffn_norm.weight", [h], 0.0, norm=True)]
            if spec.get("layernorm"):
                out += [tensor(p + "ffn_norm.bias", [h], 0.05)]
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
        elif spec.get("experts"):
            # Stacked experts: one tensor holds every expert's slice, and the
            # router projects hidden -> expert_count. Routed experts may be
            # narrower than the dense FFN, so the width is its own setting.
            experts = spec["experts"]
            ew = spec.get("expert_width", i)
            out += [
                tensor(p + "ffn_gate_inp.weight", [h, experts], 0.12),
                tensor(p + "ffn_gate_exps.weight", [h, ew, experts], 0.16),
                tensor(p + "ffn_up_exps.weight", [h, ew, experts], 0.16),
                tensor(p + "ffn_down_exps.weight", [ew, h, experts], 0.16),
            ]
            if spec.get("shared_expert"):
                # A dense FFN beside the routed experts, plus the one-vector
                # projection that produces its per-token scalar gate.
                sw = spec.get("shared_width", i)
                out += [
                    tensor(p + "ffn_gate_inp_shexp.weight", [h], 0.30),
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
        else:
            out += [
                tensor(p + "ffn_gate.weight", [h, i], 0.16),
                tensor(p + "ffn_up.weight", [h, i], 0.16),
                tensor(p + "ffn_down.weight", [i, h], 0.16),
            ]
        if spec.get("post_norms"):
            out += [tensor(p + "ffn_post_norm.weight", [h], 0.0, norm=True)]
    out += [tensor("output_norm.weight", [h], 0.0, norm=True)]
    if spec.get("layernorm"):
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
        f"{arch}.feed_forward_length": ("u32", spec["intermediate"]),
        f"{arch}.attention.head_count": ("u32", spec["heads"]),
        f"{arch}.attention.head_count_kv": ("u32", spec["kv_heads"]),
        f"{arch}.attention.key_length": ("u32", spec["head_dim"]),
        f"{arch}.attention.value_length": ("u32", spec["head_dim"]),
        # LayerNorm files declare a different epsilon key, and that key is what
        # the adapter uses as evidence for the norm type.
        (f"{arch}.attention.layer_norm_epsilon" if spec.get("layernorm")
         else f"{arch}.attention.layer_norm_rms_epsilon"): ("f32", 1e-5),
        f"{arch}.rope.dimension_count": ("u32", spec["rope_dim"]),
        f"{arch}.rope.freq_base": ("f32", 10000.0),
        f"{arch}.context_length": ("u32", spec["context"]),
        f"{arch}.vocab_size": ("u32", V),
        "tokenizer.ggml.model": ("string", "freemoe.byte-v1"),
        "tokenizer.ggml.bos_token_id": ("u32", BOS),
        "tokenizer.ggml.eos_token_id": ("u32", EOS),
        "tokenizer.ggml.tokens": ("array_string", tokenizer_tokens()),
    }
    if spec.get("experts"):
        md[f"{arch}.expert_count"] = ("u32", spec["experts"])
        md[f"{arch}.expert_used_count"] = ("u32", spec.get("expert_used", 2))
    if spec.get("qkv_bias"):
        md[f"{arch}.attention.qkv_bias"] = ("bool", True)
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
    if spec.get("attn_softcap"):
        md[f"{arch}.attn_logit_softcapping"] = ("f32", spec["attn_softcap"])
    if spec.get("final_softcap"):
        md[f"{arch}.final_logit_softcapping"] = ("f32", spec["final_softcap"])
    if spec.get("sliding_window"):
        md[f"{arch}.attention.sliding_window"] = ("u32", spec["sliding_window"])
    return md


SPECS: list[dict[str, Any]] = [
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
        # Grouped-query attention with an exact head ratio, no bias. This is the
        # shape most current small checkpoints use.
        "slug": "llama-gqa",
        "arch": "llama",
        "name": "FreeMoe synthetic llama with grouped-query attention",
        "kv_heads": 2,
        "rope_dim": SHAPE["head_dim"],
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
    resolved.update({k: spec[k] for k in ("arch", "kv_heads", "rope_dim", "name")})
    resolved.setdefault("qkv_bias", False)
    resolved["kv_heads"] = spec.get("kv_heads", resolved["heads"])

    model_path = fixture_path(spec["slug"])
    corpus = {**SHAPE, **spec}
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
