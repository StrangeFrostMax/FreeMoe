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
        tensors[t["name"]] = (list(t["dims"]), _f32_values(t["data"]))

    hidden = int(key("embedding_length"))
    heads = int(key("attention.head_count"))
    hd = int(key("attention.key_length", hidden // heads))
    cfg = {
        "arch": arch,
        "H": hidden,
        "I": int(key("feed_forward_length")),
        "L": int(key("block_count")),
        "heads": heads,
        "kv_heads": int(key("attention.head_count_kv", heads)),
        "hd": hd,
        "V": int(key("vocab_size", len(md.get("tokenizer.ggml.tokens", [])))),
        "ctx": int(key("context_length")),
        "eps": float(key("attention.layer_norm_rms_epsilon", 1e-5)),
        "theta": float(key("rope.freq_base", 10000.0)),
        "rope_dim": int(key("rope.dimension_count", hd)),
        "qkv_bias": bool(key("attention.qkv_bias", False)),
        # The prompt is tokenized with the checkpoint's own BOS id, exactly as
        # the engine does; hard-coding 1 would hide a real disagreement.
        "bos": int(md.get("tokenizer.ggml.bos_token_id", 1)),
        # Presence of the per-head q/k scales is the evidence qwen3 uses them.
        "qk_norm": "blk.0.attn_q_norm.weight" in tensors,
        # phi3 fuses q/k/v into one projection (and one optional bias).
        "fused_qkv": "blk.0.attn_qkv.weight" in tensors,
        # gemma family: scaled embeddings, zero-centred norm scales, GeGLU.
        "embedding_scale": arch in ("gemma", "gemma2", "gemma3"),
        "norm_add_one": arch in ("gemma", "gemma2", "gemma3"),
        "activation": "gelu" if arch in ("gemma", "gemma2", "gemma3") else "silu",
        "post_norms": arch in ("gemma2", "gemma3"),
        "attn_softcap": float(key("attn_logit_softcapping", 0.0)),
        "final_softcap": float(key("final_logit_softcapping", 0.0)),
        "window": int(key("attention.sliding_window", 0)),
        "tensors": tensors,
    }
    if cfg["heads"] * cfg["hd"] != cfg["H"]:
        raise ReferenceError("heads * head_dim != hidden")
    if cfg["kv_heads"] > cfg["heads"] or cfg["heads"] % cfg["kv_heads"]:
        raise ReferenceError("inconsistent grouped-query head counts")
    if cfg["rope_dim"] % 2 or cfg["rope_dim"] > cfg["hd"]:
        raise ReferenceError("invalid rotary dimension")
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


def rmsnorm(x: list[float], w: list[float], eps: float, add_one: bool = False) -> list[float]:
    mean_sq = sum(v * v for v in x) / len(x)
    scale = 1.0 / math.sqrt(mean_sq + eps)
    bias = 1.0 if add_one else 0.0
    return [x[i] * scale * (bias + w[i]) for i in range(len(x))]


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


def rope_half(vec: list[float], head_dim: int, heads: int, pos: int,
              rope_dim: int, theta: float) -> None:
    """Rotates each head's vector in place over `rope_dim` dimensions.

    This is ggml's default (`GGML_ROPE_TYPE_NORM`) layout, the one llama.cpp
    uses for llama-family checkpoints: the rotating pairs are *adjacent*
    elements (2i, 2i+1). The `NEOX` layout pairs (i, i + rope_dim/2) instead and
    belongs to different architectures; using it here silently rotates the wrong
    coordinates, which looks like a plausible model that talks nonsense.

    Verified against a real checkpoint: on tinyllama-15M the adjacent-pair
    layout produces fluent text and the half-pair layout produces word salad.
    """
    for head in range(heads):
        base = head * head_dim
        for j in range(rope_dim // 2):
            freq = 1.0 / (theta ** (2.0 * j / rope_dim))
            angle = pos * freq
            c, s = math.cos(angle), math.sin(angle)
            a = vec[base + 2 * j]
            b = vec[base + 2 * j + 1]
            vec[base + 2 * j] = a * c - b * s
            vec[base + 2 * j + 1] = a * s + b * c


def softmax(values: list[float]) -> list[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


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
        self.tensors = cfg["tensors"]

    def tensor(self, name: str) -> list[float]:
        if name not in self.tensors:
            raise ReferenceError(f"missing tensor {name}")
        return self.tensors[name][1]

    def step(self, token: int) -> list[float]:
        cfg = self.cfg
        H, hd, heads, kv_heads = cfg["H"], cfg["hd"], cfg["heads"], cfg["kv_heads"]
        kv_dim = kv_heads * hd
        group = heads // kv_heads
        embd = self.tensor("token_embd.weight")
        x = [embd[token * H + i] for i in range(H)]
        if cfg["embedding_scale"]:
            x = [v * math.sqrt(H) for v in x]

        for layer in range(cfg["L"]):
            p = f"blk.{layer}."
            h = rmsnorm(x, self.tensor(p + "attn_norm.weight"), cfg["eps"],
                        cfg["norm_add_one"])
            if cfg["fused_qkv"]:
                fused = matvec(self.tensor(p + "attn_qkv.weight"), h, H + 2 * kv_dim, H)
                if cfg["qkv_bias"]:
                    bias = self.tensor(p + "attn_qkv.bias")
                    fused = [fused[i] + bias[i] for i in range(H + 2 * kv_dim)]
                q = fused[:H]
                k = fused[H:H + kv_dim]
                v = fused[H + kv_dim:]
            else:
                q = matvec(self.tensor(p + "attn_q.weight"), h, H, H)
                k = matvec(self.tensor(p + "attn_k.weight"), h, kv_dim, H)
                v = matvec(self.tensor(p + "attn_v.weight"), h, kv_dim, H)
                if cfg["qkv_bias"]:
                    bq = self.tensor(p + "attn_q.bias")
                    bk = self.tensor(p + "attn_k.bias")
                    bv = self.tensor(p + "attn_v.bias")
                    q = [q[i] + bq[i] for i in range(H)]
                    k = [k[i] + bk[i] for i in range(kv_dim)]
                    v = [v[i] + bv[i] for i in range(kv_dim)]
            if cfg["qk_norm"]:
                q = head_rmsnorm(q, heads, self.tensor(p + "attn_q_norm.weight"), cfg["eps"])
                k = head_rmsnorm(k, kv_heads, self.tensor(p + "attn_k_norm.weight"), cfg["eps"])
            rope_half(q, hd, heads, self.pos, cfg["rope_dim"], cfg["theta"])
            rope_half(k, hd, kv_heads, self.pos, cfg["rope_dim"], cfg["theta"])
            self.cache[layer].append((k, v))

            attention = [0.0] * H
            scale = 1.0 / math.sqrt(hd)
            for head in range(heads):
                qb = head * hd
                kvb = (head // group) * hd
                cache = self.cache[layer]
                window = cfg["window"]
                first = 0 if (window == 0 or self.pos < window) else self.pos + 1 - window
                scores = []
                for t_index in range(first, len(cache)):
                    kt = cache[t_index][0]
                    dot = sum(q[qb + j] * kt[kvb + j] for j in range(hd)) * scale
                    if cfg["attn_softcap"] > 0.0:
                        dot = cfg["attn_softcap"] * math.tanh(dot / cfg["attn_softcap"])
                    scores.append(dot)
                probs = softmax(scores)
                acc = [0.0] * hd
                for offset, t_index in enumerate(range(first, len(cache))):
                    vt = cache[t_index][1]
                    p_i = probs[offset]
                    for j in range(hd):
                        acc[j] += p_i * vt[kvb + j]
                attention[qb:qb + hd] = acc
            attn_out = matvec(self.tensor(p + "attn_output.weight"), attention, H, H)
            if cfg["post_norms"]:
                attn_out = rmsnorm(attn_out, self.tensor(p + "attn_post_norm.weight"),
                                   cfg["eps"], cfg["norm_add_one"])
            x = [x[i] + attn_out[i] for i in range(H)]

            h = rmsnorm(x, self.tensor(p + "ffn_norm.weight"), cfg["eps"],
                        cfg["norm_add_one"])
            gate = matvec(self.tensor(p + "ffn_gate.weight"), h, cfg["I"], H)
            up = matvec(self.tensor(p + "ffn_up.weight"), h, cfg["I"], H)
            if cfg["activation"] == "gelu":
                act = [gelu(gate[i]) * up[i] for i in range(cfg["I"])]
            else:
                act = [gate[i] / (1.0 + math.exp(-gate[i])) * up[i] for i in range(cfg["I"])]
            down = matvec(self.tensor(p + "ffn_down.weight"), act, H, cfg["I"])
            if cfg["post_norms"]:
                down = rmsnorm(down, self.tensor(p + "ffn_post_norm.weight"),
                               cfg["eps"], cfg["norm_add_one"])
            x = [x[i] + down[i] for i in range(H)]

        final = rmsnorm(x, self.tensor("output_norm.weight"), cfg["eps"],
                        cfg["norm_add_one"])
        out_name = "output.weight" if "output.weight" in self.tensors else "token_embd.weight"
        self.pos += 1
        logits = matvec(self.tensor(out_name), final, cfg["V"], H)
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
        if token == 2:  # EOS
            break
        logits = session.step(token)
    return {"input_ids": ids, "generated_ids": generated, "prefill_logits": logits}


def write_golden(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1) + "\n")


if __name__ == "__main__":
    model = load_model(Path(sys.argv[1]))
    print(json.dumps(greedy(model, "the", 4), indent=1)[:400])
