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
        "eps": float(key("attention.layer_norm_epsilon", 1e-5)
                     if f"{arch}.attention.layer_norm_epsilon" in md
                     else key("attention.layer_norm_rms_epsilon", 1e-5)),
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
        # glm4 keeps gate and up in one tensor: gate first, then up. The width is
        # the evidence -- phi2 also has no gate, but its up is [hidden, I].
        "fused_gate_up": ("blk.0.ffn_gate.weight" not in tensors
                          and "blk.0.ffn_up.weight" in tensors
                          and "blk.0.ffn_gate_exps.weight" not in tensors
                          and tensors["blk.0.ffn_up.weight"][0][1] == 2 * int(key("feed_forward_length"))),
        # phi2's block: LayerNorm, biases on the norms/FFN/logits, a gate-free
        # GELU feed-forward fed by the attention norm, and one shared residual.
        # Every one of these is read from the tensors or the epsilon key.
        "layer_norm": f"{arch}.attention.layer_norm_epsilon" in md,
        "norm_bias": "blk.0.attn_norm.bias" in tensors,
        "parallel_residual": "blk.0.ffn_norm.weight" not in tensors,
        "ffn_bias": "blk.0.ffn_up.bias" in tensors,
        "attn_output_bias": "blk.0.attn_output.bias" in tensors,
        "logit_bias": "output.bias" in tensors,
        # gemma family: scaled embeddings, zero-centred norm scales, GeGLU.
        # A multiplier, not a flag: gemma uses sqrt(hidden), minicpm 12.0.
        "embedding_scale": (math.sqrt(hidden) if arch in ("gemma", "gemma2", "gemma3")
                            else float(key("embedding_scale", 12.0))
                            if arch == "minicpm" else 1.0),
        "norm_add_one": arch in ("gemma", "gemma2", "gemma3"),
        "activation": "gelu" if arch in ("gemma", "gemma2", "gemma3") else "silu",
        # Detected from the tensors, exactly as the engine's adapter does:
        # keying this off the architecture name silently skipped glm4's norms.
        "post_norms": "blk.0.attn_post_norm.weight" in tensors,
        "attn_softcap": float(key("attn_logit_softcapping", 0.0)),
        "final_softcap": float(key("final_logit_softcapping", 0.0)),
        "window": int(key("attention.sliding_window", 0)),
        "tensors": tensors,
        # Routing semantics straight from the checkpoint. Renormalising over the
        # selected experts is only equivalent to a full softmax when every expert
        # is selected, so it cannot be assumed.
        "experts": int(key("expert_count", 0)),
        "top_k": int(key("expert_used_count", 0)),
        # Routed experts and the shared expert can each be narrower than the
        # dense FFN, so the three widths are separate.
        "I_exp": int(key("expert_feed_forward_length",
                         key("feed_forward_length"))),
        "I_shexp": int(key("expert_shared_feed_forward_length",
                           key("feed_forward_length"))),
        # Evidence, not architecture name: the tensors are what the forward pass
        # reads, so their presence is what decides whether the branch runs.
        "shared_expert": "blk.0.ffn_gate_shexp.weight" in tensors,
        "expert_norm": bool(key("expert_weights_norm", False)),
        "expert_scale": float(key("expert_weights_scale", 1.0)),
    }
    # A block with up/down but no gate has no gated activation. GGUF cannot say
    # which activation, so phi2's GELU is applied and anything else is refused --
    # the same rule the engine's adapter enforces.
    cfg["ffn_no_gate"] = (cfg["experts"] == 0 and "blk.0.ffn_gate.weight" not in tensors
                          and "blk.0.ffn_up.weight" in tensors and not cfg["fused_gate_up"])
    if cfg["ffn_no_gate"]:
        if arch != "phi2":
            raise ReferenceError("gate-free feed-forward is only defined for phi2")
        cfg["activation"] = "gelu"
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


def expert_output(cfg: dict[str, Any], layer: int, expert: int,
                  x: list[float], prefix: str) -> list[float]:
    """One expert's SwiGLU/GeGLU output.

    The stacked tensors hold every expert back to back, so expert `e` occupies a
    contiguous slice. The slice geometry comes from the declared shapes, never
    from an assumption about float sizes.
    """
    H, I = cfg["H"], cfg["I_exp"]
    count = cfg["experts"]
    tensors = cfg["tensors"]
    out = []
    for name, out_dim, in_dim in (
            (prefix + "ffn_gate_exps.weight", I, H),
            (prefix + "ffn_up_exps.weight", I, H)):
        dims, data = tensors[name]
        per_expert = out_dim * in_dim
        base = expert * per_expert
        out.append(matvec(data[base:base + per_expert], x, out_dim, in_dim))
    gate_vals, up_vals = out
    if cfg["activation"] == "gelu":
        act = [gelu(gate_vals[i]) * up_vals[i] for i in range(I)]
    else:
        act = [gate_vals[i] / (1.0 + math.exp(-gate_vals[i])) * up_vals[i]
               for i in range(I)]
    dims, data = tensors[prefix + "ffn_down_exps.weight"]
    per_expert = H * I
    base = expert * per_expert
    return matvec(data[base:base + per_expert], act, H, I)


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
        if cfg["embedding_scale"] != 1.0:
            scale = cfg["embedding_scale"]
            x = [v * scale for v in x]

        def norm_of(vec: list[float], prefix: str) -> list[float]:
            """The one place both norm types are chosen, mirroring the engine."""
            w = self.tensor(prefix + ".weight")
            if cfg["layer_norm"]:
                return layernorm(vec, w, self.tensor(prefix + ".bias"), cfg["eps"])
            return rmsnorm(vec, w, cfg["eps"], cfg["norm_add_one"])

        for layer in range(cfg["L"]):
            p = f"blk.{layer}."
            h = norm_of(x, p + "attn_norm")
            attn_normed = list(h) if cfg["parallel_residual"] else None
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
            if cfg["attn_output_bias"]:
                bias = self.tensor(p + "attn_output.bias")
                attn_out = [attn_out[i] + bias[i] for i in range(H)]
            if cfg["post_norms"]:
                attn_out = rmsnorm(attn_out, self.tensor(p + "attn_post_norm.weight"),
                                   cfg["eps"], cfg["norm_add_one"])
            x = [x[i] + attn_out[i] for i in range(H)]

            h = attn_normed if cfg["parallel_residual"] else norm_of(x, p + "ffn_norm")
            if cfg["ffn_no_gate"]:
                # phi2: up -> gelu -> down, with a bias on both projections.
                up = matvec(self.tensor(p + "ffn_up.weight"), h, cfg["I"], H)
                if cfg["ffn_bias"]:
                    bias = self.tensor(p + "ffn_up.bias")
                    up = [up[i] + bias[i] for i in range(cfg["I"])]
                act = [gelu(v) for v in up]
                down = matvec(self.tensor(p + "ffn_down.weight"), act, H, cfg["I"])
                if cfg["ffn_bias"]:
                    bias = self.tensor(p + "ffn_down.bias")
                    down = [down[i] + bias[i] for i in range(H)]
            elif cfg["experts"]:
                # Router -> full softmax -> top-k -> combine. The selected experts
                # are renormalised only when the checkpoint asks for it.
                logits = matvec(self.tensor(p + "ffn_gate_inp.weight"), h,
                                cfg["experts"], H)
                probs = softmax(logits)
                order = sorted(range(cfg["experts"]), key=lambda e: (-probs[e], e))
                chosen = order[:cfg["top_k"]]
                weights = [probs[e] for e in chosen]
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
                if cfg["activation"] == "gelu":
                    act = [gelu(gate[i]) * up[i] for i in range(cfg["I"])]
                else:
                    act = [gate[i] / (1.0 + math.exp(-gate[i])) * up[i]
                           for i in range(cfg["I"])]
                down = matvec(self.tensor(p + "ffn_down.weight"), act, H, cfg["I"])
            if cfg["shared_expert"]:
                # qwen2moe's shared expert: a dense FFN beside the routed ones,
                # scaled by sigmoid(dot(gate_inp, h)). The reference writes that
                # gate as silu(s)/s, which is the same number except at s == 0
                # (0/0); the limit 0.5 is used there.
                s = matvec(self.tensor(p + "ffn_gate_inp_shexp.weight"), h, 1, H)[0]
                gate_scalar = (s / (1.0 + math.exp(-s)) / s) if s != 0.0 else 0.5
                W = cfg["I_shexp"]
                s_gate = matvec(self.tensor(p + "ffn_gate_shexp.weight"), h, W, H)
                s_up = matvec(self.tensor(p + "ffn_up_shexp.weight"), h, W, H)
                if cfg["activation"] == "gelu":
                    s_act = [gelu(s_gate[i]) * s_up[i] for i in range(W)]
                else:
                    s_act = [s_gate[i] / (1.0 + math.exp(-s_gate[i])) * s_up[i]
                             for i in range(W)]
                s_down = matvec(self.tensor(p + "ffn_down_shexp.weight"), s_act, H, W)
                down = [down[i] + gate_scalar * s_down[i] for i in range(H)]
            if cfg["post_norms"]:
                down = rmsnorm(down, self.tensor(p + "ffn_post_norm.weight"),
                               cfg["eps"], cfg["norm_add_one"])
            x = [x[i] + down[i] for i in range(H)]

        final = norm_of(x, "output_norm")
        out_name = "output.weight" if "output.weight" in self.tensors else "token_embd.weight"
        self.pos += 1
        logits = matvec(self.tensor(out_name), final, cfg["V"], H)
        if cfg["logit_bias"]:
            bias = self.tensor("output.bias")
            logits = [logits[i] + bias[i] for i in range(cfg["V"])]
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
