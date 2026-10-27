#!/usr/bin/env python3
"""Independent scalar reference runtime for the FreeMoe tiny GGUF fixtures.

This is intentionally a separate implementation from the C++ engine. It reads
the GGUF file, implements the documented `freemoe-tiny-v1` arithmetic in plain
Python, and emits deterministic logits/token/routing fixtures. It is not a
production LLM runtime and does not support arbitrary GGUF checkpoints.
"""
from __future__ import annotations

import argparse
import array
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_format import fnv1a64, parse_gguf  # noqa: E402

BOS_ID = 256
EOS_ID = 257
VOCAB_SIZE = 258
N_GREEDY = 16


class OracleError(RuntimeError):
    pass


def f32_values(data: bytes) -> list[float]:
    if len(data) % 4:
        raise OracleError("F32 tensor is not 4-byte aligned")
    if sys.byteorder != "little":
        arr = array.array("f")
        arr.frombytes(data)
        return list(arr)
    # struct.unpack produces native Python doubles from the exact float32 bytes.
    return list(struct.unpack(f"<{len(data)//4}f", data))


def load_model(path: Path) -> dict[str, Any]:
    doc = parse_gguf(path.read_bytes())
    md = doc["metadata"]
    required = [
        "general.architecture", "freemoe.adapter", "freemoe.model_kind",
        "freemoe.hidden_size", "freemoe.intermediate_size",
        "freemoe.num_hidden_layers", "freemoe.num_attention_heads",
        "freemoe.num_key_value_heads", "freemoe.head_dim",
        "freemoe.vocab_size", "freemoe.max_context_length",
        "freemoe.rms_norm_epsilon", "freemoe.rope_theta",
    ]
    for key in required:
        if key not in md:
            raise OracleError(f"missing metadata {key}")
    if md["general.architecture"] != "freemoe-tiny" or md["freemoe.adapter"] != "freemoe-tiny-v1":
        raise OracleError("unsupported architecture")
    if md["tokenizer.ggml.model"] != "freemoe.byte-v1":
        raise OracleError("unsupported tokenizer")
    tensors: dict[str, tuple[list[int], list[float]]] = {}
    for t in doc["tensors"]:
        if t["type"] != 0:
            raise OracleError("oracle supports F32 only")
        tensors[t["name"]] = (list(t["dims"]), f32_values(t["data"]))
    cfg = {
        "path": path,
        "metadata": md,
        "tensors": tensors,
        "H": int(md["freemoe.hidden_size"]),
        "I": int(md["freemoe.intermediate_size"]),
        "L": int(md["freemoe.num_hidden_layers"]),
        "heads": int(md["freemoe.num_attention_heads"]),
        "kv_heads": int(md["freemoe.num_key_value_heads"]),
        "hd": int(md["freemoe.head_dim"]),
        "V": int(md["freemoe.vocab_size"]),
        "ctx": int(md["freemoe.max_context_length"]),
        "eps": float(md["freemoe.rms_norm_epsilon"]),
        "theta": float(md["freemoe.rope_theta"]),
        "kind": str(md["freemoe.model_kind"]),
        "E": int(md.get("freemoe.expert_count", 0)),
        "K": int(md.get("freemoe.expert_top_k", 0)),
        "doc": doc,
    }
    if cfg["heads"] * cfg["hd"] != cfg["H"] or cfg["kv_heads"] != cfg["heads"]:
        raise OracleError("only MHA with heads*head_dim=hidden is supported")
    if cfg["hd"] % 2:
        raise OracleError("head_dim must be even")
    if cfg["V"] != VOCAB_SIZE:
        raise OracleError("fixture vocabulary mismatch")
    return cfg


def matvec(w: list[float], x: list[float], out_dim: int, in_dim: int) -> list[float]:
    y = [0.0] * out_dim
    for o in range(out_dim):
        base = o * in_dim
        acc = 0.0
        for k in range(in_dim):
            acc += w[base + k] * x[k]
        y[o] = acc
    return y


def rmsnorm(x: list[float], w: list[float], eps: float) -> list[float]:
    n = len(x)
    ss = 0.0
    for v in x:
        ss += v * v
    scale = 1.0 / math.sqrt(ss / n + eps)
    return [x[i] * scale * w[i] for i in range(n)]


def rope_half(q: list[float], k: list[float], pos: int, heads: int, hd: int, theta_base: float) -> tuple[list[float], list[float]]:
    q = list(q)
    k = list(k)
    half = hd // 2
    for h in range(heads):
        base = h * hd
        for i in range(half):
            inv = theta_base ** (-(2.0 * i) / hd)
            angle = pos * inv
            c = math.cos(angle)
            s = math.sin(angle)
            for vec, b in ((q, base), (k, base)):
                a0 = vec[b + 2 * i]
                a1 = vec[b + 2 * i + 1]
                vec[b + 2 * i] = a0 * c - a1 * s
                vec[b + 2 * i + 1] = a0 * s + a1 * c
    return q, k


def softmax(values: list[float]) -> list[float]:
    m = max(values)
    exps = [math.exp(v - m) for v in values]
    total = sum(exps)
    return [v / total for v in exps]


def attention(cfg: dict[str, Any], layer: int, q: list[float], k: list[float], v: list[float], cache: list[tuple[list[float], list[float]]], pos: int) -> list[float]:
    heads, hd = cfg["heads"], cfg["hd"]
    q, k = rope_half(q, k, pos, heads, hd, cfg["theta"])
    cache.append((k, v))
    scale = 1.0 / math.sqrt(float(hd))
    out: list[float] = []
    for h in range(heads):
        qb = h * hd
        scores = [0.0] * len(cache)
        for t, (kt, _) in enumerate(cache):
            kb = h * hd
            dot = 0.0
            for j in range(hd):
                dot += q[qb + j] * kt[kb + j]
            scores[t] = dot * scale
        probs = softmax(scores)
        acc = [0.0] * hd
        for t, (_, vt) in enumerate(cache):
            p = probs[t]
            vb = h * hd
            for j in range(hd):
                acc[j] += p * vt[vb + j]
        out.extend(acc)
    return out


def silu(x: float) -> float:
    return x / (1.0 + math.exp(-x))


def expert_output(cfg: dict[str, Any], layer: int, expert: int, x: list[float]) -> list[float]:
    H, I = cfg["H"], cfg["I"]
    gate = cfg["tensors"][f"blk.{layer}.ffn_gate_exps.weight"][1]
    up = cfg["tensors"][f"blk.{layer}.ffn_up_exps.weight"][1]
    down = cfg["tensors"][f"blk.{layer}.ffn_down_exps.weight"][1]
    gate_slice = gate[expert * I * H:(expert + 1) * I * H]
    up_slice = up[expert * I * H:(expert + 1) * I * H]
    down_slice = down[expert * H * I:(expert + 1) * H * I]
    if len(gate_slice) != I * H or len(up_slice) != I * H or len(down_slice) != H * I:
        raise OracleError("bad expert slice")
    g = matvec(gate_slice, x, I, H)
    u = matvec(up_slice, x, I, H)
    act = [silu(g[i]) * u[i] for i in range(I)]
    return matvec(down_slice, act, H, I)


def router_topk(cfg: dict[str, Any], layer: int, x: list[float]) -> tuple[list[int], list[float], list[float]]:
    E, K = cfg["E"], cfg["K"]
    router = cfg["tensors"][f"blk.{layer}.ffn_router.weight"][1]
    scores = matvec(router, x, E, cfg["H"])
    ids = sorted(range(E), key=lambda e: (-scores[e], e))[:K]
    selected = [scores[e] for e in ids]
    weights = softmax(selected)
    return ids, weights, scores


class Session:
    def __init__(self, cfg: dict[str, Any]):
        self.cfg = cfg
        self.caches = [[] for _ in range(cfg["L"])]
        self.pos = 0
        self.routing: list[tuple[int, int, list[int]]] = []
        self.selected_count = 0
        self.executed_count = 0

    def step(self, token: int) -> list[float]:
        cfg = self.cfg
        if token < 0 or token >= cfg["V"]:
            raise OracleError("token out of range")
        if self.pos >= cfg["ctx"]:
            raise OracleError("context exhausted")
        emb = cfg["tensors"]["token_embd.weight"][1]
        x = list(emb[token * cfg["H"]:(token + 1) * cfg["H"]])
        for layer in range(cfg["L"]):
            H, I = cfg["H"], cfg["I"]
            h = rmsnorm(x, cfg["tensors"][f"blk.{layer}.attn_norm.weight"][1], cfg["eps"])
            q = matvec(cfg["tensors"][f"blk.{layer}.attn_q.weight"][1], h, H, H)
            k = matvec(cfg["tensors"][f"blk.{layer}.attn_k.weight"][1], h, H, H)
            v = matvec(cfg["tensors"][f"blk.{layer}.attn_v.weight"][1], h, H, H)
            a = attention(cfg, layer, q, k, v, self.caches[layer], self.pos)
            ao = matvec(cfg["tensors"][f"blk.{layer}.attn_output.weight"][1], a, H, H)
            x = [x[i] + ao[i] for i in range(H)]
            h = rmsnorm(x, cfg["tensors"][f"blk.{layer}.ffn_norm.weight"][1], cfg["eps"])
            if cfg["kind"] == "dense":
                gate = matvec(cfg["tensors"][f"blk.{layer}.ffn_gate.weight"][1], h, I, H)
                up = matvec(cfg["tensors"][f"blk.{layer}.ffn_up.weight"][1], h, I, H)
                act = [silu(gate[i]) * up[i] for i in range(I)]
                ffn = matvec(cfg["tensors"][f"blk.{layer}.ffn_down.weight"][1], act, H, I)
            elif cfg["kind"] == "moe":
                ids, weights, _ = router_topk(cfg, layer, h)
                self.routing.append((self.pos, layer, ids))
                self.selected_count += len(ids)
                ffn = [0.0] * H
                for expert, weight in zip(ids, weights):
                    y = expert_output(cfg, layer, expert, h)
                    for i in range(H):
                        ffn[i] += weight * y[i]
                    self.executed_count += 1
                if self.selected_count != self.executed_count:
                    raise OracleError("selected/executed expert mismatch")
            else:
                raise OracleError("unknown model kind")
            x = [x[i] + ffn[i] for i in range(H)]
        x = rmsnorm(x, cfg["tensors"]["output_norm.weight"][1], cfg["eps"])
        logits = matvec(cfg["tensors"]["output.weight"][1], x, cfg["V"], cfg["H"])
        self.pos += 1
        return logits


def greedy(logits: list[float]) -> int:
    best = 0
    for i in range(1, len(logits)):
        if logits[i] > logits[best]:
            best = i
    return best


def tokenize(prompt: str) -> list[int]:
    raw = prompt.encode("utf-8")
    if len(raw) < 8 or len(raw) > 512:
        raise OracleError("fixture prompt must be 8..512 bytes")
    return [BOS_ID] + list(raw)


def run_prompt(cfg: dict[str, Any], prompt: str, n_greedy: int = N_GREEDY) -> dict[str, Any]:
    session = Session(cfg)
    ids = tokenize(prompt)
    logits: list[float] = []
    for token in ids:
        logits = session.step(token)
    prefill_logits = list(logits)
    generated: list[int] = []
    for _ in range(n_greedy):
        token = greedy(logits)
        generated.append(token)
        logits = session.step(token)
    return {
        "input_ids": ids,
        "prefill_logits": prefill_logits,
        "logits": logits,
        "generated_ids": generated,
        "routing": session.routing,
        "selected_count": session.selected_count,
        "executed_count": session.executed_count,
        "positions": session.pos,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--tokens", type=int, default=N_GREEDY)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()
    cfg = load_model(Path(args.model))
    result = run_prompt(cfg, args.prompt, args.tokens)
    if args.json:
        out = dict(result)
        out["model_kind"] = cfg["kind"]
        out["expert_count"] = cfg["E"]
        out["expert_top_k"] = cfg["K"]
        print(json.dumps(out, separators=(",", ":")))
    else:
        print("logits=" + " ".join(f"{x:.9g}" for x in result["logits"]))
        print("generated=" + " ".join(map(str, result["generated_ids"])))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
