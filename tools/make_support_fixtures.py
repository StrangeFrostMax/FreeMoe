#!/usr/bin/env python3
"""Generate fixtures that look like real published checkpoints.

The conformance fixtures model FreeMoe's own adapter. These fixtures model what
everybody else ships: `general.architecture = llama`, architecture-prefixed
hyperparameters, standard llama.cpp tensor names, grouped-query attention, a
GPT-2 style BPE tokenizer, and block-quantised tensors.

They exist so `freemoe probe` is tested against realistic shapes instead of only
against the toy adapter, and so the support matrix in `docs/MODEL_SUPPORT.md`
has a reproducible example of each blocker.

Everything is synthetic and MIT; no third-party weights are involved.

Usage: python3 tools/make_support_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_format import GGUFWriter, TensorSpec  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MANIFEST = FIXTURES / "support_fixtures.json"

Q4_0 = 2
Q8_0 = 8
ALIGN = 32

H = 64
I = 128
LAYERS = 1
HEADS = 8
KV_HEADS = 2
HEAD_DIM = 8
VOCAB = 64


def f32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def deterministic(name: str, count: int, scale: float = 0.05) -> bytes:
    digest = hashlib.sha256(f"support:{name}".encode()).digest()
    out = []
    state = int.from_bytes(digest[:8], "little")
    for _ in range(count):
        state = (state * 6364136223846793005 + 1442695040888963407) & 0xFFFFFFFFFFFFFFFF
        out.append(((state >> 33) / 2**31 - 1.0) * scale)
    return f32_bytes(out)


def quantised_bytes(name: str, count: int, block_bytes: int) -> bytes:
    """Block payload with finite scales; contents are irrelevant, layout is not."""
    raw = deterministic(name, count)
    values = struct.unpack(f"<{count}f", raw)
    payload = bytearray()
    block = 32
    for start in range(0, count, block):
        chunk = values[start : start + block]
        amax = max(abs(v) for v in chunk) or 1e-3
        scale = amax / 127.0
        payload += struct.pack("<e", scale)
        payload += struct.pack(f"<{block}b",
                              *[max(-127, min(127, int(v / scale))) for v in chunk])
    assert len(payload) == (count // block) * block_bytes
    return bytes(payload)


def llama_tensors(quantised: bool) -> list[TensorSpec]:
    type_id = Q8_0 if quantised else 0
    spec: list[TensorSpec] = []

    def matrix(name: str, rows: int, cols: int) -> None:
        if quantised:
            spec.append(TensorSpec(name, [cols, rows],
                                   quantised_bytes(name, rows * cols, 34), type_id))
        else:
            spec.append(TensorSpec(name, [cols, rows], deterministic(name, rows * cols), 0))

    def vector(name: str, count: int) -> None:
        spec.append(TensorSpec(name, [count], deterministic(name, count, 0.02), 0))

    matrix("token_embd.weight", VOCAB, H)
    matrix("output.weight", VOCAB, H)
    vector("output_norm.weight", H)
    kv_dim = KV_HEADS * HEAD_DIM
    for layer in range(LAYERS):
        prefix = f"blk.{layer}."
        vector(prefix + "attn_norm.weight", H)
        vector(prefix + "ffn_norm.weight", H)
        matrix(prefix + "attn_q.weight", HEADS * HEAD_DIM, H)
        matrix(prefix + "attn_k.weight", kv_dim, H)
        matrix(prefix + "attn_v.weight", kv_dim, H)
        matrix(prefix + "attn_output.weight", H, HEADS * HEAD_DIM)
        matrix(prefix + "ffn_gate.weight", I, H)
        matrix(prefix + "ffn_up.weight", I, H)
        matrix(prefix + "ffn_down.weight", H, I)
    return spec


def llama_metadata(tensors: list[TensorSpec], *, name: str, quantised: bool,
                   tokenizer: str) -> dict:
    meta = {
        "general.architecture": ("string", "llama"),
        "general.name": ("string", name),
        "general.alignment": ("u32", ALIGN),
        "general.file_type": ("u32", Q8_0 if quantised else 0),
        "llama.block_count": ("u32", LAYERS),
        "llama.embedding_length": ("u32", H),
        "llama.feed_forward_length": ("u32", I),
        "llama.attention.head_count": ("u32", HEADS),
        "llama.attention.head_count_kv": ("u32", KV_HEADS),
        "llama.attention.key_length": ("u32", HEAD_DIM),
        "llama.attention.layer_norm_rms_epsilon": ("f32", 1e-5),
        "llama.rope.dimension_count": ("u32", HEAD_DIM),
        "llama.rope.freq_base": ("f32", 10000.0),
        "llama.context_length": ("u32", 512),
        "llama.vocab_size": ("u32", VOCAB),
        "tokenizer.ggml.model": ("string", tokenizer),
        "tokenizer.ggml.bos_token_id": ("u32", 1),
        "tokenizer.ggml.eos_token_id": ("u32", 2),
    }
    if tokenizer == "gpt2":
        tokens = ["<|bos|>", "<|eos|>"] + [f"tok{i}" for i in range(VOCAB - 2)]
        merges = [f"tok{i} tok{i + 1}" for i in range(1, VOCAB - 2)]
        meta["tokenizer.ggml.tokens"] = ("array_string", tokens)
        meta["tokenizer.ggml.merges"] = ("array_string", merges)
        meta["tokenizer.chat_template"] = ("string", "{{ messages }}")
    return meta


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    outputs = {}

    # 1. A realistic llama checkpoint: GQA + BPE tokenizer + Q8_0 tensors.
    tensors = llama_tensors(quantised=True)
    path = FIXTURES / "llama-shape-q8_0.gguf"
    path.write_bytes(GGUFWriter(
        llama_metadata(tensors, name="Synthetic llama-shaped Q8_0 (GQA, BPE)",
                       quantised=True, tokenizer="gpt2"), tensors).build())
    outputs[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "bytes": path.stat().st_size,
                          "purpose": "GQA + BPE tokenizer + Q8_0 blockers"}

    # 2. The same shape in F32 with the byte tokenizer: every blocker that is
    #    about the architecture rather than the format, and nothing else.
    tensors = llama_tensors(quantised=False)
    path = FIXTURES / "llama-shape-f32.gguf"
    path.write_bytes(GGUFWriter(
        llama_metadata(tensors, name="Synthetic llama-shaped F32 (GQA)",
                       quantised=False, tokenizer="gpt2"), tensors).build())
    outputs[path.name] = {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                          "bytes": path.stat().st_size,
                          "purpose": "architecture-only blocker set"}

    manifest = {
        "schema": "freemoe.support-fixtures.v1",
        "generator": "tools/make_support_fixtures.py",
        "synthetic": True,
        "outputs": outputs,
        "note": ("These files are shaped like published llama.cpp checkpoints "
                 "(standard keys, standard tensor names, GQA, BPE tokenizer). They "
                 "exist to test `freemoe probe` and are not runnable models."),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "written", "files": list(outputs)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
