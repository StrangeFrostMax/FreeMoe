#!/usr/bin/env python3
"""Build a quantised conformance fixture pair with block-aligned dimensions.

The canonical `freemoe-tiny-v1` fixture is deliberately tiny (hidden 16), and
GGUF requires the innermost tensor dimension to be a whole number of
quantisation blocks. So this tool builds a separate, still synthetic, model
family with block-aligned dimensions and emits two files:

  tests/fixtures/quant-moe-q8_0.gguf    every matrix tensor stored as Q8_0
  tests/fixtures/quant-moe-f32ref.gguf  the same model in F32, holding exactly
                                        the values Q8_0 decoding must produce

Running both must agree on logits and greedy tokens. That proves the decode and
execution path end to end on real block-formatted data without shipping a
downloaded checkpoint.

Everything here is generated in-repo and MIT licensed; no third-party weights
are involved.

Usage: python3 tools/make_quant_fixtures.py
"""
from __future__ import annotations

import hashlib
import json
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_format import GGUFWriter, TensorSpec, fnv1a64  # noqa: E402
import quant_reference as qref  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
MANIFEST = FIXTURES / "quant_fixtures.json"

Q8_0 = 8
Q4_0 = 2
Q4_K = 12
BLOCK = 32
ALIGN = 32

# Variants: (label, gguf type id, block size, block bytes). Each variant is
# written as quant-moe-<label>.gguf plus its F32 reference twin.
VARIANTS = (
    ("q8_0", Q8_0, 32, 34),
    ("q4_0", Q4_0, 32, 18),
    ("q4_k", Q4_K, 256, 144),
)


SEED = 0xFEEE0E


class Cfg:
    """Synthetic MoE shape. `hidden` and `intermediate` must be multiples of the
    variant's block size, which is why the K-quant variant is wider."""

    def __init__(self, hidden, intermediate, layers, heads, head_dim, vocab, context, experts, top_k):
        self.hidden = hidden
        self.intermediate = intermediate
        self.layers = layers
        self.heads = heads
        self.head_dim = head_dim
        self.vocab = vocab
        self.context = context
        self.experts = experts
        self.top_k = top_k


# 32-value block formats (Q8_0, Q4_0) work at hidden 64; Q4_K needs 256-wide rows.
CFG_32 = Cfg(64, 128, 2, 4, 16, 258, 512, 4, 2)
CFG_256 = Cfg(256, 256, 1, 4, 64, 258, 256, 2, 1)


def f32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def encode_f16(value: float) -> bytes:
    bits = struct.unpack("<I", struct.pack("<f", value))[0]
    sign = (bits >> 16) & 0x8000
    exp = ((bits >> 23) & 0xFF) - 127 + 15
    man = bits & 0x7FFFFF
    if exp <= 0:
        return struct.pack("<H", sign)
    if exp >= 0x1F:
        return struct.pack("<H", sign | 0x7C00)
    return struct.pack("<H", sign | (exp << 10) | (man >> 13))


def deterministic(name: str, count: int, scale: float) -> list[float]:
    digest = hashlib.sha256(f"FreeMoe-quant:{SEED}:{name}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "little"))
    return [(rng.random() * 2.0 - 1.0) * scale for _ in range(count)]


def quantize_q8_0(values: list[float]) -> tuple[bytes, list[float]]:
    if len(values) % BLOCK:
        raise ValueError("Q8_0 requires a whole number of 32-value blocks")
    payload = bytearray()
    decoded: list[float] = []
    for start in range(0, len(values), BLOCK):
        block = values[start : start + BLOCK]
        amax = max(abs(v) for v in block) if block else 0.0
        scale = amax / 127.0 if amax > 0 else 0.0
        payload += encode_f16(scale)
        if scale == 0.0:
            payload += bytes(BLOCK)
            decoded += [0.0] * BLOCK
            continue
        quants = [max(-127, min(127, int(round(v / scale)))) for v in block]
        payload += struct.pack(f"<{BLOCK}b", *quants)
        # The block stores the scale as binary16, so the decoded values must use
        # the *rounded* scale. Using the exact scale here silently drifts the
        # reference by ~5e-4 relative, which the parity test would flag.
        stored = struct.unpack("<e", encode_f16(scale))[0]
        decoded += [stored * q for q in quants]
    return bytes(payload), decoded


def encode_q4_0(values: list[float]) -> bytes:
    """Q4_0: half d, 16 bytes of nibbles, values are (q - 8) * d."""
    payload = bytearray()
    for start in range(0, len(values), 32):
        block = values[start : start + 32]
        amax = max(abs(v) for v in block)
        d = amax / 8.0 if amax > 0 else 0.0
        payload += encode_f16(d)
        packed = bytearray()
        for j in range(16):
            lo = max(0, min(15, int(round(block[j] / d)) + 8)) if d else 8
            hi = max(0, min(15, int(round(block[16 + j] / d)) + 8)) if d else 8
            packed.append(lo | (hi << 4))
        payload += packed
    return bytes(payload)


def encode_q4_k_scales(sc: list[int], mn: list[int]) -> bytes:
    """Inverse of the reference 6-bit scale/min unpacking."""
    q = bytearray(12)
    for j in range(4):
        q[j] = (sc[j] & 63) | (((mn[j + 4] >> 4) & 3) << 6)
        q[j + 4] = (mn[j] & 63) | (((sc[j + 4] >> 4) & 3) << 6)
    for j in range(4):
        q[8 + j] = (sc[4 + j] & 0x0F) | ((mn[4 + j] & 0x0F) << 4)
    return bytes(q)


def encode_q4_k(values: list[float]) -> bytes:
    """Q4_K at fixed d/dmin: each 32-value sub-block gets its own 6-bit scale.

    The quantisation is deliberately simple (no min offset, uniform d). What
    matters for conformance is that the stored block round-trips exactly through
    an independent decoder.
    """
    payload = bytearray()
    for start in range(0, len(values), 256):
        block = values[start : start + 256]
        d = max(abs(v) for v in block) / (63 * 15) if any(block) else 0.0
        sc, mn = [], []
        quants: list[int] = []
        for sub in range(8):
            part = block[sub * 32 : (sub + 1) * 32]
            # 63 keeps the 6-bit scale at full range; the min stays zero.
            sc.append(63)
            mn.append(0)
            scale = d * 63
            quants += [
                max(0, min(15, int(round(v / scale)))) if scale else 0 for v in part
            ]
        payload += encode_f16(d) + encode_f16(0.0)
        payload += encode_q4_k_scales(sc, mn)
        for group in range(4):
            chunk = bytearray()
            for l in range(32):
                lo = quants[group * 64 + l]
                hi = quants[group * 64 + 32 + l]
                chunk.append(lo | (hi << 4))
            payload += chunk
    return bytes(payload)


ENCODERS = {
    "q8_0": lambda values: quantize_q8_0(values)[0],
    "q4_0": encode_q4_0,
    "q4_k": encode_q4_k,
}


def decode_reference(type_id: int, payload: bytes, block_bytes: int) -> list[float]:
    """Independent decode via the Python reference decoders.

    Cross-checking against Python rather than re-using the C++ decoder is what
    makes this a conformance test instead of a tautology.
    """
    decoder = {
        Q8_0: qref.q8_0,
        Q4_0: qref.q4_0,
        Q4_K: qref.q4_k,
    }[type_id]
    out: list[float] = []
    for start in range(0, len(payload), block_bytes):
        out += decoder(payload[start : start + block_bytes])
    return out


def product(dims: list[int]) -> int:
    out = 1
    for d in dims:
        out *= d
    return out


def layer_name(layer: int, suffix: str) -> str:
    return f"blk.{layer}.{suffix}"


def tokenizer_tokens(vocab: int) -> list[str]:
    tokens = [f"<0x{b:02X}>" for b in range(256)]
    tokens += ["<bos>", "<eos>"] + [f"<pad{i}>" for i in range(vocab - 258)]
    assert len(tokens) == vocab
    return tokens


def build_tensors(cfg: Cfg) -> list[tuple[str, list[int], list[float]]]:
    spec: list[tuple[str, list[int], list[float]]] = []
    H, V, I, E = cfg.hidden, cfg.vocab, cfg.intermediate, cfg.experts
    spec.append(("token_embd.weight", [H, V], deterministic("embd", H * V, 0.08)))
    spec.append(("output.weight", [H, V], deterministic("output", H * V, 0.08)))
    spec.append(("output_norm.weight", [H], [1.0 + x for x in deterministic("onorm", H, 0.05)]))
    for layer in range(cfg.layers):
        p = f"{layer}."
        spec.append((layer_name(layer, "attn_norm.weight"), [H],
                     [1.0 + x for x in deterministic(p + "anorm", H, 0.05)]))
        spec.append((layer_name(layer, "ffn_norm.weight"), [H],
                     [1.0 + x for x in deterministic(p + "fnorm", H, 0.05)]))
        for suffix, scale in (
            ("attn_q.weight", 0.08), ("attn_k.weight", 0.08), ("attn_v.weight", 0.08),
            ("attn_output.weight", 0.08),
        ):
            spec.append((layer_name(layer, suffix), [H, H],
                         deterministic(p + suffix, H * H, scale)))
        spec.append((layer_name(layer, "ffn_router.weight"), [H, E],
                     deterministic(p + "router", H * E, 0.1)))
        spec.append((layer_name(layer, "ffn_gate_exps.weight"), [H, I, E],
                     deterministic(p + "gate", H * I * E, 0.08)))
        spec.append((layer_name(layer, "ffn_up_exps.weight"), [H, I, E],
                     deterministic(p + "up", H * I * E, 0.08)))
        spec.append((layer_name(layer, "ffn_down_exps.weight"), [I, H, E],
                     deterministic(p + "down", I * H * E, 0.08)))
    return spec


def expert_checksums(tensors: list[TensorSpec], cfg: Cfg) -> list[int]:
    """Deterministic payload checksum per (layer, expert), matching the runtime."""
    by_name = {t.name: t for t in tensors}
    checksums: list[int] = []
    H, I, E = cfg.hidden, cfg.intermediate, cfg.experts
    for layer in range(cfg.layers):
        gate = by_name[layer_name(layer, "ffn_gate_exps.weight")].data
        up = by_name[layer_name(layer, "ffn_up_exps.weight")].data
        down = by_name[layer_name(layer, "ffn_down_exps.weight")].data
        for expert in range(E):
            payload = bytearray()
            for data, elements in ((gate, H * I), (up, H * I), (down, I * H)):
                stride = len(data) // E
                payload += data[expert * stride : (expert + 1) * stride]
            checksums.append(fnv1a64(bytes(payload)))
    return checksums


def metadata(tensors: list[TensorSpec], cfg: Cfg, *, name: str, file_type: int) -> dict:
    H, I, L, HEADS, HD, V, CTX, E, TOPK = (
        cfg.hidden, cfg.intermediate, cfg.layers, cfg.heads, cfg.head_dim,
        cfg.vocab, cfg.context, cfg.experts, cfg.top_k,
    )
    return {
        "general.architecture": ("string", "freemoe-tiny"),
        "general.name": ("string", name),
        "general.alignment": ("u32", ALIGN),
        "general.file_type": ("u32", file_type),
        "general.quantization_version": ("u32", 1),
        "freemoe.adapter": ("string", "freemoe-tiny-v1"),
        "freemoe.model_kind": ("string", "moe"),
        "freemoe.hidden_size": ("u32", H),
        "freemoe.intermediate_size": ("u32", I),
        "freemoe.num_hidden_layers": ("u32", L),
        "freemoe.num_attention_heads": ("u32", HEADS),
        "freemoe.num_key_value_heads": ("u32", HEADS),
        "freemoe.head_dim": ("u32", HD),
        "freemoe.vocab_size": ("u32", V),
        "freemoe.max_context_length": ("u32", CTX),
        "freemoe.rms_norm_epsilon": ("f32", 1e-5),
        "freemoe.rope_theta": ("f32", 10000.0),
        "freemoe.rope_mode": ("string", "half"),
        "freemoe.tokenizer_model": ("string", "freemoe.byte-v1"),
        "freemoe.router_operation": ("string", "linear-topk-softmax-v1"),
        "freemoe.shared_expert": ("bool", False),
        "freemoe.expert_layout": ("string", "stacked-f32"),  # experts stay block-encoded in a stacked tensor
        "freemoe.expert_count": ("u32", E),
        "freemoe.expert_top_k": ("u32", TOPK),
        "freemoe.expert_checksums": ("array_u64", expert_checksums(tensors, cfg)),
        "freemoe.seed": ("u32", SEED),
        "tokenizer.ggml.model": ("string", "freemoe.byte-v1"),
        "tokenizer.ggml.bos_token_id": ("u32", 256),
        "tokenizer.ggml.eos_token_id": ("u32", 257),
        "tokenizer.ggml.tokens": ("array_string", tokenizer_tokens(V)),
        "tokenizer.chat_template": ("string", "raw-v1"),
    }


def main() -> int:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, dict] = {}
    for label, type_id, block_size, block_bytes in VARIANTS:
        cfg = CFG_256 if block_size > 32 else CFG_32
        spec = build_tensors(cfg)
        encoder = ENCODERS[label]
        quantised: list[TensorSpec] = []
        reference: list[TensorSpec] = []
        quant_bytes = 0
        tensor_count = 0
        total_f32 = sum(len(values) * 4 for _, _, values in spec)
        for name, dims, values in spec:
            quantisable = len(dims) >= 2 and dims[0] % block_size == 0
            if not quantisable:
                data = f32_bytes(values)
                quantised.append(TensorSpec(name, list(dims), data, 0))
                reference.append(TensorSpec(name, list(dims), data, 0))
                quant_bytes += len(data)
                continue
            payload = encoder(values)
            decoded = decode_reference(type_id, payload, block_bytes)
            if len(decoded) != len(values):
                raise SystemExit(f"{label}: decoder returned {len(decoded)} values for {len(values)}")
            quantised.append(TensorSpec(name, list(dims), payload, type_id))
            reference.append(TensorSpec(name, list(dims), f32_bytes(decoded), 0))
            quant_bytes += len(payload)
            tensor_count += 1

        q_path = FIXTURES / f"quant-moe-{label}.gguf"
        r_path = FIXTURES / f"quant-moe-{label}-f32ref.gguf"
        q_path.write_bytes(GGUFWriter(metadata(quantised, cfg,
                                               name=f"FreeMoe synthetic quant MoE {label.upper()}",
                                               file_type=type_id), quantised).build())
        r_path.write_bytes(GGUFWriter(metadata(reference, cfg,
                                               name=f"FreeMoe synthetic quant MoE {label.upper()} reference",
                                               file_type=0), reference).build())
        outputs[label] = {
            "model": q_path.name,
            "model_sha256": hashlib.sha256(q_path.read_bytes()).hexdigest(),
            "model_bytes": q_path.stat().st_size,
            "reference": r_path.name,
            "reference_sha256": hashlib.sha256(r_path.read_bytes()).hexdigest(),
            "reference_bytes": r_path.stat().st_size,
            "quantised_tensors": tensor_count,
            "quantised_payload_bytes": quant_bytes,
            "payload_ratio": round(total_f32 / quant_bytes, 3) if quant_bytes else 0,
            "config": {
                "hidden": cfg.hidden, "intermediate": cfg.intermediate,
                "layers": cfg.layers, "heads": cfg.heads, "head_dim": cfg.head_dim,
                "vocab": cfg.vocab, "context": cfg.context, "experts": cfg.experts,
                "top_k": cfg.top_k,
            },
        }

    manifest = {
        "schema": "freemoe.quant-fixtures.v2",
        "generator": "tools/make_quant_fixtures.py",
        "synthetic": True,
        "variants": outputs,
        "note": (
            "Each -f32ref twin stores exactly the values an *independent Python "
            "decoder* (tools/quant_reference.py) produces for the quantised file, "
            "so matching greedy tokens and logits prove both the C++ decoder and "
            "the execution path."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "written", "variants": {k: v["model_bytes"] for k, v in outputs.items()}}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
