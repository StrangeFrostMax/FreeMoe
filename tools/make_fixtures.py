#!/usr/bin/env python3
"""Generate all deterministic FreeMoe correctness fixtures and reference data.

Usage:
    python3 tools/make_fixtures.py --all
    python3 tools/make_fixtures.py --verify

All files are generated locally with Python's standard library only. The tiny
checkpoints are synthetic conformance fixtures, not pretrained language models.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import struct
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gguf_format import (  # noqa: E402
    GGUFWriter, TensorSpec, fnv1a64, pack_string, pack_value, tensor_byte_size, write_gguf,
)
import reference_oracle as oracle  # noqa: E402

SEED = 0xFEEE0E
H = 16
I = 32
L = 2
HEADS = 4
HD = 4
V = 258
CTX = 1024
E = 8
TOPK = 2
ALIGN = 32
N_GREEDY = 16
DENSE_PATH = "tests/fixtures/dense-fm.gguf"
MOE_PATH = "tests/fixtures/moe-fm.gguf"
METADATA_PATH = "tests/fixtures/metadata-types.gguf"


def f32_bytes(values: list[float]) -> bytes:
    return struct.pack(f"<{len(values)}f", *values)


def deterministic_values(name: str, count: int, scale: float) -> list[float]:
    digest = hashlib.sha256(f"FreeMoe-v1:{SEED}:{name}".encode("utf-8")).digest()
    rng = random.Random(int.from_bytes(digest[:8], "little"))
    return [(rng.random() * 2.0 - 1.0) * scale for _ in range(count)]


def norm_values(name: str, count: int) -> list[float]:
    raw = deterministic_values(name, count, 0.05)
    return [1.0 + x for x in raw]


def product(dims: list[int]) -> int:
    out = 1
    for d in dims:
        out *= d
    return out


def make_tensor(name: str, dims: list[int], scale: float, *, norm: bool = False) -> TensorSpec:
    n = product(dims)
    values = norm_values(name, n) if norm else deterministic_values(name, n, scale)
    return TensorSpec(name, list(dims), f32_bytes(values), 0)


def base_tensors(kind: str) -> list[TensorSpec]:
    t = [make_tensor("token_embd.weight", [H, V], 0.35)]
    for layer in range(L):
        p = f"blk.{layer}."
        t += [
            make_tensor(p + "attn_norm.weight", [H], 0.0, norm=True),
            make_tensor(p + "attn_q.weight", [H, H], 0.18),
            make_tensor(p + "attn_k.weight", [H, H], 0.18),
            make_tensor(p + "attn_v.weight", [H, H], 0.18),
            make_tensor(p + "attn_output.weight", [H, H], 0.18),
            make_tensor(p + "ffn_norm.weight", [H], 0.0, norm=True),
        ]
        if kind == "dense":
            t += [
                make_tensor(p + "ffn_gate.weight", [H, I], 0.18),
                make_tensor(p + "ffn_up.weight", [H, I], 0.18),
                make_tensor(p + "ffn_down.weight", [I, H], 0.18),
            ]
        else:
            t += [
                make_tensor(p + "ffn_router.weight", [H, E], 0.25),
                make_tensor(p + "ffn_gate_exps.weight", [H, I, E], 0.18),
                make_tensor(p + "ffn_up_exps.weight", [H, I, E], 0.18),
                make_tensor(p + "ffn_down_exps.weight", [I, H, E], 0.18),
            ]
    t += [
        make_tensor("output_norm.weight", [H], 0.0, norm=True),
        make_tensor("output.weight", [H, V], 0.25),
    ]
    return t


def expert_checksums(tensors: list[TensorSpec], kind: str) -> list[int]:
    if kind != "moe":
        return []
    by_name = {t.name: t for t in tensors}
    out: list[int] = []
    for layer in range(L):
        gate = by_name[f"blk.{layer}.ffn_gate_exps.weight"].data
        up = by_name[f"blk.{layer}.ffn_up_exps.weight"].data
        down = by_name[f"blk.{layer}.ffn_down_exps.weight"].data
        expert_bytes = H * I * 4
        for expert in range(E):
            payload = b"".join((
                gate[expert * expert_bytes:(expert + 1) * expert_bytes],
                up[expert * expert_bytes:(expert + 1) * expert_bytes],
                down[expert * expert_bytes:(expert + 1) * expert_bytes],
            ))
            out.append(fnv1a64(payload))
    return out


def tokenizer_tokens() -> list[str]:
    return [f"<byte_{i:02x}>" for i in range(256)] + ["<bos>", "<eos>"]


def metadata(kind: str, tensors: list[TensorSpec], *, overrides: dict[str, tuple[Any, ...]] | None = None,
             checksum_override: list[int] | None = None) -> dict[str, tuple[Any, ...]]:
    md: dict[str, tuple[Any, ...]] = {
        "general.architecture": ("string", "freemoe-tiny"),
        "general.name": ("string", f"FreeMoe synthetic {kind} F32"),
        "general.alignment": ("u32", ALIGN),
        "general.file_type": ("u32", 0),
        "general.quantization_version": ("u32", 1),
        "freemoe.adapter": ("string", "freemoe-tiny-v1"),
        "freemoe.model_kind": ("string", kind),
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
        "freemoe.expert_layout": ("string", "stacked-f32" if kind == "moe" else "none"),
        "freemoe.expert_count": ("u32", E if kind == "moe" else 0),
        "freemoe.expert_top_k": ("u32", TOPK if kind == "moe" else 0),
        "freemoe.expert_checksums": ("array_u64", checksum_override if checksum_override is not None else expert_checksums(tensors, kind)),
        "freemoe.seed": ("u32", SEED),
        "tokenizer.ggml.model": ("string", "freemoe.byte-v1"),
        "tokenizer.ggml.bos_token_id": ("u32", 256),
        "tokenizer.ggml.eos_token_id": ("u32", 257),
        "tokenizer.ggml.tokens": ("array_string", tokenizer_tokens()),
        "tokenizer.chat_template": ("string", "raw-v1"),
    }
    if overrides:
        md.update(overrides)
    return md


def build_model_bytes(kind: str, *, overrides: dict[str, tuple[Any, ...]] | None = None,
                      drop_tensor: str | None = None, extra_tensor: TensorSpec | None = None,
                      duplicate_first_name: bool = False,
                      checksum_override: list[int] | None = None) -> tuple[bytes, dict[str, Any], dict[str, tuple[Any, ...]], list[TensorSpec]]:
    tensors = base_tensors(kind)
    if drop_tensor:
        tensors = [t for t in tensors if t.name != drop_tensor]
    if extra_tensor:
        tensors.append(extra_tensor)
    if duplicate_first_name and len(tensors) >= 2:
        tensors[1] = TensorSpec(tensors[0].name, tensors[1].dims, tensors[1].data, tensors[1].type_id)
    md = metadata(kind, tensors, overrides=overrides, checksum_override=checksum_override)
    # Build in memory. Only generate_all() writes the two canonical checkpoints;
    # malformed variants must never overwrite them.
    writer = GGUFWriter(md, tensors)
    blob = writer.build()
    return blob, writer.layout, md, tensors


ROLE_SUFFIXES = {
    "attn_norm.weight": "ATTN_NORM",
    "attn_q.weight": "ATTN_Q",
    "attn_k.weight": "ATTN_K",
    "attn_v.weight": "ATTN_V",
    "attn_output.weight": "ATTN_OUTPUT",
    "ffn_norm.weight": "FFN_NORM",
    "ffn_gate.weight": "FFN_GATE",
    "ffn_up.weight": "FFN_UP",
    "ffn_down.weight": "FFN_DOWN",
    "ffn_router.weight": "FFN_ROUTER",
    "ffn_gate_exps.weight": "FFN_GATE_EXPS",
    "ffn_up_exps.weight": "FFN_UP_EXPS",
    "ffn_down_exps.weight": "FFN_DOWN_EXPS",
}


def role_for(name: str) -> tuple[str, int]:
    if name == "token_embd.weight":
        return "TOKEN_EMBD", -1
    if name == "output.weight":
        return "OUTPUT", -1
    if name == "output_norm.weight":
        return "OUTPUT_NORM", -1
    if name.startswith("blk."):
        parts = name.split(".", 2)
        if len(parts) == 3 and parts[0] == "blk" and parts[1].isdigit() and parts[2] in ROLE_SUFFIXES:
            return ROLE_SUFFIXES[parts[2]], int(parts[1])
    return "UNKNOWN", -1


def golden_inspect(kind: str, source_path: str, blob: bytes, layout: dict[str, Any], md: dict[str, tuple[Any, ...]], tensors: list[TensorSpec]) -> dict[str, Any]:
    tlayouts = {t.name: t for t in layout["tensors"]}
    roles = []
    total_bytes = 0
    for spec in sorted(tensors, key=lambda t: t.name):
        info = tlayouts[spec.name]
        role, layer = role_for(spec.name)
        size = tensor_byte_size(spec.dims, spec.type_id)
        total_bytes += size
        roles.append({
            "name": spec.name,
            "role": role,
            "layer": layer,
            "expert": -1,
            "type": "F32",
            "dims": spec.dims,
            "offset": info.absolute_offset,
            "bytes": size,
        })
    expert_records = []
    if kind == "moe":
        checksums = md["freemoe.expert_checksums"][1]
        for layer in range(L):
            for expert in range(E):
                extents = []
                for role_name in ("FFN_GATE_EXPS", "FFN_UP_EXPS", "FFN_DOWN_EXPS"):
                    suffix = {
                        "FFN_GATE_EXPS": "ffn_gate_exps.weight",
                        "FFN_UP_EXPS": "ffn_up_exps.weight",
                        "FFN_DOWN_EXPS": "ffn_down_exps.weight",
                    }[role_name]
                    tname = f"blk.{layer}.{suffix}"
                    tinfo = tlayouts[tname]
                    stride = H * I * 4
                    extents.append({
                        "role": role_name,
                        "offset": tinfo.absolute_offset + expert * stride,
                        "bytes": stride,
                    })
                expert_records.append({
                    "layer": layer,
                    "expert": expert,
                    "bytes": 3 * H * I * 4,
                    "checksum": f"{checksums[layer * E + expert]:016x}",
                    "extents": extents,
                })
    summary_keys = [
        "general.architecture", "general.name", "general.alignment",
        "freemoe.adapter", "freemoe.model_kind", "freemoe.hidden_size",
        "freemoe.intermediate_size", "freemoe.num_hidden_layers",
        "freemoe.num_attention_heads", "freemoe.num_key_value_heads",
        "freemoe.head_dim", "freemoe.vocab_size", "freemoe.max_context_length",
        "freemoe.rope_mode", "freemoe.tokenizer_model",
        "freemoe.router_operation", "freemoe.shared_expert",
        "freemoe.expert_layout", "freemoe.expert_count",
        "freemoe.expert_top_k", "freemoe.seed", "tokenizer.chat_template",
    ]
    summary = {}
    for key in summary_keys:
        value = md[key][1]
        if isinstance(value, bool):
            summary[key] = "true" if value else "false"
        else:
            summary[key] = str(value)
    return {
        "schema": "freemoe.inspect.v1",
        "status": "validated",
        "source_path": source_path,
        "file_size": len(blob),
        "data_offset": layout["data_offset"],
        "gguf_version": 3,
        "alignment": ALIGN,
        "metadata_count": len(md),
        "tensor_count": len(tensors),
        "tensor_bytes": total_bytes,
        "source_fingerprint": f"{layout['source_fingerprint']:016x}",
        "architecture_adapter": {"id": "freemoe-tiny-v1", "version": 1},
        "model_kind": kind,
        "quantization_types": ["F32"],
        "tokenizer": {"model": "freemoe.byte-v1", "bos_token_id": 256, "eos_token_id": 257, "chat_template": "raw-v1"},
        "metadata_summary": summary,
        "expert_count": E if kind == "moe" else 0,
        "expert_top_k": TOPK if kind == "moe" else 0,
        "expert_layout": "stacked-f32" if kind == "moe" else "none",
        "roles": roles,
        "unresolved_roles": [],
        "expert_records": expert_records,
        "validation": {"errors": [], "warnings": []},
    }


def prompts() -> list[str]:
    base = "the quick brown fox jumps over lazy dogs while free moe pages experts from disk and verifies every selected weight "
    out = []
    for i in range(20):
        length = 8 + round(i * (512 - 8) / 19)
        text = (base * math.ceil(length / len(base)))[:length]
        assert len(text.encode("utf-8")) == length
        out.append(text)
    return out


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_reference_outputs(kind: str, model_path: Path, prompt_list: list[str]) -> list[Path]:
    cfg = oracle.load_model(model_path)
    written: list[Path] = []
    ref_dir = ROOT / "tests" / "reference" / kind
    for i, prompt in enumerate(prompt_list):
        stem = f"prompt_{i:02d}"
        result = oracle.run_prompt(cfg, prompt, N_GREEDY)
        input_ids = ROOT / "tests" / "reference" / f"{stem}.input.ids"
        oracle.write_text(input_ids, "\n".join(map(str, result["input_ids"])) + "\n")
        written.append(input_ids)
        logits = ref_dir / f"{stem}.logits"
        oracle.write_text(logits, f"{cfg['V']}\n" + " ".join(f"{x:.9g}" for x in result["prefill_logits"]) + "\n")
        written.append(logits)
        greedy_ids = ref_dir / f"{stem}.greedy.ids"
        oracle.write_text(greedy_ids, "\n".join(map(str, result["generated_ids"])) + "\n")
        written.append(greedy_ids)
        if kind == "moe":
            routing = ref_dir / f"{stem}.routing.txt"
            lines = [f"{token} {layer} " + " ".join(map(str, ids)) for token, layer, ids in result["routing"]]
            oracle.write_text(routing, "\n".join(lines) + "\n")
            written.append(routing)
        if result["selected_count"] != result["executed_count"]:
            raise RuntimeError("reference selected/executed mismatch")
    return written


def make_quant_fixture() -> tuple[bytes, dict[str, Any]]:
    """Valid Q8_0 block geometry probe; used by C++ parser unit tests."""
    md: dict[str, tuple[Any, ...]] = {
        "general.architecture": ("string", "freemoe-quant-probe"),
        "general.alignment": ("u32", 32),
    }
    tensor = TensorSpec("probe.weight", [32, 2], b"\x00" * 68, 8)  # 2 * Q8_0 blocks
    writer = GGUFWriter(md, [tensor])
    blob = writer.build()
    (ROOT / "tests" / "fixtures" / "quant-blocks.gguf").write_bytes(blob)
    return blob, writer.layout


def make_metadata_types_fixture() -> None:
    md: dict[str, tuple[Any, ...]] = {
        "general.architecture": ("string", "freemoe-metadata-types"),
        "general.alignment": ("u32", 32),
        "test.u8": ("u8", 200),
        "test.i8": ("i8", -100),
        "test.u16": ("u16", 60000),
        "test.i16": ("i16", -30000),
        "test.u32": ("u32", 4000000000),
        "test.i32": ("i32", -2000000000),
        "test.f32": ("f32", 1.25),
        "test.bool": ("bool", True),
        "test.string": ("string", "metadata types"),
        "test.u64": ("u64", 18000000000000000000),
        "test.i64": ("i64", -9000000000000000000),
        "test.f64": ("f64", 3.5),
        "test.array_u8": ("array_u8", [1, 2, 3]),
        "test.array_i8": ("array_i8", [-1, 0, 1]),
        "test.array_u16": ("array_u16", [1, 65535]),
        "test.array_i16": ("array_i16", [-32768, 32767]),
        "test.array_u32": ("array_u32", [1, 2, 3]),
        "test.array_i32": ("array_i32", [-1, 2]),
        "test.array_f32": ("array_f32", [1.0, 2.5]),
        "test.array_bool": ("array_bool", [True, False]),
        "test.array_string": ("array_string", ["a", "bb", "ccc"]),
        "test.array_u64": ("array_u64", [1, 2]),
        "test.array_i64": ("array_i64", [-1, 2]),
        "test.array_f64": ("array_f64", [1.5, -2.5]),
        "test.nested_array_u32": ("array_array_u32", [[1, 2], [3, 4, 5]]),
    }
    write_gguf(ROOT / METADATA_PATH, md, [])


def make_malformed(dense_blob: bytes, dense_layout: dict[str, Any], dense_md: dict[str, tuple[Any, ...]], dense_tensors: list[TensorSpec], q8_blob: bytes, q8_layout: dict[str, Any]) -> None:
    out_dir = ROOT / "tests" / "malformed_models"
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.gguf"):
        old.unlink()
    rows: list[tuple[str, str, str, str]] = []

    def add(name: str, blob: bytes, category: str, detail: str) -> None:
        path = out_dir / f"{name}.gguf"
        path.write_bytes(blob)
        rows.append((path.name, category, "TYPED_ERROR", detail))

    # 30 truncation cases: no truncation is at or beyond the real file size.
    for i in range(30):
        cut = max(0, min(len(dense_blob) - 1, (i * (len(dense_blob) - 1)) // 30))
        add(f"trunc_{i:02d}", dense_blob[:cut], "truncation", f"cut={cut}")

    def mutate(name: str, category: str, detail: str, func) -> None:
        b = bytearray(dense_blob)
        func(b)
        add(name, bytes(b), category, detail)

    mutate("bad_magic", "header", "bad magic", lambda b: b.__setitem__(slice(0, 4), b"NOPE"))
    mutate("bad_version", "header", "unsupported GGUF version", lambda b: struct.pack_into("<I", b, 4, 2))
    mutate("tensor_count_overflow", "header", "unreasonable tensor count", lambda b: struct.pack_into("<Q", b, 8, 1 << 62))
    mutate("metadata_count_overflow", "header", "unreasonable metadata count", lambda b: struct.pack_into("<Q", b, 16, 1 << 62))
    first_key = next(iter(dense_md))
    key_len_off = dense_layout["metadata_offsets"][first_key]["key_len_offset"]
    value_type_off = dense_layout["metadata_offsets"][first_key]["value_type_offset"]
    mutate("key_length_overflow", "metadata", "oversized string", lambda b: struct.pack_into("<Q", b, key_len_off, 1 << 62))
    mutate("invalid_value_type", "metadata", "invalid KV type", lambda b: struct.pack_into("<I", b, value_type_off, 99))
    checksums_off = dense_layout["metadata_offsets"]["freemoe.expert_checksums"]["value_offset"]
    mutate("array_count_overflow", "metadata", "oversized array", lambda b: struct.pack_into("<Q", b, checksums_off + 4, 1 << 62))
    align_off = dense_layout["metadata_offsets"]["general.alignment"]["value_offset"]
    mutate("bad_alignment", "metadata", "alignment not power of two", lambda b: struct.pack_into("<I", b, align_off, 3))

    first_tensor_layout = dense_layout["tensors"][0]
    mutate("dimension_count_overflow", "tensor_directory", "invalid dimension count", lambda b: struct.pack_into("<I", b, first_tensor_layout.n_dims_offset, 0xFFFFFFFF))
    def dim_overflow(b):
        struct.pack_into("<Q", b, first_tensor_layout.dims_offset, (1 << 32) - 1)
        struct.pack_into("<Q", b, first_tensor_layout.dims_offset + 8, (1 << 32) - 1)
    mutate("dimension_overflow", "tensor_directory", "dimension product overflow", dim_overflow)
    mutate("unknown_quant_type", "tensor_directory", "unknown quantisation type", lambda b: struct.pack_into("<I", b, first_tensor_layout.type_offset, 999))
    mutate("offset_overflow", "tensor_directory", "offset overflow", lambda b: struct.pack_into("<Q", b, first_tensor_layout.offset_offset, (1 << 64) - 1))
    mutate("tensor_out_of_bounds", "tensor_directory", "tensor beyond file", lambda b: struct.pack_into("<Q", b, first_tensor_layout.offset_offset, len(dense_blob)))
    mutate("misaligned_offset", "tensor_directory", "misaligned tensor offset", lambda b: struct.pack_into("<Q", b, first_tensor_layout.offset_offset, 1))
    mutate("shape_mismatch", "adapter", "embedding hidden dimension mismatch", lambda b: struct.pack_into("<Q", b, first_tensor_layout.dims_offset, H - 1))

    # String substitutions with equal byte lengths.
    def replace_once(name: str, category: str, detail: str, old: bytes, new: bytes) -> None:
        assert len(old) == len(new)
        b = bytearray(dense_blob)
        pos = b.find(old)
        assert pos >= 0
        b[pos:pos + len(old)] = new
        add(name, bytes(b), category, detail)

    replace_once("bad_architecture", "adapter", "unknown architecture", b"freemoe-tiny", b"unknown-tiny")
    replace_once("bad_tokenizer", "tokenizer", "unknown tokenizer model", b"freemoe.byte-v1", b"unknown.byte-v1")
    replace_once("bad_chat_template", "tokenizer", "unknown chat template", b"raw-v1", b"weird1")

    # Structural variants built from the same deterministic weights.
    variants = [
        ("missing_required_tensor", {"drop_tensor": "blk.0.attn_q.weight"}),
        ("unknown_tensor", {"extra_tensor": make_tensor("mystery.weight", [H], 0.1)}),
        ("duplicate_tensor_name", {"duplicate_first_name": True}),
        ("expert_checksum_count_mismatch", {"checksum_override": [1, 2, 3]}),
        ("expert_count_mismatch", {"overrides": {"freemoe.expert_count": ("u32", E - 1)}}),
    ]
    # The last two are MoE-specific but can be injected as dense metadata
    # overrides too; the C++ adapter must reject inconsistent expert metadata.
    for name, kwargs in variants:
        blob, _, _, _ = build_model_bytes("dense", **kwargs)
        add(name, blob, "adapter", name.replace("_", " "))

    # A Q8_0 tensor whose ne0 is not a whole quantisation block must fail.
    q8_bad = bytearray(q8_blob)
    struct.pack_into("<Q", q8_bad, q8_layout["tensors"][0].dims_offset, 31)
    add("quant_block_misaligned", bytes(q8_bad), "quantisation", "Q8_0 ne0 is not block aligned")

    manifest = ROOT / "tests" / "malformed_models" / "manifest.tsv"
    lines = ["fixture\tcategory\texpected_code\tdetail"]
    lines += ["\t".join(row) for row in rows]
    manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    if len(rows) < 50:
        raise RuntimeError(f"expected at least 50 malformed fixtures, got {len(rows)}")


def write_checkpoint_docs(files: dict[str, Path]) -> None:
    docs = ROOT / "docs"
    docs.mkdir(exist_ok=True)
    for kind, path in (("DENSE", files["dense"]), ("MOE", files["moe"])):
        doc = f"""# Checkpoint {kind}

| Field | Value |
|---|---|
| Name | FreeMoe synthetic {kind.lower()} F32 conformance fixture |
| Source URL | No external URL; generated in-repo by `tools/make_fixtures.py` |
| Local path | `{path.relative_to(ROOT).as_posix()}` |
| SHA-256 | `{sha256(path)}` |
| Architecture string | `freemoe-tiny` |
| Adapter | `freemoe-tiny-v1` |
| Quantisation | F32 only |
| Licence name | MIT |
| Licence text | `LICENSES/CHECKPOINTS-MIT.txt` |
| Model kind | `{kind.lower()}` |
| Expert count | {E if kind == 'MOE' else 0} |
| Top-k | {TOPK if kind == 'MOE' else 0} |

This is a deterministic synthetic checkpoint with no pretrained weights. It is
used to test parser, adapter, forward-pass and expert-paging semantics. It is
not evidence of support for arbitrary Llama or MoE GGUF models.
"""
        (docs / f"CHECKPOINT_{kind}.md").write_text(doc, encoding="utf-8")


def write_manifest(paths: list[Path]) -> None:
    out = ROOT / "tests" / "reference" / "manifest.sha256"
    lines = []
    for path in sorted(set(paths)):
        lines.append(f"{sha256(path)}  {path.relative_to(ROOT).as_posix()}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_all() -> None:
    (ROOT / "tests" / "fixtures").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests" / "golden").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests" / "reference").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests" / "reference" / "dense").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests" / "reference" / "moe").mkdir(parents=True, exist_ok=True)
    (ROOT / "LICENSES").mkdir(parents=True, exist_ok=True)

    dense_blob, dense_layout, dense_md, dense_tensors = build_model_bytes("dense")
    moe_blob, moe_layout, moe_md, moe_tensors = build_model_bytes("moe")
    (ROOT / DENSE_PATH).write_bytes(dense_blob)
    (ROOT / MOE_PATH).write_bytes(moe_blob)
    make_metadata_types_fixture()
    q8_blob, q8_layout = make_quant_fixture()

    dense_golden = golden_inspect("dense", DENSE_PATH, dense_blob, dense_layout, dense_md, dense_tensors)
    moe_golden = golden_inspect("moe", MOE_PATH, moe_blob, moe_layout, moe_md, moe_tensors)
    (ROOT / "tests" / "golden" / "inspect_dense.json").write_text(
        json.dumps(dense_golden, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")
    (ROOT / "tests" / "golden" / "inspect_moe.json").write_text(
        json.dumps(moe_golden, separators=(",", ":"), ensure_ascii=False) + "\n", encoding="utf-8")

    prompt_list = prompts()
    prompts_path = ROOT / "tests" / "reference" / "prompts.txt"
    prompts_path.write_text("\n".join(prompt_list) + "\n", encoding="utf-8")
    for i, prompt in enumerate(prompt_list):
        (ROOT / "tests" / "reference" / f"prompt_{i:02d}.txt").write_text(prompt + "\n", encoding="utf-8")

    tracked = [
        ROOT / DENSE_PATH, ROOT / MOE_PATH, ROOT / METADATA_PATH,
        ROOT / "tests" / "fixtures" / "quant-blocks.gguf", prompts_path,
        ROOT / "tests" / "golden" / "inspect_dense.json",
        ROOT / "tests" / "golden" / "inspect_moe.json",
        ROOT / "tools" / "gguf_format.py", ROOT / "tools" / "reference_oracle.py",
        ROOT / "tools" / "make_fixtures.py",
    ]
    tracked += [ROOT / "tests" / "reference" / f"prompt_{i:02d}.txt" for i in range(20)]
    tracked += write_reference_outputs("dense", ROOT / DENSE_PATH, prompt_list)
    tracked += write_reference_outputs("moe", ROOT / MOE_PATH, prompt_list)
    make_malformed(dense_blob, dense_layout, dense_md, dense_tensors, q8_blob, q8_layout)
    tracked += list((ROOT / "tests" / "malformed_models").glob("*.gguf"))
    tracked += [ROOT / "tests" / "malformed_models" / "manifest.tsv"]
    write_manifest(tracked)
    write_checkpoint_docs({"dense": ROOT / DENSE_PATH, "moe": ROOT / MOE_PATH})
    print(json.dumps({
        "dense_sha256": sha256(ROOT / DENSE_PATH),
        "moe_sha256": sha256(ROOT / MOE_PATH),
        "prompts": len(prompt_list),
        "prompt_bytes": [len(p) for p in prompt_list],
        "malformed": len(list((ROOT / "tests" / "malformed_models").glob("*.gguf"))),
        "reference_files": len(tracked),
    }, indent=2))


def verify() -> None:
    import subprocess
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--all"],
                            cwd=ROOT, capture_output=True, text=True)
    # Regeneration into a clean checkout should be byte-identical. Caller can
    # use git status to detect drift; print the regenerated summary.
    print(result.stdout)
    if result.returncode:
        print(result.stderr, file=sys.stderr)
    raise SystemExit(result.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    if args.verify:
        return verify()
    if not args.all:
        ap.error("pass --all or --verify")
    generate_all()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
