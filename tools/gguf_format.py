#!/usr/bin/env python3
"""Minimal deterministic GGUF v3 writer/scanner used only to build fixtures.

This is deliberately independent of the C++ runtime parser. It supports every
GGUF metadata value type used by the fixtures, including arrays and one nested
array. It is not a general-purpose GGUF conversion tool.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

MAGIC = b"GGUF"
VERSION = 3

# GGUF value type ids.
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)

# (block_size, block_bytes, name). Only types whose geometry is verified here.
QUANT_TABLE = {
    0: (1, 4, "F32"),
    1: (1, 2, "F16"),
    2: (32, 18, "Q4_0"),
    3: (32, 20, "Q4_1"),
    6: (32, 22, "Q5_0"),
    7: (32, 24, "Q5_1"),
    8: (32, 34, "Q8_0"),
    9: (32, 40, "Q8_1"),
    10: (256, 84, "Q2_K"),
    11: (256, 110, "Q3_K"),
    12: (256, 144, "Q4_K"),
    13: (256, 176, "Q5_K"),
    14: (256, 210, "Q6_K"),
    15: (256, 292, "Q8_K"),
    20: (32, 18, "IQ4_NL"),
    24: (1, 1, "I8"),
    25: (1, 2, "I16"),
    26: (1, 4, "I32"),
    27: (1, 8, "I64"),
    28: (1, 8, "F64"),
    30: (1, 2, "BF16"),
    # Ternary family, including the two PrismML-private ids so a Bonsai-shaped
    # fixture can be written and inspected.
    34: (256, 54, "TQ1_0"),
    35: (256, 66, "TQ2_0"),
    41: (128, 18, "Q1_0"),
    42: (64, 18, "Q2_0"),
    142: (128, 34, "PQ2_0"),
    143: (128, 28, "PTQ1_0"),
}

TYPE_BY_NAME = {
    "u8": U8, "i8": I8, "u16": U16, "i16": I16,
    "u32": U32, "i32": I32, "f32": F32, "bool": BOOL,
    "string": STRING, "array": ARRAY, "u64": U64, "i64": I64, "f64": F64,
}


class GGUFError(ValueError):
    pass


def fnv1a64(data: bytes) -> int:
    h = 0xCBF29CE484222325
    for b in data:
        h ^= b
        h = (h * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def align_up(value: int, alignment: int) -> int:
    if alignment <= 0 or (alignment & (alignment - 1)):
        raise GGUFError("alignment must be a power of two")
    return (value + alignment - 1) & ~(alignment - 1)


def pack_string(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _scalar_spec(name: str) -> tuple[int, str]:
    table = {
        "u8": (U8, "<B"), "i8": (I8, "<b"),
        "u16": (U16, "<H"), "i16": (I16, "<h"),
        "u32": (U32, "<I"), "i32": (I32, "<i"),
        "f32": (F32, "<f"), "bool": (BOOL, "<?"),
        "u64": (U64, "<Q"), "i64": (I64, "<q"),
        "f64": (F64, "<d"),
    }
    if name not in table:
        raise GGUFError(f"unsupported scalar type {name!r}")
    return table[name]


def pack_value(value: tuple[Any, ...], *, _depth: int = 0) -> bytes:
    kind = value[0]
    if kind == "string":
        return struct.pack("<I", STRING) + pack_string(value[1])
    if kind == "array_string":
        if _depth >= 4:
            raise GGUFError("array nesting too deep")
        payload = struct.pack("<I", STRING) + struct.pack("<Q", len(value[1]))
        for item in value[1]:
            payload += pack_string(item)
        return struct.pack("<I", ARRAY) + payload
    if kind == "array_array_u32":
        if _depth >= 4:
            raise GGUFError("array nesting too deep")
        payload = struct.pack("<I", ARRAY) + struct.pack("<Q", len(value[1]))
        for row in value[1]:
            # Nested array payload: element type + count + values (without top-level type).
            payload += struct.pack("<I", U32) + struct.pack("<Q", len(row))
            for item in row:
                payload += struct.pack("<I", item)
        return struct.pack("<I", ARRAY) + payload
    if kind.startswith("array_"):
        if _depth >= 4:
            raise GGUFError("array nesting too deep")
        elem = kind[6:]
        if elem.startswith("array_"):
            etype = ARRAY
            payload = struct.pack("<Q", len(value[1]))
            for item in value[1]:
                payload += pack_value(item, _depth=_depth + 1)[4:]
        else:
            etype, fmt = _scalar_spec(elem)
            payload = struct.pack("<I", etype) + struct.pack("<Q", len(value[1]))
            if elem in ("string", "str"):
                raise GGUFError("use array_string")
            for item in value[1]:
                payload += struct.pack(fmt, item)
        return struct.pack("<I", ARRAY) + payload
    if kind == "array_string":
        if _depth >= 4:
            raise GGUFError("array nesting too deep")
        payload = struct.pack("<I", STRING) + struct.pack("<Q", len(value[1]))
        for item in value[1]:
            payload += pack_string(item)
        return struct.pack("<I", ARRAY) + payload
    if kind == "array_array_u32":
        if _depth >= 4:
            raise GGUFError("array nesting too deep")
        payload = struct.pack("<I", ARRAY) + struct.pack("<Q", len(value[1]))
        for row in value[1]:
            # Nested array payload: element type + count + values (without top-level type).
            payload += struct.pack("<I", U32) + struct.pack("<Q", len(row))
            for item in row:
                payload += struct.pack("<I", item)
        return struct.pack("<I", ARRAY) + payload
    etype, fmt = _scalar_spec(kind)
    return struct.pack("<I", etype) + struct.pack(fmt, value[1])


@dataclass
class TensorSpec:
    name: str
    dims: list[int]
    data: bytes
    type_id: int = 0  # F32


@dataclass
class TensorLayout:
    name: str
    dims: list[int]
    type_id: int
    relative_offset: int
    byte_size: int
    name_len_offset: int
    n_dims_offset: int
    dims_offset: int
    type_offset: int
    offset_offset: int
    absolute_offset: int


class GGUFWriter:
    def __init__(self, metadata: dict[str, tuple[Any, ...]], tensors: list[TensorSpec]):
        self.metadata = metadata
        self.tensors = tensors
        self.layout: dict[str, Any] = {}

    def _metadata_bytes(self) -> bytes:
        out = bytearray()
        offsets: dict[str, dict[str, int]] = {}
        for key, value in self.metadata.items():
            key_len_offset = 24 + len(out)
            packed_key = pack_string(key)
            out += packed_key
            type_offset = 24 + len(out)
            packed_value = pack_value(value)
            out += packed_value
            offsets[key] = {
                "key_len_offset": key_len_offset,
                "value_type_offset": type_offset,
                "value_offset": type_offset + 4,
            }
        self.layout["metadata_offsets"] = offsets
        return bytes(out)

    def _tensor_directory_bytes(self, relative_offsets: list[int]) -> bytes:
        out = bytearray()
        directory_start = 24 + self.layout["metadata_size"]
        tensor_layouts: list[TensorLayout] = []
        for tensor, rel in zip(self.tensors, relative_offsets):
            name_len_offset = directory_start + len(out)
            out += pack_string(tensor.name)
            n_dims_offset = directory_start + len(out)
            out += struct.pack("<I", len(tensor.dims))
            dims_offset = directory_start + len(out)
            for dim in tensor.dims:
                out += struct.pack("<Q", dim)
            type_offset = directory_start + len(out)
            out += struct.pack("<I", tensor.type_id)
            offset_offset = directory_start + len(out)
            out += struct.pack("<Q", rel)
            block_size, block_bytes, _ = tensor_size(tensor.dims, tensor.type_id)
            byte_size = 1
            for d in tensor.dims:
                byte_size *= d
            byte_size = byte_size // block_size * block_bytes
            tensor_layouts.append(TensorLayout(
                tensor.name, list(tensor.dims), tensor.type_id, rel, byte_size,
                name_len_offset, n_dims_offset, dims_offset, type_offset,
                offset_offset, 0,
            ))
        self.layout["tensors"] = tensor_layouts
        return bytes(out)

    def build(self) -> bytes:
        metadata = self._metadata_bytes()
        self.layout["metadata_size"] = len(metadata)
        alignment = 32
        # Extract alignment from metadata when present.
        for key, value in self.metadata.items():
            if key == "general.alignment":
                alignment = int(value[1])
                break
        self.layout["alignment"] = alignment

        # Payload offsets are computed before the directory length is known, but
        # each payload is aligned relative to the data section, so this is stable.
        rel = 0
        relative_offsets: list[int] = []
        payload = bytearray()
        for tensor in self.tensors:
            rel = align_up(rel, alignment)
            relative_offsets.append(rel)
            payload += b"\x00" * (rel - len(payload))
            if len(tensor.data) != tensor_byte_size(tensor.dims, tensor.type_id):
                raise GGUFError(f"tensor {tensor.name} payload size mismatch")
            payload += tensor.data
            rel += len(tensor.data)

        directory = self._tensor_directory_bytes(relative_offsets)
        data_offset = align_up(24 + len(metadata) + len(directory), alignment)
        prefix = (
            MAGIC
            + struct.pack("<I", VERSION)
            + struct.pack("<Q", len(self.tensors))
            + struct.pack("<Q", len(self.metadata))
            + metadata
            + directory
        )
        prefix += b"\x00" * (data_offset - len(prefix))
        blob = prefix + bytes(payload)
        for layout in self.layout["tensors"]:
            layout.absolute_offset = data_offset + layout.relative_offset
        self.layout["data_offset"] = data_offset
        self.layout["file_size"] = len(blob)
        self.layout["source_fingerprint"] = fnv1a64(prefix)
        return blob


def tensor_byte_size(dims: list[int], type_id: int) -> int:
    if not dims or any(d <= 0 for d in dims):
        raise GGUFError("invalid dimensions")
    block_size, block_bytes, _ = tensor_size(dims, type_id)
    elements = 1
    for d in dims:
        elements *= d
        if elements > (1 << 62):
            raise GGUFError("dimension overflow")
    if elements % block_size:
        raise GGUFError("row is not a whole number of quantisation blocks")
    return elements // block_size * block_bytes


def tensor_size(dims: list[int], type_id: int) -> tuple[int, int, str]:
    if type_id not in QUANT_TABLE:
        raise GGUFError(f"unknown quantisation type {type_id}")
    block_size, block_bytes, name = QUANT_TABLE[type_id]
    if dims and dims[0] % block_size:
        raise GGUFError(f"ne0={dims[0]} is not divisible by block size {block_size}")
    return block_size, block_bytes, name


def write_gguf(path: str | Path, metadata: dict[str, tuple[Any, ...]], tensors: list[TensorSpec]) -> tuple[bytes, dict[str, Any]]:
    writer = GGUFWriter(metadata, tensors)
    blob = writer.build()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_bytes(blob)
    return blob, writer.layout


def _read_value(blob: bytes, off: int, vtype: int, depth: int = 0) -> tuple[Any, int]:
    def need(n: int) -> None:
        if off + n > len(blob):
            raise GGUFError("truncated value")
    if vtype in (U8, I8, BOOL):
        need(1)
        fmt = {U8: "<B", I8: "<b", BOOL: "<?"}[vtype]
        return struct.unpack_from(fmt, blob, off)[0], off + 1
    fixed = {
        U16: ("<H", 2), I16: ("<h", 2), U32: ("<I", 4), I32: ("<i", 4),
        F32: ("<f", 4), U64: ("<Q", 8), I64: ("<q", 8), F64: ("<d", 8),
    }
    if vtype in fixed:
        fmt, size = fixed[vtype]
        need(size)
        return struct.unpack_from(fmt, blob, off)[0], off + size
    if vtype == STRING:
        need(8)
        n = struct.unpack_from("<Q", blob, off)[0]
        if n > len(blob):
            raise GGUFError("string too long")
        need(8 + n)
        return blob[off + 8:off + 8 + n].decode("utf-8"), off + 8 + n
    if vtype == ARRAY:
        if depth >= 4:
            raise GGUFError("array nesting too deep")
        need(8)
        etype, count = struct.unpack_from("<IQ", blob, off)
        if count > len(blob):
            raise GGUFError("array too long")
        pos = off + 12
        values = []
        for _ in range(count):
            value, pos = _read_value(blob, pos, etype, depth + 1)
            values.append(value)
        return values, pos
    raise GGUFError(f"invalid value type {vtype}")


def parse_gguf(blob: bytes) -> dict[str, Any]:
    if len(blob) < 24:
        raise GGUFError("truncated header")
    if blob[:4] != MAGIC:
        raise GGUFError("bad magic")
    version, tensor_count, metadata_count = struct.unpack_from("<IQQ", blob, 4)
    if version != VERSION:
        raise GGUFError("unsupported version")
    off = 24
    metadata: dict[str, Any] = {}
    for _ in range(metadata_count):
        if off + 8 > len(blob):
            raise GGUFError("truncated metadata key")
        key_len = struct.unpack_from("<Q", blob, off)[0]
        if key_len > len(blob):
            raise GGUFError("key too long")
        off += 8
        if off + key_len + 4 > len(blob):
            raise GGUFError("truncated metadata")
        key = blob[off:off + key_len].decode("utf-8")
        off += key_len
        vtype = struct.unpack_from("<I", blob, off)[0]
        off += 4
        value, off = _read_value(blob, off, vtype)
        metadata[key] = value
    tensors: list[dict[str, Any]] = []
    for _ in range(tensor_count):
        if off + 8 > len(blob):
            raise GGUFError("truncated tensor name")
        name_len = struct.unpack_from("<Q", blob, off)[0]
        if name_len > len(blob):
            raise GGUFError("tensor name too long")
        off += 8
        if off + name_len + 8 > len(blob):
            raise GGUFError("truncated tensor")
        name = blob[off:off + name_len].decode("utf-8")
        off += name_len
        n_dims = struct.unpack_from("<I", blob, off)[0]
        off += 4
        if n_dims > 8 or off + n_dims * 8 + 12 > len(blob):
            raise GGUFError("invalid tensor dimensions")
        dims = list(struct.unpack_from(f"<{n_dims}Q", blob, off))
        off += n_dims * 8
        type_id, rel = struct.unpack_from("<IQ", blob, off)
        off += 12
        tensors.append({"name": name, "dims": dims, "type": type_id, "offset": rel})
    alignment = int(metadata.get("general.alignment", 32))
    data_offset = align_up(off, alignment)
    result = []
    for t in tensors:
        size = tensor_byte_size(t["dims"], t["type"])
        start = data_offset + t["offset"]
        if start + size > len(blob):
            raise GGUFError("tensor out of bounds")
        row = dict(t)
        row["data"] = blob[start:start + size]
        row["absolute_offset"] = start
        row["byte_size"] = size
        result.append(row)
    return {
        "version": version,
        "metadata": metadata,
        "tensors": result,
        "data_offset": data_offset,
        "source_fingerprint": fnv1a64(blob[:data_offset]),
        "file_size": len(blob),
    }
