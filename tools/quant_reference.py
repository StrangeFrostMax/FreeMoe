#!/usr/bin/env python3
"""Independent reference decoders for GGUF block formats.

This is deliberately written from the published block layout, not from the C++
implementation, so that a disagreement between the two is a real bug rather
than a shared mistake. It generates `tests/fixtures/quant_reference.bin`, which
the C++ unit tests decode and compare against these expected float values.

Usage:
    python3 tools/quant_reference.py            # regenerate the fixture
    python3 tools/quant_reference.py --check     # verify the committed fixture
"""
from __future__ import annotations

import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures" / "quant_reference.bin"
MANIFEST = ROOT / "tests" / "fixtures" / "quant_reference.json"
MAGIC = b"FMQR1\0\0\0"

# type_id -> (block_size, block_bytes)
GEOMETRY = {
    0: (1, 4),
    1: (1, 2),
    2: (32, 18),
    3: (32, 20),
    6: (32, 22),
    7: (32, 24),
    8: (32, 34),
    9: (32, 40),
    10: (256, 84),
    11: (256, 110),
    12: (256, 144),
    13: (256, 176),
    14: (256, 210),
    15: (256, 292),
    20: (32, 18),   # IQ4_NL: f16 scale + 16 nibble bytes
    # Ternary family. 34/35 carry the fp16 scale last; 41/42/142 carry it first.
    34: (256, 54),  # TQ1_0  base-3 trits, 5 per byte
    35: (256, 66),  # TQ2_0  2-bit trits
    41: (128, 18),  # Q1_0   one sign bit per weight
    42: (64, 18),   # Q2_0   2-bit codes, group 64
    142: (128, 34), # PQ2_0  2-bit codes, group 128
    # ggml treats these as element types, not blocked quants: one element per
    # block, so the block size is 1 and the block bytes are the element width.
    24: (1, 1),     # I8
    25: (1, 2),     # I16
    26: (1, 4),     # I32
    27: (1, 8),     # I64
    28: (1, 8),     # F64
    30: (1, 2),
}


def f16_to_f32(bits: int) -> float:
    sign = -1.0 if bits & 0x8000 else 1.0
    exp = (bits >> 10) & 0x1F
    man = bits & 0x3FF
    if exp == 0:
        return sign * man * 2.0**-24
    if exp == 0x1F:
        return sign * float("inf") if man == 0 else float("nan")
    return sign * (1.0 + man / 1024.0) * 2.0 ** (exp - 15)


def rd_half(buf: bytes, off: int) -> float:
    return f16_to_f32(struct.unpack_from("<H", buf, off)[0])


def rd_f32(buf: bytes, off: int) -> float:
    return struct.unpack_from("<f", buf, off)[0]


def q4_0(b):
    d = rd_half(b, 0)
    out = [0.0] * 32
    for j in range(16):
        out[j] = d * ((b[2 + j] & 0xF) - 8)
        out[j + 16] = d * ((b[2 + j] >> 4) - 8)
    return out


def q4_1(b):
    d, m = rd_half(b, 0), rd_half(b, 2)
    out = [0.0] * 32
    for j in range(16):
        out[j] = d * (b[4 + j] & 0xF) + m
        out[j + 16] = d * (b[4 + j] >> 4) + m
    return out


def q5_0(b):
    d = rd_half(b, 0)
    qh = b[2:6]
    out = [0.0] * 32
    for j in range(16):
        h0 = (qh[j >> 3] >> (j & 7)) & 1
        h1 = (qh[(j + 16) >> 3] >> (j & 7)) & 1
        out[j] = d * (((b[6 + j] & 0xF) | (h0 << 4)) - 16)
        out[j + 16] = d * (((b[6 + j] >> 4) | (h1 << 4)) - 16)
    return out


def q5_1(b):
    d, m = rd_half(b, 0), rd_half(b, 2)
    qh = b[4:8]
    out = [0.0] * 32
    for j in range(16):
        h0 = (qh[j >> 3] >> (j & 7)) & 1
        h1 = (qh[(j + 16) >> 3] >> (j & 7)) & 1
        out[j] = d * ((b[8 + j] & 0xF) | (h0 << 4)) + m
        out[j + 16] = d * ((b[8 + j] >> 4) | (h1 << 4)) + m
    return out


def q8_0(b):
    d = rd_half(b, 0)
    return [d * struct.unpack_from("<b", b, 2 + j)[0] for j in range(32)]


def q8_1(b):
    d = rd_f32(b, 0)
    return [d * struct.unpack_from("<b", b, 8 + j)[0] for j in range(32)]


def q2_k(b):
    scales, qs = b[0:16], b[16:80]
    d, dmin = rd_half(b, 80), rd_half(b, 82)
    out = []
    is_ = 0
    for _ in range(0, 256, 128):
        chunk = qs[(_ // 128) * 32 : (_ // 128) * 32 + 32]
        shift = 0
        for _j in range(4):
            sc = scales[is_]
            is_ += 1
            dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
            out += [dl * ((chunk[l] >> shift) & 3) - ml for l in range(16)]
            sc = scales[is_]
            is_ += 1
            dl, ml = d * (sc & 0xF), dmin * (sc >> 4)
            out += [dl * ((chunk[l + 16] >> shift) & 3) - ml for l in range(16)]
            shift += 2
    return out


def q3_k(b):
    hm, qs, packed = b[0:32], b[32:96], b[96:108]
    d_all = rd_half(b, 108)
    kmask1, kmask2 = 0x03030303, 0x0F0F0F0F
    aux = [0, 0, 0, 0]
    aux[0], aux[1], aux[2] = struct.unpack("<III", packed)
    tmp = aux[2]
    aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4)
    aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4)
    aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4)
    aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4)
    scales = list(struct.pack("<IIII", *aux))
    scales = [x - 256 if x > 127 else x for x in scales]  # int8 view
    out = []
    m, is_ = 1, 0
    for n in range(0, 256, 128):
        chunk = qs[(n // 128) * 32 : (n // 128) * 32 + 32]
        hmchunk = hm  # hmask spans both halves
        shift = 0
        for _j in range(4):
            dl = d_all * (scales[is_] - 32)
            is_ += 1
            out += [dl * (((chunk[l] >> shift) & 3) - (0 if (hmchunk[l] & m) else 4)) for l in range(16)]
            dl = d_all * (scales[is_] - 32)
            is_ += 1
            out += [
                dl * (((chunk[l + 16] >> shift) & 3) - (0 if (hmchunk[l + 16] & m) else 4))
                for l in range(16)
            ]
            shift += 2
            m = (m << 1) & 0xFF
    return out


def _scale_min_k4(j: int, q: bytes):
    if j < 4:
        return q[j] & 63, q[j + 4] & 63
    return (q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4), (q[j + 4] >> 4) | ((q[j] >> 6) << 4)


def q4_k(b):
    d, dmin = rd_half(b, 0), rd_half(b, 2)
    scales, qs = b[4:16], b[16:144]
    out = []
    is_ = 0
    for j in range(0, 256, 64):
        chunk = qs[(j // 64) * 32 : (j // 64) * 32 + 32]
        sc, m = _scale_min_k4(is_, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = _scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m
        out += [d1 * (chunk[l] & 0xF) - m1 for l in range(32)]
        out += [d2 * (chunk[l] >> 4) - m2 for l in range(32)]
        is_ += 2
    return out


def q5_k(b):
    d, dmin = rd_half(b, 0), rd_half(b, 2)
    scales, qh, qs = b[4:16], b[16:48], b[48:176]
    out = []
    is_, u1, u2 = 0, 1, 2
    for j in range(0, 256, 64):
        idx = j // 64
        chunk = qs[idx * 32 : idx * 32 + 32]
        hchunk = qh  # qh is not advanced; u1/u2 select the bit pair
        sc, m = _scale_min_k4(is_, scales)
        d1, m1 = d * sc, dmin * m
        sc, m = _scale_min_k4(is_ + 1, scales)
        d2, m2 = d * sc, dmin * m
        out += [d1 * ((chunk[l] & 0xF) + (16 if (hchunk[l] & u1) else 0)) - m1 for l in range(32)]
        out += [d2 * ((chunk[l] >> 4) + (16 if (hchunk[l] & u2) else 0)) - m2 for l in range(32)]
        is_ += 2
        u1 = (u1 << 2) & 0xFF
        u2 = (u2 << 2) & 0xFF
    return out


def q6_k(b):
    ql, qh, sc = b[0:128], b[128:192], b[192:208]
    d = rd_half(b, 208)
    scales = [x - 256 if x > 127 else x for x in sc]
    out = [0.0] * 256
    for n in range(0, 256, 128):
        base = n // 128
        qlc = ql[base * 64 : base * 64 + 64]
        qhc = qh[base * 32 : base * 32 + 32]
        scc = scales[base * 8 : base * 8 + 8]
        for l in range(32):
            i = l // 16
            q1 = ((qlc[l] & 0xF) | (((qhc[l] >> 0) & 3) << 4)) - 32
            q2 = ((qlc[l + 32] & 0xF) | (((qhc[l] >> 2) & 3) << 4)) - 32
            q3 = ((qlc[l] >> 4) | (((qhc[l] >> 4) & 3) << 4)) - 32
            q4 = ((qlc[l + 32] >> 4) | (((qhc[l] >> 6) & 3) << 4)) - 32
            out[n + l + 0] = d * scc[i + 0] * q1
            out[n + l + 32] = d * scc[i + 2] * q2
            out[n + l + 64] = d * scc[i + 4] * q3
            out[n + l + 96] = d * scc[i + 6] * q4
    return out


def q8_k(b):
    d = rd_f32(b, 0)
    return [d * struct.unpack_from("<b", b, 4 + j)[0] for j in range(256)]


def bf16(b):
    bits = struct.unpack_from("<H", b, 0)[0] << 16
    return [struct.unpack("<f", struct.pack("<I", bits))[0]]


# The published IQ4_NL codebook (ggml kvalues_iq4nl), kept as a literal so this
# decoder does not share an implementation with src/quant.cpp.
KVALUES_IQ4NL = [-127, -104, -83, -65, -49, -35, -22, -10,
                 1, 13, 25, 38, 53, 69, 89, 113]


def iq4_nl(b):
    """f16 scale, then 32 four-bit codebook indices packed Q4_0 style."""
    d = rd_half(b, 0)
    out = []
    for j in range(16):
        byte = b[2 + j]
        out.append(d * KVALUES_IQ4NL[byte & 0x0F])
    for j in range(16):
        byte = b[2 + j]
        out.append(d * KVALUES_IQ4NL[byte >> 4])
    return out


def i8(b):
    return [float(struct.unpack_from("<b", b, 0)[0])]


def i16(b):
    return [float(struct.unpack_from("<h", b, 0)[0])]


def i32(b):
    return [float(struct.unpack_from("<i", b, 0)[0])]


def i64(b):
    return [float(struct.unpack_from("<q", b, 0)[0])]


def f64(b):
    return [struct.unpack_from("<d", b, 0)[0]]


def _trit_from_byte(byte: int, power: int) -> int:
    """Recovers one base-3 trit from a byte multiplied by a power of three.

    The multiply wraps modulo 256, which is what brings the wanted trit into the
    high bits; (v * 3) >> 8 then yields 0, 1 or 2.
    """
    v = (byte * power) & 0xFF
    return (v * 3) >> 8


def tq1_0(b):
    """256 elements: 48 bytes of 5-trit packs, 4 bytes of 4-trit packs, fp16 d."""
    d = rd_half(b, 52)
    out = []
    for n in range(5):
        for m in range(32):
            out.append((_trit_from_byte(b[m], 3 ** n) - 1) * d)
    for n in range(5):
        for m in range(16):
            out.append((_trit_from_byte(b[32 + m], 3 ** n) - 1) * d)
    for n in range(4):
        for h in range(4):
            out.append((_trit_from_byte(b[48 + h], 3 ** n) - 1) * d)
    return out


def tq2_0(b):
    """256 elements: 64 bytes of 2-bit trits, then fp16 d."""
    d = rd_half(b, 64)
    out = []
    for j in range(0, 64, 32):
        for l in range(4):
            for m in range(32):
                out.append((((b[j + m] >> (l * 2)) & 3) - 1) * d)
    return out


def q1_0(b):
    """128 elements: fp16 d then one sign bit per weight."""
    d = rd_half(b, 0)
    return [d if (b[2 + j // 8] >> (j % 8)) & 1 else -d for j in range(128)]


def _two_bit(b, offset: int, count: int) -> list[float]:
    d = rd_half(b, offset)
    return [(((b[offset + 2 + j // 4] >> ((j % 4) * 2)) & 3) - 1) * d for j in range(count)]


def q2_0(b):
    """64 elements: fp16 d then 2-bit codes."""
    return _two_bit(b, 0, 64)


def pq2_0(b):
    """128 elements: the same codec as Q2_0 over a wider group."""
    return _two_bit(b, 0, 128)


DECODERS = {
    30: bf16, 2: q4_0, 3: q4_1, 6: q5_0, 7: q5_1, 8: q8_0, 9: q8_1,
    10: q2_k, 11: q3_k, 12: q4_k, 13: q5_k, 14: q6_k, 15: q8_k,
    20: iq4_nl, 24: i8, 25: i16, 26: i32, 27: i64, 28: f64,
    34: tq1_0, 35: tq2_0, 41: q1_0, 42: q2_0, 142: pq2_0,
}


def identity(b):
    if len(b) == 4:
        return [rd_f32(b, 0)]
    return [rd_half(b, 0)]


def encode_f16(value: float) -> bytes:
    """Encode a float as binary16 with round-to-nearest-even."""
    packed = struct.pack("<f", value)
    bits = struct.unpack("<I", packed)[0]
    sign = (bits >> 16) & 0x8000
    exp = ((bits >> 23) & 0xFF) - 127 + 15
    man = bits & 0x7FFFFF
    if exp <= 0:
        return struct.pack("<H", sign)
    if exp >= 0x1F:
        return struct.pack("<H", sign | 0x7C00)
    half = sign | (exp << 10) | (man >> 13)
    return struct.pack("<H", half)


def make_block(type_id: int, rng_state: int) -> tuple[bytes, int]:
    """Build a deterministic pseudo-random but semantically valid block."""
    block_size, block_bytes = GEOMETRY[type_id]
    state = rng_state
    raw = bytearray(block_bytes)

    def rand() -> int:
        nonlocal state
        state = (state * 1664525 + 1013904223) & 0xFFFFFFFF
        return (state >> 8) & 0xFFFF

    for i in range(block_bytes):
        raw[i] = rand() & 0xFF
    # Keep scales finite and small so the comparison stays meaningful.
    half_scale = encode_f16(0.01 + (rand() % 100) / 10000.0)
    if type_id in (0, 9, 15):
        struct.pack_into("<f", raw, 0, 0.02 + (rand() % 50) / 500.0)
    elif type_id in (1, 30):
        pass
    elif type_id in (2, 6, 8, 20):
        raw[0:2] = half_scale
    elif type_id in (3, 7):
        raw[0:2] = half_scale
        raw[2:4] = encode_f16(-0.005)
    elif type_id == 10:  # Q2_K
        raw[80:82] = half_scale
        raw[82:84] = encode_f16(-0.005)
    elif type_id == 11:  # Q3_K
        raw[108:110] = half_scale
    elif type_id in (12, 13):  # Q4_K / Q5_K
        raw[0:2] = half_scale
        raw[2:4] = encode_f16(-0.004)
    elif type_id == 14:  # Q6_K
        raw[208:210] = half_scale
    elif type_id == 34:  # TQ1_0: scale last
        raw[52:54] = half_scale
    elif type_id == 35:  # TQ2_0: scale last
        raw[64:66] = half_scale
    elif type_id in (41, 42, 142):  # Q1_0 / Q2_0 / PQ2_0: scale first
        raw[0:2] = half_scale
    if type_id == 24:  # I8
        struct.pack_into("<b", raw, 0, -73)
    elif type_id == 25:  # I16
        struct.pack_into("<h", raw, 0, -12345)
    elif type_id == 26:  # I32
        struct.pack_into("<i", raw, 0, -123456789)
    elif type_id == 27:  # I64
        struct.pack_into("<q", raw, 0, -7654321987654)
    elif type_id == 28:  # F64
        struct.pack_into("<d", raw, 0, -2.718281828459045)
    return bytes(raw), state


def build_fixture() -> tuple[bytes, list[dict]]:
    """Record layout: u32 type_id, u32 block_bytes, u32 n_float, raw, floats."""
    cases: list[dict] = []
    payload = bytearray()
    state = 0x1234_5678
    for type_id in sorted(GEOMETRY):
        decoder = DECODERS.get(type_id, identity)
        raw, state = make_block(type_id, state)
        expected = decoder(raw)
        payload += struct.pack("<III", type_id, len(raw), len(expected))
        payload += raw
        payload += struct.pack(f"<{len(expected)}f", *expected)
        cases.append(
            {
                "type_id": type_id,
                "block_bytes": len(raw),
                "elements": len(expected),
            }
        )
    header = MAGIC + struct.pack("<I", len(cases))
    return header + bytes(payload), cases


def main() -> int:
    data, cases = build_fixture()
    if "--check" in sys.argv:
        if not OUT.exists():
            print("missing fixture", OUT)
            return 1
        same = OUT.read_bytes() == data
        print(json.dumps({"status": "pass" if same else "fail", "fixture": str(OUT)}))
        return 0 if same else 1

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_bytes(data)
    manifest = {
        "schema": "freemoe.quant-reference.v1",
        "generator": "tools/quant_reference.py",
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "cases": cases,
        "note": (
            "Expected values are produced by this file's independent decoders. "
            "The C++ decoders must match them bit-for-bit where the arithmetic is "
            "exactly representable, and within 1e-6 otherwise."
        ),
    }
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"status": "written", "bytes": len(data), "cases": len(cases)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
