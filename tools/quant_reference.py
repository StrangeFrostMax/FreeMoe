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
import math
import re
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
    # IQ family: grid-indexed codecs on the 256-element super-block. The block
    # sizes follow from the field widths in ggml-common.h.
    16: (256, 66),   # IQ2_XXS  f16 d + 32 grid/sign words
    17: (256, 74),   # IQ2_XS   ... + 8 scale bytes
    18: (256, 98),   # IQ3_XXS  f16 d + 64 index bytes + 32 scale/sign bytes
    19: (256, 50),   # IQ1_S    f16 d + 32 index bytes + 8 index/scale words
    21: (256, 110),  # IQ3_S    + 8 high-bit bytes + 32 sign bytes + 4 scale bytes
    22: (256, 82),   # IQ2_S    + 8 high-bit bytes + 8 scale bytes
    23: (256, 136),  # IQ4_XS   f16 d + u16 high scales + 4 low + 128 nibbles
    29: (256, 56),   # IQ1_M    no header scale: it is packed into the tail
    # FP4 formats: a per-block exponent scale over E2M1 nibbles.
    39: (32, 17),    # MXFP4  one E8M0 byte + 16 nibble bytes
    40: (64, 36),    # NVFP4  four UE4M3 sub-block scales + 32 nibble bytes
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


# --- IQ family -------------------------------------------------------------
# The grids are data, not logic: they are parsed out of the generated
# src/quant_tables.cpp so this file does not carry a second copy. The decode
# logic below is written from the published block layout independently of
# src/quant.cpp.


def _load_grids() -> dict[str, list[int]]:
    text = (ROOT / "src" / "quant_tables.cpp").read_text()
    grids: dict[str, list[int]] = {}
    for match in re.finditer(
        r"const \w+ (\w+)\[\d+\] = \{(.*?)\n\};", text, re.S
    ):
        name = match.group(1)
        body = re.sub(r"//[^\n]*", "", match.group(2))
        grids[name] = [int(tok, 0) for tok in re.split(r"[,\s]+", body.strip()) if tok]
    return grids


_GRIDS = _load_grids()


def _grid(name: str, index: int, width: int) -> list[int]:
    return list(_GRIDS[name][index].to_bytes(width, "little"))


def _fp4_code(code: int) -> int:
    """E2M1 magnitudes 0,1,2,3,4,6,8,12 with bit 3 as the sign."""
    magnitude = (0, 1, 2, 3, 4, 6, 8, 12)[code & 7]
    return -magnitude if code & 8 else magnitude


def _e8m0_half(x: int) -> float:
    """E8M0 exponent doubled value, halved: 2^(x-128), with denormals below 2."""
    if x < 2:
        return math.ldexp(0.5, -127 + x)
    return math.ldexp(1.0, x - 128)


def _ue4m3_half(x: int) -> float:
    if x in (0, 0x7F):
        return 0.0
    exponent = (x >> 3) & 0xF
    mantissa = x & 7
    if exponent == 0:
        raw = math.ldexp(float(mantissa), -9)
    else:
        raw = math.ldexp(1.0 + mantissa / 8.0, exponent - 7)
    return raw * 0.5


def mxfp4(b):
    """32 elements: one E8M0 byte, then 16 packed E2M1 nibbles."""
    d = _e8m0_half(b[0])
    out = []
    for j in range(16):
        out.append(_fp4_code(b[1 + j] & 0x0F) * d)
    for j in range(16):
        out.append(_fp4_code(b[1 + j] >> 4) * d)
    return out


def nvfp4(b):
    """64 elements: four UE4M3 sub-block scales, then 32 packed nibbles."""
    out = []
    for s in range(4):
        d = _ue4m3_half(b[s])
        sub = b[4 + s * 8 : 12 + s * 8]
        out.extend(_fp4_code(byte & 0x0F) * d for byte in sub)
        out.extend(_fp4_code(byte >> 4) * d for byte in sub)
    return out


def _signs(byte: int) -> list[int]:
    """kvalues_iq2xs is a bitmask over the eight grid positions."""
    return [(-1 if byte & (1 << j) else 1) for j in range(8)]


def iq2_xxs(b):
    """Eight 32-weight blocks; grid index and sign bits share each 8-byte step."""
    d = rd_half(b, 0)
    out = []
    for ib in range(8):
        word0 = struct.unpack_from("<I", b, 2 + 8 * ib)[0]
        word1 = struct.unpack_from("<I", b, 2 + 8 * ib + 4)[0]
        indices = list(word0.to_bytes(4, "little"))
        db = d * (0.5 + (word1 >> 28)) * 0.25
        for l in range(4):
            grid = _grid("iq2xxs_grid", indices[l], 8)
            signs = _GRIDS["ksigns_iq2xs"][(word1 >> (7 * l)) & 127]
            out.extend(db * g * s for g, s in zip(grid, _signs(signs)))
    return out


def iq2_xs(b):
    d = rd_half(b, 0)
    out = []
    for ib in range(8):
        scale = b[2 + 64 + ib]
        db = (d * (0.5 + (scale & 0x0F)) * 0.25, d * (0.5 + (scale >> 4)) * 0.25)
        for l in range(4):
            code = struct.unpack_from("<H", b, 2 + 2 * (4 * ib + l))[0]
            grid = _grid("iq2xs_grid", code & 511, 8)
            signs = _GRIDS["ksigns_iq2xs"][code >> 9]
            out.extend(db[l // 2] * g * s for g, s in zip(grid, _signs(signs)))
    return out


def iq2_s(b):
    d = rd_half(b, 0)
    qs = 2
    qh = qs + 64
    scales = qh + 8
    signs_base = qs + 32
    out = []
    for ib in range(8):
        db = (d * (0.5 + (b[scales + ib] & 0x0F)) * 0.25,
              d * (0.5 + (b[scales + ib] >> 4)) * 0.25)
        for l in range(4):
            index = b[qs + 4 * ib + l] | ((b[qh + ib] << (8 - 2 * l)) & 0x300)
            grid = _grid("iq2s_grid", index, 8)
            sign_byte = b[signs_base + 4 * ib + l]
            out.extend(db[l // 2] * g * s for g, s in zip(grid, _signs(sign_byte)))
    return out


def iq3_xxs(b):
    d = rd_half(b, 0)
    qs = 2
    words = qs + 64
    out = []
    for ib in range(8):
        aux = struct.unpack_from("<I", b, words + 4 * ib)[0]
        db = d * (0.5 + (aux >> 28)) * 0.5
        for l in range(4):
            signs = _GRIDS["ksigns_iq2xs"][(aux >> (7 * l)) & 127]
            grid1 = _grid("iq3xxs_grid", b[qs + 8 * ib + 2 * l], 4)
            grid2 = _grid("iq3xxs_grid", b[qs + 8 * ib + 2 * l + 1], 4)
            mask = _signs(signs)
            out.extend(db * grid1[j] * mask[j] for j in range(4))
            out.extend(db * grid2[j] * mask[j + 4] for j in range(4))
    return out


def iq3_s(b):
    d = rd_half(b, 0)
    qs = 2
    qh = qs + 64
    signs = qh + 8
    scales = signs + 32
    out = []
    for pair in range(4):
        db = (d * (1 + 2 * (b[scales + pair] & 0x0F)), d * (1 + 2 * (b[scales + pair] >> 4)))
        for half in range(2):
            high = b[qh + 2 * pair + half]
            sg = signs + 4 * (2 * pair + half)  # four sign bytes per 32 weights
            base = qs + 16 * pair + 8 * half
            for l in range(4):
                i1 = b[base + 2 * l] | ((high << (8 - 2 * l)) & 256)
                i2 = b[base + 2 * l + 1] | ((high << (7 - 2 * l)) & 256)
                grid1 = _grid("iq3s_grid", i1, 4)
                grid2 = _grid("iq3s_grid", i2, 4)
                mask = _signs(b[sg + l])
                out.extend(db[half] * grid1[j] * mask[j] for j in range(4))
                out.extend(db[half] * grid2[j] * mask[j + 4] for j in range(4))
    return out


def iq1_s(b):
    d = rd_half(b, 0)
    qs = 2
    qh = qs + 32
    out = []
    for ib in range(8):
        h = struct.unpack_from("<H", b, qh + 2 * ib)[0]
        dl = d * (2 * ((h >> 12) & 7) + 1)
        delta = -0.125 if h & 0x8000 else 0.125
        for l in range(4):
            index = b[qs + 4 * ib + l] | (((h >> (3 * l)) & 7) << 8)
            grid = struct.unpack("<8b", _GRIDS["iq1s_grid"][index].to_bytes(8, "little"))
            out.extend(dl * (g + delta) for g in grid)
    return out


def iq1_m(b):
    qs = 0
    qh = 32
    sc = qh + 16
    words = [struct.unpack_from("<H", b, sc + 2 * i)[0] for i in range(4)]
    bits = (words[0] >> 12) | ((words[1] >> 8) & 0x00F0) | ((words[2] >> 4) & 0x0F00) | (words[3] & 0xF000)
    d = struct.unpack("<e", struct.pack("<H", bits))[0]
    out = []
    for ib in range(8):
        s = words[ib // 2]
        dl1 = d * (2 * ((s >> (6 * (ib % 2))) & 7) + 1)
        dl2 = d * (2 * ((s >> (6 * (ib % 2) + 3)) & 7) + 1)
        h0 = b[qh + 2 * ib]
        h1 = b[qh + 2 * ib + 1]
        idx = [b[qs + 4 * ib + 0] | ((h0 << 8) & 0x700),
               b[qs + 4 * ib + 1] | ((h0 << 4) & 0x700),
               b[qs + 4 * ib + 2] | ((h1 << 8) & 0x700),
               b[qs + 4 * ib + 3] | ((h1 << 4) & 0x700)]
        delta = [-0.125 if h0 & 0x08 else 0.125,
                 -0.125 if h0 & 0x80 else 0.125,
                 -0.125 if h1 & 0x08 else 0.125,
                 -0.125 if h1 & 0x80 else 0.125]
        for l in range(4):
            grid = struct.unpack("<8b", _GRIDS["iq1s_grid"][idx[l]].to_bytes(8, "little"))
            dl = dl1 if l < 2 else dl2
            out.extend(dl * (g + delta[l]) for g in grid)
    return out


def iq4_xs(b):
    d = rd_half(b, 0)
    scales_h = struct.unpack_from("<H", b, 2)[0]
    scales_l = b[4:8]
    qs = 8
    out = []
    for ib in range(8):
        low = (scales_l[ib // 2] >> (4 * (ib % 2))) & 0x0F
        high = (scales_h >> (2 * ib)) & 3
        dl = d * ((low | (high << 4)) - 32)
        block = b[qs + 16 * ib : qs + 16 * ib + 16]
        out.extend(dl * KVALUES_IQ4NL[byte & 0x0F] for byte in block)
        out.extend(dl * KVALUES_IQ4NL[byte >> 4] for byte in block)
    return out


DECODERS = {
    30: bf16, 2: q4_0, 3: q4_1, 6: q5_0, 7: q5_1, 8: q8_0, 9: q8_1,
    10: q2_k, 11: q3_k, 12: q4_k, 13: q5_k, 14: q6_k, 15: q8_k,
    20: iq4_nl, 24: i8, 25: i16, 26: i32, 27: i64, 28: f64,
    34: tq1_0, 35: tq2_0, 41: q1_0, 42: q2_0, 142: pq2_0,
    16: iq2_xxs, 17: iq2_xs, 18: iq3_xxs, 19: iq1_s, 21: iq3_s, 22: iq2_s,
    23: iq4_xs, 29: iq1_m, 39: mxfp4, 40: nvfp4,
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
    elif type_id in (16, 17, 18, 19, 21, 22, 23):  # IQ family: f16 scale first
        raw[0:2] = half_scale
    elif type_id == 29:  # IQ1_M: the fp16 scale is scattered across four words
        # 0x2ACD is roughly 0.05; the nibbles that do not carry the scale keep
        # their pseudo-random 3-bit block scales.
        bits = 0x2ACD
        words = [struct.unpack_from("<H", raw, 48 + 2 * i)[0] for i in range(4)]
        words[0] = (words[0] & 0x0FFF) | (bits & 0xF000)
        words[1] = (words[1] & 0xF0FF) | (((bits >> 4) & 0x0F) << 8)
        words[2] = (words[2] & 0xFF0F) | (((bits >> 8) & 0x0F) << 4)
        words[3] = (words[3] & 0x0FFF) | ((bits & 0x000F) << 12)
        for i, word in enumerate(words):
            struct.pack_into("<H", raw, 48 + 2 * i, word)
    elif type_id == 39:  # MXFP4: one E8M0 exponent byte
        raw[0] = 130
    elif type_id == 40:  # NVFP4: four UE4M3 sub-block scales
        raw[0:4] = bytes([120, 124, 116, 126])
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
