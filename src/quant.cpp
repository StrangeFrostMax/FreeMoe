// Block-format decoders for the GGUF tensor types used by real checkpoints.
//
// Every decoder here is a direct transcription of the published block layout.
// None of them "round" or approximate: a value either decodes exactly as the
// format defines it or the caller gets a typed error.
//
// Scalar reference implementations only. They are the oracle that faster
// kernels are checked against, and they never claim execution support for a
// type whose semantics are not implemented here.

#include "fm/quant.hpp"
#include "fm/quant_tables.hpp"

#include <string>

#include <cmath>
#include <cstring>

namespace fm {
namespace {

// IEEE 754 binary16 -> binary32.
float fp16_to_fp32(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000u) << 16;
    uint32_t exponent = (h >> 10) & 0x1Fu;
    uint32_t mantissa = h & 0x3FFu;
    uint32_t bits;
    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            exponent = 127 - 15 + 1;
            while ((mantissa & 0x400u) == 0) {
                mantissa <<= 1;
                --exponent;
            }
            mantissa &= 0x3FFu;
            bits = sign | (exponent << 23) | (mantissa << 13);
        }
    } else if (exponent == 0x1F) {
        bits = sign | 0x7F800000u | (mantissa << 13);
    } else {
        bits = sign | ((exponent + 127 - 15) << 23) | (mantissa << 13);
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

uint16_t read_u16(const uint8_t *p) {
    uint16_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

float read_f32(const uint8_t *p) {
    float v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// Six-bit scale/min unpacking shared by Q4_K and Q5_K.
void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = static_cast<uint8_t>(q[j] & 63);
        *m = static_cast<uint8_t>(q[j + 4] & 63);
    } else {
        *d = static_cast<uint8_t>((q[j + 4] & 0x0F) | ((q[j - 4] >> 6) << 4));
        *m = static_cast<uint8_t>((q[j + 4] >> 4) | ((q[j] >> 6) << 4));
    }
}

uint32_t read_u32(const uint8_t *p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

// E8M0 exponent -> float, halved. The halving is part of the format: MXFP4's
// 4-value codes are stored doubled, so the scale absorbs the factor of two.
float e8m0_to_fp32_half(uint8_t x) {
    uint32_t bits;
    if (x < 2) {
        // 2^(-128) and 2^(-127) as denormals.
        bits = 0x00200000u << x;
    } else {
        bits = static_cast<uint32_t>(x - 1) << 23;
    }
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

// UE4M3 (unsigned, 4 exponent bits with bias 7, 3 mantissa bits), halved for the
// same reason as MXFP4. 0x00 and 0x7F decode to zero.
float ue4m3_to_fp32(uint8_t x) {
    if (x == 0 || x == 0x7F) return 0.0f;
    const int exponent = (x >> 3) & 0xF;
    const int mantissa = x & 0x7;
    const float raw = exponent == 0
                          ? std::ldexp(static_cast<float>(mantissa), -9)
                          : std::ldexp(1.0f + static_cast<float>(mantissa) / 8.0f, exponent - 7);
    return raw * 0.5f;
}

// The 16 four-bit codes of the FP4 formats, in sign-magnitude order: the low
// three bits are the magnitude index and bit 3 is the sign.
const int8_t kvalues_fp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// Grid lookup for the IQ formats. The grid entry is a packed little-endian word
// whose bytes are the elements in order, so the entry is copied out and read as
// bytes. It is copied into the caller's buffer rather than a shared one because
// several decoders read two grid entries at once.
void load_grid(uint64_t entry, uint8_t out[8]) { std::memcpy(out, &entry, 8); }

void load_grid32(uint32_t entry, uint8_t out[4]) { std::memcpy(out, &entry, 4); }

} // namespace

float fp16_to_f32(uint16_t value) { return fp16_to_fp32(value); }

float bf16_to_f32(uint16_t value) {
    const uint32_t bits = static_cast<uint32_t>(value) << 16;
    float out;
    std::memcpy(&out, &bits, sizeof(out));
    return out;
}

std::string quant_geometry_or_name(uint32_t type_id) {
    const auto &table = quantisation_table();
    auto it = table.find(type_id);
    if (it != table.end()) return it->second.name;
    return "TYPE_" + std::to_string(type_id);
}

const QuantGeometry &quant_geometry(uint32_t type_id) {
    const auto &table = quantisation_table();
    auto it = table.find(type_id);
    if (it == table.end()) {
        // Types that exist in a fork but not in the GGUF type table cannot be
        // bounded without inventing a block layout, so they are refused by name
        // rather than by number. A bare "unknown tensor type 36" sends the
        // reader looking in the wrong place.
        static const std::map<uint32_t, std::string> external = {
            {36, "I2_S is bitnet.cpp's private ternary type and is not part of the GGUF "
                 "type table; re-quantise the checkpoint to TQ1_0 or TQ2_0, which this "
                 "build decodes"},
        };
        auto external_it = external.find(type_id);
        if (external_it != external.end())
            fail(ErrorCode::UNSUPPORTED_QUANTISATION,
                 "tensor type " + std::to_string(type_id) + ": " + external_it->second);
        fail(ErrorCode::UNSUPPORTED_QUANTISATION,
             "unknown tensor type " + std::to_string(type_id));
    }
    return it->second;
}

bool quant_is_k_block(uint32_t type_id) {
    return type_id >= 10 && type_id <= 15;
}

void dequantize_block(uint32_t type_id, const uint8_t *src, float *dst) {
    switch (type_id) {
    case 0: // F32
        std::memcpy(dst, src, 4 * 1);
        return;
    case 1: // F16
        dst[0] = fp16_to_fp32(read_u16(src));
        return;
    case 30: { // BF16
        const uint32_t bits = static_cast<uint32_t>(read_u16(src)) << 16;
        float out;
        std::memcpy(&out, &bits, sizeof(out));
        dst[0] = out;
        return;
    }
    case 2: { // Q4_0: half d, 16 packed nibbles
        const float d = fp16_to_fp32(read_u16(src));
        const uint8_t *qs = src + 2;
        for (int j = 0; j < 16; ++j) {
            dst[j] = d * static_cast<float>(static_cast<int>(qs[j] & 0x0F) - 8);
            dst[j + 16] = d * static_cast<float>(static_cast<int>(qs[j] >> 4) - 8);
        }
        return;
    }
    case 3: { // Q4_1: half d, half m, 16 packed nibbles
        const float d = fp16_to_fp32(read_u16(src));
        const float m = fp16_to_fp32(read_u16(src + 2));
        const uint8_t *qs = src + 4;
        for (int j = 0; j < 16; ++j) {
            dst[j] = d * static_cast<float>(qs[j] & 0x0F) + m;
            dst[j + 16] = d * static_cast<float>(qs[j] >> 4) + m;
        }
        return;
    }
    case 6: { // Q5_0: half d, 4 high-bit bytes, 16 packed nibbles
        const float d = fp16_to_fp32(read_u16(src));
        const uint8_t *qh = src + 2;
        const uint8_t *qs = src + 6;
        for (int j = 0; j < 16; ++j) {
            const uint8_t h0 = static_cast<uint8_t>((qh[j >> 3] >> (j & 7)) & 1);
            const uint8_t h1 = static_cast<uint8_t>((qh[(j + 16) >> 3] >> (j & 7)) & 1);
            dst[j] = d * static_cast<float>(static_cast<int>((qs[j] & 0x0F) | (h0 << 4)) - 16);
            dst[j + 16] =
                d * static_cast<float>(static_cast<int>((qs[j] >> 4) | (h1 << 4)) - 16);
        }
        return;
    }
    case 7: { // Q5_1
        const float d = fp16_to_fp32(read_u16(src));
        const float m = fp16_to_fp32(read_u16(src + 2));
        const uint8_t *qh = src + 4;
        const uint8_t *qs = src + 8;
        for (int j = 0; j < 16; ++j) {
            const uint8_t h0 = static_cast<uint8_t>((qh[j >> 3] >> (j & 7)) & 1);
            const uint8_t h1 = static_cast<uint8_t>((qh[(j + 16) >> 3] >> (j & 7)) & 1);
            dst[j] = d * static_cast<float>((qs[j] & 0x0F) | (h0 << 4)) + m;
            dst[j + 16] = d * static_cast<float>((qs[j] >> 4) | (h1 << 4)) + m;
        }
        return;
    }
    case 8: { // Q8_0: half d, 32 int8
        const float d = fp16_to_fp32(read_u16(src));
        const int8_t *qs = reinterpret_cast<const int8_t *>(src + 2);
        for (int j = 0; j < 32; ++j) dst[j] = d * static_cast<float>(qs[j]);
        return;
    }
    case 9: { // Q8_1: f32 d, f32 s, 32 int8
        const float d = read_f32(src);
        const int8_t *qs = reinterpret_cast<const int8_t *>(src + 8);
        for (int j = 0; j < 32; ++j) dst[j] = d * static_cast<float>(qs[j]);
        return;
    }
    case 10: { // Q2_K
        const uint8_t *scales = src;
        const uint8_t *qs = src + 16;
        const float d = fp16_to_fp32(read_u16(src + 80));
        const float min = fp16_to_fp32(read_u16(src + 82));
        int is = 0;
        float *y = dst;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                uint8_t sc = scales[is++];
                float dl = d * static_cast<float>(sc & 0x0F);
                float ml = min * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * static_cast<float>((qs[l] >> shift) & 3) - ml;
                sc = scales[is++];
                dl = d * static_cast<float>(sc & 0x0F);
                ml = min * static_cast<float>(sc >> 4);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * static_cast<float>((qs[l + 16] >> shift) & 3) - ml;
                shift += 2;
            }
            qs += 32;
        }
        return;
    }
    case 11: { // Q3_K
        const uint8_t *hm = src;
        const uint8_t *qs = src + 32;
        const uint8_t *packed = src + 96;
        const float d_all = fp16_to_fp32(read_u16(src + 108));
        constexpr uint32_t kmask1 = 0x03030303u;
        constexpr uint32_t kmask2 = 0x0F0F0F0Fu;
        uint32_t aux[4] = {0, 0, 0, 0};
        std::memcpy(aux, packed, 12);
        const uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
        const int8_t *scales = reinterpret_cast<const int8_t *>(aux);
        uint8_t m = 1;
        int is = 0;
        float *y = dst;
        for (int n = 0; n < 256; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {
                float dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * static_cast<float>(static_cast<int>((qs[l] >> shift) & 3) -
                                                   ((hm[l] & m) ? 0 : 4));
                dl = d_all * static_cast<float>(scales[is++] - 32);
                for (int l = 0; l < 16; ++l)
                    *y++ = dl * static_cast<float>(static_cast<int>((qs[l + 16] >> shift) & 3) -
                                                   ((hm[l + 16] & m) ? 0 : 4));
                shift += 2;
                m = static_cast<uint8_t>(m << 1);
            }
            qs += 32;
        }
        return;
    }
    case 12: { // Q4_K
        const float d = fp16_to_fp32(read_u16(src));
        const float min = fp16_to_fp32(read_u16(src + 2));
        const uint8_t *scales = src + 4;
        const uint8_t *qs = src + 16;
        int is = 0;
        float *y = dst;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * static_cast<float>(qs[l] & 0x0F) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * static_cast<float>(qs[l] >> 4) - m2;
            qs += 32;
            is += 2;
        }
        return;
    }
    case 13: { // Q5_K
        const float d = fp16_to_fp32(read_u16(src));
        const float min = fp16_to_fp32(read_u16(src + 2));
        const uint8_t *scales = src + 4;
        const uint8_t *qh = src + 16;
        const uint8_t *qs = src + 48;
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        float *y = dst;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, scales, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min * m;
            get_scale_min_k4(is + 1, scales, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min * m;
            for (int l = 0; l < 32; ++l)
                *y++ = d1 * static_cast<float>((qs[l] & 0x0F) + ((qh[l] & u1) ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l)
                *y++ = d2 * static_cast<float>((qs[l] >> 4) + ((qh[l] & u2) ? 16 : 0)) - m2;
            qs += 32;
            is += 2;
            u1 = static_cast<uint8_t>(u1 << 2);
            u2 = static_cast<uint8_t>(u2 << 2);
        }
        return;
    }
    case 14: { // Q6_K
        const uint8_t *ql = src;
        const uint8_t *qh = src + 128;
        const int8_t *scales = reinterpret_cast<const int8_t *>(src + 192);
        const float d = fp16_to_fp32(read_u16(src + 208));
        float *y = dst;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; ++l) {
                const int is = l / 16;
                const int q1 = static_cast<int>((ql[l + 0] & 0x0F) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int q2 = static_cast<int>((ql[l + 32] & 0x0F) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int q3 = static_cast<int>((ql[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int q4 = static_cast<int>((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l + 0] = d * scales[is + 0] * static_cast<float>(q1);
                y[l + 32] = d * scales[is + 2] * static_cast<float>(q2);
                y[l + 64] = d * scales[is + 4] * static_cast<float>(q3);
                y[l + 96] = d * scales[is + 6] * static_cast<float>(q4);
            }
            y += 128;
            ql += 64;
            qh += 32;
            scales += 8;
        }
        return;
    }

    case 39: { // MXFP4: one E8M0 exponent byte, then 16 bytes of 4-bit codes
        const float d = e8m0_to_fp32_half(src[0]);
        const uint8_t *qs = src + 1;
        for (int j = 0; j < 16; ++j) {
            dst[j] = static_cast<float>(kvalues_fp4[qs[j] & 0x0F]) * d;
            dst[j + 16] = static_cast<float>(kvalues_fp4[qs[j] >> 4]) * d;
        }
        return;
    }
    case 40: { // NVFP4: four UE4M3 sub-block scales, then 32 bytes of 4-bit codes
        const uint8_t *qs = src + 4;
        // 64 elements in four sub-blocks of 16, each with its own scale. The
        // codes of a sub-block are split: the low nibbles come first, then the
        // high ones, which is why both halves of the sub-block are written in
        // one pass.
        for (int s = 0; s < 4; ++s) {
            const float d = ue4m3_to_fp32(src[s]);
            float *out = dst + s * 16;
            const uint8_t *sub = qs + s * 8;
            for (int j = 0; j < 8; ++j) {
                out[j] = static_cast<float>(kvalues_fp4[sub[j] & 0x0F]) * d;
                out[j + 8] = static_cast<float>(kvalues_fp4[sub[j] >> 4]) * d;
            }
        }
        return;
    }
    case 16: { // IQ2_XXS: 8 grid indices per 32-element block, sign bits in aux32[1]
        const float d = fp16_to_fp32(read_u16(src));
        const uint8_t *qs = src + 2;
        int out = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            // qs is an array of 16-bit words in the reference, so one 32-weight
            // step advances it by four words -- eight bytes, not four.
            const uint32_t aux0 = read_u32(qs + 8 * ib32);
            const uint32_t aux1 = read_u32(qs + 8 * ib32 + 4);
            uint8_t idx[4];
            std::memcpy(idx, &aux0, sizeof(idx));
            // The scale is a 4-bit value in the top nibble of the second word,
            // offset by 0.5 so it cannot collapse to zero.
            const float db = d * (0.5f + static_cast<float>(aux1 >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                uint8_t grid[8];
                load_grid(iq2xxs_grid[idx[l]], grid);
                const uint8_t signs = ksigns_iq2xs[(aux1 >> (7 * l)) & 127];
                for (int j = 0; j < 8; ++j) {
                    const float v = db * static_cast<float>(grid[j]);
                    dst[out++] = (signs & kmask_iq2xs[j]) ? -v : v;
                }
            }
        }
        return;
    }
    case 17: { // IQ2_XS: 9-bit grid index and 7-bit sign index per 8 weights
        const float d = fp16_to_fp32(read_u16(src));
        const uint8_t *qs = src + 2;
        const uint8_t *scales = qs + 64;
        int out = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const float db[2] = {d * (0.5f + static_cast<float>(scales[ib32] & 0x0F)) * 0.25f,
                                 d * (0.5f + static_cast<float>(scales[ib32] >> 4)) * 0.25f};
            for (int l = 0; l < 4; ++l) {
                const uint16_t code = read_u16(qs + 2 * (4 * ib32 + l));
                uint8_t grid[8];
                load_grid(iq2xs_grid[code & 511], grid);
                const uint8_t signs = ksigns_iq2xs[code >> 9];
                for (int j = 0; j < 8; ++j) {
                    const float v = db[l / 2] * static_cast<float>(grid[j]);
                    dst[out++] = (signs & kmask_iq2xs[j]) ? -v : v;
                }
            }
        }
        return;
    }
    case 22: { // IQ2_S: 10-bit grid index split between qs and qh, explicit signs
        const float d = fp16_to_fp32(read_u16(src));
        const uint8_t *qs = src + 2;             // 64 bytes, first half is indices
        const uint8_t *qh = qs + 64;             // 2 extra index bits per sub-block
        const uint8_t *scales = qh + 8;
        const uint8_t *signs = qs + 32;          // sign bytes live in the upper qs half
        int out = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const float db[2] = {d * (0.5f + static_cast<float>(scales[ib32] & 0x0F)) * 0.25f,
                                 d * (0.5f + static_cast<float>(scales[ib32] >> 4)) * 0.25f};
            for (int l = 0; l < 4; ++l) {
                const uint32_t index = static_cast<uint32_t>(qs[4 * ib32 + l]) |
                                       ((static_cast<uint32_t>(qh[ib32]) << (8 - 2 * l)) & 0x300u);
                uint8_t grid[8];
                load_grid(iq2s_grid[index], grid);
                const uint8_t sign = signs[4 * ib32 + l];
                for (int j = 0; j < 8; ++j) {
                    const float v = db[l / 2] * static_cast<float>(grid[j]);
                    dst[out++] = (sign & kmask_iq2xs[j]) ? -v : v;
                }
            }
        }
        return;
    }
    case 18: { // IQ3_XXS: two 8-bit grid indices per 8 weights, 4-byte scale+sign words
        const float d = fp16_to_fp32(read_u16(src));
        const uint8_t *qs = src + 2;             // 64 index bytes then 32 words
        const uint8_t *scales_and_signs = qs + 64;
        int out = 0;
        for (int ib32 = 0; ib32 < 8; ++ib32) {
            const uint32_t aux = read_u32(scales_and_signs + 4 * ib32);
            const float db = d * (0.5f + static_cast<float>(aux >> 28)) * 0.5f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t signs = ksigns_iq2xs[(aux >> (7 * l)) & 127];
                uint8_t grid1[4];
                uint8_t grid2[4];
                load_grid32(iq3xxs_grid[qs[8 * ib32 + 2 * l]], grid1);
                load_grid32(iq3xxs_grid[qs[8 * ib32 + 2 * l + 1]], grid2);
                for (int j = 0; j < 4; ++j) {
                    const float v0 = db * static_cast<float>(grid1[j]);
                    const float v1 = db * static_cast<float>(grid2[j]);
                    dst[out + j] = (signs & kmask_iq2xs[j]) ? -v0 : v0;
                    dst[out + j + 4] = (signs & kmask_iq2xs[j + 4]) ? -v1 : v1;
                }
                out += 8;
            }
        }
        return;
    }
    case 21: { // IQ3_S: 9-bit grid indices (qh supplies the 9th bit), 4-bit scales
        const float d = fp16_to_fp32(read_u16(src));
        const uint8_t *qs = src + 2;             // 64 index bytes
        const uint8_t *qh = qs + 64;             // 8 high-bit bytes
        const uint8_t *signs = qh + 8;           // 32 sign bytes
        const uint8_t *scales = signs + 32;      // 4 packed scale bytes
        int out = 0;
        for (int ib32 = 0; ib32 < 8; ib32 += 2) {
            const float db[2] = {d * (1.0f + 2.0f * static_cast<float>(scales[ib32 / 2] & 0x0F)),
                                 d * (1.0f + 2.0f * static_cast<float>(scales[ib32 / 2] >> 4))};
            for (int half = 0; half < 2; ++half) {
                const uint8_t high = qh[ib32 / 2 * 2 + half];
                const uint8_t *sg = signs + (ib32 / 2 * 2 + half) * 4;
                for (int l = 0; l < 4; ++l) {
                    const uint32_t i1 = static_cast<uint32_t>(qs[2 * l]) |
                                        ((static_cast<uint32_t>(high) << (8 - 2 * l)) & 256u);
                    const uint32_t i2 = static_cast<uint32_t>(qs[2 * l + 1]) |
                                        ((static_cast<uint32_t>(high) << (7 - 2 * l)) & 256u);
                    uint8_t grid1[4];
                    uint8_t grid2[4];
                    load_grid32(iq3s_grid[i1], grid1);
                    load_grid32(iq3s_grid[i2], grid2);
                    for (int j = 0; j < 4; ++j) {
                        const float v0 = db[half] * static_cast<float>(grid1[j]);
                        const float v1 = db[half] * static_cast<float>(grid2[j]);
                        dst[out + j] = (sg[l] & kmask_iq2xs[j]) ? -v0 : v0;
                        dst[out + j + 4] = (sg[l] & kmask_iq2xs[j + 4]) ? -v1 : v1;
                    }
                    out += 8;
                }
                qs += 8;
            }
        }
        return;
    }
    case 19: { // IQ1_S: 11-bit grid index, 3-bit scale, one delta bit per block
        const float d = fp16_to_fp32(read_u16(src));
        const uint8_t *qs = src + 2;             // 32 index bytes
        const uint8_t *qh_bytes = qs + 32;       // 8 uint16: high index bits, scale, sign
        int out = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const uint16_t h = read_u16(qh_bytes + 2 * ib);
            const float dl = d * (2.0f * static_cast<float>((h >> 12) & 7) + 1.0f);
            // The delta is a fixed offset applied to every weight of the block,
            // not a per-weight sign: that is what makes 1-bit grids usable.
            const float delta = (h & 0x8000) ? -0.125f : 0.125f;
            for (int l = 0; l < 4; ++l) {
                const uint32_t index = static_cast<uint32_t>(qs[4 * ib + l]) |
                                       ((static_cast<uint32_t>(h >> (3 * l)) & 7u) << 8);
                int8_t grid[8];
                load_grid(iq1s_grid[index], reinterpret_cast<uint8_t *>(grid));
                for (int j = 0; j < 8; ++j) {
                    dst[out++] = dl * (static_cast<float>(grid[j]) + delta);
                }
            }
        }
        return;
    }
    case 29: { // IQ1_M: the scale itself is packed across the four scale words
        const uint8_t *qs = src;                  // 32 index bytes
        const uint8_t *qh = qs + 32;              // 16 bytes: high index bits, deltas
        const uint8_t *sc_bytes = qh + 16;        // 8 bytes read as four uint16
        uint16_t sc[4];
        for (int i = 0; i < 4; ++i) sc[i] = read_u16(sc_bytes + 2 * i);
        // The fp16 scale is assembled bit by bit from the four words; it is not
        // stored contiguously, which is easy to miss and produces garbage scales.
        const uint16_t scale_bits = static_cast<uint16_t>((sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) |
                                                          ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000));
        const float d = fp16_to_fp32(scale_bits);
        int out = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const uint16_t s = sc[ib / 2];
            const float dl1 = d * (2.0f * static_cast<float>((s >> (6 * (ib % 2))) & 7) + 1.0f);
            const float dl2 = d * (2.0f * static_cast<float>((s >> (6 * (ib % 2) + 3)) & 7) + 1.0f);
            const uint8_t *q = qs + 4 * ib;
            const uint8_t *h = qh + 2 * ib;
            const uint32_t idx[4] = {
                static_cast<uint32_t>(q[0]) | ((static_cast<uint32_t>(h[0]) << 8) & 0x700u),
                static_cast<uint32_t>(q[1]) | ((static_cast<uint32_t>(h[0]) << 4) & 0x700u),
                static_cast<uint32_t>(q[2]) | ((static_cast<uint32_t>(h[1]) << 8) & 0x700u),
                static_cast<uint32_t>(q[3]) | ((static_cast<uint32_t>(h[1]) << 4) & 0x700u),
            };
            const float delta[4] = {(h[0] & 0x08) ? -0.125f : 0.125f,
                                    (h[0] & 0x80) ? -0.125f : 0.125f,
                                    (h[1] & 0x08) ? -0.125f : 0.125f,
                                    (h[1] & 0x80) ? -0.125f : 0.125f};
            for (int l = 0; l < 4; ++l) {
                int8_t grid[8];
                load_grid(iq1s_grid[idx[l]], reinterpret_cast<uint8_t *>(grid));
                const float dl = l < 2 ? dl1 : dl2;
                for (int j = 0; j < 8; ++j) {
                    dst[out++] = dl * (static_cast<float>(grid[j]) + delta[l]);
                }
            }
        }
        return;
    }
    case 23: { // IQ4_XS: 6-bit signed scales across two arrays, non-linear 4-bit codes
        const float d = fp16_to_fp32(read_u16(src));
        const uint16_t scales_h = read_u16(src + 2);
        const uint8_t *scales_l = src + 4;
        const uint8_t *qs = scales_l + 4;
        int out = 0;
        for (int ib = 0; ib < 8; ++ib) {
            const int low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
            const int high = (scales_h >> (2 * ib)) & 3;
            const float dl = d * static_cast<float>((low | (high << 4)) - 32);
            for (int j = 0; j < 16; ++j) {
                dst[out + j] = dl * static_cast<float>(kvalues_iq4nl[qs[j] & 0x0F]);
                dst[out + j + 16] = dl * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
            }
            out += 32;
            qs += 16;
        }
        return;
    }
    case 20: { // IQ4_NL: half d, 16 bytes of nibbles indexing a non-linear table
        // Table from the format definition. The four-bit codes are not a linear
        // scale: they are indices into these signed values, so a linear
        // (q - 8) * d approximation would be wrong, not just imprecise.
        const float d = fp16_to_fp32(read_u16(src));
        const uint8_t *qs = src + 2;
        for (int j = 0; j < 16; ++j) {
            dst[j] = d * static_cast<float>(kvalues_iq4nl[qs[j] & 0x0F]);
            dst[j + 16] = d * static_cast<float>(kvalues_iq4nl[qs[j] >> 4]);
        }
        return;
    }
    case 24: // I8
        dst[0] = static_cast<float>(static_cast<int8_t>(src[0]));
        return;
    case 25: { // I16
        int16_t v;
        std::memcpy(&v, src, sizeof(v));
        dst[0] = static_cast<float>(v);
        return;
    }
    case 26: { // I32
        int32_t v;
        std::memcpy(&v, src, sizeof(v));
        dst[0] = static_cast<float>(v);
        return;
    }
    case 27: { // I64
        int64_t v;
        std::memcpy(&v, src, sizeof(v));
        dst[0] = static_cast<float>(v);
        return;
    }
    case 28: { // F64
        double v;
        std::memcpy(&v, src, sizeof(v));
        dst[0] = static_cast<float>(v);
        return;
    }
    case 34: { // TQ1_0: base-3 packed trits, 256 elements, fp16 scale last
        // A byte holds five trits (3^5 = 243 < 256). Multiplication by a power
        // of three modulo 256 moves a chosen trit into the top of the byte,
        // where (q * 3) >> 8 recovers it. The layout is not uniform: the first
        // 160 elements come from qs[0..31], the next 80 from qs[32..47], and the
        // last 16 from qh.
        const float d = fp16_to_fp32(read_u16(src + 52));
        const uint8_t pow3[5] = {1, 3, 9, 27, 81};
        const uint8_t *qs = src;
        const uint8_t *qh = src + 48;
        int out = 0;
        auto emit = [&](uint8_t byte, uint8_t power) {
            // The multiply is on uint8_t, so it wraps modulo 256 by design.
            const uint8_t q = static_cast<uint8_t>(byte * power);
            const int trit = (static_cast<uint16_t>(q) * 3) >> 8;
            dst[out++] = static_cast<float>(trit - 1) * d;
        };
        for (size_t n = 0; n < 5; ++n)
            for (size_t m = 0; m < 32; ++m) emit(qs[m], pow3[n]);
        for (size_t n = 0; n < 5; ++n)
            for (size_t m = 0; m < 16; ++m) emit(qs[32 + m], pow3[n]);
        for (size_t n = 0; n < 4; ++n)
            for (size_t h = 0; h < 4; ++h) emit(qh[h], pow3[n]);
        return;
    }
    case 35: { // TQ2_0: two-bit trits, 256 elements, fp16 scale last
        const float d = fp16_to_fp32(read_u16(src + 64));
        const uint8_t *qs = src;
        int out = 0;
        for (size_t j = 0; j < 64; j += 32)
            for (size_t l = 0; l < 4; ++l)
                for (size_t m = 0; m < 32; ++m) {
                    const int q = (qs[j + m] >> (l * 2)) & 3;
                    dst[out++] = static_cast<float>(q - 1) * d;
                }
        return;
    }
    case 41: { // Q1_0: one sign bit per weight, 128 elements, fp16 scale first
        const float d = fp16_to_fp32(read_u16(src));
        const uint8_t *qs = src + 2;
        for (int j = 0; j < 128; ++j) {
            const uint8_t bit = (qs[j / 8] >> (j % 8)) & 1;
            dst[j] = bit ? d : -d;
        }
        return;
    }
    case 42: { // Q2_0: two-bit codes, 64 elements, fp16 scale first
        const float d = fp16_to_fp32(read_u16(src));
        const uint8_t *qs = src + 2;
        // 00=-1, 01=0, 10=+1, 11=+2
        for (int j = 0; j < 64; ++j)
            dst[j] = static_cast<float>(((qs[j / 4] >> ((j % 4) * 2)) & 3) - 1) * d;
        return;
    }
    case 142: { // PQ2_0: the Q2_0 codec at group size 128
        const float d = fp16_to_fp32(read_u16(src));
        const uint8_t *qs = src + 2;
        for (int j = 0; j < 128; ++j)
            dst[j] = static_cast<float>(((qs[j / 4] >> ((j % 4) * 2)) & 3) - 1) * d;
        return;
    }
    case 15: { // Q8_K
        const float d = read_f32(src);
        const int8_t *qs = reinterpret_cast<const int8_t *>(src + 4);
        for (int j = 0; j < 256; ++j) dst[j] = d * static_cast<float>(qs[j]);
        return;
    }
    default:
        fail(ErrorCode::UNSUPPORTED_QUANTISATION,
             "no block decoder for tensor type " + std::to_string(type_id));
    }
}

void dequantize_tensor_region(uint32_t type_id, const uint8_t *src, float *dst,
                              uint64_t element_count) {
    const QuantGeometry &geo = quant_geometry(type_id);
    if (element_count % geo.block_size != 0)
        fail(ErrorCode::SHAPE_MISMATCH, "region is not a whole number of quantisation blocks");
    const uint64_t blocks = element_count / geo.block_size;
    for (uint64_t b = 0; b < blocks; ++b) {
        dequantize_block(type_id, src + b * geo.block_bytes, dst + b * geo.block_size);
    }
}

} // namespace fm
