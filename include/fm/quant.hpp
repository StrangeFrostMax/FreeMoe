#pragma once

#include "fm/common.hpp"
#include "fm/gguf.hpp"

#include <cstdint>
#include <string>

namespace fm {

// IEEE binary16 / bfloat16 to binary32. Exact, no approximation.
float fp16_to_f32(uint16_t value);
float bf16_to_f32(uint16_t value);

// Display name for a tensor type, or "TYPE_<id>" when the geometry is unknown.
std::string quant_geometry_or_name(uint32_t type_id);

// Block geometry for a tensor type. Fails with a typed error for unknown types.
const QuantGeometry &quant_geometry(uint32_t type_id);

// True for the 256-element super-block K-quants (Q2_K..Q8_K).
bool quant_is_k_block(uint32_t type_id);

// Decodes exactly one quantisation block into `dst`. `dst` must have room for
// `quant_geometry(type_id).block_size` floats.
void dequantize_block(uint32_t type_id, const uint8_t *src, float *dst);

// Decodes `element_count` values. `element_count` must be a whole number of
// blocks. This is the scalar oracle; faster row kernels must match it exactly.
void dequantize_tensor_region(uint32_t type_id, const uint8_t *src, float *dst,
                              uint64_t element_count);

} // namespace fm
