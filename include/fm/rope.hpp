#pragma once
// Rotary position embedding, including llama.cpp's context-extension modes.
//
// This is deliberately its own unit rather than a branch inside the attention
// loop: the YaRN maths has four interacting parameters (scale factor, original
// context, correction ramp, magnitude scale) and getting any of them wrong
// produces fluent nonsense instead of a crash, so it needs to be testable
// against the formulas directly.
//
// The formulas are ggml's (`ggml_rope_yarn_corr_dim`, `rope_yarn_ramp`,
// `rope_yarn` in ggml/src/ggml-cpu/ops.cpp) read through llama.cpp's model and
// context code, so a checkpoint that runs in llama.cpp rotates the same way
// here.
#include <array>
#include <cstdint>

#include "fm/gguf.hpp"

namespace fm {

// Number of dimension *pairs* that rotate for a head of `n_dims` rotated dims.
uint32_t rope_pairs(uint32_t n_dims);

// ggml's `ggml_rope_yarn_corr_dims`: the pair indices where interpolation starts
// and stops. Pairs below `dims[0]` keep their exact positions (their period is
// much shorter than the original context), pairs above `dims[1]` are fully
// interpolated, and the band between them blends linearly.
//
// `dims` receives {start, end}, both clamped into [0, n_dims - 1].
void rope_yarn_corr_dims(uint32_t n_dims, double n_ctx_orig, double freq_base,
                         double beta_fast, double beta_slow, double dims[2]);

// Fill in the derived fields of `cfg` from the raw `rope.scaling.*` metadata:
// `rope_freq_scale`, `rope_ext_factor` and `rope_mscale`.
//
// llama.cpp's kernel multiplies its magnitude scale by
// `1 + 0.1*log(1/freq_scale)` internally whenever extrapolation is on, and its
// context setup divides that same factor out again; the two cancel. There is no
// such hidden factor here, so `rope_mscale` is already the net value. Folding it
// back in would double-count it.
void resolve_rope_scaling(ModelConfig &cfg);

// Everything needed to rotate one tensor, precomputed once per call: the ramp
// bounds and the magnitude scale do not depend on the position.
struct RopePlan {
    uint32_t n_dims = 0;
    double freq_base = 10000.0;
    double freq_scale = 1.0;
    double ext_factor = 0.0;
    double corr_low = 0.0;
    double corr_high = 0.0;
    double mscale = 1.0;
    // Per-pair frequency divisors (`rope_freqs.weight`), or null for the usual
    // case where every pair uses its own base frequency unchanged.
    const float *freq_factors = nullptr;

    // `n_dims` is the number of rotated dimensions, which for most families is
    // the head width but is smaller when `rope.dimension_count` says so.
    void init(const ModelConfig &cfg, uint32_t n_dims);

    // mRoPE sections, in rotary *pairs*, from `{arch}.rope.dimension_sections`:
    // time, height, width, extra. A zero total means plain rotary.
    std::array<uint32_t, 4> sections = {0, 0, 0, 0};
    // Interleaved sectioning (qwen3-vl, qwen3.5) cycles t/t/y/x instead of
    // assigning runs of pairs in order.
    bool interleaved_sections = false;

    // Which of the four mRoPE axes pair `pair` reads its position from. Exposed
    // because it is the only part of mRoPE that a text-only prompt cannot
    // exercise: with four equal positions the choice makes no difference, so the
    // assignment has to be tested directly.
    uint32_t axis_of(uint32_t pair) const;

    // cos/sin for pair `pair` (elements `2*pair`, `2*pair+1`) at `position`.
    void rotation(uint32_t pair, double position, double &cos_theta,
                  double &sin_theta) const;

    // cos/sin for a prompt that carries one position per mRoPE axis. For text
    // all four are equal, which makes this identical to the scalar overload --
    // that equivalence is what lets the text path stay one code path.
    void rotation(uint32_t pair, const std::array<double, 4> &positions, double &cos_theta,
                  double &sin_theta) const;
};

}  // namespace fm
