#include "fm/rope.hpp"

#include <algorithm>
#include <cmath>

#include "fm/common.hpp"

namespace fm {
namespace {

constexpr double kPi = 3.14159265358979323846;

// ggml's `ggml_rope_yarn_corr_dim`. Solving `n_rot = 2*pi*x*base^(2*max_pos/n)`
// for the dimension `x` gives the pair index whose rotary period equals
// `n_rot * 2*pi` turns over the original context.
double yarn_corr_dim(uint32_t n_dims, double n_ctx_orig, double n_rot, double freq_base) {
    return static_cast<double>(n_dims) * std::log(n_ctx_orig / (n_rot * 2.0 * kPi)) /
           (2.0 * std::log(freq_base));
}

// ggml's `rope_yarn_ramp`. Takes the *pair* index (ggml passes its element index
// `i0 = 2*pair` and divides by two). Returns 1 at high frequency (extrapolate)
// and 0 at low frequency (interpolate); `high - low` is floored at 1e-3 so a
// degenerate ramp does not divide by zero.
double yarn_ramp(double low, double high, uint32_t pair) {
    const double y = (static_cast<double>(pair) - low) / std::max(0.001, high - low);
    return 1.0 - std::min(1.0, std::max(0.0, y));
}

// llama.cpp's `get_mscale`: how much to multiply the rotation by so attention
// scores keep their variance once positions are interpolated.
double yarn_mscale(double scale, double mscale) {
    return scale <= 1.0 ? 1.0 : 0.1 * mscale * std::log(scale) + 1.0;
}

}  // namespace

uint32_t rope_pairs(uint32_t n_dims) { return n_dims / 2; }

void rope_yarn_corr_dims(uint32_t n_dims, double n_ctx_orig, double freq_base,
                         double beta_fast, double beta_slow, double dims[2]) {
    const double start = std::floor(yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    const double end = std::ceil(yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));
    dims[0] = std::max(0.0, start);
    dims[1] = std::min(static_cast<double>(n_dims) - 1.0, end);
}

void resolve_rope_scaling(ModelConfig &cfg) {
    const double factor = cfg.rope_scaling_factor > 0.0 ? cfg.rope_scaling_factor : 1.0;
    if (cfg.rope_scaling_type == "none") {
        cfg.rope_freq_scale = 1.0;  // never scale when the type says not to
        cfg.rope_ext_factor = 0.0;
        cfg.rope_mscale = cfg.rope_scaling_attn_factor;
        return;
    }
    if (cfg.rope_scaling_type != "linear" && cfg.rope_scaling_type != "yarn")
        fail(ErrorCode::NOT_IMPLEMENTED,
             "rope.scaling.type '" + cfg.rope_scaling_type +
                 "' is not implemented; 'none', 'linear' and 'yarn' are, and "
                 "'longrope' needs per-layer short/long factors that no checkpoint "
                 "here uses");
    if (cfg.rope_n_ctx_orig <= 0.0)
        fail(ErrorCode::METADATA_MISMATCH,
             "rope.scaling is set without an original context length, so the YaRN "
             "correction ramp has no anchor");
    cfg.rope_freq_scale = 1.0 / factor;
    const bool yarn = cfg.rope_scaling_type == "yarn";
    cfg.rope_ext_factor = yarn ? 1.0 : 0.0;

    double mscale = 1.0;
    if (yarn) {
        if (cfg.rope_yarn_log_mul != 0.0) {
            // deepseek2 stores a magnitude scale other than 1 here and expects it
            // as the numerator, not just as `mscale_all_dim`; llama.cpp
            // special-cases exactly that architecture (TAG_DEEPSEEK2_YARN_LOG_MUL_FIX).
            double mscale_dim = 1.0;
            if (cfg.architecture == "deepseek2" && cfg.rope_yarn_log_mul != 1.0)
                mscale_dim = cfg.rope_yarn_log_mul;
            mscale = yarn_mscale(factor, mscale_dim) / yarn_mscale(factor, cfg.rope_yarn_log_mul);
        } else {
            mscale = yarn_mscale(factor, 1.0);
        }
    }
    cfg.rope_mscale = mscale * cfg.rope_scaling_attn_factor;
}

void RopePlan::init(const ModelConfig &cfg, uint32_t n_dims) {
    this->n_dims = n_dims;
    freq_base = cfg.rope_theta > 0.0 ? cfg.rope_theta : 10000.0;
    freq_scale = cfg.rope_freq_scale > 0.0 ? cfg.rope_freq_scale : 1.0;
    ext_factor = cfg.rope_ext_factor;
    mscale = cfg.rope_mscale;
    corr_low = 0.0;
    corr_high = 0.0;
    if (ext_factor != 0.0) {
        double dims[2] = {0.0, 0.0};
        rope_yarn_corr_dims(n_dims, cfg.rope_n_ctx_orig, freq_base, cfg.rope_yarn_beta_fast,
                            cfg.rope_yarn_beta_slow, dims);
        corr_low = dims[0];
        corr_high = dims[1];
    }
}

void RopePlan::rotation(uint32_t pair, double position, double &cos_theta,
                        double &sin_theta) const {
    if (n_dims == 0 || n_dims % 2 != 0)
        fail(ErrorCode::SHAPE_MISMATCH, "rotary dimension must be even and non-zero");
    // Frequency of this pair, before any scaling: base^(-2i/n).
    const double theta_extrap =
        std::pow(freq_base, -(2.0 * static_cast<double>(pair)) / static_cast<double>(n_dims));
    double theta = freq_scale * theta_extrap;
    if (ext_factor != 0.0) {
        const double ramp = yarn_ramp(corr_low, corr_high, pair) * ext_factor;
        theta = theta * (1.0 - ramp) + theta_extrap * ramp;
    }
    const double angle = position * theta;
    cos_theta = std::cos(angle) * mscale;
    sin_theta = std::sin(angle) * mscale;
}

}  // namespace fm
