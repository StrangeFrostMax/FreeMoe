#pragma once
// The gated feed-forward activation, in one place.
//
// This used to be a lambda inside Engine::forward_token, which made it
// untestable: the only way to check the clamped variants was to build a whole
// checkpoint and compare logits, and a 0.6% error in the bound hides easily in
// a logit comparison. It is a pure function of (config, gate, up, layer), so it
// lives here and the unit tests call it directly.
//
// Two upstream shapes exist for the clamped variant and they do not agree:
//
//   fused      silu(min(gate, limit)) * clamp(up, +-limit)     ggml_swiglu_clamp
//   non-fused  min(silu(gate), limit) * clamp(up, +-limit)     the graph fallback
//
// llama.cpp chooses per architecture (src/llama-graph.cpp:2449). Treating the
// choice as cosmetic is wrong: for gate=5, limit=2, up=3 the fused form gives
// 3.5231883 and the unfused form gives 4.0.
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "fm/gguf.hpp"

namespace fm {

namespace detail {
// Upstream's threshold for "this layer is clamped at all"
// (llama-graph.cpp:2448). A bound at or below it is no bound.
inline constexpr double kClampEpsilon = 1e-6;
} // namespace detail

// SwiGLU: the plain path, which is what every non-clamped family uses.
inline double silu(double x) { return x / (1.0 + std::exp(-x)); }

// The un-gated activation: phi2 and gpt-neox have a single projection, so there
// is no gate to multiply by. One implementation for the engine's gate-free path
// and for the gate branch of the gated one.
inline float pointwise_activation(const ModelConfig &cfg, float x) {
    const std::string &kind = cfg.activation;
    const double v = x;
    if (kind == "gelu") {
        const double inner = 0.7978845608028654 * (v + 0.044715 * v * v * v);
        return static_cast<float>(0.5 * v * (1.0 + std::tanh(inner)));
    }
    if (kind == "gelu_erf")
        return static_cast<float>(0.5 * v * (1.0 + std::erf(v / std::sqrt(2.0))));
    if (kind == "relu") return static_cast<float>(std::max(0.0, v));
    if (kind == "relu_sq") {
        const double r = std::max(0.0, v);
        return static_cast<float>(r * r);
    }
    if (kind == "softplus") return static_cast<float>(std::log1p(std::exp(v)));
    return static_cast<float>(silu(v));
}

// One feed-forward element's activation product.
//
// `layer` selects the per-layer bound; it is ignored by the unclamped forms.
// The switch is on the recorded activation name, which is derived from the
// checkpoint, so a family that gains a clamp in a future file needs no code
// change here.
inline float gated_activation(const ModelConfig &cfg, float gate, float up, uint32_t layer) {
    const std::string &kind = cfg.activation;
    if (kind == "swiglu-oai") {
        // gpt-oss / minimax-m3: ggml_swiglu_oai, alpha 1.702, limit 7.0.
        const double limit = cfg.activation_limit;
        const double x = std::min(static_cast<double>(gate), limit);
        const double y = std::clamp(static_cast<double>(up), -limit, limit);
        return static_cast<float>((x / (1.0 + std::exp(cfg.activation_alpha * (-x)))) * (y + 1.0));
    }
    if (kind == "silu-clamp") {
        const double limit = layer < cfg.swiglu_clamp.size() ? cfg.swiglu_clamp[layer]
                                                             : cfg.activation_limit;
        if (limit <= detail::kClampEpsilon)
            return static_cast<float>(silu(static_cast<double>(gate)) * up);
        const double u = std::clamp(static_cast<double>(up), -limit, limit);
        if (cfg.swiglu_clamp_fused) {
            const double g = std::min(static_cast<double>(gate), limit);
            return static_cast<float>(silu(g) * u);
        }
        const double g = std::min(silu(static_cast<double>(gate)), limit);
        return static_cast<float>(g * u);
    }
    return pointwise_activation(cfg, gate) * up;
}

} // namespace fm
