#!/usr/bin/env python3
"""Classify every upstream llama.cpp architecture by what its graph actually needs.

Reads `src/models/*.cpp` from a llama.cpp checkout and reports, per architecture,
whether this build's existing block covers it (`row-only`), or which feature is
missing. The point is to stop guessing: an architecture is only "one adapter row
away" if its graph uses nothing this engine does not already implement.

    triage_upstream_archs.py --llama-dir /path/to/llama.cpp [--json]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# Features this build's engine implements today (see src/engine.cpp).
SUPPORTED = {
    "silu", "gelu", "gelu_erf", "relu", "relu_sqr", "swiglu-oai", "swiglu-clamp",
    "rmsnorm", "layernorm", "gqa", "mla", "rope", "rope-scaling", "sliding-window",
    "softcap", "moe-topk", "shared-expert", "router-bias", "sinks", "sub-norms",
    "proj-scales", "ssm-mamba", "ssm-mamba2", "biases", "tied-output", "qk-norm",
    "partial-rotary", "fused-qkv", "fused-gate-up", "post-norms", "embedding-scale",
    "norm-zero-centred", "mrope", "attn-temp", "mla-lite", "expert-groups",
}

# Markers that mean the graph needs a block this build does not have. Each maps
# to a short reason so the report says *why*, not just "unsupported".
NEEDS = [
    (r"build_moe_ffn\(", "moe"),
    (r"ffn_norm_exps", "dual dense+moe ffn"),
    (r"ggml_ssm_", "ssm"),
    (r"build_mamba", "ssm"),
    (r"rwkv|RWKV", "rwkv scan"),
    (r"delta_net|ggml_gated_delta|build_delta", "gated delta-net"),
    (r"shortconv|short_conv|conv1d", "short convolution"),
    (r"hc_|hyper_connection|dsv4_hc", "hyper-connections"),
    (r"indexer", "dsa indexer"),
    (r"attn_comp", "attention compressor"),
    (r"mtp\.|nextn\.", "mtp head"),
    (r"build_inp_embd_per_layer|altup|AltUp", "altup / per-layer embeddings"),
    (r"ggml_conv|vision|mm_projector|clip", "multimodal"),
    (r"experts_tid2eid|tid2eid", "token-id expert routing"),
    (r"attn_temp", "attn temperature"),
    (r"ggml_cross|encoder", "encoder-decoder"),
    (r"build_recurrent|linear_attn|gated_linear", "linear attention"),
]

MATCHERS = {
    "silu": r"LLM_FFN_SILU",
    "gelu": r"LLM_FFN_GELU\b",
    "gelu_erf": r"LLM_FFN_GELU_ERF",
    "relu": r"LLM_FFN_RELU\b",
    "relu_sqr": r"LLM_FFN_RELU_SQR",
    "swiglu-oai": r"LLM_FFN_SWIGLU_OAI",
    "softcap": r"softcap",
    "sinks": r"attn_sinks",
    "sub-norms": r"attn_sub_norm|ffn_sub_norm",
    "proj-scales": r"\.scale\b|_s\)",
    "qk-norm": r"attn_q_norm|attn_k_norm",
    "post-norms": r"post_norm",
    "mrope": r"rope_sections|mrope",
    "mla": r"attn_kv_a|wkv_a|kv_a_norm",
    "rope-scaling": r"rope_scaling|yarn|freq_scale",
    "sliding-window": r"n_swa|sliding_window",
    "partial-rotary": r"n_rot(\s*<|\s*!=)",
    "fused-qkv": r"TENSOR_ATTN_QKV",
    "fused-gate-up": r"gate_up",
    "tied-output": r"TENSOR_DUPLICATED",
    "embedding-scale": r"f_embedding_scale|embedding_scale",
    "norm-zero-centred": r"norm_add_one|f_norm_rms_eps.*1\.0|zero-centred",
    "attn-temp": r"attn_temp",
    "moe": r"build_moe_ffn\(",
    "shared-expert": r"ffn_gate_shexp|ffn_up_shexp",
    "router-bias": r"ffn_exp_probs_b|exp_probs_b",
}

# Architectures whose *only* difference from the llama block is metadata or a
# name. Kept explicit so the report never invents one.
KNOWN_BLOCKS = {
    "llama", "qwen2", "qwen3", "phi2", "phi3", "gemma", "gemma2", "gemma3",
}


def graph_text(path: Path) -> str:
    text = path.read_text(errors="ignore")
    # Keep the graph body only: the loader section creates tensors, which is a
    # different question from what the forward pass does.
    marker = "::graph::graph("
    idx = text.find(marker)
    if idx < 0:
        idx = text.find("::graph(")
    return text[idx:] if idx >= 0 else text


def classify(arch: str, path: Path) -> dict:
    text = path.read_text(errors="ignore")
    graph = graph_text(path)
    reasons = [reason for pattern, reason in NEEDS if re.search(pattern, graph)]
    # A dense FFN built *and* a MoE FFN built in the same layer means the two
    # outputs are summed; that is not per-layer routing, which is what this
    # engine implements.
    if len(re.findall(r"build_ffn\(", graph)) and re.search(r"build_moe_ffn\(", graph):
        if "dual dense+moe ffn" not in reasons:
            reasons.append("dense ffn and moe in the same layer")
    features = sorted(name for name, pattern in MATCHERS.items() if re.search(pattern, graph))
    needs_weighted_norm = bool(re.search(r"build_norm\(\s*[^,]+,\s*NULL\s*,\s*NULL", graph))
    if needs_weighted_norm and "layernorm" not in features:
        reasons.append("norm without a weight tensor")
    return {
        "arch": arch,
        "file": path.name,
        "verdict": "row-only" if not reasons else "needs-block",
        "reasons": reasons,
        "features": features,
        "note": "" if not reasons else "; ".join(reasons),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--llama-dir", required=True)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    models = Path(args.llama_dir) / "src" / "models"
    if not models.is_dir():
        print(f"{models} is not a directory", file=sys.stderr)
        return 2
    rows = []
    for path in sorted(models.glob("*.cpp")):
        if path.name in ("models.cpp",):
            continue
        arch = path.stem.replace("-", "_") if False else path.stem
        rows.append(classify(arch, path))

    row_only = [r for r in rows if r["verdict"] == "row-only"]
    if args.json:
        print(json.dumps(rows, indent=2))
        return 0
    print(f"{len(rows)} model files: {len(row_only)} row-only, "
          f"{len(rows) - len(row_only)} need a block\n")
    print("ROW-ONLY (candidates for an adapter row):")
    for r in row_only:
        print(f"  {r['arch']:<22} {' '.join(r['features'][:6])}")
    print("\nNEEDS A BLOCK:")
    for r in rows:
        if r["verdict"] != "row-only":
            print(f"  {r['arch']:<22} {r['note']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
