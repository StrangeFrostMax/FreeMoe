#!/usr/bin/env python3
"""Summarise repeated FreeMoe benchmark reports without making claims."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

FIELDS = {
    "prefill_tokens_per_s": ("timing", "prefill_tokens_per_s"),
    "decode_tokens_per_s": ("timing", "decode_tokens_per_s"),
    "first_token_latency_ms": ("timing", "first_token_latency_ms"),
    "median_inter_token_ms": ("timing", "median_inter_token_ms"),
    "p95_inter_token_ms": ("timing", "p95_inter_token_ms"),
    "model_load_ms": ("timing", "model_load_ms"),
    "peak_process_ram_bytes": ("process", "peak_process_ram_bytes"),
    "physical_bytes": ("io", "physical_bytes"),
    "useful_bytes": ("io", "useful_bytes"),
    "cache_hit_rate": ("residency", "cache", "hit_rate"),
}


def get_value(report, path):
    node = report
    for key in path:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    try:
        return float(node)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--target-p95-ms", type=float, default=None)
    ap.add_argument("--min-prefill-tps", type=float, default=None)
    args = ap.parse_args()
    root = Path(args.root)
    reports = sorted(root.glob("*/report.json"))
    if not reports:
        print(json.dumps({"status": "fail", "reason": "no report.json files found"}))
        return 2
    values = {name: [] for name in FIELDS}
    for path in reports:
        report = json.loads(path.read_text(encoding="utf-8"))
        for name, keys in FIELDS.items():
            value = get_value(report, keys)
            if value is not None:
                values[name].append(value)
    summary = {"status": "measured", "repetitions": len(reports), "report_files": [str(p) for p in reports]}
    for name, series in values.items():
        if not series:
            summary[name] = None
            continue
        summary[name] = {
            "median": statistics.median(series),
            "min": min(series),
            "max": max(series),
            "spread": max(series) - min(series),
            "samples": series,
        }
    p95 = summary.get("p95_inter_token_ms")
    prefill = summary.get("prefill_tokens_per_s")
    target_pass = True
    if args.target_p95_ms is not None:
        target_pass = target_pass and bool(p95 and p95["median"] <= args.target_p95_ms)
    if args.min_prefill_tps is not None:
        target_pass = target_pass and bool(prefill and prefill["median"] >= args.min_prefill_tps)
    summary["target_p95_ms"] = args.target_p95_ms
    summary["min_prefill_tps"] = args.min_prefill_tps
    summary["target_pass"] = target_pass
    summary["target_status"] = "pass" if target_pass else "fail"
    print(json.dumps(summary, indent=2))
    return 0 if target_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
