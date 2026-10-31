#!/usr/bin/env python3
"""Archive one benchmark run into a committed evidence file.

`results/` is git-ignored, so a benchmark that is only in `results/` is a number
nobody can check after the checkout is gone. This script turns a run directory
into `docs/evidence/benchmark_zen3.json`: the medians and spreads from
`summary.json`, the reference comparison from each `report.json`, and — the part
that makes the run a measurement of the *target* rather than of the host — the
ISA every compute operation actually dispatched on.

It is also a gate. When the run pinned a profile (`extra_args` in `protocol.txt`,
e.g. `--kernels zen3`), every compute operation must have run that profile; a
single `cpu-avx512` op makes the archive `fail` and exits non-zero, because that
run would not describe a Zen 3 machine.

Usage:
    python3 scripts/archive_benchmark.py \
        --run-dir results/benchmark/synthetic-moe-buffered \
        --prefill-json build/prefill_zen3.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# The dispatch a pinned profile must produce. `zen3` and `avx2` are the same
# AVX2+FMA3 path on every host that has it; `scalar` is the reference.
PINNED_DISPATCH = {
    "zen3": {"cpu-avx2_fma"},
    "avx2": {"cpu-avx2_fma"},
    "scalar": {"cpu-scalar"},
    "avx512": {"cpu-avx512"},
}

METRICS = (
    ("prefill_tokens_per_s", "prefill_tokens_per_s", 3),
    ("decode_tokens_per_s", "decode_tokens_per_s", 3),
    ("first_token_latency_ms", "first_token_latency_ms", 3),
    ("median_inter_token_ms", "median_inter_token_ms", 6),
    ("p95_inter_token_ms", "p95_inter_token_ms", 6),
    ("peak_process_ram_bytes", "peak_process_ram_bytes", 0),
)


def rel_to_root(path: Path) -> str:
    """Path as written in the archive: repo-relative when it is inside the tree."""
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def parse_protocol(path: Path) -> dict[str, str]:
    fields: dict[str, str] = {}
    if not path.exists():
        return fields
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            fields[key.strip()] = value.strip()
    return fields


def requested_profile(extra_args: str) -> str:
    words = extra_args.split()
    for index, word in enumerate(words):
        if word == "--kernels" and index + 1 < len(words):
            return words[index + 1]
        if word.startswith("--kernels="):
            return word.split("=", 1)[1]
    return "auto"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="results/benchmark/synthetic-moe-buffered")
    ap.add_argument("--out", default="docs/evidence/benchmark_zen3.json")
    ap.add_argument("--prefill-json", default="",
                    help="tool output of tools/bench_prefill.py --kernels <profile>")
    ap.add_argument("--allow-unpinned", action="store_true",
                    help="record an unpinned run without the dispatch gate")
    args = ap.parse_args()

    run_dir = (ROOT / args.run_dir) if not Path(args.run_dir).is_absolute() else Path(args.run_dir)
    if not run_dir.is_dir():
        raise SystemExit(f"no run directory at {run_dir}; run benchmarks/run_repeats.sh first")
    summary = load_json(run_dir / "summary.json")
    if not summary:
        raise SystemExit(f"{run_dir}/summary.json is missing or invalid")
    protocol = parse_protocol(run_dir / "protocol.txt")
    extra_args = protocol.get("extra_args", "")
    profile = requested_profile(extra_args)

    # Per-repetition facts the summary does not carry: which ISA ran, and whether
    # the reference comparison passed.
    dispatch: dict[str, int] = {}
    comparison: dict[str, object] = {}
    device = ""
    reports = sorted(run_dir.glob("*/report.json"))
    for path in reports:
        report = load_json(path) or {}
        device = device or (report.get("device") or {}).get("backend", "")
        for op in (report.get("backend") or {}).get("operations", []):
            backend = op.get("backend", "unknown")
            dispatch[backend] = dispatch.get(backend, 0) + 1
        cmp = report.get("comparison") or {}
        if cmp:
            worst = comparison.get("max_rel_diff")
            if worst is None or float(cmp.get("max_rel_diff", 0.0)) > float(worst):
                comparison = {
                    "prompts": cmp.get("prompts"),
                    "prompts_within_tolerance": cmp.get("prompts_within_tolerance"),
                    "prompts_token_match": cmp.get("prompts_token_match"),
                    "prompts_routing_match": cmp.get("prompts_routing_match"),
                    "matching_tokens": cmp.get("matching_tokens"),
                    "reference_tokens": cmp.get("reference_tokens"),
                    "max_abs_diff": cmp.get("max_abs_diff"),
                    "max_rel_diff": cmp.get("max_rel_diff"),
                    "status": cmp.get("status"),
                    "report": rel_to_root(path),
                }

    metrics = {}
    for out_key, summary_key, digits in METRICS:
        block = summary.get(summary_key) or {}
        if not block:
            continue
        metrics[out_key] = {
            "median": round(float(block["median"]), digits) if digits else block["median"],
            "min": round(float(block["min"]), digits) if digits else block["min"],
            "max": round(float(block["max"]), digits) if digits else block["max"],
        }

    prefill_bench = None
    if args.prefill_json:
        # A named file that cannot be read is an error: silently archiving the run
        # without it would produce a record that looks complete and is not.
        prefill_bench = load_json(Path(args.prefill_json).expanduser())
        if prefill_bench is None:
            raise SystemExit(f"cannot read --prefill-json {args.prefill_json}")
        if prefill_bench is not None and prefill_bench.get("kernels") != profile:
            raise SystemExit(
                f"prefill benchmark ran --kernels {prefill_bench.get('kernels')!r} but the "
                f"run pinned {profile!r}; the two cannot be archived together")

    # The gate: a pinned profile that did not dispatch is a failed archive, not a
    # footnote. `scalar-control` is the upload/control path, which is not SIMD.
    compute = {name: count for name, count in dispatch.items() if name != "scalar-control"}
    expected = PINNED_DISPATCH.get(profile)
    problems = []
    if expected is not None:
        unexpected = sorted(name for name in compute if name not in expected)
        if unexpected:
            problems.append(
                f"pinned --kernels {profile} but dispatched {', '.join(unexpected)}")
        if not compute:
            problems.append("no compute operation was recorded")

    archive = {
        "schema": "freemoe.benchmark-archive.v1",
        "status": "fail" if problems else "pass",
        "problems": problems,
        "source_run": rel_to_root(run_dir),
        "commit": protocol.get("commit", ""),
        "host": {
            "cpu": device,
            "note": protocol.get("note", ""),
        },
        "protocol": {
            "repetitions": summary.get("repetitions"),
            "model": protocol.get("model", ""),
            "target_p95_ms": summary.get("target_p95_ms"),
            "min_prefill_tps": summary.get("min_prefill_tps"),
            "extra_args": extra_args,
            "requested_profile": profile,
        },
        "isa_dispatch": dict(sorted(dispatch.items())),
        "metrics": metrics,
        "comparison": comparison,
        "target_status": summary.get("target_status"),
        "prefill_bench": prefill_bench,
    }

    out = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(archive, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": archive["status"], "output": str(out),
                      "requested_profile": profile, "dispatch": archive["isa_dispatch"]}))
    return 0 if archive["status"] == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
