#!/usr/bin/env python3
"""Measure the batched prefill path against the sequential one.

Builds a temporary dense checkpoint big enough that prefill is dominated by
weight traffic, then runs `freemoe run` with `--prefill batched` and
`--prefill sequential` at several prompt lengths. The two runs execute the same
arithmetic, so the only difference is how often each weight row is fetched.

Usage:
    python3 tools/bench_prefill.py [--hidden 512] [--layers 8] [--repeat 2]
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import make_arch_fixtures as fixtures  # noqa: E402

FREEMOE = ROOT / "build" / "freemoe"


def build_shape(path: Path, hidden: int, intermediate: int, layers: int, heads: int,
                head_dim: int, context: int) -> None:
    """Write a dense llama checkpoint of the requested geometry."""
    saved_shape = dict(fixtures.SHAPE)
    saved_path = fixtures.fixture_path
    try:
        fixtures.SHAPE.update({"hidden": hidden, "intermediate": intermediate,
                               "layers": layers, "heads": heads, "head_dim": head_dim,
                               "context": context})
        fixtures.fixture_path = lambda slug: path
        fixtures.build({"slug": "prefill-bench", "arch": "llama",
                        "name": "FreeMoe prefill benchmark shape", "kv_heads": heads // 2})
    finally:
        fixtures.SHAPE.clear()
        fixtures.SHAPE.update(saved_shape)
        fixtures.fixture_path = saved_path


def run(model: Path, prompt: str, mode: str, budget_mb: int) -> float:
    out = subprocess.run(
        [str(FREEMOE), "run", str(model), "--tokens", "1", "--budget-mb", str(budget_mb),
         "--prefill", mode, "--prompt", prompt],
        capture_output=True, text=True, cwd=ROOT)
    if out.returncode != 0:
        raise SystemExit(f"{mode} prefill failed: {out.stdout[-400:]}{out.stderr[-400:]}")
    return float(json.loads(out.stdout)["prefill_ms"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--intermediate", type=int, default=2048)
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--context", type=int, default=512)
    ap.add_argument("--budget-mb", type=int, default=512)
    ap.add_argument("--repeat", type=int, default=2)
    ap.add_argument("--reps", default="4,8", help="comma-separated prompt repeat counts")
    args = ap.parse_args()

    if not FREEMOE.exists():
        raise SystemExit("build/freemoe is missing; run `make` first")
    with tempfile.TemporaryDirectory() as tmp:
        model = Path(tmp) / "prefill-bench.gguf"
        build_shape(model, args.hidden, args.intermediate, args.layers, args.heads,
                    args.head_dim, args.context)
        rows = []
        for reps in [int(v) for v in args.reps.split(",")]:
            prompt = "the cat sat on the mat and looked at the door " * reps
            seq = [run(model, prompt, "sequential", args.budget_mb) for _ in range(args.repeat)]
            bat = [run(model, prompt, "batched", args.budget_mb) for _ in range(args.repeat)]
            rows.append({"reps": reps, "sequential_ms": statistics.median(seq),
                         "batched_ms": statistics.median(bat)})
    for row in rows:
        speedup = row["sequential_ms"] / row["batched_ms"]
        print(f"prompt x{row['reps']:<3} sequential {row['sequential_ms']:9.1f} ms   "
              f"batched {row['batched_ms']:9.1f} ms   {speedup:5.2f}x")
    print(json.dumps({"status": "ok", "shape": {"hidden": args.hidden, "layers": args.layers,
                                                "intermediate": args.intermediate,
                                                "heads": args.heads},
                      "rows": rows}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
