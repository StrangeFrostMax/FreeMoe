#!/usr/bin/env python3
"""Replay a FreeMoe routing trace through an LRU cache simulator."""
from __future__ import annotations

import argparse
import json
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    events: list[dict[str, Any]] = []
    for line in Path(args.trace).read_text(encoding="utf-8").splitlines():
        if line.strip():
            events.append(json.loads(line))
    budget = next((e for e in events if e.get("event") == "budget"), None)
    if not budget or int(budget.get("slot_count", 0)) <= 0:
        print(json.dumps({"status": "error", "reason": "missing budget/slot_count"}))
        return 2
    slot_count = int(budget["slot_count"])

    demands: list[tuple[int, int]] = []
    for event in events:
        if event.get("event") != "expert_selected":
            continue
        layer = int(event["layer"])
        for expert in event.get("experts", []):
            demands.append((layer, int(expert)))

    cache: OrderedDict[tuple[int, int], bool] = OrderedDict()
    hits = misses = 0
    for key in demands:
        if key in cache:
            hits += 1
            cache.move_to_end(key)
        else:
            misses += 1
            if len(cache) >= slot_count:
                cache.popitem(last=False)
            cache[key] = True

    metrics = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    runtime = metrics["residency"]["cache"]
    runtime_hits = int(runtime["cache_hits"])
    runtime_misses = int(runtime["cache_misses"])
    match = hits == runtime_hits and misses == runtime_misses
    out = {
        "status": "pass" if match else "fail",
        "slot_count": slot_count,
        "demand_events": len(demands),
        "simulated_hits": hits,
        "simulated_misses": misses,
        "runtime_hits": runtime_hits,
        "runtime_misses": runtime_misses,
        "counts_match": match,
    }
    print(json.dumps(out, indent=2))
    if args.check and not match:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
