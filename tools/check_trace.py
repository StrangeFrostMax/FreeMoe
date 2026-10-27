#!/usr/bin/env python3
"""Check that a suite of FreeMoe traces emits every documented event type."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED = [
    "model_load", "weights_uploaded", "budget", "token", "expert_selected",
    "expert_executed", "cache_hit", "cache_miss", "evict", "io_queued",
    "host_ready", "upload_queued", "upload_complete", "acquire", "release",
    "prefetch_issue", "prefetch_useful", "prefetch_wasted", "source_verify",
    "io_complete", "runtime_released",
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("traces", nargs="+")
    ap.add_argument("--required-file")
    args = ap.parse_args()
    required = REQUIRED
    if args.required_file:
        required = [line.strip() for line in Path(args.required_file).read_text().splitlines()
                    if line.strip() and not line.startswith("#")]
    seen: set[str] = set()
    for path in args.traces:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                seen.add(json.loads(line).get("event", ""))
    missing = [event for event in required if event not in seen]
    out = {
        "status": "pass" if not missing else "fail",
        "trace_files": args.traces,
        "required_event_types": required,
        "observed_event_types": sorted(seen),
        "missing_event_types": missing,
    }
    print(json.dumps(out, indent=2))
    return 0 if not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
