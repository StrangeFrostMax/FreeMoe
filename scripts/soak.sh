#!/usr/bin/env bash
# Three N_soak-token sessions with bounded paged expert residency.
set -euo pipefail
mkdir -p build/soak
for i in 1 2 3; do
    out="build/soak/session_$i"
    mkdir -p "$out"
    ./build/freemoe run \
        --model tests/fixtures/moe-fm.gguf \
        --backend cpu \
        --reader buffered \
        --cache-slots 1 \
        --tokens 256 \
        --prefetch 0 \
        --t-wait-ms 5 \
        --prompt-file tests/reference/prompt_00.txt \
        --metrics "$out/metrics.json" > "$out/stdout.json"
    python3 - "$out/metrics.json" "$out/stdout.json" <<'PY'
import json, sys
metrics = json.load(open(sys.argv[1]))
stdout = json.load(open(sys.argv[2]))
live = metrics["backend"]["allocations"]["live_allocations"]
tokens = len(stdout.get("generated_ids", []))
assert live == 0, f"outstanding allocations after session: {live}"
assert tokens == 256, f"expected 256 generated tokens, got {tokens}"
print(json.dumps({"status": "pass", "live_allocations": live, "generated_tokens": tokens}))
PY
done
echo "soak: three sessions passed"
