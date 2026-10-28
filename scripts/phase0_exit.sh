#!/usr/bin/env bash
# Full clean verification and evidence collection. Takes about 2-4 minutes
# because the declared fuzz duration F is 72 seconds.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

must() {
    echo "+ $*"
    if ! "$@"; then
        echo "FATAL: command failed: $*" >&2
        exit 1
    fi
}

make clean >/dev/null 2>&1 || true
rm -rf build/phase0 results/final-dense-cpu results/final-moe-cpu \
       results/final-moe-vnpu results/final-moe-direct results/example-local
mkdir -p build/phase0

must bash -c 'make -j1 > build/phase0/build.log 2>&1'
must bash -c 'make test > build/phase0/test.log 2>&1'
must bash -c 'make fuzz FUZZ_SECONDS=72 > build/phase0/fuzz.log 2>&1'
must bash -c 'make soak > build/phase0/soak.log 2>&1'

# R3 record-script checks.
if bash scripts/run_record.sh > build/phase0/run_record_no_config.log 2>&1; then
    echo "FATAL: run_record.sh accepted a missing config" >&2
    exit 1
fi
must bash -c 'bash scripts/run_record.sh configs/example.env > build/phase0/run_record.log 2>&1'

run_compare() {
    local run_id="$1"; shift
    echo "+ freemoe compare $run_id"
    mkdir -p "results/$run_id"
    set +e
    ./build/freemoe compare --run-id "$run_id" --results-dir results "$@" \
        > "results/$run_id/stdout.json" 2> "results/$run_id/stderr.txt"
    local rc=$?
    set -e
    echo "$rc" > "results/$run_id/exit_code.txt" 2>/dev/null || true
    if [[ $rc -ne 0 ]]; then
        echo "WARNING: comparison $run_id exited $rc" >&2
        FINAL_RC=1
    fi
}
FINAL_RC=0
run_compare final-dense-cpu --model tests/fixtures/dense-fm.gguf --backend cpu \
    --reader buffered --cache-slots 0 --prefetch 0 --limit 20 --tokens 16
run_compare final-moe-cpu --model tests/fixtures/moe-fm.gguf --backend cpu \
    --reader buffered --cache-slots 1 --prefetch 1 --limit 20 --tokens 16
run_compare final-moe-vnpu --model tests/fixtures/moe-fm.gguf --backend vnpu \
    --reader buffered --cache-slots 1 --prefetch 0 --limit 20 --tokens 16
run_compare final-moe-direct --model tests/fixtures/moe-fm.gguf --backend cpu \
    --reader direct --cache-slots 1 --prefetch 0 --limit 20 --tokens 16

# Deterministic outputs used for Phase 0 exit hashes.
must bash -c './build/tests > build/phase0/unit.log 2>&1'
must bash -c './build/freemoe inspect tests/fixtures/dense-fm.gguf --json > build/phase0/inspect_dense.json'
must bash -c './build/freemoe inspect tests/fixtures/moe-fm.gguf --json > build/phase0/inspect_moe.json'
must bash -c 'python3 tools/trace_replay.py --trace build/test-tmp/replay-trace.jsonl --metrics build/test-tmp/replay-metrics.json --check > build/phase0/replay.json'
must bash -c 'python3 tools/check_trace.py build/test-tmp/replay-trace.jsonl build/test-tmp/prefetch-trace.jsonl build/test-tmp/cachehit-trace.jsonl > build/phase0/trace_events.json'
must bash -c 'python3 scripts/check_architecture.py > build/phase0/architecture.json'
must bash -c 'python3 scripts/check_dependencies.py > build/phase0/dependencies.json'
must bash -c 'python3 scripts/zen3_fallback_check.py > build/phase0/zen3_fallback.log 2>&1'
must bash -c './build/freemoe bench --iterations 20 --out build/phase0/bench.json > /dev/null'

if ! python3 scripts/collect_evidence.py > build/phase0/collect_evidence.json; then
    echo "FATAL: evidence collection reported a failed gate" >&2
    cat build/phase0/collect_evidence.json >&2
    exit 1
fi
if [[ $FINAL_RC -ne 0 ]]; then
    echo "FATAL: one or more final comparisons failed" >&2
    exit "$FINAL_RC"
fi
echo "phase0 exit: PASS; evidence in docs/evidence/"
