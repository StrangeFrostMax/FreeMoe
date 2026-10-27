#!/usr/bin/env bash
# Run R repeated process-cold (not OS-cold) synthetic benchmark reports.
# Usage: bash benchmarks/run_repeats.sh configs/benchmark.env
set -euo pipefail
if [[ $# -ne 1 || ! -f "$1" ]]; then
    echo "usage: bash benchmarks/run_repeats.sh <config.env>" >&2
    exit 2
fi
CONFIG="$1"
set -a
# shellcheck disable=SC1090
source "$CONFIG"
set +a

: "${RUN_ID:?RUN_ID required}"
: "${MODEL:?MODEL required}"
: "${BACKEND:=cpu}"
: "${READER:=buffered}"
: "${CACHE_SLOTS:=1}"
: "${TOKENS:=16}"
: "${LIMIT:=20}"
: "${PREFETCH:=0}"
: "${REFERENCE_ROOT:=tests/reference}"
: "${R:=5}"
: "${TARGET_P95_MS:=50}"
: "${MIN_PREFILL_TPS:=1000}"
: "${EXTRA_ARGS:=}"

OUT="results/benchmark/$RUN_ID"
mkdir -p "$OUT"
{
    echo "protocol=process-cold; OS cache state unknown"
    echo "repetitions=$R"
    echo "model=$MODEL backend=$BACKEND reader=$READER cache_slots=$CACHE_SLOTS"
    echo "target_p95_ms=$TARGET_P95_MS min_prefill_tps=$MIN_PREFILL_TPS"
    echo "commit=$(git rev-parse HEAD)"
} > "$OUT/protocol.txt"

for i in $(seq 1 "$R"); do
    rep="${RUN_ID}_r${i}"
    repdir="$OUT/$rep"
    mkdir -p "$repdir"
    set +e
    # EXTRA_ARGS is intentionally word-split.
    # shellcheck disable=SC2086
    ./build/freemoe compare \
        --model "$MODEL" --backend "$BACKEND" --reader "$READER" \
        --cache-slots "$CACHE_SLOTS" --tokens "$TOKENS" --limit "$LIMIT" \
        --prefetch "$PREFETCH" --reference-root "$REFERENCE_ROOT" \
        --run-id "$rep" --results-dir "$OUT" $EXTRA_ARGS \
        > "$repdir/stdout.json" 2> "$repdir/stderr.txt"
    rc=$?
    set -e
    echo "$rc" > "$repdir/exit_code.txt"
done

set +e
python3 scripts/summarize_benchmark.py "$OUT" \
    --target-p95-ms "$TARGET_P95_MS" \
    --min-prefill-tps "$MIN_PREFILL_TPS" > "$OUT/summary.json"
summary_rc=$?
set -e
echo "$summary_rc" > "$OUT/target_status.txt"
python3 scripts/update_results_index.py > "$OUT/results_index.txt"
echo "benchmark summary: $OUT/summary.json (target_exit=$summary_rc)"
