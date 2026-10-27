#!/usr/bin/env bash
# Record one reproducible FreeMoe correctness/benchmark run.
# Usage: bash scripts/run_record.sh configs/example.env
set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
    echo "usage: bash scripts/run_record.sh <config.env>" >&2
    exit 2
fi
CONFIG="$1"
declare -A CFG=()
while IFS='=' read -r key value || [[ -n "${key:-}" ]]; do
    key="${key//[[:space:]]/}"
    [[ -z "$key" || "$key" == \#* ]] && continue
    value="${value%$'\r'}"
    CFG["$key"]="$value"
done < "$CONFIG"

for required in RUN_ID MODEL BACKEND; do
    if [[ -z "${CFG[$required]:-}" ]]; then
        echo "missing required config key: $required" >&2
        exit 2
    fi
done

RUN_ID="${CFG[RUN_ID]}"
MODEL="${CFG[MODEL]}"
BACKEND="${CFG[BACKEND]}"
READER="${CFG[READER]:-buffered}"
CACHE_SLOTS="${CFG[CACHE_SLOTS]:-1}"
TOKENS="${CFG[TOKENS]:-16}"
LIMIT="${CFG[LIMIT]:-20}"
PREFETCH="${CFG[PREFETCH]:-0}"
REFERENCE_ROOT="${CFG[REFERENCE_ROOT]:-tests/reference}"
EXTRA_ARGS="${CFG[EXTRA_ARGS]:-}"

OUT="results/$RUN_ID"
mkdir -p "$OUT"
cp "$CONFIG" "$OUT/config.env"
git rev-parse HEAD > "$OUT/commit.txt"
{
    echo "# uname"
    uname -a
    echo "# compiler"
    "${CXX:-g++}" --version | head -1
    echo "# cpu"
    lscpu 2>/dev/null || true
    echo "# memory"
    free -h
    echo "# backend"
    ./build/freemoe device-info --backend "$BACKEND" --json
} > "$OUT/machine.txt"

set +e
# EXTRA_ARGS is intentionally word-split: it is a config-supplied CLI suffix.
# shellcheck disable=SC2086
./build/freemoe compare \
    --model "$MODEL" \
    --backend "$BACKEND" \
    --reader "$READER" \
    --cache-slots "$CACHE_SLOTS" \
    --tokens "$TOKENS" \
    --limit "$LIMIT" \
    --prefetch "$PREFETCH" \
    --reference-root "$REFERENCE_ROOT" \
    --run-id "$RUN_ID" \
    --results-dir results \
    $EXTRA_ARGS > "$OUT/stdout.json" 2> "$OUT/stderr.txt"
rc=$?
set -e
echo "$rc" > "$OUT/exit_code.txt"
python3 scripts/update_results_index.py > "$OUT/results_index.txt"
find "$OUT" -maxdepth 1 -type f ! -name SHA256SUMS.txt -print0 \
    | sort -z | xargs -0 sha256sum > "$OUT/SHA256SUMS.txt"
echo "recorded $OUT (exit=$rc)"
exit "$rc"
