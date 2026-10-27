#!/usr/bin/env python3
"""Generate docs/BENCHMARK_PROTOCOL.md including D8 fixture hashes."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "BENCHMARK_PROTOCOL.md"
D5_PIN = "deee8aa67b9615a0ef8cd73aaa4c8aca6469cfa5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    manifest = {}
    for line in (ROOT / "tests/reference/manifest.sha256").read_text().splitlines():
        if not line.strip():
            continue
        digest, path = line.split(None, 1)
        manifest[path] = digest
    d8_paths = []
    for i in range(20):
        stem = f"prompt_{i:02d}"
        d8_paths += [
            f"tests/reference/{stem}.txt",
            f"tests/reference/{stem}.input.ids",
            f"tests/reference/dense/{stem}.greedy.ids",
            f"tests/reference/moe/{stem}.greedy.ids",
        ]
    rows = ["| File | SHA-256 |", "|---|---|"]
    for path in d8_paths:
        rows.append(f"| `{path}` | `{manifest.get(path, 'MISSING')}` |")

    oracle_hash = sha256(ROOT / "tools/reference_oracle.py")
    generator_hash = sha256(ROOT / "tools/make_fixtures.py")
    manifest_hash = sha256(ROOT / "tests/reference/manifest.sha256")
    text = f"""# Benchmark and correctness protocol

Status: committed provisional protocol for the **synthetic correctness
prototype**. This is not a performance-qualified release and does not claim
support for production GGUF checkpoints.

## Reference

| Field | Value |
|---|---|
| Reference runtime | Independent scalar Python oracle, `tools/reference_oracle.py` |
| D5 pinned tree | `{D5_PIN}` |
| Oracle file SHA-256 | `{oracle_hash}` |
| Fixture generator SHA-256 | `{generator_hash}` |
| Full reference manifest SHA-256 | `{manifest_hash}` |
| Python | 3.11.2 standard library only |
| Build command | none (Python source is executed directly) |
| Compiler/build flags | not applicable |
| Reference output command | `python3 tools/make_fixtures.py --all` |
| Reference logits files | `tests/reference/<kind>/prompt_XX.logits` |
| Reference token/routing files | `tests/reference/<kind>/prompt_XX.greedy.ids`, `tests/reference/<kind>/prompt_XX.routing.txt` |

`tests/reference/manifest.sha256` is the authoritative SHA-256 list for every
reference artefact, including logits and routing traces. The D8 prompt and ID
file hashes are also reproduced below.

## Tolerances (D4)

| Data type | Absolute tolerance | Relative tolerance | Used by |
|---|---:|---:|---|
| F32 logits | 0.005 | 0.005 | D2 dense and D3 MoE reference comparisons |

A comparison passes when `abs(actual-reference) <= abs_tol + rel_tol *
abs(reference)` for every logit, greedy token IDs match for `N_greedy`, and MoE
routing IDs match for every layer/token. F16/Q8_0 and K-quants are parsed and
size-checked but are not executable inference types in this release. Any
tolerance change requires a commit whose message states the reason and date.

## Parameters (D7)

| Parameter | Value | Notes |
|---|---:|---|
| `N_greedy` | 16 | generated tokens compared per prompt |
| `N_soak` | 256 | tokens per soak session; three consecutive sessions |
| `F` | 0.02 h (72 s) | declared ASan/UBSan fuzz duration |
| `T_wait` | 100 ms | slot wait/fence deadline in benchmark runs |
| `G_bytes` | 4096 | maximum gap for read coalescing |
| `R` | 5 | exploratory benchmark repetitions |

Deterministic unit tests override `T_wait` to 5 ms to keep fault-injection
tests fast. Benchmark and soak commands use the committed value unless a config
explicitly declares otherwise.

## Provisional performance target

The user delegated parameter choice on 2026-10-27. The following provisional
targets apply only to the synthetic D3 MoE fixture on the recorded sandbox:

- median prefill throughput >= 1000 prompt tokens/s;
- median p95 inter-token latency <= 50 ms;
- 20/20 prompt correctness comparisons pass;
- no invalid state transitions, checksum failures or unbounded allocations.

These are project inputs, not measured hardware claims. The exact target
Zen 3 CPU model, RAM, storage, GPU/NPU SDK, driver and usable VRAM are **unknown
(D1)**, so the performance-qualified release gate remains closed until a real
D1 machine record is committed and benchmarked.

## Benchmark controls

- Fix model, quantisation, prompts, generated token count, context, backend,
  reader, cache slots, replacement policy, prefetch, queue depth, coalescing
  gap and thread count in `configs/benchmark.env`.
- Record commit, compiler, OS, CPU, RAM, backend/API/driver and device query.
- Separate model load, prefill and decode in `results/<run-id>/report.json`.
- Run `R=5` repetitions and report median plus min/max spread using
  `scripts/summarize_benchmark.py`.
- Validate generated tokens/routing and logits; do not accept throughput alone.
- No benchmark number from a paper, upstream issue or different backend is
  transplanted into this project's results.

## Cold and warm protocol

- A process restart is **not** an OS-cold run.
- The sandbox cannot safely control OS page cache. Its runs are labelled
  `process-cold first run; OS cache state unknown`.
- Warm runs first read the model in a separate unmeasured warming pass and
  record the method. Cold/warm run sets are never mixed in one comparison.
- Runs with unknown cache state are labelled `unknown` and excluded from
  cold-versus-warm comparisons.
- Failed runs remain in `results/` and are listed by
  `scripts/update_results_index.py` in `docs/RESULTS_INDEX.md`.

## Commands

```sh
make clean && make -j2
make test
make fuzz FUZZ_SECONDS=72
make soak
bash scripts/run_record.sh configs/example.env
bash benchmarks/run_repeats.sh configs/benchmark.env
```

## D8 prompt and ID file hashes

{chr(10).join(rows)}

The 20 prompts range from 8 to 512 byte tokens. Prompt whitespace is
significant; readers strip CR/LF only, never trailing spaces.
"""
    OUT.write_text(text, encoding="utf-8")
    print(json.dumps({"status": "pass", "output": OUT.relative_to(ROOT).as_posix(),
                      "hash_rows": len(rows) - 2, "d5_pin": D5_PIN}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
