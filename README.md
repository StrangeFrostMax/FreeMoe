# FreeMoe

A standalone C++17 correctness prototype for the Llama-like disk-resident MoE
inference design in `Final_Llama_Like_Disk_Resident_MoE_Spec.md`.

**Read this first:** this is not yet a general-purpose Llama/MoE runtime. It
supports one explicitly documented synthetic GGUF family, F32 weights, two
real backends and one deliberately failing GPU placeholder. It exists to prove
parser, adapter, forward-pass, expert-residency, paging and instrumentation
correctness before adding model-family breadth or performance features.

## What is implemented

- Bounded, fuzzable GGUF v3 parser and `model inspect` / `WAKE_UP` report.
- Strict `freemoe-tiny-v1` architecture adapter with canonical tensor roles,
  stacked expert extents and per-expert checksums.
- Llama-like dense and MoE forward pass: byte tokenizer, embeddings, RMSNorm,
  half-rotation RoPE, MHA KV cache, SwiGLU, router top-k, expert execution and
  greedy sampling.
- Independent scalar Python reference runtime and 20 prompts from 8 to 512
  byte tokens, with logits, generated IDs and MoE routing fixtures.
- Three-tier paged expert delivery prototype: unchanged GGUF cold source,
  bounded host staging/prefetch cache, and backend-owned hot expert slots.
- Buffered, mmap, Linux `O_DIRECT` and resident readers; coalescing, demand
  priority, cancellation and useful/wasted prefetch metrics.
- MoE residency manager with the authoritative state table, generation IDs,
  reference counts, LRU and clock replacement, safe eviction, checksums and
  fault injection.
- Scalar CPU backend, vendored **Virtual NPU** backend, and a physical GPU
  placeholder that fails loudly unless `--allow-cpu-fallback` is explicit.
- JSON metrics/report, JSONL traces, trace replay, malformed fixtures,
  ASan/UBSan fuzz target, soak test, run recorder and benchmark scripts.

## NPU fix for Zen 3

The supplied [npuxl-improved-rinai](https://github.com/StrangeFrostMax/npuxl-improved-rinai)
project is a CPU software device, not a physical NPU or GPU. Its source is
vendored at a pinned commit under `third_party/vnpu/` and patched for Zen 3:

- two independent 256-bit AVX2/FMA3 accumulator chains in the F32 GEMM kernels;
- no AVX-512 dependency;
- detection-driven fallback: `znver3` takes precedence over AVX-512 capability
  bits, so Zen 3 always starts on dual 256-bit AVX2/FMA3;
- safe explicit ISA forcing with CPU-feature checks;
- visible fallback diagnostics in `device-info` and the `backend_init` trace;
- regression tests against the scalar oracle, including simulated Zen 3
  detection on non-Zen-3 hosts.

Use `--vnpu-isa zen3-auto` to select runtime auto dispatch off Zen 3 and the
dual-pipe fallback on Zen 3. The test-only flag `--test-simulate-zen3` (with
`device-info`) forces that detection branch for verification:

```sh
./build/freemoe device-info --backend vnpu --vnpu-isa zen3-auto --json
./build/freemoe device-info --backend vnpu --vnpu-isa zen3-auto \
  --test-simulate-zen3 --json
```

See `third_party/vnpu/PATCHES_FREEMOE.md` and `docs/BACKEND_SUPPORT.md`.

## Quick start

Requirements: a C++17 compiler, GNU Make, POSIX threads, Python 3 (standard
library only) and a Linux/POSIX environment. CMake is optional.

```sh
make clean && make -j2          # build/freemoe
make test                       # full unit/integration/vendor/fuzz-quick suite
./build/freemoe inspect tests/fixtures/moe-fm.gguf --json
./build/freemoe device-info --backend vnpu --json
```

Run one prompt through the paged MoE runtime:

```sh
./build/freemoe run \
  --model tests/fixtures/moe-fm.gguf \
  --backend cpu --reader buffered --cache-slots 1 \
  --prefetch 1 --tokens 16 \
  --prompt-file tests/reference/prompt_00.txt \
  --trace build/trace.jsonl --metrics build/metrics.json
```

Compare all 20 prompts against the independent reference:

```sh
./build/freemoe compare \
  --model tests/fixtures/moe-fm.gguf \
  --backend vnpu --reader direct --cache-slots 1 \
  --prefetch 0 --limit 20 --tokens 16 \
  --run-id local-moe --results-dir results
```

Other useful commands:

```sh
make fuzz FUZZ_SECONDS=72       # declared ASan/UBSan fuzz duration
make soak                       # three 256-token sessions
make zen3-fallback              # focused Zen 3 detection/fallback evidence
bash scripts/run_record.sh configs/example.env
bash benchmarks/run_repeats.sh configs/benchmark.env
bash scripts/phase0_exit.sh     # full clean verification + evidence
```

## Backends

| `--backend` | Meaning |
|---|---|
| `cpu` | Scalar correctness oracle. |
| `vnpu` | Vendored Virtual NPU, AVX2/FMA3 dual-pipe path. Major matvecs run through its C ABI; control ops are explicitly logged as scalar. |
| `gpu` | Physical GPU placeholder. Fails with `GPU_BACKEND_UNIMPLEMENTED` unless explicit CPU fallback is enabled. |

No backend reports dedicated VRAM in this prototype. The hot tier is host RAM;
metrics say `dedicated_vram_bytes=0`. Exact physical GPU/NPU hardware, SDK and
driver remain unknown until recorded in `docs/TEST_MACHINE.md`.

## Verification status

`make test` exercises:

- 54 malformed GGUF fixtures and all GGUF metadata value types;
- golden dense/MoE inspect reports and Q8_0 block geometry;
- every state/event pair in the residency state table;
- budgets, coalescing, direct-I/O fallback and source-change detection;
- allocation tracker, NPU dual-path matvec, Zen 3 fallback planner/detection
  diagnostics and GPU fallback policy;
- slot generations, deferred fences, stale completions, safe eviction,
  all-slots-busy and injected transfer/fence/device-loss failures;
- 20-prompt dense/MoE CPU and VNPU reference comparisons;
- resident/buffered/mmap/direct reader bit-equivalence;
- one-slot vs all-experts cache bit-equivalence;
- LRU vs clock output equivalence;
- operation log, trace event types, trace replay and three soak sessions.

Generated run data lives in ignored `results/` and `build/` directories.
Committed evidence and phase gates live in `docs/evidence/` and
`docs/PHASE0_EXIT.md`, `docs/M1_EXIT.md`, `docs/MOE_M3_EXIT.md`, and
`docs/RELEASE_CHECKLIST.md`.

## Current release gates

- **Correctness prototype:** PASS for the declared synthetic dense and MoE
  fixtures, CPU and Virtual NPU.
- **Paged MoE prototype:** PASS for demand-paged selected experts, safe
  eviction, checksums and cold-cache output correctness on the sandbox.
- **Performance-qualified:** CLOSED. No target hardware profile or controlled
  cold/warm benchmark on that machine has been approved. No throughput number
  from another project or hardware is claimed.

## Documentation

- `docs/ARCHITECTURE.md` — module inventory, tiers, dependency licences.
- `docs/SUPPORTED_MODELS.md` — exact adapter semantics and unsupported cases.
- `docs/BENCHMARK_PROTOCOL.md` — D4/D7 values, reference pin, D8 hashes, protocol.
- `docs/FORMAT_AND_QUANTISATION.md` — GGUF and block geometry boundaries.
- `docs/BACKEND_SUPPORT.md` — CPU/VNPU/GPU status and Zen 3 patch.
- `docs/FAILURE_MODES.md` — typed errors and fixed CLI exit codes.
- `docs/TEST_MACHINE.md` — D1 machine record and unknowns.
- `docs/DECISIONS.md` — D1–D9 decision log.
- `docs/RESULTS_INDEX.md` — generated run index, including failed runs.

## Licence

FreeMoe code and synthetic checkpoints are MIT licensed. The vendored Virtual
NPU is MIT licensed; see `LICENSES/`.
