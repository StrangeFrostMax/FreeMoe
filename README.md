# FreeMoe

A standalone C++17 correctness prototype for the Llama-like disk-resident MoE
inference design in `Final_Llama_Like_Disk_Resident_MoE_Spec.md`.

**Read this first:** this is not a general-purpose Llama/MoE runtime. It
supports one explicitly documented synthetic GGUF family, F32 weights, two
real backends and one deliberately failing physical-GPU placeholder. Its goal
is to prove parser, adapter, forward-pass, expert-residency, paging and
instrumentation correctness before adding model-family breadth or performance
features.

## Contents

- [What is implemented](#what-is-implemented)
- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Build and test](#build-and-test)
- [CLI reference](#cli-reference)
- [Common workflows](#common-workflows)
- [Zen 3 fallback](#zen-3-fallback)
- [Verification, evidence and troubleshooting](#verification-evidence-and-troubleshooting)
- [Current release gates](#current-release-gates)
- [Documentation](#documentation)
- [Licence](#licence)

## What is implemented

- Bounded, fuzzable GGUF v3 parser and `inspect` / `WAKE_UP` reports.
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
- MoE residency manager with authoritative state table, generation IDs,
  reference counts, LRU and clock replacement, safe eviction, checksums and
  fault injection.
- Scalar CPU backend, vendored **Virtual NPU** backend, and a physical GPU
  placeholder that fails loudly unless `--allow-cpu-fallback` is explicit.
- JSON metrics/report, JSONL traces, trace replay, malformed fixtures,
  ASan/UBSan fuzz target, soak test, run recorder and benchmark scripts.

## Repository layout

| Path | Contents |
|---|---|
| `Final_Llama_Like_Disk_Resident_MoE_Spec.md` | Original architecture and phase-gate specification. |
| `src/`, `include/fm/` | C++17 runtime: GGUF parser, readers, backends, residency, engine and CLI. |
| `third_party/vnpu/` | Vendored Virtual NPU source, pinned upstream commit and local patches. |
| `tests/fixtures/`, `tests/reference/`, `tests/golden/` | Synthetic GGUF checkpoints, independent reference outputs and golden inspect reports. |
| `tests/malformed_models/`, `tests/unit/`, `tests/fuzz_gguf.cpp` | Fail-closed parser fixtures, unit/integration tests and fuzz target. |
| `tools/` | Independent Python GGUF writer/parser, reference oracle, fixture generator and trace checks. |
| `docs/` | Architecture, supported models, backend support, decisions, failure modes, test machine and phase-exit records. |
| `docs/evidence/` | Committed verification logs, final reports and release-status JSON. |
| `scripts/` | Build verification, run recording, evidence collection, benchmark and index scripts. |
| `configs/`, `benchmarks/` | Example run-record and exploratory benchmark configurations. |
| `build/`, `results/`, `traces/` | Generated and gitignored build/run artefacts. |

## Requirements

- Linux/POSIX environment; GNU Make and a C++17 compiler (verified with GCC
  12.2).
- POSIX threads.
- Python 3 with the standard library only, for fixtures, trace tools and
  evidence scripts.
- Optional: CMake 3.16 or newer. **GNU Make is the verified path** used for the
  committed evidence; CMake build files are provided but are not the primary
  verification path in this environment.
- No model download, network access or external C++ dependency is required.
  Synthetic checkpoints and the MIT-licensed Virtual NPU source are committed.

## Build and test

### GNU Make — verified path

From a fresh checkout:

```sh
git clone https://github.com/StrangeFrostMax/FreeMoe.git
cd FreeMoe

make clean && make -j2          # produces build/freemoe
make test                       # full unit/integration/vendor/quick-fuzz suite
./build/freemoe help
./build/freemoe version
```

Useful Make targets:

```sh
make -j2                        # build CLI only
make test                       # unit, vendor selftest, quick ASan/UBSan fuzz, trace and dependency checks
make fuzz FUZZ_SECONDS=72       # declared ASan/UBSan fuzz duration
make soak                       # three 256-token sessions
make zen3-fallback              # focused Zen 3 detection/fallback evidence
make benchmark                  # exploratory process-cold repeats
make fixtures                   # regenerate deterministic fixtures and references
make clean                      # remove build/
make distclean                  # remove build/, results/ and traces/
make help                       # list targets
```

`make fixtures` changes the deterministic fixture/reference set. Only use it
when intentionally updating the D5 reference pin; update `docs/DECISIONS.md`
and regenerate evidence afterwards.

### CMake — optional

```sh
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake -j2
ctest --test-dir build-cmake --output-on-failure
```

The CMake path builds `freemoe`, `freemoe-tests`, `vnpu-selftest`, and
`freemoe-fuzz`. CTest runs the core unit, vendor-selftest, quick-fuzz and
trace-check tests. It does not run the architecture/dependency scripts or
regenerate `docs/evidence/`; use `make test` and `scripts/phase0_exit.sh` for
the full committed evidence path. `cmake` was not installed on the recorded
sandbox, so the GNU Make path is the verified one.

## CLI reference

Run `./build/freemoe help` for the compact built-in usage. Options accept
`--key value` or `--key=value`; boolean flags can be passed bare.

### Commands

| Command | Purpose |
|---|---|
| `help`, `--help`, `-h` | Print compact CLI usage. |
| `version` | Print `FreeMoe 0.1.0 correctness prototype`. |
| `inspect <model.gguf> [--json] [--parse-only] [--output path]` | Parse and validate a GGUF file, or report the adapter manifest. |
| `WAKE_UP <model.gguf> [--json]` | Alias for `inspect`. |
| `device-info --backend cpu\|vnpu\|gpu [--json]` | Report backend name, memory type, SIMD width and notes. |
| `run --model <model.gguf> [options]` | Run one prompt and emit generated token IDs. |
| `compare --model <model.gguf> [options]` | Compare all selected fixture prompts against the independent reference. |

`--model` is also accepted by `inspect`/`WAKE_UP`. `compare` writes `trace.jsonl`,
`metrics.json` and `report.json` under `<results-dir>/<run-id>/` by default and
returns exit code `22` if any prompt fails logits, greedy-token or routing
comparison. Runtime errors print a JSON object with `code`, `message` and
`exit_code` on stderr; see `docs/FAILURE_MODES.md`.

### Common options

| Option | Default | Meaning |
|---|---|---|
| `--backend cpu\|vnpu\|gpu` | `cpu` | Select scalar CPU, vendored Virtual NPU, or GPU placeholder. |
| `--reader resident\|buffered\|mmap\|direct` | `buffered` | Cold-source reader. `direct` uses Linux `O_DIRECT` where supported. |
| `--cache-slots N` | `0` | Hot expert slots; `0` derives the maximum from the memory budget. |
| `--queue-depth N` | `2` | I/O scheduler queue depth. |
| `--coalesce-gap N` | `4096` | Maximum byte gap for coalescing adjacent extents. |
| `--t-wait-ms N` | `100` | Wait for a busy slot before returning a typed error. |
| `--tokens N` | `16` | Greedy tokens to generate. |
| `--limit N` | `0` | Number of reference prompts; `0` selects all 20. |
| `--prefetch 0\|1` | `1` | Enable or disable expert prefetch. |
| `--replacement lru\|clock` | `lru` | Expert-slot replacement policy. |
| `--max-prefetch N` | queue depth | Maximum in-flight prefetch requests. |
| `--host-cache-records N` | `4` | Bounded host staging records. |
| `--budget-mb N` | `64` | Total hot-tier budget in MiB. |
| `--workspace-kb N` | `1024` | Workspace reservation in KiB. |
| `--staging-kb N` | `1024` | Staging reservation in KiB. |
| `--margin-kb N` | `4096` | Safety margin reservation in KiB. |
| `--vnpu-isa PROFILE` | `avx2_dual` | `zen3-auto\|auto\|avx2_dual\|zen3\|avx2\|avx512\|scalar`. |
| `--vnpu-threads N` | `0` | VNPU worker threads; `0` uses the runtime default. |
| `--allow-cpu-fallback` | off | Permit explicit, logged CPU fallback for the GPU placeholder. |
| `--force-direct-fallback` | off | Force the direct-reader fallback path for tests. |
| `--test-simulate-zen3` | off | `device-info` only: simulate Zen 3 detection. |
| `--trace path` | none | JSONL trace output. |
| `--metrics path` | none | JSON metrics output. |
| `--report path` | none | JSON report output. |
| `--run-id ID` | `local` | Run identifier and results subdirectory. |
| `--results-dir path` | `results` | Root directory for result artefacts. |
| `--reference-root path` | `tests/reference` | Independent reference fixtures. |
| `--abs-tol X`, `--rel-tol X` | `0.005` | Logits comparison tolerances. |
| `--prompt TEXT` | first fixture prompt | Inline prompt text. |
| `--prompt-file PATH` | first fixture prompt | Read prompt from a file. |
| `--dump-logits PATH` | none | Write prefill logits for `run`. |
| `--json` | off | JSON output for `inspect` or `device-info`. |
| `--parse-only` | off | `inspect` only parses GGUF, without adapter validation. |
| `--output path` | stdout | Also write `inspect` output to a file. |

## Common workflows

### Inspect a checkpoint

```sh
./build/freemoe inspect tests/fixtures/moe-fm.gguf
./build/freemoe inspect tests/fixtures/dense-fm.gguf --json
./build/freemoe WAKE_UP tests/fixtures/moe-fm.gguf --json
./build/freemoe inspect tests/fixtures/moe-fm.gguf --parse-only --output build/inspect.json
```

### Query backends

```sh
./build/freemoe device-info --backend cpu  --json
./build/freemoe device-info --backend vnpu --json
./build/freemoe device-info --backend gpu  --json   # expected typed error: no physical GPU backend
```

All backends report `dedicated_vram_bytes=0`: the hot tier is host RAM and the
Virtual NPU is a CPU software device. The GPU placeholder is intentionally not
a GPU implementation.

### Run one prompt

```sh
./build/freemoe run \
  --model tests/fixtures/moe-fm.gguf \
  --backend cpu --reader buffered --cache-slots 1 \
  --prefetch 1 --tokens 16 \
  --prompt-file tests/reference/prompt_00.txt \
  --trace build/trace.jsonl --metrics build/metrics.json \
  --report build/report.json --dump-logits build/logits.txt
```

### Compare against the independent reference

```sh
./build/freemoe compare \
  --model tests/fixtures/moe-fm.gguf \
  --backend vnpu --reader direct --cache-slots 1 \
  --prefetch 0 --limit 20 --tokens 16 \
  --run-id local-moe --results-dir results
```

If `O_DIRECT` is unavailable, the direct reader can emit an explicit logged
fallback; use `--reader buffered` or `--reader mmap` for a portable path.

### Verify traces

```sh
python3 tools/trace_replay.py \
  --trace results/local-moe/trace.jsonl \
  --metrics results/local-moe/metrics.json --check
python3 tools/check_trace.py \
  build/test-tmp/replay-trace.jsonl \
  build/test-tmp/prefetch-trace.jsonl \
  build/test-tmp/cachehit-trace.jsonl
```

The three `build/test-tmp` traces are generated by `./build/tests` and are used
by `make test` to verify that every documented event type is emitted.

### Record a run or run exploratory benchmarks

```sh
bash scripts/run_record.sh configs/example.env
bash benchmarks/run_repeats.sh configs/benchmark.env
# equivalent: make benchmark
```

`configs/example.env` requires `RUN_ID`, `MODEL` and `BACKEND`.
`configs/benchmark.env` adds `R`, `TARGET_P95_MS` and `MIN_PREFILL_TPS`. These
are process-cold exploratory repeats, not controlled OS-cold benchmarks; the
performance-qualified gate remains closed.

### Regenerate all committed evidence

```sh
bash scripts/phase0_exit.sh
```

This performs a clean build, `make test`, a 72-second fuzz, soak sessions,
final dense/MoE comparisons and evidence collection. It takes about four
minutes on the recorded sandbox and rewrites files under `docs/evidence/` and
several generated phase-exit documents.

## Zen 3 fallback

The supplied [npuxl-improved-rinai](https://github.com/StrangeFrostMax/npuxl-improved-rinai)
project is a CPU software device, not a physical NPU or GPU. Its source is
vendored at a pinned commit under `third_party/vnpu/` and patched for Zen 3:

- two independent 256-bit AVX2/FMA3 accumulator chains in the F32 GEMM kernels;
- no AVX-512 dependency on the Zen 3 path;
- detection-driven fallback: `znver3` takes precedence over AVX-512 capability
  bits, so Zen 3 always starts on dual 256-bit AVX2/FMA3;
- safe explicit ISA forcing with CPU-feature checks;
- visible fallback diagnostics in `device-info` and the `backend_init` trace;
- regression tests against the scalar oracle, including simulated Zen 3
  detection on non-Zen-3 hosts.

Use `--vnpu-isa zen3-auto` to select runtime auto dispatch off Zen 3 and the
dual-pipe fallback on Zen 3. On Zen 3, `auto` and even explicit `avx512`
requests are rerouted to the dual-pipe path. On other x86 hosts, `zen3-auto`
uses runtime auto dispatch and may legitimately select AVX-512.

```sh
./build/freemoe device-info --backend vnpu --vnpu-isa zen3-auto --json
./build/freemoe device-info --backend vnpu --vnpu-isa zen3-auto \
  --test-simulate-zen3 --json
make zen3-fallback
```

`--test-simulate-zen3` is accepted only by `device-info`; it is a verification
aid, not an inference mode. `make zen3-fallback` writes
`docs/evidence/zen3_fallback.json` and `docs/evidence/zen3_fallback.log` with a
machine-checkable verdict. See `third_party/vnpu/PATCHES_FREEMOE.md` and
`docs/BACKEND_SUPPORT.md` for the full policy.

## Verification, evidence and troubleshooting

`make test` exercises 54 malformed GGUF fixtures, all GGUF metadata value
types, golden inspect reports, Q8_0 geometry, every residency state/event
pair, budgets, coalescing, direct-I/O fallback, source-change detection,
allocation tracking, Zen 3 fallback diagnostics, GPU fallback policy, slot
generations, fences, stale completions, eviction, injected failures, 20-prompt
dense/MoE CPU and VNPU reference comparisons, reader bit-equivalence, LRU vs
clock equivalence, operation logs, trace replay and three soak sessions.

Recorded evidence in `docs/evidence/` reports:

- `make test`: **587 checks, 0 failures** and VNPU selftest **19 tests, 0
  failures**;
- 72-second ASan/UBSan fuzz: **72,000 iterations, 0 sanitizer findings**;
- dense CPU, MoE CPU, MoE VNPU and MoE direct final comparisons: **20/20
  prompts pass**;
- `release_status.json`: correctness and paged-MoE prototypes pass, with
  `zen3_fallback: true`.

Generated run data lives in gitignored `results/`, `traces/` and `build/`.
Committed evidence and phase gates live in `docs/evidence/`,
`docs/PHASE0_EXIT.md`, `docs/M1_EXIT.md`, `docs/MOE_M3_EXIT.md` and
`docs/RELEASE_CHECKLIST.md`.

Common failures:

| Symptom | What to do |
|---|---|
| `GPU_BACKEND_UNIMPLEMENTED` / `CPU_FALLBACK_DISABLED` | Expected for `--backend gpu`. Use `cpu`/`vnpu`, or pass `--allow-cpu-fallback` for an explicit, logged CPU fallback. |
| `DIRECT_IO_FALLBACK` or short direct reads | Use `--reader buffered` or `--reader mmap`; `--force-direct-fallback` is for tests. |
| `BUDGET_TOO_SMALL` | Increase `--budget-mb`, reduce `--cache-slots`, or lower workspace/staging/margin reservations. |
| `SOURCE_CHANGED` | Do not modify the GGUF file while a run is active. |
| `INVALID_ARGUMENT` (exit 2) | Check `./build/freemoe help` and option spelling/values. |
| `compare` exits 22 | Inspect `results/<run-id>/report.json` and the emitted trace/metrics. |
| Missing `cmake` | Use the GNU Make path; CMake is optional. |

## Current release gates

- **Correctness prototype:** PASS for the declared synthetic dense and MoE
  fixtures, CPU and Virtual NPU.
- **Paged MoE prototype:** PASS for demand-paged selected experts, safe
  eviction, checksums and cold-cache output correctness on the recorded
  sandbox.
- **Performance-qualified:** CLOSED. No target hardware profile or controlled
  cold/warm benchmark on that machine has been approved. No throughput number
  from another project or hardware is claimed.
- **Physical GPU:** CLOSED. No GPU backend, SDK, driver or VRAM record exists;
  the GPU option remains a placeholder.

## Documentation

- `docs/ARCHITECTURE.md` — module inventory, tiers, dependency licences.
- `docs/SUPPORTED_MODELS.md` — exact adapter semantics and unsupported cases.
- `docs/BENCHMARK_PROTOCOL.md` — D4/D7 values, reference pin, D8 hashes, protocol.
- `docs/FORMAT_AND_QUANTISATION.md` — GGUF and block geometry boundaries.
- `docs/BACKEND_SUPPORT.md` — CPU/VNPU/GPU status and Zen 3 fallback.
- `docs/FAILURE_MODES.md` — typed errors and fixed CLI exit codes.
- `docs/TEST_MACHINE.md` — D1 machine record and unknowns.
- `docs/DECISIONS.md` — D1–D9 decision log.
- `docs/RESULTS_INDEX.md` — generated run index, including failed runs.
- `docs/EXTERNAL_DOCS_ANALYSIS.md` — analysis of externally supplied planning docs; extracted copies live in `docs/external/riteshnair-freemoe/`.
- `docs/ROADMAP_INFERENCE_AND_TRAINING.md` — what remains for high-performance inference and native training, with phases and gates.
- `docs/PHASE0_EXIT.md`, `docs/M1_EXIT.md`, `docs/MOE_M3_EXIT.md`,
  `docs/RELEASE_CHECKLIST.md` — generated phase gates and release criteria.

## Licence

FreeMoe code and synthetic checkpoints are MIT licensed. The vendored Virtual
NPU is MIT licensed; see `LICENSE` and `LICENSES/`.
