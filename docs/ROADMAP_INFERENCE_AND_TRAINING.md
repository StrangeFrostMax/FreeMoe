# FreeMoe roadmap: advanced inference and training

**Status:** planning document, 2026-10-28. Branch `arena/19bfe20d-freemoe`, based on
`main` at `ba2d39d`.

**Purpose:** state exactly what remains between the current correctness prototype
and (a) an advanced, extremely high-performing inference engine and (b) a
training system for the `freemoe-coder-v1` architecture described in
`docs/external/riteshnair-freemoe/FreeMoe_Model_Math_GGUF_Creation_and_Runtime_Spec.md`.

Nothing in this document is implemented yet unless it is listed under
"already done" below. Every milestone here has an acceptance gate in the same
style as `docs/RELEASE_CHECKLIST.md`.

---

## 1. Where the project actually stands

Measured facts from this tree, not aspirations:

| Area | Current state | Evidence |
|---|---|---|
| Code size | ~7,700 lines across `src/`, `include/`, `tools/`, `scripts/` | `wc -l` |
| Tests | 587 unit/integration checks, 19 VNPU selftests, ASan/UBSan fuzz, soak | `docs/evidence/test_summary.json` |
| Model support | exactly one adapter: `freemoe-tiny` / `freemoe-tiny-v1` | `docs/SUPPORTED_MODELS.md` |
| Fixture size | 2 layers, hidden 16, intermediate 32, 4 heads, head_dim 4, vocab 258, context 1024 | `tools/make_fixtures.py` |
| Tensor types | F32 execution only; quant table used for parsing/geometry only | `src/gguf.cpp` |
| Attention | MHA only (`kv_heads == heads`), no GQA, no sliding window | `src/gguf.cpp` adapter checks |
| Tokenizer | raw byte IDs, no real BPE/SentencePiece, **no detokenization** | `Engine::tokenize()` |
| Sampling | greedy argmax only | `Engine::greedy()` |
| Compute kernel | naive scalar `matvec`, single-threaded, no SIMD, no blocking | `CpuBackend::matvec()` |
| Measured speed | prefill 443 tok/s, decode 587 tok/s **on the 16-wide toy model** | `docs/evidence/final_moe_cpu.json` |
| Expert cache | 0 hits / 22,160 misses in the committed run (1 slot, tiny model) | same report |
| GPU | failing placeholder, `dedicated_vram_bytes = 0` | `docs/BACKEND_SUPPORT.md` |
| Training | **none**: no autograd, no backward pass, no optimizer, no checkpointing | — |
| Blob store | none | — |
| Server/streaming | none; CLI only | `src/main.cpp` |
| Platforms | Linux/POSIX GNU Make verified; Windows/CMake unverified | `docs/TEST_MACHINE.md` |

The important consequence: the current numbers say nothing about real-model
performance. A 16-wide model fits in L1; a 1.3B model streams ~0.9 GB of active
weights per token in F32. Everything below is about closing that gap.

---

## 2. What "advanced" and "extremely high performing" mean here

Proposed, measurable targets. All provisional until hardware is pinned (D1).

### 2.1 Inference targets

| Scenario | Target | Basis |
|---|---|---|
| `freemoe-coder-v1` (1.58B total, 374M active core), Q4_K/F16 mix, Zen 3 8-core | **40–80 tok/s decode**, 250–600 tok/s prefill | memory-bandwidth bound; see §2.2 |
| Same model, small dense 135M assistant | 150+ tok/s decode | bandwidth + compute both comfortable |
| Cold-start first token (disk → first token) | < 2 s for < 2 GB model on NVMe | read + parse + first prefill |
| Expert cache hit rate, 27B-class top-2/8, 12 GB budget | > 90% steady state | slot sizing math |
| 27B-class MoE disk-resident decode | 5–15 tok/s | NVMe random read + PCIe transfer bound |
| p99 inter-token jitter | < 2× p50 | scheduling, prefetch, no re-read of hot experts |
| Peak RSS | within configured budget ±5% | budget enforcement |
| Numerical parity | logits within 5e-3 abs/rel vs F32 oracle; greedy IDs exact | D4 |

### 2.2 Why quantization is mandatory, not optional

Decode is memory-bandwidth bound: every token re-reads all active weights.

Active parameters per token for `freemoe-coder-v1`:

- experts: `20 layers × 2 experts × 3 × 1280 × 1920` = **294.9M**
- attention: `20 × (1280² + 2×1280×256 + 1280²)` = **78.6M**
- router: **0.2M**
- output head (tied, V = 32k example): **41.0M**
- total ≈ **415M parameter reads per token**

| Weight format | Bytes read per token | Floor at 20 GB/s effective | Floor at 40 GB/s |
|---|---:|---:|---:|
| F32 | 1.66 GB | ~83 ms → 12 tok/s | ~42 ms → 24 tok/s |
| F16 | 0.83 GB | ~42 ms → 24 tok/s | ~21 ms → 48 tok/s |
| Q6_K | ~0.35 GB | ~18 ms → 56 tok/s | ~9 ms → 111 tok/s |
| Q4_K | ~0.24 GB | ~12 ms → 83 tok/s | ~6 ms → 166 tok/s |

Compute is not the first wall on Zen 3: dual 256-bit AVX2/FMA3 gives roughly
100+ GFLOP/s theoretical per core, and one token needs ~0.83 GFLOP. But the
current kernel is scalar and single-threaded, so compute *is* the wall today.
Both must be fixed, bandwidth first.

**Conclusion:** Q4_K/Q6_K kernels, a real SIMD/threaded GEMM, and an expert
cache that actually holds hot experts are the three highest-value items for
inference performance. Nothing else comes close.

---

## 3. Inference workstreams

Each workstream lists what exists, what is missing, the concrete tasks, and the
gate that closes it. IDs are stable so they can be used in issues and commits.

### I0 — Foundation refactors (blocking everything else)

Exists: strict adapter, manifest, residency, traces, backends.
Missing: separation between *container parsing* and *architecture semantics*;
a tensor-type capability matrix; a backend kernel interface that can express
quantized types and batched GEMM.

Tasks:

1. Split `gguf.cpp` into container parser (`gguf_doc`), capability/quant table,
   and per-architecture adapters selected by `general.architecture`.
2. Add `TensorTypeSupport { parse, dequantize, cpu_execute, device_execute }`
   reported per tensor type and surfaced in `inspect --json`.
3. Replace `Backend::matvec` with `Backend::gemm` + `Backend::gemv`
   `(dtype, m, n, k, transpose, epilogue)` and an explicit capability query.
4. Version the inspect/report JSON schemas (`freemoe.inspect.v2`) and keep a
   golden test for both versions.

**Gate:** existing 587 checks still pass unchanged; `freemoe-tiny-v1` remains
bit-identical; new schema golden file committed.

Effort: 1–2 weeks.

### I1 — Real model support (the single biggest functional gap)

Exists: one synthetic adapter, MHA, separate output weight, F32.
Missing: any real architecture.

Tasks, in order:

1. **Generic GGUF inspector** that works on any GGUF file without executing it
   (metadata dump, tensor directory, dtypes, offsets, alignment, support
   classification). This is DOCX/ZIP Gate 1 and unlocks every later step.
2. **GGUF writer** so fixtures and converted checkpoints can be produced in
   C++ as well as Python.
3. **`llama`-family adapter**: Llama/Mistral/Qwen2 shapes, GQA, tied/untied
   output, RoPE variants, per-layer norm names, chat template metadata.
4. **Real tokenizer**: SentencePiece BPE and byte-level BPE from GGUF metadata,
   with encode/decode/special-token/chat-template tests; remove the raw-byte
   stopgap from the production path (keep it for `freemoe-tiny-v1`).
5. **`freemoe-coder-v1` adapter** exactly as specified: 20 layers, hidden 1280,
   GQA 20/4, head_dim 64, 8 experts top-2, SwiGLU, tied output, MoE-only FFN.
6. **Detokenization and text output** in the CLI (currently tokens only).
7. **Quantized formats**: F16 → Q8_0 → Q6_K → Q4_K, each with known-vector
   dequantization tests, block-boundary tests, and a separate "parsed vs
   executable" flag.

**Gate:** load a real Llama-family GGUF and a real MoE GGUF, produce greedy
tokens matching a reference implementation stage-by-stage (embedding, norm,
Q/K/V, RoPE, attention, router, each expert, residual, final norm, logits).

Effort: 4–6 weeks for steps 1–4, +2–3 weeks for `freemoe-coder-v1` and
quantized execution.

### I2 — Compute kernels (performance-critical)

Exists: scalar double loop.
Missing: SIMD, threading, tiling, quantized dot products, fused ops.

Tasks:

1. AVX2/FMA3 GEMV with the same dual-accumulator strategy already patched into
   the vendored VNPU (reuse it rather than duplicating).
2. **Runtime dispatch** through the VNPU/FreeMoe ISA layer: Zen 3 → dual 256-bit
   AVX2 (already implemented), AVX-512 only on non-Zen-3 hosts, scalar fallback.
3. **Thread pool** with work-stealing over output rows; persistent threads; no
   per-token thread spawn. Target 4–8 cores.
4. **Tiled GEMM for prefill** (batch of prompt tokens) with micro-kernels sized
   for 256-bit registers and L2 blocking.
5. **Quantized kernels**: Q8_0 (int8 dot with FMA3), Q6_K/Q4_K (block dequant +
   FMA in registers). Dot-product accumulation in int32 where safe.
6. **Fused epilogues**: RMSNorm+GEMM, RoPE in place, SwiGLU fuse, residual add.
7. **KV cache**: GQA-aware layout, optional F16/Q8 cache, paged blocks instead
   of one contiguous buffer.
8. **Prefetch hints** (`_mm_prefetch`) for the next weight block.

**Gate:** microbenchmarks committed; GEMV reaches ≥60% of the 20 GB/s
single-core bandwidth ceiling on the sandbox; decode on `freemoe-coder-v1`
Q4_K ≥40 tok/s on 8 cores; all outputs still within D4 tolerances against the
F32 oracle.

Effort: 4–6 weeks.

### I3 — Expert residency and cache quality

Exists: state machine, generations, LRU/clock, prefetch from previous routing,
checksums, safe eviction.
Missing: cache sizing that reflects reality, hit-rate instrumentation per
layer, frequency-aware replacement, prefetch accuracy, admission control.

Committed evidence shows 0 cache hits and 1 eviction per expert execution in
the 1-slot configuration. That is a configuration artifact, but it also shows
the cache is not yet a performance feature.

Tasks:

1. LRU **plus expert-frequency** admission (LFU/ARC-style); paper-tune on
   recorded routing traces.
2. Prefetch across layers (depth-2 lookahead), not just next token.
3. Track per-layer hit rate, wasted prefetch bytes, and stall time in the
   report; expose `--cache-stats`.
4. Guard the invariant "evict only unreferenced, fence-complete slots" with a
   stress test under 4+ worker threads.
5. Slot granularity: one expert record (gate+up+down) today; support splitting
   a large expert across slots for VRAM-constrained devices.

**Gate:** ≥80% expert-cache hit rate at 12 GB-equivalent budget on a recorded
27B-class routing trace; zero stale-generation corruptions in a 10-minute
multi-threaded soak.

Effort: 2–3 weeks.

### I4 — I/O engine

Exists: buffered/mmap/`O_DIRECT` readers, coalescing, demand/prefetch futures,
source-change detection.
Missing: real device tuning, GPUDirect/`io_uring`, compression, measured
sustained throughput.

Tasks:

1. `io_uring` (Linux) and overlapped I/O (Windows) backends behind the existing
   `PositionalReader` interface.
2. Adaptive read sizes from measured latency; separate queues for demand vs
   prefetch; starvation guard (already partially present).
3. Optional per-expert compression (zstd) with checksum-then-decompress.
4. Batch multiple expert reads into one `preadv` when extents are close.

**Gate:** sustained ≥2 GB/s cold reads with <1% demand-starved tokens on the
test NVMe; measured cold vs warm first-token latency recorded.

Effort: 2–3 weeks.

### I5 — Device backends (real GPU)

Exists: placeholder that fails closed. This is correct and stays.
Missing: any real GPU execution.

Tasks (only after I1–I3 gates pass on CPU):

1. Pick **one** API first (CUDA, HIP, or Vulkan) based on the actual hardware.
2. Device buffer manager, pinned host staging, async copy streams, events.
3. Port GEMV/GEMM/attention/quantized kernels; keep CPU oracle as reference.
4. Real VRAM accounting; refuse to advertise VRAM we do not own.
5. `--backend gpu` must fail loudly when no device is present, never fall back
   silently (already the rule; keep it).

**Gate:** identical outputs vs CPU within D4 tolerance on real hardware;
measured VRAM high-water mark; no silent fallback events in traces.

Effort: 6–10 weeks for one API.

### I6 — Serving layer

Exists: nothing.
Missing: everything a user expects from an inference server.

Tasks:

1. `freemoe serve` with HTTP/JSON: `/v1/completions`, `/v1/chat/completions`,
   `/health`, `/metrics` (OpenAI-compatible shapes).
2. **Streaming** via SSE with token-level flush.
3. **Continuous batching**: admission, per-sequence KV blocks, prefill/decode
   interleaving, fair scheduling.
4. Sampling suite: temperature, top-k, top-p, min-p, repetition penalty, seed,
   stop strings; deterministic under seed.
5. Chat templates from GGUF metadata; tool/JSON grammar constraints later.
6. Cancellation, timeouts, max-context enforcement, backpressure.
7. Concurrency tests: 8+ simultaneous streams, no cross-sequence KV corruption.

**Gate:** concurrent load test with per-stream output equality vs sequential
runs at temperature 0; documented throughput at N=1/4/8.

Effort: 4–6 weeks (without continuous batching) / +3–4 weeks with batching.

### I7 — Advanced inference features

Deferred until I1–I6 are green:

- speculative decoding (draft model, acceptance statistics);
- KV cache reuse across requests (prefix caching);
- paged attention with eviction of cold prefixes;
- LoRA/adapter hot-swap on top of frozen base weights;
- multi-GPU expert sharding with all-to-all;
- constrained decoding (JSON schema / grammar);
- multimodal input (only with a concrete model contract).

Each needs a dated row in `docs/DECISIONS.md` with hypothesis, baseline, gate
and rollback, per the existing rule.

### I8 — Observability and evidence

Exists: JSON traces, metrics, report, evidence collector.
Missing: per-op profiling, flamegraph-friendly output, CI-enforced budgets.

Tasks: per-kernel timing histograms, allocation high-water tracking per tag,
`perf`/Tracy hooks, a `--profile` mode, and regression thresholds in CI so a
slow commit fails the build rather than being discovered later.

### I9 — Platform and packaging

- Windows x64 build (MSVC + CMake) with PowerShell docs, as the external plan
  assumes; currently Linux-only verified.
- CMake parity with GNU Make (currently optional and unverified).
- Static release binaries, versioned artefacts, reproducible builds.
- GitHub Actions: build matrix, unit tests, fuzz smoke, evidence regeneration.

### I10 — Security and robustness

- Fuzz every new parser surface (tokenizer, chat template, GGUF writer,
  server JSON).
- Bound all request sizes; reject NaN/Inf logits; timeout pathological prompts.
- Path-traversal and model-controlled-path protection in the server.
- No arbitrary shell execution from the governor/tool layer.

---

## 4. Training workstreams (`freemoe-coder-v1`)

The supplied architecture asks for a real training pipeline in native C++ with
no mandatory PyTorch. That is a large, separate system. Nothing exists today.

### T0 — Decisions that block training

Per the external spec, training must not start until these are fixed and
recorded:

1. teacher checkpoint + revision + licence;
2. tokenizer (vocabulary size follows from it — do not invent it);
3. dataset manifests, licences, filtering, dedup;
4. target hardware and memory;
5. training framework and resume strategy;
6. validation benchmark and held-out split;
7. seeds and hyperparameters (`α, β, λ, τ` recorded, not hidden constants).

**Gate:** a `docs/TRAINING_DECISIONS.md` row per item with hashes.

### T1 — Tensor and model-definition layer

- Versioned `freemoe-coder-v1` config with schema validation.
- Canonical logical tensor registry (role → name/shape/dtype), independent of
  source naming.
- Loader that builds a trainable module tree from config + checkpoint.

### T2 — Autograd

- Explicit tape or static graph with double-precision gradient checking.
- Backward for: embedding lookup, RMSNorm, linear (incl. transposed weight),
  RoPE, causal GQA attention (with mask), softmax, SwiGLU, top-k router with
  straight-through/selected-softmax semantics, residual add, cross-entropy.
- Finite-difference gradient tests for every op at small sizes.

**Gate:** max relative gradient error < 1e-5 in F64 for every op; a 2-layer
tiny model trains to overfit 20 examples and the loss curve is reproducible
under seed.

### T3 — Optimizer, schedule, precision

- AdamW with decoupled weight decay, gradient clipping, LR warmup/cosine.
- Mixed precision (F16/BF16 params + F32 master weights) where the platform
  supports it; deterministic accumulation order for reproducibility.
- Gradient accumulation and micro-batching.

### T4 — Memory strategy for training

Disk-first means bounded memory, not memory-free. Requires:

- activation checkpointing (recompute attention/MLP blocks);
- optimizer state offload to disk in the native blob store;
- gradient-ready expert streaming (only selected experts' gradients are
  materialized, but all selected experts must receive updates — no dropped
  experts);
- budget enforcement identical to inference.

**Gate:** a 20-layer MoE forward+backward runs inside a declared RAM budget with
peak RSS measured and stored in evidence.

### T5 — Native blob store and checkpoints

Per the external docs and `docs/external/.../BLOB_FORMAT.md`:

- fixed-width header, tensor directory, payload, checksums, manifest generations;
- staging writes, atomic publish, never overwrite the last good checkpoint;
- recovery from truncation, checksum failure and process kill;
- separate training state (step, optimizer, RNG, dataset position, config hash);
- byte-exact round-trip test: source → blob → reconstructed tensor.

**Gate:** kill -9 during checkpoint write always resumes from the last valid
generation; corruption injection suite passes; round-trip is byte-exact for
F32/F16.

### T6 — Data pipeline

- Deterministic sharded dataset reader with a manifest and position tracking.
- Tokenization with the pinned tokenizer, packing, masking, dedup.
- Teacher-generated example collection (portable across architectures).
- Optional logit/hidden-state distillation when the teacher runtime exposes it.

### T7 — Training loop and evaluation

- `train`, `resume`, `evaluate`, `compare-candidate` commands with strict
  schemas; no no-op commands.
- Coding evaluation harness (compile/run generated code, pass@k protocol).
- Router telemetry: entropy, per-expert token counts, dead experts, max load.
- Regression rule: keep the last known-good checkpoint if evaluation regresses.

**Gate:** documented held-out improvement over the initial checkpoint; resumed
run matches uninterrupted run within a recorded tolerance.

### T8 — Distillation and export

- Distillation loss with tokenizer-alignment guard (same tokenizer first).
- Export to GGUF: independent parser must accept the file; per-tensor manifest;
  F16 first, then quantized variants.
- Runtime parity against the training reference.

**Gate:** exported GGUF runs in FreeMoe and reproduces the training-time logits
within tolerance.

### Honest sizing

A correct, tested, native C++ training stack for this architecture is **3–5
months** of focused work, and it is only worth doing after the inference kernel
and quantization work exists, because training reuses those kernels. The
external plan is right to stage it behind the storage and parity gates.

---

## 5. Recommended sequencing

Ordered so each phase makes the next one cheaper and nothing is built on an
unverified foundation.

| Phase | Content | Exit gate |
|---|---|---|
| P0 (now) | I0 refactors + I1.1 generic GGUF inspector + I1.2 GGUF writer | Any GGUF inspectable; `freemoe-tiny-v1` unchanged |
| P1 | I1.3 Llama adapter + I1.4 real tokenizer + I1.6 detokenize | Real small Llama GGUF generates correct greedy tokens |
| P2 | I2 kernels (AVX2 GEMV, threads) + I7 observability basics | ≥40 tok/s on a 135M F16 model; microbench committed |
| P3 | I1.7 Q8_0/Q6_K/Q4_K + I2 quantized kernels | `freemoe-coder-v1` Q4_K ≥40 tok/s, parity held |
| P4 | I3 cache quality + I4 I/O engine | 27B trace ≥80% hit rate; ≥2 GB/s cold reads |
| P5 | I6 serving layer | OpenAI-compatible streaming server, 8 concurrent streams |
| P6 | I5 real GPU backend (one API) | GPU parity + real VRAM accounting |
| P7 | T0–T2 training foundations (autograd + gradcheck) | Gradient gates green |
| P8 | T5 blob store + T4 memory strategy | Crash-safe resume, bounded-RSS training |
| P9 | T3/T6/T7 full training + eval | Documented held-out improvement |
| P10 | T8 distillation/export + I7 advanced inference | Exported model runs; speculative/LoRA gated |

Phases P0–P3 are the shortest path to "fast inference on a real model".
P7–P10 are the shortest path to "our own trained architecture".

---

## 6. What is missing that only you can decide

These block real progress and cannot be guessed:

1. **Tokenizer and vocabulary.** `V` determines embedding size and the whole
   parameter count. "Must match tokenizer" is not a value.
2. **Teacher model** and its licence, if distillation is the plan.
3. **Target hardware**: exact CPU, RAM, NVMe, and whether there is a real GPU
   (vendor, model, VRAM, driver). Every performance number depends on this.
4. **GPU API preference**: CUDA, HIP, or Vulkan.
5. **Windows support**: required for release, or Linux-first?
6. **Quantization default** for the first shipped model (Q4_K vs Q6_K vs F16).
7. **Scope of the first release**: inference-only, or inference + training in
   the same milestone (I recommend inference first).

---

## 7. Immediate next three pull requests

Small, reviewable, each with tests:

1. **PR A — Generic GGUF inspector.** Container-level inspection of arbitrary
   GGUF files, capability matrix, no execution. Closes external Gate 1.
2. **PR B — C++ GGUF writer + round-trip test.** Write a `freemoe-tiny-v1`
   file from C++ and prove byte-identical reopen against the Python fixtures.
3. **PR C — AVX2 GEMV + thread pool behind the backend interface.** Keep the
   scalar path as the oracle; add a benchmark harness that fails CI on
   regression beyond a threshold.

PR A and B are low risk and unlock everything else. PR C is where the
"extremely high performing" work begins.

---

## 8. Summary answer

**For advanced inference, what remains is:** real model adapters, a real
tokenizer with detokenization, quantized kernels (F16/Q8_0/Q6_K/Q4_K), a
SIMD + multi-threaded GEMM, a per-layer-aware expert cache, a tuned I/O engine,
a real GPU backend for one API, a streaming OpenAI-compatible server with
continuous batching, sampling beyond greedy, KV paging/reuse, speculative
decoding, LoRA, observability with CI-enforced budgets, Windows/CMake parity
and packaging.

**For training, what remains is:** pinned teacher/tokenizer/dataset decisions,
a native autograd with gradient checks, backward kernels for every op in the
`freemoe-coder-v1` graph, AdamW/mixed precision, activation checkpointing and
optimizer offload, the crash-safe native blob store, a deterministic data
pipeline, the training/eval CLI, router telemetry, distillation and GGUF
export with parity.

**The single most valuable next step** is not a new feature: it is making the
compute and memory path real (quantized kernels + threading + a cache that
holds hot experts), because on the current 16-wide fixture every performance
number is meaningless and every real model would be unusably slow.
