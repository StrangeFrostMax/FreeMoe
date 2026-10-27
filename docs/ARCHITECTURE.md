# FreeMoe architecture

## Status

This repository implements a **correctness prototype** of the architecture in
`Final_Llama_Like_Disk_Resident_MoE_Spec.md`. It deliberately supports one
documented synthetic model family (`freemoe-tiny-v1`) rather than pretending
to support arbitrary GGUF Llama/MoE checkpoints.

The cold/warm/hot tiers are:

- **cold:** unchanged GGUF source file, read through buffered, mmap or Linux
  `O_DIRECT` paths;
- **warm:** bounded host staging, prefetch host cache and resident-expert mode;
- **hot:** backend-owned buffers used as expert slots. For the CPU and vendored
  Virtual NPU these are host RAM buffers; the code never calls them VRAM.

A physical GPU backend exists only as a failing placeholder. No CUDA, HIP,
Vulkan, DirectStorage or GPUDirect path is claimed.

## Logical component layout

```text
CLI / run recorder
  -> MappedFile + bounded GGUF parser
  -> architecture adapter + canonical model map
  -> Engine (tokenizer, RMSNorm, RoPE, attention/KV, FFN, sampling)
  -> Backend (scalar CPU | patched Virtual NPU | GPU placeholder)
  -> ResidencyManager (state machine, generations, LRU/clock, prefetch)
  -> ReadScheduler (coalescing, demand/prefetch futures, source guard)
  -> PositionalReader (resident/buffered/mmap/direct)
  -> TraceWriter + Json metrics/report
```

## Source module inventory

| Status | Path | Responsibility |
|---|---|---|
| EXISTS | `include/fm/common.hpp` | Typed errors, JSON writer, FNV-1a, file identity, timers, process metrics, trace writer. |
| EXISTS | `include/fm/gguf.hpp` | GGUF value/document types, quantisation geometry, architecture adapter and inspect report. |
| EXISTS | `include/fm/io.hpp` | Extents, readers, coalescing, bounded async scheduler, I/O metrics. |
| EXISTS | `include/fm/backend.hpp` | Device buffers, allocation tags/tracker, fences, operation log, backend interface. |
| EXISTS | `include/fm/residency.hpp` | Slot state machine, budget, replacement, residency manager, prefetch host cache. |
| EXISTS | `include/fm/engine.hpp` | Run configuration, tokenizer, forward pass, reference comparison, run metrics. |
| EXISTS | `src/common.cpp` | Error names/exit codes, file identity with sparse content probe, process snapshot, file helpers. |
| EXISTS | `src/gguf.cpp` | Bounded GGUF v3 parser, strict adapter, golden inspect JSON. |
| EXISTS | `src/io.cpp` | Buffered/mmap/direct readers, coalescing, async prefetch, source-change detection. |
| EXISTS | `src/backend.cpp` | Scalar CPU, patched Virtual NPU and explicit GPU placeholder backends. |
| EXISTS | `src/residency.cpp` | Budget calculator, authoritative state table, safe eviction, checksums, fault handling. |
| EXISTS | `src/engine.cpp` | Llama-like forward pass, MoE routing/execution, reference comparison and reports. |
| EXISTS | `src/main.cpp` | `inspect`/`WAKE_UP`, `device-info`, `run`, `compare` and `version` commands. |
| EXISTS | `tests/unit/tests.cpp` | Unit, malformed-input, integration, correctness and soak tests. |
| EXISTS | `tests/fuzz_gguf.cpp` | Duration-bounded deterministic GGUF mutation fuzz target. |
| EXISTS | `tools/gguf_format.py` | Independent fixture GGUF writer/parser. |
| EXISTS | `tools/reference_oracle.py` | Independent scalar reference runtime. |
| EXISTS | `tools/make_fixtures.py` | Deterministic checkpoints, prompts, references and malformed fixtures. |
| PLANNED | `src/backend/cuda/` | Real CUDA backend; requires D1/D6 hardware and kernel correctness gates. |
| PLANNED | `src/backend/vulkan/` | Real Vulkan backend; requires Windows/AMD evidence. |
| PLANNED | `src/backend/hip/` | Real HIP backend; requires Linux/AMD evidence. |

No empty backend directories are created for PLANNED modules.

## Correctness boundaries

- The adapter rejects unknown tensors, missing roles, wrong shapes, unsupported
  architecture/tokenizer/chat-template values and inconsistent expert metadata.
- Selected experts are never dropped or replaced by zeros. A selected/executed
  mismatch aborts with a fixed error code.
- Source and slot payload checksums are compared against checksums stored in
  GGUF metadata at generation time.
- State transitions, generation IDs, reference counts and fences guard slot
  reuse. Prefetch is explicitly a prediction from previous-token routing.
- CPU fallback for the GPU placeholder is disabled by default and logged per
  operation when explicitly enabled.

## Build dependency inventory

| Name | Version | Licence | Build reference |
|---|---|---|---|
| Virtual NPU | 1.0.0, upstream `4298c454aa56b90dac240e81f80ed4467221c1dd` plus local patches | MIT | `third_party/vnpu/src` |
| C++17 standard library | GCC 12.2 libstdc++ on D1 sandbox | compiler runtime | implicit |
| POSIX threads | OS-provided | OS licence | `Threads::Threads` |

Licence texts are committed in `LICENSES/`. Python tools use only the Python
standard library and are not linked into the C++ runtime. `scripts/check_dependencies.py`
checks this table against the build files.
