# Virtual NPU rewrite — engineering report

## Outcome

The PCem dependency is gone from the buildable runtime. In its place is a
standalone virtual NPU that can be consumed as a shared library, static
library, Windows DLL (via CMake), CLI tool, or network node. The original
experiments are preserved under `legacy/` and `archive/`, but no PCem/Voodoo
renderer, SDL, wxWidgets or emulator source is compiled.

All correctness tests pass:

- `16 tests, 0 failures` in `vnpu-selftest`;
- `120 iterations, 0 failures` in `vnpu-fuzz`;
- `make test` including local cluster and speculative demos;
- dynamic-loader C example verified;
- Python ctypes example verified;
- `make pcem-check` confirms buildable sources are PCem-free.

## What was built

### 1. Stable C ABI and device model

`include/vnpu.h` and `include/vnpu_backend.h` define a versioned C ABI:

- device/stream/event/buffer/packed-weight handles;
- fixed-size `vnpu_command_t` packets with 17 opcodes;
- virtual MMIO registers, doorbell, IRQ mask/callback;
- status counters for commands, FLOPs, bytes, tiles, culls, DMA;
- `vnpu_get_backend_v1()` function table for `dlopen`/`LoadLibrary`.

### 2. Three real compute-unit schedulers

- **Voodoo** – FIFO command-stream tiles;
- **Wildcat** – interleaved work-stealing tiles with optional cull mask;
- **Cell** – SPU-style workers with explicit DMA into 256 KiB local stores.

They share runtime-dispatched kernels but have different tile geometry,
ordering, culling and memory behavior. `AUTO` and `vnpu_autotune_gemm()`
choose among them empirically.

### 3. SIMD and quantized kernels

- AVX-512F 4×16 F32 FMA GEMM;
- AVX2/FMA 4×8 F32 GEMM;
- AArch64 NEON 4×4 F32 GEMM;
- scalar correctness path;
- transposed-B dot-product path for decode;
- AVX-512 VNNI INT8 GEMM with packed weights, per-row/per-column scales,
  broadcast-A reuse, 4 M rows × up to 64 N columns, local-store A-vector
  precompute for large K;
- RMSNorm, Softmax, RoPE, SwiGLU, GELU, ADD, MUL, streaming attention,
  quantization and MTP head.

A thread-pool lifetime bug found by AddressSanitizer was fixed with a
heap-allocated parallel state and worker-done barrier.

### 4. Sherlock tracing and autotune

Sherlock now records real nanosecond timestamps instead of the original
hard-coded 0.5 ms. It emits text roofline summaries and Chrome Trace JSON with
per-command bytes, FLOPs, opcode, unit and worker. Autotune sweeps tile sizes
and units with tracing disabled, then restores the previous state.

### 5. Cluster and distributed speculation

- `vnpu-node` binary TCP server with token, persistent connections, resident
  weights and GEMM requests;
- `vnpu_cluster_*` API splitting GEMM rows across local devices/remote nodes;
- `vnpu_speculative_step()` DFlash/MTP-style draft/verify coordinator;
- `vnpu-cli cluster-demo` and `specdemo` start two in-process nodes and
  demonstrate both modes.

v1 uses greedy acceptance; exact stochastic acceptance and TLS/RDMA are on the
roadmap.

### 6. Legacy compatibility

`src/compat/compat.cpp` implements the original `virtual_npu_*`, `vgpu_npu_*`
and Sherlock APIs without PCem. Graphics-only calls are deterministic no-ops;
matmul and unit-1 ALU compute are real. Address truncation and cull semantics
were fixed.

### 7. Documentation, bindings and tooling

- root `README.md`;
- `docs/ARCHITECTURE.md`, `INTEGRATION.md`, `CLUSTER.md`, `PERFORMANCE.md`,
  `TUNING.md`, `COMPAT.md`, `SECURITY.md`, `ROADMAP.md`;
- `examples/dynamic_load.c`, `examples/engine_adapter.cpp`;
- `bindings/python/vnpu.py`;
- `scripts/bench_matrix.sh`, `autotune_all.py`, `trace_report.py`;
- `tests/fuzz_shapes.cpp`;
- `Makefile` and `CMakeLists.txt`.

## Measured gains

Reference VM: 2 logical Xeon vCPUs with AVX-512 VNNI. Baseline is a naive
scalar triple-loop in the same binary.

- 512³ F32 Wildcat: **2.69 ms, 99.8 GFLOP/s, 59.1×**;
- 512³ INT8 AUTO: **3.15 ms, 85.2 GFLOP/s, 50.5×**;
- autotuned 512³ INT8 VNNI: **2.02 ms, 132.8 GFLOP/s, 85.5×**;
- 256³ INT8 Wildcat: **0.443 ms, 75.8 GFLOP/s, 41.3×**;
- synthetic transformer block prefill: **1,959 tok/s F32**,
  **2,741 tok/s INT8**;
- decode: **88.4 tok/s F32**, **96.0 tok/s INT8**.

The 12–20× target is exceeded for medium/large GEMMs. Tiny decode shapes remain
latency-bound; clustering over loopback TCP is slower than local execution for
small kernels, as expected from network overhead.

## Honest limitations

- This is a CPU software device, not a physical NPU.
- Speedups are vs a naive scalar baseline, not optimized BLAS.
- No full model loader, tokenizer, sampler or KV-cache manager is included.
- Network v1 is plaintext/native-endian; use a VPN or TLS proxy.
- Speculative v1 uses greedy acceptance.
- GPU support is reserved in the ABI but not implemented.

## Reproduce

```bash
make clean
make -j$(nproc)
make test
make benchmark
make release
```

Deliverables are `lib/libvnpu.so`, `lib/libvnpu.a`, `bin/vnpu-cli`,
`bin/vnpu-node`, `bin/vnpu-selftest`, `bin/vnpu-fuzz`, and
`release/vnpu-standalone-1.0.0.tar.gz`.
