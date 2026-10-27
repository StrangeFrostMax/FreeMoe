# Virtual NPU 1.0 — standalone CPU accelerator for LLM workloads

**Virtual NPU** is a dependency-free software device that turns ordinary CPU-only
VMs into accelerator-style inference targets. It replaces the old PCem/Voodoo
emulator coupling entirely: there is no SDL, wxWidgets, Voodoo renderer, PCem
header, BIOS, display or graphics driver anywhere in the build.

The runtime can be consumed in four ways:

1. **`libvnpu.so` / `libvnpu.a` / `vnpu.dll`** – stable C ABI with a dynamic
   backend vtable (`vnpu_get_backend_v1`).
2. **`vnpu-cli`** – info, selftest, benchmark, autotune, trace, node server,
   cluster demo, speculative demo and synthetic LLM block benchmark.
3. **`vnpu-node`** – a standalone network appliance that keeps weights resident
   and answers binary GEMM requests.
4. **Legacy compatibility** – the original `virtual_npu_*` and `vgpu_npu_*`
   APIs still compile, but now execute on the new kernels.

It is deliberately called a *virtual* NPU: it is not pretending to be a physical
NPU, and it does not bypass the CPU. It gives an engine a device-like contract —
streams, buffers, command packets, doorbells, interrupts, status registers,
completion events, DMA and telemetry — while mapping the work onto AVX2/FMA,
AVX-512/VNNI, NEON or scalar CPU kernels.

---

## Why this rewrite exists

The original project had two coupled trees:

- `npu/` – a standalone batched/tensor matmul experiment;
- `pcem/src/video/vgpu_npu.c` – a PCem Voodoo renderer turned into a compute
  unit with a 2 GB flat address space and a tiny unit-1 ALU.

That design was hard to use from a real LLM engine: PCem was a hard dependency,
matmul was expressed incorrectly in one of the PCem tests, Sherlock tracing used
a hard-coded `0.0005` second delay, the API was not `dlopen`-friendly, and there
was no async device model, quantization, transformer operators, cluster, or
remote execution.

This repository keeps the old experiments under `legacy/` and the original zip
under `archive/`, but the buildable product is now a clean runtime under
`include/`, `src/`, `tools/` and `bindings/`.

---

## Quick start

```bash
make -j$(nproc)
make test                      # selftest + shape fuzzer + mini cluster/spec demos
./bin/vnpu-cli info
./bin/vnpu-selftest
./bin/vnpu-fuzz 100
./bin/vnpu-cli bench --shape 512x512x512 --dtype both --iterations 5
./bin/vnpu-cli autotune --shape 1024x1024x1024 --dtype i8
./bin/vnpu-cli trace --shape 512x512x512 --output trace.json
cc -O2 -Iinclude examples/dynamic_load.c -ldl -o /tmp/dynamic_load
VNPU_LIBRARY=$PWD/lib/libvnpu.so /tmp/dynamic_load
```

Artifacts:

```text
lib/libvnpu.so        shared library (C ABI)
lib/libvnpu.so.1      SONAME symlink
lib/libvnpu.a         static library
bin/vnpu-cli          all-in-one command line tool
bin/vnpu-node         network node executable
bin/vnpu-selftest     standalone correctness suite
```

CMake also produces a Windows DLL and static `.lib`:

```bash
cmake -S . -B build-cmake -DCMAKE_BUILD_TYPE=Release
cmake --build build-cmake --parallel
```

No BLAS, oneDNN, OpenMP, protobuf, gRPC, CUDA, ROCm, SDL, wxWidgets or PCem is
required. Only a C++17 compiler and POSIX/WinSock threads.

---

## What “proper NPU” means here

The runtime exposes a small virtual device rather than a bag of functions:

| Device feature | Implementation |
|---|---|
| Command processor | fixed-size `vnpu_command_t` packets on ordered streams |
| Compute units | Voodoo FIFO, Wildcat work-stealing, Cell SPU-style DMA |
| Device memory | 64-byte aligned buffers, host/device copy direction flags |
| Streams | one worker thread per stream, FIFO ordering |
| Events | completion objects with timed waits |
| Doorbell / MMIO | `vnpu_mmio_read32/write32`, IRQ mask and callbacks |
| Status registers | commands, FLOPs, bytes, tiles, culls, DMA, errors |
| DMA | explicit copy/memset commands and Cell local-store staging |
| Telemetry | Sherlock text summaries and Chrome Trace JSON |
| Autotuning | empirical tile/unit sweep driven by measured timings |
| Cluster | local multi-device and remote `vnpu-node` workers |
| Future GPU path | `VNPU_UNIT_GPU` and `VNPU_FEATURE_GPU_FUTURE` reserved |

### The three compute-unit styles are real schedulers

They are not cosmetic names over one loop:

- **VOODOO** – classic command-stream pipeline: static FIFO tile order with
  persistent row/column bases. Best for large dense prefill GEMMs.
- **WILDCAT** – programmable work-stealing tile scheduler. It interleaves tile
  order, supports a per-tile cull mask and skips whole zero tiles. Best for
  sparse or masked workloads.
- **CELL** – SPU-style workers. Each worker DMAs A/B panels into its own 256 KiB
  aligned local store before a register-blocked microkernel runs. Best for
  small independent chunks and remote/distributed workers.

`VNPU_UNIT_AUTO` selects a unit from shape, configuration flags and cull mask.
`vnpu-cli autotune` replaces the heuristic with measurements.

---

## Operator surface

The v1 command stream supports:

- `MATMUL_F32` with optional bias, `alpha/beta`, transposed B, fused ReLU/SiLU/GELU;
- `MATMUL_I8` with packed resident INT8 weights and per-row/per-column scales;
- `RMSNORM_F32`, `SOFTMAX_F32`, `ROPE_F32`, `SWIGLU_F32`, `GELU_F32`;
- `ADD_F32`, `MUL_F32`, `ATTENTION_F32` (streaming online softmax);
- `QUANTIZE_I8`, `PACK_WEIGHTS_I8`, `MTP_HEAD_F32`;
- `COPY`, `MEMSET`, `BARRIER`, `NOP`.

Kernels use runtime dispatch:

| CPU | Path |
|---|---|
| AVX-512 VNNI | 4-row × 64-column INT8 microkernel with broadcast-A reuse |
| AVX-512F | 4×16 F32 FMA microkernel and vectorized transformer ops |
| AVX2/FMA | 4×8 F32 microkernel and AVX2 INT8 fallback |
| AArch64 NEON | 4×4 F32 microkernel |
| Generic | scalar correctness path |

All ISA-specific functions are compiled with target attributes, so a single
binary runs on machines without AVX-512. `VNPU_FORCE_ISA=scalar|avx2|avx512`
can downgrade dispatch for debugging or sanitizer builds.

---

## Integrating an LLM engine

A C/C++ engine either links `-lvnpu` or dynamically resolves the vtable:

```c
#include "vnpu_backend.h"

void *lib = dlopen("libvnpu.so", RTLD_NOW);
const vnpu_backend_v1_t *(*get)(void) =
    (const vnpu_backend_v1_t *(*)(void))dlsym(lib, "vnpu_get_backend_v1");
const vnpu_backend_v1_t *npu = get();

vnpu_config_t cfg;
npu->config_init(&cfg);          /* or use vnpu_config_init from vnpu.h */
cfg.num_threads = 0;             /* auto */
vnpu_device_t *dev;
npu->device_create(&cfg, &dev);

vnpu_stream_t *stream;
npu->stream_create(dev, 0, &stream);

vnpu_matmul_f32_args_t mm = {
    .A = activations, .B = weights, .bias = bias, .C = output,
    .M = tokens, .K = hidden, .N = out_dim,
    .alpha = 1.0f, .beta = 0.0f,
    .epilogue = VNPU_EPILOGUE_NONE,
    .unit = VNPU_UNIT_AUTO
};
npu->matmul_f32(stream, &mm);
npu->stream_sync(stream);
```

`vnpu_backend_v1_t` contains every public function as a pointer, so an engine
can ship one binary and discover the NPU at runtime. See `docs/INTEGRATION.md`
for GGML/llama.cpp-style op mapping, Python bindings and out-of-process usage.

---

## Sherlock tracing and iterative tuning

Sherlock is now real instrumentation:

- monotonic nanosecond timestamps, no hard-coded delays;
- per-command duration, bytes, FLOPs, opcode, unit and worker id;
- dispatch / memory / compute / sync buckets;
- per-opcode call counts and hot-spot list;
- roofline-style verdict;
- Chrome Trace JSON output for `chrome://tracing` or Perfetto UI;
- ~25–45 ns/event overhead on the test machine.

```bash
VNPU_TRACE=1 ./bin/vnpu-cli llm-bench --hidden 512 --intermediate 1376 \
  --heads 8 --layers 2 --prefill 32 --decode 1 --dtype both \
  --trace llm-trace.json
./bin/vnpu-cli autotune --shape 2048x4096x11008 --dtype i8 --iterations 3
```

The autotuner sweeps tile geometries and Voodoo/Wildcat/Cell schedulers,
disables tracing during measurement, restores the previous state, and writes
the winning configuration back to the device.

---

## Clustering and distributed speculative decoding

`vnpu_cluster_*` treats local devices and remote `vnpu-node` processes as
weighted workers. Weight matrices are uploaded once with a stable key; each
inference call sends only activations and receives only results.

```bash
# machine A (draft/MTP model or head)
./bin/vnpu-node --port 9449 --token secret --name draft

# machine B (main verifier model)
./bin/vnpu-node --port 9450 --token secret --name verify
```

The client can then:

- split one GEMM by rows across both nodes;
- run a DFlash/MTP-style draft head on machine A;
- run the main verifier on machine B;
- verify a batch of `gamma` draft tokens in one network round trip.

`vnpu-cli specdemo` starts two nodes in-process and demonstrates the pipeline.
`vnpu_speculative_step()` accepts arbitrary logits callbacks, so it can wrap a
real draft model, an MTP head, or a remote inference service. v1 uses greedy
acceptance; temperature affects draft/bonus sampling. Exact stochastic
acceptance needs draft probabilities in a future callback version.

Network v1 is a compact binary TCP protocol with a token. It is intentionally
**not encrypted**; run it over WireGuard, Tailscale, an SSH tunnel or an mTLS
sidecar. See `docs/CLUSTER.md`.

---

## Measured performance

Sandbox: 2 logical Intel Xeon vCPUs, AVX-512F + AVX-512 VNNI, GCC 14, `-O3`,
`make benchmark`. These numbers compare against a naive scalar triple-loop
baseline in the same binary; they are **not** a claim against optimized BLAS.
The VM is noisy, so `benchmarks/*.json` is the source of truth for a given run.

### Dense GEMM, 512×512×512

| Path | Time | Throughput | Speedup vs scalar |
|---|---:|---:|---:|
| Naive scalar baseline | ~168–205 ms | ~1.3 GFLOP/s | 1.0× |
| F32 Wildcat (`bench`) | 2.69 ms | 99.8 GFLOP/s | 59.1× |
| F32 Voodoo (`bench`) | 3.16 ms | 85.0 GFLOP/s | 50.3× |
| INT8 AUTO (`bench`) | 3.15 ms | 85.2 GFLOP/s | 50.5× |
| Autotuned F32 | 4.38 ms | 61.3 GFLOP/s | 38.3× |
| Autotuned INT8 VNNI | 2.02 ms | 132.8 GFLOP/s | 85.5× |

Other representative results from the final run:

- 256³ F32 Wildcat: 0.528 ms, 63.5 GFLOP/s, 34.6×;
- 256³ INT8 Wildcat: 0.443 ms, 75.8 GFLOP/s, 41.3×;
- 128³ F32 Wildcat: 0.199 ms, 21.0 GFLOP/s, 12.8×.

The original 12–20× goal is exceeded for medium/large GEMMs on this CPU-only
VM. Tiny decode-shaped GEMMs are dominated by dispatch and memory latency; use
`CELL`, fuse operators, keep weights packed, or use cluster/speculative
batching.

### Synthetic transformer block

`llm-bench --hidden 512 --intermediate 1376 --heads 8 --layers 2`:

| Phase | F32 | INT8 |
|---|---:|---:|
| prefill 32 tokens | 1,959 tok/s, 25.0 GFLOP/s | 2,741 tok/s, 35.0 GFLOP/s |
| decode 1 token | 88.4 tok/s | 96.0 tok/s |

This is a realistic op mix (RMSNorm, RoPE, attention, QKV/O/gate-up/down GEMMs,
SwiGLU), not a tokenizer or sampling engine.

### Cluster / speculative demo on one 2-vCPU box

- Two loopback nodes, 256³ GEMM: correct result, 1.35× effective worker
  parallelism, but 0.86× wall-clock vs one local device because ~1.8 MB crossed
  TCP for a ~5 ms kernel. On separate machines and larger work this inverts.
- Distributed draft/verify demo: 20 tokens in 13.8 ms, 16/20 draft tokens
  accepted, theoretical 1.08–1.47× for the toy model. Real gains require a
  cheap draft head and one batched verifier round trip.

Full JSON/Chrome outputs are in `benchmarks/`.

## Repository layout

```text
include/                  public C ABI, backend vtable, legacy shims
src/core/                 runtime, scheduler, autotune, backend table
src/kernels/              SIMD GEMM + transformer kernels
src/cluster/              sockets, node server, cluster, speculation
src/compat/               PCem-free virtual_npu/vgpu_npu implementation
tools/                    CLI, node, selftest, demos, LLM benchmark
bindings/python/          ctypes wrapper
examples/                 dynamic loader and engine integration examples
tests/                    fuzz/property tests
docs/                     architecture, integration, cluster, tuning, compat
scripts/                  benchmark matrix and trace report helpers
legacy/                   original experiments (not built)
archive/                  original npu_experiments.zip (not built)
benchmarks/               generated measurement artifacts
release/                  release tarball
```

`make pcem-check` verifies that no buildable runtime source includes PCem,
Voodoo renderer, SDL or wxWidgets headers.

---

## Compatibility and migration

Old tests can include `virtual_npu.h` or the original
`includes/private/video/vgpu_npu.h` path; these are shims over
`vnpu_compat.h`. Graphics-only calls (`draw_triangle`, `swapbuffers`, texture
upload/bind) are deterministic no-ops with status accounting. Matmul and unit-1
compute are real.

New code should use `vnpu.h` directly. See `docs/COMPAT.md`.

---

## Roadmap

- GPU backend behind `VNPU_UNIT_GPU` with the same stream/event/buffer ABI.
- INT4 / grouped quantization and FP8/BF16 activation paths.
- Real GGUF/GGML backend plugin and ONNX Runtime execution-provider shim.
- RDMA/libfabric transport and weight sharding instead of row splitting.
- Exact stochastic speculative acceptance with draft probabilities.
- NUMA-aware worker placement and cgroup-aware thread detection.
- Windows DLL and ARM NEON CI matrix.

See `docs/ROADMAP.md` for the full plan.

---

## License

MIT. See `LICENSE`.

**Honest note:** a virtual NPU cannot create compute that the CPU does not have.
The gains come from removing the emulator, keeping data resident, using SIMD and
VNNI, batching commands, culling empty work, and overlapping local/remote
workers. Measure against your own model and baseline; the benchmarking and
tracing tools are included specifically for that.
