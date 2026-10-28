# CPU kernels and dispatch

Status: implemented and measured on the recorded sandbox. Schema of the
machine-generated report: `freemoe.kernel-bench.v1` in
`docs/evidence/kernels_bench.json`.

## Why this exists

Before this change the CPU backend had exactly one kernel:

```cpp
for (uint32_t o = 0; o < out_dim; ++o) {
    float acc = 0.0f;
    for (uint32_t k = 0; k < in_dim; ++k) acc += row[k] * x[k];
    y[o] = acc;
}
```

That is a single-threaded scalar dot product with no vectorisation. It is the
correctness oracle and it stays — but at ~1.7 GFLOP/s it made any real model
unusable, because decoding is dominated by GEMV over hundreds of millions of
weights per token.

## Design

`include/fm/kernels.hpp` / `src/kernels.cpp`:

| Component | Purpose |
|---|---|
| `gemv_scalar` | Deterministic ascending-`k` reference. Never removed, always available. |
| `gemv_isa` | One-call vectorised GEMV with 8-lane (AVX2/FMA3) or 16-lane (AVX-512) dual accumulator chains. |
| `WorkerPool` | Persistent pool, created lazily, used for row-range partitioning. No per-token thread creation. |
| `gemv` | Public entry point: resolves ISA, decides whether threading is worth it, dispatches. |
| `benchmark_gemv` | Deterministic benchmark; used by `freemoe bench` and by evidence collection. |

Key properties:

- **Runtime dispatch without global `-mavx2`.** Vector paths use function-level
  `__attribute__((target("avx2,fma")))`, so the binary still runs on CPUs without
  AVX2 and the scalar path remains valid.
- **Zen 3 policy is shared.** `resolve_kernel_isa(Auto)` calls the same
  `host_is_zen3()` detection as the NPU layer, so a Zen 3 host never selects
  AVX-512 even though the capability bit may be set.
- **Threading has a work threshold** (`2^16` multiply-adds). Below it, the pool
  is not touched, because waking a worker costs more than the GEMV itself.
- **No hidden per-call allocation.** An earlier revision called
  `std::thread::hardware_concurrency()` on every GEMV, which cost ~2.5 µs per
  call and made small operations 40× slower than scalar. It is now cached once
  per process; the regression is covered by the `cpu_kernels` unit test.
- **Determinism.** Reduction order differs between scalar and vector paths, so
  results match to floating-point tolerance, not bit-exactly. The
  `cpu_kernels` test pins that tolerance; the model-level D4 gate (0.005
  abs/rel, exact greedy IDs and routing) still passes unchanged.

## Measured results

Sandbox: `Intel(R) Xeon(R) Processor @ 2.60GHz`, 2 vCPUs, AVX2+FMA and AVX-512
available, not Zen 3. Reproduce with:

```bash
make bench
# or
./build/freemoe bench --iterations 20 --out docs/evidence/kernels_bench.json
```

`best_ms` is the best of the recorded iterations for one GEMV.

| Shape (out × in) | scalar | AVX2/FMA, 1 thread | threaded | speedup (scalar → threaded) |
|---|---:|---:|---:|---:|
| 1280 × 1280 (attention Q/K/V/O) | 1.932 ms | 0.184 ms | 0.156 ms | **12.4×** |
| 1920 × 1280 (expert gate/up) | 2.909 ms | 0.302 ms | 0.262 ms | **11.1×** |
| 1280 × 1920 (expert down) | 2.942 ms | 0.293 ms | 0.188 ms | **15.7×** |
| 32000 × 1280 (output head) | 51.114 ms | 14.874 ms | 8.391 ms | **6.1×** |

Effective throughput reaches 19–52 GB/s of weight streaming, which is at or
above the memory bandwidth a small VM slice provides — i.e. the fast path is
now closer to bandwidth-bound than compute-bound, which is the correct regime
for decode.

## What this means for a real model

Per decoded token, `freemoe-coder-v1` touches roughly 415M active parameters
(295M in experts, 79M attention, 41M tied output head at V=32k). At the measured
streaming rates:

| Weight format | Bytes per token | At 40 GB/s | At 20 GB/s |
|---|---:|---:|---:|
| F32 | 1.66 GB | ~24 tok/s | ~12 tok/s |
| F16 | 0.83 GB | ~48 tok/s | ~24 tok/s |
| Q4_K | 0.24 GB | ~166 tok/s | ~83 tok/s |

These are ceilings, not promises: they assume every byte is streamed once at
full bandwidth. They are the reason quantised kernels are the next milestone
(`docs/ROADMAP_INFERENCE_AND_TRAINING.md`, phase P3), and they are computed from
committed measurements rather than guessed.

## Block-quantised GEMV

Real checkpoints are stored in block formats, so the quantised GEMV fuses
decode and multiply-accumulate instead of dequantising a row into a scratch
buffer and summing it in a second pass. The AVX2/FMA3 form is the one that
matters on the Zen 3 target, where AVX-512 is deliberately not used.

Formats with a fused AVX2 kernel: Q4_0, Q4_1, Q4_K, Q8_0, Q8_1, F16, BF16.
Every other supported format (Q5_0, Q5_1, Q2_K, Q3_K, Q5_K, Q6_K, Q8_K) runs
through the fused scalar path, which decodes one block into a stack buffer and
uses the same ISA dot product. Both paths are checked against the scalar oracle
in `src/quant.cpp` at a relative tolerance of `1e-5`.

Measured on the same 2-vCPU sandbox host, `--iterations 20`
(`docs/evidence/kernels_bench.json`, `quant_results`):

| Shape | Scalar | AVX2 1T | Threaded | Scalar → threaded |
| --- | --- | --- | --- | --- |
| qkv 1280×1280 Q8_0 | 2.360 ms | 0.225 ms | 0.205 ms | 11.5× |
| qkv 1280×1280 Q4_K | 3.260 ms | 0.353 ms | 0.203 ms | 16.0× |
| expert gate 1792×1280 Q4_K | 4.589 ms | 0.490 ms | 0.291 ms | 15.8× |
| expert down 1280×1792 Q4_K | 4.594 ms | 0.491 ms | 0.443 ms | 10.4× |
| output head 32000×1280 Q4_K | 82.853 ms | 9.382 ms | 4.738 ms | 17.5× |

Q4_K reaches ~3.2 GB/s of stored bytes at 1 thread (~5 GB/s threaded) against a
19–52 GB/s memory ceiling, so the decoder — not the memory system — is still the
limit. The obvious next step is to widen Q4_K/Q6_K to two 256-bit accumulator
chains per sub-block.

Correctness for the new path is enforced by fixtures rather than by inspection:
`tools/make_quant_fixtures.py` writes, for each of Q8_0, Q4_0 and Q4_K, a
quantised model and an F32 twin holding exactly the values an **independent
Python decoder** (`tools/quant_reference.py`) produces. `quant_model_parity`
requires identical greedy tokens and logits within `1e-4`. That test is what
caught two real bugs during development: a reference generator that used the
unrounded f16 scale, and nibble kernels that dropped half of every block.

## Known limits (honest list)

- The fused AVX2 kernels cover seven formats. The K-quants other than Q4_K
  (Q2_K, Q3_K, Q5_K, Q6_K, Q8_K) still decode scalar-per-block, which is roughly
  3× slower than a fused AVX2 kernel would be.
- No AVX-512 quantised kernel. The project targets Zen 3, so AVX2 was the
  priority; an AVX-512 variant is a separate measurement.
- GEMV, not GEMM. Prefill of long prompts still processes one token at a time;
  batched/tiled GEMM is a separate milestone.
- Reductions use two 256-bit chains per row. More chains, software pipelining
  and cache blocking are untested optimisations.
- Threading partitions by output row only. NUMA placement and affinity are not
  controlled.
- The sandbox has 2 vCPUs and unknown throttling, so absolute numbers are not
  a hardware target; `performance_qualified` remains **closed** in
  `docs/evidence/release_status.json`.

## Reproduction and gates

- `freemoe bench` also reports the block-quantised shapes (`quant_results`).
- `make test` runs the `cpu_kernels` unit test: vector paths vs scalar oracle,
  tail handling, threaded determinism, ISA policy, benchmark sanity.
- `make bench` writes `docs/evidence/kernels_bench.json`.
- `bash scripts/phase0_exit.sh` regenerates it as part of the full evidence set
  and `release_status.json` records `cpu_kernel_benchmark`.
- Any change to a kernel must keep `cpu_kernels` green and the four final
  comparison runs within D4 tolerance.
