# Decisions log

All entries are dated 2026-10-27 unless stated otherwise. Values delegated to
the agent by the user (“you choose a best way”) are marked **provisional** and
must be changed through a commit that records the reason and date.

| ID | Decision | Value / evidence | Status |
|---|---|---|---|
| D1 | Test machine | Validation sandbox recorded in `docs/TEST_MACHINE.md` with `lscpu`, `free -h` and device-query output. Target profile is Zen 3/AVX2+FMA3; exact CPU model, RAM, storage, physical GPU/NPU, driver and usable VRAM are **unknown (D1)**. | Sandbox complete; physical-GPU gate closed. |
| D2 | Dense checkpoint | `tests/fixtures/dense-fm.gguf`, SHA-256 `4d81aca51f3d087c977fc4b40d3227401d225b45373878b618a72059d35885a1`, `freemoe-tiny`, F32, MIT. Record: `docs/CHECKPOINT_DENSE.md`. | Complete. |
| D3 | MoE checkpoint | `tests/fixtures/moe-fm.gguf`, SHA-256 `8f1703d7863588347e318000526938a8b1313bfaf2eb0bee1bc7c1aaa985ebc8`, 8 experts, top-2, F32, MIT. Record: `docs/CHECKPOINT_MOE.md`. | Complete. |
| D4 | Numerical tolerances | F32 logits: absolute `0.005`, relative `0.005`; greedy token IDs and MoE routing IDs must match exactly. See `docs/BENCHMARK_PROTOCOL.md`. | Provisional, complete for synthetic fixtures. |
| D5 | Reference runtime | Independent scalar Python oracle `tools/reference_oracle.py`, pinned at tree `deee8aa67b9615a0ef8cd73aaa4c8aca6469cfa5`, Python 3.11.2, no build flags. Fixture generator `tools/make_fixtures.py` at same tree. | Complete. |
| D6 | Backend | Dual path: scalar CPU oracle + vendored patched Virtual NPU; physical GPU is a failing placeholder. VNPU ABI 1, vendored C++ runtime, no dedicated VRAM. See `docs/BACKEND_SUPPORT.md`. | CPU/VNPU complete; physical GPU closed. |
| D7 | Parameters | `N_greedy=16`, `N_soak=256`, `F=0.02h` (72 s), `T_wait=100ms`, `G_bytes=4096`, `R=5`. Provisional target: median prefill >= 1000 tok/s and p95 inter-token latency <= 50 ms on the sandbox synthetic fixture. | Provisional, committed before runs. |
| D8 | Prompts/reference IDs | `tests/reference/prompts.txt` has 20 prompts from 8 to 512 byte tokens. Each prompt has `prompt_XX.txt`, `prompt_XX.input.ids`, dense `prompt_XX.greedy.ids`, MoE `prompt_XX.greedy.ids`, logits and routing traces. SHA-256 values are in `tests/reference/manifest.sha256` and reproduced in `docs/BENCHMARK_PROTOCOL.md`. | Complete. |
| D9 | Decision log | This file; any change to D1–D8 must add a dated row here and update the affected spec artefact. | Complete. |

## Scope decisions from the user

1. **First target:** support both a scalar CPU correctness oracle and the
   supplied Virtual NPU backend, with a placeholder for a later physical GPU.
2. **First deliverable:** a paged MoE prototype, not a documentation-only plan.
3. **NPU integration:** vendor the pinned source into this repository rather
   than dynamic-loading a separate build.
4. **Checkpoints:** the agent may select tiny permissive synthetic checkpoints
   and an independent reference implementation.
5. **Numeric parameters:** the agent may choose conservative provisional values.
6. **Zen 3 ISA:** target dual 256-bit AVX2/FMA3, not AVX-512.

## NPU patch decision

The supplied `npuxl-improved-rinai` repository described itself as a CPU
software device. It was vendored at commit
`4298c454aa56b90dac240e81f80ed4467221c1dd` and patched locally for Zen 3:

- dual independent AVX2/FMA3 accumulator chains for NN and transposed-B GEMM;
- safe explicit ISA forcing using the ABI reserved config slot;
- Zen 3 detection and `VNPU_FEATURE_AVX2_DUAL_FMA`;
- detection-driven fallback: `znver3` overrides AVX-512 capability dispatch,
  and FreeMoe's `zen3-auto`/`auto`/`avx512` profiles all start on dual AVX2 on
  Zen 3;
- explicit `device-info`/trace diagnostics, simulated-Zen-3 tests,
  scalar-oracle regression tests and GCC 12 warning fix.

See `third_party/vnpu/PATCHES_FREEMOE.md` and `docs/BACKEND_SUPPORT.md`.
No AVX-512 result from the upstream README/REPORT is used as a target result.

## Deferred decisions

- Phase 7 custom packed store: **not scheduled**. GGUF-backed reads and
  coalescing measurements do not yet justify a new format.
- Phase 8 advanced optimisations (GPU-side remapping, learned prefetch,
  speculative decoding, batching, multi-GPU): **not scheduled**. Each needs a
  decision row, hypothesis, baseline, correctness gate and rollback path.
- Exact physical GPU backend/API version: **unknown (D6)** until hardware/SDK
  details are supplied.
- Production checkpoint support: deferred until one real dense and one real MoE
  GGUF with known-good outputs, licence and tokenizer provenance are recorded.
