# Backend support

## D6 decision

The first working release supports two real backends plus one explicit
placeholder:

| CLI name | Status | Memory | Acceleration claim |
|---|---|---|---|
| `cpu` | implemented and tested | host RAM | Scalar correctness oracle. |
| `vnpu` | implemented and tested | host RAM virtual device buffers | Vendored Virtual NPU 1.0, patched for AVX2/FMA3. CPU software device, not a physical NPU. |
| `gpu` | placeholder only | none | Fails loudly with `GPU_BACKEND_UNIMPLEMENTED`; optional explicit CPU fallback. |

The user selected a dual path on 2026-10-27: scalar CPU oracle, Virtual NPU
backend, and a placeholder for a later physical GPU backend. No CUDA, HIP or
Vulkan backend exists. A physical GPU SDK/driver/VRAM record remains **unknown
(D1/D6)** and the relevant release gate is closed.

## Zen 3 / AVX2/FMA3 NPU patch

Zen 3 has no AVX-512. It provides two 256-bit AVX2/FMA3 pipes, so the vendored
Virtual NPU was patched at pinned upstream commit
`4298c454aa56b90dac240e81f80ed4467221c1dd`:

1. Added `ISA_AVX2_DUAL` and two independent 256-bit accumulator chains to the
   normal and transposed-B F32 GEMM kernels. Both FMA pipes have ready work
   without executing an AVX-512 instruction.
2. Added Zen 3 CPU detection (`znver3`) and `VNPU_FEATURE_AVX2_DUAL_FMA`.
   Detection takes precedence: the testable `select_detected_isa()` policy
   selects dual AVX2/FMA3 even when AVX-512 capability bits are also reported,
   and the AVX-512 feature is not advertised on that path.
3. Reused the ABI's `reserved0` config slot as `forced_isa`; explicit
   `avx2_dual`, `zen3`, `avx2`, `avx512`, `scalar` or `auto` requests are
   validated against advertised CPU features. Unsupported explicit requests
   fail instead of executing illegal instructions.
4. Added selftests comparing the dual-pipe NN/NT kernels to the scalar oracle,
   verifying forced-ISA validation, and simulating Zen 3 detection with
   contradictory AVX-512 feature bits.
5. Suppressed a GCC 12 false-positive warning from AVX-512 intrinsic headers
   so the vendored source builds under `-Wall -Wextra -Werror`.

## Zen 3 detection fallback

FreeMoe requests `avx2_dual` by default for the VNPU backend. If a host lacks
AVX2, the backend falls back to scalar with an explicit note. In addition,
`host_is_zen3()` uses the compiler's `znver3` CPU identification, and the pure
`plan_vnpu_isa_attempts(profile, zen3_detected)` helper makes the detection
decision testable on any host:

| `--vnpu-isa` | Zen 3 detected | Other x86 hosts |
|---|---|---|
| `zen3-auto` (or `zen3_auto`) | Dual 256-bit AVX2/FMA3 first, then AVX2, then scalar | Runtime auto dispatch, then dual AVX2, AVX2, scalar |
| `auto` | Same Zen 3 fallback chain | Runtime auto dispatch, then dual AVX2, AVX2, scalar |
| `avx512` | Same Zen 3 fallback chain; AVX-512 is never attempted | AVX-512, then dual AVX2, AVX2, scalar |
| `avx2_dual` / `zen3` | Dual AVX2, then AVX2, then scalar | Same |
| `avx2` | AVX2, then scalar | Same |
| `scalar` | Scalar | Scalar |

Thus Zen 3 detection does not merely fall back after an AVX-512 device-create
error: it removes AVX-512 from the attempt plan. `zen3-auto` on a non-target
host may legitimately select AVX-512 through runtime auto dispatch; that is
not the Zen 3 path. Unknown profiles are hard errors.

Diagnostics are explicit. `device-info --json` notes contain
`Zen 3 detected; AVX-512 dispatch disabled in favor of dual 256-bit AVX2/FMA3`
(or `Zen 3 not detected`), plus the selected attempt. The `backend_init` trace
records `requested_profile`, `selected_attempt`, `fallback_used`,
`zen3_detected`, `zen3_detection_simulated` and `zen3_fallback_selected`. Here
`fallback_used` means a later attempt was needed after `VNPU_ERR_UNSUPPORTED`;
`zen3_fallback_selected` means detection chose the dual-pipe path immediately.
The CLI flag
`--test-simulate-zen3` is accepted only by `device-info`; it forces the
detection branch on non-Zen-3 machines for verification, never for inference.

Tests:

- `tests/unit/tests.cpp::test_zen3_fallback` checks the pure attempt planner,
  an end-to-end simulated Zen 3 backend on an AVX-512-capable CI host, device
  info notes and backend init trace fields.
- `third_party/vnpu/tools/selftest.cpp::zen3_detection_fallback` checks that
  simulated `znver3` plus AVX-512 capability bits still selects
  `ISA_AVX2_DUAL`.
- `make zen3-fallback` runs `scripts/zen3_fallback_check.py`; it writes
  `docs/evidence/zen3_fallback.json` and `docs/evidence/zen3_fallback.log`
  with the focused reproduction and machine-checkable verdict.

## Backend responsibilities

- allocate/free tagged buffers (`weights`, `expert_cache`, `kv`, `workspace`,
  `staging`);
- upload host bytes and expose fence completion;
- execute F32 matrix-vector products;
- record every dispatched operation with its actual backend and byte counts;
- inject transfer, fence and device-loss faults for tests;
- report device memory and ISA details.

Transformer control operations (RMSNorm, RoPE, attention reduction, softmax,
SwiGLU, residuals, sampling) currently execute in scalar C++ and are recorded
as `scalar-control`. This composition is explicit: the VNPU backend accelerates
the major matvec/expert work, but no result is labelled “GPU inference”.

## No fake acceleration

The `gpu` backend fails with a fixed exit code when no fallback is enabled.
With `--allow-cpu-fallback`, every fallen-back operation prints a `[fallback]`
stderr line and is recorded as `cpu-fallback` in the operation log. There is no
silent CPU substitution for missing GPU kernels.

## Device query

```sh
./build/freemoe device-info --backend cpu  --json
./build/freemoe device-info --backend vnpu --json
./build/freemoe device-info --backend gpu  --json
```

On the recorded sandbox the default VNPU query reports
`AVX2+FMA dual-pipe(forced)`, 256-bit SIMD, VNPU ABI 1, host RAM and
`dedicated_vram_bytes=0`. `zen3-auto` on that non-Zen-3 sandbox reports
`Zen 3 not detected` and may select `AVX512F+VNNI`; with
`--test-simulate-zen3` it reports the dual 256-bit path and the Zen 3 fallback
note. Those values describe the validation sandbox, not the user's Zen 3
deployment machine.
