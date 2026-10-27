# FreeMoe local patches to Virtual NPU

Pinned upstream: `4298c454aa56b90dac240e81f80ed4467221c1dd`.

1. **Zen 3 / dual AVX2 FMA path.** Added `ISA_AVX2_DUAL`, two independent
   256-bit accumulator chains for both NN and transposed-B GEMM, Zen 3 CPU
   detection, and `VNPU_FEATURE_AVX2_DUAL_FMA`. Transformer AVX2 helpers now
   accept this ISA. No AVX-512 instruction is required. Detection uses the
   testable `select_detected_isa()` policy: `znver3` overrides any later
   AVX-512 capability block, so a Zen 3 CPU selects `ISA_AVX2_DUAL` and never
   advertises `VNPU_FEATURE_AVX512F` on that path. AVX2/FMA remains advertised
   as an available feature for safe explicit forcing.
2. **Safe ISA forcing.** `vnpu_config_t.reserved0` is now `forced_isa`
   (same ABI layout). Config takes precedence over `VNPU_FORCE_ISA`; explicit
   AVX2/AVX-512 requests are rejected with `VNPU_ERR_UNSUPPORTED` when the host
   lacks the required feature. Added `avx2_dual`/`zen3` selectors and invalid
   value checks.
3. **GCC 12+ warning fix.** Suppressed the known GCC false-positive
   `-Wmaybe-uninitialized` originating inside AVX-512 intrinsic headers so the
   vendored source builds under `-Wall -Wextra -Werror`.
4. **Regression tests.** `tools/selftest.cpp` compares the dual-pipe NN/NT
   kernels against the scalar reference, verifies forced-ISA validation, and
   simulates Zen 3 detection with contradictory AVX-512 feature bits through
   `zen3_detection_fallback`. The FreeMoe unit suite additionally simulates the
   full backend fallback decision on non-Zen-3 hosts.

Validate with:

```sh
VNPU_FORCE_ISA=avx2 make -j2 all
VNPU_FORCE_ISA=avx2 make test
```

On a Zen 3 host, omit `VNPU_FORCE_ISA` to exercise automatic detection. On any
host, `./build/vnpu-selftest zen3_detection_fallback` verifies the simulated
detection precedence without needing Zen 3 hardware.
