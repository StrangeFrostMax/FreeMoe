# Release checklist

Status source: `docs/evidence/release_status.json` (generated, not hand-edited).

## Correctness prototype

| Criterion | Status | Artefact |
|---|---|---|
| One declared dense model completes inference | PASS | `docs/evidence/final_dense_cpu.json` |
| One declared MoE model completes inference | PASS | `docs/evidence/final_moe_cpu.json` |
| Model mappings and unsupported cases explicit | PASS | `docs/SUPPORTED_MODELS.md`, `docs/FORMAT_AND_QUANTISATION.md` |
| Reference comparisons pass | PASS | `docs/evidence/final_dense_cpu.json`, `docs/evidence/final_moe_cpu.json`, `docs/evidence/final_moe_vnpu.json` |
| No selected expert silently skipped | PASS | selected/executed equality in `docs/evidence/final_moe_*.json` |
| Memory bounded and reported | PASS | `residency.budget`, `backend.allocations.live_allocations=0` in final reports |
| Malformed/fuzz tests pass fail-closed | PASS | `tests/malformed_models/manifest.tsv`, `docs/evidence/test_summary.json`, `docs/evidence/fuzz_72s.json` |
| Reproducible build/test instructions | PASS | `README.md`, `Makefile`, `CMakeLists.txt`, `docs/PHASE0_EXIT.md` |
| Zen 3 detection falls back to dual AVX2/FMA3 with visible diagnostics | PASS | `docs/evidence/zen3_fallback.json`, `docs/evidence/zen3_fallback.log`, `docs/BACKEND_SUPPORT.md` |

## Paged MoE prototype

| Criterion | Status | Artefact |
|---|---|---|
| Selected experts fetched from declared storage tier | PASS | `docs/evidence/final_moe_direct.json` (`reader=direct`, demand reads and checksum checks) |
| Cache eviction and async completion safe | PASS | `docs/evidence/phase0_unit.log`, slot state/generation/fence tests; stale-completion counters |
| Cold-cache inference correct | PASS | `docs/evidence/final_moe_cpu.json` with one-slot cache and prompt set |
| Disk/RAM/VRAM metrics captured | PASS | `io`, `residency`, `process` and `device` sections of final reports; VRAM is zero because no physical GPU exists |
| No fake acceleration or silent fallback | PASS | `docs/BACKEND_SUPPORT.md`, GPU placeholder test, explicit `cpu-fallback` operation log |

## Performance-qualified release

Status: **CLOSED**. The exact target hardware/software profile is unknown and
no controlled cold/warm R-run benchmark on that machine has been approved.
`docs/BENCHMARK_PROTOCOL.md`, `benchmarks/run_repeats.sh` and
`docs/RESULTS_INDEX.md` define the procedure; no external benchmark number is
claimed.
