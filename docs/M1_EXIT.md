# M1 exit: GGUF reader, inspection and dense CPU correctness

| ID | Status | Artefact |
|---|---|---|
| P1.1 | PASS | `src/gguf.cpp` header bounds checks; `docs/evidence/phase0_unit.log` truncation suite |
| P1.2 | PASS | `tests/fixtures/metadata-types.gguf`; `docs/evidence/phase0_unit.log` all-value-type test |
| P1.3 | PASS | `src/gguf.cpp` tensor directory checks; 54 malformed fixtures |
| P1.4 | PASS | `docs/FORMAT_AND_QUANTISATION.md`; `tests/fixtures/quant-blocks.gguf`; `docs/evidence/phase0_unit.log` |
| P1.5 | PASS | `docs/evidence/inspect_dense.json`, `docs/evidence/inspect_moe.json` equal `tests/golden/*.json` |
| P1.6 | PASS | `src/gguf.cpp` strict adapter; `docs/SUPPORTED_MODELS.md`; missing/unknown tensor fixtures |
| P1.7 | PASS | `src/engine.cpp` raw byte tokenizer; `tests/reference/prompt_XX.input.ids`; `docs/evidence/reference_manifest.sha256` |
| P1.8 | PASS | `docs/evidence/final_dense_cpu.json` (20/20 logits within D4 tolerance) |
| P1.9 | PASS | `docs/evidence/final_dense_cpu.json` (20/20 greedy token matches) |
| P1.10 | PASS | `tests/malformed_models/manifest.tsv` and 54 fixtures; `docs/evidence/phase0_unit.log` |
| P1.11 | PASS | `docs/evidence/fuzz_72s.json` (F=0.02h, ASan/UBSan, zero findings) |
| P1.12 | PASS | `src/engine.cpp` `compare_reference()`; `docs/evidence/final_*.json` |
| P1.13 | PASS | this file |

M1 exit: dense Llama-like synthetic D2 model, CPU only, F32, greedy output
matches the independent reference for all 20 prompts.
