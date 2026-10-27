# M3 exit: one MoE family, all selected experts executed

| ID | Status | Artefact |
|---|---|---|
| P3.1 | PASS | `docs/SUPPORTED_MODELS.md` field/citation table; D5 oracle `tools/reference_oracle.py` |
| P3.2 | PASS | `src/gguf.cpp` MoE adapter; router/gate/up/down stacked-tensor fixtures; missing/unknown-tensor tests |
| P3.3 | PASS | `src/gguf.cpp` expert extent validation; `docs/FORMAT_AND_QUANTISATION.md`; Q8_0 block test |
| P3.4 | PASS | `src/engine.cpp` `topk()/softmax()`; `docs/evidence/final_moe_cpu.json` 20/20 routing matches |
| P3.5 | PASS | `src/engine.cpp` expert SwiGLU/down/weighted sum; 20/20 logits within D4 tolerance |
| P3.6 | PASS | resident reader preload and `docs/evidence/final_moe_cpu.json`; `resident_expert_bytes` equals stacked tensor bytes in unit tests |
| P3.7 | PASS | `ResidencyManager::note_selected()`, `with_expert()`, `assert_selected_executed()`; selected/executed equality in all final reports |
| P3.8 | PASS | `docs/evidence/final_moe_cpu.json`, `docs/evidence/final_moe_vnpu.json`, `docs/evidence/final_moe_direct.json` |
| P3.9 | PASS | this file |

M3 exit: the `freemoe-tiny-v1` MoE model completes all 20 prompts on CPU and
patched Virtual NPU. Resident, buffered, mmap and direct readers produce
bit-identical outputs on the same kernel path. No selected expert is skipped.
