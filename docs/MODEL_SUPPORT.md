# Model support matrix

This page answers one question honestly: **which of the requested checkpoints can
this build actually run, and if not, what exactly is missing?**

It is generated from two things that cannot drift apart:

- `freemoe probe --model FILE` — a per-file report of architecture,
  hyperparameters, tensor types, tokenizer and every concrete blocker;
- the fixtures in `tests/fixtures/llama-shape-*.gguf`, which are shaped like
  published llama.cpp checkpoints so the probe itself is tested
  (`model_probe` unit test).

Model weights are deliberately **not** stored in this repository: the fixtures
that prove the loader are synthetic (a few hundred KiB), and the real
checkpoints stay wherever you keep them.

Run it yourself on the real files:

```bash
./build/freemoe probe --model /path/to/Qwen2.5-1.5B-MoE-A0.9B-Q4_K_M.gguf
./build/freemoe probe /path/to/models/*.gguf --out support-matrix.json
```

Never trust the table below over the tool: the table records what we expect from
each family, the tool records what a specific file contains.

## What works today

| Capability | State |
| --- | --- |
| GGUF v3 container parse, any architecture | WORKS |
| Generic inspect / tensor inventory / type census | WORKS |
| Block decoders: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K | WORKS |
| Quantised GEMV (fused AVX2 + scalar), typed errors for unknown types | WORKS |
| `freemoe-tiny-v1` adapter: MHA, RoPE (half), SwiGLU, top-2 MoE, byte tokenizer | WORKS |
| Standard llama-family adapter (llama.cpp keys and tensor names) | **MISSING** |
| Grouped-query attention (`head_count_kv < head_count`) | **MISSING** |
| BPE / sentencepiece tokenizer (`tokenizer.ggml.model = gpt2` / `llama` / `spm`) | **MISSING** |
| Per-architecture variants: gemma, phi3, glm4, qwen3(moe), minicpm | **MISSING** |
| Ternary block formats (`PTQ1_0`, `TQ1_0`, `TQ2_0`) | **MISSING** |
| Vision towers (MiniCPM-V) | **MISSING**, and out of scope for a text engine |

## Why the gap is where it is

FreeMoe was built from its own specification, so its adapter reads
`freemoe.*` metadata, `freemoe-tiny` architecture and a raw-byte tokenizer.
Every checkpoint in the list below instead declares a standard architecture
(`llama`, `qwen2`, `gemma3`, `phi3`, …) with architecture-prefixed keys
(`llama.block_count`), llama.cpp tensor names and a real tokenizer.

The quantisation layer those files need — the part that used to be the hard
blocker — is now done and verified. What remains is the adapter and tokenizer
layer, which is enumerated below as ordered work.

## Requested models

Family inference is from the file naming; the probe gives the ground truth per
file. "Format" is the storage format the probe will confirm.

| # | File | Expected architecture | Format | Blockers |
| --- | --- | --- | --- | --- |
| 1 | Firefly-v5.Q4_K_M | llama-derived | Q4_K | adapter, tokenizer |
| 2 | gemma-4-E4B-it-Q4_K_M | gemma3/gemma4 | Q4_K | gemma semantics, tokenizer |
| 3 | GLM-4.7-Flash-Q4_K_M | glm4 | Q4_K | glm4 semantics, tokenizer, MoE |
| 4 | L3.2-8X3B-MOE-…-Q8_0 | llama + experts | Q8_0 | adapter, MoE layout, tokenizer |
| 5 | Llama-3.2-1B-4B-Quad-MoE.Q4_K_M | llama + experts | Q4_K | adapter, MoE layout, tokenizer |
| 6 | Llama-3.2-1B-Instruct-Q8_0 | llama | Q8_0 | adapter, GQA, tokenizer |
| 7 | Mellum2.1-12B-A2.5B-Thinking-Q4_K_M | llama/qwen2 + MoE | Q4_K | adapter, MoE layout, tokenizer |
| 8 | MiniCPM-V-4_6-Q8_0 | minicpm-v (multimodal) | Q8_0 | adapter, vision tower, tokenizer |
| 9 | MiniCPM5-1B-Q8_0 | minicpm/llama | Q8_0 | adapter, tokenizer |
| 10 | MiniCPM5-1B.i1-Q6_K | minicpm/llama | Q6_K | adapter, tokenizer, Q6_K AVX2 kernel |
| 11 | Phi-3-mini-4k-instruct-q4 | phi3 | Q4_K | phi3 semantics (fused qkv, partial rope), tokenizer |
| 12 | Phi-4-reasoning-plus-Q3_K_M | phi3 | Q3_K | phi3 semantics, tokenizer, Q3_K AVX2 kernel |
| 13 | qwen2.5-0.5b-instruct-q2_k | qwen2 | Q2_K | adapter, GQA, qkv bias, tokenizer |
| 14 | Qwen2.5-1.5B-MoE-A0.9B-Q4_K_M | qwen2 + MoE | Q4_K | adapter, MoE layout, tokenizer |
| 15 | Qwen3.5-35B-A3B-UD-Q4_K_XL | qwen3moe | Q4_K + Q6_K | qwen3 semantics, MoE, tokenizer |
| 16 | Qwen3.5-9B-Q4_K_M | qwen3 | Q4_K | qwen3 semantics, GQA, tokenizer |
| 17 | Qwen3.8-4B-Q4_K_M | qwen3 | Q4_K | qwen3 semantics, tokenizer |
| 18 | qwen3.8-flash-next-reap-288-Q4_K_M | qwen3 | Q4_K | qwen3 semantics, tokenizer |
| 19 | smollm-135m-instruct-add-basics-q8_0 | llama | Q8_0 | adapter, tokenizer |
| 20 | Ternary-Bonsai-2-27B-…-PTQ1_0-MTP-Q4_0 | ternary hybrid | PTQ1_0 + Q4_0 | ternary decoder, adapter, tokenizer |
| 21 | Ternary-Bonsai-2-2B-PTQ1_0 | ternary | PTQ1_0 | ternary decoder, adapter, tokenizer |
| 22 | tiny-llama3-test-Q3_K_M | llama | Q3_K | adapter, tokenizer |
| 23 | Tiny-LLM.Q4_K_M | llama-derived | Q4_K | adapter, tokenizer |
| 24 | Tiny-Moe.Q4_K_M | llama + experts | Q4_K | adapter, MoE layout |
| 25 | tinyllama-1.1b-chat-v1.0-q2_k | llama | Q2_K | adapter, tokenizer |
| 26 | tinyllama-15M-stories-Q2_K | llama | Q2_K | adapter, tokenizer |
| 27 | tinyllama-15M-stories-Q3_K_M | llama | Q3_K | adapter, tokenizer |
| 28 | TinyLLama-v0.Q8_0 | llama | Q8_0 | adapter, tokenizer |
| 29 | tinystories-gpt-0.1-3m.Q2_K | llama/gpt2 | Q2_K | adapter, tokenizer |
| 30 | Uraion-Agent-Steer-Q4_K_M | llama-derived | Q4_K | adapter, tokenizer |

**Runnable today: 0 of 30.** Every one needs the standard adapter and a real
tokenizer. Sixteen also need MoE or per-architecture semantics, and two need a
ternary decoder.

Being precise about that is the point of this page: the previous state of the
project could not even read these files' weights, and claimed nothing about it.

## Ordered work to close the gap

Each step is independently testable, and each is gated by a fixture pair the
way the quantisation work was (a synthetic checkpoint plus an independent
reference).

1. **Standard metadata + tensor-name mapping (`llama` family).** Read
   `<arch>.*` keys, map llama.cpp tensor names (`blk.N.attn_q`, `ffn_gate`,
   `ffn_up`, `ffn_down`, `token_embd`, `output`, `*_norm`) onto the existing
   role enum. Covers #6, #19, #22, #23, #25, #26, #27, #28, #29, #30 and the
   text path of #1.
2. **BPE tokenizer** (`tokenizer.ggml.model = gpt2|llama|spm`): merges, scores,
   byte fallback, add-space prefix. Needed by all 30.
3. **Grouped-query attention** in the attention kernel: `head_count_kv <
   head_count`, with the corresponding KV cache layout. Needed by #6, #13, #16
   and most modern models.
4. **Standard MoE FFN**: `ffn_gate_inp`, `ffn_gate_exps`, `ffn_up_exps`,
   `ffn_down_exps` (and the newer per-expert splat naming), top-k routing with
   optional shared expert. Needed by #4, #5, #7, #14, #15, #24.
5. **Q6_K / Q3_K / Q2_K / Q5_K AVX2 kernels** — correctness is already there via
   the scalar path; this is the throughput step for #10, #12, #15, #22, #25-27.
6. **Architecture variants**: `gemma` (embedding scale, `1 + w` norms, GeGLU),
   `phi3` (fused qkv, partial rotary), `qwen2` (`qkv_bias`), `qwen3`
   (`attn_q_norm`/`attn_k_norm`), `glm4`, `minicpm`.
7. **Ternary formats** (`PTQ1_0`/`TQ1_0`/`TQ2_0`) for #20 and #21.
8. **Multimodal** (#8) is out of scope for this engine: the text decoder can be
   supported, the vision tower cannot, and pretending otherwise would be the
   exact kind of fake capability this project refuses.

## Acceptance rule

A model counts as supported only when all of these hold:

1. `freemoe probe` reports `"runnable": true` for the actual file;
2. greedy generation from a fixed prompt is byte-identical between the new
   adapter and an independent reference (llama.cpp, or a committed golden file
   produced by it);
3. `make test` stays green, including the quantised parity fixtures;
4. the model's tensor types all report `execute: true` in the probe.

Anything less is "loads", not "supported", and this page will say so.
