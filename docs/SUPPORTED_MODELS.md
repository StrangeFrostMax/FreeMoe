# Supported models

## Supported family

FreeMoe currently supports exactly one adapter:

| Field | Value | Citation |
|---|---|---|
| Architecture string | `freemoe-tiny` | GGUF key `general.architecture`; `build_manifest()` in `src/gguf.cpp`. |
| Adapter ID/version | `freemoe-tiny-v1` / 1 | GGUF key `freemoe.adapter`; `docs/ARCHITECTURE.md`. |
| Source format | GGUF v3, little-endian, F32 tensor payloads | `src/gguf.cpp`, `tests/fixtures/*.gguf`. |
| Model kinds | `dense` and `moe` | GGUF key `freemoe.model_kind`. |
| Hidden / intermediate / layers | 16 / 32 / 2 | `freemoe.hidden_size`, `freemoe.intermediate_size`, `freemoe.num_hidden_layers`. |
| Attention | 4 heads, head dimension 4, MHA only | `freemoe.num_attention_heads`, `freemoe.num_key_value_heads`, `freemoe.head_dim`. |
| Context / vocabulary | 1024 / 258 | `freemoe.max_context_length`, `freemoe.vocab_size`. |
| Tokenizer | raw byte tokenizer plus BOS; no chat template | `tokenizer.ggml.model`, `tokenizer.ggml.tokens`, `tokenizer.chat_template`; `Engine::tokenize()`. |
| Normalisation | RMSNorm with metadata epsilon | `freemoe.rms_norm_epsilon`; `tools/reference_oracle.py` `rmsnorm()`. |
| Positional encoding | half-rotation RoPE, theta from metadata | `freemoe.rope_mode`, `freemoe.rope_theta`; `rope_half()` in both runtimes. |
| FFN | SwiGLU gate/up/down | Tensor names `blk.L.ffn_gate.weight`, `ffn_up.weight`, `ffn_down.weight`; oracle `Session.step()`. |
| Router operation | linear scores, stable top-k (ties: lower expert ID), softmax only over selected scores | `freemoe.router_operation`; `router_topk()` in oracle and `Engine::topk()/softmax()`. |
| Selected weight normalisation | softmax over the `top_k` selected router scores | `topk` then `softmax(selected_scores)`. |
| Expert count / top-k | 8 / 2 | `freemoe.expert_count`, `freemoe.expert_top_k`; stacked tensor `ne[2]`. |
| Shared expert | none | `freemoe.shared_expert=false`; no shared-expert tensors are accepted. |
| Expert tensor layout | three stacked F32 tensors per layer: gate, up, down | `freemoe.expert_layout=stacked-f32`; `blk.L.ffn_gate_exps.weight`, `ffn_up_exps.weight`, `ffn_down_exps.weight`. |
| Expert slicing | one contiguous `hidden*intermediate*4` byte slice per expert per projection | `build_manifest()` expert extent validation; `freemoe.expert_checksums`. |
| Quantisation | F32 inference only | `tensor.type_id==0`; quant table is for safe parsing/size checks. |

The reference source for every arithmetic field is
`tools/reference_oracle.py` at D5 tree `deee8aa67b9615a0ef8cd73aaa4c8aca6469cfa5`.
The C++ engine and Python oracle are separate implementations of the same
documented operations.

## Declared checkpoints

| Role | Path | SHA-256 | Licence |
|---|---|---|---|
| Dense D2 | `tests/fixtures/dense-fm.gguf` | `4d81aca51f3d087c977fc4b40d3227401d225b45373878b618a72059d35885a1` | MIT |
| MoE D3 | `tests/fixtures/moe-fm.gguf` | `8f1703d7863588347e318000526938a8b1313bfaf2eb0bee1bc7c1aaa985ebc8` | MIT |

These are deterministic synthetic conformance fixtures, not pretrained models.
They contain no third-party weights and are safe to redistribute under MIT.

## Explicitly unsupported

- arbitrary Llama, Mixtral, DeepSeek, Qwen, Gemma or other GGUF architectures;
- F16, BF16, Q4/Q5/Q8, K-quant or imatrix quantised inference;
- grouped-query attention, sliding-window attention, ALiBi, YaRN, multimodal
  towers, LoRA, speculative decoding, continuous batching or multi-GPU;
- per-expert GGUF tensor names, fused gate-up experts, shared experts or
  architecture-specific router scaling not documented above;
- chat templates other than raw prompt mode;
- any model whose tensor semantics cannot be mapped without guessing.

Attempting to run such a model produces a typed error naming the unsupported
metadata, tensor or quantisation type. The runtime never guesses an expert
index from arbitrary digits in a tensor name.
