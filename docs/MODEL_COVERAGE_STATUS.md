# Model coverage status: the requested list, one row per model

This page exists because "how many models does this run?" has two different
answers depending on whether you mean *executed here* or *reviewed against
upstream*. Every row says which one it is.

Three labels are used, and nothing else:

| Label | Means |
| --- | --- |
| **WORKS** | A fixture shaped like that family's checkpoint is executed in `architecture_parity` and matches an independent oracle. Not the real weights — no vendor checkpoint is in this workspace — but the block, tensor mapping, rotary and routing are exercised. |
| **PARTIAL** | The block is implemented except for the primitives named in the row. Those primitives are *missing*, so the model is refused rather than approximated. |
| **BLOCKED** | Needs a change to the block itself (a new attention family, a new tensor role, a new quantisation decoder). Nothing about it is guessed. |

Named models are grouped by the *primitive* that unblocks them, because that is
what the work is: the 50 model names reduce to about eight capabilities.

## Implemented and executed

| Capability | Evidence |
| --- | --- |
| Byte-level BPE (4 pre-tokenizer families), SentencePiece by score | `tools/tokenizer_parity.py`: 8 vocabularies at 46/46, 0 failures |
| Linear + YaRN context extension, `rope_freqs` divisors | `rope_scaling` checks, 4 fixtures |
| Attention sinks, NoPE layers | `attention_sinks`, `rope_scaling`, fixtures |
| MLA (DeepSeek-V2/V3 style), sigmoid/softmax/sqrt-softplus gating, group-limited routing, selection bias, shared expert | `deepseek2-mla` fixture + oracle |
| GQA/MQA, per-head qk-norm, partial rotary, fused qkv, fused gate-up, post-norms, sliding window, tanh softcap, LayerNorm variants, silu/gelu | 23 architecture fixtures |
| 36 quantisation types incl. ternary `TQ1_0`/`TQ2_0`/`Q1_0`/`Q2_0`/`PQ2_0`, the whole IQ family (IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS/NL) and FP4 (MXFP4, NVFP4) | `quant_reference_decode`, `quant_model_parity`; bit-exact against llama.cpp's own dequantizers via `tools/quant_upstream_check.py` (26 types, 0 failures) |
| Fused `ffn_gate_up_exps`, clamped SwiGLU (`swiglu-oai`, per-layer `swiglu_clamp`), router renormalisation floor | `ffn_variants`, `qwen3moe-fused-up` and `qwen3moe-clamp` fixtures |
| mRoPE section split (`rope.dimension_sections`), plain and interleaved | `rope_mrope`; `glm4-mrope` fixture |
| Adapter rows tagged in llama.cpp's vocabulary (`inspect --tags`, `freemoe adapters`) | `adapter_tags`, 85 checks |

Supported architecture strings today (each has a fixture unless noted):
`llama`, `qwen2`, `qwen3`, `qwen2moe`, `qwen3moe`, `gemma`, `gemma2`, `gemma3`,
`phi2`, `phi3`, `glm4`, `minicpm`, `deepseek2`, `internlm2`, `exaone`, `orion`,
`gptneox`, `stablelm`, `smollm`, `smollm2`, `smollm3`, `tinyllama`, plus the
native `freemoe-tiny`. `smollm`/`smollm2`/`tinyllama` are the same llama block.

## The requested list

### Runnable today

| Requested model | GGUF arch | State |
| --- | --- | --- |
| Qwen3 (dense + MoE families) | `qwen3`, `qwen3moe` | **WORKS** |
| Qwen2.5 / Qwen2 family | `qwen2`, `qwen2moe` | **WORKS** |
| LLaMA-3 / 3.1 / 3.2 text, LLaMA-3-Nemotron | `llama` | **WORKS** (tokenizer `llama-bpe` matches 46/46) |
| DeepSeek-V3 / V3.1 text path | `deepseek2` | **WORKS** for the dense+MoE text block; the V3 *vision* variant is out of scope |
| Gemma 3 text | `gemma3` | **WORKS** |
| GLM-4 text | `glm4` | **WORKS** |
| SmolLM3 | `smollm3` | **WORKS** (NoPE layers implemented in this change) |
| DeepSeek-R1 distills on Qwen/Llama | `qwen2`, `llama` | **WORKS** |

### One primitive away

| Requested model | GGUF arch | Blocked by |
| --- | --- | --- |
| Qwen3-VL (text path), Qwen3.5-MoE, Qwen3.5 dense, Ernie 4.5 / 4.5-MoE, Ling/Ring (Bailing-MoE3), GLM-4 / 4.5 MoE, Hunyuan MoE/dense | `qwen3vl`, `qwen35`, `qwen35moe`, `ernie4_5*`, `bailingmoe*`, `glm4moe`, `hunyuan-*` | **mRoPE implemented** (`rope_mrope`, `glm4-mrope` fixture): sections are parsed, validated against the head width and drive per-axis positions, plain and interleaved. Remaining work is the adapter row per architecture plus their block variants; the vision path stays out of scope |
| DeepSeek-V4, V4.1, V4-Flash-Vision (text), Mimo-2.5, MiniMax-M2/M3 | `deepseek4`, `deepseek41`, `mimo*`, `minimax-m2`, `minimax-m3` | **clamped SwiGLU** (`SWIGLU_OAI_MOE`) plus sinks and a `1e-20` router-sum epsilon. Sinks landed in this change; the clamp and the epsilon have not |
| dots.llm1 / dots3note, Cohere2-MoE (North Mini Code), Gemma 4, GLM-DSA, DeepSeek-3.2 | `dots3note`, `cohere2moe`, `gemma4`, `glm-dsa`, `deepseek32` | **fused `ffn_gate_up_exps`**: gate and up stacked in one expert tensor. The expert record layout assumes three separate projections, so this is a residency/extent change, not a rename |
| Qwen3-Next, Qwen3.5-Next, Kimi-Linear | `qwen3next`, `qwen35*`, `kimi-linear` | **gated delta-net** (linear attention with a per-head state). New cache shape, new kernel; the largest single item left |
| Nemotron-H, Jamba-style hybrids, Falcon-H1, Granite-Hybrid, Plamo2 | `nemotron_h`, `nemotron_h_moe`, `falcon-h1`, `granite-hybrid`, `plamo2` | **recurrent/SSM layers** interleaved with attention (+8 models) |
| LFM2.5, LFM2-MoE | `lfm2`, `lfm2moe`, `lfm3` | **short convolution** layers before attention |
| BitNet-b1.58-2B-4T | `bitnet` | **`I2_S` decoder** and the `SUBLN` placement. The ternary family is decoded for PrismML's types; BitNet's own does not overlap |
| Command-A, Command-R+ | `command-r`, `cohere2` | **adapter row + verification** (their graph is close to what exists; not yet checked line by line) |
| Mimo-V2-Flash, MiniMax-01 | `mimo*`, `minimax-01` | **adapter row + verification** (MoE with sinks and its own router normalisation) |
| Mellum2, Laguna XS.2 / M.1, OpenPangu, Seed-OSS, Nanbeige 4.2, MiniCPM5, GigaChat3, Step-3.5-Flash, Mistral 4, K2-Horizon, Muse-Glimmer, DSpark, LLaMA-4 | `mellum`, `laguna`, `pangu*`, `seed_oss`, `nanbeige`, `minicpm3`/`minicpm5`, `gigachat`, `step35`, `mistral4`, `kimi-k2`?, `muse*`, `dspark`, `llama4` | **not yet assessed line by line in this build.** Each needs its upstream `src/models/*.cpp` read and its tensor set compared with the existing flags before a row is added; several (LLaMA-4, Mistral 4) also need fused `gate_up_exps` or the sinks+clamp pair above |

### Permanently out of scope (as specified)

Vision towers, audio (`wavtokenizer`, `pockettts`), OCR, embeddings-only
encoders and diffusion heads are not causal text decoders. Rows marked "text
path" above mean exactly that: the language model runs, the tower does not.

## What this page is not

* No vendor checkpoint has been executed. The fixtures are synthetic, shaped
  like each family, and the goldens come from `tools/llama_oracle.py`, an
  independent implementation. That catches implementation bugs, not a misread
  specification.
* The per-model rows above are static readings of upstream's model files
  (`src/models/*.cpp`) crossed with what this build implements. Where a row says
  "not yet assessed", it means precisely that — no claim is being made.

## Order the remaining work, and why

1. ~~**mRoPE**~~ **done** — sections parsed, validated, and applied per axis,
   plain and interleaved (`rope_mrope`, `glm4-mrope`). The six architectures now
   need their adapter rows, not rotary work.
2. ~~**Fused `ffn_gate_up_exps`**~~ **done** (`qwen3moe-fused-up`).
3. ~~**Clamped SwiGLU + router epsilon**~~ **done** (`qwen3moe-clamp`).
   `SITU` and `GEGLU_ERF` are what remains of this group.
4. ~~**IQ family + MXFP4/NVFP4 decoders**~~ **done**, bit-exact against upstream.
   `I2_S` is *not* a llama.cpp type: it is bitnet.cpp's private type 36, and the
   ternary checkpoints llama.cpp reads are `TQ1_0`/`TQ2_0`, which decode here.
   A file carrying type 36 is refused by name with that explanation.
5. **Short convolution** (+4: `lfm2`, `lfm2moe`, `lfm3`, `lfm-base`) and
   **recurrent/SSM** (+17) — the second needs a new cache shape and scan
   kernels, so it is the first item that is not a variant of the existing block.
6. **Gated delta-net** (+9) — same size class, and it shares the state work with
   the recurrent group.
