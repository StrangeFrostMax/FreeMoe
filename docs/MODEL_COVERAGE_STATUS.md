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
| GQA/MQA, per-head qk-norm, partial rotary, fused qkv, fused gate-up, post-norms, sliding window (model-wide and per layer), tanh softcap, LayerNorm variants, silu/gelu/relu²/gelu_erf/swiglu_oai, gate-free FFN | 36 architecture fixtures (`architecture_parity`, 266 checks) |
| 36 quantisation types incl. ternary `TQ1_0`/`TQ2_0`/`Q1_0`/`Q2_0`/`PQ2_0`, the whole IQ family (IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_XS/NL) and FP4 (MXFP4, NVFP4) | `quant_reference_decode`, `quant_model_parity` (Q8_0/Q4_0/Q4_K/Q6_K end to end); **matches llama.cpp's own dequantizers to ≤1e-6 relative** of block magnitude via `tools/quant_upstream_check.py` (26 types, 0 failures; worst absolute 5.0e-06 on IQ4_NL, MXFP4/NVFP4 exact). Not claimed bit-exact — see `docs/GAP_ANALYSIS.md` |
| Fused `ffn_gate_up_exps`, clamped SwiGLU (`swiglu-oai`, per-layer `swiglu_clamp`), router renormalisation floor | `ffn_variants`, `activation_clamp_forms` (13 checks), `qwen3moe-fused-up` and `qwen3moe-clamp` fixtures |
| Mixed-precision expert stacks: per-projection types inside one expert record (Q4_K gate/up + Q6_K down), fused and separate | `quant_expert_mixed_types` (32 checks), `quant-expert-{fused,separate}.gguf` + F32 twins |
| Pointwise activation primitives (`relu`, `relu_sq`, `gelu_erf`, `softplus`) as upstream's `llm_ffn_op_type` defines them | `pointwise_activations` (12 checks) pins the arithmetic. Three of the four are now **reachable end to end**: `relu_sq` through `arcee`/`nemotron`/`jais2`/`plm` (`gate_free_families`, 33 checks, and their parity fixtures — with silu substituted `arcee` fails the golden by 0.98 absolute), and `gelu_erf` through `mistral3`'s adaptive norm MLP, whose activation upstream writes as `LLM_FFN_GELU_ERF` with `LLM_FFN_SEQ` (`mistral3_block`). `relu` and `softplus` stay primitive-only: no architecture in llama.cpp's `src/models/` selects either, so a fixture that applied one would be inventing a family |
| Refusal paths: unmapped architecture, undecodable tensor type, missing output head | `refusal_paths` (17 checks) — error code, message and probe blocker JSON |
| mRoPE section split (`rope.dimension_sections`), plain and interleaved | `rope_mrope`; `glm4-mrope`, `qwen3vl` and `qwen3vlmoe` fixtures. The section split is a **required key** for the mRoPE families, as it is upstream (`mrope_families`), and the interleaved layout is selected by architecture the way `llama_rope_type()` does |
| Recurrent (state-space) layers: mamba and mamba2, including the dt/B/C norms | `mamba`, `mamba-dtrms` and `mamba2` fixtures + oracle goldens, one parity row each, and `recurrent_families` (45 checks). The state replaces the kv cache and the budget reserves it as such |
| Attention sinks, biased routers, per-expert projection biases, post-attention norm as the FFN norm, alternating full/windowed layers | `gpt-oss` fixture + `gpt_oss_block`; the same flags are what `minimax-m3` needs |
| Adapter rows tagged in llama.cpp's vocabulary (`inspect --tags`, `freemoe adapters`) | `adapter_tags`, 85 checks |

Supported architecture strings today (each has a fixture unless noted):
`llama`, `qwen2`, `qwen3`, `qwen2moe`, `qwen3moe`, `gemma`, `gemma2`, `gemma3`,
`phi2`, `phi3`, `glm4`, `minicpm`, `deepseek2`, `internlm2`, `exaone`, `orion`,
`gptneox`, `stablelm`, `smollm`, `smollm2`, `smollm3`, `tinyllama`, `bitnet`,
`arcee`, `nemotron`, `jais2`, `gpt-oss`, `plm`, `mistral3`, `qwen3vl`,
`qwen3vlmoe`, `mamba`, `mamba2`,
plus the native `freemoe-tiny`. `smollm`/`smollm2`/`tinyllama` are the same llama
block. The three lists that decide this — the adapter table, the loader's family
list and the probe's known-architecture table — are bound to each other by the
`adapter_binding` test (323 checks), which is what caught `smollm3` and `bitnet`
missing from the probe's table while their adapters existed.

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
| BitNet-b1.58 block, any decoded tensor type | `bitnet` | **WORKS** for the block (`bitnet` fixture); the published `I2_S` files stay refused by type (see the row below) |
| Arcee AFM, Nemotron (dense), JAIS-2 (text), PLM | `arcee`, `nemotron`, `jais2`, `plm` | **WORKS** for the block: gate-free `relu²` FFN (the activation GGUF cannot state comes from the architecture table), LayerNorm-with-bias variants, tied head. `plm` adds MLA with a one-step q projection |
| gpt-oss (20B / 120B text) | `gpt-oss` | **WORKS** for the text block: attention sinks, the post-attention norm as the MoE's input norm, a biased router, per-expert projection biases, `swiglu_oai` (alpha 1.702, limit 7.0), `softmax-weight` gating and alternating full/windowed layers. The MXFP4 *container* is supported separately by the quantisation layer |
| Mistral 3 (text), Voxtral realtime text path | `mistral3` | **WORKS** for the block: biases on all three FFN projections, the adaptive RMSNorm over the sinusoidal delay embedding (its MLP is a `gelu_erf`), and llama.cpp's per-position attention temperature |
| Qwen3-VL (text path), Qwen3-VL-MoE (text path) | `qwen3vl`, `qwen3vlmoe` | **WORKS** for the text block, mRoPE sections included. The vision tower stays out of scope |
| Mamba, Mamba-2, Falcon-Mamba-style blocks | `mamba`, `mamba2` | **WORKS**: convolution + state-space scan, no attention, and a pure file has no FFN. The state replaces the kv cache |

### One primitive away

| Requested model | GGUF arch | Blocked by |
| --- | --- | --- |
| Qwen3.5-MoE, Qwen3.5 dense, Ernie 4.5 / 4.5-MoE, Ling/Ring (Bailing-MoE3), GLM-4 / 4.5 MoE, Hunyuan MoE/dense | `qwen35`, `qwen35moe`, `ernie4_5*`, `bailingmoe*`, `glm4moe`, `hunyuan-*` | **Adapter row missing, not the block — except qwen35.** mRoPE is implemented and now proven by `qwen3vl`/`qwen3vlmoe` parity rows. `qwen35`/`qwen35moe` remain blocked by their **gated delta-net layers**, not by the rotary: their recurrent layers are a different scan from the mamba one this build now has. The rest need only an adapter row per architecture; the vision path stays out of scope |
| DeepSeek-V4 / V4.1 (text), Mimo-2.5, MiniMax-M2/M3 | `deepseek4`, `deepseek41`, `mimo*`, `minimax-m2`, `minimax-m3` | **Partly closed.** `swiglu_oai` is no longer primitive-only: `gpt-oss` runs it end to end with a golden, and the flags `minimax-m3` adds (sinks, a biased router, per-expert projection biases, q/k per-head norms) are the same ones (`gpt_oss_block`). `deepseek4`/`deepseek41` carry their own per-layer clamp bounds *and* a much larger set of unmodelled tensors — the DSA indexer, hyper-connections, the MTP head — so a row for them would be a claim this build cannot back; the clamp arithmetic itself is covered by `activation_clamp_forms` |
| dots.llm1 / dots3note, Cohere2-MoE (North Mini Code), Gemma 4, GLM-DSA, DeepSeek-3.2 | `dots3note`, `cohere2moe`, `gemma4`, `glm-dsa`, `deepseek32` | **Fused `ffn_gate_up_exps` is implemented** (`qwen3moe-fused-up`, and a mixed-type fused expert stack in `quant_expert_mixed_types`): one tensor, gate block first, two extents per expert record, each carrying its own type. What is missing is the **adapter row** for each of these strings |
| Qwen3-Next, Qwen3.5-Next, Kimi-Linear | `qwen3next`, `qwen35*`, `kimi-linear` | **gated delta-net** (linear attention with a per-head state). New cache shape, new kernel; the largest single item left |
| Nemotron-H, Jamba-style hybrids, Falcon-H1, Granite-Hybrid, Plamo2 | `nemotron_h`, `nemotron_h_moe`, `falcon-h1`, `granite-hybrid`, `plamo2` | **The recurrent block itself now runs** (`mamba`, `mamba2`, `mamba-dtrms` fixtures). What these need on top is the *hybrid* layout — recurrent layers interleaved with attention ones, whose attention layers need a cache beside the state. That mix is refused explicitly today (`recurrent_families`) rather than half-run |
| LFM2.5, LFM2-MoE | `lfm2`, `lfm2moe`, `lfm3` | **short convolution** layers before attention |
| BitNet-b1.58-2B-4T | `bitnet` | **Block done** (`bitnet` fixture + oracle: SUBLN sub-norms on the attention and FFN outputs, optional one-element `*.scale` per projection, tied output). What remains is the **container**: Microsoft's published files store weights as `I2_S`, bitnet.cpp's private type 36, whose element layout cannot be confirmed from a published source — a file carrying it is refused by name. A BitNet-shaped file in a decoded ternary type runs |
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
4. ~~**IQ family + MXFP4/NVFP4 decoders**~~ **done**, matching upstream's own
   dequantizers to ≤1e-6 relative (26 types, worst absolute 5.0e-06).
   `I2_S` is *not* a llama.cpp type: it is bitnet.cpp's private type 36, and the
   ternary checkpoints llama.cpp reads are `TQ1_0`/`TQ2_0`, which decode here.
   A file carrying type 36 is refused by name with that explanation.
5. **Short convolution** (+4: `lfm2`, `lfm2moe`, `lfm3`, `lfm-base`) and
   **recurrent/SSM** (+17) — the second needs a new cache shape and scan
   kernels, so it is the first item that is not a variant of the existing block.
6. **Gated delta-net** (+9) — same size class, and it shares the state work with
   the recurrent group.
