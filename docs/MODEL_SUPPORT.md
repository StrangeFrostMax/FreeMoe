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
| Block decoders: F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K, IQ4_NL, I8, I16, I32, I64, F64 | WORKS |
| Ternary decoders: `TQ1_0` (34), `TQ2_0` (35), `Q1_0` (41), `Q2_0` (42), `PQ2_0` (142) | WORKS |
| `PTQ1_0` (143) geometry: parses, sizes and expert ranges exact, decode refused | **PARTIAL** |
| Quantised GEMV (fused AVX2 + scalar), typed errors for unknown types | WORKS |
| `freemoe-tiny-v1` adapter: MHA, RoPE, SwiGLU, top-2 MoE, byte tokenizer | WORKS |
| Standard llama-family adapter (llama.cpp keys and tensor names) | WORKS |
| Grouped-query attention (`head_count_kv < head_count`), KV cache sized per kv head | WORKS |
| SentencePiece and BPE tokenizers, byte fallback, honest coverage report | WORKS |
| Byte-level BPE with real merge resolution and four pre-tokenizer families (`gpt2`, `gpt2-digits`, `llama3`, `qwen2`) | WORKS — matches llama.cpp's own vectors on 6/6 vocabularies |
| Other pre-tokenizer families (DeepSeek, Falcon, ChatGLM, Yi, …) | **MISSING**, refused by name; see "Tokenizer" below |
| `qwen2`: q/k/v projection biases (split and fused) | WORKS |
| `qwen3`: per-head RMSNorm on q and k before RoPE | WORKS |
| `gemma`: embedding scale, zero-centred norm scales, GeGLU, tied output | WORKS |
| `gemma2`/`gemma3`: post-attention/post-FFN norms, tanh softcapping, sliding window | WORKS |
| `phi3`: fused q/k/v projection with fused bias, partial rotary | WORKS |
| `glm4`: fused gate+up FFN, post-attention and post-FFN norms | WORKS |
| `minicpm`: embedding scale (12.0 by default), tied output | WORKS |
| `qwen3moe`: qwen3 q/k norm with stacked-expert FFN and router | WORKS |
| `qwen2moe`: routed experts plus a dense shared expert, gated per token | WORKS |
| `phi2`: LayerNorm with biases, gate-free GELU FFN, one shared residual | WORKS |
| `deepseek2`: MLA (low-rank q, kv latent, per-head `k_b`/`v_b` expansion), dense-lead blocks, sigmoid/softmax/sqrt-softplus gating, selection bias, group-limited routing, shared expert without a scalar gate | WORKS |
| Long context: `rope.scaling.type` `linear` and `yarn` (factor, original context, per-dimension ramp, magnitude scale incl. `yarn_log_multiplier`) | WORKS |
| `smollm3`: the llama block with every fourth layer's rotary step removed (NoPE) | WORKS |
| Attention sinks (`blk.N.attn_sinks.weight`): one extra softmax logit per head, included in the denominator and never read | WORKS |
| Per-pair rotary frequency divisors (`blk.N.rope_freqs.weight`): those pairs advance slower, layer 0's vector serves every layer | WORKS |
| `rope.scaling.type = longrope` (per-layer short/long factors, phi-3-small) | **MISSING**, refused by name |
| Generation stops at the checkpoint's own `eos_token_id` | WORKS |
| Chat templates: a restricted Jinja interpreter over the checkpoint's own `tokenizer.chat_template` | WORKS (see below) |
| Built-in templates for 11 marker families + evidence-based selection from the token table | WORKS |
| `internlm2`, `exaone`, `orion`, `gptneox`, `stablelm` adapters | WORKS |
| `glm4-moe`, `minicpm3`, and the remaining architectures in llama.cpp's `src/models/` | **MISSING** — see `docs/MODEL_COVERAGE_STATUS.md` for the per-model reason |
| PrismML rotated weight basis (`prism.hadamard.*`) and Hadamard activation transform | **MISSING**, hard blocker |
| Hybrid linear attention (`ssm_*`, `conv1d` state) | **MISSING**, hard blocker |
| Shared-expert MoE (`expert_shared_count`) | **MISSING**, refused rather than dropped |
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
| 20 | Ternary-Bonsai-2-27B-…-PTQ1_0-MTP-Q4_0 | ternary hybrid | PTQ1_0 + Q4_0 | rotated basis, linear attention, PTQ1_0 layout |
| 21 | Ternary-Bonsai-2-2B-PTQ1_0 | ternary | PTQ1_0 | rotated basis, linear attention, PTQ1_0 layout |
| 22 | tiny-llama3-test-Q3_K_M | llama | Q3_K | adapter, tokenizer |
| 23 | Tiny-LLM.Q4_K_M | llama-derived | Q4_K | adapter, tokenizer |
| 24 | Tiny-Moe.Q4_K_M | llama + experts | Q4_K | adapter, MoE layout |
| 25 | tinyllama-1.1b-chat-v1.0-q2_k | llama | Q2_K | adapter, tokenizer |
| 26 | tinyllama-15M-stories-Q2_K | llama | Q2_K | adapter, tokenizer |
| 27 | tinyllama-15M-stories-Q3_K_M | llama | Q3_K | adapter, tokenizer |
| 28 | TinyLLama-v0.Q8_0 | llama | Q8_0 | adapter, tokenizer |
| 29 | tinystories-gpt-0.1-3m.Q2_K | llama/gpt2 | Q2_K | adapter, tokenizer |
| 30 | Uraion-Agent-Steer-Q4_K_M | llama-derived | Q4_K | adapter, tokenizer |

**Verified runnable on the real checkpoints available here: 2 of 30.**

| # | File | Result |
| --- | --- | --- |
| 24 | Tiny-Moe.Q4_K_M.gguf | runs; MoE routing, Q4_K/Q6_K experts, 4096-token SentencePiece table |
| 27 | tinyllama-15M-stories-Q3_K_M.gguf | runs; IQ4_NL/Q5_0/Q5_1/Q8_0/Q4_K tensors, 32000-token SentencePiece table |

`freemoe probe` reports `"runnable": true` for both, and greedy generation
produces fluent continuations (for example tinyllama answers *"Once upon a
time"* with *", there was a little girl named Lily. She loved to play outside
in"*).

The other 28 files were **not obtainable in this environment**: the Drive
connector caps a single download at 100 MiB and every remaining file is larger
(145 MB, 429 MB, 1.18 GB, …). So no claim is made about them. What can be
claimed is narrower and checkable: the adapters those families need now exist
and are verified against an independent implementation — see the parity harness
below.

The "Blockers" column below is the **pre-adapter expectation** recorded when the
list was written. It is stale for every family the capability table now lists as
WORKS. `freemoe probe` on the actual file is the only authoritative answer.

## The ternary models (Bonsai 2)

`Ternary-Bonsai-2-2B-PTQ1_0.gguf` and the 27B MTP variant are the two entries in
the requested list that use a format no mainline runtime reads. Researching them
against the published sources (`ggml-org/llama.cpp`, `PrismML-Eng/llama.cpp`,
the `prism-ml` model cards) established four things, all of which the code now
reflects:

1. **The type ids are outside upstream.** `PQ2_0` is 142 and `PTQ1_0` is 143;
   upstream stops at 43. The four upstream ternary ids — `TQ1_0` 34, `TQ2_0` 35,
   `Q1_0` 41, `Q2_0` 42 — *are* implemented here, with decoders verified against
   an independent reference. `PQ2_0` is implemented too: it is exactly the
   `Q2_0` codec at group size 128, so its layout is not in doubt.

2. **`PTQ1_0` has no verifiable element layout.** PrismML's own CPU reference
   and their Vulkan shader disagree about how many elements a 128-wide block
   holds — the CPU path advances through 88 elements per block while the shader
   maps all 128, and the CPU loop reads past the end of its 24-byte `qs` array.
   Neither can be checked here because no `PTQ1_0` file is downloadable in this
   environment (429 MB). A decoder written today would be a guess, so `PTQ1_0`
   parses (geometry, sizes, pageability correct) and **refuses to decode**.

3. **The weights are in a rotated basis.** Each matrix is transformed by a
   blockwise Hadamard rotation before ternary assignment, and the runtime must
   apply the matching transform to activations. The file declares this with
   `prism.hadamard.*` metadata (`transform`, `block_size`, `sign_mode`,
   `weight_names`, …). This is the dangerous case: the bytes decode to
   plausible numbers and the model emits garbage, so the probe reports
   `ROTATED_WEIGHT_BASIS` as a hard blocker the moment those keys appear.

4. **The blocks are hybrid.** Roughly three quarters of the layers use linear
   attention with `ssm_*` state and `conv1d`. The probe reports
   `LINEAR_ATTENTION` for any checkpoint carrying that state.

So a Bonsai file today yields `verdict: inspect-only` with every reason named,
instead of either a parse failure or a confident wrong answer.

## How a new architecture is added

Every architecture lands the same way, and none of it is "it produced plausible
words":

1. `tools/make_arch_fixtures.py` writes a checkpoint shaped like the real thing —
   standard `<arch>.*` metadata, llama.cpp tensor names, and that family's layout
   quirks (fused qkv for phi3, zero-centred norm scales for gemma, …).
2. `tools/llama_oracle.py` — a **second implementation** of the forward pass that
   shares no code with `src/engine.cpp` — computes reference logits for a fixed
   prompt.
3. The `architecture_parity` unit test runs the engine on the fixture and
   requires its logits to match the oracle within tolerance and its greedy token
   sequence to match exactly.

Fixtures are keyed by shape and architecture, so a fixture that silently ignored
a feature (biases, q/k norm, embedding scale, softcapping, sliding window) would
produce logits identical to the plain fixture and the test would catch it.

Current parity fixtures: `llama-gqa`, `qwen2-bias`, `qwen3-qknorm`, `gemma`,
`gemma2`, `phi3-fused`, `glm4`, `minicpm`, `qwen3moe`, `qwen2moe`,
`phi2-layernorm`, `deepseek2-mla`, `internlm2`, `exaone`, `orion`, `gptneox`,
`stablelm` — 17 architectures.

### What each architecture actually needed

Read from llama.cpp's `src/models/`, not guessed from the model names:

| Architecture | The differences that matter |
| --- | --- |
| `glm4` | `ffn_up` is `[hidden, 2*intermediate]` holding gate then up (`ggml_swiglu` splits it); post-attention and post-FFN RMSNorm applied before each residual add |
| `minicpm` | token embedding scaled (12.0 by default) before the first block; tied output |
| `qwen3moe` | qwen3's per-head q/k norm combined with a stacked-expert FFN and router |
| `qwen2moe` | routed experts plus a *dense* shared expert, whose whole branch is scaled by one scalar per token; three separate feed-forward widths |
| `phi2` | LayerNorm with a bias instead of RMSNorm; no `ffn_norm` at all, because the FFN runs on the attention's normalised input and both branches add to one residual; gate-free GELU FFN with biases; biases on the attention output and the logits |
| `deepseek2` | multi-head latent attention (see below), `leading_dense_block_count` plain-FFN lead-in blocks, a router whose gating function is `expert_gating_func`, a selection-only `ffn_exp_probs_b.bias`, group-limited routing, and a shared expert that has no scalar gate |

### Long context: what `rope.scaling` does here

The rotary maths lives in `src/rope.cpp`, not inside the attention loop, because
a wrong ramp does not crash -- it produces fluent text at the wrong positions.

| `rope.scaling.type` | Behaviour |
| --- | --- |
| absent / `none` | plain rotation; the context-extension fields stay identity |
| `linear` | every angle divided by `factor`; `rope.scaling.attn_factor` multiplies the result |
| `yarn` | `factor` scales the frequency, and the correction ramp keeps low-index (fast) pairs exact while interpolating slow ones; magnitude `1 + 0.1*log(factor)`. `yarn_log_multiplier` supplies `mscale_all_dim`, with llama.cpp's deepseek2 numerator special case |
| `longrope` | refused by name: it needs per-layer short/long factors keyed on the effective context, and no checkpoint we target uses it |

The ramp bounds come from ggml's `ggml_rope_yarn_corr_dim`, solved from the
original context length and `beta_fast`/`beta_slow`. llama.cpp folds a
`1 + 0.1*log(1/freq_scale)` factor into the kernel and divides it back out in
context setup; since there is no such hidden factor here, `rope_mscale` is the
net value and must not be multiplied again.

Evidence: three fixtures + oracle goldens (`llama-yarn`, `qwen2-linear`,
`llama-yarn-mscale`). The YaRN one is chosen so the ramp actually bends --
for an 8-wide rotation over 64 tokens the correction dims are {0, 2}, so pair 0
keeps its exact angle, pair 1 blends, and pairs 2-3 interpolate. `rope_scaling`
unit checks pin those numbers, then confirm the fixture's raw metadata reaches
`ModelConfig` (a checkpoint that parsed but silently rotated unscaled would pass
a logits-only test).

`glm4` also parses `rope.dimension_sections` for multimodal RoPE; a text-only
checkpoint omits it and uses standard RoPE, which is what is implemented.

### Multi-head latent attention (`deepseek2`)

MLA has no `attn_k`/`attn_v` tensors at all, so the four things a reader has to
get right cannot be inferred from the plain attention path:

* **Query.** `wq_a [hidden, q_lora_rank]` → RMSNorm of width `q_lora_rank` →
  `wq_b [q_lora_rank, heads * key_length_mla]`. `q_lora_rank == 0` is the "lite"
  layout with a single `attn_q`.
* **Keys and values.** `attn_kv_a_mqa [hidden, kv_lora_rank + rope_dim]` splits
  into the latent and the shared rotary key `k_pe`. The latent is RMSNorm'd at
  width `kv_lora_rank`, then each head expands it: `attn_k_b` has ggml dims
  `{nope, kv_lora_rank, heads}` with **ne0 = nope**, so element `(o, i, head)`
  sits at `o + nope * (i + kv_lora_rank * head)` — the latent index is the slow
  axis of each row, not the fast one. `attn_v_b` is
  `{kv_lora_rank, head_v, heads}` and *does* have the latent as ne0. Rotating
  both the same way is a plausible-looking mistake that produces numeric
  nonsense on a real checkpoint.
* **Rotary.** Applied to the trailing `rope_dim` of every query head and to the
  shared `k_pe` only; the nope part of the key is never rotated.
* **Value width.** `attention.value_length_mla` makes the value wider than the
  key, so the KV cache stride is `kv_dim + v_dim` and the attention output is
  `heads * head_v` wide. Under plain multi-head attention those coincide, which
  is why this only breaks on MLA or grouped-query models.

Two further traps cost real debugging time here, and both are pinned by tests:

* A norm must divide by the **vector's** length, not by `hidden`. MLA normalises
  latents of width `q_lora_rank` and `kv_lora_rank`; dividing by `hidden` scales
  them by `sqrt(hidden / width)` and every logit moves.
* Presence checks must scan **all** layers. A deepseek checkpoint's leading
  dense blocks carry none of the MoE tensors, so `blk.0.ffn_gate_shexp` and
  `blk.0.ffn_exp_probs_b.bias` both report "absent" on a model that has them.
  `has_any_layer()` in the loader and the oracle's `any(...)` scan are the fix.

Routing also differs by family and is read from `expert_gating_func`:
`1` softmax, `2` sigmoid (V2/V3), `3` softmax over the selected weights only,
`4` sqrt-softplus. The `ffn_exp_probs_b.bias` is added for **selection only** —
the weights that multiply the expert outputs stay unbiased — and group-limited
routing masks whole groups by setting their selection score to −inf.

### Chat templates

`tokenizer.chat_template` is Jinja2, and without it a chat model is unusable:
the role markers, the BOS placement and the trailing assistant prefix are all
part of the prompt the model was trained on, so feeding raw text produces
confidently wrong output rather than an error.

FreeMoe ships a **restricted Jinja interpreter** (`src/chat_template.cpp`) and
renders whatever the checkpoint declares. The alternative — llama.cpp's approach
of a table of C++ string builders plus a chain of substring heuristics — needs
one builder per known template and *silently* produces a wrong prompt for
anything unrecognised. Reading the template is smaller (one engine instead of
sixty builders) and more correct; the cost is a language subset, which is why an
unsupported construct is a named, typed `NOT_IMPLEMENTED` error raised at
*chat* time only, so plain completion keeps working on the same checkpoint.

Supported: `{% for %}` (with `if` filters, `else`, and `loop.index/index0/first/
last/length/revindex`), `{% if %}/{% elif %}/{% else %}`, `{% set %}` including
`namespace()` field assignment, `{{ }}` output, `{# #}` comments, whitespace
control (`{%- -%}`), the operators `+ - * / // % ~ == != < <= > >= and or not in
in`, `is defined/none/string/number`, indexing, slices, `[a, b]` lists, and the
filters `trim strip upper lower length string int float first last list join
replace default tojson`. `raise_exception()` is honoured, which is how gemma
rejects a system turn and llama2 rejects non-alternating roles — those come back
as typed errors instead of malformed prompts.

Not supported, and reported as such: `{% macro %}`, `{% include %}`, `{% import
%}`, `{% filter %}`, `{% with %}`, `{% set %}` inside a filter expression, and
attribute access on anything but maps.

**Choosing a template** (`resolve_chat_template`):

1. `--chat-template-file` or `--chat-template` (a built-in name or literal
   template text) wins.
2. Otherwise the checkpoint's own `tokenizer.chat_template`.
3. Otherwise a built-in picked from the **token table** — does the vocabulary
   actually contain `<|start_header_id|>`, `<start_of_turn>`, `<|im_start|>`,
   `[INST]`? That is evidence, and it is stronger than the architecture name
   (`llama` covers both the llama2 and llama3 marker sets). The architecture
   name is only the last resort.
4. Nothing applies: `--chat` fails with an error naming the architecture rather
   than inventing a prompt. `run` without `--chat` is unaffected.

Built-in templates: `chatml`, `llama2`, `llama3`, `gemma`, `phi3`, `deepseek2`,
`orion`, `zephyr`, `vicuna`, `raw` (the native adapter's protocol constant), and
`gptneox` has none — its checkpoints carry no standard markers.

The provenance of every render is reported: `chat_template_source` is
`checkpoint`, `builtin:<name>` or `override`. "The file told us" and "we used a
default" are different claims about whether the prompt is right, so the CLI
never blurs them. `--render-only` prints the exact bytes handed to the
tokenizer, which is the only way a template is observable.

### Tokenizer: merge order, not longest match

An encoder can round-trip perfectly and still disagree with every other runtime
about the ids, which is the worst kind of bug: the model computes correct logits
for the wrong prompt.

Byte-level BPE is therefore implemented as BPE — the family pre-tokeniser cuts
words, every byte goes through the GPT-2 alphabet, and merges are applied
lowest-rank-first. Four splitting families cover the vocabularies in use:

| Family | Selected by `tokenizer.ggml.pre` | Splitting rule |
|---|---|---|
| `llama3` | `llama3`, `llama-v3`, `llama-bpe`, `chatglm4`, `dbrx`, `smaug`, `falcon3`, `lfm2` | contractions, `[^\r\n\p{L}\p{N}]?\p{L}+`, `\p{N}{1,3}`, punctuation runs, CR/LF runs; whole-word lookup first (`ignore_merges`) |
| `qwen2` | `qwen2`, `stablelm2`, `hunyuan`, `solar-open`, `deepseek-r1-qwen` | as `llama3` with one digit per word |
| `gpt2` | `gpt-2`, `phi-2`, `mpt`, `olmo`, `jais`, `granite-docling`, `gigachat`, `mellum` | `'s\|'t\|…\| ?\p{L}+\| ?\p{N}+\| ?[^\s\p{L}\p{N}]+\|\s+(?!\S)` |
| `gpt2-digits` | `starcoder`, `refact`, `command-r`, `smollm`, `codeshell`, `exaone`, `minerva`, `mellum2` | as `gpt2` with each digit split off first |

Verification is against **llama.cpp's own expected token ids** for the
vocabularies it ships (`models/ggml-vocab-*.gguf` + `.inp`/`.out`):

```sh
python3 tools/tokenizer_parity.py --vocab-dir /path/to/llama.cpp/models
```

Last run: `gpt-2`, `llama-bpe`, `llama-spm`, `phi-3`, `qwen2`, `starcoder`,
`refact`, `command-r` all match on **46/46** cases each. `mpt` matches 39/46 —
every mismatch is a whitespace-only string whose reference vector uses the
raw-space token while the rest of the same file uses the GPT-2 escaped form; the
code path is the one that passes `gpt-2` outright, so the vectors are
inconsistent rather than the encoder.

SentencePiece vocabularies are segmented the way the reference does it, by
score, not by longest match: every UTF-8 character starts as a symbol, and the
highest-scoring adjacent pair whose concatenation is a piece is merged until
nothing merges. The dummy space prefix is added unconditionally — including for
text that already starts with a space, which is why `" hello"` starts with two
markers — because that is what the reference does and the difference is visible
in the expected vectors.

Unicode categories come from tables generated from Python's `unicodedata`
(`tools/gen_unicode_tables.py`), because a wrong category silently moves a word
boundary. `make test` fails if the committed table is stale.

## Breadth: the block is data, not code

Five further adapters landed with no new forward-pass code: `internlm2`,
`exaone`, `orion`, `gptneox` and `stablelm`. Each differs from the existing
block only in facts the checkpoint can state — the norm type (from which epsilon
key the file declares), norm biases (from the tensors), the rotary width, a
fused q/k/v projection, and the activation.

The activation is the one fact GGUF cannot express for a *gate-free* FFN (up →
activation → down has no gate to disambiguate it), so it is read from the
architecture. That table is deliberately small and cites llama.cpp's own model
files: `phi2` and `gptneox` are `LLM_FFN_GELU` with `LLM_FFN_SEQ`. Any other
architecture with a gate-free FFN is still refused rather than guessed.

## Ordered work that remains

1. ~~Standard metadata + tensor-name mapping (`llama` family).~~ **DONE**
2. ~~BPE / SentencePiece tokenizer.~~ **DONE**
3. ~~Grouped-query attention.~~ **DONE**
4. **Standard MoE FFN** — stacked `ffn_*_exps` with `ffn_gate_inp` routing works
   and runs a real checkpoint (#24). ~~`expert_shared_count`~~ **DONE** for
   `qwen2moe`: the shared expert runs at its own width and is gated per token.
   Still missing: layouts where a checkpoint narrows the routed experts below the
   dense width on architectures other than `qwen2moe`, and the newer per-expert
   splat naming.
5. **Q6_K / Q3_K / Q2_K / Q5_K AVX2 kernels** — correctness is already there via
   the scalar path; this is the throughput step.
6. ~~Architecture variants: `gemma`, `phi3`, `qwen2`, `qwen3`, `glm4`,
   `minicpm`, `qwen3moe`, `qwen2moe`, `phi2`.~~ **DONE**
   ~~MLA (`deepseek2`)~~ **DONE** — see the section above.
   Still open: `glm4-moe`, `minicpm3`, `olmo2` (its q/k norm is *not* per head,
   so it needs a second norm placement), `olmoe`/`mixtral` (MoE variants of the
   same block), the MTP/`nextn` heads (parsed and reported as ignored), and the
   rest of llama.cpp's `src/models/` tree.
6b. ~~**Chat templates.**~~ **DONE** — restricted Jinja interpreter, 10 built-in
   templates, evidence-based selection, `--render-only` for inspection.
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
