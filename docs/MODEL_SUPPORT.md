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
| `glm4-moe`, `minicpm3`, MLA attention (`deepseek2`), and the remaining 158 architectures in llama.cpp's `src/models/` | **MISSING** |
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
`phi2-layernorm`.

### What each architecture actually needed

Read from llama.cpp's `src/models/`, not guessed from the model names:

| Architecture | The differences that matter |
| --- | --- |
| `glm4` | `ffn_up` is `[hidden, 2*intermediate]` holding gate then up (`ggml_swiglu` splits it); post-attention and post-FFN RMSNorm applied before each residual add |
| `minicpm` | token embedding scaled (12.0 by default) before the first block; tied output |
| `qwen3moe` | qwen3's per-head q/k norm combined with a stacked-expert FFN and router |
| `qwen2moe` | routed experts plus a *dense* shared expert, whose whole branch is scaled by one scalar per token; three separate feed-forward widths |
| `phi2` | LayerNorm with a bias instead of RMSNorm; no `ffn_norm` at all, because the FFN runs on the attention's normalised input and both branches add to one residual; gate-free GELU FFN with biases; biases on the attention output and the logits |

`glm4` also parses `rope.dimension_sections` for multimodal RoPE; a text-only
checkpoint omits it and uses standard RoPE, which is what is implemented.

### The one thing GGUF does not say

Everything above is decided from tensors, shapes or a metadata key: the norm
type comes from which epsilon key the file declares, the biases from the
tensors, the fused layouts from the shapes, and a missing `ffn_norm` from the
absence of the tensor.

The *activation function* is the exception. GGUF has no key for it, so a
gate-free FFN — up, activation, down, with no gate to disambiguate it — cannot
be identified from the file at all. FreeMoe therefore accepts a gate-free FFN
only for `phi2`, applies GELU there, and refuses every other architecture with
a `NOT_IMPLEMENTED` error rather than guessing. A parity fixture is the check,
and `phi2-layernorm` is that fixture.

### A reference that does not do what it documents

minicpm declares three scales: `embedding_scale`, `residual_scale` and
`logit_scale`. llama.cpp reads and logs all three, but a repository-wide search
across the graph code finds only `embedding_scale` actually applied. The other
two are dead metadata in the reference.

FreeMoe therefore applies the embedding scale and **records but does not apply**
the other two, because applying them would disagree with the reference on every
logit. The probe says so in `scale_note` rather than leaving it to be inferred
from silence, and `minicpm_scales` pins the decision in a test.

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
   Still open: `glm4-moe`, `minicpm3`, MLA (`deepseek2`), and the rest of
   llama.cpp's `src/models/` tree. MLA is the largest remaining capability:
   it needs low-rank q/kv compression plus per-layer dense-lead blocks, so it
   is its own milestone rather than a variant of the attention already here.
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
