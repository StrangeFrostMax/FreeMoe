# Gap analysis: design vs code vs model codes

> **Status of this document.** Section C's findings C1, C4, C6, C7, C9 (activation
> half) and C10 are fixed:
> BPE and SentencePiece are real encoders now (12 vocabularies at 46/46,
> `tools/tokenizer_parity.py`), linear/YaRN/`rope_freqs` context extension and
> attention sinks are implemented, `auto` no longer dispatches AVX-512 on a
> Zen 3 target, and the sampler has min-p, the three penalties, per-token logit
> bias and `--n-probs`. The measurements below are the state at `f704e4b` and are kept
> as the baseline this work is measured against; current numbers live in
> `docs/MODEL_COVERAGE_STATUS.md` (3241 checks, 55 architecture fixtures).
>
> What changed since the baseline, in the terms this document uses:
>
> * **C7 closed.** The IQ family (IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
>   IQ3_XXS, IQ3_S, IQ4_XS) and both FP4 formats (MXFP4, NVFP4) decode and
>   execute. Every decoder **matches** llama.cpp's own `dequantize_row_*` to
>   within 1e-6 relative of the block magnitude -- not bit-exactly, because the
>   comparison is between two differently-ordered float reductions
>   (`tools/quant_upstream_check.py`, 26 types, 0 failures, worst absolute
>   5.0e-06 on IQ4_NL, MXFP4/NVFP4 exact). The grids come from a generated table
>   file so the transcription is mechanical, and `make test` re-checks that table
>   structurally (`tools/gen_quant_tables.py --check`, no upstream checkout
>   needed). Only `PTQ1_0` stays geometry-only: no verifiable layout exists, so
>   it is refused by name rather than guessed.
> * **C9 closed for the reachable families.** `swiglu-oai` is no longer a
>   primitive with no user: `gpt-oss` runs it end to end (`gpt-oss` fixture, an
>   oracle golden, a parity row and `gpt_oss_block`), together with the rest of
>   what `models/openai-moe.cpp` does -- sinks, the post-attention norm used as
>   the MoE's input norm, the biased router, per-expert projection biases,
>   `softmax-weight` gating and alternating full/windowed layers. Per-layer
>   `swiglu_clamp` stays covered by `activation_clamp_forms`; `deepseek4/4.1`
>   remain unrowed because their graph carries far more than the clamp (DSA
>   indexer, hyper-connections, MTP), not because the arithmetic is missing.
>   `SITU` remains open -- it is unimplemented upstream too (`GGML_ABORT`).
>   `GEGLU_ERF` is now reachable as well: `mistral3` builds its adaptive-norm MLP
>   with exactly that activation (`LLM_FFN_GELU_ERF`, `LLM_FFN_SEQ`).
> * **mRoPE is wired.** `qwen3vl` and `qwen3vlmoe` have fixtures, goldens and
>   parity rows, the section split is a required key for those families as it is
>   upstream, and the interleaved layout is chosen by architecture the way
>   `llama_rope_type()` does. `qwen35`/`qwen35moe` remain blocked by their
>   gated delta-net layers, which is a different scan from the mamba one below.
> * **The recurrent block runs.** `mamba`, `mamba2` and the falcon-mamba
>   `dt_b_c_rms` variant have fixtures, oracle goldens and parity rows: a 1D
>   convolution plus the `ggml_ssm_scan` arithmetic, with the state sized and
>   reserved per layer instead of a per-position cache. The **hybrid layout**
>   runs as well (`jamba` fixture, golden and parity row): recurrent and
>   attention layers interleaved in one stack, with the split, the feed-forward
>   and the router all read per layer, and one cache region holding a
>   per-position cache for the attention layers and a rolling state for the
>   recurrent ones.
> * **Adapter rows carry llama.cpp's tag vocabulary** (`inspect --tags`,
>   `freemoe adapters`), so a failure can be compared with an upstream
>   `llama_model_loader` message tensor by tensor.

Measured on `f704e4b` (merged PR #10), 2026-10-28, against `ggml-org/llama.cpp`
`master` fetched the same day (169 files under `src/models/`).

Three questions, three sections:

* **A. Design → code.** Does the code meet what
  `Final_Llama_Like_Disk_Resident_MoE_Spec.md` promises, phase by phase, and are
  the gates met?
* **B. Code → model codes.** Of the 169 architectures llama.cpp implements, how
  many run here today, how many are one adapter row away, and which single
  capability unlocks the most of the rest?
* **C. Code → correctness.** Where does the code *claim* a capability but not
  match reference semantics? These are the dangerous gaps, because nothing
  fails — the numbers are just quietly different.

Every claim below is either a result produced in this tree or a citation into
`src/` or upstream `src/models/`.

---

## Baseline evidence

| Measurement | Value |
| --- | --- |
| Unit / parity / fuzz checks | **3702, 0 failures** (`./build/tests`) |
| Architecture parity fixtures | **70** slugs in `architecture_parity`, 91 `.gguf` fixtures (31 MB) — compared against `tools/llama_oracle.py` goldens, not against this engine |
| Model adapters | **67** entries = 66 real architectures + native `freemoe-tiny` (`src/support.cpp`, `freemoe adapters`) |
| Upstream architectures with an adapter | **61 of the 171** files in upstream `src/models/` match an adapter row by name, plus five that upstream spells differently (`gpt-oss`→`openai-moe`, `granitemoe`→`granite-moe`, `smollm`, `smollm2`, `tinyllama`→the `llama` block) |
| Quantisation types decodable | **26** (`src/quant.cpp`) |
| Error codes | **28** non-OK codes (`include/fm/common.hpp:27`) |
| Benchmark vs declared targets | **pass** on the synthetic fixture; the recorded runs predate `--kernels` and so were dispatched on AVX-512 (C10) |

### The one benchmark that exists

`bash benchmarks/run_repeats.sh configs/benchmark.env` → `results/benchmark/synthetic-moe-buffered/`
(git-ignored), 5 process-cold repetitions, 20 prompts × 16 tokens, model
`tests/fixtures/moe-fm.gguf` (native 2-layer fixture, vocab 258), buffered
reader, `CACHE_SLOTS=1`:

| Metric | Median | Range | Declared target | Verdict |
| --- | --- | --- | --- | --- |
| Prefill | 3051 tok/s | 2991–3072 | ≥ 1000 | pass |
| p95 inter-token | 0.41 ms | 0.39–0.58 | ≤ 50 ms | pass |
| First token | 85.5 ms | 84.9–87.2 | — | — |
| Peak process RAM | 251 MB | ±78 kB | — | — |
| Reference comparison | 20/20 prompts, 320/320 tokens, 0 routing mismatches, max rel diff 4.3e-4 | — | 20/20 | pass |

Two things this does **not** establish, both already noted in
`docs/BENCHMARK_PROTOCOL.md`: the model is a 2-layer synthetic fixture, so
3051 tok/s says nothing about a real MoE checkpoint or about disk paging
(`cache_hit_rate` is 0.0 because there is one slot and no eviction); and the
run dispatched `cpu-avx512` kernels, which is not the Zen 3 profile. See C10.

---

## A. Design → code

The spec's own gates in order. "Met" means the spec's stated exit condition,
not "some code exists".

| Phase | Spec gate | State | Evidence / what is missing |
| --- | --- | --- | --- |
| 0 — repo / dependency audit | clean build on the declared target | **MET** | `make -j2`, `docs/PHASE0_EXIT.md`, `scripts/collect_evidence.py`. |
| 1 — GGUF inspection + dense CPU correctness | malformed models fail safely, tensor roles validate, output matches reference | **MET** | 1338 checks, 30 fixtures in `tests/malformed_models/`, 17-architecture parity against an independent oracle (`tools/llama_oracle.py`). |
| 2 — dense GPU baseline | no silent fallback, stable output, bounded allocations | **NOT MET** | `src/backend.cpp:545` is a placeholder raising `GPU_BACKEND_UNIMPLEMENTED`; no CUDA/Vulkan/HIP code exists. It refuses rather than faking it — correct, but the phase is open and every VRAM gate with it. |
| 3 — one MoE family, resident experts | selected experts never dropped, routing matches reference | **MET for stacked-expert MoE** | `ResidencyManager::assert_selected_executed` (`src/residency.cpp:785`) is called after every MoE layer (`src/engine.cpp:1070`) and fails if a selected expert never ran; `qwen2moe`, `qwen3moe`, `deepseek2` parity; 0 routing mismatches in the benchmark report. **Not** met for per-expert tensor layouts (C8). |
| 4 — VRAM expert slot pool | stress tests with forced eviction, no stale reads | **MET in host RAM, structurally blocked in VRAM** | Slot state machine with generation ids and publish-after-fence (`src/residency.cpp:59–77`), LRU + clock (`src/residency.cpp:385`), checksum verification, stale-completion rejection. All tested. The pool lives in host memory because Phase 2 is unstarted, so the *VRAM* half of the gate cannot be met. |
| 5 — trace + cache policy | policy beats LRU/clock on decode p95 across repeat runs | **PARTIAL** | Trace events (`RoutingEvent`, `--trace`), both baselines implemented, repeat harness exists — but the committed config runs one slot, so **no replacement happens and no policy can be compared**. The gate needs an eviction-exercising config (`CACHE_SLOTS` < working set). |
| 6 — NVMe-backed expert delivery | cold-cache correctness, bounded queues, meets target | **PARTIAL** | `pread` / `mmap` / `O_DIRECT` readers (`src/io.cpp:167`), extent coalescing (`coalesce_extents`, `src/io.cpp:46`), prefetch issue/poll/cancel. No cold/warm suite, no working-set instrumentation, and the only measured run is buffered with one slot. |
| 7 — optional layout optimisation | A/B proves end-to-end gain | **NOT STARTED** (correctly gated) | Blocked on Phase 6 evidence, as the spec requires. |
| 8 — advanced optimisations | each a separate experiment | **NOT STARTED** | No speculative decoding, no batching, no multi-GPU, no dequantisation cache, no fused expert kernels. |

### The performance gate

Targets *are* declared (`docs/BENCHMARK_PROTOCOL.md:54`, provisional, sandbox
scoped: prefill ≥ 1000 tok/s, p95 inter-token ≤ 50 ms). They pass — see the
table above. But the same paragraph pins the release gate to a real D1 machine
record, and the machine here is an Intel Xeon KVM guest with 2 vCPUs using
AVX-512 dispatch. So: **target edges met on a synthetic fixture; the
"performance-qualified" label in spec §16 remains unearned**, and earning it
requires (a) a config that exercises eviction, (b) an AVX2-only path, (c) a
non-synthetic model.

### Spec sections only partly honoured

| Spec § | Requirement | Gap |
| --- | --- | --- |
| §7.2 / §7.3 | stacked **and** per-expert tensor layouts | **Stacked only** (`ffn_*_exps` + `ffn_gate_inp`). Per-expert `ffn_gate.E.weight` naming is refused. |
| §9.2 | direct I/O usable, fallback explicit | `O_DIRECT` opened on request; alignment handled; buffered-vs-direct never measured on this machine. |
| §10.2 | prefill measured separately, with per-layer unique-expert counts and cache hit rate during prefill | Metrics exist, but prefill is **token-serial** (`src/engine.cpp:1146`), so prefill expert working set is indistinguishable from decode and the §10.2 prefill rows are uninformative rather than wrong. |
| §10.3 | prefetch policy comparable | Prefetch implemented; no policy comparison run. |
| §11.4 | declared tokens/s target recorded before Phase 4 | Recorded, provisional and sandbox-scoped; the real machine is `unknown (D1)`. |
| §12.2 | one GPU backend chosen | Not chosen; placeholder only. |
| §14 | typed, located errors | Met broadly. One hole: **KV exhaustion** is checked at admission only (`prompt + generated > context`), not per step. |
| §15 | malformed fixtures, sanitizers, declared fuzz duration | Fixtures and fuzz target exist (`tests/fuzz_gguf.cpp`); a sanitizer run is not part of `make test` and no duration is recorded in a committed artefact. |
| §16 | "correctness prototype" | **Met**: dense and MoE complete inference, mappings explicit, reference comparisons pass, no expert silently skipped, memory bounded and reported. "Paged MoE prototype" met in host RAM only. |

---

## B. Code → model codes

### The census

**171** model files upstream (`src/models/*.cpp`, counted this change). The
table below is set arithmetic against that count, not an estimate: the 110 files
with no adapter row were listed by name, then partitioned into the three groups
under it.

| Bucket | Count | Meaning |
| --- | --- | --- |
| Adapter today | **61** | Files in upstream `src/models/` whose name matches an adapter row. `freemoe adapters` reports **66** rows: the extra five are upstream aliases (`gpt-oss`→`openai-moe`, `granitemoe`→`granite-moe`) and block renames (`smollm`, `smollm2`, `tinyllama`→`llama`). `docs/MODEL_COVERAGE_STATUS.md` groups them by capability |
| **Unmapped** | **110** | `comm` between `freemoe adapters` and the upstream file list |
| — not a causal text decoder | 24 | `bert` `nomic-bert` `nomic-bert-moe` `jina-bert-v2` `jina-bert-v3` `eurobert` `modern-bert` `neo-bert` `gemma-embedding` `gemma-embedding2` `pangu-embed` `t5` `t5encoder` `clip` `cogvlm` `qwen3tts` `wavtokenizer-dec` `llada` `llada-moe` `rnd1` `dream` `dflash` `deepseek2ocr` `vllm-decision2` — pooling/no-causal-mask encoders, diffusion, audio or vision |
| — recurrent, linear attention or MTP | 30 | `rwkv6` `rwkv6-base` `rwkv6qwen2` `rwkv7` `rwkv7-base` `arwkv7` `mamba-base` `bailingmoe` `bailingmoe2` `bailingmoe3` `qwen35` `qwen35moe` `qwen3next` `kimi-linear` `kimi-k3` `falcon-h1` `nemotron-h` `nemotron-h-moe` `granite-hybrid` `granite-switch` `hy-v3` `hy-v4` `delta-net-base` `lfm-base` `lfm2` `lfm2moe` `lfm3` `hrm-text` `spark2-5` `k2-horizon` — a new cache shape and kernel, not a row |
| — a different causal block | 56 | `afmoe` `apertus` `arctic` `beam` `chameleon` `clef` `cohere2moe` `dbrx` `deepseek` `deepseek32` `dots1` `dots3note` `eagle3` `ernie4-5` `ernie4-5-moe` `exaone-moe` `exaone4` `flint` `gemma3n` `gemma4` `gemma4-assistant` `glm-dsa` `glm4-moe` `glm5-next` `granite-moe` `granite-swa` `grok` `grovemoe` `hunyuan-moe` `hunyuan-vl` `kolibri1` `laguna` `limite` `llama4` `maion-coder` `maple` `mellum` `mimo2` `minicpm3` `minimax-01` `minimax-m2` `minimax-m3` `mistral-large4` `mistral4` `muse-glimmer` `muse-spark` `openai-moe` `openelm` `plamo2` `plamo3` `qwen2vl` `qwen4exp` `refact` `sensenova-u1` `step35` `talkie` — two of these are aliases already covered (`granite-moe`, `openai-moe`), so **54** distinct blocks |

The earlier version of this table said "169 files" and split the remainder into
"needs only a row" and "needs one more primitive" by hand. That split was not
reproducible, so it is gone: what is reproducible is the three groups above, and
each row in the 54 is claimed only after its upstream file has been read — which
is the rule the last three batches followed.

### What one capability unlocks

Greedy set-cover over the pending architectures — each step adds the single
capability that completes the most models given everything before it.

| Add | Newly fully supported | Which |
| --- | --- | --- |
| `rope_freqs` / YaRN frequency factors | **+14** | apertus cogvlm deci exaone-moe exaone4 gemma4-assistant granite granite-moe llama4 mistral3 nanbeige pangu-embed refact step35 |
| gated-delta-net (linear attention) | **+9** | clef delta-net-base glm5-next kimi-linear muse-spark qwen35 qwen35moe qwen3next qwen4exp |
| lightning / generic recurrent attention | **+8** | falcon-h1 granite-hybrid jamba mamba mamba-base mamba2 nemotron-h plamo2 |
| mRoPE (sectioned rotary) | **+6** | bailingmoe3 ernie4-5 glm4-moe hunyuan-vl qwen2vl qwen3vl qwen3vlmoe |
| fused `ffn_gate_up_exps` | **+6** | cohere2moe deepseek32 dots3note gemma4 glm-dsa hy-v3 |
| attention sinks | **+6** | deepseek4 deepseek41 flint granite-swa hy-v4 mimo2 |
| RWKV | **+4** | arwkv7 rwkv6 rwkv6-base rwkv7 rwkv7-base (5 files, 4 complete) |
| short-conv (LFM) | **+4** | lfm-base lfm2 lfm2moe lfm3 |

Cost-effectiveness ordering is therefore: the 54 causal blocks first (mostly no
engine code), then the recurrent families together (one cache shape shared by
many files), then the small token-level primitives.

Three of those buckets have since been closed and the table above is kept as the
baseline it was measured at: **mRoPE** (`qwen3vl`, `qwen3vlmoe`), **attention
sinks combined with `swiglu_oai`** (`gpt-oss`, and with it everything
`minimax-m3` adds except the indexer), and **the generic recurrent block**
(`mamba`, `mamba2`, `mamba-dtrms`). What each bucket still needs is now a
different thing in each case: rows for the mRoPE families that have no other
missing piece, an indexer/MLA-dense-tail graph for `deepseek4/41`, and the
hybrid layout for the recurrent ones whose attention layers need a cache
beside the state.

### The 10 that stay hard

`gemma3n kimi-k3 minimax-m3 nemotron-h nemotron-h-moe openai-moe plm` — each
needs a change to the block itself, not a variant of it: AltUp / per-layer
embeddings (`gemma3n`), KDA plus the `SITU` timing nonlinearity (`kimi-k3`),
clamped `SWIGLU_OAI_MOE` with attention sinks (`openai-moe`, `minimax-m3`),
recurrent layers interleaved with attention (`nemotron-h*`), MLA plus `relu`
no-gate (`plm`).

`arcee`, `nemotron` and `jais2` used to be on this list and should not have
been: their block *is* the llama block with a gate-free `relu²` FFN, which needs
an entry in the gate-free activation table rather than a new layer. All three
now have an adapter row, a fixture and an oracle golden.

### Caveats on this census

The signatures are static, so they **overestimate** readiness where the
subtlety is not a tensor name:

* `olmo2` — its `attn_q_norm` is `{n_embd}`, i.e. **not per head**, unlike every
  other qk-norm family here. Needs a second norm placement.
* `gemma3n` — AltUp and per-layer embeddings; not a data change at all.
* `llama4` / `grovemoe` — router special cases upstream hard-codes.
* `chameleon`, `talkie`, `sensenova-u1` — qk-norm plus their own embedding
  peculiarities.
* `deepseek2ocr`, `qwen*VL`, `hunyuan-vl` — the **text** path would be runnable;
  the vision towers are not, and the spec puts multimodal out of scope.

Honest summary: **18 runnable, 56 one row away, 71 one primitive away, 24 out of
scope.** "All models" is a direction, not a state; the number that matters today
is 18/169.

---

## C. Code → correctness gaps (the dangerous ones)

These do not fail loudly. Ordered by damage.

### C1. ~~BPE encoding is max-match, not BPE~~ — **FIXED at `f704e4b`; verified at `main`**

> **FIXED.** `merge_rank_` is consulted (`src/tokenizer.cpp`), the encoders are
> real BPE/SentencePiece, and llama.cpp's own vectors agree on 12 vocabularies
> (`tools/tokenizer_parity.py`, 46/46 each). The body below is the state at the
> baseline and is kept as the record.

`src/tokenizer.cpp` parses `tokenizer.ggml.merges` into `merges_` and
`merge_rank_` (lines 119–130) and then **never consults them**. `encode()` does
a greedy longest-match over the piece table (lines 195–227). Real byte-level BPE
(llama 3, Qwen 2.5/3, GPT-2, starcoder) pre-tokenises with the GPT-2 regex and
then applies merges by rank. Max-match yields different ids on plenty of
ordinary text.

Consequence: for those models the prompt is tokenized differently from every
other runtime, so **logits can be perfect and the output still wrong** — and a
stage-by-stage comparison would blame arithmetic. Invisible today because every
architecture fixture uses the byte tokenizer and `test_tokenizer_real_vocabularies`
(`tests/unit/tests.cpp:1565`) only checks a hand-built SentencePiece table plus
round-tripping.

Fix: GPT-2 pre-tokenisation regex (`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`),
lowest-rank merge loop, byte-level `Ġ` remapping on decode, then a reference
test against a real vocabulary.

### C2. ~~Prefill is sequential~~ — **FIXED; measured**

> **FIXED.** `Engine::prefill_batch` evaluates the whole prompt in one
> layer-major pass: each projection becomes a `gemm_quant` call over all prompt
> rows (`include/fm/kernels.hpp`), the KV cache is filled for the batch before
> attention runs, and logits are computed for the last row only. `run_prompt`
> takes that path whenever `supports_batched_prefill()` accepts the block, and
> `--prefill sequential` (`BackendConfig::batched_prefill = false`) keeps the
> token-at-a-time path for comparison. Both paths are pinned against each other
> by `batched_prefill` (62 checks) — identical tokenisation, logits and greedy
> ids — and the seven block features the batch path does not model yet are
> pinned by name, so a family cannot silently take a path that ignores its
> detail.
>
> Measured here (`tools/bench_prefill.py`, synthetic dense llama, hidden 512 /
> intermediate 2048 / 8 layers, 2-core host, `--prefill` toggled, median of 2):
>
> | prompt | sequential | batched | speedup |
> | --- | --- | --- | --- |
> | 185 tokens (x4) | 1443.1 ms | 491.4 ms | **2.94x** |
> | 369 tokens (x8) | 3765.5 ms | 1382.2 ms | **2.72x** |
>
> The original finding follows. `run_prompt` fed prompt tokens one at a time.
> Correct, but it cost N forward passes instead of one GEMM-heavy pass, and it
> made the §10.2 prefill metrics meaningless (see §A).

### C3. Sliding-window models reserve and keep the full cache

`kv_bytes = layers * context * (kv_dim + v_dim) * 4` (`src/residency.cpp:179`)
ignores `sliding_window`. The mask is applied correctly per head
(`first_visible`, `src/engine.cpp:834`), so results are right, but a gemma-2/3 or
cohere2 checkpoint pays for full-context KV it can never read. No ring buffer.

### C4. ~~No RoPE scaling of any kind~~ — **FIXED; verified at `main`**

> **FIXED.** `src/rope.cpp` implements linear and YaRN scaling and refuses
> `longrope` by name. The body below is the baseline record.

No `rope_scaling`, `rope_freqs`, YaRN or linear scaling anywhere in `src/`
(26 upstream files use per-layer `rope_freqs` or frequency factors). Those
models will produce wrong positions outside the native window — and for
long-context checkpoints, inside it too.

### C5. Router renormalisation — **FIXED; verified at `main`**

> **FIXED.** The denominator is config-driven: upstream's clamp for every family
> and `+1e-20` for `deepseek41` (`src/engine.cpp`, wired from the checkpoint).
> The body below is the baseline record.

Upstream (`llama-graph.cpp:2338–2358`) sums the selected weights after any
`SOFTMAX_WEIGHT` step and clamps the denominator to `6.103515625e-5`
(smallest F16 normal), **except** `DEEPSEEK41`, which uses `sum + 1e-20`.
This engine (`src/engine.cpp:1010–1018`) computes the sum *before* the softmax
and guards with `total > 0.0f`, with no clamp and no per-architecture rule.

* The missing clamp is harmless while the sum stays above 6.1e-5.
* The ordering matters only for `SOFTMAX_WEIGHT` gating — used upstream by
  `openai-moe` alone, which is not supported here. So the bug is **latent, not
  reachable today**; it must be fixed in the same change that adds that family.

### C6. ~~Sampling is thin~~ — **FIXED; verified by mutation**

`SamplingConfig` had temperature, top-k, top-p and a seed: enough for parity
runs, wrong for chat. It now also carries **min-p**, **repetition**,
**presence** and **frequency** penalties with a bounded penalty window, a
**per-token logit bias** (`--logit-bias 12:+4,900:-2.5`), and `--n-probs N`,
which reports the alternatives that survived truncation next to each sampled
token.

Two details are load-bearing. The penalties are applied to the logits from a
pure function, `apply_sampling_bias(logits, config, history)`, rather than
inside the sampler: they need the tokens already produced, and a sampler that
guessed at history would be untestable against upstream's ordering. And the
penalty direction is upstream's, not the obvious one — a **positive** logit is
divided by the repetition penalty and a **negative** one multiplied, so the
penalty always moves a repeated token *down*.

Evidence: `sampling_bias` (20 checks). Six mutations each turn it red —
min-p truncation removed, repetition direction inverted, presence applied per
occurrence instead of once, logit bias ignored, `n_probs` unreported, penalty
window ignored. End to end on a fixture, `--repetition-penalty 4` turns the
greedy loop `65 65 65 65 …` into `65 93 230 16 …`, and `--n-probs 3` returns
three normalised, descending alternatives per token.

**Not** implemented, and not claimed: DRY sampling, grammar constraints,
typical-p, XTC and `xtc_probability`. `n_probs` reports the post-truncation
distribution, so it cannot describe the full softmax the way llama.cpp's
`n_probs` does when truncation is off.

### C7. ~~Quantisation coverage has holes~~ — **FIXED at `f704e4b`; verified at `main`**

> **FIXED.** The IQ family and both FP4 formats decode and execute, matched to
> llama.cpp's own dequantizers (`tools/quant_upstream_check.py`). The body below
> is the baseline record.

Decodable: F32/F16/BF16, Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1, Q2_K–Q6_K, Q8_K,
IQ4_NL, TQ1_0/TQ2_0, Q1_0/Q2_0/PQ2_0, I8/I16/I32/I64/F64.

Missing and used by published files: **IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S,
IQ4_XS** (the small-MoE quants that make big mixes fit), **MXFP4 / NVFP4**
(gpt-oss, minimax), **I2_S** (bitnet). Each is refused with a typed error —
correct behaviour, missing capability.

### C8. Per-expert tensor layouts

Upstream still writes `blk.N.ffn_gate.E.weight` for some converted files; this
engine requires the stacked layout and fails otherwise. Recorded in
`docs/MODEL_SUPPORT.md:330`.

### C9. Activation and attention variants — **FIXED (all but SITU); verified at `main`**

> **FIXED** for `attn_sinks`, `SWIGLU_OAI_MOE` and `GEGLU_ERF`. `SITU` is an
> `GGML_ABORT` upstream, so refusing it matches the reference. The body below is
> the baseline record.

No `attn_sinks`, no `SWIGLU_OAI_MOE` (clamped SwiGLU), no `SITU`, no
`GEGLU_ERF`. Small self-contained kernels — but they are what `gpt-oss`,
`minimax-*`, `deepseek4/41` and `kimi-k3` need.

### C11. Architecture census: 171 upstream families, 48 mapped — the rest need named work

`tools/triage_upstream_archs.py` classifies every `src/models/*.cpp` in a
llama.cpp checkout by what its graph *calls*, not by its tensor names: 171 files,
81 whose graph body is a llama-shaped block and 90 that build something else
(mamba, delta-net, AltUp, MTP, vision towers). A marker scan is only a first
pass, so each candidate then went through three further checks, and a row was
added only when all three passed:

1. **Tensor roles.** Every `LLM_TENSOR_*` the architecture's file names is
   resolved through `LLM_TENSOR_NAMES` and looked up in `role_from_name`. A
   single unresolved name means the file would load and then fail on a tensor
   the engine never asked for — the "name-only row" this repo refuses.
2. **Graph read.** The block body was read for norm placement, activation,
   rotary and residual structure, because tensor names do not distinguish a
   weightless LayerNorm from a weighted one (`olmo2`, `arcee`).
3. **Parity.** A fixture shaped like that family's checkpoint and an oracle
   golden; the engine must match it, or the row does not stay.

Fourteen rows survived (see `docs/MODEL_COVERAGE_STATUS.md`); this is what the
other candidates are blocked on, in the terms of the file that blocks them:

| Blocking fact | Architectures | What it needs |
| --- | --- | --- |
| `ggml_xielu` with per-layer `alpha_n/alpha_p/beta/eps` | `apertus` | A new activation with four per-layer parameters |
| Attention output gate (`attn_gate`), applied before the output projection | `muse-glimmer`, `spark2_5`, `hrm-text` | A tensor role and one multiply; `hrm-text` also needs `hrm.z_l_init` |
| Post-norm-only block (no pre-norm tensors at all) | `olmo2`, `exaone4`, `seed_oss` | A block whose residual additions run before the norms; the current post-norm support assumes both |
| Fused qkv plus per-layer head counts / widths | `openelm`, `plamo3` | Per-layer attention geometry; the manifest currently holds one head count |
| MLA variant with a single `attn_kv_b` (no `attn_k_b`/`attn_v_b`) | `minicpm3`, `plm`-adjacent | A second MLA layout |
| Dual dense+MoE FFN in one layer (both branches summed) | `arctic`, `ernie4_5`, `granite-moe` | A block that runs both, as `arctic` does |
| No kv cache at all (diffusion / embedding decoders) | `dream`, `llada`, `eurobert` | A caching policy, not a tensor |
| Encoder-only with `cls.*`/`token_types`/pooling | `modern-bert`, `neo-bert`, `jina-bert-v2/v3`, `nomic-bert*`, `gemma-embedding*` | A pooling head and a segment embedding |
| ALiBi instead of rotary | ~~`bloom`, `jais`~~ **implemented**; ~~`falcon`, `mpt`~~ **now mapped** — `falcon` landed with `attn_norm_2`, and `mpt`/`gpt2`/`starcoder` with the position table | Their tensor sets, not the bias. What `mpt` still skips is the optional AWQ `ffn.act` scale tensor and the optional full-width q/k LayerNorm, both absent from the fixture and from stock checkpoints |
| ~~Weightless norms~~ **implemented** | ~~`olmo`, `talkie`'s layer norms~~ **mapped for olmo** | `build_norm(x, NULL, NULL, ...)`: no scale tensor at all. talkie still needs its residual-skip stream |
| ~~Per-head LayerNorm q/k norm~~ **implemented** | ~~`command-r`, `chameleon`~~ **mapped for command-r** | one row per head, mean-centred, with the file's own epsilon. chameleon additionally switches its residual structure (`swin_norm`) and suppresses image-token logits, neither of which is modelled |
| ~~Output logit scale~~ **implemented** | ~~`command-r`~~ **mapped** | granite records the same key and never applies it, so the application is per architecture |
| ~~Absolute position embeddings~~ **implemented** | ~~`gpt2`, `starcoder`, `mpt` (`position_embd`)~~ **mapped**: tensor role `POSITION_EMBD`, one row lookup added to the embedding, no rotation. Dropping the lookup fails `gpt2`/`starcoder`/`mpt` greedies | Done |
| A second pre-attention norm (`attn_norm_2`) | ~~`falcon`~~ **mapped**; the same shape is what falcon-7B omits (falls back to `attn_norm`) | Done; the bias of the second norm is decoded like every other norm bias |
| T5 encoder/decoder with relative-position buckets | `t5` | Two graphs and a bucket bias |

**Fixed alongside:** llama.cpp renamed the sandwich-norm tensors
(`attn_post_norm` → `post_attention_norm`, `ffn_post_norm` → `post_ffw_norm`).
A checkpoint converted today therefore failed to load in a build that claimed
`gemma2`, `gemma3` or `glm4`. The loader now normalises both spellings to one
name at parse time (`canonical_layer_name`, `src/gguf.cpp`), the oracle does the
same independently, and a `gemma2-renamed` fixture — byte-identical arithmetic,
current names — holds a parity row. Mutating the rename table makes that row
fail, so the alias cannot be dropped silently.

---

### C11a. ALiBi and the embedding norm — **FIXED**; both are executed arithmetic now

Upstream computes the ALiBi bias in the *mask tensor*: `-|query - key|`
(`src/llama-graph.cpp:511`), which `soft_max_ext` multiplies by a per-head slope
derived from `max_bias` in `ggml_compute_forward_soft_max_f32`
(`ggml/src/ggml-cpu/ops.cpp:5831-5850`):

```
n_head_log2 = 2^floor(log2(n_head));  m0 = 2^(-max_bias/n_head_log2);
m1 = 2^(-max_bias/2/n_head_log2)
slope(h) = h < n_head_log2 ? m0^(h+1) : m1^(2(h-n_head_log2)+1)
```

The engine computes the slopes once per sequence and adds `slope(h) * -(query -
key)` to the scaled score before the softmax, and the oracle derives the same
slopes independently. `bloom` hardcodes `max_bias = 8.0` in its own
`load_arch_hparams` (upstream's comment calls making it metadata a TODO), so a
bloom file is biased whether or not it carries the key; `jais` reads the key and
rotates nothing at all.

`bloom` also normalises the token embedding once before block 0 with a biased
LayerNorm (`token_embd_norm`), which is a norm site no other mapped family has:
it carries no `blk.N.` prefix, so it cannot ride on the block's norm path. Both
halves are required, as upstream's two `create_tensor` calls demand.

Evidence: `bloom` and `jais` fixtures, oracle goldens, parity rows. Mutating the
bias away fails by 0.077 (bloom) and 0.444 (jais); mutating the embedding norm
away fails bloom by 1.440.

### C10. ~~Benchmarks run on an ISA the target CPU does not have~~ — **FIXED (code); re-run still owed**

The knob exists now: `--kernels auto|scalar|avx2|zen3|avx512` selects the CPU
backend's ISA, `resolve_kernel_isa` applies the Zen 3 policy (`zen3` and `avx2`
pin AVX2/FMA3, and `auto` **never** selects AVX-512 because the target cannot
execute it), and each backend operation reports the ISA it actually ran
(`record_operation(..., "cpu-" + kernel_isa_name(...))`). `device-info` prints
`resolved_isa`. Pinned by the `kernel_isa` checks: scalar stays scalar, `zen3`
parses to `avx2_fma`, an unknown profile is refused.

What is still owed is the **measurement**, not the code: the benchmark reports
in `results/` were produced before the knob, and re-running
`benchmarks/run_repeats.sh` with `--kernels zen3` on this host is what turns
"the numbers describe the target" from a claim into a record. Until that run is
pasted here, treat the AVX-512-dispatched figures as historical.

---

## D. Documentation drift

| Document | Claim | Reality |
| --- | --- | --- |
| `docs/ROADMAP_INFERENCE_AND_TRAINING.md:123` | "I1 … Exists: one synthetic adapter, MHA, separate output weight, F32. Missing: any real architecture." | 22 adapters, GQA/MQA/MLA, 26 quant types |
| `docs/ROADMAP_INFERENCE_AND_TRAINING.md:101` | "I0 … missing: separation between container parsing and architecture semantics" | Still true: `src/gguf.cpp` is a 2191-line file mixing container, capability and adapters |
| `docs/SUPPORTED_MODELS.md:51` | "## Explicitly unsupported: arbitrary Llama, Mixtral, DeepSeek, Qwen, Gemma …" | Scope note exists but is weak; this section describes the *native* adapter only and reads as a global claim |
| `docs/TEST_MACHINE.md:13` | "AVX-512 disabled for FreeMoe backend tests" | Not enforced (C10) |
| `docs/ARCHITECTURE.md` module inventory | every file under `src/` is listed | Was **not** true after PR #10: `src/chat_template.cpp` was missing, so `scripts/check_architecture.py` — and therefore `make test` — failed on `main`. Fixed here; `make test` now exits 0. |
| `docs/ARCHITECTURE.md:53` | `src/tokenizer.cpp` provides "GPT-2 style BPE" | It provides GPT-2 style *piece tables*; the merge algorithm is missing (C1) |

---

## E. What to do, in order

Ranked by (models unlocked) ÷ (new concepts introduced). Nothing here requires
fabricating a capability.

| # | Work | Why first | Size |
| --- | --- | --- | --- |
| 1 | **BPE correctness** (C1) | The only gap that produces wrong output *today* on models that already "work" | M |
| 2 | **56 adapter rows** | Zero engine work: table entry + fixture + golden per family, the pattern PR #10 established | M (breadth) |
| 3 | **YaRN / `rope_freqs`** (C4) | +14 models, one rotary change, already isolated behind `rope_dim` | S |
| 4 | **CPU ISA override + AVX2-only re-benchmark** (C10) | Today's only performance evidence is from an ISA the target lacks | S |
| 5 | **Eviction-exercising benchmark config** (§A Phase 5/6) | Without it the cache-policy and prefetch gates cannot be evaluated at all | S |
| 6 | ~~**Batched prefill** (C2)~~ **done** (2.72-2.94x measured) | Largest throughput win at prompt length > 32; makes §10.2 metrics honest | L |
| 7 | ~~**IQ1–IQ4 / MXFP4 decoders** (C7)~~ **done** | Unlocks quantised checkpoints of MoE families that are already supported | M |
| 8 | ~~**GEGLU_ERF**~~ **done**; **SITU** (C9 remainder) | `gelu_erf` is reachable through `mistral3`; `SITU` is unimplemented upstream too (`GGML_ABORT`), so no file can be checked against it | S |
| 9 | ~~**mamba/mamba2**~~ **done**; **gated delta-net** | The mamba scan, its state layout and the `dt_b_c_rms` norms now run with fixtures and goldens. Delta-net is a different scan (per-head matrix state) and is what `qwen35`, `qwen3next` and `kimi-linear` still need | L |
| 10 | **Sampling** (C6), **SWA cache** (C3), **per-expert tensors** (C8), **deepseek41/clamp rule** (C5) | Completeness of things already claimed | S–M |
| 11 | **GPU backend** (§A Phase 2) | Required for every VRAM gate; unstarted | XL |
| 12 | **Serving layer**, speculative decoding, batching | Post-correctness | XL |

Anti-goals, per spec §17/§18 and the project rules: do not claim a model runs
without executing it; do not ship a 169-row adapter table over files whose
primitives are missing; do not declare a tokens/s target to make a gate pass;
do not present the synthetic-fixture benchmark as a performance qualification.

---

## F. What this analysis cannot tell you

* **No vendor weights are present.** `google_drive/` is not in this workspace,
  so nothing here is verified against a real DeepSeek, Qwen, Mistral or Llama
  checkpoint. Everything is synthetic fixtures plus an oracle written from the
  same specification — that catches implementation bugs, not misread
  specifications. Section B's per-model claims are static signatures, not runs.
* **No target-hardware numbers.** The host is an Intel Xeon KVM guest with
  2 vCPUs; Zen 3 has never run this code. The GPU path does not exist.
* **Upstream is a moving target.** The census is `master` on 2026-10-28;
  upstream adds model files monthly, so "169" has a shelf life.

---

## Appendix: reproduce

```sh
make -j2 test                                   # 1338 checks
./build/tests architecture                      # 17-architecture parity
./build/freemoe probe --model <file.gguf>       # per-file capability report
bash benchmarks/run_repeats.sh configs/benchmark.env   # the numbers in §Baseline
python3 tools/make_arch_fixtures.py --check     # fixtures up to date

# census (needs the upstream tree)
curl -sSL https://codeload.github.com/ggml-org/llama.cpp/tar.gz/refs/heads/master \
  | tar -xz --wildcards 'llama.cpp-master/src/*'
ls llama.cpp-master/src/models/*.cpp | wc -l    # 169
```
