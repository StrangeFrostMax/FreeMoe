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
| Benchmark vs declared targets | **pass** on the synthetic fixture with `--kernels zen3`: 637,100 of 637,100 compute operations dispatched `cpu-avx2_fma` and none on AVX-512 (C10, `docs/evidence/benchmark_zen3.json`) |

### The one benchmark that exists

`bash benchmarks/run_repeats.sh configs/benchmark.env` → `results/benchmark/synthetic-moe-buffered/`
(git-ignored), 5 process-cold repetitions, 20 prompts × 16 tokens, model
`tests/fixtures/moe-fm.gguf` (native 2-layer fixture, vocab 258), buffered
reader, `CACHE_SLOTS=1`, **`--kernels zen3`** (recorded in the run's
`protocol.txt`, which now carries `extra_args`):

<!-- benchmark-metrics:begin (generated by scripts/benchmark_table.py; do not edit by hand) -->

Record: commit `8384bff`, requested profile `zen3`, dispatch `cpu-avx2_fma` 637,100, `scalar-control` 638,705, archive status `pass`.

| Metric | Median | Range | Declared target | Verdict |
| --- | --- | --- | --- | --- |
| Prefill | 4,329 tok/s | 4,206–4,375 | ≥ 1,000 | pass |
| Decode | 4,048 tok/s | 3,960–4,078 | — | — |
| p95 inter-token | 0.312 ms | 0.298–0.329 | ≤ 50 ms | pass |
| Median inter-token | 0.242 ms | 0.242–0.246 | — | — |
| First token | 60.3 ms | 59.7–62.1 | — | — |
| Peak process RAM | 251.7 MB | ±84 kB | — | — |
| Reference comparison | 20/20 prompts, 320/320 tokens, 0 routing mismatches, max rel diff 1.4e-3 | — | 20/20 | pass |

<!-- benchmark-metrics:end -->

The ISA is now part of the record rather than an assumption. Every one of the
637,100 compute operations in the five reports names `cpu-avx2_fma`; there is no
`cpu-avx512` operation anywhere in the logs. `python3 scripts/archive_benchmark.py`
re-derives that from the reports into `docs/evidence/benchmark_zen3.json` and
**exits non-zero** if a pinned profile did not dispatch, so a future run cannot
quietly fall back to AVX-512 and still be archived as the target's numbers.

The record is `docs/evidence/benchmark_zen3.json`, written by
`scripts/archive_benchmark.py` from the run directory and the
`tools/bench_prefill.py` output; it carries the commit it was taken at. The
table above is rendered from that file (`scripts/benchmark_table.py`, `--check`
in `make test`), so no figure in it is retyped.

Three records of the *identical* configuration now exist on this host, all with
`--kernels zen3` and no engine or kernel change between them:

| Record | Prefill (median) | p95 inter-token | First token |
| --- | --- | --- | --- |
| `31288ab` (git history) | 3692 tok/s | 0.351 ms | 70.7 ms |
| `11339a4` (git history) | 3144 tok/s | 0.414 ms | 83.0 ms |
| `8384bff` — the table above | 4329 tok/s | 0.312 ms | 60.3 ms |

That is a 38% spread between the fastest and slowest record of the same binary on
the same 2-vCPU guest, in both directions. None of it is a regression and none of
it is an improvement: between `31288ab` and `8384bff` the engine gained an adapter
rule for MTP heads and nothing the MoE fixture's path executes. The committed
archive named the `11339a4` run until this batch, while this table named
`8384bff`; both now name the run the record was actually taken at. The absolute
number is a property of the *record*, which is why each one carries its commit,
why the table above is rendered from the archive rather than retyped, and why
`docs/KERNELS.md` refuses to compare numbers across records. The declared targets
pass in all three; that is the part the spread does not change.

Three things this does **not** establish, all noted in
`docs/BENCHMARK_PROTOCOL.md`:

* the model is a 2-layer synthetic fixture, so the prefill figure says nothing
  about a real MoE checkpoint, and `cache_hit_rate` is 0.0 because there is one slot and
  no eviction, so it says nothing about disk paging either;
* the host is a 2-vCPU KVM guest on an Intel Xeon that *does* have AVX-512. The
  code path measured is the target's (AVX2 + FMA3), but the microarchitecture is
  not a Zen 3 part, so these are not target cycle counts — D1's exact machine is
  still unknown;
* the AVX-512-dispatched figures that stood here previously (3051 tok/s, p95
  0.41 ms, first token 85.5 ms) were a measurement of a path the target cannot
  execute; they are replaced, not compared. The two records cannot be read as an
  ISA comparison in either direction: the code changed between them (C2's
  prefill ordering, C3's cache), and a 2-vCPU guest has enough variance that a
  single pair of medians would not separate the causes.

---

## A. Design → code

The spec's own gates in order. "Met" means the spec's stated exit condition,
not "some code exists".

| Phase | Spec gate | State | Evidence / what is missing |
| --- | --- | --- | --- |
| 0 — repo / dependency audit | clean build on the declared target | **MET** | `make -j2`, `docs/PHASE0_EXIT.md`, `scripts/collect_evidence.py`. |
| 1 — GGUF inspection + dense CPU correctness | malformed models fail safely, tensor roles validate, output matches reference | **MET** | 1338 checks, 30 fixtures in `tests/malformed_models/`, 17-architecture parity against an independent oracle (`tools/llama_oracle.py`). |
| 2 — dense GPU baseline | no silent fallback, stable output, bounded allocations | **NOT MET** | `src/backend.cpp:545` is a placeholder raising `GPU_BACKEND_UNIMPLEMENTED`; no CUDA/Vulkan/HIP code exists. It refuses rather than faking it — correct, but the phase is open and every VRAM gate with it. |
| 3 — one MoE family, resident experts | selected experts never dropped, routing matches reference | **MET for stacked and per-expert MoE** | `ResidencyManager::assert_selected_executed` (`src/residency.cpp:785`) is called after every MoE layer (`src/engine.cpp:1070`) and fails if a selected expert never ran; `qwen2moe`, `qwen3moe`, `deepseek2` parity; 0 routing mismatches in the benchmark report. Per-expert tensor layouts load and run through the same records (C8). |
| 4 — VRAM expert slot pool | stress tests with forced eviction, no stale reads | **MET in host RAM, structurally blocked in VRAM** | Slot state machine with generation ids and publish-after-fence (`src/residency.cpp:59–77`), LRU + clock (`src/residency.cpp:385`), checksum verification, stale-completion rejection. All tested. The pool lives in host memory because Phase 2 is unstarted, so the *VRAM* half of the gate cannot be met. |
| 5 — trace + cache policy | policy beats LRU/clock on decode p95 across repeat runs | **PARTIAL** | Trace events (`RoutingEvent`, `--trace`), both baselines implemented, repeat harness exists — but the committed config runs one slot, so **no replacement happens and no policy can be compared**. The gate needs an eviction-exercising config (`CACHE_SLOTS` < working set). |
| 6 — NVMe-backed expert delivery | cold-cache correctness, bounded queues, meets target | **PARTIAL** | `pread` / `mmap` / `O_DIRECT` readers (`src/io.cpp:167`), extent coalescing (`coalesce_extents`, `src/io.cpp:46`), prefetch issue/poll/cancel. No cold/warm suite, no working-set instrumentation, and the only measured run is buffered with one slot. |
| 7 — optional layout optimisation | A/B proves end-to-end gain | **NOT STARTED** (correctly gated) | Blocked on Phase 6 evidence, as the spec requires. |
| 8 — advanced optimisations | each a separate experiment | **NOT STARTED** | No speculative decoding, no batching, no multi-GPU, no dequantisation cache, no fused expert kernels. |

### The performance gate

Targets *are* declared (`docs/BENCHMARK_PROTOCOL.md:54`, provisional, sandbox
scoped: prefill ≥ 1000 tok/s, p95 inter-token ≤ 50 ms). They pass — see the
table above. But the same paragraph pins the release gate to a real D1 machine
record, and the machine here is an Intel Xeon KVM guest with 2 vCPUs — running
the AVX2+FMA3 path the Zen 3 target executes, on silicon the target is not. So:
**target edges met on a synthetic fixture; the "performance-qualified" label in
spec §16 remains unearned**, and earning it now requires (a) a config that
exercises eviction, (b) a Zen 3 part, (c) a non-synthetic model. (b) used to read
"an AVX2-only path"; that half is done — see C10.

### Spec sections only partly honoured

| Spec § | Requirement | Gap |
| --- | --- | --- |
| §7.2 / §7.3 | stacked **and** per-expert tensor layouts | **Both.** Stacked (`ffn_*_exps` + `ffn_gate_inp`) and per-expert (`ffn_gate.E.weight`) are accepted per layer; a layer carrying both is refused as ambiguous. |
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

The partition is not written here. It is `docs/evidence/arch_census.json`,
rendered into §C11 by `tools/triage_upstream_archs.py` from a **pinned** llama.cpp
tree (`0ec4027a2a01…`): **171** `src/models/*.cpp`, of which **61** have an
adapter row and a graph that names nothing this build lacks, **2** have a row
and also build a block the row does not model, **43** are blocked by a marker
for a capability in the table below, and **65** need a graph read before
anything at all is claimed.

An earlier revision partitioned the 110 unmapped files by hand into
"not a causal text decoder / recurrent, linear attention or MTP / a different
causal block" (24 / 30 / 56) and its own first version was off by two files. A
hand-kept count with no artefact behind it is how the previous 81/90 split
survived a whole review, so it is gone. `blocked` is now the only bucket that
assigns a cause, because it is the only one that can point at a marker in the
file; everything else has to be read.

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
> rows (`include/fm/kernels.hpp`), each row's key/value is written and attended
> to before the next row is written (order matters for C3's ring, see below),
> and logits are computed for the last row only. `run_prompt`
> takes that path whenever `supports_batched_prefill()` accepts the block, and
> `--prefill sequential` (`BackendConfig::batched_prefill = false`) keeps the
> token-at-a-time path for comparison. Both paths are pinned against each other
> by `batched_prefill` (76 checks) — identical tokenisation, logits and greedy
> ids — and the seven block features the batch path does not model yet are
> pinned by name, so a family cannot silently take a path that ignores its
> detail.
>
> Measured here on a 2-vCPU Xeon sandbox, `--prefill` toggled; the same run is
> archived as `prefill_bench` in `docs/evidence/benchmark_zen3.json`:
>
<!-- prefill-table:begin (generated by scripts/benchmark_table.py; do not edit by hand) -->
> 
> `tools/bench_prefill.py --kernels zen3`, hidden 512 / intermediate 2048 / 8 layers, median of 3 per mode:
> 
> | prompt | sequential | batched | speedup |
> | --- | --- | --- | --- |
> | 185 tokens (x4) | 2264.0 ms | 488.2 ms | **4.64x** |
> | 369 tokens (x8) | 4617.5 ms | 1608.6 ms | **2.87x** |
> 
<!-- prefill-table:end -->
>
> The batched half of that ratio is the more stable one — within about a fifth
> across sessions — while the sequential half moves by up to 40% on this 2-vCPU guest
> (records in git history). The ratio therefore belongs to the run that is
> archived, with the commit it was taken at, and not to the code in general.
> These are AVX2 numbers: the 2.94x / 2.72x this section used to carry came from
> runs that dispatched AVX-512 (C10).
>
> The original finding follows. `run_prompt` fed prompt tokens one at a time.
> Correct, but it cost N forward passes instead of one GEMM-heavy pass, and it
> made the §10.2 prefill metrics meaningless (see §A).

### C3. ~~Sliding-window models reserve and keep the full cache~~ — **FIXED**

> **FIXED.** A windowed layer now reserves and addresses a ring of
> `min(context, sliding_window)` positions instead of the whole context, from one
> helper (`ModelConfig::cache_positions`, `cache_index`,
> `attention_cache_floats` in `include/fm/gguf.hpp`). `compute_budget`
> (`src/residency.cpp`) multiplies that function by `sizeof(float)`, so the
> reservation and the address cannot drift apart. Full-attention layers keep the
> context, so a per-layer pattern (gpt-oss style) pays the window only where the
> window is used.
>
> Both attention paths use the ring. The batched prefill pass used to write the
> whole batch before reading it; on a 4-position ring that evicted history the
> same pass still had to attend to, and `architecture_parity` caught it
> (`gemma2` greedy diverged at the second generated token). The pass now writes
> each row's key/value and attends to it before moving on, so batch length no
> longer interacts with ring size.
>
> Verified here: `architecture_parity` 490/490, full suite 3722/3722,
> `./build/freemoe run --model tests/fixtures/gemma2-shape.gguf --prompt "the cat"
> --tokens 8` → `[57,108,224,195,139,162,191,191]`, the golden line.
> `sliding_window_cache` (25 checks) pins the arithmetic: reservation is one ring
> per windowed layer, `cache_index(layer, capacity) == cache_index(layer, 0)`,
> layer regions do not overlap, the budget equals the reservation, and the mixed
> pattern keeps context on full-attention layers. Mutations (each rebuilt,
> each red): capacity ignoring the window → 6 failures; address using the
> position instead of the ring slot → the wrap checks fail; layer offset using
> the context → the region check fails; batched write using linear addressing →
> `architecture_parity` segfaults out of bounds; batched attention ignoring
> `first_visible` → 2 `gemma2` parity failures; the zero-context refusal removed
> → the modulus guard fails.
>
> Both paths are also pinned against each other on the windowed fixture with a
> prompt longer than the window: `sliding_window_cache` runs `gemma2-shape` with
> `batched_prefill` on and off and requires the same eight tokens and the same
> prefill logits.
>
> Not verified: a real gemma-2/gemma-3 checkpoint — the fixtures carry the shape
> and the metadata, not the weights — and the ring with anything other than an
> F32 cache, because this build has no quantised kv cache to reserve.

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

### C8. ~~Per-expert tensor layouts~~ — **FIXED**

> **FIXED.** A routed layer may now store one tensor per (expert, projection)
> — `blk.N.ffn_gate.E.weight`, `ffn_up.E.weight`, `ffn_down.E.weight` — as an
> alternative to one stacked tensor per (layer, projection). The layout is
> decided per layer from the tensor names; a layer carrying both spellings is
> refused as ambiguous, a layer carrying only some of the three projections is
> refused as incomplete, and a per-expert name on a layer that is not
> per-expert is refused rather than ignored.
>
> The reader no longer looks a tensor name up at all. `ExpertExtent` now carries
> the projection's stored type alongside its offset and length, so one expert
> record describes a stacked slice and a whole per-expert tensor in exactly the
> same way and `ResidencyManager` concatenates the three extents in role order
> for both. That was the property missing before: the engine read gate/up byte
> counts from the stacked tensor's name, which a per-expert file does not have.
>
> Verified here: `qwen3moe-per-expert` is the `qwen3moe` fixture with the same
> weights stored per expert (the generator slices the stacked tensors, so the
> two files hold identical numbers). Its oracle golden is identical to
> `qwen3moe`'s apart from the fixture line, and the engine's prefill logits and
> eight greedy tokens match that golden through the per-expert file.
> `per_expert_layout` (31 checks) pins the record arithmetic: one record per
> (layer, expert), payload = three whole tensors, distinct offsets, the type on
> the extent, expert bytes excluded from fixed weights, both refusals, the probe
> tag `moe=per-expert`, and the logits above. Mutations (each rebuilt, each
> red): layout never detected → the file is refused as an unknown tensor;
> stacked stride arithmetic applied to a whole tensor → payload and slot
> assertions plus a reader length error; ambiguity refusal removed → the
> mixed-layout case loads; expert tensors marked by the stacked name list only
> → expert bytes charged as fixed weights; payload omitting the down projection
> → four record assertions.
>
> Not verified: a real vendor per-expert checkpoint (the fixture is synthetic and
> F32), a per-expert file with mixed quantisation types, and per-expert
> projection **biases** — those stay stacked-only, and a file that carries them
> with a per-expert layout fails closed on the unexpected name.

### C9. Activation and attention variants — **FIXED (all but SITU); verified at `main`**

> **FIXED** for `attn_sinks`, `SWIGLU_OAI_MOE` and `GEGLU_ERF`. `SITU` is an
> `GGML_ABORT` upstream, so refusing it matches the reference. The body below is
> the baseline record.

No `attn_sinks`, no `SWIGLU_OAI_MOE` (clamped SwiGLU), no `SITU`, no
`GEGLU_ERF`. Small self-contained kernels — but they are what `gpt-oss`,
`minimax-*`, `deepseek4/41` and `kimi-k3` need.

### C11. Architecture census — every upstream family classified, the rest need named work

`tools/triage_upstream_archs.py` classifies every `src/models/*.cpp` in a pinned
llama.cpp checkout by what its graph *calls*, not by its tensor names. The
artefact is committed (`docs/evidence/arch_census.json`), the table below is
rendered from it, and `--check` re-derives both from the pinned tree.

The numbers this section used to quote — "81 llama-shaped, 90 need a block" —
were wrong in both directions, because the tool's capability list had drifted
from the engine: it treated `build_moe_ffn(` as a missing block, so `llama`,
`granite`, `phi3`, `qwen3moe` and twelve other **mapped** families were counted
as blocked, while `t5`, `llada` and the BERT/embedding encoders were counted as
one-row-away because their files name no primitive the engine lacks — their
difference is structural. A census whose capability list is not derived from the
engine measures the list, not the engine.

The second drift ran the other way and hid a half-implemented rule. `C-MTP-HEAD`
was listed as a blocker for 17 families, but a head is not something a decoder
runs: upstream builds it only when the loader asks for it (`ml.load_mtp`, false
in every inference run), so a checkpoint carrying one is a checkpoint whose trunk
runs unchanged. Removing it from the capability list was right; the rule behind
that claim was then only half true.

Upstream stores the head *inside* the block count. `llama-model.cpp` reads
`block_count` into `n_layer_all` and runs `n_layer_all - n_layer_nextn` trunk
layers, so a head layer's weights are ordinary `blk.{n_layer}.*` tensors — the
same names a trunk layer has — plus the `nextn.*` ones the head owns. A rule that
ignores names is therefore not enough: a real GLM-4.5-Air-class file would have
been refused on `blk.31.attn_q.weight`. The fixture now carries both halves
(`block_count` 3, `nextn_predict_layers` 1, a full head block plus the `nextn.*`
names, and a per-layer `attention.head_count_kv` array sized by the block count,
which is how llama.cpp writes per-layer arrays). The engine subtracts the
declared head layers, ignores exactly those indices, and truncates a per-layer
array that is sized by the block count instead of the trunk; the reference
implements the same subtraction independently.

`mtp_head_tolerated` (66 checks) holds all of it: every head tensor is reported
in `inspect` with its reason, logits and greedy ids are **bit-identical** to the
same weights without a head, and four counterfactuals keep the rule from growing
— dropping the key makes the head a runnable third layer, an index past the head
is `UNKNOWN_TENSOR`, a head larger than the block count is `METADATA_MISMATCH`,
and a head tensor renamed out of the block namespace is still refused. Those
families did not become supported: they moved from "blocked by MTP" to
`unclassified`, where the rest of their block has to be read. The loader also
tolerates an `mtp.` prefix that no upstream file at this pin writes; it is kept
for third-party converters and no fixture exercises it, which is stated here
rather than counted as coverage.

<!-- arch-census:begin (generated by tools/triage_upstream_archs.py; do not edit by hand) -->

Upstream `ggml-org/llama.cpp` at `0ec4027a2a01…`, `src/models/*.cpp`: **171 families**.
This build has **75 adapter rows**. The partition:

| Verdict | Families | Meaning |
|---|---:|---|
| mapped | 69 | adapter row, and the graph names nothing this build lacks |
| mapped, with a caveat | 2 | adapter row, but the file also builds a block the row does not model |
| blocked | 36 | no row; the graph calls a primitive this build has no implementation of |
| unclassified | 64 | no row and no marker hit — needs the graph read before a row can be claimed |

`unclassified` is not a polite word for "almost supported". Upstream can differ
structurally without naming a missing primitive (an encoder stack, a pooling
head, post-norm placement), so a family only leaves that bucket once its block
has been read and it has either a row with a fixture and an oracle golden, or a
named capability.

The unclassified rows, ordered by how much of their call set is already in a
mapped family. Overlap is a Jaccard score over the `build_*`/`ggml_*` calls in
the file — a hint about which graph to read first, **not** a verdict: two files
can make the same calls and wire them differently. The last column is what the
read has to account for.

| Family | Closest mapped | Overlap | Calls the read must explain |
|---|---|---:|---|
| `deepseek2ocr` | `phimoe` | 1.00 | — |
| `ernie4-5` | `xverse` | 1.00 | — |
| `hunyuan-moe` | `ernie4-5-moe` | 1.00 | — |
| `hunyuan-vl` | `glm4` | 1.00 | — |
| `jina-bert-v2` | `phimoe` | 1.00 | — |
| `jina-bert-v3` | `phimoe` | 1.00 | — |
| `maple` | `openai-moe` | 1.00 | — |
| `mistral4` | `phimoe` | 1.00 | — |
| `nomic-bert` | `phimoe` | 1.00 | — |
| `nomic-bert-moe` | `phimoe` | 1.00 | — |
| `pangu-embed` | `xverse` | 1.00 | — |
| `qwen2vl` | `paddleocr` | 1.00 | — |
| `refact` | `jais` | 1.00 | — |
| `minicpm3` | `plm` | 0.95 | — |
| `exaone4` | `olmo2` | 0.93 | — |
| `talkie` | `phi2` | 0.93 | — |
| `exaone-moe` | `openai-moe` | 0.93 | — |
| `flint` | `openai-moe` | 0.93 | — |
| `grovemoe` | `qwen3moe` | 0.93 | — |
| `maion-coder` | `openai-moe` | 0.93 | — |
| `granite-swa` | `granite` | 0.90 | — |
| `grok` | `phi3` | 0.88 | — |
| `apertus` | `xverse` | 0.86 | `ggml_xielu` |
| `dream` | `xverse` | 0.86 | — |
| `llada` | `xverse` | 0.86 | — |
| `llada-moe` | `qwen3moe` | 0.86 | — |
| `rnd1` | `qwen3moe` | 0.86 | — |
| `afmoe` | `phi3` | 0.83 | `ggml_sigmoid` |
| `kolibri1` | `openai-moe` | 0.81 | `ggml_argsort_top_k` |
| `openelm` | `olmo2` | 0.81 | — |
| `falcon-h1` | `xverse` | 0.80 | — |
| `gemma-embedding` | `phi2` | 0.80 | — |
| `granite-hybrid` | `granite` | 0.80 | — |
| `llama4` | `mistral3` | 0.80 | — |
| `laguna` | `olmo2` | 0.79 | `ggml_softplus` |
| `muse-glimmer` | `gemma2` | 0.79 | `ggml_sigmoid` |
| `spark2-5` | `olmo2` | 0.78 | `ggml_sigmoid` |
| `chameleon` | `xverse` | 0.77 | `ggml_clamp`, `ggml_element_size`, `ggml_set_1d` |
| `plamo3` | `olmo2` | 0.77 | `ggml_element_size` |
| `hy-v3` | `mistral3` | 0.76 | — |
| `beam` | `gemma2` | 0.71 | `ggml_sigmoid` |
| `eurobert` | `xverse` | 0.71 | — |
| `neo-bert` | `xverse` | 0.71 | — |
| `k2-horizon` | `phi3` | 0.70 | `build_routed_value`, `ggml_softplus` |
| `mellum` | `smallthinker` | 0.67 | `build_inp_embd_h`, `ggml_build_forward_select` |
| `nemotron-h` | `jamba` | 0.67 | `build_ffn_layer` |
| `cohere2moe` | `phi3` | 0.65 | `build_inp_embd_h`, `ggml_build_forward_select` |
| `step35` | `mistral3` | 0.64 | `ggml_sigmoid` |
| `muse-spark` | `gemma2` | 0.64 | `ggml_sigmoid` |
| `bert` | `llama` | 0.60 | `ggml_build_forward_order` |
| `cogvlm` | `starcoder` | 0.60 | `ggml_element_size`, `ggml_rope` |

The other 13 unclassified rows share no mapped shape above 0.6 and need a read from the top: `gemma-embedding2`, `gemma4`, `gemma4-assistant`, `granite-switch`, `hrm-text`, `mimo2`, `minimax-01`, `modern-bert`, `nemotron-h-moe`, `qwen3tts`, `sensenova-u1`, `t5`, `vllm-decision2`.

| Capability | Families | Of which |
|---|---:|---|
| `C-ALTUP` — AltUp / per-layer token embeddings | 1 | `gemma3n` |
| `C-DSA-INDEXER` — sparse-attention (DSA) indexer | 6 | `deepseek32`, `dots3note`, `glm-dsa`, `hy-v4`, `minimax-m3`, `mistral-large4` |
| `C-DUAL-FFN` — dense and MoE FFN summed in one layer | 1 | `arctic` |
| `C-ENC-DEC` — encoder/decoder cross-attention stack | 3 | `dflash`, `eagle3`, `t5encoder` |
| `C-HYPERCONN` — hyper-connections (per-layer residual streams) | 1 | `limite` |
| `C-LINEAR-ATTN` — linear/recurrent attention layer | 2 | `delta-net-base`, `rwkv6-base` |
| `C-MULTIMODAL` — vision tower / multimodal projector | 1 | `clip` |
| `C-RWKV` — RWKV scan | 5 | `arwkv7`, `rwkv6`, `rwkv6qwen2`, `rwkv7`, `rwkv7-base` |
| `C-SHORTCONV` — short convolution / conv1d in the block | 16 | `bailingmoe3`, `clef`, `glm5-next`, `kimi-k3`, `kimi-linear`, `lfm-base`, `lfm2`, `lfm2moe`, `lfm3`, `mamba-base`, `plamo2`, `qwen35`, `qwen35moe`, `qwen3next`, `qwen4exp`, `wavtokenizer-dec` |

**Not blocking:** 17 families carry a NextN / MTP head. Upstream builds the head only when a loader asks for it (`ml.load_mtp`), and it stores the head inside `block_count`: the trunk is `block_count - nextn_predict_layers` layers, so the head's weights are ordinary `blk.{n_layer}.*` blocks *plus* the `nextn.*` names it owns. This engine subtracts the declared count and maps both halves to `IGNORED` with the reason recorded in `inspect`. Pinned by the `qwen3moe-mtp` fixture and the `mtp_head_tolerated` test: the file loads, every head tensor is reported, and the logits are bit-identical to the same checkpoint without the head. Removing either half (`nextn_predict_layers`, or the index rule) changes what runs, which the test asserts.

Families carrying one: `bailingmoe3`, `cohere2moe`, `deepseek32`, `deepseek4`, `dflash`, `glm-dsa`, `glm4-moe`, `glm5-next`, `hy-v3`, `mellum`, `mimo2`, `nemotron-h-moe`, `qwen35`, `qwen35moe`, `qwen3next`, `qwen4exp`, `step35`.

Mapped rows whose file also builds a block the row does not model:

- `deepseek4` — `C-HYPERCONN`, `C-TID2EID`
- `deepseek41` — `C-HYPERCONN`, `C-MULTIMODAL`

Known false positives of the marker scan, each with its reason:

- `mamba2` / `C-SHORTCONV` — the `conv1d` in this file is `ssm_conv1d`, the SSM short convolution the mamba2 block already implements — not a generic shortconv layer.

<!-- arch-census:end -->

**How this closes.** Every row in the queue above ends in one of exactly two
states: an adapter row — after the three checks below — or a named capability.
The census is finished when `unclassified` reads 0 and each `blocked` capability
has an implementation with a fixture. It cannot be finished by relabelling: the
offline gate fails when a family with an adapter row is marked blocked, when a
blocked family names no declared capability, or when the table is not what the
artefact renders.

A marker scan is only a first pass, so a family only becomes a row after three
further checks, and a row was added only when all three passed:

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
other candidates are blocked on, in the terms of the file that blocks them.

**The read queue, in order.** The `unclassified` column above is the work. It is
ordered so that each batch is a whole *shape* rather than a slice of the
alphabet, because a batch of near-identical files shares one graph read:

1. **MoE trunks that differ only in router and FFN wiring** — `bailingmoe`,
   `bailingmoe2`, `deepseek`, `dots1`, `dbrx`, `ernie4-5-moe`, `exaone-moe`,
   `flint`, `glm4-moe`, `grovemoe`, `hunyuan-moe`, `maion-coder`, `maple`,
   `minimax-m2`. Each is a pre-norm block with routed experts, so what the read
   has to settle is the router (softmax or sigmoid, renormalised or not, bias or
   not) and the expert FFN (shared expert gated, ungated or absent; activation;
   width), not the attention. Eight of the fourteen are rows now
   (`bailingmoe`, `bailingmoe2`, `dbrx`, `deepseek`, `dots1`, `ernie4-5-moe`,
   `glm4-moe`, `minimax-m2`); of the rest, `exaone-moe`, `flint` and `maple` need
   per-layer rotary configuration, and `grovemoe`, `hunyuan-moe` and
   `maion-coder` need a block rather than a flag.
2. **Encoder and embedding stacks** — `jina-bert-v2`, `jina-bert-v3`,
   `nomic-bert`, `nomic-bert-moe`, `gemma-embedding`, `gemma-embedding2`,
   `neo-bert`, `modern-bert`. These are expected to end in a named capability
   (pooling head, segment embeddings, no kv cache) rather than in a row, and a
   named capability is a finished outcome, not a deferral.

A batch is finished when every family in it is either mapped — row, fixture and
oracle golden, with a mutation that turns the golden red — or moved to a named
capability with its reason written here. Overlap with a mapped family is where a
read starts; it is never why a row is added.

> The table below is **hand-read detail, not the partition**: it was written by
> reading the graphs of the families that were candidates at the time, several of
> which have since been mapped (`falcon`, `mpt`, `olmo`, `command-r`, `bloom`,
> `jais` are struck through for that reason). The generated table above is the
> authoritative statement of where each family stands; this one says what each
> *capability* costs. Where the two disagree, the generated table is right.

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

### C10. ~~Benchmarks run on an ISA the target CPU does not have~~ — **FIXED; measured**

The knob: `--kernels auto|scalar|avx2|zen3|avx512` selects the CPU backend's ISA,
`resolve_kernel_isa` applies the Zen 3 policy (`zen3` and `avx2` pin AVX2/FMA3,
and `auto` **never** selects AVX-512 because the target cannot execute it), and
each backend operation reports the ISA it actually ran
(`record_operation(..., "cpu-" + kernel_isa_name(...))`). `freemoe bench --json`
reports `resolved_isa` for the host's `auto` policy; the per-run record is the
operation log, not that summary line. Pinned by the `kernel_isa` checks: scalar
stays scalar, `zen3` parses to `avx2_fma`, an unknown profile is refused.

The **measurement** was the part owed, and it is now recorded:

* `configs/benchmark.env` pins `EXTRA_ARGS="--kernels zen3"`, and
  `benchmarks/run_repeats.sh` writes `extra_args` into the run's `protocol.txt`.
  A run record that does not say which profile ran cannot be re-checked; the old
  records did not.
* `python3 scripts/archive_benchmark.py` reads the five reports of that run,
  tallies `backend.operations[].backend`, and writes
  `docs/evidence/benchmark_zen3.json`. Result: **637,100 `cpu-avx2_fma`
  operations, zero `cpu-avx512`**. The script exits non-zero if a pinned profile
  did not dispatch, so "the numbers describe the target" is now enforced at
  archive time rather than asserted in prose.
* The re-run replaces the AVX-512 record in the baseline section above, and the
  two are too close to separate (the AVX-512 record was 3051 tok/s, p95 0.41 ms,
  first token 85.5 ms, all inside the spread the same configuration shows across
  sessions) — a 2-layer model spends its time outside the GEMV kernels — so this
  is a record of the path the target executes, not a measurement of what the ISA
  costs. The kernel-level comparison is a block generated in `docs/KERNELS.md`
  from its own archive, and it does not show a consistent AVX-512 win: at these
  widths each path is the faster one on some shapes, so no percentage is quoted
  for the pair anywhere in this tree.

Still open, and not something a re-run on this host can close: the sandbox is a
2-vCPU KVM guest on an Intel Xeon, not a Zen 3 machine. The *code path* is the
target's, the *microarchitecture* is not, so these are not target cycle counts.
That needs the D1 hardware record (`docs/TEST_MACHINE.md`) to exist.

---

## D. Documentation drift

| Document | Claim | Reality |
| --- | --- | --- |
| `docs/ROADMAP_INFERENCE_AND_TRAINING.md:123` | "I1 … Exists: one synthetic adapter, MHA, separate output weight, F32. Missing: any real architecture." | 22 adapters, GQA/MQA/MLA, 26 quant types |
| `docs/ROADMAP_INFERENCE_AND_TRAINING.md:101` | "I0 … missing: separation between container parsing and architecture semantics" | Still true: `src/gguf.cpp` is a 2191-line file mixing container, capability and adapters |
| `docs/SUPPORTED_MODELS.md:51` | "## Explicitly unsupported: arbitrary Llama, Mixtral, DeepSeek, Qwen, Gemma …" | Scope note exists but is weak; this section describes the *native* adapter only and reads as a global claim |
| `docs/TEST_MACHINE.md:13` | "AVX-512 disabled for FreeMoe backend tests" | Now enforced where it is measurable: `configs/benchmark.env` pins `--kernels zen3`, the run's `protocol.txt` records it, and `scripts/archive_benchmark.py` archives a run `fail` (exit 1) if a single compute operation dispatched anything else (C10) |
| `docs/ARCHITECTURE.md` module inventory | every file under `src/` is listed | Was **not** true after PR #10: `src/chat_template.cpp` was missing, so `scripts/check_architecture.py` — and therefore `make test` — failed on `main`. Fixed here; `make test` now exits 0. |
| `docs/ARCHITECTURE.md:53` | `src/tokenizer.cpp` provides "GPT-2 style BPE" | It provides GPT-2 style *piece tables*; the merge algorithm is missing (C1) |

---

## E. What to do, in order

Ranked by (models unlocked) ÷ (new concepts introduced). Nothing here requires
fabricating a capability.

| # | Work | Why first | Size |
| --- | --- | --- | --- |
| 1 | **BPE correctness** (C1) | The only gap that produces wrong output *today* on models that already "work" | M |
| 2 | **Adapter rows for what is already understood** | Near-zero engine work: table entry + fixture + golden per family, the pattern PR #10 established. The census names the candidates and their read queue in §C11 | M (breadth) |
| 3 | **YaRN / `rope_freqs`** (C4) | +14 models, one rotary change, already isolated behind `rope_dim` | S |
| 4 | ~~**CPU ISA override + AVX2-only re-benchmark** (C10)~~ **done** — the pin is in `configs/benchmark.env` and the archive is gated on the dispatch | Today's only performance evidence was from an ISA the target lacks | S |
| 5 | **Eviction-exercising benchmark config** (§A Phase 5/6) | Without it the cache-policy and prefetch gates cannot be evaluated at all | S |
| 6 | ~~**Batched prefill** (C2)~~ **done** (`--kernels zen3`; the measured ratios are in §C2) | Largest throughput win at prompt length > 32; makes §10.2 metrics honest | L |
| 7 | ~~**IQ1–IQ4 / MXFP4 decoders** (C7)~~ **done** | Unlocks quantised checkpoints of MoE families that are already supported | M |
| 8 | ~~**GEGLU_ERF**~~ **done**; **SITU** (C9 remainder) | `gelu_erf` is reachable through `mistral3`; `SITU` is unimplemented upstream too (`GGML_ABORT`), so no file can be checked against it | S |
| 9 | ~~**mamba/mamba2**~~ **done**; **gated delta-net** | The mamba scan, its state layout and the `dt_b_c_rms` norms now run with fixtures and goldens. Delta-net is a different scan (per-head matrix state) and is what `qwen35`, `qwen3next` and `kimi-linear` still need | L |
| 10 | ~~**SWA cache** (C3)~~ **done**, ~~**per-expert tensors** (C8)~~ **done**; **deepseek41/clamp rule** (C5) | Completeness of things already claimed | S–M |
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
* **Upstream is a moving target.** The census is pinned to
  `0ec4027a2a01…`; upstream adds model files monthly. Bumping the pin is a
  deliberate commit (`make census LLAMA_DIR=…`) that regenerates the artefact
  and the §C11 table, so the count cannot drift silently.

---

## Appendix: reproduce

```sh
make -j2 test                                   # unit, fuzz, traces, census, docs
./build/tests architecture                      # adapter parity rows
./build/freemoe probe --model <file.gguf>       # per-file capability report
bash benchmarks/run_repeats.sh configs/benchmark.env   # the numbers in §Baseline
python3 tools/make_arch_fixtures.py --check     # fixtures up to date

# census (needs the upstream tree, at the pinned commit)
git clone --filter=blob:none --sparse --depth 1 \
    https://github.com/ggml-org/llama.cpp /tmp/llama-census
git -C /tmp/llama-census sparse-checkout set src/models
git -C /tmp/llama-census checkout 0ec4027a2a0183de87c007069adddc06e665f7de
make census LLAMA_DIR=/tmp/llama-census
```
