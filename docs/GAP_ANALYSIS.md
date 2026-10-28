# Gap analysis: design vs code vs model codes

> **Status of this document.** Section C's findings C1, C4 and C10 are fixed:
> BPE and SentencePiece are real encoders now (8 vocabularies at 46/46,
> `tools/tokenizer_parity.py`), linear/YaRN/`rope_freqs` context extension and
> attention sinks are implemented, and `auto` no longer dispatches AVX-512 on a
> Zen 3 target. The measurements below are the state at `f704e4b` and are kept
> as the baseline this work is measured against; current numbers live in
> `docs/MODEL_COVERAGE_STATUS.md` (1993 checks, 23 architecture fixtures).

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
| Unit / parity / fuzz checks | **1338, 0 failures** (`./build/tests`) |
| Architecture parity fixtures | **17** architectures, 30 `.gguf` fixtures (9.3 MB) |
| Model adapters | **22** entries = 21 real architectures + native `freemoe-tiny` (`src/support.cpp`) |
| Upstream architectures with an adapter | **18 of 169** model files |
| Quantisation types decodable | **26** (`src/quant.cpp`) |
| Error codes | **28** non-OK codes (`include/fm/common.hpp:27`) |
| Benchmark vs declared targets | **pass** — but on the synthetic fixture, and on the wrong ISA (below) |

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

169 model files upstream. Requirement signatures were extracted statically
(which tensors and `build_ffn`/`build_norm` constants each file uses) and
intersected with what this engine implements.

| Bucket | Count | Meaning |
| --- | --- | --- |
| Adapter today | **18** | `llama qwen2 qwen3 gemma gemma2 gemma3 phi3 minicpm qwen3moe glm4 qwen2moe phi2 internlm2 orion exaone gptneox stablelm deepseek2` (`smollm`, `smollm2`, `tinyllama` are the same `llama` block) |
| Not a causal text decoder | **24** | BERT/Nomic/Jina/EuroBERT/ModernBERT/NeoBERT encoders, `llama-embed`, `gemma-embedding*`, `t5`/`t5encoder`, `clip`, `pockettts`, `qwen3tts`, `wavtokenizer-dec`, `paddleocr`, `llada`, `llada-moe`, `rnd1`, `dream`, `dflash`, `vllm-decision2` — no-causal-mask attention and pooling, or diffusion/audio/vision |
| **Needs only an adapter row** | **56** | Every primitive already exists: stacked-expert MoE, shared experts with/without a scalar gate, group-limited routing, per-head q/k norm, post-norms, sliding window, partial RoPE, fused q/k/v, LayerNorm, silu/gelu/relu/relu², MLA. `baichuan bloom chatglm gpt2 starcoder2 falcon mpt olmo olmoe phimoe dbrx arctic grok minimax-01 minimax-m2 qwen mistral4 command-r cohere2 …` |
| Needs one more primitive | 71 | below |

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

Cost-effectiveness ordering is therefore: the 56 adapter rows first (no engine
code), then YaRN (+14, one rotary change), then the two recurrent families
(+17 together), then the small token-level primitives.

### The 10 that stay hard

`arcee gemma3n jais2 kimi-k3 minimax-m3 nemotron nemotron-h nemotron-h-moe
openai-moe plm` — each needs a change to the block itself, not a variant of it:
AltUp / per-layer embeddings (`gemma3n`), KDA plus the `SITU` timing
nonlinearity (`kimi-k3`), clamped `SWIGLU_OAI_MOE` with attention sinks
(`openai-moe`, `minimax-m3`), ReLU² layers with per-layer quirks
(`nemotron*`), MLA plus `RELU` (`plm`).

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

### C1. BPE encoding is max-match, not BPE — **highest severity**

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

### C2. Prefill is sequential

`run_prompt` feeds prompt tokens one at a time (`src/engine.cpp:1146`). Correct,
but it costs N forward passes instead of one GEMM-heavy pass, and it makes the
§10.2 prefill metrics meaningless (see §A).

### C3. Sliding-window models reserve and keep the full cache

`kv_bytes = layers * context * (kv_dim + v_dim) * 4` (`src/residency.cpp:179`)
ignores `sliding_window`. The mask is applied correctly per head
(`first_visible`, `src/engine.cpp:834`), so results are right, but a gemma-2/3 or
cohere2 checkpoint pays for full-context KV it can never read. No ring buffer.

### C4. No RoPE scaling of any kind

No `rope_scaling`, `rope_freqs`, YaRN or linear scaling anywhere in `src/`
(26 upstream files use per-layer `rope_freqs` or frequency factors). Those
models will produce wrong positions outside the native window — and for
long-context checkpoints, inside it too.

### C5. Router renormalisation differs from upstream in two ways

Upstream (`llama-graph.cpp:2338–2358`) sums the selected weights after any
`SOFTMAX_WEIGHT` step and clamps the denominator to `6.103515625e-5`
(smallest F16 normal), **except** `DEEPSEEK41`, which uses `sum + 1e-20`.
This engine (`src/engine.cpp:1010–1018`) computes the sum *before* the softmax
and guards with `total > 0.0f`, with no clamp and no per-architecture rule.

* The missing clamp is harmless while the sum stays above 6.1e-5.
* The ordering matters only for `SOFTMAX_WEIGHT` gating — used upstream by
  `openai-moe` alone, which is not supported here. So the bug is **latent, not
  reachable today**; it must be fixed in the same change that adds that family.

### C6. Sampling is thin

`SamplingConfig` (`include/fm/engine.hpp:24`) has temperature, top-k, top-p,
seed. No min-p, no repetition/presence/frequency penalty, no DRY, no grammar or
logit-bias constraints, no `n_probs`. Fine for parity tests, wrong for chat.

### C7. Quantisation coverage has holes that matter for real checkpoints

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

### C9. Activation and attention variants missing

No `attn_sinks`, no `SWIGLU_OAI_MOE` (clamped SwiGLU), no `SITU`, no
`GEGLU_ERF`. Small self-contained kernels — but they are what `gpt-oss`,
`minimax-*`, `deepseek4/41` and `kimi-k3` need.

### C10. Benchmarks run on an ISA the target CPU does not have

`host_has_avx512()` (`src/kernels.cpp:60`) probes the *host*, so on this Xeon
the CPU backend dispatches `cpu-avx512` kernels — confirmed by every matmul row
in `results/benchmark/synthetic-moe-buffered/*/report.json`. `docs/TEST_MACHINE.md:13`
states "AVX2 + FMA3 forced; AVX-512 disabled for FreeMoe backend tests", but
nothing forces it: `VNPU_FORCE_ISA` governs the Java NPU device only, and there
is no CPU-backend equivalent.

So the one benchmark we have was measured on an ISA the stated target (Zen 3, no
AVX-512) cannot execute. Fix: an ISA-selection knob for the CPU backend plus a
config that pins `avx2`, then re-run `benchmarks/run_repeats.sh`.

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
| 6 | **Batched prefill** (C2) | Largest throughput win at prompt length > 32; makes §10.2 metrics honest | L |
| 7 | **IQ1–IQ4 / MXFP4 decoders** (C7) | Unlocks quantised checkpoints of MoE families that are already supported | M |
| 8 | **mRoPE, fused `gate_up_exps`, sinks, clamped SwiGLU, SITU, GEGLU_ERF** (C9) | +18 models in five small kernels | M |
| 9 | **One recurrent family** (delta-net or mamba) | +17 models, but introduces state, scan kernels and a new cache shape | XL |
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
