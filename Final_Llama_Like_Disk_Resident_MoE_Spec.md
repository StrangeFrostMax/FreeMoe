# Llama-Like Disk-Resident MoE Inference Engine
## Final Architecture, Evidence Review, and Implementation Specification

**Status:** Proposed implementation specification  
**Scope:** Standalone C/C++ inference engine; no Kraken dependency  
**Primary use case:** Run Llama-family dense models and supported MoE models whose full weights do not fit in a GPU with approximately 12 GiB VRAM, by coordinating NVMe, system RAM and VRAM.  
**Initial model input:** GGUF.  
**Primary correctness rule:** Every selected expert must execute with its real weights. No silent expert dropping, random/zero substitution, or unreported CPU fallback.  
**Evidence rule:** A proposal, paper, issue, branch, benchmark report, or upstream feature must be labelled according to its actual status. No invented APIs, commands, implementation status, or performance claims.

---

## 1. Executive decision

Build a conventional Llama-like inference engine with a separately testable **MoE Residency Manager**. Do not start by inventing a new model format, custom filesystem, GPU cache kernel, speculative decoding system, or custom quantisation scheme.

Use this order:

1. **Establish correct baseline inference** for a small, known-supported GGUF dense Llama-family model.
2. Add **model-family metadata and tensor mapping** with strict rejection of unknown tensor semantics.
3. Add correct MoE forward execution with all required experts available, initially using a simple CPU-resident expert path if needed.
4. Add bounded GPU expert caching and CPU-to-GPU staging.
5. Add trace-driven expert locality measurement and conservative prefetch.
6. Add NVMe-backed lazy expert delivery only after RAM-resident MoE correctness and performance are measured.
7. Evaluate I/O modes, pinned-memory strategies and GPU-side cache mechanisms individually, using backend-specific implementations and measurements.
8. Consider a custom packed deployment store only if GGUF-backed paging demonstrates a measurable bottleneck that a new layout can solve.

This is a **Llama-like engine**, not a fork of Kraken and not a claim that llama.cpp itself is being reimplemented in full. This specification supersedes the earlier Kraken-x-based disk-resident MoE drafts. Their CPU-first converter and custom deployment store are not carried forward; a custom store is evidence-gated under section 9.5. “Llama-like” describes the conventional transformer execution structure and GGUF-oriented model-loading approach. It does not mean every Llama architecture variant, every GGUF quantisation, or every MoE checkpoint is supported from day one.

### Non-negotiable architecture rules

- Correctness before optimisation.
- GGUF first; custom storage is an evidence-gated later option.
- Keep attention, embeddings, normalisation, router, shared experts, output projection and KV cache separate from routed-expert residency decisions.
- Explicitly budget VRAM for model-resident tensors, expert cache, KV cache, compute graph/workspace, staging buffers and allocator overhead.
- Never assume that a 12 GiB-labelled GPU has 12 GiB safely available for experts.
- Never assume that a high expert-cache hit rate alone proves good throughput.
- No backend is “supported” until it builds, passes correctness tests and has benchmark results on that backend.
- No numerical performance target is accepted without hardware, model, quantisation, context, batch, software version, command line and measurement method.
- A missing or ambiguous tensor mapping is a hard error, not an invitation to guess.
- CPU fallback must be explicit in logs and metrics, never silent.

---

## 2. Evidence review: what is real, what is transferable, what is not

This section deliberately separates observed upstream interfaces, research systems, reports from development discussions, and proposed work.

### 2.1 Relevant current evidence

| Approach | Evidence status | Useful idea | Limits and transfer conditions |
|---|---|---|---|
| llama.cpp CPU MoE offload | Publicly documented options in the llama.cpp server documentation | Keep MoE weights on CPU and configure a GPU expert cache; separately control GPU layer placement | Exact options and behaviour must be checked against the pinned revision used for testing. CUDA/Metal/HIP/Vulkan paths may differ. Do not assume identical semantics across backends. |
| llama.cpp persistent expert-slot pool proposal | Public GitHub discussion/RFC with a companion branch and reported CUDA validation | Persistent slots, expert-ID remapping and a prefill-specific fallback can reduce repeated expert movement | The discussion is not proof that the design is merged into every upstream build. Treat it as a reference design, not a dependency. Validate branch, commit, license and implementation before adapting. |
| llama.cpp on-demand pinning report | Public issue discussing a lazy-mmap and pin-on-first-touch path, with CUDA benchmark reports | Pin only host pages that will be transferred, avoiding whole-model host-memory locking | Reported figures are specific to a CUDA configuration and the reported model/workload. Do not quote them as expected AMD/Vulkan/Windows performance. |
| llama.cpp MoE disk-paging proof of concept | Public discussion/prototype | Fixed expert slots and on-demand reads from the model file can avoid loading every expert into RAM | The prototype and its reported Apple Silicon result are not a general Windows/AMD implementation or a product guarantee. |
| vllm-metal expert offload | Public documentation | Expert cache, separate memory reservation for KV/cache/workspace, and a different workload profile for prefill versus decode | The documented path is MLX/Apple-specific and has explicit checkpoint/layout limits. It is a conceptual reference, not a drop-in backend for Windows/AMD or GGUF. |
| MoE-Infinity | Public repository and documentation | Host/SSD offload, hot-expert caching, trace/prefetch ideas and serving integration | Its supported models, dependencies, data formats and runtime differ from this C++ engine. Reuse concepts only after reviewing implementation and licence. |
| SSD-LLaMA (2026 paper) | Research paper/preprint | Three-tier SSD/RAM/VRAM delivery and CPU/GPU coordination are active research directions | Paper results are not a benchmark of this proposed engine. Reproduce comparable workloads before making performance claims. |
| GPU-side expert cache proposal | Public GitHub issue/discussion with reported results on specific hardware | Avoid CPU readback of selected expert IDs by doing cache decisions/remapping on the GPU where feasible | Reported implementation/results are hardware- and backend-specific. It is not assumed available or portable. |

### 2.2 Evidence sources

The following are the sources examined for this design. These are references, not a claim that every listed feature is merged into a single release. Before release, each source needs an access date, a commit or version where applicable, and the specific claim it supports. Until that record exists, every entry is `unverified`.

1. llama.cpp server options and load modes: https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md
2. llama.cpp persistent expert-slot pool RFC: https://github.com/ggml-org/llama.cpp/discussions/28248
3. llama.cpp MoE disk-paging RFC/prototype: https://github.com/ggml-org/llama.cpp/discussions/23324
4. llama.cpp on-demand pinning discussion: https://github.com/ggml-org/llama.cpp/issues/26448
5. llama.cpp GPU-side expert-cache proposal: https://github.com/ggml-org/llama.cpp/issues/29949
6. vllm-metal MoE offload documentation: https://docs.vllm.ai/projects/vllm-metal/en/latest/moe_offload/
7. MoE-Infinity repository: https://github.com/EfficientMoE/MoE-Infinity
8. SSD-LLaMA research: https://arxiv.org/abs/2609.18110

### 2.3 Do not transplant benchmark numbers

The sources report results across different GPUs, operating systems, backends, model sizes, quantisations, context lengths and workloads. A CUDA RTX 4090 result does not establish a result for an AMD 12 GiB GPU. An Apple unified-memory result does not establish Windows Vulkan behaviour. A paper result does not establish the performance of this codebase.

Any performance number copied into project documentation must include:
- exact source and source date/commit;
- model and exact quantisation;
- GPU, VRAM, RAM, CPU, storage device and operating system;
- backend, compiler and driver;
- prompt length, generation length, batch/concurrency and KV configuration;
- cold/warm cache state;
- benchmark command and measurement method;
- whether loading, prefill or decode time is included.

---

## 3. Goals and non-goals

### 3.1 Goals

- Run supported Llama-family dense GGUF models correctly.
- Add a documented, testable subset of GGUF MoE model families.
- Allow a model's total weight storage to exceed VRAM, and eventually RAM, by keeping cold expert weights on NVMe and delivering selected experts through bounded staging.
- Maintain a bounded VRAM expert cache with safe slot replacement.
- Support CPU-side routing/expert dispatch initially, with backend-specific optimisation only after a correct baseline.
- Measure prefill and decode separately.
- Provide explicit memory budgets, I/O counters, cache metrics and fallback diagnostics.
- Keep storage format, model-family mapping, inference graph, and residency policy separate.
- Design the interfaces so Windows and Linux can be supported without pretending that their I/O and GPU memory APIs are identical.

### 3.2 Non-goals for the first working release

- Training, fine-tuning, distillation or architecture surgery.
- Changing the model's number of experts, hidden dimensions, router semantics or layer count.
- Universal support for all GGUF quantisation types and all model families.
- Multimodal models, vision towers, audio models, LoRA, speculative decoding, continuous batching, tensor parallelism or multi-GPU execution before single-request correctness is stable.
- A custom filesystem driver or raw-partition model store.
- A proprietary new quantisation format.
- Assuming DirectStorage, GPU direct storage, zero-copy PCIe access, asynchronous disk-to-VRAM transfers or GPU cache management is available on every target.
- Promising a particular tokens-per-second number merely because the model fits on disk.

---

## 4. Target hardware and memory model

The design target is a system with a GPU advertised as having around 12 GiB VRAM. The implementation must query actual available memory at runtime. The design does not assume NVIDIA, AMD, Vulkan, CUDA, HIP, Windows or Linux unless a build profile explicitly names that target. The 12 GiB VRAM figure is a design assumption, not a verified hardware fact. The first test machine must be named in the project inputs (section 18), and its usable VRAM must be measured rather than assumed. The earlier Kraken-x drafts used an Intel Core Ultra 5 135H laptop whose integrated Arc GPU shares system memory and has no dedicated 12 GiB VRAM, so that machine cannot validate the 12 GiB path.

### 4.1 Three-tier weight delivery

**Cold tier: NVMe model source**
- Stores the original GGUF initially.
- Provides random access to validated tensor/expert byte ranges.
- Uses an I/O abstraction with platform-specific implementations.
- Records physical bytes read, request count, request size, latency distribution and read amplification.

**Warm tier: system RAM**
- Holds the OS file cache and/or explicit bounded host cache.
- Holds any staging buffers needed for upload.
- May hold pinned/registered pages only where the backend requires them and the operating system permits it.
- Must have a hard memory budget. Whole-model pinning is prohibited by default.

**Hot tier: VRAM**
- Holds tensors required on every token where budget allows.
- Holds a bounded pool of routed experts.
- Holds KV cache, compute workspace, graph buffers and transfer staging allocations.
- May hold a small number of prefetched experts if measured benefit exceeds cache pollution and memory cost.

The tiers are not equivalent. The same expert can be represented in a file, host cache and GPU slot simultaneously. Accounting must avoid treating one physical copy as multiple free copies.

### 4.2 VRAM budget equation

At runtime:

`VRAM_total = VRAM_fixed_weights + VRAM_expert_cache + VRAM_KV + VRAM_compute + VRAM_staging + VRAM_allocator_margin`

The usable expert cache is not a fixed percentage of nominal VRAM. It is the remainder after reserving the other components and a safety margin:

`VRAM_expert_cache <= VRAM_available - fixed_weights - KV - compute_peak - staging - margin`

Measure compute peak for both prefill and decode. Prefill workspace may be substantially larger than decode workspace. Do not size the cache using decode-only allocation measurements.

The first implementation should expose explicit budget settings, but it must reject unsafe settings rather than blindly allocate the requested amount. The log must show the budget calculation and each reservation.

### 4.3 Host RAM budget

`RAM_total_process = model_metadata + explicit_host_expert_cache + I/O_staging + pinned_or_registered_pages + KV_if_hosted + runtime + safety_margin`

Do not assume mmap means the file's pages consume no physical RAM. Do not assume OS cache pages are instantly reclaimable under load. Track RSS/working set, locked/pinned memory, page faults, major faults and process/system memory pressure.

### 4.4 KV cache is an independent budget

KV-cache growth depends on architecture, context length, layers, KV heads, head dimension, data type, batch and concurrency. The expert cache must not consume memory reserved for the declared context and concurrency target. Report the maximum supported context for each tested memory profile.

---

## 5. Runtime architecture

### 5.1 Component layout

```text
Application / CLI / optional HTTP server
                 |
         Request + tokenizer
                 |
      Model manifest / GGUF reader
                 |
       Architecture family adapter
                 |
       Canonical tensor/model map
                 |
          Model execution graph
                 |
   +-------------+------------------+
   |                                |
Dense tensor execution        MoE layer execution
                                    |
                              Router/top-k
                                    |
                            Expert ID requests
                                    |
                           MoE Residency Manager
                                    |
                   +----------------+----------------+
                   |                |                |
                VRAM slots       RAM cache        NVMe reader
                   |                |                |
                   +-------- bounded async staging --+
                                    |
                          backend compute kernels
```

This is a logical diagram. It does not prescribe a specific GPU queue, API or zero-copy mechanism.

### 5.2 Main modules

1. **GGUF Reader**: parses header, metadata, tensor descriptors and offsets with bounds checks.
2. **Architecture Adapter**: recognises a specific supported architecture family and maps named tensors to known logical roles.
3. **Model Validator**: checks required tensors, dimensions, quantisation types, tensor alignment and architecture invariants.
4. **Tokenizer and Chat Template**: use the model's supported tokenizer metadata/template or a verified implementation. Never silently invent a chat template.
5. **Execution Graph**: embedding, attention, RoPE where applicable, normalisation, dense FFN or MoE FFN, residuals and output head.
6. **Backend Interface**: CPU first, then a named GPU backend. No backend is advertised as functional until tested.
7. **MoE Residency Manager**: maps requested expert IDs to resident slots or schedules delivery.
8. **I/O Scheduler**: bounded asynchronous reads, cancellation and request coalescing.
9. **Host Staging Manager**: bounded buffers and optional page pin/registration.
10. **VRAM Slot Pool**: fixed-size or size-classed slots; exact tensor packing rules are architecture-adapter driven.
11. **KV Cache Manager**: independent from expert cache and independently measured.
12. **Metrics/Trace Recorder**: per-layer routing, cache hits/misses, bytes, latency, stalls and correctness diagnostics.
13. **Reference Test Harness**: compares logits and token outputs against a known-good implementation/configuration.

### 5.3 Suggested repository layout

```text
llama-moe-runtime/
  CMakeLists.txt
  README.md
  LICENSES/
  docs/
    ARCHITECTURE.md
    SUPPORTED_MODELS.md
    BENCHMARK_PROTOCOL.md
    FORMAT_AND_QUANTISATION.md
    BACKEND_SUPPORT.md
    FAILURE_MODES.md
  include/
    runtime/
    model/
    moe/
    io/
    backend/
    metrics/
  src/
    app/
    gguf/
    model/
    tokenizer/
    graph/
    moe/
    io/
    memory/
    backend/
      cpu/
      vulkan/       # only if selected and implemented
      cuda/         # only if selected and implemented
      hip/          # only if selected and implemented
    metrics/
  shaders/          # only for a GPU backend that actually uses shaders
  tests/
    unit/
    malformed_models/
    integration/
    reference/
    stress/
  benchmarks/
    io/
    expert_cache/
    prefill/
    decode/
  tools/
    model_inspect/
    trace_replay/
```

This is a proposed layout, not an existing repository inventory. Do not create empty backend directories as a substitute for working code. Include only the backends actually selected for implementation.

---

## 6. Model formats and structural discovery

### 6.1 GGUF is the initial source format

GGUF provides a useful import contract for tensor metadata, offsets, metadata and quantised tensor payloads. It does not itself provide the complete runtime policy required for an explicit three-tier MoE cache. That is a reason to separate the runtime architecture from the file layout, not a reason to immediately replace GGUF.

First implement read-only GGUF loading. Retain the source file unchanged. Do not rewrite quantised payloads during initial discovery.

### 6.2 `WAKE_UP` discovery command

A diagnostic command may be named `WAKE_UP`, but its behaviour must be precisely defined and testable. It must not be described as an existing feature until implemented.

Proposed behaviour:

`llama-moe-runtime model inspect <path.gguf>`

Optional human-friendly alias:
`llama-moe-runtime WAKE_UP <path.gguf>`

The command:
- parses the GGUF header and metadata;
- enumerates tensor names, dimensions, types, offsets and byte sizes;
- identifies an architecture only if the metadata and tensor pattern match a registered adapter;
- maps tensors to known roles where unambiguous;
- identifies likely expert tensor groups only where the adapter knows the exact naming/layout semantics;
- validates offsets, bounds, alignment and quantisation block sizes;
- reports missing, duplicate, unknown or conflicting mappings;
- emits a machine-readable report and a readable summary;
- does not alter the source model or claim that a model can execute merely because it can be parsed.

Example report fields:
- source path and file size;
- GGUF version and metadata summary;
- architecture adapter ID and adapter version;
- tensor count and total logical tensor bytes;
- tensor role mapping and unresolved roles;
- expert count/top-k/layer count only when grounded in validated metadata;
- quantisation types found;
- source range map for tensors/expert records;
- validation errors and warnings;
- a stable source fingerprint.

### 6.3 Canonical model map

The source mapper is an internal intermediate representation, not a new model architecture. It should capture:
- original tensor name;
- exact source byte range;
- tensor shape and layout;
- element/quantisation type and block geometry;
- logical role (attention, router, expert projection, shared expert, embedding, output, etc.);
- layer and expert indices only when verified;
- packing/stride information;
- whether the tensor can be paged independently without violating a quantisation block;
- adapter version and validation status.

Never infer the expert index by parsing arbitrary digits unless the specific architecture adapter defines that rule. Never assume every MoE family uses the same `w1/w2/w3` layout or expert tensor stacking convention.

### 6.4 Source and target formats

Initial release:
- GGUF is the only required input.
- No custom output container is required to prove correct inference or expert caching.
- A separate index/cache layer may be used in memory without rewriting model payloads.

Possible later target store:
- Only introduce a custom packed deployment store if controlled benchmarks show a real limitation in the GGUF layout or access path.
- Keep the format versioned and documented.
- Preserve source quantised bytes exactly for lossless repacking.
- Add per-record lengths, offsets, quantisation type, alignment, checksum, architecture/adapter identifier and compatibility flags.
- Require atomic manifest publication and recovery from interrupted conversion.
- Provide a verifier and round-trip byte/hash tests.
- Do not call a custom format “faster” until it wins a repeatable benchmark against GGUF.

---

## 7. Correct Llama-like execution

### 7.1 Dense baseline first

Implement one well-defined Llama-family dense configuration end to end:
- tokenizer and prompt template;
- embedding;
- RMSNorm or the exact normalisation specified by the architecture;
- RoPE or exact positional encoding variant;
- Q/K/V projections;
- attention and KV cache;
- output projection;
- residual connections;
- feed-forward path;
- final norm and logits;
- sampling.

The exact operations depend on model metadata and the selected adapter. Do not generalise one Llama variant's assumptions to all descendants.

### 7.2 MoE execution path

For each supported MoE layer:
1. Compute router scores using the model's specified router operation.
2. Apply the exact top-k and routing/normalisation semantics.
3. Resolve selected expert IDs and their real weight records.
4. Ensure all required expert weights are resident or delivered before execution.
5. Execute the selected experts with the correct quantised kernels and shapes.
6. Combine expert outputs according to the model's specified weights/scaling.
7. Continue the residual/normalisation path exactly as specified.

The first correct implementation may execute experts on CPU. GPU cache optimisation comes after the reference output matches. No selected expert may be skipped. Loading or executing unselected experts is prohibited by default. A prefill or batching optimisation that touches unselected experts is a separate experiment: it must produce outputs identical to the routed-only path, and its extra bytes, compute and memory are reported as separate metrics.

### 7.3 Stacked versus per-expert tensors

The adapter must distinguish:
- one tensor per expert;
- multiple experts stacked in a tensor;
- separate gate/up/down projections;
- fused gate-up projections;
- shared expert weights;
- routed expert weights;
- architecture-specific quantised packing.

For stacked tensors, a selected expert's byte range must be computed from validated dimensions and the quantisation layout. A naive `expert_index * logical_float_bytes` calculation is invalid for block-quantised storage unless the format explicitly proves that the resulting range aligns with complete quantisation blocks.

### 7.4 Numerical correctness

Correctness has two categories:

**Same encoded weights, same arithmetic path:** demand exact deterministic equivalence where deterministic execution is expected. Paging should change the location/timing of weights, not their values.

**Different backend/kernel or arithmetic path:** compare logits and generated tokens using defined absolute/relative tolerances, then evaluate perplexity or task quality if appropriate. Document nondeterministic reductions and precision changes.

The runtime must not use a lower-quality approximation to hide an expert cache miss.

---

## 8. MoE Residency Manager

### 8.1 Responsibilities

The manager owns:
- per-layer expert identity;
- expert tensor records and source extents;
- current residency state;
- VRAM slot assignment and generation;
- pending I/O and upload requests;
- cache admission/eviction policy;
- pinning/registration lifetime where supported;
- subscriber cancellation;
- demand versus prefetch priority;
- trace collection and failure propagation.

It does not own the model router's mathematical decision. The router selects experts; the residency manager fulfils those exact requests.

### 8.2 State machine

Each expert/slot request should move through an explicit state machine, for example:

`ABSENT -> IO_QUEUED -> HOST_READY -> UPLOAD_QUEUED -> GPU_READY -> IN_USE -> GPU_READY -> EVICTABLE -> ABSENT`

This is the happy path. `ERROR` and `CANCELLED` are reachable from in-flight states as the table below defines.

Failure and cancellation states are `ERROR` and `CANCELLED`. State transitions must be guarded against stale asynchronous completions. The table below is the authoritative transition set. Any transition not in it is invalid, must be rejected, and must be logged.

| Current state | Event / guard | Next state | Required action |
|---|---|---|---|
| ABSENT | Demand or prefetch request accepted | IO_QUEUED | Reserve slot; record key, generation and subscriber type |
| IO_QUEUED | Read completes and validates | HOST_READY | Publish host copy only after validation |
| IO_QUEUED | Cancel; only prefetch subscribers remain | CANCELLED | Cancel I/O if possible; discard result on completion |
| IO_QUEUED | Cancel; a demand subscriber remains | IO_QUEUED | Keep load alive; never cancel for prefetch pressure |
| IO_QUEUED, HOST_READY, UPLOAD_QUEUED | I/O, checksum or timeout error | ERROR | Notify demand subscribers with a typed error; release after callbacks retire |
| HOST_READY | Upload queued | UPLOAD_QUEUED | Record H2D fence and generation |
| UPLOAD_QUEUED | Fence complete and generation matches | GPU_READY | Publish to slot only after fence completes |
| UPLOAD_QUEUED | Upload fails | ERROR | Retire fence before any reuse of the slot |
| GPU_READY | Compute acquires reference | IN_USE | Increment reference count |
| IN_USE | Last reference released | GPU_READY | Decrement only; no eviction while referenced |
| IN_USE, GPU_READY, UPLOAD_QUEUED | Device loss or reset | ERROR | Mark all slots on that device invalid; apply the documented fallback or fail |
| GPU_READY | Eviction selected; zero references; no demand waiter on this generation | EVICTABLE | Prevent new references |
| EVICTABLE | All GPU fences complete | ABSENT | Release slot; clear key |
| ERROR | Callbacks retired; references zero | ABSENT | Log and record metrics |
| CANCELLED | I/O and upload retired | ABSENT | Release slot |
| Any non-terminal | Shutdown | CANCELLED or ERROR (teardown) | Stop new work; drain or cancel; release only after completion |

A slot record should include:
- layer and expert identity;
- source-record fingerprint;
- slot index and generation;
- payload byte size and quantisation/layout ID;
- last-used sequence number;
- demand-use count;
- prefetch-use count;
- upload/read completion fence or equivalent;
- current users/reference count;
- error state.

Generation IDs prevent a late upload completion for expert A from marking a slot containing expert B as ready.

### 8.3 Safe eviction

A slot may be overwritten only if:
- it is not in use by an outstanding compute operation;
- all relevant GPU fences/events have completed;
- no demand request is waiting on the slot's old generation;
- its state transition is valid;
- the new expert's complete payload fits the slot/layout.

If all slots are busy, use a bounded transient staging path, wait, or execute on the explicitly supported CPU path. Never overwrite live weights to make room.

### 8.4 Cache policy progression

Start with LRU or clock-style replacement, not a complex learned predictor.

Then compare:
- LRU/clock;
- per-layer locality-aware policy;
- frequency-aware policy;
- router-trace-based prefetch;
- any learned predictor only if a simple baseline shows a material opportunity.

Metrics:
- hit rate by layer and globally;
- miss rate by demand/prefetch;
- wasted prefetch bytes;
- evictions per generated token;
- slot occupancy;
- queue wait;
- I/O latency;
- H2D latency;
- compute stall time due to experts;
- tokens/s and inter-token latency.

A cache hit is meaningful only if it avoids a costly fetch and does not create another bottleneck.

### 8.5 Persistent expert slots

A persistent GPU slot pool is a candidate design because public llama.cpp discussions describe this approach. The first implementation should use the simplest correct host-controlled cache. A GPU-side expert-ID remapping kernel or graph-integrated cache is a later optimisation, not a prerequisite.

If adapting an external implementation:
- pin and record its exact commit;
- inspect its licence;
- verify its build/backend requirements;
- identify the model and quantisation assumptions;
- port only the independently understood mechanism;
- re-run correctness and performance tests on the target backend.

---

## 9. I/O and host-memory strategy

### 9.1 Begin with ordinary file access

Benchmark the real GGUF workload using:
1. buffered reads;
2. mmap-backed access;
3. aligned direct I/O where the platform and filesystem support it;
4. optional explicit host cache.

Do not assert that NTFS is inherently too slow or that a raw partition is inherently faster. The result depends on access pattern, OS cache state, storage device, alignment, request size, memory pressure and whether reads are sequential or random.

### 9.2 Direct I/O requirements

Direct I/O has alignment and size requirements that vary by operating system and filesystem. The I/O abstraction must:
- query or configure valid alignment;
- align offsets, sizes and buffer addresses;
- handle final partial records correctly;
- use a bounded bounce buffer when needed;
- fall back only through an explicit, logged policy;
- measure the fallback rate;
- never silently misalign or truncate a request.

On Linux, `O_DIRECT` may be tested where supported. On Windows, `FILE_FLAG_NO_BUFFERING` may be tested with its required alignment rules. These are separate implementations, not one portable flag.

### 9.3 Asynchronous I/O and coalescing

The I/O scheduler should:
- bound outstanding requests;
- coalesce adjacent ranges only when doing so does not read excessive unused bytes;
- prioritise demand fetches above speculative prefetch;
- cancel prefetch when its subscriber is gone;
- avoid cancelling a shared request while another demand subscriber needs it;
- record useful bytes versus read bytes;
- handle short reads and errors;
- avoid unbounded request queues.

### 9.4 Pinned host memory

Pinned/registered memory can improve asynchronous host-to-device transfer on some backends, but it is a limited resource and backend-specific.

Default policy:
- do not pin the entire model;
- pin only bounded staging buffers or selected pages when supported;
- account locked bytes against a configured budget;
- unpin/unregister only after all transfers using the region have completed;
- handle registration granularity and adjacent-region rules;
- fail gracefully if registration is unavailable;
- compare against pageable transfer and mmap paths.

A reported CUDA result must not be used as evidence that a Vulkan or AMD path supports equivalent behaviour.

### 9.5 File layout and custom storage

Keep GGUF until measurements show that expert payload placement causes excessive read amplification or unacceptable access latency. If a later custom store is justified, its primary advantage must be demonstrated by:
- lower bytes read per useful expert;
- lower p95/p99 fetch latency;
- less CPU overhead or fewer system calls;
- no correctness regressions;
- no unacceptable conversion time or storage expansion.

A raw partition/custom filesystem is not in the initial scope. It adds provisioning, recovery, portability and data-loss risks and must beat normal file access in controlled tests before being considered.

---

## 10. Prefill versus decode

Do not use one cache policy or one benchmark to represent both phases.

### 10.1 Decode

Each generated token usually routes a small number of experts per MoE layer, but the exact count is model-specific. Decode performance can be dominated by:
- sequential dependency between layers;
- expert misses;
- NVMe and host-to-device latency;
- small matrix/vector kernels;
- CPU launch/scheduling overhead;
- synchronisation;
- KV-cache traffic.

Measure per-token I/O and stall time, not only aggregate bandwidth.

### 10.2 Prefill

Prompt processing may touch a broader set of experts, particularly with longer prompts or batches. It can require larger temporary buffers and can change the best cache policy.

Measure:
- prompt tokens/s;
- peak compute/workspace memory;
- unique experts used per layer and per prompt;
- bytes fetched per prompt;
- expert-cache hit rate during prefill;
- time blocked on expert availability;
- memory use at the target context/batch.

Do not extrapolate decode cache-hit rates to prefill.

### 10.3 Prefetch policy

Begin with no speculative prefetch and measure demand misses. Then add a conservative trace-based policy:
- use only known routing history or information available without violating the model's computation dependencies;
- bound the number and bytes of outstanding prefetches;
- lower prefetch priority than demand;
- cancel stale prefetches;
- report useful-prefetch rate and wasted bytes.

Do not claim future expert IDs are known before router execution unless a specific algorithm supplies that information. Any predictive prefetch is a prediction, not certainty.

---

## 11. Performance model and feasibility gates

### 11.1 Data movement model

For a simplified decode estimate:

```text
E[bytes_per_token] = Σ_layers Σ_experts P(selected(layer, expert)) × (1 − P(hit | layer, expert)) × bytes(layer, expert)
```

where `P(selected)` is the probability that the router selects the expert for a token, and `P(hit)` is the probability that it is resident when required. Estimate both from the trace.

A uniform approximation such as `L * k * (1-h) * s` may be used only when `L`, `k`, `h` and `s` are defined consistently and expert payload sizes are sufficiently uniform. For real reports, compute from the trace because layer sizes, selected experts and cache behaviour can vary.

An idealised lower bound for transfer time is:

`ideal_transfer_time >= bytes_moved / measured_effective_bandwidth`

This excludes queueing, random-read latency, CPU scheduling, synchronisation, kernel execution and dependencies, so it is not a token/s prediction.

### 11.2 Mandatory metrics

For every benchmark:
- prompt/prefill tokens/s;
- decode tokens/s;
- median and p95 inter-token latency;
- model load time;
- first-token latency;
- peak VRAM;
- peak process RAM/working set;
- pinned/registered host bytes;
- NVMe bytes read and read request count;
- useful expert bytes versus total bytes read;
- host-to-device bytes;
- expert cache hit/miss rate;
- prefetch useful/wasted bytes;
- GPU utilisation and idle/stall time where reliably measurable;
- CPU utilisation;
- page faults where available;
- output/logit correctness status;
- cold-cache and warm-cache state.

### 11.3 Benchmark controls

- Fix model, quantisation, prompt, generation length, context, batch, sampling and thread count.
- Record exact software commit, build flags, backend, driver and hardware.
- Separate model loading, prefill and decode.
- Run repeated trials and report median plus spread.
- Include a cold-storage/cache condition and a warm condition.
- Avoid comparing one run's cold start with another run's warm cache.
- Track thermal and power state where possible.
- Validate generated output, not just throughput counters.
- Use an external known-good runtime as a reference when the same model/backend semantics are available.

### 11.3A Cold and warm protocol

- A process restart is not a cold run.
- **Warm:** read the relevant files until they are resident. Record the method. Keep the warming pass separate from measured runs.
- **OS-cold:** use a file set larger than available RAM, or a controlled cache-clearing method where it is safe and permitted. On Linux, dropping caches is intrusive and belongs only in a controlled test environment. On Windows, if eviction cannot be guaranteed, label the run `process-cold`, not `OS-cold`.
- Warm-up for code paths and kernel selection is separate from cold I/O measurements.
- Five repetitions is the minimum exploratory count. Use more when variance is high. Report median and spread.
- A run whose cache state is unknown is labelled `unknown` and excluded from cold-versus-warm comparisons.

### 11.4 Feasibility gate

Before expanding model support, estimate:
1. total expert bytes required by the selected architecture and quantisation;
2. measured expert miss bytes/token;
3. measured NVMe and H2D latency/bandwidth;
4. CPU/backend compute time;
5. dependency stalls per layer;
6. achievable cache hit rate on actual routing traces.

Two bounds are reported per token. `transfer_lb` is defined in section 11.1 and `compute` is measured.
- Optimistic lower bound with perfect overlap: `T_lb = max(transfer_lb, compute)`. This is a valid lower bound on token time only where transfer and compute can overlap. Decode is layer-sequential (section 10.1), so the achievable overlap must be measured, not assumed.
- Conservative no-overlap estimate: `T_seq = transfer_lb + compute`. Reported for planning. It is not a rejection criterion.

Rejection rule: if `T_lb` exceeds the required inter-token latency, the configuration is infeasible. Change the target workload, cache/memory design, quantisation or hardware assumptions before building more features. Do not hide an infeasible result behind a high nominal bandwidth number.

No universal minimum tokens/s is specified here because the target model, exact GPU, storage, CPU and user latency requirement have not been fixed. Set that acceptance number, record who approved it, and store it in `docs/BENCHMARK_PROTOCOL.md` before Phase 4 begins. Phase 4 and later are blocked until this is done. Timings from earlier phases may be reported, but they are not compared against a target.

---

## 12. Backend strategy

### 12.1 CPU reference backend

Required first. It provides:
- correctness oracle for supported operations;
- deterministic test path where possible;
- baseline timings;
- fallback only when explicitly enabled and visible.

The CPU reference is not automatically fast enough for production. It is the correctness anchor.

### 12.2 GPU backend selection

Choose one target backend based on the actual first deployment environment. Do not implement CUDA, HIP and Vulkan simultaneously without a demonstrated need.

Backend interface responsibilities:
- device discovery and memory budget;
- allocation/deallocation;
- buffer upload/download;
- kernel dispatch and synchronisation;
- quantised matmul support;
- attention/KV operations required by the chosen model;
- event/fence lifecycle;
- timing and error reporting.

If the target is Windows plus AMD, Vulkan is a plausible portable route, but the specific quantised kernels, transfer path, queue selection and driver behaviour must be measured. If the target is Linux plus AMD, HIP/ROCm may be considered as a separate backend. Neither is assumed implemented by this document.

### 12.3 No fake acceleration

A GPU backend must fail loudly if required kernels are unavailable. It must not report “GPU inference” when the major matmul or expert work silently runs on CPU. Logs and metrics should identify each operation's actual backend and bytes transferred.

---

## 13. Implementation phases and release gates

### Phase 0: repository and dependency audit

Deliver:
- repository tree and build instructions;
- dependency/licence inventory;
- supported compiler/OS/backend matrix;
- a clear list of existing code versus planned code;
- reproducible build and test command.

Gate: clean build on the declared target. No feature work based on assumed files or functions.

### Phase 1: GGUF inspection and dense CPU correctness

Deliver:
- bounded, fuzzable GGUF reader;
- `inspect`/`WAKE_UP` report;
- one dense Llama-family adapter;
- tokenizer/chat-template handling for the selected model;
- CPU forward pass and deterministic greedy generation;
- reference logits/output comparison.

Gate: malformed models fail safely; all required tensor roles validate; the supported test model matches reference within declared tolerance. Do not proceed to paging if this fails.

### Phase 2: dense GPU baseline

Only for the selected backend.

Deliver:
- required dense kernels;
- device/VRAM accounting;
- prefill/decode timing;
- CPU versus GPU correctness report;
- backend operation log.

Gate: no silent CPU fallback; stable output; bounded allocations; repeatable measurements.

### Phase 3: one MoE family, resident experts

Deliver:
- exact family adapter;
- router and top-k semantics;
- per-expert and/or stacked tensor mapping;
- all experts resident in host memory or a simple known-good path;
- correct MoE layer output and full-model output.

Gate: selected experts are never dropped; routing and output match the reference. Add only one model family at a time.

### Phase 4: VRAM expert slot pool

**Entry gate:** the performance target in section 11.4 is set, approved and recorded.

Deliver:
- bounded slot allocation;
- slot state machine and generation IDs;
- safe eviction/fence handling;
- host-to-device staging;
- expert cache metrics;
- all-slots-busy behaviour.

Gate: stress tests with forced eviction and delayed transfers show no stale-slot reads, corruption, deadlocks or hidden fallbacks.

### Phase 5: trace and cache policy

Deliver:
- per-layer expert trace;
- LRU/clock baseline;
- cache-size sweep;
- prefill and decode measurements;
- useful/wasted prefetch metrics.

Gate: the chosen cache policy reduces decode p95 inter-token latency against the LRU/clock baseline, at the same VRAM budget, prompt set and sampling settings, across repeat runs. Hit rate is reported but is not the pass criterion. Output correctness is unchanged and the memory budget holds.

### Phase 6: NVMe-backed expert delivery

Deliver:
- bounded async reader;
- mmap/buffered/direct-I/O comparison;
- optional bounded pinned/registered host staging;
- demand-priority scheduling;
- request coalescing and cancellation;
- cold/warm benchmark suite.

Gate: all selected experts execute correctly under cold-cache pressure; RAM, VRAM, pinned memory and I/O queues remain bounded; measured end-to-end performance meets the project target.

### Phase 7: optional layout optimisation

Only if Phase 6 demonstrates that the source layout/access path is a material bottleneck.

Candidate work:
- packed expert records or an index sidecar;
- larger/coalesced reads;
- an optional versioned deployment store.

Gate: controlled A/B test proves an end-to-end improvement over unmodified GGUF on the declared workload, with round-trip integrity and recovery tests. Otherwise retain GGUF.

### Phase 8: advanced optimisations

Each is a separate experiment, not a bundle:
- GPU-side slot remapping/cache;
- more advanced prefetch;
- dequantisation cache;
- fused expert kernels;
- speculative decoding;
- batching/concurrency;
- multi-GPU support.

Each requires a hypothesis, baseline, correctness gate, benchmark and rollback path.

---

## 14. Failure handling and safety

The engine must explicitly handle:
- corrupt/truncated GGUF;
- unsupported GGUF version or quantisation;
- unknown model architecture;
- tensor shape mismatch;
- invalid expert ID;
- invalid or misaligned source extent;
- short/failed disk read;
- host staging allocation failure;
- page pin/registration failure;
- GPU allocation failure;
- transfer or kernel failure;
- stale asynchronous completion;
- cache slot eviction while in use;
- cancellation during I/O or upload;
- out-of-memory during prefill;
- KV cache exhaustion;
- device loss/reset;
- model file modified after index creation.

Required behaviour:
- propagate an error with the failing layer/tensor/expert and operation;
- do not substitute weights;
- invalidate cache/index state when the source fingerprint changes;
- release resources only after asynchronous users are finished;
- stop generation if correctness cannot be guaranteed;
- report backend fallback explicitly if a documented fallback is enabled.

---

## 15. Test plan

### 15.1 Unit tests
- GGUF metadata parsing and boundary checks;
- tensor offset/size validation;
- quantisation block-size arithmetic;
- expert index mapping;
- stacked-tensor slicing at quantisation block boundaries;
- source fingerprint and checksum;
- slot generation/state transitions;
- cache admission and eviction;
- cancellation/subscriber accounting;
- I/O alignment;
- budget calculation.

### 15.2 Malformed-input and fuzz tests
- truncated headers and metadata;
- unreasonable tensor counts/dimensions;
- offset overflow;
- overlapping or out-of-range extents;
- unknown quantisation type;
- missing required tensors;
- duplicate names;
- invalid tokenizer metadata;
- corrupt expert map.

### 15.3 Integration tests
- dense CPU inference;
- dense GPU inference for the selected backend;
- MoE CPU/reference inference;
- MoE GPU execution with all experts resident;
- cache hit path;
- demand miss path;
- prefetch hit and wasted prefetch;
- all slots busy;
- disk read failure;
- upload failure;
- cancellation during read/upload;
- forced eviction under active compute;
- cold-cache generation;
- model source changed after index creation.

### 15.4 Correctness tests
- compare layer outputs for the supported model;
- compare router scores/top-k IDs;
- compare expert outputs;
- compare final logits;
- deterministic greedy output where supported;
- record numerical tolerance for different kernels;
- run multiple prompt lengths and contexts;
- test all supported quantisation types separately.

### 15.5 Stress and soak tests
- long decode sessions;
- repeated cache churn;
- varying prompts to reduce locality;
- long prefill with broad expert activation;
- memory pressure;
- repeated load/unload;
- device reset/failure handling where testable;
- concurrent requests only after single-request correctness is stable.

---

## 16. Acceptance criteria

A release may be labelled **correctness prototype** only if:
- one declared dense model completes inference;
- one declared MoE model completes inference if MoE is claimed;
- model mappings and unsupported cases are explicit;
- reference comparisons pass;
- no selected expert is silently skipped;
- memory use is bounded and reported;
- malformed-input and fuzz tests (section 15.2) pass: every malformed fixture fails closed with a typed error and no partial state, there are zero sanitizer findings and zero crashes, and the fuzz duration is declared in advance and recorded in the release report;
- build/test instructions are reproducible.

A release may be labelled **paged MoE prototype** only if:
- selected experts can be fetched from the declared storage tier;
- cache eviction and asynchronous completion are safe;
- cold-cache inference is correct;
- disk/RAM/VRAM metrics are captured;
- no fake acceleration or silent fallback exists.

A release may be labelled **performance-qualified** only if:
- the exact model/hardware/backend profile is published;
- prefill, decode, first-token latency and memory are reported;
- cold and warm results are separated;
- repeated-run variation is shown;
- a predeclared project target is met;
- the result is compared with a documented baseline.

A report saying “not fast enough” is useful evidence but is not a completed performance release. The team must then either change the design/target and retest, or explicitly mark the performance gate as failed.

---

## 17. Kill criteria and decision points

Stop or redesign the relevant phase if:
- the model adapter cannot establish exact tensor semantics;
- the baseline output cannot match a reference;
- paging changes the values or identity of selected experts;
- the implementation requires unbounded RAM, VRAM or pinned memory;
- direct I/O performs worse than buffered/mmap access on the target workload;
- prefetch consumes bandwidth but rarely avoids a demand miss;
- cache hit rate rises while end-to-end decode becomes slower;
- the GPU path is launch-bound or synchronisation-bound enough that extra cache machinery makes no net difference;
- the optimistic transfer lower bound already exceeds the required latency;
- a custom storage format adds complexity without measurable end-to-end gain.

A failed experiment should be recorded with its configuration and result. Do not erase failed results or relabel them as successes.

---

## 18. Decisions that remain open and must not be fabricated

The specification intentionally does not guess:
- exact target GPU model or backend;
- exact 12 GiB GPU's usable VRAM after display/driver reservations;
- target CPU, RAM size, NVMe model or OS;
- first test machine: name, GPU model, whether its VRAM is dedicated, and usable RAM;
- first MoE checkpoint and exact quantisation;
- first supported model architecture beyond the planned Llama-family dense baseline;
- required context length and concurrency;
- minimum acceptable decode tokens/s and prefill tokens/s;
- whether the first GPU backend is Vulkan, CUDA or HIP;
- whether a custom model store is justified;
- whether a public RFC/prototype is merged into a particular upstream release;
- whether GPU-side caching or direct I/O is available and beneficial on the target system.

These are project inputs or measurements. If they are not specified, the implementation must record them as unknown and keep the relevant release gate closed.

---

## 19. Recommended first experiment

Use one known-good dense Llama-family GGUF to validate parsing, tokenizer, forward pass and CPU reference. Then choose one exact MoE GGUF with known-good output from an existing runtime. Before use, record that checkpoint's licence and redistribution terms, its exact quantisation and its tokenizer provenance.

For that MoE model, measure four configurations without changing any other setting:
1. reference runtime baseline;
2. this engine with all required expert weights in RAM and no GPU expert cache;
3. this engine with a small fixed GPU expert cache and host-resident remaining experts;
4. this engine with the same cache and NVMe demand paging enabled.

Record model load, prefill, decode, peak RAM/VRAM, bytes read, H2D bytes, cache hits/misses and output comparison. Only then add prefetch or change storage layout.

This experiment isolates whether the dominant limitation is model execution, expert transfer, disk I/O, cache policy, or CPU/GPU scheduling. It avoids building a custom filesystem or predictor before evidence establishes the need.

---

## 20. Final architecture decision

The proposed system is a **standalone Llama-like inference engine with a modular MoE residency layer**:

- GGUF import first.
- A validated architecture adapter and canonical tensor map.
- A conventional dense Llama-family execution path as the first correctness baseline.
- MoE routing and expert execution implemented exactly for each supported family.
- A bounded persistent VRAM expert pool, with host-resident weights and safe slot management.
- NVMe-backed demand paging only after host-resident MoE execution is correct.
- Trace-driven prefetch and GPU-side cache logic only as evidence-gated optimisations.
- Explicit memory budgets across weights, expert cache, KV, workspace and staging.
- Backend-specific I/O and transfer implementations.
- A custom deployment format only if a controlled benchmark proves it is needed.
- No Kraken dependency, no fictitious features, no unsupported performance claims, and no marking a phase complete without its tests.

The immediate engineering priority is not to maximise feature count. It is to prove one model end to end, then add one residency optimisation at a time while retaining correctness and measuring the actual bottleneck.


---

## 23. Execution plan

Scope: correctness, instrumentation and release artefacts. All benchmark runs, performance targets and kill-criteria evaluation (sections 11, 12, 17) are owned by the user and are excluded from this plan. Phases 7 and 8 are not scheduled; each requires a decision recorded in `docs/DECISIONS.md`.

Rules:
- No item in 23.2 or later starts until D1, D2, D5, D7 and D8 are complete.
- An item is complete when its "Done when" condition is met and its artefact is committed.
- Every value declared in D4 and D7 is committed before the first run that uses it.

### 23.1 Decisions

| ID | Action | Needs | Done when |
|---|---|---|---|
| D1 | Create `docs/TEST_MACHINE.md` with CPU model, RAM size, OS and version, GPU model, driver version and VRAM reported by the device query. | none | File committed; contains `lscpu` or `systeminfo` output and device query output. |
| D2 | Select the dense test checkpoint. Create `docs/CHECKPOINT_DENSE.md` with name, source URL, SHA-256, architecture string, quantisation type and licence name. Copy the licence text to `LICENSES/`. | D5 | Hash output recorded; licence file present; architecture and quantisation read from GGUF metadata by the reference tool at the D5 commit. |
| D3 | Select the MoE test checkpoint. Create `docs/CHECKPOINT_MOE.md` with the D2 fields plus expert count and top-k read from metadata. | D5 | Same condition as D2, with expert count and top-k recorded. Needed by Phase 3 only. |
| D4 | Set numeric tolerances in `docs/BENCHMARK_PROTOCOL.md` under "Tolerances": one row per data type used by D2 and D3, with absolute and relative tolerance. | D5 | Table committed with a value in every cell. Changing a value requires a commit whose message gives the reason and date. |
| D5 | Pin the reference runtime. Record commit hash, build flags, compiler version and build command in `docs/BENCHMARK_PROTOCOL.md` under "Reference". | none | Reference builds at the recorded commit; running it on one D8 prompt writes a logits dump. |
| D6 | Select the GPU backend (Vulkan, CUDA or HIP). Record choice and reason in `docs/BACKEND_SUPPORT.md`. | D1 | Entry names the backend, API version and driver version present on D1. Needed by Phase 2 only. |
| D7 | Declare parameters in `docs/BENCHMARK_PROTOCOL.md` under "Parameters": N_greedy (generated tokens compared), N_soak (tokens per soak session), F (fuzz duration in hours), T_wait (slot wait deadline in ms), G_bytes (maximum gap for read coalescing in bytes), R (repetitions, minimum 5). | none | All six values present and committed. |
| D8 | Commit `tests/reference/prompts.txt` with 20 prompts whose lengths range from 8 to 512 tokens. Generate reference token IDs with the D5 reference and commit one ID file per prompt. | D5 | 20 prompts and 20 ID files committed; SHA-256 of each file listed in `docs/BENCHMARK_PROTOCOL.md`. |
| D9 | Create `docs/DECISIONS.md`. Record every change to a D1 to D8 value with date and reason. | none | File exists; contains one entry per decision made so far. |

### 23.2 Phase 0: repository and audit

| ID | Action | Needs | Done when |
|---|---|---|---|
| P0.1 | Create the repository tree from section 5.3, containing only directories used by Phases 1 and 3. Do not create `vulkan/`, `cuda/` or `hip/`. | none | `find` output matches the listed tree; no empty directories. |
| P0.2 | Create `docs/ARCHITECTURE.md`, `docs/SUPPORTED_MODELS.md`, `docs/BENCHMARK_PROTOCOL.md`, `docs/FORMAT_AND_QUANTISATION.md`, `docs/BACKEND_SUPPORT.md`, `docs/FAILURE_MODES.md`. | P0.1 | All six files exist; `grep -n TBD docs/*` returns only lines tagged with a D-number. |
| P0.3 | In `docs/ARCHITECTURE.md`, list every source module with status `EXISTS` and its file path, or `PLANNED`. | P0.1 | Script confirms every `EXISTS` path exists and every file in `src/` is listed. |
| P0.4 | Create `LICENSES/` with the licence text of every build dependency. In `docs/ARCHITECTURE.md`, add a table of name, version and licence. | P0.1 | Script confirms table rows equal dependencies in the build file. |
| P0.5 | Make the build run from a clean checkout with one command using `-Wall -Wextra -Werror`. Record the compiler version in `docs/TEST_MACHINE.md`. | P0.1, D1 | Command exits 0 on the D1 machine. |
| P0.6 | Write `docs/PHASE0_EXIT.md` listing each command run and the SHA-256 of its output. | P0.2 to P0.5 | Re-running each listed command produces the listed hash. |

### 23.3 Phase 1: GGUF reader, inspection and dense CPU correctness (M1)

M1 exit: dense Llama-family model D2, CPU only, one quantisation type, greedy output matches the reference.

| ID | Action | Needs | Done when |
|---|---|---|---|
| P1.1 | Implement GGUF header parsing: magic, version, tensor count, metadata count. Bounds-check every read against file size. Return a typed error on failure. | D2 | Unit test truncates the header at every byte offset; each case returns a typed error; AddressSanitizer clean. |
| P1.2 | Implement metadata key-value parsing for all GGUF value types. Bound string and array lengths before allocation. | P1.1 | Unit test per value type; oversized length fields rejected before allocation. |
| P1.3 | Implement tensor directory parsing: name, dimensions, type, offset. Use overflow-checked arithmetic for every size and offset computation. | P1.1 | Tests cover dimension product overflow and offset plus size exceeding file size. |
| P1.4 | Implement block-size tables for every quantisation type used by D2. Compute tensor byte size as rows times blocks-per-row times block bytes. | P1.3 | Table-driven test; computed size equals the size implied by the tensor directory for every tensor in D2. |
| P1.5 | Implement `llama-moe-runtime model inspect <path>` producing JSON with the fields listed in section 6.2. | P1.1 to P1.4 | Output for D2 equals `tests/golden/inspect_dense.json`. |
| P1.6 | Implement the dense adapter. Map tensors to roles: embedding, attention norm, Q, K, V, O, FFN norm, FFN gate, up, down, output norm, output. A missing or unknown required tensor is a hard error that names the tensor. | P1.5 | Test removes one required tensor: error names it. Test adds one unknown tensor: error names it. |
| P1.7 | Implement the tokenizer type used by D2. Read the chat template from GGUF metadata only. | D2, P1.5 | Token IDs for all 20 prompts equal the D8 reference IDs. |
| P1.8 | Implement the forward pass: embedding, RMSNorm with epsilon from metadata, RoPE with parameters from metadata, attention with KV cache, FFN, output head. | P1.6, P1.7 | Logits for all 20 prompts are within D4 tolerance. |
| P1.9 | Implement greedy decoding. | P1.8 | For each prompt, the first N_greedy generated tokens equal the reference tokens exactly. |
| P1.10 | Create `tests/malformed_models/` with at least 50 fixtures: truncation at 10 offsets, count overflow, offset overflow, unknown quantisation type, duplicate tensor name, missing required tensor, invalid KV type, string length overflow, misaligned offset, missing tokenizer metadata. | P1.1 to P1.6 | Each fixture returns a typed error code; no crash; returned model object is null. |
| P1.11 | Build fuzz targets for the GGUF reader and metadata parser under AddressSanitizer and UndefinedBehaviorSanitizer. Save the corpus. | P1.10, D7 | Fuzzer runs for F hours with zero findings; corpus committed. |
| P1.12 | Write the reference comparison tool. For each D8 prompt it dumps engine logits and reference logits and writes `results/<run-id>/report.json` with maximum absolute difference, maximum relative difference and matching token count. | P1.9, D5 | One command produces the report for all 20 prompts. |
| P1.13 | Write `docs/M1_EXIT.md` listing every P1 item with pass status and artefact path. | P1.1 to P1.12 | Every item shows pass and an existing artefact path. |

### 23.4 Phase 2: dense GPU correctness

Entry: M1 exit complete and D6 complete.

| ID | Action | Needs | Done when |
|---|---|---|---|
| P2.1 | Implement the device query. Output: total VRAM, free VRAM at init, API version, driver version. | D6 | JSON output for D1 committed. |
| P2.2 | Implement the allocation tracker. Every allocation is tagged `weights`, `kv`, `workspace` or `staging`. | P2.1 | Test allocates and frees each tag; tracker total equals the sum of live allocation sizes at each step. |
| P2.3 | Implement dense kernels for the D2 quantisation type: matmul, RMSNorm, RoPE, attention, softmax, SwiGLU, residual add. | P2.2 | Each kernel matches the CPU reference on fixed random inputs within D4 tolerance; result table committed. |
| P2.4 | If a required kernel is missing, abort with the operation name and a fixed exit code. Permit CPU fallback only when `--allow-cpu-fallback` is set, and log each fallen-back operation. | P2.3 | Test removes one kernel: exit code matches. Test with the flag set: log has one line per fallen-back operation. |
| P2.5 | Log every operation in the forward pass with backend, input bytes and output bytes. | P2.3 | One-token log contains one entry per operation; test counts entries against the operation count. |
| P2.6 | Run the GPU forward pass for all 20 prompts. | P2.3 to P2.5, P1.9 | Logits within D4 tolerance; greedy tokens match for N_greedy. |
| P2.7 | Run three consecutive sessions of N_soak tokens each. | P2.6 | Tracker reports zero outstanding allocations at the end of each session; no device-lost events. |

### 23.5 Phase 3: one MoE family, all experts resident

Entry: M1 exit complete and D3 complete.

| ID | Action | Needs | Done when |
|---|---|---|---|
| P3.1 | Fill `docs/SUPPORTED_MODELS.md` with the D3 record: expert count, top-k, router operation, normalisation of selected weights, shared expert presence, expert tensor layout (stacked or per-expert). Cite each field to a GGUF metadata key or to a line in the reference source at the D5 commit. | D3 | Every field has a citation; no field is blank. |
| P3.2 | Implement the MoE adapter: router tensor, expert gate, up and down tensors in the layout recorded in P3.1, shared expert tensors. A missing or unknown tensor is a hard error. | P3.1 | Fixture tests cover missing router, missing expert tensor and unknown tensor. |
| P3.3 | Implement expert byte-range computation. For stacked tensors: start equals tensor offset plus expert index times expert bytes, where expert bytes comes from validated dimensions and block layout. Reject if expert bytes is not a whole number of blocks or the range exceeds the tensor. | P3.2 | Unit tests: aligned case, misaligned case, last expert, out-of-range index. |
| P3.4 | Implement the router: scores, top-k selection, normalisation as recorded in P3.1, tie-breaking as in the D5 reference. | P3.2 | Router logits and selected expert IDs match the reference for every layer for all 20 prompts. |
| P3.5 | Implement expert execution: gate and up projections, activation, down projection, weighted sum by router weights. | P3.3, P3.4 | Per-layer expert output within D4 tolerance for every layer for all 20 prompts. |
| P3.6 | Load all expert records into host memory at init. Log total bytes. | P3.3 | Logged total equals the sum of expert tensor sizes in the `model inspect` output. |
| P3.7 | Count per layer: experts selected by the router and experts executed. | P3.5 | Test asserts the two counts are equal for every layer; any mismatch aborts with a fixed exit code. |
| P3.8 | Run the full MoE forward pass for all 20 prompts. | P3.4 to P3.7 | Logits within D4 tolerance; greedy tokens match for N_greedy. |
| P3.9 | Write `docs/MOE_M3_EXIT.md` listing every P3 item with pass status and artefact path. | P3.1 to P3.8 | Every item shows pass and an existing artefact path. |

### 23.6 Phase 4: VRAM expert slot pool

Entry: P3.9 complete.

| ID | Action | Needs | Done when |
|---|---|---|---|
| P4.1 | Implement the budget calculator from section 4.2. Print every reservation. Refuse startup if the expert cache is smaller than one slot. | P3.9 | Test with a hand-computed budget: printed reservations equal the hand values. |
| P4.2 | Implement the slot pool. slot_bytes equals the largest expert record from P3.3. slot_count equals floor(cache_bytes / slot_bytes). | P4.1 | Test asserts slot_count. Startup failure output lists all reservations. |
| P4.3 | Implement the state machine exactly as the transition table in section 8.2. | P4.2 | Unit test enumerates every (state, event) pair; table rows pass; every pair not in the table is rejected and logged. |
| P4.4 | Attach a generation counter to every load. Discard and count any completion whose generation does not match the slot. | P4.3 | Test injects a stale completion: slot state unchanged; discard counter equals 1. |
| P4.5 | Implement host-to-device staging. Buffer count and size come from the budget. A slot publishes only after its upload fence signals. | P4.3 | Test delays the fence: slot state stays UPLOAD_QUEUED; compute request on that slot returns an error. |
| P4.6 | Implement LRU eviction. Eviction is permitted only for GPU_READY slots with zero references and no demand waiter on the current generation. | P4.3 | Test requests eviction of a referenced slot: refused. Test verifies victim order equals LRU order. |
| P4.7 | Implement the all-slots-busy path: wait up to T_wait, then return error code SLOTS_BUSY. | P4.6, D7 | Test pins all slots: SLOTS_BUSY returned after T_wait. |
| P4.8 | After each eviction and reload, compare the slot payload checksum with the source record checksum. | P4.6 | Test cycles every expert through the pool once; all checksums match. |
| P4.9 | Implement fault injection for transfer failure, fence timeout and simulated device loss. Map each to an error code listed in `docs/FAILURE_MODES.md`. | P4.5 | Each injected fault produces its listed code. Test asserts no CPU fallback occurred unless the flag is set. |
| P4.10 | Run all 20 prompts with cache equal to all experts resident, then with cache equal to one slot. | P4.2, P3.8 | Outputs are bit-identical on the same kernel path. |
| P4.11 | Run N_soak tokens with cache equal to one slot. | P4.10 | Zero errors; tracker reports zero outstanding allocations at the end. |

### 23.7 Phase 5: trace and replacement policy

Entry: P4.11 complete.

| ID | Action | Needs | Done when |
|---|---|---|---|
| P5.1 | Implement the trace writer as JSONL, emitting every event type listed in section 18. | P4.3 | A test run emits every event type at least once; a script checks the list. |
| P5.2 | Write a per-layer expert trace for each of the 20 prompts to `traces/`. Fields: prompt ID, token index, layer, selected expert IDs. | P5.1 | 20 trace files committed. |
| P5.3 | Implement the trace replay tool, which runs a cache simulator over a trace. | P5.2 | For a trace from an LRU run, replay hit and miss counts equal the runtime counts from that run. |
| P5.4 | Implement clock replacement. | P4.6 | Unit test: victim sequence equals the sequence in the test file. |
| P5.5 | Add config value `replacement`, accepting `lru` or `clock`. | P5.4 | Both policies pass P4.10 and P4.11. |
| P5.6 | Implement router-trace prefetch. Count issued, useful and wasted prefetch bytes. | P5.3 | Replay counters equal runtime counters for the same trace. |

### 23.8 Phase 6: NVMe-backed expert delivery

Entry: P4.11 complete.

| ID | Action | Needs | Done when |
|---|---|---|---|
| P6.1 | Implement the positional buffered reader. | P3.3 | For every expert record in D3, bytes read equal the file slice. |
| P6.2 | Implement the memory-mapped reader. | P3.3 | Same condition as P6.1. |
| P6.3 | Implement direct I/O: `O_DIRECT` on Linux, `FILE_FLAG_NO_BUFFERING` on Windows. Query alignment at runtime. Use aligned bounce buffers. Handle the final partial record. Log each fallback to buffered I/O. | P6.1 | Byte-equal reads for every record including the last. Test forces misalignment: fallback counter increments by one per forced read. |
| P6.4 | Implement a bounded async read queue. Queue depth comes from config. | P6.1 | Test asserts queue depth never exceeds the bound. |
| P6.5 | Coalesce adjacent ranges only when the gap is at most G_bytes (D7). | P6.4, D7 | Test: gap equal to G_bytes coalesces; gap equal to G_bytes plus one does not. |
| P6.6 | Implement scheduling: demand reads before prefetch reads; cancel a prefetch when its subscriber is gone; never cancel a request that has a demand subscriber. | P6.4, P5.6 | Test for each of the three rules. |
| P6.7 | Before each read batch, verify file size and modification time against the values recorded at index creation. On mismatch, refuse with error code SOURCE_CHANGED. | P1.5 | Test modifies one payload byte: next read batch refused with SOURCE_CHANGED. |
| P6.8 | Run all 20 prompts with cache equal to one slot, reading every expert through the file reader (P6.1 or P6.3). | P6.3, P4.10 | Outputs bit-identical to the resident run. |
| P6.9 | Pin only transfer staging buffers. Count pinned bytes against the budget. Unpin only after the transfer fence signals. | P4.5 | Pinned bytes never exceed the budget; unpin occurs after the fence signals. |
| P6.10 | On registration failure, fall back to pageable staging and log the event. | P6.9 | Test forces registration failure: fallback event logged; outputs bit-identical to the resident run. |

### 23.9 Release and results

| ID | Action | Needs | Done when |
|---|---|---|---|
| R1 | Write `docs/RELEASE_CHECKLIST.md` mapping each correctness-prototype criterion in section 16 to an artefact path. | P1.13, P3.9, P4.11, P6.8 | Every criterion has a path that exists. |
| R2 | Make one command build and test from a clean checkout on the D1 machine. | P0.5 | Command exits 0 on D1. |
| R3 | Write `scripts/run_record.sh` for Linux and `scripts/run_record.ps1` for Windows. Each creates `results/<run-id>/` containing the config file, commit hash, machine record and outputs. Each refuses to run without a config file. | P1.12 | Run without config exits non-zero; run with config creates the directory with all four elements. |
| R4 | Add to `docs/BENCHMARK_PROTOCOL.md`: failed runs remain in `results/` and are listed in `docs/RESULTS_INDEX.md`. | R3 | Rule committed; index lists every run directory. |
