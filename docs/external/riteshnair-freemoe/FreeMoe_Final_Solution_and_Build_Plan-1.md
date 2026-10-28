FREEMOE
# Final Solution & Build-Ready Plan
Model-agnostic tensor storage, incremental construction and staged training
Platform
Windows x64, native C++ and CMake
Core constraint
No mandatory PyTorch dependency
Governor candidate
minicpm5-1b.i1-q6_k.gguf
First target
Small GGUF model already supported by the existing engine
First gate
GGUF inspection and native tensor-blob round-trip
# FreeMoe: Final Solution and Build-Ready Implementation Plan
Version: 1.0
Date: 28 October 2026
Platform: Windows x64, native C++ and CMake
Core constraint: no mandatory PyTorch dependency
Starting governor candidate: minicpm5-1b.i1-q6_k.gguf
Initial target: a small existing GGUF model, ideally around 135M parameters if the current engine supports its architecture
Purpose: turn the FreeMoe idea into a staged, testable implementation that an AI coding agent can build.
## 1. Executive summary
FreeMoe is intended to be a model-construction system that separates a model's logical architecture from the physical files that store its tensors. It should reuse existing trained models where possible, support incremental and recoverable tensor updates, and eventually allow sparse MoE models whose experts can be stored and paged independently.
The first practical step is not to train a large model and not to build every subsystem at once. The user already has a partially built Llama-like C++ inference engine and a MiniCPM5-1B GGUF that may act as a governor. The initial work is to inspect the existing engine, determine how it loads GGUF, add safe tensor inspection and a native binary tensor-blob store, then round-trip a small supported GGUF model while proving the original model still runs.
The governor, target model, training engine and storage layer have distinct roles:
**Governor:** MiniCPM5-1B GGUF, run by a compatible inference runtime. It proposes tasks and calls allow-listed tools.
**Teacher/source model:** an existing GGUF model from which FreeMoe can collect outputs and inspect metadata/tensor layout. Its original file remains immutable.
**Target model:** a separate model package being converted, adapted or eventually trained.
**Blob store:** native storage for actual tensor bytes plus metadata, checksums and version information.
**Training engine:** a separate component that performs real forward/backward/optimiser operations. Writing tensors to blobs alone is not learning.
**MCP server:** a later automation interface that exposes safe, structured operations. MCP orchestrates tools; it does not implement training mathematics.
The first release should prove inspection, extraction, byte-correct round-trip, integrity checking, safe versioning and preserved inference. Only then add training, MCP automation, and MoE construction.
## 2. The intended end state
FreeMoe should ultimately support this lifecycle:
Load a small governor model for task orchestration.
Load one or more source/teacher models when needed.
Inspect source GGUF metadata, tensor names, dimensions, data types and quantisation encoding.
Collect teacher-generated examples and, where available, additional training signals.
Build a separate target model in FreeMoe's native tensor-blob package.
Train or adapt the target using a compatible native training backend.
Write recoverable tensor/checkpoint versions to disk in bounded chunks.
Evaluate the candidate and retain the last known-good version if it regresses.
Optionally assemble/export the final model to GGUF when its architecture and tensor encodings are supported.
10. Use the resulting model for inference, including eventual disk-resident MoE expert paging.
The system should be model-agnostic at the storage and orchestration interfaces. Model import, execution, training and export still require architecture-specific adapters and supported tensor encodings.
## 3. Important technical boundaries
### 3.1 A GGUF is not a training dataset
GGUF contains metadata and tensor data, often quantised. It does not contain a neatly recoverable database of the model's facts. Tensor inspection reveals structure and numerical parameters, not explicit prose knowledge.
### 3.2 IMatrix is not learned knowledge
An importance matrix is calibration/statistical information used to guide quantisation. It is not a substitute for model weights, a training corpus or a distillation dataset.
### 3.3 Writing blobs is not training
A model learns when a training algorithm computes a valid objective and changes the appropriate parameters. Extracting tensors, changing their file format, or writing the same bytes into a new package proves storage correctness, not learning.
### 3.4 Dense-to-MoE conversion is not a file conversion
A small dense model, such as a 135M instruct model, is a useful first target for storage and inference validation. Turning it into an MoE requires defining expert modules, routing, compatible dimensions and a training or initialisation strategy. The parameter count alone does not create useful experts. The first MoE should be a separately specified experimental architecture, not a promise that arbitrary dense weights can be mechanically split into a capable MoE.
### 3.5 Disk-first is bounded-memory, not memory-free
Active tensors, intermediate activations, gradients and optimiser state still require working memory. The goal is to avoid requiring the entire model and all training state to be resident simultaneously. Whether disk-first training is faster or more efficient must be measured.
## 4. First implementation target
### Phase A: inspect the existing engine
Before changing code, the coding agent must inspect the current repository and report:
how GGUF files are opened and parsed;
which model architectures and tensor types are actually supported;
whether the current inference path can load two independent model instances;
how tensor data is represented in memory;
whether quantised tensors can be read without dequantising;
current test and benchmark commands;
known gaps and any existing tensor-inspection or tracing facilities.
Do not assume the engine supports arbitrary Llama-family models just because it is called a Llama-like engine. Confirm support from code and tests.
### Phase B: governor and target roles
Governor candidate: `minicpm5-1b.i1-q6_k.gguf`.
Target/source test model: choose a small GGUF whose architecture is already supported by the existing engine. A roughly 135M instruct model is attractive if a compatible file is available.
Keep governor and target instances separate. If simultaneous loading exceeds available resources, allow sequential loading or unloading.
The governor must not need to know the internal blob layout. It calls structured tools.
The original GGUF must remain read-only and unchanged.
### Phase C: native tensor blob round-trip
Implement a read-only GGUF inspection/import path for supported tensor encodings, then store the tensor bytes in FreeMoe's native blob package. Verify that exported/reconstructed data matches the source representation exactly where the conversion is byte-preserving.
If the source uses quantised blocks, preserve the encoding and block metadata. Do not silently dequantise and requantise while claiming byte-exact round-trip. If the first implementation cannot represent a tensor type, fail explicitly and record it as unsupported.
### Phase D: preserve inference
After import, run the original model and the blob-backed representation on identical prompts, settings and runtime paths. If the blob store is only a storage layer and the inference engine decodes the same tensor representation, outputs should match under a deterministic configuration. If numerical or sampling differences are expected, document and test the permitted tolerance.
### Phase E: tiny training proof
Only after the storage gates pass, implement a tiny CPU model in native C++ or a supported training subset. Validate gradients, prove the loss improves on a deterministic task, save all necessary training state and resume after interruption. This tiny task proves the training pipeline; it does not prove that a transformer or MoE can already be trained.
## 5. Native tensor blob format
A blob is a binary file containing actual tensor bytes. It is not merely a list of paths. A blob may contain multiple tensors or tensor tiles; a logical tensor may span multiple blobs; an expert may span several blobs. Physical file boundaries must not define the model architecture.
### 5.1 Proposed logical layout
Header: magic, format version, header size, blob ID, directory/payload offsets and lengths, flags.
Tensor directory: stable tensor ID/name, dimensions, data type, encoding/quantisation type, payload offset, stored length, logical length if different, checksum.
Payload: raw tensor bytes or a documented encoded tensor representation.
Integrity metadata: checksums and commit/version metadata.
Manifest: maps logical tensor IDs and versions to blob ranges.
Use fixed-width fields and explicit byte order. Do not serialise native C++ structs directly, because padding and ABI layout are not a stable file format.
### 5.2 Blob size
512 MiB or 1 GiB can be a later configurable shard target. Initial tests should use tiny files. Blob size is a storage policy, not a tensor, layer or expert boundary.
### 5.3 Versioning and recovery
Never overwrite the only known-good model.
Write to staging files or copy-on-write blobs.
Validate the staged payload and checksums before commit.
Publish a new manifest generation only after all referenced data is verified.
Keep the previous valid generation until the new version has passed verification.
Store training state separately: optimiser state, step, RNG state where relevant, dataset position and configuration/model-definition hashes.
Test process interruption and corruption. Treat power-loss durability as a separate claim requiring platform-specific testing.
## 6. Governor and MCP
MCP should be added after stable native APIs exist. Start with direct C++ APIs and CLI commands, then expose them through an MCP adapter.
Candidate allow-listed tools:
`inspect_model`
`list_tensors`
`blob_import`
`blob_verify`
`model_validate`
`create_training_job`
`training_status`
`checkpoint`
`resume_job`
`run_evaluation`
`compare_candidate`
`export_model`
Each tool must have a strict schema and bounded arguments. The governor must not execute arbitrary shell commands, choose arbitrary filesystem paths, certify its own output or overwrite the last known-good model. The controller enforces policy in deterministic code.
## 7. Training from existing models
FreeMoe can use several transfer methods, but they are not interchangeable:
**Teacher-generated examples:** most portable across architectures. Query source models, validate answers, and train the target on examples.
**Logit/hidden-state distillation:** possible only when the source runtime exposes suitable signals and the target training objective supports them.
**Compatible weight initialisation or tensor transfer:** possible when tensor semantics, shapes and architecture are compatible and verified.
**Quantisation-guided conversion:** IMatrix may help choose quantisation parameters, but it does not transfer knowledge into a new model by itself.
The initial goal is a working storage pipeline, not a claim that a 1B governor can automatically extract every capability from a 7B teacher.
## 8. Proposed repository layout
FreeMoe/  CMakeLists.txt  README.md  docs/    FINAL_SOLUTION.md    BLOB_FORMAT.md    IMPLEMENTATION_GATES.md    RISKS_AND_BASELINES.md  agent/    CODING_AGENT_BRIEF.md    PROJECT_RULES.md  schemas/    model-definition.schema.json    manifest.schema.json  include/freemoe/    model_definition.hpp    tensor.hpp    blob_store.hpp    manifest.hpp    gguf_adapter.hpp    training_engine.hpp    job_controller.hpp  src/    model_definition/    tensor/    blob_store/    manifest/    gguf_adapter/    training/    evaluation/    controller/    cli/    mcp_adapter/       # later    exporter/          # later  examples/    tiny-model.json    tiny-dataset.jsonl  tests/    gguf_inspection/    blob_roundtrip/    corruption_recovery/    inference_parity/    train_resume/
This is a recommended layout. The coding agent must first inspect the existing repository and adapt to its real structure instead of duplicating or deleting existing components.
## 9. Implementation gates
### Gate 0: repository reconnaissance and clean build
Inspect the current code.
Record actual support and gaps.
Build Debug and Release using documented Windows commands.
Run existing tests before editing.
Pass: baseline is reproducible and the implementation plan is grounded in the real code.
### Gate 1: GGUF inspection
Read metadata and tensor directory for one supported small model.
Report tensor names, shapes, types, quantisation encodings and byte ranges.
Do not change the source file.
Pass: inspection output matches the source model metadata and invalid files fail safely.
### Gate 2: blob store and round-trip
Implement the native blob header, directory, payload, checksums and manifest.
Import supported tensor data without changing its representation.
Verify source-to-blob-to-reconstructed tensor byte equality.
Pass: exact byte equality for supported, byte-preserved tensors; unsupported tensor encodings fail explicitly.
### Gate 3: inference parity
Run original GGUF and blob-backed tensors with identical prompts and settings.
Compare logits if available; otherwise compare deterministic output and document the weaker guarantee.
Verify the original GGUF remains unchanged.
Pass: agreed parity criteria pass and tests are repeatable.
### Gate 4: transactional recovery
Add staging writes, journal/manifest generations and recovery.
Inject truncation, checksum errors and process interruption.
Pass: recovery selects the last valid committed version and never promotes incomplete data.
### Gate 5: tiny native training and resume
Implement a tiny linear model or MLP, deterministic dataset, forward/loss/backward/optimiser.
Add finite-difference gradient checks and checkpoint/resume.
Pass: valid gradients, measurable held-out improvement and resumed run matching uninterrupted run within documented tolerance.
### Gate 6: governor tools and MCP
Wrap the stable APIs in structured tools.
Add a MiniCPM5 inference adapter using a runtime that supports the model.
Enforce tool allow-list and path/resource policy.
Pass: malformed calls are rejected, jobs are auditable and resumable.
### Gate 7: small pretrained target and adaptation
Choose a model architecture actually supported by the runtime and training engine.
Use a suitable trainable checkpoint or implement a clearly defined adapter-training path.
Do not assume an inference-only GGUF can be backpropagated through.
Pass: training changes weights, evaluation demonstrates improvement, and inference still works.
### Gate 8: experimental MoE and Vulkan
Specify an explicit MoE architecture, expert shapes and routing.
Establish how experts are initialised and trained.
Add Vulkan only after CPU correctness, one operation at a time.
Pass: correctness, recovery, expert routing and performance are independently measured.
## 10. Required CLI surface
These are proposed commands, not claims that they already exist:
freemoe.exe versionfreemoe.exe self-testfreemoe.exe inspect-gguf --model "path\\to\\small-model.gguf"freemoe.exe blob-import --model "path\\to\\small-model.gguf" --out "models\\candidate.fmp"freemoe.exe blob-verify --model "models\\candidate.fmp"freemoe.exe model-validate --definition "examples\\tiny-model.json"freemoe.exe train --model "examples\\tiny-model.json" --dataset "examples\\tiny-dataset.jsonl"freemoe.exe resume --checkpoint "runs\\run-0001\\checkpoint.json"freemoe.exe evaluate --run "runs\\run-0001"
The coding agent should only implement commands for the current gate. Do not create no-op commands that pretend to perform unsupported operations.
## 11. Metrics and diagnostics
Record:
model and definition IDs/hashes;
run ID, step ID and operation;
tensor IDs and byte ranges;
read/write bytes, cache hits and commit latency;
RAM peak, VRAM use where applicable, CPU/GPU time;
training loss and held-out metrics;
checksum and validation outcomes;
recovery decision and error category.
Use the user's Sherlock-style tracing approach to find the actual bottleneck rather than assuming it is storage or compute.
## 12. Risks and stop/go decision
The file format itself is not automatically novel. Chunked storage, tensor formats, CPU training, offload, LoRA and tool orchestration all have established precedents. FreeMoe's potential value is the integration of recoverable tensor construction, bounded-memory updates, architecture-independent indexing, expert-aware storage and safe automation, if this offers a measured advantage.
Continue if the storage pipeline is correct, recoverable and useful.
Narrow the scope if it is useful as a tensor store but does not improve training efficiency.
Redesign or stop if the custom format adds significant complexity without a measurable benefit over an established format.
## 13. Tomorrow's checklist
Open the existing Llama-like repository and ask the coding agent to inspect before changing files.
Build and run the existing tests; record the baseline.
Confirm the exact MiniCPM5 GGUF path and whether the current inference runtime loads it.
Pick a small target GGUF whose architecture the engine demonstrably supports.
Implement tensor inspection only.
Add blob import/verify for the first supported tensor types.
Prove exact round-trip and preserve original inference.
Stop and review results before adding training or MCP.
The first goal is not “create an intelligent MoE overnight”. It is to establish a correct bridge from existing model files to FreeMoe's own tensor store. That is the foundation on which genuine training and MoE construction can be built.
