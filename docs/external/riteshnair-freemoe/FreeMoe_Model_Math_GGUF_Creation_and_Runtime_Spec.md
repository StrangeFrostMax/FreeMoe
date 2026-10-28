# FreeMoe: Complete Model Mathematics, GGUF Schema, Conversion, Inspection and Runtime Specification

**Status:** implementation-ready proposal for a first student-model experiment and a separate disk-resident MoE runtime.  
**Repository:** https://github.com/riteshnair/FreeMoe  
**Important:** this document defines the proposed production model and file contract. It does not claim that the student has been trained or that a production converter or physical GPU backend already exists.

## 0. What this project is building

FreeMoe has two related but separate deliverables. Do not merge their acceptance criteria.

### A. FreeMoe-Coder: a new compact coding MoE

Train or distil a new Llama-like causal language model with sparse MoE feed-forward layers. Initial target: 1B–2B total parameters, approximately 700M active parameters under the stated accounting convention, and a deployment profile intended to fit comfortably within an 8 GB GPU budget. The initial release is text-only.

### B. FreeMoe Runtime: execute larger existing MoE models with bounded residency

Load a supported existing MoE checkpoint, potentially 27B–32B total parameters, without requiring all expert weights to be in VRAM. Keep the original model's learned weights, router and mathematical operations intact. The runtime fetches the selected experts from a cold source on NVMe, stages data through bounded host RAM and places active experts in a bounded device-resident slot pool when a real GPU backend exists.

**Chunking is not distillation.** The large-model runtime pages the original model's expert tensors. FreeMoe-Coder is a separate model with newly trained/distilled weights.

### C. Explicitly retained project principles

- MoE-only feed-forward blocks: no conventional dense FFN and no shared expert in the first student version.
- Dense attention, embeddings, normalisation and output projection are still required Transformer operations. “MoE-only” refers to the FFN path, not to making every operation an expert.
- GGUF is the initial container format. It stores metadata and tensor data; it does not itself implement an architecture or inference engine.
- The model file contains immutable weights and metadata. Runtime code, kernels and routing execution live in the engine.
- A separate LMDB knowledge/learning cache may hold externally editable records. It is not silently mixed into the base checkpoint.
- Every selected expert must execute or inference must fail. Never silently skip an expert, substitute zeros, or claim a GPU backend when falling back to CPU.
- Keep the synthetic `freemoe-tiny-v1` fixture family as a stable correctness test. The production student is a different versioned model contract.

---

# 1. Decisions, proposals and unresolved inputs

## 1.1 Previously established project direction

| Item | Direction |
|---|---|
| Model family | Llama-like causal decoder |
| Student purpose | Coding-specialised |
| Student topology | Sparse MoE FFN in every Transformer layer |
| Teacher | A larger coding-capable model in the 27B/32B class was discussed |
| Student size target | Roughly 1B–2B total parameters |
| Active parameter target | Around 700M, depending on output-head accounting |
| Student deployment | Intended for an 8 GB GPU profile |
| Large-model inference | Preserve original MoE model and page experts from NVMe through RAM to VRAM |
| Initial container | GGUF |
| Data/learning cache | Separate LMDB store, initially discussed at about 2 GB |
| Runtime principle | Router selects experts; residency manager makes selected expert weights available |
| Model inspection | A `WAKE_UP`-style report that examines metadata, tensors, tokenizer, compatibility, size and integrity |
| Validation style | CPU oracle, exact fixtures, differential tests, hashes, trace events and benchmark evidence |

## 1.2 Proposed FreeMoe-Coder v0 configuration

These dimensions are a concrete starting proposal, not a historical claim that a trained checkpoint exists.

| Parameter | Proposed value |
|---|---:|
| `model_type` | `freemoe-coder` |
| `model_version` | `1` |
| Transformer layers `L` | 20 |
| Hidden width `d_model` | 1,280 |
| Query heads `n_heads` | 20 |
| Key/value heads `n_kv_heads` | 4 |
| Head dimension `d_head` | 64 |
| Attention type | Causal grouped-query attention (GQA) |
| FFN type | Sparse MoE SwiGLU in every layer |
| Experts per layer `E` | 8 |
| Selected experts per token `K` | 2 |
| Expert intermediate width `d_ff` | 1,920 |
| Shared expert | None |
| Dense FFN fallback | None |
| Normalisation | Pre-norm RMSNorm |
| RMSNorm epsilon | 1e-5 |
| Position encoding | RoPE, initial theta 10,000 |
| RoPE scaling | None in v0 |
| Biases | None in linear layers in v0 |
| Embeddings | Token embedding matrix |
| Output head | Tied to token embeddings in v0 |
| Activation | SwiGLU: `SiLU(gate) * up` |
| Router | Linear logits, stable top-2, softmax over selected logits |
| Context target | 4,096 initially; 8,192 only after tests |
| Tokenizer | Must be selected and frozen before training; preferably reuse the exact teacher tokenizer for the first distillation experiment |
| Modality | Text-only |
| Initial training/export dtype | F32/F16/BF16 as supported by the chosen training stack; export/quantisation is a separate stage |
| Runtime quantisation target | F16 baseline, then Q8_0/Q6_K/Q4_K experiments after kernels exist |

**Tokenizer is not decided by the dimensions above.** A vocabulary size must come from the selected tokenizer, not from a guess. The parameter arithmetic below uses a clearly labelled example vocabulary of 250,000 solely to illustrate the scaling.

---

# 2. Mathematical definition of FreeMoe-Coder v0

## 2.1 Symbols and shapes

- Batch size: \(B\)
- Sequence length: \(T\)
- Hidden width: \(d=1280\)
- Layers: \(L=20\)
- Query heads: \(H_q=20\)
- KV heads: \(H_{kv}=4\)
- Head width: \(d_h=64\)
- Experts per layer: \(E=8\)
- Top-k experts: \(K=2\)
- Expert intermediate width: \(m=1920\)
- Vocabulary size: \(V\), determined by the chosen tokenizer
- Hidden state at layer \(\ell\): \(X_\ell\in\mathbb{R}^{B\times T\times d}\)

All tensor orientation in the converter must be explicit. The mathematical shapes below use input-by-output projection notation; the runtime may store matrices transposed if its kernels require it, but the converter must record that transform and test it.

## 2.2 Token embedding

For token IDs \(I\in\{0,\ldots,V-1\}^{B\times T}\):

\[
X_0 = \operatorname{Embedding}(I, W_{\mathrm{emb}})
\]

with:

\[
W_{\mathrm{emb}}\in\mathbb{R}^{V\times d}
\]

Use the exact tokenizer's BOS/EOS/PAD/UNK and any special-token IDs. Do not invent IDs or assume that token ID zero is padding.

## 2.3 Pre-norm Transformer layer

For layer \(\ell\), first apply RMSNorm and causal self-attention:

\[
U_\ell = \operatorname{RMSNorm}(X_\ell;\gamma^{(1)}_\ell,\epsilon)
\]

\[
Q=U_\ell W_Q,\quad K=U_\ell W_K,\quad V=U_\ell W_V
\]

Projection shapes in the stated convention:

- \(W_Q\in\mathbb{R}^{d\times(H_qd_h)}=\mathbb{R}^{1280\times1280}\)
- \(W_K\in\mathbb{R}^{d\times(H_{kv}d_h)}=\mathbb{R}^{1280\times256}\)
- \(W_V\in\mathbb{R}^{d\times(H_{kv}d_h)}=\mathbb{R}^{1280\times256}\)
- \(W_O\in\mathbb{R}^{(H_qd_h)\times d}=\mathbb{R}^{1280\times1280}\)

Apply RoPE to Q and K using the configured rotary dimensions and theta. v0 uses the standard half-rotation convention, full head dimension 64, theta 10,000, and no scaling. The converter must record the exact RoPE convention expected by the runtime.

Causal GQA attention:

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+M_{\mathrm{causal}}\right)
\]

Each group of \(H_q/H_{kv}=5\) query heads shares one KV head. Attention output is:

\[
C=\operatorname{ConcatHeads}(AV)W_O
\]

with residual:

\[
Y_\ell=X_\ell+C
\]

For incremental decoding, append new K/V rows to the KV cache; do not recompute old keys/values. Cache layout, cache dtype, context position, RoPE position and cache reset rules must be tested separately.

## 2.4 MoE-only feed-forward block

Normalise the attention result:

\[
Z_\ell=\operatorname{RMSNorm}(Y_\ell;\gamma^{(2)}_\ell,\epsilon)
\]

Router logits for token hidden state \(z\in\mathbb{R}^{d}\):

\[
r=zW_R+b_R
\]

v0 has no router bias, so \(b_R=0\), and:

\[
W_R\in\mathbb{R}^{d\times E}=\mathbb{R}^{1280\times8}
\]

Select the top \(K=2\) expert IDs by router logit. Ties must be deterministic: lower expert ID wins. Softmax is applied only to selected logits:

\[
S=\operatorname{TopK}(r,K)
\]

\[
g_i=\frac{\exp(r_i-\max_{j\in S}r_j)}
{\sum_{j\in S}\exp(r_j-\max_{j\in S}r_j)},\quad i\in S
\]

For expert \(i\), with separate gate/up/down matrices:

\[
W^{(i)}_{\mathrm{gate}},W^{(i)}_{\mathrm{up}}
\in\mathbb{R}^{d\times m}
\]

\[
W^{(i)}_{\mathrm{down}}\in\mathbb{R}^{m\times d}
\]

\[
\operatorname{Expert}_i(z)=
\left(\operatorname{SiLU}(zW^{(i)}_{\mathrm{gate}})
\odot(zW^{(i)}_{\mathrm{up}})\right)W^{(i)}_{\mathrm{down}}
\]

The MoE output is the weighted sum of all selected experts:

\[
F_\ell(z)=\sum_{i\in S}g_i\operatorname{Expert}_i(z)
\]

Final residual:

\[
X_{\ell+1}=Y_\ell+F_\ell(Z_\ell)
\]

There is no dense FFN bypass and no shared expert in v0. Every token must execute exactly two selected experts per layer. If an expert cannot be loaded or executed, return an explicit error.

## 2.5 Final normalisation and tied output head

\[
H=\operatorname{RMSNorm}(X_L;\gamma_f,\epsilon)
\]

Tied output logits:

\[
\operatorname{logits}=H W_{\mathrm{emb}}^\top
\]

where \(W_{\mathrm{emb}}\) is the same matrix used for token embeddings. Tying must be recorded explicitly and validated. If training produces untied weights, either reject them for v0 or define a new model version; do not silently discard the output matrix.

Sampling is a runtime concern, not part of the stored weights. v0 should first validate greedy argmax, then add temperature/top-p/top-k as separate features.

## 2.6 Router training loss and expert-balance controls

The base next-token objective is causal cross-entropy:

\[
\mathcal{L}_{\mathrm{LM}}=-\sum_t
\log p_\theta(x_t\mid x_{<t})
\]

A router auxiliary load-balancing loss may be used during training, but it must be precisely defined in the training configuration and stored in the training manifest. It is not an inference-time model operation. Record router entropy, per-expert token counts, fraction of dead experts, maximum load fraction and selected-expert distribution. Do not claim a specific auxiliary-loss coefficient until the training experiment selects it.

A distillation run can combine the language-model objective with teacher distributions:

\[
\mathcal{L}=\alpha\mathcal{L}_{\mathrm{LM}}
+\beta\tau^2\,\mathrm{KL}
\left(p_{\mathrm{teacher}}^\tau\Vert p_{\mathrm{student}}^\tau\right)
+\lambda\mathcal{L}_{\mathrm{router}}
\]

where \(\alpha,\beta,\lambda,\tau\) are recorded experiment parameters, not silently fixed constants. When teacher and student tokenizers differ, token-by-token logit KL is not directly aligned. For the first experiment, use the same tokenizer if licensing and training setup permit.

---

# 3. Parameter-count mathematics and memory estimates

## 3.1 Expert parameters

One SwiGLU expert has three matrices:

\[
P_{\mathrm{expert}}=3dm
=3(1280)(1920)=7{,}372{,}800
\]

All eight experts in one layer:

\[
P_{\mathrm{experts/layer}}=E P_{\mathrm{expert}}
=8(7{,}372{,}800)=58{,}982{,}400
\]

Across 20 layers:

\[
P_{\mathrm{experts,total}}=L E 3dm
=1{,}179{,}648{,}000
\]

## 3.2 Attention parameters

Per layer, ignoring biases:

\[
P_{\mathrm{attention/layer}}
=d(d)+d(H_{kv}d_h)+d(H_{kv}d_h)+(H_qd_h)d
\]

\[
=1280^2+1280(256)+1280(256)+1280^2
=3{,}932{,}160
\]

Across 20 layers:

\[
P_{\mathrm{attention,total}}=78{,}643{,}200
\]

Router weights:

\[
P_{\mathrm{router}}=L d E=20(1280)(8)=204{,}800
\]

RMSNorm parameters are small relative to the expert matrices. Exact counts depend on whether each layer has one or two learned norm vectors plus final norm; the v0 definition has two per layer and one final vector.

## 3.3 Total parameter estimate

With tied embeddings and example vocabulary \(V=250{,}000\):

\[
P_{\mathrm{emb}}=Vd=250{,}000(1280)=320{,}000{,}000
\]

Approximate total:

\[
P_{\mathrm{total}}\approx
P_{\mathrm{experts,total}}+
P_{\mathrm{attention,total}}+
P_{\mathrm{router}}+
P_{\mathrm{emb}}+
P_{\mathrm{norms}}
\]

\[
P_{\mathrm{total}}\approx1.579\text{ billion}
\]

This is an example only. The actual vocabulary size and exact tokenizer determine embedding parameter count. If \(V=32{,}000\), the embedding matrix is only 40.96M parameters, and total size falls substantially.

## 3.4 What “700M active” means

Per token, top-2 expert execution uses:

\[
P_{\mathrm{experts,active/token}}=L K 3dm
=20(2)(7{,}372{,}800)=294{,}912{,}000
\]

Add all attention matrices and routers:

\[
P_{\mathrm{active,core}}\approx294.912M+78.643M+0.205M
\approx373.76M
\]

If counting the full tied output projection over a 250k vocabulary on every token, add approximately 320M output weights, giving about 694M parameter uses under that convention. The embedding lookup itself touches one row, not the entire embedding table. State the accounting convention whenever quoting active parameters; the phrase “700M active” is not meaningful without it.

## 3.5 Approximate raw weight storage

Ignoring metadata, alignment, quantisation scales, tokenizer assets and temporary buffers:

| Representation | Rough raw weight size for 1.58B parameters |
|---|---:|
| F32 | 6.32 GB decimal |
| F16/BF16 | 3.16 GB decimal |
| 8-bit idealised | 1.58 GB decimal |
| 4-bit idealised | 0.79 GB decimal |

Real quantised formats have block scales/minimums and format overhead. Use actual tensor byte sizes, not `parameter_count * bits/8`, for final memory accounting.

The proposed student should fit in an 8 GB VRAM deployment after weights, KV cache, scratch/workspace, command buffers and allocator reserve are included. Validate this on real hardware rather than inferring it from raw weight size.

## 3.6 KV cache

For F16 K and V, one sequence, batch 1, context \(T\):

\[
M_{\mathrm{KV}}=2 L T H_{kv}d_h\cdot2\text{ bytes}
\]

At \(L=20,T=4096,H_{kv}=4,d_h=64\):

\[
M_{\mathrm{KV}}=83{,}886{,}080\text{ bytes}\approx80\text{ MiB}
\]

At 8192 tokens this doubles to about 160 MiB. Multiply by batch/sequence count and adjust for cache dtype, allocator alignment and any sliding-window policy. v0 does not use sliding-window attention.

---

# 4. Model configuration file

The canonical configuration must be saved beside training outputs and embedded as GGUF metadata where representable. Keep a machine-readable JSON copy in the checkpoint directory.

Illustrative `config.json`:

```json
{
  "model_type": "freemoe-coder",
  "model_version": 1,
  "architectures": ["FreeMoeForCausalLM"],
  "hidden_size": 1280,
  "num_hidden_layers": 20,
  "num_attention_heads": 20,
  "num_key_value_heads": 4,
  "head_dim": 64,
  "intermediate_size": 1920,
  "num_local_experts": 8,
  "num_experts_per_tok": 2,
  "moe_every_layer": true,
  "shared_expert_count": 0,
  "dense_ffn_layers": 0,
  "hidden_act": "silu",
  "ffn_type": "swiglu",
  "norm_type": "rmsnorm",
  "rms_norm_eps": 1e-5,
  "position_embedding_type": "rope",
  "rope_theta": 10000.0,
  "rope_scaling": null,
  "attention_bias": false,
  "mlp_bias": false,
  "tie_word_embeddings": true,
  "max_position_embeddings": 4096,
  "vocab_size": "MUST_MATCH_TOKENIZER",
  "tokenizer_id": "MUST_BE_PINNED",
  "tokenizer_revision": "MUST_BE_PINNED",
  "tokenizer_sha256": "MUST_BE_COMPUTED",
  "bos_token_id": "MUST_MATCH_TOKENIZER",
  "eos_token_id": "MUST_MATCH_TOKENIZER"
}
```

The quoted placeholder strings are intentionally invalid for a training config. Replace them with actual integers/IDs/hashes and validate before training or conversion. Never write a made-up vocabulary size.

Recommended accompanying files:

```text
model/
  config.json
  tokenizer.json                 # or exact tokenizer asset set required by source
  tokenizer_config.json
  special_tokens_map.json
  generation_config.json          # if applicable; sampling defaults are not weights
  model.safetensors.index.json    # if sharded source checkpoint
  model-00001-of-000NN.safetensors
  ...
  training_manifest.json
  data_manifest.json
  licence_and_provenance.md
  SHA256SUMS
```

Exact filenames depend on the selected training/export framework. The invariant is that all tokenizer assets, configuration, shards, licences, hashes and provenance are retained.

---

# 5. Tensor schema for `freemoe-coder-v1`

Use a canonical logical tensor registry independent of any source framework's naming. The exporter maps source names into this registry; the loader maps the GGUF names to engine roles.

## 5.1 Required logical tensor roles

For every layer `L` from 0 through 19:

| Logical role | Logical shape | Notes |
|---|---|---|
| `token_embd.weight` | `[V, 1280]` | Shared with output head |
| `blk.L.attn_norm.weight` | `[1280]` | Pre-attention RMSNorm |
| `blk.L.attn_q.weight` | `[1280, 1280]` | 20 query heads |
| `blk.L.attn_k.weight` | `[1280, 256]` | 4 KV heads |
| `blk.L.attn_v.weight` | `[1280, 256]` | 4 KV heads |
| `blk.L.attn_output.weight` | `[1280, 1280]` | Attention output projection |
| `blk.L.ffn_norm.weight` | `[1280]` | Pre-MoE RMSNorm |
| `blk.L.router.weight` | `[1280, 8]` | Router logits |
| `blk.L.experts_gate.weight` | `[8, 1280, 1920]` | Stacked expert gate matrices |
| `blk.L.experts_up.weight` | `[8, 1280, 1920]` | Stacked expert up matrices |
| `blk.L.experts_down.weight` | `[8, 1920, 1280]` | Stacked expert down matrices |
| `output_norm.weight` | `[1280]` | Final RMSNorm |

These are **logical shapes**. GGUF tensor dimension order (`ne[]`) and matrix orientation must be chosen once and tested against the writer/reader; do not assume that the displayed logical order maps directly to serialized `ne[]` order. Document the convention in the adapter and test every tensor's logical and serialized shape.

For the tied output head, no duplicate weight tensor is required if the adapter supports the tie metadata and references `token_embd.weight`. If a downstream tool requires an explicit output tensor, materialise it only in a conversion mode that records the duplication and verifies equality.

## 5.2 Tensor validation rules

- Exact required tensor-role set for `freemoe-coder-v1`; no unknown tensor accepted unless the version explicitly allows it.
- Every layer index must appear exactly once for every required per-layer role.
- All dimensions, products and byte sizes checked with overflow-safe arithmetic.
- Tensor extents aligned to GGUF alignment and wholly inside the data section.
- No overlap between unrelated tensor extents.
- Stacked expert axis must be exactly 8; every expert slice must have the expected byte length.
- All expert weights and router tensors finite where checked; reject NaN/Inf in a strict validation mode.
- Norm epsilon, RoPE settings, tokenizer hash and tied-weight semantics must match the config.
- Quantisation type is recorded per tensor; unsupported execution types are reported as unsupported rather than interpreted as F32.
- Every tensor gets SHA-256 in an external conversion manifest. Expert payloads also get per-expert SHA-256 in the manifest for paging/integrity tests. A fast non-cryptographic checksum can be used in the hot path only as an additional check, not as provenance proof.

---

# 6. GGUF creation and conversion contract

## 6.1 Architecture identifier and compatibility

For the new student, propose GGUF `general.architecture = "freemoe"` and a namespaced version key such as `freemoe.model_version = 1`. This is a **custom architecture ID**. A generic GGUF reader can parse the file, but llama.cpp or another runtime will not automatically execute it merely because it is valid GGUF. FreeMoe must implement the adapter and operations. If upstream compatibility is later required, submit/maintain a real upstream architecture adapter or convert to an already-supported architecture only if the mathematical semantics truly match.

Do not use the synthetic architecture string `freemoe-tiny` for the production student. Keep:
- `freemoe-tiny-v1`: synthetic correctness fixtures;
- `freemoe-coder-v1`: proposed trained student;
- each imported third-party architecture: its actual architecture and its own adapter contract.

## 6.2 Proposed GGUF metadata

Use standard GGUF keys where a standard key accurately expresses the field. Use namespaced `freemoe.*` keys for FreeMoe-specific semantics. The exact standard key spelling must be confirmed against the version of GGUF/llama.cpp used by the exporter.

Required semantic fields:

```text
general.architecture = "freemoe"
general.name = "FreeMoe-Coder"
general.file_type = <actual encoded file type>
general.quantization_version = <when applicable>

freemoe.model_version = 1
freemoe.model_kind = "moe"
freemoe.hidden_size = 1280
freemoe.num_hidden_layers = 20
freemoe.num_attention_heads = 20
freemoe.num_key_value_heads = 4
freemoe.head_dim = 64
freemoe.intermediate_size = 1920
freemoe.expert_count = 8
freemoe.expert_top_k = 2
freemoe.moe_every_layer = true
freemoe.shared_expert_count = 0
freemoe.dense_ffn_layers = 0
freemoe.norm_type = "rmsnorm"
freemoe.rms_norm_epsilon = 1e-5
freemoe.activation = "swiglu"
freemoe.router_operation = "linear_topk_softmax_selected"
freemoe.router_tie_break = "lower_expert_id"
freemoe.expert_layout = "stacked"
freemoe.rope_mode = "half_rotation"
freemoe.rope_theta = 10000.0
freemoe.rope_scaling = "none"
freemoe.tie_word_embeddings = true
freemoe.context_length_trained = 4096
freemoe.tokenizer_sha256 = <actual SHA-256>
freemoe.source_checkpoint_sha256 = <actual hash or manifest reference>
freemoe.conversion_manifest_sha256 = <actual hash>
freemoe.expert_checksums = <typed array/records defined by writer contract>
```

This is a semantic inventory, not a literal GGUF metadata dump. Implement metadata types with the GGUF writer and verify the resulting file with an independent parser. Do not store a placeholder string or an invented checksum.

## 6.3 Conversion pipeline

The converter must be a separate tool, for example `freemoe-convert`, not a hidden feature of the runtime.

1. **Inspect source**: identify framework, architecture, config, tokenizer, tensor shards, dtype, licence and source hashes.
2. **Validate source**: check required tensors, dimensions, shard completeness, finite values and config-to-tensor consistency.
3. **Resolve architecture**: select the exact source adapter. Do not assume all MoE models share expert tensor order, router scaling, normalisation, RoPE, shared experts or attention semantics.
4. **Map tensors**: create a complete mapping report from source tensor name/shape/dtype to canonical role/shape/dtype and transformation.
5. **Dry run**: calculate output metadata, tensor shapes, byte counts, alignments and estimated output size without writing the output file.
6. **Convert**: transpose/reshape only as explicitly defined; cast/quantise only with recorded method and parameters.
7. **Write GGUF**: write metadata, tensor descriptors, alignment and tensor payloads. The file must be readable by an independent GGUF parser.
8. **Write sidecar manifest**: record source revision/hashes, tokenizer hash, config hash, converter version, mapping, transformations, per-tensor source/output hashes, quantisation settings, warnings and unsupported fields.
9. **Reopen and inspect**: use FreeMoe's parser and an independent parser; compare expected and observed metadata, dimensions, offsets and file size.
10. **Numerical validation**: compare source framework and converted checkpoint using fixed token IDs, logits, selected experts and greedy tokens.
11. **Publish** only after all mandatory gates pass. Keep the source and conversion report.

Suggested CLI contract:

```text
freemoe-convert inspect-source --input <checkpoint-dir> --json report.json
freemoe-convert validate-source --input <checkpoint-dir> --strict
freemoe-convert map --input <checkpoint-dir> --target freemoe-coder-v1 --out mapping.json
freemoe-convert dry-run --input <checkpoint-dir> --target freemoe-coder-v1 --format gguf --out plan.json
freemoe-convert export --input <checkpoint-dir> --target freemoe-coder-v1 --format gguf --out FreeMoe-Coder-F16.gguf --manifest conversion.json
freemoe-convert verify --input FreeMoe-Coder-F16.gguf --manifest conversion.json --strict
```

These are proposed command interfaces. Implement them before documenting them as available commands.

## 6.4 Quantisation sequence

Use this order:

1. F32/F16 reference checkpoint.
2. F16 GGUF export and source-vs-export parity.
3. Q8_0 export and quantised-kernel parity.
4. Q6_K only after block-accurate Q6_K dequantisation and execution kernels are implemented.
5. Q4_K after Q6_K/Q8 tests and quality/performance gates.
6. Lower-bit formats only when kernel support and quality criteria exist.

Quantisation changes values, so bit-exact logit equality is not the expected criterion for low-bit files. Use numerical tolerances, greedy-token agreement, task-level coding evaluation and regression thresholds. Keep F16 as the reference. Never mark a format “supported” merely because its tensor size can be parsed.

---

# 7. Model examination: `WAKE_UP` report

A model inspector must be usable before inference and must not allocate all weights to RAM/VRAM. Output human-readable text and machine-readable JSON.

Required report sections:

1. File identity: path, file size, SHA-256, modification time and source fingerprint.
2. Container: GGUF version, endianness, alignment, metadata/tensor counts, data-section offset.
3. Architecture: architecture ID/version, adapter chosen, support status, reason if unsupported.
4. Config: layer count, hidden width, attention/KV heads, head dimension, context, RoPE, norm, FFN/MoE settings.
5. Tokenizer: tokenizer model/assets, vocabulary size, special-token IDs, chat-template presence, hashes.
6. Tensor inventory: name, logical role, serialized dimensions, type, byte length, offset, alignment and hash status.
7. MoE map: expert count per layer, top-k, expert tensor layout, exact byte ranges, checksum state, shared-expert presence, routing semantics.
8. Quantisation: per-tensor formats and whether parsing, dequantisation and execution are each supported.
9. Memory plan: total weights by type, estimated VRAM resident budget, RAM staging budget, KV-cache estimate by context/batch, scratch reserve, maximum hot experts.
10. I/O plan: cold source, reader mode, expert record ranges, read alignment, prefetch policy and cache slots.
11. Integrity: config/hash consistency, missing/duplicate tensors, bad extents, overlap, checksum mismatch, source mutation detection.
12. Compatibility: supported, inspect-only, CPU-executable, GPU-executable, or rejected, with explicit reason.
13. Provenance: source checkpoint, conversion tool/version, quantisation parameters, license and manifest hashes.

Example command design:

```text
freemoe model inspect model.gguf --json inspect.json
freemoe model wake-up model.gguf --strict --json wake-up.json
freemoe model verify model.gguf --manifest conversion.json
freemoe model memory-plan model.gguf --vram-budget 8GiB --ram-budget 16GiB --json memory.json
```

Never say “model supported” when only the GGUF container parsed successfully. Report parser support, architecture support, tensor dtype support, CPU execution support and GPU execution support separately.

---

# 8. Large-model disk-resident MoE: preserve the source model

The large-model path must not pretend that arbitrary expert chunks are standalone language models. A routed expert is one component of a full Transformer layer; it needs the original layer state, attention, residuals, router scores and other selected experts.

## 8.1 Logical execution

For token hidden state \(h_\ell\):

1. Execute embedding, norm and attention according to the source model architecture.
2. Execute the source model's router exactly, including any architecture-specific scaling, bias, correction terms, group-limited routing, shared expert or normalization.
3. Obtain selected expert IDs and weights.
4. Resolve each expert's tensor extents in the original checkpoint.
5. For each selected expert, check whether it is resident; otherwise schedule a demand read.
6. Fetch the full required tensor payload for that expert, validate it, stage it in bounded host memory and transfer it to a free device slot when the backend is GPU.
7. Wait for the correct transfer completion/fence and verify slot generation before kernel use.
8. Execute every selected expert and combine outputs using the source model's exact routing weights.
9. Update traces and metrics: selected/executed expert IDs, read bytes, transfer bytes, wait time, slot hits/misses, prefetch usefulness and wasted bytes.
10. Evict only an unreferenced slot whose work has completed. A stale asynchronous completion must never overwrite a newer slot generation.

The FreeMoe-Coder v0 router semantics must not be imposed on a third-party model. Imported model families require their own architecture adapters.

## 8.2 Residency tiers

- **Cold tier:** original GGUF file or another supported immutable source. GGUF tensor offsets are authoritative for GGUF inputs.
- **Warm tier:** bounded RAM staging and prefetch cache.
- **Hot tier:** actual backend-owned buffers. On CPU/VNPU they are host RAM, not VRAM. On a real GPU backend they can be device memory.

For an expert record containing gate/up/down tensors, a sidecar index can describe ranges, but it must point to exact tensor byte ranges in the source container and bind to the source file hash. Do not assume the three tensors are physically adjacent unless inspection proves it.

## 8.3 Optional chunk index / manifest

Use a sidecar JSON index first rather than inventing a new raw disk filesystem. A record should include:

```json
{
  "schema": "freemoe-expert-index-v1",
  "source_file": "original-model.gguf",
  "source_sha256": "MUST_BE_COMPUTED",
  "architecture": "SOURCE_ARCHITECTURE_ID",
  "architecture_adapter": "ADAPTER_ID_AND_VERSION",
  "layers": [
    {
      "layer": 0,
      "experts": [
        {
          "expert_id": 0,
          "tensors": [
            {
              "role": "gate",
              "tensor_name": "EXACT_SOURCE_TENSOR_NAME",
              "file_offset": "INTEGER",
              "byte_length": "INTEGER",
              "sha256": "MUST_BE_COMPUTED"
            }
          ]
        }
      ]
    }
  ]
}
```

Replace every placeholder with values read from the actual source checkpoint. Use integer byte offsets in the actual manifest, not strings. Validate source hash before trusting the index. Include schema version, byte order, container data-section semantics, tensor type, serialized dimensions and expert slice boundaries.

A custom chunk file format with headers/end markers can be explored later if profiling demonstrates a benefit. It must not lose provenance, tensor type, block boundaries, alignment or integrity information. Never split a quantised tensor at an arbitrary byte boundary: respect its block geometry and logical row layout.

## 8.4 Memory budget

At all times enforce:

\[
M_{\mathrm{weights,hot}}+M_{\mathrm{KV}}+M_{\mathrm{scratch}}
+M_{\mathrm{transfer}}+M_{\mathrm{allocator\ reserve}}
\le M_{\mathrm{usable\ VRAM}}
\]

The usable VRAM is less than the nominal card capacity. For a 12 GB card, do not assume all 12 GB is available to model weights. Reserve capacity for KV cache, workspaces, driver allocations and display use.

RAM budget:

\[
M_{\mathrm{host\ staging}}+M_{\mathrm{prefetch}}+
M_{\mathrm{index}}+M_{\mathrm{runtime}}
\le M_{\mathrm{host\ budget}}
\]

If the hot slot pool is too small for the active experts, demand-load and evict safely. Prefetch is speculative; a prediction is not permission to skip a required expert.

## 8.5 Performance reality

Disk-resident inference may reduce VRAM requirements but does not guarantee high token throughput. Decode may become storage-latency or transfer-bandwidth bound if selected expert weights cannot remain resident. Measure:
- model open/inspect time;
- cold first-token latency;
- warm first-token latency;
- prefill tokens/s;
- decode tokens/s and p50/p95 inter-token latency;
- NVMe bytes and read latency per token;
- host-to-device bytes and transfer latency per token;
- expert cache hit/miss rate;
- prefetch useful/wasted bytes;
- CPU/GPU utilisation and memory high-water marks.

Report repeated runs with warm/cold state explicitly identified. Do not include one-time model load in per-token decode I/O.

---

# 9. Separate LMDB knowledge cache

The previously discussed approximately 2 GB LMDB store is a separate, optional subsystem. It must not be represented as pretrained model parameters.

Possible roles:
- coding snippets and validated examples;
- symbol/API documentation records;
- retrieval metadata, embeddings if implemented, and provenance;
- externally editable learned facts or task memory.

Suggested record fields:
- stable record ID;
- record type;
- language/library/version;
- content or content reference;
- source and license;
- created/updated timestamps;
- content hash;
- validation status;
- retrieval metadata;
- optional embedding format/version.

Rules:
- LMDB records are not automatically training data.
- Retrieved content is context supplied to the model, not a silent weight update.
- Mutable cache contents must not change the base GGUF checksum.
- Define eviction, compaction, backup/rebuild and corruption-recovery policies.
- Measure the actual database size; “2 GB” is a capacity target, not a guarantee that every record fits after indexes/metadata.

Later, an explicitly separate adapter/delta store may hold LoRA weights or other learned state. Define composition mathematically and version it; do not mutate the base GGUF in place.

---

# 10. Training and distillation plan

## 10.1 Before training

Block training until these inputs are selected and recorded:
- exact teacher checkpoint and revision;
- license and permitted derivative use;
- tokenizer files and hash;
- source framework/version;
- dataset manifests, licenses, filtering and deduplication rules;
- target hardware and memory;
- training framework and checkpoint-resume strategy;
- validation coding benchmark and held-out split;
- experiment configuration and random seeds.

A 27B/32B teacher was discussed, but no exact checkpoint is established in this specification. Do not train against a guessed model or silently mix tokenizers.

## 10.2 Staged plan

1. **Architecture smoke test:** initialise a tiny random model from the v0 config and validate every tensor shape, forward pass, gradients and save/reload.
2. **Tokenizer contract:** encode/decode fixed strings, special tokens, Unicode, whitespace, code indentation, long lines and round trips.
3. **Small data overfit:** demonstrate that the model can overfit a tiny lawful training set; verify loss decreases and save/resume is stable.
4. **Teacher data generation:** produce permitted teacher targets or logits with exact teacher revision, prompt formatting and tokenizer.
5. **Distillation:** train with recorded \(\alpha,\beta,\lambda,\tau\), optimiser, schedule, batch/sequence length and seed.
6. **Router validation:** inspect expert load, router entropy, dead experts, selected expert IDs and determinism.
7. **Coding evaluation:** compile/run generated code where appropriate; evaluate held-out tasks, syntax, tests, pass@k protocol and regressions.
8. **Checkpoint export:** save a full-precision reference and all provenance.
9. **GGUF conversion:** run the converter and manifest generation.
10. **Runtime parity:** compare source framework against FreeMoe CPU, then real GPU backend.
11. **Quantisation:** add Q8_0/Q6_K/Q4_K as independent experiments, recording quality and speed trade-offs.
12. **Release:** publish only with model card, license, tokenizer, hashes, eval report and exact inference commands.

A successful synthetic fixture test proves engine logic for that fixture only; it does not prove student quality or compatibility with third-party models.

---

# 11. Test matrix and release gates

## 11.1 Converter and container tests

- independent GGUF parser accepts the file;
- all metadata types and arrays round-trip;
- expected architecture/version and tokenizer hashes match;
- tensor count, names, types, shapes, offsets and sizes match manifest;
- no overlap, out-of-range extents or integer overflow;
- all file/tensor hashes verify;
- tied embedding/output semantics preserved;
- missing, duplicate and unknown tensors fail closed;
- source checkpoint with missing shards fails before output is published;
- interrupted export leaves no apparently valid final file;
- converter output is reproducible or any nondeterminism is documented.

## 11.2 Mathematical parity tests

Use fixed token IDs and compare each stage, not just final text:
- embedding outputs;
- RMSNorm;
- Q/K/V projections;
- RoPE Q/K;
- attention scores and output;
- router logits and selected expert IDs;
- selected router weights;
- each expert's gate/up/SwiGLU/down result;
- weighted MoE sum;
- residual stream;
- final norm and logits;
- greedy next token and multi-token continuation.

Run both single-token decode and multi-token prefill. Include context-boundary and KV-cache reset tests.

## 11.3 MoE paging tests

- all experts resident versus one-slot cache gives same outputs within declared tolerance;
- every selected expert is executed exactly once per token/layer;
- selected and executed expert sets match;
- cold and warm runs have the same mathematical output;
- slot generation prevents stale completion corruption;
- no slot eviction while referenced or while device work is pending;
- source file mutation detected;
- corrupt expert checksum fails closed;
- demand read cannot be starved indefinitely by prefetch;
- prefetch cancellation does not corrupt a subsequent demand request;
- budget enforcement prevents allocations above configured caps;
- device-loss/transfer-failure paths report explicit errors;
- no fake VRAM metrics on CPU/VNPU backend.

## 11.4 Quantisation tests

For each type:
- known block dequantisation vectors;
- block-boundary and multi-block tests;
- non-aligned logical dimensions where format allows them;
- row/stride and expert-slice boundaries;
- F16/F32 reference comparison;
- NaN/Inf and malformed scale handling;
- tensor byte count versus independent implementation;
- full-model logits/token agreement and coding task metrics.

Q6_K must not be declared executable until dequantisation and inference kernels pass these gates. Correctly parsing Q6_K geometry is not sufficient.

## 11.5 Performance gate

Performance is qualified only on named target hardware with recorded:
- CPU/GPU model, RAM/VRAM, storage, OS, compiler, backend API, driver and runtime versions;
- exact model/checkpoint hash, quantisation, context, prompt set and generated token count;
- cache state, reader, cache slots, prefetch settings and thread count;
- at least five measured repetitions after warm-up, retaining all raw results;
- separate load, prefill and decode windows;
- p50/p95 latency, throughput and memory high-water marks.

Do not publish a performance claim based only on the synthetic tiny fixture.

---

# 12. Suggested repository layout

```text
FreeMoe/
  README.md
  model-spec/
    freemoe-coder-v1.md
    config.schema.json
    tensor-schema.json
    gguf-metadata.md
    conversion-manifest.schema.json
    expert-index.schema.json
  tools/
    freemoe-convert/
    freemoe-inspect/
    reference/
  src/
    model/
      config.cpp
      tensor_registry.cpp
      adapter_freemoe_coder.cpp
      adapter_freemoe_tiny.cpp
    gguf/
      reader.cpp
      writer.cpp
      validation.cpp
    kernels/
      cpu/
      vulkan/        # only when implemented
      hip/           # only when implemented
    runtime/
      engine.cpp
      kv_cache.cpp
      router.cpp
      residency.cpp
      read_scheduler.cpp
      trace.cpp
  training/
    configs/
    data_manifests/
    scripts/
    eval/
  tests/
    fixtures/
    conversion/
    parity/
    quantisation/
    paging/
    malformed/
    reference/
  docs/
    supported_models.md
    conversion_protocol.md
    test_protocol.md
    benchmark_protocol.md
    model_card.md
```

This is a proposed layout, not a claim that these paths already exist. Preserve the current repository's existing structure where practical and avoid duplicate modules with the same responsibility.

---

# 13. Implementation order to get to the first real GGUF

## Gate 1: Freeze the model contract
- Choose teacher and tokenizer.
- Replace all placeholder fields in `config.json`.
- Finalise GGUF tensor dimension order and projection orientation.
- Define the exact RoPE convention, tokenizer metadata, special tokens and chat template.
- Freeze `freemoe-coder-v1` version and schema.

## Gate 2: Build the random-weight model and oracle
- Generate deterministic small weights.
- Run source/reference forward pass.
- Save/reload the checkpoint.
- Compare every intermediate operation.
- Keep `freemoe-tiny-v1` unchanged as a separate engine regression target.

## Gate 3: Write a minimal GGUF exporter
- Start with F32 or F16 only.
- Write config, tokenizer metadata and every tensor.
- Emit manifest with tensor hashes and offsets.
- Reopen using FreeMoe parser and independent GGUF parser.
- Reject export if any required metadata/tensor is missing.

## Gate 4: Convert a real checkpoint
- Only after the chosen source architecture genuinely matches the student contract or a correct source-to-student mapping/training export exists.
- A third-party 27B/32B MoE is not automatically convertible to FreeMoe-Coder merely by renaming tensors.
- For distillation, the output student weights are the weights trained for FreeMoe-Coder, not copied slices of the teacher.

## Gate 5: Execute the exported model
- CPU first, with fixed-token parity.
- Then add one real GPU backend and validate the same tensors/operations.
- Add quantisation after full-precision parity.
- Integrate expert paging for imported large models without weakening correctness.

## Gate 6: Evaluate and release
- Run held-out coding evaluation.
- Run memory and context stress tests.
- Run conversion reproducibility and integrity checks.
- Publish exact source hashes, config, tokenizer, licence, model card and results.

---

# 14. Important design decisions not to make implicitly

1. **Do not invent the vocabulary size.** It follows from the exact tokenizer.
2. **Do not claim the 700M active count without stating whether the full output projection is counted.**
3. **Do not treat a GGUF custom architecture ID as automatic llama.cpp support.**
4. **Do not convert an arbitrary 27B/32B MoE checkpoint into the student by chunking.** Chunking belongs to the large-model runtime; distillation belongs to student creation.
5. **Do not assume all third-party MoEs use top-2 routing.** Preserve source-model router semantics.
6. **Do not assume three expert matrices are adjacent on disk.** Derive exact extents from the GGUF tensor directory.
7. **Do not split quantised tensors on arbitrary byte boundaries.** Respect block geometry.
8. **Do not count host RAM as VRAM.** The current repository's Virtual NPU is a CPU software device; its device buffers are host memory.
9. **Do not advertise quantisation execution just because the parser understands tensor sizes.**
10. **Do not embed the LMDB cache into the base model file as mutable state.**
11. **Do not treat the proposed 20-layer dimensions as settled trained-model facts.** Freeze them only after tokenizer/teacher/training feasibility checks.
12. **Do not claim performance from a sandbox or synthetic model as if it were measured on the target GPU.**

---

# 15. Sources and provenance

Project source files to reconcile with implementation before merging this specification:

- Repository README: https://github.com/riteshnair/FreeMoe/blob/main/README.md
- Architecture inventory: https://github.com/riteshnair/FreeMoe/blob/main/docs/ARCHITECTURE.md
- Supported-model contract: https://github.com/riteshnair/FreeMoe/blob/main/docs/SUPPORTED_MODELS.md
- GGUF/quantisation contract: https://github.com/riteshnair/FreeMoe/blob/main/docs/FORMAT_AND_QUANTISATION.md
- Benchmark protocol: https://github.com/riteshnair/FreeMoe/blob/main/docs/BENCHMARK_PROTOCOL.md
- GGUF format overview: https://github.com/ggml-org/ggml/blob/master/docs/gguf.md
- llama.cpp model adapter guide: https://github.com/ggml-org/llama.cpp/blob/master/docs/development/HOWTO-add-model.md

The repository currently documents `freemoe-tiny-v1` as its synthetic F32-only conformance family, with a physical GPU placeholder. The FreeMoe-Coder architecture in this document is a proposed new production model contract. The actual teacher checkpoint, tokenizer, trained weights, converter and real GPU implementation must be selected, built and tested rather than assumed to exist.
