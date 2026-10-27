# Checkpoint DENSE

| Field | Value |
|---|---|
| Name | FreeMoe synthetic dense F32 conformance fixture |
| Source URL | No external URL; generated in-repo by `tools/make_fixtures.py` |
| Local path | `tests/fixtures/dense-fm.gguf` |
| SHA-256 | `4d81aca51f3d087c977fc4b40d3227401d225b45373878b618a72059d35885a1` |
| Architecture string | `freemoe-tiny` |
| Adapter | `freemoe-tiny-v1` |
| Quantisation | F32 only |
| Licence name | MIT |
| Licence text | `LICENSES/CHECKPOINTS-MIT.txt` |
| Model kind | `dense` |
| Expert count | 0 |
| Top-k | 0 |

This is a deterministic synthetic checkpoint with no pretrained weights. It is
used to test parser, adapter, forward-pass and expert-paging semantics. It is
not evidence of support for arbitrary Llama or MoE GGUF models.
