# Checkpoint MOE

| Field | Value |
|---|---|
| Name | FreeMoe synthetic moe F32 conformance fixture |
| Source URL | No external URL; generated in-repo by `tools/make_fixtures.py` |
| Local path | `tests/fixtures/moe-fm.gguf` |
| SHA-256 | `8f1703d7863588347e318000526938a8b1313bfaf2eb0bee1bc7c1aaa985ebc8` |
| Architecture string | `freemoe-tiny` |
| Adapter | `freemoe-tiny-v1` |
| Quantisation | F32 only |
| Licence name | MIT |
| Licence text | `LICENSES/CHECKPOINTS-MIT.txt` |
| Model kind | `moe` |
| Expert count | 8 |
| Top-k | 2 |

This is a deterministic synthetic checkpoint with no pretrained weights. It is
used to test parser, adapter, forward-pass and expert-paging semantics. It is
not evidence of support for arbitrary Llama or MoE GGUF models.
