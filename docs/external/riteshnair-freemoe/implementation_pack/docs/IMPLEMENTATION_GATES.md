# Implementation Gates
0. Inspect repo; build baseline; run existing tests.
1. Read-only GGUF inspection for one supported model.
2. Native blob import/verify and byte-exact round-trip for supported encodings.
3. Original versus blob-backed inference parity.
4. Transactional commits and interruption recovery.
5. Tiny C++ training, gradient checks, held-out improvement, checkpoint/resume.
6. Governor adapter and MCP tools.
7. Supported pretrained model adaptation.
8. Explicit MoE design and later Vulkan backend.
Do not start a later gate until the current gate has documented evidence.
