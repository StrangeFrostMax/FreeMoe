# Coding Agent Brief: FreeMoe Initial Vertical Slice

You are working on an existing, partially built Llama-like C++ inference engine on Windows. First inspect the repository and report what is actually implemented. Do not replace the project with a new scaffold without justification.

## Goal
Add a small, verifiable path from a supported GGUF file to FreeMoe's native tensor blob package while preserving the original inference path. No PyTorch dependency in the core.

## Do first
1. Inspect build system, GGUF parser, supported architectures, tensor types, tests and tracing.
2. Build Debug and Release and run current tests before editing.
3. Report the exact supported small target models/tensor encodings from code and tests.
4. Implement only read-only GGUF metadata/tensor inspection.
5. Implement a minimal native blob writer/reader for explicitly supported tensor encodings.
6. Add checksums, bounds checks, manifest and round-trip tests.
7. Verify original GGUF remains unchanged.
8. Stop and report before adding training, MCP, MoE or Vulkan.

## Rules
- Do not assume support based on a model name.
- Do not silently dequantise/requantise while claiming byte-exact round-trip.
- Do not serialize native C++ structs directly.
- Check integer overflow, offsets, dimensions and lengths before allocation/access.
- Do not overwrite the only known-good file.
- No arbitrary shell execution or model-controlled paths.
- No fake commands or no-op features.
- Do not claim training from tensor extraction.
- Run and report actual tests, with exact commands and output summaries.
- Keep unsupported tensor types explicit.
- Avoid introducing Python or PyTorch into the core.
- MCP is later; first stabilise direct C++ APIs and CLI.

## Deliverables
- Code changes appropriate to the existing repository.
- Updated README with Windows PowerShell commands.
- Tests for valid round-trip, corruption, truncation, malformed metadata and unsupported encodings.
- A report of supported architectures/types, test outcomes, performance, memory use, and remaining gaps.

## Stop condition
Stop after GGUF inspection and blob round-trip pass. Do not proceed to training until reviewed.
