# FreeMoe Project Rules

1. Inspect the existing repository before changing it.
2. Work one implementation gate at a time.
3. No mandatory PyTorch dependency in the core.
4. Do not treat GGUF as a complete training checkpoint by default.
5. Never overwrite the only known-good model version.
6. Validate every path, offset, length, shape product and enum.
7. Do not trust LLM-generated commands, paths or arguments.
8. Keep source teacher files immutable.
9. No network access in blob-format unit tests.
10. Add a test for every bug fix and format rule.
11. Report actual build/test outcomes; never invent success.
12. Separate observed facts from assumptions.
13. Measure I/O and memory before performance claims.
14. No MoE, transformer training, MCP automation or Vulkan until the current gate passes.
15. If the current repo structure differs from the suggested design, adapt to it rather than duplicating code.
