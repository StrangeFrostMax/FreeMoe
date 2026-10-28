# Risks and Baselines
Chunked storage, tensor formats, CPU training, offload, LoRA, tool calling and GGUF export are not novel individually. FreeMoe's possible value is a measured integration of recoverable tensor construction, bounded-memory updates, expert-aware indexing and safe orchestration.

Compare later with: an in-memory tiny trainer; a standard checkpoint workflow; a tensor format such as Safetensors; and adapter training. Measure I/O amplification, memory, recovery, correctness and held-out quality. Stop or narrow scope if custom storage adds complexity without measurable value. Disk-first does not mean memory-free.
