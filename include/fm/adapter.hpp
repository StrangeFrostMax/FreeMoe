#pragma once
// Adapter rows and their tags.
//
// Debugging a checkpoint that will not load is mostly the question "which
// adapter row was selected, and what did it decide?". llama.cpp answers it by
// tagging every tensor with an `LLM_TENSOR_*` enum in a per-architecture table
// (`src/llama-arch.cpp`) and printing the architecture's graph name. This does
// the same: one row per architecture carrying the upstream enum it corresponds
// to, plus a tag for every tensor the row maps and a tag for every branch
// decision the row made.
//
// The tags are the same vocabulary llama.cpp uses (`ATTN_Q`, `FFN_GATE_EXPS`,
// `FFN_GATE_INP`, `ATTN_OUT`, …), so a mismatch can be compared directly against
// an upstream `llama_model_loader` error instead of being translated first.
#include <string>
#include <vector>

#include "fm/gguf.hpp"

namespace fm {

// One adapter row: the architecture string, the upstream enum it mirrors, the
// upstream graph that builds it, and a one-line description of the block.
struct AdapterRow {
    const char *arch;
    // Upstream `LLM_ARCH_*` enum name, so an error here can be searched in
    // llama.cpp's own source without guessing which family was meant.
    const char *upstream_enum = nullptr;
    // Upstream graph builder name (`llm_build_llama`, …), or the file the block
    // was transcribed from when there is no single builder.
    const char *graph = nullptr;
    const char *block = nullptr;
};

// The row for `arch`, or nullptr when no adapter claims it. Never throws: the
// probe has to be able to report an unknown architecture rather than fail on it.
const AdapterRow *adapter_row(const std::string &arch);

// Every implemented adapter row, for tables and the coverage report.
std::vector<AdapterRow> adapter_rows();

// llama.cpp's tensor tag for a role (`ATTN_Q`, `FFN_GATE_INP`, …). Distinct from
// `tensor_role_name` where upstream names the same role differently, so a
// comparison against llama.cpp is literal rather than approximate.
const char *adapter_tensor_tag(TensorRole role);

// The ggml operation the tensor feeds, in the same spirit as upstream's
// `ggml_tensor_impl`. Used only for reporting.
const char *adapter_tensor_op(TensorRole role);

// Branch decisions resolved from the checkpoint, as `key=value` tags. These are
// the things that make two `llama` checkpoints behave differently, so they are
// what a log line has to show.
std::vector<std::string> adapter_config_tags(const ModelConfig &cfg);

}  // namespace fm
