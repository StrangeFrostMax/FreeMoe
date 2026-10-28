#pragma once

#include "fm/common.hpp"
#include "fm/gguf.hpp"

#include <string>

namespace fm {

// Architecture-agnostic support probe.
//
// `probe_model` answers one question for *any* GGUF file, including files whose
// architecture this build has no adapter for: what exactly would FreeMoe need in
// order to run it, and which of those things are missing today?
//
// It never guesses. It reads the container, the architecture metadata, the
// tokenizer description and the tensor inventory, and reports supported / not
// supported together with the specific blockers. That makes "we support this
// model" a claim you can check instead of a claim you have to trust.
Json probe_model(const GGUFDocument &doc, const std::string &source_path);

} // namespace fm
