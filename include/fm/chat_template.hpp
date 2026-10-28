#pragma once

#include <string>
#include <vector>

namespace fm {

// Chat templates.
//
// GGUF carries `tokenizer.chat_template`, which is Jinja2. Without it a chat
// model is unusable: the role markers, the BOS placement and the trailing
// assistant prefix are all part of the prompt the model was trained on, and
// feeding raw text instead produces confidently wrong answers rather than an
// error. This header is the interface to a *restricted* interpreter for the
// subset those templates actually use, plus built-in templates for the
// families whose checkpoints ship without one.
//
// The restriction is deliberate and reported: an unsupported construct raises
// a typed error naming it, at chat time only, so plain completion on the same
// checkpoint keeps working.
struct ChatMessage {
    std::string role;
    std::string content;
};

struct ChatTemplateConfig {
    // Appended after the last message so the model starts its reply. Templates
    // guard this with `{% if add_generation_prompt %}`, so it is a variable
    // rather than something the engine splices in.
    bool add_generation_prompt = true;
    std::string bos_token;
    std::string eos_token;
};

// What a chat render produced, and where the template came from. Reporting the
// provenance matters: "the checkpoint told us" and "we used a built-in default"
// are very different claims about whether the prompt is right.
struct ChatRender {
    std::string prompt;
    std::string template_source;  // "checkpoint", "builtin:<name>", "override"
    std::string template_name;    // built-in name when one applied
};

// Renders `template_text` over `messages`. Throws Error(ErrorCode::NOT_IMPLEMENTED)
// when the template uses a Jinja construct outside the supported subset, and
// Error(ErrorCode::INVALID_ARGUMENT) when it raises or the messages are
// malformed (for example a template that forbids a system role).
std::string render_chat_template(const std::string &template_text,
                                 const std::vector<ChatMessage> &messages,
                                 const ChatTemplateConfig &config);

// The built-in template for an architecture, as a Jinja string, or an empty
// string when the architecture has no known template. Rendering goes through
// the same interpreter as a checkpoint's own template, so a built-in cannot
// drift from the one the model ships with.
std::string builtin_chat_template(const std::string &architecture);

// The built-in template for an explicit template *name* ("chatml", "llama3",
// ...). Empty when the name is unknown.
std::string builtin_chat_template_by_name(const std::string &name);

// Names the built-in template that `builtin_chat_template` would return, for
// the probe and for `--chat-template` overrides.
std::string builtin_chat_template_name(const std::string &architecture);

// Picks a built-in from the *token table*: the marker a template would emit is
// either in the vocabulary or it is not, which is stronger evidence than the
// architecture name (`llama` covers both the llama2 and llama3 marker sets).
// Returns an empty string when the tokens match nothing known.
std::string builtin_chat_template_from_tokens(
    const std::string &architecture, bool has_llama3_header, bool has_im_start,
    bool has_start_of_turn, bool has_phi3_markers, bool has_inst, bool has_zephyr_markers);

// Resolves what to use for a model: the checkpoint's own template when it has
// one, the architecture's built-in otherwise. `override_text` wins over both.
// Returns an empty string when nothing applies (the caller then sends the
// prompt verbatim, which is the honest behaviour for a base model).
std::string resolve_chat_template(const std::string &checkpoint_template,
                                  const std::string &architecture,
                                  const std::string &override_text);

// True when the template only uses constructs this interpreter implements.
// Used by the probe so `chat_template_present` is not mistaken for "usable".
bool chat_template_supported(const std::string &template_text, std::string *reason);

} // namespace fm
