#pragma once

#include "fm/backend.hpp"
#include "fm/chat_template.hpp"
#include "fm/gguf.hpp"
#include "fm/tokenizer.hpp"
#include "fm/io.hpp"
#include "fm/residency.hpp"

#include <map>
#include <random>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace fm {

constexpr double DEFAULT_ABS_TOL = 5e-3;
constexpr double DEFAULT_REL_TOL = 5e-3;
constexpr uint32_t DEFAULT_N_GREEDY = 16;

// Greedy is the default: temperature 0 reproduces the reference comparison
// behaviour exactly. Non-zero temperature enables top-k/top-p/min-p sampling.
//
// The penalties act on the *logits* and need to know which tokens have already
// been produced, which is why `apply_sampling_bias` takes the history rather
// than living inside the sampler: a sampler that guessed at history would be
// untestable against llama.cpp's own ordering.
struct SamplingConfig {
    double temperature = 0.0;
    uint32_t top_k = 0;
    double top_p = 1.0;
    // Keep the smallest set of tokens whose probability is at least
    // `min_p` times the most likely token's. 0 disables it.
    double min_p = 0.0;
    // > 1 divides a repeated token's positive logit (and multiplies a negative
    // one); 1.0 is the identity.
    double repetition_penalty = 1.0;
    // Additive, once per distinct token in the window / once per occurrence.
    double presence_penalty = 0.0;
    double frequency_penalty = 0.0;
    // How many tokens back the three penalties look. 0 = the whole history,
    // which is what llama.cpp's `penalty_last_n = -1` means.
    uint32_t penalty_window = 0;
    // A static per-token bias, applied after the penalties and before the
    // temperature. Sorted for a deterministic report.
    std::map<uint32_t, double> logit_bias;
    // How many alternatives to report alongside each sampled token.
    uint32_t n_probs = 0;
    uint64_t seed = 0;
    bool greedy() const { return temperature <= 0.0; }
};

// The bias stage of sampling, as a pure function of the logits and the tokens
// produced so far: repetition (multiplicative), presence and frequency
// (additive), then the static per-token bias. Returns a copy; the caller keeps
// the raw logits for reporting.
std::vector<float> apply_sampling_bias(const std::vector<float> &logits,
                                       const SamplingConfig &config,
                                       const std::vector<int32_t> &history);

// One token and, when `n_probs` asks for them, the alternatives with their
// probabilities after truncation.
struct SampledToken {
    uint32_t id = 0;
    std::vector<std::pair<uint32_t, double>> alternatives;
};

// Pure sampler, independent of the engine so it can be unit tested directly.
// temperature <= 0 means greedy argmax (lowest token id wins ties).
uint32_t sample_from_logits(const std::vector<float> &logits, const SamplingConfig &config,
                            std::mt19937_64 &rng);
SampledToken sample_detailed(const std::vector<float> &logits, const SamplingConfig &config,
                             std::mt19937_64 &rng);

struct RunConfig {
    std::string model_path;
    std::string run_id = "local";
    BackendConfig backend;
    ReaderMode reader_mode = ReaderMode::Buffered;
    BudgetConfig budget;
    ResidencyConfig residency;
    uint32_t cache_slots = 0;
    uint32_t queue_depth = 2;
    uint64_t coalesce_gap = 4096;
    uint32_t t_wait_ms = 100;
    uint32_t generated_tokens = DEFAULT_N_GREEDY;
    uint32_t prompt_limit = 20;
    // Non-zero caps the KV context. This is a memory decision, not a
    // correctness one: the real weights and real arithmetic are unchanged, so a
    // 2048-context checkpoint still runs when only a few hundred tokens are used.
    uint32_t context_limit = 0;
    bool force_direct_fallback = false;
    std::string trace_path;
    std::string metrics_path;
    std::string report_path;
    std::string reference_root = "tests/reference";
    std::string results_dir = "results";
    SamplingConfig sampling;
    double abs_tol = DEFAULT_ABS_TOL;
    double rel_tol = DEFAULT_REL_TOL;
};

struct RoutingEvent {
    uint64_t token_index = 0;
    uint32_t layer = 0;
    std::vector<uint32_t> experts;
};

struct PromptResult {
    std::string prompt;
    std::vector<int32_t> input_ids;
    std::vector<float> prefill_logits;
    std::vector<int32_t> generated_ids;
    // One row per generated token when `--n-probs` asks for it: the alternatives
    // that survived truncation, most likely first.
    std::vector<std::vector<std::pair<uint32_t, double>>> alternatives;
    std::vector<RoutingEvent> routing;
    std::vector<double> inter_token_ms;
    double prefill_ms = 0.0;
    double decode_ms = 0.0;
    double first_token_ms = 0.0;
    double total_ms = 0.0;

    Json to_json(bool include_logits = false) const;
};

struct PromptComparison {
    uint32_t index = 0;
    uint32_t vocab = 0;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    uint32_t matching_tokens = 0;
    uint32_t reference_tokens = 0;
    uint32_t routing_events = 0;
    uint32_t routing_mismatches = 0;
    bool logits_within_tolerance = false;
    bool tokens_match = false;
    bool routing_matches = false;
    Json to_json() const;
};

struct ComparisonResult {
    std::string model_kind;
    std::string reference_dir;
    uint32_t prompts = 0;
    uint32_t prompts_within_tolerance = 0;
    uint32_t prompts_token_match = 0;
    uint32_t prompts_routing_match = 0;
    uint64_t matching_tokens = 0;
    uint64_t reference_tokens = 0;
    double max_abs_diff = 0.0;
    double max_rel_diff = 0.0;
    SamplingConfig sampling;
    double abs_tol = DEFAULT_ABS_TOL;
    double rel_tol = DEFAULT_REL_TOL;
    std::vector<PromptComparison> per_prompt;
    Json to_json() const;
};

struct RunMetrics {
    uint64_t model_load_ns = 0;
    uint64_t prefill_ns = 0;
    uint64_t decode_ns = 0;
    uint64_t total_ns = 0;
    uint64_t prompt_tokens = 0;
    uint64_t generated_tokens = 0;
    uint64_t prompt_count = 0;
    double first_token_latency_sum_ms = 0.0;
    double prefill_tokens_per_s = 0.0;
    double decode_tokens_per_s = 0.0;
    double first_token_latency_ms = 0.0;
    double median_inter_token_ms = 0.0;
    double p95_inter_token_ms = 0.0;
    std::vector<double> inter_token_ms;
    std::string cache_state = "process-cold first run; OS cache state unknown";

    void finalize();
    Json to_json() const;
};

class Engine {
public:
    explicit Engine(RunConfig config);
    ~Engine();

    void initialize();
    PromptResult run_prompt(const std::string &prompt, uint32_t generated_tokens);
    ComparisonResult compare_reference(const std::string &reference_root, uint32_t prompt_limit,
                                       uint32_t generated_tokens);
    Json metrics_json() const;
    Json report_json(const ComparisonResult *comparison = nullptr) const;
    void release_runtime();
    void write_metrics(const std::string &path) const;
    void write_report(const std::string &path, const ComparisonResult *comparison = nullptr) const;

    // Public text helpers: token ids in, bytes out.
    std::string detokenize(const std::vector<int32_t> &ids) const;

    // Renders a conversation into the exact prompt bytes the model was trained
    // on, using the checkpoint's own `tokenizer.chat_template` when it has one.
    // `template_override` accepts either a built-in name ("chatml", "llama3",
    // ...) or literal template text. Fails rather than guessing when neither
    // the checkpoint nor the built-in table provides a template.
    ChatRender render_chat(const std::vector<ChatMessage> &messages, bool add_generation_prompt,
                           const std::string &template_override) const;
    // Batched prefill. `forward_token` evaluates one token at a time, which
    // re-reads every weight row once per prompt token; `prefill_batch` runs the
    // same block arithmetic for the whole prompt at once so each row is fetched
    // once. It covers the dense families and declines everything whose block
    // differs structurally (MLA, recurrent layers, routed experts, BitNet's
    // sub-norms, phi2's parallel residual, gpt-oss's post-norm FFN input,
    // mistral3's adaptive norm); `run_prompt` falls back per checkpoint.
    bool supports_batched_prefill() const;
    std::vector<float> prefill_batch(const std::vector<int32_t> &tokens, uint64_t start_position);
    const ModelManifest &manifest() const { return manifest_; }
    const GGUFDocument &document() const { return doc_; }
    Backend &backend() { return *backend_; }
    ReadScheduler &scheduler() { return *scheduler_; }
    ResidencyManager *residency() { return residency_.get(); }
    TraceWriter &trace() { return *trace_; }
    const RunConfig &config() const { return config_; }
    const std::vector<PromptResult> &results() const { return results_; }
    const RunMetrics &run_metrics() const { return run_metrics_; }
    const ComparisonResult *last_comparison() const { return last_comparison_.get(); }

private:
    void load_fixed_weights();
    void initialize_tokenizer();
    // Resident tensor view: byte pointer plus the block type so quantised
    // tensors can be executed without materialising F32 copies.
    struct TensorView {
        const uint8_t *data = nullptr;
        uint32_t type_id = 0;
    };
    TensorView tensor_view(const std::string &name) const;
    const float *weights(const std::string &name) const;
    // RMSNorm vectors are 1-D and may be stored in a block format; decode them
    // once at load time so the hot path keeps a plain F32 view.
    const float *norm_weights(const std::string &name) const;
    const float *bias_weights(const std::string &name) const;
    bool has_bias(const std::string &name) const;
    // A one-element tensor (BitNet's `*.scale`), or nothing when the file does
    // not carry it. Kept separate from `weights()` because absence is normal.
    std::optional<float> scalar_weight(const std::string &name) const;
    // Multiply a projection's output by its optional scale, in place. Returns
    // whether a scale was applied so the caller can log it; llama.cpp applies
    // this inside `build_lora_mm`, i.e. before any bias is added.
    bool apply_projection_scale(std::vector<float> &vec, const std::string &name,
                                const std::string &op);
    std::vector<float> matvec_view(const TensorView &view, const std::vector<float> &x,
                                   uint32_t out_dim, uint32_t in_dim,
                                   const std::string &op);
    std::vector<float> matvec(const float *w, const std::vector<float> &x,
                              uint32_t out_dim, uint32_t in_dim, const std::string &op);
    std::vector<float> forward_token(int32_t token, uint64_t position, const std::string &phase,
                                     std::vector<RoutingEvent> *routing);
    std::vector<int32_t> tokenize(const std::string &prompt) const;
    uint32_t greedy(const std::vector<float> &logits) const;
    // Samples one token. Deterministic for a given seed and logits.
    uint32_t sample(const std::vector<float> &logits);
    std::mt19937_64 &rng() { return rng_; }
    std::vector<uint32_t> topk(const std::vector<float> &scores, uint32_t k) const;
    std::vector<float> softmax(const std::vector<float> &values) const;
    // `freq_base` is the layer's rotary base (0 = the file's single base); a
    // windowed layer can carry its own (`rope.freq_base_swa`).
    void rope_half(std::vector<float> &q, uint32_t q_heads, std::vector<float> &k,
                   uint32_t k_heads, uint64_t position, double freq_base = 0.0) const;
    void read_reference_logits(const std::string &path, std::vector<float> &out) const;
    std::vector<int32_t> read_reference_ids(const std::string &path) const;
    std::vector<RoutingEvent> read_reference_routing(const std::string &path) const;

    RunConfig config_;
    std::unique_ptr<MappedFile> file_;
    std::unique_ptr<Tokenizer> tokenizer_;
    GGUFDocument doc_;
    ModelManifest manifest_;
    std::unique_ptr<TraceWriter> trace_;
    std::unique_ptr<Backend> backend_;
    std::unique_ptr<ReadScheduler> scheduler_;
    BudgetResult budget_;
    std::unique_ptr<ResidencyManager> residency_;
    std::unique_ptr<DeviceBuffer> weights_buffer_;
    std::unique_ptr<DeviceBuffer> kv_buffer_;
    std::unique_ptr<DeviceBuffer> workspace_buffer_;
    std::unique_ptr<DeviceBuffer> staging_buffer_;
    float *kv_ = nullptr;
    std::map<std::string, uint64_t> weight_offsets_;
    std::map<std::string, std::vector<float>> decoded_norms_;
    // Attention projection biases, decoded once at load. Empty when the
    // checkpoint does not use them.
    std::map<std::string, std::vector<float>> decoded_biases_;
    std::vector<PromptResult> results_;
    RunMetrics run_metrics_;
    Json released_residency_snapshot_ = Json::null();
    std::unique_ptr<ComparisonResult> last_comparison_;
    bool initialized_ = false;
    std::mt19937_64 rng_{0};
};

} // namespace fm
