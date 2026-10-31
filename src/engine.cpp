#include "fm/activation.hpp"
#include "fm/engine.hpp"
#include "fm/kernels.hpp"
#include "fm/quant.hpp"
#include "fm/rope.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <set>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>

namespace fm {

// ---------------------------------------------------------------------------
// Result JSON
// ---------------------------------------------------------------------------

Json PromptResult::to_json(bool include_logits) const {
    Json j = Json::object();
    j.set("prompt_bytes", static_cast<uint64_t>(prompt.size()));
    j.set("input_tokens", static_cast<uint64_t>(input_ids.size()));
    j.set("generated_tokens", static_cast<uint64_t>(generated_ids.size()));
    j.set("prefill_ms", fixed_double(prefill_ms, 6));
    j.set("decode_ms", fixed_double(decode_ms, 6));
    j.set("first_token_ms", fixed_double(first_token_ms, 6));
    j.set("total_ms", fixed_double(total_ms, 6));
    j.set("median_inter_token_ms", fixed_double(median(inter_token_ms), 6));
    j.set("p95_inter_token_ms", fixed_double(percentile95(inter_token_ms), 6));
    Json generated = Json::array();
    for (int32_t id : generated_ids) generated.push(Json::integer(id));
    j.set("generated_ids", std::move(generated));
    j.set("routing_events", static_cast<uint64_t>(routing.size()));
    if (!alternatives.empty()) {
        Json rows = Json::array();
        for (const auto &row : alternatives) {
            Json entry = Json::array();
            for (const auto &[id, prob] : row) {
                Json pair = Json::object();
                pair.set("id", static_cast<int64_t>(id));
                pair.set("prob", Json::string(fixed_double(prob, 9)));
                entry.push(std::move(pair));
            }
            rows.push(std::move(entry));
        }
        j.set("alternatives", std::move(rows));
    }
    if (include_logits) {
        Json logits = Json::array();
        for (float v : prefill_logits) logits.push(Json::string(fixed_double(v, 9)));
        j.set("prefill_logits_text", std::move(logits));
    }
    return j;
}

Json PromptComparison::to_json() const {
    Json j = Json::object();
    j.set("index", static_cast<int64_t>(index));
    j.set("vocab", static_cast<int64_t>(vocab));
    j.set("max_abs_diff", fixed_double(max_abs_diff, 9));
    j.set("max_rel_diff", fixed_double(max_rel_diff, 9));
    j.set("matching_tokens", static_cast<int64_t>(matching_tokens));
    j.set("reference_tokens", static_cast<int64_t>(reference_tokens));
    j.set("routing_events", static_cast<int64_t>(routing_events));
    j.set("routing_mismatches", static_cast<int64_t>(routing_mismatches));
    j.set("logits_within_tolerance", logits_within_tolerance);
    j.set("tokens_match", tokens_match);
    j.set("routing_matches", routing_matches);
    return j;
}

Json ComparisonResult::to_json() const {
    Json j = Json::object();
    j.set("model_kind", model_kind);
    j.set("reference_dir", reference_dir);
    j.set("prompts", static_cast<int64_t>(prompts));
    j.set("prompts_within_tolerance", static_cast<int64_t>(prompts_within_tolerance));
    j.set("prompts_token_match", static_cast<int64_t>(prompts_token_match));
    j.set("prompts_routing_match", static_cast<int64_t>(prompts_routing_match));
    j.set("matching_tokens", matching_tokens);
    j.set("reference_tokens", reference_tokens);
    j.set("max_abs_diff", fixed_double(max_abs_diff, 9));
    j.set("max_rel_diff", fixed_double(max_rel_diff, 9));
    j.set("abs_tol", fixed_double(abs_tol, 9));
    j.set("rel_tol", fixed_double(rel_tol, 9));
    bool pass = prompts > 0 && prompts_within_tolerance == prompts &&
                prompts_token_match == prompts &&
                (model_kind != "moe" || prompts_routing_match == prompts);
    j.set("status", pass ? "pass" : "fail");
    Json per = Json::array();
    for (const auto &p : per_prompt) per.push(p.to_json());
    j.set("per_prompt", std::move(per));
    return j;
}

void RunMetrics::finalize() {
    prefill_tokens_per_s = prefill_ns ? static_cast<double>(prompt_tokens) / (static_cast<double>(prefill_ns) / 1e9) : 0.0;
    decode_tokens_per_s = decode_ns ? static_cast<double>(generated_tokens) / (static_cast<double>(decode_ns) / 1e9) : 0.0;
    first_token_latency_ms = prompt_count ? first_token_latency_sum_ms / static_cast<double>(prompt_count) : 0.0;
    median_inter_token_ms = median(inter_token_ms);
    p95_inter_token_ms = percentile95(inter_token_ms);
}

Json RunMetrics::to_json() const {
    Json j = Json::object();
    j.set("model_load_ms", fixed_double(static_cast<double>(model_load_ns) / 1e6, 6));
    j.set("prefill_ms", fixed_double(static_cast<double>(prefill_ns) / 1e6, 6));
    j.set("decode_ms", fixed_double(static_cast<double>(decode_ns) / 1e6, 6));
    j.set("total_ms", fixed_double(static_cast<double>(total_ns) / 1e6, 6));
    j.set("prefill_tokens_per_s", fixed_double(prefill_tokens_per_s, 6));
    j.set("decode_tokens_per_s", fixed_double(decode_tokens_per_s, 6));
    j.set("first_token_latency_ms", fixed_double(first_token_latency_ms, 6));
    j.set("median_inter_token_ms", fixed_double(median_inter_token_ms, 6));
    j.set("p95_inter_token_ms", fixed_double(p95_inter_token_ms, 6));
    j.set("prompt_tokens", prompt_tokens);
    j.set("generated_tokens", generated_tokens);
    j.set("prompt_count", prompt_count);
    j.set("cache_state", cache_state);
    return j;
}

// ---------------------------------------------------------------------------
// Engine lifecycle
// ---------------------------------------------------------------------------

Engine::Engine(RunConfig config) : config_(std::move(config)) {}
Engine::~Engine() = default;

void Engine::initialize() {
    rng_.seed(config_.sampling.seed);
    if (initialized_) return;
    Timer load_timer;
    file_ = std::make_unique<MappedFile>(config_.model_path);
    doc_ = parse_gguf(file_->data(), file_->size());
    manifest_ = build_manifest(doc_, config_.model_path);
    // A smaller KV context is a memory decision, not a correctness one: the
    // model still runs with its real weights and its real arithmetic.
    if (config_.context_limit != 0 && config_.context_limit < manifest_.config.context)
        manifest_.config.context = config_.context_limit;
    trace_ = std::make_unique<TraceWriter>(config_.trace_path);

    config_.budget.requested_slots = config_.cache_slots;
    config_.residency.t_wait_ms = config_.t_wait_ms;
    config_.residency.resident_mode = config_.reader_mode == ReaderMode::Resident;
    config_.residency.max_prefetch = std::min(config_.residency.max_prefetch, config_.queue_depth);
    if (config_.residency.host_cache_records == 0) config_.residency.host_cache_records = 1;
    if (config_.residency.resident_mode) config_.residency.enable_prefetch = false;

    backend_ = Backend::create(config_.backend, trace_.get());

    SchedulerConfig scheduler_config;
    scheduler_config.mode = config_.reader_mode;
    scheduler_config.path = config_.model_path;
    scheduler_config.expected_identity = file_->identity();
    scheduler_config.queue_depth = config_.queue_depth;
    scheduler_config.coalesce_gap = config_.coalesce_gap;
    scheduler_config.force_direct_fallback = config_.force_direct_fallback;
    auto reader = make_reader(config_.reader_mode, config_.model_path, file_.get(),
                              config_.force_direct_fallback);
    scheduler_ = std::make_unique<ReadScheduler>(scheduler_config, std::move(reader), trace_.get());

    budget_ = compute_budget(manifest_, config_.budget, backend_->info());
    residency_ = std::make_unique<ResidencyManager>(manifest_, *backend_, *scheduler_, *trace_,
                                                    budget_, config_.residency);
    load_fixed_weights();
    initialize_tokenizer();
    residency_->initialize();
    residency_->preload_resident_experts();

    run_metrics_.model_load_ns = load_timer.elapsed_ns();
    initialized_ = true;
    Json fields = Json::object();
    fields.set("path", config_.model_path);
    fields.set("file_size", doc_.file_size);
    fields.set("adapter", manifest_.config.adapter);
    fields.set("model_kind", manifest_.config.model_kind);
    fields.set("backend", backend_->name());
    fields.set("reader", reader_mode_name(config_.reader_mode));
    fields.set("load_ms", fixed_double(static_cast<double>(run_metrics_.model_load_ns) / 1e6, 6));
    trace_->event("model_load", std::move(fields));
}

void Engine::initialize_tokenizer() {
    tokenizer_ = std::make_unique<Tokenizer>(
        Tokenizer::from_gguf(manifest_.doc, manifest_.config.vocab));
    Json fields = Json::object();
    fields.set("model", tokenizer_->kind_name());
    fields.set("vocab", static_cast<uint64_t>(manifest_.config.vocab));
    trace_->event("tokenizer_ready", std::move(fields));
}

void Engine::load_fixed_weights() {
    if (manifest_.fixed_weight_bytes == 0)
        fail(ErrorCode::INVALID_ARGUMENT, "model has no fixed weights");
    weights_buffer_ = backend_->allocate(manifest_.fixed_weight_bytes, AllocationTag::Weights);
    kv_buffer_ = backend_->allocate(budget_.kv_bytes, AllocationTag::KV);
    workspace_buffer_ = backend_->allocate(budget_.workspace_bytes, AllocationTag::Workspace);
    staging_buffer_ = backend_->allocate(budget_.staging_bytes, AllocationTag::Staging);
    kv_ = kv_buffer_->f32();
    // A recurrent model's state must start at zero: the convolution history and
    // the scan state are read before they are written. The KV cache does not
    // care, but it costs the same single pass and removes a class of
    // uninitialised-read bugs from the first token.
    if (!manifest_.config.ssm_kind.empty()) {
        const uint64_t attention_floats =
            manifest_.config.attention_cache_floats(manifest_.config.context);
        std::memset(kv_ + attention_floats, 0,
                    static_cast<size_t>(manifest_.config.recurrent_state_floats() *
                                        sizeof(float)));
    }

    std::vector<uint8_t> host_weights(static_cast<size_t>(manifest_.fixed_weight_bytes));
    uint64_t offset = 0;
    for (const auto &tensor : manifest_.doc.tensors) {
        if (tensor.expert_tensor) continue;
        const std::string &key = tensor.canonical.empty() ? tensor.name : tensor.canonical;
        weight_offsets_[key] = offset;
        std::memcpy(host_weights.data() + offset, file_->data() + tensor.absolute_offset,
                    static_cast<size_t>(tensor.byte_size));
        offset += tensor.byte_size;
    }
    // Decode the 1-D vectors that are not weight matrices: RMSNorm scales and
    // the optional attention biases. Both may be stored in a block format.
    for (const auto &tensor : manifest_.doc.tensors) {
        if (tensor.expert_tensor) continue;
        // Per-expert bias columns are 2-D ([width, n_expert]) and live in the
        // fixed buffer, unlike the stacked expert weights they belong to, so
        // they are decoded here rather than paged.
        const bool proj_bias_2d = tensor.role == TensorRole::FFN_GATE_EXPS_BIAS ||
                                  tensor.role == TensorRole::FFN_UP_EXPS_BIAS ||
                                  tensor.role == TensorRole::FFN_DOWN_EXPS_BIAS;
        if (tensor.dims.size() != 1 && !proj_bias_2d) continue;
        // Every 1-D vector is either a norm scale (read as a scale, possibly
        // with a bias) or a projection bias (added). phi2 adds biases to the
        // norms, the FFN and the output, so this is a role test rather than a
        // list of attention-only roles.
        const bool is_bias = tensor.role == TensorRole::ATTN_Q_BIAS ||
                             tensor.role == TensorRole::ATTN_K_BIAS ||
                             tensor.role == TensorRole::ATTN_V_BIAS ||
                             tensor.role == TensorRole::ATTN_OUTPUT_BIAS ||
                             tensor.role == TensorRole::ATTN_QKV_BIAS ||
                             tensor.role == TensorRole::ATTN_NORM_BIAS ||
                             tensor.role == TensorRole::ATTN_NORM_2_BIAS ||
                             tensor.role == TensorRole::FFN_NORM_BIAS ||
                             tensor.role == TensorRole::OUTPUT_NORM_BIAS ||
                             tensor.role == TensorRole::TOKEN_EMBD_NORM_BIAS ||
                             tensor.role == TensorRole::FFN_UP_BIAS ||
                             tensor.role == TensorRole::FFN_DOWN_BIAS ||
                             tensor.role == TensorRole::FFN_GATE_BIAS ||
                             tensor.role == TensorRole::OUTPUT_BIAS ||
                             tensor.role == TensorRole::FFN_EXP_PROBS_B ||
                             tensor.role == TensorRole::FFN_ROUTER_BIAS ||
                             tensor.role == TensorRole::SSM_CONV1D_BIAS ||
                             tensor.role == TensorRole::SSM_DT_BIAS ||
                             proj_bias_2d;
        if (tensor.type_id == 0 && !is_bias) continue; // already plain floats
        size_t count = 1;
        for (uint32_t d : tensor.dims) count *= d;
        std::vector<float> values(count);
        dequantize_row(tensor.type_id, file_->data() + tensor.absolute_offset,
                       values.data(), count);
        const std::string key = tensor.canonical.empty() ? tensor.name : tensor.canonical;
        if (is_bias) decoded_biases_.emplace(key, std::move(values));
        else decoded_norms_.emplace(key, std::move(values));
    }
    if (offset != manifest_.fixed_weight_bytes)
        fail(ErrorCode::INTERNAL_ERROR, "fixed weight size accounting mismatch");
    Fence fence = backend_->upload(*weights_buffer_, host_weights.data(), host_weights.size());
    backend_->wait_fence(fence, config_.t_wait_ms);
    backend_->record_control("fixed_weight_upload", host_weights.size(), host_weights.size());
    Json fields = Json::object();
    fields.set("bytes", host_weights.size());
    fields.set("tensors", static_cast<uint64_t>(weight_offsets_.size()));
    trace_->event("weights_uploaded", std::move(fields));
}

const float *Engine::bias_weights(const std::string &name) const {
    auto it = decoded_biases_.find(name);
    if (it == decoded_biases_.end())
        fail(ErrorCode::MISSING_REQUIRED_TENSOR, "attention bias is not resident: " + name);
    return it->second.data();
}

bool Engine::has_bias(const std::string &name) const {
    return decoded_biases_.find(name) != decoded_biases_.end();
}

std::optional<float> Engine::scalar_weight(const std::string &name) const {
    std::string resolved = name;
    auto alias = manifest_.alias.find(name);
    if (alias != manifest_.alias.end()) resolved = alias->second;
    // A one-element vector that arrived in a block format was decoded at load
    // time (it is 1-D and not a bias), so both stores have to be consulted.
    auto decoded = decoded_norms_.find(resolved);
    if (decoded != decoded_norms_.end()) {
        if (decoded->second.empty())
            fail(ErrorCode::SHAPE_MISMATCH, "scalar tensor is empty: " + name);
        return decoded->second.front();
    }
    auto it = weight_offsets_.find(resolved);
    if (it == weight_offsets_.end()) return std::nullopt;
    return *reinterpret_cast<const float *>(
        static_cast<const uint8_t *>(weights_buffer_->data) + it->second);
}

bool Engine::apply_projection_scale(std::vector<float> &vec, const std::string &name,
                                    const std::string &op) {
    const std::optional<float> scale = scalar_weight(name);
    if (!scale) return false;
    const float s = *scale;
    for (float &v : vec) v *= s;
    backend_->record_control(op, vec.size() * sizeof(float), vec.size() * sizeof(float));
    return true;
}

const float *Engine::norm_weights(const std::string &name) const {
    auto decoded = decoded_norms_.find(name);
    if (decoded != decoded_norms_.end()) return decoded->second.data();
    return weights(name);
}

const float *Engine::weights(const std::string &name) const {
    auto it = weight_offsets_.find(name);
    if (it == weight_offsets_.end())
        fail(ErrorCode::MISSING_REQUIRED_TENSOR, "tensor is not resident in fixed weight buffer: " + name);
    return reinterpret_cast<const float *>(static_cast<const uint8_t *>(weights_buffer_->data) + it->second);
}

Engine::TensorView Engine::tensor_view(const std::string &name) const {
    // Shared tensors (tied embeddings) are addressed through an alias so the
    // forward pass keeps one name per logical weight.
    std::string resolved = name;
    auto alias = manifest_.alias.find(name);
    if (alias != manifest_.alias.end()) resolved = alias->second;
    auto it = weight_offsets_.find(resolved);
    if (it == weight_offsets_.end())
        fail(ErrorCode::MISSING_REQUIRED_TENSOR,
             "tensor is not resident in fixed weight buffer: " + name);
    const TensorInfo *info = manifest_.tensor(resolved);
    if (!info) fail(ErrorCode::MISSING_REQUIRED_TENSOR, "tensor is not in the manifest: " + name);
    TensorView view;
    view.data = static_cast<const uint8_t *>(weights_buffer_->data) + it->second;
    view.type_id = info->type_id;
    return view;
}

std::vector<float> Engine::matvec_view(const TensorView &view, const std::vector<float> &x,
                                       uint32_t out_dim, uint32_t in_dim,
                                       const std::string &op) {
    std::vector<float> y(out_dim, 0.0f);
    if (view.type_id == 0) {
        backend_->matvec(reinterpret_cast<const float *>(view.data), x.data(), y.data(),
                         out_dim, in_dim, op);
    } else {
        KernelConfig kernels;
        kernels.requested = config_.backend.kernel_isa;
        kernels.threads = config_.backend.kernel_threads;
        gemv_quant(kernels, view.type_id, view.data, x.data(), y.data(), out_dim, in_dim);
        backend_->record_control(op, in_dim * static_cast<uint64_t>(out_dim) *
                                          quant_geometry(view.type_id).block_bytes /
                                          quant_geometry(view.type_id).block_size,
                                 out_dim * sizeof(float), "cpu-quant");
    }
    return y;
}

std::vector<float> Engine::matvec(const float *w, const std::vector<float> &x,
                                  uint32_t out_dim, uint32_t in_dim, const std::string &op) {
    std::vector<float> y(out_dim, 0.0f);
    backend_->matvec(w, x.data(), y.data(), out_dim, in_dim, op);
    return y;
}

std::vector<float> Engine::softmax(const std::vector<float> &values) const {
    std::vector<float> out(values.size(), 0.0f);
    if (values.empty()) return out;
    float max_value = *std::max_element(values.begin(), values.end());
    double sum = 0.0;
    std::vector<double> exps(values.size());
    for (size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i]))
            fail(ErrorCode::INTERNAL_ERROR, "non-finite value in softmax");
        exps[i] = std::exp(static_cast<double>(values[i]) - max_value);
        sum += exps[i];
    }
    for (size_t i = 0; i < values.size(); ++i)
        out[i] = static_cast<float>(exps[i] / sum);
    return out;
}

std::vector<uint32_t> Engine::topk(const std::vector<float> &scores, uint32_t k) const {
    std::vector<uint32_t> order(scores.size());
    for (uint32_t i = 0; i < order.size(); ++i) order[i] = i;
    std::stable_sort(order.begin(), order.end(), [&](uint32_t a, uint32_t b) {
        // A NaN, or a positive infinity, means the router has diverged. A
        // negative infinity is the documented way to mask an expert out (group
        // limited routing does exactly that), so it is allowed through.
        auto bad = [](float v) { return std::isnan(v) || v == std::numeric_limits<float>::infinity(); };
        if (bad(scores[a]) || bad(scores[b]))
            fail(ErrorCode::INTERNAL_ERROR, "non-finite router score");
        if (scores[a] == scores[b]) return a < b;
        return scores[a] > scores[b];
    });
    if (k > order.size()) fail(ErrorCode::INVALID_ARGUMENT, "top-k exceeds expert count");
    order.resize(k);
    return order;
}

uint32_t Engine::greedy(const std::vector<float> &logits) const {
    if (logits.empty()) fail(ErrorCode::INTERNAL_ERROR, "empty logits");
    uint32_t best = 0;
    for (uint32_t i = 1; i < logits.size(); ++i) {
        if (!std::isfinite(logits[i])) fail(ErrorCode::INTERNAL_ERROR, "non-finite logit");
        if (logits[i] > logits[best]) best = i;
    }
    return best;
}

void Engine::rope_half(std::vector<float> &q, uint32_t q_heads, std::vector<float> &k,
                       uint32_t k_heads, uint64_t position, double freq_base) const {
    const ModelConfig &c = manifest_.config;
    // Rotary dimensions come from the file when present, otherwise the whole
    // head. Rotating more than rope_dim would be wrong, not merely slower.
    const uint32_t rotate = std::min(c.rope_dim ? c.rope_dim : c.head_dim, c.head_dim);
    const uint32_t pairs = rope_pairs(rotate);
    // Ramp bounds and magnitude scale are position-independent, so they are
    // derived once per call rather than once per pair.
    RopePlan plan;
    plan.init(c, rotate, freq_base);
    // Layer 0 carries the divisors and llama.cpp duplicates them, so one vector
    // describes every layer. Dividing by them stretches those pairs' positions,
    // which is how granite/llama4-style checkpoints extend context without YaRN.
    if (c.rope_freqs) plan.freq_factors = weights("blk.0.rope_freqs.weight");
    // mRoPE reads one position per axis. A text token occupies the same place on
    // all four, so the text path fills the array with the single position; the
    // section assignment then cannot change the result, which is exactly why the
    // two agree with llama.cpp on text-only prompts.
    const std::array<double, 4> positions = {static_cast<double>(position),
                                             static_cast<double>(position),
                                             static_cast<double>(position),
                                             static_cast<double>(position)};
    // Query and key can have different head counts under grouped-query
    // attention, so each vector is walked independently.
    for (uint32_t h = 0; h < std::max(q_heads, k_heads); ++h) {
        const bool in_q = h < q_heads;
        const bool in_k = h < k_heads;
        if (!in_q && !in_k) break;
        size_t base = static_cast<size_t>(h) * c.head_dim;
        for (uint32_t i = 0; i < pairs; ++i) {
            double cs = 0.0;
            double sn = 0.0;
            plan.rotation(i, positions, cs, sn);
            for (std::vector<float> *vec : {in_q ? &q : nullptr, in_k ? &k : nullptr}) {
                if (!vec) continue;
                float a0 = (*vec)[base + 2 * i];
                float a1 = (*vec)[base + 2 * i + 1];
                (*vec)[base + 2 * i] = static_cast<float>(a0 * cs - a1 * sn);
                (*vec)[base + 2 * i + 1] = static_cast<float>(a0 * sn + a1 * cs);
            }
        }
    }
    backend_->record_control("rope_half", (q.size() + k.size()) * sizeof(float),
                             (q.size() + k.size()) * sizeof(float));
}


uint32_t Engine::sample(const std::vector<float> &logits) {
    return sample_from_logits(logits, config_.sampling, rng_);
}

std::vector<float> apply_sampling_bias(const std::vector<float> &logits,
                                       const SamplingConfig &config,
                                       const std::vector<int32_t> &history) {
    std::vector<float> biased = logits;
    const bool any_penalty = config.repetition_penalty != 1.0 ||
                             config.presence_penalty != 0.0 ||
                             config.frequency_penalty != 0.0;
    if (any_penalty && !history.empty()) {
        // A window of 0 is llama.cpp's `penalty_last_n = -1`: the whole history.
        size_t first = 0;
        if (config.penalty_window > 0 && history.size() > config.penalty_window)
            first = history.size() - config.penalty_window;
        std::map<uint32_t, uint32_t> counts;
        for (size_t i = first; i < history.size(); ++i) {
            if (history[i] < 0) continue;
            ++counts[static_cast<uint32_t>(history[i])];
        }
        for (const auto &[token, count] : counts) {
            if (token >= biased.size()) continue;  // a prompt token outside this vocabulary
            float &value = biased[token];
            if (config.repetition_penalty != 1.0) {
                // Upstream divides the positive logits and multiplies the
                // negative ones, so the penalty always moves a token *down*.
                value = value > 0.0f
                            ? static_cast<float>(value / config.repetition_penalty)
                            : static_cast<float>(value * config.repetition_penalty);
            }
            if (config.presence_penalty != 0.0)
                value = static_cast<float>(value - config.presence_penalty);
            if (config.frequency_penalty != 0.0)
                value = static_cast<float>(value - config.frequency_penalty * count);
        }
    }
    for (const auto &[token, bias] : config.logit_bias) {
        if (token >= biased.size())
            fail(ErrorCode::INVALID_ARGUMENT,
                 "logit bias names token " + std::to_string(token) + " outside a vocabulary of " +
                     std::to_string(biased.size()));
        biased[token] = static_cast<float>(static_cast<double>(biased[token]) + bias);
    }
    return biased;
}

namespace {

// One sampling pass over already-biased logits. Shared by the two public
// entry points so the greedy rule, the truncation order and the reported
// probabilities cannot drift apart.
SampledToken sample_core(const std::vector<float> &logits, const SamplingConfig &sampling,
                         std::mt19937_64 &rng) {
    if (logits.empty()) fail(ErrorCode::INVALID_ARGUMENT, "cannot sample from empty logits");
    SampledToken out;
    if (sampling.greedy()) {
        // Greedy inside the sampler: highest logit, lowest id on ties.
        uint32_t best = 0;
        for (uint32_t i = 1; i < logits.size(); ++i)
            if (logits[i] > logits[best]) best = i;
        out.id = best;
        if (sampling.n_probs > 0) out.alternatives.emplace_back(best, 1.0);
        return out;
    }
    const double temperature = sampling.temperature;

    std::vector<std::pair<double, uint32_t>> scored;
    scored.reserve(logits.size());
    for (uint32_t i = 0; i < logits.size(); ++i) {
        const float v = logits[i];
        if (!std::isfinite(v))
            fail(ErrorCode::INVALID_ARGUMENT, "non-finite logit during sampling");
        scored.emplace_back(static_cast<double>(v) / temperature, i);
    }
    // Stable ordering: higher score first, lower token id wins ties.
    std::stable_sort(scored.begin(), scored.end(),
                     [](const std::pair<double, uint32_t> &a,
                        const std::pair<double, uint32_t> &b) {
                         if (a.first != b.first) return a.first > b.first;
                         return a.second < b.second;
                     });
    if (sampling.top_k > 0 && scored.size() > sampling.top_k)
        scored.resize(sampling.top_k);

    // Softmax over the surviving candidates.
    const double max_score = scored.empty() ? 0.0 : scored.front().first;
    double total = 0.0;
    std::vector<double> probs(scored.size(), 0.0);
    for (size_t i = 0; i < scored.size(); ++i) {
        probs[i] = std::exp(scored[i].first - max_score);
        total += probs[i];
    }
    if (!(total > 0.0)) fail(ErrorCode::INTERNAL_ERROR, "sampling probability underflow");
    for (double &p : probs) p /= total;

    // min-p keeps the tokens whose probability is at least `min_p` times the
    // most likely token's. It runs before the nucleus cut, as upstream's
    // sampler chain does, and always keeps one candidate.
    if (sampling.min_p > 0.0 && !probs.empty()) {
        const double floor_prob = sampling.min_p * probs.front();
        size_t keep = 1;
        while (keep < probs.size() && probs[keep] >= floor_prob) ++keep;
        scored.resize(keep);
        probs.resize(keep);
        total = 0.0;
        for (double p : probs) total += p;
        for (double &p : probs) p /= total;
    }

    // Nucleus truncation, keeping at least one candidate.
    if (sampling.top_p > 0.0 && sampling.top_p < 1.0) {
        double cumulative = 0.0;
        size_t keep = 0;
        for (; keep < probs.size(); ++keep) {
            cumulative += probs[keep];
            if (cumulative >= sampling.top_p) {
                ++keep;
                break;
            }
        }
        keep = std::max<size_t>(1, std::min(keep, probs.size()));
        scored.resize(keep);
        probs.resize(keep);
        total = 0.0;
        for (double p : probs) total += p;
        for (double &p : probs) p /= total;
    }

    std::uniform_real_distribution<double> uniform(0.0, 1.0);
    const double target = uniform(rng);
    size_t chosen = probs.size() - 1;
    double cumulative = 0.0;
    for (size_t i = 0; i < probs.size(); ++i) {
        cumulative += probs[i];
        if (target <= cumulative) {
            chosen = i;
            break;
        }
    }
    out.id = scored[chosen].second;
    if (sampling.n_probs > 0) {
        const size_t report = std::min<size_t>(sampling.n_probs, scored.size());
        out.alternatives.reserve(report);
        for (size_t i = 0; i < report; ++i)
            out.alternatives.emplace_back(scored[i].second, probs[i]);
    }
    return out;
}

}  // namespace

uint32_t sample_from_logits(const std::vector<float> &logits, const SamplingConfig &sampling,
                            std::mt19937_64 &rng) {
    return sample_core(logits, sampling, rng).id;
}

SampledToken sample_detailed(const std::vector<float> &logits, const SamplingConfig &sampling,
                             std::mt19937_64 &rng) {
    return sample_core(logits, sampling, rng);
}

std::string Engine::detokenize(const std::vector<int32_t> &ids) const {
    const ModelConfig &c = manifest_.config;
    return tokenizer_->decode(ids, c.bos_id, c.eos_id);
}

ChatRender Engine::render_chat(const std::vector<ChatMessage> &messages,
                                bool add_generation_prompt,
                                const std::string &template_override) const {
    if (!initialized_) fail(ErrorCode::INTERNAL_ERROR, "engine is not initialized");
    const ModelConfig &c = manifest_.config;

    ChatRender render;
    std::string template_text;
    if (!template_override.empty()) {
        template_text = resolve_chat_template(std::string(), std::string(), template_override);
        const std::string name = builtin_chat_template_by_name(template_override).empty()
                                     ? std::string()
                                     : template_override;
        render.template_source = name.empty() ? "override" : "builtin:" + name;
        render.template_name = name;
    } else if (c.chat_template == "raw-v1") {
        // The native `freemoe-tiny` adapter stores a protocol constant here, not
        // Jinja: its prompts are raw bytes. Rendering it as a template would
        // emit the literal string "raw-v1".
        template_text = builtin_chat_template("raw");
        render.template_source = "builtin:raw";
        render.template_name = "raw";
    } else if (!c.chat_template.empty()) {
        template_text = c.chat_template;
        render.template_source = "checkpoint";
    } else {
        // No template in the file: pick one from the token table, which is
        // evidence, and fall back to the architecture only when the vocabulary
        // carries none of the known marker sets.
        const std::string name = builtin_chat_template_from_tokens(
            c.architecture, tokenizer_->contains_piece("<|start_header_id|>"),
            tokenizer_->contains_piece("<|im_start|>"),
            tokenizer_->contains_piece("<start_of_turn>"),
            tokenizer_->contains_piece("<|user|>") && tokenizer_->contains_piece("<|end|>"),
            tokenizer_->contains_piece("[INST]"),
            tokenizer_->contains_piece("<|user|>") && tokenizer_->contains_piece("<|endoftext|>"));
        // `name` is a template name, not an architecture: look it up directly.
        if (!name.empty()) template_text = builtin_chat_template_by_name(name);
        render.template_source = name.empty() ? "" : "builtin:" + name;
        render.template_name = name;
    }
    if (template_text.empty()) {
        fail(ErrorCode::INVALID_ARGUMENT,
             "this checkpoint carries no chat template and no built-in is known for architecture '" +
                 c.architecture +
                 "'; pass --chat-template NAME or --chat-template-file PATH, or run without --chat");
    }

    ChatTemplateConfig config;
    config.add_generation_prompt = add_generation_prompt;
    config.bos_token = tokenizer_->piece(c.bos_id);
    config.eos_token = tokenizer_->piece(c.eos_id);
    render.prompt = render_chat_template(template_text, messages, config);
    return render;
}

std::vector<int32_t> Engine::tokenize(const std::string &prompt) const {
    const ModelConfig &c = manifest_.config;
    if (prompt.empty() || prompt.size() + 2 > c.context)
        fail(ErrorCode::INVALID_ARGUMENT, "prompt is empty or exceeds context");
    return tokenizer_->encode(prompt, c.bos_id);
}

std::vector<float> Engine::forward_token(int32_t token, uint64_t position,
                                         const std::string &phase,
                                         std::vector<RoutingEvent> *routing) {
    const ModelConfig &c = manifest_.config;
    if (token < 0 || static_cast<uint32_t>(token) >= c.vocab)
        fail(ErrorCode::INVALID_ARGUMENT, "token id out of range");
    if (position >= c.context)
        fail(ErrorCode::INVALID_ARGUMENT, "KV context exhausted");

    residency_->poll_prefetches();
    residency_->cancel_stale_prefetches(position);
    {
        Json fields = Json::object();
        fields.set("position", position);
        fields.set("token", static_cast<int64_t>(token));
        fields.set("phase", phase);
        trace_->event("token", std::move(fields));
    }

    // gemma stores its norm scales zero-centred, so the effective scale is
    // 1 + w. Reading them as w would leave the model roughly un-normalised.
    const double norm_bias = c.norm_add_one ? 1.0 : 0.0;
    auto rmsnorm = [&](const std::vector<float> &x, const float *w, const std::string &name) {
        double ss = 0.0;
        for (float v : x) ss += static_cast<double>(v) * v;
        // RMS over the vector's own length, not the hidden size: MLA normalises
        // the q and kv latents, which are narrower than `hidden`, and dividing
        // by `hidden` scales those by sqrt(hidden/width).
        double scale = 1.0 / std::sqrt(ss / static_cast<double>(x.size()) + c.rms_eps);
        std::vector<float> y(x.size());
        for (uint32_t i = 0; i < x.size(); ++i) {
            // A null weight is upstream's `build_norm` with no weight tensor:
            // the vector is normalised and not scaled, i.e. unit gain. That is
            // how `ssm.dt_b_c_rms` behaves when it is set without the weights.
            const float gain = w == nullptr ? 1.0f : static_cast<float>(norm_bias + w[i]);
            y[i] = static_cast<float>(x[i] * scale * gain);
        }
        backend_->record_control(name, x.size() * sizeof(float) * 2, y.size() * sizeof(float));
        return y;
    };
    // LayerNorm: mean-centred, with a bias. phi2 is the first architecture here
    // that needs it, and the norm type is chosen from the checkpoint's epsilon
    // key rather than from the architecture name.
    auto layernorm = [&](const std::vector<float> &x, const float *w, const float *b,
                         const std::string &name) {
        double mean = 0.0;
        for (float v : x) mean += v;
        mean /= x.size();
        double var = 0.0;
        for (float v : x) {
            const double d = static_cast<double>(v) - mean;
            var += d * d;
        }
        var /= x.size();
        const double scale = 1.0 / std::sqrt(var + c.rms_eps);
        std::vector<float> y(x.size());
        for (uint32_t i = 0; i < x.size(); ++i) {
            // A null weight is upstream's `build_norm(x, NULL, NULL, ...)`: the
            // vector is normalised and not scaled (olmo, talkie).
            const double gain = w == nullptr ? 1.0 : static_cast<double>(w[i]);
            y[i] = static_cast<float>((static_cast<double>(x[i]) - mean) * scale * gain +
                                      (b == nullptr ? 0.0 : static_cast<double>(b[i])));
        }
        backend_->record_control(name, x.size() * sizeof(float) * 3, y.size() * sizeof(float));
        return y;
    };
    // One call site for both norm types keeps every layer honest about which
    // one the checkpoint asked for.
    auto norm = [&](const std::vector<float> &x, const std::string &prefix,
                    const std::string &label) {
        const std::string w = prefix + ".weight";
        const std::string b = prefix + ".bias";
        // llama.cpp passes the bias pointer through unchanged, so a LayerNorm
        // with no `*.bias` tensor is a bias-free LayerNorm, not a broken file
        // (cohere2's decoder passes NULL for both norms).
        if (c.weightless_norm)
            return c.layer_norm ? layernorm(x, nullptr, nullptr, label)
                                : rmsnorm(x, nullptr, label);
        if (c.layer_norm)
            return layernorm(x, norm_weights(w), has_bias(b) ? bias_weights(b) : nullptr, label);
        return rmsnorm(x, norm_weights(w), label);
    };
    // The activation is a per-family fact read from the checkpoint: llama's
    // SwiGLU, gpt-neox's GELU, gemma's GeGLU. Both the gated and the gate-free
    // form live in include/fm/activation.hpp, so there is one implementation of
    // each and the clamp variants are testable in isolation.
    if (!is_known_activation(c.activation))
        fail(ErrorCode::NOT_IMPLEMENTED, "unsupported activation '" + c.activation + "'");
    // ALiBi slopes, exactly as `ggml_compute_forward_soft_max_f32` derives them:
    // n_head_log2 = 2^floor(log2(n_head)); m0 = 2^(-max_bias/n_head_log2);
    // m1 = 2^(-max_bias/2/n_head_log2); slope = m0^(h+1) for the first
    // n_head_log2 heads and m1^(2(h-n_head_log2)+1) for the rest. The bias is
    // slope * -(query_pos - key_pos), added to the scaled score inside the
    // softmax. Computed once and reused by every layer and position.
    std::vector<double> alibi_slopes;
    if (c.alibi_max_bias > 0.0 && c.heads > 0) {
        uint32_t n_head_log2 = 1;
        while (n_head_log2 * 2 <= c.heads) n_head_log2 *= 2;
        const double m0 = std::pow(2.0, -(c.alibi_max_bias) / n_head_log2);
        const double m1 = std::pow(2.0, -(c.alibi_max_bias / 2.0) / n_head_log2);
        alibi_slopes.resize(c.heads);
        for (uint32_t h = 0; h < c.heads; ++h)
            alibi_slopes[h] = h < n_head_log2 ? std::pow(m0, static_cast<double>(h) + 1.0)
                                              : std::pow(m1, 2.0 * static_cast<double>(h - n_head_log2) + 1.0);
    }
    auto activate = [&](float x) -> float { return pointwise_activation(c, x); };
    // The gated activation lives in include/fm/activation.hpp so the two clamp
    // forms can be tested against hand-computed values instead of only through
    // a whole-checkpoint logit comparison.
    auto gated = [&](float gate, float up, uint32_t layer) -> float {
        return gated_activation(c, gate, up, layer);
    };
    auto residual_add = [&](std::vector<float> &x, const std::vector<float> &delta, const std::string &name) {
        for (size_t i = 0; i < x.size(); ++i) x[i] += delta[i];
        backend_->record_control(name, (x.size() + delta.size()) * sizeof(float), x.size() * sizeof(float));
    };

    std::vector<float> x(c.hidden);
    {
        // Embedding lookup must respect the stored block type: a quantised
        // token table cannot be read as raw floats.
        const TensorView emb = tensor_view("token_embd.weight");
        if (emb.type_id == 0) {
            std::memcpy(x.data(),
                        reinterpret_cast<const float *>(emb.data) +
                            static_cast<size_t>(token) * c.hidden,
                        static_cast<size_t>(c.hidden) * sizeof(float));
        } else {
            const QuantGeometry &geo = quant_geometry(emb.type_id);
            if (c.hidden % geo.block_size != 0)
                fail(ErrorCode::SHAPE_MISMATCH, "embedding row is not whole quantisation blocks");
            const uint64_t row_bytes = (c.hidden / geo.block_size) * geo.block_bytes;
            const uint8_t *row = emb.data + static_cast<uint64_t>(token) * row_bytes;
            for (uint32_t k = 0; k < c.hidden; k += geo.block_size) {
                const uint32_t block_index = k / geo.block_size;
                dequantize_block(emb.type_id, row + block_index * geo.block_bytes, x.data() + k);
            }
        }
        if (c.embedding_scale != 1.0) {
            // gemma scales by sqrt(hidden), minicpm by a fixed constant. Both
            // happen before the first block, so skipping it changes every
            // subsequent activation.
            const float scale = static_cast<float>(c.embedding_scale);
            for (float &v : x) v *= scale;
        }
        if (c.token_embd_norm) {
            // bloom normalises the embedding once, before block 0
            // (models/bloom.cpp's `inpL = build_norm(inpL, tok_norm, ...)`).
            // It is not a per-layer norm and has no `blk.N.` prefix, so it
            // cannot ride on the block's norm path.
            x = layernorm(x, norm_weights("token_embd_norm.weight"),
                          bias_weights("token_embd_norm.bias"), "layernorm.inp");
        }
        if (c.position_embd) {
            // gpt2 / starcoder: the position is a learned vector added to the
            // token embedding (`pos = get_rows(pos_embd, inp_pos)`), which is the
            // only place either architecture encodes position at all.
            const TensorInfo *table = manifest_.tensor("position_embd.weight");
            if (table == nullptr || table->dims.size() < 2)
                fail(ErrorCode::INTERNAL_ERROR, "position_embd.weight is not a table");
            const uint64_t rows = table->dims[1];
            if (position >= rows)
                fail(ErrorCode::METADATA_MISMATCH,
                     "position " + std::to_string(position) +
                         " is outside the checkpoint's position table (" +
                         std::to_string(rows) + " rows)");
            const TensorView view = tensor_view("position_embd.weight");
            if (view.type_id != 0)
                fail(ErrorCode::UNSUPPORTED_QUANTISATION,
                     "a quantised position table is not supported");
            const float *row = reinterpret_cast<const float *>(view.data) +
                               static_cast<size_t>(position) * c.hidden;
            for (uint32_t i = 0; i < c.hidden; ++i) x[i] += row[i];
            backend_->record_control("position_embd", x.size() * sizeof(float), 0);
        }
    }

    // Where the two halves of the cache region begin. A pure attention model has
    // no recurrent layers, a pure mamba model has no attention cache, and a
    // hybrid has both: the attention cache is layer-major over attention layers
    // only, and the recurrent states follow it.
    const uint64_t attention_floats = c.attention_cache_floats(c.context);
    // `ggml_compute_softplus_f32`: a plain log(1+e^x) below 20, and the identity
    // above it, where the exponential would overflow. The threshold is part of
    // the reference's arithmetic, not an optimisation.
    auto softplus = [](float v) -> float {
        return v > 20.0f ? v : static_cast<float>(std::log1p(std::exp(static_cast<double>(v))));
    };
    // A recurrent layer replaces attention entirely: a 1D convolution over the
    // projected input, then a state-space scan whose state rolls forward in the
    // activation buffer. llama.cpp keeps that state in its recurrent memory and
    // nothing in a kv cache, so the same region of this build's kv buffer holds
    // `ssm_state_floats()` floats per recurrent layer, in the same layout.
    auto ssm_block = [&](uint32_t layer, const std::string &prefix,
                         const std::vector<float> &h) -> std::vector<float> {
        const std::string lp = std::to_string(layer);
        const uint32_t inner = c.ssm_inner;
        const uint32_t state_size = c.ssm_state;
        const uint32_t conv_width = c.ssm_conv;
        const uint32_t channels = c.ssm_conv_channels();
        // The buffer holds the attention layers' per-position cache first, then
        // one rolling state per recurrent layer, so a hybrid stack can have both.
        float *slot = kv_ + attention_floats +
                      static_cast<size_t>(c.recurrent_slot(layer)) * c.ssm_state_floats();
        // d_conv - 1 columns of history, then the state itself.
        float *history = slot;
        float *ssm_state = slot + static_cast<size_t>(conv_width - 1) * channels;
        const uint32_t history_rows = conv_width - 1;

        const bool mamba2 = c.ssm_kind == "mamba2";
        const uint32_t in_dim =
            mamba2 ? inner + channels + c.ssm_dt_rank : 2 * inner;
        std::vector<float> projected = matvec_view(tensor_view(prefix + "ssm_in.weight"), h,
                                                   in_dim, c.hidden, "ssm_in.L" + lp);
        std::vector<float> z(projected.begin(), projected.begin() + inner);
        std::vector<float> conv_in;
        std::vector<float> dt;
        if (mamba2) {
            conv_in.assign(projected.begin() + inner, projected.begin() + inner + channels);
            // mamba2 projects dt inside `ssm_in` and stores only its bias, so
            // there is no dt weight to apply here (models/mamba2.cpp).
            dt.assign(projected.begin() + inner + channels, projected.end());
        } else {
            conv_in.assign(projected.begin() + inner, projected.end());
            // dt/B/C come from the convolved x, so they are produced below.
        }

        // 1D convolution: `ggml_ssm_conv` slides a window of d_conv columns over
        // the concatenated (history | new input) and dots it with the kernel.
        // The window is then rolled forward by one column.
        std::vector<float> convolved(channels, 0.0f);
        const float *kernel = weights(prefix + "ssm_conv1d.weight");
        auto convolve = [&](const std::vector<float> &current) {
            for (uint32_t i = 0; i < channels; ++i) {
                double acc = 0.0;
                for (uint32_t k = 0; k < conv_width; ++k) {
                    // Rows 0..d_conv-2 are history (oldest first); the last is
                    // this token.
                    const float value = k + 1 < conv_width
                                            ? history[static_cast<size_t>(k) * channels + i]
                                            : current[i];
                    acc += static_cast<double>(value) * kernel[k + conv_width * i];
                }
                convolved[i] = static_cast<float>(acc);
            }
            // Roll the history: drop the oldest column, append this token.
            for (uint32_t k = 0; k + 1 < history_rows; ++k)
                for (uint32_t i = 0; i < channels; ++i)
                    history[static_cast<size_t>(k) * channels + i] =
                        history[static_cast<size_t>(k + 1) * channels + i];
            if (history_rows > 0)
                for (uint32_t i = 0; i < channels; ++i)
                    history[static_cast<size_t>(history_rows - 1) * channels + i] = current[i];
        };
        convolve(conv_in);
        const float *conv_bias = bias_weights(prefix + "ssm_conv1d.bias");
        for (uint32_t i = 0; i < channels; ++i)
            convolved[i] = static_cast<float>(silu(static_cast<double>(convolved[i] + conv_bias[i])));
        // The convolution reads its kernel and the window it slides (history plus
        // this token's input) and writes one column.
        backend_->record_control("ssm_conv.L" + lp,
                                 static_cast<uint64_t>(channels) * (2 * conv_width - 1) *
                                     sizeof(float),
                                 channels * sizeof(float));

        const float *A = weights(prefix + "ssm_a");
        const float *D = weights(prefix + "ssm_d");
        std::vector<float> y(inner, 0.0f);
        if (mamba2) {
            // dt is already projected; B and C ride along in the convolution.
            // No dt norm here: upstream's `build_mamba2_layer` has none, and the
            // only norm in that block is the grouped `ssm_norm` after the scan.
            const float *dt_bias = bias_weights(prefix + "ssm_dt.bias");
            for (uint32_t i = 0; i < c.ssm_dt_rank; ++i) dt[i] += dt_bias[i];
            const uint32_t head_dim = inner / c.ssm_dt_rank;
            const uint32_t group = c.ssm_dt_rank / c.ssm_groups;
            const float *B = convolved.data() + inner;
            const float *C = convolved.data() + inner + c.ssm_groups * state_size;
            for (uint32_t head = 0; head < c.ssm_dt_rank; ++head) {
                const float dt_sp = softplus(dt[head]);
                const float dA = static_cast<float>(std::exp(dt_sp * A[head]));
                const uint32_t g = head / group;
                for (uint32_t i = 0; i < head_dim; ++i) {
                    const uint32_t ii = i + head * head_dim;
                    const float x_dt = convolved[ii] * dt_sp;
                    double acc = 0.0;
                    float *st = ssm_state + static_cast<size_t>(ii) * state_size;
                    for (uint32_t k = 0; k < state_size; ++k) {
                        const float value = st[k] * dA + B[k + g * state_size] * x_dt;
                        st[k] = value;
                        acc += static_cast<double>(value) * C[k + g * state_size];
                    }
                    y[ii] = static_cast<float>(acc) + convolved[ii] * D[head];
                }
            }
            for (uint32_t i = 0; i < inner; ++i) y[i] = static_cast<float>(silu(z[i])) * y[i];
            // The grouped RMS norm, one group of `inner / groups` channels each.
            const float *norm_w = norm_weights(prefix + "ssm_norm.weight");
            const uint32_t width = inner / c.ssm_groups;
            for (uint32_t g = 0; g < c.ssm_groups; ++g) {
                double acc = 0.0;
                for (uint32_t i = 0; i < width; ++i) {
                    const double v = y[g * width + i];
                    acc += v * v;
                }
                const double scale = 1.0 / std::sqrt(acc / width + c.rms_eps);
                for (uint32_t i = 0; i < width; ++i)
                    y[g * width + i] = static_cast<float>(y[g * width + i] * scale *
                                                          norm_w[i + width * g]);
            }
            backend_->record_control("ssm_norm.L" + lp, y.size() * sizeof(float) * 2,
                                     y.size() * sizeof(float));
        } else {
            // mamba-1: dt, B and C are projected from the convolved x.
            const uint32_t dt_rank = c.ssm_dt_rank;
            std::vector<float> dbc =
                matvec_view(tensor_view(prefix + "ssm_x.weight"), convolved,
                            dt_rank + 2 * state_size, inner, "ssm_x.L" + lp);
            std::vector<float> dt_in(dbc.begin(), dbc.begin() + dt_rank);
            // falcon-mamba and jamba normalise dt, B and C before the scan, when
            // the flag or the tensor set says so (`gguf.cpp` applies upstream's
            // `flag || all three` gate). A weight the file does not carry means
            // unit gain, which is what `build_norm` with a null weight does.
            if (c.ssm_vector_norms) {
                dt_in = rmsnorm(dt_in,
                                c.ssm_dt_norm_present
                                    ? norm_weights(prefix + "ssm_dt_norm.weight")
                                    : nullptr,
                                "ssm_dt_norm.L" + lp);
                std::vector<float> b_vec(dbc.begin() + dt_rank, dbc.begin() + dt_rank + state_size);
                std::vector<float> c_vec(dbc.begin() + dt_rank + state_size, dbc.end());
                b_vec = rmsnorm(b_vec,
                                c.ssm_b_norm_present
                                    ? norm_weights(prefix + "ssm_b_norm.weight")
                                    : nullptr,
                                "ssm_b_norm.L" + lp);
                c_vec = rmsnorm(c_vec,
                                c.ssm_c_norm_present
                                    ? norm_weights(prefix + "ssm_c_norm.weight")
                                    : nullptr,
                                "ssm_c_norm.L" + lp);
                std::copy(b_vec.begin(), b_vec.end(), dbc.begin() + dt_rank);
                std::copy(c_vec.begin(), c_vec.end(), dbc.begin() + dt_rank + state_size);
            }
            const float *B = dbc.data() + dt_rank;
            const float *C = dbc.data() + dt_rank + state_size;
            dt = matvec_view(tensor_view(prefix + "ssm_dt.weight"), dt_in, inner, dt_rank,
                             "ssm_dt.L" + lp);
            const float *dt_bias = bias_weights(prefix + "ssm_dt.bias");
            for (uint32_t i = 0; i < inner; ++i) dt[i] += dt_bias[i];
            // One channel is one head, and the state wears the decay per state
            // index: dA = exp(softplus(dt) * A[channel, k]).
            for (uint32_t h_index = 0; h_index < inner; ++h_index) {
                const float dt_sp = softplus(dt[h_index]);
                const float x_dt = convolved[h_index] * dt_sp;
                double acc = 0.0;
                float *st = ssm_state + static_cast<size_t>(h_index) * state_size;
                const float *a_row = A + static_cast<size_t>(h_index) * state_size;
                for (uint32_t k = 0; k < state_size; ++k) {
                    const float decay = static_cast<float>(std::exp(dt_sp * a_row[k]));
                    const float value = st[k] * decay + B[k] * x_dt;
                    st[k] = value;
                    acc += static_cast<double>(value) * C[k];
                }
                y[h_index] = static_cast<float>(acc) + convolved[h_index] * D[h_index];
            }
            // `ggml_swiglu_split(z, y)`: silu of the first argument times the
            // second, so the gate is the z half here.
            for (uint32_t i = 0; i < inner; ++i) y[i] = static_cast<float>(silu(z[i])) * y[i];
        }
        // One scan step: the state it read and the state it wrote, plus the
        // convolution output the scan consumed.
        backend_->record_control("ssm_scan.L" + lp,
                                 static_cast<uint64_t>(state_size) * inner * sizeof(float) * 2 +
                                     channels * sizeof(float),
                                 static_cast<uint64_t>(state_size) * inner * sizeof(float));
        return matvec_view(tensor_view(prefix + "ssm_out.weight"), y, c.hidden, inner,
                           "ssm_out.L" + lp);
    };

    for (uint32_t layer = 0; layer < c.layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        // olmo2 has no pre-norms: attention and the feed-forward read the residual
        // stream, and only their *outputs* are normalised (that is `post_norms`).
        std::vector<float> h = c.no_branch_norms
                                   ? x
                                   : norm(x, prefix + "attn_norm",
                                          "attn_norm.L" + std::to_string(layer));
        if (c.layer_is_recurrent(layer)) {
            residual_add(x, ssm_block(layer, prefix, h),
                         "residual.ssm.L" + std::to_string(layer));
        } else {
        // phi2 runs the FFN on the *attention's* normalised input and adds both
        // branches to the same residual, so `h` is reused rather than replaced.
        const std::vector<float> attn_normed = c.parallel_residual ? h : std::vector<float>();
        // Grouped-query attention: `kv_heads` key/value heads feed `heads`
        // query heads, each query head repeating a contiguous group.
        // Under MLA the value width differs from the key width, so the two are
        // tracked separately; for every other family v_dim == kv_dim.
        const uint32_t kv_dim = c.kv_heads * c.head_dim;
        const uint32_t v_dim = c.v_dim();
        // The cache stores one value per kv head, but every query head gets its
        // own output slice, so the attention output is heads*head_v wide. These
        // are the same under multi-head attention and differ under GQA.
        const uint32_t attn_out_dim = c.heads * c.head_v_dim();
        const uint32_t kv_stride = kv_dim + v_dim;
        const uint32_t group = c.heads / c.kv_heads;
        // falcon-40b normalises the block input a second time for the attention
        // projections, while the FFN's residual still runs off `attn_norm`.
        std::vector<float> h_attn;
        const std::vector<float> *attn_input = &h;
        if (c.attn_norm_2) {
            h_attn = norm(x, prefix + "attn_norm_2", "attn_norm_2.L" + std::to_string(layer));
            attn_input = &h_attn;
        }
        std::vector<float> q, k, v;
        if (c.mla) {
            // Multi-head latent attention. The file stores no keys or values: a
            // per-token latent is normalised, and each head expands it with its
            // own slice of k_b / v_b. llama.cpp keeps the latent in the cache
            // and absorbs k_b into the query; because k_b and v_b are linear and
            // the attention weights do not depend on them, expanding per head is
            // the same arithmetic, and it keeps one attention path for every
            // architecture. Reported as `mla_expand` in the operation log.
            std::vector<float> qflat;
            if (c.q_lora_rank != 0) {
                std::vector<float> qa = matvec_view(tensor_view(prefix + "attn_q_a.weight"), *attn_input,
                                                    c.q_lora_rank, c.hidden,
                                                    "attn_q_a.L" + std::to_string(layer));
                qa = rmsnorm(qa, norm_weights(prefix + "attn_q_a_norm.weight"),
                             "attn_q_a_norm.L" + std::to_string(layer));
                qflat = matvec_view(tensor_view(prefix + "attn_q_b.weight"), qa,
                                    c.heads * c.head_dim, c.q_lora_rank,
                                    "attn_q_b.L" + std::to_string(layer));
            } else {
                qflat = matvec_view(tensor_view(prefix + "attn_q.weight"), *attn_input,
                                    c.heads * c.head_dim, c.hidden,
                                    "attn_q.L" + std::to_string(layer));
            }
            const uint32_t latent = c.kv_lora_rank;
            std::vector<float> kvpe =
                matvec_view(tensor_view(prefix + "attn_kv_a_mqa.weight"), *attn_input, latent + c.rope_dim,
                            c.hidden, "attn_kv_a_mqa.L" + std::to_string(layer));
            std::vector<float> ckv(kvpe.begin(), kvpe.begin() + latent);
            std::vector<float> kpe(kvpe.begin() + latent, kvpe.end());
            // MLA rotates the same `rope_dim` slice in both tensors, so one plan
            // serves both. Context extension applies here too: V3 length-extended
            // checkpoints carry YaRN.
            RopePlan plan;
            plan.init(c, c.rope_dim);
            if (c.rope_freqs) plan.freq_factors = weights("blk.0.rope_freqs.weight");
            // Text prompts put the token at the same position on every mRoPE
            // axis, so one value fills the array here as it does above.
            const std::array<double, 4> positions = {static_cast<double>(position),
                                                     static_cast<double>(position),
                                                     static_cast<double>(position),
                                                     static_cast<double>(position)};
            auto rope_part = [&](std::vector<float> &vec, uint32_t heads, uint32_t stride,
                                 uint32_t offset, const std::string &name) {
                // Rotary positions, applied to the rope slice only: the trailing
                // `rope_dim` of each query head, and the whole shared k_pe.
                const uint32_t width = c.rope_dim;
                for (uint32_t head = 0; head < heads; ++head) {
                    const size_t base = static_cast<size_t>(head) * stride + offset;
                    for (uint32_t i = 0; i < rope_pairs(width); ++i) {
                        double cs = 0.0;
                        double sn = 0.0;
                        plan.rotation(i, positions, cs, sn);
                        const float a0 = vec[base + 2 * i];
                        const float a1 = vec[base + 2 * i + 1];
                        vec[base + 2 * i] = static_cast<float>(a0 * cs - a1 * sn);
                        vec[base + 2 * i + 1] = static_cast<float>(a0 * sn + a1 * cs);
                    }
                }
                backend_->record_control(name, vec.size() * sizeof(float),
                                         vec.size() * sizeof(float));
            };
            rope_part(qflat, c.heads, c.head_dim, c.head_dim - c.rope_dim,
                      "mla_rope_q.L" + std::to_string(layer));
            rope_part(kpe, 1, c.rope_dim, 0, "mla_rope_k.L" + std::to_string(layer));
            ckv = rmsnorm(ckv, norm_weights(prefix + "attn_kv_a_norm.weight"),
                          "attn_kv_a_norm.L" + std::to_string(layer));

            const uint32_t nope = c.head_dim - c.rope_dim;
            const uint32_t head_v = c.head_v_dim();
            k.assign(static_cast<size_t>(c.heads) * c.head_dim, 0.0f);
            v.assign(static_cast<size_t>(c.heads) * head_v, 0.0f);
            const float *kb = weights(prefix + "attn_k_b.weight");
            const float *vb = weights(prefix + "attn_v_b.weight");
            for (uint32_t head = 0; head < c.heads; ++head) {
                float *k_head = k.data() + static_cast<size_t>(head) * c.head_dim;
                // k_b is [nope, kv_lora_rank, heads], ggml's ne0-fastest order,
                // so element (o, i, head) sits at o + nope*(i + latent*head): the
                // latent index is the *slow* axis of each row, not the fast one.
                const size_t kb_head = static_cast<size_t>(head) * nope * latent;
                for (uint32_t o = 0; o < nope; ++o) {
                    double acc = 0.0;
                    for (uint32_t i = 0; i < latent; ++i)
                        acc += kb[kb_head + static_cast<size_t>(i) * nope + o] * ckv[i];
                    k_head[o] = static_cast<float>(acc);
                }
                for (uint32_t i = 0; i < c.rope_dim; ++i) k_head[nope + i] = kpe[i];
                // v_b is [kv_lora_rank, head_v, heads].
                float *v_head = v.data() + static_cast<size_t>(head) * head_v;
                for (uint32_t o = 0; o < head_v; ++o) {
                    const float *row = vb + (static_cast<size_t>(head) * head_v + o) * latent;
                    double acc = 0.0;
                    for (uint32_t i = 0; i < latent; ++i) acc += row[i] * ckv[i];
                    v_head[o] = static_cast<float>(acc);
                }
            }
            q = qflat;
            backend_->record_control(
                "mla_expand.L" + std::to_string(layer),
                static_cast<uint64_t>(c.heads) * (nope + head_v) * latent * sizeof(float),
                (k.size() + v.size()) * sizeof(float));
        } else if (c.fused_qkv) {
            // phi3: one projection produces q, k and v back to back.
            std::vector<float> qkv =
                matvec_view(tensor_view(prefix + "attn_qkv.weight"), *attn_input, c.hidden + 2 * kv_dim,
                            c.hidden, "attn_qkv.L" + std::to_string(layer));
            if (c.attn_qkv_bias) {
                const float *b = bias_weights(prefix + "attn_qkv.bias");
                for (uint32_t i = 0; i < c.hidden + 2 * kv_dim; ++i) qkv[i] += b[i];
                backend_->record_control("attn_qkv_bias.L" + std::to_string(layer),
                                         qkv.size() * sizeof(float), 0);
            }
            q.assign(qkv.begin(), qkv.begin() + c.hidden);
            k.assign(qkv.begin() + c.hidden, qkv.begin() + c.hidden + kv_dim);
            v.assign(qkv.begin() + c.hidden + kv_dim, qkv.end());
        } else {
            q = matvec_view(tensor_view(prefix + "attn_q.weight"), *attn_input, c.hidden, c.hidden,
                            "attn_q.L" + std::to_string(layer));
            k = matvec_view(tensor_view(prefix + "attn_k.weight"), *attn_input, kv_dim, c.hidden,
                            "attn_k.L" + std::to_string(layer));
            v = matvec_view(tensor_view(prefix + "attn_v.weight"), *attn_input, kv_dim, c.hidden,
                            "attn_v.L" + std::to_string(layer));
            // BitNet's ternary projections carry a one-element scale that
            // multiplies the projection output. llama.cpp folds it into
            // `build_lora_mm`, so it lands before the bias and before RoPE.
            if (c.projection_scales) {
                const std::string lp = "L" + std::to_string(layer);
                apply_projection_scale(q, prefix + "attn_q.scale", "attn_q_scale." + lp);
                apply_projection_scale(k, prefix + "attn_k.scale", "attn_k_scale." + lp);
                apply_projection_scale(v, prefix + "attn_v.scale", "attn_v_scale." + lp);
            }
        }
        if (c.attn_qkv_bias && !c.fused_qkv) {
            // qwen2-style: bias is added to the projection output, before RoPE.
            const std::string lp = "blk." + std::to_string(layer) + ".";
            const float *bq = bias_weights(lp + "attn_q.bias");
            const float *bk = bias_weights(lp + "attn_k.bias");
            const float *bv = bias_weights(lp + "attn_v.bias");
            for (uint32_t i = 0; i < c.hidden; ++i) q[i] += bq[i];
            for (uint32_t i = 0; i < kv_dim; ++i) {
                k[i] += bk[i];
                v[i] += bv[i];
            }
            backend_->record_control("attn_bias.L" + std::to_string(layer),
                                     (c.hidden + 2 * kv_dim) * sizeof(float), 0);
        }
        // Per-head q/k normalisation. Which layout, which norm and where it sits
        // relative to the rotation are three separate facts (see
        // `ModelConfig::qk_norm_matrix`): qwen3 is a shared RMSNorm scale applied
        // before the rope, command-r and chameleon are one LayerNorm row per head
        // (optionally biased), and talkie normalises after the rope.
        const std::string qk_prefix = "blk." + std::to_string(layer) + ".";
        auto head_norm = [&](std::vector<float> &vec, uint32_t heads, const float *scale,
                             const std::string &name) {
            for (uint32_t h = 0; h < heads; ++h) {
                const size_t base = static_cast<size_t>(h) * c.head_dim;
                // A matrix-shaped weight carries one row per head; a shared one
                // is a single row every head reuses.
                const float *h_scale = c.qk_norm_matrix
                                           ? scale + static_cast<size_t>(h) * c.head_dim
                                           : scale;
                double mean = 0.0;
                if (c.qk_norm_layernorm) {
                    for (uint32_t j = 0; j < c.head_dim; ++j) mean += vec[base + j];
                    mean /= c.head_dim;
                }
                double ss = 0.0;
                for (uint32_t j = 0; j < c.head_dim; ++j) {
                    const double d = static_cast<double>(vec[base + j]) - mean;
                    ss += d * d;
                }
                const double s = 1.0 / std::sqrt(ss / c.head_dim + c.qk_norm_eps);
                for (uint32_t j = 0; j < c.head_dim; ++j)
                    vec[base + j] = static_cast<float>((static_cast<double>(vec[base + j]) - mean) *
                                                       s * h_scale[j]);
            }
            backend_->record_control(name, vec.size() * sizeof(float) * 2,
                                     vec.size() * sizeof(float));
        };
        // olmoe normalises the concatenated heads as one vector, with a scale as
        // wide as the whole projection. It is not a per-head norm: the RMS is
        // taken over every element, so one head's magnitude affects another's.
        auto whole_vector_norm = [&](std::vector<float> &vec, const std::string &prefix,
                                     const std::string &name) {
            const float *w = norm_weights(qk_prefix + prefix + ".weight");
            double ss = 0.0;
            for (float v : vec) ss += static_cast<double>(v) * v;
            const double s = 1.0 / std::sqrt(ss / static_cast<double>(vec.size()) + c.qk_norm_eps);
            for (size_t i = 0; i < vec.size(); ++i)
                vec[i] = static_cast<float>(vec[i] * s * w[i]);
            backend_->record_control(name, vec.size() * sizeof(float) * 2,
                                     vec.size() * sizeof(float));
        };
        auto apply_qk_norm = [&]() {
            if (c.qk_norm_whole_vector) {
                whole_vector_norm(q, "attn_q_norm", "qk_norm.q.L" + std::to_string(layer));
                whole_vector_norm(k, "attn_k_norm", "qk_norm.k.L" + std::to_string(layer));
                return;
            }
            head_norm(q, c.heads, norm_weights(qk_prefix + "attn_q_norm.weight"),
                      "qk_norm.q.L" + std::to_string(layer));
            head_norm(k, c.kv_heads, norm_weights(qk_prefix + "attn_k_norm.weight"),
                      "qk_norm.k.L" + std::to_string(layer));
        };
        if (c.qk_norm && !c.qk_norm_after_rope) apply_qk_norm();
        // NoPE layers (SmolLM3) skip the rotation entirely: their keys stay as the
        // projection produced them, and the position information reaches the model
        // through the layers that do rotate.
        // jamba's attention layers carry no rotary position at all -- upstream's
        // block simply never calls the rope -- so a file that states none is not
        // rotated at head width by default.
        const bool rotate = !c.rope_disabled && c.layer_has_rope(layer);
        if (!c.mla && rotate)
            rope_half(q, c.heads, k, c.kv_heads, position, c.layer_rope_theta(layer));
        // talkie's block normalises q and k *after* the rotation.
        if (c.qk_norm && c.qk_norm_after_rope && !c.mla) apply_qk_norm();
        // llama.cpp's per-position attention temperature (mistral3, llama4-style
        // checkpoints): every query component is scaled by
        //   log(floor(pos / floor) + 1) * temperature_scale + 1
        // *after* the rotation, so it is a multiplier on the logits of the whole
        // head rather than on a part of the score.
        if (c.attn_temp_scale != 0.0) {
            const double temperature =
                std::log(std::floor(static_cast<double>(position) / c.attn_temp_floor) + 1.0) *
                    c.attn_temp_scale +
                1.0;
            for (float &v : q) v = static_cast<float>(v * temperature);
            backend_->record_control("attn_temperature.L" + std::to_string(layer),
                                     q.size() * sizeof(float), q.size() * sizeof(float));
        }

        // Only attention layers own a slot in the per-position cache, so a hybrid
        // indexes it by the layer's position among them.
        // A windowed layer keeps a ring of `sliding_window` positions instead of
        // the whole context, so the address is the position modulo the capacity
        // -- the same number the budget reserves. The live set at any token is
        // [first_visible, position], which is at most one window wide, so no two
        // live positions can collide.
        auto cache_index = [&](uint32_t l, uint64_t pos) -> size_t {
            return static_cast<size_t>(c.cache_index(l, pos, c.context) * sizeof(float) / sizeof(float));
        };
        (void)cache_index;
        const size_t current_base =
            static_cast<size_t>(c.cache_index(layer, position, c.context));
        std::memcpy(kv_ + current_base, k.data(), static_cast<size_t>(kv_dim) * sizeof(float));
        std::memcpy(kv_ + current_base + kv_dim, v.data(), static_cast<size_t>(v_dim) * sizeof(float));

        std::vector<float> attention(attn_out_dim, 0.0f);
        double scale = c.attention_scale > 0.0
                           ? c.attention_scale
                           : 1.0 / std::sqrt(static_cast<double>(c.head_dim));
        for (uint32_t head = 0; head < c.heads; ++head) {
            const size_t hb = static_cast<size_t>(head) * c.head_dim;
            const size_t kvb = static_cast<size_t>(head / group) * c.head_dim;
            // gemma2/3 restrict most layers to a sliding window, and gpt-oss
            // alternates windowed and full layers, so the decision is per layer.
            // The first position that is visible is max(0, position + 1 - window).
            const bool windowed = c.layer_uses_window(layer);
            const uint64_t first_visible =
                !windowed || position < c.sliding_window
                    ? 0
                    : position + 1 - c.sliding_window;
            std::vector<double> scores(position + 1);
            for (uint64_t t = first_visible; t <= position; ++t) {
                const size_t base = static_cast<size_t>(c.cache_index(layer, t, c.context));
                const float *kt = kv_ + base + kvb;
                double dot = 0.0;
                for (uint32_t j = 0; j < c.head_dim; ++j)
                    dot += static_cast<double>(q[hb + j]) * kt[j];
                double value = dot * scale;
                // ALiBi: the reference puts the bias in the mask tensor as
                // -(query - key) and lets `soft_max_ext` scale it by the head's
                // slope, so it is part of the score the softmax sees.
                if (!alibi_slopes.empty())
                    value += alibi_slopes[head] * -static_cast<double>(position - t);
                scores[t] = value;
            }
            // Masked positions are excluded before the softmax rather than
            // given zero weight afterwards, so the normalisation stays exact.
            std::vector<float> score_floats;
            score_floats.reserve(position + 1 - first_visible);
            for (uint64_t t = first_visible; t <= position; ++t) {
                double value = scores[t];
                if (c.attn_logit_softcap > 0.0)
                    value = c.attn_logit_softcap *
                            std::tanh(value / c.attn_logit_softcap);
                score_floats.push_back(static_cast<float>(value));
            }
            // Attention sink: the reference appends one extra learnable logit per
            // head to the softmax and then discards its probability, so the
            // denominator includes exp(sink) and the head can attend to nothing.
            // Appending it to the vector and dropping the last entry is exactly
            // that -- no separate normalisation path to drift.
            if (c.attn_sinks) {
                const float *sinks = norm_weights(prefix + "attn_sinks.weight");
                score_floats.push_back(sinks[head]);
            }
            std::vector<float> probs = softmax(score_floats);
            std::vector<double> acc(c.head_v_dim(), 0.0);
            const size_t vb = static_cast<size_t>(head / group) * c.head_v_dim();
            for (uint64_t t = first_visible; t <= position; ++t) {
                const size_t base = static_cast<size_t>(c.cache_index(layer, t, c.context));
                const float *vt = kv_ + base + kv_dim + vb;
                double p = probs[t - first_visible];
                for (uint32_t j = 0; j < c.head_v_dim(); ++j) acc[j] += p * vt[j];
            }
            const size_t ab = static_cast<size_t>(head) * c.head_v_dim();
            for (uint32_t j = 0; j < c.head_v_dim(); ++j)
                attention[ab + j] = static_cast<float>(acc[j]);
        }
        backend_->record_control("attention.L" + std::to_string(layer),
                                 (position + 1) * kv_stride * sizeof(float),
                                 attention.size() * sizeof(float));
        // BitNet's SUBLN: the attention output is normalised *inside* the block,
        // after the head mix and before the output projection. It cannot be
        // expressed as a post-norm, which would run after the projection.
        if (c.sub_norms)
            attention = rmsnorm(attention, norm_weights(prefix + "attn_sub_norm.weight"),
                                "rmsnorm.attn_sub.L" + std::to_string(layer));
        std::vector<float> attn_out =
            matvec_view(tensor_view(prefix + "attn_output.weight"), attention, c.hidden,
                        attn_out_dim, "attn_output.L" + std::to_string(layer));
        if (c.projection_scales)
            apply_projection_scale(attn_out, prefix + "attn_output.scale",
                                   "attn_output_scale.L" + std::to_string(layer));
        if (c.attn_output_bias) {
            const float *b = bias_weights(prefix + "attn_output.bias");
            for (uint32_t i = 0; i < c.hidden; ++i) attn_out[i] += b[i];
            backend_->record_control("attn_output_bias.L" + std::to_string(layer),
                                     attn_out.size() * sizeof(float), 0);
        }
        if (c.post_norms)
            attn_out = rmsnorm(attn_out, norm_weights(prefix + "attn_post_norm.weight"),
                               "rmsnorm.attn_post.L" + std::to_string(layer));
        if (c.residual_scale_applied) {
            // granite multiplies each branch output by `*.residual_scale` before
            // the add (models/granite.cpp `build_layer_ffn`), which changes the
            // residual stream itself rather than one projection.
            const float s = static_cast<float>(c.residual_scale);
            for (float &v : attn_out) v *= s;
            backend_->record_control("residual_scale.attn.L" + std::to_string(layer),
                                     attn_out.size() * sizeof(float), attn_out.size() * sizeof(float));
        }
        residual_add(x, attn_out, "residual.attn.L" + std::to_string(layer));

        if (c.parallel_residual) {
            h = attn_normed;
        } else {
            // gpt-oss normalises after the attention residual and calls that
            // tensor attn_post_norm; the block arithmetic is unchanged.
            h = c.no_branch_norms
                    ? x
                    : norm(x, prefix + (c.ffn_norm_is_post ? "attn_post_norm" : "ffn_norm"),
                           "ffn_norm.L" + std::to_string(layer));
        }
        if (c.ffn_norm_ada) {
            // mistral3's adaptive RMSNorm: the normalised input is scaled by
            // 1 + mlp(t_cond), where t_cond is the sinusoidal embedding of the
            // configured output delay (constant for the whole session) and the
            // mlp is down -> gelu_erf -> up.
            const TensorInfo *down_info = manifest_.tensor(prefix + "ffn_norm_ada_down.weight");
            const uint32_t bottleneck =
                down_info && down_info->dims.size() == 2 ? down_info->dims[1] : 0;
            // cos of the first half, sin of the second, at geometrically spaced
            // frequencies -- upstream's `input_t_cond`, which is a session
            // constant rather than a per-token value.
            const uint32_t half = c.hidden / 2;
            std::vector<float> t_cond(c.hidden, 0.0f);
            for (uint32_t i = 0; i < half; ++i) {
                const double freq = std::exp(-std::log(10000.0) * static_cast<double>(i) / half) *
                                    c.realtime_delay;
                t_cond[i] = static_cast<float>(std::cos(freq));
                t_cond[half + i] = static_cast<float>(std::sin(freq));
            }
            std::vector<float> ada =
                matvec_view(tensor_view(prefix + "ffn_norm_ada_down.weight"), t_cond, bottleneck,
                            c.hidden, "ffn_norm_ada_down.L" + std::to_string(layer));
            // gelu_erf, the activation llama.cpp builds this MLP with
            // (`build_ffn(..., LLM_FFN_GELU_ERF, LLM_FFN_SEQ, il)`).
            for (float &v : ada) {
                const double x = v;
                v = static_cast<float>(0.5 * x * (1.0 + std::erf(x / std::sqrt(2.0))));
            }
            backend_->record_control("gelu_erf.L" + std::to_string(layer),
                                     ada.size() * sizeof(float), ada.size() * sizeof(float));
            std::vector<float> scale =
                matvec_view(tensor_view(prefix + "ffn_norm_ada_up.weight"), ada, c.hidden,
                            bottleneck, "ffn_norm_ada_up.L" + std::to_string(layer));
            for (uint32_t j = 0; j < c.hidden; ++j)
                h[j] = static_cast<float>(h[j] * (1.0f + scale[j]));
            backend_->record_control("ffn_norm_ada.L" + std::to_string(layer),
                                     h.size() * sizeof(float), h.size() * sizeof(float));
        }
        }  // end of the attention branch
        if (!c.layer_has_ffn(layer)) continue;
        if (c.layer_is_recurrent(layer)) {
            // A hybrid block: the recurrent layer has already taken its residual
            // and the feed-forward reads the post-residual norm, exactly as in
            // the attention branch.
            h = norm(x, prefix + "ffn_norm", "ffn_norm.L" + std::to_string(layer));
        }
        std::vector<float> ffn(c.hidden, 0.0f);
        // The leading blocks of an MoE model can still carry a plain FFN, and a
        // hybrid can place its experts on one kind of layer only (jamba mixes
        // dense and MoE blocks), so the router tensors decide per layer.
        const bool moe_layer = c.layer_is_moe(layer);
        if (!moe_layer && c.ffn_no_gate) {
            // phi2: one up projection, GELU, one down projection, both biased.
            std::vector<float> act = matvec_view(tensor_view(prefix + "ffn_up.weight"), h,
                                                 c.intermediate, c.hidden,
                                                 "ffn_up.L" + std::to_string(layer));
            if (c.ffn_bias) {
                const float *b = bias_weights(prefix + "ffn_up.bias");
                for (uint32_t i = 0; i < c.intermediate; ++i) act[i] += b[i];
                backend_->record_control("ffn_up_bias.L" + std::to_string(layer),
                                         act.size() * sizeof(float), 0);
            }
            for (float &v : act) v = activate(v);
            backend_->record_control(c.activation + ".L" + std::to_string(layer),
                                     act.size() * sizeof(float), act.size() * sizeof(float));
            ffn = matvec_view(tensor_view(prefix + "ffn_down.weight"), act, c.hidden,
                              c.intermediate, "ffn_down.L" + std::to_string(layer));
            if (c.ffn_bias) {
                const float *b = bias_weights(prefix + "ffn_down.bias");
                for (uint32_t i = 0; i < c.hidden; ++i) ffn[i] += b[i];
                backend_->record_control("ffn_down_bias.L" + std::to_string(layer),
                                         ffn.size() * sizeof(float), 0);
            }
        } else if (!moe_layer) {
            std::vector<float> gate, up;
            if (c.fused_gate_up) {
                // glm4: one projection emits both halves, gate first then up
                // (the convention ggml_swiglu splits on).
                std::vector<float> both =
                    matvec_view(tensor_view(prefix + "ffn_up.weight"), h, 2 * c.intermediate,
                                c.hidden, "ffn_gate_up.L" + std::to_string(layer));
                gate.assign(both.begin(), both.begin() + c.intermediate);
                up.assign(both.begin() + c.intermediate, both.end());
            } else {
                gate = matvec_view(tensor_view(prefix + "ffn_gate.weight"), h, c.intermediate,
                                   c.hidden, "ffn_gate.L" + std::to_string(layer));
                up = matvec_view(tensor_view(prefix + "ffn_up.weight"), h, c.intermediate,
                                 c.hidden, "ffn_up.L" + std::to_string(layer));
            }
            if (c.ffn_gate_bias) {
                // mistral3 can carry a bias on the gate projection alone; it is
                // added before the activation, like every other FFN bias.
                const float *b = bias_weights(prefix + "ffn_gate.bias");
                for (uint32_t i = 0; i < c.intermediate; ++i) gate[i] += b[i];
                backend_->record_control("ffn_gate_bias.L" + std::to_string(layer),
                                         gate.size() * sizeof(float), 0);
            }
            if (c.ffn_bias) {
                const float *b = bias_weights(prefix + "ffn_up.bias");
                for (uint32_t i = 0; i < c.intermediate; ++i) up[i] += b[i];
                backend_->record_control("ffn_up_bias.L" + std::to_string(layer),
                                         up.size() * sizeof(float), 0);
            }
            if (c.projection_scales) {
                // Applied to the projection outputs, before the gate product
                // (`build_ffn` multiplies `up`/`gate` by their scales first).
                const std::string lp = "L" + std::to_string(layer);
                apply_projection_scale(gate, prefix + "ffn_gate.scale", "ffn_gate_scale." + lp);
                apply_projection_scale(up, prefix + "ffn_up.scale", "ffn_up_scale." + lp);
            }
            std::vector<float> act(c.intermediate);
            for (uint32_t i = 0; i < c.intermediate; ++i) act[i] = gated(gate[i], up[i], layer);
            backend_->record_control("swiglu.L" + std::to_string(layer),
                                     (gate.size() + up.size()) * sizeof(float),
                                     act.size() * sizeof(float));
            // BitNet's second SUBLN sits on the gated FFN output, before the
            // down projection.
            if (c.sub_norms)
                act = rmsnorm(act, norm_weights(prefix + "ffn_sub_norm.weight"),
                              "rmsnorm.ffn_sub.L" + std::to_string(layer));
            ffn = matvec_view(tensor_view(prefix + "ffn_down.weight"), act, c.hidden,
                              c.intermediate, "ffn_down.L" + std::to_string(layer));
            if (c.ffn_bias) {
                const float *b = bias_weights(prefix + "ffn_down.bias");
                for (uint32_t i = 0; i < c.hidden; ++i) ffn[i] += b[i];
                backend_->record_control("ffn_down_bias.L" + std::to_string(layer),
                                         ffn.size() * sizeof(float), 0);
            }
            if (c.projection_scales)
                apply_projection_scale(ffn, prefix + "ffn_down.scale",
                                       "ffn_down_scale.L" + std::to_string(layer));
        } else {
            std::vector<float> scores =
                matvec_view(tensor_view(prefix + "ffn_router.weight"), h, c.expert_count, c.hidden,
                            "router.L" + std::to_string(layer));
            // gpt-oss biases the router *logits* before any gating function
            // runs (llama.cpp's gate_inp_b). This is a different tensor from
            // `ffn_exp_probs_b`, which is added after gating and only steers
            // selection, so applying one in place of the other would route a
            // real file wrong.
            if (c.expert_router_bias) {
                const float *b = bias_weights(prefix + "ffn_gate_inp.bias");
                for (size_t i = 0; i < scores.size(); ++i) scores[i] += b[i];
                backend_->record_control("router_logit_bias.L" + std::to_string(layer),
                                         scores.size() * sizeof(float), 0);
            }
            // Which function turns logits into probabilities is a per-family
            // fact: V2 softmaxes, V3 and later use a sigmoid. The weights that
            // come out are always the *unbiased* probabilities; a selection
            // bias only changes which experts are picked.
            std::vector<float> probs(scores.size());
            if (c.expert_gating == "sigmoid") {
                for (size_t i = 0; i < scores.size(); ++i)
                    probs[i] = static_cast<float>(1.0 / (1.0 + std::exp(-static_cast<double>(scores[i]))));
            } else if (c.expert_gating == "sqrt-softplus") {
                for (size_t i = 0; i < scores.size(); ++i)
                    probs[i] = static_cast<float>(
                        std::sqrt(std::log1p(std::exp(static_cast<double>(scores[i])))));
            } else if (c.expert_gating == "softmax-weight") {
                // Weights are softmaxed over the selected experts below, so the
                // probabilities here are the raw logits.
                probs = scores;
            } else {
                probs = softmax(scores);
            }
            backend_->record_control("router_gating.L" + std::to_string(layer),
                                     scores.size() * sizeof(float), probs.size() * sizeof(float));

            std::vector<float> selection(probs);
            if (c.expert_selection_bias) {
                const float *b = bias_weights(prefix + "ffn_exp_probs_b.bias");
                for (size_t i = 0; i < selection.size(); ++i) selection[i] += b[i];
                backend_->record_control("router_bias.L" + std::to_string(layer),
                                         selection.size() * sizeof(float), 0);
            }
            if (c.expert_groups > 1) {
                // Group-limited routing (deepseek V3): keep the groups whose two
                // best experts score highest in total, and mask the rest out of
                // the selection.
                const uint32_t per_group = c.expert_count / c.expert_groups;
                std::vector<std::pair<float, uint32_t>> group_scores;
                group_scores.reserve(c.expert_groups);
                for (uint32_t g = 0; g < c.expert_groups; ++g) {
                    std::vector<float> best;
                    for (uint32_t e = 0; e < per_group; ++e)
                        best.push_back(selection[g * per_group + e]);
                    std::sort(best.begin(), best.end(), std::greater<float>());
                    float sum = 0.0f;
                    for (uint32_t k = 0; k < std::min<uint32_t>(2, per_group); ++k) sum += best[k];
                    group_scores.emplace_back(sum, g);
                }
                std::stable_sort(group_scores.begin(), group_scores.end(),
                                 [](const auto &a, const auto &b) {
                                     if (a.first != b.first) return a.first > b.first;
                                     return a.second < b.second;
                                 });
                std::vector<bool> keep(c.expert_groups, false);
                for (uint32_t i = 0; i < std::min(c.expert_groups_used, c.expert_groups); ++i)
                    keep[group_scores[i].second] = true;
                for (uint32_t g = 0; g < c.expert_groups; ++g) {
                    if (keep[g]) continue;
                    for (uint32_t e = 0; e < per_group; ++e)
                        selection[g * per_group + e] = -std::numeric_limits<float>::infinity();
                }
                backend_->record_control("router_groups.L" + std::to_string(layer),
                                         selection.size() * sizeof(float),
                                         c.expert_groups * sizeof(uint32_t));
            }
            std::vector<uint32_t> ids = topk(selection, c.expert_top_k);
            std::vector<float> selected_weights(c.expert_top_k);
            float total = 0.0f;
            for (uint32_t i = 0; i < c.expert_top_k; ++i) {
                selected_weights[i] = probs[ids[i]];
                total += selected_weights[i];
            }
            if (c.expert_gating == "softmax-weight")
                selected_weights = softmax(selected_weights);
            if (c.expert_weights_norm) {
                // The reference clamps the denominator instead of guarding with a
                // comparison, and deepseek4.1 adds 1e-20 rather than clamping to
                // the smallest F16 normal. Both are recorded in the config, so
                // the arithmetic matches rather than merely avoiding a divide by
                // zero.
                const double denominator =
                    c.expert_weights_epsilon_additive
                        ? static_cast<double>(total) + c.expert_weights_epsilon
                        : std::max(static_cast<double>(total), c.expert_weights_epsilon);
                for (float &w : selected_weights)
                    w = static_cast<float>(w / denominator);
            }
            for (float &w : selected_weights)
                w = static_cast<float>(w * c.expert_weights_scale);
            backend_->record_control("router_topk.L" + std::to_string(layer),
                                     scores.size() * sizeof(float),
                                     ids.size() * sizeof(uint32_t));
            residency_->note_selected(layer, ids, position);
            if (routing) routing->push_back({position, layer, ids});
            for (uint32_t expert : ids)
                residency_->issue_prefetch(layer, expert, position + 1);

            const uint32_t ewidth = c.layer_expert_width(layer);
            for (uint32_t i = 0; i < ids.size(); ++i) {
                float weight = selected_weights[i];
                uint32_t expert = ids[i];
                // Expert payloads keep their stored block type; each projection
                // is a contiguous slice inside the expert record.
                // The manifest's record is the only description of an expert's
                // bytes: it carries one extent per projection with the projection's
                // own offset, length and stored type. That is what lets a stacked
                // file (one tensor, an expert axis, possibly mixed quantisation)
                // and a per-expert file (one tensor per projection) be read by the
                // same code — and why no tensor name is looked up here.
                const ExpertRecord *record = manifest_.expert(layer, expert);
                if (!record)
                    fail(ErrorCode::MISSING_REQUIRED_TENSOR,
                         "no expert record for layer " + std::to_string(layer) + " expert " +
                             std::to_string(expert));
                const uint64_t gate_bytes = record->extents[0].bytes;
                const uint64_t up_bytes = record->extents[1].bytes;
                const uint32_t gate_type = record->extents[0].type_id;
                const uint32_t up_type = record->extents[1].type_id;
                const uint32_t down_type = record->extents[2].type_id;
                residency_->with_expert(layer, expert, position, [&](const float *expert_data) {
                    const uint8_t *base = reinterpret_cast<const uint8_t *>(expert_data);
                    TensorView gate_view{base, gate_type};
                    TensorView up_view{base + gate_bytes, up_type};
                    TensorView down_view{base + gate_bytes + up_bytes, down_type};
                    std::vector<float> gate = matvec_view(
                        gate_view, h, ewidth, c.hidden,
                        "expert_gate.L" + std::to_string(layer) + ".E" + std::to_string(expert));
                    std::vector<float> up = matvec_view(
                        up_view, h, ewidth, c.hidden,
                        "expert_up.L" + std::to_string(layer) + ".E" + std::to_string(expert));
                    if (c.expert_projection_bias) {
                        // One bias column per expert, added after the matmul and
                        // before the activation, matching llama.cpp's
                        // `ggml_add_id(gate_exps_b, selected_experts)`.
                        const float *gb = bias_weights(prefix + "ffn_gate_exps.bias");
                        const float *ub = bias_weights(prefix + "ffn_up_exps.bias");
                        for (uint32_t j = 0; j < ewidth; ++j) {
                            gate[j] += gb[static_cast<size_t>(expert) * ewidth + j];
                            up[j] += ub[static_cast<size_t>(expert) * ewidth + j];
                        }
                        backend_->record_control("expert_proj_bias.L" + std::to_string(layer),
                                                 (ewidth * 2) * sizeof(float), 0);
                    }
                    std::vector<float> act(ewidth);
                    for (uint32_t j = 0; j < ewidth; ++j) act[j] = gated(gate[j], up[j], layer);
                    backend_->record_control("expert_swiglu.L" + std::to_string(layer),
                                             (gate.size() + up.size()) * sizeof(float),
                                             act.size() * sizeof(float));
                    std::vector<float> down = matvec_view(
                        down_view, act, c.hidden, ewidth,
                        "expert_down.L" + std::to_string(layer) + ".E" + std::to_string(expert));
                    if (c.expert_projection_bias) {
                        const float *db = bias_weights(prefix + "ffn_down_exps.bias");
                        for (uint32_t j = 0; j < c.hidden; ++j)
                            down[j] += db[static_cast<size_t>(expert) * c.hidden + j];
                    }
                    for (uint32_t j = 0; j < c.hidden; ++j) ffn[j] += weight * down[j];
                });
            }
            residency_->assert_selected_executed();
            backend_->record_control("expert_combine.L" + std::to_string(layer),
                                     ids.size() * c.hidden * sizeof(float),
                                     c.hidden * sizeof(float));

            if (c.shared_expert) {
                // qwen2moe runs a dense FFN beside the routed experts and gates
                // the whole branch with one scalar per token. The reference
                // computes that scalar as silu(s)/s, which is sigmoid(s); the
                // division is NaN at exactly s == 0, so the limit 0.5 is used
                // there instead. Every other value is identical.
                const uint32_t swidth = c.shared_expert_intermediate;
                float gate_scalar = 1.0f;
                if (c.shared_expert_gate) {
                    std::vector<float> s_in = matvec_view(
                        tensor_view(prefix + "ffn_gate_inp_shexp.weight"), h, 1, c.hidden,
                        "shexp_gate_inp.L" + std::to_string(layer));
                    const float s = s_in[0];
                    gate_scalar = s != 0.0f ? activate(s) / s : 0.5f;
                }
                std::vector<float> s_gate = matvec_view(
                    tensor_view(prefix + "ffn_gate_shexp.weight"), h, swidth, c.hidden,
                    "shexp_gate.L" + std::to_string(layer));
                std::vector<float> s_up = matvec_view(
                    tensor_view(prefix + "ffn_up_shexp.weight"), h, swidth, c.hidden,
                    "shexp_up.L" + std::to_string(layer));
                std::vector<float> s_act(swidth);
                for (uint32_t j = 0; j < swidth; ++j) s_act[j] = gated(s_gate[j], s_up[j], layer);
                backend_->record_control("shexp_swiglu.L" + std::to_string(layer),
                                         (s_gate.size() + s_up.size()) * sizeof(float),
                                         s_act.size() * sizeof(float));
                std::vector<float> s_down = matvec_view(
                    tensor_view(prefix + "ffn_down_shexp.weight"), s_act, c.hidden, swidth,
                    "shexp_down.L" + std::to_string(layer));
                for (uint32_t j = 0; j < c.hidden; ++j)
                    ffn[j] += gate_scalar * s_down[j];
                backend_->record_control("shexp_combine.L" + std::to_string(layer),
                                         c.hidden * sizeof(float), c.hidden * sizeof(float));
            }
        }
        if (c.post_norms)
            ffn = rmsnorm(ffn, norm_weights(prefix + "ffn_post_norm.weight"),
                          "rmsnorm.ffn_post.L" + std::to_string(layer));
        if (c.residual_scale_applied) {
            const float s = static_cast<float>(c.residual_scale);
            for (float &v : ffn) v *= s;
            backend_->record_control("residual_scale.ffn.L" + std::to_string(layer),
                                     ffn.size() * sizeof(float), ffn.size() * sizeof(float));
        }
        residual_add(x, ffn, "residual.ffn.L" + std::to_string(layer));
    }

    std::vector<float> final_norm = norm(x, "output_norm", "output_norm.final");
    std::vector<float> logits = matvec_view(tensor_view("output.weight"), final_norm, c.vocab,
                                            c.hidden, "output_head");
    if (c.logit_bias) {
        const float *b = bias_weights("output.bias");
        for (uint32_t i = 0; i < c.vocab; ++i) logits[i] += b[i];
        backend_->record_control("output_bias", logits.size() * sizeof(float), 0);
    }
    if (c.logit_scale_inverse) {
        // granite: `ggml_scale(ctx0, cur, 1.0f / hparams.f_logit_scale)` after the
        // head. Dividing is not multiplying, so it cannot share command-r's flag.
        const float s = static_cast<float>(1.0 / c.logit_scale);
        for (float &v : logits) v *= s;
        backend_->record_control("logit_scale_inverse", logits.size() * sizeof(float),
                                 logits.size() * sizeof(float));
    }
    if (c.logit_scale_applied) {
        // command-r multiplies the output logits by `*.logit_scale`; granite
        // records the same key and never applies it.
        const float s = static_cast<float>(c.logit_scale);
        for (float &v : logits) v *= s;
        backend_->record_control("logit_scale", logits.size() * sizeof(float),
                                 logits.size() * sizeof(float));
    }
    if (c.final_logit_softcap > 0.0) {
        for (float &v : logits)
            v = static_cast<float>(c.final_logit_softcap *
                                   std::tanh(static_cast<double>(v) / c.final_logit_softcap));
        backend_->record_control("logit_softcap", logits.size() * sizeof(float),
                                 logits.size() * sizeof(float));
    }
    return logits;
}

bool Engine::supports_batched_prefill() const {
    const ModelConfig &c = manifest_.config;
    // Every rejection below is a structural difference in the block, not a
    // performance preference: MLA keeps a latent instead of keys and values,
    // a recurrent layer carries a rolling state instead of a cache, routed
    // experts choose per token, BitNet normalises inside the attention branch,
    // phi2 has no ffn_norm to share, gpt-oss feeds the FFN from a post-norm and
    // mistral3 scales its norm with an MLP. Batching those would mean writing a
    // second implementation of each; they keep the sequential path.
    if (c.mla) return false;
    if (!c.ssm_kind.empty()) return false;
    if (c.expert_count > 0) return false;
    if (c.sub_norms) return false;
    if (c.parallel_residual) return false;
    if (c.ffn_norm_is_post) return false;
    if (c.ffn_norm_ada) return false;
    if (c.no_branch_norms) return false;
    // command-r and talkie normalize q/k with their own layout, norm type and
    // position; the batch path only knows qwen3's shared pre-rope RMSNorm.
    if (c.qk_norm_matrix || c.qk_norm_layernorm || c.qk_norm_after_rope)
        return false;
    if (!c.has_ffn) return false;
    return true;
}

std::vector<float> Engine::prefill_batch(const std::vector<int32_t> &tokens,
                                         uint64_t start_position) {
    const ModelConfig &c = manifest_.config;
    if (!supports_batched_prefill())
        fail(ErrorCode::INTERNAL_ERROR, "this checkpoint cannot take the batched prefill path");
    const uint32_t batch = static_cast<uint32_t>(tokens.size());
    if (batch == 0) fail(ErrorCode::INVALID_ARGUMENT, "empty prefill batch");
    if (start_position + batch > c.context)
        fail(ErrorCode::INVALID_ARGUMENT, "KV context exhausted");
    residency_->poll_prefetches();

    const uint32_t H = c.hidden;
    const double norm_bias = c.norm_add_one ? 1.0 : 0.0;
    // `x` is the batch: batch rows of `hidden` floats.
    std::vector<float> x(static_cast<size_t>(batch) * H, 0.0f);

    const TensorView emb = tensor_view("token_embd.weight");
    const QuantGeometry &emb_geo = quant_geometry(emb.type_id);
    const uint64_t emb_row = emb.type_id == 0
        ? static_cast<uint64_t>(H) * sizeof(float)
        : static_cast<uint64_t>(H / emb_geo.block_size) * emb_geo.block_bytes;
    if (emb.type_id != 0 && H % emb_geo.block_size != 0)
        fail(ErrorCode::SHAPE_MISMATCH, "embedding row is not whole quantisation blocks");
    // One row of a norm, in place. Same two rules as `forward_token`: the type is
    // the checkpoint's, and a LayerNorm whose bias tensor is absent is biased by
    // nothing rather than refused.
    auto norm_row = [&](const float *in, uint32_t width, const std::string &prefix,
                        float *out, const std::string &label, uint32_t rows) {
        // A weightless block normalises and does not scale (olmo, talkie), which
        // is upstream's build_norm(x, NULL, NULL, ...).
        const float *w = c.weightless_norm ? nullptr : norm_weights(prefix + ".weight");
        if (c.layer_norm) {
            const std::string bias_name = prefix + ".bias";
            const float *b = has_bias(bias_name) ? bias_weights(bias_name) : nullptr;
            double mean = 0.0;
            for (uint32_t i = 0; i < width; ++i) mean += in[i];
            mean /= width;
            double var = 0.0;
            for (uint32_t i = 0; i < width; ++i) {
                const double d = static_cast<double>(in[i]) - mean;
                var += d * d;
            }
            var /= width;
            const double scale = 1.0 / std::sqrt(var + c.rms_eps);
            for (uint32_t i = 0; i < width; ++i) {
                const double gain = w == nullptr ? 1.0 : static_cast<double>(w[i]);
                out[i] = static_cast<float>((static_cast<double>(in[i]) - mean) * scale * gain +
                                            (b == nullptr ? 0.0 : static_cast<double>(b[i])));
            }
        } else {
            double ss = 0.0;
            for (uint32_t i = 0; i < width; ++i) ss += static_cast<double>(in[i]) * in[i];
            const double scale = 1.0 / std::sqrt(ss / static_cast<double>(width) + c.rms_eps);
            for (uint32_t i = 0; i < width; ++i) {
                const double gain = w == nullptr ? 1.0 : norm_bias + w[i];
                out[i] = static_cast<float>(in[i] * scale * gain);
            }
        }
        backend_->record_control(label, static_cast<uint64_t>(width) * rows * sizeof(float) * 2,
                                 static_cast<uint64_t>(width) * rows * sizeof(float));
    };
    auto norm_batch = [&](const std::vector<float> &in, uint32_t width,
                          const std::string &prefix, const std::string &label) {
        std::vector<float> out(in.size(), 0.0f);
        for (uint32_t t = 0; t < batch; ++t)
            norm_row(in.data() + static_cast<size_t>(t) * width, width, prefix,
                     out.data() + static_cast<size_t>(t) * width, label, batch);
        return out;
    };
    for (uint32_t t = 0; t < batch; ++t) {
        const int32_t token = tokens[t];
        if (token < 0 || static_cast<uint32_t>(token) >= c.vocab)
            fail(ErrorCode::INVALID_ARGUMENT, "token id out of range");
        float *row = x.data() + static_cast<size_t>(t) * H;
        if (emb.type_id == 0) {
            std::memcpy(row, reinterpret_cast<const float *>(emb.data) +
                                 static_cast<size_t>(token) * H,
                        static_cast<size_t>(H) * sizeof(float));
        } else {
            const uint8_t *src = emb.data + static_cast<uint64_t>(token) * emb_row;
            for (uint32_t k = 0; k < H; k += emb_geo.block_size)
                dequantize_block(emb.type_id, src + (k / emb_geo.block_size) * emb_geo.block_bytes,
                                 row + k);
        }
        if (c.embedding_scale != 1.0) {
            const float scale = static_cast<float>(c.embedding_scale);
            for (uint32_t i = 0; i < H; ++i) row[i] *= scale;
        }
        if (c.token_embd_norm) {
            // bloom's pre-block embedding norm, on every row of the batch.
            std::vector<float> normed(H, 0.0f);
            norm_row(row, H, "token_embd_norm", normed.data(), "layernorm.inp", batch);
            std::memcpy(row, normed.data(), static_cast<size_t>(H) * sizeof(float));
        }
        if (c.position_embd) {
            // The learned position table, one row per absolute position.
            const TensorInfo *table = manifest_.tensor("position_embd.weight");
            if (table == nullptr || table->dims.size() < 2)
                fail(ErrorCode::INTERNAL_ERROR, "position_embd.weight is not a table");
            const uint64_t rows = table->dims[1];
            const uint64_t position = start_position + t;
            if (position >= rows)
                fail(ErrorCode::METADATA_MISMATCH,
                     "position " + std::to_string(position) +
                         " is outside the checkpoint's position table (" +
                         std::to_string(rows) + " rows)");
            const TensorView view = tensor_view("position_embd.weight");
            if (view.type_id != 0)
                fail(ErrorCode::UNSUPPORTED_QUANTISATION,
                     "a quantised position table is not supported");
            const float *row_pos = reinterpret_cast<const float *>(view.data) +
                                   static_cast<size_t>(position) * H;
            for (uint32_t i = 0; i < H; ++i) row[i] += row_pos[i];
        }
        Json fields = Json::object();
        fields.set("position", start_position + t);
        fields.set("token", static_cast<int64_t>(token));
        fields.set("phase", std::string("prefill"));
        fields.set("batch", static_cast<uint64_t>(batch));
        trace_->event("token", std::move(fields));
    }

    // Batched projection: one weight row, every token.
    auto project = [&](const std::string &name, const std::vector<float> &in,
                       uint32_t in_dim, uint32_t out_dim, const std::string &op) {
        const TensorView view = tensor_view(name);
        std::vector<float> out(static_cast<size_t>(batch) * out_dim, 0.0f);
        KernelConfig kernels;
        kernels.requested = config_.backend.kernel_isa;
        kernels.threads = config_.backend.kernel_threads;
        // Stride the input at `in_dim`, which is the projection's own width: a
        // shared batch buffer can be wider than the projection it feeds.
        gemm_quant(kernels, view.type_id, view.data, in.data(), out.data(), batch, out_dim, in_dim);
        backend_->record_control(op + ".batch",
                                 static_cast<uint64_t>(out_dim) * in_dim * batch *
                                     (view.type_id == 0 ? 4 : quant_geometry(view.type_id).block_bytes /
                                                              quant_geometry(view.type_id).block_size),
                                 static_cast<uint64_t>(out_dim) * batch * sizeof(float),
                                 "cpu-batched");
        return out;
    };

    const uint32_t kv_dim = c.kv_heads * c.head_dim;
    const uint32_t v_dim = c.v_dim();
    const uint32_t attn_out_dim = c.heads * c.head_v_dim();
    const uint32_t kv_stride = kv_dim + v_dim;
    const uint32_t group = c.heads / c.kv_heads;

    for (uint32_t layer = 0; layer < c.layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        const std::string lp = "L" + std::to_string(layer);
        // --- attention ------------------------------------------------------
        const std::vector<float> h = norm_batch(x, H, prefix + "attn_norm",
                                               "rmsnorm.attn." + lp);
        // falcon's second pre-attention norm, as in the sequential path.
        std::vector<float> h_attn;
        const std::vector<float> *attn_h = &h;
        if (c.attn_norm_2) {
            h_attn = norm_batch(x, H, prefix + "attn_norm_2", "rmsnorm.attn2." + lp);
            attn_h = &h_attn;
        }
        std::vector<float> qflat(static_cast<size_t>(batch) * attn_out_dim, 0.0f);
        std::vector<float> kflat(static_cast<size_t>(batch) * kv_dim, 0.0f);
        std::vector<float> vflat(static_cast<size_t>(batch) * kv_dim, 0.0f);
        if (c.fused_qkv) {
            const uint32_t fused_out = H + 2 * kv_dim;
            const std::vector<float> both =
                project(prefix + "attn_qkv.weight", *attn_h, H, fused_out, "attn_qkv." + lp);
            for (uint32_t t = 0; t < batch; ++t) {
                const float *src = both.data() + static_cast<size_t>(t) * fused_out;
                std::memcpy(qflat.data() + static_cast<size_t>(t) * attn_out_dim, src,
                            static_cast<size_t>(H) * sizeof(float));
                std::memcpy(kflat.data() + static_cast<size_t>(t) * kv_dim, src + H,
                            static_cast<size_t>(kv_dim) * sizeof(float));
                std::memcpy(vflat.data() + static_cast<size_t>(t) * kv_dim, src + H + kv_dim,
                            static_cast<size_t>(kv_dim) * sizeof(float));
            }
        } else {
            qflat = project(prefix + "attn_q.weight", *attn_h, H, attn_out_dim, "attn_q." + lp);
            kflat = project(prefix + "attn_k.weight", *attn_h, H, kv_dim, "attn_k." + lp);
            vflat = project(prefix + "attn_v.weight", *attn_h, H, kv_dim, "attn_v." + lp);
        }
        for (uint32_t t = 0; t < batch; ++t) {
            float *q = qflat.data() + static_cast<size_t>(t) * attn_out_dim;
            float *k = kflat.data() + static_cast<size_t>(t) * kv_dim;
            float *v = vflat.data() + static_cast<size_t>(t) * kv_dim;
            if (c.projection_scales) {
                std::vector<float> qv(q, q + attn_out_dim);
                std::vector<float> kv_(k, k + kv_dim);
                std::vector<float> vv(v, v + kv_dim);
                apply_projection_scale(qv, prefix + "attn_q.scale", "attn_q_scale." + lp);
                apply_projection_scale(kv_, prefix + "attn_k.scale", "attn_k_scale." + lp);
                apply_projection_scale(vv, prefix + "attn_v.scale", "attn_v_scale." + lp);
                std::memcpy(q, qv.data(), qv.size() * sizeof(float));
                std::memcpy(k, kv_.data(), kv_.size() * sizeof(float));
                std::memcpy(v, vv.data(), vv.size() * sizeof(float));
            }
            if (c.attn_qkv_bias) {
                if (c.fused_qkv) {
                    const float *b = bias_weights(prefix + "attn_qkv.bias");
                    for (uint32_t i = 0; i < attn_out_dim; ++i) q[i] += b[i];
                    for (uint32_t i = 0; i < kv_dim; ++i) {
                        k[i] += b[attn_out_dim + i];
                        v[i] += b[attn_out_dim + kv_dim + i];
                    }
                } else {
                    const float *bq = bias_weights(prefix + "attn_q.bias");
                    const float *bk = bias_weights(prefix + "attn_k.bias");
                    const float *bv = bias_weights(prefix + "attn_v.bias");
                    for (uint32_t i = 0; i < attn_out_dim; ++i) q[i] += bq[i];
                    for (uint32_t i = 0; i < kv_dim; ++i) {
                        k[i] += bk[i];
                        v[i] += bv[i];
                    }
                }
            }
            if (c.qk_norm) {
                const float *qs = norm_weights(prefix + "attn_q_norm.weight");
                const float *ks = norm_weights(prefix + "attn_k_norm.weight");
                auto head_norm = [&](float *vec, uint32_t heads, const float *scale) {
                    for (uint32_t hh = 0; hh < heads; ++hh) {
                        float *base = vec + static_cast<size_t>(hh) * c.head_dim;
                        double ss = 0.0;
                        for (uint32_t j = 0; j < c.head_dim; ++j)
                            ss += static_cast<double>(base[j]) * base[j];
                        const double s = 1.0 / std::sqrt(ss / c.head_dim + c.rms_eps);
                        for (uint32_t j = 0; j < c.head_dim; ++j)
                            base[j] = static_cast<float>(base[j] * s * scale[j]);
                    }
                };
                head_norm(q, c.heads, qs);
                head_norm(k, c.kv_heads, ks);
                backend_->record_control("qk_norm." + lp,
                                         static_cast<uint64_t>(attn_out_dim + kv_dim) * sizeof(float),
                                         0);
            }
        }
        const uint64_t position0 = start_position;
        // Rotary, then the per-position temperature, then the cache write: the
        // same order `forward_token` uses.
        const bool rotate = !c.rope_disabled && c.layer_has_rope(layer);
        std::vector<double> alibi_slopes;
        if (c.alibi_max_bias > 0.0 && c.heads > 0) {
            uint32_t n_head_log2 = 1;
            while (n_head_log2 * 2 <= c.heads) n_head_log2 *= 2;
            const double m0 = std::pow(2.0, -(c.alibi_max_bias) / n_head_log2);
            const double m1 = std::pow(2.0, -(c.alibi_max_bias / 2.0) / n_head_log2);
            alibi_slopes.resize(c.heads);
            for (uint32_t hh = 0; hh < c.heads; ++hh)
                alibi_slopes[hh] =
                    hh < n_head_log2
                        ? std::pow(m0, static_cast<double>(hh) + 1.0)
                        : std::pow(m1, 2.0 * static_cast<double>(hh - n_head_log2) + 1.0);
        }
        std::vector<float> attention(static_cast<size_t>(batch) * attn_out_dim, 0.0f);
        const bool windowed = c.layer_uses_window(layer);
        double scale = c.attention_scale > 0.0
                           ? c.attention_scale
                           : 1.0 / std::sqrt(static_cast<double>(c.head_dim));
        // Each token's key and value is written and then attended to before the
        // next token is written. A windowed layer keeps a ring of only
        // `sliding_window` positions, so writing the whole batch up front would
        // evict history that the batch's earlier tokens still have to read; this
        // order makes the ring safe for any batch length.
        for (uint32_t t = 0; t < batch; ++t) {
            std::vector<float> q(qflat.begin() + static_cast<size_t>(t) * attn_out_dim,
                                 qflat.begin() + static_cast<size_t>(t + 1) * attn_out_dim);
            std::vector<float> k(kflat.begin() + static_cast<size_t>(t) * kv_dim,
                                 kflat.begin() + static_cast<size_t>(t + 1) * kv_dim);
            const uint64_t position = position0 + t;
            if (rotate)
                rope_half(q, c.heads, k, c.kv_heads, position, c.layer_rope_theta(layer));
            if (c.attn_temp_scale != 0.0) {
                const double temperature =
                    std::log(std::floor(static_cast<double>(position) / c.attn_temp_floor) + 1.0) *
                        c.attn_temp_scale +
                    1.0;
                for (float &value : q) value = static_cast<float>(value * temperature);
            }
            std::memcpy(qflat.data() + static_cast<size_t>(t) * attn_out_dim, q.data(),
                        q.size() * sizeof(float));
            std::memcpy(kflat.data() + static_cast<size_t>(t) * kv_dim, k.data(),
                        k.size() * sizeof(float));
            const size_t base = static_cast<size_t>(c.cache_index(layer, position, c.context));
            std::memcpy(kv_ + base, k.data(), static_cast<size_t>(kv_dim) * sizeof(float));
            std::memcpy(kv_ + base + kv_dim,
                        vflat.data() + static_cast<size_t>(t) * kv_dim,
                        static_cast<size_t>(v_dim) * sizeof(float));
            // --- attention for this token ------------------------------------
            const uint64_t first_visible =
                !windowed || position < c.sliding_window ? 0 : position + 1 - c.sliding_window;
            const float *qp = q.data();
            std::vector<float> attn_row(attn_out_dim, 0.0f);
            for (uint32_t head = 0; head < c.heads; ++head) {
                const size_t hb = static_cast<size_t>(head) * c.head_dim;
                const size_t kvb = static_cast<size_t>(head / group) * c.head_dim;
                std::vector<float> score_floats;
                score_floats.reserve(position + 1 - first_visible);
                for (uint64_t tt = first_visible; tt <= position; ++tt) {
                    const size_t sbase = static_cast<size_t>(c.cache_index(layer, tt, c.context));
                    const float *kt = kv_ + sbase + kvb;
                    double dot = 0.0;
                    for (uint32_t j = 0; j < c.head_dim; ++j)
                        dot += static_cast<double>(qp[hb + j]) * kt[j];
                    double value = dot * scale;
                    if (c.attn_logit_softcap > 0.0)
                        value = c.attn_logit_softcap * std::tanh(value / c.attn_logit_softcap);
                    if (!alibi_slopes.empty())
                        value += alibi_slopes[head] * -static_cast<double>(position - tt);
                    score_floats.push_back(static_cast<float>(value));
                }
                if (c.attn_sinks) {
                    const float *sinks = norm_weights(prefix + "attn_sinks.weight");
                    score_floats.push_back(sinks[head]);
                }
                const std::vector<float> probs = softmax(score_floats);
                std::vector<double> acc(c.head_v_dim(), 0.0);
                const size_t vb = static_cast<size_t>(head / group) * c.head_v_dim();
                for (uint64_t tt = first_visible; tt <= position; ++tt) {
                    const size_t sbase = static_cast<size_t>(c.cache_index(layer, tt, c.context));
                    const float *vt = kv_ + sbase + kv_dim + vb;
                    const double p = probs[tt - first_visible];
                    for (uint32_t j = 0; j < c.head_v_dim(); ++j) acc[j] += p * vt[j];
                }
                const size_t ab = static_cast<size_t>(head) * c.head_v_dim();
                for (uint32_t j = 0; j < c.head_v_dim(); ++j)
                    attn_row[ab + j] = static_cast<float>(acc[j]);
            }
            std::memcpy(attention.data() + static_cast<size_t>(t) * attn_out_dim, attn_row.data(),
                        attn_row.size() * sizeof(float));
        }
        backend_->record_control("attention." + lp, kv_stride * batch * sizeof(float),
                                 attention.size() * sizeof(float));
        std::vector<float> attn_out =
            project(prefix + "attn_output.weight", attention, attn_out_dim, H, "attn_output." + lp);
        for (uint32_t t = 0; t < batch; ++t) {
            std::vector<float> row(attn_out.begin() + static_cast<size_t>(t) * H,
                                   attn_out.begin() + static_cast<size_t>(t + 1) * H);
            if (c.projection_scales)
                apply_projection_scale(row, prefix + "attn_output.scale", "attn_output_scale." + lp);
            if (c.attn_output_bias) {
                const float *b = bias_weights(prefix + "attn_output.bias");
                for (uint32_t i = 0; i < H; ++i) row[i] += b[i];
            }
            if (c.post_norms) {
                std::vector<float> normed(H, 0.0f);
                norm_row(row.data(), H, prefix + "attn_post_norm", normed.data(),
                         "rmsnorm.attn_post." + lp, batch);
                row.swap(normed);
            }
            if (c.residual_scale_applied) {
                // granite, same rule as the sequential path: the branch output is
                // scaled before it joins the residual stream.
                const float s = static_cast<float>(c.residual_scale);
                for (float &v : row) v *= s;
                backend_->record_control("residual_scale.attn." + lp, H * sizeof(float),
                                         H * sizeof(float));
            }
            float *xt = x.data() + static_cast<size_t>(t) * H;
            for (uint32_t i = 0; i < H; ++i) xt[i] += row[i];
        }
        if (!c.layer_has_ffn(layer)) continue;
        // --- feed-forward ---------------------------------------------------
        const std::vector<float> h2 = norm_batch(x, H, prefix + "ffn_norm",
                                                "rmsnorm.ffn." + lp);
        std::vector<float> down;
        if (c.ffn_no_gate) {
            std::vector<float> up = project(prefix + "ffn_up.weight", h2, H, c.intermediate,
                                            "ffn_up." + lp);
            for (uint32_t t = 0; t < batch; ++t) {
                float *row = up.data() + static_cast<size_t>(t) * c.intermediate;
                if (c.ffn_bias) {
                    const float *b = bias_weights(prefix + "ffn_up.bias");
                    for (uint32_t i = 0; i < c.intermediate; ++i) row[i] += b[i];
                }
                for (uint32_t i = 0; i < c.intermediate; ++i) row[i] = pointwise_activation(c, row[i]);
            }
            backend_->record_control(c.activation + "." + lp,
                                     static_cast<uint64_t>(c.intermediate) * batch * sizeof(float), 0);
            down = project(prefix + "ffn_down.weight", up, c.intermediate, H, "ffn_down." + lp);
        } else {
            std::vector<float> gate;
            std::vector<float> up;
            if (c.fused_gate_up) {
                const std::vector<float> both =
                    project(prefix + "ffn_up.weight", h2, H, 2 * c.intermediate, "ffn_gate_up." + lp);
                gate.resize(static_cast<size_t>(batch) * c.intermediate);
                up.resize(static_cast<size_t>(batch) * c.intermediate);
                for (uint32_t t = 0; t < batch; ++t) {
                    const float *src = both.data() + static_cast<size_t>(t) * 2 * c.intermediate;
                    std::memcpy(gate.data() + static_cast<size_t>(t) * c.intermediate, src,
                                static_cast<size_t>(c.intermediate) * sizeof(float));
                    std::memcpy(up.data() + static_cast<size_t>(t) * c.intermediate,
                                src + c.intermediate,
                                static_cast<size_t>(c.intermediate) * sizeof(float));
                }
            } else {
                gate = project(prefix + "ffn_gate.weight", h2, H, c.intermediate, "ffn_gate." + lp);
                up = project(prefix + "ffn_up.weight", h2, H, c.intermediate, "ffn_up." + lp);
            }
            std::vector<float> act(static_cast<size_t>(batch) * c.intermediate, 0.0f);
            for (uint32_t t = 0; t < batch; ++t) {
                float *g = gate.data() + static_cast<size_t>(t) * c.intermediate;
                float *u = up.data() + static_cast<size_t>(t) * c.intermediate;
                float *a = act.data() + static_cast<size_t>(t) * c.intermediate;
                if (c.ffn_gate_bias) {
                    const float *b = bias_weights(prefix + "ffn_gate.bias");
                    for (uint32_t i = 0; i < c.intermediate; ++i) g[i] += b[i];
                }
                if (c.ffn_bias) {
                    const float *b = bias_weights(prefix + "ffn_up.bias");
                    for (uint32_t i = 0; i < c.intermediate; ++i) u[i] += b[i];
                }
                if (c.projection_scales) {
                    std::vector<float> gv(g, g + c.intermediate);
                    std::vector<float> uv(u, u + c.intermediate);
                    apply_projection_scale(gv, prefix + "ffn_gate.scale", "ffn_gate_scale." + lp);
                    apply_projection_scale(uv, prefix + "ffn_up.scale", "ffn_up_scale." + lp);
                    std::memcpy(g, gv.data(), gv.size() * sizeof(float));
                    std::memcpy(u, uv.data(), uv.size() * sizeof(float));
                }
                for (uint32_t i = 0; i < c.intermediate; ++i)
                    a[i] = gated_activation(c, g[i], u[i], layer);
            }
            backend_->record_control("swiglu." + lp,
                                     static_cast<uint64_t>(c.intermediate) * batch * sizeof(float) * 2,
                                     static_cast<uint64_t>(c.intermediate) * batch * sizeof(float));
            down = project(prefix + "ffn_down.weight", act, c.intermediate, H, "ffn_down." + lp);
        }
        for (uint32_t t = 0; t < batch; ++t) {
            float *row = down.data() + static_cast<size_t>(t) * H;
            if (c.ffn_bias) {
                const float *b = bias_weights(prefix + "ffn_down.bias");
                for (uint32_t i = 0; i < H; ++i) row[i] += b[i];
            }
            if (c.projection_scales) {
                std::vector<float> rv(row, row + H);
                apply_projection_scale(rv, prefix + "ffn_down.scale", "ffn_down_scale." + lp);
                std::memcpy(row, rv.data(), rv.size() * sizeof(float));
            }
            if (c.post_norms) {
                std::vector<float> normed(H, 0.0f);
                norm_row(row, H, prefix + "ffn_post_norm", normed.data(),
                         "rmsnorm.ffn_post." + lp, batch);
                std::memcpy(row, normed.data(), static_cast<size_t>(H) * sizeof(float));
            }
            if (c.residual_scale_applied) {
                const float s = static_cast<float>(c.residual_scale);
                for (uint32_t i = 0; i < H; ++i) row[i] *= s;
                backend_->record_control("residual_scale.ffn." + lp, H * sizeof(float),
                                         H * sizeof(float));
            }
            float *xt = x.data() + static_cast<size_t>(t) * H;
            for (uint32_t i = 0; i < H; ++i) xt[i] += row[i];
        }
    }

    // Only the last token's logits are needed, and every earlier position's
    // head would be discarded.
    const float *last = x.data() + static_cast<size_t>(batch - 1) * H;
    std::vector<float> final_norm(H, 0.0f);
    norm_row(last, H, "output_norm", final_norm.data(), "output_norm.final", 1);
    std::vector<float> logits = matvec_view(tensor_view("output.weight"), final_norm, c.vocab,
                                            H, "output_head");
    if (c.logit_bias) {
        const float *b = bias_weights("output.bias");
        for (uint32_t i = 0; i < c.vocab; ++i) logits[i] += b[i];
    }
    if (c.logit_scale_inverse) {
        // granite divides the logits by `*.logit_scale` after the head.
        const float s = static_cast<float>(1.0 / c.logit_scale);
        for (float &value : logits) value *= s;
        backend_->record_control("logit_scale_inverse", logits.size() * sizeof(float),
                                 logits.size() * sizeof(float));
    }
    if (c.logit_scale_applied) {
        // command-r multiplies them instead; the two flags never share a file.
        const float s = static_cast<float>(c.logit_scale);
        for (float &value : logits) value *= s;
        backend_->record_control("logit_scale", logits.size() * sizeof(float),
                                 logits.size() * sizeof(float));
    }
    if (c.final_logit_softcap > 0.0) {
        for (float &value : logits)
            value = static_cast<float>(c.final_logit_softcap *
                                       std::tanh(static_cast<double>(value) / c.final_logit_softcap));
    }
    return logits;
}

PromptResult Engine::run_prompt(const std::string &prompt, uint32_t generated_tokens) {
    if (!initialized_) fail(ErrorCode::INTERNAL_ERROR, "engine is not initialized");
    PromptResult result;
    result.prompt = prompt;
    result.input_ids = tokenize(prompt);
    if (result.input_ids.size() + generated_tokens + 1 > manifest_.config.context)
        fail(ErrorCode::INVALID_ARGUMENT, "prompt plus generated tokens exceeds context");

    Timer total_timer;
    Timer prefill_timer;
    std::vector<float> logits;
    uint64_t position = 0;
    if (config_.backend.batched_prefill && supports_batched_prefill()) {
        // One pass over the whole prompt: every weight row is fetched once
        // rather than once per token. The result is the same arithmetic in the
        // same order per token, which is what the parity goldens check.
        const std::vector<int32_t> prompt_ids(result.input_ids.begin(), result.input_ids.end());
        logits = prefill_batch(prompt_ids, 0);
        position = prompt_ids.size();
        backend_->record_control("prefill_batched", prompt_ids.size() * sizeof(int32_t),
                                 logits.size() * sizeof(float), "cpu-batched");
    } else {
        for (int32_t token : result.input_ids) {
            logits = forward_token(token, position, "prefill", &result.routing);
            ++position;
        }
    }
    result.prefill_logits = logits;
    result.prefill_ms = prefill_timer.elapsed_ms();
    result.first_token_ms = result.prefill_ms;

    Timer decode_timer;
    // The penalties need to know what has already been produced, so the history
    // starts as the prompt and grows with each sampled token. `--n-probs` rides
    // on the same pass rather than sampling twice.
    std::vector<int32_t> sampling_history(result.input_ids.begin(), result.input_ids.end());
    for (uint32_t i = 0; i < generated_tokens; ++i) {
        Timer step_timer;
        const std::vector<float> biased =
            apply_sampling_bias(logits, config_.sampling, sampling_history);
        const SampledToken sampled = sample_detailed(biased, config_.sampling, rng_);
        const int32_t token = static_cast<int32_t>(sampled.id);
        result.generated_ids.push_back(token);
        if (config_.sampling.n_probs > 0) result.alternatives.push_back(sampled.alternatives);
        sampling_history.push_back(token);
        backend_->record_control(config_.sampling.greedy() ? "greedy_sampling" : "sampled_sampling",
                                 logits.size() * sizeof(float), sizeof(int32_t));
        // Generation ends at the checkpoint's end-of-sequence token. Emitting it
        // and continuing would produce text the model asked not to produce, and
        // it is the one stopping rule every reference implementation shares.
        if (static_cast<uint32_t>(token) == manifest_.config.eos_id) {
            result.inter_token_ms.push_back(step_timer.elapsed_ms());
            break;
        }
        logits = forward_token(token, position, "decode", &result.routing);
        ++position;
        result.inter_token_ms.push_back(step_timer.elapsed_ms());
    }
    result.decode_ms = decode_timer.elapsed_ms();
    result.total_ms = total_timer.elapsed_ms();

    run_metrics_.prompt_count += 1;
    run_metrics_.prompt_tokens += result.input_ids.size();
    run_metrics_.generated_tokens += result.generated_ids.size();
    run_metrics_.prefill_ns += static_cast<uint64_t>(result.prefill_ms * 1e6);
    run_metrics_.decode_ns += static_cast<uint64_t>(result.decode_ms * 1e6);
    run_metrics_.total_ns += static_cast<uint64_t>(result.total_ms * 1e6);
    run_metrics_.first_token_latency_sum_ms += result.first_token_ms;
    run_metrics_.inter_token_ms.insert(run_metrics_.inter_token_ms.end(),
                                       result.inter_token_ms.begin(), result.inter_token_ms.end());
    run_metrics_.finalize();
    results_.push_back(result);
    return result;
}

void Engine::read_reference_logits(const std::string &path, std::vector<float> &out) const {
    std::ifstream in(path);
    if (!in) fail(ErrorCode::IO_ERROR, "cannot open reference logits " + path);
    uint64_t count = 0;
    in >> count;
    out.resize(static_cast<size_t>(count));
    for (auto &v : out) {
        if (!(in >> v)) fail(ErrorCode::IO_ERROR, "short reference logits file " + path);
    }
}

std::vector<int32_t> Engine::read_reference_ids(const std::string &path) const {
    std::ifstream in(path);
    if (!in) fail(ErrorCode::IO_ERROR, "cannot open reference IDs " + path);
    std::vector<int32_t> ids;
    int32_t id;
    while (in >> id) ids.push_back(id);
    return ids;
}

std::vector<RoutingEvent> Engine::read_reference_routing(const std::string &path) const {
    std::ifstream in(path);
    if (!in) fail(ErrorCode::IO_ERROR, "cannot open reference routing " + path);
    std::vector<RoutingEvent> events;
    std::string line;
    while (std::getline(in, line)) {
        if (trim(line).empty()) continue;
        std::istringstream ss(line);
        RoutingEvent event;
        uint64_t token = 0;
        uint32_t layer = 0;
        ss >> token >> layer;
        event.token_index = token;
        event.layer = layer;
        uint32_t expert;
        while (ss >> expert) event.experts.push_back(expert);
        events.push_back(std::move(event));
    }
    return events;
}

ComparisonResult Engine::compare_reference(const std::string &reference_root, uint32_t prompt_limit,
                                           uint32_t generated_tokens) {
    if (!initialized_) fail(ErrorCode::INTERNAL_ERROR, "engine is not initialized");
    uint64_t load_ns = run_metrics_.model_load_ns;
    results_.clear();
    run_metrics_ = RunMetrics{};
    run_metrics_.model_load_ns = load_ns;

    std::filesystem::path root(reference_root);
    std::filesystem::path prompts_dir = root;
    std::filesystem::path ref_dir = root / manifest_.config.model_kind;
    if (root.filename() == manifest_.config.model_kind) {
        prompts_dir = root.parent_path();
        ref_dir = root;
    }

    ComparisonResult comparison;
    comparison.model_kind = manifest_.config.model_kind;
    comparison.reference_dir = ref_dir.string();
    comparison.abs_tol = config_.abs_tol;
    comparison.rel_tol = config_.rel_tol;
    uint32_t count = std::min<uint32_t>(prompt_limit ? prompt_limit : 20, 20);
    for (uint32_t i = 0; i < count; ++i) {
        char stem[32];
        std::snprintf(stem, sizeof(stem), "prompt_%02u", i);
        std::string prompt_path = (prompts_dir / (std::string(stem) + ".txt")).string();
        std::string prompt = strip_eol(read_text_file(prompt_path));
        PromptResult result = run_prompt(prompt, generated_tokens);
        std::vector<float> reference_logits;
        read_reference_logits((ref_dir / (std::string(stem) + ".logits")).string(), reference_logits);
        std::vector<int32_t> reference_ids = read_reference_ids((ref_dir / (std::string(stem) + ".greedy.ids")).string());

        PromptComparison pc;
        pc.index = i;
        pc.vocab = static_cast<uint32_t>(reference_logits.size());
        if (result.prefill_logits.size() != reference_logits.size())
            fail(ErrorCode::INTERNAL_ERROR, "logits vocabulary mismatch for prompt " + std::to_string(i));
        bool within = true;
        for (size_t j = 0; j < reference_logits.size(); ++j) {
            double a = result.prefill_logits[j];
            double b = reference_logits[j];
            double diff = std::abs(a - b);
            double denom = std::max(std::abs(b), 1e-6);
            double rel = diff / denom;
            pc.max_abs_diff = std::max(pc.max_abs_diff, diff);
            pc.max_rel_diff = std::max(pc.max_rel_diff, rel);
            if (!(diff <= config_.abs_tol + config_.rel_tol * std::abs(b))) within = false;
        }
        pc.logits_within_tolerance = within;
        pc.reference_tokens = static_cast<uint32_t>(reference_ids.size());
        uint32_t matching = 0;
        while (matching < reference_ids.size() && matching < result.generated_ids.size() &&
               reference_ids[matching] == result.generated_ids[matching]) ++matching;
        pc.matching_tokens = matching;
        pc.tokens_match = matching == reference_ids.size() &&
                          result.generated_ids.size() >= reference_ids.size();
        if (comparison.model_kind == "moe") {
            std::vector<RoutingEvent> reference_routing =
                read_reference_routing((ref_dir / (std::string(stem) + ".routing.txt")).string());
            pc.routing_events = static_cast<uint32_t>(result.routing.size());
            pc.routing_matches = reference_routing.size() == result.routing.size();
            if (pc.routing_matches) {
                for (size_t r = 0; r < reference_routing.size(); ++r) {
                    const auto &a = reference_routing[r];
                    const auto &b = result.routing[r];
                    if (a.token_index != b.token_index || a.layer != b.layer || a.experts != b.experts) {
                        ++pc.routing_mismatches;
                        pc.routing_matches = false;
                    }
                }
            } else {
                pc.routing_mismatches = static_cast<uint32_t>(std::max(reference_routing.size(), result.routing.size()));
            }
        } else {
            pc.routing_matches = true;
        }
        comparison.prompts += 1;
        if (pc.logits_within_tolerance) ++comparison.prompts_within_tolerance;
        if (pc.tokens_match) ++comparison.prompts_token_match;
        if (pc.routing_matches) ++comparison.prompts_routing_match;
        comparison.matching_tokens += pc.matching_tokens;
        comparison.reference_tokens += pc.reference_tokens;
        comparison.max_abs_diff = std::max(comparison.max_abs_diff, pc.max_abs_diff);
        comparison.max_rel_diff = std::max(comparison.max_rel_diff, pc.max_rel_diff);
        comparison.per_prompt.push_back(pc);
    }
    run_metrics_.finalize();
    last_comparison_ = std::make_unique<ComparisonResult>(comparison);
    return comparison;
}

Json Engine::metrics_json() const {
    RssSnapshot process = process_snapshot();
    Json j = Json::object();
    j.set("schema", "freemoe.run.v1");
    j.set("run_id", config_.run_id);
    j.set("model_path", config_.model_path);
    j.set("model_kind", manifest_.config.model_kind);
    j.set("architecture_adapter", manifest_.config.adapter);
    j.set("reader", reader_mode_name(config_.reader_mode));
    j.set("backend", backend_ ? backend_->metrics_json() : Json::object());
    j.set("device", backend_ ? backend_->info().to_json() : Json::object());
    if (residency_) j.set("residency", residency_->metrics_json());
    else if (released_residency_snapshot_.type() != Json::Type::Null)
        j.set("residency", released_residency_snapshot_);
    else j.set("residency", Json::object());
    j.set("io", scheduler_ ? scheduler_->metrics().to_json() : Json::object());
    j.set("timing", run_metrics_.to_json());

    Json proc = Json::object();
    proc.set("peak_process_ram_bytes", process.max_rss_kib * 1024);
    proc.set("minor_faults", process.minor_faults);
    proc.set("major_faults", process.major_faults);
    proc.set("user_cpu_s", fixed_double(process.user_cpu_s, 6));
    proc.set("system_cpu_s", fixed_double(process.system_cpu_s, 6));
    proc.set("peak_vram_bytes", 0);
    proc.set("pinned_host_bytes", 0);
    proc.set("gpu_utilisation", Json::null());
    proc.set("cpu_utilisation", Json::null());
    j.set("process", std::move(proc));

    std::string correctness = "not_evaluated";
    if (last_comparison_) {
        bool pass = last_comparison_->prompts > 0 &&
                    last_comparison_->prompts_within_tolerance == last_comparison_->prompts &&
                    last_comparison_->prompts_token_match == last_comparison_->prompts &&
                    (manifest_.config.model_kind != "moe" ||
                     last_comparison_->prompts_routing_match == last_comparison_->prompts);
        correctness = pass ? "pass" : "fail";
    }
    j.set("output_correctness_status", correctness);
    j.set("cache_state", run_metrics_.cache_state);
    j.set("trace_path", trace_ ? trace_->path() : std::string());
    Json events = Json::array();
    if (trace_) for (const auto &event : trace_->event_types()) events.push(Json::string(event));
    j.set("trace_event_types", std::move(events));

    Json prompts = Json::array();
    for (const auto &result : results_) prompts.push(result.to_json(false));
    j.set("prompts", std::move(prompts));
    return j;
}

Json Engine::report_json(const ComparisonResult *comparison) const {
    Json j = metrics_json();
    j.set("schema", "freemoe.report.v1");
    if (comparison) j.set("comparison", comparison->to_json());
    else if (last_comparison_) j.set("comparison", last_comparison_->to_json());
    else j.set("comparison", Json::null());
    return j;
}

void Engine::release_runtime() {
    if (residency_) {
        released_residency_snapshot_ = residency_->metrics_json();
        residency_.reset();
    }
    if (scheduler_) scheduler_->wait_idle();
    staging_buffer_.reset();
    workspace_buffer_.reset();
    kv_buffer_.reset();
    weights_buffer_.reset();
    kv_ = nullptr;
    weight_offsets_.clear();
    if (trace_) trace_->event("runtime_released", Json::object());
}

void Engine::write_metrics(const std::string &path) const {
    write_text_file(path, metrics_json().dump() + "\n");
}

void Engine::write_report(const std::string &path, const ComparisonResult *comparison) const {
    write_text_file(path, report_json(comparison).dump() + "\n");
}

} // namespace fm
