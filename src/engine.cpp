#include "fm/engine.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
    if (initialized_) return;
    Timer load_timer;
    file_ = std::make_unique<MappedFile>(config_.model_path);
    doc_ = parse_gguf(file_->data(), file_->size());
    manifest_ = build_manifest(doc_, config_.model_path);
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

void Engine::load_fixed_weights() {
    if (manifest_.fixed_weight_bytes == 0)
        fail(ErrorCode::INVALID_ARGUMENT, "model has no fixed weights");
    weights_buffer_ = backend_->allocate(manifest_.fixed_weight_bytes, AllocationTag::Weights);
    kv_buffer_ = backend_->allocate(budget_.kv_bytes, AllocationTag::KV);
    workspace_buffer_ = backend_->allocate(budget_.workspace_bytes, AllocationTag::Workspace);
    staging_buffer_ = backend_->allocate(budget_.staging_bytes, AllocationTag::Staging);
    kv_ = kv_buffer_->f32();

    std::vector<uint8_t> host_weights(static_cast<size_t>(manifest_.fixed_weight_bytes));
    uint64_t offset = 0;
    for (const auto &tensor : manifest_.doc.tensors) {
        if (tensor.expert_tensor) continue;
        weight_offsets_[tensor.name] = offset;
        std::memcpy(host_weights.data() + offset, file_->data() + tensor.absolute_offset,
                    static_cast<size_t>(tensor.byte_size));
        offset += tensor.byte_size;
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

const float *Engine::weights(const std::string &name) const {
    auto it = weight_offsets_.find(name);
    if (it == weight_offsets_.end())
        fail(ErrorCode::MISSING_REQUIRED_TENSOR, "tensor is not resident in fixed weight buffer: " + name);
    return reinterpret_cast<const float *>(static_cast<const uint8_t *>(weights_buffer_->data) + it->second);
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
        if (!std::isfinite(scores[a]) || !std::isfinite(scores[b]))
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

void Engine::rope_half(std::vector<float> &q, std::vector<float> &k, uint64_t position) const {
    const ModelConfig &c = manifest_.config;
    uint32_t half = c.head_dim / 2;
    for (uint32_t h = 0; h < c.heads; ++h) {
        size_t base = static_cast<size_t>(h) * c.head_dim;
        for (uint32_t i = 0; i < half; ++i) {
            double inv = std::pow(c.rope_theta, -(2.0 * static_cast<double>(i)) / c.head_dim);
            double angle = static_cast<double>(position) * inv;
            double cs = std::cos(angle);
            double sn = std::sin(angle);
            for (std::vector<float> *vec : {&q, &k}) {
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

std::vector<int32_t> Engine::tokenize(const std::string &prompt) const {
    const ModelConfig &c = manifest_.config;
    if (prompt.empty() || prompt.size() + 2 > c.context)
        fail(ErrorCode::INVALID_ARGUMENT, "prompt is empty or exceeds context");
    std::vector<int32_t> ids;
    ids.reserve(prompt.size() + 1);
    ids.push_back(static_cast<int32_t>(c.bos_id));
    for (unsigned char byte : prompt) ids.push_back(static_cast<int32_t>(byte));
    return ids;
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

    auto rmsnorm = [&](const std::vector<float> &x, const float *w, const std::string &name) {
        double ss = 0.0;
        for (float v : x) ss += static_cast<double>(v) * v;
        double scale = 1.0 / std::sqrt(ss / c.hidden + c.rms_eps);
        std::vector<float> y(x.size());
        for (uint32_t i = 0; i < x.size(); ++i)
            y[i] = static_cast<float>(x[i] * scale * w[i]);
        backend_->record_control(name, x.size() * sizeof(float) * 2, y.size() * sizeof(float));
        return y;
    };
    auto silu = [](float x) {
        return static_cast<float>(static_cast<double>(x) / (1.0 + std::exp(-static_cast<double>(x))));
    };
    auto residual_add = [&](std::vector<float> &x, const std::vector<float> &delta, const std::string &name) {
        for (size_t i = 0; i < x.size(); ++i) x[i] += delta[i];
        backend_->record_control(name, (x.size() + delta.size()) * sizeof(float), x.size() * sizeof(float));
    };

    std::vector<float> x(c.hidden);
    const float *embedding = weights("token_embd.weight");
    std::memcpy(x.data(), embedding + static_cast<size_t>(token) * c.hidden,
                static_cast<size_t>(c.hidden) * sizeof(float));

    for (uint32_t layer = 0; layer < c.layers; ++layer) {
        const std::string prefix = "blk." + std::to_string(layer) + ".";
        std::vector<float> h = rmsnorm(x, weights(prefix + "attn_norm.weight"),
                                       "rmsnorm.attn.L" + std::to_string(layer));
        std::vector<float> q = matvec(weights(prefix + "attn_q.weight"), h, c.hidden, c.hidden,
                                      "attn_q.L" + std::to_string(layer));
        std::vector<float> k = matvec(weights(prefix + "attn_k.weight"), h, c.hidden, c.hidden,
                                      "attn_k.L" + std::to_string(layer));
        std::vector<float> v = matvec(weights(prefix + "attn_v.weight"), h, c.hidden, c.hidden,
                                      "attn_v.L" + std::to_string(layer));
        rope_half(q, k, position);

        size_t current_base = (static_cast<size_t>(layer) * c.context + position) * 2 * c.hidden;
        std::memcpy(kv_ + current_base, k.data(), static_cast<size_t>(c.hidden) * sizeof(float));
        std::memcpy(kv_ + current_base + c.hidden, v.data(), static_cast<size_t>(c.hidden) * sizeof(float));

        std::vector<float> attention(c.hidden, 0.0f);
        double scale = 1.0 / std::sqrt(static_cast<double>(c.head_dim));
        for (uint32_t head = 0; head < c.heads; ++head) {
            size_t hb = static_cast<size_t>(head) * c.head_dim;
            std::vector<double> scores(position + 1);
            for (uint64_t t = 0; t <= position; ++t) {
                size_t base = (static_cast<size_t>(layer) * c.context + t) * 2 * c.hidden;
                const float *kt = kv_ + base + hb;
                double dot = 0.0;
                for (uint32_t j = 0; j < c.head_dim; ++j)
                    dot += static_cast<double>(q[hb + j]) * kt[j];
                scores[t] = dot * scale;
            }
            std::vector<float> score_floats(scores.size());
            for (size_t i = 0; i < scores.size(); ++i) score_floats[i] = static_cast<float>(scores[i]);
            std::vector<float> probs = softmax(score_floats);
            std::vector<double> acc(c.head_dim, 0.0);
            for (uint64_t t = 0; t <= position; ++t) {
                size_t base = (static_cast<size_t>(layer) * c.context + t) * 2 * c.hidden;
                const float *vt = kv_ + base + c.hidden + hb;
                double p = probs[t];
                for (uint32_t j = 0; j < c.head_dim; ++j) acc[j] += p * vt[j];
            }
            for (uint32_t j = 0; j < c.head_dim; ++j)
                attention[hb + j] = static_cast<float>(acc[j]);
        }
        backend_->record_control("attention.L" + std::to_string(layer),
                                 (position + 1) * c.hidden * 2 * sizeof(float),
                                 c.hidden * sizeof(float));
        std::vector<float> attn_out = matvec(weights(prefix + "attn_output.weight"), attention,
                                             c.hidden, c.hidden, "attn_output.L" + std::to_string(layer));
        residual_add(x, attn_out, "residual.attn.L" + std::to_string(layer));

        h = rmsnorm(x, weights(prefix + "ffn_norm.weight"), "rmsnorm.ffn.L" + std::to_string(layer));
        std::vector<float> ffn(c.hidden, 0.0f);
        if (c.model_kind == "dense") {
            std::vector<float> gate = matvec(weights(prefix + "ffn_gate.weight"), h,
                                             c.intermediate, c.hidden, "ffn_gate.L" + std::to_string(layer));
            std::vector<float> up = matvec(weights(prefix + "ffn_up.weight"), h,
                                           c.intermediate, c.hidden, "ffn_up.L" + std::to_string(layer));
            std::vector<float> act(c.intermediate);
            for (uint32_t i = 0; i < c.intermediate; ++i) act[i] = silu(gate[i]) * up[i];
            backend_->record_control("swiglu.L" + std::to_string(layer),
                                     (gate.size() + up.size()) * sizeof(float),
                                     act.size() * sizeof(float));
            ffn = matvec(weights(prefix + "ffn_down.weight"), act,
                         c.hidden, c.intermediate, "ffn_down.L" + std::to_string(layer));
        } else {
            std::vector<float> scores = matvec(weights(prefix + "ffn_router.weight"), h,
                                               c.expert_count, c.hidden, "router.L" + std::to_string(layer));
            std::vector<uint32_t> ids = topk(scores, c.expert_top_k);
            std::vector<float> selected_scores(c.expert_top_k);
            for (uint32_t i = 0; i < c.expert_top_k; ++i) selected_scores[i] = scores[ids[i]];
            std::vector<float> selected_weights = softmax(selected_scores);
            backend_->record_control("router_topk.L" + std::to_string(layer),
                                     scores.size() * sizeof(float),
                                     ids.size() * sizeof(uint32_t));
            residency_->note_selected(layer, ids, position);
            if (routing) routing->push_back({position, layer, ids});
            for (uint32_t expert : ids)
                residency_->issue_prefetch(layer, expert, position + 1);

            for (uint32_t i = 0; i < ids.size(); ++i) {
                float weight = selected_weights[i];
                uint32_t expert = ids[i];
                residency_->with_expert(layer, expert, position, [&](const float *expert_data) {
                    std::vector<float> gate = matvec(expert_data, h, c.intermediate, c.hidden,
                                                      "expert_gate.L" + std::to_string(layer) + ".E" + std::to_string(expert));
                    std::vector<float> up = matvec(expert_data + static_cast<size_t>(c.intermediate) * c.hidden,
                                                   h, c.intermediate, c.hidden,
                                                   "expert_up.L" + std::to_string(layer) + ".E" + std::to_string(expert));
                    std::vector<float> act(c.intermediate);
                    for (uint32_t j = 0; j < c.intermediate; ++j) act[j] = silu(gate[j]) * up[j];
                    backend_->record_control("expert_swiglu.L" + std::to_string(layer),
                                             (gate.size() + up.size()) * sizeof(float),
                                             act.size() * sizeof(float));
                    std::vector<float> down = matvec(expert_data + 2ull * c.intermediate * c.hidden,
                                                     act, c.hidden, c.intermediate,
                                                     "expert_down.L" + std::to_string(layer) + ".E" + std::to_string(expert));
                    for (uint32_t j = 0; j < c.hidden; ++j) ffn[j] += weight * down[j];
                });
            }
            residency_->assert_selected_executed();
            backend_->record_control("expert_combine.L" + std::to_string(layer),
                                     ids.size() * c.hidden * sizeof(float),
                                     c.hidden * sizeof(float));
        }
        residual_add(x, ffn, "residual.ffn.L" + std::to_string(layer));
    }

    std::vector<float> final_norm = rmsnorm(x, weights("output_norm.weight"), "rmsnorm.final");
    std::vector<float> logits = matvec(weights("output.weight"), final_norm,
                                       c.vocab, c.hidden, "output_head");
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
    for (int32_t token : result.input_ids) {
        logits = forward_token(token, position, "prefill", &result.routing);
        ++position;
    }
    result.prefill_logits = logits;
    result.prefill_ms = prefill_timer.elapsed_ms();
    result.first_token_ms = result.prefill_ms;

    Timer decode_timer;
    for (uint32_t i = 0; i < generated_tokens; ++i) {
        Timer step_timer;
        int32_t token = static_cast<int32_t>(greedy(logits));
        result.generated_ids.push_back(token);
        backend_->record_control("greedy_sampling", logits.size() * sizeof(float), sizeof(int32_t));
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
