#include "fm/backend.hpp"
#include "fm/engine.hpp"
#include "fm/gguf.hpp"
#include "fm/quant.hpp"
#include "fm/support.hpp"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

namespace {

using fm::Json;

struct Args {
    std::vector<std::string> positional;
    std::map<std::string, std::string> options;

    bool has(const std::string &key) const { return options.count(key) != 0; }
    std::string get(const std::string &key, const std::string &fallback = "") const {
        auto it = options.find(key);
        return it == options.end() ? fallback : it->second;
    }
    uint64_t get_u64(const std::string &key, uint64_t fallback) const {
        if (!has(key)) return fallback;
        try {
            size_t used = 0;
            std::string value = get(key);
            unsigned long long parsed = std::stoull(value, &used, 0);
            if (used != value.size()) throw std::invalid_argument("trailing characters");
            return static_cast<uint64_t>(parsed);
        } catch (const std::exception &) {
            fm::fail(fm::ErrorCode::INVALID_ARGUMENT, "invalid integer for --" + key + ": " + get(key));
        }
    }
    uint32_t get_u32(const std::string &key, uint32_t fallback) const {
        uint64_t value = get_u64(key, fallback);
        if (value > 0xffffffffull)
            fm::fail(fm::ErrorCode::INVALID_ARGUMENT, "value out of range for --" + key);
        return static_cast<uint32_t>(value);
    }
    double get_double(const std::string &key, double fallback) const {
        if (!has(key)) return fallback;
        try {
            size_t used = 0;
            std::string value = get(key);
            double parsed = std::stod(value, &used);
            if (used != value.size()) throw std::invalid_argument("trailing characters");
            return parsed;
        } catch (const std::exception &) {
            fm::fail(fm::ErrorCode::INVALID_ARGUMENT, "invalid number for --" + key + ": " + get(key));
        }
    }
    bool get_bool(const std::string &key, bool fallback) const {
        if (!has(key)) return fallback;
        std::string value = get(key);
        std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (value == "0" || value == "false" || value == "no" || value == "off") return false;
        if (value == "1" || value == "true" || value == "yes" || value == "on" || value.empty()) return true;
        fm::fail(fm::ErrorCode::INVALID_ARGUMENT, "invalid boolean for --" + key + ": " + get(key));
    }
};

Args parse_args(int argc, char **argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        std::string token = argv[i];
        if (token.rfind("--", 0) == 0) {
            std::string key = token.substr(2);
            std::string value = "1";
            auto eq = key.find('=');
            if (eq != std::string::npos) {
                value = key.substr(eq + 1);
                key = key.substr(0, eq);
            } else if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
                value = argv[++i];
            }
            args.options[key] = value;
        } else {
            args.positional.push_back(token);
        }
    }
    return args;
}

void print_usage() {
    std::cout <<
        "FreeMoe correctness prototype\n"
        "Usage:\n"
        "  freemoe inspect <model.gguf> [--json] [--parse-only] [--generic] [--output path]\n"
        "  freemoe WAKE_UP <model.gguf> [--json]     # alias for inspect\n"
        "  freemoe device-info --backend cpu|vnpu|gpu [--json]\n"
        "  freemoe run --model <model.gguf> [--backend cpu|vnpu|gpu] [options]\n"
        "  freemoe compare --model <model.gguf> [--backend cpu|vnpu|gpu] [options]\n"
        "  freemoe bench [--iterations N] [--out path]      # CPU kernel benchmark\n"
        "\n"
        "Common options:\n"
        "  --reader resident|buffered|mmap|direct  --cache-slots N\n"
        "  --context N                    cap the KV context for large models\n"
        "  --budget-mb N                  host memory budget for weights and expert cache\n"
        "  --replacement lru|clock                --tokens N\n"
        "  --queue-depth N --coalesce-gap N       --t-wait-ms N\n"
        "  --prefetch 0|1 --budget-mb N           --vnpu-isa PROFILE\n"
        "      PROFILE = zen3-auto|auto|avx2_dual|zen3|avx2|avx512|scalar\n"
        "  --kernels auto|scalar|avx2|avx512 --threads N   # CPU kernel dispatch\n"
        "  --temperature X --top-k N --top-p X --seed N    # sampling (run only)\n"
        "  --allow-cpu-fallback                    --force-direct-fallback\n"
        "  --test-simulate-zen3 (device-info only)# force Zen 3 fallback diagnostics\n"
        "  --trace path --metrics path --report path --run-id ID --results-dir path\n"
        "  --reference-root path --limit N --abs-tol X --rel-tol X\n"
        "  --prompt TEXT --prompt-file PATH --dump-logits PATH\n"
        "\n"
        "  freemoe probe --model FILE [FILE ...]  report what this build can do with a model\n";
}

std::string model_path(const Args &args) {
    if (args.has("model")) return args.get("model");
    if (args.positional.size() >= 2) return args.positional[1];
    fm::fail(fm::ErrorCode::INVALID_ARGUMENT, "missing model path");
}

fm::RunConfig build_run_config(const Args &args) {
    fm::RunConfig cfg;
    cfg.model_path = model_path(args);
    cfg.run_id = args.get("run-id", "local");
    cfg.backend.kind = fm::parse_backend_kind(args.get("backend", "cpu"));
    cfg.backend.allow_cpu_fallback = args.has("allow-cpu-fallback");
    cfg.backend.vnpu_isa = args.get("vnpu-isa", "avx2_dual");
    cfg.backend.vnpu_threads = args.get_u32("vnpu-threads", 0);
    cfg.backend.kernel_isa = fm::parse_kernel_isa(args.get("kernels", "auto"));
    cfg.backend.kernel_threads = args.get_u32("threads", 0);
    cfg.reader_mode = fm::parse_reader_mode(args.get("reader", "buffered"));
    cfg.cache_slots = args.get_u32("cache-slots", 0);
    cfg.queue_depth = std::max(1u, args.get_u32("queue-depth", 2));
    cfg.coalesce_gap = args.get_u64("coalesce-gap", 4096);
    cfg.t_wait_ms = args.get_u32("t-wait-ms", 100);
    cfg.generated_tokens = args.get_u32("tokens", fm::DEFAULT_N_GREEDY);
    cfg.context_limit = args.get_u32("context", 0);
    cfg.prompt_limit = args.get_u32("limit", 0);
    cfg.force_direct_fallback = args.has("force-direct-fallback");
    cfg.budget.total_hot_bytes = args.get_u64("budget-mb", 64) * 1024ull * 1024ull;
    cfg.budget.workspace_bytes = args.get_u64("workspace-kb", 1024) * 1024ull;
    cfg.budget.staging_bytes = args.get_u64("staging-kb", 1024) * 1024ull;
    cfg.budget.margin_bytes = args.get_u64("margin-kb", 4096) * 1024ull;
    cfg.residency.replacement = fm::parse_replacement_policy(args.get("replacement", "lru"));
    cfg.residency.enable_prefetch = args.get_bool("prefetch", true);
    cfg.residency.max_prefetch = args.get_u32("max-prefetch", cfg.queue_depth);
    cfg.residency.host_cache_records = args.get_u32("host-cache-records", 4);
    cfg.trace_path = args.get("trace", "");
    cfg.metrics_path = args.get("metrics", "");
    cfg.report_path = args.get("report", "");
    cfg.reference_root = args.get("reference-root", "tests/reference");
    cfg.results_dir = args.get("results-dir", "results");
    cfg.sampling.temperature = args.get_double("temperature", 0.0);
    cfg.sampling.top_k = args.get_u32("top-k", 0);
    cfg.sampling.top_p = args.get_double("top-p", 1.0);
    cfg.sampling.seed = args.get_u64("seed", 0);
    if (cfg.sampling.temperature < 0.0)
        fm::fail(fm::ErrorCode::INVALID_ARGUMENT, "--temperature must not be negative");
    if (cfg.sampling.top_p < 0.0 || cfg.sampling.top_p > 1.0)
        fm::fail(fm::ErrorCode::INVALID_ARGUMENT, "--top-p must be within [0, 1]");
    cfg.abs_tol = args.get_double("abs-tol", fm::DEFAULT_ABS_TOL);
    cfg.rel_tol = args.get_double("rel-tol", fm::DEFAULT_REL_TOL);
    return cfg;
}

std::string load_prompt(const Args &args) {
    if (args.has("prompt")) return args.get("prompt");
    if (args.has("prompt-file")) return fm::strip_eol(fm::read_text_file(args.get("prompt-file")));
    std::ifstream in("tests/reference/prompts.txt");
    std::string line;
    if (in && std::getline(in, line)) return fm::strip_eol(line);
    return "the quick brown fox";
}

// The synthetic tokenizer emits raw bytes, which are not always valid UTF-8.
// Render them with printable ASCII preserved and everything else as \xNN so the
// JSON stays valid and the output is unambiguous.
std::string display_text(const std::string &bytes) {
    static const char *hex = "0123456789abcdef";
    std::string out;
    out.reserve(bytes.size());
    for (unsigned char c : bytes) {
        if (c >= 0x20 && c < 0x7F && c != '\\') {
            out.push_back(static_cast<char>(c));
        } else {
            out += "\\x";
            out.push_back(hex[(c >> 4) & 0xF]);
            out.push_back(hex[c & 0xF]);
        }
    }
    return out;
}

Json json_result(const std::string &status) {
    Json j = Json::object();
    j.set("status", status);
    return j;
}

int command_inspect(const Args &args) {
    std::string path = model_path(args);
    auto doc = fm::parse_gguf_file(path);
    if (args.has("generic")) {
        Json view = fm::generic_inspect_json(doc, path);
        if (args.has("out") || args.has("output"))
            fm::write_text_file(args.get("output", args.get("out")), view.dump() + "\n");
        std::cout << view.dump() << "\n";
        return 0;
    }
    if (args.has("parse-only")) {
        Json j = Json::object();
        j.set("schema", "freemoe.gguf-parse.v1");
        j.set("status", "parsed");
        j.set("source_path", path);
        j.set("gguf_version", static_cast<int64_t>(doc.version));
        j.set("metadata_count", doc.metadata_count);
        j.set("tensor_count", doc.tensor_count);
        j.set("data_offset", doc.data_offset);
        j.set("source_fingerprint", fm::hex64(doc.source_fingerprint));
        Json types = Json::array();
        std::set<uint32_t> unique;
        for (const auto &kv : doc.metadata) unique.insert(static_cast<uint32_t>(kv.second.type));
        for (uint32_t type : unique) types.push(Json::integer(type));
        j.set("metadata_value_types", std::move(types));
        std::string output = j.dump() + "\n";
        if (args.has("output")) fm::write_text_file(args.get("output"), output);
        std::cout << output;
        return 0;
    }
    auto manifest = fm::build_manifest(doc, path);
    Json report = fm::inspect_json(manifest, path);
    if (args.has("json") || args.has("output")) {
        std::string output = report.dump() + "\n";
        if (args.has("output")) fm::write_text_file(args.get("output"), output);
        std::cout << output;
    } else {
        std::cout << "model: " << path << "\n"
                  << "adapter: " << manifest.config.adapter << " (" << manifest.config.model_kind << ")\n"
                  << "GGUF v" << doc.version << ", tensors=" << doc.tensors.size()
                  << ", tensor_bytes=" << [&]() {
                         uint64_t total = 0;
                         for (const auto &t : doc.tensors) total += t.byte_size;
                         return total;
                     }() << "\n"
                  << "expert_records=" << manifest.experts.size()
                  << ", slot_bytes=" << manifest.slot_bytes << "\n"
                  << "source_fingerprint=" << fm::hex64(doc.source_fingerprint) << "\n"
                  << "validation: OK\n";
    }
    return 0;
}

int command_device_info(const Args &args) {
    fm::BackendConfig cfg;
    cfg.kind = fm::parse_backend_kind(args.get("backend", "cpu"));
    cfg.allow_cpu_fallback = args.has("allow-cpu-fallback");
    cfg.vnpu_isa = args.get("vnpu-isa", "avx2_dual");
    cfg.vnpu_threads = args.get_u32("vnpu-threads", 0);
    cfg.test_simulate_zen3 = args.has("test-simulate-zen3");
    fm::TraceWriter trace(args.get("trace", ""));
    auto backend = fm::Backend::create(cfg, trace.path().empty() ? nullptr : &trace);
    Json info = backend->info().to_json();
    if (args.has("json")) {
        std::cout << info.dump() << "\n";
    } else {
        for (const auto &member : info.members()) {
            if (member.second.type() == Json::Type::String)
                std::cout << member.first << ": " << member.second.str() << "\n";
            else if (member.second.type() == Json::Type::Int)
                std::cout << member.first << ": " << member.second.integer_value() << "\n";
            else if (member.second.type() == Json::Type::Bool)
                std::cout << member.first << ": " << (member.second.bool_value() ? "true" : "false") << "\n";
        }
    }
    return 0;
}

int command_run(const Args &args) {
    fm::RunConfig cfg = build_run_config(args);
    fm::Engine engine(cfg);
    engine.initialize();
    std::string prompt = load_prompt(args);
    auto result = engine.run_prompt(prompt, cfg.generated_tokens);

    Json out = json_result("ok");
    out.set("run_id", cfg.run_id);
    out.set("model", cfg.model_path);
    out.set("backend", engine.backend().name());
    out.set("reader", fm::reader_mode_name(cfg.reader_mode));
    out.set("prompt_bytes", static_cast<uint64_t>(prompt.size()));
    out.set("input_tokens", static_cast<uint64_t>(result.input_ids.size()));
    out.set("first_token_ms", fm::fixed_double(result.first_token_ms, 6));
    out.set("prefill_ms", fm::fixed_double(result.prefill_ms, 6));
    out.set("decode_ms", fm::fixed_double(result.decode_ms, 6));
    Json generated = Json::array();
    for (int32_t id : result.generated_ids) generated.push(Json::integer(id));
    out.set("generated_ids", std::move(generated));
    out.set("text", display_text(engine.detokenize(result.generated_ids)));
    out.set("sampling_temperature", fm::fixed_double(cfg.sampling.temperature, 4));
    out.set("sampling_top_k", static_cast<int64_t>(cfg.sampling.top_k));
    out.set("sampling_top_p", fm::fixed_double(cfg.sampling.top_p, 4));
    out.set("sampling_seed", cfg.sampling.seed);
    if (args.has("dump-logits")) {
        std::ostringstream os;
        os << result.prefill_logits.size() << "\n";
        for (size_t i = 0; i < result.prefill_logits.size(); ++i) {
            if (i) os << ' ';
            os << fm::fixed_double(result.prefill_logits[i], 9);
        }
        os << "\n";
        fm::write_text_file(args.get("dump-logits"), os.str());
        out.set("dumped_logits", args.get("dump-logits"));
    }
    engine.release_runtime();
    if (!cfg.metrics_path.empty()) {
        engine.write_metrics(cfg.metrics_path);
        out.set("metrics_path", cfg.metrics_path);
    }
    if (!cfg.report_path.empty()) {
        engine.write_report(cfg.report_path);
        out.set("report_path", cfg.report_path);
    }
    if (!cfg.trace_path.empty()) out.set("trace_path", cfg.trace_path);
    std::cout << out.dump() << "\n";
    return 0;
}


Json kernel_bench_json(uint32_t iterations) {
    struct Shape { const char *label; uint32_t out_dim; uint32_t in_dim; };
    static const Shape shapes[] = {
        {"qkv_1280x1280", 1280, 1280},
        {"expert_gate_1920x1280", 1920, 1280},
        {"expert_down_1280x1920", 1280, 1920},
        {"output_head_32000x1280", 32000, 1280},
        {"toy_hidden_16x32", 16, 32},
    };
    struct Candidate { const char *label; fm::KernelIsa isa; uint32_t threads; };
    std::vector<Candidate> candidates = {
        {"scalar_1t", fm::KernelIsa::Scalar, 1},
        {"isa_1t", fm::KernelIsa::Auto, 1},
        {"isa_threads", fm::KernelIsa::Auto, 0},
    };
#if defined(__x86_64__) || defined(__i386__)
    if (fm::host_has_avx2_fma()) candidates.push_back({"avx2_1t", fm::KernelIsa::Avx2Fma, 1});
    if (fm::host_has_avx512()) candidates.push_back({"avx512_1t", fm::KernelIsa::Avx512, 1});
#endif

    Json out = Json::object();
    out.set("schema", "freemoe.kernel-bench.v1");
    out.set("status", "pass");
    out.set("host_cpu", fm::host_cpu_name());
    out.set("detected_avx2_fma", fm::host_has_avx2_fma());
    out.set("detected_avx512", fm::host_has_avx512());
    out.set("zen3_detected", fm::host_is_zen3());
    fm::KernelConfig auto_cfg;
    out.set("resolved_isa", fm::kernel_isa_name(fm::resolved_isa(auto_cfg)));
    out.set("resolved_threads", static_cast<int64_t>(fm::resolved_threads(auto_cfg)));
    out.set("iterations", static_cast<int64_t>(iterations));

    Json results = Json::array();
    for (const auto &shape : shapes) {
        for (const auto &candidate : candidates) {
            fm::KernelConfig cfg;
            cfg.requested = candidate.isa;
            cfg.threads = candidate.threads;
            auto bench = fm::benchmark_gemv(cfg, shape.out_dim, shape.in_dim, iterations);
            Json row = Json::object();
            row.set("shape", shape.label);
            row.set("candidate", candidate.label);
            row.set("isa", bench.isa);
            row.set("threads", static_cast<int64_t>(bench.threads));
            row.set("out_dim", static_cast<int64_t>(bench.out_dim));
            row.set("in_dim", static_cast<int64_t>(bench.in_dim));
            row.set("best_ms", fm::fixed_double(bench.best_ms, 6));
            row.set("gflops", fm::fixed_double(bench.gflops, 4));
            row.set("gbytes_per_s", fm::fixed_double(bench.gbytes_per_s, 4));
            results.push(std::move(row));
        }
    }
    out.set("results", std::move(results));

    // Quantised GEMV: same shapes, the two formats that dominate real
    // checkpoints. Bandwidth is the ceiling here, so report GB/s alongside.
    struct QuantShape { const char *label; uint32_t out_dim; uint32_t in_dim; uint32_t type_id; };
    static const QuantShape quant_shapes[] = {
        {"qkv_1280x1280_q8_0", 1280, 1280, 8},
        {"qkv_1280x1280_q4_k", 1280, 1280, 12},
        {"expert_gate_1792x1280_q4_k", 1792, 1280, 12},
        {"expert_down_1280x1792_q4_k", 1280, 1792, 12},
        {"output_head_32000x1280_q4_k", 32000, 1280, 12},
    };
    Json quant_results = Json::array();
    for (const auto &shape : quant_shapes) {
        for (const auto &candidate : candidates) {
            fm::KernelConfig cfg;
            cfg.requested = candidate.isa;
            cfg.threads = candidate.threads;
            auto bench = fm::benchmark_gemv_quant(cfg, shape.type_id, shape.out_dim,
                                                  shape.in_dim, iterations);
            Json row = Json::object();
            row.set("shape", shape.label);
            row.set("candidate", candidate.label);
            row.set("type", fm::quant_geometry(shape.type_id).name);
            row.set("isa", bench.isa);
            row.set("threads", static_cast<int64_t>(bench.threads));
            row.set("out_dim", static_cast<int64_t>(bench.out_dim));
            row.set("in_dim", static_cast<int64_t>(bench.in_dim));
            row.set("best_ms", fm::fixed_double(bench.best_ms, 6));
            row.set("gflops", fm::fixed_double(bench.gflops, 4));
            row.set("gbytes_per_s", fm::fixed_double(bench.gbytes_per_s, 4));
            quant_results.push(std::move(row));
        }
    }
    out.set("quant_results", std::move(quant_results));
    return out;
}

int command_probe(const Args &args) {
    std::vector<std::string> paths;
    if (args.has("model")) paths.push_back(args.get("model"));
    // positional[0] is the command name itself.
    for (size_t i = 1; i < args.positional.size(); ++i) paths.push_back(args.positional[i]);
    if (paths.empty()) fm::fail(fm::ErrorCode::INVALID_ARGUMENT, "missing model path");

    Json out = Json::object();
    if (paths.size() == 1) {
        auto doc = fm::parse_gguf_file(paths.front());
        out = fm::probe_model(doc, paths.front());
    } else {
        // Several files at once produce a support matrix, which is what you want
        // when triaging a directory of downloaded checkpoints.
        Json models = Json::array();
        uint64_t runnable = 0;
        for (const auto &path : paths) {
            Json row;
            try {
                auto doc = fm::parse_gguf_file(path);
                row = fm::probe_model(doc, path);
            } catch (const fm::Error &error) {
                row = Json::object();
                row.set("schema", "freemoe.model-probe.v1");
                row.set("source", path);
                row.set("runnable", false);
                row.set("verdict", "unreadable");
                Json blocker = Json::object();
                blocker.set("code", fm::error_name(error.code));
                blocker.set("detail", error.what());
                Json blockers = Json::array();
                blockers.push(std::move(blocker));
                row.set("blockers", std::move(blockers));
            }
            if (row.dump().find("\"runnable\":true") != std::string::npos) ++runnable;
            models.push(std::move(row));
        }
        out.set("schema", "freemoe.support-matrix.v1");
        out.set("models", std::move(models));
        out.set("model_count", static_cast<int64_t>(paths.size()));
        out.set("runnable_count", runnable);
    }
    if (args.has("out")) fm::write_text_file(args.get("out"), out.dump() + "\n");
    std::cout << out.dump() << "\n";
    return 0;
}

int command_bench(const Args &args) {
    uint32_t iterations = std::max(1u, args.get_u32("iterations", 5));
    Json out = kernel_bench_json(iterations);
    if (args.has("out")) fm::write_text_file(args.get("out"), out.dump() + "\n");
    std::cout << out.dump() << "\n";
    return 0;
}

int command_compare(const Args &args) {
    fm::RunConfig cfg = build_run_config(args);
    std::filesystem::path run_dir = std::filesystem::path(cfg.results_dir) / cfg.run_id;
    if (cfg.trace_path.empty()) cfg.trace_path = (run_dir / "trace.jsonl").string();
    if (cfg.metrics_path.empty()) cfg.metrics_path = (run_dir / "metrics.json").string();
    if (cfg.report_path.empty()) cfg.report_path = (run_dir / "report.json").string();
    std::filesystem::create_directories(run_dir);

    fm::Engine engine(cfg);
    engine.initialize();
    auto comparison = engine.compare_reference(cfg.reference_root, cfg.prompt_limit,
                                               cfg.generated_tokens);
    engine.release_runtime();
    engine.write_metrics(cfg.metrics_path);
    engine.write_report(cfg.report_path, &comparison);

    bool pass = comparison.prompts > 0 &&
                comparison.prompts_within_tolerance == comparison.prompts &&
                comparison.prompts_token_match == comparison.prompts &&
                (comparison.model_kind != "moe" || comparison.prompts_routing_match == comparison.prompts);
    Json out = json_result(pass ? "pass" : "fail");
    out.set("run_id", cfg.run_id);
    out.set("model", cfg.model_path);
    out.set("backend", engine.backend().name());
    out.set("reader", fm::reader_mode_name(cfg.reader_mode));
    out.set("prompts", static_cast<int64_t>(comparison.prompts));
    out.set("prompts_within_tolerance", static_cast<int64_t>(comparison.prompts_within_tolerance));
    out.set("prompts_token_match", static_cast<int64_t>(comparison.prompts_token_match));
    out.set("prompts_routing_match", static_cast<int64_t>(comparison.prompts_routing_match));
    out.set("max_abs_diff", fm::fixed_double(comparison.max_abs_diff, 9));
    out.set("max_rel_diff", fm::fixed_double(comparison.max_rel_diff, 9));
    out.set("report_path", cfg.report_path);
    out.set("metrics_path", cfg.metrics_path);
    out.set("trace_path", cfg.trace_path);
    std::cout << out.dump() << "\n";
    return pass ? 0 : 22;
}

} // namespace

int main(int argc, char **argv) {
    Args args = parse_args(argc, argv);
    if (args.positional.empty()) {
        print_usage();
        return 2;
    }
    const std::string &command = args.positional[0];
    try {
        if (command == "inspect" || command == "WAKE_UP") return command_inspect(args);
        if (command == "device-info") return command_device_info(args);
        if (command == "run") return command_run(args);
        if (command == "compare") return command_compare(args);
        if (command == "bench") return command_bench(args);
        if (command == "probe" || command == "support") return command_probe(args);
        if (command == "version") {
            std::cout << "FreeMoe 0.1.0 correctness prototype\n";
            return 0;
        }
        if (command == "help" || command == "--help" || command == "-h") {
            print_usage();
            return 0;
        }
        fm::fail(fm::ErrorCode::INVALID_ARGUMENT, "unknown command: " + command);
    } catch (const fm::Error &e) {
        Json error = Json::object();
        error.set("code", fm::error_name(e.code));
        error.set("message", e.what());
        error.set("exit_code", static_cast<int64_t>(fm::error_exit_code(e.code)));
        Json out = json_result("error");
        out.set("error", std::move(error));
        std::cerr << out.dump() << "\n";
        return fm::error_exit_code(e.code);
    } catch (const std::exception &e) {
        Json error = Json::object();
        error.set("code", "INTERNAL_ERROR");
        error.set("message", e.what());
        error.set("exit_code", static_cast<int64_t>(fm::error_exit_code(fm::ErrorCode::INTERNAL_ERROR)));
        Json out = json_result("error");
        out.set("error", std::move(error));
        std::cerr << out.dump() << "\n";
        return fm::error_exit_code(fm::ErrorCode::INTERNAL_ERROR);
    }
}
