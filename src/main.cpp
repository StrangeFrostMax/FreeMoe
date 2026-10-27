#include "fm/backend.hpp"
#include "fm/engine.hpp"
#include "fm/gguf.hpp"

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
        "  freemoe inspect <model.gguf> [--json] [--parse-only] [--output path]\n"
        "  freemoe WAKE_UP <model.gguf> [--json]     # alias for inspect\n"
        "  freemoe device-info --backend cpu|vnpu|gpu [--json]\n"
        "  freemoe run --model <model.gguf> [--backend cpu|vnpu|gpu] [options]\n"
        "  freemoe compare --model <model.gguf> [--backend cpu|vnpu|gpu] [options]\n"
        "\n"
        "Common options:\n"
        "  --reader resident|buffered|mmap|direct  --cache-slots N\n"
        "  --replacement lru|clock                --tokens N\n"
        "  --queue-depth N --coalesce-gap N       --t-wait-ms N\n"
        "  --prefetch 0|1 --budget-mb N           --vnpu-isa PROFILE\n"
        "      PROFILE = zen3-auto|auto|avx2_dual|zen3|avx2|avx512|scalar\n"
        "  --allow-cpu-fallback                    --force-direct-fallback\n"
        "  --test-simulate-zen3 (device-info only)# force Zen 3 fallback diagnostics\n"
        "  --trace path --metrics path --report path --run-id ID --results-dir path\n"
        "  --reference-root path --limit N --abs-tol X --rel-tol X\n"
        "  --prompt TEXT --prompt-file PATH --dump-logits PATH\n";
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
    cfg.reader_mode = fm::parse_reader_mode(args.get("reader", "buffered"));
    cfg.cache_slots = args.get_u32("cache-slots", 0);
    cfg.queue_depth = std::max(1u, args.get_u32("queue-depth", 2));
    cfg.coalesce_gap = args.get_u64("coalesce-gap", 4096);
    cfg.t_wait_ms = args.get_u32("t-wait-ms", 100);
    cfg.generated_tokens = args.get_u32("tokens", fm::DEFAULT_N_GREEDY);
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

Json json_result(const std::string &status) {
    Json j = Json::object();
    j.set("status", status);
    return j;
}

int command_inspect(const Args &args) {
    std::string path = model_path(args);
    auto doc = fm::parse_gguf_file(path);
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
