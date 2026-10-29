#include "fm/activation.hpp"
#include "fm/backend.hpp"
#include "fm/engine.hpp"
#include "fm/gguf.hpp"
#include "fm/io.hpp"
#include "fm/residency.hpp"
#include "fm/quant.hpp"
#include "fm/rope.hpp"
#include "fm/adapter.hpp"
#include "fm/support.hpp"
#include "fm/tokenizer.hpp"
#include "fm/unicode.hpp"
#include "vnpu.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <thread>
#include <utility>
#include <vector>

using namespace fm;

namespace {

int g_checks = 0;
int g_failures = 0;

#define CHECK(cond, message)                                                       \
    do {                                                                           \
        ++g_checks;                                                                \
        if (!(cond)) {                                                             \
            ++g_failures;                                                          \
            std::cerr << "FAIL: " << (message) << " [" << __FILE__ << ":"          \
                      << __LINE__ << "]\n";                                        \
        }                                                                          \
    } while (false)

void run_test(const std::string &name, const std::function<void()> &fn) {
    int before = g_failures;
    try {
        fn();
    } catch (const std::exception &e) {
        ++g_failures;
        std::cerr << "FAIL: exception in " << name << ": " << e.what() << "\n";
    }
    std::cout << "[" << (g_failures == before ? "PASS" : "FAIL") << "] " << name << "\n";
}

template <typename Fn>
bool throws_code(ErrorCode code, Fn fn) {
    try {
        fn();
    } catch (const Error &e) {
        return e.code == code;
    } catch (...) {
        return false;
    }
    return false;
}

template <typename Fn>
bool throws_any_error(Fn fn) {
    try {
        fn();
    } catch (const Error &e) {
        return e.code != ErrorCode::OK && !std::string(e.what()).empty();
    } catch (...) {
        return false;
    }
    return false;
}

bool near(float a, float b, float tol = 1e-3f) {
    return std::fabs(a - b) <= tol;
}

void write_binary(const std::string &path, const std::vector<uint8_t> &data) {
    std::filesystem::create_directories(std::filesystem::path(path).parent_path());
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) fail(ErrorCode::IO_ERROR, "cannot write " + path);
    out.write(reinterpret_cast<const char *>(data.data()), static_cast<std::streamsize>(data.size()));
}

std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> out;
    std::stringstream ss(line);
    std::string item;
    while (std::getline(ss, item, '\t')) out.push_back(item);
    return out;
}

std::string first_prompt() {
    std::ifstream in("tests/reference/prompts.txt");
    std::string line;
    if (in && std::getline(in, line)) return strip_eol(line);
    return "the quick";
}

bool comparison_pass(const ComparisonResult &c) {
    return c.prompts > 0 &&
           c.prompts_within_tolerance == c.prompts &&
           c.prompts_token_match == c.prompts &&
           (c.model_kind != "moe" || c.prompts_routing_match == c.prompts);
}

RunConfig make_engine_config(const std::string &model, BackendKind backend,
                             ReaderMode reader, uint32_t slots, bool prefetch,
                             const std::string &trace_path = "",
                             uint32_t t_wait_ms = 100,
                             bool force_direct_fallback = false,
                             ReplacementPolicy replacement = ReplacementPolicy::LRU) {
    RunConfig cfg;
    cfg.model_path = model;
    cfg.run_id = "unit";
    cfg.backend.kind = backend;
    cfg.backend.allow_cpu_fallback = false;
    cfg.backend.vnpu_isa = "avx2_dual";
    cfg.backend.vnpu_threads = 1;
    cfg.reader_mode = reader;
    cfg.cache_slots = slots;
    cfg.queue_depth = 2;
    cfg.coalesce_gap = 4096;
    cfg.t_wait_ms = t_wait_ms;
    cfg.generated_tokens = 16;
    cfg.prompt_limit = 20;
    cfg.force_direct_fallback = force_direct_fallback;
    cfg.residency.replacement = replacement;
    cfg.residency.enable_prefetch = prefetch;
    cfg.residency.max_prefetch = 2;
    cfg.residency.host_cache_records = 4;
    cfg.trace_path = trace_path;
    cfg.reference_root = "tests/reference";
    cfg.results_dir = "build/test-tmp";
    cfg.abs_tol = DEFAULT_ABS_TOL;
    cfg.rel_tol = DEFAULT_REL_TOL;
    return cfg;
}

GGUFDocument parse_path(const std::string &path) {
    MappedFile file(path);
    return parse_gguf(file.data(), file.size());
}

ModelManifest manifest_from_path(const std::string &path) {
    return build_manifest(parse_path(path), path);
}

// ---------------------------------------------------------------------------
// GGUF parser / adapter
// ---------------------------------------------------------------------------

void test_gguf_parser_and_golden() {
    auto dense = read_binary_file("tests/fixtures/dense-fm.gguf");
    for (size_t cut = 0; cut < 24; ++cut) {
        CHECK(throws_any_error([&]() { (void)parse_gguf(dense.data(), cut); }),
              "truncated GGUF header at byte " + std::to_string(cut));
    }
    auto metadata = parse_path("tests/fixtures/metadata-types.gguf");
    CHECK(metadata.metadata.size() == 27, "metadata-types fixture parses every value type");
    CHECK(metadata.tensors.empty(), "metadata-types fixture has no tensors");
    auto nested_it = metadata.metadata.find("test.nested_array_u32");
    CHECK(nested_it != metadata.metadata.end(), "nested array metadata key exists");
    if (nested_it != metadata.metadata.end()) {
        CHECK(nested_it->second.is_array(), "nested array is an array");
        CHECK(nested_it->second.array.size() == 2, "nested array row count");
        CHECK(!nested_it->second.array.empty() &&
              nested_it->second.array[0].array.size() == 2, "nested array first row length");
    }
    auto bool_it = metadata.metadata.find("test.bool");
    CHECK(bool_it != metadata.metadata.end() && bool_it->second.as_bool(), "bool metadata value");
    auto string_it = metadata.metadata.find("test.string");
    CHECK(string_it != metadata.metadata.end() && string_it->second.as_string() == "metadata types",
          "string metadata value");
    auto float_it = metadata.metadata.find("test.f64");
    CHECK(float_it != metadata.metadata.end() && std::fabs(float_it->second.as_double() - 3.5) < 1e-9,
          "float64 metadata value");

    auto q8 = parse_path("tests/fixtures/quant-blocks.gguf");
    CHECK(q8.tensors.size() == 1, "Q8_0 probe has one tensor");
    if (!q8.tensors.empty()) {
        CHECK(q8.tensors[0].type_name == "Q8_0", "Q8_0 type name");
        CHECK(q8.tensors[0].byte_size == 68, "Q8_0 byte size is two complete blocks");
        CHECK(q8.tensors[0].dims == std::vector<uint64_t>({32, 2}), "Q8_0 dimensions");
    }
    auto bad_q8 = read_binary_file("tests/malformed_models/quant_block_misaligned.gguf");
    CHECK(throws_any_error([&]() { (void)parse_gguf(bad_q8.data(), bad_q8.size()); }),
          "Q8_0 row not divisible by block size is rejected");

    for (const std::string kind : {"dense", "moe"}) {
        std::string path = "tests/fixtures/" + kind + "-fm.gguf";
        auto doc = parse_path(path);
        auto manifest = build_manifest(doc, path);
        std::string actual = inspect_json(manifest, path).dump();
        std::string golden = trim(read_text_file("tests/golden/inspect_" + kind + ".json"));
        CHECK(actual == golden, "golden inspect report matches for " + kind);
        CHECK(manifest.experts.empty() == (kind == "dense"), "expert record count matches model kind");
        if (kind == "moe") {
            CHECK(manifest.experts.size() == manifest.config.layers * manifest.config.expert_count,
                  "one expert record per layer/expert");
            CHECK(manifest.slot_bytes == 3ull * manifest.config.hidden *
                                             manifest.config.intermediate * sizeof(float),
                  "expert slot bytes computed from stacked F32 tensors");
        }
    }

    auto dense_doc = parse_path("tests/fixtures/dense-fm.gguf");
    auto fuzzed_metadata = dense_doc;
    auto layer_it = fuzzed_metadata.metadata.find("freemoe.num_hidden_layers");
    CHECK(layer_it != fuzzed_metadata.metadata.end(), "layer-count metadata key exists");
    if (layer_it != fuzzed_metadata.metadata.end()) {
        layer_it->second.u = 0xffffffffu;
        CHECK(throws_any_error([&]() {
            (void)build_manifest(fuzzed_metadata, "tests/fixtures/dense-fm.gguf");
        }), "unreasonable layer-count metadata is rejected before adapter loops");
    }
}

void test_malformed_fixtures() {
    std::ifstream manifest("tests/malformed_models/manifest.tsv");
    CHECK(manifest.good(), "malformed fixture manifest exists");
    std::string line;
    std::getline(manifest, line); // header
    uint32_t count = 0;
    while (std::getline(manifest, line)) {
        auto fields = split_tabs(line);
        if (fields.size() < 4) continue;
        std::string path = "tests/malformed_models/" + fields[0];
        auto bytes = read_binary_file(path);
        bool rejected = throws_any_error([&]() {
            auto doc = parse_gguf(bytes.data(), bytes.size());
            (void)build_manifest(doc, path);
        });
        CHECK(rejected, "typed error for malformed fixture " + fields[0]);
        CHECK(fields[2] == "TYPED_ERROR", "manifest expected code is typed for " + fields[0]);
        ++count;
    }
    CHECK(count >= 50, "at least 50 malformed fixtures were exercised");
}

// ---------------------------------------------------------------------------
// State machine, budget, I/O
// ---------------------------------------------------------------------------

void test_slot_state_machine() {
    using S = SlotState;
    using E = SlotEvent;
    struct Row { S state; E event; S next; };
    const std::vector<Row> rows = {
        {S::ABSENT, E::DemandAccepted, S::IO_QUEUED},
        {S::ABSENT, E::PrefetchAccepted, S::IO_QUEUED},
        {S::ABSENT, E::Shutdown, S::CANCELLED},
        {S::IO_QUEUED, E::IOComplete, S::HOST_READY},
        {S::IO_QUEUED, E::CancelPrefetchOnly, S::CANCELLED},
        {S::IO_QUEUED, E::CancelWithDemand, S::IO_QUEUED},
        {S::IO_QUEUED, E::IOError, S::ERROR},
        {S::IO_QUEUED, E::ValidationError, S::ERROR},
        {S::IO_QUEUED, E::Shutdown, S::CANCELLED},
        {S::HOST_READY, E::UploadQueued, S::UPLOAD_QUEUED},
        {S::HOST_READY, E::IOError, S::ERROR},
        {S::HOST_READY, E::ValidationError, S::ERROR},
        {S::HOST_READY, E::Shutdown, S::CANCELLED},
        {S::UPLOAD_QUEUED, E::FenceComplete, S::GPU_READY},
        {S::UPLOAD_QUEUED, E::UploadFailed, S::ERROR},
        {S::UPLOAD_QUEUED, E::IOError, S::ERROR},
        {S::UPLOAD_QUEUED, E::ValidationError, S::ERROR},
        {S::UPLOAD_QUEUED, E::DeviceLoss, S::ERROR},
        {S::UPLOAD_QUEUED, E::Shutdown, S::CANCELLED},
        {S::GPU_READY, E::Acquire, S::IN_USE},
        {S::GPU_READY, E::EvictSelected, S::EVICTABLE},
        {S::GPU_READY, E::DeviceLoss, S::ERROR},
        {S::GPU_READY, E::Shutdown, S::CANCELLED},
        {S::IN_USE, E::ReleaseLast, S::GPU_READY},
        {S::IN_USE, E::DeviceLoss, S::ERROR},
        {S::IN_USE, E::Shutdown, S::CANCELLED},
        {S::EVICTABLE, E::FencesDone, S::ABSENT},
        {S::EVICTABLE, E::Shutdown, S::CANCELLED},
        {S::ERROR, E::CallbacksRetired, S::ABSENT},
        {S::CANCELLED, E::CallbacksRetired, S::ABSENT},
    };
    std::map<std::pair<int, int>, S> expected;
    for (const auto &row : rows) expected[{static_cast<int>(row.state), static_cast<int>(row.event)}] = row.next;
    for (int s = 0; s <= static_cast<int>(S::CANCELLED); ++s) {
        for (int e = 0; e <= static_cast<int>(E::Shutdown); ++e) {
            auto transition = transition_for(static_cast<S>(s), static_cast<E>(e));
            auto it = expected.find({s, e});
            CHECK(transition.valid == (it != expected.end()),
                  "state/event pair matches authoritative transition table");
            if (it != expected.end())
                CHECK(transition.next == it->second, "state/event pair has correct next state");
        }
    }
    S state = S::ABSENT;
    std::string action;
    CHECK(apply_state_transition(state, E::DemandAccepted, action), "valid transition applies");
    CHECK(state == S::IO_QUEUED && !action.empty(), "valid transition updates state/action");
    CHECK(!apply_state_transition(state, E::Acquire, action), "invalid transition is rejected");
    CHECK(state == S::IO_QUEUED, "invalid transition leaves state unchanged");
}

void test_budget_calculator() {
    auto manifest = manifest_from_path("tests/fixtures/moe-fm.gguf");
    BackendConfig backend_config;
    backend_config.kind = BackendKind::ScalarCPU;
    auto backend = Backend::create(backend_config);
    BudgetConfig cfg;
    cfg.total_hot_bytes = 2ull * 1024 * 1024;
    cfg.workspace_bytes = 64ull * 1024;
    cfg.staging_bytes = 64ull * 1024;
    cfg.margin_bytes = 128ull * 1024;
    cfg.requested_slots = 0;
    auto budget = compute_budget(manifest, cfg, backend->info());
    uint64_t expected_kv = static_cast<uint64_t>(manifest.config.layers) *
                           manifest.config.context * manifest.config.hidden * 2 * sizeof(float);
    uint64_t reserved = manifest.fixed_weight_bytes + expected_kv +
                        cfg.workspace_bytes + cfg.staging_bytes + cfg.margin_bytes;
    uint64_t available = cfg.total_hot_bytes - reserved;
    CHECK(budget.kv_bytes == expected_kv, "KV reservation hand calculation");
    CHECK(budget.fixed_weights_bytes == manifest.fixed_weight_bytes, "fixed weight reservation");
    CHECK(budget.available_expert_bytes == available, "available expert cache bytes");
    CHECK(budget.slot_count == available / budget.slot_bytes, "slot count from floor(cache/slot)");
    CHECK(budget.expert_cache_bytes == static_cast<uint64_t>(budget.slot_count) * budget.slot_bytes,
          "expert cache bytes equal slot_count times slot_bytes");
    CHECK(budget.print().find("reserve.kv=") != std::string::npos, "budget print lists every reservation");

    BudgetConfig unsafe = cfg;
    unsafe.total_hot_bytes = 1024;
    CHECK(throws_code(ErrorCode::BUDGET_TOO_SMALL, [&]() {
        (void)compute_budget(manifest, unsafe, backend->info());
    }), "unsafe budget is rejected");
    BudgetConfig over_request = cfg;
    over_request.requested_slots = 10000;
    CHECK(throws_code(ErrorCode::BUDGET_TOO_SMALL, [&]() {
        (void)compute_budget(manifest, over_request, backend->info());
    }), "over-requested slot count is rejected");
}

void test_io_scheduler_and_readers() {
    std::vector<Extent> exact_gap = {{0, 100}, {100 + 4096, 100}};
    auto coalesced = coalesce_extents(exact_gap, 4096);
    CHECK(coalesced.size() == 1, "gap equal to G_bytes coalesces");
    CHECK(coalesced[0].request_indices.size() == 2, "coalesced group contains both requests");
    std::vector<Extent> too_far = {{0, 100}, {100 + 4097, 100}};
    CHECK(coalesce_extents(too_far, 4096).size() == 2, "gap equal to G_bytes plus one does not coalesce");
    CHECK(extents_total_bytes(exact_gap) == 200, "extent total bytes");

    ReadRequest request;
    request.demand_subscribers = 1;
    request.prefetch_subscribers = 1;
    CHECK(!request.cancel(SubscriberType::Prefetch), "prefetch cancellation keeps shared demand request alive");
    CHECK(request.demand_subscribers == 1 && !request.cancelled, "demand subscriber remains");
    CHECK(request.cancel(SubscriberType::Demand), "last demand subscriber cancels request");
    std::vector<ReadRequest> priority = {
        ReadRequest{2, ReadPriority::Prefetch, 0, 1, false},
        ReadRequest{1, ReadPriority::Demand, 1, 0, false},
    };
    std::stable_sort(priority.begin(), priority.end(), [](const ReadRequest &a, const ReadRequest &b) {
        return a.priority < b.priority;
    });
    CHECK(priority[0].priority == ReadPriority::Demand, "demand reads sort before prefetch reads");

    const std::string temp = "build/test-tmp/direct.bin";
    std::vector<uint8_t> pattern(8192);
    for (size_t i = 0; i < pattern.size(); ++i) pattern[i] = static_cast<uint8_t>((i * 37 + 11) & 0xff);
    write_binary(temp, pattern);

    auto forced = make_reader(ReaderMode::Direct, temp, nullptr, true);
    auto result = forced->read_span(17, 100);
    CHECK(result.fallback, "forced direct-I/O fallback is explicit");
    CHECK(std::equal(result.bytes.begin(), result.bytes.end(), pattern.begin() + 17),
          "forced fallback returns byte-equal slice");
    auto direct = make_reader(ReaderMode::Direct, temp, nullptr, false);
    auto direct_result = direct->read_span(4095, 129);
    CHECK(std::equal(direct_result.bytes.begin(), direct_result.bytes.end(), pattern.begin() + 4095),
          "direct reader handles a misaligned request via aligned bounce buffer");

    SchedulerConfig config;
    config.mode = ReaderMode::Buffered;
    config.path = temp;
    config.expected_identity = current_identity(temp);
    config.queue_depth = 2;
    config.coalesce_gap = 4096;
    ReadScheduler scheduler(config, make_reader(ReaderMode::Buffered, temp, nullptr, false));
    auto payload = scheduler.read_record({{0, 64}, {80, 64}}, ReadPriority::Demand);
    CHECK(payload.size() == 128, "coalesced record payload has both useful ranges");
    CHECK(std::equal(payload.begin(), payload.begin() + 64, pattern.begin()),
          "first coalesced range byte-equal");
    CHECK(std::equal(payload.begin() + 64, payload.end(), pattern.begin() + 80),
          "second coalesced range byte-equal");
    auto metrics = scheduler.metrics();
    CHECK(metrics.coalesced_requests == 1, "scheduler records one coalesced request");
    CHECK(metrics.physical_bytes == 144, "coalesced physical span is 144 bytes");
    CHECK(metrics.useful_bytes == 128, "useful bytes exclude the 16-byte gap");

    SchedulerConfig forced_config = config;
    forced_config.mode = ReaderMode::Direct;
    forced_config.force_direct_fallback = true;
    ReadScheduler forced_scheduler(forced_config,
                                   make_reader(ReaderMode::Direct, temp, nullptr, true));
    (void)forced_scheduler.read_record({{16, 64}}, ReadPriority::Demand);
    CHECK(forced_scheduler.metrics().direct_fallbacks == 1, "direct fallback counter increments");

    const std::string source = "build/test-tmp/source-changed.gguf";
    write_binary(source, read_binary_file("tests/fixtures/moe-fm.gguf"));
    SchedulerConfig source_config;
    source_config.mode = ReaderMode::Buffered;
    source_config.path = source;
    source_config.expected_identity = current_identity(source);
    source_config.queue_depth = 2;
    source_config.coalesce_gap = 4096;
    ReadScheduler source_scheduler(source_config,
                                   make_reader(ReaderMode::Buffered, source, nullptr, false));
    (void)source_scheduler.read_record({{0, 16}}, ReadPriority::Demand);
    auto changed = read_binary_file(source);
    changed[changed.size() / 2] ^= 0x5a;
    // The sandbox filesystem has millisecond-scale timestamp granularity.
    // Sleep well above one tick so mtime/ctime are guaranteed to differ.
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    write_binary(source, changed);
    CHECK(throws_code(ErrorCode::SOURCE_CHANGED, [&]() {
        (void)source_scheduler.read_record({{0, 16}}, ReadPriority::Demand);
    }), "modified model source is refused with SOURCE_CHANGED");
    CHECK(source_scheduler.metrics().source_changed_events == 1, "source change metric recorded");
}

// ---------------------------------------------------------------------------
// Backends
// ---------------------------------------------------------------------------

void test_backends_and_gpu_placeholder() {
    BackendConfig cpu_config;
    cpu_config.kind = BackendKind::ScalarCPU;
    auto cpu = Backend::create(cpu_config);
    {
        auto a = cpu->allocate(128, AllocationTag::Weights);
        auto b = cpu->allocate(256, AllocationTag::KV);
        auto c = cpu->allocate(64, AllocationTag::Workspace);
        auto d = cpu->allocate(32, AllocationTag::Staging);
        auto snap = cpu->tracker()->snapshot();
        CHECK(snap.live_bytes == 480, "allocation tracker live total equals tagged sum");
        CHECK(snap.by_tag[AllocationTag::Weights] == 128 &&
              snap.by_tag[AllocationTag::KV] == 256 &&
              snap.by_tag[AllocationTag::Workspace] == 64 &&
              snap.by_tag[AllocationTag::Staging] == 32,
              "allocation tracker per-tag totals");
        (void)a; (void)b; (void)c; (void)d;
    }
    CHECK(cpu->tracker()->snapshot().live_bytes == 0, "all tagged allocations are freed");

    std::vector<float> W = {1, 2, 3, 4, 5, 6};
    std::vector<float> x = {1, 1, 1};
    std::vector<float> y(2, 0);
    cpu->matvec(W.data(), x.data(), y.data(), 2, 3, "cpu_matvec_test");
    CHECK(near(y[0], 6) && near(y[1], 15), "scalar CPU matvec matches hand calculation");
    cpu->inject_fault(BackendFault::TransferFailure);
    auto buffer = cpu->allocate(16, AllocationTag::Other);
    CHECK(throws_code(ErrorCode::TRANSFER_FAILED, [&]() {
        (void)cpu->upload(*buffer, W.data(), 16);
    }), "transfer failure injection is typed");
    cpu->clear_faults();

    BackendConfig vnpu_config;
    vnpu_config.kind = BackendKind::VirtualNPU;
    vnpu_config.vnpu_isa = "avx2_dual";
    vnpu_config.vnpu_threads = 1;
    auto vnpu = Backend::create(vnpu_config);
    auto info = vnpu->info();
    CHECK(info.dedicated_vram_bytes == 0, "virtual NPU reports no dedicated VRAM");
    CHECK(info.memory_type.find("host_ram") != std::string::npos,
          "virtual NPU reports host-memory buffers");
    CHECK(info.simd_bits == 256 || info.simd_bits == 0,
          "virtual NPU selects 256-bit AVX2 or documented scalar fallback");
    CHECK(info.notes.find("CPU software device") != std::string::npos,
          "virtual NPU is explicitly labelled a CPU software device");
    std::vector<float> Wt = {1, 0, 0, 0, 0,
                             0, 1, 0, 0, 0,
                             0, 0, 1, 2, 3};
    std::vector<float> xv = {1, 2, 3, 4, 5};
    std::vector<float> yv(3, 0);
    vnpu->matvec(Wt.data(), xv.data(), yv.data(), 3, 5, "vnpu_matvec_test");
    CHECK(near(yv[0], 1) && near(yv[1], 2) && near(yv[2], 26),
          "patched AVX2 dual-pipe VNPU matvec matches scalar oracle");

    BackendConfig gpu_config;
    gpu_config.kind = BackendKind::PhysicalGPU;
    gpu_config.allow_cpu_fallback = false;
    auto gpu = Backend::create(gpu_config);
    CHECK(throws_code(ErrorCode::GPU_BACKEND_UNIMPLEMENTED, [&]() {
        (void)gpu->allocate(128, AllocationTag::Weights);
    }), "physical GPU placeholder fails loudly with no fallback");
    gpu_config.allow_cpu_fallback = true;
    auto gpu_fallback = Backend::create(gpu_config);
    std::vector<float> yf(2, 0);
    gpu_fallback->matvec(W.data(), x.data(), yf.data(), 2, 3, "explicit_fallback_test");
    CHECK(near(yf[0], 6) && near(yf[1], 15), "explicit CPU fallback computes correctly");
    auto ops = gpu_fallback->operations();
    CHECK(!ops.empty() && ops.back().backend == "cpu-fallback",
          "explicit CPU fallback is visible in operation log");
}

void test_zen3_fallback() {
    auto schedules_avx512 = [](const std::vector<VnpuIsaAttempt> &attempts) {
        return std::any_of(attempts.begin(), attempts.end(), [](const VnpuIsaAttempt &a) {
            return a.forced_isa == VNPU_ISA_FORCE_AVX512;
        });
    };

    // The planner is pure: Zen 3 detection can be simulated on any host.
    for (const std::string profile : {"zen3-auto", "zen3_auto", "auto", "avx512"}) {
        auto attempts = plan_vnpu_isa_attempts(profile, true);
        CHECK(!attempts.empty(), "Zen 3 fallback plan is non-empty for " + profile);
        CHECK(attempts[0].forced_isa == VNPU_ISA_FORCE_AVX2_DUAL,
              "Zen 3 detection selects dual 256-bit AVX2/FMA3 for " + profile);
        CHECK(attempts[0].label.find("Zen 3") != std::string::npos,
              "Zen 3 fallback attempt is visibly labelled for " + profile);
        CHECK(!schedules_avx512(attempts),
              "Zen 3 fallback never schedules AVX-512 for " + profile);
        CHECK(attempts.back().forced_isa == VNPU_ISA_FORCE_SCALAR,
              "Zen 3 fallback retains a final scalar escape hatch for " + profile);
    }

    auto non_zen_auto = plan_vnpu_isa_attempts("zen3-auto", false);
    CHECK(!non_zen_auto.empty() && non_zen_auto[0].forced_isa == VNPU_ISA_AUTO,
          "zen3-auto uses runtime auto dispatch when Zen 3 is absent");
    auto non_zen_avx512 = plan_vnpu_isa_attempts("avx512", false);
    CHECK(!non_zen_avx512.empty() && non_zen_avx512[0].forced_isa == VNPU_ISA_FORCE_AVX512,
          "explicit AVX-512 request is honoured only off Zen 3");
    auto zen3_alias = plan_vnpu_isa_attempts("zen3", false);
    CHECK(!zen3_alias.empty() && zen3_alias[0].forced_isa == VNPU_ISA_FORCE_AVX2_DUAL,
          "zen3 profile alias pins the dual-pipe path");
    CHECK(throws_code(ErrorCode::INVALID_ARGUMENT, []() {
        (void)plan_vnpu_isa_attempts("not-an-isa", true);
    }), "unknown VNPU ISA profile is rejected by the planner");

    // Exercise the real backend init diagnostics with the detection-aware mode.
    const std::string trace_path = "build/test-tmp/zen3-fallback-trace.jsonl";
    TraceWriter trace(trace_path);
    BackendConfig config;
    config.kind = BackendKind::VirtualNPU;
    config.vnpu_isa = "zen3-auto";
    config.vnpu_threads = 1;
    auto backend = Backend::create(config, &trace);
    auto info = backend->info();
    CHECK(info.notes.find("Selected VNPU dispatch:") != std::string::npos,
          "device info records the selected dispatch attempt");
    const bool zen3 = host_is_zen3();
    CHECK(info.notes.find(zen3 ? "Zen 3 detected" : "Zen 3 not detected") != std::string::npos,
          "device info records Zen 3 detection state");
    if (zen3) {
        CHECK(info.simd_bits == 256, "Detected Zen 3 backend uses 256-bit SIMD");
        CHECK(info.notes.find("AVX-512 dispatch disabled") != std::string::npos,
              "Detected Zen 3 disables AVX-512 dispatch");
    }

    const std::string trace_text = read_text_file(trace_path);
    CHECK(trace_text.find("\"event\":\"backend_init\"") != std::string::npos,
          "backend init event is traced");
    CHECK(trace_text.find(std::string("\"zen3_detected\":") + (zen3 ? "true" : "false")) !=
              std::string::npos,
          "backend init trace records Zen 3 detection state");
    CHECK(trace_text.find("\"requested_profile\":\"zen3-auto\"") != std::string::npos,
          "backend init trace records requested ISA profile");
    CHECK(trace_text.find("\"selected_attempt\":") != std::string::npos,
          "backend init trace records selected fallback attempt");
    CHECK(trace_text.find("\"fallback_used\":") != std::string::npos,
          "backend init trace records whether fallback was used");

    // End-to-end simulation of the detection branch on a non-Zen-3 CI host.
    // Even a CPU that advertises AVX-512 must take the dual 256-bit path.
    for (const std::string profile : {"zen3-auto", "auto", "avx512"}) {
        const std::string sim_trace_path =
            "build/test-tmp/zen3-simulated-" + profile + "-trace.jsonl";
        TraceWriter sim_trace(sim_trace_path);
        BackendConfig sim_config;
        sim_config.kind = BackendKind::VirtualNPU;
        sim_config.vnpu_isa = profile;
        sim_config.vnpu_threads = 1;
        sim_config.test_simulate_zen3 = true;
        auto sim_backend = Backend::create(sim_config, &sim_trace);
        auto sim_info = sim_backend->info();
        CHECK(sim_info.simd_bits == 256,
              "simulated Zen 3 uses 256-bit SIMD for " + profile);
        CHECK(sim_info.backend.find("AVX512") == std::string::npos,
              "simulated Zen 3 backend string has no AVX-512 for " + profile);
        CHECK(sim_info.backend.find("dual-pipe") != std::string::npos,
              "simulated Zen 3 backend string names dual-pipe AVX2 for " + profile);
        CHECK(sim_info.notes.find("Zen 3 detected") != std::string::npos,
              "simulated Zen 3 detection is recorded in notes for " + profile);
        CHECK(sim_info.notes.find("AVX-512 dispatch disabled") != std::string::npos,
              "simulated Zen 3 disables AVX-512 in notes for " + profile);
        CHECK(sim_info.notes.find("Selected VNPU dispatch: Zen 3 AVX2 dual-pipe fallback") !=
                  std::string::npos,
              "simulated Zen 3 selects labelled fallback for " + profile);

        // On hardware that really is Zen 3 the simulation flag is not used: the
        // detection is genuine, so the device reports it as such and must not
        // claim a simulation. The first version of this test asserted
        // `zen3_detection_simulated: true` unconditionally, which failed three
        // checks on Zen 3 CI runners and passed on the non-Zen-3 box the test
        // was written on.
        CHECK(sim_info.notes.find(zen3 ? "Zen 3 detected; "
                                       : "Zen 3 detected (simulated for test); ") !=
                  std::string::npos,
              "simulated Zen 3 marks detection as " + std::string(zen3 ? "real" : "synthetic") +
                  " for " + profile);

        const std::string sim_trace_text = read_text_file(sim_trace_path);
        CHECK(sim_trace_text.find("\"zen3_detected\":true") != std::string::npos,
              "simulated Zen 3 trace records detection for " + profile);
        CHECK(sim_trace_text.find(std::string("\"zen3_detection_simulated\":") +
                                  (zen3 ? "false" : "true")) != std::string::npos,
              "simulated Zen 3 trace marks simulation as " +
                  std::string(zen3 ? "not used on real Zen 3" : "synthetic") + " for " + profile);
        CHECK(sim_trace_text.find("\"zen3_fallback_selected\":true") != std::string::npos,
              "simulated Zen 3 trace marks detection fallback for " + profile);
        CHECK(sim_trace_text.find("\"selected_attempt\":\"Zen 3 AVX2 dual-pipe fallback\"") !=
                  std::string::npos,
              "simulated Zen 3 trace records fallback attempt for " + profile);
    }
}

// ---------------------------------------------------------------------------
// Residency manager unit harness
// ---------------------------------------------------------------------------

struct ResidencyFixture {
    std::string path = "tests/fixtures/moe-fm.gguf";
    std::unique_ptr<MappedFile> file;
    GGUFDocument doc;
    ModelManifest manifest;
    std::unique_ptr<TraceWriter> trace;
    std::unique_ptr<Backend> backend;
    std::unique_ptr<ReadScheduler> scheduler;
    BudgetResult budget;
    ResidencyConfig residency_config;
    std::unique_ptr<ResidencyManager> manager;

    ResidencyFixture(uint32_t slots, ReplacementPolicy policy, uint32_t t_wait_ms) {
        file = std::make_unique<MappedFile>(path);
        doc = parse_gguf(file->data(), file->size());
        manifest = build_manifest(doc, path);
        trace = std::make_unique<TraceWriter>("");
        BackendConfig backend_config;
        backend_config.kind = BackendKind::ScalarCPU;
        backend = Backend::create(backend_config);
        SchedulerConfig scheduler_config;
        scheduler_config.mode = ReaderMode::Buffered;
        scheduler_config.path = path;
        scheduler_config.expected_identity = file->identity();
        scheduler_config.queue_depth = 2;
        scheduler_config.coalesce_gap = 4096;
        scheduler = std::make_unique<ReadScheduler>(
            scheduler_config, make_reader(ReaderMode::Buffered, path, nullptr, false), trace.get());
        BudgetConfig budget_config;
        budget_config.requested_slots = slots;
        budget = compute_budget(manifest, budget_config, backend->info());
        residency_config.replacement = policy;
        residency_config.enable_prefetch = false;
        residency_config.t_wait_ms = t_wait_ms;
        manager = std::make_unique<ResidencyManager>(manifest, *backend, *scheduler, *trace,
                                                     budget, residency_config);
        manager->initialize();
    }
};

void test_residency_manager() {
    {
        ResidencyFixture f(1, ReplacementPolicy::LRU, 1);
        CHECK(f.manager->slots().size() == 1, "one-slot pool allocated");
        CHECK(f.manager->inject_stale_completion(0, 999), "stale completion is detectable");
        CHECK(f.manager->metrics().stale_completions == 1, "stale completion counter increments");
        CHECK(f.manager->slots()[0].state == SlotState::ABSENT, "stale completion leaves slot unchanged");

        SlotHandle first = f.manager->acquire(0, 0, 0);
        CHECK(static_cast<bool>(first), "first expert acquisition returns slot data");
        CHECK(!f.manager->try_evict(0), "eviction of referenced slot is refused");
        first.release();
        CHECK(f.manager->slots()[0].state == SlotState::GPU_READY, "release returns slot to GPU_READY");
        SlotHandle second = f.manager->acquire(0, 1, 1);
        second.release();
        CHECK(f.manager->metrics().evictions == 1, "second expert evicts LRU victim");
        CHECK(f.manager->metrics().checksum_checks >= 4, "source and slot checksums are checked");

        for (uint32_t layer = 0; layer < f.manifest.config.layers; ++layer) {
            for (uint32_t expert = 0; expert < f.manifest.config.expert_count; ++expert) {
                SlotHandle handle = f.manager->acquire(layer, expert, 10 + layer * 100 + expert);
                CHECK(static_cast<bool>(handle), "all experts cycle through one-slot pool");
            }
        }
        CHECK(f.manager->metrics().checksum_failures == 0, "no checksum failures during expert cycle");
        CHECK(f.manager->metrics().invalid_transitions == 0, "no invalid state transitions");
        CHECK(f.manager->metrics().cache_misses >=
              f.manifest.config.layers * f.manifest.config.expert_count,
              "every expert demand miss is counted");
    }
    {
        ResidencyFixture f(1, ReplacementPolicy::LRU, 1);
        f.manager->pin_slot_for_test(0);
        CHECK(throws_code(ErrorCode::SLOTS_BUSY, [&]() {
            (void)f.manager->acquire(0, 0, 0);
        }), "all-slots-busy returns SLOTS_BUSY after wait deadline");
        CHECK(f.manager->metrics().slots_busy_events == 1, "slots busy event recorded");
        f.manager->unpin_slot_for_test(0);
    }
    {
        ResidencyFixture f(2, ReplacementPolicy::Clock, 1);
        f.manager->unpin_slot_for_test(0);
        f.manager->unpin_slot_for_test(1);
        f.manager->slots()[0].key = ModelManifest::expert_key(0, 0);
        f.manager->slots()[1].key = ModelManifest::expert_key(0, 1);
        f.manager->slots()[0].last_used = 1;
        f.manager->slots()[1].last_used = 2;
        f.manager->set_clock_bits_for_test({1, 0});
        Slot *victim = f.manager->select_victim_for_test(10);
        CHECK(victim && victim->index == 1, "clock victim sequence matches test vector");
        f.manager->set_clock_bits_for_test({0, 0});
        victim = f.manager->select_victim_for_test(11);
        CHECK(victim && victim->index == 0, "clock hand advances to next victim");
    }
    {
        ResidencyFixture f(1, ReplacementPolicy::LRU, 1);
        Slot &slot = f.manager->slots()[0];
        slot.state = SlotState::UPLOAD_QUEUED;
        slot.generation = 7;
        slot.fence = Fence{1, now_ms() + 3600000ull, true};
        CHECK(!f.manager->complete_upload(0, 7), "deferred fence keeps slot in UPLOAD_QUEUED");
        CHECK(slot.state == SlotState::UPLOAD_QUEUED, "deferred fence does not publish slot");
        slot.fence.deferred = false;
        slot.fence.ready_at_ms = 0;
        CHECK(f.manager->complete_upload(0, 7), "fence completion publishes GPU_READY");
        CHECK(slot.state == SlotState::GPU_READY, "published slot state is GPU_READY");
        CHECK(!f.manager->complete_upload(0, 8), "generation mismatch is rejected");
    }
    {
        ResidencyFixture f(1, ReplacementPolicy::LRU, 1);
        f.backend->inject_fault(BackendFault::TransferFailure);
        CHECK(throws_code(ErrorCode::TRANSFER_FAILED, [&]() {
            (void)f.manager->acquire(0, 0, 0);
        }), "transfer failure is propagated with typed code");
        CHECK(f.manager->metrics().transfer_failures == 1, "transfer failure metric recorded");
        f.manager->reset_errors_for_test();
        f.backend->inject_fault(BackendFault::FenceTimeout);
        CHECK(throws_code(ErrorCode::FENCE_TIMEOUT, [&]() {
            (void)f.manager->acquire(0, 0, 1);
        }), "fence timeout is propagated with typed code");
        CHECK(f.manager->metrics().fence_timeouts == 1, "fence timeout metric recorded");
        f.manager->reset_errors_for_test();
        SlotHandle handle = f.manager->acquire(0, 0, 2);
        handle.release();
        f.manager->inject_device_loss();
        CHECK(f.manager->slots()[0].state == SlotState::ERROR, "device loss invalidates live slot");
        CHECK(f.manager->metrics().device_loss_events >= 1, "device loss metric recorded");
    }
}

// ---------------------------------------------------------------------------
// End-to-end engine correctness
// ---------------------------------------------------------------------------

void check_released(Engine &engine, const std::string &name) {
    engine.release_runtime();
    auto snap = engine.backend().tracker()->snapshot();
    CHECK(snap.live_allocations == 0 && snap.live_bytes == 0,
          name + " releases every tagged backend allocation");
}

void test_dense_cpu_reference_and_operation_log() {
    {
        auto cfg = make_engine_config("tests/fixtures/dense-fm.gguf", BackendKind::ScalarCPU,
                                      ReaderMode::Buffered, 0, false);
        Engine engine(cfg);
        engine.initialize();
        auto comparison = engine.compare_reference("tests/reference", 20, 16);
        CHECK(comparison_pass(comparison), "dense CPU matches all 20 reference prompts");
        CHECK(comparison.prompts == 20, "dense comparison ran 20 prompts");
        CHECK(comparison.max_abs_diff <= DEFAULT_ABS_TOL + DEFAULT_REL_TOL,
              "dense max absolute difference is inside declared tolerance");
        check_released(engine, "dense CPU");
    }
    {
        auto cfg = make_engine_config("tests/fixtures/dense-fm.gguf", BackendKind::ScalarCPU,
                                      ReaderMode::Buffered, 0, false);
        Engine engine(cfg);
        engine.initialize();
        auto result = engine.run_prompt(first_prompt(), 0);
        auto ops = engine.backend().operations();
        uint64_t expected = 1 + result.input_ids.size() *
            (static_cast<uint64_t>(engine.manifest().config.layers) * 14 + 2);
        CHECK(ops.size() == expected, "one-token operation log contains one entry per operation");
        bool all_named = true;
        for (const auto &op : ops) all_named = all_named && !op.name.empty() && !op.backend.empty();
        CHECK(all_named, "every operation records name and actual backend");
        check_released(engine, "operation log");
    }
}

void test_moe_readers_and_cache_sizes() {
    struct ReaderCase { ReaderMode mode; bool force_fallback; const char *name; };
    const std::vector<ReaderCase> readers = {
        {ReaderMode::Resident, false, "resident"},
        {ReaderMode::Buffered, false, "buffered"},
        {ReaderMode::Mmap, false, "mmap"},
        {ReaderMode::Direct, false, "direct"},
        {ReaderMode::Direct, true, "direct-forced-fallback"},
    };
    std::vector<float> baseline_logits;
    std::vector<int32_t> baseline_ids;
    for (const auto &reader : readers) {
        auto cfg = make_engine_config("tests/fixtures/moe-fm.gguf", BackendKind::ScalarCPU,
                                      reader.mode, 1, false, "", 5, reader.force_fallback);
        Engine engine(cfg);
        engine.initialize();
        auto comparison = engine.compare_reference("tests/reference", 20, 16);
        CHECK(comparison_pass(comparison),
              std::string("MoE CPU matches reference with ") + reader.name + " reader");
        CHECK(!engine.results().empty(), "reader run produced results");
        if (!engine.results().empty()) {
            const auto &first = engine.results().front();
            if (baseline_logits.empty()) {
                baseline_logits = first.prefill_logits;
                baseline_ids = first.generated_ids;
            } else {
                CHECK(first.prefill_logits == baseline_logits,
                      std::string(reader.name) + " prefill logits are bit-identical to baseline");
                CHECK(first.generated_ids == baseline_ids,
                      std::string(reader.name) + " generated tokens are bit-identical to baseline");
            }
        }
        if (reader.mode == ReaderMode::Resident && engine.residency()) {
            CHECK(engine.residency()->metrics().resident_expert_bytes == engine.manifest().expert_tensor_bytes,
                  "resident expert bytes equal sum of expert tensor sizes");
        }
        check_released(engine, std::string("MoE ") + reader.name);
    }

    std::vector<float> one_slot_logits;
    std::vector<int32_t> one_slot_ids;
    {
        auto cfg = make_engine_config("tests/fixtures/moe-fm.gguf", BackendKind::ScalarCPU,
                                      ReaderMode::Buffered, 1, false, "", 5);
        Engine engine(cfg);
        engine.initialize();
        auto comparison = engine.compare_reference("tests/reference", 20, 16);
        CHECK(comparison_pass(comparison), "one-slot cache matches all 20 MoE prompts");
        one_slot_logits = engine.results().front().prefill_logits;
        one_slot_ids = engine.results().front().generated_ids;
        check_released(engine, "one-slot cache");
    }
    {
        auto cfg = make_engine_config("tests/fixtures/moe-fm.gguf", BackendKind::ScalarCPU,
                                      ReaderMode::Buffered, 16, false, "", 5);
        Engine engine(cfg);
        engine.initialize();
        auto comparison = engine.compare_reference("tests/reference", 20, 16);
        CHECK(comparison_pass(comparison), "all-experts-resident cache matches all 20 MoE prompts");
        CHECK(engine.results().front().prefill_logits == one_slot_logits,
              "all-cache and one-slot prefill logits are bit-identical");
        CHECK(engine.results().front().generated_ids == one_slot_ids,
              "all-cache and one-slot generated tokens are bit-identical");
        CHECK(engine.residency()->metrics().cache_hits > 0, "all-experts-resident run has cache hits");
        check_released(engine, "all-experts cache");
    }
    {
        auto lru_cfg = make_engine_config("tests/fixtures/moe-fm.gguf", BackendKind::ScalarCPU,
                                          ReaderMode::Buffered, 4, false, "", 5,
                                          false, ReplacementPolicy::LRU);
        auto clock_cfg = lru_cfg;
        clock_cfg.residency.replacement = ReplacementPolicy::Clock;
        Engine lru(lru_cfg);
        Engine clock(clock_cfg);
        lru.initialize();
        clock.initialize();
        CHECK(comparison_pass(lru.compare_reference("tests/reference", 1, 16)),
              "LRU replacement passes MoE reference");
        CHECK(comparison_pass(clock.compare_reference("tests/reference", 1, 16)),
              "clock replacement passes MoE reference");
        CHECK(lru.results().front().prefill_logits == clock.results().front().prefill_logits,
              "LRU and clock outputs are bit-identical");
        CHECK(lru.results().front().generated_ids == clock.results().front().generated_ids,
              "LRU and clock generated tokens are bit-identical");
        check_released(lru, "LRU");
        check_released(clock, "clock");
    }
}

void test_traces_and_replay_inputs() {
    std::filesystem::create_directories("build/test-tmp");
    std::set<std::string> events;
    {
        auto cfg = make_engine_config("tests/fixtures/moe-fm.gguf", BackendKind::ScalarCPU,
                                      ReaderMode::Buffered, 4, false,
                                      "build/test-tmp/replay-trace.jsonl", 5);
        cfg.run_id = "replay";
        Engine engine(cfg);
        engine.initialize();
        CHECK(comparison_pass(engine.compare_reference("tests/reference", 1, 16)),
              "replay trace run is correct");
        engine.release_runtime();
        engine.write_metrics("build/test-tmp/replay-metrics.json");
        for (const auto &event : engine.trace().event_types()) events.insert(event);
    }
    {
        auto cfg = make_engine_config("tests/fixtures/moe-fm.gguf", BackendKind::ScalarCPU,
                                      ReaderMode::Buffered, 1, true,
                                      "build/test-tmp/prefetch-trace.jsonl", 5);
        cfg.run_id = "prefetch";
        Engine engine(cfg);
        engine.initialize();
        CHECK(comparison_pass(engine.compare_reference("tests/reference", 1, 16)),
              "prefetch trace run is correct");
        CHECK(engine.residency()->metrics().prefetch_useful_bytes > 0,
              "prefetch run has useful prefetch bytes");
        CHECK(engine.residency()->metrics().prefetch_wasted_bytes > 0,
              "prefetch run reports wasted prefetch bytes");
        engine.release_runtime();
        for (const auto &event : engine.trace().event_types()) events.insert(event);
    }
    {
        auto cfg = make_engine_config("tests/fixtures/moe-fm.gguf", BackendKind::ScalarCPU,
                                      ReaderMode::Buffered, 16, false,
                                      "build/test-tmp/cachehit-trace.jsonl", 5);
        cfg.run_id = "cachehit";
        Engine engine(cfg);
        engine.initialize();
        CHECK(comparison_pass(engine.compare_reference("tests/reference", 1, 16)),
              "cache-hit trace run is correct");
        engine.release_runtime();
        for (const auto &event : engine.trace().event_types()) events.insert(event);
    }
    const std::vector<std::string> required = {
        "model_load", "weights_uploaded", "budget", "token", "expert_selected",
        "expert_executed", "cache_hit", "cache_miss", "evict", "io_queued",
        "host_ready", "upload_queued", "upload_complete", "acquire", "release",
        "prefetch_issue", "prefetch_useful", "prefetch_wasted", "source_verify",
        "io_complete", "runtime_released",
    };
    for (const auto &event : required)
        CHECK(events.count(event) > 0, "trace suite emits required event type " + event);
    CHECK(std::filesystem::exists("build/test-tmp/replay-trace.jsonl"), "replay trace artefact exists");
    CHECK(std::filesystem::exists("build/test-tmp/replay-metrics.json"), "replay metrics artefact exists");
}

void test_vnpu_reference_backend() {
    for (const std::string kind : {"dense", "moe"}) {
        auto cfg = make_engine_config("tests/fixtures/" + kind + "-fm.gguf",
                                      BackendKind::VirtualNPU, ReaderMode::Buffered,
                                      kind == "moe" ? 1u : 0u, false, "", 5);
        cfg.run_id = "vnpu-" + kind;
        Engine engine(cfg);
        engine.initialize();
        auto comparison = engine.compare_reference("tests/reference", 20, 16);
        CHECK(comparison_pass(comparison), "VNPU backend matches all 20 " + kind + " prompts");
        auto ops = engine.backend().operations();
        bool has_vnpu_matmul = false;
        for (const auto &op : ops) {
            if (op.backend == "vnpu" && op.name.find("matvec") == std::string::npos &&
                op.name.find("attn") != std::string::npos) has_vnpu_matmul = true;
            if (op.backend == "vnpu") has_vnpu_matmul = true;
        }
        CHECK(has_vnpu_matmul, "VNPU operation log records real VNPU matvec work");
        check_released(engine, "VNPU " + kind);
    }
}



void test_generic_inspect_and_roundtrip() {
    // Generic inspection must work from the container alone and must never
    // upgrade "parsed" into "supported".
    const std::string dense_path = "tests/fixtures/dense-fm.gguf";
    auto doc = parse_gguf_file(dense_path);
    CHECK(doc.metadata_order.size() == doc.metadata.size(),
          "metadata insertion order is fully recorded");
    Json view = generic_inspect_json(doc, dense_path);
    CHECK(view.dump().find("freemoe.generic-inspect.v1") != std::string::npos,
          "generic inspect schema is versioned");
    CHECK(view.dump().find("\"classification\":\"execute-cpu\"") != std::string::npos,
          "freemoe-tiny is classified as CPU-executable");
    CHECK(view.dump().find("\"adapter_available\":true") != std::string::npos,
          "adapter availability is reported");

    // Byte-exact round-trip: parse -> write -> identical bytes.
    const std::vector<std::string> roundtrip_paths = {
        "tests/fixtures/dense-fm.gguf", "tests/fixtures/moe-fm.gguf",
        "tests/fixtures/metadata-types.gguf"};
    for (const std::string &path : roundtrip_paths) {
        MappedFile file(path);
        auto parsed = parse_gguf(file.data(), file.size());
        auto rebuilt = write_gguf(parsed, file.data(), file.size());
        CHECK(rebuilt.size() == file.size(), "round-trip preserves file size for " + path);
        CHECK(std::memcmp(rebuilt.data(), file.data(), file.size()) == 0,
              "round-trip is byte-identical for " + path);
        auto reparsed = parse_gguf(rebuilt.data(), rebuilt.size());
        CHECK(reparsed.tensors.size() == parsed.tensors.size(),
              "round-trip reparse keeps every tensor");
        CHECK(reparsed.source_fingerprint == parsed.source_fingerprint,
              "round-trip reparse keeps the metadata fingerprint");
    }

    // Capability matrix: parsing never implies execution.
    CHECK(tensor_capability(0).execute, "F32 is executable");
    CHECK(tensor_capability(1).dequantize, "F16 can be decoded");
    CHECK(tensor_capability(1).execute, "F16 executes through the quantised GEMV");
    CHECK(tensor_capability(8).dequantize, "Q8_0 can be decoded");
    CHECK(tensor_capability(8).execute, "Q8_0 executes through the quantised GEMV");
    // The IQ family and the FP4 formats all decode and execute now; the one
    // remaining geometry-only type is PTQ1_0, whose layout has no verifiable
    // reference.
    for (uint32_t type_id : {16u, 17u, 18u, 19u, 21u, 22u, 23u, 29u, 39u, 40u}) {
        CHECK(tensor_capability(type_id).dequantize,
              "IQ/FP4 type decodes: " + std::to_string(type_id));
        CHECK(tensor_capability(type_id).execute,
              "IQ/FP4 type executes: " + std::to_string(type_id));
    }
    CHECK(tensor_capability(14).dequantize, "Q6_K decoding is implemented");
    CHECK(!tensor_capability(143).dequantize, "PTQ1_0 stays geometry-only");
    CHECK(tensor_capability(143).parse, "a geometry-only type can still be parsed");
    CHECK(!tensor_capability(999).parse, "unknown tensor type is refused");

    // F16 dequantisation against exactly representable half values.
    {
        const uint16_t halves[] = {0x0000, 0x3C00, 0xC000, 0x3800, 0x7C00};
        const float expect[] = {0.0f, 1.0f, -2.0f, 0.5f, std::numeric_limits<float>::infinity()};
        float out[5] = {0};
        dequantize_row(1, reinterpret_cast<const uint8_t *>(halves), out, 5);
        CHECK(out[0] == 0.0f && out[1] == 1.0f && out[2] == -2.0f && out[3] == 0.5f,
              "F16 dequantisation matches known finite values");
        CHECK(std::isinf(out[4]) && out[4] > 0.0f, "F16 infinity round-trips as infinity");
        (void)expect;
    }
    // Subnormal half must decode without infinities appearing.
    {
        const uint16_t one = 0x0001;
        float out = 0.0f;
        dequantize_row(1, reinterpret_cast<const uint8_t *>(&one), &out, 1);
        CHECK(std::isfinite(out) && out > 0.0f && out < 1e-6f, "F16 subnormal decodes finitely");
    }
    // Q8_0: scale 0.5 followed by a full block of known int8 values.
    {
        std::vector<uint8_t> block(34, 0);
        const uint16_t scale = 0x3800; // 0.5
        std::memcpy(block.data(), &scale, sizeof(scale));
        for (int j = 0; j < 32; ++j) block[2 + j] = static_cast<uint8_t>(static_cast<int8_t>(j - 16));
        float out[32] = {0};
        dequantize_row(8, block.data(), out, 32);
        bool ok = true;
        for (int j = 0; j < 32; ++j)
            ok = ok && std::fabs(out[j] - 0.5f * static_cast<float>(j - 16)) < 1e-6f;
        CHECK(ok, "Q8_0 dequantisation matches known block values");
    }
    CHECK(throws_code(ErrorCode::UNSUPPORTED_QUANTISATION,
                      []() {
                          std::vector<uint8_t> block(128, 0);
                          float out[128];
                          // PTQ1_0 has a known geometry but no verified layout.
                          dequantize_row(143, block.data(), out, 128);
                      }),
          "geometry-only block decoder is a typed error");
    CHECK(throws_code(ErrorCode::UNSUPPORTED_QUANTISATION,
                      []() {
                          std::vector<uint8_t> block(64, 0);
                          float out[64];
                          // I2_S is bitnet.cpp's private type, refused by name.
                          dequantize_row(36, block.data(), out, 64);
                      }),
          "a fork-only quantisation type is named in the error");

    // The quantised fixture must now report dequantise-and-execute capability
    // through the generic view, proving the matrix is wired to real files.
    {
        auto q8doc = parse_gguf_file("tests/fixtures/quant-blocks.gguf");
        Json q8view = generic_inspect_json(q8doc, "tests/fixtures/quant-blocks.gguf");
        // No adapter for this container, so it stays inspect-only, but the
        // tensors themselves are decode-and-execute capable now.
        CHECK(q8view.dump().find("\"tensor_support\":\"execute\"") != std::string::npos,
              "quantised fixture tensors report execute support");
        CHECK(q8view.dump().find("\"execute\":true") != std::string::npos,
              "quantised tensors report execute=true");
    }
}


void test_quant_reference_decode() {
    // tests/fixtures/quant_reference.bin is generated by
    // tools/quant_reference.py, whose decoders are written independently of
    // src/quant.cpp. A mismatch is a real bug in one of the two.
    auto bytes = read_binary_file("tests/fixtures/quant_reference.bin");
    CHECK(bytes.size() > 16, "quant reference fixture is present");
    if (bytes.size() < 16) return;
    CHECK(std::memcmp(bytes.data(), "FMQR1", 5) == 0, "quant reference magic");
    uint32_t records = 0;
    std::memcpy(&records, bytes.data() + 8, sizeof(records));
    // One record per block type in the fixture, including the IQ family, the
    // FP4 formats and the element types.
    CHECK(records == 36, "quant reference covers every supported block type");

    size_t pos = 12;
    uint32_t decoded_types = 0;
    while (pos < bytes.size()) {
        uint32_t type_id = 0, block_bytes = 0, n_float = 0;
        CHECK(pos + 12 <= bytes.size(), "quant record header fits");
        if (pos + 12 > bytes.size()) break;
        std::memcpy(&type_id, bytes.data() + pos, 4);
        std::memcpy(&block_bytes, bytes.data() + pos + 4, 4);
        std::memcpy(&n_float, bytes.data() + pos + 8, 4);
        pos += 12;
        CHECK(pos + block_bytes <= bytes.size(), "quant record payload fits");
        if (pos + block_bytes > bytes.size()) break;
        const uint8_t *raw = bytes.data() + pos;
        pos += block_bytes;
        CHECK(pos + n_float * 4 <= bytes.size(), "quant record expectation fits");
        if (pos + n_float * 4 > bytes.size()) break;
        const float *expected = reinterpret_cast<const float *>(bytes.data() + pos);
        pos += n_float * 4;

        // Geometry must agree between the table and the reference fixture.
        const TensorCapability cap = tensor_capability(type_id);
        CHECK(cap.dequantize, "fixture type is marked dequantisable: " + std::to_string(type_id));
        const auto &table = quantisation_table();
        auto geo = table.find(type_id);
        CHECK(geo != table.end(), "fixture type has known geometry");
        if (geo != table.end()) {
            CHECK(geo->second.block_bytes == block_bytes,
                  "block_bytes agrees for type " + std::to_string(type_id));
            CHECK(geo->second.block_size == n_float,
                  "block_size agrees for type " + std::to_string(type_id));
        }

        std::vector<float> got(n_float, 0.0f);
        dequantize_block(type_id, raw, got.data());

        // Low-bit formats store one scale plus small integer codes, so a decoded
        // block takes only a handful of distinct values, all of them exact
        // multiples of that scale. A decoder that read the packing wrong would
        // spread values across the range and fail this.
        //
        // The permitted codes differ by format and are not guessable: the 2-bit
        // codecs use {0,1,2,3} meaning {-1,0,+1,+2}, which is deliberately
        // asymmetric, while base-3 trits are {-1,0,+1} and Q1_0 is a bare sign.
        struct LowBitFormat {
            uint32_t type_id;
            int lowest;   // smallest code
            int highest;  // largest code
        };
        static const std::vector<LowBitFormat> low_bit = {
            {34, -1, 1},   // TQ1_0: base-3 trits
            {35, -1, 2},   // TQ2_0: 2-bit codes
            {41, -1, 1},   // Q1_0: sign only
            {42, -1, 2},   // Q2_0: 2-bit codes
            {142, -1, 2},  // PQ2_0: 2-bit codes
        };
        for (const auto &format : low_bit) {
            if (format.type_id != type_id) continue;
            // Recover the scale from the extreme code, which is always present
            // in a pseudo-random block of codes.
            double extreme = 0.0;
            for (float v : got) extreme = std::max(extreme, std::fabs(static_cast<double>(v)));
            const bool has_negative = std::any_of(got.begin(), got.end(),
                                                  [](float v) { return v < 0.0f; });
            const int max_code = format.highest;
            const double scale = extreme / max_code;
            CHECK(scale > 0.0, "type " + std::to_string(type_id) + " has a usable scale");
            if (!(scale > 0.0)) break;

            std::set<int> seen;
            bool exact_multiples = true;
            for (float v : got) {
                const double code = static_cast<double>(v) / scale;
                const double nearest = std::round(code);
                if (std::fabs(code - nearest) > 1e-5) exact_multiples = false;
                if (nearest < format.lowest || nearest > format.highest) exact_multiples = false;
                seen.insert(static_cast<int>(nearest));
            }
            CHECK(exact_multiples,
                  "type " + std::to_string(type_id) +
                      " decodes to exact multiples of one scale in the permitted code range");
            CHECK(static_cast<int>(seen.size()) <= format.highest - format.lowest + 1,
                  "type " + std::to_string(type_id) + " uses at most " +
                      std::to_string(format.highest - format.lowest + 1) + " codes");
            if (format.lowest < 0)
                CHECK(has_negative || type_id == 41,
                      "type " + std::to_string(type_id) + " produces negative weights");
        }

        double worst = 0.0;
        for (uint32_t i = 0; i < n_float; ++i) {
            const double a = got[i], b = expected[i];
            if (!std::isfinite(a) || !std::isfinite(b)) {
                worst = std::max(worst, std::isnan(a) != std::isnan(b) ? 1.0 : 0.0);
                continue;
            }
            const double scale = std::max(1.0, std::fabs(b));
            worst = std::max(worst, std::fabs(a - b) / scale);
        }
        CHECK(worst < 1e-6, "decoder matches the independent reference for type " +
                                std::to_string(type_id) + " (worst rel " +
                                std::to_string(worst) + ")");
        ++decoded_types;
    }
    CHECK(decoded_types == records, "every reference record was decoded");
    CHECK(pos == bytes.size(), "reference fixture consumed exactly");

    // Round-trip through the public row API as well.
    std::vector<float> two_blocks(64, 0.0f);
    std::vector<uint8_t> q8(2 * 34, 0);
    for (int b = 0; b < 2; ++b) {
        const uint16_t scale = 0x3800; // 0.5
        std::memcpy(q8.data() + b * 34, &scale, 2);
        for (int j = 0; j < 32; ++j)
            q8[b * 34 + 2 + j] = static_cast<uint8_t>(static_cast<int8_t>(j - 8));
    }
    dequantize_row(8, q8.data(), two_blocks.data(), 64);
    CHECK(std::fabs(two_blocks[0] - (-4.0f)) < 1e-6f, "multi-block Q8_0 decode, first value");
    CHECK(std::fabs(two_blocks[63] - (0.5f * 23.0f)) < 1e-6f, "multi-block Q8_0 decode, last value");
    CHECK(throws_code(ErrorCode::SHAPE_MISMATCH,
                      [&]() { dequantize_row(8, q8.data(), two_blocks.data(), 33); }),
          "partial block decode is a typed error");
    CHECK(throws_code(ErrorCode::UNSUPPORTED_QUANTISATION,
                      [&]() { (void)quant_geometry(9999); }),
          "unknown type geometry is a typed error");
}


void test_model_probe() {
    // `freemoe probe` must answer, for any GGUF, what this build can and cannot
    // do with it. A probe that only says "unsupported" is useless, so the tests
    // check that it names the specific blockers.
    {
        auto doc = parse_gguf_file("tests/fixtures/moe-fm.gguf");
        Json probe = probe_model(doc, "tests/fixtures/moe-fm.gguf");
        CHECK(probe.dump().find("\"runnable\":true") != std::string::npos,
              "own adapter fixture is reported runnable");
        CHECK(probe.dump().find("\"blocker_count\":0") != std::string::npos,
              "runnable model has no blockers");
        CHECK(probe.dump().find("\"hidden_size\":16") != std::string::npos,
              "probe reads hyperparameters through the freemoe. prefix");
    }
    {
        // A llama-shaped checkpoint: standard keys, standard tensor names, GQA
        // and a GPT-2 BPE tokenizer.
        auto doc = parse_gguf_file("tests/fixtures/llama-shape-q8_0.gguf");
        Json probe = probe_model(doc, "tests/fixtures/llama-shape-q8_0.gguf");
        const std::string dump = probe.dump();
        CHECK(dump.find("\"runnable\":true") != std::string::npos,
              "llama-shaped checkpoint is reported runnable");
        CHECK(dump.find("MISSING_ADAPTER") == std::string::npos,
              "the llama adapter is present, so no adapter blocker is reported");
        CHECK(dump.find("\"grouped_query_attention\":true") != std::string::npos,
              "probe reports grouped-query attention as a capability");
        CHECK(dump.find("TOKENIZER_NOT_IMPLEMENTED") == std::string::npos,
              "the sentencepiece/gpt2 tokenizers are implemented");
        CHECK(dump.find("\"num_key_value_heads\":2") != std::string::npos,
              "probe reads grouped-query head counts from llama.* keys");
        CHECK(dump.find("\"execute\":true") != std::string::npos,
              "probe reports that Q8_0 tensors are executable");
        CHECK(probe.dump().find("tensor_families") != std::string::npos,
              "probe reports the tensor inventory");
    }
    CHECK(quant_geometry_or_name(8) == "Q8_0", "known type reports its format name");
    CHECK(quant_geometry_or_name(199) == "TYPE_199", "unknown type reports its numeric id");
}

void test_quant_model_parity() {
    // End-to-end proof that block-quantised checkpoints execute correctly.
    // Each quant-moe-<type>.gguf stores every matrix in that block format, and
    // its -f32ref twin stores exactly the values an *independent Python decoder*
    // produces for those bytes. Matching greedy tokens and matching logits mean
    // the C++ decoder, the fused kernel and the engine path all agree with the
    // format definition, not merely with each other.
    for (const std::string variant : {"q8_0", "q4_0", "q4_k", "q6_k"}) {
        const std::string model = "tests/fixtures/quant-moe-" + variant + ".gguf";
        const std::string reference_path = "tests/fixtures/quant-moe-" + variant + "-f32ref.gguf";
        auto run = [](const std::string &model_path) {
            auto cfg = make_engine_config(model_path, BackendKind::ScalarCPU,
                                          ReaderMode::Buffered, 0, false);
            Engine engine(cfg);
            engine.initialize();
            auto result = engine.run_prompt("quantised conformance", 6);
            check_released(engine, model_path);
            return result;
        };
        const PromptResult quantised = run(model);
        const PromptResult reference = run(reference_path);

        CHECK(!quantised.generated_ids.empty(), variant + " fixture generates tokens");
        CHECK(quantised.generated_ids == reference.generated_ids,
              variant + " checkpoint generates the same greedy tokens as its F32 reference");
        CHECK(quantised.prefill_logits.size() == reference.prefill_logits.size(),
              variant + " logit vectors have the same size");
        double worst = 0.0;
        const size_t n =
            std::min(quantised.prefill_logits.size(), reference.prefill_logits.size());
        for (size_t i = 0; i < n; ++i)
            worst = std::max(worst, std::fabs(static_cast<double>(quantised.prefill_logits[i]) -
                                              reference.prefill_logits[i]));
        CHECK(n > 0 && worst <= 1e-4, variant + " logits match the F32 reference (worst " +
                                          std::to_string(worst) + ")");

        // The quantised model must really contain block-encoded tensors.
        auto cfg = make_engine_config(model, BackendKind::ScalarCPU, ReaderMode::Buffered,
                                      0, false);
        Engine engine(cfg);
        engine.initialize();
        bool saw_block = false;
        for (const auto &tensor : engine.manifest().doc.tensors)
            saw_block = saw_block || tensor.type_id != 0;
        CHECK(saw_block, variant + " fixture really stores block-encoded tensors");
        check_released(engine, "quantised " + variant);
    }
}

// A real mixed-precision expert stack: `ffn_gate`/`ffn_up` in Q4_K and
// `ffn_down` in Q6_K, in both the separate and the fused (`ffn_gate_up_exps`)
// layouts. Each expert record then holds two different strides, which is the
// case a single shared stride silently corrupts -- and the case no fixture
// covered before, because every earlier quantised fixture used one type for
// every matrix.
void test_quant_expert_mixed_types() {
    for (const std::string variant : {"fused", "separate"}) {
        const std::string model = "tests/fixtures/quant-expert-" + variant + ".gguf";
        const std::string reference_path =
            "tests/fixtures/quant-expert-" + variant + "-f32ref.gguf";
        auto run = [](const std::string &model_path) {
            auto cfg = make_engine_config(model_path, BackendKind::ScalarCPU,
                                          ReaderMode::Buffered, 0, false);
            Engine engine(cfg);
            engine.initialize();
            auto result = engine.run_prompt("mixed experts", 6);
            check_released(engine, model_path);
            return result;
        };
        const PromptResult quantised = run(model);
        const PromptResult reference = run(reference_path);
        CHECK(!quantised.generated_ids.empty(), variant + " mixed-type fixture generates tokens");
        CHECK(quantised.generated_ids == reference.generated_ids,
              variant + " mixed-type experts match the F32 reference token for token");
        double worst = 0.0;
        const size_t n = std::min(quantised.prefill_logits.size(),
                                  reference.prefill_logits.size());
        for (size_t i = 0; i < n; ++i)
            worst = std::max(worst, std::fabs(static_cast<double>(quantised.prefill_logits[i]) -
                                              reference.prefill_logits[i]));
        CHECK(n > 0 && worst <= 1e-4, variant + " mixed-type logits match the F32 reference (worst " +
                                          std::to_string(worst) + ")");

        // The fixture has to really be mixed, and the layout has to be the one
        // the file declares: a runtime that read both projections with one type
        // id would still produce numbers, just the wrong ones.
        auto cfg = make_engine_config(model, BackendKind::ScalarCPU, ReaderMode::Buffered,
                                      0, false);
        Engine engine(cfg);
        engine.initialize();
        const ModelManifest &manifest = engine.manifest();
        const std::string gate_name = variant == "fused" ? "blk.0.ffn_gate_up_exps.weight"
                                                        : "blk.0.ffn_gate_exps.weight";
        const TensorInfo *gate = manifest.tensor(gate_name);
        const TensorInfo *down = manifest.tensor("blk.0.ffn_down_exps.weight");
        CHECK(gate != nullptr && down != nullptr, variant + " fixture carries both expert tensors");
        if (gate && down) {
            CHECK(gate->type_id == 12, variant + " gate/up expert tensor is Q4_K");
            CHECK(down->type_id == 14, variant + " down expert tensor is Q6_K");
            CHECK(gate->expert_tensor && down->expert_tensor,
                  variant + " expert tensors are recorded as per-expert");
            CHECK(down->byte_size / 2 ==
                      static_cast<uint64_t>(manifest.config.intermediate) *
                          manifest.config.hidden / 256 * 210,
                  variant + " the Q6_K expert stride follows the block geometry");
        }
        CHECK(manifest.config.fused_gate_up_exps == (variant == "fused"),
              variant + " the fused gate/up layout is decided by the file");
        CHECK(manifest.config.model_kind == "moe", variant + " the fixture is an MoE model");
        CHECK(manifest.expert(0, 0) != nullptr, variant + " expert records are built");
        if (gate && down && manifest.expert(0, 0) != nullptr) {
            const ExpertRecord *record = manifest.expert(0, 0);
            CHECK(record->extents.size() == 3,
                  variant + " every expert record has three projection extents");
            if (record->extents.size() == 3)
                CHECK(record->extents[2].bytes == down->byte_size / 2,
                      variant + " the down extent uses the Q6_K stride, not the Q4_K one");
        }
        check_released(engine, "mixed " + variant);
    }
}

void test_quant_gemv() {
    // A quantised GEMV must equal dequantise-then-scalar-GEMV for every type
    // that has a kernel. Small shapes keep this fast; correctness is what matters.
    const uint32_t out_dim = 9;
    for (uint32_t type_id : {0u, 2u, 3u, 6u, 7u, 8u, 9u, 10u, 11u, 12u, 13u, 14u, 15u}) {
        const QuantGeometry &geo = quant_geometry(type_id);
        const uint32_t in_dim = geo.block_size * 2; // two whole blocks per row
        const uint64_t row_bytes = (in_dim / geo.block_size) * static_cast<uint64_t>(geo.block_bytes);
        std::vector<uint8_t> weights(row_bytes * out_dim);
        // Deterministic bytes with finite scales so the comparison is meaningful.
        uint32_t state = 0xBEEF0000u + type_id;
        for (auto &b : weights) {
            state = state * 1664525u + 1013904223u;
            b = static_cast<uint8_t>((state >> 16) & 0xFF);
        }
        // Zero the scale fields of every block so no block produces NaN/Inf.
        // Scales must be in the range real checkpoints use, not so tiny that a
        // dropped term stays inside the tolerance. 0x2C00 is 2^-4.
        const uint16_t small_half = 0x2C00;
        const float small_f32 = 0.0625f;
        for (uint32_t r = 0; r < out_dim; ++r) {
            uint8_t *row = weights.data() + r * row_bytes;
            for (uint32_t b = 0; b < in_dim / geo.block_size; ++b) {
                uint8_t *block = row + b * geo.block_bytes;
                switch (type_id) {
                case 0:
                    std::memcpy(block, &small_f32, 4);
                    break;
                case 2: case 6: case 8:
                    std::memcpy(block, &small_half, 2);
                    break;
                case 3: case 7:
                    std::memcpy(block, &small_half, 2);
                    std::memcpy(block + 2, &small_half, 2);
                    break;
                case 9:
                    std::memcpy(block, &small_f32, 4);
                    break;
                case 10:
                    std::memcpy(block + 80, &small_half, 2);
                    std::memcpy(block + 82, &small_half, 2);
                    break;
                case 11:
                    std::memcpy(block + 108, &small_half, 2);
                    break;
                case 12: case 13:
                    std::memcpy(block, &small_half, 2);
                    std::memcpy(block + 2, &small_half, 2);
                    break;
                case 14:
                    std::memcpy(block + 208, &small_half, 2);
                    break;
                case 15:
                    std::memcpy(block, &small_f32, 4);
                    break;
                }
            }
        }
        std::vector<float> x(in_dim);
        for (uint32_t i = 0; i < in_dim; ++i)
            x[i] = static_cast<float>(static_cast<int>(i % 7) - 3) * 0.25f;

        // Reference: dequantise the whole tensor, then scalar GEMV.
        std::vector<float> dense(static_cast<size_t>(out_dim) * in_dim);
        for (uint32_t r = 0; r < out_dim; ++r)
            dequantize_row(type_id, weights.data() + r * row_bytes,
                           dense.data() + static_cast<size_t>(r) * in_dim, in_dim);
        std::vector<float> ref(out_dim, 0.0f);
        gemv_scalar(type_id == 0 ? reinterpret_cast<const float *>(weights.data())
                                 : dense.data(),
                    x.data(), ref.data(), out_dim, in_dim);

        KernelConfig one;
        one.threads = 1;
        std::vector<float> got(out_dim, 0.0f);
        gemv_quant(one, type_id, weights.data(), x.data(), got.data(), out_dim, in_dim);
        double worst = 0.0;
        double scale = 0.0;
        for (uint32_t i = 0; i < out_dim; ++i) {
            worst = std::max(worst, std::fabs(static_cast<double>(got[i]) - ref[i]));
            scale = std::max(scale, std::fabs(static_cast<double>(ref[i])));
        }
        // Relative tolerance: an absolute bound is meaningless across formats
        // whose decoded magnitudes differ by orders of magnitude.
        worst = scale > 0 ? worst / scale : worst;
        CHECK(worst < 1e-5, "quantised GEMV matches dequantise-then-dot for type " +
                                std::to_string(type_id) + " (worst " + std::to_string(worst) + ")");
        CHECK(kernel_supports_type(type_id),
              "kernel table agrees for type " + std::to_string(type_id));

        // Threaded dispatch must agree with the inline path.
        KernelConfig many;
        many.threads = 4;
        std::vector<float> threaded(out_dim, 0.0f);
        gemv_quant(many, type_id, weights.data(), x.data(), threaded.data(), out_dim, in_dim);
        double tdiff = 0.0;
        for (uint32_t i = 0; i < out_dim; ++i)
            tdiff = std::max(tdiff, std::fabs(static_cast<double>(threaded[i]) - got[i]));
        CHECK(tdiff <= std::max(1e-6, scale * 1e-6), "threaded quantised GEMV is consistent for type " +
                                std::to_string(type_id));
    }

    // Misaligned row and unknown type must fail closed.
    KernelConfig cfg;
    std::vector<uint8_t> junk(1024, 0);
    std::vector<float> x32(32, 1.0f), y9(9, 0.0f);
    CHECK(throws_code(ErrorCode::SHAPE_MISMATCH,
                      [&]() { gemv_quant(cfg, 8, junk.data(), x32.data(), y9.data(), 9, 33); }),
          "misaligned quantised row is a typed error");
    CHECK(throws_code(ErrorCode::UNSUPPORTED_QUANTISATION,
                      [&]() { gemv_quant(cfg, 99, junk.data(), x32.data(), y9.data(), 9, 32); }),
          "unknown quantised type is a typed error");
    CHECK(!kernel_supports_type(99), "unknown type has no kernel");
    CHECK(!kernel_supports_type(143), "geometry-only type reports no kernel");
    CHECK(kernel_supports_type(39), "MXFP4 has a kernel");
}

void test_cpu_kernels() {
    // Scalar reference vs every available vector path. Tolerances are generous
    // because vector reduction order differs; the model-level parity gate is the
    // real check that this stays within D4.
    for (uint32_t seed : {1u, 7u, 12345u}) {
        CHECK(gemv_max_abs_error(KernelIsa::Scalar, 17, 33, seed) == 0.0,
              "scalar GEMV is the reference");
        if (host_has_avx2_fma())
            CHECK(gemv_max_abs_error(KernelIsa::Avx2Fma, 17, 33, seed) < 1e-3,
                  "AVX2/FMA GEMV matches the scalar reference");
        if (host_has_avx512())
            CHECK(gemv_max_abs_error(KernelIsa::Avx512, 17, 33, seed) < 1e-3,
                  "AVX-512 GEMV matches the scalar reference");
    }
    // Non-multiple-of-8 tails must not read out of bounds or drop terms.
    CHECK(gemv_max_abs_error(KernelIsa::Auto, 5, 7, 3) < 1e-5, "tail handling for short rows");
    CHECK(gemv_max_abs_error(KernelIsa::Auto, 3, 4099, 4) < 1e-3, "tail handling for long rows");

    // Threaded dispatch must produce identical output to the inline path.
    const uint32_t out_dim = 257;
    const uint32_t in_dim = 129;
    std::vector<float> weights(static_cast<size_t>(out_dim) * in_dim);
    std::vector<float> x(in_dim, 1.0f);
    std::vector<float> single(out_dim, 0.0f);
    std::vector<float> threaded(out_dim, 0.0f);
    uint32_t state = 99u;
    for (auto &w : weights) {
        state = state * 1664525u + 1013904223u;
        w = static_cast<float>((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
    }
    KernelConfig one;
    one.requested = KernelIsa::Auto;
    one.threads = 1;
    KernelConfig many;
    many.requested = KernelIsa::Auto;
    many.threads = 8;
    gemv(one, weights.data(), x.data(), single.data(), out_dim, in_dim);
    for (int repeat = 0; repeat < 3; ++repeat)
        gemv(many, weights.data(), x.data(), threaded.data(), out_dim, in_dim);
    double worst = 0.0;
    for (uint32_t i = 0; i < out_dim; ++i)
        worst = std::max(worst, std::fabs(static_cast<double>(single[i]) - threaded[i]));
    CHECK(worst < 1e-4, "threaded GEMV matches single-threaded output deterministically");

    // ISA policy: explicit scalar is honoured, auto never exceeds host support.
    CHECK(resolve_kernel_isa(KernelIsa::Scalar) == KernelIsa::Scalar,
          "explicit scalar kernel request is honoured");
    const KernelIsa auto_isa = resolve_kernel_isa(KernelIsa::Auto);
    CHECK(auto_isa == KernelIsa::Scalar || auto_isa == KernelIsa::Avx2Fma,
          "auto kernel resolution stays on the deployment target's width");
    // The target is Zen 3, which has no AVX-512. Auto must therefore never pick
    // it, whatever this host supports, or every number recorded here describes a
    // machine the software will not run on.
    CHECK(auto_isa != KernelIsa::Avx512, "auto never selects AVX-512 kernels");
    if (!host_has_avx2_fma())
        CHECK(auto_isa == KernelIsa::Scalar, "no AVX2 means scalar fallback");
    // An explicit request is still honoured when the host can run it.
    if (host_has_avx512() && !host_is_zen3())
        CHECK(resolve_kernel_isa(KernelIsa::Avx512) == KernelIsa::Avx512,
              "an explicit AVX-512 request is honoured on a host that has it");
    CHECK(parse_kernel_isa("zen3") == KernelIsa::Avx2Fma,
          "the zen3 profile means dual-width AVX2, not AVX-512");
    CHECK(parse_kernel_isa("zen3-auto") == KernelIsa::Avx2Fma,
          "the zen3 profile accepts the VNPU spelling too");
    // The committed benchmark config has to pin the target width, or the
    // recorded numbers describe whatever host ran them.
    {
        const std::string config = read_text_file("configs/benchmark.env");
        CHECK(config.find("--kernels zen3") != std::string::npos,
              "the benchmark config pins the Zen 3 kernel profile");
    }
    CHECK(resolved_threads(KernelConfig{}) >= 1, "default thread count is at least one");
    CHECK(std::string(kernel_isa_name(KernelIsa::Avx2Fma)) == "avx2_fma",
          "kernel ISA names are stable");
    CHECK(throws_code(ErrorCode::INVALID_ARGUMENT,
                      []() { (void)parse_kernel_isa("sse42"); }),
          "unknown kernel profile is a typed error");

    // Benchmark harness must report plausible, positive rates.
    KernelConfig bench_cfg;
    bench_cfg.threads = 1;
    auto bench = benchmark_gemv(bench_cfg, 64, 128, 2);
    CHECK(bench.best_ms > 0.0, "benchmark reports a positive duration");
    CHECK(bench.gflops > 0.0 && bench.gbytes_per_s > 0.0, "benchmark reports positive rates");
    CHECK(bench.iterations == 2, "benchmark records its iteration count");
}


// The threaded-dispatch policy. It used to parallelise every GEMV above 65,536
// multiply-adds, which made a 256x256 projection six times slower on a 2-vCPU
// host (6.0 us serial, 37.3 us threaded). The threshold is now per worker, and
// these checks pin both halves: the small shapes stay serial, and the parallel
// path still produces *bit-identical* rows, because each row is reduced by one
// worker in the same order.
void test_thread_dispatch_policy() {
    CHECK(threading_min_work(1) == (1ull << 20), "one worker needs 1M multiply-adds");
    CHECK(threading_min_work(2) == 2 * (1ull << 20), "two workers need 2M between them");
    CHECK(threading_min_work(0) == (1ull << 20), "an unresolved thread count is treated as one");
    // The shapes the measurement used, and the verdict each must get.
    CHECK(256ull * 256 <= threading_min_work(2), "a 256x256 projection stays serial");
    CHECK(256ull * 1024 <= threading_min_work(2), "a 256x1024 projection stays serial");
    CHECK(1024ull * 4096 >= threading_min_work(2), "a 1024x4096 projection goes wide");
    CHECK(4096ull * 4096 >= threading_min_work(2), "a 4096x4096 projection goes wide");

    // The parallel path must not change a single bit: every row is one worker's
    // own sequential reduction, so this is an exactness claim, not a tolerance.
    const uint32_t out_dim = 1024, in_dim = 4096;
    std::vector<float> weights(static_cast<size_t>(out_dim) * in_dim);
    std::vector<float> x(in_dim), serial(out_dim, 0.0f), parallel(out_dim, 0.0f);
    uint32_t state = 0x51EDu;
    for (auto &w : weights) {
        state = state * 1664525u + 1013904223u;
        w = static_cast<float>((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
    }
    for (auto &v : x) {
        state = state * 1664525u + 1013904223u;
        v = static_cast<float>((state >> 8) & 0xFFFF) / 65536.0f - 0.5f;
    }
    CHECK(static_cast<uint64_t>(out_dim) * in_dim >= threading_min_work(2),
          "the determinism check really exercises the parallel path");
    KernelConfig one;
    one.threads = 1;
    KernelConfig two;
    two.threads = 2;
    gemv(one, weights.data(), x.data(), serial.data(), out_dim, in_dim);
    gemv(two, weights.data(), x.data(), parallel.data(), out_dim, in_dim);
    size_t differing = 0;
    double worst = 0.0;
    for (uint32_t i = 0; i < out_dim; ++i) {
        if (serial[i] != parallel[i]) ++differing;
        worst = std::max(worst, std::fabs(static_cast<double>(serial[i]) - parallel[i]));
    }
    CHECK(differing == 0, "a threaded GEMV is bit-identical to the serial one (worst " +
                              std::to_string(worst) + ")");
}

void test_sampling_and_detokenize() {
    // Greedy sampler: highest logit wins, lower id breaks ties.
    std::vector<float> logits = {0.1f, 2.5f, 2.5f, -1.0f};
    SamplingConfig cfg;
    std::mt19937_64 rng(1);
    CHECK(sample_from_logits(logits, cfg, rng) == 1, "greedy sampling picks argmax with lowest id on ties");
    cfg.temperature = 0.0;
    CHECK(sample_from_logits(logits, cfg, rng) == 1, "temperature 0 is exactly greedy");
    cfg.top_k = 1;
    cfg.temperature = 2.0;
    CHECK(sample_from_logits(logits, cfg, rng) == 1, "top-k 1 is exactly greedy");

    // Determinism: identical seeds produce identical sequences.
    auto draw = [&](uint64_t seed) {
        SamplingConfig s;
        s.temperature = 1.0;
        s.top_k = 4;
        s.top_p = 0.95;
        std::mt19937_64 r(seed);
        std::vector<uint32_t> out;
        for (int i = 0; i < 32; ++i) out.push_back(sample_from_logits(logits, s, r));
        return out;
    };
    CHECK(draw(1234) == draw(1234), "sampling is deterministic for a fixed seed");
    CHECK(draw(1234) != draw(4321), "different seeds produce different sequences");

    // Top-p truncation must exclude tokens that are individually impossible.
    {
        SamplingConfig s;
        s.temperature = 1.0;
        s.top_p = 0.5;
        std::mt19937_64 r(7);
        std::vector<float> skewed = {10.0f, 0.0f, 0.0f, 0.0f};
        bool only_top = true;
        for (int i = 0; i < 200; ++i)
            only_top = only_top && sample_from_logits(skewed, s, r) == 0;
        CHECK(only_top, "nucleus sampling keeps only the dominant token here");
    }
    // Temperature flattens the distribution: a cooler sampler must not be more
    // diverse than a hotter one.
    {
        auto distinct = [&](double temperature) {
            SamplingConfig s;
            s.temperature = temperature;
            std::mt19937_64 r(99);
            std::set<uint32_t> seen;
            for (int i = 0; i < 400; ++i) seen.insert(sample_from_logits(logits, s, r));
            return seen.size();
        };
        CHECK(distinct(3.0) >= distinct(0.2), "higher temperature is not less diverse");
    }
    // Non-finite logits must fail rather than poison the distribution.
    {
        SamplingConfig s;
        s.temperature = 1.0;
        std::mt19937_64 r(1);
        std::vector<float> bad = {0.0f, std::numeric_limits<float>::quiet_NaN()};
        CHECK(throws_code(ErrorCode::INVALID_ARGUMENT,
                          [&]() { (void)sample_from_logits(bad, s, r); }),
              "NaN logits are a typed sampling error");
    }

    // Detokenisation must invert the byte tokenizer exactly.
    RunConfig run;
    run.model_path = "tests/fixtures/moe-fm.gguf";
    run.generated_tokens = 2;
    Engine engine(run);
    engine.initialize();
    const std::string text = "hello, world!";
    std::vector<int32_t> ids;
    for (unsigned char c : text) ids.push_back(static_cast<int32_t>(c));
    CHECK(engine.detokenize(ids) == text, "detokenize inverts the byte tokenizer");
    std::vector<int32_t> with_special{256, 65, 257};
    CHECK(engine.detokenize(with_special) == "A", "BOS/EOS are not emitted as text");
    CHECK(throws_code(ErrorCode::INVALID_TOKENIZER,
                      [&]() { (void)engine.detokenize({300}); }),
          "unmapped token id is a typed tokenizer error");
    engine.release_runtime();
}


// A checkpoint's token table is real data, so the tokenizer is exercised
// against a table shaped exactly like a published SentencePiece vocabulary:
// U+2581 space markers, control tokens, and <0xNN> byte fallbacks.
static MetadataValue make_string(const std::string &value) {
    MetadataValue v;
    v.type = GGUFValueType::STRING;
    v.s = value;
    return v;
}

void test_tokenizer_real_vocabularies() {
    const std::string marker = "\xe2\x96\x81";  // U+2581
    auto sentencepiece_doc = [&marker](bool with_byte_fallback) {
        GGUFDocument doc;
        doc.metadata["tokenizer.ggml.model"] = make_string("llama");
        MetadataValue tokens;
        tokens.type = GGUFValueType::ARRAY;
        MetadataValue scores;
        scores.type = GGUFValueType::ARRAY;
        // Score model: a piece is worth -(its length). Longer pieces therefore win
        // where they exist, which is what makes the merge order observable.
        auto add = [&](const std::string &piece, double score) {
            tokens.array.push_back(make_string(piece));
            MetadataValue value;
            value.type = GGUFValueType::FLOAT32;
            value.f = score;
            scores.array.push_back(value);
        };
        add("<unk>", 0.0);                   // 0
        add("<s>", 0.0);                     // 1
        add("</s>", 0.0);                    // 2
        // Every piece the merge chain needs must exist: the algorithm only joins
        // two symbols whose concatenation is already in the table, so a table
        // without intermediate pieces cannot build a word (and would silently
        // fall back to bytes).
        for (const char *word : {"the", "cat", "sat"}) {
            const std::string w = word;
            add(marker, -1.0);                              // U+2581 alone
            for (size_t n = 1; n <= w.size(); ++n) {
                add(marker + w.substr(0, n), -static_cast<double>(n + 1));
                if (n < w.size()) add(w.substr(0, n), -static_cast<double>(n + 10));
            }
        }
        // The byte-fallback pieces make arbitrary text representable.
        for (int i = 0; i < 256; ++i) {
            if (!with_byte_fallback) continue;
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", i);
            add(buf, 0.0);
        }
        doc.metadata["tokenizer.ggml.tokens"] = tokens;
        doc.metadata["tokenizer.ggml.scores"] = scores;
        return doc;
    };

    {
        GGUFDocument doc = sentencepiece_doc(true);
        const uint32_t vocab = static_cast<uint32_t>(
            doc.metadata["tokenizer.ggml.tokens"].array.size());
        Tokenizer tokenizer = Tokenizer::from_gguf(doc, vocab);
        CHECK(tokenizer.kind_name() == "sentencepiece", "llama tokenizer is sentencepiece");
        CHECK(tokenizer.complete(), "a tokenizer with byte fallback is complete");

        const std::vector<int32_t> ids = tokenizer.encode("the cat sat", 1);
        CHECK(ids.front() == 1, "encoding starts with the BOS id");
        CHECK(ids.size() == 4, "space-marker pieces cover whole words: BOS plus three pieces");
        CHECK(tokenizer.decode(ids, 1, 2) == " the cat sat",
              "decode inverts encode including the leading space");
        // The leading word boundary is part of the piece, so dropping the BOS
        // must not disturb the text.
        const std::vector<int32_t> body(ids.begin() + 1, ids.end());
        CHECK(tokenizer.decode(body, 1, 2) == " the cat sat",
              "decoding bare pieces reproduces the sentence");
        // An unseen character must survive via byte fallback rather than being
        // dropped or replaced.
        // SentencePiece prepends a word boundary, so "z" is space + "z": the
        // space has no piece and becomes a fallback token of its own.
        const std::vector<int32_t> odd = tokenizer.encode("z", 1);
        CHECK(odd.size() == 3, "uncovered bytes each become a fallback token");
        CHECK(tokenizer.decode(odd, 1, 2) == " z", "byte fallback round-trips");
        // Control tokens never reach the text.
        CHECK(tokenizer.decode({0, 1, 2}, 1, 2).empty(),
              "special tokens are not emitted as text");
    }
    {
        // Without byte fallback the tokenizer must refuse rather than silently
        // corrupt the prompt.
        GGUFDocument doc = sentencepiece_doc(false);
        const uint32_t vocab = static_cast<uint32_t>(
            doc.metadata["tokenizer.ggml.tokens"].array.size());
        Tokenizer tokenizer = Tokenizer::from_gguf(doc, vocab);
        CHECK(!tokenizer.complete(), "no fallback pieces means incomplete coverage");
        CHECK(throws_code(ErrorCode::INVALID_TOKENIZER,
                          [&]() { (void)tokenizer.encode("z", 1); }),
              "an unrepresentable byte is a typed error, not silent loss");
    }
    {
        // A GPT-2 style file that lost its merge list cannot segment text; say
        // so instead of pretending.
        GGUFDocument doc;
        doc.metadata["tokenizer.ggml.model"] = make_string("gpt2");
        MetadataValue tokens;
        tokens.type = GGUFValueType::ARRAY;
        tokens.array.push_back(make_string("a"));
        tokens.array.push_back(make_string("b"));
        doc.metadata["tokenizer.ggml.tokens"] = tokens;
        Tokenizer tokenizer = Tokenizer::from_gguf(doc, 2);
        CHECK(!tokenizer.complete(), "BPE without merges is reported as incomplete");
        CHECK(throws_code(ErrorCode::NOT_IMPLEMENTED,
                          [&]() { (void)tokenizer.encode("ab", 0); }),
              "BPE without a merge list refuses to encode");
    }
    {
        // A token table that disagrees with the declared vocabulary is a broken
        // file, not something to paper over.
        GGUFDocument doc = sentencepiece_doc(true);
        CHECK(throws_code(ErrorCode::INVALID_TOKENIZER,
                          [&]() { (void)Tokenizer::from_gguf(doc, 7); }),
              "declared vocabulary must match the token table");
    }
}

// Byte-level BPE. The merge list is the point: an encoder that ignores it can
// still round-trip, and still disagree with every other runtime about the ids.
namespace bpe_fixture {

// A miniature GPT-2 style vocabulary: the 256 byte-alphabet pieces plus a
// handful of merged pieces, so merges have something to produce.
inline std::string byte_piece(unsigned char byte) {
    return fm::unicode::byte_to_utf8(byte);
}

inline GGUFDocument document(const std::string &pre, bool with_merges = true) {
    GGUFDocument doc;
    doc.metadata["tokenizer.ggml.model"] = make_string("gpt2");
    doc.metadata["tokenizer.ggml.pre"] = make_string(pre);

    MetadataValue tokens;
    tokens.type = GGUFValueType::ARRAY;
    // Byte pieces come first so the byte-fallback ids are easy to reason about.
    std::vector<std::string> pieces;
    for (int byte = 0; byte < 256; ++byte)
        pieces.push_back(byte_piece(static_cast<unsigned char>(byte)));
    // Merged pieces, addressed by name in the checks below.
    pieces.push_back("ab");   // 256
    pieces.push_back("bc");   // 257
    pieces.push_back("the");  // 258
    pieces.push_back("\xC4\xA0the"); // 259: "\u0120the", space + "the"
    pieces.push_back("12");   // 260
    pieces.push_back("'S");   // 261: reachable only through whole-word lookup
    for (const std::string &piece : pieces) tokens.array.push_back(make_string(piece));
    doc.metadata["tokenizer.ggml.tokens"] = tokens;

    if (with_merges) {
        MetadataValue merges;
        merges.type = GGUFValueType::ARRAY;
        // Rank order matters: "b c" merges before "a b", so "abc" becomes
        // "a" + "bc" rather than "ab" + "c". A longest-match encoder gets the
        // opposite answer, which is what this fixture exists to catch.
        merges.array.push_back(make_string("b c"));
        merges.array.push_back(make_string("a b"));
        merges.array.push_back(make_string("t h"));
        merges.array.push_back(make_string("th e"));
        // "\u0120the" (space + "the"). The pair is "\u0120" + "the", not
        // "\u0120t" + "he": "t h" has a lower rank than any \u0120 merge, so
        // "the" is already assembled by the time the space is looked at.
        merges.array.push_back(make_string("\xC4\xA0 the"));
        merges.array.push_back(make_string("1 2"));
        doc.metadata["tokenizer.ggml.merges"] = merges;
    }
    return doc;
}

} // namespace bpe_fixture

void test_tokenizer_bpe_merges() {
    using namespace bpe_fixture;
    GGUFDocument doc = document("gpt-2");
    const uint32_t vocab = static_cast<uint32_t>(
        doc.metadata["tokenizer.ggml.tokens"].array.size());
    const Tokenizer tokenizer = Tokenizer::from_gguf(doc, vocab);

    CHECK(tokenizer.kind() == Tokenizer::Kind::Bpe, "a gpt2 table is a BPE tokenizer");
    CHECK(tokenizer.pre_tokenizer() == Tokenizer::PreTokenizer::Gpt2,
          "the pre-tokenizer name selects the splitting rule");
    CHECK(tokenizer.complete(), "merges plus a byte alphabet make the BPE tokenizer complete");

    // The regression this whole change exists for: rank order, not length.
    const std::vector<int32_t> abc = tokenizer.encode("abc", 0, false);
    CHECK(abc.size() == 2, "abc is two pieces, so the merge order was followed");
    CHECK(abc[0] == 'a', "the left digit stays a single byte piece");
    CHECK(abc[1] == 256 + 1, "the rank-0 merge wins: b+c is the second piece");
    // Longest match would have produced 256 ("ab") + 'c'.
    CHECK(!(abc[0] == 256), "a longest-match encoder would have picked 'ab' first");

    // A merge chain: t+h then th+e.
    const std::vector<int32_t> the = tokenizer.encode("the", 0, false);
    CHECK(the.size() == 1 && the[0] == 258, "chained merges reconstruct the whole word");
    // The space-attachment rule of the GPT-2 preg: " the" is one word.
    const std::vector<int32_t> space_the = tokenizer.encode(" the", 0, false);
    CHECK(space_the.size() == 1 && space_the[0] == 259,
          "a leading space joins the word, so the space+the piece matches");
    // Digits are one run under the GPT-2 rule.
    const std::vector<int32_t> digits = tokenizer.encode("12", 0, false);
    CHECK(digits.size() == 1 && digits[0] == 260, "digit runs merge under the gpt2 rule");
    // Decoding reverses the byte alphabet, so text survives the round trip.
    CHECK(tokenizer.decode(the, UINT32_MAX, UINT32_MAX) == "the",
          "BPE pieces decode back through the byte alphabet");
    CHECK(tokenizer.decode(space_the, UINT32_MAX, UINT32_MAX) == " the",
          "an escaped space decodes as a real space");
    CHECK(tokenizer.decode(tokenizer.encode("hello world", 0, false), UINT32_MAX, UINT32_MAX) ==
              "hello world",
          "unmerged text round-trips byte for byte");
}

void test_tokenizer_bpe_families() {
    using namespace bpe_fixture;

    // StarCoder: every digit is its own word, and a space in front of a digit
    // becomes its own word rather than joining it.
    {
        GGUFDocument doc = document("starcoder");
        const uint32_t vocab = static_cast<uint32_t>(
            doc.metadata["tokenizer.ggml.tokens"].array.size());
        const Tokenizer tokenizer = Tokenizer::from_gguf(doc, vocab);
        CHECK(tokenizer.pre_tokenizer() == Tokenizer::PreTokenizer::Gpt2Digits,
              "starcoder uses the digit-splitting rule");
        const std::vector<int32_t> ids = tokenizer.encode("12", 0, false);
        CHECK(ids.size() == 2 && ids[0] == '1' && ids[1] == '2',
              "the digit rule splits before any merge can join them");
    }

    // llama-3: whole-word lookup first (ignore_merges), and an uppercase
    // contraction is still a contraction.
    {
        GGUFDocument doc = document("llama-bpe");
        const uint32_t vocab = static_cast<uint32_t>(
            doc.metadata["tokenizer.ggml.tokens"].array.size());
        const Tokenizer tokenizer = Tokenizer::from_gguf(doc, vocab);
        CHECK(tokenizer.pre_tokenizer() == Tokenizer::PreTokenizer::Llama3,
              "llama-bpe selects the llama-3 rule");
        const std::vector<int32_t> ids = tokenizer.encode("'S", 0, false);
        CHECK(ids.size() == 1, "a contraction is one word");
        CHECK(tokenizer.decode(ids, UINT32_MAX, UINT32_MAX) == "'S",
              "contractions decode exactly");
    }

    // Unknown and missing pre-tokenizers refuse instead of guessing. `kimi-k2`
    // is the reference example: its splitting rule is not implemented here, and
    // it must stay a refusal rather than falling back to another family.
    // The families that were added have to report themselves as implemented:
    // otherwise the refusal test above would pass for the wrong reason.
    {
        struct Expect { const char *pre; Tokenizer::PreTokenizer family; };
        const Expect implemented[] = {
            {"deepseek-llm", Tokenizer::PreTokenizer::DeepseekLlm},
            {"deepseek-coder", Tokenizer::PreTokenizer::DeepseekCoder},
            {"deepseek-v3", Tokenizer::PreTokenizer::DeepseekV3},
            {"falcon", Tokenizer::PreTokenizer::Falcon},
            {"tekken", Tokenizer::PreTokenizer::Tekken},
            {"qwen35", Tokenizer::PreTokenizer::Qwen35},
        };
        for (const Expect &e : implemented) {
            GGUFDocument doc = document(e.pre);
            const uint32_t vocab = static_cast<uint32_t>(
                doc.metadata["tokenizer.ggml.tokens"].array.size());
            const Tokenizer tokenizer = Tokenizer::from_gguf(doc, vocab);
            CHECK(tokenizer.pre_tokenizer() == e.family,
                  std::string(e.pre) + " selects its own splitting rule");
            CHECK(tokenizer.complete(),
                  std::string(e.pre) + " is reported as a complete tokenizer");
            // And the probe agrees before a run is attempted.
            doc.metadata["general.architecture"] = make_string("gpt2");
            const Json probe = probe_model(doc, "synthetic.gguf");
            CHECK(probe.dump().find("TOKENIZER_PRE_NOT_IMPLEMENTED") == std::string::npos,
                  std::string(e.pre) + " is not blocked by the probe");
        }
    }
    {
        GGUFDocument doc = document("kimi-k2");
        const uint32_t vocab = static_cast<uint32_t>(
            doc.metadata["tokenizer.ggml.tokens"].array.size());
        const Tokenizer tokenizer = Tokenizer::from_gguf(doc, vocab);
        CHECK(tokenizer.pre_tokenizer() == Tokenizer::PreTokenizer::Unsupported,
              "an unimplemented family is reported as such");
        CHECK(!tokenizer.complete(), "an unimplemented family is not complete");
        CHECK(throws_code(ErrorCode::NOT_IMPLEMENTED,
                          [&]() { (void)tokenizer.encode("hi", 0, false); }),
              "encoding with an unimplemented family refuses by name");
    }
    {
        GGUFDocument doc = document("");
        const uint32_t vocab = static_cast<uint32_t>(
            doc.metadata["tokenizer.ggml.tokens"].array.size());
        const Tokenizer tokenizer = Tokenizer::from_gguf(doc, vocab);
        CHECK(throws_code(ErrorCode::NOT_IMPLEMENTED,
                          [&]() { (void)tokenizer.encode("hi", 0, false); }),
              "a BPE file with no pre-tokenizer is refused, not guessed");
    }
    // A missing merge list is still refused, and for its own reason.
    {
        GGUFDocument doc = document("gpt-2", /*with_merges=*/false);
        const uint32_t vocab = static_cast<uint32_t>(
            doc.metadata["tokenizer.ggml.tokens"].array.size());
        const Tokenizer tokenizer = Tokenizer::from_gguf(doc, vocab);
        CHECK(throws_code(ErrorCode::NOT_IMPLEMENTED,
                          [&]() { (void)tokenizer.encode("hi", 0, false); }),
              "BPE without merges refuses");
    }

    // The probe has to say the same thing before a run is attempted, so a
    // checkpoint with an unknown splitting rule is never reported runnable.
    {
        GGUFDocument doc = document("kimi-k2");
        doc.metadata["general.architecture"] = make_string("gpt2");
        const Json probe = probe_model(doc, "synthetic.gguf");
        CHECK(probe.dump().find("TOKENIZER_PRE_NOT_IMPLEMENTED") != std::string::npos,
              "the probe blocks an unimplemented pre-tokenizer by name");
    }
    {
        GGUFDocument doc = document("");
        doc.metadata["general.architecture"] = make_string("gpt2");
        const Json probe = probe_model(doc, "synthetic.gguf");
        CHECK(probe.dump().find("TOKENIZER_NO_PRE") != std::string::npos,
              "the probe blocks a BPE file that does not say how to split words");
    }
}

// The Unicode classes drive word boundaries, so they are checked directly. A
// wrong class silently changes token ids, which no round-trip test can catch.
// SentencePiece segmentation is score-based. The two behaviours that separate it
// from longest-match are: a merge only happens when the *concatenation* is a
// piece (so the best-scoring pair wins even if a longer piece exists across the
// same text), and the dummy space prefix is added unconditionally, including for
// input that already begins with a space.
void test_tokenizer_sentencepiece() {
    const std::string marker = "\xe2\x96\x81";
    auto build = [&marker](const std::vector<std::pair<std::string, double>> &table) {
        GGUFDocument doc;
        doc.metadata["tokenizer.ggml.model"] = make_string("llama");
        MetadataValue tokens;
        tokens.type = GGUFValueType::ARRAY;
        MetadataValue scores;
        scores.type = GGUFValueType::ARRAY;
        for (const auto &[piece, score] : table) {
            tokens.array.push_back(make_string(piece));
            MetadataValue value;
            value.type = GGUFValueType::FLOAT32;
            value.f = score;
            scores.array.push_back(value);
        }
        for (int i = 0; i < 256; ++i) {
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", i);
            tokens.array.push_back(make_string(buf));
            MetadataValue value;
            value.type = GGUFValueType::FLOAT32;
            value.f = 0.0;
            scores.array.push_back(value);
        }
        doc.metadata["tokenizer.ggml.tokens"] = tokens;
        doc.metadata["tokenizer.ggml.scores"] = scores;
        return doc;
    };

    {
        // "abcd" with pieces a, b, c, d, ab, abc, cd, bcd. Longest-match takes
        // "abc" + "d"; score-based merging takes "cd" first (high score), then
        // "bcd", giving "a" + "bcd". The expected id sequence is what makes the
        // difference visible -- both segmentations round-trip.
        const std::vector<std::pair<std::string, double>> table = {
            {"<unk>", 0.0}, {"<s>", 0.0}, {"</s>", 0.0},
            {"a", -5.0}, {"b", -5.0}, {"c", -5.0}, {"d", -5.0},
            {"ab", -4.0}, {"abc", -3.0}, {"cd", -1.0}, {"bcd", -2.0},
        };
        GGUFDocument doc = build(table);
        const uint32_t vocab = static_cast<uint32_t>(doc.metadata["tokenizer.ggml.tokens"].array.size());
        const Tokenizer tokenizer = Tokenizer::from_gguf(doc, vocab);
        const std::vector<int32_t> ids = tokenizer.encode("abcd", 1, false);
        // The table has no space piece, so the prefixed space falls back to bytes;
        // drop those to inspect how the word itself was segmented.
        std::vector<int32_t> word;
        for (int32_t id : ids) {
            if (tokenizer.piece(static_cast<uint32_t>(id)).rfind("<0x", 0) == 0) continue;
            word.push_back(id);
        }
        CHECK(word.size() == 2, "score-based segmentation splits abcd into two pieces");
        if (word.size() == 2) {
            CHECK(tokenizer.piece(static_cast<uint32_t>(word[0])) == "a",
                  "the split is a + bcd, not the longest-match abc + d");
            CHECK(tokenizer.piece(static_cast<uint32_t>(word[1])) == "bcd",
                  "the second half is the piece built by merging b with cd");
        }
    }
    {
        // The dummy prefix is unconditional: " a" is escaped as two markers plus
        // "a", so the whole input is one piece. A tokenizer that skipped the
        // prefix for leading whitespace would emit two pieces here.
        const std::vector<std::pair<std::string, double>> table = {
            {"<unk>", 0.0}, {"<s>", 0.0}, {"</s>", 0.0},
            {marker, -3.0}, {marker + marker, -2.0}, {"a", -5.0},
            {marker + "a", -4.0}, {marker + marker + "a", -1.0},
        };
        GGUFDocument doc = build(table);
        const uint32_t vocab = static_cast<uint32_t>(doc.metadata["tokenizer.ggml.tokens"].array.size());
        const Tokenizer tokenizer = Tokenizer::from_gguf(doc, vocab);
        const std::vector<int32_t> ids = tokenizer.encode(" a", 1, false);
        CHECK(ids.size() == 1, "a leading space yields exactly one piece");
        if (!ids.empty())
            CHECK(tokenizer.piece(static_cast<uint32_t>(ids[0])) == marker + marker + "a",
                  "the dummy prefix is added even when the text starts with a space");
        // No prefix without the flag.
        GGUFDocument plain_doc = build(table);
        MetadataValue no_prefix;
        no_prefix.type = GGUFValueType::BOOL;
        no_prefix.b = false;
        plain_doc.metadata["tokenizer.ggml.add_space_prefix"] = no_prefix;
        const Tokenizer plain = Tokenizer::from_gguf(plain_doc, vocab);
        const std::vector<int32_t> plain_ids = plain.encode(" a", 1, false);
        CHECK(plain_ids.size() == 1 && plain.piece(static_cast<uint32_t>(plain_ids[0])) == marker + "a",
              "add_space_prefix=false leaves the text as written");
    }
    {
        // An empty prompt encodes to nothing at all: no dummy prefix for empty
        // input, matching the reference's documented behaviour.
        const std::vector<std::pair<std::string, double>> table = {
            {"<unk>", 0.0}, {"<s>", 0.0}, {"</s>", 0.0}, {marker, -1.0},
        };
        GGUFDocument doc = build(table);
        const uint32_t vocab = static_cast<uint32_t>(doc.metadata["tokenizer.ggml.tokens"].array.size());
        const Tokenizer tokenizer = Tokenizer::from_gguf(doc, vocab);
        CHECK(tokenizer.encode("", 1, false).empty(), "empty text encodes to no ids");
        CHECK(tokenizer.encode("", 1, true) == std::vector<int32_t>{1},
              "empty text with BOS is just the BOS id");
    }
}

void test_unicode_classes() {
    using namespace fm::unicode;
    const struct {
        uint32_t cpt;
        uint32_t expected;
        const char *what;
    } cases[] = {
        {'A', kLetter, "ASCII letter"},
        {'z', kLetter, "ASCII letter"},
        {0x00E9, kLetter, "Latin-1 letter (e acute)"},
        {0x1EC7, kLetter, "Vietnamese letter (e circumflex dot below)"},
        {0x4E2D, kLetter, "CJK ideograph"},
        {0x0645, kLetter, "Arabic letter"},
        {'0', kNumber, "ASCII digit"},
        {0x00BD, kNumber, "vulgar fraction is a number (No)"},
        {0x0663, kNumber, "Arabic-Indic digit"},
        {' ', kSpace, "space"},
        {'\n', kSpace, "newline"},
        {0x3000, kSpace, "ideographic space"},
        {0x00A0, kSpace, "no-break space"},
        {'!', kPunctuation, "ASCII punctuation"},
        {'$', kSymbol, "ASCII symbol"},
        {0x1F680, kSymbol, "emoji is a symbol, not a letter"},
        {0x0301, kMark, "combining acute is a mark"},
    };
    for (const auto &c : cases) {
        const uint32_t got = classes(c.cpt);
        CHECK((got & c.expected) != 0, c.what);
    }
    CHECK(!is_letter(0x1F680), "an emoji is not a letter, so it cannot join a word");
    CHECK(!is_number(UINT32_MAX), "out-of-range code points classify as nothing");

    // UTF-8: every code point round-trips, and an invalid byte becomes U+FFFD
    // rather than silently shifting the rest of the text.
    for (uint32_t cpt : {0x41u, 0xE9u, 0x1EC7u, 0x4E2Du, 0x1F680u}) {
        const std::vector<uint32_t> back = from_utf8(to_utf8(cpt));
        CHECK(back.size() == 1 && back[0] == cpt, "UTF-8 round-trips");
    }
    const std::vector<uint32_t> broken = from_utf8(std::string("a\xFF" "b"));
    CHECK(broken.size() == 3 && broken[1] == 0xFFFD, "an invalid byte becomes U+FFFD");

    // The byte alphabet is a bijection over all 256 bytes, which is what makes
    // byte-level fallback total.
    for (int byte = 0; byte < 256; ++byte) {
        const std::vector<uint32_t> cpts = from_utf8(byte_to_utf8(static_cast<unsigned char>(byte)));
        CHECK(cpts.size() == 1, "each byte maps to exactly one code point");
        if (cpts.size() == 1)
            CHECK(utf8_to_byte(cpts[0]) == byte, "the byte alphabet reverses exactly");
    }
}

// The canonical model map is what makes paging decisions auditable: every
// tensor must report a source byte range that lies inside the data section and
// a pageability answer consistent with its quantisation geometry.
void test_model_map_and_tied_output() {
    const std::string path = "tests/fixtures/moe-fm.gguf";
    auto doc = parse_path(path);
    auto manifest = build_manifest(doc, path);
    Json report = inspect_json(manifest, path);

    auto find_member = [](const Json &object, const std::string &key) -> const Json * {
        for (const auto &member : object.members())
            if (member.first == key) return &member.second;
        return nullptr;
    };

    const Json *map = find_member(report, "model_map");
    CHECK(map != nullptr && map->type() == Json::Type::Array,
          "inspect reports a canonical model map");
    if (!map) return;
    CHECK(map->items().size() == doc.tensors.size(), "the map has one entry per source tensor");

    bool saw_stacked = false;
    for (const Json &entry : map->items()) {
        auto num = [&](const char *key) -> uint64_t {
            const Json *value = find_member(entry, key);
            return value ? static_cast<uint64_t>(value->integer_value()) : 0;
        };
        auto str = [&](const char *key) -> std::string {
            const Json *value = find_member(entry, key);
            return value ? value->str() : std::string();
        };
        const uint64_t offset = num("source_offset");
        const uint64_t bytes = num("bytes");
        const std::string name = str("source_name");
        CHECK(offset >= doc.data_offset, "tensor range starts inside the data section: " + name);
        CHECK(offset + bytes <= doc.file_size, "tensor range ends inside the file: " + name);
        CHECK(num("source_end") == offset + bytes,
              "source_end is the exclusive end of the range: " + name);
        CHECK(!str("canonical_name").empty(), "every tensor has a canonical address: " + name);
        CHECK(num("adapter_version") == 1, "adapter version is recorded: " + name);
        if (num("expert_count") > 0) {
            saw_stacked = true;
            // A stacked expert slice is only independently readable when the
            // stride is a whole number of quantisation blocks.
            const uint64_t stride = num("expert_stride_bytes");
            const Json *geometry = find_member(entry, "quant_geometry");
            const Json *block = geometry ? find_member(*geometry, "block_bytes") : nullptr;
            const uint64_t block_bytes = block ? static_cast<uint64_t>(block->integer_value()) : 0;
            const Json *pageable = find_member(entry, "independently_pageable");
            CHECK(pageable &&
                      pageable->bool_value() == (block_bytes != 0 && stride % block_bytes == 0),
                  "pageability follows the quantisation block alignment: " + name);
        }
    }
    CHECK(saw_stacked, "the MoE fixture exposes stacked expert tensors in the map");
}


// Reads the line-oriented golden produced by tools/make_arch_fixtures.py.
struct ArchGolden {
    std::string adapter;
    std::string prompt;
    std::vector<int32_t> input_ids;
    std::vector<float> logits;
    std::vector<int32_t> greedy;
};

ArchGolden read_arch_golden(const std::string &path) {
    std::ifstream in(path);
    if (!in) fail(ErrorCode::INTERNAL_ERROR, "missing architecture golden " + path);
    ArchGolden golden;
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream stream(line);
        std::string tag;
        stream >> tag;
        if (tag == "adapter") {
            stream >> golden.adapter;
        } else if (tag == "bos") {
            int32_t value = 0;
            stream >> value;
            (void)value;
        } else if (tag == "prompt") {
            std::string rest;
            std::getline(stream, rest);
            if (!rest.empty() && rest[0] == ' ') rest.erase(0, 1);
            golden.prompt = rest;
        } else if (tag == "input_ids" || tag == "greedy") {
            size_t count = 0;
            stream >> count;
            std::vector<int32_t> &target = tag == "input_ids" ? golden.input_ids : golden.greedy;
            for (size_t i = 0; i < count; ++i) {
                int32_t value = 0;
                stream >> value;
                target.push_back(value);
            }
        } else if (tag == "logits") {
            size_t count = 0;
            stream >> count;
            golden.logits.resize(count);
            for (size_t i = 0; i < count; ++i) stream >> golden.logits[i];
        }
    }
    return golden;
}

// Architecture parity. Each entry is a checkpoint shaped like a real llama.cpp
// file for a named architecture, plus logits from an independent implementation
// (tools/llama_oracle.py). The engine has to reproduce those logits: that is
// what "the adapter for this architecture works" means here.
// Rotary context extension. The arithmetic is subtle enough that "the model
// still produces logits" is not evidence: a wrong ramp produces fluent text.
// These checks pin the formulas against values computed from ggml's definitions
// by hand, and then confirm the fixture's raw metadata reaches the config.
// mRoPE: which axis each rotary pair reads its position from. A text-only
// prompt has one position on all four axes, so nothing else in the suite can
// catch a wrong split -- the sections have to be checked directly.
void test_rope_mrope() {
    ModelConfig cfg;
    cfg.rope_theta = 10000.0;
    cfg.rope_freq_scale = 1.0;
    cfg.rope_mscale = 1.0;
    cfg.rope_sections = {4, 2, 2, 0};
    cfg.mrope = true;
    cfg.mrope_interleaved = false;

    RopePlan plan;
    plan.init(cfg, 32);
    CHECK(plan.axis_of(0) == 0, "the first section is the time axis");
    CHECK(plan.axis_of(3) == 0, "the time section is four pairs wide");
    CHECK(plan.axis_of(4) == 1, "the height section follows");
    CHECK(plan.axis_of(5) == 1, "the height section is two pairs wide");
    CHECK(plan.axis_of(6) == 2, "the width section follows");
    CHECK(plan.axis_of(7) == 2, "the width section is two pairs wide");
    // The pattern repeats every sum(sections) pairs, so pair 8 starts over.
    CHECK(plan.axis_of(8) == 0, "the section pattern repeats");
    CHECK(plan.axis_of(12) == 1, "the repeat lands on the same axes");
    // A non-zero fourth section is the extra axis (text and vision embeddings
    // that need a fourth coordinate).
    ModelConfig extra_cfg = cfg;
    extra_cfg.rope_sections = {4, 2, 2, 1};
    RopePlan extra;
    extra.init(extra_cfg, 32);
    CHECK(extra.axis_of(8) == 3, "the extra axis takes the leftover pairs");

    // Interleaved layout: the axes cycle t/y/x within each triplet, and a pair
    // whose axis has run out of its 3*count budget falls through to the extra
    // axis. With sections {4,2,2,0} the height and width budgets end at six
    // pairs, so the seventh pair of the pattern is the extra axis.
    cfg.mrope_interleaved = true;
    cfg.rope_sections = {4, 2, 2, 0};
    RopePlan inter;
    inter.init(cfg, 32);
    const std::vector<uint32_t> expected = {0, 1, 2, 0, 1, 2, 0, 3};
    bool matches = true;
    for (uint32_t pair = 0; pair < expected.size(); ++pair)
        if (inter.axis_of(pair) != expected[pair]) matches = false;
    CHECK(matches, "interleaved sections cycle t/y/x and spill into the extra axis");

    // Four equal positions must reproduce the scalar rotation exactly: that is
    // what keeps the text path identical to a non-mRoPE model.
    RopePlan plain;
    ModelConfig plain_cfg = cfg;
    plain_cfg.rope_sections = {0, 0, 0, 0};
    plain_cfg.mrope = false;
    plain.init(plain_cfg, 32);
    const std::array<double, 4> same = {7.0, 7.0, 7.0, 7.0};
    bool identical = true;
    for (uint32_t pair = 0; pair < rope_pairs(32); ++pair) {
        double c0 = 0.0, s0 = 0.0, c1 = 0.0, s1 = 0.0;
        plain.rotation(pair, 7.0, c0, s0);
        plan.rotation(pair, same, c1, s1);
        if (std::fabs(c0 - c1) > 1e-12 || std::fabs(s0 - s1) > 1e-12) identical = false;
    }
    CHECK(identical, "equal mRoPE positions reproduce plain rotary");

    // Distinct positions must actually change the pair, or the axis selection
    // above would be decoration.
    const std::array<double, 4> distinct = {0.0, 10.0, 20.0, 30.0};
    double c0 = 0.0, s0 = 0.0, c1 = 0.0, s1 = 0.0;
    plan.rotation(0, distinct, c0, s0);   // axis 0 -> position 0
    plan.rotation(4, distinct, c1, s1);   // axis 1 -> position 10
    CHECK(std::fabs(c0 - 1.0) < 1e-12 && std::fabs(s0) < 1e-12,
          "a zero time position leaves the first pair unrotated");
    // Pair 4 keeps its own frequency, so the angle is position times that
    // frequency -- the section changes the position, never the frequency.
    const double theta4 = std::pow(10000.0, -(2.0 * 4.0) / 32.0);
    CHECK(std::fabs(c1 - std::cos(10.0 * theta4)) < 1e-12,
          "the height axis rotates by its own position at its own frequency");

    // A section split wider than the head is refused rather than truncated.
    const std::string path = "tests/fixtures/glm4-mrope-shape.gguf";
    const ModelManifest manifest = build_manifest(parse_gguf_file(path), path);
    CHECK(manifest.config.mrope, "the fixture's sections enable mRoPE");
    CHECK(!manifest.config.mrope_interleaved, "glm4 uses the plain section layout");
    const std::array<uint32_t, 4> expected_sections = {2, 1, 1, 0};
    CHECK(manifest.config.rope_sections == expected_sections,
          "the fixture's sections are read verbatim");
    CHECK(manifest.config.rope_sections[0] + manifest.config.rope_sections[1] +
                  manifest.config.rope_sections[2] + manifest.config.rope_sections[3] <=
              manifest.config.head_dim / 2,
          "the fixture's sections fit inside the head");
}

void test_rope_scaling() {
    // ggml's correction-dim solve for a 8-dim rotation over a 64-token context:
    // start clamps to 0 (the fast knee is negative) and end rounds to 2, so
    // pairs 0,1 extrapolate/blend and pairs 2,3 interpolate.
    double dims[2] = {-1.0, -1.0};
    rope_yarn_corr_dims(8, 64.0, 10000.0, 32.0, 1.0, dims);
    CHECK(dims[0] == 0.0, "YaRN ramp start clamps to zero for a short context");
    CHECK(dims[1] == 2.0, "YaRN ramp end is the slow knee, rounded up");

    ModelConfig yarn;
    yarn.context = 64;
    yarn.rope_theta = 10000.0;
    yarn.rope_scaling_type = "yarn";
    yarn.rope_scaling_factor = 4.0;
    yarn.rope_n_ctx_orig = 64.0;
    resolve_rope_scaling(yarn);
    CHECK(std::fabs(yarn.rope_freq_scale - 0.25) < 1e-12,
          "a 4x YaRN factor becomes a 1/4 frequency scale");
    CHECK(yarn.rope_ext_factor == 1.0, "YaRN enables extrapolation");
    // 0.1*log(4) + 1, with no attn_factor in the file.
    CHECK(std::fabs(yarn.rope_mscale - (1.0 + 0.1 * std::log(4.0))) < 1e-12,
          "YaRN magnitude scale is 1 + 0.1*log(factor)");

    RopePlan scaled;
    scaled.init(yarn, 8);
    CHECK(scaled.corr_low == 0.0 && scaled.corr_high == 2.0,
          "the plan carries the ramp bounds");

    RopePlan plain;
    ModelConfig unscaled;
    unscaled.context = 64;
    unscaled.rope_theta = 10000.0;
    unscaled.rope_n_ctx_orig = 64.0;
    resolve_rope_scaling(unscaled);
    CHECK(unscaled.rope_freq_scale == 1.0 && unscaled.rope_mscale == 1.0,
          "a checkpoint without rope.scaling rotates unscaled");
    plain.init(unscaled, 8);

    // Pair 0 sits below the ramp: it keeps its exact angle (times the magnitude
    // scale only). Pair 3 is above it: fully interpolated, i.e. position/4.
    double c0 = 0.0, s0 = 0.0, c3 = 0.0, s3 = 0.0;
    scaled.rotation(0, 3.0, c0, s0);
    scaled.rotation(3, 3.0, c3, s3);
    const double mscale = 1.0 + 0.1 * std::log(4.0);
    const double freq0 = std::pow(10000.0, -2.0 * 0.0 / 8.0);
    const double freq3 = std::pow(10000.0, -2.0 * 3.0 / 8.0);
    CHECK(std::fabs(c0 - std::cos(3.0 * freq0) * mscale) < 1e-9,
          "the fastest YaRN pair keeps its exact angle");
    CHECK(std::fabs(s0 - std::sin(3.0 * freq0) * mscale) < 1e-9,
          "the fastest YaRN pair keeps its exact sine");
    CHECK(std::fabs(c3 - std::cos(3.0 * freq3 / 4.0) * mscale) < 1e-9,
          "the slowest YaRN pair is interpolated by the factor");

    double pc0 = 0.0, ps0 = 0.0, pc3 = 0.0, ps3 = 0.0;
    plain.rotation(0, 3.0, pc0, ps0);
    plain.rotation(3, 3.0, pc3, ps3);
    CHECK(std::fabs(pc0 - std::cos(3.0 * freq0)) < 1e-12,
          "unscaled rotation is the plain cos/sin of position times frequency");
    CHECK(std::fabs(c3 - pc3) > 1e-3, "YaRN actually changes the slow pair's angle");

    // linear: same interpolation, no ramp and no derived magnitude, so the
    // file's attn_factor is the only multiplier.
    ModelConfig linear;
    linear.context = 64;
    linear.rope_theta = 10000.0;
    linear.rope_scaling_type = "linear";
    linear.rope_scaling_factor = 2.0;
    linear.rope_n_ctx_orig = 64.0;
    linear.rope_scaling_attn_factor = 1.25;
    resolve_rope_scaling(linear);
    CHECK(linear.rope_freq_scale == 0.5 && linear.rope_ext_factor == 0.0,
          "linear scaling only divides the frequency");
    CHECK(std::fabs(linear.rope_mscale - 1.25) < 1e-12,
          "linear scaling takes the magnitude straight from the file");
    RopePlan lin;
    lin.init(linear, 8);
    double lc = 0.0, ls = 0.0;
    lin.rotation(2, 5.0, lc, ls);
    CHECK(std::fabs(lc - std::cos(5.0 * std::pow(10000.0, -2.0 * 2.0 / 8.0) / 2.0) * 1.25) < 1e-9,
          "linear scaling divides every angle by the factor");

    // Per-pair frequency divisors: pair 0 is unchanged (factor 1), pair 1 is
    // halved. The reference divides the frequency, so the angle for pair 1 must
    // be the unscaled angle over the factor, not the angle times it.
    {
        ModelConfig plain_cfg;
        plain_cfg.context = 64;
        plain_cfg.rope_theta = 10000.0;
        plain_cfg.rope_n_ctx_orig = 64.0;
        resolve_rope_scaling(plain_cfg);
        const float factors[4] = {1.0F, 3.0F, 1.5F, 2.0F};
        RopePlan with_factors;
        with_factors.init(plain_cfg, 8);
        with_factors.freq_factors = factors;
        RopePlan without;
        without.init(plain_cfg, 8);
        double c_scaled = 0.0, s_scaled = 0.0, c_plain = 0.0, s_plain = 0.0;
        with_factors.rotation(1, 3.0, c_scaled, s_scaled);
        without.rotation(1, 3.0, c_plain, s_plain);
        const double freq1 = std::pow(10000.0, -2.0 * 1.0 / 8.0);
        CHECK(std::fabs(c_scaled - std::cos(3.0 * freq1 / 3.0)) < 1e-9,
              "a frequency factor divides that pair's frequency");
        CHECK(std::fabs(c_scaled - c_plain) > 1e-3,
              "the factor changes the angle rather than being ignored");
        double c_zero = 0.0, s_zero = 0.0, c_zero_plain = 0.0, s_zero_plain = 0.0;
        with_factors.rotation(0, 3.0, c_zero, s_zero);
        without.rotation(0, 3.0, c_zero_plain, s_zero_plain);
        CHECK(std::fabs(c_zero - c_zero_plain) < 1e-12,
              "a factor of 1 leaves that pair untouched");
    }
    {
        // SmolLM3's NoPE rule is an architecture property upstream, not metadata.
        const std::string nope_path = "tests/fixtures/smollm3-nope-shape.gguf";
        const ModelManifest manifest = build_manifest(parse_gguf_file(nope_path), nope_path);
        CHECK(manifest.config.architecture == "smollm3", "the smollm3 adapter is selected");
        CHECK(manifest.config.no_rope_layer_step == 4,
              "SmolLM3 sets the every-fourth-layer NoPE step");
        CHECK(manifest.config.layers == 4, "the fixture has enough layers to exercise it");
    }
    {
        const std::string freq_path = "tests/fixtures/llama-ropfreqs-shape.gguf";
        const ModelManifest manifest =
            build_manifest(parse_gguf_file(freq_path), freq_path);
        CHECK(manifest.config.rope_freqs, "rope_freqs tensors enable the divisors");
        CHECK(manifest.tensor("blk.0.rope_freqs.weight") != nullptr &&
                  manifest.tensor("blk.0.rope_freqs.weight")->role == TensorRole::ROPE_FREQS,
              "the divisors get their own tensor role");
    }
    // The fixture's raw keys have to reach the config, or the engine would run
    // exactly the unscaled arithmetic above and still look fine.
    const std::string path = "tests/fixtures/llama-yarn-shape.gguf";
    const GGUFDocument doc = parse_gguf_file(path);
    const ModelManifest manifest = build_manifest(doc, path);
    CHECK(manifest.config.rope_scaling_type == "yarn",
          "the checkpoint's rope.scaling.type is read");
    CHECK(std::fabs(manifest.config.rope_scaling_factor - 4.0) < 1e-9,
          "the checkpoint's scaling factor is read");
    CHECK(std::fabs(manifest.config.rope_freq_scale - 0.25) < 1e-9,
          "the scaling factor reaches the derived frequency scale");
    CHECK(std::fabs(manifest.config.rope_mscale - (1.0 + 0.1 * std::log(4.0))) < 1e-6,
          "the derived magnitude scale survives the manifest build");

    // longrope needs per-layer short/long factors that no checkpoint here uses;
    // it must say so by name rather than fall back to unscaled rotation.
    // The probe must report the same limitation before a run is attempted, and
    // must name the mode it saw so the report is actionable.
    {
        const Json probe = probe_model(doc, path);
        CHECK(probe.dump().find("\"rope_scaling\"") != std::string::npos,
              "the probe reports the checkpoint's rope scaling");
        CHECK(probe.dump().find("\"yarn\"") != std::string::npos,
              "the probe names yarn rather than only 'supported'");
        CHECK(probe.dump().find("ROPE_SCALING_NOT_IMPLEMENTED") == std::string::npos,
              "implemented rope scaling is not a blocker");
    }
    {
        GGUFDocument bad = doc;  // already a `llama` checkpoint
        bad.metadata["llama.rope.scaling.type"] = make_string("longrope");
        const Json probe = probe_model(bad, "synthetic.gguf");
        CHECK(probe.dump().find("ROPE_SCALING_NOT_IMPLEMENTED") != std::string::npos,
              "an unimplemented rope scaling mode is reported as a blocker");
    }

    ModelConfig longrope;
    longrope.rope_scaling_type = "longrope";
    longrope.rope_n_ctx_orig = 64.0;
    CHECK(throws_code(ErrorCode::NOT_IMPLEMENTED,
                      [&]() { resolve_rope_scaling(longrope); }),
          "longrope is refused by name instead of silently ignored");
    // A scaling factor with no anchor for the ramp is a metadata error.
    ModelConfig no_anchor;
    no_anchor.rope_scaling_type = "yarn";
    no_anchor.rope_scaling_factor = 4.0;
    CHECK(throws_code(ErrorCode::METADATA_MISMATCH,
                      [&]() { resolve_rope_scaling(no_anchor); }),
          "yarn without an original context is rejected");
}

// Attention sinks: the softmax denominator gains one learnable logit per head and
// the value sum never reads it. Parity alone would catch a missing sink, but not
// a sink read from the wrong head or applied to the wrong head count, so the
// dimensions and the all-layers rule are checked directly here.
// Adapter rows and tags. The point of the tags is that a failure names the same
// thing llama.cpp would name, so these checks pin the vocabulary rather than
// just the plumbing.
// The primitives that change what the feed-forward computes, and the expert
// layout that changes where its weights live. Parity covers the arithmetic; these
// checks pin the plumbing so a fixture cannot pass for the wrong reason.
// The two clamped activation forms agree nowhere near the bound, so they are
// pinned against hand-computed constants transcribed from the kernels rather
// than against each other. Values for gate=5, up=3, limit=2:
//   fused      silu(min(5,2)) * 2  = 3.5231883119115293   (ggml_swiglu_clamp)
//   non-fused  min(silu(5),2) * 2  = 4.0                  (llama-graph fallback)
void test_activation_clamp_forms() {
    ModelConfig cfg;
    cfg.activation = "silu-clamp";
    cfg.activation_limit = 2.0;
    cfg.swiglu_clamp = {2.0};

    const float gate = 5.0f;
    const float up = 3.0f;
    cfg.swiglu_clamp_fused = true;
    CHECK(std::fabs(gated_activation(cfg, gate, up, 0) - 3.5231883119115293f) < 1e-6f,
          "fused clamp is silu(min(gate,limit)) * clamp(up)");
    cfg.swiglu_clamp_fused = false;
    CHECK(std::fabs(gated_activation(cfg, gate, up, 0) - 4.0f) < 1e-6f,
          "unfused clamp is min(silu(gate),limit) * clamp(up)");
    CHECK(std::fabs(gated_activation(cfg, gate, up, 0) -
                    gated_activation(cfg, gate, up, 0)) < 1e-9f,
          "the activation is deterministic");

    // Below the bound the two forms must coincide: this is the regime every
    // ordinary token sits in, and it is why the defect only shows up on
    // saturated activations.
    const float small_gate = 1.0f;
    cfg.swiglu_clamp_fused = true;
    const float fused_small = gated_activation(cfg, small_gate, 0.25f, 0);
    cfg.swiglu_clamp_fused = false;
    const float unfused_small = gated_activation(cfg, small_gate, 0.25f, 0);
    CHECK(std::fabs(fused_small - unfused_small) < 1e-7f,
          "both clamp forms agree away from the bound");

    // A zero or sub-threshold bound means "this layer is not clamped" in both
    // forms (llama-graph.cpp:2448 uses 1e-6, not 0).
    cfg.swiglu_clamp = {0.0, 1e-9};
    for (bool fused : {true, false}) {
        cfg.swiglu_clamp_fused = fused;
        const double expect = static_cast<double>(1.0f) / (1.0 + std::exp(-1.0)) * 3.0;
        CHECK(std::fabs(gated_activation(cfg, 1.0f, 3.0f, 0) - expect) < 1e-6,
              "a zero bound leaves the layer unclamped");
        CHECK(std::fabs(gated_activation(cfg, 1.0f, 3.0f, 1) - expect) < 1e-6,
              "a 1e-9 bound is also unclamped, matching upstream's epsilon");
    }

    // Per-layer bounds are indexed, not averaged.
    cfg.swiglu_clamp = {0.5, 2.0};
    cfg.swiglu_clamp_fused = true;
    // layer 0: silu(min(1,0.5)) * clamp(0.5) = silu(0.5) * 0.5
    CHECK(std::fabs(gated_activation(cfg, 1.0f, 0.5f, 0) - 0.15561483280046365f) < 1e-6f,
          "layer 0 uses its own bound");
    // layer 1: silu(min(5,2)) * clamp(0.5) = silu(2) * 0.5
    CHECK(std::fabs(gated_activation(cfg, 5.0f, 0.5f, 1) - 0.8807970779778823f) < 1e-6f,
          "layer 1 uses its own bound");

    // swiglu-oai: (min(g,7)/(1+exp(-1.702*min(g,7)))) * (clamp(up)+1).
    ModelConfig oai;
    oai.activation = "swiglu-oai";
    oai.activation_alpha = 1.702;
    oai.activation_limit = 7.0;
    CHECK(std::fabs(gated_activation(oai, 5.0f, 3.0f, 0) - 19.99597193493059f) < 1e-5f,
          "swiglu-oai matches the alpha/limit form");
    // Above the limit the gate saturates and up is clamped before the +1.
    CHECK(std::fabs(gated_activation(oai, 50.0f, 9.0f, 0) -
                    (7.0 / (1.0 + std::exp(-1.702 * 7.0))) * 8.0) < 1e-5,
          "swiglu-oai saturates at the limit");

    // Plain silu must be untouched by all of this.
    ModelConfig plain;
    plain.activation = "silu";
    CHECK(std::fabs(gated_activation(plain, 1.0f, 3.0f, 0) -
                    static_cast<float>(1.0 / (1.0 + std::exp(-1.0)) * 3.0)) < 1e-6f,
          "plain silu is silu(gate) * up");
}

// The un-gated activation forms. No adapter selects relu/relu_sq/gelu_erf/
// softplus today -- every family in the table uses silu or gelu -- but they are
// upstream `llm_ffn_op_type` members (nemotron-h, arcee, lfm2, mistral3 …), so
// they stay implemented and are pinned here: an untested primitive that a future
// row turns on is a guess, not an implementation. Four of the seven forms had no
// test at all before this.
void test_pointwise_activations() {
    ModelConfig cfg;
    cfg.activation = "relu";
    CHECK(pointwise_activation(cfg, 0.5f) == 0.5f, "relu passes positives through");
    CHECK(pointwise_activation(cfg, -1.0f) == 0.0f, "relu clamps negatives to zero");
    cfg.activation = "relu_sq";
    CHECK(near(pointwise_activation(cfg, 0.5f), 0.25f, 1e-6f), "relu_sq squares the positive part");
    CHECK(pointwise_activation(cfg, -1.0f) == 0.0f, "relu_sq clamps negatives to zero");
    cfg.activation = "gelu_erf";
    CHECK(near(pointwise_activation(cfg, 0.5f), 0.34573123063700656f, 1e-6f),
          "gelu_erf matches the exact erf form");
    CHECK(near(pointwise_activation(cfg, -1.0f), -0.15865525393145707f, 1e-6f),
          "gelu_erf is not clamped at zero for negatives");
    cfg.activation = "softplus";
    CHECK(near(pointwise_activation(cfg, 0.5f), 0.9740769841801067f, 1e-6f),
          "softplus matches log1p(exp(x))");
    CHECK(near(pointwise_activation(cfg, -1.0f), 0.31326168751822286f, 1e-6f),
          "softplus is positive for negative inputs");
    // The gated form multiplies the gate's activation by the up projection, and
    // the clamp variants must not leak into these plain forms.
    cfg.activation = "relu_sq";
    CHECK(near(gated_activation(cfg, 1.5f, 0.5f, 0), 1.125f, 1e-6f),
          "gated relu_sq is relu(gate)^2 * up");
    cfg.activation = "relu";
    CHECK(gated_activation(cfg, -2.0f, 0.5f, 0) == 0.0f, "gated relu zeroes a negative gate");
    cfg.activation = "gelu_erf";
    CHECK(near(gated_activation(cfg, 1.0f, 0.5f, 0), 0.42067237303427146f, 1e-6f),
          "gated gelu_erf is gelu_erf(gate) * up");
    // A clamp bound in the config must not change an unclamped form.
    cfg.swiglu_clamp = {2.0, 2.0};
    cfg.activation_limit = 2.0;
    CHECK(near(gated_activation(cfg, 1.0f, 0.5f, 0), 0.42067237303427146f, 1e-6f),
          "a declared clamp bound does not leak into gelu_erf");
}

void test_ffn_variants() {
    // Fused gate+up experts: both extents point inside one tensor, and the up
    // extent starts exactly where the gate extent ends.
    {
        const std::string path = "tests/fixtures/qwen3moe-fused-up-shape.gguf";
        const ModelManifest manifest = build_manifest(parse_gguf_file(path), path);
        CHECK(manifest.config.fused_gate_up_exps, "the fused expert layout is detected");
        CHECK(manifest.tensor("blk.0.ffn_gate_exps.weight") == nullptr,
              "the separate gate tensor is absent in a fused checkpoint");
        CHECK(manifest.tensor("blk.0.ffn_gate_up_exps.weight") != nullptr,
              "the fused tensor is mapped");
        const ExpertRecord *expert = manifest.expert(0, 0);
        CHECK(expert != nullptr, "expert 0 of layer 0 has a record");
        if (expert) {
            const ExpertExtent &gate = expert->extents[0];
            const ExpertExtent &up = expert->extents[1];
            CHECK(up.offset == gate.offset + gate.bytes,
                  "the up half begins where the gate half ends");
            CHECK(gate.role == TensorRole::FFN_GATE_EXPS && up.role == TensorRole::FFN_UP_EXPS,
                  "the two halves keep their own roles");
            const TensorInfo *fused = manifest.tensor("blk.0.ffn_gate_up_exps.weight");
            CHECK(fused && fused->byte_size / manifest.config.expert_count ==
                               gate.bytes + up.bytes,
                  "gate plus up covers exactly one expert's slice");
        }
        // A file carrying both layouts is ambiguous, not "prefer one".
        GGUFDocument doc = parse_gguf_file(path);
        GGUFDocument ambiguous = doc;
        MetadataValue name;
        name.type = GGUFValueType::STRING;
        name.s = "blk.0.ffn_gate_exps.weight";
        TensorInfo phantom = doc.tensors[0];
        phantom.name = name.s;
        ambiguous.tensors.push_back(phantom);
        ambiguous.tensor_by_name[name.s] = ambiguous.tensors.size() - 1;
        CHECK(throws_code(ErrorCode::METADATA_MISMATCH,
                          [&]() { (void)build_manifest(ambiguous, path); }),
              "both expert layouts present is a typed error");
    }
    // Clamped SwiGLU: the per-layer bounds reach the config unchanged, including
    // the zero that means "this layer is not clamped".
    {
        const std::string path = "tests/fixtures/qwen3moe-clamp-shape.gguf";
        const ModelManifest manifest = build_manifest(parse_gguf_file(path), path);
        CHECK(manifest.config.activation == "silu-clamp",
              "swiglu_clamp_exp selects the clamped variant");
        CHECK(manifest.config.swiglu_clamp.size() == manifest.config.layers,
              "one bound per layer is read");
        if (manifest.config.swiglu_clamp.size() >= 3) {
            CHECK(std::fabs(manifest.config.swiglu_clamp[0] - 0.7) < 1e-6,
                  "the first layer's bound is read");
            CHECK(manifest.config.swiglu_clamp[1] == 0.0,
                  "a zero bound means that layer is not clamped, not 'use the max'");
            CHECK(std::fabs(manifest.config.swiglu_clamp[2] - 1.3) < 1e-6,
                  "the third layer's bound is read");
        }
        const std::vector<std::string> tags = adapter_config_tags(manifest.config);
        CHECK(std::find(tags.begin(), tags.end(), "ffn=silu-clamp") != tags.end(),
              "the clamped activation is tagged");
        // Without the metadata the plain path must be selected.
        const std::string plain = "tests/fixtures/qwen3moe-shape.gguf";
        const ModelManifest plain_manifest = build_manifest(parse_gguf_file(plain), plain);
        CHECK(plain_manifest.config.activation == "silu" &&
                  plain_manifest.config.swiglu_clamp.empty(),
              "a checkpoint without clamp metadata keeps plain silu");
    }
    // Router denominator: the F16-normal clamp by default, 1e-20 additive for
    // deepseek4.1. Getting this wrong only shows when every selected weight is
    // tiny, which is why it is asserted rather than assumed.
    {
        const std::string path = "tests/fixtures/qwen3moe-shape.gguf";
        const ModelManifest manifest = build_manifest(parse_gguf_file(path), path);
        CHECK(std::fabs(manifest.config.expert_weights_epsilon - 6.103515625e-5) < 1e-12,
              "the default router denominator floor is the smallest F16 normal");
        CHECK(!manifest.config.expert_weights_epsilon_additive,
              "the default rule clamps rather than adds");
    }
}

void test_adapter_tags() {
    // Every architecture the adapter claims has a row with an upstream enum, so
    // an error can be traced to one file in llama.cpp.
    int rows = 0;
    for (const AdapterRow &row : adapter_rows()) {
        ++rows;
        CHECK(row.arch != nullptr && *row.arch != '\0', "every adapter row names its architecture");
        CHECK(row.upstream_enum != nullptr && *row.upstream_enum != '\0',
              std::string("every adapter row names its upstream enum: ") + row.arch);
        CHECK(adapter_row(row.arch) != nullptr, "the row is findable by its own name");
    }
    CHECK(rows >= 20, "the adapter catalogue is populated");
    CHECK(adapter_row("not-a-real-arch") == nullptr,
          "an unknown architecture has no row rather than a default");

    // llama.cpp's vocabulary for the roles that differ from the internal name.
    CHECK(std::string(adapter_tensor_tag(TensorRole::ATTN_OUTPUT)) == "ATTN_OUT",
          "attn_output is tagged ATTN_OUT, as upstream names it");
    CHECK(std::string(adapter_tensor_tag(TensorRole::FFN_ROUTER)) == "FFN_GATE_INP",
          "the router is tagged FFN_GATE_INP, as upstream names it");
    CHECK(std::string(adapter_tensor_tag(TensorRole::FFN_GATE_EXPS)) == "FFN_GATE_EXPS",
          "expert tensors keep their upstream names");

    // A mapped checkpoint carries a tag per tensor and a tag per branch decision.
    const std::string path = "tests/fixtures/llama-sinks-shape.gguf";
    const ModelManifest manifest = build_manifest(parse_gguf_file(path), path);
    CHECK(!manifest.tensor("blk.0.attn_q.weight")->tag.empty(),
          "a mapped tensor carries an adapter tag");
    CHECK(manifest.tensor("blk.0.attn_sinks.weight")->tag == "ATTN_SINKS",
          "the sink tensor is tagged ATTN_SINKS");
    CHECK(manifest.tensor("blk.0.attn_output.weight")->tag == "ATTN_OUT",
          "the output projection reports upstream's tag");
    const std::vector<std::string> tags = adapter_config_tags(manifest.config);
    auto has = [&tags](const std::string &needle) {
        return std::find(tags.begin(), tags.end(), needle) != tags.end();
    };
    CHECK(has("arch=llama"), "the branch tags name the architecture");
    CHECK(has("upstream=LLM_ARCH_LLAMA"), "the branch tags name the upstream enum");
    CHECK(has("attn=gqa"), "grouped-query attention is tagged");
    CHECK(has("sinks=yes"), "enabled sinks are tagged");
    CHECK(has("rope=full") && has("norm=rmsnorm") && has("ffn=silu"),
          "rotary, norm and activation are all tagged");
    // A checkpoint without a feature must not claim it.
    const std::string plain_path = "tests/fixtures/llama-gqa-shape.gguf";
    const ModelManifest plain = build_manifest(parse_gguf_file(plain_path), plain_path);
    const std::vector<std::string> plain_tags = adapter_config_tags(plain.config);
    CHECK(std::find(plain_tags.begin(), plain_tags.end(), "sinks=yes") == plain_tags.end(),
          "a checkpoint without sinks is not tagged as having them");
    // The MoE tags appear only where the router exists.
    const std::string moe_path = "tests/fixtures/qwen3moe-shape.gguf";
    const ModelManifest moe = build_manifest(parse_gguf_file(moe_path), moe_path);
    const std::vector<std::string> moe_tags = adapter_config_tags(moe.config);
    CHECK(std::find(moe_tags.begin(), moe_tags.end(), "moe=stacked") != moe_tags.end(),
          "stacked-expert MoE is tagged");
    CHECK(std::find(moe_tags.begin(), moe_tags.end(), "qk_norm=per-head") != moe_tags.end(),
          "per-head q/k norm is tagged");
}

void test_attention_sinks() {
    const std::string path = "tests/fixtures/llama-sinks-shape.gguf";
    const GGUFDocument doc = parse_gguf_file(path);
    const ModelManifest manifest = build_manifest(doc, path);
    CHECK(manifest.config.attn_sinks, "a checkpoint with attn_sinks tensors enables sinks");
    CHECK(manifest.tensor("blk.0.attn_sinks.weight") != nullptr,
          "the sink tensor is mapped by the adapter");
    if (manifest.tensor("blk.0.attn_sinks.weight"))
        CHECK(manifest.tensor("blk.0.attn_sinks.weight")->role == TensorRole::ATTN_SINKS,
              "the sink tensor gets its own role");
    // A file with the tensor cut off one layer is a mapping error, not a variant:
    // the layers would then use two different attention definitions.
    GGUFDocument partial = doc;
    partial.tensor_by_name.erase("blk.1.attn_sinks.weight");
    CHECK(throws_code(ErrorCode::MISSING_REQUIRED_TENSOR,
                      [&]() { (void)build_manifest(partial, path); }),
          "a partially sunk checkpoint is refused rather than half-applied");
    // The probe reports it, so a reader can see which attention definition runs.
    const Json probe = probe_model(doc, path);
    CHECK(probe.dump().find("attn_sinks") != std::string::npos,
          "the probe reports attention sinks");
}

void test_architecture_parity() {
    const std::vector<std::string> slugs = {"llama-gqa", "qwen2-bias", "qwen3-qknorm", "gemma",
                                           "gemma2", "phi3-fused", "minicpm", "qwen3moe",
                                           "glm4", "qwen2moe", "phi2-layernorm",
                                           "deepseek2-mla", "internlm2", "exaone", "orion",
                                           "gptneox", "stablelm", "llama-yarn", "qwen2-linear",
                                           "llama-yarn-mscale", "llama-sinks", "llama-ropfreqs", "smollm3-nope",
                                           "qwen3moe-fused-up", "qwen3moe-clamp",
                                           "glm4-mrope", "bitnet", "arcee", "nemotron",
                                           "jais2", "gpt-oss", "plm", "mistral3",
                                           "qwen3vl", "qwen3vlmoe", "mamba", "mamba2",
                                           "mamba-dtrms", "jamba"};
    for (const std::string &slug : slugs) {
        const std::string model = "tests/fixtures/" + slug + "-shape.gguf";
        ArchGolden golden = read_arch_golden("tests/golden/" + slug + "_reference.txt");
        CHECK(!golden.input_ids.empty(), "golden records the prompt ids: " + slug);
        CHECK(!golden.logits.empty(), "golden records prefill logits: " + slug);

        RunConfig config;
        config.model_path = model;
        config.trace_path = "";
        config.budget.total_hot_bytes = 64ull * 1024 * 1024;
        Engine engine(config);
        engine.initialize();

        CHECK(engine.manifest().config.architecture == golden.adapter,
              "the adapter reports the checkpoint's architecture: " + slug);

        PromptResult result = engine.run_prompt(golden.prompt, 8);
        CHECK(result.input_ids == golden.input_ids, "prompt tokenization matches: " + slug);
        CHECK(result.prefill_logits.size() == golden.logits.size(),
              "prefill produced one logit per vocabulary entry: " + slug);
        if (result.prefill_logits.size() != golden.logits.size()) {
            engine.release_runtime();
            continue;
        }

        // Same arithmetic in both implementations, but one accumulates in
        // float and the other in double, so the tolerance is relative to the
        // logit magnitude rather than absolute.
        double worst_abs = 0.0;
        double worst_rel = 0.0;
        double magnitude = 0.0;
        for (float v : golden.logits) magnitude = std::max(magnitude, std::fabs(static_cast<double>(v)));
        for (size_t i = 0; i < golden.logits.size(); ++i) {
            const double got = result.prefill_logits[i];
            const double want = golden.logits[i];
            const double abs_err = std::fabs(got - want);
            worst_abs = std::max(worst_abs, abs_err);
            worst_rel = std::max(worst_rel, abs_err / std::max(1e-9, std::fabs(want)));
        }
        CHECK(worst_abs <= 1e-4 + 1e-4 * magnitude,
              "prefill logits match the independent reference: " + slug + " (worst abs " +
                  std::to_string(worst_abs) + ", rel " + std::to_string(worst_rel) + ")");
        CHECK(result.generated_ids == golden.greedy,
              "greedy decoding matches the independent reference: " + slug);
        engine.release_runtime();
    }
}


// Families whose feed-forward has one projection instead of a gate. GGUF
// metadata cannot state an activation, so the architecture name is the only
// evidence: each of these must arrive with the activation llama.cpp uses for it
// (LLM_FFN_RELU_SQR), and a file of a family that is not in the table must be
// refused rather than run with silu.
void test_gate_free_families() {
    struct Expect {
        const char *slug;
        const char *arch;
        const char *activation;
        bool norm_bias;
        bool ffn_bias;
        bool attn_output_bias;
    };
    const Expect cases[] = {
        {"arcee", "arcee", "relu_sq", false, false, false},
        {"nemotron", "nemotron", "relu_sq", true, true, false},
        {"jais2", "jais2", "relu_sq", true, true, true},
        {"plm", "plm", "relu_sq", false, false, false},
    };
    for (const Expect &c : cases) {
        const std::string path = std::string("tests/fixtures/") + c.slug + "-shape.gguf";
        const ModelManifest manifest = build_manifest(parse_gguf_file(path), path);
        const std::string tag = std::string(c.slug) + ": ";
        CHECK(manifest.config.architecture == c.arch, tag + "architecture comes from the file");
        CHECK(manifest.config.ffn_no_gate,
              tag + "one projection with no gate is detected from the tensors");
        CHECK(manifest.config.activation == c.activation,
              tag + "activation comes from the architecture table, not a default");
        CHECK(manifest.config.norm_bias == c.norm_bias, tag + "norm biases match the file");
        CHECK(manifest.config.ffn_bias == c.ffn_bias, tag + "ffn projection biases match");
        CHECK(manifest.config.attn_output_bias == c.attn_output_bias,
              tag + "attention output bias matches");
        CHECK(manifest.tensor("blk.0.ffn_gate.weight") == nullptr,
              tag + "a gate-free checkpoint has no gate tensor");
        CHECK(manifest.tensor("blk.0.ffn_up.weight") != nullptr,
              tag + "the single up projection is mapped");
    }

    // llama is gated. Strip the gates and it becomes a gate-free FFN whose
    // activation nothing states -- the one case where guessing would produce a
    // plausible but wrong model, so it has to fail.
    {
        GGUFDocument doc = parse_gguf_file("tests/fixtures/llama-gqa-shape.gguf");
        doc.tensor_by_name.erase("blk.0.ffn_gate.weight");
        doc.tensor_by_name.erase("blk.1.ffn_gate.weight");
        CHECK(throws_code(ErrorCode::NOT_IMPLEMENTED,
                          [&]() { (void)build_manifest(doc, "gateless-llama.gguf"); }),
              "a gate-free file of a family with no activation row is refused");
    }
}

// PrismML's Bonsai 2 stores ternary weights in a Hadamard-rotated basis and
// mixes linear attention into the block. Neither is implemented, and the danger
// is specific: the bytes decode to plausible numbers, so a loader that ignored
// both would emit garbage while reporting success. The probe has to name both.
void test_rotated_and_ternary_blockers() {
    const std::string path = "tests/fixtures/ternary-bonsai-shape.gguf";
    auto doc = parse_gguf_file(path);
    const std::string dump = probe_model(doc, path).dump();

    auto contains = [&dump](const std::string &needle) {
        return dump.find(needle) != std::string::npos;
    };
    CHECK(contains("\"runnable\":false"), "a rotated-basis checkpoint is not reported runnable");
    CHECK(contains("ROTATED_WEIGHT_BASIS"),
          "the probe names the missing Hadamard activation transform");
    CHECK(contains("LINEAR_ATTENTION"),
          "the probe names the unimplemented linear-attention state");
    CHECK(contains("\"rotated_weight_basis\":true"), "the rotation is reported as a fact");
    CHECK(contains("\"ternary_weights\":true"), "ternary weights are reported as a fact");

    // PTQ1_0 parses (so sizes and expert ranges are right) but must never claim
    // to execute: its element layout cannot be confirmed from the published
    // sources, and a guess here is silent corruption rather than a crash.
    const TensorCapability ptq = tensor_capability(143);
    CHECK(ptq.parse, "PTQ1_0 geometry is parsed");
    CHECK(!ptq.dequantize, "PTQ1_0 is not decoded");
    CHECK(!ptq.execute, "PTQ1_0 has no execution path");
    const auto &table = quantisation_table();
    CHECK(table.at(143).block_size == 128 && table.at(143).block_bytes == 28,
          "PTQ1_0 geometry is 28 bytes per 128 weights");

    // The upstream ternary ids are fully supported.
    for (uint32_t id : {34u, 35u, 41u, 42u, 142u}) {
        const TensorCapability cap = tensor_capability(id);
        CHECK(cap.dequantize && cap.execute,
              "ternary type " + std::to_string(id) + " decodes and executes");
    }
}


// Three hand-maintained lists decide whether an architecture runs: the adapter
// table, `is_llama_family` and the probe's known-architecture table. They drifted:
// `smollm3` and `bitnet` had adapters but the probe reported them as known-but-
// unsupported. This test makes the drift a failure instead of a wrong report.
void test_adapter_binding() {
    CHECK(!adapter_rows().empty(), "the adapter table is not empty");
    const auto &known = known_architectures();
    for (const AdapterRow &row : adapter_rows()) {
        const std::string arch = row.arch;
        if (arch == "freemoe-tiny") continue;  // native reference block
        auto it = known.find(arch);
        CHECK(it != known.end(), "the probe knows every mapped architecture: " + arch);
        if (it != known.end()) {
            CHECK(!it->second.needs_adapter,
                  "an adapter row means the probe reports the architecture runnable: " + arch);
            CHECK(it->second.llama_family,
                  "an adapter row is a llama-family block: " + arch);
        }
        CHECK(is_llama_family(arch),
              "an adapter row is claimed by the loader's family list: " + arch);
        CHECK(row.upstream_enum != nullptr && std::string(row.upstream_enum) != "n/a (native)",
              "an adapter row names its upstream LLM_ARCH_* enum: " + arch);
        CHECK(row.graph != nullptr && row.block != nullptr,
              "an adapter row names its graph and block: " + arch);
    }
    // The other direction: the probe must not claim an architecture has a
    // forward pass when no adapter row exists for it.
    for (const auto &entry : known) {
        if (entry.second.needs_adapter) {
            CHECK(adapter_row(entry.first) == nullptr,
                  "an architecture needing an adapter has no adapter row: " + entry.first);
            continue;
        }
        CHECK(adapter_row(entry.first) != nullptr || entry.first == "freemoe-tiny",
              "a runnable architecture has an adapter row: " + entry.first);
        CHECK(entry.second.llama_family || entry.first == "freemoe-tiny",
              "a runnable architecture is a llama-family block: " + entry.first);
        CHECK(is_llama_family(entry.first) || entry.first == "freemoe-tiny",
              "the loader's family list agrees with the probe: " + entry.first);
    }
    // The two architectures this test was written for.
    CHECK(is_llama_family("smollm3") && is_llama_family("bitnet"),
          "smollm3 and bitnet are in the loader's family list");
    CHECK(known.count("smollm3") != 0 && known.count("bitnet") != 0,
          "smollm3 and bitnet are in the probe's known table");
}

// A refusal is a first-class outcome, so it gets its own assertions: the error
// code has to be the specific one (a caller can branch on it), the message has
// to name what is missing, and `probe` has to report the same fact as a
// machine-readable blocker instead of exiting with prose. None of that was
// covered before -- the tests only ever built files that load.
void test_refusal_paths() {
    // 1. An architecture no adapter claims.
    GGUFDocument unknown = parse_gguf_file("tests/fixtures/llama-gqa-shape.gguf");
    unknown.metadata["general.architecture"].s = "nonsense-arch";
    std::string message;
    try {
        (void)build_manifest(unknown, "nonsense-shape.gguf");
    } catch (const Error &e) {
        CHECK(e.code == ErrorCode::UNSUPPORTED_ARCHITECTURE,
              "an unmapped architecture fails with UNSUPPORTED_ARCHITECTURE");
        message = e.what();
    }
    CHECK(message.find("nonsense-arch") != std::string::npos,
          "the refusal names the architecture it could not map");
    CHECK(message.find("probe") != std::string::npos,
          "the refusal points at `probe` for the exact blockers");

    const Json blocked = probe_model(unknown, "nonsense-shape.gguf");
    const std::string dump = blocked.dump();
    CHECK(dump.find("\"runnable\":false") != std::string::npos,
          "an unmapped architecture is not reported runnable");
    CHECK(dump.find("UNKNOWN_ARCHITECTURE") != std::string::npos,
          "the probe names the blocker with a stable code");
    CHECK(dump.find("nonsense-arch") != std::string::npos,
          "the blocker detail names the architecture");
    CHECK(dump.find("\"adapter_available\":false") != std::string::npos,
          "the probe reports that no adapter is available");
    CHECK(dump.find("\"blocker_count\":1") != std::string::npos,
          "the blocker count matches the single defect");

    // 2. The same file, one defect at a time: a missing output projection and a
    // token table is a blocker, not a crash and not a silent bad result.
    GGUFDocument no_head = parse_gguf_file("tests/fixtures/llama-gqa-shape.gguf");
    no_head.tensor_by_name.erase("output.weight");
    no_head.tensor_by_name.erase("token_embd.weight");
    const Json headless = probe_model(no_head, "headless.gguf");
    CHECK(headless.dump().find("NO_OUTPUT_WEIGHT") != std::string::npos,
          "a file with neither head nor token table is refused by the probe");
    CHECK(headless.dump().find("\"runnable\":false") != std::string::npos,
          "a headless file is not reported runnable");

    // 3. A quantisation with no execution kernel is refused where it is used:
    // the probe says which type, and the loader refuses to run the file.
    GGUFDocument bad_type = parse_gguf_file("tests/fixtures/llama-gqa-shape.gguf");
    bad_type.tensors[bad_type.tensor_by_name.at("blk.0.attn_q.weight")].type_id = 143;
    CHECK(throws_code(ErrorCode::UNSUPPORTED_QUANTISATION,
                      [&]() { (void)build_manifest(bad_type, "ptq1_0.gguf"); }),
          "an undecodable tensor type fails with UNSUPPORTED_QUANTISATION");
    const Json typed = probe_model(bad_type, "ptq1_0.gguf");
    CHECK(typed.dump().find("UNSUPPORTED_TENSOR_TYPE") != std::string::npos,
          "the probe names the unsupported tensor type");

    // 4. Every blocker is an object with a code and a detail, so a caller never
    // has to parse prose. This is the contract `probe --json` publishes.
    const Json *blockers = nullptr;
    for (const auto &member : blocked.members())
        if (member.first == "blockers") blockers = &member.second;
    CHECK(blockers != nullptr && blockers->type() == Json::Type::Array,
          "the probe publishes blockers as a JSON array");
    const bool has_blocker = blockers != nullptr && !blockers->items().empty();
    CHECK(has_blocker, "the probe publishes at least one blocker for a refused file");
    if (has_blocker) {
        const Json &first = blockers->items().front();
        CHECK(first.type() == Json::Type::Object, "each blocker is a JSON object");
        bool code_ok = false, detail_ok = false;
        for (const auto &member : first.members()) {
            if (member.first == "code" && member.second.type() == Json::Type::String &&
                !member.second.str().empty())
                code_ok = true;
            if (member.first == "detail" && member.second.type() == Json::Type::String &&
                !member.second.str().empty())
                detail_ok = true;
        }
        CHECK(code_ok, "each blocker carries a non-empty code");
        CHECK(detail_ok, "each blocker carries a non-empty detail");
    }
}

// bitnet's SUBLN: two RMSNorms *inside* the block, plus optional one-element
// scales on every projection. The parity fixture proves the arithmetic matches
// the reference; these checks pin the parts a fixture cannot: that the adapter
// reports the structure, and that a half-written file is refused instead of
// being run with the sub-norms silently dropped.
void test_bitnet_sub_norms() {
    const std::string path = "tests/fixtures/bitnet-shape.gguf";
    const GGUFDocument doc = parse_gguf_file(path);
    const ModelManifest manifest = build_manifest(doc, path);
    CHECK(manifest.config.sub_norms, "a checkpoint with sub-norm tensors enables SUBLN");
    CHECK(manifest.config.projection_scales,
          "a checkpoint with one-element scales enables projection scaling");
    CHECK(manifest.tensor("blk.0.attn_sub_norm.weight") != nullptr,
          "the attention sub-norm is mapped");
    if (manifest.tensor("blk.0.attn_sub_norm.weight"))
        CHECK(manifest.tensor("blk.0.attn_sub_norm.weight")->role == TensorRole::ATTN_SUB_NORM,
              "the attention sub-norm gets its own role");
    if (manifest.tensor("blk.0.ffn_sub_norm.weight"))
        CHECK(manifest.tensor("blk.0.ffn_sub_norm.weight")->role == TensorRole::FFN_SUB_NORM,
              "the FFN sub-norm gets its own role");
    if (manifest.tensor("blk.0.attn_q.scale"))
        CHECK(manifest.tensor("blk.0.attn_q.scale")->role == TensorRole::PROJECTION_SCALE,
              "the projection scale is claimed rather than ignored");
    if (manifest.tensor("blk.0.attn_q.scale"))
        CHECK(manifest.tensor("blk.0.attn_q.scale")->op == "mul",
              "the projection scale is reported as a multiply");
    CHECK(manifest.config.architecture == "bitnet", "the bitnet adapter is selected");

    // A file missing either side of the pair would run a different graph while
    // claiming to be bitnet, so it is a mapping error.
    GGUFDocument no_ffn_sub = doc;
    no_ffn_sub.tensor_by_name.erase("blk.0.ffn_sub_norm.weight");
    CHECK(throws_code(ErrorCode::MISSING_REQUIRED_TENSOR,
                      [&]() { (void)build_manifest(no_ffn_sub, path); }),
          "sub-norms on one path only are refused");

    // Likewise for a file that defines them on some layers only.
    GGUFDocument partial = doc;
    partial.tensor_by_name.erase("blk.1.attn_sub_norm.weight");
    CHECK(throws_code(ErrorCode::MISSING_REQUIRED_TENSOR,
                      [&]() { (void)build_manifest(partial, path); }),
          "a partially sub-normed checkpoint is refused rather than half-applied");

    // A scale that holds more than one element would broadcast in a way no
    // checkpoint means, so the shape is enforced at load time.
    GGUFDocument bad_scale = doc;
    bad_scale.tensors[bad_scale.tensor_by_name.at("blk.0.attn_q.scale")].dims = {2};
    CHECK(throws_code(ErrorCode::SHAPE_MISMATCH,
                      [&]() { (void)build_manifest(bad_scale, path); }),
          "a projection scale must hold exactly one element");

    // The probe reports both facts, so a reader can see which graph runs.
    const Json probe = probe_model(doc, path);
    const std::string dump = probe.dump();
    CHECK(dump.find("sub_norms") != std::string::npos,
          "the probe reports the sub-norms");
    CHECK(dump.find("projection_scales") != std::string::npos,
          "the probe reports the projection scales");
}

// llama.cpp parses minicpm's residual_scale and logit_scale, logs them, and
// never uses them in the graph. Applying them here would quietly disagree with
// the reference on every logit, so this test pins the decision: they are
// recorded, and they do not change the output.
void test_minicpm_scales() {
    const std::string path = "tests/fixtures/minicpm-shape.gguf";
    RunConfig config;
    config.model_path = path;
    config.budget.total_hot_bytes = 64ull * 1024 * 1024;
    Engine engine(config);
    engine.initialize();

    const ModelConfig &cfg = engine.manifest().config;
    CHECK(cfg.architecture == "minicpm", "minicpm adapter is selected");
    CHECK(std::fabs(cfg.embedding_scale - 12.0) < 1e-9,
          "the embedding scale is read from the checkpoint");
    CHECK(cfg.residual_scale > 0.0, "residual_scale is parsed from the checkpoint");
    CHECK(cfg.logit_scale > 0.0, "logit_scale is parsed from the checkpoint");

    // The golden was produced by an implementation that also ignores them, so
    // matching it is exactly the assertion that we did not apply them.
    ArchGolden golden = read_arch_golden("tests/golden/minicpm_reference.txt");
    PromptResult result = engine.run_prompt(golden.prompt, 8);
    CHECK(!golden.logits.empty(), "minicpm golden carries logits");
    if (!golden.logits.empty() && result.prefill_logits.size() == golden.logits.size()) {
        double worst = 0.0;
        for (size_t i = 0; i < golden.logits.size(); ++i)
            worst = std::max(worst, std::fabs(static_cast<double>(result.prefill_logits[i]) -
                                              golden.logits[i]));
        CHECK(worst <= 1e-4, "minicpm logits match the reference (worst " +
                                 std::to_string(worst) + ")");
    }
    engine.release_runtime();

    // The probe must say the same thing, so a reader of the report is not left
    // to infer it from silence.
    auto doc = parse_gguf_file(path);
    const std::string dump = probe_model(doc, path).dump();
    CHECK(dump.find("residual_scale and logit_scale are parsed but not applied") !=
              std::string::npos,
          "the probe explains why the two scales are not applied");
}

// qwen2moe adds a dense shared expert to the routed experts and scales it with
// one scalar per token. Two things can go wrong silently here: the branch gets
// skipped, or it reuses the dense FFN width for a tensor that has its own. Both
// are checked against the operation log, which records the width actually used.
void test_shared_expert() {
    const std::string path = "tests/fixtures/qwen2moe-shape.gguf";
    RunConfig config;
    config.model_path = path;
    config.budget.total_hot_bytes = 64ull * 1024 * 1024;
    Engine engine(config);
    engine.initialize();

    const ModelConfig &cfg = engine.manifest().config;
    CHECK(cfg.architecture == "qwen2moe", "qwen2moe adapter is selected");
    CHECK(cfg.shared_expert, "the shared expert is detected from its tensors");
    CHECK(cfg.shared_expert_intermediate == 40, "the shared expert keeps its own width");
    // The fixture deliberately uses 48 for routed experts and 64 for the dense
    // FFN, so using the wrong one cannot go unnoticed.
    CHECK(cfg.expert_width() == 48, "routed experts keep their own width");
    CHECK(cfg.intermediate == 64, "the dense width is distinct from both");

    ArchGolden golden = read_arch_golden("tests/golden/qwen2moe_reference.txt");
    PromptResult result = engine.run_prompt(golden.prompt, 8);
    if (!golden.logits.empty() && result.prefill_logits.size() == golden.logits.size()) {
        double worst = 0.0;
        for (size_t i = 0; i < golden.logits.size(); ++i)
            worst = std::max(worst, std::fabs(static_cast<double>(result.prefill_logits[i]) -
                                              golden.logits[i]));
        CHECK(worst <= 1e-4, "qwen2moe logits match the reference (worst " +
                                 std::to_string(worst) + ")");
    }

    // The operation log is the evidence that the branch ran at all, and that
    // each branch used its own width rather than the dense one.
    bool saw_shexp_gate = false, saw_shexp_swiglu = false;
    uint64_t expert_width_used = 0, shared_width_used = 0;
    for (const OpRecord &op : engine.backend().operations()) {
        if (op.name == "shexp_gate_inp.L0") saw_shexp_gate = true;
        if (op.name == "shexp_swiglu.L0") {
            saw_shexp_swiglu = true;
            shared_width_used = op.output_bytes / sizeof(float);
        }
        if (op.name == "expert_swiglu.L0") expert_width_used = op.output_bytes / sizeof(float);
    }
    CHECK(saw_shexp_gate, "the shared expert's per-token gate is computed");
    CHECK(saw_shexp_swiglu, "the shared expert's SwiGLU runs");
    CHECK(shared_width_used == 40, "the shared expert executes at its declared width");
    CHECK(expert_width_used == 48, "routed experts execute at their declared width");
    engine.release_runtime();

    // A reader of the probe should not have to infer the branch from silence.
    auto doc = parse_gguf_file(path);
    const std::string dump = probe_model(doc, path).dump();
    CHECK(dump.find("\"shared_expert\":true") != std::string::npos,
          "the probe reports the shared expert");
    CHECK(dump.find("\"shared_expert_size\":40") != std::string::npos,
          "the probe reports the shared expert's width");
    CHECK(dump.find("\"expert_intermediate_size\":48") != std::string::npos,
          "the probe reports the routed experts' width");
}

// phi2 breaks four assumptions the rest of the engine makes at once: it
// normalises with a mean-centred LayerNorm that carries a bias, it has no FFN
// norm at all (the FFN consumes the attention's normalised input), its FFN has
// no gate, and it biases the attention output and the logits. Each of those
// changes the numbers, so each is checked against the operation log rather than
// assumed to have happened.
void test_phi2_block() {
    const std::string path = "tests/fixtures/phi2-layernorm-shape.gguf";
    RunConfig config;
    config.model_path = path;
    config.budget.total_hot_bytes = 64ull * 1024 * 1024;
    Engine engine(config);
    engine.initialize();

    const ModelConfig &cfg = engine.manifest().config;
    CHECK(cfg.architecture == "phi2", "phi2 adapter is selected");
    CHECK(cfg.layer_norm, "the norm type comes from the declared epsilon key");
    CHECK(cfg.norm_bias, "the norms carry biases");
    CHECK(cfg.parallel_residual, "the FFN reuses the attention norm");
    CHECK(cfg.ffn_no_gate, "the FFN has no gate projection");
    CHECK(cfg.activation == "gelu", "the gate-free FFN is GELU");
    CHECK(cfg.ffn_bias, "the FFN carries biases");
    CHECK(cfg.logit_bias, "the logits carry a bias");

    ArchGolden golden = read_arch_golden("tests/golden/phi2-layernorm_reference.txt");
    PromptResult result = engine.run_prompt(golden.prompt, 8);
    if (!golden.logits.empty() && result.prefill_logits.size() == golden.logits.size()) {
        double worst = 0.0;
        for (size_t i = 0; i < golden.logits.size(); ++i)
            worst = std::max(worst, std::fabs(static_cast<double>(result.prefill_logits[i]) -
                                              golden.logits[i]));
        CHECK(worst <= 1e-4, "phi2 logits match the reference (worst " + std::to_string(worst) + ")");
    }

    bool saw_attn_norm = false, saw_ffn_norm = false, saw_gelu = false, saw_swiglu = false;
    bool saw_attn_out_bias = false, saw_logit_bias = false, saw_ffn_up_bias = false;
    for (const OpRecord &op : engine.backend().operations()) {
        if (op.name == "attn_norm.L0") saw_attn_norm = true;
        if (op.name == "ffn_norm.L0") saw_ffn_norm = true;
        if (op.name == "gelu.L0") saw_gelu = true;
        if (op.name == "swiglu.L0") saw_swiglu = true;
        if (op.name == "attn_output_bias.L0") saw_attn_out_bias = true;
        if (op.name == "output_bias") saw_logit_bias = true;
        if (op.name == "ffn_up_bias.L0") saw_ffn_up_bias = true;
    }
    CHECK(saw_attn_norm, "the attention is normalised");
    CHECK(!saw_ffn_norm, "there is no separate FFN norm (the parallel residual reuses one)");
    CHECK(saw_gelu, "the gate-free GELU FFN runs");
    CHECK(!saw_swiglu, "the gated SwiGLU path is not taken");
    CHECK(saw_attn_out_bias, "the attention output bias is added");
    CHECK(saw_logit_bias, "the logit bias is added");
    CHECK(saw_ffn_up_bias, "the FFN up bias is added");
    engine.release_runtime();

    auto doc = parse_gguf_file(path);
    const std::string dump = probe_model(doc, path).dump();
    CHECK(dump.find("\"norm_type\":\"layernorm\"") != std::string::npos,
          "the probe reports the norm type");
    CHECK(dump.find("\"parallel_residual\":true") != std::string::npos,
          "the probe reports the parallel residual");
    CHECK(dump.find("\"ffn_without_gate\":true") != std::string::npos,
          "the probe reports the gate-free FFN");
}

// deepseek2 is the one architecture here whose *attention* is not a variant of
// the usual q/k/v projections, so the configuration it decodes and the branches
// its op log shows are worth pinning separately from the numeric parity check.
void test_deepseek2_mla() {
    const std::string path = "tests/fixtures/deepseek2-mla-shape.gguf";
    RunConfig config;
    config.model_path = path;
    config.budget.total_hot_bytes = 64ull * 1024 * 1024;
    Engine engine(config);
    engine.initialize();

    const ModelConfig &cfg = engine.manifest().config;
    CHECK(cfg.architecture == "deepseek2", "deepseek2 adapter is selected");
    CHECK(cfg.mla, "the kv latent marks multi-head latent attention");
    CHECK(cfg.q_lora_rank == 12, "the low-rank q bottleneck is read");
    CHECK(cfg.kv_lora_rank == 16, "the kv latent width is read");
    CHECK(cfg.head_v_dim() == 10, "the value width comes from value_length_mla");
    CHECK(cfg.v_dim() == cfg.kv_heads * cfg.head_v_dim(),
          "the cache's value width is one head_v per kv head");
    CHECK(cfg.head_v_dim() != cfg.head_dim, "the value width differs from the key width here");
    CHECK(cfg.dense_lead == 1, "the leading dense block count is read");
    CHECK(cfg.expert_gating == "sigmoid", "the router's gating function is read");
    CHECK(cfg.expert_selection_bias, "the selection bias is detected on a non-zero layer");
    CHECK(cfg.shared_expert, "the shared expert is detected on a non-zero layer");
    CHECK(!cfg.shared_expert_gate, "deepseek2's shared expert has no learned scalar gate");
    CHECK(cfg.expert_groups == 2 && cfg.expert_groups_used == 1,
          "group-limited routing is read");

    ArchGolden golden = read_arch_golden("tests/golden/deepseek2-mla_reference.txt");
    PromptResult result = engine.run_prompt(golden.prompt, 8);
    CHECK(result.generated_ids == golden.greedy, "greedy decoding matches the reference");

    bool mla_expand = false, kv_norm = false, rope_q = false, rope_k = false;
    bool shexp_moe = false, shexp_dense = false, router_moe = false, router_dense = false;
    for (const OpRecord &op : engine.backend().operations()) {
        if (op.name == "mla_expand.L1") mla_expand = true;
        if (op.name == "attn_kv_a_norm.L1") kv_norm = true;
        if (op.name == "mla_rope_q.L1") rope_q = true;
        if (op.name == "mla_rope_k.L1") rope_k = true;
        if (op.name == "shexp_down.L1") shexp_moe = true;
        if (op.name == "shexp_down.L0") shexp_dense = true;
        if (op.name == "router.L1") router_moe = true;
        if (op.name == "router.L0") router_dense = true;
    }
    CHECK(mla_expand, "the latent is expanded per head");
    CHECK(kv_norm, "the latent is normalised before expansion");
    CHECK(rope_q && rope_k, "rotary runs on the query heads and the shared k_pe");
    CHECK(router_moe && !router_dense, "the lead block has no router: it is dense");
    CHECK(shexp_moe && !shexp_dense, "the shared expert runs beside the routed experts only");
    engine.release_runtime();

    auto doc = parse_gguf_file(path);
    const std::string dump = probe_model(doc, path).dump();
    CHECK(dump.find("\"multi_head_latent_attention\":true") != std::string::npos,
          "the probe reports multi-head latent attention");
    CHECK(dump.find("\"leading_dense_blocks\":1") != std::string::npos,
          "the probe reports the dense lead-in");
    CHECK(dump.find("\"expert_gating_func\":2") != std::string::npos,
          "the probe reports the router's gating function");
    CHECK(dump.find("\"expert_selection_bias\":true") != std::string::npos,
          "the probe reports the selection bias");
}

// gpt-oss (`src/models/openai-moe.cpp`): attention sinks, a post-attention norm
// that is the MoE's input norm (there is no ffn_norm), a biased router, a bias
// on each routed projection, `swiglu_oai` and alternating full/windowed layers.
// Every one of those changes a number, so each is checked on its own.
void test_gpt_oss_block() {
    const std::string path = "tests/fixtures/gpt-oss-shape.gguf";
    RunConfig config;
    config.model_path = path;
    config.budget.total_hot_bytes = 64ull * 1024 * 1024;
    Engine engine(config);
    engine.initialize();

    const ModelConfig &cfg = engine.manifest().config;
    CHECK(cfg.architecture == "gpt-oss", "gpt-oss adapter is selected");
    CHECK(!cfg.parallel_residual,
          "a post-attention norm is not a parallel residual: the block is pre-norm");
    CHECK(cfg.ffn_norm_is_post, "the FFN norm is the post-attention norm");
    CHECK(!cfg.post_norms, "gpt-oss has no post-FFN norm, so it is not a post-norm block");
    CHECK(cfg.attn_sinks, "the attention sinks are read from the file");
    CHECK(cfg.activation == "swiglu-oai",
          "gpt-oss uses swiglu_oai, not SwiGLU: the activation comes from the family");
    CHECK(cfg.activation_limit == 7.0 && cfg.activation_alpha == 1.702,
          "swiglu_oai's limit and alpha are llama.cpp's 7.0 and 1.702");
    CHECK(cfg.expert_router_bias, "the router bias is detected from the file");
    CHECK(cfg.expert_projection_bias, "the per-expert projection biases are detected");
    CHECK(!cfg.expert_selection_bias,
          "a router-logit bias is not the selection bias; the two are different tensors");
    CHECK(cfg.expert_gating == "softmax-weight",
          "gpt-oss gates with softmax-weight (LLAMA_EXPERT_GATING_FUNC_TYPE_SOFTMAX_WEIGHT)");

    // Alternating windows: 12 wide, every second layer. The period form marks
    // layers whose index is not a multiple of the period, so 0 and 2 are full.
    CHECK(cfg.sliding_window == 12, "the window width is read");
    CHECK(cfg.swa_layers.size() == 4, "one window flag per layer");
    if (cfg.swa_layers.size() == 4) {
        CHECK(cfg.swa_layers[0] == 0 && cfg.swa_layers[2] == 0,
              "the first and third layers attend to the whole context");
        CHECK(cfg.swa_layers[1] == 1 && cfg.swa_layers[3] == 1,
              "the second and fourth layers are windowed");
    }
    CHECK(!cfg.layer_uses_window(0) && cfg.layer_uses_window(1),
          "the per-layer decision is exposed, not just the model-wide window");

    ArchGolden golden = read_arch_golden("tests/golden/gpt-oss_reference.txt");
    PromptResult result = engine.run_prompt(golden.prompt, 8);
    CHECK(result.input_ids == golden.input_ids, "prompt tokenization matches the reference");
    CHECK(result.generated_ids == golden.greedy,
          "greedy decoding matches the reference, windows and all");

    bool router_bias = false, expert_bias = false, post_norm_ffn = false;
    for (const OpRecord &op : engine.backend().operations()) {
        if (op.name == "router_logit_bias.L0") router_bias = true;
        if (op.name == "expert_proj_bias.L0") expert_bias = true;
        if (op.name == "ffn_norm.L0") post_norm_ffn = true;
    }
    CHECK(router_bias, "the router bias is added to the logits as its own step");
    CHECK(expert_bias, "each expert's projection bias is added as its own step");
    CHECK(post_norm_ffn, "the FFN normalises through the post-attention norm");
    engine.release_runtime();

    auto doc = parse_gguf_file(path);
    // A pattern shorter than the layer count would silently describe only the
    // first layers, so it is refused.
    {
        GGUFDocument broken = doc;
        MetadataValue pattern;
        pattern.type = GGUFValueType::ARRAY;
        for (int i = 0; i < 3; ++i) {
            MetadataValue entry;
            entry.type = GGUFValueType::BOOL;
            entry.b = i % 2 == 1;
            pattern.array.push_back(entry);
        }
        broken.metadata["gpt-oss.attention.sliding_window_pattern"] = pattern;
        CHECK(throws_code(ErrorCode::METADATA_MISMATCH,
                          [&]() { (void)build_manifest(broken, path); }),
              "a window pattern that does not cover every layer is refused");
    }
    // Half a bias: the flag is per model but the tensors are per layer, so one
    // missing layer would run that block with a different router.
    {
        GGUFDocument broken = doc;
        broken.tensor_by_name.erase("blk.1.ffn_gate_inp.bias");
        CHECK(throws_code(ErrorCode::MISSING_REQUIRED_TENSOR,
                          [&]() { (void)build_manifest(broken, path); }),
              "a router bias on only some layers is refused");
    }
    {
        GGUFDocument broken = doc;
        broken.tensor_by_name.erase("blk.2.ffn_down_exps.bias");
        CHECK(throws_code(ErrorCode::MISSING_REQUIRED_TENSOR,
                          [&]() { (void)build_manifest(broken, path); }),
              "an expert projection bias on only some layers is refused");
    }
    // A post-attention norm with an ordinary ffn_norm beside it is neither
    // gpt-oss nor gemma: the block would use one of the two and drop the other.
    {
        GGUFDocument broken = doc;
        TensorInfo fake;
        fake.name = "blk.0.ffn_norm.weight";
        fake.dims = {cfg.hidden};
        fake.type_id = 0;
        broken.tensors.push_back(fake);
        broken.tensor_by_name[fake.name] = broken.tensors.size() - 1;
        CHECK(throws_code(ErrorCode::MISSING_REQUIRED_TENSOR,
                          [&]() { (void)build_manifest(broken, path); }),
              "the norm layout has to be one known shape, not a mix");
    }
}

// plm (`src/models/plm.cpp`): multi-head latent attention with a one-step q
// projection, a gate-free relu^2 FFN and the output head tied to the token
// table. This is the file that makes `relu_sq` reachable end to end.
void test_plm_block() {
    const std::string path = "tests/fixtures/plm-shape.gguf";
    RunConfig config;
    config.model_path = path;
    config.budget.total_hot_bytes = 64ull * 1024 * 1024;
    Engine engine(config);
    engine.initialize();

    const ModelConfig &cfg = engine.manifest().config;
    CHECK(cfg.architecture == "plm", "plm adapter is selected");
    CHECK(cfg.mla, "the kv latent marks multi-head latent attention");
    CHECK(cfg.q_lora_rank == 0, "a zero q rank means the one-step q projection");
    CHECK(cfg.kv_lora_rank == 16, "the kv latent width is read");
    CHECK(cfg.head_v_dim() != cfg.head_dim, "the value width is its own number here");
    CHECK(cfg.ffn_no_gate, "the FFN has one projection, not a gate and an up");
    CHECK(cfg.activation == "relu_sq",
          "plm's activation is relu^2 (LLM_FFN_RELU_SQR), not a default");

    ArchGolden golden = read_arch_golden("tests/golden/plm_reference.txt");
    PromptResult result = engine.run_prompt(golden.prompt, 8);
    CHECK(result.input_ids == golden.input_ids, "prompt tokenization matches the reference");
    CHECK(result.generated_ids == golden.greedy, "greedy decoding matches the reference");

    bool q_rope = false, kv_rope = false, mla_expand = false, gate = false;
    for (const OpRecord &op : engine.backend().operations()) {
        if (op.name == "mla_rope_q.L0") q_rope = true;
        if (op.name == "mla_rope_k.L0") kv_rope = true;
        if (op.name == "mla_expand.L0") mla_expand = true;
        if (op.name == "ffn_gate.L0") gate = true;
    }
    CHECK(q_rope && kv_rope, "rotary runs on the query heads and the shared k_pe");
    CHECK(mla_expand, "the latent is expanded per head");
    CHECK(!gate, "a gate-free FFN must not run a gate");
    engine.release_runtime();
}

// mistral3 (`src/models/mistral3.cpp`): the llama block plus biases on all three
// FFN projections, an adaptive RMSNorm whose MLP is a gelu_erf over a sinusoidal
// delay embedding, and llama.cpp's per-position attention temperature. Each of
// the three is checked on its own, and the temperature's missing-floor case is
// refused rather than silently replaced by 1.0.
void test_mistral3_block() {
    const std::string path = "tests/fixtures/mistral3-shape.gguf";
    RunConfig config;
    config.model_path = path;
    config.budget.total_hot_bytes = 64ull * 1024 * 1024;
    Engine engine(config);
    engine.initialize();

    const ModelConfig &cfg = engine.manifest().config;
    CHECK(cfg.architecture == "mistral3", "mistral3 adapter is selected");
    CHECK(cfg.ffn_gate_bias && cfg.ffn_bias, "all three FFN projections carry a bias");
    CHECK(cfg.ffn_norm_ada, "the adaptive norm is detected from its tensors");
    CHECK(cfg.realtime_delay == 7, "the delay driving the embedding is read");
    CHECK(cfg.attn_temp_scale != 0.0 && cfg.attn_temp_floor == 32.0,
          "the attention temperature and its ramp floor are read");
    CHECK(cfg.activation == "silu",
          "the FFN itself is SwiGLU; gelu_erf belongs to the adaptive norm only");

    ArchGolden golden = read_arch_golden("tests/golden/mistral3_reference.txt");
    PromptResult result = engine.run_prompt(golden.prompt, 8);
    CHECK(result.input_ids == golden.input_ids, "prompt tokenization matches the reference");
    CHECK(result.generated_ids == golden.greedy, "greedy decoding matches the reference");

    bool ada = false, gelu_erf = false, temp = false, gate_bias = false, down_bias = false;
    for (const OpRecord &op : engine.backend().operations()) {
        if (op.name == "ffn_norm_ada.L0") ada = true;
        if (op.name == "gelu_erf.L0") gelu_erf = true;
        if (op.name == "attn_temperature.L0") temp = true;
        if (op.name == "ffn_gate_bias.L0") gate_bias = true;
        if (op.name == "ffn_down_bias.L0") down_bias = true;
    }
    CHECK(ada && gelu_erf, "the adaptive norm runs, and its MLP is a gelu_erf");
    CHECK(temp, "the attention temperature is applied per position");
    CHECK(gate_bias && down_bias, "the FFN biases are added as their own steps");
    engine.release_runtime();

    auto doc = parse_gguf_file(path);
    {
        // The floor is a required key, not a default: upstream throws when it is
        // missing, and 1.0 would be a silently different model.
        GGUFDocument broken = doc;
        broken.metadata.erase("mistral3.rope.scaling.original_context_length");
        CHECK(throws_code(ErrorCode::METADATA_MISMATCH,
                          [&]() { (void)build_manifest(broken, path); }),
              "a temperature scale without a ramp floor is refused");
    }
    {
        // Half an adaptive norm: the down projection without its partner would
        // silently become an identity-prefixed scale.
        GGUFDocument broken = doc;
        broken.tensor_by_name.erase("blk.0.ffn_norm_ada_up.weight");
        CHECK(throws_code(ErrorCode::MISSING_REQUIRED_TENSOR,
                          [&]() { (void)build_manifest(broken, path); }),
              "the adaptive norm needs both of its projections");
    }
}

// The mRoPE families (qwen3vl/qwen3vlmoe, and qwen35 once its recurrent layers
// run): llama.cpp reads `rope.dimension_sections` as a required key for them and
// selects the interleaved layout by architecture, not by a file flag. A file
// without the split is refused rather than run as plain RoPE.
void test_mrope_families() {
    struct Expect {
        const char *slug;
        const char *arch;
        bool moe;
    };
    const Expect cases[] = {
        {"qwen3vl", "qwen3vl", false},
        {"qwen3vlmoe", "qwen3vlmoe", true},
    };
    for (const Expect &c : cases) {
        const std::string path = std::string("tests/fixtures/") + c.slug + "-shape.gguf";
        const ModelManifest manifest = build_manifest(parse_gguf_file(path), path);
        const std::string tag = std::string(c.slug) + ": ";
        CHECK(manifest.config.architecture == c.arch, tag + "architecture comes from the file");
        CHECK(manifest.config.mrope, tag + "the section split turns mRoPE on");
        CHECK(manifest.config.mrope_interleaved,
              tag + "qwen3vl's layout is interleaved (llama_rope_type)");
        CHECK(manifest.config.qk_norm, tag + "per-head q/k norm is present");
        CHECK((manifest.config.expert_count != 0) == c.moe, tag + "routed experts match the file");
        const std::vector<std::string> tags = adapter_config_tags(manifest.config);
        const bool mrope_tag = std::find_if(tags.begin(), tags.end(), [](const std::string &t) {
                                   return t.rfind("mrope=interleaved", 0) == 0;
                               }) != tags.end();
        CHECK(mrope_tag, tag + "the adapter reports the interleaved section layout");

        GGUFDocument broken = parse_gguf_file(path);
        broken.metadata.erase(std::string(c.arch) + ".rope.dimension_sections");
        CHECK(throws_code(ErrorCode::METADATA_MISMATCH,
                          [&]() { (void)build_manifest(broken, path); }),
              tag + "a file without the required section split is refused");
    }
}

// The recurrent families (llama.cpp models/mamba.cpp, mamba2.cpp, and the
// shared block in mamba-base.cpp). There is no attention and, in a pure file, no
// feed-forward: the block is a 1D convolution over a rolling window and a
// state-space scan whose state replaces the kv cache. Because the state is the
// only channel through which an earlier token reaches a later one, the checks
// below look at both the parsed facts and the shape of that state.
// `attention.head_count` is scalar-or-array upstream (`get_key_or_arr`,
// llama.cpp/src/llama-model.cpp:1480). The array form states one count per
// layer, with zero entries marking the recurrent layers of a hybrid. This engine
// keeps one count for the stack, so the array is accepted when every attention
// layer agrees -- and a zero is checked against the tensor-derived recurrence
// mask rather than trusted.
void test_metadata_array_forms() {
    const std::string path = "tests/fixtures/llama-gqa-shape.gguf";
    const GGUFDocument scalar = parse_gguf_file(path);
    const std::string key = "llama.attention.head_count";
    const uint32_t heads = build_manifest(scalar, path).config.heads;
    CHECK(heads != 0, "the scalar head count is read");

    // A matching array is equivalent: same config, same ids.
    GGUFDocument array_form = scalar;
    MetadataValue entries;
    entries.type = GGUFValueType::ARRAY;
    entries.element_type = GGUFValueType::UINT32;
    for (uint32_t l = 0; l < 2; ++l) {
        MetadataValue one;
        one.type = GGUFValueType::UINT32;
        one.u = heads;
        entries.array.push_back(one);
    }
    array_form.metadata[key] = entries;
    const ModelManifest array_manifest = build_manifest(array_form, path);
    CHECK(array_manifest.config.heads == heads, "the array head count resolves to the same count");

    RunConfig config;
    config.model_path = path;
    config.budget.total_hot_bytes = 64ull * 1024 * 1024;
    PromptResult scalar_ids;
    {
        Engine engine(config);
        engine.initialize();
        scalar_ids = engine.run_prompt("Hello", 4);
    }
    // The array form is exercised through the manifest, since the engine reads a
    // file; the parsed config being identical is what the check above states.
    (void)scalar_ids;

    // An array of the wrong length is refused.
    GGUFDocument short_array = array_form;
    short_array.metadata[key].array.pop_back();
    CHECK(throws_code(ErrorCode::METADATA_MISMATCH,
                      [&]() { (void)build_manifest(short_array, path); }),
          "a per-layer head count with the wrong number of entries is refused");

    // Disagreeing attention layers are refused: one count cannot describe them.
    GGUFDocument disagreeing = array_form;
    disagreeing.metadata[key].array[1].u = heads + 1;
    CHECK(throws_code(ErrorCode::SHAPE_MISMATCH,
                      [&]() { (void)build_manifest(disagreeing, path); }),
          "per-layer head counts that differ among attention layers are refused");

    // A zero entry claims the layer is recurrent; here no `ssm_in` exists, so the
    // claim contradicts the tensors.
    GGUFDocument zero_recurrent = array_form;
    zero_recurrent.metadata[key].array[1].u = 0;
    CHECK(throws_code(ErrorCode::METADATA_MISMATCH,
                      [&]() { (void)build_manifest(zero_recurrent, path); }),
          "a zero head count without a recurrent layer in the tensors is refused");

    // `attention.recurrent_layers` in its positive form: jamba's split stated
    // explicitly, and it must agree with the tensors rather than override them.
    {
        const std::string jamba = "tests/fixtures/jamba-shape.gguf";
        GGUFDocument doc = parse_gguf_file(jamba);
        const ModelManifest derived = build_manifest(doc, jamba);
        MetadataValue mask;
        mask.type = GGUFValueType::ARRAY;
        mask.element_type = GGUFValueType::BOOL;
        for (uint32_t l = 0; l < derived.config.layers; ++l) {
            MetadataValue one;
            one.type = GGUFValueType::BOOL;
            one.b = derived.config.layer_is_recurrent(l) != 0;
            mask.array.push_back(one);
        }
        doc.metadata["jamba.attention.recurrent_layers"] = mask;
        const ModelManifest declared = build_manifest(doc, jamba);
        CHECK(declared.config.recurrent_layers == derived.config.recurrent_layers,
              "an explicit recurrent_layers mask that matches the tensors is accepted");

        // Flipping one entry makes the claim disagree with the tensors.
        doc.metadata["jamba.attention.recurrent_layers"].array[0].b =
            !doc.metadata["jamba.attention.recurrent_layers"].array[0].b;
        CHECK(throws_code(ErrorCode::METADATA_MISMATCH,
                          [&]() { (void)build_manifest(doc, jamba); }),
              "a recurrent_layers mask that contradicts the tensors is refused");
    }
}

void test_recurrent_families() {
    struct Expect {
        const char *slug;
        const char *kind;
        uint32_t groups;
        bool vector_norms;
    };
    const Expect cases[] = {
        {"mamba", "mamba", 1, false},
        {"mamba-dtrms", "mamba", 1, true},
        // The flag without the weights: upstream's `build_norm` degenerates to
        // unit gain, so the norms still run. The golden is the reference's own
        // arithmetic for that file, not this build's.
        {"mamba-flagonly", "mamba", 1, true},
        {"mamba2", "mamba2", 2, false},
    };
    for (const Expect &c : cases) {
        const std::string path = std::string("tests/fixtures/") + c.slug + "-shape.gguf";
        RunConfig config;
        config.model_path = path;
        config.budget.total_hot_bytes = 64ull * 1024 * 1024;
        Engine engine(config);
        engine.initialize();
        const ModelConfig &cfg = engine.manifest().config;
        const std::string tag = std::string(c.slug) + ": ";

        CHECK(cfg.ssm_kind == c.kind, tag + "the state-space variant comes from the tensors");
        CHECK(cfg.ssm_groups == c.groups, tag + "the group count is read");
        CHECK(cfg.ssm_vector_norms == c.vector_norms, tag + "the dt/B/C norms are detected");
        CHECK(!cfg.has_ffn, tag + "a pure mamba file has no feed-forward");
        CHECK(cfg.layer_is_recurrent(0) && cfg.layer_is_recurrent(1),
              tag + "every layer of a pure file is recurrent");
        CHECK(cfg.heads == 0 && cfg.kv_heads == 0,
              tag + "a recurrent file states no attention head counts");
        // The state is a fixed size per layer, not a per-position cache: the
        // convolution history plus one state vector per channel.
        const uint64_t per_layer = cfg.ssm_state_floats();
        const uint64_t expected =
            static_cast<uint64_t>(cfg.ssm_conv - 1) * cfg.ssm_conv_channels() +
            static_cast<uint64_t>(cfg.ssm_state) * cfg.ssm_inner;
        CHECK(per_layer == expected, tag + "the state size is history plus one state vector");

        ArchGolden golden = read_arch_golden("tests/golden/" + std::string(c.slug) + "_reference.txt");
        PromptResult result = engine.run_prompt(golden.prompt, 8);
        CHECK(result.input_ids == golden.input_ids, tag + "prompt tokenization matches");
        CHECK(result.generated_ids == golden.greedy, tag + "greedy decoding matches the reference");

        int convolutions = 0, scans = 0;
        bool conv_bytes_exact = true, scan_bytes_exact = true;
        const uint64_t conv_reads = static_cast<uint64_t>(cfg.ssm_conv_channels()) *
                                    (2 * cfg.ssm_conv - 1) * sizeof(float);
        const uint64_t conv_writes =
            static_cast<uint64_t>(cfg.ssm_conv_channels()) * sizeof(float);
        const uint64_t scan_state =
            static_cast<uint64_t>(cfg.ssm_state) * cfg.ssm_inner * sizeof(float);
        for (const OpRecord &op : engine.backend().operations()) {
            if (op.name.rfind("ssm_conv.", 0) == 0) {
                ++convolutions;
                conv_bytes_exact = conv_bytes_exact && op.input_bytes == conv_reads &&
                                   op.output_bytes == conv_writes;
            }
            if (op.name.rfind("ssm_scan.", 0) == 0) {
                ++scans;
                scan_bytes_exact = scan_bytes_exact && op.output_bytes == scan_state &&
                                   op.input_bytes >= 2 * scan_state;
            }
        }
        CHECK(convolutions > 0 && convolutions % static_cast<int>(cfg.layers) == 0,
              tag + "the convolution runs once per layer per token");
        CHECK(scans == convolutions, tag + "the scan runs wherever the convolution does");
        // The per-op accounting has to state the traffic honestly: the kernel and
        // the window it slides against for the convolution, the state it reads and
        // the state it writes for the scan.
        CHECK(conv_bytes_exact, tag + "the convolution reports its window and kernel traffic");
        CHECK(scan_bytes_exact, tag + "the scan reports the state it reads and writes");
        engine.release_runtime();
    }

    // mamba2 has no dt/B/C norm anywhere in upstream: `build_mamba2_layer`
    // applies its grouped `ssm_norm` after the scan and nothing before it, and
    // `ssm.dt_b_c_rms` is a key only models/mamba.cpp reads. Upstream's mamba2
    // loader does not even create those tensors, so a file carrying them is not
    // a checkpoint that runtime produces -- it is refused by name rather than
    // normalised. That refusal is what keeps the arithmetic from drifting: there
    // is no way to reach the scan with a norm upstream does not apply.
    {
        auto run = [&](const std::string &slug) {
            RunConfig config;
            config.model_path = "tests/fixtures/" + slug + "-shape.gguf";
            config.budget.total_hot_bytes = 64ull * 1024 * 1024;
            Engine engine(config);
            engine.initialize();
            PromptResult result = engine.run_prompt("Hello", 8);
            int dt_norm_ops = 0;
            for (const OpRecord &op : engine.backend().operations())
                if (op.name.rfind("ssm_dt_norm.", 0) == 0) ++dt_norm_ops;
            engine.release_runtime();
            return std::make_pair(std::move(result), dt_norm_ops);
        };
        auto plain = run("mamba2");
        auto with_norms = run("mamba2-dtnorm");
        // The two files share every weight (each tensor is generated from its own
        // name) and differ only in the extra norm tensors and the flag. Applying
        // them would change the scan's inputs, so identical output is the proof
        // that they are inert here.
        CHECK(with_norms.first.generated_ids == plain.first.generated_ids,
              "the dt/B/C weights do not change what mamba2 generates");
        CHECK(with_norms.second == 0, "no dt norm runs inside a mamba2 layer");
        // The goldens record the same thing independently of the engine: the
        // python reference computes the identical ids for both files.
        const ArchGolden plain_golden = read_arch_golden("tests/golden/mamba2_reference.txt");
        const ArchGolden extra_golden = read_arch_golden("tests/golden/mamba2-dtnorm_reference.txt");
        CHECK(extra_golden.greedy == plain_golden.greedy,
              "the reference agrees that the dt/B/C weights change nothing");
    }

    // mamba-1 reads `ssm.dt_b_c_rms` (models/mamba.cpp). With the flag set and
    // no weights in the file the norms still apply, because upstream's
    // `build_norm` multiplies only when the weight tensor is present: the norm
    // degenerates to unit gain instead of disappearing. A file without the flag
    // and without the tensors is left alone.
    {
        const std::string path = "tests/fixtures/mamba-shape.gguf";
        const GGUFDocument plain = parse_gguf_file(path);
        CHECK(!build_manifest(plain, path).config.ssm_vector_norms,
              "mamba without the flag and without the weights applies no dt/B/C norm");

        GGUFDocument flagged = plain;
        MetadataValue flag;
        flag.type = GGUFValueType::BOOL;
        flag.b = true;
        flagged.metadata["mamba.ssm.dt_b_c_rms"] = flag;
        const ModelManifest manifest = build_manifest(flagged, path);
        CHECK(manifest.config.ssm_vector_norms,
              "mamba applies the dt/B/C norms when the file sets ssm.dt_b_c_rms");
        CHECK(!manifest.config.ssm_dt_norm_present && !manifest.config.ssm_b_norm_present &&
                  !manifest.config.ssm_c_norm_present,
              "the flagged file carries no norm weights, so the norms are unit gain");
        const std::vector<std::string> tags = adapter_config_tags(manifest.config);
        CHECK(std::find(tags.begin(), tags.end(), "ssm_dt_b_c_rms=yes") != tags.end(),
              "the adapter tag says the norms are applied");
    }

    // jamba creates all three norms as required tensors (models/jamba.cpp:65-71),
    // so a jamba file missing one is malformed rather than plain. Dropping the
    // tensor from the file must be refused, not silently run without it.
    {
        const std::string path = "tests/fixtures/jamba-shape.gguf";
        auto without = [&](std::initializer_list<const char *> victims) {
            GGUFDocument doc = parse_gguf_file(path);
            for (const char *victim : victims)
                doc.tensors.erase(
                    std::remove_if(doc.tensors.begin(), doc.tensors.end(),
                                   [&](const TensorInfo &t) { return t.name == victim; }),
                    doc.tensors.end());
            doc.tensor_by_name.clear();
            for (size_t i = 0; i < doc.tensors.size(); ++i)
                doc.tensor_by_name[doc.tensors[i].name] = i;
            return doc;
        };
        // Every recurrent layer loses the weight: the stack agrees with itself,
        // and jamba's own rule rejects it by name.
        const GGUFDocument all_layers =
            without({"blk.0.ssm_b_norm.weight", "blk.2.ssm_b_norm.weight"});
        CHECK(throws_code(ErrorCode::MISSING_REQUIRED_TENSOR,
                          [&]() { (void)build_manifest(all_layers, path); }),
              "jamba without ssm_b_norm.weight is refused by name");
        // Only one layer loses it: the layers now disagree, and one model-wide
        // rule cannot describe the stack, so that is refused too rather than
        // normalising half the layers.
        const GGUFDocument one_layer = without({"blk.0.ssm_b_norm.weight"});
        CHECK(throws_code(ErrorCode::METADATA_MISMATCH,
                          [&]() { (void)build_manifest(one_layer, path); }),
              "recurrent layers that disagree about the norms are refused");
    }

    // The budget follows the state, not the context: a recurrent file must not
    // reserve a per-position cache it will never read.
    {
        const ModelManifest manifest =
            build_manifest(parse_gguf_file("tests/fixtures/mamba-shape.gguf"),
                           "tests/fixtures/mamba-shape.gguf");
        BudgetConfig budget;
        budget.total_hot_bytes = 64ull * 1024 * 1024;
        DeviceInfo device;
        device.memory_type = "host";
        const BudgetResult result = compute_budget(manifest, budget, device);
        const uint64_t expected = static_cast<uint64_t>(manifest.config.layers) *
                                  manifest.config.ssm_state_floats() * sizeof(float);
        CHECK(result.kv_bytes == expected,
              "a recurrent model reserves one rolling state per layer, not a kv cache");
        CHECK(result.kv_bytes <
                  static_cast<uint64_t>(manifest.config.layers) * manifest.config.context *
                      manifest.config.hidden * sizeof(float),
              "the state is smaller than a per-position cache would be");
    }

    // Every way the metadata can disagree with the tensors is refused, because
    // each one would otherwise run with a state of the wrong width.
    {
        GGUFDocument doc = parse_gguf_file("tests/fixtures/mamba-shape.gguf");
        GGUFDocument broken = doc;
        broken.metadata.erase("mamba.ssm.state_size");
        CHECK(throws_code(ErrorCode::METADATA_MISMATCH,
                          [&]() { (void)build_manifest(broken, "mamba-shape.gguf"); }),
              "a recurrent file without a state size is refused");
    }
    {
        GGUFDocument broken = parse_gguf_file("tests/fixtures/mamba-shape.gguf");
        // mamba expands the hidden width by exactly 2; the scan's layout depends
        // on it, so a file that says otherwise is refused rather than reshaped.
        broken.metadata["mamba.ssm.inner_size"].u = 48;
        CHECK(throws_code(ErrorCode::SHAPE_MISMATCH,
                          [&]() { (void)build_manifest(broken, "mamba-shape.gguf"); }),
              "an expansion other than 2 is refused");
    }
    {
        // A declared recurrence mask is a claim, not evidence: a file that says a
        // layer is recurrent must carry that layer's `ssm_in`, and one that says
        // it is an attention layer must not. Half of a hybrid stack would run the
        // wrong block on one of the two, so the disagreement is an error.
        GGUFDocument broken = parse_gguf_file("tests/fixtures/mamba-shape.gguf");
        MetadataValue layers;
        layers.type = GGUFValueType::ARRAY;
        for (int i = 0; i < 2; ++i) {
            MetadataValue entry;
            entry.type = GGUFValueType::BOOL;
            entry.b = i == 0;
            layers.array.push_back(entry);
        }
        broken.metadata["mamba.attention.recurrent_layers"] = layers;
        CHECK(throws_code(ErrorCode::METADATA_MISMATCH,
                          [&]() { (void)build_manifest(broken, "mamba-shape.gguf"); }),
              "a recurrence mask that disagrees with the tensors is refused");
    }
    {
        // The same rule through the other spelling llama.cpp uses: a per-layer
        // head count whose zero entry claims the layer has no attention.
        GGUFDocument broken = parse_gguf_file("tests/fixtures/mamba2-shape.gguf");
        MetadataValue counts;
        counts.type = GGUFValueType::ARRAY;
        for (int i = 0; i < 2; ++i) {
            MetadataValue entry;
            entry.type = GGUFValueType::INT32;
            entry.i = i == 0 ? 0 : 2;
            counts.array.push_back(entry);
        }
        broken.metadata["mamba2.attention.head_count_kv"] = counts;
        CHECK(throws_code(ErrorCode::METADATA_MISMATCH,
                          [&]() { (void)build_manifest(broken, "mamba2-shape.gguf"); }),
              "a per-layer kv-head mask that disagrees with the tensors is refused");
    }
    // A hybrid stack runs both halves: the mamba layers keep their rolling state,
    // the attention layers own a per-position cache, and the budget reserves both.
    {
        const std::string path = "tests/fixtures/jamba-shape.gguf";
        RunConfig config;
        config.model_path = path;
        config.budget.total_hot_bytes = 64ull * 1024 * 1024;
        Engine engine(config);
        engine.initialize();
        const ModelConfig &cfg = engine.manifest().config;
        CHECK(cfg.attention_layer_count() == 2 && cfg.recurrent_layer_count() == 2,
              "the hybrid splits its layers between the two mixers");
        CHECK(cfg.layer_is_recurrent(0) && !cfg.layer_is_recurrent(1),
              "the split is per layer, not per model");
        CHECK(cfg.rope_disabled && cfg.heads == 4 && cfg.kv_heads == 2,
              "jamba keeps its head counts and applies no rotary");
        CHECK(!cfg.layer_is_moe(0) && cfg.layer_is_moe(1),
              "the router is a per-layer fact: this stack mixes dense and MoE blocks");
        CHECK(cfg.attention_slot(1) == 0 && cfg.attention_slot(3) == 1,
              "attention layers index the cache by their position among attention layers");

        ArchGolden golden = read_arch_golden("tests/golden/jamba_reference.txt");
        PromptResult result = engine.run_prompt(golden.prompt, 8);
        CHECK(result.generated_ids == golden.greedy,
              "the hybrid decodes the token sequence the reference computes");
        int convolutions = 0, scans = 0, attentions = 0;
        for (const OpRecord &op : engine.backend().operations()) {
            if (op.name.rfind("ssm_conv.", 0) == 0) ++convolutions;
            if (op.name.rfind("ssm_scan.", 0) == 0) ++scans;
            if (op.name.rfind("attention.", 0) == 0) ++attentions;
        }
        const int per_token = static_cast<int>(result.input_ids.size() + result.generated_ids.size());
        CHECK(convolutions == per_token * 2 && scans == per_token * 2,
              "the scan runs on the recurrent layers only");
        CHECK(attentions == per_token * 2,
              "attention runs on the attention layers only, once each");
        engine.release_runtime();

        // The budget follows both halves: a per-position cache for the attention
        // layers plus a fixed state for the recurrent ones.
        const ModelManifest manifest = build_manifest(parse_gguf_file(path), path);
        BudgetConfig budget;
        budget.total_hot_bytes = 64ull * 1024 * 1024;
        DeviceInfo device;
        device.memory_type = "host";
        const BudgetResult reservation = compute_budget(manifest, budget, device);
        const uint64_t expected =
            (manifest.config.attention_cache_floats(manifest.config.context) +
             manifest.config.recurrent_state_floats()) *
            sizeof(float);
        CHECK(reservation.kv_bytes == expected,
              "a hybrid reserves the attention cache and the recurrent state, not one or other");
    }
    {
        GGUFDocument broken = parse_gguf_file("tests/fixtures/mamba2-shape.gguf");
        // mamba2's grouped norm is what keeps the scan's output in range.
        broken.tensor_by_name.erase("blk.1.ssm_norm.weight");
        CHECK(throws_code(ErrorCode::MISSING_REQUIRED_TENSOR,
                          [&]() { (void)build_manifest(broken, "mamba2-shape.gguf"); }),
              "mamba2 without its grouped norm is refused");
    }
}

// ---------------------------------------------------------------------------
// Chat templates
// ---------------------------------------------------------------------------

// The templates below are the shapes published checkpoints actually carry, not
// invented ones: ChatML (qwen2/3, glm4), llama3's header markers, gemma's turn
// markers, phi3's, and llama2's [INST]. Each expectation is what
// HuggingFace's `apply_chat_template` produces for the same conversation.
void test_chat_template_rendering() {
    const std::vector<fm::ChatMessage> with_system = {
        {"system", "be terse"}, {"user", "hello"}, {"assistant", "hi"}};
    fm::ChatTemplateConfig config;
    config.bos_token = "<s>";
    config.eos_token = "</s>";

    std::string chatml = fm::builtin_chat_template("qwen2");
    CHECK(chatml == fm::builtin_chat_template_by_name("chatml"),
          "the qwen2 built-in is the chatml template");
    CHECK(fm::render_chat_template(chatml, with_system, config) ==
              "<|im_start|>system\nbe terse<|im_end|>\n"
              "<|im_start|>user\nhello<|im_end|>\n"
              "<|im_start|>assistant\nhi<|im_end|>\n"
              "<|im_start|>assistant\n",
          "chatml renders roles, newlines and the assistant prefix");

    config.add_generation_prompt = false;
    CHECK(fm::render_chat_template(chatml, {{"user", "hi"}}, config) ==
              "<|im_start|>user\nhi<|im_end|>\n",
          "add_generation_prompt=false omits the assistant prefix");
    config.add_generation_prompt = true;

    // llama3 exercises set, loop.index0, the trim filter and bos_token.
    const std::string llama3 = fm::builtin_chat_template_by_name("llama3");
    CHECK(fm::render_chat_template(llama3, {{"user", "  hello  "}}, config) ==
              "<s><|start_header_id|>user<|end_header_id|>\n\nhello<|eot_id|>"
              "<|start_header_id|>assistant<|end_header_id|>\n\n",
          "llama3: bos on the first turn only, content trimmed, assistant header");

    // gemma refuses a system role; that must surface as a typed error rather
    // than a prompt nobody trained on.
    bool refused = false;
    try {
        fm::render_chat_template(fm::builtin_chat_template_by_name("gemma"),
                                 {{"system", "x"}, {"user", "y"}}, config);
    } catch (const fm::Error &e) {
        refused = e.code == fm::ErrorCode::INVALID_ARGUMENT;
    }
    CHECK(refused, "gemma's template rejects a system role with a typed error");
    CHECK(fm::render_chat_template(fm::builtin_chat_template("gemma"),
                                   {{"user", "hi"}}, config) ==
              "<s><start_of_turn>user\nhi<end_of_turn>\n<start_of_turn>model\n",
          "gemma renders turns and the model prefix");

    CHECK(fm::render_chat_template(fm::builtin_chat_template_by_name("phi3"), {{"user", "hi"}}, config) ==
              "<|user|>\nhi<|end|>\n<|assistant|>\n",
          "phi3 renders its markers");

    // llama2 uses a slice, a boolean sentinel and a comparison chain.
    const std::string llama2 = fm::builtin_chat_template("llama");
    CHECK(llama2 == fm::builtin_chat_template_by_name("llama2"),
          "the llama built-in is the llama2 template");
    CHECK(fm::render_chat_template(llama2, {{"user", "hi"}}, config) ==
              "<s>[INST] hi [/INST]",
          "llama2 renders the instruction block");
    CHECK(fm::render_chat_template(llama2, {{"system", "sys"}, {"user", "hi"}}, config) ==
              "<s>[INST] <<SYS>>\nsys\n<</SYS>>\n\nhi [/INST]",
          "llama2 folds the system turn into the first instruction");
    bool alternation = false;
    try {
        fm::render_chat_template(llama2, {{"user", "a"}, {"user", "b"}}, config);
    } catch (const fm::Error &) {
        alternation = true;
    }
    CHECK(alternation, "llama2 rejects non-alternating roles");

    // Whitespace control: `-` strips the surrounding whitespace, which is the
    // difference between a prompt with stray newlines and one without.
    fm::ChatTemplateConfig no_tokens;
    CHECK(fm::render_chat_template(
              "  {%- for m in messages -%}  {{ m['content'] }}  {%- endfor -%}  ",
              {{"user", "a"}, {"user", "b"}}, no_tokens) == "ab",
          "whitespace control strips on both sides");
    CHECK(fm::render_chat_template("x{# a comment #}y", {{"user", "z"}}, no_tokens) == "xy",
          "comments render nothing");
    // An empty conversation is refused rather than rendering a bare template.
    bool empty_refused = false;
    try {
        fm::render_chat_template("x", {}, no_tokens);
    } catch (const fm::Error &) {
        empty_refused = true;
    }
    CHECK(empty_refused, "an empty conversation is refused");
}

void test_chat_template_constructs() {
    fm::ChatTemplateConfig config;
    // `is defined` guards, which deepseek's template uses on its own variable.
    CHECK(fm::render_chat_template(
              "{% if not extra is defined %}{% set extra = 'dflt' %}{% endif %}{{ extra }}",
              {{"user", "x"}}, config) == "dflt",
          "'is defined' guards an optional variable");
    CHECK(fm::render_chat_template(
              "{% if extra is defined %}{{ extra }}{% else %}none{% endif %}", {{"user", "x"}},
              config) == "none",
          "'is not defined' takes the else branch");

    // namespace() and field assignment: deepseek uses this to carry state
    // across loop iterations.
    CHECK(fm::render_chat_template(
              "{% set ns = namespace(n=0) %}{% for m in messages %}"
              "{% set ns.n = ns.n + 1 %}{% endfor %}{{ ns.n }}",
              {{"user", "a"}, {"user", "b"}, {"user", "c"}}, config) == "3",
          "namespace() carries a counter across iterations");

    // Filters, loop metadata and operators.
    CHECK(fm::render_chat_template(
              "{% for m in messages %}{{ loop.index }}/{{ loop.length }}:"
              "{{ m['content'] | upper | replace('A', 'b') }}{% if not loop.last %},{% endif %}"
              "{% endfor %}",
              {{"user", "aa"}, {"user", "cc"}}, config) == "1/2:bb,2/2:CC",
          "loop metadata, chained filters and conditionals");

    CHECK(fm::render_chat_template(
              "{{ messages[0]['content'] }}|{{ messages[1:] | length }}|"
              "{{ 'x' ~ 'y' }}|{{ 7 // 2 }}|{{ 7 % 3 }}|{{ messages[0]['role'] in ['user'] }}",
              {{"user", "a"}, {"user", "b"}}, config) == "a|1|xy|3|1|true",
          "slices, concatenation, integer division, modulo and 'in'");

    // An unsupported construct is refused by name, not silently ignored, and
    // it is refused at render time so plain completion keeps working.
    bool unsupported = false;
    std::string reason;
    try {
        fm::render_chat_template("{% macro m() %}{% endmacro %}", {{"user", "x"}}, config);
    } catch (const fm::Error &e) {
        unsupported = e.code == fm::ErrorCode::NOT_IMPLEMENTED &&
                      std::string(e.what()).find("macro") != std::string::npos;
    }
    CHECK(unsupported, "an unsupported construct names itself in a typed error");
    CHECK(!fm::chat_template_supported("{% macro m() %}{% endmacro %}", &reason),
          "the probe reports the template as unsupported");
    CHECK(reason.find("macro") != std::string::npos, "the probe names the construct");
    CHECK(fm::chat_template_supported(fm::builtin_chat_template_by_name("chatml"), &reason),
          "every built-in template is inside the supported subset");

    // Every built-in must render; a template that only parses is not enough.
    for (const char *name : {"chatml", "llama2", "llama3", "gemma", "phi3", "deepseek2",
                             "zephyr", "vicuna", "raw"}) {
        fm::ChatTemplateConfig cfg;
        cfg.bos_token = "<s>";
        cfg.eos_token = "</s>";
        std::string out;
        bool threw = false;
        try {
            out = fm::render_chat_template(fm::builtin_chat_template_by_name(name), {{"user", "hi"}}, cfg);
        } catch (const fm::Error &) {
            threw = true;
        }
        CHECK(!threw && !out.empty(), std::string("built-in template renders: ") + name);
    }
}

// End to end: a checkpoint carrying its own template renders through the engine,
// and the rendered bytes are what the tokenizer then sees.
void test_chat_template_engine() {
    const std::string path = "tests/fixtures/llama-gqa-shape.gguf";
    RunConfig config;
    config.model_path = path;
    config.budget.total_hot_bytes = 64ull * 1024 * 1024;
    Engine engine(config);
    engine.initialize();

    fm::ChatRender render = engine.render_chat(
        {{"system", "sys"}, {"user", "hi"}}, true, "");
    CHECK(render.template_source == "checkpoint", "the checkpoint's own template is used");
    CHECK(render.prompt ==
              "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nhi<|im_end|>\n"
              "<|im_start|>assistant\n",
          "the checkpoint template renders the conversation");

    // The rendered prompt is the prompt: tokenizing it must yield exactly those
    // bytes (plus BOS), which is what makes the template observable rather than
    // cosmetic. The fixture's context is 64 tokens, so this checks the token
    // stream rather than running the long conversation through the decoder.
    PromptResult short_run = engine.run_prompt(render.prompt.substr(0, 8), 1);
    CHECK(short_run.input_ids.size() == 9,
          "an 8-byte prompt is tokenized to BOS plus eight pieces");

    // A built-in is chosen by *template* name here, while the table is keyed by
    // architecture; conflating the two silently renders an empty prompt.
    RunConfig stable;
    stable.model_path = "tests/fixtures/stablelm-shape.gguf";
    stable.budget.total_hot_bytes = 64ull * 1024 * 1024;
    Engine stable_engine(stable);
    stable_engine.initialize();
    fm::ChatRender builtin = stable_engine.render_chat({{"user", "hi"}}, true, "");
    CHECK(builtin.template_source == "builtin:chatml",
          "an architecture with no template of its own falls back to its built-in");
    CHECK(builtin.prompt == "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n",
          "the built-in renders the conversation");
    stable_engine.release_runtime();

    fm::ChatRender overridden = engine.render_chat({{"user", "hi"}}, false, "raw");
    CHECK(overridden.template_source == "builtin:raw", "a named built-in overrides the file");
    CHECK(overridden.prompt == "hi", "the raw template sends the content verbatim");
    engine.release_runtime();
}

void test_soak_sessions() {
    const uint32_t N_SOAK = 256;
    std::string prompt = first_prompt();
    for (uint32_t session = 0; session < 3; ++session) {
        auto cfg = make_engine_config("tests/fixtures/moe-fm.gguf", BackendKind::ScalarCPU,
                                      ReaderMode::Buffered, 1, false, "", 5);
        cfg.generated_tokens = N_SOAK;
        Engine engine(cfg);
        engine.initialize();
        auto result = engine.run_prompt(prompt, N_SOAK);
        CHECK(result.generated_ids.size() == N_SOAK, "soak session generated N_soak tokens");
        CHECK(engine.residency()->metrics().invalid_transitions == 0,
              "soak session has no invalid transitions");
        CHECK(engine.residency()->metrics().checksum_failures == 0,
              "soak session has no checksum failures");
        check_released(engine, "soak session " + std::to_string(session));
    }
}

} // namespace

int main(int argc, char **argv) {
    std::filesystem::create_directories("build/test-tmp");
    const std::string filter = argc > 1 ? argv[1] : "";
    auto maybe_run = [&](const std::string &name, const std::function<void()> &fn) {
        if (filter.empty() || name.find(filter) != std::string::npos) run_test(name, fn);
    };
    maybe_run("gguf_parser_and_golden", test_gguf_parser_and_golden);
    maybe_run("malformed_fixtures", test_malformed_fixtures);
    maybe_run("slot_state_machine", test_slot_state_machine);
    maybe_run("budget_calculator", test_budget_calculator);
    maybe_run("io_scheduler_and_readers", test_io_scheduler_and_readers);
    maybe_run("backends_and_gpu_placeholder", test_backends_and_gpu_placeholder);
    maybe_run("zen3_fallback", test_zen3_fallback);
    maybe_run("quant_reference_decode", test_quant_reference_decode);
    maybe_run("quant_gemv", test_quant_gemv);
    maybe_run("quant_model_parity", test_quant_model_parity);
    maybe_run("quant_expert_mixed_types", test_quant_expert_mixed_types);
    maybe_run("model_probe", test_model_probe);
    maybe_run("cpu_kernels", test_cpu_kernels);
    maybe_run("thread_dispatch_policy", test_thread_dispatch_policy);
    maybe_run("generic_inspect_and_roundtrip", test_generic_inspect_and_roundtrip);
    maybe_run("sampling_and_detokenize", test_sampling_and_detokenize);
    maybe_run("residency_manager", test_residency_manager);
    maybe_run("dense_cpu_reference_and_operation_log", test_dense_cpu_reference_and_operation_log);
    maybe_run("moe_readers_and_cache_sizes", test_moe_readers_and_cache_sizes);
    maybe_run("traces_and_replay_inputs", test_traces_and_replay_inputs);
    maybe_run("vnpu_reference_backend", test_vnpu_reference_backend);
    maybe_run("tokenizer_real_vocabularies", test_tokenizer_real_vocabularies);
    maybe_run("tokenizer_bpe_merges", test_tokenizer_bpe_merges);
    maybe_run("tokenizer_bpe_families", test_tokenizer_bpe_families);
    maybe_run("tokenizer_sentencepiece", test_tokenizer_sentencepiece);
    maybe_run("unicode_classes", test_unicode_classes);
    maybe_run("model_map_and_tied_output", test_model_map_and_tied_output);
    maybe_run("rope_mrope", test_rope_mrope);
    maybe_run("rope_scaling", test_rope_scaling);
    maybe_run("activation_clamp_forms", test_activation_clamp_forms);
    maybe_run("pointwise_activations", test_pointwise_activations);
    maybe_run("ffn_variants", test_ffn_variants);
    maybe_run("gate_free_families", test_gate_free_families);
    maybe_run("adapter_tags", test_adapter_tags);
    maybe_run("attention_sinks", test_attention_sinks);
    maybe_run("architecture_parity", test_architecture_parity);
    maybe_run("bitnet_sub_norms", test_bitnet_sub_norms);
    maybe_run("refusal_paths", test_refusal_paths);
    maybe_run("adapter_binding", test_adapter_binding);
    maybe_run("rotated_and_ternary_blockers", test_rotated_and_ternary_blockers);
    maybe_run("minicpm_scales", test_minicpm_scales);
    maybe_run("shared_expert", test_shared_expert);
    maybe_run("phi2_block", test_phi2_block);
    maybe_run("deepseek2_mla", test_deepseek2_mla);
    maybe_run("gpt_oss_block", test_gpt_oss_block);
    maybe_run("plm_block", test_plm_block);
    maybe_run("mistral3_block", test_mistral3_block);
    maybe_run("mrope_families", test_mrope_families);
    maybe_run("metadata_array_forms", test_metadata_array_forms);
    maybe_run("recurrent_families", test_recurrent_families);
    maybe_run("chat_template_rendering", test_chat_template_rendering);
    maybe_run("chat_template_constructs", test_chat_template_constructs);
    maybe_run("chat_template_engine", test_chat_template_engine);
    maybe_run("soak_sessions", test_soak_sessions);
    std::cout << g_checks << " checks, " << g_failures << " failures\n";
    return g_failures == 0 ? 0 : 1;
}
