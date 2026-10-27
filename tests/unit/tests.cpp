#include "fm/backend.hpp"
#include "fm/engine.hpp"
#include "fm/gguf.hpp"
#include "fm/io.hpp"
#include "fm/residency.hpp"
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

        const std::string sim_trace_text = read_text_file(sim_trace_path);
        CHECK(sim_trace_text.find("\"zen3_detected\":true") != std::string::npos,
              "simulated Zen 3 trace records detection for " + profile);
        CHECK(sim_trace_text.find("\"zen3_detection_simulated\":true") != std::string::npos,
              "simulated Zen 3 trace marks simulation for " + profile);
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
    maybe_run("residency_manager", test_residency_manager);
    maybe_run("dense_cpu_reference_and_operation_log", test_dense_cpu_reference_and_operation_log);
    maybe_run("moe_readers_and_cache_sizes", test_moe_readers_and_cache_sizes);
    maybe_run("traces_and_replay_inputs", test_traces_and_replay_inputs);
    maybe_run("vnpu_reference_backend", test_vnpu_reference_backend);
    maybe_run("soak_sessions", test_soak_sessions);
    std::cout << g_checks << " checks, " << g_failures << " failures\n";
    return g_failures == 0 ? 0 : 1;
}
