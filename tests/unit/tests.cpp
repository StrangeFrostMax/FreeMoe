#include "fm/backend.hpp"
#include "fm/engine.hpp"
#include "fm/gguf.hpp"
#include "fm/io.hpp"
#include "fm/residency.hpp"
#include "fm/quant.hpp"
#include "fm/support.hpp"
#include "fm/tokenizer.hpp"
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
    CHECK(!tensor_capability(19).dequantize, "IQ-family types remain decode-unsupported");
    CHECK(tensor_capability(14).dequantize, "Q6_K decoding is implemented");
    CHECK(!tensor_capability(19).parse, "unknown type cannot be parsed");
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
                          std::vector<uint8_t> block(256, 0);
                          float out[256];
                          // Type 19 (IQ-family) has no decoder here.
                          dequantize_row(19, block.data(), out, 256);
                      }),
          "unimplemented block decoder is a typed error");

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
    CHECK(records == 26, "quant reference covers every supported block type");

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
    for (const std::string variant : {"q8_0", "q4_0", "q4_k"}) {
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
    CHECK(!kernel_supports_type(19), "unimplemented type reports no kernel");
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
    CHECK(auto_isa == KernelIsa::Scalar || auto_isa == KernelIsa::Avx2Fma ||
              auto_isa == KernelIsa::Avx512,
          "auto kernel resolution returns a valid ISA");
    if (!host_has_avx2_fma())
        CHECK(auto_isa == KernelIsa::Scalar, "no AVX2 means scalar fallback");
    if (host_is_zen3())
        CHECK(auto_isa != KernelIsa::Avx512, "Zen 3 never selects AVX-512 kernels");
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
    auto sentencepiece_doc = [](bool with_byte_fallback) {
        GGUFDocument doc;
        doc.metadata["tokenizer.ggml.model"] = make_string("llama");
        MetadataValue tokens;
        tokens.type = GGUFValueType::ARRAY;
        auto add = [&](const std::string &piece) {
            tokens.array.push_back(make_string(piece));
        };
        add("<unk>");                        // 0
        add("<s>");                          // 1
        add("</s>");                         // 2
        add(std::string("\xe2\x96\x81") + "the");   // 3  " the"
        add(std::string("\xe2\x96\x81") + "cat");   // 4  " cat"
        add("cat");                          // 5
        add("s");                            // 6
        add(std::string("\xe2\x96\x81") + "sat");   // 7
        // The byte-fallback pieces make arbitrary text representable.
        for (int i = 0; i < 256; ++i) {
            if (!with_byte_fallback) continue;
            char buf[8];
            std::snprintf(buf, sizeof(buf), "<0x%02X>", i);
            add(buf);
        }
        doc.metadata["tokenizer.ggml.tokens"] = tokens;
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
        CHECK(tokenizer.decode({3, 4, 7}, 1, 2) == " the cat sat",
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
void test_architecture_parity() {
    const std::vector<std::string> slugs = {"llama-gqa", "qwen2-bias", "qwen3-qknorm", "gemma",
                                           "gemma2", "phi3-fused", "minicpm", "qwen3moe",
                                           "glm4", "qwen2moe", "phi2-layernorm"};
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
    maybe_run("model_probe", test_model_probe);
    maybe_run("cpu_kernels", test_cpu_kernels);
    maybe_run("generic_inspect_and_roundtrip", test_generic_inspect_and_roundtrip);
    maybe_run("sampling_and_detokenize", test_sampling_and_detokenize);
    maybe_run("residency_manager", test_residency_manager);
    maybe_run("dense_cpu_reference_and_operation_log", test_dense_cpu_reference_and_operation_log);
    maybe_run("moe_readers_and_cache_sizes", test_moe_readers_and_cache_sizes);
    maybe_run("traces_and_replay_inputs", test_traces_and_replay_inputs);
    maybe_run("vnpu_reference_backend", test_vnpu_reference_backend);
    maybe_run("tokenizer_real_vocabularies", test_tokenizer_real_vocabularies);
    maybe_run("model_map_and_tied_output", test_model_map_and_tied_output);
    maybe_run("architecture_parity", test_architecture_parity);
    maybe_run("rotated_and_ternary_blockers", test_rotated_and_ternary_blockers);
    maybe_run("minicpm_scales", test_minicpm_scales);
    maybe_run("shared_expert", test_shared_expert);
    maybe_run("phi2_block", test_phi2_block);
    maybe_run("soak_sessions", test_soak_sessions);
    std::cout << g_checks << " checks, " << g_failures << " failures\n";
    return g_failures == 0 ? 0 : 1;
}
