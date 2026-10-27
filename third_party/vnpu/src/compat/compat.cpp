/*
 * compat.cpp - PCem-free implementation of the original NPU APIs.
 *
 * The legacy flat-memory command model is preserved, but all matmul work is
 * now routed through the optimized standalone runtime.  Graphics-only PCem
 * calls are deterministic no-ops with status accounting; matmul and unit-1
 * compute are real.
 */
#include "vnpu_compat.h"
#include "../internal.h"

#include <cinttypes>

namespace {

bool mul_overflow3(size_t a, size_t b, size_t c)
{
    if (a && b > std::numeric_limits<size_t>::max() / a) return true;
    size_t ab = a * b;
    return c && ab > std::numeric_limits<size_t>::max() / c;
}

uint32_t read_u32(const uint8_t *p)
{
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

void write_u32(uint8_t *p, uint32_t v)
{
    std::memcpy(p, &v, sizeof(v));
}

uint32_t unit1_op(uint32_t a, uint32_t b, uint32_t op)
{
    switch (op & 0xffu) {
    case 0x01: return a + b;
    case 0x02: return a * b;
    case 0x03: return a ^ b;
    case 0x04: return (a & b) | ((~a) & b);
    case 0x05: return (a & b) | (a & ~b);
    case 0x06: return (a & b) | ((~a) & b);
    case 0x07: return (a & b) | (a & ~b);
    default: return a;
    }
}

/* ------------------------------------------------------------------------- */
/* Global Sherlock compatibility state                                       */
/* ------------------------------------------------------------------------- */

struct LegacySherlock {
    std::mutex mutex;
    int enabled = 1;
    double dispatch_time = 0.0;
    double memory_time = 0.0;
    double compute_time = 0.0;
    double sync_time = 0.0;
    uint64_t bytes_read = 0;
    uint64_t bytes_written = 0;
    uint64_t cache_misses = 0;
    uint64_t ops_count = 0;
    uint64_t start_ns = vnpu::now_ns();
};

LegacySherlock g_sherlock;
thread_local uint64_t g_sherlock_start[SHERLOCK_TRAP_COUNT] = {};

/* ------------------------------------------------------------------------- */
/* virtual_npu legacy object                                                  */
/* ------------------------------------------------------------------------- */

struct LegacyCommand {
    size_t a_addr, b_addr, c_addr, ctl_addr;
    size_t M, K, N;
    size_t tile_m, tile_k, tile_n;
    uint32_t op_kind;
    uint32_t op_code;
    uint32_t cull_threshold;
};

struct LegacyArea {
    void *addr;
    size_t size;
    int lock;
};

struct LegacyTeam {
    int id;
    std::string name;
};

struct LegacyThread {
    int id;
    int team;
    int priority;
};

uint32_t legacy_unit_for_op(uint32_t op_kind)
{
    switch (op_kind) {
    case NPU_OP_VOODOO:
    case NPU_OP_MERGED:
        return VNPU_UNIT_VOODOO;
    case NPU_OP_WILDCAT:
    case NPU_OP_SHADER_GEMM:
    case NPU_OP_SHADER_MATMUL:
        return VNPU_UNIT_WILDCAT;
    case NPU_OP_FUSED_GEMM:
    case NPU_OP_TENSOR_CORE:
        return VNPU_UNIT_CELL;
    case NPU_OP_CPU:
    case NPU_OP_VNPU_AUTO:
    default:
        return VNPU_UNIT_AUTO;
    }
}

/* ------------------------------------------------------------------------- */
/* vgpu_npu legacy object                                                     */
/* ------------------------------------------------------------------------- */

enum VgpuCmdType : uint32_t {
    VGPU_CMD_FILL = 1,
    VGPU_CMD_BLIT = 2,
    VGPU_CMD_TEX = 3,
    VGPU_CMD_TRI = 4,
    VGPU_CMD_SWAP = 5,
    VGPU_CMD_COMPUTE = 6,
    VGPU_CMD_MATMUL = 7,
};

struct VgpuCommand {
    uint32_t type = 0;
    uint32_t a = 0, b = 0, c = 0, d = 0;
    uint32_t e = 0, f = 0, g = 0, h = 0;
    uint32_t op = 0;
    uint32_t cull_threshold = 0;
    std::vector<uint8_t> payload;
};

struct TmuState {
    uint32_t base = 0;
    int format = 0;
    int w = 0, h = 0, tmudiv = 0;
};

} /* anonymous namespace */

struct virtual_npu {
    vnpu_device_t *device = nullptr;
    vnpu_stream_t *stream = nullptr;
    uint8_t *vram = nullptr;
    size_t vram_size = 0;
    std::vector<LegacyCommand> commands;
    virtual_npu_status_t acc{};
    std::vector<uint64_t> bitmap;
    size_t bitmap_pages = 0;
    int bitmap_enabled = 1;
    std::vector<LegacyArea> areas;
    std::vector<LegacyTeam> teams;
    std::vector<LegacyThread> threads;
    size_t heap_base = 0x100000;
    int next_team_id = 1;
    int next_thread_id = 1;
    std::mutex mutex;
};

struct vgpu_npu {
    vnpu_device_t *device = nullptr;
    vnpu_stream_t *stream = nullptr;
    uint8_t *vram = nullptr;
    size_t vram_size = 0;
    std::vector<VgpuCommand> commands;
    uint64_t results[9] = {};
    uint64_t unit1_accum[9] = {};
    TmuState tmu[4];
    uint32_t prog_base = 0;
    uint32_t prog_size = 0;
    uint32_t hier_z_addr = 0;
    uint32_t hier_z_scale = 0;
    int busy = 1;
    int error = 0;
    int unit = 1;
    std::mutex mutex;
};

/* ------------------------------------------------------------------------- */
/* virtual_npu API                                                            */
/* ------------------------------------------------------------------------- */

extern "C" {

virtual_npu_t *virtual_npu_create(size_t vram_bytes)
{
    if (!vram_bytes) vram_bytes = 256ull * 1024 * 1024;
    try {
        auto *n = new virtual_npu_t();
        n->vram_size = vram_bytes;
        n->vram = static_cast<uint8_t *>(
            vnpu::reserve_virtual_memory(vram_bytes));
        if (!n->vram) { delete n; return nullptr; }
        vnpu_config_t cfg;
        vnpu_config_init(&cfg);
        cfg.node_name = "virtual_npu_compat";
        if (vnpu_device_create(&cfg, &n->device) != VNPU_OK) {
            vnpu::release_virtual_memory(n->vram, n->vram_size);
            delete n;
            return nullptr;
        }
        if (vnpu_stream_create(n->device, 0, &n->stream) != VNPU_OK) {
            vnpu_device_destroy(n->device);
            vnpu::release_virtual_memory(n->vram, n->vram_size);
            delete n;
            return nullptr;
        }
        n->bitmap_pages = (vram_bytes + 4095) / 4096;
        n->bitmap.resize((n->bitmap_pages + 63) / 64, 0);
        n->teams.push_back(LegacyTeam{1, "npu_init"});
        n->threads.push_back(LegacyThread{1, 1, 10});
        n->acc.busy = 1;
        return n;
    } catch (...) {
        return nullptr;
    }
}

void virtual_npu_destroy(virtual_npu_t *n)
{
    if (!n) return;
    if (n->stream) vnpu_stream_destroy(n->stream);
    if (n->device) vnpu_device_destroy(n->device);
    if (n->vram) vnpu::release_virtual_memory(n->vram, n->vram_size);
    delete n;
}

uint8_t *virtual_npu_memory(virtual_npu_t *n)
{
    return n ? n->vram : nullptr;
}

size_t virtual_npu_memory_size(virtual_npu_t *n)
{
    return n ? n->vram_size : 0;
}

void virtual_npu_set_memory_bitmap(virtual_npu_t *n, int enable)
{
    if (n) n->bitmap_enabled = enable ? 1 : 0;
}

uint64_t virtual_npu_get_bitmap_mask(virtual_npu_t *n, size_t page)
{
    if (!n || !n->bitmap_enabled || page >= n->bitmap_pages) return 0;
    return (n->bitmap[page / 64] >> (page % 64)) & 1ull;
}

int virtual_npu_create_team(virtual_npu_t *n, const char *name)
{
    if (!n) return -1;
    std::lock_guard<std::mutex> lock(n->mutex);
    int id = n->next_team_id++;
    n->teams.push_back(LegacyTeam{id, name ? name : "team"});
    n->threads.push_back(LegacyThread{n->next_thread_id++, id, 10});
    return id;
}

int virtual_npu_create_thread(virtual_npu_t *n, int team, int priority)
{
    if (!n) return -1;
    std::lock_guard<std::mutex> lock(n->mutex);
    int id = n->next_thread_id++;
    n->threads.push_back(LegacyThread{id, team,
                                      priority > 0 && priority <= 100 ?
                                      priority : 10});
    return id;
}

void virtual_npu_reschedule(virtual_npu_t *n)
{
    (void)n; /* cooperative scheduler stub */
}

void *virtual_npu_create_area(virtual_npu_t *n, const char *name,
                              size_t size, int lock)
{
    (void)name;
    if (!n || !size) return nullptr;
    std::lock_guard<std::mutex> guard(n->mutex);
    size_t aligned = vnpu::align_up(size, 4096);
    if (n->heap_base > n->vram_size ||
        aligned > n->vram_size - n->heap_base)
        return nullptr;
    void *addr = n->vram + n->heap_base;
    n->areas.push_back(LegacyArea{addr, aligned, lock});
    size_t start_page = n->heap_base / 4096;
    size_t pages = aligned / 4096;
    for (size_t p = 0; p < pages && start_page + p < n->bitmap_pages; ++p)
        n->bitmap[(start_page + p) / 64] |=
            1ull << ((start_page + p) % 64);
    n->heap_base += aligned;
    return addr;
}

int virtual_npu_delete_area(virtual_npu_t *n, void *addr)
{
    if (!n || !addr) return -1;
    std::lock_guard<std::mutex> guard(n->mutex);
    for (auto it = n->areas.begin(); it != n->areas.end(); ++it) {
        if (it->addr != addr) continue;
        size_t start_page = (static_cast<uint8_t *>(addr) - n->vram) / 4096;
        size_t pages = it->size / 4096;
        for (size_t p = 0; p < pages && start_page + p < n->bitmap_pages; ++p)
            n->bitmap[(start_page + p) / 64] &=
                ~(1ull << ((start_page + p) % 64));
        n->areas.erase(it);
        return 0;
    }
    return -1;
}

void virtual_npu_read_status(virtual_npu_t *n, virtual_npu_status_t *s)
{
    if (!n || !s) return;
    *s = n->acc;
}

void virtual_npu_reset_acc(virtual_npu_t *n)
{
    if (!n) return;
    n->acc.ops = 0;
    n->acc.processed = 0;
    n->acc.culled = 0;
    n->acc.dispatches = 0;
    n->acc.batches = 0;
}

void virtual_npu_invalidate(virtual_npu_t *n)
{
    if (!n) return;
    n->commands.clear();
    n->acc = {};
    n->acc.busy = 1;
}

void virtual_npu_reset(virtual_npu_t *n)
{
    virtual_npu_invalidate(n);
}

void virtual_npu_dump(virtual_npu_t *n)
{
    if (!n) return;
    std::fprintf(stderr,
                 "[virtual_npu] vram=%zu cmds=%zu busy=%d error=%d "
                 "ops=%llu proc=%llu cull=%llu disp=%llu batch=%llu\n",
                 n->vram_size, n->commands.size(), n->acc.busy,
                 n->acc.error,
                 static_cast<unsigned long long>(n->acc.ops),
                 static_cast<unsigned long long>(n->acc.processed),
                 static_cast<unsigned long long>(n->acc.culled),
                 static_cast<unsigned long long>(n->acc.dispatches),
                 static_cast<unsigned long long>(n->acc.batches));
}

void virtual_npu_compute(virtual_npu_t *n,
                         size_t a_addr, size_t b_addr, size_t c_addr,
                         size_t ctl_addr, size_t M, size_t K, size_t N,
                         size_t tile_m, size_t tile_k, size_t tile_n,
                         uint32_t op_kind, uint32_t op_code,
                         uint32_t cull_threshold)
{
    if (!n || !M || !K || !N) return;
    if (mul_overflow3(M, K, 4) || mul_overflow3(K, N, 4) ||
        mul_overflow3(M, N, 4) ||
        M > std::numeric_limits<uint32_t>::max() ||
        K > std::numeric_limits<uint32_t>::max() ||
        N > std::numeric_limits<uint32_t>::max())
        return;
    size_t a_bytes = M * K * 4;
    size_t b_bytes = K * N * 4;
    size_t c_bytes = M * N * 4;
    if (a_addr > n->vram_size || a_bytes > n->vram_size - a_addr ||
        b_addr > n->vram_size || b_bytes > n->vram_size - b_addr ||
        c_addr > n->vram_size || c_bytes > n->vram_size - c_addr) {
        n->acc.error = 1;
        return;
    }
    if (cull_threshold && ctl_addr &&
        (ctl_addr > n->vram_size || c_bytes > n->vram_size - ctl_addr)) {
        n->acc.error = 1;
        return;
    }
    LegacyCommand cmd{};
    cmd.a_addr = a_addr;
    cmd.b_addr = b_addr;
    cmd.c_addr = c_addr;
    cmd.ctl_addr = ctl_addr;
    cmd.M = M; cmd.K = K; cmd.N = N;
    cmd.tile_m = tile_m ? tile_m : 32;
    cmd.tile_k = tile_k ? tile_k : 64;
    cmd.tile_n = tile_n ? tile_n : 32;
    cmd.op_kind = op_kind;
    cmd.op_code = op_code;
    cmd.cull_threshold = cull_threshold;
    n->commands.push_back(cmd);
}

static void legacy_apply_element_cull(uint8_t *vram,
                                      const LegacyCommand &cmd,
                                      uint64_t *culled)
{
    float *C = reinterpret_cast<float *>(vram + cmd.c_addr);
    const uint8_t *ctl = vram + cmd.ctl_addr;
    size_t count = cmd.M * cmd.N;
    for (size_t i = 0; i < count; ++i) {
        if (read_u32(ctl + i * 4) < cmd.cull_threshold) {
            write_u32(reinterpret_cast<uint8_t *>(&C[i]), 0);
            (*culled)++;
        }
    }
}

static void legacy_apply_tile_cull(uint8_t *vram,
                                   const LegacyCommand &cmd,
                                   uint64_t *culled)
{
    float *C = reinterpret_cast<float *>(vram + cmd.c_addr);
    const uint8_t *ctl = vram + cmd.ctl_addr;
    size_t tm = std::max<size_t>(1, cmd.tile_m);
    size_t tn = std::max<size_t>(1, cmd.tile_n);
    for (size_t mr = 0; mr < cmd.M; mr += tm) {
        size_t r = std::min(tm, cmd.M - mr);
        for (size_t nr = 0; nr < cmd.N; nr += tn) {
            size_t c = std::min(tn, cmd.N - nr);
            size_t idx = mr * cmd.N + nr;
            if (idx < cmd.M * cmd.N &&
                read_u32(ctl + idx * 4) < cmd.cull_threshold) {
                for (size_t i = 0; i < r; ++i)
                    for (size_t j = 0; j < c; ++j)
                        write_u32(reinterpret_cast<uint8_t *>(
                            &C[(mr + i) * cmd.N + nr + j]), 0);
                *culled += r * c;
            }
        }
    }
}

void virtual_npu_process(virtual_npu_t *n)
{
    if (!n) return;
    n->acc.busy = 1;
    n->acc.error = 0;
    for (const auto &cmd : n->commands) {
        vnpu_matmul_f32_args_t args{};
        args.A = reinterpret_cast<const float *>(n->vram + cmd.a_addr);
        args.B = reinterpret_cast<const float *>(n->vram + cmd.b_addr);
        args.C = reinterpret_cast<float *>(n->vram + cmd.c_addr);
        args.M = static_cast<uint32_t>(cmd.M);
        args.K = static_cast<uint32_t>(cmd.K);
        args.N = static_cast<uint32_t>(cmd.N);
        args.alpha = 1.0f;
        args.beta = 0.0f;
        args.unit = legacy_unit_for_op(cmd.op_kind);
        vnpu_status_t rc = vnpu_matmul_f32(n->stream, &args);
        if (rc != VNPU_OK) {
            n->acc.error = 1;
            continue;
        }
        uint64_t culled = 0;
        if (cmd.cull_threshold && cmd.ctl_addr) {
            bool tile_cull = cmd.op_kind == NPU_OP_WILDCAT ||
                             cmd.op_kind == NPU_OP_MERGED;
            if (tile_cull)
                legacy_apply_tile_cull(n->vram, cmd, &culled);
            else
                legacy_apply_element_cull(n->vram, cmd, &culled);
        }
        uint64_t outputs = static_cast<uint64_t>(cmd.M) * cmd.N;
        n->acc.ops += 2ull * cmd.M * cmd.K * cmd.N;
        n->acc.processed += outputs - std::min<uint64_t>(outputs, culled);
        n->acc.culled += culled;
        n->acc.dispatches++;
        n->acc.batches++;
    }
    n->commands.clear();
    n->acc.busy = 0;
}

/* ------------------------------------------------------------------------- */
/* Sherlock compatibility API                                                */
/* ------------------------------------------------------------------------- */

void sherlock_enable_tracing(int enable)
{
    std::lock_guard<std::mutex> lock(g_sherlock.mutex);
    g_sherlock.enabled = enable ? 1 : 0;
    if (enable) sherlock_reset_trace();
}

void sherlock_reset_trace(void)
{
    std::lock_guard<std::mutex> lock(g_sherlock.mutex);
    g_sherlock.dispatch_time = 0.0;
    g_sherlock.memory_time = 0.0;
    g_sherlock.compute_time = 0.0;
    g_sherlock.sync_time = 0.0;
    g_sherlock.bytes_read = 0;
    g_sherlock.bytes_written = 0;
    g_sherlock.cache_misses = 0;
    g_sherlock.ops_count = 0;
    g_sherlock.start_ns = vnpu::now_ns();
}

void sherlock_get_trace(sherlock_trace_t *trace)
{
    if (!trace) return;
    std::lock_guard<std::mutex> lock(g_sherlock.mutex);
    trace->dispatch_time = g_sherlock.dispatch_time;
    trace->memory_time = g_sherlock.memory_time;
    trace->compute_time = g_sherlock.compute_time;
    trace->sync_time = g_sherlock.sync_time;
    trace->total_time = static_cast<double>(
        vnpu::now_ns() - g_sherlock.start_ns) / 1e9;
    trace->bytes_read = g_sherlock.bytes_read;
    trace->bytes_written = g_sherlock.bytes_written;
    trace->cache_misses = g_sherlock.cache_misses;
    trace->ops_count = g_sherlock.ops_count;
    trace->traps_enabled = g_sherlock.enabled;
}

void sherlock_dump_trace(const char *label)
{
    sherlock_trace_t t;
    sherlock_get_trace(&t);
    if (!t.traps_enabled) return;
    double total = t.total_time > 0.0 ? t.total_time : 1.0;
    std::printf("\n=== SHERLOCK TRACE [%s] ===\n",
                label ? label : "UNKNOWN");
    std::printf("Total time:     %.6f ms\n", t.total_time * 1000.0);
    std::printf("Dispatch:       %.6f ms (%.1f%%)\n",
                t.dispatch_time * 1000.0, 100.0 * t.dispatch_time / total);
    std::printf("Memory:         %.6f ms (%.1f%%)\n",
                t.memory_time * 1000.0, 100.0 * t.memory_time / total);
    std::printf("Compute:        %.6f ms (%.1f%%)\n",
                t.compute_time * 1000.0, 100.0 * t.compute_time / total);
    std::printf("Sync:           %.6f ms (%.1f%%)\n",
                t.sync_time * 1000.0, 100.0 * t.sync_time / total);
    std::printf("Ops:            %" PRIu64 "\n", t.ops_count);
    std::printf("Bytes read:     %" PRIu64 "\n", t.bytes_read);
    std::printf("Bytes written:  %" PRIu64 "\n", t.bytes_written);
    std::printf("=============================\n\n");
}

void sherlock_trace_start(virtual_npu_t *n, int trap_type, const char *name)
{
    (void)n; (void)name;
    if (trap_type < 0 || trap_type >= SHERLOCK_TRAP_COUNT) return;
    g_sherlock_start[trap_type] = vnpu::now_ns();
}

void sherlock_trace_end(virtual_npu_t *n, int trap_type, uint64_t ops,
                        uint64_t bytes)
{
    (void)n;
    if (trap_type < 0 || trap_type >= SHERLOCK_TRAP_COUNT) return;
    uint64_t start = g_sherlock_start[trap_type];
    double elapsed = start ?
        static_cast<double>(vnpu::now_ns() - start) / 1e9 : 0.0;
    std::lock_guard<std::mutex> lock(g_sherlock.mutex);
    if (!g_sherlock.enabled) return;
    switch (trap_type) {
    case SHERLOCK_TRAP_MEMORY:
    case SHERLOCK_TRAP_ALLOC:
    case SHERLOCK_TRAP_TRANSFER:
        g_sherlock.memory_time += elapsed;
        break;
    case SHERLOCK_TRAP_DISPATCH:
        g_sherlock.dispatch_time += elapsed;
        break;
    case SHERLOCK_TRAP_SYNC:
    case SHERLOCK_TRAP_DEPENDENCY:
        g_sherlock.sync_time += elapsed;
        break;
    case SHERLOCK_TRAP_COMPUTE:
    case SHERLOCK_TRAP_INSTRUCTION:
    case SHERLOCK_TRAP_ACCURACY:
    default:
        g_sherlock.compute_time += elapsed;
        break;
    }
    g_sherlock.ops_count += ops;
    g_sherlock.bytes_read += bytes / 2;
    g_sherlock.bytes_written += bytes / 2;
    g_sherlock_start[trap_type] = 0;
}

/* ------------------------------------------------------------------------- */
/* vgpu_npu API                                                               */
/* ------------------------------------------------------------------------- */

vgpu_npu_t *vgpu_npu_create_ex(size_t vram_bytes)
{
    if (!vram_bytes) vram_bytes = 2ull * 1024 * 1024 * 1024;
    try {
        auto *n = new vgpu_npu_t();
        n->vram_size = vram_bytes;
        n->vram = static_cast<uint8_t *>(
            vnpu::reserve_virtual_memory(vram_bytes));
        if (!n->vram) { delete n; return nullptr; }
        vnpu_config_t cfg;
        vnpu_config_init(&cfg);
        cfg.node_name = "vgpu_npu_compat";
        if (vnpu_device_create(&cfg, &n->device) != VNPU_OK) {
            vnpu::release_virtual_memory(n->vram, n->vram_size);
            delete n;
            return nullptr;
        }
        if (vnpu_stream_create(n->device, 0, &n->stream) != VNPU_OK) {
            vnpu_device_destroy(n->device);
            vnpu::release_virtual_memory(n->vram, n->vram_size);
            delete n;
            return nullptr;
        }
        return n;
    } catch (...) {
        return nullptr;
    }
}

vgpu_npu_t *vgpu_npu_create(void)
{
    return vgpu_npu_create_ex(2ull * 1024 * 1024 * 1024);
}

void vgpu_npu_destroy(vgpu_npu_t *n)
{
    if (!n) return;
    if (n->stream) vnpu_stream_destroy(n->stream);
    if (n->device) vnpu_device_destroy(n->device);
    if (n->vram) vnpu::release_virtual_memory(n->vram, n->vram_size);
    delete n;
}

uint8_t *vgpu_npu_vram_ptr(vgpu_npu_t *n) { return n ? n->vram : nullptr; }
size_t vgpu_npu_vram_size(vgpu_npu_t *n) { return n ? n->vram_size : 0; }

uint32_t vgpu_npu_readl(vgpu_npu_t *n, uint32_t addr)
{
    if (!n || static_cast<size_t>(addr) + 4 > n->vram_size) return 0;
    return read_u32(n->vram + addr);
}

void vgpu_npu_writel(vgpu_npu_t *n, uint32_t addr, uint32_t val)
{
    if (!n || static_cast<size_t>(addr) + 4 > n->vram_size) return;
    write_u32(n->vram + addr, val);
}

void vgpu_npu_read_status(vgpu_npu_t *n, vgpu_npu_status_t *s)
{
    if (!n || !s) return;
    s->busy = n->busy;
    s->error = n->error;
    s->unit = n->unit;
    for (int i = 0; i < 9; ++i) s->results[i] = n->results[i];
}

static void vgpu_push(vgpu_npu_t *n, VgpuCommand cmd)
{
    if (!n) return;
    std::lock_guard<std::mutex> lock(n->mutex);
    n->commands.push_back(std::move(cmd));
}

void vgpu_npu_clear_npu(vgpu_npu_t *n, uint32_t addr, uint32_t count,
                        uint32_t value)
{
    VgpuCommand c{};
    c.type = VGPU_CMD_FILL;
    c.a = addr; c.b = count; c.c = value;
    vgpu_push(n, std::move(c));
}

void vgpu_npu_fill_npu(vgpu_npu_t *n, uint32_t addr, uint32_t count,
                       uint32_t value)
{
    vgpu_npu_clear_npu(n, addr, count, value);
}

void vgpu_npu_blit_npu(vgpu_npu_t *n, uint32_t src_addr,
                       uint32_t dst_addr, uint32_t count, uint32_t ctrl)
{
    (void)ctrl;
    VgpuCommand c{};
    c.type = VGPU_CMD_BLIT;
    c.a = src_addr; c.b = dst_addr; c.c = count;
    vgpu_push(n, std::move(c));
}

void vgpu_npu_texture_upload_npu(vgpu_npu_t *n, int tmu, uint32_t base,
                                 const uint8_t *src, int w, int h,
                                 size_t stride, int format)
{
    if (!n || !src || w <= 0 || h <= 0 || tmu < 0 || tmu >= 4) return;
    if (!stride) stride = static_cast<size_t>(w) * 4;
    VgpuCommand c{};
    c.type = VGPU_CMD_TEX;
    c.a = base;
    c.b = static_cast<uint32_t>(w);
    c.c = static_cast<uint32_t>(h);
    c.d = static_cast<uint32_t>(stride);
    c.e = static_cast<uint32_t>(format);
    c.f = static_cast<uint32_t>(tmu);
    c.payload.resize(static_cast<size_t>(h) * stride);
    std::memcpy(c.payload.data(), src, c.payload.size());
    vgpu_push(n, std::move(c));
}

void vgpu_npu_bind_tmu_npu(vgpu_npu_t *n, int tmu, uint32_t base,
                           int format, int w, int h, int tmudiv)
{
    if (!n || tmu < 0 || tmu >= 4) return;
    n->tmu[tmu] = TmuState{base, format, w, h, tmudiv};
}

void vgpu_npu_draw_triangle_npu(vgpu_npu_t *n,
                                int32_t ax, int32_t ay,
                                int32_t bx, int32_t by,
                                int32_t cx, int32_t cy,
                                uint32_t color0, uint32_t color1,
                                uint32_t fbzMode, uint32_t textureMode0,
                                uint32_t textureMode1, int bilinear)
{
    (void)ax; (void)ay; (void)bx; (void)by; (void)cx; (void)cy;
    (void)color0; (void)color1; (void)fbzMode;
    (void)textureMode0; (void)textureMode1; (void)bilinear;
    VgpuCommand c{};
    c.type = VGPU_CMD_TRI;
    vgpu_push(n, std::move(c));
}

void vgpu_npu_swapbuffers_npu(vgpu_npu_t *n)
{
    VgpuCommand c{};
    c.type = VGPU_CMD_SWAP;
    vgpu_push(n, std::move(c));
}

void vgpu_npu_unit1_shader_load(vgpu_npu_t *n, uint32_t prog_base,
                                uint32_t prog_size)
{
    if (!n) return;
    n->prog_base = prog_base;
    n->prog_size = prog_size;
}

void vgpu_npu_unit1_compute_npu(vgpu_npu_t *n, uint32_t src_addr,
                                uint32_t dst_addr, uint32_t count,
                                uint32_t op, uint32_t cull_addr,
                                uint32_t cull_threshold)
{
    if (!n) return;
    VgpuCommand c{};
    c.type = VGPU_CMD_COMPUTE;
    c.a = src_addr;
    c.b = dst_addr;
    c.c = count;
    c.op = op;
    c.d = cull_addr;
    c.cull_threshold = cull_threshold;
    vgpu_push(n, std::move(c));
}

void vgpu_npu_matmul_npu(vgpu_npu_t *n,
                         uint32_t a_addr, uint32_t b_addr, uint32_t c_addr,
                         uint32_t M, uint32_t K, uint32_t N,
                         uint32_t unit)
{
    if (!n || !M || !K || !N) return;
    VgpuCommand c{};
    c.type = VGPU_CMD_MATMUL;
    c.a = a_addr; c.b = b_addr; c.c = c_addr;
    c.d = M; c.e = K; c.f = N; c.g = unit;
    vgpu_push(n, std::move(c));
}

void vgpu_npu_unit1_hierarchical_z_load(vgpu_npu_t *n,
                                        uint32_t cull_addr,
                                        uint32_t cull_scale)
{
    if (!n) return;
    n->hier_z_addr = cull_addr;
    n->hier_z_scale = cull_scale;
    n->unit1_accum[NPU_RESULT_CULLED] = cull_scale;
}

uint64_t vgpu_npu_unit1_read_result(vgpu_npu_t *n, int index)
{
    if (!n || index < 0 || index >= 9) return 0;
    return n->unit1_accum[index];
}

void vgpu_npu_unit1_reset_results(vgpu_npu_t *n)
{
    if (!n) return;
    for (auto &x : n->unit1_accum) x = 0;
}

static void vgpu_execute_unit1(vgpu_npu_t *n, const VgpuCommand &c)
{
    uint32_t count = c.c;
    uint32_t src = c.a;
    uint32_t dst = c.b;
    uint32_t cull_addr = c.d;
    uint32_t threshold = c.cull_threshold;
    uint64_t processed = 0;
    uint64_t culled = 0;
    uint64_t ops = 0;
    size_t blocks = (count + 63) / 64;
    for (size_t b = 0; b < blocks; ++b) {
        size_t head = b * 64;
        size_t tail = std::min<size_t>(64, count - head);
        bool keep = true;
        if (cull_addr && threshold) {
            size_t ctrl_off = static_cast<size_t>(cull_addr) + head * 4;
            uint32_t ctrl = (ctrl_off + 4 <= n->vram_size) ?
                read_u32(n->vram + ctrl_off) : 0;
            keep = ctrl >= threshold;
        }
        if (!keep) {
            culled += tail;
            continue;
        }
        for (size_t i = 0; i < tail; ++i) {
            size_t si = static_cast<size_t>(src) + (head + i) * 4;
            size_t di = static_cast<size_t>(dst) + (head + i) * 4;
            if (si + 4 > n->vram_size || di + 4 > n->vram_size) break;
            uint32_t a = read_u32(n->vram + si);
            uint32_t out = unit1_op(a, a, c.op);
            write_u32(n->vram + di, out);
            if (c.op & (1u << 8))
                n->unit1_accum[NPU_RESULT_ALU_OPS] += out;
            ops++;
            processed++;
        }
    }
    n->results[NPU_RESULT_PIXELS] += count;
    n->results[NPU_RESULT_PROCESSED] += processed;
    n->results[NPU_RESULT_CULLED] += culled;
    n->results[NPU_RESULT_ALU_OPS] += ops;
    n->unit1_accum[NPU_RESULT_PROCESSED] += processed;
    n->unit1_accum[NPU_RESULT_CULLED] += culled;
    n->unit1_accum[NPU_RESULT_ALU_OPS] += ops;
}

static void vgpu_execute_matmul(vgpu_npu_t *n, const VgpuCommand &c)
{
    uint32_t M = c.d, K = c.e, N = c.f;
    uint64_t a_bytes = static_cast<uint64_t>(M) * K * 4;
    uint64_t b_bytes = static_cast<uint64_t>(K) * N * 4;
    uint64_t c_bytes = static_cast<uint64_t>(M) * N * 4;
    if (static_cast<uint64_t>(c.a) + a_bytes > n->vram_size ||
        static_cast<uint64_t>(c.b) + b_bytes > n->vram_size ||
        static_cast<uint64_t>(c.c) + c_bytes > n->vram_size) {
        n->error = 1;
        return;
    }
    vnpu_matmul_f32_args_t args{};
    args.A = reinterpret_cast<const float *>(n->vram + c.a);
    args.B = reinterpret_cast<const float *>(n->vram + c.b);
    args.C = reinterpret_cast<float *>(n->vram + c.c);
    args.M = M; args.K = K; args.N = N;
    args.alpha = 1.0f;
    args.beta = 0.0f;
    args.unit = c.g;
    if (vnpu_matmul_f32(n->stream, &args) != VNPU_OK) {
        n->error = 1;
        return;
    }
    n->results[NPU_RESULT_ALU_OPS] += 2ull * M * K * N;
    n->results[NPU_RESULT_PROCESSED] += static_cast<uint64_t>(M) * N;
}

void vgpu_npu_process(vgpu_npu_t *n)
{
    if (!n) return;
    n->busy = 1;
    n->error = 0;
    std::vector<VgpuCommand> commands;
    {
        std::lock_guard<std::mutex> lock(n->mutex);
        commands.swap(n->commands);
    }
    for (const auto &c : commands) {
        switch (c.type) {
        case VGPU_CMD_FILL: {
            size_t off = c.a;
            for (uint32_t i = 0; i < c.b; ++i) {
                if (off + 4 > n->vram_size) break;
                write_u32(n->vram + off, c.c);
                off += 4;
            }
            n->results[NPU_RESULT_FILLS] += c.b;
            n->results[NPU_RESULT_PIXELS] += c.b;
            break;
        }
        case VGPU_CMD_BLIT: {
            size_t src = c.a, dst = c.b;
            for (uint32_t i = 0; i < c.c; ++i) {
                if (src + 4 > n->vram_size || dst + 4 > n->vram_size) break;
                uint32_t v = read_u32(n->vram + src);
                write_u32(n->vram + dst, v);
                src += 4; dst += 4;
            }
            n->results[NPU_RESULT_BLITS] += c.c;
            break;
        }
        case VGPU_CMD_TEX: {
            size_t base = c.a;
            size_t stride = c.d;
            size_t bytes = static_cast<size_t>(c.c) * stride;
            if (base + bytes <= n->vram_size && c.payload.size() >= bytes)
                std::memcpy(n->vram + base, c.payload.data(), bytes);
            n->results[NPU_RESULT_TEXELS] +=
                static_cast<uint64_t>(c.b) * c.c;
            n->results[NPU_RESULT_TEX_UPLOAD_BYTES] += bytes;
            break;
        }
        case VGPU_CMD_TRI:
            n->results[NPU_RESULT_TRIS]++;
            break;
        case VGPU_CMD_SWAP:
            n->busy = 0;
            break;
        case VGPU_CMD_COMPUTE:
            vgpu_execute_unit1(n, c);
            break;
        case VGPU_CMD_MATMUL:
            vgpu_execute_matmul(n, c);
            break;
        default:
            n->error = 1;
            break;
        }
    }
    n->busy = 0;
}

} /* extern "C" */
