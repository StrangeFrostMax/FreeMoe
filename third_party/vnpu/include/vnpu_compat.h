/*
 * vnpu_compat.h - PCem-free compatibility layer for the original NPU APIs.
 *
 * This header reproduces the original virtual_npu_* and vgpu_npu_* surfaces so
 * old experiments keep compiling.  Every symbol is now backed by the new
 * standalone runtime; no PCem headers, Voodoo renderer, SDL or wxWidgets are
 * required.
 */
#ifndef VNPU_COMPAT_H
#define VNPU_COMPAT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct virtual_npu virtual_npu_t;

virtual_npu_t *virtual_npu_create(size_t vram_bytes);
void            virtual_npu_destroy(virtual_npu_t *n);

uint8_t *virtual_npu_memory(virtual_npu_t *n);
size_t   virtual_npu_memory_size(virtual_npu_t *n);
void     virtual_npu_set_memory_bitmap(virtual_npu_t *n, int enable);
uint64_t virtual_npu_get_bitmap_mask(virtual_npu_t *n, size_t page);

int     virtual_npu_create_team(virtual_npu_t *n, const char *name);
int     virtual_npu_create_thread(virtual_npu_t *n, int team, int priority);
void    virtual_npu_reschedule(virtual_npu_t *n);
void   *virtual_npu_create_area(virtual_npu_t *n, const char *name,
                                size_t size, int lock);
int     virtual_npu_delete_area(virtual_npu_t *n, void *addr);

typedef struct virtual_npu_status {
    int busy;
    int error;
    uint64_t ops;
    uint64_t processed;
    uint64_t culled;
    uint64_t dispatches;
    uint64_t batches;
} virtual_npu_status_t;

void virtual_npu_read_status(virtual_npu_t *n, virtual_npu_status_t *s);
void virtual_npu_reset_acc(virtual_npu_t *n);
void virtual_npu_invalidate(virtual_npu_t *n);
void virtual_npu_reset(virtual_npu_t *n);
void virtual_npu_dump(virtual_npu_t *n);

enum {
    NPU_OP_CPU = 0,
    NPU_OP_VOODOO = 1,
    NPU_OP_WILDCAT = 2,
    NPU_OP_MERGED = 3,
    NPU_OP_SHADER_GEMM = 4,
    NPU_OP_SHADER_MATMUL = 5,
    NPU_OP_FUSED_GEMM = 6,
    NPU_OP_TENSOR_CORE = 7,
    NPU_OP_VNPU_AUTO = 8
};

void virtual_npu_compute(virtual_npu_t *n,
                         size_t a_addr,
                         size_t b_addr,
                         size_t c_addr,
                         size_t ctl_addr,
                         size_t M,
                         size_t K,
                         size_t N,
                         size_t tile_m,
                         size_t tile_k,
                         size_t tile_n,
                         uint32_t op_kind,
                         uint32_t op_code,
                         uint32_t cull_threshold);
void virtual_npu_process(virtual_npu_t *n);

/* Sherlock tracing (real timers; the old hard-coded 0.5 ms is gone). */
enum {
    SHERLOCK_TRAP_DISPATCH = 0,
    SHERLOCK_TRAP_MEMORY = 1,
    SHERLOCK_TRAP_INSTRUCTION = 2,
    SHERLOCK_TRAP_SYNC = 3,
    SHERLOCK_TRAP_ALLOC = 4,
    SHERLOCK_TRAP_TRANSFER = 5,
    SHERLOCK_TRAP_DEPENDENCY = 6,
    SHERLOCK_TRAP_ACCURACY = 7,
    SHERLOCK_TRAP_COMPUTE = 8,
    SHERLOCK_TRAP_COUNT = 9
};

typedef struct sherlock_trace {
    double dispatch_time;
    double memory_time;
    double compute_time;
    double sync_time;
    double total_time;
    uint64_t bytes_read;
    uint64_t bytes_written;
    uint64_t cache_misses;
    uint64_t ops_count;
    int traps_enabled;
} sherlock_trace_t;

void sherlock_enable_tracing(int enable);
void sherlock_reset_trace(void);
void sherlock_dump_trace(const char *label);
void sherlock_get_trace(sherlock_trace_t *trace);
void sherlock_trace_start(virtual_npu_t *n, int trap_type, const char *name);
void sherlock_trace_end(virtual_npu_t *n, int trap_type, uint64_t ops,
                        uint64_t bytes);

/* ------------------------------------------------------------------------- */
/* vgpu_npu compatibility API                                                */
/* ------------------------------------------------------------------------- */

typedef struct vgpu_npu vgpu_npu_t;

vgpu_npu_t *vgpu_npu_create(void);
vgpu_npu_t *vgpu_npu_create_ex(size_t vram_bytes);
void         vgpu_npu_destroy(vgpu_npu_t *npu);

uint8_t *vgpu_npu_vram_ptr(vgpu_npu_t *npu);
size_t   vgpu_npu_vram_size(vgpu_npu_t *npu);
uint32_t vgpu_npu_readl(vgpu_npu_t *npu, uint32_t addr);
void     vgpu_npu_writel(vgpu_npu_t *npu, uint32_t addr, uint32_t val);

typedef struct vgpu_npu_status {
    int busy;
    int error;
    int unit;
    uint64_t results[9];
} vgpu_npu_status_t;

void vgpu_npu_read_status(vgpu_npu_t *npu, vgpu_npu_status_t *s);

enum {
    NPU_RESULT_PIXELS = 0,
    NPU_RESULT_TEXELS = 1,
    NPU_RESULT_TRIS = 2,
    NPU_RESULT_FILLS = 3,
    NPU_RESULT_BLITS = 4,
    NPU_RESULT_TEX_UPLOAD_BYTES = 5,
    NPU_RESULT_ALU_OPS = 6,
    NPU_RESULT_CULLED = 7,
    NPU_RESULT_PROCESSED = 8
};

void vgpu_npu_clear_npu(vgpu_npu_t *npu, uint32_t addr, uint32_t count,
                        uint32_t value);
void vgpu_npu_fill_npu(vgpu_npu_t *npu, uint32_t addr, uint32_t count,
                       uint32_t value);
void vgpu_npu_blit_npu(vgpu_npu_t *npu, uint32_t src_addr,
                       uint32_t dst_addr, uint32_t count, uint32_t ctrl);
void vgpu_npu_texture_upload_npu(vgpu_npu_t *npu, int tmu, uint32_t base,
                                 const uint8_t *src, int w, int h,
                                 size_t stride, int format);
void vgpu_npu_bind_tmu_npu(vgpu_npu_t *npu, int tmu, uint32_t base,
                           int format, int w, int h, int tmudiv);
void vgpu_npu_draw_triangle_npu(vgpu_npu_t *npu,
                                int32_t ax, int32_t ay,
                                int32_t bx, int32_t by,
                                int32_t cx, int32_t cy,
                                uint32_t color0, uint32_t color1,
                                uint32_t fbzMode, uint32_t textureMode0,
                                uint32_t textureMode1, int bilinear);
void vgpu_npu_swapbuffers_npu(vgpu_npu_t *npu);
void vgpu_npu_unit1_shader_load(vgpu_npu_t *npu, uint32_t prog_base,
                                uint32_t prog_size);
void vgpu_npu_unit1_compute_npu(vgpu_npu_t *npu, uint32_t src_addr,
                                uint32_t dst_addr, uint32_t count,
                                uint32_t op, uint32_t cull_addr,
                                uint32_t cull_threshold);
void vgpu_npu_unit1_hierarchical_z_load(vgpu_npu_t *npu,
                                        uint32_t cull_addr,
                                        uint32_t cull_scale);
uint64_t vgpu_npu_unit1_read_result(vgpu_npu_t *npu, int index);
void vgpu_npu_unit1_reset_results(vgpu_npu_t *npu);
void vgpu_npu_process(vgpu_npu_t *npu);

/* Extension: a real matmul command using the new virtual NPU kernels. */
void vgpu_npu_matmul_npu(vgpu_npu_t *npu,
                         uint32_t a_addr, uint32_t b_addr, uint32_t c_addr,
                         uint32_t M, uint32_t K, uint32_t N,
                         uint32_t unit);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VNPU_COMPAT_H */
