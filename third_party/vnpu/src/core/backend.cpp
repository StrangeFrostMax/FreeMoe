/*
 * backend.cpp - stable dynamic backend vtable.
 */
#include "vnpu_backend.h"

extern "C" {

static const vnpu_backend_v1_t g_backend_v1 = {
    VNPU_ABI_VERSION,
    sizeof(vnpu_backend_v1_t),
    "Arena Virtual NPU v1",
    vnpu_device_create,
    vnpu_device_destroy,
    vnpu_device_get_info,
    vnpu_device_get_status,
    vnpu_stream_create,
    vnpu_stream_destroy,
    vnpu_stream_sync,
    vnpu_stream_submit,
    vnpu_event_create,
    vnpu_event_destroy,
    vnpu_event_wait,
    vnpu_buffer_alloc,
    vnpu_buffer_free,
    vnpu_buffer_host_ptr,
    vnpu_matmul_f32,
    vnpu_matmul_i8,
    vnpu_quantize_i8,
    vnpu_pack_weights_i8,
    vnpu_packed_weights_destroy,
    vnpu_rmsnorm_f32,
    vnpu_softmax_f32,
    vnpu_rope_f32,
    vnpu_swiglu_f32,
    vnpu_gelu_f32,
    vnpu_attention_f32,
    vnpu_mtp_head_f32,
    vnpu_trace_enable,
    vnpu_trace_get_summary,
    vnpu_trace_dump,
    vnpu_trace_write_chrome,
    vnpu_autotune_gemm,
    vnpu_cluster_create,
    vnpu_cluster_destroy,
    vnpu_cluster_add_device,
    vnpu_cluster_add_remote,
    vnpu_cluster_cache_weights_f32,
    vnpu_cluster_matmul_f32,
    vnpu_cluster_get_stats,
    vnpu_node_run,
    vnpu_speculative_step,
    vnpu_status_string,
    vnpu_version_string,
    {}
};

VNPU_API const vnpu_backend_v1_t * VNPU_CALL vnpu_get_backend_v1(void)
{
    return &g_backend_v1;
}

} /* extern "C" */
