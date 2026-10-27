/*
 * vnpu_backend.h - dynamic backend discovery.
 *
 * An LLM engine can dlopen/LoadLibrary the shared object, call
 * vnpu_get_backend_v1(), and use the returned function table without linking
 * against the library at build time.  The table never owns memory and never
 * changes within a process.
 */
#ifndef VNPU_BACKEND_H
#define VNPU_BACKEND_H

#include "vnpu.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VNPU_BACKEND_V1_SIZE 48u

typedef vnpu_status_t (VNPU_CALL *vnpu_device_create_fn)(
    const vnpu_config_t *, vnpu_device_t **);
typedef void (VNPU_CALL *vnpu_device_destroy_fn)(vnpu_device_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_device_get_info_fn)(
    const vnpu_device_t *, vnpu_device_info_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_device_get_status_fn)(
    const vnpu_device_t *, vnpu_device_status_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_stream_create_fn)(
    vnpu_device_t *, int, vnpu_stream_t **);
typedef void (VNPU_CALL *vnpu_stream_destroy_fn)(vnpu_stream_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_stream_sync_fn)(vnpu_stream_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_stream_submit_fn)(
    vnpu_stream_t *, const vnpu_command_t *, uint64_t, vnpu_event_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_event_create_fn)(vnpu_event_t **);
typedef void (VNPU_CALL *vnpu_event_destroy_fn)(vnpu_event_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_event_wait_fn)(vnpu_event_t *,
                                                      uint32_t);
typedef vnpu_status_t (VNPU_CALL *vnpu_buffer_alloc_fn)(
    vnpu_device_t *, uint64_t, vnpu_buffer_t **);
typedef void (VNPU_CALL *vnpu_buffer_free_fn)(vnpu_buffer_t *);
typedef void * (VNPU_CALL *vnpu_buffer_host_ptr_fn)(vnpu_buffer_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_matmul_f32_fn)(
    vnpu_stream_t *, const vnpu_matmul_f32_args_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_matmul_i8_fn)(
    vnpu_stream_t *, const vnpu_matmul_i8_args_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_quantize_i8_fn)(
    vnpu_stream_t *, const vnpu_quantize_i8_args_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_pack_weights_i8_fn)(
    vnpu_device_t *, const int8_t *, const float *, uint32_t, uint32_t,
    uint32_t, uint32_t, vnpu_packed_weights_t **);
typedef void (VNPU_CALL *vnpu_packed_weights_destroy_fn)(
    vnpu_packed_weights_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_rmsnorm_f32_fn)(
    vnpu_stream_t *, const vnpu_rmsnorm_f32_args_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_softmax_f32_fn)(
    vnpu_stream_t *, const vnpu_softmax_f32_args_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_rope_f32_fn)(
    vnpu_stream_t *, const vnpu_rope_f32_args_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_swiglu_f32_fn)(
    vnpu_stream_t *, const vnpu_swiglu_f32_args_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_gelu_f32_fn)(
    vnpu_stream_t *, const vnpu_unary_f32_args_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_attention_f32_fn)(
    vnpu_stream_t *, const vnpu_attention_f32_args_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_mtp_head_f32_fn)(
    vnpu_stream_t *, const vnpu_mtp_head_f32_args_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_trace_enable_fn)(vnpu_device_t *, int);
typedef vnpu_status_t (VNPU_CALL *vnpu_trace_get_summary_fn)(
    const vnpu_device_t *, vnpu_trace_summary_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_trace_dump_fn)(
    const vnpu_device_t *, const char *);
typedef vnpu_status_t (VNPU_CALL *vnpu_trace_write_chrome_fn)(
    const vnpu_device_t *, const char *);
typedef vnpu_status_t (VNPU_CALL *vnpu_autotune_gemm_fn)(
    vnpu_device_t *, uint32_t, uint32_t, uint32_t, uint32_t,
    vnpu_dtype_t, vnpu_tune_result_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_cluster_create_fn)(
    const vnpu_cluster_config_t *, vnpu_cluster_t **);
typedef void (VNPU_CALL *vnpu_cluster_destroy_fn)(vnpu_cluster_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_cluster_add_device_fn)(
    vnpu_cluster_t *, vnpu_device_t *, uint32_t);
typedef vnpu_status_t (VNPU_CALL *vnpu_cluster_add_remote_fn)(
    vnpu_cluster_t *, const vnpu_remote_node_config_t *,
    vnpu_remote_node_t **);
typedef vnpu_status_t (VNPU_CALL *vnpu_cluster_cache_weights_f32_fn)(
    vnpu_cluster_t *, const char *, const float *, uint32_t, uint32_t,
    uint32_t, vnpu_weight_handle_t **);
typedef vnpu_status_t (VNPU_CALL *vnpu_cluster_matmul_f32_fn)(
    vnpu_cluster_t *, vnpu_weight_handle_t *, const float *, float *,
    uint32_t, uint32_t, uint32_t, uint32_t);
typedef vnpu_status_t (VNPU_CALL *vnpu_cluster_get_stats_fn)(
    const vnpu_cluster_t *, vnpu_cluster_stats_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_node_run_fn)(
    const vnpu_node_config_t *);
typedef vnpu_status_t (VNPU_CALL *vnpu_speculative_step_fn)(
    vnpu_cluster_t *, const vnpu_speculative_config_t *, const int32_t *,
    uint32_t, vnpu_logits_fn_t, void *, vnpu_logits_fn_t, void *,
    vnpu_speculative_result_t *);
typedef const char * (VNPU_CALL *vnpu_status_string_fn)(vnpu_status_t);
typedef const char * (VNPU_CALL *vnpu_version_string_fn)(void);

typedef struct vnpu_backend_v1_t {
    uint32_t abi_version;
    uint32_t table_size;
    const char *backend_name;
    vnpu_device_create_fn device_create;
    vnpu_device_destroy_fn device_destroy;
    vnpu_device_get_info_fn device_get_info;
    vnpu_device_get_status_fn device_get_status;
    vnpu_stream_create_fn stream_create;
    vnpu_stream_destroy_fn stream_destroy;
    vnpu_stream_sync_fn stream_sync;
    vnpu_stream_submit_fn stream_submit;
    vnpu_event_create_fn event_create;
    vnpu_event_destroy_fn event_destroy;
    vnpu_event_wait_fn event_wait;
    vnpu_buffer_alloc_fn buffer_alloc;
    vnpu_buffer_free_fn buffer_free;
    vnpu_buffer_host_ptr_fn buffer_host_ptr;
    vnpu_matmul_f32_fn matmul_f32;
    vnpu_matmul_i8_fn matmul_i8;
    vnpu_quantize_i8_fn quantize_i8;
    vnpu_pack_weights_i8_fn pack_weights_i8;
    vnpu_packed_weights_destroy_fn packed_weights_destroy;
    vnpu_rmsnorm_f32_fn rmsnorm_f32;
    vnpu_softmax_f32_fn softmax_f32;
    vnpu_rope_f32_fn rope_f32;
    vnpu_swiglu_f32_fn swiglu_f32;
    vnpu_gelu_f32_fn gelu_f32;
    vnpu_attention_f32_fn attention_f32;
    vnpu_mtp_head_f32_fn mtp_head_f32;
    vnpu_trace_enable_fn trace_enable;
    vnpu_trace_get_summary_fn trace_get_summary;
    vnpu_trace_dump_fn trace_dump;
    vnpu_trace_write_chrome_fn trace_write_chrome;
    vnpu_autotune_gemm_fn autotune_gemm;
    vnpu_cluster_create_fn cluster_create;
    vnpu_cluster_destroy_fn cluster_destroy;
    vnpu_cluster_add_device_fn cluster_add_device;
    vnpu_cluster_add_remote_fn cluster_add_remote;
    vnpu_cluster_cache_weights_f32_fn cluster_cache_weights_f32;
    vnpu_cluster_matmul_f32_fn cluster_matmul_f32;
    vnpu_cluster_get_stats_fn cluster_get_stats;
    vnpu_node_run_fn node_run;
    vnpu_speculative_step_fn speculative_step;
    vnpu_status_string_fn status_string;
    vnpu_version_string_fn version_string;
    void *reserved[8];
} vnpu_backend_v1_t;

VNPU_API const vnpu_backend_v1_t * VNPU_CALL vnpu_get_backend_v1(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* VNPU_BACKEND_H */
