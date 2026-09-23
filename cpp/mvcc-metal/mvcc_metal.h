// mvcc Metal shim: a flat C ABI over the pieces of Metal the runtime needs.
// Deliberately thin: no policy lives here, only 1:1 wrappers, so the Rust runtime owns all decisions.
// Handles are retained Objective-C objects; every *_release() balances one *_new()/create.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* mvccm_device_t;
typedef void* mvccm_buffer_t;
typedef void* mvccm_residency_t;
typedef void* mvccm_library_t;
typedef void* mvccm_pipeline_t;
typedef void* mvccm_queue_t;
typedef void* mvccm_cmdbuf_t;
typedef void* mvccm_encoder_t;   // compute encoder
typedef void* mvccm_blit_t;
typedef void* mvccm_event_t;     // MTLSharedEvent
typedef void* mvccm_archive_t;   // MTLBinaryArchive

// ---- device ----
mvccm_device_t mvccm_device_create(void);
void mvccm_device_release(mvccm_device_t);
// name copied into buf (NUL-terminated, truncated to cap)
void mvccm_device_name(mvccm_device_t, char* buf, size_t cap);
uint64_t mvccm_device_recommended_working_set(mvccm_device_t);
uint64_t mvccm_device_current_allocated(mvccm_device_t);
uint64_t mvccm_device_registry_id(mvccm_device_t);
uint32_t mvccm_device_max_threadgroup_memory(mvccm_device_t);
uint32_t mvccm_device_core_count(mvccm_device_t);  // IORegistry gpu-core-count; 0 if unavailable
int mvccm_device_supports_metal4(mvccm_device_t);
uint64_t mvccm_physical_memory(void);

// ---- buffers (shared storage, hazard tracking untracked) ----
mvccm_buffer_t mvccm_buffer_new(mvccm_device_t, uint64_t length);
// wrap page-aligned host memory without copying; returns NULL if not possible
mvccm_buffer_t mvccm_buffer_wrap_host(mvccm_device_t, void* ptr, uint64_t length);
void* mvccm_buffer_contents(mvccm_buffer_t);
uint64_t mvccm_buffer_gpu_address(mvccm_buffer_t);
uint64_t mvccm_buffer_length(mvccm_buffer_t);
void mvccm_buffer_release(mvccm_buffer_t);

// ---- residency ----
mvccm_residency_t mvccm_residency_new(mvccm_device_t);
void mvccm_residency_add_buffer(mvccm_residency_t, mvccm_buffer_t);
void mvccm_residency_remove_buffer(mvccm_residency_t, mvccm_buffer_t);
void mvccm_residency_commit(mvccm_residency_t);
void mvccm_residency_release(mvccm_residency_t);

// ---- libraries / pipelines ----
// err_out (optional): malloc'd NUL-terminated message the caller frees with mvccm_free_string
mvccm_library_t mvccm_library_from_source(mvccm_device_t, const char* src, size_t len, int metal4, int fast_math, char** err_out);
/* same, with preprocessor definitions: `defines` is a NUL-terminated list of "NAME" or "NAME=VALUE" separated by ';' (may be NULL) */
mvccm_library_t mvccm_library_from_source_defs(mvccm_device_t, const char* src, size_t len, int metal4, int fast_math, const char* defines, char** err_out);
mvccm_library_t mvccm_library_from_data(mvccm_device_t, const void* bytes, size_t len, char** err_out);
void mvccm_library_release(mvccm_library_t);
mvccm_pipeline_t mvccm_pipeline_new(mvccm_device_t, mvccm_library_t, const char* fn_name, uint32_t max_threads_hint, mvccm_archive_t archive_or_null, char** err_out);
uint32_t mvccm_pipeline_max_threads(mvccm_pipeline_t);
uint32_t mvccm_pipeline_exec_width(mvccm_pipeline_t);
uint32_t mvccm_pipeline_static_tg_mem(mvccm_pipeline_t);
void mvccm_pipeline_release(mvccm_pipeline_t);
mvccm_archive_t mvccm_archive_open(mvccm_device_t, const char* path_or_null, char** err_out);
int mvccm_archive_serialize(mvccm_archive_t, const char* path, char** err_out);
void mvccm_archive_release(mvccm_archive_t);
void mvccm_free_string(char*);

// ---- queues / command buffers ----
mvccm_queue_t mvccm_queue_new(mvccm_device_t, mvccm_residency_t residency_or_null);
void mvccm_queue_release(mvccm_queue_t);
mvccm_cmdbuf_t mvccm_cmdbuf_new(mvccm_queue_t);
void mvccm_cmdbuf_release(mvccm_cmdbuf_t);
// Ordering model: every buffer is hazard-tracking-untracked (kernels receive raw GPU addresses), so Metal
// inserts no dependencies on its own. CUDA stream order is rebuilt explicitly: a memory barrier after each
// dispatch (concurrent-type encoder), an MTLFence handed from encoder to encoder, and a shared-event wait at
// the start of each command buffer on the value its predecessor signals.
typedef void* mvccm_fence_t;
mvccm_fence_t mvccm_device_new_fence(mvccm_device_t);
void mvccm_fence_release(mvccm_fence_t);
mvccm_encoder_t mvccm_compute_begin(mvccm_cmdbuf_t);   // concurrent dispatch type; call mvccm_compute_barrier between dependent dispatches
void mvccm_compute_wait_fence(mvccm_encoder_t, mvccm_fence_t);
void mvccm_compute_update_fence(mvccm_encoder_t, mvccm_fence_t);
void mvccm_blit_wait_fence(mvccm_blit_t, mvccm_fence_t);
void mvccm_blit_update_fence(mvccm_blit_t, mvccm_fence_t);
void mvccm_compute_set_pipeline(mvccm_encoder_t, mvccm_pipeline_t);
void mvccm_compute_set_bytes(mvccm_encoder_t, const void* data, size_t len, uint32_t index);
void mvccm_compute_set_buffer(mvccm_encoder_t, mvccm_buffer_t, uint64_t offset, uint32_t index);
// Untracked buffer the encoder will read/write via a raw GPU address in the param block.
void mvccm_compute_use_buffer(mvccm_encoder_t, mvccm_buffer_t);
void mvccm_compute_set_tg_mem(mvccm_encoder_t, uint32_t len, uint32_t index);
void mvccm_compute_dispatch(mvccm_encoder_t, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t bx, uint32_t by, uint32_t bz);
void mvccm_compute_dispatch_indirect(mvccm_encoder_t, mvccm_buffer_t args, uint64_t offset, uint32_t bx, uint32_t by, uint32_t bz);
void mvccm_compute_barrier(mvccm_encoder_t);  // memoryBarrierWithScope(buffers)
void mvccm_compute_end(mvccm_encoder_t);
mvccm_blit_t mvccm_blit_begin(mvccm_cmdbuf_t);
void mvccm_blit_copy(mvccm_blit_t, mvccm_buffer_t src, uint64_t soff, mvccm_buffer_t dst, uint64_t doff, uint64_t len);
void mvccm_blit_fill(mvccm_blit_t, mvccm_buffer_t dst, uint64_t off, uint64_t len, uint8_t value);
void mvccm_blit_end(mvccm_blit_t);
void mvccm_cmdbuf_signal_event(mvccm_cmdbuf_t, mvccm_event_t, uint64_t value);
void mvccm_cmdbuf_wait_event(mvccm_cmdbuf_t, mvccm_event_t, uint64_t value);
// callback runs on Metal's completion thread with `ctx`
void mvccm_cmdbuf_on_completed(mvccm_cmdbuf_t, void (*cb)(void* ctx, int32_t status), void* ctx);
void mvccm_cmdbuf_commit(mvccm_cmdbuf_t);
void mvccm_cmdbuf_wait(mvccm_cmdbuf_t);
// 0 = not enqueued/committed, 1 = committed/scheduled, 2 = completed ok, 3 = error
int32_t mvccm_cmdbuf_status(mvccm_cmdbuf_t);
// error message (malloc'd) or NULL
char* mvccm_cmdbuf_error(mvccm_cmdbuf_t);
double mvccm_cmdbuf_gpu_start(mvccm_cmdbuf_t);
double mvccm_cmdbuf_gpu_end(mvccm_cmdbuf_t);

// ---- shared events ----
mvccm_event_t mvccm_event_new(mvccm_device_t);
uint64_t mvccm_event_value(mvccm_event_t);
void mvccm_event_set_value(mvccm_event_t, uint64_t);
// blocks until signaled value >= value or timeout (ms; <0 = forever). returns 1 if reached.
int mvccm_event_wait(mvccm_event_t, uint64_t value, int64_t timeout_ms);
void mvccm_event_release(mvccm_event_t);

// ---- misc ----
double mvccm_now_seconds(void);  // same clock as GPU timestamps (mach_absolute_time based)

#ifdef __cplusplus
}
#endif
