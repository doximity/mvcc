// mvcc Metal shim implementation. See mvcc_metal.h.
#import <Foundation/Foundation.h>
#import <IOKit/IOKitLib.h>
#import <Metal/Metal.h>
#include <mach/mach_time.h>
#include <sys/sysctl.h>
#include <cstdlib>
#include <cstring>
#include "mvcc_metal.h"

// Anchor defined in mvcc_autolink.s; referencing it drags the LC_LINKER_OPTION member out of the static archive.
extern "C" int __mvcc_autolink_anchor;
extern "C" __attribute__((used)) void* mvccm_autolink_ref(void) { return &__mvcc_autolink_anchor; }

#define RET(obj) ((__bridge_retained void*)(obj))
#define GET(T, h) ((__bridge T)(h))
#define REL(h) do { if (h) { id __o = (__bridge_transfer id)(h); (void)__o; } } while (0)

static char* dupstr(NSString* s) {
  if (!s) return nullptr;
  const char* u = s.UTF8String;
  char* r = (char*)malloc(strlen(u) + 1);
  strcpy(r, u);
  return r;
}
static void set_err(char** out, NSError* e, const char* fallback) {
  if (!out) return;
  *out = e ? dupstr(e.localizedDescription) : dupstr([NSString stringWithUTF8String:fallback]);
}
void mvccm_free_string(char* s) { free(s); }

// ---- device ----
mvccm_device_t mvccm_device_create(void) { @autoreleasepool { return RET(MTLCreateSystemDefaultDevice()); } }
void mvccm_device_release(mvccm_device_t d) { REL(d); }
void mvccm_device_name(mvccm_device_t d, char* buf, size_t cap) {
  @autoreleasepool { const char* n = GET(id<MTLDevice>, d).name.UTF8String; strncpy(buf, n, cap - 1); buf[cap - 1] = 0; }
}
uint64_t mvccm_device_recommended_working_set(mvccm_device_t d) { return GET(id<MTLDevice>, d).recommendedMaxWorkingSetSize; }
uint64_t mvccm_device_current_allocated(mvccm_device_t d) { return GET(id<MTLDevice>, d).currentAllocatedSize; }
uint64_t mvccm_device_registry_id(mvccm_device_t d) { return GET(id<MTLDevice>, d).registryID; }
uint32_t mvccm_device_max_threadgroup_memory(mvccm_device_t d) { return (uint32_t)GET(id<MTLDevice>, d).maxThreadgroupMemoryLength; }
int mvccm_device_supports_metal4(mvccm_device_t d) {
  if (@available(macOS 26.0, *)) return [GET(id<MTLDevice>, d) supportsFamily:MTLGPUFamilyMetal4];
  return 0;
}
uint64_t mvccm_physical_memory(void) { return [NSProcessInfo processInfo].physicalMemory; }
uint32_t mvccm_device_core_count(mvccm_device_t d) {
  @autoreleasepool {
    uint64_t rid = GET(id<MTLDevice>, d).registryID;
    io_iterator_t it;
    if (IOServiceGetMatchingServices(kIOMainPortDefault, IORegistryEntryIDMatching(rid), &it) != KERN_SUCCESS) return 0;
    uint32_t cores = 0;
    io_object_t obj;
    while ((obj = IOIteratorNext(it))) {
      CFTypeRef v = IORegistryEntrySearchCFProperty(obj, kIOServicePlane, CFSTR("gpu-core-count"), kCFAllocatorDefault, kIORegistryIterateRecursively | kIORegistryIterateParents);
      if (v) { if (CFGetTypeID(v) == CFNumberGetTypeID()) { int32_t n = 0; CFNumberGetValue((CFNumberRef)v, kCFNumberSInt32Type, &n); cores = (uint32_t)n; } CFRelease(v); }
      IOObjectRelease(obj);
      if (cores) break;
    }
    IOObjectRelease(it);
    return cores;
  }
}

// ---- buffers ----
static const MTLResourceOptions kBufOpts = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked | MTLResourceCPUCacheModeDefaultCache;
mvccm_buffer_t mvccm_buffer_new(mvccm_device_t d, uint64_t length) {
  @autoreleasepool { return RET([GET(id<MTLDevice>, d) newBufferWithLength:(NSUInteger)length options:kBufOpts]); }
}
mvccm_buffer_t mvccm_buffer_wrap_host(mvccm_device_t d, void* ptr, uint64_t length) {
  @autoreleasepool {
    if (((uintptr_t)ptr & 0x3fff) != 0 || (length & 0x3fff) != 0) return nullptr;
    return RET([GET(id<MTLDevice>, d) newBufferWithBytesNoCopy:ptr length:(NSUInteger)length options:kBufOpts deallocator:nil]);
  }
}
// ARC retains and autoreleases the receiver of these property reads. Without a
// pool in scope the release is deferred to one that never arrives: a runtime
// loaded as a library is never inside someone else's pool, so every call would
// pin the buffer a little harder and cudaFree would stop returning its memory.
void* mvccm_buffer_contents(mvccm_buffer_t b) { @autoreleasepool { return GET(id<MTLBuffer>, b).contents; } }
uint64_t mvccm_buffer_gpu_address(mvccm_buffer_t b) { @autoreleasepool { return GET(id<MTLBuffer>, b).gpuAddress; } }
uint64_t mvccm_buffer_length(mvccm_buffer_t b) { @autoreleasepool { return GET(id<MTLBuffer>, b).length; } }
void mvccm_buffer_release(mvccm_buffer_t b) { REL(b); }

// ---- residency ----
mvccm_residency_t mvccm_residency_new(mvccm_device_t d) {
  @autoreleasepool {
    MTLResidencySetDescriptor* desc = [MTLResidencySetDescriptor new];
    desc.initialCapacity = 1024;
    NSError* err = nil;
    id<MTLResidencySet> rs = [GET(id<MTLDevice>, d) newResidencySetWithDescriptor:desc error:&err];
    return RET(rs);
  }
}
void mvccm_residency_add_buffer(mvccm_residency_t r, mvccm_buffer_t b) { [GET(id<MTLResidencySet>, r) addAllocation:GET(id<MTLBuffer>, b)]; }
void mvccm_residency_remove_buffer(mvccm_residency_t r, mvccm_buffer_t b) { [GET(id<MTLResidencySet>, r) removeAllocation:GET(id<MTLBuffer>, b)]; }
void mvccm_residency_commit(mvccm_residency_t r) { [GET(id<MTLResidencySet>, r) commit]; }
void mvccm_residency_release(mvccm_residency_t r) { REL(r); }

// ---- libraries / pipelines ----
mvccm_library_t mvccm_library_from_source(mvccm_device_t d, const char* src, size_t len, int metal4, int fast_math, char** err_out) {
  return mvccm_library_from_source_defs(d, src, len, metal4, fast_math, nullptr, err_out);
}
mvccm_library_t mvccm_library_from_source_defs(mvccm_device_t d, const char* src, size_t len, int metal4, int fast_math, const char* defines, char** err_out) {
  @autoreleasepool {
    NSString* s = [[NSString alloc] initWithBytes:src length:len encoding:NSUTF8StringEncoding];
    MTLCompileOptions* o = [MTLCompileOptions new];
    o.languageVersion = metal4 ? MTLLanguageVersion4_0 : MTLLanguageVersion3_2;
    o.mathMode = fast_math ? MTLMathModeFast : MTLMathModeSafe;
    o.mathFloatingPointFunctions = fast_math ? MTLMathFloatingPointFunctionsFast : MTLMathFloatingPointFunctionsPrecise;
    if (defines && *defines) {
      NSMutableDictionary* macros = [NSMutableDictionary new];
      for (NSString* item in [[NSString stringWithUTF8String:defines] componentsSeparatedByString:@";"]) {
        if (item.length == 0) continue;
        NSRange eq = [item rangeOfString:@"="];
        if (eq.location == NSNotFound) macros[item] = @"1";
        else macros[[item substringToIndex:eq.location]] = [item substringFromIndex:eq.location + 1];
      }
      o.preprocessorMacros = macros;
    }
    NSError* err = nil;
    id<MTLLibrary> lib = [GET(id<MTLDevice>, d) newLibraryWithSource:s options:o error:&err];
    if (!lib) { set_err(err_out, err, "newLibraryWithSource failed"); return nullptr; }
    return RET(lib);
  }
}
mvccm_library_t mvccm_library_from_data(mvccm_device_t d, const void* bytes, size_t len, char** err_out) {
  @autoreleasepool {
    dispatch_data_t dd = dispatch_data_create(bytes, len, nullptr, DISPATCH_DATA_DESTRUCTOR_DEFAULT);
    NSError* err = nil;
    id<MTLLibrary> lib = [GET(id<MTLDevice>, d) newLibraryWithData:dd error:&err];
    if (!lib) { set_err(err_out, err, "newLibraryWithData failed"); return nullptr; }
    return RET(lib);
  }
}
void mvccm_library_release(mvccm_library_t l) { REL(l); }
mvccm_pipeline_t mvccm_pipeline_new(mvccm_device_t d, mvccm_library_t l, const char* fn_name, uint32_t max_threads_hint, mvccm_archive_t archive, char** err_out) {
  @autoreleasepool {
    id<MTLFunction> fn = [GET(id<MTLLibrary>, l) newFunctionWithName:[NSString stringWithUTF8String:fn_name]];
    if (!fn) { set_err(err_out, nil, "kernel not found in library"); return nullptr; }
    MTLComputePipelineDescriptor* desc = [MTLComputePipelineDescriptor new];
    desc.computeFunction = fn;
    desc.label = [NSString stringWithUTF8String:fn_name];
    if (max_threads_hint) desc.maxTotalThreadsPerThreadgroup = max_threads_hint;
    // Threadgroup size is fixed per launch in CUDA, so let the compiler assume the grid is a multiple of the width.
    desc.threadGroupSizeIsMultipleOfThreadExecutionWidth = YES;
    if (archive) desc.binaryArchives = @[GET(id<MTLBinaryArchive>, archive)];
    NSError* err = nil;
    id<MTLComputePipelineState> ps = [GET(id<MTLDevice>, d) newComputePipelineStateWithDescriptor:desc options:MTLPipelineOptionNone reflection:nil error:&err];
    if (!ps) { set_err(err_out, err, "newComputePipelineState failed"); return nullptr; }
    if (archive) { NSError* aerr = nil; [GET(id<MTLBinaryArchive>, archive) addComputePipelineFunctionsWithDescriptor:desc error:&aerr]; }
    return RET(ps);
  }
}
uint32_t mvccm_pipeline_max_threads(mvccm_pipeline_t p) { return (uint32_t)GET(id<MTLComputePipelineState>, p).maxTotalThreadsPerThreadgroup; }
uint32_t mvccm_pipeline_exec_width(mvccm_pipeline_t p) { return (uint32_t)GET(id<MTLComputePipelineState>, p).threadExecutionWidth; }
uint32_t mvccm_pipeline_static_tg_mem(mvccm_pipeline_t p) { return (uint32_t)GET(id<MTLComputePipelineState>, p).staticThreadgroupMemoryLength; }
void mvccm_pipeline_release(mvccm_pipeline_t p) { REL(p); }
mvccm_archive_t mvccm_archive_open(mvccm_device_t d, const char* path, char** err_out) {
  @autoreleasepool {
    MTLBinaryArchiveDescriptor* desc = [MTLBinaryArchiveDescriptor new];
    if (path) desc.url = [NSURL fileURLWithPath:[NSString stringWithUTF8String:path]];
    NSError* err = nil;
    id<MTLBinaryArchive> a = [GET(id<MTLDevice>, d) newBinaryArchiveWithDescriptor:desc error:&err];
    if (!a) { set_err(err_out, err, "newBinaryArchive failed"); return nullptr; }
    return RET(a);
  }
}
int mvccm_archive_serialize(mvccm_archive_t a, const char* path, char** err_out) {
  @autoreleasepool {
    NSError* err = nil;
    BOOL ok = [GET(id<MTLBinaryArchive>, a) serializeToURL:[NSURL fileURLWithPath:[NSString stringWithUTF8String:path]] error:&err];
    if (!ok) set_err(err_out, err, "serialize failed");
    return ok ? 1 : 0;
  }
}
void mvccm_archive_release(mvccm_archive_t a) { REL(a); }

// ---- queues / command buffers ----
mvccm_queue_t mvccm_queue_new(mvccm_device_t d, mvccm_residency_t r) {
  @autoreleasepool {
    MTLCommandQueueDescriptor* desc = [MTLCommandQueueDescriptor new];
    desc.maxCommandBufferCount = 512;
    id<MTLCommandQueue> q = [GET(id<MTLDevice>, d) newCommandQueueWithDescriptor:desc];
    if (r) [q addResidencySet:GET(id<MTLResidencySet>, r)];
    return RET(q);
  }
}
void mvccm_queue_release(mvccm_queue_t q) { REL(q); }
mvccm_cmdbuf_t mvccm_cmdbuf_new(mvccm_queue_t q) {
  @autoreleasepool {
    // Unretained references: the runtime's registry keeps every resource alive for as long as it can be in flight.
    MTLCommandBufferDescriptor* desc = [MTLCommandBufferDescriptor new];
    desc.retainedReferences = NO;
    desc.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus;
    return RET([GET(id<MTLCommandQueue>, q) commandBufferWithDescriptor:desc]);
  }
}
void mvccm_cmdbuf_release(mvccm_cmdbuf_t c) { REL(c); }
mvccm_encoder_t mvccm_compute_begin(mvccm_cmdbuf_t c) {
  @autoreleasepool {
    return RET([GET(id<MTLCommandBuffer>, c) computeCommandEncoderWithDispatchType:MTLDispatchTypeConcurrent]);
  }
}
void mvccm_compute_set_pipeline(mvccm_encoder_t e, mvccm_pipeline_t p) { [GET(id<MTLComputeCommandEncoder>, e) setComputePipelineState:GET(id<MTLComputePipelineState>, p)]; }
void mvccm_compute_set_bytes(mvccm_encoder_t e, const void* data, size_t len, uint32_t index) { [GET(id<MTLComputeCommandEncoder>, e) setBytes:data length:len atIndex:index]; }
void mvccm_compute_set_buffer(mvccm_encoder_t e, mvccm_buffer_t b, uint64_t off, uint32_t index) { [GET(id<MTLComputeCommandEncoder>, e) setBuffer:GET(id<MTLBuffer>, b) offset:(NSUInteger)off atIndex:index]; }
void mvccm_compute_use_buffer(mvccm_encoder_t e, mvccm_buffer_t b) {
  [GET(id<MTLComputeCommandEncoder>, e) useResource:GET(id<MTLBuffer>, b) usage:MTLResourceUsageRead | MTLResourceUsageWrite];
}
void mvccm_compute_set_tg_mem(mvccm_encoder_t e, uint32_t len, uint32_t index) { [GET(id<MTLComputeCommandEncoder>, e) setThreadgroupMemoryLength:len atIndex:index]; }
void mvccm_compute_dispatch(mvccm_encoder_t e, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t bx, uint32_t by, uint32_t bz) {
  [GET(id<MTLComputeCommandEncoder>, e) dispatchThreadgroups:MTLSizeMake(gx, gy, gz) threadsPerThreadgroup:MTLSizeMake(bx, by, bz)];
}
void mvccm_compute_dispatch_indirect(mvccm_encoder_t e, mvccm_buffer_t args, uint64_t off, uint32_t bx, uint32_t by, uint32_t bz) {
  [GET(id<MTLComputeCommandEncoder>, e) dispatchThreadgroupsWithIndirectBuffer:GET(id<MTLBuffer>, args) indirectBufferOffset:(NSUInteger)off threadsPerThreadgroup:MTLSizeMake(bx, by, bz)];
}
mvccm_fence_t mvccm_device_new_fence(mvccm_device_t d) { @autoreleasepool { return RET([GET(id<MTLDevice>, d) newFence]); } }
void mvccm_fence_release(mvccm_fence_t f) { REL(f); }
void mvccm_compute_wait_fence(mvccm_encoder_t e, mvccm_fence_t f) { [GET(id<MTLComputeCommandEncoder>, e) waitForFence:GET(id<MTLFence>, f)]; }
void mvccm_compute_update_fence(mvccm_encoder_t e, mvccm_fence_t f) { [GET(id<MTLComputeCommandEncoder>, e) updateFence:GET(id<MTLFence>, f)]; }
void mvccm_blit_wait_fence(mvccm_blit_t e, mvccm_fence_t f) { [GET(id<MTLBlitCommandEncoder>, e) waitForFence:GET(id<MTLFence>, f)]; }
void mvccm_blit_update_fence(mvccm_blit_t e, mvccm_fence_t f) { [GET(id<MTLBlitCommandEncoder>, e) updateFence:GET(id<MTLFence>, f)]; }
void mvccm_compute_barrier(mvccm_encoder_t e) { [GET(id<MTLComputeCommandEncoder>, e) memoryBarrierWithScope:MTLBarrierScopeBuffers]; }
void mvccm_compute_end(mvccm_encoder_t e) { [GET(id<MTLComputeCommandEncoder>, e) endEncoding]; REL(e); }
mvccm_blit_t mvccm_blit_begin(mvccm_cmdbuf_t c) { @autoreleasepool { return RET([GET(id<MTLCommandBuffer>, c) blitCommandEncoder]); } }
void mvccm_blit_copy(mvccm_blit_t e, mvccm_buffer_t src, uint64_t soff, mvccm_buffer_t dst, uint64_t doff, uint64_t len) {
  [GET(id<MTLBlitCommandEncoder>, e) copyFromBuffer:GET(id<MTLBuffer>, src) sourceOffset:(NSUInteger)soff toBuffer:GET(id<MTLBuffer>, dst) destinationOffset:(NSUInteger)doff size:(NSUInteger)len];
}
void mvccm_blit_fill(mvccm_blit_t e, mvccm_buffer_t dst, uint64_t off, uint64_t len, uint8_t value) {
  [GET(id<MTLBlitCommandEncoder>, e) fillBuffer:GET(id<MTLBuffer>, dst) range:NSMakeRange((NSUInteger)off, (NSUInteger)len) value:value];
}
void mvccm_blit_end(mvccm_blit_t e) { [GET(id<MTLBlitCommandEncoder>, e) endEncoding]; REL(e); }
void mvccm_cmdbuf_signal_event(mvccm_cmdbuf_t c, mvccm_event_t ev, uint64_t v) { [GET(id<MTLCommandBuffer>, c) encodeSignalEvent:GET(id<MTLSharedEvent>, ev) value:v]; }
void mvccm_cmdbuf_wait_event(mvccm_cmdbuf_t c, mvccm_event_t ev, uint64_t v) { [GET(id<MTLCommandBuffer>, c) encodeWaitForEvent:GET(id<MTLSharedEvent>, ev) value:v]; }
void mvccm_cmdbuf_on_completed(mvccm_cmdbuf_t c, void (*cb)(void*, int32_t), void* ctx) {
  [GET(id<MTLCommandBuffer>, c) addCompletedHandler:^(id<MTLCommandBuffer> b) { cb(ctx, b.error ? 3 : 2); }];
}
void mvccm_cmdbuf_commit(mvccm_cmdbuf_t c) { [GET(id<MTLCommandBuffer>, c) commit]; }
void mvccm_cmdbuf_wait(mvccm_cmdbuf_t c) { [GET(id<MTLCommandBuffer>, c) waitUntilCompleted]; }
int32_t mvccm_cmdbuf_status(mvccm_cmdbuf_t c) {
  id<MTLCommandBuffer> b = GET(id<MTLCommandBuffer>, c);
  switch (b.status) {
    case MTLCommandBufferStatusNotEnqueued: return 0;
    case MTLCommandBufferStatusEnqueued: case MTLCommandBufferStatusCommitted: case MTLCommandBufferStatusScheduled: return 1;
    case MTLCommandBufferStatusCompleted: return 2;
    default: return 3;
  }
}
char* mvccm_cmdbuf_error(mvccm_cmdbuf_t c) {
  @autoreleasepool {
    id<MTLCommandBuffer> b = GET(id<MTLCommandBuffer>, c);
    if (!b.error) return nullptr;
    NSMutableString* s = [NSMutableString stringWithString:b.error.localizedDescription];
    NSArray* infos = b.error.userInfo[MTLCommandBufferEncoderInfoErrorKey];
    for (id<MTLCommandBufferEncoderInfo> info in infos) {
      [s appendFormat:@"\n  encoder '%@' state=%ld", info.label, (long)info.errorState];
      for (NSString* sig in info.debugSignposts) [s appendFormat:@" [%@]", sig];
    }
    return dupstr(s);
  }
}
double mvccm_cmdbuf_gpu_start(mvccm_cmdbuf_t c) { return GET(id<MTLCommandBuffer>, c).GPUStartTime; }
double mvccm_cmdbuf_gpu_end(mvccm_cmdbuf_t c) { return GET(id<MTLCommandBuffer>, c).GPUEndTime; }

// ---- shared events ----
mvccm_event_t mvccm_event_new(mvccm_device_t d) { @autoreleasepool { return RET([GET(id<MTLDevice>, d) newSharedEvent]); } }
uint64_t mvccm_event_value(mvccm_event_t e) { return GET(id<MTLSharedEvent>, e).signaledValue; }
void mvccm_event_set_value(mvccm_event_t e, uint64_t v) { GET(id<MTLSharedEvent>, e).signaledValue = v; }
int mvccm_event_wait(mvccm_event_t e, uint64_t value, int64_t timeout_ms) {
  @autoreleasepool {
    id<MTLSharedEvent> ev = GET(id<MTLSharedEvent>, e);
    if (ev.signaledValue >= value) return 1;
    if (@available(macOS 15.0, *)) {
      return [ev waitUntilSignaledValue:value timeoutMS:(timeout_ms < 0 ? UINT64_MAX : (uint64_t)timeout_ms)] ? 1 : 0;
    }
    // fallback: listener + semaphore
    dispatch_semaphore_t sem = dispatch_semaphore_create(0);
    MTLSharedEventListener* l = [[MTLSharedEventListener alloc] init];
    [ev notifyListener:l atValue:value block:^(id<MTLSharedEvent>, uint64_t) { dispatch_semaphore_signal(sem); }];
    long r = dispatch_semaphore_wait(sem, timeout_ms < 0 ? DISPATCH_TIME_FOREVER : dispatch_time(DISPATCH_TIME_NOW, timeout_ms * 1000000));
    return r == 0 ? 1 : (ev.signaledValue >= value ? 1 : 0);
  }
}
void mvccm_event_release(mvccm_event_t e) { REL(e); }

double mvccm_now_seconds(void) {
  static mach_timebase_info_data_t tb;
  if (!tb.denom) mach_timebase_info(&tb);
  return (double)mach_absolute_time() * tb.numer / tb.denom * 1e-9;
}
