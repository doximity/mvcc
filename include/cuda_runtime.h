// mvcc: the header every CUDA translation unit gets. Orchestrates:
//   - attribute macros and API types (ours, clean-room)
//   - clang's own CUDA device-side support headers (Apache-2.0 w/ LLVM exception),
//     which target libdevice *names* (__nv_*) that mvcc-ir2msl implements in MSL
//   - device atomics, half/bf16/fp8 types (ours)
// Works in three modes: clang CUDA host pass, clang CUDA device pass, and plain C++ host code.
#ifndef MVCC_CUDA_RUNTIME_H
#define MVCC_CUDA_RUNTIME_H

#include <mvcc/host_defines.h>
#include <driver_types.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <cuda_runtime_api.h>

#if defined(__CUDA__) && defined(__clang__)
// ---------------------------------------------------------------------------
// clang CUDA mode (host or device pass)
// ---------------------------------------------------------------------------
#define __CUDACC__ 1

// Forward-declare device math overloads before the C++ standard library sees <cmath>.
#include <__clang_cuda_math_forward_declares.h>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <stdlib.h>
#include <string.h>

// Built-in variables (threadIdx, blockIdx, blockDim, gridDim, warpSize) via NVVM special registers.
#include <__clang_cuda_builtin_vars.h>

// clang's device headers key feature availability off __CUDA_ARCH__. During the host pass it is
// undefined; define it temporarily to the device pass's arch (the driver passes it as __MVCC_CUDA_ARCH__)
// so the same device functions get declared (they are never emitted for host).
#pragma push_macro("__CUDA_ARCH__")
#ifndef __CUDA_ARCH__
#ifdef __MVCC_CUDA_ARCH__
#define __CUDA_ARCH__ __MVCC_CUDA_ARCH__
#else
#define __CUDA_ARCH__ 890
#endif
#endif

// The sm_90 block of clang's intrinsics header (__isCtaShared, __cluster_map_shared_multicast) calls two
// functions NVIDIA's headers declare ahead of it; mvcc defines them in <mvcc/device_misc.h>, included below.
static __device__ __forceinline__ bool __isShared(const void* p);
static __device__ __forceinline__ size_t __cvta_generic_to_shared(const void* p);

#include <__clang_cuda_libdevice_declares.h>
#include <__clang_cuda_device_functions.h>
#include <__clang_cuda_math.h>
#include <__clang_cuda_intrinsics.h>

#pragma pop_macro("__CUDA_ARCH__")

// <cmath>-style overloads for device code (std::exp(float) etc.)
#include <__clang_cuda_cmath.h>

#include <mvcc/device_atomics.h>
#include <mvcc/device_misc.h>
#include <cuda_device_runtime_api.h>

#else
// ---------------------------------------------------------------------------
// Plain host compiler
// ---------------------------------------------------------------------------
#include <math.h>
#include <string.h>
#endif

#if defined(__cplusplus)
// ---------------------------------------------------------------------------
// C++ conveniences from the CUDA runtime header (templated overloads)
// ---------------------------------------------------------------------------
template <class T>
static inline cudaError_t cudaMalloc(T** devPtr, size_t size) { return ::cudaMalloc((void**)(void*)devPtr, size); }
template <class T>
static inline cudaError_t cudaMallocHost(T** ptr, size_t size, unsigned int flags = 0) { (void)flags; return ::cudaMallocHost((void**)(void*)ptr, size); }
template <class T>
static inline cudaError_t cudaHostAlloc(T** ptr, size_t size, unsigned int flags) { return ::cudaHostAlloc((void**)(void*)ptr, size, flags); }
template <class T>
static inline cudaError_t cudaMallocManaged(T** devPtr, size_t size, unsigned int flags = cudaMemAttachGlobal) { return ::cudaMallocManaged((void**)(void*)devPtr, size, flags); }
template <class T>
static inline cudaError_t cudaMallocAsync(T** devPtr, size_t size, cudaStream_t stream) { return ::cudaMallocAsync((void**)(void*)devPtr, size, stream); }
template <class T>
static inline cudaError_t cudaHostGetDevicePointer(T** pDevice, void* pHost, unsigned int flags) { return ::cudaHostGetDevicePointer((void**)(void*)pDevice, pHost, flags); }
template <class T>
static inline cudaError_t cudaFuncSetAttribute(T* entry, enum cudaFuncAttribute attr, int value) { return ::cudaFuncSetAttribute((const void*)entry, attr, value); }
template <class T>
static inline cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, T* entry) { return ::cudaFuncGetAttributes(attr, (const void*)entry); }
template <class T>
static inline cudaError_t cudaFuncSetCacheConfig(T* func, enum cudaFuncCache cacheConfig) { return ::cudaFuncSetCacheConfig((const void*)func, cacheConfig); }
template <class T>
static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, T func, int blockSize, size_t dynamicSMemSize) {
  return ::cudaOccupancyMaxActiveBlocksPerMultiprocessor(numBlocks, (const void*)func, blockSize, dynamicSMemSize);
}
template <class T>
static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, T func, int blockSize, size_t dynamicSMemSize, unsigned int flags) {
  return ::cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(numBlocks, (const void*)func, blockSize, dynamicSMemSize, flags);
}
template <class T>
static inline cudaError_t cudaGetSymbolAddress(void** devPtr, const T& symbol) { return ::cudaGetSymbolAddress(devPtr, (const void*)&symbol); }
template <class T>
static inline cudaError_t cudaGetSymbolSize(size_t* size, const T& symbol) { return ::cudaGetSymbolSize(size, (const void*)&symbol); }
template <class T>
static inline cudaError_t cudaMemcpyToSymbol(const T& symbol, const void* src, size_t count, size_t offset = 0, enum cudaMemcpyKind kind = cudaMemcpyHostToDevice) {
  return ::cudaMemcpyToSymbol((const void*)&symbol, src, count, offset, kind);
}
template <class T>
static inline cudaError_t cudaMemcpyFromSymbol(void* dst, const T& symbol, size_t count, size_t offset = 0, enum cudaMemcpyKind kind = cudaMemcpyDeviceToHost) {
  return ::cudaMemcpyFromSymbol(dst, (const void*)&symbol, count, offset, kind);
}
template <class T>
static inline cudaError_t cudaMemcpyToSymbolAsync(const T& symbol, const void* src, size_t count, size_t offset = 0, enum cudaMemcpyKind kind = cudaMemcpyHostToDevice, cudaStream_t stream = 0) {
  return ::cudaMemcpyToSymbolAsync((const void*)&symbol, src, count, offset, kind, stream);
}
template <class T>
static inline cudaError_t cudaLaunchKernel(const T* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem = 0, cudaStream_t stream = 0) {
  return ::cudaLaunchKernel((const void*)func, gridDim, blockDim, args, sharedMem, stream);
}
template <typename... ExpTypes, typename... ActTypes>
static inline cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t* config, void (*kernel)(ExpTypes...), ActTypes&&... args) {
  void* pArgs[] = {(void*)&args..., nullptr};
  return ::cudaLaunchKernelExC(config, (const void*)kernel, pArgs);
}
template <class T>
static inline cudaError_t cudaStreamAttachMemAsync(cudaStream_t, T*, size_t = 0, unsigned int = cudaMemAttachSingle) { return cudaSuccess; }
static inline cudaError_t cudaEventCreate(cudaEvent_t* event, unsigned int flags) { return ::cudaEventCreateWithFlags(event, flags); }
static inline cudaError_t cudaMallocHost(void** ptr, size_t size, unsigned int flags) { (void)flags; return ::cudaMallocHost(ptr, size); }
#endif // __cplusplus

#endif // MVCC_CUDA_RUNTIME_H
