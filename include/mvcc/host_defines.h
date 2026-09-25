// mvcc: CUDA language attribute macros for clang's CUDA frontend.
// Clean-room; names follow the public CUDA C++ Programming Guide.
#ifndef MVCC_HOST_DEFINES_H
#define MVCC_HOST_DEFINES_H

#if defined(__CUDA__) && defined(__clang__)
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#define __grid_constant__ __attribute__((grid_constant))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __cluster_dims__(...)
#else
// Plain host compiler (e.g. a .cpp file including cuda_runtime.h): attributes vanish.
#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __managed__
#define __grid_constant__
#define __launch_bounds__(...)
#define __cluster_dims__(...)
#endif

#if defined(__CUDA__) && defined(__clang__) && defined(__cplusplus)
// Uninitialized storage for a `__shared__ T x;` whose T has default member initializers: nvcc accepts the
// declaration and leaves the memory uninitialized; clang rejects it, so mvcc rewrites the declaration to
// `__shared__ ::mvcc::shared_storage<T> __mvcc_ss_x; auto& x = __mvcc_ss_x.get();` (README.md).
namespace mvcc {
template <class T> struct shared_storage {
  alignas(T) unsigned char bytes[sizeof(T)];
  __host__ __device__ T& get() { return *reinterpret_cast<T*>(bytes); }
};
}
#endif

#define __forceinline__ inline __attribute__((always_inline))
// __noinline__ is a keyword in clang's CUDA mode; defining it as a macro breaks libc++'s __has_attribute(__noinline__).
#define __align__(n) __attribute__((aligned(n)))
#define __thread__ __thread
#define __import__
#define __export__
#define __cdecl
#define __annotate__(a) __attribute__((a))
#define __location__(a) __annotate__(a)
#define CUDARTAPI
#define CUDART_CB
#define __CUDA_ALIGN__(n) __align__(n)
#define __CUDA_HOSTDEVICE__ __host__ __device__
#define __CUDA_FP16_DECL__ static __device__ __inline__
#define __CUDA_HOSTDEVICE_FP16_DECL__ static __host__ __device__ __inline__

// Projects can test `#ifdef __MVCC__` to adapt to mvcc (README.md, "Differences from CUDA").
#ifndef __MVCC__
#define __MVCC__ 1
#endif
#define MVCC_VERSION_MAJOR 1
#define MVCC_VERSION_MINOR 0
#define MVCC_VERSION_PATCH 1

// Values mirroring what nvcc reports; needed by code that checks them.
#ifndef CUDA_VERSION
#define CUDA_VERSION 12080
#endif
#ifndef CUDART_VERSION
#define CUDART_VERSION 12080
#endif
#ifndef __CUDACC_VER_MAJOR__
#define __CUDACC_VER_MAJOR__ 12
#define __CUDACC_VER_MINOR__ 8
#define __CUDACC_VER_BUILD__ 61
#endif

#endif // MVCC_HOST_DEFINES_H
