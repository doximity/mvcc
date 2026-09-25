// mvcc: device helpers that clang's headers don't provide.
#ifndef MVCC_DEVICE_MISC_H
#define MVCC_DEVICE_MISC_H
#if defined(__CUDA__) && defined(__clang__)

// Generic-to-shared address conversion: the address-space cast is what matters; mvcc-ir2msl
// recovers the threadgroup pointer from it when the value feeds inline PTX.
static __device__ __forceinline__ size_t __cvta_generic_to_shared(const void* p) {
  return (size_t)(__attribute__((address_space(3))) const void*)p;
}
static __device__ __forceinline__ size_t __cvta_generic_to_global(const void* p) {
  return (size_t)(__attribute__((address_space(1))) const void*)p;
}
static __device__ __forceinline__ void* __cvta_shared_to_generic(size_t a) {
  return (void*)(__attribute__((address_space(3))) void*)a;
}
static __device__ __forceinline__ void* __cvta_global_to_generic(size_t a) {
  return (void*)(__attribute__((address_space(1))) void*)a;
}
static __device__ __forceinline__ bool __isShared(const void* p) { return __nvvm_isspacep_shared(p); }
static __device__ __forceinline__ bool __isGlobal(const void* p) { return __nvvm_isspacep_global(p); }
static __device__ __forceinline__ bool __isConstant(const void* p) { return __nvvm_isspacep_const(p); }
static __device__ __forceinline__ bool __isLocal(const void* p) { return __nvvm_isspacep_local(p); }

// Cache-hint loads/stores (__ldcg, __ldcs, __stcg, ...) come from clang's __clang_cuda_intrinsics.h as inline PTX;
// mvcc-ir2msl lowers ld.global.<hint>/st.global.<hint> to plain accesses, except the L2 forms (.cg, .cv, .wt) and
// the volatile / scoped forms, which are coherent accesses (prelude, "coherent device accesses").

// __nanosleep is sm_70+ inline PTX in clang's header when CUDA_VERSION >= 10; we supply a version
// that lowers cleanly (the emitter maps nanosleep to a no-op / yield).
#if !defined(__MVCC_HAVE_NANOSLEEP)
static __device__ __forceinline__ void __nanosleep(unsigned int) {}
#endif

// Byte / half-word dot products (sm_61+ __dp4a / __dp2a). Plain arithmetic: the emitter lowers it as such.
static __device__ __forceinline__ int __dp4a(int a, int b, int c) {
  return c + (int)(signed char)(a) * (int)(signed char)(b) + (int)(signed char)(a >> 8) * (int)(signed char)(b >> 8)
           + (int)(signed char)(a >> 16) * (int)(signed char)(b >> 16) + (int)(signed char)(a >> 24) * (int)(signed char)(b >> 24);
}
static __device__ __forceinline__ unsigned int __dp4a(unsigned int a, unsigned int b, unsigned int c) {
  return c + (a & 0xffu) * (b & 0xffu) + ((a >> 8) & 0xffu) * ((b >> 8) & 0xffu) + ((a >> 16) & 0xffu) * ((b >> 16) & 0xffu) + (a >> 24) * (b >> 24);
}
static __device__ __forceinline__ int __dp2a_lo(int a, int b, int c) {
  return c + (int)(short)(a) * (int)(signed char)(b) + (int)(short)(a >> 16) * (int)(signed char)(b >> 8);
}
static __device__ __forceinline__ int __dp2a_hi(int a, int b, int c) {
  return c + (int)(short)(a) * (int)(signed char)(b >> 16) + (int)(short)(a >> 16) * (int)(signed char)(b >> 24);
}

// assert(): device-side assert is a no-op in release builds under nvcc too (NDEBUG). We follow NDEBUG.
#ifndef NDEBUG
#define __MVCC_DEVICE_ASSERT(x) ((void)0)
#endif

// Device printf: nvcc's is host+device. clang's <stdio.h> is host-only, so a kernel or a
// __host__ __device__ function that calls printf fails ("call to __host__ function from __global__").
// Visible on both passes (the host pass still type-checks __global__ bodies). Metal has no CUDA
// printf buffer; the device overload compiles and returns 0.
static __device__ inline int printf(const char *, ...) { return 0; }

#endif
#endif // MVCC_DEVICE_MISC_H
