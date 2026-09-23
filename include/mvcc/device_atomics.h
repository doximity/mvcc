// mvcc: CUDA device atomics implemented on clang's NVVM atomic builtins, which lower to
// LLVM atomicrmw/cmpxchg. mvcc-ir2msl maps 32-bit forms to Metal atomics and 64-bit integer forms
// to the prelude's spinlocked __mvcc_atomic64 (Apple GPUs have no 64-bit atomic instructions).
#ifndef MVCC_DEVICE_ATOMICS_H
#define MVCC_DEVICE_ATOMICS_H
#if defined(__CUDA__) && defined(__clang__)

#define __MVCC_ATOMIC static __device__ __forceinline__

// add
__MVCC_ATOMIC int atomicAdd(int* p, int v) { return __nvvm_atom_add_gen_i(p, v); }
__MVCC_ATOMIC unsigned int atomicAdd(unsigned int* p, unsigned int v) { return (unsigned int)__nvvm_atom_add_gen_i((int*)p, (int)v); }
__MVCC_ATOMIC unsigned long long atomicAdd(unsigned long long* p, unsigned long long v) { return (unsigned long long)__nvvm_atom_add_gen_ll((long long*)p, (long long)v); }
__MVCC_ATOMIC float atomicAdd(float* p, float v) { return __nvvm_atom_add_gen_f(p, v); }
__MVCC_ATOMIC double atomicAdd(double* p, double v) { return __nvvm_atom_add_gen_d(p, v); }
// sub
__MVCC_ATOMIC int atomicSub(int* p, int v) { return __nvvm_atom_sub_gen_i(p, v); }
__MVCC_ATOMIC unsigned int atomicSub(unsigned int* p, unsigned int v) { return (unsigned int)__nvvm_atom_sub_gen_i((int*)p, (int)v); }
// exchange
__MVCC_ATOMIC int atomicExch(int* p, int v) { return __nvvm_atom_xchg_gen_i(p, v); }
__MVCC_ATOMIC unsigned int atomicExch(unsigned int* p, unsigned int v) { return (unsigned int)__nvvm_atom_xchg_gen_i((int*)p, (int)v); }
__MVCC_ATOMIC unsigned long long atomicExch(unsigned long long* p, unsigned long long v) { return (unsigned long long)__nvvm_atom_xchg_gen_ll((long long*)p, (long long)v); }
__MVCC_ATOMIC float atomicExch(float* p, float v) { int iv; __builtin_memcpy(&iv, &v, 4); int r = __nvvm_atom_xchg_gen_i((int*)p, iv); float f; __builtin_memcpy(&f, &r, 4); return f; }
// min/max
__MVCC_ATOMIC int atomicMin(int* p, int v) { return __nvvm_atom_min_gen_i(p, v); }
__MVCC_ATOMIC unsigned int atomicMin(unsigned int* p, unsigned int v) { return __nvvm_atom_min_gen_ui(p, v); }
__MVCC_ATOMIC long long atomicMin(long long* p, long long v) { return __nvvm_atom_min_gen_ll(p, v); }
__MVCC_ATOMIC unsigned long long atomicMin(unsigned long long* p, unsigned long long v) { return __nvvm_atom_min_gen_ull(p, v); }
__MVCC_ATOMIC int atomicMax(int* p, int v) { return __nvvm_atom_max_gen_i(p, v); }
__MVCC_ATOMIC unsigned int atomicMax(unsigned int* p, unsigned int v) { return __nvvm_atom_max_gen_ui(p, v); }
__MVCC_ATOMIC long long atomicMax(long long* p, long long v) { return __nvvm_atom_max_gen_ll(p, v); }
__MVCC_ATOMIC unsigned long long atomicMax(unsigned long long* p, unsigned long long v) { return __nvvm_atom_max_gen_ull(p, v); }
// inc/dec (wrapping)
__MVCC_ATOMIC unsigned int atomicInc(unsigned int* p, unsigned int v) { return __nvvm_atom_inc_gen_ui(p, v); }
__MVCC_ATOMIC unsigned int atomicDec(unsigned int* p, unsigned int v) { return __nvvm_atom_dec_gen_ui(p, v); }
// compare-and-swap
__MVCC_ATOMIC int atomicCAS(int* p, int cmp, int v) { return __nvvm_atom_cas_gen_i(p, cmp, v); }
__MVCC_ATOMIC unsigned int atomicCAS(unsigned int* p, unsigned int cmp, unsigned int v) { return (unsigned int)__nvvm_atom_cas_gen_i((int*)p, (int)cmp, (int)v); }
__MVCC_ATOMIC unsigned long long atomicCAS(unsigned long long* p, unsigned long long cmp, unsigned long long v) { return (unsigned long long)__nvvm_atom_cas_gen_ll((long long*)p, (long long)cmp, (long long)v); }
__MVCC_ATOMIC unsigned short atomicCAS(unsigned short* p, unsigned short cmp, unsigned short v) {
  unsigned short e = cmp; __atomic_compare_exchange_n(p, &e, v, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED); return e;
}
// bitwise
__MVCC_ATOMIC int atomicAnd(int* p, int v) { return __nvvm_atom_and_gen_i(p, v); }
__MVCC_ATOMIC unsigned int atomicAnd(unsigned int* p, unsigned int v) { return (unsigned int)__nvvm_atom_and_gen_i((int*)p, (int)v); }
__MVCC_ATOMIC unsigned long long atomicAnd(unsigned long long* p, unsigned long long v) { return (unsigned long long)__nvvm_atom_and_gen_ll((long long*)p, (long long)v); }
__MVCC_ATOMIC int atomicOr(int* p, int v) { return __nvvm_atom_or_gen_i(p, v); }
__MVCC_ATOMIC unsigned int atomicOr(unsigned int* p, unsigned int v) { return (unsigned int)__nvvm_atom_or_gen_i((int*)p, (int)v); }
__MVCC_ATOMIC unsigned long long atomicOr(unsigned long long* p, unsigned long long v) { return (unsigned long long)__nvvm_atom_or_gen_ll((long long*)p, (long long)v); }
__MVCC_ATOMIC int atomicXor(int* p, int v) { return __nvvm_atom_xor_gen_i(p, v); }
__MVCC_ATOMIC unsigned int atomicXor(unsigned int* p, unsigned int v) { return (unsigned int)__nvvm_atom_xor_gen_i((int*)p, (int)v); }
__MVCC_ATOMIC unsigned long long atomicXor(unsigned long long* p, unsigned long long v) { return (unsigned long long)__nvvm_atom_xor_gen_ll((long long*)p, (long long)v); }

// _block / _system variants: same operation (a single GPU, one address space).
#define __MVCC_SCOPED(name) \
  template <class T, class U> __MVCC_ATOMIC auto name##_block(T* p, U v) -> decltype(name(p, v)) { return name(p, v); } \
  template <class T, class U> __MVCC_ATOMIC auto name##_system(T* p, U v) -> decltype(name(p, v)) { return name(p, v); }
__MVCC_SCOPED(atomicAdd) __MVCC_SCOPED(atomicSub) __MVCC_SCOPED(atomicExch) __MVCC_SCOPED(atomicMin) __MVCC_SCOPED(atomicMax)
__MVCC_SCOPED(atomicInc) __MVCC_SCOPED(atomicDec) __MVCC_SCOPED(atomicAnd) __MVCC_SCOPED(atomicOr) __MVCC_SCOPED(atomicXor)
#undef __MVCC_SCOPED
template <class T> __MVCC_ATOMIC T atomicCAS_block(T* p, T c, T v) { return atomicCAS(p, c, v); }
template <class T> __MVCC_ATOMIC T atomicCAS_system(T* p, T c, T v) { return atomicCAS(p, c, v); }

#undef __MVCC_ATOMIC
#endif
#endif // MVCC_DEVICE_ATOMICS_H
