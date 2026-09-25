// mvcc MSL prelude. Spliced by mvcc between MVCC_PRELUDE_BEGIN / MVCC_PRELUDE_END in generated sources.
// Every helper is guarded by an MVCC_NEED_<group> macro that mvcc-ir2msl defines when the kernel uses it, so unused
// helpers cost nothing (Metal compiles from source at runtime; smaller source = faster pipeline creation).
//
// Layout facts used below are measured, not assumed: see README.md.

// ---------------------------------------------------------------------------------------------------------------
// Where a kernel's shared memory lives. Apple GPUs give a threadgroup 32 KB; CUDA kernels routinely stage 40-200 KB
// (opt-in dynamic shared memory). The generated source spells every shared-memory pointer with __MVCC_SMEM, so the
// same text compiles two ways: the normal build (threadgroup memory, [[threadgroup(0)]]) and the spill build
// (-D__MVCC_SPILL: the block's shared memory is a slot of a device-memory pool, [[buffer(5)]]). The runtime compiles
// the spill build of a module lazily, the first time a launch needs more shared memory than the device has, and
// uses it only for those launches (README.md, "Differences from CUDA").
//
// Pool layout: header (slot count, slot stride, data offset), one uint counter per slot at byte 4096 (0 = free,
// else the number of threads of the block holding it), then the slots. A block takes a slot at entry (thread 0
// probes from a hash of its block index; a block that finds none free spins - blocks holding slots are resident
// and finish, so it terminates) and every thread releases its share on exit (__MVCC_RETURN on every return path),
// the last one freeing the slot: a kernel whose threads exit early never needs a barrier at the end.
// ---------------------------------------------------------------------------------------------------------------
#ifdef __MVCC_SPILL
#define __MVCC_SMEM device
#define __MVCC_SMEM_FLAGS (mem_flags::mem_threadgroup | mem_flags::mem_device)
struct __mvcc_spill_hdr { uint slots; uint stride; uint data_off; uint pad; };
inline uint __mvcc_spill_acquire(device uchar* pool, uint start, uint nthreads) {
  const uint P = ((device __mvcc_spill_hdr*)pool)->slots;
  device atomic_uint* c = (device atomic_uint*)(pool + 4096);
  uint i = start % P;
  while (true) {
    uint e = 0u;
    if (atomic_compare_exchange_weak_explicit(c + i, &e, nthreads, memory_order_relaxed, memory_order_relaxed)) {
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      return i;
    }
    i = (i + 1u == P) ? 0u : i + 1u;
  }
}
inline void __mvcc_spill_release(device uchar* pool, uint slot) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  atomic_fetch_sub_explicit((device atomic_uint*)(pool + 4096) + slot, 1u, memory_order_relaxed);
}
#define __MVCC_RETURN { __mvcc_spill_release(__mvcc_spill, __mvcc_slot); return; }
// A PTX shared-memory address is 32 bits (cvta.to.shared); CUDA code does arithmetic on it and hands the result
// to cp.async/ldmatrix/st.shared. In threadgroup memory the truncation is harmless; in a device-memory slot the
// address is rebuilt from the slot's high half (the runtime keeps every slot inside one 4 GiB window).
#define __MVCC_SMEM_ADDR(x) (__mvcc_smem_hi | (ulong)(uint)(ulong)(x))
#else
#define __MVCC_SMEM threadgroup
#define __MVCC_SMEM_FLAGS mem_flags::mem_threadgroup
#define __MVCC_RETURN return;
#define __MVCC_SMEM_ADDR(x) ((ulong)(x))
#endif

// ---------------------------------------------------------------------------------------------------------------
// Warp shuffles with exact PTX shfl.sync semantics (clamp/segmask packed in c: c[4:0]=clamp, c[12:8]=segmask).
// CUDA's __shfl_*_sync(mask, v, l, width) passes c = ((32-width)<<8) | 0x1f (idx/bfly/down) or | 0 (up).
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_shfl
namespace mvcc_detail {
inline uint shfl_pick(uint lane, uint c, uint j, bool pval) { return pval ? j : lane; }
}
inline uint __mvcc_shfl_idx(uint v, uint b, uint c, uint lane) {
  uint segmask = (c >> 8) & 0x1f, cval = c & 0x1f;
  uint maxLane = (lane & segmask) | (cval & ~segmask), minLane = lane & segmask;
  uint j = minLane | (b & 0x1f & ~segmask);
  return simd_shuffle(v, (ushort)mvcc_detail::shfl_pick(lane, c, j, j <= maxLane));
}
inline uint __mvcc_shfl_bfly(uint v, uint b, uint c, uint lane) {
  if (c == 0x1f) return simd_shuffle_xor(v, (ushort)(b & 0x1f));   // full-warp fast path (folds when c is a literal)
  uint segmask = (c >> 8) & 0x1f, cval = c & 0x1f;
  uint maxLane = (lane & segmask) | (cval & ~segmask);
  uint j = lane ^ (b & 0x1f);
  return simd_shuffle(v, (ushort)mvcc_detail::shfl_pick(lane, c, j, j <= maxLane));
}
inline uint __mvcc_shfl_up(uint v, uint b, uint c, uint lane) {
  uint segmask = (c >> 8) & 0x1f, cval = c & 0x1f;
  uint maxLane = (lane & segmask) | (cval & ~segmask);
  int j = (int)lane - (int)(b & 0x1f);
  bool pval = j >= (int)maxLane;
  return simd_shuffle(v, (ushort)(pval ? (uint)j : lane));
}
inline uint __mvcc_shfl_down(uint v, uint b, uint c, uint lane) {
  uint segmask = (c >> 8) & 0x1f, cval = c & 0x1f;
  uint maxLane = (lane & segmask) | (cval & ~segmask);
  uint j = lane + (b & 0x1f);
  return simd_shuffle(v, (ushort)mvcc_detail::shfl_pick(lane, c, j, j <= maxLane));
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// Under-aligned scalar loads/stores (LLVM emitted align < size). Assembled bytewise; rare in practice.
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_unaligned
#define __MVCC_UNALIGNED_FOR(AS)                                                                     \
  template <typename T> inline T __mvcc_load_unaligned(AS uchar* p) {                                \
    T r; thread uchar* d = (thread uchar*)&r;                                                      \
    for (uint i = 0; i < sizeof(T); i++) d[i] = p[i];                                              \
    return r;                                                                                      \
  }                                                                                                \
  template <typename T> inline void __mvcc_store_unaligned(AS uchar* p, T v) {                       \
    thread uchar* s = (thread uchar*)&v;                                                           \
    for (uint i = 0; i < sizeof(T); i++) p[i] = s[i];                                              \
  }
__MVCC_UNALIGNED_FOR(device) __MVCC_UNALIGNED_FOR(threadgroup) __MVCC_UNALIGNED_FOR(thread)
template <typename T> inline T __mvcc_load_unaligned(constant uchar* p) {
  T r; thread uchar* d = (thread uchar*)&r;
  for (uint i = 0; i < sizeof(T); i++) d[i] = p[i];
  return r;
}
#undef __MVCC_UNALIGNED_FOR
#endif

// ---------------------------------------------------------------------------------------------------------------
// Coherent device accesses: CUDA volatile / ld.volatile / .relaxed / .acquire / .cg / .cv loads and volatile /
// .release / .cg / .wt stores of device memory are performed at L2 (they see and publish other SMs' stores; a spin
// on a flag re-reads memory). An Apple GPU's plain device load is served from a core's L1 for the kernel's life,
// so a `device volatile` access alone never sees another core's store. A 4-byte access is a relaxed device-scope
// atomic (performed at the coherence point); any other width is a volatile access next to a device-scope fence.
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_coherent
template <typename T> inline T __mvcc_ld_coherent(device T* p) {
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  return *(device volatile T*)p;
}
inline uint __mvcc_ld_coherent(device uint* p) { return atomic_load_explicit((device atomic_uint*)p, memory_order_relaxed); }
inline int __mvcc_ld_coherent(device int* p) { return atomic_load_explicit((device atomic_int*)p, memory_order_relaxed); }
inline float __mvcc_ld_coherent(device float* p) { return as_type<float>(atomic_load_explicit((device atomic_uint*)p, memory_order_relaxed)); }
// vectors of 4-byte lanes: one relaxed atomic per lane (PTX promises no atomicity across the lanes of a vector access)
#define __MVCC_LD_COHERENT_VEC(T, N)                                                                 \
  inline T##N __mvcc_ld_coherent(device T##N* p) {                                                   \
    T##N r;                                                                                          \
    for (uint i = 0; i < N; i++) r[i] = __mvcc_ld_coherent((device T*)p + i);                        \
    return r;                                                                                        \
  }                                                                                                  \
  inline void __mvcc_st_coherent(device T##N* p, T##N v) {                                           \
    for (uint i = 0; i < N; i++) __mvcc_st_coherent((device T*)p + i, v[i]);                         \
  }
template <typename T> inline void __mvcc_st_coherent(device T* p, T v) {
  *(device volatile T*)p = v;
  atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
}
inline void __mvcc_st_coherent(device uint* p, uint v) { atomic_store_explicit((device atomic_uint*)p, v, memory_order_relaxed); }
inline void __mvcc_st_coherent(device int* p, int v) { atomic_store_explicit((device atomic_int*)p, v, memory_order_relaxed); }
inline void __mvcc_st_coherent(device float* p, float v) { atomic_store_explicit((device atomic_uint*)p, as_type<uint>(v), memory_order_relaxed); }
__MVCC_LD_COHERENT_VEC(uint, 2) __MVCC_LD_COHERENT_VEC(uint, 4) __MVCC_LD_COHERENT_VEC(int, 2) __MVCC_LD_COHERENT_VEC(int, 4)
__MVCC_LD_COHERENT_VEC(float, 2) __MVCC_LD_COHERENT_VEC(float, 4)
#undef __MVCC_LD_COHERENT_VEC
#endif

// ---------------------------------------------------------------------------------------------------------------
// Float atomic max/min without a CAS loop: for v >= 0 signed-int max on the bit pattern orders like float; for
// v < 0 unsigned min does. Both halves are single hardware atomics. Returns the old value like atomicMax.
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_atomic_fmaxmin
#define __MVCC_ATOMIC_FMM(AS)                                                                        \
  inline float __mvcc_atomic_fmax(AS atomic_float* p, float v) {                                     \
    int old = (v >= 0.f) ? atomic_fetch_max_explicit((AS atomic_int*)p, as_type<int>(v), memory_order_relaxed) \
                         : (int)atomic_fetch_min_explicit((AS atomic_uint*)p, as_type<uint>(v), memory_order_relaxed); \
    return as_type<float>(old);                                                                    \
  }                                                                                                \
  inline float __mvcc_atomic_fmin(AS atomic_float* p, float v) {                                     \
    int old = (v >= 0.f) ? atomic_fetch_min_explicit((AS atomic_int*)p, as_type<int>(v), memory_order_relaxed) \
                         : (int)atomic_fetch_max_explicit((AS atomic_uint*)p, as_type<uint>(v), memory_order_relaxed); \
    return as_type<float>(old);                                                                    \
  }
__MVCC_ATOMIC_FMM(device) __MVCC_ATOMIC_FMM(threadgroup)
#undef __MVCC_ATOMIC_FMM
#endif

// ---------------------------------------------------------------------------------------------------------------
// 64-bit atomics. Apple GPUs have no 64-bit atomic RMW/CAS, so each operation takes one of 16K runtime-provided
// spinlocks (hashed by address, [[buffer(4)]]) around a plain aligned 8-byte read-modify-write. Acquire and
// release happen in the same loop iteration, so lockstep SIMD lanes contending for one lock cannot starve the
// holder. Concurrent plain 64-bit loads observe either the old or the new value (aligned 8-byte accesses are
// single-copy atomic on Apple GPUs); see README.md, "Differences from CUDA".
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_atomic64
enum { MVCC_A64_ADD, MVCC_A64_SUB, MVCC_A64_AND, MVCC_A64_OR, MVCC_A64_XOR, MVCC_A64_XCHG, MVCC_A64_MAX, MVCC_A64_MIN,
       MVCC_A64_UMAX, MVCC_A64_UMIN, MVCC_A64_CAS };
#define __MVCC_A64(AS)                                                                                            \
  inline ulong __mvcc_atomic64(device atomic_uint* locks, AS ulong* p, int op, ulong v, ulong cmp, thread bool* ok) { \
    device atomic_uint* lk = locks + (uint)(((ulong)p >> 3) & 16383u);                                           \
    ulong old = 0; bool done = false;                                                                            \
    while (!done) {                                                                                              \
      uint e = 0;                                                                                                \
      if (atomic_compare_exchange_weak_explicit(lk, &e, 1u, memory_order_relaxed, memory_order_relaxed)) {       \
        atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_device); \
        old = *p; ulong nv = old;                                                                                \
        switch (op) {                                                                                            \
          case MVCC_A64_ADD: nv = old + v; break;                                                                 \
          case MVCC_A64_SUB: nv = old - v; break;                                                                 \
          case MVCC_A64_AND: nv = old & v; break;                                                                 \
          case MVCC_A64_OR: nv = old | v; break;                                                                  \
          case MVCC_A64_XOR: nv = old ^ v; break;                                                                 \
          case MVCC_A64_XCHG: nv = v; break;                                                                      \
          case MVCC_A64_MAX: nv = (ulong)max((long)old, (long)v); break;                                          \
          case MVCC_A64_MIN: nv = (ulong)min((long)old, (long)v); break;                                          \
          case MVCC_A64_UMAX: nv = max(old, v); break;                                                            \
          case MVCC_A64_UMIN: nv = min(old, v); break;                                                            \
          default: *ok = (old == cmp); if (*ok) nv = v; break;                                                   \
        }                                                                                                        \
        if (nv != old) *p = nv;                                                                                  \
        atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_device); \
        atomic_store_explicit(lk, 0u, memory_order_relaxed);                                                     \
        done = true;                                                                                             \
      }                                                                                                          \
    }                                                                                                            \
    return old;                                                                                                  \
  }
__MVCC_A64(device) __MVCC_A64(threadgroup)
#undef __MVCC_A64
#endif

// ---------------------------------------------------------------------------------------------------------------
// Tagged pointers. A CUDA generic pointer that provably refers to thread (private) memory on some paths and to
// device/threadgroup/constant memory on others has no single MSL address space. The emitter carries such values
// as a ulong: the address in the low 62 bits and the space in the top two (0 device, 1 thread, 2 threadgroup,
// 3 constant). Every access through one branches on the tag.
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_tagged
#define __MVCC_TAG_MASK 0x3ffffffffffffffful
template <typename P> inline ulong __mvcc_tag(P p, uint space) { return ((ulong)p & __MVCC_TAG_MASK) | ((ulong)space << 62); }
inline ulong __mvcc_untag(ulong t) { return t & __MVCC_TAG_MASK; }
template <typename T> inline T __mvcc_ld_tagged(ulong t, uint align) {
  ulong a = t & __MVCC_TAG_MASK;
  switch (t >> 62) {
    case 1u: return *(thread T*)a;
    case 2u: return *(__MVCC_SMEM T*)a;
    case 3u: return *(constant T*)a;
    default: return *(device T*)a;
  }
}
template <typename T> inline void __mvcc_st_tagged(ulong t, T v, uint align) {
  ulong a = t & __MVCC_TAG_MASK;
  switch (t >> 62) {
    case 1u: *(thread T*)a = v; break;
    case 2u: *(__MVCC_SMEM T*)a = v; break;
    default: *(device T*)a = v; break;
  }
}
inline void __mvcc_memcpy_tagged(ulong d, ulong s, ulong n) { for (ulong i = 0; i < n; i++) __mvcc_st_tagged<uchar>(d + i, __mvcc_ld_tagged<uchar>(s + i, 1u), 1u); }
inline void __mvcc_memset_tagged(ulong d, uchar v, ulong n) { for (ulong i = 0; i < n; i++) __mvcc_st_tagged<uchar>(d + i, v, 1u); }
#endif

// ---------------------------------------------------------------------------------------------------------------
// Bit helpers
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_bits
// PTX fns: position of the offset-th set bit of mask counting from base (offset > 0: upward, base inclusive;
// offset < 0: downward; offset == 0: base itself if set); 0xffffffff when there is no such bit.
inline uint __mvcc_fns(uint mask, uint base, int offset) {
  if (base > 31) return 0xffffffffu;
  if (offset == 0) return ((mask >> base) & 1u) ? base : 0xffffffffu;
  if (offset > 0) { for (int i = (int)base; i < 32; i++) if ((mask >> i) & 1u) { if (--offset == 0) return (uint)i; } return 0xffffffffu; }
  for (int i = (int)base; i >= 0; i--) if ((mask >> i) & 1u) { if (++offset == 0) return (uint)i; }
  return 0xffffffffu;
}
inline ushort __mvcc_bswap(ushort x) { return (ushort)((x >> 8) | (x << 8)); }
inline uint __mvcc_bswap(uint x) { return (x >> 24) | ((x >> 8) & 0xff00u) | ((x << 8) & 0xff0000u) | (x << 24); }
inline ulong __mvcc_bswap(ulong x) { return ((ulong)__mvcc_bswap((uint)x) << 32) | (ulong)__mvcc_bswap((uint)(x >> 32)); }
inline uint __mvcc_fshl(uint a, uint b, uint s) { s &= 31; return s ? (a << s) | (b >> (32 - s)) : a; }
inline uint __mvcc_fshr(uint a, uint b, uint s) { s &= 31; return s ? (a << (32 - s)) | (b >> s) : b; }
inline ulong __mvcc_fshl(ulong a, ulong b, ulong s) { s &= 63; return s ? (a << s) | (b >> (64 - s)) : a; }
inline ulong __mvcc_fshr(ulong a, ulong b, ulong s) { s &= 63; return s ? (a << (64 - s)) | (b >> s) : b; }
inline ushort __mvcc_fshl(ushort a, ushort b, ushort s) { s &= 15; return s ? (ushort)((a << s) | (b >> (16 - s))) : a; }
inline ushort __mvcc_fshr(ushort a, ushort b, ushort s) { s &= 15; return s ? (ushort)((a << (16 - s)) | (b >> s)) : b; }
#endif

// log1p / expm1: not in <metal_math>; the classic compensated forms keep full precision near 0.
#if MVCC_NEED_log1p
inline float __mvcc_log1p(float x) {
  float u = 1.0f + x;
  if (u == 1.0f) return x;
  if (isinf(u)) return u;
  return precise::log(u) * (x / (u - 1.0f));
}
inline float __mvcc_expm1(float x) {
  float u = precise::exp(x);
  if (u == 1.0f) return x;
  if (u - 1.0f == -1.0f) return -1.0f;
  if (isinf(u)) return u;
  return (u - 1.0f) * (x / precise::log(u));
}
#endif

// erff / erfcf: Metal has no erf. Rational approximations from FreeBSD msun s_erff.c, float coefficients as in the
// FreeBSD / openlibm version; ~1 ulp. s_erff.c carries this notice:
//
//   Copyright (C) 1993 by Sun Microsystems, Inc. All rights reserved.
//
//   Developed at SunPro, a Sun Microsystems, Inc. business.
//   Permission to use, copy, modify, and distribute this
//   software is freely granted, provided that this notice
//   is preserved.
#if MVCC_NEED_erf
namespace mvcc_erf {
constant constexpr float efx = 1.2837916613e-01f, efx8 = 1.0270333290e+00f;
constant constexpr float pp0 = 1.28379166e-01f, pp1 = -3.36030394e-01f, pp2 = -1.86260219e-03f;
constant constexpr float qq1 = 3.12324286e-01f, qq2 = 2.16070302e-02f, qq3 = -1.98859419e-03f;
constant constexpr float erx = 8.42697144e-01f;
constant constexpr float pa0 = 3.64939137e-06f, pa1 = 4.15109694e-01f, pa2 = -1.65179938e-01f, pa3 = 1.10914491e-01f;
constant constexpr float qa1 = 6.02074385e-01f, qa2 = 5.35934687e-01f, qa3 = 1.68576106e-01f, qa4 = 5.62181212e-02f;
constant constexpr float ra0 = -9.87132732e-03f, ra1 = -5.53605914e-01f, ra2 = -2.17589188e+00f, ra3 = -1.43268085e+00f;
constant constexpr float sa1 = 5.45995426e+00f, sa2 = 6.69798088e+00f, sa3 = 1.43113089e+00f, sa4 = -5.77397496e-02f;
constant constexpr float rb0 = -9.86494310e-03f, rb1 = -6.25171244e-01f, rb2 = -6.16498327e+00f, rb3 = -1.66696873e+01f, rb4 = -9.53764343e+00f;
constant constexpr float sb1 = 1.26884899e+01f, sb2 = 4.51839523e+01f, sb3 = 4.72810211e+01f, sb4 = 8.93033314e+00f;
// exp(-x^2) * exp(R/S) tail shared by erf and erfc for |x| >= 1.25
inline float tail(float ax, uint ix) {
  const float s = 1.0f / (ax * ax);
  float R, S;
  if (ix < 0x4036DB6Eu) { R = ra0 + s * (ra1 + s * (ra2 + s * ra3)); S = 1.0f + s * (sa1 + s * (sa2 + s * (sa3 + s * sa4))); }
  else { R = rb0 + s * (rb1 + s * (rb2 + s * (rb3 + s * rb4))); S = 1.0f + s * (sb1 + s * (sb2 + s * (sb3 + s * sb4))); }
  const float z = as_type<float>(ix & 0xffffe000u);
  return precise::exp(-z * z - 0.5625f) * precise::exp((z - ax) * (z + ax) + R / S);
}
}  // namespace mvcc_erf
inline float __mvcc_erf(float x) {
  using namespace mvcc_erf;
  const uint hx = as_type<uint>(x), ix = hx & 0x7fffffffu;
  if (ix >= 0x7f800000u) return (float)(1 - (int)((hx >> 31) << 1)) + 1.0f / x;  // nan -> nan, +-inf -> +-1
  if (ix < 0x3f580000u) {                                                          // |x| < 0.84375
    if (ix < 0x38800000u) return ix < 0x04000000u ? (8.0f * x + efx8 * x) / 8.0f : x + efx * x;
    const float z = x * x;
    const float r = pp0 + z * (pp1 + z * pp2), s = 1.0f + z * (qq1 + z * (qq2 + z * qq3));
    return x + x * (r / s);
  }
  if (ix < 0x3fa00000u) {  // 0.84375 <= |x| < 1.25
    const float s = fabs(x) - 1.0f;
    const float P = pa0 + s * (pa1 + s * (pa2 + s * pa3)), Q = 1.0f + s * (qa1 + s * (qa2 + s * (qa3 + s * qa4)));
    return (int)hx >= 0 ? erx + P / Q : -erx - P / Q;
  }
  if (ix >= 0x40800000u) return (int)hx >= 0 ? 1.0f - 1e-30f : 1e-30f - 1.0f;  // |x| >= 4
  const float ax = fabs(x), r = tail(ax, ix);
  return (int)hx >= 0 ? 1.0f - r / ax : r / ax - 1.0f;
}
inline float __mvcc_erfc(float x) {
  using namespace mvcc_erf;
  const uint hx = as_type<uint>(x), ix = hx & 0x7fffffffu;
  if (ix >= 0x7f800000u) return (float)((hx >> 31) << 1) + 1.0f / x;  // +-inf -> 0, 2
  if (ix < 0x3f580000u) {
    if (ix < 0x33800000u) return 1.0f - x;
    const float z = x * x;
    const float r = pp0 + z * (pp1 + z * pp2), s = 1.0f + z * (qq1 + z * (qq2 + z * qq3));
    const float y = r / s;
    if ((int)hx < 0x3e800000) return 1.0f - (x + x * y);
    return 0.5f - (x * y + (x - 0.5f));
  }
  if (ix < 0x3fa00000u) {
    const float s = fabs(x) - 1.0f;
    const float P = pa0 + s * (pa1 + s * (pa2 + s * pa3)), Q = 1.0f + s * (qa1 + s * (qa2 + s * (qa3 + s * qa4)));
    return (int)hx >= 0 ? (1.0f - erx) - P / Q : 1.0f + (erx + P / Q);
  }
  if (ix < 0x41300000u) {  // |x| < 28
    if ((int)hx < 0 && ix >= 0x40a00000u) return 2.0f - 1e-30f;
    const float ax = fabs(x), r = tail(ax, ix);
    return (int)hx > 0 ? r / ax : 2.0f - r / ax;
  }
  return (int)hx > 0 ? 0.0f : 2.0f - 1e-30f;
}
#endif

// prmt / __byte_perm (default mode): pick 4 bytes out of {b:a} by selector nibbles; nibble bit 3 sign-replicates.
#if MVCC_NEED_prmt
inline uint __mvcc_prmt(uint a, uint b, uint sel) {
  uint r = 0;
  for (uint i = 0; i < 4; i++) {
    uint s = (sel >> (4 * i)) & 0xf, idx = s & 7;
    uint byte = idx < 4 ? (a >> (8 * idx)) & 0xff : (b >> (8 * (idx - 4))) & 0xff;
    if (s & 8) byte = (byte & 0x80) ? 0xff : 0;
    r |= byte << (8 * i);
  }
  return r;
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// memcpy/memmove/memset across address spaces (LLVM emits these for struct copies and array init). Word-copy when
// both sides are 4-aligned, else bytewise.
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_mem
#define __MVCC_MEMCPY(DAS, SAS)                                                                      \
  inline void __mvcc_memcpy(DAS uchar* d, SAS uchar* s, ulong n) {                                   \
    ulong i = 0;                                                                                   \
    if ((((ulong)d | (ulong)s) & 3) == 0) {                                                        \
      for (; i + 16 <= n; i += 16) *(DAS uint4*)(d + i) = *(SAS uint4*)(s + i);                    \
      for (; i + 4 <= n; i += 4) *(DAS uint*)(d + i) = *(SAS uint*)(s + i);                        \
    }                                                                                              \
    for (; i < n; i++) d[i] = s[i];                                                                \
  }                                                                                                \
  inline void __mvcc_memmove(DAS uchar* d, SAS uchar* s, ulong n) {                                  \
    if ((ulong)d < (ulong)s || (ulong)d >= (ulong)s + n) { for (ulong i = 0; i < n; i++) d[i] = s[i]; } \
    else { for (ulong i = n; i-- > 0;) d[i] = s[i]; }                                              \
  }
#define __MVCC_MEMSET(DAS)                                                                           \
  inline void __mvcc_memset(DAS uchar* d, uchar v, ulong n) {                                        \
    ulong i = 0;                                                                                   \
    if (((ulong)d & 3) == 0) { uint w = (uint)v * 0x01010101u; for (; i + 4 <= n; i += 4) *(DAS uint*)(d + i) = w; } \
    for (; i < n; i++) d[i] = v;                                                                   \
  }
__MVCC_MEMCPY(device, device) __MVCC_MEMCPY(device, threadgroup) __MVCC_MEMCPY(device, thread) __MVCC_MEMCPY(device, constant)
__MVCC_MEMCPY(threadgroup, device) __MVCC_MEMCPY(threadgroup, threadgroup) __MVCC_MEMCPY(threadgroup, thread) __MVCC_MEMCPY(threadgroup, constant)
__MVCC_MEMCPY(thread, device) __MVCC_MEMCPY(thread, threadgroup) __MVCC_MEMCPY(thread, thread) __MVCC_MEMCPY(thread, constant)
__MVCC_MEMSET(device) __MVCC_MEMSET(threadgroup) __MVCC_MEMSET(thread)
#undef __MVCC_MEMCPY
#undef __MVCC_MEMSET
#endif

// ---------------------------------------------------------------------------------------------------------------
// cp.async with src-size (zero-fill): copy src_size bytes, zero the remainder of the 4/8/16-byte chunk.
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_cpasync
inline void __mvcc_cp_async_zfill_16(__MVCC_SMEM uchar* dst, device uchar* src, uint src_size) {
  if (src_size >= 16) { *(__MVCC_SMEM uint4*)dst = *(device uint4*)src; return; }
  uint4 v = uint4(0);
  thread uchar* vb = (thread uchar*)&v;
  for (uint i = 0; i < src_size; i++) vb[i] = src[i];
  *(__MVCC_SMEM uint4*)dst = v;
}
inline void __mvcc_cp_async_zfill_8(__MVCC_SMEM uchar* dst, device uchar* src, uint src_size) {
  if (src_size >= 8) { *(__MVCC_SMEM uint2*)dst = *(device uint2*)src; return; }
  uint2 v = uint2(0);
  thread uchar* vb = (thread uchar*)&v;
  for (uint i = 0; i < src_size; i++) vb[i] = src[i];
  *(__MVCC_SMEM uint2*)dst = v;
}
inline void __mvcc_cp_async_zfill_4(__MVCC_SMEM uchar* dst, device uchar* src, uint src_size) {
  if (src_size >= 4) { *(__MVCC_SMEM uint*)dst = *(device uint*)src; return; }
  uint v = 0;
  thread uchar* vb = (thread uchar*)&v;
  for (uint i = 0; i < src_size; i++) vb[i] = src[i];
  *(__MVCC_SMEM uint*)dst = v;
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// ldmatrix.sync.aligned.m8n8.{x1,x2,x4}{.trans}.shared.b16
// Lanes 8i..8i+7 supply the row addresses of matrix i. Result for lane l (g=l>>2, t=l&3):
//   row:   elements (g, 2t), (g, 2t+1)   -> a 32-bit load at rowaddr(g) + 4t
//   trans: elements (2t, g), (2t+1, g)   -> two 16-bit loads at rowaddr(2t)+2g and rowaddr(2t+1)+2g
// The address of another lane is fetched with a shuffle of the 64-bit pointer (as two 32-bit halves).
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_ldmatrix
namespace mvcc_detail {
inline ulong shfl_addr(ulong a, uint src) {
  uint2 h = as_type<uint2>(a);
  h.x = simd_shuffle(h.x, (ushort)src);
  h.y = simd_shuffle(h.y, (ushort)src);
  return as_type<ulong>(h);
}
}
inline uint __mvcc_ldmatrix_row(ulong a, uint i, uint lane) {
  uint g = lane >> 2, t = lane & 3;
  ulong ra = mvcc_detail::shfl_addr(a, 8u * i + g);
  return *(__MVCC_SMEM uint*)(ra + 4u * t);
}
inline uint __mvcc_ldmatrix_trans(ulong a, uint i, uint lane) {
  uint g = lane >> 2, t = lane & 3;
  ulong r0 = mvcc_detail::shfl_addr(a, 8u * i + 2u * t);
  ulong r1 = mvcc_detail::shfl_addr(a, 8u * i + 2u * t + 1u);
  uint lo = *(__MVCC_SMEM ushort*)(r0 + 2u * g);
  uint hi = *(__MVCC_SMEM ushort*)(r1 + 2u * g);
  return lo | (hi << 16);
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// mma.sync on simdgroup_matrix 8x8 tiles.
//
// PTX fragment layout (g = lane>>2, t = lane&3): each 32-bit register of a 16-bit-typed 8x8 block holds the pair
// (row g, cols 2t..2t+1) — for A row-major blocks, and for B ("col") the pair (k=2t..2t+1, n=g) i.e. the same shape
// on B^T. f32 accumulators: (c0,c1) = (row g, cols 2t..2t+1) of the top block, (c2,c3) the bottom block.
//
// simdgroup_matrix<T,8,8> layout (measured): lane l holds (row = ((l>>1)&3)|((l>>4)<<2), cols c..c+1 with
// c = ((l&1)<<1)|(((l>>3)&1)<<2)). Both layouts hold one horizontally adjacent pair per lane, so PTX<->Metal is a
// pure lane permutation: one shuffle per register. B must additionally be transposed, which costs two shuffles per
// 8x8 block (the two halves of the pair come from different PTX lanes).
//
// The accumulate C is added after the multiply in PTX layout, saving its shuffles: D = shuffle_back(A*B) + C.
// fp32 accumulation over 8-term dot products of half products is exact for integers < 2^24, so the s8/u8 variants
// are computed exactly through the half path (products <= 2^14, 32-term sums < 2^20). fp8 e4m3/e5m2 are widened to
// half exactly (e4m3 range fits half; e5m2 is a half with a truncated mantissa).
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_mma
namespace mvcc_detail {
// PTX lane holding the pair (r, p) with r=row, p=pair index (cols 2p, 2p+1)
inline uint ptx_lane(uint r, uint p) { return 4u * r + p; }
// Metal lane holding the pair (r, p)
inline uint metal_lane(uint r, uint p) { return (p & 1u) | ((r & 3u) << 1) | ((p >> 1) << 3) | ((r >> 2) << 4); }
// (row, pair) a Metal lane holds
inline uint metal_row(uint l) { return ((l >> 1) & 3u) | ((l >> 4) << 2); }
inline uint metal_pair(uint l) { return (l & 1u) | (((l >> 3) & 1u) << 1); }

// Gather an A-shaped (row-major pair) 8x8 block from a PTX register into Metal layout.
inline uint gather_rowmajor(uint reg, uint lane) {
  return simd_shuffle(reg, (ushort)ptx_lane(metal_row(lane), metal_pair(lane)));
}
// Gather B (PTX holds B^T pairs: register at PTX lane 4n+p = B[2p..2p+1][n]) into a Metal 8x8 block of B[k][n]:
// Metal lane needs B[k=r][n=c], B[r][c+1] with c = 2*pair: from PTX lanes 4c + (r>>1) and 4(c+1) + (r>>1), half r&1.
inline uint gather_B(uint reg, uint lane) {
  uint r = metal_row(lane), c = 2u * metal_pair(lane);
  uint v0 = simd_shuffle(reg, (ushort)(4u * c + (r >> 1)));
  uint v1 = simd_shuffle(reg, (ushort)(4u * (c + 1u) + (r >> 1)));
  uint sh = (r & 1u) * 16u;
  return ((v0 >> sh) & 0xffffu) | (((v1 >> sh) & 0xffffu) << 16);
}
// Return a Metal-layout float pair (packed as two floats) to PTX layout.
inline float2 scatter_back(float2 v, uint lane) {
  uint src = metal_lane(lane >> 2, lane & 3u);
  return float2(simd_shuffle(v.x, (ushort)src), simd_shuffle(v.y, (ushort)src));
}
template <typename T> inline simdgroup_matrix<T, 8, 8> mat_from_pair(uint packed) {
  simdgroup_matrix<T, 8, 8> m;
  vec<T, 2> e = as_type<vec<T, 2>>(packed);
  m.thread_elements()[0] = e.x; m.thread_elements()[1] = e.y;
  return m;
}
inline simdgroup_float8x8 zero_f32() { simdgroup_float8x8 z; z.thread_elements()[0] = 0.f; z.thread_elements()[1] = 0.f; return z; }
inline float2 pair_of(simdgroup_float8x8 m) { return float2(m.thread_elements()[0], m.thread_elements()[1]); }

// D(16x8, f32 pairs) = A(16x16 as 4 PTX regs) * B(16x8 as 2 PTX regs), element type T (half or bfloat)
template <typename T>
inline void mma_16x8x16(thread float2& d_top, thread float2& d_bot, uint4 a, uint2 b, uint lane) {
  simdgroup_matrix<T, 8, 8> A00 = mat_from_pair<T>(gather_rowmajor(a.x, lane));
  simdgroup_matrix<T, 8, 8> A10 = mat_from_pair<T>(gather_rowmajor(a.y, lane));
  simdgroup_matrix<T, 8, 8> A01 = mat_from_pair<T>(gather_rowmajor(a.z, lane));
  simdgroup_matrix<T, 8, 8> A11 = mat_from_pair<T>(gather_rowmajor(a.w, lane));
  simdgroup_matrix<T, 8, 8> B0 = mat_from_pair<T>(gather_B(b.x, lane));
  simdgroup_matrix<T, 8, 8> B1 = mat_from_pair<T>(gather_B(b.y, lane));
  simdgroup_float8x8 Dt = zero_f32(), Db = zero_f32();
  simdgroup_multiply_accumulate(Dt, A00, B0, Dt);
  simdgroup_multiply_accumulate(Dt, A01, B1, Dt);
  simdgroup_multiply_accumulate(Db, A10, B0, Db);
  simdgroup_multiply_accumulate(Db, A11, B1, Db);
  d_top = scatter_back(pair_of(Dt), lane);
  d_bot = scatter_back(pair_of(Db), lane);
}
template <typename T>
inline void mma_16x8x8(thread float2& d_top, thread float2& d_bot, uint2 a, uint b, uint lane) {
  simdgroup_matrix<T, 8, 8> A0 = mat_from_pair<T>(gather_rowmajor(a.x, lane));
  simdgroup_matrix<T, 8, 8> A1 = mat_from_pair<T>(gather_rowmajor(a.y, lane));
  simdgroup_matrix<T, 8, 8> B0 = mat_from_pair<T>(gather_B(b, lane));
  simdgroup_float8x8 Dt = zero_f32(), Db = zero_f32();
  simdgroup_multiply_accumulate(Dt, A0, B0, Dt);
  simdgroup_multiply_accumulate(Db, A1, B0, Db);
  d_top = scatter_back(pair_of(Dt), lane);
  d_bot = scatter_back(pair_of(Db), lane);
}
inline uint4 add_c_f32(float2 t, float2 b, uint4 c) {
  return uint4(as_type<uint>(t.x + as_type<float>(c.x)), as_type<uint>(t.y + as_type<float>(c.y)),
               as_type<uint>(b.x + as_type<float>(c.z)), as_type<uint>(b.y + as_type<float>(c.w)));
}

// --- 8-bit -> half widening. Each 32-bit register holds 4 consecutive-k elements of one row/column; the k32 mma is
// computed as two k16 halves under a consistent k-permutation (k is a reduction index, so any bijection shared by A
// and B is valid): the 4 bytes of a PTX k32 register become the two 16-bit pairs of the k16 registers (lo pair, hi pair).
inline uint pack_half2(half lo, half hi) { return (uint)as_type<ushort>(lo) | ((uint)as_type<ushort>(hi) << 16); }
inline uint2 s8x4_to_half2x2(uint v) {
  char4 c = as_type<char4>(v);
  return uint2(pack_half2((half)c.x, (half)c.y), pack_half2((half)c.z, (half)c.w));
}
inline uint2 u8x4_to_half2x2(uint v) {
  uchar4 c = as_type<uchar4>(v);
  return uint2(pack_half2((half)c.x, (half)c.y), pack_half2((half)c.z, (half)c.w));
}
inline half e4m3_to_half(uint b) {
  uint s = (b & 0x80u) << 8, e = (b >> 3) & 0xfu, m = b & 7u;
  // normal: exponent rebias 7 -> 15; subnormal: m * 2^-9; 0x7f (either sign) is NaN, no infinities.
  ushort bits = (ushort)(s | ((e + 8u) << 10) | (m << 7));
  half h = as_type<half>(bits);
  if (e == 0u) h = (half)((s ? -1.f : 1.f) * (float)m * 0.001953125f);  // +-m * 2^-9
  if ((b & 0x7fu) == 0x7fu) h = as_type<half>((ushort)0x7e00u);
  return h;
}
inline half e5m2_to_half(uint b) { return as_type<half>((ushort)((b & 0xffu) << 8)); }
inline uint2 e4m3x4_to_half2x2(uint v) {
  return uint2(pack_half2(e4m3_to_half(v), e4m3_to_half(v >> 8)), pack_half2(e4m3_to_half(v >> 16), e4m3_to_half(v >> 24)));
}
inline uint2 e5m2x4_to_half2x2(uint v) {
  return uint2(pack_half2(e5m2_to_half(v), e5m2_to_half(v >> 8)), pack_half2(e5m2_to_half(v >> 16), e5m2_to_half(v >> 24)));
}
// k32 mma over 8-bit inputs given a per-register widening function: returns the fp32 pair results before C.
template <typename F>
inline void mma_16x8x32_via_half(thread float2& d_top, thread float2& d_bot, uint4 a, uint2 b, uint lane, F widen) {
  uint2 a0 = widen(a.x), a1 = widen(a.y), a2 = widen(a.z), a3 = widen(a.w), b0 = widen(b.x), b1 = widen(b.y);
  // first k16 slice: k=0..15 of PTX (registers a0,a1,b0); second: k=16..31 (a2,a3,b1)
  float2 t0, b0d, t1, b1d;
  mma_16x8x16<half>(t0, b0d, uint4(a0.x, a1.x, a0.y, a1.y), uint2(b0.x, b0.y), lane);
  mma_16x8x16<half>(t1, b1d, uint4(a2.x, a3.x, a2.y, a3.y), uint2(b1.x, b1.y), lane);
  d_top = t0 + t1; d_bot = b0d + b1d;
}
inline uint4 add_c_s32(float2 t, float2 b, uint4 c) {
  return uint4((uint)((int)rint(t.x) + (int)c.x), (uint)((int)rint(t.y) + (int)c.y),
               (uint)((int)rint(b.x) + (int)c.z), (uint)((int)rint(b.y) + (int)c.w));
}
}  // namespace mvcc_detail

inline void __mvcc_mma_m16n8k16_f32_f16_f16_f32(thread uint4& d, uint4 a, uint2 b, uint4 c, uint lane) {
  float2 t, bo; mvcc_detail::mma_16x8x16<half>(t, bo, a, b, lane); d = mvcc_detail::add_c_f32(t, bo, c);
}
inline void __mvcc_mma_m16n8k16_f32_bf16_bf16_f32(thread uint4& d, uint4 a, uint2 b, uint4 c, uint lane) {
  float2 t, bo; mvcc_detail::mma_16x8x16<bfloat>(t, bo, a, b, lane); d = mvcc_detail::add_c_f32(t, bo, c);
}
inline void __mvcc_mma_m16n8k8_f32_f16_f16_f32(thread uint4& d, uint2 a, uint b, uint4 c, uint lane) {
  float2 t, bo; mvcc_detail::mma_16x8x8<half>(t, bo, a, b, lane); d = mvcc_detail::add_c_f32(t, bo, c);
}
inline void __mvcc_mma_m16n8k8_f32_bf16_bf16_f32(thread uint4& d, uint2 a, uint b, uint4 c, uint lane) {
  float2 t, bo; mvcc_detail::mma_16x8x8<bfloat>(t, bo, a, b, lane); d = mvcc_detail::add_c_f32(t, bo, c);
}
// half accumulate: PTX c/d are 2 regs of half2: c0 = (row g, cols 2t..2t+1) top, c1 = bottom
inline void __mvcc_mma_m16n8k16_f16_f16_f16_f16(thread uint2& d, uint4 a, uint2 b, uint2 c, uint lane) {
  float2 t, bo; mvcc_detail::mma_16x8x16<half>(t, bo, a, b, lane);
  half2 ct = as_type<half2>(c.x), cb = as_type<half2>(c.y);
  half2 dt = half2((half)t.x + ct.x, (half)t.y + ct.y), db = half2((half)bo.x + cb.x, (half)bo.y + cb.y);
  d = uint2(as_type<uint>(dt), as_type<uint>(db));
}
inline void __mvcc_mma_m16n8k32_s32_s8_s8_s32(thread uint4& d, uint4 a, uint2 b, uint4 c, uint lane) {
  float2 t, bo; mvcc_detail::mma_16x8x32_via_half(t, bo, a, b, lane, [](uint v) { return mvcc_detail::s8x4_to_half2x2(v); });
  d = mvcc_detail::add_c_s32(t, bo, c);
}
inline void __mvcc_mma_m16n8k32_s32_u8_u8_s32(thread uint4& d, uint4 a, uint2 b, uint4 c, uint lane) {
  float2 t, bo; mvcc_detail::mma_16x8x32_via_half(t, bo, a, b, lane, [](uint v) { return mvcc_detail::u8x4_to_half2x2(v); });
  d = mvcc_detail::add_c_s32(t, bo, c);
}
// m16n8k16 s8: A regs a0 = A[g][4t..4t+3], a1 = A[g+8][4t..]; b0 = B[4t..4t+3][g]. One k16 slice.
inline void __mvcc_mma_m16n8k16_s32_s8_s8_s32(thread uint4& d, uint2 a, uint b, uint4 c, uint lane) {
  uint2 a0 = mvcc_detail::s8x4_to_half2x2(a.x), a1 = mvcc_detail::s8x4_to_half2x2(a.y), b0 = mvcc_detail::s8x4_to_half2x2(b);
  float2 t, bo; mvcc_detail::mma_16x8x16<half>(t, bo, uint4(a0.x, a1.x, a0.y, a1.y), uint2(b0.x, b0.y), lane);
  d = mvcc_detail::add_c_s32(t, bo, c);
}
inline void __mvcc_mma_m16n8k32_f32_e4m3_e4m3_f32(thread uint4& d, uint4 a, uint2 b, uint4 c, uint lane) {
  float2 t, bo; mvcc_detail::mma_16x8x32_via_half(t, bo, a, b, lane, [](uint v) { return mvcc_detail::e4m3x4_to_half2x2(v); });
  d = mvcc_detail::add_c_f32(t, bo, c);
}
inline void __mvcc_mma_m16n8k32_f32_e5m2_e5m2_f32(thread uint4& d, uint4 a, uint2 b, uint4 c, uint lane) {
  float2 t, bo; mvcc_detail::mma_16x8x32_via_half(t, bo, a, b, lane, [](uint v) { return mvcc_detail::e5m2x4_to_half2x2(v); });
  d = mvcc_detail::add_c_f32(t, bo, c);
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// mvcc::warp_tile (include/mvcc/tile.cuh) on Metal 4 TensorOps. One simdgroup owns an M x N fp32 accumulator
// held as M*N/32 floats per lane in the cooperative tensor's own order; __mvcc_tile_coord exposes that order.
// A is M x K (row stride lda), B is N x K (row stride ldb): C += A . B^T, so B is passed with transpose_right.
// The accumulator round-trips through the cooperative tensor per call; the Metal compiler keeps both in registers.
// ---------------------------------------------------------------------------------------------------------------
// ---------------------------------------------------------------------------------------------------------------
// Tensor-pipeline recovery (README.md, "Tensor cores"). The compiler rewrites ldmatrix -> mma.sync grids into these
// helpers. Everything rests on one measured fact, the cooperative-tensor layout law, which the runtime checks on the
// device before it enables recovered kernels (verify_tensor_ops, crates/mvcc-cudart/src/module.rs):
//
//   slot s of lane l in an R x C cooperative tile holds (row, col) with, for the non-transposed law,
//     row = ((l>>1)&3) | ((l>>4)<<2) | ((s>>2)&1)<<3        col = (s&3) | ((l&1)<<2) | (((l>>3)&1)<<3)
//   the transposed law swaps the two; then bit 4 extends: if C == 32, col |= ((s>>3)&1)<<4 and (if R == 32)
//   row |= ((s>>4)&1)<<4; else if R == 32, row |= ((s>>3)&1)<<4.
//
// Consequences the code below relies on: slots come in runs of 4 along the contiguous dimension (8-byte loads);
// every 16x16 sub-block of a fused tile is exactly the 8 slots with fixed (s3, s4) and follows the 16x16 law;
// left/right inputs and the destination obey the same law (inputs pick the transposed law via the descriptor's
// transpose flag, which the compiler sets from the ldmatrix .trans modifier). The runtime verifies the law on the
// running OS before using any recovered kernel (emitVerifyKernels); if it ever changes, the exact twins run.
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_tp
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
namespace mvcc_tp {
constexpr uint law_row(uint l, uint s, bool transposed, int R, int C) {
  uint br = ((l >> 1) & 3u) | ((l >> 4) << 2) | (((s >> 2) & 1u) << 3);
  uint bc = (s & 3u) | ((l & 1u) << 2) | (((l >> 3) & 1u) << 3);
  uint row = transposed ? bc : br;
  if (C == 32) { if (R == 32) row |= ((s >> 4) & 1u) << 4; }
  else if (R == 32) row |= ((s >> 3) & 1u) << 4;
  return row;
}
constexpr uint law_col(uint l, uint s, bool transposed, int R, int C) {
  uint br = ((l >> 1) & 3u) | ((l >> 4) << 2) | (((s >> 2) & 1u) << 3);
  uint bc = (s & 3u) | ((l & 1u) << 2) | (((l >> 3) & 1u) << 3);
  uint col = transposed ? br : bc;
  if (C == 32) col |= ((s >> 3) & 1u) << 4;
  return col;
}
}  // namespace mvcc_tp

// Fill this lane's 8 slots (4 packed words) of one 16x16 operand block from the ldmatrix row addresses held across
// the simdgroup. addr: this lane's PTX ldmatrix operand (32-bit shared address; lanes 8i..8i+7 address the rows of
// 8x8 matrix i). kind 0: A block (rows m, cols k); kind 1: B block (rows k, cols n). trans: the ldmatrix .trans
// modifier. blockmap: 2 bits per 8x8 block r = (row>=8) + 2*(col>=8) giving the matrix index that block came from.
inline void __mvcc_tp_ld(thread uint* w, ulong addr, uint lane, int kind, int trans, uint blockmap) {
  // contiguous dimension of the block in memory: A non-trans / B trans -> columns; A trans / B non-trans -> rows
  const bool lawT = kind == 0 ? (trans != 0) : (trans == 0);
  uint2 h = as_type<uint2>(addr);
  for (uint j = 0; j < 2; j++) {
    uint row = mvcc_tp::law_row(lane, 4u * j, lawT, 16, 16), col = mvcc_tp::law_col(lane, 4u * j, lawT, 16, 16);
    uint contig = lawT ? row : col, rowlike = lawT ? col : row;
    uint r = (row >> 3) + ((col >> 3) << 1);
    uint mat = (blockmap >> (2u * r)) & 3u;
    ushort src = (ushort)(8u * mat + (rowlike & 7u));
    uint2 ra = uint2(simd_shuffle(h.x, src), simd_shuffle(h.y, src));
    uint2 v = *(__MVCC_SMEM uint2*)(as_type<ulong>(ra) + 2u * (contig & 7u));
    w[2 * j] = v.x; w[2 * j + 1] = v.y;
  }
}

// Register-built operand block (tensor_recovery.cpp, L1'): the four PTX-layout words of a 16x16 block held across the
// simdgroup - A: a0..a3 of mma.sync (lane 4g+t: a_{(m>=8)+2(k>=8)} = row g(+8), k 2t,2t+1 (+8)); B: x.b0, x.b1, y.b0,
// y.b1 of two n8 tiles (b_{k>=8} = k 2t,2t+1 (+8), n g; y = n+8) - moved into this lane's 4 slot words. Slot quad j
// of lane l is (A, non-transposed law) row base+8j, cols 8hi+4(l&1)..+3, (B, transposed law) col base+8j, rows
// 8hi+4(l&1)..+3, with base = ((l>>1)&3)|((l>>4)<<2) and hi = (l>>3)&1: word 2j+i is the PTX word of lane
// 4*base+2(l&1)+i, register j+2hi (A) or field hi+2j (B). The register index depends on the reading lane, so both
// candidates are shuffled and selected: 8 shuffles per block.
inline void __mvcc_tp_frag2slot(thread uint* w, uint f0, uint f1, uint f2, uint f3, uint lane, int kind) {
  const bool hi = ((lane >> 3) & 1u) != 0u;
  const ushort s0 = (ushort)(4u * (((lane >> 1) & 3u) | ((lane >> 4) << 2)) + 2u * (lane & 1u)), s1 = s0 + 1;
  uint p0 = simd_shuffle(f0, s0), p1 = simd_shuffle(f1, s0), p2 = simd_shuffle(f2, s0), p3 = simd_shuffle(f3, s0);
  uint q0 = simd_shuffle(f0, s1), q1 = simd_shuffle(f1, s1), q2 = simd_shuffle(f2, s1), q3 = simd_shuffle(f3, s1);
  if (kind == 0) { w[0] = hi ? p2 : p0; w[1] = hi ? q2 : q0; w[2] = hi ? p3 : p1; w[3] = hi ? q3 : q1; }
  else           { w[0] = hi ? p1 : p0; w[1] = hi ? q1 : q0; w[2] = hi ? p3 : p2; w[3] = hi ? q3 : q2; }
}

// Level 4 direct fills (staging_layout.cpp): the compiler re-evaluates the ldmatrix address expression for the
// lane whose row feeds slot quad j (__mvcc_tp_srclane), so the fill loads straight from that address: no shuffles.
inline uint __mvcc_tp_srclane(uint lane, int j, int kind, int trans, uint blockmap) {
  const bool lawT = kind == 0 ? (trans != 0) : (trans == 0);
  uint row = mvcc_tp::law_row(lane, 4u * (uint)j, lawT, 16, 16), col = mvcc_tp::law_col(lane, 4u * (uint)j, lawT, 16, 16);
  uint rowlike = lawT ? col : row;
  uint r = (row >> 3) + ((col >> 3) << 1);
  uint mat = (blockmap >> (2u * r)) & 3u;
  return 8u * mat + (rowlike & 7u);
}
inline void __mvcc_tp_ld_direct(thread uint* w, ulong a0, ulong a1, uint lane, int kind, int trans, int elem) {
  const bool lawT = kind == 0 ? (trans != 0) : (trans == 0);
  for (uint j = 0; j < 2; j++) {
    uint row = mvcc_tp::law_row(lane, 4u * j, lawT, 16, 16), col = mvcc_tp::law_col(lane, 4u * j, lawT, 16, 16);
    uint contig = lawT ? row : col;
    if (elem == 1) {
#if MVCC_NEED_mma
      uint v = *(__MVCC_SMEM uint*)((j ? a1 : a0) + (contig & 7u));
      uint2 h = mvcc_detail::e4m3x4_to_half2x2(v);
      w[2 * j] = h.x; w[2 * j + 1] = h.y;
#else
      w[2 * j] = 0u; w[2 * j + 1] = 0u;
#endif
    } else {
      uint2 v = *(__MVCC_SMEM uint2*)((j ? a1 : a0) + 2u * (contig & 7u));
      w[2 * j] = v.x; w[2 * j + 1] = v.y;
    }
  }
}
inline void __mvcc_tp_ld_direct(thread uint* w, ulong a0, ulong a1, uint lane, int kind, int trans) {
  __mvcc_tp_ld_direct(w, a0, a1, lane, kind, trans, 2);
}

// Level 4 pointer ring (staging_elim.cpp). A 16-byte chunk the CUDA kernel staged at shared address a lives, as an
// 8-byte device address, at slot(a) = obj + (a - obj) / 2 inside its shared-memory object (objects are 16-byte
// aligned, chunks are 16-byte aligned: slots are distinct and 8-byte aligned). zmode: 0 every copy is 16 bytes;
// 1 a copy is 16 bytes or all zero (src-size 0 -> null slot); 2 general src-size (16 - size in the address's low
// 4 bits; the source is 16-byte aligned by the cp.async contract).
inline void __mvcc_tp_stage(ulong dst, ulong obj, ulong src, int srcsize, int zmode) {
  ulong v = src;
  if (zmode == 1) v = srcsize ? src : 0ul;
  else if (zmode == 2) v = srcsize ? (src | (ulong)((16 - srcsize) & 15)) : 0ul;
  *(__MVCC_SMEM ulong*)(obj + ((dst - obj) >> 1)) = v;
}
// 8 bytes at byte offset off (0 or 8) of the chunk whose ring entry is q.
inline uint2 __mvcc_tp_ring_load(ulong q, uint off, int zmode) {
  if (zmode == 0) return *(device uint2*)(q + off);
  if (q == 0ul) return uint2(0u);
  if (zmode == 1) return *(device uint2*)(q + off);
  uint n = 16u - (uint)(q & 15ul);                 // valid source bytes [0, n)
  ulong base = q & ~15ul;
  if (off + 8u <= n) return *(device uint2*)(base + off);
  uint2 v = uint2(0u);
  for (uint b = off; b < n; b++) { uint byte = *(device uchar*)(base + b); if (b - off < 4u) v.x |= byte << (8u * (b - off)); else v.y |= byte << (8u * (b - off - 4u)); }
  return v;
}
inline void __mvcc_tp_ld_ring(thread uint* w, ulong addr, ulong obj, uint lane, int kind, int trans, uint blockmap, int zmode) {
  const bool lawT = kind == 0 ? (trans != 0) : (trans == 0);
  ulong p = *(__MVCC_SMEM ulong*)(obj + ((addr - obj) >> 1));   // this lane's row: its chunk's device address
  uint2 h = as_type<uint2>(p);
  for (uint j = 0; j < 2; j++) {
    uint row = mvcc_tp::law_row(lane, 4u * j, lawT, 16, 16), col = mvcc_tp::law_col(lane, 4u * j, lawT, 16, 16);
    uint contig = lawT ? row : col, rowlike = lawT ? col : row;
    uint r = (row >> 3) + ((col >> 3) << 1);
    uint mat = (blockmap >> (2u * r)) & 3u;
    ushort src = (ushort)(8u * mat + (rowlike & 7u));
    uint2 ra = uint2(simd_shuffle(h.x, src), simd_shuffle(h.y, src));
    uint2 v = __mvcc_tp_ring_load(as_type<ulong>(ra), 2u * (contig & 7u), zmode);
    w[2 * j] = v.x; w[2 * j + 1] = v.y;
  }
}

// Compile-time loops. Every array index below must be a constant after expansion: a `for` the Metal compiler
// declines to unroll would turn the accumulator/fragment arrays into memory-resident objects (measured: 15x slower
// kernels). Recursion over a template index cannot fail to expand.
template <int I> struct __mvcc_ic { static constexpr constant int value = I; };
template <int I, int N, typename F> inline void __mvcc_static_for(F f) {
  if constexpr (I < N) { f(__mvcc_ic<I>{}); __mvcc_static_for<I + 1, N>(f); }
}

// Cooperative matmul2d: a = M*K/64 words (A slots), b = K*N/64 words, c/d = M*N/32 accumulator slots.
template <typename T, int M, int N, int K, bool TL, bool TR>
inline void __mvcc_tp_mma(thread float* d, const thread uint* a, const thread uint* b, const thread float* c) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(M, N, K, TL, TR, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<1>> op;
  auto ca = op.template get_left_input_cooperative_tensor<T, T, float>();
  auto cb = op.template get_right_input_cooperative_tensor<T, T, float>();
  __mvcc_static_for<0, M * K / 64>([&](auto i) { metal::vec<T, 2> v = as_type<metal::vec<T, 2>>(a[i.value]); ca[2 * i.value] = v[0]; ca[2 * i.value + 1] = v[1]; });
  __mvcc_static_for<0, K * N / 64>([&](auto i) { metal::vec<T, 2> v = as_type<metal::vec<T, 2>>(b[i.value]); cb[2 * i.value] = v[0]; cb[2 * i.value + 1] = v[1]; });
  auto ct = op.template get_destination_cooperative_tensor<decltype(ca), decltype(cb), float>();
  __mvcc_static_for<0, M * N / 32>([&](auto i) { ct[i.value] = c[i.value]; });
  op.run(ca, cb, ct);
  __mvcc_static_for<0, M * N / 32>([&](auto i) { d[i.value] = ct[i.value]; });
}

// Accumulator layout conversions go through a per-simdgroup threadgroup scratch block (8 rows x 16 cols of float =
// 512 bytes, reserved by the compiler next to the kernel's static shared memory; a tile converts one block per
// round) rather than through simd_shuffle. Both are 2*M*N/32 operations per lane, but shuffling the
// cooperative-tensor results proved catastrophically slow in Metal's compiler (a 64x64 GEMM tile went from 1.4 ms to
// 17 ms for one conversion at the very end of the kernel); the scratch version costs nothing measurable.
// Coordinates: slot s of lane l is (row, col) by the layout law; mma.sync position p = (mt*(N/8) + nt)*4 + r of lane
// l is D[16mt + (l>>2) + 8(r>>1)][8nt + 2(l&3) + (r&1)]. Block membership (row>>3, col>>4) of a slot or a position
// does not depend on the lane (row bit 3 is a slot bit, col bit 3 the only lane-dependent bit below 4), so every
// scratch index below is static apart from the in-block lane coordinates. 8x16 is the smallest such block.
template <int M, int N>
inline void __mvcc_tp_acc2ptx(thread float* out, const thread float* slots, uint lane, __MVCC_SMEM float* scratch) {
  const uint g = lane >> 2, t = lane & 3;
  __mvcc_static_for<0, (M / 8) * (N / 16)>([&](auto bi) {
    constexpr int rb = bi.value / (N / 16), cb = bi.value % (N / 16);
    __mvcc_static_for<0, M * N / 32>([&](auto si) {
      constexpr int s = si.value;
      constexpr int sr = (int)mvcc_tp::law_row(0u, (uint)s, false, M, N) >> 3, sc = (int)mvcc_tp::law_col(0u, (uint)s, false, M, N) >> 4;
      if constexpr (sr == rb && sc == cb) {
        const uint m = mvcc_tp::law_row(lane, (uint)s, false, M, N), n = mvcc_tp::law_col(lane, (uint)s, false, M, N);
        scratch[(m & 7u) * 16u + (n & 15u)] = slots[s];
      }
    });
    simdgroup_barrier(__MVCC_SMEM_FLAGS);
    __mvcc_static_for<0, M * N / 32>([&](auto pi) {
      constexpr int p = pi.value, tile = p >> 2, r = p & 3, mt = tile / (N / 8), nt = tile % (N / 8);
      if constexpr (2 * mt + (r >> 1) == rb && nt / 2 == cb) {
        out[p] = scratch[g * 16u + 8u * (nt & 1) + 2u * t + (r & 1)];
      }
    });
    simdgroup_barrier(__MVCC_SMEM_FLAGS);
  });
}

// Epilogue store (epilogue_elim.cpp): the tile's M*N/32 per-lane values, already in slot order and converted to the
// output element type T, go to device memory through a cooperative destination tensor of T. The f16/bf16 destination
// tensors of a matmul2d follow the same layout law as the f32 one (verified by the runtime with the matmul law,
// emitVerifyKernels), so slot i of the T tensor is the same element as slot i of the accumulator. The tensor store is
// Metal's own layout-aware write, clipped to the extents (rows, cols) - the CUDA epilogue's edge predicates.
template <typename T, int M, int N, int K, bool TL, bool TR>
inline void __mvcc_tp_ctstore(const thread T* v, device T* base, int ld, int rows, int cols) {
  if (rows <= 0 || cols <= 0) return;
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(M, N, K, TL, TR, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<1>> op;
  using TA = decltype(op.template get_left_input_cooperative_tensor<T, T, float>());
  using TB = decltype(op.template get_right_input_cooperative_tensor<T, T, float>());
  auto ct = op.template get_destination_cooperative_tensor<TA, TB, T>();
  __mvcc_static_for<0, M * N / 32>([&](auto i) { ct[i.value] = v[i.value]; });
  metal::tensor<device T, metal::dextents<int, 2>, metal::tensor_inline> t(base, metal::dextents<int, 2>(cols, rows), metal::array<int, 2>({1, ld}));
  ct.store(t);
}

template <int M, int N>
inline void __mvcc_tp_ptx2acc(thread float* out, const thread float* ptx, uint lane, __MVCC_SMEM float* scratch) {
  const uint g = lane >> 2, t = lane & 3;
  __mvcc_static_for<0, (M / 8) * (N / 16)>([&](auto bi) {
    constexpr int rb = bi.value / (N / 16), cb = bi.value % (N / 16);
    __mvcc_static_for<0, M * N / 32>([&](auto pi) {
      constexpr int p = pi.value, tile = p >> 2, r = p & 3, mt = tile / (N / 8), nt = tile % (N / 8);
      if constexpr (2 * mt + (r >> 1) == rb && nt / 2 == cb) {
        scratch[g * 16u + 8u * (nt & 1) + 2u * t + (r & 1)] = ptx[p];
      }
    });
    simdgroup_barrier(__MVCC_SMEM_FLAGS);
    __mvcc_static_for<0, M * N / 32>([&](auto si) {
      constexpr int s = si.value;
      constexpr int sr = (int)mvcc_tp::law_row(0u, (uint)s, false, M, N) >> 3, sc = (int)mvcc_tp::law_col(0u, (uint)s, false, M, N) >> 4;
      if constexpr (sr == rb && sc == cb) {
        const uint m = mvcc_tp::law_row(lane, (uint)s, false, M, N), n = mvcc_tp::law_col(lane, (uint)s, false, M, N);
        out[s] = scratch[(m & 7u) * 16u + (n & 15u)];
      }
    });
    simdgroup_barrier(__MVCC_SMEM_FLAGS);
  });
}
#endif

#if MVCC_NEED_tile
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
namespace mvcc_tile {
template <typename T, int M, int N, int K> struct op_t {
  static constexpr constant auto desc = mpp::tensor_ops::matmul2d_descriptor(
      M, N, K, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  using op = mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroups<1>>;
  using dev_tensor = metal::tensor<device T, metal::dextents<int, 2>, metal::tensor_inline>;
};
}  // namespace mvcc_tile
#define MVCC_TILE_MMA(SA, SB)                                                                                     \
  template <typename T, int M, int N, int K>                                                                     \
  inline void __mvcc_tile_mma(thread float* acc, SA T* a, int lda, SB T* b, int ldb) {                             \
    using TA = metal::tensor<SA T, metal::dextents<int, 2>, metal::tensor_inline>;                               \
    using TB = metal::tensor<SB T, metal::dextents<int, 2>, metal::tensor_inline>;                               \
    TA tA(a, metal::dextents<int, 2>(K, M), metal::array<int, 2>({1, lda}));                                     \
    TB tB(b, metal::dextents<int, 2>(K, N), metal::array<int, 2>({1, ldb}));                                     \
    typename mvcc_tile::op_t<T, M, N, K>::op op;                                                                   \
    auto ct = op.template get_destination_cooperative_tensor<TA, TB, float>();                                   \
    _Pragma("unroll") for (uint16_t i = 0; i < ct.get_capacity(); ++i) ct[i] = acc[i];                          \
    op.run(tA, tB, ct);                                                                                          \
    _Pragma("unroll") for (uint16_t i = 0; i < ct.get_capacity(); ++i) acc[i] = ct[i];                          \
  }
MVCC_TILE_MMA(device, device)
MVCC_TILE_MMA(threadgroup, threadgroup)
MVCC_TILE_MMA(device, threadgroup)
MVCC_TILE_MMA(threadgroup, device)
#undef MVCC_TILE_MMA
template <typename T, int M, int N, int K>
inline int __mvcc_tile_coord(int i) {
  using TT = typename mvcc_tile::op_t<T, M, N, K>::dev_tensor;
  typename mvcc_tile::op_t<T, M, N, K>::op op;
  auto ct = op.template get_destination_cooperative_tensor<TT, TT, float>();
  if (i >= (int)ct.get_capacity() || !ct.is_valid_element((uint16_t)i)) return -1;
  auto ids = ct.get_multidimensional_index((uint16_t)i);
  return (ids[1] << 16) | ids[0];
}
#endif

// ---------------------------------------------------------------------------------------------------------------
// IEEE-754 binary64 in software. Apple GPUs have no fp64; `double` in device code is carried as mvcc_f64 (the 64
// bits) and every operation is emulated with 64-bit integer arithmetic, correctly rounded to nearest-even
// (add/sub/mul/div/sqrt/conversions) exactly as the hardware would. Slow (tens of integer ops per operation):
// meant for the odd score accumulator or reference computation, not for hot loops (README.md).
// ---------------------------------------------------------------------------------------------------------------
#if MVCC_NEED_f64
struct mvcc_f64 { ulong b; };
namespace mvcc_f64d {
constant ulong SIGN = 0x8000000000000000ul;
constant ulong QNAN = 0x7ff8000000000000ul;
inline int exp_of(ulong b) { return (int)((b >> 52) & 0x7ff); }
inline ulong frac_of(ulong b) { return b & 0xfffffffffffffull; }
inline bool is_nan(ulong b) { return exp_of(b) == 0x7ff && frac_of(b) != 0; }
inline bool is_inf(ulong b) { return exp_of(b) == 0x7ff && frac_of(b) == 0; }
inline ulong shr_jam(ulong x, int n) { return n == 0 ? x : (n < 64 ? ((x >> n) | ((x << (64 - n)) != 0 ? 1ul : 0ul)) : (x != 0 ? 1ul : 0ul)); }
// pack a significand with 10 extra low bits (bit 62 = leading 1, i.e. sig in [2^62, 2^63)) and biased exponent e
// (for that position) with round-to-nearest-even; handles overflow and subnormal results.
inline ulong round_pack(bool sign, int e, ulong sig) {
  if (e <= 0) {  // subnormal or zero: shift right by 1 - e and jam
    sig = shr_jam(sig, 1 - e);
    e = 0;
  }
  ulong round_bits = sig & 0x3ff;
  ulong r = sig >> 10;
  if (round_bits > 0x200 || (round_bits == 0x200 && (r & 1))) r++;
  if (r == 0) return (sign ? SIGN : 0ul);
  if (e == 0) {
    // subnormal: r < 2^52 stays subnormal, r == 2^52 became the smallest normal (exponent field 1 via the add)
    return (sign ? SIGN : 0ul) | r;  // r includes the implicit bit position for the normal case
  }
  if (r >= (1ul << 53)) { r >>= 1; e++; }
  if (e >= 0x7ff) return (sign ? SIGN : 0ul) | 0x7ff0000000000000ul;
  return (sign ? SIGN : 0ul) | ((ulong)e << 52) | (r & 0xfffffffffffffull);
}
// normalize a nonzero significand into [2^62, 2^63) adjusting e
inline ulong norm_pack(bool sign, int e, ulong sig) {
  int s = clz(sig) - 1;  // want bit 62 set
  if (s > 0) { sig <<= s; e -= s; } else if (s < 0) { sig = shr_jam(sig, -s); e -= s; }
  return round_pack(sign, e, sig);
}
inline ulong add_mag(bool sign, int ea, ulong sa, int eb, ulong sb) {
  // sa, sb: significands with the implicit bit at bit 52, shifted left by 10 -> bit 62; ea >= eb
  if (ea == 0x7ff) return (sign ? SIGN : 0ul) | 0x7ff0000000000000ul;
  sb = shr_jam(sb, ea - eb);
  ulong sum = sa + sb;  // may carry into bit 63
  if (sum & 0x8000000000000000ul) { sum = shr_jam(sum, 1); ea++; }
  return round_pack(sign, ea, sum);
}
inline ulong sub_mag(bool sign, int ea, ulong sa, int eb, ulong sb) {
  // |a| >= |b| by exponent (and by significand when equal exponents); result sign is `sign`.
  // Exponent distance 0 or 1: the difference is exact (one extra bit of headroom keeps b's low bit), and
  // cancellation may need a long normalizing shift. Distance >= 2: at most one shift, the jam bit stays sticky.
  if (ea - eb <= 1) {
    ulong a2 = sa << 1, b2 = (ea == eb) ? (sb << 1) : sb;  // both scaled by 2: exact
    ulong diff = a2 - b2;
    if (diff == 0) return 0ul;  // exact cancellation -> +0 (round-to-nearest)
    return norm_pack(sign, ea - 1, diff);
  }
  sb = shr_jam(sb, ea - eb);
  ulong diff = sa - sb;
  return norm_pack(sign, ea, diff);
}
inline void unpack(ulong b, thread int& e, thread ulong& sig) {
  e = exp_of(b);
  ulong f = frac_of(b);
  if (e == 0) {  // subnormal: normalize so bit 62 holds the leading 1
    if (f == 0) { sig = 0; return; }
    int s = clz(f) - 1;
    sig = f << s;  // leading 1 at bit 62; value = f * 2^-1074 = sig * 2^(-1074 - s)  =>  biased e = 11 - s
    e = 11 - s;
    return;
  }
  sig = (f | 0x10000000000000ul) << 10;
}
inline ulong add(ulong a, ulong b, bool negate_b) {
  if (negate_b) b ^= SIGN;
  bool sa = (a & SIGN) != 0, sb = (b & SIGN) != 0;
  int ea = exp_of(a), eb = exp_of(b);
  if (ea == 0x7ff || eb == 0x7ff) {
    if (is_nan(a) || is_nan(b)) return QNAN;
    if (ea == 0x7ff && eb == 0x7ff) return sa == sb ? a : QNAN;
    return ea == 0x7ff ? a : b;
  }
  int xa, xb; ulong ma, mb;
  unpack(a, xa, ma); unpack(b, xb, mb);
  if (ma == 0) return mb == 0 ? ((sa && sb) ? SIGN : 0ul) : b;
  if (mb == 0) return a;
  // order by magnitude
  bool swap = (xa < xb) || (xa == xb && ma < mb);
  if (swap) { int t = xa; xa = xb; xb = t; ulong u = ma; ma = mb; mb = u; bool v = sa; sa = sb; sb = v; }
  if (sa == sb) return add_mag(sa, xa, ma, xb, mb);
  return sub_mag(sa, xa, ma, xb, mb);
}
inline ulong mul(ulong a, ulong b) {
  bool sign = ((a ^ b) & SIGN) != 0;
  int ea = exp_of(a), eb = exp_of(b);
  if (ea == 0x7ff || eb == 0x7ff) {
    if (is_nan(a) || is_nan(b)) return QNAN;
    bool za = ea == 0 && frac_of(a) == 0, zb = eb == 0 && frac_of(b) == 0;
    if (za || zb) return QNAN;
    return (sign ? SIGN : 0ul) | 0x7ff0000000000000ul;
  }
  int xa, xb; ulong ma, mb;
  unpack(a, xa, ma); unpack(b, xb, mb);
  if (ma == 0 || mb == 0) return sign ? SIGN : 0ul;
  // significands at bit 62 (53 bits << 10); use the 53-bit forms for the product
  ulong pa = ma >> 10, pb = mb >> 10;                   // in [2^52, 2^53)
  ulong hi = mulhi(pa, pb), lo = pa * pb;               // 128-bit product in [2^104, 2^106)
  // bring the product to a 63-bit significand: take bits [105:43], jam the rest
  // product = hi:lo; we want sig = product >> 43 (63 bits max when product < 2^106)
  ulong sig = (hi << 21) | (lo >> 43);
  ulong rest = lo << 21;
  if (rest) sig |= 1;
  // value = sig * 2^(xa + xb - 2046 - 61): biased exponent for a bit-62 significand is xa + xb - 1022
  int e = xa + xb - 1022;
  if ((sig & 0x4000000000000000ul) == 0) { sig <<= 1; e -= 1; }  // sig in [2^61, 2^63): normalize to bit 62
  return round_pack(sign, e, sig);
}
inline ulong div(ulong a, ulong b) {
  bool sign = ((a ^ b) & SIGN) != 0;
  int ea = exp_of(a), eb = exp_of(b);
  if (is_nan(a) || is_nan(b)) return QNAN;
  bool ia = ea == 0x7ff, ib = eb == 0x7ff;
  int xa, xb; ulong ma, mb;
  unpack(a, xa, ma); unpack(b, xb, mb);
  if (ia && ib) return QNAN;
  if (ia) return (sign ? SIGN : 0ul) | 0x7ff0000000000000ul;
  if (ib) return sign ? SIGN : 0ul;
  if (mb == 0) return ma == 0 ? QNAN : ((sign ? SIGN : 0ul) | 0x7ff0000000000000ul);
  if (ma == 0) return sign ? SIGN : 0ul;
  ulong n = ma >> 10, d = mb >> 10;  // 53-bit
  // long division producing 55 quotient bits (1 integer bit position + 52 + guard/round) then sticky
  int e = xa - xb + 0x3ff;
  if (n < d) { n <<= 1; e -= 1; }
  ulong q = 0, r = n;
  for (int i = 0; i < 55; i++) {
    q <<= 1;
    if (r >= d) { r -= d; q |= 1; }
    r <<= 1;
  }
  // q has its leading 1 at bit 54; place it at bit 62 (shift 8) and jam the remainder
  ulong sig = (q << 8) | (r != 0 ? 1ul : 0ul);
  return round_pack(sign, e, sig);
}
inline ulong sqrt_(ulong a) {
  if (is_nan(a)) return QNAN;
  bool sa = (a & SIGN) != 0;
  int xa; ulong ma;
  unpack(a, xa, ma);
  if (ma == 0) return a;  // +-0
  if (sa) return QNAN;
  if (xa == 0x7ff) return a;
  int e = xa - 0x3ff;  // unbiased
  ulong m = ma >> 10;  // 53 bits, value = m * 2^(e-52)
  // make the exponent even: sqrt(m * 2^e') with e' even
  if (e & 1) { m <<= 1; e -= 1; }
  // integer sqrt of m << 54 (so the root has 53+1 bits): root = floor(sqrt(m * 2^54)), value * 2^((e-54)/2)...
  // compute root of N = m << 54 where m < 2^54 -> N < 2^108: do digit-by-digit with 128-bit remainder in two limbs.
  // N is placed at the top of the 128-bit pair (N << 20): 54 digit-by-digit steps consume exactly its 108 bits
  ulong nhi = m << 10, nlo = 0;
  ulong root = 0, remhi = 0, remlo = 0;
  for (int i = 0; i < 54; i++) {
    // shift 2 bits of N into rem
    remhi = (remhi << 2) | (remlo >> 62);
    remlo = (remlo << 2) | (nhi >> 62);
    nhi = (nhi << 2) | (nlo >> 62);
    nlo <<= 2;
    // trial = (root << 2) | 1
    ulong thi = root >> 62, tlo = (root << 2) | 1ul;
    root <<= 1;
    bool ge = (remhi > thi) || (remhi == thi && remlo >= tlo);
    if (ge) { ulong borrow = remlo < tlo ? 1ul : 0ul; remlo -= tlo; remhi -= thi + borrow; root |= 1ul; }
  }
  // root has 54 bits (leading 1 at bit 53); place at bit 62 and jam the remainder
  ulong sig = (root << 9) | ((remhi | remlo) != 0 ? 1ul : 0ul);
  int re = (e >> 1) + 0x3ff;
  return round_pack(false, re, sig);
}
inline bool lt(ulong a, ulong b) {
  if (is_nan(a) || is_nan(b)) return false;
  bool sa = (a & SIGN) != 0, sb = (b & SIGN) != 0;
  if (sa != sb) return sa && ((a | b) & ~SIGN) != 0;
  return sa ? (a > b) : (a < b);
}
inline bool le(ulong a, ulong b) {
  if (is_nan(a) || is_nan(b)) return false;
  bool sa = (a & SIGN) != 0, sb = (b & SIGN) != 0;
  if (sa != sb) return sa || ((a | b) & ~SIGN) == 0;
  return sa ? (a >= b) : (a <= b);
}
inline bool eq(ulong a, ulong b) {
  if (is_nan(a) || is_nan(b)) return false;
  return a == b || ((a | b) & ~SIGN) == 0;
}
inline bool unord(ulong a, ulong b) { return is_nan(a) || is_nan(b); }
inline ulong from_u64(ulong v) {
  if (v == 0) return 0ul;
  int s = clz(v) - 1;
  ulong sig = s >= 0 ? (v << s) : shr_jam(v, -s);
  int e = 0x3ff + 62 - s;
  return round_pack(false, e, sig);
}
inline ulong from_i64(long v) { ulong r = from_u64((ulong)(v < 0 ? -v : v)); return v < 0 ? (r | SIGN) : r; }
inline ulong from_f32(float f) {
  uint u = as_type<uint>(f);
  bool sign = (u >> 31) != 0; int e = (int)((u >> 23) & 0xff); uint m = u & 0x7fffff;
  if (e == 0xff) return (sign ? SIGN : 0ul) | 0x7ff0000000000000ul | (m ? 0x8000000000000ul : 0ul);
  if (e == 0) { if (m == 0) return sign ? SIGN : 0ul; ulong r = from_u64(m); r -= (ulong)(149) << 52; return sign ? (r | SIGN) : r; }
  return (sign ? SIGN : 0ul) | ((ulong)(e - 127 + 1023) << 52) | ((ulong)m << 29);
}
inline float to_f32(ulong a) {
  bool sign = (a & SIGN) != 0; int e = exp_of(a); ulong f = frac_of(a);
  if (e == 0x7ff) return as_type<float>((sign ? 0x80000000u : 0u) | 0x7f800000u | (f ? 0x400000u : 0u));
  if (e == 0 && f == 0) return as_type<float>(sign ? 0x80000000u : 0u);
  int xa; ulong ma; unpack(a, xa, ma);  // ma: bit 62 leading
  int fe = xa - 1023 + 127;
  // float significand: 24 bits from bit 62..39; round with the 39 low bits
  if (fe >= 0xff) return as_type<float>((sign ? 0x80000000u : 0u) | 0x7f800000u);
  if (fe <= 0) {  // subnormal float: shift right by 1 - fe more
    ma = shr_jam(ma, 1 - fe); fe = 0;
  }
  ulong r = ma >> 39; ulong rb = ma & ((1ul << 39) - 1);
  if (rb > (1ul << 38) || (rb == (1ul << 38) && (r & 1))) r++;
  uint out;
  if (fe == 0) { out = (uint)r; }  // may carry into the exponent field: correct by construction
  else { if (r >= (1ul << 24)) { r >>= 1; fe++; } if (fe >= 0xff) return as_type<float>((sign ? 0x80000000u : 0u) | 0x7f800000u); out = ((uint)fe << 23) | ((uint)r & 0x7fffff); }
  return as_type<float>((sign ? 0x80000000u : 0u) | out);
}
// truncating conversions with CUDA's saturation (NaN -> 0)
inline long to_i64(ulong a) {
  if (is_nan(a)) return 0;
  bool sign = (a & SIGN) != 0; int e = exp_of(a);
  if (e < 0x3ff) return 0;
  int sh = e - 0x3ff;
  if (sh >= 63) return sign ? (long)0x8000000000000000ul : 0x7fffffffffffffffl;
  ulong m = frac_of(a) | 0x10000000000000ul;
  ulong v = sh >= 52 ? (m << (sh - 52)) : (m >> (52 - sh));
  return sign ? -(long)v : (long)v;
}
inline ulong to_u64(ulong a) {
  if (is_nan(a) || (a & SIGN)) return 0;
  int e = exp_of(a);
  if (e < 0x3ff) return 0;
  int sh = e - 0x3ff;
  if (sh >= 64) return 0xfffffffffffffffful;
  ulong m = frac_of(a) | 0x10000000000000ul;
  return sh >= 52 ? (m << (sh - 52)) : (m >> (52 - sh));
}
inline int to_i32(ulong a) { long v = to_i64(a); return v > 2147483647l ? 2147483647 : (v < -2147483648l ? (int)0x80000000 : (int)v); }
inline uint to_u32(ulong a) { ulong v = to_u64(a); return v > 0xfffffffful ? 0xffffffffu : (uint)v; }
inline ulong trunc_(ulong a) {
  int e = exp_of(a);
  if (e >= 0x3ff + 52 || is_nan(a)) return a;
  if (e < 0x3ff) return a & SIGN;
  ulong mask = (1ul << (0x3ff + 52 - e)) - 1;
  return a & ~mask;
}
inline ulong floor_(ulong a) { ulong t = trunc_(a); if (t != a && (a & SIGN) && !is_nan(a)) return add(t, 0x3ff0000000000000ul, true); return t; }
inline ulong ceil_(ulong a) { ulong t = trunc_(a); if (t != a && !(a & SIGN) && !is_nan(a)) return add(t, 0x3ff0000000000000ul, false); return t; }
inline ulong rint_(ulong a) {
  int e = exp_of(a);
  if (e >= 0x3ff + 52 || is_nan(a)) return a;
  // add and subtract 2^52 with the operand's sign: exact ties-to-even rounding in binary64 arithmetic
  ulong big = 0x4330000000000000ul | (a & SIGN);
  ulong r = add(add(a, big, false), big, true);
  return (r & ~SIGN) == 0 ? (a & SIGN) : r;
}
inline ulong round_(ulong a) {  // half away from zero
  int e = exp_of(a);
  if (e >= 0x3ff + 52 || is_nan(a)) return a;
  ulong h = 0x3fe0000000000000ul | (a & SIGN);
  return trunc_(add(a, h, false));
}
}  // namespace mvcc_f64d
inline mvcc_f64 mvcc_f64_add(mvcc_f64 a, mvcc_f64 b) { return mvcc_f64{mvcc_f64d::add(a.b, b.b, false)}; }
inline mvcc_f64 mvcc_f64_sub(mvcc_f64 a, mvcc_f64 b) { return mvcc_f64{mvcc_f64d::add(a.b, b.b, true)}; }
inline mvcc_f64 mvcc_f64_mul(mvcc_f64 a, mvcc_f64 b) { return mvcc_f64{mvcc_f64d::mul(a.b, b.b)}; }
inline mvcc_f64 mvcc_f64_div(mvcc_f64 a, mvcc_f64 b) { return mvcc_f64{mvcc_f64d::div(a.b, b.b)}; }
inline mvcc_f64 mvcc_f64_sqrt(mvcc_f64 a) { return mvcc_f64{mvcc_f64d::sqrt_(a.b)}; }
inline mvcc_f64 mvcc_f64_neg(mvcc_f64 a) { return mvcc_f64{a.b ^ mvcc_f64d::SIGN}; }
inline mvcc_f64 mvcc_f64_abs(mvcc_f64 a) { return mvcc_f64{a.b & ~mvcc_f64d::SIGN}; }
inline mvcc_f64 mvcc_f64_copysign(mvcc_f64 a, mvcc_f64 s) { return mvcc_f64{(a.b & ~mvcc_f64d::SIGN) | (s.b & mvcc_f64d::SIGN)}; }
inline mvcc_f64 mvcc_f64_floor(mvcc_f64 a) { return mvcc_f64{mvcc_f64d::floor_(a.b)}; }
inline mvcc_f64 mvcc_f64_ceil(mvcc_f64 a) { return mvcc_f64{mvcc_f64d::ceil_(a.b)}; }
inline mvcc_f64 mvcc_f64_trunc(mvcc_f64 a) { return mvcc_f64{mvcc_f64d::trunc_(a.b)}; }
inline mvcc_f64 mvcc_f64_rint(mvcc_f64 a) { return mvcc_f64{mvcc_f64d::rint_(a.b)}; }
inline mvcc_f64 mvcc_f64_round(mvcc_f64 a) { return mvcc_f64{mvcc_f64d::round_(a.b)}; }
inline mvcc_f64 mvcc_f64_fmin(mvcc_f64 a, mvcc_f64 b) { if (mvcc_f64d::is_nan(a.b)) return b; if (mvcc_f64d::is_nan(b.b)) return a; return mvcc_f64d::lt(b.b, a.b) ? b : a; }
inline mvcc_f64 mvcc_f64_fmax(mvcc_f64 a, mvcc_f64 b) { if (mvcc_f64d::is_nan(a.b)) return b; if (mvcc_f64d::is_nan(b.b)) return a; return mvcc_f64d::lt(a.b, b.b) ? b : a; }
inline bool mvcc_f64_lt(mvcc_f64 a, mvcc_f64 b) { return mvcc_f64d::lt(a.b, b.b); }
inline bool mvcc_f64_le(mvcc_f64 a, mvcc_f64 b) { return mvcc_f64d::le(a.b, b.b); }
inline bool mvcc_f64_eq(mvcc_f64 a, mvcc_f64 b) { return mvcc_f64d::eq(a.b, b.b); }
inline bool mvcc_f64_unord(mvcc_f64 a, mvcc_f64 b) { return mvcc_f64d::unord(a.b, b.b); }
inline bool mvcc_f64_isnan(mvcc_f64 a) { return mvcc_f64d::is_nan(a.b); }
inline bool mvcc_f64_isinf(mvcc_f64 a) { return mvcc_f64d::is_inf(a.b); }
inline mvcc_f64 mvcc_f64_from_f32(float f) { return mvcc_f64{mvcc_f64d::from_f32(f)}; }
inline float mvcc_f64_to_f32(mvcc_f64 a) { return mvcc_f64d::to_f32(a.b); }
inline mvcc_f64 mvcc_f64_from_u64(ulong v) { return mvcc_f64{mvcc_f64d::from_u64(v)}; }
inline mvcc_f64 mvcc_f64_from_i64(long v) { return mvcc_f64{mvcc_f64d::from_i64(v)}; }
inline ulong mvcc_f64_to_u64(mvcc_f64 a) { return mvcc_f64d::to_u64(a.b); }
inline long mvcc_f64_to_i64(mvcc_f64 a) { return mvcc_f64d::to_i64(a.b); }
inline uint mvcc_f64_to_u32(mvcc_f64 a) { return mvcc_f64d::to_u32(a.b); }
inline int mvcc_f64_to_i32(mvcc_f64 a) { return mvcc_f64d::to_i32(a.b); }
#endif
