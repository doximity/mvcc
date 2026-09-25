// mvcc: __nv_bfloat16 / __nv_bfloat162 (public cuda_bf16 API).
// Storage is the 16-bit pattern. On the device the float<->bf16 conversions go through clang's __bf16 so they
// lower to the native Metal `bfloat` conversions (round-to-nearest-even); on the host they are explicit bit
// manipulation with the same rounding. Arithmetic is done in float, then rounded, matching the documented
// "convert to float, operate, round" semantics of the sm_80 path.
#ifndef MVCC_CUDA_BF16_H
#define MVCC_CUDA_BF16_H
#include <mvcc/host_defines.h>
#include <cuda_runtime.h>  // vector types, warp shuffles: this header is usable on its own, as in CUDA
#include <cuda_fp16.h>
#include <stdint.h>
#include <string.h>

#define __CUDA_BF16_TYPES_EXIST__
#define MVCC_HD __host__ __device__ __forceinline__

struct __nv_bfloat16_raw { unsigned short x; };
struct __align__(4) __nv_bfloat162_raw { unsigned short x, y; };

namespace __mvcc_bf16 {
#if defined(__CUDA_ARCH__)
// Device: clang's __bf16 lowers to LLVM `bfloat`; mvcc-ir2msl emits the native Metal conversions (one instruction
// each on M-series), so the round-to-nearest-even is done by hardware.
MVCC_HD unsigned short f2bf_rn(float f) { __bf16 b = (__bf16)f; unsigned short r; memcpy(&r, &b, 2); return r; }
MVCC_HD float bf2f(unsigned short b) { __bf16 h; memcpy(&h, &b, 2); return (float)h; }
#else
MVCC_HD unsigned short f2bf_rn(float f) {
  unsigned int u; memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return (unsigned short)((u >> 16) | 0x40);  // NaN: quiet, keep sign
  unsigned int lsb = (u >> 16) & 1u;
  u += 0x7fffu + lsb;
  return (unsigned short)(u >> 16);
}
MVCC_HD float bf2f(unsigned short b) { unsigned int u = (unsigned int)b << 16; float f; memcpy(&f, &u, 4); return f; }
#endif
MVCC_HD unsigned short f2bf_rz(float f) { unsigned int u; memcpy(&u, &f, 4); if ((u & 0x7f800000u) == 0x7f800000u && (u & 0x007fffffu)) return (unsigned short)((u >> 16) | 0x40); return (unsigned short)(u >> 16); }
MVCC_HD unsigned short f2bf_rd(float f) { unsigned int u; memcpy(&u, &f, 4); if ((u & 0x7f800000u) == 0x7f800000u) return f2bf_rz(f); unsigned short t = (unsigned short)(u >> 16); if ((u & 0xffffu) && (u & 0x80000000u)) t++; return t; }
MVCC_HD unsigned short f2bf_ru(float f) { unsigned int u; memcpy(&u, &f, 4); if ((u & 0x7f800000u) == 0x7f800000u) return f2bf_rz(f); unsigned short t = (unsigned short)(u >> 16); if ((u & 0xffffu) && !(u & 0x80000000u)) t++; return t; }
}  // namespace __mvcc_bf16

struct __align__(2) __nv_bfloat16 {
  unsigned short __x;
  __nv_bfloat16() = default;
  MVCC_HD __nv_bfloat16(const __nv_bfloat16_raw& r) : __x(r.x) {}
  MVCC_HD __nv_bfloat16(float f) : __x(__mvcc_bf16::f2bf_rn(f)) {}
  MVCC_HD __nv_bfloat16(double d) : __x(__mvcc_bf16::f2bf_rn((float)d)) {}
  MVCC_HD __nv_bfloat16(int v) : __x(__mvcc_bf16::f2bf_rn((float)v)) {}
  MVCC_HD __nv_bfloat16(unsigned int v) : __x(__mvcc_bf16::f2bf_rn((float)v)) {}
  MVCC_HD __nv_bfloat16(short v) : __x(__mvcc_bf16::f2bf_rn((float)v)) {}
  MVCC_HD __nv_bfloat16(unsigned short v) : __x(__mvcc_bf16::f2bf_rn((float)v)) {}
  MVCC_HD __nv_bfloat16(long long v) : __x(__mvcc_bf16::f2bf_rn((float)v)) {}
  MVCC_HD __nv_bfloat16(unsigned long long v) : __x(__mvcc_bf16::f2bf_rn((float)v)) {}
  MVCC_HD operator float() const { return __mvcc_bf16::bf2f(__x); }
  MVCC_HD operator __nv_bfloat16_raw() const { __nv_bfloat16_raw r; r.x = __x; return r; }
  MVCC_HD explicit operator short() const { return (short)(float)*this; }
  MVCC_HD explicit operator unsigned short() const { return (unsigned short)(float)*this; }
  MVCC_HD explicit operator int() const { return (int)(float)*this; }
  MVCC_HD explicit operator unsigned int() const { return (unsigned int)(float)*this; }
  MVCC_HD explicit operator long long() const { return (long long)(float)*this; }
  MVCC_HD explicit operator unsigned long long() const { return (unsigned long long)(float)*this; }
  MVCC_HD explicit operator bool() const { return (__x & 0x7fff) != 0; }
  MVCC_HD __nv_bfloat16& operator=(const __nv_bfloat16_raw& r) { __x = r.x; return *this; }
  MVCC_HD __nv_bfloat16& operator=(float f) { __x = __mvcc_bf16::f2bf_rn(f); return *this; }
};

struct __align__(4) __nv_bfloat162 {
  __nv_bfloat16 x, y;
  __nv_bfloat162() = default;
  MVCC_HD __nv_bfloat162(const __nv_bfloat16& a, const __nv_bfloat16& b) : x(a), y(b) {}
  MVCC_HD __nv_bfloat162(const __nv_bfloat162_raw& r) { x.__x = r.x; y.__x = r.y; }
  MVCC_HD operator __nv_bfloat162_raw() const { __nv_bfloat162_raw r; r.x = x.__x; r.y = y.__x; return r; }
  MVCC_HD __nv_bfloat162& operator=(const __nv_bfloat162_raw& r) { x.__x = r.x; y.__x = r.y; return *this; }
};
typedef __nv_bfloat16 nv_bfloat16;
typedef __nv_bfloat162 nv_bfloat162;

// --- conversions ---
MVCC_HD __nv_bfloat16 __float2bfloat16(float f) { return __nv_bfloat16(f); }
MVCC_HD __nv_bfloat16 __float2bfloat16_rn(float f) { return __nv_bfloat16(f); }
MVCC_HD __nv_bfloat16 __float2bfloat16_rz(float f) { __nv_bfloat16_raw r; r.x = __mvcc_bf16::f2bf_rz(f); return __nv_bfloat16(r); }
MVCC_HD __nv_bfloat16 __float2bfloat16_rd(float f) { __nv_bfloat16_raw r; r.x = __mvcc_bf16::f2bf_rd(f); return __nv_bfloat16(r); }
MVCC_HD __nv_bfloat16 __float2bfloat16_ru(float f) { __nv_bfloat16_raw r; r.x = __mvcc_bf16::f2bf_ru(f); return __nv_bfloat16(r); }
MVCC_HD __nv_bfloat16 __double2bfloat16(double d) { return __nv_bfloat16((float)d); }
MVCC_HD float __bfloat162float(const __nv_bfloat16 b) { return (float)b; }
MVCC_HD __nv_bfloat16 __int2bfloat16_rn(int v) { return __nv_bfloat16((float)v); }
MVCC_HD __nv_bfloat16 __uint2bfloat16_rn(unsigned int v) { return __nv_bfloat16((float)v); }
MVCC_HD __nv_bfloat16 __short2bfloat16_rn(short v) { return __nv_bfloat16((float)v); }
MVCC_HD __nv_bfloat16 __ushort2bfloat16_rn(unsigned short v) { return __nv_bfloat16((float)v); }
MVCC_HD int __bfloat162int_rn(const __nv_bfloat16 b) { float f = (float)b; return (int)(f + (f >= 0 ? 0.5f : -0.5f)); }
MVCC_HD int __bfloat162int_rz(const __nv_bfloat16 b) { return (int)(float)b; }
MVCC_HD unsigned int __bfloat162uint_rz(const __nv_bfloat16 b) { return (unsigned int)(float)b; }
MVCC_HD short __bfloat162short_rz(const __nv_bfloat16 b) { return (short)(float)b; }
MVCC_HD unsigned short __bfloat162ushort_rz(const __nv_bfloat16 b) { return (unsigned short)(float)b; }
MVCC_HD short __bfloat16_as_short(const __nv_bfloat16 b) { return (short)b.__x; }
MVCC_HD unsigned short __bfloat16_as_ushort(const __nv_bfloat16 b) { return b.__x; }
MVCC_HD __nv_bfloat16 __short_as_bfloat16(short v) { __nv_bfloat16_raw r; r.x = (unsigned short)v; return __nv_bfloat16(r); }
MVCC_HD __nv_bfloat16 __ushort_as_bfloat16(unsigned short v) { __nv_bfloat16_raw r; r.x = v; return __nv_bfloat16(r); }
MVCC_HD __nv_bfloat16 __half2bfloat16(const __half h) { return __nv_bfloat16((float)h); }
MVCC_HD __half __bfloat162half(const __nv_bfloat16 b) { return __half((float)b); }

MVCC_HD __nv_bfloat162 __floats2bfloat162_rn(float a, float b) { return __nv_bfloat162(__nv_bfloat16(a), __nv_bfloat16(b)); }
MVCC_HD __nv_bfloat162 __float2bfloat162_rn(float a) { return __nv_bfloat162(__nv_bfloat16(a), __nv_bfloat16(a)); }
MVCC_HD __nv_bfloat162 __float22bfloat162_rn(float2 f) { return __nv_bfloat162(__nv_bfloat16(f.x), __nv_bfloat16(f.y)); }
MVCC_HD float2 __bfloat1622float2(const __nv_bfloat162 b) { return make_float2((float)b.x, (float)b.y); }
MVCC_HD __nv_bfloat162 __halves2bfloat162(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __nv_bfloat162(a, b); }
MVCC_HD __nv_bfloat162 __bfloat162bfloat162(const __nv_bfloat16 a) { return __nv_bfloat162(a, a); }
MVCC_HD __nv_bfloat16 __low2bfloat16(const __nv_bfloat162 b) { return b.x; }
MVCC_HD __nv_bfloat16 __high2bfloat16(const __nv_bfloat162 b) { return b.y; }
MVCC_HD float __low2float(const __nv_bfloat162 b) { return (float)b.x; }
MVCC_HD float __high2float(const __nv_bfloat162 b) { return (float)b.y; }
MVCC_HD __nv_bfloat162 __low2bfloat162(const __nv_bfloat162 b) { return __nv_bfloat162(b.x, b.x); }
MVCC_HD __nv_bfloat162 __high2bfloat162(const __nv_bfloat162 b) { return __nv_bfloat162(b.y, b.y); }
MVCC_HD __nv_bfloat162 __lows2bfloat162(const __nv_bfloat162 a, const __nv_bfloat162 b) { return __nv_bfloat162(a.x, b.x); }
MVCC_HD __nv_bfloat162 __highs2bfloat162(const __nv_bfloat162 a, const __nv_bfloat162 b) { return __nv_bfloat162(a.y, b.y); }
MVCC_HD __nv_bfloat162 __lowhigh2highlow(const __nv_bfloat162 b) { return __nv_bfloat162(b.y, b.x); }
MVCC_HD unsigned int __bfloat162_as_uint(const __nv_bfloat162 b) { return (unsigned int)b.x.__x | ((unsigned int)b.y.__x << 16); }
MVCC_HD __nv_bfloat162 __uint_as_bfloat162(unsigned int u) { __nv_bfloat162_raw r; r.x = (unsigned short)u; r.y = (unsigned short)(u >> 16); return __nv_bfloat162(r); }

// --- arithmetic ---
MVCC_HD __nv_bfloat16 __hadd(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __nv_bfloat16((float)a + (float)b); }
MVCC_HD __nv_bfloat16 __hsub(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __nv_bfloat16((float)a - (float)b); }
MVCC_HD __nv_bfloat16 __hmul(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __nv_bfloat16((float)a * (float)b); }
MVCC_HD __nv_bfloat16 __hdiv(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __nv_bfloat16((float)a / (float)b); }
MVCC_HD __nv_bfloat16 __hfma(const __nv_bfloat16 a, const __nv_bfloat16 b, const __nv_bfloat16 c) { return __nv_bfloat16(__builtin_fmaf((float)a, (float)b, (float)c)); }
MVCC_HD __nv_bfloat16 __hneg(const __nv_bfloat16 a) { __nv_bfloat16_raw r; r.x = a.__x ^ 0x8000; return __nv_bfloat16(r); }
MVCC_HD __nv_bfloat16 __habs(const __nv_bfloat16 a) { __nv_bfloat16_raw r; r.x = a.__x & 0x7fff; return __nv_bfloat16(r); }
MVCC_HD __nv_bfloat16 __hmax(const __nv_bfloat16 a, const __nv_bfloat16 b) { float fa = (float)a, fb = (float)b; if (fa != fa) return b; if (fb != fb) return a; return fa > fb ? a : b; }
MVCC_HD __nv_bfloat16 __hmin(const __nv_bfloat16 a, const __nv_bfloat16 b) { float fa = (float)a, fb = (float)b; if (fa != fa) return b; if (fb != fb) return a; return fa < fb ? a : b; }
MVCC_HD __nv_bfloat16 __hmax_nan(const __nv_bfloat16 a, const __nv_bfloat16 b) { float fa = (float)a, fb = (float)b; if (fa != fa) return a; if (fb != fb) return b; return fa > fb ? a : b; }
MVCC_HD __nv_bfloat16 __hmin_nan(const __nv_bfloat16 a, const __nv_bfloat16 b) { float fa = (float)a, fb = (float)b; if (fa != fa) return a; if (fb != fb) return b; return fa < fb ? a : b; }
MVCC_HD __nv_bfloat16 __hadd_sat(const __nv_bfloat16 a, const __nv_bfloat16 b) { float f = (float)a + (float)b; return __nv_bfloat16(f < 0.f ? 0.f : f > 1.f ? 1.f : f); }
MVCC_HD __nv_bfloat16 __hmul_sat(const __nv_bfloat16 a, const __nv_bfloat16 b) { float f = (float)a * (float)b; return __nv_bfloat16(f < 0.f ? 0.f : f > 1.f ? 1.f : f); }
MVCC_HD __nv_bfloat16 __hadd_rn(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __hadd(a, b); }
MVCC_HD __nv_bfloat16 __hmul_rn(const __nv_bfloat16 a, const __nv_bfloat16 b) { return __hmul(a, b); }
MVCC_HD __nv_bfloat162 __hadd2(const __nv_bfloat162 a, const __nv_bfloat162 b) { return __nv_bfloat162(__hadd(a.x, b.x), __hadd(a.y, b.y)); }
MVCC_HD __nv_bfloat162 __hsub2(const __nv_bfloat162 a, const __nv_bfloat162 b) { return __nv_bfloat162(__hsub(a.x, b.x), __hsub(a.y, b.y)); }
MVCC_HD __nv_bfloat162 __hmul2(const __nv_bfloat162 a, const __nv_bfloat162 b) { return __nv_bfloat162(__hmul(a.x, b.x), __hmul(a.y, b.y)); }
MVCC_HD __nv_bfloat162 __h2div(const __nv_bfloat162 a, const __nv_bfloat162 b) { return __nv_bfloat162(__hdiv(a.x, b.x), __hdiv(a.y, b.y)); }
MVCC_HD __nv_bfloat162 __hfma2(const __nv_bfloat162 a, const __nv_bfloat162 b, const __nv_bfloat162 c) { return __nv_bfloat162(__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y)); }
MVCC_HD __nv_bfloat162 __hneg2(const __nv_bfloat162 a) { return __nv_bfloat162(__hneg(a.x), __hneg(a.y)); }
MVCC_HD __nv_bfloat162 __habs2(const __nv_bfloat162 a) { return __nv_bfloat162(__habs(a.x), __habs(a.y)); }
MVCC_HD __nv_bfloat162 __hmax2(const __nv_bfloat162 a, const __nv_bfloat162 b) { return __nv_bfloat162(__hmax(a.x, b.x), __hmax(a.y, b.y)); }
MVCC_HD __nv_bfloat162 __hmin2(const __nv_bfloat162 a, const __nv_bfloat162 b) { return __nv_bfloat162(__hmin(a.x, b.x), __hmin(a.y, b.y)); }

// --- comparisons ---
MVCC_HD bool __heq(const __nv_bfloat16 a, const __nv_bfloat16 b) { return (float)a == (float)b; }
MVCC_HD bool __hne(const __nv_bfloat16 a, const __nv_bfloat16 b) { float fa = (float)a, fb = (float)b; return fa != fb && fa == fa && fb == fb; }
MVCC_HD bool __hlt(const __nv_bfloat16 a, const __nv_bfloat16 b) { return (float)a < (float)b; }
MVCC_HD bool __hle(const __nv_bfloat16 a, const __nv_bfloat16 b) { return (float)a <= (float)b; }
MVCC_HD bool __hgt(const __nv_bfloat16 a, const __nv_bfloat16 b) { return (float)a > (float)b; }
MVCC_HD bool __hge(const __nv_bfloat16 a, const __nv_bfloat16 b) { return (float)a >= (float)b; }
MVCC_HD bool __hneu(const __nv_bfloat16 a, const __nv_bfloat16 b) { return !((float)a == (float)b); }
MVCC_HD bool __hisnan(const __nv_bfloat16 a) { return (a.__x & 0x7fff) > 0x7f80; }
MVCC_HD bool __hisinf(const __nv_bfloat16 a) { return (a.__x & 0x7fff) == 0x7f80; }
MVCC_HD bool __hbeq2(const __nv_bfloat162 a, const __nv_bfloat162 b) { return __heq(a.x, b.x) && __heq(a.y, b.y); }

// --- math ---
MVCC_HD __nv_bfloat16 hsqrt(const __nv_bfloat16 a) { return __nv_bfloat16(__builtin_sqrtf((float)a)); }
MVCC_HD __nv_bfloat16 hrsqrt(const __nv_bfloat16 a) { return __nv_bfloat16(1.0f / __builtin_sqrtf((float)a)); }
MVCC_HD __nv_bfloat16 hrcp(const __nv_bfloat16 a) { return __nv_bfloat16(1.0f / (float)a); }
MVCC_HD __nv_bfloat16 hexp(const __nv_bfloat16 a) { return __nv_bfloat16(__builtin_expf((float)a)); }
MVCC_HD __nv_bfloat16 hexp2(const __nv_bfloat16 a) { return __nv_bfloat16(__builtin_exp2f((float)a)); }
MVCC_HD __nv_bfloat16 hlog(const __nv_bfloat16 a) { return __nv_bfloat16(__builtin_logf((float)a)); }
MVCC_HD __nv_bfloat16 hlog2(const __nv_bfloat16 a) { return __nv_bfloat16(__builtin_log2f((float)a)); }
MVCC_HD __nv_bfloat16 hsin(const __nv_bfloat16 a) { return __nv_bfloat16(__builtin_sinf((float)a)); }
MVCC_HD __nv_bfloat16 hcos(const __nv_bfloat16 a) { return __nv_bfloat16(__builtin_cosf((float)a)); }
MVCC_HD __nv_bfloat16 hfloor(const __nv_bfloat16 a) { return __nv_bfloat16(__builtin_floorf((float)a)); }
MVCC_HD __nv_bfloat16 hceil(const __nv_bfloat16 a) { return __nv_bfloat16(__builtin_ceilf((float)a)); }
MVCC_HD __nv_bfloat16 htrunc(const __nv_bfloat16 a) { return __nv_bfloat16(__builtin_truncf((float)a)); }
MVCC_HD __nv_bfloat16 hrint(const __nv_bfloat16 a) { return __nv_bfloat16(__builtin_rintf((float)a)); }

// --- operators ---
MVCC_HD __nv_bfloat16 operator+(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hadd(a, b); }
MVCC_HD __nv_bfloat16 operator-(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hsub(a, b); }
MVCC_HD __nv_bfloat16 operator*(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hmul(a, b); }
MVCC_HD __nv_bfloat16 operator/(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hdiv(a, b); }
MVCC_HD __nv_bfloat16 operator-(const __nv_bfloat16& a) { return __hneg(a); }
MVCC_HD __nv_bfloat16 operator+(const __nv_bfloat16& a) { return a; }
MVCC_HD __nv_bfloat16& operator+=(__nv_bfloat16& a, const __nv_bfloat16& b) { a = a + b; return a; }
MVCC_HD __nv_bfloat16& operator-=(__nv_bfloat16& a, const __nv_bfloat16& b) { a = a - b; return a; }
MVCC_HD __nv_bfloat16& operator*=(__nv_bfloat16& a, const __nv_bfloat16& b) { a = a * b; return a; }
MVCC_HD __nv_bfloat16& operator/=(__nv_bfloat16& a, const __nv_bfloat16& b) { a = a / b; return a; }
MVCC_HD bool operator==(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __heq(a, b); }
MVCC_HD bool operator!=(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hneu(a, b); }
MVCC_HD bool operator<(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hlt(a, b); }
MVCC_HD bool operator<=(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hle(a, b); }
MVCC_HD bool operator>(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hgt(a, b); }
MVCC_HD bool operator>=(const __nv_bfloat16& a, const __nv_bfloat16& b) { return __hge(a, b); }
MVCC_HD __nv_bfloat162 operator+(const __nv_bfloat162& a, const __nv_bfloat162& b) { return __hadd2(a, b); }
MVCC_HD __nv_bfloat162 operator-(const __nv_bfloat162& a, const __nv_bfloat162& b) { return __hsub2(a, b); }
MVCC_HD __nv_bfloat162 operator*(const __nv_bfloat162& a, const __nv_bfloat162& b) { return __hmul2(a, b); }
MVCC_HD __nv_bfloat162 operator/(const __nv_bfloat162& a, const __nv_bfloat162& b) { return __h2div(a, b); }
MVCC_HD __nv_bfloat162 operator-(const __nv_bfloat162& a) { return __hneg2(a); }
MVCC_HD __nv_bfloat162& operator+=(__nv_bfloat162& a, const __nv_bfloat162& b) { a = a + b; return a; }
MVCC_HD __nv_bfloat162& operator-=(__nv_bfloat162& a, const __nv_bfloat162& b) { a = a - b; return a; }
MVCC_HD __nv_bfloat162& operator*=(__nv_bfloat162& a, const __nv_bfloat162& b) { a = a * b; return a; }
MVCC_HD bool operator==(const __nv_bfloat162& a, const __nv_bfloat162& b) { return __hbeq2(a, b); }
MVCC_HD bool operator!=(const __nv_bfloat162& a, const __nv_bfloat162& b) { return !__hbeq2(a, b); }

MVCC_HD __nv_bfloat16 __ldg(const __nv_bfloat16* p) { return *p; }
MVCC_HD __nv_bfloat162 __ldg(const __nv_bfloat162* p) { return *p; }
// __ldcg reads at L2 (a value another SM stored): a volatile load, which mvcc-ir2msl performs coherently
MVCC_HD __nv_bfloat16 __ldcg(const __nv_bfloat16* p) { __nv_bfloat16 r; *reinterpret_cast<unsigned short*>(&r) = *reinterpret_cast<const volatile unsigned short*>(p); return r; }
MVCC_HD __nv_bfloat162 __ldcg(const __nv_bfloat162* p) { __nv_bfloat162 r; *reinterpret_cast<unsigned int*>(&r) = *reinterpret_cast<const volatile unsigned int*>(p); return r; }
MVCC_HD __nv_bfloat16 __ldcs(const __nv_bfloat16* p) { return *p; }
MVCC_HD __nv_bfloat162 __ldcs(const __nv_bfloat162* p) { return *p; }
MVCC_HD void __stcg(__nv_bfloat16* p, __nv_bfloat16 v) { *p = v; }
MVCC_HD void __stcs(__nv_bfloat16* p, __nv_bfloat16 v) { *p = v; }
MVCC_HD void __stcs(__nv_bfloat162* p, __nv_bfloat162 v) { *p = v; }

#if defined(__CUDA__) && defined(__clang__)
static __device__ __forceinline__ __nv_bfloat162 __shfl_sync(unsigned mask, __nv_bfloat162 v, int src, int width = 32) { return __uint_as_bfloat162((unsigned)__shfl_sync(mask, (int)__bfloat162_as_uint(v), src, width)); }
static __device__ __forceinline__ __nv_bfloat162 __shfl_xor_sync(unsigned mask, __nv_bfloat162 v, int m, int width = 32) { return __uint_as_bfloat162((unsigned)__shfl_xor_sync(mask, (int)__bfloat162_as_uint(v), m, width)); }
static __device__ __forceinline__ __nv_bfloat162 __shfl_down_sync(unsigned mask, __nv_bfloat162 v, unsigned d, int width = 32) { return __uint_as_bfloat162((unsigned)__shfl_down_sync(mask, (int)__bfloat162_as_uint(v), d, width)); }
static __device__ __forceinline__ __nv_bfloat16 __shfl_sync(unsigned mask, __nv_bfloat16 v, int src, int width = 32) { return __ushort_as_bfloat16((unsigned short)__shfl_sync(mask, (int)v.__x, src, width)); }
static __device__ __forceinline__ __nv_bfloat16 __shfl_xor_sync(unsigned mask, __nv_bfloat16 v, int m, int width = 32) { return __ushort_as_bfloat16((unsigned short)__shfl_xor_sync(mask, (int)v.__x, m, width)); }
static __device__ __forceinline__ __nv_bfloat16 __shfl_down_sync(unsigned mask, __nv_bfloat16 v, unsigned d, int width = 32) { return __ushort_as_bfloat16((unsigned short)__shfl_down_sync(mask, (int)v.__x, d, width)); }
#endif

#undef MVCC_HD
#endif // MVCC_CUDA_BF16_H
