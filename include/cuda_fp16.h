// mvcc: __half / __half2 (public cuda_fp16 API), built on clang's _Float16.
// _Float16 is native on arm64 hosts and lowers to LLVM `half` on the device, so the same code
// serves both passes and the emitter sees ordinary half arithmetic.
#ifndef MVCC_CUDA_FP16_H
#define MVCC_CUDA_FP16_H
#include <mvcc/host_defines.h>
#include <cuda_runtime.h>  // vector types, warp shuffles: this header is usable on its own, as in CUDA
#include <stdint.h>
#include <string.h>

#define __CUDA_FP16_TYPES_EXIST__
#define MVCC_HD __host__ __device__ __forceinline__

typedef _Float16 __mvcc_f16;

struct __half_raw { unsigned short x; };
struct __align__(4) __half2_raw { unsigned short x, y; };

struct __align__(2) __half {
  __mvcc_f16 __x;
  __half() = default;
  MVCC_HD __half(const __half_raw& r) { unsigned short v = r.x; memcpy(&__x, &v, 2); }
  MVCC_HD __half(float f) : __x((__mvcc_f16)f) {}
  MVCC_HD __half(double d) : __x((__mvcc_f16)(float)d) {}
  MVCC_HD __half(int v) : __x((__mvcc_f16)(float)v) {}
  MVCC_HD __half(unsigned int v) : __x((__mvcc_f16)(float)v) {}
  MVCC_HD __half(short v) : __x((__mvcc_f16)(float)v) {}
  MVCC_HD __half(unsigned short v) : __x((__mvcc_f16)(float)v) {}
  MVCC_HD __half(long long v) : __x((__mvcc_f16)(float)v) {}
  MVCC_HD __half(unsigned long long v) : __x((__mvcc_f16)(float)v) {}
  MVCC_HD operator float() const { return (float)__x; }
  MVCC_HD operator __half_raw() const { __half_raw r; memcpy(&r.x, &__x, 2); return r; }
  MVCC_HD explicit operator short() const { return (short)(float)__x; }
  MVCC_HD explicit operator unsigned short() const { return (unsigned short)(float)__x; }
  MVCC_HD explicit operator int() const { return (int)(float)__x; }
  MVCC_HD explicit operator unsigned int() const { return (unsigned int)(float)__x; }
  MVCC_HD explicit operator long long() const { return (long long)(float)__x; }
  MVCC_HD explicit operator unsigned long long() const { return (unsigned long long)(float)__x; }
  MVCC_HD explicit operator bool() const { return (float)__x != 0.0f; }
  MVCC_HD __half& operator=(const __half_raw& r) { unsigned short v = r.x; memcpy(&__x, &v, 2); return *this; }
  MVCC_HD __half& operator=(float f) { __x = (__mvcc_f16)f; return *this; }
};

struct __align__(4) __half2 {
  __half x, y;
  __half2() = default;
  MVCC_HD __half2(const __half& a, const __half& b) : x(a), y(b) {}
  MVCC_HD __half2(const __half2_raw& r) { x = __half_raw{r.x}; y = __half_raw{r.y}; }
  MVCC_HD operator __half2_raw() const { __half2_raw r; r.x = ((__half_raw)x).x; r.y = ((__half_raw)y).x; return r; }
  MVCC_HD __half2& operator=(const __half2_raw& r) { x = __half_raw{r.x}; y = __half_raw{r.y}; return *this; }
};
typedef __half half;
typedef __half2 half2;

// --- conversions ---
MVCC_HD __half __float2half(float f) { return __half(f); }
MVCC_HD __half __float2half_rn(float f) { return __half(f); }
MVCC_HD __half __float2half_rz(float f) {
  // truncate: convert, then step toward zero if rounding went away from zero
  __half h(f); float b = (float)h;
  if ((b > f && f >= 0.0f) || (b < f && f <= 0.0f)) { __half_raw r = h; r.x -= 1; h = r; }
  return h;
}
MVCC_HD __half __float2half_rd(float f) { __half h(f); float b = (float)h; if (b > f) { __half_raw r = h; r.x += (f < 0.0f) ? 1 : -1; h = r; } return h; }
MVCC_HD __half __float2half_ru(float f) { __half h(f); float b = (float)h; if (b < f) { __half_raw r = h; r.x += (f < 0.0f) ? -1 : 1; h = r; } return h; }
MVCC_HD __half __double2half(double d) { return __half((float)d); }
MVCC_HD float __half2float(const __half h) { return (float)h; }
MVCC_HD __half __int2half_rn(int v) { return __half((float)v); }
MVCC_HD __half __uint2half_rn(unsigned int v) { return __half((float)v); }
MVCC_HD __half __short2half_rn(short v) { return __half((float)v); }
MVCC_HD __half __ushort2half_rn(unsigned short v) { return __half((float)v); }
MVCC_HD __half __ll2half_rn(long long v) { return __half((float)v); }
MVCC_HD __half __ull2half_rn(unsigned long long v) { return __half((float)v); }
MVCC_HD int __half2int_rn(const __half h) { float f = (float)h; return (int)(f + (f >= 0 ? 0.5f : -0.5f)); }
MVCC_HD int __half2int_rz(const __half h) { return (int)(float)h; }
MVCC_HD int __half2int_rd(const __half h) { float f = (float)h; int i = (int)f; return (f < (float)i) ? i - 1 : i; }
MVCC_HD int __half2int_ru(const __half h) { float f = (float)h; int i = (int)f; return (f > (float)i) ? i + 1 : i; }
MVCC_HD unsigned int __half2uint_rn(const __half h) { return (unsigned int)((float)h + 0.5f); }
MVCC_HD unsigned int __half2uint_rz(const __half h) { return (unsigned int)(float)h; }
MVCC_HD short __half2short_rn(const __half h) { return (short)__half2int_rn(h); }
MVCC_HD short __half2short_rz(const __half h) { return (short)(float)h; }
MVCC_HD unsigned short __half2ushort_rn(const __half h) { return (unsigned short)((float)h + 0.5f); }
MVCC_HD unsigned short __half2ushort_rz(const __half h) { return (unsigned short)(float)h; }
MVCC_HD long long __half2ll_rz(const __half h) { return (long long)(float)h; }
MVCC_HD unsigned long long __half2ull_rz(const __half h) { return (unsigned long long)(float)h; }
MVCC_HD short __half_as_short(const __half h) { __half_raw r = h; return (short)r.x; }
MVCC_HD unsigned short __half_as_ushort(const __half h) { __half_raw r = h; return r.x; }
MVCC_HD __half __short_as_half(short v) { __half_raw r; r.x = (unsigned short)v; return __half(r); }
MVCC_HD __half __ushort_as_half(unsigned short v) { __half_raw r; r.x = v; return __half(r); }

MVCC_HD __half2 __floats2half2_rn(float a, float b) { return __half2(__half(a), __half(b)); }
MVCC_HD __half2 __float2half2_rn(float a) { return __half2(__half(a), __half(a)); }
MVCC_HD __half2 __float22half2_rn(float2 f) { return __half2(__half(f.x), __half(f.y)); }
MVCC_HD float2 __half22float2(const __half2 h) { return make_float2((float)h.x, (float)h.y); }
MVCC_HD __half2 __halves2half2(const __half a, const __half b) { return __half2(a, b); }
MVCC_HD __half2 __half2half2(const __half a) { return __half2(a, a); }
MVCC_HD __half __low2half(const __half2 h) { return h.x; }
MVCC_HD __half __high2half(const __half2 h) { return h.y; }
MVCC_HD float __low2float(const __half2 h) { return (float)h.x; }
MVCC_HD float __high2float(const __half2 h) { return (float)h.y; }
MVCC_HD __half2 __low2half2(const __half2 h) { return __half2(h.x, h.x); }
MVCC_HD __half2 __high2half2(const __half2 h) { return __half2(h.y, h.y); }
MVCC_HD __half2 __lows2half2(const __half2 a, const __half2 b) { return __half2(a.x, b.x); }
MVCC_HD __half2 __highs2half2(const __half2 a, const __half2 b) { return __half2(a.y, b.y); }
MVCC_HD __half2 __lowhigh2highlow(const __half2 h) { return __half2(h.y, h.x); }
MVCC_HD int __half2_as_int(const __half2 h) { int i; memcpy(&i, &h, 4); return i; }
MVCC_HD unsigned int __half2_as_uint(const __half2 h) { unsigned int i; memcpy(&i, &h, 4); return i; }
MVCC_HD __half2 __int_as_half2(int i) { __half2 h; memcpy(&h, &i, 4); return h; }
MVCC_HD __half2 __uint_as_half2(unsigned int i) { __half2 h; memcpy(&h, &i, 4); return h; }

// --- arithmetic (half precision, round-to-nearest) ---
MVCC_HD __half __hadd(const __half a, const __half b) { __half r; r.__x = a.__x + b.__x; return r; }
MVCC_HD __half __hsub(const __half a, const __half b) { __half r; r.__x = a.__x - b.__x; return r; }
MVCC_HD __half __hmul(const __half a, const __half b) { __half r; r.__x = a.__x * b.__x; return r; }
MVCC_HD __half __hdiv(const __half a, const __half b) { __half r; r.__x = a.__x / b.__x; return r; }
MVCC_HD __half __hfma(const __half a, const __half b, const __half c) { return __half(__builtin_fmaf((float)a, (float)b, (float)c)); }
MVCC_HD __half __hneg(const __half a) { __half r; r.__x = -a.__x; return r; }
MVCC_HD __half __habs(const __half a) { __half_raw r = a; r.x &= 0x7fff; return __half(r); }
MVCC_HD __half __hmax(const __half a, const __half b) { float fa = (float)a, fb = (float)b; if (fa != fa) return b; if (fb != fb) return a; return fa > fb ? a : b; }
MVCC_HD __half __hmin(const __half a, const __half b) { float fa = (float)a, fb = (float)b; if (fa != fa) return b; if (fb != fb) return a; return fa < fb ? a : b; }
MVCC_HD __half __hmax_nan(const __half a, const __half b) { float fa = (float)a, fb = (float)b; if (fa != fa) return a; if (fb != fb) return b; return fa > fb ? a : b; }
MVCC_HD __half __hmin_nan(const __half a, const __half b) { float fa = (float)a, fb = (float)b; if (fa != fa) return a; if (fb != fb) return b; return fa < fb ? a : b; }
MVCC_HD __half __hadd_sat(const __half a, const __half b) { float f = (float)a + (float)b; return __half(f < 0.f ? 0.f : f > 1.f ? 1.f : f); }
MVCC_HD __half __hsub_sat(const __half a, const __half b) { float f = (float)a - (float)b; return __half(f < 0.f ? 0.f : f > 1.f ? 1.f : f); }
MVCC_HD __half __hmul_sat(const __half a, const __half b) { float f = (float)a * (float)b; return __half(f < 0.f ? 0.f : f > 1.f ? 1.f : f); }
MVCC_HD __half __hfma_sat(const __half a, const __half b, const __half c) { float f = __builtin_fmaf((float)a, (float)b, (float)c); return __half(f < 0.f ? 0.f : f > 1.f ? 1.f : f); }
MVCC_HD __half __hadd_rn(const __half a, const __half b) { return __hadd(a, b); }
MVCC_HD __half __hsub_rn(const __half a, const __half b) { return __hsub(a, b); }
MVCC_HD __half __hmul_rn(const __half a, const __half b) { return __hmul(a, b); }

MVCC_HD __half2 __hadd2(const __half2 a, const __half2 b) { return __half2(__hadd(a.x, b.x), __hadd(a.y, b.y)); }
MVCC_HD __half2 __hsub2(const __half2 a, const __half2 b) { return __half2(__hsub(a.x, b.x), __hsub(a.y, b.y)); }
MVCC_HD __half2 __hmul2(const __half2 a, const __half2 b) { return __half2(__hmul(a.x, b.x), __hmul(a.y, b.y)); }
MVCC_HD __half2 __h2div(const __half2 a, const __half2 b) { return __half2(__hdiv(a.x, b.x), __hdiv(a.y, b.y)); }
MVCC_HD __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c) { return __half2(__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y)); }
MVCC_HD __half2 __hneg2(const __half2 a) { return __half2(__hneg(a.x), __hneg(a.y)); }
MVCC_HD __half2 __habs2(const __half2 a) { return __half2(__habs(a.x), __habs(a.y)); }
MVCC_HD __half2 __hmax2(const __half2 a, const __half2 b) { return __half2(__hmax(a.x, b.x), __hmax(a.y, b.y)); }
MVCC_HD __half2 __hmin2(const __half2 a, const __half2 b) { return __half2(__hmin(a.x, b.x), __hmin(a.y, b.y)); }
MVCC_HD __half2 __hadd2_sat(const __half2 a, const __half2 b) { return __half2(__hadd_sat(a.x, b.x), __hadd_sat(a.y, b.y)); }
MVCC_HD __half2 __hmul2_sat(const __half2 a, const __half2 b) { return __half2(__hmul_sat(a.x, b.x), __hmul_sat(a.y, b.y)); }
MVCC_HD __half2 __hfma2_sat(const __half2 a, const __half2 b, const __half2 c) { return __half2(__hfma_sat(a.x, b.x, c.x), __hfma_sat(a.y, b.y, c.y)); }
MVCC_HD __half2 __hadd2_rn(const __half2 a, const __half2 b) { return __hadd2(a, b); }
MVCC_HD __half2 __hmul2_rn(const __half2 a, const __half2 b) { return __hmul2(a, b); }
MVCC_HD __half2 __hcmadd(const __half2 a, const __half2 b, const __half2 c) {
  float ar = (float)a.x, ai = (float)a.y, br = (float)b.x, bi = (float)b.y;
  return __half2(__half(ar * br - ai * bi + (float)c.x), __half(ar * bi + ai * br + (float)c.y));
}

// --- comparisons ---
MVCC_HD bool __heq(const __half a, const __half b) { return (float)a == (float)b; }
MVCC_HD bool __hne(const __half a, const __half b) { return (float)a != (float)b && (float)a == (float)a && (float)b == (float)b; }
MVCC_HD bool __hlt(const __half a, const __half b) { return (float)a < (float)b; }
MVCC_HD bool __hle(const __half a, const __half b) { return (float)a <= (float)b; }
MVCC_HD bool __hgt(const __half a, const __half b) { return (float)a > (float)b; }
MVCC_HD bool __hge(const __half a, const __half b) { return (float)a >= (float)b; }
MVCC_HD bool __hequ(const __half a, const __half b) { return !((float)a < (float)b) && !((float)a > (float)b); }
MVCC_HD bool __hneu(const __half a, const __half b) { return !((float)a == (float)b); }
MVCC_HD bool __hltu(const __half a, const __half b) { return !((float)a >= (float)b); }
MVCC_HD bool __hleu(const __half a, const __half b) { return !((float)a > (float)b); }
MVCC_HD bool __hgtu(const __half a, const __half b) { return !((float)a <= (float)b); }
MVCC_HD bool __hgeu(const __half a, const __half b) { return !((float)a < (float)b); }
MVCC_HD bool __hisnan(const __half a) { return (float)a != (float)a; }
MVCC_HD bool __hisinf(const __half a) { __half_raw r = a; return (r.x & 0x7fff) == 0x7c00; }
MVCC_HD __half2 __heq2(const __half2 a, const __half2 b) { return __half2(__half(__heq(a.x, b.x) ? 1.f : 0.f), __half(__heq(a.y, b.y) ? 1.f : 0.f)); }
MVCC_HD __half2 __hne2(const __half2 a, const __half2 b) { return __half2(__half(__hne(a.x, b.x) ? 1.f : 0.f), __half(__hne(a.y, b.y) ? 1.f : 0.f)); }
MVCC_HD __half2 __hlt2(const __half2 a, const __half2 b) { return __half2(__half(__hlt(a.x, b.x) ? 1.f : 0.f), __half(__hlt(a.y, b.y) ? 1.f : 0.f)); }
MVCC_HD __half2 __hle2(const __half2 a, const __half2 b) { return __half2(__half(__hle(a.x, b.x) ? 1.f : 0.f), __half(__hle(a.y, b.y) ? 1.f : 0.f)); }
MVCC_HD __half2 __hgt2(const __half2 a, const __half2 b) { return __half2(__half(__hgt(a.x, b.x) ? 1.f : 0.f), __half(__hgt(a.y, b.y) ? 1.f : 0.f)); }
MVCC_HD __half2 __hge2(const __half2 a, const __half2 b) { return __half2(__half(__hge(a.x, b.x) ? 1.f : 0.f), __half(__hge(a.y, b.y) ? 1.f : 0.f)); }
MVCC_HD bool __hbeq2(const __half2 a, const __half2 b) { return __heq(a.x, b.x) && __heq(a.y, b.y); }
MVCC_HD bool __hbne2(const __half2 a, const __half2 b) { return __hne(a.x, b.x) && __hne(a.y, b.y); }
MVCC_HD bool __hblt2(const __half2 a, const __half2 b) { return __hlt(a.x, b.x) && __hlt(a.y, b.y); }
MVCC_HD bool __hble2(const __half2 a, const __half2 b) { return __hle(a.x, b.x) && __hle(a.y, b.y); }
MVCC_HD bool __hbgt2(const __half2 a, const __half2 b) { return __hgt(a.x, b.x) && __hgt(a.y, b.y); }
MVCC_HD bool __hbge2(const __half2 a, const __half2 b) { return __hge(a.x, b.x) && __hge(a.y, b.y); }
MVCC_HD __half2 __hisnan2(const __half2 a) { return __half2(__half(__hisnan(a.x) ? 1.f : 0.f), __half(__hisnan(a.y) ? 1.f : 0.f)); }

// --- math ---
MVCC_HD __half hsqrt(const __half a) { return __half(__builtin_sqrtf((float)a)); }
MVCC_HD __half hrsqrt(const __half a) { return __half(1.0f / __builtin_sqrtf((float)a)); }
MVCC_HD __half hrcp(const __half a) { return __half(1.0f / (float)a); }
MVCC_HD __half hexp(const __half a) { return __half(__builtin_expf((float)a)); }
MVCC_HD __half hexp2(const __half a) { return __half(__builtin_exp2f((float)a)); }
MVCC_HD __half hexp10(const __half a) { return __half(__builtin_powf(10.0f, (float)a)); }
MVCC_HD __half hlog(const __half a) { return __half(__builtin_logf((float)a)); }
MVCC_HD __half hlog2(const __half a) { return __half(__builtin_log2f((float)a)); }
MVCC_HD __half hlog10(const __half a) { return __half(__builtin_log10f((float)a)); }
MVCC_HD __half hsin(const __half a) { return __half(__builtin_sinf((float)a)); }
MVCC_HD __half hcos(const __half a) { return __half(__builtin_cosf((float)a)); }
MVCC_HD __half hfloor(const __half a) { return __half(__builtin_floorf((float)a)); }
MVCC_HD __half hceil(const __half a) { return __half(__builtin_ceilf((float)a)); }
MVCC_HD __half htrunc(const __half a) { return __half(__builtin_truncf((float)a)); }
MVCC_HD __half hrint(const __half a) { return __half(__builtin_rintf((float)a)); }
MVCC_HD __half2 h2sqrt(const __half2 a) { return __half2(hsqrt(a.x), hsqrt(a.y)); }
MVCC_HD __half2 h2rsqrt(const __half2 a) { return __half2(hrsqrt(a.x), hrsqrt(a.y)); }
MVCC_HD __half2 h2rcp(const __half2 a) { return __half2(hrcp(a.x), hrcp(a.y)); }
MVCC_HD __half2 h2exp(const __half2 a) { return __half2(hexp(a.x), hexp(a.y)); }
MVCC_HD __half2 h2exp2(const __half2 a) { return __half2(hexp2(a.x), hexp2(a.y)); }
MVCC_HD __half2 h2log(const __half2 a) { return __half2(hlog(a.x), hlog(a.y)); }
MVCC_HD __half2 h2log2(const __half2 a) { return __half2(hlog2(a.x), hlog2(a.y)); }
MVCC_HD __half2 h2sin(const __half2 a) { return __half2(hsin(a.x), hsin(a.y)); }
MVCC_HD __half2 h2cos(const __half2 a) { return __half2(hcos(a.x), hcos(a.y)); }
MVCC_HD __half2 h2floor(const __half2 a) { return __half2(hfloor(a.x), hfloor(a.y)); }
MVCC_HD __half2 h2ceil(const __half2 a) { return __half2(hceil(a.x), hceil(a.y)); }
MVCC_HD __half2 h2trunc(const __half2 a) { return __half2(htrunc(a.x), htrunc(a.y)); }
MVCC_HD __half2 h2rint(const __half2 a) { return __half2(hrint(a.x), hrint(a.y)); }

// --- operators ---
MVCC_HD __half operator+(const __half& a, const __half& b) { return __hadd(a, b); }
MVCC_HD __half operator-(const __half& a, const __half& b) { return __hsub(a, b); }
MVCC_HD __half operator*(const __half& a, const __half& b) { return __hmul(a, b); }
MVCC_HD __half operator/(const __half& a, const __half& b) { return __hdiv(a, b); }
MVCC_HD __half operator-(const __half& a) { return __hneg(a); }
MVCC_HD __half operator+(const __half& a) { return a; }
MVCC_HD __half& operator+=(__half& a, const __half& b) { a = a + b; return a; }
MVCC_HD __half& operator-=(__half& a, const __half& b) { a = a - b; return a; }
MVCC_HD __half& operator*=(__half& a, const __half& b) { a = a * b; return a; }
MVCC_HD __half& operator/=(__half& a, const __half& b) { a = a / b; return a; }
MVCC_HD __half& operator++(__half& a) { a = a + __half(1.0f); return a; }
MVCC_HD __half& operator--(__half& a) { a = a - __half(1.0f); return a; }
MVCC_HD __half operator++(__half& a, int) { __half r = a; a = a + __half(1.0f); return r; }
MVCC_HD __half operator--(__half& a, int) { __half r = a; a = a - __half(1.0f); return r; }
MVCC_HD bool operator==(const __half& a, const __half& b) { return __heq(a, b); }
MVCC_HD bool operator!=(const __half& a, const __half& b) { return __hneu(a, b); }
MVCC_HD bool operator<(const __half& a, const __half& b) { return __hlt(a, b); }
MVCC_HD bool operator<=(const __half& a, const __half& b) { return __hle(a, b); }
MVCC_HD bool operator>(const __half& a, const __half& b) { return __hgt(a, b); }
MVCC_HD bool operator>=(const __half& a, const __half& b) { return __hge(a, b); }
MVCC_HD __half2 operator+(const __half2& a, const __half2& b) { return __hadd2(a, b); }
MVCC_HD __half2 operator-(const __half2& a, const __half2& b) { return __hsub2(a, b); }
MVCC_HD __half2 operator*(const __half2& a, const __half2& b) { return __hmul2(a, b); }
MVCC_HD __half2 operator/(const __half2& a, const __half2& b) { return __h2div(a, b); }
MVCC_HD __half2 operator-(const __half2& a) { return __hneg2(a); }
MVCC_HD __half2& operator+=(__half2& a, const __half2& b) { a = a + b; return a; }
MVCC_HD __half2& operator-=(__half2& a, const __half2& b) { a = a - b; return a; }
MVCC_HD __half2& operator*=(__half2& a, const __half2& b) { a = a * b; return a; }
MVCC_HD __half2& operator/=(__half2& a, const __half2& b) { a = a / b; return a; }
MVCC_HD bool operator==(const __half2& a, const __half2& b) { return __hbeq2(a, b); }
MVCC_HD bool operator!=(const __half2& a, const __half2& b) { return !__hbeq2(a, b); }

// --- loads with cache hints (plain loads) ---
MVCC_HD __half __ldg(const __half* p) { return *p; }
MVCC_HD __half2 __ldg(const __half2* p) { return *p; }
// __ldcg reads at L2 (a value another SM stored): a volatile load, which mvcc-ir2msl performs coherently
MVCC_HD __half __ldcg(const __half* p) { __half r; *reinterpret_cast<unsigned short*>(&r) = *reinterpret_cast<const volatile unsigned short*>(p); return r; }
MVCC_HD __half2 __ldcg(const __half2* p) { __half2 r; *reinterpret_cast<unsigned int*>(&r) = *reinterpret_cast<const volatile unsigned int*>(p); return r; }
MVCC_HD __half __ldca(const __half* p) { return *p; }
MVCC_HD __half2 __ldca(const __half2* p) { return *p; }
MVCC_HD __half __ldcs(const __half* p) { return *p; }
MVCC_HD __half2 __ldcs(const __half2* p) { return *p; }
MVCC_HD void __stcg(__half* p, __half v) { *p = v; }
MVCC_HD void __stcg(__half2* p, __half2 v) { *p = v; }
MVCC_HD void __stcs(__half* p, __half v) { *p = v; }
MVCC_HD void __stcs(__half2* p, __half2 v) { *p = v; }

// --- warp shuffles ---
#if defined(__CUDA__) && defined(__clang__)
static __device__ __forceinline__ __half2 __shfl_sync(unsigned mask, __half2 v, int src, int width = 32) { return __uint_as_half2((unsigned)__shfl_sync(mask, (int)__half2_as_uint(v), src, width)); }
static __device__ __forceinline__ __half2 __shfl_xor_sync(unsigned mask, __half2 v, int m, int width = 32) { return __uint_as_half2((unsigned)__shfl_xor_sync(mask, (int)__half2_as_uint(v), m, width)); }
static __device__ __forceinline__ __half2 __shfl_down_sync(unsigned mask, __half2 v, unsigned d, int width = 32) { return __uint_as_half2((unsigned)__shfl_down_sync(mask, (int)__half2_as_uint(v), d, width)); }
static __device__ __forceinline__ __half2 __shfl_up_sync(unsigned mask, __half2 v, unsigned d, int width = 32) { return __uint_as_half2((unsigned)__shfl_up_sync(mask, (int)__half2_as_uint(v), d, width)); }
static __device__ __forceinline__ __half __shfl_sync(unsigned mask, __half v, int src, int width = 32) { return __ushort_as_half((unsigned short)__shfl_sync(mask, (int)__half_as_ushort(v), src, width)); }
static __device__ __forceinline__ __half __shfl_xor_sync(unsigned mask, __half v, int m, int width = 32) { return __ushort_as_half((unsigned short)__shfl_xor_sync(mask, (int)__half_as_ushort(v), m, width)); }
static __device__ __forceinline__ __half __shfl_down_sync(unsigned mask, __half v, unsigned d, int width = 32) { return __ushort_as_half((unsigned short)__shfl_down_sync(mask, (int)__half_as_ushort(v), d, width)); }
static __device__ __forceinline__ __half __shfl_up_sync(unsigned mask, __half v, unsigned d, int width = 32) { return __ushort_as_half((unsigned short)__shfl_up_sync(mask, (int)__half_as_ushort(v), d, width)); }
// atomics on half2 (CAS loop on the 32-bit word)
static __device__ __forceinline__ __half2 atomicAdd(__half2* address, __half2 val) {
  unsigned int* p = (unsigned int*)address; unsigned int old = *p, assumed;
  do { assumed = old; __half2 cur = __uint_as_half2(assumed); __half2 nv = __hadd2(cur, val); old = atomicCAS(p, assumed, __half2_as_uint(nv)); } while (assumed != old);
  return __uint_as_half2(old);
}
#endif

#undef MVCC_HD
#endif // MVCC_CUDA_FP16_H
