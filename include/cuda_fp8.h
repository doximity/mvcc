// mvcc: FP8 (E4M3 / E5M2) storage types and conversions (public cuda_fp8 API).
// Formats per the OCP FP8 specification: E4M3 has bias 7, no infinities, NaN = S.1111.111;
// E5M2 has bias 15 and IEEE-style inf/NaN. Round-to-nearest-even; saturation per __nv_saturation_t.
#ifndef MVCC_CUDA_FP8_H
#define MVCC_CUDA_FP8_H
#include <mvcc/host_defines.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdint.h>
#include <string.h>

#define MVCC_HD __host__ __device__ __forceinline__

typedef unsigned char __nv_fp8_storage_t;
typedef unsigned short __nv_fp8x2_storage_t;
typedef unsigned int __nv_fp8x4_storage_t;

typedef enum __nv_saturation_t { __NV_NOSAT, __NV_SATFINITE } __nv_saturation_t;
typedef enum __nv_fp8_interpretation_t { __NV_E4M3, __NV_E5M2 } __nv_fp8_interpretation_t;

namespace __mvcc_fp8 {
// float -> fp8 with RNE. mbits = mantissa bits (3 or 2), bias, has_inf.
MVCC_HD unsigned char f2fp8(float f, int mbits, int bias, bool e5m2, bool sat) {
  unsigned int u; memcpy(&u, &f, 4);
  unsigned char sign = (unsigned char)((u >> 31) << 7);
  unsigned int exp = (u >> 23) & 0xff; unsigned int man = u & 0x7fffff;
  const int emax_biased = e5m2 ? 30 : 15;  // largest finite exponent field (e4m3: 1111 with man != 111; handled below)
  if (exp == 0xff) {
    if (man) return (unsigned char)(sign | (e5m2 ? 0x7f : 0x7f));           // NaN (e5m2: S.11111.11 is one NaN encoding)
    if (e5m2) return sat ? (unsigned char)(sign | 0x7b) : (unsigned char)(sign | 0x7c);  // inf
    return sat ? (unsigned char)(sign | 0x7e) : (unsigned char)(sign | 0x7f);            // e4m3 has no inf: NaN unless saturating
  }
  if (exp == 0 && man == 0) return sign;
  // unbiased exponent and normalized mantissa with hidden bit at bit 23
  int e = (int)exp - 127;
  unsigned int m = man | 0x800000u;
  if (exp == 0) { // float subnormal: too small for any fp8 format
    return sign;
  }
  int fe = e + bias;  // fp8 biased exponent
  int shift = 23 - mbits;
  unsigned int mant;
  if (fe <= 0) {
    // fp8 subnormal: value = m * 2^(e-23) ; fp8 subnormal unit = 2^(1-bias-mbits)
    int extra = 1 - fe;  // additional right shift
    shift += extra;
    if (shift > 31) return sign;
    fe = 0;
    unsigned int rem = m & ((1u << shift) - 1u); unsigned int half = 1u << (shift - 1);
    mant = m >> shift;
    if (rem > half || (rem == half && (mant & 1u))) mant++;
    if (mant >> mbits) { fe = 1; mant &= (1u << mbits) - 1u; }
    return (unsigned char)(sign | (fe << mbits) | mant);
  }
  unsigned int rem = m & ((1u << shift) - 1u); unsigned int half = 1u << (shift - 1);
  mant = (m >> shift) & ((1u << mbits) - 1u);
  if (rem > half || (rem == half && (mant & 1u))) { mant++; if (mant >> mbits) { mant = 0; fe++; } }
  // overflow checks
  if (e5m2) {
    if (fe >= 31) return sat ? (unsigned char)(sign | 0x7b) : (unsigned char)(sign | 0x7c);
    return (unsigned char)(sign | (fe << 2) | mant);
  }
  // e4m3: max finite is 1111.110 (448). 1111.111 is NaN.
  if (fe > 15 || (fe == 15 && mant == 7)) return sat ? (unsigned char)(sign | 0x7e) : (unsigned char)(sign | 0x7f);
  (void)emax_biased;
  return (unsigned char)(sign | (fe << 3) | mant);
}
MVCC_HD float fp82f(unsigned char v, bool e5m2) {
  unsigned int sign = (v & 0x80u) << 24;
  unsigned int e, m;
  if (e5m2) { e = (v >> 2) & 0x1f; m = v & 3; }
  else { e = (v >> 3) & 0xf; m = v & 7; }
  int mbits = e5m2 ? 2 : 3; int bias = e5m2 ? 15 : 7;
  float r;
  if (e5m2 && e == 0x1f) { unsigned int u = sign | 0x7f800000u | (m ? (m << 21) : 0); memcpy(&r, &u, 4); return r; }
  if (!e5m2 && e == 0xf && m == 7) { unsigned int u = sign | 0x7fc00000u; memcpy(&r, &u, 4); return r; }
  if (e == 0) {
    if (m == 0) { unsigned int u = sign; memcpy(&r, &u, 4); return r; }
    // subnormal: m * 2^(1-bias-mbits)
    float val = (float)m; int sc = 1 - bias - mbits;
    unsigned int pu = (unsigned int)(127 + sc) << 23; float p; memcpy(&p, &pu, 4);
    r = val * p; unsigned int ru; memcpy(&ru, &r, 4); ru |= sign; memcpy(&r, &ru, 4); return r;
  }
  unsigned int u = sign | ((unsigned int)((int)e - bias + 127) << 23) | (m << (23 - mbits));
  memcpy(&r, &u, 4); return r;
}
}  // namespace __mvcc_fp8

namespace __mvcc_fp8 {
// float -> e4m3 with RNE and finite saturation, without branches: f2fp8(x, 3, 7, false, true) bit for bit
// (the common quantization case; f2fp8's data-dependent branches and variable shifts cost ~10x this on Apple GPUs).
// Normals round in the fp32 bit pattern (add half an fp8 ulp plus the RNE tie bit, drop the low 20 bits; a carry
// into the exponent is a correct round-up), subnormals as rint of the value in units of 2^-9 (whose code is the
// integer itself, 8 being the smallest normal). Magnitudes above 448 clamp to 448 first: SATFINITE gives 448 for
// them either way (RNE from (448, 480) lands on 448, the rest overflow to the saturation value 448).
MVCC_HD unsigned char f2e4m3_sat(float f) {
  unsigned int u; memcpy(&u, &f, 4);
  const unsigned char sign = (unsigned char)((u >> 31) << 7);
  const bool nan = (u & 0x7fffffffu) > 0x7f800000u;
  float ax = fabsf(f);
  ax = ax > 448.f ? 448.f : ax;
  unsigned int a; memcpy(&a, &ax, 4);
  const unsigned int rn = a + 0x7ffffu + ((a >> 20) & 1u);
  const unsigned int normal = ((rn >> 20) - (120u << 3)) & 0x7fu;
  const unsigned int sub = (unsigned int)rintf(ax * 512.f);
  const unsigned int code = a >= 0x3c800000u ? normal : sub;  // 2^-6: smallest e4m3 normal
  return (unsigned char)(sign | (nan ? 0x7fu : code));
}
}  // namespace __mvcc_fp8

MVCC_HD __nv_fp8_storage_t __nv_cvt_float_to_fp8(float x, __nv_saturation_t sat, __nv_fp8_interpretation_t interp) {
  if (interp == __NV_E4M3 && sat == __NV_SATFINITE) return __mvcc_fp8::f2e4m3_sat(x);
  return interp == __NV_E4M3 ? __mvcc_fp8::f2fp8(x, 3, 7, false, sat == __NV_SATFINITE) : __mvcc_fp8::f2fp8(x, 2, 15, true, sat == __NV_SATFINITE);
}
MVCC_HD __nv_fp8x2_storage_t __nv_cvt_float2_to_fp8x2(float2 x, __nv_saturation_t sat, __nv_fp8_interpretation_t interp) {
  return (__nv_fp8x2_storage_t)__nv_cvt_float_to_fp8(x.x, sat, interp) | ((__nv_fp8x2_storage_t)__nv_cvt_float_to_fp8(x.y, sat, interp) << 8);
}
MVCC_HD __nv_fp8_storage_t __nv_cvt_double_to_fp8(double x, __nv_saturation_t sat, __nv_fp8_interpretation_t interp) { return __nv_cvt_float_to_fp8((float)x, sat, interp); }
MVCC_HD __nv_fp8_storage_t __nv_cvt_halfraw_to_fp8(__half_raw x, __nv_saturation_t sat, __nv_fp8_interpretation_t interp) { return __nv_cvt_float_to_fp8((float)__half(x), sat, interp); }
MVCC_HD __nv_fp8x2_storage_t __nv_cvt_halfraw2_to_fp8x2(__half2_raw x, __nv_saturation_t sat, __nv_fp8_interpretation_t interp) {
  return (__nv_fp8x2_storage_t)__nv_cvt_halfraw_to_fp8(__half_raw{x.x}, sat, interp) | ((__nv_fp8x2_storage_t)__nv_cvt_halfraw_to_fp8(__half_raw{x.y}, sat, interp) << 8);
}
MVCC_HD __nv_fp8_storage_t __nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw x, __nv_saturation_t sat, __nv_fp8_interpretation_t interp) { return __nv_cvt_float_to_fp8((float)__nv_bfloat16(x), sat, interp); }
MVCC_HD __nv_fp8x2_storage_t __nv_cvt_bfloat16raw2_to_fp8x2(__nv_bfloat162_raw x, __nv_saturation_t sat, __nv_fp8_interpretation_t interp) {
  return (__nv_fp8x2_storage_t)__nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw{x.x}, sat, interp) | ((__nv_fp8x2_storage_t)__nv_cvt_bfloat16raw_to_fp8(__nv_bfloat16_raw{x.y}, sat, interp) << 8);
}
MVCC_HD __half_raw __nv_cvt_fp8_to_halfraw(__nv_fp8_storage_t x, __nv_fp8_interpretation_t interp) {
  return (__half_raw)__half(__mvcc_fp8::fp82f(x, interp == __NV_E5M2));
}
MVCC_HD __half2_raw __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t x, __nv_fp8_interpretation_t interp) {
  __half2_raw r; r.x = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(x & 0xff), interp).x; r.y = __nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t)(x >> 8), interp).x; return r;
}
MVCC_HD float __mvcc_fp8_to_float(__nv_fp8_storage_t x, __nv_fp8_interpretation_t interp) { return __mvcc_fp8::fp82f(x, interp == __NV_E5M2); }

// --- C++ wrapper classes ---
struct __nv_fp8_e4m3 {
  __nv_fp8_storage_t __x;
  __nv_fp8_e4m3() = default;
  MVCC_HD explicit __nv_fp8_e4m3(float f) : __x(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E4M3)) {}
  MVCC_HD explicit __nv_fp8_e4m3(double f) : __x(__nv_cvt_float_to_fp8((float)f, __NV_SATFINITE, __NV_E4M3)) {}
  MVCC_HD explicit __nv_fp8_e4m3(__half h) : __x(__nv_cvt_float_to_fp8((float)h, __NV_SATFINITE, __NV_E4M3)) {}
  MVCC_HD explicit __nv_fp8_e4m3(__nv_bfloat16 b) : __x(__nv_cvt_float_to_fp8((float)b, __NV_SATFINITE, __NV_E4M3)) {}
  MVCC_HD explicit __nv_fp8_e4m3(int v) : __x(__nv_cvt_float_to_fp8((float)v, __NV_SATFINITE, __NV_E4M3)) {}
  MVCC_HD explicit operator float() const { return __mvcc_fp8::fp82f(__x, false); }
  MVCC_HD explicit operator __half() const { return __half((float)*this); }
  MVCC_HD explicit operator __nv_bfloat16() const { return __nv_bfloat16((float)*this); }
  MVCC_HD explicit operator bool() const { return (__x & 0x7f) != 0; }
};
struct __nv_fp8_e5m2 {
  __nv_fp8_storage_t __x;
  __nv_fp8_e5m2() = default;
  MVCC_HD explicit __nv_fp8_e5m2(float f) : __x(__nv_cvt_float_to_fp8(f, __NV_SATFINITE, __NV_E5M2)) {}
  MVCC_HD explicit __nv_fp8_e5m2(double f) : __x(__nv_cvt_float_to_fp8((float)f, __NV_SATFINITE, __NV_E5M2)) {}
  MVCC_HD explicit __nv_fp8_e5m2(__half h) : __x(__nv_cvt_float_to_fp8((float)h, __NV_SATFINITE, __NV_E5M2)) {}
  MVCC_HD explicit operator float() const { return __mvcc_fp8::fp82f(__x, true); }
  MVCC_HD explicit operator __half() const { return __half((float)*this); }
  MVCC_HD explicit operator bool() const { return (__x & 0x7f) != 0; }
};
struct __nv_fp8x2_e4m3 {
  __nv_fp8x2_storage_t __x;
  __nv_fp8x2_e4m3() = default;
  MVCC_HD explicit __nv_fp8x2_e4m3(float2 f) : __x(__nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E4M3)) {}
  MVCC_HD explicit __nv_fp8x2_e4m3(__half2 h) : __x(__nv_cvt_halfraw2_to_fp8x2((__half2_raw)h, __NV_SATFINITE, __NV_E4M3)) {}
  MVCC_HD explicit operator float2() const { return make_float2(__mvcc_fp8::fp82f((unsigned char)__x, false), __mvcc_fp8::fp82f((unsigned char)(__x >> 8), false)); }
  MVCC_HD explicit operator __half2() const { return __half2(__nv_cvt_fp8x2_to_halfraw2(__x, __NV_E4M3)); }
};
struct __nv_fp8x2_e5m2 {
  __nv_fp8x2_storage_t __x;
  __nv_fp8x2_e5m2() = default;
  MVCC_HD explicit __nv_fp8x2_e5m2(float2 f) : __x(__nv_cvt_float2_to_fp8x2(f, __NV_SATFINITE, __NV_E5M2)) {}
  MVCC_HD explicit operator float2() const { return make_float2(__mvcc_fp8::fp82f((unsigned char)__x, true), __mvcc_fp8::fp82f((unsigned char)(__x >> 8), true)); }
  MVCC_HD explicit operator __half2() const { return __half2(__nv_cvt_fp8x2_to_halfraw2(__x, __NV_E5M2)); }
};
struct __nv_fp8x4_e4m3 {
  __nv_fp8x4_storage_t __x;
  __nv_fp8x4_e4m3() = default;
  MVCC_HD explicit __nv_fp8x4_e4m3(float4 f) : __x((unsigned)__nv_cvt_float2_to_fp8x2(make_float2(f.x, f.y), __NV_SATFINITE, __NV_E4M3) | ((unsigned)__nv_cvt_float2_to_fp8x2(make_float2(f.z, f.w), __NV_SATFINITE, __NV_E4M3) << 16)) {}
  MVCC_HD explicit operator float4() const {
    return make_float4(__mvcc_fp8::fp82f((unsigned char)__x, false), __mvcc_fp8::fp82f((unsigned char)(__x >> 8), false), __mvcc_fp8::fp82f((unsigned char)(__x >> 16), false), __mvcc_fp8::fp82f((unsigned char)(__x >> 24), false));
  }
};

#undef MVCC_HD
#endif // MVCC_CUDA_FP8_H
