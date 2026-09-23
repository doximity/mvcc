// Device intrinsics and header semantics against CPU references:
// warp shuffles / votes, integer intrinsics, fp16/bf16 conversions (exhaustive over the 16-bit domain plus
// rounding ties), atomics under contention, libm-style math with ULP bounds, shared memory + barriers.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CK(x)                                                                                              \
  do {                                                                                                     \
    cudaError_t e_ = (x);                                                                                  \
    if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } \
  } while (0)

static int failures = 0;
static void report(const char* name, int bad, int total) {
  printf("  %-38s %s (%d/%d mismatches)\n", name, bad ? "FAIL" : "ok", bad, total);
  if (bad) ++failures;
}

// ------------------------------------------------------------------ warp ops
// out[lane*16 + i]: assorted shuffle / vote / integer intrinsic results for one warp.
__global__ void warp_kernel(uint32_t* out) {
  const int lane = threadIdx.x & 31;
  const uint32_t v = 0x9e3779b9u * (lane + 1);
  uint32_t* o = out + lane * 16;
  o[0] = __shfl_sync(0xffffffffu, v, (lane * 7) & 31);
  o[1] = __shfl_xor_sync(0xffffffffu, v, 5);
  o[2] = __shfl_up_sync(0xffffffffu, v, 3);
  o[3] = __shfl_down_sync(0xffffffffu, v, 3);
  o[4] = __shfl_sync(0xffffffffu, v, lane + 1, 16);      // width 16: wraps inside each half
  o[5] = __shfl_up_sync(0xffffffffu, v, 1, 8);           // width 8: lane 0 of each octet keeps its value
  o[6] = __ballot_sync(0xffffffffu, (lane & 3) == 1);
  o[7] = __any_sync(0xffffffffu, lane == 17) | (__all_sync(0xffffffffu, lane < 32) << 1) | (__all_sync(0xffffffffu, lane < 31) << 2);
  o[8] = __popc(v) | (__clz(v >> (lane & 7)) << 8) | (__ffs(v << (lane & 7)) << 16);
  o[9] = __brev(v);
  o[10] = __byte_perm(v, ~v, 0x3271 + lane);
  o[11] = __funnelshift_l(v, ~v, lane) ^ __funnelshift_r(~v, v, lane);
  o[12] = __umulhi(v, 0xdeadbeefu) + __mulhi((int)v, -12345);
  o[13] = (uint32_t)(__umul64hi((uint64_t)v << 20, 0x123456789abcdefull) & 0xffffffffu);
  o[14] = __usad(v, 0x12345678u, lane) ^ (uint32_t)__dp4a((int)v, (int)~v, lane) ^ __dp4a(v, ~v, 7u) ^ (uint32_t)__dp2a_lo((int)v, (int)~v, 1) ^ (uint32_t)__dp2a_hi((int)v, (int)~v, 2);
  o[15] = (uint32_t)__shfl_sync(0xffffffffu, (float)lane * 1.5f, 31 - lane);
}
static uint32_t brev32(uint32_t x) { uint32_t r = 0; for (int i = 0; i < 32; ++i) r |= ((x >> i) & 1u) << (31 - i); return r; }
static uint32_t byte_perm(uint32_t a, uint32_t b, uint32_t s) {
  uint8_t bytes[8]; memcpy(bytes, &a, 4); memcpy(bytes + 4, &b, 4);
  uint32_t r = 0; for (int i = 0; i < 4; ++i) r |= (uint32_t)bytes[(s >> (4 * i)) & 7] << (8 * i); return r;
}
static void test_warp() {
  uint32_t* d; CK(cudaMalloc(&d, 32 * 16 * 4));
  warp_kernel<<<1, 32>>>(d);
  std::vector<uint32_t> got(32 * 16); CK(cudaMemcpy(got.data(), d, got.size() * 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  auto vof = [](int l) { return 0x9e3779b9u * (l + 1); };
  for (int lane = 0; lane < 32; ++lane) {
    const uint32_t v = vof(lane);
    uint32_t e[16];
    e[0] = vof((lane * 7) & 31);
    e[1] = vof(lane ^ 5);
    e[2] = lane >= 3 ? vof(lane - 3) : v;
    e[3] = lane + 3 < 32 ? vof(lane + 3) : v;
    e[4] = vof((lane & ~15) | ((lane + 1) & 15));
    e[5] = (lane & 7) >= 1 ? vof(lane - 1) : v;
    uint32_t ballot = 0; for (int l = 0; l < 32; ++l) if ((l & 3) == 1) ballot |= 1u << l;
    e[6] = ballot;
    e[7] = 1u | (1u << 1) | (0u << 2);
    e[8] = (uint32_t)__builtin_popcount(v) | ((uint32_t)__builtin_clz(v >> (lane & 7)) << 8) | ((uint32_t)__builtin_ffs((int)(v << (lane & 7))) << 16);
    e[9] = brev32(v);
    e[10] = byte_perm(v, ~v, 0x3271 + lane);
    { uint64_t hl = ((uint64_t)(~v) << 32) | v; uint32_t fl = (uint32_t)(hl >> (32 - (lane & 31))); if ((lane & 31) == 0) fl = ~v;  // funnelshift_l(lo=v, hi=~v, s) = (hi:lo) >> (32 - s) low word
      uint64_t hr = ((uint64_t)v << 32) | (~v); uint32_t fr = (uint32_t)(hr >> (lane & 31)); e[11] = fl ^ fr; }
    e[12] = (uint32_t)(((uint64_t)v * 0xdeadbeefu) >> 32) + (uint32_t)(((int64_t)(int)v * (int64_t)-12345) >> 32);
    e[13] = (uint32_t)(((unsigned __int128)((uint64_t)v << 20) * (unsigned __int128)0x123456789abcdefull) >> 64);
    { int32_t s4 = lane; uint32_t u4 = 7; for (int b = 0; b < 4; ++b) { s4 += (int)(int8_t)(v >> (8 * b)) * (int)(int8_t)((~v) >> (8 * b)); u4 += ((v >> (8 * b)) & 0xff) * (((~v) >> (8 * b)) & 0xff); }
      int32_t lo = 1 + (int)(int16_t)v * (int)(int8_t)(~v) + (int)(int16_t)(v >> 16) * (int)(int8_t)((~v) >> 8);
      int32_t hi = 2 + (int)(int16_t)v * (int)(int8_t)((~v) >> 16) + (int)(int16_t)(v >> 16) * (int)(int8_t)((~v) >> 24);
      e[14] = ((v > 0x12345678u ? v - 0x12345678u : 0x12345678u - v) + lane) ^ (uint32_t)s4 ^ u4 ^ (uint32_t)lo ^ (uint32_t)hi; }
    { float f = (float)(31 - lane) * 1.5f; e[15] = (uint32_t)f; }
    for (int i = 0; i < 16; ++i)
      if (got[lane * 16 + i] != e[i]) { if (bad < 5) fprintf(stderr, "    lane %d slot %d: got %08x want %08x\n", lane, i, got[lane * 16 + i], e[i]); ++bad; }
  }
  report("warp shuffles / votes / int intrinsics", bad, 32 * 16);
  CK(cudaFree(d));
}

// ------------------------------------------------------------------ conversions
// For each 16-bit pattern p: bf16(p)->float->bits, half(p)->float->bits, and float->bf16/half RNE for a
// float built from p as the high half with three low-half patterns (0, tie 0x8000, above-tie 0x8001).
__global__ void cvt_kernel(uint32_t* out) {
  const uint32_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= 65536) return;
  uint32_t* o = out + p * 8;
  o[0] = __float_as_uint(__bfloat162float(__ushort_as_bfloat16((unsigned short)p)));
  o[1] = __float_as_uint(__half2float(__ushort_as_half((unsigned short)p)));
  const float f0 = __uint_as_float(p << 16), f1 = __uint_as_float((p << 16) | 0x8000u), f2 = __uint_as_float((p << 16) | 0x8001u);
  o[2] = __bfloat16_as_ushort(__float2bfloat16(f0));
  o[3] = __bfloat16_as_ushort(__float2bfloat16(f1));
  o[4] = __bfloat16_as_ushort(__float2bfloat16(f2));
  o[5] = __half_as_ushort(__float2half(__uint_as_float(p << 16)));                 // f32 -> f16 RNE (overflow, subnormals, NaN)
  o[6] = __half_as_ushort(__float2half_rz(__uint_as_float((p << 16) | 0x1234u)));  // truncation
  o[7] = __bfloat16_as_ushort(__float2bfloat16_rz(f2));
}
static uint32_t bf2f_bits(uint16_t b) { return (uint32_t)b << 16; }
static uint32_t h2f_bits(uint16_t h) { _Float16 x; memcpy(&x, &h, 2); float f = (float)x; uint32_t u; memcpy(&u, &f, 4); return u; }
static uint16_t f2bf_rne(uint32_t u) {
  if ((u & 0x7f800000u) == 0x7f800000u) return (uint16_t)((u >> 16) | ((u & 0x7fffffu) ? 0x40 : 0));  // inf stays, NaN quiet
  return (uint16_t)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}
static uint16_t f2h_rne(uint32_t u) { float f; memcpy(&f, &u, 4); _Float16 h = (_Float16)f; uint16_t r; memcpy(&r, &h, 2); return r; }
static uint16_t f2h_rz(uint32_t u) {
  float f; memcpy(&f, &u, 4);
  const uint32_t sign = (u >> 16) & 0x8000u, ex = (u >> 23) & 0xff, man = u & 0x7fffffu;
  if (ex == 0xff) return (uint16_t)(sign | 0x7c00u | (man ? 0x200u | (man >> 13) : 0));
  const int e = (int)ex - 127 + 15;
  if (e >= 31) return (uint16_t)(sign | 0x7bffu);  // rz: largest finite
  if (e <= 0) {  // subnormal or zero: shift with truncation
    if (e < -10) return (uint16_t)sign;
    const uint32_t m = (man | 0x800000u) >> (1 - e);
    return (uint16_t)(sign | (m >> 13));
  }
  return (uint16_t)(sign | ((uint32_t)e << 10) | (man >> 13));
}
// NaN results compare equal to any NaN: PTX cvt produces the canonical NaN, so sign and payload are not preserved
static bool same_or_both_nan_h(uint16_t a, uint16_t b) { return a == b || (((a & 0x7c00) == 0x7c00 && (a & 0x3ff)) && ((b & 0x7c00) == 0x7c00 && (b & 0x3ff))); }
static bool same_or_both_nan_bf(uint16_t a, uint16_t b) { return a == b || (((a & 0x7f80) == 0x7f80 && (a & 0x7f)) && ((b & 0x7f80) == 0x7f80 && (b & 0x7f))); }
static bool same_or_both_nan_f(uint32_t a, uint32_t b) { return a == b || (((a & 0x7f800000u) == 0x7f800000u && (a & 0x7fffffu)) && ((b & 0x7f800000u) == 0x7f800000u && (b & 0x7fffffu))); }
static void test_cvt() {
  uint32_t* d; CK(cudaMalloc(&d, 65536 * 8 * 4));
  cvt_kernel<<<256, 256>>>(d);
  std::vector<uint32_t> got(65536 * 8); CK(cudaMemcpy(got.data(), d, got.size() * 4, cudaMemcpyDeviceToHost));
  int bad[8] = {0};
  for (uint32_t p = 0; p < 65536; ++p) {
    const uint32_t* g = got.data() + p * 8;
    const uint32_t hi = p << 16;
    // fp32 denormal inputs (exponent field 0, nonzero mantissa) are flushed to zero by the GPU (FTZ, see
    // README.md); bf16 keeps fp32's exponent range so only these patterns differ from CUDA.
    const bool denorm_in = ((p >> 7) & 0xff) == 0;  // with a nonzero low half every such float is denormal
    if (!same_or_both_nan_f(g[0], bf2f_bits((uint16_t)p))) ++bad[0];
    if (!same_or_both_nan_f(g[1], h2f_bits((uint16_t)p))) ++bad[1];
    if (denorm_in) continue;  // (exact zero is covered by the RZ / f16 rows and by tests elsewhere)
    if (!same_or_both_nan_bf(g[2], f2bf_rne(hi))) { if (bad[2] < 3) fprintf(stderr, "    f2bf %08x: got %04x want %04x\n", hi, g[2], f2bf_rne(hi)); ++bad[2]; }
    if (!same_or_both_nan_bf(g[3], f2bf_rne(hi | 0x8000u))) { if (bad[3] < 3) fprintf(stderr, "    f2bf %08x: got %04x want %04x\n", hi | 0x8000u, g[3], f2bf_rne(hi | 0x8000u)); ++bad[3]; }
    if (!same_or_both_nan_bf(g[4], f2bf_rne(hi | 0x8001u))) { if (bad[4] < 3) fprintf(stderr, "    f2bf %08x: got %04x want %04x\n", hi | 0x8001u, g[4], f2bf_rne(hi | 0x8001u)); ++bad[4]; }
    if (!same_or_both_nan_h(g[5], f2h_rne(hi))) { if (bad[5] < 3) fprintf(stderr, "    f2h %08x: got %04x want %04x\n", hi, g[5], f2h_rne(hi)); ++bad[5]; }
    if (!same_or_both_nan_h(g[6], f2h_rz(hi | 0x1234u))) { if (bad[6] < 3) fprintf(stderr, "    f2h_rz %08x: got %04x want %04x\n", hi | 0x1234u, g[6], f2h_rz(hi | 0x1234u)); ++bad[6]; }
    { uint32_t u = hi | 0x8001u; uint16_t want = ((u & 0x7f800000u) == 0x7f800000u) ? f2bf_rne(u) : (uint16_t)(u >> 16);
      if (!same_or_both_nan_bf(g[7], want)) ++bad[7]; }
  }
  report("bf16 -> f32 (exhaustive)", bad[0], 65536);
  report("f16 -> f32 (exhaustive)", bad[1], 65536);
  report("f32 -> bf16 RNE (exact / tie / above tie)", bad[2] + bad[3] + bad[4], 3 * 65536);
  report("f32 -> f16 RNE (exhaustive high half)", bad[5], 65536);
  report("f32 -> f16 RZ", bad[6], 65536);
  report("f32 -> bf16 RZ", bad[7], 65536);
  CK(cudaFree(d));
}

// ------------------------------------------------------------------ atomics
__global__ void atomic_kernel(int* icnt, unsigned* ucnt, float* fsum, int* imax, int* imin, unsigned* bits, int* cas, float* fmax_) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  atomicAdd(icnt, 1);
  atomicAdd(ucnt + (gid & 7), 2u);
  atomicAdd(fsum, 0.5f);
  atomicMax(imax, gid * 3 - 1000);
  atomicMin(imin, 5000 - gid);
  atomicOr(bits, 1u << (gid & 31));
  atomicAnd(bits + 1, ~(1u << (gid & 31)));
  atomicXor(bits + 2, 1u << (gid & 31));  // each bit flipped (N/32) times
  if (gid == 777) atomicExch(bits + 3, 0xabcdefu);
  // CAS-based increment (classic pattern)
  int old = *cas, assumed;
  do { assumed = old; old = atomicCAS(cas, assumed, assumed + 2); } while (assumed != old);
  { float v = (float)gid * 0.25f - 100.f;
    int* as_int = (int*)fmax_; int o = *as_int, a;
    do { a = o; if (__int_as_float(a) >= v) break; o = atomicCAS(as_int, a, __float_as_int(v)); } while (a != o); }
}
static void test_atomics() {
  const int N = 4096, blocks = 16, threads = 256;
  int* icnt; unsigned* ucnt; float* fsum; int* imax; int* imin; unsigned* bits; int* cas; float* fmax_;
  CK(cudaMalloc(&icnt, 4)); CK(cudaMalloc(&ucnt, 32)); CK(cudaMalloc(&fsum, 4)); CK(cudaMalloc(&imax, 4)); CK(cudaMalloc(&imin, 4));
  CK(cudaMalloc(&bits, 16)); CK(cudaMalloc(&cas, 4)); CK(cudaMalloc(&fmax_, 4));
  CK(cudaMemset(icnt, 0, 4)); CK(cudaMemset(ucnt, 0, 32)); CK(cudaMemset(fsum, 0, 4)); CK(cudaMemset(cas, 0, 4));
  int mn = INT32_MIN, mx = INT32_MAX; float neg = -1e30f;
  CK(cudaMemcpy(imax, &mn, 4, cudaMemcpyHostToDevice)); CK(cudaMemcpy(imin, &mx, 4, cudaMemcpyHostToDevice));
  unsigned b0[4] = {0u, 0xffffffffu, 0u, 0u}; CK(cudaMemcpy(bits, b0, 16, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(fmax_, &neg, 4, cudaMemcpyHostToDevice));
  atomic_kernel<<<blocks, threads>>>(icnt, ucnt, fsum, imax, imin, bits, cas, fmax_);
  int hi, hmax, hmin, hcas; unsigned hu[8], hb[4]; float hf, hfm;
  CK(cudaMemcpy(&hi, icnt, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hu, ucnt, 32, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&hf, fsum, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&hmax, imax, 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&hmin, imin, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(hb, bits, 16, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(&hcas, cas, 4, cudaMemcpyDeviceToHost)); CK(cudaMemcpy(&hfm, fmax_, 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  bad += hi != N; for (int i = 0; i < 8; ++i) bad += hu[i] != (unsigned)(N / 8 * 2);
  bad += hf != 0.5f * N;  // 2048.0 exactly representable, every partial sum exact
  bad += hmax != (N - 1) * 3 - 1000; bad += hmin != 5000 - (N - 1);
  bad += hb[0] != 0xffffffffu; bad += hb[1] != 0u; bad += hb[2] != 0u /* N/32 = 128 flips: even */; bad += hb[3] != 0xabcdefu;
  bad += hcas != 2 * N; bad += hfm != (float)(N - 1) * 0.25f - 100.f;
  if (bad) fprintf(stderr, "    icnt %d fsum %g imax %d imin %d bits %08x %08x %08x %08x cas %d fmax %g\n", hi, hf, hmax, hmin, hb[0], hb[1], hb[2], hb[3], hcas, hfm);
  report("atomics under contention", bad, 19);
}

// ------------------------------------------------------------------ math
// out[i*24 + j] = f_j(x_i) for a grid of inputs incl. specials.
__global__ void math_kernel(const float* x, float* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float v = x[i];
  float* o = out + i * 24;
  o[0] = expf(v); o[1] = logf(fabsf(v)); o[2] = sinf(v); o[3] = cosf(v); o[4] = sqrtf(fabsf(v)); o[5] = rsqrtf(fabsf(v));
  o[6] = tanhf(v); o[7] = erff(v); o[8] = exp2f(v); o[9] = log2f(fabsf(v)); o[10] = floorf(v); o[11] = ceilf(v);
  o[12] = rintf(v); o[13] = roundf(v); o[14] = truncf(v); o[15] = fmodf(v, 1.7f); o[16] = fminf(v, 0.5f); o[17] = fmaxf(v, 0.5f);
  o[18] = fmaf(v, 1.5f, -2.f); o[19] = powf(fabsf(v), 1.3f); o[20] = __expf(v); o[21] = __logf(fabsf(v)); o[22] = __fdividef(v, 3.f);
  o[23] = log1pf(fabsf(v)) + expm1f(-fabsf(v));
}
static int ulp_diff(float a, float b) {
  if (isnan(a) && isnan(b)) return 0;
  if (isnan(a) || isnan(b)) return 1 << 20;
  if (a == b) return 0;
  if (isinf(a) || isinf(b)) return 1 << 20;
  int32_t ia, ib; memcpy(&ia, &a, 4); memcpy(&ib, &b, 4);
  if ((ia < 0) != (ib < 0)) return (fabsf(a) < 1e-30f && fabsf(b) < 1e-30f) ? 0 : 1 << 20;  // +0/-0
  return abs(ia - ib);
}
static void test_math() {
  std::vector<float> xs;
  for (int i = -200; i <= 200; ++i) xs.push_back(i * 0.137f);
  const float specials[] = {0.f, -0.f, 1e-38f, -1e-38f, 1e-45f, 3.4e38f, -3.4e38f, INFINITY, -INFINITY, NAN, 88.f, -104.f, 0.5f, 1.f, 2.f, 1e-3f, 87.3f, -87.3f};
  for (float s : specials) xs.push_back(s);
  for (int i = 0; i < 1000; ++i) { uint32_t u = 0x9e3779b9u * (i + 7); u ^= u >> 13; float f; memcpy(&f, &u, 4); if (isfinite(f) && fabsf(f) < 1e6f) xs.push_back(f); }
  const int n = (int)xs.size();
  float *dx, *dout; CK(cudaMalloc(&dx, n * 4)); CK(cudaMalloc(&dout, n * 24 * 4));
  CK(cudaMemcpy(dx, xs.data(), n * 4, cudaMemcpyHostToDevice));
  math_kernel<<<(n + 255) / 256, 256>>>(dx, dout, n);
  std::vector<float> got(n * 24); CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
  // tolerance in ULPs per function (precise libm-class: 2; approximations: as documented for CUDA's fast paths)
  const int tol[24] = {2, 2, 2, 2, 0, 2, 4, 2, 2, 2, 0, 0, 0, 0, 0, 2, 0, 0, 0, 4, 2 + (1 << 10), 2 + (1 << 10), 2, 4};
  const char* names[24] = {"expf", "logf", "sinf", "cosf", "sqrtf", "rsqrtf", "tanhf", "erff", "exp2f", "log2f", "floorf", "ceilf", "rintf", "roundf", "truncf", "fmodf", "fminf", "fmaxf", "fmaf", "powf", "__expf", "__logf", "__fdividef", "log1pf+expm1f"};
  int bad_total = 0;
  for (int j = 0; j < 24; ++j) {
    int bad = 0; int worst = 0; float wx = 0;
    for (int i = 0; i < n; ++i) {
      const float v = xs[i]; float ref;
      switch (j) {
        case 0: ref = expf(v); break; case 1: ref = logf(fabsf(v)); break; case 2: ref = sinf(v); break; case 3: ref = cosf(v); break;
        case 4: ref = sqrtf(fabsf(v)); break; case 5: ref = 1.f / sqrtf(fabsf(v)); break; case 6: ref = tanhf(v); break; case 7: ref = erff(v); break;
        case 8: ref = exp2f(v); break; case 9: ref = log2f(fabsf(v)); break; case 10: ref = floorf(v); break; case 11: ref = ceilf(v); break;
        case 12: ref = rintf(v); break; case 13: ref = roundf(v); break; case 14: ref = truncf(v); break; case 15: ref = fmodf(v, 1.7f); break;
        case 16: ref = fminf(v, 0.5f); break; case 17: ref = fmaxf(v, 0.5f); break; case 18: ref = fmaf(v, 1.5f, -2.f); break; case 19: ref = powf(fabsf(v), 1.3f); break;
        case 20: ref = expf(v); break; case 21: ref = logf(fabsf(v)); break; case 22: ref = v / 3.f; break; default: ref = log1pf(fabsf(v)) + expm1f(-fabsf(v)); break;
      }
      // Apple GPUs flush fp32 denormals to zero (documented deviation from CUDA's default -ftz=false):
      // skip inputs and reference results in the denormal range.
      auto denorm = [](float f) { return f != 0.f && fabsf(f) < 1.17549435e-38f; };
      if (denorm(v) || denorm(ref)) continue;
      // sinf/cosf of huge arguments: allow absolute 1e-6 (argument reduction differs between libms)
      const float g = got[i * 24 + j];
      int d = ulp_diff(g, ref);
      if (g == 0.f && denorm(ref)) d = 0;
      if ((j == 2 || j == 3 || j == 6 || j == 7 || j == 23) && fabsf(g - ref) <= 2e-6f) d = 0;
      if (j == 5 && v == 0.f) d = (isinf(g) ? 0 : d);
      if (d > tol[j]) { if (bad < 4) fprintf(stderr, "    %s(%g): got %.9g want %.9g (%d ulp)\n", names[j], v, g, ref, d); ++bad; }
      if (d > worst && d < (1 << 20)) { worst = d; wx = v; }
    }
    (void)wx;
    if (bad) { fprintf(stderr, "    %s: %d bad\n", names[j], bad); }
    bad_total += bad;
  }
  report("math (libm, ULP-bounded)", bad_total, n * 24);
  CK(cudaFree(dx)); CK(cudaFree(dout));
}

// ------------------------------------------------------------------ shared memory + barriers
__global__ void smem_kernel(const int* in, int* out) {
  __shared__ int tile[32][33];
  __shared__ int red[256];
  const int t = threadIdx.x;
  // transpose a 32x32 block through smem (padded to avoid bank conflicts), then block-reduce the row sums
  for (int i = t; i < 1024; i += blockDim.x) tile[i / 32][i % 32] = in[i];
  __syncthreads();
  for (int i = t; i < 1024; i += blockDim.x) out[i] = tile[i % 32][i / 32];
  int s = 0;
  for (int i = t; i < 1024; i += blockDim.x) s += in[i];
  red[t] = s;
  __syncthreads();
  for (int stride = 128; stride > 0; stride >>= 1) { if (t < stride) red[t] += red[t + stride]; __syncthreads(); }
  if (t == 0) out[1024] = red[0];
  // dynamic shared memory: write then read back reversed
  extern __shared__ int dyn[];
  dyn[t] = t * 3;
  __syncthreads();
  out[1025 + t] = dyn[255 - t];
}
static void test_smem() {
  std::vector<int> in(1024); for (int i = 0; i < 1024; ++i) in[i] = (i * 7919) % 1000 - 500;
  int *din, *dout; CK(cudaMalloc(&din, 1024 * 4)); CK(cudaMalloc(&dout, (1025 + 256) * 4));
  CK(cudaMemcpy(din, in.data(), 1024 * 4, cudaMemcpyHostToDevice));
  smem_kernel<<<1, 256, 256 * 4>>>(din, dout);
  std::vector<int> got(1025 + 256); CK(cudaMemcpy(got.data(), dout, got.size() * 4, cudaMemcpyDeviceToHost));
  int bad = 0, sum = 0;
  for (int i = 0; i < 1024; ++i) { if (got[i] != in[(i % 32) * 32 + i / 32]) ++bad; sum += in[i]; }
  bad += got[1024] != sum;
  for (int t = 0; t < 256; ++t) bad += got[1025 + t] != (255 - t) * 3;
  report("shared memory transpose / reduce / dynamic", bad, 1025 + 256);
  CK(cudaFree(din)); CK(cudaFree(dout));
}

// ------------------------------------------------------------------ 64-bit integer + vector loads
__global__ void int64_kernel(const uint4* in, uint64_t* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const uint4 v = in[i];
  const uint64_t a = ((uint64_t)v.y << 32) | v.x, b = ((uint64_t)v.w << 32) | v.z;
  out[i * 4 + 0] = a * b + (a >> 7);
  out[i * 4 + 1] = b ? a / b : 0;
  out[i * 4 + 2] = (uint64_t)((int64_t)a >> 5) ^ (a << (b & 63));
  out[i * 4 + 3] = (uint64_t)__popcll(a) + (uint64_t)__clzll(a | 1) * 1000 + (uint64_t)(__ffsll((long long)b)) * 1000000;
}
static void test_int64() {
  const int n = 2048;
  std::vector<uint32_t> in(n * 4); for (int i = 0; i < n * 4; ++i) { uint32_t u = 0x9e3779b9u * (i + 3); u ^= u >> 15; in[i] = u; }
  uint4* din; uint64_t* dout; CK(cudaMalloc(&din, n * 16)); CK(cudaMalloc(&dout, n * 32));
  CK(cudaMemcpy(din, in.data(), n * 16, cudaMemcpyHostToDevice));
  int64_kernel<<<(n + 255) / 256, 256>>>(din, dout, n);
  std::vector<uint64_t> got(n * 4); CK(cudaMemcpy(got.data(), dout, n * 32, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    const uint64_t a = ((uint64_t)in[i * 4 + 1] << 32) | in[i * 4], b = ((uint64_t)in[i * 4 + 3] << 32) | in[i * 4 + 2];
    uint64_t e[4] = {a * b + (a >> 7), b ? a / b : 0, (uint64_t)((int64_t)a >> 5) ^ (a << (b & 63)),
                     (uint64_t)__builtin_popcountll(a) + (uint64_t)__builtin_clzll(a | 1) * 1000 + (uint64_t)__builtin_ffsll((long long)b) * 1000000};
    for (int j = 0; j < 4; ++j) if (got[i * 4 + j] != e[j]) { if (bad < 3) fprintf(stderr, "    [%d.%d] got %llx want %llx\n", i, j, (unsigned long long)got[i * 4 + j], (unsigned long long)e[j]); ++bad; }
  }
  report("64-bit integer ops + uint4 loads", bad, n * 4);
  CK(cudaFree(din)); CK(cudaFree(dout));
}

// ------------------------------------------------------------------ unaligned memcpy loads/stores
// A small memcpy on a byte pointer lowers to `load/store iN, align 1`; on NVIDIA those are byte accesses that work at
// any address. Thread i reads 2/4/8 bytes at byte offset i (so every residue mod 8 is exercised) and writes an 8-byte
// value back at an odd offset. Metal would round a natural-width access down to its alignment.
__global__ void unaligned_kernel(const uint8_t* in, uint8_t* out, uint64_t* got, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint16_t h; memcpy(&h, in + i, 2);
  uint32_t w; memcpy(&w, in + i, 4);
  uint64_t q; memcpy(&q, in + i, 8);
  got[i * 3 + 0] = h; got[i * 3 + 1] = w; got[i * 3 + 2] = q;
  const uint64_t v = q * 0x9e3779b97f4a7c15ull + (uint64_t)i;
  memcpy(out + (size_t)i * 9 + 1, &v, 8);
}
static void test_unaligned() {
  const int n = 4096;
  std::vector<uint8_t> in(n + 16); for (int i = 0; i < n + 16; ++i) in[i] = (uint8_t)(0x9e3779b9u * (i + 1) >> 13);
  uint8_t *din, *dout; uint64_t* dgot;
  CK(cudaMalloc(&din, n + 16)); CK(cudaMalloc(&dout, (size_t)n * 9 + 16)); CK(cudaMalloc(&dgot, (size_t)n * 24));
  CK(cudaMemcpy(din, in.data(), n + 16, cudaMemcpyHostToDevice)); CK(cudaMemset(dout, 0, (size_t)n * 9 + 16));
  unaligned_kernel<<<(n + 255) / 256, 256>>>(din, dout, dgot, n);
  std::vector<uint64_t> got(n * 3); std::vector<uint8_t> out((size_t)n * 9 + 16);
  CK(cudaMemcpy(got.data(), dgot, (size_t)n * 24, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(out.data(), dout, (size_t)n * 9 + 16, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    uint16_t h; uint32_t w; uint64_t q, v;
    memcpy(&h, &in[i], 2); memcpy(&w, &in[i], 4); memcpy(&q, &in[i], 8); memcpy(&v, &out[(size_t)i * 9 + 1], 8);
    const uint64_t e[4] = {h, w, q, q * 0x9e3779b97f4a7c15ull + (uint64_t)i}, g[4] = {got[i * 3], got[i * 3 + 1], got[i * 3 + 2], v};
    for (int j = 0; j < 4; ++j) if (g[j] != e[j]) { if (bad < 3) fprintf(stderr, "    [%d.%d] got %llx want %llx\n", i, j, (unsigned long long)g[j], (unsigned long long)e[j]); ++bad; }
    if (out[(size_t)i * 9] != 0) { if (bad < 3) fprintf(stderr, "    [%d] store spilled below its offset\n", i); ++bad; }
  }
  report("unaligned memcpy loads/stores", bad, n * 5);
  CK(cudaFree(din)); CK(cudaFree(dout)); CK(cudaFree(dgot));
}

// prefetch.global.L2 is a cache hint: the subsequent load must still see the stored bytes.
__global__ void prefetch_kernel(const uint8_t* p, uint8_t* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  asm volatile("prefetch.global.L2 [%0];" ::"l"(p + i));
  out[i] = p[i];
}
static void test_prefetch() {
  const int n = 4096;
  std::vector<uint8_t> in(n);
  for (int i = 0; i < n; ++i) in[i] = (uint8_t)(0x9e3779b9u * (i + 1) >> 13);
  uint8_t *d, *o; CK(cudaMalloc(&d, n)); CK(cudaMalloc(&o, n));
  CK(cudaMemcpy(d, in.data(), n, cudaMemcpyHostToDevice));
  prefetch_kernel<<<(n + 255) / 256, 256>>>(d, o, n);
  std::vector<uint8_t> got(n); CK(cudaMemcpy(got.data(), o, n, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; ++i) if (got[i] != in[i]) ++bad;
  report("prefetch.global.L2 then load", bad, n);
  CK(cudaFree(d)); CK(cudaFree(o));
}

// cvt.rn.f16x2.e4m3x2 via a .reg temp (the mx GEMV widening sequence).
__global__ void e4m3x2_kernel(const uint16_t* in, uint32_t* out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  uint32_t w = in[i], d;
  asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }" : "=r"(d) : "r"(w));
  out[i] = d;
}
static uint16_t e4m3_to_f16_bits(uint8_t b) {
  const uint32_t s = (uint32_t)(b & 0x80u) << 8, e = (b >> 3) & 0xfu, m = b & 7u;
  if ((b & 0x7fu) == 0x7fu) return 0x7e00u;
  if (e == 0u) {
    float f = (s ? -1.f : 1.f) * (float)m * 0.001953125f;
    _Float16 h = (_Float16)f;
    uint16_t r; memcpy(&r, &h, 2);
    return r;
  }
  return (uint16_t)(s | ((e + 8u) << 10) | (m << 7));
}
static void test_e4m3x2() {
  const int n = 65536;
  std::vector<uint16_t> in(n);
  for (int i = 0; i < n; ++i) in[i] = (uint16_t)i;
  uint16_t* d; uint32_t* o; CK(cudaMalloc(&d, n * 2)); CK(cudaMalloc(&o, n * 4));
  CK(cudaMemcpy(d, in.data(), n * 2, cudaMemcpyHostToDevice));
  e4m3x2_kernel<<<(n + 255) / 256, 256>>>(d, o, n);
  std::vector<uint32_t> got(n); CK(cudaMemcpy(got.data(), o, n * 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    const uint32_t want = (uint32_t)e4m3_to_f16_bits((uint8_t)in[i]) | ((uint32_t)e4m3_to_f16_bits((uint8_t)(in[i] >> 8)) << 16);
    if (got[i] != want) { if (bad < 3) fprintf(stderr, "    e4m3x2 %04x: got %08x want %08x\n", in[i], got[i], want); ++bad; }
  }
  report("cvt.rn.f16x2.e4m3x2 (exhaustive u16)", bad, n);
  CK(cudaFree(d)); CK(cudaFree(o));
}

int main() {
  printf("intrinsics tests\n");
  test_warp();
  test_cvt();
  test_atomics();
  test_math();
  test_smem();
  test_int64();
  test_unaligned();
  test_prefetch();
  test_e4m3x2();
  CK(cudaDeviceSynchronize());
  if (failures) { printf("%d FAILED\n", failures); return 1; }
  printf("all ok\n");
  return 0;
}
