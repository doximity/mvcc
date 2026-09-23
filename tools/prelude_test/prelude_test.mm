// mvcc-prelude-test: runs every prelude helper on the GPU against a CPU reference. This is the correctness gate for
// the mma/ldmatrix/shfl lowering; run it after any change to msl/mvcc_prelude.metal or the emitter's PTX lowering.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static id<MTLDevice> dev;
static id<MTLCommandQueue> q;
static int failures = 0;

static uint16_t f2h(float f) {
  uint32_t x; memcpy(&x, &f, 4);
  uint32_t s = (x >> 16) & 0x8000, e = (x >> 23) & 0xff, m = x & 0x7fffff;
  if (e == 0xff) return s | 0x7c00 | (m ? 0x200 : 0);
  int ne = (int)e - 127 + 15;
  if (ne >= 31) return s | 0x7c00;
  if (ne <= 0) {
    if (ne < -10) return s;
    m |= 0x800000; int sh = 14 - ne;
    uint32_t r = m >> sh, rem = m & ((1u << sh) - 1), half = 1u << (sh - 1);
    if (rem > half || (rem == half && (r & 1))) r++;
    return s | r;
  }
  uint32_t r = s | (ne << 10) | (m >> 13), rem = m & 0x1fff;
  if (rem > 0x1000 || (rem == 0x1000 && (r & 1))) r++;
  return r;
}
static float h2f(uint16_t h) {
  uint32_t s = (h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, x;
  if (e == 0) { if (!m) x = s; else { float f = std::ldexp((float)m, -24); memcpy(&x, &f, 4); x |= s; } }
  else if (e == 31) x = s | 0x7f800000 | (m << 13);
  else x = s | ((e + 112) << 23) | (m << 13);
  float f; memcpy(&f, &x, 4); return f;
}
static float bf2f(uint16_t b) { uint32_t x = (uint32_t)b << 16; float f; memcpy(&f, &x, 4); return f; }
static uint16_t f2bf(float f) { uint32_t x; memcpy(&x, &f, 4); uint32_t r = x + 0x7fff + ((x >> 16) & 1); return r >> 16; }
static float e4m3f(uint8_t b) {
  int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  if ((b & 0x7f) == 0x7f) return NAN;
  float v = e == 0 ? std::ldexp((float)m, -9) : std::ldexp(1.f + m / 8.f, e - 7);
  return s ? -v : v;
}
static float e5m2f(uint8_t b) { return h2f((uint16_t)b << 8); }

static id<MTLLibrary> compile(const std::string& src) {
  NSError* err = nil;
  MTLCompileOptions* o = [MTLCompileOptions new];
  o.languageVersion = MTLLanguageVersion4_0;
  o.mathMode = MTLMathModeSafe;
  id<MTLLibrary> lib = [dev newLibraryWithSource:[NSString stringWithUTF8String:src.c_str()] options:o error:&err];
  if (!lib) { fprintf(stderr, "COMPILE ERROR:\n%s\n", err.localizedDescription.UTF8String); exit(1); }
  return lib;
}
// run kernel `name` with one threadgroup of 32 threads; buffers are bound in order
static void run(id<MTLLibrary> lib, const char* name, std::vector<id<MTLBuffer>> bufs, unsigned threads = 32) {
  NSError* err = nil;
  id<MTLFunction> fn = [lib newFunctionWithName:[NSString stringWithUTF8String:name]];
  if (!fn) { fprintf(stderr, "no kernel %s\n", name); exit(1); }
  id<MTLComputePipelineState> ps = [dev newComputePipelineStateWithFunction:fn error:&err];
  if (!ps) { fprintf(stderr, "%s: %s\n", name, err.localizedDescription.UTF8String); exit(1); }
  id<MTLCommandBuffer> cb = [q commandBuffer];
  id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
  [enc setComputePipelineState:ps];
  for (size_t i = 0; i < bufs.size(); i++) [enc setBuffer:bufs[i] offset:0 atIndex:i];
  [enc setThreadgroupMemoryLength:4096 atIndex:0];
  [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
  [enc endEncoding];
  [cb commit]; [cb waitUntilCompleted];
  if (cb.error) { fprintf(stderr, "%s: GPU error %s\n", name, cb.error.localizedDescription.UTF8String); exit(1); }
}
static id<MTLBuffer> buf(const void* data, size_t n) {
  id<MTLBuffer> b = [dev newBufferWithLength:n options:MTLResourceStorageModeShared];
  if (data) memcpy(b.contents, data, n); else memset(b.contents, 0, n);
  return b;
}
static void check(const char* what, bool ok, const char* detail = "") {
  printf("%-48s %s %s\n", what, ok ? "[ok]" : "[FAIL]", detail);
  if (!ok) failures++;
}

// ------------------------------------------------------------------------------------------------
// mma: for each variant, random A (16xK), B (Kx8), C (16x8) in PTX fragment layout, compare D with reference.
// ------------------------------------------------------------------------------------------------
struct MmaVariant { const char* helper; int K; int elemBits; const char* a; const char* b; const char* c; bool s32; bool f16acc; };

static void test_mma(id<MTLLibrary> lib, const MmaVariant& v, std::mt19937& rng) {
  const int M = 16, N = 8, K = v.K;
  std::vector<float> A(M * K), B(K * N), C(M * N), D(M * N, 0);
  // raw fragments: 32 lanes x (aRegs uints)
  int aRegs = (M * K * v.elemBits / 8) / (32 * 4), bRegs = (K * N * v.elemBits / 8) / (32 * 4), cRegs = v.f16acc ? 2 : 4;
  std::vector<uint32_t> fa(32 * aRegs), fb(32 * bRegs), fc(32 * cRegs), fd(32 * cRegs);
  std::uniform_int_distribution<int> di(-4, 4);
  // fill logical matrices with values representable in every element type
  auto encode = [&](float x, std::string t) -> uint32_t {
    if (t == "f16") return f2h(x);
    if (t == "bf16") return f2bf(x);
    if (t == "s8") return (uint8_t)(int8_t)x;
    if (t == "u8") return (uint8_t)x;
    if (t == "e4m3") { int s = x < 0; float ax = std::fabs(x); if (ax == 0) return s << 7; int e = (int)std::floor(std::log2(ax)); int m = (int)std::round((ax / std::ldexp(1.f, e) - 1) * 8); return (s << 7) | ((e + 7) << 3) | m; }
    if (t == "e5m2") return f2h(x) >> 8;
    return 0;
  };
  auto decode = [&](uint32_t bits, std::string t) -> float {
    if (t == "f16") return h2f(bits); if (t == "bf16") return bf2f(bits); if (t == "s8") return (float)(int8_t)bits;
    if (t == "u8") return (float)(uint8_t)bits; if (t == "e4m3") return e4m3f(bits); if (t == "e5m2") return e5m2f(bits); return 0;
  };
  auto gen = [&](std::string t) -> float {
    if (t == "u8") return (float)std::uniform_int_distribution<int>(0, 255)(rng);
    if (t == "s8") return (float)std::uniform_int_distribution<int>(-128, 127)(rng);
    if (t == "e4m3" || t == "e5m2") { static const float vals[] = {0, 0.5f, 1, 1.5f, 2, 3, -1, -0.5f, -2, 0.25f, 4}; return vals[rng() % 11]; }
    return (float)di(rng);
  };
  for (auto& x : A) x = gen(v.a);
  for (auto& x : B) x = gen(v.b);
  for (auto& x : C) x = (float)di(rng);
  // encode fragments
  int epr = 32 / v.elemBits;  // elements per 32-bit register
  for (int lane = 0; lane < 32; lane++) {
    int g = lane >> 2, t = lane & 3;
    auto packA = [&](int reg, int row, int k0) { uint32_t w = 0; for (int i = 0; i < epr; i++) w |= encode(A[row * K + k0 + i], v.a) << (i * v.elemBits); fa[lane * aRegs + reg] = w; };
    auto packB = [&](int reg, int k0, int n) { uint32_t w = 0; for (int i = 0; i < epr; i++) w |= encode(B[(k0 + i) * N + n], v.b) << (i * v.elemBits); fb[lane * bRegs + reg] = w; };
    if (v.elemBits == 16) {
      if (K == 16) { packA(0, g, 2 * t); packA(1, g + 8, 2 * t); packA(2, g, 2 * t + 8); packA(3, g + 8, 2 * t + 8); packB(0, 2 * t, g); packB(1, 2 * t + 8, g); }
      else { packA(0, g, 2 * t); packA(1, g + 8, 2 * t); packB(0, 2 * t, g); }
    } else {
      if (K == 32) { packA(0, g, 4 * t); packA(1, g + 8, 4 * t); packA(2, g, 4 * t + 16); packA(3, g + 8, 4 * t + 16); packB(0, 4 * t, g); packB(1, 4 * t + 16, g); }
      else { packA(0, g, 4 * t); packA(1, g + 8, 4 * t); packB(0, 4 * t, g); }
    }
    if (v.f16acc) {
      fc[lane * 2 + 0] = f2h(C[g * N + 2 * t]) | (f2h(C[g * N + 2 * t + 1]) << 16);
      fc[lane * 2 + 1] = f2h(C[(g + 8) * N + 2 * t]) | (f2h(C[(g + 8) * N + 2 * t + 1]) << 16);
    } else {
      float cs[4] = {C[g * N + 2 * t], C[g * N + 2 * t + 1], C[(g + 8) * N + 2 * t], C[(g + 8) * N + 2 * t + 1]};
      for (int i = 0; i < 4; i++) { if (v.s32) fc[lane * 4 + i] = (uint32_t)(int32_t)cs[i]; else memcpy(&fc[lane * 4 + i], &cs[i], 4); }
    }
  }
  // reference
  for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
    float acc = 0; for (int k = 0; k < K; k++) acc += decode(encode(A[m * K + k], v.a), v.a) * decode(encode(B[k * N + n], v.b), v.b);
    D[m * N + n] = acc + C[m * N + n];
  }
  id<MTLBuffer> ba = buf(fa.data(), fa.size() * 4), bb = buf(fb.data(), fb.size() * 4), bc = buf(fc.data(), fc.size() * 4), bd = buf(nullptr, fd.size() * 4);
  std::string kname = std::string("k_") + v.helper;
  run(lib, kname.c_str(), {ba, bb, bc, bd});
  memcpy(fd.data(), bd.contents, fd.size() * 4);
  int bad = 0; char detail[128] = "";
  for (int lane = 0; lane < 32; lane++) {
    int g = lane >> 2, t = lane & 3;
    int idx[4] = {g * N + 2 * t, g * N + 2 * t + 1, (g + 8) * N + 2 * t, (g + 8) * N + 2 * t + 1};
    for (int i = 0; i < 4; i++) {
      float got;
      if (v.f16acc) got = h2f((fd[lane * 2 + i / 2] >> (16 * (i & 1))) & 0xffff);
      else if (v.s32) got = (float)(int32_t)fd[lane * 4 + i];
      else memcpy(&got, &fd[lane * 4 + i], 4);
      float want = D[idx[i]];
      float tol = v.f16acc ? 0.05f * std::max(1.f, std::fabs(want)) : 1e-3f;
      if (std::fabs(got - want) > tol) { if (!bad) snprintf(detail, sizeof detail, "lane %d reg %d: got %g want %g", lane, i, got, want); bad++; }
    }
  }
  check(v.helper, bad == 0, detail);
}

int main() {
  @autoreleasepool {
    dev = MTLCreateSystemDefaultDevice(); q = [dev newCommandQueue];
    NSString* preludePath = [NSString stringWithUTF8String:PRELUDE_PATH];
    NSError* err = nil;
    NSString* prelude = [NSString stringWithContentsOfFile:preludePath encoding:NSUTF8StringEncoding error:&err];
    if (!prelude) { fprintf(stderr, "cannot read prelude %s\n", PRELUDE_PATH); return 1; }
    std::string src = "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n"
      "#define MVCC_NEED_shfl 1\n#define MVCC_NEED_unaligned 1\n#define MVCC_NEED_atomic_fmaxmin 1\n#define MVCC_NEED_bits 1\n"
      "#define MVCC_NEED_prmt 1\n#define MVCC_NEED_mem 1\n#define MVCC_NEED_cpasync 1\n#define MVCC_NEED_ldmatrix 1\n#define MVCC_NEED_mma 1\n";
    src += prelude.UTF8String;
    // --- test kernels ---
    src += R"MSL(
kernel void k_shfl(device uint* in [[buffer(0)]], device uint* out [[buffer(1)]], device uint* params [[buffer(2)]], uint lane [[thread_index_in_simdgroup]]) {
  uint v = in[lane], b = params[0], c = params[1], mode = params[2];
  uint r = mode == 0 ? __mvcc_shfl_idx(v, b, c, lane) : mode == 1 ? __mvcc_shfl_up(v, b, c, lane) : mode == 2 ? __mvcc_shfl_down(v, b, c, lane) : __mvcc_shfl_bfly(v, b, c, lane);
  out[lane] = r;
}
kernel void k_ldmatrix(device ushort* in [[buffer(0)]], device uint* out [[buffer(1)]], threadgroup uchar* smem [[threadgroup(0)]], uint lane [[thread_index_in_simdgroup]]) {
  // 4 matrices of 8x8 b16, stored contiguously row-major with row stride 16 elements (32 bytes) to exercise strides
  for (uint i = lane; i < 4 * 8 * 16; i += 32) ((threadgroup ushort*)smem)[i] = in[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint mat = lane >> 3, row = lane & 7;
  threadgroup uchar* rowp = smem + (mat * 8 + row) * 32;
  ulong a = (ulong)rowp;
  for (uint i = 0; i < 4; i++) out[i * 32 + lane] = __mvcc_ldmatrix_row(a, i, lane);
  for (uint i = 0; i < 4; i++) out[128 + i * 32 + lane] = __mvcc_ldmatrix_trans(a, i, lane);
}
kernel void k_prmt(device uint* in [[buffer(0)]], device uint* out [[buffer(1)]], uint lane [[thread_index_in_simdgroup]]) {
  out[lane] = __mvcc_prmt(in[0], in[1], in[2 + lane]);
}
kernel void k_atomic_fmax(device float* p [[buffer(0)]], device float* in [[buffer(1)]], uint lane [[thread_index_in_simdgroup]]) {
  __mvcc_atomic_fmax((device atomic_float*)p, in[lane]);
  __mvcc_atomic_fmin((device atomic_float*)(p + 1), in[lane]);
}
kernel void k_bits(device uint* out [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
  out[0] = __mvcc_bswap(0x11223344u); out[1] = __mvcc_fshl(0x80000001u, 0xffffffffu, 4u); out[2] = __mvcc_fshr(0x80000001u, 0x0000000fu, 4u);
  ulong b = __mvcc_bswap(0x1122334455667788ul); out[3] = (uint)b; out[4] = (uint)(b >> 32);
}
kernel void k_mem(device uint* out [[buffer(0)]], threadgroup uchar* smem [[threadgroup(0)]], uint lane [[thread_index_in_simdgroup]]) {
  if (lane == 0) {
    uchar tmp[37]; for (uint i = 0; i < 37; i++) tmp[i] = (uchar)(i * 7 + 1);
    __mvcc_memcpy(smem + 3, (thread uchar*)tmp, 37ul);
    __mvcc_memset(smem + 64, (uchar)0xab, 9ul);
    __mvcc_memcpy((device uchar*)out, smem, 128ul);
  }
}
)MSL";
    auto mmaKernel = [&](const char* helper, const char* aT, const char* bT, const char* cT) {
      char k[1024];
      snprintf(k, sizeof k,
        "kernel void k_%s(device %s* A [[buffer(0)]], device %s* B [[buffer(1)]], device %s* C [[buffer(2)]], device %s* D [[buffer(3)]], uint lane [[thread_index_in_simdgroup]]) {\n"
        "  %s d; %s(d, A[lane], B[lane], C[lane], lane); D[lane] = d; }\n", helper, aT, bT, cT, cT, cT, helper);
      src += k;
    };
    std::vector<MmaVariant> variants = {
      {"__mvcc_mma_m16n8k16_f32_f16_f16_f32", 16, 16, "f16", "f16", "f32", false, false},
      {"__mvcc_mma_m16n8k16_f32_bf16_bf16_f32", 16, 16, "bf16", "bf16", "f32", false, false},
      {"__mvcc_mma_m16n8k16_f16_f16_f16_f16", 16, 16, "f16", "f16", "f16", false, true},
      {"__mvcc_mma_m16n8k8_f32_f16_f16_f32", 8, 16, "f16", "f16", "f32", false, false},
      {"__mvcc_mma_m16n8k8_f32_bf16_bf16_f32", 8, 16, "bf16", "bf16", "f32", false, false},
      {"__mvcc_mma_m16n8k32_s32_s8_s8_s32", 32, 8, "s8", "s8", "s32", true, false},
      {"__mvcc_mma_m16n8k32_s32_u8_u8_s32", 32, 8, "u8", "u8", "s32", true, false},
      {"__mvcc_mma_m16n8k16_s32_s8_s8_s32", 16, 8, "s8", "s8", "s32", true, false},
      {"__mvcc_mma_m16n8k32_f32_e4m3_e4m3_f32", 32, 8, "e4m3", "e4m3", "f32", false, false},
      {"__mvcc_mma_m16n8k32_f32_e5m2_e5m2_f32", 32, 8, "e5m2", "e5m2", "f32", false, false},
    };
    for (auto& v : variants) {
      int aRegs = (16 * v.K * v.elemBits / 8) / 128, bRegs = (v.K * 8 * v.elemBits / 8) / 128;
      std::string aT = aRegs == 4 ? "uint4" : "uint2", bT = bRegs == 2 ? "uint2" : "uint", cT = v.f16acc ? "uint2" : "uint4";
      mmaKernel(v.helper, aT.c_str(), bT.c_str(), cT.c_str());
    }
    id<MTLLibrary> lib = compile(src);
    std::mt19937 rng(1234);

    // shfl: compare with PTX semantics reference
    {
      std::vector<uint32_t> in(32); for (int i = 0; i < 32; i++) in[i] = 1000 + i;
      struct Case { uint32_t b, c, mode; const char* name; } cases[] = {
        {5, 0x1f, 0, "shfl.idx full"}, {3, 0x1f, 1 | 0, "shfl.up full (c=0)"}, {3, 0x1f, 2, "shfl.down full"}, {16, 0x1f, 3, "shfl.bfly 16"},
        {1, 0x1f, 3, "shfl.bfly 1"}, {2, 0x1f, 0, "shfl.idx 2"}, {1, 0x1f | (16 << 8), 3, "shfl.bfly width16"}, {2, 0x1f | (24 << 8), 0, "shfl.idx width8"},
        {2, 0x1f | (16 << 8), 2, "shfl.down width16"}, {2, 0 | (16 << 8), 1, "shfl.up width16"}, {40, 0x1f, 0, "shfl.idx b>31 wraps"},
      };
      for (auto& cs : cases) {
        uint32_t params[3] = {cs.b, cs.c, cs.mode};
        if (cs.mode == 1) params[1] = cs.c & ~0x1fu;  // up uses clamp 0
        id<MTLBuffer> bi = buf(in.data(), 128), bo = buf(nullptr, 128), bp = buf(params, 12);
        run(lib, "k_shfl", {bi, bo, bp});
        int bad = 0;
        for (uint32_t lane = 0; lane < 32; lane++) {
          uint32_t c = params[1], segmask = (c >> 8) & 0x1f, cval = c & 0x1f, bval = cs.b & 0x1f;
          uint32_t maxLane = (lane & segmask) | (cval & ~segmask), minLane = lane & segmask; int j; bool pval;
          switch (cs.mode) {
            case 0: j = minLane | (bval & ~segmask); pval = (uint32_t)j <= maxLane; break;
            case 1: j = (int)lane - (int)bval; pval = j >= (int)maxLane; break;
            case 2: j = lane + bval; pval = (uint32_t)j <= maxLane; break;
            default: j = lane ^ bval; pval = (uint32_t)j <= maxLane; break;
          }
          if (!pval) j = lane;
          if (((uint32_t*)bo.contents)[lane] != in[j]) bad++;
        }
        check(cs.name, bad == 0);
      }
    }
    // ldmatrix
    {
      std::vector<uint16_t> in(4 * 8 * 16); for (size_t i = 0; i < in.size(); i++) in[i] = (uint16_t)(i * 3 + 7);
      id<MTLBuffer> bi = buf(in.data(), in.size() * 2), bo = buf(nullptr, 256 * 4);
      run(lib, "k_ldmatrix", {bi, bo});
      uint32_t* o = (uint32_t*)bo.contents; int badR = 0, badT = 0;
      for (int i = 0; i < 4; i++) for (int lane = 0; lane < 32; lane++) {
        int g = lane >> 2, t = lane & 3;
        auto el = [&](int r, int c) { return (uint32_t)in[(i * 8 + r) * 16 + c]; };
        uint32_t wantR = el(g, 2 * t) | (el(g, 2 * t + 1) << 16), wantT = el(2 * t, g) | (el(2 * t + 1, g) << 16);
        if (o[i * 32 + lane] != wantR) badR++;
        if (o[128 + i * 32 + lane] != wantT) badT++;
      }
      check("ldmatrix x4 row", badR == 0); check("ldmatrix x4 trans", badT == 0);
    }
    // prmt
    {
      std::vector<uint32_t> in = {0x33221100u, 0x77665544u};
      for (int i = 0; i < 32; i++) in.push_back(rng());
      id<MTLBuffer> bi = buf(in.data(), in.size() * 4), bo = buf(nullptr, 128);
      run(lib, "k_prmt", {bi, bo}); int bad = 0;
      for (int lane = 0; lane < 32; lane++) {
        uint32_t sel = in[2 + lane], r = 0; uint64_t all = ((uint64_t)in[1] << 32) | in[0];
        for (int i = 0; i < 4; i++) { uint32_t s = (sel >> (4 * i)) & 0xf, byte = (all >> (8 * (s & 7))) & 0xff; if (s & 8) byte = (byte & 0x80) ? 0xff : 0; r |= byte << (8 * i); }
        if (((uint32_t*)bo.contents)[lane] != r) bad++;
      }
      check("prmt / __byte_perm", bad == 0);
    }
    // atomic fmax/fmin
    {
      float vals[32]; float mx = -INFINITY, mn = INFINITY;
      for (int i = 0; i < 32; i++) { vals[i] = (float)std::uniform_real_distribution<float>(-100, 100)(rng); mx = std::max(mx, vals[i]); mn = std::min(mn, vals[i]); }
      float init[2] = {-50.f, 50.f}; mx = std::max(mx, init[0]); mn = std::min(mn, init[1]);
      id<MTLBuffer> bp = buf(init, 8), bi = buf(vals, 128);
      run(lib, "k_atomic_fmax", {bp, bi});
      float* r = (float*)bp.contents;
      check("atomic fmax/fmin", r[0] == mx && r[1] == mn);
    }
    // bits
    {
      id<MTLBuffer> bo = buf(nullptr, 64); run(lib, "k_bits", {bo}); uint32_t* o = (uint32_t*)bo.contents;
      uint64_t bs = __builtin_bswap64(0x1122334455667788ull);
      check("bswap/fshl/fshr", o[0] == 0x44332211u && o[1] == 0x0000001fu && o[2] == 0x10000000u && o[3] == (uint32_t)bs && o[4] == (uint32_t)(bs >> 32),
            (std::to_string(o[1]) + " " + std::to_string(o[2])).c_str());
    }
    // mem
    {
      id<MTLBuffer> bo = buf(nullptr, 128); run(lib, "k_mem", {bo}); uint8_t* o = (uint8_t*)bo.contents; int bad = 0;
      for (int i = 0; i < 37; i++) if (o[3 + i] != (uint8_t)(i * 7 + 1)) bad++;
      for (int i = 0; i < 9; i++) if (o[64 + i] != 0xab) bad++;
      if (o[73] != 0 || o[0] != 0) bad++;
      check("memcpy/memset", bad == 0);
    }
    // mma variants, several random seeds each
    for (auto& v : variants) for (int rep = 0; rep < 3; rep++) test_mma(lib, v, rng);

    printf("\n%d failure(s)\n", failures);
    return failures ? 1 : 0;
  }
}
