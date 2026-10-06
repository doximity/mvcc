// mma.sync fed by plain 32-bit loads, as hand-written kernels write it: no cp.async ring, no ldmatrix. Each lane
// loads its fragment words straight from device memory (or from a shared-memory tile) at its warp's tile origin, and
// recovery proves the words form one tile of that memory (matmul2d ..._dev, ..._dev_tgA).
//   gemm_dev_kernel   C = A B^T, bf16 m16n8k16 or s8 m16n8k32, 64-row blocks of 2 x WN warps of 32 x 32 tiles, K in
//                     chunks of 128. A warp's origin comes from its index (tid.x >> 5): one value per warp.
//   attention_kernel  per warp of 32 queries over 32-key tiles in a causal window: S = Q K^T with both operands in
//                     device memory, P = bf16(S) to the warp's shared tile, O += P V^T with A from shared memory.
//                     The pointers are fields of a by-value struct argument; the window start is max(0, q0 - window).
// Inputs are small multiples of powers of two, so every product and partial sum is exact and results compare equal.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
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

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ uint32_t ld32(const void* p) { return *reinterpret_cast<const uint32_t*>(p); }

template <class X> struct mma_t;
template <> struct mma_t<bf16> {
  typedef float acc;
  static constexpr int k = 16;
  static __device__ __forceinline__ void run(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
};
template <> struct mma_t<int8_t> {
  typedef int acc;
  static constexpr int k = 32;
  static __device__ __forceinline__ void run(int (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
};

// Lane l = 4g + t holds A rows g and g + 8 and B row g, each at the word of columns kw t (kw = 2 bf16 or 4 s8
// elements) and the word 16 bytes further; p points at the lane's first word and ld is the row pitch in elements.
template <class X> __device__ __forceinline__ void load_a(uint32_t (&f)[4], const X* p, size_t ld) {
  constexpr int half = 16 / sizeof(X);
  f[0] = ld32(p);
  f[1] = ld32(p + 8 * ld);
  f[2] = ld32(p + half);
  f[3] = ld32(p + 8 * ld + half);
}
template <class X> __device__ __forceinline__ void load_b(uint32_t (&f)[2], const X* p) {
  constexpr int half = 16 / sizeof(X);
  f[0] = ld32(p);
  f[1] = ld32(p + half);
}

template <class X, int WN>
__global__ void __launch_bounds__(64 * WN) gemm_dev_kernel(const X* __restrict__ A, const X* __restrict__ B,
                                                           typename mma_t<X>::acc* __restrict__ C, int N, int K) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wm = warp / WN, wn = warp % WN, g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.y * 64 + wm * 32, col0 = blockIdx.x * 32 * WN + wn * 32;
  const X* a = A + (size_t)(row0 + g) * K + 4 / sizeof(X) * t;
  const X* b = B + (size_t)(col0 + g) * K + 4 / sizeof(X) * t;
  typename mma_t<X>::acc acc[2][4][4] = {};
  for (int k0 = 0; k0 < K; k0 += 128) {
#pragma unroll
    for (int kk = 0; kk < 128; kk += mma_t<X>::k) {
      uint32_t fa[2][4], fb[4][2];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) load_a(fa[mt], a + (size_t)16 * mt * K + k0 + kk, K);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) load_b(fb[nt], b + (size_t)8 * nt * K + k0 + kk);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma_t<X>::run(acc[mt][nt], fa[mt], fb[nt]);
    }
  }
  // accumulator r of tile (mt, nt): row 16 mt + g + 8 (r >> 1), column 8 nt + 2t + (r & 1)
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        C[(size_t)(row0 + 16 * mt + g + 8 * (r >> 1)) * N + col0 + 8 * nt + 2 * t + (r & 1)] = acc[mt][nt][r];
}

constexpr int HD = 64, KT = 32, WARPS = 4;

struct Attention {
  const bf16* q;   // [heads][n][HD]
  const bf16* k;   // [heads][n][HD]
  const bf16* vt;  // [heads][HD][n]
  int n, window;
  float* out;      // [heads][n][HD]
};

__global__ void __launch_bounds__(32 * WARPS) attention_kernel(Attention a) {
  __shared__ __align__(16) bf16 probs[WARPS][32 * KT];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3, h = blockIdx.y;
  const int q0 = (blockIdx.x * WARPS + warp) * 32;
  const bf16* Q = a.q + ((size_t)h * a.n + q0 + g) * HD + 2 * t;
  const bf16* K = a.k + ((size_t)h * a.n + g) * HD + 2 * t;
  const bf16* VT = a.vt + ((size_t)h * HD + g) * a.n + 2 * t;
  bf16* P = probs[warp];
  float o[2][8][4] = {};
  const int first = max(0, q0 - a.window) / KT * KT;
  for (int k0 = first; k0 < q0 + 32; k0 += KT) {
    float s[2][4][4] = {};
#pragma unroll
    for (int kk = 0; kk < HD; kk += 16) {
      uint32_t fa[2][4], fb[4][2];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) load_a(fa[mt], Q + 16 * mt * HD + kk, HD);
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) load_b(fb[nt], K + (size_t)(k0 + 8 * nt) * HD + kk);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma_t<bf16>::run(s[mt][nt], fa[mt], fb[nt]);
    }
    // keep 0 <= q - k <= window: element (r, c) of this lane has q - k = d0 + (r - g) - (c - 2t)
    const bool edge = k0 + KT > q0 || k0 < q0 + 31 - a.window;
    const int d0 = q0 - k0 + g - 2 * t;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt)
#pragma unroll
        for (int hh = 0; hh < 2; ++hh) {
          bf16 p[2];
#pragma unroll
          for (int j = 0; j < 2; ++j) {
            const bool keep = !edge || (unsigned)(d0 + 16 * mt + 8 * hh - 8 * nt - j) <= (unsigned)a.window;
            p[j] = __float2bfloat16(keep ? s[mt][nt][2 * hh + j] : 0.f);
          }
          *reinterpret_cast<__nv_bfloat162*>(P + (16 * mt + g + 8 * hh) * KT + 8 * nt + 2 * t) = __halves2bfloat162(p[0], p[1]);
        }
    __syncwarp();
#pragma unroll
    for (int kk = 0; kk < KT; kk += 16) {
      uint32_t fa[2][4], fb[8][2];
#pragma unroll
      for (int mt = 0; mt < 2; ++mt) load_a(fa[mt], P + (16 * mt + g) * KT + kk + 2 * t, KT);
#pragma unroll
      for (int nt = 0; nt < 8; ++nt) load_b(fb[nt], VT + (size_t)8 * nt * a.n + k0 + kk);
#pragma unroll
      for (int mt = 0; mt < 2; ++mt)
#pragma unroll
        for (int nt = 0; nt < 8; ++nt) mma_t<bf16>::run(o[mt][nt], fa[mt], fb[nt]);
    }
    __syncwarp();
  }
  float* dst = a.out + ((size_t)h * a.n + q0) * HD;
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 8; ++nt)
#pragma unroll
      for (int r = 0; r < 4; ++r) dst[(size_t)(16 * mt + g + 8 * (r >> 1)) * HD + 8 * nt + 2 * t + (r & 1)] = o[mt][nt][r];
}

// ---------------------------------------------------------------- host
static uint16_t bf16_bits(float f) { uint32_t u; memcpy(&u, &f, 4); return (uint16_t)(u >> 16); }  // exact inputs only
static void put(bf16& d, float v) { const uint16_t u = bf16_bits(v); memcpy(&d, &u, 2); }
static void put(int8_t& d, float v) { d = (int8_t)v; }

static int failures = 0;

static void report(const char* what, int bad, size_t count) {
  printf("  %-44s %s", what, bad ? "FAIL" : "ok");
  if (bad) printf(" (%d of %zu differ)", bad, count);
  printf("\n");
  if (bad) failures++;
}

template <class X, int WN>
static void gemm_case(int M, int N, int K, uint32_t seed) {
  typedef typename mma_t<X>::acc Acc;
  const bool s8 = sizeof(X) == 1;
  std::vector<float> a((size_t)M * K), b((size_t)N * K);
  srand(seed);
  for (auto& x : a) x = s8 ? (float)(rand() % 255 - 127) : (float)(rand() % 9 - 4) * 0.25f;
  for (auto& x : b) x = s8 ? (float)(rand() % 255 - 127) : (float)(rand() % 9 - 4) * 0.25f;
  std::vector<X> ha(a.size()), hb(b.size());
  for (size_t i = 0; i < a.size(); ++i) put(ha[i], a[i]);
  for (size_t i = 0; i < b.size(); ++i) put(hb[i], b[i]);
  X *dA, *dB; Acc* dC;
  CK(cudaMalloc(&dA, ha.size() * sizeof(X))); CK(cudaMalloc(&dB, hb.size() * sizeof(X))); CK(cudaMalloc(&dC, (size_t)M * N * sizeof(Acc)));
  CK(cudaMemcpy(dA, ha.data(), ha.size() * sizeof(X), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hb.data(), hb.size() * sizeof(X), cudaMemcpyHostToDevice));
  CK(cudaMemset(dC, 0, (size_t)M * N * sizeof(Acc)));
  gemm_dev_kernel<X, WN><<<dim3(N / (32 * WN), M / 64), 64 * WN>>>(dA, dB, dC, N, K);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<Acc> hc((size_t)M * N);
  CK(cudaMemcpy(hc.data(), dC, hc.size() * sizeof(Acc), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      double ref = 0;
      for (int k = 0; k < K; ++k) ref += (double)a[(size_t)m * K + k] * b[(size_t)n * K + k];
      if ((double)hc[(size_t)m * N + n] != ref && bad++ < 5) fprintf(stderr, "    mismatch at (%d,%d): got %g want %g\n", m, n, (double)hc[(size_t)m * N + n], ref);
    }
  char what[96];
  snprintf(what, sizeof what, "%s gemm %dx%dx%d, 2x%d warps", s8 ? "s8  " : "bf16", M, N, K, WN);
  report(what, bad, hc.size());
  CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC));
}

static void attention_case(int heads, int n, int window, uint32_t seed) {
  std::vector<float> q((size_t)heads * n * HD), k(q.size()), vt(q.size());
  srand(seed);
  for (auto* v : {&q, &k, &vt})
    for (auto& x : *v) x = (float)(rand() % 5 - 2) * 0.5f;
  auto upload = [](const std::vector<float>& v) {
    std::vector<uint16_t> h(v.size());
    for (size_t i = 0; i < v.size(); ++i) h[i] = bf16_bits(v[i]);
    bf16* d; CK(cudaMalloc(&d, h.size() * 2)); CK(cudaMemcpy(d, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    return d;
  };
  Attention args;
  args.q = upload(q); args.k = upload(k); args.vt = upload(vt);
  args.n = n; args.window = window;
  CK(cudaMalloc(&args.out, q.size() * 4));
  CK(cudaMemset(args.out, 0, q.size() * 4));
  attention_kernel<<<dim3(n / (32 * WARPS), heads), 32 * WARPS>>>(args);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<float> out(q.size());
  CK(cudaMemcpy(out.data(), args.out, out.size() * 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  std::vector<double> s(n);
  for (int h = 0; h < heads; ++h)
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        s[j] = 0;
        if (j > i || i - j > window) continue;
        for (int d = 0; d < HD; ++d) s[j] += (double)q[((size_t)h * n + i) * HD + d] * k[((size_t)h * n + j) * HD + d];
      }
      for (int c = 0; c < HD; ++c) {
        double ref = 0;
        for (int j = 0; j < n; ++j) ref += s[j] * vt[((size_t)h * HD + c) * n + j];
        const float got = out[((size_t)h * n + i) * HD + c];
        if ((double)got != ref && bad++ < 5) fprintf(stderr, "    mismatch at head %d (%d,%d): got %g want %g\n", h, i, c, got, ref);
      }
    }
  char what[96];
  snprintf(what, sizeof what, "attention %d heads x %d, window %d", heads, n, window);
  report(what, bad, out.size());
  CK(cudaFree((void*)args.q)); CK(cudaFree((void*)args.k)); CK(cudaFree((void*)args.vt)); CK(cudaFree(args.out));
}

int main() {
  printf("gemm_dev: mma.sync fed by 32-bit device and shared loads vs CPU\n");
  gemm_case<bf16, 16>(128, 1024, 384, 1);
  gemm_case<int8_t, 16>(128, 1024, 384, 2);
  gemm_case<bf16, 4>(192, 256, 256, 3);
  attention_case(2, 512, 200, 4);
  printf("%d failure%s\n", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
