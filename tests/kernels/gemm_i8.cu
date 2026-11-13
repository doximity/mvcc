// s8 mma.sync m16n8k32 on cooperative operands: fragments that are not plain device loads.
//   gemm_ldm_kernel  C = A B^T, A and B staged to shared memory and read with ldmatrix.x4 (non-.trans): 2 x 2 warps of
//                    WM x WN tiles, K in steps of 32.
//   gemm_reg_kernel  the same product with every fragment word computed in registers (bytes stored offset by 128,
//                    flipped back to s8 with an xor), as a kernel that slices or quantizes its operands writes it.
// Integer products sum exactly, so results compare equal to the CPU.
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>

#define CK(x)                                                                                              \
  do {                                                                                                     \
    cudaError_t e_ = (x);                                                                                  \
    if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } \
  } while (0)

__device__ __forceinline__ void mma(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldm4(uint32_t (&r)[4], const void* p) {
  const uint32_t s = (uint32_t)__cvta_generic_to_shared(p);
  asm volatile("ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(s));
}

constexpr int KC = 32, LDS = KC + 16;

template <int WM, int WN>
__global__ void __launch_bounds__(128) gemm_ldm_kernel(const int8_t* __restrict__ A, const int8_t* __restrict__ B, int* __restrict__ C, int N, int K) {
  constexpr int BM = 2 * WM, BN = 2 * WN, MT = WM / 16, NP = WN / 16;
  __shared__ __align__(16) int8_t as[BM * LDS], bs[BN * LDS];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wm = warp >> 1, wn = warp & 1;
  const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
  int acc[MT][2 * NP][4] = {};
  for (int k0 = 0; k0 < K; k0 += KC) {
    for (int i = threadIdx.x; i < BM * KC / 4; i += 128) {
      const int r = i / (KC / 4), c = 4 * (i % (KC / 4));
      *(uint32_t*)&as[r * LDS + c] = *(const uint32_t*)&A[(size_t)(row0 + r) * K + k0 + c];
    }
    for (int i = threadIdx.x; i < BN * KC / 4; i += 128) {
      const int r = i / (KC / 4), c = 4 * (i % (KC / 4));
      *(uint32_t*)&bs[r * LDS + c] = *(const uint32_t*)&B[(size_t)(col0 + r) * K + k0 + c];
    }
    __syncthreads();
    uint32_t fa[MT][4], fb[NP][4];
    // A x4: matrices (rows 0-7, k 0-15), (rows 8-15), (k 16-31), (both); B x4: (n 0-7, k 0-15), (k 16-31), (n 8-15), (both)
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) ldm4(fa[mt], &as[(wm * WM + 16 * mt + (lane & 15)) * LDS + 16 * (lane >> 4)]);
#pragma unroll
    for (int np = 0; np < NP; ++np) ldm4(fb[np], &bs[(wn * WN + 16 * np + (lane & 7) + 8 * (lane >> 4)) * LDS + 16 * ((lane >> 3) & 1)]);
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
      for (int np = 0; np < NP; ++np) {
        mma(acc[mt][2 * np], fa[mt], fb[np][0], fb[np][1]);
        mma(acc[mt][2 * np + 1], fa[mt], fb[np][2], fb[np][3]);
      }
    __syncthreads();
  }
  const int g = lane >> 2, t = lane & 3;
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int nt = 0; nt < 2 * NP; ++nt)
#pragma unroll
      for (int r = 0; r < 4; ++r)
        C[(size_t)(row0 + wm * WM + 16 * mt + g + 8 * (r >> 1)) * N + col0 + wn * WN + 8 * nt + 2 * t + (r & 1)] = acc[mt][nt][r];
}

__global__ void __launch_bounds__(128) gemm_reg_kernel(const uint8_t* __restrict__ A, const uint8_t* __restrict__ B, int* __restrict__ C, int N, int K, uint32_t flip) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int row0 = blockIdx.y * 64 + (warp >> 1) * 32, col0 = blockIdx.x * 64 + (warp & 1) * 32;
  int acc[2][4][4] = {};
  auto w = [&](const uint8_t* p) { return *(const uint32_t*)p ^ flip; };
  for (int k0 = 0; k0 < K; k0 += 32) {
    uint32_t fa[2][4], fb[4][2];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
      const uint8_t* p = A + (size_t)(row0 + 16 * mt + g) * K + k0 + 4 * t;
      fa[mt][0] = w(p); fa[mt][1] = w(p + 8 * K); fa[mt][2] = w(p + 16); fa[mt][3] = w(p + 8 * K + 16);
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
      const uint8_t* p = B + (size_t)(col0 + 8 * nt + g) * K + k0 + 4 * t;
      fb[nt][0] = w(p); fb[nt][1] = w(p + 16);
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt)
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) mma(acc[mt][nt], fa[mt], fb[nt][0], fb[nt][1]);
  }
#pragma unroll
  for (int mt = 0; mt < 2; ++mt)
#pragma unroll
    for (int nt = 0; nt < 4; ++nt)
#pragma unroll
      for (int r = 0; r < 4; ++r) C[(size_t)(row0 + 16 * mt + g + 8 * (r >> 1)) * N + col0 + 8 * nt + 2 * t + (r & 1)] = acc[mt][nt][r];
}

static int failures = 0;

template <class F>
static void run_case(const char* what, int M, int N, int K, uint32_t seed, F launch) {
  std::vector<int> a((size_t)M * K), b((size_t)N * K);
  srand(seed);
  for (auto& x : a) x = rand() % 255 - 127;
  for (auto& x : b) x = rand() % 255 - 127;
  std::vector<int8_t> ha(a.begin(), a.end()), hb(b.begin(), b.end());
  int8_t *dA, *dB; int* dC;
  CK(cudaMalloc(&dA, ha.size())); CK(cudaMalloc(&dB, hb.size())); CK(cudaMalloc(&dC, (size_t)M * N * 4));
  CK(cudaMemcpy(dA, ha.data(), ha.size(), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hb.data(), hb.size(), cudaMemcpyHostToDevice));
  CK(cudaMemset(dC, 0, (size_t)M * N * 4));
  launch(dA, dB, dC);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  std::vector<int> hc((size_t)M * N);
  CK(cudaMemcpy(hc.data(), dC, hc.size() * 4, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int m = 0; m < M; ++m)
    for (int n = 0; n < N; ++n) {
      long ref = 0;
      for (int k = 0; k < K; ++k) ref += (long)a[(size_t)m * K + k] * b[(size_t)n * K + k];
      if (hc[(size_t)m * N + n] != ref && bad++ < 5) fprintf(stderr, "    mismatch at (%d,%d): got %d want %ld\n", m, n, hc[(size_t)m * N + n], ref);
    }
  printf("  %-44s %s", what, bad ? "FAIL" : "ok");
  if (bad) printf(" (%d of %zu differ)", bad, hc.size());
  printf("\n");
  failures += bad != 0;
  CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC));
}

int main() {
  printf("gemm_i8: s8 mma.sync from ldmatrix and register-built fragments vs CPU\n");
  const int M = 128, N = 256, K = 512;
  run_case("ldmatrix 32x32 warp tiles", M, N, K, 1, [&](int8_t* A, int8_t* B, int* C) {
    gemm_ldm_kernel<32, 32><<<dim3(N / 64, M / 64), 128>>>(A, B, C, N, K);
  });
  run_case("ldmatrix 16x16 warp tiles", M, N, K, 2, [&](int8_t* A, int8_t* B, int* C) {
    gemm_ldm_kernel<16, 16><<<dim3(N / 32, M / 32), 128>>>(A, B, C, N, K);
  });
  run_case("register-built fragments", M, N, K, 3, [&](int8_t* A, int8_t* B, int* C) {
    // the kernel reads bytes offset by 128 and flips them back: stored s8 x reads as (x ^ 0x80) ^ 0x80
    std::vector<uint8_t> h((size_t)M * K), g((size_t)N * K);
    CK(cudaMemcpy(h.data(), A, h.size(), cudaMemcpyDeviceToHost)); CK(cudaMemcpy(g.data(), B, g.size(), cudaMemcpyDeviceToHost));
    for (auto& x : h) x ^= 0x80; for (auto& x : g) x ^= 0x80;
    CK(cudaMemcpy(A, h.data(), h.size(), cudaMemcpyHostToDevice)); CK(cudaMemcpy(B, g.data(), g.size(), cudaMemcpyHostToDevice));
    gemm_reg_kernel<<<dim3(N / 64, M / 64), 128>>>((const uint8_t*)A, (const uint8_t*)B, C, N, K, 0x80808080u);
  });
  printf("%d failure%s\n", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
