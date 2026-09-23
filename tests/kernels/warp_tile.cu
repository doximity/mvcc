// mvcc::warp_tile correctness: C[M,N] = A[M,K] . B[N,K]^T on the tensor path (Metal 4 TensorOps under mvcc,
// mma.sync under nvcc) against a CPU reference. Operands come from device memory and, in a second variant, from
// shared memory. Inputs are small multiples of 1/4 so every product and partial sum is exact in fp32 and the
// comparison is bit-exact; a third pass uses random reals with a relative tolerance.
#include <cuda_runtime.h>
#include <mvcc/tile.cuh>
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

template <typename T> struct conv;
template <> struct conv<__nv_bfloat16> {
  static uint16_t from_float(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
  static float to_float(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
  static const char* name() { return "bf16"; }
};
template <> struct conv<__half> {
  static uint16_t from_float(float f) { _Float16 h = (_Float16)f; uint16_t r; memcpy(&r, &h, 2); return r; }
  static float to_float(uint16_t b) { _Float16 h; memcpy(&h, &b, 2); return (float)h; }
  static const char* name() { return "f16"; }
};

// One warp: full M x N tile, K streamed in chunks of TK. SMEM: stage A and B tiles in shared memory first.
template <int M, int N, int TK, typename T, bool SMEM>
__global__ void tile_kernel(const T* __restrict__ A, const T* __restrict__ B, float* __restrict__ C, int K) {
  __shared__ __align__(16) T As[SMEM ? M * TK : 1];
  __shared__ __align__(16) T Bs[SMEM ? N * TK : 1];
  mvcc::warp_tile<M, N, TK, T> tile;
  tile.zero();
  for (int k0 = 0; k0 < K; k0 += TK) {
    if (SMEM) {
      __syncthreads();
      for (int i = threadIdx.x; i < M * TK; i += blockDim.x) As[i] = A[(i / TK) * K + k0 + (i % TK)];
      for (int i = threadIdx.x; i < N * TK; i += blockDim.x) Bs[i] = B[(i / TK) * K + k0 + (i % TK)];
      __syncthreads();
      tile.mma(As, TK, Bs, TK);
    } else {
      tile.mma(A + k0, K, B + k0, K);
    }
  }
  tile.foreach_c([&](int r, int c, float& v) { C[r * N + c] = v; });
}

static int failures = 0;

template <int M, int N, int TK, typename T, bool SMEM>
static void run(int K, bool exact, uint32_t seed) {
  std::vector<uint16_t> ha((size_t)M * K), hb((size_t)N * K);
  srand(seed);
  auto gen = [&]() -> float {
    if (exact) return (float)((rand() % 9) - 4) * 0.25f;  // {-1, -0.75, ..., 1}: exact in bf16/f16
    return (float)(rand() & 0xffff) / 32768.f - 1.f;
  };
  for (auto& v : ha) v = conv<T>::from_float(gen());
  for (auto& v : hb) v = conv<T>::from_float(gen());
  std::vector<float> ref((size_t)M * N, 0.f);
  for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j) {
      double acc = 0;
      for (int k = 0; k < K; ++k) acc += (double)conv<T>::to_float(ha[(size_t)i * K + k]) * conv<T>::to_float(hb[(size_t)j * K + k]);
      ref[(size_t)i * N + j] = (float)acc;
    }
  T *dA, *dB; float* dC;
  CK(cudaMalloc(&dA, ha.size() * 2)); CK(cudaMalloc(&dB, hb.size() * 2)); CK(cudaMalloc(&dC, ref.size() * 4));
  CK(cudaMemcpy(dA, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemset(dC, 0xff, ref.size() * 4));  // NaN sentinel: any coordinate foreach_c misses shows up
  tile_kernel<M, N, TK, T, SMEM><<<1, 32>>>(dA, dB, dC, K);
  CK(cudaGetLastError());
  std::vector<float> out(ref.size());
  CK(cudaMemcpy(out.data(), dC, out.size() * 4, cudaMemcpyDeviceToHost));
  int bad = 0; float worst = 0.f;
  for (size_t i = 0; i < ref.size(); ++i) {
    const float d = fabsf(out[i] - ref[i]);
    const float tol = exact ? 0.f : 1e-3f * (1.f + fabsf(ref[i]));
    if (!(d <= tol)) { if (bad < 3) fprintf(stderr, "    [%zu] got %g want %g\n", i, out[i], ref[i]); ++bad; }
    if (d > worst) worst = d;
  }
  printf("  %-4s M=%d N=%d K=%d TK=%d %-6s %-6s : %s (max |err| %.3g)\n", conv<T>::name(), M, N, K, TK,
         SMEM ? "smem" : "device", exact ? "exact" : "random", bad ? "FAIL" : "ok", worst);
  if (bad) ++failures;
  CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dC));
}

int main() {
  printf("warp_tile tests\n");
  run<32, 32, 32, __nv_bfloat16, false>(32, true, 1);
  run<32, 32, 32, __nv_bfloat16, false>(256, true, 2);
  run<32, 32, 32, __nv_bfloat16, true>(256, true, 3);
  run<32, 32, 64, __nv_bfloat16, false>(256, true, 4);
  run<32, 32, 32, __nv_bfloat16, false>(1024, false, 5);
  run<32, 32, 32, __half, false>(256, true, 6);
  run<32, 32, 32, __half, true>(256, true, 7);
  run<32, 32, 32, __half, false>(1024, false, 8);
  run<64, 32, 32, __nv_bfloat16, false>(128, true, 9);
  run<32, 64, 32, __nv_bfloat16, false>(128, true, 10);
  run<64, 64, 32, __nv_bfloat16, true>(128, true, 11);
  run<16, 16, 16, __nv_bfloat16, false>(64, true, 12);
  if (failures) { printf("%d FAILED\n", failures); return 1; }
  printf("all ok\n");
  return 0;
}
