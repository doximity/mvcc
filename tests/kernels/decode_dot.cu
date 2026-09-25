// Decode-shaped DOT contraction: Y[n] += scale[k/32] · ⟨unpack(W[n,k]), x[k]⟩
// plus a 32-lane xor-sum. Not named gemv — STIR must recover it from the IR.
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>

__global__ void __launch_bounds__(32) packed_dot(const uint8_t* __restrict__ W, const float* __restrict__ x,
                                                const float* __restrict__ scale, float* __restrict__ y, int N, int K) {
  const int n = blockIdx.x;
  if (n >= N) return;
  const int lane = threadIdx.x;
  float acc = 0.f;
  for (int k = lane; k < K; k += 32) {
    const float w = (float)(W[(size_t)n * (size_t)K + k] & 0x0F) - 8.f;
    acc = fmaf(w * scale[k >> 5], x[k], acc);
  }
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  if (lane == 0) y[n] = acc;
}

// NT GEMM / block_sum shape: plain bf16×bf16 + xor-sum, then a tid-predicated
// second tree. Must not be recovered as gemv-dot (that hung prefill).
__global__ void __launch_bounds__(128) plain_nt_dot(const __nv_bfloat16* __restrict__ a,
                                                   const __nv_bfloat16* __restrict__ w, float* __restrict__ y, int K) {
  float acc = 0.f;
  for (int k = threadIdx.x; k < K; k += 128) acc += __bfloat162float(a[k]) * __bfloat162float(w[k]);
  for (int off = 16; off > 0; off >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, off);
  __shared__ float sm[4];
  if ((threadIdx.x & 31) == 0) sm[threadIdx.x >> 5] = acc;
  __syncthreads();
  float yv = (threadIdx.x < 4) ? sm[threadIdx.x] : 0.f;
  if (threadIdx.x < 32) {
    for (int off = 16; off > 0; off >>= 1) yv += __shfl_xor_sync(0xffffffffu, yv, off);
    if (threadIdx.x == 0) y[blockIdx.x] = yv;
  }
}

int main() {
  const int N = 8, K = 256;
  std::vector<uint8_t> hW((size_t)N * K);
  std::vector<float> hx(K), hs((K + 31) / 32), hy(N), href(N);
  for (int i = 0; i < N * K; i++) hW[i] = (uint8_t)((i * 7 + 3) & 0xFF);
  for (int k = 0; k < K; k++) hx[k] = 0.01f * (float)((k % 17) + 1);
  for (int g = 0; g < (int)hs.size(); g++) hs[g] = 0.25f + 0.05f * (float)g;
  for (int n = 0; n < N; n++) {
    float acc = 0.f;
    for (int k = 0; k < K; k++) {
      const float w = (float)(hW[(size_t)n * K + k] & 0x0F) - 8.f;
      acc += w * hs[k >> 5] * hx[k];
    }
    href[n] = acc;
  }
  uint8_t* dW;
  float *dx, *ds, *dy;
  if (cudaMalloc(&dW, hW.size()) || cudaMalloc(&dx, K * sizeof(float)) || cudaMalloc(&ds, hs.size() * sizeof(float))
      || cudaMalloc(&dy, N * sizeof(float)))
    return 2;
  cudaMemcpy(dW, hW.data(), hW.size(), cudaMemcpyHostToDevice);
  cudaMemcpy(dx, hx.data(), K * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(ds, hs.data(), hs.size() * sizeof(float), cudaMemcpyHostToDevice);
  packed_dot<<<N, 32>>>(dW, dx, ds, dy, N, K);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(hy.data(), dy, N * sizeof(float), cudaMemcpyDeviceToHost);
  int bad = 0;
  for (int n = 0; n < N; n++) {
    float d = fabsf(hy[n] - href[n]);
    float t = 1e-3f * (fabsf(href[n]) + 1.f);
    if (d > t) bad++;
  }
  cudaFree(dW);
  cudaFree(dx);
  cudaFree(ds);
  cudaFree(dy);
  if (bad) {
    printf("packed_dot: %d / %d mismatch\n", bad, N);
    return 1;
  }
  printf("packed_dot: %d ok (gemv-dot recovered)\n", N);
  return 0;
}
