// mvcc smoke test: the classic. Checks launch, cudaMalloc/cudaMemcpy, events, and error reporting.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

__global__ void vectorAdd(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("FAIL %s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

int main() {
  const int n = 1 << 20;
  std::vector<float> ha(n), hb(n), hc(n);
  for (int i = 0; i < n; i++) { ha[i] = i * 0.5f; hb[i] = 1000.f - i; }
  float *da, *db, *dc;
  CK(cudaMalloc(&da, n * sizeof(float))); CK(cudaMalloc(&db, n * sizeof(float))); CK(cudaMalloc(&dc, n * sizeof(float)));
  CK(cudaMemcpy(da, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  CK(cudaMemcpy(db, hb.data(), n * sizeof(float), cudaMemcpyHostToDevice));
  cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  vectorAdd<<<(n + 255) / 256, 256>>>(da, db, dc, n);  // warm-up: pipeline compile
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(e0));
  for (int it = 0; it < 10; it++) vectorAdd<<<(n + 255) / 256, 256>>>(da, db, dc, n);
  CK(cudaEventRecord(e1));
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
  CK(cudaMemcpy(hc.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int i = 0; i < n; i++) if (hc[i] != ha[i] + hb[i]) { if (bad < 5) printf("mismatch at %d: %f vs %f\n", i, hc[i], ha[i] + hb[i]); bad++; }
  cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, 0));
  printf("device: %s, %d cores, %.1f GB\n", p.name, p.multiProcessorCount, p.totalGlobalMem / 1e9);
  printf("10 launches of %d elements: %.3f ms (%.1f GB/s)\n", n, ms, 10.0 * 3 * n * sizeof(float) / (ms * 1e6));
  printf(bad ? "FAIL (%d mismatches)\n" : "PASS\n", bad);
  CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(dc));
  return bad ? 1 : 0;
}
