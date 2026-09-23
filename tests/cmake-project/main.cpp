// Host half of the CMake test project: runs the kernel in kern.cu and checks the result.
#include <cuda_runtime.h>
#include <cstdio>
extern "C" cudaError_t run_scale(float* x, float s, int n, cudaStream_t st);
int main() {
  float* d; int n = 1000; float h[1000];
  for (int i = 0; i < n; i++) h[i] = i;
  cudaMalloc(&d, n * 4); cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice);
  cudaStream_t st; cudaStreamCreate(&st);
  if (run_scale(d, 2.f, n, st) != cudaSuccess) { printf("launch failed\n"); return 1; }
  cudaMemcpyAsync(h, d, n * 4, cudaMemcpyDeviceToHost, st); cudaStreamSynchronize(st);
  int bad = 0; for (int i = 0; i < n; i++) bad += h[i] != 2.f * i;
  printf(bad ? "FAIL\n" : "PASS\n"); return bad != 0;
}
