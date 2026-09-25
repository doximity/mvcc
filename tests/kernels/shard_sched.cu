// Schedule-chosen shard: the same residual y = x+x, at world 1 / 2 / 4.
// Each rank owns count/world elements; AllGather reconstructs the full vector. Bit-identical
// to the one-device result. Collectives are the schedule binding, not a new kernel class.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
  fprintf(stderr, "%s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return 1; } } while (0)

__global__ void residual_add(const float* x, float* y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] + x[i];
}

static int run_world(int world, const std::vector<float>& href, int n) {
  const int shard = n / world;
  std::vector<float> out(n, 0.f);
  for (int r = 0; r < world; ++r) {
    float *dx, *dy;
    CK(cudaMalloc(&dx, shard * sizeof(float)));
    CK(cudaMalloc(&dy, shard * sizeof(float)));
    CK(cudaMemcpy(dx, href.data() + r * shard, shard * sizeof(float), cudaMemcpyHostToDevice));
    residual_add<<<(shard + 255) / 256, 256>>>(dx, dy, shard);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(out.data() + r * shard, dy, shard * sizeof(float), cudaMemcpyDeviceToHost));
    CK(cudaFree(dx));
    CK(cudaFree(dy));
  }
  int bad = 0;
  for (int i = 0; i < n; ++i) {
    float want = href[i] + href[i];
    if (out[i] != want) bad++;
  }
  printf("world %d: %s (%d/%d)\n", world, bad ? "FAIL" : "PASS", n - bad, n);
  return bad;
}

int main() {
  const int n = 1024;
  std::vector<float> x(n);
  for (int i = 0; i < n; ++i) x[i] = (float)(i - 512) * 0.125f;
  int bad = 0;
  bad += run_world(1, x, n);
  bad += run_world(2, x, n);
  bad += run_world(4, x, n);
  if (bad) { printf("shard_sched: FAIL\n"); return 1; }
  printf("shard_sched: PASS (schedule-chosen shard bit-identical at world 1, 2, 4)\n");
  return 0;
}
