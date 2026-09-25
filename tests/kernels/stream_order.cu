// Stream ordering regression test.
//
// Metal buffers in the runtime are hazard-tracking-untracked (kernels take raw GPU addresses), so nothing
// orders two dispatches unless the runtime does it explicitly. This test chains thousands of tiny dependent
// kernels, interleaves device-to-device copies, memsets and host-to-device copies on the same stream, and
// spans several command buffers. Any missing barrier/fence shows up as a wrong final value.
//
//   nvcc -O2 -o stream_order stream_order.cu && ./stream_order
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess) {                                                                         \
      fprintf(stderr, "%s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__);    \
      return 1;                                                                                      \
    }                                                                                                \
  } while (0)

// y[i] = x[i] * 3 + 1  (each step depends on the previous one's full output)
__global__ void step_kernel(const int* __restrict__ x, int* __restrict__ y, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = x[i] * 3 + 1;
}
// Reduction into a single counter so that a stale read is visible even for one element.
__global__ void sum_kernel(const int* __restrict__ x, int n, unsigned* out) {
  __shared__ unsigned part[256];
  unsigned acc = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) acc += x[i];
  part[threadIdx.x] = acc;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) part[threadIdx.x] += part[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicAdd(out, part[0]);
}

int main() {
  const int n = 1 << 16;
  const int steps = 3000;  // > runtime commit threshold: spans several command buffers
  int *a, *b;
  unsigned* total;
  CK(cudaMalloc(&a, n * sizeof(int)));
  CK(cudaMalloc(&b, n * sizeof(int)));
  CK(cudaMalloc(&total, sizeof(unsigned)));

  // reference on the host: value after k steps from seed s is s*3^k + (3^k - 1)/2, all mod 2^32
  std::vector<int> h(n);
  for (int i = 0; i < n; ++i) h[i] = i & 0xff;
  CK(cudaMemcpy(a, h.data(), n * sizeof(int), cudaMemcpyHostToDevice));
  CK(cudaMemset(total, 0, sizeof(unsigned)));

  std::vector<unsigned> ref(n);
  for (int i = 0; i < n; ++i) ref[i] = (unsigned)h[i];

  int fails = 0;
  for (int k = 0; k < steps; ++k) {
    step_kernel<<<n / 256, 256>>>(a, b, n);
    for (int i = 0; i < n; ++i) ref[i] = ref[i] * 3u + 1u;
    if (k % 97 == 13) {
      // D2D copy back instead of pointer swap: exercises blit <-> compute ordering
      CK(cudaMemcpyAsync(a, b, n * sizeof(int), cudaMemcpyDeviceToDevice, 0));
    } else if (k % 97 == 50) {
      // reset one slab through memset then a fresh H2D copy: blit fill + staged upload in the stream
      CK(cudaMemsetAsync(b, 0, 1024 * sizeof(int), 0));
      for (int i = 0; i < 1024; ++i) ref[i] = 0;
      std::vector<int> patch(1024, k);
      CK(cudaMemcpyAsync(b, patch.data(), 1024 * sizeof(int), cudaMemcpyHostToDevice, 0));
      for (int i = 0; i < 1024; ++i) ref[i] = (unsigned)k;
      std::swap(a, b);
    } else {
      std::swap(a, b);
    }
  }
  sum_kernel<<<64, 256>>>(a, n, total);
  std::vector<int> out(n);
  CK(cudaMemcpy(out.data(), a, n * sizeof(int), cudaMemcpyDeviceToHost));
  unsigned got_total = 0;
  CK(cudaMemcpy(&got_total, total, sizeof(unsigned), cudaMemcpyDeviceToHost));
  unsigned want_total = 0;
  for (int i = 0; i < n; ++i) {
    want_total += ref[i];
    if ((unsigned)out[i] != ref[i]) { if (fails < 5) fprintf(stderr, "mismatch at %d: got %u want %u\n", i, (unsigned)out[i], ref[i]); ++fails; }
  }
  if (got_total != want_total) { fprintf(stderr, "sum mismatch: got %u want %u\n", got_total, want_total); ++fails; }
  printf("stream_order: %d steps, %d mismatches -> %s\n", steps, fails, fails ? "FAIL" : "PASS");
  return fails ? 1 : 0;
}
