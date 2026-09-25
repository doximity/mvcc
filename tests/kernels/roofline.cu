// Roofline probes for README.md, written as a plain CUDA program so the numbers are what CUDA
// code actually sees through mvcc (barriers, encoders and all).
//
//   nvcc -O3 -o roofline roofline.cu && ./roofline
//
// Reports: streaming read bandwidth (decode ceiling), copy bandwidth, per-dispatch overhead for tiny kernels
// inside one stream, and cudaDeviceSynchronize round-trip cost.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>

#define CK(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess) {                                                                         \
      fprintf(stderr, "%s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__);    \
      exit(1);                                                                                       \
    }                                                                                                \
  } while (0)

static double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// Sum-reduce a buffer with 16-byte loads; result written per block so the compiler cannot elide loads.
__global__ void __launch_bounds__(256) read_bw_kernel(const uint4* __restrict__ p, size_t n16, float* __restrict__ out) {
  float acc = 0.f;
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) {
    const uint4 v = p[i];
    acc += __uint_as_float(v.x) + __uint_as_float(v.y) + __uint_as_float(v.z) + __uint_as_float(v.w);
  }
  for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
  if ((threadIdx.x & 31) == 0) out[blockIdx.x * 8 + (threadIdx.x >> 5)] = acc;
}
__global__ void __launch_bounds__(256) copy_kernel(const uint4* __restrict__ src, uint4* __restrict__ dst, size_t n16) {
  for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n16; i += (size_t)gridDim.x * blockDim.x) dst[i] = src[i];
}
__global__ void tiny_kernel(int* p) { if (threadIdx.x == 0) p[blockIdx.x] += 1; }

int main() {
  const size_t bytes = (size_t)2 << 30;  // 2 GiB working set: far beyond any cache
  uint4 *a, *b;
  float* out;
  int* ctr;
  CK(cudaMalloc(&a, bytes));
  CK(cudaMalloc(&b, bytes));
  CK(cudaMalloc(&out, 1 << 20));
  CK(cudaMalloc(&ctr, 1 << 16));
  CK(cudaMemset(a, 0, bytes));
  CK(cudaMemset(b, 0, bytes));
  const size_t n16 = bytes / 16;
  int dev = 0, sms = 0;
  CK(cudaGetDevice(&dev));
  CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev));
  const int grid = sms * 32;

  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));

  // warm (pipeline compile)
  read_bw_kernel<<<grid, 256>>>(a, n16, out);
  copy_kernel<<<grid, 256>>>(a, b, n16);
  tiny_kernel<<<1, 32>>>(ctr);
  CK(cudaDeviceSynchronize());

  // streaming read
  {
    const int reps = 5;
    CK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r) read_bw_kernel<<<grid, 256>>>(a, n16, out);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("read  bandwidth : %7.1f GB/s  (%zu MiB x %d in %.1f ms)\n", (double)bytes * reps / (ms * 1e-3) / 1e9, bytes >> 20, reps, ms);
  }
  // copy (read + write)
  {
    const int reps = 5;
    CK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r) copy_kernel<<<grid, 256>>>(a, b, n16);
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("copy  bandwidth : %7.1f GB/s  (2 x %zu MiB x %d in %.1f ms)\n", 2.0 * bytes * reps / (ms * 1e-3) / 1e9, bytes >> 20, reps, ms);
  }
  // blit copy through cudaMemcpyAsync D2D
  {
    const int reps = 5;
    CK(cudaEventRecord(e0));
    for (int r = 0; r < reps; ++r) CK(cudaMemcpyAsync(b, a, bytes, cudaMemcpyDeviceToDevice, 0));
    CK(cudaEventRecord(e1));
    CK(cudaEventSynchronize(e1));
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("blit  bandwidth : %7.1f GB/s  (cudaMemcpy D2D, 2 x %zu MiB x %d in %.1f ms)\n", 2.0 * bytes * reps / (ms * 1e-3) / 1e9, bytes >> 20, reps, ms);
  }
  // dispatch overhead: N tiny dependent kernels in one stream, GPU time and wall time
  for (int n : {1000, 4000}) {
    const double t0 = now();
    CK(cudaEventRecord(e0));
    for (int i = 0; i < n; ++i) tiny_kernel<<<1, 32>>>(ctr);
    CK(cudaEventRecord(e1));
    const double t_enc = now() - t0;
    CK(cudaEventSynchronize(e1));
    const double t_all = now() - t0;
    float ms = 0;
    CK(cudaEventElapsedTime(&ms, e0, e1));
    printf("dispatch x%5d : GPU %6.2f us/kernel, encode %6.2f us/kernel (CPU), wall %6.2f us/kernel\n", n, ms * 1e3 / n,
           t_enc * 1e6 / n, t_all * 1e6 / n);
  }
  // sync round trip
  {
    const int n = 200;
    const double t0 = now();
    for (int i = 0; i < n; ++i) { tiny_kernel<<<1, 32>>>(ctr); CK(cudaDeviceSynchronize()); }
    printf("launch+sync     : %6.1f us round trip\n", (now() - t0) * 1e6 / n);
  }
  {
    const int n = 200;
    int h = 0;
    const double t0 = now();
    for (int i = 0; i < n; ++i) { tiny_kernel<<<1, 32>>>(ctr); CK(cudaMemcpy(&h, ctr, 4, cudaMemcpyDeviceToHost)); }
    printf("launch+D2H 4B   : %6.1f us round trip\n", (now() - t0) * 1e6 / n);
  }
  return 0;
}
