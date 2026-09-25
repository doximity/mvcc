// Device half of the CMake test project.
#include <cuda_runtime.h>
__global__ void scale(float* x, float s, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] *= s; }
extern "C" cudaError_t run_scale(float* x, float s, int n, cudaStream_t st) { scale<<<(n + 127) / 128, 128, 0, st>>>(x, s, n); return cudaGetLastError(); }
