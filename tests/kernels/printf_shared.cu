// Device printf (no-op) and __shared__ T with default member initializers — two common CUDA
// forms clang rejects without mvcc's device printf and shared rewrite.
#include <cuda_runtime.h>
#include <stdio.h>

struct Biases {
    int count = 0;
    int ids[4] = {};
};

__host__ __device__ inline void note(const char* s) { printf("%s\n", s); }

__global__ void k(int* out) {
    __shared__ Biases s;
    if (threadIdx.x == 0) {
        s.count = 1;
        s.ids[0] = 7;
        note("device");
    }
    __syncthreads();
    if (threadIdx.x == 0) *out = s.count + s.ids[0];
}

int main() {
    int *d, h = 0;
    cudaMalloc(&d, sizeof(int));
    k<<<1, 32>>>(d);
    cudaMemcpy(&h, d, sizeof(int), cudaMemcpyDeviceToHost);
    printf("host %d\n", h);
    return h == 8 ? 0 : 1;
}
