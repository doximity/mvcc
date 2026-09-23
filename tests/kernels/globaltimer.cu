// Inline PTX %globaltimer, as kernel timing code reads it; Metal has no CUDA timer, so the read is 0.
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void k(unsigned long long* out) {
    unsigned long long t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
    if (threadIdx.x == 0) *out = t;
}

int main() {
    unsigned long long *d, h = 1;
    cudaMalloc(&d, sizeof(h));
    k<<<1, 32>>>(d);
    cudaMemcpy(&h, d, sizeof(h), cudaMemcpyDeviceToHost);
    printf("globaltimer %llu\n", (unsigned long long)h);
    return h == 0 ? 0 : 1;
}
