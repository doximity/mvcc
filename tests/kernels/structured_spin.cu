// Early return + nested ifs + volatile spin (the grid_sync shape). LLVM's PDT
// IDom of those branches is the virtual root; structured emission must still
// produce if/for, not a switch state machine.
#include <cuda_runtime.h>

__global__ void structured_spin(int layer, int n_layers, int* flag, volatile int* gen, int* out, int n) {
    if (layer < 0 || layer >= n_layers) return;
    if (flag) {
        if (flag[0] && flag[1]) {
            if ((int)blockIdx.x < n) out[blockIdx.x] = 1;
            __syncthreads();
            if (threadIdx.x == 0) {
                const int g = *gen;
                if (atomicAdd(flag + 2, 1) == 0) {
                    *gen = g + 1;
                } else {
                    while (*gen == g) {
                    }
                }
            }
            __syncthreads();
        }
        for (int i = (int)blockIdx.x; i < n; i += (int)gridDim.x) out[i] = i;
    }
}
