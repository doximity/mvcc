// Stream order with 1x1 grids: a kernel that writes at a device-side cursor, then a kernel on the same stream that
// advances the cursor. The second must not overtake the first.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace {

__global__ void store_at_cursor(int* out, const int* cursor, int offset, const int* src, int n) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int base = cursor[0];
    for (int i = 0; i < n; ++i) {
        out[base + offset + i] = src[i];
    }
}

__global__ void advance_cursor(const int* accepted, int* cursor, int* total) {
    if (blockIdx.x != 0 || threadIdx.x != 0) {
        return;
    }
    const int step = accepted[0] + 1;
    cursor[0] += step;
    total[0] += step;
}

} // namespace

int main() {
    int* out = nullptr;
    int* cursor = nullptr;
    int* src = nullptr;
    int* accepted = nullptr;
    int* total = nullptr;
    if (cudaMalloc(&out, 8 * sizeof(int)) != cudaSuccess || cudaMalloc(&cursor, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&src, 3 * sizeof(int)) != cudaSuccess || cudaMalloc(&accepted, sizeof(int)) != cudaSuccess ||
        cudaMalloc(&total, sizeof(int)) != cudaSuccess) {
        std::fprintf(stderr, "malloc failed\n");
        return 1;
    }
    const int h_out[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    const int h_cursor = 2;
    const int h_src[3] = {11, 22, 33};
    const int h_accepted = 1;
    const int h_total = 5;
    cudaMemcpy(out, h_out, sizeof(h_out), cudaMemcpyHostToDevice);
    cudaMemcpy(cursor, &h_cursor, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(src, h_src, 3 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(accepted, &h_accepted, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(total, &h_total, sizeof(int), cudaMemcpyHostToDevice);

    store_at_cursor<<<1, 1>>>(out, cursor, 0, src, 3);
    advance_cursor<<<1, 1>>>(accepted, cursor, total);
    cudaDeviceSynchronize();

    int got[8] = {};
    int got_cursor = -1;
    int got_total = -1;
    cudaMemcpy(got, out, sizeof(got), cudaMemcpyDeviceToHost);
    cudaMemcpy(&got_cursor, cursor, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&got_total, total, sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(out);
    cudaFree(cursor);
    cudaFree(src);
    cudaFree(accepted);
    cudaFree(total);

    if (got[2] != 11 || got[3] != 22 || got[4] != 33) {
        std::fprintf(stderr, "store failed: got %d %d %d\n", got[2], got[3], got[4]);
        return 2;
    }
    if (got_cursor != 4 || got_total != 7) {
        std::fprintf(stderr, "advance failed: cursor=%d total=%d\n", got_cursor, got_total);
        return 3;
    }
    std::printf("single_thread_order ok\n");
    return 0;
}
