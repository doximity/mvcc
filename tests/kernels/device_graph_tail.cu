// Device cudaGraphLaunch lowering: __mvcc_device_graph_launch -> ABI offsets -> runtime graph replay before dispatch.
#include <cuda_runtime.h>
#include <cuda_device_runtime_api.h>
#include <cstdio>

struct LaunchArgs {
    cudaGraphExec_t a;
    cudaGraphExec_t b;
};

__global__ void tail_kernel(LaunchArgs args) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    if (args.a) cudaGraphLaunch(args.a, cudaStreamGraphTailLaunch);
    if (args.b) cudaGraphLaunch(args.b, cudaStreamGraphTailLaunch);
}

__global__ void child_kernel(int* out) { if (threadIdx.x == 0) *out = 42; }

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { std::printf("fail %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); return 1; } } while (0)

int main() {
    int* dev = nullptr;
    CK(cudaMalloc(&dev, sizeof(int)));
    CK(cudaMemset(dev, 0, sizeof(int)));

    cudaStream_t s = nullptr;
    CK(cudaStreamCreate(&s));

    cudaGraph_t cg = nullptr;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    child_kernel<<<1, 32, 0, s>>>(dev);
    CK(cudaStreamEndCapture(s, &cg));
    cudaGraphExec_t child = nullptr;
    CK(cudaGraphInstantiate(&child, cg, 0));
    CK(cudaGraphDestroy(cg));

    LaunchArgs la{child, nullptr};
    cudaGraph_t tg = nullptr;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal));
    tail_kernel<<<1, 1, 0, s>>>(la);
    CK(cudaStreamEndCapture(s, &tg));
    cudaGraphExec_t tail = nullptr;
    CK(cudaGraphInstantiate(&tail, tg, 0));
    CK(cudaGraphDestroy(tg));

    CK(cudaGraphLaunch(tail, s));
    CK(cudaStreamSynchronize(s));
    int h = 0;
    CK(cudaMemcpy(&h, dev, sizeof(int), cudaMemcpyDeviceToHost));
    if (h != 42) {
        std::printf("expected 42 got %d\n", h);
        return 2;
    }
    CK(cudaGraphExecDestroy(tail));
    CK(cudaGraphExecDestroy(child));
    CK(cudaStreamDestroy(s));
    CK(cudaFree(dev));
    return 0;
}
