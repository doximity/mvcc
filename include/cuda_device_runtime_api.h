// mvcc: device-side graph launch is lowered in mvcc-ir2msl to host stream replay (see KernelABI
// device_graph_launch_exec_offsets); this entry point exists so the call survives LLVM IR.
#ifndef MVCC_DEVICE_RUNTIME_API_H
#define MVCC_DEVICE_RUNTIME_API_H

#include <cuda_runtime_api.h>

#if defined(__CUDA__) && defined(__clang__)

#ifdef __cplusplus
extern "C" {
#endif

__device__ cudaError_t __mvcc_device_graph_launch(cudaGraphExec_t graphExec, cudaStream_t stream);

__device__ __forceinline__ cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream) {
  return __mvcc_device_graph_launch(graphExec, stream);
}

#ifdef __cplusplus
}
#endif

#endif // __CUDA__ && __clang__
#endif // MVCC_DEVICE_RUNTIME_API_H
