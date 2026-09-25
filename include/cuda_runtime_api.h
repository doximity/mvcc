// mvcc: CUDA Runtime API surface implemented by libcudart (Metal 4 backend).
// Every function here is implemented in crates/mvcc-cudart; anything declared but unimplemented returns
// cudaErrorNotSupported so the failure is explicit.
#ifndef MVCC_CUDA_RUNTIME_API_H
#define MVCC_CUDA_RUNTIME_API_H
#include <driver_types.h>
#include <vector_types.h>

// Default arguments as in the CUDA header (C++ only).
#ifdef __cplusplus
#define __dv(v) = v
extern "C" {
#else
#define __dv(v)
#endif

// --- device management ---
cudaError_t cudaGetDeviceCount(int* count);
cudaError_t cudaGetDevice(int* device);
cudaError_t cudaSetDevice(int device);
cudaError_t cudaGetDeviceProperties(struct cudaDeviceProp* prop, int device);
cudaError_t cudaGetDeviceProperties_v2(struct cudaDeviceProp* prop, int device);
cudaError_t cudaDeviceGetAttribute(int* value, enum cudaDeviceAttr attr, int device);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceReset(void);
cudaError_t cudaDeviceSetLimit(enum cudaLimit limit, size_t value);
cudaError_t cudaDeviceGetLimit(size_t* pValue, enum cudaLimit limit);
cudaError_t cudaDeviceSetCacheConfig(enum cudaFuncCache cacheConfig);
cudaError_t cudaDeviceGetCacheConfig(enum cudaFuncCache* pCacheConfig);
cudaError_t cudaDeviceSetSharedMemConfig(enum cudaSharedMemConfig config);
cudaError_t cudaDeviceGetStreamPriorityRange(int* leastPriority, int* greatestPriority);
cudaError_t cudaDeviceGetPCIBusId(char* pciBusId, int len, int device);
cudaError_t cudaDeviceCanAccessPeer(int* canAccessPeer, int device, int peerDevice);
cudaError_t cudaDeviceEnablePeerAccess(int peerDevice, unsigned int flags);
cudaError_t cudaDeviceDisablePeerAccess(int peerDevice);
cudaError_t cudaSetDeviceFlags(unsigned int flags);
cudaError_t cudaGetDeviceFlags(unsigned int* flags);
cudaError_t cudaChooseDevice(int* device, const struct cudaDeviceProp* prop);
cudaError_t cudaDriverGetVersion(int* driverVersion);
cudaError_t cudaRuntimeGetVersion(int* runtimeVersion);
cudaError_t cudaDeviceGetDefaultMemPool(cudaMemPool_t* memPool, int device);
cudaError_t cudaDeviceSetMemPool(int device, cudaMemPool_t memPool);
cudaError_t cudaDeviceGetMemPool(cudaMemPool_t* memPool, int device);
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void* value);
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t memPool, enum cudaMemPoolAttr attr, void* value);

// --- errors ---
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
const char* cudaGetErrorName(cudaError_t error);
const char* cudaGetErrorString(cudaError_t error);

// --- memory ---
cudaError_t cudaMalloc(void** devPtr, size_t size);
cudaError_t cudaFree(void* devPtr);
cudaError_t cudaMallocHost(void** ptr, size_t size);
cudaError_t cudaFreeHost(void* ptr);
cudaError_t cudaHostAlloc(void** pHost, size_t size, unsigned int flags);
cudaError_t cudaHostRegister(void* ptr, size_t size, unsigned int flags);
cudaError_t cudaHostUnregister(void* ptr);
cudaError_t cudaHostGetDevicePointer(void** pDevice, void* pHost, unsigned int flags);
cudaError_t cudaHostGetFlags(unsigned int* pFlags, void* pHost);
cudaError_t cudaMallocManaged(void** devPtr, size_t size, unsigned int flags __dv(cudaMemAttachGlobal));
cudaError_t cudaMallocPitch(void** devPtr, size_t* pitch, size_t width, size_t height);
cudaError_t cudaMallocAsync(void** devPtr, size_t size, cudaStream_t stream);
cudaError_t cudaFreeAsync(void* devPtr, cudaStream_t stream);
cudaError_t cudaMemGetInfo(size_t* free, size_t* total);
cudaError_t cudaMemcpy(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t count, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
cudaError_t cudaMemcpy2D(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind);
cudaError_t cudaMemcpy2DAsync(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, enum cudaMemcpyKind kind, cudaStream_t stream __dv(0));
cudaError_t cudaMemcpyPeer(void* dst, int dstDevice, const void* src, int srcDevice, size_t count);
cudaError_t cudaMemcpyPeerAsync(void* dst, int dstDevice, const void* src, int srcDevice, size_t count, cudaStream_t stream __dv(0));
cudaError_t cudaMemcpyToSymbol(const void* symbol, const void* src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice));
cudaError_t cudaMemcpyFromSymbol(void* dst, const void* symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost));
cudaError_t cudaMemcpyToSymbolAsync(const void* symbol, const void* src, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyHostToDevice), cudaStream_t stream __dv(0));
cudaError_t cudaMemcpyFromSymbolAsync(void* dst, const void* symbol, size_t count, size_t offset __dv(0), enum cudaMemcpyKind kind __dv(cudaMemcpyDeviceToHost), cudaStream_t stream __dv(0));
cudaError_t cudaMemset(void* devPtr, int value, size_t count);
cudaError_t cudaMemsetAsync(void* devPtr, int value, size_t count, cudaStream_t stream __dv(0));
cudaError_t cudaMemset2D(void* devPtr, size_t pitch, int value, size_t width, size_t height);
cudaError_t cudaMemset2DAsync(void* devPtr, size_t pitch, int value, size_t width, size_t height, cudaStream_t stream __dv(0));
cudaError_t cudaMemPrefetchAsync(const void* devPtr, size_t count, int dstDevice, cudaStream_t stream __dv(0));
cudaError_t cudaMemAdvise(const void* devPtr, size_t count, enum cudaMemoryAdvise advice, int device);
cudaError_t cudaPointerGetAttributes(struct cudaPointerAttributes* attributes, const void* ptr);
cudaError_t cudaGetSymbolAddress(void** devPtr, const void* symbol);
cudaError_t cudaGetSymbolSize(size_t* size, const void* symbol);

// --- streams ---
cudaError_t cudaStreamCreate(cudaStream_t* pStream);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t* pStream, unsigned int flags);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t* pStream, unsigned int flags, int priority);
cudaError_t cudaStreamDestroy(cudaStream_t stream);
cudaError_t cudaStreamSynchronize(cudaStream_t stream);
cudaError_t cudaStreamQuery(cudaStream_t stream);
cudaError_t cudaStreamWaitEvent(cudaStream_t stream, cudaEvent_t event, unsigned int flags __dv(0));
cudaError_t cudaStreamAddCallback(cudaStream_t stream, cudaStreamCallback_t callback, void* userData, unsigned int flags __dv(0));
cudaError_t cudaLaunchHostFunc(cudaStream_t stream, cudaHostFn_t fn, void* userData);
cudaError_t cudaStreamGetFlags(cudaStream_t stream, unsigned int* flags);
cudaError_t cudaStreamGetPriority(cudaStream_t stream, int* priority);
cudaError_t cudaStreamBeginCapture(cudaStream_t stream, enum cudaStreamCaptureMode mode __dv(cudaStreamCaptureModeGlobal));
cudaError_t cudaStreamEndCapture(cudaStream_t stream, cudaGraph_t* pGraph);
cudaError_t cudaStreamIsCapturing(cudaStream_t stream, enum cudaStreamCaptureStatus* pCaptureStatus);
cudaError_t cudaStreamGetCaptureInfo(cudaStream_t stream, enum cudaStreamCaptureStatus* captureStatus_out, unsigned long long* id_out __dv(0), cudaGraph_t* graph_out __dv(0), const cudaGraphNode_t** dependencies_out __dv(0), size_t* numDependencies_out __dv(0));
cudaError_t cudaThreadExchangeStreamCaptureMode(enum cudaStreamCaptureMode* mode);

// --- events ---
cudaError_t cudaEventCreate(cudaEvent_t* event);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t* event, unsigned int flags);
cudaError_t cudaEventDestroy(cudaEvent_t event);
cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream __dv(0));
cudaError_t cudaEventRecordWithFlags(cudaEvent_t event, cudaStream_t stream __dv(0), unsigned int flags __dv(0));
cudaError_t cudaEventQuery(cudaEvent_t event);
cudaError_t cudaEventSynchronize(cudaEvent_t event);
cudaError_t cudaEventElapsedTime(float* ms, cudaEvent_t start, cudaEvent_t end);

// --- execution ---
cudaError_t cudaLaunchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem __dv(0), cudaStream_t stream __dv(0));
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t* config, const void* func, void** args);
cudaError_t cudaLaunchCooperativeKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem __dv(0), cudaStream_t stream __dv(0));
cudaError_t cudaFuncSetAttribute(const void* func, enum cudaFuncAttribute attr, int value);
cudaError_t cudaFuncGetAttributes(struct cudaFuncAttributes* attr, const void* func);
cudaError_t cudaFuncSetCacheConfig(const void* func, enum cudaFuncCache cacheConfig);
cudaError_t cudaFuncSetSharedMemConfig(const void* func, enum cudaSharedMemConfig config);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int* numBlocks, const void* func, int blockSize, size_t dynamicSMemSize, unsigned int flags __dv(0));
cudaError_t cudaOccupancyAvailableDynamicSMemPerBlock(size_t* dynamicSmemSize, const void* func, int numBlocks, int blockSize);

// --- graphs ---
cudaError_t cudaGraphCreate(cudaGraph_t* pGraph, unsigned int flags);
cudaError_t cudaGraphDestroy(cudaGraph_t graph);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags __dv(0));
cudaError_t cudaGraphInstantiateWithFlags(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, unsigned long long flags);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t graphExec);
cudaError_t cudaGraphLaunch(cudaGraphExec_t graphExec, cudaStream_t stream);
cudaError_t cudaGraphUpload(cudaGraphExec_t graphExec, cudaStream_t stream);
cudaError_t cudaGraphGetNodes(cudaGraph_t graph, cudaGraphNode_t* nodes, size_t* numNodes);
cudaError_t cudaGraphAddKernelNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaKernelNodeParams* pNodeParams);
cudaError_t cudaGraphAddMemcpyNode1D(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, void* dst, const void* src, size_t count, enum cudaMemcpyKind kind);
cudaError_t cudaGraphAddMemsetNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies, const struct cudaMemsetParams* pMemsetParams);
cudaError_t cudaGraphAddEmptyNode(cudaGraphNode_t* pGraphNode, cudaGraph_t graph, const cudaGraphNode_t* pDependencies, size_t numDependencies);
cudaError_t cudaGraphExecKernelNodeSetParams(cudaGraphExec_t hGraphExec, cudaGraphNode_t node, const struct cudaKernelNodeParams* pNodeParams);
cudaError_t cudaGraphKernelNodeSetParams(cudaGraphNode_t node, const struct cudaKernelNodeParams* pNodeParams);
cudaError_t cudaGraphKernelNodeGetParams(cudaGraphNode_t node, struct cudaKernelNodeParams* pNodeParams);
cudaError_t cudaGraphNodeGetType(cudaGraphNode_t node, enum cudaGraphNodeType* pType);
cudaError_t cudaGraphDebugDotPrint(cudaGraph_t graph, const char* path, unsigned int flags);

// --- profiler / misc no-ops ---
cudaError_t cudaProfilerStart(void);
cudaError_t cudaProfilerStop(void);
cudaError_t cudaThreadSynchronize(void);
cudaError_t cudaThreadExit(void);

// --- internal registration ABI used by clang-generated host code ---
void** __cudaRegisterFatBinary(void* fatCubin);
void __cudaRegisterFatBinaryEnd(void** fatCubinHandle);
void __cudaUnregisterFatBinary(void** fatCubinHandle);
void __cudaRegisterFunction(void** fatCubinHandle, const char* hostFun, char* deviceFun, const char* deviceName, int thread_limit, uint3* tid, uint3* bid, dim3* bDim, dim3* gDim, int* wSize);
void __cudaRegisterVar(void** fatCubinHandle, char* hostVar, char* deviceAddress, const char* deviceName, int ext, size_t size, int constant, int global);
void __cudaRegisterManagedVar(void** fatCubinHandle, void** hostVarPtrAddress, char* deviceAddress, const char* deviceName, int ext, size_t size, int constant, int global);
void __cudaRegisterSurface(void** fatCubinHandle, const void* hostVar, const void** deviceAddress, const char* deviceName, int dim, int ext);
void __cudaRegisterTexture(void** fatCubinHandle, const void* hostVar, const void** deviceAddress, const char* deviceName, int dim, int norm, int ext);
unsigned __cudaPushCallConfiguration(dim3 gridDim, dim3 blockDim, size_t sharedMem __dv(0), cudaStream_t stream __dv(0));
cudaError_t __cudaPopCallConfiguration(dim3* gridDim, dim3* blockDim, size_t* sharedMem, void* stream);

// --- mvcc extensions (mvcc_ prefix; not part of CUDA) ---
// Returns the Metal device / command queue as opaque pointers for interop (id<MTLDevice>, id<MTLCommandQueue>).
void* mvccGetMetalDevice(void);
void* mvccGetMetalCommandQueue(cudaStream_t stream);
// Returns the MTLBuffer and offset backing a device pointer, or NULL.
void* mvccGetMetalBuffer(const void* devPtr, size_t* offset);
// Runtime statistics for tuning.
void mvccGetStats(unsigned long long* launches, unsigned long long* commandBuffers, unsigned long long* pipelineCompiles);

#ifdef __cplusplus
}
// Pre-12.0 five-argument form, kept by CUDA 12 as a C++ overload.
static inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t* pGraphExec, cudaGraph_t graph, cudaGraphNode_t* pErrorNode, char* pLogBuffer, size_t bufferSize) {
  (void)pErrorNode; if (pLogBuffer && bufferSize) pLogBuffer[0] = 0;
  return cudaGraphInstantiate(pGraphExec, graph, 0ull);
}
#endif
#undef __dv
#endif // MVCC_CUDA_RUNTIME_API_H
