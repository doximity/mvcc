// mvcc: minimal CUDA driver API surface. Most projects only need the types; the handful
// of entry points below are implemented in terms of the runtime.
#ifndef MVCC_CUDA_H
#define MVCC_CUDA_H
// clang's CUDA installation detector reads this literal line to learn the toolkit version.
#define CUDA_VERSION 12080
#include <mvcc/host_defines.h>
#include <stddef.h>
#include <stdint.h>

typedef int CUdevice;
typedef struct CUctx_st* CUcontext;
typedef struct CUmod_st* CUmodule;
typedef struct CUfunc_st* CUfunction;
typedef struct CUstream_st* CUstream;
typedef struct CUevent_st* CUevent;
typedef unsigned long long CUdeviceptr;
typedef enum cudaError_enum {
  CUDA_SUCCESS = 0, CUDA_ERROR_INVALID_VALUE = 1, CUDA_ERROR_OUT_OF_MEMORY = 2, CUDA_ERROR_NOT_INITIALIZED = 3, CUDA_ERROR_DEINITIALIZED = 4,
  CUDA_ERROR_NO_DEVICE = 100, CUDA_ERROR_INVALID_DEVICE = 101, CUDA_ERROR_INVALID_IMAGE = 200, CUDA_ERROR_INVALID_CONTEXT = 201,
  CUDA_ERROR_NOT_FOUND = 500, CUDA_ERROR_NOT_READY = 600, CUDA_ERROR_ILLEGAL_ADDRESS = 700, CUDA_ERROR_LAUNCH_FAILED = 719,
  CUDA_ERROR_NOT_SUPPORTED = 801, CUDA_ERROR_UNKNOWN = 999
} CUresult;

#ifdef __cplusplus
extern "C" {
#endif
CUresult cuInit(unsigned int flags);
CUresult cuDriverGetVersion(int* version);
CUresult cuDeviceGetCount(int* count);
CUresult cuDeviceGet(CUdevice* device, int ordinal);
CUresult cuDeviceGetName(char* name, int len, CUdevice dev);
CUresult cuDeviceTotalMem(size_t* bytes, CUdevice dev);
CUresult cuDevicePrimaryCtxRetain(CUcontext* pctx, CUdevice dev);
CUresult cuDevicePrimaryCtxRelease(CUdevice dev);
CUresult cuCtxGetCurrent(CUcontext* pctx);
CUresult cuCtxSetCurrent(CUcontext ctx);
CUresult cuCtxSynchronize(void);
CUresult cuGetErrorString(CUresult error, const char** pStr);
CUresult cuGetErrorName(CUresult error, const char** pStr);
CUresult cuMemAlloc(CUdeviceptr* dptr, size_t bytesize);
CUresult cuMemFree(CUdeviceptr dptr);
CUresult cuMemcpyHtoD(CUdeviceptr dst, const void* src, size_t n);
CUresult cuMemcpyDtoH(void* dst, CUdeviceptr src, size_t n);
CUresult cuMemcpyDtoD(CUdeviceptr dst, CUdeviceptr src, size_t n);
#ifdef __cplusplus
}
#endif
#endif // MVCC_CUDA_H
