// mvcc: NCCL API over the logical devices. Implemented by libcudart.dylib (libnccl.dylib is the same
// library); see crates/mvcc-cudart/src/nccl.rs for what is implemented: communicators, groups, the copy
// collectives (AllGather, Broadcast/Bcast, Send/Recv) and AllReduce / Reduce / ReduceScatter as
// gather-and-sum or gather-and-avg (ncclSum / ncclAvg on f32/i32). Prod / max / min are refused.
#ifndef NCCL_H_
#define NCCL_H_

#include <cuda_runtime.h>
#include <cuda_fp16.h>

#define NCCL_MAJOR 2
#define NCCL_MINOR 28
#define NCCL_PATCH 3
#define NCCL_SUFFIX ""
#define NCCL_VERSION_CODE 22803
#define NCCL_VERSION(X, Y, Z) (((X) <= 2 && (Y) <= 8) ? (X) * 1000 + (Y) * 100 + (Z) : (X) * 10000 + (Y) * 100 + (Z))

#ifdef __cplusplus
extern "C" {
#endif

typedef struct ncclComm* ncclComm_t;

#define NCCL_UNIQUE_ID_BYTES 128
typedef struct { char internal[NCCL_UNIQUE_ID_BYTES]; } ncclUniqueId;

typedef enum {
    ncclSuccess = 0,
    ncclUnhandledCudaError = 1,
    ncclSystemError = 2,
    ncclInternalError = 3,
    ncclInvalidArgument = 4,
    ncclInvalidUsage = 5,
    ncclRemoteError = 6,
    ncclInProgress = 7,
    ncclNumResults = 8
} ncclResult_t;

typedef enum {
    ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3, ncclAvg = 4, ncclNumOps = 5, ncclMaxRedOp = 0x7fffffff
} ncclRedOp_t;

typedef enum {
    ncclInt8 = 0, ncclChar = 0,
    ncclUint8 = 1,
    ncclInt32 = 2, ncclInt = 2,
    ncclUint32 = 3,
    ncclInt64 = 4,
    ncclUint64 = 5,
    ncclFloat16 = 6, ncclHalf = 6,
    ncclFloat32 = 7, ncclFloat = 7,
    ncclFloat64 = 8, ncclDouble = 8,
    ncclBfloat16 = 9,
    ncclFloat8e4m3 = 10,
    ncclFloat8e5m2 = 11,
    ncclNumTypes = 12
} ncclDataType_t;

typedef enum { ncclScalarDevice = 0, ncclScalarHostImmediate = 1 } ncclScalarResidence_t;

typedef struct ncclConfig_v21700 {
    size_t size;
    unsigned int magic;
    unsigned int version;
    int blocking;
    int cgaClusterSize;
    int minCTAs;
    int maxCTAs;
    const char* netName;
    int splitShare;
} ncclConfig_t;
#define NCCL_CONFIG_INITIALIZER { sizeof(ncclConfig_t), 0xcafebeef, NCCL_VERSION_CODE, -2147483648, -2147483648, -2147483648, -2147483648, 0, -2147483648 }

ncclResult_t ncclGetVersion(int* version);
ncclResult_t ncclGetUniqueId(ncclUniqueId* uniqueId);
ncclResult_t ncclCommInitRank(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank);
ncclResult_t ncclCommInitRankConfig(ncclComm_t* comm, int nranks, ncclUniqueId commId, int rank, ncclConfig_t* config);
ncclResult_t ncclCommInitAll(ncclComm_t* comm, int ndev, const int* devlist);
ncclResult_t ncclCommFinalize(ncclComm_t comm);
ncclResult_t ncclCommDestroy(ncclComm_t comm);
ncclResult_t ncclCommAbort(ncclComm_t comm);
ncclResult_t ncclCommCount(const ncclComm_t comm, int* count);
ncclResult_t ncclCommCuDevice(const ncclComm_t comm, int* device);
ncclResult_t ncclCommUserRank(const ncclComm_t comm, int* rank);
ncclResult_t ncclCommGetAsyncError(ncclComm_t comm, ncclResult_t* asyncError);
ncclResult_t ncclCommRegister(const ncclComm_t comm, void* buff, size_t size, void** handle);
ncclResult_t ncclCommDeregister(const ncclComm_t comm, void* handle);
ncclResult_t ncclMemAlloc(void** ptr, size_t size);
ncclResult_t ncclMemFree(void* ptr);
const char* ncclGetErrorString(ncclResult_t result);
const char* ncclGetLastError(ncclComm_t comm);

ncclResult_t ncclGroupStart(void);
ncclResult_t ncclGroupEnd(void);

ncclResult_t ncclAllGather(const void* sendbuff, void* recvbuff, size_t sendcount, ncclDataType_t datatype, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBroadcast(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclBcast(void* buff, size_t count, ncclDataType_t datatype, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclSend(const void* sendbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclRecv(void* recvbuff, size_t count, ncclDataType_t datatype, int peer, ncclComm_t comm, cudaStream_t stream);
// reductions: not implemented over the logical devices (ncclInvalidUsage; a program sums gathered slabs itself)
ncclResult_t ncclAllReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduce(const void* sendbuff, void* recvbuff, size_t count, ncclDataType_t datatype, ncclRedOp_t op, int root, ncclComm_t comm, cudaStream_t stream);
ncclResult_t ncclReduceScatter(const void* sendbuff, void* recvbuff, size_t recvcount, ncclDataType_t datatype, ncclRedOp_t op, ncclComm_t comm, cudaStream_t stream);

// the p* aliases NCCL declares
#define pncclGetVersion ncclGetVersion
#define pncclGetUniqueId ncclGetUniqueId
#define pncclCommInitRank ncclCommInitRank
#define pncclCommInitAll ncclCommInitAll
#define pncclCommDestroy ncclCommDestroy
#define pncclAllGather ncclAllGather
#define pncclAllReduce ncclAllReduce
#define pncclBroadcast ncclBroadcast
#define pncclBcast ncclBcast
#define pncclReduce ncclReduce
#define pncclReduceScatter ncclReduceScatter
#define pncclSend ncclSend
#define pncclRecv ncclRecv
#define pncclGroupStart ncclGroupStart
#define pncclGroupEnd ncclGroupEnd

#ifdef __cplusplus
}
#endif

#endif  // NCCL_H_
