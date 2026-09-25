// mvcc: CUDA runtime API types and enums. Names and numeric values follow the public
// CUDA Runtime API reference so that code comparing against them behaves identically.
#ifndef MVCC_DRIVER_TYPES_H
#define MVCC_DRIVER_TYPES_H
#include <mvcc/host_defines.h>
#include <stddef.h>
#include <stdint.h>
#include <vector_types.h>

typedef uint32_t cuuint32_t;
typedef uint64_t cuuint64_t;

enum cudaError {
  cudaSuccess = 0,
  cudaErrorInvalidValue = 1,
  cudaErrorMemoryAllocation = 2,
  cudaErrorInitializationError = 3,
  cudaErrorCudartUnloading = 4,
  cudaErrorProfilerDisabled = 5,
  cudaErrorInvalidConfiguration = 9,
  cudaErrorInvalidPitchValue = 12,
  cudaErrorInvalidSymbol = 13,
  cudaErrorInvalidHostPointer = 16,
  cudaErrorInvalidDevicePointer = 17,
  cudaErrorInvalidTexture = 18,
  cudaErrorInvalidTextureBinding = 19,
  cudaErrorInvalidChannelDescriptor = 20,
  cudaErrorInvalidMemcpyDirection = 21,
  cudaErrorAddressOfConstant = 22,
  cudaErrorInvalidFilterSetting = 26,
  cudaErrorInvalidNormSetting = 27,
  cudaErrorStubLibrary = 34,
  cudaErrorInsufficientDriver = 35,
  cudaErrorCallRequiresNewerDriver = 36,
  cudaErrorInvalidSurface = 37,
  cudaErrorDuplicateVariableName = 43,
  cudaErrorDuplicateTextureName = 44,
  cudaErrorDuplicateSurfaceName = 45,
  cudaErrorDevicesUnavailable = 46,
  cudaErrorIncompatibleDriverContext = 49,
  cudaErrorMissingConfiguration = 52,
  cudaErrorLaunchMaxDepthExceeded = 65,
  cudaErrorSyncDepthExceeded = 68,
  cudaErrorLaunchPendingCountExceeded = 69,
  cudaErrorInvalidDeviceFunction = 98,
  cudaErrorNoDevice = 100,
  cudaErrorInvalidDevice = 101,
  cudaErrorDeviceNotLicensed = 102,
  cudaErrorSoftwareValidityNotEstablished = 103,
  cudaErrorStartupFailure = 127,
  cudaErrorInvalidKernelImage = 200,
  cudaErrorDeviceUninitialized = 201,
  cudaErrorMapBufferObjectFailed = 205,
  cudaErrorUnmapBufferObjectFailed = 206,
  cudaErrorArrayIsMapped = 207,
  cudaErrorAlreadyMapped = 208,
  cudaErrorNoKernelImageForDevice = 209,
  cudaErrorAlreadyAcquired = 210,
  cudaErrorNotMapped = 211,
  cudaErrorNotMappedAsArray = 212,
  cudaErrorNotMappedAsPointer = 213,
  cudaErrorECCUncorrectable = 214,
  cudaErrorUnsupportedLimit = 215,
  cudaErrorDeviceAlreadyInUse = 216,
  cudaErrorPeerAccessUnsupported = 217,
  cudaErrorInvalidPtx = 218,
  cudaErrorInvalidGraphicsContext = 219,
  cudaErrorNvlinkUncorrectable = 220,
  cudaErrorJitCompilerNotFound = 221,
  cudaErrorUnsupportedPtxVersion = 222,
  cudaErrorJitCompilationDisabled = 223,
  cudaErrorUnsupportedExecAffinity = 224,
  cudaErrorUnsupportedDevSideSync = 225,
  cudaErrorInvalidSource = 300,
  cudaErrorFileNotFound = 301,
  cudaErrorSharedObjectSymbolNotFound = 302,
  cudaErrorSharedObjectInitFailed = 303,
  cudaErrorOperatingSystem = 304,
  cudaErrorInvalidResourceHandle = 400,
  cudaErrorIllegalState = 401,
  cudaErrorSymbolNotFound = 500,
  cudaErrorNotReady = 600,
  cudaErrorIllegalAddress = 700,
  cudaErrorLaunchOutOfResources = 701,
  cudaErrorLaunchTimeout = 702,
  cudaErrorLaunchIncompatibleTexturing = 703,
  cudaErrorPeerAccessAlreadyEnabled = 704,
  cudaErrorPeerAccessNotEnabled = 705,
  cudaErrorSetOnActiveProcess = 708,
  cudaErrorContextIsDestroyed = 709,
  cudaErrorAssert = 710,
  cudaErrorTooManyPeers = 711,
  cudaErrorHostMemoryAlreadyRegistered = 712,
  cudaErrorHostMemoryNotRegistered = 713,
  cudaErrorHardwareStackError = 714,
  cudaErrorIllegalInstruction = 715,
  cudaErrorMisalignedAddress = 716,
  cudaErrorInvalidAddressSpace = 717,
  cudaErrorInvalidPc = 718,
  cudaErrorLaunchFailure = 719,
  cudaErrorCooperativeLaunchTooLarge = 720,
  cudaErrorNotPermitted = 800,
  cudaErrorNotSupported = 801,
  cudaErrorSystemNotReady = 802,
  cudaErrorSystemDriverMismatch = 803,
  cudaErrorCompatNotSupportedOnDevice = 804,
  cudaErrorMpsConnectionFailed = 805,
  cudaErrorMpsRpcFailure = 806,
  cudaErrorMpsServerNotReady = 807,
  cudaErrorMpsMaxClientsReached = 808,
  cudaErrorMpsMaxConnectionsReached = 809,
  cudaErrorMpsClientTerminated = 810,
  cudaErrorCdpNotSupported = 811,
  cudaErrorCdpVersionMismatch = 812,
  cudaErrorStreamCaptureUnsupported = 900,
  cudaErrorStreamCaptureInvalidated = 901,
  cudaErrorStreamCaptureMerge = 902,
  cudaErrorStreamCaptureUnmatched = 903,
  cudaErrorStreamCaptureUnjoined = 904,
  cudaErrorStreamCaptureIsolation = 905,
  cudaErrorStreamCaptureImplicit = 906,
  cudaErrorCapturedEvent = 907,
  cudaErrorStreamCaptureWrongThread = 908,
  cudaErrorTimeout = 909,
  cudaErrorGraphExecUpdateFailure = 910,
  cudaErrorExternalDevice = 911,
  cudaErrorInvalidClusterSize = 912,
  cudaErrorUnknown = 999,
  cudaErrorApiFailureBase = 10000
};
typedef enum cudaError cudaError_t;

enum cudaMemcpyKind { cudaMemcpyHostToHost = 0, cudaMemcpyHostToDevice = 1, cudaMemcpyDeviceToHost = 2, cudaMemcpyDeviceToDevice = 3, cudaMemcpyDefault = 4 };

enum cudaMemoryType { cudaMemoryTypeUnregistered = 0, cudaMemoryTypeHost = 1, cudaMemoryTypeDevice = 2, cudaMemoryTypeManaged = 3 };

enum cudaComputeMode { cudaComputeModeDefault = 0, cudaComputeModeExclusive = 1, cudaComputeModeProhibited = 2, cudaComputeModeExclusiveProcess = 3 };

enum cudaLimit { cudaLimitStackSize = 0, cudaLimitPrintfFifoSize = 1, cudaLimitMallocHeapSize = 2, cudaLimitDevRuntimeSyncDepth = 3, cudaLimitDevRuntimePendingLaunchCount = 4, cudaLimitMaxL2FetchGranularity = 5, cudaLimitPersistingL2CacheSize = 6 };

enum cudaFuncCache { cudaFuncCachePreferNone = 0, cudaFuncCachePreferShared = 1, cudaFuncCachePreferL1 = 2, cudaFuncCachePreferEqual = 3 };
enum cudaSharedMemConfig { cudaSharedMemBankSizeDefault = 0, cudaSharedMemBankSizeFourByte = 1, cudaSharedMemBankSizeEightByte = 2 };

enum cudaFuncAttribute {
  cudaFuncAttributeMaxDynamicSharedMemorySize = 8,
  cudaFuncAttributePreferredSharedMemoryCarveout = 9,
  cudaFuncAttributeClusterDimMustBeSet = 10,
  cudaFuncAttributeRequiredClusterWidth = 11,
  cudaFuncAttributeRequiredClusterHeight = 12,
  cudaFuncAttributeRequiredClusterDepth = 13,
  cudaFuncAttributeNonPortableClusterSizeAllowed = 14,
  cudaFuncAttributeClusterSchedulingPolicyPreference = 15,
  cudaFuncAttributeMax
};

enum cudaDeviceAttr {
  cudaDevAttrMaxThreadsPerBlock = 1, cudaDevAttrMaxBlockDimX = 2, cudaDevAttrMaxBlockDimY = 3, cudaDevAttrMaxBlockDimZ = 4,
  cudaDevAttrMaxGridDimX = 5, cudaDevAttrMaxGridDimY = 6, cudaDevAttrMaxGridDimZ = 7, cudaDevAttrMaxSharedMemoryPerBlock = 8,
  cudaDevAttrTotalConstantMemory = 9, cudaDevAttrWarpSize = 10, cudaDevAttrMaxPitch = 11, cudaDevAttrMaxRegistersPerBlock = 12,
  cudaDevAttrClockRate = 13, cudaDevAttrTextureAlignment = 14, cudaDevAttrGpuOverlap = 15, cudaDevAttrMultiProcessorCount = 16,
  cudaDevAttrKernelExecTimeout = 17, cudaDevAttrIntegrated = 18, cudaDevAttrCanMapHostMemory = 19, cudaDevAttrComputeMode = 20,
  cudaDevAttrConcurrentKernels = 31, cudaDevAttrEccEnabled = 32, cudaDevAttrPciBusId = 33, cudaDevAttrPciDeviceId = 34, cudaDevAttrTccDriver = 35,
  cudaDevAttrMemoryClockRate = 36, cudaDevAttrGlobalMemoryBusWidth = 37, cudaDevAttrL2CacheSize = 38, cudaDevAttrMaxThreadsPerMultiProcessor = 39,
  cudaDevAttrAsyncEngineCount = 40, cudaDevAttrUnifiedAddressing = 41, cudaDevAttrPciDomainId = 50, cudaDevAttrTexturePitchAlignment = 51,
  cudaDevAttrComputeCapabilityMajor = 75, cudaDevAttrComputeCapabilityMinor = 76, cudaDevAttrStreamPrioritiesSupported = 78,
  cudaDevAttrGlobalL1CacheSupported = 79, cudaDevAttrLocalL1CacheSupported = 80, cudaDevAttrMaxSharedMemoryPerMultiprocessor = 81,
  cudaDevAttrMaxRegistersPerMultiprocessor = 82, cudaDevAttrManagedMemory = 83, cudaDevAttrIsMultiGpuBoard = 84, cudaDevAttrMultiGpuBoardGroupID = 85,
  cudaDevAttrHostNativeAtomicSupported = 86, cudaDevAttrSingleToDoublePrecisionPerfRatio = 87, cudaDevAttrPageableMemoryAccess = 88,
  cudaDevAttrConcurrentManagedAccess = 89, cudaDevAttrComputePreemptionSupported = 90, cudaDevAttrCanUseHostPointerForRegisteredMem = 91,
  cudaDevAttrCooperativeLaunch = 95, cudaDevAttrCooperativeMultiDeviceLaunch = 96, cudaDevAttrMaxSharedMemoryPerBlockOptin = 97,
  cudaDevAttrCanFlushRemoteWrites = 98, cudaDevAttrHostRegisterSupported = 99, cudaDevAttrPageableMemoryAccessUsesHostPageTables = 100,
  cudaDevAttrDirectManagedMemAccessFromHost = 101, cudaDevAttrMaxBlocksPerMultiprocessor = 106, cudaDevAttrMaxPersistingL2CacheSize = 108,
  cudaDevAttrMaxAccessPolicyWindowSize = 109, cudaDevAttrReservedSharedMemoryPerBlock = 111, cudaDevAttrSparseCudaArraySupported = 112,
  cudaDevAttrHostRegisterReadOnlySupported = 113, cudaDevAttrTimelineSemaphoreInteropSupported = 114, cudaDevAttrMemoryPoolsSupported = 115,
  cudaDevAttrGPUDirectRDMASupported = 116, cudaDevAttrGPUDirectRDMAFlushWritesOptions = 117, cudaDevAttrGPUDirectRDMAWritesOrdering = 118,
  cudaDevAttrMemoryPoolSupportedHandleTypes = 119, cudaDevAttrClusterLaunch = 120, cudaDevAttrDeferredMappingCudaArraySupported = 121,
  cudaDevAttrIpcEventSupport = 124, cudaDevAttrMemSyncDomainCount = 125, cudaDevAttrNumaConfig = 130, cudaDevAttrNumaId = 131,
  cudaDevAttrMpsEnabled = 133, cudaDevAttrHostNumaId = 134, cudaDevAttrMax
};

enum cudaMemoryAdvise { cudaMemAdviseSetReadMostly = 1, cudaMemAdviseUnsetReadMostly = 2, cudaMemAdviseSetPreferredLocation = 3, cudaMemAdviseUnsetPreferredLocation = 4, cudaMemAdviseSetAccessedBy = 5, cudaMemAdviseUnsetAccessedBy = 6 };

enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal = 0, cudaStreamCaptureModeThreadLocal = 1, cudaStreamCaptureModeRelaxed = 2 };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone = 0, cudaStreamCaptureStatusActive = 1, cudaStreamCaptureStatusInvalidated = 2 };

enum cudaGraphInstantiateFlags { cudaGraphInstantiateFlagAutoFreeOnLaunch = 1, cudaGraphInstantiateFlagUpload = 2, cudaGraphInstantiateFlagDeviceLaunch = 4, cudaGraphInstantiateFlagUseNodePriority = 8 };
enum cudaGraphNodeType { cudaGraphNodeTypeKernel = 0, cudaGraphNodeTypeMemcpy = 1, cudaGraphNodeTypeMemset = 2, cudaGraphNodeTypeHost = 3, cudaGraphNodeTypeGraph = 4, cudaGraphNodeTypeEmpty = 5, cudaGraphNodeTypeWaitEvent = 6, cudaGraphNodeTypeEventRecord = 7, cudaGraphNodeTypeCount };

enum cudaGraphDebugDotFlags {
  cudaGraphDebugDotFlagsNone = 0x0,
  cudaGraphDebugDotFlagsVerbose = 0x1,
  cudaGraphDebugDotFlagsKernelNodeParams = 0x4,
  cudaGraphDebugDotFlagsMemcpyNodeParams = 0x8,
  cudaGraphDebugDotFlagsMemsetNodeParams = 0x10,
  cudaGraphDebugDotFlagsHostNodeParams = 0x20,
  cudaGraphDebugDotFlagsEventNodeParams = 0x40,
  cudaGraphDebugDotFlagsExtSemasNodeParams = 0x80,
  cudaGraphDebugDotFlagsExtSemasSignalNodeParams = 0x100,
  cudaGraphDebugDotFlagsExtSemasWaitNodeParams = 0x200,
  cudaGraphDebugDotFlagsMemoryNodeParams = 0x400,
  cudaGraphDebugDotFlagsMemAllocNodeParams = 0x800
};

#define cudaHostAllocDefault 0x00
#define cudaHostAllocPortable 0x01
#define cudaHostAllocMapped 0x02
#define cudaHostAllocWriteCombined 0x04
#define cudaHostRegisterDefault 0x00
#define cudaHostRegisterPortable 0x01
#define cudaHostRegisterMapped 0x02
#define cudaHostRegisterIoMemory 0x04
#define cudaHostRegisterReadOnly 0x08
#define cudaMemAttachGlobal 0x01
#define cudaMemAttachHost 0x02
#define cudaMemAttachSingle 0x04
#define cudaStreamDefault 0x00
#define cudaStreamNonBlocking 0x01
#define cudaEventDefault 0x00
#define cudaEventBlockingSync 0x01
#define cudaEventDisableTiming 0x02
#define cudaEventInterprocess 0x04
#define cudaEventWaitDefault 0x00
#define cudaEventWaitExternal 0x01
#define cudaEventRecordDefault 0x00
#define cudaEventRecordExternal 0x01
#define cudaDeviceScheduleAuto 0x00
#define cudaDeviceScheduleSpin 0x01
#define cudaDeviceScheduleYield 0x02
#define cudaDeviceScheduleBlockingSync 0x04
#define cudaDeviceMapHost 0x08
#define cudaDeviceLmemResizeToMax 0x10
#define cudaDeviceSyncMemops 0x80
#define cudaArrayDefault 0x00
#define cudaOccupancyDefault 0x00
#define cudaOccupancyDisableCachingOverride 0x01
#define cudaCpuDeviceId ((int)-1)
#define cudaInvalidDeviceId ((int)-2)
#define cudaStreamLegacy ((cudaStream_t)0x1)
#define cudaStreamPerThread ((cudaStream_t)0x2)
#define cudaStreamGraphTailLaunch ((cudaStream_t)0x3)
#define cudaStreamGraphFireAndForget ((cudaStream_t)0x4)

typedef struct CUstream_st* cudaStream_t;
typedef struct CUevent_st* cudaEvent_t;
typedef struct CUgraph_st* cudaGraph_t;
typedef struct CUgraphNode_st* cudaGraphNode_t;
typedef struct CUgraphExec_st* cudaGraphExec_t;
typedef struct CUfunc_st* cudaFunction_t;
typedef struct CUkern_st* cudaKernel_t;
typedef struct CUmemPoolHandle_st* cudaMemPool_t;
typedef struct cudaArray* cudaArray_t;
typedef const struct cudaArray* cudaArray_const_t;
typedef struct CUexternalMemory_st* cudaExternalMemory_t;
typedef struct CUexternalSemaphore_st* cudaExternalSemaphore_t;
typedef struct CUuserObject_st* cudaUserObject_t;
typedef unsigned long long cudaSurfaceObject_t;
typedef unsigned long long cudaTextureObject_t;
typedef void (*cudaStreamCallback_t)(cudaStream_t stream, cudaError_t status, void* userData);
typedef void (*cudaHostFn_t)(void* userData);

struct cudaUUID_t { char bytes[16]; };
typedef struct cudaUUID_t cudaUUID_t;

struct cudaDeviceProp {
  char name[256];
  cudaUUID_t uuid;
  char luid[8];
  unsigned int luidDeviceNodeMask;
  size_t totalGlobalMem;
  size_t sharedMemPerBlock;
  int regsPerBlock;
  int warpSize;
  size_t memPitch;
  int maxThreadsPerBlock;
  int maxThreadsDim[3];
  int maxGridSize[3];
  int clockRate;
  size_t totalConstMem;
  int major;
  int minor;
  size_t textureAlignment;
  size_t texturePitchAlignment;
  int deviceOverlap;
  int multiProcessorCount;
  int kernelExecTimeoutEnabled;
  int integrated;
  int canMapHostMemory;
  int computeMode;
  int maxTexture1D;
  int maxTexture1DMipmap;
  int maxTexture1DLinear;
  int maxTexture2D[2];
  int maxTexture2DMipmap[2];
  int maxTexture2DLinear[3];
  int maxTexture2DGather[2];
  int maxTexture3D[3];
  int maxTexture3DAlt[3];
  int maxTextureCubemap;
  int maxTexture1DLayered[2];
  int maxTexture2DLayered[3];
  int maxTextureCubemapLayered[2];
  int maxSurface1D;
  int maxSurface2D[2];
  int maxSurface3D[3];
  int maxSurface1DLayered[2];
  int maxSurface2DLayered[3];
  int maxSurfaceCubemap;
  int maxSurfaceCubemapLayered[2];
  size_t surfaceAlignment;
  int concurrentKernels;
  int ECCEnabled;
  int pciBusID;
  int pciDeviceID;
  int pciDomainID;
  int tccDriver;
  int asyncEngineCount;
  int unifiedAddressing;
  int memoryClockRate;
  int memoryBusWidth;
  int l2CacheSize;
  int persistingL2CacheMaxSize;
  int maxThreadsPerMultiProcessor;
  int streamPrioritiesSupported;
  int globalL1CacheSupported;
  int localL1CacheSupported;
  size_t sharedMemPerMultiprocessor;
  int regsPerMultiprocessor;
  int managedMemory;
  int isMultiGpuBoard;
  int multiGpuBoardGroupID;
  int hostNativeAtomicSupported;
  int singleToDoublePrecisionPerfRatio;
  int pageableMemoryAccess;
  int concurrentManagedAccess;
  int computePreemptionSupported;
  int canUseHostPointerForRegisteredMem;
  int cooperativeLaunch;
  int cooperativeMultiDeviceLaunch;
  size_t sharedMemPerBlockOptin;
  int pageableMemoryAccessUsesHostPageTables;
  int directManagedMemAccessFromHost;
  int maxBlocksPerMultiProcessor;
  int accessPolicyMaxWindowSize;
  size_t reservedSharedMemPerBlock;
  int hostRegisterSupported;
  int sparseCudaArraySupported;
  int hostRegisterReadOnlySupported;
  int timelineSemaphoreInteropSupported;
  int memoryPoolsSupported;
  int gpuDirectRDMASupported;
  unsigned int gpuDirectRDMAFlushWritesOptions;
  int gpuDirectRDMAWritesOrdering;
  unsigned int memoryPoolSupportedHandleTypes;
  int deferredMappingCudaArraySupported;
  int ipcEventSupported;
  int clusterLaunch;
  int unifiedFunctionPointers;
  int reserved[63];
};

struct cudaFuncAttributes {
  size_t sharedSizeBytes;
  size_t constSizeBytes;
  size_t localSizeBytes;
  int maxThreadsPerBlock;
  int numRegs;
  int ptxVersion;
  int binaryVersion;
  int cacheModeCA;
  int maxDynamicSharedSizeBytes;
  int preferredShmemCarveout;
  int clusterDimMustBeSet;
  int requiredClusterWidth;
  int requiredClusterHeight;
  int requiredClusterDepth;
  int clusterSchedulingPolicyPreference;
  int nonPortableClusterSizeAllowed;
  int reserved[16];
};

struct cudaPointerAttributes {
  enum cudaMemoryType type;
  int device;
  void* devicePointer;
  void* hostPointer;
};

struct cudaExtent { size_t width, height, depth; };
struct cudaPos { size_t x, y, z; };
struct cudaPitchedPtr { void* ptr; size_t pitch, xsize, ysize; };
struct cudaMemcpy3DParms {
  cudaArray_t srcArray; struct cudaPos srcPos; struct cudaPitchedPtr srcPtr;
  cudaArray_t dstArray; struct cudaPos dstPos; struct cudaPitchedPtr dstPtr;
  struct cudaExtent extent; enum cudaMemcpyKind kind;
};

struct cudaMemLocation { int type; int id; };
struct cudaMemAccessDesc { struct cudaMemLocation location; int flags; };
struct cudaMemPoolProps { int allocType; int handleTypes; struct cudaMemLocation location; void* win32SecurityAttributes; size_t maxSize; unsigned short usage; unsigned char reserved[54]; };
enum cudaMemPoolAttr { cudaMemPoolReuseFollowEventDependencies = 1, cudaMemPoolReuseAllowOpportunistic = 2, cudaMemPoolReuseAllowInternalDependencies = 3, cudaMemPoolAttrReleaseThreshold = 4, cudaMemPoolAttrReservedMemCurrent = 5, cudaMemPoolAttrReservedMemHigh = 6, cudaMemPoolAttrUsedMemCurrent = 7, cudaMemPoolAttrUsedMemHigh = 8 };

struct cudaKernelNodeParams { void* func; struct dim3 gridDim; struct dim3 blockDim; unsigned int sharedMemBytes; void** kernelParams; void** extra; };
struct cudaMemsetParams { void* dst; size_t pitch; unsigned int value; unsigned int elementSize; size_t width; size_t height; };
struct cudaHostNodeParams { cudaHostFn_t fn; void* userData; };

enum cudaLaunchAttributeID { cudaLaunchAttributeIgnore = 0, cudaLaunchAttributeAccessPolicyWindow = 1, cudaLaunchAttributeCooperative = 2, cudaLaunchAttributeSynchronizationPolicy = 3, cudaLaunchAttributeClusterDimension = 4, cudaLaunchAttributeClusterSchedulingPolicyPreference = 5, cudaLaunchAttributeProgrammaticStreamSerialization = 6, cudaLaunchAttributeProgrammaticEvent = 7, cudaLaunchAttributePriority = 8, cudaLaunchAttributeMemSyncDomainMap = 9, cudaLaunchAttributeMemSyncDomain = 10 };
typedef union cudaLaunchAttributeValue { char pad[64]; int cooperative; int priority; struct { unsigned int x, y, z; } clusterDim; } cudaLaunchAttributeValue;
typedef struct cudaLaunchAttribute_st { enum cudaLaunchAttributeID id; char pad[4]; cudaLaunchAttributeValue val; } cudaLaunchAttribute;
typedef struct cudaLaunchConfig_st { struct dim3 gridDim; struct dim3 blockDim; size_t dynamicSmemBytes; cudaStream_t stream; cudaLaunchAttribute* attrs; unsigned int numAttrs; } cudaLaunchConfig_t;

// Texture/surface types exist only so that code declaring them compiles; no texture support.
struct cudaChannelFormatDesc { int x, y, z, w; int f; };
struct cudaResourceDesc { int resType; unsigned char pad[120]; };
struct cudaTextureDesc { unsigned char pad[128]; };

#endif // MVCC_DRIVER_TYPES_H
