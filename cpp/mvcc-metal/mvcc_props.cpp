// Fills cudaDeviceProp from runtime-provided facts. Lives in C++ so the struct layout comes straight from our header.
#include <cstring>
#include "../../include/driver_types.h"

extern "C" {
struct MvccDeviceFacts {
  const char* name;
  uint64_t total_mem;
  uint32_t core_count;
  uint32_t max_tg_mem;
  uint32_t max_threads_per_core;
  uint32_t major, minor;
  uint64_t registry_id;
};

void mvcc_fill_device_prop(struct cudaDeviceProp* p, const struct MvccDeviceFacts* f) {
  memset(p, 0, sizeof(*p));
  strncpy(p->name, f->name, sizeof(p->name) - 1);
  memcpy(p->uuid.bytes, &f->registry_id, 8);
  p->totalGlobalMem = f->total_mem;
  /* CUDA's limits, not the device's 32 KB: a block that needs more runs from the spill build (shared memory in a
     device-memory pool; mvcc-cudart runtime.rs), exactly as an NVIDIA launch would after cudaFuncSetAttribute. */
  p->sharedMemPerBlock = 49152;
  p->sharedMemPerBlockOptin = 232448;
  p->sharedMemPerMultiprocessor = 233472;
  p->reservedSharedMemPerBlock = 0;
  p->regsPerBlock = 65536;
  p->regsPerMultiprocessor = 65536;
  p->warpSize = 32;
  p->memPitch = 2147483647;
  p->maxThreadsPerBlock = 1024;
  p->maxThreadsDim[0] = 1024; p->maxThreadsDim[1] = 1024; p->maxThreadsDim[2] = 1024;
  p->maxGridSize[0] = 2147483647; p->maxGridSize[1] = 65535; p->maxGridSize[2] = 65535;
  p->clockRate = 1400000;  // kHz, nominal
  p->totalConstMem = 65536;
  p->major = (int)f->major; p->minor = (int)f->minor;
  p->textureAlignment = 512; p->texturePitchAlignment = 32;
  p->deviceOverlap = 1;
  p->multiProcessorCount = (int)f->core_count;
  p->kernelExecTimeoutEnabled = 0;
  p->integrated = 1;
  p->canMapHostMemory = 1;
  p->computeMode = 0;
  p->concurrentKernels = 1;
  p->ECCEnabled = 0;
  p->asyncEngineCount = 1;
  p->unifiedAddressing = 1;
  p->memoryClockRate = 0; p->memoryBusWidth = 0;
  p->l2CacheSize = 24 << 20;
  p->persistingL2CacheMaxSize = 0;
  p->maxThreadsPerMultiProcessor = (int)f->max_threads_per_core;
  p->streamPrioritiesSupported = 0;
  p->globalL1CacheSupported = 1; p->localL1CacheSupported = 1;
  p->managedMemory = 1;
  p->isMultiGpuBoard = 0;
  p->hostNativeAtomicSupported = 1;
  p->singleToDoublePrecisionPerfRatio = 0;
  p->pageableMemoryAccess = 0;
  p->concurrentManagedAccess = 1;
  p->computePreemptionSupported = 1;
  p->canUseHostPointerForRegisteredMem = 0;
  p->cooperativeLaunch = 0;
  p->maxBlocksPerMultiProcessor = 32;
  p->accessPolicyMaxWindowSize = 0;
  p->hostRegisterSupported = 1;
  p->memoryPoolsSupported = 0;
}
}
