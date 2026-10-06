// mvcc: cuRAND host API. Only the Philox4_32_10 generator is implemented.
//
// curandGenerate / curandGenerateUniform write one subsequence, the same sequence as
// curand_init(seed, 0, offset) on the device. That is CURAND_ORDERING_PSEUDO_BEST / DEFAULT.
// CURAND_ORDERING_PSEUDO_LEGACY (the 65536-wide interleave) is not implemented.
// Other generator types return CURAND_STATUS_TYPE_ERROR.
//
// libcurand.dylib is libcudart.dylib. Generation runs on the CPU and is copied to the
// device pointer; if a stream was set, that stream is synchronized first.
#ifndef CURAND_H_
#define CURAND_H_

#include <cuda_runtime_api.h>

#define CURAND_VERSION 10300

#ifdef __cplusplus
extern "C" {
#endif

typedef enum curandStatus {
    CURAND_STATUS_SUCCESS = 0,
    CURAND_STATUS_VERSION_MISMATCH = 100,
    CURAND_STATUS_NOT_INITIALIZED = 101,
    CURAND_STATUS_ALLOCATION_FAILED = 102,
    CURAND_STATUS_TYPE_ERROR = 103,
    CURAND_STATUS_OUT_OF_RANGE = 104,
    CURAND_STATUS_LENGTH_NOT_MULTIPLE = 105,
    CURAND_STATUS_DOUBLE_PRECISION_REQUIRED = 106,
    CURAND_STATUS_LAUNCH_FAILURE = 201,
    CURAND_STATUS_PREEXISTING_FAILURE = 202,
    CURAND_STATUS_INITIALIZATION_FAILED = 203,
    CURAND_STATUS_ARCH_MISMATCH = 204,
    CURAND_STATUS_INTERNAL_ERROR = 999
} curandStatus_t;

typedef enum curandRngType {
    CURAND_RNG_TEST = 0,
    CURAND_RNG_PSEUDO_DEFAULT = 100,
    CURAND_RNG_PSEUDO_XORWOW = 101,
    CURAND_RNG_PSEUDO_MRG32K3A = 121,
    CURAND_RNG_PSEUDO_MTGP32 = 141,
    CURAND_RNG_PSEUDO_MT19937 = 142,
    CURAND_RNG_PSEUDO_PHILOX4_32_10 = 161,
    CURAND_RNG_QUASI_DEFAULT = 200,
    CURAND_RNG_QUASI_SOBOL32 = 201,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL32 = 202,
    CURAND_RNG_QUASI_SOBOL64 = 203,
    CURAND_RNG_QUASI_SCRAMBLED_SOBOL64 = 204
} curandRngType_t;

typedef enum curandOrdering {
    CURAND_ORDERING_PSEUDO_BEST = 100,
    CURAND_ORDERING_PSEUDO_DEFAULT = 101,
    CURAND_ORDERING_PSEUDO_SEEDED = 102,
    CURAND_ORDERING_PSEUDO_LEGACY = 103,
    CURAND_ORDERING_PSEUDO_DYNAMIC = 104,
    CURAND_ORDERING_QUASI_DEFAULT = 201
} curandOrdering_t;

typedef struct curandGenerator_st* curandGenerator_t;

curandStatus_t curandCreateGenerator(curandGenerator_t* generator, curandRngType_t rng_type);
curandStatus_t curandCreateGeneratorHost(curandGenerator_t* generator, curandRngType_t rng_type);
curandStatus_t curandDestroyGenerator(curandGenerator_t generator);
curandStatus_t curandSetPseudoRandomGeneratorSeed(curandGenerator_t generator, unsigned long long seed);
curandStatus_t curandSetGeneratorOffset(curandGenerator_t generator, unsigned long long offset);
curandStatus_t curandSetStream(curandGenerator_t generator, cudaStream_t stream);
curandStatus_t curandSetGeneratorOrdering(curandGenerator_t generator, curandOrdering_t order);
curandStatus_t curandGenerate(curandGenerator_t generator, unsigned int* outputPtr, size_t num);
curandStatus_t curandGenerateUniform(curandGenerator_t generator, float* outputPtr, size_t num);
curandStatus_t curandGetVersion(int* version);

#ifdef __cplusplus
}
#endif

#endif // CURAND_H_
