// Host Philox fill used by the cuRAND host API. The device header is the single implementation.
#include "curand_kernel.h"
#include <stdint.h>

extern "C" void mvcc_philox_fill(uint64_t seed, uint64_t offset, uint32_t* out, uint64_t n) {
    curandStatePhilox4_32_10_t st;
    curand_init(seed, 0, offset, &st);
    for (uint64_t i = 0; i < n; i++) out[i] = curand(&st);
}

extern "C" void mvcc_philox_fill_uniform(uint64_t seed, uint64_t offset, float* out, uint64_t n) {
    curandStatePhilox4_32_10_t st;
    curand_init(seed, 0, offset, &st);
    for (uint64_t i = 0; i < n; i++) out[i] = curand_uniform(&st);
}
