// mvcc: cuRAND device API for Philox4_32_10.
//
// Philox4x32-10 is the counter-based generator from Salmon, Moraes, Dror and Shaw,
// "Parallel Random Numbers: As Easy as 1, 2, 3" (SC11). The counter layout follows the
// public cuRAND device API: each seed selects a sequence, subsequence i skips 2^66
// outputs (2^64 blocks of four), and offset counts 32-bit outputs inside that subsequence.
//
// XORWOW, MRG32k3a, MTGP32 and the quasirandom generators are not implemented.
#ifndef CURAND_KERNEL_H_
#define CURAND_KERNEL_H_

#include <cuda_runtime.h>

struct curandStatePhilox4_32_10 {
    uint4 ctr;                 // next Philox block when STATE == 0; one past the current block otherwise
    uint4 output;              // four outputs of the current block, valid when STATE != 0
    uint2 key;                 // seed, low word then high word
    unsigned int STATE;        // index of the next output word in the current block (0..3)
    int boxmuller_flag;        // 1 when boxmuller_extra holds the second normal of a pair
    float boxmuller_extra;
};
typedef struct curandStatePhilox4_32_10 curandStatePhilox4_32_10_t;

// Paper constants for Philox4x32 (10 rounds).
static const unsigned int MVCC_PHILOX_M0 = 0xD2511F53u;
static const unsigned int MVCC_PHILOX_M1 = 0xCD9E8D57u;
static const unsigned int MVCC_PHILOX_W0 = 0x9E3779B9u;
static const unsigned int MVCC_PHILOX_W1 = 0xBB67AE85u;

static __inline__ __host__ __device__ uint4 mvcc_philox4x32_10(uint4 ctr, uint2 key) {
    for (int round = 0; round < 10; round++) {
        unsigned long long p0 = (unsigned long long)MVCC_PHILOX_M0 * ctr.x;
        unsigned long long p1 = (unsigned long long)MVCC_PHILOX_M1 * ctr.z;
        uint4 next;
        next.x = (unsigned int)(p1 >> 32) ^ ctr.y ^ key.x;
        next.y = (unsigned int)p1;
        next.z = (unsigned int)(p0 >> 32) ^ ctr.w ^ key.y;
        next.w = (unsigned int)p0;
        ctr = next;
        key.x += MVCC_PHILOX_W0;
        key.y += MVCC_PHILOX_W1;
    }
    return ctr;
}

static __inline__ __host__ __device__ void mvcc_philox_bump(uint4* ctr) {
    if (++ctr->x) return;
    if (++ctr->y) return;
    if (++ctr->z) return;
    ++ctr->w;
}

// Next sample as (block, word). STATE != 0 means ctr already points one past that block.
static __inline__ __host__ __device__ void mvcc_philox_load(const curandStatePhilox4_32_10_t* s,
                                                            unsigned long long* blo, unsigned long long* bhi, unsigned int* word) {
    unsigned int w = s->STATE & 3u;
    unsigned long long lo = (unsigned long long)s->ctr.x | ((unsigned long long)s->ctr.y << 32);
    unsigned long long hi = (unsigned long long)s->ctr.z | ((unsigned long long)s->ctr.w << 32);
    if (w != 0) {
        unsigned long long nlo = lo - 1;
        hi -= (lo == 0ull);
        lo = nlo;
    }
    *blo = lo;
    *bhi = hi;
    *word = w;
}

static __inline__ __host__ __device__ void mvcc_philox_store(curandStatePhilox4_32_10_t* s,
                                                             unsigned long long blo, unsigned long long bhi, unsigned int word) {
    s->boxmuller_flag = 0;
    if (word == 0) {
        s->ctr = make_uint4((unsigned int)blo, (unsigned int)(blo >> 32), (unsigned int)bhi, (unsigned int)(bhi >> 32));
        s->STATE = 0;
        return;
    }
    uint4 block = make_uint4((unsigned int)blo, (unsigned int)(blo >> 32), (unsigned int)bhi, (unsigned int)(bhi >> 32));
    s->output = mvcc_philox4x32_10(block, s->key);
    unsigned long long nlo = blo + 1;
    if (nlo == 0ull) bhi++;
    s->ctr = make_uint4((unsigned int)nlo, (unsigned int)(nlo >> 32), (unsigned int)bhi, (unsigned int)(bhi >> 32));
    s->STATE = word;
}

static __inline__ __host__ __device__ void skipahead(unsigned long long n, curandStatePhilox4_32_10_t* state) {
    if (n == 0ull || !state) return;
    unsigned long long blo, bhi;
    unsigned int word;
    mvcc_philox_load(state, &blo, &bhi, &word);
    unsigned long long blocks = n >> 2;
    unsigned int sum = word + (unsigned int)(n & 3ull);
    blocks += (unsigned long long)(sum >> 2);
    unsigned long long nlo = blo + blocks;
    bhi += (nlo < blo);
    mvcc_philox_store(state, nlo, bhi, sum & 3u);
}

// One subsequence is 2^64 blocks (2^66 outputs): add n to the high half of the block counter.
static __inline__ __host__ __device__ void skipahead_sequence(unsigned long long n, curandStatePhilox4_32_10_t* state) {
    if (n == 0ull || !state) return;
    unsigned long long blo, bhi;
    unsigned int word;
    mvcc_philox_load(state, &blo, &bhi, &word);
    bhi += n;
    mvcc_philox_store(state, blo, bhi, word);
}

static __inline__ __host__ __device__ void curand_init(unsigned long long seed, unsigned long long subsequence,
                                                       unsigned long long offset, curandStatePhilox4_32_10_t* state) {
    if (!state) return;
    state->key = make_uint2((unsigned int)seed, (unsigned int)(seed >> 32));
    state->ctr = make_uint4(0, 0, 0, 0);
    state->output = make_uint4(0, 0, 0, 0);
    state->STATE = 0;
    state->boxmuller_flag = 0;
    state->boxmuller_extra = 0.f;
    skipahead_sequence(subsequence, state);
    skipahead(offset, state);
}

static __inline__ __host__ __device__ unsigned int curand(curandStatePhilox4_32_10_t* state) {
    unsigned int word = state->STATE & 3u;
    if (word == 0) {
        state->output = mvcc_philox4x32_10(state->ctr, state->key);
        mvcc_philox_bump(&state->ctr);
    }
    unsigned int r;
    if (word == 0) r = state->output.x;
    else if (word == 1) r = state->output.y;
    else if (word == 2) r = state->output.z;
    else r = state->output.w;
    state->STATE = (word + 1u) & 3u;
    return r;
}

static __inline__ __host__ __device__ uint4 curand4(curandStatePhilox4_32_10_t* state) {
    if ((state->STATE & 3u) == 0) {
        uint4 r = mvcc_philox4x32_10(state->ctr, state->key);
        mvcc_philox_bump(&state->ctr);
        return r;
    }
    uint4 r;
    r.x = curand(state);
    r.y = curand(state);
    r.z = curand(state);
    r.w = curand(state);
    return r;
}

// (0, 1]. The added half ulp keeps the result off zero and off the denormal range; some inputs round to 1.
static __inline__ __host__ __device__ float mvcc_curand_uniform_bits(unsigned int x) {
    return ((float)x + 0.5f) * 2.3283064365386963e-10f;
}

static __inline__ __host__ __device__ float curand_uniform(curandStatePhilox4_32_10_t* state) {
    return mvcc_curand_uniform_bits(curand(state));
}

static __inline__ __host__ __device__ float4 curand_uniform4(curandStatePhilox4_32_10_t* state) {
    uint4 u = curand4(state);
    float4 r;
    r.x = mvcc_curand_uniform_bits(u.x);
    r.y = mvcc_curand_uniform_bits(u.y);
    r.z = mvcc_curand_uniform_bits(u.z);
    r.w = mvcc_curand_uniform_bits(u.w);
    return r;
}

// Two outputs joined into a 53-bit fraction, still in (0, 1].
static __inline__ __host__ __device__ double curand_uniform_double(curandStatePhilox4_32_10_t* state) {
    unsigned long long hi = curand(state);
    unsigned long long lo = curand(state);
    unsigned long long mant = (hi << 21) | (lo >> 11);
    return ((double)mant + 0.5) * 1.1102230246251565e-16; // 2^-53
}

static __inline__ __host__ __device__ float curand_normal(curandStatePhilox4_32_10_t* state) {
    if (state->boxmuller_flag) {
        state->boxmuller_flag = 0;
        return state->boxmuller_extra;
    }
    float u = curand_uniform(state);
    float v = curand_uniform(state);
    float radius = sqrtf(-2.0f * logf(u));
    float theta = 6.283185307179586f * v;
    state->boxmuller_extra = radius * sinf(theta);
    state->boxmuller_flag = 1;
    return radius * cosf(theta);
}

static __inline__ __host__ __device__ float4 curand_normal4(curandStatePhilox4_32_10_t* state) {
    float4 r;
    r.x = curand_normal(state);
    r.y = curand_normal(state);
    r.z = curand_normal(state);
    r.w = curand_normal(state);
    return r;
}

static __inline__ __host__ __device__ float curand_log_normal(curandStatePhilox4_32_10_t* state, float mean, float stddev) {
    return expf(mean + stddev * curand_normal(state));
}

#endif // CURAND_KERNEL_H_
