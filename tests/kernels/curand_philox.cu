// Philox4_32_10: known answers, subsequence/offset layout, host API matches the device.
#include <cuda_runtime.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cstdint>
#include <cmath>
#include <cstdio>
#include <vector>

static int fails = 0;
#define CHECK(cond) do { if (!(cond)) { std::printf("FAIL %s\n", #cond); fails++; } } while (0)

__global__ void fill_u32(unsigned long long seed, unsigned long long sub, unsigned long long off, unsigned int* o, int n) {
    curandStatePhilox4_32_10_t st;
    curand_init(seed, sub, off, &st);
    for (int i = 0; i < n; i++) o[i] = curand(&st);
}

__global__ void fill_threads(unsigned long long seed, unsigned int* o, int n) {
    int id = (int)(threadIdx.x + blockIdx.x * blockDim.x);
    if (id >= n) return;
    curandStatePhilox4_32_10_t st;
    curand_init(seed, (unsigned long long)id, 0, &st);
    o[id] = curand(&st);
}

__global__ void fill_uniform(unsigned long long seed, float* o, int n) {
    curandStatePhilox4_32_10_t st;
    curand_init(seed, 0, 0, &st);
    for (int i = 0; i < n; i++) o[i] = curand_uniform(&st);
}

__global__ void fill_normal(unsigned long long seed, float* o, int n) {
    curandStatePhilox4_32_10_t st;
    curand_init(seed, 0, 0, &st);
    for (int i = 0; i < n; i++) o[i] = curand_normal(&st);
}

static void host_fill(unsigned long long seed, unsigned long long sub, unsigned long long off, unsigned int* o, int n) {
    curandStatePhilox4_32_10_t st;
    curand_init(seed, sub, off, &st);
    for (int i = 0; i < n; i++) o[i] = curand(&st);
}

int main() {
    curandStatePhilox4_32_10_t st;
    curand_init(0, 0, 0, &st);
    uint4 kat = curand4(&st);
    CHECK(kat.x == 0x6627e8d5u && kat.y == 0xe169c58du && kat.z == 0xbc57ac4cu && kat.w == 0x9b00dbd8u);

    // The other Random123 vectors are the raw block function, reached by planting counter and key.
    st = curandStatePhilox4_32_10_t{};
    st.ctr = make_uint4(0xffffffffu, 0xffffffffu, 0xffffffffu, 0xffffffffu);
    st.key = make_uint2(0xffffffffu, 0xffffffffu);
    kat = curand4(&st);
    CHECK(kat.x == 0x408f276du && kat.y == 0x41c83b0eu && kat.z == 0xa20bc7c6u && kat.w == 0x6d5451fdu);
    st = curandStatePhilox4_32_10_t{};
    st.ctr = make_uint4(0x243f6a88u, 0x85a308d3u, 0x13198a2eu, 0x03707344u);
    st.key = make_uint2(0xa4093822u, 0x299f31d0u);
    kat = curand4(&st);
    CHECK(kat.x == 0xd16cfe09u && kat.y == 0x94fdccebu && kat.z == 0x5001e420u && kat.w == 0x24126ea1u);

    const unsigned long long seed = 0x123456789abcdef0ull;
    unsigned int seq[16];
    host_fill(seed, 0, 0, seq, 16);
    unsigned int one[4];
    host_fill(seed, 0, 5, one, 4);
    CHECK(one[0] == seq[5] && one[1] == seq[6] && one[2] == seq[7] && one[3] == seq[8]);

    curandStatePhilox4_32_10_t skipped;
    curand_init(seed, 0, 0, &skipped);
    skipahead_sequence(3, &skipped);
    skipahead(2, &skipped);
    curandStatePhilox4_32_10_t direct;
    curand_init(seed, 3, 2, &direct);
    CHECK(curand(&skipped) == curand(&direct));

    // subsequence 1 is block counter high-half = 1, not a shift of the low half.
    curandStatePhilox4_32_10_t planted{};
    planted.key = make_uint2((unsigned int)seed, (unsigned int)(seed >> 32));
    planted.ctr = make_uint4(0, 0, 1, 0);
    curand_init(seed, 1, 0, &direct);
    CHECK(curand(&planted) == curand(&direct));

    curand_init(seed, 0, 0, &st);
    uint4 block = curand4(&st);
    CHECK(block.x == seq[0] && block.y == seq[1] && block.z == seq[2] && block.w == seq[3]);

    float u = 0;
    curand_init(seed, 0, 0, &st);
    int uniform_bad = 0;
    for (int i = 0; i < 1024; i++) {
        u = curand_uniform(&st);
        if (!(u > 0.f && u <= 1.f)) uniform_bad++;
    }
    CHECK(uniform_bad == 0);

    const int n = 64;
    std::vector<unsigned int> h(n), d(n);
    host_fill(seed, 7, 9, h.data(), n);
    unsigned int* dev = nullptr;
    if (cudaMalloc(&dev, n * sizeof(unsigned int)) != cudaSuccess) { std::printf("FAIL cudaMalloc\n"); return 1; }
    fill_u32<<<1, 1>>>(seed, 7, 9, dev, n);
    if (cudaDeviceSynchronize() != cudaSuccess) { std::printf("FAIL sync fill\n"); return 1; }
    cudaMemcpy(d.data(), dev, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; i++) CHECK(d[i] == h[i]);

    const int threads = 128;
    std::vector<unsigned int> ht(threads), dt(threads);
    for (int i = 0; i < threads; i++) host_fill(seed, (unsigned long long)i, 0, &ht[i], 1);
    unsigned int* devt = nullptr;
    cudaMalloc(&devt, threads * sizeof(unsigned int));
    fill_threads<<<threads / 32, 32>>>(seed, devt, threads);
    cudaDeviceSynchronize();
    cudaMemcpy(dt.data(), devt, threads * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < threads; i++) CHECK(dt[i] == ht[i]);
    int distinct = 0;
    for (int i = 1; i < threads; i++) if (dt[i] != dt[0]) distinct++;
    CHECK(distinct > threads / 2);

    std::vector<float> hu(32), du(32);
    curand_init(seed, 0, 0, &st);
    for (int i = 0; i < 32; i++) hu[i] = curand_uniform(&st);
    float* devu = nullptr;
    cudaMalloc(&devu, 32 * sizeof(float));
    fill_uniform<<<1, 1>>>(seed, devu, 32);
    cudaDeviceSynchronize();
    cudaMemcpy(du.data(), devu, 32 * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 32; i++) CHECK(du[i] == hu[i]);

    const int nn = 4096;
    std::vector<float> dn(nn);
    float* devn = nullptr;
    cudaMalloc(&devn, nn * sizeof(float));
    fill_normal<<<1, 1>>>(seed, devn, nn);
    cudaDeviceSynchronize();
    cudaMemcpy(dn.data(), devn, nn * sizeof(float), cudaMemcpyDeviceToHost);
    double sum = 0, sum2 = 0;
    int nonfinite = 0;
    for (int i = 0; i < nn; i++) {
        if (!std::isfinite(dn[i])) nonfinite++;
        sum += dn[i];
        sum2 += (double)dn[i] * dn[i];
    }
    double mean = sum / nn;
    double var = sum2 / nn - mean * mean;
    CHECK(nonfinite == 0);
    CHECK(mean > -0.15 && mean < 0.15);
    CHECK(var > 0.7 && var < 1.3);

    CHECK(curandDestroyGenerator((curandGenerator_t)1) == CURAND_STATUS_NOT_INITIALIZED);
    CHECK(curandSetPseudoRandomGeneratorSeed((curandGenerator_t)1, 0) == CURAND_STATUS_NOT_INITIALIZED);

    curandGenerator_t gen = nullptr;
    CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_XORWOW) == CURAND_STATUS_TYPE_ERROR);
    CHECK(curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_PHILOX4_32_10) == CURAND_STATUS_SUCCESS);
    CHECK(reinterpret_cast<uintptr_t>(gen) > 0xffffffffu);
    CHECK(curandSetPseudoRandomGeneratorSeed(gen, seed) == CURAND_STATUS_SUCCESS);
    CHECK(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_LEGACY) == CURAND_STATUS_TYPE_ERROR);
    CHECK(curandSetGeneratorOrdering(gen, CURAND_ORDERING_PSEUDO_DEFAULT) == CURAND_STATUS_SUCCESS);
    unsigned int* devg = nullptr;
    cudaMalloc(&devg, 16 * sizeof(unsigned int));
    CHECK(curandGenerate(gen, devg, 16) == CURAND_STATUS_SUCCESS);
    std::vector<unsigned int> hg(16);
    cudaMemcpy(hg.data(), devg, 16 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    for (int i = 0; i < 16; i++) CHECK(hg[i] == seq[i]);
    CHECK(curandSetGeneratorOffset(gen, 5) == CURAND_STATUS_SUCCESS);
    CHECK(curandGenerate(gen, devg, 1) == CURAND_STATUS_SUCCESS);
    unsigned int hopped = 0;
    cudaMemcpy(&hopped, devg, sizeof(hopped), cudaMemcpyDeviceToHost);
    CHECK(hopped == seq[5]);

    curandGenerator_t host = nullptr;
    CHECK(curandCreateGeneratorHost(&host, CURAND_RNG_PSEUDO_PHILOX4_32_10) == CURAND_STATUS_SUCCESS);
    CHECK(curandSetPseudoRandomGeneratorSeed(host, 0) == CURAND_STATUS_SUCCESS);
    float uf[4];
    CHECK(curandGenerateUniform(host, uf, 4) == CURAND_STATUS_SUCCESS);
    curand_init(0, 0, 0, &st);
    for (int i = 0; i < 4; i++) CHECK(uf[i] == curand_uniform(&st));
    int ver = 0;
    CHECK(curandGetVersion(&ver) == CURAND_STATUS_SUCCESS && ver == CURAND_VERSION);
    CHECK(curandDestroyGenerator(gen) == CURAND_STATUS_SUCCESS);
    CHECK(curandDestroyGenerator(host) == CURAND_STATUS_SUCCESS);
    CHECK(curandDestroyGenerator(gen) == CURAND_STATUS_NOT_INITIALIZED);

    cudaFree(dev); cudaFree(devt); cudaFree(devu); cudaFree(devn); cudaFree(devg);
    if (fails) std::printf("FAIL (%d)\n", fails);
    else std::printf("PASS\n");
    return fails ? 1 : 0;
}
