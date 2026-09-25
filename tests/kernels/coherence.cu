// Coherent device accesses: what a CUDA program that synchronizes through device memory relies on.
//
// On NVIDIA a volatile / ld.volatile / .relaxed.gpu / .acquire.gpu / __ldcg load is performed at L2, so a thread that
// spins on a flag sees the store another SM makes; a volatile / .release / __stcg store is published there. An Apple
// GPU serves a plain device load from a core's L1 for the life of the kernel, so mvcc-ir2msl performs these accesses
// coherently (prelude, "coherent device accesses"). This checks:
//   - within one kernel, block 0 spins on a flag block 1 sets: volatile (u32, u64, u16, float), __ldcg, ld.acquire /
//     st.release inline PTX, ld.volatile inline PTX;
//   - across two streams (two logical devices under MVCC_DEVICES=2), a pair of kernels that wait on each other's
//     flag: the runtime commits every open buffer when the host waits, so both are on the GPU together (the peer
//     all-reduce kernel of a sharded model);
//   - a coherent read of data another block wrote after a fence + flag handshake (a split-K partial).
//
//   nvcc -O2 -std=c++17 -o coherence coherence.cu && ./coherence && MVCC_DEVICES=2 ./coherence
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#define CK(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess) {                                                                         \
      fprintf(stderr, "%s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__);    \
      return 1;                                                                                      \
    }                                                                                                \
  } while (0)

constexpr unsigned kSpinLimit = 1u << 26;
constexpr unsigned kTimeout = 0xdeadu;

__device__ __forceinline__ void delay() {
  unsigned x = 0;
  for (unsigned i = 0; i < (1u << 16); ++i) x = x * 1664525u + 1013904223u;
  if (x == 7u) __threadfence();   // keep the loop
}

// mode 0: volatile u32; 1: volatile u64; 2: volatile u16; 3: volatile float; 4: __ldcg u32; 5: ld.acquire / st.release
// PTX; 6: ld.volatile PTX; 7: data handshake (block 1 writes 1024 floats, fence, flag; block 0 reads them with __ldcg)
__global__ void flag_kernel(unsigned* flag32, unsigned long long* flag64, unsigned short* flag16, float* flagf,
                            float* data, unsigned* out, int mode) {
  if (threadIdx.x != 0 && mode != 7) return;
  if (blockIdx.x == 1) {
    if (mode == 7) {
      for (int i = threadIdx.x; i < 1024; i += blockDim.x) data[i] = static_cast<float>(i) * 0.5f;
      __threadfence();
      __syncthreads();
      if (threadIdx.x == 0) *(volatile unsigned*)flag32 = 1u;
      return;
    }
    delay();
    switch (mode) {
      case 0: case 4: *(volatile unsigned*)flag32 = 1u; break;
      case 1: *(volatile unsigned long long*)flag64 = 0x100000001ull; break;
      case 2: *(volatile unsigned short*)flag16 = 1u; break;
      case 3: *(volatile float*)flagf = 1.f; break;
      case 5: asm volatile("st.release.gpu.global.u32 [%0], %1;" ::"l"(flag32), "r"(1u) : "memory"); break;
      case 6: asm volatile("st.volatile.global.u32 [%0], %1;" ::"l"(flag32), "r"(1u) : "memory"); break;
      default: break;
    }
    return;
  }
  if (mode == 7) {
    if (threadIdx.x == 0) {
      unsigned spins = 0;
      while (*(volatile unsigned*)flag32 == 0u) { if (++spins > kSpinLimit) { out[0] = kTimeout; break; } }
      out[1] = spins;
    }
    __syncthreads();
    if (out[0] == kTimeout) return;
    __threadfence();
    unsigned bad = 0;
    for (int i = threadIdx.x; i < 1024; i += blockDim.x) if (__ldcg(data + i) != static_cast<float>(i) * 0.5f) ++bad;
    if (bad) atomicAdd(out, bad);
    return;
  }
  unsigned spins = 0;
  auto set = [&]() -> bool {
    switch (mode) {
      case 0: return *(volatile unsigned*)flag32 != 0u;
      case 1: return *(volatile unsigned long long*)flag64 == 0x100000001ull;
      case 2: return *(volatile unsigned short*)flag16 != 0u;
      case 3: return *(volatile float*)flagf != 0.f;
      case 4: return __ldcg(flag32) != 0u;
      case 5: { unsigned v; asm volatile("ld.acquire.gpu.global.u32 %0, [%1];" : "=r"(v) : "l"(flag32) : "memory"); return v != 0u; }
      case 6: { unsigned v; asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(v) : "l"(flag32) : "memory"); return v != 0u; }
      default: return true;
    }
  };
  while (!set()) { if (++spins > kSpinLimit) { out[0] = kTimeout; return; } }
  out[0] = 0;
  out[1] = spins;
}

// two streams: each kernel sets its flag and waits for the other's
__global__ void pair_kernel(volatile unsigned* mine, volatile unsigned* theirs, unsigned s, unsigned* out) {
  if (threadIdx.x != 0) return;
  *mine = s;
  __threadfence_system();
  unsigned spins = 0;
  while (*theirs < s) { if (++spins > kSpinLimit) { out[0] = kTimeout; return; } }
  out[0] = spins;
}

int main() {
  int devices = 0;
  CK(cudaGetDeviceCount(&devices));
  static const char* names[] = {"volatile u32", "volatile u64", "volatile u16", "volatile float", "__ldcg u32", "ld.acquire/st.release", "ld.volatile/st.volatile", "data handshake (__ldcg)"};
  unsigned* flag32; unsigned long long* flag64; unsigned short* flag16; float* flagf; float* data; unsigned* out;
  CK(cudaMalloc(&flag32, 4)); CK(cudaMalloc(&flag64, 8)); CK(cudaMalloc(&flag16, 2)); CK(cudaMalloc(&flagf, 4));
  CK(cudaMalloc(&data, 1024 * sizeof(float))); CK(cudaMalloc(&out, 8));
  int failures = 0;
  for (int mode = 0; mode < 8; ++mode) {
    CK(cudaMemset(flag32, 0, 4)); CK(cudaMemset(flag64, 0, 8)); CK(cudaMemset(flag16, 0, 2)); CK(cudaMemset(flagf, 0, 4));
    CK(cudaMemset(data, 0, 1024 * sizeof(float))); CK(cudaMemset(out, 0, 8));
    flag_kernel<<<2, 128>>>(flag32, flag64, flag16, flagf, data, out, mode);
    CK(cudaDeviceSynchronize());
    unsigned h[2];
    CK(cudaMemcpy(h, out, 8, cudaMemcpyDeviceToHost));
    if (h[0] == kTimeout) { printf("%-26s TIMEOUT: the spinning block never saw the store\n", names[mode]); ++failures; }
    else if (h[0] != 0) { printf("%-26s %u stale values read after the flag\n", names[mode], h[0]); ++failures; }
    else printf("%-26s ok (%u spins)\n", names[mode], h[1]);
  }

  // cross-stream pair
  {
    const int world = 2;
    int dev[2] = {0, devices >= 2 ? 1 : 0};
    unsigned* flags[2]; unsigned* pout[2]; cudaStream_t st[2];
    for (int r = 0; r < world; ++r) {
      CK(cudaSetDevice(dev[r]));
      CK(cudaMalloc(&flags[r], 4)); CK(cudaMemset(flags[r], 0, 4)); CK(cudaMalloc(&pout[r], 4)); CK(cudaStreamCreate(&st[r]));
    }
    for (unsigned s = 1; s <= 3; ++s) {
      for (int r = 0; r < world; ++r) { CK(cudaSetDevice(dev[r])); pair_kernel<<<1, 32, 0, st[r]>>>(flags[r], flags[1 - r], s, pout[r]); }
      // the host waits on rank 0 only: rank 1's launch must reach the GPU all the same
      CK(cudaSetDevice(dev[0]));
      CK(cudaStreamSynchronize(st[0]));
      CK(cudaSetDevice(dev[1]));
      CK(cudaStreamSynchronize(st[1]));
      unsigned h[2];
      for (int r = 0; r < world; ++r) CK(cudaMemcpy(&h[r], pout[r], 4, cudaMemcpyDeviceToHost));
      if (h[0] == kTimeout || h[1] == kTimeout) { printf("cross-stream pair round %u: TIMEOUT (%s)\n", s, devices >= 2 ? "two logical devices" : "one device"); ++failures; }
      else printf("cross-stream pair round %u: ok (%u / %u spins, %s)\n", s, h[0], h[1], devices >= 2 ? "two logical devices" : "one device");
    }
  }
  printf("%s\n", failures ? "coherence: FAILED" : "coherence: PASS");
  return failures ? 1 : 0;
}
