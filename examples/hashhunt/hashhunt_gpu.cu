// GPU hash search: MD5 every n-digit string against a target digest.
//
// Code of this kind usually ships as CUDA with no Metal source. mvcc compiles the .cu.
// Default is the 4-digit demo (1337). --bench is a full 8-digit scan against
// hashhunt_cpu.cpp. The bench kernel is specialized: register-only unrolled MD5,
// ASCII increment, many candidates per thread.
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include "md5.hpp"

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("FAIL %s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

#define MD5_F(x, y, z) ((z) ^ ((x) & ((y) ^ (z))))
#define MD5_G(x, y, z) ((y) ^ ((z) & ((x) ^ (y))))
#define MD5_H(x, y, z) ((x) ^ (y) ^ (z))
#define MD5_I(x, y, z) ((y) ^ ((x) | ~(z)))
#define MD5_STEP(f, a, b, c, d, x, t, s) \
  (a) += f((b), (c), (d)) + (x) + (t); \
  (a) = md5_rol((a), (s)) + (b)

// 8 ASCII digits, little-endian words: w0 = msg[0..3], w1 = msg[4..7].
// Padding for a 8-byte message is x2=0x80, x14=64 bits; other words are 0.
__device__ __forceinline__ bool md5_digits8_eq(uint32_t w0, uint32_t w1,
                                            uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3) {
  const uint32_t x2 = 0x80u, x14 = 64u;
  uint32_t a = 0x67452301u, b = 0xefcdab89u, c = 0x98badcfeu, d = 0x10325476u;

  MD5_STEP(MD5_F, a, b, c, d, w0, 0xd76aa478u, 7);
  MD5_STEP(MD5_F, d, a, b, c, w1, 0xe8c7b756u, 12);
  MD5_STEP(MD5_F, c, d, a, b, x2, 0x242070dbu, 17);
  MD5_STEP(MD5_F, b, c, d, a, 0, 0xc1bdceeeu, 22);
  MD5_STEP(MD5_F, a, b, c, d, 0, 0xf57c0fafu, 7);
  MD5_STEP(MD5_F, d, a, b, c, 0, 0x4787c62au, 12);
  MD5_STEP(MD5_F, c, d, a, b, 0, 0xa8304613u, 17);
  MD5_STEP(MD5_F, b, c, d, a, 0, 0xfd469501u, 22);
  MD5_STEP(MD5_F, a, b, c, d, 0, 0x698098d8u, 7);
  MD5_STEP(MD5_F, d, a, b, c, 0, 0x8b44f7afu, 12);
  MD5_STEP(MD5_F, c, d, a, b, 0, 0xffff5bb1u, 17);
  MD5_STEP(MD5_F, b, c, d, a, 0, 0x895cd7beu, 22);
  MD5_STEP(MD5_F, a, b, c, d, 0, 0x6b901122u, 7);
  MD5_STEP(MD5_F, d, a, b, c, 0, 0xfd987193u, 12);
  MD5_STEP(MD5_F, c, d, a, b, x14, 0xa679438eu, 17);
  MD5_STEP(MD5_F, b, c, d, a, 0, 0x49b40821u, 22);

  MD5_STEP(MD5_G, a, b, c, d, w1, 0xf61e2562u, 5);
  MD5_STEP(MD5_G, d, a, b, c, 0, 0xc040b340u, 9);
  MD5_STEP(MD5_G, c, d, a, b, 0, 0x265e5a51u, 14);
  MD5_STEP(MD5_G, b, c, d, a, w0, 0xe9b6c7aau, 20);
  MD5_STEP(MD5_G, a, b, c, d, 0, 0xd62f105du, 5);
  MD5_STEP(MD5_G, d, a, b, c, 0, 0x02441453u, 9);
  MD5_STEP(MD5_G, c, d, a, b, 0, 0xd8a1e681u, 14);
  MD5_STEP(MD5_G, b, c, d, a, 0, 0xe7d3fbc8u, 20);
  MD5_STEP(MD5_G, a, b, c, d, 0, 0x21e1cde6u, 5);
  MD5_STEP(MD5_G, d, a, b, c, x14, 0xc33707d6u, 9);
  MD5_STEP(MD5_G, c, d, a, b, 0, 0xf4d50d87u, 14);
  MD5_STEP(MD5_G, b, c, d, a, 0, 0x455a14edu, 20);
  MD5_STEP(MD5_G, a, b, c, d, 0, 0xa9e3e905u, 5);
  MD5_STEP(MD5_G, d, a, b, c, x2, 0xfcefa3f8u, 9);
  MD5_STEP(MD5_G, c, d, a, b, 0, 0x676f02d9u, 14);
  MD5_STEP(MD5_G, b, c, d, a, 0, 0x8d2a4c8au, 20);

  MD5_STEP(MD5_H, a, b, c, d, 0, 0xfffa3942u, 4);
  MD5_STEP(MD5_H, d, a, b, c, 0, 0x8771f681u, 11);
  MD5_STEP(MD5_H, c, d, a, b, 0, 0x6d9d6122u, 16);
  MD5_STEP(MD5_H, b, c, d, a, x14, 0xfde5380cu, 23);
  MD5_STEP(MD5_H, a, b, c, d, w1, 0xa4beea44u, 4);
  MD5_STEP(MD5_H, d, a, b, c, 0, 0x4bdecfa9u, 11);
  MD5_STEP(MD5_H, c, d, a, b, 0, 0xf6bb4b60u, 16);
  MD5_STEP(MD5_H, b, c, d, a, 0, 0xbebfbc70u, 23);
  MD5_STEP(MD5_H, a, b, c, d, 0, 0x289b7ec6u, 4);
  MD5_STEP(MD5_H, d, a, b, c, w0, 0xeaa127fau, 11);
  MD5_STEP(MD5_H, c, d, a, b, 0, 0xd4ef3085u, 16);
  MD5_STEP(MD5_H, b, c, d, a, 0, 0x04881d05u, 23);
  MD5_STEP(MD5_H, a, b, c, d, 0, 0xd9d4d039u, 4);
  MD5_STEP(MD5_H, d, a, b, c, 0, 0xe6db99e5u, 11);
  MD5_STEP(MD5_H, c, d, a, b, 0, 0x1fa27cf8u, 16);
  MD5_STEP(MD5_H, b, c, d, a, x2, 0xc4ac5665u, 23);

  MD5_STEP(MD5_I, a, b, c, d, w0, 0xf4292244u, 6);
  MD5_STEP(MD5_I, d, a, b, c, 0, 0x432aff97u, 10);
  MD5_STEP(MD5_I, c, d, a, b, x14, 0xab9423a7u, 15);
  MD5_STEP(MD5_I, b, c, d, a, 0, 0xfc93a039u, 21);
  MD5_STEP(MD5_I, a, b, c, d, 0, 0x655b59c3u, 6);
  MD5_STEP(MD5_I, d, a, b, c, 0, 0x8f0ccc92u, 10);
  MD5_STEP(MD5_I, c, d, a, b, 0, 0xffeff47du, 15);
  MD5_STEP(MD5_I, b, c, d, a, w1, 0x85845dd1u, 21);
  MD5_STEP(MD5_I, a, b, c, d, 0, 0x6fa87e4fu, 6);
  MD5_STEP(MD5_I, d, a, b, c, 0, 0xfe2ce6e0u, 10);
  MD5_STEP(MD5_I, c, d, a, b, 0, 0xa3014314u, 15);
  MD5_STEP(MD5_I, b, c, d, a, 0, 0x4e0811a1u, 21);
  MD5_STEP(MD5_I, a, b, c, d, 0, 0xf7537e82u, 6);
  MD5_STEP(MD5_I, d, a, b, c, 0, 0xbd3af235u, 10);
  MD5_STEP(MD5_I, c, d, a, b, x2, 0x2ad7d2bbu, 15);
  MD5_STEP(MD5_I, b, c, d, a, 0, 0xeb86d391u, 21);

  return (0x67452301u + a) == t0 && (0xefcdab89u + b) == t1 &&
         (0x98badcfeu + c) == t2 && (0x10325476u + d) == t3;
}

// Two ASCII digits for n in 0..99, packed in the low 16 bits.
__device__ __forceinline__ uint32_t ascii_pair(int n) {
  return (uint32_t)('0' + n / 10) | ((uint32_t)('0' + n % 10) << 8);
}

__device__ __forceinline__ void digits8_words(int v, uint32_t& w0, uint32_t& w1) {
  int p0 = v / 1000000;
  int r0 = v - p0 * 1000000;
  int p1 = r0 / 10000;
  int r1 = r0 - p1 * 10000;
  int p2 = r1 / 100;
  int p3 = r1 - p2 * 100;
  w0 = ascii_pair(p0) | (ascii_pair(p1) << 16);
  w1 = ascii_pair(p2) | (ascii_pair(p3) << 16);
}

// Add 1 to an 8-digit ASCII value stored in (w0,w1). Carry from the ones place.
__device__ __forceinline__ void digits8_inc(uint32_t& w0, uint32_t& w1) {
  uint32_t b = (w1 >> 24) + 1u;
  if (b <= (uint32_t)'9') { w1 = (w1 & 0x00ffffffu) | (b << 24); return; }
  w1 = (w1 & 0x00ffffffu) | 0x30000000u;
  b = ((w1 >> 16) & 0xffu) + 1u;
  if (b <= (uint32_t)'9') { w1 = (w1 & 0xff00ffffu) | (b << 16); return; }
  w1 = (w1 & 0xff00ffffu) | 0x00300000u;
  b = ((w1 >> 8) & 0xffu) + 1u;
  if (b <= (uint32_t)'9') { w1 = (w1 & 0xffff00ffu) | (b << 8); return; }
  w1 = (w1 & 0xffff00ffu) | 0x00003000u;
  b = (w1 & 0xffu) + 1u;
  if (b <= (uint32_t)'9') { w1 = (w1 & 0xffffff00u) | b; return; }
  w1 = (w1 & 0xffffff00u) | 0x30u;
  b = (w0 >> 24) + 1u;
  if (b <= (uint32_t)'9') { w0 = (w0 & 0x00ffffffu) | (b << 24); return; }
  w0 = (w0 & 0x00ffffffu) | 0x30000000u;
  b = ((w0 >> 16) & 0xffu) + 1u;
  if (b <= (uint32_t)'9') { w0 = (w0 & 0xff00ffffu) | (b << 16); return; }
  w0 = (w0 & 0xff00ffffu) | 0x00300000u;
  b = ((w0 >> 8) & 0xffu) + 1u;
  if (b <= (uint32_t)'9') { w0 = (w0 & 0xffff00ffu) | (b << 8); return; }
  w0 = (w0 & 0xffff00ffu) | 0x00003000u;
  w0 = (w0 & 0xffffff00u) | ((w0 + 1u) & 0xffu);
}

__global__ void hunt4(uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3, int* found) {
  int cand = (int)(blockIdx.x * blockDim.x + threadIdx.x);
  if (cand > 9999) return;
  uint8_t msg[4];
  to_digits(cand, 4, msg);
  uint32_t h[4];
  md5_short(msg, 4, h);
  if (h[0] == t0 && h[1] == t1 && h[2] == t2 && h[3] == t3) atomicMin(found, cand);
}

// 64 consecutive candidates per thread: one encode, then ASCII increment.
constexpr int kPerThread = 64;

__global__ void hunt8(uint32_t t0, uint32_t t1, uint32_t t2, uint32_t t3, int space, int* found) {
  int first = (int)((blockIdx.x * blockDim.x + threadIdx.x) * kPerThread);
  if (first >= space) return;
  int n = space - first;
  if (n > kPerThread) n = kPerThread;
  uint32_t w0, w1;
  digits8_words(first, w0, w1);
  int hit = space;
  #pragma unroll 8
  for (int i = 0; i < n; i++) {
    if (md5_digits8_eq(w0, w1, t0, t1, t2, t3)) hit = first + i;
    if (i + 1 < n) digits8_inc(w0, w1);
  }
  if (hit < space) atomicMin(found, hit);
}

int main(int argc, char** argv) {
  int bench = 0;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--bench")) bench = 1;
    else {
      fprintf(stderr, "usage: hashhunt_gpu [--bench]\n");
      return 2;
    }
  }
  const int digits = bench ? 8 : 4;
  const int space = bench ? 100000000 : 10000;
  const char* preimage = bench ? "13371337" : "1337";

  uint32_t target[4];
  md5_short(reinterpret_cast<const uint8_t*>(preimage), digits, target);
  char digest[33];
  md5_hex(target, digest);
  printf("target  md5(\"%s\") = %s\n", preimage, digest);
  printf("kernel  hunt  %d-digit space %d  (CUDA only — no authored .metal)\n", digits, space);

  int* dfound;
  int not_found = space;
  CK(cudaMalloc(&dfound, sizeof(int)));
  CK(cudaMemcpy(dfound, &not_found, sizeof(int), cudaMemcpyHostToDevice));

  int blocks = bench ? (space + kPerThread * 256 - 1) / (kPerThread * 256)
                    : (space + 255) / 256;
  auto launch = [&]() {
    if (bench) hunt8<<<blocks, 256>>>(target[0], target[1], target[2], target[3], space, dfound);
    else hunt4<<<blocks, 256>>>(target[0], target[1], target[2], target[3], dfound);
  };
  launch();
  CK(cudaDeviceSynchronize());

  CK(cudaMemcpy(dfound, &not_found, sizeof(int), cudaMemcpyHostToDevice));
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0));
  CK(cudaEventCreate(&e1));
  CK(cudaEventRecord(e0));
  launch();
  CK(cudaEventRecord(e1));
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  float ms = 0;
  CK(cudaEventElapsedTime(&ms, e0, e1));

  int found = not_found;
  CK(cudaMemcpy(&found, dfound, sizeof(int), cudaMemcpyDeviceToHost));
  cudaDeviceProp p;
  CK(cudaGetDeviceProperties(&p, 0));
  int want = bench ? 13371337 : 1337;
  double mhs = (space / 1e6) / (ms / 1e3);
  printf("device  %s\n", p.name);
  printf("gpu     %.3f ms  %.1f MH/s\n", ms, mhs);
  if (found >= space) {
    printf("FAIL    no candidate matched\n");
    return 1;
  }
  printf("found   %0*d\n", digits, found);
  printf("%s\n", found == want ? "PASS" : "FAIL");
  CK(cudaFree(dfound));
  return found == want ? 0 : 1;
}
