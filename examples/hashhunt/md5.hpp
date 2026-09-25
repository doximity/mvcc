// RFC 1321 MD5 of a message shorter than 56 bytes (one block after padding).
#pragma once
#include <cstdint>

#ifndef __CUDACC__
#define __host__
#define __device__
#endif

__host__ __device__ static inline uint32_t md5_rol(uint32_t x, int n) {
  return (x << n) | (x >> (32 - n));
}

__host__ __device__ static inline void md5_short(const uint8_t* msg, int n, uint32_t out[4]) {
  uint8_t blk[64];
  for (int i = 0; i < 64; i++) blk[i] = 0;
  for (int i = 0; i < n; i++) blk[i] = msg[i];
  blk[n] = 0x80;
  uint32_t bits = (uint32_t)n * 8u;
  blk[56] = (uint8_t)(bits);
  blk[57] = (uint8_t)(bits >> 8);
  blk[58] = (uint8_t)(bits >> 16);
  blk[59] = (uint8_t)(bits >> 24);

  uint32_t w[16];
  for (int i = 0; i < 16; i++) {
    w[i] = (uint32_t)blk[4 * i] | ((uint32_t)blk[4 * i + 1] << 8) |
           ((uint32_t)blk[4 * i + 2] << 16) | ((uint32_t)blk[4 * i + 3] << 24);
  }

  uint32_t a = 0x67452301u, b = 0xefcdab89u, c = 0x98badcfeu, d = 0x10325476u;
  const int s[64] = {
    7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22,
    5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20, 5, 9, 14, 20,
    4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23,
    6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21
  };
  const uint32_t T[64] = {
    0xd76aa478u, 0xe8c7b756u, 0x242070dbu, 0xc1bdceeeu, 0xf57c0fafu, 0x4787c62au, 0xa8304613u, 0xfd469501u,
    0x698098d8u, 0x8b44f7afu, 0xffff5bb1u, 0x895cd7beu, 0x6b901122u, 0xfd987193u, 0xa679438eu, 0x49b40821u,
    0xf61e2562u, 0xc040b340u, 0x265e5a51u, 0xe9b6c7aau, 0xd62f105du, 0x02441453u, 0xd8a1e681u, 0xe7d3fbc8u,
    0x21e1cde6u, 0xc33707d6u, 0xf4d50d87u, 0x455a14edu, 0xa9e3e905u, 0xfcefa3f8u, 0x676f02d9u, 0x8d2a4c8au,
    0xfffa3942u, 0x8771f681u, 0x6d9d6122u, 0xfde5380cu, 0xa4beea44u, 0x4bdecfa9u, 0xf6bb4b60u, 0xbebfbc70u,
    0x289b7ec6u, 0xeaa127fau, 0xd4ef3085u, 0x04881d05u, 0xd9d4d039u, 0xe6db99e5u, 0x1fa27cf8u, 0xc4ac5665u,
    0xf4292244u, 0x432aff97u, 0xab9423a7u, 0xfc93a039u, 0x655b59c3u, 0x8f0ccc92u, 0xffeff47du, 0x85845dd1u,
    0x6fa87e4fu, 0xfe2ce6e0u, 0xa3014314u, 0x4e0811a1u, 0xf7537e82u, 0xbd3af235u, 0x2ad7d2bbu, 0xeb86d391u
  };

  uint32_t A = a, B = b, C = c, D = d;
  for (int i = 0; i < 64; i++) {
    uint32_t f, g;
    if (i < 16) { f = (B & C) | ((~B) & D); g = (uint32_t)i; }
    else if (i < 32) { f = (D & B) | ((~D) & C); g = (5u * (uint32_t)i + 1u) & 15u; }
    else if (i < 48) { f = B ^ C ^ D; g = (3u * (uint32_t)i + 5u) & 15u; }
    else { f = C ^ (B | (~D)); g = (7u * (uint32_t)i) & 15u; }
    uint32_t t = A + f + T[i] + w[g];
    A = D;
    D = C;
    C = B;
    B = B + md5_rol(t, s[i]);
  }
  out[0] = a + A;
  out[1] = b + B;
  out[2] = c + C;
  out[3] = d + D;
}

__host__ __device__ static inline void to_digits(int v, int digits, uint8_t* msg) {
  for (int i = digits - 1; i >= 0; i--) {
    msg[i] = (uint8_t)('0' + (v % 10));
    v /= 10;
  }
}

static inline void md5_hex(const uint32_t h[4], char* dst) {
  const char* hex = "0123456789abcdef";
  for (int i = 0; i < 4; i++) {
    for (int b = 0; b < 4; b++) {
      uint8_t v = (uint8_t)(h[i] >> (8 * b));
      dst[8 * i + 2 * b] = hex[v >> 4];
      dst[8 * i + 2 * b + 1] = hex[v & 0xf];
    }
  }
  dst[32] = 0;
}
