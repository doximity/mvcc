// mma.sync m16n8k16 with f32 accumulate must return an NVIDIA tensor core's bits, not just a close product: the
// hashes are of an A10G's outputs (sm_86) on these seeded inputs, which span 25 binades with cancellation.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

constexpr int kTests = 2048;

__global__ void run(const uint16_t* A, const uint16_t* B, const float* C, float* D, int f16) {
  int t = blockIdx.x, lane = threadIdx.x, g = lane >> 2, q = lane & 3;
  const uint16_t* a = A + t * 256;
  const uint16_t* b = B + t * 128;
  const float* c = C + t * 128;
  float* d = D + t * 128;
  auto pk = [](const uint16_t* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 16); };
  uint32_t a0 = pk(a + g * 16 + 2 * q), a1 = pk(a + (g + 8) * 16 + 2 * q);
  uint32_t a2 = pk(a + g * 16 + 2 * q + 8), a3 = pk(a + (g + 8) * 16 + 2 * q + 8);
  uint32_t b0 = pk(b + g * 16 + 2 * q), b1 = pk(b + g * 16 + 2 * q + 8);
  float c0 = c[g * 8 + 2 * q], c1 = c[g * 8 + 2 * q + 1], c2 = c[(g + 8) * 8 + 2 * q], c3 = c[(g + 8) * 8 + 2 * q + 1];
  float d0, d1, d2, d3;
  if (f16)
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
                 : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1), "f"(c0), "f"(c1), "f"(c2), "f"(c3));
  d[g * 8 + 2 * q] = d0;
  d[g * 8 + 2 * q + 1] = d1;
  d[(g + 8) * 8 + 2 * q] = d2;
  d[(g + 8) * 8 + 2 * q + 1] = d3;
}

static uint64_t rng = 0x9e3779b97f4a7c15ull;
static uint64_t next() {
  uint64_t z = (rng += 0x9e3779b97f4a7c15ull);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
  return z ^ (z >> 31);
}
static int range(int lo, int hi) { return lo + (int)(next() % (uint64_t)(hi - lo + 1)); }

int main() {
  const uint64_t want[2] = {0xd970ab25f7b68aadull, 0xe79722894af5e279ull};
  int failures = 0;
  for (int f16 = 0; f16 < 2; ++f16) {
    std::vector<uint16_t> A(kTests * 256), B(kTests * 128);
    std::vector<float> C(kTests * 128), D(kTests * 128);
    for (int t = 0; t < kTests; ++t) {
      const int spread[4] = {0, 2, 5, f16 ? 6 : 12}, s = spread[t % 4], cm = (t / 4) % 4;
      auto val = [&]() -> uint16_t {
        if (next() % 10 == 0) return 0;
        const int m = (int)(next() % (f16 ? 1024 : 128)), e = range(-s, s), sign = (int)(next() & 1);
        return f16 ? (uint16_t)((sign << 15) | ((e + 15) << 10) | m) : (uint16_t)((sign << 15) | ((e + 127) << 7) | m);
      };
      for (int i = 0; i < 256; ++i) A[t * 256 + i] = val();
      for (int i = 0; i < 128; ++i) B[t * 128 + i] = val();
      const int lo[4] = {0, -2 * s - 8, -2 * s - 1, 2 * s}, hi[4] = {0, -2 * s, 2 * s + 1, 2 * s + 10};
      for (int i = 0; i < 128; ++i) {
        if (cm == 0) { C[t * 128 + i] = 0.f; continue; }
        const int sign = next() & 1 ? -1 : 1, e = range(lo[cm], hi[cm]);
        const double m = 1.0 + (double)(next() % (1u << 23)) / (1u << 23);
        C[t * 128 + i] = (float)(sign * std::ldexp(m, e));
      }
    }
    uint16_t *dA, *dB;
    float *dC, *dD;
    cudaMalloc(&dA, A.size() * 2);
    cudaMalloc(&dB, B.size() * 2);
    cudaMalloc(&dC, C.size() * 4);
    cudaMalloc(&dD, D.size() * 4);
    cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, B.data(), B.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dC, C.data(), C.size() * 4, cudaMemcpyHostToDevice);
    run<<<kTests, 32>>>(dA, dB, dC, dD, f16);
    cudaMemcpy(D.data(), dD, D.size() * 4, cudaMemcpyDeviceToHost);
    uint64_t h = 1469598103934665603ull;
    for (float f : D) { uint32_t u; std::memcpy(&u, &f, 4); h = (h ^ u) * 1099511628211ull; }
    std::printf("  %-5s %016llx %s\n", f16 ? "f16" : "bf16", (unsigned long long)h, h == want[f16] ? "ok" : "FAIL (A10G differs)");
    failures += h != want[f16];
  }
  return failures != 0;
}
