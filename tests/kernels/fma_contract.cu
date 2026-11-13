// nvcc's default -fmad=true turns `a * b + c` into one fma, so a kernel's last bits depend on the contraction.
// Each expression below must round as the fma nvcc emits for it (cpp/mvcc-llvm/fma_fusion.cpp), not as a
// multiply and an add.
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

constexpr int kShapes = 6;

__global__ void shapes(const float* a, const float* b, const float* c, const float* d, float* out, int n) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
    const float m = a[i] * b[i];
    float* o = out + kShapes * i;
    o[0] = m + c[i];                  // fma(a, b, c)
    o[1] = m - d[i];                  // fma(a, b, -d), the same fmul fused again
    o[2] = c[i] - a[i] * b[i];        // fma(-a, b, c)
    o[3] = a[i] * b[i] + c[i] * d[i]; // fma(a, b, c * d)
    o[4] = -(a[i] * b[i]) - c[i];     // fma(-a, b, -c)
    o[5] = d[i] + -(c[i] * b[i]);     // fma(-c, b, d)
  }
}

static uint32_t bits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

int main() {
  const int n = 1 << 20;
  std::mt19937 rng(11);
  std::uniform_real_distribution<float> u(-4.f, 4.f);
  std::vector<float> h[4];
  for (auto& v : h) { v.resize(n); for (float& x : v) x = u(rng); }
  float* dev[4];
  for (int k = 0; k < 4; ++k) {
    cudaMalloc(&dev[k], 4 * n);
    cudaMemcpy(dev[k], h[k].data(), 4 * n, cudaMemcpyHostToDevice);
  }
  float* dout;
  cudaMalloc(&dout, 4 * kShapes * n);
  shapes<<<512, 256>>>(dev[0], dev[1], dev[2], dev[3], dout, n);
  std::vector<float> out(kShapes * n);
  cudaMemcpy(out.data(), dout, 4 * kShapes * n, cudaMemcpyDeviceToHost);
  const char* names[kShapes] = {"a*b + c", "a*b - d (shared fmul)", "c - a*b", "a*b + c*d", "-(a*b) - c", "d + -(c*b)"};
  int failures = 0;
  for (int s = 0; s < kShapes; ++s) {
    int bad = 0;
    for (int i = 0; i < n; ++i) {
      const float a = h[0][i], b = h[1][i], c = h[2][i], d = h[3][i];
      volatile float cd = c * d;
      const float ref[kShapes] = {std::fma(a, b, c), std::fma(a, b, -d), std::fma(-a, b, c),
                                  std::fma(a, b, cd), std::fma(-a, b, -c), std::fma(-c, b, d)};
      bad += bits(out[kShapes * i + s]) != bits(ref[s]);
    }
    std::printf("  %-24s %s (%d/%d)\n", names[s], bad ? "FAIL" : "ok", bad, n);
    failures += bad != 0;
  }
  return failures != 0;
}
