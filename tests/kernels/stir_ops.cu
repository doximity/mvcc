// Corpus for the semantic iteration graph / coordinate discovery / STIR golden tests.
// Compiled with MVCC_SIG=1 MVCC_SIG_ONLY=1 MVCC_SIG_DUMP=<file>; the dump is compared against tests/stir/stir_ops.golden.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
// elementwise: o=(i) singleton D
__global__ void __launch_bounds__(256) silu_mul(const float* __restrict__ a, const float* __restrict__ g, float* __restrict__ out, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) { float x = a[i]; out[i] = x / (1.f + __expf(-x)) * g[i]; }
}
// row reduction with a shuffle ladder: o=(row), r=(k)
__global__ void __launch_bounds__(128) rowsum(const float* __restrict__ x, float* __restrict__ out, int K) {
  int row = blockIdx.x; float s = 0.f;
  for (int k = threadIdx.x; k < K; k += 128) s += x[(size_t)row * K + k];
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  __shared__ float part[4];
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) out[row] = part[0] + part[1] + part[2] + part[3];
}
// gemv: o=(n), r=(k), one output per thread
__global__ void __launch_bounds__(64) gemv(const __nv_bfloat16* __restrict__ W, const float* __restrict__ x, float* __restrict__ y, int N, int K) {
  int n = blockIdx.x * 64 + threadIdx.x; if (n >= N) return;
  float acc = 0.f;
  for (int k = 0; k < K; k += 4) {
    acc += __bfloat162float(W[(size_t)n * K + k]) * x[k] + __bfloat162float(W[(size_t)n * K + k + 1]) * x[k + 1]
         + __bfloat162float(W[(size_t)n * K + k + 2]) * x[k + 2] + __bfloat162float(W[(size_t)n * K + k + 3]) * x[k + 3];
  }
  y[n] = acc;
}
// gather: o=(row, d) via index tensor
__global__ void __launch_bounds__(128) gather(const float* __restrict__ src, const int* __restrict__ idx, float* __restrict__ dst, int D) {
  int row = blockIdx.x; int r = idx[row];
  for (int d = threadIdx.x; d < D; d += 128) dst[(size_t)row * D + d] = src[(size_t)r * D + d];
}
// online softmax over a row: r=(k) with the (m, l) tuple recurrence
__global__ void __launch_bounds__(128) softmax_row(const float* __restrict__ x, float* __restrict__ out, int K) {
  int row = blockIdx.x;
  float m = -INFINITY, l = 0.f;
  for (int k = threadIdx.x; k < K; k += 128) {
    float v = x[(size_t)row * K + k];
    float mn = fmaxf(m, v);
    l = l * __expf(m - mn) + __expf(v - mn);
    m = mn;
  }
  for (int o = 16; o > 0; o >>= 1) {
    float mo = __shfl_xor_sync(0xffffffffu, m, o), lo = __shfl_xor_sync(0xffffffffu, l, o);
    float mn = fmaxf(m, mo); l = l * __expf(m - mn) + lo * __expf(mo - mn); m = mn;
  }
  __shared__ float pm[4], pl[4];
  if ((threadIdx.x & 31) == 0) { pm[threadIdx.x >> 5] = m; pl[threadIdx.x >> 5] = l; }
  __syncthreads();
  if (threadIdx.x < 32) {
    float M = fmaxf(fmaxf(pm[0], pm[1]), fmaxf(pm[2], pm[3]));
    float L = pl[0] * __expf(pm[0] - M) + pl[1] * __expf(pm[1] - M) + pl[2] * __expf(pm[2] - M) + pl[3] * __expf(pm[3] - M);
    for (int k = threadIdx.x; k < K; k += 32) out[(size_t)row * K + k] = __expf(x[(size_t)row * K + k] - M) / L;
  }
}
// RMSNorm: sum of squares over r, then broadcast scale
__global__ void __launch_bounds__(128) rmsnorm(const float* __restrict__ x, const float* __restrict__ w, float* __restrict__ out, int K) {
  int row = blockIdx.x; float s = 0.f;
  for (int k = threadIdx.x; k < K; k += 128) { float v = x[(size_t)row * K + k]; s += v * v; }
  for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
  __shared__ float part[4];
  if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = s;
  __syncthreads();
  if (threadIdx.x == 0) {
    float tot = part[0] + part[1] + part[2] + part[3];
    part[0] = rsqrtf(tot / (float)K + 1e-6f);
  }
  __syncthreads();
  float inv = part[0];
  for (int k = threadIdx.x; k < K; k += 128) out[(size_t)row * K + k] = x[(size_t)row * K + k] * inv * w[k];
}
// RoPE pair-broadcast: o=(t,d) reads x[t,d] and x[t,d+D/2]
__global__ void __launch_bounds__(64) rope(const float* __restrict__ x, const float* __restrict__ cos, const float* __restrict__ sin, float* __restrict__ out, int D) {
  int t = blockIdx.x, d = threadIdx.x, half = D >> 1;
  if (d >= half) return;
  float a = x[(size_t)t * D + d], b = x[(size_t)t * D + d + half];
  float c = cos[d], s = sin[d];
  out[(size_t)t * D + d] = a * c - b * s;
  out[(size_t)t * D + d + half] = a * s + b * c;
}
// decode attention: one query, online (max,sum,acc) over KV (STIR, not an mma pattern)
__global__ void __launch_bounds__(32) attn(const float* __restrict__ q, const float* __restrict__ K, const float* __restrict__ V, float* __restrict__ out, int N, int D) {
  int d = blockIdx.x * 32 + threadIdx.x; if (d >= D) return;
  float m = -INFINITY, l = 0.f, acc = 0.f;
  for (int n = 0; n < N; n++) {
    float s = 0.f;
    for (int k = 0; k < D; k++) s += q[k] * K[(size_t)n * D + k];
    float mn = fmaxf(m, s), a = __expf(m - mn), p = __expf(s - mn);
    acc = acc * a + p * V[(size_t)n * D + d];
    l = l * a + p;
    m = mn;
  }
  out[d] = acc / l;
}
// scalar GEMM: C[m,n] += A[m,k]*B[k,n]
// 16×16×32 half tile: tensor-op contraction (__mvcc_tp_mma)
__global__ void __launch_bounds__(32) hgemm16(const __half* __restrict__ A, const __half* __restrict__ B, float* __restrict__ C) {
  int tid = threadIdx.x;
  for (int i = tid; i < 256; i += 32) {
    int m = i / 16, n = i % 16;
    float acc = 0.f;
    for (int k = 0; k < 32; k++) acc += __half2float(A[m * 32 + k]) * __half2float(B[k * 16 + n]);
    C[i] = acc;
  }
}
// gemm_ptx-shaped ABI (A, B, bias, C): same 16×16×32 tile plus a bias vector
__global__ void __launch_bounds__(32) hgemm16b(const __half* __restrict__ A, const __half* __restrict__ B,
                                              const float* __restrict__ bias, float* __restrict__ C) {
  int tid = threadIdx.x;
  for (int i = tid; i < 256; i += 32) {
    int m = i / 16, n = i % 16;
    float acc = 0.f;
    for (int k = 0; k < 32; k++) acc += __half2float(A[m * 32 + k]) * __half2float(B[k * 16 + n]);
    C[i] = acc + bias[n];
  }
}
__global__ void __launch_bounds__(64) sgemm(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  int o = blockIdx.x * 64 + threadIdx.x; if (o >= M * N) return;
  int m = o / N, n = o % N;
  float acc = 0.f;
  for (int k = 0; k < K; k++) acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
  C[o] = acc;
}
// GDN linear state: H_t = a_t H_{t-1} + B_t, sequential in t
__global__ void __launch_bounds__(64) gdn(const float* __restrict__ a, const float* __restrict__ B, float* __restrict__ H, int N, int steps) {
  int d = blockIdx.x * 64 + threadIdx.x; if (d >= N) return;
  float h = 0.f;
  for (int t = 0; t < steps; t++) h = a[t] * h + B[(size_t)t * N + d];
  H[d] = h;
}
// scatter-add through an index tensor: the destination row is data-dependent per element
__global__ void __launch_bounds__(128) scatter_add(const float* __restrict__ src, const int* __restrict__ idx, float* __restrict__ dst, int D) {
  int row = blockIdx.x;
  for (int d = threadIdx.x; d < D; d += 128) atomicAdd(&dst[(size_t)idx[row * D + d] * D + d], src[(size_t)row * D + d]);
}
static int relBad(const float* got, const float* ref, int n, float tol) {
  int bad = 0;
  for (int i = 0; i < n; i++) {
    float e = fabsf(got[i] - ref[i]) / (fabsf(ref[i]) + 1e-6f);
    if (e > tol) bad++;
  }
  return bad;
}
static float bf16bits(uint16_t h) {
  uint32_t u = (uint32_t)h << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
int main() {
  const int n = 1000;
  std::vector<float> ha(n), hg(n), href(n), hout(n);
  for (int i = 0; i < n; i++) {
    ha[i] = 0.01f * (float)(i - 400);
    hg[i] = 1.5f - 0.001f * (float)i;
    float x = ha[i];
    href[i] = (x / (1.f + expf(-x))) * hg[i];
  }
  float *a, *g, *out;
  if (cudaMalloc(&a, n * sizeof(float)) || cudaMalloc(&g, n * sizeof(float)) || cudaMalloc(&out, n * sizeof(float))) return 2;
  cudaMemcpy(a, ha.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(g, hg.data(), n * sizeof(float), cudaMemcpyHostToDevice);
  silu_mul<<<(n + 255) / 256, 256>>>(a, g, out, n);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(hout.data(), out, n * sizeof(float), cudaMemcpyDeviceToHost);
  int bad = relBad(hout.data(), href.data(), n, 1e-5f);
  cudaFree(a); cudaFree(g); cudaFree(out);
  if (bad) { printf("silu_mul: %d / %d mismatch\n", bad, n); return 1; }
  printf("silu_mul: %d ok (check+emit elementwise)\n", n);

  const int rows = 8, K = 256;
  std::vector<float> hx((size_t)rows * K), hrefs(rows), houts(rows);
  for (int r = 0; r < rows; r++) {
    float s = 0.f;
    for (int k = 0; k < K; k++) {
      float v = 0.01f * (float)((r + 1) * (k - 80));
      hx[(size_t)r * K + k] = v;
      s += v;
    }
    hrefs[r] = s;
  }
  float *dx, *dy;
  if (cudaMalloc(&dx, hx.size() * sizeof(float)) || cudaMalloc(&dy, rows * sizeof(float))) return 2;
  cudaMemcpy(dx, hx.data(), hx.size() * sizeof(float), cudaMemcpyHostToDevice);
  rowsum<<<rows, 128>>>(dx, dy, K);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(houts.data(), dy, rows * sizeof(float), cudaMemcpyDeviceToHost);
  bad = relBad(houts.data(), hrefs.data(), rows, 1e-4f);
  if (bad) { printf("rowsum: %d / %d mismatch\n", bad, rows); return 1; }
  printf("rowsum: %d rows ok (check+emit reduction)\n", rows);

  const int D = 64, V = 16;
  std::vector<float> hsrc((size_t)V * D), hdst((size_t)rows * D), hrefg((size_t)rows * D);
  std::vector<int> hidx(rows);
  for (int i = 0; i < V * D; i++) hsrc[i] = 0.1f * (float)(i % 17);
  for (int r = 0; r < rows; r++) {
    hidx[r] = (3 * r + 1) % V;
    for (int d = 0; d < D; d++) hrefg[(size_t)r * D + d] = hsrc[(size_t)hidx[r] * D + d];
  }
  float *dsrc, *ddst; int *didx;
  if (cudaMalloc(&dsrc, hsrc.size() * sizeof(float)) || cudaMalloc(&ddst, hdst.size() * sizeof(float)) || cudaMalloc(&didx, rows * sizeof(int))) return 2;
  cudaMemcpy(dsrc, hsrc.data(), hsrc.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(didx, hidx.data(), rows * sizeof(int), cudaMemcpyHostToDevice);
  gather<<<rows, 128>>>(dsrc, didx, ddst, D);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(hdst.data(), ddst, hdst.size() * sizeof(float), cudaMemcpyDeviceToHost);
  bad = relBad(hdst.data(), hrefg.data(), rows * D, 1e-6f);
  cudaFree(dsrc); cudaFree(ddst); cudaFree(didx);
  if (bad) { printf("gather: %d / %d mismatch\n", bad, rows * D); return 1; }
  printf("gather: %d ok (check+emit IndexTensor)\n", rows * D);

  const int N = 32, GK = 64;
  std::vector<uint16_t> hW((size_t)N * GK);
  std::vector<float> hxg(GK), hy(N), hrefy(N);
  for (int i = 0; i < N * GK; i++) hW[i] = (uint16_t)(0x3C00 + (i % 32)); // bf16 near 1
  for (int k = 0; k < GK; k++) hxg[k] = 0.02f * (float)(k + 1);
  for (int i = 0; i < N; i++) {
    float acc = 0.f;
    for (int k = 0; k < GK; k++) acc += bf16bits(hW[(size_t)i * GK + k]) * hxg[k];
    hrefy[i] = acc;
  }
  __nv_bfloat16 *dW; float *dxv, *dyv;
  if (cudaMalloc(&dW, hW.size() * sizeof(uint16_t)) || cudaMalloc(&dxv, GK * sizeof(float)) || cudaMalloc(&dyv, N * sizeof(float))) return 2;
  cudaMemcpy(dW, hW.data(), hW.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dxv, hxg.data(), GK * sizeof(float), cudaMemcpyHostToDevice);
  gemv<<<(N + 63) / 64, 64>>>(dW, dxv, dyv, N, GK);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(hy.data(), dyv, N * sizeof(float), cudaMemcpyDeviceToHost);
  bad = relBad(hy.data(), hrefy.data(), N, 1e-4f);
  cudaFree(dW); cudaFree(dxv); cudaFree(dyv);
  if (bad) { printf("gemv: %d / %d mismatch\n", bad, N); return 1; }
  printf("gemv: %d ok (check+emit scalar-fma)\n", N);

  const int srows = 4, SK = 64;
  std::vector<float> hsx((size_t)srows * SK), hso((size_t)srows * SK), hrefsft((size_t)srows * SK);
  for (int r = 0; r < srows; r++) {
    float m = -INFINITY, L = 0.f;
    for (int k = 0; k < SK; k++) {
      float v = 0.05f * (float)((r + 1) * (k - 20));
      hsx[(size_t)r * SK + k] = v;
      m = fmaxf(m, v);
    }
    for (int k = 0; k < SK; k++) L += expf(hsx[(size_t)r * SK + k] - m);
    for (int k = 0; k < SK; k++) hrefsft[(size_t)r * SK + k] = expf(hsx[(size_t)r * SK + k] - m) / L;
  }
  float *dsx, *dso;
  if (cudaMalloc(&dsx, hsx.size() * sizeof(float)) || cudaMalloc(&dso, hso.size() * sizeof(float))) return 2;
  cudaMemcpy(dsx, hsx.data(), hsx.size() * sizeof(float), cudaMemcpyHostToDevice);
  softmax_row<<<srows, 128>>>(dsx, dso, SK);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(hso.data(), dso, hso.size() * sizeof(float), cudaMemcpyDeviceToHost);
  bad = relBad(hso.data(), hrefsft.data(), srows * SK, 2e-3f);
  cudaFree(dsx); cudaFree(dso); cudaFree(dx); cudaFree(dy);
  if (bad) { printf("softmax_row: %d / %d mismatch\n", bad, srows * SK); return 1; }
  printf("softmax_row: %d ok (check+emit online)\n", srows * SK);

  const int rrows = 4, RK = 128;
  std::vector<float> hrx((size_t)rrows * RK), hrw(RK), hro((size_t)rrows * RK), hrr((size_t)rrows * RK);
  for (int k = 0; k < RK; k++) hrw[k] = 0.8f + 0.002f * (float)k;
  for (int r = 0; r < rrows; r++) {
    float ss = 0.f;
    for (int k = 0; k < RK; k++) {
      float v = 0.03f * (float)((r + 1) * (k - 40));
      hrx[(size_t)r * RK + k] = v;
      ss += v * v;
    }
    float inv = 1.f / sqrtf(ss / (float)RK + 1e-6f);
    for (int k = 0; k < RK; k++) hrr[(size_t)r * RK + k] = hrx[(size_t)r * RK + k] * inv * hrw[k];
  }
  float *drx, *drw, *dro;
  if (cudaMalloc(&drx, hrx.size() * sizeof(float)) || cudaMalloc(&drw, RK * sizeof(float)) || cudaMalloc(&dro, hro.size() * sizeof(float))) return 2;
  cudaMemcpy(drx, hrx.data(), hrx.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(drw, hrw.data(), RK * sizeof(float), cudaMemcpyHostToDevice);
  rmsnorm<<<rrows, 128>>>(drx, drw, dro, RK);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(hro.data(), dro, hro.size() * sizeof(float), cudaMemcpyDeviceToHost);
  bad = relBad(hro.data(), hrr.data(), rrows * RK, 2e-3f);
  cudaFree(drx); cudaFree(drw); cudaFree(dro);
  if (bad) { printf("rmsnorm: %d / %d mismatch\n", bad, rrows * RK); return 1; }
  printf("rmsnorm: %d ok (check+emit sum-of-squares + scale)\n", rrows * RK);

  const int tokens = 3, RD = 64;
  std::vector<float> hxx((size_t)tokens * RD), hcos(RD / 2), hsin(RD / 2), hrope((size_t)tokens * RD), hrefp((size_t)tokens * RD);
  for (int d = 0; d < RD / 2; d++) { hcos[d] = cosf(0.1f * (float)d); hsin[d] = sinf(0.1f * (float)d); }
  for (int t = 0; t < tokens; t++) for (int d = 0; d < RD; d++) hxx[(size_t)t * RD + d] = 0.02f * (float)(t + 1) * (float)(d - 10);
  for (int t = 0; t < tokens; t++) {
    for (int d = 0; d < RD / 2; d++) {
      float a = hxx[(size_t)t * RD + d], b = hxx[(size_t)t * RD + d + RD / 2];
      hrefp[(size_t)t * RD + d] = a * hcos[d] - b * hsin[d];
      hrefp[(size_t)t * RD + d + RD / 2] = a * hsin[d] + b * hcos[d];
    }
  }
  float *dxx, *dcos, *dsin, *drope;
  if (cudaMalloc(&dxx, hxx.size() * sizeof(float)) || cudaMalloc(&dcos, hcos.size() * sizeof(float)) || cudaMalloc(&dsin, hsin.size() * sizeof(float)) || cudaMalloc(&drope, hrope.size() * sizeof(float))) return 2;
  cudaMemcpy(dxx, hxx.data(), hxx.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dcos, hcos.data(), hcos.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dsin, hsin.data(), hsin.size() * sizeof(float), cudaMemcpyHostToDevice);
  rope<<<tokens, 64>>>(dxx, dcos, dsin, drope, RD);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(hrope.data(), drope, hrope.size() * sizeof(float), cudaMemcpyDeviceToHost);
  bad = relBad(hrope.data(), hrefp.data(), tokens * RD, 1e-5f);
  cudaFree(dxx); cudaFree(dcos); cudaFree(dsin); cudaFree(drope);
  if (bad) { printf("rope: %d / %d mismatch\n", bad, tokens * RD); return 1; }
  printf("rope: %d ok (check+emit pair-broadcast)\n", tokens * RD);

  const int AN = 8, AD = 16;
  std::vector<float> hq(AD), hK((size_t)AN * AD), hV((size_t)AN * AD), hao(AD), hrefa(AD);
  for (int k = 0; k < AD; k++) hq[k] = 0.05f * (float)(k - 4);
  for (int n = 0; n < AN; n++) for (int k = 0; k < AD; k++) {
    hK[(size_t)n * AD + k] = 0.04f * (float)((n + 1) * (k - 3));
    hV[(size_t)n * AD + k] = 0.03f * (float)((n + 2) * (k - 1));
  }
  for (int d = 0; d < AD; d++) {
    float m = -INFINITY, L = 0.f, acc = 0.f;
    for (int n = 0; n < AN; n++) {
      float s = 0.f;
      for (int k = 0; k < AD; k++) s += hq[k] * hK[(size_t)n * AD + k];
      float mn = fmaxf(m, s), a = expf(m - mn), p = expf(s - mn);
      acc = acc * a + p * hV[(size_t)n * AD + d];
      L = L * a + p;
      m = mn;
    }
    hrefa[d] = acc / L;
  }
  float *dq, *dK, *dV, *dao;
  if (cudaMalloc(&dq, AD * sizeof(float)) || cudaMalloc(&dK, hK.size() * sizeof(float)) || cudaMalloc(&dV, hV.size() * sizeof(float)) || cudaMalloc(&dao, AD * sizeof(float))) return 2;
  cudaMemcpy(dq, hq.data(), AD * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dK, hK.data(), hK.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dV, hV.data(), hV.size() * sizeof(float), cudaMemcpyHostToDevice);
  attn<<<1, 32>>>(dq, dK, dV, dao, AN, AD);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(hao.data(), dao, AD * sizeof(float), cudaMemcpyDeviceToHost);
  bad = relBad(hao.data(), hrefa.data(), AD, 3e-3f);
  cudaFree(dq); cudaFree(dK); cudaFree(dV); cudaFree(dao);
  if (bad) { printf("attn: %d / %d mismatch\n", bad, AD); return 1; }
  printf("attn: %d ok (check+emit online max,sum,acc)\n", AD);

  const int GN = 32, GT = 6;
  std::vector<float> hga(GT), hgB((size_t)GT * GN), hgH(GN), hrefgdn(GN);
  for (int t = 0; t < GT; t++) hga[t] = 0.7f + 0.03f * (float)t;
  for (int t = 0; t < GT; t++) for (int d = 0; d < GN; d++) hgB[(size_t)t * GN + d] = 0.01f * (float)((t + 1) * (d - 8));
  for (int d = 0; d < GN; d++) {
    float h = 0.f;
    for (int t = 0; t < GT; t++) h = hga[t] * h + hgB[(size_t)t * GN + d];
    hrefgdn[d] = h;
  }
  float *dga, *dgB, *dgH;
  if (cudaMalloc(&dga, GT * sizeof(float)) || cudaMalloc(&dgB, hgB.size() * sizeof(float)) || cudaMalloc(&dgH, GN * sizeof(float))) return 2;
  cudaMemcpy(dga, hga.data(), GT * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dgB, hgB.data(), hgB.size() * sizeof(float), cudaMemcpyHostToDevice);
  gdn<<<1, 64>>>(dga, dgB, dgH, GN, GT);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(hgH.data(), dgH, GN * sizeof(float), cudaMemcpyDeviceToHost);
  bad = relBad(hgH.data(), hrefgdn.data(), GN, 1e-5f);
  cudaFree(dga); cudaFree(dgB); cudaFree(dgH);
  if (bad) { printf("gdn: %d / %d mismatch\n", bad, GN); return 1; }
  printf("gdn: %d ok (check+emit linear-state scan)\n", GN);

  const int GM = 8, GN2 = 8, GK2 = 8;
  std::vector<float> hA((size_t)GM * GK2), hB((size_t)GK2 * GN2), hC((size_t)GM * GN2), hrefC((size_t)GM * GN2);
  for (int i = 0; i < GM * GK2; i++) hA[i] = 0.02f * (float)(i - 3);
  for (int i = 0; i < GK2 * GN2; i++) hB[i] = 0.03f * (float)(i - 5);
  for (int m = 0; m < GM; m++) for (int n = 0; n < GN2; n++) {
    float acc = 0.f;
    for (int k = 0; k < GK2; k++) acc += hA[(size_t)m * GK2 + k] * hB[(size_t)k * GN2 + n];
    hrefC[(size_t)m * GN2 + n] = acc;
  }
  float *dA, *dB, *dC;
  if (cudaMalloc(&dA, hA.size() * sizeof(float)) || cudaMalloc(&dB, hB.size() * sizeof(float)) || cudaMalloc(&dC, hC.size() * sizeof(float))) return 2;
  cudaMemcpy(dA, hA.data(), hA.size() * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(dB, hB.data(), hB.size() * sizeof(float), cudaMemcpyHostToDevice);
  sgemm<<<1, 64>>>(dA, dB, dC, GM, GN2, GK2);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(hC.data(), dC, hC.size() * sizeof(float), cudaMemcpyDeviceToHost);
  bad = relBad(hC.data(), hrefC.data(), GM * GN2, 1e-5f);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
  if (bad) { printf("sgemm: %d / %d mismatch\n", bad, GM * GN2); return 1; }
  printf("sgemm: %d ok (check+emit scalar GEMM)\n", GM * GN2);

  const int TM = 16, TK = 32;
  std::vector<uint16_t> hAh((size_t)TM * TK), hBh((size_t)TK * TM);
  std::vector<float> hCh((size_t)TM * TM), hrefH((size_t)TM * TM);
  auto toHalf = [](float x) -> uint16_t {
    union { float f; uint32_t u; } v; v.f = x;
    uint32_t u = v.u, s = (u >> 16) & 0x8000, e = (u >> 23) & 0xff, m = u & 0x7fffff;
    if (e == 0) return (uint16_t)s;
    int ne = (int)e - 127 + 15;
    if (ne <= 0) return (uint16_t)s;
    if (ne >= 31) return (uint16_t)(s | 0x7c00);
    return (uint16_t)(s | ((unsigned)ne << 10) | (m >> 13));
  };
  for (int i = 0; i < TM * TK; i++) hAh[i] = toHalf(0.02f * (float)(i - 7));
  for (int i = 0; i < TK * TM; i++) hBh[i] = toHalf(0.03f * (float)(i - 4));
  auto fromHalf = [](uint16_t h) -> float {
    uint32_t s = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, u;
    if (!e) u = s;
    else if (e == 31) u = s | 0x7f800000u | (m << 13);
    else u = s | ((e + 127 - 15) << 23) | (m << 13);
    union { float f; uint32_t u; } v; v.u = u; return v.f;
  };
  for (int m = 0; m < TM; m++) for (int n = 0; n < TM; n++) {
    float acc = 0.f;
    for (int k = 0; k < TK; k++) acc += fromHalf(hAh[(size_t)m * TK + k]) * fromHalf(hBh[(size_t)k * TM + n]);
    hrefH[(size_t)m * TM + n] = acc;
  }
  __half *dAh, *dBh; float *dCh;
  if (cudaMalloc(&dAh, hAh.size() * sizeof(uint16_t)) || cudaMalloc(&dBh, hBh.size() * sizeof(uint16_t)) || cudaMalloc(&dCh, hCh.size() * sizeof(float))) return 2;
  cudaMemcpy(dAh, hAh.data(), hAh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dBh, hBh.data(), hBh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  hgemm16<<<1, 32>>>(dAh, dBh, dCh);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(hCh.data(), dCh, hCh.size() * sizeof(float), cudaMemcpyDeviceToHost);
  bad = relBad(hCh.data(), hrefH.data(), TM * TM, 2e-3f);
  cudaFree(dAh); cudaFree(dBh); cudaFree(dCh);
  if (bad) { printf("hgemm16: %d / %d mismatch\n", bad, TM * TM); return 1; }
  printf("hgemm16: %d ok (check+emit tensor-op 16x16x32)\n", TM * TM);

  std::vector<float> hBias(TM), hrefHb((size_t)TM * TM);
  for (int n = 0; n < TM; n++) hBias[n] = 0.1f * (float)(n - 3);
  for (int m = 0; m < TM; m++) for (int n = 0; n < TM; n++)
    hrefHb[(size_t)m * TM + n] = hrefH[(size_t)m * TM + n] + hBias[n];
  float *dBias;
  if (cudaMalloc(&dAh, hAh.size() * sizeof(uint16_t)) || cudaMalloc(&dBh, hBh.size() * sizeof(uint16_t))
      || cudaMalloc(&dBias, hBias.size() * sizeof(float)) || cudaMalloc(&dCh, hCh.size() * sizeof(float))) return 2;
  cudaMemcpy(dAh, hAh.data(), hAh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dBh, hBh.data(), hBh.size() * sizeof(uint16_t), cudaMemcpyHostToDevice);
  cudaMemcpy(dBias, hBias.data(), hBias.size() * sizeof(float), cudaMemcpyHostToDevice);
  hgemm16b<<<1, 32>>>(dAh, dBh, dBias, dCh);
  if (cudaDeviceSynchronize() != cudaSuccess) return 3;
  cudaMemcpy(hCh.data(), dCh, hCh.size() * sizeof(float), cudaMemcpyDeviceToHost);
  bad = relBad(hCh.data(), hrefHb.data(), TM * TM, 2e-3f);
  cudaFree(dAh); cudaFree(dBh); cudaFree(dBias); cudaFree(dCh);
  if (bad) { printf("hgemm16b: %d / %d mismatch\n", bad, TM * TM); return 1; }
  printf("hgemm16b: %d ok (check+emit tensor-op 16x16x32 +bias, gemm_ptx ABI)\n", TM * TM);
  return 0;
}
