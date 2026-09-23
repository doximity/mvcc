// NVPTX oracle pin: CUDA → STIR emit → llc PTX, then run that PTX on NVIDIA next to the source kernel.
// add256 / sgemm have no libdevice and no tensor-op helpers — recovered bodies are ordinary NVPTX.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>
#include <cmath>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)

__global__ void __launch_bounds__(256) add256(const float* __restrict__ a, const float* __restrict__ b, float* __restrict__ o, int n) {
  int i = blockIdx.x * 256 + threadIdx.x;
  if (i < n) o[i] = a[i] + b[i];
}

__global__ void __launch_bounds__(64) sgemm(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int M, int N, int K) {
  int o = blockIdx.x * 64 + threadIdx.x; if (o >= M * N) return;
  int m = o / N, n = o % N;
  float acc = 0.f;
  for (int k = 0; k < K; k++) acc += A[(size_t)m * K + k] * B[(size_t)k * N + n];
  C[o] = acc;
}

__global__ void __launch_bounds__(32) hgemm16(const __half* __restrict__ A, const __half* __restrict__ B, float* __restrict__ C) {
  int tid = threadIdx.x;
  for (int i = tid; i < 256; i += 32) {
    int m = i / 16, n = i % 16;
    float acc = 0.f;
    for (int k = 0; k < 32; k++) acc += __half2float(A[m * 32 + k]) * __half2float(B[k * 16 + n]);
    C[i] = acc;
  }
}

#ifdef MVCC_ORACLE_HOST
#include <cuda.h>
#define CU(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { const char* s = nullptr; \
  cuGetErrorString(e, &s); fprintf(stderr, "%s: %s\n", #x, s ? s : "?"); return 1; } } while (0)

static int loadFn(CUmodule mod, const char* a, const char* b, CUfunction* fn) {
  CUresult ge = cuModuleGetFunction(fn, mod, a);
  if (ge != CUDA_SUCCESS) ge = cuModuleGetFunction(fn, mod, b);
  if (ge != CUDA_SUCCESS) {
    const char* s = nullptr; cuGetErrorString(ge, &s);
    fprintf(stderr, "cuModuleGetFunction %s: %s\n", a, s ? s : "?");
    return 1;
  }
  return 0;
}

static int cmpBuf(const float* rec, const float* src, int n, const char* tag, float tol) {
  int bad = 0; double maxe = 0;
  for (int i = 0; i < n; i++) {
    double e = fabs((double)rec[i] - (double)src[i]);
    if (e > maxe) maxe = e;
    if (e > tol) { if (bad < 4) printf("  %s i=%d rec=%g src=%g\n", tag, i, rec[i], src[i]); bad++; }
  }
  printf("nvptx-oracle %s n=%d: %s (max err %.3g, %d bad) recovered-ptx vs NVIDIA source\n",
         tag, n, bad ? "FAIL" : "ok", maxe, bad);
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s recovered.ptx\n", argv[0]); return 2; }
  CK(cudaSetDevice(0));
  CU(cuInit(0));
  CUmodule mod; CU(cuModuleLoad(&mod, argv[1]));
  int fails = 0;

  {
    const int n = 4096;
    std::vector<float> a(n), b(n), src(n), rec(n);
    for (int i = 0; i < n; i++) { a[i] = 0.01f * (float)i; b[i] = 1.0f - 0.001f * (float)i; }
    float *da, *db, *ds, *dr;
    CK(cudaMalloc(&da, n * 4)); CK(cudaMalloc(&db, n * 4));
    CK(cudaMalloc(&ds, n * 4)); CK(cudaMalloc(&dr, n * 4));
    CK(cudaMemcpy(da, a.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, b.data(), n * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(ds, 0, n * 4)); CK(cudaMemset(dr, 0, n * 4));
    add256<<<(n + 255) / 256, 256>>>(da, db, ds, n);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(src.data(), ds, n * 4, cudaMemcpyDeviceToHost));
    CUfunction fn;
    if (loadFn(mod, "_Z6add256PKfS0_Pfi", "add256", &fn)) return 1;
    void* args[] = { &da, &db, &dr, (void*)&n };
    CU(cuLaunchKernel(fn, (n + 255) / 256, 1, 1, 256, 1, 1, 0, 0, args, nullptr));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(rec.data(), dr, n * 4, cudaMemcpyDeviceToHost));
    fails += cmpBuf(rec.data(), src.data(), n, "add256", 1e-6f);
    CK(cudaFree(da)); CK(cudaFree(db)); CK(cudaFree(ds)); CK(cudaFree(dr));
  }

  {
    const int M = 32, N = 32, K = 48, n = M * N;
    std::vector<float> A((size_t)M * K), B((size_t)K * N), src(n), rec(n);
    for (int i = 0; i < M * K; i++) A[i] = 0.01f * (float)(i - 20);
    for (int i = 0; i < K * N; i++) B[i] = 0.02f * (float)((i % 17) - 8);
    float *dA, *dB, *ds, *dr;
    CK(cudaMalloc(&dA, A.size() * 4)); CK(cudaMalloc(&dB, B.size() * 4));
    CK(cudaMalloc(&ds, n * 4)); CK(cudaMalloc(&dr, n * 4));
    CK(cudaMemcpy(dA, A.data(), A.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B.data(), B.size() * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(ds, 0, n * 4)); CK(cudaMemset(dr, 0, n * 4));
    sgemm<<<(n + 63) / 64, 64>>>(dA, dB, ds, M, N, K);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(src.data(), ds, n * 4, cudaMemcpyDeviceToHost));
    CUfunction fn;
    if (loadFn(mod, "_Z5sgemmPKfS0_Pfiii", "sgemm", &fn)) return 1;
    void* args[] = { &dA, &dB, &dr, (void*)&M, (void*)&N, (void*)&K };
    CU(cuLaunchKernel(fn, (n + 63) / 64, 1, 1, 64, 1, 1, 0, 0, args, nullptr));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(rec.data(), dr, n * 4, cudaMemcpyDeviceToHost));
    fails += cmpBuf(rec.data(), src.data(), n, "sgemm", 1e-4f);
    CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(ds)); CK(cudaFree(dr));
  }

  {
    const int n = 256;
    std::vector<uint16_t> A(16 * 32), B(32 * 16);
    std::vector<float> src(n), rec(n);
    for (int i = 0; i < 16 * 32; i++) { __half h = __float2half(0.25f * (float)((i % 9) - 4)); memcpy(&A[i], &h, 2); }
    for (int i = 0; i < 32 * 16; i++) { __half h = __float2half(0.25f * (float)((i % 7) - 3)); memcpy(&B[i], &h, 2); }
    __half *dA, *dB; float *ds, *dr;
    CK(cudaMalloc(&dA, A.size() * 2)); CK(cudaMalloc(&dB, B.size() * 2));
    CK(cudaMalloc(&ds, n * 4)); CK(cudaMalloc(&dr, n * 4));
    CK(cudaMemcpy(dA, A.data(), A.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dB, B.data(), B.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemset(ds, 0, n * 4)); CK(cudaMemset(dr, 0, n * 4));
    hgemm16<<<1, 32>>>(dA, dB, ds);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(src.data(), ds, n * 4, cudaMemcpyDeviceToHost));
    CUfunction fn;
    if (loadFn(mod, "_Z7hgemm16PK6__halfS1_Pf", "hgemm16", &fn)) return 1;
    CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 98304));
    void* args[] = { &dA, &dB, &dr };
    CU(cuLaunchKernel(fn, 1, 1, 1, 32, 1, 1, 98304, 0, args, nullptr));
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(rec.data(), dr, n * 4, cudaMemcpyDeviceToHost));
    fails += cmpBuf(rec.data(), src.data(), n, "hgemm16", 1e-3f);
    CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(ds)); CK(cudaFree(dr));
  }

  return fails ? 1 : 0;
}
#else
int main() {
  printf("nvptx_oracle_pin: compile with -DMVCC_ORACLE_HOST and recovered PTX to compare on NVIDIA\n");
  return 0;
}
#endif
