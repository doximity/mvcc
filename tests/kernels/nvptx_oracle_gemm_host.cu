// NVIDIA-only host: load recovered gemm_ptx PTX and compare to CPU.
#ifdef MVCC_ORACLE_HOST
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e)); return 1; } } while (0)
#define CU(x) do { CUresult e = (x); if (e != CUDA_SUCCESS) { const char* s = nullptr; \
  cuGetErrorString(e, &s); fprintf(stderr, "%s: %s\n", #x, s ? s : "?"); return 1; } } while (0)

static uint16_t f16_from_f(float f) {
  __half h = __float2half(f); uint16_t r; memcpy(&r, &h, 2); return r;
}
static float f16_to_f(uint16_t b) {
  __half h; memcpy(&h, &b, 2); return __half2float(h);
}
static uint16_t bf16_from_f(float f) {
  __nv_bfloat16 h = __float2bfloat16(f); uint16_t r; memcpy(&r, &h, 2); return r;
}
static float bf16_to_f(uint16_t b) {
  __nv_bfloat16 h; memcpy(&h, &b, 2); return __bfloat162float(h);
}

constexpr int kLd = 40;

struct Case {
  const char* name;
  const char* tag;
  int M, N, K, Bm, Bn, stages, Wm, Wn;
  int bf16;
};

static int run_one(CUmodule mod, const Case& c, int seed) {
  const int kThreads = 32 * (c.Bm / c.Wm) * (c.Bn / c.Wn);
  const size_t kern_smem = (size_t)c.stages * (c.Bm + c.Bn) * kLd * 2;
  const size_t smem = kern_smem < 98304 ? 98304 : kern_smem;
  std::vector<uint16_t> ha((size_t)c.M * c.K), hb((size_t)c.N * c.K), hbias(c.N), hc((size_t)c.M * c.N);
  srand(seed);
  auto pack = c.bf16 ? bf16_from_f : f16_from_f;
  auto unpack = c.bf16 ? bf16_to_f : f16_to_f;
  for (auto& x : ha) x = pack((float)((rand() % 9) - 4) * 0.25f);
  for (auto& x : hb) x = pack((float)((rand() % 9) - 4) * 0.25f);
  for (auto& x : hbias) x = pack((float)((rand() % 9) - 4) * 0.25f);

  uint16_t *dA, *dB, *dBias, *dC;
  CK(cudaMalloc(&dA, ha.size() * 2)); CK(cudaMalloc(&dB, hb.size() * 2));
  CK(cudaMalloc(&dBias, hbias.size() * 2)); CK(cudaMalloc(&dC, hc.size() * 2));
  CK(cudaMemcpy(dA, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dBias, hbias.data(), hbias.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemset(dC, 0, hc.size() * 2));

  CUfunction fn;
  CUresult gr = cuModuleGetFunction(&fn, mod, c.name);
  if (gr != CUDA_SUCCESS) {
    const char* s = nullptr; cuGetErrorString(gr, &s);
    fprintf(stderr, "missing %s: %s\n", c.tag, s ? s : "?");
    cudaFree(dA); cudaFree(dB); cudaFree(dBias); cudaFree(dC);
    return 1;
  }
  CU(cuFuncSetAttribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)smem));
  int m = c.M, n = c.N, k = c.K;
  void* args[] = { &dA, &dB, &dBias, &dC, &m, &n, &k };
  dim3 grid((c.N + c.Bn - 1) / c.Bn, (c.M + c.Bm - 1) / c.Bm);
  CU(cuLaunchKernel(fn, grid.x, grid.y, 1, kThreads, 1, 1, (unsigned)smem, 0, args, nullptr));
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hc.data(), dC, hc.size() * 2, cudaMemcpyDeviceToHost));

  int bad = 0;
  for (int r = 0; r < c.M; r++)
    for (int col = 0; col < c.N; col++) {
      double s = 0;
      for (int kk = 0; kk < c.K; kk++)
        s += (double)unpack(ha[(size_t)r * c.K + kk]) * (double)unpack(hb[(size_t)col * c.K + kk]);
      s += (double)unpack(hbias[col]);
      float ref = unpack(pack((float)s));
      float got = unpack(hc[(size_t)r * c.N + col]);
      if (got != ref) {
        if (bad < 4) fprintf(stderr, "  mismatch %s (%d,%d) got %g want %g\n", c.tag, r, col, got, ref);
        bad++;
      }
    }
  printf("nvptx-oracle gemm_ptx %s: %s (%d bad)\n", c.tag, bad ? "FAIL" : "ok", bad);
  CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dBias)); CK(cudaFree(dC));
  return bad ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc < 2) { fprintf(stderr, "usage: %s recovered.ptx [all|exact|recovered|direct]\n", argv[0]); return 2; }
  CK(cudaSetDevice(0));
  CU(cuInit(0));
  FILE* pf = fopen(argv[1], "rb");
  if (!pf) { fprintf(stderr, "open %s\n", argv[1]); return 1; }
  fseek(pf, 0, SEEK_END); long psz = ftell(pf); rewind(pf);
  std::vector<char> ptx((size_t)psz + 1, 0);
  size_t nread = fread(ptx.data(), 1, (size_t)psz, pf); fclose(pf);
  if (nread != (size_t)psz) { fprintf(stderr, "short PTX read\n"); return 1; }
  char errlog[8192] = {}, infolog[2048] = {};
  CUjit_option opts[] = { CU_JIT_ERROR_LOG_BUFFER, CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES, CU_JIT_INFO_LOG_BUFFER, CU_JIT_INFO_LOG_BUFFER_SIZE_BYTES };
  void* optv[] = { errlog, (void*)(uintptr_t)sizeof(errlog), infolog, (void*)(uintptr_t)sizeof(infolog) };
  CUmodule mod;
  CUresult lr = cuModuleLoadDataEx(&mod, ptx.data(), 4, opts, optv);
  if (lr != CUDA_SUCCESS) {
    const char* s = nullptr; cuGetErrorString(lr, &s);
    fprintf(stderr, "cuModuleLoadDataEx: %s\nJIT error:\n%s\n", s ? s : "?", errlog);
    return 1;
  }

  const char* mode = argc > 2 ? argv[2] : "all";
  const char* rec = "_Z14gemm_tc_kernelILi64ELi64ELi2ELi32ELi32E6__halfLb1ELb0ELb0EEvPKT4_S3_S3_PS1_iii";
  const char* recd = "_Z14gemm_tc_kernelILi64ELi64ELi2ELi32ELi32E6__halfLb0ELb0ELb0EEvPKT4_S3_S3_PS1_iii";
  const char* exact = "_Z14gemm_tc_kernelILi64ELi64ELi2ELi32ELi32E6__halfLb1ELb0ELb0EEvPKT4_S3_S3_PS1_iii__mvcc_exact";
  if (!strcmp(mode, "exact")) {
    Case c{exact, "64x64x64 f16 reuse exact+bias", 64, 64, 64, 64, 64, 2, 32, 32, 0};
    return run_one(mod, c, 1);
  }
  if (!strcmp(mode, "recovered")) {
    Case c{rec, "64x64x64 f16 reuse recovered", 64, 64, 64, 64, 64, 2, 32, 32, 0};
    return run_one(mod, c, 1);
  }
  if (!strcmp(mode, "direct")) {
    Case c{recd, "64x64x64 f16 direct recovered", 64, 64, 64, 64, 64, 2, 32, 32, 0};
    return run_one(mod, c, 1);
  }

  Case cases[] = {
    {rec, "f16 reuse 32x32", 64, 64, 64, 64, 64, 2, 32, 32, 0},
    {rec, "f16 reuse 32x32 edge 100x72x128", 100, 72, 128, 64, 64, 2, 32, 32, 0},
    {recd, "f16 direct 32x32", 64, 64, 64, 64, 64, 2, 32, 32, 0},
    {exact, "f16 reuse exact-twin", 64, 64, 64, 64, 64, 2, 32, 32, 0},
    {"_Z14gemm_tc_kernelILi64ELi64ELi2ELi32ELi32E13__nv_bfloat16Lb1ELb0ELb0EEvPKT4_S3_S3_PS1_iii",
     "bf16 reuse 32x32", 64, 64, 64, 64, 64, 2, 32, 32, 1},
    {"_Z14gemm_tc_kernelILi64ELi64ELi2ELi32ELi32E6__halfLb1ELb1ELb0EEvPKT4_S3_S3_PS1_iii",
     "f16 reuse x2 32x32", 64, 64, 64, 64, 64, 2, 32, 32, 0},
    {"_Z14gemm_tc_kernelILi64ELi64ELi2ELi32ELi16E6__halfLb0ELb0ELb1EEvPKT4_S3_S3_PS1_iii",
     "f16 direct regb 32x16", 64, 64, 64, 64, 64, 2, 32, 16, 0},
    {"_Z14gemm_tc_kernelILi32ELi32ELi2ELi32ELi8E6__halfLb1ELb0ELb1EEvPKT4_S3_S3_PS1_iii",
     "f16 reuse regb 32x8", 64, 64, 64, 32, 32, 2, 32, 8, 0},
    {"_Z14gemm_tc_kernelILi64ELi64ELi3ELi32ELi32E6__halfLb1ELb0ELb0EEvPKT4_S3_S3_PS1_iii",
     "f16 reuse s3 32x32", 128, 128, 64, 64, 64, 3, 32, 32, 0},
    {"_Z14gemm_tc_kernelILi128ELi64ELi2ELi64ELi32E6__halfLb1ELb0ELb0EEvPKT4_S3_S3_PS1_iii",
     "f16 reuse 64x32", 128, 64, 64, 128, 64, 2, 64, 32, 0},
    {"_Z14gemm_tc_kernelILi64ELi64ELi2ELi32ELi32E13__nv_bfloat16Lb0ELb0ELb0EEvPKT4_S3_S3_PS1_iii",
     "bf16 direct 32x32", 64, 64, 64, 64, 64, 2, 32, 32, 1},
    {"_Z14gemm_tc_kernelILi32ELi32ELi2ELi32ELi8E6__halfLb0ELb0ELb1EEvPKT4_S3_S3_PS1_iii",
     "f16 direct regb 32x8", 64, 64, 64, 32, 32, 2, 32, 8, 0},
  };
  int fails = 0;
  for (auto& c : cases) fails += run_one(mod, c, 1);
  printf("nvptx-oracle gemm_ptx recovered suite: %s (%d failed)\n", fails ? "FAIL" : "ok", fails);
  return fails ? 1 : 0;
}
#else
int main() { printf("nvptx_oracle_gemm_host: NVIDIA host only\n"); return 0; }
#endif
