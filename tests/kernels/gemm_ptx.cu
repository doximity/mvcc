// Tensor-core GEMM: cp.async -> XOR-swizzled shared-memory ring -> ldmatrix.x4
// -> mma.sync.m16n8k16 (f16 or bf16, f32 accumulate), double-buffered register fragments, fused bias epilogue.
// C[M,N] = A[M,K] . B[N,K]^T + bias.   A, B row-major with K contiguous ("row.col" mma).
//
// Corpus kernel for tensor recovery (README.md): exact PTX twin plus recovered TensorOps.
// The host runs the same kernel against a CPU reference.
//
// Two epilogues:
//   REUSE  - stage the fp16 C tile through the idle pipeline shared memory, then 16-byte row stores;
//   DIRECT - each lane stores its accumulator pairs straight from the mma fragment layout.
// Three operand idioms: ldmatrix.x4 for A and for pairs of n8 B tiles; X2 - as attention and decode kernels write
// it - the A fragment assembled from four 32-bit shared loads per lane and one ldmatrix.x2 per n8 B tile; REGB - the
// fp8 decode idiom - both fragments assembled from 32-bit shared loads, one n8 B tile per warp (kWn = 8), so the B block
// has no n8 neighbor and recovery zero-pads it to 16 columns.
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>

#define CK(x)                                                                                              \
  do {                                                                                                     \
    cudaError_t e_ = (x);                                                                                  \
    if (e_ != cudaSuccess) { fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #x, cudaGetErrorString(e_)); exit(1); } \
  } while (0)

// ---------------------------------------------------------------- PTX wrappers (m16n8k16 / ldmatrix.x4)
__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }
__device__ __forceinline__ void cp_async_16(uint32_t dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(src));
}
__device__ __forceinline__ void cp_async_16_zfill(uint32_t dst, const void* src, int src_bytes) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n"); }
template <int N> __device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
template <typename T> struct mma_t;
template <> struct mma_t<__half> {
  static __device__ __forceinline__ void run(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
};
template <> struct mma_t<__nv_bfloat16> {
  static __device__ __forceinline__ void run(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
};

// ---------------------------------------------------------------- kernel
constexpr int kBk = 32;        // K slab per stage (two m16n8k16 steps)
constexpr int kLd = kBk + 8;   // padded row: 40 halves = 80 B (odd number of 16-byte groups)

// XOR swizzle of the 16-byte chunk index within a padded row; keeps ldmatrix row reads conflict-free.
__device__ __forceinline__ int smem_off(int row, int col) { return row * kLd + (((col >> 3) ^ (row & 3)) << 3) + (col & 7); }

template <int kBm, int kBn, int kStages, int kWm, int kWn, typename T, bool REUSE, bool X2 = false, bool REGB = false>
__global__ void __launch_bounds__(32 * (kBm / kWm) * (kBn / kWn))
gemm_tc_kernel(const T* __restrict__ A, const T* __restrict__ B, const T* __restrict__ bias, T* __restrict__ C, int M, int N, int K) {
  constexpr int kThreads = 32 * (kBm / kWm) * (kBn / kWn);
  constexpr int kWarpsN = kBn / kWn;
  constexpr int kMt = kWm / 16, kNt = kWn / 8;
  static_assert(kNt % 2 == 0 || REGB, "B fragments load pairwise");
  constexpr int kRowsPerPass = kThreads / 4;  // 4 x 16 B chunks per 64 B row slab
  static_assert(kBm % kRowsPerPass == 0 && kBn % kRowsPerPass == 0, "tile rows must fill staging passes");
  constexpr int kCLd = kBn + 8;

  extern __shared__ __align__(16) unsigned char smem_raw[];
  T* a_s = reinterpret_cast<T*>(smem_raw);            // [kStages][kBm * kLd]
  T* b_s = a_s + kStages * kBm * kLd;                  // [kStages][kBn * kLd]

  const int block_n0 = blockIdx.x * kBn, block_m0 = blockIdx.y * kBm;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int warp_m0 = (warp / kWarpsN) * kWm, warp_n0 = (warp % kWarpsN) * kWn;
  const int k_tiles = K / kBk;

  auto stage_tile = [&](int stage, int kt) {
    const int k0 = kt * kBk;
#pragma unroll
    for (int pass = 0; pass < kBm / kRowsPerPass; ++pass) {
      const int linear = pass * kThreads + tid, row = linear >> 2, chunk = (linear & 3) << 3;
      const int grow = block_m0 + row;
      const uint32_t dst = smem_u32(&a_s[stage * kBm * kLd + smem_off(row, chunk)]);
      if (grow < M) cp_async_16(dst, A + (size_t)grow * K + k0 + chunk);
      else cp_async_16_zfill(dst, A, 0);
    }
#pragma unroll
    for (int pass = 0; pass < kBn / kRowsPerPass; ++pass) {
      const int linear = pass * kThreads + tid, row = linear >> 2, chunk = (linear & 3) << 3;
      const int gcol = block_n0 + row;
      const uint32_t dst = smem_u32(&b_s[stage * kBn * kLd + smem_off(row, chunk)]);
      if (gcol < N) cp_async_16(dst, B + (size_t)gcol * K + k0 + chunk);
      else cp_async_16_zfill(dst, B, 0);
    }
    cp_async_commit();
  };

  float acc[kMt][kNt][4] = {};

#pragma unroll
  for (int s = 0; s < kStages - 1; ++s) {
    if (s < k_tiles) stage_tile(s, s); else cp_async_commit();
  }

  // ldmatrix lane addressing (PTX: lanes 8i..8i+7 address the rows of 8x8 matrix i).
  // A x4: matrices = (m0-7,k0-7) (m8-15,k0-7) (m0-7,k8-15) (m8-15,k8-15) -> lane l: row l&15, k-half l>>4.
  // B x4 (two adjacent n8 tiles): (n0-7,k0-7) (n0-7,k8-15) (n8-15,k0-7) (n8-15,k8-15) -> row (l&7)+((l>>4)<<3), k-half (l>>3)&1.
  const int a_row = lane & 15, a_kh = (lane >> 4) << 3;
  const int b_row = (lane & 7) + ((lane >> 4) << 3), b_kh = ((lane >> 3) & 1) << 3;
  // X2: register-built A (lane 4g+t: a0 = A[g][2t..2t+1], a1 = A[g+8][..], a2 = A[g][2t+8..], a3 = A[g+8][2t+8..]);
  // B x2 per n8 tile: matrices (n0-7,k0-7) (n0-7,k8-15) -> lane l addresses row l&7, k-half (l>>3)&1 (bit 4 unused).
  const int a_g = lane >> 2, a_t2 = (lane & 3) << 1;
  const int b2_row = lane & 7, b2_kh = ((lane >> 3) & 1) << 3;

  uint32_t a_frag[2][kMt][4];
  uint32_t b_frag[2][kNt][2];
  auto load_frags = [&](int buf, int stage, int kk) {
    if (X2 || REGB) {
      const T* a_base = a_s + stage * kBm * kLd;
#pragma unroll
      for (int mt = 0; mt < kMt; ++mt) {
        const int row = warp_m0 + mt * 16 + a_g;
        a_frag[buf][mt][0] = *reinterpret_cast<const uint32_t*>(a_base + smem_off(row, kk + a_t2));
        a_frag[buf][mt][1] = *reinterpret_cast<const uint32_t*>(a_base + smem_off(row + 8, kk + a_t2));
        a_frag[buf][mt][2] = *reinterpret_cast<const uint32_t*>(a_base + smem_off(row, kk + 8 + a_t2));
        a_frag[buf][mt][3] = *reinterpret_cast<const uint32_t*>(a_base + smem_off(row + 8, kk + 8 + a_t2));
      }
      if (REGB) {
        // register-built B (lane 4g+t: b0 = B[n=g][k=2t..2t+1], b1 = B[g][2t+8..2t+9])
        const T* b_base = b_s + stage * kBn * kLd;
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) {
          b_frag[buf][nt][0] = *reinterpret_cast<const uint32_t*>(b_base + smem_off(warp_n0 + nt * 8 + a_g, kk + a_t2));
          b_frag[buf][nt][1] = *reinterpret_cast<const uint32_t*>(b_base + smem_off(warp_n0 + nt * 8 + a_g, kk + 8 + a_t2));
        }
        return;
      }
#pragma unroll
      for (int nt = 0; nt < kNt; ++nt)
        ldmatrix_x2(b_frag[buf][nt], smem_u32(&b_s[stage * kBn * kLd + smem_off(warp_n0 + nt * 8 + b2_row, kk + b2_kh)]));
      return;
    }
#pragma unroll
    for (int mt = 0; mt < kMt; ++mt)
      ldmatrix_x4(a_frag[buf][mt], smem_u32(&a_s[stage * kBm * kLd + smem_off(warp_m0 + mt * 16 + a_row, kk + a_kh)]));
#pragma unroll
    for (int np = 0; np < kNt / 2; ++np) {
      uint32_t pair[4];
      ldmatrix_x4(pair, smem_u32(&b_s[stage * kBn * kLd + smem_off(warp_n0 + np * 16 + b_row, kk + b_kh)]));
      b_frag[buf][2 * np][0] = pair[0]; b_frag[buf][2 * np][1] = pair[1];
      b_frag[buf][2 * np + 1][0] = pair[2]; b_frag[buf][2 * np + 1][1] = pair[3];
    }
  };

  cp_async_wait<kStages - 2>();
  __syncthreads();
  load_frags(0, 0, 0);

  for (int kt = 0; kt < k_tiles; ++kt) {
    const int stage = kt % kStages;
#pragma unroll
    for (int kk_step = 0; kk_step < kBk / 16; ++kk_step) {
      const int buf = kk_step & 1;
      if (kk_step + 1 < kBk / 16) {
        load_frags(buf ^ 1, stage, (kk_step + 1) * 16);
      } else {
        const int produce = kt + kStages - 1;
        if (produce < k_tiles) stage_tile(produce % kStages, produce); else cp_async_commit();
        cp_async_wait<kStages - 2>();
        __syncthreads();
        if (kt + 1 < k_tiles) load_frags(buf ^ 1, (kt + 1) % kStages, 0);
      }
#pragma unroll
      for (int mt = 0; mt < kMt; ++mt)
#pragma unroll
        for (int nt = 0; nt < kNt; ++nt) mma_t<T>::run(acc[mt][nt], a_frag[buf][mt], b_frag[buf][nt]);
    }
  }

  // Epilogue. Fragment layout: lane holds D[g][2t..2t+1] (c0,c1) and D[g+8][2t..2t+1] (c2,c3), g = lane>>2, t = lane&3.
  const int d_row = lane >> 2, d_col = (lane & 3) << 1;
  if (REUSE) {
    __syncthreads();
    T* c_s = reinterpret_cast<T*>(smem_raw);  // [kBm][kCLd]
#pragma unroll
    for (int mt = 0; mt < kMt; ++mt)
#pragma unroll
      for (int nt = 0; nt < kNt; ++nt)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int r = warp_m0 + mt * 16 + d_row + half * 8, c = warp_n0 + nt * 8 + d_col;
          float v0 = acc[mt][nt][half * 2], v1 = acc[mt][nt][half * 2 + 1];
          if (bias) {
            const int n = block_n0 + c;
            if (n < N) v0 += (float)bias[n];
            if (n + 1 < N) v1 += (float)bias[n + 1];
          }
          c_s[r * kCLd + c] = (T)v0;
          c_s[r * kCLd + c + 1] = (T)v1;
        }
    __syncthreads();
    constexpr int kChunksPerRow = kBn / 8, kChunks = kBm * kChunksPerRow;
    for (int i = tid; i < kChunks; i += kThreads) {
      const int r = i / kChunksPerRow, c = (i % kChunksPerRow) * 8;
      const int m = block_m0 + r, n = block_n0 + c;
      if (m >= M || n >= N) continue;
      T* dst = C + (size_t)m * N + n;
      const T* src = c_s + r * kCLd + c;
      if (n + 8 <= N && (N & 7) == 0) *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
      else for (int j = 0; j < 8 && n + j < N; ++j) dst[j] = src[j];
    }
  } else {
#pragma unroll
    for (int mt = 0; mt < kMt; ++mt)
#pragma unroll
      for (int nt = 0; nt < kNt; ++nt)
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int m = block_m0 + warp_m0 + mt * 16 + d_row + half * 8;
          const int n = block_n0 + warp_n0 + nt * 8 + d_col;
          if (m >= M) continue;
          float v0 = acc[mt][nt][half * 2], v1 = acc[mt][nt][half * 2 + 1];
          if (bias) { if (n < N) v0 += (float)bias[n]; if (n + 1 < N) v1 += (float)bias[n + 1]; }
          if (n < N) C[(size_t)m * N + n] = (T)v0;
          if (n + 1 < N) C[(size_t)m * N + n + 1] = (T)v1;
        }
  }
}

// ---------------------------------------------------------------- host
template <typename T> struct conv;
template <> struct conv<__nv_bfloat16> {
  static uint16_t from_float(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fffu + ((u >> 16) & 1u); return (uint16_t)(u >> 16); }
  static float to_float(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
  static const char* name() { return "bf16"; }
};
template <> struct conv<__half> {
  static uint16_t from_float(float f) { __half h = __float2half(f); uint16_t r; memcpy(&r, &h, 2); return r; }
  static float to_float(uint16_t b) { __half h; memcpy(&h, &b, 2); return __half2float(h); }
  static const char* name() { return "f16"; }
};

static int failures = 0;

template <int kBm, int kBn, int kStages, int kWm, int kWn, typename T, bool REUSE, bool X2 = false, bool REGB = false>
static void run_case(int M, int N, int K, bool exact, bool with_bias, uint32_t seed, int reps) {
  constexpr int kThreads = 32 * (kBm / kWm) * (kBn / kWn);
  const size_t smem = (size_t)kStages * (kBm + kBn) * kLd * sizeof(T);
  std::vector<uint16_t> ha((size_t)M * K), hb((size_t)N * K), hbias(N), hc((size_t)M * N);
  srand(seed);
  auto gen = [&]() -> float {
    if (exact) return (float)((rand() % 9) - 4) * 0.25f;
    return (float)rand() / (float)RAND_MAX * 2.f - 1.f;
  };
  for (auto& x : ha) x = conv<T>::from_float(gen());
  for (auto& x : hb) x = conv<T>::from_float(gen());
  for (auto& x : hbias) x = conv<T>::from_float(gen());
  T *dA, *dB, *dBias, *dC;
  CK(cudaMalloc(&dA, ha.size() * 2)); CK(cudaMalloc(&dB, hb.size() * 2)); CK(cudaMalloc(&dBias, hbias.size() * 2)); CK(cudaMalloc(&dC, hc.size() * 2));
  CK(cudaMemcpy(dA, ha.data(), ha.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dB, hb.data(), hb.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dBias, hbias.data(), hbias.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemset(dC, 0, hc.size() * 2));
  auto kern = gemm_tc_kernel<kBm, kBn, kStages, kWm, kWn, T, REUSE, X2, REGB>;
  CK(cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem));
  dim3 grid((N + kBn - 1) / kBn, (M + kBm - 1) / kBm);
  kern<<<grid, kThreads, smem>>>(dA, dB, with_bias ? dBias : nullptr, dC, M, N, K);
  CK(cudaGetLastError());
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(hc.data(), dC, hc.size() * 2, cudaMemcpyDeviceToHost));

  // CPU reference (fp32 accumulate in k order; the GPU may reassociate, hence the tolerance for inexact inputs).
  int bad = 0; double max_rel = 0;
  for (int m = 0; m < M; m++)
    for (int n = 0; n < N; n++) {
      double s = 0;
      for (int k = 0; k < K; k++) s += (double)conv<T>::to_float(ha[(size_t)m * K + k]) * conv<T>::to_float(hb[(size_t)n * K + k]);
      if (with_bias) s += conv<T>::to_float(hbias[n]);
      float ref = conv<T>::to_float(conv<T>::from_float((float)s));
      float got = conv<T>::to_float(hc[(size_t)m * N + n]);
      bool ok;
      if (exact) ok = got == ref;
      else { double rel = fabs(got - ref) / fmax(fabs(ref), 1.0); if (rel > max_rel) max_rel = rel; ok = rel <= (sizeof(T) == 2 && conv<T>::name()[0] == 'b' ? 2e-2 : 4e-3); }
      if (!ok && bad++ < 5) fprintf(stderr, "    mismatch at (%d,%d): got %g want %g\n", m, n, got, ref);
    }
  double ms = 0;
  if (reps > 0) {
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    for (int i = 0; i < 3; i++) kern<<<grid, kThreads, smem>>>(dA, dB, with_bias ? dBias : nullptr, dC, M, N, K);
    CK(cudaEventRecord(e0));
    for (int i = 0; i < reps; i++) kern<<<grid, kThreads, smem>>>(dA, dB, with_bias ? dBias : nullptr, dC, M, N, K);
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float t; CK(cudaEventElapsedTime(&t, e0, e1)); ms = t / reps;
  }
  printf("  %-4s %dx%dx%d tile %dx%d/%dx%d s%d %-6s %s%s%s: %s", conv<T>::name(), M, N, K, kBm, kBn, kWm, kWn, kStages, REUSE ? "reuse" : "direct",
         exact ? "exact" : "real ", with_bias ? "+bias" : "     ", REGB ? " regb" : X2 ? " x2" : "", bad ? "FAIL" : "ok");
  if (!exact) printf(" (max rel %.2e)", max_rel);
  if (reps > 0) printf("  %.3f ms  %.1f TFLOP/s", ms, 2.0 * M * N * K / (ms * 1e-3) / 1e12);
  printf("\n");
  if (bad) failures++;
  CK(cudaFree(dA)); CK(cudaFree(dB)); CK(cudaFree(dBias)); CK(cudaFree(dC));
}

int main(int argc, char** argv) {
  const bool bench = argc > 1 && !strcmp(argv[1], "--bench");
  if (argc > 1 && !strcmp(argv[1], "--bench-64")) {  // one shape, for lowering experiments
    run_case<64, 64, 2, 32, 32, __half, true>(2048, 2048, 2048, false, true, 10, 20);
    return failures ? 1 : 0;
  }
  printf("gemm_ptx: cp.async / ldmatrix / mma.sync GEMM vs CPU\n");
  // Correctness: small shapes with edges (M, N not multiples of the tile), both epilogues, both types.
  run_case<64, 64, 2, 32, 32, __half, true>(64, 64, 64, true, true, 1, 0);
  run_case<64, 64, 2, 32, 32, __half, false>(64, 64, 64, true, true, 2, 0);
  run_case<64, 64, 2, 32, 32, __half, true>(100, 72, 128, true, true, 3, 0);
  run_case<64, 64, 2, 32, 32, __half, false>(100, 72, 128, true, false, 4, 0);
  run_case<64, 64, 3, 32, 32, __half, true>(128, 128, 256, false, true, 5, 0);
  run_case<64, 64, 2, 32, 32, __nv_bfloat16, true>(64, 64, 64, true, true, 6, 0);
  run_case<64, 64, 2, 32, 32, __nv_bfloat16, false>(100, 72, 128, false, true, 7, 0);
  run_case<128, 64, 2, 64, 32, __half, true>(256, 128, 256, false, true, 8, 0);
  run_case<128, 64, 2, 64, 32, __nv_bfloat16, false>(256, 128, 256, false, false, 9, 0);
  // register-built A + ldmatrix.x2 B (the attention/decode idiom): recovered through the x2 fusion and the
  // register-built direct fills
  run_case<64, 64, 2, 32, 32, __half, true, true>(64, 64, 64, true, true, 15, 0);
  run_case<64, 64, 2, 32, 32, __half, false, true>(100, 72, 128, true, true, 16, 0);
  run_case<64, 64, 3, 32, 32, __nv_bfloat16, false, true>(128, 128, 256, false, true, 17, 0);
  run_case<128, 64, 2, 64, 32, __nv_bfloat16, true, true>(256, 128, 256, false, false, 18, 0);
  // register-built A and B (the fp8 decode idiom): n8 tile pairs, then one n8 tile per warp, whose B block is
  // zero-padded to 16 columns
  run_case<64, 64, 2, 32, 16, __half, false, false, true>(64, 64, 64, true, true, 23, 0);
  run_case<32, 32, 2, 32, 8, __half, true, false, true>(64, 64, 64, true, true, 20, 0);
  run_case<32, 32, 2, 32, 8, __half, false, false, true>(100, 72, 128, true, true, 21, 0);
  run_case<32, 32, 3, 32, 8, __nv_bfloat16, false, false, true>(128, 128, 256, false, true, 22, 0);
  if (bench) {
    printf("bench (M=N=K=2048):\n");
    run_case<64, 64, 2, 32, 32, __half, true>(2048, 2048, 2048, false, true, 10, 10);
    run_case<64, 64, 2, 32, 32, __half, false>(2048, 2048, 2048, false, true, 11, 10);
    run_case<128, 64, 2, 64, 32, __half, true>(2048, 2048, 2048, false, true, 12, 10);
    run_case<128, 64, 2, 64, 32, __half, false>(2048, 2048, 2048, false, true, 13, 10);
    run_case<128, 64, 2, 64, 32, __nv_bfloat16, false>(2048, 2048, 2048, false, true, 14, 10);
    run_case<64, 64, 2, 32, 32, __half, true, true>(2048, 2048, 2048, false, true, 19, 10);
  }
  printf("%d failure%s\n", failures, failures == 1 ? "" : "s");
  return failures ? 1 : 0;
}
