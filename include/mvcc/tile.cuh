// mvcc: portable warp-level MMA tile (mvcc::warp_tile).
//
// One header, two implementations of the same API:
//   * NVIDIA (nvcc / clang CUDA): mma.sync m16n8k16 with fragments loaded straight from memory.
//   * mvcc (__MVCC__): lowered by mvcc-ir2msl to Metal 4 TensorOps (mpp::tensor_ops::matmul2d) with the
//     accumulator resident in a cooperative tensor. Operands stream from device or threadgroup memory.
//
// Both targets:
//
//   mvcc::warp_tile<M, N, K, T> tile;      // T = __nv_bfloat16 or __half; M,N multiples of 16 / 8, K multiple of 16
//   tile.zero();
//   for (int k = 0; k < Ktot; k += K) tile.mma(A + k, lda, B + k, ldb);   // C += A[M][K] · B[N][K]^T  ("row.col")
//   tile.foreach_c([&](int row, int col, float& v) { out[row * ldo + col] = v; });
//
// All 32 lanes of a warp must call every method (warp-collective). A is M x K row-major with leading dimension
// lda (elements); B is N x K row-major with leading dimension ldb (weights as [out, in]). Accumulation is fp32.
// Element (row, col) ownership is implementation-defined; foreach_c hands out coordinates.
//
// mvcc tile-size guidance (measured, M5 Pro): 32x32 per warp with 4x4 warps per block (512 threads) streaming
// from device memory with K chunks of 128 reaches ~8.5 TFLOP/s bf16 on a 512x4096x2048 GEMM vs ~3 TFLOP/s for the
// mma.sync emulation. See README.md.
#ifndef MVCC_TILE_CUH
#define MVCC_TILE_CUH
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace mvcc {

namespace detail {
template <typename T> struct tile_elem;
template <> struct tile_elem<__nv_bfloat16> { static constexpr int id = 0; };
template <> struct tile_elem<__half> { static constexpr int id = 1; };
}  // namespace detail

#if defined(__MVCC__)
// ---------------------------------------------------------------- mvcc: compiler builtins (mvcc-ir2msl lowers)
extern "C" {
// acc[M*N/32] += A · B^T for one K chunk. elem: 0 = bf16, 1 = f16. M, N, K must be compile-time constants.
__device__ void __mvcc_tile_mma(float* acc, const void* a, int lda, const void* b, int ldb, int M, int N, int K, int elem);
// coordinate of this lane's i-th accumulator element: (row << 16) | col, or -1 if the slot is not a valid element
__device__ int __mvcc_tile_coord(int M, int N, int K, int elem, int i);
}

template <int M, int N, int K, typename T>
struct warp_tile {
  static_assert(M % 16 == 0 && N % 16 == 0 && K % 16 == 0, "tile dims must be multiples of 16");
  static constexpr int CAP = M * N / 32;
  float acc[CAP];

  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int i = 0; i < CAP; ++i) acc[i] = 0.f;
  }
  __device__ __forceinline__ void mma(const T* a, int lda, const T* b, int ldb) {
    __mvcc_tile_mma(acc, a, lda, b, ldb, M, N, K, detail::tile_elem<T>::id);
  }
  template <class F>
  __device__ __forceinline__ void foreach_c(F&& f) {
#pragma unroll
    for (int i = 0; i < CAP; ++i) {
      const int c = __mvcc_tile_coord(M, N, K, detail::tile_elem<T>::id, i);
      if (c >= 0) f(c >> 16, c & 0xffff, acc[i]);
    }
  }
};

#else
// ---------------------------------------------------------------- NVIDIA: mma.sync m16n8k16 (sm_80+)
namespace detail {
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], __nv_bfloat16*) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ void mma16816(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2], __half*) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
               : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
}  // namespace detail

template <int M, int N, int K, typename T>
struct warp_tile {
  static_assert(M % 16 == 0 && N % 8 == 0 && K % 16 == 0, "tile dims: M%16, N%8, K%16");
  static constexpr int MT = M / 16, NT = N / 8, CAP = M * N / 32;
  float acc[CAP];  // acc[(i*NT + j)*4 + q]: mma tile (i, j), fragment register q

  __device__ __forceinline__ void zero() {
#pragma unroll
    for (int i = 0; i < CAP; ++i) acc[i] = 0.f;
  }
  // Fragment layout (g = lane>>2, t = lane&3): a0 = A[g][2t..2t+1], a1 = A[g+8][2t..], a2 = A[g][2t+8..], a3 = A[g+8][2t+8..];
  // b0 = B[n=g][k=2t..2t+1], b1 = B[g][2t+8..]. K is a reduction index, so lanes read their 8-element runs
  // (4 x 16 B per 32-wide k step) with a fixed permutation shared by A and B.
  __device__ __forceinline__ void mma(const T* a, int lda, const T* b, int ldb) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int k0 = 0; k0 < K; k0 += 16) {
      uint32_t bf[NT][2];
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const uint2 v = *reinterpret_cast<const uint2*>(b + (int64_t)(j * 8 + g) * ldb + k0 + t * 4);
        bf[j][0] = v.x; bf[j][1] = v.y;  // k-pairs (4t, 4t+1), (4t+2, 4t+3): permuted k, same permutation for A
      }
#pragma unroll
      for (int i = 0; i < MT; ++i) {
        const uint2 r0 = *reinterpret_cast<const uint2*>(a + (int64_t)(i * 16 + g) * lda + k0 + t * 4);
        const uint2 r1 = *reinterpret_cast<const uint2*>(a + (int64_t)(i * 16 + g + 8) * lda + k0 + t * 4);
        const uint32_t af[4] = {r0.x, r1.x, r0.y, r1.y};
#pragma unroll
        for (int j = 0; j < NT; ++j) {
          float (&d)[4] = *reinterpret_cast<float (*)[4]>(acc + (i * NT + j) * 4);
          detail::mma16816(d, af, bf[j], (T*)nullptr);
        }
      }
    }
  }
  template <class F>
  __device__ __forceinline__ void foreach_c(F&& f) {
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        float* d = acc + (i * NT + j) * 4;
        f(i * 16 + g, j * 8 + 2 * t, d[0]);
        f(i * 16 + g, j * 8 + 2 * t + 1, d[1]);
        f(i * 16 + g + 8, j * 8 + 2 * t, d[2]);
        f(i * 16 + g + 8, j * 8 + 2 * t + 1, d[3]);
      }
  }
};
#endif

}  // namespace mvcc

#endif // MVCC_TILE_CUH
