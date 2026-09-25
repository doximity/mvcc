// Minimal dependency-free CUDA kernels for Qwen3.5-MoE (hybrid GDN + gated attention, INT4 experts).
//
// Everything here is plain CUDA C++: warp shuffles, shared memory, 16-byte vector loads. No cuBLAS, no
// libraries. The kernel set is the smallest one that runs the full model for prefill (T tokens) and decode
// (T == 1):
//
//   embed_kernel               token id -> bf16 row gather
//   rmsnorm_kernel             y = rms(x) * w                      (w has the +1 folded in by convert.py)
//   residual_rmsnorm_kernel    x += d ; y = rms(x) * w
//   gemv_bf16_kernel           y[T,N] = x[T,K] . W[N,K]^T          (warp per output row, T <= 4 per pass)
//   gemm_bf16_kernel           same, smem-tiled fp32 FMA for T >= 8
//   gemm_bf16_tc_kernel        same on tensor cores (mvcc::warp_tile: mma.sync / Metal TensorOps) for T >= 32
//   gemv_i8_kernel/gemm_i8*    int8 (group-128 scales) variants of the dense projections
//   gdn_*                      conv1d + SiLU, fused q/k L2 norm + gate (alpha/beta), delta-rule recurrence, gated norm
//   attn_*                     q/k norm + RoPE + KV append, split-K online-softmax attention, sigmoid output gate
//   router_topk_kernel         softmax-normalized top-k routing
//   moe_gate_up_int4_kernel    silu(x.Wg) * (x.Wu) for the routed experts, INT4 weights
//   moe_down_int4_kernel       weighted expert down projection
//   moe_*_tile_kernel          prefill variants: pairs sorted by expert, INT4 rows dequantized to smem once per
//                              32-token tile, MMA on the tensor path (mvcc::warp_tile)
//   shared_gate_combine_kernel hidden += routed + sigmoid(x.g) * shared
//   argmax_*                   two-stage argmax over the vocabulary
//
// Conventions: activations flow as bf16 (uint16_t) except the residual stream and small per-head values,
// which are fp32. All matrices are row-major [out, in]. INT4 rows pack k=2j (low nibble) / k=2j+1 (high
// nibble) with an implicit zero point of 8 and one fp16 scale per 128 inputs.
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include <mvcc/tile.cuh>

namespace qk {

typedef uint16_t bf16;

// ------------------------------------------------------------------ scalar helpers

__device__ __forceinline__ float bf2f(bf16 b) { return __bfloat162float(__ushort_as_bfloat16(b)); }  // native cvt
__device__ __forceinline__ bf16 f2bf(float f) { return __bfloat16_as_ushort(__float2bfloat16(f)); }   // native RNE
__device__ __forceinline__ float lo_bf(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hi_bf(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }  // native cvt
__device__ __forceinline__ void bf16x8(const uint4 v, float* f) {
  f[0] = lo_bf(v.x); f[1] = hi_bf(v.x); f[2] = lo_bf(v.y); f[3] = hi_bf(v.y);
  f[4] = lo_bf(v.z); f[5] = hi_bf(v.z); f[6] = lo_bf(v.w); f[7] = hi_bf(v.w);
}
__device__ __forceinline__ float silu(float x) { return x / (1.f + __expf(-x)); }
__device__ __forceinline__ float sigmoid(float x) { return 1.f / (1.f + __expf(-x)); }
__device__ __forceinline__ float softplus(float x) { return x > 20.f ? x : log1pf(__expf(x)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}
// Block reduction for blockDim.x <= 1024 (multiple of 32). Every thread gets the total.
__device__ __forceinline__ float block_sum(float v, float* scratch /* >= 32 floats */) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  float t = lane < warps ? scratch[lane] : 0.f;
  t = warp_sum(t);
  return t;
}

// ------------------------------------------------------------------ embedding / norms

__global__ void embed_kernel(const int32_t* __restrict__ ids, const bf16* __restrict__ table, float* __restrict__ out,
                             int T, int H) {
  const int t = blockIdx.x;
  const bf16* row = table + (int64_t)ids[t] * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) out[(int64_t)t * H + i] = bf2f(row[i]);
}

// One block per row. x fp32, w bf16, y bf16.
__global__ void rmsnorm_kernel(const float* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ y, int H,
                               float eps) {
  __shared__ float sc[32];
  const float* row = x + (int64_t)blockIdx.x * H;
  float ss = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) { const float v = row[i]; ss += v * v; }
  ss = block_sum(ss, sc);
  const float inv = rsqrtf(ss / (float)H + eps);
  bf16* out = y + (int64_t)blockIdx.x * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) out[i] = f2bf(row[i] * inv * bf2f(w[i]));
}

// x += d (fp32 residual stream, d bf16 or fp32), then y = rms(x) * w.
__device__ __forceinline__ float load_delta(const float* p) { return *p; }
__device__ __forceinline__ float load_delta(const bf16* p) { return bf2f(*p); }
template <typename DT>
__global__ void residual_rmsnorm_kernel(float* __restrict__ x, const DT* __restrict__ d, const bf16* __restrict__ w,
                                        bf16* __restrict__ y, int H, float eps) {
  __shared__ float sc[32];
  float* row = x + (int64_t)blockIdx.x * H;
  const DT* drow = d + (int64_t)blockIdx.x * H;
  float ss = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    float v = row[i] + load_delta(drow + i);
    row[i] = v;
    ss += v * v;
  }
  ss = block_sum(ss, sc);
  const float inv = rsqrtf(ss / (float)H + eps);
  bf16* out = y + (int64_t)blockIdx.x * H;
  for (int i = threadIdx.x; i < H; i += blockDim.x) out[i] = f2bf(row[i] * inv * bf2f(w[i]));
}

// ------------------------------------------------------------------ dense projections

template <typename OutT> __device__ __forceinline__ void store_out(OutT* p, float v);
template <> __device__ __forceinline__ void store_out<float>(float* p, float v) { *p = v; }
template <> __device__ __forceinline__ void store_out<bf16>(bf16* p, float v) { *p = f2bf(v); }

// y[t][n] = sum_k x[t][k] * W[n][k].  One warp per n, TT tokens per pass (blockIdx.y selects the token group).
// K must be a multiple of 256. 8 warps per block.
template <int TT, typename OutT>
__global__ void __launch_bounds__(256) gemv_bf16_kernel(const bf16* __restrict__ x, const bf16* __restrict__ W,
                                                        OutT* __restrict__ y, int T, int N, int K) {
  const int lane = threadIdx.x & 31;
  const int n = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int t0 = blockIdx.y * TT;
  if (n >= N) return;
  const uint4* wrow = reinterpret_cast<const uint4*>(W + (int64_t)n * K);
  float acc[TT];
#pragma unroll
  for (int i = 0; i < TT; ++i) acc[i] = 0.f;
  const int nvec = K >> 3;  // uint4 per row
  for (int v = lane; v < nvec; v += 32) {
    float wf[8];
    bf16x8(wrow[v], wf);
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const int t = t0 + i;
      if (t < T) {
        float xf[8];
        bf16x8(reinterpret_cast<const uint4*>(x + (int64_t)t * K)[v], xf);
#pragma unroll
        for (int j = 0; j < 8; ++j) acc[i] = fmaf(wf[j], xf[j], acc[i]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < TT; ++i) {
    const float s = warp_sum(acc[i]);
    if (lane == 0 && t0 + i < T) store_out(y + (int64_t)(t0 + i) * N + n, s);
  }
}

// Shared-memory tiled GEMM for prefill: block tile 64 (N) x 32 (T), K step 32. Thread (tn, tg): tn = tid & 63
// owns column n0+tn, tg = tid >> 6 owns tokens tg*8 .. tg*8+7. Requires K % 32 == 0.
template <typename OutT>
__global__ void __launch_bounds__(256) gemm_bf16_kernel(const bf16* __restrict__ x, const bf16* __restrict__ W,
                                                        OutT* __restrict__ y, int T, int N, int K) {
  __shared__ float ws[64][33];
  __shared__ float xs[32][33];
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * 64, t0 = blockIdx.y * 32;
  const int tn = tid & 63, tg = tid >> 6;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  for (int k0 = 0; k0 < K; k0 += 32) {
    // load W tile 64x32: 2048 elems / 256 threads = 8 each (one uint4)
    {
      const int r = tid >> 2, c = (tid & 3) * 8;
      const int n = n0 + r;
      float f[8];
      if (n < N) bf16x8(*reinterpret_cast<const uint4*>(W + (int64_t)n * K + k0 + c), f);
      else {
#pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = 0.f;
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) ws[r][c + j] = f[j];
    }
    // load x tile 32x32: 1024 elems / 256 threads = 4 each
    {
      const int r = tid >> 3, c = (tid & 7) * 4;
      const int t = t0 + r;
      float f[4] = {0.f, 0.f, 0.f, 0.f};
      if (t < T) {
        const uint2 v = *reinterpret_cast<const uint2*>(x + (int64_t)t * K + k0 + c);
        f[0] = lo_bf(v.x); f[1] = hi_bf(v.x); f[2] = lo_bf(v.y); f[3] = hi_bf(v.y);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) xs[r][c + j] = f[j];
    }
    __syncthreads();
#pragma unroll 8
    for (int k = 0; k < 32; ++k) {
      const float w = ws[tn][k];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = fmaf(w, xs[tg * 8 + i][k], acc[i]);
    }
    __syncthreads();
  }
  const int n = n0 + tn;
  if (n < N) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int t = t0 + tg * 8 + i;
      if (t < T) store_out(y + (int64_t)t * N + n, acc[i]);
    }
  }
}

// INT8 weights (per-row, group-128 fp16 scales): W8 [N,K] int8, S [N, K/128] fp16. A warp owns R consecutive
// rows; the lane holds 32 consecutive k of each (two 16-byte loads per row, one scale), so 2R loads are in
// flight per lane and the warp covers 1024 k per pass. Block = 8 warps = 8R rows.
template <int TT, int R, typename OutT>
__global__ void __launch_bounds__(256) gemv_i8_kernel(const bf16* __restrict__ x, const int8_t* __restrict__ W8,
                                                      const uint16_t* __restrict__ S, OutT* __restrict__ y, int T,
                                                      int N, int K) {
  const int lane = threadIdx.x & 31;
  const int n0 = (blockIdx.x * 8 + (threadIdx.x >> 5)) * R;
  const int t0 = blockIdx.y * TT;
  if (n0 >= N) return;
  const int groups = K >> 7;
  float acc[R][TT];
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int i = 0; i < TT; ++i) acc[r][i] = 0.f;
  for (int k0 = lane * 32; k0 < K; k0 += 1024) {
    uint4 w[R][2];
    float sc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int8_t* wrow = W8 + (int64_t)(n0 + r) * K + k0;
      w[r][0] = *reinterpret_cast<const uint4*>(wrow);
      w[r][1] = *reinterpret_cast<const uint4*>(wrow + 16);
      sc[r] = h2f(S[(int64_t)(n0 + r) * groups + (k0 >> 7)]);
    }
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const int t = t0 + i;
      if (t < T) {
        const uint4* xr = reinterpret_cast<const uint4*>(x + (int64_t)t * K + k0);
        float xf[32];
#pragma unroll
        for (int q = 0; q < 4; ++q) bf16x8(xr[q], xf + q * 8);
#pragma unroll
        for (int r = 0; r < R; ++r) {
          const uint32_t ws[8] = {w[r][0].x, w[r][0].y, w[r][0].z, w[r][0].w, w[r][1].x, w[r][1].y, w[r][1].z, w[r][1].w};
          float d = 0.f;
#pragma unroll
          for (int j = 0; j < 8; ++j) {
            char4 c; __builtin_memcpy(&c, &ws[j], 4);  // 4 signed bytes -> vector convert
            d = fmaf((float)c.x, xf[j * 4 + 0], d);
            d = fmaf((float)c.y, xf[j * 4 + 1], d);
            d = fmaf((float)c.z, xf[j * 4 + 2], d);
            d = fmaf((float)c.w, xf[j * 4 + 3], d);
          }
          acc[r][i] = fmaf(d, sc[r], acc[r][i]);
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int i = 0; i < TT; ++i) {
      const float v = warp_sum(acc[r][i]);
      if (lane == 0 && t0 + i < T && n0 + r < N) store_out(y + (int64_t)(t0 + i) * N + n0 + r, v);
    }
}

// INT8 tiled GEMM (dequantizes into the smem tile).
template <typename OutT>
__global__ void __launch_bounds__(256) gemm_i8_kernel(const bf16* __restrict__ x, const int8_t* __restrict__ W8,
                                                      const uint16_t* __restrict__ S, OutT* __restrict__ y, int T,
                                                      int N, int K) {
  __shared__ float ws[64][33];
  __shared__ float xs[32][33];
  const int tid = threadIdx.x;
  const int n0 = blockIdx.x * 64, t0 = blockIdx.y * 32;
  const int tn = tid & 63, tg = tid >> 6;
  float acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  for (int k0 = 0; k0 < K; k0 += 32) {
    {
      const int r = tid >> 2, c = (tid & 3) * 8;
      const int n = n0 + r;
      float f[8];
      if (n < N) {
        const uint2 v = *reinterpret_cast<const uint2*>(W8 + (int64_t)n * K + k0 + c);
        const float sc = h2f(S[(int64_t)n * (K >> 7) + ((k0 + c) >> 7)]);
        f[0] = (float)(int8_t)(v.x & 0xff) * sc; f[1] = (float)(int8_t)((v.x >> 8) & 0xff) * sc;
        f[2] = (float)(int8_t)((v.x >> 16) & 0xff) * sc; f[3] = (float)(int8_t)(v.x >> 24) * sc;
        f[4] = (float)(int8_t)(v.y & 0xff) * sc; f[5] = (float)(int8_t)((v.y >> 8) & 0xff) * sc;
        f[6] = (float)(int8_t)((v.y >> 16) & 0xff) * sc; f[7] = (float)(int8_t)(v.y >> 24) * sc;
      } else {
#pragma unroll
        for (int j = 0; j < 8; ++j) f[j] = 0.f;
      }
#pragma unroll
      for (int j = 0; j < 8; ++j) ws[r][c + j] = f[j];
    }
    {
      const int r = tid >> 3, c = (tid & 7) * 4;
      const int t = t0 + r;
      float f[4] = {0.f, 0.f, 0.f, 0.f};
      if (t < T) {
        const uint2 v = *reinterpret_cast<const uint2*>(x + (int64_t)t * K + k0 + c);
        f[0] = lo_bf(v.x); f[1] = hi_bf(v.x); f[2] = lo_bf(v.y); f[3] = hi_bf(v.y);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) xs[r][c + j] = f[j];
    }
    __syncthreads();
#pragma unroll 8
    for (int k = 0; k < 32; ++k) {
      const float w = ws[tn][k];
#pragma unroll
      for (int i = 0; i < 8; ++i) acc[i] = fmaf(w, xs[tg * 8 + i][k], acc[i]);
    }
    __syncthreads();
  }
  const int n = n0 + tn;
  if (n < N) {
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int t = t0 + tg * 8 + i;
      if (t < T) store_out(y + (int64_t)t * N + n, acc[i]);
    }
  }
}

// Tensor-core GEMM: y[T,N] = x[T,K] . W[N,K]^T through the portable mvcc::warp_tile API (mvcc/tile.cuh):
// mma.sync m16n8k16 on NVIDIA, Metal 4 TensorOps (M5 Neural Accelerators) on mvcc. Block = WM x WN warps,
// each owning a 32x32 output tile; K streams in chunks of 128 straight from memory (no smem staging: on Apple
// GPUs the tensor path is fastest reading operands directly, and 32 KB of threadgroup memory would cap occupancy).
// Rows >= T are computed on whatever lies in the (padded) activation buffer and discarded; callers guarantee the
// buffer holds round_up(T, 32*WM) rows.
template <int WM, int WN, typename OutT>
__global__ void __launch_bounds__(32 * WM * WN) gemm_bf16_tc_kernel(const bf16* __restrict__ x, const bf16* __restrict__ W,
                                                                    OutT* __restrict__ y, int T, int N, int K) {
  constexpr int TM = 32, TN = 32, TK = 128;
  const int warp = threadIdx.x >> 5;
  const int t0 = blockIdx.x * (TM * WM) + (warp / WN) * TM;
  const int n0 = blockIdx.y * (TN * WN) + (warp % WN) * TN;
  if (n0 >= N) return;  // warp-uniform
  mvcc::warp_tile<TM, TN, TK, __nv_bfloat16> tile;
  tile.zero();
  const __nv_bfloat16* a = reinterpret_cast<const __nv_bfloat16*>(x) + (int64_t)t0 * K;
  const __nv_bfloat16* b = reinterpret_cast<const __nv_bfloat16*>(W) + (int64_t)n0 * K;
  for (int k = 0; k < K; k += TK) tile.mma(a + k, K, b + k, K);
  tile.foreach_c([&](int r, int c, float& v) {
    const int t = t0 + r, n = n0 + c;
    if (t < T && n < N) store_out(y + (int64_t)t * N + n, v);
  });
}

// INT8 GEMM on the tensor path: block = WM x WN warps (32x32 tiles each); the block's 32*WN weight rows are
// dequantized to bf16 in shared memory per K step (TK = 32), activations stream straight from memory.
template <int WM, int WN, typename OutT>
__global__ void __launch_bounds__(32 * WM * WN) gemm_i8_tc_kernel(const bf16* __restrict__ x, const int8_t* __restrict__ W8,
                                                                  const uint16_t* __restrict__ S, OutT* __restrict__ y, int T,
                                                                  int N, int K) {
  constexpr int TK = 32, NB = 32 * WN, NT = 32 * WM * WN;
  __shared__ __align__(16) bf16 Bs[NB * TK];
  const int tid = threadIdx.x, warp = tid >> 5;
  const int t0 = blockIdx.x * (32 * WM) + (warp / WN) * 32;
  const int n0 = blockIdx.y * NB;
  const int groups = K >> 7;
  mvcc::warp_tile<32, 32, TK, __nv_bfloat16> tile;
  tile.zero();
  const __nv_bfloat16* a = reinterpret_cast<const __nv_bfloat16*>(x) + (int64_t)t0 * K;
  for (int k0 = 0; k0 < K; k0 += TK) {
    __syncthreads();
    for (int c = tid; c < NB * (TK / 16); c += NT) {  // 16 int8 per chunk
      const int r = c / (TK / 16), ch = c % (TK / 16);
      const int n = n0 + r;
      const uint4 q = *reinterpret_cast<const uint4*>(W8 + (int64_t)n * K + k0 + ch * 16);
      const float sc = h2f(S[(int64_t)n * groups + (k0 >> 7)]);
      const uint32_t qw[4] = {q.x, q.y, q.z, q.w};
      uint32_t p[8];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float v0 = (float)(int8_t)(qw[j] & 0xff) * sc, v1 = (float)(int8_t)((qw[j] >> 8) & 0xff) * sc;
        const float v2 = (float)(int8_t)((qw[j] >> 16) & 0xff) * sc, v3 = (float)(int8_t)(qw[j] >> 24) * sc;
        p[2 * j] = (uint32_t)f2bf(v0) | ((uint32_t)f2bf(v1) << 16);
        p[2 * j + 1] = (uint32_t)f2bf(v2) | ((uint32_t)f2bf(v3) << 16);
      }
      uint4* o = reinterpret_cast<uint4*>(Bs + r * TK + ch * 16);
      o[0] = make_uint4(p[0], p[1], p[2], p[3]);
      o[1] = make_uint4(p[4], p[5], p[6], p[7]);
    }
    __syncthreads();
    tile.mma(a + k0, K, reinterpret_cast<const __nv_bfloat16*>(Bs + (warp % WN) * 32 * TK), TK);
  }
  tile.foreach_c([&](int r, int c, float& v) {
    const int t = t0 + r, n = n0 + (warp % WN) * 32 + c;
    if (t < T) store_out(y + (int64_t)t * N + n, v);
  });
}

// ------------------------------------------------------------------ GDN (gated delta net)

// Causal depthwise conv1d (4 taps) + SiLU over the concatenated qkv stream, splitting into q/k/v.
// qkv [T,C] bf16 (raw projections), conv_w [C,4], state [3,C] holds the 3 raw inputs preceding token 0.
// Outputs bf16. The state update is a separate kernel so reads and writes never race.
// When T == 1 (decode) the thread owning channel c is the only reader and writer of its state column, so the
// state shift is fused in (UPDATE_STATE); for T > 1 gdn_conv_state_kernel runs afterwards.
template <bool UPDATE_STATE>
__global__ void gdn_conv_kernel(const bf16* __restrict__ qkv, const bf16* __restrict__ conv_w,
                                bf16* __restrict__ state, bf16* __restrict__ q, bf16* __restrict__ k,
                                bf16* __restrict__ v, int T, int C, int dq, int dk) {
  const int64_t n = (int64_t)T * C;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const int t = (int)(i / C), c = (int)(i - (int64_t)t * C);
    const float w0 = bf2f(conv_w[c * 4 + 0]), w1 = bf2f(conv_w[c * 4 + 1]);
    const float w2 = bf2f(conv_w[c * 4 + 2]), w3 = bf2f(conv_w[c * 4 + 3]);
    const bf16 st0 = state[c], st1 = state[C + c], st2 = state[2 * C + c];
    const float s0 = t >= 3 ? bf2f(qkv[(int64_t)(t - 3) * C + c]) : bf2f(t == 0 ? st0 : t == 1 ? st1 : st2);
    const float s1 = t >= 2 ? bf2f(qkv[(int64_t)(t - 2) * C + c]) : bf2f(t == 0 ? st1 : st2);
    const float s2 = t >= 1 ? bf2f(qkv[(int64_t)(t - 1) * C + c]) : bf2f(st2);
    if (UPDATE_STATE) { state[c] = st1; state[C + c] = st2; state[2 * C + c] = qkv[i]; }  // T == 1 only
    const float x = bf2f(qkv[i]);
    const bf16 o = f2bf(silu(w0 * s0 + w1 * s1 + w2 * s2 + w3 * x));
    if (c < dq) q[(int64_t)t * dq + c] = o;
    else if (c < dq + dk) k[(int64_t)t * dk + (c - dq)] = o;
    else v[(int64_t)t * (C - dq - dk) + (c - dq - dk)] = o;
  }
}
// state <- last 3 raw inputs (shifting in the old state when T < 3). One thread per channel owns all
// three rows, so the read-then-write of the old state never races across threads.
__global__ void gdn_conv_state_kernel(const bf16* __restrict__ qkv, bf16* __restrict__ state, int T, int C) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= C) return;
  const bf16 o0 = state[c], o1 = state[C + c], o2 = state[2 * C + c];
  const bf16 old[3] = {o0, o1, o2};
  bf16 nw[3];
#pragma unroll
  for (int r = 0; r < 3; ++r) {
    const int t = T - 3 + r;
    nw[r] = t >= 0 ? qkv[(int64_t)t * C + c] : old[r + T];
  }
  state[c] = nw[0]; state[C + c] = nw[1]; state[2 * C + c] = nw[2];
}


// Fused per-token prep for the recurrence: warps [0, T*Hk) L2-normalize q rows (scaled by 1/sqrt(Dk)), the next
// T*Hk warps normalize k rows, the last T*Hv warps compute the alpha/beta gates (see gdn_gate_kernel).
__device__ __forceinline__ void gdn_gate_row(const bf16* __restrict__ x, const bf16* __restrict__ ab,
                                             const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                             float* __restrict__ alpha, float* __restrict__ beta, int job, int Hv, int H);
__global__ void __launch_bounds__(256) gdn_prep_kernel(bf16* __restrict__ q, bf16* __restrict__ k, int T, int Hk, int Dk,
                                                       const bf16* __restrict__ x, const bf16* __restrict__ ab,
                                                       const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                                       float* __restrict__ alpha, float* __restrict__ beta, int Hv, int H,
                                                       float eps) {
  const int lane = threadIdx.x & 31;
  int job = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int qk_rows = T * Hk;
  if (job < 2 * qk_rows) {
    bf16* r = (job < qk_rows ? q + (int64_t)job * Dk : k + (int64_t)(job - qk_rows) * Dk);
    const float scale = job < qk_rows ? rsqrtf((float)Dk) : 1.f;
    float ss = 0.f;
    for (int d = lane; d < Dk; d += 32) { const float v = bf2f(r[d]); ss += v * v; }
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss + eps) * scale;
    for (int d = lane; d < Dk; d += 32) r[d] = f2bf(bf2f(r[d]) * inv);
    return;
  }
  job -= 2 * qk_rows;
  if (job < T * Hv) gdn_gate_row(x, ab, A_log, dt_bias, alpha, beta, job, Hv, H);
}

// alpha/beta gate: a = x.Wa, b = x.Wb (Hv outputs each, ab = [Wa; Wb] rows), then
// alpha = exp(-exp(A_log) * softplus(a + dt_bias)), beta = sigmoid(b). One warp per (t, h).
__device__ __forceinline__ void gdn_gate_row(const bf16* __restrict__ x, const bf16* __restrict__ ab,
                                             const float* __restrict__ A_log, const float* __restrict__ dt_bias,
                                             float* __restrict__ alpha, float* __restrict__ beta, int job, int Hv, int H) {
  const int lane = threadIdx.x & 31;
  const int t = job / Hv, h = job - t * Hv;
  const uint4* xr = reinterpret_cast<const uint4*>(x + (int64_t)t * H);
  const uint4* ar = reinterpret_cast<const uint4*>(ab + (int64_t)h * H);
  const uint4* br = reinterpret_cast<const uint4*>(ab + (int64_t)(Hv + h) * H);
  float pa = 0.f, pb = 0.f;
  for (int v = lane; v < (H >> 3); v += 32) {
    float xf[8], af[8], bf[8];
    bf16x8(xr[v], xf); bf16x8(ar[v], af); bf16x8(br[v], bf);
#pragma unroll
    for (int j = 0; j < 8; ++j) { pa = fmaf(xf[j], af[j], pa); pb = fmaf(xf[j], bf[j], pb); }
  }
  pa = warp_sum(pa); pb = warp_sum(pb);
  if (lane == 0) {
    alpha[job] = __expf(-__expf(A_log[h]) * softplus(pa + dt_bias[h]));
    beta[job] = sigmoid(pb);
  }
}

// Gated delta rule recurrence. State S [Hv][Dv][Dk] fp32 (persistent). One warp per (h, iv) column:
// lane owns S[h][iv][4*lane .. 4*lane+3] (so k and q come in as one 8-byte load per lane). For each token:
// S *= alpha; delta = beta * (v - S.k); S += delta k; o = S.q.
// q,k [T,Hk,Dk] bf16 ; v,o [T,Hv,Dv] bf16 ; alpha,beta [T,Hv]. The block's warps share one head, so the per-token
// k/q rows and gates are staged through shared memory in chunks of CH tokens.
template <int DK, int CH>  // CH = tokens staged per chunk; CH == 0 selects the decode path (T == 1, no smem)
__global__ void __launch_bounds__(256) gdn_recurrence_kernel(const bf16* __restrict__ q, const bf16* __restrict__ k,
                                                             const bf16* __restrict__ v,
                                                             const float* __restrict__ alpha,
                                                             const float* __restrict__ beta, float* __restrict__ S,
                                                             bf16* __restrict__ o, int T, int Hk, int Hv, int Dv) {
  static_assert(DK == 128, "lane layout assumes Dk == 128");
  constexpr int WARPS = 8, SM = CH > 0 ? CH : 1;
  __shared__ __align__(16) bf16 ks[SM * DK];
  __shared__ __align__(16) bf16 qs[SM * DK];
  __shared__ float als[SM], bes[SM];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int job0 = blockIdx.x * WARPS;          // 8 consecutive columns of one head (Dv % 8 == 0)
  const int job = job0 + warp;
  const int h = job0 / Dv, iv = job - h * Dv;
  const int hk = h / (Hv / Hk);
  float* Scol = S + ((int64_t)h * Dv + iv) * DK;
  float s[4];
  {
    const float4 f = *reinterpret_cast<const float4*>(Scol + lane * 4);
    s[0] = f.x; s[1] = f.y; s[2] = f.z; s[3] = f.w;
  }
  if (CH == 0) {  // decode: no staging, no barriers
    const uint2 kw = *reinterpret_cast<const uint2*>(k + (int64_t)hk * DK + lane * 4);
    const uint2 qw = *reinterpret_cast<const uint2*>(q + (int64_t)hk * DK + lane * 4);
    const float kf[4] = {lo_bf(kw.x), hi_bf(kw.x), lo_bf(kw.y), hi_bf(kw.y)};
    const float qf[4] = {lo_bf(qw.x), hi_bf(qw.x), lo_bf(qw.y), hi_bf(qw.y)};
    const float al = alpha[h], be = beta[h], vv = bf2f(v[(int64_t)h * Dv + iv]);
    float sk = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { s[i] *= al; sk = fmaf(s[i], kf[i], sk); }
    sk = warp_sum(sk);
    const float dlt = be * (vv - sk);
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) { s[i] = fmaf(dlt, kf[i], s[i]); acc = fmaf(s[i], qf[i], acc); }
    acc = warp_sum(acc);
    if (lane == 0) o[(int64_t)h * Dv + iv] = f2bf(acc);
    *reinterpret_cast<float4*>(Scol + lane * 4) = make_float4(s[0], s[1], s[2], s[3]);
    return;
  }
  for (int t0 = 0; t0 < T; t0 += SM) {
    const int nt = min(SM, T - t0);
    __syncthreads();
    for (int c = threadIdx.x; c < SM * DK / 8; c += 256) {  // 16-byte chunks: 16 per token row
      const int r = c >> 4, ch = c & 15;
      if (r < nt) {
        const int64_t src = ((int64_t)(t0 + r) * Hk + hk) * DK + ch * 8;
        *reinterpret_cast<uint4*>(ks + r * DK + ch * 8) = *reinterpret_cast<const uint4*>(k + src);
        *reinterpret_cast<uint4*>(qs + r * DK + ch * 8) = *reinterpret_cast<const uint4*>(q + src);
      }
    }
    if (threadIdx.x < SM && threadIdx.x < nt) {
      als[threadIdx.x] = alpha[(t0 + threadIdx.x) * Hv + h];
      bes[threadIdx.x] = beta[(t0 + threadIdx.x) * Hv + h];
    }
    __syncthreads();
    for (int r = 0; r < nt; ++r) {
      const int t = t0 + r;
      const float al = als[r], be = bes[r];
      const float vv = bf2f(v[((int64_t)t * Hv + h) * Dv + iv]);
      const uint2 kw = *reinterpret_cast<const uint2*>(ks + r * DK + lane * 4);
      const uint2 qw = *reinterpret_cast<const uint2*>(qs + r * DK + lane * 4);
      const float kf[4] = {lo_bf(kw.x), hi_bf(kw.x), lo_bf(kw.y), hi_bf(kw.y)};
      const float qf[4] = {lo_bf(qw.x), hi_bf(qw.x), lo_bf(qw.y), hi_bf(qw.y)};
      float sk = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) { s[i] *= al; sk = fmaf(s[i], kf[i], sk); }
      sk = warp_sum(sk);
      const float dlt = be * (vv - sk);
      float acc = 0.f;
#pragma unroll
      for (int i = 0; i < 4; ++i) { s[i] = fmaf(dlt, kf[i], s[i]); acc = fmaf(s[i], qf[i], acc); }
      acc = warp_sum(acc);
      if (lane == 0) o[((int64_t)t * Hv + h) * Dv + iv] = f2bf(acc);
    }
  }
  *reinterpret_cast<float4*>(Scol + lane * 4) = make_float4(s[0], s[1], s[2], s[3]);
}

// o[t,h,:] = rms(o) * w * silu(z). One warp per (t, h) row of width Dv.
__global__ void gdn_gated_norm_kernel(bf16* __restrict__ o, const bf16* __restrict__ w, const bf16* __restrict__ z,
                                      int rows, int Dv, float eps) {
  const int lane = threadIdx.x & 31;
  const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (row >= rows) return;
  bf16* r = o + (int64_t)row * Dv;
  const bf16* zr = z + (int64_t)row * Dv;
  float ss = 0.f;
  for (int d = lane; d < Dv; d += 32) { const float v = bf2f(r[d]); ss += v * v; }
  ss = warp_sum(ss);
  const float inv = rsqrtf(ss / (float)Dv + eps);
  for (int d = lane; d < Dv; d += 32) r[d] = f2bf(bf2f(r[d]) * inv * bf2f(w[d]) * silu(bf2f(zr[d])));
}

// ------------------------------------------------------------------ full attention

// qkv [T, (Hq + 2 Hkv) * D] bf16 -> q [T,Hq,D] (normed + rope), K/V cache [Hkv][ctx][D] at pos0 + t.
// One warp per (t, head) where head < Hq + Hkv (v rows are copied by the k warps' block... simpler: heads
// index over Hq + 2*Hkv, v heads just copy).
__global__ void attn_qk_norm_rope_kernel(const bf16* __restrict__ qkv, const bf16* __restrict__ q_norm,
                                         const bf16* __restrict__ k_norm, bf16* __restrict__ q, bf16* __restrict__ kc,
                                         bf16* __restrict__ vc, int T, int Hq, int Hkv, int D, int rope_dim,
                                         float theta, int pos0, int ctx, float eps) {
  const int lane = threadIdx.x & 31;
  const int job = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  const int heads = Hq + 2 * Hkv;
  if (job >= T * heads) return;
  const int t = job / heads, hh = job - t * heads;
  const bf16* src = qkv + (int64_t)t * heads * D + (int64_t)hh * D;
  const int pos = pos0 + t;
  if (hh >= Hq + Hkv) {  // v: plain copy into cache
    bf16* dst = vc + ((int64_t)(hh - Hq - Hkv) * ctx + pos) * D;
    for (int d = lane; d < D; d += 32) dst[d] = src[d];
    return;
  }
  const bool is_q = hh < Hq;
  const bf16* w = is_q ? q_norm : k_norm;
  bf16* dst = is_q ? q + ((int64_t)t * Hq + hh) * D : kc + ((int64_t)(hh - Hq) * ctx + pos) * D;
  // D <= 256: each lane holds up to 8 values
  float vals[8];
  float ss = 0.f;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int d = lane + i * 32;
    vals[i] = d < D ? bf2f(src[d]) : 0.f;
    ss += vals[i] * vals[i];
  }
  ss = warp_sum(ss);
  const float inv = rsqrtf(ss / (float)D + eps);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int d = lane + i * 32;
    if (d < D) vals[i] = vals[i] * inv * bf2f(w[d]);
  }
  // rotate-half RoPE over the first rope_dim dims: pair (d, d + rope_dim/2)
  const int half = rope_dim >> 1;
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int d = lane + i * 32;
    if (d < half) {
      const int j = d + half;  // partner index; both live in this lane iff half % 32 == 0
      const float inv_freq = __powf(theta, -(float)(2 * d) / (float)rope_dim);
      const float ang = (float)pos * inv_freq;
      float c, s;
      __sincosf(ang, &s, &c);
      const float a = vals[i], b = vals[j >> 5];  // j>>5 = i + half/32
      vals[i] = a * c - b * s;
      vals[j >> 5] = b * c + a * s;
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const int d = lane + i * 32;
    if (d < D) dst[d] = f2bf(vals[i]);
  }
}

// Split-K online-softmax attention over the KV cache (flash-decoding), GQA-aware. Grid (T*Hkv, nsplit):
// block (job, sp) handles keys [sp*chunk, min((sp+1)*chunk, nkeys)) for one (t, kv head). Its 8 warps cover
// the G = Hq/Hkv query heads of that kv head (8/G warps per head, splitting the keys), so each K/V row is
// fetched from DRAM once per block instead of once per query head. Lane owns dims [8*lane, 8*lane+8) and
// loads 16 bytes at a time (D == 256). Writes unnormalized partials (acc[D], m, l) to
// part[(t*Hq + h)*nsplit + sp][D+2]; attn_combine_kernel merges the splits.
template <int D>
__global__ void __launch_bounds__(256) attn_partial_kernel(const bf16* __restrict__ q, const bf16* __restrict__ kc,
                                                           const bf16* __restrict__ vc, float* __restrict__ part,
                                                           int T, int Hq, int Hkv, int ctx, int pos0, int nsplit,
                                                           float scale) {
  static_assert(D == 256, "lane owns 8 dims: D must be 256");
  __shared__ float red_m[8], red_l[8];
  __shared__ float red_acc[8][D];
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int job = blockIdx.x, sp = blockIdx.y;
  const int t = job / Hkv, kvh = job - t * Hkv;
  const int G = Hq / Hkv;               // query heads per kv head (must divide 8)
  const int wph = 8 / G;                // warps per query head
  const int hq = kvh * G + warp / wph;  // this warp's query head
  const int sub = warp % wph;           // this warp's slice of the keys
  const int nkeys = pos0 + t + 1;
  const int chunk = (nkeys + nsplit - 1) / nsplit;
  const int k0 = sp * chunk, k1 = min(nkeys, k0 + chunk);
  float qf[8];
  bf16x8(reinterpret_cast<const uint4*>(q + ((int64_t)t * Hq + hq) * D)[lane], qf);
#pragma unroll
  for (int i = 0; i < 8; ++i) qf[i] *= scale;
  const uint4* kb = reinterpret_cast<const uint4*>(kc + (int64_t)kvh * ctx * D);
  const uint4* vb = reinterpret_cast<const uint4*>(vc + (int64_t)kvh * ctx * D);
  float m = -1e30f, l = 0.f, acc[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) acc[i] = 0.f;
  for (int j = k0 + sub; j < k1; j += wph) {
    float kf[8], vf[8];
    bf16x8(kb[(int64_t)j * (D / 8) + lane], kf);
    bf16x8(vb[(int64_t)j * (D / 8) + lane], vf);
    float sc = 0.f;
#pragma unroll
    for (int i = 0; i < 8; ++i) sc = fmaf(qf[i], kf[i], sc);
    sc = warp_sum(sc);
    const float m_new = fmaxf(m, sc);
    const float corr = __expf(m - m_new), p = __expf(sc - m_new);
    l = l * corr + p;
#pragma unroll
    for (int i = 0; i < 8; ++i) acc[i] = fmaf(p, vf[i], acc[i] * corr);
    m = m_new;
  }
  if (lane == 0) { red_m[warp] = m; red_l[warp] = l; }
#pragma unroll
  for (int i = 0; i < 8; ++i) red_acc[warp][lane * 8 + i] = acc[i];
  __syncthreads();
  if (sub == 0) {  // first warp of each head merges its wph warps
    const int w0 = warp;
    float gm = -1e30f;
    for (int w = w0; w < w0 + wph; ++w) gm = fmaxf(gm, red_m[w]);
    float gl = 0.f, out[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = 0.f;
    for (int w = w0; w < w0 + wph; ++w) {
      const float c = red_l[w] > 0.f ? __expf(red_m[w] - gm) : 0.f;
      gl += red_l[w] * c;
#pragma unroll
      for (int i = 0; i < 8; ++i) out[i] = fmaf(red_acc[w][lane * 8 + i], c, out[i]);
    }
    float* dst = part + (((int64_t)t * Hq + hq) * nsplit + sp) * (D + 2);
#pragma unroll
    for (int i = 0; i < 8; ++i) dst[lane * 8 + i] = out[i];
    if (lane == 0) { dst[D] = gm; dst[D + 1] = gl; }
  }
}

// Merge the nsplit partials of each (t, head) and normalize. One warp per job; lane owns 8 dims.
template <int D>
__global__ void attn_combine_kernel(const float* __restrict__ part, bf16* __restrict__ o, int jobs, int nsplit) {
  const int lane = threadIdx.x & 31;
  const int job = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (job >= jobs) return;
  const float* src = part + (int64_t)job * nsplit * (D + 2);
  float gm = -1e30f;
  for (int sp = 0; sp < nsplit; ++sp) gm = fmaxf(gm, src[sp * (D + 2) + D]);
  float gl = 0.f, out[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) out[i] = 0.f;
  for (int sp = 0; sp < nsplit; ++sp) {
    const float* p = src + sp * (D + 2);
    const float pl = p[D + 1];
    const float c = pl > 0.f ? __expf(p[D] - gm) : 0.f;
    gl += pl * c;
#pragma unroll
    for (int i = 0; i < 8; ++i) out[i] = fmaf(p[lane * 8 + i], c, out[i]);
  }
  const float inv = 1.f / gl;
  bf16* dst = o + (int64_t)job * D;
#pragma unroll
  for (int i = 0; i < 8; ++i) dst[lane * 8 + i] = f2bf(out[i] * inv);
}

// o *= sigmoid(g), elementwise over n values (bf16).
__global__ void sigmoid_gate_kernel(bf16* __restrict__ o, const bf16* __restrict__ g, int64_t n) {
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x)
    o[i] = f2bf(bf2f(o[i]) * sigmoid(bf2f(g[i])));
}

// ------------------------------------------------------------------ MoE

// logits [T, E] fp32 -> topk indices/weights (softmax over the selected k, i.e. renormalized).
// One warp per token; E <= 256, k <= 8.
__global__ void router_topk_kernel(const float* __restrict__ logits, int32_t* __restrict__ idx,
                                   float* __restrict__ wts, int T, int E, int K) {
  const int lane = threadIdx.x & 31;
  const int t = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
  if (t >= T) return;
  const float* row = logits + (int64_t)t * E;
  float vals[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) { const int e = lane + i * 32; vals[i] = e < E ? row[e] : -1e30f; }
  float pv[8]; int pi[8];
  for (int kk = 0; kk < K; ++kk) {
    float bv = -1e30f; int bi = 0x7fffffff;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      const int e = lane + i * 32;
      if (vals[i] > bv || (vals[i] == bv && e < bi)) { bv = vals[i]; bi = e; }
    }
    // warp argmax (value desc, index asc)
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
      const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
      if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
    }
    pv[kk] = bv; pi[kk] = bi;
#pragma unroll
    for (int i = 0; i < 8; ++i) if (lane + i * 32 == bi) vals[i] = -1e30f;
  }
  if (lane == 0) {
    float mx = pv[0];
    for (int kk = 1; kk < K; ++kk) mx = fmaxf(mx, pv[kk]);
    float den = 0.f;
    for (int kk = 0; kk < K; ++kk) { pv[kk] = __expf(pv[kk] - mx); den += pv[kk]; }
    for (int kk = 0; kk < K; ++kk) { idx[t * K + kk] = pi[kk]; wts[t * K + kk] = pv[kk] / den; }
  }
}

// Dequantized INT4 dot for one lane's 16-byte chunk (32 nibbles) against 32 activations.
__device__ __forceinline__ float int4_dot32(const uint4 w, const float* xf) {
  float acc = 0.f;
  const uint32_t ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t u = ws[j];
#pragma unroll
    for (int b = 0; b < 8; ++b) acc = fmaf((float)(int)((u >> (4 * b)) & 0xF) - 8.f, xf[j * 8 + b], acc);
  }
  return acc;
}

// Routed experts, gate/up: inter[t][slot][i] = silu(x[t].Wg[e][i]) * (x[t].Wu[e][i]).
// Wq [E][2I][K/2] u8, Ws [E][2I][K/G] f16 (G = 128 => one scale per 64 bytes = 4 lanes). K % 1024 == 0.
// One warp per (t, slot, i); 8 warps per block.
__global__ void __launch_bounds__(256) moe_gate_up_int4_kernel(const bf16* __restrict__ x,
                                                               const uint8_t* __restrict__ Wq,
                                                               const uint16_t* __restrict__ Ws,
                                                               const int32_t* __restrict__ idx,
                                                               bf16* __restrict__ inter, int T, int TOPK, int I,
                                                               int K) {
  const int lane = threadIdx.x & 31;
  const int job = blockIdx.x * 8 + (threadIdx.x >> 5);
  const int total = T * TOPK * I;
  if (job >= total) return;
  const int i = job % I;
  const int row = job / I;  // t * TOPK + slot
  const int t = row / TOPK;
  const int e = idx[row];
  const int64_t rowbytes = K >> 1;
  const int groups = K >> 7;
  const uint4* gw = reinterpret_cast<const uint4*>(Wq + ((int64_t)e * 2 * I + i) * rowbytes);
  const uint4* uw = reinterpret_cast<const uint4*>(Wq + ((int64_t)e * 2 * I + I + i) * rowbytes);
  const uint16_t* gs = Ws + ((int64_t)e * 2 * I + i) * groups;
  const uint16_t* us = Ws + ((int64_t)e * 2 * I + I + i) * groups;
  const uint4* xr = reinterpret_cast<const uint4*>(x + (int64_t)t * K);
  float g = 0.f, u = 0.f;
  const int nchunks = K >> 5;  // 16-byte chunks per row
  for (int c = lane; c < nchunks; c += 32) {
    float xf[32];
#pragma unroll
    for (int j = 0; j < 4; ++j) bf16x8(xr[c * 4 + j], xf + j * 8);
    const int grp = c >> 2;
    g = fmaf(int4_dot32(gw[c], xf), h2f(gs[grp]), g);
    u = fmaf(int4_dot32(uw[c], xf), h2f(us[grp]), u);
  }
  g = warp_sum(g); u = warp_sum(u);
  if (lane == 0) inter[(int64_t)row * I + i] = f2bf(silu(g) * u);
}

// Routed experts, down: routed[t][h] = sum_slot w[t][slot] * (inter[t][slot] . Wd[e][h]).
// Wd [E][H][I/2] u8, Sd [E][H][I/G] f16. I % 512 == 0 (each lane: I/32 nibbles = I/64 bytes, 8B when I=512).
// One warp per (t, h).
__global__ void __launch_bounds__(256) moe_down_int4_kernel(const bf16* __restrict__ inter,
                                                            const uint8_t* __restrict__ Wd,
                                                            const uint16_t* __restrict__ Sd,
                                                            const int32_t* __restrict__ idx,
                                                            const float* __restrict__ wts, float* __restrict__ routed,
                                                            int T, int TOPK, int H, int I) {
  const int lane = threadIdx.x & 31;
  const int job = blockIdx.x * 8 + (threadIdx.x >> 5);
  if (job >= T * H) return;
  const int t = job / H, h = job - t * H;
  const int64_t rowbytes = I >> 1;
  const int groups = I >> 7;
  const int nchunks = I >> 4;  // 8-byte chunks per row (16 nibbles)
  float acc = 0.f;
  for (int slot = 0; slot < TOPK; ++slot) {
    const int e = idx[t * TOPK + slot];
    const float w = wts[t * TOPK + slot];
    const uint2* wr = reinterpret_cast<const uint2*>(Wd + ((int64_t)e * H + h) * rowbytes);
    const uint16_t* sr = Sd + ((int64_t)e * H + h) * groups;
    const uint4* xr = reinterpret_cast<const uint4*>(inter + ((int64_t)t * TOPK + slot) * I);
    float part = 0.f;
    for (int c = lane; c < nchunks; c += 32) {
      float xf[16];
      bf16x8(xr[c * 2], xf); bf16x8(xr[c * 2 + 1], xf + 8);
      const uint2 wv = wr[c];
      float d = 0.f;
#pragma unroll
      for (int b = 0; b < 8; ++b) d = fmaf((float)(int)((wv.x >> (4 * b)) & 0xF) - 8.f, xf[b], d);
#pragma unroll
      for (int b = 0; b < 8; ++b) d = fmaf((float)(int)((wv.y >> (4 * b)) & 0xF) - 8.f, xf[8 + b], d);
      part = fmaf(d, h2f(sr[c >> 3]), part);
    }
    acc = fmaf(w, part, acc);
  }
  acc = warp_sum(acc);
  if (lane == 0) routed[(int64_t)t * H + h] = acc;
}

// ---- grouped (prefill) MoE: sort (token, slot) pairs by expert so each expert's weights are dequantized
// once per tile of tokens instead of once per token.

// Single block. pairs = T*TOPK (<= 32768). Writes offsets[E+1] (exclusive prefix over expert counts) and
// sorted[pairs] = t*TOPK+slot grouped by expert (order within an expert is arbitrary).
__global__ void __launch_bounds__(256) moe_sort_kernel(const int32_t* __restrict__ idx, int pairs, int E,
                                                       int32_t* __restrict__ offsets, int32_t* __restrict__ sorted) {
  __shared__ int cnt[256];
  __shared__ int cursor[256];
  for (int e = threadIdx.x; e < E; e += blockDim.x) cnt[e] = 0;
  __syncthreads();
  for (int p = threadIdx.x; p < pairs; p += blockDim.x) atomicAdd(&cnt[idx[p]], 1);
  __syncthreads();
  if (threadIdx.x == 0) {  // E <= 256: serial scan is fine
    int run = 0;
    for (int e = 0; e < E; ++e) { offsets[e] = run; cursor[e] = run; run += cnt[e]; }
    offsets[E] = run;
  }
  __syncthreads();
  for (int p = threadIdx.x; p < pairs; p += blockDim.x) sorted[atomicAdd(&cursor[idx[p]], 1)] = p;
}

// 32 INT4 values (k0..k0+31, packed low nibble first) at fp32 scale s -> 32 bf16 into smem (64 bytes, 16-aligned).
__device__ __forceinline__ void dequant32_bf16(const uint4 w, float s, bf16* __restrict__ out) {
  const uint32_t ww[4] = {w.x, w.y, w.z, w.w};
  uint4* o = reinterpret_cast<uint4*>(out);
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    uint32_t p[4];
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const float lo = ((float)(int)((ww[q] >> (8 * b)) & 0xF) - 8.f) * s;
      const float hi = ((float)(int)((ww[q] >> (8 * b + 4)) & 0xF) - 8.f) * s;
      p[b] = (uint32_t)f2bf(lo) | ((uint32_t)f2bf(hi) << 16);
    }
    o[q] = make_uint4(p[0], p[1], p[2], p[3]);
  }
}

// Tensor-core gate/up for one expert over tiles of 32 of its (token, slot) pairs. Grid (I / (NB/2), E), NB/32
// warps. The block owns NB/2 gate rows [i0, i0+NB/2) and their NB/2 up rows [I+i0, ..): the first half of the
// warps take gate rows, the second half up rows, 32 rows each. Per K step the block gathers the 32 activation
// rows and dequantizes its NB INT4 weight rows into shared memory (once per token tile, not once per token),
// then each warp runs a 32x32xTK MMA on the tensor path.
template <int TK, int NB>
__global__ void __launch_bounds__(NB) moe_gate_up_tile_kernel(const bf16* __restrict__ x,
                                                               const uint8_t* __restrict__ Wq,
                                                               const uint16_t* __restrict__ Ws,
                                                               const int32_t* __restrict__ offsets,
                                                               const int32_t* __restrict__ sorted,
                                                               bf16* __restrict__ inter, int TOPK, int I, int K) {
  static_assert(TK == 32 || TK == 64 || TK == 128, "TK must divide the 128-wide scale group");
  static_assert(NB == 64 || NB == 128, "NB rows per block");
  constexpr int MT = 32, NT = NB, HALF = NB / 2;
  constexpr int STAGE = (MT + NB) * TK * 2, EPI = MT * NB * 4;
  __shared__ __align__(16) uint8_t smem[STAGE > EPI ? STAGE : EPI];
  __shared__ int tok[MT];
  bf16* As = reinterpret_cast<bf16*>(smem);
  bf16* Bs = As + MT * TK;
  float* Cs = reinterpret_cast<float*>(smem);
  const int e = blockIdx.y;
  const int p0 = offsets[e], p1 = offsets[e + 1];
  if (p0 == p1) return;
  const int tid = threadIdx.x, warp = tid >> 5;
  const int i0 = blockIdx.x * HALF;
  const int myrow = tid < HALF ? i0 + tid : I + i0 + tid - HALF;
  const int64_t rowbytes = K >> 1;
  const int groups = K >> 7;
  const uint8_t* wrow = Wq + ((int64_t)e * 2 * I + myrow) * rowbytes;
  const uint16_t* srow = Ws + ((int64_t)e * 2 * I + myrow) * groups;
  mvcc::warp_tile<32, 32, TK, __nv_bfloat16> tile;
  for (int pt = p0; pt < p1; pt += MT) {
    const int nt = min(MT, p1 - pt);
    __syncthreads();  // previous tile's epilogue finished with Cs / tok
    if (tid < MT) tok[tid] = tid < nt ? sorted[pt + tid] : -1;
    tile.zero();
    for (int k0 = 0; k0 < K; k0 += TK) {
      __syncthreads();  // previous MMA done reading As/Bs (and tok visible on the first step)
      for (int c = tid; c < MT * TK / 8; c += NT) {
        const int r = c / (TK / 8), ch = c % (TK / 8);
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (tok[r] >= 0) v = *reinterpret_cast<const uint4*>(x + (int64_t)(tok[r] / TOPK) * K + k0 + ch * 8);
        *reinterpret_cast<uint4*>(As + r * TK + ch * 8) = v;
      }
      const float sc = h2f(srow[k0 >> 7]);
#pragma unroll
      for (int q = 0; q < TK / 32; ++q)
        dequant32_bf16(*reinterpret_cast<const uint4*>(wrow + (k0 >> 1) + q * 16), sc, Bs + tid * TK + q * 32);
      __syncthreads();
      tile.mma(reinterpret_cast<const __nv_bfloat16*>(As), TK,
               reinterpret_cast<const __nv_bfloat16*>(Bs + warp * 32 * TK), TK);
    }
    __syncthreads();  // MMA done with As/Bs before Cs aliases them
    tile.foreach_c([&](int r, int c, float& v) { Cs[r * NB + warp * 32 + c] = v; });
    __syncthreads();
    for (int idx = tid; idx < MT * HALF; idx += NT) {
      const int r = idx / HALF, i = idx % HALF;
      if (tok[r] >= 0) inter[(int64_t)tok[r] * I + i0 + i] = f2bf(silu(Cs[r * NB + i]) * Cs[r * NB + HALF + i]);
    }
  }
}

// Tensor-core down projection per expert over its token tiles; writes routing-weighted per-slot partials to
// slot_out[(t*TOPK+slot)][H]. Grid (H / NB, E), NB/32 warps x 32 output rows.
template <int TK, int NB>
__global__ void __launch_bounds__(NB) moe_down_tile_kernel(const bf16* __restrict__ inter,
                                                            const uint8_t* __restrict__ Wd,
                                                            const uint16_t* __restrict__ Sd,
                                                            const int32_t* __restrict__ offsets,
                                                            const int32_t* __restrict__ sorted,
                                                            const float* __restrict__ wts,
                                                            float* __restrict__ slot_out, int H, int I) {
  static_assert(TK == 32 || TK == 64 || TK == 128, "TK must divide the 128-wide scale group");
  constexpr int MT = 32, NT = NB;
  __shared__ __align__(16) uint8_t smem[(MT + NB) * TK * 2];
  __shared__ int tok[MT];
  bf16* As = reinterpret_cast<bf16*>(smem);
  bf16* Bs = As + MT * TK;
  const int e = blockIdx.y;
  const int p0 = offsets[e], p1 = offsets[e + 1];
  if (p0 == p1) return;
  const int tid = threadIdx.x, warp = tid >> 5;
  const int h0 = blockIdx.x * NB;
  const int64_t rowbytes = I >> 1;
  const int groups = I >> 7;
  const uint8_t* wrow = Wd + ((int64_t)e * H + h0 + tid) * rowbytes;
  const uint16_t* srow = Sd + ((int64_t)e * H + h0 + tid) * groups;
  mvcc::warp_tile<32, 32, TK, __nv_bfloat16> tile;
  for (int pt = p0; pt < p1; pt += MT) {
    const int nt = min(MT, p1 - pt);
    __syncthreads();
    if (tid < MT) tok[tid] = tid < nt ? sorted[pt + tid] : -1;
    tile.zero();
    for (int k0 = 0; k0 < I; k0 += TK) {
      __syncthreads();
      for (int c = tid; c < MT * TK / 8; c += NT) {
        const int r = c / (TK / 8), ch = c % (TK / 8);
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (tok[r] >= 0) v = *reinterpret_cast<const uint4*>(inter + (int64_t)tok[r] * I + k0 + ch * 8);
        *reinterpret_cast<uint4*>(As + r * TK + ch * 8) = v;
      }
      const float sc = h2f(srow[k0 >> 7]);
#pragma unroll
      for (int q = 0; q < TK / 32; ++q)
        dequant32_bf16(*reinterpret_cast<const uint4*>(wrow + (k0 >> 1) + q * 16), sc, Bs + tid * TK + q * 32);
      __syncthreads();
      tile.mma(reinterpret_cast<const __nv_bfloat16*>(As), TK,
               reinterpret_cast<const __nv_bfloat16*>(Bs + warp * 32 * TK), TK);
    }
    tile.foreach_c([&](int r, int c, float& v) {
      const int pair = tok[r];
      if (pair >= 0) slot_out[(int64_t)pair * H + h0 + warp * 32 + c] = v * wts[pair];
    });
  }
}

// routed[t][h] = sum over slots of slot_out[t*TOPK+slot][h]  (deterministic reduction, no atomics).
__global__ void moe_slot_reduce_kernel(const float* __restrict__ slot_out, float* __restrict__ routed, int T,
                                       int TOPK, int H) {
  const int64_t n = (int64_t)T * H;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const int t = (int)(i / H), h = (int)(i - (int64_t)t * H);
    float a = 0.f;
    for (int s = 0; s < TOPK; ++s) a += slot_out[((int64_t)t * TOPK + s) * H + h];
    routed[i] = a;
  }
}

// inter = silu(gu[:, :I]) * gu[:, I:] for the shared expert (gu bf16 [T, 2I]).
__global__ void silu_mul_kernel(const bf16* __restrict__ gu, bf16* __restrict__ out, int T, int I) {
  const int64_t n = (int64_t)T * I;
  for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (int64_t)gridDim.x * blockDim.x) {
    const int t = (int)(i / I), j = (int)(i - (int64_t)t * I);
    const bf16* r = gu + (int64_t)t * 2 * I;
    out[i] = f2bf(silu(bf2f(r[j])) * bf2f(r[I + j]));
  }
}

// hidden[t] += routed[t] + sigmoid(x[t].g) * shared[t]. One block per token.
__global__ void shared_gate_combine_kernel(float* __restrict__ hidden, const float* __restrict__ routed,
                                           const bf16* __restrict__ shared, const bf16* __restrict__ x,
                                           const bf16* __restrict__ g, int H) {
  __shared__ float sc[32];
  const int t = blockIdx.x;
  float acc = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) acc = fmaf(bf2f(x[(int64_t)t * H + i]), bf2f(g[i]), acc);
  acc = block_sum(acc, sc);
  const float gate = sigmoid(acc);
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    const int64_t o = (int64_t)t * H + i;
    hidden[o] += routed[o] + gate * bf2f(shared[o]);
  }
}

// Same, fused with the next layer's input RMSNorm: normed = rms(hidden) * w_next (saves a launch per layer).
__global__ void shared_gate_combine_rmsnorm_kernel(float* __restrict__ hidden, const float* __restrict__ routed,
                                                   const bf16* __restrict__ shared, const bf16* __restrict__ x,
                                                   const bf16* __restrict__ g, const bf16* __restrict__ w_next,
                                                   bf16* __restrict__ normed, int H, float eps) {
  __shared__ float sc[32];
  const int t = blockIdx.x;
  float acc = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) acc = fmaf(bf2f(x[(int64_t)t * H + i]), bf2f(g[i]), acc);
  acc = block_sum(acc, sc);
  const float gate = sigmoid(acc);
  float ss = 0.f;
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    const int64_t o = (int64_t)t * H + i;
    const float h = hidden[o] + routed[o] + gate * bf2f(shared[o]);
    hidden[o] = h;
    ss = fmaf(h, h, ss);
  }
  ss = block_sum(ss, sc);
  const float inv = rsqrtf(ss / (float)H + eps);
  for (int i = threadIdx.x; i < H; i += blockDim.x) {
    const int64_t o = (int64_t)t * H + i;
    normed[o] = f2bf(hidden[o] * inv * bf2f(w_next[i]));
  }
}

// ------------------------------------------------------------------ argmax over logits

// Stage 1: each block scans a strided slice of logits[V] and writes (max, idx) partials.
__global__ void argmax_partial_kernel(const float* __restrict__ logits, int V, float* __restrict__ pv,
                                      int32_t* __restrict__ pi) {
  __shared__ float sv[256];
  __shared__ int si[256];
  float bv = -1e30f; int bi = 0;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < V; i += gridDim.x * blockDim.x) {
    const float v = logits[i];
    if (v > bv || (v == bv && i < bi)) { bv = v; bi = i; }
  }
  sv[threadIdx.x] = bv; si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = sv[threadIdx.x + s]; const int oi = si[threadIdx.x + s];
      if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = ov; si[threadIdx.x] = oi; }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) { pv[blockIdx.x] = sv[0]; pi[blockIdx.x] = si[0]; }
}
// Stage 2: single block reduces the partials.
__global__ void argmax_final_kernel(const float* __restrict__ pv, const int32_t* __restrict__ pi, int n,
                                    int32_t* __restrict__ out) {
  __shared__ float sv[256];
  __shared__ int si[256];
  float bv = -1e30f; int bi = 0;
  for (int i = threadIdx.x; i < n; i += blockDim.x) {
    if (pv[i] > bv || (pv[i] == bv && pi[i] < bi)) { bv = pv[i]; bi = pi[i]; }
  }
  sv[threadIdx.x] = bv; si[threadIdx.x] = bi;
  __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      const float ov = sv[threadIdx.x + s]; const int oi = si[threadIdx.x + s];
      if (ov > sv[threadIdx.x] || (ov == sv[threadIdx.x] && oi < si[threadIdx.x])) { sv[threadIdx.x] = ov; si[threadIdx.x] = oi; }
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) out[0] = si[0];
}

}  // namespace qk
