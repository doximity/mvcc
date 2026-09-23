// Qwen3.5-MoE inference with nothing but CUDA kernels (no cuBLAS, no frameworks), INT4 routed experts.
//
//   nvcc -O3 -std=c++17 -o qwen qwen.cu
//   ./qwen --pack model.mvccq --tokenizer tokenizer.bin --prompt "Explain RoPE briefly." --max-tokens 200
//
// Model: 40 layers alternating 3x gated delta-net (linear attention) + 1x gated full attention, each followed
// by a 256-expert MoE (top-8) plus a shared expert. Weights come from convert.py (GPTQ-Int4 experts, bf16 or
// int8 dense). The forward pass is one stream of kernel launches; prefill runs T tokens per launch, decode
// runs T == 1.
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <type_traits>
#include <vector>

#include "kernels.cuh"

// Kernel tuning knobs (overridable with -D): rows per warp in the int8 decode GEMV; K step and weight rows per
// block in the tensor-core MoE tiles.
#ifndef QWEN_GEMV_R
#define QWEN_GEMV_R 1
#endif
#ifndef QWEN_MOE_TK
#define QWEN_MOE_TK 32
#endif
#ifndef QWEN_MOE_NB
#define QWEN_MOE_NB 128
#endif
#include "pack.hpp"
#include "tokenizer.hpp"

using qk::bf16;

#define CK(x)                                                                                              \
  do {                                                                                                     \
    cudaError_t _e = (x);                                                                                  \
    if (_e != cudaSuccess) {                                                                               \
      fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #x, __FILE__, __LINE__, cudaGetErrorString(_e));     \
      exit(1);                                                                                             \
    }                                                                                                      \
  } while (0)

static inline int cdiv(int64_t a, int64_t b) { return (int)((a + b - 1) / b); }
static constexpr int kMaxSplit = 32;

// Split-K attention: pick the split count so a decode step fills the GPU (>= ~64 blocks) while each split
// still streams a few dozen keys.
static void attention(const bf16* q, const bf16* kc, const bf16* vc, float* part, bf16* o, int T, int Hq, int Hkv,
                      int D, int ctx, int pos0, cudaStream_t s) {
  const int blocks = T * Hkv, jobs = T * Hq;
  const int nkeys = pos0 + T;
  if (8 % (Hq / Hkv) != 0) { fprintf(stderr, "attention: Hq/Hkv must divide 8\n"); exit(1); }
  // enough blocks to cover the GPU, but keep >= 32 keys per split so the combine stays cheap
  const int nsplit = std::max(1, std::min({kMaxSplit, cdiv(nkeys, 32), cdiv(64, blocks)}));
  qk::attn_partial_kernel<256><<<dim3(blocks, nsplit), 256, 0, s>>>(q, kc, vc, part, T, Hq, Hkv, ctx, pos0, nsplit,
                                                                    1.f / sqrtf((float)D));
  qk::attn_combine_kernel<256><<<cdiv(jobs, 8), 256, 0, s>>>(part, o, jobs, nsplit);
}

// --debug: synchronize after every stage, surface launch errors, and print activation statistics.
static bool g_debug = false;
static float host_bf2f(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
template <typename T>
static void dbg(const char* what, const T* dptr, int64_t n) {
  if (!g_debug) return;
  cudaError_t e = cudaDeviceSynchronize();
  if (e == cudaSuccess) e = cudaGetLastError();
  if (e != cudaSuccess) { fprintf(stderr, "[debug] %s: CUDA error %s\n", what, cudaGetErrorString(e)); exit(1); }
  std::vector<T> h(n);
  CK(cudaMemcpy(h.data(), dptr, n * sizeof(T), cudaMemcpyDeviceToHost));
  double sum = 0, sq = 0; float amax = 0; int64_t nans = 0;
  for (int64_t i = 0; i < n; ++i) {
    float v = std::is_same<T, float>::value ? (float)h[i] : host_bf2f((uint16_t)h[i]);
    if (std::isnan(v) || std::isinf(v)) { ++nans; continue; }
    sum += v; sq += (double)v * v; amax = std::max(amax, fabsf(v));
  }
  fprintf(stderr, "[debug] %-28s n=%-9lld mean=%+.5f rms=%.5f absmax=%.4f nan/inf=%lld  [%g %g %g %g]\n", what,
          (long long)n, sum / n, sqrt(sq / n), amax, (long long)nans,
          std::is_same<T, float>::value ? (float)h[0] : host_bf2f((uint16_t)h[0]),
          std::is_same<T, float>::value ? (float)h[1] : host_bf2f((uint16_t)h[1]),
          std::is_same<T, float>::value ? (float)h[2] : host_bf2f((uint16_t)h[2]),
          std::is_same<T, float>::value ? (float)h[3] : host_bf2f((uint16_t)h[3]));
}

template <typename T> static T* dalloc(size_t n) {
  void* p = nullptr;
  CK(cudaMalloc(&p, n * sizeof(T) + 16));
  return (T*)p;
}

// ------------------------------------------------------------------ weights

struct Linear {  // y = x . W^T ; W [N, K]
  const bf16* w = nullptr;      // bf16 weights
  const int8_t* q8 = nullptr;   // or int8 weights ...
  const uint16_t* s8 = nullptr; // ... with fp16 group scales
  int N = 0, K = 0;
  bool valid() const { return w || q8; }
};

struct Layer {
  int kind = 0;  // 1 = GDN, 0 = attention
  const bf16 *rms1, *rms2, *router;
  Linear shared_gate_up, shared_down;
  const bf16* shared_gate;
  const uint8_t *exp_gu_q, *exp_dn_q;
  const uint16_t *exp_gu_s, *exp_dn_s;
  // GDN
  Linear gdn_qkv, gdn_z, gdn_out;
  const bf16 *gdn_ab, *gdn_conv, *gdn_norm;
  const float *gdn_A_log, *gdn_dt;
  bf16* conv_state;  // [3, C]
  float* S;          // [Hv, Dv, Dk]
  // attention
  Linear attn_qkv, attn_gate, attn_o;
  const bf16 *q_norm, *k_norm;
  bf16 *kc, *vc;  // [Hkv][ctx][D]
};

struct Model {
  Pack pack;
  int H, L, E, TOPK, I, Is, Hq, Hkv, D, Hk, Hv, Dk, Dv, V, rope_dim, conv_k;
  float eps, theta;
  bool attn_gate;
  std::vector<int> eos;
  const bf16 *embed, *final_norm;
  Linear lm_head;
  std::vector<Layer> layers;
  int ctx = 0;

  Linear linear(const std::string& name, int N, int K) {
    Linear l;
    l.N = N; l.K = K;
    if (pack.has(name)) {
      const auto& ti = pack.info(name);
      if (ti.shape.size() != 2 || ti.shape[0] != N || ti.shape[1] != K) {
        fprintf(stderr, "tensor %s has shape [%lld,%lld], expected [%d,%d]\n", name.c_str(), (long long)ti.shape[0],
                (long long)ti.shape[1], N, K);
        exit(1);
      }
      l.w = pack.dev<bf16>(name);
    } else if (pack.has(name + ".q8")) {
      l.q8 = pack.dev<int8_t>(name + ".q8");
      l.s8 = pack.dev<uint16_t>(name + ".s8");
    } else {
      fprintf(stderr, "missing tensor %s\n", name.c_str());
      exit(1);
    }
    return l;
  }

  void load(const std::string& path, int max_ctx) {
    pack.open(path);
    const Json& c = pack.config;
    H = c["hidden"].i(); L = c["layers"].i(); E = c["experts"].i(); TOPK = c["topk"].i(); I = c["inter"].i();
    Is = c["shared_inter"].i(); Hq = c["q_heads"].i(); Hkv = c["kv_heads"].i(); D = c["head_dim"].i();
    Hk = c["gdn_k_heads"].i(); Hv = c["gdn_v_heads"].i(); Dk = c["gdn_k_dim"].i(); Dv = c["gdn_v_dim"].i();
    V = c["vocab"].i(); rope_dim = c["rope_dim"].i(); conv_k = c["conv_k"].i(); eps = c["eps"].f();
    theta = c["rope_theta"].f(); attn_gate = c["attn_gate"].b;
    for (const auto& e : c["eos"].arr) eos.push_back(e.i());
    ctx = max_ctx;
    if (Dk != 128 || D != 256 || conv_k != 4 || rope_dim % 64 != 0 || H % 256 != 0 || I % 512 != 0) {
      fprintf(stderr, "unsupported geometry (Dk=%d D=%d conv=%d rope=%d H=%d I=%d)\n", Dk, D, conv_k, rope_dim, H, I);
      exit(1);
    }
    embed = pack.dev<bf16>("embed");
    final_norm = pack.dev<bf16>("final_norm");
    lm_head = linear("lm_head", V, H);
    layers.resize(L);
    const int C = 2 * Hk * Dk + Hv * Dv;
    for (int i = 0; i < L; ++i) {
      Layer& l = layers[i];
      std::string p = "L" + std::to_string(i) + ".";
      l.kind = c["layer_kinds"][i].i();
      l.rms1 = pack.dev<bf16>(p + "rms1");
      l.rms2 = pack.dev<bf16>(p + "rms2");
      l.router = pack.dev<bf16>(p + "router");
      l.shared_gate_up = linear(p + "shared.gate_up", 2 * Is, H);
      l.shared_down = linear(p + "shared.down", H, Is);
      l.shared_gate = pack.dev<bf16>(p + "shared.gate");
      l.exp_gu_q = pack.dev<uint8_t>(p + "exp.gate_up.q");
      l.exp_gu_s = pack.dev<uint16_t>(p + "exp.gate_up.s");
      l.exp_dn_q = pack.dev<uint8_t>(p + "exp.down.q");
      l.exp_dn_s = pack.dev<uint16_t>(p + "exp.down.s");
      if (l.kind == 1) {
        l.gdn_qkv = linear(p + "gdn.qkv", C, H);
        l.gdn_z = linear(p + "gdn.z", Hv * Dv, H);
        l.gdn_out = linear(p + "gdn.out", H, Hv * Dv);
        l.gdn_ab = pack.dev<bf16>(p + "gdn.ab");
        l.gdn_conv = pack.dev<bf16>(p + "gdn.conv");
        l.gdn_norm = pack.dev<bf16>(p + "gdn.norm");
        l.gdn_A_log = pack.dev<float>(p + "gdn.A_log");
        l.gdn_dt = pack.dev<float>(p + "gdn.dt_bias");
        l.conv_state = dalloc<bf16>((size_t)3 * C);
        l.S = dalloc<float>((size_t)Hv * Dv * Dk);
      } else {
        l.attn_qkv = linear(p + "attn.qkv", (Hq + 2 * Hkv) * D, H);
        if (attn_gate) l.attn_gate = linear(p + "attn.gate", Hq * D, H);
        l.attn_o = linear(p + "attn.o", H, Hq * D);
        l.q_norm = pack.dev<bf16>(p + "attn.q_norm");
        l.k_norm = pack.dev<bf16>(p + "attn.k_norm");
        l.kc = dalloc<bf16>((size_t)Hkv * ctx * D);
        l.vc = dalloc<bf16>((size_t)Hkv * ctx * D);
      }
    }
  }

  void reset_state() {
    const int C = 2 * Hk * Dk + Hv * Dv;
    for (auto& l : layers) {
      if (l.kind == 1) {
        CK(cudaMemset(l.conv_state, 0, (size_t)3 * C * sizeof(bf16)));
        CK(cudaMemset(l.S, 0, (size_t)Hv * Dv * Dk * sizeof(float)));
      }
    }
  }
};

// ------------------------------------------------------------------ workspace

struct Work {
  int maxT;
  float* hidden;    // [T,H]
  bf16* normed;     // [T,H]
  bf16* qkv;        // [T, max(C, (Hq+2Hkv)D)]
  bf16* z;          // [T, Hv*Dv]  (also attention gate)
  bf16* q;          // [T, Hk*Dk]  / [T, Hq*D]
  bf16* k;          // [T, Hk*Dk]
  bf16* v;          // [T, Hv*Dv]
  bf16* o;          // [T, Hv*Dv] / [T, Hq*D]
  float* alpha;     // [T, Hv]
  float* beta;      // [T, Hv]
  bf16* attn_out;   // [T, H]
  float* logits_r;  // [T, E] router logits
  int32_t* topk_i;  // [T, TOPK]
  float* topk_w;    // [T, TOPK]
  bf16* inter;      // [T, TOPK, I]
  float* routed;    // [T, H]
  int32_t* moe_off; // [E+1] grouped MoE: expert offsets into moe_sorted
  int32_t* moe_sorted;  // [T*TOPK]
  float* slot_out;  // [T*TOPK, H] per-slot down-projection partials
  bf16* sgu;        // [T, 2 Is]
  bf16* sinter;     // [T, Is]
  bf16* sout;       // [T, H]
  float* attn_part; // [T*Hq][kMaxSplit][D+2] split-K attention partials
  float* logits;    // [V]
  float* pmax;      // [1024]
  int32_t* pidx;    // [1024]
  int32_t* ids;     // [T]
  int32_t* next;    // [1]

  void alloc(const Model& m, int T) {
    T = (T + 127) / 128 * 128;  // tensor-core GEMM tiles read (and discard) rows up to the next multiple of 128
    maxT = T;
    const int C = 2 * m.Hk * m.Dk + m.Hv * m.Dv;
    const int qkv_w = std::max(C, (m.Hq + 2 * m.Hkv) * m.D);
    const int wide = std::max(m.Hv * m.Dv, m.Hq * m.D);
    hidden = dalloc<float>((size_t)T * m.H);
    normed = dalloc<bf16>((size_t)T * m.H);
    qkv = dalloc<bf16>((size_t)T * qkv_w);
    z = dalloc<bf16>((size_t)T * wide);
    q = dalloc<bf16>((size_t)T * wide);
    k = dalloc<bf16>((size_t)T * m.Hk * m.Dk);
    v = dalloc<bf16>((size_t)T * m.Hv * m.Dv);
    o = dalloc<bf16>((size_t)T * wide);
    alpha = dalloc<float>((size_t)T * m.Hv);
    beta = dalloc<float>((size_t)T * m.Hv);
    attn_out = dalloc<bf16>((size_t)T * m.H);
    logits_r = dalloc<float>((size_t)T * m.E);
    topk_i = dalloc<int32_t>((size_t)T * m.TOPK);
    topk_w = dalloc<float>((size_t)T * m.TOPK);
    inter = dalloc<bf16>((size_t)T * m.TOPK * m.I);
    routed = dalloc<float>((size_t)T * m.H);
    moe_off = dalloc<int32_t>((size_t)m.E + 1);
    moe_sorted = dalloc<int32_t>((size_t)T * m.TOPK);
    slot_out = dalloc<float>((size_t)T * m.TOPK * m.H);
    sgu = dalloc<bf16>((size_t)T * 2 * m.Is);
    sinter = dalloc<bf16>((size_t)T * m.Is);
    sout = dalloc<bf16>((size_t)T * m.H);
    logits = dalloc<float>((size_t)m.V);
    attn_part = dalloc<float>((size_t)T * m.Hq * kMaxSplit * (m.D + 2));
    pmax = dalloc<float>(1024);
    pidx = dalloc<int32_t>(1024);
    ids = dalloc<int32_t>((size_t)T);
    next = dalloc<int32_t>(1);
  }
};

// ------------------------------------------------------------------ launch helpers

template <typename OutT>
static void dense(const bf16* x, const Linear& w, OutT* y, int T, cudaStream_t s) {
  const int N = w.N, K = w.K;
  if (w.w) {
    if (T >= 32 && K % 128 == 0 && N % 32 == 0) {
      // token tiles innermost so consecutive blocks share a weight tile; 4x4 warps when the problem is big enough
      if (T >= 128 && N >= 2048) {
        dim3 grid(cdiv(T, 128), cdiv(N, 128));
        qk::gemm_bf16_tc_kernel<4, 4, OutT><<<grid, 512, 0, s>>>(x, w.w, y, T, N, K);
      } else {
        dim3 grid(cdiv(T, 64), cdiv(N, 64));
        qk::gemm_bf16_tc_kernel<2, 2, OutT><<<grid, 128, 0, s>>>(x, w.w, y, T, N, K);
      }
    } else if (T >= 8) {
      dim3 grid(cdiv(N, 64), cdiv(T, 32));
      qk::gemm_bf16_kernel<OutT><<<grid, 256, 0, s>>>(x, w.w, y, T, N, K);
    } else if (T == 1) {
      qk::gemv_bf16_kernel<1, OutT><<<dim3(cdiv(N, 8), 1), 256, 0, s>>>(x, w.w, y, T, N, K);
    } else if (T == 2) {
      qk::gemv_bf16_kernel<2, OutT><<<dim3(cdiv(N, 8), 1), 256, 0, s>>>(x, w.w, y, T, N, K);
    } else {
      qk::gemv_bf16_kernel<4, OutT><<<dim3(cdiv(N, 8), cdiv(T, 4)), 256, 0, s>>>(x, w.w, y, T, N, K);
    }
  } else {
    if (T >= 32 && K % 128 == 0 && N % 128 == 0) {
      // 4x4 warps only when it still yields enough blocks to fill the GPU (the smem staging serializes a block)
      if (T >= 128 && (int64_t)cdiv(T, 128) * cdiv(N, 128) >= 128) {
        dim3 grid(cdiv(T, 128), cdiv(N, 128));
        qk::gemm_i8_tc_kernel<4, 4, OutT><<<grid, 512, 0, s>>>(x, w.q8, w.s8, y, T, N, K);
      } else {
        dim3 grid(cdiv(T, 64), cdiv(N, 64));
        qk::gemm_i8_tc_kernel<2, 2, OutT><<<grid, 128, 0, s>>>(x, w.q8, w.s8, y, T, N, K);
      }
    } else if (T >= 8) {
      dim3 grid(cdiv(N, 64), cdiv(T, 32));
      qk::gemm_i8_kernel<OutT><<<grid, 256, 0, s>>>(x, w.q8, w.s8, y, T, N, K);
    } else if (T == 1) {
      qk::gemv_i8_kernel<1, QWEN_GEMV_R, OutT><<<dim3(cdiv(N, 8 * QWEN_GEMV_R), 1), 256, 0, s>>>(x, w.q8, w.s8, y, T, N, K);
    } else if (T == 2) {
      qk::gemv_i8_kernel<2, QWEN_GEMV_R, OutT><<<dim3(cdiv(N, 8 * QWEN_GEMV_R), 1), 256, 0, s>>>(x, w.q8, w.s8, y, T, N, K);
    } else {
      qk::gemv_i8_kernel<4, QWEN_GEMV_R, OutT><<<dim3(cdiv(N, 8 * QWEN_GEMV_R), cdiv(T, 4)), 256, 0, s>>>(x, w.q8, w.s8, y, T, N, K);
    }
  }
}

// Routed experts. Decode (small T): one warp per (token, slot, row), bandwidth-bound. Prefill: group the
// (token, slot) pairs by expert so weights are dequantized once per tile of tokens.
static void moe(const Model& m, Work& w, const Layer& l, int T, cudaStream_t s) {
  const int H = m.H;
  if (T < 8) {
    qk::moe_gate_up_int4_kernel<<<cdiv((int64_t)T * m.TOPK * m.I, 8), 256, 0, s>>>(w.normed, l.exp_gu_q, l.exp_gu_s,
                                                                                    w.topk_i, w.inter, T, m.TOPK, m.I,
                                                                                    H);
    qk::moe_down_int4_kernel<<<cdiv((int64_t)T * H, 8), 256, 0, s>>>(w.inter, l.exp_dn_q, l.exp_dn_s, w.topk_i,
                                                                      w.topk_w, w.routed, T, m.TOPK, H, m.I);
    return;
  }
  const int pairs = T * m.TOPK;
  qk::moe_sort_kernel<<<1, 256, 0, s>>>(w.topk_i, pairs, m.E, w.moe_off, w.moe_sorted);
  qk::moe_gate_up_tile_kernel<QWEN_MOE_TK, QWEN_MOE_NB><<<dim3(m.I / (QWEN_MOE_NB / 2), m.E), QWEN_MOE_NB, 0, s>>>(
      w.normed, l.exp_gu_q, l.exp_gu_s, w.moe_off, w.moe_sorted, w.inter, m.TOPK, m.I, H);
  qk::moe_down_tile_kernel<QWEN_MOE_TK, QWEN_MOE_NB><<<dim3(H / QWEN_MOE_NB, m.E), QWEN_MOE_NB, 0, s>>>(
      w.inter, l.exp_dn_q, l.exp_dn_s, w.moe_off, w.moe_sorted, w.topk_w, w.slot_out, H, m.I);
  qk::moe_slot_reduce_kernel<<<std::min(cdiv((int64_t)T * H, 256), 4096), 256, 0, s>>>(w.slot_out, w.routed, T,
                                                                                        m.TOPK, H);
}

// GDN recurrence: decode variant has no shared-memory staging.
static void gdn_recurrence(const bf16* q, const bf16* k, const bf16* v, const float* alpha, const float* beta, float* S,
                           bf16* o, int T, const Model& m, cudaStream_t s) {
  const dim3 grid(m.Hv * m.Dv / 8);
  if (T == 1) qk::gdn_recurrence_kernel<128, 0><<<grid, 256, 0, s>>>(q, k, v, alpha, beta, S, o, T, m.Hk, m.Hv, m.Dv);
  else qk::gdn_recurrence_kernel<128, 32><<<grid, 256, 0, s>>>(q, k, v, alpha, beta, S, o, T, m.Hk, m.Hv, m.Dv);
}

// One forward pass over T tokens starting at absolute position pos0. Leaves logits of the last token in
// w.logits and its argmax in w.next.
static void forward(const Model& m, Work& w, int T, int pos0, cudaStream_t s) {
  const int H = m.H;
  const int C = 2 * m.Hk * m.Dk + m.Hv * m.Dv;
  const int dq = m.Hk * m.Dk, dk = m.Hk * m.Dk, dv = m.Hv * m.Dv;
  const int warps_per_block = 8;

  qk::embed_kernel<<<T, 256, 0, s>>>(w.ids, m.embed, w.hidden, T, H);
  dbg("embed", w.hidden, (int64_t)T * H);
  qk::rmsnorm_kernel<<<T, 256, 0, s>>>(w.hidden, m.layers[0].rms1, w.normed, H, m.eps);
  dbg("rmsnorm0", w.normed, (int64_t)T * H);

  for (int li = 0; li < m.L; ++li) {
    const Layer& l = m.layers[li];
    if (l.kind == 1) {
      dense(w.normed, l.gdn_qkv, w.qkv, T, s);
      dense(w.normed, l.gdn_z, w.z, T, s);
      if (T == 1) {
        qk::gdn_conv_kernel<true><<<cdiv(C, 256), 256, 0, s>>>(w.qkv, l.gdn_conv, l.conv_state, w.q, w.k, w.v, T, C, dq, dk);
      } else {
        qk::gdn_conv_kernel<false><<<std::min(cdiv((int64_t)T * C, 256), 4096), 256, 0, s>>>(
            w.qkv, l.gdn_conv, l.conv_state, w.q, w.k, w.v, T, C, dq, dk);
        qk::gdn_conv_state_kernel<<<cdiv(C, 256), 256, 0, s>>>(w.qkv, l.conv_state, T, C);
      }
      const int prep_jobs = 2 * T * m.Hk + T * m.Hv;
      qk::gdn_prep_kernel<<<cdiv(prep_jobs, warps_per_block), 256, 0, s>>>(w.q, w.k, T, m.Hk, m.Dk, w.normed, l.gdn_ab,
                                                                          l.gdn_A_log, l.gdn_dt, w.alpha, w.beta, m.Hv, H,
                                                                          1e-6f);
      gdn_recurrence(w.q, w.k, w.v, w.alpha, w.beta, l.S, w.o, T, m, s);
      qk::gdn_gated_norm_kernel<<<cdiv(T * m.Hv, warps_per_block), 256, 0, s>>>(w.o, l.gdn_norm, w.z, T * m.Hv, m.Dv,
                                                                                m.eps);
      if (g_debug && li < 4) {
        dbg("gdn.qkv", w.qkv, (int64_t)T * C); dbg("gdn.q(conv,norm)", w.q, (int64_t)T * dq);
        dbg("gdn.k", w.k, (int64_t)T * dk); dbg("gdn.v", w.v, (int64_t)T * dv);
        dbg("gdn.alpha", w.alpha, (int64_t)T * m.Hv); dbg("gdn.beta", w.beta, (int64_t)T * m.Hv);
        dbg("gdn.o(normed)", w.o, (int64_t)T * dv);
      }
      dense(w.o, l.gdn_out, w.attn_out, T, s);
    } else {
      dense(w.normed, l.attn_qkv, w.qkv, T, s);
      if (m.attn_gate) dense(w.normed, l.attn_gate, w.z, T, s);
      const int heads = m.Hq + 2 * m.Hkv;
      qk::attn_qk_norm_rope_kernel<<<cdiv(T * heads, warps_per_block), 256, 0, s>>>(
          w.qkv, l.q_norm, l.k_norm, w.q, l.kc, l.vc, T, m.Hq, m.Hkv, m.D, m.rope_dim, m.theta, pos0, m.ctx, m.eps);
      attention(w.q, l.kc, l.vc, w.attn_part, w.o, T, m.Hq, m.Hkv, m.D, m.ctx, pos0, s);
      if (m.attn_gate)
        qk::sigmoid_gate_kernel<<<std::min(cdiv((int64_t)T * m.Hq * m.D, 256), 4096), 256, 0, s>>>(
            w.o, w.z, (int64_t)T * m.Hq * m.D);
      if (g_debug && li < 4) {
        dbg("attn.qkv", w.qkv, (int64_t)T * (m.Hq + 2 * m.Hkv) * m.D); dbg("attn.q", w.q, (int64_t)T * m.Hq * m.D);
        dbg("attn.o(gated)", w.o, (int64_t)T * m.Hq * m.D);
      }
      dense(w.o, l.attn_o, w.attn_out, T, s);
    }
    if (g_debug && li < 4) dbg("attn_out", w.attn_out, (int64_t)T * H);
    qk::residual_rmsnorm_kernel<bf16><<<T, 256, 0, s>>>(w.hidden, w.attn_out, l.rms2, w.normed, H, m.eps);

    // MoE
    Linear router; router.w = l.router; router.N = m.E; router.K = H;
    dense(w.normed, router, w.logits_r, T, s);
    qk::router_topk_kernel<<<cdiv(T, warps_per_block), 256, 0, s>>>(w.logits_r, w.topk_i, w.topk_w, T, m.E, m.TOPK);
    moe(m, w, l, T, s);
    dense(w.normed, l.shared_gate_up, w.sgu, T, s);
    qk::silu_mul_kernel<<<std::min(cdiv((int64_t)T * m.Is, 256), 4096), 256, 0, s>>>(w.sgu, w.sinter, T, m.Is);
    dense(w.sinter, l.shared_down, w.sout, T, s);
    if (g_debug && li < 4) {
      dbg("router.logits", w.logits_r, (int64_t)T * m.E); dbg("topk.w", w.topk_w, (int64_t)T * m.TOPK);
      dbg("moe.inter", w.inter, (int64_t)T * m.TOPK * m.I); dbg("moe.routed", w.routed, (int64_t)T * H);
      dbg("shared.out", w.sout, (int64_t)T * H);
    }
    if (li + 1 < m.L)
      qk::shared_gate_combine_rmsnorm_kernel<<<T, 256, 0, s>>>(w.hidden, w.routed, w.sout, w.normed, l.shared_gate,
                                                               m.layers[li + 1].rms1, w.normed, H, m.eps);
    else
      qk::shared_gate_combine_kernel<<<T, 256, 0, s>>>(w.hidden, w.routed, w.sout, w.normed, l.shared_gate, H);
    if (g_debug) { char nm[64]; snprintf(nm, sizeof nm, "hidden after L%d", li); dbg(nm, w.hidden, (int64_t)T * H); }
  }
  // final norm on the last token only, then lm_head + argmax
  qk::rmsnorm_kernel<<<1, 256, 0, s>>>(w.hidden + (int64_t)(T - 1) * H, m.final_norm, w.normed, H, m.eps);
  dense(w.normed, m.lm_head, w.logits, 1, s);
  dbg("logits", w.logits, m.V);
  qk::argmax_partial_kernel<<<1024, 256, 0, s>>>(w.logits, m.V, w.pmax, w.pidx);
  qk::argmax_final_kernel<<<1, 256, 0, s>>>(w.pmax, w.pidx, 1024, w.next);
}


// ------------------------------------------------------------------ --bench: per-kernel decode roofline

// Bytes of weights a decode step (T = 1) streams: the bandwidth-bound floor for one token.
static double decode_weight_bytes(const Model& m) {
  auto lin = [](const Linear& l) -> double { return l.w ? 2.0 * l.N * l.K : 1.0 * l.N * l.K + 2.0 * l.N * (l.K / 128); };
  double b = 0;
  for (const Layer& l : m.layers) {
    if (l.kind == 1) b += lin(l.gdn_qkv) + lin(l.gdn_z) + lin(l.gdn_out) + 2.0 * 2 * m.Hv * m.H + 2.0 * m.Hv * m.Dv * 4 + 2.0 * m.Hv * m.Dv;
    else b += lin(l.attn_qkv) + (m.attn_gate ? lin(l.attn_gate) : 0) + lin(l.attn_o);
    b += 2.0 * m.E * m.H;                                                       // router
    b += (double)m.TOPK * (2.0 * m.I * m.H / 2 + m.H * m.I / 2 + 2.0 * (2.0 * m.I * (m.H / 128) + m.H * (m.I / 128)));  // int4 experts
    b += lin(l.shared_gate_up) + lin(l.shared_down) + 2.0 * m.H;
    b += 4.0 * m.Hv * m.Dv * m.Dk;  // GDN state read + write (fp32) counted once each below
    b += 2.0 * 2 * m.H;             // norms
  }
  b += lin(m.lm_head) + 2.0 * m.H;
  return b;
}

#if defined(__MVCC__)
extern "C" void mvccGetStats(unsigned long long* launches, unsigned long long* cbs, unsigned long long* compiles);
static unsigned long long launches() { unsigned long long l = 0, c = 0, k = 0; mvccGetStats(&l, &c, &k); return l; }
#else
static unsigned long long launches() { return 0; }
#endif

template <typename F>
static double time_ms(F&& f, int reps, cudaStream_t s) {
  cudaEvent_t e0, e1;
  CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
  f(); CK(cudaStreamSynchronize(s));  // warm
  CK(cudaEventRecord(e0, s));
  for (int i = 0; i < reps; ++i) f();
  CK(cudaEventRecord(e1, s));
  CK(cudaEventSynchronize(e1));
  float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
  CK(cudaEventDestroy(e0)); CK(cudaEventDestroy(e1));
  return ms / reps;
}

static void bench(const Model& m, Work& w, cudaStream_t s) {
  const int reps = 50;
  const Layer& g = m.layers[0];
  const Layer* a = nullptr;
  for (const Layer& l : m.layers) if (l.kind == 0) { a = &l; break; }
  const int C = 2 * m.Hk * m.Dk + m.Hv * m.Dv;
  auto row = [&](const char* name, double ms, double bytes) {
    fprintf(stderr, "  %-34s %8.3f ms  %7.1f GB/s  (%.1f MB)\n", name, ms, bytes / (ms * 1e-3) / 1e9, bytes / 1e6);
  };
  auto lb = [](const Linear& l) -> double { return l.w ? 2.0 * l.N * l.K : 1.0 * l.N * l.K + 2.0 * l.N * (l.K / 128); };
  fprintf(stderr, "[bench] dense projections stored as %s\n", g.gdn_qkv.w ? "bf16" : "int8 (group-128 scales)");
  fprintf(stderr, "[bench] decode kernels, T=1, %d reps each (bandwidth = weight bytes / time)\n", reps);
  double ms;
  ms = time_ms([&] { dense(w.normed, g.gdn_qkv, w.qkv, 1, s); }, reps, s); row("gdn qkv proj (dense gemv)", ms, lb(g.gdn_qkv));
  ms = time_ms([&] { dense(w.normed, g.gdn_z, w.z, 1, s); }, reps, s); row("gdn z proj (dense gemv)", ms, lb(g.gdn_z));
  ms = time_ms([&] { dense(w.o, g.gdn_out, w.attn_out, 1, s); }, reps, s); row("gdn out proj (dense gemv)", ms, lb(g.gdn_out));
  ms = time_ms([&] { gdn_recurrence(w.q, w.k, w.v, w.alpha, w.beta, g.S, w.o, 1, m, s); }, reps, s);
  row("gdn recurrence (fp32 state r+w)", ms, 8.0 * m.Hv * m.Dv * m.Dk);
  if (a) {
    ms = time_ms([&] { dense(w.normed, a->attn_qkv, w.qkv, 1, s); }, reps, s); row("attn qkv proj (dense gemv)", ms, lb(a->attn_qkv));
    ms = time_ms([&] { attention(w.q, a->kc, a->vc, w.attn_part, w.o, 1, m.Hq, m.Hkv, m.D, m.ctx, 1023, s); }, reps, s);
    row("attn decode @1024 keys (KV bf16)", ms, 2.0 * 2 * m.Hkv * 1024 * m.D);
    ms = time_ms([&] { attention(w.q, a->kc, a->vc, w.attn_part, w.o, 1, m.Hq, m.Hkv, m.D, m.ctx, 4095, s); }, reps, s);
    row("attn decode @4096 keys (KV bf16)", ms, 2.0 * 2 * m.Hkv * 4096 * m.D);
  }
  Linear router; router.w = g.router; router.N = m.E; router.K = m.H;
  ms = time_ms([&] { dense(w.normed, router, w.logits_r, 1, s); }, reps, s); row("router (bf16 gemv)", ms, 2.0 * m.E * m.H);
  ms = time_ms([&] { qk::moe_gate_up_int4_kernel<<<cdiv((int64_t)m.TOPK * m.I, 8), 256, 0, s>>>(w.normed, g.exp_gu_q, g.exp_gu_s, w.topk_i, w.inter, 1, m.TOPK, m.I, m.H); }, reps, s);
  row("moe gate/up (int4, 8 experts)", ms, (double)m.TOPK * (2.0 * m.I * m.H / 2 + 2.0 * 2 * m.I * (m.H / 128)));
  ms = time_ms([&] { qk::moe_down_int4_kernel<<<cdiv((int64_t)m.H, 8), 256, 0, s>>>(w.inter, g.exp_dn_q, g.exp_dn_s, w.topk_i, w.topk_w, w.routed, 1, m.TOPK, m.H, m.I); }, reps, s);
  row("moe down (int4, 8 experts)", ms, (double)m.TOPK * (1.0 * m.H * m.I / 2 + 2.0 * m.H * (m.I / 128)));
  ms = time_ms([&] { dense(w.normed, g.shared_gate_up, w.sgu, 1, s); }, reps, s); row("shared gate/up (dense gemv)", ms, lb(g.shared_gate_up));
  ms = time_ms([&] { dense(w.normed, m.lm_head, w.logits, 1, s); }, reps, s); row("lm_head", ms, lb(m.lm_head));
  ms = time_ms([&] { qk::argmax_partial_kernel<<<1024, 256, 0, s>>>(w.logits, m.V, w.pmax, w.pidx); qk::argmax_final_kernel<<<1, 256, 0, s>>>(w.pmax, w.pidx, 1024, w.next); }, reps, s);
  row("argmax over vocab", ms, 4.0 * m.V);
  unsigned long long l0 = launches();
  ms = time_ms([&] { forward(m, w, 1, 64, s); }, 10, s);
  const unsigned long long per_step = (launches() - l0) / 11;
  const double bytes = decode_weight_bytes(m);
  fprintf(stderr, "  %-34s %8.3f ms  %7.1f GB/s  (%.2f GB weights/token, %llu launches)\n", "full decode step (forward T=1)", ms,
          bytes / (ms * 1e-3) / 1e9, bytes / 1e9, per_step);

  // prefill: same kernels at T = maxT (compute-bound; report GFLOP/s)
  const int T = w.maxT;
  if (T < 8) return;
  auto frow = [&](const char* name, double ms, double flops) {
    fprintf(stderr, "  %-34s %8.3f ms  %7.1f GFLOP/s\n", name, ms, flops / (ms * 1e-3) / 1e9);
  };
  fprintf(stderr, "[bench] prefill kernels, T=%d\n", T);
  ms = time_ms([&] { dense(w.normed, g.gdn_qkv, w.qkv, T, s); }, 10, s); frow("gdn qkv proj (dense gemm)", ms, 2.0 * T * C * m.H);
  ms = time_ms([&] { dense(w.o, g.gdn_out, w.attn_out, T, s); }, 10, s); frow("gdn out proj (dense gemm)", ms, 2.0 * T * m.H * m.Hv * m.Dv);
  ms = time_ms([&] { qk::gdn_conv_kernel<false><<<std::min(cdiv((int64_t)T * C, 256), 4096), 256, 0, s>>>(w.qkv, g.gdn_conv, g.conv_state, w.q, w.k, w.v, T, C, m.Hk * m.Dk, m.Hk * m.Dk); }, 10, s);
  frow("gdn conv1d+silu", ms, 8.0 * T * C);
  ms = time_ms([&] { gdn_recurrence(w.q, w.k, w.v, w.alpha, w.beta, g.S, w.o, T, m, s); }, 10, s);
  frow("gdn recurrence (sequential T)", ms, 6.0 * T * m.Hv * m.Dv * m.Dk);
  if (a) {
    ms = time_ms([&] { attention(w.q, a->kc, a->vc, w.attn_part, w.o, T, m.Hq, m.Hkv, m.D, m.ctx, 0, s); }, 10, s);
    frow("attention prefill (causal)", ms, 2.0 * 2 * m.Hq * m.D * (double)T * (T + 1) / 2);
  }
  // realistic routing for the MoE benchmark: route the bench tokens through the real router on real activations
  {
    std::vector<int> ids(T);
    for (int t = 0; t < T; ++t) ids[t] = 1000 + t * 37;
    CK(cudaMemcpyAsync(w.ids, ids.data(), T * sizeof(int), cudaMemcpyHostToDevice, s));
    qk::embed_kernel<<<T, 256, 0, s>>>(w.ids, m.embed, w.hidden, T, m.H);
    qk::rmsnorm_kernel<<<T, 256, 0, s>>>(w.hidden, g.rms2, w.normed, m.H, m.eps);
    dense(w.normed, router, w.logits_r, T, s);
    qk::router_topk_kernel<<<cdiv(T, 8), 256, 0, s>>>(w.logits_r, w.topk_i, w.topk_w, T, m.E, m.TOPK);
  }
  ms = time_ms([&] { moe(m, w, g, T, s); }, 10, s);
  frow("moe gate/up + down (int4, tiles+TC)", ms, 2.0 * T * m.TOPK * 3 * m.I * m.H);
  ms = time_ms([&] { dense(w.normed, g.shared_gate_up, w.sgu, T, s); }, 10, s); frow("shared gate/up (dense gemm)", ms, 2.0 * T * 2 * m.Is * m.H);
  ms = time_ms([&] { forward(m, w, T, 0, s); }, 3, s);
  fprintf(stderr, "  %-34s %8.3f ms  %7.1f tok/s\n", "full prefill step (forward)", ms, T / (ms * 1e-3));
}

// ------------------------------------------------------------------ main

static void usage() {
  fprintf(stderr,
          "usage: qwen --pack model.mvccq --tokenizer tokenizer.bin [--prompt TEXT | --raw TEXT] [--max-tokens N]\n"
          "            [--ctx N] [--chunk N] [--think] [--system TEXT] [--prefetch] [--verbose]\n");
}

int main(int argc, char** argv) {
  std::string pack_path, tok_path, prompt, raw, system;
  int max_tokens = 256, ctx = 4096, chunk = 128;
  bool think = false, prefetch = false, verbose = false, do_bench = false;
  for (int i = 1; i < argc; ++i) {
    std::string a = argv[i];
    auto next = [&]() -> std::string { if (i + 1 >= argc) { usage(); exit(2); } return argv[++i]; };
    if (a == "--pack") pack_path = next();
    else if (a == "--tokenizer") tok_path = next();
    else if (a == "--prompt") prompt = next();
    else if (a == "--raw") raw = next();
    else if (a == "--system") system = next();
    else if (a == "--max-tokens") max_tokens = atoi(next().c_str());
    else if (a == "--ctx") ctx = atoi(next().c_str());
    else if (a == "--chunk") chunk = atoi(next().c_str());
    else if (a == "--think") think = true;
    else if (a == "--prefetch") prefetch = true;
    else if (a == "--verbose") verbose = true;
    else if (a == "--debug") g_debug = true;
    else if (a == "--bench") do_bench = true;
    else { usage(); return 2; }
  }
  if (pack_path.empty() || tok_path.empty() || (prompt.empty() && raw.empty() && !do_bench)) { usage(); return 2; }

  auto t_load0 = std::chrono::steady_clock::now();
  Model m;
  m.load(pack_path, ctx);
  Tokenizer tok;
  tok.load(tok_path);
  if (prefetch) m.pack.prefetch();
  auto t_load1 = std::chrono::steady_clock::now();
  if (verbose)
    fprintf(stderr, "[load] %.2f GB pack, %d layers, ctx %d in %.2fs\n", m.pack.size() / 1e9, m.L, ctx,
            std::chrono::duration<double>(t_load1 - t_load0).count());

  std::string text = raw;
  if (text.empty() && !prompt.empty()) {
    if (!system.empty()) text += "<|im_start|>system\n" + system + "<|im_end|>\n";
    text += "<|im_start|>user\n" + prompt + "<|im_end|>\n<|im_start|>assistant\n";
    text += think ? "<think>\n" : "<think>\n\n</think>\n\n";
  }
  std::vector<int> ids = tok.encode(text);
  if (verbose) {
    fprintf(stderr, "[prompt] %zu tokens:", ids.size());
    for (size_t i = 0; i < std::min<size_t>(ids.size(), 64); ++i) fprintf(stderr, " %d", ids[i]);
    fprintf(stderr, "\n");
  }
  if ((int)ids.size() + max_tokens > ctx) { fprintf(stderr, "prompt + max_tokens exceeds --ctx %d\n", ctx); return 1; }

  Work w;
  w.alloc(m, std::max(chunk, 1));
  m.reset_state();
  cudaStream_t s = 0;
  if (do_bench) {
    CK(cudaMemsetAsync(w.topk_i, 0, m.TOPK * sizeof(int), s));
    CK(cudaMemsetAsync(w.ids, 0, sizeof(int), s));
    bench(m, w, s);
    if (prompt.empty() && raw.empty()) return 0;
  }

  // prefill in chunks
  auto t0 = std::chrono::steady_clock::now();
  int pos = 0;
  int next_tok = -1;
  for (size_t off = 0; off < ids.size(); off += chunk) {
    const int T = (int)std::min<size_t>(chunk, ids.size() - off);
    CK(cudaMemcpyAsync(w.ids, ids.data() + off, T * sizeof(int), cudaMemcpyHostToDevice, s));
    forward(m, w, T, pos, s);
    pos += T;
  }
  CK(cudaMemcpy(&next_tok, w.next, sizeof(int), cudaMemcpyDeviceToHost));
  CK(cudaGetLastError());
  auto t1 = std::chrono::steady_clock::now();
  const double prefill_s = std::chrono::duration<double>(t1 - t0).count();

  // decode
  Utf8Printer out;
  int produced = 0;
  std::vector<int> generated;
  auto is_eos = [&](int t) { return std::find(m.eos.begin(), m.eos.end(), t) != m.eos.end(); };
  while (produced < max_tokens && !is_eos(next_tok)) {
    generated.push_back(next_tok);
    out.feed(tok.decode(next_tok));
    ++produced;
    CK(cudaMemcpyAsync(w.ids, &next_tok, sizeof(int), cudaMemcpyHostToDevice, s));
    forward(m, w, 1, pos, s);
    pos += 1;
    CK(cudaMemcpy(&next_tok, w.next, sizeof(int), cudaMemcpyDeviceToHost));
  }
  auto t2 = std::chrono::steady_clock::now();
  const double decode_s = std::chrono::duration<double>(t2 - t1).count();
  printf("\n");
  fprintf(stderr, "\n[stats] prefill %zu tok in %.2fs (%.1f tok/s) | decode %d tok in %.2fs (%.2f tok/s)%s\n",
          ids.size(), prefill_s, ids.size() / prefill_s, produced, decode_s, produced / std::max(decode_s, 1e-9),
          is_eos(next_tok) ? " | eos" : "");
  return 0;
}
