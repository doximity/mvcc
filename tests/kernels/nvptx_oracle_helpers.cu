// nvptx-oracle target runtime for STIR tensor-op helpers (engine::nvptxOracle).
// Metal realizes matmul2d as cooperative tensors (slot law + acc2ptx). This target
// realizes the same calls as mma.sync; C stays in fragment layout, so acc2ptx is identity.
// 16×16×32 (hgemm16) still unpacks slots with the cooperative-tensor law.
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ unsigned law_row(unsigned l, unsigned s, bool tr, int R, int C) {
  unsigned br = ((l >> 1) & 3u) | ((l >> 4) << 2) | (((s >> 2) & 1u) << 3);
  unsigned bc = (s & 3u) | ((l & 1u) << 2) | (((l >> 3) & 1u) << 3);
  unsigned row = tr ? bc : br;
  if (C == 32) { if (R == 32) row |= ((s >> 4) & 1u) << 4; }
  else if (R == 32) row |= ((s >> 3) & 1u) << 4;
  return row;
}
__device__ __forceinline__ unsigned law_col(unsigned l, unsigned s, bool tr, int R, int C) {
  unsigned br = ((l >> 1) & 3u) | ((l >> 4) << 2) | (((s >> 2) & 1u) << 3);
  unsigned bc = (s & 3u) | ((l & 1u) << 2) | (((l >> 3) & 1u) << 3);
  unsigned col = tr ? br : bc;
  if (C == 32) col |= ((s >> 3) & 1u) << 4;
  return col;
}
__device__ __forceinline__ float u16_f16(unsigned short h) {
  return __half2float(__ushort_as_half(h));
}
__device__ __forceinline__ float u16_bf16(unsigned short h) {
  union { unsigned u; float f; } x; x.u = (unsigned)h << 16; return x.f;
}

__device__ __align__(16) unsigned char __mvcc_zero_storage[4096] = {};
extern "C" __device__ void* __mvcc_zero_page() { return __mvcc_zero_storage; }

extern "C" __device__ void __mvcc_cp_async_zsel_16(unsigned dst, const void* src, bool valid) {
  const void* p = valid ? src : (const void*)__mvcc_zero_storage;
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(dst), "l"(p));
}

struct W4 { unsigned x, y, z, w; };

extern "C" __device__ __attribute__((convergent)) W4 __mvcc_tp_frag2slot(int kind, unsigned f0, unsigned f1, unsigned f2, unsigned f3) {
  unsigned lane = threadIdx.x & 31u;
  const bool hi = ((lane >> 3) & 1u) != 0u;
  const unsigned s0 = 4u * (((lane >> 1) & 3u) | ((lane >> 4) << 2)) + 2u * (lane & 1u), s1 = s0 + 1;
  unsigned p0 = __shfl_sync(0xffffffffu, f0, s0), p1 = __shfl_sync(0xffffffffu, f1, s0);
  unsigned p2 = __shfl_sync(0xffffffffu, f2, s0), p3 = __shfl_sync(0xffffffffu, f3, s0);
  unsigned q0 = __shfl_sync(0xffffffffu, f0, s1), q1 = __shfl_sync(0xffffffffu, f1, s1);
  unsigned q2 = __shfl_sync(0xffffffffu, f2, s1), q3 = __shfl_sync(0xffffffffu, f3, s1);
  W4 o;
  if (kind == 0) { o.x = hi ? p2 : p0; o.y = hi ? q2 : q0; o.z = hi ? p3 : p1; o.w = hi ? q3 : q1; }
  else           { o.x = hi ? p1 : p0; o.y = hi ? q1 : q0; o.z = hi ? p3 : p2; o.w = hi ? q3 : q2; }
  return o;
}

// Ground-truth fill: the source kernel's ldmatrix.x4 + the same frag2slot law.
// Every recovered gemm_ptx call is trans=0 blockmap=0xE4 (identity).
extern "C" __device__ __attribute__((convergent)) W4 __mvcc_tp_ld(unsigned addr, int kind, int trans, int /*blockmap*/) {
  unsigned r0, r1, r2, r3;
  if (trans) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
  } else {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
  }
  return __mvcc_tp_frag2slot(kind, r0, r1, r2, r3);
}

// Scratch lives in dynamic shared ABOVE the kernel's staging (smem_raw). File-scope
// .shared would sit at offset 0 and move smem_raw, which breaks tp_ld.
extern __shared__ char __mvcc_scratch[];
enum { kMmaOff = 65536, kWarpBytes = 2 * 32 * 32 * (int)sizeof(float) };

// Inverse of frag2slot: slot words of one 16×16 block → the ldmatrix.x4 / mma.sync fragments.
__device__ __forceinline__ __attribute__((convergent)) W4 slot2frag(int kind, unsigned w0, unsigned w1, unsigned w2, unsigned w3) {
  const unsigned D = threadIdx.x & 31u, even = D & ~1u;
  const unsigned L0 = ((even >> 4) & 1u) * 16u + ((even >> 2) & 3u) * 2u + ((even >> 1) & 1u);
  const unsigned L1 = L0 + 8u, odd = D & 1u;
  const unsigned a0 = __shfl_sync(0xffffffffu, w0, L0), a1 = __shfl_sync(0xffffffffu, w1, L0);
  const unsigned a2 = __shfl_sync(0xffffffffu, w2, L0), a3 = __shfl_sync(0xffffffffu, w3, L0);
  const unsigned b0 = __shfl_sync(0xffffffffu, w0, L1), b1 = __shfl_sync(0xffffffffu, w1, L1);
  const unsigned b2 = __shfl_sync(0xffffffffu, w2, L1), b3 = __shfl_sync(0xffffffffu, w3, L1);
  W4 f;
  if (kind == 0) { f.x = odd ? a1 : a0; f.y = odd ? a3 : a2; f.z = odd ? b1 : b0; f.w = odd ? b3 : b2; }
  else           { f.x = odd ? a1 : a0; f.z = odd ? a3 : a2; f.y = odd ? b1 : b0; f.w = odd ? b3 : b2; }
  return f;
}

__device__ __forceinline__ void mma16(float& d0, float& d1, float& d2, float& d3,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned b0, unsigned b1, int type) {
  if (type) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  } else {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3) : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  }
}

template <int M, int N, int Cap>
__device__ void acc2ptx(float* out, const float* slots) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned g = lane >> 2, t = lane & 3;
  for (int p = 0; p < Cap; p++) {
    const int tile = p >> 2, r = p & 3, mt = tile / (N / 8), nt = tile % (N / 8);
    const unsigned row = 16u * (unsigned)mt + g + 8u * (unsigned)(r >> 1);
    const unsigned col = 8u * (unsigned)nt + 2u * t + (unsigned)(r & 1);
    float v = 0.f;
    for (int s = 0; s < Cap; s++) {
      if (law_row(lane, (unsigned)s, false, M, N) == row && law_col(lane, (unsigned)s, false, M, N) == col)
        v = slots[s];
    }
    out[p] = v;
  }
}

template <int M, int N, int Cap>
__device__ void ptx2acc(float* slots, const float* ptx) {
  const unsigned lane = threadIdx.x & 31u;
  const unsigned g = lane >> 2, t = lane & 3;
  for (int s = 0; s < Cap; s++) {
    const unsigned row = law_row(lane, (unsigned)s, false, M, N);
    const unsigned col = law_col(lane, (unsigned)s, false, M, N);
    float v = 0.f;
    for (int p = 0; p < Cap; p++) {
      const int tile = p >> 2, r = p & 3, mt = tile / (N / 8), nt = tile % (N / 8);
      const unsigned prow = 16u * (unsigned)mt + g + 8u * (unsigned)(r >> 1);
      const unsigned pcol = 8u * (unsigned)nt + 2u * t + (unsigned)(r & 1);
      if (prow == row && pcol == col) v = ptx[p];
    }
    slots[s] = v;
  }
}

// 32×N×32: invert frag2slot per 16×16 block, mma.sync.m16n8k16.
// On NVIDIA the recovered "slots" are kept in mma.sync fragment (PTX) order; acc2ptx is identity.
template <int N>
__device__ __attribute__((convergent)) void mma_tc32(float* d, const unsigned* a, const unsigned* b, const float* c, int type) {
  constexpr int Cap = 32 * N / 32, n8 = N / 8, nN = N / 16;
  float ptx[Cap];
  for (int i = 0; i < Cap; i++) ptx[i] = c[i];
  for (int mb = 0; mb < 2; mb++)
    for (int kb = 0; kb < 2; kb++) {
      const int ab = 4 * (kb + 2 * mb);
      W4 Af = slot2frag(0, a[ab], a[ab + 1], a[ab + 2], a[ab + 3]);
      for (int nb = 0; nb < nN; nb++) {
        const int bb = 4 * (nb + nN * kb);
        W4 Bf = slot2frag(1, b[bb], b[bb + 1], b[bb + 2], b[bb + 3]);
        for (int h = 0; h < 2; h++) {
          const int tile = mb * n8 + 2 * nb + h;
          mma16(ptx[4 * tile], ptx[4 * tile + 1], ptx[4 * tile + 2], ptx[4 * tile + 3],
                Af.x, Af.y, Af.z, Af.w, h ? Bf.z : Bf.x, h ? Bf.w : Bf.y, type);
        }
      }
    }
  for (int i = 0; i < Cap; i++) d[i] = ptx[i];
}

template <int M, int N, int K>
__device__ void mma_run(float* d, const unsigned* a, const unsigned* b, const float* c, int type, int tl, int tr) {
  if (M == 32 && K == 32 && (N == 32 || N == 16)) { mma_tc32<N>(d, a, b, c, type); return; }
  const unsigned lane = threadIdx.x & 31u;
  const int w = (int)(threadIdx.x >> 5);
  const int nA = M * K / 64, cap = M * N / 32;
  float* As = (float*)(__mvcc_scratch + kMmaOff + w * kWarpBytes);
  float* Bs = As + 32 * 32;
  auto at = [&](float* t, unsigned r, unsigned col) -> float& { return t[r * 32u + col]; };
  auto unpack = [&](unsigned word, int s0, bool atr, int R, int C, float* dest) {
    unsigned lo = word & 0xffffu, hi = word >> 16;
    float x = type ? u16_bf16((unsigned short)lo) : u16_f16((unsigned short)lo);
    float y = type ? u16_bf16((unsigned short)hi) : u16_f16((unsigned short)hi);
    at(dest, law_row(lane, (unsigned)s0, atr, R, C), law_col(lane, (unsigned)s0, atr, R, C)) = x;
    at(dest, law_row(lane, (unsigned)s0 + 1u, atr, R, C), law_col(lane, (unsigned)s0 + 1u, atr, R, C)) = y;
  };
  for (int i = 0; i < nA; i++) unpack(a[i], 2 * i, tl != 0, M, K, As);
  for (int i = 0; i < K * N / 64; i++) unpack(b[i], 2 * i, tr != 0, K, N, Bs);
  __syncwarp();
  for (int s = 0; s < cap; s++) {
    unsigned row = law_row(lane, (unsigned)s, false, M, N);
    unsigned col = law_col(lane, (unsigned)s, false, M, N);
    float acc = c[s];
    for (int k = 0; k < K; k++) acc += at(As, row, (unsigned)k) * at(Bs, (unsigned)k, col);
    d[s] = acc;
  }
  __syncwarp();
}

struct Acc32 { float f[32]; };
struct Acc16 { float f[16]; };
struct Acc8 { float f[8]; };

extern "C" __device__ __attribute__((convergent)) Acc8 __mvcc_tp_mma_16x16x32(
    int, int, int, int type, int tl, int tr,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned a4, unsigned a5, unsigned a6, unsigned a7,
    unsigned b0, unsigned b1, unsigned b2, unsigned b3, unsigned b4, unsigned b5, unsigned b6, unsigned b7,
    float c0, float c1, float c2, float c3, float c4, float c5, float c6, float c7) {
  unsigned a[8] = {a0,a1,a2,a3,a4,a5,a6,a7};
  unsigned b[8] = {b0,b1,b2,b3,b4,b5,b6,b7};
  float c[8] = {c0,c1,c2,c3,c4,c5,c6,c7};
  Acc8 d; mma_run<16, 16, 32>(d.f, a, b, c, type, tl, tr); return d;
}

extern "C" __device__ __attribute__((convergent)) Acc32 __mvcc_tp_mma_32x32x32(
    int, int, int, int type, int tl, int tr,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned a4, unsigned a5, unsigned a6, unsigned a7,
    unsigned a8, unsigned a9, unsigned a10, unsigned a11, unsigned a12, unsigned a13, unsigned a14, unsigned a15,
    unsigned b0, unsigned b1, unsigned b2, unsigned b3, unsigned b4, unsigned b5, unsigned b6, unsigned b7,
    unsigned b8, unsigned b9, unsigned b10, unsigned b11, unsigned b12, unsigned b13, unsigned b14, unsigned b15,
    float c0, float c1, float c2, float c3, float c4, float c5, float c6, float c7,
    float c8, float c9, float c10, float c11, float c12, float c13, float c14, float c15,
    float c16, float c17, float c18, float c19, float c20, float c21, float c22, float c23,
    float c24, float c25, float c26, float c27, float c28, float c29, float c30, float c31) {
  unsigned a[16] = {a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15};
  unsigned b[16] = {b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15};
  float c[32] = {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15,c16,c17,c18,c19,c20,c21,c22,c23,c24,c25,c26,c27,c28,c29,c30,c31};
  Acc32 d; mma_run<32, 32, 32>(d.f, a, b, c, type, tl, tr); return d;
}

extern "C" __device__ __attribute__((convergent)) Acc16 __mvcc_tp_mma_32x16x32(
    int, int, int, int type, int tl, int tr,
    unsigned a0, unsigned a1, unsigned a2, unsigned a3, unsigned a4, unsigned a5, unsigned a6, unsigned a7,
    unsigned a8, unsigned a9, unsigned a10, unsigned a11, unsigned a12, unsigned a13, unsigned a14, unsigned a15,
    unsigned b0, unsigned b1, unsigned b2, unsigned b3, unsigned b4, unsigned b5, unsigned b6, unsigned b7,
    float c0, float c1, float c2, float c3, float c4, float c5, float c6, float c7,
    float c8, float c9, float c10, float c11, float c12, float c13, float c14, float c15) {
  unsigned a[16] = {a0,a1,a2,a3,a4,a5,a6,a7,a8,a9,a10,a11,a12,a13,a14,a15};
  unsigned b[8] = {b0,b1,b2,b3,b4,b5,b6,b7};
  float c[16] = {c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15};
  Acc16 d; mma_run<32, 16, 32>(d.f, a, b, c, type, tl, tr); return d;
}

extern "C" __device__ Acc32 __mvcc_tp_acc2ptx_32(int, int, float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7,
    float s8, float s9, float s10, float s11, float s12, float s13, float s14, float s15,
    float s16, float s17, float s18, float s19, float s20, float s21, float s22, float s23,
    float s24, float s25, float s26, float s27, float s28, float s29, float s30, float s31) {
  Acc32 o; o.f[0]=s0;o.f[1]=s1;o.f[2]=s2;o.f[3]=s3;o.f[4]=s4;o.f[5]=s5;o.f[6]=s6;o.f[7]=s7;
  o.f[8]=s8;o.f[9]=s9;o.f[10]=s10;o.f[11]=s11;o.f[12]=s12;o.f[13]=s13;o.f[14]=s14;o.f[15]=s15;
  o.f[16]=s16;o.f[17]=s17;o.f[18]=s18;o.f[19]=s19;o.f[20]=s20;o.f[21]=s21;o.f[22]=s22;o.f[23]=s23;
  o.f[24]=s24;o.f[25]=s25;o.f[26]=s26;o.f[27]=s27;o.f[28]=s28;o.f[29]=s29;o.f[30]=s30;o.f[31]=s31;
  return o;
}

extern "C" __device__ Acc16 __mvcc_tp_acc2ptx_16(int, int, float s0, float s1, float s2, float s3, float s4, float s5, float s6, float s7,
    float s8, float s9, float s10, float s11, float s12, float s13, float s14, float s15) {
  Acc16 o; o.f[0]=s0;o.f[1]=s1;o.f[2]=s2;o.f[3]=s3;o.f[4]=s4;o.f[5]=s5;o.f[6]=s6;o.f[7]=s7;
  o.f[8]=s8;o.f[9]=s9;o.f[10]=s10;o.f[11]=s11;o.f[12]=s12;o.f[13]=s13;o.f[14]=s14;o.f[15]=s15;
  return o;
}
