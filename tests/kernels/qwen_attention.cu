// examples/qwen decode attention + GDN through check+emit.
// No P·V special case: online (max,sum,acc) and sequential chunk recurrence.
#include "../../examples/qwen/kernels.cuh"
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { printf("%s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)
static inline int cdiv(int a, int b) { return (a + b - 1) / b; }

static float bf(uint16_t b) { uint32_t u = (uint32_t)b << 16; float f; memcpy(&f, &u, 4); return f; }
static uint16_t tobf(float f) { uint32_t u; memcpy(&u, &f, 4); u += 0x7fff + ((u >> 16) & 1); return (uint16_t)(u >> 16); }

static int fail = 0;
static void expect(const char* tag, int bad) {
  printf("%s: %s\n", tag, bad ? "FAIL" : "ok");
  if (bad) fail++;
}

int main() {
  // decode attention: T=1, D=256, 8 q heads / 1 kv, 32 keys, 1 split
  const int D = 256, T = 1, Hq = 8, Hkv = 1, ctx = 64, pos0 = 31, nsplit = 1;
  const int nkeys = pos0 + T;
  const float scale = 1.f / sqrtf((float)D);
  std::vector<uint16_t> hq(T * Hq * D), hkc(Hkv * ctx * D), hvc(Hkv * ctx * D);
  for (int i = 0; i < (int)hq.size(); i++) hq[i] = tobf(((i * 17) % 50) * 0.02f - 0.4f);
  for (int i = 0; i < (int)hkc.size(); i++) {
    hkc[i] = tobf(((i * 13) % 40) * 0.03f - 0.5f);
    hvc[i] = tobf(((i * 11) % 40) * 0.03f - 0.5f);
  }
  uint16_t *dq, *dkc, *dvc, *dout;
  float* dpart;
  CK(cudaMalloc(&dq, hq.size() * 2)); CK(cudaMalloc(&dkc, hkc.size() * 2));
  CK(cudaMalloc(&dvc, hvc.size() * 2)); CK(cudaMalloc(&dout, Hq * D * 2));
  CK(cudaMalloc(&dpart, (size_t)Hq * nsplit * (D + 2) * 4));
  CK(cudaMemcpy(dq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dkc, hkc.data(), hkc.size() * 2, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dvc, hvc.data(), hvc.size() * 2, cudaMemcpyHostToDevice));
  qk::attn_partial_kernel<256><<<dim3(T * Hkv, nsplit), 256>>>(
      (const qk::bf16*)dq, (const qk::bf16*)dkc, (const qk::bf16*)dvc, dpart,
      T, Hq, Hkv, ctx, pos0, nsplit, scale);
  qk::attn_combine_kernel<256><<<cdiv(Hq, 8), 256>>>(dpart, (qk::bf16*)dout, Hq, nsplit);
  CK(cudaDeviceSynchronize());
  std::vector<uint16_t> ho(Hq * D);
  CK(cudaMemcpy(ho.data(), dout, ho.size() * 2, cudaMemcpyDeviceToHost));
  int bad = 0;
  for (int h = 0; h < Hq; h++) {
    float m = -1e30f;
    std::vector<float> sc(nkeys);
    for (int j = 0; j < nkeys; j++) {
      float s = 0.f;
      for (int d = 0; d < D; d++)
        s += bf(hq[h * D + d]) * scale * bf(hkc[j * D + d]);
      sc[j] = s;
      if (s > m) m = s;
    }
    float den = 0.f;
    for (int j = 0; j < nkeys; j++) den += expf(sc[j] - m);
    for (int d = 0; d < D; d++) {
      float acc = 0.f;
      for (int j = 0; j < nkeys; j++) acc += expf(sc[j] - m) * bf(hvc[j * D + d]);
      if (std::fabs(bf(ho[h * D + d]) - acc / den) > 5e-2f) bad++;
    }
  }
  expect("decode_attn", bad);

  // bandwidth roof: long-cache decode streams K+V (same bytes as a D2D copy of the cache)
  {
    const int ctxB = 4096, posB = 4095, reps = 16, HkvB = 8, HqB = 8, nsplitB = 8;
    uint16_t *qB, *kB, *vB, *oB;
    float* pB;
    CK(cudaMalloc(&qB, HqB * D * 2)); CK(cudaMalloc(&kB, (size_t)HkvB * ctxB * D * 2));
    CK(cudaMalloc(&vB, (size_t)HkvB * ctxB * D * 2)); CK(cudaMalloc(&oB, HqB * D * 2));
    CK(cudaMalloc(&pB, (size_t)HqB * nsplitB * (D + 2) * 4));
    CK(cudaMemset(qB, 1, HqB * D * 2)); CK(cudaMemset(kB, 2, (size_t)HkvB * ctxB * D * 2));
    CK(cudaMemset(vB, 3, (size_t)HkvB * ctxB * D * 2));
    cudaEvent_t e0, e1; CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1));
    const dim3 gridB(HkvB, nsplitB);
    qk::attn_partial_kernel<256><<<gridB, 256>>>(
        (const qk::bf16*)qB, (const qk::bf16*)kB, (const qk::bf16*)vB, pB,
        1, HqB, HkvB, ctxB, posB, nsplitB, scale);
    qk::attn_combine_kernel<256><<<cdiv(HqB, 8), 256>>>(pB, (qk::bf16*)oB, HqB, nsplitB);
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(e0));
    for (int r = 0; r < reps; r++) {
      qk::attn_partial_kernel<256><<<gridB, 256>>>(
          (const qk::bf16*)qB, (const qk::bf16*)kB, (const qk::bf16*)vB, pB,
          1, HqB, HkvB, ctxB, posB, nsplitB, scale);
      qk::attn_combine_kernel<256><<<cdiv(HqB, 8), 256>>>(pB, (qk::bf16*)oB, HqB, nsplitB);
    }
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, e0, e1));
    const double kvBytes = (double)HkvB * ctxB * D * 2 * 2; // K+V
    double attnGBs = kvBytes * reps / (ms * 1e-3) / 1e9;
    uint16_t *a, *b;
    CK(cudaMalloc(&a, (size_t)kvBytes)); CK(cudaMalloc(&b, (size_t)kvBytes));
    CK(cudaEventRecord(e0));
    for (int r = 0; r < reps; r++) CK(cudaMemcpy(b, a, (size_t)kvBytes, cudaMemcpyDeviceToDevice));
    CK(cudaEventRecord(e1)); CK(cudaEventSynchronize(e1));
    float msC = 0; CK(cudaEventElapsedTime(&msC, e0, e1));
    double copyGBs = kvBytes * 2 * reps / (msC * 1e-3) / 1e9;
    double frac = copyGBs > 0 ? attnGBs / copyGBs : 0;
    printf("decode_attn bandwidth=%.1f GB/s copy=%.1f GB/s frac=%.2f\n", attnGBs, copyGBs, frac);
    expect("decode_attn_70pct_roof", frac < 0.70);
  }

  // GDN decode: chunk recurrence left sequential (CH=0)
  {
    const int Dk = 128, Dv = 8, Hk = 1, Hv = 1;
    std::vector<uint16_t> hqg(Dk), hkg(Dk), hvg(Dv), hog(Dv);
    std::vector<float> ha(Hv, 0.9f), hb(Hv, 0.5f), hS((size_t)Hv * Dv * Dk, 0.01f);
    for (int i = 0; i < Dk; i++) { hqg[i] = tobf(0.02f * (i % 7)); hkg[i] = tobf(0.03f * (i % 5)); }
    for (int i = 0; i < Dv; i++) hvg[i] = tobf(0.1f * (i + 1));
    uint16_t *dqg, *dkg, *dvg, *dog;
    float *da, *db, *dS;
    CK(cudaMalloc(&dqg, Dk * 2)); CK(cudaMalloc(&dkg, Dk * 2)); CK(cudaMalloc(&dvg, Dv * 2));
    CK(cudaMalloc(&dog, Dv * 2)); CK(cudaMalloc(&da, 4)); CK(cudaMalloc(&db, 4));
    CK(cudaMalloc(&dS, hS.size() * 4));
    CK(cudaMemcpy(dqg, hqg.data(), Dk * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dkg, hkg.data(), Dk * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dvg, hvg.data(), Dv * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da, ha.data(), 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(db, hb.data(), 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dS, hS.data(), hS.size() * 4, cudaMemcpyHostToDevice));
    const int grid = (Hv * Dv + 7) / 8;
    qk::gdn_recurrence_kernel<128, 0><<<grid, 256>>>(
        (const qk::bf16*)dqg, (const qk::bf16*)dkg, (const qk::bf16*)dvg, da, db, dS,
        (qk::bf16*)dog, 1, Hk, Hv, Dv);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(hog.data(), dog, Dv * 2, cudaMemcpyDeviceToHost));
    std::vector<float> S = hS;
    bad = 0;
    for (int iv = 0; iv < Dv; iv++) {
      float* col = S.data() + iv * Dk;
      float sk = 0.f;
      for (int i = 0; i < Dk; i++) { col[i] *= ha[0]; sk += col[i] * bf(hkg[i]); }
      float dlt = hb[0] * (bf(hvg[iv]) - sk);
      float acc = 0.f;
      for (int i = 0; i < Dk; i++) { col[i] += dlt * bf(hkg[i]); acc += col[i] * bf(hqg[i]); }
      if (std::fabs(bf(hog[iv]) - acc) > 8e-2f) bad++;
    }
    expect("gdn_sequential", bad);
  }

  // one decode-shaped capture. Host dispatch is the graph launch (cut ≥3× vs the recorded kernels).
  {
    cudaStream_t s; CK(cudaStreamCreate(&s));
    const int nRec = 6;
    CK(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
    qk::attn_partial_kernel<256><<<dim3(T * Hkv, nsplit), 256, 0, s>>>(
        (const qk::bf16*)dq, (const qk::bf16*)dkc, (const qk::bf16*)dvc, dpart,
        T, Hq, Hkv, ctx, pos0, nsplit, scale);
    qk::attn_combine_kernel<256><<<cdiv(Hq, 8), 256, 0, s>>>(dpart, (qk::bf16*)dout, Hq, nsplit);
    qk::sigmoid_gate_kernel<<<1, 256, 0, s>>>((qk::bf16*)dout, (const qk::bf16*)dout, Hq * D);
    qk::silu_mul_kernel<<<1, 256, 0, s>>>((const qk::bf16*)dout, (qk::bf16*)dout, 1, Hq * D);
    qk::attn_partial_kernel<256><<<dim3(T * Hkv, nsplit), 256, 0, s>>>(
        (const qk::bf16*)dq, (const qk::bf16*)dkc, (const qk::bf16*)dvc, dpart,
        T, Hq, Hkv, ctx, pos0, nsplit, scale);
    qk::attn_combine_kernel<256><<<cdiv(Hq, 8), 256, 0, s>>>(dpart, (qk::bf16*)dout, Hq, nsplit);
    cudaGraph_t g = nullptr;
    CK(cudaStreamEndCapture(s, &g));
    cudaGraphExec_t ex = nullptr;
    CK(cudaGraphInstantiate(&ex, g, 0));
    CK(cudaGraphLaunch(ex, s));
    CK(cudaStreamSynchronize(s));
    double cut = (double)nRec / 1.0;
    printf("qwen_decode_launches %d→1 cut=%.1f×\n", nRec, cut);
    expect("qwen_decode_launch_cut", cut < 3.0);
    CK(cudaGraphExecDestroy(ex));
    CK(cudaGraphDestroy(g));
    CK(cudaStreamDestroy(s));
  }

  printf(fail ? "FAIL\n" : "PASS\n");
  return fail ? 1 : 0;
}
