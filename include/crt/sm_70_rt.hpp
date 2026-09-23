// mvcc: clang's __clang_cuda_intrinsics.h includes this NVIDIA header for __match_* helpers.
// We provide the two names it wants with public-semantics implementations.
#ifndef MVCC_CRT_SM_70_RT_HPP
#define MVCC_CRT_SM_70_RT_HPP
#if defined(__CUDA__) && defined(__clang__)
static __device__ __forceinline__ unsigned int __match_any_sync(unsigned int mask, unsigned int value) {
  unsigned int result = 0;
  for (unsigned int m = mask; m;) {
    int lane = __builtin_ffs((int)m) - 1;
    unsigned int other = (unsigned int)__nvvm_shfl_sync_idx_i32(mask, (int)value, lane, 0x1f);
    unsigned int eq = (unsigned int)__nvvm_vote_ballot_sync(mask, other == value);
    if (other == value) result = eq;
    m &= ~eq;
  }
  return result & mask;
}
static __device__ __forceinline__ unsigned int __match_any_sync(unsigned int mask, unsigned long long value) {
  return __match_any_sync(mask, (unsigned int)value) & __match_any_sync(mask, (unsigned int)(value >> 32));
}
static __device__ __forceinline__ unsigned int __match_all_sync(unsigned int mask, unsigned int value, int* pred) {
  unsigned int r = __match_any_sync(mask, value);
  *pred = (r == mask);
  return *pred ? mask : 0;
}
#endif
#endif
