# Changelog

## Unreleased

- Publish the toolkit as the Homebrew formula `cuda-mvcc` (`Formula/cuda-mvcc.rb`). A release asset tagged `v<version>` updates the formula's checksum. The release workflow accepts only a `vX.Y.Z` tag, and the formula text is filled from that restricted alphabet. A prefix install refuses a world-writable destination, a symlink that leaves the prefix, and a non-absolute `libcudart` install name, and ad-hoc signs the toolchain with the hardened runtime. The compiler drops `DYLD_*`, `CPATH`, and `LIBRARY_PATH` before it runs clang.
- cuRAND Philox4_32_10: device API in `curand_kernel.h`, and a host generator (`curand.h`, `libcurand`) for that generator only. Generator handles are random, and `libcudart`'s install name is an absolute path.
- Tensor recovery maps `mma.sync.m16n8k32` with s8/u8 inputs (s32 accumulate) onto int8 TensorOps. Fragments built from plain 32-bit loads of device memory or a shared-memory tile (no `ldmatrix`) become matmul operands read in place, including at per-warp tile origins and through pointers in a by-value struct argument. `tests/kernels/gemm_dev.cu` covers bf16 and s8 GEMMs and an attention-shaped kernel.

## 1.0.1

First open-source release, licensed under the Apache License 2.0.
