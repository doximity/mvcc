# Changelog

## 1.3.0

- Removed `MVCC_PORTABLE_NUMERICS`. Its NVIDIA tensor-core rounding ran f16/bf16 `mma.sync` on the ALUs, about 80 times slower than `matmul2d`. Code that needs the same bits as an NVIDIA GPU keeps its tensor-core sums exact instead (README.md).
- s8 and u8 `mma.sync` fed by `ldmatrix` or by fragments computed in registers are recovered onto `matmul2d`. They used to be recovered only when every fragment was a plain load from memory, so a kernel that slices or quantizes its operands in registers ran each `mma.sync` on the ALUs. 8-bit cooperative tensors follow the same layout law as half, four elements to a 32-bit word; the runtime verifies it on the device. `tests/kernels/gemm_i8.cu`.

## 1.2.0

- `a * b + c` that clang marks `contract` fuses into one fma, which is what nvcc does with `-fmad=true`. The emitter used to write the multiply and the add separately, and Metal's safe math does not fuse them, so those expressions rounded twice and about 10% of random inputs differed in the last place from an A10G. `-fmad=false` compiles the device pass with `-ffp-contract=off`. float and half only: double is emulated and bfloat has no Metal fma.
- `MVCC_PORTABLE_NUMERICS=1` (runtime): f16 and bf16 `mma.sync` with an f32 accumulator rounds each k8 step as an NVIDIA tensor core does: the eight products and the accumulator align to 24 fraction bits, sum, and round toward zero. Recovered kernels run their exact twins, since `matmul2d` rounds as Apple's matrix units do. Off by default. `tests/kernels/mma_nvidia.cu` checks `m16n8k16` against an A10G bit for bit.

## 1.1.2

- A stream's command buffer also commits once its estimated GPU time reaches about a second, the estimate being the threads it launched times the GPU ns per thread measured on the stream's completed buffers. A long prefill on a throttled GPU packed 512 heavy launches into one buffer, and macOS aborted it (`kIOGPUCommandBufferCallbackErrorTimeout`), which surfaced as an illegal memory access. Decode-sized buffers are unchanged.
- A host wait faults only after 30 s in which none of the stream's command buffers completed. It used to give all of a stream's queued work 8 s, so a slow but progressing GPU read as a wedge.

## 1.1.1

- `mvcc-ir2msl` loads Homebrew Z3 through `libz3.dylib` instead of the versioned soname (`libz3.5.1.dylib`). A newer `z3` keg no longer leaves `nvcc` unable to start.

## 1.1.0

- Publish the toolkit as the Homebrew formula `cuda-mvcc` (`Formula/cuda-mvcc.rb`), which keeps the toolkit in `libexec` and links only `nvcc` and `mvcc`. A release asset tagged `v<version>` updates the formula's checksum. The release workflow accepts only a `vX.Y.Z` tag, and the formula text is filled from that restricted alphabet. A prefix install refuses a world-writable destination, a symlink that leaves the prefix, and a non-absolute `libcudart` install name, and ad-hoc signs the toolchain with the hardened runtime. The compiler drops `DYLD_*`, `CPATH`, and `LIBRARY_PATH` before it runs clang.
- `nvcc` uses clang only from the LLVM major release that `mvcc-ir2msl` and its device pass were built against, now recorded in `version.json`: `llvm@<major>` first, then Homebrew `llvm`, then `PATH`. It refuses any other clang, including one named by `MVCC_CLANG`, with an error naming both versions. `nvcc -v` prints `TOP` and `NVVMIR_LIBRARY_DIR`, so CMake finds the toolkit through a symlinked `nvcc`.
- cuRAND Philox4_32_10: device API in `curand_kernel.h`, and a host generator (`curand.h`, `libcurand`) for that generator only. Generator handles are random, and `libcudart`'s install name is an absolute path.
- Tensor recovery maps `mma.sync.m16n8k32` with s8/u8 inputs (s32 accumulate) onto int8 TensorOps. Fragments built from plain 32-bit loads of device memory or a shared-memory tile (no `ldmatrix`) become matmul operands read in place, including at per-warp tile origins and through pointers in a by-value struct argument. `tests/kernels/gemm_dev.cu` covers bf16 and s8 GEMMs and an attention-shaped kernel.

## 1.0.1

First open-source release, licensed under the Apache License 2.0.
