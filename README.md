# mvcc

mvcc compiles and runs CUDA C++ on Apple silicon GPUs. It consists of an `nvcc`-compatible compiler that translates device code to Metal Shading Language, and a CUDA runtime (`libcudart`, `libcuda` and `libnccl`) built on Metal 4. CUDA programs and CMake projects build and run on a Mac without source changes, within the limits described under [Limitations](#limitations).

No NVIDIA software is involved at build or run time. There is no CUDA toolkit, driver or `ptxas`, and no cuBLAS or other NVIDIA library.

## Requirements

- An Apple silicon Mac running macOS 26
- Xcode Command Line Tools
- Homebrew `llvm` 21 or later (clang with the NVPTX backend), `cmake` and `ninja`
- Rust, installed with [rustup](https://rustup.rs)

## Install

```bash
brew install llvm cmake ninja rust
git clone https://github.com/doximity/mvcc.git
cd mvcc
tools/install_toolkit.sh
export PATH="$PWD/toolkit/bin:$PATH"
nvcc --version
```

`tools/install_toolkit.sh` builds the compiler and runtime into `toolkit/`, which is laid out like a CUDA 12.8 installation so that CMake's `enable_language(CUDA)` and `find_package(CUDAToolkit)` work unchanged. `nvcc --version` identifies the build as `mvcc_<version>_0`.

### Ruby gem

The gem builds the same toolkit when it is installed, so it needs the Homebrew packages and Rust listed above:

```bash
gem build mvcc.gemspec --output mvcc.gem
gem install ./mvcc.gem
```

The installer then adds the toolkit's `bin` directory to `PATH` in your shell startup files: the file for `$SHELL`, plus any of `.zshrc`, `.zprofile`, `.bashrc`, `.bash_profile`, `.profile` and Fish's `config.fish` that already exist. Set `MVCC_SKIP_PATH=1` (or `CI=1`) to leave those files alone, or `MVCC_ASK_PATH=1` to be asked first.

## Usage

Use the toolkit's `nvcc` as you would NVIDIA's:

```bash
nvcc -O3 -std=c++17 -o app app.cu
./app
```

For a CMake project, point CMake at the toolkit:

```bash
cmake -S . -B build \
  -DCMAKE_CUDA_COMPILER=/path/to/mvcc/toolkit/bin/nvcc \
  -DCUDAToolkit_ROOT=/path/to/mvcc/toolkit \
  -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build
```

The driver accepts nvcc's command line, including the options CMake generates. Options that have no meaning on Metal are ignored, and unrecognized options are passed to the host compiler with a warning, as `nvcc -forward-unknown-to-host-compiler` does. `-arch` and `-gencode` choose the CUDA feature level that device code is compiled for (`__CUDA_ARCH__`, up to `sm_90`; `sm_89` by default), not a GPU target: Apple's compiler generates code for the GPU that is present when the program first runs. `cudaGetDeviceProperties` reports compute capability 8.9 unless `MVCC_ARCH` asks for a higher one. `mvcc --help` lists the environment variables that control the compiler.

### Ruby

The gem can also compile CUDA at run time and call its `extern "C"` functions from Ruby:

```ruby
require "mvcc"

lib = Mvcc.compile(<<~'CU')
  #include <cuda_runtime.h>

  __global__ void fill(char* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = 'a' + i % 26;
  }

  extern "C" void alphabet(char* host, int n) {
    char* dev;
    cudaMalloc(&dev, n);
    fill<<<(n + 255) / 256, 256>>>(dev, n);
    cudaMemcpy(host, dev, n, cudaMemcpyDeviceToHost);
    cudaFree(dev);
  }
CU

buf = Mvcc.buffer(26)
lib.alphabet(buf, 26)
puts buf.to_s # abcdefghijklmnopqrstuvwxyz
```

`Mvcc.compile` takes CUDA source or the path of a `.cu` file, and caches the compiled library under `~/Library/Caches/mvcc/ruby`. Integers are passed as `int`, floats as `float`, and `Mvcc.buffer` objects as pointers to host memory.

## Examples

- [`examples/qwen`](examples/qwen): inference for Qwen3.5-35B-A3B, a mixture-of-experts language model, written as ordinary CUDA C++. It decodes at about 80 tokens/s on an M5 Pro.
- [`examples/hashhunt`](examples/hashhunt): a brute-force MD5 search, checked against the same search in C++.
- [`examples/ruby`](examples/ruby): three terminal demos of the Ruby API, two fractal renderers and a forest you can fly through with the arrow keys.

## Limitations

- mvcc runs only on Apple silicon Macs with macOS 26. It does not bring CUDA to Intel Macs, Linux or Windows, and it does not generate code for NVIDIA GPUs.
- It implements CUDA C++ and the CUDA runtime, not NVIDIA's libraries. Programs that use cuBLAS, cuDNN, cuFFT, cuRAND, cuSPARSE, Thrust or TensorRT do not build, so frameworks built on those libraries, such as PyTorch's CUDA backend, cannot use mvcc.
- Kernels must be compiled by mvcc's `nvcc`. NVRTC is not provided, PTX and cubin files cannot be loaded at run time, and the driver API covers only devices, contexts and memory.
- Texture and surface memory are not supported.
- A Mac has one GPU. `MVCC_DEVICES` presents up to 16 logical devices on it, which is enough to run multi-GPU code paths and NCCL collectives but adds no compute or memory.
- NVIDIA's debugging and profiling tools (cuda-gdb, compute-sanitizer, Nsight) cannot be used. Device-side `printf` prints nothing, and the profiler API calls do nothing.
- Apple GPUs have no fp64 hardware. fp64 arithmetic is emulated in software and is much slower than fp32.
- Tensor-core code is fast only when mvcc recovers the matrix product its loop computes (see [Tensor cores](#tensor-cores)). Code it does not recognize runs as an exact translation, which is correct but much slower.
- This is the first public release, tested with the programs in `tests/` and `examples/`. Code that uses a feature outside the [supported CUDA surface](#supported-cuda-surface) fails to build, with an error that names the feature.

Behavior that differs from CUDA on NVIDIA hardware is described under [Differences from CUDA](#differences-from-cuda).

## How it works

The toolkit's `nvcc` is mvcc's compiler driver. For each CUDA source file, it:

1. compiles the device code to LLVM IR with Homebrew clang (`-nocudainc -nocudalib`);
2. translates that IR to Metal Shading Language with `mvcc-ir2msl`;
3. compiles the host code with clang, embedding the MSL in the object file where clang would embed an NVIDIA fatbinary;
4. links the program against mvcc's `libcudart.dylib`.

When a program first launches a kernel, the runtime passes the MSL to `MTLDevice.newLibraryWithSource`, so Apple's compiler generates code for the GPU that is actually present. Compiled pipelines are cached in `~/Library/Caches/mvcc/`.

The runtime maps CUDA directly onto Metal. `cudaMalloc` returns unified memory, streams submit their work as Metal command buffers, thread blocks are threadgroups, and warps are 32-wide SIMD groups. `libcudart.dylib` and `libcuda.dylib` are the same Rust library.

## Tensor cores

Tensor-core code (`mma.sync`, `ldmatrix`, `cp.async`) has an exact translation built on `simdgroup_matrix`. The Neural Accelerators in M5 GPUs are much faster, but Metal exposes them only through Metal 4 TensorOps, which operate on whole tiles and cannot execute a CUDA GEMM loop instruction by instruction.

mvcc works from the whole loop instead. It recovers the matrix product the loop computes, proves that a Metal 4 `matmul2d` on cooperative tensors computes the same result, and emits that. Every recovered kernel is also compiled as an exact translation of the original code. The runtime uses the exact version when the GPU does not support Metal 4, when on-device layout verification fails, or when a launch shape is outside what the proof covers. `MVCC_TENSOR_DIAG=1` reports what recovery did for each kernel, and `MVCC_TENSOR_EXACT=1` runs the exact versions.

For new code, `mvcc::warp_tile` in `include/mvcc/tile.cuh` is a portable warp-level matrix-multiply API. It compiles to `mma.sync` with NVIDIA's `nvcc` and to `matmul2d` with mvcc.

## Differences from CUDA

Behavior not listed here should match CUDA on an NVIDIA GPU. If it doesn't, please [open an issue](https://github.com/doximity/mvcc/issues).

### Numerics

- fp32 denormals are flushed to zero.
- fp64 is emulated in software: IEEE binary64 `+`, `-`, `*`, `/`, `sqrt`, comparisons, rounding and conversions. fp64 transcendental functions are compile errors.
- `__expf`, `__logf` and `__sinf` use Metal's `fast::` functions, and `expf`, `logf` and `sinf` use `precise::`. `-use_fast_math` selects `fast::` throughout.
- `lgammaf` and `tgammaf` are compile errors.
- bf16 and f16 conversions are native casts.
- Floating-point results and timing can differ from NVIDIA hardware.

### Execution

- Warps are 32 threads wide, and a block can have up to 1024 threads.
- The threads of a warp run in lockstep, without the independent thread scheduling of Volta and later GPUs. A spin lock that threads of the same warp contend for can hang.
- A threadgroup has 32 KB of shared memory. When a launch needs more (up to 232,448 bytes, the value reported as `sharedMemPerBlockOptin`), the runtime compiles a variant of the module that keeps shared memory in a device-memory pool, and uses it for those launches.
- 64-bit integer atomics are implemented with hashed spinlocks. Concurrent aligned 64-bit loads see either the old or the new value. Atomics on `double`, such as `atomicAdd(double*)`, are compile errors.
- Device-side `printf` compiles and returns 0 without printing anything, because Metal has no counterpart to CUDA's printf buffer.
- Inline PTX reads of `%globaltimer`, `%clock` and `%clock64` compile and return 0.
- `clock()`, dynamic parallelism and grid-wide cooperative-groups synchronization are not available.

### Memory and streams

- `cudaMalloc` returns unified memory.
- Work within a stream runs in order, and the default stream synchronizes with other streams as in CUDA.
- Stream capture records kernel launches, copies and memsets. Building a graph node by node through the explicit graph API is not supported.

### Device code

- Every `__device__` function is inlined into its kernel. Recursion is not supported, and a kernel can only call device functions defined in the same translation unit: `-rdc=true` and `-dlink` are accepted, but the device-link object is empty.
- Local arrays need a size known at compile time.
- Irreducible control flow is rewritten as a state machine where possible.
- `__shared__` variables whose type has default member initializers are accepted and left uninitialized, as nvcc does.
- Mixed pointers, which point to thread-local memory on some paths and to device or threadgroup memory on others, support plain loads and stores. Atomic or volatile accesses through them, and other operations that need a single address space, are compile errors.

### Libraries

NVIDIA's libraries (cuBLAS, cuDNN, cuFFT, cuRAND, Thrust and the rest) are not included. `cub/cub.cuh` provides a small subset of CUB, and `nccl.h` implements NCCL across logical devices.

## Supported CUDA surface

- Host API: 132 runtime API and 18 driver API entry points, covering device queries, memory, copies, streams, events, stream capture, host callbacks, occupancy, profiler stubs and clang's registration hooks.
- Device code: 69 `llvm.nvvm.*` intrinsic prefixes and 118 libdevice (`__nv_*`) functions.
- Inline PTX: 29 instructions, including `mma.sync` (10 f16, bf16, s8 and fp8 variants), `ldmatrix`, `cp.async`, `lop3`, `prmt` and `shf`.
- Headers: `cuda_runtime.h`, `cuda.h`, `cuda_fp16.h`, `cuda_bf16.h`, `cuda_fp8.h`, `nccl.h`, `cub/cub.cuh`, `mvcc/tile.cuh` and the rest of `include/`.
- NCCL: communicators, copy collectives, and AllReduce, Reduce and ReduceScatter (sum or average, on f32 or i32) across the logical devices set by `MVCC_DEVICES`.

Anything outside this surface fails at build time: a missing host API function is a link error, and an unsupported intrinsic, libdevice function or PTX instruction is a compile error that names it. `tools/gen_surface.py` prints the complete list. `tools/gen_diagnostics.py` prints every message the compiler and runtime can produce, with advice on what to do about each one.

## Performance

Measured on an Apple M5 Pro (20-core GPU, 48 GB of unified memory) running macOS 26.5.1:

| Measurement                                                               | Result                                          |
| ------------------------------------------------------------------------- | ----------------------------------------------- |
| f16 GEMM, M = N = K = 2048, 64x64 tiles, recovered kernel                 | 19.1 TFLOP/s                                    |
| Recovered vs. exact kernels, f16 GEMMs                                    | 8-21x faster                                    |
| Recovered vs. exact kernels, INT4 decode experts                          | 15-25x faster                                   |
| Metal 4 TensorOps, 32x32x32 and 64x32x32 tiles, operands in device memory | 17.6-18.5 TFLOP/s                               |
| `simdgroup_matrix`, 16x16 tiles                                           | 7.4 TFLOP/s                                     |
| Streaming read                                                            | 243 GB/s                                        |
| 1 GB copy, read plus write                                                | about 250 GB/s                                  |
| `examples/qwen` decode                                                    | 79-81 tokens/s                                  |
| `examples/qwen` pipeline compilation (50 kernels)                         | about 1.3 s on first launch, 2-5 ms once cached |

Recovered kernels match the CPU reference as closely as the exact translations do (maximum relative error 4.9e-4 in fp16).

### Precompiled Metal libraries

To skip compiling MSL on first launch, build `.metallib` files ahead of time with Xcode's Metal toolchain (`xcrun metal`, `xcrun metallib`) and point the runtime at them:

```bash
MVCC_KEEP=1 nvcc -O3 -std=c++17 -o app app.cu
tools/precompile_metallib.sh app.metal ./metallibs
MVCC_METALLIB_DIR=./metallibs ./app
```

## Environment variables

| Variable                  | Effect                                                                                      |
| ------------------------- | ------------------------------------------------------------------------------------------- |
| `MVCC_KEEP=1`             | Keep `.device.ll`, `.metal` and `.abi.json` next to the output                              |
| `MVCC_VERBOSE=1`          | Print each compile step, and the runtime's module and pipeline events                       |
| `MVCC_VERIFY=0`           | Skip the build-time Metal compile check of generated MSL                                    |
| `MVCC_TENSOR_DIAG=1`      | Report, for each kernel, what tensor-core recovery did or why it declined                   |
| `MVCC_TENSOR_EXACT=1`     | Launch the exact translations instead of recovered kernels                                  |
| `MVCC_METALLIB_DIR=<dir>` | Load precompiled `<hash>.metallib` files from `<dir>`                                       |
| `MVCC_DEVICES=<n>`        | Present `n` logical devices (1-16)                                                          |
| `MVCC_GRAPH=<file>`       | Kernel graph for compile-time fusion (by default, inferred from kernel order in the module) |
| `MVCC_ARCH=sm_XX`         | CUDA architecture for `__CUDA_ARCH__`, overriding `-arch`                                   |
| `MVCC_CLANG`, `MVCC_ROOT` | Paths to clang and to the toolkit (see `mvcc --help`)                                       |
| `MVCC_MEM_FRACTION`       | Limit on `cudaMalloc`, as a fraction of Metal's recommended working set (default 0.97)      |

`mvcc explain app.cu` compiles a file and prints, for each kernel, the computation mvcc recovered from it. The same analysis is available during a normal build through `MVCC_SIG=1`, `MVCC_SIG_ONLY=1`, `MVCC_SIG_DUMP=<file>`, `MVCC_SCHED=1` and `MVCC_EXPLAIN=1`; `tests/run.sh` shows them in use.

## Repository layout

| Path                 | Contents                                                                                        |
| -------------------- | ----------------------------------------------------------------------------------------------- |
| `crates/mvcc-driver` | The `nvcc` and `mvcc` compiler driver (Rust)                                                    |
| `cpp/mvcc-llvm`      | `mvcc-ir2msl`, which translates device LLVM IR to MSL and recovers tensor-core code (C++, LLVM) |
| `crates/mvcc-cudart` | The CUDA runtime, driver and NCCL APIs (Rust)                                                   |
| `cpp/mvcc-metal`     | Objective-C++ layer between the runtime and Metal                                               |
| `include`            | CUDA-compatible headers                                                                         |
| `msl`                | Metal prelude included in generated shaders                                                     |
| `lib`, `ext`         | Ruby gem                                                                                        |
| `examples`           | Example programs                                                                                |
| `tests`              | Test suite (`tests/run.sh`)                                                                     |
| `tools`              | Build, release and maintenance scripts                                                          |

## Contributing

Bug reports and pull requests are welcome. A small `.cu` file that reproduces the problem makes a bug much easier to fix.

Contributors must sign the Doximity Individual Contributor License Agreement, which is reproduced in [CONTRIBUTING.md](CONTRIBUTING.md); submitting a contribution means you agree to it.

Before opening a pull request, run `tests/run.sh` on an Apple silicon Mac. `tests/run.sh --host` builds and checks the compiler without running kernels on the GPU.

Please report security vulnerabilities privately, as described in [Doximity's security policy](https://www.doximity.com/about/security), rather than in a public issue.

## License

mvcc is licensed under the [Apache License, Version 2.0](LICENSE). Copyright 2026 Doximity, Inc. Third-party notices are in [NOTICE](NOTICE). Contributions are accepted under the contributor license agreement in [CONTRIBUTING.md](CONTRIBUTING.md).

`mvcc-ir2msl` statically links LLVM (Apache 2.0 with LLVM Exceptions) and links Z3 (MIT) and zstd (BSD) from Homebrew. The Rust crates depend on `serde`, `serde_json`, `sha2`, `parking_lot` and `cc` (MIT or Apache 2.0). `erff` and `erfcf` in `msl/mvcc_prelude.metal` are derived from FreeBSD's msun, under the Sun Microsystems license reproduced in NOTICE. Everything else is original work under the Apache License 2.0.

mvcc is provided as is, without warranty, under the terms of the Apache License 2.0.

### NVIDIA and Apple

mvcc is an independent project, developed separately from NVIDIA and Apple.

The repository and its release archives are mvcc's own work. They contain no CUDA toolkit headers, no `libcudart`, `libcuda` or cuBLAS sources or binaries, no libdevice bitcode, and no PTX tools or driver components. mvcc's headers and libraries carry CUDA's names (`cuda_runtime.h`, `libcudart.12.dylib`, `bin/nvcc`) so that existing build systems find them, but they are mvcc's own files. The runtime API, driver API, language extensions, intrinsics and the PTX subset mvcc accepts were implemented from NVIDIA's published documentation, treating NVIDIA's compiler, driver, libraries and firmware as a black box.

mvcc uses Apple software only through public interfaces: the Metal, Foundation and IOKit frameworks, and the Metal shader compiler included with macOS.

CUDA, nvcc, cuBLAS, PTX and NVIDIA are trademarks of NVIDIA. Apple, Metal, Xcode, macOS and Apple silicon are trademarks of Apple Inc. They are used here only to identify the interfaces mvcc implements and the platform it runs on.
