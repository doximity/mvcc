#!/bin/bash
# NVPTX oracle pin: CUDA→STIR→llc PTX, then run on NVIDIA when a device is present.
# Usage: tests/nvptx_oracle_pin.sh
#   no GPU  — dump + helpers.ll + llvm-link + llc must succeed
#   NVIDIA  — also launch recovered gemm_tc_kernel / add256 / sgemm / hgemm16
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${MVCC_TEST_OUT:-/tmp/mvcc-tests}"
mkdir -p "$OUT/oracle"
export PATH="$ROOT/toolkit/bin:${PATH}"

CLANG="${MVCC_CLANG:-/opt/homebrew/opt/llvm/bin/clang++}"
LINK="${MVCC_LLVM_LINK:-/opt/homebrew/opt/llvm/bin/llvm-link}"
LLC="${MVCC_LLC:-/opt/homebrew/opt/llvm/bin/llc}"
if [ ! -x "$CLANG" ]; then CLANG="$(command -v clang++ || true)"; fi
if [ ! -x "$LINK" ]; then LINK="$(command -v llvm-link || true)"; fi
if [ ! -x "$LLC" ]; then LLC="$(command -v llc || true)"; fi

echo "== dump gemm_ptx recovered LLVM =="
MVCC_NVPTX_ORACLE="$OUT/oracle/gemm_ptx.ll" nvcc -O2 -std=c++17 -c \
  -o "$OUT/oracle/gemm_ptx.o" "$ROOT/tests/kernels/gemm_ptx.cu" \
  >"$OUT/oracle/gemm_ptx.dump.log" 2>&1
test -s "$OUT/oracle/gemm_ptx.ll"

echo "== dump pin LLVM =="
MVCC_NVPTX_ORACLE="$OUT/oracle/pin.ll" nvcc -O2 -std=c++17 -c \
  -o "$OUT/oracle/pin.o" "$ROOT/tests/kernels/nvptx_oracle_pin.cu" \
  >"$OUT/oracle/pin.dump.log" 2>&1
test -s "$OUT/oracle/pin.ll"

if [ ! -x "$CLANG" ] || [ ! -x "$LINK" ] || [ ! -x "$LLC" ]; then
  echo "skip link/llc (no LLVM tools)"
  exit 0
fi

echo "== helpers.ll + link + llc =="
$CLANG -x cuda --cuda-path="$ROOT/toolkit" -nocudainc -nocudalib \
  -Xclang -target-sdk-version=12.8 -isystem "$ROOT/include" \
  -D__NVCC__ -D__CUDACC__ -D__CUDACC_VER_MAJOR__=12 -D__CUDACC_VER_MINOR__=8 -D__CUDACC_VER_BUILD__=61 \
  -D__MVCC_CUDA_ARCH__=890 -Wno-unknown-cuda-version -include cuda_runtime.h \
  --cuda-device-only --cuda-gpu-arch=sm_89 --cuda-feature=+ptx86 \
  -O2 -std=c++17 -emit-llvm -S -o "$OUT/oracle/helpers.ll" \
  "$ROOT/tests/kernels/nvptx_oracle_helpers.cu"
$LINK "$OUT/oracle/gemm_ptx.ll" "$OUT/oracle/helpers.ll" -o "$OUT/oracle/gemm_linked.bc"
$LLC -march=nvptx64 -mcpu=sm_89 -o "$OUT/oracle/gemm_linked.ptx" "$OUT/oracle/gemm_linked.bc"
$LINK "$OUT/oracle/pin.ll" "$OUT/oracle/helpers.ll" -o "$OUT/oracle/pin_linked.bc"
$LLC -march=nvptx64 -mcpu=sm_89 -o "$OUT/oracle/pin_linked.ptx" "$OUT/oracle/pin_linked.bc"
test -s "$OUT/oracle/gemm_linked.ptx" && test -s "$OUT/oracle/pin_linked.ptx"
echo "linked PTX written"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "skip device run (no NVIDIA GPU)"
  exit 0
fi

echo "== device run =="
NVCC="$(command -v nvcc || true)"
if [ -x /usr/local/cuda/bin/nvcc ]; then NVCC=/usr/local/cuda/bin/nvcc; fi
"$NVCC" -arch=sm_89 -O2 -std=c++17 -lcuda -DMVCC_ORACLE_HOST \
  -o "$OUT/oracle/oracle_run" "$ROOT/tests/kernels/nvptx_oracle_pin.cu"
"$NVCC" -arch=sm_89 -O2 -std=c++17 -lcuda -DMVCC_ORACLE_HOST \
  -o "$OUT/oracle/oracle_gemm" "$ROOT/tests/kernels/nvptx_oracle_gemm_host.cu"
"$OUT/oracle/oracle_run" "$OUT/oracle/pin_linked.ptx"
"$OUT/oracle/oracle_gemm" "$OUT/oracle/gemm_linked.ptx" all
echo "nvptx-oracle device pin: ok"
