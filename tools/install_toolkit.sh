#!/bin/bash
# Build everything and assemble toolkit/ so that CMake's FindCUDAToolkit / enable_language(CUDA) and plain
# Makefiles see a CUDA toolkit: bin/nvcc, include/, lib64/libcudart.dylib, version.json.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TK="$ROOT/toolkit"
export PATH="$HOME/.cargo/bin:/opt/homebrew/bin:$PATH"
PROFILE="${1:-release}"
FROM_GEM="${MVCC_INSTALL_FROM_GEM:-}"

# `-w /dev/tty` is true even without a controlling terminal (scripts, CI, editor shells);
# only the open() tells. Probe in a subshell so a failure cannot exit this script.
if [ -n "$FROM_GEM" ] && ( : >/dev/tty ) 2>/dev/null; then
  exec >/dev/tty 2>&1
fi

run_cmake() {
  if [ -n "$FROM_GEM" ]; then cmake "$@"; else cmake "$@" >/dev/null; fi
}

run_cmake_build() {
  if [ -n "$FROM_GEM" ]; then cmake --build "$@"; else cmake --build "$@" 2>&1 | tail -1; fi
}

run_cargo() {
  if [ -n "$FROM_GEM" ]; then cargo "$@"; else cargo "$@" 2>&1 | tail -1; fi
}

echo "== cpp (mvcc-ir2msl, mvcc-passes)"
run_cmake -S "$ROOT/cpp/mvcc-llvm" -B "$ROOT/build/mvcc-llvm" -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLVM_DIR="$(brew --prefix llvm)/lib/cmake/llvm"
run_cmake_build "$ROOT/build/mvcc-llvm" --target mvcc-ir2msl mvcc-passes

echo "== objc tools (mvcc-mslc)"
mkdir -p "$ROOT/build"
clang++ -std=c++17 -O2 -fobjc-arc -framework Metal -framework Foundation "$ROOT/tools/mslc/mslc.mm" -o "$ROOT/build/mvcc-mslc"

echo "== rust ($PROFILE)"
if [ "$PROFILE" = release ]; then run_cargo build --release --workspace; OUT="$ROOT/target/release"; else run_cargo build --workspace; OUT="$ROOT/target/debug"; fi

echo "== toolkit layout"
# tools/release.sh ships toolkit/ whole: start empty so nothing an earlier build left behind goes out with it.
rm -rf "$TK"
mkdir -p "$TK/bin" "$TK/lib64" "$TK/nvvm/libdevice" "$TK/share/mvcc"
ln -sfn ../include "$TK/include"
# Executables and dylibs go in as new files (unlink, then copy): the kernel caches a Mach-O's code signature by
# inode, and a binary overwritten in place is killed at exec (SIGKILL, "Killed: 9") until the cache drops it.
put() { rm -f "$2"; cp "$1" "$2"; }
put "$OUT/mvcc" "$TK/bin/mvcc"
ln -sfn mvcc "$TK/bin/nvcc"
put "$ROOT/build/mvcc-llvm/mvcc-ir2msl" "$TK/bin/mvcc-ir2msl"
put "$ROOT/build/mvcc-llvm/libmvcc-passes.dylib" "$TK/lib64/libmvcc-passes.dylib"   # clang -fpass-plugin for the device pass
put "$ROOT/build/mvcc-mslc" "$TK/bin/mvcc-mslc"
cp -f "$ROOT/msl/mvcc_prelude.metal" "$TK/share/mvcc/mvcc_prelude.metal"

# runtime: one dylib serving both the runtime (libcudart) and driver (libcuda) APIs
put "$OUT/libcudart.dylib" "$TK/lib64/libcudart.dylib"
# Gem install paths are longer than @rpath/libcudart.dylib. Without headerpad
# this fails and `set -e` aborts at "== toolkit layout". Keep going if rewrite
# still cannot fit; the dylib then stays at @rpath.
if ! install_name_tool -id "$TK/lib64/libcudart.dylib" "$TK/lib64/libcudart.dylib"; then
  echo "warning: install_name_tool could not set libcudart id; leaving @rpath"
fi
ln -sfn libcudart.dylib "$TK/lib64/libcudart.12.dylib"
ln -sfn libcudart.dylib "$TK/lib64/libcuda.dylib"
ln -sfn libcudart.dylib "$TK/lib64/libcuda.1.dylib"
# NCCL over the logical devices (include/nccl.h): the same dylib; `find_library(nccl)` / `-lnccl` find it here
ln -sfn libcudart.dylib "$TK/lib64/libnccl.dylib"
ln -sfn libcudart.dylib "$TK/lib64/libnccl.2.dylib"

# static runtime: the rust staticlib plus an object carrying the framework/link dependencies via
# LC_LINKER_OPTION so `-lcudart_static` alone links.
TMP="$(mktemp -d)"
cat > "$TMP/autolink.ll" <<'EOF'
target triple = "arm64-apple-macosx14.0.0"
!llvm.linker.options = !{!0, !1, !2, !3, !4}
!0 = !{!"-framework", !"Metal"}
!1 = !{!"-framework", !"Foundation"}
!2 = !{!"-framework", !"IOKit"}
!3 = !{!"-framework", !"CoreFoundation"}
!4 = !{!"-lc++"}
EOF
clang -c "$TMP/autolink.ll" -o "$TMP/mvcc_autolink.o" -Wno-override-module
cp -f "$OUT/libcudart.a" "$TK/lib64/libcudart_static.a"
ar q "$TK/lib64/libcudart_static.a" "$TMP/mvcc_autolink.o" 2>/dev/null
ranlib "$TK/lib64/libcudart_static.a" 2>/dev/null || true

# libraries CMake/nvcc link unconditionally that have no content here
printf 'static int __mvcc_cudadevrt_unused;\n' > "$TMP/devrt.c"
clang -c "$TMP/devrt.c" -o "$TMP/devrt.o"
for lib in cudadevrt rt; do rm -f "$TK/lib64/lib$lib.a"; ar rcs "$TK/lib64/lib$lib.a" "$TMP/devrt.o" 2>/dev/null; done
rm -rf "$TMP"

# version.json: what CMake's FindCUDAToolkit reads (the CUDA release we present) plus our own version (the workspace
# version in Cargo.toml, the single source) and the macOS range this version is validated on (README.md).
MVCC_VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"
cat > "$TK/version.json" <<EOF
{
  "cuda": { "name": "CUDA SDK", "version": "12.8.0" },
  "cuda_cudart": { "name": "CUDA Runtime (cudart)", "version": "12.8.57" },
  "cuda_nvcc": { "name": "CUDA NVCC", "version": "12.8.61" },
  "mvcc": { "name": "mvcc", "version": "$MVCC_VERSION", "macos_validated": ">=26.0, <27.0" }
}
EOF
echo "toolkit ready: $TK"
# sed reads to EOF: `head -1` closes the pipe after one line, and under pipefail the driver's panic on the broken
# pipe (Rust ignores SIGPIPE; exit 101) failed the build whenever the driver lost the race
"$TK/bin/nvcc" --version | sed -n 1p
