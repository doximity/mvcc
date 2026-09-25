#!/bin/bash
# Build the CUDA-only hash search, show that this folder has no Metal, print the
# MSL mvcc generated, then run the binary.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
export PATH="$ROOT/toolkit/bin:${HOME}/.cargo/bin:/opt/homebrew/bin:$PATH"
if ! command -v nvcc >/dev/null; then
  echo "nvcc not on PATH; run tools/install_toolkit.sh from the repo root" >&2
  exit 1
fi

cd "$HERE"
echo "== authored sources (no Metal)"
ls -1 "$HERE" | sed 's/^/  /'
if ls "$HERE"/*.metal >/dev/null 2>&1; then
  echo "unexpected .metal in examples/hashhunt" >&2
  exit 1
fi

KEEP="$HERE/.keep"
rm -rf "$KEEP"
mkdir -p "$KEEP"
echo
echo "== mvcc compile (MVCC_KEEP=1): generated MSL, not written by hand"
(cd "$KEEP" && MVCC_KEEP=1 nvcc -O2 -std=c++17 -c -o hashhunt.o "$HERE/hashhunt_gpu.cu")
METAL=$(ls "$KEEP"/*.metal 2>/dev/null | head -1)
if [ -z "${METAL:-}" ]; then
  echo "no generated .metal under $KEEP" >&2
  exit 1
fi
echo "  $(wc -l < "$METAL" | tr -d ' ') lines  $METAL"
echo "  --- first 16 lines ---"
sed -n '1,16p' "$METAL"
echo "  ..."

echo
echo "== run (4-digit demo)"
nvcc -O3 -std=c++17 -o hashhunt_gpu hashhunt_gpu.cu
c++ -O3 -std=c++17 -o hashhunt_cpu hashhunt_cpu.cpp
./hashhunt_gpu
echo
echo "== bench (all 100e6 8-digit strings)"
./hashhunt_gpu --bench
./hashhunt_cpu --bench --threads 1
./hashhunt_cpu --bench
