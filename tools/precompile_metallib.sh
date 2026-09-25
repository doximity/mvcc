#!/bin/bash
# Offline-compile an mvcc module's MSL (from `nvcc -keep` / MVCC_KEEP=1: <name>.metal) into a .metallib that
# the runtime loads instead of compiling source, when MVCC_METALLIB_DIR points at the output directory.
# Requires Xcode's Metal toolchain (`xcrun metal`, `xcrun metallib`). See README.md.
#
# usage: tools/precompile_metallib.sh <module.metal> [outdir=.]
set -euo pipefail
SRC="${1:?usage: precompile_metallib.sh <module.metal> [outdir]}"
OUT="${2:-.}"
mkdir -p "$OUT"
# Same key the runtime uses: FNV-1a 64 over the exact MSL bytes.
HASH="$(python3 - "$SRC" <<'EOF'
import sys
h = 0xcbf29ce484222325
for b in open(sys.argv[1], 'rb').read():
    h = ((h ^ b) * 0x100000001b3) & 0xffffffffffffffff
print(f"{h:016x}")
EOF
)"
TMP="$(mktemp -d)"
xcrun -sdk macosx metal -std=metal4.0 -O2 -fno-fast-math -c "$SRC" -o "$TMP/m.air"
xcrun -sdk macosx metallib "$TMP/m.air" -o "$OUT/$HASH.metallib"
rm -rf "$TMP"
echo "$OUT/$HASH.metallib"
