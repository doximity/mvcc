#!/bin/bash
# Fail if NVIDIA-owned copyright text or a libdevice bitcode reference appears in the tree.
# This script and the test runner mention those strings; exclude them from the search.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

exclude='^\./(\.git/|tests/run\.sh|tools/check_provenance\.sh)'

hits=$(grep -rIl --exclude-dir=target --exclude-dir=build --exclude-dir=toolkit --exclude-dir=.git \
  -e "NVIDIA CORPORATION" -e "Copyright.*NVIDIA" -e "NVIDIA Corporation" . \
  | grep -Ev "$exclude" || true)
if [ -n "$hits" ]; then
  echo "$hits"
  echo "NVIDIA copyright text found"
  exit 1
fi

hits=$(grep -rIl --exclude-dir=target --exclude-dir=build --exclude-dir=toolkit --exclude-dir=.git \
  -e "libdevice.10.bc" . | grep -Ev "$exclude" || true)
if [ -n "$hits" ]; then
  echo "$hits"
  echo "libdevice reference found"
  exit 1
fi

echo "provenance ok"
