#!/bin/bash
# Build a release artifact: mvcc-<version>-macos-arm64.tar.gz containing the toolkit/ layout (what CMake and
# plain nvcc invocations expect), plus a .sha256. Optionally code-signs the Mach-O binaries.
#
#   tools/release.sh [--no-build] [--no-test] [--out DIR]
#   CODESIGN_IDENTITY="Developer ID Application: ..."  sign bin/* and lib64/*.dylib (codesign --options runtime)
#   NOTARIZE_PROFILE=<keychain profile>                submit the tarball with notarytool and staple (needs the identity)
#
# Version: the workspace version in Cargo.toml (single source; install_toolkit.sh writes it into version.json together
# with the validated macOS range). Refuses to package a dirty tree unless RELEASE_ALLOW_DIRTY=1.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TK="$ROOT/toolkit"
BUILD=1; TEST=1; OUT="$ROOT/dist"
while [ $# -gt 0 ]; do case "$1" in --no-build) BUILD=0 ;; --no-test) TEST=0 ;; --out) OUT="$2"; shift ;; *) echo "unknown option $1" >&2; exit 2 ;; esac; shift; done

VERSION="$(sed -n 's/^version = "\(.*\)"/\1/p' "$ROOT/Cargo.toml" | head -1)"
[ -n "$VERSION" ] || { echo "release: no workspace version in Cargo.toml" >&2; exit 1; }
if [ -n "$(cd "$ROOT" && git status --porcelain --untracked-files=no)" ] && [ "${RELEASE_ALLOW_DIRTY:-0}" != 1 ]; then
  echo "release: working tree has uncommitted changes (RELEASE_ALLOW_DIRTY=1 to override)" >&2; exit 1
fi
COMMIT="$(cd "$ROOT" && git rev-parse --short HEAD)"
HOST_MACOS="$(sw_vers -productVersion)"

if [ "$BUILD" = 1 ]; then "$ROOT/tools/install_toolkit.sh" release; fi

# the release must pass its own suite on the building machine (quick set: no roofline/qwen)
if [ "$TEST" = 1 ]; then "$ROOT/tests/run.sh" --no-build --quick; fi


if [ -n "${CODESIGN_IDENTITY:-}" ]; then
  echo "== codesign ($CODESIGN_IDENTITY)"
  for f in "$TK"/bin/mvcc "$TK"/bin/mvcc-ir2msl "$TK"/bin/mvcc-mslc "$TK"/lib64/libcudart.dylib; do
    codesign --force --timestamp --options runtime --sign "$CODESIGN_IDENTITY" "$f"
  done
else
  echo "== not signed (set CODESIGN_IDENTITY; unsigned binaries need 'xattr -d com.apple.quarantine' after download)"
fi

NAME="mvcc-$VERSION-macos-arm64"
mkdir -p "$OUT"
STAGE="$(mktemp -d)"; trap 'rm -rf "$STAGE"' EXIT
mkdir -p "$STAGE/$NAME"
# toolkit/ with symlinks preserved; include/ is a symlink to ../include in the tree, so copy the real headers
rsync -a --copy-unsafe-links --exclude '.DS_Store' "$TK/" "$STAGE/$NAME/toolkit/"
rm -rf "$STAGE/$NAME/toolkit/include"; cp -R "$ROOT/include" "$STAGE/$NAME/toolkit/include"
# stamp the staged copy only (toolkit/version.json in the tree stays as install_toolkit.sh wrote it)
python3 - "$STAGE/$NAME/toolkit/version.json" "$COMMIT" "$HOST_MACOS" <<'EOF'
import json, sys
p, commit, host = sys.argv[1:]
v = json.load(open(p))
v["mvcc"]["commit"] = commit
v["mvcc"]["built_on_macos"] = host
json.dump(v, open(p, "w"), indent=2); open(p, "a").write("\n")
EOF
cp "$ROOT/LICENSE" "$ROOT/NOTICE" "$ROOT/README.md" "$ROOT/CHANGELOG.md" "$STAGE/$NAME/"
LLVM_LICENSE="$(brew --prefix llvm)/LICENSE.TXT"
[ -f "$LLVM_LICENSE" ] || { echo "release: $LLVM_LICENSE not found (LLVM's license must ship with mvcc-ir2msl)" >&2; exit 1; }
cp "$LLVM_LICENSE" "$STAGE/$NAME/LICENSE-LLVM.txt"
# tracked files only: a working tree's examples/ also holds binaries, caches and .DS_Store from local runs
(cd "$ROOT" && git ls-files -z examples | tar -c --null -T - -f -) | tar -x -f - -C "$STAGE/$NAME"
cat > "$STAGE/$NAME/INSTALL.md" <<EOF
# mvcc $VERSION ($COMMIT)

A CUDA-compatible toolchain for Apple silicon. Built and validated on macOS $HOST_MACOS; see toolkit/version.json for
the macOS range this release is validated on. Requires Homebrew llvm (clang with the NVPTX target) at compile time.

    export PATH="\$PWD/toolkit/bin:\$PATH"
    nvcc --version
    nvcc -O2 -o vector_add tests/vector_add.cu   # any CUDA source

CMake: -DCMAKE_CUDA_COMPILER=\$PWD/toolkit/bin/nvcc -DCUDAToolkit_ROOT=\$PWD/toolkit -DCMAKE_CUDA_ARCHITECTURES=89

Unsigned builds: macOS quarantines downloaded binaries; run  xattr -dr com.apple.quarantine toolkit  once.
Docs: README.md (usage, tensor cores, differences from CUDA, supported CUDA surface, performance, environment variables).
EOF
mkdir -p "$STAGE/$NAME/tests"; cp "$ROOT"/tests/kernels/vector_add.cu "$STAGE/$NAME/tests/"
tar -C "$STAGE" -czf "$OUT/$NAME.tar.gz" "$NAME"
(cd "$OUT" && shasum -a 256 "$NAME.tar.gz" > "$NAME.tar.gz.sha256")

if [ -n "${NOTARIZE_PROFILE:-}" ]; then
  echo "== notarize ($NOTARIZE_PROFILE)"
  ditto -c -k --keepParent "$STAGE/$NAME" "$OUT/$NAME.zip"
  xcrun notarytool submit "$OUT/$NAME.zip" --keychain-profile "$NOTARIZE_PROFILE" --wait
fi

echo "release: $OUT/$NAME.tar.gz"
cat "$OUT/$NAME.tar.gz.sha256"
