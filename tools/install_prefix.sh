#!/bin/bash
# Copy toolkit/ into DEST with the CUDA layout a Homebrew prefix (or any relocatable install) needs.
#
#   tools/install_prefix.sh DEST
#
# DEST receives bin/, lib64/, lib/ (symlinks into lib64), include/, share/, nvvm/ and version.json.
# include/ in a source-tree toolkit is a symlink back at the repo; this copies the real headers.
#
# MVCC_DYLIB_ID   LC_ID_DYLIB for libcudart (default: $DEST/lib64/libcudart.dylib)
# MVCC_Z3_LIB     directory of libz3 to retarget in mvcc-ir2msl (Homebrew opt lib)
# MVCC_ZSTD_LIB   directory of libzstd to retarget in mvcc-ir2msl
# MVCC_TOOLKIT    toolkit tree to copy (default: <repo>/toolkit)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-}"
SRC="${MVCC_TOOLKIT:-$ROOT/toolkit}"
case "$DEST" in
  ""|"/"|"$ROOT"|"$SRC") echo "install_prefix: refusing DEST='$DEST'" >&2; exit 1 ;;
esac
[ -x "$SRC/bin/mvcc" ] || { echo "install_prefix: no toolkit at $SRC (run tools/install_toolkit.sh)" >&2; exit 1; }
[ -f "$SRC/lib64/libcudart.dylib" ] || { echo "install_prefix: $SRC/lib64/libcudart.dylib missing" >&2; exit 1; }

mkdir -p "$DEST"
DEST="$(cd "$DEST" && pwd -P)"
python3 - "$DEST" <<'PY'
import os, stat, sys
mode = os.stat(sys.argv[1]).st_mode
if mode & stat.S_IWOTH:
    sys.exit("install_prefix: destination is world-writable")
PY
rm -rf "$DEST/bin" "$DEST/lib" "$DEST/lib64" "$DEST/include" "$DEST/share" "$DEST/nvvm"
cp -a "$SRC/bin" "$DEST/bin"
cp -a "$SRC/lib64" "$DEST/lib64"
cp -a "$SRC/share" "$DEST/share"
if [ -d "$SRC/nvvm" ]; then cp -a "$SRC/nvvm" "$DEST/nvvm"; fi
cp -a "$SRC/version.json" "$DEST/version.json"
mkdir -p "$DEST/include"
if [ -L "$SRC/include" ]; then
  cp -R "$ROOT/include/." "$DEST/include/"
else
  cp -R "$SRC/include/." "$DEST/include/"
fi

mkdir -p "$DEST/lib"
for f in "$DEST/lib64"/*; do
  ln -sfn "../lib64/$(basename "$f")" "$DEST/lib/$(basename "$f")"
done

python3 - "$DEST" <<'PY'
import os, sys
dest = os.path.realpath(sys.argv[1])
bad = []
for dirpath, dirnames, filenames in os.walk(dest, followlinks=False):
    for name in list(dirnames) + list(filenames):
        p = os.path.join(dirpath, name)
        if not os.path.islink(p):
            continue
        real = os.path.realpath(p)
        if real != dest and not real.startswith(dest + os.sep):
            bad.append("%s -> %s" % (p, real))
if bad:
    sys.stderr.write("install_prefix: symlink escapes the prefix:\n" + "\n".join(bad) + "\n")
    sys.exit(1)
PY

ID="${MVCC_DYLIB_ID:-$DEST/lib64/libcudart.dylib}"
case "$ID" in
  @*|*@executable_path*|*@loader_path*|*@rpath*|*..*)
    echo "install_prefix: refusing library id '$ID'" >&2; exit 1 ;;
esac
case "$ID" in
  /*libcudart.dylib) ;;
  *) echo "install_prefix: library id must be an absolute path ending in libcudart.dylib" >&2; exit 1 ;;
esac
for extra in "${MVCC_Z3_LIB:-}" "${MVCC_ZSTD_LIB:-}"; do
  case "$extra" in
    ""|/*) ;;
    *) echo "install_prefix: dependency directory must be absolute ($extra)" >&2; exit 1 ;;
  esac
done
install_name_tool -id "$ID" "$DEST/lib64/libcudart.dylib"
got_id="$(otool -D "$DEST/lib64/libcudart.dylib" | sed -n '2p')"
if [ "$got_id" != "$ID" ]; then
  echo "install_prefix: libcudart id is '$got_id', want '$ID'" >&2
  exit 1
fi

# mvcc-ir2msl records the build machine's Homebrew z3 and zstd. Point them at this machine's opt libs
# when the caller says where those are (the formula sets MVCC_Z3_LIB / MVCC_ZSTD_LIB).
relink() {
  local file="$1" dest_dir="$2" needle="$3" generic="$4"
  [ -n "$dest_dir" ] || return 0
  local lib base target
  while IFS= read -r lib; do
    case "$lib" in
      /*) ;;
      *) continue ;;
    esac
    case "$lib" in
      *"$needle"*)
        base="$(basename "$lib")"
        if [ -e "$dest_dir/$base" ]; then target="$dest_dir/$base"
        elif [ -e "$dest_dir/$generic" ]; then target="$dest_dir/$generic"
        else echo "install_prefix: $dest_dir has neither $base nor $generic" >&2; exit 1; fi
        case "$target" in
          /*) ;;
          *) echo "install_prefix: refusing relative dependency $target" >&2; exit 1 ;;
        esac
        if [ "$lib" != "$target" ]; then
          install_name_tool -change "$lib" "$target" "$file"
        fi
        ;;
    esac
  done < <(otool -L "$file" | awk 'NR>1 {print $1}')
}
relink "$DEST/bin/mvcc-ir2msl" "${MVCC_Z3_LIB:-}" "/z3/" "libz3.dylib"
relink "$DEST/bin/mvcc-ir2msl" "${MVCC_ZSTD_LIB:-}" "/zstd/" "libzstd.dylib"

# Hardened runtime blocks DYLD_INSERT_LIBRARIES. Library validation is off so Homebrew z3/zstd still load.
ENT="$(mktemp)"
cat > "$ENT" <<'EOF'
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>com.apple.security.cs.disable-library-validation</key><true/>
</dict></plist>
EOF
for bin in "$DEST/bin/mvcc" "$DEST/bin/mvcc-ir2msl" "$DEST/bin/mvcc-mslc" "$DEST/lib64/libcudart.dylib" "$DEST/lib64/libmvcc-passes.dylib"; do
  [ -f "$bin" ] || continue
  codesign --force --sign - --options runtime --entitlements "$ENT" "$bin"
done
rm -f "$ENT"

echo "installed mvcc toolkit: $DEST"
"$DEST/bin/nvcc" --version | sed -n 1p
