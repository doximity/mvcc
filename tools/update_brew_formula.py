#!/usr/bin/env python3
"""Fill the stable url and sha256 in Formula/cuda-mvcc.rb from a release tarball.

    tools/update_brew_formula.py dist/mvcc-1.0.1-macos-arm64.tar.gz

The tarball is the one tools/release.sh writes. Publish it as a GitHub release asset named
mvcc-<version>-macos-arm64.tar.gz on tag v<version>, then run this and commit the formula.
"""
from __future__ import annotations

import hashlib
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FORMULA = os.path.join(ROOT, "Formula", "cuda-mvcc.rb")
BEGIN = "  # brew-stable:begin"
END = "  # brew-stable:end"
NAME = re.compile(r"^mvcc-(\d+\.\d+\.\d+(?:-rc\.\d+)?)-macos-arm64\.tar\.gz$")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: tools/update_brew_formula.py dist/mvcc-<version>-macos-arm64.tar.gz", file=sys.stderr)
        return 2
    path = sys.argv[1]
    base = os.path.basename(path)
    m = NAME.match(base)
    if not m:
        print(f"tarball name {base!r} is not mvcc-<version>-macos-arm64.tar.gz", file=sys.stderr)
        return 1
    version = m.group(1)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        print("sha256 was not 64 hex characters", file=sys.stderr)
        return 1
    # version and sha are drawn from a fixed alphabet, so they cannot break out of the Ruby string.
    text = open(FORMULA, encoding="utf-8").read()
    if BEGIN not in text or END not in text:
        print(f"{FORMULA} is missing brew-stable markers", file=sys.stderr)
        return 1
    url = f"https://github.com/doximity/mvcc/releases/download/v{version}/{base}"
    pattern = re.compile(re.escape(BEGIN) + r".*?" + re.escape(END), re.S)
    new, n = pattern.subn(f'{BEGIN}\n  url "{url}"\n  sha256 "{sha}"\n{END}', text, count=1)
    if n != 1:
        print("could not replace the brew-stable block", file=sys.stderr)
        return 1
    with open(FORMULA, "w", encoding="utf-8") as f:
        f.write(new)
    print(f"cuda-mvcc {version} sha256 {sha}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
