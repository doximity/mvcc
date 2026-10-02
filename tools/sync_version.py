#!/usr/bin/env python3
"""Copy the Cargo.toml workspace version into VERSION and the MVCC_VERSION_*
macros in include/mvcc/host_defines.h.

Usage: tools/sync_version.py [--check]
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def cargo_version() -> str:
    cargo = re.search(r'(?m)^version = "([^"]+)"', read("Cargo.toml"))
    if cargo is None:
        print("no workspace version in Cargo.toml", file=sys.stderr)
        sys.exit(1)
    version = cargo.group(1)
    if not re.fullmatch(r"\d+\.\d+\.\d+(-rc\.\d+)?", version):
        print(f"version {version!r} must be X.Y.Z or X.Y.Z-rc.N", file=sys.stderr)
        sys.exit(1)
    return version


def header_version(header: str) -> str | None:
    macros = [re.search(rf"(?m)^#define MVCC_VERSION_{name} (\d+)$", header) for name in ("MAJOR", "MINOR", "PATCH")]
    if not all(macros):
        return None
    return ".".join(m.group(1) for m in macros)


def check(version: str) -> int:
    file_version = read("VERSION").strip()
    if file_version != version:
        print(
            f"VERSION file {file_version!r} != Cargo.toml {version!r} (run tools/sync_version.py)",
            file=sys.stderr,
        )
        return 1
    got = header_version(read("include/mvcc/host_defines.h"))
    if got != version.split("-")[0]:
        print(
            f"MVCC_VERSION_* in include/mvcc/host_defines.h {got!r} != Cargo.toml {version!r} (run tools/sync_version.py)",
            file=sys.stderr,
        )
        return 1
    print("version checks ok")
    return 0


def write(version: str) -> int:
    with open(os.path.join(ROOT, "VERSION"), "w", encoding="utf-8") as f:
        f.write(version + "\n")
    parts = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if not parts:
        print(f"version {version!r} does not start with X.Y.Z", file=sys.stderr)
        return 1
    header_path = os.path.join(ROOT, "include/mvcc/host_defines.h")
    header = read("include/mvcc/host_defines.h")
    for name, number in zip(("MAJOR", "MINOR", "PATCH"), parts.groups()):
        header, n = re.subn(rf"(?m)^(#define MVCC_VERSION_{name} )\d+$", rf"\g<1>{number}", header)
        if n != 1:
            print(f"could not update MVCC_VERSION_{name} in include/mvcc/host_defines.h", file=sys.stderr)
            return 1
    with open(header_path, "w", encoding="utf-8") as f:
        f.write(header)
    print(f"synced VERSION and host_defines.h to {version}")
    return 0


def main() -> int:
    version = cargo_version()
    if "--check" in sys.argv:
        return check(version)
    return write(version)


if __name__ == "__main__":
    sys.exit(main())
