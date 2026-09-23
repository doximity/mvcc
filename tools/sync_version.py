#!/usr/bin/env python3
"""Copy lib/mvcc/version.rb into VERSION, Cargo.toml (workspace.package) and the MVCC_VERSION_*
macros in include/mvcc/host_defines.h."""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main() -> int:
    rb = open(os.path.join(ROOT, "lib/mvcc/version.rb"), encoding="utf-8").read()
    m = re.search(r'VERSION = "([^"]+)"', rb)
    if not m:
        print("no VERSION = \"...\" in lib/mvcc/version.rb", file=sys.stderr)
        return 1
    version = m.group(1)
    open(os.path.join(ROOT, "VERSION"), "w", encoding="utf-8").write(version + "\n")
    cargo_path = os.path.join(ROOT, "Cargo.toml")
    cargo = open(cargo_path, encoding="utf-8").read()
    cargo_new, n = re.subn(r'(?m)^(version = ")[^"]+(")', r"\g<1>" + version + r"\g<2>", cargo, count=1)
    if n != 1:
        print("could not update workspace version in Cargo.toml", file=sys.stderr)
        return 1
    open(cargo_path, "w", encoding="utf-8").write(cargo_new)
    parts = re.match(r"(\d+)\.(\d+)\.(\d+)", version)
    if not parts:
        print(f"version {version!r} does not start with X.Y.Z", file=sys.stderr)
        return 1
    header_path = os.path.join(ROOT, "include/mvcc/host_defines.h")
    header = open(header_path, encoding="utf-8").read()
    for name, number in zip(("MAJOR", "MINOR", "PATCH"), parts.groups()):
        header, n = re.subn(rf"(?m)^(#define MVCC_VERSION_{name} )\d+$", rf"\g<1>{number}", header)
        if n != 1:
            print(f"could not update MVCC_VERSION_{name} in include/mvcc/host_defines.h", file=sys.stderr)
            return 1
    open(header_path, "w", encoding="utf-8").write(header)
    print(f"synced VERSION, Cargo.toml and host_defines.h to {version}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
