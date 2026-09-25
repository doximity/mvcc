#!/usr/bin/env python3
"""Guards for the gem release path: one version across lib/mvcc/version.rb,
VERSION, Cargo.toml and include/mvcc/host_defines.h, a gemspec that builds
the toolkit from source on `gem install`, and pushes that can only go to
rubygems.org.

Usage: tools/check_gem_publish.py --check
"""
from __future__ import annotations

import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read(rel: str) -> str:
    with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
        return f.read()


def fail(msg: str) -> None:
    print(msg, file=sys.stderr)
    sys.exit(1)


def main() -> int:
    if "--check" not in sys.argv:
        print(__doc__.strip())
        return 2

    rb = re.search(r'VERSION = "([^"]+)"', read("lib/mvcc/version.rb"))
    if rb is None:
        fail("lib/mvcc/version.rb must set VERSION = \"X.Y.Z\"")
    version = rb.group(1)
    if not re.fullmatch(r"\d+\.\d+\.\d+(-rc\.\d+)?", version):
        fail(f"VERSION {version!r} must be X.Y.Z or X.Y.Z-rc.N")
    if read("VERSION").strip() != version:
        fail(f"VERSION file {read('VERSION').strip()!r} != lib/mvcc/version.rb {version!r} (run tools/sync_version.py)")

    cargo = re.search(r'(?m)^version = "([^"]+)"', read("Cargo.toml"))
    if cargo is None:
        fail("no workspace version in Cargo.toml")
    if cargo.group(1) != version:
        fail(f"Cargo.toml {cargo.group(1)!r} != lib/mvcc/version.rb {version!r} (run tools/sync_version.py)")
    header = read("include/mvcc/host_defines.h")
    macros = [re.search(rf"(?m)^#define MVCC_VERSION_{name} (\d+)$", header) for name in ("MAJOR", "MINOR", "PATCH")]
    header_version = ".".join(m.group(1) for m in macros) if all(macros) else None
    if header_version != version.split("-")[0]:
        fail(f"MVCC_VERSION_* in include/mvcc/host_defines.h {header_version!r} != lib/mvcc/version.rb {version!r} (run tools/sync_version.py)")

    gemspec = read("mvcc.gemspec")
    if "require_relative \"lib/mvcc/version\"" not in gemspec or "Mvcc::VERSION" not in gemspec:
        fail("mvcc.gemspec must require lib/mvcc/version.rb and set s.version = Mvcc::VERSION")
    ruby = read("lib/mvcc.rb")
    if 'require_relative "mvcc/version"' not in ruby:
        fail("lib/mvcc.rb must require mvcc/version")
    if 'require_relative "mvcc/cuda"' not in ruby:
        fail("lib/mvcc.rb must require mvcc/cuda")
    if '"allowed_push_host" => "https://rubygems.org"' not in gemspec:
        fail("mvcc.gemspec must set allowed_push_host to https://rubygems.org")
    if 's.extensions = ["ext/mvcc/extconf.rb"]' not in gemspec:
        fail("mvcc.gemspec must compile via ext/mvcc/extconf.rb on gem install")
    if "tools/install_toolkit.sh" not in gemspec:
        fail("mvcc.gemspec must pack tools/install_toolkit.sh")
    for needed in ("cpp/mvcc-llvm", "cpp/mvcc-metal", "crates/", "include/", "Cargo.toml"):
        if needed not in gemspec:
            fail(f"mvcc.gemspec must pack {needed} so gem install can build from source")

    extconf = read("ext/mvcc/extconf.rb")
    if "Mvcc::Installer.run!" not in extconf:
        fail("ext/mvcc/extconf.rb must run Mvcc::Installer")

    installer = read("lib/mvcc/installer.rb")
    if "tools/install_toolkit.sh" not in installer and "install_toolkit.sh" not in installer:
        fail("lib/mvcc/installer.rb must run tools/install_toolkit.sh")
    if "MVCC_ASK_PATH" not in installer:
        fail("lib/mvcc/installer.rb must honor MVCC_ASK_PATH")
    if "Add mvcc to PATH in your shell profile(s)?" not in installer:
        fail("lib/mvcc/installer.rb must prompt before writing profiles when MVCC_ASK_PATH=1")
    if "MVCC_SKIP_PATH" not in installer:
        fail("lib/mvcc/installer.rb must honor MVCC_SKIP_PATH / CI")
    if "$PWD/toolkit/bin" not in installer:
        fail("the PATH prompt must mention the checkout-equivalent $PWD/toolkit/bin")
    if '%(export PATH="#{path}:$PATH")' not in installer:
        fail("export_line must write the installed toolkit bindir, not $PWD")

    print("gem publish checks ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
