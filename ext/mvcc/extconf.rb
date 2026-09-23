# frozen_string_literal: true

# Native-extension hook so `gem install mvcc` builds the toolkit from the
# sources packed in the gem. The dummy Makefile satisfies RubyGems; the
# compiler lives in toolkit/ after tools/install_toolkit.sh.

require_relative "../../lib/mvcc/installer"

Mvcc::Installer.run!

File.write(File.expand_path("Makefile", __dir__), <<~MAKE)
  .PHONY: all install clean
  all:
  \t@true
  install:
  \t@true
  clean:
  \t@true
MAKE
