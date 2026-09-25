# frozen_string_literal: true

require_relative "lib/mvcc/version"

Gem::Specification.new do |s|
  s.name = "mvcc"
  # Shared with VERSION and Cargo.toml; tools/check_gem_publish.py refuses drift.
  s.version = Mvcc::VERSION
  s.summary = "CUDA-compatible toolchain for Apple silicon"
  s.description = "Compiles CUDA C++ to native Metal 4 GPU code and ships a cudart-compatible runtime."
  s.authors = ["Doximity"]
  s.homepage = "https://github.com/doximity/mvcc"
  s.license = "Apache-2.0"
  s.metadata = {
    "source_code_uri" => "https://github.com/doximity/mvcc",
    "changelog_uri" => "https://github.com/doximity/mvcc/blob/master/CHANGELOG.md",
    "bug_tracker_uri" => "https://github.com/doximity/mvcc/issues",
    "allowed_push_host" => "https://rubygems.org"
  }
  skip = %w[build target toolkit dist .git]
  s.files = Dir[
    "lib/**/*.rb",
    "ext/mvcc/extconf.rb",
    "tools/install_toolkit.sh",
    "tools/mslc/**/*",
    "cpp/mvcc-llvm/**/*",
    "cpp/mvcc-metal/**/*",
    "crates/**/*",
    "include/**/*",
    "msl/**/*",
    "Cargo.toml",
    "Cargo.lock",
    "VERSION",
    "LICENSE",
    "NOTICE",
    "README.md"
  ].select { |f| File.file?(f) }.reject do |f|
    f.split("/").any? { |part| skip.include?(part) }
  end
  s.extensions = ["ext/mvcc/extconf.rb"]
  s.require_paths = ["lib"]
  s.required_ruby_version = ">= 3.1"
end
