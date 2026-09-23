# frozen_string_literal: true

require_relative "mvcc/version"
require_relative "mvcc/cuda"

module Mvcc
  module_function

  def root
    File.expand_path("..", __dir__)
  end

  def toolkit
    File.join(root, "toolkit")
  end

  def bindir
    File.join(toolkit, "bin")
  end

  # Prefer NVCC, then this install's toolkit, then another installed mvcc gem,
  # then PATH. Skip-build local installs reuse a previously built toolkit.
  def nvcc
    explicit = ENV["NVCC"]
    return explicit if explicit && !explicit.empty? && File.executable?(explicit)

    each_nvcc_candidate.find { |path| File.executable?(path) } || File.join(bindir, "nvcc")
  end

  def each_nvcc_candidate
    return enum_for(__method__) unless block_given?

    yield File.join(bindir, "nvcc")
    if defined?(Gem)
      Gem::Specification.find_all_by_name("mvcc").each do |spec|
        yield File.join(spec.full_gem_path, "toolkit", "bin", "nvcc")
      end
    end
    ENV.fetch("PATH", "").split(File::PATH_SEPARATOR).each do |dir|
      next if dir.empty?

      yield File.join(dir, "nvcc")
    end
  end
end
