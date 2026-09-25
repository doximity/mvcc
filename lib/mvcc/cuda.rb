# frozen_string_literal: true

require "digest"
require "fileutils"
require "fiddle"
require "open3"
require "rbconfig"

module Mvcc
  module_function

  # Compile a .cu path (or source string) and return a callable library.
  def compile(src)
    Cuda.new(src)
  end

  # Host buffer the device can write. `puts buffer.lines(width)` is ASCII;
  # `puts buffer.halfblocks(width)` is truecolor ▀ (RGB, even pixel rows).
  def buffer(size)
    Buffer.new(size)
  end

  def libext
    RUBY_PLATFORM.include?("darwin") ? "dylib" : "so"
  end

  def shared_flags
    RUBY_PLATFORM.include?("darwin") ? [] : ["-Xcompiler", "-fPIC", "-arch", "sm_89"]
  end

  class Buffer
    attr_reader :size

    def initialize(size)
      @size = Integer(size)
      raise ArgumentError, "buffer size must be positive" unless @size.positive?

      @ptr = Fiddle::Pointer.malloc(@size, Fiddle::RUBY_FREE)
    end

    def to_ptr
      @ptr
    end

    def to_s
      @ptr[0, @size]
    end

    def [](off, len = nil)
      len ? @ptr[off, len] : @ptr[off]
    end

    def lines(width)
      width = Integer(width)
      raise ArgumentError, "line width must be positive" unless width.positive?

      height = @size / width
      (0...height).map { |row| self[row * width, width] }.join("\n")
    end

    def halfblocks(width)
      width = Integer(width)
      raise ArgumentError, "line width must be positive" unless width.positive?
      raise ArgumentError, "RGB buffer (3 bytes/pixel)" unless (@size % (width * 3)).zero?

      rows = @size / (width * 3)
      raise ArgumentError, "even pixel-row count for ▀" unless rows.even?

      raw = to_s.b
      (0...rows).step(2).map do |y|
        cells = width.times.map do |x|
          t = (y * width + x) * 3
          b = ((y + 1) * width + x) * 3
          format(
            "\e[38;2;%d;%d;%dm\e[48;2;%d;%d;%dm▀",
            raw.getbyte(t), raw.getbyte(t + 1), raw.getbyte(t + 2),
            raw.getbyte(b), raw.getbyte(b + 1), raw.getbyte(b + 2)
          )
        end
        "#{cells.join}\e[0m"
      end.join("\n")
    end
  end

  class Cuda
    attr_reader :so

    def self.open(so)
      new(nil, so: so)
    end

    def initialize(src, so: nil, flags: [])
      if so
        @so = File.expand_path(so)
      else
        raise ArgumentError, "CUDA source or path required" if src.nil?

        src = File.read(src) if src_path?(src)
        @so = compile!(src, flags)
      end
      @lib = Fiddle.dlopen(@so)
    end

    def call(name, *args)
      types = args.map { |a| self.class.fiddle_type(a) }
      native = args.map { |a| a.is_a?(Buffer) ? a.to_ptr : a }
      Fiddle::Function.new(@lib[name.to_s], types, Fiddle::TYPE_VOID).call(*native)
      args.find { |a| a.is_a?(Buffer) } || self
    end

    def method_missing(name, *args)
      call(name, *args)
    rescue Fiddle::DLError
      super
    end

    def respond_to_missing?(name, include_private = false)
      @lib[name.to_s]
      true
    rescue Fiddle::DLError
      super
    end

    def self.fiddle_type(arg)
      case arg
      when Integer then Fiddle::TYPE_INT
      when Float then Fiddle::TYPE_FLOAT
      else Fiddle::TYPE_VOIDP
      end
    end

    private

    def src_path?(src)
      src.is_a?(String) && !src.include?("\n") && File.file?(src)
    end

    def compile!(src, flags)
      dest = cache_dir(src)
      cu = File.join(dest, "kernel.cu")
      so = File.join(dest, "kernel.#{Mvcc.libext}")
      return so if File.file?(so)

      FileUtils.mkdir_p(dest)
      File.write(cu, src)
      cmd = [Mvcc.nvcc, "-shared", *Mvcc.shared_flags, *flags, "-o", so, cu]
      warn("mvcc: #{cmd.join(" ")}") if ENV["MVCC_VERBOSE"] == "1"
      out, status = Open3.capture2e(*cmd)
      abort(out.empty? ? "mvcc: #{cmd.join(" ")} failed" : out) unless status.success?
      so
    end

    def cache_dir(src)
      key = Digest::SHA256.hexdigest([src, Mvcc.nvcc, *Mvcc.shared_flags].join("\0"))
      File.join(self.class.cache_root, key)
    end

    def self.cache_root
      if RUBY_PLATFORM.include?("darwin")
        File.join(Dir.home, "Library", "Caches", "mvcc", "ruby")
      else
        File.join(ENV["XDG_CACHE_HOME"] || File.join(Dir.home, ".cache"), "mvcc", "ruby")
      end
    end
  end
end
