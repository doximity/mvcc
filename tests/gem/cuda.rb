#!/usr/bin/env ruby
# frozen_string_literal: true

# Ruby sugar for inline CUDA. Does not need a GPU: Buf / Fiddle types use a
# host .dylib, compile uses a fake NVCC. Run: ruby tests/gem/cuda.rb

require "open3"
require "rbconfig"
require "tmpdir"

require_relative "../../lib/mvcc"

fail = 0
def check(name, cond)
  if cond
    puts "  [ok]   #{name}"
  else
    $fail += 1
    puts "  [FAIL] #{name}"
  end
end
$fail = 0

example = File.expand_path("../../examples/ruby/mandelbrot.rb", __dir__)
src = File.read(example)
check("example requires mvcc only", src.match?(/^require "mvcc"$/) && !src.include?("require \"fiddle\""))
check("example uses Mvcc.compile and Mvcc.buffer", src.include?("Mvcc.compile") && src.include?("Mvcc.buffer"))
check("example calls render and lines", src.include?("k.render") && src.include?(".lines("))
check("example inlines CUDA with a heredoc", src.include?("<<~'CU'") && src.include?("__global__"))
check("example animates a 3d structure", src.include?("kaleidoscopic IFS") && src.include?("0.step"))

quat = File.read(File.expand_path("../../examples/ruby/quaternion.rb", __dir__))
check("quaternion requires mvcc only", quat.match?(/^require "mvcc"$/) && !quat.include?("require \"fiddle\""))
check("quaternion uses halfblocks", quat.include?("Mvcc.compile") && quat.include?("halfblocks"))
check("quaternion is z^2+c on quaternions", quat.include?("qmul") && quat.include?("0.5f * r * logf"))

forest = File.read(File.expand_path("../../examples/ruby/forest.rb", __dir__))
check("forest requires mvcc only", forest.match?(/^require "mvcc"$/) && !forest.include?("require \"fiddle\""))
check("forest uses halfblocks", forest.include?("halfblocks") && forest.include?("Mvcc.compile"))
check("forest is a flight grid", forest.include?("float tree(") && forest.include?("yaw") && forest.include?("readpartial"))

buf = Mvcc.buffer(6)
check("buffer size", buf.size == 6)
buf.to_ptr[0, 6] = "abcdef"
check("buffer to_s", buf.to_s == "abcdef")
check("buffer lines", buf.lines(3) == "abc\ndef")
check("buffer slice", buf[3, 3] == "def")
rgb = Mvcc.buffer(12)
rgb.to_ptr[0, 12] = [255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255].pack("C*")
hb = rgb.halfblocks(2)
check("halfblocks pairs RGB rows", hb.include?("▀") && hb.include?("38;2;255;0;0") && hb.include?("48;2;0;0;255"))

check("int type", Mvcc::Cuda.fiddle_type(48) == Fiddle::TYPE_INT)
check("float type", Mvcc::Cuda.fiddle_type(1.5) == Fiddle::TYPE_FLOAT)
check("buffer type", Mvcc::Cuda.fiddle_type(buf) == Fiddle::TYPE_VOIDP)
check("libext", Mvcc.libext == (RUBY_PLATFORM.include?("darwin") ? "dylib" : "so"))
check("shared flags empty on darwin", !RUBY_PLATFORM.include?("darwin") || Mvcc.shared_flags.empty?)

cc = RbConfig::CONFIG["CC"] || "cc"
so = File.join(Dir.mktmpdir("mvcc-ruby-test-"), "host.#{Mvcc.libext}")
c_src = <<~C
  void mandelbrot(char* host, int w, int h) {
    int n = w * h, i;
    for (i = 0; i < n; i++) host[i] = (char)((i % w == 0) ? '#' : '.');
  }
  void rotate90ccw(char* out, const char* in, int w, int h) {
    int i;
    for (i = 0; i < w * h; i++) {
      int x = i % w, y = i / w;
      out[(w - 1 - x) * h + y] = in[i];
    }
  }
C
out, status = Open3.capture2e(cc, "-shared", "-o", so, "-x", "c", "-", stdin_data: c_src)
check("host shared lib builds", status.success?)
unless status.success?
  puts out
end

if status.success?
  frame = Mvcc.buffer(8)
  lib = Mvcc::Cuda.open(so)
  ret = lib.mandelbrot(frame, 4, 2)
  check("method_missing calls mandelbrot", frame.to_s == "#...#...")
  check("mandelbrot returns the buffer", ret.equal?(frame))
  check("lines after call", frame.lines(4) == "#...\n#...")
  rot = lib.rotate90ccw(Mvcc.buffer(8), frame, 4, 2)
  check("rotate90ccw dest-first", rot.to_s == "......##" && rot.lines(2) == "..\n..\n..\n##")
end

Dir.mktmpdir("mvcc-ruby-nvcc-") do |dir|
  fake = File.join(dir, "nvcc")
  File.write(fake, <<~'SH')
    #!/bin/bash
    out=""
    while [ $# -gt 0 ]; do
      if [ "$1" = "-o" ]; then out="$2"; shift 2; continue; fi
      shift
    done
    echo 'void mandelbrot(char* h, int w, int ht) { int i; for (i = 0; i < w * ht; i++) h[i] = 46; }' \
      | cc -shared -o "$out" -x c -
  SH
  File.chmod(0o755, fake)
  ENV["NVCC"] = fake
  begin
    # Unique source so we do not reuse a previous cache entry.
    src = "/* #{dir} */\nextern \"C\" void mandelbrot(char* h, int w, int ht);"
    compiled = Mvcc.compile(src)
    frame = Mvcc.buffer(4)
    compiled.mandelbrot(frame, 2, 2)
    check("cuda compiles through NVCC", frame.to_s == "....")
    check("nvcc env wins", File.expand_path(Mvcc.nvcc) == File.expand_path(fake))
  ensure
    ENV.delete("NVCC")
  end
end

root = File.expand_path("../..", __dir__)
check("lib/mvcc.rb requires cuda", File.read(File.join(root, "lib/mvcc.rb")).include?('require_relative "mvcc/cuda"'))
check("VERSION matches the VERSION file", Mvcc::VERSION == File.read(File.join(root, "VERSION")).strip)

exit($fail.zero? ? 0 : 1)
