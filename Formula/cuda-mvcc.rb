class CudaMvcc < Formula
  desc "CUDA C++ compiler and runtime for Apple silicon"
  homepage "https://github.com/doximity/mvcc"
  license "Apache-2.0"
  # brew-stable:begin
  # Filled by tools/update_brew_formula.py when a GitHub release asset exists.
  # Until then this is a head-only formula: brew install --HEAD cuda-mvcc
  # brew-stable:end
  head "https://github.com/doximity/mvcc.git", branch: "master"

  livecheck do
    url :homepage
    strategy :github_latest
  end

  depends_on "llvm"
  depends_on "z3"
  depends_on "zstd"
  depends_on arch: :arm64
  depends_on macos: :tahoe

  on_head do
    depends_on "cmake" => :build
    depends_on "ninja" => :build
    depends_on "rust" => :build
  end

  def install
    ENV["MVCC_DYLIB_ID"] = "#{opt_prefix}/lib64/libcudart.dylib"
    ENV["MVCC_Z3_LIB"] = Formula["z3"].opt_lib.to_s
    ENV["MVCC_ZSTD_LIB"] = Formula["zstd"].opt_lib.to_s
    if build.head?
      ENV["LLVM_DIR"] = (Formula["llvm"].opt_lib/"cmake/llvm").to_s
      ENV["MVCC_PREFIX"] = prefix.to_s
      system "tools/install_toolkit.sh", "release"
    else
      system "tools/install_prefix.sh", prefix
    end
  end

  def caveats
    <<~EOS
      nvcc is on your PATH. Point CMake at this prefix:

        cmake -S . -B build \\
          -DCMAKE_CUDA_COMPILER=#{opt_bin}/nvcc \\
          -DCUDAToolkit_ROOT=#{opt_prefix} \\
          -DCMAKE_CUDA_ARCHITECTURES=89

      Compiling CUDA still needs Homebrew llvm (a dependency). The runtime is
      Philox-only for cuRAND; see the mvcc README.
    EOS
  end

  test do
    (testpath/"hello.cu").write <<~CUDA
      __global__ void k(int* p) { *p = 1; }
      int main() { return 0; }
    CUDA
    ENV["MVCC_CLANG"] = (Formula["llvm"].opt_bin/"clang++").to_s
    system bin/"nvcc", "-O2", "-c", "hello.cu", "-o", "hello.o"
  end
end
