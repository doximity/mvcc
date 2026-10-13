class CudaMvcc < Formula
  desc "CUDA C++ compiler and runtime for Apple silicon"
  homepage "https://github.com/doximity/mvcc"
  # brew-stable:begin
  url "https://github.com/doximity/mvcc/releases/download/v1.1.0/mvcc-1.1.0-macos-arm64.tar.gz"
  sha256 "0fc5e105d58341f6f4f4541a4e4ce4f41360a6fe42c950264bb40c9ec75acf4f"
  # brew-stable:end
  license "Apache-2.0"

  livecheck do
    url :homepage
    strategy :github_latest
  end

  head do
    url "https://github.com/doximity/mvcc.git", branch: "master"

    depends_on "cmake" => :build
    depends_on "ninja" => :build
    depends_on "rust" => :build
  end

  depends_on arch: :arm64
  # nvcc needs clang from the LLVM major recorded in libexec/version.json and looks for llvm@<major> before llvm.
  depends_on "llvm"
  depends_on macos: :tahoe
  depends_on "z3"
  depends_on "zstd"

  def install
    # The toolkit stays in libexec: linking its include/ and lib/ would put cuda.h, libcuda, libcudart and a stub
    # librt.a into HOMEBREW_PREFIX/include and lib, where unrelated builds pick them up.
    ENV["MVCC_DYLIB_ID"] = "#{opt_libexec}/lib64/libcudart.dylib"
    ENV["MVCC_Z3_LIB"] = formula_opt_lib("z3").to_s
    ENV["MVCC_ZSTD_LIB"] = formula_opt_lib("zstd").to_s
    if build.head?
      ENV["LLVM_DIR"] = (formula_opt_lib("llvm")/"cmake/llvm").to_s
      ENV["MVCC_PREFIX"] = libexec.to_s
      system "tools/install_toolkit.sh", "release"
    else
      system "tools/install_prefix.sh", libexec
    end
    bin.install_symlink libexec/"bin/nvcc", libexec/"bin/mvcc"
  end

  def caveats
    <<~EOS
      nvcc and mvcc are on your PATH. The CUDA headers and libraries stay in
      #{opt_libexec}, outside Homebrew's include and lib. For CMake:

        cmake -S . -B build \\
          -DCMAKE_CUDA_COMPILER=#{opt_bin}/nvcc \\
          -DCUDAToolkit_ROOT=#{opt_libexec} \\
          -DCMAKE_CUDA_ARCHITECTURES=89

      Compiling CUDA needs clang from the LLVM release recorded in
      #{opt_libexec}/version.json. After Homebrew's llvm moves past
      it, install llvm@<that major>; nvcc looks for it before llvm.
      The runtime is Philox-only for cuRAND; see the mvcc README.
    EOS
  end

  test do
    assert_match "Cuda compiler driver", shell_output("#{bin}/nvcc --version")
    (testpath/"hello.cu").write <<~CUDA
      __global__ void k(int* p) { *p = 1; }
      int main() { return 0; }
    CUDA
    system bin/"nvcc", "-O2", "-c", "hello.cu", "-o", "hello.o"
    assert_path_exists testpath/"hello.o"
  end
end
