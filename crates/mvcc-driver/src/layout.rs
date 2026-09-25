//! Locate the pieces of a mvcc installation relative to this executable.
//!
//! Two layouts are supported:
//!   installed:  <root>/bin/nvcc (this), <root>/bin/mvcc-ir2msl, <root>/bin/mvcc-mslc, <root>/include,
//!               <root>/lib64/libcudart.dylib, <root>/lib64/libmvcc-passes.dylib, <root>/share/mvcc/mvcc_prelude.metal
//!   source tree: <repo>/toolkit/bin/nvcc -> <repo>/target/{release,debug}/mvcc, tools in <repo>/build/...

use std::path::{Path, PathBuf};

pub struct Layout {
    pub toolkit: PathBuf,
    pub include: PathBuf,
    pub lib64: PathBuf,
    pub clang: PathBuf,
    pub ir2msl: PathBuf,
    pub passes: PathBuf,   // clang pass plugin for the device pass (cpp/mvcc-llvm/nvptx_convergence.cpp)
    pub mslc: Option<PathBuf>,
    pub prelude: PathBuf,
}

impl Layout {
    pub fn discover() -> Result<Layout, String> {
        let exe = std::env::current_exe().map_err(|e| e.to_string())?;
        let real = std::fs::canonicalize(&exe).unwrap_or(exe.clone());
        let mut roots: Vec<PathBuf> = Vec::new();
        if let Ok(r) = std::env::var("MVCC_ROOT") { roots.push(PathBuf::from(r)); }
        // invoked as <root>/bin/nvcc (symlink or copy)
        if let Some(bin) = exe.parent() { if let Some(r) = bin.parent() { roots.push(r.to_path_buf()); } }
        if let Some(bin) = real.parent() {
            if let Some(r) = bin.parent() { roots.push(r.to_path_buf()); }
            // <repo>/target/<profile>/mvcc
            if let Some(repo) = bin.parent().and_then(|t| t.parent()) { roots.push(repo.join("toolkit")); roots.push(repo.to_path_buf()); }
        }

        let toolkit = roots.iter().find(|r| r.join("include/cuda_runtime.h").exists() && (r.join("lib64").exists() || r.join("bin").exists()))
            .cloned().ok_or_else(|| format!("cannot locate the mvcc toolkit (looked in {:?}); set MVCC_ROOT", roots))?;
        let repo = toolkit.parent().map(|p| p.to_path_buf()).unwrap_or_else(|| toolkit.clone());

        let find = |names: &[PathBuf]| names.iter().find(|p| p.exists()).cloned();
        let ir2msl = find(&[toolkit.join("bin/mvcc-ir2msl"), repo.join("build/mvcc-llvm/mvcc-ir2msl")])
            .ok_or("cannot find mvcc-ir2msl (build cpp/mvcc-llvm or install the toolkit)")?;
        let passes = find(&[toolkit.join("lib64/libmvcc-passes.dylib"), repo.join("build/mvcc-llvm/libmvcc-passes.dylib")])
            .ok_or("cannot find libmvcc-passes.dylib (build cpp/mvcc-llvm or install the toolkit)")?;
        let mslc = find(&[toolkit.join("bin/mvcc-mslc"), repo.join("build/mvcc-mslc")]);
        let prelude = find(&[toolkit.join("share/mvcc/mvcc_prelude.metal"), repo.join("msl/mvcc_prelude.metal")])
            .ok_or("cannot find mvcc_prelude.metal")?;
        let clang = find_clang()?;
        Ok(Layout { include: toolkit.join("include"), lib64: toolkit.join("lib64"), toolkit, clang, ir2msl, passes, mslc, prelude })
    }

    pub fn dump(&self) {
        eprintln!("#$ _MVCC_TOOLKIT_={}", self.toolkit.display());
        eprintln!("#$ _MVCC_CLANG_={}", self.clang.display());
        eprintln!("#$ _MVCC_IR2MSL_={}", self.ir2msl.display());
        eprintln!("#$ _MVCC_PRELUDE_={}", self.prelude.display());
        // Lines CMake's CMakeNVCCParseImplicitInfo reads.
        eprintln!("#$ PATH={}", std::env::var("PATH").unwrap_or_default());
        eprintln!("#$ INCLUDES=\"-I{}\"", self.include.display());
        eprintln!("#$ SYSTEM_INCLUDES=");
        eprintln!("#$ LIBRARIES={}", self.libraries_string());
        eprintln!("#$ device compilation: -arch compute_89 (Metal; __CUDA_ARCH__=890)");
    }

    /// The exact text CMake expects to find again inside the link command line.
    pub fn libraries_string(&self) -> String { format!("\"-L{}\" -lcudadevrt -lcudart", self.lib64.display()) }
}

fn find_clang() -> Result<PathBuf, String> {
    if let Ok(c) = std::env::var("MVCC_CLANG") { let p = PathBuf::from(c); if p.exists() { return Ok(p); } return Err(format!("MVCC_CLANG={} does not exist", p.display())); }
    let candidates = [
        "/opt/homebrew/opt/llvm/bin/clang++",
        "/usr/local/opt/llvm/bin/clang++",
        "/opt/homebrew/opt/llvm@23/bin/clang++",
        "/opt/homebrew/opt/llvm@22/bin/clang++",
        "/opt/homebrew/opt/llvm@21/bin/clang++",
    ];
    for c in candidates { let p = Path::new(c); if p.exists() { return Ok(p.to_path_buf()); } }
    // any clang++ on PATH that knows the NVPTX target
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            let p = Path::new(dir).join("clang++");
            if p.exists() && clang_has_nvptx(&p) { return Ok(p); }
        }
    }
    Err("no clang++ with NVPTX support found; `brew install llvm` or set MVCC_CLANG".into())
}

fn clang_has_nvptx(p: &Path) -> bool {
    std::process::Command::new(p).arg("-print-targets").output().map(|o| String::from_utf8_lossy(&o.stdout).contains("nvptx64")).unwrap_or(false)
}
