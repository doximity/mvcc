//! Locate the pieces of a mvcc installation relative to this executable.
//!
//! The search does not walk parent directories of a symlink. `current_exe` is canonicalized first, so a
//! `bin/nvcc` symlink planted next to a fake `lib64/` and `mvcc-ir2msl` cannot redirect the compiler or the
//! `-fpass-plugin` dylib. Every tool path must canonicalize to a file inside that same root.
//!
//! Two layouts are supported:
//!   installed:  <root>/bin/nvcc (this), <root>/bin/mvcc-ir2msl, <root>/bin/mvcc-mslc, <root>/include,
//!               <root>/lib64/libcudart.dylib (or <root>/lib/, for a Homebrew prefix), libmvcc-passes.dylib,
//!               <root>/share/mvcc/mvcc_prelude.metal
//!   source tree: <repo>/target/{release,debug}/mvcc, tools in <repo>/toolkit and <repo>/build

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
        let real = std::fs::canonicalize(&exe).map_err(|e| format!("cannot resolve {}: {}", exe.display(), e))?;
        let trust = if let Ok(r) = std::env::var("MVCC_ROOT") {
            let p = PathBuf::from(&r);
            if !p.is_absolute() { return Err("MVCC_ROOT must be an absolute path".into()); }
            std::fs::canonicalize(&p).map_err(|e| format!("MVCC_ROOT={}: {}", r, e))?
        } else {
            trust_root(&real)?
        };
        if !contained(&trust, &real)? {
            return Err(format!("{} resolves outside {}", real.display(), trust.display()));
        }
        let toolkit = resolve_toolkit(&trust, &real)?;
        let cudart = require_inside(&trust, &runtime_dylib(&toolkit)?, "libcudart")?;
        let libdir = cudart.parent().map(|p| p.to_path_buf()).ok_or("libcudart has no directory")?;
        let repo_build = trust.join("build/mvcc-llvm");
        let find_inside = |names: &[PathBuf]| -> Option<PathBuf> {
            names.iter().find_map(|p| require_inside(&trust, p, "tool").ok())
        };
        let ir2msl = find_inside(&[toolkit.join("bin/mvcc-ir2msl"), repo_build.join("mvcc-ir2msl")])
            .ok_or("cannot find mvcc-ir2msl inside the toolkit (build cpp/mvcc-llvm or install the toolkit)")?;
        let passes = find_inside(&[
            libdir.join("libmvcc-passes.dylib"),
            toolkit.join("lib64/libmvcc-passes.dylib"),
            toolkit.join("lib/libmvcc-passes.dylib"),
            repo_build.join("libmvcc-passes.dylib"),
        ]).ok_or("cannot find libmvcc-passes.dylib inside the toolkit")?;
        let mslc = find_inside(&[toolkit.join("bin/mvcc-mslc"), trust.join("build/mvcc-mslc")]);
        let prelude = find_inside(&[toolkit.join("share/mvcc/mvcc_prelude.metal"), trust.join("msl/mvcc_prelude.metal")])
            .ok_or("cannot find mvcc_prelude.metal inside the toolkit")?;
        let include = require_inside(&trust, &toolkit.join("include/cuda_runtime.h"), "cuda_runtime.h")?;
        let include = include.parent().map(|p| p.to_path_buf()).ok_or("include has no directory")?;
        let clang = find_clang()?;
        Ok(Layout { include, lib64: libdir, toolkit, clang, ir2msl, passes, mslc, prelude })
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

/// Install root of a canonical executable: `<root>/bin/mvcc`, the repo when the binary is
/// `<repo>/toolkit/bin/mvcc` (headers live next to `toolkit/`, not inside it), or the repo when
/// the binary is `target/<profile>/mvcc`.
fn trust_root(real_exe: &Path) -> Result<PathBuf, String> {
    let parent = real_exe.parent().ok_or_else(|| format!("{} has no directory", real_exe.display()))?;
    if parent.file_name().and_then(|s| s.to_str()) == Some("bin") {
        let root = parent.parent().ok_or_else(|| "bin has no parent".to_string())?;
        if root.file_name().and_then(|s| s.to_str()) == Some("toolkit") {
            if let Some(repo) = root.parent() { return Ok(repo.to_path_buf()); }
        }
        return Ok(root.to_path_buf());
    }
    if parent.parent().and_then(|p| p.file_name()).and_then(|s| s.to_str()) == Some("target") {
        if let Some(repo) = parent.parent().and_then(|t| t.parent()) { return Ok(repo.to_path_buf()); }
    }
    Err(format!("refusing to search parent directories of {} for a toolkit; set MVCC_ROOT to an absolute path", real_exe.display()))
}

fn resolve_toolkit(trust: &Path, real_exe: &Path) -> Result<PathBuf, String> {
    let mut cands: Vec<PathBuf> = Vec::new();
    if real_exe.starts_with(trust.join("bin")) { cands.push(trust.to_path_buf()); }
    cands.push(trust.join("toolkit"));
    if !cands.iter().any(|c| c == trust) { cands.push(trust.to_path_buf()); }
    for c in &cands {
        if is_toolkit(c) && contained(trust, c).unwrap_or(false) { return Ok(c.clone()); }
    }
    Err(format!("cannot locate the mvcc toolkit under {}; set MVCC_ROOT", trust.display()))
}

fn is_toolkit(root: &Path) -> bool {
    root.join("include/cuda_runtime.h").exists() && root.join("bin/mvcc-ir2msl").exists() && runtime_dylib(root).is_ok()
}

fn runtime_dylib(root: &Path) -> Result<PathBuf, String> {
    for rel in ["lib64/libcudart.dylib", "lib/libcudart.dylib"] {
        let p = root.join(rel);
        if p.exists() { return Ok(p); }
    }
    Err(format!("no libcudart.dylib under {}", root.display()))
}

fn contained(root: &Path, path: &Path) -> Result<bool, String> {
    let root = std::fs::canonicalize(root).map_err(|e| format!("{}: {}", root.display(), e))?;
    let path = std::fs::canonicalize(path).map_err(|e| format!("{}: {}", path.display(), e))?;
    Ok(path == root || path.starts_with(&root))
}

fn require_inside(root: &Path, path: &Path, what: &str) -> Result<PathBuf, String> {
    if !path.exists() { return Err(format!("{what} {} does not exist", path.display())); }
    let real = std::fs::canonicalize(path).map_err(|e| format!("{what} {}: {e}", path.display()))?;
    let base = std::fs::canonicalize(root).map_err(|e| format!("{root}: {e}", root = root.display()))?;
    if real != base && !real.starts_with(&base) {
        return Err(format!("{what} resolves to {}, outside {}; refusing a swapped or symlinked tool", real.display(), base.display()));
    }
    Ok(real)
}

fn find_clang() -> Result<PathBuf, String> {
    // Keep the `clang++` path. Canonicalizing it follows the symlink to `clang`, and argv[0] `clang`
    // does not link libc++.
    if let Ok(c) = std::env::var("MVCC_CLANG") {
        let p = PathBuf::from(&c);
        if !p.is_absolute() { return Err("MVCC_CLANG must be an absolute path".into()); }
        if !p.exists() { return Err(format!("MVCC_CLANG={} does not exist", c)); }
        return Ok(p);
    }
    let candidates = [
        "/opt/homebrew/opt/llvm/bin/clang++",
        "/usr/local/opt/llvm/bin/clang++",
        "/opt/homebrew/opt/llvm@23/bin/clang++",
        "/opt/homebrew/opt/llvm@22/bin/clang++",
        "/opt/homebrew/opt/llvm@21/bin/clang++",
    ];
    for c in candidates {
        let p = Path::new(c);
        if p.exists() { return Ok(p.to_path_buf()); }
    }
    // Absolute PATH entries only. An empty component is the working directory, which is a binary plant.
    if let Ok(path) = std::env::var("PATH") {
        for dir in path.split(':') {
            if dir.is_empty() || !Path::new(dir).is_absolute() { continue; }
            let p = Path::new(dir).join("clang++");
            if p.exists() && clang_has_nvptx(&p) { return Ok(p); }
        }
    }
    Err("no clang++ with NVPTX support found; `brew install llvm` or set MVCC_CLANG to an absolute path".into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symlink_outside_the_toolkit_is_refused() {
        let tmp = std::env::temp_dir().join(format!("mvcc-layout-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0)));
        let _ = std::fs::remove_dir_all(&tmp);
        std::fs::create_dir_all(tmp.join("trust/bin")).unwrap();
        std::fs::create_dir_all(tmp.join("outside")).unwrap();
        std::fs::write(tmp.join("outside/evil"), b"x").unwrap();
        std::os::unix::fs::symlink(tmp.join("outside/evil"), tmp.join("trust/bin/evil")).unwrap();
        let err = require_inside(&tmp.join("trust"), &tmp.join("trust/bin/evil"), "tool");
        assert!(err.is_err(), "{err:?}");
        std::fs::write(tmp.join("trust/bin/ok"), b"x").unwrap();
        assert!(require_inside(&tmp.join("trust"), &tmp.join("trust/bin/ok"), "tool").is_ok());
        let _ = std::fs::remove_dir_all(&tmp);
    }
}

fn clang_has_nvptx(p: &Path) -> bool {
    std::process::Command::new(p).arg("-print-targets").output().map(|o| String::from_utf8_lossy(&o.stdout).contains("nvptx64")).unwrap_or(false)
}
