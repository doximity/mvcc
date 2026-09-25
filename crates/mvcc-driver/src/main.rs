//! mvcc: an nvcc-compatible driver that compiles CUDA C++ to Metal.
//!
//! Accepts nvcc's command line (the subset CMake, Make and hand-written builds use), and for each CUDA
//! translation unit runs:
//!   1. clang -x cuda --cuda-device-only  -> NVPTX-flavoured LLVM IR (no PTX is ever produced)
//!   2. mvcc-ir2msl                        -> Metal Shading Language + kernel ABI (JSON)
//!   3. prelude splice + optional Metal compile check (mvcc-mslc)
//!   4. clang -x cuda --cuda-host-only -fcuda-include-gpubinary <blob> -> host object with the device code
//!      embedded in __NV_CUDA,__nv_fatbin and clang's standard __cudaRegister* constructors.
//! Host-only inputs (.cpp/.c/.o) go straight to clang. Linking is clang++ with -lcudart.

mod args;
mod layout;

use args::{Mode, Opts};
use layout::Layout;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const CUDA_RELEASE: &str = "12.8";
const CUDA_VERSION_FULL: &str = "12.8.61";
const DEFAULT_ARCH: &str = "sm_89";

fn main() -> ExitCode {
    let raw: Vec<String> = std::env::args().skip(1).collect();
    if raw.first().map(|s| s.as_str()) == Some("explain") {
        // recover Y[o] = ⊕_r F(...) for every kernel and print it
        std::env::set_var("MVCC_SIG", "1");
        std::env::set_var("MVCC_SIG_ONLY", "1");
        std::env::set_var("MVCC_SCHED", "1");
        std::env::set_var("MVCC_EXPLAIN", "1");
        let mut rest: Vec<String> = raw.iter().skip(1).cloned().collect();
        if !rest.iter().any(|a| a == "-c" || a == "--compile") { rest.insert(0, "-c".into()); }
        if !rest.iter().any(|a| a == "-o" || a.starts_with("--output-file")) {
            rest.push("-o".into());
            rest.push("/tmp/mvcc-explain.o".into());
        }
        let expanded = match args::expand_options_files(&rest) { Ok(a) => a, Err(e) => return die(&e) };
        let opts = match args::parse(&expanded) { Ok(o) => o, Err(e) => return die(&e) };
        let layout = match Layout::discover() { Ok(l) => l, Err(e) => return die(&e) };
        return match run(&opts, &layout) {
            Ok(()) => ExitCode::SUCCESS,
            Err(e) => { if !e.is_empty() { eprintln!("[MVCC] error: {}", e); } ExitCode::from(1) }
        };
    }
    if raw.first().map(|s| s.as_str()) == Some("diff") {
        // two schedule texts, costs and legality against a synthetic contraction
        if raw.len() < 3 { return die("usage: mvcc diff <sched_a> <sched_b>"); }
        let layout = match Layout::discover() { Ok(l) => l, Err(e) => return die(&e) };
        let st = std::process::Command::new(&layout.ir2msl)
            .args(["--sched-diff", &raw[1], &raw[2]])
            .status();
        return match st {
            Ok(s) if s.success() => ExitCode::SUCCESS,
            Ok(_) => ExitCode::from(1),
            Err(e) => die(&format!("cannot run mvcc-ir2msl --sched-diff: {}", e)),
        };
    }
    if raw.first().map(|s| s.as_str()) == Some("stir-in") {
        // a Program without CUDA recovery (named kind, or a .stir file with `kind gemm`)
        if raw.len() < 2 { return die("usage: mvcc stir-in elementwise|gemm|attention|<file.stir> [--sched-in <file.sched>]"); }
        let layout = match Layout::discover() { Ok(l) => l, Err(e) => return die(&e) };
        let mut args = vec!["--stir-in".to_string(), raw[1].clone()];
        if raw.len() >= 4 && raw[2] == "--sched-in" { args.push("--sched-in".into()); args.push(raw[3].clone()); }
        let st = std::process::Command::new(&layout.ir2msl).args(&args).status();
        return match st {
            Ok(s) if s.success() => ExitCode::SUCCESS,
            Ok(_) => ExitCode::from(1),
            Err(e) => die(&format!("cannot run mvcc-ir2msl --stir-in: {}", e)),
        };
    }
    let expanded = match args::expand_options_files(&raw) { Ok(a) => a, Err(e) => return die(&e) };
    let opts = match args::parse(&expanded) { Ok(o) => o, Err(e) => return die(&e) };

    if opts.mode == Mode::Version { print_version(); return ExitCode::SUCCESS; }
    if opts.mode == Mode::Help { print_help(); return ExitCode::SUCCESS; }

    let layout = match Layout::discover() { Ok(l) => l, Err(e) => return die(&e) };
    if opts.verbose { layout.dump(); }

    match run(&opts, &layout) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => { if !e.is_empty() { eprintln!("[MVCC] error: {}", e); } ExitCode::from(1) }
    }
}

fn die(msg: &str) -> ExitCode { eprintln!("[MVCC] error: {}", msg); ExitCode::from(1) }

fn print_version() {
    // CMake's CMakeDetermineCUDACompiler matches "nvcc: [^\n]+ Cuda compiler driver" and " V([0-9]+)\.([0-9]+)\.([0-9]+)".
    println!("nvcc: mvcc (CUDA-compatible toolchain for Apple silicon) Cuda compiler driver");
    println!("Copyright (c) 2026 Doximity, Inc. Apache License 2.0.");
    println!("Built on Apple silicon; device code targets Metal, not PTX.");
    println!("Cuda compilation tools, release {}, V{}", CUDA_RELEASE, CUDA_VERSION_FULL);
    println!("Build mvcc_{}_0", env!("CARGO_PKG_VERSION"));
}

fn print_help() {
    println!("usage: mvcc [nvcc options] <inputs>...\n\n\
Accepts nvcc's options. mvcc-specific environment variables:\n  \
MVCC_ARCH=sm_XX        virtual CUDA arch for __CUDA_ARCH__ (default: the highest -arch/-gencode asked for, up to sm_90; {DEFAULT_ARCH} when none)\n  \
MVCC_CLANG=<path>      clang++ with NVPTX support (default: Homebrew llvm)\n  \
MVCC_ROOT=<dir>        mvcc install/toolkit root\n  \
MVCC_VERIFY=0          skip the Metal compile check of generated MSL\n  \
MVCC_KEEP=1            keep intermediates next to the output (.ll, .metal, .abi.json)\n  \
MVCC_VERBOSE=1         print every sub-command\n  \
MVCC_TENSOR_DIAG=1     explain, per kernel, what tensor recovery did or why it declined\n  \
MVCC_TENSOR_EXACT=1    launch exact twins at runtime\n  \
MVCC_METALLIB_DIR      load precompiled <hash>.metallib files\n  \
MVCC_DEVICES=n         present n logical devices (1–16)\n  \
MVCC_GRAPH=<file>      compile-time kernel graph for fusion (default: infer sequential edges)\n\n\
Subcommands:\n  \
mvcc explain <file.cu> recover Y[o] = ⊕_r F(...) for every kernel\n  \
mvcc diff <a> <b>      compare two schedule texts\n  \
mvcc stir-in <kind>    Program without CUDA recovery (elementwise|gemm|attention [--sched-in])");
}

// ------------------------------------------------------------------------------------------------ orchestration

fn run(o: &Opts, l: &Layout) -> Result<(), String> {
    if o.mode == Mode::DeviceLink {
        // Separable compilation "device link" step: every TU already carries whole-program device code,
        // so the device-link object is an empty relocatable object.
        let out = o.output.clone().unwrap_or_else(|| PathBuf::from("a_dlink.o"));
        return emit_empty_object(l, &out, o);
    }
    if o.inputs.is_empty() { return Err("no input files".into()); }

    match o.mode {
        Mode::Compile | Mode::Preprocess | Mode::Ptx => {
            if o.inputs.len() > 1 && o.output.is_some() && o.mode != Mode::Preprocess {
                return Err("-o cannot be used with multiple input files in -c mode".into());
            }
            for inp in &o.inputs {
                let out = match &o.output { Some(p) => p.clone(), None => default_object_name(inp, o.mode) };
                compile_one(o, l, inp, &out)?;
            }
            Ok(())
        }
        Mode::Link | Mode::Lib => {
            let mut objects: Vec<PathBuf> = Vec::new();
            let mut temps: Vec<PathBuf> = Vec::new();
            let tmpdir = tempdir("mvcc-link")?;
            // MVCC_KEEP: build objects next to the final output so the kept intermediates (.metal, .ll) land there too
            let keep_dir = if std::env::var("MVCC_KEEP").is_ok() {
                o.output.as_ref().and_then(|p| p.parent().map(|d| d.to_path_buf())).filter(|d| !d.as_os_str().is_empty()).or_else(|| Some(PathBuf::from(".")))
            } else { None };
            for inp in &o.inputs {
                if is_source(inp, o) {
                    let out = keep_dir.as_ref().unwrap_or(&tmpdir).join(format!("{}.o", stem(inp)));
                    compile_one(o, l, inp, &out)?;
                    temps.push(out.clone());
                    objects.push(out);
                } else {
                    objects.push(inp.clone());
                }
            }
            let r = if o.mode == Mode::Lib { archive(o, &objects) } else { link(o, l, &objects) };
            if std::env::var("MVCC_KEEP").is_err() { let _ = std::fs::remove_dir_all(&tmpdir); }
            r
        }
        Mode::Version | Mode::Help | Mode::DeviceLink => unreachable!(),
    }
}

fn is_source(p: &Path, o: &Opts) -> bool {
    if o.force_lang.is_some() { return true; }
    matches!(ext(p).as_str(), "cu" | "cuh" | "cpp" | "cc" | "cxx" | "c" | "mm" | "m" | "ii" | "i")
}
fn is_cuda(p: &Path, o: &Opts) -> bool {
    match o.force_lang.as_deref() {
        Some("cu") => true,
        Some(_) => false,
        None => matches!(ext(p).as_str(), "cu" | "cuh"),
    }
}
fn ext(p: &Path) -> String { p.extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default() }
fn stem(p: &Path) -> String { p.file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "out".into()) }
fn default_object_name(inp: &Path, mode: Mode) -> PathBuf {
    match mode {
        Mode::Preprocess => PathBuf::from("-"),
        Mode::Ptx => PathBuf::from(format!("{}.metal", stem(inp))),
        _ => PathBuf::from(format!("{}.o", stem(inp))),
    }
}
fn tempdir(tag: &str) -> Result<PathBuf, String> {
    let base = std::env::var("TMPDIR").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/tmp"));
    let d = base.join(format!("{}-{}-{}", tag, std::process::id(), nanos()));
    std::fs::create_dir_all(&d).map_err(|e| format!("cannot create {}: {}", d.display(), e))?;
    Ok(d)
}
fn nanos() -> u128 { std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0) }

// ------------------------------------------------------------------------------------------------ one TU

fn compile_one(o: &Opts, l: &Layout, input: &Path, output: &Path) -> Result<(), String> {
    if !is_cuda(input, o) {
        // plain host source: clang with our includes so cuda_runtime.h resolves
        let mut c = clang(l, o);
        c.args(host_common_flags(o, l));
        if o.mode == Mode::Preprocess { c.arg("-E"); } else { c.arg("-c"); }
        if let Some(lang) = &o.force_lang { c.arg("-x").arg(clang_lang(lang)); }
        c.arg(input);
        if output != Path::new("-") { c.arg("-o").arg(output); }
        return exec(c, o);
    }

    if o.mode == Mode::Preprocess {
        let mut c = clang(l, o);
        c.args(cuda_common_flags(o, l)).arg("--cuda-host-only").arg("-E").arg(input);
        if output != Path::new("-") { c.arg("-o").arg(output); }
        return exec(c, o);
    }

    let keep = std::env::var("MVCC_KEEP").is_ok();
    let work = if keep { output.parent().map(|p| p.to_path_buf()).filter(|p| !p.as_os_str().is_empty()).unwrap_or_else(|| PathBuf::from(".")) } else { tempdir("mvcc-cc")? };
    let base = work.join(stem(output));
    let ll = with_ext(&base, "device.ll");
    let metal = with_ext(&base, "metal");
    let abi = with_ext(&base, "abi.json");
    let blob = with_ext(&base, "mvcc");

    let result = (|| -> Result<(), String> {
        // 1. device IR
        let dev_flags = |c: &mut Command| {
            c.args(cuda_common_flags(o, l));
            c.arg("--cuda-device-only").arg(format!("--cuda-gpu-arch={}", arch(o))).arg("--cuda-feature=+ptx86");
            c.arg(format!("-O{}", device_opt(o)));
            // `#pragma unroll` unrolls under nvcc whatever the loop looks like; clang's unroller has three rules nvcc
            // has not, and reports -Wpass-failed "loop not unrolled" for each loop they stop:
            //  * a loop with a bounded but inexact trip count (`for (b = 0; b < kSpan; ++b) if (b >= n) break;`,
            //    `for (c = tid; c < N; c += THREADS)`) unrolls to its bound only when the bound is <= 8;
            //  * a pragma'd loop is unrolled only up to 16K instructions;
            //  * a loop holding a `convergent` call gets no remainder loop and no bounded unroll, and clang marks every
            //    CUDA call convergent, cp.async inline asm and libdevice math included (mvcc-passes takes it off the
            //    per-thread ones: cpp/mvcc-llvm/nvptx_convergence.cpp).
            c.arg("-mllvm").arg("-unroll-max-upperbound=128");
            c.arg("-mllvm").arg("-pragma-unroll-threshold=65536");
            c.arg(format!("-fpass-plugin={}", l.passes.display()));
            for w in &o.device_extra { c.arg(w); }
        };
        let mut c = clang(l, o);
        dev_flags(&mut c);
        c.arg("-emit-llvm").arg("-S").arg("-o").arg(&ll).arg(input);
        let mut shared_rewrite: Option<Vec<SharedDecl>> = None;
        if let Err(e) = exec_capture(c, o) {
            let decls = shared_init_decls(&e);
            if decls.is_empty() { eprint!("{}", e); return Err(String::new()); }
            // nvcc accepts `__shared__ T x;` for a T with default member initializers (the initializers are not run:
            // shared memory is uninitialized). clang rejects it (err_shared_var_init) with no override, so the TU is
            // preprocessed and each diagnosed declaration is rewritten to uninitialized storage plus a reference
            // with the original name (tests/kernels/printf_shared.cu).
            let pre = with_ext(&base, "device.cu.i");
            let mut c = clang(l, o);
            dev_flags(&mut c);
            c.arg("-E").arg("-o").arg(&pre).arg(input);
            exec(c, o)?;
            rewrite_shared_decls(&pre, &decls)?;
            let mut c = clang(l, o);
            dev_flags(&mut c);
            preprocessed_input(&mut c);
            c.arg("-emit-llvm").arg("-S").arg("-o").arg(&ll).arg(&pre);
            exec(c, o)?;
            shared_rewrite = Some(decls);
        }

        // 2. IR -> MSL + ABI
        let mut m = Command::new(&l.ir2msl);
        m.arg(&ll).arg("-o").arg(&metal).arg("-abi").arg(&abi);
        if o.fast_math { m.arg("-fast-math"); }
        for x in &o.ir2msl_extra { m.arg(x); }
        exec(m, o)?;
        stamp_abi_virtual_arch(&abi, o)?;

        // 3. splice prelude, verify, pack blob
        let msl = splice_prelude(&metal, &l.prelude)?;
        std::fs::write(&metal, &msl).map_err(|e| e.to_string())?;
        if o.mode == Mode::Ptx { std::fs::copy(&metal, output).map_err(|e| e.to_string())?; return Ok(()); }
        if verify_enabled() && !msl_has_no_kernels(&abi)? {
            if let Some(mslc) = &l.mslc {
                let mut v = Command::new(mslc);
                v.arg(&metal).arg("--quiet");
                if let Err(e) = exec_capture(v, o) {
                    let kept = keep_failed_msl(&metal);
                    let detail = if e.trim().is_empty() {
                        " (re-run with MVCC_VERBOSE=1; see mslc output above)".to_string()
                    } else {
                        format!("\n{}", e.trim())
                    };
                    return Err(format!("generated Metal for {} failed to compile{}{}", input.display(), kept, detail));
                }
            }
        }
        write_blob(&blob, &msl, &abi)?;

        // 4. host object with embedded device code
        let mut c = clang(l, o);
        c.args(cuda_common_flags(o, l));
        c.arg("--cuda-host-only");
        c.arg("-Xclang").arg("-fcuda-include-gpubinary").arg("-Xclang").arg(&blob);
        c.arg(format!("-O{}", o.opt.as_deref().unwrap_or("0")));
        for h in &o.host_extra { c.arg(h); }
        if let Some(decls) = &shared_rewrite {
            // The host pass parses the same declarations: preprocess, rewrite, compile the .i. Dependencies are
            // generated from the original input in a separate pass so the build system still tracks the headers.
            if !o.dep_flags.is_empty() {
                let mut d = clang(l, o);
                d.args(cuda_common_flags(o, l)).arg("--cuda-host-only").arg("-E").arg("-o").arg("/dev/null");
                for f in &o.dep_flags { d.arg(f); }
                d.arg(input);
                exec(d, o)?;
            }
            let pre = with_ext(&base, "host.cu.i");
            let mut e = clang(l, o);
            e.args(cuda_common_flags(o, l)).arg("--cuda-host-only").arg("-E").arg("-o").arg(&pre).arg(input);
            exec(e, o)?;
            rewrite_shared_decls(&pre, decls)?;
            preprocessed_input(&mut c);
            c.arg("-c").arg("-o").arg(output).arg(&pre);
            return exec(c, o);
        }
        for d in &o.dep_flags { c.arg(d); }
        c.arg("-c").arg("-o").arg(output).arg(input);
        exec(c, o)
    })();

    if !keep { let _ = std::fs::remove_dir_all(&work); }
    result
}

fn with_ext(base: &Path, e: &str) -> PathBuf { PathBuf::from(format!("{}.{}", base.display(), e)) }

fn verify_enabled() -> bool { std::env::var("MVCC_VERIFY").map(|v| v != "0").unwrap_or(true) }

fn msl_has_no_kernels(abi: &Path) -> Result<bool, String> {
    let s = std::fs::read_to_string(abi).map_err(|e| e.to_string())?;
    Ok(!s.contains("\"name\""))
}

fn keep_failed_msl(metal: &Path) -> String {
    let dst = PathBuf::from("/tmp").join(format!("mvcc_failed_{}", metal.file_name().map(|f| f.to_string_lossy().into_owned()).unwrap_or_default()));
    match std::fs::copy(metal, &dst) { Ok(_) => format!(" (MSL kept at {})", dst.display()), Err(_) => String::new() }
}

/// Insert the prelude between the MVCC_PRELUDE_BEGIN/END markers emitted by mvcc-ir2msl, after the MVCC_NEED_* defines.
fn splice_prelude(metal: &Path, prelude: &Path) -> Result<String, String> {
    let src = std::fs::read_to_string(metal).map_err(|e| format!("read {}: {}", metal.display(), e))?;
    let pre = std::fs::read_to_string(prelude).map_err(|e| format!("read prelude {}: {}", prelude.display(), e))?;
    let marker = "#define MVCC_PRELUDE_END\n";
    match src.find(marker) {
        Some(i) => {
            let mut out = String::with_capacity(src.len() + pre.len() + 64);
            out.push_str(&src[..i]);
            out.push_str("// ---- mvcc prelude ----\n");
            out.push_str(&pre);
            out.push_str("\n// ---- end prelude ----\n");
            out.push_str(&src[i..]);
            Ok(out)
        }
        None => Ok(src),
    }
}

/// Record the TU's `-arch` / `-gencode` in the ABI JSON so libcudart reports matching compute capability.
fn stamp_abi_virtual_arch(abi_path: &Path, o: &Opts) -> Result<(), String> {
    let v = arch_macro(o);
    let mut j: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(abi_path).map_err(|e| e.to_string())?)
        .map_err(|e| format!("ABI json: {}", e))?;
    if let Some(obj) = j.as_object_mut() {
        obj.insert("virtual_arch".into(), serde_json::Value::from(v));
    }
    std::fs::write(abi_path, serde_json::to_string(&j).map_err(|e| e.to_string())?).map_err(|e| e.to_string())
}

/// MVC1 device-code blob: magic(4) version(4) reserved(8) msl_len(8) abi_len(8) msl abi
fn write_blob(path: &Path, msl: &str, abi: &Path) -> Result<(), String> {
    let abi_bytes = std::fs::read(abi).map_err(|e| e.to_string())?;
    let mut b = Vec::with_capacity(32 + msl.len() + abi_bytes.len());
    b.extend_from_slice(b"MVC1");
    b.extend_from_slice(&1u32.to_le_bytes());
    b.extend_from_slice(&0u64.to_le_bytes());
    b.extend_from_slice(&(msl.len() as u64).to_le_bytes());
    b.extend_from_slice(&(abi_bytes.len() as u64).to_le_bytes());
    b.extend_from_slice(msl.as_bytes());
    b.extend_from_slice(&abi_bytes);
    b.push(0);
    std::fs::write(path, b).map_err(|e| format!("write {}: {}", path.display(), e))
}

fn emit_empty_object(l: &Layout, out: &Path, o: &Opts) -> Result<(), String> {
    let dir = tempdir("mvcc-dlink")?;
    let src = dir.join("dlink.c");
    std::fs::write(&src, "/* mvcc: device linking is whole-program per TU; nothing to link */\nstatic int __mvcc_dlink_unused;\n").map_err(|e| e.to_string())?;
    let mut c = Command::new(&l.clang);
    c.arg("-x").arg("c").arg("-c").arg(&src).arg("-o").arg(out);
    for h in &o.host_extra { if h.starts_with("-fPIC") || h.starts_with("-fvisibility") || h.starts_with("-m") { c.arg(h); } }
    let r = exec(c, o);
    let _ = std::fs::remove_dir_all(&dir);
    r
}

// ------------------------------------------------------------------------------------------------ link / archive

fn link(o: &Opts, l: &Layout, objects: &[PathBuf]) -> Result<(), String> {
    let mut c = Command::new(&l.clang);
    if o.shared { c.arg("-shared"); }
    for obj in objects { c.arg(obj); }
    c.arg("-o").arg(o.output.clone().unwrap_or_else(|| PathBuf::from("a.out")));
    for d in &o.lib_dirs { c.arg(format!("-L{}", d.display())); }
    for lib in &o.libs { c.arg(format!("-l{}", lib)); }
    for x in &o.linker_extra { c.arg(x); }
    for h in &o.host_extra { if h.starts_with("-fsanitize") || h.starts_with("-stdlib") || h == "-pthread" || h.starts_with("-fopenmp") || h.starts_with("-fprofile") || h.starts_with("-fuse-ld") || h.starts_with("-framework") { c.arg(h); } }
    if o.cudart != "none" {
        c.arg(format!("-L{}", l.lib64.display())).arg("-lcudadevrt").arg("-lcudart");
        c.arg(format!("-Wl,-rpath,{}", l.lib64.display()));
    }
    if o.verbose || std::env::var("MVCC_VERBOSE").is_ok() {
        // CMake extracts the host link launcher (first word) and implicit link libraries/dirs from this
        // line; the LIBRARIES= text printed at startup must appear verbatim.
        let mut line = l.clang.display().to_string();
        for a in c.get_args() {
            let a = a.to_string_lossy();
            if a.starts_with("-L") && a.contains(&*l.lib64.display().to_string()) { continue; }
            if a == "-lcudadevrt" || a == "-lcudart" { continue; }
            line.push(' ');
            if a.contains(' ') { line.push('"'); line.push_str(&a); line.push('"'); } else { line.push_str(&a); }
        }
        if o.cudart != "none" { line.push(' '); line.push_str(&l.libraries_string()); }
        eprintln!("#$ {}", line);
        if o.dry_run { return Ok(()); }
        let st = c.status().map_err(|e| format!("cannot run {}: {}", c.get_program().to_string_lossy(), e))?;
        return if st.success() { Ok(()) } else { Err(String::new()) };
    }
    exec(c, o)
}

fn archive(o: &Opts, objects: &[PathBuf]) -> Result<(), String> {
    let out = o.output.clone().unwrap_or_else(|| PathBuf::from("a.a"));
    let _ = std::fs::remove_file(&out);
    let mut c = Command::new("ar");
    c.arg("rcs").arg(&out);
    for obj in objects { c.arg(obj); }
    exec(c, o)
}

// ------------------------------------------------------------------------------------------------ clang flag sets

fn clang(l: &Layout, _o: &Opts) -> Command { Command::new(&l.clang) }

/// The newest virtual arch whose clang header blocks mvcc lowers. sm_90 adds the cluster / TMA / `atom.add.v2.f32`
/// declarations of `__clang_cuda_intrinsics.h` (declared, emitted only when a kernel uses them); sm_100+ would pull in
/// blocks that have not been looked at.
const MAX_ARCH: u32 = 90;

/// The virtual arch of the device pass: `MVCC_ARCH` when set, else the highest `sm_NN` / `compute_NN` the command line
/// asks for (`-arch`, `-gencode arch=compute_90,code=[compute_90,sm_90]`, `-code`; CMake's `CMAKE_CUDA_ARCHITECTURES=90`
/// spells the second), clamped to [`MAX_ARCH`], else [`DEFAULT_ARCH`]. The user's -arch is about NVIDIA hardware, and
/// the only thing it decides here is `__CUDA_ARCH__` (which source paths compile) and the header blocks clang declares.
fn arch(o: &Opts) -> String {
    if let Ok(a) = std::env::var("MVCC_ARCH") { return a; }
    let mut best: Option<u32> = None;
    for a in &o.archs {
        for tok in a.split(|c: char| !c.is_ascii_alphanumeric() && c != '_') {
            let n = tok.strip_prefix("sm_").or_else(|| tok.strip_prefix("compute_")).or_else(|| tok.strip_prefix("lto_"));
            let Some(n) = n else { continue };
            let digits: String = n.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(v) = digits.parse::<u32>() { if best.is_none_or(|b| v > b) { best = Some(v); } }
        }
    }
    match best { Some(v) => format!("sm_{}", v.min(MAX_ARCH)), None => DEFAULT_ARCH.to_string() }
}

/// `__CUDA_ARCH__` as the device pass will see it (`sm_90` -> 900), for the host pass's fallback definition.
fn arch_macro(o: &Opts) -> u32 {
    let a = arch(o);
    let digits: String = a.trim_start_matches(|c: char| !c.is_ascii_digit()).chars().take_while(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().map(|v| v * 10).unwrap_or(890)
}

fn device_opt(o: &Opts) -> String {
    // Device IR is always optimized: the MSL emitter wants mem2reg'd, inlined IR, and Apple's compiler
    // optimizes again. -O0/-G builds still get -O1 so debugging by printf stays viable.
    match o.opt.as_deref() { Some("0") | None if o.debug_device => "1".into(), Some("0") | None => "3".into(), Some(x) => x.to_string() }
}

/// Flags shared by the device and host passes of a CUDA TU.
fn cuda_common_flags(o: &Opts, l: &Layout) -> Vec<String> {
    let mut v: Vec<String> = vec![
        "-x".into(), "cuda".into(),
        format!("--cuda-path={}", l.toolkit.display()),
        "-nocudainc".into(), "-nocudalib".into(),
        // clang keys the launch ABI (__cudaPushCallConfiguration) off -target-sdk-version, which on
        // Darwin the driver also uses for the macOS SDK; the last one wins, so ours goes through -Xclang.
        "-Xclang".into(), format!("-target-sdk-version={}", CUDA_RELEASE),
        "-isystem".into(), l.include.display().to_string(),
        // nvcc's predefined macros; clang only defines __CUDACC__ from the wrapper header we skip
        "-D__NVCC__".into(), "-D__CUDACC__".into(),
        "-D__CUDACC_VER_MAJOR__=12".into(), "-D__CUDACC_VER_MINOR__=8".into(), "-D__CUDACC_VER_BUILD__=61".into(),
        // the device pass's __CUDA_ARCH__, so cuda_runtime.h declares the same device functions in the host pass
        format!("-D__MVCC_CUDA_ARCH__={}", arch_macro(o)),
        "-Wno-unknown-cuda-version".into(),
        // nvcc pre-includes cuda_runtime.h into every CUDA TU, so __host__/__device__ and the runtime API are
        // visible before the first user include (clang's cuda_wrappers/<algorithm> and sources that spell
        // __device__ in a .c/.h included first depend on it).
        "-include".into(), "cuda_runtime.h".into(),
    ];
    v.extend(o.std.iter().map(|s| format!("-std={}", s)));
    if o.std.is_empty() { v.push("-std=c++17".into()); }
    if o.relaxed_constexpr { v.push("-Xclang".into()); v.push("-fcuda-allow-variadic-functions".into()); }
    for d in &o.defines { v.push(format!("-D{}", d)); }
    for u in &o.undefines { v.push(format!("-U{}", u)); }
    for i in &o.includes { v.push("-I".into()); v.push(i.display().to_string()); }
    for i in &o.isystem { v.push("-isystem".into()); v.push(i.display().to_string()); }
    for i in &o.force_includes { v.push("-include".into()); v.push(i.display().to_string()); }
    if o.debug_host { v.push("-g".into()); } else if o.lineinfo { v.push("-gline-tables-only".into()); }
    for w in &o.warn_flags { v.push(w.clone()); }
    if o.fast_math { v.push("-ffast-math".into()); }
    v
}

/// Flags for plain host sources (.cpp) compiled through mvcc.
fn host_common_flags(o: &Opts, l: &Layout) -> Vec<String> {
    let mut v: Vec<String> = vec!["-isystem".into(), l.include.display().to_string()];
    v.extend(o.std.iter().map(|s| format!("-std={}", s)));
    for d in &o.defines { v.push(format!("-D{}", d)); }
    for u in &o.undefines { v.push(format!("-U{}", u)); }
    for i in &o.includes { v.push("-I".into()); v.push(i.display().to_string()); }
    for i in &o.isystem { v.push("-isystem".into()); v.push(i.display().to_string()); }
    for i in &o.force_includes { v.push("-include".into()); v.push(i.display().to_string()); }
    v.push(format!("-O{}", o.opt.as_deref().unwrap_or("0")));
    if o.debug_host { v.push("-g".into()); }
    for w in &o.warn_flags { v.push(w.clone()); }
    for h in &o.host_extra { v.push(h.clone()); }
    for d in &o.dep_flags { v.push(d.clone()); }
    v
}

fn clang_lang(l: &str) -> &'static str {
    match l { "cu" => "cuda", "c" => "c", "c++" | "cpp" | "cxx" => "c++", _ => "c++" }
}

// ------------------------------------------------------------------------------------------------ process helpers

fn exec(mut c: Command, o: &Opts) -> Result<(), String> {
    if o.verbose || std::env::var("MVCC_VERBOSE").is_ok() {
        eprintln!("#$ {}", render(&c));
    }
    if o.dry_run { return Ok(()); }
    let st = c.status().map_err(|e| format!("cannot run {}: {}", c.get_program().to_string_lossy(), e))?;
    if st.success() { Ok(()) } else { Err(String::new()) }
}

/// Like `exec`, but returns the child's stderr on failure (printed by the caller unless it handles it).
fn exec_capture(mut c: Command, o: &Opts) -> Result<(), String> {
    if o.verbose || std::env::var("MVCC_VERBOSE").is_ok() { eprintln!("#$ {}", render(&c)); }
    if o.dry_run { return Ok(()); }
    let out = c.output().map_err(|e| format!("cannot run {}: {}", c.get_program().to_string_lossy(), e))?;
    let err = String::from_utf8_lossy(&out.stderr).into_owned();
    if out.status.success() { eprint!("{}", err); Ok(()) } else { Err(err) }
}

// ------------------------------------------------------------------------------------------------ __shared__ NSDMI rewrite

#[derive(Clone, Debug)]
struct SharedDecl { file: PathBuf, line: usize, name: String }

/// Parse clang's `file:line:col: error: initialization is not supported for __shared__ variables` diagnostics
/// and read the variable name at that position from the source file.
fn shared_init_decls(stderr: &str) -> Vec<SharedDecl> {
    let mut v = Vec::new();
    for l in stderr.lines() {
        let Some(idx) = l.find(": error: initialization is not supported for __shared__ variables") else { continue };
        let loc = &l[..idx];
        let mut parts = loc.rsplitn(3, ':');
        let (Some(col), Some(line), Some(file)) = (parts.next(), parts.next(), parts.next()) else { continue };
        let (Ok(col), Ok(line)) = (col.parse::<usize>(), line.parse::<usize>()) else { continue };
        let Ok(src) = std::fs::read_to_string(file) else { continue };
        let Some(text) = src.lines().nth(line.saturating_sub(1)) else { continue };
        let bytes = text.as_bytes();
        let start = col.saturating_sub(1).min(bytes.len());
        let mut end = start;
        while end < bytes.len() && (bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_') { end += 1; }
        if end == start { continue; }
        v.push(SharedDecl { file: PathBuf::from(file), line, name: text[start..end].to_string() });
    }
    v
}

/// Rewrite `__attribute__((shared)) T name[dims];` (one per diagnosed presumed location) in a preprocessed file to
/// `__attribute__((shared)) ::mvcc::shared_storage<T[dims]> __mvcc_ss_name; auto& name = __mvcc_ss_name.get();`.
fn rewrite_shared_decls(pre: &Path, decls: &[SharedDecl]) -> Result<(), String> {
    let src = std::fs::read_to_string(pre).map_err(|e| format!("{}: {}", pre.display(), e))?;
    let mut out = String::with_capacity(src.len() + 1024);
    let (mut cur_file, mut cur_line) = (String::new(), 0usize);
    let mut done = vec![false; decls.len()];
    for l in src.lines() {
        if let Some(rest) = l.strip_prefix("# ") {
            let mut it = rest.splitn(2, ' ');
            if let (Some(n), Some(f)) = (it.next(), it.next()) {
                if let Ok(n) = n.parse::<usize>() {
                    cur_line = n;
                    if let Some(q) = f.strip_prefix('"') { cur_file = q.split('"').next().unwrap_or("").to_string(); }
                    out.push_str(l); out.push('\n');
                    continue;
                }
            }
        }
        let mut line = l.to_string();
        for (i, d) in decls.iter().enumerate() {
            if done[i] || d.line != cur_line || !same_file(&d.file, &cur_file) { continue; }
            if let Some(r) = rewrite_shared_line(&line, &d.name) { line = r; done[i] = true; }
        }
        out.push_str(&line); out.push('\n');
        cur_line += 1;
    }
    // Host and device preprocesses can disagree on the #line for the same declaration (launch_bounds
    // and other attributes eat lines on one side). Retry unmatched names anywhere in that file.
    if done.iter().any(|ok| !ok) {
        out = rewrite_shared_decls_by_name(&out, decls, &mut done);
    }
    if let Some(i) = done.iter().position(|d| !d) {
        return Err(format!("could not rewrite the __shared__ declaration of '{}' at {}:{} (declaration form not understood: declare one variable per line, without an initializer, e.g. '__shared__ T name[N];')", decls[i].name, decls[i].file.display(), decls[i].line));
    }
    std::fs::write(pre, out).map_err(|e| format!("{}: {}", pre.display(), e))
}

fn rewrite_shared_decls_by_name(src: &str, decls: &[SharedDecl], done: &mut [bool]) -> String {
    let mut cur_file = String::new();
    let mut out = String::with_capacity(src.len() + 64);
    for l in src.lines() {
        if let Some(rest) = l.strip_prefix("# ") {
            let mut it = rest.splitn(2, ' ');
            if let (Some(n), Some(f)) = (it.next(), it.next()) {
                if n.parse::<usize>().is_ok() {
                    if let Some(q) = f.strip_prefix('"') { cur_file = q.split('"').next().unwrap_or("").to_string(); }
                    out.push_str(l); out.push('\n');
                    continue;
                }
            }
        }
        let mut line = l.to_string();
        for (i, d) in decls.iter().enumerate() {
            if done[i] || !same_file(&d.file, &cur_file) { continue; }
            if let Some(r) = rewrite_shared_line(&line, &d.name) { line = r; done[i] = true; }
        }
        out.push_str(&line); out.push('\n');
    }
    out
}

/// Compile a preprocessed `.i` as CUDA again: the pre-included cuda_runtime.h is already expanded in the text (a
/// second inclusion would redefine it), and the re-run of the preprocessor is harmless (no directives remain).
fn preprocessed_input(c: &mut Command) {
    let args: Vec<std::ffi::OsString> = c.get_args().map(|a| a.to_os_string()).collect();
    let prog = c.get_program().to_os_string();
    let mut n = Command::new(prog);
    let mut skip = false;
    for (i, a) in args.iter().enumerate() {
        if skip { skip = false; continue; }
        if a == "-include" && args.get(i + 1).map(|x| x == "cuda_runtime.h").unwrap_or(false) { skip = true; continue; }
        n.arg(a);
    }
    n.arg("-x").arg("cuda").arg("-Wno-unused-command-line-argument");
    *c = n;
}

fn same_file(a: &Path, b: &str) -> bool {
    let b = Path::new(b);
    if a == b { return true; }
    match (std::fs::canonicalize(a), std::fs::canonicalize(b)) { (Ok(x), Ok(y)) => x == y, _ => false }
}

fn rewrite_shared_line(line: &str, name: &str) -> Option<String> {
    const ATTR: &str = "__attribute__((shared))";
    let mut search = 0;
    while let Some(off) = line[search..].find(ATTR) {
        let start = search + off;
        let after = start + ATTR.len();
        let Some(semi_off) = line[after..].find(';') else { return None };
        let decl = &line[after..after + semi_off];
        // whole-token occurrence of the name
        let mut found = None;
        let mut from = 0;
        while let Some(p) = decl[from..].find(name) {
            let a = from + p; let b = a + name.len();
            let before_ok = a == 0 || !(decl.as_bytes()[a - 1].is_ascii_alphanumeric() || decl.as_bytes()[a - 1] == b'_');
            let after_ok = b >= decl.len() || !(decl.as_bytes()[b].is_ascii_alphanumeric() || decl.as_bytes()[b] == b'_');
            if before_ok && after_ok { found = Some((a, b)); break; }
            from = b;
        }
        let Some((a, b)) = found else { search = after; continue };
        let ty_text = decl[..a].trim();
        let dims = decl[b..].trim();
        if ty_text.is_empty() || decl.contains(',') && !ty_text.contains('<') || dims.contains('=') { return None; }
        if !dims.is_empty() && !(dims.starts_with('[') && dims.ends_with(']')) { return None; }
        // move alignment attributes from the type to the storage variable (they are not part of a type-id)
        let mut attrs = String::new();
        let mut ty_clean = String::new();
        let mut rest = ty_text;
        while let Some(p) = rest.find("__attribute__((") {
            ty_clean.push_str(&rest[..p]);
            let mut depth = 0; let mut end = p;
            for (i, ch) in rest[p..].char_indices() { match ch { '(' => depth += 1, ')' => { depth -= 1; if depth == 0 { end = p + i + 1; break; } } _ => {} } }
            if end == p { return None; }
            attrs.push_str(&rest[p..end]); attrs.push(' ');
            rest = &rest[end..];
        }
        ty_clean.push_str(rest);
        let ty_clean = ty_clean.split_whitespace().collect::<Vec<_>>().join(" ");
        let ty_clean = ty_clean.strip_prefix("static ").map(|t| t.to_string()).unwrap_or(ty_clean);
        let storage = format!("__mvcc_ss_{}", name);
        let repl = format!("{} {}::mvcc::shared_storage<{}{}> {}; auto& {} = {}.get()", ATTR, attrs, ty_clean, dims, storage, name, storage);
        return Some(format!("{}{}{}", &line[..start], repl, &line[after + semi_off..]));
    }
    None
}

fn render(c: &Command) -> String {
    let mut s = c.get_program().to_string_lossy().into_owned();
    for a in c.get_args() {
        let a = a.to_string_lossy();
        s.push(' ');
        if a.contains(' ') || a.contains('"') { s.push('"'); s.push_str(&a.replace('"', "\\\"")); s.push('"'); } else { s.push_str(&a); }
    }
    s
}
