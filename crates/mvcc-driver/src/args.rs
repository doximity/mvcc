//! nvcc command-line parsing. nvcc's grammar: `-s[=| ]value`, `--long[=| ]value`, booleans without values,
//! and `-I/-D/-U/-l/-L/-O` also in attached form. Unknown options are forwarded to the host compiler
//! (nvcc's `-forward-unknown-to-host-compiler` behavior, which CMake always requests).

use std::path::PathBuf;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Mode { Compile, Link, Lib, DeviceLink, Preprocess, Ptx, Version, Help }

#[derive(Default, Debug)]
pub struct Opts {
    pub mode_set: Option<Mode>,
    pub mode: Mode,
    pub inputs: Vec<PathBuf>,
    pub output: Option<PathBuf>,
    pub includes: Vec<PathBuf>,
    pub isystem: Vec<PathBuf>,
    pub force_includes: Vec<PathBuf>,
    pub defines: Vec<String>,
    pub undefines: Vec<String>,
    pub std: Vec<String>,
    pub opt: Option<String>,
    pub debug_host: bool,
    pub debug_device: bool,
    pub lineinfo: bool,
    pub archs: Vec<String>,
    pub host_extra: Vec<String>,     // -Xcompiler and forwarded unknowns
    pub device_extra: Vec<String>,   // extra clang flags for the device pass
    pub ir2msl_extra: Vec<String>,   // -Xmvcc=...
    pub linker_extra: Vec<String>,   // -Xlinker
    pub warn_flags: Vec<String>,     // -W... applied to both passes
    pub dep_flags: Vec<String>,      // -MD/-MT/-MF/...
    pub libs: Vec<String>,
    pub lib_dirs: Vec<PathBuf>,
    pub force_lang: Option<String>,
    pub relaxed_constexpr: bool,
    pub fast_math: bool,
    pub shared: bool,
    pub cudart: String,
    pub verbose: bool,
    pub dry_run: bool,
}

impl Default for Mode { fn default() -> Self { Mode::Link } }

/// Expand `--options-file f` / `-optf f` response files (also `@file`, which some generators emit).
pub fn expand_options_files(args: &[String]) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < args.len() {
        let a = &args[i];
        let file = if a == "--options-file" || a == "-optf" { i += 1; args.get(i).cloned() }
            else if let Some(f) = a.strip_prefix("--options-file=").or_else(|| a.strip_prefix("-optf=")) { Some(f.to_string()) }
            else if let Some(f) = a.strip_prefix('@') { if std::path::Path::new(f).exists() { Some(f.to_string()) } else { None } }
            else { None };
        match file {
            Some(f) => {
                let text = std::fs::read_to_string(&f).map_err(|e| format!("cannot read options file {}: {}", f, e))?;
                let nested = split_shell_words(&text);
                out.extend(expand_options_files(&nested)?);
            }
            None => out.push(a.clone()),
        }
        i += 1;
    }
    Ok(out)
}

fn split_shell_words(s: &str) -> Vec<String> {
    let mut words = Vec::new();
    let mut cur = String::new();
    let mut in_word = false;
    let mut quote: Option<char> = None;
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        match quote {
            Some(q) => {
                if c == q { quote = None; }
                else if c == '\\' && q == '"' { if let Some(n) = chars.next() { if n != '"' && n != '\\' { cur.push('\\'); } cur.push(n); } }
                else { cur.push(c); }
            }
            None => {
                if c.is_whitespace() { if in_word { words.push(std::mem::take(&mut cur)); in_word = false; } }
                else if c == '"' || c == '\'' { quote = Some(c); in_word = true; }
                else if c == '\\' { if let Some(n) = chars.next() { cur.push(n); } in_word = true; }
                else { cur.push(c); in_word = true; }
            }
        }
    }
    if in_word { words.push(cur); }
    words
}

/// Options that take a value (either `=v` or a following token). Value is returned as-is.
const VALUE_OPTS: &[&str] = &[
    "-o", "--output-file", "-x", "--x", "-arch", "--gpu-architecture", "-code", "--gpu-code", "-gencode", "--generate-code",
    "-ccbin", "--compiler-bindir", "-std", "--std", "-Xcompiler", "--compiler-options", "-Xptxas", "--ptxas-options",
    "-Xnvlink", "--nvlink-options", "-Xcudafe", "--cudafe-options", "-Xfatbin", "--fatbin-options", "-Xlinker", "--linker-options",
    "-Xarchive", "--archive-options", "-Xmvcc", "-maxrregcount", "--maxrregcount", "-keep-dir", "--keep-dir", "-t", "--threads",
    "-default-stream", "--default-stream", "-cudart", "--cudart", "-cudadevrt", "--cudadevrt",
    "-ftz", "--ftz", "-prec-div", "--prec-div", "-prec-sqrt", "--prec-sqrt", "-fmad", "--fmad", "-rdc", "--relocatable-device-code",
    "-diag-suppress", "--diag-suppress", "-diag-error", "--diag-error", "-diag-warn", "--diag-warn", "-isystem", "-include", "-idirafter", "-iquote",
    "-MT", "-MF", "-MQ", "-odir", "--output-directory", "-objtemp", "-libdevice-directory", "-ldir", "-target-dir", "-m", "--machine",
    "-time", "--time", "-split-compile", "--split-compile", "-dryrun-alias", "-Xopencc", "-lto", "--lto", "-dlto", "-brief",
    "-lineinfo-target", "-static-global-template-stub", "--static-global-template-stub", "-entries", "--entries",
];

/// Boolean options we recognize and ignore (nvcc-specific, no Metal analog).
const IGNORED_BOOL: &[&str] = &[
    "-keep", "--keep", "-save-temps", "--save-temps", "-G", "--device-debug", "-res-usage", "--resource-usage", "-src-in-ptx", "--source-in-ptx",
    "-Wno-deprecated-gpu-targets", "-Wno-deprecated-declarations-nvcc", "-forward-unknown-to-host-compiler", "--forward-unknown-to-host-compiler",
    "-forward-unknown-opts", "--forward-unknown-opts", "-forward-unknown-to-host-linker", "--forward-unknown-to-host-linker",
    "-extended-lambda", "--extended-lambda", "-expt-extended-lambda", "--expt-extended-lambda", "-m64", "--m64",
    "-dopt", "--dopt", "-noprof", "-restrict", "--restrict", "-use-local-env", "-dryrun", "-clean", "--clean-targets",
    "-w", "--disable-warnings", "-Wreorder", "-Wdefault-stream-launch", "-Wmissing-launch-bounds", "-Wext-lambda-captures-this",
    "-Wno-deprecated-declarations", "--no-host-device-initializer-list", "-no-host-device-initializer-list", "--no-host-device-move-forward",
    "-generate-line-info-target", "-augment-host-linker-script", "-extensible-whole-program", "-ewp", "-static-global-template-stub",
    "--display-error-number", "-err-no", "--no-display-error-number", "-no-err-no", "-fdevice-syntax-only", "--fdevice-syntax-only",
    "-minimal", "--minimal", "-qpp-config", "-device-int128", "--device-int128", "-device-float128", "--device-float128",
    "-frandom-seed", "-jump-table-density", "--jump-table-density", "-no-compress", "--no-compress", "-compress-all", "--compress-all",
    "-brief", "--brief", "-ptxas-options-nodebug", "-fvisibility-nvcc",
];

pub fn parse(args: &[String]) -> Result<Opts, String> {
    let mut o = Opts { cudart: "static".into(), ..Default::default() };
    let mut i = 0;
    let n = args.len();

    // helpers
    let take = |i: &mut usize, a: &str| -> Result<String, String> {
        if let Some(p) = a.find('=') { if a.starts_with("--") || a.len() > 2 { return Ok(a[p + 1..].to_string()); } }
        *i += 1;
        args.get(*i).cloned().ok_or_else(|| format!("option {} requires a value", a))
    };

    while i < n {
        let a = args[i].clone();
        let key = a.split('=').next().unwrap_or(&a).to_string();

        if !a.starts_with('-') || a == "-" {
            o.inputs.push(PathBuf::from(&a));
            i += 1; continue;
        }

        // ---- modes
        match key.as_str() {
            "--version" | "-V" => { o.mode_set = Some(Mode::Version); i += 1; continue; }
            "--help" | "-h" => { o.mode_set = Some(Mode::Help); i += 1; continue; }
            "-c" | "--compile" | "-dc" | "--device-c" | "-dw" | "--device-w" => { o.mode_set = Some(Mode::Compile); i += 1; continue; }
            "-dlink" | "--device-link" => { o.mode_set = Some(Mode::DeviceLink); i += 1; continue; }
            "-lib" | "--lib" => { o.mode_set = Some(Mode::Lib); i += 1; continue; }
            "-E" | "--preprocess" => { o.mode_set = Some(Mode::Preprocess); i += 1; continue; }
            "-ptx" | "--ptx" => { o.mode_set = Some(Mode::Ptx); i += 1; continue; }
            "-cubin" | "--cubin" | "-fatbin" | "--fatbin" | "-optix-ir" | "--optix-ir" | "-cuda" | "--cuda" =>
                return Err(format!("{} produces NVIDIA binary formats; mvcc compiles to Metal (use -c)", key)),
            "-link" | "--link" => { o.mode_set = Some(Mode::Link); i += 1; continue; }
            "-shared" | "--shared" => { o.shared = true; i += 1; continue; }
            "-v" | "--verbose" => { o.verbose = true; i += 1; continue; }
            "-dryrun" | "--dryrun" => { o.dry_run = true; o.verbose = true; i += 1; continue; }
            _ => {}
        }

        // ---- attached-form options
        if let Some(v) = attached(&a, "-I") { o.includes.push(PathBuf::from(v)); i += 1; continue; }
        if let Some(v) = attached(&a, "-D") { o.defines.push(v); i += 1; continue; }
        if let Some(v) = attached(&a, "-U") { o.undefines.push(v); i += 1; continue; }
        if a.starts_with("-O") && a.len() <= 3 && a != "-O" { o.opt.push_str_opt(&a[2..]); i += 1; continue; }
        if a == "-O" || a == "--optimize" { let v = take(&mut i, &a)?; o.opt = Some(v); i += 1; continue; }
        if let Some(v) = a.strip_prefix("--optimize=") { o.opt = Some(v.to_string()); i += 1; continue; }
        if let Some(v) = a.strip_prefix("-std=").or_else(|| a.strip_prefix("--std=")) { o.std = vec![v.to_string()]; i += 1; continue; }
        if let Some(v) = a.strip_prefix("--include-path=") { o.includes.push(PathBuf::from(v)); i += 1; continue; }
        if let Some(v) = a.strip_prefix("--define-macro=") { o.defines.push(v.to_string()); i += 1; continue; }
        if let Some(v) = a.strip_prefix("--library=") { o.libs.push(v.to_string()); i += 1; continue; }
        if let Some(v) = a.strip_prefix("--library-path=") { o.lib_dirs.push(PathBuf::from(v)); i += 1; continue; }
        if let Some(v) = a.strip_prefix("--pre-include=") { o.force_includes.push(PathBuf::from(v)); i += 1; continue; }
        if let Some(v) = a.strip_prefix("--output-file=") { o.output = Some(PathBuf::from(v)); i += 1; continue; }

        match key.as_str() {
            "-g" | "--debug" => { o.debug_host = true; i += 1; continue; }
            "-G" | "--device-debug" => { o.debug_device = true; i += 1; continue; }
            "-lineinfo" | "--generate-line-info" => { o.lineinfo = true; i += 1; continue; }
            "--expt-relaxed-constexpr" | "-expt-relaxed-constexpr" => { o.relaxed_constexpr = true; i += 1; continue; }
            "--use_fast_math" | "-use_fast_math" => { o.fast_math = true; i += 1; continue; }
            "-MD" | "-MMD" | "-M" | "-MM" | "-MP" | "-MG" | "-MV" | "--generate-dependencies" | "--generate-nonsystem-dependencies"
            | "--generate-dependencies-with-compile" | "--generate-nonsystem-dependencies-with-compile" => {
                let f = match key.as_str() { "--generate-dependencies-with-compile" | "--generate-dependencies" => "-MD", "--generate-nonsystem-dependencies-with-compile" | "--generate-nonsystem-dependencies" => "-MMD", k => k };
                o.dep_flags.push(f.to_string()); i += 1; continue;
            }
            "-MT" | "-MF" | "-MQ" => { let v = take(&mut i, &a)?; o.dep_flags.push(key.clone()); o.dep_flags.push(v); i += 1; continue; }
            "--dependency-output" | "-dependency-output" => { let v = take(&mut i, &a)?; o.dep_flags.push("-MF".into()); o.dep_flags.push(v); i += 1; continue; }
            "--dependency-target-name" => { let v = take(&mut i, &a)?; o.dep_flags.push("-MT".into()); o.dep_flags.push(v); i += 1; continue; }
            "-I" | "--include-path" => { let v = take(&mut i, &a)?; o.includes.push(PathBuf::from(v)); i += 1; continue; }
            "-isystem" => { let v = take(&mut i, &a)?; o.isystem.push(PathBuf::from(v)); i += 1; continue; }
            "-include" | "--pre-include" => { let v = take(&mut i, &a)?; o.force_includes.push(PathBuf::from(v)); i += 1; continue; }
            "-D" | "--define-macro" => { let v = take(&mut i, &a)?; o.defines.push(v); i += 1; continue; }
            "-U" | "--undefine-macro" => { let v = take(&mut i, &a)?; o.undefines.push(v); i += 1; continue; }
            "-l" | "--library" => { let v = take(&mut i, &a)?; o.libs.push(v); i += 1; continue; }
            "-L" | "--library-path" => { let v = take(&mut i, &a)?; o.lib_dirs.push(PathBuf::from(v)); i += 1; continue; }
            _ => {}
        }

        if key == "-Werror" || key == "--Werror" {
            // nvcc: `-Werror <kind>[,<kind>]`; clang: `-Werror` / `-Werror=<warning>`
            if a.contains('=') { o.warn_flags.push(a.clone()); i += 1; continue; }
            if let Some(next) = args.get(i + 1) { if !next.starts_with('-') && ["all-warnings", "cross-execution-space-call", "reorder", "default-stream-launch", "missing-launch-bounds", "ext-lambda-captures-this", "deprecated-declarations"].iter().any(|k| next.split(',').all(|x| *k == x)) { i += 2; continue; } }
            o.warn_flags.push("-Werror".into()); i += 1; continue;
        }
        if VALUE_OPTS.contains(&key.as_str()) {
            let v = take(&mut i, &a)?;
            match key.as_str() {
                "-o" | "--output-file" => o.output = Some(PathBuf::from(v)),
                "-x" | "--x" => o.force_lang = match v.as_str() { "cu" => Some("cu".into()), "c++" | "cpp" => Some("c++".into()), "c" => Some("c".into()), "none" => None, other => return Err(format!("unknown language '{}' for -x", other)) },
                "-arch" | "--gpu-architecture" | "-code" | "--gpu-code" | "-gencode" | "--generate-code" => o.archs.push(v),
                "-std" | "--std" => o.std = vec![v],
                "-Xcompiler" | "--compiler-options" => o.host_extra.extend(split_commas(&v)),
                "-Xlinker" | "--linker-options" => for x in split_commas(&v) { o.linker_extra.push("-Xlinker".into()); o.linker_extra.push(x); },
                "-Xmvcc" => o.ir2msl_extra.extend(split_commas(&v)),
                "-cudart" | "--cudart" => o.cudart = v,
                "-isystem" => o.isystem.push(PathBuf::from(v)),
                "-include" => o.force_includes.push(PathBuf::from(v)),
                "-m" | "--machine" => { if v != "64" { return Err("only 64-bit targets are supported".into()); } }
                "-default-stream" | "--default-stream" => { if v.starts_with("per-thread") { o.defines.push("CUDA_API_PER_THREAD_DEFAULT_STREAM=1".into()); } }
                // -Xptxas/-Xcudafe/-Xnvlink/-ccbin/-maxrregcount/-Werror kinds/-rdc/... have no Metal meaning
                _ => {}
            }
            i += 1; continue;
        }

        if IGNORED_BOOL.contains(&key.as_str()) { i += 1; continue; }

        if let Some(v) = attached(&a, "-l") { o.libs.push(v); i += 1; continue; }
        if let Some(v) = attached(&a, "-L") { o.lib_dirs.push(PathBuf::from(v)); i += 1; continue; }

        // clang-style flags that mean the same thing for us
        if a.starts_with("-W") { o.warn_flags.push(a.clone()); i += 1; continue; }
        if a == "-fPIC" || a == "-fPIE" || a.starts_with("-fvisibility") || a == "-pthread" || a.starts_with("-fno-") || a.starts_with("-f") || a.starts_with("-m") || a.starts_with("-p") || a.starts_with("-stdlib") || a.starts_with("-g") || a.starts_with("-isysroot") || a.starts_with("-mmacos") {
            if a == "-isysroot" { let v = take(&mut i, &a)?; o.host_extra.push(a.clone()); o.host_extra.push(v); i += 1; continue; }
            o.host_extra.push(a.clone()); i += 1; continue;
        }
        if a.starts_with("-Xclang") { let v = take(&mut i, &a)?; o.host_extra.push("-Xclang".into()); o.host_extra.push(v); i += 1; continue; }

        // last resort: forward to the host compiler, as nvcc -forward-unknown-to-host-compiler does
        eprintln!("[MVCC] warning: forwarding unknown option '{}' to the host compiler", a);
        o.host_extra.push(a.clone());
        i += 1;
    }

    o.mode = o.mode_set.unwrap_or(Mode::Link);
    if o.debug_device && o.opt.is_none() { o.opt = Some("0".into()); }
    Ok(o)
}

fn attached(a: &str, flag: &str) -> Option<String> {
    if a.len() > flag.len() && a.starts_with(flag) && !a.starts_with("--") {
        let rest = &a[flag.len()..];
        let rest = rest.strip_prefix('=').unwrap_or(rest);
        if !rest.is_empty() { return Some(rest.to_string()); }
    }
    None
}

/// nvcc -Xcompiler values are comma-separated lists of host flags (commas inside quotes are literal).
fn split_commas(v: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut depth = 0;
    let mut in_q = false;
    for c in v.chars() {
        match c {
            '"' => { in_q = !in_q; cur.push(c); }
            '(' | '[' | '{' if !in_q => { depth += 1; cur.push(c); }
            ')' | ']' | '}' if !in_q => { depth -= 1; cur.push(c); }
            ',' if !in_q && depth == 0 => { if !cur.is_empty() { out.push(std::mem::take(&mut cur)); } }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() { out.push(cur); }
    out.into_iter().map(|s| s.trim_matches('"').to_string()).collect()
}

trait PushOpt { fn push_str_opt(&mut self, s: &str); }
impl PushOpt for Option<String> { fn push_str_opt(&mut self, s: &str) { *self = Some(if s.is_empty() { "2".to_string() } else { s.to_string() }); } }
