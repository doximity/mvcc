//! Device code modules: the MVC1 blob embedded by mvcc, kernel ABI descriptors, lazy MSL compilation and the
//! pipeline-state cache (in memory + MTLBinaryArchive on disk).
use crate::metal::{Archive, Device, Library, Pipeline};
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Deserialize, Clone, Debug)]
pub struct ParamAbi {
    pub kind: String,
    pub size: u32,
    pub align: u32,
    pub offset: u32,
}
#[derive(Deserialize, Clone, Debug)]
pub struct KernelAbi {
    pub name: String,
    pub param_block_size: u32,
    pub static_smem: u32,
    pub dynamic_smem: bool,
    pub max_threads: u32,
    pub params: Vec<ParamAbi>,
    /// Tensor recovery (README.md): this kernel was rewritten onto TensorOps and `exact_variant` names
    /// its exact-semantics twin in the same module. The runtime launches the twin when recovery is disabled, when the
    /// device lacks Metal 4 TensorOps, or when the module's layout verification fails on this device/OS.
    #[serde(default)]
    pub exact_variant: Option<String>,
    #[serde(default)]
    pub is_variant: bool,
    /// The recovered body reserves per-simdgroup conversion scratch for at most this many threads; a launch with a
    /// larger block runs the exact twin instead.
    #[serde(default)]
    pub recovered_max_threads: u32,
    /// Recovered body was proved for exactly this 1-D block size; any other block shape runs the exact twin.
    #[serde(default)]
    pub staging_threads: u32,
    /// Exact byte footprint of the dynamic shared object; the launch allocates this instead of the host request. 0 = unknown.
    #[serde(default)]
    pub dyn_smem_used: u32,
    /// Recovered body runs `thread_scale` simdgroups per CUDA warp and expects
    /// block.x * thread_scale threads (each physical thread executes virtual thread p % block.x for op p / block.x).
    #[serde(default)]
    pub thread_scale: u32,
    /// Recovered body dropped the staging ring: bytes at `[dyn_smem_cut, ...)` moved down by `dyn_smem_shrink`.
    /// The launch allocates `smem - dyn_smem_shrink`.
    #[serde(default)]
    pub dyn_smem_cut: u32,
    #[serde(default)]
    pub dyn_smem_shrink: u32,
    /// Conditions over the kernel arguments the recovered body assumes: s-expressions evaluated on the
    /// parameter block at launch; the exact twin runs when one fails.
    #[serde(default)]
    pub assume: Vec<String>,
    /// The kernel reads module-scope globals: the module's globals buffer is bound at [[buffer(2)]] (constant view)
    /// and [[buffer(3)]] (device view).
    #[serde(default)]
    pub globals: bool,
    /// The compiler made guarded device loads speculatable by redirecting them to a page of zeros when the guard is
    /// false (guarded_loads.h); the kernel takes that page at [[buffer(1)]].
    #[serde(default)]
    pub zero_page: bool,
    #[serde(default)]
    pub lock_page: bool,
    /// Lowered device cudaGraphLaunch sites: byte offsets of exec handles in the param block (call order).
    #[serde(default)]
    pub device_graph_launch_exec_offsets: Vec<u32>,
}
/// A `__mvcc_tp_verify_*` kernel: one per matmul2d shape the module uses. Run once per module load; it exercises the
/// same prelude path recovered kernels use and writes the product in mma.sync fragment layout for a CPU check.
#[derive(Deserialize, Clone, Debug)]
pub struct VerifyKernelAbi {
    pub name: String,
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub tl: bool,
    pub tr: bool,
    #[serde(rename = "type")]
    pub ty: String,
}
#[derive(Deserialize, Debug, Clone)]
pub struct GlobalSymbolAbi { pub name: String, pub offset: u32, pub size: u32 }

/// Dynamic shared memory a host request of `requested` bytes costs the recovered body `k`: layout recovery's proven
/// footprint when it established one below the request; less the decode retiling's shrink when the retiling moved
/// the object's tail down (the host must have requested at least the cut - `retile_smem_fits`).
pub fn effective_dyn_smem(k: &KernelAbi, requested: u32) -> u32 {
    let mut smem = if k.dyn_smem_used != 0 && k.dyn_smem_used < requested { k.dyn_smem_used } else { requested };
    if k.dyn_smem_shrink != 0 && smem >= k.dyn_smem_cut { smem -= k.dyn_smem_shrink; }
    smem
}
/// Whether a retiled body's relocated shared layout applies to a host request of `requested` dynamic bytes: a request
/// below the cut would not have held the original object either (the exact twin runs, and reports the overrun).
pub fn retile_smem_fits(k: &KernelAbi, requested: u32) -> bool { k.dyn_smem_shrink == 0 || requested >= k.dyn_smem_cut }

/// Module-scope variables the runtime materializes in one device buffer.
#[derive(Deserialize, Debug, Clone, Default)]
pub struct GlobalsAbi {
    pub size: u32,
    #[serde(default)]
    pub align: u32,
    /// initial contents as hex
    pub init: String,
    /// (slot offset, target offset): slot <- buffer gpu address + target
    #[serde(default)]
    pub relocs: Vec<[u32; 2]>,
    #[serde(default)]
    pub symbols: Vec<GlobalSymbolAbi>,
}

/// A collective the compiler emitted (NCCL copy). Reductions are never silent.
#[derive(Deserialize, Debug, Clone)]
pub struct CollectiveAbi {
    pub kind: String,
    #[serde(default)]
    pub world: u32,
    #[serde(default)]
    pub dtype: String,
}

#[derive(Deserialize, Debug)]
pub struct ModuleAbi {
    pub kernels: Vec<KernelAbi>,
    #[serde(default)]
    pub verify_kernels: Vec<VerifyKernelAbi>,
    #[serde(default)]
    pub globals: Option<GlobalsAbi>,
    #[serde(default)]
    pub collectives: Vec<CollectiveAbi>,
    /// Virtual SM the TU was compiled for (`__CUDA_ARCH__`: 890 = sm_89, 900 = sm_90). Drives cudaDeviceGetAttribute CC.
    #[serde(default)]
    pub virtual_arch: u32,
}

/// `virtual_arch` / `__CUDA_ARCH__` (e.g. 890, 900) -> CUDA major/minor (8/9, 9/0).
pub fn compute_capability_from_virtual_arch(v: u32) -> (u32, u32) {
    if v == 0 { return (8, 9); }
    let sm = v / 10;
    (sm / 10, sm % 10)
}

/// Parse `sm_89`, `89`, `compute_90`, etc. into `__CUDA_ARCH__` form (890, 900).
pub fn parse_sm_arch_string(s: &str) -> u32 {
    let digits: String = s.chars().filter(|c| c.is_ascii_digit()).collect();
    digits.parse::<u32>().map(|v| v * 10).unwrap_or(890)
}

/// Which body a recovered kernel runs with.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum TensorPolicy { Recovered, Exact }

pub struct Module {
    pub msl: String,
    pub abi: ModuleAbi,
    pub hash: String,
    library: Option<Arc<Library>>,
    /// the module compiled with -D__MVCC_SPILL (shared memory in a device-memory pool; prelude "spill build"),
    /// built the first time a launch needs more shared memory than the device's threadgroup memory
    spill_library: Option<Arc<Library>>,
    pipelines: HashMap<String, Arc<KernelPipeline>>,
    archive: Option<Archive>,
    archive_path: Option<String>,
    archive_dirty: bool,
    tensor_policy: Option<TensorPolicy>,
    /// device address of the materialized globals buffer (allocated on first use)
    pub globals_addr: Option<u64>,
    /// this module's index in the runtime's table (set by register_module)
    pub index: usize,
}

pub struct KernelPipeline {
    pub pipeline: Pipeline,
    pub abi: KernelAbi,
    /// index of the owning module in the runtime's module table
    pub module: usize,
    pub max_threads: u32,
    pub static_tg_mem: u32,
    /// spill build: the block's shared memory is a slot of the runtime's pool at [[buffer(5)]], not threadgroup memory
    pub spill: bool,
}

/// Parse the MVC1 container: magic, version, total_len, msl_len, abi_len (all little-endian), then MSL then ABI JSON.
pub fn parse_blob(p: *const u8) -> Result<(String, ModuleAbi), String> {
    unsafe {
        let magic = std::slice::from_raw_parts(p, 4);
        if magic != b"MVC1" { return Err(format!("bad device-code magic {:?}", magic)); }
        let rd = |off: usize| -> u64 { u64::from_le_bytes(std::slice::from_raw_parts(p.add(off), 8).try_into().unwrap()) };
        let version = u32::from_le_bytes(std::slice::from_raw_parts(p.add(4), 4).try_into().unwrap());
        if version != 1 { return Err(format!("unsupported device-code version {}", version)); }
        let msl_len = rd(16) as usize;
        let abi_len = rd(24) as usize;
        let msl = std::str::from_utf8(std::slice::from_raw_parts(p.add(32), msl_len)).map_err(|e| e.to_string())?.to_string();
        let abi_bytes = std::slice::from_raw_parts(p.add(32 + msl_len), abi_len);
        let abi: ModuleAbi = serde_json::from_slice(abi_bytes).map_err(|e| format!("ABI json: {}", e))?;
        Ok((msl, abi))
    }
}

/// f32 -> IEEE half bits (round-to-nearest-even; enough for the small exact values used in verification).
fn f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let mant = b & 0x7f_ffff;
    if exp == 0 && mant == 0 { return sign; }
    let e = exp - 127 + 15;
    if e <= 0 { return sign; }        // underflow to zero (not reached for |x| >= 2^-14)
    if e >= 31 { return sign | 0x7c00; }
    let m = mant >> 13;
    let rem = mant & 0x1fff;
    let mut h = sign | ((e as u16) << 10) | (m as u16);
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) { h += 1; }
    h
}

/// IEEE half bits -> f32 (normals, zero and infinities; subnormals are not produced by the verification values).
fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h as u32) & 0x8000) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    if e == 0 { return f32::from_bits(sign) + if m == 0 { 0.0 } else { (m as f32) * (2.0f32).powi(-24) * if sign != 0 { -1.0 } else { 1.0 } }; }
    if e == 31 { return f32::from_bits(sign | 0x7f80_0000 | (m << 13)); }
    f32::from_bits(sign | ((e + 112) << 23) | (m << 13))
}

fn fnv1a(data: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in data { h ^= *b as u64; h = h.wrapping_mul(0x100000001b3); }
    h
}

fn cache_dir() -> Option<std::path::PathBuf> {
    let home = std::env::var("HOME").ok()?;
    let d = std::path::PathBuf::from(home).join("Library/Caches/mvcc");
    std::fs::create_dir_all(&d).ok()?;
    Some(d)
}

impl Module {
    pub fn new(msl: String, abi: ModuleAbi) -> Module {
        if !abi.collectives.is_empty() {
            let seq: Vec<String> = abi.collectives.iter()
                .map(|c| format!("{}:world={}:{}", c.kind, c.world, if c.dtype.is_empty() { "copy" } else { &c.dtype }))
                .collect();
            eprintln!("[MVCC] emitted collective sequence: {} (ncclAllGather copy; AllReduce/Reduce/ReduceScatter are gather-and-sum or avg)", seq.join(" "));
        }
        let hash = format!("{:016x}", fnv1a(msl.as_bytes()));
        Module { msl, abi, hash, library: None, spill_library: None, pipelines: HashMap::new(), archive: None, archive_dirty: false, archive_path: None, tensor_policy: None, globals_addr: None, index: 0 }
    }

    pub fn kernel_abi(&self, name: &str) -> Option<&KernelAbi> { self.abi.kernels.iter().find(|k| k.name == name) }

    /// Static shared memory a launch of `name` needs at minimum: the exact twin's when the kernel was recovered
    /// (the runtime falls back to it whenever the recovered body's scratch would not fit).
    pub fn static_smem_floor(&self, name: &str) -> u32 {
        let Some(k) = self.kernel_abi(name) else { return 0 };
        match &k.exact_variant { Some(t) => self.kernel_abi(t).map(|a| a.static_smem.min(k.static_smem)).unwrap_or(k.static_smem), None => k.static_smem }
    }

    /// Dynamic shared memory a host request of `requested` bytes actually costs: the proven footprint of the
    /// recovered body when layout recovery established one below the request (runtime.rs launches with it).
    pub fn effective_dyn_smem(&self, name: &str, requested: u32) -> u32 {
        match self.kernel_abi(name) { Some(k) => effective_dyn_smem(k, requested), None => requested }
    }

    fn ensure_library(&mut self, dev: &Device) -> Result<Arc<Library>, String> {
        if let Some(l) = &self.library { return Ok(l.clone()); }
        let t0 = std::time::Instant::now();
        let verbose = std::env::var("MVCC_VERBOSE").is_ok();
        // Offline-compiled path (README.md): $MVCC_METALLIB_DIR/<hash>.metallib, produced by
        // tools/precompile_metallib.sh from the `nvcc -keep` .metal. Any failure falls back to the source path.
        if let Ok(dir) = std::env::var("MVCC_METALLIB_DIR") {
            let path = std::path::Path::new(&dir).join(format!("{}.metallib", self.hash));
            match std::fs::read(&path).map_err(|e| e.to_string()).and_then(|bytes| dev.library_from_data(&bytes)) {
                Ok(lib) => {
                    if verbose { eprintln!("[MVCC] loaded module {} from {} in {:.1} ms", self.hash, path.display(), t0.elapsed().as_secs_f64() * 1e3); }
                    self.open_archive(dev);
                    let l = Arc::new(lib);
                    self.library = Some(l.clone());
                    return Ok(l);
                }
                Err(e) => if verbose { eprintln!("[MVCC] {}: {} (falling back to MSL source)", path.display(), e); },
            }
        }
        let lib = dev.library_from_source(&self.msl, true, false)?;
        if verbose { eprintln!("[MVCC] compiled module {} ({} kernels) in {:.1} ms", self.hash, self.abi.kernels.len(), t0.elapsed().as_secs_f64() * 1e3); }
        self.open_archive(dev);
        let l = Arc::new(lib);
        self.library = Some(l.clone());
        Ok(l)
    }

    /// Open (or create) the binary archive holding this module's compiled pipelines.
    fn open_archive(&mut self, dev: &Device) {
        if let Some(dir) = cache_dir() {
            let path = dir.join(format!("{}.metalar", self.hash));
            let ps = path.to_string_lossy().into_owned();
            let existing = path.exists();
            match dev.open_archive(if existing { Some(&ps) } else { None }) {
                Ok(a) => { self.archive = Some(a); self.archive_path = Some(ps); }
                Err(_) if existing => { let _ = std::fs::remove_file(&path); self.archive = dev.open_archive(None).ok(); self.archive_path = Some(ps); }
                Err(_) => {}
            }
        }
    }

    /// The spill build of the module (compiled on first use; the source is the same, so any MSL diagnostic would
    /// already have shown on the normal build).
    fn ensure_spill_library(&mut self, dev: &Device) -> Result<Arc<Library>, String> {
        if let Some(l) = &self.spill_library { return Ok(l.clone()); }
        let t0 = std::time::Instant::now();
        let lib = dev.library_from_source_defs(&self.msl, true, false, &["__MVCC_SPILL=1"]).map_err(|e| format!("spill build: {}", e))?;
        if std::env::var("MVCC_VERBOSE").is_ok() { eprintln!("[MVCC] compiled spill build of module {} ({} kernels) in {:.1} ms", self.hash, self.abi.kernels.len(), t0.elapsed().as_secs_f64() * 1e3); }
        let l = Arc::new(lib);
        self.spill_library = Some(l.clone());
        Ok(l)
    }

    pub fn pipeline(&mut self, dev: &Device, name: &str) -> Result<Arc<KernelPipeline>, String> { self.pipeline_in(dev, name, false) }

    /// `spill`: the pipeline from the spill build (launches whose shared memory exceeds threadgroup memory).
    pub fn pipeline_in(&mut self, dev: &Device, name: &str, spill: bool) -> Result<Arc<KernelPipeline>, String> {
        let key = if spill { format!("{}\u{1}spill", name) } else { name.to_string() };
        if let Some(p) = self.pipelines.get(&key) { return Ok(p.clone()); }
        let mut abi = self.kernel_abi(name).cloned().ok_or_else(|| format!("kernel {} not in module ABI", name))?;
        let lib = self.ensure_library(dev)?;
        let lib = if spill { self.ensure_spill_library(dev)? } else { lib };
        // Recovered kernels: decide once per module which body runs, then create the pipeline for that body under
        // the requested kernel's name. The twin has the identical parameter block, but its own ABI row: the
        // recovered body's shared-memory footprint (dyn_smem_used) and block-size proofs do not apply to it.
        let mut body = name.to_string();
        if let Some(twin) = abi.exact_variant.clone() {
            if self.tensor_policy(dev, &lib) == TensorPolicy::Exact {
                body = twin.clone();
                if let Some(t) = self.kernel_abi(&twin).cloned() { abi = KernelAbi { name: abi.name.clone(), ..t }; }
            }
        }
        let t0 = std::time::Instant::now();
        // The MSL attribute and this hint must match (Metal rejects a mismatch). A 64-thread
        // __launch_bounds__ copied to both leaves one GEMV group per core; omitting both
        // (hint 0) lets Metal plan occupancy. Do not raise the hint: that caps later launches
        // (prefix-sum 256 failed at a 128 floor).
        let hint = if abi.max_threads > 0 && abi.max_threads < 128 { 0 } else { abi.max_threads };
        let ps = dev.new_pipeline(&lib, &body, hint, if spill { None } else { self.archive.as_ref() })?;
        if std::env::var("MVCC_VERBOSE").is_ok() { eprintln!("[MVCC] pipeline {}{}{} in {:.1} ms (maxThreads={}, tgmem={})", name, if body != name { " (exact twin)" } else if abi.exact_variant.is_some() { " (TensorOps recovered)" } else { "" }, if spill { " [spill build]" } else { "" }, t0.elapsed().as_secs_f64() * 1e3, ps.max_threads(), ps.static_tg_mem()); }
        if !spill { self.archive_dirty = true; }
        let kp = Arc::new(KernelPipeline { max_threads: ps.max_threads(), static_tg_mem: ps.static_tg_mem(), pipeline: ps, abi, module: self.index, spill });
        self.pipelines.insert(key, kp.clone());
        Ok(kp)
    }

    /// Recovered-vs-exact policy for this module (cached). `MVCC_TENSOR_EXACT=1` and a device without Metal 4
    /// TensorOps both select the exact twins; otherwise the module's verification kernels must pass on this
    /// device before any recovered body is used. A failure is reported once and is not an error: the exact
    /// path is always available.
    fn tensor_policy(&mut self, dev: &Device, lib: &Arc<Library>) -> TensorPolicy {
        if let Some(p) = self.tensor_policy { return p; }
        let verbose = std::env::var("MVCC_VERBOSE").is_ok() || std::env::var("MVCC_TENSOR_DIAG").is_ok();
        let forced_exact = std::env::var("MVCC_TENSOR_EXACT").map(|v| v == "1").unwrap_or(false);
        let policy = if forced_exact {
            if verbose { eprintln!("[MVCC] tensor recovery: exact twins selected by environment"); }
            TensorPolicy::Exact
        } else if !dev.supports_metal4() {
            eprintln!("[MVCC] tensor recovery: device has no Metal 4 TensorOps; using exact kernels");
            TensorPolicy::Exact
        } else {
            match self.verify_tensor_ops(dev, lib) {
                Ok(n) => { if verbose { eprintln!("[MVCC] tensor recovery: {} matmul2d shape(s) verified on this device; recovered kernels enabled", n); } TensorPolicy::Recovered }
                Err(e) => { eprintln!("[MVCC] tensor recovery: verification failed ({}); using exact kernels", e); TensorPolicy::Exact }
            }
        };
        self.tensor_policy = Some(policy);
        policy
    }

    /// Run every `__mvcc_tp_verify_*` kernel with small integer operands and compare against a CPU product, bit-exact.
    fn verify_tensor_ops(&self, dev: &Device, lib: &Arc<Library>) -> Result<usize, String> {
        let verifies = self.abi.verify_kernels.clone();
        if verifies.is_empty() { return Ok(0); }
        let queue = dev.new_queue(None);
        for v in &verifies {
            let (m, n, k) = (v.m as usize, v.n as usize, v.k as usize);
            let cap = m * n / 32;
            // deterministic small integers, exactly representable in f16 and bf16 and summing exactly in f32
            let mut seed: u32 = 0x9e3779b9 ^ ((m * 131 + n * 17 + k) as u32);
            let mut next = || { seed ^= seed << 13; seed ^= seed >> 17; seed ^= seed << 5; ((seed >> 8) % 9) as i32 - 4 };
            let a: Vec<i32> = (0..m * k).map(|_| next()).collect();  // logical A[m][k]
            let b: Vec<i32> = (0..k * n).map(|_| next()).collect();  // logical B[k][n]
            let enc = |x: i32| -> u16 {
                if v.ty == "bf16" { ((x as f32).to_bits() >> 16) as u16 } else { f16_bits(x as f32) }
            };
            // memory layouts the verify kernel stages: A rows are m (k contiguous) unless tl -> rows k; B rows are n (k contiguous) when tr, else rows k
            let mut am = vec![0u16; m * k];
            for i in 0..m { for j in 0..k { let idx = if v.tl { j * m + i } else { i * k + j }; am[idx] = enc(a[i * k + j]); } }
            let mut bm = vec![0u16; k * n];
            for i in 0..k { for j in 0..n { let idx = if v.tr { j * k + i } else { i * n + j }; bm[idx] = enc(b[i * n + j]); } }
            let ab = dev.new_buffer((am.len() * 2) as u64).ok_or("buffer")?;
            let bb = dev.new_buffer((bm.len() * 2) as u64).ok_or("buffer")?;
            // fragments (32*cap f32), round-trip errors (32 f32), then the epilogue store: the tile as T in full
            // (M*N) and clipped to (M-3) x (N-5) at row pitch N+2 (M*(N+2), pre-filled with 0xffff)
            let tile_bytes = (m * n + m * (n + 2)) * 2;
            let ob = dev.new_buffer(((32 * cap + 32) * 4 + tile_bytes) as u64).ok_or("buffer")?;
            unsafe {
                std::ptr::copy_nonoverlapping(am.as_ptr() as *const u8, ab.contents(), am.len() * 2);
                std::ptr::copy_nonoverlapping(bm.as_ptr() as *const u8, bb.contents(), bm.len() * 2);
                std::ptr::write_bytes(ob.contents(), 0xff, (32 * cap + 32) * 4 + tile_bytes);
            }
            let ps = dev.new_pipeline(lib, &v.name, 32, None).map_err(|e| format!("{}: {}", v.name, e))?;
            let cb = queue.new_cmdbuf();
            let e = cb.compute();
            e.set_pipeline(&ps); e.set_buffer(&ab, 0, 0); e.set_buffer(&bb, 0, 1); e.set_buffer(&ob, 0, 2);
            e.dispatch([1, 1, 1], [32, 1, 1]);
            e.end();
            cb.commit();
            cb.wait_completed();
            if let Some(err) = cb.error() { return Err(format!("{}: {}", v.name, err)); }
            let out: Vec<f32> = unsafe { std::slice::from_raw_parts(ob.contents() as *const f32, 32 * cap + 32).to_vec() };
            // expected, in mma.sync fragment layout: lane l position p holds D[16mt + g + 8(r>>1)][8nt + 2t + (r&1)]
            let mut bad = 0usize; let mut first = String::new();
            for lane in 0..32usize { for p in 0..cap {
                let (tile, r) = (p / 4, p % 4); let (mt, nt) = (tile / (n / 8), tile % (n / 8));
                let mm = 16 * mt + (lane >> 2) + 8 * (r >> 1); let nn = 8 * nt + 2 * (lane & 3) + (r & 1);
                let mut acc = 0i64; for kk in 0..k { acc += (a[mm * k + kk] * b[kk * n + nn]) as i64; }
                let got = out[lane * cap + p];
                if got != acc as f32 { bad += 1; if first.is_empty() { first = format!("lane {} reg {} (D[{}][{}]) = {} expected {}", lane, p, mm, nn, got, acc); } }
            } }
            let rt: f32 = out[32 * cap..32 * cap + 32].iter().sum();
            if bad > 0 { return Err(format!("{}: {} of {} fragment values wrong; first: {}", v.name, bad, 32 * cap, first)); }
            if rt != 0.0 { return Err(format!("{}: slot<->fragment conversion round trip differs (sum |err| = {})", v.name, rt)); }
            // the epilogue store through the T destination tensor: the full tile, then the clipped one and its border
            let tile: Vec<u16> = unsafe { std::slice::from_raw_parts((ob.contents() as *const u8).add((32 * cap + 32) * 4) as *const u16, m * n + m * (n + 2)).to_vec() };
            let dec = |x: u16| -> f32 { if v.ty == "bf16" { f32::from_bits((x as u32) << 16) } else { f16_to_f32(x) } };
            let mut bad_full = 0usize; let mut bad_clip = 0usize; let mut border = 0usize;
            for mm in 0..m { for nn in 0..n {
                let mut acc = 0i64; for kk in 0..k { acc += (a[mm * k + kk] * b[kk * n + nn]) as i64; }
                if dec(tile[mm * n + nn]) != acc as f32 { bad_full += 1; }
                let c = tile[m * n + mm * (n + 2) + nn];
                if mm < m - 3 && nn < n - 5 { if dec(c) != acc as f32 { bad_clip += 1; } } else if c != 0xffff { border += 1; }
            } }
            if bad_full > 0 || bad_clip > 0 || border > 0 { return Err(format!("{}: epilogue tensor store: {} of {} values wrong in the full tile, {} in the clipped tile, {} written outside its extents", v.name, bad_full, m * n, bad_clip, border)); }
        }
        Ok(verifies.len())
    }

    /// Build every kernel's pipeline (cudaGraphUpload).
    pub fn prewarm(&mut self, dev: &Device) -> Result<(), String> {
        let names: Vec<String> = self.abi.kernels.iter().filter(|k| !k.is_variant).map(|k| k.name.clone()).collect();
        for n in names { self.pipeline(dev, &n)?; }
        Ok(())
    }

    pub fn flush_cache(&mut self) {
        if !self.archive_dirty { return; }
        if let (Some(a), Some(p)) = (&self.archive, &self.archive_path) {
            let tmp = format!("{}.tmp{}", p, std::process::id());
            if a.serialize(&tmp).is_ok() { let _ = std::fs::rename(&tmp, p); } else { let _ = std::fs::remove_file(&tmp); }
        }
        self.archive_dirty = false;
    }
}

/// Registered kernel: which module and which kernel name a host stub address maps to.
#[derive(Clone)]
pub struct KernelRef {
    pub module: usize,
    pub name: String,
    /// cudaFuncSetAttribute(MaxDynamicSharedMemorySize)
    pub max_dynamic_smem: u32,
    pub preferred_carveout: i32,
}
