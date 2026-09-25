//! NCCL over the logical devices (`include/nccl.h`; `lib64/libnccl.dylib` is this library under another name).
//!
//! A multi-GPU program drives its ranks from one process: a communicator per rank (`ncclCommInitAll`, or
//! `ncclGetUniqueId` + `ncclCommInitRank` per rank), one stream per rank, and the collectives posted for every
//! rank between `ncclGroupStart` / `ncclGroupEnd`. The logical devices share one address space, so a collective
//! is a set of stream-ordered copies: each rank's stream waits for an event recorded on every other rank's stream
//! (the operands are ready) and copies the slabs it needs. Ranks driven from several threads work too: a post
//! outside a group waits (30 s) for the other ranks' matching posts.
//!
//! Implemented: AllGather, Broadcast / Bcast, Send / Recv, and AllReduce / Reduce / ReduceScatter as
//! gather-and-sum or gather-and-avg (`ncclSum` / `ncclAvg` on `ncclFloat32` / `ncclInt32` / `ncclUint32`).
//! Prod / max / min still return `ncclInvalidUsage`. `world == 1` collectives are copies. The reduce is
//! host-side after stream-ordered D2H of every rank's send (no hand-written Metal). Reduce writes to
//! `root` only; ReduceScatter treats each send as `recvcount * world` elements.
#![allow(non_snake_case, non_upper_case_globals, clippy::missing_safety_doc)]
use crate::api::{cudaEventCreateWithFlags, cudaEventDestroy, cudaEventRecord, cudaFree, cudaGetDevice, cudaMalloc, cudaMemcpyAsync, cudaStreamSynchronize, cudaStreamWaitEvent};
use crate::error::SUCCESS;
use parking_lot::Mutex;
use std::ffi::{c_char, c_int, c_void};
use std::sync::OnceLock;

pub type NcclResult = c_int;
pub const ncclSuccess: NcclResult = 0;
pub const ncclUnhandledCudaError: NcclResult = 1;
pub const ncclSystemError: NcclResult = 2;
pub const ncclInternalError: NcclResult = 3;
pub const ncclInvalidArgument: NcclResult = 4;
pub const ncclInvalidUsage: NcclResult = 5;
pub const ncclRemoteError: NcclResult = 6;
pub const ncclInProgress: NcclResult = 7;

const NCCL_VERSION: c_int = 22803;   // 2.28.3: the header's NCCL_VERSION_CODE
const UNIQUE_ID_BYTES: usize = 128;

/// One rank's communicator. Communicators created together (one `ncclCommInitAll`, or one unique id) form a
/// clique; a collective completes when every rank of the clique has posted its part.
#[repr(C)]
pub struct Comm { clique: u64, rank: c_int, world: c_int, device: c_int }

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Kind {
    AllGather,
    Broadcast { root: c_int },
    Send { peer: c_int },
    Recv { peer: c_int },
    AllReduce { op: c_int, ty: c_int },
    Reduce { op: c_int, ty: c_int, root: c_int },
    ReduceScatter { op: c_int, ty: c_int },
}

#[derive(Clone, Copy)]
struct Post { kind: Kind, rank: c_int, send: usize, recv: usize, bytes: usize, send_bytes: usize, stream: *mut c_void }
unsafe impl Send for Post {}

struct Clique { world: c_int, joined: c_int, pending: Vec<Post> }

struct State { next_clique: u64, cliques: std::collections::HashMap<u64, Clique>, last_error: Option<String>, noted: std::collections::HashSet<&'static str> }

static STATE: OnceLock<Mutex<State>> = OnceLock::new();
fn state() -> &'static Mutex<State> {
    STATE.get_or_init(|| Mutex::new(State { next_clique: 1, cliques: Default::default(), last_error: None, noted: Default::default() }))
}
thread_local! {
    static GROUP_DEPTH: std::cell::Cell<u32> = const { std::cell::Cell::new(0) };
    static GROUP_POSTS: std::cell::RefCell<Vec<(u64, Post)>> = const { std::cell::RefCell::new(Vec::new()) };
}

fn note(key: &'static str, msg: String) -> NcclResult {
    let mut s = state().lock();
    if s.noted.insert(key) { eprintln!("[mvcc nccl] {}", msg); }
    s.last_error = Some(msg);
    ncclInvalidUsage
}
fn fail(code: NcclResult, msg: String) -> NcclResult { state().lock().last_error = Some(msg); code }

fn type_bytes(t: c_int) -> Option<usize> {
    Some(match t { 0 | 1 => 1, 2 | 3 => 4, 4 | 5 => 8, 6 => 2, 7 => 4, 8 => 8, 9 => 2, 10 | 11 => 1, _ => return None })
}

fn comm<'a>(c: *mut c_void) -> Option<&'a Comm> { if c.is_null() { None } else { Some(unsafe { &*(c as *const Comm) }) } }

// ------------------------------------------------------------------ communicators

fn new_clique(world: c_int) -> u64 {
    let mut s = state().lock();
    let id = s.next_clique; s.next_clique += 1;
    s.cliques.insert(id, Clique { world, joined: 0, pending: Vec::new() });
    id
}
fn make_comm(clique: u64, rank: c_int, world: c_int, device: c_int) -> *mut c_void {
    state().lock().cliques.get_mut(&clique).map(|c| c.joined += 1);
    Box::into_raw(Box::new(Comm { clique, rank, world, device })) as *mut c_void
}

#[no_mangle] pub extern "C" fn ncclGetVersion(v: *mut c_int) -> NcclResult { if v.is_null() { return ncclInvalidArgument; } unsafe { *v = NCCL_VERSION }; ncclSuccess }
#[no_mangle] pub extern "C" fn ncclGetUniqueId(id: *mut u8) -> NcclResult {
    if id.is_null() { return ncclInvalidArgument; }
    // the clique is created when the first rank initializes with the id (its world is known then)
    let n = { let mut s = state().lock(); let n = s.next_clique; s.next_clique += 1; n };
    unsafe { std::ptr::write_bytes(id, 0, UNIQUE_ID_BYTES); std::ptr::copy_nonoverlapping(b"mvcc-nccl".as_ptr(), id, 9); std::ptr::copy_nonoverlapping(n.to_le_bytes().as_ptr(), id.add(16), 8); }
    ncclSuccess
}
/// `ncclUniqueId`, passed by value (128 bytes: indirect in the arm64 C ABI, which repr(C) follows).
#[repr(C)] #[derive(Clone, Copy)] pub struct UniqueId { internal: [u8; UNIQUE_ID_BYTES] }
#[no_mangle] pub extern "C" fn ncclCommInitRank(comm: *mut *mut c_void, nranks: c_int, id: UniqueId, rank: c_int) -> NcclResult {
    if comm.is_null() || nranks < 1 || rank < 0 || rank >= nranks { return ncclInvalidArgument; }
    if &id.internal[..9] != b"mvcc-nccl" { return fail(ncclInvalidArgument, "ncclCommInitRank: the unique id was not made by ncclGetUniqueId".into()); }
    let clique = u64::from_le_bytes(id.internal[16..24].try_into().unwrap());
    { let mut s = state().lock(); s.cliques.entry(clique).or_insert(Clique { world: nranks, joined: 0, pending: Vec::new() }); }
    let mut dev = 0; cudaGetDevice(&mut dev);
    unsafe { *comm = make_comm(clique, rank, nranks, dev) };
    ncclSuccess
}
#[no_mangle] pub extern "C" fn ncclCommInitRankConfig(comm: *mut *mut c_void, nranks: c_int, id: UniqueId, rank: c_int, _config: *const c_void) -> NcclResult { ncclCommInitRank(comm, nranks, id, rank) }
#[no_mangle] pub extern "C" fn ncclCommInitAll(comms: *mut *mut c_void, ndev: c_int, devlist: *const c_int) -> NcclResult {
    if comms.is_null() || ndev < 1 { return ncclInvalidArgument; }
    let mut count = 0; if crate::api::cudaGetDeviceCount(&mut count) != SUCCESS { return ncclUnhandledCudaError; }
    let clique = new_clique(ndev);
    for r in 0..ndev {
        let dev = if devlist.is_null() { r } else { unsafe { *devlist.add(r as usize) } };
        if dev < 0 || dev >= count { return fail(ncclInvalidArgument, format!("ncclCommInitAll: device {} of {} (MVCC_DEVICES presents {})", dev, ndev, count)); }
        unsafe { *comms.add(r as usize) = make_comm(clique, r, ndev, dev) };
    }
    ncclSuccess
}
#[no_mangle] pub extern "C" fn ncclCommDestroy(c: *mut c_void) -> NcclResult {
    if c.is_null() { return ncclInvalidArgument; }
    let b = unsafe { Box::from_raw(c as *mut Comm) };
    let mut s = state().lock();
    if let Some(cl) = s.cliques.get_mut(&b.clique) { cl.joined -= 1; if cl.joined <= 0 { s.cliques.remove(&b.clique); } }
    ncclSuccess
}
#[no_mangle] pub extern "C" fn ncclCommAbort(c: *mut c_void) -> NcclResult { ncclCommDestroy(c) }
#[no_mangle] pub extern "C" fn ncclCommFinalize(_c: *mut c_void) -> NcclResult { ncclSuccess }
#[no_mangle] pub extern "C" fn ncclCommCount(c: *mut c_void, n: *mut c_int) -> NcclResult { match (comm(c), n.is_null()) { (Some(cm), false) => { unsafe { *n = cm.world }; ncclSuccess } _ => ncclInvalidArgument } }
#[no_mangle] pub extern "C" fn ncclCommCuDevice(c: *mut c_void, d: *mut c_int) -> NcclResult { match (comm(c), d.is_null()) { (Some(cm), false) => { unsafe { *d = cm.device }; ncclSuccess } _ => ncclInvalidArgument } }
#[no_mangle] pub extern "C" fn ncclCommUserRank(c: *mut c_void, r: *mut c_int) -> NcclResult { match (comm(c), r.is_null()) { (Some(cm), false) => { unsafe { *r = cm.rank }; ncclSuccess } _ => ncclInvalidArgument } }
#[no_mangle] pub extern "C" fn ncclCommGetAsyncError(c: *mut c_void, e: *mut NcclResult) -> NcclResult { if c.is_null() || e.is_null() { return ncclInvalidArgument; } unsafe { *e = ncclSuccess }; ncclSuccess }
#[no_mangle] pub extern "C" fn ncclCommRegister(_c: *mut c_void, _buf: *mut c_void, _n: usize, handle: *mut *mut c_void) -> NcclResult { if !handle.is_null() { unsafe { *handle = 1 as *mut c_void } } ncclSuccess }
#[no_mangle] pub extern "C" fn ncclCommDeregister(_c: *mut c_void, _handle: *mut c_void) -> NcclResult { ncclSuccess }
#[no_mangle] pub extern "C" fn ncclMemAlloc(p: *mut *mut c_void, n: usize) -> NcclResult { if cudaMalloc(p, n) == SUCCESS { ncclSuccess } else { ncclUnhandledCudaError } }
#[no_mangle] pub extern "C" fn ncclMemFree(p: *mut c_void) -> NcclResult { if cudaFree(p) == SUCCESS { ncclSuccess } else { ncclUnhandledCudaError } }
#[no_mangle] pub extern "C" fn ncclGetErrorString(r: NcclResult) -> *const c_char {
    match r {
        ncclSuccess => c"no error".as_ptr(),
        ncclUnhandledCudaError => c"unhandled cuda error (run with NCCL_DEBUG=INFO for details)".as_ptr(),
        ncclSystemError => c"unhandled system error".as_ptr(),
        ncclInternalError => c"internal error".as_ptr(),
        ncclInvalidArgument => c"invalid argument".as_ptr(),
        ncclInvalidUsage => c"invalid usage (mvcc: see the [mvcc nccl] note)".as_ptr(),
        ncclRemoteError => c"remote process exited or there was a network error".as_ptr(),
        ncclInProgress => c"NCCL operation in progress".as_ptr(),
        _ => c"unknown result code".as_ptr(),
    }
}
#[no_mangle] pub extern "C" fn ncclGetLastError(_c: *mut c_void) -> *const c_char {
    static LAST: Mutex<Option<std::ffi::CString>> = Mutex::new(None);
    let msg = state().lock().last_error.clone().unwrap_or_default();
    let mut l = LAST.lock();
    *l = Some(std::ffi::CString::new(msg).unwrap_or_default());
    l.as_ref().unwrap().as_ptr()
}

// ------------------------------------------------------------------ groups and collectives

#[no_mangle] pub extern "C" fn ncclGroupStart() -> NcclResult { GROUP_DEPTH.with(|d| d.set(d.get() + 1)); ncclSuccess }
#[no_mangle] pub extern "C" fn ncclGroupEnd() -> NcclResult {
    let depth = GROUP_DEPTH.with(|d| d.get());
    if depth == 0 { return fail(ncclInvalidUsage, "ncclGroupEnd without ncclGroupStart".into()); }
    GROUP_DEPTH.with(|d| d.set(depth - 1));
    if depth > 1 { return ncclSuccess; }
    let posts: Vec<(u64, Post)> = GROUP_POSTS.with(|p| std::mem::take(&mut *p.borrow_mut()));
    // park every post first (the sets complete as the later ranks' posts arrive), then wait only for the posts
    // whose other ranks are driven from other threads
    let mut rc = ncclSuccess;
    let mut parked = Vec::new();
    for (clique, p) in posts {
        match park_or_run(clique, p) { Ok(true) => {} Ok(false) => parked.push((clique, p)), Err(e) => rc = e }
    }
    for (clique, p) in parked { let r = wait_pending(clique, p); if r != ncclSuccess { rc = r; } }
    rc
}

/// Queue a rank's part of a collective; run the collective once every rank of the clique has posted its part
/// (in a group: at ncclGroupEnd; outside one: now, waiting for the other threads' posts).
fn post(c: *mut c_void, kind: Kind, send: *const c_void, recv: *mut c_void, count: usize, ty: c_int, stream: *mut c_void) -> NcclResult {
    let Some(cm) = comm(c) else { return ncclInvalidArgument };
    let Some(tb) = type_bytes(ty) else { return ncclInvalidArgument };
    if count > 0 && (recv.is_null() || (send.is_null() && !matches!(kind, Kind::Recv { .. }))) { return ncclInvalidArgument; }
    let bytes = count * tb;
    let send_bytes = match kind {
        Kind::ReduceScatter { .. } => bytes * cm.world as usize,
        _ => bytes,
    };
    let p = Post { kind, rank: cm.rank, send: send as usize, recv: recv as usize, bytes, send_bytes, stream };
    if GROUP_DEPTH.with(|d| d.get()) > 0 { GROUP_POSTS.with(|g| g.borrow_mut().push((cm.clique, p))); return ncclSuccess; }
    submit(cm.clique, p)
}

fn timeout() -> std::time::Duration {
    std::time::Duration::from_millis(30_000)
}

/// The posts a collective needs from the clique's pending list: for a collective, one per rank of the same kind
/// (posted first by each rank); for Send/Recv, the matching pair.
fn take_set(cl: &mut Clique, p: &Post) -> Option<Vec<Post>> {
    match p.kind {
        Kind::Send { peer } => {
            let i = cl.pending.iter().position(|q| q.rank == peer && q.kind == Kind::Recv { peer: p.rank })?;
            Some(vec![*p, cl.pending.remove(i)])
        }
        Kind::Recv { peer } => {
            let i = cl.pending.iter().position(|q| q.rank == peer && q.kind == Kind::Send { peer: p.rank })?;
            Some(vec![cl.pending.remove(i), *p])
        }
        kind => {
            let mut idx = Vec::new();
            for r in 0..cl.world {
                if r == p.rank { continue; }
                let i = cl.pending.iter().position(|q| q.rank == r && q.kind == kind)?;
                idx.push(i);
            }
            let mut set: Vec<Post> = idx.iter().map(|&i| cl.pending[i]).collect();
            idx.sort_unstable_by(|a, b| b.cmp(a));
            for i in idx { cl.pending.remove(i); }
            set.push(*p);
            set.sort_by_key(|q| q.rank);
            Some(set)
        }
    }
}

/// Run the collective if this post completes its set; else park the post. `Ok(true)`: ran; `Ok(false)`: parked.
fn park_or_run(clique: u64, p: Post) -> Result<bool, NcclResult> {
    // world 1: the collective is this rank's own copy
    let world = match state().lock().cliques.get(&clique) { Some(cl) => cl.world, None => return Err(fail(ncclInvalidArgument, "collective on a destroyed communicator".into())) };
    if world == 1 {
        return match p.kind {
            Kind::Send { .. } | Kind::Recv { .. } => Err(fail(ncclInvalidUsage, "ncclSend/ncclRecv to oneself on a one-rank communicator".into())),
            _ => { let r = run(&[p]); if r == ncclSuccess { Ok(true) } else { Err(r) } }
        };
    }
    let set = {
        let mut s = state().lock();
        let cl = s.cliques.get_mut(&clique).unwrap();
        match take_set(cl, &p) { Some(set) => Some(set), None => { cl.pending.push(p); None } }
    };
    match set { Some(set) => { let r = run(&set); if r == ncclSuccess { Ok(true) } else { Err(r) } } None => Ok(false) }
}

/// A post outside a group: run it now, or wait for the other ranks' posts (other threads).
fn submit(clique: u64, p: Post) -> NcclResult {
    match park_or_run(clique, p) { Ok(true) => ncclSuccess, Ok(false) => wait_pending(clique, p), Err(e) => e }
}

/// Wait for a parked post's set to complete: another thread completes it (and runs it), or a late post lets us.
fn wait_pending(clique: u64, p: Post) -> NcclResult {
    let deadline = std::time::Instant::now() + timeout();
    let mut first = true;
    loop {
        if !first { std::thread::sleep(std::time::Duration::from_micros(200)); }
        first = false;
        {
            let mut s = state().lock();
            let Some(cl) = s.cliques.get_mut(&clique) else { return fail(ncclInvalidArgument, "communicator destroyed while a collective was pending".into()) };
            // our post completed as part of a set another thread took: it is no longer pending
            if !cl.pending.iter().any(|q| q.rank == p.rank && q.kind == p.kind && q.send == p.send && q.recv == p.recv) { return ncclSuccess; }
            if let Some(i) = cl.pending.iter().position(|q| q.rank == p.rank && q.kind == p.kind && q.send == p.send && q.recv == p.recv) {
                let me = cl.pending.remove(i);
                match take_set(cl, &me) { Some(set) => { drop(s); return run(&set); } None => cl.pending.push(me) }
            }
        }
        if std::time::Instant::now() > deadline {
            let mut s = state().lock();
            if let Some(cl) = s.cliques.get_mut(&clique) { cl.pending.retain(|q| !(q.rank == p.rank && q.kind == p.kind && q.send == p.send && q.recv == p.recv)); }
            drop(s);
            return fail(ncclInvalidUsage, format!("{:?} from rank {}: the other ranks did not post theirs within the timeout (post every rank's part inside ncclGroupStart/End)", p.kind, p.rank));
        }
    }
}

/// Host ⊕ of `n` slabs of `bytes` (ncclSum on f32 / i32). Used by AllReduce / Reduce / ReduceScatter.
fn sum_slabs(host: &[u8], n: usize, bytes: usize, ty: c_int) -> Vec<u8> {
    let mut acc = vec![0u8; bytes];
    if bytes == 0 { return acc; }
    let count = bytes / 4;
    if ty == 7 {
        let out = unsafe { std::slice::from_raw_parts_mut(acc.as_mut_ptr() as *mut f32, count) };
        for r in 0..n {
            let sl = unsafe { std::slice::from_raw_parts(host[r * bytes..].as_ptr() as *const f32, count) };
            for i in 0..count { out[i] += sl[i]; }
        }
    } else {
        let out = unsafe { std::slice::from_raw_parts_mut(acc.as_mut_ptr() as *mut i32, count) };
        for r in 0..n {
            let sl = unsafe { std::slice::from_raw_parts(host[r * bytes..].as_ptr() as *const i32, count) };
            for i in 0..count { out[i] = out[i].wrapping_add(sl[i]); }
        }
    }
    acc
}

fn sum_ok(op: c_int, ty: c_int, name: &str) -> Result<(), NcclResult> {
    if op != 0 && op != 4 { return Err(fail(ncclInvalidUsage, format!("{name}: only ncclSum / ncclAvg are implemented"))); }
    if ty != 2 && ty != 3 && ty != 7 {
        return Err(fail(ncclInvalidUsage, format!("{name}: only ncclInt32 / ncclUint32 / ncclFloat32 (gather-and-sum / avg)")));
    }
    Ok(())
}

/// Run a complete collective: every stream waits for every other rank's operands, copies, then waits for every
/// other rank's copies (the collective completes on all streams together, as NCCL's does).
fn run(set: &[Post]) -> NcclResult {
    let mut events: Vec<*mut c_void> = Vec::with_capacity(set.len());
    let mut rc = ncclSuccess;
    let mut cuda = |e: c_int| { if e != SUCCESS && rc == ncclSuccess { rc = ncclUnhandledCudaError; } };
    for q in set { let mut ev = std::ptr::null_mut(); cuda(cudaEventCreateWithFlags(&mut ev, 2)); cuda(cudaEventRecord(ev, q.stream)); events.push(ev); }
    for (i, p) in set.iter().enumerate() {
        for (j, ev) in events.iter().enumerate() { if i != j { cuda(cudaStreamWaitEvent(p.stream, *ev, 0)); } }
        match p.kind {
            Kind::AllGather => {
                // recv[r * bytes ..] = rank r's send, for every rank (in place when send == recv + rank * bytes)
                for q in set { let dst = p.recv + q.rank as usize * p.bytes; if dst != q.send && p.bytes > 0 { cuda(cudaMemcpyAsync(dst as *mut c_void, q.send as *const c_void, p.bytes, 3, p.stream)); } }
            }
            Kind::Broadcast { root } => {
                let Some(r) = set.iter().find(|q| q.rank == root) else { return fail(ncclInvalidArgument, format!("broadcast root {} is not a rank", root)) };
                if p.recv != r.send && p.bytes > 0 { cuda(cudaMemcpyAsync(p.recv as *mut c_void, r.send as *const c_void, p.bytes, 3, p.stream)); }
            }
            Kind::AllReduce { op, ty } | Kind::Reduce { op, ty, .. } | Kind::ReduceScatter { op, ty } => {
                if i != 0 { continue; }
                if let Err(e) = sum_ok(op, ty, "nccl reduction") { return e; }
                let n = set.len();
                let slab = set[0].send_bytes;
                if slab == 0 { continue; }
                let mut host = vec![0u8; n * slab];
                for q in set {
                    cuda(cudaMemcpyAsync(host[q.rank as usize * slab..].as_mut_ptr() as *mut c_void,
                                        q.send as *const c_void, slab, 2, q.stream));
                }
                for q in set { cuda(cudaStreamSynchronize(q.stream)); }
                let mut acc = sum_slabs(&host, n, slab, ty);
                if op == 4 && n > 0 && slab >= 4 {
                    let count = slab / 4;
                    if ty == 7 {
                        let out = unsafe { std::slice::from_raw_parts_mut(acc.as_mut_ptr() as *mut f32, count) };
                        let nf = n as f32;
                        for i in 0..count { out[i] /= nf; }
                    } else {
                        let out = unsafe { std::slice::from_raw_parts_mut(acc.as_mut_ptr() as *mut i32, count) };
                        let ni = n as i32;
                        for i in 0..count { out[i] /= ni; }
                    }
                }
                match p.kind {
                    Kind::AllReduce { .. } => {
                        for q in set {
                            cuda(cudaMemcpyAsync(q.recv as *mut c_void, acc.as_ptr() as *const c_void, q.bytes, 1, q.stream));
                        }
                    }
                    Kind::Reduce { root, .. } => {
                        let Some(r) = set.iter().find(|q| q.rank == root) else {
                            return fail(ncclInvalidArgument, format!("reduce root {} is not a rank", root));
                        };
                        cuda(cudaMemcpyAsync(r.recv as *mut c_void, acc.as_ptr() as *const c_void, r.bytes, 1, r.stream));
                    }
                    Kind::ReduceScatter { .. } => {
                        for q in set {
                            let off = q.rank as usize * q.bytes;
                            if off + q.bytes > acc.len() {
                                return fail(ncclInvalidArgument, "ncclReduceScatter: recvcount * world exceeds send size".into());
                            }
                            cuda(cudaMemcpyAsync(q.recv as *mut c_void, acc[off..].as_ptr() as *const c_void, q.bytes, 1, q.stream));
                        }
                    }
                    _ => {}
                }
            }
            Kind::Send { .. } => {}
            Kind::Recv { .. } => {
                let s = &set[0];
                if p.bytes != s.bytes { return fail(ncclInvalidArgument, format!("ncclRecv of {} bytes against ncclSend of {}", p.bytes, s.bytes)); }
                if p.recv != s.send && p.bytes > 0 { cuda(cudaMemcpyAsync(p.recv as *mut c_void, s.send as *const c_void, p.bytes, 3, p.stream)); }
            }
        }
    }
    // join: a collective completes on every stream at once, so a rank's later write to its send buffer (or read
    // of its receive buffer) cannot race another rank's copy of it
    let mut done: Vec<*mut c_void> = Vec::with_capacity(set.len());
    for q in set { let mut ev = std::ptr::null_mut(); cuda(cudaEventCreateWithFlags(&mut ev, 2)); cuda(cudaEventRecord(ev, q.stream)); done.push(ev); }
    for (i, p) in set.iter().enumerate() { for (j, ev) in done.iter().enumerate() { if i != j { cuda(cudaStreamWaitEvent(p.stream, *ev, 0)); } } }
    for ev in events.into_iter().chain(done) { cudaEventDestroy(ev); }
    rc
}

#[no_mangle] pub extern "C" fn ncclAllGather(send: *const c_void, recv: *mut c_void, count: usize, ty: c_int, c: *mut c_void, stream: *mut c_void) -> NcclResult {
    post(c, Kind::AllGather, send, recv, count, ty, stream)
}
#[no_mangle] pub extern "C" fn ncclBroadcast(send: *const c_void, recv: *mut c_void, count: usize, ty: c_int, root: c_int, c: *mut c_void, stream: *mut c_void) -> NcclResult {
    post(c, Kind::Broadcast { root }, send, recv, count, ty, stream)
}
#[no_mangle] pub extern "C" fn ncclBcast(buf: *mut c_void, count: usize, ty: c_int, root: c_int, c: *mut c_void, stream: *mut c_void) -> NcclResult {
    post(c, Kind::Broadcast { root }, buf, buf, count, ty, stream)
}
#[no_mangle] pub extern "C" fn ncclSend(send: *const c_void, count: usize, ty: c_int, peer: c_int, c: *mut c_void, stream: *mut c_void) -> NcclResult {
    post(c, Kind::Send { peer }, send, send as *mut c_void, count, ty, stream)
}
#[no_mangle] pub extern "C" fn ncclRecv(recv: *mut c_void, count: usize, ty: c_int, peer: c_int, c: *mut c_void, stream: *mut c_void) -> NcclResult {
    post(c, Kind::Recv { peer }, recv, recv, count, ty, stream)
}
fn reduce_args(name: &'static str, op: c_int, ty: c_int) -> NcclResult {
    if op != 0 && op != 4 { return note("nccl.op", format!("{name}: only ncclSum / ncclAvg are implemented")); }
    if ty != 2 && ty != 3 && ty != 7 { return note("nccl.ty", format!("{name}: only ncclInt32 / ncclUint32 / ncclFloat32")); }
    ncclSuccess
}
#[no_mangle] pub extern "C" fn ncclAllReduce(send: *const c_void, recv: *mut c_void, count: usize, ty: c_int, op: c_int, c: *mut c_void, stream: *mut c_void) -> NcclResult {
    if reduce_args("ncclAllReduce", op, ty) != ncclSuccess { return ncclInvalidUsage; }
    post(c, Kind::AllReduce { op, ty }, send, recv, count, ty, stream)
}
#[no_mangle] pub extern "C" fn ncclReduce(send: *const c_void, recv: *mut c_void, count: usize, ty: c_int, op: c_int, root: c_int, c: *mut c_void, stream: *mut c_void) -> NcclResult {
    if reduce_args("ncclReduce", op, ty) != ncclSuccess { return ncclInvalidUsage; }
    post(c, Kind::Reduce { op, ty, root }, send, recv, count, ty, stream)
}
#[no_mangle] pub extern "C" fn ncclReduceScatter(send: *const c_void, recv: *mut c_void, count: usize, ty: c_int, op: c_int, c: *mut c_void, stream: *mut c_void) -> NcclResult {
    if reduce_args("ncclReduceScatter", op, ty) != ncclSuccess { return ncclInvalidUsage; }
    post(c, Kind::ReduceScatter { op, ty }, send, recv, count, ty, stream)
}
