//! The runtime proper: device singleton, streams with open-command-buffer batching, events on shared events,
//! copies, kernel launches, and graph capture/replay (replayed on the host from the recorded op list).
use crate::error::*;
use crate::memory::{Allocator, Loc};
use crate::metal::{self, Buffer, CmdBuf, Device, Encoder, Queue, Residency, SharedEvent};
use crate::module::{KernelPipeline, KernelRef, Module};
use std::collections::{HashMap, HashSet};
use std::ffi::c_void;
use std::sync::Arc;

pub const STREAM_LEGACY: u64 = 0;
pub const STREAM_PER_THREAD: u64 = 2;
const COMMIT_THRESHOLD: u32 = 512;
/// After a host sync the GPU would otherwise sit idle for a full threshold's worth of encoding (~0.4 ms of a decode step).
const COMMIT_FIRST: u32 = 32;
const GPU_WAIT_MS: i64 = 8000;
const MALLOC_BUDGET_FRAC: f64 = 0.97;

fn commit_threshold() -> u32 { COMMIT_THRESHOLD }
fn commit_first() -> u32 { COMMIT_FIRST }
fn gpu_wait_ms() -> i64 { GPU_WAIT_MS }
fn malloc_budget_frac() -> f64 {
    std::env::var("MVCC_MEM_FRACTION")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .filter(|&f| f > 0.0 && f <= 1.0)
        .unwrap_or(MALLOC_BUDGET_FRAC)
}
const FIRST_USER_STREAM: u64 = 16;
/// Most shared memory a block may use through the spill build (sm_90's opt-in limit; cudaDevAttrMaxSharedMemoryPerBlockOptin).
pub const MAX_SPILL_SMEM: u32 = 232448;
/// Device-side copies/fills up to this size are done by the CPU when the stream is idle.
const IDLE_CPU_COPY_MAX: u64 = 1 << 20;

pub struct DeviceInfo {
    pub name: String,
    pub total_mem: u64,
    pub core_count: u32,
    pub max_tg_mem: u32,
    pub registry_id: u64,
    pub major: u32,
    pub minor: u32,
}

struct Open {
    cb: Arc<CmdBuf>,
    enc: Option<Encoder>,
    enc_fence: Arc<metal::Fence>,
    ops: u32,
    /// buffers that must outlive this command buffer (staging copies)
    keepalive: Vec<Arc<Buffer>>,
}
impl Open {
    /// Ends the open compute encoder (if any), publishing its writes through the stream fence.
    fn close_encoder(&mut self) {
        if let Some(e) = self.enc.take() { e.update_fence(&self.enc_fence); e.end(); }
    }
}

pub struct Stream {
    pub id: u64,
    queue: Queue,
    pub non_blocking: bool,
    pub priority: i32,
    open: Option<Open>,
    fence: Arc<SharedEvent>,
    /// MTLFence handed from encoder to encoder: buffers are untracked, so this is what orders a blit after
    /// the kernels before it (and vice versa) inside one command buffer
    enc_fence: Arc<metal::Fence>,
    /// value the most recently committed buffer signals (0 = nothing committed yet)
    fence_value: u64,
    last_committed: Option<Arc<CmdBuf>>,
    /// event waits to encode into the next command buffer
    pending_waits: Vec<(Arc<SharedEvent>, u64)>,
    host_fence: Arc<SharedEvent>,
    host_fence_value: u64,
    /// committed buffers whose staging buffers can be dropped once fence >= value
    retired: Vec<(u64, Vec<Arc<Buffer>>)>,
    pub capture: Option<u64>,
    /// legacy-sync bookkeeping: peer stream id -> fence value already waited for
    waited: HashMap<u64, u64>,
    /// ops the open buffer commits at: commit_first() when the GPU had drained this stream's work when the
    /// buffer opened, else commit_threshold()
    commit_at: u32,
}

struct Rec {
    stream: u64,
    fence: Arc<SharedEvent>,
    value: u64,
    /// command buffer that carried the record (timing events)
    cb: Option<Arc<CmdBuf>>,
    host_time: f64,
}
pub struct Event {
    pub timing: bool,
    rec: Option<Rec>,
    /// last recorded inside a capture: (session, op index)
    captured: Option<(u64, usize)>,
}

#[derive(Clone)]
pub enum Op {
    Kernel { kp: Arc<KernelPipeline>, params: Vec<u8>, grid: [u32; 3], block: [u32; 3], tg_mem: u32 },
    Copy { src: Arc<Buffer>, soff: u64, dst: Arc<Buffer>, doff: u64, n: u64 },
    Fill { dst: Arc<Buffer>, off: u64, n: u64, value: u8 },
    /// pageable host -> device: memcpy into staging at launch, then blit
    HostIn { host: usize, dst: Arc<Buffer>, doff: u64, n: u64 },
    /// device -> pageable host: blit to staging, memcpy after completion
    HostOut { src: Arc<Buffer>, soff: u64, host: usize, n: u64 },
    HostFn { f: usize, data: usize },
    /// an event record node (cudaEventRecordWithFlags(cudaEventRecordExternal) inside a capture): every replay
    /// records the event at this point of the launching stream, so the host can time or wait on it
    EventRecord { ev: u64 },
}
pub struct Capture {
    pub origin: u64,
    pub streams: HashSet<u64>,
    pub ops: Vec<Op>,
    /// Dependency level of each op: 0 for an op with no predecessor, else 1 + the highest level it depends on
    /// (the previous op on its stream, and every op before an event it waited on). Ops of one level are mutually
    /// independent under CUDA stream semantics, so replay orders levels and lets a level's ops run concurrently.
    pub levels: Vec<u32>,
    /// index of the last op recorded on each stream of the session
    pub last: HashMap<u64, usize>,
    /// lower bound on the next op's level per stream, from event waits (fork/join)
    pub join: HashMap<u64, u32>,
    pub mode: i32,
}
pub struct Graph { pub ops: Vec<Op>, pub levels: Vec<u32> }
pub struct GraphExec { pub ops: Vec<Op>, pub levels: Vec<u32>, pub launches: u64 }

pub struct Runtime {
    pub dev: Device,
    pub res: Residency,
    pub alloc: Allocator,
    pub modules: Vec<Module>,
    pub kernels: HashMap<usize, KernelRef>,
    /// host shadow of a __device__/__constant__ variable -> (module, mangled name)
    pub symbols: HashMap<usize, (usize, String)>,
    /// module handle (from __cudaRegisterFatBinary) -> module index
    pub handles: HashMap<usize, usize>,
    streams: HashMap<u64, Stream>,
    next_id: u64,
    pub events: HashMap<u64, Event>,
    pub captures: HashMap<u64, Capture>,
    pub graphs: HashMap<u64, Graph>,
    pub execs: HashMap<u64, GraphExec>,
    pub failed: Option<String>,
    pub info: DeviceInfo,
    pub verbose: bool,
    tensor_fallback_reported: std::collections::HashSet<String>,
    spill_reported: std::collections::HashSet<String>,
    /// 4 KiB of zeros bound at [[buffer(1)]] for kernels whose ABI asks for it (`zero_page`): the target of device
    /// loads whose bounds guard is false, so the compiler can issue them unconditionally.
    zero_page: Option<Buffer>,
    /// 64 KiB of zeroed spinlocks bound at [[buffer(4)]] for kernels that use 64-bit atomics (`lock_page`).
    lock_page: Option<Buffer>,
    /// Shared-memory spill pools (prelude, "spill build"): one per slot-stride class, grown on demand. Old pools
    /// stay alive: in-flight command buffers may still reference them.
    spill_pools: Vec<(u32, Arc<Buffer>)>,
}

fn now() -> f64 { metal::now_seconds() }

impl Runtime {
    fn virtual_arch_from_env() -> u32 {
        std::env::var("MVCC_ARCH").ok().map(|a| crate::module::parse_sm_arch_string(&a)).unwrap_or(0)
    }

    fn apply_virtual_arch(info: &mut DeviceInfo, v: u32) {
        if v == 0 { return; }
        let (maj, min) = crate::module::compute_capability_from_virtual_arch(v);
        if maj > info.major || (maj == info.major && min > info.minor) {
            info.major = maj;
            info.minor = min;
        }
    }

    pub fn new() -> Result<Runtime, CudaError> {
        let dev = Device::create().ok_or(NO_DEVICE)?;
        let res = dev.new_residency();
        let cores = dev.core_count();
        let mut info = DeviceInfo {
            name: dev.name(),
            total_mem: dev.recommended_working_set(),
            core_count: if cores == 0 { 8 } else { cores },
            max_tg_mem: dev.max_threadgroup_memory(),
            registry_id: dev.registry_id(),
            major: 8, minor: 9,
        };
        Self::apply_virtual_arch(&mut info, Self::virtual_arch_from_env());
        let verbose = std::env::var("MVCC_VERBOSE").is_ok();
        if verbose { eprintln!("[MVCC] device {} cores={} mem={}MiB tgmem={} metal4={}", info.name, info.core_count, info.total_mem >> 20, info.max_tg_mem, dev.supports_metal4()); }
        let mut rt = Runtime {
            dev, res, alloc: Allocator::new(), modules: Vec::new(), kernels: HashMap::new(), symbols: HashMap::new(), handles: HashMap::new(),
            streams: HashMap::new(), next_id: FIRST_USER_STREAM, events: HashMap::new(), captures: HashMap::new(),
            graphs: HashMap::new(), execs: HashMap::new(), failed: None, info, verbose, tensor_fallback_reported: Default::default(), spill_reported: Default::default(), zero_page: None, lock_page: None, spill_pools: Vec::new(),
        };
        rt.make_stream(STREAM_LEGACY, false, 0);
        rt.make_stream(STREAM_PER_THREAD, false, 0);
        Ok(rt)
    }

    // ------------------------------------------------------------------ streams

    fn make_stream(&mut self, id: u64, non_blocking: bool, priority: i32) {
        let s = Stream {
            id, queue: self.dev.new_queue(Some(&self.res)), non_blocking, priority, open: None,
            fence: Arc::new(self.dev.new_shared_event()), enc_fence: Arc::new(self.dev.new_fence()), fence_value: 0, last_committed: None, pending_waits: Vec::new(),
            host_fence: Arc::new(self.dev.new_shared_event()), host_fence_value: 0, retired: Vec::new(), capture: None, waited: HashMap::new(),
            commit_at: commit_first(),
        };
        self.streams.insert(id, s);
    }
    pub fn create_stream(&mut self, non_blocking: bool, priority: i32) -> u64 {
        let id = self.next_id; self.next_id += 1;
        self.make_stream(id, non_blocking, priority);
        id
    }
    pub fn destroy_stream(&mut self, id: u64) -> CudaError {
        if id < FIRST_USER_STREAM { return INVALID_RESOURCE_HANDLE; }
        if !self.streams.contains_key(&id) { return INVALID_RESOURCE_HANDLE; }
        // CUDA: destroying a stream with pending work is fine; the work completes. We commit and drop.
        self.commit(id);
        self.streams.remove(&id);
        SUCCESS
    }
    /// Map a user-visible cudaStream_t to a stream id. 0 and 1 are the legacy stream, 2 is per-thread.
    pub fn stream_id(&self, h: usize) -> Option<u64> {
        let id = match h { 0 | 1 => STREAM_LEGACY, 2 => STREAM_PER_THREAD, x => x as u64 };
        if self.streams.contains_key(&id) { Some(id) } else { None }
    }
    pub fn stream(&self, id: u64) -> &Stream { &self.streams[&id] }
    pub fn stream_mut(&mut self, id: u64) -> &mut Stream { self.streams.get_mut(&id).unwrap() }
    pub fn stream_ids(&self) -> Vec<u64> { self.streams.keys().copied().collect() }

    /// Legacy default-stream semantics: work on the legacy stream orders after all blocking streams, and vice versa.
    fn legacy_dependencies(&mut self, id: u64) -> Vec<(Arc<SharedEvent>, u64)> {
        let mut deps = Vec::new();
        let peers: Vec<u64> = if id == STREAM_LEGACY {
            self.streams.values().filter(|s| s.id != STREAM_LEGACY && !s.non_blocking && s.capture.is_none()).map(|s| s.id).collect()
        } else if !self.streams[&id].non_blocking {
            vec![STREAM_LEGACY]
        } else { Vec::new() };
        for p in peers {
            if self.streams[&p].open.is_some() { self.commit(p); }
            let (fence, v) = { let s = &self.streams[&p]; (s.fence.clone(), s.fence_value) };
            if v == 0 { continue; }
            let s = self.streams.get_mut(&id).unwrap();
            let seen = s.waited.get(&p).copied().unwrap_or(0);
            if v > seen { s.waited.insert(p, v); deps.push((fence, v)); }
        }
        deps
    }

    fn open(&mut self, id: u64) {
        if self.streams[&id].open.is_some() { return; }
        let deps = self.legacy_dependencies(id);
        let s = self.streams.get_mut(&id).unwrap();
        let cb = Arc::new(s.queue.new_cmdbuf());
        // untracked resources: successive command buffers on one queue may overlap unless told otherwise
        if s.fence_value > 0 { cb.wait(&s.fence, s.fence_value); }
        for (ev, v) in deps { cb.wait(&ev, v); }
        for (ev, v) in s.pending_waits.drain(..) { cb.wait(&ev, v); }
        // drop staging buffers of completed work
        let done = s.fence.value();
        s.retired.retain(|(v, _)| *v > done);
        s.open = Some(Open { cb, enc: None, enc_fence: s.enc_fence.clone(), ops: 0, keepalive: Vec::new() });
    }

    fn encoder(&mut self, id: u64) -> &Encoder {
        self.open(id);
        let o = self.streams.get_mut(&id).unwrap().open.as_mut().unwrap();
        if o.enc.is_none() {
            let e = o.cb.compute();
            e.wait_fence(&o.enc_fence);
            o.enc = Some(e);
        }
        o.enc.as_ref().unwrap()
    }
    fn end_encoder(&mut self, id: u64) {
        if let Some(o) = self.streams.get_mut(&id).unwrap().open.as_mut() { o.close_encoder(); }
    }
    fn blit(&mut self, id: u64) -> metal::Blit {
        self.open(id);
        self.end_encoder(id);
        let o = self.streams.get_mut(&id).unwrap().open.as_mut().unwrap();
        o.ops += 1;
        let b = o.cb.blit();
        b.wait_fence(&o.enc_fence);
        b
    }
    /// Blits end by updating the stream fence so the next encoder orders after them.
    fn end_blit(&mut self, id: u64, b: metal::Blit) {
        let o = self.streams.get_mut(&id).unwrap().open.as_mut().unwrap();
        b.update_fence(&o.enc_fence);
        b.end();
    }
    fn bump_ops(&mut self, id: u64) {
        let (ops, at) = {
            let s = self.streams.get_mut(&id).unwrap();
            let o = s.open.as_mut().unwrap();
            o.ops += 1;
            if o.ops == 1 {
                // First op of the buffer: if the GPU has finished everything this stream committed, it is
                // idle and waiting on us; start over with a small buffer.
                let drained = s.last_committed.as_ref().map(|c| c.status() >= 2).unwrap_or(true);
                s.commit_at = if drained { commit_first() } else { commit_threshold() };
            }
            (o.ops, s.commit_at.max(1))
        };
        if ops >= at { self.commit(id); }
    }
    fn keepalive(&mut self, id: u64, b: Arc<Buffer>) {
        if let Some(o) = self.streams.get_mut(&id).unwrap().open.as_mut() { o.keepalive.push(b); }
    }

    /// Commit the stream's open command buffer (if any). Every committed buffer signals the stream fence.
    pub fn commit(&mut self, id: u64) {
        let s = self.streams.get_mut(&id).unwrap();
        let Some(mut o) = s.open.take() else { return };
        o.close_encoder();
        s.fence_value += 1;
        o.cb.signal(&s.fence, s.fence_value);
        o.cb.commit();
        if !o.keepalive.is_empty() { s.retired.push((s.fence_value, std::mem::take(&mut o.keepalive))); }
        s.last_committed = Some(o.cb.clone());
    }
    fn commit_with_handler(&mut self, id: u64, cb_fn: extern "C" fn(*mut c_void, i32), ctx: *mut c_void) {
        self.open(id);
        let s = self.streams.get_mut(&id).unwrap();
        let o = s.open.as_mut().unwrap();
        o.close_encoder();
        o.cb.on_completed(cb_fn, ctx);
        self.commit(id);
    }

    fn check_cb(&mut self, cb: &CmdBuf) -> CudaError {
        if cb.status() == 3 {
            let msg = cb.error().unwrap_or_else(|| "GPU command buffer failed".into());
            if self.failed.is_none() { eprintln!("[MVCC] GPU fault: {}", msg); self.failed = Some(msg); }
            return ILLEGAL_ADDRESS;
        }
        SUCCESS
    }
    /// Wait for a stream's last command buffer without `waitUntilCompleted`.
    /// A timed shared-event wait returns on fault or wedge; an infinite Metal
    /// wait on a wedged buffer can take down the machine.
    fn await_stream(&mut self, sid: u64) -> CudaError {
        if self.failed.is_some() { return ILLEGAL_ADDRESS; }
        let s = &self.streams[&sid];
        let (fence, value, cb) = (s.fence.clone(), s.fence_value, s.last_committed.clone());
        let Some(cb) = cb else { return SUCCESS; };
        if cb.status() >= 2 { return self.check_cb(&cb); }
        let signaled = if value > 0 { fence.wait(value, gpu_wait_ms()) } else { false };
        if cb.status() >= 2 { return self.check_cb(&cb); }
        if signaled { return SUCCESS; }
        let msg = format!("GPU wait timed out after {} ms (command buffer still in flight)", gpu_wait_ms());
        eprintln!("[MVCC] {}", msg);
        self.failed = Some(msg);
        ILLEGAL_ADDRESS
    }
    /// The host is about to wait on (or poll) GPU progress: commit every stream's open buffer, not only the awaited
    /// one. A kernel on one stream may spin on a flag a kernel on another stream stores (a peer all-reduce over the
    /// logical devices): CUDA submits every launch promptly, so it runs; here the other stream's launch would sit in
    /// an open buffer until its own threshold or sync, and the awaited kernel would never finish.
    fn commit_all(&mut self) {
        let ids: Vec<u64> = self.streams.iter().filter(|(_, s)| s.open.is_some() && s.capture.is_none()).map(|(id, _)| *id).collect();
        for id in ids { self.commit(id); }
    }
    pub fn sync_stream(&mut self, id: u64) -> CudaError {
        if self.failed.is_some() { return ILLEGAL_ADDRESS; }
        self.commit(id);
        self.commit_all();
        self.await_stream(id)
    }
    pub fn query_stream(&mut self, id: u64) -> CudaError {
        self.commit(id);
        self.commit_all();
        let cb = self.streams[&id].last_committed.clone();
        match cb { None => SUCCESS, Some(cb) => match cb.status() { 2 => SUCCESS, 3 => self.check_cb(&cb), _ => NOT_READY } }
    }
    pub fn sync_device(&mut self) -> CudaError {
        let ids = self.stream_ids();
        let mut rc = SUCCESS;
        for id in &ids { self.commit(*id); }
        for id in ids { let r = self.sync_stream(id); if r != SUCCESS { rc = r; } }
        rc
    }
    /// Is the stream idle: nothing open and the last committed buffer finished?
    fn idle(&self, id: u64) -> bool {
        let s = &self.streams[&id];
        s.open.is_none() && s.pending_waits.is_empty() && s.last_committed.as_ref().map(|c| c.status() >= 2).unwrap_or(true)
    }

    // ------------------------------------------------------------------ events

    pub fn create_event(&mut self, timing: bool) -> u64 {
        let id = self.next_id; self.next_id += 1;
        self.events.insert(id, Event { timing, rec: None, captured: None });
        id
    }
    /// The level the next op recorded on `sid` in session `sess` gets.
    fn capture_next_level(&self, sess: u64, sid: u64) -> u32 {
        let c = &self.captures[&sess];
        let chain = c.last.get(&sid).map(|&i| c.levels[i] + 1).unwrap_or(0);
        chain.max(c.join.get(&sid).copied().unwrap_or(0))
    }
    /// Record an op on a captured stream.
    fn capture_push(&mut self, sess: u64, sid: u64, op: Op) {
        let level = self.capture_next_level(sess, sid);
        let c = self.captures.get_mut(&sess).unwrap();
        c.ops.push(op);
        c.levels.push(level);
        let idx = c.ops.len() - 1;
        c.last.insert(sid, idx);
        c.join.remove(&sid);
    }
    /// cudaEventRecordWithFlags: `external` (cudaEventRecordExternal) inside a capture records an event node - the
    /// event is recorded on every replay of the graph - where a plain record is only a dependency marker.
    pub fn record_event_flags(&mut self, ev: u64, sid: u64, external: bool) -> CudaError {
        if !self.events.contains_key(&ev) { return INVALID_RESOURCE_HANDLE; }
        if let Some(sess) = self.streams[&sid].capture {
            if external { self.capture_push(sess, sid, Op::EventRecord { ev }); return SUCCESS; }
        }
        self.record_event(ev, sid)
    }
    pub fn record_event(&mut self, ev: u64, sid: u64) -> CudaError {
        if !self.events.contains_key(&ev) { return INVALID_RESOURCE_HANDLE; }
        if let Some(sess) = self.streams[&sid].capture {
            // the event stands for everything recorded on this stream so far: a waiter's next op goes above it
            let bound = self.capture_next_level(sess, sid) as usize;
            let e = self.events.get_mut(&ev).unwrap();
            e.captured = Some((sess, bound)); e.rec = None;
            return SUCCESS;
        }
        let timing = self.events[&ev].timing;
        if timing { self.commit(sid); }
        let s = &self.streams[&sid];
        let (value, cb) = if s.open.is_some() { (s.fence_value + 1, None) } else { (s.fence_value, s.last_committed.clone()) };
        let rec = Rec { stream: sid, fence: s.fence.clone(), value, cb, host_time: now() };
        let e = self.events.get_mut(&ev).unwrap();
        e.rec = Some(rec); e.captured = None;
        SUCCESS
    }
    /// Make sure the work an event refers to has been committed.
    fn flush_event(&mut self, ev: u64) {
        let Some(r) = self.events[&ev].rec.as_ref() else { return };
        let (sid, v) = (r.stream, r.value);
        if let Some(s) = self.streams.get(&sid) { if s.fence_value < v { self.commit(sid); } }
    }
    pub fn sync_event(&mut self, ev: u64) -> CudaError {
        if !self.events.contains_key(&ev) { return INVALID_RESOURCE_HANDLE; }
        self.flush_event(ev);
        self.commit_all();
        let Some(r) = self.events[&ev].rec.as_ref() else { return SUCCESS };
        let (f, v, sid) = (r.fence.clone(), r.value, r.stream);
        if let Some(cb) = self.streams.get(&sid).and_then(|s| s.last_committed.clone()) {
            if cb.status() >= 2 { return self.check_cb(&cb); }
        }
        if !f.wait(v, gpu_wait_ms()) {
            if let Some(cb) = self.streams.get(&sid).and_then(|s| s.last_committed.clone()) {
                if cb.status() >= 2 { return self.check_cb(&cb); }
            }
            let msg = format!("GPU event wait timed out after {} ms", gpu_wait_ms());
            eprintln!("[MVCC] {}", msg);
            self.failed = Some(msg);
            return ILLEGAL_ADDRESS;
        }
        if let Some(cb) = self.streams.get(&sid).and_then(|s| s.last_committed.clone()) {
            return self.check_cb(&cb);
        }
        SUCCESS
    }
    pub fn query_event(&mut self, ev: u64) -> CudaError {
        if !self.events.contains_key(&ev) { return INVALID_RESOURCE_HANDLE; }
        self.flush_event(ev);
        self.commit_all();
        let Some(r) = self.events[&ev].rec.as_ref() else { return SUCCESS };
        if r.fence.value() >= r.value { SUCCESS } else { NOT_READY }
    }
    pub fn elapsed_ms(&mut self, a: u64, b: u64) -> Result<f32, CudaError> {
        let (Some(ea), Some(eb)) = (self.events.get(&a), self.events.get(&b)) else { return Err(INVALID_RESOURCE_HANDLE) };
        if !ea.timing || !eb.timing { return Err(INVALID_RESOURCE_HANDLE); }
        let (Some(ra), Some(rb)) = (&ea.rec, &eb.rec) else { return Err(INVALID_RESOURCE_HANDLE) };
        if ra.fence.value() < ra.value || rb.fence.value() < rb.value { return Err(NOT_READY); }
        // The shared-event signal lands slightly before Metal finalizes the command buffer's GPUEndTime, so
        // wait for the buffer itself before reading the timestamp.
        // The fence has already signaled (checked above), so completion is imminent: spin briefly, bounded,
        // instead of waitUntilCompleted. Never mix a host clock with a GPU clock: that gave negative probes.
        let stamp = |r: &Rec| -> Option<f64> {
            let c = r.cb.as_ref()?;
            let deadline = std::time::Instant::now() + std::time::Duration::from_millis(gpu_wait_ms() as u64);
            while c.status() < 2 {
                if std::time::Instant::now() >= deadline { return None; }
                std::thread::yield_now();
            }
            Some(c.gpu_end()).filter(|t| *t > 0.0)
        };
        let (ta, tb) = match (stamp(ra), stamp(rb)) {
            (Some(a), Some(b)) => (a, b),
            _ => (ra.host_time, rb.host_time),
        };
        Ok(((tb - ta) * 1000.0) as f32)
    }
    pub fn stream_wait_event(&mut self, sid: u64, ev: u64) -> CudaError {
        if !self.events.contains_key(&ev) { return INVALID_RESOURCE_HANDLE; }
        if let Some((sess, bound)) = self.events[&ev].captured {
            // fork/join inside a capture: the waiting stream joins the session, and its next op is ordered after
            // everything the event stands for (dependency levels; replay runs a level's ops concurrently)
            if self.streams[&sid].capture.is_none() {
                if self.streams[&sid].open.is_some() { self.commit(sid); }
                self.streams.get_mut(&sid).unwrap().capture = Some(sess);
                self.captures.get_mut(&sess).unwrap().streams.insert(sid);
            }
            let j = self.captures.get_mut(&sess).unwrap().join.entry(sid).or_insert(0);
            *j = (*j).max(bound as u32);
            return SUCCESS;
        }
        if self.streams[&sid].capture.is_some() { return SUCCESS; }  // relaxed: ignore non-captured dependencies
        self.flush_event(ev);
        let Some(r) = self.events[&ev].rec.as_ref() else { return SUCCESS };
        if r.stream == sid { return SUCCESS; }
        let (f, v) = (r.fence.clone(), r.value);
        if self.streams[&sid].open.is_some() {
            self.end_encoder(sid);
            let cb = self.streams[&sid].open.as_ref().unwrap().cb.clone();
            cb.wait(&f, v);
        } else {
            self.streams.get_mut(&sid).unwrap().pending_waits.push((f, v));
        }
        SUCCESS
    }
    pub fn destroy_event(&mut self, ev: u64) -> CudaError {
        if self.events.remove(&ev).is_some() { SUCCESS } else { INVALID_RESOURCE_HANDLE }
    }

    // ------------------------------------------------------------------ host functions

    pub fn launch_host_func(&mut self, sid: u64, f: usize, data: usize) -> CudaError {
        if let Some(sess) = self.streams[&sid].capture {
            self.capture_push(sess, sid, Op::HostFn { f, data });
            return SUCCESS;
        }
        let (hf, v) = { let s = self.streams.get_mut(&sid).unwrap(); s.host_fence_value += 1; (s.host_fence.clone(), s.host_fence_value) };
        // if nothing is open, the buffer must still order after previously committed work
        if self.streams[&sid].open.is_none() {
            let (fence, fv) = { let s = &self.streams[&sid]; (s.fence.clone(), s.fence_value) };
            self.open(sid);
            if fv > 0 { self.streams[&sid].open.as_ref().unwrap().cb.wait(&fence, fv); }
        }
        let ctx = Box::new(HostFnCtx { f, data, fence: hf.clone(), value: v });
        self.commit_with_handler(sid, host_fn_trampoline, Box::into_raw(ctx) as *mut c_void);
        self.streams.get_mut(&sid).unwrap().pending_waits.push((hf, v));
        SUCCESS
    }

    // ------------------------------------------------------------------ memory

    pub fn malloc(&mut self, size: u64) -> Result<u64, CudaError> {
        let used = self.dev.current_allocated();
        let cap = (self.info.total_mem as f64 * malloc_budget_frac()) as u64;
        if used.saturating_add(size) > cap {
            eprintln!(
                "[MVCC] refusing {} MiB alloc: {} + {} > {} MiB budget ({:.0}% of {} MiB recommended)",
                size >> 20, used >> 20, size >> 20, cap >> 20, malloc_budget_frac() * 100.0, self.info.total_mem >> 20
            );
            return Err(MEMORY_ALLOCATION);
        }
        let (dev, res) = (&self.dev, &self.res);
        self.alloc.alloc_device(dev, res, size).ok_or(MEMORY_ALLOCATION)
    }
    pub fn free(&mut self, addr: u64) -> CudaError {
        if addr == 0 { return SUCCESS; }
        if self.alloc.resolve_device(addr).map(|(a, off)| off != 0 || a.gpu_addr != addr).unwrap_or(true) { return INVALID_VALUE; }
        // cudaFree is synchronous with respect to the device: anything in flight may still use the memory.
        let busy = self.stream_ids().iter().any(|id| !self.idle(*id));
        if busy { self.sync_device(); }
        self.alloc.free_device(&self.res, addr);
        SUCCESS
    }
    pub fn host_alloc(&mut self, size: u64) -> Result<*mut u8, CudaError> {
        let (dev, res) = (&self.dev, &self.res);
        self.alloc.alloc_host(dev, res, size).ok_or(MEMORY_ALLOCATION)
    }
    pub fn host_free(&mut self, p: usize) -> CudaError {
        if p == 0 { return SUCCESS; }
        let busy = self.stream_ids().iter().any(|id| !self.idle(*id));
        if busy { self.sync_device(); }
        if self.alloc.free_host(&self.res, p).is_some() { SUCCESS } else { INVALID_VALUE }
    }
    pub fn mem_info(&self) -> (u64, u64) {
        let total = self.info.total_mem;
        let used = self.dev.current_allocated();
        (total.saturating_sub(used), total)
    }

    fn resolve_dev(&self, addr: u64) -> Option<Loc> { self.alloc.resolve_device(addr).map(|(a, off)| Loc::from_alloc(a, off)) }
    fn resolve_host(&self, p: usize) -> Option<Loc> { self.alloc.resolve_host(p).map(|(a, off)| Loc::from_alloc(a, off)) }

    /// Blit copy between two Metal-backed locations on a stream (or record it during capture).
    fn copy_dev(&mut self, sid: u64, src: &Loc, dst: &Loc, n: u64) {
        if let Some(sess) = self.streams[&sid].capture {
            self.capture_push(sess, sid, Op::Copy { src: src.buf.clone(), soff: src.offset, dst: dst.buf.clone(), doff: dst.offset, n });
            return;
        }
        // Small copies on an idle stream: a CPU memcpy beats a ~150 us GPU round trip. Large ones go to the
        // blit engine so the caller is not blocked and the copy runs at GPU bandwidth.
        if n <= IDLE_CPU_COPY_MAX && self.idle(sid) {
            unsafe { std::ptr::copy(src.cpu, dst.cpu, n as usize) };
            return;
        }
        let b = self.blit(sid);
        b.copy(&src.buf, src.offset, &dst.buf, dst.offset, n);
        self.end_blit(sid, b);
    }

    /// memcpy semantics. `kind`: 0 H2H, 1 H2D, 2 D2H, 3 D2D, 4 default. `sync` = cudaMemcpy (legacy stream, blocking).
    pub fn memcpy(&mut self, dst: usize, src: usize, n: u64, kind: i32, sid: u64, sync: bool) -> CudaError {
        if n == 0 { return SUCCESS; }
        let dst_dev = self.resolve_dev(dst as u64);
        let src_dev = self.resolve_dev(src as u64);
        let (dst_is_dev, src_is_dev) = match kind {
            0 => (false, false), 1 => (true, false), 2 => (false, true), 3 => (true, true),
            4 => (dst_dev.is_some(), src_dev.is_some()),
            _ => return INVALID_MEMCPY_DIRECTION,
        };
        let capturing = self.streams[&sid].capture.is_some();
        match (dst_is_dev, src_is_dev) {
            (false, false) => {
                if capturing { return STREAM_CAPTURE_UNSUPPORTED; }
                if sync { self.sync_blocking_streams(); }
                unsafe { std::ptr::copy(src as *const u8, dst as *mut u8, n as usize) };
                SUCCESS
            }
            (true, true) => {
                let (Some(d), Some(s)) = (dst_dev, src_dev) else { return INVALID_DEVICE_POINTER };
                if sync && !capturing { self.sync_blocking_streams(); }
                self.copy_dev(sid, &s, &d, n);
                SUCCESS
            }
            (true, false) => {
                let Some(d) = dst_dev else { return INVALID_DEVICE_POINTER };
                if let Some(s) = self.resolve_host(src) { self.copy_dev(sid, &s, &d, n); return SUCCESS; }
                // pageable host source
                if let Some(sess) = self.streams[&sid].capture {
                    self.capture_push(sess, sid, Op::HostIn { host: src, dst: d.buf.clone(), doff: d.offset, n });
                    return SUCCESS;
                }
                if sync { self.sync_blocking_streams(); }
                if self.idle(sid) { unsafe { std::ptr::copy(src as *const u8, d.cpu, n as usize) }; return SUCCESS; }
                // stream busy: stage through a transient buffer so the copy stays ordered and asynchronous
                let Some(stg) = self.dev.new_buffer(n) else { return MEMORY_ALLOCATION };
                unsafe { std::ptr::copy(src as *const u8, stg.contents(), n as usize) };
                self.res.add(&stg); self.res.commit();
                let stg = Arc::new(stg);
                let b = self.blit(sid);
                b.copy(&stg, 0, &d.buf, d.offset, n);
                self.end_blit(sid, b);
                self.keepalive(sid, stg);
                SUCCESS
            }
            (false, true) => {
                let Some(s) = src_dev else { return INVALID_DEVICE_POINTER };
                if let Some(d) = self.resolve_host(dst) { self.copy_dev(sid, &s, &d, n); return SUCCESS; }
                if let Some(sess) = self.streams[&sid].capture {
                    self.capture_push(sess, sid, Op::HostOut { src: s.buf.clone(), soff: s.offset, host: dst, n });
                    return SUCCESS;
                }
                // pageable destination: the copy must observe all prior work on the stream, so wait for it
                if sync { self.sync_blocking_streams(); }
                let rc = self.sync_stream(sid);
                if rc != SUCCESS { return rc; }
                unsafe { std::ptr::copy(s.cpu, dst as *mut u8, n as usize) };
                SUCCESS
            }
        }
    }
    /// cudaMemcpy/cudaMemset (no stream) run on the legacy stream, which orders after all blocking streams.
    fn sync_blocking_streams(&mut self) {
        let ids: Vec<u64> = self.streams.values().filter(|s| !s.non_blocking && s.capture.is_none()).map(|s| s.id).collect();
        for id in ids { self.sync_stream(id); }
    }

    pub fn memcpy2d(&mut self, dst: usize, dpitch: u64, src: usize, spitch: u64, width: u64, height: u64, kind: i32, sid: u64, sync: bool) -> CudaError {
        if width > dpitch || width > spitch { return INVALID_PITCH_VALUE; }
        if dpitch == width && spitch == width { return self.memcpy(dst, src, width * height, kind, sid, sync); }
        for r in 0..height {
            let rc = self.memcpy(dst + (r * dpitch) as usize, src + (r * spitch) as usize, width, kind, sid, sync);
            if rc != SUCCESS { return rc; }
        }
        SUCCESS
    }

    pub fn memset(&mut self, dst: usize, value: u8, n: u64, sid: u64, sync: bool) -> CudaError {
        if n == 0 { return SUCCESS; }
        let Some(d) = self.resolve_dev(dst as u64) else {
            if let Some(h) = self.resolve_host(dst) { let _ = h; unsafe { std::ptr::write_bytes(dst as *mut u8, value, n as usize) }; return SUCCESS; }
            return INVALID_DEVICE_POINTER;
        };
        if let Some(sess) = self.streams[&sid].capture {
            self.capture_push(sess, sid, Op::Fill { dst: d.buf.clone(), off: d.offset, n, value });
            return SUCCESS;
        }
        if sync { self.sync_blocking_streams(); }
        // Shared MTLBuffers are CPU-mapped: an idle stream fills on the CPU at any size (no GPU wiring,
        // no command buffer). A busy stream keeps the blit engine; never force a stream sync here: a
        // sync per fill would serialize the whole stream.
        if self.idle(sid) {
            unsafe { std::ptr::write_bytes(d.cpu, value, n as usize) };
            return SUCCESS;
        }
        let b = self.blit(sid);
        b.fill(&d.buf, d.offset, n, value);
        self.end_blit(sid, b);
        SUCCESS
    }

    pub fn pointer_kind(&self, p: usize) -> (i32, usize, usize) {
        // returns (cudaMemoryType, devicePointer, hostPointer): 0 unregistered, 1 host, 2 device, 3 managed
        if let Some((a, off)) = self.alloc.resolve_device(p as u64) { return (2, p, unsafe { a.cpu_ptr().add(off as usize) } as usize); }
        if let Some((a, off)) = self.alloc.resolve_host(p) { return (1, (a.gpu_addr + off) as usize, p); }
        (0, 0, p)
    }

    // ------------------------------------------------------------------ modules & launch

    pub fn register_module(&mut self, blob: *const u8) -> Result<usize, String> {
        let (msl, abi) = crate::module::parse_blob(blob)?;
        Self::apply_virtual_arch(&mut self.info, abi.virtual_arch);
        let mut m = Module::new(msl, abi);
        if self.verbose {
            eprintln!("[MVCC] registered module {} with {} kernels (virtual sm_{}{})", m.hash, m.abi.kernels.len(), self.info.major, self.info.minor);
        }
        m.index = self.modules.len();
        self.modules.push(m);
        let idx = self.modules.len() - 1;
        self.handles.insert(blob as usize, idx);
        Ok(idx)
    }
    pub fn module_index(&self, blob: usize) -> Option<usize> { self.handles.get(&blob).copied() }
    pub fn register_kernel(&mut self, module: usize, host_fn: usize, name: &str) -> Result<(), String> {
        if self.modules[module].kernel_abi(name).is_none() { return Err(format!("kernel {} not found in module {}", name, self.modules[module].hash)); }
        self.kernels.insert(host_fn, KernelRef { module, name: name.to_string(), max_dynamic_smem: 0, preferred_carveout: -1 });
        Ok(())
    }
    pub fn kernel_pipeline(&mut self, host_fn: usize) -> Result<(KernelRef, Arc<KernelPipeline>), CudaError> {
        let Some(k) = self.kernels.get(&host_fn).cloned() else { return Err(INVALID_DEVICE_FUNCTION) };
        let dev = &self.dev;
        let kp = self.modules[k.module].pipeline(dev, &k.name).map_err(|e| { eprintln!("[MVCC] pipeline creation failed for {}: {}", k.name, e); INVALID_KERNEL_IMAGE })?;
        Ok((k, kp))
    }

    pub fn launch(&mut self, host_fn: usize, grid: [u32; 3], block: [u32; 3], args: *const *const c_void, smem: u32, sid: u64) -> CudaError {
        if self.failed.is_some() { return ILLEGAL_ADDRESS; }
        let (k, mut kp) = match self.kernel_pipeline(host_fn) { Ok(x) => x, Err(e) => return e };
        let threads = block[0] as u64 * block[1] as u64 * block[2] as u64;
        if threads == 0 || grid.iter().any(|g| *g == 0) { return INVALID_CONFIGURATION; }
        // Recovered body reserves conversion scratch (sized for recovered_max_threads) next to static shared
        // memory. A launch that exceeds that thread count, or that no longer fits in threadgroup memory with the
        // scratch, runs the exact twin. Dynamic shared uses the proved footprint (dyn_smem_used / relocate);
        // the twin keeps the host's request. Assumptions on arguments are evaluated below.
        let effective_smem = |kp: &KernelPipeline, smem: u32| crate::module::effective_dyn_smem(&kp.abi, smem);
        // pack parameters (the recovered body and its exact twin share the parameter block)
        let mut params = vec![0u8; kp.abi.param_block_size.max(4) as usize];
        for (i, p) in kp.abi.params.iter().enumerate() {
            unsafe {
                let src = *args.add(i) as *const u8;
                if src.is_null() { return INVALID_VALUE; }
                std::ptr::copy_nonoverlapping(src, params.as_mut_ptr().add(p.offset as usize), p.size as usize);
            }
        }
        // Recovered body runs thread_scale × the host's block (physical thread p is virtual thread p % block.x).
        let mut block = block;
        let mut threads = threads;
        if let Some(twin) = kp.abi.exact_variant.clone() {
            let scale = kp.abi.thread_scale.max(1) as u64;
            let too_many = (kp.abi.recovered_max_threads != 0 && threads * scale > kp.abi.recovered_max_threads as u64) || threads * scale > kp.max_threads as u64;
            // does not fit in threadgroup memory: the twin runs only when it fits itself; otherwise both would
            // spill to the device-memory pool and the recovered body's (smaller, TensorOps) footprint is the better spill
            let twin_fits = self.modules[k.module].kernel_abi(&twin).map(|t| t.static_smem + smem <= self.info.max_tg_mem).unwrap_or(false);
            let too_big = kp.abi.static_smem + effective_smem(&kp, smem) > self.info.max_tg_mem && twin_fits;
            let wrong_shape = kp.abi.staging_threads != 0 && (block[0] != kp.abi.staging_threads || block[1] != 1 || block[2] != 1);
            let too_small = !crate::module::retile_smem_fits(&kp.abi, smem);
            let assumption = if kp.abi.assume.is_empty() { Ok(()) } else { crate::assume::check(&kp.abi.assume, &params, &kp.abi.params) };
            if too_many || too_big || wrong_shape || too_small || assumption.is_err() {
                let dev = &self.dev;
                kp = match self.modules[k.module].pipeline(dev, &twin) { Ok(p) => p, Err(e) => { eprintln!("[MVCC] pipeline creation failed for {}: {}", twin, e); return INVALID_KERNEL_IMAGE; } };
                if std::env::var("MVCC_TENSOR_DIAG").is_ok() && self.tensor_fallback_reported.insert(k.name.clone()) {
                    let why = if too_many { "block larger than the recovered body supports".to_string() } else if too_big { "recovered body does not fit in threadgroup memory with this dynamic shared size".to_string() } else if wrong_shape { "block shape differs from the one the staging elimination was proven for".to_string() } else if too_small { "dynamic shared request below the retiled layout's cut".to_string() } else { format!("launch assumption failed: {}", assumption.unwrap_err()) };
                    eprintln!("[MVCC] tensor recovery: {} launched as exact twin ({})", k.name, why);
                }
            } else if scale > 1 { block[0] *= scale as u32; threads *= scale; }
        }
        let smem = effective_smem(&kp, smem);
        let total_smem = kp.abi.static_smem + smem;
        // Beyond threadgroup memory: the spill build keeps the block's shared memory in a slot of a device-memory
        // pool (prelude, "spill build"; README.md). Same source, same semantics; slower staging.
        if total_smem > self.info.max_tg_mem {
            if total_smem > MAX_SPILL_SMEM {
                eprintln!("[MVCC] launch of {} needs {} bytes of shared memory ({} static + {} dynamic); the limit is {}", k.name, total_smem, kp.abi.static_smem, smem, MAX_SPILL_SMEM);
                return INVALID_VALUE;
            }
            let body = kp.abi.name.clone();
            let dev = &self.dev;
            kp = match self.modules[k.module].pipeline_in(dev, &body, true) { Ok(p) => p, Err(e) => { eprintln!("[MVCC] spill pipeline creation failed for {}: {}", body, e); return INVALID_KERNEL_IMAGE; } };
            if self.verbose && self.spill_reported.insert(k.name.clone()) { eprintln!("[MVCC] {}: {} bytes of shared memory ({} static + {} dynamic) exceed threadgroup memory ({}); running from the spill build", k.name, total_smem, kp.abi.static_smem, smem, self.info.max_tg_mem); }
        }
        if threads > kp.max_threads as u64 {
            eprintln!("[MVCC] launch of {} with {} threads/block exceeds the pipeline limit of {} (register pressure); reduce the block size for Apple GPUs", k.name, threads, kp.max_threads);
            return LAUNCH_OUT_OF_RESOURCES;
        }
        let tg_mem = total_smem.max(16);
        if let Some(sess) = self.streams[&sid].capture {
            self.capture_push(sess, sid, Op::Kernel { kp, params, grid, block, tg_mem });
            return SUCCESS;
        }
        self.encode_kernel(sid, &kp, &params, grid, block, tg_mem, true);
        SUCCESS
    }
    /// The module's globals buffer (allocated and initialized on first use): (buffer, offset inside it, gpu address).
    fn globals_binding(&mut self, module: usize) -> Option<(Arc<Buffer>, u64, u64)> {
        let g = self.modules[module].abi.globals.clone()?;
        if g.size == 0 { return None; }
        let addr = match self.modules[module].globals_addr {
            Some(a) => a,
            None => {
                let addr = self.malloc(g.size.max(16) as u64).ok()?;
                let (a, off) = self.alloc.resolve_device(addr)?;
                let base = unsafe { a.cpu_ptr().add(off as usize) };
                let bytes: Vec<u8> = (0..g.init.len() / 2).map(|i| u8::from_str_radix(&g.init[2 * i..2 * i + 2], 16).unwrap_or(0)).collect();
                unsafe {
                    std::ptr::write_bytes(base, 0, g.size as usize);
                    std::ptr::copy_nonoverlapping(bytes.as_ptr(), base, bytes.len().min(g.size as usize));
                    for r in &g.relocs { std::ptr::write_unaligned(base.add(r[0] as usize) as *mut u64, addr + r[1] as u64); }
                }
                if self.verbose { eprintln!("[MVCC] module {}: {} bytes of module-scope globals at {:#x} ({} relocations, {} symbols)", self.modules[module].hash, g.size, addr, g.relocs.len(), g.symbols.len()); }
                self.modules[module].globals_addr = Some(addr);
                addr
            }
        };
        let (a, off) = self.alloc.resolve_device(addr)?;
        Some((a.buf.clone(), a.offset + off, addr))
    }

    /// Device address and size of a module-scope symbol registered by __cudaRegisterVar.
    pub fn symbol_address(&mut self, host_sym: usize) -> Option<(u64, u64)> {
        let (module, name) = self.symbols.get(&host_sym).cloned()?;
        let (_, _, base) = self.globals_binding(module)?;
        let g = self.modules[module].abi.globals.as_ref()?;
        let s = g.symbols.iter().find(|s| s.name == name)?;
        Some((base + s.offset as u64, s.size as u64))
    }
    pub fn register_symbol(&mut self, module: usize, host_sym: usize, name: &str) -> Result<(), String> {
        let known = self.modules[module].abi.globals.as_ref().map(|g| g.symbols.iter().any(|s| s.name == name)).unwrap_or(false);
        if !known { return Err(format!("module-scope variable {} not found in module {}", name, self.modules[module].hash)); }
        self.symbols.insert(host_sym, (module, name.to_string()));
        Ok(())
    }

    /// Encode one kernel dispatch on the stream's open encoder. `barrier`: order the next dispatch after this one
    /// (CUDA stream order); graph replay passes false inside a level of independent ops.
    fn encode_kernel(&mut self, sid: u64, kp: &KernelPipeline, params: &[u8], grid: [u32; 3], block: [u32; 3], tg_mem: u32, barrier: bool) {
        for &off in &kp.abi.device_graph_launch_exec_offsets {
            let o = off as usize;
            if o + 8 > params.len() { continue; }
            let exec = u64::from_le_bytes(params[o..o + 8].try_into().unwrap());
            if exec == 0 { continue; }
            if self.graph_launch(exec, sid) != SUCCESS {
                if self.failed.is_none() { self.failed = Some("device-side graph launch lowered to host replay".into()); }
                return;
            }
        }
        let globals = if kp.abi.globals { self.globals_binding(kp.module) } else { None };
        if kp.abi.zero_page && self.zero_page.is_none() {
            let b = self.dev.new_buffer(4096).expect("zero page allocation");
            unsafe { std::ptr::write_bytes(b.contents(), 0, 4096); }
            self.zero_page = Some(b);
        }
        if kp.abi.lock_page && self.lock_page.is_none() {
            let b = self.dev.new_buffer(65536).expect("lock page allocation");
            unsafe { std::ptr::write_bytes(b.contents(), 0, 65536); }
            self.lock_page = Some(b);
        }
        let zero = if kp.abi.zero_page { self.zero_page.take() } else { None };
        let locks = if kp.abi.lock_page { self.lock_page.take() } else { None };
        let pool = if kp.spill { Some(self.spill_pool(tg_mem)) } else { None };
        let enc = self.encoder(sid);
        enc.set_pipeline(&kp.pipeline);
        enc.set_bytes(params, 0);
        if let Some(z) = &zero { enc.set_buffer(z, 0, 1); }
        if let Some(l) = &locks { enc.set_buffer(l, 0, 4); }
        if let Some((b, off, _)) = &globals { enc.set_buffer(b, *off, 2); enc.set_buffer(b, *off, 3); }
        if let Some(p) = &pool { enc.set_buffer(p, 0, 5); } else { enc.set_tg_mem(tg_mem, 0); }
        enc.dispatch(grid, block);
        if barrier { enc.barrier(); }  // CUDA stream order: the next kernel sees this one's writes
        if zero.is_some() { self.zero_page = zero; }
        if locks.is_some() { self.lock_page = locks; }
        self.bump_ops(sid);
    }

    /// The spill pool for launches needing `smem` bytes of shared memory per block: header (slot count, stride,
    /// data offset), one counter per slot at byte 4096, then the slots. 16 slots per GPU core: more than the
    /// resident blocks of any kernel that needs > 32 KB; a block that finds every slot taken waits for one.
    fn spill_pool(&mut self, smem: u32) -> Arc<Buffer> {
        let stride = (smem + 32767) / 32768 * 32768;
        if let Some((_, b)) = self.spill_pools.iter().find(|(s, _)| *s >= stride) { return b.clone(); }
        let slots: u32 = (self.info.core_count * 16).max(128);
        let data_off = (4096 + slots * 4 + 4095) / 4096 * 4096;
        let bytes = data_off as u64 + slots as u64 * stride as u64;
        let b = self.dev.new_buffer(bytes).expect("shared-memory spill pool allocation");
        let base = b.gpu_address();
        let mut straddling = 0u32;
        unsafe {
            std::ptr::write_bytes(b.contents(), 0, data_off as usize);
            let h = b.contents() as *mut u32;
            *h = slots; *h.add(1) = stride; *h.add(2) = data_off;
            // kernels rebuild 32-bit PTX shared addresses from a slot's high 32 bits (prelude __MVCC_SMEM_ADDR): a
            // slot crossing a 4 GiB boundary is marked permanently taken
            for i in 0..slots {
                let lo = base + data_off as u64 + i as u64 * stride as u64;
                if (lo >> 32) != ((lo + stride as u64 - 1) >> 32) { *h.add(1024 + i as usize) = 0x8000_0000; straddling += 1; }
            }
        }
        if self.verbose { eprintln!("[MVCC] shared-memory spill pool: {} slots x {} bytes ({} MiB){}", slots, stride, bytes >> 20, if straddling > 0 { format!(", {} unusable (4 GiB boundary)", straddling) } else { String::new() }); }
        let b = Arc::new(b);
        self.spill_pools.push((stride, b.clone()));
        b
    }

    pub fn occupancy_blocks(&mut self, host_fn: usize, block: u32, dyn_smem: u32) -> Result<i32, CudaError> {
        let (_, kp) = self.kernel_pipeline(host_fn)?;
        if block == 0 || block > kp.max_threads { return Ok(0); }
        // Apple GPU core: up to 2048 threads (64 simdgroups) resident per core when register use permits;
        // the pipeline's maxTotalThreadsPerThreadgroup already reflects register pressure (1024 = <= 64 regs/thread).
        let threads_per_core: u32 = if kp.max_threads >= 1024 { 2048 } else { kp.max_threads * 2 };
        let by_threads = threads_per_core / block;
        let smem = kp.abi.static_smem + dyn_smem;
        // spilled shared memory (beyond threadgroup memory) lives in device memory: it does not bound residency
        let by_smem = if smem == 0 || smem > self.info.max_tg_mem { 32 } else { self.info.max_tg_mem / smem };
        Ok(by_threads.min(by_smem).min(32) as i32)
    }

    // ------------------------------------------------------------------ graphs

    pub fn begin_capture(&mut self, sid: u64, mode: i32) -> CudaError {
        if self.streams[&sid].capture.is_some() { return ILLEGAL_STATE; }
        if sid == STREAM_LEGACY { return STREAM_CAPTURE_UNSUPPORTED; }
        // work already queued stays queued; capture records from here on
        self.commit(sid);
        let id = self.next_id; self.next_id += 1;
        let mut streams = HashSet::new(); streams.insert(sid);
        self.captures.insert(id, Capture { origin: sid, streams, ops: Vec::new(), levels: Vec::new(), last: HashMap::new(), join: HashMap::new(), mode });
        self.streams.get_mut(&sid).unwrap().capture = Some(id);
        SUCCESS
    }
    pub fn end_capture(&mut self, sid: u64) -> Result<u64, CudaError> {
        let Some(sess) = self.streams[&sid].capture else { return Err(STREAM_CAPTURE_UNMATCHED) };
        let cap = self.captures.remove(&sess).unwrap();
        if cap.origin != sid {
            self.captures.insert(sess, cap);
            return Err(STREAM_CAPTURE_UNMATCHED);
        }
        for s in &cap.streams { if let Some(st) = self.streams.get_mut(s) { st.capture = None; } }
        for e in self.events.values_mut() { if e.captured.map(|(s, _)| s == sess).unwrap_or(false) { e.captured = None; } }
        let id = self.next_id; self.next_id += 1;
        if self.verbose {
            let top = cap.levels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
            let mut width = vec![0u32; top as usize];
            for &l in &cap.levels { width[l as usize] += 1; }
            let multi = width.iter().filter(|&&w| w > 1).count();
            eprintln!("[MVCC] graph captured: {} ops on {} streams, {} levels ({} with >1 op, widest {})", cap.ops.len(), cap.streams.len(), top, multi, width.iter().copied().max().unwrap_or(0));
        }
        self.graphs.insert(id, Graph { ops: cap.ops, levels: cap.levels });
        Ok(id)
    }
    pub fn capture_status(&self, sid: u64) -> (i32, u64) {
        match self.streams[&sid].capture { Some(s) => (1, s), None => (0, 0) }
    }
    pub fn instantiate(&mut self, graph: u64) -> Result<u64, CudaError> {
        let Some(g) = self.graphs.get(&graph) else { return Err(INVALID_RESOURCE_HANDLE) };
        let id = self.next_id; self.next_id += 1;
        self.execs.insert(id, GraphExec { ops: g.ops.clone(), levels: g.levels.clone(), launches: 0 });
        Ok(id)
    }
    pub fn graph_node_handle(container: u64, op_index: usize) -> *mut std::ffi::c_void {
        ((container << 32) | (op_index as u64 + 1)) as *mut std::ffi::c_void
    }
    fn graph_node_index(node: *mut std::ffi::c_void) -> Option<usize> {
        let v = node as u64;
        if v == 0 { return None; }
        let lo = (v & 0xffff_ffff) as u32;
        if lo == 0 { return None; }
        Some((lo - 1) as usize)
    }
    fn op_graph_node_type(op: &Op) -> i32 {
        match op {
            Op::Kernel { .. } => 0,
            Op::Copy { .. } | Op::HostIn { .. } | Op::HostOut { .. } => 1,
            Op::Fill { .. } => 2,
            Op::HostFn { .. } => 3,
            Op::EventRecord { .. } => 7,
        }
    }
    pub fn graph_node_get_type(&self, node: *mut std::ffi::c_void, out: *mut i32) -> CudaError {
        if out.is_null() { return INVALID_VALUE; }
        let idx = match Self::graph_node_index(node) { Some(i) => i, None => return INVALID_VALUE };
        let container = (node as u64) >> 32;
        let ty = self.graphs.get(&container).and_then(|g| g.ops.get(idx).map(Self::op_graph_node_type))
            .or_else(|| self.execs.get(&container).and_then(|e| e.ops.get(idx).map(Self::op_graph_node_type)));
        match ty {
            Some(t) => { unsafe { *out = t }; SUCCESS }
            None => INVALID_VALUE,
        }
    }
    pub fn graph_get_nodes(&self, graph: u64, nodes: *mut *mut std::ffi::c_void, n: *mut usize) -> CudaError {
        let Some(gr) = self.graphs.get(&graph) else { return INVALID_RESOURCE_HANDLE };
        if n.is_null() { return SUCCESS; }
        let count = gr.ops.len();
        unsafe {
            if nodes.is_null() {
                *n = count;
            } else {
                for i in 0..count {
                    *nodes.add(i) = Self::graph_node_handle(graph, i);
                }
                *n = count;
            }
        }
        SUCCESS
    }
    pub fn graph_exec_kernel_node_set_params(&mut self, exec: u64, node: *mut std::ffi::c_void, func: *const std::ffi::c_void, grid: [u32; 3], block: [u32; 3], smem: u32, args: *const *const std::ffi::c_void) -> CudaError {
        let idx = match Self::graph_node_index(node) { Some(i) => i, None => return INVALID_VALUE };
        let op_count = match self.execs.get(&exec) { Some(e) => e.ops.len(), None => return INVALID_RESOURCE_HANDLE };
        if idx >= op_count { return INVALID_VALUE; }
        let new_kp = if func.is_null() {
            None
        } else {
            Some(match self.kernel_pipeline(func as usize) { Ok((_, kp)) => kp, Err(e) => return e })
        };
        let Some(e) = self.execs.get_mut(&exec) else { return INVALID_RESOURCE_HANDLE };
        let Op::Kernel { kp, params, grid: g, block: b, tg_mem } = &mut e.ops[idx] else { return INVALID_VALUE };
        if let Some(nkp) = new_kp { *kp = nkp; }
        *g = grid;
        *b = block;
        let total_smem = kp.abi.static_smem + smem;
        *tg_mem = total_smem.max(16);
        if args.is_null() { return SUCCESS; }
        for (i, p) in kp.abi.params.iter().enumerate() {
            unsafe {
                let src = *args.add(i) as *const u8;
                if src.is_null() { return INVALID_VALUE; }
                std::ptr::copy_nonoverlapping(src, params.as_mut_ptr().add(p.offset as usize), p.size as usize);
            }
        }
        SUCCESS
    }
    pub fn graph_launch(&mut self, exec: u64, sid: u64) -> CudaError {
        if self.failed.is_some() { return ILLEGAL_ADDRESS; }
        let Some(e) = self.execs.get_mut(&exec) else { return INVALID_RESOURCE_HANDLE };
        e.launches += 1;
        let ops = e.ops.clone();
        let levels = e.levels.clone();
        if let Some(sess) = self.streams[&sid].capture {
            // graph launched into another capture: inline it above everything the launching stream has recorded,
            // keeping its internal levels; the stream's next op goes above all of it
            let base = self.capture_next_level(sess, sid);
            let top = levels.iter().copied().max().unwrap_or(0);
            let c = self.captures.get_mut(&sess).unwrap();
            c.ops.extend(ops);
            c.levels.extend(levels.iter().map(|l| base + l));
            c.last.remove(&sid);
            c.join.insert(sid, base + top + 1);
            return SUCCESS;
        }
        // Level order: ops of one level are independent (captured on streams with no event ordering between them)
        // and dispatch without barriers, so they may overlap on the GPU; a barrier separates levels.
        let mut order: Vec<usize> = (0..ops.len()).collect();
        order.sort_by_key(|&i| levels[i]);
        let mut cur_level = u32::MAX;
        for &i in &order {
            let op = &ops[i];
            if levels[i] != cur_level {
                if cur_level != u32::MAX { self.encoder(sid).barrier(); }
                cur_level = levels[i];
            }
            match op {
                Op::Kernel { kp, params, grid, block, tg_mem } => self.encode_kernel(sid, kp, params, *grid, *block, *tg_mem, false),
                Op::Copy { src, soff, dst, doff, n } => { let b = self.blit(sid); b.copy(src, *soff, dst, *doff, *n); self.end_blit(sid, b); }
                Op::Fill { dst, off, n, value } => { let b = self.blit(sid); b.fill(dst, *off, *n, *value); self.end_blit(sid, b); }
                Op::HostIn { host, dst, doff, n } => {
                    let Some(stg) = self.dev.new_buffer(*n) else { return MEMORY_ALLOCATION };
                    unsafe { std::ptr::copy(*host as *const u8, stg.contents(), *n as usize) };
                    self.res.add(&stg); self.res.commit();
                    let stg = Arc::new(stg);
                    let b = self.blit(sid); b.copy(&stg, 0, dst, *doff, *n); self.end_blit(sid, b);
                    self.keepalive(sid, stg);
                }
                Op::HostOut { src, soff, host, n } => {
                    let Some(stg) = self.dev.new_buffer(*n) else { return MEMORY_ALLOCATION };
                    self.res.add(&stg); self.res.commit();
                    let stg = Arc::new(stg);
                    let b = self.blit(sid); b.copy(src, *soff, &stg, 0, *n); self.end_blit(sid, b);
                    self.keepalive(sid, stg.clone());
                    // copy out on completion, then let later work proceed
                    let ctx = Box::new(HostOutCtx { stg, host: *host, n: *n });
                    self.commit_with_handler(sid, host_out_trampoline, Box::into_raw(ctx) as *mut c_void);
                }
                Op::HostFn { f, data } => { let rc = self.launch_host_func(sid, *f, *data); if rc != SUCCESS { return rc; } }
                Op::EventRecord { ev } => { let rc = self.record_event(*ev, sid); if rc != SUCCESS { return rc; } }
            }
        }
        // work after the graph on this stream is ordered after its last level
        if !ops.is_empty() { self.encoder(sid).barrier(); }
        SUCCESS
    }
    pub fn graph_upload(&mut self, exec: u64) -> CudaError {
        let Some(e) = self.execs.get(&exec) else { return INVALID_RESOURCE_HANDLE };
        let _ = e;  // pipelines were built at capture time (launch resolves them); nothing else to warm
        SUCCESS
    }

    fn graph_op_dot_label(op: &Op, verbose: bool) -> String {
        match op {
            Op::Kernel { kp, grid, block, tg_mem, .. } => {
                if verbose {
                    format!(
                        "kernel\\n{}\\ngrid=[{},{},{}] block=[{},{},{}] smem={}",
                        kp.abi.name, grid[0], grid[1], grid[2], block[0], block[1], block[2], tg_mem
                    )
                } else {
                    format!("kernel {}", kp.abi.name)
                }
            }
            Op::Copy { n, .. } => format!("memcpy {}B", n),
            Op::HostIn { n, .. } | Op::HostOut { n, .. } => format!("host_memcpy {}B", n),
            Op::Fill { n, value, .. } => format!("memset {}B val={}", n, value),
            Op::HostFn { .. } => "host_fn".into(),
            Op::EventRecord { ev, .. } => format!("event_record {}", ev),
        }
    }

    /// Write a Graphviz DOT snapshot of a captured graph (cudaGraphDebugDotPrint).
    pub fn graph_debug_dot_print(&self, graph: u64, path: &str, flags: u32) -> CudaError {
        let Some(g) = self.graphs.get(&graph) else { return INVALID_RESOURCE_HANDLE };
        let verbose = (flags & 0x1) != 0;
        let mut out = String::from("digraph mvcc_cuda_graph {\n  rankdir=LR;\n");
        for (i, op) in g.ops.iter().enumerate() {
            let label = Self::graph_op_dot_label(op, verbose);
            let level = g.levels.get(i).copied().unwrap_or(0);
            out.push_str(&format!(
                "  n{i} [label=\"{}\\nlevel={}\"];\n",
                label.replace('\\', "\\\\").replace('"', "\\\""),
                level
            ));
        }
        for i in 1..g.ops.len() {
            out.push_str(&format!("  n{} -> n{};\n", i - 1, i));
        }
        out.push_str("}\n");
        if std::fs::write(path, out).is_err() { UNKNOWN } else { SUCCESS }
    }

    pub fn flush_caches(&mut self) { for m in &mut self.modules { m.flush_cache(); } }
}

struct HostFnCtx { f: usize, data: usize, fence: Arc<SharedEvent>, value: u64 }
extern "C" fn host_fn_trampoline(ctx: *mut c_void, _status: i32) {
    let c = unsafe { Box::from_raw(ctx as *mut HostFnCtx) };
    let f: extern "C" fn(*mut c_void) = unsafe { std::mem::transmute(c.f) };
    f(c.data as *mut c_void);
    c.fence.set_value(c.value);
}
struct HostOutCtx { stg: Arc<Buffer>, host: usize, n: u64 }
extern "C" fn host_out_trampoline(ctx: *mut c_void, status: i32) {
    let c = unsafe { Box::from_raw(ctx as *mut HostOutCtx) };
    if status == 2 { unsafe { std::ptr::copy(c.stg.contents(), c.host as *mut u8, c.n as usize) }; }
}
