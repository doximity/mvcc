//! Device memory: suballocation from large shared MTLBuffers plus a pointer registry.
//!
//! Device pointers handed to user code are GPU virtual addresses (`MTLBuffer.gpuAddress + offset`), so kernels can
//! use them directly as `device T*`. The CPU side of the same memory is reached through the registry
//! (`resolve` → buffer contents), which is what cudaMemcpy uses. Host (pinned) allocations live in their own
//! registry keyed by CPU address.
use crate::metal::{Buffer, Device, Residency};
use std::collections::BTreeMap;
use std::sync::Arc;

pub const ALIGN: u64 = 256;
const CHUNK_SIZE: u64 = 256 << 20;   // suballocation chunk
const DEDICATED_MIN: u64 = 32 << 20; // allocations at least this big get their own buffer

pub struct Chunk {
    pub buf: Arc<Buffer>,
    pub gpu_base: u64,
    pub size: u64,
    /// offset -> length of free ranges
    free: BTreeMap<u64, u64>,
    pub used: u64,
}

impl Chunk {
    fn new(buf: Buffer) -> Chunk {
        let size = buf.len();
        let mut free = BTreeMap::new();
        free.insert(0, size);
        Chunk { gpu_base: buf.gpu_address(), buf: Arc::new(buf), size, free, used: 0 }
    }
    fn alloc(&mut self, n: u64) -> Option<u64> {
        // first fit
        let mut pick = None;
        for (&off, &len) in &self.free {
            if len >= n { pick = Some((off, len)); break; }
        }
        let (off, len) = pick?;
        self.free.remove(&off);
        if len > n { self.free.insert(off + n, len - n); }
        self.used += n;
        Some(off)
    }
    fn release(&mut self, off: u64, n: u64) {
        self.used -= n;
        let mut start = off;
        let mut len = n;
        // merge with previous
        if let Some((&poff, &plen)) = self.free.range(..off).next_back() {
            if poff + plen == off { self.free.remove(&poff); start = poff; len += plen; }
        }
        // merge with next
        if let Some(&nlen) = self.free.get(&(off + n)) {
            self.free.remove(&(off + n));
            len += nlen;
        }
        self.free.insert(start, len);
    }
}

#[derive(Clone)]
pub struct Allocation {
    pub buf: Arc<Buffer>,
    /// offset of this allocation inside `buf`
    pub offset: u64,
    pub size: u64,
    pub gpu_addr: u64,
    /// index into `chunks`, or None for a dedicated buffer
    chunk: Option<usize>,
}

impl Allocation {
    pub fn cpu_ptr(&self) -> *mut u8 { unsafe { self.buf.contents().add(self.offset as usize) } }
}

pub struct Allocator {
    chunks: Vec<Chunk>,
    /// gpu address -> allocation (device pointers)
    device: BTreeMap<u64, Allocation>,
    /// cpu address -> (buffer, size) for cudaHostAlloc / registered host memory
    host: BTreeMap<usize, Allocation>,
    pub bytes_in_use: u64,
}

impl Allocator {
    pub fn new() -> Allocator { Allocator { chunks: Vec::new(), device: BTreeMap::new(), host: BTreeMap::new(), bytes_in_use: 0 } }

    pub fn alloc_device(&mut self, dev: &Device, res: &Residency, size: u64) -> Option<u64> {
        let n = ((size.max(1) + ALIGN - 1) / ALIGN) * ALIGN;
        if n >= DEDICATED_MIN {
            let buf = dev.new_buffer(n)?;
            res.add(&buf); res.commit();
            let a = Allocation { gpu_addr: buf.gpu_address(), buf: Arc::new(buf), offset: 0, size: n, chunk: None };
            let addr = a.gpu_addr;
            self.device.insert(addr, a);
            self.bytes_in_use += n;
            return Some(addr);
        }
        for (ci, c) in self.chunks.iter_mut().enumerate() {
            if let Some(off) = c.alloc(n) {
                let a = Allocation { buf: c.buf.clone(), offset: off, size: n, gpu_addr: c.gpu_base + off, chunk: Some(ci) };
                let addr = a.gpu_addr;
                self.device.insert(addr, a);
                self.bytes_in_use += n;
                return Some(addr);
            }
        }
        let buf = dev.new_buffer(CHUNK_SIZE)?;
        res.add(&buf); res.commit();
        let mut c = Chunk::new(buf);
        let off = c.alloc(n)?;
        let ci = self.chunks.len();
        let a = Allocation { buf: c.buf.clone(), offset: off, size: n, gpu_addr: c.gpu_base + off, chunk: Some(ci) };
        self.chunks.push(c);
        let addr = a.gpu_addr;
        self.device.insert(addr, a);
        self.bytes_in_use += n;
        Some(addr)
    }

    /// Frees an allocation; returns the released allocation (caller decides whether to sync first).
    pub fn free_device(&mut self, res: &Residency, addr: u64) -> Option<Allocation> {
        let a = self.device.remove(&addr)?;
        self.bytes_in_use -= a.size;
        match a.chunk {
            Some(ci) => self.chunks[ci].release(a.offset, a.size),
            None => { res.remove(&a.buf); res.commit(); }
        }
        Some(a)
    }

    pub fn alloc_host(&mut self, dev: &Device, res: &Residency, size: u64) -> Option<*mut u8> {
        let n = ((size.max(1) + 4095) / 4096) * 4096;
        let buf = dev.new_buffer(n)?;
        res.add(&buf); res.commit();
        let p = buf.contents();
        let a = Allocation { gpu_addr: buf.gpu_address(), buf: Arc::new(buf), offset: 0, size: n, chunk: None };
        self.host.insert(p as usize, a);
        Some(p)
    }
    pub fn free_host(&mut self, res: &Residency, p: usize) -> Option<Allocation> {
        let a = self.host.remove(&p)?;
        res.remove(&a.buf); res.commit();
        Some(a)
    }
    pub fn register_host(&mut self, dev: &Device, res: &Residency, p: *mut u8, size: u64) -> bool {
        // wrap page-aligned ranges without copying; unaligned registrations are accepted as no-ops (copies fall back to CPU memcpy)
        let start = (p as usize) & !4095;
        let end = ((p as usize) + size as usize + 4095) & !4095;
        if let Some(buf) = dev.wrap_host(start as *mut _, (end - start) as u64) {
            res.add(&buf); res.commit();
            let a = Allocation { gpu_addr: buf.gpu_address(), buf: Arc::new(buf), offset: 0, size: (end - start) as u64, chunk: None };
            self.host.insert(start, a);
            true
        } else { false }
    }
    pub fn unregister_host(&mut self, res: &Residency, p: usize) -> Option<Allocation> {
        let key = *self.host.range(..=p).next_back().map(|(k, _)| k)?;
        let a = self.host.remove(&key)?;
        res.remove(&a.buf); res.commit();
        Some(a)
    }

    /// Resolve a device pointer to (allocation, byte offset within the allocation).
    pub fn resolve_device(&self, addr: u64) -> Option<(&Allocation, u64)> {
        let (_, a) = self.device.range(..=addr).next_back()?;
        if addr < a.gpu_addr + a.size { Some((a, addr - a.gpu_addr)) } else { None }
    }
    /// Resolve a host pointer that belongs to a pinned/registered allocation.
    pub fn resolve_host(&self, p: usize) -> Option<(&Allocation, u64)> {
        let (&base, a) = self.host.range(..=p).next_back()?;
        if p < base + a.size as usize { Some((a, (p - base) as u64)) } else { None }
    }
    pub fn device_ptr_count(&self) -> usize { self.device.len() }
}

/// A resolved memory location usable by both CPU and GPU.
#[derive(Clone)]
pub struct Loc {
    pub buf: Arc<Buffer>,
    pub offset: u64,   // offset inside buf
    pub cpu: *mut u8,
}
impl Loc {
    pub fn from_alloc(a: &Allocation, off: u64) -> Loc {
        Loc { buf: a.buf.clone(), offset: a.offset + off, cpu: unsafe { a.cpu_ptr().add(off as usize) } }
    }
}
