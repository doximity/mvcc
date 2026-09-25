//! The C ABI: CUDA runtime API entry points. Thin argument marshaling over `Runtime`.
#![allow(non_snake_case, clippy::missing_safety_doc)]
use crate::error::{self as err, *};
use crate::runtime::{Runtime, STREAM_LEGACY};
use parking_lot::{Mutex, MutexGuard};
use std::cell::RefCell;
use std::ffi::{c_char, c_int, c_void, CStr};
use std::sync::OnceLock;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct Dim3 { pub x: u32, pub y: u32, pub z: u32 }

static RT: OnceLock<Result<Mutex<Runtime>, CudaError>> = OnceLock::new();
static STATS_LAUNCHES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

fn rt() -> Result<MutexGuard<'static, Runtime>, CudaError> {
    match RT.get_or_init(|| Runtime::new().map(Mutex::new)) {
        Ok(m) => { let mut g = m.lock(); drain_pending(&mut g); Ok(g) }
        Err(e) => Err(*e),
    }
}
/// Run `f` with the runtime; converts init failure into an error code and records it as last error.
fn with<F: FnOnce(&mut Runtime) -> CudaError>(f: F) -> CudaError {
    match rt() { Ok(mut r) => record(f(&mut r)), Err(e) => record(e) }
}
fn sid(r: &Runtime, h: *const c_void) -> Result<u64, CudaError> { r.stream_id(h as usize).ok_or(INVALID_RESOURCE_HANDLE) }

// ------------------------------------------------------------------ device

/// Logical devices. The one Metal GPU is device 0; MVCC_DEVICES=N (1..=16) presents N logical devices that all
/// map to it, so a multi-GPU program (ranks on devices 0..world-1, one stream each, peer access between every
/// pair) runs its real multi-device path on one chip: cudaSetDevice selects per thread, every device shares the
/// address space (peer copies are plain copies), and the streams, events and captures are the single runtime's.
fn device_count() -> c_int {
    static N: OnceLock<c_int> = OnceLock::new();
    *N.get_or_init(|| std::env::var("MVCC_DEVICES").ok().and_then(|v| v.trim().parse::<c_int>().ok()).map(|n| n.clamp(1, 16)).unwrap_or(1))
}
thread_local! { static CURRENT_DEVICE: std::cell::Cell<c_int> = const { std::cell::Cell::new(0) }; }
fn valid_device(d: c_int) -> bool { d >= 0 && d < device_count() }

#[no_mangle] pub extern "C" fn cudaGetDeviceCount(count: *mut c_int) -> CudaError {
    if count.is_null() { return record(INVALID_VALUE); }
    match rt() { Ok(_) => { unsafe { *count = device_count() }; SUCCESS } Err(_) => { unsafe { *count = 0 }; record(NO_DEVICE) } }
}
#[no_mangle] pub extern "C" fn cudaGetDevice(device: *mut c_int) -> CudaError { if device.is_null() { return record(INVALID_VALUE); } unsafe { *device = CURRENT_DEVICE.get() }; with(|_| SUCCESS) }
#[no_mangle] pub extern "C" fn cudaSetDevice(device: c_int) -> CudaError { if !valid_device(device) { return record(INVALID_DEVICE); } CURRENT_DEVICE.set(device); with(|_| SUCCESS) }
#[no_mangle] pub extern "C" fn cudaSetDeviceFlags(_flags: u32) -> CudaError { with(|_| SUCCESS) }
#[no_mangle] pub extern "C" fn cudaGetDeviceFlags(flags: *mut u32) -> CudaError { if !flags.is_null() { unsafe { *flags = 0 } } with(|_| SUCCESS) }
#[no_mangle] pub extern "C" fn cudaDeviceReset() -> CudaError { with(|r| r.sync_device()) }
#[no_mangle] pub extern "C" fn cudaDeviceSynchronize() -> CudaError { with(|r| r.sync_device()) }
#[no_mangle] pub extern "C" fn cudaThreadSynchronize() -> CudaError { cudaDeviceSynchronize() }
#[no_mangle] pub extern "C" fn cudaThreadExit() -> CudaError { SUCCESS }
#[no_mangle] pub extern "C" fn cudaDriverGetVersion(v: *mut c_int) -> CudaError { if v.is_null() { return record(INVALID_VALUE); } unsafe { *v = 12060 }; SUCCESS }
#[no_mangle] pub extern "C" fn cudaRuntimeGetVersion(v: *mut c_int) -> CudaError { if v.is_null() { return record(INVALID_VALUE); } unsafe { *v = 12060 }; SUCCESS }
#[no_mangle] pub extern "C" fn cudaDeviceGetStreamPriorityRange(least: *mut c_int, greatest: *mut c_int) -> CudaError {
    if !least.is_null() { unsafe { *least = 0 } }
    if !greatest.is_null() { unsafe { *greatest = -1 } }
    SUCCESS
}
#[no_mangle] pub extern "C" fn cudaDeviceGetPCIBusId(buf: *mut c_char, len: c_int, device: c_int) -> CudaError {
    if !valid_device(device) || buf.is_null() || len < 13 { return record(INVALID_VALUE); }
    let s = format!("0000:{:02x}:00.0\0", device);
    unsafe { std::ptr::copy_nonoverlapping(s.as_ptr(), buf as *mut u8, s.len()) };
    SUCCESS
}
// (logical devices share the one address space: every pair is a peer, and enabling access is a no-op)
#[no_mangle] pub extern "C" fn cudaDeviceCanAccessPeer(can: *mut c_int, d: c_int, p: c_int) -> CudaError {
    if can.is_null() { return record(INVALID_VALUE); }
    if !valid_device(d) || !valid_device(p) { return record(INVALID_DEVICE); }
    unsafe { *can = (d != p) as c_int };
    SUCCESS
}
#[no_mangle] pub extern "C" fn cudaDeviceEnablePeerAccess(p: c_int, _f: u32) -> CudaError {
    if !valid_device(p) { return record(INVALID_DEVICE); }
    if p == CURRENT_DEVICE.get() { return record(INVALID_DEVICE); }
    SUCCESS
}
#[no_mangle] pub extern "C" fn cudaDeviceDisablePeerAccess(p: c_int) -> CudaError { if !valid_device(p) { return record(INVALID_DEVICE); } SUCCESS }
#[no_mangle] pub extern "C" fn cudaDeviceSetLimit(_l: c_int, _v: usize) -> CudaError { SUCCESS }
#[no_mangle] pub extern "C" fn cudaDeviceGetLimit(v: *mut usize, limit: c_int) -> CudaError {
    if v.is_null() { return record(INVALID_VALUE); }
    unsafe { *v = match limit { 0 => 1024, 1 => 1 << 20, 2 => 8192, 3 => 0, 4 => 0, _ => 0 } };
    SUCCESS
}
#[no_mangle] pub extern "C" fn cudaDeviceSetCacheConfig(_c: c_int) -> CudaError { SUCCESS }
#[no_mangle] pub extern "C" fn cudaDeviceGetCacheConfig(c: *mut c_int) -> CudaError { if !c.is_null() { unsafe { *c = 0 } } SUCCESS }
#[no_mangle] pub extern "C" fn cudaDeviceSetSharedMemConfig(_c: c_int) -> CudaError { SUCCESS }
#[no_mangle] pub extern "C" fn cudaChooseDevice(device: *mut c_int, _prop: *const c_void) -> CudaError { if device.is_null() { return record(INVALID_VALUE); } unsafe { *device = 0 }; SUCCESS }
#[no_mangle] pub extern "C" fn cudaDeviceGetDefaultMemPool(pool: *mut *mut c_void, _d: c_int) -> CudaError { if !pool.is_null() { unsafe { *pool = 1 as *mut c_void } } SUCCESS }
#[no_mangle] pub extern "C" fn cudaDeviceSetMemPool(_d: c_int, _p: *mut c_void) -> CudaError { SUCCESS }
#[no_mangle] pub extern "C" fn cudaDeviceGetMemPool(pool: *mut *mut c_void, _d: c_int) -> CudaError { if !pool.is_null() { unsafe { *pool = 1 as *mut c_void } } SUCCESS }
#[no_mangle] pub extern "C" fn cudaMemPoolSetAttribute(_p: *mut c_void, _a: c_int, _v: *mut c_void) -> CudaError { SUCCESS }
#[no_mangle] pub extern "C" fn cudaMemPoolGetAttribute(_p: *mut c_void, _a: c_int, _v: *mut c_void) -> CudaError { record(NOT_SUPPORTED) }

/// Shared memory a block may use without opting in (CUDA's 48 KB); more needs cudaFuncSetAttribute, as on NVIDIA.
/// Launches above the device's threadgroup memory run from the spill build (runtime.rs).
const SMEM_DEFAULT_LIMIT: u32 = 49152;
#[repr(C)]
struct DeviceFacts { name: *const c_char, total_mem: u64, core_count: u32, max_tg_mem: u32, max_threads_per_core: u32, major: u32, minor: u32, registry_id: u64 }
extern "C" { fn mvcc_fill_device_prop(p: *mut c_void, f: *const DeviceFacts); }

#[no_mangle] pub extern "C" fn cudaGetDeviceProperties(prop: *mut c_void, device: c_int) -> CudaError {
    if prop.is_null() { return record(INVALID_VALUE); }
    if !valid_device(device) { return record(INVALID_DEVICE); }
    with(|r| {
        let name = std::ffi::CString::new(r.info.name.clone()).unwrap();
        let f = DeviceFacts { name: name.as_ptr(), total_mem: r.info.total_mem, core_count: r.info.core_count, max_tg_mem: r.info.max_tg_mem, max_threads_per_core: 2048, major: r.info.major, minor: r.info.minor, registry_id: r.info.registry_id };
        unsafe { mvcc_fill_device_prop(prop, &f) };
        SUCCESS
    })
}
#[no_mangle] pub extern "C" fn cudaGetDeviceProperties_v2(prop: *mut c_void, device: c_int) -> CudaError { cudaGetDeviceProperties(prop, device) }

#[no_mangle] pub extern "C" fn cudaDeviceGetAttribute(value: *mut c_int, attr: c_int, device: c_int) -> CudaError {
    if value.is_null() { return record(INVALID_VALUE); }
    if !valid_device(device) { return record(INVALID_DEVICE); }
    with(|r| {
        let v: i64 = match attr {
            1 => 1024, 2 | 3 | 4 => 1024, 5 => 2147483647, 6 | 7 => 65535,
            // 8 MaxSharedMemoryPerBlock, 97 MaxSharedMemoryPerBlockOptin: report the Metal
            // threadgroup limit so launchers pick tiles that fit. Larger CUDA
            // opt-in values would select NVIDIA 48–99 KB configs that then spill to DRAM.
            8 => r.info.max_tg_mem as i64, 9 => 65536, 10 => 32, 11 => 2147483647, 12 => 65536, 13 => 1400000, 14 => 512,
            15 => 1, 16 => r.info.core_count as i64, 17 => 0, 18 => 1, 19 => 1, 20 => 0, 31 => 1, 32 => 0, 33 | 34 | 35 => 0,
            36 => 0, 37 => 0, 38 => 24 << 20, 39 => 2048, 40 => 1, 41 => 1, 50 => 0, 51 => 32,
            75 => r.info.major as i64, 76 => r.info.minor as i64, 78 => 0, 79 | 80 => 1, 81 => (crate::runtime::MAX_SPILL_SMEM + 1024) as i64, 82 => 65536,
            83 => 1, 84 => 0, 85 => 0, 86 => 1, 87 => 0, 88 => 0, 89 => 1, 90 => 1, 91 => 0, 95 => 0, 96 => 0,
            97 => r.info.max_tg_mem as i64, 98 => 0, 99 => 1, 100 => 0, 101 => 0, 106 => 32, 108 => 0, 109 => 0, 111 => 0,
            112 => 0, 113 => 0, 114 => 0, 115 => 0, 116 => 0, 117 => 0, 118 => 0, 119 => 0, 120 => 0, 121 => 0, 124 => 0, 125 => 1,
            _ => return INVALID_VALUE,
        };
        unsafe { *value = v as c_int };
        SUCCESS
    })
}

// ------------------------------------------------------------------ errors

#[no_mangle] pub extern "C" fn cudaGetLastError() -> CudaError { err::take_last() }
#[no_mangle] pub extern "C" fn cudaPeekAtLastError() -> CudaError { err::peek_last() }
#[no_mangle] pub extern "C" fn cudaGetErrorName(e: CudaError) -> *const c_char {
    static NAMES: OnceLock<std::collections::HashMap<i32, std::ffi::CString>> = OnceLock::new();
    let m = NAMES.get_or_init(|| {
        let mut h = std::collections::HashMap::new();
        for e in 0..1000 { h.insert(e, std::ffi::CString::new(err::name(e)).unwrap()); }
        h
    });
    m.get(&e).or_else(|| m.get(&999)).unwrap().as_ptr()
}
#[no_mangle] pub extern "C" fn cudaGetErrorString(e: CudaError) -> *const c_char {
    static STRS: OnceLock<std::collections::HashMap<i32, std::ffi::CString>> = OnceLock::new();
    let m = STRS.get_or_init(|| {
        let mut h = std::collections::HashMap::new();
        for e in 0..1000 { h.insert(e, std::ffi::CString::new(err::string(e)).unwrap()); }
        h
    });
    m.get(&e).or_else(|| m.get(&999)).unwrap().as_ptr()
}

// ------------------------------------------------------------------ memory

#[no_mangle] pub extern "C" fn cudaMalloc(dev_ptr: *mut *mut c_void, size: usize) -> CudaError {
    if dev_ptr.is_null() { return record(INVALID_VALUE); }
    with(|r| match r.malloc(size as u64) { Ok(a) => { unsafe { *dev_ptr = a as *mut c_void }; SUCCESS } Err(e) => { unsafe { *dev_ptr = std::ptr::null_mut() }; e } })
}
#[no_mangle] pub extern "C" fn cudaMallocManaged(dev_ptr: *mut *mut c_void, size: usize, _flags: u32) -> CudaError {
    // No identity mapping between CPU and GPU address spaces: managed memory behaves as pinned host memory whose
    // pointer kernels cannot dereference directly. Report the limitation loudly the first time.
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| eprintln!("[MVCC] cudaMallocManaged: returning pinned host memory; pass it to kernels via cudaHostGetDevicePointer"));
    cudaHostAlloc(dev_ptr, size, 0)
}
#[no_mangle] pub extern "C" fn cudaMallocPitch(dev_ptr: *mut *mut c_void, pitch: *mut usize, width: usize, height: usize) -> CudaError {
    if dev_ptr.is_null() || pitch.is_null() { return record(INVALID_VALUE); }
    let p = (width + 255) & !255;
    unsafe { *pitch = p };
    cudaMalloc(dev_ptr, p * height.max(1))
}
#[no_mangle] pub extern "C" fn cudaMallocAsync(dev_ptr: *mut *mut c_void, size: usize, _stream: *mut c_void) -> CudaError { cudaMalloc(dev_ptr, size) }
#[no_mangle] pub extern "C" fn cudaFreeAsync(dev_ptr: *mut c_void, _stream: *mut c_void) -> CudaError { cudaFree(dev_ptr) }
#[no_mangle] pub extern "C" fn cudaFree(dev_ptr: *mut c_void) -> CudaError { with(|r| r.free(dev_ptr as u64)) }
#[no_mangle] pub extern "C" fn cudaMallocHost(ptr: *mut *mut c_void, size: usize) -> CudaError { cudaHostAlloc(ptr, size, 0) }
#[no_mangle] pub extern "C" fn cudaHostAlloc(ptr: *mut *mut c_void, size: usize, _flags: u32) -> CudaError {
    if ptr.is_null() { return record(INVALID_VALUE); }
    with(|r| match r.host_alloc(size as u64) { Ok(p) => { unsafe { *ptr = p as *mut c_void }; SUCCESS } Err(e) => e })
}
#[no_mangle] pub extern "C" fn cudaFreeHost(ptr: *mut c_void) -> CudaError { with(|r| r.host_free(ptr as usize)) }
#[no_mangle] pub extern "C" fn cudaHostRegister(ptr: *mut c_void, size: usize, _flags: u32) -> CudaError {
    with(|r| { let (dev, res) = (&r.dev, &r.res); r.alloc.register_host(dev, res, ptr as *mut u8, size as u64); SUCCESS })
}
#[no_mangle] pub extern "C" fn cudaHostUnregister(ptr: *mut c_void) -> CudaError { with(|r| { r.alloc.unregister_host(&r.res, ptr as usize); SUCCESS }) }
#[no_mangle] pub extern "C" fn cudaHostGetDevicePointer(pdev: *mut *mut c_void, phost: *mut c_void, _flags: u32) -> CudaError {
    if pdev.is_null() { return record(INVALID_VALUE); }
    with(|r| match r.alloc.resolve_host(phost as usize) { Some((a, off)) => { unsafe { *pdev = (a.gpu_addr + off) as *mut c_void }; SUCCESS } None => INVALID_VALUE })
}
#[no_mangle] pub extern "C" fn cudaHostGetFlags(flags: *mut u32, _phost: *mut c_void) -> CudaError { if !flags.is_null() { unsafe { *flags = 0 } } SUCCESS }
#[no_mangle] pub extern "C" fn cudaMemGetInfo(free: *mut usize, total: *mut usize) -> CudaError {
    with(|r| { let (f, t) = r.mem_info(); if !free.is_null() { unsafe { *free = f as usize } } if !total.is_null() { unsafe { *total = t as usize } } SUCCESS })
}
#[no_mangle] pub extern "C" fn cudaMemcpy(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int) -> CudaError {
    with(|r| r.memcpy(dst as usize, src as usize, n as u64, kind, STREAM_LEGACY, true))
}
#[no_mangle] pub extern "C" fn cudaMemcpyAsync(dst: *mut c_void, src: *const c_void, n: usize, kind: c_int, stream: *mut c_void) -> CudaError {
    with(|r| { let s = match sid(r, stream) { Ok(s) => s, Err(e) => return e }; r.memcpy(dst as usize, src as usize, n as u64, kind, s, false) })
}
#[no_mangle] pub extern "C" fn cudaMemcpyPeer(dst: *mut c_void, _dd: c_int, src: *const c_void, _sd: c_int, n: usize) -> CudaError { cudaMemcpy(dst, src, n, 3) }
#[no_mangle] pub extern "C" fn cudaMemcpyPeerAsync(dst: *mut c_void, _dd: c_int, src: *const c_void, _sd: c_int, n: usize, s: *mut c_void) -> CudaError { cudaMemcpyAsync(dst, src, n, 3, s) }
#[no_mangle] pub extern "C" fn cudaMemcpy2D(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, w: usize, h: usize, kind: c_int) -> CudaError {
    with(|r| r.memcpy2d(dst as usize, dpitch as u64, src as usize, spitch as u64, w as u64, h as u64, kind, STREAM_LEGACY, true))
}
#[no_mangle] pub extern "C" fn cudaMemcpy2DAsync(dst: *mut c_void, dpitch: usize, src: *const c_void, spitch: usize, w: usize, h: usize, kind: c_int, stream: *mut c_void) -> CudaError {
    with(|r| { let s = match sid(r, stream) { Ok(s) => s, Err(e) => return e }; r.memcpy2d(dst as usize, dpitch as u64, src as usize, spitch as u64, w as u64, h as u64, kind, s, false) })
}
// Module-scope __device__ / __constant__ variables live in the module's globals buffer; the host shadow registered
// by __cudaRegisterVar names the symbol. Copies go through the ordinary memcpy paths on the device address.
fn symbol_copy(sym: *const c_void, off: usize, n: usize, kind: c_int, stream: *mut c_void, sync: bool, to_symbol: bool, host_side: *mut c_void) -> CudaError {
    with(|r| {
        let Some((addr, size)) = r.symbol_address(sym as usize) else { return INVALID_SYMBOL };
        if off + n > size as usize { return INVALID_VALUE; }
        let s = match sid(r, stream) { Ok(s) => s, Err(e) => return e };
        // kind is relative to the symbol side being device memory; cudaMemcpyDefault resolves the other pointer
        let k = match kind { 4 => 4, 0 | 1 | 2 | 3 => if to_symbol { if kind == 3 { 3 } else { 1 } } else if kind == 3 { 3 } else { 2 }, _ => return INVALID_MEMCPY_DIRECTION };
        let dev = addr as usize + off;
        if to_symbol { r.memcpy(dev, host_side as usize, n as u64, k, s, sync) } else { r.memcpy(host_side as usize, dev, n as u64, k, s, sync) }
    })
}
#[no_mangle] pub extern "C" fn cudaMemcpyToSymbol(s: *const c_void, src: *const c_void, n: usize, off: usize, k: c_int) -> CudaError { symbol_copy(s, off, n, k, std::ptr::null_mut(), true, true, src as *mut c_void) }
#[no_mangle] pub extern "C" fn cudaMemcpyFromSymbol(d: *mut c_void, s: *const c_void, n: usize, off: usize, k: c_int) -> CudaError { symbol_copy(s, off, n, k, std::ptr::null_mut(), true, false, d) }
#[no_mangle] pub extern "C" fn cudaMemcpyToSymbolAsync(s: *const c_void, src: *const c_void, n: usize, off: usize, k: c_int, st: *mut c_void) -> CudaError { symbol_copy(s, off, n, k, st, false, true, src as *mut c_void) }
#[no_mangle] pub extern "C" fn cudaMemcpyFromSymbolAsync(d: *mut c_void, s: *const c_void, n: usize, off: usize, k: c_int, st: *mut c_void) -> CudaError { symbol_copy(s, off, n, k, st, false, false, d) }
#[no_mangle] pub extern "C" fn cudaGetSymbolAddress(d: *mut *mut c_void, s: *const c_void) -> CudaError {
    if d.is_null() { return record(INVALID_VALUE); }
    with(|r| match r.symbol_address(s as usize) { Some((addr, _)) => { unsafe { *d = addr as *mut c_void }; SUCCESS } None => INVALID_SYMBOL })
}
#[no_mangle] pub extern "C" fn cudaGetSymbolSize(sz: *mut usize, s: *const c_void) -> CudaError {
    if sz.is_null() { return record(INVALID_VALUE); }
    with(|r| match r.symbol_address(s as usize) { Some((_, size)) => { unsafe { *sz = size as usize }; SUCCESS } None => INVALID_SYMBOL })
}
#[no_mangle] pub extern "C" fn cudaMemset(dst: *mut c_void, value: c_int, n: usize) -> CudaError { with(|r| r.memset(dst as usize, value as u8, n as u64, STREAM_LEGACY, true)) }
#[no_mangle] pub extern "C" fn cudaMemsetAsync(dst: *mut c_void, value: c_int, n: usize, stream: *mut c_void) -> CudaError {
    with(|r| { let s = match sid(r, stream) { Ok(s) => s, Err(e) => return e }; r.memset(dst as usize, value as u8, n as u64, s, false) })
}
#[no_mangle] pub extern "C" fn cudaMemset2D(dst: *mut c_void, pitch: usize, value: c_int, w: usize, h: usize) -> CudaError {
    with(|r| { for row in 0..h { let rc = r.memset(dst as usize + row * pitch, value as u8, w as u64, STREAM_LEGACY, true); if rc != SUCCESS { return rc; } } SUCCESS })
}
#[no_mangle] pub extern "C" fn cudaMemset2DAsync(dst: *mut c_void, pitch: usize, value: c_int, w: usize, h: usize, stream: *mut c_void) -> CudaError {
    with(|r| { let s = match sid(r, stream) { Ok(s) => s, Err(e) => return e }; for row in 0..h { let rc = r.memset(dst as usize + row * pitch, value as u8, w as u64, s, false); if rc != SUCCESS { return rc; } } SUCCESS })
}
#[no_mangle] pub extern "C" fn cudaMemPrefetchAsync(_p: *const c_void, _n: usize, _d: c_int, _s: *mut c_void) -> CudaError { SUCCESS }
#[no_mangle] pub extern "C" fn cudaMemAdvise(_p: *const c_void, _n: usize, _a: c_int, _d: c_int) -> CudaError { SUCCESS }
#[repr(C)] pub struct PointerAttributes { ty: c_int, device: c_int, device_pointer: *mut c_void, host_pointer: *mut c_void }
#[no_mangle] pub extern "C" fn cudaPointerGetAttributes(attr: *mut PointerAttributes, ptr: *const c_void) -> CudaError {
    if attr.is_null() { return record(INVALID_VALUE); }
    with(|r| {
        let (ty, dp, hp) = r.pointer_kind(ptr as usize);
        unsafe { *attr = PointerAttributes { ty, device: 0, device_pointer: dp as *mut c_void, host_pointer: hp as *mut c_void } };
        SUCCESS
    })
}

// ------------------------------------------------------------------ streams

#[no_mangle] pub extern "C" fn cudaStreamCreate(p: *mut *mut c_void) -> CudaError { cudaStreamCreateWithPriority(p, 0, 0) }
#[no_mangle] pub extern "C" fn cudaStreamCreateWithFlags(p: *mut *mut c_void, flags: u32) -> CudaError { cudaStreamCreateWithPriority(p, flags, 0) }
#[no_mangle] pub extern "C" fn cudaStreamCreateWithPriority(p: *mut *mut c_void, flags: u32, priority: c_int) -> CudaError {
    if p.is_null() { return record(INVALID_VALUE); }
    with(|r| { let id = r.create_stream(flags & 1 != 0, priority); unsafe { *p = id as *mut c_void }; SUCCESS })
}
#[no_mangle] pub extern "C" fn cudaStreamDestroy(s: *mut c_void) -> CudaError { with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(e) => return e }; r.destroy_stream(s) }) }
#[no_mangle] pub extern "C" fn cudaStreamSynchronize(s: *mut c_void) -> CudaError { with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(e) => return e }; r.sync_stream(s) }) }
#[no_mangle] pub extern "C" fn cudaStreamQuery(s: *mut c_void) -> CudaError {
    // NOT_READY is not an error for last-error purposes
    match rt() { Ok(mut r) => { let s = match sid(&r, s) { Ok(s) => s, Err(e) => return record(e) }; let rc = r.query_stream(s); if rc == NOT_READY { rc } else { record(rc) } } Err(e) => record(e) }
}
#[no_mangle] pub extern "C" fn cudaStreamWaitEvent(s: *mut c_void, ev: *mut c_void, _flags: u32) -> CudaError {
    with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(e) => return e }; r.stream_wait_event(s, ev as u64) })
}
#[no_mangle] pub extern "C" fn cudaLaunchHostFunc(s: *mut c_void, f: *const c_void, data: *mut c_void) -> CudaError {
    if f.is_null() { return record(INVALID_VALUE); }
    with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(e) => return e }; r.launch_host_func(s, f as usize, data as usize) })
}
struct CallbackCtx { cb: usize, stream: *mut c_void, data: *mut c_void }
extern "C" fn stream_callback_trampoline(p: *mut c_void) {
    let c = unsafe { Box::from_raw(p as *mut CallbackCtx) };
    let f: extern "C" fn(*mut c_void, CudaError, *mut c_void) = unsafe { std::mem::transmute(c.cb) };
    f(c.stream, SUCCESS, c.data);
}
#[no_mangle] pub extern "C" fn cudaStreamAddCallback(s: *mut c_void, cb: *const c_void, data: *mut c_void, _flags: u32) -> CudaError {
    if cb.is_null() { return record(INVALID_VALUE); }
    let ctx = Box::into_raw(Box::new(CallbackCtx { cb: cb as usize, stream: s, data }));
    cudaLaunchHostFunc(s, stream_callback_trampoline as *const c_void, ctx as *mut c_void)
}
#[no_mangle] pub extern "C" fn cudaStreamGetFlags(s: *mut c_void, flags: *mut u32) -> CudaError {
    with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(e) => return e }; if !flags.is_null() { unsafe { *flags = r.stream(s).non_blocking as u32 } } SUCCESS })
}
#[no_mangle] pub extern "C" fn cudaStreamGetPriority(s: *mut c_void, p: *mut c_int) -> CudaError {
    with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(e) => return e }; if !p.is_null() { unsafe { *p = r.stream(s).priority } } SUCCESS })
}
#[no_mangle] pub extern "C" fn cudaStreamSetAttribute(_s: *mut c_void, _attr: c_int, _v: *const c_void) -> CudaError { SUCCESS }
#[no_mangle] pub extern "C" fn cudaStreamGetAttribute(_s: *mut c_void, _attr: c_int, _v: *mut c_void) -> CudaError { record(NOT_SUPPORTED) }
#[no_mangle] pub extern "C" fn cudaStreamBeginCapture(s: *mut c_void, mode: c_int) -> CudaError {
    with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(e) => return e }; r.begin_capture(s, mode) })
}
#[no_mangle] pub extern "C" fn cudaStreamEndCapture(s: *mut c_void, graph: *mut *mut c_void) -> CudaError {
    if graph.is_null() { return record(INVALID_VALUE); }
    with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(e) => return e }; match r.end_capture(s) { Ok(g) => { unsafe { *graph = g as *mut c_void }; SUCCESS } Err(e) => { unsafe { *graph = std::ptr::null_mut() }; e } } })
}
#[no_mangle] pub extern "C" fn cudaStreamIsCapturing(s: *mut c_void, status: *mut c_int) -> CudaError {
    if status.is_null() { return record(INVALID_VALUE); }
    with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(e) => return e }; unsafe { *status = r.capture_status(s).0 }; SUCCESS })
}
#[no_mangle] pub extern "C" fn cudaStreamGetCaptureInfo(s: *mut c_void, status: *mut c_int, id: *mut u64, graph: *mut *mut c_void, deps: *mut *const c_void, ndeps: *mut usize) -> CudaError {
    // CUDA 12 signature. The capturing graph and its dependency frontier are not exposed (capture is a private
    // op list until cudaStreamEndCapture): callers get a null graph and an empty dependency set.
    with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(e) => return e }; let (st, i) = r.capture_status(s); if !status.is_null() { unsafe { *status = st } } if !id.is_null() { unsafe { *id = i } } if !graph.is_null() { unsafe { *graph = std::ptr::null_mut() } } if !deps.is_null() { unsafe { *deps = std::ptr::null() } } if !ndeps.is_null() { unsafe { *ndeps = 0 } } SUCCESS })
}
#[no_mangle] pub extern "C" fn cudaThreadExchangeStreamCaptureMode(mode: *mut c_int) -> CudaError { if !mode.is_null() { unsafe { *mode = 0 } } SUCCESS }

// ------------------------------------------------------------------ events

#[no_mangle] pub extern "C" fn cudaEventCreate(ev: *mut *mut c_void) -> CudaError { cudaEventCreateWithFlags(ev, 0) }
#[no_mangle] pub extern "C" fn cudaEventCreateWithFlags(ev: *mut *mut c_void, flags: u32) -> CudaError {
    if ev.is_null() { return record(INVALID_VALUE); }
    with(|r| { let id = r.create_event(flags & 2 == 0); unsafe { *ev = id as *mut c_void }; SUCCESS })
}
#[no_mangle] pub extern "C" fn cudaEventDestroy(ev: *mut c_void) -> CudaError { with(|r| r.destroy_event(ev as u64)) }
#[no_mangle] pub extern "C" fn cudaEventRecord(ev: *mut c_void, s: *mut c_void) -> CudaError {
    with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(e) => return e }; r.record_event(ev as u64, s) })
}
#[no_mangle] pub extern "C" fn cudaEventRecordWithFlags(ev: *mut c_void, s: *mut c_void, flags: u32) -> CudaError {
    with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(e) => return e }; r.record_event_flags(ev as u64, s, flags & 0x01 != 0) })
}
#[no_mangle] pub extern "C" fn cudaEventQuery(ev: *mut c_void) -> CudaError {
    match rt() { Ok(mut r) => { let rc = r.query_event(ev as u64); if rc == NOT_READY { rc } else { record(rc) } } Err(e) => record(e) }
}
#[no_mangle] pub extern "C" fn cudaEventSynchronize(ev: *mut c_void) -> CudaError { with(|r| r.sync_event(ev as u64)) }
#[no_mangle] pub extern "C" fn cudaEventElapsedTime(ms: *mut f32, a: *mut c_void, b: *mut c_void) -> CudaError {
    if ms.is_null() { return record(INVALID_VALUE); }
    with(|r| match r.elapsed_ms(a as u64, b as u64) { Ok(v) => { unsafe { *ms = v }; SUCCESS } Err(e) => e })
}

// ------------------------------------------------------------------ execution

#[no_mangle] pub extern "C" fn cudaLaunchKernel(func: *const c_void, grid: Dim3, block: Dim3, args: *mut *mut c_void, smem: usize, stream: *mut c_void) -> CudaError {
    STATS_LAUNCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    with(|r| {
        let s = match sid(r, stream) { Ok(s) => s, Err(e) => return e };
        r.launch(func as usize, [grid.x, grid.y, grid.z], [block.x, block.y, block.z], args as *const *const c_void, smem as u32, s)
    })
}
#[repr(C)] pub struct LaunchConfig { grid: Dim3, block: Dim3, smem: usize, stream: *mut c_void, attrs: *mut c_void, num_attrs: u32 }
#[no_mangle] pub extern "C" fn cudaLaunchKernelExC(cfg: *const LaunchConfig, func: *const c_void, args: *mut *mut c_void) -> CudaError {
    if cfg.is_null() { return record(INVALID_VALUE); }
    let c = unsafe { &*cfg };
    cudaLaunchKernel(func, c.grid, c.block, args, c.smem, c.stream)
}
#[no_mangle] pub extern "C" fn cudaLaunchCooperativeKernel(func: *const c_void, grid: Dim3, block: Dim3, args: *mut *mut c_void, smem: usize, stream: *mut c_void) -> CudaError {
    // No grid-wide sync primitive on Metal; a cooperative launch degrades to a normal launch. Kernels using grid.sync() fail at compile time.
    cudaLaunchKernel(func, grid, block, args, smem, stream)
}
#[no_mangle] pub extern "C" fn cudaFuncSetAttribute(func: *const c_void, attr: c_int, value: c_int) -> CudaError {
    with(|r| {
        let Some(k) = r.kernels.get_mut(&(func as usize)) else { return INVALID_DEVICE_FUNCTION };
        match attr {
            8 => { // MaxDynamicSharedMemorySize
                let static_smem = r.modules[k.module].static_smem_floor(&k.name);
                // a request above the device limit is fine when the recovered body's proven footprint fits (the
                // launch uses that footprint; the exact twin is only chosen when the recovered body cannot run)
                let effective = r.modules[k.module].effective_dyn_smem(&k.name, value.max(0) as u32);
                // beyond threadgroup memory the launch runs from the spill build (runtime.rs), up to the opt-in limit
                if value < 0 || (effective + static_smem) > crate::runtime::MAX_SPILL_SMEM { return INVALID_VALUE; }
                k.max_dynamic_smem = value as u32; SUCCESS
            }
            9 => { k.preferred_carveout = value; SUCCESS }
            _ => SUCCESS,
        }
    })
}
#[repr(C)] pub struct FuncAttributes { shared: usize, const_: usize, local: usize, max_threads: c_int, num_regs: c_int, ptx: c_int, bin: c_int, cache_ca: c_int, max_dyn: c_int, carveout: c_int, cluster_must: c_int, req_w: c_int, req_h: c_int, req_d: c_int, cluster_policy: c_int, non_portable: c_int, reserved: [c_int; 16] }
#[no_mangle] pub extern "C" fn cudaFuncGetAttributes(attr: *mut FuncAttributes, func: *const c_void) -> CudaError {
    if attr.is_null() { return record(INVALID_VALUE); }
    with(|r| {
        let (k, kp) = match r.kernel_pipeline(func as usize) { Ok(x) => x, Err(e) => return e };
        let static_floor = r.modules[k.module].static_smem_floor(&k.name);
        let a = FuncAttributes { shared: static_floor as usize, const_: 0, local: 0, max_threads: kp.max_threads as c_int, num_regs: if kp.max_threads >= 1024 { 32 } else { 128 }, ptx: 89, bin: 89, cache_ca: 0,
            max_dyn: if k.max_dynamic_smem > 0 { k.max_dynamic_smem as c_int } else { (SMEM_DEFAULT_LIMIT - static_floor.min(SMEM_DEFAULT_LIMIT)) as c_int }, carveout: k.preferred_carveout, cluster_must: 0, req_w: 0, req_h: 0, req_d: 0, cluster_policy: 0, non_portable: 0, reserved: [0; 16] };
        unsafe { *attr = a };
        SUCCESS
    })
}
#[no_mangle] pub extern "C" fn cudaFuncSetCacheConfig(_f: *const c_void, _c: c_int) -> CudaError { SUCCESS }
#[no_mangle] pub extern "C" fn cudaFuncSetSharedMemConfig(_f: *const c_void, _c: c_int) -> CudaError { SUCCESS }
#[no_mangle] pub extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessor(n: *mut c_int, func: *const c_void, block: c_int, dyn_smem: usize) -> CudaError {
    if n.is_null() { return record(INVALID_VALUE); }
    with(|r| match r.occupancy_blocks(func as usize, block.max(0) as u32, dyn_smem as u32) { Ok(v) => { unsafe { *n = v }; SUCCESS } Err(e) => e })
}
#[no_mangle] pub extern "C" fn cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(n: *mut c_int, func: *const c_void, block: c_int, dyn_smem: usize, _f: u32) -> CudaError {
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(n, func, block, dyn_smem)
}
#[no_mangle] pub extern "C" fn cudaOccupancyAvailableDynamicSMemPerBlock(out: *mut usize, func: *const c_void, _num_blocks: c_int, _block: c_int) -> CudaError {
    if out.is_null() { return record(INVALID_VALUE); }
    with(|r| match r.kernel_pipeline(func as usize) { Ok((k, _)) => { let f = r.modules[k.module].static_smem_floor(&k.name); unsafe { *out = crate::runtime::MAX_SPILL_SMEM.saturating_sub(f) as usize }; SUCCESS } Err(e) => e })
}

// ------------------------------------------------------------------ graphs

#[no_mangle] pub extern "C" fn cudaGraphCreate(g: *mut *mut c_void, _flags: u32) -> CudaError { let _ = g; record(NOT_SUPPORTED) }
#[no_mangle] pub extern "C" fn cudaGraphDestroy(g: *mut c_void) -> CudaError { with(|r| if r.graphs.remove(&(g as u64)).is_some() { SUCCESS } else { INVALID_RESOURCE_HANDLE }) }
#[no_mangle] pub extern "C" fn cudaGraphInstantiate(exec: *mut *mut c_void, g: *mut c_void, _flags: u64) -> CudaError {
    if exec.is_null() { return record(INVALID_VALUE); }
    with(|r| match r.instantiate(g as u64) { Ok(e) => { unsafe { *exec = e as *mut c_void }; SUCCESS } Err(e) => { unsafe { *exec = std::ptr::null_mut() }; e } })
}
#[no_mangle] pub extern "C" fn cudaGraphInstantiateWithFlags(exec: *mut *mut c_void, g: *mut c_void, flags: u64) -> CudaError { cudaGraphInstantiate(exec, g, flags) }
#[no_mangle] pub extern "C" fn cudaGraphExecDestroy(e: *mut c_void) -> CudaError { with(|r| if r.execs.remove(&(e as u64)).is_some() { SUCCESS } else { INVALID_RESOURCE_HANDLE }) }
#[no_mangle] pub extern "C" fn cudaGraphLaunch(e: *mut c_void, s: *mut c_void) -> CudaError {
    with(|r| { let s = match sid(r, s) { Ok(s) => s, Err(err) => return err }; r.graph_launch(e as u64, s) })
}
#[no_mangle] pub extern "C" fn cudaGraphUpload(e: *mut c_void, _s: *mut c_void) -> CudaError { with(|r| r.graph_upload(e as u64)) }
#[no_mangle] pub extern "C" fn cudaGraphGetNodes(g: *mut c_void, nodes: *mut *mut c_void, n: *mut usize) -> CudaError {
    with(|r| r.graph_get_nodes(g as u64, nodes, n))
}
#[no_mangle] pub extern "C" fn cudaGraphAddKernelNode(_n: *mut *mut c_void, _g: *mut c_void, _d: *const c_void, _nd: usize, _p: *const c_void) -> CudaError { record(NOT_SUPPORTED) }
#[no_mangle] pub extern "C" fn cudaGraphAddMemcpyNode1D(_n: *mut *mut c_void, _g: *mut c_void, _d: *const c_void, _nd: usize, _dst: *mut c_void, _src: *const c_void, _c: usize, _k: c_int) -> CudaError { record(NOT_SUPPORTED) }
#[no_mangle] pub extern "C" fn cudaGraphAddMemsetNode(_n: *mut *mut c_void, _g: *mut c_void, _d: *const c_void, _nd: usize, _p: *const c_void) -> CudaError { record(NOT_SUPPORTED) }
#[no_mangle] pub extern "C" fn cudaGraphAddEmptyNode(_n: *mut *mut c_void, _g: *mut c_void, _d: *const c_void, _nd: usize) -> CudaError { record(NOT_SUPPORTED) }
#[repr(C)]
struct KernelNodeParams { func: *mut c_void, grid_dim: Dim3, block_dim: Dim3, shared_mem_bytes: u32, kernel_params: *mut *mut c_void, extra: *mut *mut c_void }
#[no_mangle] pub extern "C" fn cudaGraphExecKernelNodeSetParams(e: *mut c_void, n: *mut c_void, p: *const c_void) -> CudaError {
    if p.is_null() { return record(INVALID_VALUE); }
    with(|r| {
        let np = unsafe { &*(p as *const KernelNodeParams) };
        let grid = [np.grid_dim.x, np.grid_dim.y, np.grid_dim.z];
        let block = [np.block_dim.x, np.block_dim.y, np.block_dim.z];
        r.graph_exec_kernel_node_set_params(e as u64, n, np.func as *const c_void, grid, block, np.shared_mem_bytes, np.kernel_params as *const *const c_void)
    })
}
#[no_mangle] pub extern "C" fn cudaGraphKernelNodeSetParams(_n: *mut c_void, _p: *const c_void) -> CudaError { record(NOT_SUPPORTED) }
#[no_mangle] pub extern "C" fn cudaGraphKernelNodeGetParams(_n: *mut c_void, _p: *mut c_void) -> CudaError { record(NOT_SUPPORTED) }
#[no_mangle] pub extern "C" fn cudaGraphNodeGetType(n: *mut c_void, ty: *mut c_int) -> CudaError {
    if ty.is_null() { return record(INVALID_VALUE); }
    with(|r| r.graph_node_get_type(n, ty as *mut i32))
}
#[no_mangle] pub extern "C" fn cudaGraphDebugDotPrint(g: *mut c_void, path: *const c_char, flags: u32) -> CudaError {
    if path.is_null() { return record(INVALID_VALUE); }
    let path = match unsafe { CStr::from_ptr(path) }.to_str() {
        Ok(p) => p,
        Err(_) => return record(INVALID_VALUE),
    };
    with(|r| r.graph_debug_dot_print(g as u64, path, flags))
}

#[no_mangle] pub extern "C" fn cudaProfilerStart() -> CudaError { SUCCESS }
#[no_mangle] pub extern "C" fn cudaProfilerStop() -> CudaError { SUCCESS }

// ------------------------------------------------------------------ registration ABI (clang host codegen)

#[repr(C)] struct FatbinWrapper { magic: u32, version: u32, data: *const u8, filename: *const c_void }

/// Device code registered by static constructors before the runtime exists. Parsing the blob and
/// creating the Metal device are deferred to first use so linking libcudart costs nothing at startup.
struct Pending { blobs: Vec<usize>, kernels: Vec<(usize, usize, String)>, vars: Vec<(usize, usize, String)> }
static PENDING: Mutex<Pending> = Mutex::new(Pending { blobs: Vec::new(), kernels: Vec::new(), vars: Vec::new() });
static PENDING_COUNT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Apply deferred registrations to a runtime (called under the RT lock).
fn drain_pending(r: &mut Runtime) {
    if PENDING_COUNT.load(std::sync::atomic::Ordering::Acquire) == 0 { return; }
    let mut p = PENDING.lock();
    for &blob in &p.blobs {
        if let Err(e) = r.register_module(blob as *const u8) { eprintln!("[MVCC] cannot register device code: {}", e); }
    }
    for (blob, host_fn, name) in p.kernels.drain(..) {
        match r.module_index(blob) {
            Some(idx) => { if let Err(e) = r.register_kernel(idx, host_fn, &name) { eprintln!("[MVCC] {}", e); } }
            None => eprintln!("[MVCC] kernel {} belongs to a module that failed to register", name),
        }
    }
    for (blob, host_sym, name) in p.vars.drain(..) {
        match r.module_index(blob) {
            // `const` tables the compiler kept as program-scope constant data have no runtime symbol (a copy to
            // them is invalid anyway); registering them is a no-op.
            Some(idx) => { let _ = r.register_symbol(idx, host_sym, &name); }
            None => eprintln!("[MVCC] variable {} belongs to a module that failed to register", name),
        }
    }
    p.blobs.clear();
    PENDING_COUNT.store(0, std::sync::atomic::Ordering::Release);
}

#[no_mangle] pub unsafe extern "C" fn __cudaRegisterFatBinary(fat: *mut c_void) -> *mut *mut c_void {
    let w = &*(fat as *const FatbinWrapper);
    if w.magic != 0x466243b1 { eprintln!("[MVCC] __cudaRegisterFatBinary: bad wrapper magic {:#x}", w.magic); return std::ptr::null_mut(); }
    if w.data.is_null() { return std::ptr::null_mut(); }
    PENDING.lock().blobs.push(w.data as usize);
    PENDING_COUNT.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    // the handle is the blob address itself, boxed so clang's generated code can store it
    Box::into_raw(Box::new(w.data as usize)) as *mut *mut c_void
}
#[no_mangle] pub extern "C" fn __cudaRegisterFatBinaryEnd(_h: *mut *mut c_void) {}
#[no_mangle] pub unsafe extern "C" fn __cudaUnregisterFatBinary(h: *mut *mut c_void) {
    if h.is_null() { return; }
    // process teardown: persist the pipeline cache once
    if let Some(Ok(m)) = RT.get() { if let Some(mut r) = m.try_lock() { r.flush_caches(); } }
    drop(Box::from_raw(h as *mut usize));
}
#[no_mangle] pub unsafe extern "C" fn __cudaRegisterFunction(h: *mut *mut c_void, host_fn: *const c_char, _device_fn: *mut c_char, device_name: *const c_char, _thread_limit: c_int,
    _tid: *mut c_void, _bid: *mut c_void, _bdim: *mut c_void, _gdim: *mut c_void, _wsize: *mut c_int) -> c_int {
    if h.is_null() { return INVALID_KERNEL_IMAGE; }
    let blob = *(h as *const usize);
    let name = CStr::from_ptr(device_name).to_string_lossy().into_owned();
    PENDING.lock().kernels.push((blob, host_fn as usize, name));
    PENDING_COUNT.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
    SUCCESS
}
#[no_mangle] pub unsafe extern "C" fn __cudaRegisterVar(h: *mut *mut c_void, host: *mut c_char, _dev: *mut c_char, name: *const c_char, _ext: c_int, _size: usize, _constant: c_int, _global: c_int) {
    if h.is_null() { return; }
    let blob = *(h as *const usize);
    let name = CStr::from_ptr(name).to_string_lossy().into_owned();
    PENDING.lock().vars.push((blob, host as usize, name));
    PENDING_COUNT.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
}
#[no_mangle] pub extern "C" fn __cudaRegisterManagedVar(_h: *mut *mut c_void, _p: *mut *mut c_void, _d: *mut c_char, _n: *const c_char, _e: c_int, _s: usize, _c: c_int, _g: c_int) {}
#[no_mangle] pub extern "C" fn __cudaRegisterSurface(_h: *mut *mut c_void, _v: *const c_void, _d: *const *const c_void, _n: *const c_char, _dim: c_int, _e: c_int) {}
#[no_mangle] pub extern "C" fn __cudaRegisterTexture(_h: *mut *mut c_void, _v: *const c_void, _d: *const *const c_void, _n: *const c_char, _dim: c_int, _norm: c_int, _e: c_int) {}

struct CallConfig { grid: Dim3, block: Dim3, smem: usize, stream: *mut c_void }
thread_local! { static CONFIGS: RefCell<Vec<CallConfig>> = const { RefCell::new(Vec::new()) }; }
#[no_mangle] pub extern "C" fn __cudaPushCallConfiguration(grid: Dim3, block: Dim3, smem: usize, stream: *mut c_void) -> u32 {
    CONFIGS.with(|c| c.borrow_mut().push(CallConfig { grid, block, smem, stream }));
    0
}
#[no_mangle] pub unsafe extern "C" fn __cudaPopCallConfiguration(grid: *mut Dim3, block: *mut Dim3, smem: *mut usize, stream: *mut *mut c_void) -> CudaError {
    let Some(c) = CONFIGS.with(|c| c.borrow_mut().pop()) else { return record(INVALID_CONFIGURATION) };
    *grid = c.grid; *block = c.block; *smem = c.smem; *stream = c.stream;
    SUCCESS
}

// ------------------------------------------------------------------ mvcc extensions

#[no_mangle] pub extern "C" fn mvccGetMetalDevice() -> *mut c_void { match rt() { Ok(r) => r.dev.0, Err(_) => std::ptr::null_mut() } }
#[no_mangle] pub extern "C" fn mvccGetMetalCommandQueue(_s: *mut c_void) -> *mut c_void { std::ptr::null_mut() }
#[no_mangle] pub extern "C" fn mvccGetMetalBuffer(p: *const c_void, offset: *mut usize) -> *mut c_void {
    match rt() { Ok(r) => match r.alloc.resolve_device(p as u64) { Some((a, off)) => { if !offset.is_null() { unsafe { *offset = (a.offset + off) as usize } } a.buf.0 } None => std::ptr::null_mut() }, Err(_) => std::ptr::null_mut() }
}
#[no_mangle] pub extern "C" fn mvccGetStats(launches: *mut u64, cbs: *mut u64, compiles: *mut u64) {
    if !launches.is_null() { unsafe { *launches = STATS_LAUNCHES.load(std::sync::atomic::Ordering::Relaxed) } }
    if !cbs.is_null() { unsafe { *cbs = 0 } }
    if !compiles.is_null() { unsafe { *compiles = 0 } }
}
