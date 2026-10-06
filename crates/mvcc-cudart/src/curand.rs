//! cuRAND host API for Philox4_32_10 (`include/curand.h`). `libcurand.dylib` is this library.
//!
//! Numbers come from `mvcc_philox_fill` in `cpp/mvcc-metal/curand_host.cpp`, which calls the same
//! header the device compiles. A device generator synchronizes its stream, then copies the buffer.
#![allow(non_snake_case, clippy::missing_safety_doc)]
use crate::api::{cudaMemcpy, cudaStreamSynchronize};
use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::OnceLock;

pub type CurandStatus = i32;
pub const CURAND_STATUS_SUCCESS: CurandStatus = 0;
pub const CURAND_STATUS_NOT_INITIALIZED: CurandStatus = 101;
pub const CURAND_STATUS_ALLOCATION_FAILED: CurandStatus = 102;
pub const CURAND_STATUS_TYPE_ERROR: CurandStatus = 103;
pub const CURAND_STATUS_LAUNCH_FAILURE: CurandStatus = 201;

const PHILOX: i32 = 161;
const ORDER_BEST: i32 = 100;
const ORDER_DEFAULT: i32 = 101;
const CURAND_VERSION: i32 = 10300;
const H2D: i32 = 1;
const CHUNK: usize = 1 << 20;

extern "C" {
    fn mvcc_philox_fill(seed: u64, offset: u64, out: *mut u32, n: u64);
    fn mvcc_philox_fill_uniform(seed: u64, offset: u64, out: *mut f32, n: u64);
}

struct Gen {
    host_mem: bool,
    seed: u64,
    offset: u64,
    stream: *mut c_void,
}
// The stream pointer is an opaque runtime handle. It is only passed back into cudaStreamSynchronize,
// which takes the runtime lock; the bits themselves are not dereferenced here.
unsafe impl Send for Gen {}

struct State {
    gens: HashMap<u64, Gen>,
}

fn state() -> &'static std::sync::Mutex<State> {
    static G: OnceLock<std::sync::Mutex<State>> = OnceLock::new();
    G.get_or_init(|| std::sync::Mutex::new(State { gens: HashMap::new() }))
}

fn note_type() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!("[mvcc curand] only Philox4_32_10 is implemented (CURAND_RNG_PSEUDO_PHILOX4_32_10)");
    });
}
fn note_order() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        eprintln!("[mvcc curand] generator ordering is one Philox subsequence (CURAND_ORDERING_PSEUDO_BEST); LEGACY interleave is not implemented");
    });
}

fn fresh_id(used: &HashMap<u64, Gen>) -> Result<u64, CurandStatus> {
    extern "C" {
        fn getentropy(buf: *mut u8, buflen: usize) -> i32;
    }
    for _ in 0..8 {
        let mut buf = [0u8; 8];
        if unsafe { getentropy(buf.as_mut_ptr(), buf.len()) } != 0 { return Err(999); }
        // Above 2^32 so a caller cannot guess a handle by trying 1, 2, 3, ... The top bit stays clear so
        // the value survives as a pointer argument on arm64 (the top byte is not an address bit we need).
        let id = u64::from_ne_bytes(buf) & 0x7fff_ffff_ffff_ffff;
        if id > 0xffff_ffff && !used.contains_key(&id) { return Ok(id); }
    }
    Err(999)
}

fn create(generator: *mut *mut c_void, rng_type: i32, host_mem: bool) -> CurandStatus {
    if generator.is_null() { return CURAND_STATUS_NOT_INITIALIZED; }
    if rng_type != PHILOX { note_type(); return CURAND_STATUS_TYPE_ERROR; }
    let mut s = state().lock().unwrap_or_else(|e| e.into_inner());
    let id = match fresh_id(&s.gens) { Ok(id) => id, Err(e) => return e };
    s.gens.insert(id, Gen { host_mem, seed: 0, offset: 0, stream: std::ptr::null_mut() });
    unsafe { *generator = id as *mut c_void; }
    CURAND_STATUS_SUCCESS
}

fn with_gen<F: FnOnce(&mut Gen) -> CurandStatus>(generator: *mut c_void, f: F) -> CurandStatus {
    if generator.is_null() { return CURAND_STATUS_NOT_INITIALIZED; }
    let mut s = state().lock().unwrap_or_else(|e| e.into_inner());
    match s.gens.get_mut(&(generator as u64)) {
        Some(g) => f(g),
        None => CURAND_STATUS_NOT_INITIALIZED,
    }
}

#[no_mangle] pub extern "C" fn curandCreateGenerator(generator: *mut *mut c_void, rng_type: i32) -> CurandStatus {
    create(generator, rng_type, false)
}
#[no_mangle] pub extern "C" fn curandCreateGeneratorHost(generator: *mut *mut c_void, rng_type: i32) -> CurandStatus {
    create(generator, rng_type, true)
}

#[no_mangle] pub extern "C" fn curandDestroyGenerator(generator: *mut c_void) -> CurandStatus {
    if generator.is_null() { return CURAND_STATUS_NOT_INITIALIZED; }
    let mut s = state().lock().unwrap_or_else(|e| e.into_inner());
    if s.gens.remove(&(generator as u64)).is_none() { CURAND_STATUS_NOT_INITIALIZED } else { CURAND_STATUS_SUCCESS }
}

#[no_mangle] pub extern "C" fn curandSetPseudoRandomGeneratorSeed(generator: *mut c_void, seed: u64) -> CurandStatus {
    with_gen(generator, |g| { g.seed = seed; g.offset = 0; CURAND_STATUS_SUCCESS })
}
#[no_mangle] pub extern "C" fn curandSetGeneratorOffset(generator: *mut c_void, offset: u64) -> CurandStatus {
    with_gen(generator, |g| { g.offset = offset; CURAND_STATUS_SUCCESS })
}
#[no_mangle] pub extern "C" fn curandSetStream(generator: *mut c_void, stream: *mut c_void) -> CurandStatus {
    with_gen(generator, |g| { g.stream = stream; CURAND_STATUS_SUCCESS })
}
#[no_mangle] pub extern "C" fn curandSetGeneratorOrdering(generator: *mut c_void, order: i32) -> CurandStatus {
    with_gen(generator, |_| {
        if order == ORDER_BEST || order == ORDER_DEFAULT { CURAND_STATUS_SUCCESS } else { note_order(); CURAND_STATUS_TYPE_ERROR }
    })
}
#[no_mangle] pub extern "C" fn curandGetVersion(version: *mut i32) -> CurandStatus {
    if version.is_null() { return CURAND_STATUS_NOT_INITIALIZED; }
    unsafe { *version = CURAND_VERSION; }
    CURAND_STATUS_SUCCESS
}

fn deliver(gen: &Gen, bytes: &[u8], dst: *mut c_void) -> CurandStatus {
    if gen.host_mem {
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), dst as *mut u8, bytes.len()); }
        return CURAND_STATUS_SUCCESS;
    }
    if !gen.stream.is_null() && cudaStreamSynchronize(gen.stream) != 0 { return CURAND_STATUS_LAUNCH_FAILURE; }
    if cudaMemcpy(dst, bytes.as_ptr() as *const c_void, bytes.len(), H2D) != 0 { CURAND_STATUS_LAUNCH_FAILURE } else { CURAND_STATUS_SUCCESS }
}

fn generate(generator: *mut c_void, dst: *mut c_void, num: usize, uniform: bool) -> CurandStatus {
    if dst.is_null() && num != 0 { return CURAND_STATUS_NOT_INITIALIZED; }
    if generator.is_null() { return CURAND_STATUS_NOT_INITIALIZED; }
    // One generator is not used concurrently. Holding the map lock across the fill keeps the offset
    // update atomic with the copy; the Philox call does not re-enter the runtime.
    let mut s = state().lock().unwrap_or_else(|e| e.into_inner());
    let (host_mem, seed, stream, mut offset) = {
        let g = match s.gens.get(&(generator as u64)) { Some(g) => g, None => return CURAND_STATUS_NOT_INITIALIZED };
        (g.host_mem, g.seed, g.stream, g.offset)
    };
    if num == 0 { return CURAND_STATUS_SUCCESS; }
    let copy = Gen { host_mem, seed, offset, stream };
    let mut done = 0usize;
    while done < num {
        let n = (num - done).min(CHUNK);
        let status = if uniform {
            let mut buf = vec![0f32; n];
            unsafe { mvcc_philox_fill_uniform(seed, offset, buf.as_mut_ptr(), n as u64); }
            let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, n * 4) };
            let dst_i = unsafe { (dst as *mut u8).add(done * 4) } as *mut c_void;
            deliver(&copy, bytes, dst_i)
        } else {
            let mut buf = vec![0u32; n];
            unsafe { mvcc_philox_fill(seed, offset, buf.as_mut_ptr(), n as u64); }
            let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, n * 4) };
            let dst_i = unsafe { (dst as *mut u8).add(done * 4) } as *mut c_void;
            deliver(&copy, bytes, dst_i)
        };
        if status != CURAND_STATUS_SUCCESS { return status; }
        offset = offset.wrapping_add(n as u64);
        done += n;
    }
    if let Some(g) = s.gens.get_mut(&(generator as u64)) { g.offset = offset; }
    CURAND_STATUS_SUCCESS
}

#[no_mangle] pub extern "C" fn curandGenerate(generator: *mut c_void, output: *mut u32, num: usize) -> CurandStatus {
    generate(generator, output as *mut c_void, num, false)
}
#[no_mangle] pub extern "C" fn curandGenerateUniform(generator: *mut c_void, output: *mut f32, num: usize) -> CurandStatus {
    generate(generator, output as *mut c_void, num, true)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn philox_kat_zero_and_offset() {
        let mut first = [0u32; 8];
        let mut skipped = [0u32; 4];
        unsafe {
            mvcc_philox_fill(0, 0, first.as_mut_ptr(), 8);
            mvcc_philox_fill(0, 5, skipped.as_mut_ptr(), 1);
        }
        // Random123 known answer: Philox4x32-10(counter 0, key 0).
        assert_eq!(&first[..4], &[0x6627e8d5, 0xe169c58d, 0xbc57ac4c, 0x9b00dbd8]);
        assert_eq!(skipped[0], first[5]);
    }
}
