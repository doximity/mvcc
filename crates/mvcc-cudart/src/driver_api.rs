//! The CUDA driver API subset (cuda.h). Same library as the runtime; libcuda.dylib is a symlink to libcudart.dylib.
#![allow(non_snake_case, clippy::missing_safety_doc)]
use crate::api::*;
use std::ffi::{c_char, c_int, c_void};

type CUresult = c_int;
const CUDA_SUCCESS: CUresult = 0;
const CUDA_ERROR_INVALID_VALUE: CUresult = 1;
const CUDA_ERROR_OUT_OF_MEMORY: CUresult = 2;
const CUDA_ERROR_NOT_INITIALIZED: CUresult = 3;
const CUDA_ERROR_NO_DEVICE: CUresult = 100;
const CUDA_ERROR_INVALID_DEVICE: CUresult = 101;
const CUDA_ERROR_UNKNOWN: CUresult = 999;

fn map(e: crate::error::CudaError) -> CUresult {
    match e { 0 => CUDA_SUCCESS, 1 => CUDA_ERROR_INVALID_VALUE, 2 => CUDA_ERROR_OUT_OF_MEMORY, 3 => CUDA_ERROR_NOT_INITIALIZED, 100 => CUDA_ERROR_NO_DEVICE, 101 => CUDA_ERROR_INVALID_DEVICE, 700 => 700, _ => CUDA_ERROR_UNKNOWN }
}

#[no_mangle] pub extern "C" fn cuInit(_flags: u32) -> CUresult { let mut n = 0; map(cudaGetDeviceCount(&mut n)) }
#[no_mangle] pub extern "C" fn cuDriverGetVersion(v: *mut c_int) -> CUresult { map(cudaDriverGetVersion(v)) }
#[no_mangle] pub extern "C" fn cuDeviceGetCount(n: *mut c_int) -> CUresult { map(cudaGetDeviceCount(n)) }
#[no_mangle] pub extern "C" fn cuDeviceGet(d: *mut c_int, ordinal: c_int) -> CUresult {
    if d.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    if ordinal != 0 { return CUDA_ERROR_INVALID_DEVICE; }
    unsafe { *d = 0 }; CUDA_SUCCESS
}
#[no_mangle] pub extern "C" fn cuDeviceGetName(buf: *mut c_char, len: c_int, _dev: c_int) -> CUresult {
    if buf.is_null() || len <= 0 { return CUDA_ERROR_INVALID_VALUE; }
    let mut prop = vec![0u8; 4096];
    let rc = cudaGetDeviceProperties(prop.as_mut_ptr() as *mut c_void, 0);
    if rc != 0 { return map(rc); }
    let n = (len as usize - 1).min(255);
    unsafe { std::ptr::copy_nonoverlapping(prop.as_ptr(), buf as *mut u8, n); *buf.add(n) = 0; }
    CUDA_SUCCESS
}
#[no_mangle] pub extern "C" fn cuDeviceTotalMem(bytes: *mut usize, _dev: c_int) -> CUresult {
    if bytes.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let mut f = 0usize; map(cudaMemGetInfo(&mut f, bytes))
}
#[no_mangle] pub extern "C" fn cuDevicePrimaryCtxRetain(ctx: *mut *mut c_void, _dev: c_int) -> CUresult { if !ctx.is_null() { unsafe { *ctx = 1 as *mut c_void } } CUDA_SUCCESS }
#[no_mangle] pub extern "C" fn cuDevicePrimaryCtxRelease(_dev: c_int) -> CUresult { CUDA_SUCCESS }
#[no_mangle] pub extern "C" fn cuCtxGetCurrent(ctx: *mut *mut c_void) -> CUresult { if !ctx.is_null() { unsafe { *ctx = 1 as *mut c_void } } CUDA_SUCCESS }
#[no_mangle] pub extern "C" fn cuCtxSetCurrent(_ctx: *mut c_void) -> CUresult { CUDA_SUCCESS }
#[no_mangle] pub extern "C" fn cuCtxSynchronize() -> CUresult { map(cudaDeviceSynchronize()) }
#[no_mangle] pub extern "C" fn cuMemAlloc(p: *mut u64, size: usize) -> CUresult {
    if p.is_null() { return CUDA_ERROR_INVALID_VALUE; }
    let mut v: *mut c_void = std::ptr::null_mut();
    let rc = cudaMalloc(&mut v, size);
    unsafe { *p = v as u64 }; map(rc)
}
#[no_mangle] pub extern "C" fn cuMemFree(p: u64) -> CUresult { map(cudaFree(p as *mut c_void)) }
#[no_mangle] pub extern "C" fn cuMemcpyHtoD(dst: u64, src: *const c_void, n: usize) -> CUresult { map(cudaMemcpy(dst as *mut c_void, src, n, 1)) }
#[no_mangle] pub extern "C" fn cuMemcpyDtoH(dst: *mut c_void, src: u64, n: usize) -> CUresult { map(cudaMemcpy(dst, src as *const c_void, n, 2)) }
#[no_mangle] pub extern "C" fn cuMemcpyDtoD(dst: u64, src: u64, n: usize) -> CUresult { map(cudaMemcpy(dst as *mut c_void, src as *const c_void, n, 3)) }
#[no_mangle] pub extern "C" fn cuGetErrorName(e: CUresult, s: *mut *const c_char) -> CUresult { if s.is_null() { return CUDA_ERROR_INVALID_VALUE; } unsafe { *s = cudaGetErrorName(e) }; CUDA_SUCCESS }
#[no_mangle] pub extern "C" fn cuGetErrorString(e: CUresult, s: *mut *const c_char) -> CUresult { if s.is_null() { return CUDA_ERROR_INVALID_VALUE; } unsafe { *s = cudaGetErrorString(e) }; CUDA_SUCCESS }
