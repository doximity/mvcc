//! Thin safe-ish wrappers over the C shim in cpp/mvcc-metal. Every wrapper owns one retained ObjC object.
#![allow(dead_code)]
use std::ffi::{c_char, c_void, CStr, CString};

mod ffi {
    use std::ffi::{c_char, c_void};
    pub type H = *mut c_void;
    extern "C" {
        pub fn mvccm_device_create() -> H;
        pub fn mvccm_device_release(d: H);
        pub fn mvccm_device_name(d: H, buf: *mut c_char, cap: usize);
        pub fn mvccm_device_recommended_working_set(d: H) -> u64;
        pub fn mvccm_device_current_allocated(d: H) -> u64;
        pub fn mvccm_device_registry_id(d: H) -> u64;
        pub fn mvccm_device_max_threadgroup_memory(d: H) -> u32;
        pub fn mvccm_device_core_count(d: H) -> u32;
        pub fn mvccm_device_supports_metal4(d: H) -> i32;
        pub fn mvccm_physical_memory() -> u64;
        pub fn mvccm_buffer_new(d: H, len: u64) -> H;
        pub fn mvccm_buffer_wrap_host(d: H, p: *mut c_void, len: u64) -> H;
        pub fn mvccm_buffer_contents(b: H) -> *mut c_void;
        pub fn mvccm_buffer_gpu_address(b: H) -> u64;
        pub fn mvccm_buffer_length(b: H) -> u64;
        pub fn mvccm_buffer_release(b: H);
        pub fn mvccm_residency_new(d: H) -> H;
        pub fn mvccm_residency_add_buffer(r: H, b: H);
        pub fn mvccm_residency_remove_buffer(r: H, b: H);
        pub fn mvccm_residency_commit(r: H);
        pub fn mvccm_residency_release(r: H);
        pub fn mvccm_library_from_source(d: H, src: *const c_char, len: usize, metal4: i32, fast: i32, err: *mut *mut c_char) -> H;
        pub fn mvccm_library_from_source_defs(d: H, src: *const c_char, len: usize, metal4: i32, fast: i32, defines: *const c_char, err: *mut *mut c_char) -> H;
        pub fn mvccm_library_from_data(d: H, bytes: *const c_void, len: usize, err: *mut *mut c_char) -> H;
        pub fn mvccm_library_release(l: H);
        pub fn mvccm_pipeline_new(d: H, l: H, name: *const c_char, max_threads: u32, archive: H, err: *mut *mut c_char) -> H;
        pub fn mvccm_pipeline_max_threads(p: H) -> u32;
        pub fn mvccm_pipeline_exec_width(p: H) -> u32;
        pub fn mvccm_pipeline_static_tg_mem(p: H) -> u32;
        pub fn mvccm_pipeline_release(p: H);
        pub fn mvccm_archive_open(d: H, path: *const c_char, err: *mut *mut c_char) -> H;
        pub fn mvccm_archive_serialize(a: H, path: *const c_char, err: *mut *mut c_char) -> i32;
        pub fn mvccm_archive_release(a: H);
        pub fn mvccm_free_string(s: *mut c_char);
        pub fn mvccm_queue_new(d: H, r: H) -> H;
        pub fn mvccm_queue_release(q: H);
        pub fn mvccm_cmdbuf_new(q: H) -> H;
        pub fn mvccm_cmdbuf_release(c: H);
        pub fn mvccm_compute_begin(c: H) -> H;
        pub fn mvccm_compute_set_pipeline(e: H, p: H);
        pub fn mvccm_compute_set_bytes(e: H, data: *const c_void, len: usize, index: u32);
        pub fn mvccm_compute_set_buffer(e: H, b: H, off: u64, index: u32);
        pub fn mvccm_compute_use_buffer(e: H, b: H);
        pub fn mvccm_compute_set_tg_mem(e: H, len: u32, index: u32);
        pub fn mvccm_compute_dispatch(e: H, gx: u32, gy: u32, gz: u32, bx: u32, by: u32, bz: u32);
        pub fn mvccm_compute_dispatch_indirect(e: H, args: H, off: u64, bx: u32, by: u32, bz: u32);
        pub fn mvccm_compute_barrier(e: H);
        pub fn mvccm_device_new_fence(d: H) -> H;
        pub fn mvccm_fence_release(f: H);
        pub fn mvccm_compute_wait_fence(e: H, f: H);
        pub fn mvccm_compute_update_fence(e: H, f: H);
        pub fn mvccm_blit_wait_fence(e: H, f: H);
        pub fn mvccm_blit_update_fence(e: H, f: H);
        pub fn mvccm_compute_end(e: H);
        pub fn mvccm_blit_begin(c: H) -> H;
        pub fn mvccm_blit_copy(e: H, src: H, soff: u64, dst: H, doff: u64, len: u64);
        pub fn mvccm_blit_fill(e: H, dst: H, off: u64, len: u64, v: u8);
        pub fn mvccm_blit_end(e: H);
        pub fn mvccm_cmdbuf_signal_event(c: H, ev: H, v: u64);
        pub fn mvccm_cmdbuf_wait_event(c: H, ev: H, v: u64);
        pub fn mvccm_cmdbuf_on_completed(c: H, cb: extern "C" fn(*mut c_void, i32), ctx: *mut c_void);
        pub fn mvccm_cmdbuf_commit(c: H);
        pub fn mvccm_cmdbuf_wait(c: H);
        pub fn mvccm_cmdbuf_status(c: H) -> i32;
        pub fn mvccm_cmdbuf_error(c: H) -> *mut c_char;
        pub fn mvccm_cmdbuf_gpu_start(c: H) -> f64;
        pub fn mvccm_cmdbuf_gpu_end(c: H) -> f64;
        pub fn mvccm_event_new(d: H) -> H;
        pub fn mvccm_event_value(e: H) -> u64;
        pub fn mvccm_event_set_value(e: H, v: u64);
        pub fn mvccm_event_wait(e: H, v: u64, timeout_ms: i64) -> i32;
        pub fn mvccm_event_release(e: H);
        pub fn mvccm_now_seconds() -> f64;
    }
}
use ffi::H;

fn take_err(p: *mut c_char) -> String {
    if p.is_null() { return "unknown Metal error".into(); }
    let s = unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned();
    unsafe { ffi::mvccm_free_string(p) };
    s
}

macro_rules! handle {
    ($name:ident, $rel:path) => {
        pub struct $name(pub(crate) H);
        unsafe impl Send for $name {}
        unsafe impl Sync for $name {}
        impl Drop for $name { fn drop(&mut self) { if !self.0.is_null() { unsafe { $rel(self.0) } } } }
    };
}
handle!(Device, ffi::mvccm_device_release);
handle!(Buffer, ffi::mvccm_buffer_release);
handle!(Residency, ffi::mvccm_residency_release);
handle!(Library, ffi::mvccm_library_release);
handle!(Pipeline, ffi::mvccm_pipeline_release);
handle!(Archive, ffi::mvccm_archive_release);
handle!(Queue, ffi::mvccm_queue_release);
handle!(CmdBuf, ffi::mvccm_cmdbuf_release);
handle!(SharedEvent, ffi::mvccm_event_release);

/// An open compute encoder; ended (and released) on drop or `end()`.
pub struct Fence(H);
unsafe impl Send for Fence {}
unsafe impl Sync for Fence {}
impl Drop for Fence { fn drop(&mut self) { if !self.0.is_null() { unsafe { ffi::mvccm_fence_release(self.0) } } } }
pub struct Encoder(H);
unsafe impl Send for Encoder {}
impl Drop for Encoder { fn drop(&mut self) { if !self.0.is_null() { unsafe { ffi::mvccm_compute_end(self.0) } } } }
pub struct Blit(H);
unsafe impl Send for Blit {}
impl Drop for Blit { fn drop(&mut self) { if !self.0.is_null() { unsafe { ffi::mvccm_blit_end(self.0) } } } }

impl Device {
    pub fn create() -> Option<Device> {
        let h = unsafe { ffi::mvccm_device_create() };
        if h.is_null() { None } else { Some(Device(h)) }
    }
    pub fn name(&self) -> String {
        let mut buf = [0 as c_char; 256];
        unsafe { ffi::mvccm_device_name(self.0, buf.as_mut_ptr(), buf.len()) };
        unsafe { CStr::from_ptr(buf.as_ptr()) }.to_string_lossy().into_owned()
    }
    pub fn recommended_working_set(&self) -> u64 { unsafe { ffi::mvccm_device_recommended_working_set(self.0) } }
    pub fn current_allocated(&self) -> u64 { unsafe { ffi::mvccm_device_current_allocated(self.0) } }
    pub fn registry_id(&self) -> u64 { unsafe { ffi::mvccm_device_registry_id(self.0) } }
    pub fn max_threadgroup_memory(&self) -> u32 { unsafe { ffi::mvccm_device_max_threadgroup_memory(self.0) } }
    pub fn core_count(&self) -> u32 { unsafe { ffi::mvccm_device_core_count(self.0) } }
    pub fn supports_metal4(&self) -> bool { unsafe { ffi::mvccm_device_supports_metal4(self.0) != 0 } }
    pub fn physical_memory() -> u64 { unsafe { ffi::mvccm_physical_memory() } }
    pub fn new_buffer(&self, len: u64) -> Option<Buffer> {
        let h = unsafe { ffi::mvccm_buffer_new(self.0, len) };
        if h.is_null() { None } else { Some(Buffer(h)) }
    }
    pub fn wrap_host(&self, p: *mut c_void, len: u64) -> Option<Buffer> {
        let h = unsafe { ffi::mvccm_buffer_wrap_host(self.0, p, len) };
        if h.is_null() { None } else { Some(Buffer(h)) }
    }
    pub fn new_residency(&self) -> Residency { Residency(unsafe { ffi::mvccm_residency_new(self.0) }) }
    pub fn library_from_source(&self, src: &str, metal4: bool, fast_math: bool) -> Result<Library, String> {
        self.library_from_source_defs(src, metal4, fast_math, &[])
    }
    /// `defines`: preprocessor definitions ("NAME" or "NAME=VALUE") for this compilation.
    pub fn library_from_source_defs(&self, src: &str, metal4: bool, fast_math: bool, defines: &[&str]) -> Result<Library, String> {
        let mut err: *mut c_char = std::ptr::null_mut();
        let defs = CString::new(defines.join(";")).unwrap();
        let h = unsafe { ffi::mvccm_library_from_source_defs(self.0, src.as_ptr() as *const c_char, src.len(), metal4 as i32, fast_math as i32, defs.as_ptr(), &mut err) };
        if h.is_null() { Err(take_err(err)) } else { Ok(Library(h)) }
    }
    pub fn library_from_data(&self, data: &[u8]) -> Result<Library, String> {
        let mut err: *mut c_char = std::ptr::null_mut();
        let h = unsafe { ffi::mvccm_library_from_data(self.0, data.as_ptr() as *const c_void, data.len(), &mut err) };
        if h.is_null() { Err(take_err(err)) } else { Ok(Library(h)) }
    }
    pub fn new_pipeline(&self, lib: &Library, name: &str, max_threads: u32, archive: Option<&Archive>) -> Result<Pipeline, String> {
        let c = CString::new(name).unwrap();
        let mut err: *mut c_char = std::ptr::null_mut();
        let h = unsafe { ffi::mvccm_pipeline_new(self.0, lib.0, c.as_ptr(), max_threads, archive.map(|a| a.0).unwrap_or(std::ptr::null_mut()), &mut err) };
        if h.is_null() { Err(take_err(err)) } else { Ok(Pipeline(h)) }
    }
    pub fn open_archive(&self, path: Option<&str>) -> Result<Archive, String> {
        let c = path.map(|p| CString::new(p).unwrap());
        let mut err: *mut c_char = std::ptr::null_mut();
        let h = unsafe { ffi::mvccm_archive_open(self.0, c.as_ref().map(|c| c.as_ptr()).unwrap_or(std::ptr::null()), &mut err) };
        if h.is_null() { Err(take_err(err)) } else { Ok(Archive(h)) }
    }
    pub fn new_queue(&self, residency: Option<&Residency>) -> Queue {
        Queue(unsafe { ffi::mvccm_queue_new(self.0, residency.map(|r| r.0).unwrap_or(std::ptr::null_mut())) })
    }
    pub fn new_shared_event(&self) -> SharedEvent { SharedEvent(unsafe { ffi::mvccm_event_new(self.0) }) }
    pub fn new_fence(&self) -> Fence { Fence(unsafe { ffi::mvccm_device_new_fence(self.0) }) }
}

impl Buffer {
    pub fn contents(&self) -> *mut u8 { unsafe { ffi::mvccm_buffer_contents(self.0) as *mut u8 } }
    pub fn gpu_address(&self) -> u64 { unsafe { ffi::mvccm_buffer_gpu_address(self.0) } }
    pub fn len(&self) -> u64 { unsafe { ffi::mvccm_buffer_length(self.0) } }
}
impl Residency {
    pub fn add(&self, b: &Buffer) { unsafe { ffi::mvccm_residency_add_buffer(self.0, b.0) } }
    pub fn remove(&self, b: &Buffer) { unsafe { ffi::mvccm_residency_remove_buffer(self.0, b.0) } }
    pub fn commit(&self) { unsafe { ffi::mvccm_residency_commit(self.0) } }
}
impl Pipeline {
    pub fn max_threads(&self) -> u32 { unsafe { ffi::mvccm_pipeline_max_threads(self.0) } }
    pub fn exec_width(&self) -> u32 { unsafe { ffi::mvccm_pipeline_exec_width(self.0) } }
    pub fn static_tg_mem(&self) -> u32 { unsafe { ffi::mvccm_pipeline_static_tg_mem(self.0) } }
}
impl Archive {
    pub fn serialize(&self, path: &str) -> Result<(), String> {
        let c = CString::new(path).unwrap();
        let mut err: *mut c_char = std::ptr::null_mut();
        if unsafe { ffi::mvccm_archive_serialize(self.0, c.as_ptr(), &mut err) } != 0 { Ok(()) } else { Err(take_err(err)) }
    }
}
impl Queue {
    pub fn new_cmdbuf(&self) -> CmdBuf { CmdBuf(unsafe { ffi::mvccm_cmdbuf_new(self.0) }) }
}
impl CmdBuf {
    pub fn compute(&self) -> Encoder { Encoder(unsafe { ffi::mvccm_compute_begin(self.0) }) }
    pub fn blit(&self) -> Blit { Blit(unsafe { ffi::mvccm_blit_begin(self.0) }) }
    pub fn signal(&self, ev: &SharedEvent, v: u64) { unsafe { ffi::mvccm_cmdbuf_signal_event(self.0, ev.0, v) } }
    pub fn wait(&self, ev: &SharedEvent, v: u64) { unsafe { ffi::mvccm_cmdbuf_wait_event(self.0, ev.0, v) } }
    pub fn on_completed(&self, cb: extern "C" fn(*mut c_void, i32), ctx: *mut c_void) { unsafe { ffi::mvccm_cmdbuf_on_completed(self.0, cb, ctx) } }
    pub fn commit(&self) { unsafe { ffi::mvccm_cmdbuf_commit(self.0) } }
    pub fn wait_completed(&self) { unsafe { ffi::mvccm_cmdbuf_wait(self.0) } }
    /// 0 not committed, 1 in flight, 2 completed, 3 error
    pub fn status(&self) -> i32 { unsafe { ffi::mvccm_cmdbuf_status(self.0) } }
    pub fn error(&self) -> Option<String> {
        let p = unsafe { ffi::mvccm_cmdbuf_error(self.0) };
        if p.is_null() { None } else { Some(take_err(p)) }
    }
    pub fn gpu_start(&self) -> f64 { unsafe { ffi::mvccm_cmdbuf_gpu_start(self.0) } }
    pub fn gpu_end(&self) -> f64 { unsafe { ffi::mvccm_cmdbuf_gpu_end(self.0) } }
}
impl Encoder {
    pub fn set_pipeline(&self, p: &Pipeline) { unsafe { ffi::mvccm_compute_set_pipeline(self.0, p.0) } }
    pub fn set_bytes(&self, data: &[u8], index: u32) { unsafe { ffi::mvccm_compute_set_bytes(self.0, data.as_ptr() as *const c_void, data.len(), index) } }
    pub fn set_buffer(&self, b: &Buffer, off: u64, index: u32) { unsafe { ffi::mvccm_compute_set_buffer(self.0, b.0, off, index) } }
    pub fn use_buffer(&self, b: &Buffer) { unsafe { ffi::mvccm_compute_use_buffer(self.0, b.0) } }
    pub fn set_tg_mem(&self, len: u32, index: u32) { unsafe { ffi::mvccm_compute_set_tg_mem(self.0, len, index) } }
    pub fn dispatch(&self, g: [u32; 3], b: [u32; 3]) { unsafe { ffi::mvccm_compute_dispatch(self.0, g[0], g[1], g[2], b[0], b[1], b[2]) } }
    pub fn dispatch_indirect(&self, args: &Buffer, off: u64, b: [u32; 3]) { unsafe { ffi::mvccm_compute_dispatch_indirect(self.0, args.0, off, b[0], b[1], b[2]) } }
    pub fn barrier(&self) { unsafe { ffi::mvccm_compute_barrier(self.0) } }
    pub fn wait_fence(&self, f: &Fence) { unsafe { ffi::mvccm_compute_wait_fence(self.0, f.0) } }
    pub fn update_fence(&self, f: &Fence) { unsafe { ffi::mvccm_compute_update_fence(self.0, f.0) } }
    pub fn end(mut self) { unsafe { ffi::mvccm_compute_end(self.0) }; self.0 = std::ptr::null_mut(); }
}
impl Blit {
    pub fn copy(&self, src: &Buffer, soff: u64, dst: &Buffer, doff: u64, len: u64) { unsafe { ffi::mvccm_blit_copy(self.0, src.0, soff, dst.0, doff, len) } }
    pub fn fill(&self, dst: &Buffer, off: u64, len: u64, v: u8) { unsafe { ffi::mvccm_blit_fill(self.0, dst.0, off, len, v) } }
    pub fn end(mut self) { unsafe { ffi::mvccm_blit_end(self.0) }; self.0 = std::ptr::null_mut(); }
    pub fn wait_fence(&self, f: &Fence) { unsafe { ffi::mvccm_blit_wait_fence(self.0, f.0) } }
    pub fn update_fence(&self, f: &Fence) { unsafe { ffi::mvccm_blit_update_fence(self.0, f.0) } }
}
impl SharedEvent {
    pub fn value(&self) -> u64 { unsafe { ffi::mvccm_event_value(self.0) } }
    pub fn set_value(&self, v: u64) { unsafe { ffi::mvccm_event_set_value(self.0, v) } }
    pub fn wait(&self, v: u64, timeout_ms: i64) -> bool { unsafe { ffi::mvccm_event_wait(self.0, v, timeout_ms) != 0 } }
}
pub fn now_seconds() -> f64 { unsafe { ffi::mvccm_now_seconds() } }
