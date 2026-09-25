//! cudaError_t values and the thread-local "last error" the runtime API exposes.
use std::cell::Cell;

pub type CudaError = i32;
pub const SUCCESS: CudaError = 0;
pub const INVALID_VALUE: CudaError = 1;
pub const MEMORY_ALLOCATION: CudaError = 2;
pub const INITIALIZATION_ERROR: CudaError = 3;
pub const INVALID_CONFIGURATION: CudaError = 9;
pub const INVALID_PITCH_VALUE: CudaError = 12;
pub const INVALID_SYMBOL: CudaError = 13;
pub const INVALID_HOST_POINTER: CudaError = 16;
pub const INVALID_DEVICE_POINTER: CudaError = 17;
pub const INVALID_MEMCPY_DIRECTION: CudaError = 21;
pub const INVALID_DEVICE_FUNCTION: CudaError = 98;
pub const NO_DEVICE: CudaError = 100;
pub const INVALID_DEVICE: CudaError = 101;
pub const INVALID_KERNEL_IMAGE: CudaError = 200;
pub const UNSUPPORTED_LIMIT: CudaError = 215;
pub const INVALID_RESOURCE_HANDLE: CudaError = 400;
pub const ILLEGAL_STATE: CudaError = 401;
pub const SYMBOL_NOT_FOUND: CudaError = 500;
pub const NOT_READY: CudaError = 600;
pub const ILLEGAL_ADDRESS: CudaError = 700;
pub const LAUNCH_OUT_OF_RESOURCES: CudaError = 701;
pub const LAUNCH_TIMEOUT: CudaError = 702;
pub const LAUNCH_FAILURE: CudaError = 719;
pub const NOT_PERMITTED: CudaError = 800;
pub const NOT_SUPPORTED: CudaError = 801;
pub const STREAM_CAPTURE_UNSUPPORTED: CudaError = 900;
pub const STREAM_CAPTURE_INVALIDATED: CudaError = 901;
pub const STREAM_CAPTURE_MERGE: CudaError = 902;
pub const STREAM_CAPTURE_UNMATCHED: CudaError = 903;
pub const STREAM_CAPTURE_UNJOINED: CudaError = 904;
pub const STREAM_CAPTURE_ISOLATION: CudaError = 905;
pub const STREAM_CAPTURE_IMPLICIT: CudaError = 906;
pub const CAPTURED_EVENT: CudaError = 907;
pub const STREAM_CAPTURE_WRONG_THREAD: CudaError = 908;
pub const UNKNOWN: CudaError = 999;

thread_local! { static LAST: Cell<CudaError> = const { Cell::new(SUCCESS) }; }

/// Record a sticky error (only non-success values are recorded; success never clears it).
pub fn record(e: CudaError) -> CudaError {
    if e != SUCCESS { LAST.with(|l| l.set(e)); }
    e
}
pub fn take_last() -> CudaError { LAST.with(|l| l.replace(SUCCESS)) }
pub fn peek_last() -> CudaError { LAST.with(|l| l.get()) }

pub fn name(e: CudaError) -> &'static str {
    match e {
        SUCCESS => "cudaSuccess",
        INVALID_VALUE => "cudaErrorInvalidValue",
        MEMORY_ALLOCATION => "cudaErrorMemoryAllocation",
        INITIALIZATION_ERROR => "cudaErrorInitializationError",
        INVALID_CONFIGURATION => "cudaErrorInvalidConfiguration",
        INVALID_PITCH_VALUE => "cudaErrorInvalidPitchValue",
        INVALID_SYMBOL => "cudaErrorInvalidSymbol",
        INVALID_HOST_POINTER => "cudaErrorInvalidHostPointer",
        INVALID_DEVICE_POINTER => "cudaErrorInvalidDevicePointer",
        INVALID_MEMCPY_DIRECTION => "cudaErrorInvalidMemcpyDirection",
        INVALID_DEVICE_FUNCTION => "cudaErrorInvalidDeviceFunction",
        NO_DEVICE => "cudaErrorNoDevice",
        INVALID_DEVICE => "cudaErrorInvalidDevice",
        INVALID_KERNEL_IMAGE => "cudaErrorInvalidKernelImage",
        UNSUPPORTED_LIMIT => "cudaErrorUnsupportedLimit",
        INVALID_RESOURCE_HANDLE => "cudaErrorInvalidResourceHandle",
        ILLEGAL_STATE => "cudaErrorIllegalState",
        SYMBOL_NOT_FOUND => "cudaErrorSymbolNotFound",
        NOT_READY => "cudaErrorNotReady",
        ILLEGAL_ADDRESS => "cudaErrorIllegalAddress",
        LAUNCH_OUT_OF_RESOURCES => "cudaErrorLaunchOutOfResources",
        LAUNCH_TIMEOUT => "cudaErrorLaunchTimeout",
        LAUNCH_FAILURE => "cudaErrorLaunchFailure",
        NOT_PERMITTED => "cudaErrorNotPermitted",
        NOT_SUPPORTED => "cudaErrorNotSupported",
        STREAM_CAPTURE_UNSUPPORTED => "cudaErrorStreamCaptureUnsupported",
        STREAM_CAPTURE_INVALIDATED => "cudaErrorStreamCaptureInvalidated",
        STREAM_CAPTURE_MERGE => "cudaErrorStreamCaptureMerge",
        STREAM_CAPTURE_UNMATCHED => "cudaErrorStreamCaptureUnmatched",
        STREAM_CAPTURE_UNJOINED => "cudaErrorStreamCaptureUnjoined",
        STREAM_CAPTURE_ISOLATION => "cudaErrorStreamCaptureIsolation",
        STREAM_CAPTURE_IMPLICIT => "cudaErrorStreamCaptureImplicit",
        CAPTURED_EVENT => "cudaErrorCapturedEvent",
        STREAM_CAPTURE_WRONG_THREAD => "cudaErrorStreamCaptureWrongThread",
        _ => "cudaErrorUnknown",
    }
}
pub fn string(e: CudaError) -> &'static str {
    match e {
        SUCCESS => "no error",
        INVALID_VALUE => "invalid argument",
        MEMORY_ALLOCATION => "out of memory",
        INITIALIZATION_ERROR => "initialization error",
        INVALID_CONFIGURATION => "invalid configuration argument",
        INVALID_PITCH_VALUE => "invalid pitch argument",
        INVALID_SYMBOL => "invalid device symbol",
        INVALID_HOST_POINTER => "invalid host pointer",
        INVALID_DEVICE_POINTER => "invalid device pointer",
        INVALID_MEMCPY_DIRECTION => "invalid copy direction for memcpy",
        INVALID_DEVICE_FUNCTION => "invalid device function",
        NO_DEVICE => "no CUDA-capable device is detected",
        INVALID_DEVICE => "invalid device ordinal",
        INVALID_KERNEL_IMAGE => "device kernel image is invalid",
        UNSUPPORTED_LIMIT => "limit is not supported on this architecture",
        INVALID_RESOURCE_HANDLE => "invalid resource handle",
        ILLEGAL_STATE => "the operation cannot be performed in the present state",
        SYMBOL_NOT_FOUND => "named symbol not found",
        NOT_READY => "device not ready",
        ILLEGAL_ADDRESS => "an illegal memory access was encountered",
        LAUNCH_OUT_OF_RESOURCES => "too many resources requested for launch",
        LAUNCH_TIMEOUT => "the launch timed out and was terminated",
        LAUNCH_FAILURE => "unspecified launch failure",
        NOT_PERMITTED => "operation not permitted",
        NOT_SUPPORTED => "operation not supported",
        STREAM_CAPTURE_UNSUPPORTED => "operation not permitted when stream is capturing",
        STREAM_CAPTURE_INVALIDATED => "operation failed due to a previous error during capture",
        STREAM_CAPTURE_MERGE => "operation would make the legacy stream depend on a capturing blocking stream",
        STREAM_CAPTURE_UNMATCHED => "capture was not initiated in this stream",
        STREAM_CAPTURE_UNJOINED => "capture sequence contains a fork that was not joined to the primary stream",
        STREAM_CAPTURE_ISOLATION => "dependency created on uncaptured work in another stream",
        STREAM_CAPTURE_IMPLICIT => "operation would make the legacy stream depend on a capturing blocking stream",
        CAPTURED_EVENT => "operation not permitted on an event last recorded in a capturing stream",
        STREAM_CAPTURE_WRONG_THREAD => "attempt to terminate a thread-local capture sequence from another thread",
        _ => "unknown error",
    }
}
