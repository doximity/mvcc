// LLVM IR (nvptx64 device module, post-normalization) -> Metal Shading Language.
#pragma once
#include <llvm/IR/Module.h>
#include <string>
#include <vector>

namespace mvcc {

struct ParamABI {
  std::string kind;   // "ptr" | "i8" | "i16" | "i32" | "i64" | "f16" | "bf16" | "f32" | "struct"
  unsigned size = 0;
  unsigned align = 0;
  unsigned offset = 0;   // byte offset in the packed argument block
};

struct KernelABI {
  std::string name;              // mangled symbol name (what __cudaRegisterFunction gets)
  std::vector<ParamABI> params;
  unsigned paramBlockSize = 0;
  unsigned staticSmem = 0;       // bytes of statically declared __shared__
  bool dynamicSmem = false;      // uses extern __shared__
  unsigned maxThreads = 0;       // __launch_bounds__ max, 0 if none
  unsigned smemAlign = 16;
  std::string exactVariant;      // tensor recovery: name of the exact-semantics twin kernel, if this one was rewritten
  std::string searchVariant;     // extra legal body <name>__mvcc_s0; runtime may pin it
  bool isVariant = false;        // this kernel is such a twin (not registered by the host; launched by substitution)
  unsigned stagingThreads = 0;   // recovered body assumed this 1-D block size; other blocks run the twin
  unsigned threadScale = 0;      // body runs threadScale × host block.x (simdgroup copies); 0/1 = none
  unsigned dynSmemUsed = 0;      // exact footprint of the dynamic shared object after layout recovery (0 = as requested by the host)
  unsigned recoveredMaxThreads = 0;  // recovered body's conversion scratch is sized for this many threads; larger blocks run the twin
  unsigned dynSmemCut = 0, dynSmemShrink = 0;  // decode retiling: dynamic bytes from dynSmemCut on live dynSmemShrink lower; the runtime allocates request - shrink
  std::vector<std::string> assume;   // conditions over the arguments the body assumes ("(=> premise conclusion)"); a failing one runs the twin
  bool zeroPage = false;         // takes the runtime's 4 KiB zero page at [[buffer(1)]] (guarded loads made speculatable)
  bool lockPage = false;         // takes the runtime's 64 KiB lock table at [[buffer(4)]] (64-bit atomics)
  bool globals = false;          // takes the module's globals buffer at [[buffer(2)]] (constant view) and [[buffer(3)]] (device view)
  /// Byte offsets in the packed param block of cudaGraphExec_t handles to replay on the launching stream
  /// (in call order) before the kernel dispatch — lowered from __mvcc_device_graph_launch in the body.
  std::vector<unsigned> deviceGraphLaunchExecOffsets;
};

// Module-scope variables the runtime materializes in one device buffer:
// every __device__ variable, and every __constant__/const table that is reached through a pointer stored in data
// (those need runtime relocation). Tables that kernels only index directly stay program-scope `constant` data.
struct GlobalsABI {
  unsigned size = 0;                    // bytes (0 = the module has no buffered globals)
  unsigned align = 16;
  std::vector<uint8_t> init;            // initial contents, relocation slots zero
  std::vector<std::pair<unsigned, unsigned>> relocs;  // (slot offset, target offset): slot <- buffer_gpu_address + target
  struct Symbol { std::string name; unsigned offset; unsigned size; };
  std::vector<Symbol> symbols;          // by mangled name (what __cudaRegisterVar reports)
};

// A TensorOps verification kernel the runtime runs once per process before trusting recovered kernels.
struct VerifyKernelABI {
  std::string name;              // __mvcc_tp_verify_<tag>
  int M = 0, N = 0, K = 0;
  bool tl = false, tr = false;
  int type = 0;                  // 0 f16, 1 bf16
};

struct EmitOptions {
  bool structured = true;        // structured control flow; false = switch state machine
  bool verbose = false;
  bool fastMath = false;         // emit metal::fast:: for math functions (nvcc --use_fast_math)
  bool allowGenericAsDevice = true;
  bool tensorRecovery = true;
  bool tensorDiag = false;       // verbose recovery notes on stderr (MVCC_TENSOR_DIAG=1 or -v; default is declines/warnings only)
  bool guardedLoads = true;
  int tpGroupM = 2, tpGroupN = 2, tpGroupK = 0;  // fusion caps, see TensorRecoveryOptions
};

struct CollectiveABI {
  std::string kind;     // "allgather"
  unsigned world = 1;
  std::string dtype;    // "copy" (reductions refused)
};

struct EmitResult {
  std::string msl;
  std::vector<KernelABI> kernels;
  std::vector<VerifyKernelABI> verifyKernels;
  GlobalsABI globals;
  std::vector<CollectiveABI> collectives; // sequence the runtime must run (NCCL copy)
  std::vector<std::string> warnings;
  std::vector<std::string> notes;
  std::string error;
};

// Runs normalization passes on M (mutates it), then emits MSL text.
bool emitModule(llvm::Module& M, const EmitOptions& opts, EmitResult& out);

} // namespace mvcc
