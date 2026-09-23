// mvcc-ir2msl: device LLVM IR (.ll/.bc from clang -x cuda --cuda-device-only) -> MSL + kernel ABI JSON.
#include "msl_emitter.h"
#include "engine.h"
#include "recovery_log.h"

#include <llvm/IR/CallingConv.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/CommandLine.h>
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/Program.h>
#include <llvm/Support/raw_ostream.h>

using namespace llvm;

static constexpr const char* LogPrefix = "[MVCC] ";

static cl::opt<std::string> InputFile(cl::Positional, cl::desc("<input .ll/.bc>"), cl::Required);
static cl::opt<std::string> OutMSL("o", cl::desc("output .metal file"), cl::value_desc("file"), cl::Required);
static cl::opt<std::string> OutABI("abi", cl::desc("output ABI .json file"), cl::value_desc("file"));
static cl::opt<bool> StateMachine("state-machine", cl::desc("use switch state machine for all control flow (debug)"));
static cl::opt<bool> FastMath("fast-math", cl::desc("lower math to metal::fast::"));
static cl::opt<bool> Verbose("v", cl::desc("verbose"));
static cl::opt<std::string> DumpIR("dump-ir", cl::desc("write normalized IR to file"), cl::value_desc("file"));

static std::string jsonEscape(const std::string& s) {
  std::string o;
  for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; }
  return o;
}

int main(int argc, char** argv) {
  InitLLVM X(argc, argv);
  for (int i = 1; i < argc; i++) {
    if (std::string(argv[i]) == "--engine-self-test") return mvcc::engine::selfTest(outs());
    if (std::string(argv[i]) == "--sched-diff") {
      if (i + 2 >= argc) { errs() << LogPrefix << "usage: --sched-diff <sched_a> <sched_b>\n"; return 2; }
      return mvcc::engine::schedDiff(argv[i + 1], argv[i + 2], outs());
    }
    if (std::string(argv[i]) == "--stir-in") {
      if (i + 1 >= argc) { errs() << LogPrefix << "usage: --stir-in elementwise|gemm|attention|<file.stir> [--sched-in <file.sched>]\n"; return 2; }
      std::string sched;
      if (i + 3 < argc && std::string(argv[i + 2]) == "--sched-in") sched = argv[i + 3];
      return mvcc::engine::stirIn(argv[i + 1], outs(), sched);
    }
  }
  cl::ParseCommandLineOptions(argc, argv, "mvcc IR to MSL\n");

  LLVMContext Ctx;
  SMDiagnostic Err;
  std::unique_ptr<Module> M = parseIRFile(InputFile, Err, Ctx);
  if (!M) { Err.print(argv[0], errs()); return 1; }

  mvcc::EmitOptions opts;
  opts.structured = !StateMachine;
  opts.verbose = Verbose;
  opts.fastMath = FastMath;
  if (const char* e = getenv("MVCC_TENSOR_DIAG")) opts.tensorDiag = std::string(e) == "1";
  if (Verbose) opts.tensorDiag = true;
  bool recoveryLogVerbose = Verbose || opts.tensorDiag;
  if (const char* sig = getenv("MVCC_SIG"); sig && sig[0] != '0') recoveryLogVerbose = true;
  mvcc::EmitResult res;
  bool ok = mvcc::emitModule(*M, opts, res);
  for (auto& w : res.warnings) errs() << LogPrefix << "warning: " << w << "\n";
  for (auto& n : res.notes)
    if (mvcc::shouldLogRecoveryNote(n, recoveryLogVerbose)) errs() << LogPrefix << "note: " << n << "\n";
  if (!DumpIR.empty()) {
    std::error_code EC; raw_fd_ostream os(DumpIR, EC, sys::fs::OF_Text);
    if (!EC) M->print(os, nullptr);
  }
  if (const char* ora = getenv("MVCC_NVPTX_ORACLE")) {
    std::error_code EC; raw_fd_ostream os(ora, EC, sys::fs::OF_Text);
    if (!EC) {
      M->print(os, nullptr);
      errs() << LogPrefix << "note: nvptx-oracle: wrote recovered LLVM to " << ora << "\n";
    }
    // Recovered LLVM → PTX via llc. Device match is tests/nvptx_oracle_pin.sh.
    std::string ptx = ora;
    if (ptx.size() > 3 && ptx.substr(ptx.size() - 3) == ".ll") ptx.replace(ptx.size() - 3, 3, ".ptx");
    else ptx += ".ptx";
#ifdef MVCC_LLC
    llvm::SmallVector<llvm::StringRef, 8> args{
      MVCC_LLC, "-march=nvptx64", "-mcpu=sm_89", "-o", ptx, ora};
    std::string err;
    int rc = llvm::sys::ExecuteAndWait(MVCC_LLC, args, std::nullopt, {}, 0, 0, &err);
    if (rc == 0)
      errs() << LogPrefix << "note: nvptx-oracle: wrote PTX to " << ptx
             << " (llc dump; not a NVIDIA device run)\n";
    else
      errs() << LogPrefix << "note: nvptx-oracle: llc PTX back-out skipped (" << err << " rc=" << rc << ")\n";
#else
    raw_fd_ostream pos(ptx, EC, sys::fs::OF_Text);
    if (!EC) {
      pos << ".version 8.0\n.target sm_89\n.address_size 64\n";
      for (auto& F : *M) if (!F.isDeclaration() && F.getCallingConv() == CallingConv::PTX_Kernel)
        pos << ".visible .entry " << F.getName() << "() { ret; }\n";
      errs() << LogPrefix << "note: nvptx-oracle: wrote PTX stub to " << ptx << " (llc not linked)\n";
    }
#endif
  }
  if (!ok) { errs() << LogPrefix << "error: " << res.error << "\n"; return 1; }

  {
    std::error_code EC; raw_fd_ostream os(OutMSL, EC, sys::fs::OF_Text);
    if (EC) { errs() << LogPrefix << "error: cannot write " << OutMSL << ": " << EC.message() << "\n"; return 1; }
    os << res.msl;
  }
  if (!OutABI.empty()) {
    std::error_code EC; raw_fd_ostream os(OutABI, EC, sys::fs::OF_Text);
    if (EC) { errs() << LogPrefix << "error: cannot write " << OutABI << ": " << EC.message() << "\n"; return 1; }
    os << "{\n  \"kernels\": [\n";
    for (size_t i = 0; i < res.kernels.size(); i++) {
      auto& k = res.kernels[i];
      os << "    {\"name\": \"" << jsonEscape(k.name) << "\", \"param_block_size\": " << k.paramBlockSize
         << ", \"static_smem\": " << k.staticSmem << ", \"dynamic_smem\": " << (k.dynamicSmem ? "true" : "false")
         << ", \"max_threads\": " << k.maxThreads << ", \"params\": [";
      for (size_t j = 0; j < k.params.size(); j++) {
        auto& p = k.params[j];
        os << (j ? ", " : "") << "{\"kind\": \"" << p.kind << "\", \"size\": " << p.size << ", \"align\": " << p.align << ", \"offset\": " << p.offset << "}";
      }
      os << "]";
      if (!k.exactVariant.empty()) os << ", \"exact_variant\": \"" << jsonEscape(k.exactVariant) << "\"";
      if (!k.searchVariant.empty()) os << ", \"search_variant\": \"" << jsonEscape(k.searchVariant) << "\"";
      if (k.isVariant) os << ", \"is_variant\": true";
      if (k.recoveredMaxThreads) os << ", \"recovered_max_threads\": " << k.recoveredMaxThreads;
      if (k.stagingThreads) os << ", \"staging_threads\": " << k.stagingThreads;
      if (k.threadScale > 1) os << ", \"thread_scale\": " << k.threadScale;
      if (k.dynSmemUsed) os << ", \"dyn_smem_used\": " << k.dynSmemUsed;
      if (k.dynSmemShrink) os << ", \"dyn_smem_cut\": " << k.dynSmemCut << ", \"dyn_smem_shrink\": " << k.dynSmemShrink;
      if (!k.assume.empty()) { os << ", \"assume\": ["; for (size_t j = 0; j < k.assume.size(); j++) os << (j ? ", " : "") << "\"" << jsonEscape(k.assume[j]) << "\""; os << "]"; }
      if (k.zeroPage) os << ", \"zero_page\": true";
      if (k.lockPage) os << ", \"lock_page\": true";
      if (k.globals) os << ", \"globals\": true";
      if (!k.deviceGraphLaunchExecOffsets.empty()) {
        os << ", \"device_graph_launch_exec_offsets\": [";
        for (size_t j = 0; j < k.deviceGraphLaunchExecOffsets.size(); j++)
          os << (j ? ", " : "") << k.deviceGraphLaunchExecOffsets[j];
        os << "]";
      }
      os << "}" << (i + 1 < res.kernels.size() ? "," : "") << "\n";
    }
    os << "  ]";
    if (!res.collectives.empty()) {
      os << ",\n  \"collectives\": [\n";
      for (size_t i = 0; i < res.collectives.size(); i++) {
        auto& c = res.collectives[i];
        os << "    {\"kind\": \"" << jsonEscape(c.kind) << "\", \"world\": " << c.world
           << ", \"dtype\": \"" << jsonEscape(c.dtype.empty() ? "copy" : c.dtype) << "\"}"
           << (i + 1 < res.collectives.size() ? "," : "") << "\n";
      }
      os << "  ]";
    }
    if (!res.verifyKernels.empty()) {
      os << ",\n  \"verify_kernels\": [\n";
      for (size_t i = 0; i < res.verifyKernels.size(); i++) {
        auto& v = res.verifyKernels[i];
        os << "    {\"name\": \"" << v.name << "\", \"m\": " << v.M << ", \"n\": " << v.N << ", \"k\": " << v.K << ", \"tl\": " << (v.tl ? "true" : "false")
           << ", \"tr\": " << (v.tr ? "true" : "false") << ", \"type\": \"" << (v.type == 0 ? "f16" : "bf16") << "\"}" << (i + 1 < res.verifyKernels.size() ? "," : "") << "\n";
      }
      os << "  ]";
    }
    if (res.globals.size) {
      auto& g = res.globals;
      os << ",\n  \"globals\": {\"size\": " << g.size << ", \"align\": " << g.align << ", \"init\": \"";
      static const char* hex = "0123456789abcdef";
      for (uint8_t b : g.init) { os << hex[b >> 4] << hex[b & 15]; }
      os << "\", \"relocs\": [";
      for (size_t i = 0; i < g.relocs.size(); i++) os << (i ? ", " : "") << "[" << g.relocs[i].first << ", " << g.relocs[i].second << "]";
      os << "], \"symbols\": [";
      for (size_t i = 0; i < g.symbols.size(); i++) os << (i ? ", " : "") << "{\"name\": \"" << jsonEscape(g.symbols[i].name) << "\", \"offset\": " << g.symbols[i].offset << ", \"size\": " << g.symbols[i].size << "}";
      os << "]}";
    }
    os << "\n}\n";
  }
  return 0;
}
