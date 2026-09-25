// mvcc-passes: clang pass plugins (-fpass-plugin) run at the start of clang's -O pipeline on the device TU:
// NVPTXConvergencePass (below) and NVPTXTensorInlinePass (end of file).
//
// clang compiles CUDA as a convergent language: every call, inline asm included, carries `convergent`, because
// clang cannot know whether the callee is a warp-collective (bar.sync, shfl.sync, mma.sync) or a per-thread
// instruction. The attribute forbids the transformations that add a control dependence to the call: the loop
// unroller declines a remainder loop, and refuses an upper-bound full unroll, for any loop whose body holds one.
// nvcc knows its own instruction set and unrolls `#pragma unroll` loops of cp.async copies with a per-thread trip
// count (`for (c = tid; c < N; c += THREADS)`) freely; under clang the same loops stay rolled and clang reports
// -Wpass-failed "loop not unrolled" for each. This pass gives clang the same knowledge: `convergent` comes off an
// inline asm whose every PTX instruction is a per-thread operation, and off libdevice (__nv_*) math calls, which
// are pure. Warp-collective asm (bar, barrier, shfl, vote, match, redux, activemask, elect, mma, wmma, ldmatrix,
// stmatrix, mbarrier ...) keeps the attribute.
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassBuilder.h"
#if __has_include("llvm/Plugins/PassPlugin.h")
#include "llvm/Plugins/PassPlugin.h"   // LLVM >= 23
#else
#include "llvm/Passes/PassPlugin.h"
#endif
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace llvm;

namespace {

// PTX instructions that act on the issuing thread alone (the opcode root, before the first '.').
static const StringSet<>& perThreadOps() {
  static const StringSet<> ops = {
      // data movement and addressing
      "ld", "st", "ldu", "mov", "cvta", "cvt", "isspacep", "prefetch", "prefetchu", "cp", "createpolicy",
      "applypriority", "discard",
      // integer and floating-point arithmetic
      "add", "sub", "mul", "mad", "mul24", "mad24", "sad", "div", "rem", "abs", "neg", "min", "max", "fma", "rcp",
      "sqrt", "rsqrt", "sin", "cos", "lg2", "ex2", "tanh", "popc", "clz", "bfind", "fns", "brev", "bfe", "bfi",
      "szext", "bmsk", "dp4a", "dp2a", "testp", "copysign",
      // comparison, selection, logic, shifts
      "set", "setp", "selp", "slct", "and", "or", "xor", "not", "cnot", "lop3", "shf", "shl", "shr", "prmt",
      // memory ordering and atomics: fences and atomics are not convergent (LLVM's own fence/atomicrmw are not)
      "membar", "fence", "atom", "red",
      // misc
      "nanosleep", "pmevent",
  };
  return ops;
}

// Every instruction of the asm text is a per-thread operation (an empty string, the `asm volatile("" ::: "memory")`
// compiler fence, qualifies). Instructions are separated by ';' or newlines; a leading `{`/`}` scope brace, a
// `@%p` predicate and a `LABEL:` are skipped to reach the opcode.
static bool asmIsPerThread(StringRef text) {
  SmallVector<StringRef, 8> lines;
  text.split(lines, '\n', -1, false);
  for (StringRef line : lines) {
    SmallVector<StringRef, 4> insts;
    line.split(insts, ';', -1, false);
    for (StringRef inst : insts) {
      inst = inst.trim(" \t\r{}");
      if (inst.empty()) continue;
      if (inst.starts_with("@")) {                  // predicate
        size_t sp = inst.find_first_of(" \t");
        if (sp == StringRef::npos) continue;
        inst = inst.substr(sp).ltrim();
      }
      StringRef op = inst.take_until([](char c) { return c == ' ' || c == '\t' || c == '.'; });
      if (op.ends_with(":")) return false;          // a label: control flow inside the asm, keep the attribute
      if (op.empty() || !perThreadOps().count(op)) return false;
    }
  }
  return true;
}

struct NVPTXConvergencePass : PassInfoMixin<NVPTXConvergencePass> {
  PreservedAnalyses run(Module& M, ModuleAnalysisManager&) {
    bool changed = false;
    for (Function& F : M) {
      // libdevice: __nv_* are pure math routines
      if (F.isDeclaration() && F.getName().starts_with("__nv_") && F.isConvergent()) { F.setNotConvergent(); changed = true; }
      for (Instruction& I : instructions(F)) {
        auto* CB = dyn_cast<CallBase>(&I);
        if (!CB || !CB->isConvergent()) continue;
        bool strip = false;
        if (auto* IA = dyn_cast<InlineAsm>(CB->getCalledOperand())) strip = asmIsPerThread(IA->getAsmString());
        else if (Function* Callee = CB->getCalledFunction()) strip = Callee->getName().starts_with("__nv_");
        if (!strip) continue;
        CB->removeFnAttr(Attribute::Convergent);
        changed = true;
      }
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

// The opcode root of a PTX instruction that is a tensor-core / staging primitive: the operations tensor recovery
// reads (mma, wmma, ldmatrix, stmatrix, cp.async) and the barriers that phase them.
static bool asmHoldsTensorPrimitive(StringRef text) {
  SmallVector<StringRef, 8> lines;
  text.split(lines, '\n', -1, false);
  for (StringRef line : lines) {
    SmallVector<StringRef, 4> insts;
    line.split(insts, ';', -1, false);
    for (StringRef inst : insts) {
      inst = inst.trim(" \t\r{}");
      if (inst.starts_with("@")) { size_t sp = inst.find_first_of(" \t"); if (sp == StringRef::npos) continue; inst = inst.substr(sp).ltrim(); }
      StringRef op = inst.take_until([](char c) { return c == ' ' || c == '\t'; });
      StringRef root = op.take_until([](char c) { return c == '.'; });
      if (root == "mma" || root == "wmma" || root == "ldmatrix" || root == "stmatrix") return true;
      if (root == "cp" && op.contains(".async")) return true;
    }
  }
  return false;
}

// nvcc inlines the device functions of a kernel's K loop; clang's inliner, costing the same body (a warp tile's worth
// of mma.sync, ldmatrix and cp.async per call, called from several places), can decline it. The tensor-core
// primitives then sit in a separate function, and tensor recovery — which reads the kernel — sees a K loop without
// matmuls (`f16_gemm_tc_bias_kernel<256, 128, ...>`: the K-step lambda, 64 mma.sync, 3 call sites, stayed a call).
// Every defined, non-kernel function whose body holds a tensor primitive, and every function that calls such a
// function, is marked `alwaysinline` (a function the source marked `noinline`, or one that is recursive, is left).
struct NVPTXTensorInlinePass : PassInfoMixin<NVPTXTensorInlinePass> {
  PreservedAnalyses run(Module& M, ModuleAnalysisManager&) {
    SmallPtrSet<Function*, 16> holds;
    for (Function& F : M) {
      if (F.isDeclaration() || F.getCallingConv() == CallingConv::PTX_Kernel) continue;
      if (F.hasFnAttribute(Attribute::NoInline)) continue;
      for (Instruction& I : instructions(F)) {
        auto* CB = dyn_cast<CallBase>(&I);
        if (!CB) continue;
        if (auto* IA = dyn_cast<InlineAsm>(CB->getCalledOperand())) if (asmHoldsTensorPrimitive(IA->getAsmString())) { holds.insert(&F); break; }
      }
    }
    // the callers of a holder hold too, up to the kernels
    bool grew = true;
    while (grew) {
      grew = false;
      for (Function* F : SmallVector<Function*, 16>(holds.begin(), holds.end()))
        for (User* U : F->users()) {
          auto* CB = dyn_cast<CallBase>(U);
          if (!CB || CB->getCalledFunction() != F) continue;
          Function* P = CB->getFunction();
          if (P->getCallingConv() == CallingConv::PTX_Kernel || P->hasFnAttribute(Attribute::NoInline)) continue;
          if (holds.insert(P).second) grew = true;
        }
    }
    bool changed = false;
    for (Function* F : holds) {
      bool recursive = false;
      for (Instruction& I : instructions(*F)) if (auto* CB = dyn_cast<CallBase>(&I)) if (CB->getCalledFunction() == F) recursive = true;
      if (recursive || F->hasFnAttribute(Attribute::AlwaysInline)) continue;
      F->removeFnAttr(Attribute::InlineHint);
      F->removeFnAttr(Attribute::OptimizeNone);
      F->addFnAttr(Attribute::AlwaysInline);
      changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
  static bool isRequired() { return true; }
};

}  // namespace

extern "C" LLVM_ATTRIBUTE_WEAK ::llvm::PassPluginLibraryInfo llvmGetPassPluginInfo() {
  return {LLVM_PLUGIN_API_VERSION, "mvcc-passes", LLVM_VERSION_STRING, [](PassBuilder& PB) {
            PB.registerPipelineStartEPCallback([](ModulePassManager& MPM, OptimizationLevel) {
              MPM.addPass(NVPTXConvergencePass());
              MPM.addPass(NVPTXTensorInlinePass());
            });
            PB.registerPipelineParsingCallback([](StringRef name, ModulePassManager& MPM, ArrayRef<PassBuilder::PipelineElement>) {
              if (name == "mvcc-nvptx-convergence") { MPM.addPass(NVPTXConvergencePass()); return true; }
              if (name == "mvcc-nvptx-tensor-inline") { MPM.addPass(NVPTXTensorInlinePass()); return true; }
              return false;
            });
          }};
}
