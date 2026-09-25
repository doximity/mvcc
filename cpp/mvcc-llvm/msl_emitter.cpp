// LLVM IR -> MSL emitter.
//
// Design:
//  * All non-kernel functions are force-inlined; only kernels are emitted.
//  * Pointers are byte pointers qualified by address space (device/threadgroup/constant/thread).
//    GEPs become byte arithmetic; loads/stores reinterpret to the accessed type.
//  * Integer SSA values use unsigned MSL types; signed operations cast at the use.
//  * SSA values used from another block (phis, loop-carried, values that escape a structured
//    region) are declared at the top of the kernel. Same-block temps are declared at first
//    assign so Metal does not treat thousands of decode-GEMV SSA names as kernel-live
//    (that register budget leaves one 64-thread group per core).
//  * Control flow: StructurizeCFG + recursive dominator/post-dominator emission.
//    If a function's CFG defeats the structured emitter, fall back to a switch state machine.
//  * NVVM intrinsics, libdevice (__nv_*) calls, and inline PTX are lowered via tables;
//    heavy lifting (mma, ldmatrix) is delegated to helpers in the MSL prelude.
#include "msl_emitter.h"
#include "llvm/Support/raw_ostream.h"
#include "ptx_asm.h"
#include "recovery_log.h"
#include "tensor_recovery.h"
#include "wide_vectors.h"
#include "staging_elim.h"

#include <algorithm>
#include <llvm/ADT/DenseMap.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/ADT/SmallVector.h>
#include <llvm/ADT/StringExtras.h>
#include <llvm/Analysis/LoopInfo.h>
#include <llvm/Analysis/PostDominators.h>
#include <llvm/IR/CFG.h>
#include <llvm/IR/ConstantRange.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/Dominators.h>
#include <llvm/IR/GetElementPtrTypeIterator.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/InlineAsm.h>
#include <llvm/IR/InstIterator.h>
#include <llvm/IR/Instructions.h>
#include <llvm/IR/IntrinsicInst.h>
#include <llvm/IR/Operator.h>
#include <llvm/ADT/MapVector.h>
#include <llvm/Support/MathExtras.h>
#include <llvm/IR/PassManager.h>
#include <llvm/IR/Verifier.h>
#include <llvm/Transforms/Utils/Local.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar/SROA.h>
#include <llvm/Transforms/Scalar/DCE.h>
#include <llvm/Transforms/Scalar/InferAddressSpaces.h>
#include <llvm/Demangle/Demangle.h>
#include <llvm/Transforms/Scalar/LoopSimplifyCFG.h>
#include <llvm/Transforms/Utils/LowerSwitch.h>
#include <llvm/Transforms/Scalar/SimplifyCFG.h>
#include <llvm/Transforms/Scalar/JumpThreading.h>
#include <llvm/Transforms/Scalar/EarlyCSE.h>
#include <llvm/Transforms/Scalar/StructurizeCFG.h>
#include <llvm/Transforms/Utils/FixIrreducible.h>
#include <llvm/Transforms/Utils/LoopSimplify.h>
#include <llvm/Transforms/Utils/UnifyLoopExits.h>
#include <llvm/Transforms/Vectorize/LoadStoreVectorizer.h>
#include <llvm/Transforms/Vectorize/SLPVectorizer.h>
#include <llvm/Analysis/TargetTransformInfo.h>
#include <llvm/Analysis/TargetTransformInfoImpl.h>

#include <functional>
#include <map>
#include <set>
#include <sstream>

using namespace llvm;

namespace mvcc {
namespace {

struct EmitError { std::string msg; };
[[noreturn]] void fail(const std::string& m) { throw EmitError{m}; }

bool isKernel(const Function& F) {
  if (F.getCallingConv() == CallingConv::PTX_Kernel) return true;
  if (F.hasFnAttribute("nvvm.kernel")) return true;
  if (auto* MD = F.getParent()->getNamedMetadata("nvvm.annotations")) {
    for (auto* Op : MD->operands()) {
      if (Op->getNumOperands() < 3) continue;
      auto* V = mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
      auto* S = dyn_cast<MDString>(Op->getOperand(1));
      if (V == &F && S && S->getString() == "kernel") return true;
    }
  }
  return false;
}

}  // namespace

unsigned kernelMaxThreads(const Function& F) {
  // LLVM 23 carries launch bounds as function attributes; older clang used nvvm.annotations.
  if (F.hasFnAttribute("nvvm.maxntid")) {
    StringRef s = F.getFnAttribute("nvvm.maxntid").getValueAsString();
    unsigned prod = 1;
    SmallVector<StringRef, 3> parts;
    s.split(parts, ',');
    for (auto p : parts) { unsigned v = 0; p.getAsInteger(10, v); if (v) prod *= v; }
    return prod;
  }
  unsigned prod = 0;
  if (auto* MD = F.getParent()->getNamedMetadata("nvvm.annotations")) {
    for (auto* Op : MD->operands()) {
      if (Op->getNumOperands() < 3) continue;
      auto* V = mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
      if (V != &F) continue;
      for (unsigned i = 1; i + 1 < Op->getNumOperands(); i += 2) {
        auto* S = dyn_cast<MDString>(Op->getOperand(i));
        auto* C = mdconst::dyn_extract_or_null<ConstantInt>(Op->getOperand(i + 1));
        if (!S || !C) continue;
        if (S->getString() == "maxntidx" || S->getString() == "maxntidy" || S->getString() == "maxntidz")
          prod = (prod ? prod : 1) * (unsigned)C->getZExtValue();
      }
    }
  }
  return prod;
}

namespace {

// ---------------------------------------------------------------------------
// Normalization passes
// ---------------------------------------------------------------------------

void kernelPointerArgsToGlobal(Function& F) {
  // Kernel pointer parameters point at device memory. Tag them so InferAddressSpaces
  // can propagate the fact through the whole kernel (mirrors NVPTXLowerArgs).
  if (F.empty()) return;
  IRBuilder<> B(&F.getEntryBlock(), F.getEntryBlock().getFirstNonPHIOrDbgOrAlloca());
  for (Argument& A : F.args()) {
    auto* PT = dyn_cast<PointerType>(A.getType());
    if (!PT || PT->getAddressSpace() != 0) continue;
    if (A.hasByValAttr()) continue;  // byval structs live in param space; handled at prologue
    Value* g = B.CreateAddrSpaceCast(&A, PointerType::get(F.getContext(), 1), A.getName() + ".global");
    Value* back = B.CreateAddrSpaceCast(g, A.getType(), A.getName() + ".gen");
    A.replaceUsesWithIf(back, [&](Use& U) { return U.getUser() != g; });
  }
}

// Static allocas are thread-private. Route their uses through addrspace(5) so InferAddressSpaces
// propagates `thread` to every derived load/store/GEP (mirrors NVPTXLowerAlloca).
void allocasToLocal(Function& F) {
  SmallVector<AllocaInst*, 8> allocas;
  for (Instruction& I : F.getEntryBlock()) if (auto* AI = dyn_cast<AllocaInst>(&I)) allocas.push_back(AI);
  for (AllocaInst* AI : allocas) {
    if (AI->getType()->getAddressSpace() != 0) continue;
    IRBuilder<> B(AI->getNextNode());
    Value* loc = B.CreateAddrSpaceCast(AI, PointerType::get(F.getContext(), 5), AI->getName() + ".local");
    Value* back = B.CreateAddrSpaceCast(loc, AI->getType(), AI->getName() + ".gen");
    AI->replaceUsesWithIf(back, [&](Use& U) { return U.getUser() != loc; });
  }
}

// A `phi ptr` whose incoming pointers live in different address spaces (e.g. `c ? &local[i] : &global[j]`
// hoisted by clang) has no MSL spelling. When every user is a plain load, sink the loads into the
// predecessors and phi the loaded values instead. Runs between the InferAddressSpaces passes.
struct SplitMixedPtrPhiPass : PassInfoMixin<SplitMixedPtrPhiPass> {
  static unsigned spaceOf(Value* V) {
    V = V->stripPointerCasts();
    if (auto* ASC = dyn_cast<AddrSpaceCastInst>(V)) return ASC->getSrcAddressSpace();
    if (auto* CE = dyn_cast<ConstantExpr>(V)) if (CE->getOpcode() == Instruction::AddrSpaceCast) return CE->getOperand(0)->getType()->getPointerAddressSpace();
    if (isa<AllocaInst>(V)) return 5;
    return V->getType()->getPointerAddressSpace();
  }
  PreservedAnalyses run(Function& F, FunctionAnalysisManager&) {
    bool changed = false;
    SmallVector<PHINode*, 8> work;
    for (BasicBlock& BB : F)
      for (PHINode& PN : BB.phis()) {
        if (!PN.getType()->isPointerTy()) continue;
        unsigned s0 = spaceOf(PN.getIncomingValue(0));
        bool mixed = false;
        for (unsigned i = 1; i < PN.getNumIncomingValues(); ++i) if (spaceOf(PN.getIncomingValue(i)) != s0) mixed = true;
        if (!mixed) continue;
        bool loadsOnly = !PN.use_empty();
        for (User* U : PN.users()) {
          auto* L = dyn_cast<LoadInst>(U);
          if (!L || L->isVolatile() || L->getPointerOperand() != &PN || L->getParent() != &BB) { loadsOnly = false; break; }
          // nothing may write memory between the block entry and the load, or sinking changes what is read
          for (Instruction& I : BB) { if (&I == L) break; if (!isa<PHINode>(I) && I.mayWriteToMemory()) { loadsOnly = false; break; } }
        }
        if (loadsOnly) work.push_back(&PN);
      }
    for (PHINode* PN : work) {
      SmallVector<LoadInst*, 4> loads;
      for (User* U : PN->users()) loads.push_back(cast<LoadInst>(U));
      for (LoadInst* L : loads) {
        PHINode* VP = PHINode::Create(L->getType(), PN->getNumIncomingValues(), L->getName() + ".sunk", PN->getIterator());
        for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i) {
          BasicBlock* Pred = PN->getIncomingBlock(i);
          IRBuilder<> B(Pred->getTerminator());
          LoadInst* NL = B.CreateAlignedLoad(L->getType(), PN->getIncomingValue(i), L->getAlign(), L->getName());
          NL->copyMetadata(*L);
          VP->addIncoming(NL, Pred);
        }
        L->replaceAllUsesWith(VP);
        L->eraseFromParent();
      }
      PN->eraseFromParent();
      changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// CUDA requires every load/store to be naturally aligned for its type (misaligned accesses fault at run time), and
// NVIDIA's compiler emits the natural-width instruction regardless of what the IR's `align` says. Clang often derives
// a weaker alignment through reinterpret_casts (e.g. a `uint32_t` view of a `__half` array gets align 2); adopt the
// CUDA semantics so the emitter and the vectorizer see the alignment the program is actually entitled to.
//
// `align 1` on a multi-byte access is the one alignment the program is *not* entitled to raise: it is what a small
// `memcpy` (e.g. `memcpy(&sig, bytes, 8)` on a `uint8_t*`) or a packed-struct member lowers to, and on NVIDIA those
// are byte accesses that work at any address. Widening them makes Metal round the address down (a `ulong` load at
// `p + 3` reads `p`), so they stay at align 1 and the emitter assembles them bytewise.
struct CudaAlignmentPass : PassInfoMixin<CudaAlignmentPass> {
  static bool keep(Align a, Type* T) { return a == Align(1) && T->isSized() && !T->isIntegerTy(8) && !T->isIntegerTy(1); }
  PreservedAnalyses run(Function& F, FunctionAnalysisManager&) {
    const DataLayout& DL = F.getParent()->getDataLayout();
    bool changed = false;
    for (BasicBlock& BB : F)
      for (Instruction& I : BB) {
        if (auto* L = dyn_cast<LoadInst>(&I)) {
          if (keep(L->getAlign(), L->getType())) continue;
          Align nat = DL.getABITypeAlign(L->getType());
          if (L->getAlign() < nat) { L->setAlignment(nat); changed = true; }
        } else if (auto* S = dyn_cast<StoreInst>(&I)) {
          if (keep(S->getAlign(), S->getValueOperand()->getType())) continue;
          Align nat = DL.getABITypeAlign(S->getValueOperand()->getType());
          if (S->getAlign() < nat) { S->setAlignment(nat); changed = true; }
        }
      }
    (void)changed;  // alignment metadata only: no analysis depends on it
    return PreservedAnalyses::all();
  }
};

// Target model for LLVM's load/store vectorizer: Metal has packed vector types, so a vector access only needs its
// element alignment (the emitter picks packed_T for under-aligned vectors), and 128-bit accesses are the widest.
struct MvccTTIImpl : TargetTransformInfoImplCRTPBase<MvccTTIImpl> {
  explicit MvccTTIImpl(const DataLayout& DL) : TargetTransformInfoImplCRTPBase<MvccTTIImpl>(DL) {}
  bool allowsMisalignedMemoryAccesses(LLVMContext&, unsigned BitWidth, unsigned, Align Alignment, unsigned* Fast) const override {
    (void)BitWidth;
    if (Fast) *Fast = 1;  // packed vector access is a single instruction
    return Alignment.value() >= 4;
  }
  unsigned getLoadStoreVecRegBitWidth(unsigned) const override { return 128; }
};

// Decode DOT loops are `p = fmaf(w[i], a[i], p)` for i in 0..7. SLP will not break a dependent fma chain;
// Metal's `dot(float4, float4)` is the instruction that product is. Walk unique-use fma/fmuladd chains
// in one block and replace each group of 2 or 4 with `__mvcc_dot4` / `__mvcc_dot2`. Opt-in: MVCC_DOT_PACK=1.
struct FmaDotPackPass : PassInfoMixin<FmaDotPackPass> {
  static bool asFma(Instruction* I, Value*& x, Value*& y, Value*& acc) {
    if (auto* BO = dyn_cast<BinaryOperator>(I)) {
      if (BO->getOpcode() == Instruction::FAdd && BO->getType()->isFloatTy()) {
        auto take = [&](Value* mulV, Value* other) {
          auto* M = dyn_cast<BinaryOperator>(mulV);
          if (!M || M->getOpcode() != Instruction::FMul || !M->hasOneUse() || !M->getType()->isFloatTy()) return false;
          x = M->getOperand(0); y = M->getOperand(1); acc = other; return true;
        };
        if (take(BO->getOperand(0), BO->getOperand(1)) || take(BO->getOperand(1), BO->getOperand(0))) return true;
      }
    }
    auto* CI = dyn_cast<CallInst>(I);
    if (!CI || !CI->getType()->isFloatTy() || CI->arg_size() < 3) return false;
    StringRef n;
    if (Function* Fn = CI->getCalledFunction()) n = Fn->getName();
    bool ok = false;
    if (auto* II = dyn_cast<IntrinsicInst>(CI)) {
      auto id = II->getIntrinsicID();
      ok = id == Intrinsic::fma || id == Intrinsic::fmuladd;
    }
    if (!ok) {
      ok = n.starts_with("llvm.nvvm.fma.rn.f") || n.starts_with("llvm.fma.") || n.starts_with("llvm.fmuladd.")
        || n == "__nv_fmaf" || n == "__nv_fmaf_rn" || n == "__nv_fma";
    }
    if (!ok) return false;
    x = CI->getArgOperand(0);
    y = CI->getArgOperand(1);
    acc = CI->getArgOperand(2);
    return x->getType()->isFloatTy() && !x->getType()->isVectorTy();
  }

  static Value* packDot(IRBuilder<>& B, ArrayRef<Value*> xs, ArrayRef<Value*> ys, Value* acc) {
    const unsigned n = xs.size();
    Type* F32 = Type::getFloatTy(B.getContext());
    SmallVector<Type*, 8> tys(n * 2, F32);
    const char* name = n == 4 ? "__mvcc_dot4" : "__mvcc_dot2";
    FunctionCallee Callee = B.GetInsertBlock()->getModule()->getOrInsertFunction(name, FunctionType::get(F32, tys, false));
    if (auto* Fn = dyn_cast<Function>(Callee.getCallee())) {
      Fn->setDoesNotAccessMemory();
      Fn->setDoesNotThrow();
    }
    SmallVector<Value*, 8> args;
    args.append(xs.begin(), xs.end());
    args.append(ys.begin(), ys.end());
    Value* d = B.CreateCall(Callee, args);
    if (auto* CFP = dyn_cast<ConstantFP>(acc); CFP && CFP->isZero()) return d;
    return B.CreateFAdd(acc, d);
  }

  PreservedAnalyses run(Function& F, FunctionAnalysisManager&) {
    return PreservedAnalyses::all();
    bool changed = false;
    for (BasicBlock& BB : F) {
      SmallVector<Instruction*, 32> insts;
      for (Instruction& I : BB) insts.push_back(&I);
      SmallPtrSet<Instruction*, 16> packed;
      for (Instruction* IP : llvm::reverse(insts)) {
        if (!IP->getParent() || packed.count(IP)) continue;
        Instruction& I = *IP;
        Value *x, *y, *acc;
        if (!asFma(&I, x, y, acc)) continue;
        struct Term { Instruction* I; Value* x; Value* y; };
        SmallVector<Term, 8> chain;
        Instruction* cur = &I;
        Value *cx = x, *cy = y, *ca = acc;
        while (cur) {
          chain.push_back({cur, cx, cy});
          auto* prev = dyn_cast<Instruction>(ca);
          Value *px, *py, *pacc;
          if (!prev || prev->getParent() != &BB || !prev->hasOneUse() || !asFma(prev, px, py, pacc)) break;
          cur = prev;
          cx = px;
          cy = py;
          ca = pacc;
        }
        // chain[0] is the sink; chain.back() is the oldest fma. Walk oldest→newest.
        std::reverse(chain.begin(), chain.end());
        if (chain.size() < 2) continue;
        Value* run = ([&]() {
          Value *ox, *oy, *oa;
          asFma(chain.front().I, ox, oy, oa);
          return oa;
        })();
        IRBuilder<> B(chain.back().I);
        unsigned i = 0;
        while (i + 4 <= chain.size()) {
          run = packDot(B, {chain[i].x, chain[i + 1].x, chain[i + 2].x, chain[i + 3].x},
                           {chain[i].y, chain[i + 1].y, chain[i + 2].y, chain[i + 3].y}, run);
          i += 4;
        }
        if (i + 2 <= chain.size()) {
          run = packDot(B, {chain[i].x, chain[i + 1].x}, {chain[i].y, chain[i + 1].y}, run);
          i += 2;
        }
        if (i == 0) continue;
        while (i < chain.size()) {
          Function* Fma = Intrinsic::getOrInsertDeclaration(B.GetInsertBlock()->getModule(), Intrinsic::fma, {run->getType()});
          run = B.CreateCall(Fma, {chain[i].x, chain[i].y, run});
          i++;
        }
        Instruction* sink = chain.back().I;
        sink->replaceAllUsesWith(run);
        for (Term& t : llvm::reverse(chain)) {
          packed.insert(t.I);
          t.I->eraseFromParent();
        }
        changed = true;
      }
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// `__shfl_xor_sync(0xffffffff, x, 16/8/4/2/1)` plus add (or max) is a 32-lane butterfly. Metal's `simd_sum` /
// `simd_max` is that tree. Opt-in: MVCC_SIMD_REDUCE=1. Must insert the call next to the uniform +16 add,
// not at a sink that may sit inside `if (lane stores)`.
struct WarpButterflyReducePass : PassInfoMixin<WarpButterflyReducePass> {
  static Value* stripCast(Value* V) {
    while (auto* BC = dyn_cast<BitCastInst>(V)) V = BC->getOperand(0);
    return V;
  }
  static bool shflXor(Value* V, Value*& src, unsigned& delta) {
    V = stripCast(V);
    auto* CI = dyn_cast<CallInst>(V);
    if (!CI || !CI->getCalledFunction()) return false;
    StringRef n = CI->getCalledFunction()->getName();
    if (!n.starts_with("llvm.nvvm.shfl.sync.bfly")) return false;
    if (CI->arg_size() < 4) return false;
    auto* D = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    auto* C = dyn_cast<ConstantInt>(CI->getArgOperand(3));
    if (!D || !C || (C->getZExtValue() & 0x1f) != 0x1f) return false;
    delta = (unsigned)D->getZExtValue() & 0x1f;
    src = stripCast(CI->getArgOperand(1));
    return true;
  }
  static bool step(Instruction* I, Value*& prev, unsigned& delta, char& kind) {
    Value *a = nullptr, *b = nullptr;
    kind = 0;
    if (auto* BO = dyn_cast<BinaryOperator>(I)) {
      if (BO->getOpcode() == Instruction::FAdd || BO->getOpcode() == Instruction::Add) {
        a = BO->getOperand(0); b = BO->getOperand(1); kind = '+';
      }
    }
    if (!kind) {
      auto* CI = dyn_cast<CallInst>(I);
      if (!CI) return false;
      Intrinsic::ID id = CI->getIntrinsicID();
      StringRef n = CI->getCalledFunction() ? CI->getCalledFunction()->getName() : "";
      if (id == Intrinsic::maxnum || id == Intrinsic::maximum || n.starts_with("llvm.nvvm.fmax") || n == "__nv_fmaxf") {
        if (CI->arg_size() < 2) return false;
        a = CI->getArgOperand(0); b = CI->getArgOperand(1); kind = 'M';
      } else {
        return false;
      }
    }
    Value *src = nullptr; unsigned d = 0;
    if (shflXor(b, src, d) && stripCast(a) == src) { prev = stripCast(a); delta = d; return true; }
    if (shflXor(a, src, d) && stripCast(b) == src) { prev = stripCast(b); delta = d; return true; }
    return false;
  }
  static Function* helper(Module* M, Type* T, char kind) {
    const char* name = kind == 'M' ? "__mvcc_simd_max" : "__mvcc_simd_sum";
    FunctionCallee C = M->getOrInsertFunction(name, FunctionType::get(T, {T}, false));
    auto* Fn = cast<Function>(C.getCallee());
    Fn->setDoesNotAccessMemory();
    Fn->setDoesNotThrow();
    Fn->addFnAttr(Attribute::Convergent);
    return Fn;
  }

  PreservedAnalyses run(Function& F, FunctionAnalysisManager&) {
    return PreservedAnalyses::all();
    bool changed = false;
    SmallVector<Instruction*, 16> sinks;
    for (BasicBlock& BB : F)
      for (Instruction& I : BB) {
        Value* prev; unsigned d; char k;
        if (step(&I, prev, d, k) && d == 1) sinks.push_back(&I);
      }
    for (Instruction* sink : sinks) {
      if (!sink->getParent()) continue;
      Instruction* cur = sink;
      Instruction* uniform = nullptr;  // the +16 add: always outside a divergent use of the total
      Value* root = nullptr;
      char kind = 0;
      const unsigned expect[] = {1, 2, 4, 8, 16};
      bool ok = true;
      for (unsigned e : expect) {
        Value* prev; unsigned d; char k;
        if (!step(cur, prev, d, k) || d != e) { ok = false; break; }
        if (!kind) kind = k;
        else if (kind != k) { ok = false; break; }
        root = prev;
        if (e == 16) uniform = cur;
        if (e != 16) {
          auto* PI = dyn_cast<Instruction>(prev);
          if (!PI) { ok = false; break; }
          cur = PI;
        }
      }
      if (!ok || !root || !uniform) continue;
      // simd_sum is convergent: it must sit with the uniform butterfly, not at a sink that
      // may be inside `if (lane stores)`, where a partial simdgroup would compute a wrong sum.
      IRBuilder<> B(uniform);
      Value* r = B.CreateCall(helper(F.getParent(), sink->getType(), kind), {root});
      sink->replaceAllUsesWith(r);
      sink->eraseFromParent();
      changed = true;
    }
    if (changed) {
      auto droppable = [](Instruction& I) -> bool {
        if (isa<BitCastInst>(I) || isa<BinaryOperator>(I)) return true;
        auto* CI = dyn_cast<CallInst>(&I);
        if (!CI || !CI->getCalledFunction()) return false;
        StringRef n = CI->getCalledFunction()->getName();
        return n.starts_with("llvm.nvvm.shfl") || n.starts_with("llvm.nvvm.fmax") || n.starts_with("llvm.maxnum")
            || n.starts_with("llvm.maximum") || n == "__nv_fmaxf";
      };
      bool progress = true;
      while (progress) {
        progress = false;
        for (BasicBlock& BB : F)
          for (Instruction& I : llvm::make_early_inc_range(BB))
            if (!I.isTerminator() && I.use_empty() && droppable(I)) {
              I.eraseFromParent();
              progress = true;
            }
      }
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// MSL vectors have at most 4 lanes. The load/store vectorizer may build <8 x i8> / <8 x i16>; split such accesses
// into 4-lane pieces glued with shufflevectors (InstCombine folds the extracts back onto the pieces).
// Constant branches left behind by InstCombine (`br i1 false`) keep dead CFG edges alive; those edges break the
// dominance facts the recovery proofs rely on (a store on the live path no longer dominates the barrier). Fold them
// and drop the unreachable blocks; nothing else is restructured.
// libdevice's integer min/max/abs (`min(g1 - gs, kGps)` bounds the group loop of the decode kernels) are opaque
// calls to ScalarEvolution, known-bits and InstCombine; the LLVM intrinsics say the same thing and every analysis
// reads them (a loop bounded by smin(x, 1) has a constant maximum trip count; the staging enumeration needs one).
struct LowerLibdeviceMinMaxPass : PassInfoMixin<LowerLibdeviceMinMaxPass> {
  PreservedAnalyses run(Function& F, FunctionAnalysisManager&) {
    static const std::map<std::string, std::pair<Intrinsic::ID, unsigned>> table = {  // name -> (intrinsic, bit width)
      {"__nv_min", {Intrinsic::smin, 32}}, {"__nv_max", {Intrinsic::smax, 32}}, {"__nv_umin", {Intrinsic::umin, 32}}, {"__nv_umax", {Intrinsic::umax, 32}},
      {"__nv_llmin", {Intrinsic::smin, 64}}, {"__nv_llmax", {Intrinsic::smax, 64}}, {"__nv_ullmin", {Intrinsic::umin, 64}}, {"__nv_ullmax", {Intrinsic::umax, 64}},
      {"__nv_abs", {Intrinsic::abs, 32}}, {"__nv_llabs", {Intrinsic::abs, 64}}};
    bool changed = false;
    for (Instruction& I : llvm::make_early_inc_range(instructions(F))) {
      auto* CI = dyn_cast<CallInst>(&I);
      Function* Callee = CI ? CI->getCalledFunction() : nullptr;
      if (!Callee || !Callee->isDeclaration()) continue;
      auto it = table.find(Callee->getName().str());
      if (it == table.end()) continue;
      Type* T = CI->getType();
      const unsigned nargs = it->second.first == Intrinsic::abs ? 1 : 2;
      if (!T->isIntegerTy(it->second.second) || CI->arg_size() != nargs) continue;
      bool typed = true;
      for (unsigned i = 0; i < nargs; i++) typed &= CI->getArgOperand(i)->getType() == T;
      if (!typed) continue;
      IRBuilder<> B(CI);
      Value* v = nargs == 1 ? B.CreateBinaryIntrinsic(Intrinsic::abs, CI->getArgOperand(0), B.getFalse())
                            : B.CreateBinaryIntrinsic(it->second.first, CI->getArgOperand(0), CI->getArgOperand(1));
      CI->replaceAllUsesWith(v);
      CI->eraseFromParent();
      changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Small local tables indexed by a run-time value (`T* buf[kStages]; ... buf[t % kStages]`): SROA leaves the alloca
// because the index is not constant, so the pointers escape into local memory and every later analysis loses their
// provenance (the staging passes see "a shared-memory address escapes to memory" and keep the layout). When every
// element is stored exactly once, before every load, a load at index i is the select chain over the stored values
// and the table disappears.
struct ScalarizeLocalTablesPass : PassInfoMixin<ScalarizeLocalTablesPass> {
  PreservedAnalyses run(Function& F, FunctionAnalysisManager& FAM) {
    const DataLayout& DL = F.getParent()->getDataLayout();
    DominatorTree& DT = FAM.getResult<DominatorTreeAnalysis>(F);
    bool changed = false;
    std::vector<AllocaInst*> allocas;
    for (Instruction& I : instructions(F)) if (auto* AI = dyn_cast<AllocaInst>(&I)) allocas.push_back(AI);
    for (AllocaInst* AI : allocas) {
      auto* AT = dyn_cast<ArrayType>(AI->getAllocatedType());
      if (!AT || AI->isArrayAllocation()) continue;
      const uint64_t n = AT->getNumElements();
      Type* ET = AT->getElementType();
      if (n < 2 || n > 16 || !(ET->isPointerTy() || ET->isIntegerTy() || ET->isFloatingPointTy())) continue;
      const uint64_t esz = DL.getTypeAllocSize(ET);
      if (DL.getTypeStoreSize(ET) != esz) continue;
      // the table's pointers: the alloca and every chain of address-space casts of it (allocasToLocal casts it to the
      // local space, the frontend casts it back to generic at the uses)
      std::vector<Value*> bases = {AI};
      for (size_t i = 0; i < bases.size(); i++) for (User* U : bases[i]->users()) if (isa<AddrSpaceCastInst>(U)) bases.push_back(U);
      auto isBase = [&](Value* v) { return std::find(bases.begin(), bases.end(), v) != bases.end(); };
      // element addressed by a pointer into the table: idx + base (idx == nullptr: the constant element base)
      auto elemOf = [&](Value* p, Value*& idx, int64_t& base) -> bool {
        idx = nullptr; base = 0;
        if (isBase(p)) return true;
        auto* G = dyn_cast<GEPOperator>(p);
        if (!G || !isBase(G->getPointerOperand())) return false;
        const unsigned bw = DL.getIndexSizeInBits(G->getPointerAddressSpace());
        SmallMapVector<Value*, APInt, 4> vo; APInt co(bw, 0);
        if (!G->collectOffset(DL, bw, vo, co)) return false;
        if (co.getSExtValue() % (int64_t)esz) return false;
        base = co.getSExtValue() / (int64_t)esz;
        if (vo.size() > 1) return false;
        if (vo.size() == 1) {
          Value* v = vo.begin()->first; const APInt& sc = vo.begin()->second;
          if (sc == esz) idx = v;
          else if (sc == 1) {
            // a byte offset computed from the element index: idx << log2(esz) or idx * esz
            if (auto* BO = dyn_cast<BinaryOperator>(v)) {
              auto* K = dyn_cast<ConstantInt>(BO->getOperand(1));
              if (K && BO->getOpcode() == Instruction::Shl && isPowerOf2_64(esz) && K->getZExtValue() == Log2_64(esz)) idx = BO->getOperand(0);
              else if (K && BO->getOpcode() == Instruction::Mul && K->getZExtValue() == esz) idx = BO->getOperand(0);
            }
            if (!idx) return false;
          } else return false;
        }
        return true;
      };
      struct Ld { LoadInst* L; Value* idx; int64_t base; };
      struct St { StoreInst* S; unsigned lane; };  // element = lane of the stored value (a vectorized store of several elements)
      std::vector<Ld> loads; std::vector<St> stores(n, St{nullptr, 0}); std::vector<Instruction*> dead; bool ok = true;
      std::function<void(Value*)> visit = [&](Value* tbl) {
        for (User* U : tbl->users()) {
          if (!ok) return;
          if (auto* II = dyn_cast<IntrinsicInst>(U)) { if (II->isLifetimeStartOrEnd()) { dead.push_back(II); continue; } ok = false; return; }
          if (isa<AddrSpaceCastInst>(U)) { dead.push_back(cast<Instruction>(U)); visit(U); continue; }
          Value* p = tbl; std::vector<User*> accs;
          if (auto* G = dyn_cast<GEPOperator>(U)) { if (G->getPointerOperand() != tbl) { ok = false; return; } p = G; for (User* V : U->users()) accs.push_back(V); dead.push_back(cast<Instruction>(U)); }
          else accs.push_back(U);
          Value* idx; int64_t base;
          if (!elemOf(p, idx, base)) { ok = false; return; }
          for (User* V : accs) {
            if (auto* L = dyn_cast<LoadInst>(V)) { if (L->getType() != ET || L->isVolatile() || L->getPointerOperand() != p) { ok = false; return; } loads.push_back({L, idx, base}); continue; }
            if (auto* S = dyn_cast<StoreInst>(V)) {
              if (S->isVolatile() || S->getPointerOperand() != p || idx || base < 0) { ok = false; return; }
              Type* VT = S->getValueOperand()->getType();
              unsigned m = 1;
              if (VT != ET) {
                // the frontend's store vectorizer: <m x iN> holding m elements (pointers as integers)
                auto* FV = dyn_cast<FixedVectorType>(VT);
                if (!FV || DL.getTypeStoreSize(FV->getElementType()) != esz || !(FV->getElementType()->isIntegerTy() || FV->getElementType() == ET)) { ok = false; return; }
                m = FV->getNumElements();
              }
              for (unsigned j = 0; j < m; j++) { if ((uint64_t)base + j >= n || stores[base + j].S) { ok = false; return; } stores[base + j] = {S, j}; }
              continue;
            }
            ok = false; return;
          }
        }
      };
      visit(AI);
      if (!ok || loads.empty()) continue;
      for (auto& s : stores) if (!s.S) ok = false;
      if (!ok) continue;
      for (auto& ld : loads) {
        if (!ld.idx && (ld.base < 0 || (uint64_t)ld.base >= n)) ok = false;
        for (auto& s : stores) if (!DT.dominates(s.S, ld.L)) ok = false;
      }
      if (!ok) continue;
      auto elemValue = [&](IRBuilder<>& B, unsigned k) -> Value* {
        Value* v = stores[k].S->getValueOperand();
        if (v->getType() == ET) return v;
        Value* e = B.CreateExtractElement(v, stores[k].lane);
        if (e->getType() == ET) return e;
        if (auto* CE = dyn_cast<ConstantExpr>(e)) if (CE->getOpcode() == Instruction::PtrToInt && CE->getOperand(0)->getType() == ET) return CE->getOperand(0);
        return B.CreateIntToPtr(e, ET);
      };
      for (auto& ld : loads) {
        IRBuilder<> B(ld.L);
        Value* v;
        if (!ld.idx) v = elemValue(B, (unsigned)ld.base);
        else {
          v = PoisonValue::get(ET);
          for (int64_t k = (int64_t)n - 1; k >= 0; k--) v = B.CreateSelect(B.CreateICmpEQ(ld.idx, ConstantInt::get(ld.idx->getType(), k - ld.base)), elemValue(B, (unsigned)k), v);
        }
        ld.L->replaceAllUsesWith(v); ld.L->eraseFromParent();
      }
      std::set<StoreInst*> erased;
      for (auto& s : stores) if (erased.insert(s.S).second) s.S->eraseFromParent();
      // GEPs and casts first (they use the alloca), then the alloca
      std::set<Instruction*> gone;
      for (bool progress = true; progress;) {
        progress = false;
        for (Instruction* d : dead) if (!gone.count(d) && d->use_empty()) { gone.insert(d); d->eraseFromParent(); progress = true; }
      }
      if (AI->use_empty()) AI->eraseFromParent();
      changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

struct FoldConstantBranchesPass : PassInfoMixin<FoldConstantBranchesPass> {
  PreservedAnalyses run(Function& F, FunctionAnalysisManager&) {
    bool changed = false;
    for (BasicBlock& BB : make_early_inc_range(F)) changed |= ConstantFoldTerminator(&BB, /*DeleteDeadConditions=*/true);
    changed |= removeUnreachableBlocks(F);
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

struct LegalizeWideVectorsPass : PassInfoMixin<LegalizeWideVectorsPass> {
  static Value* loadPieces(IRBuilder<>& B, LoadInst* L, FixedVectorType* VT) {
    unsigned n = VT->getNumElements(), w = 4;
    Type* PT = FixedVectorType::get(VT->getElementType(), w);
    unsigned esz = B.GetInsertBlock()->getModule()->getDataLayout().getTypeAllocSize(VT->getElementType());
    SmallVector<Value*, 4> pieces;
    for (unsigned i = 0; i < n; i += w) {
      Value* p = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), L->getPointerOperand(), (uint64_t)i * esz);
      Align a = commonAlignment(L->getAlign(), (uint64_t)i * esz);
      LoadInst* NL = B.CreateAlignedLoad(PT, p, a, L->isVolatile());
      pieces.push_back(NL);
    }
    // concatenate pairwise
    while (pieces.size() > 1) {
      SmallVector<Value*, 4> next;
      for (size_t i = 0; i + 1 < pieces.size(); i += 2) {
        unsigned m = cast<FixedVectorType>(pieces[i]->getType())->getNumElements();
        SmallVector<int, 16> mask; for (unsigned k = 0; k < 2 * m; ++k) mask.push_back((int)k);
        next.push_back(B.CreateShuffleVector(pieces[i], pieces[i + 1], mask));
      }
      if (pieces.size() & 1) next.push_back(pieces.back());
      pieces = next;
    }
    return pieces[0];
  }
  // extractelement (bitcast <n x iW> %v to <m x iN>), idx  ->  trunc(lshr(extractelement %v, idx / r), (idx % r) * N)
  // (InstCombine's canonical form for byte extraction out of a vectorized load; little-endian lane order)
  static bool rewriteWideExtract(ExtractElementInst* EE) {
    auto* BC = dyn_cast<BitCastInst>(EE->getVectorOperand());
    if (!BC) return false;
    auto* WT = dyn_cast<FixedVectorType>(BC->getType());
    auto* ST = dyn_cast<FixedVectorType>(BC->getSrcTy());
    if (!WT || !ST || WT->getNumElements() <= 4) return false;
    if (!WT->getElementType()->isIntegerTy() || !ST->getElementType()->isIntegerTy()) return false;
    auto* CI = dyn_cast<ConstantInt>(EE->getIndexOperand());
    if (!CI) return false;
    unsigned N = WT->getElementType()->getIntegerBitWidth(), W = ST->getElementType()->getIntegerBitWidth();
    if (W % N) return false;
    unsigned r = W / N, idx = (unsigned)CI->getZExtValue();
    IRBuilder<> B(EE);
    Value* word = B.CreateExtractElement(BC->getOperand(0), B.getInt64(idx / r));
    if (idx % r) word = B.CreateLShr(word, (idx % r) * N);
    Value* v = B.CreateTrunc(word, WT->getElementType());
    EE->replaceAllUsesWith(v);
    EE->eraseFromParent();
    return true;
  }
  PreservedAnalyses run(Function& F, FunctionAnalysisManager&) {
    bool changed = false;
    SmallVector<Instruction*, 8> work;
    SmallVector<ExtractElementInst*, 16> extracts;
    for (BasicBlock& BB : F)
      for (Instruction& I : BB) {
        if (auto* EE = dyn_cast<ExtractElementInst>(&I)) { extracts.push_back(EE); continue; }
        Type* T = isa<LoadInst>(I) ? I.getType() : isa<StoreInst>(I) ? cast<StoreInst>(I).getValueOperand()->getType() : nullptr;
        auto* VT = T ? dyn_cast<FixedVectorType>(T) : nullptr;
        if (VT && VT->getNumElements() > 4 && VT->getNumElements() % 4 == 0) work.push_back(&I);
      }
    for (ExtractElementInst* EE : extracts) changed |= rewriteWideExtract(EE);
    for (Instruction* I : work) {
      IRBuilder<> B(I);
      if (auto* L = dyn_cast<LoadInst>(I)) {
        Value* v = loadPieces(B, L, cast<FixedVectorType>(L->getType()));
        L->replaceAllUsesWith(v);
        L->eraseFromParent();
      } else {
        auto* S = cast<StoreInst>(I);
        auto* VT = cast<FixedVectorType>(S->getValueOperand()->getType());
        unsigned n = VT->getNumElements(), esz = F.getParent()->getDataLayout().getTypeAllocSize(VT->getElementType());
        for (unsigned i = 0; i < n; i += 4) {
          SmallVector<int, 4> mask; for (unsigned k = 0; k < 4; ++k) mask.push_back((int)(i + k));
          Value* piece = B.CreateShuffleVector(S->getValueOperand(), PoisonValue::get(VT), mask);
          Value* p = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), S->getPointerOperand(), (uint64_t)i * esz);
          B.CreateAlignedStore(piece, p, commonAlignment(S->getAlign(), (uint64_t)i * esz), S->isVolatile());
        }
        S->eraseFromParent();
      }
      changed = true;
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Merge all `ret` blocks into one so every function has a single exit (post-dominator tree root).
struct MergeReturnPass : PassInfoMixin<MergeReturnPass> {
  PreservedAnalyses run(Function& F, FunctionAnalysisManager&) {
    SmallVector<BasicBlock*, 8> rets, unreach;
    for (BasicBlock& BB : F) {
      if (isa<ReturnInst>(BB.getTerminator())) rets.push_back(&BB);
      else if (isa<UnreachableInst>(BB.getTerminator())) unreach.push_back(&BB);
    }
    if (rets.size() + unreach.size() <= 1) return PreservedAnalyses::all();
    BasicBlock* Ret = BasicBlock::Create(F.getContext(), "mvcc.unified.ret", &F);
    PHINode* PN = nullptr;
    if (!F.getReturnType()->isVoidTy()) PN = PHINode::Create(F.getReturnType(), rets.size(), "mvcc.retval", Ret);
    ReturnInst::Create(F.getContext(), PN, Ret);
    for (BasicBlock* BB : rets) {
      auto* RI = cast<ReturnInst>(BB->getTerminator());
      if (PN) PN->addIncoming(RI->getReturnValue(), BB);
      RI->eraseFromParent();
      BranchInst::Create(Ret, BB);
    }
    // `unreachable` (after __trap) also flows to the unified exit so the CFG has one sink.
    for (BasicBlock* BB : unreach) {
      BB->getTerminator()->eraseFromParent();
      if (PN) PN->addIncoming(PoisonValue::get(F.getReturnType()), BB);
      BranchInst::Create(Ret, BB);
    }
    return PreservedAnalyses::none();
  }
};

// If-conversion of small pure regions. A single-entry / single-exit, loop-free region whose blocks hold nothing but
// speculatable arithmetic and comparisons (no memory access, no call) is collapsed into its entry: every phi becomes
// a select over the edge predicates, the branches go. The frontend leaves short-circuit boolean logic as control
// flow — `take = a < b || (a == b && ia > ib)` is a chain of compare-and-branch blocks whose exits meet in a
// four-predecessor phi, which none of SimplifyCFG's folds (two-entry phi, branch to common destination) match — and
// a warp bitonic compare-exchange is one such chain per step: 8 blocks, thousands of steps in an unrolled top-k
// scan. ptxas predicates all of it; in MSL every block is an `if` with flow predicates and phi copies, and Metal's
// compiler holds ~40 KB per MSL line — a 216k-line top-k kernel put it past its 8 GB
// and it aborted. Exact by construction: the moved instructions are speculatable (no trap, no side effect), and a
// select takes the value of the path the branches would have taken. Regions are collapsed innermost first; the
// dominator trees stay valid across one sweep because a collapsed region's blocks are only cut off, not deleted,
// until the sweep ends.
struct IfConvertPureRegionsPass : PassInfoMixin<IfConvertPureRegionsPass> {
  static constexpr unsigned kMaxBlocks = 24, kMaxInsts = 160;
  static bool pureBlock(BasicBlock* B, unsigned& insts) {
    auto* Br = dyn_cast<BranchInst>(B->getTerminator());
    if (!Br) return false;
    for (Instruction& I : *B) {
      if (isa<PHINode>(I) || I.isTerminator()) continue;
      if (auto* CB = dyn_cast<CallBase>(&I)) { if (!isa<IntrinsicInst>(CB) || CB->isConvergent()) return false; }
      if (I.mayReadOrWriteMemory() || I.mayHaveSideEffects() || !isSafeToSpeculativelyExecute(&I)) return false;
      insts++;
    }
    return true;
  }
  // one sweep: collapse every candidate region found with the trees as they are
  bool sweep(Function& F, DominatorTree& DT, PostDominatorTree& PDT, LoopInfo& LI) {
    bool changed = false;
    SmallPtrSet<BasicBlock*, 32> dead;
    SmallVector<BasicBlock*, 64> order;
    for (BasicBlock& B : F) order.push_back(&B);
    for (BasicBlock* E : order) {
      if (dead.count(E)) continue;
      auto* EBr = dyn_cast<BranchInst>(E->getTerminator());
      if (!EBr || !EBr->isConditional() || EBr->getSuccessor(0) == EBr->getSuccessor(1)) continue;
      auto* PN = PDT.getNode(E); if (!PN || !PN->getIDom()) continue;
      BasicBlock* X = PN->getIDom()->getBlock(); if (!X) continue;
      // the region: what E reaches before X
      SmallVector<BasicBlock*, 32> region; SmallPtrSet<BasicBlock*, 32> inR; SmallVector<BasicBlock*, 32> work;
      for (BasicBlock* S : successors(E)) if (S != X && inR.insert(S).second) work.push_back(S);
      bool ok = true; unsigned insts = 0;
      while (!work.empty() && ok) {
        BasicBlock* B = work.pop_back_val();
        region.push_back(B);
        if (region.size() > kMaxBlocks || dead.count(B) || B == E || !DT.dominates(E, B) || LI.isLoopHeader(B) || LI.getLoopFor(B) != LI.getLoopFor(E) || !pureBlock(B, insts)) { ok = false; break; }
        for (BasicBlock* S : successors(B)) { if (S == E) { ok = false; break; } if (S != X && inR.insert(S).second) work.push_back(S); }
      }
      if (!ok || region.empty() || insts > kMaxInsts) continue;
      // single entry: every predecessor of a region block is E or in the region
      for (BasicBlock* B : region) for (BasicBlock* P : predecessors(B)) if (P != E && !inR.count(P)) { ok = false; break; }
      if (!ok) continue;
      // topological order within the region (E first)
      SmallVector<BasicBlock*, 32> topo; SmallPtrSet<BasicBlock*, 32> done; done.insert(E);
      while (topo.size() < region.size()) {
        bool progress = false;
        for (BasicBlock* B : region) {
          if (done.count(B)) continue;
          bool ready = true; for (BasicBlock* P : predecessors(B)) if (!done.count(P)) { ready = false; break; }
          if (ready) { topo.push_back(B); done.insert(B); progress = true; }
        }
        if (!progress) { ok = false; break; }   // a cycle (should not happen: loop-free)
      }
      if (!ok) continue;
      // collapse
      IRBuilder<> IB(EBr);
      Type* i1 = Type::getInt1Ty(F.getContext());
      DenseMap<BasicBlock*, Value*> pred;  // block -> condition under which it executes
      pred[E] = ConstantInt::getTrue(i1);
      auto edgeCond = [&](BasicBlock* P, BasicBlock* S) -> Value* {
        auto* Br = cast<BranchInst>(P->getTerminator());
        if (!Br->isConditional()) return pred[P];
        Value* c = Br->getCondition();
        if (Br->getSuccessor(0) == S && Br->getSuccessor(1) == S) return pred[P];
        if (Br->getSuccessor(1) == S) c = IB.CreateNot(c);
        // select, not `and`: on a path P did not take, c may be poison (an overflowed `add nsw` feeding it), and
        // `and false, poison` is poison where `select false, poison, false` is false
        return IB.CreateSelect(pred[P], c, ConstantInt::getFalse(i1));
      };
      auto selectChain = [&](PHINode* P, ArrayRef<BasicBlock*> froms) -> Value* {
        Value* v = P->getIncomingValueForBlock(froms[0]);
        for (size_t i = 1; i < froms.size(); i++) v = IB.CreateSelect(edgeCond(froms[i], P->getParent()), P->getIncomingValueForBlock(froms[i]), v);
        return v;
      };
      for (BasicBlock* B : topo) {
        Value* p = nullptr;
        for (BasicBlock* P : predecessors(B)) { Value* c = edgeCond(P, B); p = p ? IB.CreateSelect(p, ConstantInt::getTrue(i1), c) : c; }
        pred[B] = p;
        SmallVector<PHINode*, 8> phis; for (PHINode& P : B->phis()) phis.push_back(&P);
        for (PHINode* P : phis) {
          SmallVector<BasicBlock*, 8> froms(P->blocks().begin(), P->blocks().end());
          Value* v = selectChain(P, froms);
          P->replaceAllUsesWith(v); P->eraseFromParent();
        }
        SmallVector<Instruction*, 32> body; for (Instruction& I : *B) if (!I.isTerminator()) body.push_back(&I);
        for (Instruction* I : body) I->moveBefore(EBr->getIterator());
      }
      // the exit's phis: one entry from E in place of the region's
      SmallVector<BasicBlock*, 8> xpreds;   // region blocks (and E) that reach X
      for (BasicBlock* B : topo) for (BasicBlock* S : successors(B)) if (S == X) { xpreds.push_back(B); break; }
      for (BasicBlock* S : successors(E)) if (S == X) { xpreds.push_back(E); break; }
      for (PHINode& P : X->phis()) {
        SmallVector<BasicBlock*, 8> froms; for (BasicBlock* B : xpreds) if (P.getBasicBlockIndex(B) >= 0) froms.push_back(B);
        if (froms.empty()) continue;
        Value* v = selectChain(&P, froms);
        for (BasicBlock* B : froms) while (P.getBasicBlockIndex(B) >= 0) P.removeIncomingValue(B, /*DeletePHIIfEmpty=*/false);
        P.addIncoming(v, E);
      }
      // E branches to X; the region's blocks are cut off (deleted after the sweep)
      for (BasicBlock* B : topo) { B->getTerminator()->eraseFromParent(); new UnreachableInst(F.getContext(), B); dead.insert(B); }
      EBr->eraseFromParent();
      BranchInst::Create(X, E);
      changed = true;
    }
    if (changed) for (BasicBlock* B : dead) { B->dropAllReferences(); }
    if (changed) for (BasicBlock* B : dead) B->eraseFromParent();
    return changed;
  }
  PreservedAnalyses run(Function& F, FunctionAnalysisManager& FAM) {
    bool any = false;
    for (unsigned round = 0; round < 16; round++) {
      auto& DT = FAM.getResult<DominatorTreeAnalysis>(F);
      auto& PDT = FAM.getResult<PostDominatorTreeAnalysis>(F);
      auto& LI = FAM.getResult<LoopAnalysis>(F);
      if (!sweep(F, DT, PDT, LI)) break;
      any = true;
      FAM.invalidate(F, PreservedAnalyses::none());
    }
    return any ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Launch-bounds ranges. A kernel with __launch_bounds__(N) (nvvm.maxntid) can only be launched with blocks of at most
// N threads: the runtime rejects anything larger (kp.max_threads), so tid.{x,y,z} < maxntid.{x,y,z} and
// ntid.{x,y,z} <= maxntid.{x,y,z} are facts. Clang's frontend attached only the architectural range(0, 1024) to the
// intrinsic declarations; the sharper per-kernel ranges let InstCombine/SimplifyCFG fold the unroller's remainder
// loops in per-thread `for (i = tid; i < C; i += blockDim)` loops, which otherwise leave merge PHIs the tensor
// recovery proofs cannot evaluate. Mirrors NVPTX's own NVVMIntrRange pass, applied after inlining.
struct LaunchBoundsRangePass : PassInfoMixin<LaunchBoundsRangePass> {
  static bool maxNtid(const Function& F, unsigned out[3]) {
    out[0] = out[1] = out[2] = 0;
    if (F.hasFnAttribute("nvvm.maxntid")) {
      SmallVector<StringRef, 3> parts; F.getFnAttribute("nvvm.maxntid").getValueAsString().split(parts, ',');
      for (unsigned i = 0; i < 3; i++) { unsigned v = 1; if (i < parts.size() && parts[i].getAsInteger(10, v)) return false; out[i] = i < parts.size() ? v : 1; }
      return out[0] && out[1] && out[2];
    }
    return false;
  }
  PreservedAnalyses run(Module& M, ModuleAnalysisManager&) {
    bool changed = false;
    for (Function& F : M) {
      if (F.isDeclaration() || !isKernel(F)) continue;
      unsigned mx[3]; if (!maxNtid(F, mx)) continue;
      for (Instruction& I : instructions(F)) {
        auto* CI = dyn_cast<CallInst>(&I); if (!CI || !CI->getCalledFunction() || !CI->getType()->isIntegerTy(32)) continue;
        StringRef n = CI->getCalledFunction()->getName();
        int dim = -1; bool nt = false;
        if (n.starts_with("llvm.nvvm.read.ptx.sreg.tid.")) dim = n.back() - 'x';
        else if (n.starts_with("llvm.nvvm.read.ptx.sreg.ntid.")) { dim = n.back() - 'x'; nt = true; }
        if (dim < 0 || dim > 2) continue;
        ConstantRange R = nt ? ConstantRange(APInt(32, 1), APInt(32, mx[dim] + 1)) : ConstantRange(APInt(32, 0), APInt(32, mx[dim]));
        if (!nt && mx[dim] == 1) R = ConstantRange(APInt(32, 0));
        CI->addRangeRetAttr(R); changed = true;
      }
    }
    return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
  }
};

// Drops `convergent` from inline-asm calls that are per-thread PTX ALU instructions, in kernels that contain a
// *guarded* mma.sync: one whose block is conditionally executed (it does not post-dominate its immediate dominator,
// which ends in a conditional branch). Only those kernels need the same-guard merging; on a kernel whose mmas run
// unconditionally jump threading would only re-shape unrelated control flow (the epilogue's bounds checks), which
// costs the staging layout its address enumeration. Returns true if the kernel qualifies.
static bool deconvergeScalarAsm(Function& F, bool& hasMma) {
  bool guardedMma = false; hasMma = false;
  DominatorTree DT(F); PostDominatorTree PDT(F); LoopInfo LI(DT);
  // guarded: some dominating block of the same loop (not its header or latch, whose branches are the trip test)
  // ends in a conditional branch that BB does not post-dominate
  auto guarded = [&](BasicBlock* BB) {
    const Loop* L = LI.getLoopFor(BB);
    for (auto* N = DT.getNode(BB)->getIDom(); N; N = N->getIDom()) {
      BasicBlock* D = N->getBlock();
      if (LI.getLoopFor(D) != L) break;
      if (L && (D == L->getHeader() || L->isLoopLatch(D))) continue;
      auto* Br = dyn_cast<BranchInst>(D->getTerminator());
      if (Br && Br->isConditional() && !PDT.dominates(BB, D)) return true;
    }
    return false;
  };
  std::vector<CallInst*> asmCalls;
  for (BasicBlock& BB : F)
    for (Instruction& I : BB) {
      auto* CI = dyn_cast<CallInst>(&I);
      if (!CI || !CI->isInlineAsm()) continue;
      std::vector<PtxInstr> ins; std::string err;
      if (!parsePtxAsm(std::string(cast<InlineAsm>(CI->getCalledOperand())->getAsmString()), ins, err) || ins.empty()) continue;
      bool allScalar = true;
      for (auto& in : ins) {
        if (in.mnemonic == "mma") { hasMma = true; if (!guardedMma && guarded(&BB)) guardedMma = true; }
        if (!ptxIsPerThreadAlu(in.mnemonic) || in.predicated()) allScalar = false;
      }
      if (allScalar && !cast<InlineAsm>(CI->getCalledOperand())->hasSideEffects() && CI->isConvergent()) asmCalls.push_back(CI);
    }
  if (!guardedMma) return false;
  for (CallInst* CI : asmCalls) CI->removeFnAttr(Attribute::Convergent);
  return true;
}

void normalizeModule(Module& M, const EmitOptions& opts, TensorRecoveryResult& tr) {
  // 1. Force-inline every defined non-kernel function.
  for (Function& F : M) {
    if (F.isDeclaration() || isKernel(F)) continue;
    F.removeFnAttr(Attribute::NoInline);
    F.removeFnAttr(Attribute::OptimizeNone);
    F.addFnAttr(Attribute::AlwaysInline);
    F.setLinkage(GlobalValue::InternalLinkage);
  }
  for (Function& F : M) if (isKernel(F)) kernelPointerArgsToGlobal(F);
  for (Function& F : M) if (!F.isDeclaration()) allocasToLocal(F);
  // ignore-src cp.async blocks (guarded_loads.h) become the merged zero-selecting copy before anything reads them
  for (Function& F : M) if (!F.isDeclaration()) lowerPredicatedCopies(F, opts.tensorDiag ? &tr.notes : nullptr);

  PassBuilder PB;
  LoopAnalysisManager LAM; FunctionAnalysisManager FAM; CGSCCAnalysisManager CGAM; ModuleAnalysisManager MAM;
  const DataLayout& DL = M.getDataLayout();
  FAM.registerPass([&] { return TargetIRAnalysis([&](const Function&) { return TargetTransformInfo(std::make_unique<MvccTTIImpl>(DL)); }); });
  PB.registerModuleAnalyses(MAM); PB.registerCGSCCAnalyses(CGAM); PB.registerFunctionAnalyses(FAM); PB.registerLoopAnalyses(LAM);
  PB.crossRegisterProxies(LAM, FAM, CGAM, MAM);

  ModulePassManager MPM;
  MPM.addPass(AlwaysInlinerPass());
  MPM.addPass(GlobalDCEPass());
  MPM.addPass(LaunchBoundsRangePass());
  {
    FunctionPassManager FPM;
    FPM.addPass(SROAPass(SROAOptions::ModifyCFG));  // closures of inlined lambdas (a captured shared pointer in an alloca would be an escape to the staging passes)
    FPM.addPass(ScalarizeLocalTablesPass());         // stage-pointer tables indexed by `t % kStages`: SROA cannot, a select chain can
    FPM.addPass(InferAddressSpacesPass(/*flat=*/0));
    FPM.addPass(SplitMixedPtrPhiPass());
    FPM.addPass(InstCombinePass());
    FPM.addPass(SimplifyCFGPass());
    FPM.addPass(InferAddressSpacesPass(0));
    FPM.addPass(SplitMixedPtrPhiPass());
    FPM.addPass(InferAddressSpacesPass(0));
    FPM.addPass(CudaAlignmentPass());
    FPM.addPass(LoadStoreVectorizerPass());
    FPM.addPass(InstCombinePass());
    FPM.addPass(LegalizeWideVectorsPass());  // after InstCombine: it would re-canonicalize byte extracts into wide bitcasts
    {
      ScalarizeWideVectorsPass SW;
      SW.onError = [&](Function& Fn, const std::string& e) { tr.warnings.push_back("in " + llvm::demangle(Fn.getName().str()) + ": wide vector left in place: " + e); };
      FPM.addPass(std::move(SW));
    }
    FPM.addPass(DCEPass());
    FPM.addPass(FoldConstantBranchesPass());
    MPM.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }
  MPM.run(M, MAM);
  // Guarded loads first (guarded_loads.h): the bounds-checked cp.async pairs of a staging loop become single
  // unconditional copies, which is the shape the level-4 proofs and Metal's scheduler both want.
  if (opts.guardedLoads) {
    GuardedLoadStats gs;
    for (Function& F : M) if (!F.isDeclaration() && isKernel(F)) ifConvertGuardedLoads(F, gs, opts.tensorDiag ? &tr.notes : nullptr);
    for (Function& F : M) FAM.invalidate(F, PreservedAnalyses::none());
    MAM.invalidate(M, PreservedAnalyses::none());
  }
  // Canonicalization for recovery. Every mma kernel: EarlyCSE and SimplifyCFG. The guarded-load if-conversion
  // leaves one operand select per mma (identical ones must merge so tiles share fragments) and the frontend leaves
  // the k-step's mmas in a chain of single-successor blocks (they must sit in one block to be paired).
  // Same-guard merging, only for kernels whose mmas are themselves conditional: a kernel that guards each mma of
  // a k-step separately with one warp-uniform predicate (`if (rows_valid) mma(...)` per n-subtile, dequantization
  // in between) leaves the batch spread over a chain of diamonds; recovery wants it in one block. Threading the
  // predicate through the in-between code joins the arms (LLVM jump threading, with the duplication budget raised
  // to the size of a dequantization step). Per-thread PTX ALU asm (lop3, fma.rn.bf16x2, prmt, cvt...) is first
  // stripped of the frontend's blanket `convergent`, which would otherwise forbid duplicating the block it sits in;
  // warp-level asm keeps it. Jump threading is skipped unless an mma itself is behind a uniform predicate:
  // threading a `bias != nullptr` epilogue through many small diamonds can hide C-tile stores from staging
  if (opts.tensorRecovery) {
    std::vector<Function*> canon, threaded;
    for (Function& F : M) {
      if (F.isDeclaration() || !isKernel(F)) continue;
      bool hasMma = false;
      const bool jt = deconvergeScalarAsm(F, hasMma);
      if (hasMma) canon.push_back(&F);
      if (jt) { threaded.push_back(&F); if (opts.tensorDiag) tr.notes.push_back(llvm::demangle(F.getName().str()) + ": same-guard merging (jump threading) applied"); }
    }
    if (!canon.empty()) {
      FunctionPassManager FPM;
      // InstCombine first: the if-conversion's selects come in both polarities (select (xor c, 1), a, b), which
      // EarlyCSE alone does not merge; the m=1 sub-tile fragments of a 2-sub-tile int4 kernel stayed
      // apart and its n8 tiles found no partner sharing A
      FPM.addPass(InstCombinePass());
      FPM.addPass(EarlyCSEPass());
      FPM.addPass(SimplifyCFGPass());
      for (Function* F : canon) { FPM.run(*F, FAM); FAM.invalidate(*F, PreservedAnalyses::none()); }
      FunctionPassManager FPMj;
      FPMj.addPass(JumpThreadingPass(/*BBDupThreshold=*/96));
      FPMj.addPass(SimplifyCFGPass());
      for (Function* F : threaded) { FPMj.run(*F, FAM); FAM.invalidate(*F, PreservedAnalyses::none()); }
      MAM.invalidate(M, PreservedAnalyses::none());
    }
  }
  // libdevice min/max/abs -> the LLVM intrinsics, after the InstCombine runs above and before recovery: the
  // intrinsics are what ScalarEvolution reads (a loop bounded by smin(x, 1) has a constant maximum trip count; the
  // staging enumeration needs one), but InstCombine would also distribute a comparison over them (`r < smin(32, x)`
  // becomes `r < 32 & r < x`, then folds differently for each constant r) and the decode retiler needs one shape of
  // guard across a tile's elements (int4 experts, 2 and 4 row sub-tiles).
  {
    FunctionPassManager FPM;
    FPM.addPass(LowerLibdeviceMinMaxPass());
    for (Function& F : M) if (!F.isDeclaration()) { FPM.run(F, FAM); FAM.invalidate(F, PreservedAnalyses::none()); }
    MAM.invalidate(M, PreservedAnalyses::none());
  }
  // Tensor-pipeline recovery runs here: wrappers are inlined and address arithmetic simplified, but the CFG has not
  // been structurized yet, so loop phis and the per-k-step mma batches are still in their natural shape.
  {
    TensorRecoveryOptions to; to.enabled = opts.tensorRecovery; to.verbose = opts.tensorDiag;
    to.groupM = opts.tpGroupM; to.groupN = opts.tpGroupN; to.groupK = opts.tpGroupK;
    runTensorRecovery(M, to, tr);
    // a rewrite that breaks SSA must be caught here: the passes below would repair the CFG around it into a
    // well-formed but wrong program
    {
      std::string verr; raw_string_ostream vos(verr);
      if (verifyModule(M, &vos)) {
        throw EmitError{"internal: tensor recovery produced invalid IR: " + verr};
      }
    }
    // recovery may change the CFG (exit edges split for the accumulator conversion): drop every cached analysis
    for (Function& F : M) FAM.invalidate(F, PreservedAnalyses::none());
    MAM.invalidate(M, PreservedAnalyses::none());
  }
  ModulePassManager MPM2;
  {
    FunctionPassManager FPM;
    FPM.addPass(DCEPass());
    // pure branchy regions become selects (IfConvertPureRegionsPass, above); its InstCombine runs before the wide
    // vector legalization below, which InstCombine would undo
    FPM.addPass(IfConvertPureRegionsPass());
    FPM.addPass(InstCombinePass());
    // SLP / dot-pack / simd-reduce stay off: on M5 Pro Qwen decode they regressed vs scalar fma + shfl.
    // the canonicalization InstCombine above folds half extracts of a 16-byte word back into `extractelement (bitcast
    // <4 x i32> to <8 x half>)`: legalize again before anything reads the types
    FPM.addPass(LegalizeWideVectorsPass());
    {
      ScalarizeWideVectorsPass SW;
      SW.onError = [&](Function& Fn, const std::string& e) { tr.warnings.push_back("in " + llvm::demangle(Fn.getName().str()) + ": wide vector left in place: " + e); };
      FPM.addPass(std::move(SW));
    }
    FPM.addPass(LowerSwitchPass());
    FPM.addPass(FixIrreduciblePass());
    FPM.addPass(LoopSimplifyPass());
    FPM.addPass(UnifyLoopExitsPass());
    FPM.addPass(MergeReturnPass());
    FPM.addPass(StructurizeCFGPass());
    MPM2.addPass(createModuleToFunctionPassAdaptor(std::move(FPM)));
  }
  MPM2.addPass(GlobalDCEPass());
  MPM2.run(M, MAM);
}

// ---------------------------------------------------------------------------
// Emitter
// ---------------------------------------------------------------------------

class Emitter {
public:
  Emitter(Module& M, const EmitOptions& o, EmitResult& r) : M(M), DL(M.getDataLayout()), opts(o), res(r) {}
  void run();

private:
  Module& M;
  const DataLayout& DL;
  const EmitOptions& opts;
  EmitResult& res;

  std::string typesOut, globalsOut, kernelsOut;
  std::map<StructType*, std::string> structNames;
  std::vector<StructType*> structOrder;
  std::set<std::string> usedHelpers;
  bool usesZeroPage = false;     // current kernel references __mvcc_zero (set while emitting the body)
  bool usesLocks = false;        // current kernel uses 64-bit atomics (lock table at [[buffer(4)]])
  bool usesGlobals = false;      // current kernel references the module globals buffer
  std::map<const GlobalVariable*, unsigned> globalOffsets;  // buffered globals -> byte offset in the globals buffer
  const KernelABI* curAbi = nullptr;  // parameter layout while emitting a kernel body
  std::vector<unsigned> curGraphLaunchExecOffsets;
  void layoutGlobals();
  void serializeInit(Constant* C, uint64_t off);
  const TensorRecoveryResult* recovery = nullptr;
 public:
  void setRecovery(const TensorRecoveryResult& r) { recovery = &r; }
 private:
  void emitTpOp(CallInst& CI, StringRef name);
  void emitVerifyKernels();

  // per-function state
  Function* F = nullptr;
  DenseMap<const Value*, std::string> names;
  std::map<const GlobalVariable*, std::string> globalNames;  // program-scope constant tables (not cleared per kernel)
  DenseMap<const BasicBlock*, unsigned> bbIds;
  std::map<const GlobalVariable*, unsigned> smemOffsets;
  unsigned smemStatic = 0; bool smemDynamic = false;
  unsigned tpScratchOff = 0, tpScratchPerSg = 0, tpScratchSgs = 0;  // tensor-recovery conversion scratch (per simdgroup)
  std::string body; unsigned indent = 1;
  std::set<const BasicBlock*> emitted;
  std::map<std::string, std::string> lateDecl;  // same-block temp name -> MSL type (declared at first assign)
  std::set<std::string> declared;
  DominatorTree* DT = nullptr; PostDominatorTree* PDT = nullptr; LoopInfo* LI = nullptr;
  BasicBlock* uniqueRet = nullptr;
  // When LLVM's PDT IDom is the virtual root (a volatile spin may not reach
  // `ret`), the real if-join is the nearest common descendant of the two arms
  // that this branch still dominates. Walking only those nodes keeps a later
  // `ret` (reached via a sibling early-return) from being picked as the merge.
  BasicBlock* nearestJoin(BasicBlock* BB, BasicBlock* T, BasicBlock* E) {
    if (!T || !E) return uniqueRet;
    if (T == E) return T;
    auto reach = [&](BasicBlock* S, DenseMap<BasicBlock*, unsigned>& dist) {
      SmallVector<BasicBlock*, 32> q;
      dist[S] = 0;
      q.push_back(S);
      for (size_t i = 0; i < q.size(); ++i)
        for (BasicBlock* N : successors(q[i]))
          if (N != BB && dist.insert({N, dist[q[i]] + 1}).second) q.push_back(N);
    };
    DenseMap<BasicBlock*, unsigned> dT, dE;
    reach(T, dT);
    reach(E, dE);
    BasicBlock* best = nullptr;
    unsigned bestD = ~0u;
    bool onlyRet = true;
    for (auto& kv : dT) {
      auto it = dE.find(kv.first);
      if (it == dE.end()) continue;
      if (kv.first != uniqueRet) onlyRet = false;
    }
    for (auto& kv : dT) {
      auto it = dE.find(kv.first);
      if (it == dE.end()) continue;
      if (!onlyRet && kv.first == uniqueRet) continue;
      unsigned d = kv.second + it->second;
      if (d < bestD) {
        bestD = d;
        best = kv.first;
      }
    }
    return best ? best : uniqueRet;
  }
  BasicBlock* branchMerge(BasicBlock* BB, Loop* clamp, BasicBlock* loopExit) {
    BasicBlock* Mb = nullptr;
    if (auto* N = PDT->getNode(BB)) if (auto* ID = N->getIDom()) Mb = ID->getBlock();
    if (!Mb) Mb = nearestJoin(BB, BB->getTerminator()->getSuccessor(0),
                              BB->getTerminator()->getNumSuccessors() > 1
                                  ? BB->getTerminator()->getSuccessor(1)
                                  : nullptr);
    if (clamp && Mb && !clamp->contains(Mb) && Mb != loopExit) Mb = loopExit;
    return Mb;
  }

  void line(const std::string& s) {
    std::string out = s;
    size_t b = 0;
    while (b < s.size() && s[b] == ' ') b++;
    if (s.size() > b + 2 && s[b] == 'v') {
      size_t i = b + 1;
      while (i < s.size() && s[i] >= '0' && s[i] <= '9') i++;
      if (i > b + 1) {
        std::string nm = s.substr(b, i - b);
        auto it = lateDecl.find(nm);
        if (it != lateDecl.end() && declared.insert(nm).second) {
          std::string rest = s.substr(b);
          if (rest.compare(nm.size(), 3, " = ") == 0) out = it->second + " " + rest;
          else out = it->second + " " + nm + "; " + rest;
        }
      }
    }
    body.append(indent * 2, ' ');
    body += out;
    body += '\n';
  }

  // types
  std::string ty(Type* T);
  std::string sty(Type* T);           // signed counterpart
  std::string vecOf(const std::string& base, unsigned n);
  std::string asSpace(unsigned as);
  static constexpr unsigned AS_MIXED = 6;  // a pointer that is provably thread memory on some paths and device/threadgroup/constant on others
  std::string ptrTy(unsigned as) { return as == AS_MIXED ? "ulong" : asSpace(as) + " uchar*"; }
  // V's pointer spelled in the representation of address space `as` (tagged ulong for AS_MIXED, see prelude "tagged pointers")
  std::string ptrIn(Value* V, unsigned as);
  static unsigned tagOf(unsigned as) { return as == 5 ? 1 : as == 3 ? 2 : as == 4 ? 3 : 0; }
  std::string declStruct(StructType* ST);
  unsigned intBits(Type* T) { return cast<IntegerType>(T->getScalarType())->getBitWidth(); }

  // values
  std::string val(Value* V);
  std::string constant(Constant* C);
  std::string constExpr(ConstantExpr* CE);
  std::string fpLiteral(const APFloat& f, Type* T);
  std::string castToSigned(Value* V) { return "(" + sty(V->getType()) + ")(" + val(V) + ")"; }
  std::string castTo(const std::string& t, const std::string& e) { return "(" + t + ")(" + e + ")"; }

  // kernels
  void emitKernel(Function& F);
  void layoutSharedMemory(Function& F);
  void emitInst(Instruction& I);
  std::string gepExpr(GEPOperator* G, std::function<std::string(Value*)> v);
  void emitCall(CallInst& CI);
  std::optional<unsigned> execParamOffset(Value* execVal);
  void recordDeviceGraphLaunch(CallInst& CI);
  void emitIntrinsic(CallInst& CI, StringRef name);
  void emitLibdevice(CallInst& CI, StringRef name);
  void emitTileOp(CallInst& CI, StringRef name);
  void emitInlineAsm(CallInst& CI);
  std::string asmOperandExpr(CallInst& CI, const std::vector<int>& inputIdx, int reg, std::vector<int>& outIdx);
  std::string atomicPtr(Value* P, Type* T, bool sign, bool fp);
  std::string atomic64(Value* P, const char* op, const std::string& v, const std::string& cmp, const std::string& ok);
  std::string mathFn(const std::string& name);

  // control flow
  void emitBodyStructured();
  void emitBodyStateMachine();
  void emitBlockInstrs(BasicBlock* BB);
  void emitPhiCopies(BasicBlock* From, BasicBlock* To);
  void emitDirectEdgeElse(BasicBlock* From, BasicBlock* Merge);
  struct LoopCtx { Loop* L; BasicBlock* exit; };
  void emitChain(BasicBlock* BB, BasicBlock* stopAt, LoopCtx* ctx);
  void gotoTarget(BasicBlock* From, BasicBlock* S, BasicBlock* stopAt, LoopCtx* ctx);
  bool forwardReachable(BasicBlock* From, BasicBlock* To);
  std::string setPred(Value* c) { return val(c); }
};

// Address space of a pointer value, looking through casts to the underlying object when the type says
// "generic" (allocas and tagged objects keep their real space even if a cast to addrspace(0) survived).
static unsigned ptrASImpl(Value* P, SmallPtrSetImpl<Value*>& visited);
static unsigned storedPtrAS(Value* V, SmallPtrSetImpl<Value*>& visited);
static unsigned slotStoresAS(AllocaInst* AI, Function* F, bool constSlot, const APInt& off, uint64_t size, SmallPtrSetImpl<Value*>& visited);
static unsigned storedElemAS(Value* V, unsigned idx, SmallPtrSetImpl<Value*>& visited);
static unsigned loadedPtrAS(LoadInst* LD, SmallPtrSetImpl<Value*>& visited);
// Join of two address-space facts: ~0u = no opinion; 0/1 both mean device; disagreement = 6 (mixed: tagged pointer).
static unsigned mergeAS(unsigned a, unsigned b) {
  if (a == ~0u) return b; if (b == ~0u) return a;
  if (a == 1) a = 0; if (b == 1) b = 0;
  return a == b ? a : 6;
}
static unsigned ptrAS(Value* P) { SmallPtrSet<Value*, 16> visited; unsigned r = ptrASImpl(P, visited); return r == ~0u ? 0 : r; }

// A pointer loaded back from private memory (a struct member SROA could not scalarize) has the space of the
// pointers stored into that slot, when they all agree.
// Pointer space carried by element `idx` of a vector value assembled with insertelement or a constant vector;
// falls back to the join over the whole value when the element cannot be singled out.
static unsigned storedElemAS(Value* V, unsigned idx, SmallPtrSetImpl<Value*>& visited) {
  Value* cur = V;
  for (int hops = 0; hops < 16; hops++) {
    if (auto* IE = dyn_cast<InsertElementInst>(cur)) {
      auto* CI = dyn_cast<ConstantInt>(IE->getOperand(2));
      if (!CI) break;
      if (CI->getZExtValue() == idx) return storedPtrAS(IE->getOperand(1), visited);
      cur = IE->getOperand(0); continue;
    }
    if (auto* C = dyn_cast<Constant>(cur)) { if (Constant* E = C->getAggregateElement(idx)) return storedPtrAS(E, visited); break; }
    break;
  }
  return storedPtrAS(V, visited);
}

// Join of the spaces of every store into the private object `AI` that can reach the `size` bytes at constant
// offset `off` (or any store when the slot is not constant). A wider (vectorized) store covering the slot
// contributes just the element at the slot.
static unsigned slotStoresAS(AllocaInst* AI, Function* F, bool constSlot, const APInt& off, uint64_t size, SmallPtrSetImpl<Value*>& visited) {
  const DataLayout& DL = F->getParent()->getDataLayout();
  unsigned a = ~0u;
  for (BasicBlock& BB : *F)
    for (Instruction& I : BB) {
      auto* ST = dyn_cast<StoreInst>(&I);
      if (!ST || getUnderlyingObject(ST->getPointerOperand()) != AI) continue;
      Value* SV = ST->getValueOperand();
      APInt soff(64, 0);
      if (constSlot && ST->getPointerOperand()->stripAndAccumulateConstantOffsets(DL, soff, true) == AI) {
        uint64_t ssize = DL.getTypeStoreSize(SV->getType());
        // a store to a different constant slot cannot reach a constant-slot load; variable-index stores might
        if (soff.getSExtValue() + (int64_t)ssize <= off.getSExtValue() || off.getSExtValue() + (int64_t)size <= soff.getSExtValue()) continue;
        if (auto* VT = dyn_cast<FixedVectorType>(SV->getType())) {
          uint64_t esz = DL.getTypeStoreSize(VT->getElementType());
          if (esz && soff.getSExtValue() <= off.getSExtValue() && ((uint64_t)(off.getSExtValue() - soff.getSExtValue())) % esz == 0) {
            a = mergeAS(a, storedElemAS(SV, (unsigned)((off.getSExtValue() - soff.getSExtValue()) / esz), visited));
            continue;
          }
        }
      }
      a = mergeAS(a, storedPtrAS(SV, visited));
    }
  return a;
}

static unsigned loadedPtrAS(LoadInst* LD, SmallPtrSetImpl<Value*>& visited) {
  const DataLayout& DL = LD->getModule()->getDataLayout();
  auto* AI = dyn_cast<AllocaInst>(getUnderlyingObject(LD->getPointerOperand()));
  if (!AI) return 0;
  APInt off(64, 0);
  bool constSlot = LD->getPointerOperand()->stripAndAccumulateConstantOffsets(DL, off, /*AllowNonInbounds=*/true) == AI;
  return slotStoresAS(AI, LD->getFunction(), constSlot, off, DL.getTypeStoreSize(LD->getType()), visited);
}

// Address space of the pointer(s) a stored value carries: a pointer, ptrtoint of one, or a constant vector/aggregate
// of those (the vectorizer merges adjacent pointer stores into `store <2 x i64> <ptrtoint ..., ptrtoint ...>`).
// ~0u when the value carries no pointer; 0 when it carries pointers of unknown or mixed spaces.
static unsigned storedPtrAS(Value* V, SmallPtrSetImpl<Value*>& visited) {
  if (V->getType()->isPointerTy()) return ptrASImpl(V, visited);
  if (auto* P = dyn_cast<PtrToIntInst>(V)) return ptrASImpl(P->getOperand(0), visited);
  if (auto* CE = dyn_cast<ConstantExpr>(V)) {
    if (CE->getOpcode() == Instruction::PtrToInt) return ptrASImpl(CE->getOperand(0), visited);
    return 0;
  }
  if (auto* C = dyn_cast<Constant>(V)) {
    if (isa<ConstantInt>(C) || isa<ConstantFP>(C) || isa<UndefValue>(C) || C->isNullValue()) return ~0u;
    if (isa<ConstantDataSequential>(C)) return ~0u;  // plain data
    unsigned a = ~0u;
    for (unsigned i = 0, n = C->getNumOperands(); i < n; i++) a = mergeAS(a, storedPtrAS(C->getOperand(i), visited));
    return a;
  }
  // vectors/aggregates assembled in place (lambda captures: `insertelement <2 x i64> poison, ptrtoint %x, 0`)
  if (isa<InsertElementInst>(V) || isa<InsertValueInst>(V) || isa<ShuffleVectorInst>(V)) {
    unsigned a = ~0u;
    auto* I = cast<Instruction>(V);
    for (unsigned i = 0; i < 2; i++) a = mergeAS(a, storedPtrAS(I->getOperand(i), visited));
    return a;
  }
  if (V->getType()->isIntegerTy(64) || V->getType()->isVectorTy() || V->getType()->isAggregateType()) {
    // opaque integer / assembled vector: might hold a pointer we cannot see
    return isa<Instruction>(V) && (isa<InsertElementInst>(V) || isa<InsertValueInst>(V) || isa<PHINode>(V) || isa<SelectInst>(V) || isa<LoadInst>(V) || isa<BitCastInst>(V)) ? 0 : ~0u;
  }
  return ~0u;
}

// Address space of the pointer an integer value carries (inttoptr operands): ptrtoint of a pointer, an element of a
// vectorized pointer load (`extractelement (load <2 x i64> closure)`, the vectorizer's shape for lambda captures
// read back together), or a plain 64-bit reload of a pointer slot.
static unsigned intPtrAS(Value* V, SmallPtrSetImpl<Value*>& visited) {
  if (auto* PI = dyn_cast<PtrToIntInst>(V)) return ptrASImpl(PI->getOperand(0), visited);
  if (auto* CE = dyn_cast<ConstantExpr>(V)) { if (CE->getOpcode() == Instruction::PtrToInt) return ptrASImpl(CE->getOperand(0), visited); return ~0u; }
  if (auto* LD = dyn_cast<LoadInst>(V)) { if (!visited.insert(LD).second) return ~0u; return loadedPtrAS(LD, visited); }
  if (auto* EE = dyn_cast<ExtractElementInst>(V)) {
    auto* CI = dyn_cast<ConstantInt>(EE->getIndexOperand());
    Value* vec = EE->getVectorOperand();
    if (auto* LD = dyn_cast<LoadInst>(vec)) {
      if (!CI || !visited.insert(LD).second) return ~0u;
      const DataLayout& DL = LD->getModule()->getDataLayout();
      auto* AI = dyn_cast<AllocaInst>(getUnderlyingObject(LD->getPointerOperand()));
      if (!AI) return ~0u;
      APInt off(64, 0);
      bool constSlot = LD->getPointerOperand()->stripAndAccumulateConstantOffsets(DL, off, true) == AI;
      uint64_t esz = DL.getTypeStoreSize(cast<VectorType>(LD->getType())->getElementType());
      APInt eoff = off + APInt(64, CI->getZExtValue() * esz);
      return slotStoresAS(AI, LD->getFunction(), constSlot, eoff, esz, visited);
    }
    if (CI) return storedElemAS(vec, (unsigned)CI->getZExtValue(), visited);
    return storedPtrAS(vec, visited);
  }
  if (auto* PN = dyn_cast<PHINode>(V)) {
    if (!visited.insert(PN).second) return ~0u;
    unsigned a = ~0u;
    for (Value* I : PN->incoming_values()) a = mergeAS(a, intPtrAS(I, visited));
    return a;
  }
  if (auto* SEL = dyn_cast<SelectInst>(V)) {
    if (!visited.insert(SEL).second) return ~0u;
    return mergeAS(intPtrAS(SEL->getTrueValue(), visited), intPtrAS(SEL->getFalseValue(), visited));
  }
  return ~0u;
}

static unsigned ptrASImpl(Value* P, SmallPtrSetImpl<Value*>& visited) {
  unsigned as = P->getType()->getPointerAddressSpace();
  if (as != 0) return as;
  Value* cur = P;
  while (true) {
    if (auto* IP = dyn_cast<IntToPtrInst>(cur)) { unsigned r = intPtrAS(IP->getOperand(0), visited); return r == ~0u ? 0 : r; }
    if (auto* GEP = dyn_cast<GEPOperator>(cur)) { cur = GEP->getPointerOperand(); continue; }
    if (auto* BC = dyn_cast<BitCastOperator>(cur)) { cur = BC->getOperand(0); continue; }
    if (auto* AC = dyn_cast<AddrSpaceCastInst>(cur)) { cur = AC->getOperand(0); continue; }
    if (auto* CE = dyn_cast<ConstantExpr>(cur)) if (CE->getOpcode() == Instruction::AddrSpaceCast) { cur = CE->getOperand(0); continue; }
    if (auto* SEL = dyn_cast<SelectInst>(cur)) {
      if (!visited.insert(SEL).second) return ~0u;
      unsigned a = ptrASImpl(SEL->getTrueValue(), visited), b = ptrASImpl(SEL->getFalseValue(), visited);
      return mergeAS(a, b);
    }
    if (isa<ConstantPointerNull>(cur) || isa<UndefValue>(cur)) return ~0u;  // says nothing about the space
    if (auto* PN = dyn_cast<PHINode>(cur)) {
      if (!visited.insert(PN).second) return ~0u;  // cycle: no opinion
      unsigned a = ~0u;
      for (Value* V : PN->incoming_values()) a = mergeAS(a, ptrASImpl(V, visited));
      return a;
    }
    if (auto* LD = dyn_cast<LoadInst>(cur)) {
      if (!visited.insert(LD).second) return ~0u;
      unsigned r = loadedPtrAS(LD, visited);
      return r;
    }
    break;
  }
  if (isa<AllocaInst>(cur)) return 5;
  if (auto* A = dyn_cast<Argument>(cur)) if (A->hasByValAttr()) return 5;
  if (auto* GV = dyn_cast<GlobalVariable>(cur)) if (GV->getAddressSpace() == 0 && GV->isConstant()) return 4;  // optimizer-made constant table  // byval structs are copied to thread memory in the prologue
  return cur->getType()->getPointerAddressSpace();
}

std::string Emitter::asSpace(unsigned as) {
  switch (as) {
    case 0: return "device";   // residual generic: treated as device
    case 1: return "device";
    case 3: return "__MVCC_SMEM";   // threadgroup memory, or a device-memory slot in the spill build (prelude)
    case 4: return "constant";
    case 5: return "thread";
    case 6: fail("a pointer that is thread memory on some paths and device/threadgroup memory on others reaches an operation that needs one address space (README.md, \"mixed pointers\")");
    default: fail("unsupported address space " + std::to_string(as));
  }
}

std::string Emitter::vecOf(const std::string& base, unsigned n) {
  if (n == 1) return base;
  if (n < 2 || n > 4) fail("vector width " + std::to_string(n) + " of " + base + " not representable in MSL");
  return base + std::to_string(n);
}

std::string Emitter::ty(Type* T) {
  if (auto* VT = dyn_cast<FixedVectorType>(T)) return vecOf(ty(VT->getElementType()), VT->getNumElements());
  if (T->isVoidTy()) return "void";
  if (T->isIntegerTy()) {
    switch (cast<IntegerType>(T)->getBitWidth()) {
      case 1: return "bool"; case 8: return "uchar"; case 16: return "ushort"; case 32: return "uint"; case 64: return "ulong";
      default: fail("unsupported integer width i" + std::to_string(cast<IntegerType>(T)->getBitWidth()));
    }
  }
  if (T->isHalfTy()) return "half";
  if (T->isBFloatTy()) return "bfloat";
  if (T->isFloatTy()) return "float";
  if (T->isDoubleTy()) { usedHelpers.insert("f64"); return "mvcc_f64"; }  // software binary64 (prelude)
  if (auto* PT = dyn_cast<PointerType>(T)) return ptrTy(PT->getAddressSpace());
  if (auto* AT = dyn_cast<ArrayType>(T)) return "array<" + ty(AT->getElementType()) + ", " + std::to_string(AT->getNumElements()) + ">";
  if (auto* ST = dyn_cast<StructType>(T)) return declStruct(ST);
  std::string s; raw_string_ostream os(s); T->print(os);
  fail("unsupported type: " + s);
}

std::string Emitter::sty(Type* T) {
  if (auto* VT = dyn_cast<FixedVectorType>(T)) return vecOf(sty(VT->getElementType()), VT->getNumElements());
  if (T->isIntegerTy()) {
    switch (cast<IntegerType>(T)->getBitWidth()) {
      case 1: return "bool"; case 8: return "char"; case 16: return "short"; case 32: return "int"; case 64: return "long";
    }
  }
  return ty(T);
}

std::string Emitter::declStruct(StructType* ST) {
  auto it = structNames.find(ST);
  if (it != structNames.end()) return it->second;
  std::string name = "S" + std::to_string(structNames.size());
  if (ST->hasName()) {
    std::string n = ST->getName().str();
    for (char& c : n) if (!isalnum((unsigned char)c)) c = '_';
    name += "_" + n;
  }
  structNames[ST] = name;
  std::string def = "struct " + name + " {\n";
  if (ST->isPacked()) def = "struct __attribute__((packed)) " + name + " {\n";
  const StructLayout* SL = DL.getStructLayout(ST);
  for (unsigned i = 0; i < ST->getNumElements(); i++) {
    Type* ET = ST->getElementType(i);
    std::string fty = ty(ET);
    // MSL vector alignment matches LLVM's for 2/4-wide; 3-wide is 16-aligned in both. Structs and
    // arrays follow the same natural rules, so element offsets line up. Assert it.
    def += "  " + fty + " f" + std::to_string(i) + ";\n";
    (void)SL;
  }
  def += "};\n";
  typesOut += def;
  structOrder.push_back(ST);
  return name;
}

std::string Emitter::fpLiteral(const APFloat& f, Type* T) {
  APInt bits = f.bitcastToAPInt();
  if (T->isFloatTy()) {
    if (f.isZero() && !f.isNegative()) return "0.0f";
    char buf[32]; snprintf(buf, sizeof buf, "0x%08xu", (unsigned)bits.getZExtValue());
    return std::string("as_type<float>(") + buf + ")";
  }
  if (T->isHalfTy()) {
    char buf[32]; snprintf(buf, sizeof buf, "0x%04x", (unsigned)bits.getZExtValue());
    return std::string("as_type<half>((ushort)") + buf + ")";
  }
  if (T->isBFloatTy()) {
    char buf[32]; snprintf(buf, sizeof buf, "0x%04x", (unsigned)bits.getZExtValue());
    return std::string("as_type<bfloat>((ushort)") + buf + ")";
  }
  if (T->isDoubleTy()) {
    usedHelpers.insert("f64");
    char buf[40]; snprintf(buf, sizeof buf, "mvcc_f64{0x%016llxul}", (unsigned long long)bits.getZExtValue());
    return buf;
  }
  fail("fp literal of unsupported type");
}

std::string Emitter::constant(Constant* C) {
  Type* T = C->getType();
  // LLVM >= 20 represents fixed-length splats as vector-typed ConstantInt / ConstantFP ("splat (half 1.0)").
  if (isa<FixedVectorType>(T) && (isa<ConstantInt>(C) || isa<ConstantFP>(C))) return ty(T) + "(" + constant(C->getSplatValue()) + ")";
  if (auto* CI = dyn_cast<ConstantInt>(C)) {
    unsigned bw = CI->getBitWidth();
    if (bw == 1) return CI->isOne() ? "true" : "false";
    uint64_t v = CI->getZExtValue();
    char buf[48];
    if (bw == 64) snprintf(buf, sizeof buf, "0x%llxul", (unsigned long long)v);
    else if (bw == 32) snprintf(buf, sizeof buf, "%uu", (unsigned)v);
    else snprintf(buf, sizeof buf, "(%s)%uu", ty(T).c_str(), (unsigned)v);
    return buf;
  }
  if (auto* CF = dyn_cast<ConstantFP>(C)) return fpLiteral(CF->getValueAPF(), T);
  if (isa<ConstantPointerNull>(C)) return "(" + ty(T) + ")0";
  if (isa<UndefValue>(C) || isa<PoisonValue>(C)) {
    if (T->isPointerTy()) return "(" + ty(T) + ")0";
    if (T->isAggregateType()) return ty(T) + "{}";
    return ty(T) + "()";
  }
  if (isa<ConstantAggregateZero>(C)) {
    if (T->isAggregateType()) return ty(T) + "{}";
    return ty(T) + "(0)";
  }
  if (auto* CDV = dyn_cast<ConstantDataVector>(C)) {
    std::string s = ty(T) + "(";
    for (unsigned i = 0; i < CDV->getNumElements(); i++) { if (i) s += ", "; s += constant(CDV->getElementAsConstant(i)); }
    return s + ")";
  }
  if (auto* CV = dyn_cast<ConstantVector>(C)) {
    std::string s = ty(T) + "(";
    for (unsigned i = 0; i < CV->getNumOperands(); i++) { if (i) s += ", "; s += constant(CV->getOperand(i)); }
    return s + ")";
  }
  if (auto* CDA = dyn_cast<ConstantDataArray>(C)) {
    std::string s = ty(T) + "{{";
    for (unsigned i = 0; i < CDA->getNumElements(); i++) { if (i) s += ", "; s += constant(CDA->getElementAsConstant(i)); }
    return s + "}}";
  }
  if (auto* CA = dyn_cast<ConstantArray>(C)) {
    std::string s = ty(T) + "{{";
    for (unsigned i = 0; i < CA->getNumOperands(); i++) { if (i) s += ", "; s += constant(CA->getOperand(i)); }
    return s + "}}";
  }
  if (auto* CS = dyn_cast<ConstantStruct>(C)) {
    std::string s = ty(T) + "{";
    for (unsigned i = 0; i < CS->getNumOperands(); i++) { if (i) s += ", "; s += constant(CS->getOperand(i)); }
    return s + "}";
  }
  if (auto* GV = dyn_cast<GlobalVariable>(C)) {
    auto it = smemOffsets.find(GV);
    if (it != smemOffsets.end()) return "(__smem + " + std::to_string(it->second) + "u)";
    auto gi = globalOffsets.find(GV);
    if (gi != globalOffsets.end()) {
      usesGlobals = true;
      return GV->getAddressSpace() == 4 ? "((constant uchar*)(__mvcc_cglobals + " + std::to_string(gi->second) + "u))"
                                        : "((device uchar*)(__mvcc_dglobals + " + std::to_string(gi->second) + "u))";
    }
    if (GV->getAddressSpace() == 4 || (GV->getAddressSpace() == 0 && GV->isConstant())) {
      auto gn = globalNames.find(GV);
      if (gn == globalNames.end()) fail("constant global '" + GV->getName().str() + "' has no definition (no initializer)");
      return "((constant uchar*)&" + gn->second + ")";
    }
    fail("unsupported global variable '" + GV->getName().str() + "' in address space " + std::to_string(GV->getAddressSpace()) +
         " (__device__ globals are not supported yet)");
  }
  if (auto* CE = dyn_cast<ConstantExpr>(C)) return constExpr(CE);
  if (isa<Function>(C)) fail("function pointers are not supported on the device");
  std::string s; raw_string_ostream os(s); C->print(os);
  fail("unsupported constant: " + s);
}

std::string Emitter::constExpr(ConstantExpr* CE) {
  switch (CE->getOpcode()) {
    case Instruction::GetElementPtr:
      return gepExpr(cast<GEPOperator>(CE), [&](Value* v) { return constant(cast<Constant>(v)); });
    case Instruction::AddrSpaceCast: {
      auto* SrcPT = cast<PointerType>(CE->getOperand(0)->getType());
      auto* DstPT = cast<PointerType>(CE->getType());
      unsigned s = SrcPT->getAddressSpace(), d = DstPT->getAddressSpace();
      if (asSpace(s) == asSpace(d)) return constant(cast<Constant>(CE->getOperand(0)));
      // casts through generic keep the MSL pointer in its real space; consumers recover it via ptrAS()
      if (s == 0 || d == 0) return constant(cast<Constant>(CE->getOperand(0)));
      fail("address space cast " + std::to_string(s) + "->" + std::to_string(d) + " survived normalization (generic pointer to " + asSpace(s) + " memory)");
    }
    case Instruction::BitCast:
      return "as_type<" + ty(CE->getType()) + ">(" + constant(cast<Constant>(CE->getOperand(0))) + ")";
    case Instruction::PtrToInt:
      return "(" + ty(CE->getType()) + ")((ulong)(" + constant(cast<Constant>(CE->getOperand(0))) + "))";
    case Instruction::IntToPtr:
      if (CE->getType()->getPointerAddressSpace() == 3) return "(" + ty(CE->getType()) + ")(__MVCC_SMEM_ADDR(" + constant(cast<Constant>(CE->getOperand(0))) + "))";
      return "(" + ty(CE->getType()) + ")(" + constant(cast<Constant>(CE->getOperand(0))) + ")";
    // Integer arithmetic on folded addresses (e.g. `ptrtoint(@smem) + 8192` for a shared-memory sub-buffer).
    case Instruction::Add: case Instruction::Sub: case Instruction::Mul: case Instruction::Shl:
    case Instruction::Xor: case Instruction::And: case Instruction::Or: {
      const char* op = CE->getOpcode() == Instruction::Add ? "+" : CE->getOpcode() == Instruction::Sub ? "-" :
                       CE->getOpcode() == Instruction::Mul ? "*" : CE->getOpcode() == Instruction::Shl ? "<<" :
                       CE->getOpcode() == Instruction::Xor ? "^" : CE->getOpcode() == Instruction::And ? "&" : "|";
      return "(" + ty(CE->getType()) + ")((" + constant(cast<Constant>(CE->getOperand(0))) + ") " + op + " (" +
             constant(cast<Constant>(CE->getOperand(1))) + "))";
    }
    default: {
      std::string s; raw_string_ostream os(s); CE->print(os);
      fail("unsupported constant expression: " + s);
    }
  }
}

std::string Emitter::val(Value* V) {
  if (auto* C = dyn_cast<Constant>(V)) return constant(C);
  auto it = names.find(V);
  if (it == names.end()) { std::string s; raw_string_ostream os(s); V->print(os); fail("unnamed value: " + s); }
  return it->second;
}

std::string Emitter::gepExpr(GEPOperator* G, std::function<std::string(Value*)> v) {
  std::string base = v(G->getPointerOperand());
  std::string off;
  int64_t constOff = 0;
  Type* cur = G->getSourceElementType();
  auto GTI = gep_type_begin(G), GTE = gep_type_end(G);
  bool first = true;
  for (auto it = G->idx_begin(); it != G->idx_end(); ++it, ++GTI) {
    Value* Idx = *it;
    if (StructType* ST = GTI.getStructTypeOrNull()) {
      unsigned fi = (unsigned)cast<ConstantInt>(Idx)->getZExtValue();
      constOff += DL.getStructLayout(ST)->getElementOffset(fi);
      cur = ST->getElementType(fi);
      first = false;
      continue;
    }
    Type* ET = first ? cur : GTI.getIndexedType();
    (void)ET;
    uint64_t stride = DL.getTypeAllocSize(GTI.getIndexedType());
    first = false;
    if (auto* CI = dyn_cast<ConstantInt>(Idx)) { constOff += CI->getSExtValue() * (int64_t)stride; continue; }
    std::string idx = "(long)(" + v(Idx) + ")";
    // indices are signed in LLVM; our ints are unsigned, so sign-extend properly
    if (Idx->getType()->isIntegerTy(32)) idx = "(long)(int)(" + v(Idx) + ")";
    else if (Idx->getType()->isIntegerTy(64)) idx = "(long)(" + v(Idx) + ")";
    else idx = "(long)(" + sty(Idx->getType()) + ")(" + v(Idx) + ")";
    if (!off.empty()) off += " + ";
    off += idx + " * " + std::to_string(stride) + "l";
  }
  (void)GTE;
  std::string r = "(" + base;
  if (!off.empty()) r += " + (" + off + ")";
  if (constOff) r += " + (" + std::to_string(constOff) + "l)";
  return r + ")";
}

// ---------------------------------------------------------------------------

static constexpr unsigned kTpDefaultMaxThreads = 256;

void Emitter::layoutSharedMemory(Function& Fn) {
  smemOffsets.clear(); smemStatic = 0; smemDynamic = false;
  // Collect addrspace(3) globals referenced by this kernel (after inlining every function is a kernel).
  std::set<const GlobalVariable*> used;
  for (auto& I : instructions(Fn))
    for (auto& Op : I.operands()) {
      SmallVector<Constant*, 4> work;
      if (auto* C = dyn_cast<Constant>(Op)) work.push_back(C);
      while (!work.empty()) {
        Constant* C = work.pop_back_val();
        if (auto* GV = dyn_cast<GlobalVariable>(C)) { if (GV->getAddressSpace() == 3) used.insert(GV); continue; }
        for (auto& U : C->operands()) if (auto* CC = dyn_cast<Constant>(U)) work.push_back(CC);
      }
    }
  // deterministic order: by name
  std::vector<const GlobalVariable*> vec(used.begin(), used.end());
  std::sort(vec.begin(), vec.end(), [](auto* a, auto* b) { return a->getName() < b->getName(); });
  const GlobalVariable* dyn = nullptr;
  for (auto* GV : vec) {
    Type* VT = GV->getValueType();
    bool isDyn = GV->isDeclaration() || (isa<ArrayType>(VT) && cast<ArrayType>(VT)->getNumElements() == 0);
    if (isDyn) { if (dyn) fail("more than one extern __shared__ array"); dyn = GV; continue; }
    unsigned align = std::max<unsigned>(16, GV->getAlign() ? (unsigned)GV->getAlign()->value() : 16);
    smemStatic = (smemStatic + align - 1) / align * align;
    smemOffsets[GV] = smemStatic;
    smemStatic += (unsigned)DL.getTypeAllocSize(VT);
  }
  smemStatic = (smemStatic + 15) / 16 * 16;
  // Tensor recovery: accumulator layout conversions need M*N floats of scratch per simdgroup (see the prelude).
  // Sized for the launch bound when there is one, else for kTpDefaultMaxThreads; the runtime launches the exact twin
  // for larger blocks (KernelABI::recoveredMaxThreads).
  tpScratchOff = tpScratchPerSg = tpScratchSgs = 0;
  for (auto& I : instructions(Fn)) {
    auto* CI = dyn_cast<CallInst>(&I);
    Function* Callee = CI ? CI->getCalledFunction() : nullptr;
    if (!Callee || !(Callee->getName().starts_with("__mvcc_tp_acc2ptx_") || Callee->getName().starts_with("__mvcc_tp_ptx2acc_"))) continue;
    auto* M_ = dyn_cast<ConstantInt>(CI->getArgOperand(0)); auto* N_ = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    if (M_ && N_) tpScratchPerSg = 8 * 16 * 4;  // conversions run one 8x16 block per round (see the prelude)
  }
  if (tpScratchPerSg) {
    unsigned mt = kernelMaxThreads(Fn);
    tpScratchSgs = (mt ? mt : kTpDefaultMaxThreads) / 32;
    tpScratchOff = smemStatic;
    // Conversions sit where the dynamic object's ring is dead: scratch aliases its first bytes instead of
    // adding to the block's static shared memory.
    if (!(dyn && Fn.getFnAttribute("mvcc-tp-scratch").getValueAsString() == "dynamic")) smemStatic += tpScratchPerSg * tpScratchSgs;
  }
  if (dyn) { smemOffsets[dyn] = smemStatic; smemDynamic = true; }
}

void Emitter::emitKernel(Function& Fn) {
  F = &Fn; names.clear(); bbIds.clear(); body.clear(); emitted.clear(); indent = 1; usesZeroPage = false; usesGlobals = false; usesLocks = false;
  uniqueRet = nullptr;
  for (BasicBlock& BB : Fn) if (isa<ReturnInst>(BB.getTerminator())) { uniqueRet = &BB; break; }
  curGraphLaunchExecOffsets.clear();
  layoutSharedMemory(Fn);

  KernelABI abi; abi.name = Fn.getName().str(); abi.staticSmem = smemStatic; abi.dynamicSmem = smemDynamic;
  abi.maxThreads = kernelMaxThreads(Fn);
  if (recovery) for (auto& ev : recovery->exactVariants) { if (ev.first == abi.name) abi.exactVariant = ev.second; if (ev.second == abi.name) abi.isVariant = true; }
  if (recovery) for (auto& ev : recovery->searchVariants) { if (ev.first == abi.name) abi.searchVariant = ev.second; if (ev.second == abi.name) abi.isVariant = true; }
  if (tpScratchPerSg) abi.recoveredMaxThreads = tpScratchSgs * 32;
  if (recovery) { auto st = recovery->stagingThreads.find(abi.name); if (st != recovery->stagingThreads.end()) abi.stagingThreads = st->second; }
  if (recovery) { auto ts = recovery->threadScale.find(abi.name); if (ts != recovery->threadScale.end()) abi.threadScale = ts->second; }
  if (recovery) { auto du = recovery->dynSmemUsed.find(abi.name); if (du != recovery->dynSmemUsed.end()) abi.dynSmemUsed = du->second; }
  if (recovery) { auto rl = recovery->dynSmemRelocate.find(abi.name); if (rl != recovery->dynSmemRelocate.end()) { abi.dynSmemCut = rl->second.first; abi.dynSmemShrink = rl->second.second; } }
  if (recovery) { auto as = recovery->assumptions.find(abi.name); if (as != recovery->assumptions.end()) abi.assume = as->second; }

  // --- parameter block ---
  std::string pstruct = "struct __P_" + abi.name + " {\n";
  unsigned off = 0, idx = 0;
  std::string prologue;
  for (Argument& A : Fn.args()) {
    ParamABI p; Type* T = A.getType(); std::string fty;
    if (A.hasByValAttr()) {
      Type* ST = A.getParamByValType();
      p.kind = "struct"; p.size = DL.getTypeAllocSize(ST);
      p.align = A.getParamAlign() ? (unsigned)A.getParamAlign()->value() : DL.getABITypeAlign(ST).value();
      fty = ty(ST);
      // copy to thread memory so the body's generic pointer is a `thread` pointer
      prologue += "  " + fty + " __bv" + std::to_string(idx) + " = __p.a" + std::to_string(idx) + ";\n";
      names[&A] = "((thread uchar*)&__bv" + std::to_string(idx) + ")";
    } else if (T->isPointerTy()) {
      p.kind = "ptr"; p.size = 8; p.align = 8; fty = "ulong";
      names[&A] = "((" + ptrTy(cast<PointerType>(T)->getAddressSpace()) + ")__p.a" + std::to_string(idx) + ")";
    } else if (T->isIntegerTy(1)) {
      p.kind = "i8"; p.size = 1; p.align = 1; fty = "uchar";
      names[&A] = "((bool)(__p.a" + std::to_string(idx) + " & 1))";
    } else {
      p.size = DL.getTypeAllocSize(T); p.align = DL.getABITypeAlign(T).value(); fty = ty(T);
      if (T->isIntegerTy()) p.kind = "i" + std::to_string(intBits(T));
      else if (T->isHalfTy()) p.kind = "f16"; else if (T->isBFloatTy()) p.kind = "bf16"; else if (T->isFloatTy()) p.kind = "f32";
      else p.kind = "struct";
      names[&A] = "__p.a" + std::to_string(idx);
    }
    // Explicit padding to the ABI offset: aggregate params are emitted as packed structs (alignment 1 in MSL),
    // which Metal would otherwise place right after the previous field, off the offset the runtime writes to.
    unsigned aligned = (off + p.align - 1) / p.align * p.align;
    if (aligned > off) pstruct += "  uchar __pad" + std::to_string(idx) + "[" + std::to_string(aligned - off) + "];\n";
    off = aligned; p.offset = off; off += p.size;
    pstruct += "  " + fty + " a" + std::to_string(idx) + ";\n";
    abi.params.push_back(p); idx++;
  }
  abi.paramBlockSize = (off + 7) / 8 * 8;
  if (abi.paramBlockSize == 0) { pstruct += "  uint __pad;\n"; abi.paramBlockSize = 8; }
  pstruct += "};\n";
  curAbi = &abi;

  // --- name values and blocks ---
  unsigned n = 0;
  for (BasicBlock& BB : Fn) {
    bbIds[&BB] = (unsigned)bbIds.size();
    for (Instruction& I : BB) if (!I.getType()->isVoidTy()) names[&I] = "v" + std::to_string(n++);
  }

  // --- declarations ---
  lateDecl.clear();
  declared.clear();
  std::string decls;
  for (BasicBlock& BB : Fn)
    for (Instruction& I : BB) {
      if (I.getType()->isVoidTy()) continue;
      if (auto* AI = dyn_cast<AllocaInst>(&I)) {
        if (!AI->isStaticAlloca()) fail("dynamic alloca");
        Type* AT = AI->getAllocatedType();
        std::string nm = names[AI];
        decls += "  " + ty(AT) + " " + nm + "_mem;\n";
        decls += "  thread uchar* " + nm + " = (thread uchar*)&" + nm + "_mem;\n";
        continue;
      }
      if (I.getType()->isTokenTy()) continue;
      if (isa<AtomicCmpXchgInst>(&I)) { decls += "  " + ty(cast<AtomicCmpXchgInst>(&I)->getCompareOperand()->getType()) + " " + names[&I] + "_old; bool " + names[&I] + "_ok;\n"; continue; }
      if (I.getType()->isStructTy() && isa<CallInst>(&I)) { /* declared by call handler */ }
      bool escape = isa<PHINode>(I);
      if (!escape)
        for (User* U : I.users()) {
          auto* UI = dyn_cast<Instruction>(U);
          if (!UI || UI->getParent() != &BB || isa<PHINode>(UI)) { escape = true; break; }
        }
      std::string t = I.getType()->isPointerTy() ? ptrTy(ptrAS(&I)) : ty(I.getType());
      // structs/vectors are first written as `v.f0 =` / `v[i] =` inside `{ ... }` helpers; keep those kernel-scope
      bool scalar = I.getType()->isFloatingPointTy() || I.getType()->isIntegerTy() || I.getType()->isPointerTy();
      if (escape || !scalar) decls += "  " + t + " " + names[&I] + ";\n";
      else lateDecl[names[&I]] = t;
    }

  // --- body ---
  DominatorTree dt(Fn); PostDominatorTree pdt(Fn); LoopInfo li(dt);
  DT = &dt; PDT = &pdt; LI = &li;
  bool structuredOK = false;
  if (opts.structured) {
    try { emitBodyStructured(); structuredOK = true; }
    catch (EmitError& e) {
      res.warnings.push_back("kernel " + abi.name + ": structured emission failed (" + e.msg + "); using state machine");
      body.clear(); emitted.clear(); declared.clear(); indent = 1;
    }
  }
  if (!structuredOK) emitBodyStateMachine();

  // --- assemble ---
  std::string k;
  k += pstruct;
  // __launch_bounds__(N) becomes the MSL thread bound, which on Apple GPUs is also the register budget: the compiler
  // may spend registers as if only N threads had to be resident, and a 64-thread bound leaves one or two simdgroups
  // per core (decode GEMMs). MVCC_MSL_MIN_MAX_THREADS raises the bound the compiler plans for (launches stay at
  // the host's block size). Default is the source bound — a floor of 256 regresses 64-thread decode GEMVs
  // (fewer registers, same launch size, spill). `0` also keeps the source bound.
  {
    unsigned mslMax = abi.maxThreads;
    // A 64-thread __launch_bounds__ copied here is also Metal's register budget: the compiler
    // spends registers as if only one or two simdgroups had to be resident (decode GEMVs, ~7 GB/s
    // instead of the 243 GB/s stream). Omit the attribute below 128 so the pipeline hint can
    // raise occupancy.
    if (mslMax >= 128) k += "[[max_total_threads_per_threadgroup(" + std::to_string(mslMax) + ")]]\n";
  }
  k += "kernel void " + abi.name + "(constant __P_" + abi.name + "& __p [[buffer(0)]],\n";
  if (usesZeroPage) { k += "    device uchar* __mvcc_zero [[buffer(1)]],\n"; abi.zeroPage = true; }
  if (usesLocks) { k += "    device atomic_uint* __mvcc_locks [[buffer(4)]],\n"; abi.lockPage = true; }
  if (usesGlobals) { k += "    constant uchar* __mvcc_cglobals [[buffer(2)]], device uchar* __mvcc_dglobals [[buffer(3)]],\n"; abi.globals = true; }
  // Shared memory: threadgroup memory, or in the spill build (-D__MVCC_SPILL, prelude) a slot of the runtime's
  // device-memory pool at [[buffer(5)]] taken at entry and released on every return path (__MVCC_RETURN).
  k += "#ifdef __MVCC_SPILL\n    device uchar* __mvcc_spill [[buffer(5)]],\n#else\n    threadgroup uchar* __smem [[threadgroup(0)]],\n#endif\n";
  k += "    uint3 __tid [[thread_position_in_threadgroup]], uint3 __ctaid [[threadgroup_position_in_grid]],\n";
  k += "    uint3 __ntid [[threads_per_threadgroup]], uint3 __nctaid [[threadgroups_per_grid]],\n";
  k += "    uint __lane [[thread_index_in_simdgroup]], uint __sgid [[simdgroup_index_in_threadgroup]]) {\n";
  k += "#ifdef __MVCC_SPILL\n";
  k += "  threadgroup uint __mvcc_slot_tg;\n";
  k += "  if ((__tid.x | __tid.y | __tid.z) == 0u) __mvcc_slot_tg = __mvcc_spill_acquire(__mvcc_spill, __ctaid.x + __nctaid.x * (__ctaid.y + __nctaid.y * __ctaid.z), __ntid.x * __ntid.y * __ntid.z);\n";
  k += "  threadgroup_barrier(mem_flags::mem_threadgroup);\n";
  k += "  const uint __mvcc_slot = __mvcc_slot_tg;\n";
  k += "  device uchar* __smem = __mvcc_spill + ((device __mvcc_spill_hdr*)__mvcc_spill)->data_off + (ulong)__mvcc_slot * (ulong)((device __mvcc_spill_hdr*)__mvcc_spill)->stride;\n";
  k += "  const ulong __mvcc_smem_hi = (ulong)__smem & ~0xfffffffful;\n";  // the slot never straddles a 4 GiB boundary (runtime.rs spill_pool)
  k += "#endif\n";
  k += prologue + decls + body + "}\n\n";
  kernelsOut += k;
  abi.deviceGraphLaunchExecOffsets = std::move(curGraphLaunchExecOffsets);
  curAbi = nullptr;
  res.kernels.push_back(abi);
}

std::optional<unsigned> Emitter::execParamOffset(Value* execVal) {
  if (!curAbi) return std::nullopt;
  Value* V = execVal;
  unsigned extra = 0;
  for (int hop = 0; hop < 24; hop++) {
    V = V->stripPointerCasts();
    if (auto* LI = dyn_cast<LoadInst>(V)) { V = LI->getPointerOperand(); continue; }
    if (auto* GEP = dyn_cast<GEPOperator>(V)) {
      APInt off(64, 0);
      if (!GEP->accumulateConstantOffset(DL, off)) return std::nullopt;
      extra += (unsigned)off.getZExtValue();
      V = GEP->getPointerOperand();
      continue;
    }
    if (auto* A = dyn_cast<Argument>(V)) {
      unsigned idx = A->getArgNo();
      if (idx >= curAbi->params.size()) return std::nullopt;
      return curAbi->params[idx].offset + extra;
    }
    if (auto* AI = dyn_cast<AllocaInst>(V)) {
      (void)AI;
      if (F && F->arg_size() > 0 && F->getArg(0)->hasByValAttr())
        return curAbi->params[0].offset + extra;
      return std::nullopt;
    }
    break;
  }
  return std::nullopt;
}

void Emitter::recordDeviceGraphLaunch(CallInst& CI) {
  if (CI.arg_size() < 1) return;
  if (auto off = execParamOffset(CI.getArgOperand(0)))
    curGraphLaunchExecOffsets.push_back(*off);
}

// ---------------------------------------------------------------------------
// Control flow
// ---------------------------------------------------------------------------

void Emitter::emitPhiCopies(BasicBlock* From, BasicBlock* To) {
  std::vector<std::pair<std::string, Value*>> copies;
  std::set<Value*> phis;
  for (PHINode& P : To->phis()) { phis.insert(&P); copies.push_back({names[&P], P.getIncomingValueForBlock(From)}); }
  if (copies.empty()) return;
  bool needTemps = false;
  for (auto& c : copies) if (phis.count(c.second)) needTemps = true;
  // a null/undef pointer takes the phi's own address space (its generic type says nothing about the space)
  auto src = [&](PHINode* P, Value* V) -> std::string { return V->getType()->isPointerTy() ? ptrIn(V, ptrAS(P)) : val(V); };
  size_t pi = 0; std::vector<PHINode*> pns; for (PHINode& P : To->phis()) pns.push_back(&P);
  if (!needTemps) { for (auto& c : copies) { line(c.first + " = " + src(pns[pi], c.second) + ";"); pi++; } return; }
  line("{");
  indent++;
  for (size_t i = 0; i < copies.size(); i++) {
    Value* V = copies[i].second;
    std::string t = V->getType()->isPointerTy() ? ptrTy(ptrAS(pns[i])) : ty(V->getType());
    line(t + " __t" + std::to_string(i) + " = " + src(pns[i], V) + ";");
  }
  for (size_t i = 0; i < copies.size(); i++) line(copies[i].first + " = __t" + std::to_string(i) + ";");
  indent--;
  line("}");
}

// Closes an `if (...) { ... }` whose other arm is the direct edge From -> Merge. That edge carries phi
// copies too (StructurizeCFG's flow predicates live exactly there), so give it an else arm when needed.
void Emitter::emitDirectEdgeElse(BasicBlock* From, BasicBlock* Merge) {
  bool need = false;
  for (PHINode& P : Merge->phis()) { (void)P; if (P.getBasicBlockIndex(From) >= 0) { need = true; break; } }
  if (!need) { line("}"); return; }
  line("} else {");
  indent++;
  emitPhiCopies(From, Merge);
  indent--;
  line("}");
}

void Emitter::emitBlockInstrs(BasicBlock* BB) {
  if (emitted.count(BB)) fail("block emitted twice: " + std::to_string(bbIds[BB]));
  emitted.insert(BB);
  for (Instruction& I : *BB) {
    if (I.isTerminator() || isa<PHINode>(I)) continue;
    emitInst(I);
  }
}

// CFG reachability that ignores backedges (successor dominates the source).
// Used so an if-arm with stopAt does not walk into the continuation, without
// treating a later loop's backedge as "already past the join".
bool Emitter::forwardReachable(BasicBlock* From, BasicBlock* To) {
  if (!From || !To) return false;
  if (From == To) return true;
  SmallPtrSet<BasicBlock*, 32> seen;
  SmallVector<BasicBlock*, 32> q;
  seen.insert(From);
  q.push_back(From);
  for (size_t i = 0; i < q.size(); ++i) {
    BasicBlock* B = q[i];
    for (BasicBlock* N : successors(B)) {
      if (DT->dominates(N, B)) continue;
      if (N == To) return true;
      if (seen.insert(N).second) q.push_back(N);
    }
  }
  return false;
}

void Emitter::gotoTarget(BasicBlock* From, BasicBlock* S, BasicBlock* stopAt, LoopCtx* ctx) {
  if (S == stopAt) { emitPhiCopies(From, S); return; }
  // The arm walked past the join: S is in the continuation (reachable from
  // stopAt). Emitting it here duplicates the shared tail.
  if (stopAt && forwardReachable(stopAt, S)) return;
  emitPhiCopies(From, S);
  if (ctx && S == ctx->L->getHeader()) { line("continue;"); return; }
  if (ctx && S == ctx->exit) { line("break;"); return; }
  if (ctx && !ctx->L->contains(S)) fail("branch leaves loop to non-exit block " + std::to_string(bbIds[S]));
  emitChain(S, stopAt, ctx);
}

void Emitter::emitChain(BasicBlock* BB, BasicBlock* stopAt, LoopCtx* ctx) {
  if (stopAt && BB != stopAt && forwardReachable(stopAt, BB)) return;
  Loop* L = LI->getLoopFor(BB);
  if (L && L->getHeader() == BB && (!ctx || ctx->L != L)) {
    BasicBlock* exit = L->getExitBlock();
    if (!exit) {
      // No exit: an infinite loop (allowed in CUDA if it ends with a return/trap inside)
      SmallVector<BasicBlock*, 4> exits; L->getExitBlocks(exits);
      if (!exits.empty()) fail("loop with multiple exit blocks");
    }
    LoopCtx inner{L, exit};
    line("for (;;) {");
    indent++;
    emitBlockInstrs(BB);
    // header terminator within loop context
    Instruction* T = BB->getTerminator();
    if (auto* Br = dyn_cast<BranchInst>(T)) {
      if (Br->isUnconditional()) gotoTarget(BB, Br->getSuccessor(0), nullptr, &inner);
      else {
        BasicBlock* Tb = Br->getSuccessor(0), *Fb = Br->getSuccessor(1);
        // merge point inside loop = immediate post-dominator, clamped to the loop
        BasicBlock* Mb = branchMerge(BB, L, exit);
        if (Tb == Mb) { line("if (!(" + val(Br->getCondition()) + ")) {"); indent++; gotoTarget(BB, Fb, Mb, &inner); indent--; emitDirectEdgeElse(BB, Mb); }
        else if (Fb == Mb) { line("if (" + val(Br->getCondition()) + ") {"); indent++; gotoTarget(BB, Tb, Mb, &inner); indent--; emitDirectEdgeElse(BB, Mb); }
        else {
          line("if (" + val(Br->getCondition()) + ") {"); indent++; gotoTarget(BB, Tb, Mb, &inner); indent--;
          line("} else {"); indent++; gotoTarget(BB, Fb, Mb, &inner); indent--; line("}");
        }
        if (Mb && Mb != exit) {
          if (Mb == BB) line("continue;");
          else emitChain(Mb, nullptr, &inner);
        } else if (Mb == exit && exit) line("break;");
      }
    } else if (isa<ReturnInst>(T)) line("__MVCC_RETURN");
    else if (isa<UnreachableInst>(T)) line("__MVCC_RETURN");
    else fail("unsupported terminator in loop header");
    indent--;
    line("}");
    if (exit && exit != stopAt) {
      if (ctx && exit == ctx->L->getHeader()) { emitPhiCopies(BB, exit); line("continue;"); return; }
      if (ctx && exit == ctx->exit) { line("break;"); return; }
      emitChain(exit, stopAt, ctx);
    }
    return;
  }

  emitBlockInstrs(BB);
  Instruction* T = BB->getTerminator();
  if (auto* Br = dyn_cast<BranchInst>(T)) {
    if (Br->isUnconditional()) { gotoTarget(BB, Br->getSuccessor(0), stopAt, ctx); return; }
    BasicBlock* Tb = Br->getSuccessor(0), *Fb = Br->getSuccessor(1);
    BasicBlock* Mb = branchMerge(BB, ctx ? ctx->L : nullptr, ctx ? ctx->exit : nullptr);
    if (!Mb) fail("conditional branch without post-dominator in block " + std::to_string(bbIds[BB]));
    // If the merge is beyond our stop point, the stop point must be the merge (structured property).
    if (stopAt && Mb != stopAt && PDT->dominates(Mb, stopAt) && Mb != stopAt) Mb = stopAt;
    // A recovered join can be the function `ret` (PDT virtual root). That is
    // the right merge only at the top level; inside a region it walks past
    // stopAt and emits the shared tail twice.
    if (stopAt && Mb == uniqueRet && Mb != stopAt) Mb = stopAt;
    if (Tb == Mb) { line("if (!(" + val(Br->getCondition()) + ")) {"); indent++; gotoTarget(BB, Fb, Mb, ctx); indent--; emitDirectEdgeElse(BB, Mb); }
    else if (Fb == Mb) { line("if (" + val(Br->getCondition()) + ") {"); indent++; gotoTarget(BB, Tb, Mb, ctx); indent--; emitDirectEdgeElse(BB, Mb); }
    else {
      line("if (" + val(Br->getCondition()) + ") {"); indent++; gotoTarget(BB, Tb, Mb, ctx); indent--;
      line("} else {"); indent++; gotoTarget(BB, Fb, Mb, ctx); indent--; line("}");
    }
    if (Mb == stopAt) return;
    if (ctx && Mb == ctx->L->getHeader()) { line("continue;"); return; }
    if (ctx && Mb == ctx->exit) { line("break;"); return; }
    emitChain(Mb, stopAt, ctx);
    return;
  }
  if (isa<ReturnInst>(T)) { line("__MVCC_RETURN"); return; }
  if (isa<UnreachableInst>(T)) { line("__MVCC_RETURN"); return; }
  std::string s; raw_string_ostream os(s); T->print(os);
  fail("unsupported terminator: " + s);
}

void Emitter::emitBodyStructured() {
  emitChain(&F->getEntryBlock(), nullptr, nullptr);
  // Sanity: every reachable block must have been emitted exactly once.
  for (BasicBlock& BB : *F) if (!emitted.count(&BB) && DT->isReachableFromEntry(&BB)) fail("reachable block not emitted: " + std::to_string(bbIds[&BB]));
}

void Emitter::emitBodyStateMachine() {
  line("uint __bb = " + std::to_string(bbIds[&F->getEntryBlock()]) + "u;");
  line("for (;;) {");
  indent++;
  line("switch (__bb) {");
  for (BasicBlock& BB : *F) {
    if (!DT->isReachableFromEntry(&BB)) continue;
    line("case " + std::to_string(bbIds[&BB]) + "u: {");
    indent++;
    for (Instruction& I : BB) { if (I.isTerminator() || isa<PHINode>(I)) continue; emitInst(I); }
    Instruction* T = BB.getTerminator();
    if (auto* Br = dyn_cast<BranchInst>(T)) {
      if (Br->isUnconditional()) { emitPhiCopies(&BB, Br->getSuccessor(0)); line("__bb = " + std::to_string(bbIds[Br->getSuccessor(0)]) + "u; break;"); }
      else {
        line("if (" + val(Br->getCondition()) + ") {"); indent++;
        emitPhiCopies(&BB, Br->getSuccessor(0)); line("__bb = " + std::to_string(bbIds[Br->getSuccessor(0)]) + "u;"); indent--;
        line("} else {"); indent++;
        emitPhiCopies(&BB, Br->getSuccessor(1)); line("__bb = " + std::to_string(bbIds[Br->getSuccessor(1)]) + "u;"); indent--;
        line("}"); line("break;");
      }
    } else if (isa<ReturnInst>(T) || isa<UnreachableInst>(T)) line("__MVCC_RETURN");
    else { std::string s; raw_string_ostream os(s); T->print(os); fail("unsupported terminator: " + s); }
    indent--;
    line("}");
  }
  line("default: __MVCC_RETURN");
  line("}");
  indent--;
  line("}");
}

// ---------------------------------------------------------------------------
// Instructions
// ---------------------------------------------------------------------------

static const char* fcmpExpr(CmpInst::Predicate P, const std::string& a, const std::string& b, std::string& out) {
  switch (P) {
    case CmpInst::FCMP_OEQ: out = a + " == " + b; break;
    case CmpInst::FCMP_OGT: out = a + " > " + b; break;
    case CmpInst::FCMP_OGE: out = a + " >= " + b; break;
    case CmpInst::FCMP_OLT: out = a + " < " + b; break;
    case CmpInst::FCMP_OLE: out = a + " <= " + b; break;
    case CmpInst::FCMP_ONE: out = "((" + a + " < " + b + ") || (" + a + " > " + b + "))"; break;
    case CmpInst::FCMP_ORD: out = "(!isnan(" + a + ") && !isnan(" + b + "))"; break;
    case CmpInst::FCMP_UNO: out = "(isnan(" + a + ") || isnan(" + b + "))"; break;
    case CmpInst::FCMP_UEQ: out = "!((" + a + " < " + b + ") || (" + a + " > " + b + "))"; break;
    case CmpInst::FCMP_UGT: out = "!(" + a + " <= " + b + ")"; break;
    case CmpInst::FCMP_UGE: out = "!(" + a + " < " + b + ")"; break;
    case CmpInst::FCMP_ULT: out = "!(" + a + " >= " + b + ")"; break;
    case CmpInst::FCMP_ULE: out = "!(" + a + " > " + b + ")"; break;
    case CmpInst::FCMP_UNE: out = "!(" + a + " == " + b + ")"; break;
    case CmpInst::FCMP_TRUE: out = "true"; break;
    case CmpInst::FCMP_FALSE: out = "false"; break;
    default: return "bad fcmp";
  }
  return nullptr;
}

std::string Emitter::atomicPtr(Value* P, Type* T, bool sign, bool fp) {
  unsigned as = ptrAS(P);
  std::string at;
  if (fp) { if (!T->isFloatTy()) fail("atomic on " + ty(T) + " not supported by Metal"); at = "atomic_float"; }
  else if (T->isIntegerTy(32)) at = sign ? "atomic_int" : "atomic_uint";
  else if (T->isIntegerTy(64)) fail("internal: 64-bit atomic reached the 32-bit atomic path");
  else fail("atomic on " + ty(T) + " not supported by Metal");
  return "((" + asSpace(as) + " " + at + "*)(" + val(P) + "))";
}

// 64-bit atomics: locked plain RMW (prelude __mvcc_atomic64, lock table from the runtime at [[buffer(4)]]).
std::string Emitter::atomic64(Value* P, const char* op, const std::string& v, const std::string& cmp, const std::string& ok) {
  usedHelpers.insert("atomic64"); usesLocks = true;
  return "__mvcc_atomic64(__mvcc_locks, (" + asSpace(ptrAS(P)) + " ulong*)(" + val(P) + "), " + op + ", " + v + ", " + cmp + ", " + ok + ")";
}

// The address space pointers stored to this slot are spelled in: what a load of the slot would infer (all stores to the
// private object, or this constant offset of it, joined). Non-private slots hold raw pointers of the value's own space.
static unsigned slotPtrAS(StoreInst* S) {
  Value* V = S->getValueOperand();
  const DataLayout& DL = S->getModule()->getDataLayout();
  auto* AI = dyn_cast<AllocaInst>(getUnderlyingObject(S->getPointerOperand()));
  if (!AI) return ptrAS(V);
  APInt off(64, 0);
  bool constSlot = S->getPointerOperand()->stripAndAccumulateConstantOffsets(DL, off, true) == AI;
  SmallPtrSet<Value*, 16> visited;
  unsigned a = slotStoresAS(AI, S->getFunction(), constSlot, off, DL.getTypeStoreSize(V->getType()), visited);
  return a == ~0u ? ptrAS(V) : a;
}

std::string Emitter::ptrIn(Value* V, unsigned as) {
  if (V->getType()->isPointerTy() && (isa<ConstantPointerNull>(V) || isa<UndefValue>(V))) return as == AS_MIXED ? "0ul" : "(" + ptrTy(as) + ")0";
  unsigned vs = ptrAS(V);
  if (mergeAS(vs, as) != AS_MIXED || vs == as) return val(V);
  if (as == AS_MIXED) { usedHelpers.insert("tagged"); return "__mvcc_tag(" + val(V) + ", " + std::to_string(tagOf(vs)) + "u)"; }
  if (vs == AS_MIXED) { usedHelpers.insert("tagged"); return "((" + ptrTy(as) + ")__mvcc_untag(" + val(V) + "))"; }
  return "((" + ptrTy(as) + ")(" + val(V) + "))";
}

void Emitter::emitInst(Instruction& I) {
  std::string nm = I.getType()->isVoidTy() ? "" : names[&I];
  auto set = [&](const std::string& e) { line(nm + " = " + e + ";"); };

  if (isa<AllocaInst>(I)) return;  // declared up front
  if (auto* BO = dyn_cast<BinaryOperator>(&I)) {
    std::string a = val(BO->getOperand(0)), b = val(BO->getOperand(1));
    Type* T = BO->getType();
    bool isBool = T->isIntegerTy(1);
    if (T->isDoubleTy()) {
      usedHelpers.insert("f64");
      switch (BO->getOpcode()) {
        case Instruction::FAdd: set("mvcc_f64_add(" + a + ", " + b + ")"); return;
        case Instruction::FSub: set("mvcc_f64_sub(" + a + ", " + b + ")"); return;
        case Instruction::FMul: set("mvcc_f64_mul(" + a + ", " + b + ")"); return;
        case Instruction::FDiv: set("mvcc_f64_div(" + a + ", " + b + ")"); return;
        default: fail("fp64 operation not emulated: frem on double (README.md)");
      }
    }
    switch (BO->getOpcode()) {
      case Instruction::Add: set(isBool ? "(" + a + " ^ " + b + ")" : a + " + " + b); return;
      case Instruction::Sub: set(isBool ? "(" + a + " ^ " + b + ")" : a + " - " + b); return;
      case Instruction::Mul: set(isBool ? "(" + a + " && " + b + ")" : a + " * " + b); return;
      case Instruction::UDiv: set(a + " / " + b); return;
      case Instruction::URem: set(a + " % " + b); return;
      case Instruction::SDiv: set("(" + ty(T) + ")((" + sty(T) + ")" + a + " / (" + sty(T) + ")" + b + ")"); return;
      case Instruction::SRem: set("(" + ty(T) + ")((" + sty(T) + ")" + a + " % (" + sty(T) + ")" + b + ")"); return;
      case Instruction::Shl: set("(" + ty(T) + ")(" + a + " << " + b + ")"); return;
      case Instruction::LShr: set(a + " >> " + b); return;
      case Instruction::AShr: set("(" + ty(T) + ")((" + sty(T) + ")" + a + " >> " + b + ")"); return;
      case Instruction::And: set(a + " & " + b); return;
      case Instruction::Or: set(a + " | " + b); return;
      case Instruction::Xor: set(a + " ^ " + b); return;
      case Instruction::FAdd: set(a + " + " + b); return;
      case Instruction::FSub: set(a + " - " + b); return;
      case Instruction::FMul: set(a + " * " + b); return;
      case Instruction::FDiv: set(a + " / " + b); return;
      case Instruction::FRem: set("fmod(" + a + ", " + b + ")"); return;
      default: break;
    }
    fail("unsupported binary operator");
  }
  if (auto* U = dyn_cast<UnaryOperator>(&I)) {
    if (U->getOpcode() == Instruction::FNeg) { set(U->getType()->isDoubleTy() ? "mvcc_f64_neg(" + val(U->getOperand(0)) + ")" : "-" + val(U->getOperand(0))); return; }
    fail("unsupported unary operator");
  }
  if (auto* C = dyn_cast<ICmpInst>(&I)) {
    std::string a = val(C->getOperand(0)), b = val(C->getOperand(1));
    Type* OT = C->getOperand(0)->getType();
    if (C->isSigned()) { a = "(" + sty(OT) + ")" + a; b = "(" + sty(OT) + ")" + b; }
    const char* op = nullptr;
    switch (C->getPredicate()) {
      case CmpInst::ICMP_EQ: op = "=="; break; case CmpInst::ICMP_NE: op = "!="; break;
      case CmpInst::ICMP_UGT: case CmpInst::ICMP_SGT: op = ">"; break;
      case CmpInst::ICMP_UGE: case CmpInst::ICMP_SGE: op = ">="; break;
      case CmpInst::ICMP_ULT: case CmpInst::ICMP_SLT: op = "<"; break;
      case CmpInst::ICMP_ULE: case CmpInst::ICMP_SLE: op = "<="; break;
      default: fail("bad icmp");
    }
    if (OT->isPointerTy()) {
      unsigned m = mergeAS(ptrAS(C->getOperand(0)), ptrAS(C->getOperand(1)));
      if (m == AS_MIXED) { a = ptrIn(C->getOperand(0), AS_MIXED); b = ptrIn(C->getOperand(1), AS_MIXED); }
      else { a = "(ulong)" + a; b = "(ulong)" + b; }
    }
    set(a + " " + op + " " + b);
    return;
  }
  if (auto* C = dyn_cast<FCmpInst>(&I)) {
    if (C->getOperand(0)->getType()->isDoubleTy()) {
      usedHelpers.insert("f64");
      std::string a = val(C->getOperand(0)), b = val(C->getOperand(1));
      auto lt = [&](const std::string& x, const std::string& y) { return "mvcc_f64_lt(" + x + ", " + y + ")"; };
      auto le = [&](const std::string& x, const std::string& y) { return "mvcc_f64_le(" + x + ", " + y + ")"; };
      std::string eq = "mvcc_f64_eq(" + a + ", " + b + ")", un = "mvcc_f64_unord(" + a + ", " + b + ")";
      std::string e;
      switch (C->getPredicate()) {
        case CmpInst::FCMP_OEQ: e = eq; break;
        case CmpInst::FCMP_OGT: e = lt(b, a); break;
        case CmpInst::FCMP_OGE: e = le(b, a); break;
        case CmpInst::FCMP_OLT: e = lt(a, b); break;
        case CmpInst::FCMP_OLE: e = le(a, b); break;
        case CmpInst::FCMP_ONE: e = "(!" + un + " && !" + eq + ")"; break;
        case CmpInst::FCMP_ORD: e = "!" + un; break;
        case CmpInst::FCMP_UNO: e = un; break;
        case CmpInst::FCMP_UEQ: e = "(" + un + " || " + eq + ")"; break;
        case CmpInst::FCMP_UGT: e = "!" + le(a, b); break;
        case CmpInst::FCMP_UGE: e = "!" + lt(a, b); break;
        case CmpInst::FCMP_ULT: e = "!" + le(b, a); break;
        case CmpInst::FCMP_ULE: e = "!" + lt(b, a); break;
        case CmpInst::FCMP_UNE: e = "!" + eq; break;
        case CmpInst::FCMP_TRUE: e = "true"; break;
        case CmpInst::FCMP_FALSE: e = "false"; break;
        default: fail("bad fcmp");
      }
      set(e); return;
    }
    std::string e; if (fcmpExpr(C->getPredicate(), val(C->getOperand(0)), val(C->getOperand(1)), e)) fail("bad fcmp");
    set(e); return;
  }
  if (auto* S = dyn_cast<SelectInst>(&I)) {
    if (S->getCondition()->getType()->isVectorTy()) set("select(" + val(S->getFalseValue()) + ", " + val(S->getTrueValue()) + ", " + val(S->getCondition()) + ")");
    else if (S->getType()->isPointerTy()) {
      // both arms spelled in the select's inferred space (null/undef arms have no space of their own)
      unsigned as = ptrAS(S);
      set("(" + val(S->getCondition()) + ") ? " + ptrIn(S->getTrueValue(), as) + " : " + ptrIn(S->getFalseValue(), as));
    }
    else set("(" + val(S->getCondition()) + ") ? (" + val(S->getTrueValue()) + ") : (" + val(S->getFalseValue()) + ")");
    return;
  }
  if (auto* CI = dyn_cast<CastInst>(&I)) {
    Value* Op = CI->getOperand(0); Type* ST = Op->getType(), *DT_ = CI->getType();
    std::string a = val(Op);
    if (ST->isDoubleTy() || DT_->isDoubleTy()) {
      usedHelpers.insert("f64");
      switch (CI->getOpcode()) {
        case Instruction::FPExt:  // half/float -> double (half goes through float, exactly)
          set("mvcc_f64_from_f32((float)(" + a + "))"); return;
        case Instruction::FPTrunc:  // double -> float exactly rounded; double -> half via float (double rounding)
          set(DT_->isFloatTy() ? "mvcc_f64_to_f32(" + a + ")" : "(" + ty(DT_) + ")mvcc_f64_to_f32(" + a + ")"); return;
        case Instruction::UIToFP: set("mvcc_f64_from_u64((ulong)(" + a + "))"); return;
        case Instruction::SIToFP: set("mvcc_f64_from_i64((long)(" + sty(ST) + ")(" + a + "))"); return;
        case Instruction::FPToUI: set(cast<IntegerType>(DT_)->getBitWidth() > 32 ? "mvcc_f64_to_u64(" + a + ")" : "(" + ty(DT_) + ")mvcc_f64_to_u32(" + a + ")"); return;
        case Instruction::FPToSI: set(cast<IntegerType>(DT_)->getBitWidth() > 32 ? "(ulong)mvcc_f64_to_i64(" + a + ")" : "(" + ty(DT_) + ")mvcc_f64_to_i32(" + a + ")"); return;
        case Instruction::BitCast: set(DT_->isDoubleTy() ? "mvcc_f64{as_type<ulong>(" + a + ")}" : "as_type<" + ty(DT_) + ">((" + a + ").b)"); return;
        default: fail("unsupported cast involving double");
      }
    }
    switch (CI->getOpcode()) {
      case Instruction::Trunc:
        if (DT_->getScalarType()->isIntegerTy(1)) set("(" + ty(DT_) + ")(" + a + " & 1)");
        else set("(" + ty(DT_) + ")(" + a + ")");
        return;
      case Instruction::ZExt: set("(" + ty(DT_) + ")(" + a + ")"); return;
      case Instruction::SExt:
        if (ST->getScalarType()->isIntegerTy(1)) {
          if (DT_->isVectorTy()) set("(" + ty(DT_) + ")(" + sty(DT_) + "(" + a + ") * -1)");  // bool vec -> 0/-1
          else set("(" + a + ") ? (" + ty(DT_) + ")(-1) : (" + ty(DT_) + ")0");
        } else set("(" + ty(DT_) + ")(" + sty(DT_) + ")(" + sty(ST) + ")(" + a + ")");
        return;
      case Instruction::FPTrunc: case Instruction::FPExt: case Instruction::UIToFP:
        set("(" + ty(DT_) + ")(" + a + ")"); return;
      case Instruction::SIToFP: set("(" + ty(DT_) + ")((" + sty(ST) + ")" + a + ")"); return;
      case Instruction::FPToUI: set("(" + ty(DT_) + ")(" + a + ")"); return;
      case Instruction::FPToSI: set("(" + ty(DT_) + ")((" + sty(DT_) + ")(" + a + "))"); return;
      case Instruction::PtrToInt:
        if (ptrAS(Op) == AS_MIXED) { usedHelpers.insert("tagged"); set("(" + ty(DT_) + ")(__mvcc_untag(" + a + "))"); return; }
        set("(" + ty(DT_) + ")((ulong)(" + a + "))"); return;
      case Instruction::IntToPtr: {
        unsigned das = cast<PointerType>(DT_)->getAddressSpace() == 0 ? ptrAS(CI) : cast<PointerType>(DT_)->getAddressSpace();
        // a 32-bit PTX shared address is a truncated pointer: rebuilt against the shared object's high half (prelude)
        if (das == 3) { set("(" + ptrTy(3) + ")(__MVCC_SMEM_ADDR(" + a + "))"); return; }
        if (das == AS_MIXED) fail("a pointer rebuilt from an integer slot that holds thread memory on some paths and device/threadgroup memory on others (README.md, \"mixed pointers\")");
        set("(" + ptrTy(das) + ")((ulong)(" + a + "))"); return;
      }
      case Instruction::BitCast:
        if (ST->isPointerTy()) set("(" + ty(DT_) + ")(" + a + ")");
        else set("as_type<" + ty(DT_) + ">(" + a + ")");
        return;
      case Instruction::AddrSpaceCast: {
        unsigned s = cast<PointerType>(ST)->getAddressSpace(), d = cast<PointerType>(DT_)->getAddressSpace();
        if (asSpace(s) == asSpace(d)) { set(a); return; }
        // casts through generic keep the MSL pointer as-is; consumers recover the real space via ptrAS()
        if (s == 0 || d == 0) { set(ptrIn(Op, ptrAS(CI))); return; }
        fail("address space cast " + std::to_string(s) + "->" + std::to_string(d) + " survived normalization: a pointer to " + asSpace(s) + " memory escapes to a generic pointer");
      }
      default: fail("unsupported cast");
    }
  }
  if (auto* G = dyn_cast<GetElementPtrInst>(&I)) { set(gepExpr(cast<GEPOperator>(G), [&](Value* v) { return val(v); })); return; }
  if (auto* L = dyn_cast<LoadInst>(&I)) {
    unsigned as = ptrAS(L->getPointerOperand());
    // a loaded pointer is spelled in the space inferred for the load itself (ptrAS looks through the private slot's stores)
    std::string T = L->getType()->isPointerTy() ? ptrTy(ptrAS(L)) : ty(L->getType());
    if (as == AS_MIXED) {
      if (!L->isSimple()) fail("atomic or volatile load through a mixed pointer (README.md, \"mixed pointers\")");
      usedHelpers.insert("tagged");
      set("__mvcc_ld_tagged<" + T + ">(" + val(L->getPointerOperand()) + ", " + std::to_string(L->getAlign().value()) + "u)");
      return;
    }
    if (L->isAtomic() && L->getType()->isIntegerTy(64)) { set("*(" + asSpace(ptrAS(L->getPointerOperand())) + " ulong*)(" + val(L->getPointerOperand()) + ")"); return; }  // single-copy atomic 8-byte load (prelude, "64-bit atomics")
    if (L->isAtomic()) { set("atomic_load_explicit(" + atomicPtr(L->getPointerOperand(), L->getType(), false, L->getType()->isFloatTy()) + ", memory_order_relaxed)"); return; }
    std::string q = L->isVolatile() ? "volatile " : "";
    unsigned align = L->getAlign().value(), sz = DL.getTypeAllocSize(L->getType());
    if (align < sz && (!L->getType()->isVectorTy() || align < DL.getTypeAllocSize(cast<FixedVectorType>(L->getType())->getElementType()))) {
      // under-aligned scalar load (or a vector below even its packed_T alignment): assemble bytewise
      set("__mvcc_load_unaligned<" + T + ">((" + asSpace(as) + " uchar*)(" + val(L->getPointerOperand()) + "))");
      usedHelpers.insert("unaligned");
      return;
    }
    // a volatile load of device memory is coherent (prelude, "coherent device accesses"): a spin on a flag another
    // core stores must see the store
    if (L->isVolatile() && (as == 0 || as == 1) && !L->getType()->isPointerTy()) {
      usedHelpers.insert("coherent");
      set("__mvcc_ld_coherent((device " + T + "*)(" + val(L->getPointerOperand()) + "))");
      return;
    }
    if (align < sz && L->getType()->isVectorTy()) {
      // packed vector load
      auto* VT = cast<FixedVectorType>(L->getType());
      std::string pk = "packed_" + ty(VT->getElementType()) + std::to_string(VT->getNumElements());
      set("(" + T + ")(*(" + asSpace(as) + " " + q + pk + "*)(" + val(L->getPointerOperand()) + "))");
      return;
    }
    // a pointer-typed pointee spells its own address space first: `device uchar* constant*`
    if (L->getType()->isPointerTy() && T != "ulong") set("*(" + T + " " + asSpace(as) + " " + q + "*)(" + val(L->getPointerOperand()) + ")");
    else set("*(" + asSpace(as) + " " + q + T + "*)(" + val(L->getPointerOperand()) + ")");
    return;
  }
  if (auto* S = dyn_cast<StoreInst>(&I)) {
    unsigned as = ptrAS(S->getPointerOperand());
    Type* VT = S->getValueOperand()->getType();
    // a stored pointer is spelled in the space of its slot (what the loads of that slot infer), tagged when the slot mixes
    unsigned slotAS = VT->isPointerTy() ? slotPtrAS(S) : 0;
    std::string T = VT->isPointerTy() ? ptrTy(slotAS) : ty(VT);
    std::string sv = VT->isPointerTy() ? ptrIn(S->getValueOperand(), slotAS) : val(S->getValueOperand());
    if (as == AS_MIXED) {
      if (!S->isSimple()) fail("atomic or volatile store through a mixed pointer (README.md, \"mixed pointers\")");
      usedHelpers.insert("tagged");
      line("__mvcc_st_tagged<" + T + ">(" + val(S->getPointerOperand()) + ", " + sv + ", " + std::to_string(S->getAlign().value()) + "u);");
      return;
    }
    if (S->isAtomic() && VT->isIntegerTy(64)) { line("*(" + asSpace(ptrAS(S->getPointerOperand())) + " ulong*)(" + val(S->getPointerOperand()) + ") = " + val(S->getValueOperand()) + ";"); return; }
    if (S->isAtomic()) { line("atomic_store_explicit(" + atomicPtr(S->getPointerOperand(), VT, false, VT->isFloatTy()) + ", " + val(S->getValueOperand()) + ", memory_order_relaxed);"); return; }
    std::string q = S->isVolatile() ? "volatile " : "";
    unsigned align = S->getAlign().value(), sz = DL.getTypeAllocSize(VT);
    if (align < sz && (!VT->isVectorTy() || align < DL.getTypeAllocSize(cast<FixedVectorType>(VT)->getElementType()))) {
      // under-aligned scalar store (or a vector below even its packed_T alignment): scatter bytewise
      usedHelpers.insert("unaligned");
      line("__mvcc_store_unaligned<" + T + ">((" + asSpace(as) + " uchar*)(" + val(S->getPointerOperand()) + "), " + sv + ");");
      return;
    }
    if (S->isVolatile() && (as == 0 || as == 1) && !VT->isPointerTy()) {
      usedHelpers.insert("coherent");
      line("__mvcc_st_coherent((device " + T + "*)(" + val(S->getPointerOperand()) + "), (" + T + ")(" + sv + "));");
      return;
    }
    if (align < sz && VT->isVectorTy()) {
      auto* FVT = cast<FixedVectorType>(VT);
      std::string pk = "packed_" + ty(FVT->getElementType()) + std::to_string(FVT->getNumElements());
      line("*(" + asSpace(as) + " " + q + pk + "*)(" + val(S->getPointerOperand()) + ") = (" + pk + ")(" + val(S->getValueOperand()) + ");");
      return;
    }
    if (VT->isPointerTy() && T != "ulong") line("*(" + T + " " + asSpace(as) + " " + q + "*)(" + val(S->getPointerOperand()) + ") = " + sv + ";");
    else line("*(" + asSpace(as) + " " + q + T + "*)(" + val(S->getPointerOperand()) + ") = " + sv + ";");
    return;
  }
  if (auto* A = dyn_cast<AtomicRMWInst>(&I)) {
    Type* T = A->getValOperand()->getType();
    std::string v = val(A->getValOperand());
    bool sign = false, fp = false; const char* op = nullptr;
    if (T->isIntegerTy(64)) {
      // Identity RMW (atomicAdd(p, 0), atomicOr(p, 0): CUDA's idiom for an atomic 64-bit read). The locked RMW writes
      // its result with one aligned 8-byte store and aligned 8-byte loads are single-copy atomic on Apple GPUs, so a
      // fenced plain load returns a value some locked RMW returned or produced, exactly as taking the lock would, at
      // no lock traffic. Every thread of a grid reading one counter this way otherwise serializes the grid on a
      // single spinlock (a tokenizer-style grid of identity atomicAdds: seconds per launch).
      if (auto* C = dyn_cast<ConstantInt>(A->getValOperand())) {
        const auto rmw = A->getOperation();
        const bool identity = (C->isZero() && (rmw == AtomicRMWInst::Add || rmw == AtomicRMWInst::Sub || rmw == AtomicRMWInst::Or || rmw == AtomicRMWInst::Xor || rmw == AtomicRMWInst::UMax)) ||
                              (C->isMinusOne() && (rmw == AtomicRMWInst::And || rmw == AtomicRMWInst::UMin));
        if (identity) {
          line("atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_device);");
          set("*(" + asSpace(ptrAS(A->getPointerOperand())) + " volatile ulong*)(" + val(A->getPointerOperand()) + ")");
          line("atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_device);");
          return;
        }
      }
      const char* code = nullptr;
      switch (A->getOperation()) {
        case AtomicRMWInst::Add: code = "MVCC_A64_ADD"; break; case AtomicRMWInst::Sub: code = "MVCC_A64_SUB"; break;
        case AtomicRMWInst::And: code = "MVCC_A64_AND"; break; case AtomicRMWInst::Or: code = "MVCC_A64_OR"; break;
        case AtomicRMWInst::Xor: code = "MVCC_A64_XOR"; break; case AtomicRMWInst::Xchg: code = "MVCC_A64_XCHG"; break;
        case AtomicRMWInst::Max: code = "MVCC_A64_MAX"; break; case AtomicRMWInst::Min: code = "MVCC_A64_MIN"; break;
        case AtomicRMWInst::UMax: code = "MVCC_A64_UMAX"; break; case AtomicRMWInst::UMin: code = "MVCC_A64_UMIN"; break;
        default: fail("unsupported 64-bit atomicrmw operation");
      }
      set(atomic64(A->getPointerOperand(), code, v, "0ul", "nullptr"));
      return;
    }
    switch (A->getOperation()) {
      case AtomicRMWInst::Xchg: op = "atomic_exchange_explicit"; break;
      case AtomicRMWInst::Add: op = "atomic_fetch_add_explicit"; break;
      case AtomicRMWInst::Sub: op = "atomic_fetch_sub_explicit"; break;
      case AtomicRMWInst::And: op = "atomic_fetch_and_explicit"; break;
      case AtomicRMWInst::Or: op = "atomic_fetch_or_explicit"; break;
      case AtomicRMWInst::Xor: op = "atomic_fetch_xor_explicit"; break;
      case AtomicRMWInst::Max: op = "atomic_fetch_max_explicit"; sign = true; break;
      case AtomicRMWInst::Min: op = "atomic_fetch_min_explicit"; sign = true; break;
      case AtomicRMWInst::UMax: op = "atomic_fetch_max_explicit"; break;
      case AtomicRMWInst::UMin: op = "atomic_fetch_min_explicit"; break;
      case AtomicRMWInst::FAdd: op = "atomic_fetch_add_explicit"; fp = true; break;
      case AtomicRMWInst::FSub: op = "atomic_fetch_sub_explicit"; fp = true; break;
      case AtomicRMWInst::FMax: case AtomicRMWInst::FMin: {
        // Metal has no float atomic max/min: CAS loop on the bit pattern (correct for all finite values incl. -0/+0 ordering by IEEE compare)
        usedHelpers.insert("atomic_fmaxmin");
        std::string fn = A->getOperation() == AtomicRMWInst::FMax ? "__mvcc_atomic_fmax" : "__mvcc_atomic_fmin";
        set(fn + "(" + atomicPtr(A->getPointerOperand(), T, false, false) .substr(0) + ", " + v + ")");
        return;
      }
      default: fail("unsupported atomicrmw operation");
    }
    if (T->isFloatTy() && !fp) fail("bitwise atomic on float");
    if (fp && !T->isFloatTy()) fail("float atomic on " + ty(T));
    if (sign) v = "(int)" + v;
    std::string e = std::string(op) + "(" + atomicPtr(A->getPointerOperand(), T, sign, fp) + ", " + v + ", memory_order_relaxed)";
    if (sign) e = "(uint)" + e;
    set(e);
    return;
  }
  if (auto* X = dyn_cast<AtomicCmpXchgInst>(&I)) {
    Type* T = X->getCompareOperand()->getType();
    if (T->isIntegerTy(64)) {
      line("{ bool __ok = false; " + nm + "_old = " + atomic64(X->getPointerOperand(), "MVCC_A64_CAS", val(X->getNewValOperand()), val(X->getCompareOperand()), "&__ok") + "; " + nm + "_ok = __ok; }");
      return;
    }
    std::string p = atomicPtr(X->getPointerOperand(), T, false, false);
    // strong semantics from Metal's weak CAS: retry while the failure was spurious (value still equal to expected)
    line("{ " + ty(T) + " __e = " + val(X->getCompareOperand()) + "; " + ty(T) + " __cmp = __e; bool __ok;");
    line("  do { __e = __cmp; __ok = atomic_compare_exchange_weak_explicit(" + p + ", &__e, " + val(X->getNewValOperand()) + ", memory_order_relaxed, memory_order_relaxed); } while (!__ok && __e == __cmp);");
    line("  " + nm + "_old = __e; " + nm + "_ok = __ok; }");
    return;
  }
  if (auto* EV = dyn_cast<ExtractValueInst>(&I)) {
    Value* Agg = EV->getAggregateOperand();
    if (isa<AtomicCmpXchgInst>(Agg)) { set(names[Agg] + (EV->getIndices()[0] == 0 ? "_old" : "_ok")); return; }
    std::string e = val(Agg); Type* T = Agg->getType();
    for (unsigned idx : EV->getIndices()) {
      if (T->isStructTy()) { e += ".f" + std::to_string(idx); T = T->getStructElementType(idx); }
      else { e += "[" + std::to_string(idx) + "]"; T = T->getArrayElementType(); }
    }
    set(e); return;
  }
  if (auto* IV = dyn_cast<InsertValueInst>(&I)) {
    set(val(IV->getAggregateOperand()));
    std::string e = nm; Type* T = IV->getType();
    for (unsigned idx : IV->getIndices()) {
      if (T->isStructTy()) { e += ".f" + std::to_string(idx); T = T->getStructElementType(idx); }
      else { e += "[" + std::to_string(idx) + "]"; T = T->getArrayElementType(); }
    }
    line(e + " = " + val(IV->getInsertedValueOperand()) + ";"); return;
  }
  if (auto* EE = dyn_cast<ExtractElementInst>(&I)) { set(val(EE->getVectorOperand()) + "[" + val(EE->getIndexOperand()) + "]"); return; }
  if (auto* IE = dyn_cast<InsertElementInst>(&I)) {
    set(val(IE->getOperand(0)));
    line(nm + "[" + val(IE->getOperand(2)) + "] = " + val(IE->getOperand(1)) + ";"); return;
  }
  if (auto* SV = dyn_cast<ShuffleVectorInst>(&I)) {
    auto* VT = cast<FixedVectorType>(SV->getOperand(0)->getType());
    unsigned n = VT->getNumElements();
    std::string a = val(SV->getOperand(0)), b = val(SV->getOperand(1));
    std::string e = ty(SV->getType()) + "(";
    ArrayRef<int> mask = SV->getShuffleMask();
    for (unsigned i = 0; i < mask.size(); i++) {
      if (i) e += ", ";
      int m = mask[i];
      if (m < 0) e += "0";
      else if ((unsigned)m < n) e += a + "[" + std::to_string(m) + "]";
      else e += b + "[" + std::to_string(m - n) + "]";
    }
    set(e + ")"); return;
  }
  if (auto* CI = dyn_cast<CallInst>(&I)) { emitCall(*CI); return; }
  if (auto* Fe = dyn_cast<FenceInst>(&I)) {
    (void)Fe;
    line("atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_device);");
    return;
  }
  if (isa<FreezeInst>(I)) { set(val(I.getOperand(0))); return; }
  std::string s; raw_string_ostream os(s); I.print(os);
  fail("unsupported instruction: " + s);
}

std::string Emitter::mathFn(const std::string& name) { return (opts.fastMath ? "fast::" : "precise::") + name; }

void Emitter::emitCall(CallInst& CI) {
  if (CI.isInlineAsm()) { emitInlineAsm(CI); return; }
  Function* Callee = CI.getCalledFunction();
  if (!Callee) fail("indirect call on device");
  StringRef name = Callee->getName();
  if (Callee->isIntrinsic()) { emitIntrinsic(CI, name); return; }
  if (name.starts_with("__nv_")) { emitLibdevice(CI, name); return; }
  if (name == "__mvcc_dot4") {
    line(names[&CI] + " = dot(float4(" + val(CI.getArgOperand(0)) + ", " + val(CI.getArgOperand(1)) + ", " + val(CI.getArgOperand(2)) + ", " + val(CI.getArgOperand(3)) + "), float4(" + val(CI.getArgOperand(4)) + ", " + val(CI.getArgOperand(5)) + ", " + val(CI.getArgOperand(6)) + ", " + val(CI.getArgOperand(7)) + "));");
    return;
  }
  if (name == "__mvcc_dot2") {
    line(names[&CI] + " = dot(float2(" + val(CI.getArgOperand(0)) + ", " + val(CI.getArgOperand(1)) + "), float2(" + val(CI.getArgOperand(2)) + ", " + val(CI.getArgOperand(3)) + "));");
    return;
  }
  if (name == "__mvcc_simd_sum") { line(names[&CI] + " = simd_sum(" + val(CI.getArgOperand(0)) + ");"); return; }
  if (name == "__mvcc_simd_max") { line(names[&CI] + " = simd_max(" + val(CI.getArgOperand(0)) + ");"); return; }
  if (name == "__mvcc_tile_mma" || name == "__mvcc_tile_coord") { emitTileOp(CI, name); return; }
  if (name == "__mvcc_zero_page") { usesZeroPage = true; line(names[&CI] + " = __mvcc_zero;"); return; }
  if (name.starts_with("__mvcc_cp_async_zsel_")) {
    // guarded_loads.h: (dst, src, valid) -> one unconditional copy; the zero page stands in for src when !valid
    unsigned bytes = std::stoul(name.substr(strlen("__mvcc_cp_async_zsel_")).str());
    std::string vt = bytes == 16 ? "uint4" : bytes == 8 ? "uint2" : "uint";
    Value* d = CI.getArgOperand(0);
    for (int hops = 0; hops < 6; hops++) {
      if (auto* T = dyn_cast<TruncInst>(d)) { d = T->getOperand(0); continue; }
      if (auto* Z = dyn_cast<ZExtInst>(d)) { d = Z->getOperand(0); continue; }
      if (auto* P = dyn_cast<PtrToIntInst>(d)) { d = P->getOperand(0); continue; }
      if (auto* AC = dyn_cast<AddrSpaceCastInst>(d)) { d = AC->getOperand(0); continue; }
      break;
    }
    std::string dst = d->getType()->isPointerTy() && cast<PointerType>(d->getType())->getAddressSpace() == 3 ? "(" + val(d) + ")" : "((__MVCC_SMEM uchar*)__MVCC_SMEM_ADDR(" + val(d) + "))";
    usesZeroPage = true;
    line("*(__MVCC_SMEM " + vt + "*)" + dst + " = *(device " + vt + "*)((" + val(CI.getArgOperand(2)) + ") ? ((device uchar*)(" + val(CI.getArgOperand(1)) + ")) : __mvcc_zero);");
    return;
  }
  if (name.starts_with("__mvcc_ld_zsel_")) {
    // Load half of a zero-selecting copy: (src, valid) → the chunk, from the zero page when !valid.
    unsigned bytes = std::stoul(name.substr(strlen("__mvcc_ld_zsel_")).str());
    std::string vt = bytes == 16 ? "uint4" : bytes == 8 ? "uint2" : "uint";
    usesZeroPage = true;
    line(names[&CI] + " = *(device " + vt + "*)((" + val(CI.getArgOperand(1)) + ") ? ((device uchar*)(" + val(CI.getArgOperand(0)) + ")) : __mvcc_zero);");
    return;
  }
  if (name == "__mvcc_device_graph_launch") {
    recordDeviceGraphLaunch(CI);
    if (!CI.getType()->isVoidTy()) line(names[&CI] + " = 0;"); // cudaSuccess
    return;
  }
  if (name.starts_with("__mvcc_tp_")) { emitTpOp(CI, name); return; }
  if (name.starts_with("__mvcc_")) {
    // pass-through to a prelude helper with the same name and argument order
    std::string e = name.str() + "(";
    for (unsigned i = 0; i < CI.arg_size(); i++) { if (i) e += ", "; e += val(CI.getArgOperand(i)); }
    e += ")";
    if (CI.getType()->isVoidTy()) line(e + ";"); else line(names[&CI] + " = " + e + ";");
    return;
  }
  fail("call to non-inlined function '" + name.str() + "' (recursive or external device functions are not supported)");
}

// mvcc::warp_tile builtins (include/mvcc/tile.cuh) -> Metal 4 TensorOps helpers in the prelude.
//   __mvcc_tile_mma(float* acc, const void* a, int lda, const void* b, int ldb, int M, int N, int K, int elem)
//   int __mvcc_tile_coord(int M, int N, int K, int elem, int i)
void Emitter::emitTileOp(CallInst& CI, StringRef name) {
  auto cst = [&](unsigned i) {
    auto* C = dyn_cast<ConstantInt>(CI.getArgOperand(i));
    if (!C) fail(name.str() + ": tile shape / element type must be compile-time constants");
    return C->getSExtValue();
  };
  auto elemTy = [&](long long e) -> std::string {
    if (e == 0) return "bfloat";
    if (e == 1) return "half";
    fail(name.str() + ": unsupported element type id " + std::to_string(e));
    return "";
  };
  usedHelpers.insert("tile");
  if (name == "__mvcc_tile_mma") {
    long long M = cst(5), N = cst(6), K = cst(7); std::string T = elemTy(cst(8));
    Value* acc = CI.getArgOperand(0); Value* a = CI.getArgOperand(1); Value* b = CI.getArgOperand(3);
    unsigned asA = ptrAS(a), asB = ptrAS(b), asAcc = ptrAS(acc);
    auto space = [&](unsigned as) -> std::string {
      std::string sp = asSpace(as);
      if (sp != "device" && sp != "__MVCC_SMEM") fail("__mvcc_tile_mma: operands must live in device or shared memory (got " + sp + ")");
      return sp;
    };
    if (asSpace(asAcc) != "thread") fail("__mvcc_tile_mma: accumulator must be a register array (thread memory), got " + asSpace(asAcc));
    std::string tmpl = "<" + T + ", " + std::to_string(M) + ", " + std::to_string(N) + ", " + std::to_string(K) + ">";
    line("__mvcc_tile_mma" + tmpl + "((thread float*)(" + val(acc) + "), (" + space(asA) + " " + T + "*)(" + val(a) + "), " +
         castToSigned(CI.getArgOperand(2)) + ", (" + space(asB) + " " + T + "*)(" + val(b) + "), " + castToSigned(CI.getArgOperand(4)) + ");");
    return;
  }
  long long M = cst(0), N = cst(1), K = cst(2); std::string T = elemTy(cst(3));
  std::string tmpl = "<" + T + ", " + std::to_string(M) + ", " + std::to_string(N) + ", " + std::to_string(K) + ">";
  line(names[&CI] + " = (uint)__mvcc_tile_coord" + tmpl + "(" + castToSigned(CI.getArgOperand(4)) + ");");
}

// Tensor recovery builtins (tensor_recovery.cpp) -> prelude helpers (msl/mvcc_prelude.metal, MVCC_NEED_tp).
//   {uint x4}  __mvcc_tp_ld(addr, kind, trans, blockmap)           operand block fill, words in slot order
//   {float xC} __mvcc_tp_mma_MxNxK(M,N,K,type,tl,tr, a..., b..., c...) cooperative matmul2d
//   {float xC} __mvcc_tp_acc2ptx_C(M,N, slots...) / __mvcc_tp_ptx2acc_C(M,N, ptx...)
void Emitter::emitTpOp(CallInst& CI, StringRef name) {
  usedHelpers.insert("tp");
  auto cst = [&](unsigned i) {
    auto* C = dyn_cast<ConstantInt>(CI.getArgOperand(i));
    if (!C) fail(name.str() + ": shape arguments must be constants");
    return (int)C->getSExtValue();
  };
  std::string nm = names[&CI];
  auto* ST = dyn_cast<StructType>(CI.getType());
  unsigned nOut = ST ? ST->getNumElements() : 0;
  // The 32-bit PTX shared address is a truncated pointer; hand the helpers the pointer itself (as the ldmatrix
  // lowering does) so Metal keeps its threadgroup provenance instead of re-deriving a pointer from an integer.
  auto sharedAddr = [&](Value* v) -> std::string {
    Value* cur = v;
    for (int hops = 0; cur && hops < 6; hops++) {
      if (auto* T = dyn_cast<TruncInst>(cur)) { cur = T->getOperand(0); continue; }
      if (auto* Z = dyn_cast<ZExtInst>(cur)) { cur = Z->getOperand(0); continue; }
      if (auto* P = dyn_cast<PtrToIntInst>(cur)) { cur = P->getOperand(0); continue; }
      if (auto* AC = dyn_cast<AddrSpaceCastInst>(cur)) { cur = AC->getOperand(0); continue; }
      break;
    }
    return (cur && cur->getType()->isPointerTy()) ? "(ulong)(" + val(cur) + ")" : "__MVCC_SMEM_ADDR(" + val(v) + ")";
  };
  if (name == "__mvcc_tp_ld") {
    line("{ uint __w[4]; __mvcc_tp_ld(__w, " + sharedAddr(CI.getArgOperand(0)) + ", __lane, " + std::to_string(cst(1)) + ", " + std::to_string(cst(2)) + ", " + std::to_string(cst(3)) + "u);");
    for (unsigned i = 0; i < 4; i++) line("  " + nm + ".f" + std::to_string(i) + " = __w[" + std::to_string(i) + "];");
    line("}");
    return;
  }
  if (name == "__mvcc_tp_frag2slot") {
    // register-built block: (kind, w0..w3 in PTX order) -> the lane's 4 slot words
    line("{ uint __w[4]; __mvcc_tp_frag2slot(__w, " + val(CI.getArgOperand(1)) + ", " + val(CI.getArgOperand(2)) + ", " + val(CI.getArgOperand(3)) + ", " + val(CI.getArgOperand(4)) + ", __lane, " + std::to_string(cst(0)) + ");");
    for (unsigned i = 0; i < 4; i++) line("  " + nm + ".f" + std::to_string(i) + " = __w[" + std::to_string(i) + "];");
    line("}");
    return;
  }
  if (name == "__mvcc_tp_widen_e4m3" || name == "__mvcc_tp_widen_e5m2") {
    // fp8 mma split: 4 bytes of one PTX register -> two half pairs (lo k pair, hi k pair)
    usedHelpers.insert("mma");
    line("{ uint2 __w = mvcc_detail::" + std::string(name == "__mvcc_tp_widen_e4m3" ? "e4m3x4_to_half2x2" : "e5m2x4_to_half2x2") + "(" + val(CI.getArgOperand(0)) + ");");
    line("  " + nm + ".f0 = __w.x; " + nm + ".f1 = __w.y; }");
    return;
  }
  if (name == "__mvcc_tp_srclane") {
    // (lane, j, kind, trans, blockmap) → lane whose ldmatrix row feeds slot quad j
    line(nm + " = __mvcc_tp_srclane(" + val(CI.getArgOperand(0)) + ", " + std::to_string(cst(1)) + ", " + std::to_string(cst(2)) + ", " + std::to_string(cst(3)) + ", " + std::to_string(cst(4)) + "u);");
    return;
  }
  if (name == "__mvcc_tp_ld_direct") {
    // (addr0, addr1, kind, trans, blockmap) → the two source lanes' row addresses
    std::string elemArg;
    if (CI.arg_size() >= 5 && cst(4) == 1) elemArg = ", 1";
    line("{ uint __w[4]; __mvcc_tp_ld_direct(__w, " + sharedAddr(CI.getArgOperand(0)) + ", " + sharedAddr(CI.getArgOperand(1)) + ", __lane, " + std::to_string(cst(2)) + ", " + std::to_string(cst(3)) + elemArg + "); ");
    for (unsigned i = 0; i < 4; i++) line("  " + nm + ".f" + std::to_string(i) + " = __w[" + std::to_string(i) + "];");
    line("}");
    return;
  }
  if (name == "__mvcc_tp_ld_ring") {
    // (addr, kind, trans, blockmap, zmode, obj) → row device address from the pointer ring
    line("{ uint __w[4]; __mvcc_tp_ld_ring(__w, " + sharedAddr(CI.getArgOperand(0)) + ", " + sharedAddr(CI.getArgOperand(5)) + ", __lane, " + std::to_string(cst(1)) + ", " + std::to_string(cst(2)) + ", " + std::to_string(cst(3)) + "u, " + std::to_string(cst(4)) + ");");
    for (unsigned i = 0; i < 4; i++) line("  " + nm + ".f" + std::to_string(i) + " = __w[" + std::to_string(i) + "];");
    line("}");
    return;
  }
  if (name == "__mvcc_tp_stage") {
    // (dst, src, srcsize, zmode, obj) → publish the chunk's device address in the ring
    Value* src = CI.getArgOperand(1);
    for (int hops = 0; src && hops < 4; hops++) { if (auto* AC = dyn_cast<AddrSpaceCastInst>(src)) { src = AC->getOperand(0); continue; } if (auto* P = dyn_cast<IntToPtrInst>(src)) { src = P->getOperand(0); continue; } break; }
    std::string srcE = src->getType()->isPointerTy() ? "(ulong)((device uchar*)(" + val(src) + "))" : "(ulong)(" + val(src) + ")";
    line("__mvcc_tp_stage(" + sharedAddr(CI.getArgOperand(0)) + ", " + sharedAddr(CI.getArgOperand(4)) + ", " + srcE + ", " + val(CI.getArgOperand(2)) + ", " + std::to_string(cst(3)) + ");");
    return;
  }
  if (name.starts_with("__mvcc_tp_mma_")) {
    int M = cst(0), N = cst(1), K = cst(2), type = cst(3), tl = cst(4), tr = cst(5);
    unsigned nA = M * K / 64, nB = K * N / 64, cap = M * N / 32;
    if (CI.arg_size() != 6 + nA + nB + cap || nOut != cap) fail("__mvcc_tp_mma: bad arity");
    std::string T = type == 0 ? "half" : "bfloat";
    std::string blk = "{ uint __a[" + std::to_string(nA) + "] = {";
    for (unsigned i = 0; i < nA; i++) blk += (i ? ", " : "") + val(CI.getArgOperand(6 + i));
    blk += "}; uint __b[" + std::to_string(nB) + "] = {";
    for (unsigned i = 0; i < nB; i++) blk += (i ? ", " : "") + val(CI.getArgOperand(6 + nA + i));
    blk += "}; float __c[" + std::to_string(cap) + "] = {";
    for (unsigned i = 0; i < cap; i++) blk += (i ? ", " : "") + val(CI.getArgOperand(6 + nA + nB + i));
    blk += "}; float __d[" + std::to_string(cap) + "];";
    line(blk);
    line("  __mvcc_tp_mma<" + T + ", " + std::to_string(M) + ", " + std::to_string(N) + ", " + std::to_string(K) + ", " + (tl ? "true" : "false") + ", " + (tr ? "true" : "false") + ">(__d, __a, __b, __c);");
    for (unsigned i = 0; i < cap; i++) line("  " + nm + ".f" + std::to_string(i) + " = __d[" + std::to_string(i) + "];");
    line("}");
    return;
  }
  if (name.starts_with("__mvcc_tp_acc2ptx_") || name.starts_with("__mvcc_tp_ptx2acc_")) {
    int M = cst(0), N = cst(1);
    unsigned cap = M * N / 32;
    if (CI.arg_size() != 2 + cap || nOut != cap) fail(name.str() + ": bad arity");
    std::string blk = "{ float __in[" + std::to_string(cap) + "] = {";
    for (unsigned i = 0; i < cap; i++) blk += (i ? ", " : "") + val(CI.getArgOperand(2 + i));
    blk += "}; float __out[" + std::to_string(cap) + "];";
    line(blk);
    if (!tpScratchPerSg) fail("internal: tensor recovery conversion without scratch reservation");
    std::string scratch = "(__MVCC_SMEM float*)(__smem + " + std::to_string(tpScratchOff) + "u + __sgid * " + std::to_string(tpScratchPerSg) + "u)";
    line("  " + std::string(name.starts_with("__mvcc_tp_acc2ptx_") ? "__mvcc_tp_acc2ptx" : "__mvcc_tp_ptx2acc") + "<" + std::to_string(M) + ", " + std::to_string(N) + ">(__out, __in, __lane, " + scratch + ");");
    for (unsigned i = 0; i < cap; i++) line("  " + nm + ".f" + std::to_string(i) + " = __out[" + std::to_string(i) + "];");
    line("}");
    return;
  }
  if (name.starts_with("__mvcc_tp_accslots_")) {
    // identity conversion: the tile's slots in slot order (the recovered epilogue reads them directly)
    unsigned cap = cst(0) * cst(1) / 32;
    if (CI.arg_size() != 2 + cap || nOut != cap) fail(name.str() + ": bad arity");
    for (unsigned i = 0; i < cap; i++) line(nm + ".f" + std::to_string(i) + " = " + val(CI.getArgOperand(2 + i)) + ";");
    return;
  }
  if (name.starts_with("__mvcc_tp_ctstore_")) {
    // (M,N,K,type,tl,tr, base, ld, rows, cols, v...) with cap = M*N/32 values of the output element type
    int M = cst(0), N = cst(1), K = cst(2), tl = cst(4), tr = cst(5);
    unsigned cap = M * N / 32;
    if (CI.arg_size() != 10 + cap) fail("__mvcc_tp_ctstore: bad arity");
    std::string T = name.ends_with("_f32") ? "float" : name.ends_with("_bf16") ? "bfloat" : "half";  // the output element type
    std::string blk = "{ " + T + " __v[" + std::to_string(cap) + "] = {";
    for (unsigned i = 0; i < cap; i++) blk += (i ? ", " : "") + val(CI.getArgOperand(10 + i));
    blk += "};";
    line(blk);
    line("  __mvcc_tp_ctstore<" + T + ", " + std::to_string(M) + ", " + std::to_string(N) + ", " + std::to_string(K) + ", " + (tl ? "true" : "false") + ", " + (tr ? "true" : "false") + ">(__v, (device " + T + "*)(" + val(CI.getArgOperand(6)) + "), " + castToSigned(CI.getArgOperand(7)) + ", " + castToSigned(CI.getArgOperand(8)) + ", " + castToSigned(CI.getArgOperand(9)) + ");");
    line("}");
    return;
  }
  fail("unknown tensor recovery builtin " + name.str());
}

// One verification kernel per matmul2d descriptor the module uses. It stages small integer-valued operands into
// threadgroup memory laid out exactly as a CUDA kernel's ldmatrix would address them, then runs the same prelude
// path the recovered kernels use (block fill -> cooperative matmul2d -> slot->fragment conversion) and writes the
// result in mma.sync fragment layout. The runtime compares against a CPU product before it trusts recovered
// kernels on this device/OS; any layout-law change in Metal shows up here, not as wrong tensors.
void Emitter::emitVerifyKernels() {
  usedHelpers.insert("tp");
  kernelsOut += "#ifndef __MVCC_SPILL\n";
  for (const TpDescriptor& d : recovery->descriptors) {
    VerifyKernelABI v; v.name = "__mvcc_tp_verify_" + d.tag(); v.M = d.M; v.N = d.N; v.K = d.K; v.tl = d.tl; v.tr = d.tr; v.type = d.type;
    res.verifyKernels.push_back(v);
    const std::string T = d.mslType();
    const int MB = d.M / 16, NB = d.N / 16, KB = d.K / 16, nA = d.M * d.K / 64, nB = d.K * d.N / 64, cap = d.M * d.N / 32;
    std::string k;
    k += "kernel void " + v.name + "(device const ushort* A [[buffer(0)]], device const ushort* B [[buffer(1)]], device float* out [[buffer(2)]],\n";
    k += "    uint __lane [[thread_index_in_simdgroup]]) {\n";
    // A logical [m][k]; memory rows m (k contiguous) unless tl -> rows k (m contiguous). B logical [k][n]; memory rows n
    // (k contiguous) when tr, else rows k (n contiguous). Row pitch = the contiguous extent (+8 halves padding).
    const int aRows = d.tl ? d.K : d.M, aCols = d.tl ? d.M : d.K, bRows = d.tr ? d.N : d.K, bCols = d.tr ? d.K : d.N;
    const int aLd = aCols + 8, bLd = bCols + 8;
    k += "  threadgroup ushort As[" + std::to_string(aRows * aLd) + "]; threadgroup ushort Bs[" + std::to_string(bRows * bLd) + "];\n";
    k += "  for (uint i = __lane; i < " + std::to_string(aRows * aCols) + "u; i += 32u) As[(i / " + std::to_string(aCols) + "u) * " + std::to_string(aLd) + "u + i % " + std::to_string(aCols) + "u] = A[i];\n";
    k += "  for (uint i = __lane; i < " + std::to_string(bRows * bCols) + "u; i += 32u) Bs[(i / " + std::to_string(bCols) + "u) * " + std::to_string(bLd) + "u + i % " + std::to_string(bCols) + "u] = B[i];\n";
    k += "  simdgroup_barrier(mem_flags::mem_threadgroup);\n";
    k += "  uint i8 = __lane >> 3, j8 = __lane & 7;\n";  // matrix i8, row j8 of it (PTX ldmatrix lane addressing)
    k += "  uint a[" + std::to_string(nA) + "]; uint b[" + std::to_string(nB) + "]; float c[" + std::to_string(cap) + "]; float dst[" + std::to_string(cap) + "]; float ptx[" + std::to_string(cap) + "];\n";
    // A sub-blocks: identity block map (matrix r = block r = (m>=8) + 2*(k>=8)); word order per tensor_recovery.cpp
    for (int mb = 0; mb < MB; mb++) for (int kb = 0; kb < KB; kb++) {
      int order = d.K == 32 ? kb + KB * mb : mb;
      std::string row, col;  // memory row / col of this lane's ldmatrix row address within the sub-block
      if (!d.tl) { row = std::to_string(mb * 16) + "u + 8u * (i8 & 1u) + j8"; col = std::to_string(kb * 16) + "u + 8u * (i8 >> 1)"; }
      else       { row = std::to_string(kb * 16) + "u + 8u * (i8 >> 1) + j8"; col = std::to_string(mb * 16) + "u + 8u * (i8 & 1u)"; }
      k += "  { ulong addr = (ulong)(threadgroup uchar*)&As[(" + row + ") * " + std::to_string(aLd) + "u + (" + col + ")];\n";
      k += "    __mvcc_tp_ld(a + " + std::to_string(4 * order) + ", addr, __lane, 0, " + (d.tl ? "1" : "0") + ", 0xE4u); }\n";  // 0xE4 = identity map 3,2,1,0 packed
    }
    // B sub-blocks: block r = 2*(n>=8) + (k>=8); matrix r = block r
    for (int nb = 0; nb < NB; nb++) for (int kb = 0; kb < KB; kb++) {
      int order = d.N == 32 ? nb + NB * kb : kb;
      std::string row, col;
      if (d.tr) { row = std::to_string(nb * 16) + "u + 8u * (i8 >> 1) + j8"; col = std::to_string(kb * 16) + "u + 8u * (i8 & 1u)"; }   // rows n, k contiguous; non-.trans ldmatrix
      else      { row = std::to_string(kb * 16) + "u + 8u * (i8 & 1u) + j8"; col = std::to_string(nb * 16) + "u + 8u * (i8 >> 1)"; }   // rows k, n contiguous; .trans ldmatrix
      k += "  { ulong addr = (ulong)(threadgroup uchar*)&Bs[(" + row + ") * " + std::to_string(bLd) + "u + (" + col + ")];\n";
      k += "    __mvcc_tp_ld(b + " + std::to_string(4 * order) + ", addr, __lane, 1, " + (d.tr ? "0" : "1") + ", 0xE4u); }\n";
    }
    k += "  for (int i = 0; i < " + std::to_string(cap) + "; i++) c[i] = 0.0f;\n";
    k += "  __mvcc_tp_mma<" + T + ", " + std::to_string(d.M) + ", " + std::to_string(d.N) + ", " + std::to_string(d.K) + ", " + (d.tl ? "true" : "false") + ", " + (d.tr ? "true" : "false") + ">(dst, a, b, c);\n";
    k += "  threadgroup float scratch[128];\n";
    k += "  __mvcc_tp_acc2ptx<" + std::to_string(d.M) + ", " + std::to_string(d.N) + ">(ptx, dst, __lane, scratch);\n";
    k += "  for (int i = 0; i < " + std::to_string(cap) + "; i++) out[__lane * " + std::to_string(cap) + " + i] = ptx[i];\n";
    // round trip through ptx2acc as well so both conversions are verified
    k += "  float back[" + std::to_string(cap) + "]; __mvcc_tp_ptx2acc<" + std::to_string(d.M) + ", " + std::to_string(d.N) + ">(back, ptx, __lane, scratch);\n";
    k += "  float err = 0.0f; for (int i = 0; i < " + std::to_string(cap) + "; i++) err += fabs(back[i] - dst[i]);\n";
    k += "  out[32 * " + std::to_string(cap) + " + __lane] = err;\n";
    // the epilogue store: the tile through the T destination tensor, once in full and once clipped to (M-3) x (N-5)
    // into a region the runtime pre-fills; both are compared with the CPU product (and the clipped one's border)
    k += "  { " + T + " hv[" + std::to_string(cap) + "]; for (int i = 0; i < " + std::to_string(cap) + "; i++) hv[i] = (" + T + ")dst[i];\n";
    k += "    device " + T + "* full = (device " + T + "*)(out + 32 * " + std::to_string(cap) + " + 32);\n";
    k += "    __mvcc_tp_ctstore<" + T + ", " + std::to_string(d.M) + ", " + std::to_string(d.N) + ", " + std::to_string(d.K) + ", " + (d.tl ? "true" : "false") + ", " + (d.tr ? "true" : "false") + ">(hv, full, " + std::to_string(d.N) + ", " + std::to_string(d.M) + ", " + std::to_string(d.N) + ");\n";
    k += "    __mvcc_tp_ctstore<" + T + ", " + std::to_string(d.M) + ", " + std::to_string(d.N) + ", " + std::to_string(d.K) + ", " + (d.tl ? "true" : "false") + ", " + (d.tr ? "true" : "false") + ">(hv, full + " + std::to_string(d.M * d.N) + ", " + std::to_string(d.N + 2) + ", " + std::to_string(d.M - 3) + ", " + std::to_string(d.N - 5) + "); }\n";
    k += "}\n\n";
    kernelsOut += k;
  }
  kernelsOut += "#endif\n";
}

void Emitter::emitIntrinsic(CallInst& CI, StringRef name) {
  std::string nm = CI.getType()->isVoidTy() ? "" : names[&CI];
  auto set = [&](const std::string& e) { line(nm + " = " + e + ";"); };
  auto arg = [&](unsigned i) { return val(CI.getArgOperand(i)); };
  auto sarg = [&](unsigned i) { return castToSigned(CI.getArgOperand(i)); };
  Type* T = CI.getType();

  // --- NVVM special registers ---
  static const std::map<std::string, std::string> sregs = {
    {"llvm.nvvm.read.ptx.sreg.tid.x", "__tid.x"}, {"llvm.nvvm.read.ptx.sreg.tid.y", "__tid.y"}, {"llvm.nvvm.read.ptx.sreg.tid.z", "__tid.z"},
    {"llvm.nvvm.read.ptx.sreg.ntid.x", "__ntid.x"}, {"llvm.nvvm.read.ptx.sreg.ntid.y", "__ntid.y"}, {"llvm.nvvm.read.ptx.sreg.ntid.z", "__ntid.z"},
    {"llvm.nvvm.read.ptx.sreg.ctaid.x", "__ctaid.x"}, {"llvm.nvvm.read.ptx.sreg.ctaid.y", "__ctaid.y"}, {"llvm.nvvm.read.ptx.sreg.ctaid.z", "__ctaid.z"},
    {"llvm.nvvm.read.ptx.sreg.nctaid.x", "__nctaid.x"}, {"llvm.nvvm.read.ptx.sreg.nctaid.y", "__nctaid.y"}, {"llvm.nvvm.read.ptx.sreg.nctaid.z", "__nctaid.z"},
    {"llvm.nvvm.read.ptx.sreg.laneid", "__lane"}, {"llvm.nvvm.read.ptx.sreg.warpsize", "32u"},
    {"llvm.nvvm.read.ptx.sreg.warpid", "__sgid"},
  };
  auto sr = sregs.find(name.str());
  if (sr != sregs.end()) { set(sr->second); return; }

  if (name.starts_with("llvm.nvvm.barrier") || name == "llvm.nvvm.barrier0" || name.starts_with("llvm.nvvm.bar.sync")) {
    line("threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"); return;
  }
  if (name.starts_with("llvm.nvvm.bar.warp.sync")) { line("simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"); return; }
  if (name.starts_with("llvm.nvvm.membar.cta")) { line("atomic_thread_fence(mem_flags::mem_threadgroup | mem_flags::mem_device, memory_order_seq_cst, thread_scope_threadgroup);"); return; }
  if (name.starts_with("llvm.nvvm.membar")) { line("atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_device);"); return; }

  // --- warp shuffles: (mask, val, lane/delta, clamp/width-pack) ---
  if (name.starts_with("llvm.nvvm.shfl.sync.")) {
    // arg1 value, arg2 lane or delta, arg3 packed clamp|segmask. CUDA's __shfl_*_sync(mask, v, l, width) encodes
    // c = ((32 - width) << 8) | 0x1f for idx/bfly, and ((32-width)<<8) | 0 for up... we implement the PTX semantics exactly.
    std::string kind = name.substr(strlen("llvm.nvvm.shfl.sync.")).split('.').first.str();
    std::string v = arg(1), b = arg(2), c = arg(3);
    usedHelpers.insert("shfl");
    std::string fn = "__mvcc_shfl_" + kind;
    if (T->isFloatTy()) set("as_type<float>(" + fn + "(as_type<uint>(" + v + "), " + b + ", " + c + ", __lane))");
    else set(fn + "(" + v + ", " + b + ", " + c + ", __lane)");
    return;
  }
  if (name.starts_with("llvm.nvvm.redux.sync.")) {
    // __reduce_*_sync(mask, v): lanes outside mask contribute the identity (CUDA requires them not to participate)
    std::string op = name.drop_front(strlen("llvm.nvvm.redux.sync.")).str();
    std::string v = arg(0), mask = arg(1), in = "(((" + mask + ") >> __lane) & 1u)";
    std::string T_ = ty(T), S_ = sty(T);
    if (op == "add") set("simd_sum(" + in + " ? " + v + " : (" + T_ + ")0)");
    else if (op == "and") set("simd_and(" + in + " ? " + v + " : (" + T_ + ")0xffffffffu)");
    else if (op == "or") set("simd_or(" + in + " ? " + v + " : (" + T_ + ")0)");
    else if (op == "xor") set("simd_xor(" + in + " ? " + v + " : (" + T_ + ")0)");
    else if (op == "umin") set("simd_min(" + in + " ? " + v + " : (" + T_ + ")0xffffffffu)");
    else if (op == "umax") set("simd_max(" + in + " ? " + v + " : (" + T_ + ")0)");
    else if (op == "min") set("(" + T_ + ")simd_min(" + in + " ? (" + S_ + ")" + v + " : (" + S_ + ")0x7fffffff)");
    else if (op == "max") set("(" + T_ + ")simd_max(" + in + " ? (" + S_ + ")" + v + " : (" + S_ + ")0x80000000)");
    else fail("unsupported warp reduction " + name.str());
    return;
  }
  if (name == "llvm.nvvm.fns") { usedHelpers.insert("bits"); set("__mvcc_fns(" + arg(0) + ", " + arg(1) + ", " + sarg(2) + ")"); return; }
  if (name.starts_with("llvm.nvvm.vote.ballot")) { set("(uint)(simd_vote::vote_t)simd_ballot(" + arg(1) + ") & " + arg(0)); return; }
  if (name.starts_with("llvm.nvvm.vote.any")) { set("simd_any(" + arg(1) + ")"); return; }
  if (name.starts_with("llvm.nvvm.vote.all")) { set("simd_all(" + arg(1) + ")"); return; }
  if (name.starts_with("llvm.nvvm.vote.uni")) { set("(simd_all(" + arg(1) + ") || !simd_any(" + arg(1) + "))"); return; }
  if (name == "llvm.nvvm.activemask") { set("(uint)(simd_vote::vote_t)simd_active_threads_mask()"); return; }

  // --- NVVM math ---
  if (name.starts_with("llvm.nvvm.fma.rn")) { set("fma(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")"); return; }
  if (name.starts_with("llvm.nvvm.rsqrt.approx")) { set("rsqrt(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.rcp.approx") || name.starts_with("llvm.nvvm.rcp.rn")) { set("1.0f / " + arg(0)); return; }
  if (name.starts_with("llvm.nvvm.sqrt.approx") || name.starts_with("llvm.nvvm.sqrt.rn")) { set("sqrt(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.ex2.approx")) { set("exp2(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.lg2.approx")) { set("log2(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.sin.approx")) { set("fast::sin(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.cos.approx")) { set("fast::cos(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.fmin")) { set("fmin(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (name.starts_with("llvm.nvvm.fmax")) { set("fmax(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (name.starts_with("llvm.nvvm.fabs")) { set("fabs(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.round")) { set("rint(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.floor")) { set("floor(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.ceil")) { set("ceil(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.trunc")) { set("trunc(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.saturate")) { set("saturate(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.f2h.rn")) { set("as_type<ushort>((half)" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.f2bf16.rn")) { set("as_type<ushort>((bfloat)" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.ff2bf16x2.rn")) { set("as_type<uint>(bfloat2((bfloat)" + arg(1) + ", (bfloat)" + arg(0) + "))"); return; }
  if (name.starts_with("llvm.nvvm.ff2f16x2.rn")) { set("as_type<uint>(half2((half)" + arg(1) + ", (half)" + arg(0) + "))"); return; }
  if (name.starts_with("llvm.nvvm.f2i.rn")) { set("(uint)(int)rint(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.f2i.rz")) { set("(uint)(int)(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.f2i.rm")) { set("(uint)(int)floor(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.f2i.rp")) { set("(uint)(int)ceil(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.f2ui.rn")) { set("(uint)rint(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.f2ui.rz")) { set("(uint)(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.i2f")) { set("(float)(int)" + arg(0)); return; }
  if (name.starts_with("llvm.nvvm.ui2f")) { set("(float)" + arg(0)); return; }
  if (name.starts_with("llvm.nvvm.mulhi.ui")) { set("mulhi(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (name.starts_with("llvm.nvvm.mulhi.i")) { set("(uint)mulhi((int)" + arg(0) + ", (int)" + arg(1) + ")"); return; }
  if (name.starts_with("llvm.nvvm.mul24.ui")) { set("(" + arg(0) + " & 0xffffffu) * (" + arg(1) + " & 0xffffffu)"); return; }
  if (name.starts_with("llvm.nvvm.prmt")) { usedHelpers.insert("prmt"); set("__mvcc_prmt(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")"); return; }
  if (name.starts_with("llvm.nvvm.bitcast")) { set("as_type<" + ty(T) + ">(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.nvvm.ptr.")) { set(arg(0)); return; }  // address space no-ops after normalization
  if (name == "llvm.nvvm.nanosleep") { return; }
  if (name.starts_with("llvm.nvvm.atomic.load.add.f32")) { set("atomic_fetch_add_explicit(" + atomicPtr(CI.getArgOperand(0), T, false, true) + ", " + arg(1) + ", memory_order_relaxed)"); return; }
  if (name.starts_with("llvm.nvvm.isspacep.shared")) { set("false"); return; }  // after IAS this only remains for genuinely generic pointers which we treat as device
  if (name.starts_with("llvm.nvvm.isspacep.global")) { set("true"); return; }

  // --- generic LLVM intrinsics ---
  if (name.starts_with("llvm.lifetime") || name.starts_with("llvm.dbg") || name.starts_with("llvm.assume") || name.starts_with("llvm.experimental.noalias") || name.starts_with("llvm.var.annotation")) return;
  if (name.starts_with("llvm.expect")) { set(arg(0)); return; }
  if (T->isDoubleTy() || (CI.arg_size() && CI.getArgOperand(0)->getType()->isDoubleTy())) {
    usedHelpers.insert("f64");
    static const std::map<std::string, std::string> f64un = {
      {"llvm.fabs.f64", "mvcc_f64_abs"}, {"llvm.sqrt.f64", "mvcc_f64_sqrt"}, {"llvm.floor.f64", "mvcc_f64_floor"}, {"llvm.ceil.f64", "mvcc_f64_ceil"},
      {"llvm.trunc.f64", "mvcc_f64_trunc"}, {"llvm.rint.f64", "mvcc_f64_rint"}, {"llvm.nearbyint.f64", "mvcc_f64_rint"}, {"llvm.roundeven.f64", "mvcc_f64_rint"},
      {"llvm.round.f64", "mvcc_f64_round"},
    };
    static const std::map<std::string, std::string> f64bin = {
      {"llvm.copysign.f64", "mvcc_f64_copysign"}, {"llvm.minnum.f64", "mvcc_f64_fmin"}, {"llvm.maxnum.f64", "mvcc_f64_fmax"},
    };
    auto u = f64un.find(name.str()); if (u != f64un.end()) { set(u->second + "(" + arg(0) + ")"); return; }
    auto b2 = f64bin.find(name.str()); if (b2 != f64bin.end()) { set(b2->second + "(" + arg(0) + ", " + arg(1) + ")"); return; }
    if (name == "llvm.fmuladd.f64") { set("mvcc_f64_add(mvcc_f64_mul(" + arg(0) + ", " + arg(1) + "), " + arg(2) + ")"); return; }
    if (name == "llvm.is.fpclass.f64" || name == "llvm.fma.f64" || name.starts_with("llvm.exp") || name.starts_with("llvm.log") || name.starts_with("llvm.pow") || name.starts_with("llvm.sin") || name.starts_with("llvm.cos"))
      fail("fp64 operation not emulated: " + name.str() + " (software binary64 covers + - * / sqrt, rounding, comparisons and conversions; README.md)");
  }
  if (name.starts_with("llvm.fma.")) { set("fma(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")"); return; }
  if (name.starts_with("llvm.fmuladd.")) { set("fma(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")"); return; }
  if (name.starts_with("llvm.fabs.")) { set("fabs(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.sqrt.")) { set(mathFn("sqrt") + "(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.minnum.")) { set("fmin(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (name.starts_with("llvm.maxnum.")) { set("fmax(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (name.starts_with("llvm.minimum.")) { set("(isnan(" + arg(0) + ")||isnan(" + arg(1) + ")) ? (" + ty(T) + ")NAN : fmin(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (name.starts_with("llvm.maximum.")) { set("(isnan(" + arg(0) + ")||isnan(" + arg(1) + ")) ? (" + ty(T) + ")NAN : fmax(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (name.starts_with("llvm.floor.")) { set("floor(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.ceil.")) { set("ceil(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.trunc.")) { set("trunc(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.rint.") || name.starts_with("llvm.nearbyint.")) { set("rint(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.round.")) { set("round(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.roundeven.")) { set("rint(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.copysign.")) { set("copysign(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (name.starts_with("llvm.exp2.")) { set(mathFn("exp2") + "(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.exp.")) { set(mathFn("exp") + "(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.log2.")) { set(mathFn("log2") + "(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.log.")) { set(mathFn("log") + "(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.pow.")) { set(mathFn("pow") + "(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (name.starts_with("llvm.sin.")) { set(mathFn("sin") + "(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.cos.")) { set(mathFn("cos") + "(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.ctpop.")) { set("(" + ty(T) + ")popcount(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.ctlz.")) { set("(" + ty(T) + ")clz(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.cttz.")) { set("(" + ty(T) + ")ctz(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.bswap.")) { usedHelpers.insert("bits"); set("__mvcc_bswap(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.bitreverse.")) { set("reverse_bits(" + arg(0) + ")"); return; }
  if (name.starts_with("llvm.fshl.")) { usedHelpers.insert("bits"); set("__mvcc_fshl(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")"); return; }
  if (name.starts_with("llvm.fshr.")) { usedHelpers.insert("bits"); set("__mvcc_fshr(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")"); return; }
  if (name.starts_with("llvm.abs.")) { set("(" + ty(T) + ")abs(" + sarg(0) + ")"); return; }
  if (name.starts_with("llvm.smax.")) { set("(" + ty(T) + ")max(" + sarg(0) + ", " + sarg(1) + ")"); return; }
  if (name.starts_with("llvm.smin.")) { set("(" + ty(T) + ")min(" + sarg(0) + ", " + sarg(1) + ")"); return; }
  if (name.starts_with("llvm.umax.")) { set("max(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (name.starts_with("llvm.umin.")) { set("min(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (name.starts_with("llvm.usub.sat.")) { set("(" + arg(0) + " > " + arg(1) + ") ? (" + arg(0) + " - " + arg(1) + ") : (" + ty(T) + ")0"); return; }
  if (name.starts_with("llvm.uadd.sat.")) { set("addsat(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (name.starts_with("llvm.sadd.sat.")) { set("(" + ty(T) + ")addsat(" + sarg(0) + ", " + sarg(1) + ")"); return; }
  if (name.starts_with("llvm.ssub.sat.")) { set("(" + ty(T) + ")subsat(" + sarg(0) + ", " + sarg(1) + ")"); return; }
  if (name.starts_with("llvm.umul.with.overflow.") || name.starts_with("llvm.uadd.with.overflow.") || name.starts_with("llvm.usub.with.overflow.") ||
      name.starts_with("llvm.smul.with.overflow.") || name.starts_with("llvm.sadd.with.overflow.") || name.starts_with("llvm.ssub.with.overflow.")) {
    usedHelpers.insert("overflow");
    std::string k = name.substr(5, 4).str();  // umul / uadd / ...
    Type* ET = cast<StructType>(T)->getElementType(0);
    set("__mvcc_" + k + "_ov<" + ty(ET) + ", " + sty(ET) + ">(" + arg(0) + ", " + arg(1) + ")");
    return;
  }
  if (name.starts_with("llvm.memcpy.")) {
    usedHelpers.insert("mem");
    unsigned das = ptrAS(CI.getArgOperand(0));
    unsigned sas = ptrAS(CI.getArgOperand(1));
    if (das == AS_MIXED || sas == AS_MIXED) {
      usedHelpers.insert("tagged");
      line("__mvcc_memcpy_tagged(" + ptrIn(CI.getArgOperand(0), AS_MIXED) + ", " + ptrIn(CI.getArgOperand(1), AS_MIXED) + ", (ulong)(" + arg(2) + "));");
      return;
    }
    line("__mvcc_memcpy((" + asSpace(das) + " uchar*)(" + arg(0) + "), (" + asSpace(sas) + " uchar*)(" + arg(1) + "), (ulong)(" + arg(2) + "));");
    return;
  }
  if (name.starts_with("llvm.memmove.")) {
    usedHelpers.insert("mem");
    unsigned das = ptrAS(CI.getArgOperand(0));
    unsigned sas = ptrAS(CI.getArgOperand(1));
    line("__mvcc_memmove((" + asSpace(das) + " uchar*)(" + arg(0) + "), (" + asSpace(sas) + " uchar*)(" + arg(1) + "), (ulong)(" + arg(2) + "));");
    return;
  }
  if (name.starts_with("llvm.memset.")) {
    usedHelpers.insert("mem");
    unsigned das = ptrAS(CI.getArgOperand(0));
    if (das == AS_MIXED) { usedHelpers.insert("tagged"); line("__mvcc_memset_tagged(" + arg(0) + ", (uchar)(" + arg(1) + "), (ulong)(" + arg(2) + "));"); return; }
    line("__mvcc_memset((" + asSpace(das) + " uchar*)(" + arg(0) + "), (uchar)(" + arg(1) + "), (ulong)(" + arg(2) + "));");
    return;
  }
  if (name.starts_with("llvm.trap") || name.starts_with("llvm.debugtrap")) { line("return;  // __trap()"); return; }
  if (name.starts_with("llvm.is.fpclass")) {
    unsigned mask = (unsigned)cast<ConstantInt>(CI.getArgOperand(1))->getZExtValue();
    std::string a = arg(0), e;
    auto add = [&](const std::string& s) { if (!e.empty()) e += " || "; e += s; };
    if (mask & 0x3) add("isnan(" + a + ")");
    if (mask & 0x204) add("isinf(" + a + ")");
    if (mask & 0x108) add("isnormal(" + a + ")");
    if (mask & 0x90) add("(fabs(" + a + ") > 0 && !isnormal(" + a + ") && !isinf(" + a + ") && !isnan(" + a + "))");
    if (mask & 0x60) add("(" + a + " == 0)");
    set(e.empty() ? "false" : "(" + e + ")");
    return;
  }
  // SLP / FmaDotPack emit horizontal reduces over the MSL-legal 2/3/4-lane vectors they just built.
  if (name.starts_with("llvm.vector.reduce.fadd.")) {
    Value* accV = CI.getArgOperand(0);
    Value* vecV = CI.getArgOperand(1);
    if (auto* BO = dyn_cast<BinaryOperator>(vecV); BO && BO->getOpcode() == Instruction::FMul) {
      std::string d = "dot(" + val(BO->getOperand(0)) + ", " + val(BO->getOperand(1)) + ")";
      if (auto* CFP = dyn_cast<ConstantFP>(accV); CFP && CFP->isZero()) { set(d); return; }
      set("(" + val(accV) + " + " + d + ")");
      return;
    }
  }
  if (name.starts_with("llvm.vector.reduce.")) {
    bool startAcc = name.starts_with("llvm.vector.reduce.fadd.") || name.starts_with("llvm.vector.reduce.fmul.");
    Value* vecOp = CI.getArgOperand(startAcc ? 1 : 0);
    auto* VT = dyn_cast<FixedVectorType>(vecOp->getType());
    if (!VT) fail("vector.reduce on a non-fixed vector");
    unsigned n = VT->getNumElements();
    std::string v = val(vecOp);
    auto lane = [&](unsigned i) { return v + "[" + std::to_string(i) + "]"; };
    const char* bin = nullptr;
    const char* fn = nullptr;
    bool signedFn = false;
    if (name.starts_with("llvm.vector.reduce.add.") || name.starts_with("llvm.vector.reduce.fadd.")) bin = "+";
    else if (name.starts_with("llvm.vector.reduce.mul.") || name.starts_with("llvm.vector.reduce.fmul.")) bin = "*";
    else if (name.starts_with("llvm.vector.reduce.and.")) bin = "&";
    else if (name.starts_with("llvm.vector.reduce.or.")) bin = "|";
    else if (name.starts_with("llvm.vector.reduce.xor.")) bin = "^";
    else if (name.starts_with("llvm.vector.reduce.smax.")) { fn = "max"; signedFn = true; }
    else if (name.starts_with("llvm.vector.reduce.smin.")) { fn = "min"; signedFn = true; }
    else if (name.starts_with("llvm.vector.reduce.umax.")) fn = "max";
    else if (name.starts_with("llvm.vector.reduce.umin.")) fn = "min";
    else if (name.starts_with("llvm.vector.reduce.fmax.")) fn = "fmax";
    else if (name.starts_with("llvm.vector.reduce.fmin.")) fn = "fmin";
    else fail("unsupported intrinsic: " + name.str());
    auto wrap = [&](const std::string& e) {
      return signedFn ? ("(" + sty(VT->getElementType()) + ")(" + e + ")") : e;
    };
    std::string acc = startAcc ? arg(0) : wrap(lane(0));
    for (unsigned i = startAcc ? 0 : 1; i < n; i++) {
      std::string x = wrap(lane(i));
      acc = fn ? (std::string(fn) + "(" + acc + ", " + x + ")") : ("(" + acc + " " + bin + " " + x + ")");
    }
    if (signedFn) acc = "(" + ty(T) + ")(" + acc + ")";
    set(acc);
    return;
  }
  fail("unsupported intrinsic: " + name.str());
}

void Emitter::emitLibdevice(CallInst& CI, StringRef name) {
  // Libdevice function *names* are the interface clang's math headers target; we implement them with MSL.
  std::string nm = CI.getType()->isVoidTy() ? "" : names[&CI];
  auto set = [&](const std::string& e) { line(nm + " = " + e + ";"); };
  auto arg = [&](unsigned i) { return val(CI.getArgOperand(i)); };
  std::string n = name.drop_front(5).str();  // strip "__nv_"
  bool fast = false;
  if (n.rfind("fast_", 0) == 0) { fast = true; n = n.substr(5); }
  auto mf = [&](const std::string& f) { return (fast || opts.fastMath ? "fast::" : "precise::") + f; };
  auto un = [&](const char* nvname, const std::string& e) { if (n == nvname) { set(e); return true; } return false; };
  #define UN(nv, msl) if (n == nv) { set(std::string(msl) + "(" + arg(0) + ")"); return; }
  #define UNM(nv, msl) if (n == nv) { set(mf(msl) + "(" + arg(0) + ")"); return; }
  #define BIN(nv, msl) if (n == nv) { set(std::string(msl) + "(" + arg(0) + ", " + arg(1) + ")"); return; }
  #define BINM(nv, msl) if (n == nv) { set(mf(msl) + "(" + arg(0) + ", " + arg(1) + ")"); return; }
  UNM("expf", "exp") UNM("exp2f", "exp2") UNM("exp10f", "exp10") UNM("logf", "log") UNM("log2f", "log2") UNM("log10f", "log10")
  UNM("sinf", "sin") UNM("cosf", "cos") UNM("tanf", "tan") UNM("tanhf", "tanh") UNM("sinhf", "sinh") UNM("coshf", "cosh")
  UNM("asinf", "asin") UNM("acosf", "acos") UNM("atanf", "atan") UNM("asinhf", "asinh") UNM("acoshf", "acosh") UNM("atanhf", "atanh")
  UNM("sqrtf", "sqrt") UNM("rsqrtf", "rsqrt") UNM("cbrtf", "cbrt") UNM("rcbrtf", "rsqrt")  // rcbrt handled below
  UN("fabsf", "fabs") UN("floorf", "floor") UN("ceilf", "ceil") UN("truncf", "trunc") UN("rintf", "rint") UN("nearbyintf", "rint") UN("roundf", "round")
  if (n == "erff") { set("__mvcc_erf(" + arg(0) + ")"); usedHelpers.insert("erf"); return; }
  if (n == "erfcf") { set("__mvcc_erfc(" + arg(0) + ")"); usedHelpers.insert("erf"); return; }
  if (n == "lgammaf" || n == "tgammaf") fail(n + ": no Metal equivalent (not implemented in the prelude yet)");
  BIN("fmaxf", "fmax") BIN("fminf", "fmin") BIN("fmodf", "fmod") BINM("powf", "pow") BINM("atan2f", "atan2") BIN("copysignf", "copysign") BIN("fdimf", "fdim") BIN("hypotf", "hypot")
  BIN("remainderf", "remainder") BIN("nextafterf", "nextafter")
  if (n == "log1pf") { set("__mvcc_log1p(" + arg(0) + ")"); usedHelpers.insert("log1p"); return; }
  if (n == "expm1f") { set("__mvcc_expm1(" + arg(0) + ")"); usedHelpers.insert("log1p"); return; }
  if (n == "rcbrtf") { set("1.0f / " + mf("cbrt") + "(" + arg(0) + ")"); return; }
  if (n == "fmaf") { set("fma(" + arg(0) + ", " + arg(1) + ", " + arg(2) + ")"); return; }
  if (n == "fdividef" || n == "fast_fdividef") { set(arg(0) + " / " + arg(1)); return; }
  if (n == "fdiv_rn") { set(arg(0) + " / " + arg(1)); return; }
  if (n == "fmul_rn") { set(arg(0) + " * " + arg(1)); return; }
  if (n == "fadd_rn") { set(arg(0) + " + " + arg(1)); return; }
  if (n == "fsub_rn") { set(arg(0) + " - " + arg(1)); return; }
  if (n == "fsqrt_rn") { set("precise::sqrt(" + arg(0) + ")"); return; }
  if (n == "frsqrt_rn") { set("precise::rsqrt(" + arg(0) + ")"); return; }
  if (n == "frcp_rn") { set("1.0f / " + arg(0)); return; }
  if (n == "saturatef") { set("saturate(" + arg(0) + ")"); return; }
  if (n == "isnanf") { set("(uint)isnan(" + arg(0) + ")"); return; }
  if (n == "isinff") { set("(uint)isinf(" + arg(0) + ")"); return; }
  if (n == "finitef") { set("(uint)isfinite(" + arg(0) + ")"); return; }
  if (n == "signbitf") { set("(uint)signbit(" + arg(0) + ")"); return; }
  if (n == "float_as_int" || n == "float_as_uint") { set("as_type<uint>(" + arg(0) + ")"); return; }
  if (n == "int_as_float" || n == "uint_as_float") { set("as_type<float>(" + arg(0) + ")"); return; }
  if (n == "float2int_rn") { set("(uint)(int)rint(" + arg(0) + ")"); return; }
  if (n == "float2int_rz") { set("(uint)(int)(" + arg(0) + ")"); return; }
  if (n == "float2int_rd") { set("(uint)(int)floor(" + arg(0) + ")"); return; }
  if (n == "float2int_ru") { set("(uint)(int)ceil(" + arg(0) + ")"); return; }
  if (n == "float2uint_rn") { set("(uint)rint(" + arg(0) + ")"); return; }
  if (n == "float2uint_rz") { set("(uint)(" + arg(0) + ")"); return; }
  if (n == "int2float_rn") { set("(float)(int)" + arg(0)); return; }
  if (n == "uint2float_rn") { set("(float)" + arg(0)); return; }
  if (n == "ll2float_rn") { set("(float)(long)" + arg(0)); return; }
  if (n == "ull2float_rn") { set("(float)" + arg(0)); return; }
  if (n == "float2ll_rn") { set("(ulong)(long)rint(" + arg(0) + ")"); return; }
  if (n == "float2ll_rz") { set("(ulong)(long)(" + arg(0) + ")"); return; }
  if (n == "float2ull_rz") { set("(ulong)(" + arg(0) + ")"); return; }
  if (n == "llrintf" || n == "llroundf") { set("(ulong)(long)" + std::string(n == "llrintf" ? "rint" : "round") + "(" + arg(0) + ")"); return; }
  if (n == "lrintf") { set("(ulong)(long)rint(" + arg(0) + ")"); return; }
  if (n == "popc") { set("popcount(" + arg(0) + ")"); return; }
  if (n == "popcll") { set("(uint)popcount(" + arg(0) + ")"); return; }
  if (n == "clz") { set("clz(" + arg(0) + ")"); return; }
  if (n == "clzll") { set("(uint)clz(" + arg(0) + ")"); return; }
  if (n == "ffs") { set("(" + arg(0) + " == 0u) ? 0u : (ctz(" + arg(0) + ") + 1u)"); return; }
  if (n == "ffsll") { set("(" + arg(0) + " == 0ul) ? 0u : ((uint)ctz(" + arg(0) + ") + 1u)"); return; }
  if (n == "brev") { set("reverse_bits(" + arg(0) + ")"); return; }
  if (n == "brevll") { set("reverse_bits(" + arg(0) + ")"); return; }
  // __byte_perm uses only bits <2:0> of each selector nibble (no sign replication, unlike raw prmt)
  if (n == "byte_perm") { usedHelpers.insert("prmt"); set("__mvcc_prmt(" + arg(0) + ", " + arg(1) + ", (" + arg(2) + ") & 0x7777u)"); return; }
  if (n == "umulhi") { set("mulhi(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (n == "mulhi") { set("(uint)mulhi((int)" + arg(0) + ", (int)" + arg(1) + ")"); return; }
  if (n == "umul64hi") { set("mulhi(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (n == "mul64hi") { set("(ulong)mulhi((long)" + arg(0) + ", (long)" + arg(1) + ")"); return; }
  if (n == "umul24") { set("(" + arg(0) + " & 0xffffffu) * (" + arg(1) + " & 0xffffffu)"); return; }
  if (n == "mul24") { set("(uint)(((int)(" + arg(0) + " << 8) >> 8) * ((int)(" + arg(1) + " << 8) >> 8))"); return; }
  if (n == "usad") { set("absdiff(" + arg(0) + ", " + arg(1) + ") + " + arg(2)); return; }
  if (n == "sad") { set("(uint)absdiff((int)" + arg(0) + ", (int)" + arg(1) + ") + " + arg(2)); return; }
  if (n == "abs") { set("(uint)abs((int)" + arg(0) + ")"); return; }
  if (n == "llabs") { set("(ulong)abs((long)" + arg(0) + ")"); return; }
  if (n == "max") { set("(uint)max((int)" + arg(0) + ", (int)" + arg(1) + ")"); return; }
  if (n == "min") { set("(uint)min((int)" + arg(0) + ", (int)" + arg(1) + ")"); return; }
  if (n == "umax") { set("max(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (n == "umin") { set("min(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (n == "llmax") { set("(ulong)max((long)" + arg(0) + ", (long)" + arg(1) + ")"); return; }
  if (n == "llmin") { set("(ulong)min((long)" + arg(0) + ", (long)" + arg(1) + ")"); return; }
  if (n == "ullmax") { set("max(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (n == "ullmin") { set("min(" + arg(0) + ", " + arg(1) + ")"); return; }
  if (n == "ldexpf" || n == "scalbnf") { set("ldexp(" + arg(0) + ", (int)" + arg(1) + ")"); return; }
  if (n == "sincosf") {
    unsigned as1 = ptrAS(CI.getArgOperand(1));
    unsigned as2 = ptrAS(CI.getArgOperand(2));
    line("*(" + asSpace(as1) + " float*)(" + arg(1) + ") = " + mf("sin") + "(" + arg(0) + "); *(" + asSpace(as2) + " float*)(" + arg(2) + ") = " + mf("cos") + "(" + arg(0) + ");");
    return;
  }
  if (n == "frexpf") {
    unsigned as = ptrAS(CI.getArgOperand(1));
    line("{ int __e; " + nm + " = frexp(" + arg(0) + ", __e); *(" + asSpace(as) + " int*)(" + arg(1) + ") = __e; }");
    return;
  }
  if (n == "modff") {
    unsigned as = ptrAS(CI.getArgOperand(1));
    line("{ float __i; " + nm + " = modf(" + arg(0) + ", __i); *(" + asSpace(as) + " float*)(" + arg(1) + ") = __i; }");
    return;
  }
  (void)un;
  #undef UN
  #undef UNM
  #undef BIN
  #undef BINM
  {
    // fp64 libdevice entry points covered by the software binary64 (README.md)
    static const std::map<std::string, std::string> f64fn = {
      {"fabs", "mvcc_f64_abs"}, {"sqrt", "mvcc_f64_sqrt"}, {"floor", "mvcc_f64_floor"}, {"ceil", "mvcc_f64_ceil"}, {"trunc", "mvcc_f64_trunc"},
      {"rint", "mvcc_f64_rint"}, {"nearbyint", "mvcc_f64_rint"}, {"round", "mvcc_f64_round"}, {"fmin", "mvcc_f64_fmin"}, {"fmax", "mvcc_f64_fmax"},
      {"copysign", "mvcc_f64_copysign"}, {"dadd_rn", "mvcc_f64_add"}, {"dsub_rn", "mvcc_f64_sub"}, {"dmul_rn", "mvcc_f64_mul"}, {"ddiv_rn", "mvcc_f64_div"},
      {"dsqrt_rn", "mvcc_f64_sqrt"}, {"double2float_rn", "mvcc_f64_to_f32"}, {"float2double_rn", "mvcc_f64_from_f32"},
      {"double2int_rz", "mvcc_f64_to_i32"}, {"double2uint_rz", "mvcc_f64_to_u32"}, {"double2ll_rz", "mvcc_f64_to_i64"}, {"double2ull_rz", "mvcc_f64_to_u64"},
      {"int2double_rn", "mvcc_f64_from_i64"}, {"uint2double_rn", "mvcc_f64_from_u64"}, {"ll2double_rn", "mvcc_f64_from_i64"}, {"ull2double_rn", "mvcc_f64_from_u64"},
      {"isnand", "mvcc_f64_isnan"}, {"isinfd", "mvcc_f64_isinf"},
    };
    auto f = f64fn.find(n);
    if (f != f64fn.end() && (CI.getType()->isDoubleTy() || (CI.arg_size() && CI.getArgOperand(0)->getType()->isDoubleTy()))) {
      usedHelpers.insert("f64");
      std::string call = f->second + "(";
      for (unsigned i = 0; i < CI.arg_size(); i++) {
        if (i) call += ", ";
        Type* AT = CI.getArgOperand(i)->getType();
        if (n == "int2double_rn" || n == "ll2double_rn") call += "(long)(" + sty(AT) + ")(" + val(CI.getArgOperand(i)) + ")";
        else if (n == "uint2double_rn" || n == "ull2double_rn") call += "(ulong)(" + val(CI.getArgOperand(i)) + ")";
        else call += val(CI.getArgOperand(i));
      }
      call += ")";
      if (CI.getType()->isIntegerTy(32) && (n == "isnand" || n == "isinfd")) call = "(uint)(" + call + ")";
      else if (CI.getType()->isIntegerTy(64) && n == "double2ll_rz") call = "(ulong)" + call;
      else if (CI.getType()->isIntegerTy(32) && n == "double2int_rz") call = "(uint)" + call;
      set(call); return;
    }
  }
  if (n.find("double") != std::string::npos || (!n.empty() && n.back() != 'f' && (n == "exp" || n == "log" || n == "sqrt" || n == "pow" || n == "sin" || n == "cos" || n == "fma" || n == "floor" || n == "fabs" || n == "rint" || n == "tanh")))
    fail("fp64 libdevice function __nv_" + n + " is not emulated (software binary64 covers + - * / sqrt, rounding, comparisons and conversions; README.md)");
  fail("unsupported libdevice function __nv_" + n);
}

// ---------------------------------------------------------------------------
// Inline PTX
// ---------------------------------------------------------------------------

void Emitter::emitInlineAsm(CallInst& CI) {
  auto* IA = cast<InlineAsm>(CI.getCalledOperand());
  std::string text = std::string(IA->getAsmString());
  std::vector<PtxInstr> instrs; std::string err;
  if (!parsePtxAsm(text, instrs, err)) fail("cannot parse inline PTX: " + err);

  // Map $N operand numbers. LLVM constraint strings list outputs ("=f") first, then inputs in order; a source
  // "+f" becomes an output plus a tied input ("0") which clang appends AFTER the ordinary inputs. Call args
  // correspond, in order, to the non-output, non-clobber constraints.
  std::string cons = std::string(IA->getConstraintString());
  std::vector<std::string> cparts; { std::stringstream ss(cons); std::string p; while (std::getline(ss, p, ',')) cparts.push_back(p); }
  unsigned nOut = 0;
  std::vector<int> inputArg;          // operand index k >= nOut  ->  call arg index (k - nOut)
  std::vector<int> tiedArgForOutput;  // output k -> call arg index of its tied input, or -1
  {
    unsigned argIdx = 0;
    for (auto& p : cparts) {
      if (p.empty() || p[0] == '~') continue;
      if (p[0] == '=' || p[0] == '+') { nOut++; tiedArgForOutput.push_back(-1); continue; }
      if (isdigit((unsigned char)p[0])) {
        unsigned o = (unsigned)atoi(p.c_str());
        if (o < tiedArgForOutput.size()) tiedArgForOutput[o] = (int)argIdx;
      }
      inputArg.push_back((int)argIdx++);
    }
  }
  auto inArg = [&](unsigned k) -> Value* {
    if (k < nOut) {
      if (tiedArgForOutput[k] < 0) fail("PTX reads write-only asm operand %" + std::to_string(k) + " in: " + text);
      return CI.getArgOperand(tiedArgForOutput[k]);
    }
    unsigned i = k - nOut;
    if (i >= inputArg.size() || (unsigned)inputArg[i] >= CI.arg_size()) fail("asm operand %" + std::to_string(k) + " out of range in: " + text);
    return CI.getArgOperand(inputArg[i]);
  };
  // Expression for operand %k as an rvalue
  auto inExpr = [&](unsigned k) -> std::string { return val(inArg(k)); };
  // Lvalue names for outputs: scalar result or struct fields
  std::string nm = CI.getType()->isVoidTy() ? "" : names[&CI];
  auto outLV = [&](unsigned k) -> std::string {
    if (k >= nOut) fail("PTX writes input operand %" + std::to_string(k) + " in: " + text);
    if (nOut == 1) return nm;
    return nm + ".f" + std::to_string(k);
  };
  // Recover the pointer behind a "shared address" integer operand (trunc(ptrtoint(addrspacecast ptr))), else int->ptr cast.
  auto sharedPtr = [&](unsigned k) -> std::string {
    Value* V = k < nOut ? nullptr : inArg(k);
    Value* cur = V;
    for (int hops = 0; cur && hops < 6; hops++) {
      if (auto* T = dyn_cast<TruncInst>(cur)) { cur = T->getOperand(0); continue; }
      if (auto* Z = dyn_cast<ZExtInst>(cur)) { cur = Z->getOperand(0); continue; }
      if (auto* P = dyn_cast<PtrToIntInst>(cur)) { cur = P->getOperand(0); continue; }
      if (auto* AC = dyn_cast<AddrSpaceCastInst>(cur)) { cur = AC->getOperand(0); continue; }
      if (auto* CE = dyn_cast<ConstantExpr>(cur)) { if (CE->getOpcode() == Instruction::PtrToInt || CE->getOpcode() == Instruction::AddrSpaceCast) { cur = CE->getOperand(0); continue; } }
      break;
    }
    if (cur && cur->getType()->isPointerTy()) {
      unsigned as = cast<PointerType>(cur->getType())->getAddressSpace();
      if (as == 3) return "(" + val(cur) + ")";
      if (as == 0) return "((__MVCC_SMEM uchar*)__MVCC_SMEM_ADDR(" + val(cur) + "))";
      fail("PTX shared-memory operand is a pointer to " + asSpace(as) + " memory in: " + text);
    }
    return "((__MVCC_SMEM uchar*)__MVCC_SMEM_ADDR(" + inExpr(k) + "))";
  };
  auto globalPtr = [&](unsigned k) -> std::string {
    Value* V = inArg(k);
    Value* cur = V;
    for (int hops = 0; cur && hops < 4; hops++) {
      if (auto* P = dyn_cast<PtrToIntInst>(cur)) { cur = P->getOperand(0); continue; }
      if (auto* AC = dyn_cast<AddrSpaceCastInst>(cur)) { cur = AC->getOperand(0); continue; }
      break;
    }
    if (cur && cur->getType()->isPointerTy()) return "((device uchar*)(" + val(cur) + "))";
    return "((device uchar*)(ulong)(" + inExpr(k) + "))";
  };
  auto reg = [&](const PtxOperand& o) { if (o.kind != PtxOperand::Reg) fail("expected register operand in: " + text); return (unsigned)o.reg; };
  // Temps from `{ .reg .b16 t; ... }`: one MSL uint per name, written and read like a $N operand.
  std::map<std::string, std::string> namedLV;
  for (const auto& in0 : instrs)
    for (const auto& o : in0.ops)
      if (o.kind == PtxOperand::Named && !namedLV.count(o.name)
          && o.name != "globaltimer" && o.name != "clock" && o.name != "clock64"
          && o.name != "laneid" && o.name != "warpid") {
        std::string v = (nm.empty() ? "__ptx" : nm) + "_" + o.name;
        namedLV[o.name] = v;
        line("uint " + v + " = 0u;");
      }
  auto opR = [&](const PtxOperand& o) -> std::string {
    if (o.kind == PtxOperand::Named) return namedLV.at(o.name);
    if (o.kind == PtxOperand::Reg) return inExpr((unsigned)o.reg);
    if (o.kind == PtxOperand::Imm) return std::to_string(o.imm) + "u";
    fail("bad rvalue operand in: " + text);
  };
  auto opW = [&](const PtxOperand& o) -> std::string {
    if (o.kind == PtxOperand::Named) return namedLV.at(o.name);
    if (o.kind == PtxOperand::Reg) return outLV((unsigned)o.reg);
    fail("bad lvalue operand in: " + text);
  };

  for (auto& in : instrs) {
    std::string guard;
    if (!in.predName.empty()) fail("inline PTX predicate register `" + in.predName + "` declared inside the asm block: only the ignore-src cp.async form (`setp; cp.async ..., p`) is understood, as a single copy (see lowerPredicatedCopies); write other predicates as C++ `if` around the asm: " + text);
    if (in.pred >= 0) guard = std::string("if (") + (in.predNeg ? "!" : "") + inExpr(in.pred) + ") ";
    const std::string& m = in.mnemonic;

    if (m == "cp" && in.mods.size() >= 1 && in.mods[0] == "async") {
      if (in.hasMod("commit_group") || in.hasMod("wait_group") || in.hasMod("wait_all")) {
        // Copies are performed synchronously at issue; the groups are already complete.
        // A simdgroup barrier keeps the "wait" a compiler scheduling barrier so following smem reads aren't hoisted.
        line("// " + in.full());
        continue;
      }
      if (in.hasMod("shared") && in.hasMod("global")) {
        if (in.ops.size() < 3) fail("cp.async needs 3 operands: " + text);
        long long bytes = in.ops[2].imm;
        std::string dst = sharedPtr(in.ops[0].reg), src = globalPtr(in.ops[1].reg);
        std::string vt = bytes == 16 ? "uint4" : bytes == 8 ? "uint2" : bytes == 4 ? "uint" : "";
        if (vt.empty()) fail("cp.async size must be 4/8/16: " + text);
        if (in.ops.size() >= 4) {
          // src-size (zfill) variant: copy src_size bytes, zero the rest
          std::string ss = inExpr(reg(in.ops[3]));
          usedHelpers.insert("cpasync");
          line(guard + "__mvcc_cp_async_zfill_" + std::to_string(bytes) + "(" + dst + ", " + src + ", " + ss + ");");
        } else {
          line(guard + "*(__MVCC_SMEM " + vt + "*)" + dst + " = *(device " + vt + "*)" + src + ";");
        }
        continue;
      }
      fail("unsupported cp.async form: " + text);
    }
    if ((m == "ld" || m == "st") && in.hasMod("global")) {
      // ld.global{.hint}{.vN}.<type> / st.global{.hint}{.vN}.<type>. Register width comes from the operand
      // (constraint r/h/l/f/d), memory type from the mnemonic; loads sign/zero-extend, stores truncate.
      std::string t = in.mods.back();
      static const std::map<std::string, std::pair<std::string, bool>> memTypes = {
        {"s8", {"char", true}}, {"u8", {"uchar", false}}, {"b8", {"uchar", false}},
        {"s16", {"short", true}}, {"u16", {"ushort", false}}, {"b16", {"ushort", false}}, {"f16", {"half", false}},
        {"s32", {"int", true}}, {"u32", {"uint", false}}, {"b32", {"uint", false}}, {"f32", {"float", false}},
        {"s64", {"long", true}}, {"u64", {"ulong", false}}, {"b64", {"ulong", false}},
      };
      if (t == "f64") fail("fp64 memory access in inline PTX (__ldcg/__stcg on double): Apple GPUs have no double");
      auto mt = memTypes.find(t);
      if (mt == memTypes.end()) fail("unsupported " + m + " type " + t + " in: " + text);
      unsigned n = in.hasMod("v4") ? 4 : in.hasMod("v2") ? 2 : 1;
      std::string et = mt->second.first; bool sgn = mt->second.second;
      std::string vt = n == 1 ? et : et + std::to_string(n);
      // Memory-ordering forms (PTX ISA memory consistency model): ld.acquire / st.release / .relaxed with a .cta
      // or .gpu scope, and .volatile; the L2 cache hints .cg / .cv (loads) and .cg / .wt (stores). Metal has relaxed
      // atomics only; an aligned 8-byte plain access is single-copy atomic on Apple GPUs (README, 64-bit atomics), so
      // the access is a coherent access (prelude, "coherent device accesses": performed at L2, so a spin on a flag
      // sees another core's store) and the ordering is a device-scope fence after the load (acquire) / before the
      // store (release).
      const bool acquire = in.hasMod("acquire"), release = in.hasMod("release");
      const bool scoped = acquire || release || in.hasMod("relaxed") || in.hasMod("volatile") || in.hasMod("gpu") || in.hasMod("cta") || in.hasMod("sys") ||
                          in.hasMod("cg") || in.hasMod("cv") || in.hasMod("wt");
      if (in.hasMod("acq_rel") || in.hasMod("mmio")) fail("unsupported " + m + " ordering in: " + text);
      if (scoped) usedHelpers.insert("coherent");
      const std::string fence = "atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_device);";
      if (m == "ld") {
        if (in.ops.size() != 2 || in.ops[1].kind != PtxOperand::Mem) fail("bad ld operands: " + text);
        std::string ptr = globalPtr(in.ops[1].reg);
        if (in.ops[1].imm) ptr = "(" + ptr + " + " + std::to_string(in.ops[1].imm) + ")";
        const std::string ld = scoped ? "__mvcc_ld_coherent((device " + vt + "*)" + ptr + ")" : "*(device " + vt + "*)" + ptr;
        // register type from the call's result
        Type* RT = CI.getType(); if (RT->isStructTy()) RT = RT->getStructElementType(0);
        std::string regT = ty(RT), sregT = sty(RT);
        auto conv = [&](const std::string& e) { return sgn ? "(" + regT + ")(" + sregT + ")(" + e + ")" : "(" + regT + ")(" + e + ")"; };
        if (n == 1) { line(guard + outLV(reg(in.ops[0])) + " = " + conv(ld) + ";"); if (acquire) line(guard + fence); continue; }
        if (in.ops[0].kind != PtxOperand::Vec || in.ops[0].regs.size() != n) fail("bad ld vector destination: " + text);
        line(guard + "{ " + vt + " __v = " + ld + ";");
        for (unsigned i = 0; i < n; i++) line("  " + outLV(in.ops[0].regs[i]) + " = " + conv("__v[" + std::to_string(i) + "]") + ";");
        if (acquire) line("  " + fence);
        line("}");
      } else {
        if (in.ops.size() != 2 || in.ops[0].kind != PtxOperand::Mem) fail("bad st operands: " + text);
        std::string ptr = globalPtr(in.ops[0].reg);
        if (in.ops[0].imm) ptr = "(" + ptr + " + " + std::to_string(in.ops[0].imm) + ")";
        if (release) line(guard + fence);
        auto st = [&](const std::string& v) {
          return scoped ? "__mvcc_st_coherent((device " + vt + "*)" + ptr + ", (" + vt + ")(" + v + "));" : "*(device " + vt + "*)" + ptr + " = " + v + ";";
        };
        if (n == 1) { line(guard + st("(" + et + ")(" + inExpr(reg(in.ops[1])) + ")")); continue; }
        if (in.ops[1].kind != PtxOperand::Vec || in.ops[1].regs.size() != n) fail("bad st vector source: " + text);
        std::string e = vt + "(";
        for (unsigned i = 0; i < n; i++) { if (i) e += ", "; e += "(" + et + ")(" + inExpr(in.ops[1].regs[i]) + ")"; }
        line(guard + st(e + ")"));
      }
      continue;
    }
    if (m == "lop3") {
      if (in.ops.size() != 5) fail("lop3 needs 5 operands: " + text);
      unsigned lut = (unsigned)in.ops[4].imm & 0xff;
      std::string v[3] = {inExpr(reg(in.ops[1])), inExpr(reg(in.ops[2])), inExpr(reg(in.ops[3]))};
      // Minimal sum of products over the truth table (bit (a<<2|b<<1|c) of lut): the prime implicants, then the
      // smallest cover (a dequantization's (x & m) | k is 2 operations, not the 5 minterms' 19)
      std::string e;
      if (lut == 0) e = "0u";
      else if (lut == 0xff) e = "0xffffffffu";
      else {
        struct Imp { int lit[3]; unsigned cover; };   // lit: 0 = complemented, 1 = plain, 2 = absent
        std::vector<Imp> primes;
        for (int p = 0; p < 27; p++) {
          Imp im; im.lit[0] = p % 3; im.lit[1] = (p / 3) % 3; im.lit[2] = p / 9; im.cover = 0;
          bool ok = true;
          for (unsigned i = 0; i < 8; i++) {
            bool inIm = true;
            for (int k = 0; k < 3; k++) if (im.lit[k] != 2 && (int)((i >> (2 - k)) & 1) != im.lit[k]) inIm = false;
            if (inIm) { im.cover |= 1u << i; if (!((lut >> i) & 1)) ok = false; }
          }
          if (ok) primes.push_back(im);
        }
        // prime: not strictly contained in another valid implicant
        std::vector<Imp> pr;
        for (auto& x : primes) { bool sub = false; for (auto& y : primes) if (y.cover != x.cover && (y.cover & x.cover) == x.cover) sub = true; if (!sub) pr.push_back(x); }
        unsigned best = ~0u, bestLits = ~0u;
        for (unsigned s = 1; s < (1u << pr.size()); s++) {
          unsigned cov = 0, lits = 0; for (size_t i = 0; i < pr.size(); i++) if ((s >> i) & 1) { cov |= pr[i].cover; for (int k = 0; k < 3; k++) lits += pr[i].lit[k] != 2; }
          if (cov != lut) continue;
          const unsigned n = __builtin_popcount(s);
          if (n < __builtin_popcount(best) || (n == __builtin_popcount(best) && lits < bestLits)) { best = s; bestLits = lits; }
        }
        for (size_t i = 0; i < pr.size(); i++) {
          if (!((best >> i) & 1)) continue;
          std::string term;
          for (int k = 0; k < 3; k++) { if (pr[i].lit[k] == 2) continue; if (!term.empty()) term += " & "; term += std::string(pr[i].lit[k] ? "" : "~") + "(" + v[k] + ")"; }
          if (term.empty()) term = "0xffffffffu";
          if (!e.empty()) e += " | ";
          e += "(" + term + ")";
        }
      }
      line(guard + outLV(reg(in.ops[0])) + " = " + e + ";");
      continue;
    }
    if ((m == "add" || m == "sub" || m == "mul" || m == "fma" || m == "max" || m == "min") && (in.hasMod("f16x2") || in.hasMod("bf16x2"))) {
      std::string vt = in.hasMod("f16x2") ? "half2" : "bfloat2";
      // bf16x2: Metal has no bfloat fma/fmax/fmin; compute in float (exact for bf16 products and sums that fit 24 bits,
      // single rounding back to bf16 like PTX .rn) and round once.
      bool viaFloat = !in.hasMod("f16x2");
      auto h = [&](unsigned k) { std::string v = "as_type<" + vt + ">(" + inExpr(k) + ")"; return viaFloat ? "float2(" + v + ")" : v; };
      std::string e;
      if (m == "fma") e = "fma(" + h(reg(in.ops[1])) + ", " + h(reg(in.ops[2])) + ", " + h(reg(in.ops[3])) + ")";
      else if (m == "max") e = "fmax(" + h(reg(in.ops[1])) + ", " + h(reg(in.ops[2])) + ")";
      else if (m == "min") e = "fmin(" + h(reg(in.ops[1])) + ", " + h(reg(in.ops[2])) + ")";
      else e = h(reg(in.ops[1])) + (m == "add" ? " + " : m == "sub" ? " - " : " * ") + h(reg(in.ops[2]));
      line(guard + outLV(reg(in.ops[0])) + " = as_type<uint>(" + vt + "(" + e + "));");
      continue;
    }
    if (m == "tanh" && in.hasMod("approx")) { line(guard + outLV(reg(in.ops[0])) + " = fast::tanh(" + inExpr(reg(in.ops[1])) + ");"); continue; }
    if (m == "ex2" && in.hasMod("approx")) { line(guard + outLV(reg(in.ops[0])) + " = fast::exp2(" + inExpr(reg(in.ops[1])) + ");"); continue; }
    if (m == "lg2" && in.hasMod("approx")) { line(guard + outLV(reg(in.ops[0])) + " = fast::log2(" + inExpr(reg(in.ops[1])) + ");"); continue; }
    if (m == "rcp" && in.hasMod("approx")) { line(guard + outLV(reg(in.ops[0])) + " = 1.0f / (" + inExpr(reg(in.ops[1])) + ");"); continue; }
    if (m == "rsqrt" && in.hasMod("approx")) { line(guard + outLV(reg(in.ops[0])) + " = rsqrt(" + inExpr(reg(in.ops[1])) + ");"); continue; }
    if (m == "mov") {
      if (in.ops.size() != 2) fail("bad mov: " + text);
      std::string src;
      if (in.ops[1].kind == PtxOperand::Reg) src = inExpr(in.ops[1].reg);
      else if (in.ops[1].kind == PtxOperand::Imm) src = std::to_string(in.ops[1].imm) + "u";
      else if (in.ops[1].kind == PtxOperand::Named) {
        if (in.ops[1].name == "laneid") src = "__lane";
        else if (in.ops[1].name == "warpid") src = "__sgid";
        else if (namedLV.count(in.ops[1].name)) src = namedLV.at(in.ops[1].name);
        else src = "0u";  // %globaltimer / %clock / %clock64: Metal has no CUDA special timer
      } else fail("bad mov source: " + text);
      line(guard + outLV(reg(in.ops[0])) + " = " + src + ";");
      continue;
    }
    if (m == "cvt") {
      // cvt.rn.bf16x2.f32 d, a, b  (d.hi = a, d.lo = b) ; cvt.rn.f16x2.f32 ; cvt.rn.f16.f32 ; cvt.f32.f16 ; cvt.rn.bf16.f32 ; cvt.f32.bf16
      if (in.hasMod("bf16x2") && in.hasMod("f32")) { line(guard + outLV(reg(in.ops[0])) + " = as_type<uint>(bfloat2((bfloat)" + inExpr(reg(in.ops[2])) + ", (bfloat)" + inExpr(reg(in.ops[1])) + "));"); continue; }
      if (in.hasMod("f16x2") && in.hasMod("f32")) { line(guard + outLV(reg(in.ops[0])) + " = as_type<uint>(half2((half)" + inExpr(reg(in.ops[2])) + ", (half)" + inExpr(reg(in.ops[1])) + "));"); continue; }
      if (in.mods.size() >= 2 && in.mods[in.mods.size() - 2] == "f16" && in.mods.back() == "f32") { line(guard + outLV(reg(in.ops[0])) + " = as_type<ushort>((half)" + inExpr(reg(in.ops[1])) + ");"); continue; }
      if (in.mods.size() >= 2 && in.mods[in.mods.size() - 2] == "f32" && in.mods.back() == "f16") { line(guard + outLV(reg(in.ops[0])) + " = (float)as_type<half>((ushort)" + inExpr(reg(in.ops[1])) + ");"); continue; }
      if (in.mods.size() >= 2 && in.mods[in.mods.size() - 2] == "bf16" && in.mods.back() == "f32") { line(guard + outLV(reg(in.ops[0])) + " = as_type<ushort>((bfloat)" + inExpr(reg(in.ops[1])) + ");"); continue; }
      if (in.mods.size() >= 2 && in.mods[in.mods.size() - 2] == "f32" && in.mods.back() == "bf16") { line(guard + outLV(reg(in.ops[0])) + " = (float)as_type<bfloat>((ushort)" + inExpr(reg(in.ops[1])) + ");"); continue; }
      if (in.hasMod("u16") && in.hasMod("u32") && in.ops.size() == 2) {
        line(guard + opW(in.ops[0]) + " = (uint)(ushort)(" + opR(in.ops[1]) + ");");
        continue;
      }
      // Two packed e4m3 bytes -> packed f16x2 (cvt.rn.f16x2.e4m3x2).
      if (in.hasMod("f16x2") && in.hasMod("e4m3x2") && in.ops.size() == 2) {
        usedHelpers.insert("mma");
        std::string a = opR(in.ops[1]);
        line(guard + opW(in.ops[0]) + " = mvcc_detail::pack_half2(mvcc_detail::e4m3_to_half(" + a + "), mvcc_detail::e4m3_to_half((" + a + ") >> 8));");
        continue;
      }
      fail("unsupported cvt: " + text);
    }
    if (m == "prmt") { usedHelpers.insert("prmt"); line(guard + outLV(reg(in.ops[0])) + " = __mvcc_prmt(" + inExpr(reg(in.ops[1])) + ", " + inExpr(reg(in.ops[2])) + ", " + inExpr(reg(in.ops[3])) + ");"); continue; }
    if (m == "shf") {  // funnel shift of {b:a} by c: .l keeps the high word, .r the low word; .clamp saturates c at 32
      usedHelpers.insert("bits");
      const bool left = in.hasMod("l"), clamp = in.hasMod("clamp");
      const std::string a = inExpr(reg(in.ops[1])), b = inExpr(reg(in.ops[2])), c = inExpr(reg(in.ops[3]));
      std::string e = std::string(left ? "__mvcc_fshl" : "__mvcc_fshr") + "((uint)(" + b + "), (uint)(" + a + "), (uint)(" + c + "))";
      if (clamp) e = "(((uint)(" + c + ") >= 32u) ? (uint)(" + (left ? a : b) + ") : " + e + ")";
      line(guard + outLV(reg(in.ops[0])) + " = " + e + ";");
      continue;
    }
    if (m == "bar" || m == "barrier") { line("threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);"); continue; }
    if (m == "membar" || m == "fence") { line("atomic_thread_fence(mem_flags::mem_device | mem_flags::mem_threadgroup, memory_order_seq_cst, thread_scope_device);"); continue; }
    if (m == "prefetch" || m == "prefetchu") {
      // PTX cache hint. Metal has no L2 prefetch; a discarded volatile device load touches the line
      // so the next consumer can start warm.
      if (!in.ops.empty()) {
        const PtxOperand& a = in.ops[0];
        int rk = (a.kind == PtxOperand::Mem || a.kind == PtxOperand::Reg) ? a.reg : -1;
        if (rk >= 0) {
          std::string ptr = globalPtr((unsigned)rk);
          if (a.imm) ptr = "(" + ptr + " + " + std::to_string(a.imm) + ")";
          line(guard + "(void)*((device volatile uchar*)" + ptr + ");");
        }
      }
      continue;
    }
    if (m == "nanosleep") { continue; }
    if (m == "trap") { line("__MVCC_RETURN"); continue; }

    if (m == "ldmatrix") {
      // ldmatrix.sync.aligned.m8n8.{x1,x2,x4}{.trans}.shared.b16 {d...}, [addr]
      unsigned x = in.hasMod("x4") ? 4 : in.hasMod("x2") ? 2 : 1;
      bool trans = in.hasMod("trans");
      if (in.ops.size() != 2) fail("bad ldmatrix operands: " + text);
      std::vector<int> dregs = in.ops[0].kind == PtxOperand::Vec ? in.ops[0].regs : std::vector<int>{in.ops[0].reg};
      if (dregs.size() != x) fail("ldmatrix destination count mismatch: " + text);
      std::string addr = sharedPtr(in.ops[1].reg);
      if (in.ops[1].imm) addr = "(" + addr + " + " + std::to_string(in.ops[1].imm) + ")";
      usedHelpers.insert("ldmatrix");
      line("{ ulong __a = (ulong)" + addr + ";");
      for (unsigned i = 0; i < x; i++)
        line("  " + outLV(dregs[i]) + " = __mvcc_ldmatrix_" + (trans ? "trans" : "row") + "(__a, " + std::to_string(i) + "u, __lane);");
      line("}");
      continue;
    }
    if (m == "mma") {
      // mma.sync.aligned.mMnNkK.row.col.DT.AT.BT.CT {d}, {a}, {b}, {c}
      std::string shape, dt, at, bt, ct;
      for (auto& md : in.mods) if (md.size() > 3 && md[0] == 'm' && isdigit((unsigned char)md[1])) shape = md;
      // types are the last four modifiers
      if (in.mods.size() < 4) fail("bad mma modifiers: " + text);
      dt = in.mods[in.mods.size() - 4]; at = in.mods[in.mods.size() - 3]; bt = in.mods[in.mods.size() - 2]; ct = in.mods[in.mods.size() - 1];
      if (in.ops.size() != 4) fail("mma needs 4 operands: " + text);
      auto vec = [&](const PtxOperand& o) { return o.kind == PtxOperand::Vec ? o.regs : std::vector<int>{o.reg}; };
      auto D = vec(in.ops[0]), A = vec(in.ops[1]), B = vec(in.ops[2]), C = vec(in.ops[3]);
      std::string helper = "__mvcc_mma_" + shape + "_" + dt + "_" + at + "_" + bt + "_" + ct;
      static const std::set<std::string> supported = {
        "__mvcc_mma_m16n8k16_f32_f16_f16_f32", "__mvcc_mma_m16n8k16_f32_bf16_bf16_f32", "__mvcc_mma_m16n8k16_f16_f16_f16_f16",
        "__mvcc_mma_m16n8k8_f32_f16_f16_f32", "__mvcc_mma_m16n8k8_f32_bf16_bf16_f32",
        "__mvcc_mma_m16n8k32_s32_s8_s8_s32", "__mvcc_mma_m16n8k32_s32_u8_u8_s32", "__mvcc_mma_m16n8k16_s32_s8_s8_s32",
        "__mvcc_mma_m16n8k32_f32_e4m3_e4m3_f32", "__mvcc_mma_m16n8k32_f32_e5m2_e5m2_f32",
      };
      if (!supported.count(helper)) fail("unsupported mma variant " + in.full());
      usedHelpers.insert("mma");
      // Operands travel as raw 32-bit lanes (uintN); float registers are bit-cast in and out.
      auto inIsF = [&](unsigned k) { return k < nOut ? (tiedArgForOutput[k] >= 0 && inArg(k)->getType()->isFloatTy()) : inArg(k)->getType()->isFloatTy(); };
      auto outIsF = [&](unsigned k) { Type* T = CI.getType(); if (T->isStructTy()) T = T->getStructElementType(k); return T->isFloatTy(); };
      auto pack = [&](const std::vector<int>& r, bool out, const char* nm_) {
        std::string s = "  " + std::string(out ? "" : "const ") + "uint" + (r.size() > 1 ? std::to_string(r.size()) : "") + " " + nm_;
        if (out) return s + ";\n";
        s += r.size() > 1 ? " = uint" + std::to_string(r.size()) + "(" : " = (";
        for (size_t i = 0; i < r.size(); i++) { if (i) s += ", "; s += inIsF(r[i]) ? "as_type<uint>(" + inExpr(r[i]) + ")" : inExpr(r[i]); }
        return s + ");\n";
      };
      std::string blk = "{\n" + pack(A, false, "__a") + pack(B, false, "__b") + pack(C, false, "__c") + pack(D, true, "__d");
      blk += "  " + helper + "(__d, __a, __b, __c, __lane);\n";
      for (size_t i = 0; i < D.size(); i++) {
        std::string e = "__d" + (D.size() > 1 ? "[" + std::to_string(i) + "]" : "");
        blk += "  " + outLV(D[i]) + " = " + (outIsF(D[i]) ? "as_type<float>(" + e + ")" : e) + ";\n";
      }
      blk += "}";
      // emit as lines
      std::stringstream ss(blk); std::string l;
      while (std::getline(ss, l)) line(guard + l);
      continue;
    }
    fail("unsupported inline PTX instruction '" + in.full() + "' in: " + text);
  }
}

// ---------------------------------------------------------------------------

// Module-scope variables that need a runtime buffer: __device__ variables (mutable, visible to the host through
// cudaMemcpyToSymbol) and any global that is reached through a pointer stored in another global's initializer
// (the stored pointer must be a real device address, so the target cannot be program-scope constant data).
void Emitter::layoutGlobals() {
  std::set<const GlobalVariable*> buffered;
  std::function<void(const Constant*, bool)> refs = [&](const Constant* C, bool inData) {
    if (auto* GV = dyn_cast<GlobalVariable>(C)) { if (inData && (GV->getAddressSpace() == 4 || GV->getAddressSpace() == 1)) buffered.insert(GV); return; }
    for (const Value* op : C->operands()) if (auto* OC = dyn_cast<Constant>(op)) refs(OC, inData);
  };
  for (GlobalVariable& GV : M.globals()) {
    if (GV.getName().starts_with("llvm.")) continue;
    // __device__ variables, and __constant__ variables that are not const in the source (the host may write them
    // with cudaMemcpyToSymbol), are buffered; `const` tables stay program-scope constant data unless pointed at
    if ((GV.getAddressSpace() == 1 || (GV.getAddressSpace() == 4 && !GV.isConstant())) && GV.hasInitializer()) buffered.insert(&GV);
    if (GV.hasInitializer()) {
      // a pointer to another global inside this initializer buffers both
      bool hasRef = false;
      std::function<void(const Constant*)> scan = [&](const Constant* C) {
        if (isa<GlobalVariable>(C)) { hasRef = true; return; }
        for (const Value* op : C->operands()) if (auto* OC = dyn_cast<Constant>(op)) scan(OC);
      };
      scan(GV.getInitializer());
      if (hasRef) { buffered.insert(&GV); refs(GV.getInitializer(), true); }
    }
  }
  if (buffered.empty()) return;
  std::vector<const GlobalVariable*> order(buffered.begin(), buffered.end());
  const DataLayout& DL = M.getDataLayout();
  std::stable_sort(order.begin(), order.end(), [&](const GlobalVariable* a, const GlobalVariable* b) {
    unsigned aa = std::max<unsigned>(a->getAlign().valueOrOne().value(), DL.getABITypeAlign(a->getValueType()).value());
    unsigned ab = std::max<unsigned>(b->getAlign().valueOrOne().value(), DL.getABITypeAlign(b->getValueType()).value());
    return aa > ab || (aa == ab && a->getName() < b->getName());
  });
  uint64_t off = 0;
  for (const GlobalVariable* GV : order) {
    unsigned al = std::max<unsigned>(GV->getAlign().valueOrOne().value(), DL.getABITypeAlign(GV->getValueType()).value());
    off = (off + al - 1) / al * al;
    globalOffsets[GV] = (unsigned)off;
    res.globals.symbols.push_back({GV->getName().str(), (unsigned)off, (unsigned)DL.getTypeAllocSize(GV->getValueType())});
    off += DL.getTypeAllocSize(GV->getValueType());
  }
  res.globals.size = (unsigned)((off + 15) / 16 * 16);
  res.globals.init.assign(res.globals.size, 0);
  for (const GlobalVariable* GV : order)
    if (GV->hasInitializer()) serializeInit(const_cast<GlobalVariable*>(GV)->getInitializer(), globalOffsets[GV]);
    else fail("module-scope variable '" + GV->getName().str() + "' has no initializer (external __device__ variables are not supported)");
}

void Emitter::serializeInit(Constant* C, uint64_t off) {
  const DataLayout& DL = M.getDataLayout();
  auto& out = res.globals.init;
  auto put = [&](uint64_t at, uint64_t v, unsigned bytes) { for (unsigned i = 0; i < bytes; i++) out[at + i] = (uint8_t)(v >> (8 * i)); };
  if (isa<ConstantAggregateZero>(C) || isa<UndefValue>(C) || isa<ConstantPointerNull>(C)) return;
  if (auto* CI = dyn_cast<ConstantInt>(C)) {
    if (CI->getBitWidth() > 64) fail("global initializer with an integer wider than 64 bits");
    put(off, CI->getZExtValue(), (unsigned)DL.getTypeStoreSize(C->getType())); return;
  }
  if (auto* CF = dyn_cast<ConstantFP>(C)) { put(off, CF->getValueAPF().bitcastToAPInt().getZExtValue(), (unsigned)DL.getTypeStoreSize(C->getType())); return; }
  if (auto* CDS = dyn_cast<ConstantDataSequential>(C)) {
    StringRef raw = CDS->getRawDataValues();
    unsigned esz = (unsigned)DL.getTypeAllocSize(CDS->getElementType()), essz = (unsigned)DL.getTypeStoreSize(CDS->getElementType());
    for (unsigned i = 0; i < CDS->getNumElements(); i++) memcpy(&out[off + (uint64_t)i * esz], raw.data() + (uint64_t)i * essz, essz);
    return;
  }
  if (auto* CA = dyn_cast<ConstantArray>(C)) {
    unsigned esz = (unsigned)DL.getTypeAllocSize(CA->getType()->getElementType());
    for (unsigned i = 0; i < CA->getNumOperands(); i++) serializeInit(CA->getOperand(i), off + (uint64_t)i * esz);
    return;
  }
  if (auto* CV = dyn_cast<ConstantVector>(C)) {
    unsigned esz = (unsigned)DL.getTypeAllocSize(CV->getType()->getElementType());
    for (unsigned i = 0; i < CV->getNumOperands(); i++) serializeInit(CV->getOperand(i), off + (uint64_t)i * esz);
    return;
  }
  if (auto* CS = dyn_cast<ConstantStruct>(C)) {
    const StructLayout* SL = DL.getStructLayout(CS->getType());
    for (unsigned i = 0; i < CS->getNumOperands(); i++) serializeInit(CS->getOperand(i), off + SL->getElementOffset(i));
    return;
  }
  // pointers: a global (plus a constant offset) becomes a relocation the runtime patches with the buffer address
  if (auto* GV = dyn_cast<GlobalVariable>(C)) {
    auto it = globalOffsets.find(GV);
    if (it == globalOffsets.end()) fail("initializer points at global '" + GV->getName().str() + "' which is not in the globals buffer");
    res.globals.relocs.push_back({(unsigned)off, it->second});
    return;
  }
  if (auto* CE = dyn_cast<ConstantExpr>(C)) {
    if (auto* GEP = dyn_cast<GEPOperator>(CE)) {
      APInt ap(64, 0);
      if (!GEP->accumulateConstantOffset(DL, ap)) fail("initializer with a non-constant address computation");
      auto* Base = dyn_cast<GlobalVariable>(GEP->getPointerOperand()->stripPointerCasts());
      if (!Base) fail("initializer address not based on a global");
      auto it = globalOffsets.find(Base);
      if (it == globalOffsets.end()) fail("initializer points at global '" + Base->getName().str() + "' which is not in the globals buffer");
      res.globals.relocs.push_back({(unsigned)off, it->second + (unsigned)ap.getZExtValue()});
      return;
    }
    switch (CE->getOpcode()) {
      case Instruction::AddrSpaceCast: case Instruction::BitCast: case Instruction::PtrToInt: case Instruction::IntToPtr:
        serializeInit(cast<Constant>(CE->getOperand(0)), off); return;
      default: break;
    }
  }
  if (isa<Function>(C)) fail("global initializer holds a function pointer (device function pointers are not supported)");
  std::string str; raw_string_ostream os(str); C->print(os);
  fail("unsupported global initializer: " + str);
}

void Emitter::run() {
  layoutGlobals();
  // constant-space globals with initializers
  for (GlobalVariable& GV : M.globals()) {
    if (globalOffsets.count(&GV)) continue;  // materialized in the runtime's globals buffer
    // AS4 __constant__ tables, and generic-space constant tables the optimizer itself creates (SimplifyCFG's
    // `switch.table.*` lookup tables), are program-scope constant data
    if ((GV.getAddressSpace() == 4 || (GV.getAddressSpace() == 0 && GV.isConstant())) && GV.hasInitializer()) {
      std::string nm = "g_" + GV.getName().str();
      for (char& c : nm) if (!isalnum((unsigned char)c)) c = '_';
      globalNames[&GV] = nm;
      // program-scope constants never reference each other: any global reached from an initializer is buffered
      globalsOut += "constant " + ty(GV.getValueType()) + " " + nm + " = " + constant(GV.getInitializer()) + ";\n";
    } else if (GV.getAddressSpace() == 1 && !GV.getName().starts_with("llvm.")) {
      fail("__device__ global variable '" + GV.getName().str() + "' is not supported yet (module-scope device globals)");
    }
  }
  for (Function& Fn : M) {
    if (Fn.isDeclaration() || !isKernel(Fn)) continue;
    try { emitKernel(Fn); }
    catch (EmitError& e) { throw EmitError{"in kernel " + llvm::demangle(Fn.getName().str()) + ": " + e.msg}; }
  }
  if (recovery && !recovery->descriptors.empty()) emitVerifyKernels();
  std::string out;
  out += "// Generated by mvcc mvcc-ir2msl. Do not edit.\n";
  out += "#include <metal_stdlib>\n#include <metal_simdgroup_matrix>\nusing namespace metal;\n";
  out += "#define MVCC_PRELUDE_BEGIN\n";  // the driver splices the prelude in here
  for (auto& h : usedHelpers) out += "#define MVCC_NEED_" + h + " 1\n";
  out += "#define MVCC_PRELUDE_END\n\n";
  out += typesOut + "\n" + globalsOut + "\n" + kernelsOut;
  res.msl = out;
}

} // namespace

bool emitModule(Module& M, const EmitOptions& opts, EmitResult& out) {
  try {
    TensorRecoveryResult tr;
    normalizeModule(M, opts, tr);
    for (auto& w : tr.warnings) out.warnings.push_back("tensor recovery: " + w);
    const bool recoveryLogVerbose = opts.tensorDiag || opts.verbose;
    for (auto& n : tr.notes) {
      const std::string prefixed = "tensor recovery: " + n;
      if (mvcc::shouldLogRecoveryNote(prefixed, recoveryLogVerbose)) out.notes.push_back(prefixed);
    }
    if (tr.kernelsWithMma && recoveryLogVerbose)
      out.notes.push_back("tensor recovery: " + std::to_string(tr.mmaKernelsRecovered) + " of "
                          + std::to_string(tr.kernelsWithMma) + " mma.sync kernel(s) recovered onto TensorOps");
    for (auto& c : tr.collectives) out.collectives.push_back({c.first, c.second, "copy"});
    std::string verr; raw_string_ostream vos(verr);
    if (verifyModule(M, &vos)) { out.error = "module failed verification after normalization: " + verr; return false; }
    Emitter E(M, opts, out);
    E.setRecovery(tr);
    E.run();
    if (!out.collectives.empty()) {
      std::string hdr = "// mvcc collective sequence:";
      for (auto& c : out.collectives)
        hdr += " " + c.kind + ":world=" + std::to_string(c.world) + ":" + (c.dtype.empty() ? "copy" : c.dtype);
      out.msl = hdr + "\n" + out.msl;
    }
    return true;
  } catch (EmitError& e) {
    out.error = e.msg;
    return false;
  }
}

} // namespace mvcc
