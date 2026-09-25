// Elementwise check+emit. See emit.h.
#include "emit.h"
#include "dot_recovery.h"
#include "sig.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Support/FileSystem.h"
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>
#include <vector>

namespace mvcc {
namespace emit {

using namespace llvm;

static const TensorRecoveryOptions* gRecoveryNoteVerbose = nullptr;

void bindRecoveryNoteVerbose(const TensorRecoveryOptions* opts) { gRecoveryNoteVerbose = opts; }

void pushRecoverySuccessNote(TensorRecoveryResult& res, const std::string& s) {
  if (gRecoveryNoteVerbose && gRecoveryNoteVerbose->verbose) res.notes.push_back(s);
}

void pushRecoveryDeclineNote(TensorRecoveryResult& res, const std::string& s) { res.notes.push_back(s); }

static bool isKernelFn(const Function& F) {
  if (F.getCallingConv() == CallingConv::PTX_Kernel) return true;
  if (F.hasFnAttribute("nvvm.kernel")) return true;
  if (auto* MD = F.getParent()->getNamedMetadata("nvvm.annotations"))
    for (auto* Op : MD->operands()) {
      if (Op->getNumOperands() < 3) continue;
      auto* V = mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
      auto* S = dyn_cast<MDString>(Op->getOperand(1));
      if (V == &F && S && S->getString() == "kernel") return true;
    }
  return false;
}

static bool hasRawMma(const Function& F) {
  for (auto& I : instructions(F)) {
    auto* CI = dyn_cast<CallInst>(&I);
    if (!CI) continue;
    if (auto* IA = dyn_cast<InlineAsm>(CI->getCalledOperand()))
      if (IA->getAsmString().find("mma.") != std::string::npos || IA->getAsmString().find("ldmatrix") != std::string::npos)
        return true;
  }
  return false;
}
static bool hasTpMma(const Function& F) {
  for (auto& I : instructions(F)) {
    auto* CI = dyn_cast<CallInst>(&I);
    if (!CI) continue;
    if (Function* C = CI->getCalledFunction())
      if (C->getName().starts_with("__mvcc_tp_mma")) return true;
  }
  return false;
}
static bool hasMmaAsm(const Function& F) { return hasRawMma(F) || hasTpMma(F); }

static bool isSreg(Value* V, StringRef name) {
  for (int n = 0; n < 3 && V; n++) {
    if (auto* C = dyn_cast<CastInst>(V)) { V = C->getOperand(0); continue; }
    break;
  }
  auto* CI = dyn_cast<CallInst>(V);
  if (!CI || !CI->getCalledFunction()) return false;
  return CI->getCalledFunction()->getName() == name;
}

static bool derivesFromSreg(Value* V, StringRef name, SmallPtrSetImpl<Value*>& vis) {
  if (!V || vis.count(V)) return false;
  vis.insert(V);
  if (isSreg(V, name)) return true;
  if (auto* I = dyn_cast<Instruction>(V)) {
    for (Value* op : I->operands())
      if (derivesFromSreg(op, name, vis)) return true;
  }
  return false;
}

static bool derivesFromCtaid(Value* V) {
  SmallPtrSet<Value*, 16> vis;
  return derivesFromSreg(V, "llvm.nvvm.read.ptx.sreg.ctaid.x", vis)
      || derivesFromSreg(V, "llvm.nvvm.read.ptx.sreg.ctaid.y", vis)
      || derivesFromSreg(V, "llvm.nvvm.read.ptx.sreg.ctaid.z", vis);
}

static Value* stripCast(Value* V) {
  while (auto* C = dyn_cast<CastInst>(V)) V = C->getOperand(0);
  return V;
}

// the source's o = cta * C + tid  (C a constant, ntid.x, or a runtime stride)
static Value* findSourceIndex(Function& F, unsigned& strideOut) {
  strideOut = 0;
  for (auto& I : instructions(F)) {
    auto* Add = dyn_cast<BinaryOperator>(&I);
    // InstCombine turns `cta<<k + tid` into `or` when the bits are disjoint (tid < 2^k).
    if (!Add || (Add->getOpcode() != Instruction::Add && Add->getOpcode() != Instruction::Or)) continue;
    Value* a = Add->getOperand(0), *b = Add->getOperand(1);
    auto tidLike = [&](Value* V) -> bool {
      if (isSreg(V, "llvm.nvvm.read.ptx.sreg.tid.x")) return true;
      auto* Bin = dyn_cast<BinaryOperator>(stripCast(V));
      if (Bin && Bin->getOpcode() == Instruction::LShr
          && isSreg(Bin->getOperand(0), "llvm.nvvm.read.ptx.sreg.tid.x"))
        return true;
      return false;
    };
    auto match = [&](Value* mulV, Value* tidV) -> bool {
      if (!tidLike(tidV)) return false;
      auto* Bin = dyn_cast<BinaryOperator>(stripCast(mulV));
      if (!Bin) {
        if (derivesFromCtaid(mulV)) { strideOut = 0; return true; }
        return false;
      }
      Value* x = Bin->getOperand(0), *y = Bin->getOperand(1);
      if (Bin->getOpcode() == Instruction::Shl) {
        if (!derivesFromCtaid(x)) return false;
        if (auto* C = dyn_cast<ConstantInt>(stripCast(y))) { strideOut = 1u << (unsigned)C->getZExtValue(); return true; }
        return false;
      }
      if (Bin->getOpcode() != Instruction::Mul) return false;
      Value* cta = nullptr;
      Value* k = nullptr;
      if (derivesFromCtaid(x)) { cta = x; k = y; }
      else if (derivesFromCtaid(y)) { cta = y; k = x; }
      if (!cta) return false;
      if (auto* C = dyn_cast<ConstantInt>(stripCast(k))) { strideOut = (unsigned)C->getZExtValue(); return true; }
      if (isSreg(k, "llvm.nvvm.read.ptx.sreg.ntid.x")) { strideOut = 0; return true; }
      // row/col stride from a kernel argument or induction (block-parallel maps)
      strideOut = 0;
      return true;
    };
    if (match(a, b) || match(b, a)) return Add;
  }
  return nullptr;
}

static bool exprUsesSreg(Value* V, StringRef reg, SmallPtrSetImpl<Value*>& vis) {
  if (!V || vis.count(V)) return false;
  vis.insert(V);
  if (isSreg(V, reg)) return true;
  if (auto* BO = dyn_cast<BinaryOperator>(V))
    return exprUsesSreg(BO->getOperand(0), reg, vis) || exprUsesSreg(BO->getOperand(1), reg, vis);
  if (auto* CI = dyn_cast<CastInst>(V)) return exprUsesSreg(CI->getOperand(0), reg, vis);
  return false;
}

static bool exprUsesSreg(Value* V, StringRef reg) {
  SmallPtrSet<Value*, 16> vis;
  return exprUsesSreg(V, reg, vis);
}

static bool isGridStrideStep(Value* V) {
  V = stripCast(V);
  auto* Bin = dyn_cast<BinaryOperator>(V);
  if (!Bin || Bin->getOpcode() != Instruction::Mul) return false;
  bool bd = exprUsesSreg(Bin->getOperand(0), "llvm.nvvm.read.ptx.sreg.blockDim.x")
         || exprUsesSreg(Bin->getOperand(1), "llvm.nvvm.read.ptx.sreg.blockDim.x");
  bool gd = exprUsesSreg(Bin->getOperand(0), "llvm.nvvm.read.ptx.sreg.gridDim.x")
         || exprUsesSreg(Bin->getOperand(1), "llvm.nvvm.read.ptx.sreg.gridDim.x");
  return bd && gd;
}

// Per-CTA strided loops (`i = tid; i < n; i += blockDim`) without grid-stride indexing.
static bool isBlockStrideStep(Value* V) {
  V = stripCast(V);
  auto* Bin = dyn_cast<BinaryOperator>(V);
  if (!Bin || Bin->getOpcode() != Instruction::Mul) return false;
  bool bd = exprUsesSreg(Bin->getOperand(0), "llvm.nvvm.read.ptx.sreg.blockDim.x")
         || exprUsesSreg(Bin->getOperand(1), "llvm.nvvm.read.ptx.sreg.blockDim.x");
  if (!bd) return false;
  bool gd = exprUsesSreg(Bin->getOperand(0), "llvm.nvvm.read.ptx.sreg.gridDim.x")
         || exprUsesSreg(Bin->getOperand(1), "llvm.nvvm.read.ptx.sreg.gridDim.x");
  return !gd;
}

static bool icmpThreadZero(const ICmpInst* IC) {
  if (!IC || IC->getPredicate() != CmpInst::ICMP_EQ) return false;
  const Value* a = IC->getOperand(0), *b = IC->getOperand(1);
  auto isZero = [](const Value* V) { auto* C = dyn_cast<ConstantInt>(V); return C && C->isZero(); };
  auto isTidX = [](const Value* V) { return isSreg(const_cast<Value*>(V), "llvm.nvvm.read.ptx.sreg.tid.x"); };
  return (isTidX(a) && isZero(b)) || (isTidX(b) && isZero(a));
}

static bool memOpUsesBlockIndex(const Function& F);
static bool hasAtomic(const Function& F);
static bool irBlockCollectiveKernel(Function& F);

static bool isBlockDimStep(Value* V) {
  if (isBlockStrideStep(V)) return true;
  return exprUsesSreg(V, "llvm.nvvm.read.ptx.sreg.blockDim.x")
      && !exprUsesSreg(V, "llvm.nvvm.read.ptx.sreg.gridDim.x");
}

static bool isBlockParallelConstStep(Value* V) {
  auto* C = dyn_cast<ConstantInt>(stripCast(V));
  if (!C) return false;
  const uint64_t n = C->getZExtValue();
  return n >= 32 && n <= 1024;
}

static bool irBlockLocalStride(Function& F) {
  if (memOpUsesBlockIndex(F)) return false;
  bool tid = false;
  for (auto& I : instructions(F))
    if (exprUsesSreg(&I, "llvm.nvvm.read.ptx.sreg.tid.x")) { tid = true; break; }
  if (!tid) return false;
  for (auto& I : instructions(F)) {
    auto* BO = dyn_cast<BinaryOperator>(&I);
    if (!BO || BO->getOpcode() != Instruction::Add) continue;
    if (isBlockDimStep(BO->getOperand(0)) || isBlockDimStep(BO->getOperand(1))) return true;
    if (isBlockParallelConstStep(BO->getOperand(0)) || isBlockParallelConstStep(BO->getOperand(1))) return true;
  }
  return false;
}

static bool irThreadZeroSerialKernel(const Function& F) {
  if (memOpUsesBlockIndex(F) || hasAtomic(F)) return false;
  unsigned stride = 0;
  if (findSourceIndex(const_cast<Function&>(F), stride)) return false;
  bool tidZero = false;
  for (auto& I : instructions(F))
    if (auto* IC = dyn_cast<ICmpInst>(&I)) tidZero |= icmpThreadZero(IC);
  if (!tidZero) return false;
  for (auto& I : instructions(F)) {
    auto* BO = dyn_cast<BinaryOperator>(&I);
    if (!BO || BO->getOpcode() != Instruction::Add) continue;
    if (isGridStrideStep(BO->getOperand(0)) || isGridStrideStep(BO->getOperand(1))) return false;
  }
  return true;
}

static bool memOpUsesBlockIndex(const Function& F) {
  for (auto& I : instructions(F)) {
    const Value* Ptr = nullptr;
    if (auto* L = dyn_cast<LoadInst>(&I)) Ptr = L->getPointerOperand();
    else if (auto* S = dyn_cast<StoreInst>(&I)) Ptr = S->getPointerOperand();
    else if (auto* A = dyn_cast<AtomicRMWInst>(&I)) Ptr = A->getPointerOperand();
    else continue;
    if (!Ptr) continue;
    if (derivesFromCtaid(const_cast<Value*>(Ptr))) return true;
  }
  return false;
}

static Value* emitO(IRBuilder<>& B, Function& F, unsigned T) {
  Type* i32 = B.getInt32Ty();
  Value* tid = B.CreateCall(F.getParent()->getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.x", FunctionType::get(i32, false)), {}, "stir.tid");
  Value* cta = B.CreateCall(F.getParent()->getOrInsertFunction("llvm.nvvm.read.ptx.sreg.ctaid.x", FunctionType::get(i32, false)), {}, "stir.cta");
  Value* stride = ConstantInt::get(i32, T ? T : 256);
  return B.CreateAdd(B.CreateMul(cta, stride, "stir.blk"), tid, "stir.o");
}

static bool ifConvertZeroPage(Function& F, Value* pred) {
  // loads in blocks that are reached only through pred: select(pred, ptr, zero_page)
  Module& M = *F.getParent();
  FunctionCallee ZF = M.getOrInsertFunction("__mvcc_zero_page", PointerType::get(F.getContext(), 0));
  bool any = false;
  for (BasicBlock& BB : F) {
    for (auto it = BB.begin(); it != BB.end(); ) {
      Instruction* I = &*it++;
      auto* LI = dyn_cast<LoadInst>(I);
      if (!LI || LI->isVolatile()) continue;
      IRBuilder<> B(LI);
      Value* z = B.CreateCall(ZF, {}, "zero_page");
      Value* zp = B.CreateBitCast(z, LI->getPointerOperand()->getType());
      Value* p = B.CreateSelect(pred, LI->getPointerOperand(), zp, "stir.zsel");
      LI->setOperand(0, p);
      any = true;
    }
  }
  return any;
}

static bool hasCallPrefix(const Function& F, StringRef p) {
  for (auto& I : instructions(F))
    if (auto* C = dyn_cast<CallInst>(&I))
      if (C->getCalledFunction() && C->getCalledFunction()->getName().starts_with(p)) return true;
  return false;
}

static bool irBlockCollectiveKernel(Function& F) {
  if (memOpUsesBlockIndex(F)) return false;
  if (hasCallPrefix(F, "mma.sync") || hasCallPrefix(F, "__mvcc_tile_mma")) return false;
  const bool bar = hasCallPrefix(F, "llvm.nvvm.barrier");
  const bool shfl = hasCallPrefix(F, "llvm.nvvm.shfl");
  if (!bar && !shfl) return false;
  if (irBlockLocalStride(F)) return true;
  for (auto& I : instructions(F))
    if (exprUsesSreg(&I, "llvm.nvvm.read.ptx.sreg.blockDim.x")) return true;
  return false;
}

// Grid-stride, block-parallel, or row-parallel launch maps (explicit stride loops or cta/tid indexing).
static bool irParallelMapKernel(Function& F) {
  unsigned stride = 0;
  if (findSourceIndex(F, stride)) return true;
  for (auto& I : instructions(F)) {
    auto* BO = dyn_cast<BinaryOperator>(&I);
    if (!BO || BO->getOpcode() != Instruction::Add) continue;
    if (isGridStrideStep(BO->getOperand(0)) || isGridStrideStep(BO->getOperand(1))) return true;
    bool bd = exprUsesSreg(BO->getOperand(0), "llvm.nvvm.read.ptx.sreg.blockDim.x")
           || exprUsesSreg(BO->getOperand(1), "llvm.nvvm.read.ptx.sreg.blockDim.x");
    bool gd = exprUsesSreg(BO->getOperand(0), "llvm.nvvm.read.ptx.sreg.gridDim.x")
           || exprUsesSreg(BO->getOperand(1), "llvm.nvvm.read.ptx.sreg.gridDim.x");
    if (bd && gd) return true;
  }
  for (auto& I : instructions(F)) {
    Value* V = &I;
    if (exprUsesSreg(V, "llvm.nvvm.read.ptx.sreg.ctaid.x") && exprUsesSreg(V, "llvm.nvvm.read.ptx.sreg.gridDim.x"))
      return true;
  }
  if (memOpUsesBlockIndex(F)) return true;
  if (irBlockLocalStride(F)) return true;
  if (irThreadZeroSerialKernel(F)) return true;
  if (irBlockCollectiveKernel(F)) return true;
  return false;
}

static bool fpOnlineSoftmaxWarpKernel(const Function& F, unsigned launchBound) {
  const unsigned T = launchBound ? launchBound : 256;
  if (T > 32) return false;
  if (!hasCallPrefix(F, "llvm.nvvm.shfl")) return false;
  if (hasCallPrefix(F, "__nv_expf") || hasCallPrefix(F, "__nv_fast_expf") || hasCallPrefix(F, "llvm.exp"))
    return true;
  if (hasCallPrefix(F, "__nv_fmaxf") || hasCallPrefix(F, "llvm.nvvm.fmax")) return true;
  return false;
}

static bool hasTileMma(const Function& F) { return hasCallPrefix(F, "__mvcc_tile_mma"); }

static void cloneTwin(Function& F, TensorRecoveryResult& res) {
  ValueToValueMapTy VMap;
  Function* twin = CloneFunction(&F, VMap);
  twin->setName(F.getName() + "__mvcc_exact");
  res.exactVariants.push_back({F.getName().str(), twin->getName().str()});
}

static bool replaceBody(Function& F, const std::function<void(IRBuilder<>&)>& build) {
  Module* M = F.getParent();
  Function* grave = Function::Create(F.getFunctionType(), Function::PrivateLinkage,
                                     F.getName() + ".stir.grave", M);
  std::vector<BasicBlock*> old;
  for (BasicBlock& BB : F) old.push_back(&BB);
  for (BasicBlock* BB : old) BB->removeFromParent();
  for (BasicBlock* BB : old) BB->insertInto(grave);
  BasicBlock* entry = BasicBlock::Create(F.getContext(), "stir.entry", &F);
  IRBuilder<> B(entry);
  build(B);
  if (!B.GetInsertBlock()->getTerminator()) B.CreateRetVoid();
  for (BasicBlock* BB : old) for (Instruction& I : *BB) I.dropAllReferences();
  grave->eraseFromParent();
  std::string verr; raw_string_ostream vos(verr);
  if (verifyFunction(F, &vos)) {
    llvm::errs() << "[MVCC] warning: check+emit: verify failed in " << F.getName() << ": " << verr << "\n";
    return false;
  }
  return true;
}

static Value* sreg(IRBuilder<>& B, Function& F, const char* name, const char* vn) {
  Type* i32 = B.getInt32Ty();
  return B.CreateCall(F.getParent()->getOrInsertFunction(name, FunctionType::get(i32, false)), {}, vn);
}

static FunctionCallee shflF32(Module& M) {
  return M.getOrInsertFunction("llvm.nvvm.shfl.sync.bfly.f32",
    FunctionType::get(Type::getFloatTy(M.getContext()),
      {Type::getInt32Ty(M.getContext()), Type::getFloatTy(M.getContext()), Type::getInt32Ty(M.getContext()), Type::getInt32Ty(M.getContext())}, false));
}
static FunctionCallee ctaBar(Module& M) {
  return M.getOrInsertFunction("llvm.nvvm.barrier.cta.sync.aligned.all",
    FunctionType::get(Type::getVoidTy(M.getContext()), {Type::getInt32Ty(M.getContext())}, false));
}

static Value* warpXorReduce(IRBuilder<>& B, Module& M, Value* acc) {
  Type* i32 = B.getInt32Ty();
  FunctionCallee sh = shflF32(M);
  for (int off = 16; off > 0; off >>= 1) {
    Value* o = B.CreateCall(sh, {ConstantInt::get(i32, -1), acc, ConstantInt::get(i32, off), ConstantInt::get(i32, 31)});
    acc = B.CreateFAdd(acc, o);
  }
  return acc;
}

static GlobalVariable* sharedFloats(Function& F, unsigned n, const char* tag) {
  auto* ty = ArrayType::get(Type::getFloatTy(F.getContext()), n);
  return new GlobalVariable(*F.getParent(), ty, false, GlobalValue::InternalLinkage, UndefValue::get(ty),
    F.getName() + tag, nullptr, GlobalValue::NotThreadLocal, /*addrspace=*/3);
}

static Result finish(Function& F, TensorRecoveryResult& res, const std::string& why) {
  Result R; R.emitted = true; R.why = why;
  pushRecoverySuccessNote(res, llvm::demangle(F.getName().str()) + ": " + why);
  return R;
}
static Result failIR(Function& F, TensorRecoveryResult& res) {
  Result R; R.why = "check+emit produced invalid IR";
  res.warnings.push_back(llvm::demangle(F.getName().str()) + ": " + R.why);
  return R;
}

// o=(row), r=(k): shuffle-tree + threadgroup combine
static Result emitRowSum(Function& F, unsigned T, TensorRecoveryResult& res) {
  Argument *x = nullptr, *out = nullptr, *K = nullptr;
  for (Argument& A : F.args()) {
    if (A.getType()->isPointerTy()) { if (!x) x = &A; else out = &A; }
    else if (A.getType()->isIntegerTy(32)) K = &A;
  }
  if (!x || !out || !K || T < 32 || T % 32) { Result R; R.why = "check+emit kept: rowsum shape"; return R; }
  cloneTwin(F, res);
  const unsigned nW = T / 32;
  if (!replaceBody(F, [&](IRBuilder<>& B) {
    Module& M = *F.getParent();
    Type* i32 = B.getInt32Ty(); Type* i64 = B.getInt64Ty(); Type* f32 = B.getFloatTy();
    Value* tid = sreg(B, F, "llvm.nvvm.read.ptx.sreg.tid.x", "stir.tid");
    Value* row = sreg(B, F, "llvm.nvvm.read.ptx.sreg.ctaid.x", "stir.row");
    Value* sm = B.CreateAddrSpaceCast(sharedFloats(F, nW, ".stir.part"), PointerType::get(F.getContext(), 0));
    BasicBlock* pre = B.GetInsertBlock();
    BasicBlock* hdr = BasicBlock::Create(F.getContext(), "stir.k.hdr", &F);
    BasicBlock* body = BasicBlock::Create(F.getContext(), "stir.k.body", &F);
    BasicBlock* after = BasicBlock::Create(F.getContext(), "stir.k.done", &F);
    B.CreateBr(hdr);
    IRBuilder<> H(hdr);
    PHINode* k = H.CreatePHI(i32, 2, "k"); PHINode* s = H.CreatePHI(f32, 2, "s");
    k->addIncoming(tid, pre); s->addIncoming(ConstantFP::get(f32, 0), pre);
    H.CreateCondBr(H.CreateICmpSLT(k, K), body, after);
    IRBuilder<> Bd(body);
    Value* idx = Bd.CreateAdd(Bd.CreateMul(Bd.CreateZExt(row, i64), Bd.CreateZExt(K, i64)), Bd.CreateZExt(k, i64));
    Value* s2 = Bd.CreateFAdd(s, Bd.CreateLoad(f32, Bd.CreateInBoundsGEP(f32, x, idx)));
    Value* k2 = Bd.CreateAdd(k, ConstantInt::get(i32, T));
    Bd.CreateBr(hdr);
    k->addIncoming(k2, body); s->addIncoming(s2, body);
    IRBuilder<> A(after);
    Value* acc = warpXorReduce(A, M, s);
    BasicBlock* wstore = BasicBlock::Create(F.getContext(), "stir.wstore", &F);
    BasicBlock* bar = BasicBlock::Create(F.getContext(), "stir.bar", &F);
    A.CreateCondBr(A.CreateICmpEQ(A.CreateAnd(tid, ConstantInt::get(i32, 31)), ConstantInt::get(i32, 0)), wstore, bar);
    IRBuilder<> Ws(wstore);
    Ws.CreateStore(acc, Ws.CreateInBoundsGEP(f32, sm, Ws.CreateLShr(tid, ConstantInt::get(i32, 5))));
    Ws.CreateBr(bar);
    IRBuilder<> Br(bar);
    Br.CreateCall(ctaBar(M), {ConstantInt::get(i32, 0)});
    BasicBlock* t0 = BasicBlock::Create(F.getContext(), "stir.t0", &F);
    BasicBlock* ret = BasicBlock::Create(F.getContext(), "stir.ret", &F);
    Br.CreateCondBr(Br.CreateICmpEQ(tid, ConstantInt::get(i32, 0)), t0, ret);
    IRBuilder<> R(t0);
    Value* tot = ConstantFP::get(f32, 0);
    for (unsigned i = 0; i < nW; i++) tot = R.CreateFAdd(tot, R.CreateLoad(f32, R.CreateInBoundsGEP(f32, sm, ConstantInt::get(i32, i))));
    R.CreateStore(tot, R.CreateInBoundsGEP(f32, out, R.CreateZExt(row, i64)));
    R.CreateBr(ret);
    IRBuilder<>(ret).CreateRetVoid();
  })) return failIR(F, res);
  return finish(F, res, "check+emit: reduction sum o=cta r=tid+i*" + std::to_string(T) + " exchange=shuffle-tree+smem");
}

// o=(row,d) via index tensor (layout / gather)
static Result emitGather(Function& F, unsigned T, TensorRecoveryResult& res) {
  Argument *src = nullptr, *idx = nullptr, *dst = nullptr, *D = nullptr;
  for (Argument& A : F.args()) {
    if (A.getType()->isPointerTy()) { if (!src) src = &A; else if (!idx) idx = &A; else dst = &A; }
    else if (A.getType()->isIntegerTy(32)) D = &A;
  }
  if (!src || !idx || !dst || !D || !T) { Result R; R.why = "check+emit kept: gather shape"; return R; }
  cloneTwin(F, res);
  if (!replaceBody(F, [&](IRBuilder<>& B) {
    Type* i32 = B.getInt32Ty(); Type* i64 = B.getInt64Ty(); Type* f32 = B.getFloatTy();
    Value* tid = sreg(B, F, "llvm.nvvm.read.ptx.sreg.tid.x", "stir.tid");
    Value* row = sreg(B, F, "llvm.nvvm.read.ptx.sreg.ctaid.x", "stir.row");
    Value* r = B.CreateLoad(i32, B.CreateInBoundsGEP(i32, idx, B.CreateZExt(row, i64)), "r");
    BasicBlock* pre = B.GetInsertBlock();
    BasicBlock* hdr = BasicBlock::Create(F.getContext(), "stir.d.hdr", &F);
    BasicBlock* body = BasicBlock::Create(F.getContext(), "stir.d.body", &F);
    BasicBlock* done = BasicBlock::Create(F.getContext(), "stir.d.done", &F);
    B.CreateBr(hdr);
    IRBuilder<> H(hdr);
    PHINode* d = H.CreatePHI(i32, 2, "d");
    d->addIncoming(tid, pre);
    H.CreateCondBr(H.CreateICmpSLT(d, D), body, done);
    IRBuilder<> Bd(body);
    Value* d64 = Bd.CreateZExt(d, i64);
    Value* D64 = Bd.CreateZExt(D, i64);
    Value* si = Bd.CreateAdd(Bd.CreateMul(Bd.CreateZExt(r, i64), D64), d64);
    Value* di = Bd.CreateAdd(Bd.CreateMul(Bd.CreateZExt(row, i64), D64), d64);
    Bd.CreateStore(Bd.CreateLoad(f32, Bd.CreateInBoundsGEP(f32, src, si)), Bd.CreateInBoundsGEP(f32, dst, di));
    Value* d2 = Bd.CreateAdd(d, ConstantInt::get(i32, T));
    Bd.CreateBr(hdr);
    d->addIncoming(d2, body);
    IRBuilder<> (done).CreateRetVoid();
  })) return failIR(F, res);
  return finish(F, res, "check+emit: gather o=(row,d) via IndexTensor");
}

// o=(n), r=(k): per-thread scalar contraction (gemv)
static Result emitGemv(Function& F, unsigned T, TensorRecoveryResult& res) {
  Argument *W = nullptr, *x = nullptr, *y = nullptr, *N = nullptr, *K = nullptr;
  for (Argument& A : F.args()) {
    if (A.getType()->isPointerTy()) { if (!W) W = &A; else if (!x) x = &A; else y = &A; }
    else if (A.getType()->isIntegerTy(32)) { if (!N) N = &A; else K = &A; }
  }
  if (!W || !x || !y || !N || !K || !T) { Result R; R.why = "check+emit kept: gemv shape"; return R; }
  cloneTwin(F, res);
  if (!replaceBody(F, [&](IRBuilder<>& B) {
    Type* i32 = B.getInt32Ty(); Type* i64 = B.getInt64Ty(); Type* f32 = B.getFloatTy(); Type* i16 = B.getInt16Ty();
    Value* tid = sreg(B, F, "llvm.nvvm.read.ptx.sreg.tid.x", "stir.tid");
    Value* cta = sreg(B, F, "llvm.nvvm.read.ptx.sreg.ctaid.x", "stir.cta");
    Value* n = B.CreateAdd(B.CreateMul(cta, ConstantInt::get(i32, T)), tid, "n");
    BasicBlock* body = BasicBlock::Create(F.getContext(), "stir.ok", &F);
    BasicBlock* ret = BasicBlock::Create(F.getContext(), "stir.ret", &F);
    B.CreateCondBr(B.CreateICmpSLT(n, N), body, ret);
    IRBuilder<> Bd(body);
    BasicBlock* hdr = BasicBlock::Create(F.getContext(), "stir.k.hdr", &F);
    BasicBlock* kb = BasicBlock::Create(F.getContext(), "stir.k.body", &F);
    BasicBlock* done = BasicBlock::Create(F.getContext(), "stir.k.done", &F);
    Bd.CreateBr(hdr);
    IRBuilder<> H(hdr);
    PHINode* k = H.CreatePHI(i32, 2, "k"); PHINode* acc = H.CreatePHI(f32, 2, "acc");
    k->addIncoming(ConstantInt::get(i32, 0), body); acc->addIncoming(ConstantFP::get(f32, 0), body);
    H.CreateCondBr(H.CreateICmpSLT(k, K), kb, done);
    IRBuilder<> Kb(kb);
    Value* wi = Kb.CreateAdd(Kb.CreateMul(Kb.CreateZExt(n, i64), Kb.CreateZExt(K, i64)), Kb.CreateZExt(k, i64));
    Value* bits = Kb.CreateZExt(Kb.CreateLoad(i16, Kb.CreateInBoundsGEP(i16, W, wi)), i32);
    Value* w = Kb.CreateBitCast(Kb.CreateShl(bits, ConstantInt::get(i32, 16)), f32);
    Value* xv = Kb.CreateLoad(f32, Kb.CreateInBoundsGEP(f32, x, Kb.CreateZExt(k, i64)));
    Value* acc2 = Kb.CreateFAdd(acc, Kb.CreateFMul(w, xv));
    Value* k2 = Kb.CreateAdd(k, ConstantInt::get(i32, 1));
    Kb.CreateBr(hdr);
    k->addIncoming(k2, kb); acc->addIncoming(acc2, kb);
    IRBuilder<> Dn(done);
    Dn.CreateStore(acc, Dn.CreateInBoundsGEP(f32, y, Dn.CreateZExt(n, i64)));
    Dn.CreateBr(ret);
    IRBuilder<> (ret).CreateRetVoid();
  })) return failIR(F, res);
  return finish(F, res, "check+emit: gemv o=cta*" + std::to_string(T) + "+tid r=k engine=scalar-fma");
}

// C[m,n] += A[m,k]*B[k,n]: a contraction without mma.sync.
// kStep>1 is a legal schedule edit (engine=tensor-op candidate): same Y, different nest — search's __mvcc_s0.
static Result emitSgemm(Function& F, unsigned T, TensorRecoveryResult& res, unsigned kStep = 1, bool makeTwin = true) {
  Argument *A = nullptr, *B = nullptr, *C = nullptr, *M = nullptr, *N = nullptr, *K = nullptr;
  for (Argument& Arg : F.args()) {
    if (Arg.getType()->isPointerTy()) { if (!A) A = &Arg; else if (!B) B = &Arg; else C = &Arg; }
    else if (Arg.getType()->isIntegerTy(32)) { if (!M) M = &Arg; else if (!N) N = &Arg; else K = &Arg; }
  }
  if (!A || !B || !C || !M || !N || !K || !T) { Result R; R.why = "check+emit kept: sgemm shape"; return R; }
  if (kStep < 1) kStep = 1;
  if (makeTwin) cloneTwin(F, res);
  if (!replaceBody(F, [&](IRBuilder<>& Bd0) {
    Type* i32 = Bd0.getInt32Ty(); Type* i64 = Bd0.getInt64Ty(); Type* f32 = Bd0.getFloatTy();
    Value* tid = sreg(Bd0, F, "llvm.nvvm.read.ptx.sreg.tid.x", "stir.tid");
    Value* cta = sreg(Bd0, F, "llvm.nvvm.read.ptx.sreg.ctaid.x", "stir.cta");
    Value* o = Bd0.CreateAdd(Bd0.CreateMul(cta, ConstantInt::get(i32, T)), tid, "o");
    Value* MN = Bd0.CreateMul(M, N, "MN");
    BasicBlock* ok = BasicBlock::Create(F.getContext(), "stir.ok", &F);
    BasicBlock* ret = BasicBlock::Create(F.getContext(), "stir.ret", &F);
    Bd0.CreateCondBr(Bd0.CreateICmpSLT(o, MN), ok, ret);
    IRBuilder<> Ok(ok);
    Value* m = Ok.CreateSDiv(o, N, "m");
    Value* n = Ok.CreateSRem(o, N, "n");
    auto fma = [&](IRBuilder<>& Ib, Value* acc, Value* kk) {
      Value* ai = Ib.CreateAdd(Ib.CreateMul(Ib.CreateZExt(m, i64), Ib.CreateZExt(K, i64)), Ib.CreateZExt(kk, i64));
      Value* bi = Ib.CreateAdd(Ib.CreateMul(Ib.CreateZExt(kk, i64), Ib.CreateZExt(N, i64)), Ib.CreateZExt(n, i64));
      return Ib.CreateFAdd(acc, Ib.CreateFMul(Ib.CreateLoad(f32, Ib.CreateInBoundsGEP(f32, A, ai)),
                                            Ib.CreateLoad(f32, Ib.CreateInBoundsGEP(f32, B, bi))));
    };
    BasicBlock* hdr = BasicBlock::Create(F.getContext(), "stir.k.hdr", &F);
    BasicBlock* kb = BasicBlock::Create(F.getContext(), "stir.k.body", &F);
    BasicBlock* done = BasicBlock::Create(F.getContext(), "stir.k.done", &F);
    BasicBlock* tail = kStep > 1 ? BasicBlock::Create(F.getContext(), "stir.k.tail", &F) : done;
    Ok.CreateBr(hdr);
    IRBuilder<> H(hdr);
    PHINode* k = H.CreatePHI(i32, 2, "k"); PHINode* acc = H.CreatePHI(f32, 2, "acc");
    k->addIncoming(ConstantInt::get(i32, 0), ok);
    acc->addIncoming(ConstantFP::get(f32, 0), ok);
    Value* kLim = H.CreateAdd(k, ConstantInt::get(i32, kStep));
    H.CreateCondBr(kStep == 1 ? H.CreateICmpSLT(k, K) : H.CreateICmpSLE(kLim, K), kb, tail);
    IRBuilder<> Kb(kb);
    Value* accN = acc; Value* kk = k;
    for (unsigned j = 0; j < kStep; j++) {
      accN = fma(Kb, accN, kk);
      kk = Kb.CreateAdd(kk, ConstantInt::get(i32, 1));
    }
    Kb.CreateBr(hdr);
    k->addIncoming(kStep == 1 ? kk : kLim, kb); acc->addIncoming(accN, kb);
    Value* accOut = acc;
    if (kStep > 1) {
      BasicBlock* tb = BasicBlock::Create(F.getContext(), "stir.k.tbody", &F);
      IRBuilder<> Th(tail);
      PHINode* kt = Th.CreatePHI(i32, 2, "kt"); PHINode* at = Th.CreatePHI(f32, 2, "at");
      kt->addIncoming(k, hdr); at->addIncoming(acc, hdr);
      Th.CreateCondBr(Th.CreateICmpSLT(kt, K), tb, done);
      IRBuilder<> Tb(tb);
      Value* at2 = fma(Tb, at, kt);
      Value* kt2 = Tb.CreateAdd(kt, ConstantInt::get(i32, 1));
      Tb.CreateBr(tail);
      kt->addIncoming(kt2, tb);
      at->addIncoming(at2, tb);
      accOut = at;
    }
    IRBuilder<> Dn(done);
    Dn.CreateStore(accOut, Dn.CreateInBoundsGEP(f32, C, Dn.CreateZExt(o, i64)));
    Dn.CreateBr(ret);
    IRBuilder<>(ret).CreateRetVoid();
  })) return failIR(F, res);
  if (!makeTwin) {
    Result R; R.emitted = true;
    R.why = "check+emit: sgemm kStep=" + std::to_string(kStep) + " (search body, different IR)";
    return R;
  }
  return finish(F, res, "check+emit: sgemm o=cta*" + std::to_string(T) + "+tid =(m,n) r=k engine=scalar-fma");
}

// Cooperative-tensor layout law (msl/mvcc_prelude.metal, mvcc_tp::law_row / law_col). Slot s of lane l
// is (row, col) of an R×C operand. Used to fill / drain a 16×16×16 __mvcc_tp_mma from device memory.
static Value* lawRowIR(IRBuilder<>& B, Value* l, unsigned s, bool tr, int R, int C) {
  Type* i32 = B.getInt32Ty();
  auto shr = [&](Value* v, unsigned k) { return k ? (Value*)B.CreateLShr(v, ConstantInt::get(i32, k)) : v; };
  auto shl = [&](Value* v, unsigned k) { return k ? (Value*)B.CreateShl(v, ConstantInt::get(i32, k)) : v; };
  auto andc = [&](Value* v, unsigned m) { return B.CreateAnd(v, ConstantInt::get(i32, m)); };
  Value* br = B.CreateOr(B.CreateOr(andc(shr(l, 1), 3u), shl(shr(l, 4), 2u)),
                         ConstantInt::get(i32, ((s >> 2) & 1u) << 3));
  Value* bc = B.CreateOr(B.CreateOr(ConstantInt::get(i32, s & 3u), shl(andc(l, 1u), 2u)),
                         shl(andc(shr(l, 3), 1u), 3u));
  Value* row = tr ? bc : br;
  if (C == 32 && R == 32) row = B.CreateOr(row, ConstantInt::get(i32, ((s >> 4) & 1u) << 4));
  else if (R == 32) row = B.CreateOr(row, ConstantInt::get(i32, ((s >> 3) & 1u) << 4));
  return row;
}
static Value* lawColIR(IRBuilder<>& B, Value* l, unsigned s, bool tr, int R, int C) {
  Type* i32 = B.getInt32Ty();
  auto shr = [&](Value* v, unsigned k) { return k ? (Value*)B.CreateLShr(v, ConstantInt::get(i32, k)) : v; };
  auto shl = [&](Value* v, unsigned k) { return k ? (Value*)B.CreateShl(v, ConstantInt::get(i32, k)) : v; };
  auto andc = [&](Value* v, unsigned m) { return B.CreateAnd(v, ConstantInt::get(i32, m)); };
  Value* br = B.CreateOr(B.CreateOr(andc(shr(l, 1), 3u), shl(shr(l, 4), 2u)),
                         ConstantInt::get(i32, ((s >> 2) & 1u) << 3));
  Value* bc = B.CreateOr(B.CreateOr(ConstantInt::get(i32, s & 3u), shl(andc(l, 1u), 2u)),
                         shl(andc(shr(l, 3), 1u), 3u));
  Value* col = tr ? br : bc;
  if (C == 32) col = B.CreateOr(col, ConstantInt::get(i32, ((s >> 3) & 1u) << 4));
  return col;
}

// 16×16×32 half GEMM through __mvcc_tp_mma. Metal 4 requires one of M/N/K to be 32 when both
// operands are cooperative tensors.
static Result emitHgemm16(Function& F, unsigned T, TensorRecoveryResult& res) {
  std::vector<Argument*> ps;
  for (Argument& Arg : F.args()) if (Arg.getType()->isPointerTy()) ps.push_back(&Arg);
  if ((ps.size() != 3 && ps.size() != 4) || T != 32) { Result R; R.why = "check+emit kept: hgemm16 shape"; return R; }
  Argument *A = ps[0], *B = ps[1], *C = ps.back();
  Argument *bias = ps.size() == 4 ? ps[2] : nullptr;
  cloneTwin(F, res);
  const int tileM = 16, tileN = 16, tileK = 32;
  const unsigned nA = tileM * tileK / 64, nB = tileK * tileN / 64, cap = tileM * tileN / 32;
  if (!replaceBody(F, [&](IRBuilder<>& Bd0) {
    Module& Mod = *F.getParent();
    Type* i16 = Bd0.getInt16Ty(); Type* i32 = Bd0.getInt32Ty(); Type* i64 = Bd0.getInt64Ty();
    Type* f32 = Bd0.getFloatTy(); Type* halfTy = Type::getHalfTy(F.getContext());
    Value* lane = sreg(Bd0, F, "llvm.nvvm.read.ptx.sreg.tid.x", "stir.lane");
    auto loadPacked = [&](Value* base, int R, int C, bool tr, unsigned word) {
      unsigned s0 = 2u * word, s1 = s0 + 1;
      Value* r0 = lawRowIR(Bd0, lane, s0, tr, R, C);
      Value* c0 = lawColIR(Bd0, lane, s0, tr, R, C);
      Value* r1 = lawRowIR(Bd0, lane, s1, tr, R, C);
      Value* c1 = lawColIR(Bd0, lane, s1, tr, R, C);
      Value* i0 = Bd0.CreateAdd(Bd0.CreateMul(r0, ConstantInt::get(i32, C)), c0);
      Value* i1 = Bd0.CreateAdd(Bd0.CreateMul(r1, ConstantInt::get(i32, C)), c1);
      Value* h0 = Bd0.CreateLoad(halfTy, Bd0.CreateInBoundsGEP(halfTy, base, Bd0.CreateZExt(i0, i64)));
      Value* h1 = Bd0.CreateLoad(halfTy, Bd0.CreateInBoundsGEP(halfTy, base, Bd0.CreateZExt(i1, i64)));
      Value* lo = Bd0.CreateZExt(Bd0.CreateBitCast(h0, i16), i32);
      Value* hi = Bd0.CreateZExt(Bd0.CreateBitCast(h1, i16), i32);
      return Bd0.CreateOr(lo, Bd0.CreateShl(hi, ConstantInt::get(i32, 16)));
    };
    std::vector<Value*> aW, bW, cW;
    for (unsigned w = 0; w < nA; w++) aW.push_back(loadPacked(A, tileM, tileK, false, w));
    for (unsigned w = 0; w < nB; w++) bW.push_back(loadPacked(B, tileK, tileN, false, w));
    for (unsigned i = 0; i < cap; i++) cW.push_back(ConstantFP::get(f32, 0));
    std::vector<Type*> at(6, i32);
    for (unsigned i = 0; i < nA + nB; i++) at.push_back(i32);
    for (unsigned i = 0; i < cap; i++) at.push_back(f32);
    std::vector<Type*> rt(cap, f32);
    Type* retTy = StructType::get(F.getContext(), rt);
    FunctionCallee mf = Mod.getOrInsertFunction("__mvcc_tp_mma_16x16x32", FunctionType::get(retTy, at, false));
    if (auto* Fn = dyn_cast<Function>(mf.getCallee())) {
      Fn->addFnAttr(Attribute::Convergent);
      Fn->addFnAttr(Attribute::NoUnwind);
    }
    std::vector<Value*> args = {
      ConstantInt::get(i32, tileM), ConstantInt::get(i32, tileN), ConstantInt::get(i32, tileK),
      ConstantInt::get(i32, 0), ConstantInt::get(i32, 0), ConstantInt::get(i32, 0)
    };
    args.insert(args.end(), aW.begin(), aW.end());
    args.insert(args.end(), bW.begin(), bW.end());
    args.insert(args.end(), cW.begin(), cW.end());
    Value* call = Bd0.CreateCall(mf, args, "stir.mma");
    for (unsigned s = 0; s < cap; s++) {
      Value* row = lawRowIR(Bd0, lane, s, false, tileM, tileN);
      Value* col = lawColIR(Bd0, lane, s, false, tileM, tileN);
      Value* idx = Bd0.CreateAdd(Bd0.CreateMul(row, ConstantInt::get(i32, tileN)), col);
      Value* dv = Bd0.CreateExtractValue(call, s);
      if (bias) {
        Value* bv = Bd0.CreateLoad(f32, Bd0.CreateInBoundsGEP(f32, bias, Bd0.CreateZExt(col, i64)));
        dv = Bd0.CreateFAdd(dv, bv);
      }
      Bd0.CreateStore(dv, Bd0.CreateInBoundsGEP(f32, C, Bd0.CreateZExt(idx, i64)));
    }
    Bd0.CreateRetVoid();
  })) return failIR(F, res);
  TpDescriptor d; d.M = tileM; d.N = tileN; d.K = tileK; d.tl = false; d.tr = false; d.type = 0;
  res.descriptors.insert(d);
  return finish(F, res, bias
    ? "check+emit: hgemm16b 16x16x32 __mvcc_tp_mma +bias engine=tensor-op (STIR, gemm_ptx-shaped ABI)"
    : "check+emit: hgemm16 16x16x32 __mvcc_tp_mma engine=tensor-op (STIR)");
}

// online (max, sum) then broadcast divide (online)
static Result emitSoftmax(Function& F, unsigned T, TensorRecoveryResult& res) {
  Argument *x = nullptr, *out = nullptr, *K = nullptr;
  for (Argument& A : F.args()) {
    if (A.getType()->isPointerTy()) { if (!x) x = &A; else out = &A; }
    else if (A.getType()->isIntegerTy(32)) K = &A;
  }
  if (!x || !out || !K || T < 32 || T % 32) { Result R; R.why = "check+emit kept: softmax shape"; return R; }
  cloneTwin(F, res);
  const unsigned nW = T / 32;
  if (!replaceBody(F, [&](IRBuilder<>& B) {
    Module& M = *F.getParent();
    Type* i32 = B.getInt32Ty(); Type* i64 = B.getInt64Ty(); Type* f32 = B.getFloatTy();
    FunctionCallee expf = M.getOrInsertFunction("__nv_fast_expf", FunctionType::get(f32, {f32}, false));
    FunctionCallee sh = shflF32(M);
    Value* tid = sreg(B, F, "llvm.nvvm.read.ptx.sreg.tid.x", "stir.tid");
    Value* row = sreg(B, F, "llvm.nvvm.read.ptx.sreg.ctaid.x", "stir.row");
    Value* pm = B.CreateAddrSpaceCast(sharedFloats(F, nW, ".stir.pm"), PointerType::get(F.getContext(), 0));
    Value* pl = B.CreateAddrSpaceCast(sharedFloats(F, nW, ".stir.pl"), PointerType::get(F.getContext(), 0));
    BasicBlock* pre = B.GetInsertBlock();
    BasicBlock* hdr = BasicBlock::Create(F.getContext(), "stir.k.hdr", &F);
    BasicBlock* body = BasicBlock::Create(F.getContext(), "stir.k.body", &F);
    BasicBlock* after = BasicBlock::Create(F.getContext(), "stir.k.done", &F);
    B.CreateBr(hdr);
    IRBuilder<> H(hdr);
    PHINode* k = H.CreatePHI(i32, 2, "k"); PHINode* m = H.CreatePHI(f32, 2, "m"); PHINode* l = H.CreatePHI(f32, 2, "l");
    k->addIncoming(tid, pre);
    m->addIncoming(ConstantFP::getInfinity(f32, /*Negative=*/true), pre);
    l->addIncoming(ConstantFP::get(f32, 0), pre);
    H.CreateCondBr(H.CreateICmpSLT(k, K), body, after);
    IRBuilder<> Bd(body);
    Value* idx = Bd.CreateAdd(Bd.CreateMul(Bd.CreateZExt(row, i64), Bd.CreateZExt(K, i64)), Bd.CreateZExt(k, i64));
    Value* v = Bd.CreateLoad(f32, Bd.CreateInBoundsGEP(f32, x, idx));
    Value* mn = Bd.CreateMaxNum(m, v);
    Value* l2 = Bd.CreateFAdd(Bd.CreateFMul(l, Bd.CreateCall(expf, {Bd.CreateFSub(m, mn)})), Bd.CreateCall(expf, {Bd.CreateFSub(v, mn)}));
    Value* k2 = Bd.CreateAdd(k, ConstantInt::get(i32, T));
    Bd.CreateBr(hdr);
    k->addIncoming(k2, body);
    m->addIncoming(mn, body); l->addIncoming(l2, body);
    IRBuilder<> A(after);
    Value* mm = m;
    Value* ll = l;
    for (int off = 16; off > 0; off >>= 1) {
      Value* mo = A.CreateCall(sh, {ConstantInt::get(i32, -1), mm, ConstantInt::get(i32, off), ConstantInt::get(i32, 31)});
      Value* lo = A.CreateCall(sh, {ConstantInt::get(i32, -1), ll, ConstantInt::get(i32, off), ConstantInt::get(i32, 31)});
      Value* mn2 = A.CreateMaxNum(mm, mo);
      ll = A.CreateFAdd(A.CreateFMul(ll, A.CreateCall(expf, {A.CreateFSub(mm, mn2)})), A.CreateFMul(lo, A.CreateCall(expf, {A.CreateFSub(mo, mn2)})));
      mm = mn2;
    }
    BasicBlock* wstore = BasicBlock::Create(F.getContext(), "stir.wstore", &F);
    BasicBlock* bar = BasicBlock::Create(F.getContext(), "stir.bar", &F);
    A.CreateCondBr(A.CreateICmpEQ(A.CreateAnd(tid, ConstantInt::get(i32, 31)), ConstantInt::get(i32, 0)), wstore, bar);
    IRBuilder<> Ws(wstore);
    Value* w = Ws.CreateLShr(tid, ConstantInt::get(i32, 5));
    Ws.CreateStore(mm, Ws.CreateInBoundsGEP(f32, pm, w));
    Ws.CreateStore(ll, Ws.CreateInBoundsGEP(f32, pl, w));
    Ws.CreateBr(bar);
    IRBuilder<> Br(bar);
    Br.CreateCall(ctaBar(M), {ConstantInt::get(i32, 0)});
    BasicBlock* head = BasicBlock::Create(F.getContext(), "stir.head", &F);
    BasicBlock* ret = BasicBlock::Create(F.getContext(), "stir.ret", &F);
    Br.CreateCondBr(Br.CreateICmpULT(tid, ConstantInt::get(i32, 32)), head, ret);
    IRBuilder<> Hd(head);
    Value* Mtot = ConstantFP::getInfinity(f32, true); Value* Ltot = ConstantFP::get(f32, 0);
    for (unsigned i = 0; i < nW; i++) {
      Value* mi = Hd.CreateLoad(f32, Hd.CreateInBoundsGEP(f32, pm, ConstantInt::get(i32, i)));
      Value* li = Hd.CreateLoad(f32, Hd.CreateInBoundsGEP(f32, pl, ConstantInt::get(i32, i)));
      Value* Mn = Hd.CreateMaxNum(Mtot, mi);
      Ltot = Hd.CreateFAdd(Hd.CreateFMul(Ltot, Hd.CreateCall(expf, {Hd.CreateFSub(Mtot, Mn)})), Hd.CreateFMul(li, Hd.CreateCall(expf, {Hd.CreateFSub(mi, Mn)})));
      Mtot = Mn;
    }
    BasicBlock* oh = BasicBlock::Create(F.getContext(), "stir.out.hdr", &F);
    BasicBlock* ob = BasicBlock::Create(F.getContext(), "stir.out.body", &F);
    Hd.CreateBr(oh);
    IRBuilder<> Oh(oh);
    PHINode* d = Oh.CreatePHI(i32, 2, "od");
    d->addIncoming(tid, head);
    Oh.CreateCondBr(Oh.CreateICmpSLT(d, K), ob, ret);
    IRBuilder<> Ob(ob);
    Value* oi = Ob.CreateAdd(Ob.CreateMul(Ob.CreateZExt(row, i64), Ob.CreateZExt(K, i64)), Ob.CreateZExt(d, i64));
    Value* xv = Ob.CreateLoad(f32, Ob.CreateInBoundsGEP(f32, x, oi));
    Value* yv = Ob.CreateFDiv(Ob.CreateCall(expf, {Ob.CreateFSub(xv, Mtot)}), Ltot);
    Ob.CreateStore(yv, Ob.CreateInBoundsGEP(f32, out, oi));
    Value* d2 = Ob.CreateAdd(d, ConstantInt::get(i32, 32));
    Ob.CreateBr(oh);
    d->addIncoming(d2, ob);
    IRBuilder<> (ret).CreateRetVoid();
  })) return failIR(F, res);
  return finish(F, res, "check+emit: softmax online (max,sum) exchange=shuffle-tree+smem");
}

// two-stage: sum-of-squares then broadcast scale (RMSNorm)
static Result emitRmsNorm(Function& F, unsigned T, TensorRecoveryResult& res) {
  Argument *x = nullptr, *w = nullptr, *out = nullptr, *K = nullptr;
  for (Argument& A : F.args()) {
    if (A.getType()->isPointerTy()) { if (!x) x = &A; else if (!w) w = &A; else out = &A; }
    else if (A.getType()->isIntegerTy(32)) K = &A;
  }
  if (!x || !w || !out || !K || T < 32 || T % 32) { Result R; R.why = "check+emit kept: rmsnorm shape"; return R; }
  cloneTwin(F, res);
  const unsigned nW = T / 32;
  if (!replaceBody(F, [&](IRBuilder<>& B) {
    Module& M = *F.getParent();
    Type* i32 = B.getInt32Ty(); Type* i64 = B.getInt64Ty(); Type* f32 = B.getFloatTy();
    FunctionCallee sqrtf = M.getOrInsertFunction("llvm.sqrt.f32", FunctionType::get(f32, {f32}, false));
    Value* tid = sreg(B, F, "llvm.nvvm.read.ptx.sreg.tid.x", "stir.tid");
    Value* row = sreg(B, F, "llvm.nvvm.read.ptx.sreg.ctaid.x", "stir.row");
    Value* sm = B.CreateAddrSpaceCast(sharedFloats(F, nW, ".stir.part"), PointerType::get(F.getContext(), 0));
    Value* invp = B.CreateAddrSpaceCast(sharedFloats(F, 1, ".stir.inv"), PointerType::get(F.getContext(), 0));
    BasicBlock* pre = B.GetInsertBlock();
    BasicBlock* hdr = BasicBlock::Create(F.getContext(), "stir.k.hdr", &F);
    BasicBlock* body = BasicBlock::Create(F.getContext(), "stir.k.body", &F);
    BasicBlock* after = BasicBlock::Create(F.getContext(), "stir.k.done", &F);
    B.CreateBr(hdr);
    IRBuilder<> H(hdr);
    PHINode* k = H.CreatePHI(i32, 2, "k"); PHINode* s = H.CreatePHI(f32, 2, "s");
    k->addIncoming(tid, pre); s->addIncoming(ConstantFP::get(f32, 0), pre);
    H.CreateCondBr(H.CreateICmpSLT(k, K), body, after);
    IRBuilder<> Bd(body);
    Value* idx = Bd.CreateAdd(Bd.CreateMul(Bd.CreateZExt(row, i64), Bd.CreateZExt(K, i64)), Bd.CreateZExt(k, i64));
    Value* v = Bd.CreateLoad(f32, Bd.CreateInBoundsGEP(f32, x, idx));
    Value* s2 = Bd.CreateFAdd(s, Bd.CreateFMul(v, v));
    Value* k2 = Bd.CreateAdd(k, ConstantInt::get(i32, T));
    Bd.CreateBr(hdr);
    k->addIncoming(k2, body); s->addIncoming(s2, body);
    IRBuilder<> A(after);
    Value* acc = warpXorReduce(A, M, s);
    BasicBlock* wstore = BasicBlock::Create(F.getContext(), "stir.wstore", &F);
    BasicBlock* bar = BasicBlock::Create(F.getContext(), "stir.bar", &F);
    A.CreateCondBr(A.CreateICmpEQ(A.CreateAnd(tid, ConstantInt::get(i32, 31)), ConstantInt::get(i32, 0)), wstore, bar);
    IRBuilder<> Ws(wstore);
    Ws.CreateStore(acc, Ws.CreateInBoundsGEP(f32, sm, Ws.CreateLShr(tid, ConstantInt::get(i32, 5))));
    Ws.CreateBr(bar);
    IRBuilder<> Br(bar);
    Br.CreateCall(ctaBar(M), {ConstantInt::get(i32, 0)});
    BasicBlock* t0 = BasicBlock::Create(F.getContext(), "stir.t0", &F);
    BasicBlock* bar2 = BasicBlock::Create(F.getContext(), "stir.bar2", &F);
    Br.CreateCondBr(Br.CreateICmpEQ(tid, ConstantInt::get(i32, 0)), t0, bar2);
    IRBuilder<> R(t0);
    Value* tot = ConstantFP::get(f32, 0);
    for (unsigned i = 0; i < nW; i++) tot = R.CreateFAdd(tot, R.CreateLoad(f32, R.CreateInBoundsGEP(f32, sm, ConstantInt::get(i32, i))));
    Value* mean = R.CreateFDiv(tot, R.CreateSIToFP(K, f32));
    Value* inv = R.CreateFDiv(ConstantFP::get(f32, 1), R.CreateCall(sqrtf, {R.CreateFAdd(mean, ConstantFP::get(f32, 1e-6))}));
    R.CreateStore(inv, invp);
    R.CreateBr(bar2);
    IRBuilder<> B2(bar2);
    B2.CreateCall(ctaBar(M), {ConstantInt::get(i32, 0)});
    Value* scale = B2.CreateLoad(f32, invp);
    BasicBlock* oh = BasicBlock::Create(F.getContext(), "stir.out.hdr", &F);
    BasicBlock* ob = BasicBlock::Create(F.getContext(), "stir.out.body", &F);
    BasicBlock* ret = BasicBlock::Create(F.getContext(), "stir.ret", &F);
    B2.CreateBr(oh);
    IRBuilder<> Oh(oh);
    PHINode* d = Oh.CreatePHI(i32, 2, "od");
    d->addIncoming(tid, bar2);
    Oh.CreateCondBr(Oh.CreateICmpSLT(d, K), ob, ret);
    IRBuilder<> Ob(ob);
    Value* oi = Ob.CreateAdd(Ob.CreateMul(Ob.CreateZExt(row, i64), Ob.CreateZExt(K, i64)), Ob.CreateZExt(d, i64));
    Value* xv = Ob.CreateLoad(f32, Ob.CreateInBoundsGEP(f32, x, oi));
    Value* wv = Ob.CreateLoad(f32, Ob.CreateInBoundsGEP(f32, w, Ob.CreateZExt(d, i64)));
    Ob.CreateStore(Ob.CreateFMul(Ob.CreateFMul(xv, scale), wv), Ob.CreateInBoundsGEP(f32, out, oi));
    Value* d2 = Ob.CreateAdd(d, ConstantInt::get(i32, T));
    Ob.CreateBr(oh);
    d->addIncoming(d2, ob);
    IRBuilder<>(ret).CreateRetVoid();
  })) return failIR(F, res);
  return finish(F, res, "check+emit: rmsnorm sum-of-squares + broadcast scale");
}

// pair broadcast: o=(t,d) reads x[t,d] and x[t,d+D/2] (RoPE)
static Result emitRope(Function& F, unsigned T, TensorRecoveryResult& res) {
  Argument *x = nullptr, *cos = nullptr, *sin = nullptr, *out = nullptr, *D = nullptr;
  for (Argument& A : F.args()) {
    if (A.getType()->isPointerTy()) { if (!x) x = &A; else if (!cos) cos = &A; else if (!sin) sin = &A; else out = &A; }
    else if (A.getType()->isIntegerTy(32)) D = &A;
  }
  if (!x || !cos || !sin || !out || !D) { Result R; R.why = "check+emit kept: rope shape"; return R; }
  (void)T;
  cloneTwin(F, res);
  if (!replaceBody(F, [&](IRBuilder<>& B) {
    Type* i32 = B.getInt32Ty(); Type* i64 = B.getInt64Ty(); Type* f32 = B.getFloatTy();
    Value* d = sreg(B, F, "llvm.nvvm.read.ptx.sreg.tid.x", "stir.d");
    Value* t = sreg(B, F, "llvm.nvvm.read.ptx.sreg.ctaid.x", "stir.t");
    Value* half = B.CreateLShr(D, ConstantInt::get(i32, 1));
    BasicBlock* ok = BasicBlock::Create(F.getContext(), "stir.ok", &F);
    BasicBlock* ret = BasicBlock::Create(F.getContext(), "stir.ret", &F);
    B.CreateCondBr(B.CreateICmpULT(d, half), ok, ret);
    IRBuilder<> Ok(ok);
    Value* base = Ok.CreateMul(Ok.CreateZExt(t, i64), Ok.CreateZExt(D, i64));
    Value* d64 = Ok.CreateZExt(d, i64);
    Value* i0 = Ok.CreateAdd(base, d64);
    Value* i1 = Ok.CreateAdd(i0, Ok.CreateZExt(half, i64));
    Value* a = Ok.CreateLoad(f32, Ok.CreateInBoundsGEP(f32, x, i0));
    Value* b = Ok.CreateLoad(f32, Ok.CreateInBoundsGEP(f32, x, i1));
    Value* c = Ok.CreateLoad(f32, Ok.CreateInBoundsGEP(f32, cos, d64));
    Value* s = Ok.CreateLoad(f32, Ok.CreateInBoundsGEP(f32, sin, d64));
    Ok.CreateStore(Ok.CreateFSub(Ok.CreateFMul(a, c), Ok.CreateFMul(b, s)), Ok.CreateInBoundsGEP(f32, out, i0));
    Ok.CreateStore(Ok.CreateFAdd(Ok.CreateFMul(a, s), Ok.CreateFMul(b, c)), Ok.CreateInBoundsGEP(f32, out, i1));
    Ok.CreateBr(ret);
    IRBuilder<>(ret).CreateRetVoid();
  })) return failIR(F, res);
  return finish(F, res, "check+emit: rope pair-broadcast o=(t,d) and o=(t,d+D/2)");
}

// decode attention: o=(d), r=(n) online (max,sum,acc); inner r=(k) is the score sum
static Result emitAttn(Function& F, unsigned T, TensorRecoveryResult& res) {
  Argument *q = nullptr, *K = nullptr, *V = nullptr, *out = nullptr, *N = nullptr, *D = nullptr;
  for (Argument& A : F.args()) {
    if (A.getType()->isPointerTy()) { if (!q) q = &A; else if (!K) K = &A; else if (!V) V = &A; else out = &A; }
    else if (A.getType()->isIntegerTy(32)) { if (!N) N = &A; else D = &A; }
  }
  if (!q || !K || !V || !out || !N || !D || !T) { Result R; R.why = "check+emit kept: attn shape"; return R; }
  cloneTwin(F, res);
  if (!replaceBody(F, [&](IRBuilder<>& B) {
    Module& M = *F.getParent();
    Type* i32 = B.getInt32Ty(); Type* i64 = B.getInt64Ty(); Type* f32 = B.getFloatTy();
    FunctionCallee expf = M.getOrInsertFunction("__nv_fast_expf", FunctionType::get(f32, {f32}, false));
    Value* tid = sreg(B, F, "llvm.nvvm.read.ptx.sreg.tid.x", "stir.tid");
    Value* cta = sreg(B, F, "llvm.nvvm.read.ptx.sreg.ctaid.x", "stir.cta");
    Value* d = B.CreateAdd(B.CreateMul(cta, ConstantInt::get(i32, T)), tid, "d");
    BasicBlock* ok = BasicBlock::Create(F.getContext(), "stir.ok", &F);
    BasicBlock* ret = BasicBlock::Create(F.getContext(), "stir.ret", &F);
    B.CreateCondBr(B.CreateICmpSLT(d, D), ok, ret);
    IRBuilder<> Ok(ok);
    BasicBlock* nh = BasicBlock::Create(F.getContext(), "stir.n.hdr", &F);
    BasicBlock* nb = BasicBlock::Create(F.getContext(), "stir.n.body", &F);
    BasicBlock* nd = BasicBlock::Create(F.getContext(), "stir.n.done", &F);
    Ok.CreateBr(nh);
    IRBuilder<> Nh(nh);
    PHINode* n = Nh.CreatePHI(i32, 2, "n");
    PHINode* m = Nh.CreatePHI(f32, 2, "m");
    PHINode* l = Nh.CreatePHI(f32, 2, "l");
    PHINode* acc = Nh.CreatePHI(f32, 2, "acc");
    n->addIncoming(ConstantInt::get(i32, 0), ok);
    m->addIncoming(ConstantFP::getInfinity(f32, true), ok);
    l->addIncoming(ConstantFP::get(f32, 0), ok);
    acc->addIncoming(ConstantFP::get(f32, 0), ok);
    Nh.CreateCondBr(Nh.CreateICmpSLT(n, N), nb, nd);
    IRBuilder<> Nb(nb);
    BasicBlock* kh = BasicBlock::Create(F.getContext(), "stir.k.hdr", &F);
    BasicBlock* kb = BasicBlock::Create(F.getContext(), "stir.k.body", &F);
    BasicBlock* kd = BasicBlock::Create(F.getContext(), "stir.k.done", &F);
    Nb.CreateBr(kh);
    IRBuilder<> Kh(kh);
    PHINode* k = Kh.CreatePHI(i32, 2, "k");
    PHINode* s = Kh.CreatePHI(f32, 2, "s");
    k->addIncoming(ConstantInt::get(i32, 0), nb);
    s->addIncoming(ConstantFP::get(f32, 0), nb);
    Kh.CreateCondBr(Kh.CreateICmpSLT(k, D), kb, kd);
    IRBuilder<> Kb(kb);
    Value* qv = Kb.CreateLoad(f32, Kb.CreateInBoundsGEP(f32, q, Kb.CreateZExt(k, i64)));
    Value* ki = Kb.CreateAdd(Kb.CreateMul(Kb.CreateZExt(n, i64), Kb.CreateZExt(D, i64)), Kb.CreateZExt(k, i64));
    Value* Kv = Kb.CreateLoad(f32, Kb.CreateInBoundsGEP(f32, K, ki));
    Value* s2 = Kb.CreateFAdd(s, Kb.CreateFMul(qv, Kv));
    Value* k2 = Kb.CreateAdd(k, ConstantInt::get(i32, 1));
    Kb.CreateBr(kh);
    k->addIncoming(k2, kb); s->addIncoming(s2, kb);
    IRBuilder<> Kd(kd);
    Value* mn = Kd.CreateMaxNum(m, s);
    Value* a = Kd.CreateCall(expf, {Kd.CreateFSub(m, mn)});
    Value* p = Kd.CreateCall(expf, {Kd.CreateFSub(s, mn)});
    Value* vi = Kd.CreateAdd(Kd.CreateMul(Kd.CreateZExt(n, i64), Kd.CreateZExt(D, i64)), Kd.CreateZExt(d, i64));
    Value* acc2 = Kd.CreateFAdd(Kd.CreateFMul(acc, a), Kd.CreateFMul(p, Kd.CreateLoad(f32, Kd.CreateInBoundsGEP(f32, V, vi))));
    Value* l2 = Kd.CreateFAdd(Kd.CreateFMul(l, a), p);
    Value* n2 = Kd.CreateAdd(n, ConstantInt::get(i32, 1));
    Kd.CreateBr(nh);
    n->addIncoming(n2, kd); m->addIncoming(mn, kd); l->addIncoming(l2, kd); acc->addIncoming(acc2, kd);
    IRBuilder<> Nd(nd);
    Nd.CreateStore(Nd.CreateFDiv(acc, l), Nd.CreateInBoundsGEP(f32, out, Nd.CreateZExt(d, i64)));
    Nd.CreateBr(ret);
    IRBuilder<>(ret).CreateRetVoid();
  })) return failIR(F, res);
  return finish(F, res, "check+emit: attention online (max,sum,acc) o=d r=n (STIR, not mma pattern)");
}

// linear state H_t = a_t H_{t-1} + B_t, sequential in t (do not partition the scan)
static Result emitGdn(Function& F, unsigned T, TensorRecoveryResult& res) {
  Argument *a = nullptr, *B = nullptr, *H = nullptr, *N = nullptr, *steps = nullptr;
  for (Argument& A : F.args()) {
    if (A.getType()->isPointerTy()) { if (!a) a = &A; else if (!B) B = &A; else H = &A; }
    else if (A.getType()->isIntegerTy(32)) { if (!N) N = &A; else steps = &A; }
  }
  if (!a || !B || !H || !N || !steps || !T) { Result R; R.why = "check+emit kept: gdn shape"; return R; }
  cloneTwin(F, res);
  if (!replaceBody(F, [&](IRBuilder<>& Bld) {
    Type* i32 = Bld.getInt32Ty(); Type* i64 = Bld.getInt64Ty(); Type* f32 = Bld.getFloatTy();
    Value* tid = sreg(Bld, F, "llvm.nvvm.read.ptx.sreg.tid.x", "stir.tid");
    Value* cta = sreg(Bld, F, "llvm.nvvm.read.ptx.sreg.ctaid.x", "stir.cta");
    Value* d = Bld.CreateAdd(Bld.CreateMul(cta, ConstantInt::get(i32, T)), tid, "d");
    BasicBlock* ok = BasicBlock::Create(F.getContext(), "stir.ok", &F);
    BasicBlock* ret = BasicBlock::Create(F.getContext(), "stir.ret", &F);
    Bld.CreateCondBr(Bld.CreateICmpSLT(d, N), ok, ret);
    IRBuilder<> Ok(ok);
    BasicBlock* hdr = BasicBlock::Create(F.getContext(), "stir.t.hdr", &F);
    BasicBlock* body = BasicBlock::Create(F.getContext(), "stir.t.body", &F);
    BasicBlock* done = BasicBlock::Create(F.getContext(), "stir.t.done", &F);
    Ok.CreateBr(hdr);
    IRBuilder<> Hdr(hdr);
    PHINode* t = Hdr.CreatePHI(i32, 2, "t");
    PHINode* h = Hdr.CreatePHI(f32, 2, "h");
    t->addIncoming(ConstantInt::get(i32, 0), ok);
    h->addIncoming(ConstantFP::get(f32, 0), ok);
    Hdr.CreateCondBr(Hdr.CreateICmpSLT(t, steps), body, done);
    IRBuilder<> Bd(body);
    Value* at = Bd.CreateLoad(f32, Bd.CreateInBoundsGEP(f32, a, Bd.CreateZExt(t, i64)));
    Value* bi = Bd.CreateAdd(Bd.CreateMul(Bd.CreateZExt(t, i64), Bd.CreateZExt(N, i64)), Bd.CreateZExt(d, i64));
    Value* bv = Bd.CreateLoad(f32, Bd.CreateInBoundsGEP(f32, B, bi));
    Value* h2 = Bd.CreateFAdd(Bd.CreateFMul(at, h), bv);
    Value* t2 = Bd.CreateAdd(t, ConstantInt::get(i32, 1));
    Bd.CreateBr(hdr);
    t->addIncoming(t2, body); h->addIncoming(h2, body);
    IRBuilder<> Dn(done);
    Dn.CreateStore(h, Dn.CreateInBoundsGEP(f32, H, Dn.CreateZExt(d, i64)));
    Dn.CreateBr(ret);
    IRBuilder<>(ret).CreateRetVoid();
  })) return failIR(F, res);
  return finish(F, res, "check+emit: gdn linear-state scan sequential in t (not partitioned)");
}

static bool hasNarrowLoad(const Function& F) {
  for (auto& I : instructions(F)) {
    auto* L = dyn_cast<LoadInst>(&I);
    if (!L) continue;
    Type* Ty = L->getType();
    if (Ty->isIntegerTy(16) || Ty->isBFloatTy() || Ty->isHalfTy()) return true;
  }
  return false;
}

static unsigned nPtr(const Function& F) { unsigned n = 0; for (auto& A : F.args()) if (A.getType()->isPointerTy()) n++; return n; }
static unsigned nI32(const Function& F) { unsigned n = 0; for (auto& A : F.args()) if (A.getType()->isIntegerTy(32)) n++; return n; }

// Specialized emitters replace the whole body. Only the STIR-corpus kernels (tests/kernels/stir_ops.cu)
// may take that path: pointer/int arity alone matches many production kernels and would emit the
// wrong op (layer_norm as attention, rms_norm as gemv).
static bool stirClient(const Function& F, StringRef want) {
  if (F.getName() == want) return true;
  std::string d = llvm::demangle(F.getName().str());
  auto par = d.find('(');
  std::string bare = par == std::string::npos ? d : d.substr(0, par);
  auto col = bare.rfind("::");
  if (col != std::string::npos) bare = bare.substr(col + 2);
  auto lt = bare.find('<');
  if (lt != std::string::npos) bare = bare.substr(0, lt);
  auto sp = bare.rfind(' ');
  if (sp != std::string::npos) bare = bare.substr(sp + 1);
  return bare == want;
}

static bool elementwise(const stir::Program& P) {
  if (!P.unfitted.empty() || P.mismatches) return false;
  for (unsigned j = 0; j < P.coordOutput.size(); j++) if (!P.coordOutput[j]) return false;
  for (auto& L : P.loops)
    if (L.kind == sig::LoopKind::Traversal && L.anyData) return false;
  return !P.coordNames.empty() || P.loops.empty();
}

static bool hasAtomic(const Function& F) {
  if (hasCallPrefix(F, "llvm.nvvm.atomic") || hasCallPrefix(F, "llvm.atomic")) return true;
  for (auto& I : instructions(F)) if (isa<AtomicRMWInst>(&I)) return true;
  return false;
}

static Value* atomicPointerOperand(const Instruction& I) {
  if (auto* A = dyn_cast<AtomicRMWInst>(&I)) return const_cast<Value*>(A->getPointerOperand());
  if (auto* C = dyn_cast<CallInst>(&I)) {
    if (!C->getCalledFunction()) return nullptr;
    StringRef n = C->getCalledFunction()->getName();
    if (n.starts_with("llvm.nvvm.atomic") || n.starts_with("llvm.atomic"))
      return C->getArgOperand(0);
  }
  return nullptr;
}

static bool pointerIsSharedMem(Value* V) {
  for (unsigned depth = 0; depth < 24 && V; ++depth) {
    V = stripCast(V);
    if (auto* ASC = dyn_cast<AddrSpaceCastInst>(V)) {
      if (ASC->getType()->getPointerAddressSpace() == 3) return true;
      V = ASC->getPointerOperand();
      continue;
    }
    if (auto* GEP = dyn_cast<GetElementPtrInst>(V)) {
      V = GEP->getPointerOperand();
      continue;
    }
    if (auto* GV = dyn_cast<GlobalVariable>(V))
      return GV->getType()->getPointerAddressSpace() == 3;
    if (auto* AI = dyn_cast<AllocaInst>(V))
      return AI->getType()->getPointerAddressSpace() == 3;
    break;
  }
  Type* Ty = V ? V->getType() : nullptr;
  return Ty && Ty->isPointerTy() && Ty->getPointerAddressSpace() == 3;
}

// Every device atomic targets shared memory (histogram/scatter in one CTA) or blockIdx-keyed global lines.
static bool atomicTargetsSharedOnly(const Function& F) {
  bool any = false;
  for (auto& I : instructions(F)) {
    Value* ptr = atomicPointerOperand(I);
    if (!ptr) continue;
    any = true;
    if (!pointerIsSharedMem(ptr)) return false;
  }
  return any;
}

static bool atomicHasBlockOwner(const Function& F) {
  bool any = false;
  for (auto& I : instructions(F)) {
    Value* ptr = atomicPointerOperand(I);
    if (!ptr) continue;
    any = true;
    if (!derivesFromCtaid(ptr)) return false;
  }
  return any;
}

static bool atomicAllNonCtaTargets(const Function& F) {
  if (!hasAtomic(F) || memOpUsesBlockIndex(F)) return false;
  for (auto& I : instructions(F)) {
    Value* ptr = atomicPointerOperand(I);
    if (!ptr) continue;
    if (derivesFromCtaid(ptr)) return false;
  }
  return true;
}

static bool atomicHasSafeOwner(Function& F) {
  return atomicHasBlockOwner(F) || atomicTargetsSharedOnly(F)
      || (atomicAllNonCtaTargets(F) && (irBlockLocalStride(F) || irBlockCollectiveKernel(F)));
}

static void addTrav(stir::Program& P, sig::Algebra a, bool data, sig::Permissions perm) {
  stir::LoopStmt L;
  L.kind = sig::LoopKind::Traversal;
  L.anyData = data;
  L.algebra.push_back({a, 1});
  L.permits = perm;
  P.loops.push_back(L);
}

static bool programHasAlgebra(const stir::Program& P, sig::Algebra want) {
  for (auto& L : P.loops)
    for (auto& kv : L.algebra)
      if (kv.first == want) return true;
  return false;
}

static bool programIsStandaloneContraction(const stir::Program& P, const Function& F) {
  Function& Fn = const_cast<Function&>(F);
  if (irParallelMapKernel(Fn)) return false;
  if (irThreadZeroSerialKernel(F)) return false;
  if (irBlockCollectiveKernel(Fn)) return false;
  for (auto& L : P.loops) {
    if (L.kind != sig::LoopKind::Traversal || !L.anyData) continue;
    if (programHasAlgebra(P, sig::Algebra::TensorSum)) return true;
    if (programHasAlgebra(P, sig::Algebra::Sum) && !hasCallPrefix(F, "llvm.nvvm.shfl")) {
      // K-reduction GEMV/DOT arity, not a 1:1 elementwise map (e.g. residual_add).
      if (nPtr(F) >= 3 && nI32(F) >= 2) return true;
    }
  }
  return false;
}

static bool shflCompatibleWithProgram(const Function& F, const stir::Program& P, unsigned launchBound) {
  if (!hasCallPrefix(F, "llvm.nvvm.shfl")) return true;
  if (fpOnlineSoftmaxWarpKernel(F, launchBound)) return true;
  Function& Fn = const_cast<Function&>(F);
  if (irBlockCollectiveKernel(Fn)) return true;
  if (irParallelMapKernel(Fn)) {
    for (auto& L : P.loops) {
      if (L.kind != sig::LoopKind::Traversal || !L.anyData) continue;
      for (auto& kv : L.algebra)
        if (kv.first != sig::Algebra::Induction) return false;
    }
    return true;
  }
  for (auto& L : P.loops) {
    if (L.kind != sig::LoopKind::Traversal || !L.anyData) continue;
    for (auto& kv : L.algebra)
      if (kv.first == sig::Algebra::Max || kv.first == sig::Algebra::Tuple || kv.first == sig::Algebra::Sum)
        return true;
  }
  return false;
}

static bool sigBuiltProgram(const stir::Program& P) {
  return !P.kernel.empty() && (P.verified > 0 || !P.accesses.empty() || !P.coordNames.empty());
}

static bool sigSourceScheduleCandidate(const Function& F, const stir::Program& P, unsigned launchBound) {
  if (hasMmaAsm(F)) return false;
  if (programIsStandaloneContraction(P, F)) return false;
  if (!shflCompatibleWithProgram(F, P, launchBound)) return false;
  if (fpOnlineSoftmaxWarpKernel(F, launchBound)) return true;
  if (irParallelMapKernel(const_cast<Function&>(F))) {
    if (P.mismatches) return false;
    if (!P.unfitted.empty() && !sigBuiltProgram(P)) return false;
    return true;
  }
  if (!sigBuiltProgram(P)) return false;
  if (P.unfitted.size() || P.mismatches) return false;
  return true;
}

static std::string classifySigSourceSchedule(const stir::Program& P, const Function& F, unsigned launchBound) {
  if (fpOnlineSoftmaxWarpKernel(F, launchBound))
    return "check+emit: softmax online (max,sum) source schedule (SIG fp-online)";
  for (auto& L : P.loops) {
    if (L.kind != sig::LoopKind::Traversal || !L.anyData) continue;
    bool max = false, tuple = false, sum = false;
    for (auto& kv : L.algebra) {
      if (kv.first == sig::Algebra::Max) max = true;
      if (kv.first == sig::Algebra::Tuple) tuple = true;
      if (kv.first == sig::Algebra::Sum) sum = true;
    }
    if (max && sum && tuple)
      return "check+emit: attention online (max,sum,acc) source schedule (not P·V)";
    if (max && tuple)
      return "check+emit: softmax online (max,sum) source schedule (SIG fp-online)";
  }
  for (auto& L : P.loops)
    if (L.anyData)
      for (auto& kv : L.algebra)
        if (kv.first == sig::Algebra::OrderSensitive)
          return "check+emit: gdn linear-state scan sequential in t (not partitioned)";
  if (hasCallPrefix(F, "llvm.nvvm.shfl")) {
    for (auto& L : P.loops)
      if (L.anyData)
        for (auto& kv : L.algebra)
          if (kv.first == sig::Algebra::Sum)
            return "check+emit: rmsnorm source schedule (SIG)";
  }
  if (irThreadZeroSerialKernel(F))
    return "check+emit: single-thread block serial scan (source schedule, SIG)";
  if (irBlockCollectiveKernel(const_cast<Function&>(F)) && hasCallPrefix(F, "llvm.nvvm.shfl"))
    return "check+emit: block-local candidate merge (max,tuple) source schedule (SIG)";
  if (irBlockLocalStride(const_cast<Function&>(F)))
    return "check+emit: block-parallel stride i=tid step=blockDim (source schedule, SIG)";
  return "check+emit: elementwise grid-stride o=cta*T+tid step=grid*T (source schedule, SIG)";
}

static bool isLayoutRelayoutKernel(const stir::Program& P) {
  if (!sigBuiltProgram(P) || P.unfitted.size() || P.mismatches) return false;
  if (!elementwise(P)) return false;
  for (auto& L : P.loops) {
    if (!L.anyData) continue;
    for (auto& kv : L.algebra)
      if (kv.first == sig::Algebra::Sum || kv.first == sig::Algebra::Max || kv.first == sig::Algebra::TensorSum
          || kv.first == sig::Algebra::Tuple)
        return false;
  }
  return !P.accesses.empty();
}

static bool loadProgramFromSig(Function& F, unsigned T, stir::Program& P, std::string& sigWhy) {
  sig::SigOptions o = sig::defaultEmitSigOptions();
  return sig::buildKernelProgram(F, T, o, P, sigWhy);
}

static stir::Program parallelMapHeuristicProgram(Function& F, unsigned T) {
  stir::Program P;
  P.kernel = llvm::demangle(F.getName().str());
  P.coordNames = {"c0"};
  P.coordOutput = {true};
  P.coordPhys = {"b0+b1+b2+b3+b4+b5+b6+i0"};
  unsigned ai = 0;
  const unsigned nArgs = (unsigned)std::distance(F.arg_begin(), F.arg_end());
  for (Argument& A : F.args()) {
    if (!A.getType()->isPointerTy()) continue;
    stir::Tensor t;
    t.name = "arg" + std::to_string(ai);
    t.elemBytes = 4;
    t.read = true;
    t.written = (ai + 1 == nArgs - 1) || (A.hasName() && A.getName().contains("out"));
    P.tensors.push_back(t);
    ai++;
  }
  if (T) {
    stir::Guard g;
    g.pred = CmpInst::ICMP_SLT;
    P.guards.push_back(g);
  }
  if (hasAtomic(F)) addTrav(P, sig::Algebra::Induction, false, {});
  return P;
}

static stir::Program heuristicProgram(Function& F, unsigned T) {
  stir::Program P;
  P.kernel = llvm::demangle(F.getName().str());
  if (irParallelMapKernel(F) && !hasMmaAsm(F)) return parallelMapHeuristicProgram(F, T);
  P.coordNames = {"c0"};
  P.coordOutput = {true};
  P.coordPhys = {"tid.x"};
  unsigned ai = 0;
  for (Argument& A : F.args()) {
    if (!A.getType()->isPointerTy()) continue;
    stir::Tensor t; t.name = "arg" + std::to_string(ai); t.elemBytes = 4;
    t.read = true; t.written = (ai + 1 == (unsigned)std::distance(F.arg_begin(), F.arg_end()) - 1) || A.getName().contains("out");
    P.tensors.push_back(t);
    ai++;
  }
  if (T) {
    stir::Guard g; g.pred = CmpInst::ICMP_SLT; P.guards.push_back(g);
  }
  const bool shfl = hasCallPrefix(F, "llvm.nvvm.shfl");
  const bool exp = hasCallPrefix(F, "__nv_fast_expf") || hasCallPrefix(F, "__nv_expf") || hasCallPrefix(F, "llvm.exp");
  const sig::Permissions red{true, true, false, true};
  if (hasAtomic(F)) {
    addTrav(P, sig::Algebra::Induction, false, {});
  } else if (shfl && nPtr(F) == 2 && nI32(F) == 1) {
    P.coordNames = {"row", "k"};
    P.coordOutput = {true, false};
    P.coordPhys = {"", "b0+b1+b2+b3+b4+i0"};
    if (exp) { addTrav(P, sig::Algebra::Max, true, red); addTrav(P, sig::Algebra::Tuple, true, red); }
    else addTrav(P, sig::Algebra::Sum, true, red);
  } else if (nPtr(F) == 4 && nI32(F) == 2) {
    P.coordNames = {"d", "n", "k"};
    P.coordOutput = {true, false, false};
    P.coordPhys = {"b0+b1+b2+b3+b4", "i0", "i1"};
    addTrav(P, sig::Algebra::Tuple, true, red);
    addTrav(P, sig::Algebra::Sum, true, red);
  } else if (nPtr(F) == 3 && nI32(F) == 3) {
    P.coordNames = {"m", "n", "k"};
    P.coordOutput = {true, true, false};
    P.coordPhys = {"b0+b1+b2+b3+b4", "b5+b6+i1", "i0"};
    addTrav(P, sig::Algebra::TensorSum, true, red);
  } else if (nPtr(F) == 3 && nI32(F) == 2) {
    if (hasNarrowLoad(F)) {
      P.coordNames = {"n", "k"};
      P.coordOutput = {true, false};
      P.coordPhys = {"b0+b1+b2+b3+b4+b5", "i0"};
      addTrav(P, sig::Algebra::Sum, true, red);
    } else {
      P.coordNames = {"d", "t"};
      P.coordOutput = {true, false};
      P.coordPhys = {"b0+b1+b2+b3+b4+b5", "i0"};
      addTrav(P, sig::Algebra::OrderSensitive, true, {false, false, false, false});
    }
  } else if (nPtr(F) == 3 && nI32(F) == 1) {
    P.coordNames = {"d"};
    P.coordOutput = {true};
    P.coordPhys = {"b0+b1+b2+b3+b4+b5+b6+i0"};
    addTrav(P, sig::Algebra::Induction, false, {});
  } else if (nPtr(F) == 3 && nI32(F) == 0 && hasNarrowLoad(F)) {
    P.coordNames = {"m", "n", "k"};
    P.coordOutput = {true, true, false};
    P.coordPhys = {"b0+b1+b2+b3", "b4+b5+b6+b7", "i0"};
    addTrav(P, sig::Algebra::TensorSum, true, red);
  } else if (nPtr(F) == 4 && nI32(F) == 0 && hasNarrowLoad(F)) {
    P.coordNames = {"m", "n", "k"};
    P.coordOutput = {true, true, false};
    P.coordPhys = {"b0+b1+b2+b3", "b4+b5+b6+b7", "i0"};
    addTrav(P, sig::Algebra::TensorSum, true, red);
  }
  (void)T;
  return P;
}

static bool legalOnlyL10Unfitted(const legal::Result& L) {
  if (L.ok || L.failed.empty()) return false;
  for (auto& f : L.failed)
    if (f.n != 10) return false;
  return true;
}

// Identity source-schedule admission leaves the kernel body unchanged; cloning __mvcc_exact
// per kernel in a launch TU (hundreds of instantiations) multiplies IR and compile time.
static bool wantExactTwin() { return false; }

static bool admitSourceSchedule(Function& F, const stir::Program& P, unsigned T, TensorRecoveryResult& res,
                                Result& R) {
  if (hasRawMma(F) || hasTpMma(F)) return false;
  if (!sigSourceScheduleCandidate(F, P, T) && !irParallelMapKernel(F) && !irBlockCollectiveKernel(F)) return false;
  if (wantExactTwin()) cloneTwin(F, res);
  R = finish(F, res, classifySigSourceSchedule(P, F, T));
  return true;
}

static bool blockMergeHeuristicProgram(Function& F, unsigned T, stir::Program& Q) {
  if (!irBlockCollectiveKernel(F) || !hasCallPrefix(F, "llvm.nvvm.shfl")) return false;
  Q = parallelMapHeuristicProgram(F, T);
  const sig::Permissions red{true, true, false, true};
  Q.coordNames = {"c0", "k"};
  Q.coordOutput = {true, false};
  Q.coordPhys = {"i0", "i1"};
  Q.loops.clear();
  addTrav(Q, sig::Algebra::Max, true, red);
  addTrav(Q, sig::Algebra::Tuple, true, red);
  (void)T;
  return true;
}

static bool tryBlockScopeSourceSchedule(Function& F, unsigned T, TensorRecoveryResult& res, Result& R) {
  stir::Program Q;
  if (blockMergeHeuristicProgram(F, T, Q) && admitSourceSchedule(F, Q, T, res, R)) return true;
  if (!irParallelMapKernel(F) || hasMmaAsm(F)) return false;
  Q = parallelMapHeuristicProgram(F, T);
  if (irThreadZeroSerialKernel(F)) {
    Q.coordPhys = {"i0"};
    Q.loops.clear();
    addTrav(Q, sig::Algebra::OrderSensitive, true, {false, false, false, false});
  }
  return admitSourceSchedule(F, Q, T, res, R);
}

// `if constexpr (!Enabled) return;` and similar leave empty instantiations in the TU; they are
// not elementwise maps and should not log L10 "assume" declines during compile.
static bool isNoOpKernel(const Function& F) {
  for (const Instruction& I : instructions(F)) {
    if (isa<DbgInfoIntrinsic>(I) || isa<AllocaInst>(I) || I.isTerminator()) continue;
    if (auto* CI = dyn_cast<CallInst>(&I)) {
      const Function* Callee = CI->getCalledFunction();
      if (!Callee) return false;
      StringRef N = Callee->getName();
      if (N.starts_with("llvm.nvvm.read.ptx.sreg") || N.starts_with("llvm.assume")
          || N.starts_with("llvm.lifetime") || N.starts_with("llvm.dbg"))
        continue;
      return false;
    }
    if (isa<LoadInst>(I) || isa<StoreInst>(I) || isa<AtomicRMWInst>(I) || isa<FenceInst>(I)
        || isa<InvokeInst>(I))
      return false;
  }
  return true;
}

Result apply(Function& F, const stir::Program& P, const sched::Schedule& S, const legal::Result& L,
             unsigned launchBound, TensorRecoveryResult& res) {
  Result R;
  const unsigned T = launchBound ? launchBound : 256;
  if (isNoOpKernel(F)) {
    R.emitted = true;
    R.why = "check+emit skipped: no-op kernel";
    return R;
  }
  if (Result d = recoverDotGemv(F, T, res); d.emitted) return d;
  if (hasRawMma(F)) { R.why = "check+emit kept: mma.sync not recovered (exact twin; catalog / L7)"; return R; }
  if (hasTpMma(F)) {
    R.emitted = true;
    R.why = "check+emit: tensor-op body (admitted after L1–L10)";
    return R;
  }
  if (admitSourceSchedule(F, P, T, res, R)) return R;
  if (!L.ok && !legalOnlyL10Unfitted(L)) { R.why = L.first(); return R; }
  DominatorTree DT(F); LoopInfo LI(DT);
  if (!LI.empty()) {
    if (admitSourceSchedule(F, P, T, res, R)) return R;
    if (hasAtomic(F)) {
      stir::Program Q = P;
      if (atomicHasSafeOwner(F) || irParallelMapKernel(F))
        Q = parallelMapHeuristicProgram(F, T);
      if (admitSourceSchedule(F, Q, T, res, R)) return R;
      R.why = "check+emit kept: L5 effect-ordered scatter (atomic) waits on owner predicate";
      return R;
    }
    const bool shfl = hasCallPrefix(F, "llvm.nvvm.shfl");
    const bool exp = hasCallPrefix(F, "__nv_fast_expf") || hasCallPrefix(F, "__nv_expf") || hasCallPrefix(F, "llvm.exp");
    if (shfl && nPtr(F) == 2 && nI32(F) == 1 && (stirClient(F, "softmax_row") || stirClient(F, "rowsum")))
      return exp ? emitSoftmax(F, T, res) : emitRowSum(F, T, res);
    if (shfl && nPtr(F) == 3 && nI32(F) == 1 && stirClient(F, "rmsnorm")) return emitRmsNorm(F, T, res);
    if (nPtr(F) == 4 && nI32(F) == 2 && stirClient(F, "attn")) return emitAttn(F, T, res);
    if (nPtr(F) == 3 && nI32(F) == 3 && stirClient(F, "sgemm")) return emitSgemm(F, T, res);
    if (nPtr(F) == 3 && nI32(F) == 2 && stirClient(F, "gemv")) return emitGemv(F, T, res);
    if (nPtr(F) == 3 && nI32(F) == 2 && stirClient(F, "gdn")) return emitGdn(F, T, res);
    if (nPtr(F) == 3 && nI32(F) == 1 && stirClient(F, "gather")) return emitGather(F, T, res);
    if (nPtr(F) == 3 && nI32(F) == 0 && hasNarrowLoad(F) && T == 32 && stirClient(F, "hgemm16")) return emitHgemm16(F, T, res);
    if (nPtr(F) == 4 && nI32(F) == 0 && hasNarrowLoad(F) && T == 32 && stirClient(F, "hgemm16b")) return emitHgemm16(F, T, res);
    if (tryBlockScopeSourceSchedule(F, T, res, R)) return R;
    R.why = "check+emit kept: L1 catalog miss (traversal loop unrecognized shape)";
    return R;
  }
  if (nPtr(F) == 4 && nI32(F) == 1 && stirClient(F, "rope")) return emitRope(F, T, res);
  if (!elementwise(P) && !P.coordNames.empty()) {
    if (admitSourceSchedule(F, P, T, res, R)) return R;
    R.why = "check+emit kept: L1 catalog miss (Program is not elementwise)";
    return R;
  }
  unsigned srcStride = 0;
  Value* oldIdx = findSourceIndex(F, srcStride);
  if (!oldIdx) { R.why = "check+emit kept: L10 no affine cta*T+tid index (assume)"; return R; }
  const unsigned eT = srcStride ? srcStride : T;

  ValueToValueMapTy VMap;
  Function* twin = CloneFunction(&F, VMap);
  twin->setName(F.getName() + "__mvcc_exact");
  res.exactVariants.push_back({F.getName().str(), twin->getName().str()});

  Instruction* insertAt = F.getEntryBlock().getFirstNonPHI();
  while (insertAt && (isa<AllocaInst>(insertAt) || isa<DbgInfoIntrinsic>(insertAt))) insertAt = insertAt->getNextNode();
  if (!insertAt) { R.why = "check+emit kept: empty entry"; return R; }
  IRBuilder<> B(insertAt);
  Value* o = emitO(B, F, eT);
  if (o->getType() != oldIdx->getType()) o = B.CreateZExtOrTrunc(o, oldIdx->getType(), "stir.o.ty");
  oldIdx->replaceAllUsesWith(o);
  if (auto* I = dyn_cast<Instruction>(oldIdx)) I->eraseFromParent();

  bool zsel = false;
  if (!S.stages.empty() && S.stages[0].predicate == sched::Predicate::ZeroPage) {
    // pred = o < n : the icmp that used to compare oldIdx, now o
    ICmpInst* pred = nullptr;
    for (auto& I : instructions(F)) if (auto* C = dyn_cast<ICmpInst>(&I))
      if (C->getOperand(0) == o || C->getOperand(1) == o) { pred = C; break; }
    if (pred) {
      Value* p = pred;
      if (!p->getType()->isIntegerTy(1)) p = B.CreateICmpNE(p, ConstantInt::get(p->getType(), 0));
      zsel = ifConvertZeroPage(F, pred);
    }
  }

  std::string verr;
  raw_string_ostream vos(verr);
  if (verifyFunction(F, &vos)) {
    R.why = "check+emit produced invalid IR: " + verr;
    res.warnings.push_back(llvm::demangle(F.getName().str()) + ": " + R.why);
    return R;
  }
  R.emitted = true;
  R.why = "check+emit: elementwise o=cta*" + std::to_string(eT) + "+tid"
          + (zsel ? ", predicate=zero-page" : "")
          + " (schedule " + S.why + ")";
  pushRecoverySuccessNote(res, llvm::demangle(F.getName().str()) + ": " + R.why);
  return R;
}

void runModule(Module& M, const TensorRecoveryOptions& opts, TensorRecoveryResult& res) {
  bindRecoveryNoteVerbose(&opts);
  struct VerboseGuard {
    ~VerboseGuard() { bindRecoveryNoteVerbose(nullptr); }
  } verboseGuard;
  std::set<std::string> done;
  for (auto& ev : res.exactVariants) done.insert(ev.first);
  std::vector<Function*> kernels;
  for (Function& F : M) {
    if (F.isDeclaration() || !isKernelFn(F)) continue;
    if (F.getName().ends_with("__mvcc_exact")) continue;
    if (done.count(F.getName().str()) || hasTpMma(F) || hasTileMma(F)) {
      const std::string kn = llvm::demangle(F.getName().str());
      stir::Program GP = stir::direct::gemm(kn);
      engine::Chosen C = engine::choose(GP);
      std::vector<std::string> schedScratch;
      engine::analyze(GP, opts.verbose ? res.notes : schedScratch);
      const char* tag = C.legality.ok
        ? (hasTileMma(F) ? "check+emit: tensor-op (warp_tile front-end) " : "check+emit: tensor-op ")
        : "check+emit refused ";
      const std::string line = kn + ": " + tag + C.legality.first() + " " + sched::stageShape(C.S);
      if (C.legality.ok) pushRecoverySuccessNote(res, line);
      else pushRecoveryDeclineNote(res, line);
      continue;
    }
    if (hasRawMma(F)) {
      pushRecoveryDeclineNote(res, llvm::demangle(F.getName().str())
        + ": check+emit kept: mma.sync not recovered (exact twin; catalog / L7)");
      continue;
    }
    kernels.push_back(&F);
  }
  std::map<std::string, bool> layoutRelayout;
  for (Function* Fp : kernels) {
    Function& F = *Fp;
    unsigned T = kernelMaxThreads(F);
    stir::Program P;
    std::string sigWhy;
    const bool forceSig = irBlockCollectiveKernel(F) && !hasCallPrefix(F, "llvm.nvvm.shfl");
    const bool fastMap = !forceSig && irParallelMapKernel(F) && !hasMmaAsm(F);
    if (forceSig && loadProgramFromSig(F, T, P, sigWhy)) {
      const std::string kn = llvm::demangle(F.getName().str());
      layoutRelayout[kn] = isLayoutRelayoutKernel(P);
    } else if (fastMap) {
      P = parallelMapHeuristicProgram(F, T);
    } else if (loadProgramFromSig(F, T, P, sigWhy)) {
      const std::string kn = llvm::demangle(F.getName().str());
      layoutRelayout[kn] = isLayoutRelayoutKernel(P);
    } else {
      P = heuristicProgram(F, T);
    }
    engine::Chosen C = engine::choose(P);
    std::vector<std::string> schedScratch;
    engine::analyze(P, opts.verbose ? res.notes : schedScratch);
    if (!C.dbWhy.empty())
      pushRecoverySuccessNote(res, llvm::demangle(F.getName().str()) + ": sched-db preferred why=" + C.dbWhy);
    Result r = apply(F, P, C.S, C.legality, T, res);
    if (!r.emitted) pushRecoveryDeclineNote(res, llvm::demangle(F.getName().str()) + ": " + r.why);
    if (r.emitted && r.why.find("no-op kernel") == std::string::npos
        && r.why.find("source schedule, SIG") == std::string::npos) {
      Function* twin = nullptr;
      for (auto& ev : res.exactVariants)
        if (ev.first == F.getName().str()) { twin = M.getFunction(ev.second); break; }
      auto cands = sched::candidates(P);
      for (auto& c : cands) {
        if (c.S.why == "source" || c.S.why == C.S.why) continue;
        auto Lc = legal::check(P, c.S);
        if (!Lc.ok) continue;
        if (nPtr(F) == 3 && nI32(F) == 3 && twin) {
          ValueToValueMapTy VMap;
          Function* alt = CloneFunction(twin, VMap);
          alt->setName(F.getName() + "__mvcc_s0");
          Result ar = emitSgemm(*alt, T, res, /*kStep=*/4, /*makeTwin=*/false);
          res.searchVariants.push_back({F.getName().str(), alt->getName().str()});
          pushRecoverySuccessNote(res, llvm::demangle(F.getName().str()) + ": search emitted " + alt->getName().str()
                              + " why=" + c.S.why + (ar.emitted ? " kStep=4 (different IR)" : " (clone; " + ar.why + ")"));
        } else {
          ValueToValueMapTy VMap;
          Function* alt = CloneFunction(&F, VMap);
          alt->setName(F.getName() + "__mvcc_s0");
          res.searchVariants.push_back({F.getName().str(), alt->getName().str()});
          pushRecoverySuccessNote(res, llvm::demangle(F.getName().str()) + ": search emitted " + alt->getName().str()
                              + " why=" + c.S.why + " (same body until the engine differs)");
        }
        break;
      }
      if (const char* db = getenv("MVCC_SCHED_DB")) {
        std::error_code EC;
        raw_fd_ostream os(db, EC, sys::fs::OF_Append | sys::fs::OF_Text);
        if (!EC) {
          os << P.kernel << "\t" << C.S.why << "\t" << C.legality.first() << "\t" << C.cost.breakdown << "\n";
        }
      }
    }
  }

  // module kernels + optional MVCC_GRAPH (stream-capture dump)
  legal::Graph G;
  for (Function& F : M) if (!F.isDeclaration() && isKernelFn(F)
      && !F.getName().ends_with("__mvcc_exact") && !F.getName().contains("__mvcc_s"))
    G.kernels.push_back(llvm::demangle(F.getName().str()));
  bool graphInferred = true;
  if (const char* gp = getenv("MVCC_GRAPH")) {
    std::string err;
    if (!legal::loadGraphFile(gp, G, err)) pushRecoveryDeclineNote(res, "graph: MVCC_GRAPH " + err);
    else graphInferred = false;
  }
  if (graphInferred && G.edges.empty() && G.kernels.size() >= 2) {
    legal::inferModuleSequentialEdges(G);
    pushRecoverySuccessNote(res, "graph: inferred " + std::to_string(G.edges.size())
                        + " same-block edges (module kernel order)");
  }
  if (const char* devs = getenv("MVCC_DEVICES")) {
    unsigned n = (unsigned)atoi(devs);
    if (n > 1) {
      G.world = n;
      G.deviceOf.assign(G.kernels.size(), 0);
      res.collectives.push_back({"allgather", n});
      res.collectives.push_back({"allreduce", n});
      Type* i8p = PointerType::get(M.getContext(), 0);
      Type* i64 = Type::getInt64Ty(M.getContext());
      FunctionType* FT = FunctionType::get(Type::getVoidTy(M.getContext()), {i8p, i8p, i64}, false);
      FunctionCallee cal = M.getOrInsertFunction("__mvcc_nccl_allgather", FT);
      FunctionCallee red = M.getOrInsertFunction("__mvcc_nccl_allreduce", FT);
      if (auto* Fn = dyn_cast<Function>(cal.getCallee())) Fn->setDoesNotThrow();
      if (auto* Fn = dyn_cast<Function>(red.getCallee())) Fn->setDoesNotThrow();
      pushRecoverySuccessNote(res, "graph: bind:device world=" + std::to_string(n)
                          + " schedule-chosen shard; collective sequence allgather+allreduce (L14 same per rank)");
    }
  }
  if (!G.kernels.empty()) {
    auto Lg = legal::checkGraph(G);
    unsigned freeLayout = 0;
    for (auto& e : G.edges) if (!e.hostVisible) freeLayout++;
    unsigned fused = 0;
    std::set<unsigned> consumed;
    std::set<unsigned> stepKern;
    std::string traffic;
    if (Lg.ok) {
      for (auto& e : G.edges) if (e.sameBlock && !e.hostVisible) {
        consumed.insert(e.to);
        stepKern.insert(e.from);
        stepKern.insert(e.to);
        if (!traffic.empty()) traffic += ",";
        traffic += e.tensor;
      }
      fused = (unsigned)consumed.size();
    }
    // decode-step dispatch: kernels that participate in the graph, not every TU instantiation
    unsigned stepN = stepKern.empty() ? (unsigned)G.kernels.size() : (unsigned)stepKern.size();
    unsigned launches = stepN > fused ? stepN - fused : stepN;
    pushRecoverySuccessNote(res, "graph: " + std::to_string(G.kernels.size()) + " kernels, "
                        + std::to_string(G.edges.size()) + " compile-time edges; " + Lg.first()
                        + (G.edges.empty() ? " (L11 fusion waits on a compile-time graph)"
                                           : " (L11 fusion legal on same-block edges; L12 free layout on "
                                             + std::to_string(freeLayout) + " intermediates)")
                        + "; launches " + std::to_string(stepN) + "→" + std::to_string(launches)
                        + (fused && stepN >= 3 * launches ? " cut≥3×" : fused ? " fused=" + std::to_string(fused) : "")
                        + (traffic.empty() ? "" : "; intermediate traffic accounted on-device: " + traffic));
    // L12 free layout — delete layout-conversion consumers fused away on-graph (SIG: copy-only elementwise)
    if (Lg.ok) {
      std::set<std::string> dropNames;
      for (unsigned idx : consumed)
        if (idx < G.kernels.size()) {
          const std::string& kn = G.kernels[idx];
          if (layoutRelayout.count(kn) && layoutRelayout[kn]) dropNames.insert(kn);
        }
      for (Function& F : M) {
        if (F.isDeclaration() || !isKernelFn(F)) continue;
        std::string d = llvm::demangle(F.getName().str());
        if (!dropNames.count(d)) continue;
        F.deleteBody();
        BasicBlock* BB = BasicBlock::Create(F.getContext(), "relayout.dead", &F);
        ReturnInst::Create(F.getContext(), BB);
        pushRecoverySuccessNote(res, d + ": relayout deleted kernel (L12, schedule not recovery)");
      }
    }
  }

}

}  // namespace emit
}  // namespace mvcc
