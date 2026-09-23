// Guarded-load if-conversion (declared in staging_elim.h).
#include "staging_elim.h"
#include "ptx_asm.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include <set>

using namespace llvm;

namespace mvcc {
namespace {

struct Copy {
  CallInst* CI = nullptr;
  Value* dst = nullptr;   // shared address operand (i32 or pointer)
  Value* src = nullptr;   // device pointer operand
  bool valid = true;      // false: src-size 0, the copy writes zeros
  unsigned bytes = 0;
};

struct Arm {
  std::vector<Instruction*> insts;   // everything but the terminator, in order
  std::vector<LoadInst*> loads;
  std::vector<CallInst*> mmas;       // mma.sync asm in the arm
  Copy copy;
};

// One arm of a guard: only speculatable arithmetic, device loads and at most one cp.async.
bool analyzeArm(BasicBlock* BB, Arm& arm, std::string& why) {
  unsigned n = 0;
  for (Instruction& I : *BB) {
    if (I.isTerminator()) break;
    if (++n > 128) { why = "arm too large"; return false; }
    if (isa<PHINode>(I)) { why = "phi in arm"; return false; }
    if (auto* L = dyn_cast<LoadInst>(&I)) {
      if (!L->isSimple()) { why = "volatile or atomic load"; return false; }
      unsigned as = L->getPointerAddressSpace();
      if (as != 0 && as != 1) { why = "load from non-device address space"; return false; }
      const Value* obj = getUnderlyingObject(L->getPointerOperand());
      if (isa<GlobalVariable>(obj) || isa<AllocaInst>(obj)) { why = "load from a global or private object"; return false; }
      if (auto* A = dyn_cast<Argument>(obj)) if (A->hasByValAttr()) { why = "load from a by-value kernel parameter (thread memory)"; return false; }
      arm.loads.push_back(L); arm.insts.push_back(&I);
      continue;
    }
    if (auto* CI = dyn_cast<CallInst>(&I)) {
      if (!CI->isInlineAsm()) {
        // pure intrinsics (tid/lane reads, math) are speculatable like arithmetic
        if (CI->getCalledFunction() && CI->getCalledFunction()->isIntrinsic() && !CI->mayReadOrWriteMemory() && !CI->mayHaveSideEffects() && !isa<DbgInfoIntrinsic>(CI)) { arm.insts.push_back(&I); continue; }
        why = "call in arm"; return false;
      }
      std::vector<PtxInstr> ins; std::string err;
      if (!parsePtxAsm(std::string(cast<InlineAsm>(CI->getCalledOperand())->getAsmString()), ins, err) || ins.empty()) { why = "asm in arm"; return false; }
      // pure register asm: per-thread ALU, and mma.sync (no memory, no side effects; executing it for a whole
      // simdgroup and discarding the result is how a skipped warp-uniform mma behaves anyway)
      if (!cast<InlineAsm>(CI->getCalledOperand())->hasSideEffects() && !CI->mayReadOrWriteMemory()) {
        bool pure = true;
        for (auto& in : ins) if (in.predicated() || !(in.mnemonic == "mma" || ptxIsPerThreadAlu(in.mnemonic))) pure = false;
        if (pure) { if (ins[0].mnemonic == "mma") arm.mmas.push_back(CI); arm.insts.push_back(&I); continue; }
      }
      if (ins.size() != 1) { why = "asm in arm"; return false; }
      const PtxInstr& in = ins[0];
      if (in.mnemonic != "cp" || in.mods.empty() || in.mods[0] != "async" || !in.hasMod("shared") || !in.hasMod("global") || in.hasMod("bulk") || in.predicated()) { why = "asm in arm"; return false; }
      if (arm.copy.CI) { why = "two copies in one arm"; return false; }
      if (in.ops.size() < 3 || in.ops.size() > 4 || in.ops[2].kind != PtxOperand::Imm) { why = "cp.async operand form"; return false; }
      long long bytes = in.ops[2].imm;
      if (bytes != 4 && bytes != 8 && bytes != 16) { why = "cp.async size"; return false; }
      bool valid = true;
      if (in.ops.size() == 4) {
        long long sz = -1;
        if (in.ops[3].kind == PtxOperand::Imm) sz = in.ops[3].imm;
        else if (in.ops[3].kind == PtxOperand::Reg && in.ops[3].reg >= 0 && (unsigned)in.ops[3].reg < CI->arg_size())
          if (auto* K = dyn_cast<ConstantInt>(CI->getArgOperand(in.ops[3].reg))) sz = K->getSExtValue();
        if (sz != 0 && sz != bytes) { why = "cp.async with a variable or partial src-size"; return false; }
        valid = sz == bytes;
      }
      if (in.ops[0].kind != PtxOperand::Reg && in.ops[0].kind != PtxOperand::Mem) { why = "cp.async destination form"; return false; }
      if (in.ops[1].kind != PtxOperand::Reg && in.ops[1].kind != PtxOperand::Mem) { why = "cp.async source form"; return false; }
      int d = in.ops[0].reg, s = in.ops[1].reg;
      if (d < 0 || s < 0 || (unsigned)d >= CI->arg_size() || (unsigned)s >= CI->arg_size()) { why = "cp.async operand indices"; return false; }
      if (!CI->getArgOperand(s)->getType()->isPointerTy()) { why = "cp.async source is not a pointer"; return false; }
      arm.copy = {CI, CI->getArgOperand(d), CI->getArgOperand(s), valid, (unsigned)bytes};
      arm.insts.push_back(&I);
      continue;
    }
    if (I.mayReadOrWriteMemory() || I.mayHaveSideEffects() || !isSafeToSpeculativelyExecute(&I)) { why = "non-speculatable instruction in arm"; return false; }
    arm.insts.push_back(&I);
  }
  return true;
}

struct Converter {
  Function& F;
  GuardedLoadStats& st;
  std::vector<std::string>* notes;
  CallInst* zeroPage = nullptr;
  std::set<BasicBlock*> dead;

  Value* zero(Type* ptrTy, Instruction* before) {
    if (!zeroPage) {
      Module& M = *F.getParent();
      FunctionCallee ZF = M.getOrInsertFunction("__mvcc_zero_page", PointerType::get(F.getContext(), 0));
      if (auto* Fn = dyn_cast<Function>(ZF.getCallee())) { Fn->setMemoryEffects(MemoryEffects::none()); Fn->addFnAttr(Attribute::NoUnwind); Fn->addFnAttr(Attribute::WillReturn); }
      IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
      zeroPage = B.CreateCall(ZF, {}, "zero_page");
    }
    if (zeroPage->getType() == ptrTy) return zeroPage;
    return new AddrSpaceCastInst(zeroPage, ptrTy, "", before->getIterator());
  }

  // Moves the arm's instructions in front of `at`; loads are redirected through the zero page when !cond (taken=true)
  // or when cond (taken=false).
  void hoist(Arm& arm, Value* cond, bool taken, Instruction* at) {
    for (Instruction* I : arm.insts) {
      if (I == arm.copy.CI) continue;  // merged separately
      I->moveBefore(at->getIterator());
      if (auto* L = dyn_cast<LoadInst>(I)) {
        Value* p = L->getPointerOperand();
        Value* z = zero(p->getType(), L);
        IRBuilder<> B(L);
        L->setOperand(0, taken ? B.CreateSelect(cond, p, z) : B.CreateSelect(cond, z, p));
        // value metadata that the zeros may violate
        L->setMetadata(LLVMContext::MD_range, nullptr); L->setMetadata(LLVMContext::MD_nonnull, nullptr);
        L->setMetadata(LLVMContext::MD_align, nullptr); L->setMetadata(LLVMContext::MD_dereferenceable, nullptr);
        L->setMetadata(LLVMContext::MD_dereferenceable_or_null, nullptr); L->setMetadata(LLVMContext::MD_noundef, nullptr);
        st.loads++;
      }
    }
  }

  // Identical computations of two arms (both arms usually recompute the same shared destination) collapse.
  void cse(Arm& keep, Arm& drop) {
    for (Instruction* I : drop.insts) {
      if (I == drop.copy.CI || isa<LoadInst>(I)) continue;
      for (Instruction* J : keep.insts) {
        if (J == keep.copy.CI || isa<LoadInst>(J) || !I->isIdenticalTo(J)) continue;
        I->replaceAllUsesWith(J); I->eraseFromParent(); break;
      }
    }
  }

  void mergeCopies(Copy& t, Copy& f, Value* cond, Instruction* at) {
    IRBuilder<> B(at);
    Value* dst = t.dst == f.dst ? t.dst : B.CreateSelect(cond, t.dst, f.dst);
    Value* src = t.src == f.src ? t.src : B.CreateSelect(cond, t.src, f.src);
    Value* valid = t.valid && f.valid ? ConstantInt::getTrue(F.getContext()) : t.valid ? cond : B.CreateNot(cond);
    Module& M = *F.getParent();
    FunctionCallee ZS = M.getOrInsertFunction("__mvcc_cp_async_zsel_" + std::to_string(t.bytes), FunctionType::get(Type::getVoidTy(F.getContext()), {dst->getType(), src->getType(), Type::getInt1Ty(F.getContext())}, false));
    if (auto* Fn = dyn_cast<Function>(ZS.getCallee())) { Fn->addFnAttr(Attribute::NoUnwind); Fn->addFnAttr(Attribute::Convergent); }
    CallInst* NC = B.CreateCall(ZS, {dst, src, valid});
    NC->setDebugLoc(t.CI->getDebugLoc());
    t.CI->eraseFromParent(); f.CI->eraseFromParent();
    st.copies++;
  }

  // Tries one branch block. Returns true if the CFG changed.
  bool tryBranch(BasicBlock* P) {
    auto* BI = dyn_cast<BranchInst>(P->getTerminator());
    if (!BI || !BI->isConditional() || isa<Constant>(BI->getCondition())) return false;
    BasicBlock* S0 = BI->getSuccessor(0); BasicBlock* S1 = BI->getSuccessor(1);
    if (S0 == S1 || S0 == P || S1 == P) return false;
    auto isArm = [&](BasicBlock* A, BasicBlock*& J) {
      if (A->getSinglePredecessor() != P || A->hasAddressTaken()) return false;
      auto* T = dyn_cast<BranchInst>(A->getTerminator());
      if (!T || T->isConditional()) return false;
      J = T->getSuccessor(0);
      return J != A && J != P;
    };
    BasicBlock *J0 = nullptr, *J1 = nullptr;
    bool a0 = isArm(S0, J0), a1 = isArm(S1, J1);
    Value* cond = BI->getCondition();
    std::string why;
    if (a0 && a1 && J0 == J1) {
      // diamond
      Arm T, Fm;
      if (!analyzeArm(S0, T, why) || !analyzeArm(S1, Fm, why)) return false;
      if ((T.copy.CI != nullptr) != (Fm.copy.CI != nullptr)) return false;   // a copy in only one arm stays guarded
      if (T.copy.CI && (T.copy.bytes != Fm.copy.bytes || (!T.copy.valid && !Fm.copy.valid) || T.copy.src->getType() != Fm.copy.src->getType() || T.copy.dst->getType() != Fm.copy.dst->getType())) return false;
      BasicBlock* J = J0;
      Instruction* at = BI;
      hoist(T, cond, true, at); hoist(Fm, cond, false, at);
      cse(T, Fm);
      if (T.copy.CI) mergeCopies(T.copy, Fm.copy, cond, at);
      IRBuilder<> B(at);
      for (PHINode& PN : J->phis()) {
        Value* vt = PN.getIncomingValueForBlock(S0); Value* vf = PN.getIncomingValueForBlock(S1);
        Value* sel = vt == vf ? vt : B.CreateSelect(cond, vt, vf);
        PN.addIncoming(sel, P);  // the arm entries go with the arms (DeleteDeadBlock)
      }
      BranchInst::Create(J, BI->getIterator());
      BI->eraseFromParent();
      dead.insert(S0); dead.insert(S1); DeleteDeadBlock(S0); DeleteDeadBlock(S1);
      st.diamonds++;
      return true;
    }
    // triangle: one arm, the other edge goes straight to the join
    BasicBlock* A = nullptr; BasicBlock* J = nullptr; bool taken = false;
    if (a0 && J0 == S1) { A = S0; J = S1; taken = true; }
    else if (a1 && J1 == S0) { A = S1; J = S0; taken = false; }
    else return false;
    Arm arm;
    if (!analyzeArm(A, arm, why) || arm.copy.CI) return false;
    Instruction* at = BI;
    hoist(arm, cond, taken, at);
    IRBuilder<> B(at);
    // `if (p) acc = mma(a, b, acc)`: the join phis take the mma's D on the arm and its own C otherwise. Selecting the
    // results would cut the accumulator chain tensor recovery keys on, so instead the operands are zeroed when the
    // arm is skipped: acc + sum(0 * 0) = acc (the sign of a negative zero excepted), and the phi is the D itself.
    std::set<PHINode*> settled;
    for (CallInst* MC : arm.mmas) {
      if (MC->arg_size() != 10 || !MC->getType()->isStructTy() || MC->getType()->getStructNumElements() != 4) continue;
      PHINode* pn[4] = {nullptr, nullptr, nullptr, nullptr};
      bool match = true;
      for (User* U : MC->users()) {
        auto* EV = dyn_cast<ExtractValueInst>(U);
        if (!EV || EV->getNumIndices() != 1 || EV->getIndices()[0] > 3) { match = false; break; }
        unsigned i = EV->getIndices()[0];
        for (User* V : EV->users()) {
          auto* PN = dyn_cast<PHINode>(V);
          if (!PN || PN->getParent() != J || PN->getIncomingValueForBlock(A) != EV || PN->getIncomingValueForBlock(P) != MC->getArgOperand(6 + i) || (pn[i] && pn[i] != PN)) { match = false; break; }
          pn[i] = PN;
        }
        if (!match) break;
      }
      if (!match) continue;
      IRBuilder<> Bm(MC);
      for (unsigned k = 0; k < 6; k++) {
        Value* w = MC->getArgOperand(k); Value* z = Constant::getNullValue(w->getType());
        MC->setArgOperand(k, taken ? Bm.CreateSelect(cond, w, z) : Bm.CreateSelect(cond, z, w));
      }
      for (unsigned i = 0; i < 4; i++) if (pn[i]) { pn[i]->setIncomingValueForBlock(P, pn[i]->getIncomingValueForBlock(A)); settled.insert(pn[i]); }
      st.mmas++;
    }
    for (PHINode& PN : J->phis()) {
      if (settled.count(&PN)) continue;
      Value* va = PN.getIncomingValueForBlock(A); Value* vp = PN.getIncomingValueForBlock(P);
      Value* sel = va == vp ? va : taken ? B.CreateSelect(cond, va, vp) : B.CreateSelect(cond, vp, va);
      PN.setIncomingValueForBlock(P, sel);  // the arm entry goes with the arm (DeleteDeadBlock)
    }
    BranchInst::Create(J, BI->getIterator());
    BI->eraseFromParent();
    dead.insert(A); DeleteDeadBlock(A);
    st.diamonds++;
    return true;
  }
};

}  // namespace

// The ignore-src form of cp.async (PTX ISA: `cp.async.cg.shared.global [dst], [src], 16, ignore-src`, a predicate
// register set by `setp` in the same asm block) is one copy that reads its source when the predicate is false and
// writes zeros when it is true. It is the same thing the if-conversion above produces from two guarded copies, so
// it is rewritten to the merged call directly: __mvcc_cp_async_zsel_N(dst, src, valid = !pred). The staging proofs
// then see one unconditional copy per chunk, as with the src-size form.
//
//   { .reg .pred p; setp.ne.b32 p, %2, 0; cp.async.cg.shared.global[.L2::128B] [%0], [%1], 16, p; }
unsigned lowerPredicatedCopies(Function& F, std::vector<std::string>* notes) {
  std::vector<CallInst*> calls;
  for (BasicBlock& BB : F) for (Instruction& I : BB) if (auto* CI = dyn_cast<CallInst>(&I)) if (CI->isInlineAsm() && CI->getType()->isVoidTy()) calls.push_back(CI);
  unsigned n = 0;
  for (CallInst* CI : calls) {
    std::vector<PtxInstr> ins; std::string err;
    if (!parsePtxAsm(std::string(cast<InlineAsm>(CI->getCalledOperand())->getAsmString()), ins, err) || ins.size() != 2) continue;
    const PtxInstr* sp = nullptr; const PtxInstr* cp = nullptr;
    for (auto& in : ins) { if (in.mnemonic == "setp") sp = &in; else if (in.mnemonic == "cp") cp = &in; }
    if (!sp || !cp || sp->predicated() || cp->predicated()) continue;
    if (cp->mods.empty() || cp->mods[0] != "async" || !cp->hasMod("shared") || !cp->hasMod("global") || cp->hasMod("bulk")) continue;
    if (cp->ops.size() != 4 || cp->ops[2].kind != PtxOperand::Imm || cp->ops[3].kind != PtxOperand::Named) continue;
    const long long bytes = cp->ops[2].imm;
    if (bytes != 4 && bytes != 8 && bytes != 16) continue;
    // setp.<cmp>.<type> p, a, b with p the copy's predicate; a and b registers or immediates
    if (sp->ops.size() != 3 || sp->ops[0].kind != PtxOperand::Named || sp->ops[0].name != cp->ops[3].name || sp->mods.size() < 2) continue;
    const std::string cmp = sp->mods[0];
    if (cmp != "eq" && cmp != "ne" && cmp != "lt" && cmp != "le" && cmp != "gt" && cmp != "ge") continue;
    const std::string& ty = sp->mods[1];
    const bool isSigned = ty[0] == 's';
    if (ty != "b32" && ty != "u32" && ty != "s32" && ty != "b64" && ty != "u64" && ty != "s64") continue;
    if (cp->ops[0].kind != PtxOperand::Reg && cp->ops[0].kind != PtxOperand::Mem) continue;
    if (cp->ops[1].kind != PtxOperand::Reg && cp->ops[1].kind != PtxOperand::Mem) continue;
    auto argOf = [&](const PtxOperand& o) -> Value* {
      if (o.kind == PtxOperand::Reg || o.kind == PtxOperand::Mem) return o.reg >= 0 && (unsigned)o.reg < CI->arg_size() ? CI->getArgOperand(o.reg) : nullptr;
      return nullptr;
    };
    Value* dst = argOf(cp->ops[0]); Value* src = argOf(cp->ops[1]);
    if (!dst || !src || !src->getType()->isPointerTy()) continue;
    IRBuilder<> B(CI);
    Value* a = nullptr; Value* b = nullptr;
    if (sp->ops[1].kind == PtxOperand::Reg) a = argOf(sp->ops[1]);
    if (sp->ops[2].kind == PtxOperand::Reg) b = argOf(sp->ops[2]);
    Type* it = a ? a->getType() : b ? b->getType() : nullptr;
    if (!it || !it->isIntegerTy()) continue;
    if (!a) a = ConstantInt::get(it, sp->ops[1].imm, true);
    if (!b) b = ConstantInt::get(it, sp->ops[2].imm, true);
    if (a->getType() != b->getType()) continue;
    CmpInst::Predicate P = cmp == "eq" ? CmpInst::ICMP_EQ : cmp == "ne" ? CmpInst::ICMP_NE
                         : cmp == "lt" ? (isSigned ? CmpInst::ICMP_SLT : CmpInst::ICMP_ULT) : cmp == "le" ? (isSigned ? CmpInst::ICMP_SLE : CmpInst::ICMP_ULE)
                         : cmp == "gt" ? (isSigned ? CmpInst::ICMP_SGT : CmpInst::ICMP_UGT) : (isSigned ? CmpInst::ICMP_SGE : CmpInst::ICMP_UGE);
    Value* ignore = B.CreateICmp(P, a, b);
    Value* valid = B.CreateNot(ignore);
    Module& M = *F.getParent();
    FunctionCallee ZS = M.getOrInsertFunction("__mvcc_cp_async_zsel_" + std::to_string(bytes), FunctionType::get(Type::getVoidTy(F.getContext()), {dst->getType(), src->getType(), Type::getInt1Ty(F.getContext())}, false));
    if (auto* Fn = dyn_cast<Function>(ZS.getCallee())) { Fn->addFnAttr(Attribute::NoUnwind); Fn->addFnAttr(Attribute::Convergent); }
    CallInst* NC = B.CreateCall(ZS, {dst, src, valid});
    NC->setDebugLoc(CI->getDebugLoc());
    CI->eraseFromParent();
    n++;
  }
  if (n && notes) notes->push_back(F.getName().str() + ": " + std::to_string(n) + " ignore-src cp.async (setp + predicate operand) lowered to single zero-selecting copies");
  return n;
}

bool ifConvertGuardedLoads(Function& F, GuardedLoadStats& st, std::vector<std::string>* notes) {
  Converter C{F, st, notes};
  GuardedLoadStats before = st;
  for (int round = 0; round < 8; round++) {
    bool changed = false;
    std::vector<BasicBlock*> blocks;
    for (BasicBlock& BB : F) blocks.push_back(&BB);
    for (BasicBlock* BB : blocks) if (!C.dead.count(BB) && C.tryBranch(BB)) changed = true;
    if (!changed) break;
  }
  if (notes && (st.diamonds != before.diamonds))
    notes->push_back(F.getName().str() + ": " + std::to_string(st.diamonds - before.diamonds) + " guarded region(s) if-converted: " + std::to_string(st.loads - before.loads) +
                     " load(s) redirected through the zero page, " + std::to_string(st.copies - before.copies) + " cp.async pair(s) merged, " + std::to_string(st.mmas - before.mmas) + " guarded mma(s) made unconditional");
  return C.zeroPage != nullptr;
}

}  // namespace mvcc
