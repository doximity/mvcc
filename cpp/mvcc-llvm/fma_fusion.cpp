// nvcc compiles `a * b + c` to one fma by default (-fmad=true): clang marks the operations `contract`, and the
// NVPTX DAG combiner fuses them while selecting each basic block. Metal fuses nothing in safe math mode, so
// without this pass every such expression rounds twice here and once on the GPU, and results drift in the last
// place. The rules are the combiner's with aggressive fusion, which NVPTX enables:
//   fadd (fmul x, y), z        -> fma x, y, z         (of two fmul operands, the one with fewer uses)
//   fsub (fmul x, y), z        -> fma x, y, -z
//   fsub x, (fmul y, z)        -> fma -y, z, x        (preferred when both are fmuls and the left has more uses)
//   fsub (fneg (fmul x, y)), z -> fma -x, y, -z
//   fadd a, (fneg b)           -> fsub a, b           (either order; the combiner's canonicalizations)
//   fsub a, (fneg b)           -> fadd a, b
// An fmul fuses even when it has other uses. The DAG holds one block, so an fmul (or fneg) from another block
// arrives as a register and never fuses, and a use outside the block or by a phi counts once. The combiner pops
// its worklist from the back, so a block is visited last instruction first. Chains that need `reassoc` (fast
// math) are not reassociated. -fmad=false compiles with -ffp-contract=off: no `contract` flags, nothing fuses.
// float and half, scalar or vector; double is emulated (mvcc_f64) and bfloat has no Metal fma.
#include "fma_fusion.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"

using namespace llvm;

namespace mvcc {
namespace {

bool fusibleTy(Type* T) {
  Type* E = T->getScalarType();
  return (E->isFloatTy() || E->isHalfTy()) && (T == E || isa<FixedVectorType>(T));
}

Instruction* local(Value* V, unsigned op, BasicBlock* BB) {
  auto* I = dyn_cast<Instruction>(V);
  return I && I->getOpcode() == op && I->getParent() == BB ? I : nullptr;
}

Instruction* contractableMul(Value* V, BasicBlock* BB) {
  Instruction* M = local(V, Instruction::FMul, BB);
  return M && M->hasAllowContract() ? M : nullptr;
}

unsigned dagUses(Instruction* M) {
  unsigned inBlock = 0;
  bool elsewhere = false;
  for (const Use& U : M->uses()) {
    auto* I = cast<Instruction>(U.getUser());
    if (I->getParent() == M->getParent() && !isa<PHINode>(I)) ++inBlock; else elsewhere = true;
  }
  return inBlock + elsewhere;
}

struct Fuser {
  IRBuilder<> B;
  Instruction* N;
  explicit Fuser(Instruction* n) : B(n), N(n) {}

  // N becomes fma(x, y, z); the fmuls and fnegs it consumed go when nothing else reads them.
  void replace(Value* x, Value* y, Value* z) {
    SmallSetVector<Instruction*, 4> dead;
    for (Value* V : N->operands()) {
      auto* I = dyn_cast<Instruction>(V);
      if (!I) continue;
      dead.insert(I);
      if (I->getOpcode() == Instruction::FNeg)
        if (auto* M = dyn_cast<Instruction>(I->getOperand(0))) dead.insert(M);
    }
    Value* F = B.CreateIntrinsic(Intrinsic::fma, {N->getType()}, {x, y, z});
    F->takeName(N);
    N->replaceAllUsesWith(F);
    N->eraseFromParent();
    for (Instruction* I : dead)
      if (I->use_empty() && (I->getOpcode() == Instruction::FMul || I->getOpcode() == Instruction::FNeg))
        I->eraseFromParent();
  }
  Value* neg(Value* V) { return B.CreateFNeg(V); }

  bool add(Value* a, Value* b) {
    BasicBlock* BB = N->getParent();
    if (Instruction* f = local(b, Instruction::FNeg, BB)) return sub(a, f->getOperand(0));
    if (Instruction* f = local(a, Instruction::FNeg, BB)) return sub(b, f->getOperand(0));
    Instruction *ma = contractableMul(a, BB), *mb = contractableMul(b, BB);
    if (ma && mb && dagUses(ma) > dagUses(mb)) { std::swap(ma, mb); std::swap(a, b); }
    if (ma) { replace(ma->getOperand(0), ma->getOperand(1), b); return true; }
    if (mb) { replace(mb->getOperand(0), mb->getOperand(1), a); return true; }
    return false;
  }

  bool sub(Value* a, Value* b) {
    BasicBlock* BB = N->getParent();
    if (Instruction* f = local(b, Instruction::FNeg, BB)) return add(a, f->getOperand(0));
    Instruction *ma = contractableMul(a, BB), *mb = contractableMul(b, BB);
    if (mb && (!ma || dagUses(ma) > dagUses(mb))) { replace(neg(mb->getOperand(0)), mb->getOperand(1), a); return true; }
    if (ma) { replace(ma->getOperand(0), ma->getOperand(1), neg(b)); return true; }
    if (Instruction* f = local(a, Instruction::FNeg, BB))
      if (Instruction* m = contractableMul(f->getOperand(0), BB)) {
        replace(neg(m->getOperand(0)), m->getOperand(1), neg(b));
        return true;
      }
    return false;
  }
};

}  // namespace

PreservedAnalyses FmaFusionPass::run(Function& F, FunctionAnalysisManager&) {
  bool changed = false;
  for (BasicBlock& BB : F) {
    SmallVector<Instruction*, 64> work;
    for (Instruction& I : llvm::reverse(BB))
      if ((I.getOpcode() == Instruction::FAdd || I.getOpcode() == Instruction::FSub) && I.hasAllowContract() &&
          fusibleTy(I.getType()))
        work.push_back(&I);
    for (Instruction* I : work) {
      Fuser f(I);
      changed |= I->getOpcode() == Instruction::FAdd ? f.add(I->getOperand(0), I->getOperand(1))
                                                     : f.sub(I->getOperand(0), I->getOperand(1));
    }
  }
  return changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}

}  // namespace mvcc
