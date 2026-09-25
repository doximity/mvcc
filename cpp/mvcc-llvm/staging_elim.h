// Staging elimination: STIR-first tensor-op emit (staging_elim.cpp) and guarded-load if-conversion
// (msl_guarded.cpp). Internal to mvcc-ir2msl.
#pragma once
#include "tensor_recovery.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include <string>

namespace mvcc {
namespace staging {

using namespace llvm;

class StagingElim {
 public:
  StagingElim(Function& F, const TensorRecoveryOptions& o, TensorRecoveryResult& r, const std::string& kname, unsigned threads)
      : F(F), opts(o), res(r), kname(kname), T(threads), DL(F.getParent()->getDataLayout()), DT(F), LI(DT),
        TLII(F.getParent()->getTargetTriple()), TLI(TLII), AC(F), SE(F, TLI, AC, DT, LI) {}
  bool emitFromStir();

 private:
  Function& F;
  const TensorRecoveryOptions& opts;
  TensorRecoveryResult& res;
  std::string kname;
  unsigned T;
  const DataLayout& DL;
  DominatorTree DT;
  LoopInfo LI;
  TargetLibraryInfoImpl TLII;
  TargetLibraryInfo TLI;
  AssumptionCache AC;
  ScalarEvolution SE;
  void note(const std::string& s) { if (opts.verbose) res.notes.push_back(kname + ": " + s); }
  // Catalog f16 bodies already have tp_mma; succeed without decode carryWalk.
  bool stageRingParent(const llvm::Loop* K, std::string& why) {
    const llvm::Loop* P = K->getParentLoop();
    if (!P) return false;
    unsigned ptc = SE.getSmallConstantTripCount(const_cast<llvm::Loop*>(P));
    if (!ptc) ptc = SE.getSmallConstantMaxTripCount(const_cast<llvm::Loop*>(P));
    if (ptc && ptc <= 8) { why = std::to_string(ptc) + "-trip stage ring"; return true; }
    if (ptc) return false;
    llvm::BasicBlock* latch = const_cast<llvm::Loop*>(P)->getLoopLatch();
    if (!latch) return false;
    auto* BI = llvm::dyn_cast<llvm::BranchInst>(latch->getTerminator());
    if (!BI || !BI->isConditional()) return false;
    auto* cmp = llvm::dyn_cast<llvm::ICmpInst>(BI->getCondition());
    if (!cmp) return false;
    for (int i = 0; i < 2; i++)
      if (auto* C = llvm::dyn_cast<llvm::ConstantInt>(cmp->getOperand(i)))
        if (C->getZExtValue() >= 1 && C->getZExtValue() <= 8) {
          why = "stage-ring latch bound " + std::to_string(C->getZExtValue());
          return true;
        }
    return false;
  }
};

}  // namespace staging

struct GuardedLoadStats {
  unsigned diamonds = 0;
  unsigned loads = 0;
  unsigned copies = 0;
  unsigned mmas = 0;
};

bool ifConvertGuardedLoads(llvm::Function& F, GuardedLoadStats& st, std::vector<std::string>* notes);
unsigned lowerPredicatedCopies(llvm::Function& F, std::vector<std::string>* notes);

}  // namespace mvcc
