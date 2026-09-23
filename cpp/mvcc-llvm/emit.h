// check+emit: legal (Program, Schedule) → recovered LLVM beside the exact twin.
#pragma once
#include "engine.h"
#include "tensor_recovery.h"
#include "llvm/IR/Function.h"

namespace mvcc {
namespace emit {

struct Result {
  bool emitted = false;
  std::string why;
};

// rewrite F when a STIR client matches (elementwise / reduction / gather / gemv / softmax / rmsnorm / rope)
Result apply(llvm::Function& F, const stir::Program& P, const sched::Schedule& S, const legal::Result& L,
             unsigned launchBound, TensorRecoveryResult& res);

// run on every kernel that is not already a tensor-recovery rewrite
void runModule(llvm::Module& M, const TensorRecoveryOptions& opts, TensorRecoveryResult& res);

void bindRecoveryNoteVerbose(const TensorRecoveryOptions* opts);
void pushRecoverySuccessNote(TensorRecoveryResult& res, const std::string& s);
void pushRecoveryDeclineNote(TensorRecoveryResult& res, const std::string& s);

}  // namespace emit
}  // namespace mvcc
