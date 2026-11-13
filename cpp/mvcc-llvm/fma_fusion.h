// nvcc's -fmad=true contraction of `contract` fmul + fadd/fsub into fma. See fma_fusion.cpp.
#pragma once
#include "llvm/IR/PassManager.h"

namespace mvcc {

struct FmaFusionPass : llvm::PassInfoMixin<FmaFusionPass> {
  llvm::PreservedAnalyses run(llvm::Function& F, llvm::FunctionAnalysisManager&);
};

}  // namespace mvcc
