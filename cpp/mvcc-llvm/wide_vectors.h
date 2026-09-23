// Scalarization of vector values MSL cannot represent (> 4 lanes, or double lanes). See wide_vectors.cpp.
#pragma once
#include "llvm/IR/PassManager.h"
#include <functional>
#include <string>

namespace mvcc {

struct ScalarizeWideVectorsPass : llvm::PassInfoMixin<ScalarizeWideVectorsPass> {
  // Called when a wide value cannot be scalarized; the emitter then reports the residual wide type.
  std::function<void(llvm::Function&, const std::string&)> onError;
  llvm::PreservedAnalyses run(llvm::Function& F, llvm::FunctionAnalysisManager&);
};

}  // namespace mvcc
