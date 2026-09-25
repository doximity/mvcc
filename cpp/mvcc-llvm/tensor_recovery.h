// Tensor-pipeline recovery: SIG → STIR → schedule → check → emit.
//
// Recognizes ldmatrix → mma.sync → register accumulators, recovers
// Y[o] = ⊕_r F(...) onto TensorOps (__mvcc_tp_* → Metal 4 matmul2d), and keeps
// an exact-semantics twin named <kernel>__mvcc_exact. The runtime chooses between them.
#pragma once
#include "llvm/IR/Module.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace mvcc {

struct TpDescriptor {
  int M = 0, N = 0, K = 0;
  bool tl = false, tr = false;  // matmul2d transpose flags (left = A stored [k][m]; right = B stored [n][k])
  int type = 0;                 // 0 = f16, 1 = bf16
  bool operator<(const TpDescriptor& o) const {
    return std::tie(M, N, K, tl, tr, type) < std::tie(o.M, o.N, o.K, o.tl, o.tr, o.type);
  }
  std::string mslType() const { return type == 0 ? "half" : "bfloat"; }
  std::string tag() const;  // "half_32x32x32_tl0_tr1"
};

struct TensorRecoveryOptions {
  bool enabled = true;
  bool verbose = false;  // per-kernel notes on stderr (MVCC_TENSOR_DIAG=1 or -v)
  // Fusion caps in 16-blocks per matmul2d along M, N, K (1 or 2; 0 = heuristic).
  // MVCC_TP_GROUPS=m,n,k overrides.
  int groupM = 2, groupN = 2, groupK = 0;
};

struct TensorRecoveryResult {
  std::vector<std::pair<std::string, std::string>> exactVariants;
  std::vector<std::pair<std::string, std::string>> searchVariants; // kernel -> <name>__mvcc_s0
  std::set<TpDescriptor> descriptors;
  std::vector<std::string> notes;
  std::vector<std::string> warnings;
  unsigned kernelsWithMma = 0, mmaKernelsRecovered = 0;
  std::map<std::string, unsigned> stagingThreads;
  std::map<std::string, unsigned> threadScale;
  std::map<std::string, unsigned> dynSmemUsed;
  std::map<std::string, std::pair<unsigned, unsigned>> dynSmemRelocate;
  std::map<std::string, std::vector<std::string>> assumptions;
  std::vector<std::pair<std::string, unsigned>> collectives;
};

bool emitFromStir(llvm::Function& F, const TensorRecoveryOptions& opts, TensorRecoveryResult& res, const std::string& kname, unsigned maxThreads);
unsigned kernelMaxThreads(const llvm::Function& F);

void runTensorRecovery(llvm::Module& M, const TensorRecoveryOptions& opts, TensorRecoveryResult& out);

}  // namespace mvcc
