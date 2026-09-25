// Decode-DOT recovery: Y[n] += scale · ⟨unpack(W[n,k]), x[k]⟩ plus a 32-lane xor-sum.
// Same shape as matmul2d for MMA — a first-class STIR body, not an MSL peephole.
#pragma once
#include "emit.h"

namespace mvcc {
namespace emit {

// Recognize the contraction, prove a Metal schedule (simd_sum + thread_scale), emit it,
// keep the exact twin. Returns emitted=false when the IR is not this Program.
Result recoverDotGemv(llvm::Function& F, unsigned launchBound, TensorRecoveryResult& res);

}  // namespace emit
}  // namespace mvcc
