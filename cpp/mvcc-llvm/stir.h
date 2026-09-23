// Semantic tile IR: the logical content of a kernel with the physical schedule removed.
//
// Built from the semantic iteration graph (sig), coordinate discovery (coords) and the recurrence algebra: tensors
// with their dimensions and unit strides, the logical coordinates with their roles, every device access as an index
// map per dimension (affine in the coordinates, offsets uniform), the predicate literals, and per traversal loop the
// algebra of its recurrences with the schedule permissions it grants.
//
// The identity round trip is checked, not assumed: every address the executor sampled is re-derived from the
// tensor's dimension strides and the access's index maps and compared symbolically (`verified` counts them).
#pragma once
#include "coords.h"
#include "sig.h"
#include <string>
#include <vector>

namespace mvcc {
namespace stir {

using namespace llvm;

struct Dim {
  std::vector<uint32_t> mono;      // the launch-parameter monomial of this dimension's stride ({} = contiguous)
  sym::EP unitBytes = nullptr;     // stride of one index step, in bytes
  std::string name;                // d0, d1, ... (outermost first)
};
struct Tensor {
  std::string name; uint32_t sym = ~0u; unsigned elemBytes = 0; bool elemFromSignature = false;
  bool read = false, written = false;
  std::vector<Dim> dims;           // contiguous dimension last
};
struct IndexMap {                  // index along one dimension: offset + sum_j coef[j] * c_j (in index units)
  sym::EP offset = nullptr;        // uniform polynomial
  std::vector<int64_t> coef;       // per logical coordinate
};
struct Access {
  unsigned tensor = 0; bool store = false, atomic = false; unsigned bytes = 0; const Instruction* site = nullptr; unsigned piece = 0, pieces = 1;
  bool viaDerived = false;
  std::vector<IndexMap> index;     // per dimension of the tensor
  sym::EP leftoverBytes = nullptr; // uniform byte offset that matched no dimension (nullptr = none)
  unsigned samples = 0, verified = 0, mismatches = 0;
};
// Closed form of one cooperative-tensor operand fill: F(X[φ]) at a frag2slot site.
// Address maps live on Access; this is the point function the emitter materializes.
struct Operand {
  const Instruction* site = nullptr;
  std::vector<sym::EP> words;      // typically two i64 slot-word pairs
};
struct Guard { unsigned pred; sym::EP lhsBase; std::vector<sym::EP> lhsCoef; unsigned families = 1; };
struct LoopStmt {
  std::string header; unsigned depth = 0; sig::LoopKind kind; unsigned trip = 0; bool uniform = true;
  std::vector<std::pair<sig::Algebra, unsigned>> algebra;   // tally of the loop-carried values' algebras
  sig::Permissions permits; bool anyData = false;
  std::string why;
};
struct Program {
  std::string kernel;
  std::vector<Tensor> tensors;
  std::vector<std::string> coordNames; std::vector<bool> coordOutput;
  std::vector<std::string> coordPhys;      // c_j as a combination of physical axes
  std::vector<std::string> derivedDefs;
  std::vector<std::string> derivedOperands;   // "d3 operand 0 = <map in the coordinates>"
  std::vector<Access> accesses;
  std::vector<Operand> operands;           // F(X[φ]) for tensor-op fills; empty = identity / no refill
  std::vector<Guard> guards;
  std::vector<LoopStmt> loops;
  unsigned unmapped = 0;                   // fitted address sites with no tensor (not an offset from a pointer argument)
  std::vector<std::string> unfitted;       // the sites coordinate discovery could not fit, with the reason (data-dependent scatter, ...)
  unsigned verified = 0, mismatches = 0;
  std::string numClass = "inherit";        // inherit (default) | explicit-budget | refused silent lower
};

// demangledSignature: the kernel's demangled name (its pointer parameters name the element types)
Program build(sym::Arena& A, const coords::Discovery& D, const std::vector<sig::LoopClass>& classes, const std::vector<LoopStmt>& loopAlgebra, const std::string& demangledSignature);
// the textual form (one finding per line; deterministic for golden tests)
std::vector<std::string> dump(sym::Arena& A, const Program& P);

// construct a Program without CUDA recovery (same checker/emitter consume it)
namespace direct {
Program elementwise(const std::string& name);
Program gemm(const std::string& name);
Program attention(const std::string& name);
}

}  // namespace stir
}  // namespace mvcc
