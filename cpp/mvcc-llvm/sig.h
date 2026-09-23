// Semantic iteration graph and recurrence algebra.
//
// Loop classification decides how the symbolic executor treats each natural loop of a kernel without the caller
// naming "the K loop": bounded loops unroll, traversal loops (loop-carried data state, trip count a function of the
// launch parameters) run a fixed number of probe iterations with their recurrences recorded, generic loops (control
// state only, e.g. the job loop) run one pass with symbolic header phis, and loops whose trip count depends on data
// computed inside them are refused - the kernel keeps the exact twin for any schedule change along them.
//
// The recurrence classifier names the algebra of every loop-carried value from its `next` expression: what `⊕` is
// in Y[o] = ⊕_r F(...), and therefore which transformations of the traversal coordinate are legal.
#pragma once
#include "symexec.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include <string>
#include <vector>

namespace mvcc {
namespace stir { struct Program; }
namespace sig {

using namespace llvm;

enum class LoopKind { Bounded, Traversal, Generic, Refused };
struct LoopClass {
  const Loop* L; LoopKind kind; unsigned trip = 0;   // Bounded: the constant trip count
  bool uniform = true;                              // Generic: header phis independent of the thread id
  bool earlyExit = false;                           // a constant-trip loop with a data-dependent exit (run as generic)
  unsigned dataPhis = 0;                            // loop-carried non-induction phis
  std::string why;                                  // Refused: the reason
};
const char* loopKindName(LoopKind k);
std::vector<LoopClass> classifyLoops(Function& F, LoopInfo& LI, ScalarEvolution& SE, unsigned maxBounded = 256);
// applies a classification to an executor (generic / traversal sets); Refused loops run as per-thread generic
void applyLoopClasses(sym::Executor& X, const std::vector<LoopClass>& classes);

// Recurrence algebra. `Sum` covers fp (fadd / fma chains) and integer additive accumulators; `Induction` is an
// integer accumulator whose increment is loop-invariant (loop control); `TensorSum` an accumulator carried through a
// __mvcc_tp_mma call's C operand; `Tuple` a recurrence that reads another header phi (online softmax's (m, l) pair) -
// its component algebra is in `detail`; `Permutation` selects among values without arithmetic; `Invariant` is
// unchanged across the loop; anything else is `OrderSensitive` with the expression's shape in `detail`.
enum class Algebra { Invariant, Induction, Sum, Max, Min, And, Or, Xor, TensorSum, Tuple, Permutation, OrderSensitive, Unknown };
const char* algebraName(Algebra a);
// Permissions the algebra grants a schedule
struct Permissions { bool partition = false, reorder = false, duplicate = false, treeCombine = false; };
Permissions permissionsOf(Algebra a);

// `permits` is permissionsOf(kind) except for Tuple, where the component structure decides: a linear recurrence
// l' = a*l + b (a, b free of l) composes associatively (partition, tree-combine); when `a` is an exponential of the
// difference of a sibling Max phi's old and new value it is the online-softmax rescaling, whose (m, l) pair is also
// commutative (reorder).
struct Classified { Algebra kind = Algebra::Unknown; unsigned terms = 0; bool predicated = false; bool initIdentity = false; std::string detail; Permissions permits; };
Classified classifyRecurrence(sym::Arena& A, const sym::Recurrence& r, const std::vector<sym::Recurrence>& siblings);

// Per-kernel SIG. `tpCalls` / `smemAccesses` are the executor's vectors.
struct Graph {
  Function* F = nullptr;
  unsigned T = 0;
  std::vector<sym::Executor::TpCall>& tpCalls;
  std::vector<sym::Executor::SmemAccess>& smemAccesses;
  Graph(Function& f, unsigned t, sym::Executor& X) : F(&f), T(t), tpCalls(X.tpCalls), smemAccesses(X.smemAccesses) {}
};

// The per-kernel SIG summary: runs the executor in retile mode over `threads` threads, classifies loops and
// recurrences, counts effects. `out` receives one line per finding. Returns false (with the reason in `out`) when
// the executor refused the kernel. Refusals are one of: data-dependent loop, effect-ordered, under-probed, time budget.
struct SigOptions { unsigned threads = 32; unsigned iterCount = 4; bool dumpExprs = false; bool dumpEffects = false; uint64_t stepBudget = 2000000; double wallBudget = 30; bool coords = true; bool stir = true; };
SigOptions defaultEmitSigOptions();
// Symbolic execution + coordinate discovery → STIR. False when SIG refuses (caller falls back to heuristics).
bool buildKernelProgram(Function& F, unsigned launchBound, const SigOptions& o, stir::Program& program, std::string& refused);
bool analyzeKernel(Function& F, unsigned launchBound, const SigOptions& o, std::vector<std::string>& out);

}  // namespace sig
}  // namespace mvcc
