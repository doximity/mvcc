// Coordinate discovery: the logical coordinates of a kernel emerge from its
// physical access set, they are not named.
//
// Input: the device reads and writes the symbolic executor recorded for a set of threads and probe iterations
// (sym::Executor::devLoads / devStores), each an address polynomial over the launch parameters for one concrete
// point of the physical iteration space Phys = (bits of tid.x, iteration index of every traversal loop level).
//
// Per access site the address is fitted as affine in Phys: addr(x) = base + sum_a stride_a * x_a with `base` and the
// strides polynomials over the uniform atoms only (arguments, block indices, uniform loop values), verified on every
// sample. The strides of every fitted site (and of every predicate literal) form the integer matrix S : Phys -> Z^M
// (one column per (site, monomial) pair); its Hermite normal form U*S = H gives the logical dimension d = rank S, the
// basis change c = V^T x (V = U^-1, first d columns) and, per site, the stride of every logical coordinate. A
// coordinate that moves a store address is an output coordinate `o`, the rest are reduction / traversal coordinates
// `r`; physical axes along which nothing moves are replication.
//
// An address through a loaded value is an index-tensor access phi = X[Load(psi)]: psi is fitted in its place and the
// site is marked. Anything else that depends on a per-thread opaque value is reported as data-dependent with the
// atom that caused it. The fit never trusts the differences: every sample is re-evaluated against the map.
#pragma once
#include "symexec.h"
#include <string>
#include <vector>

namespace mvcc {
namespace coords {

using namespace llvm;

struct Axis { enum Kind { TidBit, Iter, Derived, Simdgroup, Lane } kind; unsigned index; const Loop* loop = nullptr; };
// Phys is simdgroup (tid.x/32) + lane (tid.x%32) + traversal inductions, not raw tid bits.

struct SiteMap {
  Instruction* site = nullptr; bool isStore = false, atomic = false; unsigned bytes = 0; unsigned piece = 0, pieces = 1;
  unsigned samples = 0;
  bool isPred = false;             // a predicate literal's polynomial (left-hand side, compared against a constant)
  unsigned pred = 0;               // Cmp predicate for isPred
  std::string tensor;              // the pointer argument (or global) the address is an offset from; "" if none
  uint32_t tensorSym = ~0u;
  bool viaIndex = false;           // the address goes through derived quantities (quotients by a runtime uniform, loaded indices)
  bool isDerivedOperand = false;   // the integer operand `operand` of derived quantity d<derived>: its affine map
  unsigned derived = 0, operand = 0;
  sym::EP base = nullptr;          // address at x = 0
  std::vector<sym::EP> stride;     // per axis (physical, then derived); nullptr = the axis does not move the address
  std::vector<sym::EP> coordStride;   // after discovery: per logical coordinate
  std::vector<std::pair<std::vector<int>, sym::EP>> points;   // the fitted samples: (physical point, value there)
  std::string why;                 // non-empty: not fitted (data-dependent, non-affine, under-sampled)
  bool fitted() const { return why.empty(); }
};

struct Discovery {
  std::vector<Axis> axes;          // physical axes first, then one per derived quantity
  std::vector<SiteMap> sites;
  unsigned d = 0;
  std::vector<std::vector<int64_t>> V;     // |axes| x d: coordinate j = sum_a V[a][j] * x_a
  std::vector<bool> isOutput;              // per coordinate
  std::vector<unsigned> replication;       // physical axes no fitted access moves along
  unsigned fittedSites = 0, dataDependent = 0, nonAffine = 0;
  std::vector<std::string> axisNames;      // per axis: b<bit>, i<level>, d<derived>
  std::vector<std::string> derivedDefs;    // per derived quantity (in axes order): its definition, for reports
  std::vector<sym::EP> derivedValue;       // per derived quantity: its placeholder as an i64 polynomial (the sampled values are over it)
  std::vector<std::string> lines;          // the report (one finding per line)
};

std::string polyStr(sym::Arena& A, sym::EP e);      // a polynomial over the uniform atoms, for reports
std::string siteName(const Instruction* I);         // "%name" / "cp.async" / "store to %p"

// T: the block's thread count (axes: the bits of tid.x below T); iterLevels: the depth of traversal-loop nesting
// the executor probed (from the recorded iteration vectors); debug adds the stride matrix and every failing sample.
Discovery discover(sym::Arena& A, const sym::Executor& X, unsigned T);

}  // namespace coords
}  // namespace mvcc
