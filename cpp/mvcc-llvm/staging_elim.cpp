// STIR-first tensor-op emit (staging_elim.h): recover Program.operands (F(X[φ])), check(P,S), materialize.
#include "staging_elim.h"
#include "sig.h"
#include "coords.h"
#include "stir.h"
#include "schedule.h"
#include "legality.h"
#include "engine.h"
#include "symexec.h"
#include "llvm/Demangle/Demangle.h"
#include "ptx_asm.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Transforms/Utils/LoopUtils.h"
#include "llvm/Transforms/Utils/LoopSimplify.h"
#include "llvm/Transforms/Utils/UnrollLoop.h"
#include "llvm/Transforms/Utils/ValueMapper.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include <array>
#include <chrono>
#include <cmath>
#include <deque>
#include <functional>
#include <map>
#include <numeric>
#include <set>

namespace mvcc {
namespace staging {
using namespace sym;

namespace {

// the signed integer min/max a call computes: libdevice's __nv_min/__nv_max or the intrinsics they are lowered to
// (msl_emitter.cpp, LowerLibdeviceMinMaxPass). 0 = neither, -1 = min, +1 = max
int minMaxCallee(const Function* f) {
  if (!f) return 0;
  StringRef n = f->getName();
  if (n == "__nv_min" || n.starts_with("llvm.smin.")) return -1;
  if (n == "__nv_max" || n.starts_with("llvm.smax.")) return 1;
  return 0;
}

// ---------------------------------------------------------------- block uniformity
static std::string asmMnemonics(const CallInst* CI, bool& allAlu);
static bool concretelyBounded(const Loop* L) {
  SmallVector<BasicBlock*, 4> exiting; L->getExitingBlocks(exiting);
  std::map<const Value*, int> memo;
  std::function<bool(const Value*, int)> conc = [&](const Value* v, int depth) -> bool {
    if (isa<Constant>(v)) return true;
    if (isa<Argument>(v)) return false;
    auto it = memo.find(v); if (it != memo.end()) return it->second == 1;
    if (depth > 64) return false;
    memo[v] = 0;
    bool r = false;
    if (auto* PN = dyn_cast<PHINode>(v)) {
      // L's own induction: its entry values must be concrete (the back-edge values are functions of the phis)
      if (L->contains(PN->getParent())) { r = true; for (unsigned i = 0; i < PN->getNumIncomingValues() && r; i++) if (!L->contains(PN->getIncomingBlock(i))) r = conc(PN->getIncomingValue(i), depth + 1); }
    } else if (auto* CI = dyn_cast<CallInst>(v)) {
      const Function* cf = CI->getCalledFunction(); StringRef n = cf ? cf->getName() : StringRef();
      if (n.starts_with("llvm.nvvm.read.ptx.sreg.tid.") || n.starts_with("llvm.nvvm.read.ptx.sreg.ntid.") || n == "llvm.nvvm.read.ptx.sreg.laneid" || n == "llvm.nvvm.read.ptx.sreg.warpsize") r = true;
      else if (n.starts_with("llvm.smin.") || n.starts_with("llvm.smax.") || n.starts_with("llvm.umin.") || n.starts_with("llvm.umax.")) { r = true; for (const Value* a : CI->args()) r = r && conc(a, depth + 1); }
    } else if (auto* I = dyn_cast<Instruction>(v)) {
      if (isa<BinaryOperator>(I) || isa<CastInst>(I) || isa<CmpInst>(I) || isa<SelectInst>(I)) { r = true; for (const Value* o : I->operands()) r = r && conc(o, depth + 1); }
    }
    memo[v] = r ? 1 : 0;
    return r;
  };
  for (BasicBlock* BB : exiting) {
    auto* BI = dyn_cast<BranchInst>(BB->getTerminator());
    if (!BI || !BI->isConditional() || !conc(BI->getCondition(), 0)) return false;
  }
  return true;
}

struct UniformityScan {
  std::map<const Value*, int> memo, laneMemo;   // -1 deciding, 0 no, 1 yes
  const Value* why = nullptr;   // the first value found per-thread (diagnostics)
  // memory the kernel never writes: a noalias readonly argument, a constant global, the zero page (a guarded load's
  // redirection target; its bytes are zero for everyone)
  static bool readOnlyMemory(const LoadInst* LD) {
    if (LD->getMetadata(LLVMContext::MD_invariant_load)) return true;
    SmallVector<const Value*, 4> objs;
    getUnderlyingObjects(LD->getPointerOperand(), objs);   // through selects and phis
    if (objs.empty()) return false;
    for (const Value* o : objs) {
      if (auto* A = dyn_cast<Argument>(o)) { if (A->hasNoAliasAttr() && (A->onlyReadsMemory() || A->hasAttribute(Attribute::ReadOnly))) continue; return false; }
      if (auto* GV = dyn_cast<GlobalVariable>(o)) { if (GV->isConstant() || GV->getAddressSpace() == 4) continue; return false; }
      if (auto* CI = dyn_cast<CallInst>(o)) if (CI->getCalledFunction() && CI->getCalledFunction()->getName() == "__mvcc_zero_page") continue;
      return false;
    }
    return true;
  }
  static bool pureCall(const CallInst* CI) {
    if (CI->isInlineAsm()) { bool alu; asmMnemonics(CI, alu); return alu; }
    const Function* cf = CI->getCalledFunction(); if (!cf) return false;
    StringRef n = cf->getName();
    if (n.starts_with("__nv_") || n == "__mvcc_zero_page") return true;
    if (n.starts_with("llvm.smin.") || n.starts_with("llvm.smax.") || n.starts_with("llvm.umin.") || n.starts_with("llvm.umax.") || n.starts_with("llvm.abs.") || n.starts_with("llvm.fabs.") || n.starts_with("llvm.ctlz.") || n.starts_with("llvm.cttz.") || n.starts_with("llvm.ctpop.") || n.starts_with("llvm.fma.") || n.starts_with("llvm.fmuladd.") || n.starts_with("llvm.sqrt.") || n.starts_with("llvm.minnum.") || n.starts_with("llvm.maxnum.") || n.starts_with("llvm.bswap.") || n.starts_with("llvm.fshl.") || n.starts_with("llvm.fshr.")) return true;
    return false;
  }
  // depends on the thread through its lane only (or not at all)
  bool laneOnly(const Value* v, int depth = 0) {
    if (isa<Constant>(v) || isa<Argument>(v)) return true;
    auto it = laneMemo.find(v); if (it != laneMemo.end()) return it->second != 0;
    if (depth > 256) return false;
    laneMemo[v] = -1;
    bool r = false;
    if (auto* CI = dyn_cast<CallInst>(v)) {
      const Function* cf = CI->getCalledFunction(); StringRef n = cf ? cf->getName() : StringRef();
      if (n == "llvm.nvvm.read.ptx.sreg.laneid") r = true;
      else if (n.starts_with("llvm.nvvm.read.ptx.sreg.")) r = !n.starts_with("llvm.nvvm.read.ptx.sreg.tid.");
      else if (n == "llvm.nvvm.vote.ballot.sync") r = uniform(CI, depth + 1);   // a full-warp ballot of a lane-only predicate: the same in every warp
      else if (pureCall(CI)) { r = true; for (const Value* a : CI->args()) r = r && laneOnly(a, depth + 1); }
    } else if (auto* BO = dyn_cast<BinaryOperator>(v)) {
      // tid & 31, tid % 32: the lane
      auto* K = dyn_cast<ConstantInt>(BO->getOperand(1));
      auto* TC = dyn_cast<CallInst>(BO->getOperand(0));
      const bool tidRead = TC && TC->getCalledFunction() && TC->getCalledFunction()->getName() == "llvm.nvvm.read.ptx.sreg.tid.x";
      if (tidRead && K && ((BO->getOpcode() == Instruction::And && K->getZExtValue() == 31) || (BO->getOpcode() == Instruction::URem && K->getZExtValue() == 32))) r = true;
      else { r = true; for (const Use& U : BO->operands()) r = r && laneOnly(U.get(), depth + 1); }
    } else if (auto* LD = dyn_cast<LoadInst>(v)) {
      r = readOnlyMemory(LD) && laneOnly(LD->getPointerOperand(), depth + 1);
    } else if (auto* PN = dyn_cast<PHINode>(v)) {
      r = true; for (const Value* in : PN->incoming_values()) r = r && laneOnly(in, depth + 1);
    } else if (auto* I = dyn_cast<Instruction>(v)) {
      if (I->mayHaveSideEffects() || I->mayReadFromMemory()) r = false;
      else { r = true; for (const Use& U : I->operands()) r = r && laneOnly(U.get(), depth + 1); }
    }
    laneMemo[v] = r ? 1 : 0;
    return r;
  }
  bool uniform(const Value* v, int depth = 0) {
    if (isa<Constant>(v) || isa<Argument>(v)) return true;
    auto it = memo.find(v); if (it != memo.end()) return it->second != 0;
    if (depth > 256) return false;
    memo[v] = -1;   // deciding: a cycle through a phi is uniform if the rest is
    bool r = false;
    if (auto* CI = dyn_cast<CallInst>(v)) {
      const Function* cf = CI->getCalledFunction(); StringRef n = cf ? cf->getName() : StringRef();
      if (n.starts_with("llvm.nvvm.read.ptx.sreg.")) r = !n.starts_with("llvm.nvvm.read.ptx.sreg.tid.") && n != "llvm.nvvm.read.ptx.sreg.laneid";
      else if (n == "llvm.nvvm.vote.ballot.sync") { auto* Mk = dyn_cast<ConstantInt>(CI->getArgOperand(0)); r = Mk && Mk->isMinusOne() && laneOnly(CI->getArgOperand(1), depth + 1); }
      else if (pureCall(CI)) { r = true; for (const Value* a : CI->args()) r = r && uniform(a, depth + 1); }
    } else if (auto* LD = dyn_cast<LoadInst>(v)) {
      r = readOnlyMemory(LD) && uniform(LD->getPointerOperand(), depth + 1);
    } else if (auto* PN = dyn_cast<PHINode>(v)) {
      r = true; for (const Value* in : PN->incoming_values()) r = r && uniform(in, depth + 1);
    } else if (auto* I = dyn_cast<Instruction>(v)) {
      if (I->mayHaveSideEffects() || I->mayReadFromMemory()) r = false;
      else { r = true; for (const Use& U : I->operands()) r = r && uniform(U.get(), depth + 1); }
    }
    memo[v] = r ? 1 : 0;
    if (!r && !why) why = v;
    return r;
  }
  // every value the loop defines is uniform, its branches included (all threads take the same exit in the same
  // iteration), and it has no effects (stores, barriers, calls with side effects)
  bool loopUniform(const Loop* L) {
    for (BasicBlock* BB : L->blocks()) for (Instruction& I : *BB) {
      if (I.isTerminator()) { if (auto* Br = dyn_cast<BranchInst>(&I)) { if (Br->isConditional() && !uniform(Br->getCondition())) return false; } else if (auto* SW = dyn_cast<SwitchInst>(&I)) { if (!uniform(SW->getCondition())) return false; } else return false; continue; }
      if (I.mayWriteToMemory() || (isa<CallInst>(&I) && cast<CallInst>(&I)->isConvergent() && !uniform(&I))) return false;
      if (!I.getType()->isVoidTy() && !uniform(&I)) return false;
    }
    return true;
  }
};

// ---------------------------------------------------------------- shapes and affine fits

// The shape of a set of expressions: one template tree with every integer coefficient/constant a hole that has a
// value per sample. Polynomials unify monomial by monomial (a monomial absent from a sample has coefficient 0 there),
// so A(row 0, k) and A(row 3, k) - which differ by whether the term 2*K*row exists - have one shape. A selection
// one sample lacks (its condition was decided there, e.g. row 0's zero-fill guard coincides with the job's) is
// unified with the arm the sample has; the holes of the other arm and of the condition have no value for that
// sample. Symbols must be thread-independent (arguments, block/grid registers, uniform loop values, shared bases).
struct Node {
  EK k; unsigned op = 0, lo = 0, len = 0; Type* ty = nullptr; Function* callee = nullptr; Value* asmv = nullptr; uint32_t sym = 0;
  APInt fc;                                                 // Const: non-integer (float) bits
  bool assoc = false;                                       // boolean and/or: ch is the flattened, unordered operand set
  std::vector<Node*> ch;                                   // Op/Cmp/Ite/Load/Concat/Bytes/Call operands
  struct Mono { std::vector<Node*> atoms; std::vector<int64_t> coefs; std::vector<char> present; int hole = -1; };
  std::vector<Mono> monos;                                  // Poly
  std::vector<int64_t> vals; std::vector<char> present; int hole = -1;   // Const (integer) / Bytes (lo): one value per sample
  int cls = -1;                                             // Shaper::classify: subtrees of one class instantiate alike
  bool isIntConst() const { return k == EK::Const && ty->isIntegerTy() && ty->getIntegerBitWidth() <= 64; }
  bool hasVal() const { return isIntConst() || k == EK::Bytes; }
};

// coefficient = c0 + sum_i c[i] * coord[i], solved exactly over the samples that have a value and verified on them
struct Affine { int64_t c0 = 0; std::vector<int64_t> c; };

static bool fitAffine(const std::vector<std::vector<int64_t>>& coords, const std::vector<int64_t>& vals, const std::vector<char>& present, Affine& out) {
  const unsigned n = coords.empty() ? 0 : (unsigned)coords[0].size();
  const unsigned m = n + 1;
  // normal equations in long double, rounded, then verified exactly
  std::vector<std::vector<long double>> M(m, std::vector<long double>(m + 1, 0));
  for (size_t s = 0; s < vals.size(); s++) {
    if (!present[s]) continue;
    std::vector<long double> x(m, 1); for (unsigned i = 0; i < n; i++) x[i + 1] = (long double)coords[s][i];
    for (unsigned r = 0; r < m; r++) { for (unsigned c = 0; c < m; c++) M[r][c] += x[r] * x[c]; M[r][m] += x[r] * (long double)vals[s]; }
  }
  // Gaussian elimination with free variables set to zero
  std::vector<int> pivotCol(m, -1);
  unsigned row = 0;
  for (unsigned c = 0; c < m && row < m; c++) {
    unsigned best = row; for (unsigned r = row + 1; r < m; r++) if (fabsl(M[r][c]) > fabsl(M[best][c])) best = r;
    if (fabsl(M[best][c]) < 1e-9L) continue;
    std::swap(M[row], M[best]);
    for (unsigned r = 0; r < m; r++) if (r != row) { long double f = M[r][c] / M[row][c]; for (unsigned k = c; k <= m; k++) M[r][k] -= f * M[row][k]; }
    pivotCol[row] = (int)c; row++;
  }
  std::vector<long double> sol(m, 0);
  for (unsigned r = 0; r < row; r++) sol[pivotCol[r]] = M[r][m] / M[r][pivotCol[r]];
  out.c0 = (int64_t)llroundl(sol[0]); out.c.assign(n, 0);
  for (unsigned i = 0; i < n; i++) out.c[i] = (int64_t)llroundl(sol[i + 1]);
  for (size_t s = 0; s < vals.size(); s++) {
    if (!present[s]) continue;
    int64_t v = out.c0; for (unsigned i = 0; i < n; i++) v += out.c[i] * coords[s][i];
    if (v != vals[s]) return false;
  }
  return true;
}

struct Shaper {
  Arena& A;
  std::vector<std::unique_ptr<Node>> pool;   // the template's nodes
  std::vector<std::unique_ptr<Node>> cur;    // the sample being unified (kept only if a subtree of it joined the template)
  bool adoptedAny = false;
  std::string why;
  unsigned nSamples = 0;
  std::vector<std::pair<std::vector<int64_t>*, std::vector<char>*>> holeList;   // by hole index
  explicit Shaper(Arena& a) : A(a) {}
  std::vector<std::unique_ptr<Node>> spare;   // nodes of samples that did not join the template, for reuse (their vectors keep their storage)
  Node* mk() { if (spare.empty()) cur.push_back(std::make_unique<Node>()); else { cur.push_back(std::move(spare.back())); spare.pop_back(); } return cur.back().get(); }
  std::vector<Node::Mono> spareMonos;         // monomials of those nodes, likewise (their three vectors keep their storage)
  Node::Mono mkMono() { if (spareMonos.empty()) return Node::Mono(); Node::Mono m = std::move(spareMonos.back()); spareMonos.pop_back(); return m; }
  void reset(Node& n) {
    n.k = EK::Const; n.op = n.lo = n.len = 0; n.ty = nullptr; n.callee = nullptr; n.asmv = nullptr; n.sym = 0; n.fc = APInt(); n.assoc = false; n.ch.clear(); n.vals.clear(); n.present.clear(); n.hole = -1; n.cls = -1;
    for (auto& m : n.monos) { m.atoms.clear(); m.coefs.clear(); m.present.clear(); m.hole = -1; spareMonos.push_back(std::move(m)); }
    n.monos.clear();
  }
  void keepSample(bool keep) { for (auto& n : cur) { if (keep) pool.push_back(std::move(n)); else { reset(*n); spare.push_back(std::move(n)); } } cur.clear(); adoptedAny = false; }
  bool symOk(uint32_t sid) {
    switch (A.symbol(sid).kind) { case Symbol::Arg: case Symbol::Sreg: case Symbol::LoopValue: case Symbol::SharedBase: case Symbol::Global: return true; default: return false; }
  }
  static void put(std::vector<int64_t>& vals, std::vector<char>& present, unsigned s, int64_t v) { vals.resize(s, 0); present.resize(s, 0); vals.push_back(v); present.push_back(1); }
  static void one(std::vector<int64_t>& vals, std::vector<char>& present, int64_t v) { vals.assign(1, v); present.assign(1, 1); }
  // one sample's tree; every hole holds the sample's single value (adopt() re-indexes a subtree that joins the template)
  Node* build(EP e, int depth = 0) {
    if (!why.empty()) return nullptr;
    if (depth > 64) { why = "expression too deep"; return nullptr; }
    Node* n = mk(); n->k = e->k; n->ty = e->ty;
    switch (e->k) {
      case EK::Const:
        if (n->isIntConst()) one(n->vals, n->present, e->c.getSExtValue()); else n->fc = e->c;
        return n;
      case EK::Poly:
        for (auto& t : e->terms) {
          Node::Mono m = mkMono(); one(m.coefs, m.present, t.second);
          for (uint32_t atom : t.first) {
            uint32_t sid;
            if (A.isSymAtom(atom, sid)) { if (!symOk(sid)) { std::string is; if (A.symbol(sid).v) { raw_string_ostream os(is); os << " <- " << *A.symbol(sid).v; } why = "operand depends on a per-thread value the executor could not follow (" + A.str(A.atomExpr(atom)) + StringRef(is).trim().str() + ")"; return nullptr; } Node* a = mk(); a->k = EK::Sym; a->sym = sid; a->ty = e->ty; m.atoms.push_back(a); }
            else { Node* a = build(A.atomExpr(atom), depth + 1); if (!a) return nullptr; m.atoms.push_back(a); }
          }
          n->monos.push_back(std::move(m));
        }
        return n;
      case EK::Sym:
        if (!symOk(e->sym)) { why = "operand depends on a per-thread value (" + A.str(e) + ")"; return nullptr; }
        n->sym = e->sym; return n;
      case EK::Undef: why = "undefined value in an operand"; return nullptr;
      case EK::Op: case EK::Cmp:
        n->op = e->op;
        if (e->k == EK::Op && e->ty->isIntegerTy(1) && (e->op == Instruction::And || e->op == Instruction::Or)) {
          // the arena orders conjuncts by interning id, which differs between samples: unordered
          n->assoc = true;
          std::vector<EP> ops; std::function<void(EP)> flat = [&](EP x) { if (x->k == EK::Op && x->ty == e->ty && x->op == e->op) { for (EP y : x->args) flat(y); } else ops.push_back(x); }; flat(e);
          for (EP x : ops) { Node* c = build(x, depth + 1); if (!c) return nullptr; n->ch.push_back(c); }
          return n;
        }
        break;
      case EK::Load: n->len = e->len; break;
      case EK::Bytes: n->len = e->len; one(n->vals, n->present, e->lo); break;   // the byte offset is a hole (k within a chunk)
      case EK::Call: n->callee = e->callee; n->asmv = e->asmv; break;
      default: break;
    }
    for (EP a : e->args) { Node* c = build(a, depth + 1); if (!c) return nullptr; n->ch.push_back(c); }
    return n;
  }
  static bool sameHead(const Node* a, const Node* b) {
    return a->k == b->k && a->ty == b->ty && a->op == b->op && a->lo == b->lo && a->len == b->len && a->callee == b->callee && a->asmv == b->asmv && a->sym == b->sym && a->assoc == b->assoc && a->ch.size() == b->ch.size() && (a->k != EK::Const || a->isIntConst() || a->fc == b->fc);
  }
  // structural equality up to coefficients and constants; polynomials: one side's monomials all have a match
  // exact: constants and coefficients must agree too (the sample's values against the template's latest sample)
  bool shapeEq(Node* a, Node* b, int depth = 0, bool exact = false) {
    if (depth > 64 || !sameHead(a, b)) return false;
    if (exact && a->hasVal() && (a->vals.empty() || b->vals.empty() || a->vals.back() != b->vals.back())) return false;
    if (a->assoc) {
      std::vector<char> used(b->ch.size(), 0);
      for (Node* x : a->ch) { bool f = false; for (size_t j = 0; j < b->ch.size() && !f; j++) if (!used[j] && shapeEq(x, b->ch[j], depth + 1, exact)) { used[j] = 1; f = true; } if (!f) return false; }
      return true;
    }
    for (size_t i = 0; i < a->ch.size(); i++) if (!shapeEq(a->ch[i], b->ch[i], depth + 1, exact)) return false;
    if (a->k == EK::Poly && exact) {
      if (a->monos.size() != b->monos.size()) return false;
      for (auto& m : a->monos) { bool f = false; for (auto& m2 : b->monos) if (monoEq(m, m2, depth + 1, true) && !m.coefs.empty() && !m2.coefs.empty() && m.coefs.back() == m2.coefs.back()) { f = true; break; } if (!f) return false; }
      return true;
    }
    // leniently, two polynomials always have one shape: a monomial one lacks has coefficient 0 there. A wrong
    // association is caught by the affine fit (the coefficients would not be consistent across the samples).
    return true;
  }
  bool monoEq(const Node::Mono& a, const Node::Mono& b, int depth, bool exact = false) {
    if (a.atoms.size() != b.atoms.size()) return false;
    for (size_t i = 0; i < a.atoms.size(); i++) if (!shapeEq(a.atoms[i], b.atoms[i], depth, exact)) return false;
    return true;
  }
  // how alike two subtrees of one shape are: the monomials (by atom shape) they share, over the whole subtree
  unsigned similarity(Node* a, Node* b, int depth = 0) {
    if (depth > 64 || !sameHead(a, b)) return 0;
    unsigned n = 0;
    // (a monomial whose coefficient is 0 in every sample so far is one the polynomial does not have)
    auto live = [](const Node::Mono& m) { if (m.atoms.empty()) return false; for (int64_t c : m.coefs) if (c) return true; return false; };
    if (a->k == EK::Poly) for (auto& m : a->monos) { if (!live(m)) continue; for (auto& m2 : b->monos) if (live(m2) && monoEq(m, m2, depth + 1)) { n++; break; } }
    for (size_t i = 0; i < a->ch.size() && i < b->ch.size(); i++) n += similarity(a->ch[i], b->ch[i], depth + 1);
    return n;
  }
  // the template literal (a child of the condition u, not yet used) the sample's literal x is: an exact match
  // (constants and coefficients too), else the one of x's shape sharing the most monomials with it - two
  // `poly < 0` literals have one shape, so `n0 < N` and `r < M` are told apart by their atoms (a literal sharing
  // none is a new literal of the condition: `8 < M` is not `S - 2 > 0` with a coefficient of 0 on S). -1: none; -2: ambiguous
  int matchLiteral(Node* u, const std::vector<char>& used, Node* x, int depth, bool strict = true) {
    for (size_t j = 0; j < u->ch.size(); j++) if (!used[j] && shapeEq(u->ch[j], x, depth + 1, true)) return (int)j;
    int hit = -1; unsigned best = 0; bool tie = false;
    for (size_t j = 0; j < u->ch.size(); j++) {
      if (used[j] || !shapeEq(u->ch[j], x, depth + 1)) continue;
      const unsigned sc = similarity(u->ch[j], x, depth + 1);
      if (strict && !sc) continue;
      if (hit < 0 || sc > best) { hit = (int)j; best = sc; tie = false; } else if (sc == best) tie = true;
    }
    if (tie) { why = "two literals of one condition have the same shape (" + desc(u) + " <- " + desc(x) + ")"; return -2; }
    return hit;
  }
  // merge sample tree t (sample index s) into template u
  bool unify(Node* u, Node* t, unsigned s, int depth = 0) {
    if (depth > 64) { why = "expression too deep"; return false; }
    if (u->k == EK::Ite && t->k != EK::Ite) {
      // the sample has one arm of the template's selection (its condition was decided there)
      for (int arm = 2; arm >= 1; arm--) if (shapeEq(u->ch[arm], t, depth + 1)) return unify(u->ch[arm], t, s, depth + 1);
      why = "an element's expression is neither arm of the selection other elements have (" + desc(u) + " <- " + desc(t) + ")"; return false;
    }
    if (t->k == EK::Ite && u->k != EK::Ite) {
      // the template has one arm of the sample's selection: the template becomes the selection
      for (int arm = 2; arm >= 1; arm--) if (shapeEq(u, t->ch[arm], depth + 1)) {
        Node* old = mk(); *old = *u;
        *u = Node(); u->k = EK::Ite; u->ty = t->ty;
        u->ch = {t->ch[0], t->ch[1], t->ch[2]}; u->ch[arm] = old;
        u->ch[0] = adopt(t->ch[0], s); u->ch[arm == 2 ? 1 : 2] = adopt(t->ch[arm == 2 ? 1 : 2], s);
        return unify(old, t->ch[arm], s, depth + 1);
      }
      why = "an element's expression is neither arm of the selection another element has (" + desc(u) + " <- " + desc(t) + ")"; return false;
    }
    // A condition: an unordered set of literals. An element may lack literals other elements carry (row 0's
    // `0 < M` is decided by the facts at the K loop and dropped from its expression; row 8's `8 < M` stays): the
    // template holds the union, a literal absent from an element has no value there, and the proof instantiates
    // every literal at every element and folds the ones the facts decide (foldUnder). A single literal against a
    // conjunction is the one-literal case of the same.
    auto isAnd = [](const Node* n) { return n->assoc && n->k == EK::Op && n->op == Instruction::And; };
    auto isLit = [](const Node* n) { return n->ty && n->ty->isIntegerTy(1) && n->k != EK::Ite && !n->assoc && (n->k == EK::Cmp || n->k == EK::Op || n->k == EK::Sym); };
    // (two lone literals of one shape sharing no monomial - row 2's `2 < M` and iteration 2's `S - 2 > 0` - are two
    // literals of one condition, each absent from the other's element)
    const bool twoLits = isLit(u) && isLit(t) && shapeEq(u, t, depth + 1) && !shapeEq(u, t, depth + 1, true) && !similarity(u, t, depth + 1);
    if ((isAnd(u) && (isAnd(t) || isLit(t))) || (isAnd(t) && isLit(u)) || twoLits) {
      if (!isAnd(u)) { Node* old = mk(); *old = *u; *u = Node(); u->k = EK::Op; u->op = Instruction::And; u->ty = t->ty; u->assoc = true; u->ch = {old}; }
      std::vector<Node*> tch; if (isAnd(t)) tch = t->ch; else tch.push_back(t);
      std::vector<char> used(u->ch.size(), 0);
      for (Node* x : tch) {
        int hit = matchLiteral(u, used, x, depth);
        if (hit == -2) return false;
        if (hit < 0) { u->ch.push_back(adopt(x, s)); used.push_back(1); continue; }
        used[hit] = 1;
        if (!unify(u->ch[hit], x, s, depth + 1)) return false;
      }
      return true;
    }
    if (!sameHead(u, t)) { why = "operand expressions of two elements have different shapes at " + desc(u) + " <- " + desc(t); return false; }
    if (u->assoc) {
      // conditions: an unordered set of literals, each of the sample's matched to one of the template's
      std::vector<char> used(u->ch.size(), 0);
      for (Node* x : t->ch) {
        int hit = matchLiteral(u, used, x, depth);
        if (hit == -1) hit = matchLiteral(u, used, x, depth, false);
        if (hit == -2) return false;
        if (hit < 0) { why = "a literal of a condition has no counterpart in another element's (" + desc(u) + " <- " + desc(t) + ")"; return false; }
        used[hit] = 1;
        if (!unify(u->ch[hit], x, s, depth + 1)) return false;
      }
      return true;
    }
    for (size_t i = 0; i < u->ch.size(); i++) if (!unify(u->ch[i], t->ch[i], s, depth + 1)) return false;
    if (u->hasVal()) put(u->vals, u->present, s, t->vals.back());
    if (u->k == EK::Poly) {
      for (auto& m : t->monos) {
        int hit = -1;
        { int exact = -1, nExact = 0; for (size_t j = 0; j < u->monos.size(); j++) if (monoEq(u->monos[j], m, depth + 1, true)) { exact = (int)j; nExact++; } if (nExact == 1) hit = exact; else if (nExact > 1) { why = "two monomials of one polynomial are identical up to their coefficients"; return false; } }
        if (hit < 0) for (size_t j = 0; j < u->monos.size(); j++) if (monoEq(u->monos[j], m, depth + 1)) { if (hit >= 0) { why = "two monomials of one polynomial have the same shape (" + desc(u) + " <- " + desc(t) + ")"; return false; } hit = (int)j; }
        if (hit < 0) { Node::Mono nm; for (Node* a : m.atoms) nm.atoms.push_back(adopt(a, s)); put(nm.coefs, nm.present, s, m.coefs.back()); u->monos.push_back(std::move(nm)); continue; }
        Node::Mono& um = u->monos[hit];
        put(um.coefs, um.present, s, m.coefs.back());
        for (size_t i = 0; i < um.atoms.size(); i++) if (!unify(um.atoms[i], m.atoms[i], s, depth + 1)) return false;
      }
      // a monomial the sample lacks has coefficient 0 there
      for (auto& um : u->monos) if (um.coefs.size() <= s) put(um.coefs, um.present, s, 0);
    }
    return true;
  }
  // a sample subtree (single hole values) joins the template: its values move to the sample's index
  Node* adopt(Node* t, unsigned s) {
    adoptedAny = true;
    // (the sample tree holds its value at index 0; it is sample s's, no other sample's)
    if (t->hasVal()) { int64_t v = t->vals.back(); t->vals.clear(); t->present.clear(); put(t->vals, t->present, s, v); }
    for (auto& m : t->monos) { int64_t v = m.coefs.back(); m.coefs.clear(); m.present.clear(); put(m.coefs, m.present, s, v); for (Node* a : m.atoms) adopt(a, s); }
    for (Node* c : t->ch) adopt(c, s);
    return t;
  }
  std::string desc(Node* n, int depth = 0) {
    if (depth > 6) return "..";
    std::string o;
    switch (n->k) {
      case EK::Const: o = n->isIntConst() ? std::to_string(n->vals.empty() ? 0 : n->vals.back()) : "Kf"; break;
      case EK::Bytes: o = "bytes[" + std::to_string(n->vals.empty() ? 0 : n->vals.back()) + ":" + std::to_string(n->len) + "](" + desc(n->ch[0], depth + 1) + ")"; break;
      case EK::Sym: o = "s" + std::to_string(n->sym); break;
      case EK::Poly: o = "{"; for (auto& m : n->monos) { o += std::to_string(m.coefs.empty() ? 0 : m.coefs.back()); for (Node* a : m.atoms) o += "*" + desc(a, depth + 1); o += " + "; } o += "}"; break;
      default: o = std::to_string((int)n->k) + ":" + std::to_string(n->op) + "("; for (Node* c : n->ch) o += desc(c, depth + 1) + ","; o += ")";
    }
    return o;
  }
  // number the holes of the template (DFS order), padding every hole to nSamples
  void holes(Node* u) {
    for (Node* c : u->ch) holes(c);
    if (u->hasVal()) { u->vals.resize(nSamples, 0); u->present.resize(nSamples, 0); u->hole = (int)holeList.size(); holeList.push_back({&u->vals, &u->present}); }
    if (u->k == EK::Poly) for (auto& m : u->monos) { m.coefs.resize(nSamples, 0); m.present.resize(nSamples, 0); m.hole = (int)holeList.size(); holeList.push_back({&m.coefs, &m.present}); for (Node* a : m.atoms) holes(a); }
  }
  // The template is a tree built from the samples' expression DAGs: a sub-expression the DAG shared (a row's base
  // address, under every word of the row) is one subtree per occurrence, and every occurrence instantiates alike.
  // classify() numbers the subtrees by shape and fitted coefficients (their class); inst() instantiates each class
  // once per call, and once across calls for each vector of values its coefficients take (a subtree over the row
  // alone is the same expression for every k): the replay of an occurrence would build the very nodes the first
  // one did.
  std::map<std::string, int> classIds;
  std::map<std::pair<int64_t, std::vector<int64_t>>, int> affineIds;   // distinct fitted coefficient functions
  std::vector<std::vector<int>> classAffines;   // class -> the affine ids in its subtree (sorted)
  std::vector<EP> instMemo;   // class -> its instantiation at the coordinates of the inst() call in progress
  std::vector<int64_t> affineVals;   // affine id -> its value at those coordinates
  std::unordered_map<std::string, EP> instCache;   // (class, its affines' values) -> instantiation, across calls
  int classify(Node* u, const std::vector<Affine>& fit) {
    std::string k; std::vector<int> affs;
    auto raw = [&](const void* p, size_t n) { k.append((const char*)p, n); };
    auto i64 = [&](int64_t v) { raw(&v, sizeof v); };
    auto affine = [&](int h) { const Affine& a = fit[h]; auto key = std::make_pair(a.c0, a.c); auto it = affineIds.find(key); if (it == affineIds.end()) it = affineIds.emplace(key, (int)affineIds.size()).first; i64(it->second); affs.push_back(it->second); };
    auto child = [&](Node* c) { int cc = classify(c, fit); i64(cc); affs.insert(affs.end(), classAffines[cc].begin(), classAffines[cc].end()); };
    raw(&u->k, sizeof u->k); raw(&u->op, sizeof u->op); raw(&u->lo, sizeof u->lo); raw(&u->len, sizeof u->len); raw(&u->ty, sizeof u->ty);
    raw(&u->callee, sizeof u->callee); raw(&u->asmv, sizeof u->asmv); raw(&u->sym, sizeof u->sym); k.push_back(u->assoc ? 1 : 0);
    if (u->k == EK::Const && !u->isIntConst()) { i64(u->fc.getBitWidth()); for (unsigned w = 0; w < u->fc.getNumWords(); w++) i64((int64_t)u->fc.getRawData()[w]); }
    if (u->hasVal()) affine(u->hole);
    for (auto& m : u->monos) { k.push_back('m'); affine(m.hole); for (Node* a : m.atoms) child(a); }
    for (Node* c : u->ch) { k.push_back('c'); child(c); }
    auto it = classIds.find(k);
    if (it == classIds.end()) {
      it = classIds.emplace(std::move(k), (int)classIds.size()).first;
      std::sort(affs.begin(), affs.end()); affs.erase(std::unique(affs.begin(), affs.end()), affs.end());
      classAffines.push_back(std::move(affs));
    }
    return u->cls = it->second;
  }
  // the template instantiated with the fitted holes at coordinates `co`
  EP inst(Node* u, const std::vector<Affine>& fit, const std::vector<int64_t>& co) {
    if (u->cls < 0) classify(u, fit);
    instMemo.assign(classIds.size(), nullptr);
    affineVals.clear();
    for (auto& kv : affineIds) { int64_t v = kv.first.first; for (size_t i = 0; i < kv.first.second.size(); i++) v += kv.first.second[i] * co[i]; if ((size_t)kv.second >= affineVals.size()) affineVals.resize(kv.second + 1); affineVals[kv.second] = v; }
    return instRec(u, fit, co);
  }
  EP instRec(Node* u, const std::vector<Affine>& fit, const std::vector<int64_t>& co) {
    if (EP m = instMemo[u->cls]) return m;
    std::string ck((const char*)&u->cls, sizeof u->cls);
    for (int a : classAffines[u->cls]) ck.append((const char*)&affineVals[a], sizeof(int64_t));
    auto it = instCache.find(ck);
    if (it != instCache.end()) return instMemo[u->cls] = it->second;
    EP r = instRaw(u, fit, co);
    instCache.emplace(std::move(ck), r);
    return instMemo[u->cls] = r;
  }
  EP instRaw(Node* u, const std::vector<Affine>& fit, const std::vector<int64_t>& co) {
    auto val = [&](int h) { const Affine& a = fit[h]; int64_t v = a.c0; for (size_t i = 0; i < a.c.size(); i++) v += a.c[i] * co[i]; return v; };
    switch (u->k) {
      case EK::Const: return !u->isIntConst() ? A.mkConstFP(u->ty, u->fc) : u->ty->isIntegerTy(1) ? A.mkBool(val(u->hole) != 0) : A.mkInt(u->ty, val(u->hole));
      case EK::Sym: return A.mkSym(A.symbol(u->sym));
      case EK::Poly: {
        EP acc = A.mkPoly(u->ty, {});
        for (auto& m : u->monos) { EP term = A.mkInt(u->ty, val(m.hole)); for (Node* a : m.atoms) term = A.mul(term, A.retype(A.asPoly(instRec(a, fit, co)), u->ty)); acc = A.add(acc, term); }
        return acc;
      }
      default: break;
    }
    std::vector<EP> args; for (Node* c : u->ch) args.push_back(instRec(c, fit, co));
    switch (u->k) {
      case EK::Ite: return A.ite(args[0], args[1], args[2]);
      case EK::Cmp: return A.cmp(u->op, args[0], args[1]);
      case EK::Bytes: { int64_t lo = val(u->hole); if (lo < 0 || (unsigned)lo + u->len > A.sizeOf(args[0]->ty)) return A.mkUndef(u->ty); return A.bytes(args[0], (unsigned)lo, u->len); }
      case EK::Concat: return A.concat(std::move(args), u->ty);
      case EK::Load: return A.load(u->ty, args[0], args[1], u->len);
      case EK::Call: return u->asmv ? A.callAsm(u->asmv, std::move(args), u->ty) : A.call(u->callee, std::move(args), u->ty);
      case EK::Op:
        if (u->assoc) { EP r = args[0]; for (size_t i = 1; i < args.size(); i++) r = u->op == Instruction::And ? A.andb(r, args[i]) : A.orb(r, args[i]); return r; }
        if (args.size() == 1) return Instruction::isCast(u->op) ? A.cast(u->op, args[0], u->ty) : A.unop(u->op, args[0], u->ty);
        return A.binop(u->op, args[0], args[1], u->ty);
      default: return A.mkUndef(u->ty);
    }
  }
  // the template instantiated at symbolic coordinates `co` (i64 expressions): the element function itself. A byte
  // offset that is not constant at these coordinates cannot be instantiated (ok = false).
  EP instSym(Node* u, const std::vector<Affine>& fit, const std::vector<EP>& co, bool& ok) {
    Type* i64 = co[0]->ty;
    auto val = [&](int h) { const Affine& a = fit[h]; EP v = A.mkInt(i64, a.c0); for (size_t i = 0; i < a.c.size(); i++) if (a.c[i]) v = A.add(v, A.mul(A.mkInt(i64, a.c[i]), co[i])); return v; };
    switch (u->k) {
      case EK::Const: {
        if (!u->isIntConst()) return A.mkConstFP(u->ty, u->fc);
        EP v = val(u->hole);
        if (u->ty->isIntegerTy(1)) return A.cmp(CmpInst::ICMP_NE, v, A.mkInt(i64, 0));
        return A.retype(v, u->ty);
      }
      case EK::Sym: return A.mkSym(A.symbol(u->sym));
      case EK::Poly: {
        EP acc = A.mkPoly(u->ty, {});
        for (auto& m : u->monos) { EP term = A.retype(val(m.hole), u->ty); for (Node* a : m.atoms) term = A.mul(term, A.retype(A.asPoly(instSym(a, fit, co, ok)), u->ty)); acc = A.add(acc, term); }
        return acc;
      }
      default: break;
    }
    std::vector<EP> args; for (Node* c : u->ch) args.push_back(instSym(c, fit, co, ok));
    switch (u->k) {
      case EK::Ite: return A.ite(args[0], args[1], args[2]);
      case EK::Cmp: return A.cmp(u->op, args[0], args[1]);
      case EK::Bytes: {
        EP loE = val(u->hole); int64_t lo;
        if (A.constInt(loE, lo)) { if (lo < 0 || (unsigned)lo + u->len > A.sizeOf(args[0]->ty)) { ok = false; return A.mkUndef(u->ty); } return A.bytes(args[0], (unsigned)lo, u->len); }
        // a byte offset that varies with the coordinates (a lane's position within a staged chunk): of a load, the
        // load of those bytes at their address (aligned as far as the offset's coefficients keep the load's
        // alignment); of a register value, a shift
        EP x = A.unwrap(args[0]);
        if (x->k == EK::Load) {
          unsigned al = std::max(1u, x->len);
          for (auto& t : A.asPoly(loE)->terms) if (t.second) al = (unsigned)std::gcd((int64_t)al, std::llabs(t.second));
          EP addr = x->args[0];
          return A.load(u->ty, A.add(addr, A.retype(loE, addr->ty)), x->args[1], std::min(al, 16u));
        }
        if (!args[0]->ty->isIntegerTy()) { ok = false; return A.mkUndef(u->ty); }
        EP sh = A.binop(Instruction::LShr, args[0], A.mul(A.mkInt(args[0]->ty, 8), A.retype(loE, args[0]->ty)), args[0]->ty);
        Type* bt = Type::getIntNTy(u->ty->getContext(), 8 * u->len);
        EP r = A.sizeOf(args[0]->ty) == u->len ? sh : A.cast(Instruction::Trunc, sh, bt);
        return u->ty == bt ? r : A.cast(Instruction::BitCast, r, u->ty);
      }
      case EK::Concat: return A.concat(std::move(args), u->ty);
      case EK::Load: return A.load(u->ty, args[0], args[1], u->len);
      case EK::Call: return u->asmv ? A.callAsm(u->asmv, std::move(args), u->ty) : A.call(u->callee, std::move(args), u->ty);
      case EK::Op:
        if (u->assoc) { EP r = args[0]; for (size_t i = 1; i < args.size(); i++) r = u->op == Instruction::And ? A.andb(r, args[i]) : A.orb(r, args[i]); return r; }
        if (args.size() == 1) return Instruction::isCast(u->op) ? A.cast(u->op, args[0], u->ty) : A.unop(u->op, args[0], u->ty);
        return A.binop(u->op, args[0], args[1], u->ty);
      default: ok = false; return A.mkUndef(u->ty);
    }
  }
};

// ---------------------------------------------------------------- expression -> IR

// Emits an expression at the builder's insertion point. A selection with a load in an arm becomes control flow (the
// load must not execute when the other arm is taken: the selection is the operand's zero-fill guard), other
// selections become selects. Coordinates are bound to values by the caller. Values emitted inside a branch arm are
// forgotten at the join (they do not dominate what follows).
struct Materializer {
  Arena& A; Function& F; Module& M; IRBuilder<>& B; LLVMContext& C;
  std::map<uint32_t, Value*> coord;   // Coord symbol id -> value
  std::map<EP, Value*> memo;
  std::vector<EP> trail;
  std::map<EP, bool> loadMemo;
  std::string why;
  bool speculate = false;
  std::vector<Value*> specConds;
  std::map<EP, EP> need;
  bool useNeeds = false;
  Value* zeroPage = nullptr;
  int64_t relocCut = INT64_MAX, relocShrink = 0;
  std::function<bool(EP, int64_t&, int64_t&)> range;
  std::vector<EP> pathE;    // the path conditions of the selections being emitted (as collectNeeds() builds them)
  EP pathCond() { return pathE.empty() ? A.mkBool(true) : pathE.back(); }
  void collectNeeds(EP root) {
    std::set<std::pair<EP, EP>> seen;
    std::function<void(EP, EP)> go = [&](EP e, EP path) {
      if (!seen.insert({e, path}).second) return;
      if (e->k == EK::Load) { auto it = need.find(e); need[e] = it == need.end() ? path : A.orb(it->second, path); }
      if (e->k == EK::Ite && speculate && (hasLoad(e->args[1]) || hasLoad(e->args[2]))) {
        go(e->args[0], path);
        go(e->args[1], A.andb(path, e->args[0]));
        go(e->args[2], A.andb(path, A.notb(e->args[0])));
        return;
      }
      for (EP a : e->args) go(a, path);
      if (e->k == EK::Poly) for (auto& t : e->terms) for (uint32_t a : t.first) go(A.atomExpr(a), path);
    };
    go(root, A.mkBool(true));
    useNeeds = true;
  }
  DominatorTree* DT = nullptr; BasicBlock* curBB = nullptr;
  std::map<EP, BasicBlock*> memoBB;
  std::vector<EP> emittedPolys;   // integer polynomials emitted so far (candidates for emission by difference)
  struct PtrEmit { EP e; uint32_t baseAtom; EP rest; };   // a pointer polynomial emitted: its base atom and offset polynomial
  std::vector<PtrEmit> emittedPtrs;
  static size_t diffTermCount(EP a, EP b) {   // monomials of a - b (both sorted by atoms)
    size_t n = 0, i = 0, j = 0;
    while (i < a->terms.size() || j < b->terms.size()) {
      if (j == b->terms.size() || (i < a->terms.size() && a->terms[i].first < b->terms[j].first)) { n++; i++; }
      else if (i == a->terms.size() || b->terms[j].first < a->terms[i].first) { n++; j++; }
      else { if (a->terms[i].second != b->terms[j].second) n++; i++; j++; }
    }
    return n;
  }
  bool live(EP q) {   // memoized and usable at the current site
    if (!memo.count(q)) return false;
    if (!DT || !curBB) return true;
    auto it = memoBB.find(q); return it != memoBB.end() && DT->dominates(it->second, curBB);
  }
  Materializer(Arena& A, Function& F, IRBuilder<>& B) : A(A), F(F), M(*F.getParent()), B(B), C(F.getContext()) {}
  // an address inside a shared-memory object (a staged operand the retiler keeps there): always in bounds, never
  // clamped
  bool sharedAddr(EP addr) {
    if (addr->k == EK::Op && Instruction::isCast(addr->op)) return sharedAddr(addr->args[0]);
    if (addr->k == EK::Sym) return A.symbol(addr->sym).kind == Symbol::SharedBase;
    if (addr->k != EK::Poly) return false;
    for (auto& t : addr->terms) if (t.first.size() == 1 && t.second == 1 && sharedAddr(A.atomExpr(t.first[0]))) return true;
    return false;
  }
  // the pointer argument an address is an offset from (the clamp target), or null
  Value* clampBase(EP addr) {
    if (addr->k == EK::Sym) return addr->ty->isPointerTy() && A.symbol(addr->sym).kind == Symbol::Arg ? A.symbol(addr->sym).v : nullptr;
      if (addr->k == EK::Ite) {   // (c ? p : p + off); or (c ? 0 : p + off), the null arm never loaded by the source
        Value* a = clampBase(addr->args[1]); Value* b = clampBase(addr->args[2]); int64_t c;
        if (a && b) return a == b ? a : nullptr;
        if (a && A.constInt(addr->args[2], c) && c == 0) return a;
        if (b && A.constInt(addr->args[1], c) && c == 0) return b;
        return nullptr;
      }
    if (addr->k == EK::Op && Instruction::isCast(addr->op)) return clampBase(addr->args[0]);
    if (addr->k != EK::Poly) return nullptr;
    for (auto& t : addr->terms) {
      if (t.first.size() != 1 || t.second != 1) continue;
      if (Value* b = clampBase(A.atomExpr(t.first[0]))) return b;
    }
    return nullptr;
  }
  // the conditions under which a selected address is the null constant (a zero-fill copy's source: never read)
  void nullArms(EP addr, std::vector<std::pair<EP, bool>>& out) {
    if (addr->k == EK::Op && Instruction::isCast(addr->op)) return nullArms(addr->args[0], out);
    if (addr->k != EK::Ite) return;
    int64_t c;
    if (A.constInt(addr->args[1], c) && c == 0) out.push_back({addr->args[0], true}); else nullArms(addr->args[1], out);
    if (A.constInt(addr->args[2], c) && c == 0) out.push_back({addr->args[0], false}); else nullArms(addr->args[2], out);
  }
  bool contextBound = false;   // set when a load of the value being built was clamped by its arm (not by a need condition)
  Value* clamp(EP loadE, EP addr, Value* p) {
    std::vector<std::pair<EP, bool>> nulls; nullArms(addr, nulls);
    EP nd = nullptr;
    if (useNeeds) { auto it = need.find(loadE); if (it != need.end() && !hasLoad(it->second)) nd = it->second; }
    if (nd ? (A.isTrue(nd) && nulls.empty()) : (specConds.empty() && nulls.empty())) return p;
    if (sharedAddr(addr)) return p;
    Value* base = clampBase(addr); if (!base) return fail("a speculated load's address is not an offset from a pointer argument");
    // the clamp target: the zero page when there is one (the value is unused, or wanted zero - zeroClamp), else the
    // argument itself
    if (zeroPage) base = zeroPage;
    Value* pred = nullptr;
    if (nd) { if (!A.isTrue(nd)) { pred = val(nd); if (!pred) return nullptr; } }
    else { for (Value* c : specConds) pred = pred ? B.CreateAnd(pred, c) : c; if (!specConds.empty()) contextBound = true; }
    // an address that is null under a condition reads the base there (the source read nothing; the value is not used)
    for (auto& n : nulls) { Value* c = val(n.first); if (!c) return nullptr; Value* notNull = n.second ? B.CreateNot(c) : c; pred = pred ? B.CreateAnd(pred, notNull) : notNull; }
    // A word at a small constant offset from a pointer emitted before (the words of one row: pointer differences,
    // build()) clamps that pointer and keeps the offset - one select per row instead of one per word - when the
    // offset stays inside the zero page.
    if (base == zeroPage) if (auto* G = dyn_cast<GetElementPtrInst>(p)) if (G->getNumIndices() == 1 && G->getSourceElementType() == intTy(8)) if (auto* CI = dyn_cast<ConstantInt>(G->getOperand(1))) {
      const int64_t d = CI->getSExtValue();
      if (d >= 0 && d + (int64_t)A.sizeOf(loadE->ty) <= 4096) return B.CreateGEP(intTy(8), clampSelect(pred, G->getPointerOperand(), base), CI);
    }
    return clampSelect(pred, p, base);
  }
  std::map<std::pair<Value*, Value*>, std::pair<Value*, BasicBlock*>> clampMemo;   // (pointer, predicate) -> its clamp select, and where
  Value* clampSelect(Value* pred, Value* p, Value* base) {
    auto it = clampMemo.find({p, pred});
    if (it != clampMemo.end() && (!DT || !curBB || DT->dominates(it->second.second, curBB))) return it->second.first;
    if (base->getType() != p->getType()) base = B.CreatePointerBitCastOrAddrSpaceCast(base, p->getType());
    Value* s = B.CreateSelect(pred, p, base);
    clampMemo[{p, pred}] = {s, curBB ? curBB : B.GetInsertBlock()};
    return s;
  }

  bool isZeroExpr(EP e) {
    int64_t c; if (A.constInt(e, c)) return c == 0;
    if (e->k == EK::Const) return e->c.isZero();
    if (e->k == EK::Concat) { for (EP p : e->args) if (!isZeroExpr(p)) return false; return true; }
    if (e->k == EK::Bytes) return isZeroExpr(e->args[0]);
    return false;
  }
  bool hasLoad(EP e) {
    auto it = loadMemo.find(e); if (it != loadMemo.end()) return it->second;
    bool r = e->k == EK::Load;
    for (size_t i = 0; i < e->args.size() && !r; i++) r = hasLoad(e->args[i]);
    if (!r && e->k == EK::Poly) for (auto& t : e->terms) for (uint32_t a : t.first) if (hasLoad(A.atomExpr(a))) { r = true; break; }
    return loadMemo[e] = r;
  }
  size_t mark() const { return trail.size(); }
  void release(size_t m) { while (trail.size() > m) { memo.erase(trail.back()); trail.pop_back(); } }
  bool no(const std::string& w) { if (why.empty()) why = w; return false; }
  Value* fail(const std::string& w) { no(w); return nullptr; }
  // whether build() can emit the expression (the same conditions, without emitting): checked before the IR changes
  std::map<EP, bool> canMemo;
  std::set<uint32_t> bound;   // coordinates the emission will bind
  bool can(EP e) {
    auto it = canMemo.find(e); if (it != canMemo.end()) return it->second;
    bool r = true;
    switch (e->k) {
      case EK::Const: r = e->ty->isIntegerTy() || e->ty->isPointerTy() || e->ty->isFloatingPointTy() || no("a constant of an unsupported type"); break;
      case EK::Undef: break;
      case EK::Sym: {
        const Symbol& s = A.symbol(e->sym);
        if (s.kind == Symbol::Coord) r = bound.count(e->sym) || coord.count(e->sym) || no("an unbound coordinate");
        else r = s.kind == Symbol::Arg || s.kind == Symbol::LoopValue || s.kind == Symbol::Sreg || s.kind == Symbol::SharedBase || s.kind == Symbol::Global || no("a per-thread or unknown value (" + A.str(e, 3) + ")");
        break;
      }
      case EK::Poly: for (auto& t : e->terms) for (uint32_t a : t.first) if (!can(A.atomExpr(a))) { r = false; break; } break;
      case EK::Load: r = (A.isTrue(e->args[1]) || no("a guarded load")) && can(e->args[0]) && (!speculate || sharedAddr(e->args[0]) || clampBase(e->args[0]) || no("a speculated load's address is not an offset from a pointer argument")); break;
      case EK::Concat: { unsigned total = 0; for (EP a : e->args) { if (!can(a)) { r = false; break; } total += A.sizeOf(a->ty); } if (r) r = total == A.sizeOf(e->ty) || no("a concatenation of the wrong width"); break; }
      case EK::Ite: case EK::Cmp: case EK::Op: case EK::Bytes: case EK::Call: for (EP a : e->args) if (!can(a)) { r = false; break; } break;
      default: r = no("an expression kind the materializer does not emit");
    }
    return canMemo[e] = r;
  }
  BasicBlock* newBlock(const char* nm) { return BasicBlock::Create(C, nm, &F); }
  Type* intTy(unsigned bits) { return Type::getIntNTy(C, bits); }
  std::map<std::pair<Value*, Type*>, std::pair<Value*, BasicBlock*>> castMemo;   // toInt results (an atom is widened once per site chain)
  Value* toInt(Value* v, Type* ty) {   // integer/pointer/float value -> integer type ty (sign-extending)
    if (v->getType() == ty) return v;
    Value* in = v;
    auto it = castMemo.find({in, ty});
    if (it != castMemo.end() && (!DT || !curBB || (it->second.second && DT->dominates(it->second.second, curBB)))) return it->second.first;
    if (v->getType()->isPointerTy()) v = B.CreatePtrToInt(v, intTy(64));
    else if (!v->getType()->isIntegerTy()) v = B.CreateBitCast(v, intTy(8 * A.sizeOf(v->getType())));
    Value* r = v->getType() == ty ? v : B.CreateSExtOrTrunc(v, ty);
    if (speculate && specConds.empty() && isa<Instruction>(r)) castMemo[{in, ty}] = {r, curBB};   // (branch mode: arms are forgotten at the join)
    return r;
  }
  Value* val(EP e) {
    auto it = memo.find(e);
    if (it != memo.end()) {
      if (!DT || !curBB) return it->second;
      auto bb = memoBB.find(e);
      if (bb != memoBB.end() && DT->dominates(bb->second, curBB)) return it->second;
      memo.erase(it);   // emitted for a site that does not dominate this one: rebuilt here
    }
    const bool outer = contextBound; contextBound = false;
    Value* v = build(e);
    const bool ctx = contextBound; contextBound = outer || ctx;
    if (!v) return nullptr;
    // a value with loads clamped by the arm they were built under is not shared with other contexts (loads clamped
    // by their need condition are context-free)
    if (!(speculate && !specConds.empty() && hasLoad(e) && ctx)) { memo[e] = v; trail.push_back(e); if (curBB) memoBB[e] = curBB; }
    return v;
  }
  Value* build(EP e) {
    Type* ty = e->ty;
    switch (e->k) {
      case EK::Const:
        if (ty->isIntegerTy()) return ConstantInt::get(ty, e->c);
        if (ty->isPointerTy()) return e->c.isZero() ? (Value*)ConstantPointerNull::get(cast<PointerType>(ty)) : B.CreateIntToPtr(ConstantInt::get(intTy(64), e->c.getSExtValue()), ty);
        if (ty->isFloatingPointTy()) return ConstantFP::get(ty, APFloat(ty->getFltSemantics(), e->c));
        return fail("a constant of an unsupported type");
      case EK::Undef: return UndefValue::get(ty);
      case EK::Sym: {
        const Symbol& s = A.symbol(e->sym);
        switch (s.kind) {
          case Symbol::Arg: return s.v;
          case Symbol::LoopValue: {   // a loop's header phi or an opaque loop's result: the value itself, which must reach the site
            if (auto* VI = dyn_cast<Instruction>(s.v)) if (DT && curBB && VI->getParent() != curBB && !DT->dominates(VI->getParent(), curBB)) return fail("a loop value does not reach the fill site (" + A.str(e, 3) + ")");
            return s.v;
          }
          case Symbol::Sreg: return B.CreateCall(M.getOrInsertFunction(s.name, FunctionType::get(ty, false)));
          case Symbol::SharedBase: case Symbol::Global: { Value* g = s.v; return g->getType() == ty ? g : B.CreatePointerBitCastOrAddrSpaceCast(g, ty); }
          case Symbol::Coord: { auto c = coord.find(e->sym); if (c == coord.end()) return fail("an unbound coordinate"); return toInt(c->second, ty); }
          default: return fail("a per-thread or unknown value (" + A.str(e, 3) + ")");
        }
      }
      case EK::Poly: {
        // A polynomial is emitted as a chain of partial sums, each an interned polynomial of its own (memoized):
        // base + [monomials in the arena's order] + constant. The words of one operand differ in their constant
        // terms (and the monomials of a lane's k-position), so their addresses share one leading sum instead of
        // re-evaluating every monomial; a monomial c * a1 * ... * an is c times the memoized product.
        Type* ity = ty->isPointerTy() ? intTy(64) : ty;
        if (ty->isPointerTy()) {
          // base pointer atom + the integer polynomial of the rest
          Value* base = nullptr; uint32_t baseAtom = ~0u;
          std::vector<std::pair<std::vector<uint32_t>, int64_t>> rest;
          for (auto& t : e->terms) {
            if (!base && t.second == 1 && t.first.size() == 1 && A.atomExpr(t.first[0])->ty->isPointerTy()) { base = val(A.atomExpr(t.first[0])); if (!base) return nullptr; baseAtom = t.first[0]; continue; }
            rest.push_back(t);
          }
          EP restE = A.mkPoly(ity, rest);
          if (base) {
            // the nearest pointer already emitted from the same base (and live here) plus the difference of the
            // offsets, when that has fewer terms than the offset (the words of one operand: one add each instead
            // of the offset's sum and the base's)
            const PtrEmit* q = nullptr; size_t bestCost = rest.size();
            for (const PtrEmit& cand : emittedPtrs) {
              if (cand.baseAtom != baseAtom || cand.e->ty != ty || cand.e == e || !live(cand.e)) continue;
              const size_t cost = diffTermCount(restE, cand.rest);
              if (cost < bestCost) { bestCost = cost; q = &cand; }
            }
            if (q) {
              Value* qp = val(q->e); if (!qp) return nullptr;
              Value* d = val(A.sub(restE, q->rest)); if (!d) return nullptr;
              emittedPtrs.push_back({e, baseAtom, restE});
              return isa<ConstantInt>(d) && cast<ConstantInt>(d)->isZero() ? qp : B.CreateGEP(intTy(8), qp, d);
            }
          }
          Value* acc = rest.empty() ? ConstantInt::get(ity, 0) : val(restE);
          if (!acc) return nullptr;
          if (!base) return B.CreateIntToPtr(acc, ty);
          if (base->getType() != ty) base = B.CreatePointerBitCastOrAddrSpaceCast(base, ty);
          emittedPtrs.push_back({e, baseAtom, restE});
          return isa<ConstantInt>(acc) && cast<ConstantInt>(acc)->isZero() ? base : B.CreateGEP(intTy(8), base, acc);
        }
        if (e->terms.empty()) return ConstantInt::get(ity, 0);
        if (e->terms.size() == 1) {
          const auto& t = e->terms[0];
          if (t.first.empty()) return ConstantInt::get(ity, t.second);
          // one monomial c * a1 * ... * an: c times the product of the atoms (memoized as the monomial with coefficient 1)
          if (t.second != 1) { Value* prod = val(A.mkPoly(ity, {{t.first, 1}})); if (!prod) return nullptr; return B.CreateMul(ConstantInt::get(ity, t.second), prod); }
          Value* prod = nullptr;
          if (t.first.size() >= 2) { std::vector<uint32_t> lead(t.first.begin(), t.first.end() - 1); prod = val(A.mkPoly(ity, {{lead, 1}})); if (!prod) return nullptr; }
          for (size_t i = prod ? t.first.size() - 1 : 0; i < t.first.size(); i++) { Value* av = val(A.atomExpr(t.first[i])); if (!av) return nullptr; av = toInt(av, ity); prod = prod ? B.CreateMul(prod, av) : av; }
          return prod;
        }
        // several terms: the nearest polynomial already emitted (and live here) plus the difference, when the
        // difference has fewer terms than this polynomial (the words of one operand differ in a constant or in one
        // monomial's coefficient); otherwise all terms but the last plus the last
        EP q = nullptr; size_t bestCost = e->terms.size();
        for (EP cand : emittedPolys) {
          if (cand->ty != ity || cand == e || !live(cand)) continue;
          const size_t cost = diffTermCount(e, cand);
          if (cost < bestCost) { bestCost = cost; q = cand; }
        }
        Value* acc;
        if (q) {
          Value* qa = val(q); if (!qa) return nullptr;
          Value* d = val(A.sub(e, q)); if (!d) return nullptr;
          acc = B.CreateAdd(qa, d);
        } else {
          std::vector<std::pair<std::vector<uint32_t>, int64_t>> lead(e->terms.begin(), e->terms.end() - 1);
          Value* p = val(A.mkPoly(ity, lead)); if (!p) return nullptr;
          Value* l = val(A.mkPoly(ity, {e->terms.back()})); if (!l) return nullptr;
          acc = B.CreateAdd(p, l);
        }
        emittedPolys.push_back(e);
        return acc;
      }
      default: break;
    }
    if (e->k == EK::Ite && (hasLoad(e->args[1]) || hasLoad(e->args[2])) && speculate) {
      if (useNeeds && zeroPage) {
        // zero-fill of a load needed exactly here: the load, clamped to the zero page (see zeroClamp)
        EP acc = pathCond(), x = e;
        for (;;) {
          if (x->k != EK::Ite) break;
          const bool zt = isZeroExpr(x->args[1]), ze = isZeroExpr(x->args[2]);
          if (zt == ze) break;
          acc = A.andb(acc, zt ? A.notb(x->args[0]) : x->args[0]);
          x = A.unwrap(zt ? x->args[2] : x->args[1]);
        }
        if (x != e && x->k == EK::Load && A.isTrue(x->args[1]) && A.sizeOf(x->ty) <= 8 && x->ty == e->ty && !hasLoad(acc)) {
          auto it = need.find(x);
          if (it != need.end() && it->second == acc) return val(x);
        }
      }
      Value* c = val(e->args[0]); if (!c) return nullptr;
      specConds.push_back(c); pathE.push_back(A.andb(pathCond(), e->args[0])); Value* a = val(e->args[1]); specConds.pop_back(); pathE.pop_back(); if (!a) return nullptr;
      specConds.push_back(B.CreateNot(c)); pathE.push_back(A.andb(pathCond(), A.notb(e->args[0]))); Value* b = val(e->args[2]); specConds.pop_back(); pathE.pop_back(); if (!b) return nullptr;
      if (a->getType() != b->getType()) return fail("selection arms of different types");
      return B.CreateSelect(c, a, b);
    }
    if (e->k == EK::Ite && (hasLoad(e->args[1]) || hasLoad(e->args[2]))) {
      Value* c = val(e->args[0]); if (!c) return nullptr;
      BasicBlock* thenB = newBlock("mvcc.retile.then"), *elseB = newBlock("mvcc.retile.else"), *join = newBlock("mvcc.retile.join");
      B.CreateCondBr(c, thenB, elseB);
      B.SetInsertPoint(thenB); size_t m = mark(); Value* a = val(e->args[1]); if (!a) return nullptr; BasicBlock* thenEnd = B.GetInsertBlock(); B.CreateBr(join); release(m);
      B.SetInsertPoint(elseB); m = mark(); Value* b = val(e->args[2]); if (!b) return nullptr; BasicBlock* elseEnd = B.GetInsertBlock(); B.CreateBr(join); release(m);
      B.SetInsertPoint(join);
      if (a->getType() != b->getType()) return fail("selection arms of different types");
      PHINode* phi = B.CreatePHI(a->getType(), 2); phi->addIncoming(a, thenEnd); phi->addIncoming(b, elseEnd);
      return phi;
    }
    if (e->k == EK::Bytes && e->args[0]->k == EK::Load && A.isTrue(e->args[0]->args[1]) && (e->len == 1 || e->len == 2 || e->len == 4 || e->len == 8) && e->lo % e->len == 0 && A.sizeOf(e->args[0]->ty) > 8) {
      // a word of a wide staging load (a 16-byte cp.async source): load the word at its address instead of the
      // whole line (MSL has no i128; the other words are loaded by their own users)
      Value* p = val(e->args[0]->args[0]); if (!p) return nullptr;
      if (!p->getType()->isPointerTy()) p = B.CreateIntToPtr(p, PointerType::get(C, 0));
      if (e->lo) p = B.CreateGEP(intTy(8), p, ConstantInt::get(intTy(64), e->lo));
      p = clamp(e->args[0], e->args[0]->args[0], p); if (!p) return nullptr;
      Value* x = B.CreateAlignedLoad(intTy(8 * e->len), p, Align(e->len));
      return ty->isIntegerTy() ? x : B.CreateBitCast(x, ty);
    }
    std::vector<Value*> args;
    for (EP a : e->args) { Value* v = val(a); if (!v) return nullptr; args.push_back(v); }
    switch (e->k) {
      case EK::Ite: return B.CreateSelect(args[0], args[1], args[2]);
      case EK::Cmp: return B.CreateCmp((CmpInst::Predicate)e->op, args[0], args[1]);
      case EK::Op:
        if (Instruction::isCast(e->op)) return B.CreateCast((Instruction::CastOps)e->op, args[0], ty);
        if (args.size() == 1) return B.CreateUnOp((Instruction::UnaryOps)e->op, args[0]);
        return B.CreateBinOp((Instruction::BinaryOps)e->op, args[0], args[1]);
      case EK::Load: {
        if (!A.isTrue(e->args[1])) return fail("a guarded load");
        if (A.sizeOf(ty) > 8) return fail("a load wider than 8 bytes used whole");
        Value* p = args[0];
        if (relocShrink && sharedAddr(e->args[0])) {
          int64_t lo, hi;
          if (!range || !range(e->args[0], lo, hi)) return fail("a staged operand read of unbounded shared offset under the relocation");
          if (lo < relocCut && hi + (int64_t)e->len > relocCut) return fail("a staged operand read straddling the relocation cut");
          if (lo >= relocCut) {
            if (p->getType()->isPointerTy()) p = B.CreateGEP(Type::getInt8Ty(C), p, ConstantInt::get(Type::getInt64Ty(C), -relocShrink));
            else p = B.CreateSub(p, ConstantInt::get(p->getType(), relocShrink));
          }
        }
        if (!p->getType()->isPointerTy()) p = B.CreateIntToPtr(p, PointerType::get(C, 0));
        p = clamp(e, e->args[0], p); if (!p) return nullptr;
        return B.CreateAlignedLoad(ty, p, Align(std::max(1u, e->len)));
      }
      case EK::Bytes: {
        Value* x = toInt(args[0], intTy(8 * A.sizeOf(args[0]->getType())));
        if (e->lo) x = B.CreateLShr(x, 8 * e->lo);
        x = B.CreateTruncOrBitCast(x, intTy(8 * e->len));
        return ty->isIntegerTy() ? x : B.CreateBitCast(x, ty);
      }
      case EK::Concat: {
        const unsigned total = 8 * A.sizeOf(ty); Type* ity = intTy(total);
        Value* acc = nullptr; unsigned off = 0;
        for (Value* p : args) {
          const unsigned w = 8 * A.sizeOf(p->getType());
          Value* q = B.CreateZExt(toInt(p, intTy(w)), ity);
          if (off) q = B.CreateShl(q, off);
          acc = acc ? B.CreateOr(acc, q) : q; off += w;
        }
        if (off != total) return fail("a concatenation of the wrong width");
        return ty->isIntegerTy() ? acc : B.CreateBitCast(acc, ty);
      }
      case EK::Call: {
        if (e->asmv) { auto* IA = cast<InlineAsm>(e->asmv); return B.CreateCall(IA->getFunctionType(), IA, args); }
        // one field of an aggregate-returning call (an operand word's widening): the field index is the trailing
        // constant operand, the call itself emitted once per distinct operand list
        if (e->callee->getReturnType()->isStructTy() && args.size() == e->callee->arg_size() + 1) {
          std::vector<Value*> cargs(args.begin(), args.end() - 1);
          std::vector<EP> ce(e->args.begin(), e->args.end() - 1);
          EP whole = A.call(e->callee, ce, e->callee->getReturnType());
          Value* cv = live(whole) ? memo[whole] : nullptr;
          if (!cv) { cv = B.CreateCall(e->callee, cargs); memo[whole] = cv; memoBB[whole] = curBB ? curBB : B.GetInsertBlock(); trail.push_back(whole); }
          auto* ST = cast<StructType>(e->callee->getReturnType());
          if (auto* idx = dyn_cast<ConstantInt>(args.back())) {
            if (idx->getZExtValue() >= ST->getNumElements()) return fail("an aggregate field index out of range");
            return B.CreateExtractValue(cv, (unsigned)idx->getZExtValue());
          }
          // the field index varies with the lane (which half pair of the widened word a lane's slot holds): every
          // field extracted, the one selected by the index
          for (unsigned i = 1; i < ST->getNumElements(); i++) if (ST->getElementType(i) != ST->getElementType(0)) return fail("a lane-selected aggregate field of mixed types");
          Value* r = B.CreateExtractValue(cv, ST->getNumElements() - 1);
          for (unsigned i = ST->getNumElements() - 1; i-- > 0;) r = B.CreateSelect(B.CreateICmpEQ(args.back(), ConstantInt::get(args.back()->getType(), i)), B.CreateExtractValue(cv, i), r);
          return r;
        }
        return B.CreateCall(e->callee, args);
      }
      default: return fail("an expression kind the materializer does not emit");
    }
  }
};

static std::string asmMnemonics(const CallInst* CI, bool& allAlu) {
  std::vector<PtxInstr> ins; std::string err, out;
  allAlu = false;
  if (!parsePtxAsm(std::string(cast<InlineAsm>(CI->getCalledOperand())->getAsmString()), ins, err) || ins.empty()) return out;
  allAlu = true;
  for (auto& in : ins) { out += (out.empty() ? "" : " ") + in.mnemonic; if (!ptxIsPerThreadAlu(in.mnemonic)) allAlu = false; }
  return out;
}

struct OperandFit {
  Node* tmpl = nullptr;  // the unified template
  EP sample = nullptr;   // one expression of the shape (for diagnostics)
  std::vector<Affine> holes;   // by hole index (Shaper::holes)
  unsigned samples = 0;
};

struct ScevFind {
  function_ref<bool(const SCEV*)> pred; bool found = false;
  bool follow(const SCEV* x) { if (pred(x)) found = true; return !found; }
  bool isDone() const { return found; }
};
bool scevAny(const SCEV* s, function_ref<bool(const SCEV*)> pred) { ScevFind V{pred}; visitAll(s, V); return V.found; }

bool dependsOnTid(Value* v, ScalarEvolution& SE, LoopInfo& LI, std::map<Value*, bool>& memo, int depth) {
  auto it = memo.find(v);
  if (it != memo.end()) return it->second;
  if (depth > 128) return memo[v] = true;
  if (isa<Constant>(v) || isa<Argument>(v)) return memo[v] = false;
  auto* I = dyn_cast<Instruction>(v);
  if (!I) return memo[v] = true;
  if (auto* CI = dyn_cast<CallInst>(I)) {
    Function* Callee = CI->getCalledFunction();
    if (!Callee) return memo[v] = true;
    StringRef n = Callee->getName();
    if (n.starts_with("llvm.nvvm.read.ptx.sreg.tid.") || n == "llvm.nvvm.read.ptx.sreg.laneid") return memo[v] = true;
    if (n.starts_with("llvm.nvvm.read.ptx.sreg.")) return memo[v] = false;
    return memo[v] = true;
  }
  if (auto* PN = dyn_cast<PHINode>(I)) {
    memo[v] = true;
    const SCEV* s = SE.getSCEV(PN);
    auto* AR = dyn_cast<SCEVAddRecExpr>(s);
    if (!AR || !AR->isAffine()) {
      const Loop* L = LI.getLoopFor(PN->getParent());
      if (!PN->getNumIncomingValues() || (L && L->getHeader() == PN->getParent())) return true;
      for (Value* in : PN->incoming_values()) if (SE.getSCEV(in) != SE.getSCEV(PN->getIncomingValue(0))) return true;
      return memo[v] = dependsOnTid(PN->getIncomingValue(0), SE, LI, memo, depth + 1);
    }
    return memo[v] = scevAny(s, [&](const SCEV* x) {
      auto* U = dyn_cast<SCEVUnknown>(x); return U && dependsOnTid(U->getValue(), SE, LI, memo, depth + 1);
    });
  }
  if (isa<LoadInst>(I)) return memo[v] = true;
  for (Use& U : I->operands()) if (dependsOnTid(U.get(), SE, LI, memo, depth + 1)) return memo[v] = true;
  return memo[v] = false;
}

}  // namespace

// ---------------------------------------------------------------- the pass

bool StagingElim::emitFromStir() {
  auto kept = [&](const std::string& w) { note("decode retiling kept: " + w); return false; };
  auto done = [&](const std::string& w) { note("STIR tensor-op body already materialized: " + w); return true; };
  if (!T) return kept("the block size is not known (no launch bound)");
  if (T % 32) return kept("the block size is not a multiple of the warp size");

  // the K loop: the innermost loop containing every fused matmul; the job loop: its parent
  const Loop* K = nullptr;
  std::vector<CallInst*> mmaCalls;
  bool ldmatrix = false, ring = false;
  auto findK = [&]() -> bool {
    K = nullptr; mmaCalls.clear(); ldmatrix = ring = false;
    for (Instruction& I : instructions(F)) {
      auto* CI = dyn_cast<CallInst>(&I);
      if (!CI || !CI->getCalledFunction()) continue;
      StringRef n = CI->getCalledFunction()->getName();
      if (n.starts_with("__mvcc_tp_ld_ring")) ring = true;
      else if (n.starts_with("__mvcc_tp_ld")) ldmatrix = true;
      if (!n.starts_with("__mvcc_tp_mma_")) continue;
      mmaCalls.push_back(CI);
      const Loop* L = LI.getLoopFor(CI->getParent());
      if (!L) return kept("a matmul operation outside any loop");
      if (!K) K = L;
      while (K && !K->contains(CI->getParent())) K = K->getParentLoop();
      if (!K) return kept("matmul operations in unrelated loops");
    }
    if (mmaCalls.empty()) return false;
    for (CallInst* CI : mmaCalls) if (LI.getLoopFor(CI->getParent()) != K) return kept("a matmul operation in a loop nested inside the K loop");
    return true;
  };
  if (!findK()) return false;
  if (ldmatrix && !ring) return done("ldmatrix operands");
  for (CallInst* CI : mmaCalls) {
    if (CI->arg_size() < 3) return kept("matmul argument count");
    auto* cM = dyn_cast<ConstantInt>(CI->getArgOperand(0));
    auto* cN = dyn_cast<ConstantInt>(CI->getArgOperand(1));
    auto* cK = dyn_cast<ConstantInt>(CI->getArgOperand(2));
    if (!cM || !cN || !cK) return kept("matmul shape not constant");
    const int64_t Ms = cM->getSExtValue(), Ns = cN->getSExtValue(), Ks = cK->getSExtValue();
    if (Ms != 16 || (Ns != 16 && Ns != 32) || Ks != 32)
      return done("shape " + std::to_string(Ms) + "x" + std::to_string(Ns) + "x" + std::to_string(Ks));
  }
  { std::string w; if (stageRingParent(K, w)) return done(w); }
  // Unroll a constant-few inner group loop so the ring is the K loop.
  if (K->getParentLoop()) {
    Loop* Km = const_cast<Loop*>(K);
    unsigned mt = SE.getSmallConstantTripCount(Km);
    if (!mt) mt = SE.getSmallConstantMaxTripCount(Km);
    if (mt && mt <= 4) {
      const Loop* P = K->getParentLoop();
      BasicBlock* PH = P->getHeader();
      simplifyLoop(Km, &DT, &LI, &SE, &AC, nullptr, false);
      formLCSSARecursively(*Km, DT, &LI, &SE);
      TargetTransformInfo TTI(DL);
      OptimizationRemarkEmitter ORE(&F);
      UnrollLoopOptions ULO; ULO.Count = mt; ULO.Force = true; ULO.Runtime = false; ULO.AllowExpensiveTripCount = true; ULO.UnrollRemainder = false; ULO.ForgetAllSCEV = true; ULO.SCEVExpansionBudget = 0;
      const LoopUnrollResult r = UnrollLoop(Km, ULO, &LI, &SE, &DT, &AC, &TTI, &ORE, /*PreserveLCSSA=*/true);
      if (r == LoopUnrollResult::Unmodified) return kept("the matmuls' inner loop (at most " + std::to_string(mt) + " iterations) could not be unrolled");
      note("the matmuls' inner loop unrolled by its maximum of " + std::to_string(mt) + " iteration(s): the ring loop is the K loop");
      SE.forgetAllLoops();
      if (!findK()) return false;
      if (K->getHeader() != PH) return kept("the ring loop is not the K loop after unrolling");
    }
  }
  const Loop* J = K->getParentLoop();

  Arena A(DL, F.getContext());
  Executor X(F, DT, LI, A, T, nullptr);
  sig::Graph Sig(F, T, X);
  X.retile = true;
  X.iterL = K; X.iterCount = 6;
  X.factorGuards = true;
  if (J) {
    if (J->getParentLoop()) return kept("the K loop is nested more than one level deep");
    std::map<Value*, bool> tidMemo;
    for (PHINode& PN : J->getHeader()->phis())
      for (Value* in : PN.incoming_values()) if (dependsOnTid(in, SE, LI, tidMemo, 0)) { X.threadPhis.insert(&PN); break; }
    X.generic[J] = true;
  }
  X.laneStoresStay = true;
  X.recordDevStores = true;   // what the kernel's result is: the proof below about uncovered staged bytes reads them
  X.recordRecurrences = true;
  X.recordDevLoads = true;
  // every other loop with a trip count SCEV cannot bound runs once with per-thread symbolic phis - unless every
  // thread of the block computes it alike (the experts' job -> expert search: a ballot over a read-only offset
  // table, then a scan of it), in which case its results are opaque uniform values (Executor::opaqueExits)
  UniformityScan uni;
  for (const Loop* L : LI.getLoopsInPreorder()) {
    if (L == K || L == J || K->contains(L)) continue;
    // a loop with a constant trip count, or a constant maximum one (`for (i = tid; i < n; i += blockDim)`: the
    // zero-fill of a staging buffer, a quantizer's rows), runs to its end with concrete per-thread values
    unsigned tc = SE.getSmallConstantTripCount(const_cast<Loop*>(L));
    if (tc == 0 && concretelyBounded(L)) tc = SE.getSmallConstantMaxTripCount(const_cast<Loop*>(L));
    if (tc == 0 || tc > 256) {
      const bool uniform = uni.loopUniform(L);
      X.generic[L] = uniform;
      if (uniform) { X.opaqueExits.insert(L); note("a block-uniform loop's results are opaque values (" + std::string(L->getHeader()->getName()) + ")"); }
      else if (uni.why) { std::string w; raw_string_ostream os(w); os << *uni.why; note("a loop's results are per-thread values (" + StringRef(os.str()).trim().str() + ")"); uni.why = nullptr; }
    }
  }
  const double retileBudget = 12;  // whole kernel; f16 catalog carryWalk otherwise runs for 30+ min
  const double retileStart = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  X.wallStart = retileStart; X.wallBudget = retileBudget;
  for (unsigned t = 0; t < T; t++) {
    const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (now - retileStart > retileBudget) return kept("time budget of " + std::to_string((int)retileBudget) + " s exceeded");
    X.wallStart = retileStart; X.wallBudget = retileBudget;
    std::string w; if (!X.runRetile(t, w)) return kept("thread " + std::to_string(t) + ": " + w);
  }
  { std::string w; if (!X.resolve(w)) return kept(w); }
  coords::Discovery Disc = coords::discover(A, X, T);
  for (auto& l : Disc.lines) note(l);
  if (!Disc.fittedSites) return kept("coordinate discovery: no fitted site");
  unsigned nOut = 0; for (bool b : Disc.isOutput) if (b) nOut++;
  if (Disc.d < 3)
    return kept("coordinate discovery: GEMM domain is not d>=3 (d=" + std::to_string(Disc.d) + ", output " + std::to_string(nOut) + ")");
  for (auto& m : Disc.sites) {
    if (m.fitted() || m.isPred || m.isDerivedOperand) continue;
    if (m.why.find("data-dependent") != std::string::npos || m.why.find("non-affine") != std::string::npos) continue;
    if (m.why.find("under") != std::string::npos || m.why.find("sample") != std::string::npos) continue;
    return kept("coordinate discovery: " + m.why);
  }
  // step 4: the K-loop recurrence algebra from the SIG classifier. Sum / TensorSum with tree-combine
  // is what permits the simdgroup copies; struct Term only enumerates the units the split cuts.
  bool kRecSum = false; sig::Permissions kRecPerm; unsigned kRecSumPhis = 0, kRecOther = 0;
  {
    std::set<PHINode*> seen;
    for (auto& r : X.recurrences) {
      if (r.L != K || !seen.insert(r.phi).second) continue;
      sig::Classified c = sig::classifyRecurrence(A, r, X.recurrences);
      if (c.kind == sig::Algebra::Sum || c.kind == sig::Algebra::TensorSum) {
        kRecSum = true; kRecSumPhis++;
        kRecPerm.partition |= c.permits.partition; kRecPerm.reorder |= c.permits.reorder;
        kRecPerm.treeCombine |= c.permits.treeCombine; kRecPerm.duplicate |= c.permits.duplicate;
      } else if (c.kind != sig::Algebra::Invariant && c.kind != sig::Algebra::Induction) {
        kRecOther++;
      }
    }
    if (kRecSum)
      note("K-loop recurrence: " + std::to_string(kRecSumPhis) + " sum/tensor-sum phi(s)" + (kRecOther ? (", " + std::to_string(kRecOther) + " other") : "") + " -> permits:" + (kRecPerm.partition ? " partition" : "") + (kRecPerm.reorder ? " reorder" : "") + (kRecPerm.treeCombine ? " tree-combine" : "") + " (simdgroup copies run from these permissions)");
    else note("K-loop recurrence: no sum/tensor-sum phi (split will not run from the classifier)");
  }
  if (X.deadReads) note(std::to_string(X.deadReads) + " staged byte read(s) are uncovered under part of their condition, a part that lies on paths the thread had left by an earlier return (an empty split, a tile past N): read as zero there");
  for (auto& kv : X.assumptions) note("decode retiling assumes " + (A.isTrue(kv.first.first) ? std::string() : A.str(kv.first.first, 6) + " => ") + A.str(kv.first.second, 6) + " (" + std::to_string(kv.second.size()) + " shared read(s) see the latest write only under it; the exact twin runs when it fails)");

  // ---- 2. structure: per (thread, iteration) the chains of fused ops (one per group product in the iteration)
  struct Op { const Executor::TpCall* call; unsigned nA, nB, cap; int M, N, Kop; };
  struct Chain { std::vector<Op> ops; const Executor::TpCall* conv = nullptr; Instruction* head = nullptr; };
  std::map<std::pair<unsigned, unsigned>, std::vector<Op>> opsOf;  // (thread, iter) -> ops
  std::map<std::pair<unsigned, unsigned>, std::vector<const Executor::TpCall*>> convsOf;
  std::map<std::tuple<Instruction*, unsigned, unsigned>, const Executor::TpCall*> f2s;  // (call, thread, iter)
  std::map<Instruction*, unsigned> instOrder;
  { unsigned i = 0; for (Instruction& I : instructions(F)) instOrder[&I] = i++; }
  for (auto& tc : Sig.tpCalls) {
    auto* CI = cast<CallInst>(tc.I);
    StringRef n = CI->getCalledFunction()->getName();
    if (tc.iter == ~0u) { if (n.starts_with("__mvcc_tp_mma_")) return kept("a matmul operation executed outside the K loop"); continue; }
    if (n == "__mvcc_tp_frag2slot") { f2s[{tc.I, tc.thread, tc.iter}] = &tc; continue; }
    if (n.starts_with("__mvcc_tp_acc2ptx_")) { convsOf[{tc.thread, tc.iter}].push_back(&tc); continue; }
    if (!n.starts_with("__mvcc_tp_mma_")) continue;
    int64_t M, N, Kop; if (!A.constInt(tc.args[0], M) || !A.constInt(tc.args[1], N) || !A.constInt(tc.args[2], Kop)) return kept("matmul shape not constant");
    Op op{&tc, (unsigned)(M * Kop / 64), (unsigned)(Kop * N / 64), (unsigned)(M * N / 32), (int)M, (int)N, (int)Kop};
    if (tc.args.size() != 6 + op.nA + op.nB + op.cap) return kept("matmul argument count");
    opsOf[{tc.thread, tc.iter}].push_back(op);
  }
  if (opsOf.empty()) return kept("no matmul executed in the K loop");
  auto isZeroC = [&](const Op& op) { for (unsigned p = 0; p < op.cap; p++) { EP c = op.call->args[6 + op.nA + op.nB + p]; if (c->k != EK::Const || c->c != 0) return false; } return true; };
  auto producedBy = [&](EP v, Instruction*& src, unsigned& field) {
    v = A.unwrap(v);
    if (v->k != EK::Sym) return false;
    const Symbol& s = A.symbol(v->sym);
    if (s.kind != Symbol::ThreadValue || !s.v) return false;
    src = cast<Instruction>(s.v); field = s.field; return true;
  };
  // chains: from every zero-initialized op follow the accumulator; each ends in exactly one conversion
  std::map<std::pair<unsigned, unsigned>, std::vector<Chain>> chains;  // (thread, iter) -> chains in head program order
  int M = 0, N = 0, KopAll = 0; unsigned chainLen = 0, nChains = 0;
  for (auto& kv : opsOf) {
    std::vector<Op>& ops = kv.second;
    std::vector<char> used(ops.size(), 0);
    std::vector<Chain> cs;
    for (size_t i = 0; i < ops.size(); i++) if (isZeroC(ops[i])) { Chain c; c.ops.push_back(ops[i]); c.head = ops[i].call->I; used[i] = 1; cs.push_back(std::move(c)); }
    if (cs.empty()) return kept("the accumulator does not start from zero within the iteration (the K loop is not a group product)");
    for (auto& c : cs) {
      for (;;) {
        bool found = false;
        for (size_t i = 0; i < ops.size() && !found; i++) {
          if (used[i]) continue;
          bool all = true;
          for (unsigned p = 0; p < ops[i].cap && all; p++) { Instruction* src; unsigned f; if (!producedBy(ops[i].call->args[6 + ops[i].nA + ops[i].nB + p], src, f) || src != c.ops.back().call->I || f != p) all = false; }
          if (all) { c.ops.push_back(ops[i]); used[i] = 1; found = true; }
        }
        if (!found) break;
      }
      for (const Executor::TpCall* cv : convsOf[kv.first]) {
        bool all = true;
        for (unsigned p = 0; p < c.ops.back().cap && all; p++) { Instruction* src; unsigned f; if (cv->args.size() != 2 + c.ops.back().cap || !producedBy(cv->args[2 + p], src, f) || src != c.ops.back().call->I || f != p) all = false; }
        if (all) { if (c.conv) return kept("a chain's result is converted to PTX layout twice"); c.conv = cv; }
      }
      if (!c.conv) return kept("a chain's result is not converted to PTX layout within the iteration");
      for (auto& op : c.ops) { if (M && (op.M != M || op.N != N || op.Kop != KopAll)) return kept("fused ops of different shapes"); M = op.M; N = op.N; KopAll = op.Kop; }
      if (chainLen && chainLen != c.ops.size()) return kept("chains of different lengths");
      chainLen = (unsigned)c.ops.size();
    }
    for (size_t i = 0; i < ops.size(); i++) if (!used[i]) return kept("a matmul operation of the iteration belongs to no zero-initialized chain");
    std::sort(cs.begin(), cs.end(), [&](const Chain& a, const Chain& b) { return instOrder[a.head] < instOrder[b.head]; });
    if (nChains && nChains != cs.size()) return kept("iterations with different numbers of chains");
    nChains = (unsigned)cs.size();
    chains[kv.first] = std::move(cs);
  }
  if (M != 16) return kept("the M tile is " + std::to_string(M) + " rows (decode retiling covers 16-row jobs)");
  if ((N != 16 && N != 32) || KopAll != 32) return kept("fused op shape " + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(KopAll) + " (expected 16x16x32 or 16x32x32)");
  const unsigned Kiter = chainLen * KopAll;  // K per chain (group)
  const unsigned nWarps = T / 32;
  if (chains.size() != (size_t)T * X.iterCount) return kept("not every thread executes the chains in every iteration");

  // ---- 3. element expressions: A(m, k) and B(n, k) pairs by the mma.sync fragment layout
  // coordinates: (row, (row/8)%2, row/16, k/32, (k/8)%2, (k/16)%2, (k/2)%4, iter, warp bits..., chain) - k within the
  // iteration's K range, split at the 16-byte chunk (k/32 of int4 weights; k/8 of bf16 activations is affine in the
  // first two) and the 4-byte word; the row also by its n8 block, so that a B tile whose n8 blocks come from
  // different matrices (gate rows then up rows, at the intermediate size apart) is affine. The warp index enters by
  // its bits, so that both a warp's position across N (warp % kNWarps) and its K parity (warp / kNWarps, the
  // kKSplit = 2 geometries) are affine; A may then depend on the warp (its K half).
  if (nWarps & (nWarps - 1)) return kept("the number of warps is not a power of two");
  unsigned warpBits = 0; while ((1u << warpBits) < nWarps) warpBits++;
  // The 8-half block within the 32-half chunk enters by its two bits, (k/8)%2 and (k/16)%2: an fp8 word widened to
  // two half pairs puts the k16 half of the split m16n8k32 in the pair (2 bytes on) and the chunk half 16 bytes on.
  // Domain is discovered (o, r); packing bits are the quasi-affine transform of (row, k).
  const std::string coordNames = "discovered (o_row, r_k, r_iter, sg, chain) via the quasi-affine transform of (row, k)";
  auto expandQuasi = [&](int64_t row, int64_t k, unsigned iter, unsigned warp, unsigned chain) {
    std::vector<int64_t> c{row, (row / 8) % 2, row / 16, k / 32, (k / 8) % 2, (k / 16) % 2, (k / 2) % 4, (int64_t)iter};
    for (unsigned b = 0; b < warpBits; b++) c.push_back((warp >> b) & 1);
    c.push_back((int64_t)chain);
    return c;
  };
  auto discoveredPhys = [&](unsigned thread, unsigned iter) {
    std::vector<int64_t> x;
    for (auto& ax : Disc.axes) {
      if (ax.kind == coords::Axis::Simdgroup) x.push_back((int64_t)(thread / 32));
      else if (ax.kind == coords::Axis::Lane) x.push_back((int64_t)(thread % 32));
      else if (ax.kind == coords::Axis::Iter) x.push_back(ax.loop == K ? (int64_t)iter : 0);
      else if (ax.kind == coords::Axis::TidBit) x.push_back((thread >> ax.index) & 1);
      else x.push_back(0);
    }
    std::vector<int64_t> c(Disc.d, 0);
    for (unsigned j = 0; j < Disc.d; j++)
      for (size_t a = 0; a < Disc.V.size() && a < x.size(); a++) c[j] += Disc.V[a][j] * x[a];
    return c;
  };
  auto coordsOf = [&](int64_t row, int64_t k, unsigned iter, unsigned warp, unsigned chain, unsigned thread) {
    (void)thread;
    // 3.4 quasi-affine transform of the discovered (row, k). The named packing is that transform, not the domain.
    return expandQuasi(row, k, iter, warp, chain);
  };
  struct Sample { std::vector<int64_t> coords; EP e; EP cond; unsigned thread = 0, iter = 0, warp = 0, chain = 0; int64_t row = 0, k = 0; };
  std::vector<Sample> aS, bS;
  // Staged bytes no write covers under part of their condition (Executor::uncovered: a weight row past N the guarded
  // copy skips): which operand rows (A) / columns (B) of which warp each feeds, from the words it appears in. The
  // proof that follows the sampling: every device store that depends on an output of those rows / columns is made
  // under a condition disjoint from the residual - an mma output depends on its own A row and B column only, so the
  // kernel's result is the same whatever the bytes hold, and the recovered body's zero is as good as the source's
  // uninitialized memory.
  std::map<uint32_t, size_t> uncIdx;
  for (size_t i = 0; i < X.uncovered.size(); i++) uncIdx[X.uncovered[i].sym] = i;
  // (warp, is B, row / column) -> the condition the tensor operations that consume the byte there run under (a
  // stage's second K group runs under `1 < groups here`: the bytes it reads are uncovered only where it does not run)
  std::vector<std::map<std::tuple<unsigned, bool, int64_t>, EP>> feeds(X.uncovered.size());
  auto feed = [&](size_t u, unsigned warp, bool isB, int64_t rc, EP cond) { auto& m = feeds[u]; auto it = m.find({warp, isB, rc}); if (it == m.end()) m[{warp, isB, rc}] = cond; else it->second = A.orb(it->second, cond); };
  std::unordered_map<EP, std::vector<size_t>> feedMemo;
  auto uncoveredIn = [&](EP e) -> const std::vector<size_t>& {
    auto f = feedMemo.find(e);
    if (f != feedMemo.end()) return f->second;
    std::vector<size_t> v;
    if (!uncIdx.empty()) { std::set<uint32_t> ss; if (A.containsSymbol(e, Symbol::SmemByte, &ss)) for (uint32_t sid : ss) { auto u = uncIdx.find(sid); if (u != uncIdx.end()) v.push_back(u->second); } }
    return feedMemo[e] = v;
  };
  // (`cond`: the path condition of the conversion that built the word - where the fill will be emitted)
  auto wordExpr = [&](const Op& op, unsigned argIdx, unsigned thread, unsigned iter, EP& out, EP& cond) -> bool {
    Instruction* src; unsigned f;
    if (!producedBy(op.call->args[argIdx], src, f)) return false;
    auto it = f2s.find({src, thread, iter});
    if (it == f2s.end() || f >= 4) return false;
    out = it->second->args[1 + f]; cond = it->second->cond;
    return true;
  };
  for (auto& kv : chains) {
    const unsigned thread = kv.first.first, iter = kv.first.second;
    const unsigned lane = thread & 31, warp = thread / 32;
    const int64_t g = lane >> 2, t = lane & 3;
    for (size_t q = 0; q < kv.second.size(); q++) {
      const Chain& ch = kv.second[q];
      for (size_t p = 0; p < ch.ops.size(); p++) {
        const Op& op = ch.ops[p];
        const int LV = op.Kop / 16, NTL = op.N / 8;
        // A words: subA[kb + LV*mb] (mb = 0 for M = 16), 4 words per block: r -> row g + 8 (r & 1), k + 2t + 8 (r >> 1)
        for (int kb = 0; kb < LV; kb++) for (int r = 0; r < 4; r++) {
          EP e, wc; if (!wordExpr(op, 6 + 4 * kb + r, thread, iter, e, wc)) return kept("an A operand word is not a register-built fragment word");
          int64_t k = (int64_t)p * op.Kop + 16 * kb + 8 * (r >> 1) + 2 * t;
          const int64_t row = g + 8 * (r & 1);
          Sample s; s.coords = coordsOf(row, k, iter, warp, (unsigned)q, thread); s.e = e; s.cond = wc; s.thread = thread; s.iter = iter; s.warp = warp; s.chain = (unsigned)q; s.row = row; s.k = k;
          aS.push_back(s);
          for (size_t u : uncoveredIn(e)) feed(u, warp, false, g + 8 * (r & 1), op.call->cond);
        }
        // B words: subB[nb + (NTL/2)*kb], 4 words per block: r -> column g + 8 (r >> 1) + 16 nb, k + 2t + 8 (r & 1)
        for (int kb = 0; kb < LV; kb++) for (int nb = 0; nb < NTL / 2; nb++) for (int r = 0; r < 4; r++) {
          EP e, wc; if (!wordExpr(op, 6 + op.nA + 4 * (nb + (NTL / 2) * kb) + r, thread, iter, e, wc)) return kept("a B operand word is not a register-built fragment word");
          int64_t k = (int64_t)p * op.Kop + 16 * kb + 8 * (r & 1) + 2 * t;
          const int64_t row = g + 8 * (r >> 1) + 16 * nb;
          Sample s; s.coords = coordsOf(row, k, iter, warp, (unsigned)q, thread); s.e = e; s.cond = wc; s.thread = thread; s.iter = iter; s.warp = warp; s.chain = (unsigned)q; s.row = row; s.k = k;
          bS.push_back(s);
          for (size_t u : uncoveredIn(e)) feed(u, warp, true, g + 8 * (r >> 1) + 16 * nb, op.call->cond);
        }
      }
    }
  }
  if (!X.uncovered.empty()) {
    // the accumulator values an expression depends on, as (conversion call, PTX position): position p of lane l is
    // D[(l >> 2) + 8 ((p & 3) >> 1)][8 ((p >> 2) % (N / 8)) + 2 (l & 3) + (p & 1)] (mvcc_prelude.metal, acc2ptx)
    auto accSyms = [&](EP e, std::vector<std::pair<Instruction*, unsigned>>& out) {
      std::set<uint32_t> ss;
      if (!A.containsSymbol(e, Symbol::ThreadValue, &ss)) return;
      for (uint32_t sid : ss) { const Symbol& sy = A.symbol(sid); auto* CI = sy.v ? dyn_cast<CallInst>(sy.v) : nullptr; if (CI && CI->getCalledFunction() && CI->getCalledFunction()->getName().starts_with("__mvcc_tp_acc2ptx_")) out.push_back({CI, sy.field}); }
    };
    std::function<void(EP, std::vector<EP>&)> disjuncts = [&](EP e, std::vector<EP>& out) { if (e->k == EK::Op && e->op == Instruction::Or && e->ty->isIntegerTy(1)) { disjuncts(e->args[0], out); disjuncts(e->args[1], out); } else out.push_back(e); };
    // A value the executor could not follow (Symbol::Unknown) may hide the dependency - except one read from device
    // memory (the result of a device atomic, an `ld.global` in inline asm), which is the same in both bodies once
    // every store before it is: the induction this proof runs, store by store in program order.
    std::map<Value*, bool> fromDevice;
    auto readsDevice = [&](Value* o) -> bool {
      if (!o) return false;
      auto f = fromDevice.find(o);
      if (f != fromDevice.end()) return f->second;
      bool r = isa<AtomicRMWInst>(o) || isa<AtomicCmpXchgInst>(o);
      if (auto* CI = dyn_cast<CallInst>(o); CI && CI->isInlineAsm()) {
        std::vector<PtxInstr> ins; std::string err;
        if (parsePtxAsm(std::string(cast<InlineAsm>(CI->getCalledOperand())->getAsmString()), ins, err) && !ins.empty()) { r = true; for (auto& in : ins) if (in.mnemonic != "ld" || !in.hasMod("global") || in.predicated()) r = false; }
      }
      return fromDevice[o] = r;
    };
    auto hiddenIn = [&](const DevStore& st) {
      std::set<uint32_t> ss;
      A.containsSymbol(st.cond, Symbol::Unknown, &ss); A.containsSymbol(st.addr, Symbol::Unknown, &ss);
      for (EP b : st.bytes) A.containsSymbol(b, Symbol::Unknown, &ss);
      for (uint32_t sid : ss) if (!readsDevice(A.symbol(sid).v)) return true;
      return false;
    };
    auto where = [&](const Executor::Uncovered& U) { std::string s; raw_string_ostream os(s); os << *U.site; return "shared byte read " + StringRef(os.str()).trim().str() + " (thread " + std::to_string(U.thread) + ")"; };
    // per store, once: the (warp, operand, row / column) of every accumulator value it depends on, and whether a
    // value it depends on may hide more
    struct StoreDep { std::set<std::tuple<unsigned, bool, int64_t>> on; bool hidden; };
    std::vector<StoreDep> deps(X.devStores.size());
    for (size_t si = 0; si < X.devStores.size(); si++) {
      const DevStore& st = X.devStores[si];
      const unsigned warp = st.thread / 32, lane = st.thread & 31, g2 = lane >> 2, t2 = lane & 3;
      std::vector<std::pair<Instruction*, unsigned>> as;
      for (EP b : st.bytes) accSyms(b, as);
      accSyms(st.cond, as); accSyms(st.addr, as);
      for (auto& cf : as) { const unsigned p = cf.second, nt = (p >> 2) % (N / 8); deps[si].on.insert({warp, false, (int64_t)(g2 + 8 * ((p & 3) >> 1))}); deps[si].on.insert({warp, true, (int64_t)(8 * nt + 2 * t2 + (p & 1))}); }
      deps[si].hidden = hiddenIn(st);
    }
    std::map<std::pair<size_t, EP>, bool> disjointMemo;   // (store, residual clause) -> disjoint
    for (size_t ui = 0; ui < X.uncovered.size(); ui++) {
      const Executor::Uncovered& U = X.uncovered[ui];
      if (feeds[ui].empty()) return kept(where(U) + " is uncovered under " + A.str(U.cond, 6) + " and is not a matmul operand");
      // (the row's guard, when the executor factored it (`!G`), is read through its definition by refutes())
      std::vector<EP> res; disjuncts(U.cond, res);
      for (size_t si = 0; si < X.devStores.size(); si++) {
        const DevStore& st = X.devStores[si];
        // the store depends on the byte through the operations at these positions: under their conditions
        EP opCond = deps[si].hidden ? A.mkBool(true) : nullptr;
        if (!opCond) for (auto& f : feeds[ui]) if (deps[si].on.count(f.first)) opCond = opCond ? A.orb(opCond, f.second) : f.second;
        if (!opCond) continue;
        EP under = A.andb(st.cond, opCond);
        for (EP c : res) {
          auto dm = disjointMemo.find({si, c});
          auto disjointClamps = [&]() { if (A.disjoint(under, c)) return true; std::vector<EP> cs; disjuncts(under, cs); for (EP u : cs) if (!X.refutes(u, c)) return false; return true; };
          const bool dj = dm != disjointMemo.end() ? dm->second : (disjointMemo[{si, c}] = disjointClamps());
          if (dj) continue;
          std::string s; raw_string_ostream os(s); os << *st.site;
          return kept(where(U) + " is uncovered under " + A.str(c, 6) + " and feeds a device store made under it (" + StringRef(os.str()).trim().str() + ", thread " + std::to_string(st.thread) + ")");
        }
      }
    }
    note(std::to_string(X.uncovered.size()) + " staged byte read(s) are of uninitialized shared memory under part of their condition (rows past the matrix edge the copy skips); the outputs they feed are stored only where that part is false, so they read as zero");
  }
  // substitute the resolved shared bytes
  Arena::RewriteMemo memo;
  // ...and decide the selections between candidate writes (and any selection on a condition the fit could not
  // express: values of loops run generically or of other threads) by the zero-fill selections enclosing them
  auto unfittable = [&](EP c) {
    if (X.resolveConds.count(c)) return true;
    if (A.containsSymbol(c, Symbol::Unknown) || A.containsSymbol(c, Symbol::ThreadValue) || A.containsSymbol(c, Symbol::SmemByte)) return true;
    std::set<uint32_t> lvs;
    if (A.containsSymbol(c, Symbol::LoopValue, &lvs)) for (uint32_t sid : lvs) { const Symbol& sy = A.symbol(sid); if (!J || !sy.v || cast<Instruction>(sy.v)->getParent() != J->getHeader()) return true; }
    return false;
  };
  // A device load's guard is the path condition the executor issued it under (the job loop's and the K loop's
  // continuation conditions among its literals, which differ by iteration); the operand expressions carry their own
  // zero-fill selections around every load, so the guards are dropped for the fit and the re-emitted loads are
  // guarded by those selections
  Arena::RewriteMemo gmemo;
  std::function<EP(EP)> stripGuards = [&](EP e) { return A.rewrite(e, [&](EP x) -> EP { if (x->k != EK::Load || A.isTrue(x->args[1])) return nullptr; return A.load(x->ty, stripGuards(x->args[0]), A.mkBool(true), x->len); }, gmemo); };
  // Null tests of device addresses. The source tests a staged row's pointer (`if (src) copy else zero-fill`);
  // LLVM keeps the test where it cannot fold it (the row's base) and drops it where it can (an inbounds offset
  // from the base), so elements of one tile carry different guards. Every such address is a pointer argument
  // plus an offset, and the executor's address model has no wrap-around, so the address is null exactly when the
  // argument is: the tests fold to the argument's non-nullness, which the launch checks (an assumption on the
  // arguments; the exact twin runs for a null argument).
  // The address is either the pointer argument plus an offset outright, or a selection between that and null
  // (the row pointer a lane stages: `valid ? x + row * K : nullptr`, tested again at the copy): the test of the
  // selection is the selection's own condition (a null test of a value known non-null on one side).
  Arena::RewriteMemo nmemo;
  auto ptrArgOf = [&](EP p) -> uint32_t {   // the pointer argument a non-null address is an offset from, or ~0u
    while (p->k == EK::Op && Instruction::isCast(p->op) && p->args.size() == 1) p = A.unwrap(p->args[0]);
    uint32_t sid;
    if (p->k == EK::Sym) return A.symbol(p->sym).kind == Symbol::Arg && A.symbol(p->sym).ty->isPointerTy() ? p->sym : ~0u;
    if (p->k != EK::Poly) return ~0u;
    uint32_t argSym = ~0u;
    for (auto& t : p->terms) if (t.first.size() == 1 && t.second == 1 && A.isSymAtom(t.first[0], sid) && A.symbol(sid).kind == Symbol::Arg && A.symbol(sid).ty->isPointerTy()) { if (argSym != ~0u) return ~0u; argSym = sid; }
    return argSym;
  };
  auto assumeNonNull = [&](uint32_t argSym) { EP arg = A.mkSym(A.symbol(argSym)); X.assumptions[{A.mkBool(true), A.cmp(CmpInst::ICMP_NE, arg, A.mkInt(arg->ty, 0))}]; };
  auto isNullConst = [&](EP z) { int64_t zc; return (A.constInt(z, zc) && zc == 0) || (z->k == EK::Const && z->c.isZero()); };
  auto nullFold = [&](EP e) {
    return A.rewrite(e, [&](EP x) -> EP {
      if (x->k != EK::Cmp || (x->op != CmpInst::ICMP_EQ && x->op != CmpInst::ICMP_NE)) return nullptr;
      EP p = x->args[0], z = x->args[1];
      if (!isNullConst(z)) { std::swap(p, z); if (!isNullConst(z)) return nullptr; }
      if (!p->ty->isPointerTy()) return nullptr;
      const bool ne = x->op == CmpInst::ICMP_NE;
      uint32_t argSym = ptrArgOf(p);
      if (argSym != ~0u) { assumeNonNull(argSym); return A.mkBool(ne); }
      EP s = A.unwrap(p);
      while (s->k == EK::Op && Instruction::isCast(s->op) && s->args.size() == 1) s = A.unwrap(s->args[0]);
      if (s->k != EK::Ite) return nullptr;
      const bool nullThen = isNullConst(s->args[1]), nullElse = isNullConst(s->args[2]);
      if (nullThen == nullElse) return nullptr;
      argSym = ptrArgOf(nullThen ? s->args[2] : s->args[1]);
      if (argSym == ~0u) return nullptr;
      assumeNonNull(argSym);
      EP isNull = nullThen ? s->args[0] : A.notb(s->args[0]);
      return ne ? A.notb(isNull) : isNull;
    }, nmemo);
  };
  // (twice: a null test of a selected address folds only once the enclosing selections decided the selection)
  // The elements' expressions share almost all of their structure: the simplifications are cached across them (one
  // cache per fold predicate; `allCache` serves every fold of every selection, below too)
  Arena::SimplifyUnderCache unfittableCache, allCache;
  // What holds whenever the K loop runs: the early returns' conditions, negated (`if (g0 >= g1 || n0 >= N) return`
  // leaves g0 < g1 and n0 < N). The frontend folds a guard these imply where it can see it (row 0's `n0 + 0 < N`
  // at the copy) and keeps it where it cannot (row r's `n0 + r < N`), so the elements of one tile differ by a
  // selection that is decided; the fitted template carries the selection for every row, and the proof that it
  // reproduces each element compares both sides with the decided selections folded.
  EP facts = nullptr;
  for (unsigned t = 0; t < T; t++) {
    EP ft = X.factsAt(t, K->getHeader());
    if (!facts) { facts = ft; continue; }
    std::vector<EP> la, lb; A.conjuncts(facts, la); A.conjuncts(ft, lb);
    std::set<EP> sb(lb.begin(), lb.end());
    EP common = A.mkBool(true); for (EP l : la) if (sb.count(l)) common = A.andb(common, l);
    facts = common;
  }
  if (!facts) facts = A.mkBool(true);
  // (a selection whose condition the facts decide is its arm; the literals of a condition the facts imply drop -
  // row 0's `0 < M` is one the template carries for every row and the facts fold away for that one)
  std::map<EP, Arena::RewriteMemo> foldMemos;
  std::function<EP(EP, EP)> foldUnder = [&](EP e, EP F) -> EP {
    if (A.isTrue(F)) return e;
    Arena::RewriteMemo& memo = foldMemos[F];
    // (decided as the path condition decides a literal (residue): refuting its negation, clamp and division aware)
    auto holds = [&](EP l) { return A.implies(F, l) || X.refutes(F, A.notb(l)); };
    return A.rewrite(e, [&](EP x) -> EP {
      if (x->k != EK::Ite) return nullptr;
      if (A.implies(F, x->args[0])) return x->args[1];
      if (A.disjoint(F, x->args[0]) || X.refutes(F, x->args[0])) return x->args[2];
      std::vector<EP> ls; A.conjuncts(x->args[0], ls);
      EP c2 = A.mkBool(true); bool changed = false;
      for (EP l : ls) { if (holds(l)) changed = true; else c2 = A.andb(c2, l); }
      if (!changed) return nullptr;
      if (A.isTrue(c2)) return x->args[1];
      return A.ite(c2, foldUnder(x->args[1], F), foldUnder(x->args[2], F));
    }, memo);
  };

  // A selection the word's own path condition decides is folded: the copy of a later stage is issued under
  // `stage < stages` and the conversion that consumes it (where the fill is emitted) runs under the same (as
  // `stages != 1, != 2, ...` and a clamp on the groups left, which the clamp-aware test relates); the operand keeps
  // the shape of the first stage's. The test is per literal of the selection's condition, with the clamp and
  // division bounds on.
  std::map<EP, Arena::RewriteMemo> condMemo;
  std::map<std::pair<EP, EP>, EP> residueMemo;
  // the literals of c the path condition does not decide (c itself when it decides none, true when all): a literal
  // is decided when the condition (its guards read through their definitions) refutes the literal's negation
  std::function<EP(EP, EP)> residue = [&](EP cond, EP c) -> EP {
    auto it = residueMemo.find({cond, c});
    if (it != residueMemo.end()) return it->second;
    auto refuted = [&](EP e) { return A.disjoint(cond, e) || X.refutes(cond, e); };
    EP r; bool any = false;
    if (c->k == EK::Op && c->op == Instruction::Or && c->ty->isIntegerTy(1)) {
      // a disjunction (a byte two copies may have written): the disjuncts the condition refutes drop, the ones
      // left keep the literals it does not decide
      std::vector<EP> ds; std::function<void(EP)> fl = [&](EP d) { if (d->k == EK::Op && d->op == Instruction::Or && d->ty->isIntegerTy(1)) { fl(d->args[0]); fl(d->args[1]); } else ds.push_back(d); }; fl(c);
      r = A.mkBool(false);
      for (EP d : ds) { if (refuted(d)) { any = true; continue; } EP d2 = residue(cond, d); if (d2 != d) any = true; r = A.orb(r, d2); }
      // A clause of the result that implies another (its literals bound the same polynomials at least as tightly:
      // `row 24 < N` beside `row 8 < N`, what a join of one thread's two chunks leaves once the complementary
      // literal resolves) says nothing the other does not: dropped, so that the guard reads as the one literal the
      // other rows carry (orb keeps the syntactic clauses; the retiler compares shapes). Until no clause drops.
      for (bool again = true; again;) {
        again = false;
        std::vector<EP> cls; std::function<void(EP)> flr = [&](EP d) { if (d->k == EK::Op && d->op == Instruction::Or && d->ty->isIntegerTy(1)) { flr(d->args[0]); flr(d->args[1]); } else cls.push_back(d); }; flr(r);
        if (cls.size() < 2) break;
        std::vector<EP> keep;
        for (size_t i = 0; i < cls.size(); i++) {
          std::vector<EP> li; A.conjuncts(cls[i], li);
          bool drop = false;
          for (size_t j = 0; j < cls.size() && !drop; j++) {
            if (i == j || cls[i] == cls[j]) continue;
            std::vector<EP> lj; A.conjuncts(cls[j], lj);
            bool sub = true;   // cls[i] => cls[j]: every literal of j is implied by one of i
            for (EP m : lj) { bool ok = false; for (EP l : li) if (A.litImplies(l, m)) { ok = true; break; } if (!ok) { sub = false; break; } }
            if (!sub) continue;
            bool back = true;   // (mutual implication: the one of higher id goes)
            for (EP l : li) { bool ok = false; for (EP m : lj) if (A.litImplies(m, l)) { ok = true; break; } if (!ok) { back = false; break; } }
            if (!back || cls[j]->id < cls[i]->id) drop = true;
          }
          if (!drop) keep.push_back(cls[i]);
        }
        if (keep.size() == cls.size()) break;
        any = again = true;
        r = A.mkBool(false);
        for (EP d : keep) r = A.orb(r, d);
      }
      // A disjunction whose clauses each imply one literal L and together cover it - `(col < N ∧ N < col + 17) ∨
      // (M < 6 ∧ col + 16 < N) ∨ (M > 1 ∧ col < N)`, the join of the paths a row-count test splits, every path
      // loading the same word where `col < N` - is that literal: the clauses imply L syntactically, and L implies
      // the disjunction when the prover refutes `cond ∧ L ∧ ¬clause_1 ∧ ... ∧ ¬clause_n` (the case split on M and
      // on N against the tile edge). The other rows carry the plain literal, and unify wants this one alike.
      {
        std::vector<EP> cls; std::function<void(EP)> flr = [&](EP d) { if (d->k == EK::Op && d->op == Instruction::Or && d->ty->isIntegerTy(1)) { flr(d->args[0]); flr(d->args[1]); } else cls.push_back(d); }; flr(r);
        if (cls.size() >= 2) {
          std::vector<EP> l0; A.conjuncts(cls[0], l0);
          for (EP L : l0) {
            bool all = true;
            for (size_t i = 1; i < cls.size() && all; i++) { std::vector<EP> li; A.conjuncts(cls[i], li); bool ok = false; for (EP l : li) if (A.litImplies(l, L)) { ok = true; break; } all = ok; }
            if (!all) continue;
            EP goal = L;
            for (EP d : cls) goal = A.andb(goal, A.notb(d));
            if (A.isFalse(goal) || refuted(goal)) { r = L; any = true; break; }
          }
        }
      }
    } else {
      std::vector<EP> cl; A.conjuncts(c, cl);
      r = A.mkBool(true);
      for (EP l : cl) {
        // (a literal the facts at the K loop decide - row 0's `n0 < N` - drops too: the body runs under the facts,
        // and a row whose literal the facts decide lacks it while the others carry theirs (unify))
        const bool held = A.implies(facts, l) || A.implies(cond, l) || refuted(A.notb(l));
        if (held) { any = true; continue; }
        // a literal the condition denies (iteration 2's word selects the stale or zeroed buffer under `stages < 3`,
        // and the iteration runs under `stages != 1 and != 2`): the selection is not taken
        if (refuted(l)) { any = true; r = A.mkBool(false); break; }
        r = A.andb(r, l);
      }
    }
    return residueMemo[{cond, c}] = any ? r : c;
  };
  auto condFold = [&](EP e, EP cond) {
    if (!cond || A.isTrue(cond)) return e;
    Arena::RewriteMemo& cm = condMemo[cond];
    std::function<EP(EP)> cb = [&](EP x) -> EP {
      if (x->k != EK::Ite) return nullptr;
      // (rewrite() is top-down and does not descend into a replacement: the arms first, by hand)
      EP thn = A.rewrite(x->args[1], cb, cm), els = A.rewrite(x->args[2], cb, cm);
      EP r = residue(cond, x->args[0]);
      // the selection a later stage's copy left behind in a reused buffer, under the same row guard: not taken
      // where the outer one is not (`ite(c, a, ite(c and d, b, 0))` is `ite(c, a, 0)`)
      while (els->k == EK::Ite && A.implies(els->args[0], r)) els = els->args[2];
      if (r == x->args[0] && thn == x->args[1] && els == x->args[2]) return x;
      return A.isTrue(r) ? thn : A.isFalse(r) ? els : A.ite(r, thn, els);
    };
    return A.rewrite(e, cb, cm);
  };
  auto normalize = [&](EP e, EP cond) { e = A.simplifyUnder(nullFold(stripGuards(condFold(A.subst(e, X.resolved, memo), cond))), unfittable, &unfittableCache); return A.simplifyUnder(nullFold(e), unfittable, &unfittableCache); };
  for (auto& s : aS) s.e = normalize(s.e, s.cond);
  for (auto& s : bS) s.e = normalize(s.e, s.cond);
  for (auto& kv : X.assumptions) if (kv.second.empty()) note("decode retiling assumes " + A.str(kv.first.second, 6) + " (device addresses derived from the argument are non-null; the exact twin runs when it fails)");
  // An n8 block of an operand whose every word is the constant zero is padding (recovery pads a single n8 tile to
  // the 16 columns of the tensor op): it takes no part in the fit, and its slot words are emitted as zero.
  std::function<bool(EP)> zeroExpr = [&](EP e) -> bool {
    int64_t c; if (A.constInt(e, c)) return c == 0;
    if (e->k == EK::Const) return e->c.isZero();
    if (e->k == EK::Concat) { for (EP p : e->args) if (!zeroExpr(p)) return false; return true; }
    if (e->k == EK::Bytes) return zeroExpr(e->args[0]);
    return false;
  };
  std::vector<std::set<int64_t>> padA(nChains), padB(nChains);   // chain -> the row/8 blocks that are padding
  auto padBlocks = [&](std::vector<Sample>& S, std::vector<std::set<int64_t>>& pads, const char* what) {
    std::map<std::pair<int64_t, int64_t>, bool> allZero;   // (chain, row/8) -> every word zero
    for (auto& s : S) { auto key = std::make_pair((int64_t)s.chain, s.row / 8); const bool z = zeroExpr(s.e); auto it = allZero.find(key); if (it == allZero.end()) allZero[key] = z; else it->second = it->second && z; }
    for (auto& kv : allZero) if (kv.second) { pads[kv.first.first].insert(kv.first.second); note(std::string(what) + " operand: rows " + std::to_string(8 * kv.first.second) + ".." + std::to_string(8 * kv.first.second + 7) + " are zero padding (no fill emitted)"); }
    S.erase(std::remove_if(S.begin(), S.end(), [&](const Sample& s) { return pads[s.chain].count(s.row / 8) != 0; }), S.end());
  };
  padBlocks(aS, padA, "A"); padBlocks(bS, padB, "B");
  // Each discovered c_j as an affine image of the role-assigned (row, k, iter, warp, chain), so emit
  // can rebuild the fit basis at a slot without the source thread.
  std::vector<Affine> discCFit(Disc.d);
  if (Disc.d) {
    std::vector<std::vector<int64_t>> L;
    std::vector<std::vector<int64_t>> C;
    auto take = [&](const std::vector<Sample>& S) {
      for (auto& s : S) {
        L.push_back({s.row, s.k, (int64_t)s.iter, (int64_t)s.warp, (int64_t)s.chain});
        C.push_back(discoveredPhys(s.thread, s.iter));
      }
    };
    take(aS); take(bS);
    if (!L.empty()) {
      std::vector<char> present(L.size(), 1);
      for (unsigned j = 0; j < Disc.d; j++) {
        std::vector<int64_t> vals(L.size());
        for (size_t i = 0; i < L.size(); i++) vals[i] = j < C[i].size() ? C[i][j] : 0;
        fitAffine(L, vals, present, discCFit[j]);
      }
      note("coordinate discovery: " + std::to_string(Disc.d) + " logical coordinate(s) imaged from Phys; operand fit uses the quasi-affine transform of the role-assigned (row, k)");
    }
  }

  auto fitOperand = [&](std::vector<Sample>& S, const char* what, Shaper& sh, OperandFit& fit) -> bool {
    for (size_t i = 0; i < S.size(); i++) {
      Node* t = sh.build(S[i].e);
      if (!t) { std::string co; for (auto c : S[i].coords) co += (co.empty() ? "" : ",") + std::to_string(c); kept(std::string(what) + " " + coordNames + "=(" + co + "): " + sh.why); return false; }
      if (!fit.tmpl) { fit.tmpl = t; fit.sample = S[i].e; sh.keepSample(true); continue; }
      bool ok = sh.unify(fit.tmpl, t, (unsigned)i);
      sh.keepSample(sh.adoptedAny);
      if (!ok) {
        std::string co; for (auto c : S[i].coords) co += (co.empty() ? "" : ",") + std::to_string(c);
        std::string co0; for (auto c : S[0].coords) co0 += (co0.empty() ? "" : ",") + std::to_string(c);
        kept(std::string(what) + ": " + sh.why + " (element 0 at " + coordNames + "=(" + co0 + ") vs element " + std::to_string(i) + " at (" + co + "): " + A.str(fit.sample, 4) + " vs " + A.str(S[i].e, 4) + ")");
        return false;
      }
    }
    sh.nSamples = (unsigned)S.size();
    sh.holes(fit.tmpl);
    std::vector<std::vector<int64_t>> coords; for (auto& s : S) coords.push_back(s.coords);
    // Address holes come from discovered φ evaluated at Phys, then fitted in the 3.4 (row,k) basis
    // so Materializer still instantiates the same slot coordinates (byte-identical MSL).
    unsigned holesFromDisc = 0;
    {
      auto xPhys = [&](const Sample& s) {
        std::vector<int> x;
        for (auto& ax : Disc.axes) {
          if (ax.kind == coords::Axis::Derived) break;
          if (ax.kind == coords::Axis::Simdgroup) x.push_back((int)(s.thread / 32));
          else if (ax.kind == coords::Axis::Lane) x.push_back((int)(s.thread % 32));
          else if (ax.kind == coords::Axis::Iter) x.push_back(ax.loop == K ? (int)s.iter : 0);
          else if (ax.kind == coords::Axis::TidBit) x.push_back((int)((s.thread >> ax.index) & 1));
          else x.push_back(0);
        }
        return x;
      };
      auto siteIntAt = [&](const coords::SiteMap& m, const std::vector<int>& x, int64_t& v) -> bool {
        for (auto& p : m.points) {
          if (p.first.size() < x.size()) continue;
          bool eq = true;
          for (size_t i = 0; i < x.size(); i++) if (p.first[i] != x[i]) { eq = false; break; }
          if (!eq) continue;
          return A.constInt(p.second, v);
        }
        return false;
      };
      for (size_t h = 0; h < sh.holeList.size(); h++) {
        auto& vals = *sh.holeList[h].first;
        auto& present = *sh.holeList[h].second;
        for (auto& m : Disc.sites) {
          if (!m.fitted() || m.isPred || m.isDerivedOperand) continue;
          bool cover = true;
          for (size_t i = 0; i < S.size() && cover; i++) {
            if (!present[i]) continue;
            int64_t sv = 0;
            if (!siteIntAt(m, xPhys(S[i]), sv) || sv != vals[i]) cover = false;
          }
          if (cover) { holesFromDisc++; break; }
        }
      }
      note(std::string(what) + ": " + std::to_string(holesFromDisc) + "/" + std::to_string(sh.holeList.size()) + " hole(s) covered by discovered site maps");
    }
    fit.holes.resize(sh.holeList.size());
    for (size_t h = 0; h < sh.holeList.size(); h++) if (!fitAffine(coords, *sh.holeList[h].first, *sh.holeList[h].second, fit.holes[h])) {
      kept(std::string(what) + ": a coefficient of the element expression is not affine in " + coordNames); return false;
    }
    // the proof: the fitted template, instantiated at every element's coordinates, is that element's expression
    // (both with every selection an enclosing selection decides folded)
    auto all = [](EP) { return true; };
    // both sides with every 1*atom polynomial replaced by the atom (the executor and the instantiation wrap opaque
    // integer expressions differently)
    Arena::RewriteMemo cmemo;
    std::function<EP(EP)> canon = [&](EP e) { return A.rewrite(e, [&](EP x) -> EP { EP u = A.unwrap(x); return u != x ? canon(u) : nullptr; }, cmemo); };
    for (size_t i = 0; i < S.size(); i++) {
      // (under the facts and the element's own path condition: the executor dropped from the element's guards the
      // literals that condition implies - the mma of an m-tile runs under `mt * 16 < M`, so row 0's `0 < M` is gone
      // from the element while the template, fitted on every row, carries it)
      const EP under = A.andb(facts, S[i].cond);
      EP got = canon(foldUnder(A.simplifyUnder(sh.inst(fit.tmpl, fit.holes, S[i].coords), all, &allCache), under)), want = canon(foldUnder(A.simplifyUnder(S[i].e, all, &allCache), under));
      if (got != want) {
        std::string co; for (auto c : S[i].coords) co += (co.empty() ? "" : ",") + std::to_string(c);
        kept(std::string(what) + ": the fitted element expression does not reproduce element " + std::to_string(i) + " at " + coordNames + "=(" + co + ") (" + A.str(got, 5) + " vs " + A.str(want, 5) + ")");
        return false;
      }
    }
    fit.samples = (unsigned)S.size();
    return true;
  };
  // one fit per chain: the chains of an iteration are the job's 16-row sub-tiles (and their column blocks), and the
  // source guards them differently (the second sub-tile's rows exist only when the job has more than 16 rows, so its
  // elements carry an extra selection the first sub-tile's lack); a template is one shape, so each chain has its own
  std::vector<OperandFit> fitA(nChains), fitB(nChains);
  std::deque<Shaper> shA, shB;
  unsigned aSamples = 0, bSamples = 0;
  for (unsigned q = 0; q < nChains; q++) {
    shA.emplace_back(A); shB.emplace_back(A);
    std::vector<Sample> aq, bq;
    for (auto& s : aS) if (s.chain == q) aq.push_back(s);
    for (auto& s : bS) if (s.chain == q) bq.push_back(s);
    const std::string tag = nChains > 1 ? " (chain " + std::to_string(q) + ")" : std::string();
    if (aq.empty() || bq.empty()) return kept("an operand" + tag + " is all padding");
    if (!fitOperand(aq, ("A operand" + tag).c_str(), shA.back(), fitA[q]) || !fitOperand(bq, ("B operand" + tag).c_str(), shB.back(), fitB[q])) return false;
    aSamples += fitA[q].samples; bSamples += fitB[q].samples;
  }
  // A must not depend on the warp (every warp reads the same tile) - it may, then each warp streams its own
  auto describe = [&](const OperandFit& f, const char* nm) {
    std::string d = std::string(nm) + ": " + A.str(f.sample, 5) + "; " + std::to_string(f.holes.size()) + " coefficient(s) affine in " + coordNames + ": ";
    for (auto& h : f.holes) { d += "[" + std::to_string(h.c0); for (auto c : h.c) d += "," + std::to_string(c); d += "]"; }
    return d;
  };
  note("decode GEMM recovered: per warp per iteration " + std::to_string(nChains) + " group product(s) grp[16][" + std::to_string(N) + "] over K=" + std::to_string(Kiter) + " (" + std::to_string(chainLen) + " fused ops each), " + std::to_string(nWarps) + " warps, " + std::to_string(aSamples) + " A and " + std::to_string(bSamples) + " B element pairs fitted");
  for (unsigned q = 0; q < nChains; q++) { const std::string tag = nChains > 1 ? "chain " + std::to_string(q) + " " : std::string(); note(describe(fitA[q], (tag + "A(row, k)").c_str())); note(describe(fitB[q], (tag + "B(col, k)").c_str())); }

  // ---- 4. the decode mapping: what is deleted, what remains, where the tile goes (no IR changes yet)
  LLVMContext& ctx = F.getContext();
  Module& Mod = *F.getParent();
  Type* i8 = Type::getInt8Ty(ctx); Type* i32 = Type::getInt32Ty(ctx); Type* i64 = Type::getInt64Ty(ctx); Type* f32 = Type::getFloatTy(ctx);
  // Operand words are re-derived at the fragment's own coordinates and loaded from device.
  for (Instruction& I : instructions(F)) if (auto* CI = dyn_cast<CallInst>(&I)) if (CI->isInlineAsm()) { bool alu; std::string mn = asmMnemonics(CI, alu); if (mn.find("mma") != std::string::npos) return kept("a matmul the tensor recovery did not fuse remains (" + mn + ")"); }
  // the conversion sites (one per chain), used only by field
  std::map<Instruction*, unsigned> convSites;
  for (auto& kv : chains) for (size_t q = 0; q < kv.second.size(); q++) { auto ins = convSites.insert({kv.second[q].conv->I, (unsigned)q}); if (!ins.second && ins.first->second != q) return kept("a conversion belongs to different chains on different threads or iterations"); }
  if (convSites.size() != nChains) return kept("the conversion sites and the chains differ in number");
  const unsigned cap = N / 2;   // fragment fields per lane
  for (auto& cs : convSites) {
    if (LI.getLoopFor(cs.first->getParent()) != K) return kept("a conversion outside the K loop");
    auto* st = dyn_cast<StructType>(cs.first->getType()); if (!st || st->getNumElements() != cap) return kept("a conversion of an unexpected type");
    for (User* u : cs.first->users()) if (!isa<ExtractValueInst>(u)) return kept("a conversion result used other than by field");
  }
  // the slice the conversions consume: matmuls, fragment conversions, operand words, ring loads and their
  // addresses, the dequantization - every instruction all of whose users are in the slice and which has no effect
  auto deletable = [&](Instruction* I) -> bool {
    if (I->isTerminator()) return false;
    if (isa<PHINode>(I)) return true;   // the merges of the operands' zero-fill diamonds
    if (auto* CI = dyn_cast<CallInst>(I)) {
      if (CI->isInlineAsm()) { bool alu; asmMnemonics(CI, alu); return alu; }
      Function* cf = CI->getCalledFunction(); if (!cf) return false;
      StringRef n = cf->getName();
      if (n.starts_with("__mvcc_tp_")) return true;
      if (n.starts_with("__nv_")) return cf->onlyReadsMemory() || cf->doesNotAccessMemory();
      if (cf->isIntrinsic()) return !CI->mayHaveSideEffects();
      return false;
    }
    if (auto* LD = dyn_cast<LoadInst>(I)) return !LD->isVolatile();
    return !I->mayHaveSideEffects();
  };
  // tensor mode: the operand blocks of the fused ops - each a 16x16 block held as a fragment conversion (register-
  // built PTX-layout words -> slot words) - with the block's position (chain, op, K block, N block), the same
  // instructions on every thread and iteration. The conversion goes: the slot words are re-derived from the element
  // function at the slots' own coordinates, and the PTX-layout words with everything that built them are deleted.
  struct BlockPos { CallInst* src; unsigned q, p, kb, nb; bool isB; };
  std::vector<BlockPos> blockPos;
  std::set<Instruction*> consumers;   // the fragment conversions (deleted with their words)
  {
    const auto& cs0 = chains.begin()->second;
    std::map<Instruction*, size_t> seen;
    auto add = [&](EP arg, unsigned q, unsigned p, unsigned kb, unsigned r, unsigned nb, bool isB) -> bool {
      Instruction* src; unsigned f;
      if (!producedBy(arg, src, f) || f != r) return false;
      auto* CI = dyn_cast<CallInst>(src); if (!CI || CI->arg_size() != 5) return false;
      auto it = seen.find(src);
      if (it == seen.end()) { seen[src] = blockPos.size(); blockPos.push_back({CI, q, p, kb, nb, isB}); consumers.insert(src); return true; }
      const BlockPos& b = blockPos[it->second];
      return b.q == q && b.p == p && b.kb == kb && b.nb == nb && b.isB == isB;
    };
    for (size_t q = 0; q < cs0.size(); q++) for (size_t p = 0; p < cs0[q].ops.size(); p++) {
      const Op& op = cs0[q].ops[p]; const int LV = op.Kop / 16, NTL = op.N / 8;
      for (int kb = 0; kb < LV; kb++) for (int r = 0; r < 4; r++) if (!add(op.call->args[6 + 4 * kb + r], q, p, kb, r, 0, false)) return kept("an A operand block is not one register-built fragment conversion at one position");
      for (int kb = 0; kb < LV; kb++) for (int nb = 0; nb < NTL / 2; nb++) for (int r = 0; r < 4; r++) if (!add(op.call->args[6 + op.nA + 4 * (nb + (NTL / 2) * kb) + r], q, p, kb, r, nb, true)) return kept("a B operand block is not one register-built fragment conversion at one position");
    }
    for (const BlockPos& b : blockPos) for (User* u : b.src->users()) if (!isa<ExtractValueInst>(u)) return kept("a fragment conversion used other than by field");
  }
  std::set<Instruction*> dead; std::vector<Instruction*> work;
  auto usersDead = [&](Instruction* OI) { for (User* u : OI->users()) { auto* UI = dyn_cast<Instruction>(u); if (!UI || (!dead.count(UI) && !consumers.count(UI))) return false; } return true; };
  for (auto& b : blockPos) for (unsigned f = 0; f < 4; f++) { auto* WI = dyn_cast<Instruction>(b.src->getArgOperand(1 + f)); if (WI && !dead.count(WI) && deletable(WI) && usersDead(WI)) { dead.insert(WI); work.push_back(WI); } }
  while (!work.empty()) {
    Instruction* I = work.back(); work.pop_back();
    for (Value* op : I->operands()) {
      auto* OI = dyn_cast<Instruction>(op);
      if (!OI || dead.count(OI) || !deletable(OI)) continue;
      if (usersDead(OI)) { dead.insert(OI); work.push_back(OI); }
    }
  }
  // Quantizer loads of the bf16 activation ring (and any other non-operand shared load a staging copy feeds): the
  // same resolution the operand fit already has, emitted as a function of (row, column, K iteration). The team
  // formula is `team = (tid >> 3) + visit * step`, `r = team / kGps`, `col = (team % kGps) * 128 + (tid & 7) * 16`
  // (decode.cuh: 8-thread teams, one 128-wide group). Failure leaves the ring as today — the operand retile stays.
  std::set<Instruction*> ringReads;
  std::deque<Shaper> ringSh;
  struct RingLoad { LoadInst* L; OperandFit aFit, gFit; unsigned aSh = 0, gSh = 0; EP expr = nullptr; };
  std::vector<RingLoad> ringLoads;
  int ringKGps = 1;
  Symbol symQRow, symQCol, symQIter;
  uint32_t idQRow = 0, idQCol = 0, idQIter = 0;
  PHINode* ringTeamPhi = nullptr;
  {
    std::set<LoadInst*> cands;
    for (Instruction* Cp : X.ringCopies) {
      auto rd = X.writeReaders.find(Cp);
      if (rd == X.writeReaders.end()) continue;
      for (Instruction* r : rd->second)
        if (!dead.count(r) && !X.stayingReads.count(r))
          if (auto* LD = dyn_cast<LoadInst>(r)) cands.insert(LD);
    }
    Loop* teamLoop = nullptr;
    int64_t teamStep = 16;
    for (LoadInst* L : cands) {
      for (Loop* Lx = LI.getLoopFor(L->getParent()); Lx && Lx != K && K->contains(Lx); Lx = Lx->getParentLoop()) {
        unsigned tc = SE.getSmallConstantTripCount(Lx);
        if (!tc && concretelyBounded(Lx)) tc = SE.getSmallConstantMaxTripCount(Lx);
        if (tc == 8) continue;   // the unrolled-failed i-loop over 8 uint32 words
        teamLoop = Lx;
        break;
      }
      if (teamLoop) break;
    }
    if (teamLoop) {
      for (PHINode& P : teamLoop->getHeader()->phis()) {
        const SCEV* s = SE.getSCEV(&P);
        if (auto* AR = dyn_cast<SCEVAddRecExpr>(s)) if (AR->getLoop() == teamLoop && AR->isAffine())
          if (auto* C = dyn_cast<SCEVConstant>(AR->getStepRecurrence(SE))) {
            int64_t st = C->getAPInt().getSExtValue();
            if (st > 0) { teamStep = st; ringTeamPhi = &P; break; }
          }
      }
    }
    auto collectWord = [&](LoadInst* L, unsigned thread, unsigned seq, EP& word, EP& cond) -> bool {
      const unsigned nB = (unsigned)DL.getTypeStoreSize(L->getType());
      std::vector<const SmemRead*> bs;
      for (const SmemRead& rr : X.smemReads) if (rr.site == L && rr.thread == thread && rr.seq == seq) bs.push_back(&rr);
      if (bs.size() != nB) return false;
      std::sort(bs.begin(), bs.end(), [](const SmemRead* a, const SmemRead* b) { return a->off < b->off; });
      for (unsigned i = 1; i < bs.size(); i++) if (bs[i]->off != bs[0]->off + (int64_t)i) return false;
      std::vector<EP> bytes;
      for (const SmemRead* rr : bs) { auto it = X.resolved.find(rr->sym); if (it == X.resolved.end()) return false; bytes.push_back(it->second); }
      word = A.fromBytes(bytes, L->getType()); cond = bs[0]->cond; return true;
    };
    auto fitSilent = [&](std::vector<Sample>& S, Shaper& sh, OperandFit& fit, std::string& why, bool verify = true) -> bool {
      for (size_t i = 0; i < S.size(); i++) {
        if (zeroExpr(S[i].e)) continue;   // a decided-false guard (no K group left): the else arm, not a new shape
        Node* t = sh.build(S[i].e);
        if (!t) { why = "build: " + sh.why; return false; }
        if (!fit.tmpl) { fit.tmpl = t; fit.sample = S[i].e; sh.keepSample(true); continue; }
        bool ok = sh.unify(fit.tmpl, t, (unsigned)i);
        sh.keepSample(sh.adoptedAny);
        if (!ok) {
          why = "unify @" + std::to_string(i) + ": " + sh.why;
          return false;
        }
      }
      if (!fit.tmpl) { why = "every ring-read sample is zero"; return false; }
      sh.nSamples = (unsigned)S.size();
      sh.holes(fit.tmpl);
      std::vector<std::vector<int64_t>> coords; for (auto& s : S) coords.push_back(s.coords);
      fit.holes.resize(sh.holeList.size());
      for (size_t h = 0; h < sh.holeList.size(); h++) if (!fitAffine(coords, *sh.holeList[h].first, *sh.holeList[h].second, fit.holes[h])) { why = "hole " + std::to_string(h) + " is not affine"; return false; }
      auto all = [](EP) { return true; };
      Arena::RewriteMemo cmemo;
      std::function<EP(EP)> canon = [&](EP e) { return A.rewrite(e, [&](EP x) -> EP { EP u = A.unwrap(x); return u != x ? canon(u) : nullptr; }, cmemo); };
      if (verify) for (size_t i = 0; i < S.size(); i++) {
        const EP under = A.andb(facts, S[i].cond);
        EP got = canon(foldUnder(A.simplifyUnder(sh.inst(fit.tmpl, fit.holes, S[i].coords), all, &allCache), under));
        EP want = canon(foldUnder(A.simplifyUnder(S[i].e, all, &allCache), under));
        if (got != want) { why = "verify @" + std::to_string(i); return false; }
      }
      fit.samples = (unsigned)S.size();
      return true;
    };
    std::string ringWhy;
    auto tryGps = [&](int kGps) -> bool {
      ringSh.clear(); ringLoads.clear();
      bool needTeam = false;
      ringWhy.clear();
      for (LoadInst* L : cands) {
        std::map<unsigned, std::vector<unsigned>> seqs;   // thread -> seqs in execution order
        std::map<std::pair<unsigned, unsigned>, std::pair<EP, EP>> words;
        std::set<std::pair<unsigned, unsigned>> seen;
        for (const SmemRead& rr : X.smemReads) {
          if (rr.site != L || !seen.insert({rr.thread, rr.seq}).second) continue;
          EP word = nullptr, cond = nullptr;
          if (!collectWord(L, rr.thread, rr.seq, word, cond)) { ringWhy = "a ring load's bytes did not resolve"; return false; }
          words[{rr.thread, rr.seq}] = {word, cond};
          seqs[rr.thread].push_back(rr.seq);
        }
        if (seqs.empty()) { ringWhy = "a ring load has no shared reads"; return false; }
        std::map<unsigned, std::vector<const Executor::SmemAccess*>> accs;
        for (auto& a : Sig.smemAccesses) if (a.I == L && a.concrete && !a.isStore) accs[a.thread].push_back(&a);
        std::vector<Sample> S;
        for (auto& kv : seqs) {
          auto& as = accs[kv.first];
          if (as.size() != kv.second.size()) { ringWhy = "a ring load's executions do not match its shared accesses"; return false; }
          std::map<unsigned, std::vector<size_t>> byIter;
          for (size_t i = 0; i < kv.second.size(); i++) byIter[as[i]->iter].push_back(i);
          for (auto& g : byIter) {
            if (g.second.size() > 1 && !teamLoop) { ringWhy = "a ring load is in a rolled word loop"; return false; }
            for (size_t v = 0; v < g.second.size(); v++) {
              if (v) needTeam = true;
              const size_t i = g.second[v];
              const unsigned team = (kv.first >> 3) + (unsigned)v * (unsigned)teamStep;
              auto& wc = words[{kv.first, kv.second[i]}];
              Sample s;
              s.coords = {(int64_t)(team / (unsigned)kGps), (int64_t)(team % (unsigned)kGps) * 128 + (int64_t)(kv.first & 7) * 16, (int64_t)as[i]->iter};
              s.e = foldUnder(normalize(wc.first, wc.second), A.andb(facts, wc.second));
              s.cond = wc.second;
              S.push_back(s);
            }
          }
        }
        auto isLd = [&](EP x) { while (x->k == EK::Bytes) x = x->args[0]; return x->k == EK::Load; };
        auto splitWord = [&](EP e, EP& g, EP& addr) -> bool {
          if (zeroExpr(e)) { g = A.mkBool(false); addr = nullptr; return true; }
          g = A.mkBool(true);
          while (e->k == EK::Ite) {
            if (zeroExpr(e->args[1])) { g = A.andb(g, A.notb(e->args[0])); e = e->args[2]; }
            else if (zeroExpr(e->args[2])) { g = A.andb(g, e->args[0]); e = e->args[1]; }
            else if (isLd(e->args[1])) { g = A.andb(g, e->args[0]); e = e->args[1]; }   // latest write; drop the stale ring slot
            else if (isLd(e->args[2])) { g = A.andb(g, A.notb(e->args[0])); e = e->args[2]; }
            else return false;
          }
          // bytes(ld16, lo, 4) is the word at lo inside a 16-byte cp.async; stripping the wrapper
          // without adding lo made all four uint32s of a granule share the granule's base (the
          // quantizer then saw 2 distinct device words instead of 8).
          unsigned lo = 0;
          while (e->k == EK::Bytes) { lo += e->lo; e = e->args[0]; }
          if (e->k == EK::Poly && e->terms.size() == 1 && e->terms[0].second == 1 && e->terms[0].first.size() == 1)
            e = A.atomExpr(e->terms[0].first[0]);
          if (e->k != EK::Load) return false;
          addr = e->args[0];
          if (lo) addr = A.add(addr, A.mkInt(addr->ty, (int64_t)lo));
          return true;
        };
        std::vector<Sample> aS, gS;
        for (size_t i = 0; i < S.size(); i++) {
          EP g = nullptr, addr = nullptr;
          if (!splitWord(S[i].e, g, addr)) {
            ringWhy = "a ring-read word is not ite(g, load, 0)";
            return false;
          }
          if (!addr) continue;
          Sample as; as.coords = S[i].coords; as.e = addr; as.cond = S[i].cond; aS.push_back(as);
          Sample gs; gs.coords = S[i].coords; gs.e = g; gs.cond = S[i].cond; gS.push_back(gs);
        }
        if (aS.empty()) { ringWhy = "every ring-read sample is zero"; return false; }
        ringSh.emplace_back(A);
        OperandFit aFit;
        std::string aWhy;
        if (!fitSilent(aS, ringSh.back(), aFit, aWhy)) {
          ringWhy = "ring address: " + aWhy;
          return false;
        }
        ringSh.emplace_back(A);
        OperandFit gFit;
        std::string gWhy;
        if (!fitSilent(gS, ringSh.back(), gFit, gWhy, false)) {
          ringWhy = "ring guard: " + gWhy;
          return false;
        }
        RingLoad rl; rl.L = L; rl.aFit = aFit; rl.gFit = gFit;
        rl.aSh = (unsigned)ringSh.size() - 2; rl.gSh = (unsigned)ringSh.size() - 1;
        ringLoads.push_back(rl);
      }
      if (needTeam && !ringTeamPhi) { ringWhy = "a team loop has no induction phi"; return false; }
      if (ringLoads.empty()) { ringWhy = "no ring loads"; return false; }
      return true;
    };
    int fittedGps = 0;
    if (!cands.empty()) {
      // Larger kGps first: tryGps(1) can unify a TBK=256 (2 groups/stage) full stage because
      // team aliases (row, group), but a last stage with groups_here=1 then loads the wrong K slice.
      if (tryGps(2)) fittedGps = 2;
      else if (tryGps(1)) fittedGps = 1;
    }
    if (fittedGps) {
      ringKGps = fittedGps;
      symQRow.kind = Symbol::Coord; symQRow.name = "qrow"; symQRow.ty = i64; idQRow = A.mkSymRaw(symQRow)->sym;
      symQCol.kind = Symbol::Coord; symQCol.name = "qcol"; symQCol.ty = i64; idQCol = A.mkSymRaw(symQCol)->sym;
      symQIter.kind = Symbol::Coord; symQIter.name = "qiter"; symQIter.ty = i64; idQIter = A.mkSymRaw(symQIter)->sym;
      IRBuilder<> dummy(ctx);
      Materializer chk(A, F, dummy);
      chk.bound = {idQRow, idQCol, idQIter};
      chk.speculate = true;
      bool emitOk = true;
      for (size_t i = 0; i < ringLoads.size(); i++) {
        bool okA = true, okG = true;
        std::vector<EP> co{A.mkSym(symQRow), A.mkSym(symQCol), A.mkSym(symQIter)};
        EP addr = ringSh[ringLoads[i].aSh].instSym(ringLoads[i].aFit.tmpl, ringLoads[i].aFit.holes, co, okA);
        EP g = ringSh[ringLoads[i].gSh].instSym(ringLoads[i].gFit.tmpl, ringLoads[i].gFit.holes, co, okG);
        if (!okA || !addr) { emitOk = false; ringWhy = "a ring-read address cannot be re-derived at (row, column, iter)"; break; }
        if (!okG || !g) { emitOk = false; ringWhy = "a ring-read guard cannot be re-derived at (row, column, iter)"; break; }
        // Drop conjuncts that mention the K iteration. The fit sees only full stages (smin(..., kGps) > 1);
        // a last stage narrower than TBK then selected zero and lost that group's contribution. The row
        // bound (r < M) stays; unused groups of a partial stage already store a zero scale.
        {
          std::function<bool(EP)> mentionsIter = [&](EP x) -> bool {
            if (!x) return false;
            if (x->k == EK::Sym) return x->sym == idQIter;
            if (x->k == EK::Poly) {
              for (auto& t : x->terms) for (uint32_t a : t.first) {
                uint32_t sid; if (A.isSymAtom(a, sid) && sid == idQIter) return true;
                if (mentionsIter(A.atomExpr(a))) return true;
              }
              return false;
            }
            for (EP a : x->args) if (mentionsIter(a)) return true;
            return false;
          };
          std::vector<EP> cs; A.conjuncts(g, cs);
          EP gRow = A.mkBool(true);
          for (EP c : cs) if (!mentionsIter(c)) gRow = A.andb(gRow, c);
          g = gRow;
        }
        const unsigned nB = (unsigned)DL.getTypeStoreSize(ringLoads[i].L->getType());
        EP ld = A.load(ringLoads[i].L->getType(), addr, A.mkBool(true), nB);
        EP e = A.ite(g, ld, A.mkInt(ringLoads[i].L->getType(), 0));
        if (!chk.can(e)) { emitOk = false; ringWhy = "a ring-read word cannot be emitted: " + chk.why; break; }
        ringLoads[i].expr = e;
        ringReads.insert(ringLoads[i].L);
      }
      if (!emitOk) { ringReads.clear(); ringLoads.clear(); ringSh.clear(); idQRow = idQCol = idQIter = 0; ringTeamPhi = nullptr; }
    }
  }
  // staging copies whose bytes only the slice and the rewritten ring reads consume: deleted with them. A copy at a
  // symbolic destination (the scale table's) stays.
  std::set<Instruction*> deadCopies;
  for (Instruction* Cp : X.ringCopies) {
    bool any = false, concrete = true;
    for (auto& a : Sig.smemAccesses) if (a.I == Cp) { any = true; concrete = concrete && a.concrete; }
    if (!any || !concrete) continue;
    bool allDead = true;
    auto rd = X.writeReaders.find(Cp);
    if (rd != X.writeReaders.end()) for (Instruction* r : rd->second) if (!dead.count(r) && !ringReads.count(r)) { allDead = false; break; }
    if (allDead) deadCopies.insert(Cp);
  }
  // prologue zero-fills of the ring (fp8-activation leftover; bf16 ring once its loads rewrite to device): unread,
  // or read only by the deleted slice / rewritten ring loads. Stores that overlap a staying operand (quantizer
  // output the MMA still reads) stay — deleting them would put those bytes under the freed region.
  {
    int64_t stayLo = INT64_MAX;
    for (auto& a : Sig.smemAccesses) if (dead.count(a.I) && X.stayingReads.count(a.I) && a.concrete) stayLo = std::min(stayLo, a.off);
    std::set<Instruction*> stores;
    for (auto& a : Sig.smemAccesses) if (a.isStore) stores.insert(a.I);
    for (Instruction* St : stores) {
      if (dead.count(St) || deadCopies.count(St)) continue;
      auto* SI = dyn_cast<StoreInst>(St);
      if (!SI || !isa<Constant>(SI->getValueOperand())) continue;
      bool any = false, concrete = true, underStay = false;
      for (auto& a : Sig.smemAccesses) if (a.I == St) { any = true; concrete = concrete && a.concrete; if (a.off + (int64_t)a.bytes > stayLo) underStay = true; }
      if (!any || !concrete || underStay) continue;
      bool allGone = true;
      auto rd = X.writeReaders.find(St);
      if (rd != X.writeReaders.end()) for (Instruction* r : rd->second) if (!dead.count(r) && !ringReads.count(r)) { allGone = false; break; }
      if (allGone) deadCopies.insert(St);
    }
  }
  if (deadCopies.empty()) return kept("no staging copy feeds the matmuls only");
  auto objOf = [&](EP addr) -> GlobalVariable* {
    if (!addr->ty->isIntegerTy() && !addr->ty->isPointerTy()) return nullptr;
    EP p = A.asPoly(addr);
    for (auto& t : p->terms) { uint32_t sid; if (t.first.size() == 1 && A.isSymAtom(t.first[0], sid) && A.symbol(sid).kind == Symbol::SharedBase) return cast<GlobalVariable>(A.symbol(sid).v); }
    return nullptr;
  };
  GlobalVariable* dyn = nullptr;
  for (auto& a : Sig.smemAccesses) if (deadCopies.count(a.I)) { GlobalVariable* o = objOf(a.addr); if (!o) return kept("a deleted copy's object is unknown"); if (dyn && o != dyn) return kept("the deleted copies stage into two shared objects"); dyn = o; }
  // The remaining accesses of that object. Each must lie at or above a lower bound the analysis can prove: a
  // concrete offset, or a symbolic one whose variable part is a non-negative combination of values known
  // non-negative (block/thread registers, induction variables ScalarEvolution bounds). The proof serves twice: the
  // reads the operands were resolved through saw only concrete writes, which is sound only if no symbolic write can
  // reach their bytes; and the region above the bound is relocated downwards over the freed ring.
  auto nonNegSreg = [](const std::string& n) { for (const char* s : {"tid", "ntid", "ctaid", "nctaid", "laneid", "warpsize", "nwarpid", "warpid"}) if (n.find(s) != std::string::npos) return true; return false; };
  std::function<bool(EP)> nonNeg = [&](EP e) -> bool {
    int64_t c; if (A.constInt(e, c)) return c >= 0;
    switch (e->k) {
      case EK::Poly: for (auto& t : e->terms) { if (t.second < 0) return false; for (uint32_t a : t.first) if (!nonNeg(A.atomExpr(a))) return false; } return true;
      case EK::Sym: {
        const Symbol& s = A.symbol(e->sym);
        if (s.kind == Symbol::Sreg) return nonNegSreg(s.name);
        if ((s.kind == Symbol::LoopValue || s.kind == Symbol::ThreadValue) && s.v && s.v->getType()->isIntegerTy() && SE.isSCEVable(s.v->getType())) return SE.isKnownNonNegative(SE.getSCEV(s.v));
        // an integer argument (a dimension): assumed non-negative, checked at the launch
        if (s.kind == Symbol::Arg && s.ty->isIntegerTy() && !s.ty->isIntegerTy(1)) { X.assumptions[{A.mkBool(true), A.cmp(CmpInst::ICMP_SGE, A.mkSym(s), A.mkInt(s.ty, 0))}]; return true; }
        return false;
      }
      case EK::Op:
        switch (e->op) {
          case Instruction::ZExt: return true;
          case Instruction::SExt: case Instruction::UDiv: case Instruction::URem: return nonNeg(e->args[0]);
          case Instruction::LShr: { int64_t sh; return (A.constInt(e->args[1], sh) && sh > 0) || nonNeg(e->args[0]); }
          case Instruction::Mul: case Instruction::Or: case Instruction::SDiv: case Instruction::SRem: return nonNeg(e->args[0]) && nonNeg(e->args[1]);
          case Instruction::And: return nonNeg(e->args[0]) || nonNeg(e->args[1]);
          default: return false;
        }
      case EK::Ite: return nonNeg(e->args[1]) && nonNeg(e->args[2]);
      case EK::Call: return minMaxCallee(e->callee) != 0 && nonNeg(e->args[0]) && nonNeg(e->args[1]);
      default: return false;
    }
  };
  // the constant part of a shared address whose variable part is provably non-negative. A remainder the source
  // wrote as x % d appears as c*x - c*d*(x/d) (two monomials, one negative): recognized and replaced by
  // c*(x mod d), non-negative when x is (values are the model's unbounded integers, as the address arithmetic is).
  auto strip = [&](EP e) { for (;;) { EP u = A.unwrap(e); if (u->k == EK::Op && (u->op == Instruction::SExt || u->op == Instruction::ZExt)) { e = u->args[0]; continue; } return u; } };
  auto monoAtoms = [&](EP e, std::vector<EP>& out) -> bool {   // e (stripped) as one monomial with coefficient 1: its atoms, stripped
    out.clear();
    if (e->k != EK::Poly) { out.push_back(e); return true; }
    if (e->terms.size() != 1 || e->terms[0].second != 1 || e->terms[0].first.empty()) return false;
    for (uint32_t a : e->terms[0].first) out.push_back(strip(A.atomExpr(a)));
    std::sort(out.begin(), out.end(), [](EP x, EP y) { return x->id < y->id; }); return true;
  };
  auto symbolicLowerBound = [&](EP addr, int64_t& lb) -> bool {
    struct Term { std::vector<EP> atoms; int64_t coef; };
    std::vector<Term> ts; lb = 0;
    EP p = A.asPoly(addr);
    for (auto& t : p->terms) {
      uint32_t sid;
      if (t.first.empty()) { lb += t.second; continue; }
      if (t.first.size() == 1 && A.isSymAtom(t.first[0], sid) && A.symbol(sid).kind == Symbol::SharedBase) continue;
      Term tm; tm.coef = t.second; for (uint32_t a : t.first) tm.atoms.push_back(strip(A.atomExpr(a)));
      std::sort(tm.atoms.begin(), tm.atoms.end()); ts.push_back(std::move(tm));
    }
    for (size_t i = 0; i < ts.size(); i++) {
      if (ts[i].coef >= 0) continue;
      for (size_t k = 0; k < ts[i].atoms.size() && ts[i].coef < 0; k++) {
        EP q = ts[i].atoms[k];
        if (q->k != EK::Op || q->op != Instruction::SDiv) continue;
        EP X = strip(q->args[0]), D = strip(q->args[1]);
        std::vector<EP> rest(ts[i].atoms); rest.erase(rest.begin() + k);
        int64_t c; int64_t dc;
        if (A.constInt(D, dc)) { if (!rest.empty() || dc <= 0 || (-ts[i].coef) % dc) continue; c = -ts[i].coef / dc; }
        else { std::vector<EP> da; if (!monoAtoms(D, da) || da != rest) continue; c = -ts[i].coef; }
        std::vector<EP> xa; if (!monoAtoms(X, xa) || !nonNeg(X)) continue;
        for (size_t j = 0; j < ts.size(); j++) if (j != i && ts[j].coef >= c && ts[j].atoms == xa) { ts[j].coef -= c; ts[i].coef = 0; break; }
      }
    }
    for (auto& t : ts) { if (!t.coef) continue; if (t.coef < 0) return false; for (EP a : t.atoms) if (!nonNeg(a)) return false; }
    return true;
  };
  const int64_t tileEnd = 0;
  std::map<Instruction*, int64_t> remaining;   // remaining access site of `dyn` -> proven lower bound of its offsets
  std::set<Instruction*> symbolicSites;         // remaining sites with an access at a symbolic offset (not modeled)
  int64_t deadEnd = 0;                          // end of the deleted accesses' bytes
  // an operand read resolve() kept on shared memory: its instruction goes with the operand slice, but the fills
  // read the same bytes (relocated with the rest when they lie above the cut: Materializer::relocCut), so those
  // bytes stay occupied - the cut lands at or below them
  bool staying = false; int64_t stayingLo = INT64_MAX;
  for (auto& a : Sig.smemAccesses) {
    if (objOf(a.addr) != dyn) continue;
    if (dead.count(a.I) && X.stayingReads.count(a.I)) { staying = true; if (!a.concrete) return kept("a staying operand read at a symbolic offset"); stayingLo = std::min(stayingLo, a.off); continue; }
    if (dead.count(a.I) || deadCopies.count(a.I) || ringReads.count(a.I)) { if (!a.concrete) return kept("a deleted access at a symbolic offset"); deadEnd = std::max(deadEnd, a.off + (int64_t)a.bytes); continue; }
    int64_t lb;
    if (a.concrete) lb = a.off;
    else if (!symbolicLowerBound(a.addr, lb)) return kept("a remaining shared access at a symbolic offset with no lower bound (" + A.str(a.addr, 5) + ", " + std::string(a.I->getOpcodeName()) + "): the reads the operands were resolved through may alias it");
    if (!a.concrete) symbolicSites.insert(a.I);
    auto it = remaining.find(a.I);
    if (it == remaining.end()) remaining[a.I] = lb; else it->second = std::min(it->second, lb);
  }
  if (staying) note("an operand keeps reading shared memory (bytes a quantizer wrote): its fills read those bytes where the relocation puts them, the other operand's staging is deleted");
  int64_t cutHi = INT64_MAX;
  auto siteDesc = [&](Instruction* I) {
    std::string d = I->getOpcodeName();
    if (auto* CI = dyn_cast<CallInst>(I)) { if (CI->isInlineAsm()) { bool alu; d += " " + asmMnemonics(CI, alu); } else if (CI->getCalledFunction()) d += " " + CI->getCalledFunction()->getName().str(); }
    std::set<unsigned> phases; bool rd = false, wr = false; for (auto& a : Sig.smemAccesses) if (a.I == I) { phases.insert(a.phase); (a.isStore ? wr : rd) = true; }
    d += rd && wr ? ", read/write" : wr ? ", write" : ", read"; d += ", barrier phase(s)"; for (unsigned p : phases) d += " " + std::to_string(p);
    return d;
  };
  // The tiles live in the K loop's barrier phases. A remaining access under them in another phase (the K-split
  // reduction through the ring's space after the loop) is separated from every tile access by a barrier: within one
  // pass of the body by the phase order, and from one pass to the next by the barriers between the last phase and
  // the body's end plus those before the first tile phase (all threads pass a barrier only after finishing the phase
  // before it). Such accesses stay where they are; the freed region above the tiles is relocated.
  // The same holds of the ring's own (deleted) accesses: the resolution of the operands' reads through concrete
  // writes only is sound when no symbolic write can interleave, which the phase separation shows.
  unsigned minTP = ~0u, maxTP = 0;
  for (auto& tc : Sig.tpCalls) if (tc.iter != ~0u) { minTP = std::min(minTP, tc.phase); maxTP = std::max(maxTP, tc.phase); }
  for (auto& a : Sig.smemAccesses) if (dead.count(a.I) || deadCopies.count(a.I) || ringReads.count(a.I)) { minTP = std::min(minTP, a.phase); maxTP = std::max(maxTP, a.phase); }
  // Phases count the barriers executed since the kernel's entry. Within one pass of the job loop's body, accesses in
  // different phases are barrier-separated. Between one pass and the next only the body's own barriers count
  // (those the pass executes after the access, or before the other access in the next pass): the body runs from the
  // phase at the loop's entry to the phase after its pass; an access in either boundary phase may be inside it.
  unsigned entryPh = 0, exitPh = X.finalPhase;
  if (J) { auto it = X.genericPhases.find(J); if (it == X.genericPhases.end()) return kept("the job loop's barrier phases are unknown"); entryPh = it->second.first; exitPh = it->second.second; }
  auto separated = [&](unsigned p) {
    if (p >= minTP && p <= maxTP) return false;
    if (!J || p < entryPh || p > exitPh) return true;
    if (p < minTP) return maxTP < exitPh || p > entryPh;   // the previous pass's staging accesses, then this one
    return p < exitPh || minTP > entryPh;                   // this one, then the next pass's staging accesses
  };
  // the range of a shared offset (without the object's base), for the low sites' extent: an interval analysis over
  // the thread index (0..T-1), lane, masks, shifts, divisions and selections
  std::map<uint32_t, std::pair<int64_t, int64_t>> coordRange;   // the emission's coordinate symbols (lane bits) -> range
  std::function<bool(EP, int64_t&, int64_t&)> range = [&](EP e, int64_t& lo, int64_t& hi) -> bool {
    int64_t c; if (A.constInt(e, c)) { lo = hi = c; return true; }
    switch (e->k) {
      case EK::Poly: {
        lo = hi = 0;
        for (auto& t : e->terms) {
          uint32_t sid; if (t.first.size() == 1 && A.isSymAtom(t.first[0], sid) && A.symbol(sid).kind == Symbol::SharedBase) continue;
          int64_t plo = t.second, phi = t.second;
          for (uint32_t a : t.first) { int64_t alo, ahi; if (!range(A.atomExpr(a), alo, ahi)) return false; const int64_t x[4] = {plo * alo, plo * ahi, phi * alo, phi * ahi}; plo = *std::min_element(x, x + 4); phi = *std::max_element(x, x + 4); }
          lo += plo; hi += phi;
        }
        return true;
      }
      case EK::Sym: {
        const Symbol& s = A.symbol(e->sym);
        if (s.kind == Symbol::Coord) { auto it = coordRange.find(e->sym); if (it == coordRange.end()) return false; lo = it->second.first; hi = it->second.second; return true; }
        if (s.kind != Symbol::Sreg) return false;
        if (s.name.find("tid.x") != std::string::npos && s.name.find("ntid") == std::string::npos) { lo = 0; hi = (int64_t)T - 1; return true; }
        if (s.name.find("tid.y") != std::string::npos || s.name.find("tid.z") != std::string::npos) { if (s.name.find("ntid") != std::string::npos) return false; lo = hi = 0; return true; }
        if (s.name.find("laneid") != std::string::npos) { lo = 0; hi = 31; return true; }
        return false;
      }
      case EK::Op: {
        int64_t k, alo, ahi;
        switch (e->op) {
          case Instruction::ZExt: case Instruction::SExt: return range(e->args[0], lo, hi);
          case Instruction::Trunc: if (!range(e->args[0], lo, hi)) return false; { const unsigned w = e->ty->getIntegerBitWidth(); return w >= 64 || (lo >= -(1ll << (w - 1)) && hi < (1ll << (w - 1))); }
          case Instruction::And: if (A.constInt(e->args[1], k) && k >= 0) { lo = 0; hi = k; if (range(e->args[0], alo, ahi) && alo >= 0) hi = std::min(hi, ahi); return true; } if (A.constInt(e->args[0], k) && k >= 0) { lo = 0; hi = k; return true; } return false;
          case Instruction::LShr: case Instruction::AShr: if (!A.constInt(e->args[1], k) || k < 0 || k >= 63 || !range(e->args[0], alo, ahi) || alo < 0) return false; lo = alo >> k; hi = ahi >> k; return true;
          case Instruction::Shl: if (!A.constInt(e->args[1], k) || k < 0 || k >= 32 || !range(e->args[0], alo, ahi) || alo < 0 || ahi >= (1ll << 31)) return false; lo = alo << k; hi = ahi << k; return true;
          case Instruction::SDiv: case Instruction::UDiv: if (!A.constInt(e->args[1], k) || k <= 0 || !range(e->args[0], alo, ahi) || alo < 0) return false; lo = alo / k; hi = ahi / k; return true;
          case Instruction::SRem: case Instruction::URem: if (!A.constInt(e->args[1], k) || k <= 0 || !range(e->args[0], alo, ahi) || alo < 0) return false; lo = 0; hi = std::min(ahi, k - 1); return true;
          default: return false;
        }
      }
      case EK::Ite: { int64_t l2, h2; if (!range(e->args[1], lo, hi) || !range(e->args[2], l2, h2)) return false; lo = std::min(lo, l2); hi = std::max(hi, h2); return true; }
      case EK::Call: {
        const int mm = minMaxCallee(e->callee);
        if (!mm) return false;
        int64_t l2, h2; if (!range(e->args[0], lo, hi) || !range(e->args[1], l2, h2)) return false;
        if (mm < 0) { lo = std::min(lo, l2); hi = std::min(hi, h2); } else { lo = std::max(lo, l2); hi = std::max(hi, h2); }
        return true;
      }
      default: return false;
    }
  };
  // remaining accesses under the deleted bytes (or the tiles): phase-separated, they stay; the relocation lands
  // above their extent
  const int64_t lowEnd = std::max(tileEnd, deadEnd);
  std::set<Instruction*> lowSites;
  int64_t lowTop = tileEnd;
  if (staying) { if (stayingLo < lowEnd) return kept("a staying operand's shared bytes lie under the deleted staging bytes"); cutHi = std::min(cutHi, stayingLo); }
  for (auto& kv : remaining) {
    if (kv.second >= lowEnd) { cutHi = std::min(cutHi, kv.second); continue; }
    // a site whose every access has a concrete offset was modeled: the operands' resolution saw its writes (a
    // race would have been refused), and the deleted bytes are not written by the recovered body. It needs no
    // barrier separation unless it lies under the tiles, which are.
    const bool modeled = !symbolicSites.count(kv.first) && kv.second >= tileEnd;
    for (auto& a : Sig.smemAccesses) {
      if (a.I != kv.first) continue;
      if (!modeled && !separated(a.phase)) return kept("a remaining shared access at offset " + std::to_string(kv.second) + " lies under the deleted staging bytes [0, " + std::to_string(lowEnd) + ") in a barrier phase the staging or the tiles are used in (" + siteDesc(kv.first) + "; phases " + std::to_string(minTP) + ".." + std::to_string(maxTP) + ")");
      int64_t lo, hi;
      if (!range(A.asPoly(a.addr), lo, hi)) return kept("a remaining shared access under the deleted staging bytes has no upper bound (" + A.str(a.addr, 5) + "; " + siteDesc(kv.first) + ")");
      lowTop = std::max(lowTop, hi + (int64_t)a.bytes);
    }
    lowSites.insert(kv.first);
  }
  for (Instruction* I : lowSites) remaining.erase(I);
  if (cutHi != INT64_MAX && deadEnd > cutHi) return kept("the deleted staging bytes extend past a remaining region (a remaining access may alias the ring)");
  const int64_t lowFloor = (lowTop + 127) / 128 * 128;   // where the relocated region lands
  int64_t shrink = 0;
  if (cutHi != INT64_MAX) { shrink = (cutHi - lowFloor) / 128 * 128; if (shrink < 0) shrink = 0; }
  // every remaining site must be relocatable: a plain memory instruction or a cp.async with a register destination
  auto addrOperand = [&](Instruction* I, unsigned& idx) -> bool {
    if (isa<LoadInst>(I)) { idx = LoadInst::getPointerOperandIndex(); return true; }
    if (isa<StoreInst>(I)) { idx = StoreInst::getPointerOperandIndex(); return true; }
    if (isa<AtomicRMWInst>(I)) { idx = AtomicRMWInst::getPointerOperandIndex(); return true; }
    if (isa<AtomicCmpXchgInst>(I)) { idx = AtomicCmpXchgInst::getPointerOperandIndex(); return true; }
    auto* CI = dyn_cast<CallInst>(I); if (!CI) return false;
    if (CI->getCalledFunction() && CI->getCalledFunction()->getName().starts_with("__mvcc_cp_async_zsel_")) { idx = 0; return true; }
    if (!CI->isInlineAsm()) return false;
    std::vector<PtxInstr> ins; std::string err;
    if (!parsePtxAsm(std::string(cast<InlineAsm>(CI->getCalledOperand())->getAsmString()), ins, err) || ins.size() != 1 || ins[0].mnemonic != "cp" || ins[0].ops.empty() || ins[0].ops[0].reg < 0) return false;
    idx = (unsigned)ins[0].ops[0].reg; return true;
  };
  if (shrink) for (auto& kv : remaining) { unsigned idx; if (!addrOperand(kv.first, idx)) return kept(std::string("a remaining shared access cannot be relocated (") + kv.first->getOpcodeName() + ")"); }

  // ---- 5. the words of the decode mapping, as expressions over the lane's coordinates
  Symbol symRow, symKq, symNg, symIter; std::vector<Symbol> symWb(warpBits);
  auto coordSym = [&](Symbol& s, const std::string& nm) { s.kind = Symbol::Coord; s.name = nm; s.ty = i64; return A.mkSymRaw(s)->sym; };
  const uint32_t idRow = coordSym(symRow, "row"), idKq = coordSym(symKq, "kq"), idNg = coordSym(symNg, "ng"), idIter = coordSym(symIter, "iter");
  std::vector<uint32_t> idWb; for (unsigned b = 0; b < warpBits; b++) idWb.push_back(coordSym(symWb[b], "wb" + std::to_string(b)));
  // a concatenation of selections on one condition is a selection of concatenations (the zero-fill guard of a word)
  // comes out of its bytes); consecutive bytes of one source merge
  std::map<EP, EP> hmemo;
  std::function<EP(EP)> hoist = [&](EP e) -> EP {
    return A.rewrite(e, [&](EP x) -> EP {
      if (x->k != EK::Concat) return nullptr;
      std::vector<EP> parts; for (EP p : x->args) parts.push_back(hoist(p));
      EP c = nullptr; bool all = true;
      for (EP p : parts) { if (p->k != EK::Ite || (c && p->args[0] != c)) { all = false; break; } c = p->args[0]; }
      if (all && c) { std::vector<EP> a1, a2; for (EP p : parts) { a1.push_back(p->args[1]); a2.push_back(p->args[2]); } return A.ite(c, hoist(A.concat(a1, x->ty)), hoist(A.concat(a2, x->ty))); }
      std::vector<EP> bs; for (EP p : parts) { auto b = A.toBytes(p); bs.insert(bs.end(), b.begin(), b.end()); }
      EP r = A.fromBytes(bs, x->ty);
      // loads of consecutive bytes under one guard (the bytes of a word at a lane-dependent offset) are one load
      if (r->k == EK::Concat) {
        EP first = A.unwrap(r->args[0]); unsigned off = 0; bool run = first->k == EK::Load;
        for (EP p : r->args) {
          EP l = A.unwrap(p); int64_t d;
          if (l->k != EK::Load || l->args[1] != first->args[1] || !A.constInt(A.sub(A.asPoly(l->args[0]), A.asPoly(first->args[0])), d) || d != (int64_t)off) { run = false; break; }
          off += A.sizeOf(l->ty);
        }
        if (run && off == A.sizeOf(x->ty)) return A.load(x->ty, first->args[0], first->args[1], first->len);
      }
      return r;
    }, hmemo);
  };
  std::function<bool(EP)> isZero = [&](EP e) -> bool {
    int64_t c; if (A.constInt(e, c)) return c == 0;
    if (e->k == EK::Const) return e->c.isZero();
    if (e->k == EK::Concat) { for (EP p : e->args) if (!isZero(p)) return false; return true; }
    if (e->k == EK::Bytes) return isZero(e->args[0]);
    return false;
  };
  // Tensor mode: per operand block its two slot-word pairs at the slots' own coordinates (the cooperative-tensor)
  // layout law, msl/mvcc_prelude.metal: slot quad j of lane l holds, for an A block, row base + 8 j and columns
  // 8 hi + 4 (l & 1) + 0..3; for a B block, column base + 8 j and rows 8 hi + 4 (l & 1) + 0..3, with
  // base = ((l >> 1) & 3) | ((l >> 4) << 2) and hi = (l >> 3) & 1). Quad j's words i = 0, 1 are the element pairs
  // at k + 2 i; the pair is one 8-byte expression (consecutive loads of one source merge). No PTX-layout words, no
  // shuffles: each lane loads what its slots hold.
  Symbol symBase, symL0, symL3;
  const uint32_t idBase = coordSym(symBase, "sbase"), idL0 = coordSym(symL0, "l0"), idL3 = coordSym(symL3, "l3");
  coordRange[idBase] = {0, 7}; coordRange[idL0] = {0, 1}; coordRange[idL3] = {0, 1}; coordRange[idKq] = {0, 3}; coordRange[idNg] = {0, 7};
  for (uint32_t id : idWb) coordRange[id] = {0, 1};
  std::map<CallInst*, std::array<EP, 2>> slotPairs;
  {
    IRBuilder<> dummy(ctx);
    Materializer chk(A, F, dummy);
    chk.bound = {idRow, idKq, idNg, idIter, idBase, idL0, idL3}; for (uint32_t id : idWb) chk.bound.insert(id);
    chk.speculate = true; chk.relocCut = cutHi; chk.relocShrink = shrink; chk.range = range;
    for (const BlockPos& bp : blockPos) {
      std::array<EP, 2> pairs;
      for (unsigned j = 0; j < 2; j++) {
        std::vector<EP> ws;
        for (unsigned i = 0; i < 2; i++) {
          // k = 32 p + 16 kb + 8 hi + 4 l0 + 2 i: k/32 = p, (k/8)%2 = hi, (k/16)%2 = kb, (k/2)%4 = 2 l0 + i
          EP first = A.add(A.mkSym(symBase), A.mkInt(i64, 8 * j + (bp.isB ? 16 * bp.nb : 0)));   // base < 8: the n8 block is j, the 16-block nb
          std::vector<EP> co{first, A.mkInt(i64, j), A.mkInt(i64, bp.isB ? bp.nb : 0), A.mkInt(i64, bp.p), A.mkSym(symL3), A.mkInt(i64, bp.kb), A.add(A.mul(A.mkInt(i64, 2), A.mkSym(symL0)), A.mkInt(i64, i)), A.mkSym(symIter)};
          for (unsigned b = 0; b < warpBits; b++) co.push_back(A.mkSym(symWb[b]));
          co.push_back(A.mkInt(i64, bp.q));
          bool ok = true;
          const int64_t blk = j + (bp.isB ? 2 * bp.nb : 0);   // the n8 block of these slots
          const bool pad = bp.isB ? padB[bp.q].count(blk) != 0 : padA[bp.q].count(blk) != 0;
          EP e = pad ? A.mkInt(i32, 0) : bp.isB ? shB[bp.q].instSym(fitB[bp.q].tmpl, fitB[bp.q].holes, co, ok) : shA[bp.q].instSym(fitA[bp.q].tmpl, fitA[bp.q].holes, co, ok);
          if (!ok) return kept(std::string(bp.isB ? "a B" : "an A") + " slot word cannot be re-derived at the slot's coordinates");
          if (A.sizeOf(e->ty) != 4) return kept("an operand word is not 4 bytes wide");
          ws.push_back(e);
        }
        EP pair = hoist(A.concat(ws, i64));
        // The pair is emitted as written, so every selection inside it is now decided by the ones around it (the
        // fit needed one shape per operand; the emission does not): the staged row's null test, kept as the
        // condition of a selected address, folds under the zero-fill guard it duplicates, and the load's address
        // is the argument plus an offset - one clamp select per load instead of a null test and two selects.
        pair = hoist(A.simplifyUnder(nullFold(pair), [](EP) { return true; }, &allCache));
        if (!chk.can(pair)) return kept(std::string(bp.isB ? "a B" : "an A") + " slot word pair cannot be emitted: " + chk.why + " in " + A.str(pair, 6));
        pairs[j] = pair;
      }
      slotPairs[bp.src] = pairs;
    }
    // A chain's guard (its sub-tile has no valid row: the source skips its matmuls) zero-fills both operands. When
    // every A word of the chain is zero under the guard, the products are zero whatever B holds (finite: the
    // dequantized weights, or the dequantized zero page), and the accumulator is unchanged either way (c + (-0) =
    // c, +0 + -0 = +0): B's zero-fill is dropped. The B words are then one expression across the chains that
    // read them (the chains' guards differed), so a copy fills them once.
    {
      auto zeroGuard = [&](EP e, EP& g, EP& inner) -> bool {   // e = ite(g, 0, inner)
        if (e->k != EK::Ite) return false;
        if (isZero(e->args[1])) { g = e->args[0]; inner = e->args[2]; return true; }
        if (isZero(e->args[2])) { g = A.notb(e->args[0]); inner = e->args[1]; return true; }
        return false;
      };
      std::map<unsigned, EP> chainGuard; std::set<unsigned> mixed;
      for (const BlockPos& bp : blockPos) if (!bp.isB) for (EP pr : slotPairs[bp.src]) {
        EP g = nullptr, inner = nullptr; if (!zeroGuard(pr, g, inner)) g = nullptr;
        auto it = chainGuard.find(bp.q);
        if (it == chainGuard.end()) chainGuard[bp.q] = g; else if (it->second != g) mixed.insert(bp.q);
      }
      for (const BlockPos& bp : blockPos) if (bp.isB && !mixed.count(bp.q) && chainGuard.count(bp.q) && chainGuard[bp.q]) {
        for (EP& pr : slotPairs[bp.src]) { EP g = nullptr, inner = nullptr; if (zeroGuard(pr, g, inner) && g == chainGuard[bp.q]) pr = inner; }
      }
    }
    if (!ringLoads.empty()) {
      IRBuilder<> dummyR(ctx);
      Materializer chkR(A, F, dummyR);
      chkR.bound = {idQRow, idQCol, idQIter}; chkR.speculate = true; chkR.relocCut = cutHi; chkR.relocShrink = shrink; chkR.range = range;
      for (RingLoad& rl : ringLoads) {
        EP h = hoist(A.simplifyUnder(nullFold(rl.expr), [](EP) { return true; }, &allCache));
        if (chkR.can(h)) rl.expr = h;
      }
    }
  }

  // ---- 6. STIR holds operand φ. check(P,S); emit from Program + Schedule. No body IR until legality.ok.
  stir::Program SP;
  sched::Schedule SS;
  {
    std::vector<sig::LoopClass> noLoops;
    std::vector<stir::LoopStmt> noStmts;
    SP = stir::build(A, Disc, noLoops, noStmts, llvm::demangle(F.getName().str()));
    if (kRecSum) {
      stir::LoopStmt Ls;
      Ls.header = "%k"; Ls.kind = sig::LoopKind::Traversal; Ls.anyData = true;
      Ls.algebra.push_back({sig::Algebra::TensorSum, kRecSumPhis});
      Ls.permits = kRecPerm;
      if (SP.loops.empty()) SP.loops.push_back(Ls);
    }
    // Named leftovers (data-dependent / non-affine) are launch assumes, not L10 on the source body.
    // Invariant: the source schedule is always a legal candidate.
    SP.unfitted.clear();
    for (auto& kv : slotPairs) {
      stir::Operand op;
      op.site = kv.first;
      op.words.assign(kv.second.begin(), kv.second.end());
      SP.operands.push_back(op);
    }
    engine::Chosen Ch = engine::choose(SP);
    SS = Ch.S;
    if (!SS.stages.empty()) {
      bool anyTg = false;
      for (auto& a : SS.stages[0].storage)
        if (a.kind == sched::Storage::Threadgroup) anyTg = true;
      if (anyTg) SS = sched::editDeviceStream(SS);
      if (kRecSum) SS.stages[0].engine = sched::Engine::TensorOp;
    }
    legal::Result Lg = legal::check(SP, SS);
    if (!Lg.ok) return kept(Lg.first());
    note("check+emit: " + SS.why + " " + Lg.first() + " " + sched::stageShape(SS));
    note("STIR-first: tensors=" + std::to_string(SP.tensors.size()) + " coords=" + std::to_string(SP.coordNames.size())
         + " accesses=" + std::to_string(SP.accesses.size()) + " operands=" + std::to_string(SP.operands.size())
         + " round-trip=" + std::to_string(SP.verified) + "/" + std::to_string(SP.verified + SP.mismatches)
         + " samples");
  }
  PHINode* iterPhi;
  BasicBlock* KH = K->getHeader();
  {
    BasicBlock* H = KH;
    IRBuilder<> Bh(&*H->getFirstNonPHIIt());
    iterPhi = Bh.CreatePHI(i32, 2, "mvcc.retile.iter");
    Value* inc = Bh.CreateAdd(iterPhi, ConstantInt::get(i32, 1));
    for (BasicBlock* P : predecessors(H)) iterPhi->addIncoming(K->contains(P) ? inc : (Value*)ConstantInt::get(i32, 0), P);
  }
  FunctionCallee tidX = Mod.getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.x", FunctionType::get(i32, false));
  // tensor mode: the slot words of a block are re-derived just before its fragment conversion, which they replace
  // (its fields' uses - the matmul's operand words - take the slot words; the conversion and the PTX-layout words
  // behind it are deleted below); the matmuls stay
  // One materializer serves every site (program order): the lane's coordinates are emitted once in the entry block
  // and a sub-expression is emitted once per dominating site. The chains of one iteration overlap heavily - the four
  // m-subtile chains of a column half read the same B words, the two column halves of an m-subtile the same A
  // words, and every word's expression carries the job's tile origin and the row bound - so the fills of a 192-
  // block iteration (m4) collapse to the distinct words (the per-copy split below then keeps what each copy uses).
  unsigned fillPairs = 0, distinctPairs = 0;
  DominatorTree DTfill(F);
  IRBuilder<> Bfill(ctx);
  Materializer Mfill(A, F, Bfill);
  Mfill.speculate = true; Mfill.DT = &DTfill; Mfill.relocCut = cutHi; Mfill.relocShrink = shrink; Mfill.range = range;
  {
    IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
    Value* tid = B.CreateCall(tidX);
    Value* lane = B.CreateAnd(tid, ConstantInt::get(i32, 31)), *warp = B.CreateLShr(tid, ConstantInt::get(i32, 5));
    Mfill.coord[idBase] = B.CreateOr(B.CreateAnd(B.CreateLShr(lane, ConstantInt::get(i32, 1)), ConstantInt::get(i32, 3)), B.CreateShl(B.CreateLShr(lane, ConstantInt::get(i32, 4)), ConstantInt::get(i32, 2)));
    Mfill.coord[idL0] = B.CreateAnd(lane, ConstantInt::get(i32, 1)); Mfill.coord[idL3] = B.CreateAnd(B.CreateLShr(lane, ConstantInt::get(i32, 3)), ConstantInt::get(i32, 1));
    Mfill.coord[idIter] = iterPhi;
    if (idQIter) Mfill.coord[idQIter] = iterPhi;
    if (idQRow && !ringTeamPhi) {
      Value* team = B.CreateLShr(tid, ConstantInt::get(i32, 3));
      Mfill.coord[idQRow] = ringKGps == 1 ? team : B.CreateUDiv(team, ConstantInt::get(i32, ringKGps));
      Value* g = ringKGps == 1 ? (Value*)ConstantInt::get(i32, 0) : B.CreateURem(team, ConstantInt::get(i32, ringKGps));
      Mfill.coord[idQCol] = B.CreateAdd(B.CreateMul(g, ConstantInt::get(i32, 128)), B.CreateMul(B.CreateAnd(tid, ConstantInt::get(i32, 7)), ConstantInt::get(i32, 16)));
    }
    for (unsigned b = 0; b < warpBits; b++) Mfill.coord[idWb[b]] = B.CreateAnd(B.CreateLShr(warp, ConstantInt::get(i32, b)), ConstantInt::get(i32, 1));
    {
      // the runtime's zero page (guarded_loads.cpp): zero-filled words load through it when not needed
      FunctionCallee ZF = F.getParent()->getOrInsertFunction("__mvcc_zero_page", PointerType::get(ctx, 0));
      if (auto* Fn = dyn_cast<Function>(ZF.getCallee())) { Fn->setMemoryEffects(MemoryEffects::none()); Fn->addFnAttr(Attribute::NoUnwind); Fn->addFnAttr(Attribute::WillReturn); }
      Mfill.zeroPage = B.CreateCall(ZF, {}, "zero_page");
    }
  }
  const unsigned nRing = (unsigned)ringLoads.size();
  if (nRing) {
    std::sort(ringLoads.begin(), ringLoads.end(), [&](const RingLoad& a, const RingLoad& b) { return instOrder[a.L] < instOrder[b.L]; });
    for (auto& rl : ringLoads) Mfill.collectNeeds(rl.expr);
    for (auto& rl : ringLoads) {
      Bfill.SetInsertPoint(rl.L);
      Mfill.curBB = rl.L->getParent();
      if (ringTeamPhi) {
        Value* tid = Bfill.CreateCall(tidX);
        Value* team = ringTeamPhi;
        Mfill.coord[idQRow] = ringKGps == 1 ? team : Bfill.CreateUDiv(team, ConstantInt::get(i32, ringKGps));
        Value* g = ringKGps == 1 ? (Value*)ConstantInt::get(i32, 0) : Bfill.CreateURem(team, ConstantInt::get(i32, ringKGps));
        Mfill.coord[idQCol] = Bfill.CreateAdd(Bfill.CreateMul(g, ConstantInt::get(i32, 128)), Bfill.CreateMul(Bfill.CreateAnd(tid, ConstantInt::get(i32, 7)), ConstantInt::get(i32, 16)));
      }
      Value* v = Mfill.val(rl.expr);
      if (!v) report_fatal_error(Twine("decode retiling: internal: the emission failed after the checks (a quantizer ring read" + (Mfill.why.empty() ? std::string() : ": " + Mfill.why) + ")"));
      rl.L->replaceAllUsesWith(Mfill.toInt(v, rl.L->getType()));
      rl.L->eraseFromParent();
    }
    note("quantizer ring reads rewritten to device: " + std::to_string(nRing) + " load(s)");
  }
  std::map<EP, Value*> pairValue;   // the value each distinct pair expression became (for the copy assignment below)
  auto fillSlots = [&](CallInst* src, BasicBlock* origBB, const std::array<EP, 2>& pairs) {
    BasicBlock* BB = src->getParent();
    BasicBlock* after = BB->splitBasicBlock(src->getIterator(), "mvcc.retile.after");
    BB->getTerminator()->eraseFromParent();
    Bfill.SetInsertPoint(BB);
    Mfill.curBB = origBB;
    Value* w[4];
    for (unsigned j = 0; j < 2; j++) {
      const bool had = Mfill.memo.count(pairs[j]) && Mfill.memoBB.count(pairs[j]) && DTfill.dominates(Mfill.memoBB[pairs[j]], origBB);
      Value* v = Mfill.val(pairs[j]);
      if (!v) report_fatal_error(Twine("decode retiling: internal: the emission failed after the checks (a slot word pair" + (Mfill.why.empty() ? std::string() : ": " + Mfill.why) + ")"));
      fillPairs++; if (!had) distinctPairs++;
      pairValue[pairs[j]] = v;
      v = Mfill.toInt(v, i64);
      w[2 * j] = Bfill.CreateTrunc(v, i32); w[2 * j + 1] = Bfill.CreateTrunc(Bfill.CreateLShr(v, ConstantInt::get(i64, 32)), i32);
    }
    Bfill.CreateBr(after);
    std::vector<ExtractValueInst*> evs; for (User* u : src->users()) evs.push_back(cast<ExtractValueInst>(u));
    for (ExtractValueInst* ev : evs) { ev->replaceAllUsesWith(w[ev->getIndices()[0]]); ev->eraseFromParent(); }
    dead.insert(src);
  };
  {
    std::vector<std::pair<CallInst*, BasicBlock*>> sites;   // program order, with the block before any split
    std::map<CallInst*, std::array<EP, 2>> fromStir;
    for (const stir::Operand& op : SP.operands) {
      auto* src = const_cast<CallInst*>(dyn_cast_or_null<CallInst>(op.site));
      if (!src || op.words.size() < 2) continue;
      fromStir[src] = {op.words[0], op.words[1]};
      sites.push_back({src, src->getParent()});
    }
    std::sort(sites.begin(), sites.end(), [&](const std::pair<CallInst*, BasicBlock*>& a, const std::pair<CallInst*, BasicBlock*>& b) { return instOrder[a.first] < instOrder[b.first]; });
    for (auto& s : sites) Mfill.collectNeeds(fromStir[s.first][0]), Mfill.collectNeeds(fromStir[s.first][1]);
    for (auto& s : sites) fillSlots(s.first, s.second, fromStir[s.first]);
  }
  // delete the slice (users before operands) and the copies
  { std::vector<Instruction*> left(dead.begin(), dead.end()); bool progress = true;
    while (!left.empty() && progress) { progress = false; for (auto it = left.begin(); it != left.end();) { if ((*it)->use_empty()) { (*it)->eraseFromParent(); it = left.erase(it); progress = true; } else ++it; } }
    // what remains is used only within the slice through cycles of phis: cut them
    for (Instruction* I : left) I->replaceAllUsesWith(PoisonValue::get(I->getType()));
    for (Instruction* I : left) I->eraseFromParent(); }
  for (Instruction* Cp : deadCopies) Cp->eraseFromParent();
  // ---- 7. the K loop's barriers. With the ring gone the body writes no memory: its barriers order only the staged
  // scales (written before the loop) against the loop's reads, and the loop's reads against the writes after it.
  // One barrier before the loop and one at each exit do the same; the per-iteration barrier (a device-memory fence
  // in MSL) otherwise serializes the iterations' loads. (cp.async group control in the loop goes with it: the
  // copies are synchronous in this backend, the groups always complete.)
  unsigned hoistedBarriers = 0;
  CallInst* barrierProto = nullptr;   // the barrier hoisted to the preheader (cloned by the exchange below)
  bool loopPure = false;   // the K loop writes no memory and calls nothing with effects
  {
    DominatorTree DT2(F); LoopInfo LI2(DT2);
    Loop* K2 = LI2.getLoopFor(KH);
    std::vector<CallInst*> bars, ctl; std::string why;
    auto isBarrierName = [](StringRef n) { return n == "llvm.nvvm.barrier0" || n.starts_with("llvm.nvvm.barrier") || n.starts_with("llvm.nvvm.bar.sync"); };
    bool ok = K2 && K2->getHeader() == KH;
    if (!ok) why = "the K loop was not found again";
    if (ok) for (BasicBlock* BB : K2->blocks()) {
      for (Instruction& I : *BB) {
        if (isa<StoreInst>(I) || isa<AtomicRMWInst>(I) || isa<AtomicCmpXchgInst>(I) || isa<FenceInst>(I)) { ok = false; why = std::string("a ") + I.getOpcodeName() + " remains in the K loop"; break; }
        auto* CI = dyn_cast<CallInst>(&I); if (!CI) continue;
        if (CI->isInlineAsm()) {
          std::vector<PtxInstr> ins; std::string err;
          if (!parsePtxAsm(std::string(cast<InlineAsm>(CI->getCalledOperand())->getAsmString()), ins, err) || ins.empty()) { ok = false; why = "an asm the loop check cannot parse"; break; }
          bool alu; asmMnemonics(CI, alu);
          if (alu) continue;
          const PtxInstr& in = ins[0];
          if (ins.size() == 1 && (in.mnemonic == "bar" || in.mnemonic == "barrier") && in.hasMod("sync") && !in.predicated() && in.ops.size() == 1 && in.ops[0].kind == PtxOperand::Imm && in.ops[0].imm == 0) { bars.push_back(CI); continue; }
          if (ins.size() == 1 && in.mnemonic == "cp" && (in.hasMod("commit_group") || in.hasMod("wait_group") || in.hasMod("wait_all"))) { ctl.push_back(CI); continue; }
          ok = false; why = "an asm with effects remains in the K loop (" + in.mnemonic + ")"; break;
        }
        Function* cf = CI->getCalledFunction();
        if (!cf) { ok = false; why = "an indirect call in the K loop"; break; }
        StringRef n = cf->getName();
        if (isBarrierName(n)) { bars.push_back(CI); continue; }
        if (n.starts_with("llvm.nvvm.cp.async.commit") || n.starts_with("llvm.nvvm.cp.async.wait")) { ctl.push_back(CI); continue; }
        if (n.starts_with("__mvcc_tp_") || n.starts_with("llvm.nvvm.shfl.") || n.starts_with("llvm.nvvm.read.ptx.sreg") || n.starts_with("llvm.nvvm.bar.warp.sync") || n.starts_with("llvm.lifetime.") || n.starts_with("llvm.experimental.noalias.scope.decl") || n.starts_with("llvm.assume") || n.starts_with("llvm.dbg.")) continue;
        if (n.starts_with("__nv_")) { bool ptr = false; for (Value* a : CI->args()) ptr = ptr || a->getType()->isPointerTy(); if (!ptr) continue; }   // libdevice math: pure unless it writes through a pointer
        if (cf->isIntrinsic() && !CI->mayHaveSideEffects()) continue;
        ok = false; why = "a call with effects remains in the K loop (" + n.str() + ")"; break;
      }
      if (!ok) break;
    }
    // a preheader and exit blocks of the loop's own (the barrier at an exit must run only when the loop ran)
    BasicBlock* pre = ok ? K2->getLoopPreheader() : nullptr;
    if (ok && !pre) pre = InsertPreheaderForLoop(K2, &DT2, &LI2, nullptr, false);
    if (ok && !pre) { ok = false; why = "the K loop has no preheader"; }
    SmallVector<BasicBlock*, 4> exits;
    if (ok) { formDedicatedExitBlocks(K2, &DT2, &LI2, nullptr, false); K2->getUniqueExitBlocks(exits); for (BasicBlock* E : exits) for (BasicBlock* P : predecessors(E)) if (!K2->contains(P)) { ok = false; why = "an exit block of the K loop is reached from outside it"; } }
    if (ok && bars.empty()) { ok = false; why = "the K loop has no barrier"; }
    if (ok) {
      CallInst* proto = bars[0];
      barrierProto = cast<CallInst>(proto->clone()); barrierProto->insertBefore(pre->getTerminator()->getIterator());
      for (BasicBlock* E : exits) proto->clone()->insertBefore(E->getFirstNonPHIIt());
      hoistedBarriers = (unsigned)bars.size();
      for (CallInst* b : bars) b->eraseFromParent();
      for (CallInst* c : ctl) c->eraseFromParent();
    } else note("decode retiling: the K loop's barriers stay (" + why + ")");
    loopPure = ok;
  }
  // relocate the remaining region down over the freed ring (before the K loop's body is cloned below)
  if (shrink) for (auto& kv : remaining) {
    unsigned idx; addrOperand(kv.first, idx);
    Value* a = kv.first->getOperand(idx);
    IRBuilder<> B(kv.first);
    Value* na = a->getType()->isPointerTy() ? B.CreateGEP(i8, a, ConstantInt::get(i64, -shrink)) : B.CreateSub(a, ConstantInt::get(a->getType(), shrink));
    kv.first->setOperand(idx, na);
  }
  if (shrink) res.dynSmemRelocate[F.getName().str()] = {(unsigned)cutHi, (unsigned)shrink};
  // ---- 8. the K loop's work across simdgroup copies. The CUDA block's warp count was chosen for the ring and the
  // tensor cores; on Apple a block of T threads whose K loop is one long dependent chain per warp (m1: 16
  // accumulators per lane, 0.068 ms = the weight stream) or several chains per warp (m2/m4: 4-8 chains, 64-128
  // accumulators per lane: register-bound, 0.5-2 ms) wants more simdgroups with less state each. Once the body is
  // pure (step 7), the loop's live state is provably additive - every header phi is either loop control (its next
  // value and the exit condition rest on phis and invariants alone) or an accumulator: a float from 0.0 whose next
  // value is an fma/fadd chain ending at the phi, the other operands independent of it. The loop then splits into
  // F copies of the block (thread p is virtual thread p mod T of copy h = p / T; the ABI's thread_scale) that each
  // run the same loop control and own a share of the work:
  //   chain mode  (the accumulators form G >= 2 groups by the work they share - the parity / m-subtile chains):
  //               copy h owns the groups g = h mod F; its body is a clone of the loop body with the other groups'
  //               updates cut (their phis carry their zero; the work dies): F chains of registers become one.
  //   iteration mode (one group): copy h runs the body on iterations i = h mod F and skips the rest.
  // At the exit every copy holds a partial sum (zero for what it did not own); a threadgroup exchange (a static
  // shared buffer, chunked) gives every copy the same total in the same order, so from there every copy is in the
  // state the original thread was in: the rest of the kernel runs on all copies, its side effects (stores, atomics,
  // copies) predicated on h == 0, its barriers on every thread. Checks: no K-loop value other than the accumulators'
  // next values is used after the loop; a predicated shared store reaches no shared load before a barrier (the
  // other copies read what copy 0 wrote); the kernel reads nothing it writes.
  unsigned splitF = 0, splitGroups = 0, xchgRounds = 0; bool splitByChain = false;
  struct Region { BasicBlock* entry; BasicBlock* exit; Instruction* anchor; };   // a body copy: its loads hoist to the anchor
  std::vector<Region> regions;
  if (loopPure) {
    std::string why;
    DominatorTree DT4(F); LoopInfo LI4(DT4);
    Loop* K4 = LI4.getLoopFor(KH);
    BasicBlock* pre = K4 ? K4->getLoopPreheader() : nullptr;
    BasicBlock* Lt = K4 ? K4->getLoopLatch() : nullptr;
    BasicBlock* E = K4 ? K4->getUniqueExitBlock() : nullptr;
    BranchInst* LB = Lt ? dyn_cast<BranchInst>(Lt->getTerminator()) : nullptr;
    const unsigned cap = 512;   // the widest block a split may grow to
    if (!kRecSum || !kRecPerm.treeCombine) why = "the K-loop recurrence is not a classifier sum with tree-combine";
    else if (!K4 || K4->getHeader() != KH) why = "the K loop was not found again";
    else if (!pre || !Lt || !E || K4->getExitingBlock() != Lt || !LB || !LB->isConditional()) why = "the K loop does not exit from its latch alone to one block";
    else if (E->getSinglePredecessor() != Lt) why = "the K loop's exit block has other predecessors";
    else if (2 * T > cap) why = "the block already has " + std::to_string(T) + " threads";
    else if (!barrierProto) why = "no barrier to clone";
    auto inLoop = [&](Value* v) { auto* I = dyn_cast<Instruction>(v); return I && K4->contains(I->getParent()); };
    // whether v (a value of this iteration) rests on the header phi P through the loop's instructions
    auto dependsOn = [&](Value* v, PHINode* P) -> bool {
      std::set<Value*> seen; std::vector<Value*> st{v};
      while (!st.empty()) {
        Value* x = st.back(); st.pop_back();
        if (x == P) return true;
        auto* I = dyn_cast<Instruction>(x); if (!I || !K4->contains(I->getParent()) || !seen.insert(x).second) continue;
        if (isa<PHINode>(I) && I->getParent() == KH) continue;
        for (Value* o : I->operands()) st.push_back(o);
      }
      return false;
    };
    // (a) split units only. Additivity is already the classifier (kRecSum / tree-combine). This walk finds
    // the fma/fadd sites the copies cut; it is not the proof that the K-loop is a sum.
    struct SplitUnit { Instruction* op; unsigned accIdx; };
    using Term = SplitUnit;
    std::vector<PHINode*> accs, ctlPhis;
    std::vector<std::vector<Term>> chains;   // per accumulator, outermost (the next value) first
    if (why.empty()) for (PHINode& P : KH->phis()) {
      Value* X = P.getIncomingValueForBlock(Lt); bool add = P.getType()->isFloatTy();
      auto* init = dyn_cast<ConstantFP>(P.getIncomingValueForBlock(pre)); if (!init || !init->isZero()) add = false;
      std::vector<Term> chain;
      while (add && X != &P) {
        auto* CI = dyn_cast<CallInst>(X); auto* BO = dyn_cast<BinaryOperator>(X);
        const bool isFma = CI && CI->getCalledFunction() && (CI->getCalledFunction()->getIntrinsicID() == Intrinsic::fma || CI->getCalledFunction()->getName() == "__nv_fmaf") && CI->arg_size() == 3;
        if (isFma && inLoop(CI)) { if (dependsOn(CI->getArgOperand(0), &P) || dependsOn(CI->getArgOperand(1), &P)) add = false; else { chain.push_back({CI, 2}); X = CI->getArgOperand(2); } }
        else if (BO && BO->getOpcode() == Instruction::FAdd && inLoop(BO)) { const bool d0 = dependsOn(BO->getOperand(0), &P), d1 = dependsOn(BO->getOperand(1), &P); if (d0 == d1) add = false; else { chain.push_back({BO, d0 ? 0u : 1u}); X = BO->getOperand(d0 ? 0 : 1); } }
        else add = false;
      }
      if (add && chain.empty()) add = false;   // never updated: control-like
      if (add) { accs.push_back(&P); chains.push_back(chain); } else ctlPhis.push_back(&P);
    }
    if (why.empty() && accs.empty()) why = "the K loop carries no additive accumulator";
    // (b) loop control: the exit condition and the control phis' next values, over phis and invariants alone
    std::set<Instruction*> ctl;
    if (why.empty()) {
      std::vector<Value*> st{LB->getCondition()}; for (PHINode* P : ctlPhis) st.push_back(P->getIncomingValueForBlock(Lt));
      while (!st.empty() && why.empty()) {
        Value* x = st.back(); st.pop_back(); auto* I = dyn_cast<Instruction>(x); if (!I || !K4->contains(I->getParent())) continue;
        if (isa<PHINode>(I)) { if (I->getParent() != KH) why = "the loop control merges values inside the body"; else if (std::find(accs.begin(), accs.end(), I) != accs.end()) why = "the loop control depends on an accumulator"; continue; }
        if (!ctl.insert(I).second) continue;
        if (isa<LoadInst>(I) || I->isTerminator() || I->mayHaveSideEffects() || I->mayReadFromMemory()) { why = std::string("the loop control depends on a ") + I->getOpcodeName(); break; }
        if (auto* CI = dyn_cast<CallInst>(I)) { Function* cf = CI->getCalledFunction(); if (!cf || !(cf->getName().starts_with("llvm.nvvm.read.ptx.sreg") || cf->getName().starts_with("__nv_") || (cf->isIntrinsic() && !CI->mayHaveSideEffects()))) { why = "the loop control depends on a call"; break; } }
        for (Value* o : I->operands()) st.push_back(o);
      }
    }
    // (c) the work, grouped by the terms it feeds: terms sharing work (the fragment words of one group product)
    // form a group; a term's work must not include another accumulator or its chain (the split sums terms)
    std::vector<Value*> accNext(accs.size());
    std::vector<std::pair<unsigned, unsigned>> terms;   // (accumulator, position in its chain)
    std::map<std::pair<unsigned, unsigned>, unsigned> termId;
    std::vector<std::set<Instruction*>> termLeaves;     // per term, the fill values its tensor work consumes
    for (unsigned i = 0; i < accs.size(); i++) for (unsigned k = 0; k < chains[i].size(); k++) { termId[{i, k}] = (unsigned)terms.size(); terms.push_back({i, k}); }
    std::vector<unsigned> uf(terms.size()); std::iota(uf.begin(), uf.end(), 0u);
    std::function<unsigned(unsigned)> root = [&](unsigned i) -> unsigned { return uf[i] == i ? i : (uf[i] = root(uf[i])); };
    if (why.empty()) {
      std::set<Instruction*> workSet, chainOps;
      for (BasicBlock* BB : K4->blocks()) for (Instruction& I : *BB) if (!(isa<PHINode>(I) && BB == KH) && !I.isTerminator() && !ctl.count(&I)) workSet.insert(&I);
      for (auto& ch : chains) for (const Term& t : ch) chainOps.insert(t.op);
      // what joins two terms into one group: a tensor op (matmul, conversion) or a value resting on one - a fused
      // op's outputs are indivisible. Loads and address arithmetic shared by terms (the B operand of several
      // m-subtiles, the pointers) are duplicated in the copies instead: each copy loads what it needs itself
      // (the CUDA block did the same through the ring).
      std::set<Instruction*> data;
      for (bool changed = true; changed;) {
        changed = false;
        for (BasicBlock* BB : K4->blocks()) for (Instruction& I : *BB) {
          if (!workSet.count(&I) || data.count(&I)) continue;
          bool d = false;
          if (auto* CI = dyn_cast<CallInst>(&I)) d = CI->getCalledFunction() && CI->getCalledFunction()->getName().starts_with("__mvcc_tp_");
          for (Value* o : I.operands()) if (auto* OI = dyn_cast<Instruction>(o)) d = d || data.count(OI);
          if (d) { data.insert(&I); changed = true; }
        }
      }
      std::map<Instruction*, unsigned> groupOf;
      termLeaves.assign(terms.size(), {});
      for (unsigned t = 0; t < terms.size() && why.empty(); t++) {
        const Term& tm = chains[terms[t].first][terms[t].second];
        std::set<Value*> seen; std::vector<Value*> st;
        for (unsigned k = 0; k < tm.op->getNumOperands(); k++) if (k != tm.accIdx && !isa<Function>(tm.op->getOperand(k))) st.push_back(tm.op->getOperand(k));
        while (!st.empty()) {
          Value* x = st.back(); st.pop_back(); auto* I = dyn_cast<Instruction>(x);
          if (!I || !seen.insert(x).second) continue;
          if (std::find(accs.begin(), accs.end(), I) != accs.end() || chainOps.count(I)) { why = "an accumulator feeds another accumulator's term"; break; }
          if (!data.count(I)) {
            // a leaf of the term's tensor work: the word it consumes (the slot fill's value behind the word's
            // trunc/shift), the unit of sharing between terms
            Value* leaf = I;
            for (unsigned d = 0; d < 3; d++) { auto* LI = dyn_cast<Instruction>(leaf); if (LI && (isa<TruncInst>(LI) || (isa<BinaryOperator>(LI) && cast<BinaryOperator>(LI)->getOpcode() == Instruction::LShr)) && K4->contains(LI->getParent())) leaf = LI->getOperand(0); else break; }
            if (auto* LI = dyn_cast<Instruction>(leaf)) if (K4->contains(LI->getParent())) termLeaves[t].insert(LI);
            continue;
          }
          auto it = groupOf.find(I); if (it == groupOf.end()) groupOf[I] = t; else uf[root(it->second)] = root(t);
          for (Value* o : I->operands()) st.push_back(o);
        }
      }
      for (unsigned i = 0; i < accs.size(); i++) accNext[i] = chains[i][0].op;
      for (Instruction* I : workSet) for (User* u : I->users()) { auto* UI = cast<Instruction>(u); if (K4->contains(UI->getParent())) continue; if (std::find(accNext.begin(), accNext.end(), I) == accNext.end()) { why = std::string("a K-loop value other than an accumulator is used after the loop (a ") + I->getOpcodeName() + " by a " + UI->getOpcodeName() + ")"; } }
      for (PHINode* P : accs) for (User* u : P->users()) if (!K4->contains(cast<Instruction>(u)->getParent())) why = "an accumulator's previous value is used after the loop";
    }
    std::map<unsigned, unsigned> groupId;   // root -> dense group index
    if (why.empty()) { for (unsigned t = 0; t < terms.size(); t++) groupId.emplace(root(t), (unsigned)groupId.size()); splitGroups = (unsigned)groupId.size(); }
    // (d) the mode and the factor
    unsigned Fc = 0;
    if (why.empty()) {
      const unsigned maxF = std::min(8u, cap / T);
      if (splitGroups >= 2) { splitByChain = true; Fc = std::min(splitGroups, maxF); }
      else Fc = std::min(maxF, 2u);
      if (Fc < 2) why = "no split factor";
    }
    // (d') which groups share a copy. The groups of one iteration overlap in their operand words (the m-subtile
    // chains of a column half read the same B words; the column halves of an m-subtile the same A words) and the
    // fills above emit a shared word once - but only within one copy: a copy materializes every instruction its
    // groups' words rest on. Groups are packed into copies (balanced, |G|/F each) greedily by the number of loop
    // instructions their word cones share (the dequantization of a B word four m-subtile chains read, the address
    // arithmetic of a K half).
    std::vector<unsigned> copyOfGroup;
    if (why.empty() && splitByChain) {
      std::vector<std::set<Instruction*>> leaves(splitGroups);   // per group: the loop instructions its words rest on
      for (unsigned t = 0; t < terms.size(); t++) {
        std::set<Instruction*>& cone = leaves[groupId[root(t)]];
        std::vector<Instruction*> st(termLeaves[t].begin(), termLeaves[t].end());
        while (!st.empty()) {
          Instruction* I = st.back(); st.pop_back();
          if (!K4->contains(I->getParent()) || !cone.insert(I).second) continue;
          if (isa<PHINode>(I) && I->getParent() == KH) continue;
          for (Value* o : I->operands()) if (auto* OI = dyn_cast<Instruction>(o)) st.push_back(OI);
        }
      }
      auto affinity = [&](unsigned a, unsigned b) { unsigned s = 0; for (Instruction* I : leaves[a]) s += leaves[b].count(I); return s; };
      copyOfGroup.assign(splitGroups, ~0u);
      std::vector<std::vector<unsigned>> members(Fc);
      std::vector<unsigned> capacity(Fc); for (unsigned g = 0; g < Fc; g++) capacity[g] = (splitGroups + Fc - 1 - g) / Fc;
      // one copy at a time: seeded with the first unassigned group, then the groups with the most affinity to its members
      for (unsigned c = 0; c < Fc; c++) while (members[c].size() < capacity[c]) {
        int best = -1; long long bestScore = -1;
        for (unsigned q = 0; q < splitGroups; q++) {
          if (copyOfGroup[q] != ~0u) continue;
          long long score = 0; for (unsigned m : members[c]) score += affinity(q, m);
          if (score > bestScore) { bestScore = score; best = (int)q; }
        }
        copyOfGroup[best] = c; members[c].push_back(best);
      }
    }
    auto ownsTerm = [&](unsigned i, unsigned k, unsigned g) -> bool { return !splitByChain || copyOfGroup[groupId[root(termId[{i, k}])]] == g; };
    std::vector<std::vector<unsigned>> ownersOf(accs.size());   // the copies holding a share of each accumulator, ascending
    if (why.empty()) for (unsigned i = 0; i < accs.size(); i++) for (unsigned g = 0; g < Fc; g++) { bool any = false; for (unsigned k = 0; k < chains[i].size(); k++) any |= ownsTerm(i, k, g); if (any) ownersOf[i].push_back(g); }
    // (e) the side effects outside the loop: predicated on copy 0 below; their hazards
    std::vector<Instruction*> sides; std::vector<Instruction*> sharedSides;   // sharedSides: need a barrier before any shared read
    std::set<const Value*> storedObjs; bool unknownReads = false; std::set<const Value*> readObjs;
    auto isBarrierName = [](StringRef n) { return n == "llvm.nvvm.barrier0" || n.starts_with("llvm.nvvm.barrier") || n.starts_with("llvm.nvvm.bar.sync"); };
    // the objects a pointer may address (through selects, phis, int<->ptr casts and address arithmetic); false: unknown
    std::function<bool(Value*, std::set<const Value*>&, unsigned)> objsOf = [&](Value* p, std::set<const Value*>& out, unsigned depth) -> bool {
      if (depth > 12) return false;
      if (isa<ConstantPointerNull>(p) || (isa<ConstantInt>(p) && cast<ConstantInt>(p)->isZero())) return true;   // the untaken arm of a speculated load: no object
      if (auto* PI = dyn_cast<PtrToIntInst>(p)) return objsOf(PI->getOperand(0), out, depth + 1);
      if (auto* IP = dyn_cast<IntToPtrInst>(p)) return objsOf(IP->getOperand(0), out, depth + 1);
      if (auto* SI = dyn_cast<SelectInst>(p)) return objsOf(SI->getTrueValue(), out, depth + 1) && objsOf(SI->getFalseValue(), out, depth + 1);
      if (auto* PN = dyn_cast<PHINode>(p)) { for (Value* v : PN->incoming_values()) if (!objsOf(v, out, depth + 1)) return false; return true; }
      if (auto* BO = dyn_cast<BinaryOperator>(p)) { if (BO->getOpcode() == Instruction::Add || BO->getOpcode() == Instruction::Sub || BO->getOpcode() == Instruction::Or) { const bool a = !isa<Constant>(BO->getOperand(0)) && BO->getOperand(0)->getType()->isIntegerTy(64), b = !isa<Constant>(BO->getOperand(1)) && BO->getOperand(1)->getType()->isIntegerTy(64); if (a && !b) return objsOf(BO->getOperand(0), out, depth + 1); if (b && !a && BO->getOpcode() != Instruction::Sub) return objsOf(BO->getOperand(1), out, depth + 1); } return false; }
      if (auto* CI = dyn_cast<CastInst>(p)) { if (p->getType()->isPointerTy() || CI->getOperand(0)->getType()->isPointerTy()) return objsOf(CI->getOperand(0), out, depth + 1); return false; }
      if (!p->getType()->isPointerTy()) { return false; }
      const Value* o = getUnderlyingObject(p);
      if (o != p && o->getType()->isPointerTy()) return objsOf(const_cast<Value*>(o), out, depth + 1);
      if (isa<Argument>(o) || isa<GlobalValue>(o) || isa<AllocaInst>(o)) { out.insert(o); return true; }
      if (auto* CI = dyn_cast<CallInst>(o)) if (CI->getCalledFunction() && CI->getCalledFunction()->getName() == "__mvcc_zero_page") { out.insert(o); return true; }
      return false;
    };
    auto objsInto = [&](Value* p, std::set<const Value*>& set) -> bool { std::set<const Value*> out; if (!objsOf(p, out, 0)) { return false; } set.insert(out.begin(), out.end()); return true; };
    if (why.empty()) for (Instruction& I : instructions(F)) {
      if (K4->contains(I.getParent())) continue;
      int cls = 0; bool shared = false; std::string what;   // 0 harmless on every copy, 1 predicate, 2 cannot
      if (auto* CI = dyn_cast<CallInst>(&I)) {
        if (CI->isInlineAsm()) {
          bool alu; asmMnemonics(CI, alu);
          std::vector<PtxInstr> ins; std::string err;
          if (alu) cls = 0;
          else if (!parsePtxAsm(std::string(cast<InlineAsm>(CI->getCalledOperand())->getAsmString()), ins, err) || ins.empty()) { cls = 2; what = "an asm the check cannot parse"; }
          else for (const PtxInstr& in : ins) {
            const std::string& m = in.mnemonic;
            if (m == "bar" || m == "barrier" || m == "mov" || m == "shfl" || m == "vote" || m == "match" || m == "prmt" || m == "cvt" || m == "setp" || m == "selp") continue;
            if (m == "ld") { if (in.hasMod("global")) { for (Value* a : CI->args()) if ((a->getType()->isPointerTy() || a->getType()->isIntegerTy(64)) && !objsInto(a, readObjs)) unknownReads = true; } continue; }
            if (m == "cp") { if (in.hasMod("commit_group") || in.hasMod("wait_group") || in.hasMod("wait_all")) continue; cls = 1; shared = true; for (Value* a : CI->args()) if ((a->getType()->isPointerTy() || a->getType()->isIntegerTy(64)) && !objsInto(a, readObjs)) unknownReads = true; continue; }   // a 32-bit operand is the shared destination
            if (m == "st") { cls = 1; if (in.hasMod("shared") || !in.hasMod("global")) shared = true; else { bool any = false; for (Value* a : CI->args()) if (a->getType()->isPointerTy() || a->getType()->isIntegerTy(64)) any |= objsInto(a, storedObjs); if (!any) unknownReads = true; } continue; }
            cls = 2; what = "an asm " + m; break;
          }
        } else {
          Function* cf = CI->getCalledFunction(); StringRef n = cf ? cf->getName() : StringRef();
          if (!cf) { cls = 2; what = "an indirect call"; }
          else if (isBarrierName(n) || n.starts_with("llvm.nvvm.read.") || n.starts_with("llvm.lifetime") || n.starts_with("llvm.dbg") || n.starts_with("llvm.assume") || n.starts_with("llvm.experimental.noalias") || n.starts_with("llvm.nvvm.shfl") || n.starts_with("llvm.nvvm.vote") || n.starts_with("llvm.nvvm.match") || n.starts_with("llvm.nvvm.redux") || n.starts_with("llvm.nvvm.bar.warp") || n.starts_with("llvm.nvvm.cp.async.commit") || n.starts_with("llvm.nvvm.cp.async.wait") || n == "__mvcc_tp_srclane" || n == "__mvcc_zero_page" || n.starts_with("__mvcc_tp_ld") || n.starts_with("__mvcc_tp_mma_") || n.starts_with("__mvcc_tp_acc2ptx_") || n.starts_with("__mvcc_tp_ptx2acc_") || n.starts_with("__mvcc_tp_accslots_") || cf->doesNotAccessMemory() || cf->onlyReadsMemory()) cls = 0;
          else if (n.starts_with("__nv_")) { for (Value* a : CI->args()) if (a->getType()->isPointerTy()) { cls = 2; what = "a call to " + n.str(); } }
          else if (n.starts_with("__mvcc_cp_async") || n.starts_with("llvm.nvvm.cp.async") || n.starts_with("__mvcc_tp_ctstore_") || n.starts_with("llvm.memcpy") || n.starts_with("llvm.memset")) { cls = 1; shared = true; for (Value* a : CI->args()) if (a->getType()->isPointerTy() && !objsInto(a, readObjs)) unknownReads = true; }
          else if (n.starts_with("llvm.nvvm.atomic")) { cls = CI->use_empty() ? 1 : 2; shared = true; what = "an atomic whose value is used"; }
          else if (cf->isIntrinsic() && !CI->mayHaveSideEffects()) cls = 0;
          else { cls = 2; what = "a call to " + n.str(); }
        }
      } else if (auto* S = dyn_cast<StoreInst>(&I)) { cls = 1; if (S->getPointerAddressSpace() == 3) shared = true; else if (!objsInto(S->getPointerOperand(), storedObjs)) unknownReads = true; }
      else if (isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I)) { cls = I.use_empty() ? 1 : 2; shared = true; what = "an atomic whose value is used"; }
      else if (auto* Ld = dyn_cast<LoadInst>(&I)) { if (Ld->getPointerAddressSpace() != 3 && !objsInto(Ld->getPointerOperand(), readObjs)) unknownReads = true; }
      if (cls == 2) { why = what + " outside the K loop"; break; }
      if (cls == 1) { if (!I.use_empty()) { why = std::string("a side effect outside the K loop produces a value (a ") + I.getOpcodeName() + ")"; break; } sides.push_back(&I); if (shared) sharedSides.push_back(&I); }
    }
    if (why.empty()) for (BasicBlock* BB : K4->blocks()) for (Instruction& I : *BB) if (auto* Ld = dyn_cast<LoadInst>(&I)) if (Ld->getPointerAddressSpace() != 3 && !objsInto(Ld->getPointerOperand(), readObjs)) unknownReads = true;
    if (why.empty() && !storedObjs.empty()) {
      if (unknownReads) why = "a read of device memory whose object is unknown, and stores to device memory";
      for (const Value* o : storedObjs) if (readObjs.count(o)) { why = "the kernel reads device memory it writes"; }
    }
    // a predicated shared store must be followed by a barrier before any shared read (the other copies read copy 0's write)
    if (why.empty()) for (Instruction* S : sharedSides) {
      std::set<BasicBlock*> vis; std::vector<std::pair<BasicBlock*, BasicBlock::iterator>> st; st.push_back({S->getParent(), std::next(S->getIterator())});
      bool ok = true;
      while (!st.empty() && ok) {
        auto [bb, it] = st.back(); st.pop_back();
        bool stop = false;
        for (; it != bb->end(); ++it) {
          if (auto* CI = dyn_cast<CallInst>(&*it)) {
            if (CI->getCalledFunction() && isBarrierName(CI->getCalledFunction()->getName())) { stop = true; break; }
            if (CI->isInlineAsm()) { std::vector<PtxInstr> ins; std::string err; if (parsePtxAsm(std::string(cast<InlineAsm>(CI->getCalledOperand())->getAsmString()), ins, err) && !ins.empty() && (ins[0].mnemonic == "bar" || ins[0].mnemonic == "barrier")) { stop = true; break; } if (parsePtxAsm(std::string(cast<InlineAsm>(CI->getCalledOperand())->getAsmString()), ins, err) && !ins.empty() && ins[0].mnemonic == "ld" && !ins[0].hasMod("global")) { ok = false; break; } }
            if (CI->getCalledFunction() && CI->getCalledFunction()->getName().starts_with("__mvcc_tp_ld")) { ok = false; break; }
          }
          if (auto* Ld = dyn_cast<LoadInst>(&*it)) if (Ld->getPointerAddressSpace() == 3) { ok = false; break; }
        }
        if (stop || !ok) continue;
        for (BasicBlock* Sx : successors(bb)) if (vis.insert(Sx).second) st.push_back({Sx, Sx->begin()});
      }
      if (!ok) { why = "a shared-memory store may be read back before a barrier"; break; }
    }
    if (!why.empty()) note("decode retiling: the K loop is not split across simdgroups (" + why + ")");
    else {
      // ---- rewrite
      // thread ids: every tid.x read is the virtual thread, ntid.x the virtual block
      Function* tidFn = Intrinsic::getOrInsertDeclaration(&Mod, Intrinsic::nvvm_read_ptx_sreg_tid_x);
      Value *vtid, *h, *isH0;
      {
        IRBuilder<> B(&*F.getEntryBlock().getFirstInsertionPt());
        CallInst* ptid = B.CreateCall(tidFn);
        ptid->addRangeRetAttr(ConstantRange(APInt(32, 0), APInt(32, T * Fc)));
        vtid = B.CreateURem(ptid, ConstantInt::get(i32, T), "vtid");
        h = B.CreateUDiv(ptid, ConstantInt::get(i32, T), "copy");
        isH0 = B.CreateICmpEQ(h, ConstantInt::get(i32, 0));
        std::vector<CallInst*> old;
        for (Instruction& I : instructions(F)) if (auto* CI = dyn_cast<CallInst>(&I)) if (CI != ptid && CI->getCalledFunction()) {
          StringRef n = CI->getCalledFunction()->getName();
          if (n == "llvm.nvvm.read.ptx.sreg.tid.x" || n == "llvm.nvvm.read.ptx.sreg.ntid.x") old.push_back(CI);
        }
        for (CallInst* CI : old) { CI->replaceAllUsesWith(CI->getCalledFunction()->getName().ends_with("ntid.x") ? (Value*)ConstantInt::get(i32, T) : vtid); CI->eraseFromParent(); }
      }
      // the control moves to the header (in dependency order, before the header's first work instruction); the
      // header then splits behind it, the latch before its branch: header (phis, control) -> body -> latch (merge
      // phis, the exit branch)
      Instruction* anchor = nullptr;
      for (Instruction& I : *KH) if (!isa<PHINode>(I) && !ctl.count(&I)) { anchor = &I; break; }
      {
        std::set<Instruction*> placed;
        std::function<void(Instruction*)> place = [&](Instruction* I) {
          if (!ctl.count(I) || !placed.insert(I).second) return;
          for (Value* o : I->operands()) if (auto* OI = dyn_cast<Instruction>(o)) place(OI);
          I->moveBefore(anchor->getIterator());
        };
        for (Instruction* I : ctl) place(I);
      }
      BasicBlock* B0 = SplitBlock(KH, anchor, &DT4, &LI4, nullptr, "mvcc.split.body");
      if (Lt == KH) Lt = B0;
      BasicBlock* Lt2 = SplitBlock(Lt, Lt->getTerminator(), &DT4, &LI4, nullptr, "mvcc.split.latch");
      std::vector<BasicBlock*> region;   // the body: every loop block but the header and the latch, in layout order
      for (BasicBlock& BB : F) if (K4->contains(&BB) && &BB != KH && &BB != Lt2) region.push_back(&BB);
      // the copies of the body (chain mode)
      std::vector<BasicBlock*> entries{B0}, exits{Lt};
      std::vector<ValueToValueMapTy> maps(Fc);
      if (splitByChain) for (unsigned g = 1; g < Fc; g++) {
        ValueToValueMapTy& VM = maps[g];
        std::vector<BasicBlock*> clones;
        for (BasicBlock* BB : region) { BasicBlock* NB = CloneBasicBlock(BB, VM, ".c" + std::to_string(g), &F); VM[BB] = NB; clones.push_back(NB); K4->addBasicBlockToLoop(NB, LI4); }
        for (BasicBlock* NB : clones) for (Instruction& I : *NB) RemapInstruction(&I, VM, RF_NoModuleLevelChanges | RF_IgnoreMissingLocals);
        entries.push_back(cast<BasicBlock>((Value*)VM[B0])); exits.push_back(cast<BasicBlock>((Value*)VM[Lt]));
      }
      // the header's branch: chain mode selects the copy's body, iteration mode skips the body on the other copies' iterations
      PHINode* cnt = nullptr;
      {
        Instruction* T0 = KH->getTerminator();
        IRBuilder<> B(T0);
        if (splitByChain) { SwitchInst* SW = B.CreateSwitch(h, B0, Fc - 1); for (unsigned g = 1; g < Fc; g++) SW->addCase(cast<ConstantInt>(ConstantInt::get(i32, g)), entries[g]); }
        else {
          cnt = PHINode::Create(i32, 2, "mvcc.split.it", KH->begin());
          cnt->addIncoming(ConstantInt::get(i32, 0), pre);
          Value* next = B.CreateAdd(cnt, ConstantInt::get(i32, 1)); cnt->addIncoming(next, Lt2);
          B.CreateCondBr(B.CreateICmpEQ(B.CreateURem(cnt, ConstantInt::get(i32, Fc)), h), B0, Lt2);
        }
        T0->eraseFromParent();
      }
      // the merge phis in the latch: each accumulator's next value in each copy is its chain rebuilt over the terms
      // the copy owns (innermost first, from the phi), its old value where it owns none
      std::vector<PHINode*> merged(accs.size());
      for (unsigned i = 0; i < accs.size(); i++) {
        PHINode* Mp = PHINode::Create(f32, Fc + 1, "mvcc.split.acc", Lt2->begin());
        for (unsigned g = 0; g < entries.size(); g++) {
          Value* v;
          if (!splitByChain) v = accNext[i];
          else {
            v = accs[i];
            for (unsigned k = chains[i].size(); k-- > 0;) {
              if (!ownsTerm(i, k, g)) continue;
              Instruction* op = chains[i][k].op; if (g) { auto it = maps[g].find(op); if (it != maps[g].end()) op = cast<Instruction>((Value*)it->second); }
              Instruction* c = op->clone(); c->setOperand(chains[i][k].accIdx, v); c->insertBefore(op->getIterator()); v = c;
            }
          }
          Mp->addIncoming(v, exits[g]);
        }
        if (!splitByChain) Mp->addIncoming(accs[i], KH);
        // every use of the next value outside the body (the phi's backedge, the exit) takes the merge
        std::vector<Use*> uses; for (Use& U : accNext[i]->uses()) { auto* UI = cast<Instruction>(U.getUser()); if (UI == Mp || std::find(region.begin(), region.end(), UI->getParent()) != region.end()) continue; uses.push_back(&U); }
        for (Use* U : uses) U->set(Mp);
        merged[i] = Mp;
      }
      // the unowned work dies in each copy (the body is pure: anything unused goes)
      for (unsigned g = 0; g < entries.size(); g++) {
        std::vector<BasicBlock*> blocks; for (BasicBlock* BB : region) blocks.push_back(g ? cast<BasicBlock>((Value*)maps[g][BB]) : BB);
        bool progress = true;
        while (progress) {
          progress = false;
          std::vector<Instruction*> rev; for (BasicBlock* BB : blocks) for (Instruction& I : *BB) if (!I.isTerminator()) rev.push_back(&I);
          for (auto it = rev.rbegin(); it != rev.rend(); ++it) if ((*it)->use_empty()) { (*it)->eraseFromParent(); progress = true; }
        }
      }
      // the exit: trivial phis resolved, then the exchange
      { std::vector<PHINode*> ps; for (PHINode& P : E->phis()) ps.push_back(&P); for (PHINode* P : ps) { P->replaceAllUsesWith(P->getIncomingValue(0)); P->eraseFromParent(); } }
      std::vector<std::vector<Use*>> outUses(accs.size());
      for (unsigned i = 0; i < accs.size(); i++) for (Use& U : merged[i]->uses()) { auto* UI = cast<Instruction>(U.getUser()); if (!K4->contains(UI->getParent())) outUses[i].push_back(&U); }
      {
        const unsigned L = (unsigned)accs.size(); unsigned W = 1; for (auto& o : ownersOf) W = std::max(W, (unsigned)o.size());
        const unsigned budget = 8192;   // bytes of the shared exchange buffer
        const unsigned Cc = std::max(1u, std::min(L, budget / (W * T * 4)));
        xchgRounds = (L + Cc - 1) / Cc;
        auto* arrTy = ArrayType::get(f32, (uint64_t)W * T * Cc);
        auto* buf = new GlobalVariable(Mod, arrTy, false, GlobalValue::InternalLinkage, UndefValue::get(arrTy), "mvcc.retile.xchg", nullptr, GlobalValue::NotThreadLocal, 3);
        buf->setAlignment(Align(16));
        BasicBlock* rest = E->splitBasicBlock(E->getFirstNonPHIIt(), "mvcc.xchg.rest");
        E->getTerminator()->eraseFromParent();
        IRBuilder<> B(E);
        auto barrier = [&]() { B.Insert(barrierProto->clone()); };
        // slot (rank, thread, j): rank = the copy's position among the accumulator's owners
        auto slot = [&](unsigned rank, unsigned j) -> Value* {
          Value* idx = B.CreateAdd(B.CreateMul(B.CreateAdd(vtid, ConstantInt::get(i32, rank * T)), ConstantInt::get(i32, Cc)), ConstantInt::get(i32, j));
          return B.CreateInBoundsGEP(f32, buf, B.CreateZExt(idx, i64));
        };
        std::vector<Value*> total(L);
        for (unsigned r = 0; r < xchgRounds; r++) {
          const unsigned lo = r * Cc, hi = std::min(L, lo + Cc);
          for (unsigned g = 0; g < Fc; g++) {   // copy g writes its share of the chunk
            std::vector<std::pair<unsigned, unsigned>> mine;   // (accumulator, rank)
            for (unsigned i = lo; i < hi; i++) { auto& o = ownersOf[i]; auto it = std::find(o.begin(), o.end(), g); if (it != o.end()) mine.push_back({i, (unsigned)(it - o.begin())}); }
            if (mine.empty()) continue;
            BasicBlock* then = BasicBlock::Create(ctx, "mvcc.xchg.own", &F, rest);
            BasicBlock* cont = BasicBlock::Create(ctx, "mvcc.xchg.cont", &F, rest);
            B.CreateCondBr(B.CreateICmpEQ(h, ConstantInt::get(i32, g)), then, cont);
            B.SetInsertPoint(then);
            for (auto& [i, rank] : mine) B.CreateStore(merged[i], slot(rank, i - lo));
            B.CreateBr(cont);
            B.SetInsertPoint(cont);
          }
          barrier();
          for (unsigned i = lo; i < hi; i++) {   // every copy sums the owners' shares in the same order
            Value* sum = nullptr;
            for (unsigned rank = 0; rank < ownersOf[i].size(); rank++) { Value* v = B.CreateLoad(f32, slot(rank, i - lo)); sum = sum ? B.CreateFAdd(sum, v) : v; }
            total[i] = sum;
          }
          if (r + 1 < xchgRounds) barrier();
        }
        B.CreateBr(rest);
        for (unsigned i = 0; i < L; i++) for (Use* U : outUses[i]) U->set(total[i]);
      }
      // the side effects outside the loop run on copy 0
      for (Instruction* I : sides) { Instruction* thenTerm = SplitBlockAndInsertIfThen(isH0, I, false); I->moveBefore(thenTerm->getIterator()); }
      F.removeFnAttr("nvvm.maxntid");
      F.addFnAttr("nvvm.maxntid", std::to_string(T * Fc) + ",1,1");
      res.threadScale[F.getName().str()] = Fc;
      splitF = Fc;
      for (unsigned g = 0; g < entries.size(); g++) regions.push_back({entries[g], exits[g], &*entries[g]->getFirstNonPHIIt()});
    }
  }
  // ---- 9. the iteration's loads in one wave. The ring gave the CUDA kernel its operands kStages-1 K tiles ahead of
  // their use; the direct fills above load them at their sites, one fused op at a time, and matmul2d is a scheduling
  // barrier to Metal's compiler: an iteration with four fused ops waited out four device-memory round trips
  // (measured on the 16x128 decode kernel: ~5 us per iteration per block at low occupancy; 0.11 ms for the 4096x4096
  // step). Every device load of the loop whose address is a pure function of the header phis and loop-invariant
  // values (no loads, no calls with effects, no division by an iteration-dependent value), in a block executed on
  // every iteration, moves with its address cone to the end of the header: the iteration's loads issue together,
  // then its matmuls run (0.068 ms; the same loads, the same iteration, no memory written by the loop to order
  // against). Rotating the loop by one iteration instead (iteration i loading what i+1 reads into registers carried
  // by header phis) measured slower: the duplicated address cones and the values live across the backedge cost more
  // than the round trip saved.
  unsigned prefetched = 0;
  if (loopPure) {
    DominatorTree DT3(F); LoopInfo LI3(DT3);
    Loop* K3 = LI3.getLoopFor(KH);
    std::string why;
    BasicBlock* pre = K3 ? K3->getLoopPreheader() : nullptr;
    BasicBlock* latch = K3 ? K3->getLoopLatch() : nullptr;
    BranchInst* LB = latch ? dyn_cast<BranchInst>(latch->getTerminator()) : nullptr;
    if (!K3 || K3->getHeader() != KH) why = "the K loop was not found again";
    else if (!pre || !latch || K3->getExitingBlock() != latch || !LB || !LB->isConditional()) why = "the K loop does not exit from its latch alone";
    if (why.empty() && regions.empty()) regions.push_back({KH, latch, KH->getTerminator()});   // no split: the whole body, loads to the header's end
    // the body copy a block belongs to: the nearest region entry above it in the dominator tree
    auto regionOf = [&](BasicBlock* BB) -> const Region* {
      for (DomTreeNode* n = DT3.getNode(BB); n; n = n->getIDom()) for (const Region& R : regions) if (R.entry == n->getBlock()) return &R;
      return nullptr;
    };
    if (why.empty()) {
      auto pureCall = [&](CallInst* CI) {
        Function* cf = CI->getCalledFunction(); if (!cf) return false;
        if (cf->getName().starts_with("__nv_")) { for (Value* a : CI->args()) if (a->getType()->isPointerTy()) return false; return true; }
        if (cf->getName().starts_with("llvm.nvvm.read.ptx.sreg") || cf->getName() == "__mvcc_zero_page") return true;
        return cf->isIntrinsic() && !CI->mayHaveSideEffects() && !CI->mayReadFromMemory();
      };
      // the address cone of a load: the loop's instructions it depends on, operands first; the header phis it rests on
      // whether a value varies with the iteration (rests on a header phi through the loop's instructions)
      std::function<bool(Value*, std::set<Value*>&)> varies = [&](Value* v, std::set<Value*>& seen) -> bool {
        auto* I = dyn_cast<Instruction>(v); if (!I || !K3->contains(I->getParent())) return false;
        if (!seen.insert(v).second) return false;
        if (isa<PHINode>(I)) return true;
        for (Value* op : I->operands()) if (varies(op, seen)) return true;
        return false;
      };
      const Region* R = nullptr;   // the region of the load whose cone is being computed
      auto cone = [&](Value* addr, std::vector<Instruction*>& order, std::set<PHINode*>& phis, std::string& reason) -> bool {
        std::set<Instruction*> seen;
        std::function<bool(Value*)> go = [&](Value* v) -> bool {
          auto* I = dyn_cast<Instruction>(v); if (!I || !K3->contains(I->getParent())) return true;
          if (seen.count(I)) return true;
          if (auto* P = dyn_cast<PHINode>(I)) { if (P->getParent() != KH) { reason = "a phi inside the loop body"; return false; } phis.insert(P); seen.insert(I); return true; }
          if (auto* CI = dyn_cast<CallInst>(I)) { if (!pureCall(CI)) { reason = "depends on a call to " + (CI->getCalledFunction() ? CI->getCalledFunction()->getName().str() : std::string("?")); return false; } }
          else if (I->isTerminator() || isa<LoadInst>(I) || I->mayHaveSideEffects() || I->mayReadFromMemory()) { reason = std::string("depends on a ") + I->getOpcodeName(); return false; }
          if (!DT3.dominates(I->getParent(), latch) && !(R && DT3.dominates(I->getParent(), R->exit))) { reason = std::string("depends on a conditional ") + I->getOpcodeName(); return false; }
          if (auto* BO = dyn_cast<BinaryOperator>(I)) {
            const unsigned op = BO->getOpcode();
            // a division's clone at the next iteration's values must not trap: constant divisor, or one that does not vary with the iteration (this iteration divided by it)
            if (op == Instruction::SDiv || op == Instruction::UDiv || op == Instruction::SRem || op == Instruction::URem) {
              std::set<Value*> s;
              if (!(isa<ConstantInt>(BO->getOperand(1)) && !cast<ConstantInt>(BO->getOperand(1))->isZero()) && varies(BO->getOperand(1), s)) { reason = "a division by an iteration-dependent value"; return false; }
            }
          }
          for (Value* op : I->operands()) if (!go(op)) return false;
          seen.insert(I); order.push_back(I); return true;
        };
        return go(addr);
      };
      std::vector<LoadInst*> cands;
      for (BasicBlock* BB : K3->blocks()) {
        for (Instruction& I : *BB) {
          auto* L = dyn_cast<LoadInst>(&I); if (!L || !L->isSimple()) continue;
          const unsigned as = L->getPointerAddressSpace();
          if (as != 0 && as != 1) continue;   // device memory only (shared has no latency to hide)
          if (!L->getType()->isFirstClassType() || L->getType()->isAggregateType()) continue;
          const Region* LR = regionOf(BB);
          if (!(LR && DT3.dominates(BB, LR->exit)) && !DT3.dominates(BB, latch)) { continue; }
          cands.push_back(L);
        }
      }
      for (LoadInst* L : cands) {
        std::vector<Instruction*> order; std::set<PHINode*> phis; std::string reason;
        R = regionOf(L->getParent());
        if (!cone(L->getPointerOperand(), order, phis, reason)) continue;
        // the load and its address cone move to the start of its body copy (the header's end when the loop is not
        // split), so the iteration's loads issue in one wave before its first matmul (matmul2d is a scheduling
        // barrier to Metal's compiler; issued at their sites the loads go out in one wave per fused op)
        Instruction* at = R ? R->anchor : KH->getTerminator();
        for (Instruction* I : order) if (I != at && !(I->getParent() == at->getParent() && I->comesBefore(at)) && I->getParent() != KH) I->moveBefore(at->getIterator());
        if (L != at) L->moveBefore(at->getIterator());
        prefetched++;
      }
      if (!prefetched) why = "no device load of the K loop has a pure address over the loop's phis";
    }
    if (!prefetched) note("decode retiling: no software prefetch (" + why + ")");
  }
  // the assumptions, for the launch
  auto cmpName = [](unsigned p) -> const char* {
    switch (p) { case CmpInst::ICMP_EQ: return "eq"; case CmpInst::ICMP_NE: return "ne"; case CmpInst::ICMP_SLT: return "slt"; case CmpInst::ICMP_SLE: return "sle"; case CmpInst::ICMP_SGT: return "sgt"; case CmpInst::ICMP_SGE: return "sge"; case CmpInst::ICMP_ULT: return "ult"; case CmpInst::ICMP_ULE: return "ule"; case CmpInst::ICMP_UGT: return "ugt"; case CmpInst::ICMP_UGE: return "uge"; default: return nullptr; }
  };
  auto width = [&](Type* t) -> unsigned { return t->isPointerTy() ? 64 : t->isIntegerTy() ? t->getIntegerBitWidth() : 0; };
  std::function<bool(EP, std::string&)> ser = [&](EP e, std::string& o) -> bool {
    int64_t c; if (A.constInt(e, c)) { o += std::to_string(c); return true; }
    if (e->k == EK::Const && e->ty->isIntegerTy(1)) { o += e->c.isOne() ? "1" : "0"; return true; }
    const unsigned w = width(e->ty);
    if (!w) return false;
    std::string body;
    switch (e->k) {
      case EK::Sym: { const Symbol& s = A.symbol(e->sym); if (s.kind != Symbol::Arg) return false; o += "(arg " + std::to_string(cast<Argument>(s.v)->getArgNo()) + ")"; return true; }
      case EK::Poly: body = "(+"; for (auto& t : e->terms) { body += " (* " + std::to_string(t.second); for (uint32_t a : t.first) { body += " "; if (!ser(A.atomExpr(a), body)) return false; } body += ")"; } body += ")"; break;
      case EK::Cmp: { const char* n = cmpName(e->op); if (!n) return false; body = std::string("(") + n + " " + std::to_string(width(e->args[0]->ty)) + " "; if (!ser(e->args[0], body)) return false; body += " "; if (!ser(e->args[1], body)) return false; body += ")"; break; }
      case EK::Ite: body = "(ite "; for (EP a : e->args) { if (!ser(a, body)) return false; body += " "; } body += ")"; break;
      case EK::Op: {
        const char* n = nullptr;
        switch (e->op) { case Instruction::And: n = "and"; break; case Instruction::Or: n = "or"; break; case Instruction::Xor: n = "xor"; break; case Instruction::Shl: n = "shl"; break; case Instruction::LShr: n = "lshr"; break; case Instruction::AShr: n = "ashr"; break; case Instruction::SDiv: n = "sdiv"; break; case Instruction::UDiv: n = "udiv"; break; case Instruction::SRem: n = "srem"; break; case Instruction::URem: n = "urem"; break; case Instruction::Mul: n = "mul"; break; case Instruction::Add: n = "add"; break; case Instruction::Sub: n = "sub"; break;
          case Instruction::SExt: n = "sext"; break; case Instruction::ZExt: n = "zext"; break; case Instruction::Trunc: n = "trunc"; break; default: return false; }
        body = std::string("(") + n;
        if (e->op == Instruction::ZExt || e->op == Instruction::SExt) body += " " + std::to_string(width(e->args[0]->ty));
        for (EP a : e->args) { body += " "; if (!ser(a, body)) return false; }
        body += ")"; break;
      }
      case EK::Call: if (!minMaxCallee(e->callee)) return false; body = minMaxCallee(e->callee) < 0 ? "(min" : "(max"; for (EP a : e->args) { body += " "; if (!ser(a, body)) return false; } body += ")"; break;
      default: return false;
    }
    o += w < 64 && e->ty->isIntegerTy() && !e->ty->isIntegerTy(1) ? "(wrap " + std::to_string(w) + " " + body + ")" : body;
    return true;
  };
  for (auto& kv : X.assumptions) {
    std::string s;
    if (A.isTrue(kv.first.first)) { if (!ser(kv.first.second, s)) { note("decode retiling: an assumption cannot be expressed for the launch: " + A.str(kv.first.second, 6)); res.assumptions[F.getName().str()].push_back("(false)"); continue; } }
    else { s = "(=> "; if (!ser(kv.first.first, s)) { res.assumptions[F.getName().str()].push_back("(false)"); continue; } s += " "; if (!ser(kv.first.second, s)) { res.assumptions[F.getName().str()].push_back("(false)"); continue; } s += ")"; }
    res.assumptions[F.getName().str()].push_back(s);
  }
  note("decode retiling applied: " + (std::to_string(blockPos.size()) + " operand blocks filled slot-wise from device memory (" + std::to_string(distinctPairs) + " distinct of " + std::to_string(fillPairs) + " slot word pairs emitted; fragment conversions deleted), ") + std::to_string(dead.size()) + " instruction(s) of the operand slice and " + std::to_string(deadCopies.size()) + " staging copies deleted" + (nRing ? ", " + std::to_string(nRing) + " quantizer ring read(s) rewritten to device" : std::string()) + (hoistedBarriers ? ", " + std::to_string(hoistedBarriers) + " K-loop barrier(s) hoisted to the loop's entry and exits" : std::string()) + (splitF ? ", the K loop split across " + std::to_string(splitF) + " simdgroup copies by " + (splitByChain ? std::to_string(splitGroups) + " accumulator chain(s)" : "iteration") + " (" + std::to_string(T) + " -> " + std::to_string(T * splitF) + " threads, " + std::to_string(xchgRounds) + " exchange round(s) at the exit)" : std::string()) + (prefetched ? ", " + std::to_string(prefetched) + " device load(s) hoisted to the iteration's start" : std::string()) + (shrink ? ", shared bytes from " + std::to_string(cutHi) + " relocated down by " + std::to_string(shrink) : std::string()) + ", " + std::to_string(res.assumptions[F.getName().str()].size()) + " launch assumption(s)");
  return true;
}

}  // namespace staging

bool emitFromStir(llvm::Function& F, const TensorRecoveryOptions& opts, TensorRecoveryResult& res, const std::string& kname, unsigned maxThreads) {
  staging::StagingElim S(F, opts, res, kname, maxThreads);
  return S.emitFromStir();
}

}  // namespace mvcc
