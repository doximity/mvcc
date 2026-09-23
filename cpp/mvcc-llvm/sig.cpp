// Semantic iteration graph: loop classification, recurrence algebra, per-kernel summary (sig.h).
#include "sig.h"
#include "coords.h"
#include "stir.h"
#include "engine.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/Analysis/AssumptionCache.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <map>
#include <set>

namespace mvcc {
namespace sig {
using namespace sym;

const char* loopKindName(LoopKind k) {
  switch (k) { case LoopKind::Bounded: return "bounded"; case LoopKind::Traversal: return "traversal"; case LoopKind::Generic: return "generic"; case LoopKind::Refused: return "refused"; }
  return "?";
}

const char* algebraName(Algebra a) {
  switch (a) {
    case Algebra::Invariant: return "invariant"; case Algebra::Induction: return "induction"; case Algebra::Sum: return "sum";
    case Algebra::Max: return "max"; case Algebra::Min: return "min"; case Algebra::And: return "and"; case Algebra::Or: return "or";
    case Algebra::Xor: return "xor"; case Algebra::TensorSum: return "tensor-sum"; case Algebra::Tuple: return "tuple";
    case Algebra::Permutation: return "permutation"; case Algebra::OrderSensitive: return "order-sensitive"; case Algebra::Unknown: return "unknown";
  }
  return "?";
}

Permissions permissionsOf(Algebra a) {
  Permissions p;
  switch (a) {
    case Algebra::Sum: case Algebra::TensorSum: case Algebra::Xor: p.partition = p.reorder = p.treeCombine = true; break;
    case Algebra::Max: case Algebra::Min: case Algebra::And: case Algebra::Or: p.partition = p.reorder = p.treeCombine = p.duplicate = true; break;
    case Algebra::Permutation: p.duplicate = true; break;
    case Algebra::Invariant: case Algebra::Induction: p.partition = p.reorder = p.duplicate = p.treeCombine = true; break;  // control: replicated freely
    default: break;
  }
  return p;
}

// ---------------------------------------------------------------- loop classification

static bool isLaunchSreg(Value* V, StringRef name) {
  for (int n = 0; n < 4 && V; n++) {
    if (auto* C = dyn_cast<CastInst>(V)) { V = C->getOperand(0); continue; }
    break;
  }
  auto* CI = dyn_cast<CallInst>(V);
  return CI && CI->getCalledFunction() && CI->getCalledFunction()->getName() == name;
}

static bool valueUsesLaunchSreg(Value* V, StringRef reg) {
  SmallPtrSet<Value*, 16> vis;
  SmallVector<Value*, 8> stk = {V};
  while (!stk.empty()) {
    Value* x = stk.pop_back_val();
    if (!x || !vis.insert(x).second) continue;
    if (isLaunchSreg(x, reg)) return true;
    if (auto* I = dyn_cast<Instruction>(x))
      for (Value* op : I->operands()) stk.push_back(op);
  }
  return false;
}

static bool isGridStrideStepValue(Value* V) {
  auto* M = dyn_cast<BinaryOperator>(V);
  if (!M || M->getOpcode() != Instruction::Mul) return false;
  bool a = valueUsesLaunchSreg(M->getOperand(0), "llvm.nvvm.read.ptx.sreg.blockDim.x")
        && valueUsesLaunchSreg(M->getOperand(1), "llvm.nvvm.read.ptx.sreg.gridDim.x");
  bool b = valueUsesLaunchSreg(M->getOperand(1), "llvm.nvvm.read.ptx.sreg.blockDim.x")
        && valueUsesLaunchSreg(M->getOperand(0), "llvm.nvvm.read.ptx.sreg.gridDim.x");
  return a || b;
}

static bool isBlockParallelStepValue(Value* V) {
  if (valueUsesLaunchSreg(V, "llvm.nvvm.read.ptx.sreg.ntid.x")) return true;
  if (auto* C = dyn_cast<ConstantInt>(V)) return !C->isZero();
  return false;
}

static bool loopLatchStepMatches(const Loop* L, const std::function<bool(Value*)>& pred) {
  BasicBlock* latch = L->getLoopLatch();
  if (!latch) return false;
  for (Instruction& I : *latch) {
    auto* BO = dyn_cast<BinaryOperator>(&I);
    if (!BO || BO->getOpcode() != Instruction::Add) continue;
    if (pred(BO->getOperand(0)) || pred(BO->getOperand(1))) return true;
  }
  return false;
}

static bool dependsOnTid(Value* v, int depth, std::set<Value*>& seen) {
  if (depth > 64 || !seen.insert(v).second) return false;
  if (auto* CI = dyn_cast<CallInst>(v)) {
    if (Function* Cf = CI->getCalledFunction()) {
      StringRef n = Cf->getName();
      if (n.starts_with("llvm.nvvm.read.ptx.sreg.tid") || n == "llvm.nvvm.read.ptx.sreg.laneid") return true;
    }
  }
  if (auto* I = dyn_cast<Instruction>(v)) for (Value* op : I->operands()) if (dependsOnTid(op, depth + 1, seen)) return true;
  return false;
}

// does the SCEV mention a value computed inside L (a load, a call, an opaque instruction)?
static bool scevDependsOnLoop(const SCEV* S, const Loop* L) {
  struct Finder {
    const Loop* L; bool dep = false;
    bool follow(const SCEV* S) {
      if (auto* U = dyn_cast<SCEVUnknown>(S)) if (auto* I = dyn_cast<Instruction>(U->getValue())) if (L->contains(I->getParent())) dep = true;
      return true;
    }
    bool isDone() const { return dep; }
  } f{L};
  visitAll(S, f);
  return f.dep;
}

static bool loopContainsBarrier(const Loop* L) {
  for (BasicBlock* BB : L->blocks()) {
    for (Instruction& I : *BB) {
      auto* C = dyn_cast<CallInst>(&I);
      if (!C || !C->getCalledFunction()) continue;
      if (C->getCalledFunction()->getName().contains("barrier")) return true;
    }
  }
  return false;
}

std::vector<LoopClass> classifyLoops(Function& F, LoopInfo& LI, ScalarEvolution& SE, unsigned maxBounded) {
  std::vector<LoopClass> out;
  for (const Loop* L : LI.getLoopsInPreorder()) {
    LoopClass c{L, LoopKind::Generic};
    const unsigned tc = SE.getSmallConstantTripCount(const_cast<Loop*>(L));
    // SCEV has no closed form for `for (i = tid; i < n; i += blockDim)` (non-constant stride); the trip count is
    // still a function of the launch parameters when every exit compares an induction of L (loop-invariant step)
    // against a loop-invariant value
    bool parametricExits = true;
    {
      SmallVector<BasicBlock*, 4> exiting; L->getExitingBlocks(exiting);
      for (BasicBlock* E : exiting) {
        auto* Br = dyn_cast<BranchInst>(E->getTerminator());
        auto* IC = Br && Br->isConditional() ? dyn_cast<ICmpInst>(Br->getCondition()) : nullptr;
        if (!IC) { parametricExits = false; break; }
        for (Value* op : IC->operands()) {
          const SCEV* s = SE.getSCEV(op);
          if (SE.isLoopInvariant(s, L)) continue;
          auto* AR = dyn_cast<SCEVAddRecExpr>(s);
          if (AR && AR->getLoop() == L && AR->isAffine() && SE.isLoopInvariant(AR->getStepRecurrence(SE), L)) continue;
          parametricExits = false;
        }
        if (!parametricExits) break;
      }
    }
    if (tc && tc <= maxBounded && parametricExits) { c.kind = LoopKind::Bounded; c.trip = tc; out.push_back(c); continue; }
    // loop-carried data: a header phi that is not an integer induction (fp, vector, struct, or an integer whose SCEV
    // is not an add recurrence of this loop)
    bool tidDep = false;
    for (PHINode& PN : L->getHeader()->phis()) {
      Type* ty = PN.getType();
      bool data = ty->isFPOrFPVectorTy() || ty->isVectorTy() || ty->isStructTy();
      if (!data && ty->isIntegerTy()) {
        const SCEV* s = SE.getSCEV(&PN);
        auto* AR = dyn_cast<SCEVAddRecExpr>(s);
        data = !(AR && AR->getLoop() == L && AR->isAffine());
      }
      if (data) c.dataPhis++;
      for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) if (!L->contains(PN.getIncomingBlock(i))) { std::set<Value*> seen; if (dependsOnTid(PN.getIncomingValue(i), 0, seen)) tidDep = true; }
    }
    c.uniform = !tidDep;
    if (tc && tc <= maxBounded && !parametricExits) {
      // a constant-trip loop with a data-dependent early exit: unrolling it accumulates one path literal per
      // iteration (quadratic conditions); it runs as one generic pass instead
      c.kind = LoopKind::Generic; c.earlyExit = true; c.trip = tc; c.uniform = false; out.push_back(c); continue;
    }
    const SCEV* btc = SE.getBackedgeTakenCount(const_cast<Loop*>(L));
    const bool dataDependent = (isa<SCEVCouldNotCompute>(btc) || scevDependsOnLoop(btc, L)) && !parametricExits;
    const bool gridStride = loopLatchStepMatches(L, isGridStrideStepValue);
    const bool blockStride = loopLatchStepMatches(L, isBlockParallelStepValue);
    // no loop-carried data: one generic pass - unless the loop is a per-thread strided traversal of a coordinate
    // (`for (d = tid; d < D; d += T)`, exits parametric, header phis inductions): its induction is a coordinate the
    // discovery must see move, so it is probed like a traversal loop
    if (c.dataPhis == 0) {
      bool trav = (tidDep && parametricExits && !L->getHeader()->phis().empty()) || gridStride
               || (blockStride && parametricExits);
      c.kind = trav ? LoopKind::Traversal : LoopKind::Generic;
    }
    else if (dataDependent) {
      // Block tournament / merge (top-k): shift-carried active with __syncthreads is not a parametric
      // traversal, but it is a uniform block-collective schedule, not an opaque data-dependent trip.
      if (loopContainsBarrier(L)) { c.kind = LoopKind::Generic; c.uniform = true; }
      else {
        c.kind = LoopKind::Refused;
        c.why = std::string("data-dependent loop: ") + (isa<SCEVCouldNotCompute>(btc) ? "trip count not computable with loop-carried data" : "trip count depends on a value computed in the loop");
      }
    } else c.kind = LoopKind::Traversal;
    out.push_back(c);
  }
  return out;
}

void applyLoopClasses(Executor& X, const std::vector<LoopClass>& classes) {
  for (auto& c : classes) {
    switch (c.kind) {
      case LoopKind::Bounded: break;
      case LoopKind::Traversal: X.traversal.insert(c.L); break;
      case LoopKind::Generic: X.generic[c.L] = c.uniform; break;
      case LoopKind::Refused: X.generic[c.L] = false; break;
    }
  }
}

// ---------------------------------------------------------------- recurrence algebra

namespace {

struct Ctx {
  Arena& A; uint32_t pid; std::set<uint32_t> siblingIds;
  std::map<EP, bool> memo;
  bool contains(EP e, uint32_t sid) {
    if (!e) return false;
    if (sid == pid) { auto it = memo.find(e); if (it != memo.end()) return it->second; }
    bool r = false;
    switch (e->k) {
      case EK::Sym: r = e->sym == sid; break;
      case EK::Poly:
        for (auto& t : e->terms) for (uint32_t a : t.first) { uint32_t s; if (A.isSymAtom(a, s)) { if (s == sid) r = true; } else if (contains(A.atomExpr(a), sid)) r = true; }
        break;
      default: for (EP x : e->args) if (contains(x, sid)) { r = true; break; }
    }
    if (sid == pid) memo[e] = r;
    return r;
  }
  bool isPhi(EP e) { e = A.unwrap(e); return e && e->k == EK::Sym && e->sym == pid; }
  bool isAnyPhi(EP e) { e = A.unwrap(e); return e && e->k == EK::Sym && (e->sym == pid || siblingIds.count(e->sym)); }
  bool free(EP e) { return !contains(e, pid); }
  std::string calleeName(EP e) {
    if (e->k != EK::Call) return "";
    if (e->callee) return e->callee->getName().str();
    return "asm";
  }
};

bool isMaxName(StringRef n) { return n.starts_with("llvm.maxnum") || n.starts_with("llvm.maximum") || n == "__nv_fmaxf" || n == "__nv_fmax" || n.starts_with("llvm.smax") || n.starts_with("llvm.umax") || n == "__nv_max" || n == "__nv_umax"; }
bool isMinName(StringRef n) { return n.starts_with("llvm.minnum") || n.starts_with("llvm.minimum") || n == "__nv_fminf" || n == "__nv_fmin" || n.starts_with("llvm.smin") || n.starts_with("llvm.umin") || n == "__nv_min" || n == "__nv_umin"; }
bool isFmaName(StringRef n) { return n.starts_with("llvm.fma.") || n.starts_with("llvm.fmuladd.") || n == "__nv_fmaf" || n == "__nv_fma" || n == "__nv_fmaf_rn"; }

bool isIdentity(Arena& A, EP init, Algebra k) {
  init = A.unwrap(init);
  if (!init) return false;
  if (init->k == EK::Poly) { int64_t v; if (!A.constInt(init, v)) return false; switch (k) { case Algebra::Sum: case Algebra::Or: case Algebra::Xor: return v == 0; case Algebra::And: return v == -1; default: return false; } }
  if (init->k == EK::Const) {
    if (init->ty && init->ty->isFloatingPointTy()) {
      const uint64_t bits = init->c.getZExtValue(); const unsigned w = init->ty->getPrimitiveSizeInBits();
      const uint64_t zero = 0, negzero = w == 32 ? 0x80000000ull : (w == 16 ? 0x8000ull : 0x8000000000000000ull);
      const uint64_t ninf = w == 32 ? 0xff800000ull : (w == 16 ? 0xfc00ull : 0xfff0000000000000ull), pinf = w == 32 ? 0x7f800000ull : (w == 16 ? 0x7c00ull : 0x7ff0000000000000ull);
      const uint64_t fmin = w == 32 ? 0xff7fffffull : 0, fmax = w == 32 ? 0x7f7fffffull : 0;
      switch (k) { case Algebra::Sum: return bits == zero || bits == negzero; case Algebra::Max: return bits == ninf || bits == fmin; case Algebra::Min: return bits == pinf || bits == fmax; default: return false; }
    }
    const int64_t v = init->c.getSExtValue();
    switch (k) { case Algebra::Sum: case Algebra::Or: case Algebra::Xor: return v == 0; case Algebra::And: return v == -1; default: return false; }
  }
  return false;
}

Classified classifyOne(Ctx& C, EP n, EP init);

// the additive chain: fma(a, b, x) / fadd(x, a) / fsub(x, a) / x + a  down to the phi
bool additiveChain(Ctx& C, EP n, unsigned& terms) {
  Arena& A = C.A;
  EP x = n; terms = 0;
  for (unsigned guard = 0; guard < 4096; guard++) {
    if (C.isPhi(x)) return terms > 0;
    if (x->k == EK::Poly) {
      // integer: exactly one monomial `1 * phi`, every other monomial free of the phi
      const uint32_t patom = A.atomOf(A.mkSymRaw(A.symbol(C.pid)));
      bool one = false;
      for (auto& t : x->terms) {
        if (t.first.size() == 1 && t.first[0] == patom) { if (t.second != 1 || one) return false; one = true; continue; }
        for (uint32_t a : t.first) { uint32_t s; if (a == patom || (!A.isSymAtom(a, s) && C.contains(A.atomExpr(a), C.pid))) return false; }
      }
      if (!one) return false;
      terms += x->terms.size() - 1; return true;
    }
    if (x->k == EK::Call && isFmaName(C.calleeName(x)) && x->args.size() >= 3) {
      if (C.free(x->args[0]) && C.free(x->args[1])) { x = x->args[2]; terms++; continue; }
      if (C.free(x->args[1]) && C.free(x->args[2])) { return false; }  // phi multiplied: linear, not additive
      return false;
    }
    if (x->k == EK::Op && (x->op == Instruction::FAdd || x->op == Instruction::Add) && x->args.size() == 2) {
      if (C.free(x->args[1])) { x = x->args[0]; terms++; continue; }
      if (C.free(x->args[0])) { x = x->args[1]; terms++; continue; }
      return false;
    }
    if (x->k == EK::Op && (x->op == Instruction::FSub || x->op == Instruction::Sub) && x->args.size() == 2 && C.free(x->args[1])) { x = x->args[0]; terms++; continue; }
    // a rounding/convert pair around the accumulator (bf16 accumulators: fpext(fptrunc(...)))
    if (x->k == EK::Op && x->args.size() == 1 && (x->op == Instruction::FPExt || x->op == Instruction::FPTrunc)) { x = x->args[0]; continue; }
    return false;
  }
  return false;
}

Classified classifyOne(Ctx& C, EP n, EP init) {
  Arena& A = C.A;
  Classified r;
  if (!n) { r.kind = Algebra::Unknown; r.detail = "no back-edge value"; return r; }
  n = A.unwrap(n);
  if (n->k == EK::Undef) { r.kind = Algebra::Unknown; r.detail = "undef"; return r; }
  if (C.isPhi(n)) { r.kind = Algebra::Invariant; return r; }
  // predicated update: ite(c, x, phi) / ite(c, phi, x) with c free of the phi
  if (n->k == EK::Ite && n->args.size() == 3 && C.free(n->args[0])) {
    if (C.isPhi(n->args[2]) && !C.isPhi(n->args[1])) { r = classifyOne(C, n->args[1], init); r.predicated = true; return r; }
    if (C.isPhi(n->args[1]) && !C.isPhi(n->args[2])) { r = classifyOne(C, n->args[2], init); r.predicated = true; return r; }
  }
  // max / min as compare + select
  if (n->k == EK::Ite && n->args.size() == 3 && n->args[0]->k == EK::Cmp && n->args[0]->args.size() == 2) {
    EP c = n->args[0]; EP a = n->args[1], b = n->args[2];
    const bool phiA = C.isPhi(a), phiB = C.isPhi(b);
    if ((phiA && C.free(b)) || (phiB && C.free(a))) {
      EP other = phiA ? b : a;
      EP l = c->args[0], rr = c->args[1];
      const bool lPhi = C.isPhi(l) && A.unwrap(rr) == A.unwrap(other), rPhi = C.isPhi(rr) && A.unwrap(l) == A.unwrap(other);
      if (lPhi || rPhi) {
        const auto pred = (CmpInst::Predicate)c->op;
        bool gt = pred == CmpInst::FCMP_OGT || pred == CmpInst::FCMP_UGT || pred == CmpInst::FCMP_OGE || pred == CmpInst::FCMP_UGE || pred == CmpInst::ICMP_SGT || pred == CmpInst::ICMP_UGT || pred == CmpInst::ICMP_SGE || pred == CmpInst::ICMP_UGE;
        bool lt = pred == CmpInst::FCMP_OLT || pred == CmpInst::FCMP_ULT || pred == CmpInst::FCMP_OLE || pred == CmpInst::FCMP_ULE || pred == CmpInst::ICMP_SLT || pred == CmpInst::ICMP_ULT || pred == CmpInst::ICMP_SLE || pred == CmpInst::ICMP_ULE;
        if (gt || lt) {
          // ite(phi > t, phi, t) = max; ite(t > phi, phi, t) = min; and the mirrored forms
          bool selectsPhiWhenLeftGreater = phiA;  // arm 1 taken when the compare holds
          bool leftIsPhi = lPhi;
          bool isMax = (gt && ((leftIsPhi && selectsPhiWhenLeftGreater) || (!leftIsPhi && !selectsPhiWhenLeftGreater))) || (lt && ((leftIsPhi && !selectsPhiWhenLeftGreater) || (!leftIsPhi && selectsPhiWhenLeftGreater)));
          r.kind = isMax ? Algebra::Max : Algebra::Min; r.terms = 1; r.initIdentity = isIdentity(A, init, r.kind); return r;
        }
      }
    }
  }
  // struct / vector recurrence: classify the parts
  if (n->k == EK::Concat) {
    std::set<Algebra> kinds; unsigned terms = 0; bool pred = false;
    for (EP part : n->args) { Classified pc = classifyOne(C, part, nullptr); kinds.insert(pc.kind); terms += pc.terms; pred |= pc.predicated; if (pc.kind == Algebra::OrderSensitive || pc.kind == Algebra::Unknown) r.detail = pc.detail; }
    if (kinds.size() == 1) { r.kind = *kinds.begin(); r.terms = terms; r.predicated = pred; return r; }
    kinds.erase(Algebra::Invariant);
    if (kinds.size() == 1) { r.kind = *kinds.begin(); r.terms = terms; r.predicated = pred; return r; }
    r.kind = Algebra::Tuple; r.detail = "mixed parts:"; for (Algebra k : kinds) r.detail += std::string(" ") + algebraName(k); return r;
  }
  // tensor accumulator: mma(..., phi, ...)
  if (n->k == EK::Call && StringRef(C.calleeName(n)).starts_with("__mvcc_tp_mma_")) {
    bool inArgs = false; for (EP a : n->args) if (!C.free(a)) inArgs = true;
    if (inArgs) { r.kind = Algebra::TensorSum; r.terms = 1; r.initIdentity = isIdentity(A, init, Algebra::Sum); return r; }
  }
  // max / min intrinsics
  if (n->k == EK::Call && n->args.size() == 2) {
    const std::string cn = C.calleeName(n);
    const bool mx = isMaxName(cn), mn = isMinName(cn);
    if ((mx || mn) && ((C.isPhi(n->args[0]) && C.free(n->args[1])) || (C.isPhi(n->args[1]) && C.free(n->args[0])))) { r.kind = mx ? Algebra::Max : Algebra::Min; r.terms = 1; r.initIdentity = isIdentity(A, init, r.kind); return r; }
  }
  // bitwise
  if (n->k == EK::Op && n->args.size() == 2 && (n->op == Instruction::And || n->op == Instruction::Or || n->op == Instruction::Xor)) {
    if ((C.isPhi(n->args[0]) && C.free(n->args[1])) || (C.isPhi(n->args[1]) && C.free(n->args[0]))) {
      r.kind = n->op == Instruction::And ? Algebra::And : n->op == Instruction::Or ? Algebra::Or : Algebra::Xor; r.terms = 1; r.initIdentity = isIdentity(A, init, r.kind); return r;
    }
  }
  // additive chains (fp and integer)
  unsigned terms = 0;
  if (additiveChain(C, n, terms)) {
    // an integer chain whose terms are all constants is loop control
    if (n->k == EK::Poly) { bool allConst = true; EP p = A.mkSymRaw(A.symbol(C.pid)); uint32_t patom = A.atomOf(p); for (auto& t : n->terms) { if (t.first.empty()) continue; if (t.first.size() == 1 && t.first[0] == patom) continue; allConst = false; } if (allConst) { r.kind = Algebra::Induction; r.terms = terms; return r; } }
    r.kind = Algebra::Sum; r.terms = terms; r.initIdentity = isIdentity(A, init, Algebra::Sum); return r;
  }
  // selection among values with no arithmetic on the phi
  if (n->k == EK::Ite) {
    bool leavesOk = true;
    std::function<void(EP)> walk = [&](EP e) { e = A.unwrap(e); if (e->k == EK::Ite) { if (!C.free(e->args[0])) leavesOk = false; walk(e->args[1]); walk(e->args[2]); return; } if (!C.isAnyPhi(e) && !C.free(e)) leavesOk = false; };
    walk(n);
    if (leavesOk) { r.kind = Algebra::Permutation; return r; }
  }
  // reads another header phi: tuple state. A linear form l' = a*l + b with a, b free of l composes associatively.
  for (uint32_t s : C.siblingIds) if (C.contains(n, s)) {
    r.kind = Algebra::Tuple; r.detail = "reads a sibling phi: " + A.str(n, 3);
    EP scale = nullptr;   // b (the shift) only has to be free of l; the classification needs a
    auto isMulByPhi = [&](EP m, EP& other) { m = A.unwrap(m); if (m->k == EK::Op && m->op == Instruction::FMul && m->args.size() == 2) { if (C.isPhi(m->args[0]) && C.free(m->args[1])) { other = m->args[1]; return true; } if (C.isPhi(m->args[1]) && C.free(m->args[0])) { other = m->args[0]; return true; } } return false; };
    EP u = A.unwrap(n);
    if (u->k == EK::Call && isFmaName(C.calleeName(u)) && u->args.size() >= 3) {
      if (C.isPhi(u->args[0]) && C.free(u->args[1]) && C.free(u->args[2])) scale = u->args[1];
      else if (C.isPhi(u->args[1]) && C.free(u->args[0]) && C.free(u->args[2])) scale = u->args[0];
    } else if (u->k == EK::Op && u->op == Instruction::FAdd && u->args.size() == 2) {
      EP o; if (isMulByPhi(u->args[0], o) && C.free(u->args[1])) scale = o; else if (isMulByPhi(u->args[1], o) && C.free(u->args[0])) scale = o;
    }
    if (scale) {
      r.permits.partition = r.permits.treeCombine = true;
      // the online-softmax rescaling: exp(m_old - m_new) with m a sibling Max phi
      bool expOfSibling = false;
      std::function<void(EP)> find = [&](EP e) { e = A.unwrap(e); if (e->k == EK::Call) { std::string nm = C.calleeName(e); if (nm.find("exp") != std::string::npos) for (uint32_t sid : C.siblingIds) if (C.contains(e, sid)) expOfSibling = true; } for (EP a : e->args) if (a) find(a); };
      find(scale);
      r.detail = expOfSibling ? "linear l' = a*l + b with a = exp(m_old - m_new): online softmax" : "linear l' = a*l + b";
      if (expOfSibling) r.permits.reorder = true;
    }
    return r;
  }
  if (n->k == EK::Sym && A.symbol(n->sym).kind == Symbol::Unknown) { r.kind = Algebra::Unknown; r.detail = "opaque value"; return r; }
  if (C.free(n)) { r.kind = Algebra::Permutation; r.detail = "overwritten each iteration"; return r; }
  r.kind = Algebra::OrderSensitive; r.detail = A.str(n, 3);
  return r;
}

}  // namespace

static bool irFeedsPhi(Value* X, PHINode* P, unsigned depth, bool& throughMma) {
  if (!X || depth > 64) return false;
  if (X == P) return true;
  if (auto* C = dyn_cast<CastInst>(X)) return irFeedsPhi(C->getOperand(0), P, depth + 1, throughMma);
  if (auto* EE = dyn_cast<ExtractElementInst>(X)) return irFeedsPhi(EE->getVectorOperand(), P, depth + 1, throughMma);
  if (auto* IE = dyn_cast<InsertElementInst>(X))
    return irFeedsPhi(IE->getOperand(0), P, depth + 1, throughMma) || irFeedsPhi(IE->getOperand(1), P, depth + 1, throughMma);
  if (auto* EV = dyn_cast<ExtractValueInst>(X)) return irFeedsPhi(EV->getAggregateOperand(), P, depth + 1, throughMma);
  if (auto* IV = dyn_cast<InsertValueInst>(X))
    return irFeedsPhi(IV->getAggregateOperand(), P, depth + 1, throughMma) || irFeedsPhi(IV->getInsertedValueOperand(), P, depth + 1, throughMma);
  if (auto* SV = dyn_cast<ShuffleVectorInst>(X))
    return irFeedsPhi(SV->getOperand(0), P, depth + 1, throughMma) || irFeedsPhi(SV->getOperand(1), P, depth + 1, throughMma);
  if (auto* S = dyn_cast<SelectInst>(X))
    return irFeedsPhi(S->getTrueValue(), P, depth + 1, throughMma) || irFeedsPhi(S->getFalseValue(), P, depth + 1, throughMma);
  if (auto* BO = dyn_cast<BinaryOperator>(X)) {
    if (BO->getOpcode() == Instruction::FAdd || BO->getOpcode() == Instruction::FSub)
      return irFeedsPhi(BO->getOperand(0), P, depth + 1, throughMma) || irFeedsPhi(BO->getOperand(1), P, depth + 1, throughMma);
  }
  if (auto* CI = dyn_cast<CallInst>(X)) {
    auto walkArgs = [&]() -> bool {
      for (Value* a : CI->args()) if (irFeedsPhi(a, P, depth + 1, throughMma)) return true;
      return false;
    };
    if (CI->isInlineAsm()) {
      if (auto* IA = dyn_cast<InlineAsm>(CI->getCalledOperand())) {
        StringRef as = IA->getAsmString();
        if (as.contains("mma.sync") || as.contains("mma.")) throughMma = true;
      }
      return walkArgs();
    }
    Function* F = CI->getCalledFunction();
    if (!F) return false;
    StringRef n = F->getName();
    if (n.starts_with("__mvcc_tp_mma") || n.starts_with("__mvcc_tp_acc") || n.contains("mma")
        || n.starts_with("llvm.nvvm.wmma") || n.starts_with("llvm.nvvm.mma")) {
      throughMma = true;
      return walkArgs();
    }
    if ((F->getIntrinsicID() == Intrinsic::fma || n == "__nv_fmaf" || n.starts_with("llvm.fma.")
         || n.starts_with("llvm.fmuladd.")) && CI->arg_size() >= 3)
      return irFeedsPhi(CI->getArgOperand(2), P, depth + 1, throughMma);
  }
  return false;
}

Classified classifyRecurrence(Arena& A, const Recurrence& r, const std::vector<Recurrence>& siblings) {
  Symbol s; s.kind = Symbol::ThreadValue; s.v = r.phi; s.ty = r.phi->getType(); s.thread = r.thread;
  Ctx C{A, A.mkSymRaw(s)->sym, {}, {}};
  for (auto& o : siblings) if (o.L == r.L && o.thread == r.thread && o.phi != r.phi) { Symbol t; t.kind = Symbol::ThreadValue; t.v = o.phi; t.ty = o.phi->getType(); t.thread = o.thread; C.siblingIds.insert(A.mkSymRaw(t)->sym); }
  Classified k = classifyOne(C, r.next, r.init);
  if ((k.kind == Algebra::Unknown || k.kind == Algebra::OrderSensitive) && r.phi && r.L) {
    bool throughMma = false, feeds = false;
    for (unsigned i = 0; i < r.phi->getNumIncomingValues(); i++) {
      if (!r.L->contains(r.phi->getIncomingBlock(i))) continue;
      if (irFeedsPhi(r.phi->getIncomingValue(i), r.phi, 0, throughMma)) { feeds = true; break; }
    }
    if (feeds) {
      k.kind = throughMma ? Algebra::TensorSum : Algebra::Sum;
      k.terms = 1;
      k.initIdentity = isIdentity(A, r.init, Algebra::Sum);
      k.detail = throughMma ? "IR mma/acc chain (opaque EP)" : "IR fma/fadd chain (opaque EP)";
    } else if (k.kind == Algebra::Unknown) {
      k.kind = Algebra::Permutation;
      if (k.detail.empty()) k.detail = k.predicated ? "predicated opaque fill" : "opaque fill";
    }
  }
  if (k.kind != Algebra::Tuple) k.permits = permissionsOf(k.kind);
  return k;
}

// ---------------------------------------------------------------- per-kernel summary

static std::string blockName(const BasicBlock* BB) { std::string s; raw_string_ostream os(s); BB->printAsOperand(os, false); return s; }
static std::string valueName(const Value* V) { std::string s; raw_string_ostream os(s); V->printAsOperand(os, false); return s; }

SigOptions defaultEmitSigOptions() {
  SigOptions o;
  o.coords = true;
  o.stir = true;
  o.threads = 32;
  o.iterCount = 4;
  o.stepBudget = 2000000;
  o.wallBudget = 12;
  return o;
}

static bool runSigProbes(Executor& X, const SigOptions& o, unsigned T, size_t nLoops, std::string& refused) {
  const unsigned run = std::min(T, o.threads);
  std::vector<unsigned> threads;
  const bool laneForce = getenv("MVCC_SIG_LANE") && getenv("MVCC_SIG_LANE")[0] == '1';
  const bool lane = laneForce || nLoops > 8;
  if (lane) {
    for (unsigned t = 0; t < T; t += 32) threads.push_back(t);
    for (unsigned b = 1; b < 32 && b < T; b <<= 1) threads.push_back(b);
    std::sort(threads.begin(), threads.end());
    threads.erase(std::unique(threads.begin(), threads.end()), threads.end());
  } else {
    for (unsigned t = 0; t < run; t++) threads.push_back(t);
    if (o.coords)
      for (unsigned b = 1; b < T; b <<= 1)
        if (b >= run) threads.push_back(b);
  }
  const auto t0 = std::chrono::steady_clock::now();
  auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  refused.clear();
  for (unsigned t : threads) {
    if (o.wallBudget > 0 && elapsed() > o.wallBudget) {
      refused = "time budget of " + std::to_string((int)o.wallBudget) + " s exceeded";
      break;
    }
    std::string w;
    if (!X.runRetile(t, w)) {
      if (w.find("does not model") != std::string::npos) w = "under-probed: " + w;
      else if (w.find("time budget") == std::string::npos && w.find("data-dependent") == std::string::npos
               && w.find("effect-ordered") == std::string::npos)
        w = "under-probed: " + w;
      refused = "thread " + std::to_string(t) + ": " + w;
      break;
    }
  }
  if (refused.empty() && o.wallBudget > 0 && elapsed() > o.wallBudget)
    refused = "time budget of " + std::to_string((int)o.wallBudget) + " s exceeded";
  return refused.empty();
}

static void fillLoopStmtsFromRecurrences(Arena& A, Executor& X, const std::vector<LoopClass>& classes,
                                         std::vector<stir::LoopStmt>& loopStmts) {
  loopStmts.clear();
  for (auto& c : classes) {
    stir::LoopStmt LS;
    LS.header = blockName(c.L->getHeader());
    LS.depth = c.L->getLoopDepth();
    LS.kind = c.kind;
    LS.trip = c.trip;
    LS.uniform = c.uniform;
    LS.why = c.why;
    loopStmts.push_back(LS);
    if (c.kind != LoopKind::Traversal) continue;
    unsigned firstThread = ~0u;
    for (auto& r : X.recurrences)
      if (r.L == c.L) firstThread = std::min(firstThread, r.thread);
    if (firstThread == ~0u) continue;
    std::map<Algebra, unsigned> tally;
    Permissions p;
    bool anyData = false;
    for (auto& r : X.recurrences) {
      if (r.L != c.L || r.thread != firstThread) continue;
      Classified k = classifyRecurrence(A, r, X.recurrences);
      tally[k.kind]++;
      if (k.kind != Algebra::Invariant && k.kind != Algebra::Induction && k.kind != Algebra::Permutation) {
        if (!anyData) {
          p = k.permits;
          anyData = true;
        } else {
          p.partition &= k.permits.partition;
          p.reorder &= k.permits.reorder;
          p.duplicate &= k.permits.duplicate;
          p.treeCombine &= k.permits.treeCombine;
        }
      }
    }
    stir::LoopStmt& ls = loopStmts.back();
    for (auto& kv : tally) ls.algebra.push_back({kv.first, kv.second});
    ls.permits = p;
    ls.anyData = anyData;
  }
}

bool buildKernelProgram(Function& F, unsigned launchBound, const SigOptions& o, stir::Program& program, std::string& refused) {
  program = {};
  refused.clear();
  if (!o.coords || !o.stir) {
    refused = "coords and stir required for program build";
    return false;
  }
  const DataLayout& DL = F.getParent()->getDataLayout();
  DominatorTree DT(F);
  LoopInfo LI(DT);
  TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple());
  TargetLibraryInfo TLI(TLII);
  AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  const unsigned T = launchBound ? launchBound : 256;
  auto classes = classifyLoops(F, LI, SE);
  for (auto& c : classes)
    if (c.kind == LoopKind::Refused) {
      refused = c.why;
      return false;
    }
  Arena A(DL, F.getContext());
  Executor X(F, DT, LI, A, T, nullptr);
  X.retile = true;
  X.recordRecurrences = true;
  X.recordDevStores = true;
  X.recordDevLoads = true;
  X.reuseUniform = true;
  X.iterCount = o.iterCount;
  X.stepBudget = o.stepBudget;
  X.wallBudget = o.wallBudget;
  applyLoopClasses(X, classes);
  if (!runSigProbes(X, o, T, classes.size(), refused)) return false;
  coords::Discovery D = coords::discover(A, X, T);
  std::vector<stir::LoopStmt> loopStmts;
  fillLoopStmtsFromRecurrences(A, X, classes, loopStmts);
  program = stir::build(A, D, classes, loopStmts, llvm::demangle(F.getName().str()));
  return true;
}

bool analyzeKernel(Function& F, unsigned launchBound, const SigOptions& o, std::vector<std::string>& out) {
  const DataLayout& DL = F.getParent()->getDataLayout();
  DominatorTree DT(F); LoopInfo LI(DT);
  TargetLibraryInfoImpl TLII(F.getParent()->getTargetTriple()); TargetLibraryInfo TLI(TLII); AssumptionCache AC(F);
  ScalarEvolution SE(F, TLI, AC, DT, LI);
  const unsigned T = launchBound ? launchBound : 256;
  const unsigned run = std::min(T, o.threads);
  auto classes = classifyLoops(F, LI, SE);
  Arena A(DL, F.getContext());
  Executor X(F, DT, LI, A, T, nullptr);
  X.retile = true; X.recordRecurrences = true; X.recordDevStores = true; X.recordDevLoads = o.coords; X.reuseUniform = true; X.iterCount = o.iterCount; X.stepBudget = o.stepBudget; X.wallBudget = o.wallBudget;
  applyLoopClasses(X, classes);
  std::string refused;
  const bool laneForce = getenv("MVCC_SIG_LANE") && getenv("MVCC_SIG_LANE")[0] == '1';
  const bool lane = laneForce || classes.size() > 8;
  const auto t0 = std::chrono::steady_clock::now();
  auto elapsed = [&]() { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); };
  runSigProbes(X, o, T, classes.size(), refused);
  unsigned nTrav = 0, nGen = 0, nBnd = 0, nRef = 0;
  for (auto& c : classes) { switch (c.kind) { case LoopKind::Traversal: nTrav++; break; case LoopKind::Generic: nGen++; break; case LoopKind::Bounded: nBnd++; break; case LoopKind::Refused: nRef++; break; } }
  unsigned threadsRun = lane ? std::min(T, (T + 31) / 32 + 5) : run;
  {
    std::string s = "sig: T=" + std::to_string(T) + (launchBound ? "" : " (assumed)") + " threads-run=" + std::to_string(threadsRun) + (lane ? " lane-symbolic" : "") + " loops=" + std::to_string(classes.size()) + " (traversal " + std::to_string(nTrav) + ", generic " + std::to_string(nGen) + ", bounded " + std::to_string(nBnd) + ", refused " + std::to_string(nRef) + ")";
    s += " device-stores=" + std::to_string(X.devStores.size()) + " shared-writes=" + std::to_string(X.smemWrites.size()) + " shared-reads=" + std::to_string(X.smemReads.size()) + " barriers=" + std::to_string(X.finalPhase) + " tensor-calls=" + std::to_string(X.tpCalls.size()) + (X.privateAccesses ? " private-accesses=" + std::to_string(X.privateAccesses) + " (alloca contents not modeled)" : "");
    if (!refused.empty()) s += " REFUSED: " + refused;
    out.push_back(s);
  }
  if (o.dumpEffects) {
    unsigned n = 0;
    for (auto& st : X.devStores) {
      if (n >= 8) break;
      out.push_back("  effect store[" + std::to_string(n) + "] " + A.str(st.addr, 4));
      n++;
    }
  }
  std::vector<stir::LoopStmt> loopStmts;
  for (auto& c : classes) {
    stir::LoopStmt LS; LS.header = blockName(c.L->getHeader()); LS.depth = c.L->getLoopDepth(); LS.kind = c.kind; LS.trip = c.trip; LS.uniform = c.uniform; LS.why = c.why;
    loopStmts.push_back(LS);
    stir::LoopStmt& ls = loopStmts.back();
    std::string s = "  loop " + blockName(c.L->getHeader()) + " depth=" + std::to_string(c.L->getLoopDepth()) + " kind=" + loopKindName(c.kind);
    if (c.kind == LoopKind::Bounded) s += "(" + std::to_string(c.trip) + ")";
    if (c.kind == LoopKind::Generic) s += c.uniform ? " uniform" : " per-thread"; if (c.earlyExit) s += " (constant trip " + std::to_string(c.trip) + " with a data-dependent exit)";
    if (c.kind == LoopKind::Refused) s += ": " + c.why;
    if (c.dataPhis) s += " data-phis=" + std::to_string(c.dataPhis);
    auto rw = X.recurrenceWhy.find(c.L);
    if (rw != X.recurrenceWhy.end()) s += " (recurrence pass failed: " + rw->second + ")";
    out.push_back(s);
    if (c.kind != LoopKind::Traversal) continue;
    // the first executed thread's recurrences of this loop
    unsigned firstThread = ~0u;
    for (auto& r : X.recurrences) if (r.L == c.L) { firstThread = std::min(firstThread, r.thread); }
    if (firstThread == ~0u) { out.push_back("    (not reached by the executed threads)"); continue; }
    std::map<Algebra, unsigned> tally; std::vector<Permissions> perms;
    for (auto& r : X.recurrences) {
      if (r.L != c.L || r.thread != firstThread) continue;
      Classified k = classifyRecurrence(A, r, X.recurrences);
      tally[k.kind]++;
      if (k.kind != Algebra::Invariant && k.kind != Algebra::Induction && k.kind != Algebra::Permutation) perms.push_back(k.permits);
      std::string line = "    phi " + valueName(r.phi) + " : " + algebraName(k.kind);
      if (k.terms) line += " terms=" + std::to_string(k.terms);
      if (k.predicated) line += " predicated";
      if (k.kind == Algebra::Sum || k.kind == Algebra::TensorSum || k.kind == Algebra::Max || k.kind == Algebra::Min || k.kind == Algebra::And || k.kind == Algebra::Or || k.kind == Algebra::Xor) line += k.initIdentity ? " init=identity" : " init=non-identity";
      if (!k.detail.empty()) line += " [" + k.detail + "]";
      if (o.dumpExprs) line += "\n      init: " + A.str(r.init, 4) + "\n      next: " + A.str(r.next, 5);
      out.push_back(line);
    }
    std::string sum = "    algebra:";
    for (auto& kv : tally) sum += " " + std::string(algebraName(kv.first)) + "x" + std::to_string(kv.second);
    Permissions p; bool anyData = false;
    for (auto& q : perms) { if (!anyData) { p = q; anyData = true; } else { p.partition &= q.partition; p.reorder &= q.reorder; p.duplicate &= q.duplicate; p.treeCombine &= q.treeCombine; } }
    if (anyData) sum += std::string(" -> permits:") + (p.partition ? " partition" : "") + (p.reorder ? " reorder" : "") + (p.duplicate ? " duplicate" : "") + (p.treeCombine ? " tree-combine" : "") + (!p.partition && !p.reorder && !p.duplicate ? " nothing (order-sensitive along this loop)" : "");
    {
      bool online = false;
      for (auto& r : X.recurrences) {
        if (r.L != c.L || r.thread != firstThread) continue;
        Classified k = classifyRecurrence(A, r, X.recurrences);
        if (k.detail.find("online softmax") != std::string::npos) online = true;
      }
      if (online && tally.count(Algebra::Max)) sum += " online (max,sum,acc)";
    }
    out.push_back(sum);
    for (auto& kv : tally) ls.algebra.push_back({kv.first, kv.second});
    ls.permits = p; ls.anyData = anyData;
  }
  if (o.coords && refused.empty() && !(o.wallBudget > 0 && elapsed() > o.wallBudget)) {
    coords::Discovery D = coords::discover(A, X, T);
    for (auto& l : D.lines) out.push_back(l);
    if (o.stir) {
      stir::Program P = stir::build(A, D, classes, loopStmts, llvm::demangle(F.getName().str()));
      for (auto& l : stir::dump(A, P)) out.push_back(l);
      engine::analyze(P, out);
    }
  }
  return refused.empty();
}

}  // namespace sig
}  // namespace mvcc
