// Coordinate discovery (coords.h).
#include "coords.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <functional>
#include <map>
#include <set>

namespace mvcc {
namespace coords {
using namespace sym;

static std::string valueName(const Value* V) { std::string s; raw_string_ostream os(s); V->printAsOperand(os, false); return s; }
std::string siteName(const Instruction* I) {
  if (!I) return "?";
  if (auto* S = dyn_cast<StoreInst>(I)) return "store to " + valueName(S->getPointerOperand());
  if (auto* C = dyn_cast<CallInst>(I)) { if (C->isInlineAsm()) return "cp.async"; if (Function* f = C->getCalledFunction()) return f->getName().str(); }
  if (I->getType()->isVoidTy()) return I->getOpcodeName();
  return valueName(I);
}
std::string polyStr(Arena& A, EP e) { return e ? A.str(e, 4) : "0"; }

namespace {

// ---------------------------------------------------------------- uniformity

// is `e` a function of the launch parameters only (arguments, block indices, uniform loop values, constants)?
// Placeholder symbols (LoopValue with a name; see Abstractor) count as uniform: they stand for a derived coordinate.
struct Uniform {
  Arena& A; std::map<EP, bool> memo; std::string blame;
  bool ok(EP e) {
    if (!e) return true;
    auto it = memo.find(e); if (it != memo.end()) return it->second;
    bool r = true;
    switch (e->k) {
      case EK::Const: case EK::Undef: break;
      case EK::Sym: {
        const Symbol& s = A.symbol(e->sym);
        switch (s.kind) {
          case Symbol::Arg: case Symbol::Global: case Symbol::SharedBase: case Symbol::LoopValue: break;
          case Symbol::Sreg: r = s.name.find("tid") == std::string::npos && s.name.find("laneid") == std::string::npos && s.name.find("warpid") == std::string::npos; if (!r) blame = "a thread-index register (" + s.name + ")"; break;
          case Symbol::ThreadValue: r = false; blame = "a per-thread value" + (s.v ? " (" + valueName(s.v) + ")" : std::string()); break;
          case Symbol::Unknown: r = false; blame = "an opaque per-thread value"; break;
          case Symbol::SmemByte: r = false; blame = "a shared-memory byte"; break;
          case Symbol::Private: r = false; blame = "a stack object"; break;
          case Symbol::Coord: r = false; blame = "a coordinate symbol"; break;
          case Symbol::Guard: r = ok(A.guardDef(e->sym)); break;
        }
        break;
      }
      case EK::Poly:
        for (auto& t : e->terms) for (uint32_t a : t.first) { uint32_t s; if (A.isSymAtom(a, s)) { if (!ok(A.mkSymRaw(A.symbol(s)))) r = false; } else if (!ok(A.atomExpr(a))) r = false; }
        break;
      case EK::Load: r = false; blame = "a loaded value"; break;
      default: for (EP x : e->args) if (!ok(x)) r = false;
    }
    memo[e] = r;
    return r;
  }
};

// the pointer-typed argument / global an address polynomial is an offset from (coefficient 1, exactly one)
bool baseOf(Arena& A, EP p, std::string& name, uint32_t& symId) {
  name.clear(); symId = ~0u;
  if (p->k != EK::Poly) return false;
  for (auto& t : p->terms) {
    uint32_t s;
    if (t.first.size() == 1 && A.isSymAtom(t.first[0], s)) {
      const Symbol& sy = A.symbol(s);
      if ((sy.kind == Symbol::Arg || sy.kind == Symbol::Global) && sy.ty && sy.ty->isPointerTy()) {
        if (symId != ~0u) return false;
        symId = s; name = sy.v ? (sy.v->hasName() ? sy.v->getName().str() : valueName(sy.v)) : "?";
      }
    }
  }
  return symId != ~0u;
}

// the leaves of a selected address: (path condition, leaf), null / zero leaves dropped (the source reads nothing there)
void leavesOf(Arena& A, EP e, EP path, std::vector<std::pair<EP, EP>>& out, unsigned depth) {
  EP u = A.unwrap(e);
  if (u->k == EK::Ite && depth < 12) {
    leavesOf(A, u->args[1], A.andb(path, u->args[0]), out, depth + 1);
    leavesOf(A, u->args[2], A.andb(path, A.notb(u->args[0])), out, depth + 1);
    return;
  }
  int64_t c;
  if (A.constInt(e, c) && c == 0) return;
  if (u->k == EK::Const && u->c.isZero()) return;
  out.push_back({path, e});
}

// ---------------------------------------------------------------- physical points and samples

struct Sample { std::vector<int> x; EP val; };   // physical point (one entry per axis) and the polynomial there

void dedupe(std::vector<Sample>& S) {
  std::sort(S.begin(), S.end(), [](const Sample& a, const Sample& b) { return a.x != b.x ? a.x < b.x : a.val->id < b.val->id; });   // ids, not pointers: deterministic
  S.erase(std::unique(S.begin(), S.end(), [](const Sample& a, const Sample& b) { return a.x == b.x && a.val == b.val; }), S.end());
}

// ---------------------------------------------------------------- derived quantities
//
// An opaque integer node whose operands depend on the physical point only through their constant terms - a division
// or remainder by a runtime uniform (`c / chunks_per_row`), a shift or mask of such a quotient, a load through an
// affine index (`idx[row]`) - is replaced by a placeholder symbol shared by every sample with the same shape (the node
// with the constant terms of its operands removed). The operands' constant terms are recorded per sample and fitted
// as affine maps of their own: the derived quantity is d = op(R_1 + P_1(x), ..., R_n + P_n(x)) with the R_i uniform.
// Addresses then fit affinely over Phys extended by the placeholders; each placeholder is one more axis (its stride
// is the coefficient of the placeholder in the fitted address).
struct Derived {
  EP placeholder; EP proto; Type* ty; EK kind; unsigned op; Function* callee; Value* asmv; unsigned loadBytes = 0;
  bool uniform = false;                          // every operand map is constant: a launch parameter, folded back
  std::vector<EP> rest;                          // per operand: the sample-independent part (uniform poly or non-integer expr)
  std::vector<bool> intArg;
  std::vector<std::vector<Sample>> consts;       // per integer operand: (x, constant term)
  std::vector<std::vector<int>> xs;              // the physical points the node was seen at
  std::string shape;                             // for the report
  int cls = -1;                                  // the class (Abstractor::classes index) the member joined
};

struct Abstractor {
  Arena& A; Type* i64;
  std::vector<Derived> members;                  // pass 1: one per distinct opaque node (constants included)
  std::vector<Derived> derived;                  // pass 2: classes - members of one shape whose sample points are disjoint
  std::map<EP, size_t> byNode;
  std::map<EP, EP> memo;   // per sample (cleared by the caller between samples)
  std::vector<int> x;      // the current sample's point
  Uniform U{A};

  EP placeholderOf(const std::string& name, Type* ty) {
    Symbol s; s.kind = Symbol::LoopValue; s.name = name; s.ty = ty;
    return A.mkSym(s);
  }
  // rebuilds `e` with every opaque non-uniform integer node abstracted; returns nullptr when a node cannot be
  // (a per-thread opaque value the executor could not follow)
  // does an opaque integer node carry a constant term in an integer operand (a possible physical-point dependence
  // the executor already folded: `c / chunks_per_row` for thread c)?
  bool constantBearing(EP e) {
    for (EP a : e->args) if (a->ty && (a->ty->isIntegerTy() || a->ty->isPointerTy())) {
      if (a->k == EK::Const) return true;
      EP p = A.asPoly(a);
      if (p->terms.empty()) return true;   // the constant 0
      for (auto& t : p->terms) if (t.first.empty() && t.second != 0) return true;
    }
    return false;
  }
  EP go(EP e, std::string& why) {
    auto it = memo.find(e); if (it != memo.end()) return it->second;
    EP r = nullptr;
    Uniform Uc{A};
    const bool opaque = e->k == EK::Op || e->k == EK::Call || e->k == EK::Load || e->k == EK::Ite || e->k == EK::Cmp;
    // a polynomial is always taken apart (its atoms may be constant-bearing quotients the executor folded)
    if (e->k != EK::Poly && Uc.ok(e) && !(opaque && constantBearing(e) && e->ty && (e->ty->isIntegerTy() || e->ty->isPointerTy()))) r = e;
    else switch (e->k) {
      case EK::Poly: {
        EP acc = A.mkPoly(e->ty, {});
        for (auto& t : e->terms) {
          EP term = A.mkInt(e->ty, t.second);
          for (uint32_t a : t.first) {
            uint32_t s;
            EP ae = A.isSymAtom(a, s) ? A.mkSymRaw(A.symbol(s)) : A.atomExpr(a);
            EP se = go(ae, why); if (!se) return memo[e] = nullptr;
            term = A.mul(term, A.retype(A.asPoly(se), e->ty));
          }
          acc = A.add(acc, term);
        }
        r = acc;
        break;
      }
      case EK::Sym: { Uniform Ub{A}; Ub.ok(e); why = Ub.blame; return memo[e] = nullptr; }
      case EK::Const: case EK::Undef: r = e; break;
      case EK::Op: case EK::Cmp: case EK::Ite: case EK::Load: case EK::Call: case EK::Concat: case EK::Bytes: {
        std::vector<EP> args;
        for (size_t i = 0; i < e->args.size(); i++) {
          // a load's guard is the access predicate, not part of the loaded value's identity: dropped from the shape
          if (e->k == EK::Load && i == 1) { args.push_back(A.mkBool(true)); continue; }
          EP sa = go(e->args[i], why); if (!sa) return memo[e] = nullptr; args.push_back(sa);
        }
        {   // uniform once its operands are, and carrying no constant (a cast of a loaded index): keep
          EP rb = A.rebuild(e, args); Uniform Ur{A};
          if (e->k != EK::Load && Ur.ok(rb) && !constantBearing(rb)) { r = rb; break; }
        }
        if (!e->ty || !(e->ty->isIntegerTy() || e->ty->isPointerTy())) {   // a float / vector node: keep with abstracted operands
          r = A.rebuild(e, args); break;
        }
        // pass 1: one member per distinct node (its operands with the inner nodes abstracted)
        EP node = A.rebuild(e, args);
        std::vector<EP> rest; std::vector<bool> intArg; std::vector<int64_t> consts;
        for (EP a : args) {
          if (a->ty && (a->ty->isIntegerTy() || a->ty->isPointerTy())) {
            EP p = A.asPoly(a); int64_t c = 0;
            std::vector<std::pair<std::vector<uint32_t>, int64_t>> terms;
            for (auto& t : p->terms) { if (t.first.empty()) c = t.second; else terms.push_back(t); }
            rest.push_back(A.mkPoly(p->ty, terms)); intArg.push_back(true); consts.push_back(c);
          } else { rest.push_back(a); intArg.push_back(false); consts.push_back(0); }
        }
        size_t j;
        auto kit = byNode.find(node);
        if (kit == byNode.end()) {
          j = members.size(); byNode[node] = j;
          Derived d; d.proto = e; d.ty = e->ty; d.kind = e->k; d.op = e->op; d.callee = e->callee; d.asmv = e->asmv; d.rest = rest; d.intArg = intArg; d.consts.resize(args.size());
          if (e->k == EK::Load) d.loadBytes = A.sizeOf(e->ty);
          for (size_t i = 0; i < args.size(); i++) if (intArg[i]) d.consts[i].push_back({{}, A.mkInt(i64, consts[i])});
          d.placeholder = placeholderOf("m" + std::to_string(j), e->ty);
          members.push_back(d);
        } else j = kit->second;
        members[j].xs.push_back(x);
        r = A.retype(A.asPoly(members[j].placeholder), e->ty);
        break;
      }
    }
    return memo[e] = r;
  }

  // pass 2: classes. Members of one shape (the node with its operands' constant terms removed, inner members
  // replaced by their classes) whose sample points are disjoint are the same physical-point-dependent quantity seen
  // at different points (`c / cpr` for c = 0, 1, 2, ...); members that co-occur at a point are distinct quantities
  // (`K / 128` and `K / 512`). Returns the substitution member placeholder -> class placeholder.
  std::map<uint32_t, EP> classify() {
    std::map<uint32_t, EP> sub; std::map<EP, EP> smemo;
    std::vector<uint32_t> memberSym(members.size());
    for (size_t j = 0; j < members.size(); j++) memberSym[j] = A.unwrap(members[j].placeholder)->sym;
    std::map<std::string, std::vector<size_t>> shapeClasses;   // shape key -> class indices
    for (bool progress = true; progress;) {
      progress = false;
      for (size_t j = 0; j < members.size(); j++) {
        Derived& m = members[j];
        if (m.cls >= 0) continue;
        // every inner member placeholder classed?
        bool ready = true;
        for (EP r : m.rest) { std::set<uint32_t> found; A.containsSymbol(r, Symbol::LoopValue, &found); for (uint32_t sid : found) for (size_t k = 0; k < members.size(); k++) if (memberSym[k] == sid && members[k].cls < 0) ready = false; }
        if (!ready) continue;
        std::vector<EP> rest; for (EP r : m.rest) rest.push_back(A.subst(r, sub, smemo));
        std::string key = std::to_string((int)m.kind) + "|" + std::to_string(m.op) + "|" + std::to_string(m.proto->lo) + "|" + std::to_string(m.proto->len) + "|" + std::to_string((uintptr_t)m.callee) + "|" + std::to_string((uintptr_t)m.asmv) + "|" + std::to_string((uintptr_t)m.ty);
        for (EP r : rest) key += "|" + std::to_string(r->id);
        std::set<std::vector<int>> mine(m.xs.begin(), m.xs.end());
        int target = -1;
        for (size_t c : shapeClasses[key]) {
          bool disjoint = true;
          for (auto& x : derived[c].xs) if (mine.count(x)) { disjoint = false; break; }
          if (disjoint) { target = (int)c; break; }
        }
        if (target < 0) {
          target = (int)derived.size();
          Derived d = m; d.rest = rest; d.xs.clear(); d.consts.assign(m.rest.size(), {});
          d.placeholder = placeholderOf("d" + std::to_string(target), m.ty);
          derived.push_back(d); shapeClasses[key].push_back(target);
        }
        Derived& d = derived[target];
        for (auto& x : m.xs) { d.xs.push_back(x); for (size_t i = 0; i < m.rest.size(); i++) if (m.intArg[i]) d.consts[i].push_back({x, m.consts[i][0].val}); }
        m.cls = target;
        sub[memberSym[j]] = A.retype(A.asPoly(d.placeholder), m.ty);
        progress = true;
      }
    }
    return sub;
  }
};

// ---------------------------------------------------------------- the affine fit over Phys (+ derived axes)

struct Fitter {
  Arena& A; Type* i64; std::vector<std::string> axisNames;
  size_t n() const { return axisNames.size(); }
  // fits val(x) = base + sum_a stride_a x_a over the physical axes (the first |x| entries of axisNames); verifies
  // every sample. False with why.
  bool fit(const std::vector<Sample>& S, EP& base, std::vector<EP>& stride, std::string& why) {
    if (S.empty()) { why = "no samples"; return false; }
    const size_t nx = S[0].x.size();
    stride.assign(nx, nullptr);
    size_t o = 0;
    for (size_t i = 1; i < S.size(); i++) if (S[i].x < S[o].x) o = i;
    std::vector<bool> have(nx, false);
    for (size_t a = 0; a < nx; a++) {
      for (size_t i = 0; i < S.size() && !have[a]; i++) for (size_t j = 0; j < S.size() && !have[a]; j++) {
        if (i == j) continue;
        int k = 0; bool oneAxis = true;
        for (size_t b = 0; b < nx; b++) { int dlt = S[j].x[b] - S[i].x[b]; if (b == a) k = dlt; else if (dlt) oneAxis = false; }
        if (!oneAxis || k == 0) continue;
        EP diff = A.sub(S[j].val, S[i].val);
        if (k != 1) {
          std::vector<std::pair<std::vector<uint32_t>, int64_t>> terms; bool exact = true;
          for (auto& t : diff->terms) { if (t.second % k) { exact = false; break; } terms.push_back({t.first, t.second / k}); }
          if (!exact) continue;
          diff = A.mkPoly(i64, terms);
        }
        stride[a] = diff; have[a] = true;
      }
    }
    for (size_t a = 0; a < nx; a++) if (!have[a]) {
      bool varies = false; for (auto& s : S) if (s.x[a] != S[o].x[a]) varies = true;
      if (varies) { why = "under-sampled along " + axisNames[a]; return false; }
      stride[a] = nullptr;
    }
    base = S[o].val;
    for (size_t a = 0; a < nx; a++) if (stride[a] && S[o].x[a]) base = A.sub(base, A.mul(stride[a], A.mkInt(i64, S[o].x[a])));
    for (size_t a = 0; a < nx; a++) if (stride[a]) { int64_t z; if (A.constInt(stride[a], z) && z == 0) stride[a] = nullptr; }
    for (auto& s : S) {
      EP pred = base;
      for (size_t a = 0; a < nx; a++) if (stride[a] && s.x[a]) pred = A.add(pred, A.mul(stride[a], A.mkInt(i64, s.x[a])));
      if (pred != s.val) { why = "not affine in Phys at " + pointName(s.x) + ": " + A.str(s.val, 4) + " vs fitted " + A.str(pred, 4); return false; }
    }
    return true;
  }
  std::string pointName(const std::vector<int>& x) const {
    std::string s = "(";
    for (size_t a = 0; a < x.size(); a++) { if (a) s += ","; s += axisNames[a] + "=" + std::to_string(x[a]); }
    return s + ")";
  }
};

// Splits a sample set into affine families: the whole set when it fits; otherwise the stride along every axis is
// taken as the most common difference between samples one step apart along that axis alone, and samples are grouped
// by the residual val - sum stride_a x_a (the family's base). Each group is then fitted (and verified) on its own.
std::vector<std::vector<Sample>> splitFamilies(Arena& A, Fitter& F, const std::vector<Sample>& S) {
  EP base; std::vector<EP> stride; std::string why;
  if (S.size() <= 1 || F.fit(S, base, stride, why)) return {S};
  const size_t n = S[0].x.size();
  Type* i64 = F.i64;
  std::vector<EP> st(n, nullptr);
  for (size_t a = 0; a < n; a++) {
    std::map<EP, unsigned> votes;
    for (size_t i = 0; i < S.size(); i++) for (size_t j = 0; j < S.size(); j++) {
      if (i == j) continue;
      bool one = true;
      for (size_t b = 0; b < n; b++) { int d = S[j].x[b] - S[i].x[b]; if ((b == a && d != 1) || (b != a && d != 0)) { one = false; break; } }
      if (one) votes[A.sub(S[j].val, S[i].val)]++;
    }
    EP best = nullptr; unsigned bestN = 0;
    for (auto& kv : votes) if (kv.second > bestN || (kv.second == bestN && best && kv.first->id < best->id)) { best = kv.first; bestN = kv.second; }
    int64_t z; if (best && A.constInt(best, z) && z == 0) best = nullptr;
    st[a] = best;
  }
  std::map<EP, std::vector<Sample>> groups; std::vector<EP> order;
  for (auto& smp : S) {
    EP res = smp.val;
    for (size_t a = 0; a < n; a++) if (st[a] && smp.x[a]) res = A.sub(res, A.mul(st[a], A.mkInt(i64, smp.x[a])));
    if (!groups.count(res)) order.push_back(res);
    groups[res].push_back(smp);
  }
  std::vector<std::vector<Sample>> out;
  for (EP r : order) out.push_back(groups[r]);
  return out;
}

// ---------------------------------------------------------------- integer Hermite normal form

using Mat = std::vector<std::vector<int64_t>>;

// Row-style HNF: transforms M (r x c) in place into row echelon form with positive pivots and reduced entries above
// them, accumulating the unimodular row transform in U (r x r, starts as the identity). Returns the rank.
unsigned hnf(Mat& M, Mat& U) {
  const size_t r = M.size(), c = r ? M[0].size() : 0;
  U.assign(r, std::vector<int64_t>(r, 0));
  for (size_t i = 0; i < r; i++) U[i][i] = 1;
  auto rowOp = [&](size_t dst, size_t src, int64_t q) {
    if (!q) return;
    for (size_t j = 0; j < c; j++) M[dst][j] -= q * M[src][j];
    for (size_t j = 0; j < r; j++) U[dst][j] -= q * U[src][j];
  };
  auto swapRows = [&](size_t a, size_t b) { if (a != b) { std::swap(M[a], M[b]); std::swap(U[a], U[b]); } };
  auto negRow = [&](size_t a) { for (auto& v : M[a]) v = -v; for (auto& v : U[a]) v = -v; };
  size_t p = 0;
  for (size_t col = 0; col < c && p < r; col++) {
    for (;;) {
      size_t best = r; int64_t bestAbs = 0;
      for (size_t i = p; i < r; i++) if (M[i][col]) { int64_t ab = M[i][col] < 0 ? -M[i][col] : M[i][col]; if (best == r || ab < bestAbs) { best = i; bestAbs = ab; } }
      if (best == r) break;
      swapRows(p, best);
      bool others = false;
      for (size_t i = p + 1; i < r; i++) if (M[i][col]) { rowOp(i, p, M[i][col] / M[p][col]); if (M[i][col]) others = true; }
      if (!others) break;
    }
    if (p < r && M[p][col]) {
      if (M[p][col] < 0) negRow(p);
      for (size_t i = 0; i < p; i++) { int64_t q = M[i][col] / M[p][col]; if (M[i][col] - q * M[p][col] < 0) q--; rowOp(i, p, q); }
      p++;
    }
  }
  return (unsigned)p;
}

Mat inverse(const Mat& U) { Mat M = U, W; hnf(M, W); return W; }

}  // namespace

Discovery discover(Arena& A, const Executor& X, unsigned T) {
  Discovery D;
  Type* i64 = Type::getInt64Ty(A.C);
  // one iteration axis per traversal loop the samples ran in (order of first appearance)
  std::map<const Loop*, unsigned> loopAxis; std::vector<const Loop*> loopsSeen;
  auto seeLoops = [&](const std::vector<const Loop*>& ls) { for (const Loop* L : ls) if (!loopAxis.count(L)) { loopAxis[L] = (unsigned)loopsSeen.size(); loopsSeen.push_back(L); } };
  for (auto& l : X.devLoads) seeLoops(l.loops);
  for (auto& ls : X.devStoreLoops) seeLoops(ls);
  const size_t levels = loopsSeen.size();
  D.axes.push_back({Axis::Simdgroup, 0});
  D.axes.push_back({Axis::Lane, 0});
  for (unsigned l = 0; l < levels; l++) D.axes.push_back({Axis::Iter, l, loopsSeen[l]});
  const size_t nPhys = D.axes.size();
  auto point = [&](unsigned thread, const std::vector<unsigned>& iters, const std::vector<const Loop*>& loops) {
    std::vector<int> x(nPhys, 0);
    x[0] = (int)(thread / 32);
    x[1] = (int)(thread % 32);
    for (size_t l = 0; l < iters.size() && l < loops.size(); l++) x[2 + loopAxis[loops[l]]] = (int)iters[l];
    return x;
  };
  Fitter F{A, i64, {}};
  for (auto& ax : D.axes) {
    if (ax.kind == Axis::Simdgroup) F.axisNames.push_back("sg");
    else if (ax.kind == Axis::Lane) F.axisNames.push_back("lane");
    else if (ax.kind == Axis::Iter) F.axisNames.push_back("i" + std::to_string(ax.index));
    else F.axisNames.push_back("b" + std::to_string(ax.index));
  }
  Abstractor AB{A, i64};

  // gather the samples per site
  struct Rec { unsigned thread; std::vector<unsigned> iters; std::vector<const Loop*> loops; EP addr, cond; };
  struct Raw { Instruction* site; bool isStore, atomic; unsigned bytes; std::vector<Rec> recs; };
  std::map<Instruction*, Raw> raws; std::vector<Instruction*> order;
  auto add = [&](Instruction* site, bool isStore, bool atomic, unsigned bytes, unsigned thread, const std::vector<unsigned>& iters, const std::vector<const Loop*>& loops, EP addr, EP cond) {
    auto it = raws.find(site);
    if (it == raws.end()) { order.push_back(site); it = raws.emplace(site, Raw{site, isStore, atomic, bytes, {}}).first; }
    it->second.recs.push_back({thread, iters, loops, addr, cond});
  };
  for (auto& l : X.devLoads) add(l.site, false, false, l.bytes, l.thread, l.iters, l.loops, l.addr, l.cond);
  for (size_t i = 0; i < X.devStores.size(); i++) { auto& s = X.devStores[i]; add(s.site, true, s.atomic, (unsigned)s.bytes.size(), s.thread, i < X.devStoreIters.size() ? X.devStoreIters[i] : std::vector<unsigned>{}, i < X.devStoreLoops.size() ? X.devStoreLoops[i] : std::vector<const Loop*>{}, s.addr, s.cond); }

  using PredKey = std::pair<unsigned, std::vector<std::vector<uint32_t>>>;
  struct PredFamily { Instruction* site; std::vector<Sample> samples; };
  std::map<PredKey, PredFamily> predFamilies; std::vector<PredKey> predOrder;
  struct Pending { Instruction* site; bool isStore, atomic; unsigned bytes; unsigned piece, pieces; bool viaDerived; std::vector<Sample> S; };
  std::vector<Pending> pending;
  for (Instruction* site : order) {
    Raw& R = raws[site];
    std::map<unsigned, std::vector<Sample>> pieceSamples; std::map<unsigned, bool> pieceDerived;
    unsigned maxPieces = 0; std::string dataWhy;
    for (auto& rec : R.recs) {
      const unsigned thread = rec.thread; EP addr = rec.addr, cond = rec.cond;
      std::vector<std::pair<EP, EP>> leaves; leavesOf(A, addr, A.mkBool(true), leaves, 0);
      if (leaves.size() > 1) {   // a bare pointer argument among the leaves is the clamp target of a guarded read: not an access
        std::vector<std::pair<EP, EP>> kept;
        for (auto& lf : leaves) {
          EP u = A.unwrap(lf.second); std::string nm; uint32_t sid;
          bool bare = u->k == EK::Sym || (u->k == EK::Poly && u->terms.size() == 1 && u->terms[0].first.size() == 1 && u->terms[0].second == 1 && baseOf(A, u, nm, sid));
          if (u->k == EK::Poly && u->terms.size() == 1 && u->terms[0].first.size() == 1) { EP at = A.atomExpr(u->terms[0].first[0]); if (at->k == EK::Call && at->callee && at->callee->getName().contains("zero_page")) bare = true; }
          if (u->k == EK::Call && u->callee && u->callee->getName().contains("zero_page")) bare = true;
          if (!bare) kept.push_back(lf);
        }
        if (!kept.empty()) leaves = kept;
      }
      maxPieces = std::max<unsigned>(maxPieces, (unsigned)leaves.size());
      AB.x = point(thread, rec.iters, rec.loops);
      for (unsigned k = 0; k < leaves.size(); k++) {
        EP leaf = A.retype(A.asPoly(leaves[k].second), i64);
        AB.memo.clear(); std::string why;
        EP sub = AB.go(leaf, why);
        if (!sub) { if (dataWhy.empty()) dataWhy = why; continue; }
        sub = A.retype(A.asPoly(sub), i64);
        if (sub != leaf) pieceDerived[k] = true;
        pieceSamples[k].push_back({AB.x, sub});
      }
      // predicate literals: every comparison literal of the sample's condition, its polynomial side (lhs - rhs) fitted
      // like an address; families keyed by predicate and monomial support (the literal shape independent of the thread)
      std::vector<EP> lits; A.conjuncts(cond, lits);
      for (EP lit : lits) {
        EP u = A.unwrap(lit);
        if (u->k == EK::Op && u->op == Instruction::Xor && u->args.size() == 2) u = A.unwrap(u->args[0]);
        if (u->k != EK::Cmp || u->args.size() != 2) continue;
        EP l = u->args[0], r = u->args[1]; if (!l->ty || !(l->ty->isIntegerTy() || l->ty->isPointerTy())) continue;
        EP lhs = A.retype(A.asPoly(A.sub(A.retype(A.asPoly(l), i64), A.retype(A.asPoly(r), i64))), i64);
        { AB.memo.clear(); std::string why; EP sub = AB.go(lhs, why); if (!sub) continue; lhs = A.retype(A.asPoly(sub), i64); }
        std::vector<std::vector<uint32_t>> support; for (auto& t : lhs->terms) if (!t.first.empty()) support.push_back(t.first);
        PredKey key{u->op, support};
        auto it = predFamilies.find(key);
        if (it == predFamilies.end()) { predOrder.push_back(key); it = predFamilies.emplace(key, PredFamily{site, {}}).first; }
        it->second.samples.push_back({AB.x, lhs});
      }
    }
    if (pieceSamples.empty()) {
      SiteMap m; m.site = site; m.isStore = R.isStore; m.atomic = R.atomic; m.bytes = R.bytes; m.samples = (unsigned)R.recs.size();
      m.why = dataWhy.empty() ? "no readable address (null on every path)" : "data-dependent: the address depends on " + dataWhy;
      D.sites.push_back(m); D.dataDependent++;
      continue;
    }
    for (auto& kv : pieceSamples) pending.push_back({site, R.isStore, R.atomic, R.bytes, kv.first, maxPieces, pieceDerived[kv.first], kv.second});
  }

  // Derived quantities whose operand maps are all constant (and whose uniform parts hold no other derived quantity)
  // are launch parameters the executor had folded a constant into (`N / 128`): folded back into the samples as the
  // original node. The rest become axes: the placeholder of derived j is one more axis; a sample's value over the
  // extended axes is its polynomial with the placeholders kept (their coefficient is the stride along that axis).
  {
    std::map<uint32_t, EP> toClass = AB.classify(); std::map<EP, EP> cmemo;
    if (!toClass.empty()) {
      for (auto& P : pending) for (auto& smp : P.S) smp.val = A.subst(smp.val, toClass, cmemo);
      for (auto& kv : predFamilies) for (auto& smp : kv.second.samples) smp.val = A.subst(smp.val, toClass, cmemo);
    }
  }
  const size_t nDerAll = AB.derived.size();
  {
    std::vector<uint32_t> phSym(nDerAll);
    for (size_t j = 0; j < nDerAll; j++) phSym[j] = A.unwrap(AB.derived[j].placeholder)->sym;
    auto holdsNonUniform = [&](EP e) {
      std::set<uint32_t> found; A.containsSymbol(e, Symbol::LoopValue, &found);
      for (uint32_t sid : found) for (size_t j = 0; j < nDerAll; j++) if (phSym[j] == sid && !AB.derived[j].uniform) return true;
      return false;
    };
    for (bool changed = true; changed;) {
      changed = false;
      for (size_t j = 0; j < nDerAll; j++) {
        Derived& d = AB.derived[j];
        if (d.uniform) continue;
        bool u = true;
        for (size_t i = 0; i < d.rest.size() && u; i++) {
          if (d.intArg[i]) { dedupe(d.consts[i]); std::set<EP> vals; for (auto& smp : d.consts[i]) vals.insert(smp.val); if (vals.size() > 1) u = false; }
          if (u && holdsNonUniform(d.rest[i])) u = false;
        }
        if (u) { d.uniform = true; changed = true; }
      }
    }
    std::map<uint32_t, EP> fold; std::map<EP, EP> fmemo;
    for (size_t j = 0; j < nDerAll; j++) {
      Derived& d = AB.derived[j];
      if (!d.uniform) continue;
      std::vector<EP> args;
      for (size_t i = 0; i < d.rest.size(); i++) {
        EP a = A.subst(d.rest[i], fold, fmemo);
        if (d.intArg[i] && !d.consts[i].empty()) { int64_t c; if (A.constInt(d.consts[i][0].val, c) && c) a = A.add(A.asPoly(a), A.mkPoly(a->ty, {{{}, c}})); }
        args.push_back(a);
      }
      EP node = A.rebuild(d.proto, args);
      fold[phSym[j]] = (node->ty->isIntegerTy() || node->ty->isPointerTy()) ? A.asPoly(node) : node;
    }
    if (!fold.empty()) {
      for (auto& P : pending) for (auto& smp : P.S) smp.val = A.subst(smp.val, fold, fmemo);
      for (auto& kv : predFamilies) for (auto& smp : kv.second.samples) smp.val = A.subst(smp.val, fold, fmemo);
      for (size_t j = 0; j < nDerAll; j++) if (!AB.derived[j].uniform) for (auto& r : AB.derived[j].rest) r = A.subst(r, fold, fmemo);
    }
  }
  std::vector<size_t> derIx;   // non-uniform derived quantities, in axis order
  for (size_t j = 0; j < nDerAll; j++) if (!AB.derived[j].uniform) derIx.push_back(j);
  const size_t nDer = derIx.size();
  for (size_t q = 0; q < nDer; q++) { D.axes.push_back({Axis::Derived, (unsigned)derIx[q]}); F.axisNames.push_back("d" + std::to_string(derIx[q])); }
  const size_t nAx = nPhys + nDer;
  std::vector<uint32_t> derAtom(nDer);
  for (size_t q = 0; q < nDer; q++) derAtom[q] = A.atomOf(A.unwrap(AB.derived[derIx[q]].placeholder));
  // split a polynomial into (placeholder-free part, coefficient poly per derived axis); nullptr when a placeholder
  // appears in a product with a physical-point-dependent or another placeholder factor (not affine in the axes)
  auto splitDerived = [&](EP p, std::vector<EP>& coef) -> EP {
    coef.assign(nDer, nullptr);
    std::vector<std::pair<std::vector<uint32_t>, int64_t>> rest;
    std::vector<std::vector<std::pair<std::vector<uint32_t>, int64_t>>> per(nDer);
    for (auto& t : p->terms) {
      int which = -1; std::vector<uint32_t> others;
      for (uint32_t a : t.first) {
        size_t j = std::find(derAtom.begin(), derAtom.end(), a) - derAtom.begin();
        if (j < nDer) { if (which >= 0) return nullptr; which = (int)j; } else others.push_back(a);
      }
      if (which < 0) rest.push_back(t); else per[which].push_back({others, t.second});
    }
    for (size_t j = 0; j < nDer; j++) if (!per[j].empty()) coef[j] = A.mkPoly(i64, per[j]);
    return A.mkPoly(i64, rest);
  };
  auto fitExtended = [&](std::vector<Sample> S, SiteMap& m) -> bool {
    // the placeholder coefficients must agree across the samples (a stride is a stride everywhere); the remainder is
    // fitted over the physical axes
    std::vector<EP> coef(nDer, nullptr); bool first = true;
    std::vector<std::pair<std::vector<int>, EP>> points;
    for (auto& s : S) {
      points.push_back({s.x, s.val});
      std::vector<EP> c; EP r = splitDerived(s.val, c);
      if (!r) { m.why = "a derived quantity multiplies a physical coordinate in " + A.str(s.val, 4); return false; }
      if (first) { coef = c; first = false; }
      else for (size_t q = 0; q < nDer; q++) if (coef[q] != c[q]) { m.why = "the stride along d" + std::to_string(derIx[q]) + " differs between samples (" + polyStr(A, coef[q]) + " vs " + polyStr(A, c[q]) + ")"; return false; }
      s.val = r;
    }
    std::vector<EP> phys;
    if (!F.fit(S, m.base, phys, m.why)) return false;
    m.points = points;
    m.stride = phys; m.stride.resize(nAx, nullptr);
    for (size_t j = 0; j < nDer; j++) m.stride[nPhys + j] = coef[j];
    return true;
  };
  auto pushSorted = [&](std::vector<SiteMap>& ms) {   // deterministic order within one site's families
    std::stable_sort(ms.begin(), ms.end(), [&](const SiteMap& a, const SiteMap& b) { if (a.fitted() != b.fitted()) return a.fitted(); if (!a.fitted()) return false; std::string sa = polyStr(A, a.base), sb = polyStr(A, b.base); for (size_t j = 0; j < a.stride.size() && j < b.stride.size(); j++) { sa += "|" + polyStr(A, a.stride[j]); sb += "|" + polyStr(A, b.stride[j]); } return sa < sb; });
    for (auto& m : ms) D.sites.push_back(m);
  };
  for (auto& P : pending) {
    dedupe(P.S);
    std::vector<std::vector<Sample>> fams = splitFamilies(A, F, P.S);
    std::vector<SiteMap> ms;
    for (auto& fs : fams) {
      SiteMap m; m.site = P.site; m.isStore = P.isStore; m.atomic = P.atomic; m.bytes = P.bytes; m.samples = (unsigned)fs.size();
      m.piece = P.piece; m.pieces = std::max<unsigned>(P.pieces, (unsigned)fams.size()); m.viaIndex = P.viaDerived;
      if (fitExtended(fs, m)) { baseOf(A, m.base, m.tensor, m.tensorSym); D.fittedSites++; } else D.nonAffine++;
      ms.push_back(m);
    }
    pushSorted(ms);
  }
  for (auto& key : predOrder) {
    PredFamily& P = predFamilies[key];
    dedupe(P.samples);
    std::vector<std::vector<Sample>> fams = splitFamilies(A, F, P.samples);
    std::vector<SiteMap> ms;
    for (auto& fs : fams) {
      SiteMap m; m.site = P.site; m.isPred = true; m.pred = key.first; m.samples = (unsigned)fs.size(); m.pieces = (unsigned)fams.size();
      if (fitExtended(fs, m)) D.fittedSites++; else D.nonAffine++;
      ms.push_back(m);
    }
    pushSorted(ms);
  }
  // the derived quantities' operand maps: d_j = op(rest_i + P_i(x)) with P_i affine in Phys
  for (size_t q = 0; q < nDer; q++) {
    const size_t j = derIx[q];
    Derived& d = AB.derived[j];
    for (size_t i = 0; i < d.consts.size(); i++) {
      if (!d.intArg[i]) continue;
      SiteMap m; m.isDerivedOperand = true; m.derived = (unsigned)j; m.operand = (unsigned)i; m.samples = 0;
      dedupe(d.consts[i]); m.samples = (unsigned)d.consts[i].size();
      std::vector<std::vector<Sample>> fams = splitFamilies(A, F, d.consts[i]);
      std::vector<SiteMap> ms;
      for (auto& fs : fams) {
        SiteMap mm = m; mm.samples = (unsigned)fs.size(); mm.pieces = (unsigned)fams.size();
        if (fitExtended(fs, mm)) { mm.base = A.add(mm.base, A.retype(A.asPoly(d.rest[i]), i64)); D.fittedSites++; } else D.nonAffine++;
        ms.push_back(mm);
      }
      pushSorted(ms);
    }
  }

  // the stride matrix: rows = axes (physical, then derived), columns = (fitted site, monomial)
  struct Col { size_t site; std::vector<uint32_t> mono; };
  std::vector<Col> cols; std::map<std::pair<size_t, std::vector<uint32_t>>, size_t> colIx;
  std::vector<size_t> siteOrder;
  for (int pass = 0; pass < 3; pass++) for (size_t s = 0; s < D.sites.size(); s++) {
    const SiteMap& m = D.sites[s];
    const int cls = m.isStore ? 0 : (m.isPred || m.isDerivedOperand) ? 1 : 2;
    if (cls == pass && m.fitted()) siteOrder.push_back(s);
  }
  for (size_t s : siteOrder) {
    for (size_t a = 0; a < nAx && a < D.sites[s].stride.size(); a++) if (D.sites[s].stride[a]) for (auto& t : D.sites[s].stride[a]->terms) {
      auto key = std::make_pair(s, t.first);
      if (!colIx.count(key)) { colIx[key] = cols.size(); cols.push_back({s, t.first}); }
    }
  }
  Mat S(nAx, std::vector<int64_t>(cols.size(), 0));
  for (size_t s = 0; s < D.sites.size(); s++) {
    if (!D.sites[s].fitted()) continue;
    for (size_t a = 0; a < nAx && a < D.sites[s].stride.size(); a++) if (D.sites[s].stride[a]) for (auto& t : D.sites[s].stride[a]->terms) S[a][colIx[{s, t.first}]] = t.second;
  }
  // Logical d is rank(S on Phys). Derived placeholders are part of φ (a sdiv of the K
  // induction is quasi-affine of r, not a second reduction coordinate of O×D).
  for (size_t a = 0; a < nPhys; a++) { bool any = false; for (auto v : S[a]) if (v) any = true; if (!any) D.replication.push_back((unsigned)a); }
  Mat Sphys(nPhys, std::vector<int64_t>(cols.size(), 0));
  for (size_t a = 0; a < nPhys && a < S.size(); a++) Sphys[a] = S[a];
  Mat H = Sphys, U;
  D.d = cols.empty() || nPhys == 0 ? 0 : hnf(H, U);
  Mat V = cols.empty() || nPhys == 0 ? Mat(nPhys, std::vector<int64_t>(nPhys, 0)) : inverse(U);
  D.V.assign(nAx, std::vector<int64_t>(D.d, 0));
  for (size_t a = 0; a < nPhys; a++) for (unsigned j = 0; j < D.d; j++) D.V[a][j] = V[a][j];
  D.isOutput.assign(D.d, false);
  for (size_t s = 0; s < D.sites.size(); s++) {
    SiteMap& m = D.sites[s];
    if (!m.fitted()) continue;
    m.coordStride.assign(D.d, nullptr);
    for (unsigned j = 0; j < D.d; j++) {
      std::vector<std::pair<std::vector<uint32_t>, int64_t>> terms;
      for (size_t c = 0; c < cols.size(); c++) if (cols[c].site == s && j < H.size() && H[j][c]) terms.push_back({cols[c].mono, H[j][c]});
      if (terms.empty()) continue;
      std::sort(terms.begin(), terms.end());
      m.coordStride[j] = A.mkPoly(i64, terms);
      if (m.isStore) D.isOutput[j] = true;
    }
    for (size_t a = 0; a < nPhys; a++) {
      EP viaCoords = nullptr;
      for (unsigned j = 0; j < D.d; j++) if (m.coordStride[j] && D.V[a][j]) { EP t = A.mul(m.coordStride[j], A.mkInt(i64, D.V[a][j])); viaCoords = viaCoords ? A.add(viaCoords, t) : t; }
      EP direct = a < m.stride.size() ? m.stride[a] : nullptr;
      int64_t z;
      if (viaCoords && A.constInt(viaCoords, z) && z == 0) viaCoords = nullptr;
      if (direct && A.constInt(direct, z) && z == 0) direct = nullptr;
      if (viaCoords != direct) { m.why = "basis change does not reproduce the stride along " + F.axisNames[a] + " (" + polyStr(A, direct) + " vs " + polyStr(A, viaCoords) + ")"; D.fittedSites--; D.nonAffine++; break; }
    }
  }

  D.axisNames = F.axisNames;
  for (size_t q = 0; q < nDer; q++) D.derivedValue.push_back(A.retype(A.asPoly(AB.derived[derIx[q]].placeholder), i64));
  for (size_t q = 0; q < nDer; q++) { Derived& d = AB.derived[derIx[q]]; D.derivedDefs.push_back("d" + std::to_string(derIx[q]) + " = " + A.str(A.rebuild(d.proto, d.rest), 6) + (d.kind == EK::Load ? " (index tensor read)" : "") + "; the integer operands' constant terms are the operand maps"); }
  // ---- report
  unsigned nOut = 0; for (bool b : D.isOutput) if (b) nOut++;
  {
    std::string s = "coords: d=" + std::to_string(D.d) + " (output " + std::to_string(nOut) + ", reduction/traversal " + std::to_string(D.d - nOut) + ") axes=" + std::to_string(nPhys) + " (sg/lane, loop levels " + std::to_string(levels) + "; derived " + std::to_string(nDer) + " in φ)";
    s += " sites fitted=" + std::to_string(D.fittedSites) + " data-dependent=" + std::to_string(D.dataDependent) + " non-affine=" + std::to_string(D.nonAffine);
    if (!D.replication.empty()) { s += " replication="; for (size_t i = 0; i < D.replication.size(); i++) s += (i ? "," : "") + F.axisNames[D.replication[i]]; }
    D.lines.push_back(s);
  }
  for (unsigned j = 0; j < D.d; j++) {
    std::string s = "  c" + std::to_string(j) + (D.isOutput[j] ? " (o) = " : " (r) = ");
    bool first = true;
    for (size_t a = 0; a < nAx; a++) if (D.V[a][j]) { s += (first ? "" : " + ") + (D.V[a][j] == 1 ? "" : std::to_string(D.V[a][j]) + "*") + F.axisNames[a]; first = false; }
    if (first) s += "0";
    D.lines.push_back(s);
  }
  for (size_t q = 0; q < nDer; q++) {
    Derived& d = AB.derived[derIx[q]];
    std::string s = "  d" + std::to_string(derIx[q]) + " = " + A.str(A.rebuild(d.proto, d.rest), 3) + (d.kind == EK::Load ? " (index tensor read)" : "") + "   [operands' constant terms are the maps below]";
    D.lines.push_back(s);
  }
  for (auto& m : D.sites) {
    std::string s;
    if (m.isDerivedOperand) s = "  d" + std::to_string(m.derived) + " operand " + std::to_string(m.operand);
    else s = std::string("  ") + (m.isPred ? "pred at " : m.atomic ? "atomic " : m.isStore ? "store " : "load ") + siteName(m.site);
    if (!m.isPred && !m.isDerivedOperand) { s += " " + std::to_string(m.bytes) + "B"; if (!m.tensor.empty()) s += " in " + m.tensor; if (m.viaIndex) s += " through derived quantities"; if (m.pieces > 1) s += " piece " + std::to_string(m.piece) + " (of " + std::to_string(m.pieces) + ")"; }
    else if (m.isPred) { s += " " + std::string(CmpInst::getPredicateName((CmpInst::Predicate)m.pred)); if (m.pieces > 1) s += " (one of " + std::to_string(m.pieces) + " families)"; }
    else if (m.pieces > 1) s += " (one of " + std::to_string(m.pieces) + " families)";
    s += " samples=" + std::to_string(m.samples);
    if (!m.fitted()) { s += ": " + m.why; D.lines.push_back(s); continue; }
    s += ": " + (m.isPred ? std::string("lhs") : m.isDerivedOperand ? std::string("value") : std::string("addr")) + " = " + polyStr(A, m.base);
    for (unsigned j = 0; j < D.d; j++) if (j < m.coordStride.size() && m.coordStride[j]) s += " + (" + polyStr(A, m.coordStride[j]) + ")*c" + std::to_string(j);
    for (size_t q = 0; q < nDer; q++) if (nPhys + q < m.stride.size() && m.stride[nPhys + q]) s += " + (" + polyStr(A, m.stride[nPhys + q]) + ")*" + F.axisNames[nPhys + q];
    D.lines.push_back(s);
  }
  return D;
}

}  // namespace coords
}  // namespace mvcc
