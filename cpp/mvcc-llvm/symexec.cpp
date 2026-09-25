// Symbolic execution engine (symexec.h).
#include "symexec.h"
#include <chrono>
#include "ptx_asm.h"
#include <limits>
#include <numeric>
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/CFG.h"
#include "llvm/Analysis/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cctype>
#include <functional>
#include <sstream>

namespace mvcc {
namespace sym {

// A condition every thread of the block computes alike: arguments, block index, a skipped loop's unknown, a
// value one thread stored to shared and the rest loaded after a barrier. Per-thread symbols make a barrier or
// an early return data-dependent (not all threads arrive).
static bool uniformAcrossBlock(Arena& A, EP e) {
  if (A.containsSymbol(e, Symbol::ThreadValue)) return false;
  if (A.containsSymbol(e, Symbol::SmemByte)) return false;
  std::set<uint32_t> sregs;
  if (A.containsSymbol(e, Symbol::Sreg, &sregs))
    for (uint32_t id : sregs) {
      StringRef n = A.symbol(id).name;
      if (n.contains("laneid")) return false;
      if (n.contains("tid.") && !n.contains("ntid")) return false;
    }
  return true;
}

// ------------------------------------------------------------------------------------------------ arena

// Hash-consing: a node is identified by its kind, type and payload (the payload fields that its kind uses); the
// table hashes and compares the nodes themselves rather than an encoded key string
static inline uint64_t hmix(uint64_t h, uint64_t v) {
  h ^= v + 0x9E3779B97F4A7C15ull + (h << 6) + (h >> 2);
  h *= 0xBF58476D1CE4E5B9ull;
  return h ^ (h >> 31);
}
size_t Arena::ExprHash::operator()(EP e) const {
  uint64_t h = hmix((uint64_t)e->k, (uint64_t)(uintptr_t)e->ty);
  auto ids = [&](const std::vector<EP>& args) { h = hmix(h, args.size()); for (EP a : args) h = hmix(h, a->id); };
  switch (e->k) {
    case EK::Const: h = hmix(h, e->c.getBitWidth()); for (unsigned w = 0; w < e->c.getNumWords(); w++) h = hmix(h, e->c.getRawData()[w]); break;
    case EK::Undef: break;
    case EK::Poly:
      h = hmix(h, e->terms.size());
      for (auto& t : e->terms) { h = hmix(h, t.first.size()); for (auto a : t.first) h = hmix(h, a); h = hmix(h, (uint64_t)t.second); }
      break;
    case EK::Sym: h = hmix(h, e->sym); break;
    case EK::Op: case EK::Cmp: h = hmix(h, e->op); ids(e->args); break;
    case EK::Ite: case EK::Concat: ids(e->args); break;
    case EK::Load: h = hmix(h, e->len); ids(e->args); break;
    case EK::Bytes: h = hmix(h, e->args[0]->id); h = hmix(h, e->lo); h = hmix(h, e->len); break;
    case EK::Call: h = hmix(h, (uint64_t)(uintptr_t)e->callee); h = hmix(h, (uint64_t)(uintptr_t)e->asmv); ids(e->args); break;
  }
  return (size_t)h;
}
bool Arena::ExprEq::operator()(EP a, EP b) const {
  if (a->k != b->k || a->ty != b->ty) return false;
  auto ids = [&]() {
    if (a->args.size() != b->args.size()) return false;
    for (size_t i = 0; i < a->args.size(); i++) if (a->args[i]->id != b->args[i]->id) return false;
    return true;
  };
  switch (a->k) {
    case EK::Const: return a->c.getBitWidth() == b->c.getBitWidth() && a->c == b->c;
    case EK::Undef: return true;
    case EK::Poly: return a->terms == b->terms;
    case EK::Sym: return a->sym == b->sym;
    case EK::Op: case EK::Cmp: return a->op == b->op && ids();
    case EK::Ite: case EK::Concat: return ids();
    case EK::Load: return a->len == b->len && ids();
    case EK::Bytes: return a->args[0]->id == b->args[0]->id && a->lo == b->lo && a->len == b->len;
    case EK::Call: return a->callee == b->callee && a->asmv == b->asmv && ids();
  }
  return false;
}

EP Arena::intern(Expr&& e) {
  auto it = table.find(&e);
  if (it != table.end()) return *it;
  e.id = (uint32_t)nodes.size();
  nodes.push_back(std::make_unique<Expr>(std::move(e)));
  EP p = nodes.back().get();
  table.insert(p);
  return p;
}

int64_t Arena::wrap(Type* ty, int64_t v) const {
  unsigned w = ty->isPointerTy() ? 64 : ty->getIntegerBitWidth();
  if (w >= 64) return v;
  uint64_t m = (1ull << w) - 1;
  uint64_t u = (uint64_t)v & m;
  if (u & (1ull << (w - 1))) return (int64_t)(u | ~m);  // polynomials are signed integers
  return (int64_t)u;
}

EP Arena::mkInt(Type* ty, int64_t v) { return polyConst(ty, wrap(ty, v)); }
EP Arena::mkConstFP(Type* ty, const APInt& bits) { Expr e; e.k = EK::Const; e.ty = ty; e.c = bits; return intern(std::move(e)); }
EP Arena::mkBool(bool b) { Expr e; e.k = EK::Const; e.ty = Type::getInt1Ty(C); e.c = APInt(1, b ? 1 : 0); return intern(std::move(e)); }
EP Arena::mkUndef(Type* ty) { Expr e; e.k = EK::Undef; e.ty = ty; return intern(std::move(e)); }

EP Arena::mkPoly(Type* ty, std::vector<std::pair<std::vector<uint32_t>, int64_t>> terms) {
  // terms already sorted, distinct and non-zero (the common case: a constant, 1*atom, a rebuilt polynomial) are
  // taken as they are
  bool canonical = true;
  for (size_t i = 0; i < terms.size() && canonical; i++) canonical = terms[i].second != 0 && (i == 0 || terms[i - 1].first < terms[i].first);
  if (!canonical) {
    std::sort(terms.begin(), terms.end(), [](auto& a, auto& b) { return a.first < b.first; });
    std::vector<std::pair<std::vector<uint32_t>, int64_t>> out;
    for (auto& t : terms) {
      if (!out.empty() && out.back().first == t.first) out.back().second += t.second;
      else out.push_back(std::move(t));
    }
    out.erase(std::remove_if(out.begin(), out.end(), [](auto& t) { return t.second == 0; }), out.end());
    terms = std::move(out);
  }
  Expr e; e.k = EK::Poly; e.ty = ty; e.terms = std::move(terms);
  return intern(std::move(e));
}

EP Arena::mkSym(const Symbol& s) {
  EP p = mkSymRaw(s);
  if (p->ty->isIntegerTy() || p->ty->isPointerTy()) return asPoly(p);
  return p;
}

EP Arena::rebuild(EP e, std::vector<EP> args) {
  Expr n = *e; n.args = std::move(args); n.id = 0;
  return intern(std::move(n));
}

EP Arena::mkSymRaw(const Symbol& s) {
  std::string k;
  raw_string_ostream os(k);
  os << (int)s.kind << '|' << (const void*)s.v << '|' << s.name << '|' << s.thread << '|' << s.field;
  uint32_t id;
  auto it = symIds.find(k);
  if (it == symIds.end()) { id = (uint32_t)symbols.size(); symbols.push_back(s); symIds[k] = id; }
  else id = it->second;
  Expr e; e.k = EK::Sym; e.ty = s.ty; e.sym = id;
  return intern(std::move(e));
}

uint32_t Arena::atomOf(EP e) {
  auto it = atomIds.find(e);
  if (it != atomIds.end()) return it->second;
  uint32_t id = (uint32_t)atoms.size();
  atoms.push_back(e);
  atomIds[e] = id;
  return id;
}

bool Arena::isSymAtom(uint32_t atom, uint32_t& symId) const {
  EP e = atoms[atom];
  if (e->k != EK::Sym) return false;
  symId = e->sym;
  return true;
}

EP Arena::asPoly(EP e) {
  if (e->k == EK::Poly) return e;
  if (e->k == EK::Const && e->ty->isIntegerTy()) return polyConst(e->ty, e->c.getSExtValue());
  return mkPoly(e->ty, {{{atomOf(e)}, 1}});
}

EP Arena::retype(EP p, Type* ty) {
  p = asPoly(p);
  if (p->ty == ty) return p;
  auto terms = p->terms;
  for (auto& t : terms) if (t.first.empty()) t.second = wrap(ty, t.second);
  return mkPoly(ty, std::move(terms));
}

bool Arena::constInt(EP e, int64_t& out) const {
  if (e->k == EK::Const && e->ty->isIntegerTy()) { out = e->c.getSExtValue(); return true; }
  if (e->k != EK::Poly) return false;
  if (e->terms.empty()) { out = 0; return true; }
  if (e->terms.size() == 1 && e->terms[0].first.empty()) { out = e->terms[0].second; return true; }
  return false;
}
bool Arena::isTrue(EP e) const { return e->k == EK::Const && e->ty->isIntegerTy(1) && e->c.isOne(); }
bool Arena::isFalse(EP e) const { return e->k == EK::Const && e->ty->isIntegerTy(1) && e->c.isZero(); }

EP Arena::add(EP a, EP b) {
  a = asPoly(a); b = asPoly(b);
  auto terms = a->terms;
  terms.insert(terms.end(), b->terms.begin(), b->terms.end());
  return mkPoly(a->ty, std::move(terms));
}
EP Arena::neg(EP a) {
  a = asPoly(a);
  auto terms = a->terms;
  for (auto& t : terms) t.second = -t.second;
  return mkPoly(a->ty, std::move(terms));
}
EP Arena::sub(EP a, EP b) { return add(a, neg(b)); }
EP Arena::mul(EP a, EP b) {
  a = asPoly(a); b = asPoly(b);
  std::vector<std::pair<std::vector<uint32_t>, int64_t>> terms;
  for (auto& x : a->terms) for (auto& y : b->terms) {
    std::vector<uint32_t> m = x.first;
    m.insert(m.end(), y.first.begin(), y.first.end());
    std::sort(m.begin(), m.end());
    terms.push_back({m, x.second * y.second});
  }
  return mkPoly(a->ty, std::move(terms));
}

EP Arena::unop(unsigned opc, EP a, Type* ty) {
  if (a->k == EK::Const && ty->isFloatingPointTy() && opc == Instruction::FNeg) {
    APFloat x(ty->getFltSemantics(), a->c); x.changeSign(); return mkConstFP(ty, x.bitcastToAPInt());
  }
  Expr e; e.k = EK::Op; e.ty = ty; e.op = opc; e.args = {a};
  return intern(std::move(e));
}

EP Arena::binop(unsigned opc, EP a, EP b, Type* ty) {
  if (ty->isIntegerTy(1)) {
    switch (opc) {
      case Instruction::And: return andb(a, b);
      case Instruction::Or: return orb(a, b);
      case Instruction::Xor: if (isTrue(b)) return notb(a); if (isTrue(a)) return notb(b); if (isFalse(b)) return a; if (isFalse(a)) return b; if (a == b) return mkBool(false); break;
      default: break;
    }
    Expr e; e.k = EK::Op; e.ty = ty; e.op = opc; e.args = {a, b};
    return intern(std::move(e));
  }
  if (ty->isIntegerTy() || ty->isPointerTy()) {
    int64_t ca, cb;
    bool ka = constInt(a, ca), kb = constInt(b, cb);
    if (ka && kb) {
      unsigned w = ty->getIntegerBitWidth();
      APInt x(w, (uint64_t)ca, true), y(w, (uint64_t)cb, true);
      APInt r(w, 0);
      switch (opc) {
        case Instruction::Add: r = x + y; break;
        case Instruction::Sub: r = x - y; break;
        case Instruction::Mul: r = x * y; break;
        case Instruction::And: r = x & y; break;
        case Instruction::Or: r = x | y; break;
        case Instruction::Xor: r = x ^ y; break;
        case Instruction::Shl: r = y.uge(w) ? APInt(w, 0) : x.shl(y); break;
        case Instruction::LShr: r = y.uge(w) ? APInt(w, 0) : x.lshr(y); break;
        case Instruction::AShr: r = y.uge(w) ? x.ashr(w - 1) : x.ashr(y); break;
        case Instruction::UDiv: if (y.isZero()) return mkUndef(ty); r = x.udiv(y); break;
        case Instruction::SDiv: if (y.isZero()) return mkUndef(ty); r = x.sdiv(y); break;
        case Instruction::URem: if (y.isZero()) return mkUndef(ty); r = x.urem(y); break;
        case Instruction::SRem: if (y.isZero()) return mkUndef(ty); r = x.srem(y); break;
        default: goto opaque;
      }
      return mkInt(ty, r.getSExtValue());
    }
    switch (opc) {
      case Instruction::Add: return add(a, b);
      case Instruction::Sub: return sub(a, b);
      case Instruction::Mul: return mul(a, b);
      case Instruction::Shl: if (kb && cb >= 0 && cb < 63) return mul(a, mkInt(ty, (int64_t)1 << cb)); break;
      case Instruction::Or:
        // or of a polynomial with a constant whose bits it cannot have set: only known for concrete values
        break;
      case Instruction::And: if (kb && cb == -1) return asPoly(a); if (kb && cb == 0) return mkInt(ty, 0); break;
      case Instruction::UDiv: case Instruction::SDiv: if (kb && cb == 1) return asPoly(a); break;
      case Instruction::URem: case Instruction::SRem: if (kb && (cb == 1 || cb == -1)) return mkInt(ty, 0); break;
      default: break;
    }
  opaque:
    Expr e; e.k = EK::Op; e.ty = ty; e.op = opc; e.args = {asPoly(a), asPoly(b)};
    return asPoly(intern(std::move(e)));
  }
  // floating point (and vectors): opaque, with constant folding for the common exact cases
  if (a->k == EK::Const && b->k == EK::Const && ty->isFloatingPointTy()) {
    const fltSemantics& sem = ty->getFltSemantics();
    APFloat x(sem, a->c), y(sem, b->c);
    bool ok = true;
    switch (opc) {
      case Instruction::FAdd: x.add(y, APFloat::rmNearestTiesToEven); break;
      case Instruction::FSub: x.subtract(y, APFloat::rmNearestTiesToEven); break;
      case Instruction::FMul: x.multiply(y, APFloat::rmNearestTiesToEven); break;
      default: ok = false;
    }
    if (ok) return mkConstFP(ty, x.bitcastToAPInt());
  }
  Expr e; e.k = EK::Op; e.ty = ty; e.op = opc; e.args = {a, b};
  return intern(std::move(e));
}

EP Arena::cast(unsigned opc, EP a, Type* ty) {
  switch (opc) {
    case Instruction::Trunc: case Instruction::ZExt: case Instruction::SExt:
    case Instruction::PtrToInt: case Instruction::IntToPtr: case Instruction::AddrSpaceCast:
      if (ty->isIntegerTy(1)) { int64_t c; if (constInt(a, c)) return mkBool(c & 1); break; }
      if (a->k == EK::Poly || (a->k == EK::Const && a->ty->isIntegerTy())) {
        if (opc == Instruction::ZExt) {
          // a concrete value zero-extends by its unsigned reading; a symbolic one is assumed in range
          int64_t c;
          if (constInt(a, c)) { unsigned w = a->ty->getIntegerBitWidth(); if (w < 64) c &= (int64_t)((1ull << w) - 1); return mkInt(ty, c); }
        }
        return retype(a, ty);
      }
      if (a->k == EK::Undef) return mkUndef(ty);
      break;
    case Instruction::BitCast:
      if (a->ty == ty) return a;
      if (a->k == EK::Undef) return mkUndef(ty);
      if (a->k == EK::Const && ty->isIntegerTy()) return mkInt(ty, a->c.getSExtValue());
      if (ty->isFloatingPointTy()) { int64_t c; if (constInt(a, c)) return mkConstFP(ty, APInt(ty->getPrimitiveSizeInBits(), (uint64_t)c)); }
      break;
    default: break;
  }
  if (a->k == EK::Const && ty->isFloatingPointTy() && a->ty->isFloatingPointTy() && (opc == Instruction::FPExt || opc == Instruction::FPTrunc)) {
    APFloat x(a->ty->getFltSemantics(), a->c);
    bool lossy;
    x.convert(ty->getFltSemantics(), APFloat::rmNearestTiesToEven, &lossy);
    return mkConstFP(ty, x.bitcastToAPInt());
  }
  Expr e; e.k = EK::Op; e.ty = ty; e.op = opc; e.args = {a};
  EP p = intern(std::move(e));
  return (ty->isIntegerTy() || ty->isPointerTy()) ? asPoly(p) : p;
}

EP Arena::cmp(unsigned pred, EP a, EP b) {
  // integer operands as polynomials (an atom and its 1*atom polynomial must give the same literal)
  if (a->ty->isIntegerTy() && !a->ty->isIntegerTy(1) && a->k != EK::Poly && a->k != EK::Undef) a = asPoly(a);
  if (b->ty->isIntegerTy() && !b->ty->isIntegerTy(1) && b->k != EK::Poly && b->k != EK::Undef) b = asPoly(b);
  if (a->ty->isPointerTy() && a->k != EK::Poly && a->k != EK::Undef) a = asPoly(a);
  if (b->ty->isPointerTy() && b->k != EK::Poly && b->k != EK::Undef) b = asPoly(b);
  int64_t ca, cb;
  if (constInt(a, ca) && constInt(b, cb)) {
    unsigned w = a->ty->isPointerTy() ? 64 : a->ty->getIntegerBitWidth();
    APInt x(w, (uint64_t)ca, true), y(w, (uint64_t)cb, true);
    return mkBool(ICmpInst::compare(x, y, (ICmpInst::Predicate)pred));
  }
  if (a->k == EK::Poly && b->k == EK::Poly) {
    // canonical signed forms: only SLT and SGT, the constant on the left (a <= b  <=>  a-1 < b;  a >= b  <=>  a+1 > b)
    if (pred == CmpInst::ICMP_SLE) return cmp(CmpInst::ICMP_SLT, sub(a, mkInt(a->ty, 1)), b);
    if (pred == CmpInst::ICMP_SGE) return cmp(CmpInst::ICMP_SGT, add(a, mkInt(a->ty, 1)), b);
    if ((pred == CmpInst::ICMP_SLT || pred == CmpInst::ICMP_SGT) && !a->ty->isPointerTy()) {
      // canonical: P > 0 or P < 0 with P's non-constant coefficients coprime and its leading one positive, so that
      // the same condition written at different scales (a loop bound compared in elements and in bytes) is one literal
      EP d = pred == CmpInst::ICMP_SLT ? sub(b, a) : sub(a, b);  // condition <=> 0 < d
      int64_t dc;
      if (constInt(d, dc)) return mkBool(dc > 0);
      if (d->k == EK::Poly) {
        int64_t g = 0, c = 0; bool lead = false, leadNeg = false;
        for (auto& t : d->terms) { if (t.first.empty()) { c = t.second; continue; } g = g ? std::gcd(g, std::llabs(t.second)) : std::llabs(t.second); if (!lead) { lead = true; leadNeg = t.second < 0; } }
        if (g) {
          std::vector<std::pair<std::vector<uint32_t>, int64_t>> et;
          for (auto& t : d->terms) if (!t.first.empty()) et.push_back({t.first, (leadNeg ? -t.second : t.second) / g});
          auto fdiv = [](int64_t x, int64_t y) { int64_t q = x / y; if ((x % y != 0) && ((x < 0) != (y < 0))) q--; return q; };
          auto cdiv = [&](int64_t x, int64_t y) { return -fdiv(-x, y); };
          Expr e; e.k = EK::Cmp; e.ty = Type::getInt1Ty(C);
          if (!leadNeg) { et.push_back({{}, -fdiv(-c, g)}); e.op = CmpInst::ICMP_SGT; }   // e > floor(-c/g)  <=>  e - floor(-c/g) > 0
          else { et.push_back({{}, -cdiv(c, g)}); e.op = CmpInst::ICMP_SLT; }           // e' < ceil(c/g)   <=>  e' - ceil(c/g) < 0
          if (et.back().second == 0) et.pop_back();
          e.args = {mkPoly(d->ty, std::move(et)), mkInt(d->ty, 0)};
          return intern(std::move(e));
        }
      }
    }
    // the constant term lives on the left (equality and signed orders only: an unsigned order does not survive a
    // shift, (unsigned)(x - 1) < 16 is the range test 1 <= x <= 16, not x < 17; it keeps its constant on the right)
    if (pred == CmpInst::ICMP_EQ || pred == CmpInst::ICMP_NE || CmpInst::isSigned((CmpInst::Predicate)pred))
      for (auto& t : b->terms) if (t.first.empty() && t.second != 0) { EP c = mkInt(b->ty, t.second); return cmp(pred, sub(a, c), sub(b, c)); }
  }
  if (a == b && (a->k == EK::Poly)) {
    switch (pred) {
      case CmpInst::ICMP_EQ: case CmpInst::ICMP_ULE: case CmpInst::ICMP_SLE: case CmpInst::ICMP_UGE: case CmpInst::ICMP_SGE: return mkBool(true);
      case CmpInst::ICMP_NE: case CmpInst::ICMP_ULT: case CmpInst::ICMP_SLT: case CmpInst::ICMP_UGT: case CmpInst::ICMP_SGT: return mkBool(false);
      default: break;
    }
  }
  Expr e; e.k = EK::Cmp; e.ty = Type::getInt1Ty(C); e.op = pred; e.args = {a, b};
  return intern(std::move(e));
}

static bool canonicalPred(unsigned p) {
  switch (p) {
    case CmpInst::ICMP_NE: case CmpInst::ICMP_SGE: case CmpInst::ICMP_UGE: case CmpInst::ICMP_SGT: case CmpInst::ICMP_UGT:
    case CmpInst::FCMP_ONE: case CmpInst::FCMP_OGE: case CmpInst::FCMP_OGT: case CmpInst::FCMP_UNE: case CmpInst::FCMP_UGE: case CmpInst::FCMP_UGT:
      return false;
    default: return true;
  }
}

EP Arena::ite(EP c, EP a, EP b) {
  if (isTrue(c)) return a;
  if (isFalse(c)) return b;
  if (a == b) return a;
  if (c->k == EK::Cmp && !canonicalPred(c->op)) return ite(notb(c), b, a);
  if (c->k == EK::Op && c->op == Instruction::Xor && c->args.size() == 2 && isTrue(c->args[1])) return ite(c->args[0], b, a);
  if (a->k == EK::Undef) return b;
  if (b->k == EK::Undef) return a;
  if (a->k == EK::Ite && a->args[0] == c) a = a->args[1];
  if (b->k == EK::Ite && b->args[0] == c) b = b->args[2];
  if (a == b) return a;
  {
    // Hoist same-size bitcasts out of both arms: ite(c, bitcast(x), bitcast(y)) = bitcast(ite(c, x, y)).
    // Keeps values in their natural (float) type so byte fusion across integer and float stores of the same
    // element agrees.
    EP ua = unwrap(a), ub = unwrap(b);
    auto isBC = [&](EP v) { return v->k == EK::Op && v->op == Instruction::BitCast && sizeOf(v->args[0]->ty) == sizeOf(v->ty) && !v->args[0]->ty->isIntegerTy(); };
    if (isBC(ua) && isBC(ub) && ua->args[0]->ty == ub->args[0]->ty)
      return cast(Instruction::BitCast, ite(c, ua->args[0], ub->args[0]), a->ty);
  }
  if (a->ty->isIntegerTy(1)) {
    if (isTrue(a) && isFalse(b)) return c;
    if (isFalse(a) && isTrue(b)) return notb(c);
    if (isTrue(a)) return orb(c, b);
    if (isFalse(a)) return andb(notb(c), b);
    if (isTrue(b)) return orb(notb(c), a);
    if (isFalse(b)) return andb(c, a);
  }
  Expr e; e.k = EK::Ite; e.ty = a->ty; e.args = {c, a, b};
  return intern(std::move(e));
}

EP Arena::notb(EP a) {
  if (isTrue(a)) return mkBool(false);
  if (isFalse(a)) return mkBool(true);
  if (a->k == EK::Op && a->op == Instruction::Xor && a->args.size() == 2 && isTrue(a->args[1])) return a->args[0];
  if (notbMemo.size() <= a->id) notbMemo.resize(nodes.size(), nullptr);
  if (EP m = notbMemo[a->id]) return m;
  EP r;
  if (a->k == EK::Cmp) r = cmp(CmpInst::getInversePredicate((CmpInst::Predicate)a->op), a->args[0], a->args[1]);
  else if (a->k == EK::Op && a->op == Instruction::And && a->ty->isIntegerTy(1)) r = orb(notb(a->args[0]), notb(a->args[1]));
  else if (a->k == EK::Op && a->op == Instruction::Or && a->ty->isIntegerTy(1)) r = andb(notb(a->args[0]), notb(a->args[1]));
  else { Expr e; e.k = EK::Op; e.ty = a->ty; e.op = Instruction::Xor; e.args = {a, mkBool(true)}; r = intern(std::move(e)); }
  return notbMemo[a->id] = r;
}
EP Arena::andb(EP a, EP b) {
  if (isTrue(a)) return b;
  if (isTrue(b)) return a;
  if (isFalse(a) || isFalse(b)) return mkBool(false);
  if (a == b) return a;
  auto it = andMemo.find({a, b});
  if (it != andMemo.end()) return it->second;
  EP r = andbRaw(a, b);
  andMemo[{a, b}] = r;
  return r;
}
EP Arena::andbRaw(EP a, EP b) {
  if (implies(a, b)) return a;
  if (implies(b, a)) return b;
  if (disjoint(a, b)) return mkBool(false);
  {
    // conjunction of literals with implied members dropped, in a canonical order
    std::vector<EP> ls; conjuncts(a, ls); conjuncts(b, ls);
    bool isOr = false;
    for (EP l : ls) if (l->k == EK::Op && l->op == Instruction::Or && l->ty->isIntegerTy(1)) isOr = true;
    if (!isOr) {
      std::vector<EP> keep;
      for (size_t i = 0; i < ls.size(); i++) {
        bool drop = false;
        for (size_t j = 0; j < ls.size() && !drop; j++) if (i != j && litImplies(ls[j], ls[i]) && !(litImplies(ls[i], ls[j]) && j > i)) drop = true;
        if (!drop) keep.push_back(ls[i]);
      }
      std::sort(keep.begin(), keep.end(), [](EP x, EP y) { return x->id < y->id; });
      keep.erase(std::unique(keep.begin(), keep.end()), keep.end());
      if (keep.size() == 1) return keep[0];
      EP r = keep[0];
      for (size_t i = 1; i < keep.size(); i++) { Expr e; e.k = EK::Op; e.ty = a->ty; e.op = Instruction::And; e.args = {r, keep[i]}; r = intern(std::move(e)); }
      return r;
    }
  }
  // disjunctive normal form: conditions are kept as disjunctions of conjunctions of literals
  if (b->k == EK::Op && b->op == Instruction::Or && b->ty->isIntegerTy(1)) return orb(andb(a, b->args[0]), andb(a, b->args[1]));
  if (a->k == EK::Op && a->op == Instruction::Or && a->ty->isIntegerTy(1)) return orb(andb(a->args[0], b), andb(a->args[1], b));
  Expr e; e.k = EK::Op; e.ty = a->ty; e.op = Instruction::And; e.args = {a, b};
  return intern(std::move(e));
}
static bool isGuardSym(const Arena& A, EP l);
// Literals are ordered by expression id, never by address: the resolution below works on sorted clauses, and the
// order it visits them in decides which resolutions happen first (and so the guards a join leaves); by address the
// result would differ from run to run.
static bool byId(EP x, EP y) { return x->id < y->id; }
static unsigned gGuardExpandMax = 0;
void Arena::noteGuardExpandDepth(unsigned d) { gGuardExpandMax = std::max(gGuardExpandMax, d); }
unsigned takeGuardExpandMax() { unsigned m = gGuardExpandMax; gGuardExpandMax = 0; return m; }
EP Arena::orb(EP a, EP b) {
  if (isFalse(a)) return b;
  if (isFalse(b)) return a;
  if (isTrue(a) || isTrue(b)) return mkBool(true);
  if (a == b) return a;
  auto it = orMemo.find({a, b});
  if (it != orMemo.end()) return it->second;
  EP r = orbRaw(a, b);
  orMemo[{a, b}] = r;
  return r;
}
EP Arena::orbRaw(EP a, EP b) {
  if (implies(a, b)) return b;
  if (implies(b, a)) return a;
  // disjunction of conjunctions: resolve clauses that differ in one complementary literal (the reaching condition of
  // a join is the condition of its dominator), drop clauses implied by others
  std::vector<std::vector<EP>> clauses;
  // Every clause enters in the form andb builds from its literals (implied literals dropped: the row guard a path
  // already established is not repeated), so that the resolution below sees the literals a clause really carries;
  // otherwise `(X and not-s) or (X and s and r)`, with X implying r, resolves to `(X and not-s) or (X and r)` and
  // rebuilds as `(X and not-s) or X`: a redundant clause that is never dropped, and the disjunction the guard
  // factoring of every join then nests (each guard defined in terms of the last), until the implication checks are
  // exponential in the number of joins executed
  // A guard literal with a shallow definition (a few clauses of plain literals: the factored condition of the last
  // join, `no stage left or row out of range`) is expanded in place, so that the join that completes its case split
  // (`... or (stage left and row in range)`) resolves to the dominator's condition; what does not resolve is
  // factored again by the caller. Deep definitions stay opaque: expanding them would multiply the clauses.
  auto shallowGuard = [&](EP l) -> const EP* {
    if (!isGuardSym(*this, l)) return nullptr;
    const EP& def = guardDefs[symbols[l->sym].field];
    unsigned n = 0; bool plain = true;
    std::function<void(EP)> scan = [&](EP c) {
      if (c->k == EK::Op && c->op == Instruction::Or && c->ty->isIntegerTy(1)) { scan(c->args[0]); scan(c->args[1]); return; }
      n++; std::vector<EP> ls; conjuncts(c, ls);
      for (EP x : ls) if (isGuardSym(*this, x)) plain = false;
    };
    scan(def);
    return plain && n <= 4 ? &def : nullptr;
  };
  std::function<void(EP, bool)> flat = [&](EP c, bool canon) {
    if (c->k == EK::Op && c->op == Instruction::Or && c->ty->isIntegerTy(1)) { flat(c->args[0], canon); flat(c->args[1], canon); return; }
    if (isFalse(c)) return;
    std::vector<EP> ls; conjuncts(c, ls);
    if (canon) {
      EP cc = mkBool(true);
      for (EP l : ls) cc = andb(cc, l);
      if (cc != c) { flat(cc, false); return; }
    }
    if (isTrue(c)) { clauses.clear(); clauses.push_back({}); return; }
    if (clauses.size() < 64)
      for (size_t i = 0; i < ls.size(); i++)
        if (const EP* def = shallowGuard(ls[i])) {
          EP rest = mkBool(true);
          for (size_t j = 0; j < ls.size(); j++) if (j != i) rest = andb(rest, ls[j]);
          std::vector<EP> ds; std::function<void(EP)> fd = [&](EP d) { if (d->k == EK::Op && d->op == Instruction::Or && d->ty->isIntegerTy(1)) { fd(d->args[0]); fd(d->args[1]); return; } ds.push_back(d); };
          fd(*def);
          for (EP d : ds) flat(andb(rest, d), true);
          return;
        }
    std::sort(ls.begin(), ls.end(), byId); clauses.push_back(ls);
  };
  flat(a, true); flat(b, true);
  for (auto& cl : clauses) if (cl.empty()) return mkBool(true);
  auto complementary = [&](EP x, EP y) {
    if (notb(x) == y) return true;
    return x->k == EK::Cmp && y->k == EK::Cmp && x->args == y->args && x->op == (unsigned)CmpInst::getInversePredicate((CmpInst::Predicate)y->op);
  };
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t i = 0; i < clauses.size() && !changed; i++)
      for (size_t j = i + 1; j < clauses.size() && !changed; j++) {
        auto& ci = clauses[i]; auto& cj = clauses[j];
        std::vector<EP> di, dj;
        std::set_difference(ci.begin(), ci.end(), cj.begin(), cj.end(), std::back_inserter(di), byId);
        std::set_difference(cj.begin(), cj.end(), ci.begin(), ci.end(), std::back_inserter(dj), byId);
        if (di.empty()) { clauses.erase(clauses.begin() + j); changed = true; break; }   // ci implies... cj implies ci: cj subsumed
        if (dj.empty()) { clauses.erase(clauses.begin() + i); changed = true; break; }
        // subsumption up to the order of the bounds: cj => ci when every literal ci has is implied by one of cj's
        // (kt < 1 is subsumed by kt < 2: the prologue's "stage s is loaded when s < ktiles" chain)
        auto subsumed = [&](const std::vector<EP>& by, const std::vector<EP>& cl) {
          for (EP m : by) { bool ok = false; for (EP l : cl) if (litImplies(l, m)) { ok = true; break; } if (!ok) return false; }
          return true;
        };
        if (semantic && subsumed(ci, cj)) { clauses.erase(clauses.begin() + j); changed = true; break; }
        if (semantic && subsumed(cj, ci)) { clauses.erase(clauses.begin() + i); changed = true; break; }
        // one differing literal each whose disjunction is a tautology (complementary, or bounds on one polynomial
        // that together cover it: `M < 10 or M > 1`): the common part
        auto covering = [&](EP x, EP y) { return complementary(x, y) || litImplies(notb(x), y) || litImplies(notb(y), x); };
        if (di.size() == 1 && dj.size() == 1 && covering(di[0], dj[0])) {
          std::vector<EP> common;
          std::set_intersection(ci.begin(), ci.end(), cj.begin(), cj.end(), std::back_inserter(common), byId);
          clauses.erase(clauses.begin() + j); clauses[i] = common; changed = true;
          break;
        }
        // (X and l) or (X and not-l and R) = (X and l) or (X and R): the literal dropped is one of the other clause's
        // own (in dj, not in the common part X - `(X' and y and l) or (X' and y and R)` with `not-l => y` keeps y in
        // the second clause: where l holds and R does but y does not, the original is false)
        auto own = [&](const std::vector<EP>& d, EP l) { return std::binary_search(d.begin(), d.end(), l, byId); };
        // A covering literal in the common part goes too when the rest of its clause implies it (it is redundant
        // there: `stage < stages` next to `stage + 1 < stages`): the same drop under a proof instead of by position.
        auto redundant = [&](const std::vector<EP>& cl, size_t k) {
          EP rest = mkBool(true);
          for (size_t m = 0; m < cl.size(); m++) if (m != k) rest = andb(rest, cl[m]);
          return implies(rest, cl[k]);
        };
        if (di.size() == 1) for (size_t k = 0; k < cj.size(); k++) if (covering(di[0], cj[k]) && (own(dj, cj[k]) || redundant(cj, k))) { cj.erase(cj.begin() + k); changed = true; break; }
        if (!changed && dj.size() == 1) for (size_t k = 0; k < ci.size(); k++) if (covering(dj[0], ci[k]) && (own(di, ci[k]) || redundant(ci, k))) { ci.erase(ci.begin() + k); changed = true; break; }
      }
  }
  EP r = nullptr;
  bool rebuilt = false;   // a resolved clause andb simplified further: the clause set may have new redundancy
  for (auto& cl : clauses) {
    EP c = mkBool(true);
    for (EP l : cl) c = andb(c, l);
    if (isTrue(c)) return c;
    if (isFalse(c)) { rebuilt = true; continue; }
    { std::vector<EP> ls; conjuncts(c, ls); if (ls.size() != cl.size()) rebuilt = true; }
    if (!r) { r = c; continue; }
    Expr e; e.k = EK::Op; e.ty = a->ty; e.op = Instruction::Or; e.args = {r, c};
    r = intern(std::move(e));
  }
  if (!r) return mkBool(false);
  if (rebuilt && orbDepth < 4 && r->k == EK::Op && r->op == Instruction::Or && r->ty->isIntegerTy(1)) { orbDepth++; r = orb(r->args[0], r->args[1]); orbDepth--; }
  // When the expansion above leaves clauses, the result is a disjunction where the caller's operands carried a
  // guard: `(X and C) or (Y and C)` at a block joined by `g and C and s` / `g and C and not-s`. It is returned as
  // is - re-factoring it here (one guard for X or Y) deepens the guard chains of every later join, and the fp8
  // decode kernels' implication checks then either run out of budget or explode. A pass that reads a block's
  // literals factors the disjunction itself (the mainloop's K bound: `factor`).
  return r;
}

EP Arena::factor(EP c) {
  if (!(c->k == EK::Op && c->op == Instruction::Or && c->ty->isIntegerTy(1))) return c;
  std::vector<std::vector<EP>> clauses;
  std::function<void(EP)> flat = [&](EP x) {
    if (x->k == EK::Op && x->op == Instruction::Or && x->ty->isIntegerTy(1)) { flat(x->args[0]); flat(x->args[1]); return; }
    std::vector<EP> ls; conjuncts(x, ls); std::sort(ls.begin(), ls.end(), byId); clauses.push_back(ls);
  };
  flat(c);
  std::vector<EP> common = clauses[0];
  for (size_t i = 1; i < clauses.size(); i++) { std::vector<EP> k; std::set_intersection(common.begin(), common.end(), clauses[i].begin(), clauses[i].end(), std::back_inserter(k), byId); common = std::move(k); }
  EP rest = nullptr;
  for (auto& cl : clauses) {
    std::vector<EP> d; std::set_difference(cl.begin(), cl.end(), common.begin(), common.end(), std::back_inserter(d), byId);
    EP x = mkBool(true); for (EP l : d) x = andb(x, l);
    if (isTrue(x)) { rest = x; break; }   // a clause reduced to the common part: the disjunction is the common part
    if (!rest) { rest = x; continue; }
    Expr e; e.k = EK::Op; e.ty = c->ty; e.op = Instruction::Or; e.args = {rest, x}; rest = intern(std::move(e));
  }
  EP g;
  if (isTrue(rest)) g = rest;
  else {
    auto it = guardOf.find(rest);
    if (it != guardOf.end()) g = it->second;
    else {
      Symbol s; s.kind = Symbol::Guard; s.name = "guard"; s.field = (unsigned)guardDefs.size(); s.ty = c->ty;
      guardDefs.push_back(rest);
      g = mkSymRaw(s);
      guardOf[rest] = g;
    }
  }
  EP r = g; for (EP l : common) r = andb(r, l);
  return r;
}

EP Arena::load(Type* ty, EP addr, EP guard, unsigned align) { Expr e; e.k = EK::Load; e.ty = ty; e.args = {addr, guard}; e.len = align; EP p = intern(std::move(e)); return (ty->isIntegerTy() || ty->isPointerTy()) ? asPoly(p) : p; }
EP Arena::call(Function* callee, std::vector<EP> args, Type* ty) { Expr e; e.k = EK::Call; e.ty = ty; e.callee = callee; e.args = std::move(args); EP p = intern(std::move(e)); return (ty->isIntegerTy() || ty->isPointerTy()) ? asPoly(p) : p; }
EP Arena::callAsm(Value* asmv, std::vector<EP> args, Type* ty) { Expr e; e.k = EK::Call; e.ty = ty; e.asmv = asmv; e.args = std::move(args); EP p = intern(std::move(e)); return (ty->isIntegerTy() || ty->isPointerTy()) ? asPoly(p) : p; }
EP Arena::concat(std::vector<EP> parts, Type* ty) {
  if (parts.size() == 1 && parts[0]->ty == ty) return parts[0];
  Expr e; e.k = EK::Concat; e.ty = ty; e.args = std::move(parts);
  return intern(std::move(e));
}
EP Arena::bytes(EP x, unsigned lo, unsigned len) {
  if (lo == 0 && len == sizeOf(x->ty)) return x;
  auto key = std::make_tuple(x, lo, len);
  auto it = bytesMemo.find(key);
  if (it != bytesMemo.end()) return it->second;
  EP r = bytesRaw(x, lo, len);
  bytesMemo[key] = r;
  return r;
}

EP Arena::bytesRaw(EP x, unsigned lo, unsigned len) {
  unsigned sz = sizeOf(x->ty);
  if (lo == 0 && len == sz) return x;
  if (x->k == EK::Undef) return mkUndef(Type::getIntNTy(C, 8 * len));
  if (x->k == EK::Bytes) return bytes(x->args[0], x->lo + lo, len);
  if (x->k == EK::Op && x->op == Instruction::BitCast && sizeOf(x->args[0]->ty) == sz) return bytes(x->args[0], lo, len);
  if (x->k == EK::Ite) return ite(x->args[0], bytes(x->args[1], lo, len), bytes(x->args[2], lo, len));
  if (x->k == EK::Poly && x->terms.size() == 1 && x->terms[0].second == 1 && x->terms[0].first.size() == 1 && sizeOf(atoms[x->terms[0].first[0]]->ty) == sz) return bytes(atoms[x->terms[0].first[0]], lo, len);
  if (x->k == EK::Concat) {
    unsigned off = 0;
    for (EP part : x->args) {
      unsigned psz = sizeOf(part->ty);
      if (lo >= off && lo + len <= off + psz) return bytes(part, lo - off, len);
      off += psz;
    }
    // straddles parts: assemble from bytes
    std::vector<EP> bs = toBytes(x);
    std::vector<EP> sel(bs.begin() + lo, bs.begin() + lo + len);
    return fromBytes(sel, Type::getIntNTy(C, 8 * len));
  }
  int64_t c;
  if (constInt(x, c)) { uint64_t u = (uint64_t)c >> (8 * lo); return mkInt(Type::getIntNTy(C, 8 * len), (int64_t)(len >= 8 ? u : (u & ((1ull << (8 * len)) - 1)))); }
  if (x->k == EK::Const) { APInt v = x->c.lshr(8 * lo).trunc(8 * len); return mkInt(Type::getIntNTy(C, 8 * len), v.getSExtValue()); }
  Expr e; e.k = EK::Bytes; e.ty = Type::getIntNTy(C, 8 * len); e.args = {x}; e.lo = lo; e.len = len;
  return intern(std::move(e));
}

std::vector<EP> Arena::toBytes(EP v) {
  auto it = byteMemo.find(v);
  if (it != byteMemo.end()) return it->second;
  std::vector<EP> out;
  unsigned sz = sizeOf(v->ty);
  if (v->k == EK::Concat) { for (EP p : v->args) { auto b = toBytes(p); out.insert(out.end(), b.begin(), b.end()); } }
  else for (unsigned i = 0; i < sz; i++) out.push_back(bytes(v, i, 1));
  byteMemo[v] = out;
  return out;
}

EP Arena::fromBytes(const std::vector<EP>& bs, Type* ty) {
  // merge runs of consecutive bytes of one source
  std::vector<EP> parts;
  size_t i = 0;
  while (i < bs.size()) {
    EP b = bs[i];
    if (b->k == EK::Bytes) {
      EP x = b->args[0]; unsigned lo = b->lo; size_t j = i + 1;
      while (j < bs.size() && bs[j]->k == EK::Bytes && bs[j]->args[0] == x && bs[j]->lo == lo + (j - i)) j++;
      parts.push_back(bytes(x, lo, (unsigned)(j - i)));
      i = j;
      continue;
    }
    int64_t c;
    if (constInt(b, c)) {
      uint64_t v = 0; size_t j = i;
      while (j < bs.size() && j - i < 8 && constInt(bs[j], c)) { v |= ((uint64_t)c & 0xff) << (8 * (j - i)); j++; }
      parts.push_back(mkInt(Type::getIntNTy(C, 8 * (unsigned)(j - i)), (int64_t)v));
      i = j;
      continue;
    }
    if (b->k == EK::Undef) {
      size_t j = i; while (j < bs.size() && bs[j]->k == EK::Undef) j++;
      parts.push_back(mkUndef(Type::getIntNTy(C, 8 * (unsigned)(j - i))));
      i = j;
      continue;
    }
    if (b->k == EK::Ite) {
      // consecutive bytes selected by one condition whose arms are consecutive bytes of one value each: the longest
      // such run is one value under the condition (a shared-memory byte resolved from alternative writes)
      EP c = b->args[0];
      size_t bestJ = 0; EP best = nullptr;
      for (size_t j = i + 1; j <= bs.size(); j++) {
        if (bs[j - 1]->k != EK::Ite || bs[j - 1]->args[0] != c) break;
        std::vector<EP> ts, fs;
        for (size_t k = i; k < j; k++) { ts.push_back(bs[k]->args[1]); fs.push_back(bs[k]->args[2]); }
        Type* it = Type::getIntNTy(C, 8 * (unsigned)(j - i));
        EP t = fromBytes(ts, it), f = fromBytes(fs, it);
        auto strip = [&](EP v) { v = unwrap(v); if (v->k == EK::Op && v->op == Instruction::BitCast && sizeOf(v->args[0]->ty) == sizeOf(v->ty)) v = v->args[0]; return v; };
        t = strip(t); f = strip(f);
        if (t->k != EK::Concat && f->k != EK::Concat) {
          if (t->ty != f->ty) { t = cast(Instruction::BitCast, t, it); f = cast(Instruction::BitCast, f, it); }
          best = ite(c, t, f); bestJ = j;
        }
      }
      if (best && bestJ > i + 1) { parts.push_back(best); i = bestJ; continue; }
    }
    parts.push_back(b); i++;
  }
  if (parts.size() == 1) {
    EP p = parts[0];
    if (p->ty == ty) return p;
    if (sizeOf(p->ty) == sizeOf(ty)) return cast(Instruction::BitCast, p, ty);
  }
  return concat(std::move(parts), ty);
}

void Arena::conjuncts(EP c, std::vector<EP>& out) const {
  if (c->k == EK::Op && c->op == Instruction::And && c->ty->isIntegerTy(1)) { conjuncts(c->args[0], out); conjuncts(c->args[1], out); return; }
  if (!isTrue(c)) out.push_back(c);
}
// x => y for single literals: identical, or the same signed comparison against the same right-hand side with x's
// left-hand side larger (SLT: x's constant >= y's is weaker... x = p+c1 < q implies p+c2 < q when c2 <= c1)
bool Arena::litImplies(EP x, EP y) const {
  if (x == y) return true;
  if (x->k != EK::Cmp || y->k != EK::Cmp || x->args[1] != y->args[1]) return false;
  auto lower = [](unsigned p) { return p == CmpInst::ICMP_SGT || p == CmpInst::ICMP_SGE; };
  auto upper = [](unsigned p) { return p == CmpInst::ICMP_SLT || p == CmpInst::ICMP_SLE; };
  if (!((lower(x->op) && lower(y->op)) || (upper(x->op) && upper(y->op)))) {
    // a bound deciding an (in)equality on the same polynomial: kt > 1 implies kt != 1
    bool out; return semantic && (y->op == CmpInst::ICMP_NE || y->op == CmpInst::ICMP_EQ) && decideLit(y, {x}, out) && out;
  }
  if (x->args[0]->k != EK::Poly || y->args[0]->k != EK::Poly || x->args[0]->ty != y->args[0]->ty) return false;
  // difference of the left-hand sides must be a constant
  auto& tx = x->args[0]->terms; auto& ty = y->args[0]->terms;
  int64_t cx = 0, cy = 0;
  size_t ix = 0, iy = 0;
  while (ix < tx.size() || iy < ty.size()) {
    if (ix < tx.size() && tx[ix].first.empty()) { cx = tx[ix].second; ix++; continue; }
    if (iy < ty.size() && ty[iy].first.empty()) { cy = ty[iy].second; iy++; continue; }
    if (ix >= tx.size() || iy >= ty.size() || tx[ix] != ty[iy]) return false;
    ix++; iy++;
  }
  // as bounds on p relative to q: p + c > q is p >= q - c + 1, p + c >= q is p >= q - c (and the mirror images
  // above); x implies y when x's bound is at least as tight
  if (lower(x->op)) { int64_t lx = -cx + (x->op == CmpInst::ICMP_SGT), ly = -cy + (y->op == CmpInst::ICMP_SGT); return lx >= ly; }
  int64_t ux = -cx - (x->op == CmpInst::ICMP_SLT), uy = -cy - (y->op == CmpInst::ICMP_SLT); return ux <= uy;
}

bool Arena::decideLit(EP l, const std::vector<EP>& facts, bool& out) const {
  if (l->k != EK::Cmp) return false;
  { int64_t z; if (constInt(l->args[1], z) && z == 0) { if (l->op == CmpInst::ICMP_UGE) { out = true; return true; } if (l->op == CmpInst::ICMP_ULT) { out = false; return true; } } }
  if (l->args[0]->k != EK::Poly) return false;
  // (polynomial without its constant, constant) of a literal's left-hand side
  auto split = [](EP lhs, int64_t& c) { std::vector<std::pair<std::vector<uint32_t>, int64_t>> p; c = 0; for (auto& t : lhs->terms) { if (t.first.empty()) c = t.second; else p.push_back(t); } return p; };
  int64_t cl; auto pl = split(l->args[0], cl);
  // an unsigned order against a constant is a range test on the signed value: (unsigned)(P + cl) < C is
  // 0 <= P + cl < C; its bounds come from the signed facts on P (which compare against 0)
  int64_t uC = 0; const bool isUnsigned = CmpInst::isUnsigned((CmpInst::Predicate)l->op);
  if (isUnsigned && !constInt(l->args[1], uC)) return false;
  auto zeroC = [&](EP e) { int64_t z; return constInt(e, z) && z == 0; };
  // the right-hand side the bounds are against: the literal's, or 0 for an unsigned range test (whose facts are the
  // signed comparisons of P against 0)
  EP rhs = isUnsigned ? nullptr : l->args[1];
  const bool rhsZero = rhs ? zeroC(rhs) : true;
  __int128 lo = std::numeric_limits<__int128>::min() / 4, hi = std::numeric_limits<__int128>::max() / 4;
  bool any = false;
  for (EP f : facts) {
    if (f->k != EK::Cmp || f->args[0]->k != EK::Poly || f->args[0]->ty != l->args[0]->ty) continue;
    int64_t cf; auto pf = split(f->args[0], cf);
    if (pf != pl) continue;
    const bool fZero = zeroC(f->args[1]);
    if (rhsZero && !fZero) {
      // an unsigned range test known true: (unsigned)(P + cf) < C is the two bounds 0 <= P + cf < C
      int64_t fC; if (!(f->op == CmpInst::ICMP_ULT || f->op == CmpInst::ICMP_ULE) || !constInt(f->args[1], fC) || fC < 0) continue;
      lo = std::max(lo, (__int128)-cf); hi = std::min(hi, (__int128)fC - cf - (f->op == CmpInst::ICMP_ULT)); any = true;
      continue;
    }
    if (rhs ? f->args[1] != rhs : !fZero) continue;
    // f: P + cf OP q  ->  bounds on P - q
    unsigned op = f->op;
    const bool isZeroRhs = fZero;
    if (isZeroRhs && op == CmpInst::ICMP_ULE) op = CmpInst::ICMP_EQ;
    if (isZeroRhs && op == CmpInst::ICMP_UGT) op = CmpInst::ICMP_NE;
    switch (op) {
      case CmpInst::ICMP_SGT: lo = std::max(lo, (__int128)-cf + 1); break;
      case CmpInst::ICMP_SGE: lo = std::max(lo, (__int128)-cf); break;
      case CmpInst::ICMP_SLT: hi = std::min(hi, (__int128)-cf - 1); break;
      case CmpInst::ICMP_SLE: hi = std::min(hi, (__int128)-cf); break;
      case CmpInst::ICMP_EQ: lo = std::max(lo, (__int128)-cf); hi = std::min(hi, (__int128)-cf); break;
      default: continue;
    }
    any = true;
  }
  if (!any) return false;
  // l: P + cl OP q, with P - q in [lo, hi]
  const __int128 vlo = lo + cl, vhi = hi + cl;   // range of (P - q) + cl
  unsigned op = l->op;
  if (isUnsigned) {
    // (unsigned)v < C for C >= 0: true when 0 <= v <= hi < C, false when v >= C or v < 0 throughout (a negative v
    // is huge); for a negative C (a range test x - c >= -c', on the top of the unsigned range): (unsigned)v < C is
    // v >= 0 or v < C
    bool ltTrue, ltFalse, leTrue, leFalse;
    if (uC >= 0) { ltTrue = vlo >= 0 && vhi < uC; ltFalse = vlo >= uC || vhi < 0; leTrue = vlo >= 0 && vhi <= uC; leFalse = vlo > uC || vhi < 0; }
    else { ltTrue = vlo >= 0 || vhi < uC; ltFalse = vlo >= uC && vhi < 0; leTrue = vlo >= 0 || vhi <= uC; leFalse = vlo > uC && vhi < 0; }
    switch (op) {
      case CmpInst::ICMP_ULT: if (ltTrue) { out = true; return true; } if (ltFalse) { out = false; return true; } return false;
      case CmpInst::ICMP_UGE: if (ltTrue) { out = false; return true; } if (ltFalse) { out = true; return true; } return false;
      case CmpInst::ICMP_ULE: if (leTrue) { out = true; return true; } if (leFalse) { out = false; return true; } return false;
      case CmpInst::ICMP_UGT: if (leTrue) { out = false; return true; } if (leFalse) { out = true; return true; } return false;
      default: return false;
    }
  }
  switch (op) {
    case CmpInst::ICMP_SGT: if (vlo > 0) { out = true; return true; } if (vhi <= 0) { out = false; return true; } return false;
    case CmpInst::ICMP_SGE: if (vlo >= 0) { out = true; return true; } if (vhi < 0) { out = false; return true; } return false;
    case CmpInst::ICMP_SLT: if (vhi < 0) { out = true; return true; } if (vlo >= 0) { out = false; return true; } return false;
    case CmpInst::ICMP_SLE: if (vhi <= 0) { out = true; return true; } if (vlo > 0) { out = false; return true; } return false;
    case CmpInst::ICMP_EQ: if (vlo == vhi) { out = vlo == 0; return true; } if (vlo > 0 || vhi < 0) { out = false; return true; } return false;
    case CmpInst::ICMP_NE: if (vlo == vhi) { out = vlo != 0; return true; } if (vlo > 0 || vhi < 0) { out = true; return true; } return false;
    default: return false;
  }
}

// c as it reads where ctx holds: the literals ctx implies dropped, guards not implied by ctx replaced by their
// definitions (a value selected under the read's path condition need not repeat that condition: the write's `row
// < N and stage s exists and ...` under a read that established all but `row < N` is `row < N`)
EP Arena::given(EP c, EP ctx) {
  if (isTrue(c) || isFalse(c)) return c;
  auto it = givenMemo.find({c, ctx});
  if (it != givenMemo.end()) return it->second;
  EP r = givenRaw(c, ctx);
  givenMemo[{c, ctx}] = r;
  return r;
}
EP Arena::givenRaw(EP c, EP ctx) {
  if (implies(ctx, c)) return mkBool(true);
  if (c->k == EK::Op && c->op == Instruction::Or && c->ty->isIntegerTy(1)) return orb(given(c->args[0], ctx), given(c->args[1], ctx));
  std::vector<EP> ls; conjuncts(c, ls);
  EP r = mkBool(true);
  for (EP l : ls) {
    if (implies(ctx, l)) continue;
    if (disjoint(ctx, l)) return mkBool(false);
    // a guard's definition holds the previous join's guard: memoized on (definition, ctx), each is expanded once
    if (isGuardSym(*this, l)) { r = andb(r, given(guardDefs[symbols[l->sym].field], ctx)); continue; }
    r = andb(r, l);
  }
  return r;
}

// A conjunction of literals with no integer solution for one of its polynomials: the bounds its orders and equalities
// put on P leave a range every point of which an inequality excludes (`stages > 0`, `stages != 1`, `stages != 2`,
// `stages < 3`: the ring's third iteration under a stage count the loop exit at iteration 1 or 2 rules out).
// Pairwise implication cannot see it; it needs the literals of one polynomial together.
bool Arena::contradictory(const std::vector<EP>& lits, bool clamps) const {
  struct Range { __int128 lo = std::numeric_limits<__int128>::min() / 4, hi = std::numeric_limits<__int128>::max() / 4; std::set<__int128> ne; };
  std::map<std::vector<std::pair<std::vector<uint32_t>, int64_t>>, Range> groups;
  Arena& me = const_cast<Arena&>(*this);
  for (EP l : lits) {
    if (l->k != EK::Cmp || !CmpInst::isIntPredicate((CmpInst::Predicate)l->op) || CmpInst::isUnsigned((CmpInst::Predicate)l->op)) continue;
    EP a = l->args[0], b = l->args[1];
    if (!a->ty->isIntegerTy() || a->ty->isIntegerTy(1) || a->ty != b->ty) continue;
    if (a->k == EK::Undef || b->k == EK::Undef) continue;
    EP d = me.sub(me.asPoly(a), me.asPoly(b));   // the literal is  d OP 0
    if (d->k != EK::Poly) continue;
    std::vector<std::pair<std::vector<uint32_t>, int64_t>> pn; int64_t c = 0;
    for (auto& t : d->terms) { if (t.first.empty()) c = t.second; else pn.push_back(t); }
    if (pn.empty()) continue;
    unsigned op = l->op;
    if (pn[0].second < 0) {   // one sign for the polynomial: -(pn) + -c  flipped-OP  0
      for (auto& t : pn) t.second = -t.second; c = -c;
      op = CmpInst::getSwappedPredicate((CmpInst::Predicate)op);
    }
    auto bound = [&](std::vector<std::pair<std::vector<uint32_t>, int64_t>> pn, int64_t c, unsigned op) {
      // the polynomial with its coefficients' common factor divided out (`128 min(g, 2) > 128` and `min(g, 2) != 1`
      // are about one quantity), the bound rounded to the integers
      int64_t g64 = 0; for (auto& t : pn) g64 = std::gcd(g64, std::llabs(t.second));
      if (g64 > 1) for (auto& t : pn) t.second /= g64; else g64 = 1;
      const __int128 g = g64;
      const __int128 t = -(__int128)c;
      auto fdiv = [](__int128 a, __int128 b) { __int128 q = a / b; if ((a % b != 0) && ((a < 0) != (b < 0))) q--; return q; };
      auto cdiv = [&](__int128 a, __int128 b) { return -fdiv(-a, b); };
      Range& r = groups[pn];
      switch (op) {   // g pn + c OP 0  <=>  pn OP t / g
        case CmpInst::ICMP_SGT: r.lo = std::max(r.lo, fdiv(t, g) + 1); break;
        case CmpInst::ICMP_SGE: r.lo = std::max(r.lo, cdiv(t, g)); break;
        case CmpInst::ICMP_SLT: r.hi = std::min(r.hi, cdiv(t, g) - 1); break;
        case CmpInst::ICMP_SLE: r.hi = std::min(r.hi, fdiv(t, g)); break;
        case CmpInst::ICMP_EQ: if (t % g) { r.lo = 1; r.hi = 0; } else { r.lo = std::max(r.lo, t / g); r.hi = std::min(r.hi, t / g); } break;
        case CmpInst::ICMP_NE: if (t % g == 0) r.ne.insert(t / g); break;
        default: break;
      }
    };
    bound(pn, c, op);
    // A clamp against a constant compared for (in)equality with that constant decides the clamped polynomial's
    // side: max(P, k) != k is P > k, min(P, k) != k is P < k; max(P, k) == v is P == v (v > k), false (v < k) or
    // P <= k (v == k), min the other way round. The frontend rotates a loop `for (st = 0; st < stages; st++)` into
    // `st != max(stages, 1)` while a copy the iteration depends on was issued under `stages > st`: the two say the
    // same thing about `stages`. Always on: it adds bounds only where a literal names the clamp's own constant.
    if (pn.size() == 1 && pn[0].first.size() == 1 && pn[0].second == 1 && (op == CmpInst::ICMP_NE || op == CmpInst::ICMP_EQ)) {
      EP atom = atoms[pn[0].first[0]];
      if (atom->k == EK::Call && atom->callee && atom->args.size() == 2 && (atom->callee->getName().starts_with("llvm.smin.") || atom->callee->getName().starts_with("llvm.smax."))) {
        const bool isMin = atom->callee->getName().starts_with("llvm.smin.");
        int64_t k; EP P = nullptr;
        if (constInt(atom->args[1], k)) P = atom->args[0]; else if (constInt(atom->args[0], k)) P = atom->args[1];
        if (P && P->ty == atom->ty) {
          const __int128 v = -(__int128)c;   // the literal is  clamp OP v
          EP dp = me.asPoly(P);
          if (dp->k == EK::Poly) {
            std::vector<std::pair<std::vector<uint32_t>, int64_t>> pp; int64_t cp = 0;
            for (auto& t2 : dp->terms) { if (t2.first.empty()) cp = t2.second; else pp.push_back(t2); }
            if (!pp.empty()) {
              // the derived literal on P:  pp + cp OP2 v2  as  pp + (cp - v2) OP2 0
              auto derive = [&](unsigned op2, __int128 v2) {
                const __int128 c2 = (__int128)cp - v2;
                if (c2 < std::numeric_limits<int64_t>::min() / 2 || c2 > std::numeric_limits<int64_t>::max() / 2) return;
                auto pq = pp; int64_t c3 = (int64_t)c2;
                if (pq[0].second < 0) { for (auto& t2 : pq) t2.second = -t2.second; c3 = -c3; op2 = CmpInst::getSwappedPredicate((CmpInst::Predicate)op2); }
                bound(pq, c3, op2);
              };
              // (and max(P, k) != v with v > k is P != v: the clamp takes the value v only from P; min likewise)
              if (op == CmpInst::ICMP_NE) { if (v == k) derive(isMin ? CmpInst::ICMP_SLT : CmpInst::ICMP_SGT, k); else if ((isMin && v < k) || (!isMin && v > k)) derive(CmpInst::ICMP_NE, v); }
              else if ((isMin && v > k) || (!isMin && v < k)) return true;
              else if (v == k) derive(isMin ? CmpInst::ICMP_SGE : CmpInst::ICMP_SLE, k);
              else derive(CmpInst::ICMP_EQ, v);
            }
          }
        }
      }
    }
    // With `clamps`: a clamp against a constant, bounded, is a bound on the clamped polynomial where the clamp does
    // not decide it. min(P, k) > t is P > t (and k > t, or the literal is false); min(P, k) < t is P < t unless k < t
    // (then it says nothing); max the other way round. The ring's "stages left" is min(groups left, per stage): `no
    // stage left` and `a third stage` contradict as bounds on the groups left, where the clamped literals are
    // unrelated. Not the default: the sharper test changes what andb/orb absorb, and one kernel's execution went
    // from 12 s to 9 minutes with it on everywhere; resolve()'s dead-residual test asks for it.
    if (clamps && pn.size() == 1 && pn[0].first.size() == 1 && pn[0].second == 1 && (op == CmpInst::ICMP_SGT || op == CmpInst::ICMP_SGE || op == CmpInst::ICMP_SLT || op == CmpInst::ICMP_SLE)) {
      EP atom = atoms[pn[0].first[0]];
      // a bound on a polynomial: P >= L (lower) or P <= U
      auto boundPoly = [&](EP P, __int128 v, bool lowerBound) {
        EP dp = me.asPoly(P);
        if (dp->k != EK::Poly) return;
        std::vector<std::pair<std::vector<uint32_t>, int64_t>> pp; __int128 cp = -v;
        for (auto& t2 : dp->terms) { if (t2.first.empty()) cp += t2.second; else pp.push_back(t2); }
        if (pp.empty() || cp < std::numeric_limits<int64_t>::min() / 2 || cp > std::numeric_limits<int64_t>::max() / 2) return;
        unsigned op2 = lowerBound ? CmpInst::ICMP_SGE : CmpInst::ICMP_SLE;   // pp + cp OP2 0
        int64_t c2 = (int64_t)cp;
        if (pp[0].second < 0) { for (auto& t2 : pp) t2.second = -t2.second; c2 = -c2; op2 = CmpInst::getSwappedPredicate((CmpInst::Predicate)op2); }
        bound(pp, c2, op2);
      };
      // A division by a positive constant, bounded, bounds the dividend: sdiv(P, k) >= m is P >= m k (m >= 1) or
      // P >= (m - 1) k + 1 (m <= 0, the quotient truncating toward zero); sdiv(P, k) <= m is P <= (m + 1) k - 1
      // (m >= 0) or P <= m k (m <= -1). The K groups of a split are `ceil(tiles / 2)`: "one group" and "a third
      // tile" contradict as bounds on the tiles.
      int64_t kdiv;
      if (atom->k == EK::Op && atom->op == Instruction::SDiv && atom->args.size() == 2 && constInt(atom->args[1], kdiv) && kdiv > 0) {
        const __int128 t = -(__int128)c;
        if (op == CmpInst::ICMP_SGT || op == CmpInst::ICMP_SGE) { const __int128 m = op == CmpInst::ICMP_SGT ? t + 1 : t; boundPoly(atom->args[0], m >= 1 ? m * kdiv : (m - 1) * kdiv + 1, true); }
        else { const __int128 m = op == CmpInst::ICMP_SLT ? t - 1 : t; boundPoly(atom->args[0], m >= 0 ? (m + 1) * kdiv - 1 : m * kdiv, false); }
      }
      if (atom->k == EK::Call && atom->callee && atom->args.size() == 2 && (atom->callee->getName().starts_with("llvm.smin.") || atom->callee->getName().starts_with("llvm.smax."))) {
        const bool isMin = atom->callee->getName().starts_with("llvm.smin.");
        int64_t k; EP P = nullptr;
        if (constInt(atom->args[1], k)) P = atom->args[0]; else if (constInt(atom->args[0], k)) P = atom->args[1];
        if (P && P->ty == atom->ty) {
          const bool lower = op == CmpInst::ICMP_SGT || op == CmpInst::ICMP_SGE;   // a lower bound on the clamp
          // the clamp's own bound on the constant: k OP -c
          const __int128 t = -(__int128)c;
          const bool kHolds = op == CmpInst::ICMP_SGT ? k > t : op == CmpInst::ICMP_SGE ? k >= t : op == CmpInst::ICMP_SLT ? k < t : k <= t;
          // min: a lower bound needs both (k fails: false), an upper bound holds by either (k holds: no information);
          // max: a lower bound holds by either, an upper bound needs both
          const bool both = isMin == lower;
          if (both && !kHolds) return true;
          if (both || !kHolds) {
            EP dp = me.asPoly(P);
            if (dp->k == EK::Poly) {
              std::vector<std::pair<std::vector<uint32_t>, int64_t>> pp; int64_t cp = c;
              for (auto& t2 : dp->terms) { if (t2.first.empty()) cp += t2.second; else pp.push_back(t2); }
              if (!pp.empty()) {
                unsigned op2 = op;
                if (pp[0].second < 0) { for (auto& t2 : pp) t2.second = -t2.second; cp = -cp; op2 = CmpInst::getSwappedPredicate((CmpInst::Predicate)op2); }
                bound(pp, cp, op2);
              }
            }
          }
        }
      }
    }
  }
  // Bounds on a dividend or a clamped polynomial bound the quotient / the clamp (monotone in P): a group whose
  // key is one such atom takes the range of its polynomial's group, mapped. `stages = (tiles + 1) / 2` with
  // 0 < tiles <= 4 leaves stages in 1..2, which `stages != 1, != 2` exclude (a third stage's second group).
  // (with `clamps` only, like the bounds above: on everywhere it changes what andb/orb absorb, and the shapes the
  // executor gives the operand words with it)
  if (clamps) for (unsigned pass = 0; pass < 2; pass++) for (auto& kv : groups) {
    if (kv.first.size() != 1 || kv.first[0].first.size() != 1 || kv.first[0].second != 1) continue;
    EP atom = atoms[kv.first[0].first[0]];
    int64_t k; EP P = nullptr; int kind = 0;   // 1 sdiv, 2 smin, 3 smax
    if (atom->k == EK::Op && atom->op == Instruction::SDiv && atom->args.size() == 2 && constInt(atom->args[1], k) && k > 0) { P = atom->args[0]; kind = 1; }
    else if (atom->k == EK::Call && atom->callee && atom->args.size() == 2 && (atom->callee->getName().starts_with("llvm.smin.") || atom->callee->getName().starts_with("llvm.smax."))) {
      kind = atom->callee->getName().starts_with("llvm.smin.") ? 2 : 3;
      if (constInt(atom->args[1], k)) P = atom->args[0]; else if (constInt(atom->args[0], k)) P = atom->args[1];
    }
    if (!P || P->ty != atom->ty) continue;
    EP dp = me.asPoly(P);
    if (dp->k != EK::Poly) continue;
    std::vector<std::pair<std::vector<uint32_t>, int64_t>> pp; __int128 cp = 0; int sgn = 1;
    for (auto& t2 : dp->terms) { if (t2.first.empty()) cp = t2.second; else pp.push_back(t2); }
    if (pp.empty()) continue;
    if (pp[0].second < 0) { for (auto& t2 : pp) t2.second = -t2.second; sgn = -1; }
    int64_t gp = 0; for (auto& t2 : pp) gp = std::gcd(gp, std::llabs(t2.second));
    if (gp > 1) for (auto& t2 : pp) t2.second /= gp; else gp = 1;   // P = sgn gp G + cp, G the group's polynomial
    auto g = groups.find(pp);
    if (g == groups.end()) continue;
    const Range& pr = g->second;
    if (pr.lo > pr.hi) return true;
    const __int128 big = std::numeric_limits<__int128>::max() / 8;
    const bool loB = pr.lo > -big, hiB = pr.hi < big;   // each side maps on its own (the map is monotone)
    const bool pLoB = sgn > 0 ? loB : hiB, pHiB = sgn > 0 ? hiB : loB;
    const __int128 plo = sgn > 0 ? gp * pr.lo + cp : -gp * pr.hi + cp, phi = sgn > 0 ? gp * pr.hi + cp : -gp * pr.lo + cp;
    auto mapQ = [&](__int128 pv) { return kind == 1 ? pv / k : kind == 2 ? std::min<__int128>(pv, k) : std::max<__int128>(pv, k); };   // C++ division truncates toward zero as sdiv does
    Range& qr = kv.second;
    if (pLoB) qr.lo = std::max(qr.lo, mapQ(plo));
    if (pHiB) qr.hi = std::min(qr.hi, mapQ(phi));
  }
  for (auto& kv : groups) {
    const Range& r = kv.second;
    if (r.lo > r.hi) return true;
    if (r.hi - r.lo + 1 <= (__int128)r.ne.size()) {
      bool all = true;
      for (__int128 v = r.lo; v <= r.hi && all; v++) all = r.ne.count(v) != 0;
      if (all) return true;
    }
  }
  return false;
}

// a guard among the literals of a conjunction: the conjunction with the guard replaced by its definition (a
// disjunction), or null when there is none
static bool isGuardSym(const Arena& A, EP l) { return l->k == EK::Sym && A.symbol(l->sym).kind == Symbol::Guard; }
EP Arena::expandGuard(const std::vector<EP>& lits) const {
  for (size_t i = 0; i < lits.size(); i++) {
    if (!isGuardSym(*this, lits[i])) continue;
    Arena& me = const_cast<Arena&>(*this);
    EP rest = me.mkBool(true);
    for (size_t j = 0; j < lits.size(); j++) if (j != i) { Expr e; e.k = EK::Op; e.ty = lits[j]->ty; e.op = Instruction::And; e.args = {rest, lits[j]}; rest = isTrue(rest) ? lits[j] : me.intern(std::move(e)); }
    EP def = guardDefs[symbols[lits[i]->sym].field];
    // distribute: (D1 or D2) and rest = (D1 and rest) or (D2 and rest), without simplification
    std::vector<EP> clauses;
    std::function<void(EP)> flat = [&](EP c) { if (c->k == EK::Op && c->op == Instruction::Or && c->ty->isIntegerTy(1)) { flat(c->args[0]); flat(c->args[1]); return; } clauses.push_back(c); };
    flat(def);
    EP r = nullptr;
    for (EP c : clauses) {
      EP cl = c;
      if (!isTrue(rest)) { Expr e; e.k = EK::Op; e.ty = c->ty; e.op = Instruction::And; e.args = {c, rest}; cl = me.intern(std::move(e)); }
      if (!r) { r = cl; continue; }
      Expr e; e.k = EK::Op; e.ty = c->ty; e.op = Instruction::Or; e.args = {r, cl}; r = me.intern(std::move(e));
    }
    return r;
  }
  return nullptr;
}
bool Arena::implies(EP a, EP b) const {
  if (isTrue(b) || a == b || isFalse(a)) return true;
  auto it = impliesMemo.find({a, b});
  if (it != impliesMemo.end()) return it->second;
  const bool r = impliesRaw(a, b);
  impliesMemo[{a, b}] = r;
  return r;
}
bool Arena::disjoint(EP a, EP b) const {
  if (isFalse(a) || isFalse(b)) return true;
  auto it = disjointMemo.find({a, b});
  if (it != disjointMemo.end()) return it->second;
  const bool r = disjointRaw(a, b);
  disjointMemo[{a, b}] = r;
  return r;
}
bool Arena::impliesRaw(EP a, EP b) const {
  if (isTrue(b) || a == b || isFalse(a)) return true;
  auto isOr = [](EP c) { return c->k == EK::Op && c->op == Instruction::Or && c->ty->isIntegerTy(1); };
  if (isOr(a)) return implies(a->args[0], b) && implies(a->args[1], b);
  if (isOr(b)) return implies(a, b->args[0]) || implies(a, b->args[1]);
  if (isGuardSym(*this, b)) return implies(a, guardDefs[symbols[b->sym].field]);
  std::vector<EP> la, lb;
  conjuncts(a, la); conjuncts(b, lb);
  for (EP l : lb) {
    bool ok = false;
    for (EP m : la) if (litImplies(m, l)) { ok = true; break; }
    // a guard among b's literals: a must imply its definition (kt > 5 implies G = (G' or kt > 4)); a negated
    // guard: a must be disjoint from it (kt > 5 implies not G = (G' or (kt > 3 and kt < 5)))
    if (!ok && isGuardSym(*this, l) && implies(a, guardDefs[symbols[l->sym].field])) ok = true;
    if (!ok && l->k == EK::Op && l->op == Instruction::Xor && l->args.size() == 2 && isTrue(l->args[1]) && isGuardSym(*this, l->args[0]) && disjoint(a, guardDefs[symbols[l->args[0]->sym].field])) ok = true;
    // the literals of a together exclude not-l (`stages > 0, != 1, != 2` imply `stages > 2`)
    if (!ok) { std::vector<EP> all(la); all.push_back(const_cast<Arena*>(this)->notb(l)); if (contradictory(all)) ok = true; }
    if (!ok) {
      // a guard among a's literals: every clause of its definition (with the other literals) must imply b
      if (EP ex = expandGuard(la)) { if (guardExpandDepth >= kGuardExpandLimit) return false; GuardExpandScope s(*this); return implies(ex, b); }
      return false;
    }
  }
  return true;
}
bool Arena::disjointRaw(EP a, EP b) const {
  if (isFalse(a) || isFalse(b)) return true;
  auto isOr = [](EP c) { return c->k == EK::Op && c->op == Instruction::Or && c->ty->isIntegerTy(1); };
  if (isOr(a)) return disjoint(a->args[0], b) && disjoint(a->args[1], b);
  if (isOr(b)) return disjoint(a, b->args[0]) && disjoint(a, b->args[1]);
  std::vector<EP> la, lb;
  conjuncts(a, la); conjuncts(b, lb);
  for (EP x : la) for (EP y : lb) {
    EP nx = const_cast<Arena*>(this)->notb(x);
    if (nx == y) return true;
    if (litImplies(y, nx)) return true;   // y => not x
  }
  { std::vector<EP> all(la); all.insert(all.end(), lb.begin(), lb.end()); if (contradictory(all)) return true; }
  EP ea = expandGuard(la), eb = ea ? nullptr : expandGuard(lb);
  if (!ea && !eb) return false;
  if (guardExpandDepth >= kGuardExpandLimit) return false;
  GuardExpandScope s(*this);
  return ea ? disjoint(ea, b) : disjoint(a, eb);
}

bool Arena::containsSymbol(EP e, Symbol::Kind kind, std::set<uint32_t>* found) {
  std::vector<EP> work{e};
  std::set<EP> seen;
  bool any = false;
  while (!work.empty()) {
    EP x = work.back(); work.pop_back();
    if (!seen.insert(x).second) continue;
    if (x->k == EK::Sym) {
      if (symbols[x->sym].kind == Symbol::Guard) { work.push_back(guardDefs[symbols[x->sym].field]); continue; }
      if (symbols[x->sym].kind == kind) { any = true; if (found) found->insert(x->sym); else return true; }
      continue;
    }
    if (x->k == EK::Poly) { for (auto& t : x->terms) for (auto a : t.first) work.push_back(atoms[a]); continue; }
    for (EP a : x->args) work.push_back(a);
  }
  return any;
}

std::string Arena::str(EP e, unsigned depth) const {
  std::string out;
  raw_string_ostream os(out);
  if (!depth) { os << "..."; return out; }
  switch (e->k) {
    case EK::Const: if (e->ty->isIntegerTy(1)) os << (e->c.isOne() ? "true" : "false"); else { os << "c"; e->c.print(os, false); } break;
    case EK::Undef: os << "undef"; break;
    case EK::Sym: {
      const Symbol& s = symbols[e->sym];
      switch (s.kind) {
        case Symbol::Arg: os << "arg" << llvm::cast<Argument>(s.v)->getArgNo(); break;
        case Symbol::Sreg: os << s.name.substr(s.name.rfind('.') + 1); break;
        case Symbol::Coord: os << '$' << s.name; break;
        case Symbol::ThreadValue: os << "tv" << s.thread << "." << s.field; if (!s.name.empty()) os << "#" << s.name; break;
        case Symbol::Unknown: os << "?" << s.field; break;
        case Symbol::SharedBase: os << "smem"; break;
        case Symbol::Global: os << "g"; break;
        case Symbol::SmemByte: os << "sb" << s.field; break;
        case Symbol::LoopValue: if (!s.name.empty()) os << s.name; else os << "lv"; break;
        case Symbol::Private: os << "priv" << s.thread; break;
        case Symbol::Guard: os << "G" << s.field; break;
      }
      break;
    }
    case EK::Poly: {
      bool first = true;
      for (auto& t : e->terms) {
        if (!first) os << "+"; first = false;
        os << t.second;
        for (auto a : t.first) os << "*" << str(atoms[a], depth - 1);
      }
      if (first) os << "0";
      break;
    }
    case EK::Op: os << Instruction::getOpcodeName(e->op) << "("; for (EP a : e->args) os << str(a, depth - 1) << ","; os << ")"; break;
    case EK::Cmp: os << CmpInst::getPredicateName((CmpInst::Predicate)e->op) << "(" << str(e->args[0], depth - 1) << "," << str(e->args[1], depth - 1) << ")"; break;
    case EK::Ite: os << "ite(" << str(e->args[0], depth - 1) << "," << str(e->args[1], depth - 1) << "," << str(e->args[2], depth - 1) << ")"; break;
    case EK::Load: os << "ld[" << str(e->args[0], depth - 1) << "]"; break;
    case EK::Concat: os << "{"; for (EP a : e->args) os << str(a, depth - 1) << ","; os << "}"; break;
    case EK::Bytes: os << str(e->args[0], depth - 1) << "[" << e->lo << ":" << e->len << "]"; break;
    case EK::Call: os << (e->callee ? e->callee->getName().str() : e->asmv ? "asm" : "call") << "("; for (EP a : e->args) os << str(a, depth - 1) << ","; os << ")"; break;
  }
  return out;
}

// ------------------------------------------------------------------------------------------------ executor

static bool isBarrierName(StringRef n) { return n == "llvm.nvvm.barrier0" || n.starts_with("llvm.nvvm.barrier") || n.starts_with("llvm.nvvm.bar.sync"); }

EP Executor::unknown(Type* ty) {
  Symbol s; s.kind = Symbol::Unknown; s.thread = tid; s.ty = ty; s.field = unkCounter++; s.v = curI;
  if (ty->isStructTy()) {
    std::vector<EP> f;
    for (unsigned i = 0; i < ty->getStructNumElements(); i++) f.push_back(unknown(ty->getStructElementType(i)));
    return A.concat(std::move(f), ty);
  }
  return A.mkSym(s);
}

EP Executor::gep(GEPOperator& G, EP base) {
  EP p = A.retype(A.asPoly(base), G.getType()->isVectorTy() ? G.getType() : G.getType());
  Type* i64 = Type::getInt64Ty(F.getContext());
  for (gep_type_iterator GTI = gep_type_begin(G), E = gep_type_end(G); GTI != E; ++GTI) {
    Value* idx = GTI.getOperand();
    if (StructType* ST = GTI.getStructTypeOrNull()) {
      unsigned fi = (unsigned)cast<ConstantInt>(idx)->getZExtValue();
      p = A.add(p, A.mkInt(p->ty, (int64_t)A.DL.getStructLayout(ST)->getElementOffset(fi)));
      continue;
    }
    int64_t esz = (int64_t)A.DL.getTypeAllocSize(GTI.getIndexedType());
    EP iv = A.retype(A.cast(Instruction::SExt, get(idx), i64), p->ty);
    p = A.add(p, A.mul(iv, A.mkInt(p->ty, esz)));
  }
  return p;
}

EP Executor::evalConstant(Constant* Cst) {
  Type* ty = Cst->getType();
  if (auto* CI = dyn_cast<ConstantInt>(Cst)) return ty->isIntegerTy(1) ? A.mkBool(CI->isOne()) : A.mkInt(ty, CI->getSExtValue());
  if (auto* CF = dyn_cast<ConstantFP>(Cst)) return A.mkConstFP(ty, CF->getValueAPF().bitcastToAPInt());
  if (isa<ConstantPointerNull>(Cst)) return A.mkInt(ty, 0);
  if (isa<UndefValue>(Cst)) return A.mkUndef(ty);
  if (auto* GV = dyn_cast<GlobalVariable>(Cst)) {
    Symbol s; s.kind = GV->getAddressSpace() == 3 ? Symbol::SharedBase : Symbol::Global; s.v = GV; s.ty = ty;
    return A.mkSym(s);
  }
  if (isa<Function>(Cst)) { Symbol s; s.kind = Symbol::Global; s.v = Cst; s.ty = ty; return A.mkSym(s); }
  if (isa<ConstantAggregateZero>(Cst)) {
    std::vector<EP> zeros(A.sizeOf(ty), A.mkInt(Type::getInt8Ty(F.getContext()), 0));
    return A.fromBytes(zeros, ty);
  }
  if (auto* CDV = dyn_cast<ConstantDataVector>(Cst)) {
    std::vector<EP> parts;
    for (unsigned i = 0; i < CDV->getNumElements(); i++) parts.push_back(evalConstant(CDV->getElementAsConstant(i)));
    return A.concat(std::move(parts), ty);
  }
  if (auto* CV = dyn_cast<ConstantVector>(Cst)) {
    std::vector<EP> parts;
    for (unsigned i = 0; i < CV->getNumOperands(); i++) parts.push_back(evalConstant(CV->getOperand(i)));
    return A.concat(std::move(parts), ty);
  }
  if (auto* CE = dyn_cast<ConstantExpr>(Cst)) {
    if (auto* G = dyn_cast<GEPOperator>(CE)) return gep(*G, evalConstant(cast<Constant>(G->getPointerOperand())));
    if (CE->isCast()) return A.cast(CE->getOpcode(), evalConstant(CE->getOperand(0)), ty);
    if (Instruction::isBinaryOp(CE->getOpcode())) return A.binop(CE->getOpcode(), evalConstant(CE->getOperand(0)), evalConstant(CE->getOperand(1)), ty);
  }
  return unknown(ty);
}

EP Executor::get(Value* v) {
  if (auto* Cst = dyn_cast<Constant>(v)) return evalConstant(Cst);
  if (auto* Arg = dyn_cast<Argument>(v)) { Symbol s; s.kind = Symbol::Arg; s.v = Arg; s.ty = Arg->getType(); return A.mkSym(s); }
  // Floating-point data that reaches the epilogue from before its first barrier is kept opaque: the epilogue's
  // rewrite can name the IR value itself (it dominates the barrier), and the store descriptions stay small.
  if (Be) if (auto* I = dyn_cast<Instruction>(v)) if (I->getType()->isFPOrFPVectorTy() && DT.dominates(I, Be)) {
    Symbol s; s.kind = Symbol::ThreadValue; s.v = I; s.thread = tid; s.ty = I->getType();
    return A.mkSym(s);
  }
  auto it = env.find(v);
  if (it != env.end()) return it->second;
  return unknown(v->getType());
}

bool Executor::classifyAddr(EP addr, GlobalVariable*& sharedObj, int64_t& off) {
  sharedObj = nullptr; off = 0;
  addr = A.asPoly(addr);
  int64_t constPart = 0;
  for (auto& t : addr->terms) {
    if (t.first.empty()) { constPart = t.second; continue; }
    uint32_t sid;
    if (t.first.size() == 1 && A.isSymAtom(t.first[0], sid) && A.symbol(sid).kind == Symbol::SharedBase) {
      if (sharedObj || t.second != 1) return false;
      sharedObj = cast<GlobalVariable>(A.symbol(sid).v);
      continue;
    }
    if (sharedObj) return false;  // a shared address with a symbolic offset
  }
  if (sharedObj) {
    // every other term must be absent (offset concrete)
    for (auto& t : addr->terms) {
      if (t.first.empty()) continue;
      uint32_t sid;
      if (t.first.size() == 1 && A.isSymAtom(t.first[0], sid) && A.symbol(sid).kind == Symbol::SharedBase) continue;
      return false;
    }
    off = constPart;
  }
  return true;
}

bool Executor::isPrivateAddr(EP addr) {
  EP p = A.asPoly(addr);
  if (p->k != EK::Poly) return false;
  for (auto& t : p->terms) for (uint32_t a : t.first) { uint32_t sid; if (A.isSymAtom(a, sid) && A.symbol(sid).kind == Symbol::Private) return true; }
  return false;
}

bool Executor::storeBytes(EP addr, const std::vector<EP>& bs, EP R, Instruction* site) {
  if (shadow) return true;  // recurrence pass: effects are not recorded
  if (retile && isPrivateAddr(addr)) { privateAccesses++; return true; }
  // a selected address with a shared object in an arm: the store to each arm under its condition (see loadBytes)
  if (EP u = A.unwrap(addr); u->k == EK::Ite && (A.containsSymbol(u->args[1], Symbol::SharedBase) || A.containsSymbol(u->args[2], Symbol::SharedBase))) {
    EP c = u->args[0], Rt = A.andb(R, c), Rf = A.andb(R, A.notb(c));
    if (!A.isFalse(Rt) && !storeBytes(u->args[1], bs, Rt, site)) return false;
    if (!A.isFalse(Rf) && !storeBytes(u->args[2], bs, Rf, site)) return false;
    return true;
  }
  GlobalVariable* obj; int64_t off;
  if (!classifyAddr(addr, obj, off)) {
    if (retile) {
      // a shared store at a symbolic offset: recorded (for the footprint) but not modeled; a later read of those bytes
      // resolves to the concrete writes only, which is unsound, so the retiler refuses objects with such stores
      GlobalVariable* o2 = nullptr; EP p = A.asPoly(addr);
      for (auto& t : p->terms) { uint32_t sid; if (t.first.size() == 1 && A.isSymAtom(t.first[0], sid) && A.symbol(sid).kind == Symbol::SharedBase) o2 = cast<GlobalVariable>(A.symbol(sid).v); }
      if (o2) { smemAccesses.push_back({site, tid, curIter, addr, false, 0, (unsigned)bs.size(), true, phase}); return true; }
      if (recordDevStores) { devStores.push_back({addr, bs, R, phase, tid, site}); devStoreIters.push_back(iterStack); devStoreLoops.push_back(loopStack); }
      return true;  // device store: no effect modeled by the retiler
    }
    return fail("a store address in the epilogue is shared memory at a symbolic offset");
  }
  if (obj && retile) {
    smemAccesses.push_back({site, tid, curIter, addr, true, off, (unsigned)bs.size(), true, phase});
    for (unsigned i = 0; i < bs.size(); i++) { smemByOff[off + i].push_back(smemWrites.size()); smemWrites.push_back({off + (int64_t)i, bs[i], R, phase, tid, seq, site}); }
    seq++;
    return true;
  }
  if (!obj && retile) { if (recordDevStores) { devStores.push_back({addr, bs, R, phase, tid, site}); devStoreIters.push_back(iterStack); devStoreLoops.push_back(loopStack); } return true; }
  if (obj) {
    if (!afterSkip) return true;  // staging writes before the epilogue are not observable after it (checked at the reads)
    if (!Be && pendingBe) { Be = pendingBe; epiPhase = pendingBePhase; beCond = pendingBeCond; }
    for (unsigned i = 0; i < bs.size(); i++) {
      smemByOff[off + i].push_back(smemWrites.size());
      smemWrites.push_back({off + (int64_t)i, bs[i], R, phase, tid, seq, site});
    }
    seq++;
    return true;
  }
  if (!afterSkip) return fail("a device-memory store before the epilogue (" + std::string(site->getOpcodeName()) + ")");
  // Split-K protocol (decode.cuh): the arrival counter and its self-reset are a global atomic's result, which
  // the executor names Unknown. Those stores are not the C tile; skip them so the accumulator stores can recover.
  if (A.containsSymbol(addr, Symbol::Unknown) || A.containsSymbol(R, Symbol::Unknown)) return true;
  for (EP b : bs) if (A.containsSymbol(b, Symbol::Unknown)) return true;
  EP sc = (Be && beCond) ? A.given(R, beCond) : R;
  devStores.push_back({addr, bs, sc, phase, tid, site});
  firstStorePhase = std::min(firstStorePhase, phase);
  return true;
}

bool Executor::loadBytes(EP addr, unsigned n, unsigned align, EP R, Instruction* site, std::vector<EP>& out) {
  GlobalVariable* obj; int64_t off;
  out.clear();
  // a selected address with a shared object in an arm (a ring stage chosen by a symbolic stage index, `stage[t % S]`
  // as a select chain): each arm under its condition, the bytes a selection of the arms' (a select over shared
  // pointers is not a device address)
  if (EP u = A.unwrap(addr); u->k == EK::Ite && (A.containsSymbol(u->args[1], Symbol::SharedBase) || A.containsSymbol(u->args[2], Symbol::SharedBase))) {
    EP c = u->args[0], Rt = A.andb(R, c), Rf = A.andb(R, A.notb(c));
    std::vector<EP> a, b;
    if (!A.isFalse(Rt) && !loadBytes(u->args[1], n, align, Rt, site, a)) return false;
    if (!A.isFalse(Rf) && !loadBytes(u->args[2], n, align, Rf, site, b)) return false;
    if (a.empty()) { out = b; return true; }
    if (b.empty()) { out = a; return true; }
    for (unsigned i = 0; i < n; i++) out.push_back(A.ite(c, a[i], b[i]));
    return true;
  }
  if (retile && isPrivateAddr(addr)) { if (!shadow) privateAccesses++; EP u = unknown(Type::getIntNTy(F.getContext(), 8 * n)); out = A.toBytes(u); return true; }
  if (shadow) {
    // recurrence pass: a shared read is an opaque per-iteration value (its resolution belongs to the real pass);
    // a device read is the pure Load expression, recorded nowhere
    if (!classifyAddr(addr, obj, off) || obj) { EP u = unknown(Type::getIntNTy(F.getContext(), 8 * n)); out = A.toBytes(u); return true; }
    out = A.toBytes(A.load(Type::getIntNTy(F.getContext(), 8 * n), addr, R, align));
    return true;
  }
  if (!classifyAddr(addr, obj, off)) {
    if (retile) {
      GlobalVariable* o2 = nullptr; EP p = A.asPoly(addr);
      for (auto& t : p->terms) { uint32_t sid; if (t.first.size() == 1 && A.isSymAtom(t.first[0], sid) && A.symbol(sid).kind == Symbol::SharedBase) o2 = cast<GlobalVariable>(A.symbol(sid).v); }
      if (o2) smemAccesses.push_back({site, tid, curIter, addr, false, 0, n, false, phase});
      EP u = unknown(Type::getIntNTy(F.getContext(), 8 * n)); out = A.toBytes(u); return true;  // shared at a symbolic offset (not an operand path), or device through an opaque pointer
    }
    return fail("a load address in the epilogue is shared memory at a symbolic offset");
  }
  if (obj && retile) smemAccesses.push_back({site, tid, curIter, addr, true, off, n, false, phase});
  if (obj) {
    if (!afterSkip && !retile) { EP u = unknown(Type::getIntNTy(F.getContext(), 8 * n)); out = A.toBytes(u); return true; }
    // one thread stored, a barrier, everyone loads (s_last / decode.cu shared atomics): fold the write now so a
    // branch on the broadcast is uniform and a reduction over k_splits can stop. Permutation loads keep the
    // writer's ThreadValue (resolve() still sees the read).
    if (afterSkip && !retile) {
      std::vector<EP> bs;
      bool ok = true;
      for (unsigned i = 0; i < n && ok; i++) {
        const SmemWrite* W = nullptr;
        auto it = smemByOff.find(off + (int64_t)i);
        if (it != smemByOff.end()) {
          for (auto wi = it->second.rbegin(); wi != it->second.rend(); ++wi) {
            const SmemWrite& wr = smemWrites[*wi];
            if (wr.phase < phase) { W = &wr; break; }
          }
        }
        if (!W) { ok = false; break; }
        bs.push_back(W->byte);
      }
      if (ok) {
        for (unsigned i = 0; i < n; i++) {
          Symbol s; s.kind = Symbol::SmemByte; s.v = obj; s.ty = Type::getInt8Ty(F.getContext()); s.field = (unsigned)smemReads.size(); s.thread = tid;
          EP e = A.mkSymRaw(s);
          smemReads.push_back({off + (int64_t)i, R, phase, tid, seq, site, obj, e->sym});
        }
        seq++;
        out = std::move(bs);
        return true;
      }
    }
    for (unsigned i = 0; i < n; i++) {
      Symbol s; s.kind = Symbol::SmemByte; s.v = obj; s.ty = Type::getInt8Ty(F.getContext()); s.field = (unsigned)smemReads.size(); s.thread = tid;
      EP e = A.mkSymRaw(s);
      smemReads.push_back({off + (int64_t)i, R, phase, tid, seq, site, obj, e->sym});
      out.push_back(A.asPoly(e));
    }
    seq++;
    return true;
  }
  // split-K / stream-K / residual: a device load may be ordered after this block's stores by a barrier
  Type* ity = Type::getIntNTy(F.getContext(), 8 * n);
  EP v = A.load(ity, addr, (afterSkip || retile) ? R : A.mkBool(true), align);
  if (retile && recordDevLoads) devLoads.push_back({addr, n, R, phase, tid, iterStack, site, loopStack});
  out = A.toBytes(v);
  return true;
}

const std::vector<Value*>& Executor::liveOuts(const Loop* L) {
  auto it = liveOutMemo.find(L);
  if (it != liveOutMemo.end()) return it->second;
  std::vector<Value*> outs;
  for (BasicBlock* BB : L->blocks())
    for (Instruction& I : *BB)
      for (User* U : I.users()) {
        auto* UI = dyn_cast<Instruction>(U);
        if (UI && !L->contains(UI->getParent())) { outs.push_back(&I); break; }
      }
  return liveOutMemo[L] = outs;
}

void Executor::edgeTo(BasicBlock* from, BasicBlock* to, EP cond, EdgeMap& out) {
  if (factorGuards) cond = A.factor(cond);
  Edge e{from, cond, {}};
  for (PHINode& PN : to->phis()) e.snap[&PN] = get(PN.getIncomingValueForBlock(from));
  // leaving loops: snapshot their live-outs
  const Loop* Lf = LI.getLoopFor(from);
  const Loop* Lt = LI.getLoopFor(to);
  for (const Loop* L = Lf; L && !L->contains(to); L = L->getParentLoop())
    for (Value* v : liveOuts(L)) e.snap[v] = get(v);
  (void)Lt;
  out[to].push_back(std::move(e));
}

static bool sliceUsesTid(Instruction& I) {
  SmallVector<Instruction*, 16> st;
  SmallPtrSet<Instruction*, 32> seen;
  st.push_back(&I);
  while (!st.empty()) {
    Instruction* J = st.pop_back_val();
    if (!seen.insert(J).second) continue;
    if (auto* C = dyn_cast<CallInst>(J))
      if (Function* F = C->getCalledFunction()) {
        StringRef n = F->getName();
        if (n == "llvm.nvvm.read.ptx.sreg.tid.x" || n == "llvm.nvvm.read.ptx.sreg.laneid") return true;
      }
    for (Value* op : J->operands())
      if (auto* OI = dyn_cast<Instruction>(op)) st.push_back(OI);
  }
  return false;
}

bool Executor::evalInst(Instruction& I, EP R) {
  curI = &I;
  if (reuseUniform && !I.mayHaveSideEffects() && !I.isTerminator() && !isa<CallInst>(&I))
    if (auto it = uniformEnv.find(&I); it != uniformEnv.end()) { env[&I] = it->second; return true; }
  if (stepBudget && ++steps > stepBudget) return fail("evaluation budget of " + std::to_string(stepBudget) + " instructions exceeded");
  if (wallBudget > 0 && ((steps & 63) == 0)) { const double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); if (now - wallStart > wallBudget) return fail("time budget of " + std::to_string((int)wallBudget) + " s exceeded"); }
  Type* ty = I.getType();
  if (auto* BO = dyn_cast<BinaryOperator>(&I)) {
    EP a = get(BO->getOperand(0)), b = get(BO->getOperand(1));
    if (ty->isVectorTy()) {
      // elementwise on vectors
      auto* VT = cast<FixedVectorType>(ty);
      unsigned n = VT->getNumElements(), esz = A.sizeOf(VT->getElementType());
      std::vector<EP> parts;
      for (unsigned i = 0; i < n; i++) parts.push_back(A.binop(BO->getOpcode(), A.fromBytes(std::vector<EP>(A.toBytes(a).begin() + i * esz, A.toBytes(a).begin() + (i + 1) * esz), VT->getElementType()),
                                                        A.fromBytes(std::vector<EP>(A.toBytes(b).begin() + i * esz, A.toBytes(b).begin() + (i + 1) * esz), VT->getElementType()), VT->getElementType()));
      env[&I] = A.concat(std::move(parts), ty);
      return true;
    }
    if (BO->getOpcode() == Instruction::Or && cast<PossiblyDisjointInst>(BO)->isDisjoint() && ty->isIntegerTy()) { env[&I] = A.add(a, b); return true; }
    env[&I] = A.binop(BO->getOpcode(), a, b, ty);
    return true;
  }
  if (auto* UO = dyn_cast<UnaryOperator>(&I)) {
    if (ty->isVectorTy()) return fail("vector unary operator in the epilogue");
    env[&I] = A.unop(UO->getOpcode(), get(UO->getOperand(0)), ty);
    return true;
  }
  if (auto* CI = dyn_cast<CastInst>(&I)) {
    EP a = get(CI->getOperand(0));
    if (ty->isVectorTy() && CI->getOpcode() != Instruction::BitCast) {
      auto* VT = cast<FixedVectorType>(ty); auto* ST = cast<FixedVectorType>(CI->getSrcTy());
      unsigned n = VT->getNumElements(), esz = A.sizeOf(ST->getElementType());
      std::vector<EP> parts; auto bs = A.toBytes(a);
      for (unsigned i = 0; i < n; i++) parts.push_back(A.cast(CI->getOpcode(), A.fromBytes(std::vector<EP>(bs.begin() + i * esz, bs.begin() + (i + 1) * esz), ST->getElementType()), VT->getElementType()));
      env[&I] = A.concat(std::move(parts), ty);
      return true;
    }
    if (CI->getOpcode() == Instruction::BitCast && A.sizeOf(ty) == A.sizeOf(CI->getSrcTy())) { env[&I] = A.fromBytes(A.toBytes(a), ty); return true; }
    env[&I] = A.cast(CI->getOpcode(), a, ty);
    return true;
  }
  if (auto* IC = dyn_cast<ICmpInst>(&I)) {
    if (ty->isVectorTy()) return fail("vector compare in the epilogue");
    // `samesign`: the operands' sign bits agree, so the unsigned order is the signed one; the signed literal is the
    // one the rest of the kernel wrote the same condition as (the row guard `n0 + r < N` next to `n0 >= N`)
    CmpInst::Predicate pred = IC->getPredicate();
    if (IC->hasSameSign() && CmpInst::isUnsigned(pred)) pred = ICmpInst::getSignedPredicate(pred);
    env[&I] = A.cmp(pred, A.asPoly(get(IC->getOperand(0))), A.asPoly(get(IC->getOperand(1))));
    return true;
  }
  if (auto* FC = dyn_cast<FCmpInst>(&I)) {
    if (ty->isVectorTy()) return fail("vector compare in the epilogue");
    env[&I] = A.cmp(FC->getPredicate(), get(FC->getOperand(0)), get(FC->getOperand(1)));
    return true;
  }
  if (auto* SI = dyn_cast<SelectInst>(&I)) {
    if (SI->getCondition()->getType()->isVectorTy()) return fail("vector select in the epilogue");
    env[&I] = A.ite(get(SI->getCondition()), get(SI->getTrueValue()), get(SI->getFalseValue()));
    return true;
  }
  if (auto* G = dyn_cast<GetElementPtrInst>(&I)) { env[&I] = gep(cast<GEPOperator>(*G), get(G->getPointerOperand())); return true; }
  if (auto* LD = dyn_cast<LoadInst>(&I)) {
    if (LD->isVolatile() || LD->isAtomic()) return fail("volatile/atomic load in the epilogue");
    std::vector<EP> bs;
    if (!loadBytes(get(LD->getPointerOperand()), A.sizeOf(ty), (unsigned)LD->getAlign().value(), R, &I, bs)) return false;
    env[&I] = A.fromBytes(bs, ty);
    return true;
  }
  if (auto* ST = dyn_cast<StoreInst>(&I)) {
    if (ST->isVolatile() || ST->isAtomic()) return fail("volatile/atomic store in the epilogue");
    return storeBytes(get(ST->getPointerOperand()), A.toBytes(get(ST->getValueOperand())), R, &I);
  }
  if (auto* EV = dyn_cast<ExtractValueInst>(&I)) {
    EP agg = get(EV->getAggregateOperand());
    if (EV->getNumIndices() == 1 && agg->k == EK::Concat && EV->getIndices()[0] < agg->args.size()) {
      env[&I] = agg->args[EV->getIndices()[0]];
      return true;
    }
    // skipped K-loop accumulator (a struct the skip named Unknown / ThreadValue): each field is a per-thread value
    if (EV->getNumIndices() == 1 && (ty->isFPOrFPVectorTy() || ty->isIntegerTy() || ty->isBFloatTy())) {
      Symbol s; s.kind = Symbol::ThreadValue; s.v = &I; s.thread = tid; s.ty = ty; s.field = EV->getIndices()[0];
      env[&I] = A.mkSym(s);
      return true;
    }
    return fail("extractvalue from an aggregate the executor does not model");
  }
  if (auto* EE = dyn_cast<ExtractElementInst>(&I)) {
    auto* idx = dyn_cast<ConstantInt>(EE->getIndexOperand());
    if (!idx) return fail("extractelement with a variable index");
    auto* VT = cast<FixedVectorType>(EE->getVectorOperandType());
    unsigned esz = A.sizeOf(VT->getElementType());
    auto bs = A.toBytes(get(EE->getVectorOperand()));
    unsigned i = (unsigned)idx->getZExtValue();
    if ((i + 1) * esz > bs.size()) return fail("extractelement out of range");
    env[&I] = A.fromBytes(std::vector<EP>(bs.begin() + i * esz, bs.begin() + (i + 1) * esz), VT->getElementType());
    return true;
  }
  if (auto* IE = dyn_cast<InsertElementInst>(&I)) {
    auto* idx = dyn_cast<ConstantInt>(IE->getOperand(2));
    if (!idx) return fail("insertelement with a variable index");
    auto* VT = cast<FixedVectorType>(ty);
    unsigned n = VT->getNumElements(), esz = A.sizeOf(VT->getElementType());
    auto bs = A.toBytes(get(IE->getOperand(0)));
    std::vector<EP> parts;
    for (unsigned i = 0; i < n; i++) {
      if (i == idx->getZExtValue()) parts.push_back(get(IE->getOperand(1)));
      else parts.push_back(A.fromBytes(std::vector<EP>(bs.begin() + i * esz, bs.begin() + (i + 1) * esz), VT->getElementType()));
    }
    env[&I] = A.concat(std::move(parts), ty);
    return true;
  }
  if (auto* SV = dyn_cast<ShuffleVectorInst>(&I)) {
    auto* VT = cast<FixedVectorType>(ty);
    auto* ST = cast<FixedVectorType>(SV->getOperand(0)->getType());
    unsigned n0 = ST->getNumElements(), esz = A.sizeOf(ST->getElementType());
    auto b0 = A.toBytes(get(SV->getOperand(0))), b1 = A.toBytes(get(SV->getOperand(1)));
    std::vector<EP> parts;
    for (int m : SV->getShuffleMask()) {
      if (m < 0) { parts.push_back(A.mkUndef(VT->getElementType())); continue; }
      auto& src = (unsigned)m < n0 ? b0 : b1; unsigned i = (unsigned)m % n0;
      parts.push_back(A.fromBytes(std::vector<EP>(src.begin() + i * esz, src.begin() + (i + 1) * esz), VT->getElementType()));
    }
    env[&I] = A.concat(std::move(parts), ty);
    return true;
  }
  if (isa<FreezeInst>(&I)) { env[&I] = get(I.getOperand(0)); return true; }
  if (isa<FenceInst>(&I)) return true;
  if (isa<AllocaInst>(&I)) {
    if (!retile) return fail("stack allocation in the kernel");
    Symbol s; s.kind = Symbol::Private; s.v = &I; s.thread = tid; s.ty = ty; env[&I] = A.mkSym(s);
    return true;
  }
    if (isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I)) {
    if (retile) {
      // a device read-modify-write: recorded as an atomic store effect (its address is what coordinate discovery fits)
      if (recordDevStores && !shadow) {
        Value* pv = isa<AtomicRMWInst>(&I) ? cast<AtomicRMWInst>(&I)->getPointerOperand() : cast<AtomicCmpXchgInst>(&I)->getPointerOperand();
        Value* vv = isa<AtomicRMWInst>(&I) ? cast<AtomicRMWInst>(&I)->getValOperand() : cast<AtomicCmpXchgInst>(&I)->getNewValOperand();
        EP addr = get(pv);
        // the bytes are the operand's (what the kernel's result may depend on through the location), not the result's
        if (addr) { DevStore ds{addr, A.toBytes(get(vv)), R, phase, tid, &I}; ds.atomic = true; devStores.push_back(ds); devStoreIters.push_back(iterStack); devStoreLoops.push_back(loopStack); }
      }
      if (!ty->isVoidTy()) env[&I] = unknown(ty); return true;
    }
    // Split-K arrival (decode.cuh) and decode.cu's shared atomics: the result is an unknown. When one thread
    // stores it to shared and the rest load it after a barrier, resolve() broadcasts that unknown as a uniform.
    if (!ty->isVoidTy()) env[&I] = unknown(ty);
    return true;
  }
  if (auto* CI = dyn_cast<CallInst>(&I)) {
    if (CI->isInlineAsm()) {
      if (retile) {
        std::vector<PtxInstr> ins; std::string err;
        auto* IA = cast<InlineAsm>(CI->getCalledOperand());
        if (parsePtxAsm(std::string(IA->getAsmString()), ins, err) && !ins.empty()) {
          if (ins.size() == 1 && ins[0].mnemonic == "cp" && ins[0].mods.size() >= 1 && ins[0].mods[0] == "async" && ins[0].hasMod("shared") && ins[0].hasMod("global") && !ins[0].hasMod("bulk")) return shadow ? true : modelCopy(CI, R);
          bool alu = !ty->isVoidTy();
          for (auto& in : ins) if (!ptxIsPerThreadAlu(in.mnemonic) || in.predicated()) alu = false;
          if (alu && !ty->isStructTy()) {
            std::vector<EP> args;
            for (unsigned i = 0; i < CI->arg_size(); i++) args.push_back(get(CI->getArgOperand(i)));
            env[&I] = A.callAsm(IA, std::move(args), ty);
            return true;
          }
        }
        if (!ty->isVoidTy()) env[&I] = unknown(ty);
        return true;
      }
      // staging copies and warp-level asm before the epilogue are irrelevant to it
      if (afterSkip) {
        std::vector<PtxInstr> ins; std::string err;
        auto* IA = cast<InlineAsm>(CI->getCalledOperand());
        std::string text = std::string(IA->getAsmString());
        if (parsePtxAsm(text, ins, err) && !ins.empty()) {
          // $N → call arg: outputs first in the constraint string, then inputs (msl_emitter.cpp)
          std::string cons = std::string(IA->getConstraintString());
          std::vector<std::string> cparts; { std::stringstream ss(cons); std::string p; while (std::getline(ss, p, ',')) cparts.push_back(p); }
          unsigned nOut = 0; std::vector<int> inputArg, tied;
          { unsigned argIdx = 0; for (auto& p : cparts) {
            if (p.empty() || p[0] == '~') continue;
            if (p[0] == '=' || p[0] == '+') { nOut++; tied.push_back(-1); continue; }
            if (isdigit((unsigned char)p[0])) { unsigned o = (unsigned)atoi(p.c_str()); if (o < tied.size()) tied[o] = (int)argIdx; }
            inputArg.push_back((int)argIdx++);
          } }
          auto inArg = [&](int k) -> Value* {
            if (k < 0) return nullptr;
            if ((unsigned)k < nOut) return (unsigned)k < tied.size() && tied[k] >= 0 && (unsigned)tied[k] < CI->arg_size() ? CI->getArgOperand((unsigned)tied[k]) : nullptr;
            unsigned i = (unsigned)k - nOut;
            return i < inputArg.size() && (unsigned)inputArg[i] < CI->arg_size() ? CI->getArgOperand((unsigned)inputArg[i]) : nullptr;
          };
          bool alu = !ty->isVoidTy();
          for (auto& in : ins) if (!ptxIsPerThreadAlu(in.mnemonic) || in.predicated()) alu = false;
          if (alu && !ty->isStructTy()) {
            std::vector<EP> args;
            for (unsigned i = 0; i < CI->arg_size(); i++) args.push_back(get(CI->getArgOperand(i)));
            env[&I] = A.callAsm(IA, std::move(args), ty);
            return true;
          }
          if (ins[0].mnemonic == "cp" && !ins[0].mods.empty() && ins[0].mods[0] == "async")
            return true;  // leftover copy / wait_all / commit_group after the skipped ring
          if (ins.size() == 1 && !ins[0].predicated() && ins[0].mnemonic == "ld" && ins[0].hasMod("global") && !ty->isVoidTy()) {
            // split-K last arriver: __ldcg of C_partial (A.load, not loadBytes: this load is ordered after the
            // partial stores by a barrier, which loadBytes refuses as a race with the C-tile copy-out)
            int areg = -1;
            for (const PtxOperand& op : ins[0].ops) if (op.kind == PtxOperand::Mem) { areg = op.reg; break; }
            Value* addrV = areg >= 0 ? inArg(areg) : (CI->arg_size() == 1 ? CI->getArgOperand(0) : nullptr);
            if (addrV) { env[&I] = A.load(ty, get(addrV), R, A.sizeOf(ty)); return true; }
          }
          if (ins.size() == 1 && !ins[0].predicated() && ins[0].mnemonic == "st" && ins[0].hasMod("global") && CI->arg_size() >= 2) {
            int areg = -1, vreg = -1;
            for (const PtxOperand& op : ins[0].ops) {
              if (op.kind == PtxOperand::Mem && areg < 0) areg = op.reg;
              else if (op.kind == PtxOperand::Reg && vreg < 0) vreg = op.reg;
            }
            Value* addrV = areg >= 0 ? inArg(areg) : nullptr;
            Value* valV = vreg >= 0 ? inArg(vreg) : nullptr;
            if (addrV && valV) {
              EP v = get(valV);
              if (!storeBytes(get(addrV), A.toBytes(v), R, CI)) return false;
              return true;
            }
          }
        }
        return fail("inline asm in the epilogue (" + (err.empty() ? text : err) + ")");
      }
      if (!ty->isVoidTy()) env[&I] = unknown(ty);
      return true;
    }
    Function* Callee = CI->getCalledFunction();
    if (!Callee) return fail("indirect call");
    StringRef n = Callee->getName();
    if (isBarrierName(n)) {
      if (shadow) return true;
      if (!A.isTrue(R) && !retile && !uniformAcrossBlock(A, R)) {
        std::string d; raw_string_ostream os(d); os << "a barrier under a data-dependent condition (block "; I.getParent()->printAsOperand(os, false); os << ")";
        return fail(d);
      }
      phase++;
      // First post-skip barrier only: later stream-K / s_last barriers must not
      // change the candidate, so every thread agrees on Be even if only tid 0
      // writes shared memory (decode.cuh s_last).
      if (afterSkip && !Be && !pendingBe) { pendingBe = &I; pendingBePhase = phase; pendingBeCond = R; }
      return true;
    }
    if (n == "llvm.nvvm.read.ptx.sreg.tid.x") { env[&I] = A.mkInt(ty, tid); return true; }
    if (n == "llvm.nvvm.read.ptx.sreg.tid.y" || n == "llvm.nvvm.read.ptx.sreg.tid.z") { env[&I] = A.mkInt(ty, 0); return true; }
    if (n == "llvm.nvvm.read.ptx.sreg.ntid.x") { env[&I] = A.mkInt(ty, T); return true; }
    if (n == "llvm.nvvm.read.ptx.sreg.ntid.y" || n == "llvm.nvvm.read.ptx.sreg.ntid.z") { env[&I] = A.mkInt(ty, 1); return true; }
    if (n == "llvm.nvvm.read.ptx.sreg.laneid") { env[&I] = A.mkInt(ty, tid & 31); return true; }
    if (n == "llvm.nvvm.read.ptx.sreg.warpsize") { env[&I] = A.mkInt(ty, 32); return true; }
    if (n.starts_with("llvm.nvvm.read.ptx.sreg.")) { Symbol s; s.kind = Symbol::Sreg; s.name = n.str(); s.ty = ty; env[&I] = A.mkSym(s); return true; }
    if (retile && n.starts_with("__mvcc_cp_async_zsel_")) return shadow ? true : modelCopy(CI, R);
    // Cross-lane operations: a shuffle is the value another lane computed - an uninterpreted function of
    // (value, source-lane expression, ...) until lane re-execution resolves it;
    // a vote is a reduction over the lane coordinate. Both keep the recurrence structure around them visible
    // (v += shfl(v, off) is a sum over lanes), where an opaque Unknown would hide it. Also in the epilogue: the
    // split-K path and a skipped inner K loop can still reach a quantizer's xor-max.
    if (n.starts_with("llvm.nvvm.shfl.sync.") || n.starts_with("llvm.nvvm.vote.") || n.starts_with("llvm.nvvm.match.") || n.starts_with("llvm.nvvm.redux.sync.")) {
      std::vector<EP> args;
      for (unsigned i = 0; i < CI->arg_size(); i++) args.push_back(get(CI->getArgOperand(i)));
      if (ty->isStructTy()) {
        std::vector<EP> f;
        for (unsigned i = 0; i < ty->getStructNumElements(); i++) f.push_back(A.call(Callee, args, ty->getStructElementType(i)));
        env[&I] = A.concat(std::move(f), ty);
      } else if (!ty->isVoidTy()) env[&I] = A.call(Callee, std::move(args), ty);
      return true;
    }
    if (n.starts_with("__mvcc_tp_") || n.starts_with("__mvcc_cp_async") || n.starts_with("llvm.nvvm.cp.async")) {
      unsigned ord = 0;
      if (retile && n.starts_with("__mvcc_tp_") && !shadow) {
        ord = ++tpOrd;
        TpCall tc{&I, tid, curIter, {}, R, phase, ord};
        for (unsigned i = 0; i < CI->arg_size(); i++) tc.args.push_back(get(CI->getArgOperand(i)));
        tpCalls.push_back(std::move(tc));
      }
      if (afterSkip && !retile && !n.starts_with("__mvcc_tp_acc2ptx") && !n.starts_with("__mvcc_tp_ptx2acc")) return fail("a tensor operation or staging copy in the epilogue (" + n.str() + ")");
      if (ty->isVoidTy()) return true;
      if (shadow && n.starts_with("__mvcc_tp_mma_")) {
        // recurrence pass: the matmul stays a Call over its operands so that an accumulator phi's next value reads
        // mma(a, b, phi) - the shape the algebra classifier recognizes as a sum over slot words
        std::vector<EP> args;
        for (unsigned i = 0; i < CI->arg_size(); i++) args.push_back(get(CI->getArgOperand(i)));
        if (ty->isStructTy()) {
          // one Call per field; the field index is appended as a trailing constant operand
          std::vector<EP> f;
          for (unsigned i = 0; i < ty->getStructNumElements(); i++) { auto a2 = args; a2.push_back(A.mkInt(Type::getInt32Ty(F.getContext()), i)); f.push_back(A.call(Callee, std::move(a2), ty->getStructElementType(i))); }
          env[&I] = A.concat(std::move(f), ty);
        } else env[&I] = A.call(Callee, std::move(args), ty);
        return true;
      }
      if (retile && n.starts_with("__mvcc_tp_widen_") && ty->isStructTy()) {
        // the 8-bit widening of an operand word is a pure function of the word: one Call per result field (the field
        // index a trailing constant operand, as the recurrence pass writes a matmul), so that an operand built from
        // staged fp8 bytes stays a function of their source (the decode retiler fits it; Materializer emits the call)
        std::vector<EP> f;
        for (unsigned i = 0; i < ty->getStructNumElements(); i++) {
          std::vector<EP> args;
          for (unsigned j = 0; j < CI->arg_size(); j++) args.push_back(get(CI->getArgOperand(j)));
          args.push_back(A.mkInt(Type::getInt32Ty(F.getContext()), i));
          f.push_back(A.call(Callee, std::move(args), ty->getStructElementType(i)));
        }
        env[&I] = A.concat(std::move(f), ty);
        return true;
      }
      if (ty->isStructTy()) {
        // retile mode: an operand fill's words name the execution that produced them (name = the call's ordinal in
        // this thread's run), so that a word consumed one iteration after it was filled (a software-pipelined
        // prefetch) is traced to the right fill. Other results keep one symbol per instruction.
        const bool perExec = retile && !shadow && n.starts_with("__mvcc_tp_ld");
        std::vector<EP> f;
        for (unsigned i = 0; i < ty->getStructNumElements(); i++) { Symbol s; s.kind = Symbol::ThreadValue; s.v = &I; s.thread = tid; s.field = i; s.ty = ty->getStructElementType(i); if (perExec) s.name = std::to_string(ord); f.push_back(A.mkSym(s)); }
        env[&I] = A.concat(std::move(f), ty);
        if (std::find(acc2ptxSites.begin(), acc2ptxSites.end(), &I) == acc2ptxSites.end()) acc2ptxSites.push_back(&I);
        return true;
      }
      Symbol s; s.kind = Symbol::ThreadValue; s.v = &I; s.thread = tid; s.ty = ty; env[&I] = A.mkSym(s);
      return true;
    }
    if (n == "__mvcc_zero_page") { env[&I] = A.call(Callee, {}, ty); return true; }
    if (auto* MI = dyn_cast<MemIntrinsic>(CI)) {
      auto* Len = dyn_cast<ConstantInt>(MI->getLength());
      if (!Len) return fail("memcpy/memset of non-constant length");
      unsigned bytes = (unsigned)Len->getZExtValue();
      std::vector<EP> bs;
      if (auto* MT = dyn_cast<MemTransferInst>(MI)) { if (!loadBytes(get(MT->getRawSource()), bytes, MT->getSourceAlign() ? (unsigned)MT->getSourceAlign()->value() : 1, R, &I, bs)) return false; }
      else { EP v = get(cast<MemSetInst>(MI)->getValue()); bs.assign(bytes, A.retype(A.asPoly(v), Type::getInt8Ty(F.getContext()))); }
      return storeBytes(get(MI->getRawDest()), bs, R, &I);
    }
    if (isa<DbgInfoIntrinsic>(CI) || n.starts_with("llvm.lifetime") || n.starts_with("llvm.assume") || n.starts_with("llvm.experimental.noalias")) return true;
    if (n.starts_with("llvm.nvvm.membar") || n.starts_with("llvm.nvvm.fence")) return true;  // ordering only; the SIG orders effects by barrier phase
    if (Callee->onlyReadsMemory() || Callee->doesNotAccessMemory() || CI->doesNotAccessMemory() || CI->onlyReadsMemory() || (Callee->isDeclaration() && n.starts_with("__nv_"))) {
      if (CI->onlyReadsMemory() && !CI->doesNotAccessMemory() && afterSkip) return fail("a memory-reading call the executor does not model (" + n.str() + ")");
      std::vector<EP> args;
      for (unsigned i = 0; i < CI->arg_size(); i++) args.push_back(get(CI->getArgOperand(i)));
      // constant folding of the integer min/max family keeps addresses closed
      int64_t x, y;
      if (args.size() == 2 && ty->isIntegerTy() && A.constInt(args[0], x) && A.constInt(args[1], y)) {
        if (n.starts_with("llvm.smax.")) { env[&I] = A.mkInt(ty, std::max(x, y)); return true; }
        if (n.starts_with("llvm.smin.")) { env[&I] = A.mkInt(ty, std::min(x, y)); return true; }
        if (n.starts_with("llvm.umax.")) { env[&I] = A.mkInt(ty, (int64_t)std::max((uint64_t)x, (uint64_t)y)); return true; }
        if (n.starts_with("llvm.umin.")) { env[&I] = A.mkInt(ty, (int64_t)std::min((uint64_t)x, (uint64_t)y)); return true; }
        if (n == "__nv_min") { env[&I] = A.mkInt(ty, std::min(x, y)); return true; }
        if (n == "__nv_max") { env[&I] = A.mkInt(ty, std::max(x, y)); return true; }
      }
      if (!ty->isVoidTy()) env[&I] = A.call(Callee, std::move(args), ty);
      return true;
    }
    return fail("a call the executor does not model (" + n.str() + ")");
  }
  return fail(std::string("an instruction the executor does not model (") + I.getOpcodeName() + ")");
}

bool Executor::execBlock(BasicBlock* BB, std::vector<Edge>& edges, const Loop* region, EdgeMap& out) {
  if (edges.empty()) return true;  // unreachable under every path taken so far
  EP R = A.mkBool(false);
  for (auto& e : edges) R = A.orb(R, e.cond);
  if (A.isFalse(R)) return true;
  // phis of this block, and values of loops the edges left (their exit values differ per exit edge)
  // in the function's order, not by address: the merges are built in this order and the expressions they create
  // are numbered by it, and the arena orders literals by that number
  if (valueRank.empty()) { unsigned n = 0; for (Argument& a : F.args()) valueRank[&a] = n++; for (Instruction& I : instructions(F)) valueRank[&I] = n++; }
  std::vector<Value*> keys;
  { std::set<Value*> ks; for (auto& e : edges) for (auto& kv : e.snap) ks.insert(kv.first); keys.assign(ks.begin(), ks.end()); }
  std::sort(keys.begin(), keys.end(), [&](Value* x, Value* y) { auto ix = valueRank.find(x), iy = valueRank.find(y); const unsigned rx = ix == valueRank.end() ? ~0u : ix->second, ry = iy == valueRank.end() ? ~0u : iy->second; return rx != ry ? rx < ry : x < y; });
  // the merged values are only used under R: the literals every edge's condition shares (the path down to the
  // branch that split them) are left out of the selections (retile mode, whose expressions are compared across
  // lanes and iterations)
  std::vector<EP> sel(edges.size());
  for (size_t i = 0; i < edges.size(); i++) sel[i] = edges[i].cond;
  if (retile && edges.size() > 1) {
    std::vector<std::vector<EP>> lits(edges.size());
    for (size_t i = 0; i < edges.size(); i++) A.conjuncts(edges[i].cond, lits[i]);
    std::set<EP> common(lits[0].begin(), lits[0].end());
    for (size_t i = 1; i < edges.size(); i++) { std::set<EP> keep; for (EP l : lits[i]) if (common.count(l)) keep.insert(l); common = std::move(keep); }
    if (!common.empty()) for (size_t i = 0; i < edges.size(); i++) { EP c = A.mkBool(true); for (EP l : lits[i]) if (!common.count(l)) c = A.andb(c, l); sel[i] = c; }
  }
  for (Value* key : keys) {
    if (opaqueValues.count(key)) { Symbol s; s.kind = Symbol::LoopValue; s.v = key; s.ty = key->getType(); s.thread = 0; env[key] = A.mkSym(s); continue; }   // an opaque loop's result: the value itself on every path
    auto val = [&](Edge& e) -> EP { auto it = e.snap.find(key); if (it != e.snap.end()) return it->second; return A.mkUndef(key->getType()); };
    EP v = val(edges.back());
    for (size_t i = edges.size() - 1; i-- > 0;) v = A.ite(sel[i], val(edges[i]), v);
    env[key] = v;
  }
  for (Instruction& I : *BB) {
    if (isa<PHINode>(&I)) continue;
    if (I.isTerminator()) break;
    if (!evalInst(I, R)) return false;
  }
  Instruction* Tm = BB->getTerminator();
  if (auto* Br = dyn_cast<BranchInst>(Tm)) {
    if (Br->isUnconditional()) { edgeTo(BB, Br->getSuccessor(0), R, out); return true; }
    EP c = get(Br->getCondition());
    if (!retile && A.containsSymbol(c, Symbol::Unknown) && !uniformAcrossBlock(A, c)) return fail("a branch condition depends on a value of the skipped loop");
    if (region && (generic.count(region) || region == iterL || traversal.count(region))) {
      // A loop-exiting branch of a generically executed loop: the body is modeled under the assumption that the
      // iteration executes, recorded as the condition of the edge that stays inside. The exit edge of a generic loop
      // is taken under the loop's entry condition (the loop terminates: whenever it was entered it is left, in some
      // iteration - the exit's per-iteration literal is not a condition on reaching the code after the loop); the
      // iterated loop's exit is taken in its last iteration only.
      bool out0 = !region->contains(Br->getSuccessor(0)), out1 = !region->contains(Br->getSuccessor(1));
      if (out0 != out1) {
        const unsigned exitIdx = out0 ? 0 : 1, inIdx = 1 - exitIdx;
        if (region == iterL || traversal.count(region)) { unsigned keep = iterLast ? exitIdx : inIdx; edgeTo(BB, Br->getSuccessor(keep), A.andb(R, keep ? A.notb(c) : c), out); return true; }
        edgeTo(BB, Br->getSuccessor(inIdx), A.andb(R, inIdx ? A.notb(c) : c), out);
        auto ge = genericEntry.find(region);
        edgeTo(BB, Br->getSuccessor(exitIdx), ge != genericEntry.end() ? ge->second : R, out);
        return true;
      }
    }
    EP ct = A.andb(R, c), cf = A.andb(R, A.notb(c));
    if (!A.isFalse(ct)) edgeTo(BB, Br->getSuccessor(0), ct, out);
    if (!A.isFalse(cf)) edgeTo(BB, Br->getSuccessor(1), cf, out);
    return true;
  }
  if (auto* SW = dyn_cast<SwitchInst>(Tm)) {
    // each case under R && x == v; the default under R && no case matched (LowerSwitch runs later in the pipeline)
    EP x = A.asPoly(get(SW->getCondition()));
    EP none = A.mkBool(true);
    for (auto& C : SW->cases()) {
      EP eq = A.cmp(CmpInst::ICMP_EQ, x, A.mkInt(x->ty, C.getCaseValue()->getSExtValue()));
      EP cc = A.andb(R, eq);
      if (!A.isFalse(cc)) edgeTo(BB, C.getCaseSuccessor(), cc, out);
      none = A.andb(none, A.notb(eq));
    }
    EP cd = A.andb(R, none);
    if (!A.isFalse(cd)) edgeTo(BB, SW->getDefaultDest(), cd, out);
    return true;
  }
  if (isa<ReturnInst>(Tm)) {
    if (!A.isTrue(R) && !retile && !uniformAcrossBlock(A, R)) return fail("a return under a data-dependent condition: " + A.str(R, 8));
    // per incoming edge: the return blocks of a kernel are usually one (the exits are unified), and what tells an
    // early return from the epilogue's is the block the path came from. The epilogue's own returns are left out
    // (their conditions are the K loop's exit conditions, deep case splits whose disjunction costs more than it
    // could say: nothing before the loop is ordered after them anyway).
    if (retile && !shadow) for (Edge& e : edges) {
      BasicBlock* from = e.from ? e.from : BB;
      auto r = fromEpilogue.find(from);
      if (r == fromEpilogue.end()) r = fromEpilogue.emplace(from, iterL && isPotentiallyReachable(iterL->getHeader(), from, nullptr, &DT, &LI)).first;
      if (r->second) continue;
      EP& c = returned[{tid, from}]; c = c ? A.orb(c, e.cond) : e.cond;
    }
    return true;
  }
  if (isa<UnreachableInst>(Tm)) return true;
  return fail(std::string("a terminator the executor does not model (") + Tm->getOpcodeName() + ")");
}

// Blocks of `region` (the function when L is null) in a topological order of the acyclic graph whose nodes are the
// region's own blocks and its immediate sub-loops (represented by their headers), from `entry`.
static void regionOrder(BasicBlock* entry, const Loop* L, LoopInfo& LI, std::vector<BasicBlock*>& order) {
  std::set<BasicBlock*> seen;
  std::function<void(BasicBlock*)> dfs = [&](BasicBlock* BB) {
    if (!seen.insert(BB).second) return;
    const Loop* inner = LI.getLoopFor(BB);
    // the node's successors: if BB heads a sub-loop, the sub-loop's exits; else BB's successors
    std::vector<BasicBlock*> succs;
    if (inner && inner != L && (!L || L->contains(inner)) ) {
      // find the immediate sub-loop of L containing BB
      const Loop* sub = inner;
      while (sub->getParentLoop() != L) sub = sub->getParentLoop();
      if (sub->getHeader() != BB) return;  // interior block of a sub-loop: not a node at this level
      SmallVector<BasicBlock*, 8> exits;
      sub->getExitBlocks(exits);
      succs.assign(exits.begin(), exits.end());
    } else {
      for (BasicBlock* S : successors(BB)) succs.push_back(S);
    }
    for (BasicBlock* S : succs) {
      if (L && !L->contains(S)) continue;          // leaving the region
      if (L && S == L->getHeader()) continue;      // back edge
      dfs(S);
    }
    order.push_back(BB);
  };
  dfs(entry);
  std::reverse(order.begin(), order.end());
}

bool Executor::execRegion(BasicBlock* entry, const Loop* L, EdgeMap& in, EdgeMap& out) {
  std::vector<BasicBlock*> order;
  regionOrder(entry, L, LI, order);
  for (BasicBlock* BB : order) {
    if (stopAt.count(BB)) { in.erase(BB); continue; }
    const Loop* inner = LI.getLoopFor(BB);
    if (inner && inner != L && (!L || L->contains(inner))) {
      const Loop* sub = inner;
      while (sub->getParentLoop() != L) sub = sub->getParentLoop();
      auto edges = std::move(in[BB]); in.erase(BB);
      if (edges.empty()) continue;
      bool ok = sub == skip ? skipLoop(sub, std::move(edges), in) : generic.count(sub) ? genericLoop(sub, std::move(edges), in) : (sub == iterL || traversal.count(sub)) ? iterLoop(sub, std::move(edges), in) : execLoop(sub, std::move(edges), in);
      if (!ok) return false;
      // the sub-loop's exits that leave L as well go to `out`
      if (L) for (auto it = in.begin(); it != in.end();) { if (!L->contains(it->first)) { auto& v = out[it->first]; v.insert(v.end(), it->second.begin(), it->second.end()); it = in.erase(it); } else ++it; }
      continue;
    }
    auto edges = std::move(in[BB]); in.erase(BB);
    EdgeMap local;
    if (!execBlock(BB, edges, L, local)) return false;
    for (auto& kv : local) {
      BasicBlock* to = kv.first;
      auto& dst = (L && (!L->contains(to) || to == L->getHeader())) ? out[to] : in[to];
      dst.insert(dst.end(), kv.second.begin(), kv.second.end());
    }
  }
  return true;
}

bool Executor::execLoop(const Loop* L, std::vector<Edge> headerEdges, EdgeMap& out) {
  BasicBlock* H = L->getHeader();
  for (unsigned iter = 0;; iter++) {
    if (iter > maxUnroll) {
      if (afterSkip) break;  // split-K reduction over a launch argument; the C-tile stores already executed
      // Setup before K (int4 experts scale copy over K_groups): one pass is enough
      // to continue to skip. A loop that contains the MMA must keep unrolling.
      bool mma = false;
      for (const BasicBlock* B : L->blocks()) {
        for (const Instruction& I : *B) {
          auto* CI = dyn_cast<const CallInst>(&I);
          if (CI && CI->getCalledFunction() && CI->getCalledFunction()->getName().starts_with("__mvcc_tp_mma")) {
            mma = true; break;
          }
        }
        if (mma) break;
      }
      if (!mma) break;
      std::string hs; raw_string_ostream os(hs); os << *H->getFirstNonPHI(); return fail("a loop in the epilogue does not unroll within " + std::to_string(maxUnroll) + " iterations (header:" + StringRef(os.str()).trim().str() + ")");
    }
    EdgeMap in, exits;
    in[H] = std::move(headerEdges);
    if (!execRegion(H, L, in, exits)) return false;
    std::vector<Edge> back;
    for (auto& kv : exits) {
      if (kv.first == H) { back.insert(back.end(), kv.second.begin(), kv.second.end()); continue; }
      auto& dst = out[kv.first];
      dst.insert(dst.end(), kv.second.begin(), kv.second.end());
    }
    EP c = A.mkBool(false);
    for (auto& e : back) c = A.orb(c, e.cond);
    if (A.isFalse(c)) break;
    // a reduction over a launch argument (decode.cuh `s < k_splits`): further iterations repeat the same
    // store shape. Per-thread copy-out loops stay data-dependent and still unroll.
    if (afterSkip && !A.isTrue(c) && uniformAcrossBlock(A, c)) break;
    headerEdges = std::move(back);
  }
  // exit-block phis and live-outs are defined from the edge snapshots when the exit blocks execute
  return true;
}

bool Executor::skipLoop(const Loop* L, std::vector<Edge> headerEdges, EdgeMap& out) {
  BasicBlock* X = L->getExitBlock();
  if (!X) return fail("the recovered K loop has several exit blocks");
  EP R = A.mkBool(false);
  for (auto& e : headerEdges) R = A.orb(R, e.cond);
  // integer / pointer values defined in the loop become unknown; floating-point live-outs are the
  // accumulator fragments the epilogue stores (decode.cuh writes C before any barrier when k_splits==1).
  // Naming the IR value is the same ThreadValue rule get() applies once Be is set.
  Edge e{L->getLoopLatch() ? L->getLoopLatch() : L->getHeader(), R, {}};
  SmallVector<BasicBlock*, 4> exiting;
  L->getExitingBlocks(exiting);
  auto asThread = [&](Value* v) {
    Symbol s; s.kind = Symbol::ThreadValue; s.v = v; s.thread = tid; s.ty = v->getType();
    return A.mkSym(s);
  };
  // Address math sunk into the K loop (row, col, pass bounds) is a function of tid and the arguments;
  // replay it so the epilogue stores are not refused as depending on the skipped IV.
  std::map<Value*, EP> replayed;
  std::function<EP(Value*)> replay = [&](Value* v) -> EP {
    if (auto it = replayed.find(v); it != replayed.end()) return it->second;
    auto bind = [&](EP e) { replayed[v] = e; return e; };
    if (auto* Cst = dyn_cast<Constant>(v)) return bind(evalConstant(Cst));
    if (isa<Argument>(v) || isa<GlobalVariable>(v)) return bind(get(v));
    auto* I = dyn_cast<Instruction>(v);
    if (!I) return bind(get(v));
    if (!L->contains(I->getParent())) return bind(get(v));
    if (I->getType()->isFPOrFPVectorTy()) return bind(asThread(I));
    if (auto* CI = dyn_cast<CallInst>(I)) {
      Function* Callee = CI->getCalledFunction();
      StringRef n = Callee ? Callee->getName() : "";
      Type* ty = I->getType();
      if (n == "llvm.nvvm.read.ptx.sreg.tid.x") return bind(A.mkInt(ty, tid));
      if (n == "llvm.nvvm.read.ptx.sreg.tid.y" || n == "llvm.nvvm.read.ptx.sreg.tid.z") return bind(A.mkInt(ty, 0));
      if (n == "llvm.nvvm.read.ptx.sreg.ntid.x") return bind(A.mkInt(ty, T));
      if (n == "llvm.nvvm.read.ptx.sreg.ntid.y" || n == "llvm.nvvm.read.ptx.sreg.ntid.z") return bind(A.mkInt(ty, 1));
      if (n == "llvm.nvvm.read.ptx.sreg.laneid") return bind(A.mkInt(ty, tid & 31));
      if (n == "llvm.nvvm.read.ptx.sreg.warpsize") return bind(A.mkInt(ty, 32));
      if (n.starts_with("llvm.nvvm.read.ptx.sreg.")) { Symbol s; s.kind = Symbol::Sreg; s.name = n.str(); s.ty = ty; return bind(A.mkSym(s)); }
    }
    if (I->isBinaryOp()) {
      EP a = replay(I->getOperand(0)), b = replay(I->getOperand(1));
      if (A.containsSymbol(a, Symbol::Unknown) || A.containsSymbol(b, Symbol::Unknown)) return bind(unknown(I->getType()));
      return bind(A.binop(I->getOpcode(), a, b, I->getType()));
    }
    if (auto* C = dyn_cast<CastInst>(I)) {
      EP a = replay(C->getOperand(0));
      if (A.containsSymbol(a, Symbol::Unknown)) return bind(unknown(C->getType()));
      return bind(A.cast(C->getOpcode(), a, C->getType()));
    }
    if (auto* IC = dyn_cast<ICmpInst>(I)) {
      EP a = replay(IC->getOperand(0)), b = replay(IC->getOperand(1));
      if (A.containsSymbol(a, Symbol::Unknown) || A.containsSymbol(b, Symbol::Unknown)) return bind(unknown(IC->getType()));
      return bind(A.cmp(IC->getPredicate(), A.asPoly(a), A.asPoly(b)));
    }
    if (auto* S = dyn_cast<SelectInst>(I)) {
      EP c = replay(S->getCondition()), t = replay(S->getTrueValue()), f = replay(S->getFalseValue());
      if (A.containsSymbol(c, Symbol::Unknown) || A.containsSymbol(t, Symbol::Unknown) || A.containsSymbol(f, Symbol::Unknown)) return bind(unknown(S->getType()));
      return bind(A.ite(c, t, f));
    }
    if (auto* G = dyn_cast<GetElementPtrInst>(I)) {
      EP p = replay(G->getPointerOperand());
      if (A.containsSymbol(p, Symbol::Unknown)) return bind(unknown(G->getType()));
      Type* i64 = Type::getInt64Ty(F.getContext());
      p = A.retype(A.asPoly(p), G->getType());
      for (gep_type_iterator GTI = gep_type_begin(G), E = gep_type_end(G); GTI != E; ++GTI) {
        Value* idx = GTI.getOperand();
        if (StructType* ST = GTI.getStructTypeOrNull()) {
          p = A.add(p, A.mkInt(p->ty, (int64_t)A.DL.getStructLayout(ST)->getElementOffset((unsigned)cast<ConstantInt>(idx)->getZExtValue())));
          continue;
        }
        EP iv = replay(idx);
        if (A.containsSymbol(iv, Symbol::Unknown)) return bind(unknown(G->getType()));
        int64_t esz = (int64_t)A.DL.getTypeAllocSize(GTI.getIndexedType());
        p = A.add(p, A.mul(A.retype(A.cast(Instruction::SExt, iv, i64), p->ty), A.mkInt(p->ty, esz)));
      }
      return bind(p);
    }
    if (auto* PN = dyn_cast<PHINode>(I)) {
      if (PN->getParent() == L->getHeader()) return bind(unknown(PN->getType()));
      EP v = nullptr;
      for (unsigned i = 0; i < PN->getNumIncomingValues(); i++) {
        EP cur = replay(PN->getIncomingValue(i));
        if (A.containsSymbol(cur, Symbol::Unknown)) return bind(unknown(PN->getType()));
        v = v ? (v == cur ? v : unknown(PN->getType())) : cur;
        if (A.containsSymbol(v, Symbol::Unknown)) return bind(unknown(PN->getType()));
      }
      return bind(v ? v : unknown(PN->getType()));
    }
    return bind(unknown(I->getType()));
  };
  for (PHINode& PN : X->phis()) {
    if (PN.getType()->isFPOrFPVectorTy()) { e.snap[&PN] = asThread(&PN); continue; }
    EP v = nullptr;
    for (unsigned i = 0; i < PN.getNumIncomingValues(); i++) {
      if (!L->contains(PN.getIncomingBlock(i))) continue;
      v = v ? (v == replay(PN.getIncomingValue(i)) ? v : unknown(PN.getType())) : replay(PN.getIncomingValue(i));
    }
    e.snap[&PN] = v ? v : unknown(PN.getType());
  }
  for (Value* v : liveOuts(L)) e.snap[v] = v->getType()->isFPOrFPVectorTy() ? asThread(v) : replay(v);
  afterSkip = true;
  phase++;  // the loop's barriers: an unknown number of phases, all before the epilogue
  out[X].push_back(std::move(e));
  return true;
}

// ---------------------------------------------------------------- retile mode

bool Executor::genericLoop(const Loop* L, std::vector<Edge> headerEdges, EdgeMap& out) {
  BasicBlock* H = L->getHeader();
  EP R = A.mkBool(false);
  for (auto& e : headerEdges) R = A.orb(R, e.cond);
  Edge e{nullptr, R, {}};
  const bool uniform = generic.at(L);
  for (PHINode& PN : H->phis()) {
    const bool u = uniform && !threadPhis.count(&PN);
    Symbol s; s.kind = u ? Symbol::LoopValue : Symbol::ThreadValue; s.v = &PN; s.ty = PN.getType(); s.thread = u ? 0 : tid;
    e.snap[&PN] = A.mkSym(s);
  }
  EdgeMap in, exits;
  in[H].push_back(std::move(e));
  genericEntry[L] = R;
  const unsigned entryPhase = phase;
  if (!execRegion(H, L, in, exits)) return false;
  genericPhases[L] = {entryPhase, phase};
  const bool opaque = opaqueExits.count(L);
  for (auto& kv : exits) {
    if (kv.first == H) continue;
    if (opaque) for (Edge& x : kv.second) for (auto& sv : x.snap) {
      // the exit phi / live-out itself, as a uniform symbol (its merge at the exit block keeps it: opaqueValues)
      Symbol s; s.kind = Symbol::LoopValue; s.v = sv.first; s.ty = sv.first->getType(); s.thread = 0;
      sv.second = A.mkSym(s);
      opaqueValues.insert(sv.first);
    }
    auto& dst = out[kv.first]; dst.insert(dst.end(), kv.second.begin(), kv.second.end());
  }
  return true;
}

bool Executor::iterLoop(const Loop* L, std::vector<Edge> headerEdges, EdgeMap& out) {
  BasicBlock* H = L->getHeader();
  if (recordRecurrences && !shadow && !recurrencePass(L, headerEdges)) { recurrenceWhy[L] = why; why.clear(); }  // informational: the real pass decides
  // nested traversal loops: the enclosing loop's iteration state is restored when this one is left
  const unsigned savedIter = curIter; const bool savedLast = iterLast;
  iterStack.push_back(0); loopStack.push_back(L);
  for (unsigned it = 0; it < iterCount; it++) {
    curIter = it; iterLast = it + 1 == iterCount; iterStack.back() = it;
    EdgeMap in, exits;
    in[H] = std::move(headerEdges);
    if (!execRegion(H, L, in, exits)) return false;
    headerEdges.clear();
    for (auto& kv : exits) {
      if (kv.first == H) { headerEdges.insert(headerEdges.end(), kv.second.begin(), kv.second.end()); continue; }
      auto& dst = out[kv.first]; dst.insert(dst.end(), kv.second.begin(), kv.second.end());
    }
    if (!iterLast && headerEdges.empty()) {
      if (L == iterL) return fail("the iterated loop has no back edge");
      break;  // a further traversal loop that ended before the probe count: fewer iterations than probes
    }
  }
  iterStack.pop_back(); loopStack.pop_back();
  curIter = savedIter; iterLast = savedLast;
  return true;
}

// One effect-free pass of L's body with every header phi a free per-thread symbol: the back-edge values are the
// recurrences `next(phi_1, ..., phi_n, iteration values)`; the entry values are `init`. Nothing this pass evaluates is
// recorded (shadow): shared reads are opaque, stores / copies / barriers / tensor calls leave no trace, and the phase
// and sequence counters are restored. The body's env entries are overwritten by the real iterations that follow.
bool Executor::recurrencePass(const Loop* L, const std::vector<Edge>& headerEdges) {
  BasicBlock* H = L->getHeader();
  EP R = A.mkBool(false);
  for (auto& e : headerEdges) R = A.orb(R, e.cond);
  Edge e{nullptr, R, {}};
  std::vector<std::pair<PHINode*, EP>> inits;
  for (PHINode& PN : H->phis()) {
    Symbol s; s.kind = Symbol::ThreadValue; s.v = &PN; s.ty = PN.getType(); s.thread = tid;
    e.snap[&PN] = A.mkSym(s);
    // init: the entry value, merged over the entry edges
    EP v = nullptr;
    for (size_t i = headerEdges.size(); i-- > 0;) { auto it = headerEdges[i].snap.find(&PN); EP cur = it != headerEdges[i].snap.end() ? it->second : A.mkUndef(PN.getType()); v = v ? A.ite(headerEdges[i].cond, cur, v) : cur; }
    inits.push_back({&PN, v ? v : A.mkUndef(PN.getType())});
  }
  const unsigned savedPhase = phase, savedSeq = seq, savedIter = curIter; const bool savedLast = iterLast;
  shadow = true; curIter = ~0u; iterLast = false;
  EdgeMap in, exits;
  in[H].push_back(std::move(e));
  const bool ok = execRegion(H, L, in, exits);
  shadow = false; phase = savedPhase; seq = savedSeq; curIter = savedIter; iterLast = savedLast;
  if (!ok) return false;
  auto back = exits.find(H);
  if (back == exits.end() || back->second.empty()) return fail("the iterated loop has no back edge");
  for (auto& [PN, init] : inits) {
    EP v = nullptr;
    for (size_t i = back->second.size(); i-- > 0;) { auto it = back->second[i].snap.find(PN); EP cur = it != back->second[i].snap.end() ? it->second : A.mkUndef(PN->getType()); v = v ? A.ite(back->second[i].cond, cur, v) : cur; }
    recurrences.push_back({L, PN, init, v, tid});
  }
  return true;
}

bool Executor::modelCopy(CallInst* CI, EP R) {
  Value* dstV = nullptr; Value* srcV = nullptr; Value* validV = nullptr; unsigned bytes = 16; Value* srcSize = nullptr;
  if (CI->isInlineAsm()) {
    std::vector<PtxInstr> ins; std::string err;
    parsePtxAsm(std::string(cast<InlineAsm>(CI->getCalledOperand())->getAsmString()), ins, err);
    const PtxInstr& in = ins[0];
    if (in.predicated() || in.ops.size() < 3 || in.ops[2].kind != PtxOperand::Imm) return fail("cp.async form the executor does not model");
    bytes = (unsigned)in.ops[2].imm;
    if ((in.ops[0].kind != PtxOperand::Reg && in.ops[0].kind != PtxOperand::Mem) || (in.ops[1].kind != PtxOperand::Reg && in.ops[1].kind != PtxOperand::Mem)) return fail("cp.async operand form");
    dstV = CI->getArgOperand(in.ops[0].reg); srcV = CI->getArgOperand(in.ops[1].reg);
    if (in.ops.size() >= 4) { if (in.ops[3].kind == PtxOperand::Imm) { if (in.ops[3].imm == 0) validV = ConstantInt::getFalse(F.getContext()); else if ((unsigned)in.ops[3].imm != bytes) return fail("partial cp.async"); } else srcSize = CI->getArgOperand(in.ops[3].reg); }
    if (in.ops.size() > 4) return fail("cp.async with cache-policy operands");
  } else {
    bytes = (unsigned)std::stoul(CI->getCalledFunction()->getName().substr(strlen("__mvcc_cp_async_zsel_")).str());
    dstV = CI->getArgOperand(0); srcV = CI->getArgOperand(1); validV = CI->getArgOperand(2);
  }
  EP dst = get(dstV), src = get(srcV);
  EP valid = validV ? get(validV) : A.mkBool(true);
  if (srcSize) { EP sz = get(srcSize); int64_t k; if (A.constInt(sz, k)) valid = A.mkBool(k != 0); else valid = A.cmp(CmpInst::ICMP_NE, sz, A.mkInt(sz->ty, 0)); }
  if (valid->ty && !valid->ty->isIntegerTy(1)) valid = A.cmp(CmpInst::ICMP_NE, valid, A.mkInt(valid->ty, 0));
  // the copy is `bytes` device bytes, zeros when not valid
  Type* i8 = Type::getInt8Ty(F.getContext());
  std::vector<EP> bs;
  if (src->ty->isIntegerTy()) src = A.retype(A.asPoly(src), PointerType::get(F.getContext(), 0));
  Type* ity = Type::getIntNTy(F.getContext(), 8 * bytes);
  EP v = A.load(ity, src, A.andb(R, valid), bytes);
  if (recordDevLoads && !shadow) devLoads.push_back({src, bytes, A.andb(R, valid), phase, tid, iterStack, CI, loopStack});
  std::vector<EP> lb = A.toBytes(v);
  for (unsigned i = 0; i < bytes; i++) bs.push_back(A.isTrue(valid) ? lb[i] : A.ite(valid, lb[i], A.mkInt(i8, 0)));
  if (dst->ty->isIntegerTy()) dst = A.retype(A.asPoly(dst), PointerType::get(F.getContext(), 3));
  if (std::find(ringCopies.begin(), ringCopies.end(), CI) == ringCopies.end()) ringCopies.push_back(CI);
  return storeBytes(dst, bs, R, CI);
}

bool Executor::runRetile(unsigned t, std::string& w) {
  tid = t; phase = 0; seq = 0; tpOrd = 0; afterSkip = false; why.clear(); env.clear(); curIter = ~0u; iterLast = false; iterStack.clear(); loopStack.clear(); shadow = false; steps = 0; wallStart = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  EdgeMap in, out;
  in[&F.getEntryBlock()].push_back(Edge{nullptr, A.mkBool(true), {}});
  bool ok = execRegion(&F.getEntryBlock(), nullptr, in, out);
  finalPhase = phase;
  if (ok && reuseUniform && uniformEnv.empty()) {
    for (auto& kv : env)
      if (auto* I = dyn_cast<Instruction>(kv.first))
        if (!I->mayHaveSideEffects() && !isa<CallInst>(I) && !isa<PHINode>(I)
            && !sliceUsesTid(*I)
            && !A.containsSymbol(kv.second, Symbol::ThreadValue)
            && !A.containsSymbol(kv.second, Symbol::Private)
            && !A.containsSymbol(kv.second, Symbol::Unknown)
            && !A.containsSymbol(kv.second, Symbol::SmemByte))
          uniformEnv[I] = kv.second;
  }
  if (!ok) w = why;
  return ok;
}

bool Executor::crossesLanes(EP e) {
  auto it = crossMemo.find(e);
  if (it != crossMemo.end()) return it->second;
  bool r = false;
  if (e->k == EK::Sym) {
    const Symbol::Kind k = A.symbol(e->sym).kind;
    r = k == Symbol::ThreadValue || k == Symbol::Unknown || k == Symbol::SmemByte;
  } else if (e->k == EK::Call && e->callee) {
    StringRef n = e->callee->getName();
    r = n.starts_with("llvm.nvvm.shfl.sync.") || n.starts_with("llvm.nvvm.vote.") || n.starts_with("llvm.nvvm.match.") || n.starts_with("llvm.nvvm.redux.sync.");
  }
  for (size_t i = 0; i < e->args.size() && !r; i++) r = crossesLanes(e->args[i]);
  if (!r && e->k == EK::Poly) for (auto& t : e->terms) { for (uint32_t a : t.first) if (crossesLanes(A.atomExpr(a))) { r = true; break; } if (r) break; }
  return crossMemo[e] = r;
}

EP Executor::returnedBefore(unsigned thread, BasicBlock* B) {
  auto f = returnedMemo.find({thread, B});
  if (f != returnedMemo.end()) return f->second;
  EP dead = A.mkBool(false);
  for (auto& kv : returned) {
    if (kv.first.first != thread) continue;
    BasicBlock* XB = kv.first.second;
    if (XB == B || isPotentiallyReachable(B, XB, nullptr, &DT, &LI)) continue;
    dead = A.orb(dead, kv.second);
  }
  return returnedMemo[{thread, B}] = dead;
}

EP Executor::factsAt(unsigned thread, BasicBlock* B) {
  std::function<void(EP, std::vector<EP>&)> disjuncts = [&](EP e, std::vector<EP>& out) { if (e->k == EK::Op && e->op == Instruction::Or && e->ty->isIntegerTy(1)) { disjuncts(e->args[0], out); disjuncts(e->args[1], out); } else out.push_back(e); };
  std::vector<EP> ways; disjuncts(returnedBefore(thread, B), ways);
  EP facts = A.mkBool(true);
  for (EP w : ways) {
    if (A.isFalse(w)) continue;
    EP ex = expandGuards(w, true);
    if (ex->k == EK::Op && ex->op == Instruction::Or && ex->ty->isIntegerTy(1)) { std::vector<EP> more; disjuncts(ex, more); for (EP m : more) if (!(m->k == EK::Op && m->op == Instruction::And && m->ty->isIntegerTy(1))) facts = A.andb(facts, A.notb(m)); continue; }
    if (ex->k == EK::Op && ex->op == Instruction::And && ex->ty->isIntegerTy(1)) continue;   // !(a and b): a disjunction, not a fact
    facts = A.andb(facts, A.notb(ex));
  }
  return facts;
}

EP Executor::expandGuards(EP clause, bool positive) {
  std::function<void(EP, std::vector<EP>&)> disjuncts = [&](EP e, std::vector<EP>& out) { if (e->k == EK::Op && e->op == Instruction::Or && e->ty->isIntegerTy(1)) { disjuncts(e->args[0], out); disjuncts(e->args[1], out); } else out.push_back(e); };
  std::vector<EP> ls; A.conjuncts(clause, ls);
  EP ex = A.mkBool(true);
  for (EP l : ls) {
    EP g = nullptr; bool neg = false;
    if (l->k == EK::Sym && A.symbol(l->sym).kind == Symbol::Guard) g = l;
    else if (l->k == EK::Op && l->op == Instruction::Xor && l->args.size() == 2 && l->args[0]->k == EK::Sym && A.symbol(l->args[0]->sym).kind == Symbol::Guard && A.isTrue(l->args[1])) { g = l->args[0]; neg = true; }
    if (!g) { ex = A.andb(ex, l); continue; }
    EP def = A.guardDef(g->sym);
    if (!neg) { if (positive) ex = A.andb(ex, def); continue; }
    // !(d1 or d2 or ...) = !d1 and !d2 and ...; a disjunct that is itself a conjunction contributes nothing (a
    // disjunction of negations: dropping it leaves a weaker clause - the callers' tests stay sound on a superset)
    std::vector<EP> ds; disjuncts(def, ds);
    for (EP d : ds) if (!(d->k == EK::Op && d->op == Instruction::And && d->ty->isIntegerTy(1))) ex = A.andb(ex, A.notb(expandGuards(d, false)));
  }
  return ex;
}

bool Executor::refutes(EP facts, EP e, unsigned budget) {
  auto isOr = [](EP x) { return x->k == EK::Op && x->op == Instruction::Or && x->ty->isIntegerTy(1); };
  auto isAnd = [](EP x) { return x->k == EK::Op && x->op == Instruction::And && x->ty->isIntegerTy(1); };
  auto guardOf = [&](EP l, bool& neg) -> EP {
    neg = false;
    if (l->k == EK::Sym && A.symbol(l->sym).kind == Symbol::Guard) return l;
    if (l->k == EK::Op && l->op == Instruction::Xor && l->args.size() == 2 && l->args[0]->k == EK::Sym && A.symbol(l->args[0]->sym).kind == Symbol::Guard && A.isTrue(l->args[1])) { neg = true; return l->args[0]; }
    return nullptr;
  };
  unsigned steps = 0, depth = 0;
  auto isNot = [&](EP x) { return x->k == EK::Op && x->op == Instruction::Xor && x->args.size() == 2 && A.isTrue(x->args[1]); };
  // a literal under the facts F: 1 held, -1 denied, 0 open (the clamp-aware checker decides)
  auto& litMemo = refuteLitMemo;
  auto litStatus = [&](EP F, EP l) -> int {
    auto it = litMemo.find({F, l});
    if (it != litMemo.end()) return it->second;
    int r = 0;
    if (A.implies(F, l)) r = 1; else if (A.disjoint(F, l)) r = -1;
    else {
      std::vector<EP> ls; A.conjuncts(F, ls);
      std::vector<EP> a(ls); a.push_back(l); if (A.contradictory(a, true)) r = -1;
      else { std::vector<EP> b(ls); b.push_back(A.notb(l)); if (A.contradictory(b, true)) r = 1; }
    }
    return litMemo[{F, l}] = r;
  };
  // the first literal of x (guards read through their definitions) the facts F leave open, skipping the parts
  // they already decide (a dead conjunct's other literals are not worth a case split); out: x's own status
  std::function<EP(EP, EP, int&)> undecided = [&](EP x, EP F, int& st) -> EP {
    if (A.isTrue(x)) { st = 1; return nullptr; } if (A.isFalse(x)) { st = -1; return nullptr; }
    bool neg; EP g = guardOf(x, neg);
    if (g) { EP r = undecided(A.guardDef(g->sym), F, st); if (neg) st = -st; return r; }
    if (isNot(x)) { EP r = undecided(x->args[0], F, st); st = -st; return r; }
    if (isOr(x) || isAnd(x)) {
      const int decides = isOr(x) ? 1 : -1;   // a held disjunct decides an or; a denied conjunct an and
      std::vector<EP> parts; if (isOr(x)) { std::function<void(EP)> fl = [&](EP d) { if (isOr(d)) { fl(d->args[0]); fl(d->args[1]); } else parts.push_back(d); }; fl(x); } else A.conjuncts(x, parts);
      EP cand = nullptr; bool allOther = true;
      for (EP a : parts) { int sa; EP r = undecided(a, F, sa); if (sa == decides) { st = decides; return nullptr; } if (sa == 0) { allOther = false; if (!cand) cand = r; } }
      st = allOther ? -decides : 0;
      return cand;
    }
    st = litStatus(F, x);
    return st ? nullptr : x;
  };
  auto& memo = refuteMemo;
  // F: the plain literals in force (a conjunction); pending: the positive guards among the facts, disjunctions the
  // proof splits on when the literals alone do not decide. prove(e): F entails e; refute(e): F contradicts e.
  std::function<bool(EP, std::vector<EP>&, EP, bool)> go0;
  std::function<bool(EP, std::vector<EP>&, EP, bool)> go = [&](EP F, std::vector<EP>& pending, EP x, bool prove) -> bool {
    const auto key = std::make_tuple(F, x, prove, pending);
    auto it = memo.find(key);
    if (it != memo.end()) return it->second;
    const bool r = go0(F, pending, x, prove);
    if (steps <= budget) memo[key] = r;
    return r;
  };
  go0 = [&](EP F, std::vector<EP>& pending, EP x, bool prove) -> bool {
    if (++steps > budget) return false;
    if (A.isTrue(x)) return prove; if (A.isFalse(x)) return !prove;
    bool neg; EP g = guardOf(x, neg);
    if (g) return go(F, pending, A.guardDef(g->sym), prove != neg);
    if (x->k == EK::Op && x->op == Instruction::Xor && x->args.size() == 2 && A.isTrue(x->args[1])) return go(F, pending, x->args[0], !prove);
    std::vector<EP> parts;
    if (isOr(x)) { std::function<void(EP)> fl = [&](EP d) { if (isOr(d)) { fl(d->args[0]); fl(d->args[1]); } else parts.push_back(d); }; fl(x); }
    else if (isAnd(x)) A.conjuncts(x, parts);
    if (!parts.empty()) {
      // prove a disjunction: one part; refute it: every part. Prove a conjunction: every part; refute it: one.
      const bool any = prove == isOr(x);
      if (!any) { for (EP p : parts) if (!go(F, pending, p, prove)) return false; return true; }
      for (EP p : parts) if (go(F, pending, p, prove)) return true;
      // no part decides alone (a disjunction whose disjuncts each hold on a different case of the facts): split
      // on a literal of the parts the facts leave open; both cases must decide
      if (depth >= 16) return false;
      int st; EP l = undecided(x, F, st);
      if (st) return (st > 0) == prove;
      if (!l) return false;
      ++depth;
      bool ok = true;
      for (EP b : {l, A.notb(l)}) {
        EP F2 = A.andb(F, b);
        if (A.isFalse(F2)) continue;
        { std::vector<EP> l2; A.conjuncts(F2, l2); if (A.contradictory(l2, true)) continue; }
        if (!go(F2, pending, x, prove)) { ok = false; break; }
      }
      --depth;
      return ok;
    }
    // a literal: by the literals in force, else by splitting a pending guard (every disjunct must decide it)
    std::vector<EP> ls; A.conjuncts(F, ls);
    if (prove ? A.implies(F, x) : A.disjoint(F, x)) return true;
    { std::vector<EP> all(ls); all.push_back(prove ? A.notb(x) : x); if (A.contradictory(all, true)) return true; }
    if (pending.empty()) { return false; }
    EP G = pending.back(); pending.pop_back();
    std::vector<EP> ds; std::function<void(EP)> fl = [&](EP d) { if (isOr(d)) { fl(d->args[0]); fl(d->args[1]); } else ds.push_back(d); }; fl(G);
    bool ok = true;
    for (EP d : ds) {
      // the disjunct's literals join F; its own positive guards join the pending list; a disjunct F contradicts
      // is not a case
      std::vector<EP> dl; A.conjuncts(d, dl);
      EP F2 = F; std::vector<EP> pend2(pending); bool dead = false;
      for (EP l : dl) { bool n2; EP g2 = guardOf(l, n2); if (g2 && !n2) { pend2.push_back(A.guardDef(g2->sym)); continue; } F2 = A.andb(F2, l); if (A.isFalse(F2)) { dead = true; break; } }
      if (!dead) { std::vector<EP> l2; A.conjuncts(F2, l2); if (A.contradictory(l2, true)) dead = true; }
      if (dead) continue;
      if (!go(F2, pend2, x, prove)) { ok = false; break; }
    }
    pending.push_back(G);
    return ok;
  };
  // the facts: plain literals in force, positive guards pending, negated guards refuted-to-be (their definitions
  // are disjunctions the facts deny: each disjunct is a fact's negation - kept as pending negations is complex;
  // they join the target instead: refuting e under (F and not G) is refuting (e and not G) under F)
  EP F = A.mkBool(true); std::vector<EP> pending; EP target = e;
  std::vector<EP> fl; A.conjuncts(facts, fl);
  for (EP l : fl) { bool neg; EP g = guardOf(l, neg); if (g && !neg) pending.push_back(A.guardDef(g->sym)); else if (g) target = A.andb(target, l); else F = A.andb(F, l); }
  if (A.isFalse(F)) return true;
  const bool r = go(F, pending, target, false);
  return r;
}

bool Executor::resolve(std::string& w) {
  std::map<uint32_t, EP> m;
  // shared object -> the first barrier phase a store at a symbolic offset (not modeled) touches it in
  std::map<GlobalVariable*, unsigned> symStoreMinPhase;
  for (auto& a : smemAccesses) {
    if (!a.isStore || a.concrete) continue;
    GlobalVariable* o = nullptr; EP p = A.asPoly(a.addr);
    for (auto& t : p->terms) { uint32_t sid; if (t.first.size() == 1 && A.isSymAtom(t.first[0], sid) && A.symbol(sid).kind == Symbol::SharedBase) o = cast<GlobalVariable>(A.symbol(sid).v); }
    if (!o) continue;
    auto it = symStoreMinPhase.find(o);
    if (it == symStoreMinPhase.end()) symStoreMinPhase[o] = a.phase; else it->second = std::min(it->second, a.phase);
  }
  auto unmodeledBefore = [&](const SmemRead& r) { auto it = symStoreMinPhase.find(r.obj); return it != symStoreMinPhase.end() && it->second <= r.phase; };
  // The residual `rem` of a read lies on paths the read's thread had left: it implies the disjunction of the
  // conditions under which the thread returned at a `ret` the read's block cannot reach (a return the read could
  // precede on some path - the epilogue's - says nothing about the read). Memoized per (thread, block).
  // Returns the part of `rem` that is not dead (false when all of it is).
  auto liveResidual = [&](const SmemRead& r, EP rem) -> EP {
    BasicBlock* RB = r.site ? r.site->getParent() : nullptr;
    if (!RB) return rem;
    EP dead = returnedBefore(r.thread, RB);
    if (A.isFalse(dead)) return rem;
    // clause by clause: a clause of the residual implies some way the thread returned. The clause's guard literals
    // are read through their definitions (a negated guard whose definition is a disjunction of literals is the
    // conjunction of their negations: the early return `if (a || b) return` leaves `!(a || b)` on the continuing
    // paths and `a || b` as the return's condition), since `implies` matches literals syntactically.
    std::vector<EP> ways, clauses;
    std::function<void(EP, std::vector<EP>&)> disjuncts = [&](EP e, std::vector<EP>& out) { if (e->k == EK::Op && e->op == Instruction::Or && e->ty->isIntegerTy(1)) { disjuncts(e->args[0], out); disjuncts(e->args[1], out); } else out.push_back(e); };
    disjuncts(dead, ways); disjuncts(rem, clauses);
    EP live = A.mkBool(false);
    for (EP C : clauses) {
      EP ex = expandGuards(C, true);
      bool any = A.isFalse(ex);
      if (!any && !(ex->k == EK::Op && ex->op == Instruction::Or && ex->ty->isIntegerTy(1))) { std::vector<EP> ls; A.conjuncts(ex, ls); any = A.contradictory(ls, true); }   // infeasible: no stage left, yet a third
      for (size_t i = 0; i < ways.size() && !any; i++) any = A.implies(ex, ways[i]);
      if (!any) live = A.orb(live, C);
    }
    return live;
  };
  // a literal over the arguments only (memoized: the reads share their conditions' literals)
  std::unordered_map<EP, bool> argsOnlyMemo;
  auto argsOnly = [&](EP e) {
    auto f = argsOnlyMemo.find(e);
    if (f != argsOnlyMemo.end()) return f->second;
    bool r = true;
    for (Symbol::Kind k : {Symbol::Sreg, Symbol::ThreadValue, Symbol::Unknown, Symbol::SharedBase, Symbol::Global, Symbol::SmemByte, Symbol::LoopValue}) if (A.containsSymbol(e, k)) { r = false; break; }
    return argsOnlyMemo[e] = r;
  };
  for (const SmemRead& r : smemReads) {
    // latest write first; the read sees the latest write whose condition held. Writes are collected until their
    // conditions cover the read's; a write that is not ordered before the read (same phase, another thread) and
    // not excluded by the conditions is a race.
    std::vector<const SmemWrite*> cover;
    EP remaining = r.cond;
    auto it = smemByOff.find(r.off);
    // the writes of a byte in program order (barrier phase, then sequence): the executor runs the threads one after
    // another, so the recording order interleaves the phases
    std::vector<size_t> order;
    if (it != smemByOff.end()) {
      order = it->second;
      std::stable_sort(order.begin(), order.end(), [&](size_t x, size_t y) {
        const SmemWrite& a = smemWrites[x]; const SmemWrite& b = smemWrites[y];
        return a.phase != b.phase ? a.phase < b.phase : a.seq < b.seq;
      });
    }
    if (!retile && it != smemByOff.end()) {
      // After the epilogue's barrier the C tile is a permutation of the fragment stores. A 16-byte
      // load's path condition (stream-K / edge joins) can be wider than any one fragment store's;
      // race-freedom plus the barrier say the latest earlier-phase write is the value.
      const SmemWrite* W = nullptr;
      for (auto wi = order.rbegin(); wi != order.rend(); ++wi) {
        const SmemWrite& wr = smemWrites[*wi];
        if (wr.phase >= r.phase) continue;
        W = &wr; break;
      }
      if (W) { m[r.sym] = W->byte; writeReaders[W->site].insert(r.site); continue; }
    }
    if (retile && it != smemByOff.end()) {
      // retile mode: the read sees the latest write that can precede it. Where the read's path condition does not
      // imply the write's, the difference is assumed - if it is a condition on the kernel's arguments alone (the
      // even K-group count a launcher can guarantee, which decides whether a ring slot was refilled before the
      // iteration that reads it) - and recorded for the launch: the recovered body runs only when the assumptions
      // hold, the exact twin otherwise. Differences on other values (a thread's, a loop's) keep the selection.
      const SmemWrite* W = nullptr;
      for (auto wi = order.rbegin(); wi != order.rend(); ++wi) {
        const SmemWrite& wr = smemWrites[*wi];
        if (wr.phase > r.phase) continue;
        if (wr.phase == r.phase && wr.thread == r.thread && wr.seq >= r.seq) continue;
        if (A.disjoint(r.cond, wr.cond)) continue;
        bool ordered = wr.phase < r.phase || (wr.phase == r.phase && wr.thread == r.thread);
        if (!ordered) { w = "shared byte " + std::to_string(r.off) + " of " + r.obj->getName().str() + " is read by thread " + std::to_string(r.thread) + " in the barrier phase thread " + std::to_string(wr.thread) + " writes it"; return false; }
        W = &wr; break;
      }
      if (W && laneStoresStay && isa<StoreInst>(W->site) && crossesLanes(W->byte)) {
        // the byte stays in shared memory: its value is the aligned 32-bit word of the staged object at the read's
        // offset (a Load whose address is the object's base plus a constant: affine in the retiler's coordinates)
        EP base = A.asPoly(evalConstant(r.obj));
        const int64_t w0 = r.off & ~(int64_t)3;
        EP word = A.load(Type::getInt32Ty(F.getContext()), A.add(base, A.mkInt(base->ty, w0)), r.cond, 4);
        m[r.sym] = A.bytes(word, (unsigned)(r.off - w0), 1);
        writeReaders[W->site].insert(r.site); stayingReads.insert(r.site);
        continue;
      }
      if (W && !A.implies(r.cond, W->cond)) {
        std::vector<EP> rl, wl; A.conjuncts(r.cond, rl); A.conjuncts(W->cond, wl);
        std::set<EP> rs(rl.begin(), rl.end());
        EP D = A.mkBool(true); bool ok = true;
        for (EP l : wl) if (!rs.count(l)) { if (!argsOnly(l)) { ok = false; break; } D = A.andb(D, l); }
        // The kernel defines the byte where the latest write does not: an earlier store of a constant (the zero
        // fill of the rows past M, before the copy `if (row < M)`) covers the difference. That is a selection the
        // program makes (`row < M ? loaded : 0`, the shape every row has), not a condition to assume at launch -
        // assumed, the row-edge literal would exclude the launches it is about and give this row a shape unlike
        // the rows whose copy condition arrives as a guard symbol (the cover path below), and the fit fails.
        if (ok) {
          for (auto pi = order.rbegin(); pi != order.rend(); ++pi) {
            const SmemWrite& pw = smemWrites[*pi];
            if (&pw == W || pw.phase > W->phase || (pw.phase == W->phase && (pw.thread != W->thread || pw.seq >= W->seq))) continue;
            const bool constant = pw.byte->k == EK::Const || (pw.byte->k == EK::Poly && (pw.byte->terms.empty() || (pw.byte->terms.size() == 1 && pw.byte->terms[0].first.empty())));
            if (!constant) continue;
            if (A.implies(r.cond, pw.cond)) ok = false;
            break;
          }
        }
        if (ok) { EP P = A.mkBool(true); for (EP l : rl) if (argsOnly(l)) P = A.andb(P, l); assumptions[{P, D}].push_back(r.sym); m[r.sym] = W->byte; writeReaders[W->site].insert(r.site); continue; }
      } else if (W) { m[r.sym] = W->byte; writeReaders[W->site].insert(r.site); continue; }
    }
    if (it != smemByOff.end()) {
      for (auto wi = order.rbegin(); wi != order.rend() && !A.isFalse(remaining); ++wi) {
        const SmemWrite& wr = smemWrites[*wi];
        if (wr.phase > r.phase) continue;  // a write of a later barrier phase (the executor ran on past the read)
        if (wr.phase == r.phase && wr.thread == r.thread && wr.seq >= r.seq) continue;  // the thread's own later write
        if (A.disjoint(remaining, wr.cond)) continue;
        bool ordered = wr.phase < r.phase || (wr.phase == r.phase && wr.thread == r.thread);
        if (!ordered) {
          w = "shared byte " + std::to_string(r.off) + " of " + r.obj->getName().str() + " is read by thread " + std::to_string(r.thread) + " in the barrier phase thread " + std::to_string(wr.thread) + " writes it";
          return false;
        }
        cover.push_back(&wr);
        remaining = A.implies(remaining, wr.cond) ? A.mkBool(false) : A.andb(remaining, A.notb(wr.cond));
      }
    }
    if (!A.isFalse(remaining) && !unmodeledBefore(r)) {
      // Bytes no write reaches under `remaining`, of an object no store at a symbolic offset (unmodeled) touches
      // before the read. Either `remaining` lies on paths the thread had already left by a return the read's block
      // cannot reach (the early return before the K loop: an empty split, a tile past N) - the read's path condition
      // is a join of case splits and over-approximates the paths that reach it, and the residual is the part it
      // admits that never executes - or the bytes are uninitialized shared memory the kernel does read (a weight row
      // past N: the copy is guarded, the mma reads the row, its outputs are never stored), recorded for the caller
      // to prove harmless from the kernel's structure. The byte resolves to zero there so that the operand keeps one
      // shape (`row < N ? loaded : 0`, the shape of a zero-filled row). The epilogue uses the same residual: a
      // 16-byte C-tile load always happens; the fragment stores that produce those bytes are edge-guarded, and the
      // device store of the chunk is too.
      EP live = liveResidual(r, remaining);
      const bool dead = A.isFalse(live);
      EP val = A.mkInt(Type::getInt8Ty(F.getContext()), 0);
      for (const SmemWrite* cw : cover) writeReaders[cw->site].insert(r.site);
      for (size_t i = cover.size(); i-- > 0;) { EP c = retile ? A.given(cover[i]->cond, r.cond) : cover[i]->cond; val = A.ite(c, cover[i]->byte, val); resolveConds.insert(c); }
      m[r.sym] = val;
      if (dead) deadReads++;
      else uncovered.push_back({r.sym, r.site, r.thread, r.phase, live});
      continue;
    }
    if (!A.isFalse(remaining) && retile) {
      // a read no modeled write covers (e.g. staged by a loop the executor ran generically): its value is unknown;
      // the retiler refuses to depend on it and keeps the instruction as written
      unresolvedReads.insert(r.sym);
      Symbol s; s.kind = Symbol::Unknown; s.thread = r.thread; s.ty = Type::getInt8Ty(F.getContext()); s.field = 0x40000000u + r.sym;
      m[r.sym] = A.mkSym(s);
      continue;
    }
    if (!A.isFalse(remaining)) {
      w = "shared byte " + std::to_string(r.off) + " of " + r.obj->getName().str() + " is read by thread " + std::to_string(r.thread) + " in the epilogue without a write in the epilogue covering it (uncovered under " + A.str(remaining, 5) + ")";
      return false;
    }
    EP val = cover.back()->byte;
    for (const SmemWrite* cw : cover) writeReaders[cw->site].insert(r.site);
    for (size_t i = cover.size() - 1; i-- > 0;) { EP c = retile ? A.given(cover[i]->cond, r.cond) : cover[i]->cond; val = A.ite(c, cover[i]->byte, val); resolveConds.insert(c); }
    // one thread stores, a barrier, everyone loads: the value is uniform (decode.cuh s_last, decode.cu shared atomics)
    if (!retile && cover.size() == 1 && cover[0]->phase < r.phase && A.containsSymbol(val, Symbol::ThreadValue)) {
      std::map<EP, EP> bm;
      val = A.rewrite(val, [&](EP n) -> EP {
        if (n->k != EK::Sym) return nullptr;
        const Symbol& s = A.symbol(n->sym);
        if (s.kind != Symbol::ThreadValue) return nullptr;
        Symbol u; u.kind = Symbol::Unknown; u.thread = 0; u.ty = s.ty; u.field = 0x50000000u + n->sym; u.v = s.v;
        return A.mkSymRaw(u);
      }, bm);
    }
    m[r.sym] = val;
  }
  if (!retile && !uncovered.empty()) {
    // A C-tile load whose residual the fragment stores do not cover is harmless when no device store
    // still names that byte (the load was already folded to the writes at the site) or when every
    // store that does names it runs under a condition that refutes the residual (the edge-guarded
    // copy-out: rows past M).
    for (const Uncovered& U : uncovered) {
      for (const DevStore& st : devStores) {
        bool feeds = false;
        std::set<uint32_t> ss;
        A.containsSymbol(st.addr, Symbol::SmemByte, &ss);
        A.containsSymbol(st.cond, Symbol::SmemByte, &ss);
        for (EP b : st.bytes) A.containsSymbol(b, Symbol::SmemByte, &ss);
        if (ss.count(U.sym)) feeds = true;
        if (!feeds) continue;
        if (A.disjoint(st.cond, U.cond) || refutes(st.cond, U.cond)) continue;
        w = "shared byte read by thread " + std::to_string(U.thread) + " in the epilogue without a write covering it (uncovered under " + A.str(U.cond, 5) + ")";
        return false;
      }
    }
  }
  resolved = m;
  std::map<EP, EP> memo;
  for (DevStore& st : devStores) {
    st.addr = A.subst(st.addr, m, memo);
    st.cond = A.subst(st.cond, m, memo);
    std::vector<EP> bs;
    for (EP b : st.bytes) { EP v = A.subst(b, m, memo); auto vb = A.toBytes(v); bs.insert(bs.end(), vb.begin(), vb.end()); }
    st.bytes = bs;
  }
  return true;
}

template <class Memo> static EP substT(Arena& A, EP e, const std::map<uint32_t, EP>& m, Memo& memo) {
  return A.rewrite(e, [&](EP n) -> EP {
    if (n->k != EK::Sym) return nullptr;
    auto f = m.find(n->sym);
    return f == m.end() ? nullptr : f->second;
  }, memo);
}
EP Arena::subst(EP e, const std::map<uint32_t, EP>& m, std::map<EP, EP>& memo) { return substT(*this, e, m, memo); }
EP Arena::subst(EP e, const std::map<uint32_t, EP>& m, RewriteMemo& memo) { return substT(*this, e, m, memo); }

EP Arena::unwrap(EP e) const {
  if (e->k == EK::Poly && e->terms.size() == 1 && e->terms[0].second == 1 && e->terms[0].first.size() == 1) {
    EP a = atoms[e->terms[0].first[0]];
    if (a->ty == e->ty) return a;
  }
  return e;
}

EP Arena::rewrite(EP e, const std::function<EP(EP)>& leaf, std::map<EP, EP>& memo) { return rewriteT(e, leaf, memo); }
EP Arena::rewrite(EP e, const std::function<EP(EP)>& leaf, RewriteMemo& memo) { return rewriteT(e, leaf, memo); }
template <class Memo> EP Arena::rewriteT(EP e, const std::function<EP(EP)>& leaf, Memo& memo) {
  auto it = memo.find(e);
  if (it != memo.end()) return it->second;
  EP r = leaf(e);
  if (!r) {
    r = e;
    switch (e->k) {
      case EK::Sym: case EK::Const: case EK::Undef: break;
      case EK::Poly: {
        if constexpr (std::is_same_v<Memo, RewriteMemo>) {
          // (hashed memo - the decode retiler's large DAGs: a polynomial none of whose atoms changes is kept as it is,
          // without the rebuild below, which interns the partial sums even when the result is the polynomial itself)
          bool any = false;
          for (auto& t : e->terms) for (uint32_t a : t.first) any |= rewrite(atoms[a], leaf, memo) != atoms[a];
          if (!any) break;
        }
        EP acc = mkPoly(e->ty, {});
        bool ch = false;
        for (auto& t : e->terms) {
          EP term = mkInt(e->ty, t.second);
          for (uint32_t a : t.first) {
            EP ae = atoms[a];
            EP se = rewrite(ae, leaf, memo);
            if (se != ae) ch = true;
            term = mul(term, se == ae ? mkPoly(e->ty, {{{a}, 1}}) : retype(asPoly(se), e->ty));
          }
          acc = add(acc, term);
        }
        if (ch) r = unwrap(acc);
        break;
      }
      default: {
        std::vector<EP> args; bool ch = false;
        for (EP a : e->args) { EP c = rewrite(a, leaf, memo); ch |= c != a; args.push_back(c); }
        if (!ch) break;
        switch (e->k) {
          case EK::Ite: r = ite(args[0], args[1], args[2]); break;
          case EK::Cmp: r = cmp(e->op, args[0], args[1]); break;
          case EK::Bytes: r = bytes(args[0], e->lo, e->len); break;
          case EK::Concat: r = concat(std::move(args), e->ty); break;
          case EK::Op:
            if (args.size() == 1) r = Instruction::isCast(e->op) ? cast(e->op, args[0], e->ty) : unop(e->op, args[0], e->ty);
            else r = unwrap(binop(e->op, args[0], args[1], e->ty));
            break;
          case EK::Load: r = unwrap(load(e->ty, args[0], args[1], e->len)); break;
          case EK::Call: r = unwrap(e->asmv ? callAsm(e->asmv, std::move(args), e->ty) : call(e->callee, std::move(args), e->ty)); break;
          default: r = rebuild(e, std::move(args));
        }
      }
    }
  }
  memo[e] = r;
  return r;
}

static void addFact(Arena& A, EP c, bool val, std::map<EP, bool>& f) {
  f[c] = val;
  if (c->k == EK::Cmp) f[A.notb(c)] = !val;
  if (c->k == EK::Op && c->ty->isIntegerTy(1) && c->args.size() == 2) {
    if (c->op == Instruction::And && val) { addFact(A, c->args[0], true, f); addFact(A, c->args[1], true, f); }
    if (c->op == Instruction::Or && !val) { addFact(A, c->args[0], false, f); addFact(A, c->args[1], false, f); }
  }
}
// Selections under the facts of the enclosing selections. Two levels:
//  - value level (a selection's arms, concatenations, byte extracts): a selection whose condition the facts decide -
//    as literals, or by the bounds they put on the condition's polynomial (sge(p-17, 0) decides slt(p-1, 0) and
//    ule(p-16, 0)) - folds. These are the merges of the executor's paths (a sub-tile skipped, a ring slot's writes)
//    and the same decision is made for every element of a tile (the facts are the tile's guards).
//  - below (addresses, conditions, operands): only the conditions the caller asks to fold, by literal facts. A
//    selection there is the element's own (a staged row's null test); deciding it by bounds would decide it for
//    some rows of a tile and not others, and the fit needs one shape.
// ---- the facts a selection can be decided by (simplifyUnder's cache keys)
uint64_t Arena::polyClassKey(EP cmp) {
  if (cmp->k != EK::Cmp || cmp->args[0]->k != EK::Poly) return 0;
  if (polyClassMemo.size() <= cmp->id) polyClassMemo.resize(nodes.size(), ~0ull);
  uint64_t& slot = polyClassMemo[cmp->id];
  if (slot != ~0ull) return slot;
  std::vector<std::pair<std::vector<uint32_t>, int64_t>> p;
  for (auto& t : cmp->args[0]->terms) if (!t.first.empty()) p.push_back(t);
  auto key = std::make_pair(std::move(p), cmp->args[0]->ty);
  auto it = polyClassIds.find(key);
  if (it == polyClassIds.end()) it = polyClassIds.emplace(std::move(key), (uint32_t)polyClassIds.size()).first;
  return slot = ((uint64_t)it->second << 1) | 1;
}
// the memo vectors hold, per node id, the sorted keys behind a leading marker element (so that an empty answer is
// told from none); the answers are returned without the marker
static bool knownKeys(std::vector<std::vector<uint64_t>>& memo, EP e, size_t n, ArrayRef<uint64_t>& out) {
  if (memo.size() <= e->id) memo.resize(n);
  const std::vector<uint64_t>& v = memo[e->id];
  if (v.empty()) return false;
  out = ArrayRef<uint64_t>(v).drop_front();
  return true;
}
static ArrayRef<uint64_t> storeKeys(std::vector<std::vector<uint64_t>>& memo, EP e, std::vector<uint64_t>&& keys) {
  keys.insert(keys.begin(), 0);   // the marker
  memo[e->id] = std::move(keys);
  return ArrayRef<uint64_t>(memo[e->id]).drop_front();
}
static void unionKeys(std::vector<uint64_t>& out, ArrayRef<uint64_t> more) {
  if (more.empty()) return;
  if (out.empty()) { out.assign(more.begin(), more.end()); return; }
  std::vector<uint64_t> m; m.reserve(out.size() + more.size());
  std::set_union(out.begin(), out.end(), more.begin(), more.end(), std::back_inserter(m));
  out.swap(m);
}
// decideCond rewrites a condition with a leaf that looks every boolean node up among the facts, and decides every
// comparison by the bounds the facts put on its polynomial: both kinds of key, for every node of the condition
ArrayRef<uint64_t> Arena::conditionKeys(EP c) {
  ArrayRef<uint64_t> known; if (knownKeys(condKeysMemo, c, nodes.size(), known)) return known;
  std::vector<uint64_t> keys;
  if (c->k == EK::Poly) { for (auto& t : c->terms) for (uint32_t a : t.first) unionKeys(keys, conditionKeys(atoms[a])); }
  else for (EP a : c->args) unionKeys(keys, conditionKeys(a));
  if (c->ty->isIntegerTy(1)) { std::vector<uint64_t> own{(uint64_t)c->id << 1}; if (uint64_t pk = polyClassKey(c)) own.push_back(pk); std::sort(own.begin(), own.end()); unionKeys(keys, own); }
  return storeKeys(condKeysMemo, c, std::move(keys));
}
// the conditions of the selections anywhere in e (a selection's condition itself is kept as written: only its
// nodes' keys, not the selections inside it, matter)
ArrayRef<uint64_t> Arena::selectionKeys(EP e) {
  ArrayRef<uint64_t> known; if (knownKeys(selKeysMemo, e, nodes.size(), known)) return known;
  std::vector<uint64_t> keys;
  if (e->k == EK::Poly) { for (auto& t : e->terms) for (uint32_t a : t.first) unionKeys(keys, selectionKeys(atoms[a])); }
  else if (e->k == EK::Ite) { unionKeys(keys, conditionKeys(e->args[0])); unionKeys(keys, selectionKeys(e->args[1])); unionKeys(keys, selectionKeys(e->args[2])); }
  else for (EP a : e->args) unionKeys(keys, selectionKeys(a));
  return storeKeys(selKeysMemo, e, std::move(keys));
}
// The cache key of a computation over e under `facts`: e and the facts among `keys` (the others cannot take part in
// any decision inside e: a literal is looked up by identity, a bound applies to its own polynomial class, and only
// facts that hold give bounds). The facts a computation with the reduced set makes are the same, so its result and
// the nodes it interns are the same.
static void appendRaw(std::string& k, const void* p, size_t n) { k.append((const char*)p, n); }
static std::string suKey(Arena& A, char tag, EP e, ArrayRef<uint64_t> keys, const std::map<EP, bool>& facts, unsigned depth, bool valueLevel, std::map<EP, bool>* relevant) {
  std::string k; k.reserve(16 + 9 * facts.size());
  k.push_back(tag); appendRaw(k, &e, sizeof e); uint8_t d = (uint8_t)std::min(depth, 255u); k.push_back((char)d); k.push_back(valueLevel ? 1 : 0);
  for (auto& f : facts) {
    bool rel = std::binary_search(keys.begin(), keys.end(), (uint64_t)f.first->id << 1);
    if (!rel && f.second) { uint64_t pk = A.polyClassKey(f.first); rel = pk && std::binary_search(keys.begin(), keys.end(), pk); }
    if (!rel) continue;
    appendRaw(k, &f.first, sizeof f.first); k.push_back(f.second ? 1 : 0);
    if (relevant) (*relevant)[f.first] = f.second;
  }
  return k;
}
static EP simplifyUnderRec(Arena& A, EP e, const std::function<bool(EP)>& fold, std::map<EP, bool>& facts, std::map<EP, EP>& memo, unsigned depth, bool valueLevel, Arena::SimplifyUnderCache& cache);
static EP decideCond(Arena& A, EP c, std::map<EP, bool>& facts, bool byBounds, Arena::SimplifyUnderCache& cache) {
  ArrayRef<uint64_t> keys = A.conditionKeys(c);
  std::string ck = suKey(A, byBounds ? 'D' : 'd', c, keys, facts, 0, false, nullptr);
  auto hit = cache.results.find(ck);
  if (hit != cache.results.end()) return hit->second;
  std::vector<EP> tf; if (byBounds) for (auto& f : facts) if (f.second && f.first->k == EK::Cmp) tf.push_back(f.first);
  std::map<EP, EP> m2;
  EP r = A.rewrite(c, [&](EP l) -> EP {
    if (!l->ty->isIntegerTy(1)) return A.conditionKeys(l).empty() ? l : nullptr;   // nothing below to decide: as it is
    auto it = facts.find(l); if (it != facts.end()) return A.mkBool(it->second);
    bool v; if (byBounds && A.decideLit(l, tf, v)) return A.mkBool(v);
    return nullptr;
  }, m2);
  cache.results[ck] = r;
  return r;
}
// an arm of a selection: under the condition's facts, with its own memo (the memo maps expressions to their forms
// under one set of facts); the entry point of the cache
static EP simplifyUnderArm(Arena& A, EP v, const std::function<bool(EP)>& fold, std::map<EP, bool>& facts, unsigned depth, bool valueLevel, Arena::SimplifyUnderCache& cache) {
  if (depth > 24) return v;
  ArrayRef<uint64_t> keys = A.selectionKeys(v);
  if (keys.empty()) return v;
  std::map<EP, bool> relevant;
  std::string ck = suKey(A, 'S', v, keys, facts, depth, valueLevel, &relevant);
  auto hit = cache.results.find(ck);
  if (hit != cache.results.end()) return hit->second;
  std::map<EP, EP> m2;
  EP r = simplifyUnderRec(A, v, fold, relevant, m2, depth, valueLevel, cache);
  cache.results[ck] = r;
  return r;
}
static EP simplifyUnderRec(Arena& A, EP e, const std::function<bool(EP)>& fold, std::map<EP, bool>& facts, std::map<EP, EP>& memo, unsigned depth, bool valueLevel, Arena::SimplifyUnderCache& cache) {
  if (depth > 24) return e;
  if (A.selectionKeys(e).empty()) return e;   // no selection: nothing to decide
  if (valueLevel) {
    auto it = memo.find(e); if (it != memo.end()) return it->second;
    EP r = e;
    if (e->k == EK::Ite) {
      EP c = e->args[0];
      EP d = facts.empty() ? c : decideCond(A, c, facts, true, cache);
      if (A.isTrue(d)) r = simplifyUnderRec(A, e->args[1], fold, facts, memo, depth + 1, true, cache);
      else if (A.isFalse(d)) r = simplifyUnderRec(A, e->args[2], fold, facts, memo, depth + 1, true, cache);
      else {
        auto arm = [&](EP v, bool val) { std::map<EP, bool> f2 = facts; addFact(A, c, val, f2); return simplifyUnderArm(A, v, fold, f2, depth + 1, true, cache); };
        r = A.ite(c, arm(e->args[1], true), arm(e->args[2], false));
      }
    } else if (e->k == EK::Concat || e->k == EK::Bytes) {
      std::vector<EP> args; bool ch = false;
      for (EP a : e->args) { EP s = simplifyUnderRec(A, a, fold, facts, memo, depth + 1, true, cache); ch |= s != a; args.push_back(s); }
      if (ch) r = e->k == EK::Concat ? A.concat(std::move(args), e->ty) : A.bytes(args[0], e->lo, e->len);
    } else r = simplifyUnderRec(A, e, fold, facts, memo, depth, false, cache);
    memo[e] = r;
    return r;
  }
  auto foldMemo = [&](EP c) -> bool {   // fold is a function of the condition: asked once per condition
    if (cache.fold.size() <= c->id) cache.fold.resize(A.size(), -1);
    int8_t& f = cache.fold[c->id];
    if (f < 0) { bool v = fold(c); if (cache.fold.size() <= c->id) cache.fold.resize(A.size(), -1); cache.fold[c->id] = v ? 1 : 0; return v; }
    return f != 0;
  };
  return A.rewrite(e, [&](EP x) -> EP {
    if (x->k != EK::Ite) return A.selectionKeys(x).empty() ? x : nullptr;   // no selection below: as it is
    if (depth > 24) return nullptr;
    EP c = x->args[0];
    if (!facts.empty() && foldMemo(c)) {
      EP d = decideCond(A, c, facts, true, cache);
      if (A.isTrue(d)) return simplifyUnderRec(A, x->args[1], fold, facts, memo, depth + 1, false, cache);
      if (A.isFalse(d)) return simplifyUnderRec(A, x->args[2], fold, facts, memo, depth + 1, false, cache);
    };
    auto arm = [&](EP v, bool val) { std::map<EP, bool> f2 = facts; addFact(A, c, val, f2); return simplifyUnderArm(A, v, fold, f2, depth + 1, false, cache); };
    return A.ite(c, arm(x->args[1], true), arm(x->args[2], false));
  }, memo);
}
EP Arena::simplifyUnder(EP e, const std::function<bool(EP)>& fold, SimplifyUnderCache* cache) {
  SimplifyUnderCache local;
  std::map<EP, bool> facts;
  return simplifyUnderArm(*this, e, fold, facts, 0, true, cache ? *cache : local);
}

EP Arena::shannon(EP e, const std::function<std::string(EP)>& litKey, unsigned maxLits) {
  // literals: Cmp nodes reachable through conditions (Ite conditions, boolean operators) and anywhere else
  std::vector<EP> lits;
  {
    std::set<EP> seen;
    std::vector<EP> work{e};
    while (!work.empty()) {
      EP x = work.back(); work.pop_back();
      if (!seen.insert(x).second) continue;
      if (x->k == EK::Cmp) { lits.push_back(x); continue; }
      if (x->k == EK::Poly) { for (auto& t : x->terms) for (auto a : t.first) work.push_back(atoms[a]); continue; }
      for (EP a : x->args) work.push_back(a);
    }
  }
  if (lits.empty()) return e;
  if (lits.size() > maxLits) return nullptr;
  // one decision variable per literal family: a literal and its inverse are keyed alike (SGT/NE through the inverse:
  // p+c > q is !(p+c-1 < q)), and the caller's key orders a family weakest first (ascending constant for SLT), so
  // that deciding a literal false settles every stronger one and the trees are threshold chains
  std::vector<std::pair<std::string, EP>> keyed;
  {
    std::set<std::string> have;
    for (EP l : lits) {
      EP k = l;
      if (l->op == CmpInst::ICMP_SGT || l->op == CmpInst::ICMP_NE || l->op == CmpInst::ICMP_UGT || l->op == CmpInst::ICMP_UGE || l->op == CmpInst::ICMP_SGE || l->op == CmpInst::FCMP_ONE || l->op == CmpInst::FCMP_UNE)
        k = cmp(CmpInst::getInversePredicate((CmpInst::Predicate)l->op), l->args[0], l->args[1]);
      if (k->k != EK::Cmp) k = l;
      std::string key = litKey(k);
      if (have.insert(key).second) keyed.push_back({key, k});
    }
  }
  std::sort(keyed.begin(), keyed.end(), [](auto& a, auto& b) { return a.first < b.first; });
  auto hasCmp = [&](EP x) {
    std::set<EP> seen; std::vector<EP> work{x};
    while (!work.empty()) {
      EP y = work.back(); work.pop_back();
      if (!seen.insert(y).second) continue;
      if (y->k == EK::Cmp) return true;
      if (y->k == EK::Poly) { for (auto& t : y->terms) for (auto a : t.first) work.push_back(atoms[a]); continue; }
      for (EP a : y->args) work.push_back(a);
    }
    return false;
  };
  std::function<EP(EP, size_t)> build = [&](EP x, size_t i) -> EP {
    if (i == keyed.size() || !hasCmp(x)) return x;
    EP l = keyed[i].second;
    EP linv = cmp(CmpInst::getInversePredicate((CmpInst::Predicate)l->op), l->args[0], l->args[1]);
    // assigning l also decides every literal it implies or excludes (linear implications between literals of one
    // family: p+c < q for different c), so that e.g. n+8 <= cols settles n < cols on that branch
    auto sub = [&](bool v) {
      EP L = v ? l : linv;
      std::map<EP, EP> memo;
      return rewrite(x, [&](EP n) -> EP {
        if (n->k != EK::Cmp) return nullptr;
        if (n == l) return mkBool(v);
        if (n == linv) return mkBool(!v);
        if (litImplies(L, n)) return mkBool(true);
        EP ninv = cmp(CmpInst::getInversePredicate((CmpInst::Predicate)n->op), n->args[0], n->args[1]);
        if (litImplies(L, ninv)) return mkBool(false);
        return nullptr;
      }, memo);
    };
    // only a literal that occurs decides here (its implications alone must not create a decision on it: the path
    // above may already have settled it)
    {
      bool occurs = false; std::set<EP> seen; std::vector<EP> work{x};
      while (!work.empty() && !occurs) {
        EP y = work.back(); work.pop_back();
        if (!seen.insert(y).second) continue;
        if (y == l || y == linv) { occurs = true; break; }
        if (y->k == EK::Poly) { for (auto& t : y->terms) for (auto a : t.first) work.push_back(atoms[a]); continue; }
        for (EP a : y->args) work.push_back(a);
      }
      if (!occurs) return build(x, i + 1);
    }
    EP t = sub(true), f = sub(false);
    if (t == x && f == x) return build(x, i + 1);   // literal absent
    t = build(t, i + 1); f = build(f, i + 1);
    if (t == f) return t;
    // a decision node, not re-folded into and/or (whose nesting would depend on construction order)
    if (t->k == EK::Undef) return f;
    if (f->k == EK::Undef) return t;
    Expr n; n.k = EK::Ite; n.ty = t->ty; n.args = {l, t, f};
    return intern(std::move(n));
  };
  EP r = build(e, 0);
  return r;
}

bool Executor::run(unsigned t, std::string& w) {
  tid = t; phase = 0; seq = 0; afterSkip = false; why.clear(); env.clear();
  Be = nullptr; beCond = nullptr; pendingBe = nullptr; pendingBeCond = nullptr; pendingBePhase = 0;
  EdgeMap in, out;
  in[&F.getEntryBlock()].push_back(Edge{nullptr, A.mkBool(true), {}});
  bool ok = execRegion(&F.getEntryBlock(), nullptr, in, out);
  if (ok && !afterSkip) ok = fail("the recovered K loop is not reached on the executed path");
  if (!Be && pendingBe) { Be = pendingBe; epiPhase = pendingBePhase; beCond = pendingBeCond; }
  if (ok && !Be && !retile && !smemReads.empty()) ok = fail("no barrier follows the K loop");
  finalPhase = phase;
  if (!ok) w = why;
  return ok;
}

}  // namespace sym
}  // namespace mvcc
