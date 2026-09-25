// Semantic tile IR. See stir.h.
#include "stir.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <set>

namespace mvcc {
namespace stir {

using namespace sym;
using coords::polyStr;
using coords::siteName;

namespace {

typedef std::vector<uint32_t> Mono;

// multiset inclusion of sorted atom lists: does `d` divide `m`?
bool divides(const Mono& d, const Mono& m) {
  size_t i = 0;
  for (uint32_t a : m) { if (i < d.size() && d[i] == a) i++; }
  return i == d.size();
}
Mono quotient(const Mono& m, const Mono& d) {
  Mono q; size_t i = 0;
  for (uint32_t a : m) { if (i < d.size() && d[i] == a) i++; else q.push_back(a); }
  return q;
}
int64_t gcd64(int64_t a, int64_t b) { a = a < 0 ? -a : a; b = b < 0 ? -b : b; while (b) { int64_t t = a % b; a = b; b = t; } return a; }

// element sizes of the pointer parameters, from the demangled signature ("void k<...>(const __half*, int, float*)")
std::vector<unsigned> pointerElemBytes(const std::string& sig) {
  std::vector<unsigned> out;
  size_t open = sig.find('('); if (open == std::string::npos) return out;
  // the parameter list is the last balanced parenthesis group
  size_t depth = 0, start = std::string::npos, end = std::string::npos;
  for (size_t i = 0; i < sig.size(); i++) {
    char c = sig[i];
    if (c == '(' || c == '<' || c == '[') { if (depth == 0 && c == '(') start = i; depth++; }
    else if (c == ')' || c == '>' || c == ']') { depth--; if (depth == 0 && c == ')') end = i; }
  }
  if (start == std::string::npos || end == std::string::npos || end <= start) return out;
  std::string params = sig.substr(start + 1, end - start - 1);
  std::vector<std::string> parts; { std::string cur; unsigned d = 0; for (char c : params) { if (c == '<' || c == '(') d++; if (c == '>' || c == ')') d--; if (c == ',' && d == 0) { parts.push_back(cur); cur.clear(); } else cur += c; } if (!cur.empty()) parts.push_back(cur); }
  for (std::string p : parts) {
    size_t star = p.find('*');
    if (star == std::string::npos) { out.push_back(0); continue; }
    std::string t = p.substr(0, star);
    auto strip = [&](const std::string& w) { size_t k; while ((k = t.find(w)) != std::string::npos) t.erase(k, w.size()); };
    strip("const"); strip("volatile"); strip("__restrict__"); strip("__restrict");
    t.erase(std::remove(t.begin(), t.end(), ' '), t.end());
    unsigned b = 0;
    if (t == "__half" || t == "__nv_bfloat16" || t == "half" || t == "short" || t == "unsignedshort" || t == "int16_t" || t == "uint16_t") b = 2;
    else if (t == "float" || t == "int" || t == "unsignedint" || t == "int32_t" || t == "uint32_t" || t == "unsigned") b = 4;
    else if (t == "unsignedchar" || t == "char" || t == "signedchar" || t == "bool" || t == "int8_t" || t == "uint8_t") b = 1;
    else if (t == "double" || t == "long" || t == "longlong" || t == "unsignedlong" || t == "unsignedlonglong" || t == "int64_t" || t == "uint64_t" || t == "size_t") b = 8;
    else if (t == "half2" || t == "__half2" || t == "__nv_bfloat162" || t == "float2" || t == "int2") b = 8;
    else if (t == "float4" || t == "int4" || t == "uint4") b = 16;
    out.push_back(b);
  }
  return out;
}

struct Term { Mono mono; int64_t coef; };
std::vector<Term> termsOf(EP p) { std::vector<Term> t; if (!p) return t; for (auto& kv : p->terms) t.push_back({kv.first, kv.second}); return t; }

}  // namespace

Program build(Arena& A, const coords::Discovery& D, const std::vector<sig::LoopClass>& classes, const std::vector<LoopStmt>& loopAlgebra, const std::string& demangledSignature) {
  Program P; P.kernel = demangledSignature;
  Type* i64 = Type::getInt64Ty(A.C);
  for (unsigned j = 0; j < D.d; j++) {
    P.coordNames.push_back("c" + std::to_string(j)); P.coordOutput.push_back(D.isOutput[j]);
    std::string s; bool first = true;
    for (size_t a = 0; a < D.axes.size(); a++) if (D.V[a][j]) { s += (first ? "" : " + ") + (D.V[a][j] == 1 ? "" : std::to_string(D.V[a][j]) + "*") + D.axisNames[a]; first = false; }
    P.coordPhys.push_back(first ? "0" : s);
  }
  P.derivedDefs = D.derivedDefs;
  P.loops = loopAlgebra;
  std::vector<unsigned> elemBySig = pointerElemBytes(demangledSignature);

  // ---- tensors: one per pointer argument some fitted access is an offset from
  std::map<uint32_t, unsigned> tensorIx;
  std::vector<std::vector<size_t>> tensorSites;
  for (size_t s = 0; s < D.sites.size(); s++) {
    const coords::SiteMap& m = D.sites[s];
    if (m.isPred || m.isDerivedOperand) continue;
    if (!m.fitted()) { P.unfitted.push_back(std::string(m.atomic ? "atomic " : m.isStore ? "store " : "load ") + std::to_string(m.bytes) + "B at " + siteName(m.site) + ": " + m.why); continue; }
    if (m.tensor.empty()) { P.unmapped++; continue; }
    auto it = tensorIx.find(m.tensorSym);
    if (it == tensorIx.end()) {
      Tensor t; t.name = m.tensor; t.sym = m.tensorSym;
      const Symbol& sy = A.symbol(m.tensorSym);
      if (sy.kind == Symbol::Arg && sy.v) if (auto* arg = dyn_cast<Argument>(sy.v)) {
        t.name = "arg" + std::to_string(arg->getArgNo());
        if (arg->getArgNo() < elemBySig.size() && elemBySig[arg->getArgNo()]) { t.elemBytes = elemBySig[arg->getArgNo()]; t.elemFromSignature = true; }
      }
      tensorIx[m.tensorSym] = (unsigned)P.tensors.size(); P.tensors.push_back(t); tensorSites.push_back({});
      it = tensorIx.find(m.tensorSym);
    }
    tensorSites[it->second].push_back(s);
    if (m.isStore) P.tensors[it->second].written = true; else P.tensors[it->second].read = true;
  }
  // the tensor's own atom in a base polynomial
  auto isTensorAtom = [&](const Mono& mono, uint32_t sym) { uint32_t s; return mono.size() == 1 && A.isSymAtom(mono[0], s) && s == sym; };

  // ---- dimensions: the distinct launch-parameter monomials of the strides; unit = gcd of every coefficient along it
  for (size_t ti = 0; ti < P.tensors.size(); ti++) {
    Tensor& T = P.tensors[ti];
    std::map<Mono, int64_t> g;   // monomial -> gcd of coefficients
    g[{}] = 0;
    for (size_t s : tensorSites[ti]) {
      const coords::SiteMap& m = D.sites[s];
      for (unsigned j = 0; j < D.d && j < m.coordStride.size(); j++) for (auto& t : termsOf(m.coordStride[j])) g[t.mono] = gcd64(g[t.mono], t.coef);
    }
    // base terms fold into the most specific dimension dividing them
    for (size_t s : tensorSites[ti]) {
      const coords::SiteMap& m = D.sites[s];
      for (auto& t : termsOf(m.base)) {
        if (isTensorAtom(t.mono, T.sym)) continue;
        const Mono* best = nullptr;
        for (auto& kv : g) if (!kv.first.empty() && divides(kv.first, t.mono) && (!best || kv.first.size() > best->size())) best = &kv.first;
        if (best) g[*best] = gcd64(g[*best], t.coef); else g[{}] = gcd64(g[{}], t.coef);
      }
      g[{}] = gcd64(g[{}], m.bytes);
    }
    // the contiguous dimension's unit is the element size when it divides everything constant
    int64_t contig = g[{}];
    if (T.elemBytes && contig % T.elemBytes == 0) contig = T.elemBytes;
    if (!T.elemBytes) T.elemBytes = (unsigned)contig;
    std::vector<std::pair<Mono, int64_t>> dims(g.begin(), g.end());
    // outermost first: higher degree first, then by atom ids; contiguous last
    std::sort(dims.begin(), dims.end(), [](const std::pair<Mono, int64_t>& a, const std::pair<Mono, int64_t>& b) { if (a.first.empty() != b.first.empty()) return b.first.empty(); if (a.first.size() != b.first.size()) return a.first.size() > b.first.size(); return a.first < b.first; });
    for (auto& d : dims) {
      Dim dm; dm.mono = d.first; dm.name = "d" + std::to_string(T.dims.size());
      int64_t u = d.first.empty() ? contig : d.second;
      if (u == 0) u = 1;
      std::vector<std::pair<std::vector<uint32_t>, int64_t>> tt{{d.first, u}};
      dm.unitBytes = A.mkPoly(i64, tt);
      T.dims.push_back(dm);
    }
  }

  // ---- accesses: per dimension, the index map (offset + coefficients) and the round-trip check on every sample
  for (size_t ti = 0; ti < P.tensors.size(); ti++) {
    Tensor& T = P.tensors[ti];
    for (size_t s : tensorSites[ti]) {
      const coords::SiteMap& m = D.sites[s];
      Access ac; ac.tensor = (unsigned)ti; ac.store = m.isStore; ac.atomic = m.atomic; ac.bytes = m.bytes; ac.site = m.site; ac.piece = m.piece; ac.pieces = m.pieces; ac.viaDerived = m.viaIndex; ac.samples = m.samples;
      ac.index.resize(T.dims.size());
      for (auto& im : ac.index) { im.coef.assign(D.d, 0); im.offset = A.mkPoly(i64, {}); }
      auto dimOf = [&](const Mono& mono) -> int {
        int best = -1;
        for (size_t k = 0; k < T.dims.size(); k++) if (!T.dims[k].mono.empty() && divides(T.dims[k].mono, mono) && (best < 0 || T.dims[k].mono.size() > T.dims[best].mono.size())) best = (int)k;
        if (best < 0) for (size_t k = 0; k < T.dims.size(); k++) if (T.dims[k].mono.empty()) best = (int)k;
        return best;
      };
      auto unitOf = [&](int k) { int64_t u = 1; A.constInt(A.mkPoly(i64, {{{}, T.dims[k].unitBytes->terms[0].second}}), u); return u; };
      for (unsigned j = 0; j < D.d && j < m.coordStride.size(); j++) for (auto& t : termsOf(m.coordStride[j])) {
        int k = dimOf(t.mono);
        int64_t u = unitOf(k);
        if (k < 0 || t.coef % u || (T.dims[k].mono != t.mono)) continue;   // cannot happen (dims are the strides' monomials); the round trip reports it
        ac.index[k].coef[j] += t.coef / u;
      }
      for (auto& t : termsOf(m.base)) {
        if (isTensorAtom(t.mono, T.sym)) continue;
        int k = dimOf(t.mono);
        int64_t u = k >= 0 ? unitOf(k) : 1;
        if (k < 0 || t.coef % u) { EP term = A.mkPoly(i64, {{t.mono, t.coef}}); ac.leftoverBytes = ac.leftoverBytes ? A.add(ac.leftoverBytes, term) : term; continue; }
        ac.index[k].offset = A.add(ac.index[k].offset, A.mkPoly(i64, {{quotient(t.mono, T.dims[k].mono), t.coef / u}}));
      }
      // round trip: tensor + sum_k unit_k * (offset_k + sum_j coef_kj * c_j(x)) + leftover == the sampled address
      const size_t nPhys = D.axes.size() - D.derivedValue.size();
      for (auto& pt : m.points) {
        std::vector<EP> c(D.d);
        for (unsigned j = 0; j < D.d; j++) {
          EP cj = A.mkPoly(i64, {});
          for (size_t a = 0; a < D.axes.size(); a++) {
            if (!D.V[a][j]) continue;
            EP xa = a < nPhys ? A.mkInt(i64, a < pt.first.size() ? pt.first[a] : 0) : D.derivedValue[a - nPhys];
            cj = A.add(cj, A.mul(A.mkInt(i64, D.V[a][j]), xa));
          }
          c[j] = cj;
        }
        EP addr = A.mkPoly(i64, {{{}, 0}});
        {   // the tensor atom
          for (auto& t : termsOf(m.base)) if (isTensorAtom(t.mono, T.sym)) addr = A.add(addr, A.mkPoly(i64, {{t.mono, t.coef}}));
        }
        for (size_t k = 0; k < T.dims.size(); k++) {
          EP idx = ac.index[k].offset;
          for (unsigned j = 0; j < D.d; j++) if (ac.index[k].coef[j]) idx = A.add(idx, A.mul(A.mkInt(i64, ac.index[k].coef[j]), c[j]));
          addr = A.add(addr, A.mul(T.dims[k].unitBytes, idx));
        }
        if (ac.leftoverBytes) addr = A.add(addr, ac.leftoverBytes);
        // Derived placeholders live in φ, not in c_j. The sampled address still carries them.
        for (size_t q = 0; q < D.derivedValue.size(); q++) {
          const size_t a = nPhys + q;
          if (a < m.stride.size() && m.stride[a] && D.derivedValue[q])
            addr = A.add(addr, A.mul(m.stride[a], D.derivedValue[q]));
        }
        EP diff = A.sub(A.retype(A.asPoly(pt.second), i64), addr);
        int64_t z;
        if (A.constInt(diff, z) && z == 0) ac.verified++; else ac.mismatches++;
      }
      P.verified += ac.verified; P.mismatches += ac.mismatches;
      P.accesses.push_back(ac);
    }
  }
  // ---- derived operand maps
  for (auto& m : D.sites) {
    if (!m.fitted() || !m.isDerivedOperand) continue;
    std::string s = "d" + std::to_string(m.derived) + " operand " + std::to_string(m.operand) + " = " + polyStr(A, m.base);
    for (unsigned j = 0; j < D.d && j < m.coordStride.size(); j++) if (m.coordStride[j]) s += " + (" + polyStr(A, m.coordStride[j]) + ")*c" + std::to_string(j);
    if (m.pieces > 1) s += " (one of " + std::to_string(m.pieces) + " families)";
    P.derivedOperands.push_back(s);
  }
  // ---- guards
  for (auto& m : D.sites) {
    if (!m.fitted() || !m.isPred) continue;
    Guard g; g.pred = m.pred; g.lhsBase = m.base; g.lhsCoef = m.coordStride; g.families = m.pieces;
    P.guards.push_back(g);
  }
  (void)classes;
  return P;
}

std::vector<std::string> dump(Arena& A, const Program& P) {
  std::vector<std::string> out;
  {
    std::string s = "stir: tensors=" + std::to_string(P.tensors.size()) + " coords=" + std::to_string(P.coordNames.size()) + " accesses=" + std::to_string(P.accesses.size()) + " guards=" + std::to_string(P.guards.size()) + " loops=" + std::to_string(P.loops.size());
    if (!P.operands.empty()) s += " operands=" + std::to_string(P.operands.size());
    s += " round-trip=" + std::to_string(P.verified) + "/" + std::to_string(P.verified + P.mismatches) + " samples";
    if (P.mismatches) s += " MISMATCH";
    if (P.unmapped) s += " unmapped-sites=" + std::to_string(P.unmapped);
    if (!P.unfitted.empty()) s += " unfitted-sites=" + std::to_string(P.unfitted.size());
    out.push_back(s);
  }
  for (auto& u : P.unfitted) out.push_back("  unfitted " + u);
  for (auto& T : P.tensors) {
    std::string s = "  tensor " + T.name + " elem=" + std::to_string(T.elemBytes) + "B" + (T.elemFromSignature ? "" : " (inferred)") + (T.read && T.written ? " read+written" : T.written ? " written" : " read") + " dims:";
    for (auto& d : T.dims) s += " " + d.name + "[stride " + polyStr(A, d.unitBytes) + "B]";
    out.push_back(s);
  }
  for (auto& ac : P.accesses) {
    const Tensor& T = P.tensors[ac.tensor];
    std::string s = std::string("  ") + (ac.atomic ? "atomic " : ac.store ? "store " : "load ") + std::to_string(ac.bytes) + "B " + T.name;
    for (size_t k = 0; k < T.dims.size(); k++) {
      const IndexMap& im = ac.index[k];
      std::string e; int64_t z;
      if (!(A.constInt(im.offset, z) && z == 0)) e = polyStr(A, im.offset);
      for (unsigned j = 0; j < im.coef.size(); j++) if (im.coef[j]) { if (!e.empty()) e += im.coef[j] < 0 ? " - " : " + "; else if (im.coef[j] < 0) e += "-"; int64_t v = im.coef[j] < 0 ? -im.coef[j] : im.coef[j]; e += (v == 1 ? "" : std::to_string(v) + "*") + P.coordNames[j]; }
      if (e.empty()) e = "0";
      s += "[" + e + "]";
    }
    if (ac.leftoverBytes) s += " + " + polyStr(A, ac.leftoverBytes) + "B";
    if (ac.viaDerived) s += " (through derived quantities)";
    if (ac.pieces > 1) s += " piece " + std::to_string(ac.piece) + "/" + std::to_string(ac.pieces);
    s += " at " + siteName(ac.site);
    if (ac.mismatches) s += " ROUND-TRIP MISMATCH " + std::to_string(ac.mismatches) + "/" + std::to_string(ac.verified + ac.mismatches);
    out.push_back(s);
  }
  for (auto& g : P.guards) {
    std::string s = "  guard " + std::string(CmpInst::getPredicateName((CmpInst::Predicate)g.pred)) + " lhs = " + polyStr(A, g.lhsBase);
    for (size_t j = 0; j < g.lhsCoef.size(); j++) if (g.lhsCoef[j]) s += " + (" + polyStr(A, g.lhsCoef[j]) + ")*" + P.coordNames[j];
    if (g.families > 1) s += " (one of " + std::to_string(g.families) + " families)";
    out.push_back(s);
  }
  for (size_t i = 0; i < P.derivedDefs.size(); i++) out.push_back("  derived " + P.derivedDefs[i]);
  for (auto& l : P.derivedOperands) out.push_back("    " + l);
  for (auto& L : P.loops) {
    std::string s = "  loop " + L.header + " depth=" + std::to_string(L.depth) + " " + sig::loopKindName(L.kind);
    if (L.kind == sig::LoopKind::Bounded) s += "(" + std::to_string(L.trip) + ")";
    if (L.kind == sig::LoopKind::Generic) s += L.uniform ? " uniform" : " per-thread";
    if (L.kind == sig::LoopKind::Refused) s += ": " + L.why;
    if (!L.algebra.empty()) {
      s += ":";
      for (auto& kv : L.algebra) s += " " + std::string(sig::algebraName(kv.first)) + "x" + std::to_string(kv.second);
      if (L.anyData) s += std::string(" -> permits:") + (L.permits.partition ? " partition" : "") + (L.permits.reorder ? " reorder" : "") + (L.permits.duplicate ? " duplicate" : "") + (L.permits.treeCombine ? " tree-combine" : "") + (!L.permits.partition && !L.permits.reorder && !L.permits.duplicate ? " nothing" : "");
    }
    out.push_back(s);
  }
  return out;
}

namespace direct {

Program elementwise(const std::string& name) {
  Program P;
  P.kernel = name;
  P.coordNames = {"i"};
  P.coordOutput = {true};
  P.coordPhys = {"b0+b1+b2+b3+b4+b5+b6+b7"};
  Tensor a, o;
  a.name = "x"; a.elemBytes = 4; a.read = true;
  o.name = "y"; o.elemBytes = 4; o.written = true;
  P.tensors = {a, o};
  Guard g; g.pred = 2; P.guards.push_back(g);
  return P;
}

Program gemm(const std::string& name) {
  Program P;
  P.kernel = name;
  P.coordNames = {"m", "n", "k"};
  P.coordOutput = {true, true, false};
  P.coordPhys = {"b0+b1+b2+b3+b4", "b5+b6+i1", "i0"};
  LoopStmt L;
  L.header = "%k"; L.depth = 1; L.kind = sig::LoopKind::Traversal;
  L.algebra.push_back({sig::Algebra::TensorSum, 1});
  L.permits = {true, true, false, true}; L.anyData = true;
  P.loops.push_back(L);
  Tensor A, B, C;
  A.name = "A"; A.elemBytes = 2; A.read = true;
  B.name = "B"; B.elemBytes = 2; B.read = true;
  C.name = "C"; C.elemBytes = 2; C.written = true;
  P.tensors = {A, B, C};
  return P;
}

Program attention(const std::string& name) {
  Program P;
  P.kernel = name;
  P.coordNames = {"d", "n", "k"};
  P.coordOutput = {true, false, false};
  P.coordPhys = {"b0+b1+b2+b3+b4", "i0", "i1"};
  LoopStmt on, sc;
  on.header = "%n"; on.depth = 1; on.kind = sig::LoopKind::Traversal;
  on.algebra.push_back({sig::Algebra::Tuple, 1});
  on.permits = {true, true, false, true}; on.anyData = true;
  sc.header = "%k"; sc.depth = 2; sc.kind = sig::LoopKind::Traversal;
  sc.algebra.push_back({sig::Algebra::Sum, 1});
  sc.permits = {true, true, false, true}; sc.anyData = true;
  P.loops = {on, sc};
  Tensor q, K, V, o;
  q.name = "Q"; q.elemBytes = 4; q.read = true;
  K.name = "K"; K.elemBytes = 4; K.read = true;
  V.name = "V"; V.elemBytes = 4; V.read = true;
  o.name = "O"; o.elemBytes = 4; o.written = true;
  P.tensors = {q, K, V, o};
  return P;
}

}  // namespace direct

}  // namespace stir
}  // namespace mvcc
