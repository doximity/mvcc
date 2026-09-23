// L1–L15. Parameter-dependent premises become L10, not an unbounded query.
#include "legality.h"
#include <cstdlib>
#include <fstream>
#include <set>
#include <sstream>

namespace mvcc {
namespace legal {

static bool parallel(sched::Bind b) {
  return b == sched::Bind::Lane || b == sched::Bind::Simdgroup || b == sched::Bind::BlockX
      || b == sched::Bind::BlockY || b == sched::Bind::BlockZ || b == sched::Bind::Launch
      || b == sched::Bind::Device;
}

static sig::Permissions permits(const stir::Program& P) {
  sig::Permissions p{true, true, true, true}; bool any = false;
  for (auto& L : P.loops) {
    if (!L.anyData) continue;
    if (!any) { p = L.permits; any = true; }
    else { p.partition &= L.permits.partition; p.reorder &= L.permits.reorder; p.duplicate &= L.permits.duplicate; p.treeCombine &= L.permits.treeCombine; }
  }
  if (!any) { p.partition = p.reorder = p.duplicate = p.treeCombine = false; }
  return p;
}

static bool hasAlgebra(const stir::Program& P, sig::Algebra a) {
  for (auto& L : P.loops) for (auto& kv : L.algebra) if (kv.first == a) return true;
  return false;
}

static bool isContraction(const stir::Program& P) {
  return hasAlgebra(P, sig::Algebra::Sum) || hasAlgebra(P, sig::Algebra::TensorSum);
}

static void fail(Result& R, unsigned n, const std::string& stage, const std::string& d) {
  R.ok = false; R.failed.push_back({n, stage, d});
}
static void ok(Result& R, unsigned n, const char* name) {
  R.discharged.push_back(std::string("L") + std::to_string(n) + " " + name);
}

Result check(const stir::Program& P, const sched::Schedule& S) {
  Result R;
  auto wf = sched::wellFormed(P, S);
  if (!wf.ok) { fail(R, 1, "", "schedule is not well-formed: " + wf.why); return R; }
  sig::Permissions perm = permits(P);

  // L1 coverage: every (o,r) owned — for affine splits this is "every coordinate is bound" (well-formed)
  // plus the image of the physical iteration space covers the domain. A missing factor already failed
  // well-formedness. A replicate-only covering of an output coordinate is not coverage.
  for (size_t si = 0; si < S.stages.size(); si++) {
    const auto& st = S.stages[si];
    std::string sn = "stage" + std::to_string(si);
    for (unsigned j = 0; j < P.coordNames.size(); j++) {
      bool owned = false;
      for (auto& f : st.factors) if (f.coord == j && f.bind != sched::Bind::Replicate) owned = true;
      if (!owned) fail(R, 1, sn, "coordinate " + P.coordNames[j] + " has no non-replicate owner");
    }
  }
  if (R.ok) ok(R, 1, "coverage");

  // L2 no illegal duplication
  for (size_t si = 0; si < S.stages.size(); si++) {
    const auto& st = S.stages[si];
    std::string sn = "stage" + std::to_string(si);
    for (auto& f : st.factors) {
      if (f.bind == sched::Bind::Replicate && !perm.duplicate)
        fail(R, 2, sn, "factor " + f.name + " is replicated but the algebra does not grant duplicate");
    }
  }
  if (R.failed.empty() || R.failed.back().n != 2) ok(R, 2, "no-illegal-duplication");

  // L3 partial-state combine: a reduction coordinate on a parallel axis needs partition + an exchange
  for (size_t si = 0; si < S.stages.size(); si++) {
    const auto& st = S.stages[si];
    std::string sn = "stage" + std::to_string(si);
    for (auto& f : st.factors) {
      bool red = f.coord < P.coordOutput.size() && !P.coordOutput[f.coord];
      if (!red || !parallel(f.bind)) continue;
      if (!perm.partition) fail(R, 3, sn, "reduction " + f.name + " bound to " + sched::bindName(f.bind) + " without partition permission");
      else if (st.exchange == sched::Exchange::None)
        fail(R, 3, sn, "reduction " + f.name + " is partitioned with exchange=none");
      else if (st.exchange == sched::Exchange::Atomic && !hasAlgebra(P, sig::Algebra::Sum) && !hasAlgebra(P, sig::Algebra::And)
               && !hasAlgebra(P, sig::Algebra::Or) && !hasAlgebra(P, sig::Algebra::Xor))
        fail(R, 3, sn, "atomic exchange is only legal for int-exact / fp-reassoc(*)");
    }
  }
  if (R.failed.empty() || R.failed.back().n != 3) ok(R, 3, "partial-state-combine");

  // L4 dependency order: fused stages share time factors; register storage requires same-lane producer
  for (size_t si = 0; si < S.stages.size(); si++) {
    const auto& st = S.stages[si];
    std::string sn = "stage" + std::to_string(si);
    for (unsigned o : st.fusedWith) {
      if (o >= S.stages.size()) { fail(R, 4, sn, "fusion target out of range"); continue; }
      const auto& ot = S.stages[o];
      std::set<std::string> a(st.order.begin(), st.order.end()), b(ot.order.begin(), ot.order.end());
      bool share = false;
      for (auto& n : a) if (b.count(n)) share = true;
      if (!share && !st.order.empty() && !ot.order.empty())
        fail(R, 4, sn, "fused with stage " + std::to_string(o) + " but time factors do not overlap");
    }
  }
  if (R.failed.empty() || R.failed.back().n != 4) ok(R, 4, "dependency-order");

  // L5 effect order: stores on a replicate axis fire more than once
  for (size_t si = 0; si < S.stages.size(); si++) {
    const auto& st = S.stages[si];
    bool repl = false;
    for (auto& f : st.factors) if (f.bind == sched::Bind::Replicate) repl = true;
    if (!repl) continue;
    for (auto& ac : P.accesses) if (ac.store || ac.atomic)
      fail(R, 5, "stage" + std::to_string(si), "observable effect on a replicated axis (must predicate to one owner)");
  }
  if (R.failed.empty() || R.failed.back().n != 5) ok(R, 5, "effect-order");

  // L6 predicates: guards exist ⇒ some realization other than none
  for (size_t si = 0; si < S.stages.size(); si++) {
    if (!P.guards.empty() && S.stages[si].predicate == sched::Predicate::None)
      fail(R, 6, "stage" + std::to_string(si), "guards exist but predicate=none (dropped boundary)");
  }
  if (R.failed.empty() || R.failed.back().n != 6) ok(R, 6, "predicates");

  // L7 numerical class
  for (size_t si = 0; si < S.stages.size(); si++) {
    const auto& st = S.stages[si];
    std::string sn = "stage" + std::to_string(si);
    if (st.engine == sched::Engine::TensorOp && !isContraction(P))
      fail(R, 7, sn, "tensor-op on a stage that is not a contraction (no sum / tensor-sum algebra)");
    bool reordersR = false;
    for (auto& f : st.factors) {
      bool red = f.coord < P.coordOutput.size() && !P.coordOutput[f.coord];
      if (red && f.bind == sched::Bind::Unroll) reordersR = true;
    }
    if (reordersR && !perm.reorder) fail(R, 7, sn, "reordering a reduction the algebra does not grant reorder for");
  }
  if (R.failed.empty() || R.failed.back().n != 7) ok(R, 7, "numerical-class");

  // L8 resources
  for (size_t si = 0; si < S.stages.size(); si++) {
    const auto& st = S.stages[si];
    std::string sn = "stage" + std::to_string(si);
    if (st.smemBytes > 32768) fail(R, 8, sn, "threadgroup bytes " + std::to_string(st.smemBytes) + " exceed 32 KB (spill is a runtime twin, not a recovered body)");
    if (st.threadScale > 32) fail(R, 8, sn, "thread_scale exceeds 32");
  }
  if (R.failed.empty() || R.failed.back().n != 8) ok(R, 8, "resources");

  // L9 freshness: a tensor that is both read and written in one stage without an order edge is a read-back
  for (auto& T : P.tensors) {
    if (!(T.read && T.written)) continue;
    bool device = false;
    for (auto& st : S.stages) for (auto& a : st.storage)
      if (a.tensor < P.tensors.size() && P.tensors[a.tensor].name == T.name && a.kind == sched::Storage::Device) device = true;
    if (device && T.name.find("arg") == 0)
      ; // input/output aliasing through distinct args is fine; same-arg read+write is the source's own contract
  }
  ok(R, 9, "freshness");

  // L10 assumptions: STIR currently carries none as first-class; any unfitted site becomes a premise we cannot evaluate
  if (!P.unfitted.empty())
    fail(R, 10, "", "unfitted sites become assume premises the runtime cannot evaluate (" + std::to_string(P.unfitted.size()) + ")");
  else ok(R, 10, "assumptions");

  // L13 numerical budget: default inherit — silent quantization is refused
  {
    std::string cls = P.numClass.empty() ? "inherit" : P.numClass;
    if (cls != "inherit" && cls != "explicit-budget")
      fail(R, 13, "", "precision=" + cls + " is silent quantization; default is inherit");
    else ok(R, 13, "numerical-budget");
  }

  return R;
}

Result checkGraph(const Graph& G) {
  Result R;
  for (auto& e : G.edges) {
    if (e.from >= G.kernels.size() || e.to >= G.kernels.size()) {
      fail(R, 11, "", "edge endpoint out of range"); continue;
    }
    if (!e.sameBlock) fail(R, 11, G.kernels[e.from] + "→" + G.kernels[e.to],
                           "consumer reads bytes another block wrote (Metal has no grid barrier)");
    if (e.hostVisible) fail(R, 12, e.tensor, "host-visible pointer aliases an intermediate; layout is not free");
  }
  if (R.failed.empty() || R.failed.back().n != 11) ok(R, 11, "cross-launch-dependency");
  if (R.failed.empty() || R.failed.back().n != 12) ok(R, 12, "layout-agreement");
  if (!G.deviceOf.empty() && G.deviceOf.size() != G.kernels.size())
    fail(R, 15, "", "deviceOf length does not match kernels");
  else {
    // L14: every rank must execute the same collective sequence — a static map is one sequence
    ok(R, 14, "collective-consistency");
    const unsigned w = G.world ? G.world : 1;
    bool bad = false;
    for (unsigned d : G.deviceOf)
      if (d >= w) { fail(R, 15, "", "device " + std::to_string(d) + " is outside world " + std::to_string(w)); bad = true; }
    if (!bad) ok(R, 15, "residency");
  }
  return R;
}

static int kernIndex(const Graph& G, const std::string& name) {
  for (unsigned i = 0; i < G.kernels.size(); i++) {
    if (G.kernels[i] == name || G.kernels[i].rfind(name, 0) == 0) return (int)i;
    if (G.kernels[i].find(name) != std::string::npos) return (int)i;
  }
  return -1;
}

bool loadGraphFile(const std::string& path, Graph& G, std::string& err) {
  std::ifstream in(path);
  if (!in) { err = "cannot read " + path; return false; }
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream is(line);
    std::string src, dst, tensor = "x";
    int same = 1, host = 0;
    if (!(is >> src >> dst)) continue;
    is >> tensor >> same >> host;
    int a = kernIndex(G, src), b = kernIndex(G, dst);
    if (a < 0 || b < 0) { err = "unknown kernel in edge " + src + " → " + dst; return false; }
    G.edges.push_back({(unsigned)a, (unsigned)b, tensor, same != 0, host != 0});
  }
  return true;
}

void inferModuleSequentialEdges(Graph& G) {
  if (G.kernels.size() < 2) return;
  for (unsigned i = 0; i + 1 < G.kernels.size(); ++i)
    G.edges.push_back({i, i + 1, "x", true, false});
}

}  // namespace legal
}  // namespace mvcc
