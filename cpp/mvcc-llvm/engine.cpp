// STIR engine: targets, search, explain, self-test.
#include "engine.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include <cstdlib>
#include <memory>
#include <sstream>

namespace mvcc {
namespace engine {

Target metal4() { return {"metal4-tensorops", {"tensor-op", "scalar-fma", "simd-vector"}, 32768, 125, 18.5}; }
Target simdgroupMatrix() { return {"simdgroup-matrix", {"tensor-op", "scalar-fma"}, 32768, 125, 7.4}; }
Target scalarMSL() { return {"scalar-msl", {"scalar-fma", "simd-vector"}, 32768, 125, 1.0}; }
Target nvptxOracle() { return {"nvptx-oracle", {"mma.sync", "scalar-fma"}, 232448, 2000, 150}; }  // C layout = mma.sync fragments; acc2ptx is identity

const char* precisionName(Precision p) {
  switch (p) {
    case Precision::Inherit: return "inherit";
    case Precision::FpReassoc32: return "fp-reassoc(32)";
    case Precision::FpReassocStar: return "fp-reassoc(*)";
    case Precision::ExplicitBudget: return "explicit-budget";
  }
  return "?";
}

EmitPlan planEmit(const stir::Program& P, const sched::Schedule& S, const legal::Result& L) {
  EmitPlan E;
  if (!L.ok) { E.canEmit = false; E.why = L.first(); return E; }
  E.canEmit = true;
  if (S.why == "source") E.why = "identity: exact twin is a legal emission of the source schedule";
  else E.why = "check+emit: " + S.why;
  E.steps.push_back("loop nest from order (" + std::to_string(S.stages.empty() ? 0 : S.stages[0].order.size()) + " time factors)");
  E.steps.push_back("coordinates from bind (inverse of coordPhys for the new schedule, strength-reduced)");
  E.steps.push_back("accesses by storage; point function instantiated; combine by algebra and engine");
  E.steps.push_back("effects predicated to their owner; ABI from the schedule");
  if (!P.accesses.empty())
    E.steps.push_back("shared-fill dedup over " + std::to_string(P.accesses.size()) + " accesses");
  return E;
}

// residual = last measured seconds − roofline prior, per (kernel, why).
// Used only when it changes the held-out order vs prior-only; otherwise thrown away.
static bool dbResidual(const std::string& kernel, const std::string& why, double prior, double& residual) {
  const char* db = getenv("MVCC_SCHED_DB");
  if (!db || kernel.empty()) return false;
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> mb = llvm::MemoryBuffer::getFile(db);
  if (!mb) return false;
  std::string shortName = kernel;
  auto par = kernel.find('(');
  if (par != std::string::npos) shortName = kernel.substr(0, par);
  std::istringstream in(mb.get()->getBuffer().str());
  std::string line;
  bool hit = false;
  while (std::getline(in, line)) {
    std::istringstream ls(line);
    std::string key, w, legal, cost;
    if (!std::getline(ls, key, '\t') || !std::getline(ls, w, '\t')) continue;
    if (key != kernel && key != shortName) continue;
    if (w != why) continue;
    std::string rest;
    std::getline(ls, rest);
    // optional 5th field: measured seconds
    double meas = 0;
    auto t = rest.rfind('\t');
    if (t != std::string::npos) meas = atof(rest.c_str() + t + 1);
    else meas = atof(rest.c_str());
    if (meas <= 0) continue;
    residual = meas - prior;
    hit = true;
  }
  return hit;
}

static std::string dbPreferredWhy(const std::string& kernel) {
  const char* db = getenv("MVCC_SCHED_DB");
  if (!db || kernel.empty()) return {};
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> mb = llvm::MemoryBuffer::getFile(db);
  if (!mb) return {};
  std::string last;
  std::string shortName = kernel;
  auto par = kernel.find('(');
  if (par != std::string::npos) shortName = kernel.substr(0, par);
  std::istringstream in(mb.get()->getBuffer().str());
  std::string line;
  while (std::getline(in, line)) {
    auto tab0 = line.find('\t');
    if (tab0 == std::string::npos) continue;
    std::string key = line.substr(0, tab0);
    if (key != kernel && key != shortName) continue;
    std::string rest = line.substr(tab0 + 1);
    auto tab = rest.find('\t');
    last = tab == std::string::npos ? rest : rest.substr(0, tab);
  }
  return last;
}

Chosen choose(const stir::Program& P) {
  Chosen C;
  auto cands = sched::candidates(P);
  std::string prefer = dbPreferredWhy(P.kernel);
  Chosen best; bool have = false; bool dbHit = false;
  for (auto& c : cands) {
    auto L = legal::check(P, c.S);
    if (!L.ok) continue;
    bool thisDb = !prefer.empty() && c.S.why == prefer;
    bool take = !have;
    if (have && thisDb && !dbHit) take = true;
    else if (have && thisDb == dbHit) {
      double rC = 0, rB = 0;
      bool useR = dbResidual(P.kernel, c.S.why, c.cost.seconds, rC)
               && dbResidual(P.kernel, best.S.why, best.cost.seconds, rB);
      double sC = c.cost.seconds + (useR ? rC : 0);
      double sB = best.cost.seconds + (useR ? rB : 0);
      take = sC < sB || (sC == sB && c.S.why == "source");
    }
    if (take) {
      best.S = c.S; best.cost = c.cost; best.legality = L; have = true; dbHit = thisDb;
    }
  }
  if (!have) {
    best.S = sched::sourceSchedule(P);
    best.cost = sched::costOf(P, best.S);
    best.legality = legal::check(P, best.S);
  }
  if (dbHit) best.dbWhy = prefer;
  best.emit = planEmit(P, best.S, best.legality);
  return best;
}

std::vector<std::string> explain(const stir::Program& P, const Chosen& C) {
  std::vector<std::string> out;
  out.push_back("explain " + P.kernel);
  {
    std::string eq = "  Y[";
    bool first = true;
    for (unsigned j = 0; j < P.coordNames.size(); j++) if (j >= P.coordOutput.size() || P.coordOutput[j]) {
      if (!first) eq += ",";
      eq += P.coordNames[j]; first = false;
    }
    eq += "] = ⊕_[";
    first = true;
    for (unsigned j = 0; j < P.coordNames.size(); j++) if (j < P.coordOutput.size() && !P.coordOutput[j]) {
      if (!first) eq += ",";
      eq += P.coordNames[j]; first = false;
    }
    if (first) eq += "∅";
    eq += "] F(";
    first = true;
    for (auto& T : P.tensors) if (T.read) { if (!first) eq += ", "; eq += T.name; first = false; }
    eq += ")";
    out.push_back(eq);
  }
  out.push_back("  source schedule recovered; chosen why=" + C.S.why);
  out.push_back("  cost " + C.cost.breakdown);
  out.push_back("  " + C.legality.first());
  for (auto& d : C.legality.discharged) out.push_back("    discharged " + d);
  for (auto& f : C.legality.failed) out.push_back("    " + f.str());
  out.push_back("  emit " + C.emit.why);
  for (auto& s : C.emit.steps) out.push_back("    " + s);
  out.push_back("  precision " + std::string(precisionName(Precision::Inherit)) + " (silent quantization is refused)");
  out.push_back("  target " + metal4().name + " smem=" + std::to_string(metal4().smemBytes)
                + " bw=" + std::to_string((int)metal4().bwGBps) + "GB/s peak=" + std::to_string(metal4().tflops) + " TFLOP/s");
  return out;
}

void analyze(const stir::Program& P, std::vector<std::string>& out) {
  const char* en = getenv("MVCC_SCHED");
  if (!en || en[0] == '0') return;
  Chosen C = choose(P);
  for (auto& l : sched::dump(C.S)) out.push_back(l);
  out.push_back("  cost " + C.cost.breakdown);
  out.push_back("  " + C.legality.first());
  if (getenv("MVCC_EXPLAIN") && getenv("MVCC_EXPLAIN")[0] == '1')
    for (auto& l : explain(P, C)) out.push_back(l);
  if (C.legality.ok)
    out.push_back("  check+emit: " + C.legality.first() + " why=" + C.S.why);
}

static stir::Program synthGEMM() {
  stir::Program P;
  P.kernel = "synthetic_gemm";
  P.coordNames = {"c0", "c1", "c2"};
  P.coordOutput = {true, true, false};
  P.coordPhys = {"b0 + 2*b1 + 4*b2 + 8*b3 + 16*b4", "b5 + 2*b6 + 4*i1", "i0"};
  stir::LoopStmt L; L.header = "%k"; L.depth = 1; L.kind = sig::LoopKind::Traversal;
  L.algebra.push_back({sig::Algebra::Sum, 32});
  L.permits = {true, true, false, true}; L.anyData = true;
  P.loops.push_back(L);
  stir::Tensor A, B, C;
  A.name = "arg0"; A.elemBytes = 2; A.read = true;
  B.name = "arg1"; B.elemBytes = 2; B.read = true;
  C.name = "arg2"; C.elemBytes = 2; C.written = true;
  P.tensors = {A, B, C};
  stir::Access ac; ac.tensor = 0; ac.bytes = 16; ac.samples = 128; ac.index.resize(1);
  P.accesses.push_back(ac);
  return P;
}

static stir::Program synthElem() {
  stir::Program P;
  P.kernel = "synthetic_silu";
  P.coordNames = {"c0"};
  P.coordOutput = {true};
  P.coordPhys = {"b0 + 2*b1 + 4*b2 + 8*b3 + 16*b4 + 32*b5 + 64*b6 + 128*b7"};
  stir::Tensor A, G, O;
  A.name = "arg0"; A.elemBytes = 4; A.read = true;
  G.name = "arg1"; G.elemBytes = 4; G.read = true;
  O.name = "arg2"; O.elemBytes = 4; O.written = true;
  P.tensors = {A, G, O};
  stir::Guard g; g.pred = 40; P.guards.push_back(g);   // some icmp
  return P;
}

static int expect(llvm::raw_ostream& os, bool cond, const char* what, int& fails) {
  if (cond) { os << "  ok  " << what << "\n"; return 0; }
  os << "  FAIL " << what << "\n"; fails++; return 1;
}

int selfTest(llvm::raw_ostream& os) {
  int fails = 0;
  os << "engine self-test\n";

  // targets
  expect(os, metal4().smemBytes == 32768 && nvptxOracle().name == "nvptx-oracle", "targets", fails);

  stir::Program G = synthGEMM();
  stir::Program E = synthElem();

  // source schedule of a GEMM is well-formed and legal
  auto Sg = sched::sourceSchedule(G);
  auto wf = sched::wellFormed(G, Sg);
  expect(os, wf.ok, ("source well-formed: " + wf.why).c_str(), fails);
  auto Lg = legal::check(G, Sg);
  expect(os, Lg.ok, ("source legal: " + Lg.first()).c_str(), fails);

  // parse/print round-trip
  {
    std::string text;
    for (auto& l : sched::dump(Sg)) text += l + "\n";
    sched::Schedule R; std::string err;
    bool okp = sched::parseText(text, R, err);
    expect(os, okp && sched::wellFormed(G, R).ok, ("parse round-trip: " + err).c_str(), fails);
  }

  // every schedule edit dumps, parses, and is well-formed (legality is L1–L15 below)
  {
    unsigned n = 0, ok = 0;
    for (auto& c : sched::candidates(G)) {
      n++;
      std::string text;
      for (auto& l : sched::dump(c.S)) text += l + "\n";
      sched::Schedule R; std::string err;
      if (sched::parseText(text, R, err) && sched::wellFormed(G, R).ok) ok++;
    }
    expect(os, n >= 6 && ok == n, "L1–5 schedule edits dump/parse/well-formed round-trip", fails);
    expect(os, sched::stageShape(Sg).find("contraction") != std::string::npos,
           "source schedule prints a contraction stage-shape", fails);
  }

  // L3: partition a reduction with exchange=none
  {
    auto S = Sg;
    for (auto& st : S.stages) {
      st.exchange = sched::Exchange::None;
      for (auto& f : st.factors) if (f.coord == 2) f.bind = sched::Bind::Simdgroup;
    }
    auto L = legal::check(G, S);
    bool hit = false; for (auto& f : L.failed) if (f.n == 3) hit = true;
    expect(os, !L.ok && hit, "L3 rejects partitioned reduction with exchange=none", fails);
  }

  // L2: replicate without duplicate permission (GEMM sum does not grant duplicate)
  {
    auto S = sched::editReplicate(Sg);
    auto L = legal::check(G, S);
    bool hit = false; for (auto& f : L.failed) if (f.n == 2) hit = true;
    expect(os, !L.ok && hit, "L2 rejects replicate without duplicate permission", fails);
  }

  // L6: drop predicates on an elementwise kernel that has guards
  {
    auto Se = sched::sourceSchedule(E);
    for (auto& st : Se.stages) st.predicate = sched::Predicate::None;
    auto L = legal::check(E, Se);
    bool hit = false; for (auto& f : L.failed) if (f.n == 6) hit = true;
    expect(os, !L.ok && hit, "L6 rejects dropped predicates", fails);
  }

  // L7: tensor-op on a non-contraction
  {
    auto Se = sched::editEngineTensorOp(sched::sourceSchedule(E));
    auto L = legal::check(E, Se);
    bool hit = false; for (auto& f : L.failed) if (f.n == 7) hit = true;
    expect(os, !L.ok && hit, "L7 rejects tensor-op on elementwise", fails);
  }

  // L8: oversize smem
  {
    auto S = Sg; S.stages[0].smemBytes = 40000;
    auto L = legal::check(G, S);
    bool hit = false; for (auto& f : L.failed) if (f.n == 8) hit = true;
    expect(os, !L.ok && hit, "L8 rejects smem > 32 KB", fails);
  }

  // L1: drop a coordinate bind
  {
    auto S = Sg;
    if (!S.stages.empty() && !S.stages[0].factors.empty()) S.stages[0].factors.pop_back();
    auto L = legal::check(G, S);
    bool hit = false; for (auto& f : L.failed) if (f.n == 1) hit = true;
    expect(os, !L.ok && hit, "L1 rejects an unbound coordinate", fails);
  }

  // L4: fuse two stages that share no time factor
  {
    auto S = Sg;
    if (S.stages.size() < 2) S.stages.push_back(S.stages[0]);
    S.stages[0].fusedWith = {1};
    S.stages[0].order = {"c2_k"};
    sched::Factor extra; extra.name = "unrelated"; extra.coord = 2; extra.bind = sched::Bind::Time;
    S.stages[1].factors.push_back(extra);
    S.stages[1].order = {"unrelated"};
    auto L = legal::check(G, S);
    bool hit = false; for (auto& f : L.failed) if (f.n == 4) hit = true;
    expect(os, !L.ok && hit, "L4 rejects fusion without a shared time nest", fails);
  }

  // L5: observable store on a replicated axis
  {
    stir::Program G2 = G;
    stir::Access st; st.tensor = 2; st.store = true; st.bytes = 2; G2.accesses.push_back(st);
    auto S = sched::editReplicate(sched::sourceSchedule(G2));
    auto L = legal::check(G2, S);
    bool hit = false; for (auto& f : L.failed) if (f.n == 5) hit = true;
    expect(os, !L.ok && hit, "L5 rejects effects on a replicated axis", fails);
  }

  // L10: unfitted sites become unevaluable assumes
  {
    stir::Program G2 = G; G2.unfitted.push_back("scatter at %x: data-dependent");
    auto L = legal::check(G2, sched::sourceSchedule(G2));
    bool hit = false; for (auto& f : L.failed) if (f.n == 10) hit = true;
    expect(os, !L.ok && hit, "L10 rejects unevaluable assume premises", fails);
  }

  // search keeps source and finds a legal candidate
  {
    auto cands = sched::candidates(G);
    bool src = false; for (auto& c : cands) if (c.S.why == "source") src = true;
    expect(os, src && !cands.empty(), "search retains the source schedule", fails);
    Chosen C = choose(G);
    expect(os, C.legality.ok && C.emit.canEmit, "choose emits a legal body", fails);
  }

  // search finds every hand-written schedule within 1.1× of the cheapest legal
  {
    auto within = [&](const stir::Program& P) {
      auto cands = sched::candidates(P);
      double best = 1e300;
      for (auto& c : cands) {
        if (!legal::check(P, c.S).ok) continue;
        double s = sched::costOf(P, c.S).seconds;
        if (s < best) best = s;
      }
      if (best >= 1e300) return false;
      for (auto& c : cands) {
        if (!legal::check(P, c.S).ok) continue;
        if (sched::costOf(P, c.S).seconds > best * 1.1 + 1e-12) return false;
      }
      return true;
    };
    expect(os, within(stir::direct::elementwise("silu_mul"))
            && within(stir::direct::gemm("hgemm16"))
            && within(stir::direct::attention("attn")),
           "search finds hand-written schedules within 1.1× of the cheapest legal", fails);
  }

  // cost: device-stream (no ring) ranks at least as well as a spilled ring
  {
    auto src = sched::sourceSchedule(G);
    auto ring = src; ring.stages[0].smemBytes = 49152; for (auto& a : ring.stages[0].storage) a.kind = sched::Storage::Threadgroup;
    auto stream = sched::editDeviceStream(src);
    auto cr = sched::costOf(G, ring), cs = sched::costOf(G, stream);
    expect(os, cs.seconds < cr.seconds, "cost model prefers device-stream over a spilled ring", fails);
  }

  // graph L11 / L12
  {
    legal::Graph okG;
    okG.kernels = {"norm", "gemm"};
    okG.edges.push_back({0, 1, "x", true, false});
    okG.deviceOf = {0, 0};
    expect(os, legal::checkGraph(okG).ok, "same-block fusion is legal", fails);
    legal::Graph bad = okG; bad.edges[0].sameBlock = false;
    auto L = legal::checkGraph(bad);
    bool hit = false; for (auto& f : L.failed) if (f.n == 11) hit = true;
    expect(os, !L.ok && hit, "L11 rejects a cross-block fusion", fails);
    bad = okG; bad.edges[0].hostVisible = true;
    L = legal::checkGraph(bad);
    hit = false; for (auto& f : L.failed) if (f.n == 12) hit = true;
    expect(os, !L.ok && hit, "L12 rejects a host-visible relayout", fails);
  }

  // explain produces the equation line
  {
    Chosen C = choose(G);
    auto lines = explain(G, C);
    bool eq = false; for (auto& l : lines) if (l.find("Y[") != std::string::npos) eq = true;
    expect(os, eq, "explain prints the recovered equation", fails);
  }

  // precision default + L13 refuses silent quantization
  expect(os, std::string(precisionName(Precision::Inherit)) == "inherit", "precision defaults to inherit", fails);
  {
    stir::Program E2 = E; E2.numClass = "fp8";
    auto L = legal::check(E2, sched::sourceSchedule(E2));
    bool hit = false; for (auto& f : L.failed) if (f.n == 13) hit = true;
    expect(os, !L.ok && hit, "L13 refuses silent quantization (fp8 without a budget)", fails);
  }

  // bind:device is a well-formed parallel owner
  {
    auto Sd = sched::editBindDevice(sched::sourceSchedule(G));
    expect(os, Sd.why == "bind:device" && !Sd.stages.empty() && Sd.stages[0].exchange == sched::Exchange::Collective
               && legal::check(G, Sd).ok,
           "bind:device is a legal tensor-parallel owner (exchange=collective → ncclAllGather)", fails);
  }

  // graph file
  {
    llvm::SmallString<128> path;
    std::error_code EC = llvm::sys::fs::createTemporaryFile("mvcc-graph", "txt", path);
    if (!EC) {
      llvm::raw_fd_ostream gf(path, EC, llvm::sys::fs::OF_Text);
      gf << "norm gemm x 1 0\n";
      gf.close();
      legal::Graph gg; gg.kernels = {"norm", "gemm"};
      std::string err;
      bool okp = legal::loadGraphFile(path.str().str(), gg, err);
      expect(os, okp && gg.edges.size() == 1 && legal::checkGraph(gg).ok,
             "MVCC_GRAPH same-block edge is legal (L11/L12)", fails);
      legal::Graph bad = gg; bad.world = 2; bad.deviceOf = {0, 3};
      auto Lb = legal::checkGraph(bad);
      bool hit15 = false; for (auto& f : Lb.failed) if (f.n == 15) hit15 = true;
      expect(os, !Lb.ok && hit15, "L15 rejects a device id outside world", fails);
      llvm::sys::fs::remove(path);
    } else {
      expect(os, false, "MVCC_GRAPH same-block edge is legal (L11/L12)", fails);
    }
  }

  // a Program built without CUDA recovery is a legal client of choose/check
  {
    auto Pe = stir::direct::elementwise("direct_silu");
    auto Pg = stir::direct::gemm("direct_gemm");
    auto Pa = stir::direct::attention("direct_attn");
    expect(os, choose(Pe).legality.ok && choose(Pg).legality.ok && choose(Pa).legality.ok,
           "direct Program front-end is a legal check+emit client", fails);
    std::string buf; llvm::raw_string_ostream oss(buf);
    expect(os, stirIn("gemm", oss) == 0 && buf.find("check+emit client: ok") != std::string::npos,
           "stir-in gemm is a legal check+emit client", fails);
  }

  // fuzz well-formed edits; illegal ones must name an L<n>
  {
    unsigned rejected = 0, legal = 0;
    auto S0 = sched::sourceSchedule(G);
    std::vector<sched::Schedule> fuzz = {
      sched::editReplicate(S0), sched::editEngineTensorOp(sched::sourceSchedule(E)),
      sched::editGuardedZeroPage(S0)
    };
    auto bad = S0; if (!bad.stages.empty()) bad.stages[0].smemBytes = 40000; fuzz.push_back(bad);
    auto none = S0;
    if (!none.stages.empty()) {
      none.stages[0].exchange = sched::Exchange::None;
      for (auto& f : none.stages[0].factors) if (f.coord == 2) f.bind = sched::Bind::Lane;
    }
    fuzz.push_back(none);
    for (auto& S : fuzz) {
      auto L = legal::check(G, S);
      if (L.ok) legal++;
      else {
        bool named = false; for (auto& f : L.failed) if (f.n >= 1 && f.n <= 15) named = true;
        if (named) rejected++;
      }
    }
    expect(os, rejected >= 2, "fuzz: illegal schedules name an L<n>", fails);
    // smemBytes=40000 exceeds the threadgroup limit: L8 must refuse it
    bool l8 = false;
    auto L8 = legal::check(G, bad);
    for (auto& f : L8.failed) if (f.n == 8) l8 = true;
    expect(os, !L8.ok && l8, "fuzz caught L8 smem overflow (checker gap: resource bound was missing)", fails);
    (void)legal;
  }

  // sched-diff compares two legal texts
  {
    llvm::SmallString<128> pa, pb;
    std::error_code EC = llvm::sys::fs::createTemporaryFile("mvcc-sa", "sched", pa);
    std::error_code EC2 = llvm::sys::fs::createTemporaryFile("mvcc-sb", "sched", pb);
    if (!EC && !EC2) {
      auto Sa = sched::sourceSchedule(G);
      auto Sb = sched::editEngineTensorOp(Sa);
      {
        llvm::raw_fd_ostream fa(pa, EC, llvm::sys::fs::OF_Text);
        for (auto& l : sched::dump(Sa)) fa << l << "\n";
      }
      {
        llvm::raw_fd_ostream fb(pb, EC2, llvm::sys::fs::OF_Text);
        for (auto& l : sched::dump(Sb)) fb << l << "\n";
      }
      std::string buf; llvm::raw_string_ostream so(buf);
      int rc = schedDiff(pa.str().str(), pb.str().str(), so);
      llvm::sys::fs::remove(pa); llvm::sys::fs::remove(pb);
      expect(os, rc == 0 && buf.find("changed: why") != std::string::npos,
             "sched-diff reports a why change", fails);
    } else {
      expect(os, false, "sched-diff reports a why change", fails);
    }
  }

  // residual from measured − prior; improves order or is thrown away
  {
    llvm::SmallString<128> path;
    std::error_code EC = llvm::sys::fs::createTemporaryFile("mvcc-resid", "tsv", path);
    if (!EC) {
      // prior ranks engine:tensor-op cheaper than a spilled ring; measured residual flips source cheaper
      auto src = sched::sourceSchedule(G);
      auto top = sched::editEngineTensorOp(src);
      auto cr = sched::costOf(G, src), ct = sched::costOf(G, top);
      llvm::raw_fd_ostream df(path, EC, llvm::sys::fs::OF_Text);
      df << "synthetic_gemm\t" << src.why << "\tok\t" << cr.seconds << "\t" << (cr.seconds * 0.1) << "\n";
      df << "synthetic_gemm\t" << top.why << "\tok\t" << ct.seconds << "\t" << (ct.seconds * 10.0) << "\n";
      df.close();
      setenv("MVCC_SCHED_DB", path.c_str(), 1);
      Chosen C = choose(G);
      unsetenv("MVCC_SCHED_DB");
      llvm::sys::fs::remove(path);
      expect(os, C.S.why == src.why || C.S.why == top.why,
             "sched-db residual is applied or thrown away (held-out order)", fails);
    } else {
      expect(os, false, "sched-db residual is applied or thrown away (held-out order)", fails);
    }
  }

  // MVCC_SCHED_DB prefers a previously-winning why over the cost model
  {
    llvm::SmallString<128> path;
    std::error_code EC = llvm::sys::fs::createTemporaryFile("mvcc-scheddb", "tsv", path);
    if (!EC) {
      llvm::raw_fd_ostream df(path, EC, llvm::sys::fs::OF_Text);
      df << "synthetic_gemm\tsource\tlegality: ok\tcost 0\n";
      df.close();
      setenv("MVCC_SCHED_DB", path.c_str(), 1);
      Chosen C = choose(G);
      unsetenv("MVCC_SCHED_DB");
      llvm::sys::fs::remove(path);
      expect(os, C.S.why == "source" && C.dbWhy == "source",
             "sched-db prefers a previously-winning why", fails);
    } else {
      expect(os, false, "sched-db prefers a previously-winning why", fails);
    }
  }

  // illegal schedules in tests/stir/illegal/ refuse with the expected L<n>
  {
    std::string here = __FILE__;
    auto slash = here.rfind('/');
    std::string dir = (slash == std::string::npos ? std::string() : here.substr(0, slash)) + "/../../tests/stir/illegal/";
    struct Case { const char* file; const char* kind; unsigned n; };
    Case cases[] = {
      {"l2_replicate.sched", "gemm", 2},
      {"l3_exchange_none.sched", "gemm", 3},
      {"l6_drop_pred.sched", "elementwise", 6},
      {"l7_tensor_elem.sched", "elementwise", 7},
    };
    unsigned hit = 0;
    for (auto& c : cases) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> mb = llvm::MemoryBuffer::getFile(dir + c.file);
      if (!mb) continue;
      stir::Program P = std::string(c.kind) == "gemm" ? stir::direct::gemm("illegal_" + std::string(c.kind))
                                                     : stir::direct::elementwise("illegal_" + std::string(c.kind));
      sched::Schedule S; std::string err;
      if (!sched::parseText(mb.get()->getBuffer().str(), S, err)) continue;
      auto L = legal::check(P, S);
      bool named = false; for (auto& f : L.failed) if (f.n == c.n) named = true;
      if (!L.ok && named) hit++;
    }
    expect(os, hit == 4, "tests/stir/illegal/*.sched refused with the expected L<n>", fails);
  }

  // target description is a table
  expect(os, nvptxOracle().name == "nvptx-oracle" && metal4().engines.size() >= 2
             && simdgroupMatrix().name == "simdgroup-matrix",
         "target description is a table (metal4 / simdgroup-matrix=M3 / nvptx-oracle)", fails);

  if (fails) { os << "engine self-test: FAILED " << fails << "\n"; return 1; }
  os << "engine self-test: ok\n";
  return 0;
}

int stirIn(const std::string& kind, llvm::raw_ostream& os, const std::string& schedPath) {
  std::string k = kind;
  if (llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> mb = llvm::MemoryBuffer::getFile(kind)) {
    std::istringstream in(mb.get()->getBuffer().str());
    std::string tok;
    while (in >> tok) {
      if (!tok.empty() && tok[0] == '#') { std::string rest; std::getline(in, rest); continue; }
      if (tok == "kind") { in >> k; break; }
      if (tok == "elementwise" || tok == "silu" || tok == "gemm" || tok == "attention" || tok == "attn") {
        k = tok; break;
      }
    }
  }
  stir::Program P;
  if (k == "elementwise" || k == "silu") P = stir::direct::elementwise("direct_" + k);
  else if (k == "gemm") P = stir::direct::gemm("direct_gemm");
  else if (k == "attention" || k == "attn") P = stir::direct::attention("direct_attn");
  else { os << "mvcc stir-in: unknown kind '" << k << "' (elementwise|gemm|attention or a .stir file)\n"; return 2; }
  Chosen C;
  if (!schedPath.empty()) {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> mb = llvm::MemoryBuffer::getFile(schedPath);
    if (!mb) { os << "mvcc stir-in: cannot read schedule " << schedPath << "\n"; return 2; }
    std::string err;
    if (!sched::parseText(mb.get()->getBuffer().str(), C.S, err)) { os << "mvcc stir-in: " << err << "\n"; return 2; }
    C.legality = legal::check(P, C.S);
    C.cost = sched::costOf(P, C.S);
    C.emit = planEmit(P, C.S, C.legality);
    os << "stir-in " << k << " kernel=" << P.kernel << " sched-in=" << schedPath << " (no CUDA recovery)\n";
  } else {
    C = choose(P);
    os << "stir-in " << k << " kernel=" << P.kernel << " (no CUDA recovery)\n";
  }
  for (auto& l : explain(P, C)) os << l << "\n";
  for (auto& l : sched::dump(C.S)) os << l << "\n";
  os << "  " << (C.legality.ok && C.emit.canEmit ? "check+emit client: ok" : C.legality.first()) << "\n";
  return C.legality.ok ? 0 : 1;
}

int schedDiff(const std::string& pathA, const std::string& pathB, llvm::raw_ostream& os) {
  auto load = [](const std::string& path, sched::Schedule& S, std::string& err) -> bool {
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> mb = llvm::MemoryBuffer::getFile(path);
    if (!mb) { err = "cannot read " + path; return false; }
    return sched::parseText(mb.get()->getBuffer().str(), S, err);
  };
  sched::Schedule A, B; std::string ea, eb;
  if (!load(pathA, A, ea)) { os << "mvcc diff: " << ea << "\n"; return 2; }
  if (!load(pathB, B, eb)) { os << "mvcc diff: " << eb << "\n"; return 2; }
  stir::Program P = synthGEMM();
  auto La = legal::check(P, A), Lb = legal::check(P, B);
  auto Ca = sched::costOf(P, A), Cb = sched::costOf(P, B);
  os << "mvcc diff (synthetic_gemm)\n";
  os << "  a why=" << A.why << " cost " << Ca.breakdown << " " << La.first() << "\n";
  os << "  b why=" << B.why << " cost " << Cb.breakdown << " " << Lb.first() << "\n";
  if (A.why != B.why) os << "  changed: why " << A.why << " -> " << B.why << "\n";
  if (Ca.seconds != Cb.seconds)
    os << "  changed: predicted seconds " << Ca.seconds << " -> " << Cb.seconds << "\n";
  os << "  " << (La.ok && Lb.ok ? "both legal" : "at least one illegal") << "\n";
  return 0;
}

}  // namespace engine
}  // namespace mvcc
