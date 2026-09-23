// Schedule language, edits, cost, search.
#include "schedule.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <set>
#include <sstream>

namespace mvcc {
namespace sched {

const char* bindName(Bind b) {
  switch (b) {
    case Bind::Lane: return "lane"; case Bind::Simdgroup: return "simdgroup";
    case Bind::BlockX: return "block.x"; case Bind::BlockY: return "block.y"; case Bind::BlockZ: return "block.z";
    case Bind::Unroll: return "unroll"; case Bind::Time: return "time";
    case Bind::Replicate: return "replicate"; case Bind::Launch: return "launch"; case Bind::Device: return "device";
  }
  return "?";
}
const char* storageName(Storage s) {
  switch (s) {
    case Storage::Device: return "device"; case Storage::Threadgroup: return "threadgroup";
    case Storage::Register: return "register"; case Storage::CoopTensor: return "coop-tensor";
    case Storage::Recompute: return "recompute";
  }
  return "?";
}
const char* engineName(Engine e) {
  switch (e) { case Engine::TensorOp: return "tensor-op"; case Engine::ScalarFma: return "scalar-fma"; case Engine::SimdVector: return "simd-vector"; }
  return "?";
}
const char* exchangeName(Exchange x) {
  switch (x) {
    case Exchange::None: return "none"; case Exchange::ShuffleTree: return "shuffle-tree";
    case Exchange::ThreadgroupBuffer: return "threadgroup-buffer"; case Exchange::Atomic: return "atomic";
    case Exchange::Collective: return "collective";
  }
  return "?";
}
const char* prefetchName(Prefetch p) {
  switch (p) { case Prefetch::AtUse: return "at-use"; case Prefetch::IterationStart: return "iteration-start"; case Prefetch::Rotate: return "rotate"; }
  return "?";
}
const char* predicateName(Predicate p) {
  switch (p) {
    case Predicate::Branch: return "branch"; case Predicate::SelectClamp: return "select+clamp";
    case Predicate::ZeroPage: return "zero-page"; case Predicate::None: return "none";
  }
  return "?";
}

static bool eq(const std::string& a, const char* b) { return a == b; }
bool parseBind(const std::string& s, Bind& o) {
  if (eq(s, "lane")) o = Bind::Lane;
  else if (eq(s, "simdgroup")) o = Bind::Simdgroup;
  else if (eq(s, "block.x")) o = Bind::BlockX;
  else if (eq(s, "block.y")) o = Bind::BlockY;
  else if (eq(s, "block.z")) o = Bind::BlockZ;
  else if (eq(s, "unroll")) o = Bind::Unroll;
  else if (eq(s, "time")) o = Bind::Time;
  else if (eq(s, "replicate")) o = Bind::Replicate;
  else if (eq(s, "launch")) o = Bind::Launch;
  else if (eq(s, "device")) o = Bind::Device;
  else return false;
  return true;
}
bool parseStorage(const std::string& s, Storage& o) {
  if (eq(s, "device")) o = Storage::Device;
  else if (eq(s, "threadgroup")) o = Storage::Threadgroup;
  else if (eq(s, "register")) o = Storage::Register;
  else if (eq(s, "coop-tensor")) o = Storage::CoopTensor;
  else if (eq(s, "recompute")) o = Storage::Recompute;
  else return false;
  return true;
}
bool parseEngine(const std::string& s, Engine& o) {
  if (eq(s, "tensor-op")) o = Engine::TensorOp;
  else if (eq(s, "scalar-fma")) o = Engine::ScalarFma;
  else if (eq(s, "simd-vector")) o = Engine::SimdVector;
  else return false;
  return true;
}
bool parseExchange(const std::string& s, Exchange& o) {
  if (eq(s, "none")) o = Exchange::None;
  else if (eq(s, "shuffle-tree")) o = Exchange::ShuffleTree;
  else if (eq(s, "threadgroup-buffer")) o = Exchange::ThreadgroupBuffer;
  else if (eq(s, "atomic")) o = Exchange::Atomic;
  else if (eq(s, "collective")) o = Exchange::Collective;
  else return false;
  return true;
}
bool parsePrefetch(const std::string& s, Prefetch& o) {
  if (eq(s, "at-use")) o = Prefetch::AtUse;
  else if (eq(s, "iteration-start")) o = Prefetch::IterationStart;
  else if (eq(s, "rotate")) o = Prefetch::Rotate;
  else return false;
  return true;
}
bool parsePredicate(const std::string& s, Predicate& o) {
  if (eq(s, "branch")) o = Predicate::Branch;
  else if (eq(s, "select+clamp")) o = Predicate::SelectClamp;
  else if (eq(s, "zero-page")) o = Predicate::ZeroPage;
  else if (eq(s, "none")) o = Predicate::None;
  else return false;
  return true;
}

WellFormed wellFormed(const stir::Program& P, const Schedule& S) {
  if (S.stages.empty()) return {false, "no stages"};
  for (size_t si = 0; si < S.stages.size(); si++) {
    const StageSchedule& st = S.stages[si];
    std::set<unsigned> covered;
    std::set<std::string> names;
    for (auto& f : st.factors) {
      if (f.coord >= P.coordNames.size() && !P.coordNames.empty())
        return {false, "stage " + std::to_string(si) + " factor " + f.name + " coord out of range"};
      if (!names.insert(f.name).second) return {false, "duplicate factor " + f.name};
      covered.insert(f.coord);
      if (f.size && (f.size & (f.size - 1)) && f.size != 3)  // 3 is allowed (planes); else power of two or *
        return {false, "factor " + f.name + " size " + std::to_string(f.size) + " is not a split the emitter can strength-reduce"};
    }
    for (unsigned j = 0; j < P.coordNames.size(); j++)
      if (!covered.count(j)) return {false, "coordinate " + P.coordNames[j] + " is not bound"};
    for (auto& n : st.order) if (!names.count(n)) return {false, "order names unknown factor " + n};
    if (st.threadScale == 0 || st.threadScale > 32) return {false, "thread_scale out of range"};
    if (st.smemBytes > 232448) return {false, "smem above the spill ceiling"};
    if (st.engine == Engine::TensorOp && (st.tensorM < 16 || st.tensorN < 8 || st.tensorK < 8))
      return {false, "tensor-op shape is below the smallest matmul2d"};
  }
  return {true, ""};
}

// ---- source schedule: invert coordPhys ("b0 + 2*b1 + 4*i0") into split/bind factors
static void addAxis(std::vector<std::pair<char, unsigned>>& axes, char kind, unsigned idx) {
  for (auto& a : axes) if (a.first == kind && a.second == idx) return;
  axes.push_back({kind, idx});
}
static std::vector<std::pair<char, unsigned>> parsePhys(const std::string& s) {
  std::vector<std::pair<char, unsigned>> axes;
  for (size_t i = 0; i < s.size(); i++) {
    char k = s[i];
    if ((k == 'b' || k == 'i' || k == 'd') && i + 1 < s.size() && s[i + 1] >= '0' && s[i + 1] <= '9') {
      unsigned v = 0; size_t j = i + 1;
      while (j < s.size() && s[j] >= '0' && s[j] <= '9') { v = v * 10 + (unsigned)(s[j] - '0'); j++; }
      addAxis(axes, k, v);
      i = j - 1;
    }
  }
  return axes;
}

static sig::Permissions programPermits(const stir::Program& P) {
  sig::Permissions p{true, true, true, true}; bool any = false;
  for (auto& L : P.loops) {
    if (!L.anyData) continue;
    if (!any) { p = L.permits; any = true; }
    else { p.partition &= L.permits.partition; p.reorder &= L.permits.reorder; p.duplicate &= L.permits.duplicate; p.treeCombine &= L.permits.treeCombine; }
  }
  return p;
}

static bool hasAlgebra(const stir::Program& P, sig::Algebra a) {
  for (auto& L : P.loops) for (auto& kv : L.algebra) if (kv.first == a) return true;
  return false;
}

Schedule sourceSchedule(const stir::Program& P) {
  Schedule S; S.kernel = P.kernel; S.why = "source";
  StageSchedule st;
  st.engine = (hasAlgebra(P, sig::Algebra::TensorSum) ? Engine::TensorOp : Engine::ScalarFma);
  st.prefetch = Prefetch::IterationStart;
  st.predicate = P.guards.empty() ? Predicate::None : Predicate::Branch;
  sig::Permissions perm = programPermits(P);
  bool anyR = false;
  for (bool o : P.coordOutput) if (!o) anyR = true;
  if (anyR && perm.partition) st.exchange = Exchange::ShuffleTree;
  for (unsigned j = 0; j < P.coordNames.size(); j++) {
    auto axes = j < P.coordPhys.size() ? parsePhys(P.coordPhys[j]) : std::vector<std::pair<char, unsigned>>{};
    unsigned tidLo = 0, tidHi = 0; bool hasTid = false, hasIter = false, hasDer = false;
    for (auto& a : axes) {
      if (a.first == 'b') { hasTid = true; if (a.second < 5) tidLo++; else tidHi++; }
      else if (a.first == 'i') hasIter = true;
      else if (a.first == 'd') hasDer = true;
    }
    if (hasTid && tidLo) {
      Factor f; f.coord = j; f.name = P.coordNames[j] + "_lane"; f.size = 1u << std::min(tidLo, 5u); f.bind = Bind::Lane;
      st.factors.push_back(f);
    }
    if (hasTid && tidHi) {
      Factor f; f.coord = j; f.name = P.coordNames[j] + "_sg"; f.size = 1u << std::min(tidHi, 5u); f.bind = Bind::Simdgroup;
      st.factors.push_back(f);
    }
    if (hasIter) {
      Factor f; f.coord = j; f.name = P.coordNames[j] + "_k"; f.size = 0; f.bind = Bind::Time;
      st.factors.push_back(f);
      st.order.push_back(f.name);
    }
    if (hasDer) {
      Factor f; f.coord = j; f.name = P.coordNames[j] + "_u"; f.size = 0; f.bind = Bind::Unroll;
      st.factors.push_back(f);
    }
    if (st.factors.empty() || st.factors.back().coord != j) {
      // no physical axis (constant / replication): still bind so well-formedness holds
      Factor f; f.coord = j; f.name = P.coordNames[j] + (P.coordOutput.empty() || P.coordOutput[j] ? "_blk" : "_r");
      f.size = 1; f.bind = (P.coordOutput.empty() || P.coordOutput[j]) ? Bind::BlockX : Bind::Time;
      st.factors.push_back(f);
      if (f.bind == Bind::Time) st.order.push_back(f.name);
    }
  }
  if (P.coordNames.empty()) {
    Factor f; f.name = "c0_lane"; f.size = 32; f.bind = Bind::Lane;
    st.factors.push_back(f);
  }
  for (unsigned t = 0; t < P.tensors.size(); t++) {
    AccessStorage a; a.tensor = t; a.kind = Storage::Device;
    st.storage.push_back(a);
  }
  unsigned nOut = 0; for (auto& T : P.tensors) if (T.written) nOut++;
  // a contraction with a traversal loop is one compute stage (+ implicit epilogue); elementwise is one stage
  S.stages.push_back(st);
  if (nOut && anyR && !P.accesses.empty()) {
    StageSchedule epi = st;
    epi.engine = Engine::ScalarFma;
    epi.exchange = Exchange::None;
    epi.order.clear();
    // the epilogue is r = (): reduction coordinates are not partitioned, they have already been combined
    for (auto& f : epi.factors) {
      bool red = f.coord < P.coordOutput.size() && !P.coordOutput[f.coord];
      if (red) f.bind = Bind::Time;
    }
  }
  return S;
}

Schedule editEngineTensorOp(Schedule S) {
  S.why = "engine:tensor-op";
  for (auto& st : S.stages) {
    st.engine = Engine::TensorOp; st.tensorM = st.tensorN = st.tensorK = 32;
    for (auto& a : st.storage) if (a.kind == Storage::Register) a.kind = Storage::CoopTensor;
  }
  return S;
}
Schedule editRestage2(Schedule S) {
  S.why = "restage:2";
  for (auto& st : S.stages) {
    for (auto& a : st.storage) if (a.kind == Storage::Threadgroup) { a.ringStages = 2; }
    st.smemBytes = st.smemBytes ? std::min(st.smemBytes, 32768u) : 16384;
  }
  return S;
}
Schedule editSimdSplit(Schedule S, unsigned F) {
  S.why = "split:simdgroup*" + std::to_string(F);
  for (auto& st : S.stages) {
    st.threadScale = F;
    bool rebound = false;
    for (auto& f : st.factors) if (f.bind == Bind::Unroll) { f.bind = Bind::Simdgroup; rebound = true; }
    if (!rebound) for (auto& f : st.factors) if (f.bind == Bind::Time && f.size != 1) { f.bind = Bind::Simdgroup; break; }
  }
  return S;
}
Schedule editDeviceStream(Schedule S) {
  S.why = "storage:device-stream";
  for (auto& st : S.stages) {
    for (auto& a : st.storage) if (a.kind == Storage::Threadgroup) a.kind = Storage::Device;
    st.prefetch = Prefetch::IterationStart;
    st.smemBytes = 0;
  }
  return S;
}
Schedule editGuardedZeroPage(Schedule S) {
  S.why = "predicate:zero-page";
  for (auto& st : S.stages) st.predicate = Predicate::ZeroPage;
  return S;
}
Schedule editReplicate(Schedule S) {
  S.why = "bind:replicate";
  for (auto& st : S.stages) for (auto& f : st.factors) if (f.bind == Bind::Lane) { f.bind = Bind::Replicate; break; }
  return S;
}
Schedule editBindDevice(Schedule S) {
  S.why = "bind:device";
  for (auto& st : S.stages) {
    for (auto& f : st.factors)
      if (f.bind == Bind::BlockX) { f.bind = Bind::Device; break; }
    st.exchange = Exchange::Collective; // partitioned o across ranks is AllGather (copy)
  }
  return S;
}

std::string stageShape(const Schedule& S) {
  if (S.stages.empty()) return "stage-shape none";
  const StageSchedule& st = S.stages[0];
  bool kTime = false;
  for (auto& f : st.factors) if (f.bind == Bind::Time) kTime = true;
  return "stage-shape contraction stages=" + std::to_string(S.stages.size())
         + " engine=" + std::string(engineName(st.engine))
         + (kTime ? " k=time" : "");
}

std::vector<std::string> dump(const Schedule& S) {
  std::vector<std::string> out;
  out.push_back("schedule " + (S.kernel.empty() ? "_" : S.kernel) + " why=" + S.why + " stages=" + std::to_string(S.stages.size()));
  for (size_t i = 0; i < S.stages.size(); i++) {
    const StageSchedule& st = S.stages[i];
    out.push_back("  stage " + std::to_string(i) + " engine=" + std::string(engineName(st.engine))
                  + " exchange=" + exchangeName(st.exchange) + " prefetch=" + prefetchName(st.prefetch)
                  + " predicate=" + predicateName(st.predicate) + " thread_scale=" + std::to_string(st.threadScale)
                  + " smem=" + std::to_string(st.smemBytes));
    for (auto& f : st.factors)
      out.push_back("    split " + f.name + " coord=" + std::to_string(f.coord) + " size=" + (f.size ? std::to_string(f.size) : "*")
                    + " bind=" + std::string(bindName(f.bind)));
    if (!st.order.empty()) {
      std::string s = "    order";
      for (auto& n : st.order) s += " " + n;
      out.push_back(s);
    }
    for (auto& a : st.storage)
      out.push_back("    storage t" + std::to_string(a.tensor) + " " + std::string(storageName(a.kind))
                    + (a.ringStages ? " ring=" + std::to_string(a.ringStages) : ""));
  }
  return out;
}

static std::string trim(const std::string& s) {
  size_t a = 0, b = s.size();
  while (a < b && (s[a] == ' ' || s[a] == '\t')) a++;
  while (b > a && (s[b - 1] == ' ' || s[b - 1] == '\t' || s[b - 1] == '\r')) b--;
  return s.substr(a, b - a);
}
static bool starts(const std::string& s, const char* p) { return s.rfind(p, 0) == 0; }
static std::string after(const std::string& s, const char* p) { return trim(s.substr(std::string(p).size())); }

bool parse(const std::vector<std::string>& lines, Schedule& S, std::string& err) {
  S = Schedule{};
  StageSchedule* st = nullptr;
  for (auto raw : lines) {
    std::string l = trim(raw);
    if (l.empty() || l[0] == '#') continue;
    if (starts(l, "schedule ")) {
      S.kernel = after(l, "schedule ");
      size_t sp = S.kernel.find(' ');
      if (sp != std::string::npos) {
        std::string rest = S.kernel.substr(sp + 1); S.kernel = S.kernel.substr(0, sp);
        if (starts(rest, "why=")) {
          rest = rest.substr(4);
          size_t e = rest.find(' ');
          S.why = e == std::string::npos ? rest : rest.substr(0, e);
        }
      }
      continue;
    }
    if (starts(l, "stage ")) {
      S.stages.push_back({});
      st = &S.stages.back();
      std::istringstream is(after(l, "stage "));
      std::string tok;
      while (is >> tok) {
        if (starts(tok, "engine=")) parseEngine(tok.substr(7), st->engine);
        else if (starts(tok, "exchange=")) parseExchange(tok.substr(9), st->exchange);
        else if (starts(tok, "prefetch=")) parsePrefetch(tok.substr(9), st->prefetch);
        else if (starts(tok, "predicate=")) parsePredicate(tok.substr(10), st->predicate);
        else if (starts(tok, "thread_scale=")) st->threadScale = (unsigned)std::stoul(tok.substr(13));
        else if (starts(tok, "smem=")) st->smemBytes = (unsigned)std::stoul(tok.substr(5));
      }
      continue;
    }
    if (!st) { err = "factor before stage"; return false; }
    if (starts(l, "split ")) {
      Factor f;
      std::istringstream is(after(l, "split "));
      is >> f.name;
      std::string tok;
      while (is >> tok) {
        if (starts(tok, "coord=")) f.coord = (unsigned)std::stoul(tok.substr(6));
        else if (starts(tok, "size=")) { std::string v = tok.substr(5); f.size = (v == "*") ? 0 : (unsigned)std::stoul(v); }
        else if (starts(tok, "bind=")) { if (!parseBind(tok.substr(5), f.bind)) { err = "bad bind " + tok; return false; } }
      }
      st->factors.push_back(f);
      continue;
    }
    if (starts(l, "order")) {
      std::istringstream is(after(l, "order"));
      std::string n; while (is >> n) st->order.push_back(n);
      continue;
    }
    if (starts(l, "storage ")) {
      AccessStorage a;
      std::istringstream is(after(l, "storage "));
      std::string t, k; is >> t >> k;
      if (!t.empty() && t[0] == 't') a.tensor = (unsigned)std::stoul(t.substr(1));
      if (!parseStorage(k, a.kind)) { err = "bad storage " + k; return false; }
      std::string extra;
      while (is >> extra) {
        if (starts(extra, "ring=")) a.ringStages = (unsigned)std::stoul(extra.substr(5));
        else if (starts(extra, "ring_bytes=")) a.ringBytes = (unsigned)std::stoul(extra.substr(11));
      }
      st->storage.push_back(a);
      continue;
    }
    err = "unrecognized schedule line: " + l;
    return false;
  }
  if (S.stages.empty()) { err = "empty schedule"; return false; }
  return true;
}
bool parseText(const std::string& text, Schedule& S, std::string& err) {
  std::vector<std::string> lines; std::string cur;
  for (char c : text) { if (c == '\n') { lines.push_back(cur); cur.clear(); } else cur += c; }
  if (!cur.empty()) lines.push_back(cur);
  return parse(lines, S, err);
}

Cost costOf(const stir::Program& P, const Schedule& S) {
  Cost c;
  // device bytes: every device-storage access, once per output point we can see (samples as a proxy)
  double accSamples = 0;
  for (auto& a : P.accesses) accSamples += a.samples ? a.samples : 1;
  if (accSamples == 0) accSamples = 256;
  bool anyDevice = false, anySmem = false;
  unsigned smem = 0;
  for (auto& st : S.stages) {
    smem = std::max(smem, st.smemBytes);
    for (auto& a : st.storage) {
      if (a.kind == Storage::Device) anyDevice = true;
      if (a.kind == Storage::Threadgroup) { anySmem = true; smem = std::max(smem, a.ringBytes ? a.ringBytes : 8192u); }
    }
    if (st.engine == Engine::TensorOp) c.tensorOps += 1;
    if (st.engine == Engine::ScalarFma) c.alu += accSamples;
    if (st.exchange == Exchange::ThreadgroupBuffer) { smem = std::max(smem, 8192u); c.barriers += 2; }
    if (st.exchange == Exchange::ShuffleTree) c.alu += 32;
    if (anySmem || st.smemBytes) c.barriers += 1;
  }
  c.deviceBytes = anyDevice ? accSamples * 16 : 0;   // 16 B as a typical vector load
  c.smemKB = smem / 1024.0;
  // occupancy law from the probe: TFLOP/s ≈ 1 / KB  (16 KB → 25, 24 KB → 17)
  c.occupancy = c.smemKB <= 0.5 ? 1.0 : std::min(1.0, 16.0 / std::max(c.smemKB, 1.0));
  const double bw = 125e9, peak = 18.5e12;
  double memT = c.deviceBytes / bw;
  double aluT = (c.tensorOps * 32.0 * 32.0 * 32.0 * 2.0) / (peak * std::max(c.occupancy, 0.05));
  double scalarT = c.alu / 1e10;
  c.seconds = std::max(memT, std::max(aluT, scalarT)) + c.barriers * 2e-7;
  // streaming from device (no ring) is cheaper on occupancy than a spilled ring — the decode lesson
  if (anySmem && smem > 32768) c.seconds *= 8;       // spill
  std::ostringstream os;
  os << "bytes=" << (unsigned)c.deviceBytes << " smemKB=" << c.smemKB << " occ=" << c.occupancy
     << " tensorOps=" << c.tensorOps << " alu=" << c.alu << " barriers=" << c.barriers
     << " est=" << c.seconds;
  c.breakdown = os.str();
  return c;
}

static std::string canon(const Schedule& S) {
  std::string t;
  for (auto& l : dump(S)) t += l + "\n";
  return t;
}

std::vector<Candidate> candidates(const stir::Program& P, unsigned depth, unsigned width) {
  std::vector<Candidate> out;
  std::set<std::string> seen;
  auto consider = [&](Schedule S) {
    auto wf = wellFormed(P, S);
    if (!wf.ok) return;
    std::string k = canon(S);
    if (!seen.insert(k).second) return;
    out.push_back({S, costOf(P, S)});
  };
  Schedule src = sourceSchedule(P);
  consider(src);
  std::vector<Schedule> frontier = {src};
  for (unsigned d = 0; d < depth; d++) {
    std::vector<Schedule> next;
    for (auto& S : frontier) {
      Schedule edits[] = {
        editEngineTensorOp(S), editRestage2(S), editSimdSplit(S, 2), editSimdSplit(S, 4),
        editDeviceStream(S), editGuardedZeroPage(S)
      };
      for (auto& E : edits) { consider(E); next.push_back(E); }
    }
    frontier.swap(next);
  }
  std::sort(out.begin(), out.end(), [](const Candidate& a, const Candidate& b) { return a.cost.seconds < b.cost.seconds; });
  if (out.size() > width) out.resize(width);
  // source is always retained even if it ranked out of the window
  bool hasSrc = false;
  for (auto& c : out) if (c.S.why == "source") hasSrc = true;
  if (!hasSrc) {
    auto wf = wellFormed(P, src);
    if (wf.ok) {
      if (out.size() == width) out.pop_back();
      out.push_back({src, costOf(P, src)});
    }
  }
  return out;
}

}  // namespace sched
}  // namespace mvcc
