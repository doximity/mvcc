// Schedule language: the ownership function Ω as a serializable object over a STIR Program.
// Well-formedness lives here; legality is legality.h. The source schedule is recovered from the physical
// facts STIR discarded (coordPhys, loop kinds, algebra permissions). Schedule edits are
// functions Schedule → Schedule. The cost model ranks; search chooses.
#pragma once
#include "stir.h"
#include <string>
#include <vector>

namespace mvcc {
namespace sched {

enum class Bind { Lane, Simdgroup, BlockX, BlockY, BlockZ, Unroll, Time, Replicate, Launch, Device };
enum class Storage { Device, Threadgroup, Register, CoopTensor, Recompute };
enum class Engine { TensorOp, ScalarFma, SimdVector };
enum class Exchange { None, ShuffleTree, ThreadgroupBuffer, Atomic, Collective };
enum class Prefetch { AtUse, IterationStart, Rotate };
enum class Predicate { Branch, SelectClamp, ZeroPage, None };

const char* bindName(Bind b);
const char* storageName(Storage s);
const char* engineName(Engine e);
const char* exchangeName(Exchange x);
const char* prefetchName(Prefetch p);
const char* predicateName(Predicate p);
bool parseBind(const std::string& s, Bind& o);
bool parseStorage(const std::string& s, Storage& o);
bool parseEngine(const std::string& s, Engine& o);
bool parseExchange(const std::string& s, Exchange& o);
bool parsePrefetch(const std::string& s, Prefetch& o);
bool parsePredicate(const std::string& s, Predicate& o);

struct Factor {
  std::string name;          // c0_lane
  unsigned coord = 0;        // logical coordinate index
  unsigned size = 0;         // 0 = *
  Bind bind = Bind::Time;
};

struct AccessStorage { unsigned tensor = 0; Storage kind = Storage::Device; unsigned ringStages = 0, ringBytes = 0; };

struct StageSchedule {
  std::vector<Factor> factors;
  std::vector<std::string> order;          // time-bound factor names, outer → inner
  std::vector<AccessStorage> storage;
  Exchange exchange = Exchange::None;
  Engine engine = Engine::ScalarFma;
  unsigned tensorM = 32, tensorN = 32, tensorK = 32;
  Prefetch prefetch = Prefetch::AtUse;
  Predicate predicate = Predicate::Branch;
  std::vector<unsigned> fusedWith;         // other stage indices sharing this nest
  unsigned threadScale = 1;
  unsigned smemBytes = 0;
};

struct Schedule {
  std::string kernel;
  std::string why;                         // "source" | edit name | "search"
  std::vector<StageSchedule> stages;
};

struct WellFormed { bool ok = true; std::string why; };

WellFormed wellFormed(const stir::Program& P, const Schedule& S);

// recover the source kernel's schedule from STIR's discarded physical facts
Schedule sourceSchedule(const stir::Program& P);

// edits
Schedule editEngineTensorOp(Schedule S);
Schedule editRestage2(Schedule S);
Schedule editSimdSplit(Schedule S, unsigned F);
Schedule editDeviceStream(Schedule S);
Schedule editGuardedZeroPage(Schedule S);
Schedule editReplicate(Schedule S);          // illegal on most programs — test fodder
Schedule editBindDevice(Schedule S);         // o → device (tensor parallel)

std::vector<std::string> dump(const Schedule& S);
bool parse(const std::vector<std::string>& lines, Schedule& S, std::string& err);
bool parseText(const std::string& text, Schedule& S, std::string& err);
// Canonical contraction shape: one compute stage, engine, k on time.
std::string stageShape(const Schedule& S);

struct Cost {
  double deviceBytes = 0;
  double smemKB = 0;
  double occupancy = 1;
  double tensorOps = 0;
  double alu = 0;
  double barriers = 0;
  double seconds = 0;                      // roofline estimate (ordering only)
  std::string breakdown;
};
Cost costOf(const stir::Program& P, const Schedule& S);

struct Candidate { Schedule S; Cost cost; };
// depth-bounded search. Source is always a candidate. First well-formed + legal is chosen
// after ranking; legality is checked by the caller (search only well-formedness + cost).
std::vector<Candidate> candidates(const stir::Program& P, unsigned depth = 3, unsigned width = 8);

}  // namespace sched
}  // namespace mvcc
