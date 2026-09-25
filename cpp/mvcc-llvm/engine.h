// STIR engine: recover a source schedule, check legality, rank candidates, emit a plan.
#pragma once
#include "legality.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace mvcc {
namespace engine {

struct Target {
  std::string name;
  std::vector<std::string> engines;
  unsigned smemBytes = 32768;
  double bwGBps = 125;
  double tflops = 18.5;
};
Target metal4();
Target simdgroupMatrix();      // M3 table row
Target scalarMSL();
Target nvptxOracle();          // CUDA → STIR → PTX: matmul2d is mma.sync, C stays fragment layout (acc2ptx = id)

enum class Precision { Inherit, FpReassoc32, FpReassocStar, ExplicitBudget };
const char* precisionName(Precision p);

struct EmitPlan {
  bool canEmit = false;
  std::string why;             // "identity: exact twin" | "check+emit" | obstacle
  std::vector<std::string> steps;
};
EmitPlan planEmit(const stir::Program& P, const sched::Schedule& S, const legal::Result& L);

struct Chosen {
  sched::Schedule S;
  sched::Cost cost;
  legal::Result legality;
  EmitPlan emit;
  std::string dbWhy;           // last MVCC_SCHED_DB winner for this kernel; empty = unused
};
// pick the cheapest legal candidate; source wins ties. A MVCC_SCHED_DB hit for this kernel
// outranks the cost model (measured winner).
Chosen choose(const stir::Program& P);

std::vector<std::string> explain(const stir::Program& P, const Chosen& C);

// append schedule / legality / search / explain lines (gated by MVCC_SCHED)
void analyze(const stir::Program& P, std::vector<std::string>& out);

// --engine-self-test: synthetic programs, illegal schedules, graph L11/L12, parse round-trip
int selfTest(llvm::raw_ostream& os);

// compare two schedule texts against a synthetic contraction (mvcc diff)
int schedDiff(const std::string& pathA, const std::string& pathB, llvm::raw_ostream& os);
// a Program built without CUDA recovery (elementwise|gemm|attention)
int stirIn(const std::string& kind, llvm::raw_ostream& os, const std::string& schedPath = {});

}  // namespace engine
}  // namespace mvcc
