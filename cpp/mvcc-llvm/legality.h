// Legality (L11–L15): given (Program, Schedule), discharge every obligation or refuse
// with a named obstacle. The checker is only trusted if it is seen rejecting — tests/stir/illegal and
// engine::selfTest construct the illegal points.
#pragma once
#include "schedule.h"
#include <string>
#include <vector>

namespace mvcc {
namespace legal {

struct Obligation {
  unsigned n = 0;            // 1–15
  std::string stage;
  std::string detail;
  std::string str() const {
    return "legality: L" + std::to_string(n) + (stage.empty() ? "" : " " + stage) + " " + detail;
  }
};

struct Result {
  bool ok = true;
  std::vector<Obligation> failed;
  std::vector<std::string> discharged;   // "L1 coverage"
  std::string first() const { return failed.empty() ? "legality: ok" : failed[0].str(); }
};

Result check(const stir::Program& P, const sched::Schedule& S);

// graph-level obligations (steps 11, 12, 16) over a list of programs and a fusion/device schedule
struct GraphEdge { unsigned from = 0, to = 0; std::string tensor; bool sameBlock = true; bool hostVisible = false; };
struct Graph {
  std::vector<std::string> kernels;
  std::vector<GraphEdge> edges;
  std::vector<unsigned> deviceOf;        // per kernel, 0 = unbound
  unsigned world = 1;                    // MVCC_DEVICES; L15 requires deviceOf[i] < world
};
Result checkGraph(const Graph& G);
// MVCC_GRAPH file: lines `src dst [tensor] [sameBlock=0|1] [hostVisible=0|1]`. src/dst match kernel-name prefixes.
bool loadGraphFile(const std::string& path, Graph& G, std::string& err);
// Same-block edges between consecutive kernels in G.kernels (compile-time module order).
void inferModuleSequentialEdges(Graph& G);

}  // namespace legal
}  // namespace mvcc
