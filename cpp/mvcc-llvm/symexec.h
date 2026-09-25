// Symbolic execution of a kernel for one thread.
//
// The executor runs the kernel's IR for a concrete thread id with every kernel argument and block index symbolic,
// skipping one loop (the recovered K loop) whose effects are abstracted: values it defines become opaque per-thread
// symbols. Integer arithmetic is kept as polynomials over atoms (arguments, block indices, opaque values) so that
// addresses and predicates come out as closed forms; control flow is executed statically with reaching conditions
// (a value defined on several paths is an if-then-else over the branch conditions), loops are unrolled until their
// back-edge condition is constantly false (a data-dependent trip count leaves the iterations' effects conditional),
// and memory is modeled byte by byte: shared-memory bytes carry the writing thread, barrier phase and condition,
// device-memory effects are recorded as (address, bytes, condition) triples. Every device-memory effect of the
// original kernel is thereby a function of the thread, the block, the arguments and the per-thread opaque values.
//
// Assumptions the polynomial model makes, stated once: integer arithmetic on addresses and indices does not wrap
// (signed overflow is undefined in the CUDA source; unsigned quantities derived from thread ids and tile constants
// are concrete and evaluated exactly), and a load reads what the last barrier-ordered write to the same bytes put
// there (the CUDA program's own memory model within a block).
#pragma once
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Operator.h"
#include <functional>
#include <map>
#include <tuple>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mvcc {
namespace sym {

using namespace llvm;

struct Expr;
using EP = const Expr*;

enum class EK : uint8_t { Const, Poly, Sym, Op, Cmp, Ite, Load, Concat, Bytes, Call, Undef };

// Atom: what a polynomial ranges over - a symbol or an opaque expression (both by id).
struct Expr {
  EK k;
  Type* ty = nullptr;
  APInt c;                                                        // Const: integer / float bits
  std::vector<std::pair<std::vector<uint32_t>, int64_t>> terms;  // Poly: sorted monomials (sorted atom ids) -> coef; {} = 1
  uint32_t sym = 0;                                               // Sym: symbol id
  unsigned op = 0;                                                // Op: Instruction opcode; Cmp: CmpInst predicate
  std::vector<EP> args;                                           // Op / Cmp(a,b) / Ite(c,a,b) / Load(addr, guard) / Concat / Bytes(x) / Call
  unsigned lo = 0, len = 0;                                       // Bytes: [lo, lo+len) bytes of args[0]; Load: len = alignment
  Function* callee = nullptr;                                     // Call
  Value* asmv = nullptr;                                          // Call: inline asm callee (per-thread ALU asm), callee null
  uint32_t id = 0;
};

struct Symbol {
  enum Kind { Arg, Sreg, ThreadValue, Unknown, SharedBase, Global, SmemByte, LoopValue, Coord, Private, Guard } kind = Arg;
  // Guard: a boolean standing for a disjunction of path conditions (Arena::factor); field = its index. Conditions
  // are kept in disjunctive normal form, whose size multiplies across independent branches; a reaching condition
  // that is a disjunction is replaced by a guard so that the code below the join runs under one clause
  // Private: the base of a per-thread stack object (alloca; retile mode). Its contents are not modeled: stores to it are
  // dropped and loads from it are opaque per-thread values (counted in Executor::privateAccesses).
  // LoopValue: a header phi of a loop executed generically (retile mode); uniform across threads. v = the phi
  // Coord: a coordinate of a re-emitted tensor program, named; bound to a value at emission
  Value* v = nullptr;   // Arg: Argument; ThreadValue: the defining instruction; SharedBase/Global/SmemByte: the GlobalVariable
  std::string name;     // Sreg: intrinsic name
  unsigned thread = 0;  // ThreadValue / Unknown
  unsigned field = 0;   // ThreadValue: aggregate field (0 for scalars); SmemByte: index of the read record
  Type* ty = nullptr;
};

// the deepest guard expansion since the last call (MVCC_TENSOR_PROGRESS reports it per kernel)
unsigned takeGuardExpandMax();
class Arena {
 public:
  Arena(const DataLayout& DL, LLVMContext& C) : DL(DL), C(C) {}
  const DataLayout& DL;
  LLVMContext& C;

  EP mkInt(Type* ty, int64_t v);
  EP mkConstFP(Type* ty, const APInt& bits);
  EP mkBool(bool b);
  EP mkUndef(Type* ty);
  EP mkPoly(Type* ty, std::vector<std::pair<std::vector<uint32_t>, int64_t>> terms);
  EP mkSym(const Symbol& s);
  EP mkSymRaw(const Symbol& s);                  // the Sym node itself (mkSym wraps integer symbols in a polynomial)
  EP rebuild(EP e, std::vector<EP> args);        // same node kind and payload, other operands (no simplification)
  uint32_t atomOf(EP e);         // an opaque integer expression as a polynomial atom
  EP asPoly(EP e);               // integer-typed expression -> polynomial (opaque ones become atoms)
  EP retype(EP poly, Type* ty);  // same polynomial, another integer/pointer type (constants wrap)

  EP add(EP a, EP b); EP sub(EP a, EP b); EP mul(EP a, EP b); EP neg(EP a);
  EP binop(unsigned opc, EP a, EP b, Type* ty);   // integer or float BinaryOperator
  EP unop(unsigned opc, EP a, Type* ty);
  EP cast(unsigned opc, EP a, Type* ty);
  EP cmp(unsigned pred, EP a, EP b);
  EP ite(EP c, EP a, EP b);
  EP andb(EP a, EP b); EP orb(EP a, EP b); EP notb(EP a);
  EP load(Type* ty, EP addr, EP guard, unsigned align);
  EP call(Function* callee, std::vector<EP> args, Type* ty);
  EP callAsm(Value* asmv, std::vector<EP> args, Type* ty);
  EP concat(std::vector<EP> parts, Type* ty);
  EP bytes(EP x, unsigned lo, unsigned len);

  bool constInt(EP e, int64_t& out) const;
  bool isTrue(EP e) const; bool isFalse(EP e) const;
  bool isPoly(EP e) const { return e->k == EK::Poly; }
  unsigned sizeOf(Type* ty) const { return (unsigned)DL.getTypeStoreSize(ty); }
  const Symbol& symbol(uint32_t id) const { return symbols[id]; }
  bool isSymAtom(uint32_t atom, uint32_t& symId) const;   // atom -> symbol id if the atom is a symbol
  EP atomExpr(uint32_t atom) const { return atoms[atom]; }

  std::vector<EP> toBytes(EP v);
  EP fromBytes(const std::vector<EP>& bytes, Type* ty);

  // literals of a conjunction
  void conjuncts(EP c, std::vector<EP>& out) const;
  bool implies(EP a, EP b) const;    // a => b, syntactically (b's literals are among a's)
  bool litImplies(EP x, EP y) const;
  bool contradictory(const std::vector<EP>& lits, bool clamps = false) const;   // no integer satisfies the literals on one polynomial
  EP given(EP c, EP ctx);                                  // c where ctx holds: implied literals dropped, guards expanded
  // whether the facts (literals that hold) decide literal l by the bounds they put on its polynomial (the same
  // polynomial up to the constant term, compared against the same right-hand side): out = its value
  bool decideLit(EP l, const std::vector<EP>& facts, bool& out) const;
  bool disjoint(EP a, EP b) const;   // a and b cannot both hold, syntactically (a literal and its negation)
  bool containsSymbol(EP e, Symbol::Kind kind, std::set<uint32_t>* found = nullptr);
  // (the memo of a rewrite over a large DAG - the decode retiler's, millions of nodes - is a hash map)
  using RewriteMemo = std::unordered_map<EP, EP>;
  EP subst(EP e, const std::map<uint32_t, EP>& m, std::map<EP, EP>& memo);   // symbols -> expressions
  EP subst(EP e, const std::map<uint32_t, EP>& m, RewriteMemo& memo);
  // rebuilds e bottom-up with the simplifying constructors; `leaf` may replace any node (nullptr = keep)
  EP rewrite(EP e, const std::function<EP(EP)>& leaf, std::map<EP, EP>& memo);
  EP rewrite(EP e, const std::function<EP(EP)>& leaf, RewriteMemo& memo);
  // canonical decision-tree form: every condition literal of e is lifted to the top in a fixed order (Shannon
  // expansion), so that semantically equal expressions built along different control-flow paths coincide
  EP shannon(EP e, const std::function<std::string(EP)>& litKey, unsigned maxLits = 10);
  // ites whose condition `fold` accepts are decided by the enclosing ite conditions where those decide it (the
  // literals of an enclosing condition, and their negations, are replaced by their values); other ites are kept
  // as written so that expressions built for different lanes keep one shape
  // The result of a call depends on the expression, the enclosing facts that can decide a selection in it, and
  // `fold`: a cache (one per `fold` predicate, owned by the caller) shares the work across calls - the elements of
  // one tile share almost all of their expressions' structure
  struct SimplifyUnderCache { std::unordered_map<std::string, EP> results; std::vector<int8_t> fold; /* fold(c) by node id: -1 unknown */ };
  EP simplifyUnder(EP e, const std::function<bool(EP)>& fold, SimplifyUnderCache* cache = nullptr);
  // The facts (literals, or bounds on a comparison's polynomial) that can decide a selection anywhere in e: the
  // identity of every boolean node of a selection's condition (key 2*id) and the class - left-hand side without its
  // constant term, and type - of every comparison there (key 2*class+1), sorted. Empty: e has no selection, and
  // simplifyUnder leaves it as it is. conditionKeys: the same for the nodes of one condition.
  ArrayRef<uint64_t> selectionKeys(EP e);
  ArrayRef<uint64_t> conditionKeys(EP c);
  uint64_t polyClassKey(EP cmp);   // the class key of a comparison's left-hand side, 0 when it is not a polynomial
  EP unwrap(EP e) const;   // 1*atom of the atom's type -> the atom
  std::string str(EP e, unsigned depth = 6) const;   // for diagnostics
  size_t size() const { return nodes.size(); }
  // a disjunction of clauses -> (the literals common to every clause) and (a guard symbol standing for the rest);
  // other conditions are returned as they are
  EP factor(EP c);
  EP guardDef(uint32_t symId) const { return guardDefs[symbols[symId].field]; }
  EP expandGuard(const std::vector<EP>& lits) const;   // the conjunction with its first guard literal replaced by its definition (null: none)
  // semantic (bound-aware) simplification of path conditions: a clause subsumed by a weaker bound is dropped when
  // disjunctions merge, and a bound implies the (in)equalities it decides. The mainloop re-emission needs it to keep
  // an unrolled body's conditions small; the decode retiler compares operand expressions shape by shape and keeps
  // the syntactic forms (off by default)
  bool semantic = false;

 private:
  EP intern(Expr&& e);
  // the hash-consing table: nodes hashed and compared on the fields that identify them (no key string is built)
  struct ExprHash { size_t operator()(EP e) const; };
  struct ExprEq { bool operator()(EP a, EP b) const; };
  std::unordered_set<EP, ExprHash, ExprEq> table;
  std::vector<std::unique_ptr<Expr>> nodes;
  std::vector<Symbol> symbols;
  std::map<std::string, uint32_t> symIds;
  std::vector<EP> atoms;                       // atom id -> expression (Sym or opaque)
  std::unordered_map<EP, uint32_t> atomIds;
  std::unordered_map<EP, std::vector<EP>> byteMemo;
  std::map<std::pair<EP, unsigned>, bool> containsMemo;
  std::vector<EP> guardDefs;                   // guard index -> the disjunction it stands for
  std::unordered_map<EP, EP> guardOf;          // disjunction -> its guard symbol
  mutable std::map<std::pair<EP, EP>, bool> impliesMemo, disjointMemo;   // expressions are immutable: the answers are too
  unsigned orbDepth = 0;                       // re-simplification rounds of orb in progress
  std::map<std::pair<EP, EP>, EP> andMemo, orMemo;   // andb/orb are functions of their operands (and of `semantic`, set once)
  // bytes() recurses into both arms of an ite; an ite DAG whose arms share subexpressions (the chains resolve()
  // builds under `given`, the uninitialized-read zeros) would be walked as a tree without this
  std::map<std::tuple<EP, unsigned, unsigned>, EP> bytesMemo;
  // selectionKeys / conditionKeys / polyClassKey per expression, by node id (immutable, so the answers are too;
  // the vectors' element 0 marks an answer known); the class ids of polynomials
  std::vector<std::vector<uint64_t>> selKeysMemo, condKeysMemo;
  std::vector<uint64_t> polyClassMemo;
  std::vector<EP> notbMemo;                    // notb by node id (a function of its operand)
  template <class Memo> EP rewriteT(EP e, const std::function<EP(EP)>& leaf, Memo& memo);
  std::map<std::pair<std::vector<std::pair<std::vector<uint32_t>, int64_t>>, Type*>, uint32_t> polyClassIds;
  EP bytesRaw(EP x, unsigned lo, unsigned len);
  std::map<std::pair<EP, EP>, EP> givenMemo;
  EP givenRaw(EP c, EP ctx);
  EP andbRaw(EP a, EP b);
  EP orbRaw(EP a, EP b);
  bool impliesRaw(EP a, EP b) const;
  bool disjointRaw(EP a, EP b) const;
  // implies/disjoint expand a guard among the literals into its definition's clauses and ask again; every clause of
  // a join's guard holds the previous join's guard, so the chain is exponential in the number of joins. Past this
  // many nested expansions the question is left unproven (the caller then does not simplify), which is sound; the
  // answer is memoized like any other, so it is deterministic and asked once.
  static constexpr unsigned kGuardExpandLimit = 32;
  mutable unsigned guardExpandDepth = 0, guardExpandMax = 0;   // current nesting; the deepest reached (diagnostics)
  struct GuardExpandScope { const Arena& A; explicit GuardExpandScope(const Arena& a) : A(a) { A.guardExpandDepth++; A.guardExpandMax = std::max(A.guardExpandMax, A.guardExpandDepth); noteGuardExpandDepth(A.guardExpandDepth); } ~GuardExpandScope() { A.guardExpandDepth--; } };
  static void noteGuardExpandDepth(unsigned d);
  EP polyConst(Type* ty, int64_t v) { return mkPoly(ty, {{{}, v}}); }
  int64_t wrap(Type* ty, int64_t v) const;
};

// A loop-carried value of a traversal loop, as the executor saw it for one thread: `init` is the value the header
// phi takes on entry, `next` the back-edge value with the phi itself a free ThreadValue symbol (v = the phi) and
// every other header phi of the same loop likewise. The raw material of the recurrence-algebra classification
// (sig.h): whether the phi is loop control, an associative accumulator, or order-sensitive state.
struct Recurrence { const Loop* L; PHINode* phi; EP init; EP next; unsigned thread; };

struct SmemWrite { int64_t off; EP byte; EP cond; unsigned phase, thread, seq; Instruction* site; };
struct SmemRead { int64_t off; EP cond; unsigned phase, thread, seq; Instruction* site; GlobalVariable* obj; uint32_t sym; };
struct DevStore { EP addr; std::vector<EP> bytes; EP cond; unsigned phase, thread; Instruction* site; bool atomic = false; };
// A device-memory read as one thread issued it (retile mode, recordDevLoads): the physical access set coordinate
// discovery (coords.h) fits. `iters` is the iteration index of every enclosing traversal loop, outermost first.
struct DevLoad { EP addr; unsigned bytes; EP cond; unsigned phase, thread; std::vector<unsigned> iters; Instruction* site; std::vector<const Loop*> loops; };

class Executor {
 public:
  Executor(Function& F, DominatorTree& DT, LoopInfo& LI, Arena& A, unsigned T, const Loop* skip)
      : F(F), DT(DT), LI(LI), A(A), T(T), skip(skip) {}
  // Executes the kernel for thread `tid`. Effects accumulate across calls (all threads of the block).
  bool run(unsigned tid, std::string& why);

  // After every thread ran: resolves each shared-memory byte read to the write that produced it (the permutation
  // proof) and substitutes the results into the device stores. False with the reason if any read is not covered.
  bool resolve(std::string& why);

  std::vector<SmemWrite> smemWrites;
  std::vector<SmemRead> smemReads;
  std::vector<DevStore> devStores;
  std::map<int64_t, std::vector<size_t>> smemByOff;   // byte offset -> indices into smemWrites
  // the epilogue: the barrier that opens the C-tile shared permutation (the last barrier before the
  // first shared store after the skipped loop), and the phase that starts there. Stream-K wait
  // barriers come first on some paths and are not Be.
  Instruction* Be = nullptr;
  EP beCond = nullptr;  // reaching condition at Be; later store conditions are given(R, beCond)
  unsigned epiPhase = 0;
  unsigned finalPhase = 0;
  unsigned firstStorePhase = ~0u;   // earliest phase with a device store (loads after it are refused)
  std::vector<Instruction*> acc2ptxSites;  // __mvcc_tp_* aggregate calls whose fields are per-thread symbols
  unsigned maxUnroll = 512;

  // Retile mode: the whole kernel is modeled from its entry - shared memory from the start,
  // staging copies as shared writes of device loads, per-thread ALU asm as opaque functions of its operands, device
  // stores and atomics as no effect - so the operands of every tensor operation resolve to device memory. Loops in
  // `generic` run once with their header phis as uniform LoopValue symbols and their exits dropped; `iterL` runs
  // `iterCount` times with its exit dropped (the body under the assumption that the iteration executes).
  bool retile = false;
  std::map<const Loop*, bool> generic;  // loop -> header phis are uniform (LoopValue) or per-thread (ThreadValue)
  std::map<const Loop*, EP> genericEntry;   // generic loop -> the condition it was entered under (its exits' condition)
  std::map<const Loop*, std::pair<unsigned, unsigned>> genericPhases;   // generic loop -> the barrier phase at its entry and after its one pass
  // Generic loops whose results are opaque: a block-uniform search (a scan of a device table for the job's expert,
  // decided by a ballot) that the caller proved every thread computes alike. What leaves the loop - the exit-block
  // phis and its live-outs - is one uniform LoopValue symbol per value, standing for the IR value itself (which
  // dominates everything after the loop), instead of a selection over the loop's symbolic exit conditions.
  std::set<const Loop*> opaqueExits;
  std::set<Value*> opaqueValues;   // the values so bound (a merge at their block keeps the symbol)
  // Header phis of a uniform generic loop that carry a per-thread value regardless (an fp32 accumulator the K
  // groups add into): a ThreadValue symbol, so that an operand depending on one is refused, not misread as uniform.
  std::set<PHINode*> threadPhis;
  // Shared bytes written by a plain store of a value no single lane can recompute from device memory (a quantizer's
  // per-row scale is a shuffle reduction) resolve to a load of the byte from shared memory rather than to the stored
  // expression: the operand keeps reading the staged bytes, and only the other operands are streamed.
  bool laneStoresStay = false;
  std::set<Instruction*> stayingReads;   // the read sites resolve() so kept on shared memory
  // Shared bytes no write covers under part of the read's condition (retile mode) resolve to zero under that
  // residual, and are counted or recorded by what stands behind it:
  //  - deadReads: the residual lies on paths the thread had left by a `ret` the read's block cannot reach (the early
  //    return before the K loop: an empty split, a tile past N). The read does not happen there. Proven here, by
  //    `implies(residual, the thread's recorded return conditions)`.
  //  - uncovered: the rest (a weight row past N the guarded copy skips, which the mma reads; a C-tile 16-byte load
  //    whose fragment stores are edge-guarded). Whether the kernel's result can depend on them is proven here for
  //    the epilogue (every device store that still names the byte refutes the residual) and by the caller for
  //    retile (the outputs those operand bytes feed are stored only where the residual is false).
  unsigned deadReads = 0;
  struct Uncovered { uint32_t sym; Instruction* site; unsigned thread, phase; EP cond; };
  std::vector<Uncovered> uncovered;
  // A clause (conjunction of literals) with its guard literals read through their definitions: a positive guard by
  // its definition when `positive` (a disjunction: the result may not be a clause), a negated guard whose definition
  // is a disjunction by the conjunction of its disjuncts' negations. The syntactic tests (implies, disjoint) match
  // literals; this puts the literals a guard stands for where they can be matched.
  EP expandGuards(EP clause, bool positive);
  // `facts` contradict `e` (guards at any depth, in either, read through their definitions): a goal-directed
  // proof - a literal by the literals in force (clamp- and division-aware), a disjunction to refute part by part,
  // a conjunction by one part; a positive guard among the facts is split on only when the literals alone do not
  // decide. Sound (a `true` is a contradiction), incomplete; `budget` bounds the steps.
  bool refutes(EP facts, EP e, unsigned budget = 20000);
  std::map<std::pair<EP, EP>, int> refuteLitMemo;                             // (facts, literal) -> held / denied / open
  std::map<std::tuple<EP, EP, bool, std::vector<EP>>, bool> refuteMemo;       // (facts, target, prove, pending) -> decided
  // The disjunction of the conditions under which `thread` returned by a way block `B` cannot reach (returns before
  // B on every path: the early returns), and the literals that therefore hold whenever B executes (the negation of
  // each such way that is a literal or a disjunction of literals: `if (a || b) return` gives !a and !b).
  EP returnedBefore(unsigned thread, BasicBlock* B);
  EP factsAt(unsigned thread, BasicBlock* B);
  std::map<std::pair<unsigned, BasicBlock*>, EP> returnedMemo;
  std::map<BasicBlock*, bool> fromEpilogue;   // a return's predecessor block is reachable from the K loop (not recorded)
  // (thread, the block a return was reached from) -> the path condition under which the thread returned that way
  // (retile mode; the exits of a kernel are usually unified into one return block, so the edge tells them apart)
  std::map<std::pair<unsigned, BasicBlock*>, EP> returned;
  bool crossesLanes(EP e);               // contains a cross-lane operation or a value the executor could not follow
  std::map<EP, bool> crossMemo;
  const Loop* iterL = nullptr; unsigned iterCount = 0;
  std::set<BasicBlock*> stopAt;   // retile mode: execution ends at these blocks (paths reaching them are dropped)
  bool factorGuards = false;      // edge conditions that are disjunctions are factored through guard symbols (Arena::factor)
  // Further traversal loops (sig.h): each runs `iterCount` times like iterL; `curIter` is the innermost's index.
  std::set<const Loop*> traversal;
  // Record a Recurrence for every header phi of every iterated loop (one extra pass of the loop body per thread,
  // effect-free: no shared traffic, barriers, copies or tensor calls are recorded during it).
  bool recordRecurrences = false;
  std::vector<Recurrence> recurrences;
  std::map<const Loop*, std::string> recurrenceWhy;   // loops whose recurrence pass failed, and why
  bool recordDevStores = false;   // retile mode: record device stores as effects (devStores) instead of dropping them
  bool recordDevLoads = false;    // retile mode: record every device read (devLoads), plain loads and cp.async sources alike
  bool reuseUniform = false;      // SIG only: reuse tid-independent values across threads. Off in production retile — it made INT4 resolve refuse a same-phase write/read.
  std::vector<DevLoad> devLoads;
  std::vector<std::vector<unsigned>> devStoreIters;   // devStores[i]'s iteration vector (parallel to devStores; retile mode)
  std::vector<std::vector<const Loop*>> devStoreLoops; // devStores[i]'s enclosing traversal loops (parallel to devStoreIters)
  unsigned privateAccesses = 0;
  uint64_t steps = 0, stepBudget = 0;   // instruction evaluations; a non-zero budget refuses the kernel when exceeded
  double wallBudget = 0; double wallStart = 0;   // seconds of wall clock per runRetile call (0 = unlimited)   // loads/stores through alloca'd memory (opaque; see Symbol::Private)
  // ord: the call's ordinal in its thread's run (1-based); an operand fill's result words carry it as their symbol name
  struct TpCall { Instruction* I; unsigned thread; unsigned iter; std::vector<EP> args; EP cond; unsigned phase; unsigned ord = 0; };
  std::vector<TpCall> tpCalls;
  struct SmemAccess { Instruction* I; unsigned thread; unsigned iter; EP addr; bool concrete; int64_t off; unsigned bytes; bool isStore; unsigned phase; };
  std::vector<SmemAccess> smemAccesses;  // every shared access, concrete offset or not (retile mode)
  std::map<uint32_t, EP> resolved;
  std::set<EP> resolveConds;   // conditions of the selections between candidate writes resolve() built
  // (premise over the arguments, assumed conclusion over the arguments) -> the reads resolved under the assumption
  // (ordered by the nodes' ids - creation order - so that the notes and the launch assumptions come out in an order
  // that does not depend on where the nodes were allocated)
  struct ByIdPair { bool operator()(const std::pair<EP, EP>& a, const std::pair<EP, EP>& b) const { return a.first->id != b.first->id ? a.first->id < b.first->id : a.second->id < b.second->id; } };
  std::map<std::pair<EP, EP>, std::vector<uint32_t>, ByIdPair> assumptions;       // SmemByte symbol -> the byte's value (after resolve)
  std::vector<Instruction*> ringCopies;  // staging copies modeled as shared writes (retile mode)
  std::map<Instruction*, std::set<Instruction*>> writeReaders;   // shared write site -> the read sites resolve() gave its bytes to
  bool runRetile(unsigned tid, std::string& why);
  unsigned curIter = ~0u;
  bool iterLast = false;                 // last iteration of iterL: its exits are taken, the back edge dropped
  std::set<uint32_t> unresolvedReads;    // retile mode: SmemByte symbols no write covers (their readers must not matter)
  // The value the last thread run computed for an IR value (null when it was not evaluated). Uniform values (a
  // loop bound, a tile origin) are the same for every thread.
  EP valueOf(Value* v) const { auto it = env.find(v); return it == env.end() ? nullptr : it->second; }
  // A shared-memory address as (object, concrete byte offset); false when it is not one
  bool sharedOffset(EP addr, GlobalVariable*& obj, int64_t& off) { return classifyAddr(addr, obj, off) && obj; }

 private:
  Function& F; DominatorTree& DT; LoopInfo& LI; Arena& A; unsigned T; const Loop* skip;
  Instruction* pendingBe = nullptr;
  EP pendingBeCond = nullptr;
  unsigned pendingBePhase = 0;
  unsigned tid = 0, phase = 0, seq = 0, unkCounter = 0, tpOrd = 0;
  bool afterSkip = false;
  bool shadow = false;                     // recurrence pass: evaluate, record nothing
  std::vector<unsigned> iterStack;         // nested traversal loops: iteration index per level (curIter = back)
  std::vector<const Loop*> loopStack;      // the traversal loop of every iterStack level
  std::string why;
  DenseMap<Value*, EP> env;
  DenseMap<Instruction*, EP> uniformEnv;  // values whose slice has no tid, reused across threads
  std::map<Value*, unsigned> valueRank;   // argument / instruction position in F (execBlock merges in this order)
  std::map<const Loop*, std::vector<Value*>> liveOutMemo;

  struct Edge { BasicBlock* from; EP cond; std::map<Value*, EP> snap; };
  using EdgeMap = std::map<BasicBlock*, std::vector<Edge>>;
  bool recurrencePass(const Loop* L, const std::vector<Edge>& headerEdges);

  bool fail(const std::string& w) { if (why.empty()) why = w; return false; }
  EP get(Value* v);
  EP evalConstant(Constant* C);
  bool execRegion(BasicBlock* entry, const Loop* L, EdgeMap& in, EdgeMap& out);
  bool execLoop(const Loop* L, std::vector<Edge> headerEdges, EdgeMap& out);
  bool genericLoop(const Loop* L, std::vector<Edge> headerEdges, EdgeMap& out);
  bool iterLoop(const Loop* L, std::vector<Edge> headerEdges, EdgeMap& out);
  bool modelCopy(CallInst* CI, EP R);   // cp.async (asm or __mvcc_cp_async_zsel_N) as a shared write of device loads
  bool skipLoop(const Loop* L, std::vector<Edge> headerEdges, EdgeMap& out);
  bool execBlock(BasicBlock* BB, std::vector<Edge>& edges, const Loop* region, EdgeMap& out);
  bool evalInst(Instruction& I, EP R);
  void edgeTo(BasicBlock* from, BasicBlock* to, EP cond, EdgeMap& out);
  const std::vector<Value*>& liveOuts(const Loop* L);
  bool storeBytes(EP addr, const std::vector<EP>& bytes, EP R, Instruction* site);
  bool loadBytes(EP addr, unsigned n, unsigned align, EP R, Instruction* site, std::vector<EP>& out);
  bool classifyAddr(EP addr, GlobalVariable*& sharedObj, int64_t& off);
  bool isPrivateAddr(EP addr);
  EP unknown(Type* ty);
  Instruction* curI = nullptr;   // the instruction being evaluated: the origin an Unknown symbol records (Symbol::v)
  EP gep(GEPOperator& G, EP base);
};

}  // namespace sym
}  // namespace mvcc
