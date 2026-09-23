// mma.sync → matmul2d recognition (catalog). Storage emit is emitFromStir.
#include "tensor_recovery.h"
#include "emit.h"
#include "legality.h"
#include "ptx_asm.h"
#include "schedule.h"
#include "sig.h"
#include "stir.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Constants.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/IntrinsicsNVPTX.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include "llvm/Support/Format.h"
#include <algorithm>
#include <chrono>
#include <functional>
#include <map>

using namespace llvm;

namespace mvcc {

std::string TpDescriptor::tag() const {
  return mslType() + "_" + std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K) + "_tl" + (tl ? "1" : "0") + "_tr" + (tr ? "1" : "0");
}

namespace {

bool isKernelFn(const Function& F) {
  if (F.getCallingConv() == CallingConv::PTX_Kernel) return true;
  if (F.hasFnAttribute("nvvm.kernel")) return true;
  if (auto* MD = F.getParent()->getNamedMetadata("nvvm.annotations"))
    for (auto* Op : MD->operands()) {
      if (Op->getNumOperands() < 3) continue;
      auto* V = mdconst::dyn_extract_or_null<Function>(Op->getOperand(0));
      auto* S = dyn_cast<MDString>(Op->getOperand(1));
      if (V == &F && S && S->getString() == "kernel") return true;
    }
  return false;
}

struct MmaRec;
struct LdRec {
  CallInst* CI = nullptr;
  bool trans = false;
  int nx = 4;                                 // matrices the instruction loads: 4 (x4) or 2 (x2)
  ExtractValueInst* field[4] = {nullptr, nullptr, nullptr, nullptr};
  int kind = -1;                              // 0 = A operand, 1 = B operand (set by the consumers)
  int matrixOfBlock[4] = {-1, -1, -1, -1};    // 8x8 block r -> ldmatrix matrix/field index
  bool reg = false;                           // register-built block: regWords are the PTX-layout words (no ldmatrix)
  bool solo = false;                          // register-built B block of a tile without an n8 neighbor: words 2, 3 are zero
  std::vector<Value*> regWords;               // A: a0..a3; B: x.b0, x.b1, y.b0, y.b1 (x = n 0-7 tile, y = n 8-15)
  // Composite block: two ldmatrix.x2 (parts[0] dominates parts[1]) that together load the four 8x8 matrices of one
  // operand block. Field f of the block is field f&1 of part f>>1. Rewritten as one x4 fill whose lanes 16-31 carry
  // the second instruction's address (see fuseX2).
  std::vector<LdRec*> parts;
  LdRec* mergedInto = nullptr;                // x2 part: the composite it belongs to (the part itself is not rewritten)
  bool composite() const { return !parts.empty(); }
  Value* slotWord[4] = {nullptr, nullptr, nullptr, nullptr};  // after rewrite: the block's words in slot order
  MmaRec* firstUse = nullptr;                 // register block: the consumer the conversion is placed before
  bool ok = true;
  std::string why;
};

struct MmaRec {
  CallInst* CI = nullptr;
  int type = 0;                               // 0 f16, 1 bf16
  Value* a[4] = {};
  Value* b[2] = {};
  Value* c[4] = {};
  ExtractValueInst* d[4] = {nullptr, nullptr, nullptr, nullptr};
  std::vector<LdRec*> aSrc, bSrc;
  int aPerm[4] = {-1, -1, -1, -1};            // a_r carries ldmatrix field aPerm[r]
  int bPerm[2] = {-1, -1};                    // b_j carries field bPerm[j]
  MmaRec* prev = nullptr;                     // accumulator chain (D of prev feeds our C)
  MmaRec* next = nullptr;
  int level = 0;
  MmaRec* pairWith = nullptr;                 // the other n8 tile sharing our B ldmatrix
  int nh = -1;                                // 0 = our B fields are the lower pair (n 0-7 of the 16-wide block)
  bool regB = false;                          // B pair is register-built (paired into a block in analyzeOperands)
  bool solo = false;                          // the only n8 tile of its B block (nh 0, no pairWith): columns 8-15 are padding
  bool ok = true;
  std::string why;
  void fail(const std::string& w) { if (ok) { ok = false; why = w; } }
};

// One matmul2d op: a sub-grid of mmas (mts x nps x levels), tile enumeration (mt_local, nt_local = 2*np_local+nh).
struct FusedOp {
  std::vector<std::vector<std::vector<MmaRec*>>> m;  // m[mtl][ntl][lvl]
  int M = 0, N = 0, K = 0;
  int type = 0;
  bool tl = false, tr = false;
  int cap() const { return M * N / 32; }
  std::vector<Value*> cList;                  // positions p = tile*4 + r (PTX order), inputs at level 0
  std::vector<Value*> dList;                  // last-level D extracts per position (a pad value at a solo block's padding positions)
  CallInst* newCall = nullptr;
  std::vector<Value*> newD;                   // extracts of newCall per position
  bool cIsSlots = false;                      // C list is slot-typed (no conversion)
  MmaRec* lastMma = nullptr;                  // insertion point
};

// Values proven to be in cooperative-slot order: value -> (group id, position)
struct SlotGroup {
  int id;
  std::vector<Value*> vals;                   // per position
  bool isPhi = false;
  BasicBlock* phiBlock = nullptr;
  FusedOp* op = nullptr;                      // when !isPhi
  int M = 0, N = 0;                           // tile shape (for conversions)
  bool valid = true;
};

class KernelRecovery {
 public:
  KernelRecovery(Function& F, const TensorRecoveryOptions& o, TensorRecoveryResult& r) : F(F), opts(o), res(r), DT(F) {}

  // Returns true if the kernel was rewritten.
  bool run();

 private:
  Function& F;
  const TensorRecoveryOptions& opts;
  TensorRecoveryResult& res;
  DominatorTree DT;
  std::vector<std::unique_ptr<LdRec>> lds;
  std::vector<std::unique_ptr<MmaRec>> mmas;
  std::map<CallInst*, LdRec*> ldByCI;
  std::map<CallInst*, MmaRec*> mmaByCI;
  std::vector<std::unique_ptr<FusedOp>> ops;
  std::vector<std::unique_ptr<SlotGroup>> groups;
  std::map<Value*, std::pair<SlotGroup*, int>> slotOf;
  // Padding positions of a solo B block (analyzeOperands). A slot-typed bundle identifies list position p with slot p
  // of the cooperative tile, and no slot is padding (the padded columns are lane bit 3, not a slot bit), so the bundle
  // must carry every position: the ops get a stand-in D value per padding position (padVals, replaced by the fused
  // call's extract and erased) and the accumulator phi webs get a phi per padding position (kept, a loop-carried slot).
  std::vector<Instruction*> padVals;
  std::string kname;

  void note(const std::string& s) { if (opts.verbose) res.notes.push_back(kname + ": " + s); }
  void warn(const std::string& s) { res.warnings.push_back(kname + ": " + s); }

  bool collect();
  void splitFp8Mmas();
  bool isRegTuple(ArrayRef<Value*> vals) const;
  LdRec* regBlock(ArrayRef<Value*> words, int kind);
  // ldmatrix.x2 pairs: -1 = every lane l and l^16 of a warp compute the same value; b >= 0 = the two may differ in
  // bit b only; -2 = unknown. Decides whether the second x2's address can serve lanes 16-31 as it stands.
  int laneHalfDiffBit(Value* v, std::map<Value*, int>& memo, int depth);
  Value* materializeAt(Value* v, Instruction* at, std::map<Value*, Value*>& memo, int depth);
  LdRec* fuseX2(LdRec* a, LdRec* b, std::string& why);
  std::map<std::pair<LdRec*, LdRec*>, LdRec*> compOf;  // composite by its ordered parts
  std::map<std::vector<Value*>, LdRec*> regRecs;      // register block by its word tuple (B: per n8 tuple)
  std::map<std::vector<Value*>, std::pair<LdRec*, int>> regBTuple;  // B n8 tuple -> (block, first field)
  bool resolveFrag(Value* v, std::vector<std::pair<LdRec*, int>>& out, SmallPtrSetImpl<Value*>& visiting);
  bool resolveTuple(ArrayRef<Value*> vals, std::vector<LdRec*>& srcs, std::vector<int>& perm, std::string& why);
  bool fieldUsesFeedMma(Value* v, SmallPtrSetImpl<Value*>& visiting, std::set<MmaRec*>& consumers, std::string& why);
  void analyzeChains();
  void analyzeOperands();
  void pairMmas();
  bool buildGrids();
  void validateFixpoint();
  void typeAccumulators();
  void rewrite();
};

// ---------------------------------------------------------------- collection

bool KernelRecovery::collect() {
  for (BasicBlock& BB : F)
    for (Instruction& I : BB) {
      auto* CI = dyn_cast<CallInst>(&I);
      if (!CI || !CI->isInlineAsm()) continue;
      auto* IA = cast<InlineAsm>(CI->getCalledOperand());
      std::vector<PtxInstr> ins; std::string err;
      if (!parsePtxAsm(std::string(IA->getAsmString()), ins, err) || ins.size() != 1) continue;
      const PtxInstr& in = ins[0];
      if (in.mnemonic == "ldmatrix") {
        const int nx = in.hasMod("x4") ? 4 : in.hasMod("x2") ? 2 : 1;
        if (nx == 1 || !in.hasMod("b16") || CI->arg_size() != 1) {
          // x1 is not recovered: its consumers fail with a clear reason. (x2 pairs are fused into x4 fills, fuseX2.)
          auto rec = std::make_unique<LdRec>(); rec->CI = CI; rec->ok = false; rec->why = "ldmatrix.x1 is not supported by recovery (x2 pairs and x4 are)";
          ldByCI[CI] = rec.get(); lds.push_back(std::move(rec));
          continue;
        }
        auto rec = std::make_unique<LdRec>();
        rec->CI = CI; rec->trans = in.hasMod("trans"); rec->nx = nx;
        for (User* U : CI->users()) {
          auto* EV = dyn_cast<ExtractValueInst>(U);
          if (!EV || EV->getNumIndices() != 1 || (int)EV->getIndices()[0] >= nx || rec->field[EV->getIndices()[0]]) { rec->ok = false; rec->why = "ldmatrix result used other than by single extractvalue per field"; break; }
          rec->field[EV->getIndices()[0]] = EV;
        }
        ldByCI[CI] = rec.get(); lds.push_back(std::move(rec));
      } else if (in.mnemonic == "mma") {
        std::string shape, dt, at, bt, ct;
        for (auto& md : in.mods) if (md.size() > 3 && md[0] == 'm' && isdigit((unsigned char)md[1])) shape = md;
        if (in.mods.size() < 4) continue;
        dt = in.mods[in.mods.size() - 4]; at = in.mods[in.mods.size() - 3]; bt = in.mods[in.mods.size() - 2]; ct = in.mods[in.mods.size() - 1];
        auto rec = std::make_unique<MmaRec>();
        rec->CI = CI;
        if (shape != "m16n8k16" || dt != "f32" || ct != "f32" || at != bt || (at != "f16" && at != "bf16") || !in.hasMod("row") || !in.hasMod("col"))
          rec->fail("mma variant " + in.full() + " is not recoverable (TensorOps on this OS: f16/bf16 inputs, f32 accumulate)");
        rec->type = at == "bf16" ? 1 : 0;
        if (CI->arg_size() != 10) rec->fail("unexpected mma operand count");
        else {
          for (int i = 0; i < 4; i++) rec->a[i] = CI->getArgOperand(i);
          for (int i = 0; i < 2; i++) rec->b[i] = CI->getArgOperand(4 + i);
          for (int i = 0; i < 4; i++) rec->c[i] = CI->getArgOperand(6 + i);
        }
        // duplicate extracts of one field (unfolded by the frontend) are merged into the dominating one
        std::vector<ExtractValueInst*> dups;
        for (User* U : CI->users()) {
          auto* EV = dyn_cast<ExtractValueInst>(U);
          if (!EV || EV->getNumIndices() != 1 || EV->getIndices()[0] > 3) { rec->fail("mma result used other than by single extractvalue per field"); break; }
          ExtractValueInst*& slot = rec->d[EV->getIndices()[0]];
          if (!slot) { slot = EV; continue; }
          if (DT.dominates(slot, EV)) { dups.push_back(EV); continue; }
          if (DT.dominates(EV, slot)) { dups.push_back(slot); slot = EV; continue; }
          rec->fail("mma result used other than by single extractvalue per field"); break;
        }
        if (rec->ok) for (ExtractValueInst* EV : dups) { EV->replaceAllUsesWith(rec->d[EV->getIndices()[0]]); EV->eraseFromParent(); }
        mmaByCI[CI] = rec.get(); mmas.push_back(std::move(rec));
      }
    }
  return !mmas.empty();
}

// ---------------------------------------------------------------- fragment provenance

bool KernelRecovery::resolveFrag(Value* v, std::vector<std::pair<LdRec*, int>>& out, SmallPtrSetImpl<Value*>& visiting) {
  if (auto* EV = dyn_cast<ExtractValueInst>(v)) {
    auto* CI = dyn_cast<CallInst>(EV->getAggregateOperand());
    auto it = CI ? ldByCI.find(CI) : ldByCI.end();
    if (it == ldByCI.end() || EV->getNumIndices() != 1) return false;
    out.push_back({it->second, (int)EV->getIndices()[0]});
    return true;
  }
  if (auto* PN = dyn_cast<PHINode>(v)) {
    if (!visiting.insert(PN).second) return true;
    for (Value* in : PN->incoming_values()) if (!resolveFrag(in, out, visiting)) return false;
    return true;
  }
  return false;
}

// Resolve an ordered tuple of fragment registers to ldmatrix sources with one consistent field permutation.
// Tuples are followed through phis *as tuples* (all members phis of the same block, incoming edge by edge) so
// that no path can mix matrices of different ldmatrix instructions into one operand block.
bool KernelRecovery::resolveTuple(ArrayRef<Value*> vals, std::vector<LdRec*>& srcs, std::vector<int>& perm, std::string& why) {
  const size_t n = vals.size();
  std::set<std::vector<Value*>> visiting;
  std::function<bool(ArrayRef<Value*>)> go = [&](ArrayRef<Value*> vs) -> bool {
    std::vector<Value*> key(vs.begin(), vs.end());
    if (!visiting.insert(key).second) return true;
    // all extracts of one ldmatrix (or, for a four-register A block, of two ldmatrix.x2 that fuse into one)?
    if (auto* EV0 = dyn_cast<ExtractValueInst>(vs[0])) {
      auto* CI = dyn_cast<CallInst>(EV0->getAggregateOperand());
      auto it = CI ? ldByCI.find(CI) : ldByCI.end();
      if (it == ldByCI.end()) { why = "fragment register is produced by something other than ldmatrix"; return false; }
      LdRec* ld = it->second;
      if (!ld->ok) { why = ld->why; return false; }
      std::vector<int> p(n, -1);
      std::vector<LdRec*> recOf(n, nullptr);
      bool oneSource = true;
      for (size_t i = 0; i < n; i++) {
        auto* EV = dyn_cast<ExtractValueInst>(vs[i]);
        auto* Ci = EV ? dyn_cast<CallInst>(EV->getAggregateOperand()) : nullptr;
        auto ii = Ci ? ldByCI.find(Ci) : ldByCI.end();
        if (ii == ldByCI.end()) { why = "operand block mixes ldmatrix fragments with other values"; return false; }
        recOf[i] = ii->second; p[i] = (int)EV->getIndices()[0];
        if (recOf[i] != ld) oneSource = false;
      }
      if (!oneSource) {
        // two x2 instructions, each contributing both of its matrices: fuse them into one four-matrix block
        LdRec* other = nullptr;
        for (LdRec* r : recOf) if (r != ld) { if (other && other != r) { why = "operand block mixes fragments of more than two ldmatrix instructions"; return false; } other = r; }
        if (n != 4 || ld->nx != 2 || other->nx != 2) { why = "operand block mixes fragments of different ldmatrix instructions"; return false; }
        if (!other->ok) { why = other->why; return false; }
        LdRec* comp = fuseX2(ld, other, why);
        if (!comp) return false;
        for (size_t i = 0; i < n; i++) p[i] += recOf[i] == comp->parts[0] ? 0 : 2;
        ld = comp;
      }
      if (perm.empty()) perm = p;
      else if (perm != p) { why = "fragment register order differs between control-flow paths"; return false; }
      if (std::find(srcs.begin(), srcs.end(), ld) == srcs.end()) srcs.push_back(ld);
      return true;
    }
    if (auto* PN0 = dyn_cast<PHINode>(vs[0])) {
      BasicBlock* BB = PN0->getParent();
      std::vector<PHINode*> phis;
      for (Value* v : vs) { auto* PN = dyn_cast<PHINode>(v); if (!PN || PN->getParent() != BB) { why = "fragment registers do not flow together through phis"; return false; } phis.push_back(PN); }
      for (unsigned e = 0; e < PN0->getNumIncomingValues(); e++) {
        BasicBlock* pred = PN0->getIncomingBlock(e);
        std::vector<Value*> in;
        for (auto* PN : phis) in.push_back(PN->getIncomingValueForBlock(pred));
        if (!go(in)) return false;
      }
      return true;
    }
    why = "fragment register is produced by something other than ldmatrix (register-built fragments are not recovered yet)";
    return false;
  };
  if (!go(vals)) return false;
  if (srcs.empty()) { why = "fragment provenance is a phi cycle without a source"; return false; }
  std::vector<int> sorted = perm; std::sort(sorted.begin(), sorted.end());
  for (size_t i = 0; i < n; i++) if (sorted[i] != (int)i && n == 4) { why = "operand block does not use all four ldmatrix fields exactly once"; return false; }
  if (n == 2 && (perm[0] == perm[1])) { why = "b0/b1 use the same ldmatrix field"; return false; }
  return true;
}

// ---------------------------------------------------------------- ldmatrix.x2 pairs
//
// ldmatrix.x2 loads two 8x8 matrices from the row addresses of lanes 0-15 (lanes 16-31 are ignored). A 16x16
// operand block then takes two instructions: a B n8 tile's (b0, b1) from one, its n8 neighbor's from the other
// (attention P.V, GDN), or an A block's four registers from two. The pair is emitted as one x4 fill on the address
//   addr4 = lane < 16 ? addr_first : addr_second
// which is exactly an x4 operand (lanes 8m..8m+7 hold matrix m's rows, m = 0..3) provided addr_second at lane l >= 16
// equals addr_second at lane l - 16. That holds when every lane-dependent leaf of the expression is masked below bit
// 4 (the idiom is `lane & 15`), proven by laneHalfDiffBit; otherwise the value is fetched from lane l - 16 with one
// shuffle. The fill is placed at the first instruction (it dominates the second); the second's address expression
// is re-materialized there when it is computed later (pure address arithmetic, cloned). Field f of the block is
// field f&1 of part f>>1, so every consumer of either instruction - directly or through the phi web of a
// register-double-buffered kernel - reads the fused fill's words instead. Exact by construction: the same 32 rows
// reach the same slots as with two separate x2 instructions.

int KernelRecovery::laneHalfDiffBit(Value* v, std::map<Value*, int>& memo, int depth) {
  auto it = memo.find(v);
  if (it != memo.end()) return it->second;
  if (depth > 96) return memo[v] = -2;
  if (isa<Constant>(v) || isa<Argument>(v)) return memo[v] = -1;
  auto* I = dyn_cast<Instruction>(v);
  if (!I) return memo[v] = -2;
  auto join = [&](int a, int b) { if (a == -2 || b == -2) return -2; if (a == -1) return b; if (b == -1) return a; return a == b ? a : -2; };
  auto uniformArgs = [&](ArrayRef<Value*> ops) { for (Value* o : ops) if (laneHalfDiffBit(o, memo, depth + 1) != -1) return false; return true; };
  int r = -2;
  if (auto* CI = dyn_cast<CallInst>(I)) {
    Function* Callee = CI->getCalledFunction();
    StringRef n = Callee ? Callee->getName() : "";
    if (n == "llvm.nvvm.read.ptx.sreg.tid.x" || n == "llvm.nvvm.read.ptx.sreg.laneid") r = 4;   // lane l and l^16 differ in bit 4
    else if (n.starts_with("llvm.nvvm.read.ptx.sreg.") && !n.starts_with("llvm.nvvm.read.ptx.sreg.tid.")) r = -1;  // block ids, sizes: warp-uniform
    else r = -2;
  } else if (auto* PN = dyn_cast<PHINode>(I)) {
    // optimistic: a phi web whose external inputs are all uniform is uniform in every iteration
    memo[v] = -1;
    for (Value* in : PN->incoming_values()) if (laneHalfDiffBit(in, memo, depth + 1) != -1) { memo[v] = -2; return -2; }
    return -1;
  } else if (auto* BO = dyn_cast<BinaryOperator>(I)) {
    int a = laneHalfDiffBit(BO->getOperand(0), memo, depth + 1), b = laneHalfDiffBit(BO->getOperand(1), memo, depth + 1);
    auto* C1 = dyn_cast<ConstantInt>(BO->getOperand(1));
    switch (BO->getOpcode()) {
      case Instruction::And:
        r = join(a, b);
        if (r >= 0) { auto* C0 = dyn_cast<ConstantInt>(BO->getOperand(0)); const ConstantInt* C = C1 ? C1 : C0; if (C && (unsigned)r < C->getBitWidth() && !C->getValue()[r]) r = -1; }
        break;
      case Instruction::Or:
        r = join(a, b);
        if (r >= 0) { auto* C0 = dyn_cast<ConstantInt>(BO->getOperand(0)); const ConstantInt* C = C1 ? C1 : C0; if (C && (unsigned)r < C->getBitWidth() && C->getValue()[r]) r = -1; }
        break;
      case Instruction::Xor: r = join(a, b); break;
      case Instruction::Shl: if (C1 && b == -1 && a >= 0) r = a + (int)C1->getZExtValue(); else if (a == -1 && b == -1) r = -1; break;
      case Instruction::LShr: if (C1 && b == -1 && a >= 0) r = a >= (int)C1->getZExtValue() ? a - (int)C1->getZExtValue() : -1; else if (a == -1 && b == -1) r = -1; break;
      case Instruction::AShr: if (C1 && b == -1 && a >= 0 && (unsigned)a + 1 < BO->getType()->getIntegerBitWidth()) r = a >= (int)C1->getZExtValue() ? a - (int)C1->getZExtValue() : -1; else if (a == -1 && b == -1) r = -1; break;
      case Instruction::Mul:
        if (C1 && b == -1 && a >= 0 && C1->getValue().isPowerOf2()) r = a + (int)C1->getValue().logBase2(); else if (a == -1 && b == -1) r = -1; break;
      case Instruction::UDiv:
        if (C1 && b == -1 && a >= 0 && C1->getValue().isPowerOf2()) { int k = (int)C1->getValue().logBase2(); r = a >= k ? a - k : -1; } else if (a == -1 && b == -1) r = -1; break;
      case Instruction::URem:
        if (C1 && b == -1 && a >= 0 && C1->getValue().isPowerOf2()) r = a < (int)C1->getValue().logBase2() ? a : -1; else if (a == -1 && b == -1) r = -1; break;
      default: r = (a == -1 && b == -1) ? -1 : -2; break;   // add/sub carry across bits
    }
    if (r >= 0 && (unsigned)r >= BO->getType()->getIntegerBitWidth()) r = -1;
  } else if (auto* T = dyn_cast<TruncInst>(I)) {
    int a = laneHalfDiffBit(T->getOperand(0), memo, depth + 1);
    r = a >= 0 && (unsigned)a >= T->getType()->getIntegerBitWidth() ? -1 : a;
  } else if (isa<ZExtInst>(I)) {
    r = laneHalfDiffBit(I->getOperand(0), memo, depth + 1);
  } else if (auto* S = dyn_cast<SExtInst>(I)) {
    int a = laneHalfDiffBit(S->getOperand(0), memo, depth + 1);
    r = a >= 0 && (unsigned)a + 1 >= S->getOperand(0)->getType()->getIntegerBitWidth() ? -2 : a;
  } else if (auto* SI = dyn_cast<SelectInst>(I)) {
    if (laneHalfDiffBit(SI->getCondition(), memo, depth + 1) != -1) r = -2;
    else r = join(laneHalfDiffBit(SI->getTrueValue(), memo, depth + 1), laneHalfDiffBit(SI->getFalseValue(), memo, depth + 1));
  } else if (isa<PtrToIntInst>(I) || isa<IntToPtrInst>(I) || isa<AddrSpaceCastInst>(I) || isa<BitCastInst>(I) || isa<FreezeInst>(I)) {
    r = laneHalfDiffBit(I->getOperand(0), memo, depth + 1);
  } else if (isa<GetElementPtrInst>(I) || isa<ICmpInst>(I)) {
    std::vector<Value*> ops; for (Use& U : I->operands()) ops.push_back(U.get());
    r = uniformArgs(ops) ? -1 : -2;
  }
  return memo[v] = r;
}

// v, or a clone of its pure address arithmetic, available before `at` (nullptr when some leaf is not).
Value* KernelRecovery::materializeAt(Value* v, Instruction* at, std::map<Value*, Value*>& memo, int depth) {
  auto it = memo.find(v);
  if (it != memo.end()) return it->second;
  if (depth > 96) return nullptr;
  if (isa<Constant>(v) || isa<Argument>(v)) return memo[v] = v;
  auto* I = dyn_cast<Instruction>(v);
  if (!I) return nullptr;
  if (DT.dominates(I, at)) return memo[v] = v;
  if (auto* CI = dyn_cast<CallInst>(I)) {
    Function* Callee = CI->getCalledFunction();
    if (!Callee || !Callee->getName().starts_with("llvm.nvvm.read.ptx.sreg.")) return nullptr;
  } else if (!(isa<BinaryOperator>(I) || isa<CastInst>(I) || isa<GetElementPtrInst>(I) || isa<SelectInst>(I) || isa<ICmpInst>(I) || isa<FreezeInst>(I))) return nullptr;
  Instruction* C = I->clone();
  for (unsigned i = 0; i < C->getNumOperands(); i++) {
    Value* o = materializeAt(C->getOperand(i), at, memo, depth + 1);
    if (!o) { C->deleteValue(); return nullptr; }
    C->setOperand(i, o);
  }
  C->insertBefore(at->getIterator());
  C->setDebugLoc(at->getDebugLoc());
  return memo[v] = C;
}

LdRec* KernelRecovery::fuseX2(LdRec* a, LdRec* b, std::string& why) {
  if (a == b || a->nx != 2 || b->nx != 2 || a->composite() || b->composite()) { why = "operand block mixes fragments of different ldmatrix instructions"; return nullptr; }
  if (a->mergedInto || b->mergedInto) {
    if (a->mergedInto && a->mergedInto == b->mergedInto) return a->mergedInto;
    why = "an ldmatrix.x2 pairs with different partners in different operand blocks"; return nullptr;
  }
  if (!a->ok) { why = a->why; return nullptr; }
  if (!b->ok) { why = b->why; return nullptr; }
  if (a->trans != b->trans) { why = "the two ldmatrix.x2 of one operand block differ in .trans"; return nullptr; }
  if (!DT.dominates(a->CI, b->CI)) { if (!DT.dominates(b->CI, a->CI)) { why = "the two ldmatrix.x2 of one operand block are not ordered by dominance"; return nullptr; } std::swap(a, b); }
  {
    // the second address must be computable at the first instruction (checked without emitting: leaves dominate)
    std::function<bool(Value*, int)> avail = [&](Value* v, int depth) -> bool {
      if (depth > 96) return false;
      if (isa<Constant>(v) || isa<Argument>(v)) return true;
      auto* I = dyn_cast<Instruction>(v);
      if (!I) return false;
      if (DT.dominates(I, a->CI)) return true;
      if (auto* CI = dyn_cast<CallInst>(I)) return CI->getCalledFunction() && CI->getCalledFunction()->getName().starts_with("llvm.nvvm.read.ptx.sreg.");
      if (!(isa<BinaryOperator>(I) || isa<CastInst>(I) || isa<GetElementPtrInst>(I) || isa<SelectInst>(I) || isa<ICmpInst>(I) || isa<FreezeInst>(I))) return false;
      for (Use& U : I->operands()) if (!avail(U.get(), depth + 1)) return false;
      return true;
    };
    if (!avail(b->CI->getArgOperand(0), 0)) { why = "the second ldmatrix.x2's address is not computable at the first (a value loaded or carried in between)"; return nullptr; }
  }
  auto rec = std::make_unique<LdRec>();
  rec->trans = a->trans; rec->nx = 4; rec->parts = {a, b};
  LdRec* R = rec.get(); lds.push_back(std::move(rec));
  a->mergedInto = b->mergedInto = R;
  compOf[{a, b}] = R;
  return R;
}

bool KernelRecovery::fieldUsesFeedMma(Value* v, SmallPtrSetImpl<Value*>& visiting, std::set<MmaRec*>& consumers, std::string& why) {
  if (!visiting.insert(v).second) return true;
  for (User* U : v->users()) {
    if (auto* CI = dyn_cast<CallInst>(U)) {
      auto it = mmaByCI.find(CI);
      if (it == mmaByCI.end()) { why = "fragment escapes to a non-mma call"; return false; }
      consumers.insert(it->second);
      continue;
    }
    if (auto* PN = dyn_cast<PHINode>(U)) { if (!fieldUsesFeedMma(PN, visiting, consumers, why)) return false; continue; }
    why = std::string("fragment escapes to ") + (isa<Instruction>(U) ? cast<Instruction>(U)->getOpcodeName() : "a non-instruction user");
    return false;
  }
  return true;
}

// ---------------------------------------------------------------- chains, operands, pairs

void KernelRecovery::analyzeChains() {
  for (auto& mp : mmas) {
    MmaRec* Y = mp.get();
    if (!Y->ok) continue;
    MmaRec* X = nullptr;
    bool chain = true;
    for (int i = 0; i < 4 && chain; i++) {
      auto* EV = dyn_cast<ExtractValueInst>(Y->c[i]);
      auto* CI = EV ? dyn_cast<CallInst>(EV->getAggregateOperand()) : nullptr;
      auto it = CI ? mmaByCI.find(CI) : mmaByCI.end();
      if (it == mmaByCI.end() || (int)EV->getIndices()[0] != i) { chain = false; break; }
      if (!X) X = it->second; else if (X != it->second) { chain = false; break; }
    }
    if (!chain || !X || !X->ok || X->next) continue;
    // every D field of X must be used only by Y (once), and X must dominate Y
    bool exclusive = true;
    for (int i = 0; i < 4; i++) if (!X->d[i] || !X->d[i]->hasOneUse() || X->d[i]->user_back() != Y->CI) exclusive = false;
    if (!exclusive || !DT.dominates(X->CI, Y->CI)) continue;
    X->next = Y; Y->prev = X;
  }
  for (auto& mp : mmas) { int l = 0; for (MmaRec* p = mp->prev; p; p = p->prev) l++; mp->level = l; }
}

// A tuple none of whose words is an ldmatrix field or a phi: an operand block the kernel assembled in registers.
// (A tuple mixing ldmatrix fields with computed words, or reaching computed words through phis, is declined.)
bool KernelRecovery::isRegTuple(ArrayRef<Value*> vals) const {
  // phis are looked through: a fragment loaded under a guard (`if (rows_valid) a = lds(...)`, the other arm undef)
  // is a register-built block whose words are the phis themselves
  SmallPtrSet<Value*, 16> seen;
  std::vector<Value*> work(vals.begin(), vals.end());
  while (!work.empty()) {
    Value* v = work.back(); work.pop_back();
    if (!seen.insert(v).second) continue;
    if (auto* PN = dyn_cast<PHINode>(v)) { for (Value* in : PN->incoming_values()) work.push_back(in); continue; }
    if (auto* EV = dyn_cast<ExtractValueInst>(v)) { auto* CI = dyn_cast<CallInst>(EV->getAggregateOperand()); if (CI && ldByCI.count(CI)) return false; }
  }
  return true;
}

LdRec* KernelRecovery::regBlock(ArrayRef<Value*> words, int kind) {
  std::vector<Value*> key(words.begin(), words.end());
  auto it = regRecs.find(key);
  if (it != regRecs.end()) return it->second;
  auto rec = std::make_unique<LdRec>();
  rec->reg = true; rec->regWords = key; rec->kind = kind; rec->trans = false;
  for (int r = 0; r < 4; r++) rec->matrixOfBlock[r] = r;   // PTX word order is block order (m/k or k/n halves)
  LdRec* R = rec.get(); lds.push_back(std::move(rec)); regRecs[key] = R;
  return R;
}

void KernelRecovery::analyzeOperands() {
  for (auto& mp : mmas) {
    MmaRec* m = mp.get();
    if (!m->ok) continue;
    std::vector<int> perm; std::string why;
    if (isRegTuple(ArrayRef<Value*>(m->a, 4))) {
      m->aSrc = {regBlock(ArrayRef<Value*>(m->a, 4), 0)};
      for (int i = 0; i < 4; i++) m->aPerm[i] = i;
    } else {
      if (!resolveTuple(ArrayRef<Value*>(m->a, 4), m->aSrc, perm, why)) { m->fail("A operand: " + why); continue; }
      for (int i = 0; i < 4; i++) m->aPerm[i] = perm[i];
    }
    perm.clear();
    if (isRegTuple(ArrayRef<Value*>(m->b, 2))) { m->regB = true; }
    else {
      if (!resolveTuple(ArrayRef<Value*>(m->b, 2), m->bSrc, perm, why)) { m->fail("B operand: " + why); continue; }
      m->bPerm[0] = perm[0]; m->bPerm[1] = perm[1];
    }
    for (LdRec* ld : m->aSrc) {
      if (ld->kind == 1) { m->fail("ldmatrix feeds both A and B operands"); break; }
      ld->kind = 0;
      for (int r = 0; r < 4; r++) {
        if (ld->matrixOfBlock[r] >= 0 && ld->matrixOfBlock[r] != m->aPerm[r]) { m->fail("ldmatrix fields feed different A blocks in different mmas"); break; }
        ld->matrixOfBlock[r] = m->aPerm[r];  // block r = (m>=8) + 2*(k>=8) <- field
      }
    }
    for (LdRec* ld : m->bSrc) {
      if (ld->kind == 0) { m->fail("ldmatrix feeds both A and B operands"); break; }
      ld->kind = 1;
    }
  }
  // Register-built B pairs: two n8 tiles that share the A tuple, the level and the block form one 16-wide block
  // (x = the first in program order takes n 0-7 of the block; which logical columns they are does not matter to
  // the matmul, every accumulator position keeps its own identity). The pairing must then hold at every k-step,
  // which buildGrids checks.
  std::vector<MmaRec*> order;
  for (auto& mp : mmas) if (mp->ok && mp->regB) order.push_back(mp.get());
  std::sort(order.begin(), order.end(), [](MmaRec* x, MmaRec* y) { return x->CI->comesBefore(y->CI); });
  for (MmaRec* m : order) {
    if (!m->ok || !m->bSrc.empty()) continue;
    std::vector<Value*> key = {m->b[0], m->b[1]};
    auto it = regBTuple.find(key);
    if (it != regBTuple.end()) {
      m->bSrc = {it->second.first}; m->bPerm[0] = it->second.second; m->bPerm[1] = it->second.second + 1;
      if (it->second.first->solo) { m->solo = true; m->nh = 0; }
      continue;
    }
    MmaRec* o = nullptr;
    for (MmaRec* c : order) {
      if (c == m || !c->ok || !c->bSrc.empty() || c->level != m->level || c->CI->getParent() != m->CI->getParent()) continue;
      if (!std::equal(m->a, m->a + 4, c->a) || (c->b[0] == m->b[0] && c->b[1] == m->b[1])) continue;
      if (regBTuple.count({c->b[0], c->b[1]})) continue;
      if (!m->CI->comesBefore(c->CI)) continue;
      o = c; break;
    }
    if (!o) {
      // A tile with no n8 neighbor in its warp (decode kernels: 4 warps x n8 across a 32-wide tile): the block's
      // columns 8-15 are zero. Exact - D = C + A.[B 0] leaves the tile's columns as with the n8 instruction and the
      // padded columns (C 0, B 0) are never read. Half the tensor op is padding; the mma chain still runs on the
      // TensorOps rather than the exact per-lane lowering.
      Value* z = ConstantInt::get(Type::getInt32Ty(F.getContext()), 0);
      LdRec* rec = regBlock({m->b[0], m->b[1], z, z}, 1);
      rec->solo = true;
      regBTuple[key] = {rec, 0};
      m->bSrc = {rec}; m->bPerm[0] = 0; m->bPerm[1] = 1; m->solo = true; m->nh = 0;
      continue;
    }
    LdRec* rec = regBlock({m->b[0], m->b[1], o->b[0], o->b[1]}, 1);
    regBTuple[key] = {rec, 0}; regBTuple[{o->b[0], o->b[1]}] = {rec, 2};
    m->bSrc = {rec}; m->bPerm[0] = 0; m->bPerm[1] = 1;
    o->bSrc = {rec}; o->bPerm[0] = 2; o->bPerm[1] = 3;
  }
}

// Two mmas sharing the A tuple and the B ldmatrix set, with complementary B fields, form one 16-wide B block.
// Two mmas sharing A whose B halves come from two ldmatrix.x2 fuse those into one block first (fuseX2).
void KernelRecovery::pairMmas() {
  for (auto& mp : mmas) {
    MmaRec* m = mp.get();
    if (!m->ok || m->pairWith || m->solo) continue;
    MmaRec* best = nullptr;
    std::string x2why;
    for (auto& np : mmas) {
      MmaRec* o = np.get();
      if (o == m || !o->ok || o->pairWith || o->solo || o->level != m->level || o->CI->getParent() != m->CI->getParent()) continue;
      if (!std::equal(m->a, m->a + 4, o->a)) continue;
      if (m->bSrc != o->bSrc) {
        // each side ldmatrix.x2 holding both k halves of its n8 tile: one 16-wide block across the two. A
        // register-double-buffered kernel reaches the mma from several x2 (prologue and loop body) through phis; the
        // two sides then fuse pairwise, block by block, with the same order on every path.
        if (m->bSrc.size() != o->bSrc.size()) continue;
        bool x2 = true;
        for (LdRec* l : m->bSrc) if (l->nx != 2 || l->composite()) x2 = false;
        for (LdRec* l : o->bSrc) if (l->nx != 2 || l->composite()) x2 = false;
        if (!x2) continue;
        std::vector<std::pair<LdRec*, LdRec*>> pairs;
        for (LdRec* lm : m->bSrc) {
          LdRec* lo = nullptr;
          for (LdRec* c : o->bSrc) if (c->CI->getParent() == lm->CI->getParent()) { if (lo) { lo = nullptr; break; } lo = c; }
          if (!lo) break;
          if (lm->mergedInto && lm->mergedInto != lo->mergedInto) { lo = nullptr; break; }
          pairs.push_back({lm, lo});
        }
        if (pairs.size() != m->bSrc.size()) { x2why = "the ldmatrix.x2 of the two n8 tiles do not pair up block by block"; continue; }
        std::vector<LdRec*> comps; int off = -1; bool ok = true;
        for (auto& pr : pairs) {
          LdRec* comp = fuseX2(pr.first, pr.second, x2why);
          if (!comp) { ok = false; break; }
          const int offM = comp->parts[0] == pr.first ? 0 : 2;
          if (off >= 0 && off != offM) { x2why = "the two ldmatrix.x2 of a B block come in different orders on different paths"; ok = false; break; }
          off = offM; comps.push_back(comp); comp->kind = 1;
        }
        if (!ok) continue;
        m->bPerm[0] += off; m->bPerm[1] += off; o->bPerm[0] += 2 - off; o->bPerm[1] += 2 - off;
        m->bSrc = comps; o->bSrc = comps;
      }
      std::set<int> f = {m->bPerm[0], m->bPerm[1], o->bPerm[0], o->bPerm[1]};
      if (f.size() != 4) continue;
      best = o; break;
    }
    if (!best) { m->fail(x2why.empty() ? "no partner mma shares A and the other half of the B ldmatrix (recovery pairs n8 tiles into 16-wide blocks)" : "B operand: " + x2why); continue; }
    // lower pair of fields -> nh 0 (n 0-7 of the block)
    int mMin = std::min(m->bPerm[0], m->bPerm[1]), oMin = std::min(best->bPerm[0], best->bPerm[1]);
    m->nh = mMin < oMin ? 0 : 1; best->nh = 1 - m->nh;
    m->pairWith = best; best->pairWith = m;
    for (LdRec* ld : m->bSrc) {
      // block r = 2*nh + kh <- field
      int map[4] = {-1, -1, -1, -1};
      map[2 * m->nh + 0] = m->bPerm[0]; map[2 * m->nh + 1] = m->bPerm[1];
      map[2 * best->nh + 0] = best->bPerm[0]; map[2 * best->nh + 1] = best->bPerm[1];
      for (int r = 0; r < 4; r++) {
        if (ld->matrixOfBlock[r] >= 0 && ld->matrixOfBlock[r] != map[r]) { m->fail("ldmatrix fields feed different B blocks in different mma pairs"); best->fail(m->why); }
        ld->matrixOfBlock[r] = map[r];
      }
    }
  }
}

// ---------------------------------------------------------------- grids -> fused ops

bool KernelRecovery::buildGrids() {
  // chains by head, grouped per head block
  std::map<BasicBlock*, std::vector<MmaRec*>> headsByBlock;
  for (auto& mp : mmas) if (mp->ok && !mp->prev) headsByBlock[mp->CI->getParent()].push_back(mp.get());
  bool any = false;
  for (auto& kv : headsByBlock) {
    std::vector<MmaRec*> heads = kv.second;
    // connected components over shared A tuple / shared B pair at level 0
    std::map<MmaRec*, int> comp;
    int nc = 0;
    for (MmaRec* h : heads) {
      if (comp.count(h)) continue;
      std::vector<MmaRec*> stack = {h}; comp[h] = nc;
      while (!stack.empty()) {
        MmaRec* x = stack.back(); stack.pop_back();
        for (MmaRec* y : heads) {
          if (comp.count(y)) continue;
          bool link = std::equal(x->a, x->a + 4, y->a) || x->bSrc == y->bSrc || x->pairWith == y;
          if (link) { comp[y] = nc; stack.push_back(y); }
        }
      }
      nc++;
    }
    for (int c = 0; c < nc; c++) {
      std::vector<MmaRec*> chains;
      for (MmaRec* h : heads) if (comp[h] == c) chains.push_back(h);
      // sort chains by program order for a deterministic enumeration
      std::sort(chains.begin(), chains.end(), [&](MmaRec* x, MmaRec* y) { return x->CI->comesBefore(y->CI); });
      auto failAll = [&](const std::string& w) { for (MmaRec* h : chains) for (MmaRec* m = h; m; m = m->next) m->fail(w); };
      // chain lengths and per-level blocks
      size_t L = 0; for (MmaRec* m = chains[0]; m; m = m->next) L++;
      bool okL = true;
      for (MmaRec* h : chains) { size_t l = 0; for (MmaRec* m = h; m; m = m->next) l++; if (l != L) okL = false; }
      if (!okL) { failAll("accumulator chains in one warp tile have different lengths"); continue; }
      for (size_t lvl = 0; lvl < L; lvl++) {
        MmaRec* ref = chains[0]; for (size_t i = 0; i < lvl; i++) ref = ref->next;
        for (MmaRec* h : chains) { MmaRec* m = h; for (size_t i = 0; i < lvl; i++) m = m->next; if (m->CI->getParent() != ref->CI->getParent() || !m->ok) okL = false; }
      }
      if (!okL) { failAll("k-step mma batches are not one basic block per level"); continue; }
      // mt / np indices from level 0
      std::vector<std::array<Value*, 4>> mtKeys; std::vector<std::vector<LdRec*>> npKeys;
      std::map<MmaRec*, std::pair<int, int>> idx;  // chain head -> (mt, np)
      for (MmaRec* h : chains) {
        std::array<Value*, 4> ak = {h->a[0], h->a[1], h->a[2], h->a[3]};
        int mt = (int)(std::find(mtKeys.begin(), mtKeys.end(), ak) - mtKeys.begin()); if (mt == (int)mtKeys.size()) mtKeys.push_back(ak);
        int np = (int)(std::find(npKeys.begin(), npKeys.end(), h->bSrc) - npKeys.begin()); if (np == (int)npKeys.size()) npKeys.push_back(h->bSrc);
        idx[h] = {mt, np};
      }
      const int MT = (int)mtKeys.size(), NP = (int)npKeys.size();
      // a solo B block (zero-padded n8 tile, analyzeOperands) holds one tile per A block; its nh 1 cells stay empty
      int expect = 0, solos = 0;
      for (auto& k : npKeys) { const bool solo = k.size() == 1 && k[0]->reg && k[0]->solo; expect += solo ? MT : 2 * MT; solos += solo; }
      if ((int)chains.size() != expect) { failAll("warp tile grid is incomplete: " + std::to_string(chains.size()) + " accumulator tiles for " + std::to_string(MT) + " A blocks x " + std::to_string(NP) + " B blocks"); continue; }
      std::vector<std::vector<std::vector<MmaRec*>>> grid(MT, std::vector<std::vector<MmaRec*>>(2 * NP, std::vector<MmaRec*>(L, nullptr)));
      bool dup = false;
      for (MmaRec* h : chains) {
        auto [mt, np] = idx[h];
        MmaRec* m = h;
        for (size_t lvl = 0; lvl < L; lvl++, m = m->next) { if (grid[mt][2 * np + h->nh][lvl]) dup = true; grid[mt][2 * np + h->nh][lvl] = m; }
      }
      if (dup) { failAll("two accumulator tiles map to the same (A block, B block, half)"); continue; }
      // consistency at deeper levels: same mt -> same A tuple, same np -> same B sources; different -> different
      std::string inconsistency;
      for (size_t lvl = 1; lvl < L && inconsistency.empty(); lvl++)
        for (int mt = 0; mt < MT && inconsistency.empty(); mt++)
          for (int nt = 0; nt < 2 * NP && inconsistency.empty(); nt++) {
            MmaRec* m = grid[mt][nt][lvl];
            if (!m) continue;
            if (m->nh != (nt & 1)) { inconsistency = "n-half of tile (" + std::to_string(mt) + "," + std::to_string(nt) + ") flips at k-step " + std::to_string(lvl); break; }
            for (int mt2 = 0; mt2 < MT; mt2++) for (int nt2 = 0; nt2 < 2 * NP; nt2++) {
              MmaRec* o = grid[mt2][nt2][lvl];
              if (!o) continue;
              bool sameA = std::equal(m->a, m->a + 4, o->a), sameB = m->bSrc == o->bSrc;
              if (sameA != (mt == mt2)) inconsistency = "A fragment sharing between tiles differs at k-step " + std::to_string(lvl);
              else if (sameB != (nt / 2 == nt2 / 2)) inconsistency = "B fragment sharing between tiles differs at k-step " + std::to_string(lvl);
              else if (mt == mt2 && nt / 2 == nt2 / 2 && (nt & 1) != (nt2 & 1) && m->pairWith != o) {
                std::string where = "outside the grid";
                for (int a = 0; a < MT; a++) for (int b = 0; b < 2 * NP; b++) for (size_t c = 0; c < L; c++) if (grid[a][b][c] == m->pairWith) where = "(" + std::to_string(a) + "," + std::to_string(b) + ",k" + std::to_string(c) + ")";
                inconsistency = "n8 tile pairing differs at k-step " + std::to_string(lvl) + ": tile (" + std::to_string(mt) + "," + std::to_string(nt) + ") pairs with " + where + " instead of (" + std::to_string(mt2) + "," + std::to_string(nt2) + ")";
              }
            }
          }
      if (!inconsistency.empty()) { failAll("A/B block sharing differs between k-steps: " + inconsistency); continue; }
      int type = chains[0]->type;
      bool sameType = true; for (MmaRec* h : chains) for (MmaRec* m = h; m; m = m->next) if (m->type != type) sameType = false;
      if (!sameType) { failAll("mixed f16/bf16 mmas in one warp tile"); continue; }
      if (solos) note(std::to_string(solos) + " B block(s) of a single n8 tile zero-padded to 16 columns (no n8 neighbor in the warp; half of each op's B is padding)");
      // transposition flags: all A ldmatrix share .trans, all B share .trans
      int aT = -1, bT = -1; bool tOk = true;
      for (MmaRec* h : chains) for (MmaRec* m = h; m; m = m->next) {
        for (LdRec* ld : m->aSrc) { if (aT >= 0 && aT != (int)ld->trans) tOk = false; aT = ld->trans; }
        for (LdRec* ld : m->bSrc) { if (bT >= 0 && bT != (int)ld->trans) tOk = false; bT = ld->trans; }
      }
      if (!tOk) { failAll("mixed .trans / non-.trans ldmatrix for one operand"); continue; }
      // partition: mt groups of <=2, np groups of <=2, level groups of <=2; each op needs a 32 somewhere
      std::vector<std::pair<int, int>> mtG, npG, lvG;
      const int gM = std::clamp(opts.groupM, 1, 2), gN = std::clamp(opts.groupN, 1, 2);
      // K depth 32 per op whenever the chain allows it: measured on M5 Pro, 32x32x32 ops beat two 32x32x16 ops for
      // every warp tile (64x128/32x64: 13.6 -> 23.3 TFLOP/s; 128x64/64x32 and 64x64/32x32: unchanged)
      const int gK = opts.groupK ? std::clamp(opts.groupK, 1, 2) : 2;
      for (int i = 0; i < MT; i += gM) mtG.push_back({i, std::min(MT, i + gM)});
      for (int i = 0; i < NP; i += gN) npG.push_back({i, std::min(NP, i + gN)});
      for (size_t i = 0; i < L; i += gK) lvG.push_back({(int)i, (int)std::min(L, i + gK)});
      bool shapeOk = true;
      std::vector<std::unique_ptr<FusedOp>> local;
      for (auto& mg : mtG) for (auto& ng : npG) for (auto& lg : lvG) {
        auto op = std::make_unique<FusedOp>();
        op->M = 16 * (mg.second - mg.first); op->N = 16 * (ng.second - ng.first); op->K = 16 * (lg.second - lg.first);
        if (op->M != 32 && op->N != 32 && op->K != 32) { shapeOk = false; }
        op->type = type; op->tl = aT == 1; op->tr = bT == 0;
        op->m.assign(mg.second - mg.first, std::vector<std::vector<MmaRec*>>(2 * (ng.second - ng.first), std::vector<MmaRec*>(lg.second - lg.first)));
        for (int mt = mg.first; mt < mg.second; mt++) for (int nt = 2 * ng.first; nt < 2 * ng.second; nt++) for (int lv = lg.first; lv < lg.second; lv++)
          op->m[mt - mg.first][nt - 2 * ng.first][lv - lg.first] = grid[mt][nt][lv];
        local.push_back(std::move(op));
      }
      if (!shapeOk) { failAll("warp tile too small for cooperative TensorOps (needs 32 in M, N or K: " + std::to_string(16 * MT) + "x" + std::to_string(16 * NP) + "x" + std::to_string(16 * L) + " per K slab)"); continue; }
      for (auto& op : local) ops.push_back(std::move(op));
      any = true;
    }
  }
  return any;
}

// Fixpoint: an ldmatrix is recoverable only if all its consumers are recovered mmas of valid fused ops and none of
// its fields escape; a fused op is valid only if all its ldmatrix sources are recoverable.
void KernelRecovery::validateFixpoint() {
  std::map<MmaRec*, FusedOp*> opOf;
  for (auto& op : ops) for (auto& a : op->m) for (auto& b : a) for (MmaRec* m : b) if (m) opOf[m] = op.get();
  for (auto& mp : mmas) if (!opOf.count(mp.get())) mp->fail(mp->ok ? "not part of a complete warp-tile grid" : mp->why);
  std::map<LdRec*, std::set<MmaRec*>> consumers;
  for (auto& lp : lds) {
    LdRec* ld = lp.get();
    if (!ld->ok || ld->reg || ld->mergedInto) continue;   // an x2 part is judged through its composite
    std::string why;
    std::vector<LdRec*> srcs = ld->composite() ? ld->parts : std::vector<LdRec*>{ld};
    for (LdRec* s : srcs) for (int i = 0; i < s->nx && ld->ok; i++) {
      if (!s->field[i]) continue;
      SmallPtrSet<Value*, 8> visiting;
      if (!fieldUsesFeedMma(s->field[i], visiting, consumers[ld], why)) { ld->ok = false; ld->why = why; }
    }
    if (ld->ok && ld->kind < 0) { ld->ok = false; ld->why = "ldmatrix has no mma consumer"; }
  }
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto& lp : lds) {
      LdRec* ld = lp.get();
      if (!ld->ok) continue;
      for (MmaRec* m : consumers[ld]) if (!m->ok) { ld->ok = false; ld->why = "consumer mma not recovered: " + m->why; changed = true; break; }
    }
    for (auto& mp : mmas) {
      MmaRec* m = mp.get();
      if (!m->ok) continue;
      for (LdRec* ld : m->aSrc) if (!ld->ok) { m->fail("A ldmatrix not recoverable: " + ld->why); changed = true; }
      for (LdRec* ld : m->bSrc) if (!ld->ok) { m->fail("B ldmatrix not recoverable: " + ld->why); changed = true; }
    }
    // an op with any failed mma fails entirely
    for (auto& op : ops) {
      bool bad = false; std::string why;
      for (auto& a : op->m) for (auto& b : a) for (MmaRec* m : b) if (m && !m->ok) { bad = true; why = m->why; }
      if (bad) for (auto& a : op->m) for (auto& b : a) for (MmaRec* m : b) if (m && m->ok) { m->fail(why); changed = true; }
    }
  }
  std::vector<std::unique_ptr<FusedOp>> kept;
  for (auto& op : ops) { bool ok = true; for (auto& a : op->m) for (auto& b : a) for (MmaRec* m : b) if (m && !m->ok) ok = false; if (ok) kept.push_back(std::move(op)); }
  ops.swap(kept);
  // Register-built blocks: the slot conversion is placed at the insertion point (last mma) of one of the fused ops
  // consuming the block - every word of the block is an operand of an mma of that op, so it is defined by then (a
  // B pair's partner words are built between the two mmas, after the first) - and that op must dominate the others.
  // Ops that cannot be served are dropped (the exact path keeps them); the loop repeats since dropping an op can
  // change which ops remain for a block.
  for (bool changed = true; changed;) {
    changed = false;
    for (auto& op : ops) {
      // the last-level mmas share one block (buildGrids); the latest of them is the op's insertion point
      op->lastMma = nullptr;
      const int LV = (int)op->m[0][0].size();
      for (auto& a : op->m) for (auto& b : a) { MmaRec* m = b[LV - 1]; if (m && (!op->lastMma || op->lastMma->CI->comesBefore(m->CI))) op->lastMma = m; }
    }
    std::map<LdRec*, std::vector<FusedOp*>> users;
    for (auto& op : ops) for (auto& a : op->m) for (auto& b : a) for (MmaRec* m : b) {
      if (!m) continue;
      for (LdRec* ld : m->aSrc) if (ld->reg) { auto& u = users[ld]; if (std::find(u.begin(), u.end(), op.get()) == u.end()) u.push_back(op.get()); }
      for (LdRec* ld : m->bSrc) if (ld->reg) { auto& u = users[ld]; if (std::find(u.begin(), u.end(), op.get()) == u.end()) u.push_back(op.get()); }
    }
    std::set<FusedOp*> drop;
    for (auto& [ld, us] : users) {
      ld->firstUse = nullptr;
      for (FusedOp* c : us) {
        bool all = true;
        for (FusedOp* o : us) if (o != c && !DT.dominates(c->lastMma->CI, o->lastMma->CI)) { all = false; break; }
        if (all) { ld->firstUse = c->lastMma; break; }
      }
      if (!ld->firstUse) for (FusedOp* o : us) drop.insert(o);
    }
    if (drop.empty()) break;
    changed = true;
    std::vector<std::unique_ptr<FusedOp>> keep;
    for (auto& op : ops) {
      if (!drop.count(op.get())) { keep.push_back(std::move(op)); continue; }
      for (auto& a : op->m) for (auto& b : a) for (MmaRec* m : b) if (m) m->fail("register-built block is shared by mma batches none of which dominates the others");
    }
    ops.swap(keep);
  }
}

// ---------------------------------------------------------------- accumulator typing

void KernelRecovery::typeAccumulators() {
  Type* f32 = Type::getFloatTy(F.getContext());
  // C/D lists per op
  for (auto& op : ops) {
    const int MTL = (int)op->m.size(), NTL = (int)op->m[0].size(), LV = (int)op->m[0][0].size();
    for (int mt = 0; mt < MTL; mt++) for (int nt = 0; nt < NTL; nt++) {
      MmaRec* first = op->m[mt][nt][0]; MmaRec* last = op->m[mt][nt][LV - 1];
      if (!first) {
        // padding positions of a solo B block: C enters as zero (nothing to carry) unless the bundle is slot-typed
        // (filled in below); D gets a stand-in so a slot-typed consumer can take the fused call's word
        for (int r = 0; r < 4; r++) {
          auto* ph = new FreezeInst(PoisonValue::get(f32), "tp.pad", op->lastMma->CI->getIterator());
          padVals.push_back(ph);
          op->cList.push_back(nullptr); op->dList.push_back(ph);
        }
        continue;
      }
      for (int r = 0; r < 4; r++) { op->cList.push_back(first->c[r]); op->dList.push_back(last->d[r]); }
      if (!op->lastMma || op->lastMma->CI->comesBefore(last->CI)) op->lastMma = last;
    }
    auto g = std::make_unique<SlotGroup>(); g->id = (int)groups.size(); g->op = op.get(); g->M = op->M; g->N = op->N;
    for (size_t p = 0; p < op->dList.size(); p++) { g->vals.push_back(op->dList[p]); if (op->dList[p]) slotOf[op->dList[p]] = {g.get(), (int)p}; }
    groups.push_back(std::move(g));
  }
  // a value list is slot-typed from group X iff every position p is X.vals[p] (padding positions are unconstrained)
  auto listSource = [&](const std::vector<Value*>& vals) -> SlotGroup* {
    SlotGroup* X = nullptr;
    for (size_t p = 0; p < vals.size(); p++) {
      if (!vals[p]) continue;
      auto it = slotOf.find(vals[p]);
      if (it == slotOf.end() || it->second.second != (int)p) return nullptr;
      if (!X) X = it->second.first; else if (X != it->second.first) return nullptr;
    }
    return X && X->valid ? X : nullptr;
  };
  auto allZero = [&](const std::vector<Value*>& vals) { for (Value* v : vals) { if (!v) continue; auto* C = dyn_cast<ConstantFP>(v); if (!C || !C->isZero()) return false; } return true; };
  auto sameList = [](const std::vector<Value*>& have, const std::vector<Value*>& want) {
    if (have.size() != want.size()) return false;
    for (size_t p = 0; p < have.size(); p++) if (want[p] && want[p] != have[p]) return false;
    return true;
  };
  // candidate phi groups from C lists (and recursively from their incoming lists). A list with padding positions gets
  // a phi per such position whose incoming values are the sources' values at that position.
  std::vector<PHINode*> padPhis;
  std::function<SlotGroup*(const std::vector<Value*>&, int, int)> phiGroup = [&](const std::vector<Value*>& vals, int M_, int N_) -> SlotGroup* {
    auto* PN0 = dyn_cast<PHINode>(vals[0]);
    if (!PN0) return nullptr;
    if (slotOf.count(PN0)) { auto [g, p] = slotOf[PN0]; return (g->isPhi && p == 0 && sameList(g->vals, vals)) ? g : nullptr; }
    std::set<Value*> uniq; size_t live = 0;
    for (Value* v : vals) if (v) { uniq.insert(v); live++; }
    if (uniq.size() != live) return nullptr;
    for (Value* v : vals) { if (!v) continue; auto* PN = dyn_cast<PHINode>(v); if (!PN || PN->getParent() != PN0->getParent() || slotOf.count(PN)) return nullptr; }
    auto g = std::make_unique<SlotGroup>(); g->id = (int)groups.size(); g->isPhi = true; g->phiBlock = PN0->getParent(); g->vals = vals; g->M = M_; g->N = N_;
    SlotGroup* gp = g.get();
    std::vector<int> padPos;
    for (size_t p = 0; p < vals.size(); p++) {
      if (vals[p]) continue;
      auto* pn = PHINode::Create(f32, PN0->getNumIncomingValues(), "tp.pad", PN0->getIterator());
      gp->vals[p] = pn; padPhis.push_back(pn); padPos.push_back((int)p);
    }
    for (size_t p = 0; p < vals.size(); p++) slotOf[gp->vals[p]] = {gp, (int)p};
    groups.push_back(std::move(g));
    for (unsigned e = 0; e < PN0->getNumIncomingValues(); e++) {
      BasicBlock* pred = PN0->getIncomingBlock(e);
      std::vector<Value*> in;
      for (Value* v : vals) in.push_back(v ? cast<PHINode>(v)->getIncomingValueForBlock(pred) : nullptr);
      SlotGroup* src = nullptr;
      if (isa<PHINode>(in[0])) src = phiGroup(in, M_, N_);
      else if (!allZero(in)) src = listSource(in);
      for (int p : padPos) {
        Value* v = allZero(in) ? ConstantFP::get(f32, 0.0) : src ? src->vals[p] : PoisonValue::get(f32);
        cast<PHINode>(gp->vals[p])->addIncoming(v, pred);
      }
    }
    return gp;
  };
  for (auto& op : ops) phiGroup(op->cList, op->M, op->N);
  bool changed = true;
  while (changed) {
    changed = false;
    for (auto& g : groups) {
      if (!g->isPhi || !g->valid) continue;
      auto* PN0 = cast<PHINode>(g->vals[0]);
      for (unsigned e = 0; e < PN0->getNumIncomingValues() && g->valid; e++) {
        std::vector<Value*> in;
        for (Value* v : g->vals) in.push_back(cast<PHINode>(v)->getIncomingValueForBlock(PN0->getIncomingBlock(e)));
        if (allZero(in)) continue;
        if (!listSource(in)) { g->valid = false; changed = true; }
      }
    }
  }
  for (auto& g : groups) if (!g->valid) for (Value* v : g->vals) if (v) slotOf.erase(v);
  // pad phis of invalid groups are not slots of anything: drop them (a valid group never takes a value from an
  // invalid one, so their only users are pad phis of invalid groups)
  std::set<PHINode*> keep;
  for (auto& g : groups) if (g->isPhi && g->valid) for (Value* v : g->vals) if (auto* pn = dyn_cast<PHINode>(v)) keep.insert(pn);
  for (PHINode* pn : padPhis) if (!keep.count(pn)) { pn->replaceAllUsesWith(PoisonValue::get(f32)); pn->eraseFromParent(); }
  for (auto& op : ops) {
    op->cIsSlots = allZero(op->cList) || listSource(op->cList) != nullptr;
    // a slot-typed C bundle carries its padding slots too: the source's value at the position (zero stays zero)
    if (SlotGroup* X = op->cIsSlots ? listSource(op->cList) : nullptr)
      for (size_t p = 0; p < op->cList.size(); p++) if (!op->cList[p]) op->cList[p] = X->vals[p];
  }
}

// ---------------------------------------------------------------- rewrite

static FunctionCallee declare(Module& M, const std::string& name, Type* ret, ArrayRef<Type*> args) {
  FunctionType* FT = FunctionType::get(ret, args, false);
  FunctionCallee FC = M.getOrInsertFunction(name, FT);
  if (auto* Fn = dyn_cast<Function>(FC.getCallee())) { Fn->addFnAttr(Attribute::Convergent); Fn->addFnAttr(Attribute::NoUnwind); }
  return FC;
}

void KernelRecovery::rewrite() {
  Module& M = *F.getParent();
  LLVMContext& C = F.getContext();
  Type* i32 = Type::getInt32Ty(C); Type* f32 = Type::getFloatTy(C);
  StructType* ld4 = StructType::get(C, {i32, i32, i32, i32});
  auto cst = [&](int v) { return ConstantInt::get(i32, v); };
  auto floatStruct = [&](int n) { std::vector<Type*> t(n, f32); return StructType::get(C, t); };

  // 1. ldmatrix -> __mvcc_tp_ld(addr, kind, trans, blockmap) : {i32 x 4} words in slot order
  // in discovery order (the ops', then their sources'), not by address: the order decides where the fills are
  // inserted and so the emitted program's order, which must not vary from run to run
  std::vector<LdRec*> usedLds;
  { std::set<LdRec*> seen; auto add = [&](LdRec* l) { if (seen.insert(l).second) usedLds.push_back(l); };
    for (auto& op : ops) for (auto& a : op->m) for (auto& b : a) for (MmaRec* m : b) if (m) { for (LdRec* l : m->aSrc) add(l); for (LdRec* l : m->bSrc) add(l); } }
  FunctionCallee ldFn = declare(M, "__mvcc_tp_ld", ld4, {i32, i32, i32, i32});
  FunctionCallee f2s = declare(M, "__mvcc_tp_frag2slot", ld4, {i32, i32, i32, i32, i32});
  for (LdRec* ld : usedLds) {
    if (!ld->reg) continue;
    IRBuilder<> B(ld->firstUse->CI);
    CallInst* nc = B.CreateCall(f2s, {cst(ld->kind), ld->regWords[0], ld->regWords[1], ld->regWords[2], ld->regWords[3]});
    nc->setDebugLoc(ld->firstUse->CI->getDebugLoc());
    for (int i = 0; i < 4; i++) ld->slotWord[i] = B.CreateExtractValue(nc, i);
    note(std::string("register-built ") + (ld->kind == 0 ? "A" : "B") + " block moved into slot order at its first mma (8 simd shuffles per 16x16 block)");
  }
  for (LdRec* ld : usedLds) {
    if (ld->reg) continue;
    unsigned bm = 0;
    for (int r = 0; r < 4; r++) bm |= (unsigned)ld->matrixOfBlock[r] << (2 * r);
    if (ld->composite()) {
      // two x2 -> one x4 fill at the first instruction: lanes 0-15 give the first instruction's rows (matrices 0, 1),
      // lanes 16-31 the second's (matrices 2, 3). The second address is taken as is when lane l and l-16 provably
      // compute the same value (laneHalfDiffBit), else fetched from lane l-16 with a shuffle.
      LdRec* p0 = ld->parts[0]; LdRec* p1 = ld->parts[1];
      IRBuilder<> B(p0->CI);
      B.SetCurrentDebugLocation(p0->CI->getDebugLoc());
      Value* a0 = p0->CI->getArgOperand(0);
      std::map<Value*, int> memo;
      const bool invariant = laneHalfDiffBit(p1->CI->getArgOperand(0), memo, 0) == -1;
      std::map<Value*, Value*> mat;
      Value* a1 = materializeAt(p1->CI->getArgOperand(0), p0->CI, mat, 0);
      if (!a1) { warn("internal: ldmatrix.x2 pair whose second address was not materializable (fuseX2 admitted it)"); continue; }
      Value* lane = B.CreateCall(Intrinsic::getOrInsertDeclaration(&M, Intrinsic::nvvm_read_ptx_sreg_laneid));
      if (!invariant) {
        if (!a1->getType()->isIntegerTy(32)) { warn("internal: ldmatrix.x2 pair with a non-32-bit address"); continue; }
        Function* shfl = Intrinsic::getOrInsertDeclaration(&M, Intrinsic::nvvm_shfl_sync_idx_i32);
        a1 = B.CreateCall(shfl, {cst(-1), a1, B.CreateAnd(lane, cst(15)), cst(0x1f)});
      }
      Value* lo = B.CreateICmpULT(lane, cst(16));
      // select the pointers behind the two addresses when both are visible (keeps threadgroup provenance for the
      // emitter and the level-4 address analyses); the truncated form otherwise
      auto behind = [](Value* v) -> Value* {
        for (int hops = 0; v && hops < 8; hops++) {
          if (auto* T = dyn_cast<TruncInst>(v)) { v = T->getOperand(0); continue; }
          if (auto* Z = dyn_cast<ZExtInst>(v)) { v = Z->getOperand(0); continue; }
          if (auto* P = dyn_cast<PtrToIntInst>(v)) { v = P->getOperand(0); continue; }
          break;
        }
        return v && v->getType()->isPointerTy() ? v : nullptr;
      };
      Value* q0 = behind(a0); Value* q1 = invariant ? behind(a1) : nullptr;
      Value* addr;
      if (q0 && q1 && q0->getType() == q1->getType()) addr = B.CreateZExtOrTrunc(B.CreatePtrToInt(B.CreateSelect(lo, q0, q1), Type::getInt64Ty(C)), a0->getType());
      else addr = B.CreateSelect(lo, a0, a1);
      CallInst* nc = B.CreateCall(ldFn, {addr, cst(ld->kind), cst(ld->trans), cst((int)bm)});
      nc->setDebugLoc(p0->CI->getDebugLoc());
      for (int i = 0; i < 4; i++) ld->slotWord[i] = B.CreateExtractValue(nc, i);
      // every field extract of either part now reads the fused fill (field f of part k is word 2k+f); the phi webs
      // and the mma operand tuples keep flowing through the same values
      for (int k = 0; k < 2; k++) {
        LdRec* p = ld->parts[k];
        std::vector<ExtractValueInst*> evs;
        for (User* U : p->CI->users()) if (auto* EV = dyn_cast<ExtractValueInst>(U)) evs.push_back(EV);
        for (ExtractValueInst* EV : evs) {
          if (EV->getNumIndices() != 1 || EV->getIndices()[0] > 1) { warn("internal: ldmatrix.x2 field extract with an unexpected index"); continue; }
          Value* NE = ld->slotWord[2 * k + EV->getIndices()[0]];
          for (auto& mp : mmas) { for (int r = 0; r < 4; r++) if (mp->a[r] == EV) mp->a[r] = NE; for (int r = 0; r < 2; r++) if (mp->b[r] == EV) mp->b[r] = NE; }
          for (int f = 0; f < 2; f++) if (p->field[f] == EV) p->field[f] = nullptr;
          EV->replaceAllUsesWith(NE);
          EV->eraseFromParent();
        }
        if (!p->CI->use_empty()) { warn("internal: ldmatrix.x2 with a non-extract user survives the fusion"); continue; }
        p->CI->eraseFromParent(); p->CI = nullptr;
      }
      note(std::string("two ldmatrix.x2 fused into one 16x16 ") + (ld->kind == 0 ? "A" : "B") + " block fill" + (invariant ? " (lanes 16-31 take the second instruction's address)" : " (one shuffle brings the second address to lanes 16-31)"));
      continue;
    }
    IRBuilder<> B(ld->CI);
    CallInst* nc = B.CreateCall(ldFn, {ld->CI->getArgOperand(0), cst(ld->kind), cst(ld->trans), cst((int)bm)});
    nc->setDebugLoc(ld->CI->getDebugLoc());
    ld->CI->replaceAllUsesWith(nc);
    ld->CI->eraseFromParent();
  }

  // 2. fused ops (C operands are placeholders = the old values; patched in step 3)
  auto cArgBase = [&](FusedOp* op) { return 6 + (op->M * op->K + op->K * op->N) / 64; };
  std::map<FusedOp*, CallInst*> ptx2accOf;
  for (auto& op : ops) {
    const int MTL = (int)op->m.size(), NTL = (int)op->m[0].size(), LV = (int)op->m[0][0].size();
    const int cap = op->cap();
    const int nA = op->M * op->K / 64, nB = op->K * op->N / 64;  // 32-bit words per lane
    IRBuilder<> B(op->lastMma->CI);
    std::vector<Value*> cVals = op->cList;
    for (Value*& v : cVals) if (!v) v = ConstantFP::get(f32, 0.0);   // padding positions of a solo B block
    if (!op->cIsSlots) {
      std::vector<Type*> at(2, i32); for (int i = 0; i < cap; i++) at.push_back(f32);
      FunctionCallee cv = declare(M, "__mvcc_tp_ptx2acc_" + std::to_string(cap), floatStruct(cap), at);
      std::vector<Value*> args = {cst(op->M), cst(op->N)}; args.insert(args.end(), cVals.begin(), cVals.end());
      CallInst* cc = B.CreateCall(cv, args);
      ptx2accOf[op.get()] = cc;
      for (int p = 0; p < cap; p++) cVals[p] = B.CreateExtractValue(cc, p);
    }
    // operand words: sub-blocks (16x16 per ldmatrix) ordered by the slot-bit assignment of the fused tile
    auto wordsOf = [&](MmaRec* m, bool isA) {
      std::vector<Value*> w(4, nullptr);
      // register-built blocks carry their slot words directly; an x4 fill (or the fused fill of an x2 pair) replaced
      // the ldmatrix, so its field extracts (the mma operands, possibly through phis) are the slot words, indexed by
      // field
      if (isA) {
        if (m->aSrc[0]->reg) { for (int r = 0; r < 4; r++) w[r] = m->aSrc[0]->slotWord[r]; }
        else for (int r = 0; r < 4; r++) w[m->aPerm[r]] = m->a[r];
      } else {
        if (m->bSrc[0]->reg) { for (int r = 0; r < 4; r++) w[r] = m->bSrc[0]->slotWord[r]; }
        else {
          MmaRec* e = m->nh == 0 ? m : m->pairWith; MmaRec* o = e->pairWith;
          w[e->bPerm[0]] = e->b[0]; w[e->bPerm[1]] = e->b[1]; w[o->bPerm[0]] = o->b[0]; w[o->bPerm[1]] = o->b[1];
        }
      }
      return w;
    };
    std::vector<std::vector<Value*>> subA(MTL * LV), subB((NTL / 2) * LV);
    // A (M x K): C==32 -> kb is slot bit 3 (fastest), mb slot bit 4; else mb is slot bit 3
    for (int mb = 0; mb < MTL; mb++) for (int kb = 0; kb < LV; kb++) subA[op->K == 32 ? kb + LV * mb : mb] = wordsOf(op->m[mb][0][kb], true);
    // B (K x N): N==32 -> nb is slot bit 3 (fastest), kb slot bit 4; else kb is slot bit 3
    for (int nb = 0; nb < NTL / 2; nb++) for (int kb = 0; kb < LV; kb++) subB[op->N == 32 ? nb + (NTL / 2) * kb : kb] = wordsOf(op->m[0][2 * nb][kb], false);
    std::vector<Value*> aWords, bWords;
    for (auto& s : subA) aWords.insert(aWords.end(), s.begin(), s.end());
    for (auto& s : subB) bWords.insert(bWords.end(), s.begin(), s.end());
    if ((int)aWords.size() != nA || (int)bWords.size() != nB) { warn("internal: operand word count mismatch"); continue; }
    std::vector<Type*> at(6, i32); for (int i = 0; i < nA + nB; i++) at.push_back(i32); for (int i = 0; i < cap; i++) at.push_back(f32);
    std::string nm = "__mvcc_tp_mma_" + std::to_string(op->M) + "x" + std::to_string(op->N) + "x" + std::to_string(op->K);
    FunctionCallee mf = declare(M, nm, floatStruct(cap), at);
    std::vector<Value*> args = {cst(op->M), cst(op->N), cst(op->K), cst(op->type), cst(op->tl), cst(op->tr)};
    args.insert(args.end(), aWords.begin(), aWords.end()); args.insert(args.end(), bWords.begin(), bWords.end()); args.insert(args.end(), cVals.begin(), cVals.end());
    op->newCall = B.CreateCall(mf, args);
    op->newCall->setDebugLoc(op->lastMma->CI->getDebugLoc());
    for (int p = 0; p < cap; p++) op->newD.push_back(B.CreateExtractValue(op->newCall, p));
    TpDescriptor d; d.M = op->M; d.N = op->N; d.K = op->K; d.tl = op->tl; d.tr = op->tr; d.type = op->type;
    res.descriptors.insert(d);
    int fused = 0; for (auto& a : op->m) for (auto& b : a) for (MmaRec* m : b) fused += m != nullptr;
    note("fused " + std::to_string(fused) + " mma.sync.m16n8k16 into matmul2d " + d.tag() + (op->cIsSlots ? "" : " (accumulator enters in fragment layout: PTX->slot conversion inserted)"));
  }

  // 3. wire slot-typed values: internal consumers take the slot values, everything else a PTX-layout conversion
  // A conversion is placed as late as possible: a use outside the loop(s) that carry the tile gets it in the exit
  // block that leads to that use, so the K loop never converts per iteration (the conversion goes through
  // threadgroup scratch; in the loop it would cost more than the matmul).
  LoopInfo LI(DT);
  std::map<std::pair<SlotGroup*, BasicBlock*>, std::vector<Value*>> convOf;
  auto ensureConv = [&](SlotGroup* g, Use* U, int p) -> std::vector<Value*>& {
    BasicBlock* defBB = g->isPhi ? g->phiBlock : cast<Instruction>(g->op->newD.back())->getParent();
    BasicBlock* useBB = cast<Instruction>(U->getUser())->getParent();
    PHINode* collapse = nullptr;  // single-entry phi at the exit: it is the value; the conversion replaces it
    if (auto* PN = dyn_cast<PHINode>(U->getUser())) {
      useBB = PN->getIncomingBlock(*U);
      // a phi outside the loop taking the tile from the latch: the conversion must run once, after the loop,
      // not in the latch. A single-entry phi is replaced by the conversion at its block; a merge with other
      // paths gets the exit edge split so the conversion has a block of its own.
      const Loop* L = LI.getLoopFor(defBB);
      if (L && L->contains(useBB) && !L->contains(PN->getParent())) {
        if (PN->getNumIncomingValues() == 1) { collapse = PN; useBB = PN->getParent(); }
        else useBB = SplitEdge(useBB, PN->getParent(), &DT, &LI);
      }
    }
    BasicBlock* atBB = nullptr;  // nullptr = right after the definition
    const Loop* outermost = nullptr;
    for (const Loop* L = LI.getLoopFor(defBB); L && !L->contains(useBB); L = L->getParentLoop()) outermost = L;
    if (outermost) {
      SmallVector<BasicBlock*, 4> exits; outermost->getExitBlocks(exits);
      for (BasicBlock* E : exits) if (DT.dominates(defBB, E) && DT.dominates(E, useBB)) { if (atBB && atBB != E) { atBB = nullptr; break; } atBB = E; }
    }
    auto finish = [&](std::vector<Value*>& out) -> std::vector<Value*>& {
      if (collapse) { collapse->replaceAllUsesWith(out[p]); U->set(PoisonValue::get(collapse->getType())); }
      return out;
    };
    auto it = convOf.find({g, atBB});
    if (it != convOf.end()) return finish(it->second);
    const int cap = (int)g->vals.size();
    std::vector<Value*> vals(cap);
    Instruction* ip;
    if (g->isPhi) { ip = &*g->phiBlock->getFirstNonPHIIt(); for (int p = 0; p < cap; p++) vals[p] = g->vals[p]; }
    else { ip = cast<Instruction>(g->op->newD.back())->getNextNode(); for (int p = 0; p < cap; p++) vals[p] = g->op->newD[p]; }
    if (atBB) ip = &*atBB->getFirstNonPHIIt();
    for (Value*& v : vals) if (!v) v = ConstantFP::get(f32, 0.0);   // padding positions of a solo B block
    IRBuilder<> B(ip);
    std::vector<Type*> at(2, i32); for (int i = 0; i < cap; i++) at.push_back(f32);
    FunctionCallee cv = declare(M, "__mvcc_tp_acc2ptx_" + std::to_string(cap), floatStruct(cap), at);
    std::vector<Value*> args = {cst(g->M), cst(g->N)}; args.insert(args.end(), vals.begin(), vals.end());
    CallInst* cc = B.CreateCall(cv, args);
    std::vector<Value*> out(cap);
    for (int p = 0; p < cap; p++) out[p] = B.CreateExtractValue(cc, p);
    note(std::string("inserted slot->PTX accumulator conversion at a tile boundary (") + (g->isPhi ? "loop-carried tile read outside the mma chain" : "tile result read by the epilogue") + (atBB ? ", at the loop exit)" : ")"));
    return finish(convOf[{g, atBB}] = out);
  };
  std::map<CallInst*, FusedOp*> opOfCall;
  for (auto& op : ops) if (op->newCall) opOfCall[op->newCall] = op.get();
  for (auto& g : groups) {
    if (!g->valid) continue;
    if (!g->isPhi && !g->op->newCall) continue;
    const int cap = (int)g->vals.size();
    for (int p = 0; p < cap; p++) {
      Value* oldV = g->isPhi ? g->vals[p] : g->op->dList[p];
      Value* slotV = g->isPhi ? g->vals[p] : g->op->newD[p];
      if (!oldV) continue;
      std::vector<Use*> uses;
      for (Use& U : oldV->uses()) uses.push_back(&U);
      for (Use* U : uses) {
        User* usr = U->getUser();
        if (auto* CI = dyn_cast<CallInst>(usr)) {
          if (mmaByCI.count(CI)) continue;                                              // old mma, about to die
          // a conversion this group already placed (a phi group's slot values are its own operands: the call is a
          // use of every later slot, and rewiring it to its own extracts made a self-referencing call)
          if (CI->getCalledFunction() && CI->getCalledFunction()->getName().starts_with("__mvcc_tp_acc2ptx_")) continue;
          auto oi = opOfCall.find(CI);
          if (oi != opOfCall.end() && oi->second->cIsSlots && (int)U->getOperandNo() == cArgBase(oi->second) + p) { U->set(slotV); continue; }
        } else if (auto* PN = dyn_cast<PHINode>(usr)) {
          auto it = slotOf.find(PN);
          if (it != slotOf.end() && it->second.first->valid && it->second.second == p) { U->set(slotV); continue; }
        } else if (isa<ExtractValueInst>(usr)) {
          continue;
        }
        std::vector<Value*>& conv = ensureConv(g.get(), U, p);
        if (!isa<PoisonValue>(U->get())) U->set(conv[p]);  // (a collapsed exit phi already took it)
      }
    }
  }

  // 4. delete old mmas (all levels) and their extracts
  std::vector<MmaRec*> dead;
  for (auto& op : ops) if (op->newCall) for (auto& a : op->m) for (auto& b : a) for (MmaRec* m : b) if (m) dead.push_back(m);
  for (MmaRec* m : dead) for (int r = 0; r < 4; r++) if (m->d[r]) { if (!m->d[r]->use_empty()) m->d[r]->replaceAllUsesWith(PoisonValue::get(f32)); m->d[r]->eraseFromParent(); }
  for (MmaRec* m : dead) m->CI->eraseFromParent();
  // the padding stand-ins were replaced by the fused calls' extracts (step 3); an op that was not rewritten leaves
  // its stand-ins poison, which is what its padding slots are
  for (Instruction* ph : padVals) { if (!ph->use_empty()) ph->replaceAllUsesWith(PoisonValue::get(f32)); ph->eraseFromParent(); }
}

// mma.sync.m16n8k32 with e4m3/e5m2 inputs -> widen every register to two half pairs and issue two m16n8k16 f16 mmas.
// PTX k32 register r holds 4 consecutive k of one row/column (k = 4t+e); the k16 slice j takes registers (a_{2j},
// a_{2j+1}; b_j) with the lo pair as k' = 2t+e (e<2) and the hi pair as k' = 8+2t+(e-2): one bijection of the
// reduction index applied to A and B alike, so the sum is over the same products. Exact by construction; it is also
// how the exact lowering computes the instruction, so declining afterwards costs nothing.
void KernelRecovery::splitFp8Mmas() {
  Module& M = *F.getParent();
  LLVMContext& C = F.getContext();
  Type* i32 = Type::getInt32Ty(C); Type* f32 = Type::getFloatTy(C);
  std::vector<std::pair<CallInst*, std::string>> todo;
  for (BasicBlock& BB : F)
    for (Instruction& I : BB) {
      auto* CI = dyn_cast<CallInst>(&I);
      if (!CI || !CI->isInlineAsm()) continue;
      auto* IA = cast<InlineAsm>(CI->getCalledOperand());
      std::vector<PtxInstr> ins; std::string err;
      if (!parsePtxAsm(std::string(IA->getAsmString()), ins, err) || ins.size() != 1 || ins[0].mnemonic != "mma") continue;
      const PtxInstr& in = ins[0];
      if (!in.hasMod("m16n8k32") || !in.hasMod("row") || !in.hasMod("col") || in.mods.size() < 4) continue;
      std::string dt = in.mods[in.mods.size() - 4], at = in.mods[in.mods.size() - 3], bt = in.mods[in.mods.size() - 2], ct = in.mods[in.mods.size() - 1];
      if (dt != "f32" || ct != "f32" || at != bt || (at != "e4m3" && at != "e5m2") || CI->arg_size() != 10) continue;
      auto* ST = dyn_cast<StructType>(CI->getType());
      if (!ST || ST->getNumElements() != 4) continue;
      bool ok = true;
      for (unsigned i = 0; i < 6; i++) if (!CI->getArgOperand(i)->getType()->isIntegerTy(32)) ok = false;
      for (unsigned i = 6; i < 10; i++) if (!CI->getArgOperand(i)->getType()->isFloatTy()) ok = false;
      for (unsigned i = 0; i < 4; i++) if (!ST->getElementType(i)->isFloatTy()) ok = false;
      if (ok) todo.push_back({CI, at});
    }
  if (todo.empty()) return;
  StructType* pair = StructType::get(C, {i32, i32});
  StructType* d4 = StructType::get(C, {f32, f32, f32, f32});
  FunctionType* FT = FunctionType::get(d4, {i32, i32, i32, i32, i32, i32, f32, f32, f32, f32}, false);
  InlineAsm* k16 = InlineAsm::get(FT, "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {$0,$1,$2,$3}, {$4,$5,$6,$7}, {$8,$9}, {$10,$11,$12,$13};",
                                  "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f", false);
  // one widening per source word, right after the word's definition, so mmas sharing a register share the widened
  // tuple too (operand identity is what the grid analysis keys on)
  std::map<std::pair<Value*, std::string>, std::pair<Value*, Value*>> widened;
  auto widenOf = [&](Value* v, const std::string& fmt) {
    auto it = widened.find({v, fmt});
    if (it != widened.end()) return it->second;
    Instruction* ip;
    if (auto* I = dyn_cast<Instruction>(v)) { ip = isa<PHINode>(I) ? &*I->getParent()->getFirstNonPHIIt() : I->getNextNode(); }
    else ip = &*F.getEntryBlock().getFirstNonPHIIt();
    IRBuilder<> B(ip);
    FunctionCallee widen = declare(M, "__mvcc_tp_widen_" + fmt, pair, {i32});
    CallInst* wc = B.CreateCall(widen, {v});
    std::pair<Value*, Value*> r = {B.CreateExtractValue(wc, 0), B.CreateExtractValue(wc, 1)};
    return widened[{v, fmt}] = r;
  };
  for (auto& [CI, fmt] : todo) {
    IRBuilder<> B(CI);
    B.SetCurrentDebugLocation(CI->getDebugLoc());
    Value* w[6][2];
    for (int i = 0; i < 6; i++) { auto r = widenOf(CI->getArgOperand(i), fmt); w[i][0] = r.first; w[i][1] = r.second; }
    Value* c[4]; for (int i = 0; i < 4; i++) c[i] = CI->getArgOperand(6 + i);
    CallInst* last = nullptr;
    for (int j = 0; j < 2; j++) {
      // slice j: A words (a_{2j}.lo, a_{2j+1}.lo, a_{2j}.hi, a_{2j+1}.hi), B words (b_j.lo, b_j.hi)
      std::vector<Value*> args = {w[2 * j][0], w[2 * j + 1][0], w[2 * j][1], w[2 * j + 1][1], w[4 + j][0], w[4 + j][1], c[0], c[1], c[2], c[3]};
      last = B.CreateCall(k16, args);
      last->addFnAttr(Attribute::Convergent);
      if (j == 0) for (int i = 0; i < 4; i++) c[i] = B.CreateExtractValue(last, i);
    }
    CI->replaceAllUsesWith(last);
    CI->eraseFromParent();
  }
  note("split " + std::to_string(todo.size()) + " mma.sync.m16n8k32." + todo[0].second + " into half-widening + 2 x m16n8k16.f16 each (one k-permutation for A and B)");
}

bool KernelRecovery::run() {
  kname = demangle(F.getName().str());
  splitFp8Mmas();
  if (!collect()) return false;
  res.kernelsWithMma++;
  analyzeChains();
  analyzeOperands();
  pairMmas();
  buildGrids();
  validateFixpoint();
  if (ops.empty()) {
    std::set<std::string> reasons;
    for (auto& m : mmas) if (!m->ok) reasons.insert(m->why);
    std::string r; for (auto& s : reasons) { if (!r.empty()) r += "; "; r += s; }
    warn("mma.sync recognized but not recovered (exact lowering kept): " + r);
    return false;
  }
  typeAccumulators();
  // check(P,S) before any IR: rewrite() is emit(P,S) for this contraction.
  {
    stir::Program GP = stir::direct::gemm(kname);
    sched::Schedule Sg = sched::sourceSchedule(GP);
    if (!Sg.stages.empty()) Sg.stages[0].engine = sched::Engine::TensorOp;
    legal::Result Lg = legal::check(GP, Sg);
    if (!Lg.ok) {
      warn("check+emit refused " + Lg.first() + " (exact twin)");
      return false;
    }
    note("check+emit: tensor-op " + Lg.first() + " " + sched::stageShape(Sg));
  }
  // exact twin before touching anything
  ValueToValueMapTy VMap;
  Function* twin = CloneFunction(&F, VMap);
  twin->setName(F.getName() + "__mvcc_exact");
  res.exactVariants.push_back({F.getName().str(), twin->getName().str()});
  rewrite();
  unsigned nm = 0; for (auto& m : mmas) if (m->ok) nm++;
  unsigned left = 0; for (auto& m : mmas) if (!m->ok) left++;
  note(std::to_string(nm) + " mma.sync recovered into " + std::to_string(ops.size()) + " matmul2d op(s)" + (left ? ", " + std::to_string(left) + " left on the exact path" : ""));
  if (left) { std::set<std::string> reasons; for (auto& m : mmas) if (!m->ok) reasons.insert(m->why); std::string r; for (auto& s : reasons) { if (!r.empty()) r += "; "; r += s; } warn(std::to_string(left) + " mma.sync not recovered: " + r); }
  res.mmaKernelsRecovered++;
  // Storage / pipeline lowering is emit(P,S) (emit::runModule).
  return true;
}

}  // namespace

// Semantic iteration graph summary (sig.h): MVCC_SIG=1 prints, per kernel, the loop classification, the algebra of
// every loop-carried value of the traversal loops, the effect counts and the discovered coordinates (coords.h).
// MVCC_SIG=2 adds the init/next expressions. MVCC_SIG_STAGE=source runs it on the kernels as written (before
// recovery) instead of as they will be emitted; MVCC_SIG_ONLY=1 skips the recovery passes altogether (analysis
// only; the module is emitted exactly). MVCC_SIG_DUMP=<file> appends the lines.
static void runSig(Module& M, TensorRecoveryResult& out) {
  const char* e = getenv("MVCC_SIG");
  if (!e || e[0] == '0') return;
  sig::SigOptions so; so.dumpExprs = e[0] == '2';
  std::vector<std::string> all;
  for (Function& F : M) {
    if (F.isDeclaration() || !isKernelFn(F)) continue;
    std::vector<std::string> lines;
    sig::analyzeKernel(F, kernelMaxThreads(F), so, lines);
    const std::string kname = llvm::demangle(F.getName().str());
    for (size_t i = 0; i < lines.size(); i++) { std::string l = (i == 0 ? kname + ": " : "") + lines[i]; out.notes.push_back(l); all.push_back(l); }
    // append after each kernel so a process wall-kill still leaves a named SIG / refusal
    if (const char* d = getenv("MVCC_SIG_DUMP")) {
      std::error_code EC; raw_fd_ostream os(d, EC, sys::fs::OF_Append | sys::fs::OF_Text);
      if (!EC) {
        for (size_t i = 0; i < lines.size(); i++) os << (i == 0 ? kname + ": " : "") << lines[i] << "\n";
        os.flush();
      }
    }
  }
}

void runTensorRecovery(Module& M, const TensorRecoveryOptions& opts, TensorRecoveryResult& out) {
  if (!opts.enabled) return;
  const char* stage = getenv("MVCC_SIG_STAGE");
  const bool source = stage && std::string(stage) == "source";
  const char* only = getenv("MVCC_SIG_ONLY");
  if (source || (only && only[0] == '1')) runSig(M, out);
  if (only && only[0] == '1') return;
  std::vector<Function*> kernels;
  for (Function& F : M) if (!F.isDeclaration() && isKernelFn(F)) kernels.push_back(&F);
  for (Function* F : kernels) {
    KernelRecovery R(*F, opts, out);
    R.run();
  }
  emit::runModule(M, opts, out);
  // SIG the recovered body only when asked — post-recovery SIG on a 256×128 mainloop blows the 30s budget
  // and is not the identity dump the goldens compare. MVCC_SIG_STAGE=recovered opts in.
  if (!source && stage && std::string(stage) == "recovered") runSig(M, out);
}

}  // namespace mvcc
