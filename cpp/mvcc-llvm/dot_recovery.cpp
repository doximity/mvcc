// Recover a decode-DOT contraction onto a Metal GEMV schedule.
//
// Program:  Y[n] = ⊕_k scale · ⟨unpack(W[n,k]), x[k]⟩
// Source:   K-loop of fma + 32-lane xor-sum (the CUDA schedule).
// Metal:    same F, exchange=simd_sum, K partitioned across thread_scale copies
//           of the host block (physical thread p is virtual thread p % T).
// Twin:     untouched source body, named <kernel>__mvcc_exact.
#include "dot_recovery.h"
#include "emit.h"
#include "engine.h"
#include "legality.h"
#include "schedule.h"
#include "stir.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/Analysis/PostDominators.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Demangle/Demangle.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Verifier.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <set>

using namespace llvm;

namespace mvcc {
namespace emit {

namespace {

Value* strip(Value* V) {
  while (V) {
    if (auto* C = dyn_cast<CastInst>(V)) { V = C->getOperand(0); continue; }
    break;
  }
  return V;
}

bool hasMmaAsm(const Function& F) {
  for (auto& I : instructions(F)) {
    auto* CI = dyn_cast<CallInst>(&I);
    if (!CI) continue;
    if (auto* IA = dyn_cast<InlineAsm>(CI->getCalledOperand())) {
      StringRef a = IA->getAsmString();
      if (a.contains("mma.") || a.contains("ldmatrix")) return true;
    }
    if (Function* C = CI->getCalledFunction())
      if (C->getName().starts_with("__mvcc_tp_mma") || C->getName().starts_with("__mvcc_tile_mma"))
        return true;
  }
  return false;
}

bool isTidX(const CallInst* CI) {
  if (!CI || !CI->getCalledFunction()) return false;
  return CI->getCalledFunction()->getName() == "llvm.nvvm.read.ptx.sreg.tid.x";
}

bool isExp(const Instruction& I) {
  auto* CI = dyn_cast<CallInst>(&I);
  if (!CI || !CI->getCalledFunction()) return false;
  StringRef n = CI->getCalledFunction()->getName();
  // not llvm.experimental.* (noalias.scope.decl in DOT loops)
  return n.starts_with("__nv_exp") || n.starts_with("__nv_fast_exp")
      || n == "llvm.exp" || n.starts_with("llvm.exp.") || n.starts_with("llvm.exp2");
}

bool isFmax(const Instruction& I) {
  if (auto* CI = dyn_cast<CallInst>(&I)) {
    Intrinsic::ID id = CI->getIntrinsicID();
    StringRef n = CI->getCalledFunction() ? CI->getCalledFunction()->getName() : "";
    if (id == Intrinsic::maxnum || id == Intrinsic::maximum) return true;
    if (n.starts_with("llvm.nvvm.fmax") || n == "__nv_fmaxf") return true;
  }
  return false;
}

bool isFmaLike(const Instruction& I, Value*& a, Value*& b, Value*& c) {
  a = b = c = nullptr;
  if (auto* BO = dyn_cast<BinaryOperator>(&I)) {
    if (BO->getOpcode() == Instruction::FMul) {
      a = BO->getOperand(0); b = BO->getOperand(1); return true;
    }
    if (BO->getOpcode() == Instruction::FAdd) {
      if (auto* M = dyn_cast<BinaryOperator>(strip(BO->getOperand(0))))
        if (M->getOpcode() == Instruction::FMul) {
          a = M->getOperand(0); b = M->getOperand(1); c = BO->getOperand(1); return true;
        }
      if (auto* M = dyn_cast<BinaryOperator>(strip(BO->getOperand(1))))
        if (M->getOpcode() == Instruction::FMul) {
          a = M->getOperand(0); b = M->getOperand(1); c = BO->getOperand(0); return true;
        }
    }
    return false;
  }
  auto* CI = dyn_cast<CallInst>(&I);
  if (!CI) return false;
  Intrinsic::ID id = CI->getIntrinsicID();
  StringRef n = CI->getCalledFunction() ? CI->getCalledFunction()->getName() : "";
  if (id == Intrinsic::fmuladd || id == Intrinsic::fma || n.starts_with("llvm.nvvm.fma") || n == "__nv_fmaf") {
    if (CI->arg_size() < 3) return false;
    a = CI->getArgOperand(0); b = CI->getArgOperand(1); c = CI->getArgOperand(2);
    return true;
  }
  return false;
}

bool shflXor(Value* V, Value*& src, unsigned& delta) {
  V = strip(V);
  auto* CI = dyn_cast<CallInst>(V);
  if (!CI || !CI->getCalledFunction()) return false;
  StringRef n = CI->getCalledFunction()->getName();
  if (!n.starts_with("llvm.nvvm.shfl.sync.bfly")) return false;
  if (CI->arg_size() < 4) return false;
  auto* D = dyn_cast<ConstantInt>(CI->getArgOperand(2));
  auto* C = dyn_cast<ConstantInt>(CI->getArgOperand(3));
  if (!D || !C || (C->getZExtValue() & 0x1f) != 0x1f) return false;
  delta = (unsigned)D->getZExtValue() & 0x1f;
  src = strip(CI->getArgOperand(1));
  return true;
}

bool xorStep(Instruction* I, Value*& prev, unsigned& delta) {
  if (!I) return false;
  Value *a = nullptr, *b = nullptr;
  if (auto* BO = dyn_cast<BinaryOperator>(I)) {
    if (BO->getOpcode() != Instruction::FAdd && BO->getOpcode() != Instruction::Add) return false;
    a = BO->getOperand(0); b = BO->getOperand(1);
  } else {
    return false;
  }
  Value* src = nullptr; unsigned d = 0;
  if (shflXor(b, src, d) && strip(a) == src) { prev = src; delta = d; return true; }
  if (shflXor(a, src, d) && strip(b) == src) { prev = src; delta = d; return true; }
  return false;
}

// Walk a 16/8/4/2/1 add-butterfly ending at sink. uniform is the +16 add.
bool xorTree(Instruction* sink, Value*& root, Instruction*& uniform, SmallPtrSet<Instruction*, 8>& nodes) {
  Instruction* cur = sink;
  root = nullptr; uniform = nullptr; nodes.clear();
  const unsigned expect[] = {1, 2, 4, 8, 16};
  for (unsigned e : expect) {
    Value* prev = nullptr; unsigned d = 0;
    if (!xorStep(cur, prev, d) || d != e) return false;
    nodes.insert(cur);
    root = prev;
    if (e == 16) uniform = cur;
    if (e != 16) {
      auto* PI = dyn_cast<Instruction>(prev);
      if (!PI) return false;
      cur = PI;
    }
  }
  return root && uniform;
}

// Packed W field: nibble / byte / fp8-in-half masks. Not bf16 widen (shl 16) and not address and/or.
bool packingMask(const ConstantInt* C) {
  if (!C) return false;
  const uint64_t v = C->getZExtValue();
  switch (v) {
    case 0x0Full:
    case 0xF0ull:
    case 0xFFull:
    case 0x0F0F0F0Full:
    case 0xF0F0F0F0ull:
    case 0x00FF00FFull:
    case 0xFF00FF00ull:
    case 0x007F007Full:
    case 0x7F007F00ull:
    case 0x00800080ull:
    case 0x80008000ull:
      return true;
    default:
      return false;
  }
}

bool packingAnd(Value* V, int depth, SmallPtrSet<Value*, 8>& seen) {
  if (!V || depth > 14 || !seen.insert(V).second) return false;
  V = strip(V);
  if (auto* BO = dyn_cast<BinaryOperator>(V)) {
    if (BO->getOpcode() == Instruction::And
        && (packingMask(dyn_cast<ConstantInt>(BO->getOperand(0)))
            || packingMask(dyn_cast<ConstantInt>(BO->getOperand(1)))))
      return true;
    if (BO->getOpcode() == Instruction::LShr || BO->getOpcode() == Instruction::AShr
        || BO->getOpcode() == Instruction::Shl)
      return packingAnd(BO->getOperand(0), depth + 1, seen)
          || packingAnd(BO->getOperand(1), depth + 1, seen);
  }
  if (auto* I = dyn_cast<Instruction>(V))
    for (Value* o : I->operands())
      if (packingAnd(o, depth + 1, seen)) return true;
  return false;
}

bool loadRoot(Value* V, Value*& base, bool& unpack, int depth, SmallPtrSet<Value*, 8>& seen) {
  if (!V || depth > 14 || !seen.insert(V).second) return false;
  V = strip(V);
  if (auto* L = dyn_cast<LoadInst>(V)) {
    base = getUnderlyingObject(L->getPointerOperand());
    return base != nullptr;
  }
  if (auto* EE = dyn_cast<ExtractValueInst>(V)) return loadRoot(EE->getAggregateOperand(), base, unpack, depth + 1, seen);
  if (auto* El = dyn_cast<ExtractElementInst>(V)) return loadRoot(El->getVectorOperand(), base, unpack, depth + 1, seen);
  if (auto* I = dyn_cast<Instruction>(V)) {
    SmallPtrSet<Value*, 8> u;
    if (packingAnd(I, 0, u)) unpack = true;
    for (Value* o : I->operands()) {
      Value* b = nullptr; bool un = unpack;
      if (loadRoot(o, b, un, depth + 1, seen)) { base = b; unpack = un; return true; }
    }
  }
  return false;
}

bool accFromZero(PHINode* P, Loop* L) {
  if (!P->getType()->isFloatTy()) return false;
  bool zero = false;
  for (unsigned i = 0; i < P->getNumIncomingValues(); i++) {
    if (L->contains(P->getIncomingBlock(i))) continue;
    auto* C = dyn_cast<ConstantFP>(P->getIncomingValue(i));
    if (C && C->isZero()) zero = true;
    else return false;
  }
  return zero;
}

bool nextIsSum(PHINode* P, Loop* L) {
  for (unsigned i = 0; i < P->getNumIncomingValues(); i++) {
    if (!L->contains(P->getIncomingBlock(i))) continue;
    Value* n = P->getIncomingValue(i);
    if (n == P) continue;
    Value *a, *b, *c;
    if (auto* I = dyn_cast<Instruction>(n)) {
      if (isFmaLike(*I, a, b, c)) return true;
      if (auto* BO = dyn_cast<BinaryOperator>(I))
        if (BO->getOpcode() == Instruction::FAdd) return true;
    }
  }
  return false;
}

Function* simdSumFn(Module* M, Type* T) {
  FunctionCallee C = M->getOrInsertFunction("__mvcc_simd_sum", FunctionType::get(T, {T}, false));
  auto* Fn = cast<Function>(C.getCallee());
  Fn->setDoesNotAccessMemory();
  Fn->setDoesNotThrow();
  Fn->addFnAttr(Attribute::Convergent);
  return Fn;
}

CallInst* barrierProto(Function& F) {
  for (auto& I : instructions(F)) {
    auto* CI = dyn_cast<CallInst>(&I);
    if (!CI || !CI->getCalledFunction()) continue;
    StringRef n = CI->getCalledFunction()->getName();
    if (n == "llvm.nvvm.barrier0" || n.starts_with("llvm.nvvm.barrier.sync")) return CI;
  }
  return nullptr;
}

void emitBarrier(IRBuilder<>& B, Function& F, CallInst* proto) {
  if (proto) { B.Insert(proto->clone()); return; }
  FunctionCallee C = F.getParent()->getOrInsertFunction("llvm.nvvm.barrier0",
                                                        FunctionType::get(B.getVoidTy(), false));
  B.CreateCall(C);
}

stir::Program gemvProgram(const std::string& name) {
  stir::Program P;
  P.kernel = name;
  P.coordNames = {"n", "k"};
  P.coordOutput = {true, false};
  // n: high tid bits + block; k: 32-lane xor-sum + the K loop on time
  P.coordPhys = {"b5+b6+b7", "b0+b1+b2+b3+b4+i0"};
  stir::LoopStmt L;
  L.header = "%k"; L.depth = 1; L.kind = sig::LoopKind::Traversal;
  L.algebra.push_back({sig::Algebra::Sum, 1});
  L.permits = {true, true, false, true};
  L.anyData = true;
  P.loops.push_back(L);
  stir::Tensor W, x, s, y;
  W.name = "W"; W.elemBytes = 1; W.read = true;
  x.name = "x"; x.elemBytes = 4; x.read = true;
  s.name = "scale"; s.elemBytes = 4; s.read = true;
  y.name = "Y"; y.elemBytes = 4; y.written = true;
  P.tensors = {W, x, s, y};
  return P;
}

sched::Schedule metalSchedule(const stir::Program& P, unsigned scale) {
  sched::Schedule S = sched::sourceSchedule(P);
  S.why = "engine:simd-vector+split:simdgroup*" + std::to_string(scale);
  if (S.stages.empty()) return S;
  auto& st = S.stages[0];
  st.engine = sched::Engine::SimdVector;
  st.exchange = sched::Exchange::ShuffleTree;
  st.prefetch = sched::Prefetch::IterationStart;
  st.threadScale = scale ? scale : 1;
  if (st.predicate == sched::Predicate::None && !P.guards.empty())
    st.predicate = sched::Predicate::Branch;
  return S;
}

bool usesTid(const Value* V, SmallPtrSet<const Value*, 16>& seen) {
  if (!V || !seen.insert(V).second) return false;
  if (auto* C = dyn_cast<CallInst>(V)) if (isTidX(C)) return true;
  if (auto* I = dyn_cast<Instruction>(V))
    for (const Value* o : I->operands())
      if (usesTid(o, seen)) return true;
  return false;
}

// simd_sum is convergent. Insert only when the +16 add is not control-dependent
// on a tid branch (`if (tid < 32)` in block_sum). A tid test that rejoins before
// the tree (`if (tid < K) skip the K-loop`) is fine — BB post-dominates it.
// block_sum's second tree: `if (tid < 32) warp_sum`. A tid-stride loop
// (`c = tid; c < N`) is not that — its xor may sit on only one inlined path.
bool warpFilter(Value* cond) {
  auto* C = dyn_cast<ICmpInst>(cond);
  if (!C) return false;
  Value *a = C->getOperand(0), *b = C->getOperand(1);
  if (C->getPredicate() != ICmpInst::ICMP_ULT && C->getPredicate() != ICmpInst::ICMP_SLT)
    return false;
  auto* K = dyn_cast<ConstantInt>(b);
  Value* t = a;
  if (!K) { K = dyn_cast<ConstantInt>(a); t = b; }
  if (!K || K->getZExtValue() != 32) return false;
  SmallPtrSet<const Value*, 16> seen;
  return usesTid(t, seen);
}

bool tidControlDep(BasicBlock* BB, DominatorTree& DT, PostDominatorTree& PDT, LoopInfo& LI) {
  Function* F = BB->getParent();
  for (BasicBlock& D : *F) {
    if (&D == BB || !DT.dominates(&D, BB)) continue;
    if (Loop* L = LI.getLoopFor(&D); L && L->getHeader() == &D) continue;
    auto* Br = dyn_cast<BranchInst>(D.getTerminator());
    if (!Br || !Br->isConditional() || !warpFilter(Br->getCondition())) continue;
    if (!PDT.dominates(BB, &D)) return true;
  }
  return false;
}

struct KLoop {
  PHINode* ind = nullptr;
  int64_t step = 0;
  PHINode* acc = nullptr;
  bool uniqueExit = false;
};

struct Match {
  PHINode* ind = nullptr;
  int64_t step = 0;
  std::vector<PHINode*> accs;
  std::vector<Instruction*> xorSinks;
  std::vector<KLoop> kloops;
  bool unpack = false;
  bool uniqueExit = false;
  unsigned nMatch = 0;
};

bool recognize(Function& F, Match& M) {
  if (hasMmaAsm(F)) return false;
  DominatorTree DT(F);
  PostDominatorTree PDT(F);
  LoopInfo LI(DT);
  Match best;
  for (Loop* L : LI.getLoopsInPreorder()) {
    if (!L->getHeader() || !L->getLoopLatch()) continue;
    bool exp = false, fmax = false, store = false, atomic = false;
    bool product = false, unpack = false;
    std::vector<PHINode*> accs;
    PHINode* ind = nullptr;
    int64_t step = 0;
    for (BasicBlock* BB : L->blocks()) {
      for (Instruction& I : *BB) {
        if (isExp(I)) exp = true;
        if (isFmax(I)) fmax = true;
        if (auto* St = dyn_cast<StoreInst>(&I)) {
          Value* u = getUnderlyingObject(St->getPointerOperand());
          if (u && !isa<AllocaInst>(u)) store = true;
        }
        if (isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I)) atomic = true;
        if (auto* CI = dyn_cast<CallInst>(&I))
          if (CI->getCalledFunction() && CI->getCalledFunction()->getName().starts_with("llvm.nvvm.atomic"))
            atomic = true;
        Value *a, *b, *c;
        if (isFmaLike(I, a, b, c)) {
          Value *ba = nullptr, *bb = nullptr; bool ua = false, ub = false;
          SmallPtrSet<Value*, 8> sa, sb;
          bool la = loadRoot(a, ba, ua, 0, sa);
          bool lb = loadRoot(b, bb, ub, 0, sb);
          if (la && lb && ba && bb && ba != bb) {
            product = true;
            unpack = unpack || ua || ub;
          }
        }
      }
    }
    if (exp || fmax || store || atomic || !product || !unpack) continue;
    BasicBlock* H = L->getHeader();
    BasicBlock* latch = L->getLoopLatch();
    for (PHINode& P : H->phis()) {
      if (accFromZero(&P, L) && nextIsSum(&P, L)) accs.push_back(&P);
      if (!P.getType()->isIntegerTy()) continue;
      Value* nxt = P.getIncomingValueForBlock(latch);
      auto* Add = dyn_cast<BinaryOperator>(nxt);
      if (!Add || Add->getOpcode() != Instruction::Add) continue;
      Value* o = Add->getOperand(0) == &P ? Add->getOperand(1) : (Add->getOperand(1) == &P ? Add->getOperand(0) : nullptr);
      auto* C = dyn_cast<ConstantInt>(o);
      if (!C || C->isZero()) continue;
      ind = &P;
      step = C->getSExtValue();
    }
    if (accs.empty()) continue;
    auto rootOfAcc = [&](Value* root) -> bool {
      root = strip(root);
      for (PHINode* A : accs) {
        if (root == A) return true;
        Value* nxt = latch ? A->getIncomingValueForBlock(latch) : nullptr;
        if (root == nxt) return true;
        if (auto* P = dyn_cast<PHINode>(root)) {
          for (unsigned i = 0; i < P->getNumIncomingValues(); i++) {
            Value* in = strip(P->getIncomingValue(i));
            if (in == A || in == nxt) return true;
          }
        }
      }
      return false;
    };
    std::vector<Instruction*> sinks;
    for (Function::iterator BBi = F.begin(), BBe = F.end(); BBi != BBe; ++BBi) {
      if (L->contains(&*BBi)) continue;
      for (Instruction& I : *BBi) {
        Value* root; Instruction* uni; SmallPtrSet<Instruction*, 8> nodes;
        if (!xorTree(&I, root, uni, nodes) || !uni) continue;
        if (!rootOfAcc(root)) continue;
        if (tidControlDep(uni->getParent(), DT, PDT, LI)) continue;
        sinks.push_back(&I);
      }
    }
    if (sinks.empty()) {
      continue;
    }
    best.nMatch++;
    best.xorSinks.insert(best.xorSinks.end(), sinks.begin(), sinks.end());
    best.kloops.push_back({ind, step, accs.front(), L->getUniqueExitBlock() != nullptr});
    if (accs.size() > best.accs.size()) {
      best.ind = ind; best.step = step; best.accs = accs;
      best.unpack = unpack;
      best.uniqueExit = L->getUniqueExitBlock() != nullptr;
    }
  }
  if (best.accs.empty() || best.xorSinks.empty()) return false;
  M = best;
  return true;
}

bool emitBody(Function& F, const Match& M, unsigned T, unsigned scale, Value*& hOut) {
  Type* i32 = Type::getInt32Ty(F.getContext());
  BasicBlock& entry = F.getEntryBlock();
  Instruction* ip = entry.getFirstNonPHI();
  while (ip && (isa<AllocaInst>(ip) || isa<DbgInfoIntrinsic>(ip))) ip = ip->getNextNode();
  if (!ip) return false;
  IRBuilder<> B(ip);
  FunctionCallee tidFn = F.getParent()->getOrInsertFunction("llvm.nvvm.read.ptx.sreg.tid.x",
                                                            FunctionType::get(i32, false));
  Value* Tv = ConstantInt::get(i32, T);
  Value* virt = nullptr;
  Value* h = ConstantInt::get(i32, 0);
  if (scale > 1) {
    Value* p = B.CreateCall(tidFn, {}, "dot.p");
    virt = B.CreateURem(p, Tv, "dot.virt");
    h = B.CreateUDiv(p, Tv, "dot.h");
    std::vector<CallInst*> old;
    for (auto& I : instructions(F))
      if (auto* C = dyn_cast<CallInst>(&I)) if (isTidX(C) && C != p) old.push_back(C);
    for (CallInst* C : old) { C->replaceAllUsesWith(virt); C->eraseFromParent(); }
  }
  hOut = h;

  DominatorTree DT0(F);
  LoopInfo LI0(DT0);
  Loop* K0 = LI0.getLoopFor(M.accs.front()->getParent());
  if (!K0) return false;
  SmallPtrSet<BasicBlock*, 16> loopBB;
  for (BasicBlock* BB : K0->blocks()) loopBB.insert(BB);
  auto splitK = [&](PHINode* ind, int64_t step) {
    if (!ind || step == 0) return;
    Loop* K = LI0.getLoopFor(ind->getParent());
    if (!K || !K->getLoopPreheader() || !K->getLoopLatch()) return;
    BasicBlock* pre = K->getLoopPreheader();
    BasicBlock* latch = K->getLoopLatch();
    Value* nxt = ind->getIncomingValueForBlock(latch);
    if (auto* Add = dyn_cast<BinaryOperator>(nxt)) {
      unsigned ci = Add->getOperand(0) == ind ? 1 : 0;
      Add->setOperand(ci, ConstantInt::get(cast<IntegerType>(Add->getOperand(ci)->getType()), step * (int64_t)scale));
    }
    IRBuilder<> Pb(pre->getTerminator());
    Value* init = ind->getIncomingValueForBlock(pre);
    Value* hs = Pb.CreateMul(Pb.CreateZExtOrTrunc(h, init->getType()),
                             ConstantInt::get(cast<IntegerType>(init->getType()), step), "dot.hstep");
    ind->setIncomingValueForBlock(pre, Pb.CreateAdd(init, hs, "dot.k0"));
    Type* iTy = ind->getType();
    for (auto& I : instructions(F)) {
      auto* Add = dyn_cast<BinaryOperator>(&I);
      if (!Add || Add->getOpcode() != Instruction::Add) continue;
      if (Add == nxt) continue;
      Value* oth = nullptr;
      if (Add->getOperand(0) == ind) oth = Add->getOperand(1);
      else if (Add->getOperand(1) == ind) oth = Add->getOperand(0);
      auto* C = dyn_cast<ConstantInt>(oth);
      if (!C || C->getSExtValue() != step) continue;
      unsigned ci = Add->getOperand(0) == ind ? 1 : 0;
      Add->setOperand(ci, ConstantInt::get(cast<IntegerType>(iTy), step * (int64_t)scale));
    }
  };
  if (scale > 1 && virt) {
    if (!M.kloops.empty()) {
      for (const KLoop& kl : M.kloops) splitK(kl.ind, kl.step);
    } else {
      splitK(M.ind, M.step);
    }
  }

  Function* sumFn = simdSumFn(F.getParent(), Type::getFloatTy(F.getContext()));
  CallInst* proto = barrierProto(F);
  const unsigned Fsc = scale > 1 ? scale : 1;
  GlobalVariable* buf = nullptr;
  if (Fsc > 1) {
    auto* ty = ArrayType::get(Type::getFloatTy(F.getContext()), (uint64_t)Fsc * T);
    buf = new GlobalVariable(*F.getParent(), ty, false, GlobalValue::InternalLinkage,
                             UndefValue::get(ty), F.getName() + ".dot.xchg", nullptr,
                             GlobalValue::NotThreadLocal, 3);
    buf->setAlignment(Align(16));
  }
  SmallPtrSet<Instruction*, 16> trees;
  for (Instruction* sink : M.xorSinks) {
    if (!sink->getParent()) continue;
    Value* root; Instruction* uni; SmallPtrSet<Instruction*, 8> nodes;
    if (!xorTree(sink, root, uni, nodes) || !uni) continue;
    trees.insert(nodes.begin(), nodes.end());
    IRBuilder<> Tb(uni);
    Value* s = Tb.CreateCall(sumFn, {root}, "dot.simd");
    Value* tot = s;
    if (Fsc > 1 && buf) {
      Value* slot = Tb.CreateAdd(Tb.CreateMul(h, Tv), virt);
      Value* gep = Tb.CreateInBoundsGEP(Type::getFloatTy(F.getContext()), buf,
                                       Tb.CreateZExt(slot, Tb.getInt64Ty()));
      Tb.CreateStore(s, gep);
      emitBarrier(Tb, F, proto);
      tot = nullptr;
      for (unsigned r = 0; r < Fsc; r++) {
        Value* rs = Tb.CreateAdd(Tb.CreateMul(ConstantInt::get(i32, r), Tv), virt);
        Value* ld = Tb.CreateLoad(Type::getFloatTy(F.getContext()),
                                 Tb.CreateInBoundsGEP(Type::getFloatTy(F.getContext()), buf,
                                                      Tb.CreateZExt(rs, Tb.getInt64Ty())));
        tot = tot ? Tb.CreateFAdd(tot, ld, "dot.xadd") : ld;
      }
      emitBarrier(Tb, F, proto);
    }
    sink->replaceAllUsesWith(tot);
  }
  bool progress = true;
  while (progress) {
    progress = false;
    for (BasicBlock& BB : F)
      for (Instruction& I : llvm::make_early_inc_range(BB)) {
        if (!I.use_empty() || I.isTerminator()) continue;
        if (trees.count(&I) || isa<BitCastInst>(&I)) { I.eraseFromParent(); progress = true; continue; }
        auto* CI = dyn_cast<CallInst>(&I);
        if (CI && CI->getCalledFunction() && CI->getCalledFunction()->getName().starts_with("llvm.nvvm.shfl")) {
          I.eraseFromParent(); progress = true;
        }
      }
  }

  if (Fsc > 1) {
    // h=1 is a K-partition copy of the same virtual block: it must still write
    // smem (scale tables, stage rings). Only device atomics would fire twice.
    std::vector<Instruction*> sides;
    for (auto& I : instructions(F)) {
      if (loopBB.count(I.getParent())) continue;
      if (isa<AtomicRMWInst>(&I) || isa<AtomicCmpXchgInst>(&I)) {
        sides.push_back(&I);
      } else if (auto* CI = dyn_cast<CallInst>(&I)) {
        if (CI->getCalledFunction() && CI->getCalledFunction()->getName().starts_with("llvm.nvvm.atomic"))
          sides.push_back(CI);
      }
    }
    for (Instruction* S : sides) {
      if (!S->getParent()) continue;
      IRBuilder<> Pb(S);
      Value* isH0 = Pb.CreateICmpEQ(h, ConstantInt::get(i32, 0), "dot.h0");
      Instruction* thenTerm = SplitBlockAndInsertIfThen(isH0, S, false);
      S->moveBefore(thenTerm->getIterator());
    }
    F.removeFnAttr("nvvm.maxntid");
    F.addFnAttr("nvvm.maxntid", std::to_string(T * Fsc) + ",1,1");
  }
  return true;
}

}  // namespace

// Off: the emitted gemv-dot bodies hung the GPU on an M5 Pro in Qwen decode, and the GPU reset took the machine down.
// Prove on decode_dot.cu before re-enabling.
constexpr bool kGemvDotRecovery = false;

Result recoverDotGemv(Function& F, unsigned launchBound, TensorRecoveryResult& res) {
  Result R;
  if (!kGemvDotRecovery) return R;
  if (F.getName().ends_with("__mvcc_exact") || F.getName().contains("__mvcc_s") || F.getName().contains(".dot."))
    return R;
  Match M;
  if (!recognize(F, M)) return R;
  const unsigned T = launchBound ? launchBound : 32;
  unsigned scale = 1;
  bool allSplit = !M.kloops.empty();
  for (const KLoop& kl : M.kloops)
    if (!kl.ind || kl.step == 0 || !kl.uniqueExit) allSplit = false;
  if (T && T <= 128 && T * 2 <= 256 && allSplit) scale = 2;
  stir::Program P = gemvProgram(llvm::demangle(F.getName().str()));
  sched::Schedule S = metalSchedule(P, scale);
  legal::Result L = legal::check(P, S);
  if (!L.ok) {
    S = metalSchedule(P, 1);
    S.why = "engine:simd-vector";
    if (!S.stages.empty()) S.stages[0].threadScale = 1;
    L = legal::check(P, S);
    scale = 1;
  }
  if (!L.ok) {
    R.why = "check+emit kept: gemv-dot legality " + L.first();
    return R;
  }
  engine::analyze(P, res.notes);

  ValueToValueMapTy VM;
  Function* twin = CloneFunction(&F, VM);
  twin->setName(F.getName() + "__mvcc_exact");
  res.exactVariants.push_back({F.getName().str(), twin->getName().str()});

  Value* h = nullptr;
  if (!emitBody(F, M, T, scale, h)) {
    F.deleteBody();
    ValueToValueMapTy BVM;
    Function::arg_iterator DA = F.arg_begin();
    for (Argument& A : twin->args()) { BVM[&A] = &*DA++; }
    SmallVector<ReturnInst*, 8> rets;
    CloneFunctionInto(&F, twin, BVM, CloneFunctionChangeType::LocalChangesOnly, rets);
    twin->eraseFromParent();
    res.exactVariants.pop_back();
    R.why = "check+emit kept: gemv-dot emit failed";
    return R;
  }
  std::string verr; raw_string_ostream vos(verr);
  if (verifyFunction(F, &vos)) {
    F.deleteBody();
    ValueToValueMapTy BVM;
    Function::arg_iterator DA = F.arg_begin();
    for (Argument& A : twin->args()) { BVM[&A] = &*DA++; }
    SmallVector<ReturnInst*, 8> rets;
    CloneFunctionInto(&F, twin, BVM, CloneFunctionChangeType::LocalChangesOnly, rets);
    twin->eraseFromParent();
    res.exactVariants.pop_back();
    R.why = "check+emit kept: gemv-dot invalid IR: " + verr;
    res.warnings.push_back(llvm::demangle(F.getName().str()) + ": " + R.why);
    return R;
  }
  if (scale > 1) res.threadScale[F.getName().str()] = scale;
  R.emitted = true;
  R.why = std::string("check+emit: gemv-dot Y[n]=⊕_k scale·⟨unpack(W),x⟩ exchange=simd_sum thread_scale=")
          + std::to_string(scale) + " engine=simd-vector " + sched::stageShape(S)
          + (M.unpack ? " unpack" : "") + " loops=" + std::to_string(M.nMatch) + " (exact twin)";
  pushRecoverySuccessNote(res, llvm::demangle(F.getName().str()) + ": " + R.why);
  return R;
}

}  // namespace emit
}  // namespace mvcc
