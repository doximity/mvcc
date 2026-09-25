// Scalarize vector values MSL cannot represent: more than 4 lanes, or lanes of double (carried as the mvcc_f64
// struct, which has no vector form). LLVM's middle end builds these freely (SLP on byte streams, <8 x bfloat>
// from 128-bit packed loads, <2 x double> from SROA); LegalizeWideVectorsPass splits the loads and stores, this
// pass takes whatever still flows through arithmetic, shuffles, bitcasts, selects and phis and rewrites each wide
// value as its lanes. Bitcasts between wide and narrow types are re-expressed as integer shifts and masks over
// the lanes (little-endian lane order, as LLVM defines it), so no value changes bits.
#include "wide_vectors.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/PostOrderIterator.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace mvcc {
namespace {

bool isWideTy(Type* T) {
  auto* VT = dyn_cast<FixedVectorType>(T);
  if (!VT) return false;
  return VT->getNumElements() > 4 || VT->getElementType()->isDoubleTy();
}

struct Scalarizer {
  Function& F;
  const DataLayout& DL;
  std::string& err;
  DenseMap<Value*, SmallVector<Value*, 16>> parts;
  SmallVector<Instruction*, 32> dead;
  SmallVector<std::pair<PHINode*, SmallVector<PHINode*, 16>>, 8> phis;  // wide phi -> lane phis (filled at the end)

  Scalarizer(Function& f, std::string& e) : F(f), DL(f.getParent()->getDataLayout()), err(e) {}

  bool fail(const std::string& m) { if (err.empty()) err = m; return false; }

  Type* intTyForBits(unsigned bits) { return Type::getIntNTy(F.getContext(), bits); }

  Value* toInt(IRBuilder<>& B, Value* v) {
    if (v->getType()->isIntegerTy()) return v;
    if (v->getType()->isPointerTy()) return B.CreatePtrToInt(v, intTyForBits(DL.getTypeSizeInBits(v->getType())));
    return B.CreateBitCast(v, intTyForBits(DL.getTypeSizeInBits(v->getType())));
  }
  Value* fromInt(IRBuilder<>& B, Value* v, Type* T) {
    if (T->isIntegerTy()) return v;
    if (T->isPointerTy()) return B.CreateIntToPtr(v, T);
    return B.CreateBitCast(v, T);
  }

  // Lanes of a (possibly wide) vector value. Constants and narrow vectors are split with extractelement.
  bool lanesOf(IRBuilder<>& B, Value* V, SmallVectorImpl<Value*>& out) {
    auto it = parts.find(V);
    if (it != parts.end()) { out.assign(it->second.begin(), it->second.end()); return true; }
    auto* VT = cast<FixedVectorType>(V->getType());
    unsigned n = VT->getNumElements();
    if (auto* C = dyn_cast<Constant>(V)) {
      for (unsigned i = 0; i < n; i++) {
        Constant* e = C->getAggregateElement(i);
        if (!e) return fail("wide vector constant without extractable elements");
        out.push_back(e);
      }
      return true;
    }
    if (auto* PN = dyn_cast<PHINode>(V)) {
      if (isWideTy(PN->getType())) { makePhis(PN); return lanesOf(B, V, out); }
    }
    if (isWideTy(VT) && isa<Instruction>(V)) return fail("wide vector value used before it was scalarized");
    for (unsigned i = 0; i < n; i++) out.push_back(B.CreateExtractElement(V, B.getInt64(i)));
    return true;
  }

  void makePhis(PHINode* PN) {
    if (parts.count(PN)) return;
    auto* VT = cast<FixedVectorType>(PN->getType());
    IRBuilder<> B(PN);
    SmallVector<PHINode*, 16> lanes;
    SmallVector<Value*, 16> vals;
    for (unsigned i = 0; i < VT->getNumElements(); i++) {
      PHINode* p = B.CreatePHI(VT->getElementType(), PN->getNumIncomingValues());
      lanes.push_back(p); vals.push_back(p);
    }
    parts[PN] = vals;
    phis.push_back({PN, lanes});
    dead.push_back(PN);
  }

  // Re-slice a list of integer lanes (each `sb` bits) into `count` lanes of `db` bits (little-endian bit order).
  bool reslice(IRBuilder<>& B, ArrayRef<Value*> src, unsigned sb, unsigned db, unsigned count, SmallVectorImpl<Value*>& out) {
    if (db > 64 || sb > 64) return fail("bitcast through a >64-bit lane");
    Type* DT = intTyForBits(db);
    for (unsigned j = 0; j < count; j++) {
      if (db <= sb) {
        unsigned bit = j * db, si = bit / sb, sh = bit % sb;
        Value* w = src[si];
        if (sh) w = B.CreateLShr(w, sh);
        out.push_back(db == sb ? w : B.CreateTrunc(w, DT));
      } else {
        Value* acc = nullptr;
        for (unsigned k = 0; k < db / sb; k++) {
          Value* w = B.CreateZExt(src[(j * db) / sb + k], DT);
          if (k) w = B.CreateShl(w, k * sb);
          acc = acc ? B.CreateOr(acc, w) : w;
        }
        out.push_back(acc);
      }
    }
    return true;
  }

  // Lanes of a value of any type (scalar, narrow or wide vector) as integers of the element width.
  bool intLanes(IRBuilder<>& B, Value* V, SmallVectorImpl<Value*>& out, unsigned& bits) {
    if (auto* VT = dyn_cast<FixedVectorType>(V->getType())) {
      SmallVector<Value*, 16> l;
      if (!lanesOf(B, V, l)) return false;
      bits = DL.getTypeSizeInBits(VT->getElementType());
      for (Value* x : l) out.push_back(toInt(B, x));
      return true;
    }
    bits = DL.getTypeSizeInBits(V->getType());
    if (bits > 64) return fail("bitcast of a >64-bit scalar to a wide vector");
    out.push_back(toInt(B, V));
    return true;
  }

  Value* assemble(IRBuilder<>& B, ArrayRef<Value*> lanes, Type* T) {
    if (auto* VT = dyn_cast<FixedVectorType>(T)) {
      Value* v = PoisonValue::get(VT);
      for (unsigned i = 0; i < lanes.size(); i++) v = B.CreateInsertElement(v, fromInt(B, lanes[i], VT->getElementType()), B.getInt64(i));
      return v;
    }
    return fromInt(B, lanes[0], T);
  }

  Function* scalarIntrinsic(Intrinsic::ID id, Type* elem, ArrayRef<Type*> extra) {
    SmallVector<Type*, 2> tys{elem};
    tys.append(extra.begin(), extra.end());
    return Intrinsic::getOrInsertDeclaration(F.getParent(), id, tys);
  }

  // Produce the lanes of a wide-typed instruction.
  bool produce(Instruction* I) {
    auto* VT = cast<FixedVectorType>(I->getType());
    unsigned n = VT->getNumElements();
    Type* ET = VT->getElementType();
    IRBuilder<> B(I);
    SmallVector<Value*, 16> out;
    if (auto* PN = dyn_cast<PHINode>(I)) { makePhis(PN); return true; }
    if (auto* BO = dyn_cast<BinaryOperator>(I)) {
      SmallVector<Value*, 16> a, b;
      if (!lanesOf(B, BO->getOperand(0), a) || !lanesOf(B, BO->getOperand(1), b)) return false;
      for (unsigned i = 0; i < n; i++) {
        Value* r = B.CreateBinOp(BO->getOpcode(), a[i], b[i]);
        if (auto* RI = dyn_cast<Instruction>(r)) RI->copyIRFlags(BO);
        out.push_back(r);
      }
    } else if (auto* U = dyn_cast<UnaryOperator>(I)) {
      SmallVector<Value*, 16> a;
      if (!lanesOf(B, U->getOperand(0), a)) return false;
      for (unsigned i = 0; i < n; i++) out.push_back(B.CreateUnOp(U->getOpcode(), a[i]));
    } else if (auto* C = dyn_cast<CmpInst>(I)) {
      SmallVector<Value*, 16> a, b;
      if (!lanesOf(B, C->getOperand(0), a) || !lanesOf(B, C->getOperand(1), b)) return false;
      for (unsigned i = 0; i < n; i++) out.push_back(isa<ICmpInst>(C) ? B.CreateICmp(C->getPredicate(), a[i], b[i]) : B.CreateFCmp(C->getPredicate(), a[i], b[i]));
    } else if (auto* S = dyn_cast<SelectInst>(I)) {
      SmallVector<Value*, 16> t, f, c;
      if (!lanesOf(B, S->getTrueValue(), t) || !lanesOf(B, S->getFalseValue(), f)) return false;
      bool vc = S->getCondition()->getType()->isVectorTy();
      if (vc && !lanesOf(B, S->getCondition(), c)) return false;
      for (unsigned i = 0; i < n; i++) out.push_back(B.CreateSelect(vc ? c[i] : S->getCondition(), t[i], f[i]));
    } else if (auto* L = dyn_cast<LoadInst>(I)) {
      unsigned esz = DL.getTypeAllocSize(ET);
      for (unsigned i = 0; i < n; i++) {
        Value* p = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), L->getPointerOperand(), (uint64_t)i * esz);
        out.push_back(B.CreateAlignedLoad(ET, p, commonAlignment(L->getAlign(), (uint64_t)i * esz), L->isVolatile()));
      }
    } else if (auto* SV = dyn_cast<ShuffleVectorInst>(I)) {
      SmallVector<Value*, 16> a, b;
      if (!lanesOf(B, SV->getOperand(0), a)) return false;
      if (!isa<PoisonValue>(SV->getOperand(1)) && !isa<UndefValue>(SV->getOperand(1)) && !lanesOf(B, SV->getOperand(1), b)) return false;
      unsigned na = a.size();
      for (int m : SV->getShuffleMask()) {
        if (m < 0) out.push_back(PoisonValue::get(ET));
        else if ((unsigned)m < na) out.push_back(a[m]);
        else if ((unsigned)(m - na) < b.size()) out.push_back(b[m - na]);
        else out.push_back(PoisonValue::get(ET));
      }
    } else if (auto* IE = dyn_cast<InsertElementInst>(I)) {
      SmallVector<Value*, 16> a;
      if (!lanesOf(B, IE->getOperand(0), a)) return false;
      auto* CI = dyn_cast<ConstantInt>(IE->getOperand(2));
      if (!CI) {
        for (unsigned i = 0; i < n; i++) out.push_back(B.CreateSelect(B.CreateICmpEQ(IE->getOperand(2), ConstantInt::get(IE->getOperand(2)->getType(), i)), IE->getOperand(1), a[i]));
      } else { out = a; if (CI->getZExtValue() < n) out[CI->getZExtValue()] = IE->getOperand(1); }
    } else if (auto* CI = dyn_cast<CastInst>(I)) {
      Value* src = CI->getOperand(0);
      auto* ST = dyn_cast<FixedVectorType>(src->getType());
      if (CI->getOpcode() != Instruction::BitCast && ST && ST->getNumElements() == n) {
        SmallVector<Value*, 16> a;
        if (!lanesOf(B, src, a)) return false;
        for (unsigned i = 0; i < n; i++) out.push_back(B.CreateCast(CI->getOpcode(), a[i], ET));
      } else if (CI->getOpcode() == Instruction::BitCast) {
        SmallVector<Value*, 16> il, rl; unsigned sb;
        if (!intLanes(B, src, il, sb)) return false;
        if (!reslice(B, il, sb, DL.getTypeSizeInBits(ET), n, rl)) return false;
        for (Value* v : rl) out.push_back(fromInt(B, v, ET));
      } else return fail("unsupported cast producing a wide vector");
    } else if (auto* Call = dyn_cast<CallInst>(I)) {
      auto* II = dyn_cast<IntrinsicInst>(Call);
      if (!II) return fail("call returning a wide vector");
      Intrinsic::ID id = II->getIntrinsicID();
      unsigned nargs = Call->arg_size();
      SmallVector<SmallVector<Value*, 16>, 3> args(nargs);
      SmallVector<Type*, 2> extra;
      bool ok = true;
      switch (id) {
        case Intrinsic::fma: case Intrinsic::fmuladd: case Intrinsic::fabs: case Intrinsic::sqrt: case Intrinsic::floor: case Intrinsic::ceil:
        case Intrinsic::trunc: case Intrinsic::rint: case Intrinsic::nearbyint: case Intrinsic::round: case Intrinsic::roundeven:
        case Intrinsic::minnum: case Intrinsic::maxnum: case Intrinsic::minimum: case Intrinsic::maximum: case Intrinsic::copysign:
        case Intrinsic::smax: case Intrinsic::smin: case Intrinsic::umax: case Intrinsic::umin: case Intrinsic::abs:
        case Intrinsic::ctpop: case Intrinsic::ctlz: case Intrinsic::cttz: case Intrinsic::bswap: case Intrinsic::bitreverse:
        case Intrinsic::fshl: case Intrinsic::fshr: case Intrinsic::exp: case Intrinsic::exp2: case Intrinsic::log: case Intrinsic::log2:
        case Intrinsic::sin: case Intrinsic::cos: case Intrinsic::pow: break;
        default: ok = false;
      }
      if (!ok) return fail("intrinsic " + II->getCalledFunction()->getName().str() + " on a wide vector");
      for (unsigned k = 0; k < nargs; k++) {
        Value* a = Call->getArgOperand(k);
        if (a->getType()->isVectorTy()) { if (!lanesOf(B, a, args[k])) return false; }
        else args[k].assign(n, a);  // scalar flag operands (ctlz/cttz is_zero_poison, abs int_min_poison)
      }
      Function* SF = scalarIntrinsic(id, ET, {});
      for (unsigned i = 0; i < n; i++) {
        SmallVector<Value*, 3> av;
        for (unsigned k = 0; k < nargs; k++) av.push_back(args[k][i]);
        out.push_back(B.CreateCall(SF, av));
      }
    } else if (auto* FR = dyn_cast<FreezeInst>(I)) {
      SmallVector<Value*, 16> a;
      if (!lanesOf(B, FR->getOperand(0), a)) return false;
      for (unsigned i = 0; i < n; i++) out.push_back(B.CreateFreeze(a[i]));
    } else {
      return fail(std::string("cannot scalarize ") + I->getOpcodeName() + " producing a wide vector");
    }
    parts[I] = out;
    dead.push_back(I);
    return true;
  }

  // Rewrite a non-wide instruction that consumes a wide vector.
  bool consume(Instruction* I) {
    IRBuilder<> B(I);
    Value* repl = nullptr;
    if (auto* EE = dyn_cast<ExtractElementInst>(I)) {
      SmallVector<Value*, 16> a;
      if (!lanesOf(B, EE->getVectorOperand(), a)) return false;
      if (auto* CI = dyn_cast<ConstantInt>(EE->getIndexOperand())) repl = CI->getZExtValue() < a.size() ? a[CI->getZExtValue()] : PoisonValue::get(EE->getType());
      else {
        repl = a[0];
        for (unsigned i = 1; i < a.size(); i++) repl = B.CreateSelect(B.CreateICmpEQ(EE->getIndexOperand(), ConstantInt::get(EE->getIndexOperand()->getType(), i)), a[i], repl);
      }
    } else if (auto* S = dyn_cast<StoreInst>(I)) {
      SmallVector<Value*, 16> a;
      if (!lanesOf(B, S->getValueOperand(), a)) return false;
      Type* ET = cast<FixedVectorType>(S->getValueOperand()->getType())->getElementType();
      unsigned esz = DL.getTypeAllocSize(ET);
      for (unsigned i = 0; i < a.size(); i++) {
        Value* p = B.CreateConstInBoundsGEP1_64(B.getInt8Ty(), S->getPointerOperand(), (uint64_t)i * esz);
        B.CreateAlignedStore(a[i], p, commonAlignment(S->getAlign(), (uint64_t)i * esz), S->isVolatile());
      }
      I->eraseFromParent();
      return true;
    } else if (auto* BC = dyn_cast<BitCastInst>(I)) {
      SmallVector<Value*, 16> il, rl; unsigned sb;
      if (!intLanes(B, BC->getOperand(0), il, sb)) return false;
      Type* DT = BC->getType();
      unsigned db = DL.getTypeSizeInBits(DT->getScalarType());
      unsigned count = isa<FixedVectorType>(DT) ? cast<FixedVectorType>(DT)->getNumElements() : 1;
      if (!reslice(B, il, sb, db, count, rl)) return false;
      repl = assemble(B, rl, DT);
    } else if (auto* SV = dyn_cast<ShuffleVectorInst>(I)) {
      SmallVector<Value*, 16> a, b;
      if (!lanesOf(B, SV->getOperand(0), a)) return false;
      if (!isa<PoisonValue>(SV->getOperand(1)) && !isa<UndefValue>(SV->getOperand(1)) && !lanesOf(B, SV->getOperand(1), b)) return false;
      auto* RT = cast<FixedVectorType>(SV->getType());
      Value* v = PoisonValue::get(RT);
      unsigned j = 0, na = a.size();
      for (int m : SV->getShuffleMask()) {
        Value* e = m < 0 ? PoisonValue::get(RT->getElementType()) : ((unsigned)m < na ? a[m] : ((unsigned)(m - na) < b.size() ? b[m - na] : PoisonValue::get(RT->getElementType())));
        v = B.CreateInsertElement(v, e, B.getInt64(j++));
      }
      repl = v;
    } else if (auto* Call = dyn_cast<CallInst>(I)) {
      auto* II = dyn_cast<IntrinsicInst>(Call);
      if (!II) return fail("call taking a wide vector");
      Intrinsic::ID id = II->getIntrinsicID();
      Instruction::BinaryOps op; bool isBin = true; Intrinsic::ID mm = Intrinsic::not_intrinsic;
      switch (id) {
        case Intrinsic::vector_reduce_add: op = Instruction::Add; break;
        case Intrinsic::vector_reduce_mul: op = Instruction::Mul; break;
        case Intrinsic::vector_reduce_and: op = Instruction::And; break;
        case Intrinsic::vector_reduce_or: op = Instruction::Or; break;
        case Intrinsic::vector_reduce_xor: op = Instruction::Xor; break;
        case Intrinsic::vector_reduce_fadd: op = Instruction::FAdd; break;
        case Intrinsic::vector_reduce_fmul: op = Instruction::FMul; break;
        case Intrinsic::vector_reduce_smax: isBin = false; mm = Intrinsic::smax; break;
        case Intrinsic::vector_reduce_smin: isBin = false; mm = Intrinsic::smin; break;
        case Intrinsic::vector_reduce_umax: isBin = false; mm = Intrinsic::umax; break;
        case Intrinsic::vector_reduce_umin: isBin = false; mm = Intrinsic::umin; break;
        case Intrinsic::vector_reduce_fmax: isBin = false; mm = Intrinsic::maxnum; break;
        case Intrinsic::vector_reduce_fmin: isBin = false; mm = Intrinsic::minnum; break;
        default: return fail("intrinsic " + II->getCalledFunction()->getName().str() + " taking a wide vector");
      }
      bool fp = id == Intrinsic::vector_reduce_fadd || id == Intrinsic::vector_reduce_fmul;
      SmallVector<Value*, 16> a;
      if (!lanesOf(B, Call->getArgOperand(fp ? 1 : 0), a)) return false;
      Value* acc = fp ? Call->getArgOperand(0) : a[0];
      Function* SF = isBin ? nullptr : scalarIntrinsic(mm, a[0]->getType(), {});
      for (unsigned i = fp ? 0 : 1; i < a.size(); i++) acc = isBin ? B.CreateBinOp(op, acc, a[i]) : B.CreateCall(SF, {acc, a[i]});
      repl = acc;
    } else {
      return fail(std::string("cannot scalarize ") + I->getOpcodeName() + " consuming a wide vector");
    }
    I->replaceAllUsesWith(repl);
    I->eraseFromParent();
    return true;
  }

  bool run() {
    // Instructions in reverse post-order so operands are (mostly) scalarized before their users; phis are created
    // on demand and their incoming lanes are filled after the walk.
    ReversePostOrderTraversal<Function*> RPOT(&F);
    SmallVector<Instruction*, 64> order;
    for (BasicBlock* BB : RPOT) for (Instruction& I : *BB) order.push_back(&I);
    for (Instruction* I : order) {
      bool wideOut = isWideTy(I->getType());
      bool wideIn = false;
      for (Value* op : I->operands()) if (isWideTy(op->getType())) { wideIn = true; break; }
      if (wideOut) { if (!produce(I)) return false; }
      else if (wideIn) { if (!consume(I)) return false; }
    }
    for (auto& [PN, lanes] : phis) {
      for (unsigned k = 0; k < PN->getNumIncomingValues(); k++) {
        BasicBlock* pred = PN->getIncomingBlock(k);
        IRBuilder<> B(pred->getTerminator());
        SmallVector<Value*, 16> a;
        if (!lanesOf(B, PN->getIncomingValue(k), a)) return false;
        for (unsigned i = 0; i < lanes.size(); i++) lanes[i]->addIncoming(a[i], pred);
      }
    }
    for (Instruction* I : dead) I->replaceAllUsesWith(PoisonValue::get(I->getType()));
    for (Instruction* I : llvm::reverse(dead)) I->eraseFromParent();
    return true;
  }
};

}  // namespace

PreservedAnalyses ScalarizeWideVectorsPass::run(Function& F, FunctionAnalysisManager&) {
  bool any = false;
  for (BasicBlock& BB : F) for (Instruction& I : BB) {
    if (isWideTy(I.getType())) { any = true; break; }
    for (Value* op : I.operands()) if (isWideTy(op->getType())) { any = true; break; }
    if (any) break;
  }
  if (!any) return PreservedAnalyses::all();
  std::string err;
  Scalarizer S(F, err);
  if (!S.run()) {
    if (onError) onError(F, err);
    return PreservedAnalyses::all();
  }
  return PreservedAnalyses::none();
}

}  // namespace mvcc
