// Inline PTX asm parser. See ptx_asm.h.
#include "ptx_asm.h"
#include <cctype>
#include <cstdlib>

namespace mvcc {

std::string PtxInstr::full() const {
  std::string s = mnemonic;
  for (auto& m : mods) s += "." + m;
  return s;
}
bool PtxInstr::hasMod(const std::string& m) const {
  for (auto& x : mods) if (x == m) return true;
  return false;
}

namespace {
struct Lexer {
  const std::string& s; size_t i = 0;
  explicit Lexer(const std::string& str) : s(str) {}
  void ws() { while (i < s.size() && (isspace((unsigned char)s[i]) || s[i] == '\n' || s[i] == '\t')) i++; }
  bool eof() { ws(); return i >= s.size(); }
  char peek() { ws(); return i < s.size() ? s[i] : 0; }
  bool accept(char c) { if (peek() == c) { i++; return true; } return false; }
  std::string ident() {
    ws(); size_t b = i;
    // mnemonics carry their modifiers: mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
    while (i < s.size() && (isalnum((unsigned char)s[i]) || s[i] == '_' || s[i] == ':' || s[i] == '.')) i++;
    return s.substr(b, i - b);
  }
};

bool parseOperand(Lexer& L, PtxOperand& op, std::string& err) {
  char c = L.peek();
  if (c == '%' || c == '$') {  // %N / $N operands; %globaltimer and other sregs are Named
    L.i++;
    if (L.i < L.s.size() && (isalpha((unsigned char)L.s[L.i]) || L.s[L.i] == '_')) {
      size_t b = L.i;
      while (L.i < L.s.size() && (isalnum((unsigned char)L.s[L.i]) || L.s[L.i] == '_')) L.i++;
      op.kind = PtxOperand::Named; op.name = L.s.substr(b, L.i - b);
      return true;
    }
    size_t b = L.i; while (L.i < L.s.size() && isdigit((unsigned char)L.s[L.i])) L.i++;
    if (b == L.i) { err = "expected register number after %"; return false; }
    op.kind = PtxOperand::Reg; op.reg = atoi(L.s.substr(b, L.i - b).c_str());
    return true;
  }
  if (c == '!') { L.i++; if (!parseOperand(L, op, err)) return false; op.negated = true; return true; }
  if (c == '[') {
    L.i++;
    PtxOperand base;
    if (!parseOperand(L, base, err)) return false;
    op.kind = PtxOperand::Mem; op.reg = base.reg; op.imm = 0;
    if (L.accept('+') || L.peek() == '-') {
      size_t b = L.i; if (L.s[L.i] == '-') L.i++;
      while (L.i < L.s.size() && isalnum((unsigned char)L.s[L.i])) L.i++;
      op.imm = strtoll(L.s.substr(b, L.i - b).c_str(), nullptr, 0);
    }
    if (!L.accept(']')) { err = "expected ]"; return false; }
    return true;
  }
  if (c == '{') {
    L.i++;
    op.kind = PtxOperand::Vec;
    while (true) {
      PtxOperand e;
      if (!parseOperand(L, e, err)) return false;
      if (e.kind != PtxOperand::Reg) { err = "vector operand must contain registers"; return false; }
      op.regs.push_back(e.reg);
      if (L.accept(',')) continue;
      if (L.accept('}')) break;
      err = "expected , or } in vector operand"; return false;
    }
    return true;
  }
  if (isdigit((unsigned char)c) || c == '-') {
    size_t b = L.i; if (L.s[L.i] == '-') L.i++;
    while (L.i < L.s.size() && (isalnum((unsigned char)L.s[L.i]) || L.s[L.i] == '.')) L.i++;
    op.kind = PtxOperand::Imm; op.imm = strtoll(L.s.substr(b, L.i - b).c_str(), nullptr, 0);
    return true;
  }
  if (isalpha((unsigned char)c) || c == '_') {  // a register declared inside the block (`.reg .pred p;`)
    size_t b = L.i;
    while (L.i < L.s.size() && (isalnum((unsigned char)L.s[L.i]) || L.s[L.i] == '_')) L.i++;
    op.kind = PtxOperand::Named; op.name = L.s.substr(b, L.i - b);
    return true;
  }
  err = std::string("unexpected character '") + c + "' in operand";
  return false;
}
} // namespace

bool parsePtxAsm(const std::string& text, std::vector<PtxInstr>& out, std::string& error) {
  Lexer L(text);
  while (!L.eof()) {
    if (L.accept(';') || L.accept('{') || L.accept('}')) continue;   // scopes carry no meaning here
    if (L.peek() == '.') {   // a directive (`.reg .pred p;`): dropped, its register shows up as a Named operand
      while (!L.eof() && L.peek() != ';') L.i++;
      continue;
    }
    PtxInstr in;
    if (L.accept('@')) {
      bool neg = L.accept('!');
      PtxOperand p; std::string e;
      if (!parseOperand(L, p, e) || (p.kind != PtxOperand::Reg && p.kind != PtxOperand::Named)) { error = "bad predicate: " + e; return false; }
      if (p.kind == PtxOperand::Reg) in.pred = p.reg; else in.predName = p.name;
      in.predNeg = neg;
    }
    std::string word = L.ident();
    if (word.empty()) { error = "expected mnemonic at offset " + std::to_string(L.i) + " in: " + text; return false; }
    // split on dots: first piece is the mnemonic, the rest are modifiers
    size_t p = 0, d;
    bool first = true;
    while (p <= word.size()) {
      d = word.find('.', p);
      std::string piece = word.substr(p, d == std::string::npos ? std::string::npos : d - p);
      if (!piece.empty()) { if (first) { in.mnemonic = piece; first = false; } else in.mods.push_back(piece); }
      if (d == std::string::npos) break;
      p = d + 1;
    }
    // operands until ';' or end
    if (!L.eof() && L.peek() != ';') {
      while (true) {
        PtxOperand op; std::string e;
        if (!parseOperand(L, op, e)) { error = e + " in: " + text; return false; }
        in.ops.push_back(op);
        if (L.accept(',')) continue;
        break;
      }
    }
    if (!L.eof() && !L.accept(';') && L.peek() != '}') { error = "expected ; after instruction in: " + text; return false; }
    out.push_back(in);
  }
  if (out.empty()) { error = "empty asm"; return false; }
  return true;
}

bool ptxIsPerThreadAlu(const std::string& m) {
  static const char* const names[] = {
    "add", "sub", "mul", "mad", "mul24", "mad24", "sad", "div", "rem", "abs", "neg", "min", "max", "fma", "rcp", "sqrt",
    "rsqrt", "sin", "cos", "lg2", "ex2", "tanh", "copysign", "testp", "cvt", "mov", "and", "or", "xor", "not", "cnot",
    "lop3", "shf", "shl", "shr", "bfe", "bfi", "bfind", "brev", "clz", "popc", "prmt", "fns", "set", "setp", "selp", "slct",
  };
  for (const char* n : names) if (m == n) return true;
  return false;
}

}  // namespace mvcc
