// Inline PTX asm parser: turns the asm strings CUDA code embeds into typed
// instructions. Grammar follows the public PTX ISA document. Only the
// mnemonics the lowering supports are accepted; anything else is a hard error
// carrying the original string so the user sees exactly what was rejected.
#pragma once
#include <string>
#include <vector>

namespace mvcc {

struct PtxOperand {
  enum Kind { Reg, Imm, Mem, Vec, Named } kind = Reg;
  int reg = -1;                 // %N for Reg / Mem base
  long long imm = 0;            // immediate or Mem offset
  std::vector<int> regs;        // Vec: {%0, %1, ...}
  bool negated = false;         // !%p
  std::string name;             // Named: `.reg` temp or a PTX special register (`%globaltimer`)
};

struct PtxInstr {
  std::string mnemonic;               // "mma", "cp", "ldmatrix", "lop3", ...
  std::vector<std::string> mods;      // dotted modifiers in order, e.g. {"sync","aligned","m16n8k16","row","col","f32","f16","f16","f32"}
  std::vector<PtxOperand> ops;
  int pred = -1;                      // @%N guard, -1 if none
  bool predNeg = false;
  std::string predName;               // @p guard on a register declared inside the block (pred stays -1)
  std::string full() const;           // mnemonic + mods joined by '.'
  bool hasMod(const std::string& m) const;
  bool predicated() const { return pred >= 0 || !predName.empty(); }
};

// Parses one asm template (may contain several ';'-separated instructions). `{ }` scopes and `.reg` declarations
// are accepted and dropped; a register they declare appears as a Named operand.
// Returns false and fills `error` on failure.
bool parsePtxAsm(const std::string& text, std::vector<PtxInstr>& out, std::string& error);

// Per-thread PTX ALU instructions: no memory, no side effects, no cross-lane communication (safe to speculate, to
// duplicate and to drop the frontend's blanket `convergent` from).
bool ptxIsPerThreadAlu(const std::string& mnemonic);

} // namespace mvcc
