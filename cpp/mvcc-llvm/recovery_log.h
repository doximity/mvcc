// Which tensor-recovery notes print by default: declines and problems. MVCC_TENSOR_DIAG=1 or -v prints all of them.
#pragma once
#include <string>

namespace mvcc {

inline bool recoveryNoteIsDeclineOrProblem(const std::string& note) {
  if (note.find("kept:") != std::string::npos) return true;
  if (note.find("refused") != std::string::npos) return true;
  if (note.find("not recovered") != std::string::npos) return true;
  if (note.find("graph: MVCC_GRAPH") != std::string::npos) return true;
  return false;
}

inline bool shouldLogRecoveryNote(const std::string& note, bool verbose) {
  return verbose || recoveryNoteIsDeclineOrProblem(note);
}

}  // namespace mvcc
