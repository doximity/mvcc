// Host-side byte-level BPE tokenizer (Qwen2 family) driven by the tables convert.py exports.
//
// tokenizer.bin: "MVCCT001" | u32 vocab | u32 merges | u32 specials | u32 pad
//                pieces: (u16 len, bytes) * vocab
//                merges: (u32 a, u32 b, u32 merged) * merges        (rank = position)
//                specials: (u32 id, u16 len, bytes) * specials
//                4 range tables (letters, marks, numbers, white space): u32 n, (u32 lo, u32 hi) * n
//
// Pre-tokenization is a hand-rolled matcher for the Qwen2 split regex:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// Input is expected to already be NFC-normalized (true for ASCII / typical prompts).
#pragma once
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

class Tokenizer {
 public:
  std::vector<std::string> pieces;
  std::unordered_map<uint64_t, std::pair<int, int>> merge;  // (a<<32|b) -> (rank, merged id)
  std::vector<std::pair<int, std::string>> specials;         // id, text
  std::unordered_map<std::string, int> special_ids;
  int byte_id[256];
  std::vector<std::pair<uint32_t, uint32_t>> letters, marks, numbers, spaces;

  void load(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open tokenizer " + path);
    char magic[8];
    if (fread(magic, 1, 8, f) != 8 || memcmp(magic, "MVCCT001", 8) != 0) throw std::runtime_error("bad tokenizer magic");
    uint32_t hdr[4];
    fread(hdr, 4, 4, f);
    pieces.resize(hdr[0]);
    for (uint32_t i = 0; i < hdr[0]; ++i) {
      uint16_t n;
      fread(&n, 2, 1, f);
      pieces[i].resize(n);
      if (n) fread(&pieces[i][0], 1, n, f);
    }
    for (uint32_t i = 0; i < hdr[1]; ++i) {
      uint32_t m[3];
      fread(m, 4, 3, f);
      merge[((uint64_t)m[0] << 32) | m[1]] = {(int)i, (int)m[2]};
    }
    for (uint32_t i = 0; i < hdr[2]; ++i) {
      uint32_t id;
      uint16_t n;
      fread(&id, 4, 1, f);
      fread(&n, 2, 1, f);
      std::string s(n, '\0');
      if (n) fread(&s[0], 1, n, f);
      specials.push_back({(int)id, s});
      special_ids[s] = (int)id;
    }
    auto read_table = [&](std::vector<std::pair<uint32_t, uint32_t>>& t) {
      uint32_t n;
      fread(&n, 4, 1, f);
      t.resize(n);
      if (n) fread(t.data(), 8, n, f);
    };
    read_table(letters);
    read_table(marks);
    read_table(numbers);
    read_table(spaces);
    fclose(f);
    for (int b = 0; b < 256; ++b) byte_id[b] = -1;
    for (size_t i = 0; i < pieces.size(); ++i)
      if (pieces[i].size() == 1) byte_id[(uint8_t)pieces[i][0]] = (int)i;
    for (int b = 0; b < 256; ++b)
      if (byte_id[b] < 0) throw std::runtime_error("vocab lacks single-byte token");
  }

  int special(const std::string& s) const {
    auto it = special_ids.find(s);
    return it == special_ids.end() ? -1 : it->second;
  }

  // ---- unicode classes
  static bool in_table(const std::vector<std::pair<uint32_t, uint32_t>>& t, uint32_t cp) {
    size_t lo = 0, hi = t.size();
    while (lo < hi) {
      size_t mid = (lo + hi) / 2;
      if (cp < t[mid].first) hi = mid;
      else if (cp > t[mid].second) lo = mid + 1;
      else return true;
    }
    return false;
  }
  bool is_L(uint32_t c) const { return in_table(letters, c); }
  bool is_M(uint32_t c) const { return in_table(marks, c); }
  bool is_N(uint32_t c) const { return in_table(numbers, c); }
  bool is_S(uint32_t c) const { return in_table(spaces, c); }
  static bool is_nl(uint32_t c) { return c == '\r' || c == '\n'; }

  // ---- utf8
  static std::vector<uint32_t> decode_utf8(const std::string& s, std::vector<size_t>& offs) {
    std::vector<uint32_t> cps;
    size_t i = 0;
    while (i < s.size()) {
      offs.push_back(i);
      uint8_t c = s[i];
      uint32_t cp;
      int n;
      if (c < 0x80) { cp = c; n = 1; }
      else if ((c >> 5) == 6) { cp = c & 0x1f; n = 2; }
      else if ((c >> 4) == 14) { cp = c & 0x0f; n = 3; }
      else if ((c >> 3) == 30) { cp = c & 0x07; n = 4; }
      else { cp = 0xfffd; n = 1; }
      for (int k = 1; k < n && i + k < s.size(); ++k) cp = (cp << 6) | (s[i + k] & 0x3f);
      cps.push_back(cp);
      i += n;
    }
    offs.push_back(s.size());
    return cps;
  }

  // ---- pre-tokenizer: returns byte spans [start, end)
  std::vector<std::pair<size_t, size_t>> pretokenize(const std::string& s) const {
    std::vector<size_t> offs;
    std::vector<uint32_t> cp = decode_utf8(s, offs);
    const size_t n = cp.size();
    std::vector<std::pair<size_t, size_t>> out;
    size_t i = 0;
    auto lower = [](uint32_t c) { return (c >= 'A' && c <= 'Z') ? c + 32 : c; };
    while (i < n) {
      size_t j = i;
      // 1. contractions
      if (cp[i] == '\'' && i + 1 < n) {
        uint32_t a = lower(cp[i + 1]);
        uint32_t b = i + 2 < n ? lower(cp[i + 2]) : 0;
        if (a == 's' || a == 't' || a == 'm' || a == 'd') j = i + 2;
        else if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) j = i + 3;
        if (j > i) { out.push_back({offs[i], offs[j]}); i = j; continue; }
      }
      // 2. [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+
      {
        size_t k = i;
        bool prefix_ok = !is_nl(cp[k]) && !is_L(cp[k]) && !is_N(cp[k]);
        size_t start_letters = k + (prefix_ok ? 1 : 0);
        size_t m = start_letters;
        while (m < n && (is_L(cp[m]) || is_M(cp[m]))) ++m;
        if (m > start_letters) { out.push_back({offs[i], offs[m]}); i = m; continue; }
        if (prefix_ok && is_M(cp[k])) {  // backtrack: optional not taken, the char itself is a mark
          m = k;
          while (m < n && (is_L(cp[m]) || is_M(cp[m]))) ++m;
          out.push_back({offs[i], offs[m]}); i = m; continue;
        }
      }
      // 3. \p{N}
      if (is_N(cp[i])) { out.push_back({offs[i], offs[i + 1]}); i += 1; continue; }
      // 4.  ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*
      {
        size_t k = i;
        if (cp[k] == ' ') ++k;
        size_t m = k;
        while (m < n && !is_S(cp[m]) && !is_L(cp[m]) && !is_M(cp[m]) && !is_N(cp[m])) ++m;
        if (m > k) {
          while (m < n && is_nl(cp[m])) ++m;
          out.push_back({offs[i], offs[m]}); i = m; continue;
        }
      }
      // 5/6/7. whitespace runs
      if (is_S(cp[i])) {
        size_t m = i;
        while (m < n && is_S(cp[m])) ++m;
        // 5. \s*[\r\n]+ : up to and including the last newline in the run
        size_t last_nl = i;
        bool any = false;
        for (size_t q = i; q < m; ++q) if (is_nl(cp[q])) { last_nl = q; any = true; }
        if (any) { out.push_back({offs[i], offs[last_nl + 1]}); i = last_nl + 1; continue; }
        // 6. \s+(?!\S) : whole run at end of text, else run minus one
        if (m == n) { out.push_back({offs[i], offs[m]}); i = m; continue; }
        if (m - i > 1) { out.push_back({offs[i], offs[m - 1]}); i = m - 1; continue; }
        // 7. \s+
        out.push_back({offs[i], offs[m]}); i = m; continue;
      }
      // fallback (should not happen): single code point
      out.push_back({offs[i], offs[i + 1]});
      i += 1;
    }
    return out;
  }

  // ---- BPE on one pre-token
  void bpe(const std::string& piece, std::vector<int>& out) const {
    std::vector<int> syms;
    syms.reserve(piece.size());
    for (unsigned char c : piece) syms.push_back(byte_id[c]);
    while (syms.size() > 1) {
      int best_rank = 0x7fffffff, best_pos = -1, best_id = -1;
      for (size_t i = 0; i + 1 < syms.size(); ++i) {
        auto it = merge.find(((uint64_t)(uint32_t)syms[i] << 32) | (uint32_t)syms[i + 1]);
        if (it != merge.end() && it->second.first < best_rank) {
          best_rank = it->second.first;
          best_pos = (int)i;
          best_id = it->second.second;
        }
      }
      if (best_pos < 0) break;
      syms[best_pos] = best_id;
      syms.erase(syms.begin() + best_pos + 1);
    }
    out.insert(out.end(), syms.begin(), syms.end());
  }

  // ---- encode text (special tokens in the text are recognized)
  std::vector<int> encode(const std::string& text) const {
    std::vector<int> out;
    size_t i = 0, seg = 0;
    auto flush = [&](size_t end) {
      if (end > seg) {
        std::string s = text.substr(seg, end - seg);
        for (auto [a, b] : pretokenize(s)) bpe(s.substr(a, b - a), out);
      }
    };
    while (i < text.size()) {
      bool matched = false;
      if (text[i] == '<') {
        for (const auto& sp : specials) {
          if (text.compare(i, sp.second.size(), sp.second) == 0) {
            flush(i);
            out.push_back(sp.first);
            i += sp.second.size();
            seg = i;
            matched = true;
            break;
          }
        }
      }
      if (!matched) ++i;
    }
    flush(text.size());
    return out;
  }

  const std::string& decode(int id) const {
    static const std::string empty;
    return (id >= 0 && (size_t)id < pieces.size()) ? pieces[id] : empty;
  }
};

// Streams decoded bytes to stdout, holding back incomplete UTF-8 sequences.
struct Utf8Printer {
  std::string pending;
  void feed(const std::string& bytes) {
    pending += bytes;
    size_t i = 0;
    while (i < pending.size()) {
      uint8_t c = pending[i];
      size_t n = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 1;
      if (i + n > pending.size()) break;
      i += n;
    }
    fwrite(pending.data(), 1, i, stdout);
    fflush(stdout);
    pending.erase(0, i);
  }
};
