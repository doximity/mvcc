// .mvccq pack reader: mmap the file, register it with the CUDA runtime once (zero-copy on unified memory,
// a staging copy elsewhere), and hand out device pointers per tensor. Includes the tiny JSON reader the
// header needs.
#pragma once
#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

// ------------------------------------------------------------------ minimal JSON

struct Json {
  enum Kind { Null, Bool, Number, String, Array, Object } kind = Null;
  bool b = false;
  double num = 0;
  std::string str;
  std::vector<Json> arr;
  std::map<std::string, Json> obj;

  const Json& operator[](const std::string& k) const {
    auto it = obj.find(k);
    if (it == obj.end()) throw std::runtime_error("json: missing key " + k);
    return it->second;
  }
  bool has(const std::string& k) const { return obj.count(k) != 0; }
  const Json& operator[](size_t i) const { return arr.at(i); }
  int i() const { return (int)num; }
  int64_t i64() const { return (int64_t)num; }
  float f() const { return (float)num; }

  static Json parse(const std::string& s) {
    size_t p = 0;
    Json j = value(s, p);
    return j;
  }

 private:
  static void ws(const std::string& s, size_t& p) { while (p < s.size() && isspace((unsigned char)s[p])) ++p; }
  static Json value(const std::string& s, size_t& p) {
    ws(s, p);
    Json j;
    if (p >= s.size()) throw std::runtime_error("json: eof");
    char c = s[p];
    if (c == '{') {
      j.kind = Object; ++p; ws(s, p);
      if (s[p] == '}') { ++p; return j; }
      while (true) {
        ws(s, p);
        Json k = value(s, p);
        ws(s, p);
        if (s[p] != ':') throw std::runtime_error("json: expected :");
        ++p;
        j.obj[k.str] = value(s, p);
        ws(s, p);
        if (s[p] == ',') { ++p; continue; }
        if (s[p] == '}') { ++p; return j; }
        throw std::runtime_error("json: expected , or }");
      }
    }
    if (c == '[') {
      j.kind = Array; ++p; ws(s, p);
      if (s[p] == ']') { ++p; return j; }
      while (true) {
        j.arr.push_back(value(s, p));
        ws(s, p);
        if (s[p] == ',') { ++p; continue; }
        if (s[p] == ']') { ++p; return j; }
        throw std::runtime_error("json: expected , or ]");
      }
    }
    if (c == '"') {
      j.kind = String; ++p;
      while (p < s.size() && s[p] != '"') {
        if (s[p] == '\\') { ++p; char e = s[p]; j.str += e == 'n' ? '\n' : e == 't' ? '\t' : e; ++p; }
        else j.str += s[p++];
      }
      ++p;
      return j;
    }
    if (s.compare(p, 4, "true") == 0) { j.kind = Bool; j.b = true; p += 4; return j; }
    if (s.compare(p, 5, "false") == 0) { j.kind = Bool; j.b = false; p += 5; return j; }
    if (s.compare(p, 4, "null") == 0) { j.kind = Null; p += 4; return j; }
    j.kind = Number;
    size_t q = p;
    while (q < s.size() && (isdigit((unsigned char)s[q]) || s[q] == '-' || s[q] == '+' || s[q] == '.' || s[q] == 'e' || s[q] == 'E')) ++q;
    j.num = strtod(s.substr(p, q - p).c_str(), nullptr);
    p = q;
    return j;
  }
};

// ------------------------------------------------------------------ pack

struct TensorInfo {
  uint64_t off = 0, nbytes = 0;
  std::string dtype;
  std::vector<int64_t> shape;
};

class Pack {
 public:
  Json config;
  std::map<std::string, TensorInfo> tensors;

  void open(const std::string& path) {
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open " + path);
    struct stat st;
    fstat(fd_, &st);
    size_ = (size_t)st.st_size;
    // MAP_SHARED (read-only): the GPU mapping of a shared file mapping is set up about 2x faster than for a
    // private one (the first command buffer has to make the whole pack resident: ~1.8 s vs ~3.7 s for 20 GB).
    base_ = mmap(nullptr, size_, PROT_READ, MAP_SHARED, fd_, 0);
    if (base_ == MAP_FAILED) throw std::runtime_error("mmap failed");
    const char* p = (const char*)base_;
    if (memcmp(p, "MVCCQ001", 8) != 0) throw std::runtime_error("bad pack magic");
    uint64_t hlen;
    memcpy(&hlen, p + 8, 8);
    Json hdr = Json::parse(std::string(p + 16, hlen));
    config = hdr["config"];
    for (const auto& kv : hdr["tensors"].obj) {
      TensorInfo ti;
      ti.off = (uint64_t)kv.second["off"].num;
      ti.nbytes = (uint64_t)kv.second["nbytes"].num;
      ti.dtype = kv.second["dtype"].str;
      for (const auto& d : kv.second["shape"].arr) ti.shape.push_back(d.i64());
      tensors[kv.first] = ti;
    }
    // Zero-copy: register the whole mapping with the runtime and resolve one device base pointer.
    cudaError_t e = cudaHostRegister(base_, size_, cudaHostRegisterDefault);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaHostRegister: ") + cudaGetErrorString(e));
    void* dev = nullptr;
    e = cudaHostGetDevicePointer(&dev, base_, 0);
    if (e != cudaSuccess) throw std::runtime_error(std::string("cudaHostGetDevicePointer: ") + cudaGetErrorString(e));
    dev_base_ = (uint8_t*)dev;
  }

  bool has(const std::string& name) const { return tensors.count(name) != 0; }
  const TensorInfo& info(const std::string& name) const {
    auto it = tensors.find(name);
    if (it == tensors.end()) throw std::runtime_error("pack: missing tensor " + name);
    return it->second;
  }
  template <typename T> const T* dev(const std::string& name) const { return (const T*)(dev_base_ + info(name).off); }
  const uint8_t* host(const std::string& name) const { return (const uint8_t*)base_ + info(name).off; }
  size_t size() const { return size_; }

  // Touch every page so the first forward pass does not pay the page-fault cost (optional warm-up).
  void prefetch() const {
    madvise(base_, size_, MADV_WILLNEED);
    volatile uint64_t sink = 0;
    const uint8_t* p = (const uint8_t*)base_;
    for (size_t i = 0; i < size_; i += 4096) sink += p[i];
    (void)sink;
  }

 private:
  int fd_ = -1;
  void* base_ = nullptr;
  size_t size_ = 0;
  uint8_t* dev_base_ = nullptr;
};
