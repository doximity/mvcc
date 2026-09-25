// Same search as hashhunt_gpu.cu, on the CPU. --bench hashes the full 8-digit
// space; default is the 4-digit demo. --threads N (default: hardware concurrency).
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <thread>
#include <vector>
#include "md5.hpp"

static int hunt_range(const uint32_t target[4], int digits, int begin, int end) {
  int found = -1;
  uint8_t msg[8];
  uint32_t h[4];
  for (int cand = begin; cand < end; cand++) {
    to_digits(cand, digits, msg);
    md5_short(msg, digits, h);
    if (h[0] == target[0] && h[1] == target[1] && h[2] == target[2] && h[3] == target[3]) {
      found = cand;
    }
  }
  return found;
}

int main(int argc, char** argv) {
  int bench = 0;
  int threads = (int)std::thread::hardware_concurrency();
  if (threads < 1) threads = 1;
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--bench")) bench = 1;
    else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
    else {
      fprintf(stderr, "usage: hashhunt_cpu [--bench] [--threads N]\n");
      return 2;
    }
  }
  if (threads < 1) threads = 1;

  const int digits = bench ? 8 : 4;
  const int space = bench ? 100000000 : 10000;
  const char* preimage = bench ? "13371337" : "1337";
  const int want = bench ? 13371337 : 1337;

  uint32_t target[4];
  md5_short(reinterpret_cast<const uint8_t*>(preimage), digits, target);
  char digest[33];
  md5_hex(target, digest);
  printf("target  md5(\"%s\") = %s\n", preimage, digest);
  printf("cpu     %d-digit space %d  %d thread%s\n", digits, space, threads, threads == 1 ? "" : "s");

  auto t0 = std::chrono::steady_clock::now();
  int found = -1;
  if (threads == 1) {
    found = hunt_range(target, digits, 0, space);
  } else {
    std::vector<std::thread> pool;
    std::vector<int> hits(threads, -1);
    int chunk = (space + threads - 1) / threads;
    for (int t = 0; t < threads; t++) {
      int b = t * chunk, e = b + chunk;
      if (e > space) e = space;
      if (b >= space) break;
      pool.emplace_back([&, t, b, e]() { hits[t] = hunt_range(target, digits, b, e); });
    }
    for (auto& th : pool) th.join();
    for (int h : hits) if (h >= 0 && (found < 0 || h < found)) found = h;
  }
  auto t1 = std::chrono::steady_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

  printf("cpu     %.3f ms  %.1f MH/s\n", ms, (space / 1e6) / (ms / 1e3));
  if (found < 0) {
    printf("FAIL    no candidate matched\n");
    return 1;
  }
  printf("found   %0*d\n", digits, found);
  printf("%s\n", found == want ? "PASS" : "FAIL");
  return found == want ? 0 : 1;
}
