# hashhunt

Brute-force search is a staple CUDA workload, and code of that kind usually exists only as CUDA.

`hashhunt_gpu.cu` MD5s every n-digit string until one matches a known digest; mvcc compiles it unchanged. `hashhunt_cpu.cpp` is the same search in host C++.

```bash
export PATH=/path/to/mvcc/toolkit/bin:$PATH
./run.sh
```

```text
target  md5("1337") = e48e13207341b6bffb7fb1622282247b
found   1337
PASS
```

## Bench

Full scan of all 100,000,000 8-digit strings (`md5("13371337")`). CPU uses the shared `md5.hpp` loop (`-O3`). The GPU bench kernel is a specialized 8-digit path: register-only unrolled MD5, ASCII increment, 64 candidates per thread (`nvcc -O3`). Timed on an M5 Pro (18 threads), macOS 26.5.1.

| impl | time | throughput | vs CPU 1 thread |
| --- | ---: | ---: | ---: |
| GPU (mvcc / Metal) | 13.5 ms | 7415 MH/s | 1353x |
| CPU C++, 18 threads | 1303 ms | 76.7 MH/s | 14x |
| CPU C++, 8 threads | 2362 ms | 42.3 MH/s | 7.7x |
| CPU C++, 4 threads | 4677 ms | 21.4 MH/s | 3.9x |
| CPU C++, 2 threads | 9213 ms | 10.9 MH/s | 2.0x |
| CPU C++, 1 thread | 18249 ms | 5.5 MH/s | 1x |

```bash
./hashhunt_gpu --bench
for n in 1 2 4 8 18; do ./hashhunt_cpu --bench --threads $n; done
```
