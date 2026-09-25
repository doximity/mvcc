// cudaFree has to give the memory back, including after the buffer has been
// touched.
//
// Reading an MTLBuffer property retains the receiver under ARC and hands the
// release to an autorelease pool. A runtime loaded as a library is never inside
// anyone else's pool, so without one of its own the release never lands and the
// buffer outlives the free. Nothing fails at the time: cudaFree returns success
// and the pointer really is gone from the allocator. The memory just never comes
// back, and a process that reloads a large corpus a few times dies.
//
// Every resolve of a device pointer reads `contents`, so any copy or fill is
// enough to trigger it. The shapes below are the ones that reach that path.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static size_t used() {
    size_t free_bytes = 0, total = 0;
    cudaMemGetInfo(&free_bytes, &total);
    return total - free_bytes;
}

static double gb(size_t b) { return static_cast<double>(b) / 1073741824.0; }

static int failures = 0;

// One alloc/free cycle with `op` applied in between; the memory has to come back.
static void cycle(const char* label, int op) {
    const size_t n = 512ull << 20;
    const size_t before = used();

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    void* device = nullptr;
    if (cudaMalloc(&device, n) != cudaSuccess) {
        std::printf("  [SKIP] %-28s could not allocate %.2f GB\n", label, gb(n));
        cudaStreamDestroy(stream);
        return;
    }

    void* host = std::malloc(1 << 20);
    std::memset(host, 7, 1 << 20);
    switch (op) {
        case 1: cudaMemsetAsync(device, 0, n, stream); break;
        case 2: cudaMemcpyAsync(device, host, 1 << 20, cudaMemcpyHostToDevice, stream); break;
        case 3:
            // Many small copies: each one resolves the pointer again, so a
            // per-resolve leak shows up multiplied rather than once.
            for (int i = 0; i < 64; ++i) {
                cudaMemcpyAsync(static_cast<char*>(device) + (i << 20), host, 1 << 20, cudaMemcpyHostToDevice, stream);
            }
            break;
        default: break;
    }
    cudaStreamSynchronize(stream);
    cudaFree(device);
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    std::free(host);

    const size_t after = used();
    const size_t leaked = after > before ? after - before : 0;
    // A little slack for anything the runtime keeps on the side; the failure
    // this guards against is the whole allocation staying resident.
    if (leaked > (32ull << 20)) {
        std::printf("  [FAIL] %-28s leaked %.2f GB of a %.2f GB buffer\n", label, gb(leaked), gb(n));
        ++failures;
    } else {
        std::printf("  [ok]   %-28s returned (%.2f GB in flight, %.0f MB retained)\n", label, gb(n),
                    leaked / 1048576.0);
    }
}

int main() {
    cycle("untouched", 0);
    cycle("after memset", 1);
    cycle("after one copy", 2);
    cycle("after 64 copies", 3);

    // Repeated cycles are how the real case fails: each reload of a corpus
    // allocates, fills and frees, and a per-cycle leak compounds until the
    // process is out of memory.
    const size_t baseline = used();
    for (int i = 0; i < 4; ++i) {
        cudaStream_t stream = nullptr;
        cudaStreamCreate(&stream);
        void* p = nullptr;
        cudaMalloc(&p, 512ull << 20);
        cudaMemsetAsync(p, 0, 512ull << 20, stream);
        cudaStreamSynchronize(stream);
        cudaFree(p);
        cudaStreamDestroy(stream);
    }
    const size_t drift = used() > baseline ? used() - baseline : 0;
    if (drift > (32ull << 20)) {
        std::printf("  [FAIL] %-28s grew %.2f GB over 4 cycles\n", "repeated alloc/fill/free", gb(drift));
        ++failures;
    } else {
        std::printf("  [ok]   %-28s flat over 4 cycles (%.0f MB drift)\n", "repeated alloc/fill/free",
                    drift / 1048576.0);
    }

    if (failures != 0) {
        std::printf("free_returns: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("free_returns: all cycles returned their memory\n");
    return 0;
}
