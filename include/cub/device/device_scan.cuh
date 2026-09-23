// mvcc: clean-room subset of the CUB device-scan interface (cub::DeviceScan::ExclusiveScanByKey and the functors it
// takes). Same two-call protocol as CUB: a null temporary-storage pointer sizes the storage and returns.
//
// Segmented (by-key) exclusive scan in three launches: per-tile aggregates, one block scanning the tile
// aggregates, per-tile rescan with the carried-in prefix. Tiles are 256 threads x 8 items. The segment state is a
// triple (flag: a segment starts inside the range, value: fold of the items since the last segment start, nonempty)
// and the combine is associative, so the tile decomposition is exact for any associative scan_op.
#ifndef MVCC_CUB_DEVICE_DEVICE_SCAN_CUH
#define MVCC_CUB_DEVICE_DEVICE_SCAN_CUH
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace cub {

struct Sum { template <class T, class U> __host__ __device__ __forceinline__ auto operator()(const T& a, const U& b) const -> decltype(a + b) { return a + b; } };
struct Max { template <class T> __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const { return a < b ? b : a; } };
struct Min { template <class T> __host__ __device__ __forceinline__ T operator()(const T& a, const T& b) const { return b < a ? b : a; } };
struct Equality { template <class T, class U> __host__ __device__ __forceinline__ bool operator()(const T& a, const U& b) const { return a == b; } };
struct Inequality { template <class T, class U> __host__ __device__ __forceinline__ bool operator()(const T& a, const U& b) const { return a != b; } };

namespace mvcc_detail {

template <class V> struct SegState { int flag; V value; int nonempty; };

template <class V, class Op>
__device__ __forceinline__ SegState<V> seg_combine(const SegState<V>& a, const SegState<V>& b, Op op) {
    SegState<V> r;
    r.flag = a.flag | b.flag;
    r.nonempty = a.nonempty | b.nonempty;
    if (b.flag) r.value = b.value;
    else if (!a.nonempty) r.value = b.value;
    else if (!b.nonempty) r.value = a.value;
    else r.value = op(a.value, b.value);
    return r;
}

constexpr int kThreads = 256;
constexpr int kItems = 8;
constexpr int kTile = kThreads * kItems;

// Fold the tile's threads' states into one; thread 0 leaves the block aggregate in sh[0] and every thread gets
// its exclusive prefix (states of the threads before it) in `excl`.
template <class V, class Op>
__device__ __forceinline__ SegState<V> block_scan_states(SegState<V> mine, SegState<V>* sh, Op op, SegState<V>& excl) {
    sh[threadIdx.x] = mine;
    __syncthreads();
    if (threadIdx.x == 0) {
        SegState<V> acc; acc.flag = 0; acc.nonempty = 0; acc.value = V();  // exclusive prefix of thread 0 is empty
        for (int t = 0; t < kThreads; t++) {
            SegState<V> cur = sh[t];
            sh[t] = acc;
            acc = seg_combine(acc, cur, op);
        }
        sh[kThreads] = acc;
    }
    __syncthreads();
    excl = sh[threadIdx.x];
    SegState<V> total = sh[kThreads];
    __syncthreads();
    return total;
}

template <class KeyIt, class ValIt, class V, class Op, class Eq>
__global__ void __launch_bounds__(kThreads) sbk_tile_aggregate(KeyIt keys, ValIt vals, int n, Op op, Eq eq, SegState<V>* tiles) {
    __shared__ SegState<V> sh[kThreads + 1];
    const int base = blockIdx.x * kTile + threadIdx.x * kItems;
    SegState<V> s; s.flag = 0; s.nonempty = 0; s.value = V();
    for (int j = 0; j < kItems; j++) {
        const int i = base + j;
        if (i >= n) break;
        SegState<V> e; e.nonempty = 1; e.value = static_cast<V>(vals[i]);
        e.flag = (i == 0) || !eq(keys[i - 1], keys[i]);
        s = seg_combine(s, e, op);
    }
    SegState<V> excl;
    SegState<V> total = block_scan_states(s, sh, op, excl);
    if (threadIdx.x == 0) tiles[blockIdx.x] = total;
}

template <class V, class Op>
__global__ void sbk_scan_tiles(SegState<V>* tiles, int ntiles, Op op) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    SegState<V> acc; acc.flag = 0; acc.nonempty = 0; acc.value = V();
    for (int t = 0; t < ntiles; t++) {
        SegState<V> cur = tiles[t];
        tiles[t] = acc;
        acc = seg_combine(acc, cur, op);
    }
}

template <class KeyIt, class ValIt, class OutIt, class V, class Op, class Eq>
__global__ void __launch_bounds__(kThreads) sbk_tile_scan(KeyIt keys, ValIt vals, OutIt out, int n, Op op, Eq eq, V init, const SegState<V>* tiles) {
    __shared__ SegState<V> sh[kThreads + 1];
    const int base = blockIdx.x * kTile + threadIdx.x * kItems;
    SegState<V> s; s.flag = 0; s.nonempty = 0; s.value = V();
    for (int j = 0; j < kItems; j++) {
        const int i = base + j;
        if (i >= n) break;
        SegState<V> e; e.nonempty = 1; e.value = static_cast<V>(vals[i]);
        e.flag = (i == 0) || !eq(keys[i - 1], keys[i]);
        s = seg_combine(s, e, op);
    }
    SegState<V> excl;
    block_scan_states(s, sh, op, excl);
    SegState<V> run = seg_combine(tiles[blockIdx.x], excl, op);
    for (int j = 0; j < kItems; j++) {
        const int i = base + j;
        if (i >= n) break;
        const bool starts = (i == 0) || !eq(keys[i - 1], keys[i]);
        V v = static_cast<V>(vals[i]);
        if (starts) { out[i] = init; run.flag = 1; run.nonempty = 1; run.value = v; }
        else { out[i] = run.nonempty ? op(init, run.value) : init; run.value = run.nonempty ? op(run.value, v) : v; run.nonempty = 1; }
    }
}

}  // namespace mvcc_detail

struct DeviceScan {
    template <class KeyIt, class ValIt, class OutIt, class ScanOp, class InitT, class EqOp = Equality>
    static cudaError_t ExclusiveScanByKey(void* d_temp_storage, size_t& temp_storage_bytes, KeyIt d_keys_in, ValIt d_values_in,
                                          OutIt d_values_out, ScanOp scan_op, InitT init_value, int num_items,
                                          EqOp equality_op = EqOp(), cudaStream_t stream = 0) {
        using namespace mvcc_detail;
        using V = InitT;
        const int ntiles = num_items > 0 ? (num_items + kTile - 1) / kTile : 1;
        const size_t need = static_cast<size_t>(ntiles) * sizeof(SegState<V>);
        if (d_temp_storage == nullptr) { temp_storage_bytes = need; return cudaSuccess; }
        if (temp_storage_bytes < need) return cudaErrorInvalidValue;
        if (num_items <= 0) return cudaSuccess;
        SegState<V>* tiles = static_cast<SegState<V>*>(d_temp_storage);
        sbk_tile_aggregate<KeyIt, ValIt, V, ScanOp, EqOp><<<ntiles, kThreads, 0, stream>>>(d_keys_in, d_values_in, num_items, scan_op, equality_op, tiles);
        sbk_scan_tiles<V, ScanOp><<<1, 1, 0, stream>>>(tiles, ntiles, scan_op);
        sbk_tile_scan<KeyIt, ValIt, OutIt, V, ScanOp, EqOp><<<ntiles, kThreads, 0, stream>>>(d_keys_in, d_values_in, d_values_out, num_items, scan_op, equality_op, static_cast<V>(init_value), tiles);
        return cudaGetLastError();
    }
};

}  // namespace cub

#endif // MVCC_CUB_DEVICE_DEVICE_SCAN_CUH
