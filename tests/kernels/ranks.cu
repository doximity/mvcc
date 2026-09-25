// Multi-rank surface test: what a sharded model's ranks ask of the runtime.
//
// Ranks live on devices 0..world-1, one stream each. Under MVCC_DEVICES=N the runtime presents N logical devices
// on the one Metal GPU; under the default (one device) the program takes the same route a CUDA program takes on
// one GPU: every rank on device 0. Either way this must hold:
//   - cudaSetDevice / cudaGetDevice select per thread; every logical device is a peer of every other;
//   - a kernel takes a by-value struct of per-rank pointers (Bufs<T>) and sums the ranks' buffers into each;
//   - a stream capture forks from rank 0's stream to the other ranks' streams through events and joins back,
//     so one graph launch covers every rank's work, and an external event record inside the capture is a node
//     that records on every replay;
//   - peer copies between logical devices are copies;
//   - NCCL (nccl.h, libnccl = libcudart): communicators over the ranks' devices, an all-gather posted per rank
//     inside a group on the ranks' streams (the deterministic all-reduce: gather, then every rank sums the slabs
//     in rank order), broadcast, send/recv, and AllReduce / Reduce / ReduceScatter as gather-and-sum or
//     gather-and-avg (ncclSum / ncclAvg on f32).
//
//   MVCC_DEVICES=2 nvcc -O2 -std=c++17 -o ranks ranks.cu && MVCC_DEVICES=2 ./ranks
#include <cuda_runtime.h>
#include <nccl.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CK(x)                                                                                        \
  do {                                                                                               \
    cudaError_t e_ = (x);                                                                            \
    if (e_ != cudaSuccess) {                                                                         \
      fprintf(stderr, "%s failed: %s (%s:%d)\n", #x, cudaGetErrorString(e_), __FILE__, __LINE__);    \
      return 1;                                                                                      \
    }                                                                                                \
  } while (0)

constexpr int kMaxWorld = 8;
template <typename T> struct Bufs { T* p[kMaxWorld]; };

// every rank's buffer becomes the rank-ordered sum (the same bits on every rank)
template <typename T>
__global__ void sum_ranks_kernel(Bufs<T> b, int world, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float acc = 0.f;
  for (int r = 0; r < world; ++r) acc += b.p[r][i];
  for (int r = 0; r < world; ++r) b.p[r][i] = acc;
}
// one rank's layer: x = x * 2 + rank
__global__ void layer_kernel(float* x, int rank, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) x[i] = x[i] * 2.f + static_cast<float>(rank);
}
// the gathered slabs [world][count] summed in rank order into out
__global__ void sum_gathered_kernel(const float* gathered, float* out, int world, size_t count) {
  const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  float acc = 0.f;
  for (int r = 0; r < world; ++r) acc += gathered[static_cast<size_t>(r) * count + i];
  out[i] = acc;
}
#define NCK(x)                                                                                       \
  do {                                                                                               \
    ncclResult_t e_ = (x);                                                                           \
    if (e_ != ncclSuccess) {                                                                         \
      fprintf(stderr, "%s failed: %s (%s:%d)\n", #x, ncclGetErrorString(e_), __FILE__, __LINE__);    \
      return 1;                                                                                      \
    }                                                                                                \
  } while (0)

int main() {
  int devices = 0;
  CK(cudaGetDeviceCount(&devices));
  const int world = 4;
  std::vector<int> dev(world);
  for (int r = 0; r < world; ++r) dev[r] = devices >= world ? r : 0;
  const bool multi = devices >= world;
  printf("devices %d, world %d (%s)\n", devices, world, multi ? "one logical device per rank" : "every rank on device 0");

  // device selection and peer access
  for (int r = 0; r < world; ++r) {
    CK(cudaSetDevice(dev[r]));
    int cur = -1;
    CK(cudaGetDevice(&cur));
    if (cur != dev[r]) { fprintf(stderr, "cudaGetDevice: %d after cudaSetDevice(%d)\n", cur, dev[r]); return 1; }
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev[r]));
    int sms = 0;
    CK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev[r]));
    if (sms <= 0) { fprintf(stderr, "device %d: %d SMs\n", dev[r], sms); return 1; }
    for (int q = 0; q < world; ++q) {
      if (dev[q] == dev[r]) continue;
      int can = 0;
      CK(cudaDeviceCanAccessPeer(&can, dev[r], dev[q]));
      if (!can) { fprintf(stderr, "device %d cannot access device %d\n", dev[r], dev[q]); return 1; }
      cudaError_t e = cudaDeviceEnablePeerAccess(dev[q], 0);
      if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) { fprintf(stderr, "peer access %d -> %d: %s\n", dev[r], dev[q], cudaGetErrorString(e)); return 1; }
      cudaGetLastError();
    }
  }
  if (cudaSetDevice(devices) != cudaErrorInvalidDevice) { fprintf(stderr, "cudaSetDevice(%d) did not fail\n", devices); return 1; }
  cudaGetLastError();

  const size_t count = 1 << 16;
  const int threads = 256;
  const unsigned blocks = static_cast<unsigned>((count + threads - 1) / threads);
  std::vector<float*> x(world);
  std::vector<cudaStream_t> streams(world);
  std::vector<cudaEvent_t> events(world);
  for (int r = 0; r < world; ++r) {
    CK(cudaSetDevice(dev[r]));
    CK(cudaMalloc(&x[r], count * sizeof(float)));
    CK(cudaStreamCreate(&streams[r]));
    CK(cudaEventCreateWithFlags(&events[r], cudaEventDisableTiming));
  }
  cudaEvent_t mark;
  CK(cudaSetDevice(dev[0]));
  CK(cudaEventCreate(&mark));

  // the step every rank runs: a layer on its stream, then the all-reduce (rank 0's stream, after every rank)
  Bufs<float> b;
  for (int r = 0; r < kMaxWorld; ++r) b.p[r] = r < world ? x[r] : nullptr;
  auto step = [&](bool capturing) {
    if (multi) {
      CK(cudaEventRecord(events[0], streams[0]));
      for (int r = 1; r < world; ++r) { CK(cudaSetDevice(dev[r])); CK(cudaStreamWaitEvent(streams[r], events[0], 0)); }
    }
    for (int r = 0; r < world; ++r) {
      CK(cudaSetDevice(dev[r]));
      layer_kernel<<<blocks, threads, 0, multi ? streams[r] : streams[0]>>>(x[r], r, count);
    }
    if (multi) {
      for (int r = 1; r < world; ++r) {
        CK(cudaSetDevice(dev[r]));
        CK(cudaEventRecord(events[r], streams[r]));
        CK(cudaSetDevice(dev[0]));
        CK(cudaStreamWaitEvent(streams[0], events[r], 0));
      }
    }
    CK(cudaSetDevice(dev[0]));
    if (capturing) CK(cudaEventRecordWithFlags(mark, streams[0], cudaEventRecordExternal));
    else CK(cudaEventRecord(mark, streams[0]));
    sum_ranks_kernel<float><<<blocks, threads, 0, streams[0]>>>(b, world, count);
    return 0;
  };
  // CPU model of the same
  std::vector<std::vector<float>> ref(world, std::vector<float>(count, 1.f));
  auto ref_step = [&]() {
    for (int r = 0; r < world; ++r) for (size_t i = 0; i < count; ++i) ref[r][i] = ref[r][i] * 2.f + static_cast<float>(r);
    for (size_t i = 0; i < count; ++i) {
      float acc = 0.f;
      for (int r = 0; r < world; ++r) acc += ref[r][i];
      for (int r = 0; r < world; ++r) ref[r][i] = acc;
    }
  };
  auto check = [&](const char* what) {
    std::vector<float> h(count);
    for (int r = 0; r < world; ++r) {
      CK(cudaSetDevice(dev[r]));
      CK(cudaMemcpy(h.data(), x[r], count * sizeof(float), cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < count; ++i) if (h[i] != ref[r][i]) { fprintf(stderr, "%s: rank %d [%zu] = %g, want %g\n", what, r, i, h[i], ref[r][i]); return 1; }
    }
    return 0;
  };
  for (int r = 0; r < world; ++r) { std::vector<float> one(count, 1.f); CK(cudaSetDevice(dev[r])); CK(cudaMemcpy(x[r], one.data(), count * sizeof(float), cudaMemcpyHostToDevice)); }

  // eager
  if (step(false)) return 1;
  ref_step();
  CK(cudaSetDevice(dev[0]));
  CK(cudaStreamSynchronize(streams[0]));
  if (check("eager")) return 1;
  printf("eager step: ok\n");

  // captured: rank 0's stream is the origin; the others fork and join
  CK(cudaSetDevice(dev[0]));
  CK(cudaStreamBeginCapture(streams[0], cudaStreamCaptureModeThreadLocal));
  cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
  CK(cudaStreamIsCapturing(streams[0], &cs));
  if (cs != cudaStreamCaptureStatusActive) { fprintf(stderr, "stream 0 not capturing\n"); return 1; }
  if (step(true)) return 1;
  cudaGraph_t graph = nullptr;
  CK(cudaStreamEndCapture(streams[0], &graph));
  size_t nodes = 0;
  CK(cudaGraphGetNodes(graph, nullptr, &nodes));
  cudaGraphExec_t exec = nullptr;
  CK(cudaGraphInstantiate(&exec, graph, 0));
  CK(cudaGraphDestroy(graph));
  const int replays = 3;
  for (int k = 0; k < replays; ++k) {
    CK(cudaGraphLaunch(exec, streams[0]));
    ref_step();
  }
  CK(cudaStreamSynchronize(streams[0]));
  if (check("graph")) return 1;
  // the external record inside the graph recorded on replay: the event is complete and queryable
  CK(cudaEventSynchronize(mark));
  CK(cudaEventQuery(mark));
  printf("captured step (%zu nodes) x %d replays: ok\n", nodes, replays);

  // peer copy between ranks' devices, then a device-side check
  if (world > 1) {
    std::vector<float> h(count);
    CK(cudaSetDevice(dev[1]));
    CK(cudaMemcpyPeerAsync(x[1], dev[1], x[0], dev[0], count * sizeof(float), streams[multi ? 1 : 0]));
    CK(cudaStreamSynchronize(streams[multi ? 1 : 0]));
    CK(cudaMemcpy(h.data(), x[1], count * sizeof(float), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) if (h[i] != ref[0][i]) { fprintf(stderr, "peer copy: [%zu] = %g, want %g\n", i, h[i], ref[0][i]); return 1; }
    printf("peer copy: ok\n");
  }

  // NCCL: the deterministic all-reduce (all-gather in a group, then the rank-ordered sum on every rank)
  {
    int v = 0;
    NCK(ncclGetVersion(&v));
    std::vector<ncclComm_t> comms(world);
    NCK(ncclCommInitAll(comms.data(), world, dev.data()));
    for (int r = 0; r < world; ++r) {
      int n = 0, d = -1, rk = -1;
      NCK(ncclCommCount(comms[r], &n)); NCK(ncclCommCuDevice(comms[r], &d)); NCK(ncclCommUserRank(comms[r], &rk));
      if (n != world || d != dev[r] || rk != r) { fprintf(stderr, "comm %d: count %d device %d rank %d\n", r, n, d, rk); return 1; }
    }
    // a fresh layer step on every rank's own stream, then the all-reduce through NCCL
    std::vector<float*> gathered(world);
    for (int r = 0; r < world; ++r) {
      CK(cudaSetDevice(dev[r]));
      CK(cudaMalloc(&gathered[r], static_cast<size_t>(world) * count * sizeof(float)));
      layer_kernel<<<blocks, threads, 0, streams[r]>>>(x[r], r, count);
    }
    NCK(ncclGroupStart());
    for (int r = 0; r < world; ++r) { CK(cudaSetDevice(dev[r])); NCK(ncclAllGather(x[r], gathered[r], count, ncclFloat32, comms[r], streams[r])); }
    NCK(ncclGroupEnd());
    for (int r = 0; r < world; ++r) {
      CK(cudaSetDevice(dev[r]));
      sum_gathered_kernel<<<blocks, threads, 0, streams[r]>>>(gathered[r], x[r], world, count);
    }
    ref_step();
    for (int r = 0; r < world; ++r) { CK(cudaSetDevice(dev[r])); CK(cudaStreamSynchronize(streams[r])); }
    if (check("nccl all-gather + sum")) return 1;
    // broadcast from rank 1 into every rank's slab 0, in a group
    NCK(ncclGroupStart());
    for (int r = 0; r < world; ++r) { CK(cudaSetDevice(dev[r])); NCK(ncclBroadcast(x[1], gathered[r], count, ncclFloat32, 1, comms[r], streams[r])); }
    NCK(ncclGroupEnd());
    // send/recv: rank 0's x to rank world-1's slab 1
    NCK(ncclGroupStart());
    CK(cudaSetDevice(dev[0]));
    NCK(ncclSend(x[0], count, ncclFloat32, world - 1, comms[0], streams[0]));
    CK(cudaSetDevice(dev[world - 1]));
    NCK(ncclRecv(gathered[world - 1] + count, count, ncclFloat32, 0, comms[world - 1], streams[world - 1]));
    NCK(ncclGroupEnd());
    {
      std::vector<float> h(count);
      for (int r = 0; r < world; ++r) {
        CK(cudaSetDevice(dev[r]));
        CK(cudaStreamSynchronize(streams[r]));
        CK(cudaMemcpy(h.data(), gathered[r], count * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) if (h[i] != ref[1][i]) { fprintf(stderr, "nccl broadcast: rank %d [%zu] = %g, want %g\n", r, i, h[i], ref[1][i]); return 1; }
      }
      CK(cudaSetDevice(dev[world - 1]));
      CK(cudaMemcpy(h.data(), gathered[world - 1] + count, count * sizeof(float), cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < count; ++i) if (h[i] != ref[0][i]) { fprintf(stderr, "nccl send/recv: [%zu] = %g, want %g\n", i, h[i], ref[0][i]); return 1; }
    }
    // AllReduce is gather-and-sum (ncclSum, f32): each recv[0..count) is the sum of every rank's x
    if (world > 1) {
      std::vector<float> href(count, 0.f), hx(count);
      for (int r = 0; r < world; ++r) {
        CK(cudaSetDevice(dev[r]));
        CK(cudaStreamSynchronize(streams[r]));
        CK(cudaMemcpy(hx.data(), x[r], count * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) href[i] += hx[i];
      }
      NCK(ncclGroupStart());
      for (int r = 0; r < world; ++r) { CK(cudaSetDevice(dev[r])); NCK(ncclAllReduce(x[r], gathered[r], count, ncclFloat32, ncclSum, comms[r], streams[r])); }
      NCK(ncclGroupEnd());
      for (int r = 0; r < world; ++r) {
        CK(cudaSetDevice(dev[r]));
        CK(cudaStreamSynchronize(streams[r]));
        CK(cudaMemcpy(hx.data(), gathered[r], count * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) if (hx[i] != href[i]) {
          fprintf(stderr, "ncclAllReduce: rank %d [%zu] = %g, want %g\n", r, i, hx[i], href[i]);
          return 1;
        }
      }
      // AllReduce ncclAvg: each recv is href / world
      NCK(ncclGroupStart());
      for (int r = 0; r < world; ++r) { CK(cudaSetDevice(dev[r])); NCK(ncclAllReduce(x[r], gathered[r], count, ncclFloat32, ncclAvg, comms[r], streams[r])); }
      NCK(ncclGroupEnd());
      for (int r = 0; r < world; ++r) {
        CK(cudaSetDevice(dev[r]));
        CK(cudaStreamSynchronize(streams[r]));
        CK(cudaMemcpy(hx.data(), gathered[r], count * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
          float want = href[i] / (float)world;
          if (fabsf(hx[i] - want) > 1e-5f * (fabsf(want) + 1.f)) {
            fprintf(stderr, "ncclAllReduce avg: rank %d [%zu] = %g, want %g\n", r, i, hx[i], want);
            return 1;
          }
        }
      }
      // Reduce to root 0: only that rank's recv is the sum
      NCK(ncclGroupStart());
      for (int r = 0; r < world; ++r) { CK(cudaSetDevice(dev[r])); NCK(ncclReduce(x[r], gathered[r], count, ncclFloat32, ncclSum, 0, comms[r], streams[r])); }
      NCK(ncclGroupEnd());
      CK(cudaSetDevice(dev[0]));
      CK(cudaStreamSynchronize(streams[0]));
      CK(cudaMemcpy(hx.data(), gathered[0], count * sizeof(float), cudaMemcpyDeviceToHost));
      for (size_t i = 0; i < count; ++i) if (hx[i] != href[i]) {
        fprintf(stderr, "ncclReduce: root [%zu] = %g, want %g\n", i, hx[i], href[i]);
        return 1;
      }
      // ReduceScatter: each send is world copies of x[r]; every recv is the same sum
      std::vector<float*> rs_send(world), rs_recv(world);
      std::vector<float> sendh(static_cast<size_t>(world) * count);
      for (int r = 0; r < world; ++r) {
        CK(cudaSetDevice(dev[r]));
        CK(cudaMalloc(&rs_send[r], static_cast<size_t>(world) * count * sizeof(float)));
        CK(cudaMalloc(&rs_recv[r], count * sizeof(float)));
        CK(cudaMemcpy(hx.data(), x[r], count * sizeof(float), cudaMemcpyDeviceToHost));
        for (int k = 0; k < world; ++k)
          for (size_t i = 0; i < count; ++i) sendh[static_cast<size_t>(k) * count + i] = hx[i];
        CK(cudaMemcpy(rs_send[r], sendh.data(), static_cast<size_t>(world) * count * sizeof(float), cudaMemcpyHostToDevice));
      }
      NCK(ncclGroupStart());
      for (int r = 0; r < world; ++r) { CK(cudaSetDevice(dev[r])); NCK(ncclReduceScatter(rs_send[r], rs_recv[r], count, ncclFloat32, ncclSum, comms[r], streams[r])); }
      NCK(ncclGroupEnd());
      for (int r = 0; r < world; ++r) {
        CK(cudaSetDevice(dev[r]));
        CK(cudaStreamSynchronize(streams[r]));
        CK(cudaMemcpy(hx.data(), rs_recv[r], count * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) if (hx[i] != href[i]) {
          fprintf(stderr, "ncclReduceScatter: rank %d [%zu] = %g, want %g\n", r, i, hx[i], href[i]);
          return 1;
        }
        CK(cudaFree(rs_send[r]));
        CK(cudaFree(rs_recv[r]));
      }
    }
    // the unique-id route (ncclUniqueId by value): a second clique, in-place bcast from rank 0 of slab 0
    {
      ncclUniqueId id;
      NCK(ncclGetUniqueId(&id));
      std::vector<ncclComm_t> c2(world);
      for (int r = 0; r < world; ++r) { CK(cudaSetDevice(dev[r])); NCK(ncclCommInitRank(&c2[r], world, id, r)); }
      NCK(ncclGroupStart());
      for (int r = 0; r < world; ++r) { CK(cudaSetDevice(dev[r])); NCK(ncclBcast(r == 0 ? x[0] : gathered[r], count, ncclFloat32, 0, c2[r], streams[r])); }
      NCK(ncclGroupEnd());
      std::vector<float> h(count);
      for (int r = 1; r < world; ++r) {
        CK(cudaSetDevice(dev[r]));
        CK(cudaStreamSynchronize(streams[r]));
        CK(cudaMemcpy(h.data(), gathered[r], count * sizeof(float), cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) if (h[i] != ref[0][i]) { fprintf(stderr, "nccl bcast (unique id): rank %d [%zu] = %g, want %g\n", r, i, h[i], ref[0][i]); return 1; }
      }
      for (int r = 0; r < world; ++r) NCK(ncclCommDestroy(c2[r]));
    }
    for (int r = 0; r < world; ++r) { CK(cudaSetDevice(dev[r])); CK(cudaFree(gathered[r])); NCK(ncclCommDestroy(comms[r])); }
    printf("nccl %d: all-gather + rank-ordered sum, broadcast, send/recv, unique-id bcast, allreduce/reduce/reducescatter sum+avg: ok\n", v);
  }

  CK(cudaGraphExecDestroy(exec));
  for (int r = 0; r < world; ++r) {
    CK(cudaSetDevice(dev[r]));
    CK(cudaEventDestroy(events[r]));
    CK(cudaStreamDestroy(streams[r]));
    CK(cudaFree(x[r]));
  }
  CK(cudaEventDestroy(mark));
  printf("ranks: PASS\n");
  return 0;
}
