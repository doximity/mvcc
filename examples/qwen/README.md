# Qwen3.5-35B-A3B in CUDA kernels

A self-contained inference program for Qwen3.5-MoE written as ordinary CUDA C++ (`qwen.cu`, `kernels.cuh`) and compiled for the Mac GPU with mvcc.

Measured on an M5 Pro (20 GPU cores, 48 GB unified), macOS 26.5.1, INT4 experts + INT8 dense:

| metric | value |
| --- | --- |
| decode | 81 tok/s (12.1 ms/token, 2.6 GB of weights read per token, 214 GB/s effective) |
| prefill | 1370 tok/s at T=128, 1950 tok/s at T=512 |
| startup | < 1 s to first token (20 GB pack mmapped, page cache warm) |
| memory | 19.6 GB pack + ~0.5 GB working set |

## Build and run

```bash
export PATH=/path/to/mvcc/toolkit/bin:$PATH
nvcc -O3 -std=c++17 -o qwen qwen.cu

# one-time: convert the GPTQ-Int4 checkpoint (python3, numpy, HF snapshot on disk)
python3 convert.py ~/models/Qwen3.5-35B-A3B-GPTQ-Int4 ~/models/qwen35.mvccq --dense int8 \
        --tokenizer-out ~/models/qwen35_tokenizer.bin

./qwen --pack ~/models/qwen35.mvccq --tokenizer ~/models/qwen35_tokenizer.bin \
       --prompt "Explain RoPE briefly." --max-tokens 200
```

Options: `--raw TEXT` (omit the chat template), `--system TEXT`, `--think` (thinking channel), `--ctx N` (KV capacity, default 4096), `--chunk N` (prefill tokens per forward, default 128), `--prefetch` (touch the pack before the first token), `--bench` (per-kernel decode and prefill microbenchmarks), `--verbose`.

`convert.py --dense bf16`, the default, keeps dense projections in bf16 (21.5 GB pack, ~52 tok/s decode). `--dense int8`, used for the numbers above, quantizes them per-row symmetric with group-128 scales.

## Model and kernels

40 layers in a repeating pattern of three gated delta-net (linear attention) layers and one gated full-attention layer, each followed by a 256-expert top-8 MoE plus a shared expert. Kernels in `kernels.cuh`:

| stage | kernel | notes |
| --- | --- | --- |
| embedding, norms | `embed`, `rmsnorm`, `residual_rmsnorm`, `shared_gate_combine_rmsnorm` | residual + shared-expert gate + next norm fused |
| dense projections, T ≤ 4 | `gemv_bf16`, `gemv_i8` | warp per output row, several rows in flight per warp |
| dense projections, T ≥ 32 | `gemm_bf16_tc`, `gemm_i8_tc` | `mvcc::warp_tile` (Metal 4 TensorOps on M5) |
| gated delta-net | `gdn_conv`, `gdn_conv_state`, `gdn_prep`, `gdn_recurrence<CH>`, `gdn_gated_norm` | conv1d+SiLU with fused state update; q/k L2-norm + alpha/beta gate fused; register-only T=1 path |
| full attention | `attn_qk_norm_rope`, `attn_decode` (split-K online softmax), `attn_combine`, `sigmoid_gate` | bf16 KV cache written in the qk-norm/RoPE kernel |
| router | `router_topk` | softmax-normalized top-8 |
| MoE, T ≤ 4 | `moe_gate_up_int4`, `moe_down_int4`, `moe_slot_reduce` | INT4 nibbles dequantized in registers |
| MoE, T > 4 | `moe_gate_up_tile`, `moe_down_tile` | (token, expert) pairs sorted by expert; INT4 rows dequantized to shared memory once per 32-token tile; `warp_tile` MMA |
| head | `gemv_i8` (lm_head), `argmax_partial` + `argmax_final` | |

The kernels are CUDA: `__shfl_sync`, `__ballot_sync`, shared-memory staging, `cuda_bf16.h` / `cuda_fp16.h`. The mvcc-specific header is `<mvcc/tile.cuh>` for `mvcc::warp_tile`, which also has an `mma.sync` implementation, so the same source builds with NVIDIA `nvcc`.

## Files

- `qwen.cu`: model, forward pass, greedy sampling, chat template and `--bench`
- `kernels.cuh`: device code
- `pack.hpp`: `.mvccq` reader (mmap, `MAP_SHARED`, `cudaHostRegister` for zero-copy weights)
- `tokenizer.hpp`: byte-level BPE with the Qwen special tokens
- `convert.py`: converts safetensors to `.mvccq` (GPTQ-Int4 experts repacked to nibble planes; dense bf16 or int8)

## Correctness

Greedy generations for factual and multi-paragraph prompts (short, and > 128 tokens to exercise tiled prefill) are coherent and consistent between the bf16 and int8 dense variants. `tests/run.sh` compiles this example and, with `MVCC_QWEN_PACK` / `MVCC_QWEN_TOK` set, runs a smoke prompt. Kernel building blocks (`warp_tile`, conversions, warp intrinsics, libm) are covered against CPU references in `tests/kernels/`.
