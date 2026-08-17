# Qwen3.8-27B EXL3 K5/K6 + MTP2 + LMCache (2x DGX Spark, TP2)

Two-node profile for Qwen3.8-27B served from an EXL3 K5/K6 quant across a
DGX Spark pair at TP2, with MTP speculative decoding, EXL3 CUDA-graph decode,
FP8 prefill and FP8 KV cache, patched-NCCL two-rail RoCE striping, prefix
caching, and an LMCache L1 + NVMe L2 tier.

Distinguishing feature versus the other Spark profiles here: this is the
EXL3 (near-BF16 fidelity) lane rather than an NVFP4 one — the checkpoint
measures **0.00276 mean KLD vs BF16**, roughly an order of magnitude tighter
than the NVFP4 builds commonly run on this hardware, at 27 tok/s single-stream.

## Model

| Parameter | Value |
|---|---|
| Model ID | malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated |
| Base | Qwen/Qwen3.8-27B @ `1d4bf0f` (hybrid: 48 GatedDeltaNet + 16 full-attention layers, vision tower, MTP head) |
| Quantization | EXL3 — MLP gate/up K5, down K6; attention K6; lm_head K6; quantized MTP head; BF16 embeddings + vision. 21.61 GB |
| Served name | qwen38 |
| Tensor parallel | 2 (one GPU per node, ray executor) |
| MTP | 2 (throughput mode; 3 = interactive alternative, +6% single-stream / −4% at 64 streams) |
| Max context | 262,144 |
| KV cache dtype | FP8 (KV pool ≈ 1.78M tokens) |
| Block size | 1600 (mamba/GDN block) |
| Max batched tokens | 3072 (LMCache mamba-align constraint — see Constraints) |
| Max sequences | 64 |
| GPU memory utilization | 40% |

## Hardware

| Component | Configuration |
|---|---|
| Nodes | 2x NVIDIA DGX Spark (GB10), ~121 GiB unified memory each |
| Compute | sm_121, driver 580.173.02 |
| Fabric | 4x direct 200G QSFP DAC between the pair; RoCEv2. Two rails used (one port per ConnectX-7 card) |
| Per-link | ~109 Gb/s (PCIe Gen5 x4 per card is the ceiling; both ports of one card share it) |
| Storage | NVMe (ext4) for the LMCache L2 tier |
| Parallelism | TP2 / MTP2 |

## Runtime

| Component | Version |
|---|---|
| Engine | Gilded Gnosis vLLM fork (`local-inference-lab/vllm`), branch from `dev/gilded-gnosis @ fa033bd4e` + PR-318 stack + two upstream ports (below) |
| Kernels | exllamav3 1.4.2 (@ `5f3c537` + aarch64 port), b12x 1.2.4 |
| Framework | torch 2.12.0+cu132, CUDA 13.2.86 toolkit |
| Transport | patched NCCL 2.30.7 (sparkring switchless-ring patches), `sha256 e69a8c24…` |
| KV tier | lmcache 0.5.2 + heartbeat patch (mandatory — see Patches) |
| Base image | `nvcr.io/nvidia/cuda@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d` |

### Patches applied (all four are required)

1. **exllamav3 aarch64 port** — upstream is x86-only (AVX intrinsics, `__builtin_ia32_pause`).
2. **vLLM #51113 port** — mamba-align chunk splitting, hand-ported into the fork's diverged
   scheduler. Prefix caching on this hybrid model is unsafe without it.
3. **vLLM #48425 port** — per-group prefix-hit divergence reconcile. Without it, LMCache plus
   KV-pressure eviction can resume generation on stale recurrent state and emit silent garbage
   (LMCache issue #4247).
4. **lmcache 0.5.2 heartbeat fix** — stock `_ensure_heartbeat_started` guards on
   `if self._heartbeats is not None:` against a field initialized to `{}`, so heartbeats never
   start, the servers reap the engine's registration after ~2.5 minutes, and **every lookup
   silently returns 0 thereafter** while stores keep working. Benchmarks mask it (they query
   inside the live window); long-lived serving loses the tier. Two-character fix.

Patches, scripts, and a step-by-step build recipe: see the companion recipe repository noted in
`manifest.json` (`recipe_repository`).

## Backend pins (driver-imposed)

Driver 580 cannot JIT the build's CUDA-13.2 FlashAttention PTX
(`cudaErrorUnsupportedPtxVersion`). Three pins are required, not preferences:
`--attention-backend TRITON_ATTN`, `--mm-encoder-attn-backend TORCH_SDPA`, and
`"attention_backend":"TRITON_ATTN"` **inside** the speculative config — the MTP drafter
resolves its own backend and crashes without its own pin.

## Apply

1. Copy `profile.env.example` to `.env` on each node and replace every `REPLACE_WITH_*`.
2. Build the runtime per the recipe repository (source builds: fork, exllamav3, lmcache,
   patched NCCL). This profile records configuration, not a published image.
3. Start in order: LMCache server on each node → ray head (rank 0) → ray worker (rank 1) →
   engine. `compose.yml` covers the per-node cache server plus the engine container;
   ray bring-up stays in the recipe's scripts because it is order-dependent across nodes.
4. Validate with `test_correctness.py` (byte-exactness, cache-hit, and the heartbeat-window
   check).

## Constraints and known limitations

- **`max_num_batched_tokens` must be in [1600, 3200) when LMCache is enabled.** lmcache's
  mamba-align guard requires every prefill step to advance exactly one 1600-token block so each
  chunk boundary holds a real recurrent-state snapshot. 3072 is the best legal value; it costs
  ~8% cold prefill versus 8192 (invisible under concurrent decode). Do not remove the guard —
  it prevents storing null-state chunks under valid content hashes.
- **LMCache chunk size must be 1600** (a multiple of the mamba block).
- **L1 must be at least as large as the longest prefix you expect to replay.** Lookups count an
  L2 hit only after staging chunks into L1, so an undersized L1 silently answers zero even when
  every chunk is on NVMe. Roughly 106 MB per 1600-token chunk per rank.
- Prompt tails shorter than one chunk always recompute.
- Two rails is the maximum useful striping: rails beyond one port per card add NCCL
  channel-kernel SM contention inside decode graphs (measured −28% single-stream at four rails)
  without adding bandwidth, since both ports of a card share one PCIe x4 link.
- `NCCL_IB_SUBNET_AWARE_ROUTING` must be **0** for parallel rails between the same rank pair,
  with one subnet per cable; otherwise all channels collapse onto one card.
- No KLD measurement on this hardware — fidelity is inherited from the checkpoint's published
  RTX 5090 receipts. Verified here by byte-exact greedy reproduction across cache states, not by
  a fidelity suite.
- Long-context (100K+) concurrency untested. Vision path functional but only trivially probed.
