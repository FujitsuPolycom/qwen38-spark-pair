# Results

Sanitized runs on the reference DGX Spark pair, TP2 / MTP2 / FP8 KV / two-rail striping /
`BATCHTOK=3072`, EXL3 CUDA-graph decode, LMCache (4 GB L1 + NVMe L2, chunk 1600).
Greedy (`temperature 0`) throughout. Harness: `local-inference-lab/llm-inference-bench`,
plus direct endpoint timings where noted.

## Single-stream decode ladder

Each row adds one change to the row above, so the ladder doubles as an ablation.

| Configuration | Decode tok/s |
|---|---:|
| 1 node, eager, no speculation | 10.7 |
| TP2 over one cable | 17.0 |
| + fork PR-318 stack + MTP3 | 23.7 |
| + EXL3 CUDA-graph decode | 29.6 |
| Production (MTP2 throughput mode, FP8 KV, two-rail) | **27.0** |
| MTP3 interactive variant | 28.6 |

MTP2 versus MTP3 is a deliberate trade: MTP2 gives up ~6% single-stream and returns
+8% at 16 streams, +12% at 32, +4% at 64, and doubles the KV pool headroom.

## Concurrency (256-token streams, uniform)

| Streams | 1 | 8 | 16 | 32 | 64 |
|---|---:|---:|---:|---:|---:|
| Aggregate tok/s | 25.4 | 139 | 189 | 228 | **275** |

Under a mixed agentic load at 64 streams the engine sustained 1,030–1,325 tok/s of prefill
concurrently with 91–128 tok/s of decode, at 72% native prefix-cache hit rate plus 17–18%
external (LMCache) hit rate, KV pool peaking at 57%.

## Domain battery (comparable methodology to public single-Spark threads)

| Domain | Decode tok/s |
|---|---:|
| Prose | 23–26 |
| Code | 29–39 |
| Reasoning | 40 |

MTP draft acceptance under live load: 76–95%, mean accepted length 2.6–2.9 at depth 2.

## Prefill

| Configuration | tok/s |
|---|---:|
| Baseline (no FP8 prefill) | 662 |
| + `VLLM_EXL3_PREFILL_FP8=1` | 1,433 (2.16x) |
| + two-rail striping, `BATCHTOK=8192` | 1,500 |
| Production with LMCache (`BATCHTOK=3072`) | ~1,375 |

The last row is the price of the mamba-align batch constraint: about 8% of cold prefill,
and only on genuinely cold prompts.

## KV reuse

18,750-token prompt, 32-token greedy completion, measured end to end:

| Scenario | Wall time | Source of the hit |
|---|---:|---|
| Cold | 16.0 s | full prefill |
| Replay after **engine** restart | 2.18 s (**7.4x**) | 11/11 chunks from server L1 |
| Replay after **engine and cache-server** restart | 2.22 s (**7.2x**) | 11/11 chunks from NVMe (`0 L1, 11 L2`), staged in 41 ms, retrieved to GPU in 11 ms |

Outputs were byte-identical in every scenario. The trailing partial chunk (1,150 tokens of a
1,600-token chunk) always recomputes — chunk granularity, by design.

Earlier prefix-cache validation without LMCache: a 9K-token prompt replayed 1.9x faster with
byte-identical greedy output (the historical silent-corruption scenario for hybrid models,
which the #51113 port is what makes safe).

## Transport

| Measurement | Value |
|---|---|
| Two-rail all-reduce, decode shapes | 37 µs |
| Two-rail all-reduce, 64-stream shapes | 158 µs (2.4x faster than single-card) |
| Large-payload aggregate | 160 Gb/s |
| Raw RDMA per link | 1.8 µs, ~109 Gb/s |

Four-rail striping measured 176 Gb/s aggregate but cost 28% of single-stream decode to NCCL
channel-kernel SM contention inside the decode CUDA graphs; two rails (one port per card,
two channels) beat both the four-rail and unstriped configurations on every serving metric.

Methodology note for anyone repeating the transport numbers: NCCL microbenchmarks run beside a
live engine on GB10 report a phantom ~2.3 ms per-operation floor caused by GPU context
timeslicing. Quiesce the GPU or trust only end-to-end serving measurements.

## Correctness

Byte-exact greedy reproduction was verified at every configuration change: cold versus warm,
across engine restarts, across cache-server restarts, and before/after each patch. CUDA-graph
decode was gated on a 3-of-3 exact-match check against eager before adoption. No fidelity (KLD)
suite has been run on this hardware; the checkpoint's published 0.00276 was measured elsewhere,
on different silicon and a different runtime revision.
