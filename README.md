# qwen38-spark-pair

**A step-by-step recipe for serving [`malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated`](https://huggingface.co/malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated) on one or two NVIDIA DGX Spark (GB10) nodes** — `TP=1` on a single Spark or `TP=2` across a pair — with MTP speculative decoding, CUDA-graph decode, FP8 prefill+KV, prefix caching, an LMCache L1+NVMe KV tier, and (at `TP=2`) 2-rail RoCE striping. Despite the repository name, one Spark is enough: `TP=1` drops the fabric phases entirely ([One Spark or two](#one-spark-or-two)).

Measured on one such pair — the "reference pair" referenced throughout: **27 tok/s single-stream · 275 tok/s aggregate @ 64 streams · ~1,375 tok/s cold prefill · 7× replay speedup from NVMe cache · byte-exact greedy outputs throughout.** A single Spark at `TP=1` serves the same stack at **23.8 tok/s single-stream · 85 tok/s aggregate @ 4 streams** with a 1.67M-token KV pool. Near-BF16 quality (checkpoint KLD 0.00276 vs BF16).

This document is written to be executed by an LLM agent with SSH access to the Spark — or both Sparks at `TP=2`. Each phase ends with a **Verify** gate — do not proceed past a failed gate. Every site-specific value is listed in [Site values](#site-values); substitute yours throughout.

> The [sparkring](https://github.com/FujitsuPolycom/sparkring) project (same maintainer) supplies two ingredients: the switchless-ring NCCL patches built in Phase 6 and the container image pin in Phase 2. This repository consumes them at pinned revisions and modifies neither sparkring nor any other upstream.

---

## What you need

- 1× or 2× DGX Spark (GB10, ~121 GB unified memory each — the second Spark is only needed for `TP=2`), DGX OS (Ubuntu 24.04.4 LTS as tested) with **driver 580.173.02** (any 580.x should behave identically) (recipe assumes the driver cannot JIT CUDA-13.2 PTX — see the Phase 9 backend pins), Docker with NVIDIA runtime.
- `TP=2` only: 2–4 direct 200G QSFP cables between the pair's ConnectX-7 ports (2 minimum, one per card; 4 harmless, only 2 are used).
- ~80 GB free disk per node (model + build trees + cache tier headroom; NVMe cache cap is configurable).
- A LAN IP for each node and SSH between them.

## Site values — edit one file

Copy the example config and fill in your hardware. **This is the only file you edit**; every
script sources it and falls back to the reference pair's values when it is absent.

**Easiest path — let it detect your hardware:**

```bash
bash scripts/detect-site.sh            # print what it finds, for review
bash scripts/detect-site.sh --write    # write scripts/site.env
```

It probes containers, work dir, LAN and fabric addresses, ConnectX netdevs in PCI order, one
RDMA device per card, the rail prefix, and the **RoCE v2** GID index (the same address also
appears as RoCE v1 at a different index — picking that one yields a fabric that misbehaves
rather than fails, so the detector filters by type). Anything it cannot determine is emitted as
`REPLACE_WITH_*` and it exits non-zero, so a partial detection is obvious. On the reference pair
it reproduces all 13 values correctly. **Review the output before using it** — it is a starting
point, not an oracle.

Or fill it in by hand:

```bash
cp scripts/site.env.example scripts/site.env
$EDITOR scripts/site.env
```

| Setting | What it is | How to find it |
|---|---|---|
| `TP` | 1 = single Spark, 2 = a pair | — |
| `C0` / `C1` | serving container names | your choice at Phase 2 |
| `WS` | host work dir, mounted as `/ws` | your choice |
| `LAN_RANK0` / `LAN_RANK1` | LAN IPs (LMCache server URLs) | `ip -br addr` |
| `FABRIC_RANK0` / `FABRIC_RANK1` | point-to-point IPs on one direct cable **[2x]** | Phase 1 |
| `NETDEV1..4` | the four ConnectX-7 netdevs, cable order **[2x]** | `ip -br link` |
| `RDMA_CARD1` / `RDMA_CARD2` | RDMA device for **one port on each card** **[2x]** | `ibv_devices` |
| `NCCL_GID_INDEX` | GID index holding the per-cable /24 **[2x]** | `cat /sys/class/infiniband/<dev>/ports/1/gids/<n>` |
| `RAIL_PREFIX` | per-cable rail subnets **[2x]** | your choice |

Settings marked **[2x]** are irrelevant with `TP=1` — a single node needs no fabric, no striping
and no second cache server.

Convention below: **[r0]** = run on rank0 host, **[r1]** = rank1, **[both]** = both. Container commands are `docker exec <container> bash -c "..."` with the work dir at `/ws`.

---

## Phase 1 — Fabric bring-up

1. [both] Confirm the direct links are up: `ip -br link | grep -E "en[pP]"` — the cabled ports show `UP`.
2. [both] Give one cable a point-to-point subnet for bootstrap (the "fabric IP" above), e.g. rank0 `sudo ip addr add 198.18.200.1/30 dev enp1s0f0np0`, rank1 `.2/30` on its cabled peer port. (If your OS manages these already, use what exists.)
3. [both] Add the per-cable rail /24s (used by NCCL striping). No sudo needed later if you use a NET_ADMIN helper container (Phase 2); with sudo now:
   `ip addr add 10.42.N.{1|2}/24 dev <Nth-cable-netdev>` for each cable N.
4. Qualify each cable: `ib_write_bw` between the pair per RDMA device. Expect **~109 Gb/s** per link (PCIe Gen5 x4 ceiling per card — this is normal on Spark, not a cabling fault).

**Verify:** ping across the fabric IP; `ib_write_bw` ≥ ~100 Gb/s on both cards' chosen ports.

## Phase 2 — Containers

1. [both] Serving/build container from the CUDA base image (sparkring-pinned digest works; any recent `nvcr.io/nvidia/cuda:13.x-devel-ubuntu24.04` aarch64 image should too):

```
docker run -d --name <ggrun|ggbuild> --restart unless-stopped \
  --gpus all --network host --ipc host --cap-add IPC_LOCK \
  --device /dev/infiniband \
  -v /home/<user>/work/qwen38-exl3:/ws \
  nvcr.io/nvidia/cuda@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d sleep infinity
  # (= the 13.0.1-devel-ubuntu24.04 tag at test time, pinned by digest)
```

2. [both] In-container deps: `apt-get update && apt-get install -y python3 python3-dev python3-venv git cmake ninja-build ccache && apt-get install -y cuda-toolkit-13-2` — **the CUDA 13.2 toolkit is required** (13.0 base cannot build the fork's kernels; 13.2.86 as tested — the meta-package floats within 13.2.x).
3. [both] **Prevent the container from losing its GPU.** With cgroups v2 and the systemd driver,
   containers can lose access to the NVIDIA devices (`Failed to initialize NVML: Unknown Error`,
   surfacing later as `CUDA error: invalid device ordinal`), often triggered by
   `systemctl daemon-reload` ([nvidia-container-toolkit#48](https://github.com/NVIDIA/nvidia-container-toolkit/issues/48)).
   Apply NVIDIA's fix once, and make it run at boot:

```bash
sudo nvidia-ctk system create-dev-char-symlinks --create-all
sudo tee /etc/cron.d/nvidia-dev-char <<< "@reboot root /usr/bin/nvidia-ctk system create-dev-char-symlinks --create-all >/dev/null 2>&1"
```

   (The symlinks do not persist across reboots — the cron.d entry recreates them at boot. The
   redirect keeps cron from mailing root the command's warnings every boot.)

   **The output looks alarming and is not.** Expect many `WARN ... unable to detect IOMMU FD`
   and `unable to get device name` lines — the tool probes for VFIO/IOMMU paths that a Spark
   does not have. On a node where some links already exist, expect `Could not create symlink:
   ... file exists` for each. The command is idempotent; neither warning is a failure.

**Verify:** `ls /dev/char | wc -l` returns a few hundred entries (357 on the reference pair, which
varies with driver capability count) and `ls -l /dev/char/195:0` resolves to `../nvidia0`.

   `scripts/boot-stack-aa42.sh` (the boot-persistence script installed in Phase 9) also detects the condition and restarts the container, but preventing it
   is better than recovering from it — the recovery costs a restart cycle and only fires at boot.

4. [both, optional] NET_ADMIN helper for sudo-less rail IPs at boot: `docker run -d --name netadm --restart unless-stopped --network host --cap-add NET_ADMIN ubuntu:24.04 sleep infinity`.

**Verify:** `docker exec <c> nvcc --version` reports 13.2; `docker exec <c> ls /dev/infiniband` shows uverbs devices.

## Phase 3 — Python env + kernel libraries

All builds happen in the rank0 container; rank1 receives the finished tree (Phase 7).

1. venv at `/ws/venv`, then:

```
pip install torch==2.12.0 torchvision==0.27.0 --index-url https://download.pytorch.org/whl/cu132
pip install b12x==1.2.4 ray==2.57.0 setuptools-rust ninja pytest
```

   (`pytest` is needed by the Phase 4 verify gate.) The full 231-package closure of the tested
   venv is committed as [`requirements-freeze.txt`](requirements-freeze.txt) — if a build fails on
   a transitive dependency, diff your venv against it before debugging anything else.
2. **exllamav3 (needs the ARM port — upstream is x86-only):**

```
cd /ws/src && git clone https://github.com/turboderp-org/exllamav3 && cd exllamav3
git checkout 5f3c537
git apply /ws/patches/exllamav3-arm.patch
TORCH_CUDA_ARCH_LIST="12.1" python setup.py bdist_wheel   # plain 12.1 here — NOT 12.0f
pip install dist/*.whl --no-deps
```

**Verify:** `python -c "from exllamav3_ext import exl3_gemm; print('ok')"`.

## Phase 4 — The vLLM fork

Two equivalent paths — **A** is the immutable one, **B** shows the provenance.

**A — pinned mirror (recommended).** The exact tested tree (base `fa033bd4e` + the PR-318 merge +
both ports, tip `229effc810ee`) is preserved at
[FujitsuPolycom/vllm](https://github.com/FujitsuPolycom/vllm/tree/gg-perf-mtp), tag
`qwen38-tested-20260817`, so it does not depend on the upstream PR staying open or unchanged:

```
cd /ws/src && git clone --branch qwen38-tested-20260817 https://github.com/FujitsuPolycom/vllm vllm-gg && cd vllm-gg
TORCH_CUDA_ARCH_LIST="12.0f" pip install -e . --no-build-isolation   # 12.0f REQUIRED (see troubleshooting)
```

`fork-ports.patch` is **already in this tree** — do not apply it again.

**B — from parts** (fails if upstream force-pushes or deletes PR 318 — which is why A exists):

```
cd /ws/src && git clone https://github.com/local-inference-lab/vllm vllm-gg && cd vllm-gg
git checkout fa033bd4e                     # dev/gilded-gnosis pin
git fetch origin pull/318/head:pr318
git merge 2b96dad45                        # PR-318 head as tested (⊇ #316 ⊇ #314); merges clean
# PR 318 was open as of the 2026-08-17 pin verification — merging `pr318` instead takes whatever its head is today,
# which may not be the state these numbers were measured on.
git am /ws/patches/fork-ports.patch        # our two ports: vLLM #51113 + #48425 (see below)
TORCH_CUDA_ARCH_LIST="12.0f" pip install -e . --no-build-isolation   # 12.0f REQUIRED (see troubleshooting)
```

The two ports in `fork-ports.patch` are **not optional**:
- **#51113 port** — mamba-align chunk splitting in the scheduler. Without it, enabling prefix caching on this GDN-hybrid model risks silent corruption.
- **#48425 port** — per-group prefix-hit divergence reconcile. Without it, LMCache + KV-pressure eviction can resume generation on stale mamba state (silent token salad; see [LMCache issue #4247](https://github.com/LMCache/LMCache/issues/4247)).

**Verify:** `python -c "import vllm; print(vllm.__version__)"`; `cd /ws/src/vllm-gg && python -m pytest tests/v1/core/test_scheduler.py -q` → expect 129/130 (one known order-dependent flake).

### Pins, verified 2026-08-17

Every external reference in this recipe resolved on that date. Full identities:

| Pin | Value |
|---|---|
| engine mirror (path A) | `FujitsuPolycom/vllm` tag `qwen38-tested-20260817` = `229effc810ee` |
| fork base | `fa033bd4e1b16d9d729ad94be2d87da5a13210ce` |
| PR-318 head as tested | `2b96dad45b2c9a1dd59b4bf3f9f33b06cb70f42a` |
| exllamav3 | `5f3c537` |
| torch / torchvision | `2.12.0+cu132` / `0.27.0+cu132` from `download.pytorch.org/whl/cu132` |
| CUDA toolkit | 13.2.86 (meta-package floats within 13.2.x) |
| lmcache | `0.5.2` from PyPI (0.5.3 exists; 0.5.2 is deliberate — the heartbeat patch targets it) |
| sparkring (NCCL stage) | `4545c4ec4740f203d4f427db265414a34bd8f5db` |
| patched NCCL artifact | sha256 `e69a8c240f45d10166bcd901d99db78bb63147adda66e586d8dd505c6d608b54` |
| HF checkpoint revision | `ab3a91a13813df8096cb4c1d560ed3669035d0cf` |
| base image | `nvcr.io/nvidia/cuda@sha256:5c36750138dc1447a17dafbb397674f167d3b44ce18d9160d769df114577b35d` |

Shorthand used in the phases: fork base `fa033bd4e`,
PR-318 head `2b96dad45`, exllamav3 `5f3c537`, lmcache `0.5.2` on PyPI (latest is 0.5.3 —
0.5.2 is deliberate, the heartbeat patch is written against it), the sparkring repository,
and the Hugging Face checkpoint. If a build fails at one of these, check whether the pin
moved before debugging anything else.

## Phase 5 — Model + template

1. Download the checkpoint **at the tested revision** (21.61 GiB — the bare repo ID is mutable;
   its model card can change while the weight files stay the same):

```
huggingface-cli download malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated \n  --revision ab3a91a13813df8096cb4c1d560ed3669035d0cf --local-dir /ws/model/Qwen3.8-27B-EXL3-K5K6-hydrated
```

   Verify SHA256s against the repo's manifest — 16/16 must match. The weight shards at this
   revision are byte-identical to the tested deployment (spot-verified via LFS OIDs).
2. Copy `scripts/chat_template_agentic.jinja` to `/ws/` (side file — the checkpoint stays byte-untouched). It maps `reasoning_effort` and renders mid-conversation system messages as `<system-reminder>` blocks (agent-CLI friendly).

## Phase 6 — Patched NCCL (2-rail striping)

Build NCCL 2.30.7 with the sparkring switchless-ring patches — the [sparkring repo](https://github.com/FujitsuPolycom/sparkring)'s `runtime/Containerfile` (commit `4545c4ec4740` as tested) has a self-contained `nccl-build` stage; run that stage and copy `libnccl.so.2.30.7` out to `/ws/nccl-patched/` [both], then create the SONAME links the scripts load:

```
cd /ws/nccl-patched && ln -sf libnccl.so.2.30.7 libnccl.so.2 && ln -sf libnccl.so.2 libnccl.so
```

Verify: `sha256sum libnccl.so.2.30.7` → `e69a8c240f45d10166bcd901d99db78bb63147adda66e586d8dd505c6d608b54` on the reference pair.

The injection and all striping env live in `scripts/tp2-env.sh` (`STRIPE=2` block). The non-obvious parts, already encoded there:
- `NCCL_IB_SUBNET_AWARE_ROUTING=0` — **required**; the subnet-aware feature collapses parallel rails between the same rank pair onto one card.
- One port per card (`NCCL_IB_HCA=<card1-port0>,<card2-port0>`), `MIN/MAX_NCHANNELS=2` — more rails/channels only add SM contention inside decode CUDA graphs (measured −28% single-stream at 4 rails).
- Injected via `LD_PRELOAD` + `VLLM_NCCL_SO_PATH`; no system libraries touched.

**Verify (only on an otherwise-idle GPU — NCCL microbenchmarks beside a live engine give phantom milliseconds on GB10):** a 2-rank all-reduce probe shows small-op latency in the tens of µs and ~160 Gb/s at large sizes.

## Phase 7 — Replicate to rank1

Copy `/ws/{venv,src,model,nccl-patched,patches}` and all `scripts/*` to rank1's work dir (rsync over the fabric IP is fastest). The venv is path-portable if the work dir path is identical on both nodes — keep it identical. SHA-verify the model copy.

**Verify [r1]:** the exllamav3 and vllm import checks from Phases 3–4 pass in rank1's container.

## Phase 8 — LMCache tier

1. [both] Build lmcache 0.5.2 from source in the venv: `TORCH_CUDA_ARCH_LIST="12.1" pip install lmcache==0.5.2 --no-build-isolation --no-deps` (**12.1, not 12.0f** — lmcache's arch parser rejects family suffixes).
2. [both] **Apply the heartbeat fix — mandatory:**

```
cd /ws/venv/lib/python3.12/site-packages && patch -p1 < /ws/patches/lmcache-0.5.2-heartbeat-fix.patch
```

Stock 0.5.2 initializes the scheduler adapter's heartbeat registry to `{}` and then guards the lazy starter with `if self._heartbeats is not None:`. An empty dict is not `None`, so the guard always returns and the thread-creation loop is unreachable: the process never logs `Started PeriodicThread: lmcache-heartbeat`, and each server's health event stays at its constructed value, so the scheduler's view of server health is permanently healthy and it never enters degraded mode. Observed on this stack before the patch: every cache lookup returned 0 while stores kept working, and benchmarks still passed because they query right after a restart. Treat the link between defect and symptom as observed rather than established -- `lmcache.mp.mq_timeout` was raised from 10 to 60 in the same change window, so the field A/B does not isolate the patch. The defect itself is unambiguous in the source and the fix is two characters, so apply it either way.

3. [both] NVMe cache dir `mkdir -p /ws/lmcache-l2`.
4. Server config is `scripts/lmcache-server.sh` — one server per rank, port 6556. Sizing rules baked in, don't lower them casually:
   - `--chunk-size 1600` — must equal this model's mamba block (1600); chunks are ~106 MB each per rank.
   - `--l1-size-gb 8` — lookups only count a hit after staging chunks into L1, so **L1 must be ≥ your largest replayed prefix** or lookups silently answer 0. A hit found on NVMe still needs an L1 write reservation, and the default trim policy truncates the answer at the first reservation failure, so a prefix one chunk too large for L1 collapses to zero hits rather than a partial hit.
   - `fs_native` L2, `use_odirect:false`, capacity to taste (`max_capacity_gb`).

## Phase 9 — Launch

Order matters: cache servers → ray head → ray worker → engine.

```
[r0] docker exec -d ggrun   bash -c "RANK=0 bash /ws/lmcache-server.sh > /ws/logs/lmcache-r0.log 2>&1"
[r1] docker exec -d ggbuild bash -c "RANK=1 bash /ws/lmcache-server.sh > /ws/logs/lmcache-r1.log 2>&1"
[r0] docker exec ggrun   bash /ws/tp2-ray-head.sh
[r1] docker exec ggbuild bash /ws/tp2-ray-worker.sh
[r0] bash scripts/start-stack.sh          # gated launcher — see below
```

**Use `scripts/start-stack.sh` rather than launching the engine by hand.** It refuses to start
on any failed precondition instead of leaving you a half-dead stack and a traceback whose real
cause is buried: no stale engine holding GPUs, ray reporting enough *free* GPUs (a not-yet-released
actor from the previous engine makes vLLM see one GPU and fail the TP2 placement group), cache
servers listening on both ranks, the lmcache heartbeat patch actually present, weights and patched
NCCL on both ranks, disk for the L2 tier. It then verifies the engine answered `/v1/models` and
that the heartbeat thread started, and on failure prints the *first* real error rather than the
traceback tail. `bash scripts/start-stack.sh --check` runs the gates read-only and leaves a running
engine untouched.

**Critical restart rule: kill engine → cycle cache servers → start engine.** The LMCache MP servers
import the engine's KV tensors through CUDA IPC (69 GB observed held by a server whose RSS was
4.8 GB). A clean engine shutdown unregisters the mapping and the server releases it immediately, but
an abrupt exit sends nothing, and release then waits on the server's reaper -- 120 to 150 s at the
default `--worker-reap-timeout-seconds 120`, whose scan runs every quarter of that interval. A
restart faster than that window fails with `Free memory on device cuda:0 … less than desired GPU
memory utilization`. Cycling the servers frees the memory at once and does not depend on how the
engine died.

The serve script's gates (each is one env var + restart to A/B):

| Gate | Production | Meaning |
|---|---|---|
| `LMC` | 1 | LMCache connector on. **Forces `BATCHTOK ∈ [1600,3200)`** — the mamba-align guard; 3072 is the best legal value (~8% cold-prefill cost, invisible under concurrent load). `LMC=0` → use `BATCHTOK=8192` |
| `APC` | 1 | prefix caching + `--mamba-cache-mode align` (safe only because of the #51113 port) |
| `STRIPE` | 2 | 2-rail patched-NCCL transport (0 = stock single-cable) |
| `MTPK` | 2 | MTP speculative depth (2 = throughput mode; 3 = +6% single-stream, −4% @cc64) |
| `KVDTYPE` | fp8 | FP8 KV cache — doubles the KV pool |
| `GPUMEM` | 0.70 | fraction of **unified** memory for the engine. Measured pools: 0.40 → 1,842,455 tokens · 0.55 → 2,852,305 · 0.70 → ~3.9M. Community guidance for GB10 is ≤0.70 — the usual discrete-GPU 0.85–0.95 does not apply, since the OS, page cache, ray and the cache servers draw from the same pool |
| `TP` | 2 | 1 = single Spark (drops ray, striping, rank-1 cache server) |
| `STAGE` | graph | EXL3 CUDA-graph decode (+25% vs eager) |
| `RECON_M` | 256 | EXL3 prefill reconstruction tile (`VLLM_EXL3_PREFILL_RECONSTRUCT_M`). Prefill-only; 128 collides with MTP-inflated decode batches at cc32 |
| `FP8PREFILL` | 1 | FP8 prefill GEMM (2.16x prefill). Prefill-only E4M3 numerics; decode keeps the exact trellis kernels |

Fixed flags worth knowing: `--attention-backend TRITON_ATTN`, `--mm-encoder-attn-backend TORCH_SDPA`, and `"attention_backend":"TRITON_ATTN"` **inside** the speculative config — all three pins exist because driver 580 cannot JIT the build's CUDA-13.2 FlashAttention PTX; the drafter crashes without its own pin. The `RECON_M` and `FP8PREFILL` gates above set `VLLM_EXL3_PREFILL_RECONSTRUCT_M` and `VLLM_EXL3_PREFILL_FP8`; both default to the production values, so leaving them unset is the same as passing them.

Boot persistence: install `scripts/boot-*.sh` as `@reboot` crontabs (edit the env gates in the serve line to match production first).

## One Spark or two

`TP` is the switch and the only thing you change. `TP=2` uses ray, two-rail RoCE striping and a
cache server per rank; `TP=1` drops all three and the launcher skips those gates.

```bash
# two Sparks
bash scripts/start-stack.sh

# one Spark
TP=1 bash scripts/start-stack.sh
```

Full production launch, either way:

```bash
TP=2 LMC=1 APC=1 STRIPE=2 MTPK=2 KVDTYPE=fp8 STAGE=graph BATCHTOK=3072 GPUMEM=0.70   bash scripts/start-stack.sh

TP=1 LMC=1 APC=1 MTPK=2 KVDTYPE=fp8 STAGE=graph BATCHTOK=3072 GPUMEM=0.70   bash scripts/start-stack.sh
```

On a single node, **phases 1, 6 and 7 are unnecessary** — no fabric to qualify, no patched NCCL
to build, nothing to replicate. All four patches still apply: they concern hybrid KV groups and
cache liveness, not node count.

### Measured

Both configurations, same harness, greedy, `GPUMEM=0.70`, `BATCHTOK=3072`, LMCache on.

**Two Sparks** — KV pool 3,935,138 tokens

| | cc1 | cc2 | cc4 |
|---|---:|---:|---:|
| decode @ 4k ctx | 30.9 | 55.7 | 101.7 |
| decode @ 16k ctx | 32.2 | 57.8 | 105.6 |

Prefill: 1,085 / 1,064 / 1,004 / 918 tok/s at 4k / 8k / 16k / 32k.
Longer runs: 27.0 tok/s single-stream at 256-token generations, 275 tok/s aggregate at 64
streams, ~1,375 tok/s cold prefill on an 11.5K prompt.

**One Spark** — KV pool 1,669,678 tokens

| | cc1 | cc2 | cc4 |
|---|---:|---:|---:|
| decode @ 4k ctx | 23.8 | 43.1 | 85.2 |
| decode @ 16k ctx | 22.2 | 42.0 | 85.3 |

Prefill: 330 / 398 / 446 / 667 tok/s at 4k / 8k / 16k / 32k. Prefill throughput *rises* with
context here — fixed per-request overhead dominates at these sizes and amortizes as prompts grow.

MTP acceptance runs 2.5–2.65 tokens per step in both configurations.

Recommended single-node adjustments: lower `--max-num-seqs` from 64 (the KV pool is smaller, so
64 streams each get a much thinner slice), and see LMCache sizing below — the chunk size doubles.

## LMCache sizing

Chunk size is fixed by the model, not by choice: it must be a multiple of the 1600-token mamba
block, so this deployment uses **1600**. What that costs in bytes depends on tensor parallelism,
because each rank stores only its share of the KV heads:

```
chunk bytes per rank = 65 layers x 2 (K,V) x 1600 tokens x 1024 B / TP
```

| | bytes per chunk, per rank | replayable tokens per GB of L1 |
|---|---:|---:|
| TP=2 | 106,496,000 (≈106 MB) | ≈15,000 |
| TP=1 | 212,992,000 (≈213 MB) | ≈7,500 |

**L1 is a hard ceiling on replay length.** A lookup only counts an L2 hit *after* staging the
chunks into L1, so a prefix longer than L1 can hold returns **zero hits — silently** — even when
every chunk is sitting on NVMe. This is the single most confusing failure in the whole stack:
the cache looks healthy, stores work, and replays quietly recompute.

| `--l1-size-gb` | TP=2 replay ceiling | TP=1 replay ceiling |
|---:|---:|---:|
| 4 | ≈59K tokens | ≈29K tokens |
| **8** (default here) | **≈118K tokens** | **≈59K tokens** |
| 16 | ≈236K tokens | ≈118K tokens |
| 17.5 / 35 | full 262K context | full 262K context |

Pick it from the longest prompt you actually expect to replay, then check it fits: L1 lives in
the same unified memory as the engine, so `GPUMEM x 121 GB + L1 + OS` must leave headroom. At
`GPUMEM=0.70` the engine takes ~85 GB, leaving roughly 19 GB — 8 GB of L1 is comfortable there,
16 GB is not.

**L2 (NVMe)** is cheap by comparison and bounded by `max_capacity_gb` (200 GB here ≈ 3.0M tokens
at TP=2). It is content-addressed and survives both engine and cache-server restarts, so it is
the tier that makes cold-start replays fast. It costs nothing but disk — size it generously.

Two operational rules that follow from the above:

- **Restart order is kill engine → cycle cache servers → start engine.** The servers hold the
  engine's KV cache through CUDA IPC and never release it on engine exit; skipping the cycle
  leaves too little memory for the next start.
- **Chunk size must stay a multiple of the mamba block**, and `max_num_batched_tokens` must stay
  in `[1600, 3200)` while LMCache is attached.

## Phase 10 — Validation gauntlet

The companion profile ships an automated version of these checks:
[`test_correctness.py`](https://github.com/FujitsuPolycom/inference-runtime-profiles/blob/master/profiles/qwen38-27b-exl3-k5k6-mtp2-lmcache-2x-spark/test_correctness.py)
(byte-exactness, cache-hit, and a separate `--stage heartbeat` that must run **>4 minutes after
engine start** — the one gate that catches the heartbeat bug, which every immediate-replay test
passes).

1. **Liveness:** `curl http://<rank0-lan>:8000/v1/models`.
2. **Exactness:** a few greedy (`temperature 0`) probes — arithmetic, instruction-following. Re-run each twice: byte-identical.
3. **Heartbeat (the fix from Phase 8 working):** engine log contains `Started PeriodicThread: lmcache-heartbeat`; server logs show **no** `Reaped GPU instance` while the engine lives.
4. **APC:** send a ~9K-token prompt twice: second is ~2× faster, byte-identical output.
5. **Cache tier end-to-end:** send a ~20K-token greedy prompt (cold, time it) → restart the engine (`pkill -f "vllm [s]erve"`, relaunch) → same prompt: expect **~7× faster**, byte-identical, server logs showing `Prefetch request completed ... retained keys` and `Retrieved N tokens`. For the full test also restart the *servers* first — hits then come from NVMe (`0 L1, N L2`).
6. **Wait 4+ minutes, replay again** — this specifically catches the heartbeat bug's signature (works-then-silently-stops). Expect hits, not a slow recompute.
7. **Perf reference (yours should be in the same ballpark):** cc1 ≈ 27 tok/s · cc64 ≈ 275 aggregate · cold prefill ≈ 1,375 tok/s @ BATCHTOK 3072.

## Troubleshooting (the mistakes already made for you)

| Symptom | Cause / fix |
|---|---|
| ptxas: `cvt with .e2m1x2 not supported on sm_120` building the fork | `TORCH_CUDA_ARCH_LIST=12.1` silently clamps to plain 12.0 under CUDA≥13. Use `"12.0f"` (family target) for the fork. exllamav3 and lmcache want plain `12.1`. |
| `cudaErrorUnsupportedPtxVersion` at startup or first MTP step | FA2 PTX vs driver 580. Pin TRITON_ATTN / TORCH_SDPA / drafter backend (already in the serve script). |
| Engine refuses graph decode | Needs `VLLM_EXL3_GRAPH_DECODE=1` **and** `--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}'` (script sets both under `STAGE=graph`). |
| `block_size <= max_num_batched_tokens < 2*block_size` error with LMC=1 | The mamba-align guard (block=1600). Set `BATCHTOK=3072`. Do **not** delete the guard — it prevents real KV corruption. |
| "LMCache chunk size should be a multiple of vLLM block size" | Chunk must be 1600 for this model (server script already is). |
| Cache stores work but replays never speed up | The heartbeat bug — you skipped Phase 8 step 2. Confirm with validation steps 3 and 6. |
| Large replays return 0 hits though chunks exist on NVMe | L1 too small to stage the prefix (silent). Raise `--l1-size-gb` (~106 MB × chunks-per-prefix per rank). |
| Striping configured but NCCL binds only one card | **`STRIPE` must be set when `ray start` runs, not when the engine starts.** NCCL executes inside the ray workers, which inherit their env from the raylet; the engine's own `STRIPE` has no effect on collectives. Confirm with `grep 'NET/IB : Using' <serve log>` — it must list one device per rail. `tp2-env.sh` defaults `STRIPE=2` for this reason; if you restart ray by hand, pass it explicitly. |
| Striping shows 1 rail busy, others idle | `NCCL_IB_SUBNET_AWARE_ROUTING` still on, or rail /24s missing. |
| −25–30% single-stream after enabling striping | Too many rails/channels (SM contention in decode graphs). 2 rails, 2 channels, one port per card. |
| Throughput drops ~18% at cc32 specifically | `VLLM_EXL3_PREFILL_RECONSTRUCT_M` left at default 128 (collides with 32×4 MTP batches). Use 256. |
| NCCL probe shows ~2.3 ms latency floor | You benchmarked beside the live engine — GB10 context timeslicing phantom. Quiesce first. |
| Container loses the GPU: `Failed to initialize NVML: Unknown Error` (often after `systemctl daemon-reload`) | Known NVIDIA container-toolkit issue with cgroups v2 + systemd driver ([#48](https://github.com/NVIDIA/nvidia-container-toolkit/issues/48)). **Prevent it** with `sudo nvidia-ctk system create-dev-char-symlinks --create-all` at boot. `docker restart <container>` recovers a container that has already lost access; `scripts/boot-stack-aa42.sh` detects and does this automatically. |
| Engine won't start: `Free memory on device cuda:0 (N/121 GiB) ... less than desired GPU memory utilization` | The cache servers are still holding the **previous** engine's KV cache through CUDA IPC — check `nvidia-smi --query-compute-apps=pid,used_memory --format=csv` for an `lmcache` PID holding tens of GB with tiny RSS. Restart the cache servers to release it. Correct restart order is always **kill engine -> cycle cache servers -> start engine**. |
| Garbage/multilingual output on shared-prefix requests under memory pressure | You skipped the #48425 port in `fork-ports.patch`. |

## Known limitations

Cold prefill pays ~8% for the LMCache batch guard. Sub-1600-token prompt tails always recompute (chunk granularity). No KLD measured on GB10 (quality inherited from the checkpoint's model-card KLD measurements, taken on an RTX 5090; extensive byte-exactness spot checks only). Concurrency is validated to 64 streams at short-to-mid context (cc1–8 at 16K and 32K, ~16K average context per stream at cc64) and single-stream to 227K tokens; many streams at 100K+ each is untested, and the 3.9M-token pool caps that regime at roughly 29 streams at 131K or 14 at the full 262K. L1 eviction in stock lmcache never demotes to NVMe (write-through covers it — but that's why the heartbeat+sizing fixes matter).

## Credits

turboderp (exllamav3) · the Gilded Gnosis fork authors · malaiwah (the checkpoint and its exemplary model card, plus the #4247 root-causing) · hasso5703 (chat template lineage) · MikeWang0316tw (LMCache #4247 investigation) · the sparkring project (NCCL patches, container pinning, cable-qual discipline) · the vLLM and LMCache maintainers.
