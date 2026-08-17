#!/bin/bash
# @reboot bootstrap for rank 0 (spark-aa42).
#
# Rewritten after a real power event on 2026-08-17, where the previous version
# reached the serve step and died with `CUDA error: invalid device ordinal`
# because the container comes back from a reboot unable to see the GPU. It
# would also have produced a misconfigured stack had it survived: it launched
# the serve script with NO environment, so every gate fell back to its default
# (APC off, MTP3, KV auto, batch 8192, LMCache off) and the cache servers were
# never started at all.
#
# Sequence: container -> GPU visibility -> rails -> ray head -> wait for rank 1
# -> cache server -> wait for rank 1's cache server -> start-stack.sh.
#
# Idempotent: exits early if the endpoint already answers, so it is safe to run
# by hand to test.  Installed via user crontab; no sudo required.

exec > "$HOME/work/qwen38-exl3/logs/boot-aa42.log" 2>&1
set -x

WS="$HOME/work/qwen38-exl3"
C0=ggrun
RANK1=198.18.200.2
API=http://127.0.0.1:8000

# 0. Already healthy? Do nothing.
if curl -sf -m 5 "$API/v1/models" >/dev/null 2>&1; then
    echo "boot: endpoint already serving, nothing to do"; exit 0
fi

# 1. Container up (docker restart policy brings it back).
for i in $(seq 1 60); do
    docker inspect -f '{{.State.Running}}' "$C0" 2>/dev/null | grep -q true && break
    sleep 5
done

# 2. GPU visible INSIDE the container. After a reboot the container often comes
#    back with a dead NVML handle ("Failed to initialize NVML"), which surfaces
#    later as the engine's "CUDA error: invalid device ordinal". A container
#    restart re-acquires the device.
if ! docker exec "$C0" nvidia-smi -L >/dev/null 2>&1; then
    echo "boot: no GPU inside $C0 — restarting the container"
    docker restart "$C0"
    sleep 20
    if ! docker exec "$C0" nvidia-smi -L >/dev/null 2>&1; then
        echo "boot: FATAL — GPU still not visible in $C0 after restart; aborting"
        exit 1
    fi
fi

# 3. Per-cable rail addresses (NET_ADMIN helper; survives container restarts).
for spec in "10.42.1.1/24 enp1s0f0np0" "10.42.2.1/24 enp1s0f1np1" \
            "10.42.3.1/24 enP2p1s0f0np0" "10.42.4.1/24 enP2p1s0f1np1"; do
    set -- $spec
    docker exec netadm ip addr add "$1" dev "$2" 2>/dev/null || true
done

# 4. Ray head. STRIPE is passed explicitly even though tp2-env.sh now defaults
#    to 2 — NCCL runs in the ray workers and inherits this at `ray start` time,
#    so being wrong here silently costs a rail.
docker exec "$C0" bash -c "STRIPE=2 bash /ws/tp2-ray-head.sh"

# 5. Wait for rank 1 to join — and re-trigger its join periodically.
#    Race this heals: tp2-ray-head.sh does `ray stop --force` before
#    `ray start --head`, so a rank 1 that joined the PREVIOUS head is silently
#    orphaned when we replace it. Rank 1's own loop breaks on a successful
#    join, so it never notices and never retries. Rank 0 is authoritative here.
for i in $(seq 1 120); do
    n=$(docker exec "$C0" bash -c ". /ws/venv/bin/activate && ray status 2>/dev/null" | grep -c "1 node_")
    [ "${n:-0}" -ge 2 ] && break
    if [ $((i % 6)) -eq 0 ]; then
        echo "boot: only $n ray node(s) after $((i*5))s — re-triggering rank 1 join"
        ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1"             "docker exec ggbuild bash -c 'STRIPE=2 bash /ws/tp2-ray-worker.sh'" >/dev/null 2>&1 || true
    fi
    sleep 5
done

# 6. Cache servers: ALWAYS cycle both, never reuse.
#    A surviving cache server holds the previous engine's KV cache through CUDA
#    IPC and never releases it, so the next engine start fails with
#    "Free memory on device cuda:0 ... less than desired GPU memory utilization".
#    Harmless on a true cold boot (nothing survives); essential whenever this
#    script is used to recover a warm machine.
docker exec "$C0" bash -c 'pkill -f "[l]mcache server"; true'
ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1"     "docker exec ggbuild bash -c 'pkill -f \"[l]mcache server\"; true'" || true
sleep 6

docker exec -d "$C0" bash -c "RANK=0 bash /ws/lmcache-server.sh > /ws/logs/lmcache-r0.log 2>&1"
ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1"     "docker exec -d ggbuild bash -c 'RANK=1 bash /ws/lmcache-server.sh > /ws/logs/lmcache-r1.log 2>&1'" || true

# 7. Wait for BOTH to bind before handing off (the launcher gates on both).
for i in $(seq 1 60); do
    ss -tln | grep -q ':6556 ' &&       ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1" "ss -tln | grep -q ':6556 '" && break
    sleep 3
done
ss -tln | grep -q ':6556 ' || { echo "boot: FATAL — rank 0 cache server never bound 6556"; exit 1; }

# 8. Hand off to the gated launcher with the production configuration.
#    It re-checks everything above and refuses to start a degraded stack.
cd "$WS" && LMC=1 APC=1 STRIPE=2 MTPK=2 KVDTYPE=fp8 STAGE=graph BATCHTOK=3072 GPUMEM=0.70 \
    bash start-stack.sh
echo "boot: start-stack.sh exited $?"
