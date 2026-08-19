#!/bin/bash
# @reboot bootstrap for rank 0, intended for installation as a user crontab
# entry (no sudo). Installing it replaces whatever @reboot line the node's
# crontab currently carries. Brings a cold or partially-up host to a serving
# stack with the production configuration, in this order:
#   container -> GPU visibility -> rail addresses -> ray head -> wait for rank 1
#   -> cache servers (always cycled) -> wait for both to bind -> start-stack.sh.
#
# Invariants this ordering maintains:
# - The GPU must be visible INSIDE the container before anything else: a
#   container can come up after reboot with a dead NVML handle, which otherwise
#   surfaces minutes later as `CUDA error: invalid device ordinal` at engine
#   start. A container restart re-acquires the device.
# - The engine must be launched through start-stack.sh with the full production
#   environment: the serve script's bare defaults are an A/B baseline (no
#   prefix caching, no LMCache, fp16 KV), not the production configuration.
# - Cache servers are cycled, never reused: a surviving server holds a previous
#   engine's KV cache through CUDA IPC and starves the next engine start.
#
# Idempotent: exits early if the endpoint already answers, so it is safe to run
# by hand.

exec > "$HOME/work/qwen38-exl3/logs/boot-aa42.log" 2>&1
set -x

# Site config (same file the other scripts read); tested values as fallbacks.
[ -f "$HOME/work/qwen38-exl3/site.env" ] && . "$HOME/work/qwen38-exl3/site.env"
RAIL_PREFIX="${RAIL_PREFIX:-10.42}"
NETDEV1="${NETDEV1:-enp1s0f0np0}"; NETDEV2="${NETDEV2:-enp1s0f1np1}"
NETDEV3="${NETDEV3:-enP2p1s0f0np0}"; NETDEV4="${NETDEV4:-enP2p1s0f1np1}"

WS="$HOME/work/qwen38-exl3"
C0=ggrun
RANK1=198.18.200.2
TP="${TP:-2}"                                  # 1 = single Spark, 2 = the pair
MULTI=$([ "$TP" -gt 1 ] && echo 1 || echo 0)
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
for spec in "$RAIL_PREFIX.1.1/24 $NETDEV1" "$RAIL_PREFIX.2.1/24 $NETDEV2" \
            "$RAIL_PREFIX.3.1/24 $NETDEV3" "$RAIL_PREFIX.4.1/24 $NETDEV4"; do
    set -- $spec
    docker exec netadm ip addr add "$1" dev "$2" 2>/dev/null || true
done

# 4. Ray head. STRIPE is passed explicitly at ray start: NCCL runs in the ray
#    workers and inherits this from the raylet, so an omission here silently
#    costs a rail regardless of tp2-env.sh's default.
if [ "$MULTI" = "1" ]; then
    docker exec "$C0" bash -c "STRIPE=2 bash /ws/tp2-ray-head.sh"
else
    echo "boot: TP=1 - no ray cluster needed"
fi

if [ "$MULTI" = "1" ]; then
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
fi

# 6. Cache servers: always cycled, never reused — both ranks' cache servers are
#    cycled at TP=2; rank 0's alone at TP=1.
#    A surviving cache server holds the previous engine's KV cache through CUDA
#    IPC and never releases it, so the next engine start fails with
#    "Free memory on device cuda:0 ... less than desired GPU memory utilization".
#    Harmless on a true cold boot (nothing survives); essential whenever this
#    script is used to recover a warm machine.
docker exec "$C0" bash -c 'pkill -f "[l]mcache server"; true'
[ "$MULTI" = "1" ] && ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1"     "docker exec ggbuild bash -c 'pkill -f \"[l]mcache server\"; true'" || true
sleep 6

docker exec -d "$C0" bash -c "RANK=0 bash /ws/lmcache-server.sh > /ws/logs/lmcache-r0.log 2>&1"
[ "$MULTI" = "1" ] && ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1"     "docker exec -d ggbuild bash -c 'RANK=1 bash /ws/lmcache-server.sh > /ws/logs/lmcache-r1.log 2>&1'" || true

# 7. Wait for BOTH to bind before handing off (the launcher gates on both).
for i in $(seq 1 60); do
    if [ "$MULTI" = "1" ]; then
      ss -tln | grep -q ':6556 ' && ssh -o BatchMode=yes -o ConnectTimeout=5 "$RANK1" "ss -tln | grep -q ':6556 '" && break
    else
      ss -tln | grep -q ':6556 ' && break
    fi
    sleep 5
done
ss -tln | grep -q ':6556 ' || { echo "boot: FATAL — rank 0 cache server never bound 6556"; exit 1; }

# 8. Hand off to the gated launcher with the production configuration.
#    It re-checks everything above and refuses to start a degraded stack.
cd "$WS" && TP=$TP LMC=1 APC=1 MTPK=3 KVDTYPE=fp8 STAGE=graph BATCHTOK=3072 GPUMEM=0.70 \
    bash start-stack.sh
echo "boot: start-stack.sh exited $?"
