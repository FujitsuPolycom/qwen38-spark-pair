#!/bin/bash
# @reboot bootstrap for rank 1 (spark-931e): container -> GPU visibility ->
# rails -> join ray -> cache server. Rank 0 waits for both the ray join and
# this node's cache server before launching the engine.
#
# Idempotent: safe to run by hand.

exec > "$HOME/work/qwen38-exl3/logs/boot-931e.log" 2>&1
set -x

C1=ggbuild

for i in $(seq 1 60); do
    docker inspect -f '{{.State.Running}}' "$C1" 2>/dev/null | grep -q true && break
    sleep 5
done

# Same post-reboot NVML hazard as rank 0.
if ! docker exec "$C1" nvidia-smi -L >/dev/null 2>&1; then
    echo "boot: no GPU inside $C1 — restarting the container"
    docker restart "$C1"
    sleep 20
    if ! docker exec "$C1" nvidia-smi -L >/dev/null 2>&1; then
        echo "boot: FATAL — GPU still not visible in $C1 after restart; aborting"
        exit 1
    fi
fi

for spec in "10.42.1.2/24 enp1s0f0np0" "10.42.2.2/24 enp1s0f1np1" \
            "10.42.3.2/24 enP2p1s0f0np0" "10.42.4.2/24 enP2p1s0f1np1"; do
    set -- $spec
    docker exec netadm ip addr add "$1" dev "$2" 2>/dev/null || true
done

# Join ray with the striping env (see rank 0 step 4).
for i in $(seq 1 120); do
    docker exec "$C1" bash -c "STRIPE=2 bash /ws/tp2-ray-worker.sh" && break
    sleep 10
done

# Cache server for this rank; rank 0 blocks on port 6556 before launching.
docker exec -d "$C1" bash -c "RANK=1 bash /ws/lmcache-server.sh > /ws/logs/lmcache-r1.log 2>&1"
echo "boot: rank 1 ready"
