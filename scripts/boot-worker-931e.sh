#!/bin/bash
# @reboot bootstrap for spark-931e (rank 1): wait for docker + ggbuild container,
# then join the ray cluster, retrying until the aa42 head answers. Installed via
# user crontab; logs to ~/work/qwen38-exl3/logs/boot-931e.log. No sudo required.
exec > "$HOME/work/qwen38-exl3/logs/boot-931e.log" 2>&1
set -x
for i in $(seq 1 60); do docker inspect -f '{{.State.Running}}' ggbuild 2>/dev/null | grep -q true && break; sleep 5; done
for spec in "10.42.1.2/24 enp1s0f0np0" "10.42.2.2/24 enp1s0f1np1" "10.42.3.2/24 enP2p1s0f0np0" "10.42.4.2/24 enP2p1s0f1np1"; do set -- $spec; docker exec netadm ip addr add $1 dev $2 2>/dev/null || true; done
for i in $(seq 1 120); do
  docker exec ggbuild bash /ws/tp2-ray-worker.sh && break
  sleep 10
done
