#!/bin/bash
# @reboot bootstrap for spark-aa42 (rank 0): wait for docker + ggrun container
# (docker restart policy brings it up), start ray head, wait for the 931e worker
# to join, then launch the TP2 serve. Installed via user crontab; logs to
# ~/work/qwen38-exl3/logs/boot-aa42.log. No sudo required.
exec > "$HOME/work/qwen38-exl3/logs/boot-aa42.log" 2>&1
set -x
for i in $(seq 1 60); do docker inspect -f '{{.State.Running}}' ggrun 2>/dev/null | grep -q true && break; sleep 5; done
for spec in "10.42.1.1/24 enp1s0f0np0" "10.42.2.1/24 enp1s0f1np1" "10.42.3.1/24 enP2p1s0f0np0" "10.42.4.1/24 enP2p1s0f1np1"; do set -- $spec; docker exec netadm ip addr add $1 dev $2 2>/dev/null || true; done
docker exec ggrun bash /ws/tp2-ray-head.sh
# wait until both nodes registered (worker joins from 931e's own boot script)
for i in $(seq 1 120); do
  n=$(docker exec ggrun bash -c ". /ws/venv/bin/activate && ray status 2>/dev/null" | grep -c "1 node_")
  [ "${n:-0}" -ge 2 ] && break
  sleep 5
done
docker exec -d ggrun bash -c "bash /ws/run-serve-tp2-v2.sh > /ws/logs/serve-boot.log 2>&1"
