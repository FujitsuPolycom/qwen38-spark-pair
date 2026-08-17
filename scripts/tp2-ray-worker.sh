#!/bin/bash
# Join ray cluster as rank-1 worker. Run inside ggrun on spark-931e.
set -u
. /ws/venv/bin/activate
RANK_IP=198.18.200.2 . /ws/tp2-env.sh
ray stop --force > /dev/null 2>&1
ray start --address=198.18.200.1:6379 --node-ip-address=198.18.200.2 --num-gpus=1 --disable-usage-stats
