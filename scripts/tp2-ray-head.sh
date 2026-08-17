#!/bin/bash
# Start ray head for the TP2 probe. Run inside ggrun on spark-aa42.
set -u
. /ws/venv/bin/activate
RANK_IP=198.18.200.1 . /ws/tp2-env.sh
ray stop --force > /dev/null 2>&1
ray start --head --node-ip-address=198.18.200.1 --port=6379 --num-gpus=1 --disable-usage-stats
