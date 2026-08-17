#!/bin/bash
# Start ray head for TP2 serving. Run inside ggrun on spark-aa42.
# Site config: scripts/site.env if present, else the reference values below.
[ -f "${SITE_ENV:-/ws/site.env}" ] && . "${SITE_ENV:-/ws/site.env}"
set -u
. /ws/venv/bin/activate
RANK_IP="${FABRIC_RANK0:-198.18.200.1}" . /ws/tp2-env.sh
ray stop --force > /dev/null 2>&1
ray start --head --node-ip-address="${FABRIC_RANK0:-198.18.200.1}" --port=6379 --num-gpus=1 --disable-usage-stats
