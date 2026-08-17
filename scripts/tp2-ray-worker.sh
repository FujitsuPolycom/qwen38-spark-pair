#!/bin/bash
# Join ray cluster as rank-1 worker. Run inside ggrun on spark-931e.
# Site config: scripts/site.env if present, else the reference values below.
[ -f "${SITE_ENV:-/ws/site.env}" ] && . "${SITE_ENV:-/ws/site.env}"
set -u
. /ws/venv/bin/activate
RANK_IP="${FABRIC_RANK1:-198.18.200.2}" . /ws/tp2-env.sh
ray stop --force > /dev/null 2>&1
ray start --address="${FABRIC_RANK0:-198.18.200.1}:6379" --node-ip-address="${FABRIC_RANK1:-198.18.200.2}" --num-gpus=1 --disable-usage-stats
