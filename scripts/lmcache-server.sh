#!/bin/bash
# Per-rank LMCache MP server (CS512 profile, adapted from the sparkring GLM lane).
# Usage: RANK=0|1 bash /ws/lmcache-server.sh   (inside the rank's container)
# Tiering is GB10-aware: tiny lazy L1 (1 GB, lives in unified memory) and the
# real capacity on NVMe L2 (200 GB, O_DIRECT so the page cache stays out of
# unified memory too).
set -u
. /ws/venv/bin/activate
export LMCACHE_DISABLE_BANNER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
exec lmcache server \
  --instance-id "qwen38-r${RANK}-cs1600" \
  --host 0.0.0.0 --port 6556 \
  --chunk-size 1600 \
  --max-gpu-workers 2 --max-cpu-workers 2 \
  --supported-transfer-mode lmcache_driven \
  --l1-size-gb 8 --l1-use-lazy --l1-init-size-gb 0 --eviction-policy LRU \
  --l2-adapter '{"type":"fs_native","base_path":"/ws/lmcache-l2","num_workers":2,"use_odirect":false,"max_capacity_gb":200}'
