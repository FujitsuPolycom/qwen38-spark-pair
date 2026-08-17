#!/bin/bash
# Per-rank LMCache MP server — CS1600 profile (chunk 1600 = this model's mamba block).
# Usage: RANK=0|1 bash /ws/lmcache-server.sh   (inside the rank's container)
# Tiering: 8 GB lazy L1 in unified memory (must cover the longest replayed prefix —
# chunks are 106 MB/rank at TP2, 213 MB at TP1, so 8 GB = ~118K / ~59K tokens) plus
# fs_native L2 on NVMe (200 GB, buffered I/O; odirect also works).
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
