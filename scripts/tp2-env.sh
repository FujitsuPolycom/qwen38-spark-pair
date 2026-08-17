#!/bin/bash
# Shared distributed-runtime environment for the two-Spark TP2 probe.
# Sourced by the ray and serve scripts. RANK_IP must be set by the caller.
export NCCL_SOCKET_IFNAME=enp1s0f0np0
export GLOO_SOCKET_IFNAME=enp1s0f0np0
export NCCL_IB_HCA=rocep1s0f0:1
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export VLLM_HOST_IP=${RANK_IP:?set RANK_IP to this node fabric IP}
# Fallback knob: export NCCL_IB_DISABLE=1 to force TCP over the fabric link.

# Tier-1 striping (STRIPE=1): patched NCCL + 4-rail multi-cable striping.
# Measured: 176 Gb/s aggregate collective bw vs 106 single-card; even 44 Gbps/NIC.
# NOTE: subnet-aware routing OFF — its connector override defeats parallel rails.
if [ "${STRIPE:-0}" = "1" ]; then
  export LD_PRELOAD=/ws/nccl-patched/libnccl.so.2
  export VLLM_NCCL_SO_PATH=/ws/nccl-patched/libnccl.so.2
  export NCCL_NET=IB
  export NCCL_IB_HCA=rocep1s0f0,rocep1s0f1,roceP2p1s0f0,roceP2p1s0f1
  export NCCL_IB_GID_INDEX=5
  export NCCL_IB_MERGE_NICS=0
  export NCCL_CROSS_NIC=1
  export NCCL_IB_SUBNET_AWARE_ROUTING=0
  export NCCL_ALGO=Ring
  export NCCL_SKIP_TREE_CONNECT=1
  export NCCL_CUMEM_ENABLE=0
  export NCCL_MIN_NCHANNELS=4
  export NCCL_MAX_NCHANNELS=4
fi

# STRIPE=2: dual-card 2-rail (one port per card, 2 channels) — keeps both PCIe
# pipes with half the channel kernels per collective. Quiet-GPU probe: 37 us
# small ops, 158 us @256 rows, 160 Gb/s large.
if [ "${STRIPE:-0}" = "2" ]; then
  export LD_PRELOAD=/ws/nccl-patched/libnccl.so.2
  export VLLM_NCCL_SO_PATH=/ws/nccl-patched/libnccl.so.2
  export NCCL_NET=IB
  export NCCL_IB_HCA=rocep1s0f0,roceP2p1s0f0
  export NCCL_IB_GID_INDEX=5
  export NCCL_IB_MERGE_NICS=0
  export NCCL_CROSS_NIC=1
  export NCCL_IB_SUBNET_AWARE_ROUTING=0
  export NCCL_ALGO=Ring
  export NCCL_SKIP_TREE_CONNECT=1
  export NCCL_CUMEM_ENABLE=0
  export NCCL_MIN_NCHANNELS=2
  export NCCL_MAX_NCHANNELS=2
fi
