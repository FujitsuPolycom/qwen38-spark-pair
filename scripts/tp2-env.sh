#!/bin/bash
# Shared distributed-runtime environment for two-Spark TP2 serving.
# Sourced by the ray and serve scripts. RANK_IP must be set by the caller.
# Site config: scripts/site.env if present, else the reference values below.
[ -f "${SITE_ENV:-/ws/site.env}" ] && . "${SITE_ENV:-/ws/site.env}"
export NCCL_SOCKET_IFNAME="${NETDEV1:-enp1s0f0np0}"
export GLOO_SOCKET_IFNAME="${NETDEV1:-enp1s0f0np0}"
export NCCL_IB_HCA="${RDMA_CARD1:-rocep1s0f0}:1"
# GID 3 = RoCE v2 entry of the /30 fabric address — used only by the unstriped
# (STRIPE=0) fallback; the STRIPE blocks below use the rail /24's GID instead.
export NCCL_IB_GID_INDEX=3
export NCCL_DEBUG=INFO
export VLLM_HOST_IP=${RANK_IP:?set RANK_IP to this node fabric IP}
# Fallback knob: export NCCL_IB_DISABLE=1 to force TCP over the fabric link.

# STRIPE defaults to 2 (the validated production transport). It MUST default here,
# because NCCL runs inside the *ray workers*, which inherit this env from the raylet
# at `ray start` time. Setting STRIPE only when launching the engine has no effect on
# collectives and silently produces a single-rail run.
STRIPE="${STRIPE:-2}"

# Tier-1 striping (STRIPE=1): patched NCCL + 4-rail multi-cable striping.
# Measured: 176 Gb/s aggregate collective bw vs 106 single-card; even 44 Gbps/NIC.
# NOTE: subnet-aware routing OFF — its connector override defeats parallel rails.
if [ "$STRIPE" = "1" ]; then
  export LD_PRELOAD=/ws/nccl-patched/libnccl.so.2
  export VLLM_NCCL_SO_PATH=/ws/nccl-patched/libnccl.so.2
  export NCCL_NET=IB
  export NCCL_IB_HCA="${RDMA_CARD1:-rocep1s0f0},rocep1s0f1,${RDMA_CARD2:-roceP2p1s0f0},roceP2p1s0f1"
  export NCCL_IB_GID_INDEX="${NCCL_GID_INDEX:-5}"
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
if [ "$STRIPE" = "2" ]; then
  export LD_PRELOAD=/ws/nccl-patched/libnccl.so.2
  export VLLM_NCCL_SO_PATH=/ws/nccl-patched/libnccl.so.2
  export NCCL_NET=IB
  export NCCL_IB_HCA="${RDMA_CARD1:-rocep1s0f0},${RDMA_CARD2:-roceP2p1s0f0}"
  export NCCL_IB_GID_INDEX="${NCCL_GID_INDEX:-5}"
  export NCCL_IB_MERGE_NICS=0
  export NCCL_CROSS_NIC=1
  export NCCL_IB_SUBNET_AWARE_ROUTING=0
  export NCCL_ALGO=Ring
  export NCCL_SKIP_TREE_CONNECT=1
  export NCCL_CUMEM_ENABLE=0
  export NCCL_MIN_NCHANNELS=2
  export NCCL_MAX_NCHANNELS=2
fi
