#!/bin/bash
# TP2 serve for malaiwah/Qwen3.8-27B-EXL3-K5K6-hydrated across spark-aa42 + spark-931e.
# Run inside the ggrun container on spark-aa42 AFTER ray head+worker are up.
#
# A/B gates (all default to the current validated config):
#   STAGE=graph|eager   graph = EXL3 CUDA-graph decode (validated 3/3 vs eager, +25%)
#   APC=0|1             1 = prefix caching + mamba align (scheduler fix ported, 20/20
#                       regression tests; not yet live-validated)
#   FP8PREFILL=1|0      FP8 prefill GEMM: 662 -> 1433 tok/s measured; prefill-only
#                       E4M3 numerics, decode keeps exact trellis kernels
#   MTPK=3              speculative depth (community favors 2 under concurrency)
#   BATCHTOK=8192       max-num-batched-tokens (sweep: 3072/4096/8192)
#   KVDTYPE=auto        fp8 = FP8 KV cache (community: avoids -33% long-depth penalty)
set -u
. /ws/venv/bin/activate
RANK_IP=198.18.200.1 . /ws/tp2-env.sh

STAGE="${STAGE:-graph}"
APC="${APC:-0}"
SPEC="${SPEC:-mtp}"   # mtp = built-in MTP head; dspark = RadixArk 1.36B drafter (see DSPARK-TRIAL.md)
MTPK="${MTPK:-3}"
BATCHTOK="${BATCHTOK:-8192}"
KVDTYPE="${KVDTYPE:-auto}"
export VLLM_EXL3_PREFILL_FP8="${FP8PREFILL:-1}"
export VLLM_EXL3_PREFILL_RECONSTRUCT_M="${RECON_M:-256}"

EXTRA=()
if [ "$STAGE" = "graph" ]; then
  export VLLM_EXL3_GRAPH_DECODE=1
  EXTRA+=(--compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}')
else
  EXTRA+=(--enforce-eager)
fi
if [ "$APC" = "1" ]; then EXTRA+=(--enable-prefix-caching --mamba-cache-mode align); fi
if [ "${LMC:-0}" = "1" ]; then
  export LMCACHE_DISABLE_BANNER=1
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False
  EXTRA+=(--kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_connector_module_path":"lmcache.integration.vllm.lmcache_mp_connector","kv_role":"kv_both","kv_load_failure_policy":"recompute","kv_connector_extra_config":{"lmcache.mp.server_urls":["tcp://192.168.0.200:6556","tcp://192.168.0.174:6556"],"lmcache.mp.mq_timeout":60,"lmcache.mp.heartbeat_interval":10}}')
fi
if [ "$KVDTYPE" != "auto" ]; then EXTRA+=(--kv-cache-dtype "$KVDTYPE"); fi

exec vllm serve /ws/model/Qwen3.8-27B-EXL3-K5K6-hydrated \
  --served-model-name qwen38 --quantization exl3 "${EXTRA[@]}" \
  --mm-processor-kwargs '{"truncation":false}' --mm-encoder-attn-backend TORCH_SDPA --attention-backend TRITON_ATTN \
  --chat-template /ws/chat_template_agentic.jinja --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --speculative-config "$( if [ "$SPEC" = "dspark" ]; then
      echo "{\"method\":\"dspark\",\"model\":\"/ws/model/Qwen3.8-27B-DSpark-gg\",\"attention_backend\":\"TRITON_ATTN\"}"
    else
      echo "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$MTPK,\"attention_backend\":\"TRITON_ATTN\"}"
    fi )" \
  --max-model-len 262144 --max-num-batched-tokens "$BATCHTOK" --gpu-memory-utilization 0.40 --max-num-seqs 64 \
  --tensor-parallel-size 2 --distributed-executor-backend ray \
  --host 0.0.0.0 --port 8000
