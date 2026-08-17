#!/bin/bash
# start-stack.sh — gated launcher for the Qwen3.8-27B EXL3 TP2 stack.
#
# Every precondition is checked BEFORE launch and the engine is verified AFTER,
# so a failed start exits non-zero naming the gate that failed instead of
# leaving a half-dead stack and a 200-line traceback whose useful line is
# somewhere in the middle.
#
# Run on rank 0. Requires passwordless ssh to rank 1.
#
#   bash start-stack.sh                 # launch with the production gates
#   GPUMEM=0.40 bash start-stack.sh     # override any serve gate
#   bash start-stack.sh --check         # run preconditions only, launch nothing
#
# Exit codes: 0 ok · 10 stale process · 11 ray · 12 cache servers · 13 model
#             14 transport · 15 lmcache patch · 16 disk · 20 engine never came up

set -uo pipefail

# ---- site configuration ----------------------------------------------------
RANK1_HOST="${RANK1_HOST:-198.18.200.2}"     # rank 1 over the fabric
C0="${C0:-ggrun}"                             # rank 0 serving container
C1="${C1:-ggbuild}"                           # rank 1 serving container
WS="${WS:-/home/code/work/qwen38-exl3}"       # host work dir (= /ws in container)
API="${API:-http://127.0.0.1:8000}"
LMCACHE_PORT="${LMCACHE_PORT:-6556}"
TP="${TP:-2}"                                 # 1 = single Spark, 2 = the pair
TP_SIZE="$TP"
MULTI=$([ "$TP" -gt 1 ] && echo 1 || echo 0)  # 1 = rank-1 / transport checks apply

# ---- serve gates (passed through to run-serve-tp2-v2.sh) -------------------
LMC="${LMC:-1}"; APC="${APC:-1}"; MTPK="${MTPK:-2}"
# Striping is a two-node concept: with TP=1 there are no collectives to stripe.
if [ "$MULTI" = "1" ]; then STRIPE="${STRIPE:-2}"; else STRIPE=0; fi
KVDTYPE="${KVDTYPE:-fp8}"; STAGE="${STAGE:-graph}"; BATCHTOK="${BATCHTOK:-3072}"
GPUMEM="${GPUMEM:-0.70}"
LOG="${LOG:-$WS/logs/serve-lmc.log}"

# ---- timeouts --------------------------------------------------------------
RAY_SETTLE_TIMEOUT="${RAY_SETTLE_TIMEOUT:-180}"   # wait for GPUs to be released
ENGINE_TIMEOUT="${ENGINE_TIMEOUT:-900}"           # wait for /v1/models

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

pass() { printf '  \033[32mPASS\033[0m  %s\n' "$1"; }
fail() { printf '  \033[31mFAIL\033[0m  %s\n' "$1" >&2; }
info() { printf '        %s\n' "$1"; }
step() { printf '\n\033[1m%s\033[0m\n' "$1"; }

die() { fail "$2"; echo; echo "Stack NOT started. Fix the above and re-run." >&2; exit "$1"; }

step "Preconditions"

# --- gate 1: no stale engine ------------------------------------------------
# A previous engine still holding GPUs is the usual cause of a TP placement
# failure on the next launch.
#
# --check MUST NOT mutate anything: reporting "an engine is running" is the
# correct result there, not a reason to kill it.
if docker exec "$C0" bash -c 'pgrep -f "vllm [s]erve" >/dev/null' 2>/dev/null; then
    if [ "$CHECK_ONLY" = "1" ]; then
        pass "an engine is already running (check mode: left untouched)"
        info "a real launch would stop it first"
        ENGINE_ALREADY_UP=1
    else
    info "an engine is running; stopping it"
    docker exec "$C0" bash -c 'pkill -f "vllm [s]erve"' 2>/dev/null
    for _ in $(seq 1 60); do
        docker exec "$C0" bash -c 'pgrep -f "vllm [s]erve" >/dev/null' 2>/dev/null || break
        sleep 2
    done
    docker exec "$C0" bash -c 'pgrep -f "vllm [s]erve" >/dev/null' 2>/dev/null \
        && die 10 "previous engine would not exit (still holding GPUs)"
    pass "no stale engine process"
    fi
else
    pass "no stale engine process"
fi
ENGINE_ALREADY_UP="${ENGINE_ALREADY_UP:-0}"

# --- gate 2: ray cluster is whole AND its GPUs are free (multi-node only) ---
# TP=1 runs in-process with no ray executor, so there is no cluster to check.
if [ "$MULTI" = "0" ]; then
    pass "single node (TP=1): ray, striping and rank-1 checks skipped"
else
# THIS is the gate that matters most. Killing an engine does not instantly
# release its ray actors; launching into a half-released cluster makes vLLM
# see 1 GPU, fail the TP2 placement group, and emit a traceback whose real
# cause ("Tensor parallel size (2) exceeds available GPUs (1)") scrolls past.
ray_gpus() {  # -> "<free>/<total>" or empty
    docker exec "$C0" bash -c '. /ws/venv/bin/activate 2>/dev/null && ray status 2>/dev/null' \
        | grep -oE '[0-9.]+/[0-9.]+ GPU' | head -1 | awk '{print $1}'
}
deadline=$((SECONDS + RAY_SETTLE_TIMEOUT))
while :; do
    g=$(ray_gpus)
    used=${g%%/*}; total=${g##*/}
    # In check mode a live engine legitimately holds its GPUs; only require
    # that the cluster has enough of them.
    if [ -n "$g" ] && [ "${total%.*}" -ge "$TP_SIZE" ] 2>/dev/null \
       && { [ "$ENGINE_ALREADY_UP" = "1" ] || [ "$(printf '%.0f' "${used:-1}")" -eq 0 ]; }; then
        if [ "$ENGINE_ALREADY_UP" = "1" ]; then
            pass "ray: ${total%.*} GPUs present (${used} in use by the running engine)"
        else
            pass "ray: ${total%.*} GPUs present, all free"
        fi
        break
    fi
    [ $SECONDS -ge $deadline ] && die 11 "ray never reached ${TP_SIZE} free GPUs (saw '${g:-no ray}'). Check both nodes joined: docker exec $C0 bash -c '. /ws/venv/bin/activate && ray status'"
    info "waiting for ray to release GPUs (${g:-no response})"
    sleep 5
done

nodes=$(docker exec "$C0" bash -c '. /ws/venv/bin/activate 2>/dev/null && ray status 2>/dev/null' | grep -c '^ 1 node_')
[ "$nodes" -ge "$TP_SIZE" ] || die 11 "ray sees $nodes node(s), need $TP_SIZE — rank 1 has not joined"
pass "ray: $nodes nodes joined"
fi

# --- gate 3: cache servers listening on both ranks --------------------------
if [ "$LMC" = "1" ]; then
    ss -tln 2>/dev/null | grep -q ":$LMCACHE_PORT " \
        || die 12 "no LMCache server on rank 0 port $LMCACHE_PORT — start it first: docker exec -d $C0 bash -c 'RANK=0 bash /ws/lmcache-server.sh > /ws/logs/lmcache-r0.log 2>&1'"
    if [ "$MULTI" = "1" ]; then
        ssh -o BatchMode=yes -o ConnectTimeout=10 "$RANK1_HOST" "ss -tln | grep -q ':$LMCACHE_PORT '" \
            || die 12 "no LMCache server on rank 1 ($RANK1_HOST) port $LMCACHE_PORT"
        pass "LMCache servers listening on both ranks"
    else
        pass "LMCache server listening (single node)"
    fi

    # The heartbeat fix is mandatory: without it the servers reap the engine
    # ~2.5 min in and every lookup silently returns 0 while stores keep working.
    if docker exec "$C0" bash -c 'grep -q "if self._heartbeats is not None:" /ws/venv/lib/python3.12/site-packages/lmcache/integration/vllm/vllm_multi_process_adapter.py' 2>/dev/null; then
        die 15 "lmcache heartbeat patch is NOT applied — cache loads would silently stop working. Apply patches/lmcache-0.5.2-heartbeat-fix.patch"
    fi
    pass "lmcache heartbeat patch present"

    avail=$(df -BG --output=avail "$WS" 2>/dev/null | tail -1 | tr -dc '0-9')
    [ "${avail:-0}" -ge 20 ] || die 16 "only ${avail}G free on $WS — the NVMe cache tier needs headroom"
    pass "disk: ${avail}G free for the L2 tier"
fi

# --- gate 4: model present on both ranks ------------------------------------
docker exec "$C0" bash -c 'ls /ws/model/Qwen3.8-27B-EXL3-K5K6-hydrated/*.safetensors >/dev/null 2>&1' \
    || die 13 "model weights missing on rank 0"
if [ "$MULTI" = "1" ]; then
ssh -o BatchMode=yes -o ConnectTimeout=10 "$RANK1_HOST" \
    "ls $WS/model/Qwen3.8-27B-EXL3-K5K6-hydrated/*.safetensors >/dev/null 2>&1" \
    || die 13 "model weights missing on rank 1 — TP2 needs the checkpoint on both nodes"
    pass "model weights present on both ranks"
else
    pass "model weights present"
fi

# --- gate 5: patched NCCL present if striping -------------------------------
if [ "${STRIPE:-0}" != "0" ]; then
    docker exec "$C0" bash -c 'test -f /ws/nccl-patched/libnccl.so.2' \
        || die 14 "STRIPE=$STRIPE but /ws/nccl-patched/libnccl.so.2 is missing on rank 0"
    ssh -o BatchMode=yes -o ConnectTimeout=10 "$RANK1_HOST" "test -f $WS/nccl-patched/libnccl.so.2" \
        || die 14 "STRIPE=$STRIPE but the patched NCCL is missing on rank 1"
    pass "patched NCCL present on both ranks (STRIPE=$STRIPE)"
fi

# --- gate 6: ray must have been started WITH the striping env ---------------
# NCCL runs inside the ray WORKERS, which inherit their environment from the
# raylet at `ray start` time — not from the engine. A ray cluster started
# without STRIPE produces a silent single-rail run that every other gate here
# passes happily. Uses `strings` to read the NUL-separated environ.
if [ "${STRIPE:-0}" != "0" ]; then
    want_hca=$(docker exec "$C0" bash -c "STRIPE=$STRIPE RANK_IP=x . /ws/tp2-env.sh >/dev/null 2>&1; echo \$NCCL_IB_HCA" 2>/dev/null | tail -1)
    # Scan EVERY raylet-matching pid: `pgrep -f raylet | head -1` can land on a
    # stale/empty process and report the env as unset (false positive).
    if [ -n "$want_hca" ]; then
        got_hca=$(docker exec "$C0" bash -c 'for p in $(pgrep -f raylet); do strings /proc/$p/environ 2>/dev/null | sed -n "s/^NCCL_IB_HCA=//p"; done | sort -u | tail -1' 2>/dev/null)
        if [ "$got_hca" != "$want_hca" ]; then
            fail "ray was started WITHOUT the STRIPE=$STRIPE environment"
            info "raylet has NCCL_IB_HCA='${got_hca:-unset}', expected '$want_hca'"
            info "NCCL runs in the ray workers, which inherit this at 'ray start' time,"
            info "so the engine's own STRIPE setting cannot fix it. Restart ray:"
            info "  [r0] docker exec $C0 bash -c 'STRIPE=$STRIPE bash /ws/tp2-ray-head.sh'"
            info "  [r1] docker exec $C1 bash -c 'STRIPE=$STRIPE bash /ws/tp2-ray-worker.sh'"
            die 17 "transport would silently run single-rail"
        fi
        pass "ray carries the striping env (NCCL_IB_HCA=$got_hca)"
    fi
fi

if [ "$CHECK_ONLY" = "1" ]; then
    step "All preconditions pass (--check: nothing launched)"
    exit 0
fi

# ---- launch ----------------------------------------------------------------
step "Launching engine"
info "LMC=$LMC APC=$APC STRIPE=$STRIPE MTPK=$MTPK KVDTYPE=$KVDTYPE STAGE=$STAGE BATCHTOK=$BATCHTOK GPUMEM=$GPUMEM"
info "log: $LOG"
# No host-side truncate: the file is root-owned inside the container, and bash
# reports a failed `>` before any `2>/dev/null` on the same line can suppress
# it. The container's own redirect below truncates it as root regardless.
docker exec -d "$C0" bash -c \
    "LMC=$LMC APC=$APC STRIPE=$STRIPE MTPK=$MTPK KVDTYPE=$KVDTYPE STAGE=$STAGE BATCHTOK=$BATCHTOK GPUMEM=$GPUMEM bash /ws/run-serve-tp2-v2.sh > ${LOG/$WS//ws} 2>&1"

# ---- verify ----------------------------------------------------------------
step "Waiting for the engine to answer (timeout ${ENGINE_TIMEOUT}s)"
deadline=$((SECONDS + ENGINE_TIMEOUT))
up=0
while [ $SECONDS -lt $deadline ]; do
    curl -sf -m 5 "$API/v1/models" >/dev/null 2>&1 && { up=1; break; }
    # Fail fast rather than burning the whole timeout on a dead process.
    docker exec "$C0" bash -c 'pgrep -f "vllm [s]erve" >/dev/null' 2>/dev/null || break
    sleep 10
done

if [ "$up" != "1" ]; then
    fail "engine did not come up"
    step "Probable cause (first real error in the log, not the traceback tail)"
    # The useful line is usually an early WARNING/ERROR far above the final
    # traceback — e.g. the TP-vs-available-GPU mismatch, or an OOM during
    # KV profiling. Surface those first.
    grep -nE "exceeds available GPUs|No available memory|out of memory|CUDA out of memory|ValueError|Cannot allocate|placement group|Failed to (bind|allocate)|ptxas|Unsupported|not supported" "$LOG" \
        | grep -viE "min_frames|max_frames|not documented" | head -5 \
        || tail -5 "$LOG"
    echo
    info "full log: $LOG"
    exit 20
fi
pass "engine answered $API/v1/models"

kv=$(grep -oE 'GPU KV cache size: [0-9,]+ tokens' "$LOG" | tail -1)
[ -n "$kv" ] && pass "$kv"
conc=$(grep -oE 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' "$LOG" | tail -1)
[ -n "$conc" ] && info "$conc"

if [ "$LMC" = "1" ]; then
    # The heartbeat thread starts LAZILY on the first request, not at boot, so
    # a probe is required before this check means anything.
    curl -sf -m 60 "$API/v1/completions" -H 'Content-Type: application/json' \
        -d '{"model":"qwen38","prompt":"ping","max_tokens":1,"temperature":0}' >/dev/null 2>&1
    sleep 3
    # Confirms the heartbeat patch is live in this process, not just on disk.
    if grep -q "PeriodicThread: lmcache-heartbeat" "$LOG"; then
        pass "lmcache heartbeat thread started"
    else
        fail "lmcache heartbeat thread NOT started — cache lookups will silently stop in ~2.5 min"
        info "the engine is serving, but the KV cache tier is degraded"
    fi
fi

# Prove the transport actually came up striped rather than trusting the env.
if [ "${STRIPE:-0}" != "0" ]; then
    case "$STRIPE" in 2) want_rails=2 ;; 1) want_rails=4 ;; *) want_rails=1 ;; esac
    bound=$(grep -a "NET/IB : Using" "$LOG" 2>/dev/null | head -1             | grep -oE "\[[0-9]+\](rocep|roceP)[a-zA-Z0-9]*" | wc -l)
    if [ "${bound:-0}" -ge "$want_rails" ]; then
        pass "NCCL bound $bound rail(s), as STRIPE=$STRIPE expects"
    else
        fail "NCCL bound only ${bound:-0} rail(s); STRIPE=$STRIPE expects $want_rails"
        info "$(grep -a 'NET/IB : Using' "$LOG" 2>/dev/null | head -1 | sed 's/.*NCCL INFO //')"
        info "engine is serving but transport is degraded (~6% single-stream, ~7% prefill)"
    fi
fi

step "Stack up"
exit 0
