#!/bin/bash
# detect-site.sh — probe this machine and emit a filled scripts/site.env.
#
# Read-only: it inspects, it never configures. Run on rank 0.
#
#   bash scripts/detect-site.sh            # print to stdout for review
#   bash scripts/detect-site.sh --write    # write scripts/site.env
#
# Anything it cannot determine is emitted as REPLACE_WITH_* with a note on
# stderr, so a partial detection is obvious rather than silently wrong.

set -uo pipefail
WRITE=0; [ "${1:-}" = "--write" ] && WRITE=1
note() { printf '  %s\n' "$1" >&2; }
warn() { printf '  !! %s\n' "$1" >&2; }

echo "detect-site: probing $(hostname -s)" >&2

# ---- containers: those with a bind mount landing on /ws ---------------------
mapfile -t WSC < <(docker ps --format '{{.Names}}' 2>/dev/null | while read -r n; do
    docker inspect "$n" --format '{{range .Mounts}}{{if eq .Destination "/ws"}}{{$.Name}} {{.Source}}{{end}}{{end}}' 2>/dev/null
done | grep -v '^$')
C0_D=$(printf '%s\n' "${WSC[@]}" | head -1 | awk '{print $1}' | sed 's|^/||')
WS_D=$(printf '%s\n' "${WSC[@]}" | head -1 | awk '{print $2}')
[ -n "$C0_D" ] && note "container: $C0_D (work dir $WS_D)" || warn "no container with a /ws mount found"

# ---- LAN address: the interface holding the default route ------------------
LAN_IF=$(ip -4 route show default 2>/dev/null | awk '{print $5; exit}')
LAN0=$(ip -4 -br addr show "$LAN_IF" 2>/dev/null | awk '{print $3}' | cut -d/ -f1)
[ -n "$LAN0" ] && note "LAN: $LAN0 (via $LAN_IF)" || warn "could not determine the LAN address"

# ---- ConnectX netdevs, in PCI order ----------------------------------------
mapfile -t NETDEVS < <(for d in /sys/class/net/*; do
    n=$(basename "$d")
    drv=$(basename "$(readlink -f "$d/device/driver" 2>/dev/null)" 2>/dev/null)
    [ "$drv" = "mlx5_core" ] || continue
    printf '%s %s\n' "$(readlink -f "$d/device" | sed 's|.*/||')" "$n"
done | sort | awk '{print $2}')
note "ConnectX netdevs: ${NETDEVS[*]:-none}"

# ---- RDMA devices, mapped to their netdev, one port per PCI card -----------
declare -A CARD_OF
for ibd in /sys/class/infiniband/*; do
    [ -e "$ibd" ] || continue
    dev=$(basename "$ibd")
    net=$(ls "$ibd/device/net" 2>/dev/null | head -1)
    pci=$(readlink -f "$ibd/device" | sed 's|.*/||')          # e.g. 0000:01:00.0
    card="${pci%.*}"                                           # strip the function
    [ -n "${CARD_OF[$card]:-}" ] || CARD_OF[$card]="$dev $net"
done
mapfile -t CARDS < <(for k in "${!CARD_OF[@]}"; do echo "$k ${CARD_OF[$k]}"; done | sort)
RDMA1=$(printf '%s\n' "${CARDS[@]}" | sed -n 1p | awk '{print $2}')
RDMA2=$(printf '%s\n' "${CARDS[@]}" | sed -n 2p | awk '{print $2}')
note "RDMA (one port per card): ${RDMA1:-none} ${RDMA2:-none}"

# ---- fabric address: point-to-point IP on the first ConnectX netdev --------
FAB0=""; FAB1=""
if [ "${#NETDEVS[@]}" -gt 0 ]; then
    FAB0=$(ip -4 -br addr show "${NETDEVS[0]}" 2>/dev/null | awk '{print $3}' | cut -d/ -f1)
    if [ -n "$FAB0" ]; then
        # peer of a /30 or /31: flip the last octet by +1 or -1
        base="${FAB0%.*}"; last="${FAB0##*.}"
        cand="$base.$((last+1))"; ping -c1 -W1 "$cand" >/dev/null 2>&1 && FAB1="$cand"
        if [ -z "$FAB1" ]; then cand="$base.$((last-1))"; ping -c1 -W1 "$cand" >/dev/null 2>&1 && FAB1="$cand"; fi
    fi
fi
[ -n "$FAB0" ] && note "fabric rank0: $FAB0" || warn "no address on the first ConnectX netdev — is the fabric configured?"
[ -n "$FAB1" ] && note "fabric rank1: $FAB1 (reachable)" || warn "no reachable peer — single node, or the pair is not cabled/addressed"

# ---- rail prefix + GID index -----------------------------------------------
RAILP=""; GIDX=""
# Look ONLY at the ConnectX netdevs: the LAN interface also carries a /24 and
# would otherwise be mistaken for a rail subnet.
railaddr=""
for nd in "${NETDEVS[@]:-}"; do
    [ -n "$nd" ] || continue
    a=$(ip -4 -br addr show "$nd" 2>/dev/null | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+/24' | head -1 | cut -d/ -f1)
    [ -n "$a" ] && { railaddr="$a"; break; }
done
if [ -n "$railaddr" ]; then
    RAILP=$(echo "$railaddr" | cut -d. -f1,2)
    if [ -n "${RDMA1:-}" ]; then
        # RoCEv2 GIDs encode IPv4 as ...ffff:AABB:CCDD — match our rail address
        hex=$(printf '%02x%02x:%02x%02x' $(echo "$railaddr" | tr . ' '))
        # The same address appears twice: once as RoCE v1, once as v2. We need
        # v2 — picking the first match silently yields v1 and a fabric that
        # misbehaves rather than fails cleanly.
        for g in /sys/class/infiniband/$RDMA1/ports/1/gids/*; do
            grep -qi "ffff:$hex" "$g" 2>/dev/null || continue
            idx=$(basename "$g")
            typ=$(cat "/sys/class/infiniband/$RDMA1/ports/1/gid_attrs/types/$idx" 2>/dev/null)
            case "$typ" in *"RoCE v2"*) GIDX="$idx"; break;; esac
        done
    fi
fi
[ -n "$RAILP" ] && note "rail prefix: $RAILP (from $railaddr)" || warn "no rail /24 found — striping needs per-cable subnets (see README.md Phase 1 — Fabric bring-up)"
[ -n "$GIDX" ] && note "GID index: $GIDX (RoCE v2, matches $railaddr on $RDMA1)" || warn "could not resolve the GID index for the rail address"

TP_D=2; [ -z "$FAB1" ] && TP_D=1
note "topology: TP=$TP_D"

# ---- rank 1's LAN address, via the fabric peer -----------------------------
LAN1=""
if [ -n "$FAB1" ]; then
    LAN1=$(ssh -o BatchMode=yes -o ConnectTimeout=5 "$FAB1"         "ip -4 route show default | awk '{print \$5; exit}' | xargs -r ip -4 -br addr show | awk '{print \$3}' | cut -d/ -f1" 2>/dev/null | head -1)
    [ -n "$LAN1" ] && note "LAN rank1: $LAN1 (via fabric peer)"                    || warn "could not reach rank 1 over the fabric to read its LAN address"
fi

OUT=$(cat <<CFG
# Generated by detect-site.sh on $(hostname -s) at $(date -u +%Y-%m-%dT%H:%M:%SZ).
# Review every value before use; anything REPLACE_WITH_* was not detectable.

TP=$TP_D

C0=${C0_D:-REPLACE_WITH_RANK0_CONTAINER}
C1=${C1_D:-ggbuild}

WS=${WS_D:-REPLACE_WITH_WORK_DIR}

LAN_RANK0=${LAN0:-REPLACE_WITH_RANK0_LAN_IP}
LAN_RANK1=${LAN1:-REPLACE_WITH_RANK1_LAN_IP}

FABRIC_RANK0=${FAB0:-REPLACE_WITH_RANK0_FABRIC_IP}
FABRIC_RANK1=${FAB1:-REPLACE_WITH_RANK1_FABRIC_IP}

NETDEV1=${NETDEVS[0]:-REPLACE_WITH_NETDEV1}
NETDEV2=${NETDEVS[1]:-REPLACE_WITH_NETDEV2}
NETDEV3=${NETDEVS[2]:-REPLACE_WITH_NETDEV3}
NETDEV4=${NETDEVS[3]:-REPLACE_WITH_NETDEV4}

RDMA_CARD1=${RDMA1:-REPLACE_WITH_CARD1_RDMA_DEV}
RDMA_CARD2=${RDMA2:-REPLACE_WITH_CARD2_RDMA_DEV}

NCCL_GID_INDEX=${GIDX:-REPLACE_WITH_GID_INDEX}
RAIL_PREFIX=${RAILP:-10.42}
CFG
)

miss=$(printf '%s' "$OUT" | grep -v '^#' | grep -c REPLACE_WITH_)
if [ "$WRITE" = "1" ]; then
    d="$(cd "$(dirname "$0")" && pwd)"; printf '%s\n' "$OUT" > "$d/site.env"
    note "wrote $d/site.env"
    [ "$miss" -gt 0 ] && warn "$miss value(s) still need filling in by hand"
else
    printf '%s\n' "$OUT"
    [ "$miss" -gt 0 ] && warn "$miss value(s) could not be detected (shown as REPLACE_WITH_*)"
fi
[ "$miss" -gt 0 ] && exit 1 || exit 0
