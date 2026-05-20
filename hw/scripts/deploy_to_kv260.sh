#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 pccxai
#
# deploy_to_kv260.sh - copy a v002 bitstream + dtbo + bit.bin set to a KV260
# and program the PL.  Idempotent: re-running uninstalls the previous
# overlay before staging the new one.
#
# Required environment:
#   PCCX_KV260_HOST   KV260 hostname / IP
#   PCCX_KV260_USER   SSH user (key-based auth assumed)
#   KVFPGA_PASSWORD   sudo password for the board user (env var only - never
#                     printed, never logged, never embedded in commands)
#
# Optional environment:
#   PCCX_REPO         worktree root (default: auto-detected from script path)
#   PCCX_BIT          .bit file to deploy (default: latest v002 build under
#                     hw/build/v002-vivado-*/)
#   PCCX_OVERLAY      overlay name (default: pccx_npu_bd)
#   PCCX_SSH_OPTS     extra ssh options
#
# Usage:
#   deploy_to_kv260.sh [--dry-run] [--no-bitbin-regen] [--skip-precheck]
#
#   --dry-run            : print commands without executing them
#   --no-bitbin-regen    : skip `make -C sw/dtbo bitbin` (use existing .bit.bin)
#   --skip-precheck      : skip the pre_deploy_check.py validation
#                          (NOT recommended)

set -euo pipefail
IFS=$'\n\t'

# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
DRY_RUN=0
DO_BITBIN_REGEN=1
DO_PRECHECK=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)          DRY_RUN=1; shift ;;
        --no-bitbin-regen)  DO_BITBIN_REGEN=0; shift ;;
        --skip-precheck)    DO_PRECHECK=0; shift ;;
        -h|--help)
            sed -n '3,30p' "$0"
            exit 0
            ;;
        *)  echo "ERROR: unknown argument: $1" >&2; exit 2 ;;
    esac
done

# ----------------------------------------------------------------------------
# Resolve paths
# ----------------------------------------------------------------------------
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PCCX_REPO="${PCCX_REPO:-$( cd "$SCRIPT_DIR/../.." && pwd )}"

if [[ -z "${PCCX_BIT:-}" ]]; then
    # Pick the newest v002 build dir
    PCCX_BIT="$( ls -1td "$PCCX_REPO"/hw/build/v002-vivado-*/pccx_v002_system_wrapper.bit 2>/dev/null | head -n1 || true )"
fi
if [[ -z "$PCCX_BIT" || ! -f "$PCCX_BIT" ]]; then
    echo "ERROR: PCCX_BIT not found (looked under $PCCX_REPO/hw/build/v002-vivado-*/)." >&2
    exit 1
fi

PCCX_OVERLAY="${PCCX_OVERLAY:-pccx_npu_bd}"
DTBO_DIR="$PCCX_REPO/sw/dtbo/build/$PCCX_OVERLAY"
BITBIN="$DTBO_DIR/$PCCX_OVERLAY.bit.bin"
DTBO="$DTBO_DIR/$PCCX_OVERLAY.dtbo"
SHELLJSON="$DTBO_DIR/shell.json"

# ----------------------------------------------------------------------------
# Env validation (never log $KVFPGA_PASSWORD)
# ----------------------------------------------------------------------------
need_env() {
    local n="$1"
    if [[ -z "${!n:-}" ]]; then
        echo "ERROR: required env var not set: $n" >&2
        return 1
    fi
}
need_env PCCX_KV260_HOST
need_env PCCX_KV260_USER
need_env KVFPGA_PASSWORD  # value is consumed only as stdin to sudo

SSH_OPTS_DEFAULT=(
    -o ConnectTimeout=10
    -o ServerAliveInterval=15
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
)
read -r -a EXTRA_SSH_OPTS <<<"${PCCX_SSH_OPTS:-}"

echo "-- deploy_to_kv260.sh --"
echo "PCCX_REPO         : $PCCX_REPO"
echo "PCCX_BIT          : $PCCX_BIT"
echo "PCCX_OVERLAY      : $PCCX_OVERLAY"
echo "DTBO_DIR          : $DTBO_DIR"
echo "PCCX_KV260_HOST   : $PCCX_KV260_HOST"
echo "PCCX_KV260_USER   : $PCCX_KV260_USER"
echo "KVFPGA_PASSWORD   : (set; value redacted)"
echo "dry-run           : $DRY_RUN"
echo "bitbin regen      : $DO_BITBIN_REGEN"
echo "pre-deploy check  : $DO_PRECHECK"
echo

run() {
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY: $*"
    else
        echo "RUN: $*"
        "$@"
    fi
}

ssh_run() {
    # Send a remote sudo command via stdin, passing password from env (never
    # via argv).  Uses `sudo -S` so the password is read from stdin only.
    #
    # Returns the underlying ssh exit code; aborts the script when the
    # remote command fails so partial deploys never silently "succeed".
    local cmd="$1"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        # Redact the password placeholder in dry-run print.
        echo "DRY: ssh $PCCX_KV260_USER@$PCCX_KV260_HOST -- sudo -S sh -c '<cmd: $cmd>'"
        return 0
    fi
    echo "REMOTE: $cmd"
    local out rc
    out=$(printf '%s\n' "$KVFPGA_PASSWORD" | \
          ssh "${SSH_OPTS_DEFAULT[@]}" "${EXTRA_SSH_OPTS[@]}" \
              "$PCCX_KV260_USER@$PCCX_KV260_HOST" \
              "sudo -S sh -c $(printf %q "$cmd")" 2>&1)
    rc=$?
    # `grep` may legitimately exit 1 when every line is filtered; do not let
    # that mask a real ssh / sudo / remote-command failure (`set -o
    # pipefail` would also catch the wrong thing).  We check $rc directly.
    if [[ $rc -ne 0 ]]; then
        # Print full output (with password lines filtered) so the user can
        # see what went wrong.
        printf '%s\n' "$out" | grep -v -E 'password for|^Password:' >&2 || true
        echo "ERROR: remote command failed (rc=$rc): $cmd" >&2
        return "$rc"
    fi
    printf '%s\n' "$out" | grep -v -E 'password for|^Password:' || true
    return 0
}

scp_to() {
    local src="$1"
    local dst="$2"
    if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "DRY: scp $src $PCCX_KV260_USER@$PCCX_KV260_HOST:$dst"
        return 0
    fi
    scp "${SSH_OPTS_DEFAULT[@]}" "${EXTRA_SSH_OPTS[@]}" "$src" \
        "$PCCX_KV260_USER@$PCCX_KV260_HOST:$dst"
}

# ----------------------------------------------------------------------------
# 1. Regenerate .bit.bin from the chosen .bit (provenance step)
# ----------------------------------------------------------------------------
if [[ "$DO_BITBIN_REGEN" -eq 1 ]]; then
    echo "[1/5] regenerate bit.bin via bootgen"
    run make -C "$PCCX_REPO/sw/dtbo" bitbin \
        BIT_SRC="$PCCX_BIT" \
        BUILD="$DTBO_DIR"
else
    echo "[1/5] skip bit.bin regen (--no-bitbin-regen)"
fi

# ----------------------------------------------------------------------------
# 2. Run the local pre-deploy check
# ----------------------------------------------------------------------------
if [[ "$DO_PRECHECK" -eq 1 ]]; then
    echo "[2/5] run pre_deploy_check.py"
    run python3 "$SCRIPT_DIR/pre_deploy_check.py" \
        --bit "$PCCX_BIT" \
        --bitbin "$BITBIN" \
        --dtbo "$DTBO" \
        --shell-json "$SHELLJSON" \
        --check-env
else
    echo "[2/5] skip pre-deploy check (--skip-precheck)"
fi

# ----------------------------------------------------------------------------
# 3. Unload any previously-loaded overlay (idempotent)
# ----------------------------------------------------------------------------
echo "[3/5] remote: unload existing overlay (best-effort)"
ssh_run "xmutil unloadapp || true"

# ----------------------------------------------------------------------------
# 4. Stage firmware into /lib/firmware/xilinx/<overlay>/
# ----------------------------------------------------------------------------
echo "[4/5] stage overlay files into board firmware tree"
ssh_run "mkdir -p /lib/firmware/xilinx/$PCCX_OVERLAY"

# We scp to a writable user dir first, then sudo mv (the scp itself does
# not have sudo capability).
TMP_REMOTE="/tmp/pccx_deploy_${PCCX_OVERLAY}_$$"
# Create the staging dir as the SSH user (not sudo) so the subsequent
# unprivileged scp can write into it; the sudo cp below moves the files
# into the firmware tree where root ownership is required.
if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "DRY: ssh $PCCX_KV260_USER@$PCCX_KV260_HOST -- mkdir -p $TMP_REMOTE"
else
    echo "REMOTE (user): mkdir -p $TMP_REMOTE"
    ssh "${SSH_OPTS_DEFAULT[@]}" "${EXTRA_SSH_OPTS[@]}" \
        "$PCCX_KV260_USER@$PCCX_KV260_HOST" \
        "mkdir -p $TMP_REMOTE"
fi
scp_to "$BITBIN"    "$TMP_REMOTE/"
scp_to "$DTBO"      "$TMP_REMOTE/"
scp_to "$SHELLJSON" "$TMP_REMOTE/"
ssh_run "cp -f $TMP_REMOTE/$PCCX_OVERLAY.bit.bin /lib/firmware/xilinx/$PCCX_OVERLAY/$PCCX_OVERLAY.bit.bin"
ssh_run "cp -f $TMP_REMOTE/$PCCX_OVERLAY.dtbo    /lib/firmware/xilinx/$PCCX_OVERLAY/$PCCX_OVERLAY.dtbo"
ssh_run "cp -f $TMP_REMOTE/shell.json            /lib/firmware/xilinx/$PCCX_OVERLAY/shell.json"
ssh_run "rm -rf $TMP_REMOTE"

# ----------------------------------------------------------------------------
# 5. Load the overlay (programs PL) and verify
# ----------------------------------------------------------------------------
echo "[5/5] xmutil loadapp $PCCX_OVERLAY"
ssh_run "xmutil loadapp $PCCX_OVERLAY"

# Allow 1s for /dev/uio* nodes to appear
ssh_run "sleep 1; ls -la /dev/uio* 2>&1 | tee /tmp/pccx_uio_after_load.txt; ls -la /sys/class/fpga_manager/fpga0/state 2>/dev/null || true; cat /sys/class/fpga_manager/fpga0/state 2>/dev/null || true"

echo
echo "deploy_to_kv260.sh: complete"
echo "next: run hw/scripts/post_deploy_smoke.py to verify on-board reachability"
