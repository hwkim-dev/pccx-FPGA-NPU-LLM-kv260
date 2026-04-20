#!/usr/bin/env bash
# Run tb_GEMM_dsp_packer_sign_recovery under xsim and emit a .pccx trace
# for pccx-lab to load.
#
# Usage:  hw/sim/run_tb_packer.sh [output_dir]
#   output_dir defaults to ./hw/sim/work

set -euo pipefail

HW_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${1:-$HW_DIR/sim/work}"
PCCX_LAB_DIR="${PCCX_LAB_DIR:-$HW_DIR/../../pccx-lab}"
PCCX_CLI_BIN="${PCCX_CLI_BIN:-$PCCX_LAB_DIR/target/release/from_xsim_log}"

mkdir -p "$OUT_DIR"
cd "$OUT_DIR"

echo "==> Compiling RTL with xvlog"
xvlog -sv \
    -i "$HW_DIR/rtl/Constants/compilePriority_Order/A_const_svh" \
    "$HW_DIR/rtl/MAT_CORE/GEMM_dsp_packer.sv" \
    "$HW_DIR/rtl/MAT_CORE/GEMM_sign_recovery.sv" \
    "$HW_DIR/tb/tb_GEMM_dsp_packer_sign_recovery.sv" \
    >/dev/null

echo "==> Elaborating"
xelab -debug typical tb_GEMM_dsp_packer_sign_recovery \
    -s tb_packer_snap >/dev/null

echo "==> Running simulation"
xsim tb_packer_snap -R | tee xsim.log

echo "==> Converting xsim log to .pccx"
if [[ ! -x "$PCCX_CLI_BIN" ]]; then
    echo "ERROR: from_xsim_log binary not found at $PCCX_CLI_BIN"
    echo "Build with: cd $PCCX_LAB_DIR && cargo build --release --bin from_xsim_log"
    exit 2
fi

"$PCCX_CLI_BIN" \
    --log "$OUT_DIR/xsim.log" \
    --output "$OUT_DIR/tb_packer.pccx" \
    --testbench tb_GEMM_dsp_packer_sign_recovery \
    --core-id 0

echo ""
echo "==> Result: $OUT_DIR/tb_packer.pccx"
ls -la "$OUT_DIR/tb_packer.pccx"
