// ===| DEPRECATED — use npu_arch.svh + kv260_device.svh instead |===============
// This file is kept as a compatibility shim so existing `include "GLOBAL_CONST.svh"
// statements continue to work during the migration period.
// Do NOT add new constants here. Add to npu_arch.svh or kv260_device.svh.
// ===============================================================================

`ifndef GLOBAL_CONST_SVH
`define GLOBAL_CONST_SVH

`include "NUMBERS.svh"
`include "kv260_device.svh"
`include "npu_arch.svh"

// ===| Legacy aliases (kept for backward compatibility) |=======================

// Boolean
`define TRUE  1'b1
`define FALSE 1'b0

// HP weight bus width aliases (used in port declarations of MAT_CORE)
`define HP_PORT_MAX_WIDTH    `HP_TOTAL_WIDTH
`define HP_PORT_SINGLE_WIDTH `HP_SINGLE_WIDTH
`define HP_PORT_CNT          `DEVICE_HP_PORT_CNT

// DSP48E2 port size aliases (used in GEMM_dsp_unit port declarations)
`define DSP48E2_POUT_SIZE    `DSP_P_OUT_WIDTH
`define DSP48E2_A_WIDTH      `DEVICE_DSP_A_WIDTH
`define DSP48E2_B_WIDTH      `DEVICE_DSP_B_WIDTH
`define PREG_SIZE            `DSP_P_OUT_WIDTH

// (Retired) GEMM_MAC_UNIT_IN_H / GEMM_MAC_UNIT_IN_V — these described the
// v001 1-MAC layout (INT4 on B-port, BF16 mantissa on A-port). v002 flips
// those roles (2 x INT4 packed on A-port via GEMM_dsp_packer, INT8 on
// B-port via BCIN/BCOUT cascade), so the old constants no longer fit.
// Use `INT4_WIDTH / `DEVICE_DSP_A_WIDTH / `DEVICE_DSP_B_WIDTH directly.

`endif // GLOBAL_CONST_SVH
