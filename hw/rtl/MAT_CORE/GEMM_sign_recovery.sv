`timescale 1ns / 1ps

// ===============================================================================
// Module: GEMM_sign_recovery
// Phase : pccx v002, Phase A §2.2 (W4A8 dual-MAC bit-packing)
//
// Purpose
// -------
//   Post-DSP48E2 unpacker for the W4A8 "1 DSP = 2 MAC" datapath.
//
//   Takes the 48-bit P-register output of a DSP48E2 that was driven by
//   GEMM_dsp_packer and extracts the two independent signed accumulator
//   values (lower channel and upper channel).
//
//   The correction needed is the classic "carry borrow" case: if the lower
//   channel's running sum goes negative, the two's-complement representation
//   of that negative value spills a `1` carry into the guard band, which
//   propagates up into the upper channel's LSB and flips its value by one.
//   Adding 1 to the upper channel whenever the lower channel's MSB is set
//   restores the mathematical sum.
//
// Bit layout (mirrors GEMM_dsp_packer)
// -------------------------------------
//   P_reg[UPPER_SHIFT-1 : 0]                 : lower channel accumulator
//   P_reg[UPPER_SHIFT+UPPER_W-1 : UPPER_SHIFT]: upper channel accumulator
//
//   Default widths (UPPER_SHIFT=21, LOWER_W=21, UPPER_W=21) give:
//     lower : P_reg[20:0]    (21-bit signed)
//     upper : P_reg[41:21]   (21-bit signed)
//
//   With 1024 INT4*INT8 accumulations per channel the magnitude bound is
//   2^20, which fits in 21 bits signed.
//
// Notes
// -----
//   * This module is pure combinational.
//   * The accumulator must be drained before UPPER_W+1 bits of magnitude
//     accumulate in either channel. The GEMM controller schedules a drain
//     every 1024 cycles to honor that bound.
// ===============================================================================

`include "GLOBAL_CONST.svh"

module GEMM_sign_recovery #(
  parameter int P_PORT_W    = `DEVICE_DSP_P_WIDTH,  // 48
  parameter int UPPER_SHIFT = 21,                   // matches GEMM_dsp_packer
  parameter int LOWER_W     = UPPER_SHIFT,          // 21 (bits 20:0)
  parameter int UPPER_W     = 21                    // 21 (bits 41:21)
) (
  input  logic signed [P_PORT_W-1:0]  in_p_accum,

  output logic signed [LOWER_W-1:0]   out_lower_sum,
  output logic signed [UPPER_W-1:0]   out_upper_sum
);

  // ===| Lower channel extraction |===============================================
  // Direct slice; bit [LOWER_W-1] is the natural sign bit.
  assign out_lower_sum = in_p_accum[LOWER_W-1:0];

  // ===| Upper channel extraction with borrow correction |=======================
  logic signed [UPPER_W-1:0] upper_raw;
  assign upper_raw = in_p_accum[UPPER_SHIFT + UPPER_W - 1 : UPPER_SHIFT];

  // If the lower accumulator is negative, a carry propagated into bit
  // UPPER_SHIFT and decremented upper by 1. Add 1 back whenever that
  // happened.
  logic lower_is_negative;
  assign lower_is_negative = in_p_accum[LOWER_W-1];

  assign out_upper_sum = lower_is_negative ? (upper_raw + {{(UPPER_W-1){1'b0}}, 1'b1})
                                           : upper_raw;

endmodule
