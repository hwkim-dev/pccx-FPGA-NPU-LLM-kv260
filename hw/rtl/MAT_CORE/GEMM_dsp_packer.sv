`timescale 1ns / 1ps

// ===============================================================================
// Module: GEMM_dsp_packer
// Phase : pccx v002, Phase A §2.2 (W4A8 dual-MAC bit-packing)
//
// Purpose
// -------
//   Pre-DSP48E2 bit-packer for the W4A8 "1 DSP = 2 MAC" datapath.
//
//   Takes a pair of stationary INT4 weights (upper and lower channel) and a
//   single streaming INT8 activation, and produces the A-port and B-port
//   values that drive the DSP48E2 primitive so that a single multiply emits
//   two independent signed MAC results in disjoint bit fields of the 48-bit
//   P accumulator.
//
//   Pairs with GEMM_sign_recovery.sv, which undoes the packing on the
//   accumulator output.
//
// Bit-width budget
// ----------------
//   Per-MAC magnitude : |w| * |a| <= 8 * 128 = 2^10
//   N = 1024 accumulations -> 2^10 * 2^10 = 2^20 per channel
//   21 signed bits per channel. Place the upper channel at bit 21 so the
//   two 21-bit fields are exactly adjacent in the 48-bit P register:
//       P[20:0]  = lower channel signed accumulator
//       P[41:21] = upper channel signed accumulator
//   Bits [47:42] are sign-extension headroom.
//
//   Multiplier: 27x18 signed. The packed A value lives in <= 25 signed bits
//   (w_upper * 2^21 + w_lower), well inside the 27-bit A budget.
//
// Encoding
// --------
//   We compute a_packed mathematically as (w_upper * 2^UPPER_SHIFT + w_lower)
//   using signed arithmetic, then sign-extend to A_PORT_W bits. Naive bit
//   concatenation is wrong in the signed case: if either weight is negative,
//   its two's-complement sign bits must ripple to the correct positions,
//   which only a signed add guarantees.
//
// Notes
// -----
//   * Pure combinational. DSP48E2's AREG / BREG register the values inside
//     the primitive; no clock is needed here.
//   * Requires UPPER_SHIFT >= INT4_BITS + INT8_BITS + log2(N) for an N-MAC
//     drain window. Default UPPER_SHIFT = 21 covers N = 1024.
// ===============================================================================

`include "GLOBAL_CONST.svh"

module GEMM_dsp_packer #(
  parameter int INT4_BITS   = `INT4_WIDTH,          // 4
  parameter int INT8_BITS   = 8,                    // `Int8Width` in dtype_pkg
  parameter int A_PORT_W    = `DEVICE_DSP_A_WIDTH,  // 30
  parameter int B_PORT_W    = `DEVICE_DSP_B_WIDTH,  // 18
  parameter int UPPER_SHIFT = 21                    // see budget above
) (
  // ===| Stationary INT4 weights |=================================================
  input  logic signed [INT4_BITS-1:0] in_w_lower,
  input  logic signed [INT4_BITS-1:0] in_w_upper,

  // ===| Streaming INT8 activation |===============================================
  input  logic signed [INT8_BITS-1:0] in_act,

  // ===| DSP48E2 port-aligned outputs |===========================================
  output logic signed [A_PORT_W-1:0]  out_a_packed,
  output logic signed [B_PORT_W-1:0]  out_b_extended
);

  // ===| A-port packing via signed arithmetic |===================================
  logic signed [A_PORT_W-1:0] w_upper_shifted;
  logic signed [A_PORT_W-1:0] w_lower_extended;

  // Widen each signed weight to A_PORT_W bits with proper sign extension, then
  // shift the upper one into its slot.
  assign w_upper_shifted  = A_PORT_W'(signed'(in_w_upper)) <<< UPPER_SHIFT;
  assign w_lower_extended = A_PORT_W'(signed'(in_w_lower));

  assign out_a_packed = w_upper_shifted + w_lower_extended;

  // ===| B-port sign-extension |===================================================
  assign out_b_extended = B_PORT_W'(signed'(in_act));

endmodule
