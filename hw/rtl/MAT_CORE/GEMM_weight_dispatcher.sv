`timescale 1ns / 1ps

// ===============================================================================
// Module: GEMM_weight_dispatcher
// Phase : pccx v002 (W4A8, 1 DSP = 2 MAC)
//
// Role
// ----
//   Registers the 32-wide INT4 weight stream that the upstream FIFO already
//   unpacks from HP0 / HP1, and fans it out to the systolic array's
//   H_in_upper / H_in_lower shift-register chains.
//
//   Upstream pipeline stages are responsible for:
//     * deserializing 128-bit HP AXI words into 32 × INT4 arrays
//     * interleaving the lane assignment: HP0 → upper channel,
//       HP1 → lower channel (so both weights of a given PE arrive in
//       the same cycle and can be packed together by GEMM_dsp_packer)
//
//   This module only does the final pipeline flop to preserve 400 MHz
//   timing and emit a single `weight_valid` that gates `i_weight_valid`
//   on every PE in the array.
//
// v001 -> v002 delta
// ------------------
//   v001 had a single unpacked stream (32 × INT4). v002 needs two streams
//   (upper + lower) because each DSP MAC now processes two weights per
//   cycle. Both streams use the same weight_cnt width.
// ===============================================================================

`include "GLOBAL_CONST.svh"
`include "GEMM_Array.svh"

module GEMM_weight_dispatcher #(
  parameter int weight_size = `INT4_WIDTH,                         // 4
  parameter int weight_cnt  = `HP_PORT_SINGLE_WIDTH / `INT4_WIDTH  // 32 = 128/4
) (
  input  logic clk,
  input  logic rst_n,

  // ===| Upper-channel input (e.g. HP0 lane, already unpacked) |================
  input  logic [weight_size-1:0] fifo_upper     [0:weight_cnt-1],
  input  logic                   fifo_upper_valid,
  output logic                   fifo_upper_ready,

  // ===| Lower-channel input (e.g. HP1 lane, already unpacked) |================
  input  logic [weight_size-1:0] fifo_lower     [0:weight_cnt-1],
  input  logic                   fifo_lower_valid,
  output logic                   fifo_lower_ready,

  // ===| Registered outputs to the systolic array |=============================
  output logic [weight_size-1:0] weight_upper [0:weight_cnt-1],
  output logic [weight_size-1:0] weight_lower [0:weight_cnt-1],
  output logic                   weight_valid
);

  // ===| Flow control — always accept while the pipeline is not stalled |=======
  assign fifo_upper_ready = 1'b1;
  assign fifo_lower_ready = 1'b1;

  // ===| Pipeline register stage |==============================================
  //   Fires only when both channels deliver valid data in the same cycle,
  //   which is how the upstream scheduler is supposed to pair them for W4A8
  //   dual-MAC. A misalignment starves the array of valid pairs, so the
  //   valid is an AND, not an OR.
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      weight_valid <= 1'b0;
      for (int i = 0; i < weight_cnt; i++) begin
        weight_upper[i] <= '0;
        weight_lower[i] <= '0;
      end
    end else begin
      weight_valid <= fifo_upper_valid & fifo_lower_valid;
      for (int i = 0; i < weight_cnt; i++) begin
        weight_upper[i] <= fifo_upper[i];
        weight_lower[i] <= fifo_lower[i];
      end
    end
  end

endmodule
