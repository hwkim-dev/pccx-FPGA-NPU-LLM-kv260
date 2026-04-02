`timescale 1ns / 1ps

`include "stlc_Array.svh"
`include "GLOBAL_CONST.svh"
/**
 * Module: stlc_systolic_top
 * Target: Kria KV260 @ 400MHz
 *
 * Architecture V2:
 * - Weight Dispatcher (Unpacker)
 * - Staggered Delay Lines for FMap & Instructions
 * - 32x32 Systolic Array Core
 * - e_max Pipe for Synchronization with Result Output
 */
module stlc_systolic_top #(
    parameter weight_lane_cnt = `HP_PORT_CNT,
    parameter weight_width_per_lane = `HP_PORT_SINGLE_WIDTH,
    parameter weight_size     = `INT4,
    parameter weight_cnt      = `HP_WEIGHT_CNT(`HP_PORT_CNT, `INT4)
)(
    input logic clk,
    input logic rst_n,
    input logic i_clear,

    // Control & Inst
    input logic global_weight_valid,
    input logic [2:0] global_inst,
    input logic global_inst_valid,

    // Feature Map Broadcast (from SRAM Cache)
    input logic [`FIXED_MANT_WIDTH-1:0] fmap_broadcast      [0:`ARRAY_SIZE_H-1],
    input logic                         fmap_broadcast_valid,

    // e_max (from Cache for Normalization alignment)
    input logic [`BF16_EXP_WIDTH-1:0] cached_emax_out[0:`ARRAY_SIZE_H-1],

    // Weight Input from FIFO (Direct)
    input  logic [`HP_PORT_MAX_WIDTH-1:0] weight_fifo_data,
    input  logic                          weight_fifo_valid,
    output logic                          weight_fifo_ready,

    // Output Results (Raw)
    output logic [`DSP48E2_POUT_SIZE-1:0] raw_res_sum      [0:`ARRAY_SIZE_H-1],
    output logic                          raw_res_sum_valid[0:`ARRAY_SIZE_H-1],

    // Delayed e_max for Normalizers
    output logic [`BF16_EXP_WIDTH-1:0] delayed_emax_32[0:`ARRAY_SIZE_H-1]
);

  // ===| Weight Dispatcher (The Unpacker) |=======
  logic [`INT4-1:0] unpacked_weights [0:weight_lane_cnt-1][0:weight_cnt-1];
  logic             weights_ready_for_array;

  stlc_weight_dispatcher #(
    .weight_lane_cnt(weight_lane_cnt),
    .weight_width_per_lane(weight_width_per_lane),
    .weight_size(weight_size),
    .weight_cnt(weight_cnt)
  )( u_weight_unpacker (
      .clk(clk),
      .rst_n(rst_n),
      .fifo_data(weight_fifo_data),
      .fifo_valid(weight_fifo_valid),
      .fifo_ready(weight_fifo_ready),
      .weight_out(unpacked_weights),
      .weight_valid(weights_ready_for_array)
  );

  // ===| Staggered Delay Line for FMap & Instructions |=======
  logic [`FIXED_MANT_WIDTH-1:0] staggered_fmap      [0:`ARRAY_SIZE_H-1];
  logic                         staggered_fmap_valid[0:`ARRAY_SIZE_H-1];
  logic [                  2:0] staggered_inst      [0:`ARRAY_SIZE_H-1];
  logic                         staggered_inst_valid[0:`ARRAY_SIZE_H-1];

  stlc_fmap_staggered_delay #(
      .DATA_WIDTH(`FIXED_MANT_WIDTH),
      .ARRAY_SIZE(`ARRAY_SIZE_H)
  ) u_delay_line (
      .clk(clk),
      .rst_n(rst_n),
      .fmap_in(fmap_broadcast),
      .fmap_valid(fmap_broadcast_valid),
      .global_inst(global_inst),
      .global_inst_valid(global_inst_valid),
      .row_data(staggered_fmap),
      .row_valid(staggered_fmap_valid),
      .row_inst(staggered_inst),
      .row_inst_valid(staggered_inst_valid)
  );

  // ===| Systolic Array Core (The Engine) |=======
  logic [`DSP48E2_POUT_SIZE-1:0] raw_res_seq[0:`ARRAY_SIZE_H-1];

  stlc_NxN_array #(
      .ARRAY_HORIZONTAL(`ARRAY_SIZE_H),
      .ARRAY_VERTICAL  (`ARRAY_SIZE_V)
  ) u_compute_core (
      .clk(clk),
      .rst_n(rst_n),
      .i_clear(i_clear),
      .i_weight_valid(global_weight_valid),

      // Horizontal: Weights
      .H_in(unpacked_weights),

      // Vertical: Feature Map Broadcast & Instructions (Staggered)
      .V_in(staggered_fmap),
      .in_valid(staggered_fmap_valid),
      .inst_in(staggered_inst),
      .inst_valid_in(staggered_inst_valid),

      .V_out(raw_res_seq),
      .V_ACC_out(raw_res_sum),
      .V_ACC_valid(raw_res_sum_valid)
  );

  // ===| e_max Delay Pipe for Normalization alignment |=======
  localparam TOTAL_LATENCY = `SYSTOLIC_TOTAL_LATENCY;
  logic [`BF16_EXP_WIDTH-1:0] emax_pipe[0:`ARRAY_SIZE_H-1][0:TOTAL_LATENCY-1];

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      for (int c = 0; c < `ARRAY_SIZE_H; c++) begin
        for (int d = 0; d < TOTAL_LATENCY; d++) begin
          emax_pipe[c][d] <= 0;
        end
      end
    end else begin
      for (int c = 0; c < `ARRAY_SIZE_H; c++) begin
        emax_pipe[c][0] <= cached_emax_out[c];
        for (int d = 1; d < TOTAL_LATENCY; d++) begin
          emax_pipe[c][d] <= emax_pipe[c][d-1];
        end
      end
    end
  end

  always_comb begin
    for (int c = 0; c < `ARRAY_SIZE_H; c++) begin
      delayed_emax_32[c] = emax_pipe[c][TOTAL_LATENCY-1];
    end
  end

endmodule
