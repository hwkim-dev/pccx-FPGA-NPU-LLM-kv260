`timescale 1ns / 1ps

// ===============================================================================
// Module: GEMM_dsp_unit_last_ROW
// Phase : pccx v002 (W4A8, 1 DSP = 2 MAC)
//
// Role
// ----
//   Specialized PE for the bottom row of the 32 × 32 systolic array. Same
//   W4A8 dual-MAC datapath as GEMM_dsp_unit, but exposes the 48-bit P as
//   `gemm_unit_results` so the result packer / barrel-shifter can sample it
//   directly (there is no next row to cascade into).
//
//   See GEMM_dsp_unit.sv for the full data-flow description.
// ===============================================================================

`include "GLOBAL_CONST.svh"
`include "GEMM_Array.svh"

module GEMM_dsp_unit_last_ROW #(
  parameter IS_TOP_ROW  = 0,     // kept for symmetry with GEMM_dsp_unit
  parameter INT4_BITS   = `INT4_WIDTH,
  parameter INT8_BITS   = 8,
  parameter A_PORT_W    = `DEVICE_DSP_A_WIDTH,
  parameter B_PORT_W    = `DEVICE_DSP_B_WIDTH,
  parameter P_PORT_W    = `DSP48E2_POUT_SIZE,
  parameter UPPER_SHIFT = 21
) (
  input  logic clk,
  input  logic rst_n,
  input  logic i_clear,

  input  logic i_valid,
  input  logic i_weight_valid,
  input  logic inst_valid_in_V,
  output logic o_valid,

  // ===| Horizontal weight shift-registers |=====================================
  input  logic [INT4_BITS-1:0] in_H_upper,
  output logic [INT4_BITS-1:0] out_H_upper,
  input  logic [INT4_BITS-1:0] in_H_lower,
  output logic [INT4_BITS-1:0] out_H_lower,

  // ===| Vertical INT8 activation |==============================================
  input  logic [INT8_BITS-1:0] in_V,
  input  logic [B_PORT_W-1:0]  BCIN_in,
  output logic [B_PORT_W-1:0]  BCOUT_out,

  // ===| VLIW instruction |======================================================
  input  logic [2:0] instruction_in_V,
  output logic [2:0] instruction_out_V,
  output logic       inst_valid_out_V,

  // ===| Partial-sum cascade |===================================================
  input  logic [P_PORT_W-1:0] V_result_in,   // PCIN from the PE above
  output logic [P_PORT_W-1:0] V_result_out,  // PCOUT (unused at the bottom)

  // ===| Final result out to the barrel-shifter |================================
  output logic [P_PORT_W-1:0] gemm_unit_results
);

  // ===| Instruction latch |======================================================
  logic [2:0] current_inst;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      current_inst <= 3'b000;
    end else if (inst_valid_in_V) begin
      current_inst <= instruction_in_V;
    end
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      instruction_out_V <= 3'b000;
      inst_valid_out_V  <= 1'b0;
    end else begin
      instruction_out_V <= instruction_in_V;
      inst_valid_out_V  <= inst_valid_in_V;
    end
  end

  // ===| Flush sequencer |========================================================
  logic [3:0] flush_sequence;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      flush_sequence <= 4'd0;
    end else begin
      flush_sequence <= {flush_sequence[2:0], 1'b0};
      if (inst_valid_in_V && instruction_in_V[2] == 1'b1) begin
        flush_sequence[0] <= 1'b1;
      end
    end
  end

  // ===| OPMODE / ALUMODE (plain PCIN path — no fabric break at the last row) |==
  logic [8:0] dynamic_opmode;
  logic [3:0] dynamic_alumode;
  logic       is_flushing;
  assign      is_flushing = flush_sequence[1] | flush_sequence[2];

  always_comb begin
    if (is_flushing) begin
      dynamic_opmode  = 9'b00_001_00_00;   // P = PCIN (drain)
      dynamic_alumode = 4'b0000;
    end else if (current_inst[0] == 1'b1) begin
      dynamic_opmode  = 9'b00_010_01_01;   // P = P_prev + A*B (local accum)
      dynamic_alumode = 4'b0000;
    end else begin
      dynamic_opmode  = 9'b00_010_00_00;
      dynamic_alumode = 4'b0000;
    end
  end

  logic dsp_ce_p;
  assign dsp_ce_p = current_inst[0] | is_flushing;

  // ===| Weight latch + horizontal shift |========================================
  logic [INT4_BITS-1:0] w_upper_reg;
  logic [INT4_BITS-1:0] w_lower_reg;

  always_ff @(posedge clk) begin
    if (!rst_n || i_clear) begin
      w_upper_reg <= '0;
      w_lower_reg <= '0;
      out_H_upper <= '0;
      out_H_lower <= '0;
    end else if (i_weight_valid) begin
      w_upper_reg <= in_H_upper;
      w_lower_reg <= in_H_lower;
      out_H_upper <= in_H_upper;
      out_H_lower <= in_H_lower;
    end
  end

  // ===| Bit packing |============================================================
  logic signed [A_PORT_W-1:0] a_packed;
  logic signed [B_PORT_W-1:0] b_extended;

  GEMM_dsp_packer #(
    .INT4_BITS   (INT4_BITS),
    .INT8_BITS   (INT8_BITS),
    .A_PORT_W    (A_PORT_W),
    .B_PORT_W    (B_PORT_W),
    .UPPER_SHIFT (UPPER_SHIFT)
  ) u_packer (
    .in_w_lower     (w_lower_reg),
    .in_w_upper     (w_upper_reg),
    .in_act         (in_V),
    .out_a_packed   (a_packed),
    .out_b_extended (b_extended)
  );

  // ===| o_valid pipeline |=======================================================
  logic valid_delay;
  always_ff @(posedge clk) begin
    if (!rst_n) valid_delay <= 1'b0;
    else        valid_delay <= i_valid;
  end
  assign o_valid = valid_delay;

  // ===| DSP48E2 primitive |======================================================
  DSP48E2 #(
    .A_INPUT    ("DIRECT"),
    .B_INPUT    (IS_TOP_ROW ? "DIRECT" : "CASCADE"),
    .AREG       (1),
    .BREG       (2),
    .CREG       (0),
    .MREG       (1),
    .PREG       (1),
    .OPMODEREG  (1),
    .ALUMODEREG (1),
    .USE_MULT   ("MULTIPLY")
  ) DSP_HARD_BLOCK (
    .CLK          (clk),
    .RSTA         (i_clear),
    .RSTB         (i_clear),
    .RSTM         (i_clear),
    .RSTP         (i_clear),
    .RSTCTRL      (i_clear),
    .RSTALLCARRYIN(i_clear),
    .RSTALUMODE   (i_clear),
    .RSTC         (i_clear),

    .CEA1     (i_weight_valid),
    .CEA2     (i_weight_valid),
    .CEB1     (i_valid),
    .CEB2     (i_valid),
    .CEM      (i_valid),
    .CEP      (dsp_ce_p),
    .CECTRL   (1'b1),
    .CEALUMODE(1'b1),

    .A     (a_packed),
    .ACIN  ('0),
    .ACOUT (),

    .B     (b_extended),
    .BCIN  (BCIN_in),
    .BCOUT (BCOUT_out),

    .C     ('0),

    .PCIN  (V_result_in),
    .PCOUT (V_result_out),

    .OPMODE (dynamic_opmode),
    .ALUMODE(dynamic_alumode),

    .P      (gemm_unit_results)
  );

endmodule
