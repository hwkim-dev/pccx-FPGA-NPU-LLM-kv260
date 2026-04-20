`timescale 1ns / 1ps

// ===============================================================================
// Module: GEMM_dsp_unit
// Phase : pccx v002 (W4A8, 1 DSP = 2 MAC)
//
// Role
// ----
//   A single PE of the 32 × 32 systolic array (with cascade break @ row 16).
//   Holds two stationary INT4 weights (w_upper / w_lower) and multiplies them
//   in parallel against an INT8 activation streaming vertically through the
//   column via B-port cascade, emitting two MACs per DSP per cycle.
//
// Data-flow summary
// -----------------
//   Weights (2 × INT4, horizontal shift-register along each row):
//     in_H_upper -> latch -> packer.w_upper -> A-port  -> out_H_upper
//     in_H_lower -> latch -> packer.w_lower -> A-port  -> out_H_lower
//
//   Activation (INT8, vertical cascade via BCIN/BCOUT):
//     top row : in_V       -> sign-ext(18) -> B  (B_INPUT = DIRECT)
//     others  : BCIN_in    -> cascade      -> B  (B_INPUT = CASCADE)
//     always  : BCOUT_out  -> next row BCIN
//
//   Partial sum (48-bit accumulator, P-port cascade):
//     top / normal rows : V_result_in via PCIN
//     break row (16)    : V_result_in via C-port (fabric), P_fabric_out
//                         feeds the merger back into the lower half.
//
// Notes
// -----
//   * A_INPUT is always DIRECT — weights are stationary per PE, not cascaded.
//   * BREG = 2 is kept so that activation can be shadow-loaded one cycle
//     ahead of the MAC fire, giving a steady one-cycle `o_valid` pipeline
//     latency. AREG = 1 is enough for weights (they rarely change).
//   * The accumulator drain window is bounded by the packer's guard band
//     (UPPER_SHIFT = 21, 1024 MACs per channel). The GEMM instruction
//     dispatcher is responsible for issuing a flush before that limit.
// ===============================================================================

`include "GLOBAL_CONST.svh"
`include "GEMM_Array.svh"

module GEMM_dsp_unit #(
  parameter IS_TOP_ROW    = 0,
  parameter BREAK_CASCADE = 0,  // 1 at row 16 of the 32-row physical array
  parameter INT4_BITS     = `INT4_WIDTH,           // 4
  parameter INT8_BITS     = 8,
  parameter A_PORT_W      = `DEVICE_DSP_A_WIDTH,   // 30
  parameter B_PORT_W      = `DEVICE_DSP_B_WIDTH,   // 18
  parameter P_PORT_W      = `DSP48E2_POUT_SIZE,    // 48
  parameter UPPER_SHIFT   = 21                     // matches GEMM_dsp_packer default
) (
  input  logic clk,
  input  logic rst_n,
  input  logic i_clear,

  input  logic i_valid,         // activation data valid
  input  logic i_weight_valid,  // latch a new pair of weights from in_H_*
  output logic o_valid,

  // ===| Horizontal weight shift-registers (two INT4 lanes per row) |============
  input  logic [INT4_BITS-1:0] in_H_upper,
  output logic [INT4_BITS-1:0] out_H_upper,
  input  logic [INT4_BITS-1:0] in_H_lower,
  output logic [INT4_BITS-1:0] out_H_lower,

  // ===| Vertical INT8 activation |==============================================
  //   Top row uses in_V (fabric). Non-top rows ignore in_V and take BCIN_in.
  input  logic [INT8_BITS-1:0] in_V,
  input  logic [B_PORT_W-1:0]  BCIN_in,
  output logic [B_PORT_W-1:0]  BCOUT_out,

  // ===| VLIW 3-bit instruction (vertical pipeline) |============================
  input  logic [2:0] instruction_in_V,
  output logic [2:0] instruction_out_V,
  input  logic       inst_valid_in_V,
  output logic       inst_valid_out_V,

  // ===| Partial-sum cascade (PCIN/PCOUT + optional fabric break) |==============
  input  logic [P_PORT_W-1:0] V_result_in,
  output logic [P_PORT_W-1:0] V_result_out,
  output logic [P_PORT_W-1:0] P_fabric_out
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
  //   Same shape as v001: a single-hot pulse walks down a 4-bit shift register
  //   whenever the dispatcher asserts instruction[2] (flush opcode). Stages 1-2
  //   clear the P-register, stage 3 latches the newly loaded weights.
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

  logic is_flushing;
  assign is_flushing = flush_sequence[1] | flush_sequence[2];

  // ===| OPMODE / ALUMODE |=======================================================
  //   Z-mux selects whether the P accumulator is continued from PCIN (cascade)
  //   or from the fabric C-port (after a cascade break at row 16).
  logic [8:0] dynamic_opmode;
  logic [3:0] dynamic_alumode;
  localparam logic [2:0] Z_MUX = BREAK_CASCADE ? 3'b011 : 3'b001;

  always_comb begin
    if (is_flushing) begin
      dynamic_opmode  = 9'b00_000_00_00;   // P = 0
      dynamic_alumode = 4'b0000;
    end else if (current_inst[0] == 1'b1) begin
      dynamic_opmode  = {2'b00, Z_MUX, 2'b01, 2'b01};  // P = P_prev + A*B
      dynamic_alumode = 4'b0000;
    end else begin
      dynamic_opmode  = {2'b00, Z_MUX, 2'b00, 2'b00};  // P = P_prev (pass)
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

  // ===| Bit packing (2 × INT4 weights -> A port, INT8 act -> B port) |==========
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

  // ===| Output-valid pipeline |==================================================
  logic valid_delay;
  always_ff @(posedge clk) begin
    if (!rst_n) valid_delay <= 1'b0;
    else        valid_delay <= i_valid;
  end
  assign o_valid = valid_delay;

  // ===| DSP48E2 primitive |======================================================
  //   A-port  : packer output (DIRECT, no cascade).
  //   B-port  : direct (top row) or BCIN cascade (non-top rows).
  //   P-port  : PCIN cascade or C-fabric when BREAK_CASCADE = 1.
  logic [P_PORT_W-1:0] p_internal;
  logic [P_PORT_W-1:0] dsp_c_input;
  logic [P_PORT_W-1:0] dsp_pcin_input;

  assign dsp_c_input    = BREAK_CASCADE ? V_result_in : '0;
  assign dsp_pcin_input = BREAK_CASCADE ? '0          : V_result_in;

  // Break row re-injects the activation from fabric (same pattern the v001
  // A-port used for BF16 mantissa). Otherwise BCIN cascade drives B.
  DSP48E2 #(
    .A_INPUT    ("DIRECT"),
    .B_INPUT    ((IS_TOP_ROW || BREAK_CASCADE) ? "DIRECT" : "CASCADE"),
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
    .CEC      (1'b1),

    .A     (a_packed),
    .ACIN  ('0),
    .ACOUT (),

    .B     (b_extended),
    .BCIN  (BCIN_in),
    .BCOUT (BCOUT_out),

    .C     (dsp_c_input),

    .PCIN  (dsp_pcin_input),
    .PCOUT (V_result_out),

    .OPMODE (dynamic_opmode),
    .ALUMODE(dynamic_alumode),
    .P      (p_internal)
  );

  assign P_fabric_out = p_internal;

endmodule
