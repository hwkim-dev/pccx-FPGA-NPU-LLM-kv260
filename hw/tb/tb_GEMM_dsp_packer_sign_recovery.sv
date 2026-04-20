`timescale 1ns / 1ps

// ===============================================================================
// Testbench: tb_GEMM_dsp_packer_sign_recovery
// Phase : pccx v002, Phase A §2.2
//
// Purpose
// -------
//   Closed-loop verification of GEMM_dsp_packer + GEMM_sign_recovery.
//
//   The testbench wraps a synthesizable DSP48E2 behaviour model (just an
//   add-accumulate on A*B) around the two modules under test, drives it
//   with random INT4 * INT8 operand pairs for N = 1024 cycles (the
//   documented per-channel accumulation bound), and compares the recovered
//   upper / lower sums against software-maintained golden accumulators.
//
//   The test fails on the first mismatch. Target simulator: Xilinx xsim.
// ===============================================================================

`include "GLOBAL_CONST.svh"

module tb_GEMM_dsp_packer_sign_recovery;

  // ===| Parameters |=============================================================
  localparam int INT4_BITS   = `INT4_WIDTH;           // 4
  localparam int INT8_BITS   = 8;
  localparam int A_PORT_W    = `DEVICE_DSP_A_WIDTH;   // 30
  localparam int B_PORT_W    = `DEVICE_DSP_B_WIDTH;   // 18
  localparam int P_PORT_W    = `DEVICE_DSP_P_WIDTH;   // 48
  localparam int UPPER_SHIFT = 21;
  localparam int LOWER_W     = UPPER_SHIFT;           // 21
  localparam int UPPER_W     = 21;                    // 21
  localparam int N_ACCUM     = 1024;

  // ===| Clock |==================================================================
  logic clk;
  initial clk = 1'b0;
  always #2 clk = ~clk;  // 250 MHz test clock — arbitrary

  // ===| Stimulus signals |=======================================================
  logic signed [INT4_BITS-1:0] w_lower, w_upper;
  logic signed [INT8_BITS-1:0] act;

  // ===| DUT #1: packer |=========================================================
  logic signed [A_PORT_W-1:0] a_packed;
  logic signed [B_PORT_W-1:0] b_extended;

  GEMM_dsp_packer #(
    .INT4_BITS   (INT4_BITS),
    .INT8_BITS   (INT8_BITS),
    .A_PORT_W    (A_PORT_W),
    .B_PORT_W    (B_PORT_W),
    .UPPER_SHIFT (UPPER_SHIFT)
  ) u_packer (
    .in_w_lower     (w_lower),
    .in_w_upper     (w_upper),
    .in_act         (act),
    .out_a_packed   (a_packed),
    .out_b_extended (b_extended)
  );

  // ===| Control |================================================================
  logic p_reset;
  logic p_enable;
  int   errors;
  int   i;

  // ===| DSP48E2 behavioural stand-in |===========================================
  // Signed A * B, add into P. This is the minimum functional slice of the
  // DSP48E2 we rely on for W4A8 packing. A real sim links against the
  // Xilinx UNISIM DSP48E2 primitive; this model keeps the testbench
  // vendor-independent.
  logic signed [P_PORT_W-1:0] p_accum;

  always_ff @(posedge clk) begin
    if (p_reset) begin
      p_accum <= '0;
    end else if (p_enable) begin
      p_accum <= p_accum + ($signed(a_packed) * $signed(b_extended));
    end
  end

  // ===| DUT #2: sign recovery |==================================================
  logic signed [LOWER_W-1:0] rec_lower;
  logic signed [UPPER_W-1:0] rec_upper;

  GEMM_sign_recovery #(
    .P_PORT_W    (P_PORT_W),
    .UPPER_SHIFT (UPPER_SHIFT),
    .LOWER_W     (LOWER_W),
    .UPPER_W     (UPPER_W)
  ) u_recovery (
    .in_p_accum    (p_accum),
    .out_lower_sum (rec_lower),
    .out_upper_sum (rec_upper)
  );

  // ===| Golden accumulators (maintained in software) |===========================
  logic signed [31:0] golden_lower;
  logic signed [31:0] golden_upper;

  initial begin
    p_reset      = 1'b1;
    p_enable     = 1'b0;
    golden_lower = '0;
    golden_upper = '0;
    errors       = 0;
    w_lower      = '0;
    w_upper      = '0;
    act          = '0;

    @(posedge clk);
    p_reset  = 1'b0;
    p_enable = 1'b1;

    for (i = 0; i < N_ACCUM; i++) begin
      // Random signed INT4 and INT8 stimuli.
      w_lower = $random;
      w_upper = $random;
      act     = $random;

      // Golden update — plain integer arithmetic, no packing tricks.
      golden_lower = golden_lower + w_lower * act;
      golden_upper = golden_upper + w_upper * act;

      @(posedge clk);
      // NBA for p_accum resolves at the end of the posedge time step.
      // Reading combinational fan-out of p_accum needs one delta-cycle
      // after the edge — otherwise we see the previous cycle's value.
      #1;

      if (rec_lower !== golden_lower[LOWER_W-1:0]) begin
        $display("[%0t] cycle %0d LOWER mismatch: rec=%0d golden=%0d",
                 $time, i, rec_lower, golden_lower);
        errors++;
      end
      if (rec_upper !== golden_upper[UPPER_W-1:0]) begin
        $display("[%0t] cycle %0d UPPER mismatch: rec=%0d golden=%0d",
                 $time, i, rec_upper, golden_upper);
        errors++;
      end
    end

    p_enable = 1'b0;
    @(posedge clk);

    if (errors == 0) begin
      $display("PASS: %0d cycles, both channels match golden.", N_ACCUM);
    end else begin
      $display("FAIL: %0d mismatches over %0d cycles.", errors, N_ACCUM);
    end
    $finish;
  end

  // ===| Safety timeout |=========================================================
  initial begin
    #100000 $display("TIMEOUT"); $finish;
  end

endmodule
