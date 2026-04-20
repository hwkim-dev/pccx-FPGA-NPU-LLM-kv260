`timescale 1ns / 1ps

// ===============================================================================
// Testbench: tb_barrel_shifter_BF16
// Phase : pccx v002 — fmap preprocess path
//
// Purpose
// -------
//   Pure-combinational verification of `barrel_shifter_BF16`.
//
//   Generates 512 random (bf16_act, e_max) pairs, recomputes the 27-bit
//   fixed-point output in a matching SV function, and compares each
//   apply against the DUT after a sub-cycle settle delay.
//
//   Emits the canonical `PASS: <N> cycles, ...` line the pccx-lab xsim
//   bridge picks up.
// ===============================================================================

`include "GLOBAL_CONST.svh"

module tb_barrel_shifter_BF16;

  localparam int N_VECTORS = 512;

  // ===| DUT IO (no clock — the DUT is combinational) |==========================
  logic [15:0] bf16_act;
  logic [ 7:0] e_max;
  logic [26:0] delayLine_in;
  logic [ 7:0] exp_out;
  logic        sign_out;

  barrel_shifter_BF16 u_dut (
    .bf16_act    (bf16_act),
    .e_max       (e_max),
    .delayLine_in(delayLine_in),
    .exp_out     (exp_out),
    .sign_out    (sign_out)
  );

  // ===| Golden model |==========================================================
  function automatic logic [26:0] golden_shift(
      input logic [15:0] bf,
      input logic [ 7:0] emax
  );
    logic       sign;
    logic [7:0] exp;
    logic [7:0] mant;
    logic [26:0] base_vec;
    logic [7:0]  delta_e;
    logic [26:0] shifted;

    sign = bf[15];
    exp  = bf[14:7];
    mant = (exp == 8'd0) ? {1'b0, bf[6:0]} : {1'b1, bf[6:0]};
    base_vec = {mant, 19'b0};
    delta_e  = emax - exp;

    if (delta_e >= 8'd27) begin
      shifted = 27'd0;
    end else begin
      shifted = base_vec >> delta_e[4:0];
    end

    return sign ? (~shifted + 27'd1) : shifted;
  endfunction

  // ===| Stimulus |==============================================================
  int errors = 0;
  logic [26:0] exp_delay;

  initial begin
    bf16_act = '0;
    e_max    = '0;

    // Fixed checkpoints first so regression artefacts show obvious intent.
    bf16_act = 16'h0000; e_max = 8'd127; #1;
    if (delayLine_in !== 27'd0) begin
      errors++;
      $display("[%0t] checkpoint zero-input failed: got=%h",
               $time, delayLine_in);
    end

    // Random sweep.
    for (int i = 0; i < N_VECTORS; i++) begin
      bf16_act = $random;
      e_max    = $random;
      #1;

      exp_delay = golden_shift(bf16_act, e_max);
      if (delayLine_in !== exp_delay) begin
        errors++;
        if (errors <= 10) begin
          $display("[%0t] mismatch  bf=%h emax=%h  got=%h exp=%h",
                   $time, bf16_act, e_max, delayLine_in, exp_delay);
        end
      end
      if (exp_out !== e_max) begin
        errors++;
        if (errors <= 10) begin
          $display("[%0t] exp_out mismatch: got=%h exp=%h",
                   $time, exp_out, e_max);
        end
      end
      if (sign_out !== bf16_act[15]) begin
        errors++;
        if (errors <= 10) begin
          $display("[%0t] sign_out mismatch: got=%b exp=%b",
                   $time, sign_out, bf16_act[15]);
        end
      end
    end

    if (errors == 0) begin
      $display("PASS: %0d cycles, both channels match golden.", N_VECTORS);
    end else begin
      $display("FAIL: %0d mismatches over %0d cycles.", errors, N_VECTORS);
    end
    $finish;
  end

  initial begin
    #100000 $display("TIMEOUT"); $finish;
  end

endmodule
