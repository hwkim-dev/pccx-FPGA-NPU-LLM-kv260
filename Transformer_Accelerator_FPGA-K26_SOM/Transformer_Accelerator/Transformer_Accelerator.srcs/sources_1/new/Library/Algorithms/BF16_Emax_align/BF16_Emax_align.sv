`timescale 1ns / 1ps
`include "GLOBAL_CONST.svh"

import bf16_math_pkg::*;

/*─────────────────────────────────────────────
  BF16_EaxAlign
  Input  : 32 × BF16 (raw 16-bit)
  Output : global emax (8-bit)
           32 × aligned value (24-bit 2's complement)

  Pipeline : purely combinational (no clock)
  Caller registers the output as needed.
─────────────────────────────────────────────*/
module BF16_EaxAlign (
    input logic [15:0] i_bf16[0:31],  // 32 raw BF16 inputs

    output logic [ 7:0] o_emax,          // global max exponent
    output logic [23:0] o_aligned[0:31]  // aligned 2's complement values
);

  /*─────────────────────────────────────────────
  Step 1 : unpack 32 raw BF16 → bf16_t structs
  ─────────────────────────────────────────────*/
  bf16_t unpacked[0:31];

  genvar gi;
  generate
    for (gi = 0; gi < 32; gi++) begin : g_unpack
      assign unpacked[gi] = to_bf16(i_bf16[gi]);
    end
  endgenerate

  /*─────────────────────────────────────────────
  Step 2 : find global emax — binary tree
  Level 0 : 32 exponents
  Level 1 : 16 maxes  (pairs)
  Level 2 :  8 maxes
  Level 3 :  4 maxes
  Level 4 :  2 maxes
  Level 5 :  1 global emax
  ─────────────────────────────────────────────*/
  logic [7:0] exp_L1[0:15];
  logic [7:0] exp_L2[ 0:7];
  logic [7:0] exp_L3[ 0:3];
  logic [7:0] exp_L4[ 0:1];

  generate
    // L1 : 32 → 16
    for (gi = 0; gi < 16; gi++) begin : g_L1
      assign exp_L1[gi] = (unpacked[gi*2].exp > unpacked[gi*2+1].exp)
                          ? unpacked[gi*2].exp : unpacked[gi*2+1].exp;
    end
    // L2 : 16 → 8
    for (gi = 0; gi < 8; gi++) begin : g_L2
      assign exp_L2[gi] = (exp_L1[gi*2] > exp_L1[gi*2+1]) ? exp_L1[gi*2] : exp_L1[gi*2+1];
    end
    // L3 : 8 → 4
    for (gi = 0; gi < 4; gi++) begin : g_L3
      assign exp_L3[gi] = (exp_L2[gi*2] > exp_L2[gi*2+1]) ? exp_L2[gi*2] : exp_L2[gi*2+1];
    end
    // L4 : 4 → 2
    for (gi = 0; gi < 2; gi++) begin : g_L4
      assign exp_L4[gi] = (exp_L3[gi*2] > exp_L3[gi*2+1]) ? exp_L3[gi*2] : exp_L3[gi*2+1];
    end
  endgenerate

  // L5 : 2 → 1
  assign o_emax = (exp_L4[0] > exp_L4[1]) ? exp_L4[0] : exp_L4[1];

  /*─────────────────────────────────────────────
  Step 3 : align all 32 values to global emax
  ─────────────────────────────────────────────*/
  generate
    for (gi = 0; gi < 32; gi++) begin : g_align
      assign o_aligned[gi] = align_to_emax(unpacked[gi], o_emax);
    end
  endgenerate

endmodule
