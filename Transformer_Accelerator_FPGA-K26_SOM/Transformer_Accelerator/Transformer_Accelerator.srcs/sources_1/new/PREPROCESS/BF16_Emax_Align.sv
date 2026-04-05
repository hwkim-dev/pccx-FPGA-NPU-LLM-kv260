`timescale 1ns / 1ps
`include "bf16_math.sv"

import bf16_math_pkg::*;

/*─────────────────────────────────────────────
  BF16_EaxAlign
  Input  : 32 × BF16 (raw 16-bit)
  Output : OUT_emax     — global max exponent (8-bit)
           OUT_aligned  — 32 × 11-bit signed mantissa
                        [10]    = sign
                        [9:2]   = {hidden_1, mantissa_7}
                        [1:0]   = guard bits
  Purely combinational — caller registers output.
─────────────────────────────────────────────*/
module BF16_Emax_Align (
    input logic [15:0] IN_bf16[0:31],

    output logic [7:0] OUT_emax,
    output logic signed [10:0] OUT_aligned[0:31]
);

  /*─────────────────────────────────────────────
  Step 1 : unpack raw 16-bit → bf16_t
  ─────────────────────────────────────────────*/
  bf16_t unpacked[0:31];

  genvar gi;
  generate
    for (gi = 0; gi < 32; gi++) begin : g_unpack
      assign unpacked[gi] = to_bf16(IN_bf16[gi]);
    end
  endgenerate

  /*─────────────────────────────────────────────
  Step 2 : find global emax — binary reduction tree
  L0 : 32 exponents
  L1 : 16  L2 : 8  L3 : 4  L4 : 2  L5 : 1
  ─────────────────────────────────────────────*/
  logic [7:0] exp_L1[0:15];
  logic [7:0] exp_L2[ 0:7];
  logic [7:0] exp_L3[ 0:3];
  logic [7:0] exp_L4[ 0:1];

  generate
    for (gi = 0; gi < 16; gi++) begin : g_L1
      assign exp_L1[gi] = (unpacked[gi*2].exp > unpacked[gi*2+1].exp)
                          ? unpacked[gi*2].exp : unpacked[gi*2+1].exp;
    end
    for (gi = 0; gi < 8; gi++) begin : g_L2
      assign exp_L2[gi] = (exp_L1[gi*2] > exp_L1[gi*2+1]) ? exp_L1[gi*2] : exp_L1[gi*2+1];
    end
    for (gi = 0; gi < 4; gi++) begin : g_L3
      assign exp_L3[gi] = (exp_L2[gi*2] > exp_L2[gi*2+1]) ? exp_L2[gi*2] : exp_L2[gi*2+1];
    end
    for (gi = 0; gi < 2; gi++) begin : g_L4
      assign exp_L4[gi] = (exp_L3[gi*2] > exp_L3[gi*2+1]) ? exp_L3[gi*2] : exp_L3[gi*2+1];
    end
  endgenerate

  assign OUT_emax = (exp_L4[0] > exp_L4[1]) ? exp_L4[0] : exp_L4[1];

  /*─────────────────────────────────────────────
  Step 3 : align 32 values to global emax
  unshifted magnitude : {hidden=1, mantissa[6:0], guard=2'b00} = 10-bit
  shift right by diff = (OUT_emax - val.exp)
  then apply sign → 11-bit 2's complement
  ─────────────────────────────────────────────*/
  generate
    for (gi = 0; gi < 32; gi++) begin : g_align
      logic [ 7:0] diff;
      logic [ 9:0] mag;  // unsigned magnitude after shift
      logic [10:0] signed_val;  // 11-bit signed result

      assign diff = OUT_emax - unpacked[gi].exp;

      // place {hidden, mantissa} at top, 2 guard bits at bottom, shift right
      assign mag = ({1'b1, unpacked[gi].mantissa, 2'b00}) >> diff;

      // 2's complement sign application
      assign signed_val = unpacked[gi].sign ? (~{1'b0, mag} + 11'd1) : {1'b0, mag};

      assign OUT_aligned[gi] = signed_val;
    end
  endgenerate

endmodule
