`timescale 1ns / 1ps
`ifndef BF16_MATH_SV
`define BF16_MATH_SV

package bf16_math_pkg;

  /*─────────────────────────────────────────────
  BF16 struct
  [15]=sign  [14:7]=exp(8b)  [6:0]=mantissa(7b)
  hidden bit is implicit (not stored)
  ─────────────────────────────────────────────*/
  typedef struct packed {
    logic       sign;
    logic [7:0] exp;
    logic [6:0] mantissa;
  } bf16_t;

  /*─────────────────────────────────────────────
  Aligned output
  24-bit 2's complement
  ─────────────────────────────────────────────*/
  typedef struct packed {
    logic [7:0]  emax;
    logic [23:0] val;
  } bf16_aligned_t;

  /*─────────────────────────────────────────────
  cast raw 16-bit → bf16_t
  ─────────────────────────────────────────────*/
  function automatic bf16_t to_bf16(input logic [15:0] raw);
    return bf16_t'{sign: raw[15], exp: raw[14:7], mantissa: raw[6:0]};
  endfunction

  /*─────────────────────────────────────────────
  align one BF16 value to a given emax
  returns 24-bit 2's complement
  ─────────────────────────────────────────────*/
  function automatic logic [23:0] align_to_emax(input bf16_t val, input logic [7:0] emax);
    logic [ 7:0] diff;
    logic [22:0] mag;
    logic [23:0] result;

    diff   = emax - val.exp;
    mag    = ({1'b1, val.mantissa, 15'd0}) >> diff;
    result = val.sign ? (~{1'b0, mag} + 24'd1) : {1'b0, mag};
    return result;
  endfunction

endpackage

`endif
