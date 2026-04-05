`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "npu_interfaces.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

module cu_npu_decoder (
    input logic clk,
    input logic rst_n,
    input logic [`ISA_WIDTH-1:0] IN_raw_instruction,  // Instruction_reg
    input logic raw_instruction_pop_valid,

    output logic OUT_fetch_PC_ready,


    //output instruction_t OUT_inst,
    output logic         OUT_memcpy_VALID;
    output memory_uop_x64_t OUT_memcpy_uop_x64;

    output logic         OUT_vdotm_VALID;
    output vdotm_uop_x64_t  OUT_vdotm_uop_x64;

    output logic         OUT_mdotm_VALID;
    output mdotm_uop_x64_t  OUT_mdotm_uop_x64;
);



  //logic [31-`X32_HEADSIZE:0] instruction_x32;
  VLIW_instruction_x64 instruction_VLIW_x64;

  //high bit -> low

  logic chain_depth;

  always_comb begin
    //assign arch_mod = `MOD_X64;
    if(chain_depth==1) begin
      assign o_inst = instruction_x64_low_t'(IN_raw_INST);
    end else begin
      assign o_inst = instruction_x64_high_t'(IN_raw_INST);
    end
  end


  //x64
  always_ff @(posedge clk) begin
    if(!rst_n) begin
      chain_depth <= 0;
    end else begin
      if (chain_depth == 1) begin
        chain_depth <= 0;
        instruction_ready <= 1;
        instruction_VLIW_x64[`X64_LOWBIT_RMOV_HD-1 : 0] <= o_inst.payload;
      end else begin
        chain_depth <= 1;
        instruction_VLIW_x64[`X64_HIGHBIT_RMOV_HD-1 : `X64_LOWBIT_RMOV_HD-1] <= o_inst.payload;
      end
    end
  end

  logic instruction_ready;

  always_ff @(posedge clk) begin
    if(!rst_n) begin
      instruction_ready <= 0;
    end else begin
      o_valid = 1'b1;
      if(instruction_ready) begin
        case (o_inst.opcode)
          OP_VDOTM: begin
            OUT_vdotm_cmd_x64  <= vdotm_uop_x64_t'(instruction_VLIW_x64);
          end
          OP_MDOTM: begin
            OUT_mdotm_cmd_x64 <= mdotm_uop_x64_t'(instruction_VLIW_x64);
          end
          OP_MEMCPY: begin
            OUT_memcpy_cmd_x64 <= memory_uop_x64_t'(instruction_VLIW_x64);
          end
          default: o_valid = 1'b0;  // unknown opcode
        endcase
      end
    end


  end


endmodule

