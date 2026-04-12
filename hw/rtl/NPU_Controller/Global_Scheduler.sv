`timescale 1ns / 1ps
`include "GEMM_Array.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

// ===| Global Scheduler |========================================================
// Receives the decoded instruction valid pulses and the raw 60-bit body.
// Casts the body to the appropriate instruction struct and produces engine uops.
//
// Note: STORE_uop is computed internally but a dedicated output port is not yet
// wired — add OUT_STORE_uop when the result-writeback path is implemented.
// ===============================================================================

module Global_Scheduler #() (
    input logic clk_core,
    input logic rst_n_core,

    // ===| From Decoder |========================================================
    input logic IN_GEMV_op_x64_valid,
    input logic IN_GEMM_op_x64_valid,
    input logic IN_memcpy_op_x64_valid,
    input logic IN_memset_op_x64_valid,
    input logic IN_cvo_op_x64_valid,

    input instruction_op_x64_t instruction,

    // ===| Engine uops |=========================================================
    output gemm_control_uop_t   OUT_GEMM_uop,
    output GEMV_control_uop_t   OUT_GEMV_uop,
    output memory_control_uop_t OUT_LOAD_uop,
    output memory_set_uop_t     OUT_mem_set_uop,
    output cvo_control_uop_t    OUT_CVO_uop
);

  // ===| Instruction type-casts (combinational) |================================
  // The 60-bit body is interpreted as different structs depending on the opcode.
  GEMV_op_x64_t   GEMV_op_x64;
  GEMM_op_x64_t   GEMM_op_x64;
  memcpy_op_x64_t memcpy_op_x64;
  memset_op_x64_t memset_op_x64;
  cvo_op_x64_t    cvo_op_x64;

  always_comb begin
    GEMV_op_x64   = GEMV_op_x64_t'(instruction.instruction);
    GEMM_op_x64   = GEMM_op_x64_t'(instruction.instruction);
    memcpy_op_x64 = memcpy_op_x64_t'(instruction.instruction);
    memset_op_x64 = memset_op_x64_t'(instruction.instruction);
    cvo_op_x64    = cvo_op_x64_t'(instruction.instruction);
  end

  // ===| MEMSET uop |============================================================
  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      OUT_mem_set_uop <= '0;
    end else begin
      if (IN_memset_op_x64_valid) begin
        OUT_mem_set_uop <= '{
            dest_cache : dest_cache_e'(memset_op_x64.dest_cache),
            dest_addr  : memset_op_x64.dest_addr,
            a_value    : memset_op_x64.a_value,
            b_value    : memset_op_x64.b_value,
            c_value    : memset_op_x64.c_value
        };
      end
    end
  end

  // ===| MEMCPY uop |============================================================
  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      OUT_LOAD_uop <= '0;
    end else begin
      if (IN_memcpy_op_x64_valid) begin
        OUT_LOAD_uop <= '{
            data_dest      : data_route_e'({memcpy_op_x64.from_device, memcpy_op_x64.to_device,
                                            6'b0}),
            dest_addr      : memcpy_op_x64.dest_addr,
            src_addr       : memcpy_op_x64.src_addr,
            shape_ptr_addr : memcpy_op_x64.shape_ptr_addr,
            async          : memcpy_op_x64.async
        };
      end
    end
  end

  // ===| GEMM uop |==============================================================
  memory_control_uop_t GEMM_STORE_uop;  // TODO: wire to result-writeback path

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      OUT_LOAD_uop  <= '0;
      OUT_GEMM_uop  <= '0;
      GEMM_STORE_uop <= '0;
    end else begin
      if (IN_GEMM_op_x64_valid) begin
        OUT_LOAD_uop <= '{
            data_dest      : from_L2_to_L1_GEMM,
            dest_addr      : '0,
            src_addr       : GEMM_op_x64.src_addr,
            shape_ptr_addr : GEMM_op_x64.shape_ptr_addr,
            async          : SYNC_OP
        };

        OUT_GEMM_uop <= '{
            flags          : GEMM_op_x64.flags,
            size_ptr_addr  : GEMM_op_x64.size_ptr_addr,
            parallel_lane  : GEMM_op_x64.parallel_lane
        };

        GEMM_STORE_uop <= '{
            data_dest      : from_GEMM_res_to_L2,
            dest_addr      : GEMM_op_x64.dest_reg,
            src_addr       : '0,
            shape_ptr_addr : GEMM_op_x64.shape_ptr_addr,
            async          : SYNC_OP
        };
      end
    end
  end

  // ===| GEMV uop |==============================================================
  memory_control_uop_t GEMV_STORE_uop;  // TODO: wire to result-writeback path

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      OUT_LOAD_uop   <= '0;
      OUT_GEMV_uop   <= '0;
      GEMV_STORE_uop <= '0;
    end else begin
      if (IN_GEMV_op_x64_valid) begin
        OUT_LOAD_uop <= '{
            data_dest      : from_L2_to_L1_GEMV,
            dest_addr      : '0,
            src_addr       : GEMV_op_x64.src_addr,
            shape_ptr_addr : GEMV_op_x64.shape_ptr_addr,
            async          : SYNC_OP
        };

        OUT_GEMV_uop <= '{
            flags          : GEMV_op_x64.flags,
            size_ptr_addr  : GEMV_op_x64.size_ptr_addr,
            parallel_lane  : GEMV_op_x64.parallel_lane
        };

        GEMV_STORE_uop <= '{
            data_dest      : from_GEMV_res_to_L2,
            dest_addr      : GEMV_op_x64.dest_reg,
            src_addr       : '0,
            shape_ptr_addr : GEMV_op_x64.shape_ptr_addr,
            async          : SYNC_OP
        };
      end
    end
  end

  // ===| CVO uop |===============================================================
  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      OUT_CVO_uop <= '0;
    end else begin
      if (IN_cvo_op_x64_valid) begin
        OUT_CVO_uop <= '{
            cvo_func : cvo_func_e'(cvo_op_x64.cvo_func),
            src_addr : cvo_op_x64.src_addr,
            dst_addr : cvo_op_x64.dst_addr,
            length   : cvo_op_x64.length,
            flags    : cvo_flags_t'(cvo_op_x64.flags),
            async    : cvo_op_x64.async
        };
      end
    end
  end

endmodule
