`timescale 1ns / 1ps

`include "GLOBAL_CONST.svh"

import isa_pkg::*;

// ===| NPU Controller Top |======================================================
// Wraps the AXI-Lite frontend and the opcode decoder.
// Outputs one valid pulse per instruction type along with the raw 60-bit body.
// ===============================================================================

module npu_controller_top #() (
    input logic clk,
    input logic rst_n,
    input logic i_clear,

    // ===| AXI4-Lite Slave : PS <-> NPU control plane |=========================
    axil_if.slave S_AXIL_CTRL,

    // ===| Decoded Instruction Valids |=========================================
    output logic OUT_GEMV_op_x64_valid,
    output logic OUT_GEMM_op_x64_valid,
    output logic OUT_memcpy_op_x64_valid,
    output logic OUT_memset_op_x64_valid,
    output logic OUT_cvo_op_x64_valid,

    // ===| Raw Instruction Body (60-bit, opcode stripped) |=====================
    output instruction_op_x64_t OUT_op_x64
);

  // ===| Internal Wires |========================================================
  logic [`ISA_WIDTH-1:0] raw_instruction;
  logic                  raw_instruction_pop_valid;
  logic                  fetch_PC_ready;

  // ===| Frontend : AXI-Lite CMD/STAT |==========================================
  ctrl_npu_frontend #() u_npu_frontend (
      .clk     (clk),
      .rst_n   (rst_n),
      .IN_clear(i_clear),

      .S_AXIL_CTRL(S_AXIL_CTRL),

      .OUT_RAW_instruction(raw_instruction),
      .OUT_kick           (raw_instruction_pop_valid),

      .IN_enc_stat ('0),
      .IN_enc_valid(1'b0),

      .IN_fetch_ready(fetch_PC_ready)
  );

  // ===| Decoder : Opcode -> Engine FIFOs |======================================
  ctrl_npu_decoder u_decoder (
      .clk                    (clk),
      .rst_n                  (rst_n),
      .IN_raw_instruction     (raw_instruction),
      .raw_instruction_pop_valid(raw_instruction_pop_valid),

      .OUT_fetch_PC_ready     (fetch_PC_ready),

      .OUT_GEMV_op_x64_valid  (OUT_GEMV_op_x64_valid),
      .OUT_GEMM_op_x64_valid  (OUT_GEMM_op_x64_valid),
      .OUT_memcpy_op_x64_valid(OUT_memcpy_op_x64_valid),
      .OUT_memset_op_x64_valid(OUT_memset_op_x64_valid),
      .OUT_cvo_op_x64_valid   (OUT_cvo_op_x64_valid),

      .OUT_op_x64(OUT_op_x64)
  );

endmodule
