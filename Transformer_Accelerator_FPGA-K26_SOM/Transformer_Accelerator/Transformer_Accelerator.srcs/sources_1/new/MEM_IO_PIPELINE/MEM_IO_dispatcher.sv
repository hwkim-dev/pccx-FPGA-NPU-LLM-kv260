`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

module MEM_IO_dispatcher #(
)(
    input logic clk_core,    // 400MHz
    input logic rst_n_core,
    input logic clk_axi,     // 250MHz
    input logic rst_axi_n,

    // ===| External AXI-Stream (From PS & DDR4) |=================
    axis_if.slave  S_AXI_HP0_WEIGHT,
    axis_if.slave  S_AXI_HP1_WEIGHT,
    axis_if.slave  S_AXI_HP2_WEIGHT,
    axis_if.slave  S_AXI_HP3_WEIGHT,
    axis_if.slave  S_AXIS_ACP_FMAP,
    axis_if.master M_AXIS_ACP_RESULT, // TO ps.

    input memory_uop_t memcpy_uop_x64

    /*
    // ===| Weight Pipeline Control (To/From Dispatcher) |=============
    input  logic [`ADDR_WIDTH_L2-1:0] IN_read_addr_hp [0:3],
    input  logic                      IN_read_en_hp   [0:3],
    output logic [             127:0] OUT_read_data_hp[0:3],

    // ===| L2 cache Pipeline Control (To/From Dispatcher)(KV,FMAP) |==
    // ACP (External) Memory Map Control
    input logic [16:0] IN_acp_base_addr,  // Dispatcher tells where to store incoming FMAP
    input logic        IN_acp_rx_start,   // Trigger to accept ACP data

    // NPU (Internal) Compute Access (Port B)
    input  logic         IN_npu_we,
    input  logic [ 16:0] IN_npu_addr,
    input  logic [127:0] IN_npu_wdata,
    output logic [127:0] OUT_npu_rdata
    */

);

    MEM_IO_top #(
    ) u_MEM_IO_dispatcher (
      .clk_core(clk_core),
      .rst_n_core(rst_n_core),

      .clk_axi(clk_axi),
      .rst_axi_n(rst_axi_n),

      axis_if.slave(S_AXI_HP0_WEIGHT),
      axis_if.slave(S_AXI_HP1_WEIGHT),
      axis_if.slave(S_AXI_HP2_WEIGHT),
      axis_if.slave(S_AXI_HP3_WEIGHT),

      // ACP      = featureMAP in, out (Full-Duplex), read & write at same time
      axis_if.slave(S_AXIS_ACP_FMAP),  // Feature Map Input 0 (128-bit, HPC0)
      axis_if.master(M_AXIS_ACP_RESULT),  // Final Result Output (128-bit)

      .IN_memcpy_uop_x64(memcpy_uop_x64)

      // ===| Weight Pipeline Control (To/From Dispatcher) |=============
      .IN_read_addr_hp(),
      .IN_read_en_hp(),
      .OUT_read_data_hp(),

      // ===| L2 cache Pipeline Control (To/From Dispatcher)(KV,FMAP) |==
      .IN_acp_base_addr(),
      .IN_acp_rx_start(),

      // NPU (Internal) Compute Access (Port B)
      .IN_npu_we(),
      .IN_npu_addr(),
      .IN_npu_wdata(),
      .OUT_npu_rdata()

    )

endmodule
