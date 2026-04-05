`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "GLOBAL_CONST.svh"

import isa_pkg::*;

module MEM_IO_top (
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
);

  // 1. Internal AXIS Interfaces (400MHz Domain)
  axis_if #(.DATA_WIDTH(128)) core_hp_bus[0:3] ();
  axis_if #(.DATA_WIDTH(128)) core_acp_rx_bus ();
  axis_if #(.DATA_WIDTH(128)) core_acp_tx_bus ();

  // 2. MEM_IO_BUFFER Instantiation
  MEM_IO_BUFFER u_buffer (
      .clk_core(clk_core),
      .rst_n_core(rst_n_core),
      .clk_axi(clk_axi),
      .rst_axi_n(rst_axi_n),

      .S_AXI_HP0_WEIGHT (S_AXI_HP0_WEIGHT),
      .M_CORE_HP0_WEIGHT(core_hp_bus[0]),
      .S_AXI_HP1_WEIGHT (S_AXI_HP1_WEIGHT),
      .M_CORE_HP1_WEIGHT(core_hp_bus[1]),
      .S_AXI_HP2_WEIGHT (S_AXI_HP2_WEIGHT),
      .M_CORE_HP2_WEIGHT(core_hp_bus[2]),
      .S_AXI_HP3_WEIGHT (S_AXI_HP3_WEIGHT),
      .M_CORE_HP3_WEIGHT(core_hp_bus[3]),

      .S_AXIS_ACP_FMAP(S_AXIS_ACP_FMAP),
      .M_CORE_ACP_RX(core_acp_rx_bus),
      .M_AXIS_ACP_RESULT(M_AXIS_ACP_RESULT),
      .S_CORE_ACP_TX(core_acp_tx_bus)
  );


  // 4. ACP RX Auto-Increment Logic (Just like HP ports!)
  logic [16:0] acp_write_ptr;
  logic        acp_write_en;

  // RX Bus is ready when Dispatcher tells it to start
  assign core_acp_rx_bus.tready = IN_acp_rx_start;
  assign acp_write_en = core_acp_rx_bus.tvalid & core_acp_rx_bus.tready;

  always_ff @(posedge clk_core) begin
    if (!rst_n_core) begin
      acp_write_ptr <= '0;
    end else if (IN_acp_rx_start && !acp_write_en) begin
      // Initialize pointer to Base Address provided by Dispatcher
      acp_write_ptr <= IN_acp_base_addr;
    end else if (acp_write_en) begin
      // Auto-increment as data pours in
      acp_write_ptr <= acp_write_ptr + 1'b1;
    end
  end

  // 5. FMAP & KV Cache (Massive 1.75MB URAM)
  MEM_IO_L2_cache #(
      .URAM_DEPTH(114688)
  ) u_fmap_kv_l2 (
      .clk_core  (clk_core),
      .rst_n_core(rst_n_core),

      // Port A [from queue Buffer] -> (ACP / DDR4 Side)
      .IN_acp_we    (acp_write_en),
      .IN_acp_addr  (acp_write_ptr),
      .IN_acp_wdata (core_acp_rx_bus.tdata),
      .OUT_acp_rdata(core_acp_tx_bus.tdata),  // (Output draining logic can be added here)

      // Port B (NPU Compute Side)
      .IN_npu_we    (IN_npu_we),
      .IN_npu_addr  (IN_npu_addr),
      .IN_npu_wdata (IN_npu_wdata),
      .OUT_npu_rdata(OUT_npu_rdata)
  );

endmodule
