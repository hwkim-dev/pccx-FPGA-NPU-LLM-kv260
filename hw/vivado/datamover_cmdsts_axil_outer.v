// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 pccxai
// =====================================================================
// datamover_cmdsts_axil_outer.v — Verilog-2001 passthrough wrapper.
//
// Same reason as npu_core_outer.v: Vivado IP Integrator's
// `create_bd_cell -type module -reference` rejects SystemVerilog as the
// top file type (filemgmt 56-195). The real body
// `datamover_cmdsts_axil.sv` uses `logic` types and other SV features,
// so the BD references this thin .v outer module and the SV body is
// elaborated during synthesis.
//
// Port list mirrors datamover_cmdsts_axil exactly. Keep them in sync.
// =====================================================================

`timescale 1ns / 1ps

module datamover_cmdsts_axil_outer #(
    parameter integer AXIL_ADDR_W = 12,
    parameter integer AXIL_DATA_W = 32,
    parameter integer CMD_WIDTH   = 72,
    parameter integer STS_WIDTH   = 8,
    parameter integer FIFO_DEPTH  = 8
) (
    input  wire                       s_axil_aclk,
    input  wire                       s_axil_aresetn,

    input  wire [AXIL_ADDR_W-1:0]     s_axil_awaddr,
    input  wire                       s_axil_awvalid,
    output wire                       s_axil_awready,
    input  wire [AXIL_DATA_W-1:0]     s_axil_wdata,
    input  wire [(AXIL_DATA_W/8)-1:0] s_axil_wstrb,
    input  wire                       s_axil_wvalid,
    output wire                       s_axil_wready,
    output wire [1:0]                 s_axil_bresp,
    output wire                       s_axil_bvalid,
    input  wire                       s_axil_bready,
    input  wire [AXIL_ADDR_W-1:0]     s_axil_araddr,
    input  wire                       s_axil_arvalid,
    output wire                       s_axil_arready,
    output wire [AXIL_DATA_W-1:0]     s_axil_rdata,
    output wire [1:0]                 s_axil_rresp,
    output wire                       s_axil_rvalid,
    input  wire                       s_axil_rready,

    output wire [CMD_WIDTH-1:0]       m_axis_cmd_tdata,
    output wire                       m_axis_cmd_tvalid,
    input  wire                       m_axis_cmd_tready,

    input  wire [STS_WIDTH-1:0]       s_axis_sts_tdata,
    input  wire                       s_axis_sts_tvalid,
    output wire                       s_axis_sts_tready,
    input  wire                       s_axis_sts_tlast,
    input  wire [(STS_WIDTH+7)/8-1:0] s_axis_sts_tkeep
);

    datamover_cmdsts_axil #(
        .AXIL_ADDR_W (AXIL_ADDR_W),
        .AXIL_DATA_W (AXIL_DATA_W),
        .CMD_WIDTH   (CMD_WIDTH),
        .STS_WIDTH   (STS_WIDTH),
        .FIFO_DEPTH  (FIFO_DEPTH)
    ) u_body (
        .s_axil_aclk       (s_axil_aclk),
        .s_axil_aresetn    (s_axil_aresetn),
        .s_axil_awaddr     (s_axil_awaddr),
        .s_axil_awvalid    (s_axil_awvalid),
        .s_axil_awready    (s_axil_awready),
        .s_axil_wdata      (s_axil_wdata),
        .s_axil_wstrb      (s_axil_wstrb),
        .s_axil_wvalid     (s_axil_wvalid),
        .s_axil_wready     (s_axil_wready),
        .s_axil_bresp      (s_axil_bresp),
        .s_axil_bvalid     (s_axil_bvalid),
        .s_axil_bready     (s_axil_bready),
        .s_axil_araddr     (s_axil_araddr),
        .s_axil_arvalid    (s_axil_arvalid),
        .s_axil_arready    (s_axil_arready),
        .s_axil_rdata      (s_axil_rdata),
        .s_axil_rresp      (s_axil_rresp),
        .s_axil_rvalid     (s_axil_rvalid),
        .s_axil_rready     (s_axil_rready),
        .m_axis_cmd_tdata  (m_axis_cmd_tdata),
        .m_axis_cmd_tvalid (m_axis_cmd_tvalid),
        .m_axis_cmd_tready (m_axis_cmd_tready),
        .s_axis_sts_tdata  (s_axis_sts_tdata),
        .s_axis_sts_tvalid (s_axis_sts_tvalid),
        .s_axis_sts_tready (s_axis_sts_tready),
        .s_axis_sts_tlast  (s_axis_sts_tlast),
        .s_axis_sts_tkeep  (s_axis_sts_tkeep)
    );

endmodule
