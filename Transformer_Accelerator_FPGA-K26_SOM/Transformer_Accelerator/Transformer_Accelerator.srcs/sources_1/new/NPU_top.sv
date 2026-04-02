`include "GLOBAL_CONST.svh"
`timescale 1ns / 1ps
`include "stlc_Array.svh"
`include "mem_IO.svh"
`include "npu_interfaces.svh"
`include "GLOBAL_CONST.svh"

/**
 * Module: NPU_top
 * Target: Kria KV260 @ 400MHz
 *
 * Architecture V2 (SystemVerilog Interface Version):
 * - HPC0/HPC1: Combined to form 256-bit Feature Map caching bus.
 * - HP0~HP3: Dedicated to high-throughput Weight streaming.
 * - HPM (MMIO): Centralized control & VLIW Instruction issuing.
 * - ACP: Coherent Result Output.
 */
module NPU_top (
    // Clock & Reset
    input logic clk,
    input logic rst_n,

    // Control Plane (MMIO)
    input  logic [31:0] mmio_npu_cmd,
    output logic [31:0] mmio_npu_stat,

    // AXI4-Stream Interfaces (Clean & Modern)
    axis_if.slave  S_AXIS_FMAP,   // Feature Map Input 0 (128-bit, HPC0)
    axis_if.slave  S_AXI_WEIGHT,  // Weight Matrix Input (512-bit)
    axis_if.master M_AXIS_RESULT, // Final Result Output (128-bit)

    // Auxiliary Streaming Ports
    axis_if.slave  S_AXIS_HP1,
    axis_if.slave  S_AXIS_HP2,
    axis_if.slave  S_AXIS_HP3,
    axis_if.slave  S_AXIS_HPC1,  // Feature Map Input 1 (128-bit, HPC1)
    axis_if.master M_AXIS_ACP,

    // Secondary MMIO Interface
    input  logic [31:0] s_axi_hpm1_vliw,
    output logic [31:0] s_axi_hpm1_stat
);

  // ===| FMap Preprocessing Pipeline (The Common Path) |=======
  logic [`FIXED_MANT_WIDTH-1:0] fmap_broadcast       [0:`ARRAY_SIZE_H-1];
  logic                         fmap_broadcast_valid;
  logic [  `BF16_EXP_WIDTH-1:0] cached_emax_out      [0:`ARRAY_SIZE_H-1];

  preprocess_fmap u_fmap_pre (
      .clk(clk),
      .rst_n(rst_n),
      .i_clear(npu_clear),

      // HPC Streaming Inputs
      .S_AXIS_FMAP0(S_AXIS_FMAP),
      .S_AXIS_FMAP1(S_AXIS_HPC1),

      // Control
      .i_rd_start(global_sram_rd_start),

      // Preprocessed Outputs (to Branch Engines)
      .o_fmap_broadcast(fmap_broadcast),
      .o_fmap_valid(fmap_broadcast_valid),
      .o_cached_emax(cached_emax_out)
  );


  // 2. Weight Pipeline (HP0~3 -> Systolic Array)
  logic [`AXI_DATA_WIDTH-1:0] weight_fifo_data [0:`AXI_WEIGHT_PORT_CNT-1];
  logic                       weight_fifo_valid[0:`AXI_WEIGHT_PORT_CNT-1];
  logic                       weight_fifo_ready[0:`AXI_WEIGHT_PORT_CNT-1];

  genvar i;
  generate
    for (i = 0; i < `AXI_WEIGHT_PORT_CNT; i++) begin : weight_fifos
      xpm_fifo_axis #(
          .FIFO_DEPTH(`XPM_FIFO_DEPTH),
          .TDATA_WIDTH(`AXI_DATA_WIDTH),
          .FIFO_MEMORY_TYPE("block"),
          .CLOCKING_MODE("common_clock")
      ) u_w_fifo (
          .s_aclk(clk),
          .s_aresetn(rst_n),
          .s_axis_tdata(s_axis_weight_tdata[i]),
          .s_axis_tvalid(s_axis_weight_tvalid[i]),
          .s_axis_tready(s_axis_weight_tready[i]),
          .m_axis_tdata(weight_fifo_data[i]),
          .m_axis_tvalid(weight_fifo_valid[i]),
          .m_axis_tready(weight_fifo_ready[i])
      );
    end
  endgenerate

  vdotm_top#(
      .line_lengt(32),
      .line_cnt(128),
      .fmap_line_cnt(32),
      .reduction_rate(4),
      .in_weight_size(`INT4),
      .in_fmap_size(`BF16),
      .in_fmap_e_size(`BF16_EXP),
      .in_fmap_m_size(`BF16_MANTISSA)
  ) (
      .clk(clk),
      .rst_n(rst_n),

      // weight
      .i_valid(),
      .IN_weight(),

      .IN_fMAP_valid(),
      .IN_feature_map(),

      .IN_fmap_broadcast(fmap_broadcast),
      .IN_fmap_broadcast_valid(fmap_broadcast_valid),

      // e_max (from Cache for Normalization alignment)
      .IN_cached_emax_out(cached_emax_out),

      .OUT_featureMAP_BF16(),
      .OUT_final_fp32(),
      .OUT_final_valid()
  );


  // 3. Systolic Array Engine (Modularized)
  logic [`DSP48E2_POUT_SIZE-1:0] raw_res_sum      [0:`ARRAY_SIZE_H-1];
  logic                          raw_res_sum_valid[0:`ARRAY_SIZE_H-1];
  logic [   `BF16_EXP_WIDTH-1:0] delayed_emax_32  [0:`ARRAY_SIZE_H-1];

  stlc_systolic_top u_systolic_engine (
      .clk(clk),
      .rst_n(rst_n),
      .i_clear(npu_clear),

      .global_weight_valid(global_weight_valid),
      .global_inst(global_inst),
      .global_inst_valid(global_inst_valid),

      .fmap_broadcast(fmap_broadcast),
      .fmap_broadcast_valid(fmap_broadcast_valid),

      .cached_emax_out(cached_emax_out),

      // Weight Input from FIFO (Direct)
      .weight_fifo_data (weight_fifo_data[0]),
      .weight_fifo_valid(weight_fifo_valid[0]),
      .weight_fifo_ready(weight_fifo_ready[0]),

      .raw_res_sum(raw_res_sum),
      .raw_res_sum_valid(raw_res_sum_valid),
      .delayed_emax_32(delayed_emax_32)
  );

  // 4. Output Pipeline (Result Normalization -> Result Packer -> FIFO)
  // Normalizers
  logic [`BF16_WIDTH-1:0] norm_res_seq      [0:`ARRAY_SIZE_H-1];
  logic                   norm_res_seq_valid[0:`ARRAY_SIZE_H-1];


  genvar n;
  generate
    for (n = 0; n < `ARRAY_SIZE_H; n++) begin : gen_norm
      stlc_result_normalizer u_norm_seq (
          .clk(clk),
          .rst_n(rst_n),
          .data_in(raw_res_sum[n]),
          .e_max(delayed_emax_32[n]),
          .valid_in(raw_res_sum_valid[n]),
          .data_out(norm_res_seq[n]),
          .valid_out(norm_res_seq_valid[n])
      );
    end
  endgenerate

  // Packer
  logic [`AXI_DATA_WIDTH-1:0] packed_res_data;
  logic                       packed_res_valid;
  logic                       packed_res_ready;

  FROM_stlc_result_packer u_packer (
      .clk(clk),
      .rst_n(rst_n),
      .row_res(norm_res_seq),
      .row_res_valid(norm_res_seq_valid),
      .packed_data(packed_res_data),
      .packed_valid(packed_res_valid),
      .packed_ready(packed_res_ready),
      .o_busy(packer_busy_status)
  );

  // Output FIFO
  xpm_fifo_axis #(
      .FIFO_DEPTH(`XPM_FIFO_DEPTH),
      .TDATA_WIDTH(`AXI_DATA_WIDTH),
      .FIFO_MEMORY_TYPE("block"),
      .CLOCKING_MODE("common_clock")
  ) u_output_fifo (
      .s_aclk(clk),
      .m_aclk(clk),
      .s_aresetn(rst_n),
      .s_axis_tdata(packed_res_data),
      .s_axis_tvalid(packed_res_valid),
      .s_axis_tready(packed_res_ready),
      .m_axis_tdata(m_axis_result_tdata),
      .m_axis_tvalid(m_axis_result_tvalid),
      .m_axis_tready(m_axis_result_tready)
  );


  // Status Assignment
  assign mmio_npu_stat[1] = 1'b0;
  assign mmio_npu_stat[31:2] = 30'd0;

endmodule
