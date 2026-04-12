`include "GLOBAL_CONST.svh"
`timescale 1ns / 1ps
`include "GEMM_Array.svh"
`include "mem_IO.svh"
`include "npu_interfaces.svh"

import isa_pkg::*;
import vec_core_pkg::*;
import bf16_math_pkg::*;

// ===| NPU Top |=================================================================
// Target: Kria KV260 @ 400 MHz
//
// Architecture V2 (SystemVerilog Interface Version):
//   HPC0 / HPC1  : 256-bit Feature Map caching bus (ACP port).
//   HP0  ~ HP3   : High-throughput Weight streaming (128-bit each).
//   HPM  (MMIO)  : Centralised control & VLIW instruction issuing (AXI-Lite).
//   ACP           : Coherent Result Output.
// ===============================================================================

module NPU_top (
    // ===| Clock & Reset |=======================================================
    input logic clk_core,
    input logic rst_n_core,

    input logic clk_axi,
    input logic rst_axi_n,

    // ===| Soft Clear (synchronous, active-high) |===============================
    input logic i_clear,

    // ===| Control Plane (MMIO) |================================================
    input  logic [31:0] mmio_npu_vliw,   // legacy MMIO word (unused — kept for pin compat)
    output logic [31:0] mmio_npu_stat,

    axil_if.slave S_AXIL_CTRL,

    // ===| HP Weight Ports — Matrix Core (Systolic) |============================
    axis_if.slave S_AXI_HP0_WEIGHT,
    axis_if.slave S_AXI_HP1_WEIGHT,

    // ===| HP Weight Ports — Vector Core (GEMV) |================================
    axis_if.slave S_AXI_HP2_WEIGHT,
    axis_if.slave S_AXI_HP3_WEIGHT,

    // ===| ACP Feature Map / Result (Full-Duplex) |==============================
    axis_if.slave  S_AXIS_ACP_FMAP,
    axis_if.master M_AXIS_ACP_RESULT
);

  // ===| Internal Wires — HP Weight (Core-side, post-CDC FIFO) |================
  axis_if #(.DATA_WIDTH(128)) M_CORE_HP0_WEIGHT ();
  axis_if #(.DATA_WIDTH(128)) M_CORE_HP1_WEIGHT ();
  axis_if #(.DATA_WIDTH(128)) M_CORE_HP2_WEIGHT ();
  axis_if #(.DATA_WIDTH(128)) M_CORE_HP3_WEIGHT ();

  // ===| Internal Wires — Instruction Path |=====================================
  logic GEMV_op_x64_valid_wire;
  logic GEMM_op_x64_valid_wire;
  logic memcpy_op_x64_valid_wire;
  logic memset_op_x64_valid_wire;
  logic cvo_op_x64_valid_wire;

  instruction_op_x64_t instruction;

  logic fifo_full_wire;

  // ===| Internal Wires — Compute Control |======================================
  // TODO: drive from Global_Scheduler once the scheduling FSM is implemented.
  logic        global_sram_rd_start;
  logic        global_weight_valid;
  logic [2:0]  global_inst;
  logic        global_inst_valid;
  logic        npu_clear;

  assign npu_clear          = i_clear;
  assign global_sram_rd_start = 1'b0;  // TODO: connect to scheduler
  assign global_weight_valid  = 1'b0;  // TODO: connect to scheduler
  assign global_inst          = 3'b0;  // TODO: connect to scheduler
  assign global_inst_valid    = 1'b0;  // TODO: connect to scheduler

  // ===| [1] NPU Controller |====================================================
  npu_controller_top #() u_npu_controller_top (
      .clk    (clk_core),
      .rst_n  (rst_n_core),
      .i_clear(i_clear),

      .S_AXIL_CTRL(S_AXIL_CTRL),

      .OUT_GEMV_op_x64_valid  (GEMV_op_x64_valid_wire),
      .OUT_GEMM_op_x64_valid  (GEMM_op_x64_valid_wire),
      .OUT_memcpy_op_x64_valid(memcpy_op_x64_valid_wire),
      .OUT_memset_op_x64_valid(memset_op_x64_valid_wire),
      .OUT_cvo_op_x64_valid   (cvo_op_x64_valid_wire),

      .OUT_op_x64(instruction)
  );

  // ===| [2] Global Scheduler |==================================================
  gemm_control_uop_t   GEMM_uop_wire;
  GEMV_control_uop_t   GEMV_uop_wire;
  memory_control_uop_t LOAD_uop_wire;
  memory_set_uop_t     mem_set_uop;
  cvo_control_uop_t    CVO_uop_wire;

  Global_Scheduler #() u_Global_Scheduler (
      .clk_core  (clk_core),
      .rst_n_core(rst_n_core),

      .IN_GEMV_op_x64_valid  (GEMV_op_x64_valid_wire),
      .IN_GEMM_op_x64_valid  (GEMM_op_x64_valid_wire),
      .IN_memcpy_op_x64_valid(memcpy_op_x64_valid_wire),
      .IN_memset_op_x64_valid(memset_op_x64_valid_wire),
      .IN_cvo_op_x64_valid   (cvo_op_x64_valid_wire),

      .instruction(instruction),

      .OUT_GEMM_uop    (GEMM_uop_wire),
      .OUT_GEMV_uop    (GEMV_uop_wire),
      .OUT_LOAD_uop    (LOAD_uop_wire),
      .OUT_mem_set_uop (mem_set_uop),
      .OUT_CVO_uop     (CVO_uop_wire)
  );

  // ===| [3] Memory Dispatcher |=================================================
  mem_dispatcher #() u_mem_dispatcher (
      .clk_core  (clk_core),
      .rst_n_core(rst_n_core),

      .clk_axi  (clk_axi),
      .rst_axi_n(rst_axi_n),

      .S_AXIS_ACP_FMAP  (S_AXIS_ACP_FMAP),
      .M_AXIS_ACP_RESULT(M_AXIS_ACP_RESULT),

      .IN_LOAD_uop   (LOAD_uop_wire),
      .IN_mem_set_uop(mem_set_uop),

      .OUT_fifo_full(fifo_full_wire)
  );

  // ===| [4] HP Weight Buffer (CDC FIFO: AXI -> Core clock) |====================
  mem_HP_buffer #() u_HP_buffer (
      .clk_core  (clk_core),
      .rst_n_core(rst_n_core),
      .clk_axi   (clk_axi),
      .rst_axi_n (rst_axi_n),

      .S_AXI_HP0_WEIGHT(S_AXI_HP0_WEIGHT),
      .S_AXI_HP1_WEIGHT(S_AXI_HP1_WEIGHT),
      .S_AXI_HP2_WEIGHT(S_AXI_HP2_WEIGHT),
      .S_AXI_HP3_WEIGHT(S_AXI_HP3_WEIGHT),

      .M_CORE_HP0_WEIGHT(M_CORE_HP0_WEIGHT),
      .M_CORE_HP1_WEIGHT(M_CORE_HP1_WEIGHT),
      .M_CORE_HP2_WEIGHT(M_CORE_HP2_WEIGHT),
      .M_CORE_HP3_WEIGHT(M_CORE_HP3_WEIGHT)
  );

  // ===| [5] FMap Preprocessing Pipeline |=======================================
  logic [`FIXED_MANT_WIDTH-1:0] fmap_broadcast      [0:`ARRAY_SIZE_H-1];
  logic                         fmap_broadcast_valid;
  logic [`BF16_EXP_WIDTH-1:0]   cached_emax_out      [0:`ARRAY_SIZE_H-1];

  // TODO: preprocess_fmap port interface updated to 2D output — adapt once
  //       PIPELINE_CNT macro is defined and the port contract is finalised.
  preprocess_fmap #() u_fmap_pre (
      .clk    (clk_core),
      .rst_n  (rst_n_core),
      .i_clear(npu_clear),

      .S_AXIS_ACP_FMAP(S_AXIS_ACP_FMAP),

      .i_rd_start(global_sram_rd_start),

      .o_fmap_broadcast(fmap_broadcast),
      .o_fmap_valid    (fmap_broadcast_valid),
      .o_cached_emax   (cached_emax_out)
  );

  // ===| [6] Systolic Array Engine (Matrix Core) |================================
  logic [`DSP48E2_POUT_SIZE-1:0] raw_res_sum      [0:`ARRAY_SIZE_H-1];
  logic                          raw_res_sum_valid [0:`ARRAY_SIZE_H-1];
  logic [`BF16_EXP_WIDTH-1:0]    delayed_emax_32   [0:`ARRAY_SIZE_H-1];

  GEMM_systolic_top #() u_systolic_engine (
      .clk    (clk_core),
      .rst_n  (rst_n_core),
      .i_clear(npu_clear),

      .global_weight_valid(global_weight_valid),
      .global_inst        (global_inst),
      .global_inst_valid  (global_inst_valid),

      .IN_fmap_broadcast      (fmap_broadcast),
      .IN_fmap_broadcast_valid(fmap_broadcast_valid),
      .IN_cached_emax_out     (cached_emax_out),

      .IN_weight_fifo_data (M_CORE_HP0_WEIGHT.tdata),
      .IN_weight_fifo_valid(M_CORE_HP0_WEIGHT.tvalid),
      .weight_fifo_ready   (M_CORE_HP0_WEIGHT.tready),

      .raw_res_sum      (raw_res_sum),
      .raw_res_sum_valid(raw_res_sum_valid),
      .delayed_emax_32  (delayed_emax_32)
  );

  // ===| [7] Result Normalizers (one per systolic column) |======================
  logic [`BF16_WIDTH-1:0] norm_res_seq      [0:`ARRAY_SIZE_H-1];
  logic                   norm_res_seq_valid[0:`ARRAY_SIZE_H-1];

  genvar n;
  generate
    for (n = 0; n < `ARRAY_SIZE_H; n++) begin : gen_norm
      gemm_result_normalizer u_norm_seq (
          .clk      (clk_core),
          .rst_n    (rst_n_core),
          .data_in  (raw_res_sum[n]),
          .e_max    (delayed_emax_32[n]),
          .valid_in (raw_res_sum_valid[n]),
          .data_out (norm_res_seq[n]),
          .valid_out(norm_res_seq_valid[n])
      );
    end
  endgenerate

  // ===| [8] Result Packer |=====================================================
  logic [`AXI_STREAM_WIDTH-1:0] packed_res_data;
  logic                         packed_res_valid;
  logic                         packed_res_ready;
  logic                         packer_busy_status;

  FROM_gemm_result_packer #() u_packer (
      .clk         (clk_core),
      .rst_n       (rst_n_core),
      .row_res     (norm_res_seq),
      .row_res_valid(norm_res_seq_valid),
      .packed_data (packed_res_data),
      .packed_valid(packed_res_valid),
      .packed_ready(packed_res_ready),
      .o_busy      (packer_busy_status)
  );

  // ===| [9] Vector Core (GEMV) |================================================
  // TODO: IN_weight_A/B/C/D expect unpacked INT4 arrays — add weight unpack
  //       bridge between M_CORE_HP*_WEIGHT.tdata (128-bit flat) and the array port.
  // TODO: IN_num_recur and IN_activated_lane need to come from Global_Scheduler.

  logic [16:0] gemv_num_recur;
  logic        gemv_activated_lane[0:VecCoreDefaultCfg.num_gemv_pipeline-1];
  assign gemv_num_recur = '0;          // TODO: connect to scheduler
  assign gemv_activated_lane = '{default: 1'b0};  // TODO: connect to scheduler

  GEMV_top #(
      .param(VecCoreDefaultCfg)
  ) u_GEMV_top (
      .clk   (clk_core),
      .rst_n (rst_n_core),

      .IN_weight_valid_A(M_CORE_HP2_WEIGHT.tvalid),
      .IN_weight_valid_B(M_CORE_HP3_WEIGHT.tvalid),
      .IN_weight_valid_C(1'b0),
      .IN_weight_valid_D(1'b0),

      .IN_weight_A(M_CORE_HP2_WEIGHT.tdata),
      .IN_weight_B(M_CORE_HP3_WEIGHT.tdata),
      .IN_weight_C('0),
      .IN_weight_D('0),

      .OUT_weight_ready_A(M_CORE_HP2_WEIGHT.tready),
      .OUT_weight_ready_B(M_CORE_HP3_WEIGHT.tready),
      .OUT_weight_ready_C(),
      .OUT_weight_ready_D(),

      .IN_fmap_broadcast      (fmap_broadcast),
      .IN_fmap_broadcast_valid(fmap_broadcast_valid),
      .IN_num_recur           (gemv_num_recur),
      .IN_cached_emax_out     (cached_emax_out),
      .IN_activated_lane      (gemv_activated_lane),

      .OUT_final_fmap_A(),
      .OUT_final_fmap_B(),
      .OUT_final_fmap_C(),
      .OUT_final_fmap_D(),

      .OUT_result_valid_A(),
      .OUT_result_valid_B(),
      .OUT_result_valid_C(),
      .OUT_result_valid_D()
  );

  // ===| [10] CVO Core |==========================================================
  // TODO: IN_data / OUT_result streams need to be routed through mem_dispatcher
  //       once L2 read/write DMA paths for CVO are implemented.

  logic [15:0] cvo_result;
  logic        cvo_result_valid;
  logic        cvo_busy;
  logic        cvo_done;

  CVO_top u_CVO_top (
      .clk             (clk_core),
      .rst_n           (rst_n_core),
      .i_clear         (npu_clear),

      .IN_uop          (CVO_uop_wire),
      .IN_uop_valid    (cvo_op_x64_valid_wire),
      .OUT_uop_ready   (),

      // ===| Streams (TODO: connect to L2 DMA paths) |===
      .IN_data         (16'd0),
      .IN_data_valid   (1'b0),
      .OUT_data_ready  (),

      .OUT_result      (cvo_result),
      .OUT_result_valid(cvo_result_valid),
      .IN_result_ready (1'b1),

      .IN_e_max        (16'd0),  // TODO: connect to cached e_max register

      .OUT_busy        (cvo_busy),
      .OUT_done        (cvo_done),
      .OUT_accm        ()
  );

  // ===| Status Register |=======================================================
  assign mmio_npu_stat[0]    = fifo_full_wire | cvo_busy;  // BUSY
  assign mmio_npu_stat[1]    = cvo_done;                   // DONE
  assign mmio_npu_stat[31:2] = 30'd0;

endmodule
