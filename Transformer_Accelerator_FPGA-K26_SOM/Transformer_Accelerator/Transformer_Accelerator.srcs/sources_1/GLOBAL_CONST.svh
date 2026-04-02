`define TRUE 1'b1
`define FALSE 1'b0

// ===| KV260's DSP48E2 |========================

`define DSP48E2_MAXIN_H 18
`define DSP48E2_MAXIN_V 30
`define DSP48E2_MAXOUT 48
`define PREG_SIZE 48
`define MREG_SIZE 48

// ===| KV260's DSP48E2 - END |==================

`define BF16 16
`define BF16_EXP 8
`define BF16_MANTISSA 7
`define INT4 4

`define DATA_WIDTH 27

`define TRUE 1
`define FALSE 0





// ===| BF16 Data Formats |==========================
`define BF16_WIDTH 16
`define BF16_EXP_WIDTH 8
`define BF16_MANT_WIDTH 7
`define FIXED_MANT_WIDTH 27
// ===| BF16 Data Formats - END |====================


// ===| FP32 Data Formats |==========================
`define FMAP_CACHE_OUT_SIZE 32
`define FP32 32
// ===| FP32 Data Formats - END |====================

// ===| INT4 Data Formats |==========================
`define INT4_TO_IDX(val) ((val) + 8)
`define INT4_MAX_VAL 7
`define INT4_MIN_VAL -8
`define INT4_RANGE 16
// ===| INT4 Data Formats - END |====================



// ===| SYSTEM-WIDE ARCHITECTURAL CONSTANTS |==============

// [AXI-Stream & DMA]
`define HP_PORT_MAX_WIDTH 512
`define HP_PORT_SINGLE_WIDTH 128
`define HP_PORT_CNT 4
`define HP_WEIGHT_CNT(P_size, W_size) (P_size / W_size)

`define AXI_DATA_WIDTH 128
`define AXI_PORT_CNT 4

// [Elastic Buffers (FIFOs)]
`define XPM_FIFO_DEPTH 512

// [Feature Map Cache (SRAM)]
`define FMAP_CACHE_DEPTH 2048
`define FMAP_ADDR_WIDTH 11 // log2(2048)

// [Pipelining & Latency Hiding]
// Calculated as Array H (32) + Array V (32) + Pipeline Overheads
`define SYSTOLIC_TOTAL_LATENCY 64


// ========================================================

// DSP48E2_MAXOUT
`define DSP48E2_POUT_SIZE 48
`define DSP48E2_AB_WIDTH 48
`define DSP48E2_C_WIDTH 48
