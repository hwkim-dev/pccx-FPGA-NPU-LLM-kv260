`define ARRAY_SIZE_H 32
`define ARRAY_SIZE_V 32

`define stlc_instruction_dispatcher_CLOCK_CONSUMPTION 1

// <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
// DSP INSTRUCTION
`define DSP_INSTRUCTION_CNT 4

`define DSP_IDLE_MOD 2'b00
`define DSP_SYSTOLIC_MOD_P 2'b01
`define DSP_GEMV_STATIONARY_MOD 2'b10 // Used for Weight-Stationary GEMV
`define DSP_SHIFT_RESULT_MOD 2'b11

/*
`define DSP_SUB_MOD = 2'b11
`define DSP_INV_DIV_MOD
*/
// >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



// INT4 - DSP48E2_MAXIN_H
`define STLC_MAC_UNIT_IN_H 4

// BFLOAT16 - DSP48E2_MAXIN_V
// aligned Mantissa size 27
`define STLC_MAC_UNIT_IN_V 27



// systolic delay line
`define MINIMUM_DELAY_LINE_LENGTH 1

// systolic delay line V | TYPE:INT4
`define INT4_WIDTH 4

// systolic delay line H | TYPE: BFLOAT 16
`define BFLOAT_WIDTH 16
