// ===| GEMM_Array.svh (compatibility shim) |====================================
// Historically this file defined ARRAY_SIZE_H / ARRAY_SIZE_V independently.
// Single-source the values from npu_arch.svh to avoid redefinition warnings
// and keep downstream `include "GEMM_Array.svh" lines working.
// ===============================================================================

`ifndef GEMM_ARRAY_SVH
`define GEMM_ARRAY_SVH

`include "npu_arch.svh"

// Extra constants previously scoped to MAT_CORE alone.
`ifndef MINIMUM_DELAY_LINE_LENGTH
`define MINIMUM_DELAY_LINE_LENGTH 1
`endif

`ifndef gemm_instruction_dispatcher_CLOCK_CONSUMPTION
`define gemm_instruction_dispatcher_CLOCK_CONSUMPTION 1
`endif

`endif // GEMM_ARRAY_SVH
