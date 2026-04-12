// ===| NPU Architecture Macros |=================================================
// NPU-level architectural constants that must be `define (used in port
// declarations and generate ranges, where localparams cannot be used).
//
// These are design choices — they change when the NPU architecture changes,
// not when the board changes. Board-specific values live in kv260_device.svh.
// ===============================================================================

`ifndef NPU_ARCH_SVH
`define NPU_ARCH_SVH

// ===| ISA |=====================================================================
`define ISA_WIDTH                   64   // VLIW instruction word width (bits)
`define ISA_OPCODE_WIDTH             4   // Top 4 bits of every instruction
`define ISA_BODY_WIDTH              60   // Instruction body after opcode is stripped

// ISA compilation mode selectors (used by ctrl_npu_decoder and test benches)
`define MOD_X64                      1   // 64-bit VLIW mode (active)
`define MOD_X32                      0   // 32-bit mode (legacy, unused)
`define U_OPERATION_WIDTH           59   // Usable body bits (ISA_BODY_WIDTH - 1 header)
`define INST_HEAD_ARCH_MOD_BIT       1   // Architecture mode selector bit position

// ===| Systolic Array (Matrix Core) |============================================
`define ARRAY_SIZE_H                32   // Horizontal: number of PE columns
`define ARRAY_SIZE_V                32   // Vertical  : number of PE rows
// Total pipeline latency (H + V + overhead), used for e_max delay pipe
`define SYSTOLIC_TOTAL_LATENCY      64

// ===| Data Type Widths (used in port declarations) |===========================
// These must remain `define because they appear in port width expressions.
// Semantic definitions live in dtype_pkg.

// BF16
`define BF16_WIDTH                  16
`define BF16_EXP_WIDTH              8
`define BF16_MANT_WIDTH             7

// INT4 (Weight)
`define INT4_WIDTH                  4

// Fixed-point mantissa width after BF16 emax alignment
// = BF16_MANT_WIDTH(7) + leading-1 + sign + 18-bit integer headroom = 27
`define FIXED_MANT_WIDTH            27

// FP32 (used for mixed-precision output path)
`define FP32_WIDTH                  32

// DSP48E2 accumulator output used as port width
`define DSP_P_OUT_WIDTH             48   // = `DEVICE_DSP_P_WIDTH

// ===| AXI Stream Port Widths |=================================================
`define AXI_STREAM_WIDTH            128  // Standard AXI-S data width used throughout

// HP port aggregated width (all 4 lanes combined for GEMM weight bus)
`define HP_TOTAL_WIDTH              512  // `DEVICE_HP_PORT_CNT * `DEVICE_HP_SINGLE_WIDTH_BIT
`define HP_SINGLE_WIDTH             128  // = `DEVICE_HP_SINGLE_WIDTH_BIT

// ===| Memory / Cache Sizes |===================================================
// FMap L1 cache (SRAM) capacity — depth in units of AXI_STREAM_WIDTH words
`define FMAP_CACHE_DEPTH            2048
`define FMAP_ADDR_WIDTH             11   // log2(FMAP_CACHE_DEPTH)

// ===| DSP48E2 Instruction Modes |==============================================
// Control codes sent along with fmap in the systolic array V-path
`define DSP_IDLE_MOD                2'b00
`define DSP_SYSTOLIC_MOD_P          2'b01   // Normal systolic MAC
`define DSP_GEMV_STATIONARY_MOD     2'b10   // Weight-stationary GEMV
`define DSP_SHIFT_RESULT_MOD        2'b11   // Shift output (for normalization)

// ===| Pipeline Type Selector |=================================================
`define PIPELINE_GEMV               0
`define PIPELINE_GEMM               1

// ===| INT4 Range Helpers |=====================================================
`define INT4_MAX_VAL                7
`define INT4_MIN_VAL                -8
`define INT4_RANGE                  16

`endif // NPU_ARCH_SVH
