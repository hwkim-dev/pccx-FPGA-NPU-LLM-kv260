`define X64_HEADSIZE 8

`define X64_HIGHBIT_RMOV_HD 26
`define X64_LOWBIT_RMOV_HD 31

`ifndef ISA_X64_SVH
`define ISA_X64_SVH

typedef logic [16:0] dest_addr_t;
typedef logic [16:0] src_addr_t;

typedef logic [7:0] loop_cnt_t;
typedef logic [7:0] long_loop_cnt_t;
typedef logic [30:0] max_loop_cnt_t;
typedef logic [55:0] VLIW_instruction_x64;


typedef struct packed {
  logic [63:0] data;
  logic [7:0]  byte_en;
} x64_payload_t;


//8bit
typedef struct packed {
  logic findemax;
  logic accm;
  logic w_scale;
  logic [4:0] reserved;
} flags_t;

/*─────────────────────────────────────────────
  Opcode table
  ─────────────────────────────────────────────*/
typedef enum logic [4:0] {
  OP_VDOTM  = 4'h0,
  OP_MDOTM  = 4'h1,
  OP_MEMCPY = 4'h2
} opcode_t;


/*─────────────────────────────────────────────
  Field layouts per opcode (26-bit payloads)
  ─────────────────────────────────────────────*/

// V dot M / M dot M
typedef struct packed {
  dest_addr_t     dest_addr;
  flags_t         flags;
  src_addr_t      src_addr;
  long_loop_cnt_t long_loop_cnt;
} payload_vdotm_t;


// M dot M
typedef struct packed {
  dest_addr_t     dest_addr;
  flags_t         flags;
  src_addr_t      src_addr;
  long_loop_cnt_t long_loop_cnt;
} payload_mdotm_t;


typedef struct packed {
  logic          to_device;
  dest_addr_t    dest_addr;
  loop_cnt_t     loop_cnt_short;
  max_loop_cnt_t loop_cnt_long;
} payload_memcpy_t;

/*─────────────────────────────────────────────
  Full 32-bit instruction word
  Fixed header (6b) + union payload (26b)
  ─────────────────────────────────────────────*/
typedef struct packed {
  logic [1:0] cmd_chaining;
  opcode_t    opcode;

  union packed {
    payload_dotm_t  dotm;   // V dot M / M dot M
    payload_memcpy_t memcpy; // memcpy
    override_memcpy_t override_memcpy;
    override_chain_memcpy_t override_chain_memcpy;
    logic [25:0]    raw;
  } payload;

} instruction_x64_high_t;

typedef struct packed {
  logic [1:0] cmd_chaining;

  union packed {
    payload_dotm_t  dotm;   // V dot M / M dot M
    payload_memcpy_t memcpy; // memcpy
    override_memcpy_t override_memcpy;
    override_chain_memcpy_t override_chain_memcpy;
    logic [29:0]    raw;
  } payload;
} instruction_x64_low_t;


/*─────────────────────────────────────────────
  [1] Compute Micro-Op (For DSP48E2 Array)
  Dispatched to the Compute Queue
  ─────────────────────────────────────────────*/
typedef struct packed {
  dest_addr_t     dest_reg;
  src_addr_t      src_addr;
  long_loop_cnt_t loop_cnt;
  flags_t         flags;
} vdotm_uop_x64_t;

typedef struct packed {
  dest_addr_t     dest_reg;
  src_addr_t      src_addr;
  long_loop_cnt_t loop_cnt;
  flags_t         flags;
} mdotm_uop_x64_t;

typedef struct packed {
  logic          to_divice;
  dest_addr_t    dest_addr;
  max_loop_cnt_t loop_cnt;
} memory_uop_x64_t;

`endif
