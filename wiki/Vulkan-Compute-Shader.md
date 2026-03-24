# Code Documentation (2/8) — Vulkan Compute Shader

> **Target Files**: `gemv_int4_vector4.comp` · `gemv_int4.comp`
> **Role**: GLSL compute shaders performing GEMV with INT4 quantized weights on the iGPU
> **Compilation**: `glslc gemv_int4_vector4.comp -o C_DLL/gemv_int4_vector4.spv`

---

## Comparison of the Two Shaders at a Glance

| Item | `gemv_int4_vector4.comp` | `gemv_int4.comp` |
| ------------------------ | --------------------------- | ------------------------ |
| **Status** | ✅ Currently in use (Production) | 🗃️ Old version (Legacy) |
| **Memory Access Unit** | `uvec4` (128-bit, 16 bytes) | `uint` (32-bit, 4 bytes) |
| **INT4 processed per loop** | 32 | 8 |
| **Push Constant Field Name** | `K_in_vector4s` | `K_in_uints` |
| **Binding 1 Type** | `uvec4[]` | `uint[]` |
| **Cache Efficiency** | High (128-bit burst) | Low (32-bit units) |

---

## 1. `gemv_int4_vector4.comp` (Currently in use)

### Overview

An optimized shader that reads a `uvec4` (128-bit vector type) at once, processing 32 INT4 values per loop. It maximizes the utilization of modern GPUs' 128-bit memory bus.

### Binding Layout

```glsl
layout(binding = 0) readonly buffer InputX  { float  x[];        };  // Input vector
layout(binding = 1) readonly buffer MatP    { uvec4  mat_vec4[]; };  // INT4 packed weights (128-bit units)
layout(binding = 2) readonly buffer Scale   { float  scale[];    };  // Row-wise dequant scale
layout(binding = 3) writeonly buffer Output { float  out_vec[];  };  // Output vector
```

### Push Constants

```glsl
layout(push_constant) uniform PushConstants {
    uint M_out;           // Number of output rows (= number of weight rows)
    uint K_in_vector4s;   // K dimension divided by uvec4 units (= K_in / 32)
} params;
```

> **Why K_in / 32?**
> 1 `uvec4` = 4 bytes × 4 = 16 bytes = 128 bits.
> Since INT4 is 4 bits, there are 2 in 1 byte, and 32 in 16 bytes → Processing 32 INT4s per 1 `uvec4`.

### Execution Structure

```
Workgroup size: local_size_x = 32
Total dispatches: ceil(M_out / 32) workgroups
→ 1 thread = Responsible for 1 output row
```

### Main Logic Detail (`main()`)

```glsl
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= params.M_out) return;     // Early exit for out-of-bounds threads

    float acc = 0.0;
    uint row_offset = row * params.K_in_vector4s;  // Starting uvec4 index for the row

    for (uint k = 0; k < params.K_in_vector4s; k++) {
        uvec4 packed_128 = mat_vec4[row_offset + k];  // ← Sweep load 128 bits
        uint x_idx = k * 32;

        for (int v = 0; v < 4; v++) {          // Iterate over the 4 uint32 elements of uvec4
            uint packed_32 = packed_128[v];    // x, y, z, w components
            uint x_v_idx = x_idx + (v * 8);   // Each uint handles 8 elements of x

            for (int i = 0; i < 4; i++) {      // Decompose uint32 in 8-bit chunks 4 times
                uint byte_val = (packed_32 >> (i * 8)) & 0xFF;

                // Lower 4 bits → INT4 (signed)
                int low  = int(byte_val & 0x0F);
                if (low  > 7) low  -= 16;

                // Upper 4 bits → INT4 (signed)
                int high = int((byte_val >> 4) & 0x0F);
                if (high > 7) high -= 16;

                acc += x[x_v_idx + i*2    ] * float(low);
                acc += x[x_v_idx + i*2 + 1] * float(high);
            }
        }
    }

    out_vec[row] = acc * scale[row];  // Dequantize: Scale multiplication
}
```

### Memory Access Pattern Visualization

```
mat_vec4[row_offset + k]  →  uvec4 (128 bits)
│
├── [v=0] packed_128.x  (uint32, 4 bytes)
│   ├── [i=0] byte[0]: low=INT4, high=INT4  → x[0], x[1]
│   ├── [i=1] byte[1]: low=INT4, high=INT4  → x[2], x[3]
│   ├── [i=2] byte[2]: low=INT4, high=INT4  → x[4], x[5]
│   └── [i=3] byte[3]: low=INT4, high=INT4  → x[6], x[7]
├── [v=1] packed_128.y  → x[8]  ~ x[15]
├── [v=2] packed_128.z  → x[16] ~ x[23]
└── [v=3] packed_128.w  → x[24] ~ x[31]

1 uvec4 load = Unpacking 32 INT4s = Dot product with 32 elements of x
```

### INT4 Sign Restoration Principle

```
nibble value range: 0x0 ~ 0xF (0 ~ 15, unsigned)
signed interpretation: 0 ~ 7   → Keep positive
                       8 ~ 15  → 8 as -8, 9 as -7, ... 15 as -1 (if val > 7: val -= 16)

Example:
  nibble = 0b1010 = 10 → 10 > 7 → 10 - 16 = -6 (signed)
  nibble = 0b0011 = 3  → 3 ≤ 7  → +3 (signed)
```

### Dequantization

$$ \text{out}[row] = \text{scale}[row] \times \sum_{k} \left( x[2k] \cdot w_{low}^{(k)} + x[2k+1] \cdot w_{high}^{(k)} \right) $$

The scale is a float32 value calculated **per row** as `max(|w|) / 7.0` in `quantize.py`.

---

## 2. `gemv_int4.comp` (Legacy)

### Overview

An older version of the shader that reads memory in `uint` (32-bit) units. The predecessor to `gemv_int4_vector4.comp`, it has the same unpacking logic but has 4 times lower memory load efficiency. Since it was replaced by `gemv_int4_vector4.spv` in `vulkan_core.cpp`, it is not actually executed.

### Binding Layout

```glsl
layout(binding = 0) readonly buffer InputX  { float x[];   };
layout(binding = 1) readonly buffer MatP    { uint  mat[]; };  // ← uint[] (32-bit units, old version)
layout(binding = 2) readonly buffer Scale   { float scale[]; };
layout(binding = 3) writeonly buffer Output { float out_vec[]; };
```

### Push Constants

```glsl
layout(push_constant) uniform PushConstants {
    uint M_out;
    uint K_in_uints;   // K dimension divided by uint units (= K_in / 8)
} params;
```

> **Why K_in / 8**: 1 `uint` = 4 bytes = 32 bits. 8 INT4 values of 4 bits each = 32 bits.

### Main Logic

```glsl
void main() {
    uint row = gl_GlobalInvocationID.x;
    if (row >= params.M_out) return;

    float acc = 0.0;
    uint row_offset = row * params.K_in_uints;

    for (uint k = 0; k < params.K_in_uints; k++) {
        uint packed_32 = mat[row_offset + k];   // ← Loads only 32 bits (limit of the old version)
        uint x_idx = k * 8;

        for (int i = 0; i < 4; i++) {           // Decompose 32 bits into 8-bit chunks 4 times
            uint byte_val = (packed_32 >> (i * 8)) & 0xFF;

            int low  = int(byte_val & 0x0F);
            if (low  > 7) low  -= 16;

            int high = int((byte_val >> 4) & 0x0F);
            if (high > 7) high -= 16;

            acc += x[x_idx + i*2    ] * float(low);
            acc += x[x_idx + i*2 + 1] * float(high);
        }
    }

    out_vec[row] = acc * scale[row];
}
```

---

## Evolution Relationship of the Two Shaders

```
gemv_int4.comp (Old version)
│
│  Problem: Reads 32 bits at a time using uint[]
│           → Processes 8 INT4s per loop
│           → 25% memory bus utilization
│
▼
gemv_int4_vector4.comp (Current)
   Improvement: Reads 128 bits at a time using uvec4[]
                → Processes 32 INT4s per loop
                → 100% memory bus utilization
                → The number of K loop iterations decreases by 4 times
```

## Build and Deploy

```bash
# Compile to SPIR-V binaries
glslc gemv_int4_vector4.comp -o C_DLL/gemv_int4_vector4.spv
glslc gemv_int4.comp         -o C_DLL/gemv_int4.spv          # Legacy, unused

# Load in vulkan_core.cpp
auto shaderCode = readFile("C_DLL/gemv_int4_vector4.spv");
```

## Complete Call Path

```
main.py
  └── hw_compute_pingpong() / hw_matmul()
        └── IGPU_CORE.py
              ├── compute_pingpong()  → vk_lib.run_vulkan_gemv_pingpong()
              └── igpu_matmul()       → vk_lib.run_vulkan_gemv()
                    └── vulkan_core.cpp
                          └── vkCmdDispatch() → gemv_int4_vector4.spv
                                                 (Executed on GPU)
```

## Performance Considerations

| Item | Value | Notes |
| ---------------- | --------- | ----------------------------------------------- |
| Workgroup Size | 32 | Recommended to match the GPU warp/wavefront size |
| Maximum M_out | 262,144 | `MAX_M` constant (LM Head vocab size) |
| Maximum K_in | 16,384 | `MAX_K` constant (FFN intermediate dimension) |
| Weight Buffer Limit | 300MB × 2 | Each ping-pong buffer. Needs splitting if FFN W_gate (~562MB) is exceeded |
| Scale Precision | float32 | Minimizes quantization error |
