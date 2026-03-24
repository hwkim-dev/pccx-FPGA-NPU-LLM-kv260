# Code Documentation (1/8) — C++ Acceleration Layer

> **Target Files**: `vulkan_core.cpp` · `my_accelerator.cpp`
> **Role**: Low-level matrix operation kernels based on iGPU (Vulkan) and CPU (AVX2/OpenMP)

---

## 1. `vulkan_core.cpp`

### Overview

A C++ shared library that performs GEMV (General Matrix-Vector Multiply) operations on INT4 quantized weights on the iGPU using the Vulkan Compute API. It is called from Python (`IGPU_CORE.py`) via `ctypes`.

**Core Design Pattern: Ping-Pong Buffering**
While the CPU asynchronously uploads the weights for the next layer to VRAM, the GPU proceeds with operations on the previous buffer. It hides the memory transfer latency between layers by overlapping it with the computation time.

```
[Layer i]   GPU Computation (Buffer A) │ CPU Asynchronous Transfer (W[i+1] → Buffer B)
[Layer i+1] GPU Computation (Buffer B) │ CPU Asynchronous Transfer (W[i+2] → Buffer A)
```

---

### Global State

| Variable | Type | Description |
|---|---|---|
| `instance` | `VkInstance` | Vulkan instance handle |
| `physicalDevice` | `VkPhysicalDevice` | Physical GPU device handle |
| `device` | `VkDevice` | Logical device handle |
| `computeQueue` | `VkQueue` | Compute-only queue |
| `computePipeline` | `VkPipeline` | Compiled compute pipeline |
| `g_matBuf[2]` | `VkBuffer[2]` | Ping-pong weight buffers (300MB each) |
| `g_xBuf` | `VkBuffer` | Input vector buffer (MAX_K × 4 bytes) |
| `g_scaleBuf` | `VkBuffer` | INT4 dequant scale buffer |
| `g_outBuf` | `VkBuffer` | Output buffer (MAX_M × 4 bytes) |
| `g_descriptorSet[2]` | `VkDescriptorSet[2]` | Descriptor sets for each ping-pong buffer |
| `weight_loader` | `std::future<void>` | Future handle of the asynchronous weight loader |

**Constants**

```cpp
#define MAX_M 262144   // Maximum output dimension (considering LM Head vocab size)
#define MAX_K 16384    // Maximum input dimension
```

g_scaleBuf allocation: 262144 × 4 = 1,048,576 bytes
LM Head requirement: 262400 × 4 = 1,049,600 bytes
                              ─────────────────
                              Short by 1,024 bytes (256 floats)

---

### Function Reference

#### `init_vulkan_engine()`

```c
extern "C" void init_vulkan_engine()
```

**Purpose**: Called exactly once at program startup to initialize the entire Vulkan pipeline.

**Initialization Sequence**:
1. Create `VkInstance` (API version 1.2)
2. Select the first physical device (`devices[0]`)
3. Create a compute queue (queue family 0)
4. Create a descriptor set layout (binding 0~3: x, mat, scale, out)
5. Define Push Constants layout (`PushConstants` structure)
6. Load SPIR-V shader (`gemv_int4_vector4.spv`) and create a compute pipeline
7. Allocate buffers (Ping-pong weights × 2, x, scale, out)
8. Create and bind descriptor pools and descriptor sets × 2
9. Create a command pool

**Buffer Memory Layout**:
```
g_matBuf[0] (300MB) ─── Ping buffer: Weights currently being computed
g_matBuf[1] (300MB) ─── Pong buffer: Prefetching next layer weights
g_xBuf      (64KB)  ─── Input vector x (float32)
g_scaleBuf  (1MB)   ─── Dequant scale (float32)
g_outBuf    (1MB)   ─── Result output (float32)
```

All buffers are allocated in CPU-GPU zero-copy shared memory with the `HOST_VISIBLE | HOST_COHERENT` flags (Optimized for APU unified memory environment).

---

#### `prefetch_weight_async()`

```c
extern "C" void prefetch_weight_async(
    const uint8_t* mat_p,  // Source: INT4 packed weight pointer in CPU RAM
    int M_out,             // Number of output rows
    int K_in,              // Input dimension (unpacked basis)
    int buf_idx            // Target buffer index (0 or 1)
)
```

**Purpose**: Asynchronously copies weight data to the specified ping-pong buffer in a **background thread** using `std::async`.

Copy size: `M_out × (K_in / 2)` bytes (INT4 packed basis)

> `weight_loader.wait()` automatically guarantees synchronization before calling `run_vulkan_gemv_pingpong()`.

---

#### `run_vulkan_gemv_pingpong()`

```c
extern "C" void run_vulkan_gemv_pingpong(
    const float* x,        // Input vector (float32, K_in size)
    const float* scale,    // Dequant scale (float32, M_out size)
    float* out,            // Output vector (float32, M_out size)
    int M_out,
    int K_in,
    int buf_idx            // Ping-pong buffer index to use
)
```

**Purpose**: Executes GPU GEMV using the weights in the specified ping-pong buffer (`buf_idx`).

**Execution Flow**:
1. `weight_loader.wait()` — Wait for asynchronous prefetch to complete
2. `memcpy` `x`, `scale` → GPU shared buffer
3. Record command buffer: Bind pipeline → Bind descriptor → Push Constants → `Dispatch`
4. `vkQueueSubmit` + `vkQueueWaitIdle` — Synchronous execution
5. `memcpy` `g_outBuf` → `out` array

**Dispatch Size**: `ceil(M_out / 32)` workgroups (Based on shader local size of 32)

**Push Constants Structure**:
```cpp
struct PushConstants {
    uint32_t M_out;          // Number of output rows
    uint32_t K_in_vector4s;     // K_in / 32 (in uvec4 units)
};
```

---

#### `run_vulkan_gemv()` *(Legacy)*

```c
extern "C" void run_vulkan_gemv(
    const float* x,
    const uint8_t* mat_p,
    const float* scale,
    float* out,
    int M_out,
    int K_in
)
```

A legacy interface that directly copies weights to buffer[0] without ping-pong and executes synchronously. Called from `igpu_matmul()` in `IGPU_CORE.py`. It is recommended to use `run_vulkan_gemv_pingpong()` in new code.

---

#### `createBuffer()` *(Internal Utility)*

```cpp
void createBuffer(
    VkDeviceSize size,
    VkBufferUsageFlags usage,
    VkMemoryPropertyFlags properties,
    VkBuffer& buffer,
    VkDeviceMemory& bufferMemory,
    void** mappedData      // Output: Mapped pointer accessible from CPU
)
```

Handles buffer creation → Querying memory requirements → Memory allocation → Binding → `vkMapMemory` mapping in one step.

---

### Dependencies

| Item | Details |
|---|---|
| Runtime Dependencies | Vulkan SDK, SPIR-V shader (`C_DLL/gemv_int4_vector4.spv`) |
| Build Flags | `-lvulkan` |
| Python Interface | `IGPU_CORE.py` (`ctypes.CDLL`) |

---

---

## 2. `my_accelerator.cpp`

### Overview

A collection of high-performance numerical computation kernels executed on the CPU. It performs AVX2-level parallel computation utilizing OpenMP SIMD directives (`#pragma omp simd`) and the GCC auto-vectorizer. Called via `ctypes` in Python from `CPU_CORE.py` and `main.py`.

**Common Rules**:
- All functions are declared within an `extern "C"` block to allow Python `ctypes` to find the symbols.
- `float* __restrict__` keyword: Guarantees to the compiler that "this pointer memory does not overlap with any other pointer," enabling SIMD optimization.
- All operations are **In-place** (overwriting the input array with the result).

---

### Function Reference

#### `run_gelu_inplace()`

```c
void run_gelu_inplace(float* x, int length)
```

**Formula**:
$$ \text{GELU}(x) = 0.5 \cdot x \cdot \left(1 + \tanh\!\left(0.7978846 \cdot (x + 0.044715 \cdot x^3)\right)\right) $$

**Implementation Features**:
- The entire loop is SIMD parallelized using `#pragma omp simd`
- The constant `GELU_CONST = 0.7978845608028654f` is defined at compile time
- The intermediate value `cube = x³` is cached in a separate variable to prevent recalculation

**Callers**: `CPU_CORE.gelu()`, `run_gemv_int4_gelu()` within `my_accelerator.cpp`

---

#### `run_RMSNorm_inplace()`

```c
void run_RMSNorm_inplace(float* x, const float* gamma, int length)
```

**Formula**:
$$ \text{RMSNorm}(x_i) = \frac{x_i}{\sqrt{\frac{1}{n}\sum x_i^2 + \varepsilon}} \cdot \gamma_i \quad (\varepsilon = 10^{-6}) $$

**Implementation Features**:
- Summation loop: `#pragma omp simd reduction(+:sum)` — Final aggregation after parallel addition
- The `sum` variable is declared as `double` to prevent float32 overflow when accumulating 2048 dimensions
- Single reciprocal calculation for `inv_rms` followed by multiplication (instead of division)

**Callers**: Wrapper function `rms_norm()` in `main.py`

---

#### `run_unpack_int4_inplace()`

```c
void run_unpack_int4_inplace(
    const uint8_t* packed,   // Input: INT4 × 2 packed uint8 array
    float scale,             // Row-wise dequant scale
    float* out,              // Output: float32 array (size = packed_length × 2)
    int packed_length
)
```

**Packing Format**:
```
packed[i] = (high_nibble << 4) | low_nibble
out[2*i]   = low_nibble  (signed -8~7) × scale
out[2*i+1] = high_nibble (signed -8~7) × scale
```

Sign restoration: `if (val > 7) val -= 16` (Two's complement 4-bit → int8 conversion)

**Callers**: `CPU_CORE.embedding()` — When looking up a single embedding row as a token ID

---

#### `run_rope_inplace()`

```c
void run_rope_inplace(
    float* x,           // [num_heads × dim] continuous float32 array (in-place)
    int pos,            // Current sequence position
    float theta_base,   // RoPE base frequency (Local: 10000, Global: 1000000)
    int num_heads,      // Number of attention heads
    int dim             // Dimensions per head (fixed at 256)
)
```

**Formula**:
$$ \text{cos\_vals}[i] = \cos\!\left(\text{pos} \cdot \theta_{\text{base}}^{-2i/d}\right), \quad
x'[i] = x[i]\cos - x[i+d/2]\sin, \quad x'[i+d/2] = x[i+d/2]\cos + x[i]\sin $$

**Implementation Optimization**:
- The cos/sin values are calculated **only once**, regardless of the number of heads (`cos_vals[128]`, `sin_vals[128]` stack caching)
- Structure: Outer loop (heads) + Inner SIMD loop (dimensions)

**Callers**: `CPU_CORE.cpu_rope()`

---

#### `run_softmax_inplace()`

```c
void run_softmax_inplace(float* logits, int length, float temperature)
```

**Formula**: Temperature scaling → Subtract Max (safe exp) → Summation → Normalization

**Implementation Features**:
- Temperature division and finding the maximum are **fused into a single loop** (`reduction(max:max_val)`)
- `exp` calculation and summation are **fused into a single loop** (`reduction(+:sum_exp)`)
- `sum_exp` is accumulated as a `double` to guarantee precision for 256,000 softmax calculations
- `temperature < 1e-8` guard: Prevents division by zero

**Callers**: The `_sample()` function in `main.py`

---

#### `run_gemv_int4()`

```c
void run_gemv_int4(
    const float* vec,        // Input vector [K_in]
    const uint8_t* mat_p,    // INT4 packed weight matrix [M_out × K_in/2]
    const float* scale,      // Row-wise dequant scale [M_out]
    float* out,              // Output vector [M_out]
    int M_out,
    int K_in
)
```

**Formula**: `out[i] = scale[i] × Σ( vec[k] × dequant(mat_p[i][k]) )`

**Implementation Features**:
- `#pragma omp parallel for` — Distributes M_out rows across all CPU cores
- `#pragma omp simd reduction(+:acc)` — Processes the K loop of each row with AVX2 SIMD
- Unpacking (nibble extraction + sign extension) is handled inline in the inner loop

**Callers**: `CPU_MATRIX_CORE.igpu_matmul()` (CPU mode)

---

#### `run_gemv_int4_gelu()`

```c
void run_gemv_int4_gelu(
    const float* vec,
    const uint8_t* mat_p,
    const float* scale,
    float* out,
    int M_out,
    int K_in
)
```

Identical to `run_gemv_int4()`, but applies GELU to the output **immediately (fusion)**. Used in FFN Gate operations, it eliminates the unnecessary round trip of writing the output to memory and reading it back again.

**Callers**: `CPU_MATRIX_CORE.igpu_matmul_gelu()` (CPU mode, layer 10 and above)

---

### Build Configuration Recommendations

```bash
g++ -O3 -march=native -fopenmp -ffast-math \
    -shared -fPIC -o C_DLL/my_accelerator.so my_accelerator.cpp
```

| Flag | Reason |
|---|---|
| `-march=native` | Enables AVX2 auto-vectorization |
| `-fopenmp` | Processes `#pragma omp` directives |
| `-ffast-math` | Allows approximations to optimize `tanh`/`exp` |
| `-O3` | Highest level of optimization |

---

### Function Call Map

```
Python (main.py / CPU_CORE.py)
    │
    ├── rms_norm()         ──→  run_RMSNorm_inplace()
    ├── gelu()             ──→  run_gelu_inplace()
    ├── cpu_rope()         ──→  run_rope_inplace()
    ├── embedding()        ──→  run_unpack_int4_inplace()
    ├── _sample()          ──→  run_softmax_inplace()
    └── CPU_MATRIX_CORE
            ├── igpu_matmul()      ──→  run_gemv_int4()
            └── igpu_matmul_gelu() ──→  run_gemv_int4_gelu()
```
