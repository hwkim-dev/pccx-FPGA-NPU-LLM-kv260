# Code Documentation (7/8) — FPGA NPU Engine & Architecture Design Document

> **Target Files**: `NPU_CORE.py` · `gemma3N_E4B_architecture.md`
> **Role**: Python control layer for FPGA-based NPU acceleration engine (`NPU_CORE`) + Reference document for overall model structure and hardware distribution (`gemma3N_E4B_architecture.md`)

---

## 1. `NPU_CORE.py`

### Overview

The **Python control layer for the FPGA RTL-based NPU hardware**. It is a **separate hardware target** from the current execution paths (`IGPU_CORE.py` / `CPU_MATRIX_CORE.py`) of the project, controlling a custom Systolic Array NPU implemented on a Xilinx/AMD FPGA via MMIO (Memory-Mapped I/O).

**Current Status**: Cannot be run directly without an FPGA board due to the `import MMIO` dependency. However, the `MMIO.SIMULATION_MODE` branch at the top of the function **completely mocks** the operation on a PC using NumPy.

---

### Hardware Architecture Overview

```
CPU (Python/NumPy)
    │  MMIO Register Control
    │  DMA Transfer
    ▼
Inside FPGA
  ┌─────────────────────────────────────┐
  │  AXI DMA  ──→  Ping-Pong BRAM      │
  │                    │               │
  │              Systolic Array NPU    │
  │              (32×32 PE Tile)       │
  │                    │               │
  │              ACC (Accumulator)     │
  │                    │               │
  │           RMSNorm / GeLU IP        │
  │                    │               │
  │              Result BRAM           │
  └─────────────────────────────────────┘
```

---

### MMIO Register Map

| Address | Role | Usage Example |
| ------------ | ------------------------------------------------------- | ----------------------------- |
| `0x00` | Control Register — Bit0: NPU_START (Pulse), Bit1: ACC_CLEAR | `write(0x00, 0x01)` Start calculation |
| `0x08` | RMSNorm denominator scalar (`mean_sq_val`) | `write(0x08, int(mean_sq))` |
| `0x0C` | DMA Switch — 0: Ping buffer, 1: Pong buffer selection | `write(0x0C, 0)` |
| `0x10` Bit0 | GeLU hardware IP enable | `write(0x10, 0x01)` |
| `0x10` Bit1 | Softmax IP enable | `write(0x10, 0x02)` |
| `0x10` Bit16 | NPU done flag (`w_npu_done`) — Polling target | `read(0x10) & 0x010000` |
| `0x14` | DMA stream type — 0: Token, 1: Weight | `write(0x14, 0 or 1)` |

> Register `0x04` was incorrectly used as the done flag in a previous version before being corrected to `0x10` (Code comment: `Polling bug fixed: 0x04 -> 0x10`).

---

### Core Function: `run_npu_matmul(x_vec, weight_mat, mean_sq_val, use_gelu=False)`

```python
def run_npu_matmul(
    x_vec:       np.ndarray,   # [2048] Input vector (Before RMSNorm)
    weight_mat:  np.ndarray,   # [2048, Output_Dim] Weight matrix
    mean_sq_val: float,        # RMSNorm denominator: mean(x²) value
    use_gelu:    bool = False, # FFN Gate specific GeLU hardware IP enable
) -> np.ndarray                # [Output_Dim] int16 (FPGA) or float16 (Simulation)
```

#### Simulation Path (`MMIO.SIMULATION_MODE = True`)

```python
inv_sqrt = 1.0 / sqrt(mean_sq_val + 1e-6)
x_f32    = x_vec.astype(np.float32) * inv_sqrt   # float32 upcasting essential
                                                   # (Risk of overflow in 2048D FP16 accumulation)
out = np.dot(x_f32, weight_mat.astype(np.float32))
if use_gelu:
    out = GELU(out)
return out.astype(np.float16)
```

**Reason for FP16 Upcasting**: Because accumulation across 2048 dimensions can exceed the maximum value of FP16 (65504), intermediate calculations must be done in FP32.

---

#### FPGA Execution Path — Tiling Structure

Decomposes the input (2048) and output (Output_Dim) into **32×32 tiles** for processing.

```
num_ic_tiles = 2048 // 32 = 64   (Number of input channel tiles)
num_oc_tiles = Out  // 32        (Number of output channel tiles)
total_tiles  = 64 × num_oc_tiles

Tile Sequence Example (Out=2048, total=4096):
  tile_idx 0   → oc=0, ic=0    (Output channel 0, Input channel 0)
  tile_idx 1   → oc=0, ic=1    (Output channel 0, Input channel 1)
  ...
  tile_idx 63  → oc=0, ic=63   ← Last ic: ACC read occurs
  tile_idx 64  → oc=1, ic=0    ← Start of ic: ACC_CLEAR occurs
```

---

#### FPGA Execution Path — Ping-Pong BRAM Pipeline

In each tile iteration, **calculation and DMA transfer for the next tile proceed simultaneously**.

```
[Prologue]
  Tile 0's token(32) + weight(32×32) → Ping buffer transfer (Synchronous)

[Main Loop: tile_idx = 0 → total_tiles-1]
  ┌─ 1. DMA Background Transfer (Asynchronous) ───────────────────────────────┐
  │   tile_idx Even (Ping calculating) → Next tile data → Pong Buffer   │
  │   tile_idx Odd (Pong calculating) → Next tile data → Ping Buffer   │
  │   Transfer Order: Token first (0x14=0) → Weight (0x14=1)             │
  └────────────────────────────────────────────────────────────────┘
  ┌─ 2. Kick NPU Calculation ────────────────────────────────────────────────┐
  │   write(0x00, 0x01)  ← START pulse (Automatically returns to 0)        │
  └────────────────────────────────────────────────────────────────┘
  ┌─ 3. Wait for Completion (Polling) ───────────────────────────────────────────┐
  │   while (read(0x10) & 0x010000) == 0: pass                      │
  └────────────────────────────────────────────────────────────────┘
  ┌─ 4. Receive Result (Only for the last ic tile) ──────────────────────────────┐
  │   if ic == num_ic_tiles - 1:                                    │
  │       DMA recv → result_buf → final_out[oc*32:(oc+1)*32]       │
  └────────────────────────────────────────────────────────────────┘
  5. Wait for DMA transfer to complete, then proceed to the next loop

[Special Handling]
  Upon entering ic == 0: ACC_CLEAR (write(0x00, 0x02))
  → Initialize the accumulator before starting to calculate a new output channel
```

**Result Reception Timing**: The results are received via DMA only after 64 dot product accumulations calculating 1 output channel (32 output neurons) are completed (`ic == 63`). In between, the internal accumulator (ACC) in the FPGA holds the values.

---

### Wrapper Functions

```python
def npu_matmul(x, weight, mean_sq):
    """ Q, K, V, O, Down — Standard matrix multiplication """
    return run_npu_matmul(x, weight, mean_sq, use_gelu=False)

def npu_matmul_gelu(x, W_gate, mean_sq):
    """ FFN Gate exclusive — 1-Cycle GeLU hardware IP enabled immediately after matrix multiplication """
    return run_npu_matmul(x, W_gate, mean_sq, use_gelu=True)
```

---

### `npu_softmax(logits)`

```python
def npu_softmax(logits: np.ndarray) -> np.ndarray  # float16
```

**Simulation Path**: Stable Softmax (subtract max) NumPy implementation.

**FPGA Path**:
Since Softmax is not a matrix multiplication, it is sent to the dedicated **Softmax IP** instead of going through the Systolic Array.

```python
MMIO.npu_control.write(0x10, 0x02)   # Softmax_EN bit ON

for i in range(0, len(logits), 32):   # Transferred in chunks of 32
    ping_token ← logits[i:i+32]
    DMA send → NPU kick → DMA recv → probs[i:i+32]
```

> Softmax completion polling uses a **different register** `0x04 & 0x01` compared to matrix multiplication (`0x10 & 0x010000`). This indicates that the done signals of the Softmax IP and the Systolic Array are implemented separately at the hardware level.

---

### Comparison of the Three Acceleration Modes

| Item | NPU_CORE.py | IGPU_CORE.py | CPU_MATRIX_CORE.py |
| -------------- | -------------------------- | ----------------------- | ------------------------ |
| Hardware | FPGA Systolic Array | iGPU (Vulkan) | CPU AVX2/OpenMP |
| Control Method | MMIO Register + DMA | Vulkan Command Buffer | OpenMP Multicore |
| Weight Format | float16 (Built-in FPGA conversion) | INT4 uint8 (GPU unpacking) | INT4 uint8 (SIMD unpacking) |
| Tile Unit | 32×32 fixed | Workgroup 32 | AVX2 register unit |
| RMSNorm Location | Built-in NPU IP (pass mean_sq) | CPU (Separate call) | CPU (Separate call) |
| GeLU Location | Built-in NPU IP (1-Cycle) | CPU (`CPU_CORE.gelu`) | C++ SIMD inline |
| Current Usage | ❌ (FPGA target, unconnected) | ✅ Default mode | ✅ CPU mode |

---

---

## 2. `gemma3N_E4B_architecture.md`

### Overview

A **reference design document recording the entire Forward Pass** of the Gemma 3N E4B model. It includes the operating principles of each step, the rationale for CPU/IGPU distribution, and core formulas based on the example input "Hello". It serves as the **Single Source of Truth** that acts as the standard for implementation decisions in the codebase.

> **Warning**: Some code snippets in the document reflect an **older version's structure** and may differ in detail from the current implementation in `main.py`. The differences are summarized in the "Differences between the document and the current code" section below.

---

### Document Structure

| Phase | Step | Hardware | Core Content |
| ----------- | ------------------------------- | -------- | --------------------------------------- |
| **Phase 1** | 1. Tokenization + Load Weights | CPU | Text → Integer ID |
| | 2. Embedding + AltUp Init | CPU/IGPU | ID → [4, 2048] 4-stream |
| **Phase 2** | 3. AltUp Router (Predict) | IGPU | Generates `xs_pred` based on Tanh |
| | 4. Pre-Attn RMSNorm + Q,K,V | IGPU | `inputs_normalized` → Q,K,V |
| | 5. QK-Norm + RoPE | CPU | Head-wise normalization + position encoding |
| | 6. KV Cache Routing + GQA | CPU | Cache reuse for layers 20~34, unscaled |
| | 7. W_o Proj + LAuReL + 1st Residual | IGPU | `1/√2` scaled sum |
| | 8. FFN Sparsity (Layers 0~9) | IGPU/CPU | Activates only the top 5% of neurons |
| | 9. 2nd Residual Connection | IGPU | `outputs += attn_output` |
| | 10. Inject PLE (xs[1~3]) | CPU | Injects layer position info only to shadow streams |
| **Phase 3** | 11. Final Norm + LM Head | IGPU | 4-stream → vocab logit |
| | 12. Softmax + Sampling | CPU | Repetition penalty + Top-p |

---

### Core Design Principles Summary

#### 1. AltUp 4-Stream Structure

```
xs[0]  = Main Stream  ← The only input for Attention/FFN operations. Never modified directly
xs[1]  = Shadow Stream 1  ┐
xs[2]  = Shadow Stream 2  ├─ Created by altup_projs, target for PLE injection
xs[3]  = Shadow Stream 3  ┘

Layer Start: xs → xs_pred (AltUp Predict, 4×4 coefficient matrix)
Layer End:   xs_pred + innovation × corr_coefs → xs_new (AltUp Correct)
```

The core of AltUp: **Computation relies entirely on `xs[0]`**, **Information is accumulated across all 4 streams**.

---

#### 2. KV Cache Routing Rules

```
Layers 0~19:  Writes K, V to its own slot + Looks up its own cache
Layers 20~34: No cache writing
              ├── i % 5 == 4 (Global Layers: 24,29,34) → Reuses K_cache[19]
              └── The rest    (Local Layers)              → Reuses K_cache[18]
```

**Rationale**: The deeper the layer, the more the Attention pattern solidifies. Since layers 18 (Local) and 19 (Global) possess the best-learned patterns, they are reused in layers after 20. This completely eliminates the cost of storing the KV cache for 15 layers (20~34).

---

#### 3. Unscaled GQA (Unscaled Attention)

Standard Attention:
$$ \text{Attn} = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V, \quad d_k=256 $$

Gemma 3N E4B:
$$ \text{Attn} = \text{Softmax}(QK^T)V \quad \leftarrow \text{No division by } \sqrt{d_k} $$

Passes the Raw Score directly to the Softmax without scaling. There is no `/ math.sqrt(256)` operation inside `cpu_gqa()`.

---

#### 4. Extreme FFN Sparsity (Layers 0~9)

$$ \text{cutoff} = \mu(\text{gate}) + 1.6449 \cdot \sigma(\text{gate}) $$
$$ \text{sparse\_gate} = \max(\text{gate} - \text{cutoff},\ 0) $$

A normal distribution of z=1.6449 corresponds to the top 5% cutoff. Since 95% of the neurons become exactly 0, it creates an opportunity for sparse operations during the W_down matrix multiplication.

```
Layers 0~9:   sparse gate (95% zero) → W_up → W_down  (Drastically reduced computation volume)
Layers 10~34: dense gate (Fused GeLU)  → W_up → W_down  (Prioritizes speed)
```

---

#### 5. LAuReL (Layer-wise Augmented Residual Learning)

$$ \text{laurel\_out} = x_n + \text{RMSNorm}(\text{right}(\text{left}(x_n))) $$
$$ \text{attn\_final} = (\text{attn\_output} + \text{laurel\_out}) \times \frac{1}{\sqrt{2}} $$

Executed in **parallel** with the W_o projection. It bolsters expressive power by adding a short bypass path that passes through two low-rank (64-dimensional) linear layers. The `1/√2` scaling maintains variance after summing the two paths.

---

#### 6. PLE (Per-Layer Embedding) Injection

```
Calculate PLE (Before entering the layer loop, once per token):
  x0 → W_ple_proj → reshape[35, 256] → RMSNorm(row-wise) + norm_ple   = x_proj_normed
  x0 → W_ple[token_id] → reshape[35, 256] × √256                  = y
  pli_all = (x_proj_normed + y) × (1/√2)   shape: [35, 256]

Inject PLE at Layer i:
  pli = pli_all[i]                          ← Position vector for the i-th layer
  gate_ple = GELU(activated @ W_ple_gate[i]) × pli
  mapped   = RMSNorm(gate_ple @ W_ple_proj[i], ple_post_ln[i])
  xs_new[1:] += mapped   ← xs[0] is untouched
```

It accumulates layer number information exclusively in the shadow streams without polluting the main computation path.

---

### Differences between the Document and the Current Code (`main.py`)

| Item | Document (`architecture.md`) | Current Code (`main.py`) |
| ---------------- | ------------------------------------------------- | ------------------------------------------------------------------ |
| KV Cache Data Structure | `K_cache = [[] for _ in range(35)]` (List) | `np.zeros([35, 2048, 512], float16)` (Pre-allocated array) |
| W_embed Format | `CPU_CORE.embedding(token_id, W_embed)` Single argument | `CPU_CORE.embedding(token_id, W_embed[0], W_embed[1])` (Tuple unpacking) |
| Residual after W_o | `attn_output += xs[0]` | `attn_output += x` (Uses `xs_pred[0].copy()`) |
| AltUp Correct | `xs_pred.copy()` then for loop | `xs_pred + corr_coefs[:, np.newaxis] * innovation` (Vectorized) |
| Ping-Pong Optimization | Not mentioned | `hw_prefetch`/`hw_compute_pingpong` in order Q→K→V→O→Gate→Up→Down |
| GeLU Location (Layers 0~9) | Applied separately after specifying `use_gelu=False` | Identical (`use_gelu=(i >= 10)`) |

---

### RoPE Layer Classification

| Layer Index | Condition | theta | Type |
| ------------------------------------------- | ------------ | --------- | ---------- |
| 4, 9, 14, 19, 24, 29, 34 | `i % 5 == 4` | 1,000,000 | **Global** |
| 0~3, 5~8, 10~13, 15~18, 20~23, 25~28, 30~33 | The rest | 10,000 | **Local** |

Global layers are responsible for capturing long-range context, while Local layers capture short-range patterns.

---

### Overall Forward Pass Data Flow Summary

```
token_id (int)
    │
    ▼ embedding() × √2048
x0 [2048]
    │
    ├──→ xs[0] = x0
    ├──→ xs[1..3] = x0 @ altup_projs[0..2]
    │
    ├──→ W_ple_proj → pli_all [35, 256]
    │
    ▼  ×35 Layers
┌──────────────────────────────────┐
│  AltUp Predict  → xs_pred        │
│  RMSNorm(xs[0]) → inputs_norm    │
│                                  │
│  Q,K,V = inputs_norm @ W_q,k,v   │
│  QK-Norm → RoPE                  │
│  KV Cache Routing                  │
│  GQA (Unscaled) → attn_raw       │
│                                  │
│  W_o + LAuReL + Residual 1           │
│                                  │
│  RMSNorm → W_gate (sparse/dense) │
│         → W_up                   │
│  hidden = gate × up              │
│  W_down + Residual 2                 │
│                                  │
│  AltUp Correct → xs_new          │
│  Inject PLE → xs_new[1:] += mapped │
└──────────────────────────────────┘
    │
    ▼
xs [4, 2048]
    │
    ▼ decode_logits()
    │  4-stream magnitude normalization + average
    │  Final RMSNorm + W_lm_head
    │
logits [262400]
    │
    ▼ _sample()
    │  Softcap(30) → Rep Penalty → Softmax → Top-p
    │
next_token (int)
```
