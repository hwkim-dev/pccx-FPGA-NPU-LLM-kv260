# Code Documentation (6/8) — Quantization & KV Cache Memory Management

> **Target Files**: `quantize.py` · `Memory_Manager.py`
> **Role**: Converts original model to INT4 quantization (`quantize`) + Pre-allocates KV cache utility (`Memory_Manager`)

---

## Location of the Two Files

From the perspective of the entire pipeline, both files are **one-time preparation tools**.

```
[Pre-preparation Stage]
  quantize.py        ← Original float model → INT4 safetensors (First time only)
  Memory_Manager.py  ← Tool for designing and verifying the size of the KV cache array

[Actual Inference]
  main.py            ← Consumes the outputs of the two files above (converted weights, cache design)
```

---

## 1. `quantize.py`

### Overview

A **script executed exactly once** to convert the large weights of the original Gemma 3N E4B model (float16/bfloat16) to **INT4 (4-bit) symmetric quantization** and save the conversion results in SafeTensors format.

**Execution**: `python quantize.py`

**Input/Output**:
```
Input: ORIGINAL_MODEL_DIR/local_gemma_3n/*.safetensors    (Original float model)
Output: SAVE_DIR/local_gemma_3n_int4/*.safetensors         (INT4 converted model)
```

---

### Module-Level Configuration

```python
ORIGINAL_MODEL_DIR = "/home/hwkim/.../local_gemma_3n"     # Original model path (hardcoded absolute path)
SAVE_DIR           = BASE_DIR + "/local_gemma_3n_int4"    # Output path
```

> `ORIGINAL_MODEL_DIR` is hardcoded as an absolute path.
> You must manually modify this value when executing in a different environment.

---

### List of Weights Targeted for Quantization (`_BIG_WEIGHT_SUFFIXES`)

Only 2D weights ending with the suffixes below are converted to INT4. The rest retain their original dtype.

| Suffix | Corresponding Weight | Size Before Conversion (Per Layer) | Size After Conversion |
| ----------------------------------- | ------------ | ----------------------- | ----------------------------------- |
| `q_proj.weight` | W_q | 2048×2048 f16 = 8MB | 2048×1024 u8 + 2048 f32 = ~2MB |
| `k_proj.weight` | W_k | 512×2048 f16 = 2MB | 512×1024 u8 + 512 f32 = ~0.5MB |
| `v_proj.weight` | W_v | 512×2048 f16 = 2MB | ~0.5MB |
| `o_proj.weight` | W_o | 2048×2048 f16 = 8MB | ~2MB |
| `gate_proj.weight` | W_gate | 16384×2048 f16 = 64MB | 16384×1024 u8 + 16384 f32 = ~16.3MB |
| `up_proj.weight` | W_up | 16384×2048 f16 = 64MB | ~16.3MB |
| `down_proj.weight` | W_down | 2048×16384 f16 = 64MB | ~16.3MB |
| `embed_tokens.weight` | W_embed | 262400×2048 f16 = 1.0GB | 262400×1024 u8 = ~257MB |
| `embed_tokens_per_layer.weight` | W_ple | 262144×8960 f16 = 4.4GB | ~1.1GB |
| `per_layer_input_gate.weight` | ple_gate | 256×2048 f16 = 1MB | ~0.27MB |
| `per_layer_model_projection.weight` | W_ple_proj | 8960×2048 f16 = 35MB | ~8.8MB |
| `laurel.linear_left.weight` | laurel_left | 64×2048 f16 = 0.25MB | ~0.065MB |
| `laurel.linear_right.weight` | laurel_right | 2048×64 f16 = 0.25MB | ~0.065MB |

**Weights not quantized**: All LayerNorm (gamma), altup coefficients, 1D weights, `ple_proj` (currently kept as float32 — subject to review for future INT4 conversion)

---

### Core Function: `quantize_to_int4(weight)`

```python
def quantize_to_int4(
    weight: np.ndarray    # [N, M] float16 or float32
) -> tuple[np.ndarray, np.ndarray]:
    # Returns: (packed [N, M//2] uint8, scale [N] float32)
```

Performs **Per-Row symmetric quantization**. Row = 1 output neuron.

#### Formula

$$ \text{scale}[i] = \frac{\max(|w_i|)}{7.0} $$

$$ w_q[i,j] = \text{clip}\!\left(\text{round}\!\left(\frac{w[i,j]}{\max(|w_i|)} \times 7.0\right),\ -8,\ 7\right) $$

$$ \text{packed}[i, j//2] = (w_q[i, 2j] \mathbin{\&} \texttt{0x0F})\ |\ ((w_q[i, 2j+1] \mathbin{\&} \texttt{0x0F}) \ll 4) $$

#### Step-by-Step Implementation

**Step 1: Upcasting to float32**
```python
w_f32 = weight.astype(np.float32)
# Errors can occur during max calculation with float16 precision → upcast to float32
```

**Step 2: Calculating scale per row**
```python
max_vals = np.max(np.abs(w_f32), axis=1, keepdims=True)  # [N, 1]
max_vals = np.maximum(max_vals, 1e-8)                     # Prevent division by 0
scale    = (max_vals / 7.0).flatten()                     # [N] — used during dequant
```

**Step 3: Normalization and Rounding**
```python
w_q = np.round(w_f32 / max_vals * 7.0).astype(np.int8)
w_q = np.clip(w_q, -8, 7)
# Range: [-8, 7] — Utilizes the entire range of signed 4-bit integers
# -8 is representable but introduces a slight asymmetric error during dequant
```

**Step 4: Packing 2 per uint8**
```python
w_q_low  = w_q[:, 0::2] & 0x0F   # Even columns → lower 4 bits
w_q_high = w_q[:, 1::2] & 0x0F   # Odd columns → upper 4 bits
packed   = (w_q_low | (w_q_high << 4)).astype(np.uint8)
# [N, M] int8 → [N, M//2] uint8  (Saves 50% memory)
```

**Packing Layout Visualization**:
```
Original  w_q[i]: [ a, b, c, d, e, f, ... ]  (int8, M items)
                └─┬─┘  └─┬─┘
packed[i]:    [ a|b<<4, c|d<<4, ... ]    (uint8, M/2 items)

Example: a=-3 (0b1101), b=5 (0b0101)
  a & 0x0F = 0x0D (lower 4 bits)
  b & 0x0F = 0x05 (upper 4 bits)
  packed   = 0x0D | (0x05 << 4) = 0x5D
```

---

### `main()` Execution Flow

```python
def main():
    for filename in sorted(glob("local_gemma_3n/*.safetensors")):
        tensors = load_file(filename)           # safetensors → torch tensor dict
        quantized_tensors = {}

        for name, tensor in tensors.items():
            is_big = any(name.endswith(s) for s in _BIG_WEIGHT_SUFFIXES)

            if is_big and len(tensor.shape) == 2:   # Quantize only 2D large weights
                weight_np = tensor.to(torch.float32).numpy()
                packed, scale = quantize_to_int4(weight_np)

                quantized_tensors[name]            = torch.from_numpy(packed)  # uint8
                quantized_tensors[name + ".scale"] = torch.from_numpy(scale)   # float32

            else:                                   # 1D, small, non-targets → keep original
                quantized_tensors[name] = tensor

        save_file(quantized_tensors, SAVE_DIR + "/" + basename(filename))

        del tensors, quantized_tensors
        gc.collect()   # Release memory per file (Prevents the entire model from residing in RAM simultaneously)
```

**Reason for File-by-File Processing**: Processes SafeTensors files one by one and immediately calls `del` + `gc.collect()`, preventing the entire original model (~9GB) from being loaded into RAM at the same time.

---

### Quantization Error Characteristics

| Item | Description |
| -------------- | ------------------------------------------------------------ |
| Method | Symmetric quantization — Centered at 0, no offset |
| Expression Range | [-8, 7] — Theoretically [-7.5, 7.5], but forced to [-8, 7] via clip |
| Scale Unit | 1 per row (Output neuron) (Per-Row) |
| Theoretical Max Error | `scale × 0.5` (Rounding error) |
| Asymmetry | -8 is represented but +8 is clipped → Slight asymmetry exists in the negative direction |
| Precision Loss | float16 → INT4: About 75% bit reduction, maintains practical quality |

---

---

## 2. `Memory_Manager.py`

### Overview

A **utility module** that pre-allocates the KV cache array as a single contiguous NumPy array. Since this task is currently performed directly with `np.zeros()` in `main.py`, this module itself is not called during inference. It serves as reference code to design and verify the cache memory layout.

---

### `allocate_KVcache(layers, token, dimension)`

```python
def allocate_KVcache(
    layers:    int,   # Number of layers (35)
    token:     int,   # Maximum sequence length (2048)
    dimension: int,   # KV head dimension (512 = 2 KV heads × 256)
) -> np.ndarray       # [layers, token, dimension] float16
```

```python
A = np.zeros((layers, token, dimension), dtype=np.float16)
return A
```

---

### KV Cache Design Rationale

**Why float16?**
While restored to float32 (`K_cache.astype(np.float32)`) during attention calculation, it is stored as float16. The impact of precision loss on attention quality is negligible, but memory is halved.

**Dimension Configuration (512)**:
```
Number of KV heads: 2 (GQA structure)
Head dimension:  256
Total:       2 × 256 = 512
```

**Memory Calculation When Called With Default Values**:
```
allocate_KVcache(35, 2048, 512)
  = 35 × 2048 × 512 × 2 bytes (float16)
  = 73,400,320 bytes
  ≈ 70 MB (K + V respectively)
  K_cache + V_cache Total ≈ 140 MB
```

---

### Actual Allocation in `main.py`

Without importing `Memory_Manager.py`, it allocates directly in `main.py` with the same structure:

```python
# main.py
K_cache = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)
V_cache = np.zeros((NUM_LAYERS, MAX_NEW_TOKENS, KV_CACHE_DIM), dtype=np.float16)
# = np.zeros((35, 2048, 512), dtype=np.float16) — Same design as Memory_Manager
```

**Indexing Method**:
```python
# Writing (Layers 0~19)
K_cache[layer_idx, pos, :] = K.astype(np.float16)   # In-place slice writing

# Reading
target_k = K_cache[layer_idx, :pos+1, :]             # Sequence slice accumulated so far
```

Compared to the previous dynamic growth method based on `np.concatenate`, the pre-allocation method replaces the O(N) reallocation that occurred with every token generation with an O(1) in-place write.

---

### Current Status and Future Directions

| Item | Current Status | Notes |
| -------------------------------- | ----------------------------------------- | -------------------------------- |
| `allocate_KVcache()` Usage | Unused in `main.py` | `main.py` calls `np.zeros` directly |
| `if __name__ == "__main__"` Block | For testing (`shape` print only) | Confirms `(35, 2048, 512)` |
| Layers 20~34 Slots | Allocated but not written to | Reuse 18/19 via KV routing |
| Potential OOM Risk | Index out of bounds if `cur_pos` exceeds 2048 | Needs `MAX_NEW_TOKENS` guard |

**Direction for Future Utilization**: When introducing KV cache initialization or sliding window methods in multi-turn conversations, it would be appropriate to extend this module to encapsulate the cache reset/compression logic.

---

### Position of the Two Files in the Entire Pipeline

```
[Step 1]  quantize.py
          Original float model (9GB+)
              ↓  Per-row symmetric INT4 quantization
          local_gemma_3n_int4/*.safetensors
          (Stored as packed uint8 + scale float32 pairs)

[Step 2]  Optim_tensor_load.py
          local_gemma_3n_int4/*.safetensors
              ↓  Decompose .npy by tensor + transpose processing
          mmap_weights/*.npy

[Step 3]  safeTensor.py  (Every execution)
          mmap_weights/*.npy → mmap virtual mapping
              ↓
          load_local_weights() return

[Step 4]  main.py  (Every execution)
          Based on Memory_Manager.py design
          K_cache, V_cache = np.zeros([35, 2048, 512], float16)
              ↓
          forward_one_token() → decode_logits() → _sample()
```
