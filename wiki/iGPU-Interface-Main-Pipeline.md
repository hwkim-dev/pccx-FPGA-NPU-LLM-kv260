# Code Documentation (4/8) — iGPU Interface & Main Pipeline

> **Target Files**: `IGPU_CORE.py` · `main.py`
> **Role**: Vulkan Python binding (`IGPU_CORE`) + Orchestration of the entire inference loop (`main`)

---

## 1. `IGPU_CORE.py`

### Overview

An **iGPU acceleration interface module** that wraps `vulkan_core.so` (C++ Vulkan engine) with Python `ctypes`.
When `ACCEL_MODE = "IGPU"` is set in `main.py`, it is imported under the name `FAST_MATRIX_CORE` and provides the exact same function signatures as `CPU_MATRIX_CORE.py`.

**Module Initialization Sequence** (Automatically executed upon import):

```
1. Load ctypes.CDLL("C_DLL/vulkan_core.so")
2. os.chdir(base_dir)           ← Essential for finding .spv shader files via relative paths
3. vk_lib.init_vulkan_engine()  ← Full initialization of Vulkan instance/pipeline/buffers (Once)
4. Register argtypes for each C++ function
```

> Without `os.chdir(base_dir)`, `readFile("C_DLL/gemv_int4_vector4.spv")` in `vulkan_core.cpp`
> would search relative to the execution location and fail to find the file.

---

### C++ DLL Binding

| C++ Function | Python Wrapper | Notes |
| -------------------------- | --------------------- | ----------------------- |
| `init_vulkan_engine` | (Called directly during initialization) | Executed automatically once upon import |
| `run_vulkan_gemv` | `igpu_matmul()` | Legacy synchronous GEMV |
| `prefetch_weight_async` | `prefetch_weight()` | Asynchronous weight prefetch |
| `run_vulkan_gemv_pingpong` | `compute_pingpong()` | Ping-pong buffer specific GEMV |

**Pay attention to the argument order of `run_vulkan_gemv_pingpong`**:
```python
# C++ Signature:
#   run_vulkan_gemv_pingpong(x, scale, out, M_out, K_in, buf_idx)
# ← packed (weights) is not an argument! It is already loaded into VRAM via prefetch_weight_async()
vk_lib.run_vulkan_gemv_pingpong.argtypes = [
    float32[1D],   # x     (Input vector)
    float32[1D],   # scale (Dequant scale)
    float32[1D],   # out   (Output vector)
    c_int,         # M_out
    c_int,         # K_in
    c_int,         # buf_idx (0 or 1)
]
```

---

### Output Buffer Pool

```python
_OUTPUT_BUF_POOL: dict[int, np.ndarray] = {}

def _get_output_buf(size: int) -> np.ndarray:
    if size not in _OUTPUT_BUF_POOL:
        _OUTPUT_BUF_POOL[size] = np.empty(size, dtype=np.float32)
    return _OUTPUT_BUF_POOL[size]
```

Allocates `np.empty` arrays by output size only once and reuses them.
When returning a result, you must call `.copy()` so the pool buffer does not become contaminated.

---

### Function Reference

#### `igpu_matmul(x_vec, weight_data)`

```python
def igpu_matmul(
    x_vec:       np.ndarray,          # [K_in] any dtype → converted to float32
    weight_data: tuple | np.ndarray,  # INT4 tuple or standard float matrix
) -> np.ndarray                       # [M_out] float32
```

**INT4 Tuple Path** (`weight_data = (packed, scale)`):
```
packed.shape[0]     → M_out
packed.shape[1] × 2 → K_in (1 uint8 = 2 INT4s)
vk_lib.run_vulkan_gemv(x, packed, scale, out_buf, M_out, K_in)
```

**Standard Matrix Path** (float matrix):
```python
return np.dot(x_f32, w_f32)
# ← Note the lack of transpose compared to np.dot(x, W.T) in CPU_MATRIX_CORE
#    weight_data must already be in [K_in, M_out] layout
```

---

#### `igpu_matmul_gelu(x_vec, weight_data)`

```python
def igpu_matmul_gelu(x_vec, weight_data) -> np.ndarray
```

Applies `CPU_CORE.gelu()` separately after calling `igpu_matmul()`.

> **Difference from CPU_MATRIX_CORE**: While CPU mode fuses GEMV+GELU in a single C++ kernel via `run_gemv_int4_gelu`, the Vulkan iGPU mode executes them separately in sequence: GEMV(GPU) → GELU(CPU). There is room for further optimization in iGPU mode by integrating GELU into the shader.

---

#### `prefetch_weight(weight_data, buf_idx)`

```python
def prefetch_weight(
    weight_data: tuple,   # (packed, scale) — no-op if not a tuple
    buf_idx:     int,     # 0 or 1 (Target ping-pong buffer)
)
```

Calls the C++ `prefetch_weight_async()` to asynchronously copy the weights to VRAM in a background thread using `std::async`. It returns immediately, and completion of the copy is guaranteed by `weight_loader.wait()` during the next `compute_pingpong()` call.

If it's not a tuple (standard float matrix), no VRAM upload is necessary, so it performs no action.

---

#### `compute_pingpong(x_vec, weight_data, buf_idx)`

```python
def compute_pingpong(
    x_vec:       np.ndarray,  # [K_in] float32
    weight_data: tuple,       # (packed, scale)
    buf_idx:     int,         # Buffer index to use
) -> np.ndarray               # [M_out] float32
```

Executes a GPU GEMV using the weights in the ping-pong buffer indicated by `buf_idx`.
The weights for this buffer must have already been loaded into VRAM by a previous `prefetch_weight(w, buf_idx)` call.

**Standard Matrix Fallback**: If not a tuple, calculates on the CPU via `np.dot(x, W.T)`.

---

#### Legacy Functions

```python
def preload_and_free(W, keys): pass   # Legacy VRAM pre-upload from previous Taichi-based version
def _get_or_upload_weight(w):  pass   # Same as above
def warmup(): print("...")            # Only prints shader load completion message
```

These are functions that existed for VRAM memory management in the older Taichi version. In the current Vulkan ping-pong architecture, weights are dynamically transferred at the time of the call, so they are unnecessary. They are kept as empty functions for interface compatibility with `main.py`.

---

---

## 2. `main.py`

### Overview

The **entry point and orchestrator** for the entire Gemma 3N E4B inference pipeline.
It controls the overall flow of Tokenization → Embedding → 35 Transformer layer iterations → Logit Decoding → Sampling.

---

### Module-Level Configuration

```python
ACCEL_MODE = "IGPU"          # Change to "CPU" to switch to CPU_MATRIX_CORE
NUM_LAYERS = 35

_IGPU_WEIGHT_KEYS = [        # List of keys targeted for ping-pong prefetch (currently passed to the no-op function)
    "W_q", "W_k", "W_v", "W_o", "W_gate", "W_up", "W_down"
]
```

Upon loading the module, it additionally registers the argtypes of two functions from `my_accelerator.so`: `run_RMSNorm_inplace` and `run_softmax_inplace`. (This structure **redundantly loads** the same DLL as `CPU_CORE.py`.)

---

### Utility Functions

#### `rms_norm(x, gamma)`

```python
def rms_norm(x: np.ndarray, gamma: np.ndarray) -> np.ndarray
```

A wrapper exclusive to main.py that calls C++ `run_RMSNorm_inplace`.
This feature is absent in `CPU_CORE` and is only used within `main.py`.

```python
x_f32   = np.ascontiguousarray(x.astype(np.float32))     # Create an independent copy
gamma_c = np.ascontiguousarray(gamma.astype(np.float32))
c_lib.run_RMSNorm_inplace(x_f32, gamma_c, x_f32.size)    # Overwrite in-place
return x_f32
```

> Since it creates an independent copy via `np.ascontiguousarray`, the original `x` remains unmodified.

---

#### `get_router_modalities(x, w_norm, w_router)`

```python
def get_router_modalities(x, w_norm, w_router) -> np.ndarray  # [4]
```

Calculates the modality vector of the AltUp router.
Normalizes `xs[0]`, projects it using the router weights, and compresses it with Tanh.

```python
x_n = rms_norm(x, w_norm) / 2048.0      # Dimension scale correction
return np.tanh(np.dot(x_n, w_router))   # shape: [4]
```

It is called **twice**: at the beginning of the layer (AltUp Predict) and at the end of the layer (AltUp Correct).

---

#### `hw_matmul(x, w, use_gelu=False)` / `hw_prefetch(w, buf_idx)` / `hw_compute_pingpong(x, w, buf_idx, use_gelu=False)`

**Three hardware adapter functions** that abstract ping-pong optimization and mode switching.

| Function | IGPU Mode | CPU Mode |
| --------------------- | ------------------------------------- | ------------------------------ |
| `hw_matmul` | `FAST_MATRIX_CORE.igpu_matmul[_gelu]` | Inline INT4 dequant + `np.dot` |
| `hw_prefetch` | `FAST_MATRIX_CORE.prefetch_weight` | no-op |
| `hw_compute_pingpong` | `FAST_MATRIX_CORE.compute_pingpong` | `hw_matmul` fallback |

CPU mode inline dequant for `hw_matmul`:
```python
# If it's a tuple, manually convert INT4 → float32
low  = (packed & 0x0F).astype(np.int8); low[low > 7]   -= 16
high = (packed >> 4  ).astype(np.int8); high[high > 7] -= 16
w_real = interleave(low, high) * scale[:, np.newaxis]
out = np.dot(x, w_real.T)
```

---

### Core Function: `forward_one_token()`

```python
def forward_one_token(
    token_id:     int,
    pos:          int,            # Current sequence position (0-indexed)
    W:            dict,           # Dictionary of weights for the 35 layers
    W_embed:      tuple,          # (packed, scale) mmap
    W_ple_packed: np.ndarray,     # [262144, 4480] uint8 mmap
    W_ple_scale:  np.ndarray,     # [262144] float32 mmap
    norm_ple:     np.ndarray,     # [256] float32
    W_ple_proj:   tuple,          # (packed, scale) INT4
    altup_projs:  list[np.ndarray],  # [3] × [2048, 2048]
    K_cache:      np.ndarray,     # [35, max_seq, 512] float16 (pre-alloc)
    V_cache:      np.ndarray,     # [35, max_seq, 512] float16 (pre-alloc)
) -> np.ndarray                   # xs: [4, 2048] float32 (4-stream output)
```

Performs **Embedding → PLE calculation → 35 layer iterations** for a single token.

#### Phase A: Embedding and AltUp Initialization

```python
# 1. Look up INT4 embedding + Gemma 3N scaling
x0 = CPU_CORE.embedding(token_id, W_embed[0], W_embed[1])
x0 = x0 * sqrt(2048.0)               # Gemma 3N specific embedding scaling

# 2. Initialize AltUp 4-Stream
xs[0] = x0                            # Main stream (Absolutely no modifications)
xs[1..3] = dot(x0, altup_projs[0..2]) # Shadow Streams
```

#### Phase B: Calculate PLE (Per-Layer Embedding)

```python
# W_ple_proj: [2048] → [35×256] projection (IGPU)
x_proj = hw_matmul(x0, W_ple_proj) / sqrt(2048.0)
x_proj = x_proj.reshape(35, 256)
x_proj_normed = RMSNorm_perrow(x_proj) * norm_ple   # Row-wise RMSNorm

# W_ple: [vocab, 8960] → Look up corresponding token row → [35, 256]
y = embedding(token_id, W_ple_packed, W_ple_scale).reshape(35, 256) * sqrt(256.0)

# Final PLE vector (Layer-wise position embedding)
pli_all = (x_proj_normed + y) * (1/sqrt(2.0))   # shape: [35, 256]
```

#### Phase C: 35-Layer Iteration Loop

**Layer Start: AltUp Predict**
```python
modalities = get_router_modalities(xs[0], W["altup_rn"][i], W["altup_router"][i])
coef_mat   = dot(W["altup_pred"][i], modalities).reshape(4, 4)  # [4, 4]
xs_pred    = xs + dot(coef_mat, xs)   # Predicted stream (temporary lens)
x          = xs_pred[0].copy()        # Attention uses pure xs_pred[0]
```

**Attention Block (Ping-Pong Order)**

```
prefetch(W_q[0], buf=0)   ← Pre-load before entering the loop

[Start of Layer i]
buf=0: Calculate W_q  │  Asynchronous: W_k → buf=1
buf=1: Calculate W_k  │  Asynchronous: W_v → buf=0
buf=0: Calculate W_v  │  Asynchronous: W_o → buf=1
       QK-Norm, RoPE, KV Cache, GQA
buf=1: Calculate W_o  │  Asynchronous: W_gate → buf=0
       Calculate LAuReL
```

**KV Cache Routing Rules**:
```python
if i < 20:
    K_cache[i, pos, :] = K      # Layers 0~19: Save to its own slot
    V_cache[i, pos, :] = V
    target_k = K_cache[i, :pos+1, :]
else:
    if i % 5 == 4:              # Global Layers (20,25,30,34): Reuse layer 19 cache
        target_k = K_cache[19, :pos+1, :]
    else:                       # Local Layers (21~24, 26~29, ...): Reuse layer 18 cache
        target_k = K_cache[18, :pos+1, :]
```

**1st Residual Connection + LAuReL**:
```python
attn_output = RMSNorm(W_o_out, post_attn_ln)
attn_output += x                             # Residual connection
# LAuReL: inputs_normalized → left → right → norm → + inputs_normalized
laurel_out_normed = inputs_normalized + RMSNorm(right(left(inputs_normalized)))
attn_output = (attn_output + laurel_out_normed) * (1/sqrt(2.0))  # Sum scaled
```

**FFN Block (Ping-Pong Order + Sparsity)**:
```
buf=0: Calculate W_gate (i≥10: Fused GELU)  │  Asynchronous: W_up → buf=1
buf=1: Calculate W_up                       │  Asynchronous: W_down → buf=0

if i < 10:   # Layers 0~9 5% sparsity surgery
    cutoff = mean(gate_out) + std(gate_out) * 1.6448536   # z=1.645 → top 5%
    sparse_gate = max(gate_out - cutoff, 0)
    hidden = gelu(sparse_gate) * up_out
else:        # Layers 10~34 dense
    hidden = gate_out * up_out

buf=0: Calculate W_down  │  Asynchronous: W_q[i+1] → buf=1 (Pre-load for next layer)
```

**2nd Residual Connection**:
```python
outputs = RMSNorm(mlp_out, post_ffn_ln) + attn_output
```

**Layer End: AltUp Correct + Inject PLE**:
```python
activated  = outputs * W["altup_scale"][i]
innovation = activated - xs_pred[0]

mod_corr   = get_router_modalities(activated, ...)
corr_coefs = dot(W["altup_corr"][i], mod_corr) + 1.0   # [4]

xs_new = xs_pred + corr_coefs[:, np.newaxis] * innovation   # Calibrate the 4 streams

# Inject PLE (Do not touch xs[0]!)
gate_ple = gelu(hw_matmul(activated, W["ple_gate"][i])) * pli_all[i]
mapped   = RMSNorm(hw_matmul(gate_ple, W["ple_proj"][i]), W["ple_post_ln"][i])
xs_new[1:] += mapped    # Inject into shadow streams 1, 2, and 3 only

xs = xs_new
```

---

### Core Function: `decode_logits(xs, altup_unprojs, W_final_norm, W_lm_head)`

```python
def decode_logits(...) -> np.ndarray  # [vocab_size=262400] float32
```

Transforms the 4-stream `xs`, the output of the 35 layers, into a single logit vector.

```python
# 1. Prefetch W_lm_head (Since LM Head is large, load asynchronously during CPU calculation)
hw_prefetch(W_lm_head, buf_idx=0)

# 2. Normalize magnitude of 4 streams and sum the average
target_mag = mean(xs[0]**2)**0.5
for k in 1..3:
    proj_x = dot(xs[k+1], altup_unprojs[k])
    proj_x *= target_mag / max(mean(proj_x**2)**0.5, 1e-12)  # Match magnitude
x_final = mean(stack([xs[0], proj_0, proj_1, proj_2]), axis=0)  # [2048]

# 3. Final RMSNorm + LM Head (Uses ping-pong buffer 0)
x_final = RMSNorm(x_final, W_final_norm)
logits  = hw_compute_pingpong(x_final, W_lm_head, buf_idx=0)   # [262400]
```

---

### Core Function: `_sample(logits, temperature, top_p, rep_penalty, generated)`

```python
def _sample(...) -> int  # Next token ID
```

**Execution Order**:

1. **Repetition Penalty**: Attenuate the logits of already generated tokens by `rep_penalty`(1.15)
   ```python
   logits[token] /= rep_penalty  if logits[token] > 0
   logits[token] *= rep_penalty  if logits[token] < 0
   ```

2. **Softcap**: `logits = 30.0 * tanh(logits / 30.0)` — *(Applied in `main()` before calling `_sample`)*

3. **Softmax + Temperature**: C++ `run_softmax_inplace(logits, size, temperature)`

4. **Top-p (Nucleus) Sampling**:
   ```python
   sorted_idx  = argsort(probs)[::-1]             # Sort in descending order
   cumsum       = cumsum(probs[sorted_idx])
   cutoff_mask  = cumsum - probs[sorted_idx] < top_p  # Retain only up to cumulative probability top_p
   probs_filtered[sorted_idx[cutoff_mask]] = probs[...]
   ```

5. **Token Sampling**: `np.random.choice(vocab_size, p=probs_filtered)`

> **Unimplemented Optimization**: Replacing `np.argsort` (O(n log n)) with `np.argpartition` (O(n)) in the Top-p stage could yield a significant speedup for vocab_size=262,400.

---

### `main()` — Overall Execution Flow

```
[Initialization]
  warmup()                        ← Warm up hardware
  load_local_weights()            ← Load mmap-based weights
  preload_and_free() (no-op)
  K_cache = zeros([35, 2048, 512], float16)   ← Pre-allocate KV cache
  V_cache = zeros([35, 2048, 512], float16)
  cur_pos = 0                     ← Global sequence position (Maintained across conversations)

[Conversation Loop]  while True:
  user_input = input()

  [Prefill]  for token in input_tokens:
    xs = forward_one_token(token, cur_pos, ...)
    cur_pos += 1

  [Generation]  for _ in range(MAX_NEW_TOKENS):
    logits     = decode_logits(xs, ...)
    logits     = 30 * tanh(logits / 30)        ← Softcap
    next_token = _sample(logits, ...)
    if next_token in [1, 106]: break            ← Detect EOS token

    current_text = tokenizer.decode(generated) ← Full re-decode (Prevents UTF-8 truncation in some languages)
    print(current_text[len(printed_text):])     ← Incremental output
    printed_text = current_text

    xs = forward_one_token(next_token, cur_pos, ...)
    cur_pos += 1

  gc.collect()   ← Free memory after turn ends
```

**Design Considerations**:

| Item | Value / Description |
| ---------------- | ---------------------------------------------------------------------------- |
| `cur_pos` Init | Initialized to 0 **outside** the conversation loop → KV cache continues to accumulate during multi-turn chats |
| KV Cache Limit | `MAX_NEW_TOKENS = 2048` — If exceeded, `cur_pos` goes out of bounds of the array |
| `history_tokens` | Declared but **not actually used** (Multi-turn history is unimplemented) |
| Stop Tokens | `[1, 106]` — ID 1: `<eos>`, ID 106: Gemma turn end token |
| Output Method | Full re-decode with `tokenizer.decode(generated)` then differential output to prevent UTF-8 truncation |

---

### Module Dependency Graph

```
main.py
  ├── ACCEL_MODE="IGPU" → IGPU_CORE.py
  │                           └── C_DLL/vulkan_core.so
  │                                 └── C_DLL/gemv_int4_vector4.spv (Executed on GPU)
  ├── CPU_CORE.py
  │     └── C_DLL/my_accelerator.so
  ├── safeTensor.py   (Loads weight mmap)
  └── C_DLL/my_accelerator.so  (RMSNorm, Softmax — Registered natively by main.py)
```
