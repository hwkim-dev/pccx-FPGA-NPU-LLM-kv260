# Code Documentation (3/8) — CPU Calculation Layer

> **Target Files**: `CPU_CORE.py` · `CPU_MATRIX_CORE.py`
> **Role**: CPU-dedicated operations like tokenizer, attention, RoPE (`CPU_CORE`) + CPU mode INT4 GEMV interface (`CPU_MATRIX_CORE`)

---

## 1. `CPU_CORE.py`

### Overview

A module that gathers **all operations dedicated to the CPU** in the model inference pipeline.
It directly links `my_accelerator.so` (C++ DLL) using `ctypes` to delegate performance-critical kernels (GELU, RMSNorm, RoPE, INT4 unpacking, Softmax) to C++ SIMD, while the Python level handles array preparation and shape transformation.

**Module Initialization Sequence** (Automatically executed upon import):

```
1. Load AutoTokenizer  ← local_gemma_3n_int4/ directory
2. Load ctypes.CDLL("C_DLL/my_accelerator.so")
3. Register argtypes / restype for each C++ function
```

---

### C++ DLL Binding

| C++ Function | Python Wrapper | Argument Types |
| ------------------------- | ------------- | --------------------------------------------------- |
| `run_gelu_inplace` | `gelu()` | `float32[1D]`, `c_int` |
| `run_unpack_int4_inplace` | `embedding()` | `uint8[1D]`, `c_float`, `float32[1D]`, `c_int` |
| `run_rope_inplace` | `cpu_rope()` | `float32[1D]`, `c_int`, `c_float`, `c_int`, `c_int` |

> There is a bug in the code where `run_gelu_inplace.restype = None` is set **twice**
> (`run_unpack_int4_inplace.restype` is unset). It has no effect on functionality but explicit correction is recommended.

---

### Function Reference

#### `tokenize(text)`

```python
def tokenize(text: str) -> np.ndarray  # shape: [T], dtype: int64
```

Converts a string to an array of token IDs using the HuggingFace `AutoTokenizer`.
Outputs the token IDs (`print`) for debugging purposes.

```python
tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
```

---

#### `embedding(token_id, W_packed, W_scale)`

```python
def embedding(
    token_id: int,
    W_packed: np.ndarray,   # [vocab_size, hidden//2], dtype=uint8  (mmap)
    W_scale:  np.ndarray,   # [vocab_size],            dtype=float32 (mmap)
) -> np.ndarray             # [hidden], dtype=float32
```

Extracts one row of a token from the INT4 embedding table and converts it to float32.

**Execution Flow**:
```
1. W_packed[token_id]  → row_packed (uint8, 1D, contiguous guaranteed)
2. W_scale[token_id]   → row_scale  (scalar float)
3. np.empty(hidden)    → out_f32    (Pre-allocate output buffer)
4. c_lib.run_unpack_int4_inplace(row_packed, scale, out_f32, packed_len)
   └── Nibble separation + sign restoration + scale multiplication performed in C++
5. return out_f32       # [hidden] float32
```

**Memory Optimization**: Since `W_packed` and `W_scale` are arrays opened via mmap, actual disk reading occurs only for that specific row (`token_id`), which is about ~5.5 KB.

---

#### `gelu(x)`

```python
def gelu(x: np.ndarray) -> np.ndarray  # Returns the same shape as input
```

A wrapper calling C++ `run_gelu_inplace`.

```python
x_flat = np.ascontiguousarray(x.flatten().astype(np.float32))
c_lib.run_gelu_inplace(x_flat, x_flat.size)
return x_flat.reshape(x.shape)
```

Handles inputs of arbitrary shapes via `flatten()` + `reshape()`.
Works transparently for both 1D and 2D.

---

#### `cpu_qk_norm(Q, K, gamma_q, gamma_k)`

```python
def cpu_qk_norm(
    Q: np.ndarray,       # [num_heads × 256] flat
    K: np.ndarray,       # [num_heads × 256] flat
    gamma_q: np.ndarray, # [256]
    gamma_k: np.ndarray, # [256]
) -> tuple[np.ndarray, np.ndarray]  # Q_norm, K_norm (flat)
```

Applies **per-head RMSNorm** to Q and K separately.
It's a QK-Norm technique that prevents attention score explosions (a feature unique to Gemma 3N).

**Formula**:
$$ Q_{norm}[h] = \frac{Q[h]}{\sqrt{\text{mean}(Q[h]^2) + \varepsilon}} \cdot \gamma_q $$

```python
Q_reshaped = Q.reshape(-1, 256)   # [num_heads, 256]
q_rms = np.sqrt(np.mean(Q_reshaped**2, axis=1, keepdims=True) + 1e-6)
Q_norm = (Q_reshaped / q_rms) * gamma_q
return Q_norm.flatten(), K_norm.flatten()
```

> The operation is performed after forcibly casting to float32.

---

#### `cpu_rope(x, pos, theta_base)`

```python
def cpu_rope(
    x:          np.ndarray,  # [num_heads × 256] flat, float32
    pos:        int,          # Current sequence position
    theta_base: float,        # Local=10,000 / Global=1,000,000
) -> np.ndarray               # [num_heads × 256] flat, float32 (in-place result)
```

A wrapper calling C++ `run_rope_inplace`.

```python
x_flat = np.ascontiguousarray(x.astype(np.float32).flatten())
c_lib.run_rope_inplace(x_flat, int(pos), float(theta_base), num_heads, 256)
return x_flat
```

**RoPE Frequency**: `cos_vals/sin_vals` caching inside C++ (Calculated once per head).
`theta_base` is determined in `main.py` according to the layer index:

```python
theta = 1_000_000.0 if (i % 5 == 4) else 10_000.0   # Global / Local
```

---

#### `cpu_gqa(Q_rope, K_cache_layer, V_cache_layer)`

```python
def cpu_gqa(
    Q_rope:          np.ndarray,  # [num_q_heads × 256] flat  (= 2×4×256 = 2048)
    K_cache_layer:   np.ndarray,  # [seq_len, 512] float16
    V_cache_layer:   np.ndarray,  # [seq_len, 512] float16
) -> np.ndarray                   # [2048] float32 (flat)
```

**Grouped Query Attention** (GQA) implementation.
Attention head configuration of Gemma 3N E4B:

| Item | Value |
| ----------- | ---------------------------- |
| Number of Q heads | 8 (= 2 groups × 4 heads) |
| Number of K/V heads | 2 (GQA: 4 Qs share 1 KV) |
| Head Dimension | 256 |

> **Crucial**: No scaling (`/ sqrt(256)`) — Gemma 3N's **Unscaled Attention** design.

**Execution Flow**:

```python
Q = Q_rope.reshape(2, 4, 256)          # [kv_heads, q_per_kv, head_dim]
K = K_cache.reshape(-1, 2, 256)        # [seq, kv_heads, head_dim]
V = V_cache.reshape(-1, 2, 256)

K_t = K.transpose(1, 2, 0)            # [kv_heads, head_dim, seq]
scores = Q @ K_t                       # [kv_heads, q_per_kv, seq]

# Stable softmax (subtract max)
scores -= scores.max(axis=-1, keepdims=True)
probs = exp(scores) / sum(exp(scores))

V_t = V.transpose(1, 0, 2)            # [kv_heads, seq, head_dim]
out = probs @ V_t                      # [kv_heads, q_per_kv, head_dim]
return out.flatten()                   # [2048]
```

---

#### `cpu_update_kv_cache()` *(Currently disabled)*

```python
def cpu_update_kv_cache(K_rope, V, token_cnt, layer_idx, K_cache, V_cache)
```

The function body is completely commented out. KV cache updating is handled directly inside `forward_one_token()` in `main.py`:

```python
# Inside main.py (Actual cache update location)
if i < 20:
    K_cache[i, pos, :] = K   # Write directly to pre-allocated [35, max_seq, 512] float16 array
    V_cache[i, pos, :] = V
```

---

#### `_get_rope_freqs(theta_base, dim=256)` *(Internal Utility)*

```python
_rope_freq_cache: dict = {}   # Module-level cache

def _get_rope_freqs(theta_base: float, dim: int = 256) -> np.ndarray
```

Calculates the RoPE frequency table and caches it in a module-level dictionary.
Currently, it is not directly called at the Python level because the C++ version of `run_rope_inplace` internally performs the same caching. (Auxiliary function of the legacy `cpu_rope` Python version)

---

### Dependencies and Initialization

```
When importing CPU_CORE.py:
  ├── transformers.AutoTokenizer  (HuggingFace)
  ├── ctypes.CDLL("C_DLL/my_accelerator.so")
  └── tokenizer = AutoTokenizer.from_pretrained("local_gemma_3n_int4/")
```

> The `tokenizer` is maintained as a module global variable and is also accessed directly as `CPU_CORE.tokenizer.decode()` from `main.py`.

---

---

## 2. `CPU_MATRIX_CORE.py`

### Overview

A **CPU-exclusive matrix multiplication module** that replaces `IGPU_CORE.py` when `ACCEL_MODE = "CPU"` is set.
It directly calls the `run_gemv_int4` / `run_gemv_int4_gelu` C++ kernels of `my_accelerator.so` to perform INT4 GEMV with OpenMP multicore + AVX2 SIMD.

Since it provides an **exactly identical interface** (`igpu_matmul`, `igpu_matmul_gelu`, `preload_and_free`, `warmup`) as `IGPU_CORE.py`, you can switch between CPU/GPU by merely changing the single `ACCEL_MODE` variable in `main.py`.

```python
# main.py
if ACCEL_MODE == "IGPU":
    import IGPU_CORE as FAST_MATRIX_CORE
elif ACCEL_MODE == "CPU":
    import CPU_MATRIX_CORE as FAST_MATRIX_CORE
# Subsequent code is called identically in the form of FAST_MATRIX_CORE.igpu_matmul()
```

---

### C++ DLL Binding

```python
c_lib.run_gemv_int4.argtypes = [
    ndpointer(float32, 1D),   # vec   [K_in]
    ndpointer(uint8,   2D),   # mat_p [M_out, K_in/2]
    ndpointer(float32, 1D),   # scale [M_out]
    ndpointer(float32, 1D),   # out   [M_out]
    c_int,                    # M_out
    c_int,                    # K_in
]

c_lib.run_gemv_int4_gelu.argtypes = [...]  # Same signature
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

`np.empty` arrays corresponding to the output size (M_out) are **allocated only once** and reused.
This removes the memory allocation cost and GC pressure that occur with every call.

> After overwriting in the C++ kernel, you **must return it with `.copy()`**.
> Returning the pool buffer directly will contaminate the value in subsequent calls.

---

### Function Reference

#### `igpu_matmul(x_vec, weight_data)`

```python
def igpu_matmul(
    x_vec:       np.ndarray,             # [K_in] float32
    weight_data: tuple | np.ndarray,     # INT4 tuple or standard float matrix
) -> np.ndarray                          # [M_out] float32
```

**INT4 Tuple Path**:
```python
packed, scale = weight_data          # packed: uint8[M_out, K_in/2]
out_buf = _get_output_buf(M_out)
c_lib.run_gemv_int4(x_f32, packed, scale, out_buf, M_out, K_in)
return out_buf.copy()
```

**Standard Matrix Fallback Path**:
```python
return np.dot(x_f32, weight_data.astype(np.float32).T)
```

> Note the `np.dot(x, W.T)` form — assumes the weight is in the `[M_out, K_in]` layout.

---

#### `igpu_matmul_gelu(x_vec, weight_data)`

```python
def igpu_matmul_gelu(
    x_vec:       np.ndarray,
    weight_data: tuple | np.ndarray,
) -> np.ndarray
```

**INT4 Tuple Path**: Calls C++ `run_gemv_int4_gelu` — **Fuses execution** of GEMV and GELU within a single kernel.
**Standard Matrix Fallback Path**: Separate call to `CPU_CORE.gelu()` after `np.dot`.

```
Advantages of fusing GEMV + GELU:
  Separate execution: GEMV → [M_out] memory write → GELU → [M_out] memory read
  Fused execution: GEMV → acc → GELU (applied immediately at the acc stage) → [M_out] memory write once
  → Saves 1 memory round trip (Saves about 64KB when M_out=16384)
```

---

#### `preload_and_free(W, keys)` / `_get_or_upload_weight(weight_data)`

```python
def preload_and_free(W: dict, keys: list): pass
def _get_or_upload_weight(weight_data):    pass
```

These are **empty functions (no-op)** for interface compatibility with `IGPU_CORE.py`.
Since there is no concept of VRAM upload in CPU mode, they do nothing.

---

#### `warmup()`

```python
def warmup()
```

Calls the C++ kernel once with a small dummy array to warm up the OpenMP thread pool and AVX2 registers.

```python
dummy_x = np.zeros(2048, dtype=np.float32)
dummy_p = np.zeros((2048, 1024), dtype=np.uint8)
dummy_s = np.zeros(2048, dtype=np.float32)
igpu_matmul(dummy_x, (dummy_p, dummy_s))
```

Prevents latency (thread spawn, cache cold start) on the first actual inference call.

---

### Interface Correspondence Table with `IGPU_CORE.py`

| Function | CPU_MATRIX_CORE | IGPU_CORE |
| ------------------------- | ------------------------------ | ----------------------------------- |
| `igpu_matmul()` | `run_gemv_int4` (C++/CPU) | `run_vulkan_gemv` (Vulkan/iGPU) |
| `igpu_matmul_gelu()` | `run_gemv_int4_gelu` (C++/CPU) | `igpu_matmul()` + `CPU_CORE.gelu()` |
| `preload_and_free()` | no-op | no-op (Legacy VRAM optimization) |
| `_get_or_upload_weight()` | no-op | Weight VRAM upload |
| `warmup()` | OpenMP warmup | Prints shader load confirmation message |
| `prefetch_weight()` | None | `prefetch_weight_async` (Asynchronous) |
| `compute_pingpong()` | None | `run_vulkan_gemv_pingpong` (Ping-pong) |

> CPU mode does not have a ping-pong prefetch feature. Ping-pong optimization is exclusive to Vulkan iGPU mode.

---

### Module Dependencies

```
main.py
  │
  ├── ACCEL_MODE = "CPU"  →  CPU_MATRIX_CORE.py
  │                              └── C_DLL/my_accelerator.so
  │                                    ├── run_gemv_int4()
  │                                    └── run_gemv_int4_gelu()
  │
  └── (Always) CPU_CORE.py
              ├── C_DLL/my_accelerator.so
              │     ├── run_gelu_inplace()
              │     ├── run_RMSNorm_inplace()
              │     ├── run_rope_inplace()
              │     └── run_unpack_int4_inplace()
              └── transformers.AutoTokenizer
```
