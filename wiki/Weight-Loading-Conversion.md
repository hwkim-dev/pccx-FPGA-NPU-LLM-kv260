# Code Documentation (5/8) — Weight Loading & Conversion Pipeline

> **Target Files**: `safeTensor.py` · `Optim_tensor_load.py`
> **Role**: Loading weight mmap during inference (`safeTensor`) + Initial one-time conversion script (`Optim_tensor_load`)

---

## Overall Weight Pipeline Overview

The two files constitute a **step-by-step pipeline**.

```
[Initial One-time Execution]
Original Model (*.safetensors)
    └── quantize.py             ← INT4 quantization + generates .scale files
        └── local_gemma_3n_int4/*.safetensors

    └── Optim_tensor_load.py   ← safetensors → decomposes into individual .npy files + transpose processing
        └── mmap_weights/*.npy  (1 file per weight)

[Every Inference Execution]
    └── safeTensor.py           ← Virtually maps mmap_weights/*.npy with mmap_mode='r'
        └── Returns load_local_weights() value → main.py
```

> `safeTensor.py` will only operate normally after `Optim_tensor_load.py` has been executed.

---

## 1. `safeTensor.py`

### Overview

A loader that virtually maps the `.npy` files in the `mmap_weights/` directory with `mmap_mode='r'`, immediately returning the entire weight dictionary **without consuming RAM**.

There are two versions of the implementation within the file:

| Version | Location | Status | Source |
| ------------- | ---------------------------- | ---------- | --------------------------------------------- |
| Old Version | Top of the file (Inside the long `'''` comment) | ❌ Inactive | Parses `local_gemma_3n_int4/*.safetensors` directly |
| **Current Version**| Bottom of the file (After `'''`) | ✅ **Active** | mmap loads `mmap_weights/*.npy` |

The old version directly read files with `safetensors.torch.load_file` and converted torch tensors to numpy, but required the entire model to be loaded into RAM during loading. The current version solves this problem.

---

### Core Principle of the mmap Strategy

```python
val = np.load("mmap_weights/some_weight.npy", mmap_mode='r')
# ↑ Data read from disk at this point: 0 bytes
# ↑ OS only registers the virtual address of the file (1 page table entry)
# Actual disk read occurs only when the corresponding array element is accessed for the first time (Demand Paging)
```

As a result, the increase in RAM usage immediately after calling `load_local_weights()` is almost 0, and each weight is automatically loaded in OS page units at the time it is actually used.

**Specialized optimization for W_embed / W_ple**: Since both weights involve a pattern of querying only one row for an embedding, mmap is particularly effective. Only about 5.5 KB is actually read per token.

---

### `load_local_weights()` Function

```python
def load_local_weights(model_dir=mmap_dir) -> tuple
```

**Returned Tuple Structure**:

```python
return (
    W_embed,          # tuple (packed[262400,1024] uint8, scale[262400] f32) — mmap
    W_ple_packed,     # ndarray [262144, 4480] uint8                         — mmap
    W_ple_scale,      # ndarray [262144] float32                             — mmap
    norm_ple,         # ndarray [256] float32
    W_ple_proj,       # tuple (packed, scale) INT4
    altup_projs,      # list[3] × ndarray [2048, 2048] float32
    altup_unprojs,    # list[3] × ndarray [2048, 2048] float32
    W_final_norm,     # ndarray [2048] float32
    W_lm_head,        # tuple — Same object as W_embed (Tied Weights)
    layers,           # dict[str, list[35]] — Layer-wise weights
)
```

**`W_lm_head = W_embed` Design**:
Gemma 3N shares the same weights for the input embedding and the output LM Head (Tied Embedding). Since it references the same mmap object without separate copying, there is no memory duplication.

---

### Detailed Execution Flow

#### Step 1: Collect File List and Separate Scales

```python
all_files = glob.glob("mmap_weights/*.npy")
all_keys  = [basename(f)[:-4] for f in all_files]  # Remove .npy from filename

# Create scale file index (Fast pair search)
scales = {k[:-6]: k for k in all_keys if k.endswith(".scale")}
# Example: {"model.language_model.layers.0.self_attn.q_proj.weight":
#      "model.language_model.layers.0.self_attn.q_proj.weight.scale"}
```

#### Step 2: mmap Virtual Mapping Loop

```python
for k in all_keys:
    if k.endswith(".scale"): continue    # Scales are handled together when loading the main body

    val = np.load(f"mmap_weights/{k}.npy", mmap_mode='r')  # RAM consumption 0

    if k in scales:                      # INT4 Tensor: Bundled as a (packed, scale) tuple
        scale_val = np.load(f"mmap_weights/{scales[k]}.npy", mmap_mode='r')
        val = (val, scale_val)

    # Extract layer index and subkey using regex
    match = re.match(r"model\.language_model\.layers\.(\d+)\.(.*)", k)
    if match:
        layer_idx = int(match.group(1))  # 0 ~ 34
        sub_key   = match.group(2)       # "self_attn.q_proj.weight" etc.
        layers[KEY_MAP[sub_key]][layer_idx] = val
    else:
        globals_dict[k] = val            # Global weight not belonging to a layer
```

#### Step 3: Decompose and Return Global Weights

```python
P = "model.language_model."

W_embed      = globals_dict[P + "embed_tokens.weight"]          # tuple (mmap)
W_ple_packed,\
W_ple_scale  = globals_dict[P + "embed_tokens_per_layer.weight"] # tuple unpacking
W_ple_proj   = globals_dict[P + "per_layer_model_projection.weight"]
norm_ple     = globals_dict[P + "per_layer_projection_norm.weight"]
altup_projs  = [globals_dict[P + f"altup_projections.{i}.weight"] for i in range(3)]
altup_unprojs= [globals_dict[P + f"altup_unembed_projections.{i}.weight"] for i in range(3)]
W_final_norm = globals_dict[P + "norm.weight"]
W_lm_head    = W_embed   # References the same object (Tied Weights)
```

---

### SafeTensor Original Key → Internal Key Mapping Table

| SafeTensor Original Key (sub_key) | `layers` Dictionary Key |
| ----------------------------------- | -------------------- |
| `self_attn.q_proj.weight` | `W_q` |
| `self_attn.k_proj.weight` | `W_k` |
| `self_attn.v_proj.weight` | `W_v` |
| `self_attn.o_proj.weight` | `W_o` |
| `self_attn.q_norm.weight` | `gamma_q` |
| `self_attn.k_norm.weight` | `gamma_k` |
| `input_layernorm.weight` | `input_ln` |
| `post_attention_layernorm.weight` | `post_attn_ln` |
| `pre_feedforward_layernorm.weight` | `pre_ffn_ln` |
| `post_feedforward_layernorm.weight` | `post_ffn_ln` |
| `mlp.gate_proj.weight` | `W_gate` |
| `mlp.up_proj.weight` | `W_up` |
| `mlp.down_proj.weight` | `W_down` |
| `per_layer_input_gate.weight` | `ple_gate` |
| `per_layer_projection.weight` | `ple_proj` |
| `post_per_layer_input_norm.weight` | `ple_post_ln` |
| `laurel.linear_left.weight` | `laurel_left` |
| `laurel.linear_right.weight` | `laurel_right` |
| `laurel.post_laurel_norm.weight` | `laurel_norm` |
| `altup.router_norm.weight` | `altup_rn` |
| `altup.modality_router.weight` | `altup_router` |
| `altup.prediction_coefs.weight` | `altup_pred` |
| `altup.correction_coefs.weight` | `altup_corr` |
| `altup.correct_output_scale` | `altup_scale` |

---

---

## 2. `Optim_tensor_load.py`

### Overview

Two independent roles are **mixed in one file**.

| Section | Description | Execution Method |
| ---------- | ---------------------------------------------------- | -------------------------------------------- |
| **Top Part** | Memory usage measurement and structure inspection utility (`debug()`) | Function — Explicit call required |
| **Bottom Part** | SafeTensors → `.npy` conversion script | **Module top-level code** — Executes automatically upon `import` |

> **Warning**: The conversion code at the bottom is exposed at the module level, not wrapped in a function.
> Just running `import Optim_tensor_load` will immediately start the conversion process.
> In actual use, it should only be used to **execute directly** with `python Optim_tensor_load.py`.

---

### Top Part: Memory Inspection Utility

#### `get_real_memory_size(obj)`

```python
def get_real_memory_size(obj) -> int  # bytes
```

Recursively calculates the **actual memory footprint** of nested Python objects.

`sys.getsizeof()` only returns the shell size of a container (list, tuple) and does not include the `.nbytes` of internal numpy arrays. This function compensates for that limitation.

```python
def get_real_memory_size(obj):
    total = sys.getsizeof(obj)          # Container shell size

    if isinstance(obj, np.ndarray):
        total += obj.nbytes             # Add actual data size
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            total += get_real_memory_size(item)  # Recursive exploration
    return total
```

**Example — Calculating the actual size of an INT4 Tuple**:
```
W_q[0] = (packed[2048, 1024] uint8, scale[2048] float32)
  get_real_memory_size(W_q[0])
    = getsizeof(tuple)          # ~56 bytes (shell)
    + getsizeof(packed)         # ~112 bytes (ndarray object)
    + packed.nbytes             # 2048 × 1024 × 1 = 2,097,152 bytes
    + getsizeof(scale)          # ~112 bytes
    + scale.nbytes              # 2048 × 4 = 8,192 bytes
    ≈ 2,105,524 bytes (~2.0 MB)
```

---

#### `inspect_matrix_structure(name, obj)`

```python
def inspect_matrix_structure(name: str, obj) -> str
```

Describes the **nested structure and actual dimensions** of a weight object as a string.
This function is used to generate the tables in `optim_tensor_size.md`.

**Recursive Processing Rules**:

| Type | Output Format |
| ------------------------ | ------------------------------------------------------------- |
| `list` | `List[N] ──> {Structure of element 0}` (Inspects only element 0 as representative) |
| `tuple` | `Tuple( {Structure of element 1}, {Structure of element 2} )` |
| `np.ndarray` (uint8, 2D) | `[ matrix: A x B , type: uint8 , (INT4 dimension: A x B*2) ]` |
| `np.ndarray` (Others) | `[ matrix: shape , type: dtype ]` |

**INT4 Dimension Correction**: Since uint8 2D arrays pack 2 INT4s each, the actual number of columns is displayed as `shape[1] * 2`.

---

#### `format_memory_size(total_bytes)` / `calculate_memory_usage(obj)`

```python
def format_memory_size(total_bytes: int) -> str  # "GB | MB | Mb" format string
def calculate_memory_usage(obj) -> str
```

Formats the bytes obtained from `get_real_memory_size()` into three units: GB/MB/Megabits.

---

#### `debug()`

```python
def debug()
```

Calls `safeTensor.load_local_weights()` and then prints the structure and memory size of all weights in a Markdown table format. Currently, it is commented out (`#if __name__ == "__main__": debug()`) so it does not run automatically.

Output Example:
```
| name | matrix                                                                                                                         | GB       | MB     | Mb      |
| ---- | ------------------------------------------------------------------------------------------------------------------------------ | -------- | ------ | ------- |
| W_q  | List[35] ──>  Tuple( [ matrix: 2048 x 1024 , type: uint8 , (INT4 dimension: 2048 x 2048) ], [ matrix: 2048 , type: float32 ] ) | 0.068636 | 70.284 | 562.269 |
...
```

---

### Bottom Part: SafeTensors → `.npy` Conversion Script

**Execution Conditions**: `local_gemma_3n_int4/*.safetensors` exist (After running quantize.py)
**Output Location**: `mmap_weights/*.npy` (1 file per tensor)

#### Execution Flow

```python
# 1. Create output directory
os.makedirs("mmap_weights/", exist_ok=True)

# 2. Iterate through all .safetensors files
for st_file in sorted(glob("local_gemma_3n_int4/*.safetensors")):
    tensors = load_file(st_file)      # safetensors → torch tensor dict

    # Identify INT4 tensors (those with a scale file)
    quantized_bases = [k[:-6] for k in tensors if k.endswith(".scale")]

    for k, val in tensors.items():
        # Convert bfloat16 → float32 (Handles dtype not supported by numpy)
        if val.dtype == torch.bfloat16:
            val = val.to(torch.float32)
        arr = val.numpy()

        # Determine Transpose
        is_quantized = (k in quantized_bases) or k.endswith(".scale")
        needs_transpose = False

        if not is_quantized:  # ← INT4 tensors are NEVER transposed (Core rule!)
            if any(suffix in k for suffix in TRANSPOSE_SUFFIXES):
                needs_transpose = True

        if needs_transpose:
            arr = np.ascontiguousarray(arr.T)
        else:
            arr = np.ascontiguousarray(arr)

        np.save(f"mmap_weights/{k}.npy", arr)
```

#### Transpose Target List

Among the non-quantized (float) weights, only those containing the suffixes below are transposed.

| Suffix | Corresponding Weight |
| ------------------------------------------------------------------ | -------------------- |
| `per_layer_model_projection.weight` | W_ple_proj |
| `altup_projections` | altup_projs |
| `altup_unembed_projections` | altup_unprojs |
| `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight` | Q, K, V, O |
| `gate_proj.weight`, `up_proj.weight`, `down_proj.weight` | FFN |
| `per_layer_input_gate.weight`, `per_layer_projection.weight` | ple_gate, (ple_proj) |
| `laurel.linear_left.weight`, `laurel.linear_right.weight` | LAuReL |
| `altup.modality_router.weight` | altup_router |

**Why transpose?**
Linear layer weights in SafeTensors are stored in the form `[out, in]` (Row: Output, Column: Input).
To perform an `x @ W` operation (Vector-Matrix multiplication) during inference, an `[in, out]` layout is required, so we convert it in advance with `.T`.

**Why not transpose INT4 tensors?**
During INT4 quantization, they are already stored in the `[out_dim, in_dim/2]` layout (Quantized per row, Row=Output).
Since the GEMV kernel (`run_gemv_int4` in `my_accelerator.cpp`) is designed to consume this layout directly, transposing it will rather cause malfunctions.

---

### Complete Pipeline Summary

```
[1] Execute quantize.py
    local_gemma_3n/ → local_gemma_3n_int4/
    (Original float16/32 → Generates INT4 packed + .scale files)

[2] Execute Optim_tensor_load.py  (First time only)
    local_gemma_3n_int4/*.safetensors → mmap_weights/*.npy
    - Converts bfloat16 → float32
    - Non-INT4 weights: Apply Transpose
    - INT4 weights: Store as is without transpose
    - Guarantees C-contiguous memory with np.ascontiguousarray

[3] safeTensor.py (Every execution)
    mmap_weights/*.npy → Virtually maps with mmap_mode='r'
    - Automatically pairs scale files → Constructs (packed, scale) tuple
    - Parses layer indices with Regex
    - Returns a 10-item tuple → Consumed by main.py
```
