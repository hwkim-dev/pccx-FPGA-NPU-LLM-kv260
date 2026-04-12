# Gemma 3N (INT4 + AltUp) Pipeline — Complete Operation Specification

Base Dimension Constants: $D=2048$ , $D_{ffn}=16384$ , $D_{patch}=256$ , $D_{router}=4$ , $Vocab=262400$  
Attention Heads: **8 Q / 2 KV**, $d_{head}=256$ , $N_{layers}=35$ 

---

## 0. Pre-definition: Core Function Mathematical Definitions

### Embedding
Extracts the row corresponding to the integer ID from the weight matrix.

$$
Output = W_{embed}[token\_id,\ :]
$$

### RMSNorm

$$
RMS = \sqrt{\dfrac{1}{N}\sum_{i=1}^{N}x_i^2 + 10^{-6}}, \qquad Output = \frac{x}{RMS} \times \gamma
$$

### GELU

$$
Output = 0.5 \times x \times \!\left(1 + \tanh\!\left(\sqrt{\tfrac{2}{\pi}}\,(x + 0.044715\,x^3)\right)\right)
$$

### RoPE (Rotary Position Embedding)

$$
Output_{2i}   = x_{2i}\cos\theta - x_{2i+1}\sin\theta
$$

$$
Output_{2i+1} = x_{2i}\sin\theta + x_{2i+1}\cos\theta
$$

### INT4 Dequantization

$$
w_0 = W_{packed} \ \land \ 0x0F, \quad w_0 = w_0 - 16 \text{ (if } w_0 > 7\text{)}
$$


$$
w_1 = W_{packed} \gg 4, \quad w_1 = w_1 - 16 \text{ (if } w_1 > 7\text{)}
$$

$$
Output = [w_0,\ w_1] \times Scale
$$

### GQA (Grouped-Query Attention)
Groups the 8 Q heads with the 2 KV heads in a 4:1 ratio. For group $g$  ( $g=0,1$ ):

$$
Q_g = Q[4g : 4g+4,\ :] \in \mathbb{R}^{4 \times d_{head}}
$$

$$
\text{scores}_g = \frac{Q_g \cdot K_g^{\,T}}{\sqrt{d_{head}}} \in \mathbb{R}^{4 \times L}
$$

$$
\text{out}_g = \text{softmax}(\text{scores}_g) \cdot V_g \in \mathbb{R}^{4 \times d_{head}}
$$

$$
GQA\_out = \text{concat}(\text{out}_0,\ \text{out}_1) \in \mathbb{R}^{8 \times d_{head}} \xrightarrow{\text{flatten}} \mathbb{R}^{1 \times D}
$$

Here, $L = \text{current sequence length}$ , and $K_g, V_g \in \mathbb{R}^{L \times d_{head}}$  are retrieved from the KV cache.

---
---

## 1. Token Embedding

$$
x_0 = Embedding(token\_id,\ W_{embed}) \times \sqrt{2048}
$$

> `token_id` is clipped to `min(token_id, 262143)` to prevent exceeding the PLE vocab range.

W_embed  [262400 × 2048]  INT4
────────────────────────────────
x₀       [1 × 2048]       float32


---

## 2. AltUp Initial Projections

Inserts $x_0$  into the 0th row of an empty matrix $ xs $ , and fills the 1st~3rd rows via dot product.

$$
xs[0] = x_0
$$

$$
xs[k+1] = x_0 \cdot altup\_projs[k], \quad k=0,1,2
$$

```
altup_projs[k]  [2048 × 2048]  float32  (k=0,1,2)
──────────────────────────────────────────────────
xs              [4 × 2048]     float32
```

---

## 3. PLE Setup (Position & Patch Embedding Pre-computation)

Pre-computes the auxiliary vector $pli\_all$  at once to be shared across all 35 layers.

**Step 1 — Linear Projection and Normalization:**

$$
x_{proj} = \frac{x_0 \cdot W_{ple\_proj}^{T}}{\sqrt{2048}} \xrightarrow{\text{reshape}} [35 \times 256]
$$

$$
x_{proj\_normed} = \frac{x_{proj}}{RMS(x_{proj})} \times norm_{ple}
$$

W_ple_proj  [8960 × 2048]  INT4   →  x_proj        [35 × 256]  float32
norm_ple    [256]          float32 →  x_proj_normed [35 × 256]  float32


**Step 2 — Patch Embedding Extraction:**

$$
y = Embedding(token\_id,\ W_{ple}) \xrightarrow{\text{reshape}} [35 \times 256] \times \sqrt{256}
$$

W_ple  [262144 × 8960]  INT4   →  y  [35 × 256]  float32


**Step 3 — Composition:**

$$
pli\_all = (x_{proj\_normed} + y) \times \frac{1}{\sqrt{2}}
$$

pli_all  [35 × 256]  float32


---

## 4. Transformer Layer (Repeated 35 Times)

> Loop index $i = 0, 1, \ldots, 34$ 

---

### A. AltUp Router & Prediction

Mixes the 4 modality vectors to generate $xs_{pred}$ .

$$
x_n = \frac{RMSNorm(xs[0],\ W_{altup\_rn})}{2048}
$$

$$
modalities = \tanh(x_n \cdot W_{altup\_router})
$$

$$
coef\_mat = (W_{altup\_pred} \cdot modalities) \xrightarrow{\text{reshape}} [4 \times 4]
$$

$$
xs_{pred} = xs + coef\_mat \cdot xs
$$

```
altup_rn     [2048]      float32  →  x_n        [1 × 2048]   float32
altup_router [2048 × 4]  float32  →  modalities [4]          float32
altup_pred   [16 × 4]    float32  →  coef_mat   [4 × 4]      float32
──────────────────────────────────────────────────────────────────────
xs_pred      [4 × 2048]  float32
```

---

### B. Attention — Q, K, V Projection

Extracts $xs_{pred}[0]$  to use as input.

$$
x_{input} = xs_{pred}[0], \qquad x_{norm} = RMSNorm(x_{input},\ W_{input\_ln})
$$

$$
Q = x_{norm} \cdot W_q^{\,T}, \quad K = x_{norm} \cdot W_k^{\,T}, \quad V = x_{norm} \cdot W_v^{\,T}
$$

input_ln  [2048]           float32
W_q       [2048 × 2048]    INT4    →  Q  [1 × 2048]  →  [8 × 256]  float32
W_k       [512  × 2048]    INT4    →  K  [1 × 512]   →  [2 × 256]  float32
W_v       [512  × 2048]    INT4    →  V  [1 × 512]   →  [2 × 256]  float32


---

### B-2. Head-wise QK-Norm

Applies RMSNorm per head (256 dimensions).

$$
Q^{head}_i = \frac{Q^{head}_i}{RMS(Q^{head}_i)} \times \gamma_q, \quad i=0,\ldots,7
$$

$$
K^{head}_j = \frac{K^{head}_j}{RMS(K^{head}_j)} \times \gamma_k, \quad j=0,1
$$

```
gamma_q  [256]  float32  (Shared, same weights for all 8 heads)
gamma_k  [256]  float32  (Shared, same weights for all 2 heads)
──────────────────────────────────────────────────────
Q_norm   [8 × 256]  float32
K_norm   [2 × 256]  float32
```

---

### B-3. Dynamic-θ RoPE

$$
\theta = \begin{cases} 1,000,000 & i \pmod{5} = 4 \\ 10,000 & \text{otherwise} \end{cases}
$$

 
$$
Q_{rope} = RoPE(Q_{norm},\ pos,\ \theta), \quad K_{rope} = RoPE(K_{norm},\ pos,\ \theta)
$$

Q_rope  [8 × 256]  float32
K_rope  [2 × 256]  float32


---

### B-4. KV Cache Storage & Asymmetric Sharing

> KV cache is stored in **float16** (downcasted from float32).  
> Shape: `K_cache[20, max_seq, 512]`, `V_cache[20, max_seq, 512]`

$$
i < 20: \quad K\_cache[i,\ pos,\ :] = K_{rope},\quad V\_cache[i,\ pos,\ :] = V
$$

$$
i \ge 20: \quad \begin{cases} 
target_K = K_{cache}[19, \ :pos+1, \ :] & i \% 5 = 4 \\ 
target_K = K_{cache}[18, \ :pos+1, \ :] & \text{otherwise} 
\end{cases}
$$


K_cache  [20 × max_seq × 512]  float16
V_cache  [20 × max_seq × 512]  float16
target_K [L × 512]  →  [L × 2 × 256]  float16   (L = pos+1)
target_V [L × 512]  →  [L × 2 × 256]  float16


---

### B-5. GQA & Output Projection

$$
attn\_raw = GQA(Q_{rope},\ target\_K,\ target\_V)
$$

$$
attn\_output = attn\_raw \cdot W_o^{\,T}
$$

scores_g  [4 × L]      float32   (g = 0,1)
out_g     [4 × 256]    float32
attn_raw  [1 × 2048]   float32
W_o       [2048 × 2048]  INT4
──────────────────────────────────
attn_output  [1 × 2048]  float32


---

### C. LAuReL + Attention Residual Composition

For LAuReL's residual connection, the **normalized input** ( $x_{norm}$ ) is added instead of the original input.

$$
laurel\_x = (x_{norm} \cdot W_{laurel\_left}^{\,T}) \cdot W_{laurel\_right}^{\,T}
$$

$$
laurel\_out = x_{norm} + RMSNorm(laurel\_x,\ W_{laurel\_norm})
$$

$$
attn\_output = RMSNorm(attn\_output,\ W_{post\_attn\_ln}) + x_{input}
$$

$$
x_{attn} = (attn\_output + laurel\_out) \times \frac{1}{\sqrt{2}}
$$

```
laurel_left   [64   × 2048]  INT4   →  laurel_x (intermediate) [1 × 64]    float32
laurel_right  [2048 × 64  ]  INT4   →  laurel_x                [1 × 2048]  float32
laurel_norm   [2048]          float32
post_attn_ln  [2048]          float32
──────────────────────────────────────────────────────────────────────────────
x_attn        [1 × 2048]     float32
```

---

### D. FFN (Gate-Up-Down)

$$
x_{n2} = RMSNorm(x_{attn},\ W_{pre\_ffn\_ln})
$$

$$
gate\_raw = x_{n2} \cdot W_{gate}^{\,T}, \qquad up\_out = x_{n2} \cdot W_{up}^{\,T}
$$

pre_ffn_ln  [2048]           float32
W_gate      [16384 × 2048]   INT4    →  gate_raw  [1 × 16384]  float32
W_up        [16384 × 2048]   INT4    →  up_out    [1 × 16384]  float32


** $i \ge 10$  — Standard GELU Gate:**

$$
gate\_out = GELU(gate\_raw), \qquad hidden = gate\_out \times up\_out
$$

** $i < 10$  — Sparse Gate:**

$$
cutoff = Mean(gate\_raw) + Std(gate\_raw) \times 1.6448536
$$

$$
sparse\_gate = \max(gate\_raw - cutoff,\ 0), \qquad hidden = GELU(sparse\_gate) \times up\_out
$$

hidden  [1 × 16384]  float32


**FFN Output:**

$$
mlp\_out = hidden \cdot W_{down}^{\,T}
$$

$$
outputs = RMSNorm(mlp\_out,\ W_{post\_ffn\_ln}) + x_{attn}
$$

```
W_down       [2048 × 16384]  INT4    →  mlp_out  [1 × 2048]  float32
post_ffn_ln  [2048]           float32
───────────────────────────────────────────────────────────────────
outputs      [1 × 2048]      float32
```

---

### E. AltUp Correction & PLE Mixing

**Step 1 — Scale & Innovation:**

$$
activated = outputs \times W_{altup\_scale}
$$

$$
innovation = activated - xs_{pred}[0]
$$

altup_scale  [2048]       float32
activated    [1 × 2048]   float32
innovation   [1 × 2048]   float32


**Step 2 — Correction Coefficients:**

$$
x_{n3} = \frac{RMSNorm(activated,\ W_{altup\_rn})}{2048}
$$

$$
mod\_corr = \tanh(x_{n3} \cdot W_{altup\_router})
$$

$$
corr\_coefs = W_{altup\_corr} \cdot mod\_corr + 1.0
$$

altup_rn      [2048]      float32  →  x_n3       [1 × 2048]  float32
altup_router  [2048 × 4]  float32  →  mod_corr   [4]         float32
altup_corr    [4 × 4]     float32  →  corr_coefs [4]         float32


**Step 3 — Modality Update (Broadcasting):**

$$
xs_{new} = xs_{pred} + corr\_coefs_{[:,1]} \times innovation_{[1,:]}
$$

corr_coefs[:,np.newaxis]  [4 × 1]      ×  innovation  [1 × 2048]  →  [4 × 2048]
xs_new                    [4 × 2048]   float32


**Step 4 — PLE Mixing:**

$$
gate\_ple = GELU(activated \cdot W_{ple\_gate}^{\,T}) \times pli
$$

$$
mapped = RMSNorm(gate\_ple \cdot W_{ple\_proj},\ W_{ple\_post\_ln})
$$

$$
xs_{new}[1:] = xs_{new}[1:] + mapped
$$

W_ple_gate   [256  × 2048]  INT4    →  gate_ple (INT4 matmul)  [1 × 256]   float32
pli          [256]          float32 →  gate_ple (×pli)         [1 × 256]   float32
W_ple_proj   [256  × 2048]  float32 →  gate_ple·W_ple_proj     [1 × 2048]  float32
ple_post_ln  [2048]         float32 →  mapped                  [1 × 2048]  float32


$$
xs \leftarrow xs_{new} \quad [4 \times 2048]
$$

---

## 5. Decode Logits

**Step 1 — Magnitude Matching & Unprojection:**

$$
target\_mag = \sqrt{Mean(xs[0]^2)}
$$

$$
{proj_x}_k = xs[k+1] \cdot {altup\_unprojs}[k], \quad k=0,1,2
$$

$$
{new\_mag}_k = \sqrt{Mean({proj\_x}_k^2)}, \qquad {proj\_x}_k \mathrel{*}= \frac{target\_mag}{\max({new\_mag}_k,\ 10^{-12})}
$$


altup_unprojs[k]  [2048 × 2048]  float32   (k=0,1,2)
proj_xₖ           [1 × 2048]     float32


**Step 2 — Average & Final Projection:**

$$
x_{final} = Mean([xs[0],\ {proj\_x}_0,\ {proj\_x}_1,\ {proj\_x}_2])
$$


$$
x_{final\_norm} = RMSNorm(x_{final},\ W_{final\_norm})
$$

$$
logits = x_{final\_norm} \cdot W_{lm\_head}^{\,T}
$$

```
W_final_norm  [2048]           float32
W_lm_head     [262400 × 2048]  INT4  
─────────────────────────────────────  
x_final_norm  [1 × 2048]       float32
logits        [1 × 262400]     float32
```

**Step 3 — Logit Soft-Capping:**

$$
Logits = 30.0 \times \tanh\!\left(\frac{logits}{30.0}\right)
$$

---

## 6. Generation & Sampling

### Repetition Penalty ( $\rho = 1.15$ )

$$
Logits_t = \begin{cases} Logits_t \times \rho & \text{if } Logits_t < 0 \\ Logits_t / \rho & \text{if } Logits_t \ge 0 \end{cases}
$$

### Temperature Softmax

** $T = 0$ :** $\quad next\_token = \arg\max(Logits)$ 

** $ T > 0 $  (SIMD C++ Kernel):**

$$
Logits_{safe} = \frac{Logits}{T} - \max\!\left(\frac{Logits}{T}\right)
$$

$$
probs_i = \frac{\exp(Logits_{safe,i})}{\sum_j \exp(Logits_{safe,j})}
$$

logits  [262400]  float32
probs   [262400]  float32


### Top-P Sampling ( $p = 0.9$ )

Sorts probabilities in descending order and keeps only the tokens whose cumulative sum is less than $p$ .

$$
sorted\_idx = \text{argsort}(probs)[::-1]
$$

$$
\text{keep} = \{i : \text{cumsum}(probs[sorted\_idx])_i - probs[sorted\_idx_i] < p\}
$$

$$
probs_{filtered}[i] = \begin{cases} probs[i] & i \in \text{keep} \\ 0 & \text{otherwise} \end{cases}
$$

Renormalizes so the sum becomes 1 after filtering.

$$
probs = \frac{probs_{filtered}}{\sum probs_{filtered}}
$$

$$
next\_token \sim \text{Categorical}(probs)
$$

---

## 7. System Optimization Architecture (Hardware Integration Reference)

### Ping-Pong Double Buffering (`hw_compute_pingpong`)
While executing the current layer's matrix operation (e.g., $Q$), it prefetches the next weights (e.g.,$ K $ ) into the opposite buffer in a background thread → hides I/O latency to 0.

### In-place Memory Overwrite (`__restrict__`)
RMSNorm, GELU, and Softmax overwrite the input memory directly without allocating separate output tensors.

### MMAP Zero-Copy Streaming
Instead of loading the entire model into RAM, it uses OS paging (Page Faults) to directly stream the necessary rows from SSD via C-Contiguous pointers.
