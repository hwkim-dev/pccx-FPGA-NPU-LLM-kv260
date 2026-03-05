import numpy as np
import math
from transformers import AutoTokenizer
import os

# 변경: 로컬 폴더 경로에서 오프라인으로 바로 가져오기!
base_dir = os.path.dirname(os.path.abspath(__file__))
model_id = os.path.join(base_dir, "local_gemma_3n")
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

def tokenize(text):
    tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
    print(f"[CPU] Tokenized IDs: {tokens}")
    return tokens

# =====================================================================
# 2. Embedding (Token ID -> 2048차원 벡터)
# =====================================================================
def embedding(token_id, W_embed_real):
    x = W_embed_real[token_id] 
    return x.astype(np.float16)

# =====================================================================
# 3. PLE (Per-Layer Embedding) 주입
# =====================================================================
# 🔥 파라미터로 진짜 PLE 가중치들을 받도록 수정!
def inject_ple(x, ple_proj_down, ple_embed, ple_proj_up, layer_idx=0):
    x_down = np.dot(x, ple_proj_down) 
    x_gated = x_down * ple_embed[0]
    ple_out = np.dot(x_gated, ple_proj_up)
    
    x_final = x + ple_out
    return x_final.astype(np.float16)

def cpu_qk_norm(Q, K, gamma_q, gamma_k):
    q_inv_sqrt = 1.0 / np.sqrt(np.mean(Q**2) + 1e-6)
    k_inv_sqrt = 1.0 / np.sqrt(np.mean(K**2) + 1e-6)
    Q_norm = (Q * q_inv_sqrt) * gamma_q
    K_norm = (K * k_inv_sqrt) * gamma_k
    return Q_norm, K_norm

# =====================================================================
# 5. RoPE (회전 위치 인코딩)
# =====================================================================
def cpu_rope(x, pos, theta_base=10000.0):
    """
    [CPU] 복소수 회전 행렬 연산 (차원이 작아서 CPU 연산이 빠름)
    x: Q_norm [2048] 또는 K_norm [512]
    """
    dim = 256 # 헤드당 차원 (head_dim)
    num_heads = len(x) // dim
    
    # [num_heads, head_dim] 형태로 View 변경 (C++ 포인터 캐스팅 느낌!)
    x_reshaped = x.reshape(num_heads, dim)
    out = np.zeros_like(x_reshaped)
    
    for h in range(num_heads):
        for i in range(0, dim, 2):
            # 주파수 계산
            freq = 1.0 / (theta_base ** (i / dim))
            val = pos * freq
            cos_val = math.cos(val)
            sin_val = math.sin(val)
            
            # 짝수/홀수 인덱스 교차 회전 연산
            x0 = x_reshaped[h, i]
            x1 = x_reshaped[h, i+1]
            out[h, i]   = x0 * cos_val - x1 * sin_val
            out[h, i+1] = x1 * cos_val + x0 * sin_val
            
    # 다시 1D 벡터로 평탄화해서 리턴
    return out.flatten()

# =====================================================================
# 6. KV 캐시 업데이트 (DDR 메모리 포인터 복사)
# =====================================================================
def cpu_update_kv_cache(K_rope, V, layer_idx, K_cache, V_cache):
    """
    [CPU] 35층 각 레이어의 캐시 리스트에 현재 턴의 K, V 벡터 추가
    K_rope: [512], V: [512]
    """
    K_cache[layer_idx].append(K_rope)
    V_cache[layer_idx].append(V)

# =====================================================================
# 7. GQA (Grouped Query Attention) - 슬라이딩 윈도우 기반
# =====================================================================
def cpu_gqa(Q_rope, K_cache_layer, V_cache_layer):
    """
    [CPU] 메모리(KV Cache)를 읽어와서 Q와 내적하는 과정.
    디코드(T=1) 단계에서는 Matrix-Vector 곱셈(GEMV) 형태라 CPU가 처리함.
    """
    # Q: [8, 256], K/V 캐시 전체: [T_total, 2, 256]
    Q_reshaped = Q_rope.reshape(8, 256)
    
    # 캐시를 numpy 배열로 변환
    K_mat = np.array(K_cache_layer) # [T_total, 2, 256]
    V_mat = np.array(V_cache_layer) # [T_total, 2, 256]
    
    attn_out = np.zeros((8, 256), dtype=np.float32)
    
    # GQA: Q 헤드 8개를 KV 헤드 2개에 4개씩 매핑 (4:1 비율)
    for q_head in range(8):
        kv_head = q_head // 4 
        
        # 1. Attention Score 계산: Q * K^T
        # Q_reshaped[q_head] 는 [256], K_mat[:, kv_head, :] 는 [T_total, 256]
        scores = np.dot(K_mat[:, kv_head, :], Q_reshaped[q_head]) / math.sqrt(256)
        
        # 2. Softmax (안정성을 위해 최대값 빼기)
        scores = scores - np.max(scores)
        probs = np.exp(scores) / np.sum(np.exp(scores))
        
        # 3. Output 계산: Score * V
        # probs 는 [T_total], V_mat[:, kv_head, :] 는 [T_total, 256]
        attn_out[q_head] = np.dot(probs, V_mat[:, kv_head, :])
        
    return attn_out.flatten() # 다시 [2048] 1D 벡터로 쫙 펴서 리턴!

def cpu_sample_token(probs):
    """ 확률값 배열에서 가장 높은 확률의 토큰 인덱스 추출 """
    return np.argmax(probs)