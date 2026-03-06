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
    x = W_embed_real[token_id].astype(np.float32)
    #  Gemma 특수 규칙: 임베딩 스케일링 (sqrt(2048))
    x = x * math.sqrt(2048) 
    return x.astype(np.float16)

# =====================================================================
# 3. PLE (Per-Layer Embedding) 주입
# =====================================================================
#  파라미터로 진짜 PLE 가중치들을 받도록 수정!
def inject_ple(x, ple_proj_down, ple_embed, ple_proj_up, layer_idx=0):
    x_down = np.dot(x, ple_proj_down) 
    x_gated = x_down * ple_embed[0]
    ple_out = np.dot(x_gated, ple_proj_up)
    
    x_final = x + ple_out
    return x_final.astype(np.float16)

def cpu_qk_norm(Q, K, gamma_q, gamma_k):
    """ 
    [CPU] Q와 K에 각각 '헤드별(Per-head)' RMSNorm 적용
    Q: [2048] -> [8, 256]
    K: [512]  -> [2, 256]
    """
    # 1. 포인터 캐스팅하듯 [헤드 수, 256] 형태로 모양(View) 변경
    Q_reshaped = Q.reshape(-1, 256) # 8개의 헤드
    K_reshaped = K.reshape(-1, 256) # 2개의 헤드
    
    # 2. 헤드별(axis=1)로 제곱의 평균 구하기 (keepdims=True로 해야 브로드캐스팅 가능)
    q_mean_sq = np.mean(Q_reshaped.astype(np.float32)**2, axis=1, keepdims=True)
    k_mean_sq = np.mean(K_reshaped.astype(np.float32)**2, axis=1, keepdims=True)
    
    q_inv_sqrt = 1.0 / np.sqrt(q_mean_sq + 1e-6)
    k_inv_sqrt = 1.0 / np.sqrt(k_mean_sq + 1e-6)
    
    # 3. 각 헤드별로 Norm 적용 + 256차원 감마 스케일링
    # Q_reshaped[8, 256] * q_inv_sqrt[8, 1] * gamma_q[256] -> 완벽하게 매핑됨!
    Q_norm = (Q_reshaped * q_inv_sqrt) * gamma_q
    K_norm = (K_reshaped * k_inv_sqrt) * gamma_k
    
    # 4. 다시 1D 벡터 [2048], [512] 로 쫙 펴서 리턴
    return Q_norm.flatten(), K_norm.flatten()

# =====================================================================
# 5. RoPE (회전 위치 인코딩)
# =====================================================================
def cpu_rope(x, pos, theta_base=10000.0):
    dim = 256 
    num_heads = len(x) // dim
    x_reshaped = x.reshape(num_heads, dim)
    out = np.zeros_like(x_reshaped, dtype=np.float32)
    
    for h in range(num_heads):
        half = dim // 2
        for i in range(half):
            #  Gemma/LLaMA 스타일 반갈죽(Half) 회전 로직! (짝수/홀수 아님)
            freq = 1.0 / (theta_base ** ((2 * i) / dim))
            val = pos * freq
            cos_val = math.cos(val)
            sin_val = math.sin(val)
            
            x0 = x_reshaped[h, i]           # 앞부분 절반
            x1 = x_reshaped[h, i + half]    # 뒷부분 절반
            
            out[h, i]        = x0 * cos_val - x1 * sin_val
            out[h, i + half] = x1 * cos_val + x0 * sin_val
            
    return out.flatten().astype(np.float16)

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
    # 캐시를 numpy 배열로 변환하고 3D 텐서로 View 캐스팅!
    K_mat = np.array(K_cache_layer).reshape(-1, 2, 256) # [T_total, 2, 256]
    V_mat = np.array(V_cache_layer).reshape(-1, 2, 256) # [T_total, 2, 256]
    
    attn_out = np.zeros((8, 256), dtype=np.float32)
    
    # GQA: Q 헤드 8개를 KV 헤드 2개에 4개씩 매핑 (4:1 비율)
    for q_head in range(8):
        kv_head = q_head // 4 
        
        # 1. Attention Score 계산: Q * K^T
        # Q_reshaped[q_head] 는 [256], K_mat[:, kv_head, :] 는 [T_total, 256]
        scores = np.dot(K_mat[:, kv_head, :].astype(np.float32), Q_reshaped[q_head].astype(np.float32)) / math.sqrt(256)
        
        #  [Gemma 3 필수] Attention Logit Softcapping: 50.0 * tanh(score / 50.0)
        # 없으면 어텐션 분포가 극단적으로 쏠려서 완전히 엉뚱한 결과 나옴!
        ATTN_SOFTCAP = 50.0
        scores = ATTN_SOFTCAP * np.tanh(scores / ATTN_SOFTCAP)
        
        # 2. Softmax (안정성을 위해 최대값 빼기, float32로 안전하게)
        scores = scores - np.max(scores)
        probs = np.exp(scores.astype(np.float64)) / np.sum(np.exp(scores.astype(np.float64)))
        
        # 3. Output 계산: Score * V
        # probs 는 [T_total], V_mat[:, kv_head, :] 는 [T_total, 256]
        attn_out[q_head] = np.dot(probs, V_mat[:, kv_head, :])
        
    return attn_out.flatten() # 다시 [2048] 1D 벡터로 쫙 펴서 리턴!

def cpu_sample_token(probs):
    """ 확률값 배열에서 가장 높은 확률의 토큰 인덱스 추출 """
    return np.argmax(probs)