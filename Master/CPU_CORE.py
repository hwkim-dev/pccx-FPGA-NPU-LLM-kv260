import numpy as np
import math
from transformers import AutoTokenizer

# =====================================================================
# [가상의 모델 가중치 로드] 
# 실제로는 safetensors나 PyTorch 모델 파일에서 읽어와야 하지만, 
# 지금은 구조를 잡는 거니까 numpy 랜덤 배열로 뼈대만 만들게!
# =====================================================================
vocab_size = 256000
d_model = 2048
ple_dim = 256

print("Loading Model Weights to Host Memory (DDR)...")
W_embed = np.random.randn(vocab_size, d_model).astype(np.float32)

# PLE 가중치 (0번 레이어 예시)
ple_proj_down = np.random.randn(d_model, ple_dim).astype(np.float32)
ple_embed     = np.random.randn(1, ple_dim).astype(np.float32) # T=1
ple_proj_up   = np.random.randn(ple_dim, d_model).astype(np.float32)

# =====================================================================
# 1. Tokenize (문자열 -> 정수 ID 배열)
# =====================================================================
# 기존: 인터넷 연결해서 가져오기
# model_id = "google/gemma-3n-e4b" 

# 변경: 로컬 폴더 경로에서 오프라인으로 바로 가져오기!
model_id = "./local_gemma_3n" 
tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)

def tokenize(text):
    """ [CPU] 입력 프롬프트를 Token ID로 변환 """
    tokens = tokenizer(text, return_tensors="np")["input_ids"][0]
    print(f"[CPU] Tokenized IDs: {tokens}")
    return tokens

# =====================================================================
# 2. Embedding (Token ID -> 2048차원 벡터)
# =====================================================================
def embedding(token_id):
    """ [CPU] Token ID를 인덱스로 삼아 2048차원 벡터 룩업 """
    # C++의 W_embed[token_id] 와 완벽히 동일한 메모리 오프셋 접근!
    # 연산(Math)이 아니라 메모리 읽기(Memory Access) 작업임.
    x = W_embed[token_id] 
    
    # NPU는 16bit(INT16 또는 FP16)를 쓰니까 데이터 타입 캐스팅!
    return x.astype(np.float16)

# =====================================================================
# 3. PLE (Per-Layer Embedding) 주입
# =====================================================================
def inject_ple(x, layer_idx=0):
    """ [CPU] Gemma 3N 특화: Low-Rank 레이어 임베딩 주입 """
    # x: [2048], ple_proj_down: [2048, 256]
    # np.dot(벡터, 행렬) -> [256] 차원 벡터 나옴
    x_down = np.dot(x, ple_proj_down) 
    
    # 요소별 곱셈 (Gating) 후 다시 2048로 Up Projection
    x_gated = x_down * ple_embed[0]
    ple_out = np.dot(x_gated, ple_proj_up)
    
    # 기존 입력 x에 Residual Add
    x_final = x + ple_out
    
    return x_final.astype(np.float16)

def cpu_qk_norm(Q, K, gamma_q, gamma_k):
    """ 
    [CPU] Q와 K에 각각 RMSNorm 적용 (디코드 단계라 Q, K는 1D 벡터)
    Q: [2048] (8헤드 x 256차원)
    K: [512]  (2헤드 x 256차원)
    """
    # 분모 구하기 (eps는 보통 1e-6)
    q_inv_sqrt = 1.0 / np.sqrt(np.mean(Q**2) + 1e-6)
    k_inv_sqrt = 1.0 / np.sqrt(np.mean(K**2) + 1e-6)
    
    # 요소별(Element-wise) 스케일링
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