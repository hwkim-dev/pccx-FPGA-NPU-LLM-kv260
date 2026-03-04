import numpy as np
import MMIO
import CPU_CORE
import NPU_CORE
import safeTensor

def main():
    print("🚀 Gemma 3N NPU Acceleration Start!")
    
    # 1. 퓨전된 가중치 + 🔥 PLE 가중치까지 한 번에 로드!
    W_embed, W_lm_head_fused, weights, ple_down, ple_emb, ple_up = safeTensor.load_local_weights()

    K_cache = [[] for _ in range(35)]
    V_cache = [[] for _ in range(35)]

    prompt = "안녕 하세요"
    input_tokens = CPU_CORE.tokenize(prompt)
    x = CPU_CORE.embedding(input_tokens[-1], W_embed)
    
    # 2. 이제 변수가 생겼으니 에러 안 터짐!
    x = CPU_CORE.inject_ple(x, ple_down, ple_emb, ple_up)
    
    # 2. 파이프라인 시작!
    for layer in range(35):
        # ---------------------------------------------------------
        # [A] Attention Block
        # ---------------------------------------------------------
        mean_sq = np.mean(x**2)
        
        # 🔥 원본 x를 그대로 넣어도 NPU가 inv_sqrt를 곱하고 Fused Weight랑 연산해서 완벽한 Norm 결과가 나옴!
        Q = NPU_CORE.npu_matmul(x, weights["W_q_fused"][layer], mean_sq)
        K = NPU_CORE.npu_matmul(x, weights["W_k_fused"][layer], mean_sq)
        V = NPU_CORE.npu_matmul(x, weights["W_v_fused"][layer], mean_sq)
        
        Q_norm, K_norm = CPU_CORE.cpu_qk_norm(Q, K, weights["gamma_q"][layer], weights["gamma_k"][layer])
        Q_rope, K_rope = CPU_CORE.cpu_rope(Q_norm, pos=10, theta_base=10000)
        
        CPU_CORE.cpu_update_kv_cache(K_rope, V, layer, K_cache, V_cache)
        attn_out = CPU_CORE.cpu_gqa(Q_rope, K_cache[layer], V_cache[layer])
        
        # W_o는 퓨전할 게 없으므로 mean_sq=1.0 (스케일링 무시)
        attn_proj = NPU_CORE.npu_matmul(attn_out, weights["W_o"][layer], mean_sq=1.0)
        x = x + attn_proj  
        
        # ---------------------------------------------------------
        # [B] FFN Block
        # ---------------------------------------------------------
        mean_sq_ffn = np.mean(x**2)
        
        # 🔥 퓨전 가중치 + 1-Cycle GeLU 가속!
        hidden = NPU_CORE.npu_matmul_gelu(x, weights["W_gate_fused"][layer], mean_sq_ffn)
        up = NPU_CORE.npu_matmul(x, weights["W_up_fused"][layer], mean_sq_ffn)
        
        hidden = hidden * up
        
        ffn_out = NPU_CORE.npu_matmul(hidden, weights["W_down"][layer], mean_sq=1.0)
        x = x + ffn_out    
        
    # ---------------------------------------------------------
    # [C] Final Output (LM Head)
    # ---------------------------------------------------------
    mean_sq_final = np.mean(x**2)
    # Final Gamma도 퓨전되어 있으므로 바로 W_lm_head_fused 때림!
    logits = NPU_CORE.npu_matmul(x, W_lm_head_fused, mean_sq_final)
    probs = NPU_CORE.npu_softmax(logits)
    
    next_token = CPU_CORE.cpu_sample_token(probs)
    print(f"✅ Generated Next Token: {next_token}")

if __name__ == "__main__":
    main()