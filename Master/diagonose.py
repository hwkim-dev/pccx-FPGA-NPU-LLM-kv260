import numpy as np
import CPU_CORE
import safeTensor
import sys

def rms_norm(x, gamma):
    x_f64 = x.astype(np.float64)
    rms = np.sqrt(np.mean(x_f64**2) + 1e-6)
    return (x / rms).astype(np.float32) * gamma

def check(tag, arr):
    has_nan = np.isnan(arr).any()
    has_inf = np.isinf(arr).any()
    max_val = np.max(np.abs(arr))
    mean_val = np.mean(arr)
    print(f"{tag:<32} | max_abs: {max_val:10.4f} | mean: {mean_val:10.4f}")
    
    if has_nan or has_inf or max_val > 80000:
        print(f"\n🚨🚨🚨 [FATAL ERROR] 데이터 오염 발생: {tag} 🚨🚨🚨")
        sys.exit(1)

def gelu(x):
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * (x**3))))

def sigmoid(x):
    x_clipped = np.clip(x, -88.0, 88.0)
    return 1.0 / (1.0 + np.exp(-x_clipped))

def softmax(x):
    x_safe = x - np.max(x)
    e_x = np.exp(x_safe)
    return e_x / np.sum(e_x)

def main():
    print("🚀 Gemma 3N [CPU Golden Reference] Debugging Start!")
    W_embed, W_ple, norm_ple, W_ple_proj, altup_projs, altup_unprojs, W_final_norm, W_lm_head, W = safeTensor.load_local_weights()

    K_cache = [[] for _ in range(35)]
    V_cache = [[] for _ in range(35)]

    prompt = "안녕 하세요"
    input_tokens = CPU_CORE.tokenize(prompt)
    
    for pos, token_id in enumerate(input_tokens):
        is_last = (pos == len(input_tokens) - 1)
        if is_last:
            print(f"\n🔥 [Last Token Debugging] Token ID: {token_id} (pos={pos})")
        
        x0 = CPU_CORE.embedding(token_id, W_embed)
        xs = np.zeros((4, 2048), dtype=np.float32)
        xs[0] = x0
        for k in range(3):
            xs[k+1] = np.dot(x0, altup_projs[k])
            
        ple_all = W_ple[min(token_id, W_ple.shape[0]-1)].astype(np.float32)

        for i in range(35):
            # --- 1. AltUp Predict ---
            r_in = rms_norm(xs[0], W["altup_rn"][i])
            router_logits = np.dot(r_in, W["altup_router"][i])
            r_weights = softmax(router_logits)
            
            coef_mat = np.dot(W["altup_pred"][i], r_weights).reshape(4, 4)
            xs_pred = np.dot(coef_mat, xs)

            # --- 2. Stream 0 본체 훈련 ---
            # 🔥 핵심 복구 1: 섞여서 나온 xs_pred[0]가 훈련장에 들어가야 피드백 루프가 완성됨!
            x = xs_pred[0].copy()
            
            # ① PLE
            ple_slice = ple_all[i*256 : (i+1)*256]
            ple_normed = rms_norm(ple_slice, norm_ple)
            gate = sigmoid(np.dot(x, W["ple_gate"][i]))            
            projected = rms_norm(np.dot(gate * ple_normed, W["ple_proj"][i]), W["ple_post_ln"][i])
            x = x + projected
            
            # ② Attention 
            x_n = rms_norm(x, W["input_ln"][i])
            Q = np.dot(x_n, W["W_q"][i])
            K = np.dot(x_n, W["W_k"][i])
            V = np.dot(x_n, W["W_v"][i])
            
            Q, K = CPU_CORE.cpu_qk_norm(Q, K, W["gamma_q"][i], W["gamma_k"][i])
            Q = CPU_CORE.cpu_rope(Q, pos=pos, theta_base=1_000_000)
            K = CPU_CORE.cpu_rope(K, pos=pos, theta_base=1_000_000)
            
            CPU_CORE.cpu_update_kv_cache(K, V, i, K_cache, V_cache)
            attn_out = CPU_CORE.cpu_gqa(Q, K_cache[i], V_cache[i])
            x = x + rms_norm(np.dot(attn_out, W["W_o"][i]), W["post_attn_ln"][i])
            
            # ③ FFN
            x_n2 = rms_norm(x, W["pre_ffn_ln"][i])
            gate_out = np.dot(x_n2, W["W_gate"][i])
            up_out   = np.dot(x_n2, W["W_up"][i])
            hidden   = gelu(gate_out) * up_out
            x = x + rms_norm(np.dot(hidden, W["W_down"][i]), W["post_ffn_ln"][i])
            
            # ④ LAuReL
            bottleneck = np.dot(x, W["laurel_left"][i])
            expanded = np.dot(bottleneck, W["laurel_right"][i])
            x = x + rms_norm(expanded, W["laurel_norm"][i])

            # --- 3. AltUp Correct ---
            # 🔥 핵심 복구 2: 변화량은 훈련 전(xs_pred[0]) 대비 훈련 후(x)의 차이!
            delta = x - xs_pred[0] 
            scaled = delta * W["altup_scale"][i]
            
            # 🔥 핵심 복구 3: 공유기 비율은 아까 Predict에서 썼던 r_weights를 그대로 재활용!
            corr_coefs = np.dot(W["altup_corr"][i], r_weights)
            
            xs_new = xs_pred.copy()
            for k in range(4):
                xs_new[k] += corr_coefs[k] * scaled
            
            xs = xs_new
            
            if is_last: 
                check(f"L{i} Post-AltUp (Main xs[0])", xs[0])
                check(f"L{i} Post-AltUp (Shadow xs[1])", xs[1])
            
        if is_last:
            x_final = xs[0].copy()
            for k in range(3):
                x_final += np.dot(xs[k+1], altup_unprojs[k])
                
            check("FINAL Un-embed x", x_final)
            
            x_final = rms_norm(x_final, W_final_norm)
            logits = np.dot(x_final, W_lm_head)
            
            FINAL_SOFTCAP = 30.0
            logits_capped = FINAL_SOFTCAP * np.tanh(logits / FINAL_SOFTCAP)
            
            logits_safe = logits_capped - np.max(logits_capped)
            probs = np.exp(logits_safe) / np.sum(np.exp(logits_safe))
            
            next_token = CPU_CORE.cpu_sample_token(probs)
            print(f"\n✅ Generated Next Token ID: {next_token}")
            print(f"🎉 Decoded Text: {CPU_CORE.tokenizer.decode([next_token])}")

if __name__ == "__main__":
    main()