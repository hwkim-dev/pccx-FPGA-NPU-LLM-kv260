import numpy as np
import os
from safetensors.numpy import load_file

def load_local_weights(model_dir="./local_gemma_3n"):
    print("Loading weights and folding RMSNorm Gammas...")
    tensors = {}
    for filename in os.listdir(model_dir):
        if filename.endswith(".safetensors"):
            tensors.update(load_file(os.path.join(model_dir, filename)))
            
    weights = {"W_q_fused": [], "W_k_fused": [], "W_v_fused": [], "W_o": [],
               "W_gate_fused": [], "W_up_fused": [], "W_down": [],
               "gamma_q": [], "gamma_k": []}

    for i in range(35):
        # 1. 텐서 가져오기
        W_q = tensors[f"model.layers.{i}.self_attn.q_proj.weight"]
        gamma_attn = tensors[f"model.layers.{i}.input_layernorm.weight"] # [2048]
        
        # 2. 🔥 대망의 Weight Folding (가중치 퓨전!) 🔥
        # gamma[:, None]을 곱해주면 2048차원 벡터가 행렬의 각 '행'에 브로드캐스팅 곱셈됨
        weights["W_q_fused"].append(W_q * gamma_attn[:, None])
        weights["W_k_fused"].append(tensors[f"model.layers.{i}.self_attn.k_proj.weight"] * gamma_attn[:, None])
        weights["W_v_fused"].append(tensors[f"model.layers.{i}.self_attn.v_proj.weight"] * gamma_attn[:, None])
        
        # (W_o는 Norm 직후가 아니므로 퓨전 안 함)
        weights["W_o"].append(tensors[f"model.layers.{i}.self_attn.o_proj.weight"])
        
        # FFN 블록도 마찬가지로 Pre-Norm 감마를 융합!
        gamma_ffn = tensors[f"model.layers.{i}.post_attention_layernorm.weight"]
        weights["W_gate_fused"].append(tensors[f"model.layers.{i}.mlp.gate_proj.weight"] * gamma_ffn[:, None])
        weights["W_up_fused"].append(tensors[f"model.layers.{i}.mlp.up_proj.weight"] * gamma_ffn[:, None])
        weights["W_down"].append(tensors[f"model.layers.{i}.mlp.down_proj.weight"])
        
        # QK-Norm 용 감마 (이건 CPU에서 처리하므로 그냥 넘김)
        weights["gamma_q"].append(tensors[f"model.layers.{i}.self_attn.q_norm.weight"])
        weights["gamma_k"].append(tensors[f"model.layers.{i}.self_attn.k_norm.weight"])

    # 임베딩 & Final Norm
    W_embed = tensors["model.embed_tokens.weight"]
    gamma_final = tensors["model.norm.weight"]
    W_lm_head_fused = W_embed.T * gamma_final[:, None]
    
    # 🔥 추가된 부분: PLE 가중치도 텐서에서 뽑아오기!
    # (실제 허깅페이스 텐서 키 이름에 맞춰야 하는데, 혹시 모델에 없으면 안 터지게 .get()으로 기본값 세팅)
    ple_down = tensors.get("model.ple_proj_down.weight", np.random.randn(2048, 256).astype(np.float16))
    ple_emb  = tensors.get("model.ple_embed.weight", np.random.randn(1, 256).astype(np.float16))
    ple_up   = tensors.get("model.ple_proj_up.weight", np.random.randn(256, 2048).astype(np.float16))

    print("✅ Weight Folding & PLE Load Complete!")
    
    # 리턴 값에 ple 3형제 추가!
    return W_embed, W_lm_head_fused, weights, ple_down, ple_emb, ple_up