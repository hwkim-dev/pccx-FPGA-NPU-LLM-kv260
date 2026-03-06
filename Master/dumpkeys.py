"""
실제 모델 키 이름 전부 출력 - 아키텍처 파악용
"""
import os, sys
import torch
from safetensors.torch import load_file

base_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(base_dir, "local_gemma_3n")

print("=" * 70)
print("모든 safetensor 파일의 키 이름 덤프")
print("=" * 70)

all_keys = {}
for filename in sorted(os.listdir(model_dir)):
    if filename.endswith(".safetensors"):
        print(f"\n {filename}")
        pt = load_file(os.path.join(model_dir, filename))
        for k, v in sorted(pt.items()):
            all_keys[k] = v.shape
            print(f"  {k:80s}  {str(tuple(v.shape)):20s}  {v.dtype}")

print("\n" + "=" * 70)
print("레이어 0 키 이름만 필터링 (구조 파악)")
print("=" * 70)
for k in sorted(all_keys.keys()):
    if "layers.0" in k or ("layer" not in k and "layers" not in k):
        print(f"  {k:80s}  {str(tuple(all_keys[k]))}")

print("\n" + "=" * 70)
print("'altup', 'ple', 'per_layer', 'lauren', 'griffin' 포함 키")
print("=" * 70)
keywords = ["altup", "ple", "per_layer", "griffin", "recurrent",
            "temporal", "conv", "audio", "vision", "image"]
for k in sorted(all_keys.keys()):
    for kw in keywords:
        if kw.lower() in k.lower():
            print(f"  [{kw}] {k}  {tuple(all_keys[k])}")
            break