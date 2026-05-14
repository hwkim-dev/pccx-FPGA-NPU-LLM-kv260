#!/usr/bin/env python3
"""Slice Gemma 3N E4B mmap INT4 weights to E2B variant for KV260 deploy.

E4B → E2B: intermediate_size 16384 → 8192 in W_gate/W_up/W_down across all 35 layers.
language_model tensors only (audio_tower / vision_tower omitted for text chat).
"""
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np


SRC = Path("/home/hwkim/Desktop/pccx-kv260-models/gemma3n-e4b")
DST = Path("/tmp/gemma3n-e2b-deploy")

# E2B target
TARGET_INTERMEDIATE = 8192   # was 16384
NUM_LAYERS = 35


def slice_rows_int4(packed_path, scale_path, out_packed, out_scale, target_rows):
    """W_gate / W_up : slice the M_out dim (rows) to target_rows."""
    packed = np.load(packed_path, mmap_mode="r")
    scale = np.load(scale_path, mmap_mode="r")
    new_packed = np.ascontiguousarray(packed[:target_rows, :])
    new_scale = np.ascontiguousarray(scale[:target_rows])
    np.save(out_packed, new_packed)
    np.save(out_scale, new_scale)
    return new_packed.nbytes + new_scale.nbytes


def slice_cols_half_int4(packed_path, scale_path, out_packed, out_scale, target_intermediate):
    """W_down : packed col dim is intermediate / 2 (since INT4 packs 2 per byte).
    Original packed shape (2048, 8192) — slice to (2048, target_intermediate//2)."""
    packed = np.load(packed_path, mmap_mode="r")
    scale = np.load(scale_path, mmap_mode="r")
    half = target_intermediate // 2
    new_packed = np.ascontiguousarray(packed[:, :half])
    new_scale = np.ascontiguousarray(scale)  # per-row scale, unchanged
    np.save(out_packed, new_packed)
    np.save(out_scale, new_scale)
    return new_packed.nbytes + new_scale.nbytes


def copy_tensor(src_path, dst_path):
    """Copy unchanged tensor (real copy, not symlink, so scp transfers the file)."""
    if dst_path.exists():
        return dst_path.stat().st_size
    shutil.copy2(src_path, dst_path)
    return dst_path.stat().st_size


def main():
    t0 = time.time()
    if DST.exists():
        print(f"[clean] removing existing {DST}")
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    mmap_src = SRC / "mmap_weights"
    mmap_dst = DST / "mmap_weights"
    mmap_dst.mkdir()

    # 1. Copy tokenizer + config (these are symlinks in src — copy real content)
    for name in [
        "chat_template.jinja",
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "preprocessor_config.json",
        "processor_config.json",
    ]:
        src = SRC / name
        if not src.exists():
            print(f"[skip] {name} not in src")
            continue
        # Follow symlinks to copy real content
        real = src.resolve()
        shutil.copy2(real, DST / name)

    print(f"[done] tokenizer + config in {time.time()-t0:.1f}s")

    # 2. Walk language_model files in mmap_weights, slice MLP, copy rest
    total_bytes = 0
    n_files = 0
    n_sliced = 0
    lm_files = sorted(p.name for p in mmap_src.glob("model.language_model.*"))
    print(f"[scan] {len(lm_files)} language_model files")

    mlp_re = re.compile(r"^model\.language_model\.layers\.(\d+)\.mlp\.(gate_proj|up_proj|down_proj)\.weight(\.scale)?\.npy$")

    for fname in lm_files:
        src_p = mmap_src / fname
        dst_p = mmap_dst / fname
        m = mlp_re.match(fname)
        if m is None:
            # non-MLP: copy as-is (resolve symlink)
            real = src_p.resolve()
            total_bytes += copy_tensor(real, dst_p)
            n_files += 1
            continue

        # MLP — defer to slicing pass: skip here, handled below
        continue

    # 3. Slice MLP per layer
    for layer in range(NUM_LAYERS):
        for kind in ("gate_proj", "up_proj"):
            packed_src = (mmap_src / f"model.language_model.layers.{layer}.mlp.{kind}.weight.npy").resolve()
            scale_src = (mmap_src / f"model.language_model.layers.{layer}.mlp.{kind}.weight.scale.npy").resolve()
            packed_dst = mmap_dst / f"model.language_model.layers.{layer}.mlp.{kind}.weight.npy"
            scale_dst = mmap_dst / f"model.language_model.layers.{layer}.mlp.{kind}.weight.scale.npy"
            total_bytes += slice_rows_int4(packed_src, scale_src, packed_dst, scale_dst, TARGET_INTERMEDIATE)
            n_sliced += 2

        packed_src = (mmap_src / f"model.language_model.layers.{layer}.mlp.down_proj.weight.npy").resolve()
        scale_src = (mmap_src / f"model.language_model.layers.{layer}.mlp.down_proj.weight.scale.npy").resolve()
        packed_dst = mmap_dst / f"model.language_model.layers.{layer}.mlp.down_proj.weight.npy"
        scale_dst = mmap_dst / f"model.language_model.layers.{layer}.mlp.down_proj.weight.scale.npy"
        total_bytes += slice_cols_half_int4(packed_src, scale_src, packed_dst, scale_dst, TARGET_INTERMEDIATE)
        n_sliced += 2

        if (layer + 1) % 5 == 0:
            print(f"[sliced] layer {layer+1}/{NUM_LAYERS}  elapsed {time.time()-t0:.1f}s")

    # 4. Manifest
    manifest = {
        "model_name": "gemma3n-e2b-sliced",
        "source": str(SRC),
        "target_intermediate": TARGET_INTERMEDIATE,
        "source_intermediate": 16384,
        "num_layers": NUM_LAYERS,
        "non_mlp_files_copied": n_files,
        "mlp_files_sliced": n_sliced,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "language_model_only": True,
        "skipped_modalities": ["audio_tower", "vision_tower"],
    }
    with open(DST / "pccx_kv260_slice_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n[final] total {total_bytes/1024/1024/1024:.2f} GB across {n_files} copies + {n_sliced} slices")
    print(f"[final] elapsed {time.time()-t0:.1f}s")
    print(f"[final] output at {DST}")


if __name__ == "__main__":
    main()
