"""Gemma 3N E4B integration scaffold for the KV260 v002 NPU.

Scope (v002.1, this commit): structural scaffold only.

What is wired:
- Model identifier and architecture knobs from docs/GEMMA_3N_E4B.md
- A :class:`GemmaPlan` that decomposes one transformer layer into NPU-friendly
  primitives (GEMM matmuls, GEMV projections, activation function via CVO)
  using the v002 ISA from :mod:`sw.runtime.isa`
- An :class:`InferenceSession` shell that opens the UIO mmap, owns the L2
  scratch / shape-pointer state, and exposes a `step()` placeholder

What is NOT yet wired (gated on subsequent commits and v002 design work):
- HP weight streams (S_AXIS_HP[0..3]_WEIGHT) — system_bd.tcl scaffold ties
  them off. AXI DMA wiring is a prerequisite for actual weight matmul.
- ACP fmap / result streams — same gating.
- INT4 quantisation of the model weights — :mod:`sw.quantize` would own
  this; the bridge here just describes the layer plan.
- Tokeniser, KV cache, sampling loop — :mod:`sw.runtime.launcher` will own
  the end-to-end loop once the underlying primitives are real.

The aim of this file is to give the v002.1 release a concrete, testable
seam for the launcher integration in :mod:`sw.runtime.npu` and the future
hardware enablement work, not to run inference today.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


GEMMA_3N_E4B = "google/gemma-3n-e4b"


@dataclass(frozen=True)
class GemmaArch:
    """Static architecture knobs taken from docs/GEMMA_3N_E4B.md.

    Numbers are placeholders pending the real model card cross-check; they
    serve only to size buffer allocations in the dispatch planner.
    """
    name: str = GEMMA_3N_E4B
    num_layers: int = 28
    hidden_dim: int = 3072
    intermediate_dim: int = 8192
    num_heads: int = 16
    num_kv_heads: int = 4
    head_dim: int = 192
    vocab_size: int = 256_000
    rope_theta: float = 10000.0


@dataclass
class LayerPlan:
    """NPU dispatch plan for one Gemma transformer layer.

    Each step is a list of v002 ISA ops to be encoded by :mod:`sw.runtime.isa`
    and pushed through :class:`sw.runtime.uio.NpuMmio`. The runtime fills in
    the actual addresses; this dataclass just describes the call sequence.
    """
    layer_idx: int
    qkv_proj_gemm: bool = True            # (q, k, v) batched matmul
    attn_score_gemv: bool = True          # softmax(QKᵀ / √d)
    attn_combine_gemv: bool = True        # attention @ V
    out_proj_gemm: bool = True            # output projection
    ffn_up_gemm: bool = True              # FFN up projection
    ffn_act_cvo: str = "GELU"             # CVO activation choice
    ffn_down_gemm: bool = True            # FFN down projection
    residual_memcpy: bool = True          # residual stream copy back to L2


@dataclass
class GemmaPlan:
    arch: GemmaArch = field(default_factory=GemmaArch)
    layers: List[LayerPlan] = field(default_factory=list)

    @classmethod
    def for_arch(cls, arch: GemmaArch = GemmaArch()) -> "GemmaPlan":
        return cls(arch=arch, layers=[LayerPlan(i) for i in range(arch.num_layers)])


@dataclass
class InferenceSession:
    """Stub orchestrator. Real impl will hold a :class:`sw.runtime.uio.NpuMmio`,
    a tokeniser handle, and the KV cache."""
    plan: GemmaPlan = field(default_factory=GemmaPlan.for_arch)
    device_path: Optional[str] = None   # set to /dev/uio4 once available

    def open(self) -> "InferenceSession":
        """Acquire the UIO mmap. No-op until the NPU read path is verified."""
        return self

    def close(self) -> None:
        """Release the UIO mmap. No-op until open() is implemented."""

    def step(self, _input_ids: Tuple[int, ...]) -> Tuple[int, ...]:
        """One forward token. Currently raises until the v002.2 hardware enablement
        wires up HP/ACP DMA and the launcher's KV cache lands."""
        raise NotImplementedError(
            "InferenceSession.step() requires HP/ACP DMA wiring (system_bd.tcl "
            "HUMAN REVIEW block) and the KV cache to land. Until then the NPU "
            "accepts control writes but cannot stream weights or return acts."
        )
