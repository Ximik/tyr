#!/usr/bin/env python
"""Generate Laguna MoE block ground-truth fixtures for Tests/RunLagunaMoe.

Builds a small MoE (hidden=64, 8 experts, top-2, moeInt=32, sharedInt=32,
scaling=2.5, norm_topk_prob=True) with random NVFP4-packed expert banks, a
random sigmoid router (with e_score_correction_bias), random shared-expert
weights and a fixed input x [5, 64], then computes the expected output with a
PyTorch reference implementing the HF math (`modeling_laguna.py` of
`poolside/Laguna-XS-2.1` (revision c5f36269): LagunaTopKRouter/LagunaExperts/
LagunaSparseMoeBlock).

The one deliberate spec-vs-HF difference: router logits use an FP32 matmul
(cast first, then matmul), matching the Lean implementation's task spec;
HF does F.linear(bf16).float(). This is far below the test tolerance.

Run with: .venv-gpu/bin/python Tests/fixtures/laguna/gen_moe_fixtures.py
"""
import os

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "moe.safetensors")

HIDDEN = 64
EXPERTS = 8
TOP_K = 2
MOE_INT = 32
SHARED_INT = 32
SCALING = 2.5
TOKENS = 5

E2M1_MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)


def e2m1_decode(nibbles: torch.Tensor) -> torch.Tensor:
    n = nibbles.to(torch.int32)
    sign = (n >> 3) & 1
    mag = E2M1_MAGS[n & 7]
    return torch.where(sign.bool(), -mag, mag)


def dequant_bank(packed: torch.Tensor, scales: torch.Tensor, global_scale: torch.Tensor) -> torch.Tensor:
    """[E, out, in/2] U8 + [E, out, in/16] F8 + [E] F32 -> F32 [E, out, in] (reference op order)."""
    e = packed.shape[0]
    b = packed.to(torch.int32)
    lo = e2m1_decode(b & 0xF)
    hi = e2m1_decode((b >> 4) & 0xF)
    vals = torch.stack((lo, hi), dim=-1).flatten(-2)
    scale_full = (scales.to(torch.float32) / global_scale.view(e, 1, 1)).repeat_interleave(16, dim=-1)
    return vals * scale_full


def make_expert_bank(out_f: int, in_f: int, g: torch.Generator):
    packed = torch.randint(0, 256, (EXPERTS, out_f, in_f // 2), generator=g, dtype=torch.uint8)
    scales = (torch.rand((EXPERTS, out_f, in_f // 16), generator=g) * 0.10 + 0.02).to(torch.float8_e4m3fn)
    global_scale = torch.rand((EXPERTS,), generator=g) * 0.4 + 0.8
    return packed, scales, global_scale


def main():
    torch.manual_seed(20240707)
    g = torch.Generator().manual_seed(987654)

    # Router (bf16 weight like the checkpoint; f32 selection bias).
    router_weight = (torch.randn((EXPERTS, HIDDEN), generator=g) * 0.08).to(torch.bfloat16)
    router_bias = (torch.rand((EXPERTS,), generator=g) * 0.2 - 0.1).to(torch.float32)

    # Packed expert banks.
    gate_packed, gate_scales, gate_global = make_expert_bank(MOE_INT, HIDDEN, g)
    up_packed, up_scales, up_global = make_expert_bank(MOE_INT, HIDDEN, g)
    down_packed, down_scales, down_global = make_expert_bank(HIDDEN, MOE_INT, g)

    # Shared expert (dense SwiGLU MLP).
    shared_gate = (torch.randn((SHARED_INT, HIDDEN), generator=g) * 0.05).to(torch.bfloat16)
    shared_up = (torch.randn((SHARED_INT, HIDDEN), generator=g) * 0.05).to(torch.bfloat16)
    shared_down = (torch.randn((HIDDEN, SHARED_INT), generator=g) * 0.05).to(torch.bfloat16)

    # Input.
    x = torch.randn((TOKENS, HIDDEN), generator=g).to(torch.bfloat16)

    # ---------------- reference forward (HF math) ----------------
    # Router: FP32 logits (spec) -> sigmoid -> biased selection -> unbiased weights.
    logits = x.to(torch.float32) @ router_weight.to(torch.float32).t()
    scores = torch.sigmoid(logits)
    selection = scores + router_bias
    _, top_idx = torch.topk(selection, TOP_K, dim=-1)
    weights = scores.gather(-1, top_idx)
    weights = weights / weights.sum(dim=-1, keepdim=True)
    weights = weights.to(x.dtype)  # bf16

    # Dequantize banks to bf16 (matches Lean dequantMatrix/dequantBank output).
    gate_w = dequant_bank(gate_packed, gate_scales, gate_global).to(torch.bfloat16)
    up_w = dequant_bank(up_packed, up_scales, up_global).to(torch.bfloat16)
    down_w = dequant_bank(down_packed, down_scales, down_global).to(torch.bfloat16)

    # Per-expert dispatch in ascending expert order (HF LagunaExperts.forward).
    final = torch.zeros_like(x)
    for e in range(EXPERTS):
        mask = top_idx == e  # [tokens, k]
        if not bool(mask.any()):
            continue
        tok, slot = mask.nonzero(as_tuple=True)
        xe = x[tok]
        h = F.silu(F.linear(xe, gate_w[e])) * F.linear(xe, up_w[e])
        oe = F.linear(h, down_w[e])
        oe = oe * weights[tok, slot].unsqueeze(-1)
        final.index_add_(0, tok, oe.to(final.dtype))

    shared = F.linear(F.silu(F.linear(x, shared_gate)) * F.linear(x, shared_up), shared_down)
    y = final * SCALING + shared

    tensors = {
        "router_weight": router_weight.contiguous(),
        "router_bias": router_bias.contiguous(),
        "gate_packed": gate_packed.contiguous(),
        "gate_scales": gate_scales.contiguous(),
        "gate_global": gate_global.contiguous(),
        "up_packed": up_packed.contiguous(),
        "up_scales": up_scales.contiguous(),
        "up_global": up_global.contiguous(),
        "down_packed": down_packed.contiguous(),
        "down_scales": down_scales.contiguous(),
        "down_global": down_global.contiguous(),
        "shared_gate": shared_gate.contiguous(),
        "shared_up": shared_up.contiguous(),
        "shared_down": shared_down.contiguous(),
        "x": x.contiguous(),
        "y_expected": y.to(torch.float32).contiguous(),
        "top_idx_expected": top_idx.to(torch.int64).contiguous(),
        "weights_expected": weights.to(torch.float32).contiguous(),
    }
    save_file(tensors, OUT)
    print(f"wrote {OUT}")
    for k, v in tensors.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype}")
    print(f"  y magnitude: max|y|={y.abs().max().item():.4f} mean|y|={y.abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
