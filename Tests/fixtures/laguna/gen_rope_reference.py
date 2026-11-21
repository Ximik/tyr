#!/usr/bin/env python
"""Generate ground-truth rotary-table fixtures for Laguna rope tests.

Computes both rotary tables (plain rope for sliding-window layers, YaRN for
full-attention layers) in torch float64 on CPU following
`modeling_laguna.py` of `poolside/Laguna-XS-2.1` (revision c5f36269) + HF `_compute_yarn_parameters`,
plus end-to-end apply fixtures (HF `apply_rotary_pos_emb`, GLM-style
non-interleaved rotate_half). Saves everything as fp32 safetensors.

Run from the repo root:
    .venv-gpu/bin/python Tests/fixtures/laguna/gen_rope_reference.py
"""

import json
import math
import struct

import torch

MAXLEN = 1024

# Rope parameters under test.
THETA_SLIDING = 10000.0
DIM_SLIDING = 128  # partial_rotary_sliding 1.0 * head_dim 128
THETA_FULL = 500000.0
DIM_FULL = 64  # partial_rotary_full 0.5 * head_dim 128
YARN_FACTOR = 32.0
YARN_ORIG_MAX = 8192
YARN_BETA_FAST = 32.0
YARN_BETA_SLOW = 1.0
YARN_ATTN_FACTOR = 1.3465735902799727

DT = torch.float64


def plain_inv_freq(dim: int, base: float) -> torch.Tensor:
    """inv_freq[j] = 1 / base^(2j/dim), j in [0, dim/2)."""
    return 1.0 / (base ** (torch.arange(0, dim, 2, dtype=DT) / dim))


def yarn_inv_freq(dim: int, base: float):
    """HF transformers `_compute_yarn_parameters` (truncated correction range)."""
    half = dim // 2
    pos_freqs = base ** (torch.arange(0, dim, 2, dtype=DT) / dim)
    inv_interp = 1.0 / (YARN_FACTOR * pos_freqs)
    inv_extrap = 1.0 / pos_freqs
    low = math.floor(
        dim * math.log(YARN_ORIG_MAX / (YARN_BETA_FAST * 2 * math.pi)) / (2 * math.log(base))
    )
    high = math.ceil(
        dim * math.log(YARN_ORIG_MAX / (YARN_BETA_SLOW * 2 * math.pi)) / (2 * math.log(base))
    )
    low = max(low, 0)
    high = min(high, dim - 1)
    ramp = ((torch.arange(half, dtype=DT) - low) / max(high - low, 1)).clamp(0.0, 1.0)
    inv_freq = inv_interp * ramp + inv_extrap * (1.0 - ramp)
    return inv_freq, low, high


def make_table(inv_freq: torch.Tensor, scale: float):
    """[maxLen, half] cos/sin tables: table[p, j] = cos|sin(p * inv_freq[j]) * scale."""
    t = torch.arange(MAXLEN, dtype=DT)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos() * scale, freqs.sin() * scale


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_partial(q: torch.Tensor, cos_half: torch.Tensor, sin_half: torch.Tensor, rotary_dim: int):
    """HF `apply_rotary_pos_emb` for q laid out as [batch, seq, heads, head_dim].

    cos_half/sin_half are [seq, rotary_dim//2] (the tyr table layout); HF's emb
    is cat(freqs, freqs), i.e. the half tables duplicated along the last dim.
    """
    cos = torch.cat([cos_half, cos_half], dim=-1)  # [seq, rotary_dim]
    sin = torch.cat([sin_half, sin_half], dim=-1)
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    emb = q_rot * cos[None, :, None, :] + rotate_half(q_rot) * sin[None, :, None, :]
    return torch.cat([emb, q_pass], dim=-1)


def save_safetensors(path: str, tensors: dict):
    """Minimal safetensors writer (F32 only) — avoids the safetensors dependency."""
    header = {}
    blobs = []
    offset = 0
    for name, t in tensors.items():
        t = t.contiguous().to(torch.float32)
        data = t.numpy().tobytes(order="C")
        header[name] = {
            "dtype": "F32",
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(data)],
        }
        blobs.append(data)
        offset += len(data)
    hj = json.dumps(header).encode("utf-8")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hj)))
        f.write(hj)
        for b in blobs:
            f.write(b)


def main():
    inv_sliding = plain_inv_freq(DIM_SLIDING, THETA_SLIDING)
    sliding_cos, sliding_sin = make_table(inv_sliding, 1.0)

    inv_full, low, high = yarn_inv_freq(DIM_FULL, THETA_FULL)
    full_cos, full_sin = make_table(inv_full, YARN_ATTN_FACTOR)

    print(f"YaRN correction range: low={low} high={high}")
    print(f"YaRN inv_freq[0..4]: {[float(v) for v in inv_full[:5]]}")
    print(f"sliding table shape: {tuple(sliding_cos.shape)}, full table shape: {tuple(full_cos.shape)}")

    # End-to-end apply fixtures (fixed seed, fp32 input).
    g = torch.Generator().manual_seed(0)
    q = torch.randn(1, 16, 4, 128, generator=g, dtype=DT)
    q_out_full = apply_rotary_partial(q, full_cos[:16], full_sin[:16], DIM_FULL)
    q_out_sliding = apply_rotary_partial(q, sliding_cos[:16], sliding_sin[:16], DIM_SLIDING)
    # Decode-step fixture: single position 5 through the partial YaRN path.
    q_out_full_pos5 = apply_rotary_partial(q[:, 5:6], full_cos[5:6], full_sin[5:6], DIM_FULL)

    tensors = {
        "sliding_cos": sliding_cos,
        "sliding_sin": sliding_sin,
        "full_cos": full_cos,
        "full_sin": full_sin,
        "yarn_inv_freq": inv_full,
        "q_in": q,
        "q_out_full_yarn": q_out_full,
        "q_out_sliding": q_out_sliding,
        "q_out_full_yarn_pos5": q_out_full_pos5,
    }
    tensors = {k: v.contiguous().to(torch.float32) for k, v in tensors.items()}
    out = "Tests/fixtures/laguna/rope_reference.safetensors"
    save_safetensors(out, tensors)
    print(f"saved {out}")
    for k, v in tensors.items():
        print(f"  {k}: {tuple(v.shape)} {v.dtype}")


if __name__ == "__main__":
    main()
