#!/usr/bin/env python3
"""Benchmark the vendored ThunderKittens H100 MHA extension.

This is intentionally narrow: it measures the upstream ThunderKittens PyTorch
extension on the shapes that can be compared to Tyr's current H100 bridge.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable

import torch


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TK_DIR = ROOT / "external" / "git" / "ThunderKittens" / "kernels" / "attention" / "mha_h100"


def repo_relative(path: Path) -> str:
    resolved = path.resolve()
    return str(resolved.relative_to(ROOT)) if resolved.is_relative_to(ROOT) else str(path)


CASES = {
    "native_dense_128x64": (1, 1, 1, 128, 64),
    "native_dense_768x64": (1, 1, 1, 768, 64),
}


def parse_csv(value: str) -> list[str]:
    if value == "native_now":
        return ["native_dense_128x64", "native_dense_768x64"]
    if value == "tk_now":
        return ["native_dense_768x64"]
    return [part for part in value.split(",") if part]


def median(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[len(ordered) // 2]


def tensor_mae(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().mean().item()


def tensor_max(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def sdpa_once(
    q_base: torch.Tensor,
    k_base: torch.Tensor,
    v_base: torch.Tensor,
    dO_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q = q_base.detach().clone().requires_grad_(True)
    k = k_base.detach().clone().requires_grad_(True)
    v = v_base.detach().clone().requires_grad_(True)
    out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=False)
    (out.float() * dO_base.float()).sum().backward()
    torch.cuda.synchronize()
    return out.detach(), q.grad.detach(), k.grad.detach(), v.grad.detach()


def time_cuda_event(fn: Callable[[], None], warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    stop = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    stop.record()
    torch.cuda.synchronize()
    return start.elapsed_time(stop) / max(1, iters)


def tk_supported(seq: int, head_dim: int) -> bool:
    # The vendored benchmark starts at N=768. Smaller rows either launch with a
    # zero grid (N=128) or fail parity in the current upstream extension (N=256).
    return head_dim in (64, 128) and seq >= 768 and seq % 256 == 0


def load_tk_module(tk_dir: Path):
    sys.path.insert(0, str(tk_dir))
    import _C as tk  # type: ignore

    return tk


def build_summary(
    case_id: str,
    backend: str,
    status: str,
    seq: int,
    head_dim: int,
    p50_ms: float,
    speedup_vs_sdpa: float,
    metrics: dict[str, float | bool],
    route: str,
) -> dict[str, object]:
    row: dict[str, object] = {
        "event": "summary",
        "caseId": case_id,
        "backendExecuted": backend,
        "status": status,
        "route": route,
        "seq": seq,
        "headDim": head_dim,
        "latencyMsP50": p50_ms,
        "speedupVsSdpaP50": speedup_vs_sdpa,
    }
    row.update(metrics)
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="native_now")
    parser.add_argument("--backend", default="torch_sdpa,thunderkittens")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260422)
    parser.add_argument("--tk-dir", type=Path, default=DEFAULT_TK_DIR)
    parser.add_argument("--jsonl-out", type=Path)
    parser.add_argument("--jsonl-stdout", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")

    torch.manual_seed(args.seed)
    cases = parse_csv(args.case)
    backends = parse_csv(args.backend)
    tk = None
    if "thunderkittens" in backends:
        tk = load_tk_module(args.tk_dir)

    out_file = None
    if args.jsonl_out is not None:
        args.jsonl_out.parent.mkdir(parents=True, exist_ok=True)
        out_file = args.jsonl_out.open("w", encoding="utf-8")

    def emit(row: dict[str, object]) -> None:
        line = json.dumps(row, sort_keys=True)
        if args.jsonl_stdout:
            print(line, flush=True)
        if out_file is not None:
            out_file.write(line + "\n")
            out_file.flush()

    emit({
        "event": "meta",
        "tool": "benchmarks/bench_tk_mha_h100.py",
        "device": "cuda:0",
        "timer": "cuda_event",
        "tkDir": repo_relative(args.tk_dir),
        "torchVersion": torch.__version__,
        "torchCuda": torch.version.cuda,
    })

    all_ok = True
    for case_id in cases:
        if case_id not in CASES:
            raise ValueError(f"unknown case: {case_id}")
        batch, q_heads, kv_heads, seq, head_dim = CASES[case_id]
        shape = (batch, q_heads, seq, head_dim)
        kv_shape = (batch, kv_heads, seq, head_dim)
        q_base = torch.randn(shape, device="cuda", dtype=torch.bfloat16)
        k_base = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
        v_base = torch.randn(kv_shape, device="cuda", dtype=torch.bfloat16)
        dO_base = torch.randn(shape, device="cuda", dtype=torch.bfloat16)

        ref = sdpa_once(q_base, k_base, v_base, dO_base)
        sdpa_samples = [
            time_cuda_event(lambda: sdpa_once(q_base, k_base, v_base, dO_base), args.warmup, args.iters)
            for _ in range(args.repeats)
        ]
        sdpa_p50 = median(sdpa_samples)

        if "torch_sdpa" in backends:
            emit(build_summary(
                case_id,
                "torch_sdpa",
                "ok",
                seq,
                head_dim,
                sdpa_p50,
                1.0,
                {
                    "correctnessOk": True,
                    "outMae": 0.0,
                    "outMax": 0.0,
                    "dqMae": 0.0,
                    "dqMax": 0.0,
                    "dkMae": 0.0,
                    "dkMax": 0.0,
                    "dvMae": 0.0,
                    "dvMax": 0.0,
                },
                "torch_sdpa",
            ))

        if "thunderkittens" in backends:
            assert tk is not None
            if not tk_supported(seq, head_dim):
                emit(build_summary(
                    case_id,
                    "thunderkittens",
                    "unsupported",
                    seq,
                    head_dim,
                    0.0,
                    0.0,
                    {
                        "correctnessOk": False,
                        "outMae": 0.0,
                        "outMax": 0.0,
                        "dqMae": 0.0,
                        "dqMax": 0.0,
                        "dkMae": 0.0,
                        "dkMax": 0.0,
                        "dvMae": 0.0,
                        "dvMax": 0.0,
                    },
                    "unsupported_seq_lt_768_or_bad_dim",
                ))
                continue

            def tk_once():
                q = q_base.detach()
                k = k_base.detach()
                v = v_base.detach()
                out, l_vec = tk.mha_forward(q, k, v, False)
                dQ, dK, dV = tk.mha_backward(q, k, v, out, l_vec, dO_base, False)
                return out, dQ, dK, dV

            got = tk_once()
            torch.cuda.synchronize()
            metrics = {
                "outMae": tensor_mae(got[0], ref[0]),
                "outMax": tensor_max(got[0], ref[0]),
                "dqMae": tensor_mae(got[1], ref[1]),
                "dqMax": tensor_max(got[1], ref[1]),
                "dkMae": tensor_mae(got[2], ref[2]),
                "dkMax": tensor_max(got[2], ref[2]),
                "dvMae": tensor_mae(got[3], ref[3]),
                "dvMax": tensor_max(got[3], ref[3]),
            }
            correctness_ok = (
                metrics["outMae"] < 5e-3 and
                metrics["dqMae"] < 5e-3 and
                metrics["dkMae"] < 5e-3 and
                metrics["dvMae"] < 5e-3
            )
            all_ok = all_ok and correctness_ok
            samples = [
                time_cuda_event(lambda: tk_once(), args.warmup, args.iters)
                for _ in range(args.repeats)
            ]
            p50 = median(samples)
            emit(build_summary(
                case_id,
                "thunderkittens",
                "ok",
                seq,
                head_dim,
                p50,
                sdpa_p50 / p50 if p50 > 0.0 else 0.0,
                {"correctnessOk": correctness_ok, **metrics},
                "vendored_tk_mha_h100_fwd_bwd",
            ))

    if out_file is not None:
        out_file.close()
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
