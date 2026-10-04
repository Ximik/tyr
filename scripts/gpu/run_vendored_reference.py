#!/usr/bin/env python3
import math
import os
import sys
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TK_ROOT = REPO_ROOT / "external" / "git" / "ThunderKittens"
TORCH_EXTENSIONS_DIR = REPO_ROOT / ".cache" / "torch_extensions"


def fail(message: str) -> "NoReturn":
    raise SystemExit(message)


try:
    import torch
    from torch.utils.cpp_extension import load as load_extension
except Exception as exc:
    fail(f"vendored_ref import failed: {exc}")


def load_fixture(path: Path) -> torch.Tensor:
    obj = torch.load(path, map_location="cuda", weights_only=False)
    if isinstance(obj, (list, tuple)):
        if len(obj) != 1:
            fail(f"expected single-tensor fixture in {path}, got {type(obj).__name__} len={len(obj)}")
        obj = obj[0]
    elif hasattr(obj, "named_parameters") and hasattr(obj, "named_buffers"):
        params = [tensor for _, tensor in obj.named_parameters()]
        buffers = [tensor for _, tensor in obj.named_buffers()]
        tensors = params + buffers
        if len(tensors) == 1:
            obj = tensors[0]
    if not isinstance(obj, torch.Tensor):
        fail(f"fixture {path} did not deserialize to a torch.Tensor")
    return obj.cuda()


def compare(label: str, expected: torch.Tensor, actual: torch.Tensor, rtol: float, atol: float) -> bool:
    expected_f = expected.float()
    actual_f = actual.float()
    diff = (actual_f - expected_f).abs()
    ok = torch.allclose(actual_f, expected_f, rtol=rtol, atol=atol)
    mae = diff.mean().item()
    max_err = diff.max().item()
    print(f"{label} ok={str(ok).lower()} mae={mae} max={max_err} rtol={rtol} atol={atol}")
    return ok


def detect_cuda_code() -> tuple[str, str]:
    major, minor = torch.cuda.get_device_capability()
    return f"compute_{major}{minor}", f"sm_{major}{minor}"


def detect_default_family() -> str:
    gpu_name = torch.cuda.get_device_name(0)
    if "GB10" in gpu_name:
        return "hopper"
    if "B200" in gpu_name or "B300" in gpu_name:
        return "blackwell"
    if "A100" in gpu_name:
        return "ampere"
    return "hopper"


def detect_default_target() -> str:
    gpu_name = torch.cuda.get_device_name(0)
    if "GB10" in gpu_name:
        return "GB10"
    if "B300" in gpu_name:
        return "B300"
    if "B200" in gpu_name:
        return "B200"
    if "A100" in gpu_name:
        return "A100"
    return "H100"


def gpu_target() -> str:
    return os.environ.get("TYR_GPU_TARGET", detect_default_target()).strip().upper()


def kernel_family() -> str:
    return os.environ.get("TYR_GPU_FAMILY", detect_default_family()).strip().lower()


def macro_for_family(family: str) -> str:
    if family == "ampere":
        return "KITTENS_AMPERE"
    if family == "blackwell":
        return "KITTENS_BLACKWELL"
    if family == "hopper":
        return "KITTENS_HOPPER"
    fail(f"unsupported TYR_GPU_FAMILY={family!r}; expected ampere, hopper, or blackwell")


def vendor_suite_support_reason(suite: str) -> str | None:
    target = gpu_target()
    family = kernel_family()
    if suite in {"rotary", "flashattn", "mha_h100"} and (target in {"GB10", "B200", "B300"} or family == "blackwell"):
        return (
            f"suite {suite!r} uses vendored ThunderKittens code that is Hopper-only "
            f"and incompatible with target={target} family={family}"
        )
    return None


@lru_cache(maxsize=None)
def load_tk_extension(name: str, rel_source: str):
    compute, code = detect_cuda_code()
    family = kernel_family()
    source = TK_ROOT / rel_source
    build_dir = TORCH_EXTENSIONS_DIR / f"{name}_{family}_{code}"
    build_dir.mkdir(parents=True, exist_ok=True)
    include_dirs = [
        str(TK_ROOT / "include"),
        str(TK_ROOT / "prototype"),
    ]
    extra_cflags = ["-O3", "-std=c++20"]
    extra_cuda_cflags = [
        "-O3",
        "-std=c++20",
        "--use_fast_math",
        "--expt-extended-lambda",
        "--expt-relaxed-constexpr",
        "-lineinfo",
        "-DTORCH_COMPILE",
        f"-D{macro_for_family(family)}",
        f"-gencode=arch={compute},code={code}",
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    extra_ldflags = ["-lcuda"]
    return load_extension(
        name=f"tyr_{name}_{family}_{code}",
        sources=[str(source)],
        build_directory=str(build_dir),
        extra_include_paths=include_dirs,
        extra_cflags=extra_cflags,
        extra_cuda_cflags=extra_cuda_cflags,
        extra_ldflags=extra_ldflags,
        verbose=os.environ.get("TYR_GPU_VERBOSE_EXT_BUILD", "0") == "1",
    )


def run_rotary(fixtures: Path) -> bool:
    ext = load_tk_extension("tk_rotary", "kernels/rotary/rotary.cu")
    x = load_fixture(fixtures / "x.pt")
    sin = load_fixture(fixtures / "sin.pt")
    cos = load_fixture(fixtures / "cos.pt")
    expected = load_fixture(fixtures / "expected.pt")
    output = ext.fused_rotary(
        x.reshape(1, 1, x.shape[0], x.shape[1]).to(torch.bfloat16),
        cos.to(torch.bfloat16),
        sin.to(torch.bfloat16),
    ).reshape_as(expected)
    return compare("rotary.vendored", expected, output, 1e-4, 1e-4)


def run_layernorm(fixtures: Path) -> bool:
    ext = load_tk_extension("tk_layernorm", "kernels/layernorm/layernorm.cu")
    x = load_fixture(fixtures / "x.pt").to(torch.bfloat16)
    residual = load_fixture(fixtures / "residual.pt").to(torch.bfloat16)
    weight = load_fixture(fixtures / "weight.pt").to(torch.bfloat16)
    bias = load_fixture(fixtures / "bias.pt").to(torch.bfloat16)
    expected_out = load_fixture(fixtures / "expected_out.pt")
    expected_resid = load_fixture(fixtures / "expected_resid.pt")
    output, output_resid = ext.fused_layernorm(x, residual, weight, bias, 0.0)
    ok_out = compare("layernorm.vendored_output", expected_out, output, 5e-2, 5e-2)
    ok_resid = compare("layernorm.vendored_residual", expected_resid, output_resid, 1e-5, 1e-5)
    return ok_out and ok_resid


def run_flashattn(fixtures: Path) -> bool:
    ext = load_tk_extension("tk_mha_h100", "kernels/attention/mha_h100/mha_h100.cu")
    q = load_fixture(fixtures / "q.pt").to(torch.bfloat16)
    k = load_fixture(fixtures / "k.pt").to(torch.bfloat16)
    v = load_fixture(fixtures / "v.pt").to(torch.bfloat16)
    expected_o = load_fixture(fixtures / "expected_o.pt")
    expected_lse = load_fixture(fixtures / "expected_lse.pt")
    output, l_vec = ext.mha_forward(q, k, v, False)
    scale = math.sqrt(q.shape[-1])
    lse = (-l_vec.squeeze(-1) / scale).reshape_as(expected_lse)
    ok_out = compare("flashattn.vendored_output", expected_o, output, 3e-2, 3e-2)
    ok_lse = compare("flashattn.vendored_lse", expected_lse, lse, 3e-2, 3e-2)
    return ok_out and ok_lse


def run_mha_h100(fixtures: Path) -> bool:
    ext = load_tk_extension("tk_mha_h100", "kernels/attention/mha_h100/mha_h100.cu")
    q = load_fixture(fixtures / "q.pt").to(torch.bfloat16)
    k = load_fixture(fixtures / "k.pt").to(torch.bfloat16)
    v = load_fixture(fixtures / "v.pt").to(torch.bfloat16)
    dO = load_fixture(fixtures / "dO.pt").to(torch.bfloat16)
    expected_o = load_fixture(fixtures / "expected_o.pt")
    expected_l = load_fixture(fixtures / "expected_l.pt")
    expected_dq = load_fixture(fixtures / "expected_dq.pt")
    expected_dk = load_fixture(fixtures / "expected_dk.pt")
    expected_dv = load_fixture(fixtures / "expected_dv.pt")
    output, l_vec = ext.mha_forward(q, k, v, False)
    dQ, dK, dV = ext.mha_backward(q, k, v, output, l_vec, dO, False)
    l_out = l_vec.squeeze(-1).reshape_as(expected_l)
    checks = [
        compare("mha_h100.vendored_output", expected_o, output, 3e-2, 3e-2),
        compare("mha_h100.vendored_l", expected_l, l_out, 3e-2, 3e-2),
        compare("mha_h100.vendored_dq", expected_dq, dQ, 3e-2, 3e-2),
        compare("mha_h100.vendored_dk", expected_dk, dK, 3e-2, 3e-2),
        compare("mha_h100.vendored_dv", expected_dv, dV, 3e-2, 3e-2),
    ]
    return all(checks)


def main() -> int:
    if len(sys.argv) != 3:
        fail("usage: run_vendored_reference.py <suite-name> <fixture-dir>")
    if not torch.cuda.is_available():
        fail("vendored_ref requires CUDA-enabled torch")
    suite = sys.argv[1]
    fixtures = Path(sys.argv[2]).resolve()
    if not fixtures.exists():
        fail(f"fixture directory does not exist: {fixtures}")
    runners = {
        "rotary": run_rotary,
        "layernorm": run_layernorm,
        "flashattn": run_flashattn,
        "mha_h100": run_mha_h100,
    }
    if suite not in runners:
        print(f"[{suite}] vendored_ref unsupported=true")
        return 0
    support_reason = vendor_suite_support_reason(suite)
    if support_reason is not None:
        print(f"[{suite}] vendored_ref unsupported=true reason={support_reason}")
        return 0
    ok = runners[suite](fixtures)
    torch.cuda.synchronize()
    print(
        f"[{suite}] vendored_ref_summary ok={str(ok).lower()} "
        f"family={kernel_family()} target={gpu_target()}"
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
