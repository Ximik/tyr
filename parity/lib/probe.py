"""Running a tyr parity probe (`.lake/build/bin/Parity<Name>`) and reading its output."""

import json
import subprocess
from pathlib import Path

import numpy as np
from safetensors.numpy import load_file

REPO = Path(__file__).resolve().parents[2]


def run(name: str, request: dict, weights: Path, device: str, out: Path) -> None:
    """Write `request` to `out/request.json` and run probe `Parity<name>` on it;
    the probe writes `out/result.json` and `out/tensors.safetensors`."""
    out.mkdir(parents=True, exist_ok=True)
    request_path = out / "request.json"
    request_path.write_text(json.dumps(request))
    binary = REPO / ".lake" / "build" / "bin" / f"Parity{name}"
    subprocess.run(
        [str(binary), "--request", str(request_path), "--weights", str(weights),
         "--device", device, "--out", str(out)],
        check=True,
    )


def read(out: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """`result.json` and the tensors of `tensors.safetensors` (empty if absent)
    in a probe's or a reference's output directory."""
    result = json.loads((out / "result.json").read_text())
    tensors_path = out / "tensors.safetensors"
    tensors = load_file(str(tensors_path)) if tensors_path.exists() else {}
    return result, tensors
