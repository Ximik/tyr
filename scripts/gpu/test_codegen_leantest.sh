#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

uv_bin="${UV_BIN:-$(command -v uv)}"
venv_python="${TYR_GPU_VENV_PYTHON:-.venv-gpu/bin/python}"
lake_bin="${LAKE_BIN:-${HOME}/.elan/bin/lake}"
lean_bin="${LEAN_BIN:-${HOME}/.elan/bin/lean}"

if [[ ! -x "${uv_bin}" ]]; then
  echo "uv not found; install uv first." >&2
  exit 1
fi

if [[ ! -x "${venv_python}" ]]; then
  echo "missing ${venv_python}; run ./scripts/gpu/setup_python_venv.sh first." >&2
  exit 1
fi

if [[ ! -x "${lake_bin}" || ! -x "${lean_bin}" ]]; then
  echo "missing Lean toolchain binaries; ensure elan is installed." >&2
  exit 1
fi

export LD_LIBRARY_PATH="${repo_root}/external/python/torch/lib:${repo_root}/cc/build:${EBROOTGCCCORE:+${EBROOTGCCCORE}/lib64:}${LD_LIBRARY_PATH:-}"

echo "[1/4] Build GPU kernel LeanTest module"
"${uv_bin}" run --python "${venv_python}" env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  "${lake_bin}" -R --quiet build Tests.TestGPUKernels

echo "[2/4] Build GPU DSL LeanTest module"
"${uv_bin}" run --python "${venv_python}" env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  "${lake_bin}" -R --quiet build Tests.TestGPUDSL

echo "[3/4] Run GPU kernel LeanTest suite"
"${uv_bin}" run --python "${venv_python}" env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  "${lake_bin}" -R env "${lean_bin}" --run Tests/RunTestGPUKernels.lean "$@"

echo "[4/4] Run GPU DSL LeanTest suite"
"${uv_bin}" run --python "${venv_python}" env LD_LIBRARY_PATH="${LD_LIBRARY_PATH}" \
  "${lake_bin}" -R env "${lean_bin}" --run Tests/RunTestGPUDSL.lean "$@"
