#!/usr/bin/env bash
# Create the .venv-gpu Python environment (torch, numpy, ninja) used by the
# Python reference and benchmark tools. The Lean build does not use it: its
# libtorch comes from ./fetch_dependencies.sh.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

uv_bin="${UV_BIN:-$(command -v uv)}"
python_bin="${TYR_GPU_BOOTSTRAP_PYTHON:-python3}"
managed_python="${TYR_GPU_MANAGED_PYTHON:-3.12.13}"
venv_dir="${TYR_GPU_VENV:-.venv-gpu}"
torch_channel="${TYR_GPU_TORCH_CHANNEL:-}"
torch_version="${TYR_GPU_TORCH_VERSION:-}"

ensure_python_headers() {
  local candidate="$1"
  local has_headers
  if ! command -v "${candidate}" >/dev/null 2>&1; then
    echo "${candidate}"
    return
  fi
  has_headers="$("${candidate}" -c 'import sysconfig; from pathlib import Path; include = sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_path("include") or ""; print("yes" if include and Path(include, "Python.h").exists() else "no")' 2>/dev/null || echo no)"
  if [[ "${has_headers}" == "yes" ]]; then
    echo "${candidate}"
    return
  fi
  echo "python headers missing for ${candidate}; installing uv-managed CPython ${managed_python}" >&2
  "${uv_bin}" python install "${managed_python}"
  echo "${managed_python}"
}

detect_torch_channel() {
  if [[ -n "${torch_channel}" ]]; then
    echo "${torch_channel}"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "cpu"
    return
  fi
  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\r')"
  case "${gpu_name}" in
    *GB10*|*B200*|*B300*|*H100*|*A100*) echo "cu130" ;;
    *) echo "cpu" ;;
  esac
}

detect_torch_version() {
  if [[ -n "${torch_version}" ]]; then
    echo "${torch_version}"
    return
  fi
  local channel="$1"
  if [[ "${channel}" == "cpu" ]]; then
    echo "2.9.1"
    return
  fi
  case "${python_mm}" in
    3.12)
      # As of April 5, 2026, this is the latest official cu130 aarch64 wheel
      # available for CPython 3.12 in the PyTorch index.
      echo "2.9.0+cu130"
      ;;
    *)
      echo "2.9.0+cu130"
      ;;
  esac
}

channel="$(detect_torch_channel)"
python_spec="$(ensure_python_headers "${python_bin}")"

"${uv_bin}" venv "${venv_dir}" --python "${python_spec}" --clear

python_mm="$("${venv_dir}/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
version="$(detect_torch_version "${channel}")"

echo "uv=${uv_bin}"
echo "python=${python_spec} (${python_mm})"
echo "torch_channel=${channel}"
echo "torch_version=${version}"

if [[ "${channel}" == "cpu" ]]; then
  "${uv_bin}" pip install \
    --python "${venv_dir}/bin/python" \
    torch=="${version}" \
    numpy \
    ninja
else
  "${uv_bin}" pip install \
    --python "${venv_dir}/bin/python" \
    --index-strategy unsafe-best-match \
    --index-url "https://download.pytorch.org/whl/${channel}" \
    --extra-index-url https://pypi.org/simple \
    "torch==${version}" \
    numpy \
    ninja
fi

python_include="$("${venv_dir}/bin/python" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_path("include") or "")')"
if [[ -z "${python_include}" || ! -f "${python_include}/Python.h" ]]; then
  echo "missing Python headers after venv creation: ${python_include}/Python.h" >&2
  exit 1
fi

"${venv_dir}/bin/python" -c 'import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())'
