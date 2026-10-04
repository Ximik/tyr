#!/usr/bin/env bash
# Create the .venv-gpu Python environment used by the Python reference and
# benchmark tools. It imports the torch that ./fetch_dependencies.sh put in
# external/python (the same files the Lean build links against) through a .pth
# file, and adds only numpy, ninja and torch's pure-Python dependencies.
#
# The fetched torch wheel is built for CPython 3.12 (WHEEL_PYTHON in
# scripts/lock_dependencies.py). Set TYR_GPU_PYTHON to a 3.12 interpreter; if
# none is found and uv is installed, uv provides one.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

venv_dir="${TYR_GPU_VENV:-.venv-gpu}"
python_bin="${TYR_GPU_PYTHON:-python3.12}"

if [[ ! -d external/python/torch ]]; then
  echo "external/python/torch is missing; run ./fetch_dependencies.sh" >&2
  exit 1
fi
if ! command -v "${python_bin}" >/dev/null 2>&1; then
  if command -v uv >/dev/null 2>&1; then
    uv python install 3.12
    python_bin="$(uv python find 3.12)"
  else
    echo "no Python 3.12 found; set TYR_GPU_PYTHON or install uv" >&2
    exit 1
  fi
fi

"${python_bin}" -m venv --clear "${venv_dir}"
venv_python="${venv_dir}/bin/python"
if [[ "$("${venv_python}" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" != 3.12 ]]; then
  echo "${python_bin} is not Python 3.12" >&2
  exit 1
fi

purelib="$("${venv_python}" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
printf '%s\n' "${repo_root}/external/python" > "${purelib}/tyr-external.pth"

# torch is already satisfied through the .pth; pip installs only what it lacks.
torch_version="$("${venv_python}" -c 'import importlib.metadata as m; print(m.version("torch"))')"
"${venv_python}" -m pip install "torch==${torch_version}" numpy ninja

# Building the vendored ThunderKittens reference as a torch extension needs Python.h.
python_include="$("${venv_python}" -c 'import sysconfig; print(sysconfig.get_config_var("INCLUDEPY") or sysconfig.get_path("include") or "")')"
if [[ -z "${python_include}" || ! -f "${python_include}/Python.h" ]]; then
  echo "missing Python headers: ${python_include}/Python.h" >&2
  exit 1
fi

"${venv_python}" -c 'import torch; print(torch.__version__, torch.__file__); print(torch.version.cuda); print(torch.cuda.is_available())'
