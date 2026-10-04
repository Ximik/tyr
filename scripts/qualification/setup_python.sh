#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
qualification_root=${TYR_QUALIFICATION_ROOT:-"$HOME/tyr-qualification"}
# The torch wheel in external/wheels is built for CPython 3.12 (WHEEL_PYTHON in
# deps/lock_wheels.py), so the venv must use the same version.
bootstrap_python=${TYR_QUALIFICATION_BOOTSTRAP_PYTHON:-python3.12}
venv_python="$qualification_root/venv/bin/python"
python_version() { "$1" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")'; }
if [[ "$(python_version "$bootstrap_python")" != 3.12 ]]; then
  echo "$bootstrap_python is not Python 3.12; set TYR_QUALIFICATION_BOOTSTRAP_PYTHON" >&2
  exit 1
fi
mkdir -p "$qualification_root"
# The venv holds only pinned pip packages, so one left over from another Python
# version is recreated rather than reused.
if [[ ! -x "$venv_python" || "$(python_version "$venv_python")" != 3.12 ]]; then
  "$bootstrap_python" -m venv --clear "$qualification_root/venv"
fi
if [[ ! -d "$repo_root/external/wheels/torch" ]]; then
  echo "external/wheels/torch is missing; run deps/fetch.sh" >&2
  exit 1
fi
# Import the exact torch the Lean build links against (deps/fetch.sh).
# Its *.dist-info lets pip treat torch and the nvidia-* wheels as installed, so
# the pinned requirements below only add the remaining reference packages.
purelib=$("$venv_python" -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')
rm -f "$purelib/spark-existing-runtime.pth"
printf '%s\n' "$repo_root/external/wheels" > "$purelib/tyr-external.pth"
"$venv_python" -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  -r "$repo_root/scripts/qualification/requirements.txt"
