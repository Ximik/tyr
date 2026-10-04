#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
qualification_root=${TYR_QUALIFICATION_ROOT:-"$HOME/tyr-qualification"}
# The torch wheel in external/wheels is built for CPython 3.12 (WHEEL_PYTHON in
# deps/lock_wheels.py), so the venv must use the same version.
bootstrap_python=${TYR_QUALIFICATION_BOOTSTRAP_PYTHON:-python3.12}
venv_python="$qualification_root/venv/bin/python"
mkdir -p "$qualification_root"
if [[ ! -x "$venv_python" ]]; then
  "$bootstrap_python" -m venv "$qualification_root/venv"
fi
if [[ "$("$venv_python" -c 'import sys; print(f"{sys.version_info[0]}.{sys.version_info[1]}")')" != 3.12 ]]; then
  echo "$qualification_root/venv must use Python 3.12 to load external/wheels; delete it and rerun" >&2
  exit 1
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
