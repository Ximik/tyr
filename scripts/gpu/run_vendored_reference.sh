#!/usr/bin/env bash
set -euo pipefail

venv_python="${TYR_GPU_PYTHON:-$PWD/.venv-gpu/bin/python}"
if [[ ! -x "${venv_python}" ]]; then
  echo "vendored_ref missing Python env: expected ${venv_python}" >&2
  echo "hint: run ./scripts/gpu/setup_python_venv.sh" >&2
  exit 1
fi

export PATH="$(dirname "${venv_python}"):${PATH}"

if command -v uv >/dev/null 2>&1; then
  exec uv run --python "${venv_python}" "$PWD/scripts/gpu/run_vendored_reference.py" "$@"
fi

exec "${venv_python}" "$PWD/scripts/gpu/run_vendored_reference.py" "$@"
