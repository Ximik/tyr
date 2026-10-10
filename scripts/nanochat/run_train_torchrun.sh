#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

source ./env.sh

export LD_LIBRARY_PATH="${REPO_ROOT}/external/wheels/torch/lib:${REPO_ROOT}/cc/build:${EBROOTGCCCORE:+${EBROOTGCCCORE}/lib64:}${LD_LIBRARY_PATH:-}"

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
if ! command -v "${TORCHRUN_BIN}" >/dev/null 2>&1; then
  echo "error: torchrun launcher not found at ${TORCHRUN_BIN}" >&2
  echo "set TORCHRUN_BIN to a working torchrun binary on this host" >&2
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "error: NPROC_PER_NODE must be a positive integer (got ${NPROC_PER_NODE})" >&2
  exit 2
fi

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  lake --quiet build TrainNanoChat
fi

if [[ $# -eq 0 ]]; then
  set -- --debug --iterations 2 --data data/nanochat --val data/nanochat
fi

echo "Running TrainNanoChat with torchrun (${NPROC_PER_NODE} processes)"
echo "Torchrun: ${TORCHRUN_BIN}"

exec env TYR_DEVICE="${TYR_DEVICE:-cuda}" \
  "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --no_python \
  ./.lake/build/bin/TrainNanoChat "$@"
