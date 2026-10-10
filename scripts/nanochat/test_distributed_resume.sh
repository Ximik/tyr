#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

export LD_LIBRARY_PATH="${REPO_ROOT}/external/wheels/torch/lib:${REPO_ROOT}/cc/build:${EBROOTGCCCORE:+${EBROOTGCCCORE}/lib64:}${LD_LIBRARY_PATH:-}"

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
if ! command -v "${TORCHRUN_BIN}" >/dev/null 2>&1; then
  echo "error: torchrun launcher not found at ${TORCHRUN_BIN}" >&2
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-2}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-/tmp/nanochat_dist_resume_smoke}"
DATA_PATH="${DATA_PATH:-data/nanochat}"
VAL_PATH="${VAL_PATH:-data/nanochat}"
FRESH_ITERS="${FRESH_ITERS:-2}"
RESUME_ITERS="${RESUME_ITERS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  lake --quiet build TrainNanoChat
fi

rm -rf "${CHECKPOINT_DIR}"
mkdir -p "${CHECKPOINT_DIR}"
FRESH_LOG="${CHECKPOINT_DIR}/fresh.log"
RESUME_LOG="${CHECKPOINT_DIR}/resume.log"

run() {
  local iters="$1"
  local log="$2"
  env CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
    TYR_DEVICE="${TYR_DEVICE:-cuda}" \
    "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --no_python \
    ./.lake/build/bin/TrainNanoChat \
    --debug \
    --iterations "${iters}" \
    --data "${DATA_PATH}" \
    --val "${VAL_PATH}" \
    --checkpoint-dir "${CHECKPOINT_DIR}" > "${log}" 2>&1
}

run "${FRESH_ITERS}" "${FRESH_LOG}"
run "${RESUME_ITERS}" "${RESUME_LOG}"

if rg -n "Duplicate GPU|Error: .*NCCL|Broadcast failed" "${FRESH_LOG}" "${RESUME_LOG}" >/dev/null; then
  echo "error: NCCL/device failure detected" >&2
  rg -n "Duplicate GPU|Error: .*NCCL|Broadcast failed" "${FRESH_LOG}" "${RESUME_LOG}" >&2 || true
  exit 1
fi

rg -n "Training complete!" "${FRESH_LOG}" >/dev/null
rg -n "Resuming from checkpoint:" "${RESUME_LOG}" >/dev/null
rg -n "Training complete!" "${RESUME_LOG}" >/dev/null
rg -n "Saving checkpoint to .*latest\\.ckpt at step 43" "${RESUME_LOG}" >/dev/null

echo "PASS: distributed fresh+resume completed"
echo "Fresh log:  ${FRESH_LOG}"
echo "Resume log: ${RESUME_LOG}"
