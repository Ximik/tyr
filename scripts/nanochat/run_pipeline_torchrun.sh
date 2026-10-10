#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"
source "${SCRIPT_DIR}/checkpoint_paths.sh"

have_glob() {
  local pattern="$1"
  compgen -G "${pattern}" > /dev/null
}

# Disable module paging for non-interactive and detached launches.
export LMOD_PAGER="${LMOD_PAGER:-none}"
export MODULES_PAGER="${MODULES_PAGER:-cat}"
export PAGER="${PAGER:-cat}"

source ./env.sh

export LD_LIBRARY_PATH="${REPO_ROOT}/external/wheels/torch/lib:${REPO_ROOT}/cc/build:${EBROOTGCCCORE:+${EBROOTGCCCORE}/lib64:}${LD_LIBRARY_PATH:-}"

TORCHRUN_BIN="${TORCHRUN_BIN:-torchrun}"
if ! command -v "${TORCHRUN_BIN}" >/dev/null 2>&1; then
  echo "error: torchrun launcher not found at ${TORCHRUN_BIN}" >&2
  exit 2
fi

NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
if ! [[ "${NPROC_PER_NODE}" =~ ^[0-9]+$ ]] || [[ "${NPROC_PER_NODE}" -lt 1 ]]; then
  echo "error: NPROC_PER_NODE must be a positive integer (got '${NPROC_PER_NODE}')" >&2
  exit 2
fi

PIPELINE_EXE="${PIPELINE_EXE:-NanoChatPipeline}"
PIPELINE_BIN="${REPO_ROOT}/.lake/build/bin/${PIPELINE_EXE}"
if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  lake --quiet build "${PIPELINE_EXE}"
elif [[ ! -x "${PIPELINE_BIN}" ]]; then
  echo "error: ${PIPELINE_BIN} is missing and SKIP_BUILD=1. Run without SKIP_BUILD or build it first." >&2
  exit 2
fi

if [[ "${QUICK_MODE_FLAG:-0}" == "1" ]]; then
  export QUICK_MODE=1
else
  unset QUICK_MODE || true
fi

if [[ "${ENABLE_RL_FLAG:-0}" == "1" ]]; then
  export ENABLE_RL=1
else
  unset ENABLE_RL || true
fi

if [[ "${NANOCHAT_USE_RUN_ID_DIR:-0}" == "1" ]]; then
  export NANOCHAT_RUN_ID="${NANOCHAT_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
  export NANOCHAT_DIR="${NANOCHAT_DIR:-${HOME}/.cache/nanochat_${NANOCHAT_RUN_ID}}"
else
  export NANOCHAT_DIR="${NANOCHAT_DIR:-${HOME}/.cache/nanochat}"
fi
mkdir -p "${NANOCHAT_DIR}"

if [[ "${TYR_DEVICE:-cuda}" == "cuda" ]] && command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT="$(nvidia-smi --query-gpu=count --format=csv,noheader 2>/dev/null | head -n 1 || echo 0)"
  if [[ "${GPU_COUNT}" =~ ^[0-9]+$ ]] && [[ "${GPU_COUNT}" -gt 0 ]] && [[ "${NPROC_PER_NODE}" -gt "${GPU_COUNT}" ]]; then
    echo "error: requested NPROC_PER_NODE=${NPROC_PER_NODE}, but only ${GPU_COUNT} CUDA devices are visible" >&2
    exit 2
  fi
fi

export DATA_PATH="${DATA_PATH:-base_data}"
export MODEL_DEPTH="${MODEL_DEPTH:-20}"
export PARAM_DATA_RATIO="${PARAM_DATA_RATIO:-20}"
export VOCAB_SIZE="${VOCAB_SIZE:-65536}"
export TOKENIZER_MAX_CHARS="${TOKENIZER_MAX_CHARS:-2000000000}"
export TOKENIZER_DOC_CAP="${TOKENIZER_DOC_CAP:-10000}"
export INITIAL_DATA_SHARDS="${INITIAL_DATA_SHARDS:-8}"
export NUM_SHARDS="${NUM_SHARDS:-240}"
export WANDB_RUN="${WANDB_RUN:-dummy}"
if [[ -z "${WANDB_ENABLED:-}" ]]; then
  if [[ "${WANDB_RUN}" == "dummy" ]]; then
    export WANDB_ENABLED=0
  else
    export WANDB_ENABLED=1
  fi
fi

# Pretrain defaults.
# Note: Lean path currently has higher activation memory than upstream nanochat,
# so use a conservative per-GPU microbatch and preserve global token batch via
# accumulation (PRETRAIN_TOTAL_BATCH_SIZE).
export PRETRAIN_ITERS="${PRETRAIN_ITERS:-21400}"
export PRETRAIN_EXTENSION_ITERS="${PRETRAIN_EXTENSION_ITERS:-0}"
export PRETRAIN_DEVICE_BATCH_SIZE="${PRETRAIN_DEVICE_BATCH_SIZE:-8}"
export PRETRAIN_TOTAL_BATCH_SIZE="${PRETRAIN_TOTAL_BATCH_SIZE:-524288}"
export PRETRAIN_VAL_INTERVAL="${PRETRAIN_VAL_INTERVAL:-250}"
export PRETRAIN_LOG_INTERVAL="${PRETRAIN_LOG_INTERVAL:-10}"
export PRETRAIN_CHECKPOINT_INTERVAL="${PRETRAIN_CHECKPOINT_INTERVAL:-21400}"
export PRETRAIN_TEXT_COLUMN="${PRETRAIN_TEXT_COLUMN:-text}"
export PRETRAIN_TOKENIZER_BATCH_SIZE="${PRETRAIN_TOKENIZER_BATCH_SIZE:-128}"

# Midtrain defaults aligned to current Lean pipeline implementation.
export MIDTRAIN_ITERS="${MIDTRAIN_ITERS:-811}"
export MIDTRAIN_EXTENSION_ITERS="${MIDTRAIN_EXTENSION_ITERS:-0}"
export MIDTRAIN_DEVICE_BATCH_SIZE="${MIDTRAIN_DEVICE_BATCH_SIZE:-8}"
export MIDTRAIN_TOTAL_BATCH_SIZE="${MIDTRAIN_TOTAL_BATCH_SIZE:-524288}"
export MIDTRAIN_VAL_INTERVAL="${MIDTRAIN_VAL_INTERVAL:-150}"
export MIDTRAIN_LOG_INTERVAL="${MIDTRAIN_LOG_INTERVAL:-10}"
export MIDTRAIN_CHECKPOINT_INTERVAL="${MIDTRAIN_CHECKPOINT_INTERVAL:-811}"

# SFT defaults.
export SFT_EPOCHS="${SFT_EPOCHS:-1}"
export SFT_DEVICE_BATCH_SIZE="${SFT_DEVICE_BATCH_SIZE:-4}"
export SFT_TARGET_EXAMPLES_PER_STEP="${SFT_TARGET_EXAMPLES_PER_STEP:-32}"

# GRPO defaults (used only if ENABLE_RL=1).
export GRPO_NUM_SAMPLES="${GRPO_NUM_SAMPLES:-16}"
export GRPO_MAX_NEW_TOKENS="${GRPO_MAX_NEW_TOKENS:-256}"
export GRPO_EXAMPLES_PER_STEP="${GRPO_EXAMPLES_PER_STEP:-16}"
export REQUIRE_BIN_DATA_PATHS="${REQUIRE_BIN_DATA_PATHS:-0}"

if [[ "${AUTO_REUSE_CACHE:-1}" == "1" ]]; then
  if [[ ! -e "${NANOCHAT_DIR}/base_data" && -d "${HOME}/.cache/nanochat/base_data" ]]; then
    ln -s "${HOME}/.cache/nanochat/base_data" "${NANOCHAT_DIR}/base_data"
  fi
  if [[ ! -e "${NANOCHAT_DIR}/tokenizer" && -d "${HOME}/.cache/nanochat/tokenizer" ]]; then
    ln -s "${HOME}/.cache/nanochat/tokenizer" "${NANOCHAT_DIR}/tokenizer"
  fi
fi

if [[ -z "${PRETRAIN_DATA_PATH:-}" ]]; then
  if [[ -d "${NANOCHAT_DIR}/base_data" ]] && have_glob "${NANOCHAT_DIR}/base_data/*.parquet"; then
    export PRETRAIN_DATA_PATH="${NANOCHAT_DIR}/base_data"
  elif [[ -d "${REPO_ROOT}/data/nanochat" ]] && have_glob "${REPO_ROOT}/data/nanochat/*.parquet"; then
    export PRETRAIN_DATA_PATH="${REPO_ROOT}/data/nanochat"
  elif [[ "${REQUIRE_BIN_DATA_PATHS}" == "1" ]] && [[ -d "${NANOCHAT_DIR}/base_data_bin" ]] && have_glob "${NANOCHAT_DIR}/base_data_bin/*.bin"; then
    export PRETRAIN_DATA_PATH="${NANOCHAT_DIR}/base_data_bin"
  elif [[ "${REQUIRE_BIN_DATA_PATHS}" == "1" ]] && [[ -d "${NANOCHAT_DIR}/base_data" ]] && have_glob "${NANOCHAT_DIR}/base_data/*.bin"; then
    export PRETRAIN_DATA_PATH="${NANOCHAT_DIR}/base_data"
  elif [[ "${REQUIRE_BIN_DATA_PATHS}" == "1" ]] && [[ -d "${REPO_ROOT}/data/nanochat" ]] && have_glob "${REPO_ROOT}/data/nanochat/*.bin"; then
    export PRETRAIN_DATA_PATH="${REPO_ROOT}/data/nanochat"
  else
    export PRETRAIN_DATA_PATH="${NANOCHAT_DIR}/base_data"
    mkdir -p "${PRETRAIN_DATA_PATH}"
  fi
fi
if [[ -z "${MIDTRAIN_DATA_PATH:-}" && -n "${PRETRAIN_DATA_PATH:-}" ]]; then
  export MIDTRAIN_DATA_PATH="${PRETRAIN_DATA_PATH}"
fi

if [[ "${REQUIRE_BIN_DATA_PATHS}" == "1" ]]; then
  if [[ -z "${PRETRAIN_DATA_PATH:-}" ]]; then
    echo "error: PRETRAIN_DATA_PATH is unset. Set PRETRAIN_DATA_PATH to a directory containing .bin token shards." >&2
    exit 2
  fi
  if [[ -z "${MIDTRAIN_DATA_PATH:-}" ]]; then
    echo "error: MIDTRAIN_DATA_PATH is unset. Set MIDTRAIN_DATA_PATH to a directory containing .bin token shards." >&2
    exit 2
  fi
  if ! have_glob "${PRETRAIN_DATA_PATH}/*.bin"; then
    echo "error: PRETRAIN_DATA_PATH has no .bin files: ${PRETRAIN_DATA_PATH}" >&2
    exit 2
  fi
  if ! have_glob "${MIDTRAIN_DATA_PATH}/*.bin"; then
    echo "error: MIDTRAIN_DATA_PATH has no .bin files: ${MIDTRAIN_DATA_PATH}" >&2
    exit 2
  fi
else
  if [[ -z "${PRETRAIN_DATA_PATH:-}" ]]; then
    echo "error: PRETRAIN_DATA_PATH is unset. Set PRETRAIN_DATA_PATH to a directory containing parquet shards." >&2
    exit 2
  fi
  if ! have_glob "${PRETRAIN_DATA_PATH}/*.parquet"; then
    echo "warning: PRETRAIN_DATA_PATH has no local .parquet files yet: ${PRETRAIN_DATA_PATH}" >&2
    echo "         proceeding; pipeline download stages are expected to populate this directory." >&2
  fi
fi

if [[ "${CLEAR_PIPELINE_CHECKPOINT:-1}" == "1" ]]; then
  rm -f "${REPO_ROOT}/.pipeline_checkpoint.json" "${NANOCHAT_DIR}/.pipeline_checkpoint.json"
fi

if [[ "${PRETRAIN_ITERS}" =~ ^[0-9]+$ ]] && [[ "${PRETRAIN_ITERS}" -eq 0 ]]; then
  if ! nanochat_checkpoint_dir "${NANOCHAT_DIR}/checkpoints/base/latest.ckpt" >/dev/null; then
    echo "error: PRETRAIN_ITERS=0 but no base checkpoint is present at ${NANOCHAT_DIR}/checkpoints/base/latest.ckpt" >&2
    echo "       set PRETRAIN_ITERS>=1 for a fresh run, or place/resume an existing base checkpoint first." >&2
    exit 2
  fi
fi

echo "Running ${PIPELINE_EXE} with torchrun (${NPROC_PER_NODE} processes)"
echo "Torchrun: ${TORCHRUN_BIN}"
echo "Executable: ${PIPELINE_EXE}"
echo "QUICK_MODE=${QUICK_MODE_FLAG:-0} ENABLE_RL=${ENABLE_RL_FLAG:-0}"
echo "NANOCHAT_DIR=${NANOCHAT_DIR}"
echo "NANOCHAT_USE_RUN_ID_DIR=${NANOCHAT_USE_RUN_ID_DIR:-0}"
echo "WANDB_RUN=${WANDB_RUN} WANDB_ENABLED=${WANDB_ENABLED}"
echo "MODEL_DEPTH=${MODEL_DEPTH} PARAM_DATA_RATIO=${PARAM_DATA_RATIO} VOCAB_SIZE=${VOCAB_SIZE}"
echo "TOKENIZER_MAX_CHARS=${TOKENIZER_MAX_CHARS} TOKENIZER_DOC_CAP=${TOKENIZER_DOC_CAP}"
echo "INITIAL_DATA_SHARDS=${INITIAL_DATA_SHARDS} NUM_SHARDS=${NUM_SHARDS}"
echo "PRETRAIN_ITERS=${PRETRAIN_ITERS} PRETRAIN_DEVICE_BATCH_SIZE=${PRETRAIN_DEVICE_BATCH_SIZE} PRETRAIN_TOTAL_BATCH_SIZE=${PRETRAIN_TOTAL_BATCH_SIZE}"
echo "MIDTRAIN_ITERS=${MIDTRAIN_ITERS} MIDTRAIN_DEVICE_BATCH_SIZE=${MIDTRAIN_DEVICE_BATCH_SIZE} MIDTRAIN_TOTAL_BATCH_SIZE=${MIDTRAIN_TOTAL_BATCH_SIZE}"
echo "SFT_EPOCHS=${SFT_EPOCHS} SFT_DEVICE_BATCH_SIZE=${SFT_DEVICE_BATCH_SIZE} SFT_TARGET_EXAMPLES_PER_STEP=${SFT_TARGET_EXAMPLES_PER_STEP}"
echo "DATA_PATH=${DATA_PATH}"
echo "PRETRAIN_DATA_PATH=${PRETRAIN_DATA_PATH:-<unset>}"
echo "MIDTRAIN_DATA_PATH=${MIDTRAIN_DATA_PATH:-<unset>}"
echo "PRETRAIN_TEXT_COLUMN=${PRETRAIN_TEXT_COLUMN} PRETRAIN_TOKENIZER_BATCH_SIZE=${PRETRAIN_TOKENIZER_BATCH_SIZE}"
echo "AUTO_REUSE_CACHE=${AUTO_REUSE_CACHE:-1}"
echo "REQUIRE_BIN_DATA_PATHS=${REQUIRE_BIN_DATA_PATHS}"

exec env TYR_DEVICE="${TYR_DEVICE:-cuda}" \
  "${TORCHRUN_BIN}" --standalone --nnodes=1 --nproc_per_node="${NPROC_PER_NODE}" --no_python \
  "${PIPELINE_BIN}"
