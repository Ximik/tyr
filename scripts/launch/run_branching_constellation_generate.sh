#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

data="${TYR_CONSTELLATION_DATA:-launch/generated/branching-constellations/dataset.jsonl}"
checkpoint="${TYR_CONSTELLATION_CHECKPOINT:-launch/generated/branching-constellations/run-3000-cap2-vocab2/checkpoint}"
split_cap="${TYR_CONSTELLATION_SPLIT_CAP:--1.5}"
if [[ "$split_cap" == -* ]]; then
  split_slug="neg${split_cap#-}"
else
  split_slug="pos${split_cap}"
fi
split_slug="${split_slug//./p}"
output_dir="${TYR_CONSTELLATION_OUTPUT:-launch/generated/branching-constellations/cohort-split${split_slug}}"
device="${TYR_CONSTELLATION_DEVICE:-cuda}"
samples="${TYR_CONSTELLATION_SAMPLES:-64}"
sample_steps="${TYR_CONSTELLATION_SAMPLE_STEPS:-96}"
max_len="${TYR_CONSTELLATION_MAX_LEN:-12}"
batch="${TYR_CONSTELLATION_BATCH:-32}"
seed="${TYR_CONSTELLATION_SEED:-20260710}"
train_seed="${TYR_CONSTELLATION_TRAIN_SEED:-20260709}"
vocab_size="${TYR_CONSTELLATION_VOCAB_SIZE:-2}"
coord_cap="${TYR_CONSTELLATION_COORD_CAP:-2.0}"
splits_weight="${TYR_CONSTELLATION_SPLITS_WEIGHT:-5.0}"
del_weight="${TYR_CONSTELLATION_DEL_WEIGHT:-1.0}"
deletion_pad="${TYR_CONSTELLATION_DELETION_PAD:-0.0}"
fixed_labels="${TYR_CONSTELLATION_FIXED_LABELS:-0}"

if [[ ! "$fixed_labels" =~ ^(0|1)$ ]]; then
  echo "TYR_CONSTELLATION_FIXED_LABELS must be 0 or 1, got: $fixed_labels" >&2
  exit 1
fi

manifest_value() {
  local manifest="$1"
  local key="$2"
  awk -v key="$key" '
    index($0, key "=") == 1 {
      count += 1
      value = substr($0, length(key) + 2)
    }
    END {
      if (count == 1) {
        print value
      } else {
        exit 1
      }
    }
  ' "$manifest"
}

binary=".lake/build/bin/BranchingFlowsMoleculeTrainGenerate"
if [[ ! -x "$binary" ]]; then
  echo "Missing $binary; run lake build BranchingFlowsMoleculeTrainGenerate first." >&2
  exit 1
fi
if [[ ! -f "$data" ]]; then
  echo "Missing constellation dataset: $data" >&2
  exit 1
fi
# Resolve CURRENT once. Hash and load that same immutable snapshot even if a
# concurrent training process publishes a newer one during generation.
checkpoint_snapshot="$checkpoint"
if [[ -e "$checkpoint/CURRENT" || -L "$checkpoint/CURRENT" ]]; then
  if [[ ! -f "$checkpoint/CURRENT" ]]; then
    echo "Invalid checkpoint snapshot pointer: $checkpoint/CURRENT" >&2
    exit 1
  fi
  snapshot_name="$(cat "$checkpoint/CURRENT")"
  if [[ ! "$snapshot_name" =~ ^snapshot-[A-Za-z0-9_-]+$ ]]; then
    echo "Invalid checkpoint snapshot pointer: $checkpoint/CURRENT" >&2
    exit 1
  fi
  checkpoint_snapshot="$checkpoint/.snapshots/$snapshot_name"
  if [[ ! -d "$checkpoint_snapshot" ]]; then
    echo "Missing checkpoint snapshot: $checkpoint_snapshot" >&2
    exit 1
  fi
fi
if [[ ! -f "$checkpoint_snapshot/meta.txt" ]]; then
  echo "Missing constellation checkpoint: $checkpoint" >&2
  exit 1
fi
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite an existing cohort path: $output_dir" >&2
  exit 1
fi

training_manifest="$(dirname "$checkpoint")/manifest.txt"
training_config_checked=false
training_fixed_labels="unknown"
training_vocab_size="unknown"
training_coord_cap="unknown"
parsed_fixed_labels=""
parsed_vocab_size=""
parsed_coord_cap=""
if [[ -f "$training_manifest" ]] &&
    parsed_fixed_labels="$(manifest_value "$training_manifest" fixed_labels 2>/dev/null)" &&
    parsed_vocab_size="$(manifest_value "$training_manifest" vocab_size 2>/dev/null)" &&
    parsed_coord_cap="$(manifest_value "$training_manifest" coord_target_cap 2>/dev/null)" &&
    [[ "$parsed_fixed_labels" =~ ^(0|1)$ ]] &&
    [[ "$parsed_vocab_size" =~ ^[0-9]+$ ]] &&
    [[ "$parsed_coord_cap" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  training_config_checked=true
  training_fixed_labels="$parsed_fixed_labels"
  training_vocab_size="$parsed_vocab_size"
  training_coord_cap="$parsed_coord_cap"
  config_mismatches=()
  if [[ "$fixed_labels" != "$training_fixed_labels" ]]; then
    config_mismatches+=("fixed_labels: checkpoint=$training_fixed_labels requested=$fixed_labels")
  fi
  if ! awk -v checkpoint="$training_vocab_size" -v requested="$vocab_size" \
      'BEGIN { exit !((checkpoint + 0) == (requested + 0)) }'; then
    config_mismatches+=("vocab_size: checkpoint=$training_vocab_size requested=$vocab_size")
  fi
  if ! awk -v checkpoint="$training_coord_cap" -v requested="$coord_cap" \
      'BEGIN { exit !((checkpoint + 0.0) == (requested + 0.0)) }'; then
    config_mismatches+=("coord_target_cap: checkpoint=$training_coord_cap requested=$coord_cap")
  fi
  if [[ "${#config_mismatches[@]}" -gt 0 ]]; then
    printf 'Checkpoint/training configuration mismatch: %s\n' "${config_mismatches[@]}" >&2
    exit 1
  fi
fi

cmd=(
  "$binary"
  --require-data
  --data "$data"
  --out-prefix "$output_dir/constellation"
  --resume-checkpoint "$checkpoint_snapshot"
  --generate-only
  --no-checkpoint
  --no-resume-optimizer
  --steps 0
  --total-steps 3000
  --architecture full
  --hidden-dim 128
  --heads 4
  --head-dim 32
  --mlp 512
  --rff-dim 32
  --layers 4
  --coord-update-layers 2
  --max-len "$max_len"
  --vocab-size "$vocab_size"
  --coord-target-cap "$coord_cap"
  --splits-weight "$splits_weight"
  --del-weight "$del_weight"
  --deletion-pad "$deletion_pad"
  --batch-size "$batch"
  --sample-steps "$sample_steps"
  --sample-count "$samples"
  --split-logit-cap "$split_cap"
  --device "$device"
  --seed "$seed"
)
if [[ "$fixed_labels" == "1" ]]; then
  cmd+=(--fixed-labels)
fi

checkpoint_meta_sha256="$(sha256sum "$checkpoint_snapshot/meta.txt" | awk '{print $1}')"
shopt -s nullglob
checkpoint_params=("$checkpoint_snapshot"/param_*.pt)
shopt -u nullglob
if [[ "${#checkpoint_params[@]}" -eq 0 ]]; then
  echo "Checkpoint contains no parameter tensors: $checkpoint" >&2
  exit 1
fi
checkpoint_params_sha256="$(sha256sum "${checkpoint_params[@]}" | sha256sum | awk '{print $1}')"
mkdir -p "$output_dir"
{
  echo "schema=tyr.branching-constellation-cohort.v1"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "base_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "device_request=$device"
  echo "evaluation_split_seed=$train_seed"
  echo "fixed_labels=$fixed_labels"
  echo "training_config_checked=$training_config_checked"
  echo "training_fixed_labels=$training_fixed_labels"
  echo "training_vocab_size=$training_vocab_size"
  echo "training_coord_target_cap=$training_coord_cap"
  echo "dataset=$data"
  echo "dataset_sha256=$(sha256sum "$data" | awk '{print $1}')"
  echo "checkpoint=$checkpoint"
  echo "checkpoint_snapshot=$checkpoint_snapshot"
  echo "checkpoint_meta_sha256=$checkpoint_meta_sha256"
  echo "checkpoint_param_count=${#checkpoint_params[@]}"
  echo "checkpoint_params_sha256=$checkpoint_params_sha256"
  echo "branchingflows_sha256=$(sha256sum Tyr/Model/BranchingFlows.lean | awk '{print $1}')"
  echo "molecule_sha256=$(sha256sum Tyr/Model/BranchingFlows/Molecule.lean | awk '{print $1}')"
  echo "molecule_transformer_sha256=$(sha256sum Tyr/Model/BranchingFlows/MoleculeTransformer.lean | awk '{print $1}')"
  echo "molecule_runner_sha256=$(sha256sum Examples/BranchingFlows/MoleculeTrainGenerate.lean | awk '{print $1}')"
  echo "evaluator_sha256=$(sha256sum scripts/launch/evaluate_branching_constellations.py | awk '{print $1}')"
  if [[ -f "$(dirname "$checkpoint")/manifest.txt" ]]; then
    echo "checkpoint_run_manifest_sha256=$(sha256sum "$(dirname "$checkpoint")/manifest.txt" | awk '{print $1}')"
  fi
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/gpu=/'
  fi
  printf 'command='
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$output_dir/manifest.txt"

set +e
"${cmd[@]}" 2>&1 | tee "$output_dir/generation.log"
generation_pipeline_status=("${PIPESTATUS[@]}")
set -e
generation_status="${generation_pipeline_status[0]}"
generation_log_status="${generation_pipeline_status[1]}"
echo "generation_exit_code=$generation_status" >> "$output_dir/manifest.txt"
echo "generation_log_exit_code=$generation_log_status" >> "$output_dir/manifest.txt"
checkpoint_meta_sha256_after="$(sha256sum "$checkpoint_snapshot/meta.txt" | awk '{print $1}')"
shopt -s nullglob
checkpoint_params_after=("$checkpoint_snapshot"/param_*.pt)
shopt -u nullglob
checkpoint_params_sha256_after="missing"
if [[ "${#checkpoint_params_after[@]}" -gt 0 ]]; then
  checkpoint_params_sha256_after="$(sha256sum "${checkpoint_params_after[@]}" | sha256sum | awk '{print $1}')"
fi
checkpoint_unchanged=true
if [[ "$checkpoint_meta_sha256_after" != "$checkpoint_meta_sha256" ||
      "${#checkpoint_params_after[@]}" -ne "${#checkpoint_params[@]}" ||
      "$checkpoint_params_sha256_after" != "$checkpoint_params_sha256" ]]; then
  checkpoint_unchanged=false
fi
echo "checkpoint_unchanged=$checkpoint_unchanged" >> "$output_dir/manifest.txt"
if [[ "$generation_status" -ne 0 || "$generation_log_status" -ne 0 ||
      "$checkpoint_unchanged" != true ]]; then
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_dir/manifest.txt"
  if [[ "$generation_status" -ne 0 ]]; then
    exit "$generation_status"
  fi
  if [[ "$generation_log_status" -ne 0 ]]; then
    exit "$generation_log_status"
  fi
  echo "Checkpoint changed during generate-only execution: $checkpoint" >&2
  exit 1
fi

shopt -s nullglob
trajectory_files=("$output_dir"/constellation*_trajectory.jsonl)
shopt -u nullglob
echo "trajectory_count=${#trajectory_files[@]}" >> "$output_dir/manifest.txt"
if [[ "${#trajectory_files[@]}" -ne "$samples" ]]; then
  echo "Expected $samples trajectories, found ${#trajectory_files[@]} in $output_dir" \
    | tee "$output_dir/evaluation.log" >&2
  echo "evaluation_exit_code=1" >> "$output_dir/manifest.txt"
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_dir/manifest.txt"
  exit 1
fi

evaluation_cmd=(
  python3 scripts/launch/evaluate_branching_constellations.py
  --data "$data"
  --trajectories "$output_dir/constellation*_trajectory.jsonl"
  --max-len "$max_len"
  --split-seed "$train_seed"
  --out "$output_dir/evaluation.json"
)
set +e
"${evaluation_cmd[@]}" | tee "$output_dir/evaluation.log"
evaluation_pipeline_status=("${PIPESTATUS[@]}")
set -e
evaluation_status="${evaluation_pipeline_status[0]}"
evaluation_log_status="${evaluation_pipeline_status[1]}"
echo "evaluation_exit_code=$evaluation_status" >> "$output_dir/manifest.txt"
echo "evaluation_log_exit_code=$evaluation_log_status" >> "$output_dir/manifest.txt"
if [[ "$evaluation_status" -ne 0 || "$evaluation_log_status" -ne 0 ]]; then
  echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_dir/manifest.txt"
  if [[ "$evaluation_status" -ne 0 ]]; then
    exit "$evaluation_status"
  fi
  exit "$evaluation_log_status"
fi
echo "evaluation_sha256=$(sha256sum "$output_dir/evaluation.json" | awk '{print $1}')" >> "$output_dir/manifest.txt"
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_dir/manifest.txt"
