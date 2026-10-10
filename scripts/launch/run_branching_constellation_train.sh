#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

data="${TYR_CONSTELLATION_DATA:-launch/generated/branching-constellations/dataset.jsonl}"
output_dir="${TYR_CONSTELLATION_OUTPUT:-launch/generated/branching-constellations/run-3000-cap2-vocab2}"
device="${TYR_CONSTELLATION_DEVICE:-cuda}"
steps="${TYR_CONSTELLATION_STEPS:-3000}"
batch="${TYR_CONSTELLATION_BATCH:-16}"
seed="${TYR_CONSTELLATION_SEED:-20260709}"
vocab_size="${TYR_CONSTELLATION_VOCAB_SIZE:-2}"
coord_cap="${TYR_CONSTELLATION_COORD_CAP:-2.0}"
splits_weight="${TYR_CONSTELLATION_SPLITS_WEIGHT:-5.0}"
fixed_labels="${TYR_CONSTELLATION_FIXED_LABELS:-0}"

if [[ ! "$fixed_labels" =~ ^(0|1)$ ]]; then
  echo "TYR_CONSTELLATION_FIXED_LABELS must be 0 or 1, got: $fixed_labels" >&2
  exit 1
fi

binary=".lake/build/bin/BranchingFlowsMoleculeTrainGenerate"
if [[ ! -x "$binary" ]]; then
  echo "Missing $binary; run lake build BranchingFlowsMoleculeTrainGenerate first." >&2
  exit 1
fi
if [[ ! -f "$data" ]]; then
  echo "Missing constellation dataset: $data" >&2
  exit 1
fi
if [[ -e "$output_dir" ]]; then
  echo "Refusing to overwrite an existing training path: $output_dir" >&2
  exit 1
fi

mkdir -p "$output_dir"

cmd=(
  "$binary"
  --require-data
  --data "$data"
  --out-prefix "$output_dir/sample"
  --checkpoint-dir "$output_dir/checkpoint"
  --architecture full
  --hidden-dim 128
  --heads 4
  --head-dim 32
  --mlp 512
  --rff-dim 32
  --layers 4
  --coord-update-layers 2
  --max-len 12
  --vocab-size "$vocab_size"
  --coord-target-cap "$coord_cap"
  --steps "$steps"
  --total-steps "$steps"
  --warmup-steps 100
  --cooldown-steps 500
  --batch-size "$batch"
  --lr 0.001
  --lr-end 0.0001
  --grad-clip 1.0
  --splits-weight "$splits_weight"
  --del-weight 1.0
  --deletion-pad 0.0
  --no-generate
  --log-every 50
  --device "$device"
  --seed "$seed"
)
if [[ "$fixed_labels" == "1" ]]; then
  cmd+=(--fixed-labels)
fi

{
  echo "schema=tyr.branching-constellation-run.v1"
  echo "started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "base_commit=$(git rev-parse HEAD 2>/dev/null || echo unknown)"
  echo "device_request=$device"
  echo "vocab_size=$vocab_size"
  echo "coord_target_cap=$coord_cap"
  echo "fixed_labels=$fixed_labels"
  echo "dataset=$data"
  echo "dataset_sha256=$(sha256sum "$data" | awk '{print $1}')"
  echo "branchingflows_sha256=$(sha256sum Tyr/Model/BranchingFlows.lean | awk '{print $1}')"
  echo "molecule_sha256=$(sha256sum Tyr/Model/BranchingFlows/Molecule.lean | awk '{print $1}')"
  echo "molecule_train_sha256=$(sha256sum Tyr/Model/BranchingFlows/MoleculeTrain.lean | awk '{print $1}')"
  echo "molecule_transformer_sha256=$(sha256sum Tyr/Model/BranchingFlows/MoleculeTransformer.lean | awk '{print $1}')"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | sed 's/^/gpu=/'
  fi
  printf 'command='
  printf '%q ' "${cmd[@]}"
  printf '\n'
} > "$output_dir/manifest.txt"

set +e
"${cmd[@]}" 2>&1 | tee "$output_dir/training.log"
status="${PIPESTATUS[0]}"
set -e
echo "finished_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$output_dir/manifest.txt"
echo "exit_code=$status" >> "$output_dir/manifest.txt"
exit "$status"
