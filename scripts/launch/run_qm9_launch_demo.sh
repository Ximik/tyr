#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

jsonl="${TYR_QM9_JSONL:-}"
if [[ -z "$jsonl" ]]; then
  jsonl="$(./scripts/launch/prepare_qm9.sh | tail -n1)"
fi

output_dir="${TYR_LAUNCH_OUTPUT_DIR:-launch/generated/qm9-learned}"
mkdir -p "$output_dir"
prefix="$output_dir/sample"
checkpoint="$output_dir/checkpoint"

steps="${TYR_QM9_STEPS:-2000}"
batch_size="${TYR_QM9_BATCH_SIZE:-4}"
device="${TYR_QM9_DEVICE:-cuda}"
log_every="${TYR_QM9_LOG_EVERY:-10}"
sample_steps="${TYR_QM9_SAMPLE_STEPS:-48}"

echo "QM9 learned branching demo"
echo "data=$jsonl steps=$steps batch_size=$batch_size device=$device"

# These outputs are owned by this harness. Remove prior trajectory frames so a
# shorter follow-up sampling schedule cannot inherit stale storyboard panels.
rm -f "${prefix}"_step_*.xyz "${prefix}"_step_*.svg

lake exe BranchingFlowsMoleculeTrainGenerate \
  --profile smoke \
  --require-data \
  --data "$jsonl" \
  --out-prefix "$prefix" \
  --checkpoint-dir "$checkpoint" \
  --architecture full \
  --hidden-dim 128 \
  --heads 4 \
  --head-dim 32 \
  --mlp 512 \
  --rff-dim 32 \
  --layers 4 \
  --coord-update-layers 2 \
  --max-len 32 \
  --steps "$steps" \
  --total-steps "$steps" \
  --batch-size "$batch_size" \
  --sample-count 1 \
  --sample-steps "$sample_steps" \
  --log-every "$log_every" \
  --device "$device" \
  --seed 20260709 | tee "$output_dir/training-and-generation.txt"

for xyz in "${prefix}"_target.xyz "${prefix}"_source.xyz "${prefix}"_generated.xyz "${prefix}"_step_*.xyz; do
  [[ -f "$xyz" ]] || continue
  ./scripts/launch/render_xyz.py "$xyz" "${xyz%.xyz}.svg"
done

./scripts/launch/render_branching_storyboard.py "$prefix" "$output_dir/branching-storyboard.svg"

echo "PASS: QM9 training, learned generation, trajectory, and storyboard saved in $output_dir"
