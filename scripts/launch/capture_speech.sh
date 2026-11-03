#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 qwen3-tts-model-directory" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

model_dir="$1"
text="${TYR_LAUNCH_TEXT:-Hello from Tyr, a dependently typed machine learning framework built in Lean.}"
output_dir="${TYR_LAUNCH_OUTPUT_DIR:-launch/generated/speech}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$output_dir/$timestamp"
mkdir -p "$run_dir"

{
  echo "timestamp_utc=$timestamp"
  echo "commit=$(git rev-parse HEAD)"
  echo "model_dir=$model_dir"
  echo "text=$text"
  echo "boundary=Lean tokenization and generation; Python speech-codec decode bridge"
} | tee "$run_dir/provenance.txt"

set +e
lake exe Qwen3TTSEndToEnd \
  --model-dir "$model_dir" \
  --text "$text" \
  --top-k 50 \
  --subtalker-top-k 50 \
  --codes-path "$run_dir/codes.txt" \
  --wav-path "$run_dir/tyr-launch.wav" 2>&1 | tee "$run_dir/transcript.txt"
status="${PIPESTATUS[0]}"
set -e
echo "$status" > "$run_dir/exit-code.txt"

if [[ "$status" -ne 0 ]]; then
  echo "Speech demo failed; logs preserved in $run_dir" >&2
  exit "$status"
fi

echo "PASS: speech record saved to $run_dir"
