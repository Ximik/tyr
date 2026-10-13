#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi is required for the GB10 capture." >&2
  exit 1
fi

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\r')"
if [[ "$gpu_name" != *GB10* ]]; then
  echo "Expected an NVIDIA GB10, found: $gpu_name" >&2
  exit 1
fi

if ! git diff --quiet --ignore-submodules=all || ! git diff --cached --quiet --ignore-submodules=all; then
  echo "Refusing hardware capture from a dirty tracked worktree." >&2
  echo "Create a clean worktree at the exact launch commit first." >&2
  exit 1
fi

output_dir="${TYR_LAUNCH_OUTPUT_DIR:-launch/generated/gb10}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_dir="$output_dir/$timestamp"
mkdir -p "$run_dir"

{
  echo "timestamp_utc=$timestamp"
  echo "commit=$(git rev-parse HEAD)"
  echo "branch=$(git branch --show-current)"
  nvidia-smi --query-gpu=name,driver_version,pci.bus_id --format=csv,noheader
  "$CUDA_HOME/bin/nvcc" --version 2>/dev/null || true
} | tee "$run_dir/provenance.txt"

echo
echo "Running GB10 generated-kernel suite..."
set +e
./scripts/gpu/test_leantest_gb10_e2e.sh 2>&1 | tee "$run_dir/gb10-suite.txt"
gb10_status="${PIPESTATUS[0]}"
set -e
echo "$gb10_status" > "$run_dir/gb10-exit-code.txt"
if [[ "$gb10_status" -ne 0 ]]; then
  echo "GB10 suite failed; logs preserved in $run_dir" >&2
  exit "$gb10_status"
fi

echo
echo "Running reusable parity suite..."
set +e
./scripts/gpu/test_parity_suite.sh 2>&1 | tee "$run_dir/parity-suite.txt"
parity_status="${PIPESTATUS[0]}"
set -e
echo "$parity_status" > "$run_dir/parity-exit-code.txt"
if [[ "$parity_status" -ne 0 ]]; then
  echo "Parity suite failed; logs preserved in $run_dir" >&2
  exit "$parity_status"
fi

echo
echo "PASS: GB10 provenance and parity evidence saved to $run_dir"
