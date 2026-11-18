#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

run() {
  echo
  echo ">>> $*"
  "$@"
}

run_built() {
  local target="$1"
  local binary=".lake/build/bin/$target"
  if [[ ! -x "$binary" ]]; then
    echo "Missing $binary; build it first with: lake build $target" >&2
    exit 1
  fi
  run "$binary"
}

run ./scripts/launch/run_shape_safety.sh
run_built BranchingFlowsContinuousTrain

echo
echo "Local showcase complete. GPU and model-weight demos are intentionally separate."
