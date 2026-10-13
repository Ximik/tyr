#!/usr/bin/env bash
set -euo pipefail

# This suite must not silently use CPU stubs.
if [[ -z "${CUDA_HOME:-}" ]]; then
  echo "CUDA_HOME is not set; the GB10 parity suite requires a CUDA toolkit" >&2
  exit 127
fi
export PATH="$CUDA_HOME/bin:$PATH"

export TYR_GPU_FAMILY=BLACKWELL
export TYR_GPU_VENDORED_REF_RUNNER="${TYR_GPU_VENDORED_REF_RUNNER:-$PWD/scripts/gpu/run_vendored_reference.sh}"

detect_gpu_target() {
  if [[ -n "${TYR_GPU_TARGET:-}" ]]; then
    echo "${TYR_GPU_TARGET}"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GB10"
    return
  fi
  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\r')"
  case "${gpu_name}" in
    *GB10*) echo "GB10" ;;
    *B300*) echo "B300" ;;
    *B200*) echo "B200" ;;
    *) echo "GB10" ;;
  esac
}

export TYR_GPU_TARGET="${TYR_GPU_TARGET:-$(detect_gpu_target)}"
modules=(
  Tyr.GPU.Kernels.MhaGB10
  Tyr.GPU.Kernels.FusedLayerNorm
  Tyr.GPU.Kernels.FusedRMSNorm
  Tyr.GPU.Kernels.RKCombine
  Tyr.GPU.Kernels.BrownianSample
)
echo "[1/2] Configure Lake and build TestGPUGB10E2E with the kernels' CUDA (GPU=${TYR_GPU_TARGET})"
lake -R -Kcuda="$CUDA_HOME" -Kgpu="$TYR_GPU_TARGET" -Kkernels="${modules[*]}" --quiet build TestGPUGB10E2E

echo "[2/2] Run LeanTest GB10 suite"
lake env ./.lake/build/bin/TestGPUGB10E2E "$@"
