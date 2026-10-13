#!/usr/bin/env bash
set -euo pipefail

export TYR_GPU_VENDORED_REF_RUNNER="${TYR_GPU_VENDORED_REF_RUNNER:-$PWD/scripts/gpu/run_vendored_reference.sh}"

detect_gpu_target() {
  if [[ -n "${TYR_GPU_TARGET:-}" ]]; then
    echo "${TYR_GPU_TARGET}"
    return
  fi
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "H100"
    return
  fi
  local gpu_name
  gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n1 | tr -d '\r')"
  case "${gpu_name}" in
    *GB10*) echo "GB10" ;;
    *B300*) echo "B300" ;;
    *B200*) echo "B200" ;;
    *A100*) echo "A100" ;;
    *) echo "H100" ;;
  esac
}

detect_gpu_family() {
  if [[ -n "${TYR_GPU_FAMILY:-}" ]]; then
    echo "${TYR_GPU_FAMILY}"
    return
  fi
  case "$(detect_gpu_target)" in
    GB10)
      # GB10 is a Blackwell-generation product, but physical SM121 does not
      # support SM100a tcgen05/TMEM. The native source gate enforces that
      # instruction distinction before NVCC.
      echo "BLACKWELL"
      ;;
    B200|B300) echo "BLACKWELL" ;;
    A100) echo "AMPERE" ;;
    *) echo "HOPPER" ;;
  esac
}

: "${CUDA_HOME:?set CUDA_HOME to the CUDA toolkit}"
gpu_target="$(detect_gpu_target)"
gpu_family="$(detect_gpu_family)"
export TYR_GPU_TARGET="${TYR_GPU_TARGET:-${gpu_target}}"
export TYR_GPU_FAMILY="${TYR_GPU_FAMILY:-${gpu_family}}"

extract_test_filter() {
  local prev=""
  for arg in "$@"; do
    if [[ "${prev}" == "--filter" ]]; then
      echo "${arg}"
      return
    fi
    prev="${arg}"
  done
}

select_modules_from_filter() {
  # The LeanTest GPU executable links all GPU test modules even when `--filter`
  # narrows runtime execution, so codegen still needs the full kernel set.
  printf '%s\n' \
    Tyr.GPU.Kernels.Copy \
    Tyr.GPU.Kernels.Rotary \
    Tyr.GPU.Kernels.FusedLayerNorm \
    Tyr.GPU.Kernels.FusedRMSNorm \
    Tyr.GPU.Kernels.MhaH100
}

test_filter="$(extract_test_filter "$@")"
mapfile -t modules < <(select_modules_from_filter "${test_filter}")
echo "[1/2] Configure Lake and build TestGPUE2E with the kernels' CUDA (GPU=${TYR_GPU_TARGET})"
lake -R -Kcuda="$CUDA_HOME" -Kgpu="$TYR_GPU_TARGET" -Kkernels="${modules[*]}" --quiet build TestGPUE2E

echo "[2/2] Run LeanTest GPU suite"
lake env ./.lake/build/bin/TestGPUE2E "$@"
