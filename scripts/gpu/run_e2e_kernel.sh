#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "Usage: $0 <KernelModule> <RunnerExe> <Label> [ExtraLeanBuildTarget ...]" >&2
  echo "Example: $0 Tyr.GPU.Kernels.Rotary RunRotary rotary" >&2
  exit 2
fi

kernel_module="$1"
runner_exe="$2"
label="$3"
shift 3
extra_build_targets=("$@")

: "${CUDA_HOME:?set CUDA_HOME to the CUDA toolkit}"

if [[ -z "${TYR_GPU_VENDORED_REF_RUNNER:-}" ]] && [[ -x "$PWD/scripts/gpu/run_vendored_reference.sh" ]]; then
  export TYR_GPU_VENDORED_REF_RUNNER="$PWD/scripts/gpu/run_vendored_reference.sh"
fi
LEAN_BIN="${TYR_LEAN_BIN:-$HOME/.elan/bin/lean}"
if [[ ! -x "$LEAN_BIN" ]]; then
  LEAN_BIN="$(command -v lean || true)"
fi
if [[ -z "${LEAN_BIN:-}" || ! -x "$LEAN_BIN" ]]; then
  echo "lean binary not found; set TYR_LEAN_BIN or install elan" >&2
  exit 127
fi

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
      echo "BLACKWELL"
      ;;
    B200|B300) echo "BLACKWELL" ;;
    A100) echo "AMPERE" ;;
    *) echo "HOPPER" ;;
  esac
}

trials="${E2E_TRIALS:-1}"
if ! [[ "$trials" =~ ^[0-9]+$ ]] || [[ "$trials" -lt 1 ]]; then
  echo "E2E_TRIALS must be a positive integer (got: $trials)" >&2
  exit 2
fi

gpu_target="$(detect_gpu_target)"
gpu_family="$(detect_gpu_family)"
export TYR_GPU_TARGET="${TYR_GPU_TARGET:-${gpu_target}}"
export TYR_GPU_FAMILY="${TYR_GPU_FAMILY:-${gpu_family}}"

echo "[1/4] Configure Lake and build libTyrC with the kernel's CUDA (${label}, GPU=${TYR_GPU_TARGET})"
lake -R -Kcuda="$CUDA_HOME" -Kgpu="$TYR_GPU_TARGET" -Kkernels="$kernel_module" --quiet build libtyr

runner_source="Examples/GPU/${runner_exe}.lean"
if [[ -f "Examples/GPU/${runner_exe}Exe.lean" ]]; then
  runner_source="Examples/GPU/${runner_exe}Exe.lean"
fi
use_source_runner=0

echo "[2/4] Build Lean executable (${runner_exe})"
if ! lake --quiet build "$runner_exe" "${extra_build_targets[@]}"; then
  if [[ -f "${runner_source}" ]]; then
    echo "[2/4] Falling back to Lean source runner (${runner_source})"
    use_source_runner=1
  else
    exit 1
  fi
fi

for i in $(seq 1 "$trials"); do
  echo "[3/4] (${i}/${trials}) Regenerate fixture tensors (${label})"
  if [[ "${use_source_runner}" -eq 1 ]]; then
    lake env "$LEAN_BIN" --run "${runner_source}" --gen-only --regen
  else
    lake env ./.lake/build/bin/"${runner_exe}" --gen-only --regen
  fi

  echo "[4/4] (${i}/${trials}) Run end-to-end check (${label})"
  if [[ "${use_source_runner}" -eq 1 ]]; then
    lake env "$LEAN_BIN" --run "${runner_source}"
  else
    lake env ./.lake/build/bin/"${runner_exe}"
  fi
done
