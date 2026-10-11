#!/usr/bin/env bash
set -euo pipefail

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

: "${CUDA_HOME:?set CUDA_HOME to the CUDA toolkit}"
gpu_target="$(detect_gpu_target)"
gpu_family="$(detect_gpu_family)"
export TYR_GPU_TARGET="${TYR_GPU_TARGET:-${gpu_target}}"
export TYR_GPU_FAMILY="${TYR_GPU_FAMILY:-${gpu_family}}"

echo "[1/2] Configure Lake and build libTyrC with the MhaH100 kernels' CUDA (GPU=${TYR_GPU_TARGET})"
lake -R -Kcuda="$CUDA_HOME" -Kgpu="$TYR_GPU_TARGET" -Kkernels=Tyr.GPU.Kernels.MhaH100 --quiet build libtyr

echo "[2/2] Run benchmark (Lean source runner Examples/GPU/RunMhaH100Train.lean)"
lake env "$LEAN_BIN" --run Examples/GPU/RunMhaH100Train.lean --benchmark --warmup 20 --bench-iters 500 --lr 200.0 --noise 0.5 "$@"
