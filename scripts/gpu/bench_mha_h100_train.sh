#!/usr/bin/env bash
set -euo pipefail

source ./env.sh

export LEAN_CC_FAST=1
export LD_LIBRARY_PATH="$PWD/external/wheels/torch/lib:$PWD/cc/build:${EBROOTGCCCORE:+${EBROOTGCCCORE}/lib64:}${LD_LIBRARY_PATH:-}"
LEAN_BIN="${TYR_LEAN_BIN:-$HOME/.elan/bin/lean}"
if [[ ! -x "$LEAN_BIN" ]]; then
  LEAN_BIN="$(command -v lean || true)"
fi
if [[ -z "${LEAN_BIN:-}" || ! -x "$LEAN_BIN" ]]; then
  echo "lean binary not found; set TYR_LEAN_BIN or install elan" >&2
  exit 127
fi

cpu_count() {
  local count
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  if command -v getconf >/dev/null 2>&1; then
    count="$(getconf _NPROCESSORS_ONLN 2>/dev/null || true)"
    if [[ "$count" =~ ^[0-9]+$ ]] && [[ "$count" -gt 0 ]]; then
      echo "$count"
      return
    fi
  fi
  if command -v sysctl >/dev/null 2>&1; then
    count="$(sysctl -n hw.logicalcpu 2>/dev/null || true)"
    if ! [[ "$count" =~ ^[0-9]+$ ]] || [[ "$count" -lt 1 ]]; then
      count="$(sysctl -n hw.ncpu 2>/dev/null || true)"
    fi
    if [[ "$count" =~ ^[0-9]+$ ]] && [[ "$count" -gt 0 ]]; then
      echo "$count"
      return
    fi
  fi
  echo 1
}

invalidate_generated_gpu_objects() {
  rm -f "$PWD"/cc/build/generated/*.o "$PWD"/cc/build/libTyrC.a "$PWD"/cc/build/libTyrC.so
}

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

gpu_target="$(detect_gpu_target)"
gpu_family="$(detect_gpu_family)"
export TYR_GPU_TARGET="${TYR_GPU_TARGET:-${gpu_target}}"
export TYR_GPU_FAMILY="${TYR_GPU_FAMILY:-${gpu_family}}"

echo "[1/5] Build Lean targets"
lake --quiet build +Tyr.GPU.Codegen.GenerateMain +Tyr.GPU.Kernels.MhaH100

echo "[2/5] Generate CUDA translation unit"
lake env "$LEAN_BIN" --run Tyr/GPU/Codegen/GenerateMain.lean Tyr.GPU.Kernels.MhaH100 --out-dir cc/src/generated

echo "[3/5] Build C++/CUDA runtime library (GPU=${TYR_GPU_TARGET}, family=${TYR_GPU_FAMILY})"
invalidate_generated_gpu_objects
make -C cc -j"$(cpu_count)" GPU="${TYR_GPU_TARGET}" GPU_FAMILY="${TYR_GPU_FAMILY}"

echo "[4/5] Use Lean source runner (Examples/GPU/RunMhaH100Train.lean)"

echo "[5/5] Run benchmark"
lake env "$LEAN_BIN" --run Examples/GPU/RunMhaH100Train.lean --benchmark --warmup 20 --bench-iters 500 --lr 200.0 --noise 0.5 "$@"
