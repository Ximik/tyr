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

# Preserve the requested module through nested Lake/Make builds. Otherwise the
# generator build falls back to MhaH100 and may compile an unrelated,
# architecture-incompatible generated translation unit.
export TYR_GPU_CODEGEN_MODULE="${kernel_module}"

source ./env.sh

export LD_LIBRARY_PATH="$PWD/external/wheels/torch/lib:$PWD/cc/build:${EBROOTGCCCORE:+${EBROOTGCCCORE}/lib64:}${LD_LIBRARY_PATH:-}"
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

trials="${E2E_TRIALS:-1}"
if ! [[ "$trials" =~ ^[0-9]+$ ]] || [[ "$trials" -lt 1 ]]; then
  echo "E2E_TRIALS must be a positive integer (got: $trials)" >&2
  exit 2
fi

echo "[1/6] Build Lean kernel + generator (${label})"
lake --quiet build +Tyr.GPU.Codegen.GenerateMain "+${kernel_module}"

echo "[2/6] Generate CUDA translation unit (${label})"
lake env "$LEAN_BIN" --run Tyr/GPU/Codegen/GenerateMain.lean "$kernel_module" --out-dir cc/src/generated

gpu_target="$(detect_gpu_target)"
gpu_family="$(detect_gpu_family)"
export TYR_GPU_TARGET="${TYR_GPU_TARGET:-${gpu_target}}"
export TYR_GPU_FAMILY="${TYR_GPU_FAMILY:-${gpu_family}}"
runner_source="Examples/GPU/${runner_exe}.lean"
if [[ -f "Examples/GPU/${runner_exe}Exe.lean" ]]; then
  runner_source="Examples/GPU/${runner_exe}Exe.lean"
fi
use_source_runner=0

echo "[3/6] Build C++/CUDA runtime library (${label}, GPU=${TYR_GPU_TARGET}, family=${TYR_GPU_FAMILY})"
invalidate_generated_gpu_objects
make -C cc -j"$(cpu_count)" GPU="${TYR_GPU_TARGET}" GPU_FAMILY="${TYR_GPU_FAMILY}"

echo "[4/6] Build Lean executable (${runner_exe})"
if ! lake --quiet build "$runner_exe" "${extra_build_targets[@]}"; then
  if [[ -f "${runner_source}" ]]; then
    echo "[4/6] Falling back to Lean source runner (${runner_source})"
    use_source_runner=1
  else
    exit 1
  fi
fi

for i in $(seq 1 "$trials"); do
  echo "[5/6] (${i}/${trials}) Regenerate fixture tensors (${label})"
  if [[ "${use_source_runner}" -eq 1 ]]; then
    lake env "$LEAN_BIN" --run "${runner_source}" --gen-only --regen
  else
    lake env ./.lake/build/bin/"${runner_exe}" --gen-only --regen
  fi

  echo "[6/6] (${i}/${trials}) Run end-to-end check (${label})"
  if [[ "${use_source_runner}" -eq 1 ]]; then
    lake env "$LEAN_BIN" --run "${runner_source}"
  else
    lake env ./.lake/build/bin/"${runner_exe}"
  fi
done
