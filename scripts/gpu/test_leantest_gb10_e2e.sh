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
# Keep Lake's extern-lib rebuild on the same generated kernel set.  Without
# this, building TestGPUGB10E2E regenerates/relinks libtyr with Lake's default
# single MhaH100 module and drops the GB10/RK/Brownian symbols generated above.
export TYR_GPU_CODEGEN_MODULE="${modules[*]}"
generator_targets=(
  +Tyr.GPU.Codegen.GenerateMain
  +Tyr.GPU.Kernels.MhaGB10
  +Tyr.GPU.Kernels.FusedLayerNorm
  +Tyr.GPU.Kernels.FusedRMSNorm
  +Tyr.GPU.Kernels.RKCombine
  +Tyr.GPU.Kernels.BrownianSample
)

echo "[1/5] Build Lean kernel generator inputs"
lake --quiet build "${generator_targets[@]}"

echo "[2/5] Generate CUDA translation units"
lake env "$LEAN_BIN" --run Tyr/GPU/Codegen/GenerateMain.lean "${modules[@]}" --out-dir cc/src/generated

echo "[3/5] Build C++/CUDA runtime library (GPU=${TYR_GPU_TARGET}, family=${TYR_GPU_FAMILY})"
invalidate_generated_gpu_objects
make -C cc -j"$(cpu_count)" GPU="${TYR_GPU_TARGET}" GPU_FAMILY="${TYR_GPU_FAMILY}"

echo "[4/5] Build LeanTest GB10 executable"
lake --quiet build TestGPUGB10E2E

echo "[5/5] Run LeanTest GB10 suite"
lake env ./.lake/build/bin/TestGPUGB10E2E "$@"
