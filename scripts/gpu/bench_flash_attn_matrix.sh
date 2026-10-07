#!/usr/bin/env bash
set -euo pipefail

source ./load_modules.sh

export LEAN_CC="${LEAN_CC:-$PWD/scripts/lean_cc_wrapper.sh}"
export LEAN_CC_FAST="${LEAN_CC_FAST:-1}"
export LAKE_NUM_JOBS="${LAKE_NUM_JOBS:-1}"
export TYR_BUILD_TYRC_DYLIB="${TYR_BUILD_TYRC_DYLIB:-0}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="$PWD/external/wheels/torch/lib:$PWD/cc/build:${EBROOTGCCCORE:+${EBROOTGCCCORE}/lib64:}${LD_LIBRARY_PATH:-}"
gpu_codegen_module="${TYR_GPU_CODEGEN_MODULE:-Tyr.GPU.Kernels.MhaH100}"

cpu_count() {
  if command -v nproc >/dev/null 2>&1; then
    nproc
    return
  fi
  echo 1
}

usage() {
  cat <<'EOF'
Usage: scripts/gpu/bench_flash_attn_matrix.sh [wrapper flags] [-- benchmark flags]

Wrapper flags:
  --ensure-native   rebuild the requested GPU-backed benchmark target
  --skip-build      do not build the Lean benchmark executable
  --build-only      stop after build steps
  --help            show this help

Environment:
  TYR_GPU_CODEGEN_MODULE=Tyr.GPU.Kernels.FlashAttn3
                    build the exact repo-local FA3 row used by
                    `--case future_flash_256x64 --backend flash_attention`

Benchmark flags are passed through to RunFlashAttnBench, for example:
  --case native_now
  --backend tyr_runtime,torch_sdpa
  --warmup 20
  --iters 200
  --repeats 5
  --seed 20260422
  --jsonl-out /tmp/flash_attn_bench.jsonl
  --jsonl-stdout
  --strict
  --list-cases
  --list-backends
EOF
}

ensure_native=0
skip_build=0
build_only=0
bench_args=()

while (($#)); do
  case "$1" in
    --ensure-native)
      ensure_native=1
      shift
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    --build-only)
      build_only=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    --)
      shift
      while (($#)); do
        bench_args+=("$1")
        shift
      done
      ;;
    *)
      bench_args+=("$1")
      shift
      ;;
  esac
done

if (( ! skip_build )); then
  echo "[1/3] Build benchmark executable"
  if (( ensure_native )); then
    echo "        codegen module: ${gpu_codegen_module}"
    lake -R run buildGpuTarget -- "${gpu_codegen_module}" RunFlashAttnBench
  else
    lake -R --quiet build RunFlashAttnBench
  fi
fi

if (( ensure_native )); then
  echo "[2/3] Native runtime regeneration was included in the buildGpuTarget step"
else
  echo "[2/3] Native runtime rebuild skipped"
fi

if (( build_only )); then
  echo "[3/3] Build-only requested; stopping here"
  exit 0
fi

echo "[3/3] Run benchmark scaffold"
lake -R run runBuiltTarget -- RunFlashAttnBench "${bench_args[@]}"
