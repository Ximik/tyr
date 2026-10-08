#!/usr/bin/env bash
# Invoke under the host-wide GPU flock (see cuda-smoke.yml).
set -euo pipefail
qualification_root=${TYR_QUALIFICATION_ROOT:-"$HOME/tyr-qualification"}
python_bin="$qualification_root/venv/bin/python"
reports=(gpu)
if [[ "${TYR_QUALIFY_MODELS:-false}" == true ]]; then reports+=(models); fi
mkdir -p output/qualification
# A persistent runner must never publish a previous run's successful report.
rm -f output/qualification/{gpu,models,summary}.json
finish() {
  local status=$?
  if ! python3 scripts/qualification/summarize.py --directory output/qualification "${reports[@]}"; then status=1; fi
  exit "$status"
}
trap finish EXIT

# The lock coordinates Tyr qualification jobs across both repository runners.
# Refuse to overlap an unrelated CUDA workload that does not use that lock.
active=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
if [[ -n "$active" ]]; then
  echo "GPU is occupied by an unrelated workload; retry after it finishes." >&2
  exit 1
fi
export PATH="$(dirname "$python_bin"):$HOME/.elan/bin:${CUDA_HOME:-/usr/local/cuda}/bin:$PATH"
# Preparation shares the GPU lock too: another repository job cannot replace
# the venv or partially download a fixture while this run uses those inputs.
bash scripts/qualification/setup_python.sh
if [[ "${TYR_QUALIFY_MODELS:-false}" == true ]]; then
  python3 scripts/qualification/prepare.py --cache "$qualification_root/fixtures" \
    --download --paths-output output/qualification/fixture-paths.json
fi
export TYR_QUALIFICATION_PYTHON="$python_bin"
# The Hopper vendor kernel includes Python.h through Torch's pybind header.
# Make must discover headers from this prepared runtime on a fresh checkout.
export PYTHON="$python_bin"
export GPU=${GPU:-GB10}
export TYR_GPU_TARGET="$GPU"
python3 scripts/qualification/gpu_plan.py "$GPU" > output/qualification/gpu-plan.json
TYR_GPU_FAMILY=$(python3 scripts/qualification/gpu_plan.py "$GPU" --field family)
export TYR_GPU_FAMILY
gpu_runner=$(python3 scripts/qualification/gpu_plan.py "$GPU" --field runner)
mapfile -t modules < <(python3 scripts/qualification/gpu_plan.py "$GPU" --field modules)
export TYR_GPU_CODEGEN_MODULE="${modules[*]}"
cuda_libraries=$(python3 scripts/qualification/cuda_runtime.py --libtorch external/wheels/torch --cuda-home "${CUDA_HOME:-/usr/local/cuda}" --previous="${LD_LIBRARY_PATH:-}")
export LD_LIBRARY_PATH="$cuda_libraries"
source ./env.sh

python3 scripts/qualification/run.py --kind gpu --python "$python_bin" \
  --check-runtime --report output/qualification/runtime.json
# Keep the known codegen bootstrap explicit on a fresh checkout.
TYR_SKIP_GPU_CODEGEN=1 lake build "${modules[@]}"
targets=("$gpu_runner" RunMhaH100Decode LagunaModelTest)
if [[ "${TYR_QUALIFY_MODELS:-false}" == true ]]; then
  targets+=(Qwen3TTSEndToEnd Qwen3ASRTranscribe)
fi
lake build "${targets[@]}"
python3 scripts/check_ffi_abi.py
python3 scripts/check_ffi_abi.py --header cc/src/tyr_owned_kv_abi.h --generated .lake/build/ir/Tyr/Inference/OwnedKV.c
python3 scripts/qualification/run.py --kind gpu --python "$python_bin" \
  --report output/qualification/gpu.json
if [[ "${TYR_QUALIFY_MODELS:-false}" == true ]]; then
  python3 scripts/qualification/run.py --kind models --python "$python_bin" \
    --cache "$qualification_root/fixtures" --report output/qualification/models.json
fi
