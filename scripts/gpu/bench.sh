#!/usr/bin/env bash
set -euo pipefail
source ./load_modules.sh

gpu_index="${BENCH_GPU_INDEX:-0}"
idle_wait_seconds="${BENCH_IDLE_WAIT_SECONDS:-30}"
monitor_interval_seconds="${BENCH_GPU_MONITOR_INTERVAL:-0.5}"

gpu_compute_processes() {
  nvidia-smi -i "$gpu_index" \
    --query-compute-apps=pid,process_name,used_memory \
    --format=csv,noheader,nounits
}

gpu_utilization() {
  nvidia-smi -i "$gpu_index" --query-gpu=utilization.gpu \
    --format=csv,noheader,nounits | head -n 1 | tr -d ' '
}

wait_for_idle_gpu() {
  local label="$1"
  local attempt processes util
  for ((attempt = 0; attempt <= idle_wait_seconds; ++attempt)); do
    processes="$(gpu_compute_processes)"
    util="$(gpu_utilization)"
    if [[ -z "$processes" && "$util" == "0" ]]; then
      echo "gpu_idle_gate=$label gpu=$gpu_index utilization_pct=0 compute_processes=none"
      return 0
    fi
    if (( attempt == idle_wait_seconds )); then
      echo "benchmark aborted: GPU $gpu_index was not idle before $label" >&2
      echo "utilization_pct=$util" >&2
      if [[ -n "$processes" ]]; then echo "$processes" >&2; fi
      return 75
    fi
    sleep 1
  done
}

pid_is_descendant_or_self() {
  local pid="$1" root="$2" parent
  while [[ "$pid" =~ ^[0-9]+$ && "$pid" -gt 1 ]]; do
    if [[ "$pid" == "$root" ]]; then
      return 0
    fi
    parent="$(ps -o ppid= -p "$pid" 2>/dev/null || true)"
    parent="${parent//[[:space:]]/}"
    if [[ -z "$parent" || "$parent" == "$pid" ]]; then
      break
    fi
    pid="$parent"
  done
  return 1
}

foreign_gpu_processes() {
  local allowed_pid="$1"
  local line pid
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    pid="${line%%,*}"
    pid="${pid//[[:space:]]/}"
    if [[ -n "$allowed_pid" ]] && pid_is_descendant_or_self "$pid" "$allowed_pid"; then
      continue
    fi
    printf '%s\n' "$line"
  done < <(gpu_compute_processes)
}

# Run one timed backend while sampling the CUDA process table. The measured
# executable itself is allowed; any other CUDA PID makes the run invalid even
# when it appears only after the pre-run idle check.
run_on_idle_gpu() {
  local label="$1"
  shift
  local contamination_file measured_pid monitor_pid status foreign_after
  wait_for_idle_gpu "$label"
  contamination_file="$(mktemp "/tmp/tyr_gpu_contamination_${label}.XXXXXX")"
  "$@" &
  measured_pid=$!
  (
    while kill -0 "$measured_pid" 2>/dev/null; do
      foreign_gpu_processes "$measured_pid" >> "$contamination_file"
      sleep "$monitor_interval_seconds"
    done
  ) &
  monitor_pid=$!
  if wait "$measured_pid"; then status=0; else status=$?; fi
  wait "$monitor_pid" || true
  foreign_after="$(foreign_gpu_processes "")"
  if [[ -n "$foreign_after" ]]; then printf '%s\n' "$foreign_after" >> "$contamination_file"; fi
  if [[ -s "$contamination_file" ]]; then
    echo "benchmark invalid: foreign CUDA process observed during $label" >&2
    sort -u "$contamination_file" >&2
    rm -f "$contamination_file"
    return 76
  fi
  rm -f "$contamination_file"
  return "$status"
}
if [[ "${1:-}" == "--idle-run" ]]; then
  shift
  label="${1:-external}"
  if [[ $# -lt 2 ]]; then
    echo "usage: $0 --idle-run <label> <command> [args...]" >&2
    exit 2
  fi
  shift
  run_on_idle_gpu "$label" "$@"
  exit $?
fi

case_id="${1:-}"
if [[ -z "$case_id" ]]; then
  echo "usage: $0 <copy|rotary|layernorm|rmsnorm|loss|optimizer|bf16_gemm|mha_gb10|all>" >&2
  exit 2
fi
shift
if [[ "$case_id" == "all" ]]; then
  for name in copy rotary layernorm rmsnorm loss optimizer bf16_gemm mha_gb10; do "$0" "$name" "$@"; done
  exit 0
fi
case "$case_id" in
  copy) module=Tyr.GPU.Kernels.Copy; exe=RunCopy; py=bench_copy_torch.py; default_iters=1000; family=BLACKWELL ;;
  rotary) module=Tyr.GPU.Kernels.Rotary; exe=RunRotary; py=bench_rotary_torch.py; default_iters=200; family=BLACKWELL ;;
  layernorm) module=Tyr.GPU.Kernels.FusedLayerNorm; exe=RunLayerNorm; py=bench_layernorm_torch.py; default_iters=200; family=BLACKWELL ;;
  rmsnorm) module=Tyr.GPU.Kernels.FusedRMSNorm; exe=RunRMSNorm; py=bench_rmsnorm_torch.py; default_iters=200; family=BLACKWELL ;;
  loss) module=Tyr.GPU.Kernels.Loss; exe=RunLoss; py=bench_loss_torch.py; default_iters=200; family=BLACKWELL ;;
  optimizer) module=Tyr.GPU.Kernels.Optimizer; exe=RunOptimizer; py=bench_optimizer_torch.py; default_iters=200; family=BLACKWELL ;;
  bf16_gemm) module=Tyr.GPU.Kernels.Bf16Gemm; exe=RunB200Bf16Gemm; py=bench_bf16_gemm_torch.py; default_iters=200; family=BLACKWELL ;;
  mha_gb10) module=Tyr.GPU.Kernels.MhaGB10; exe=RunMhaGB10; py=bench_mha_gb10_torch.py; default_iters=200; family=BLACKWELL ;;
  *) echo "unknown benchmark case: $case_id" >&2; exit 2 ;;
esac
warmup="${BENCH_WARMUP:-20}"
iters="${BENCH_ITERS:-$default_iters}"
repeats="${BENCH_REPEATS:-7}"
out="${BENCH_JSONL_OUT:-/tmp/tyr_${case_id}_bench.jsonl}"
run_id="${case_id}_bench_$(date -u +%Y%m%dT%H%M%SZ)_$$"
export TYR_GPU_CODEGEN_MODULE="$module"
export TYR_GPU_TARGET="${TYR_GPU_TARGET:-GB10}"
export TYR_GPU_FAMILY="${TYR_GPU_FAMILY:-$family}"
export TYR_BUILD_TYRC_DYLIB=0
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-/tmp/tyr_torchinductor_cache}"
export LD_LIBRARY_PATH="$PWD/external/python/torch/lib:$PWD/cc/build:${LD_LIBRARY_PATH:-}"
# Build the selected registration before generation, then replace the native
# archive deliberately. The normal extern-lib dependency currently has a cycle
# through the kernel registration dynlib and can otherwise emit CUDA from the
# previous source revision; keep this explicit sequence until that graph is cut.
LEAN_BIN="${LEAN_BIN:-$(lake -R env lean --print-prefix)/bin/lean}"
if [[ "${BENCH_SKIP_BUILD:-0}" == "1" ]]; then build_skipped=true; else build_skipped=false; fi
if [[ "${BENCH_SKIP_BUILD:-0}" != "1" ]]; then
  TYR_SKIP_GPU_CODEGEN=1 lake -R --quiet build +Tyr.GPU.Codegen.GenerateMain "$module:dynlib"
  lake -R env "$LEAN_BIN" --run Tyr/GPU/Codegen/GenerateMain.lean "$module" --out-dir cc/src/generated
  rm -f cc/build/generated/*.o cc/build/libTyrC.a cc/build/libTyrC.so
  make -C cc -j"${TYR_GPU_BUILD_JOBS:-$(getconf _NPROCESSORS_ONLN)}" GPU="$TYR_GPU_TARGET" GPU_FAMILY="$TYR_GPU_FAMILY"
  TYR_SKIP_GPU_CODEGEN=1 lake -R build "$exe"
else
  test -x "./.lake/build/bin/$exe"
  test -f "cc/src/generated/${module//./_}.cu"
  test -f "cc/build/generated/${module//./_}.o"
fi
rm -f "$out"
run_on_idle_gpu tyr "./.lake/build/bin/$exe" --benchmark --run-id "$run_id" \
  --warmup "$warmup" --iters "$iters" --repeats "$repeats" --jsonl-out "$out" "$@"
run_on_idle_gpu pytorch .venv-gpu/bin/python "benchmarks/$py" --run-id "$run_id" \
  --warmup "$warmup" --iters "$iters" --repeats "$repeats" --jsonl-out "$out" "$@"

# Append one run-scoped provenance record shared by every backend row. Keep
# expensive/static metadata out of the timed executables and collect it only
# after all measurements have completed.
generated_source="cc/src/generated/${module//./_}.cu"
generated_object="cc/build/generated/${module//./_}.o"
git_revision="$(git rev-parse HEAD)"
if [[ -n "$(git status --porcelain)" ]]; then git_dirty=true; else git_dirty=false; fi
generated_sha256="$(sha256sum "$generated_source" | awk '{print $1}')"
IFS=',' read -r gpu_name gpu_uuid compute_capability driver_version < <(
  nvidia-smi --query-gpu=name,uuid,compute_cap,driver_version --format=csv,noheader,nounits | head -n 1
)
gpu_name="${gpu_name# }"; gpu_name="${gpu_name% }"
gpu_uuid="${gpu_uuid# }"; gpu_uuid="${gpu_uuid% }"
compute_capability="${compute_capability# }"; compute_capability="${compute_capability% }"
driver_version="${driver_version# }"; driver_version="${driver_version% }"
cuda_compiler="$(nvcc --version | tail -n 1)"
resource_usage="$(cuobjdump --dump-resource-usage "$generated_object" 2>&1)"
jq -nc \
  --arg runId "$run_id" --arg module "$module" \
  --arg target "$TYR_GPU_TARGET" --arg family "$TYR_GPU_FAMILY" \
  --argjson buildSkipped "$build_skipped" \
  --arg gitRevision "$git_revision" --argjson gitDirty "$git_dirty" \
  --arg generatedSource "$generated_source" --arg generatedSourceSha256 "$generated_sha256" \
  --arg gpuName "$gpu_name" --arg gpuUuid "$gpu_uuid" \
  --arg computeCapability "$compute_capability" --arg driverVersion "$driver_version" \
  --arg cudaCompiler "$cuda_compiler" --arg resourceUsage "$resource_usage" \
  '{event:"meta",schemaVersion:1,runId:$runId,module:$module,physicalTarget:$target,
    idleGpuGate:"nvidia-smi pre-run zero utilization plus foreign-PID monitoring",
    implementationFamily:$family,gitRevision:$gitRevision,gitDirty:$gitDirty,
    buildSkipped:$buildSkipped,
    generatedSource:$generatedSource,generatedSourceSha256:$generatedSourceSha256,
    gpuName:$gpuName,gpuUuid:$gpuUuid,computeCapability:$computeCapability,
    driverVersion:$driverVersion,cudaCompiler:$cudaCompiler,resourceUsage:$resourceUsage}' >> "$out"
echo "benchmark_case=$case_id benchmark_jsonl=$out"
