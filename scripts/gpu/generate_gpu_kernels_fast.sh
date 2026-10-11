#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

usage() {
  cat <<'EOF'
Usage: scripts/gpu/generate_gpu_kernels_fast.sh [options] [KernelModule ...]

Compile the codegen/kernel Lean modules directly, relink the GPU kernel
generator from Lake's recorded link trace, then execute that generator. This
intentionally avoids `lake exe` and `lake build <KernelModule>`, which can pull
in unrelated extern library work in this workspace.

Options:
  --out-dir <path>   Output directory for generated .cu files (default: cc/src/generated)
  --no-clean         Keep existing generated .cu files in the output directory
  --skip-build       Run the current relinked generator without rebuilding Lean targets
  --kernel-only      Rebuild only requested kernel module(s), reusing existing codegen/generator
  --build-only       Rebuild/relink but do not run the generator
  --help             Show this help

Environment:
  LAKE_NUM_JOBS      Lake parallelism (default: 1)

Examples:
  scripts/gpu/generate_gpu_kernels_fast.sh Tyr.GPU.Kernels.MhaH100
  scripts/gpu/generate_gpu_kernels_fast.sh --no-clean Tyr.GPU.Kernels.MhaH100 Tyr.GPU.Kernels.Rotary
EOF
}

out_dir="cc/src/generated"
clean_args=()
skip_build=0
kernel_only=0
build_only=0
modules=()

while (($#)); do
  case "$1" in
    --out-dir)
      if [[ $# -lt 2 ]]; then
        echo "--out-dir expects a path argument" >&2
        exit 2
      fi
      out_dir="$2"
      shift 2
      ;;
    --no-clean)
      clean_args+=("--no-clean")
      shift
      ;;
    --skip-build)
      skip_build=1
      shift
      ;;
    --kernel-only)
      kernel_only=1
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
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      modules+=("$1")
      shift
      ;;
  esac
done

if [[ ${#modules[@]} -eq 0 ]]; then
  modules=("Tyr.GPU.Kernels.MhaH100")
fi

export LAKE_NUM_JOBS="${LAKE_NUM_JOBS:-1}"

lean_prefix="$(lean --print-prefix)"
lean_include="$lean_prefix/include"
export LEAN_PATH="$repo_root/.lake/packages/LeanTest/.lake/build/lib/lean:$repo_root/.lake/build/lib/lean:$lean_prefix/lib/lean${LEAN_PATH:+:$LEAN_PATH}"

generator="/tmp/tyr_relinked/GenerateGpuKernels"

module_rel() {
  local module="$1"
  printf '%s\n' "${module//./\/}"
}

extract_trace_command() {
  local trace="$1"
  local cmd
  cmd="$(awk '/^ *"\.> / { sub(/^ *"\.> /, ""); print; exit }' "$trace")"
  cmd="${cmd%\",}"
  printf '%s\n' "$cmd"
}

run_trace_command() {
  local trace="$1"
  local output_from="${2:-}"
  local output_to="${3:-}"
  local link_cmd

  link_cmd="$(extract_trace_command "$trace")"
  if [[ -z "$link_cmd" ]]; then
    echo "Could not extract link command from $trace" >&2
    exit 1
  fi
  if [[ -n "$output_from" && -n "$output_to" ]]; then
    link_cmd="${link_cmd//$output_from/$output_to}"
  fi
  bash -lc "$link_cmd"
}

relink_module_dynlib() {
  local module="$1"
  local dylib=".lake/build/lib/lean/tyr_${module//./_}.so"
  local trace="$dylib.trace"
  if [[ -f "$trace" ]]; then
    echo "  relink $(basename "$dylib")"
    run_trace_command "$trace"
  fi
}

compile_module() {
  local module="$1"
  local require_object="${2:-0}"
  local relink_dynlib="${3:-0}"
  local rel src olean ilean c setup obj
  rel="$(module_rel "$module")"
  src="$rel.lean"
  olean=".lake/build/lib/lean/$rel.olean"
  ilean=".lake/build/lib/lean/$rel.ilean"
  c=".lake/build/ir/$rel.c"
  setup=".lake/build/ir/$rel.setup.json"
  obj=".lake/build/ir/$rel.c.o.export"

  if [[ ! -f "$src" ]]; then
    echo "Missing Lean source for module $module: $src" >&2
    exit 1
  fi

  mkdir -p "$(dirname "$olean")" "$(dirname "$ilean")" "$(dirname "$c")"
  if [[ -f "$setup" ]]; then
    echo "  lean+cc $module"
    lean "$src" -o "$olean" -i "$ilean" -c "$c" --setup "$setup" --json >/dev/null
    "$lean_prefix/bin/leanc" -c -o "$obj" "$c" -I "$lean_include" \
      -fstack-clash-protection -fdata-sections -ffunction-sections -fPIC \
      -fvisibility=hidden -Wno-unused-command-line-argument \
      -O3 -DNDEBUG -DLEAN_EXPORTING
    if [[ "$relink_dynlib" == "1" ]]; then
      relink_module_dynlib "$module"
    fi
  else
    if [[ "$require_object" == "1" ]]; then
      echo "Missing Lean setup file for generator module $module: $setup" >&2
      echo "Run a normal Lake build once to materialize setup metadata." >&2
      exit 1
    fi
    echo "  lean $module"
    lean "$src" -o "$olean" -i "$ilean" --json >/dev/null
  fi
}

relink_generator() {
  local trace=".lake/build/bin/GenerateGpuKernels.trace"
  local original="$repo_root/.lake/build/bin/GenerateGpuKernels"

  if [[ ! -f "$trace" ]]; then
    echo "Missing generator link trace: $trace" >&2
    echo "Run a normal Lake build once to materialize the generator trace." >&2
    exit 1
  fi

  mkdir -p "$(dirname "$generator")"
  run_trace_command "$trace" "$original" "$generator"
  chmod +x "$generator"
}

if [[ "$skip_build" -eq 0 ]]; then
  if [[ "$kernel_only" -eq 0 ]]; then
    echo "[1/3] Compiling codegen support modules"
    codegen_modules=(
      Tyr.GPU.Codegen.TileTypes
      Tyr.GPU.Codegen.IR
      Tyr.GPU.Codegen.Monad
      Tyr.GPU.Codegen.GlobalLayout
      Tyr.GPU.Codegen.Primitives
      Tyr.GPU.Codegen.Loop
      Tyr.GPU.Codegen.EmitNew
      Tyr.GPU.Codegen.Attribute
      Tyr.GPU.Codegen.FFI
      Tyr.GPU.GenerateGpuKernels
      Tyr.GPU.Codegen.Macros
      Tyr.GPU.Kernels.Prelude
    )
    for module in "${codegen_modules[@]}"; do
      compile_module "$module" 1 1
    done

    echo "[2/3] Compiling requested kernel module target(s)"
  else
    echo "[1/2] Compiling requested kernel module target(s)"
  fi
  for module in "${modules[@]}"; do
    compile_module "$module" 0 1
  done

  if [[ "$kernel_only" -eq 0 ]]; then
    echo "[3/3] Relinking generator"
    relink_generator
  else
    echo "[2/2] Reusing existing generator"
  fi
else
  echo "[1/2] Skipping Lean rebuild"
fi

if [[ ! -x "$generator" ]]; then
  if [[ -x ".lake/build/bin/GenerateGpuKernels" ]]; then
    generator=".lake/build/bin/GenerateGpuKernels"
  else
    echo "Missing built generator: $generator" >&2
    echo "Run without --skip-build first." >&2
    exit 1
  fi
fi

if [[ ! -x "$generator" ]]; then
  echo "Missing built generator: $generator" >&2
  echo "Run without --skip-build first." >&2
  exit 1
fi

if [[ "$build_only" -eq 1 ]]; then
  echo "Build complete; generator not executed (--build-only)"
  exit 0
fi

echo "[2/2] Generating CUDA into $out_dir"
"$generator" "${modules[@]}" --out-dir "$out_dir" "${clean_args[@]}"
