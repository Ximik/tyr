#!/usr/bin/env bash
# Isolate fatal callback failures from the test process, using real generated Lean C.
set -euo pipefail
ulimit -c 0
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
torch_dir=${LIBTORCH_DIR:-"$repo_root/external/wheels/torch"}
lean_dir=${LEAN_HOME:-$(lean --print-prefix)}
native_dir=${TYR_NATIVE_LIB_DIR:-"$repo_root/cc/build"}
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/tyr-callback-errors.XXXXXX")
source "$repo_root/scripts/native_tensor_test_common.sh"
trap native_test_cleanup EXIT
native_test_exception_smoke

# Link the callback module with the project modules it imports.
modules=()
pending=(Tests.Native.FunctionalGradCallback)
while ((${#pending[@]})); do
  module=${pending[0]}; pending=("${pending[@]:1}")
  [[ " ${modules[*]} " == *" $module "* ]] && continue
  modules+=("$module")
  while read -r _ import; do
    [[ $import == Tyr.* ]] && pending+=("$import")
  done < <(grep '^import ' "$repo_root/${module//.//}.lean")
done
(cd "$repo_root" && lake build "${modules[@]/%/:o.export}")
objects=()
for module in "${modules[@]}"; do objects+=("$repo_root/.lake/build/ir/${module//.//}.c.o.export"); done
"${CXX:-c++}" "${native_cxx_flags[@]}" -I"$repo_root/cc/src" -I"$lean_dir/include" \
  "$repo_root/cc/tools/test_functional_grad_callback.cpp" "${objects[@]}" \
  "${native_link_flags[@]}" \
  -o "$test_dir/callback-test"

python3 - "$test_dir/callback-test" <<'PY'
import signal
import subprocess
import sys

executable = sys.argv[1]
for mode in ("success", "setup-error", "result-error", "callback-error"):
    result = subprocess.run([executable, mode], capture_output=True, text=True, timeout=20)
    if mode == "callback-error":
        assert result.returncode == -signal.SIGABRT, (mode, result.returncode, result.stdout, result.stderr)
        assert "unrecoverable native exception in pure Lean callback" in result.stderr, result.stderr
        assert "shape" in result.stderr, result.stderr
        print("callback-error terminated with the expected diagnostic")
    else:
        assert result.returncode == 0, (mode, result.returncode, result.stdout, result.stderr)
        print(result.stdout, end="")
PY
native_test_complete=1
