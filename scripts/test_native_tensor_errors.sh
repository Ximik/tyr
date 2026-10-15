#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
torch_dir=${LIBTORCH_DIR:-"$repo_root/external/wheels/torch"}
lean_dir=${LEAN_HOME:-$(lean --print-prefix)}
native_dir=${TYR_NATIVE_LIB_DIR:-"$repo_root/cc/build"}
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/tyr-native-errors.XXXXXX")
test_complete=0
# Bash 3.2 can enter EXIT with status zero after a nounset expansion failure.
# Require the final success marker as well as the command status.
trap 'test_exit=$?; rm -rf -- "$test_dir"; if [[ $test_complete != 1 && $test_exit == 0 ]]; then test_exit=1; fi; exit "$test_exit"' EXIT
test_flags=(-O2)
if [[ ${TYR_CHECKED_BASELINE:-0} == 1 ]]; then
  test_flags+=(-DTYR_CHECKED_BASELINE)
fi

# Link the existing native library. C++ catches escaped native exceptions so
# regressions are reported without terminating a shared Lean test process.
"${CXX:-c++}" -std=c++20 "${test_flags[@]}" \
  -I"$repo_root/cc/src" -I"$lean_dir/include" \
  -I"$torch_dir/include" -I"$torch_dir/include/torch/csrc/api/include" \
  "$repo_root/cc/tools/test_tensor_errors.cpp" \
  -L"$lean_dir/lib/lean" -Wl,-rpath,"$lean_dir/lib/lean" -lleanshared \
  -L"$native_dir" -Wl,-rpath,"$native_dir" -lTyrC \
  -L"$torch_dir/lib" -Wl,-rpath,"$torch_dir/lib" -ltorch -ltorch_cpu -lc10 \
  -o "$test_dir/test_tensor_errors"
"$test_dir/test_tensor_errors"
test_complete=1
