#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
torch_dir=${LIBTORCH_DIR:-"$repo_root/external/wheels/torch"}
lean_dir=${LEAN_HOME:-$(lean --print-prefix)}
native_dir=${TYR_NATIVE_LIB_DIR:-"$repo_root/cc/build"}
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/tyr-native-distributed.XXXXXX")
source "$repo_root/scripts/native_tensor_test_common.sh"
trap native_test_cleanup EXIT
if [[ ${TYR_DISTRIBUTED_BASELINE:-0} == 1 ]]; then
  native_cxx_flags+=(-DTYR_DISTRIBUTED_BASELINE)
fi
native_test_exception_smoke
"${CXX:-c++}" "${native_cxx_flags[@]}" \
  -I"$repo_root/cc/src" -I"$lean_dir/include" \
  "$repo_root/cc/tools/test_distributed_recovery.cpp" \
  "${native_link_flags[@]}" -o "$test_dir/test_distributed_recovery"
native_test_run "$test_dir/test_distributed_recovery"
if [[ ${TYR_DISTRIBUTED_BASELINE:-0} != 1 ]]; then
  "${CXX:-c++}" "${native_cxx_flags[@]}" -DTYR_DISTRIBUTED_UNIT \
    -I"$repo_root/cc/include" -I"$repo_root/cc/src" -I"$lean_dir/include" -I"$torch_dir/include" \
    -I"$torch_dir/include/torch/csrc/api/include" -I"${GLOO_DIR:-$repo_root/external/git/gloo}" \
    "$repo_root/cc/tools/test_distributed_recovery.cpp" \
    "${native_link_flags[@]}" -o "$test_dir/test_distributed_state"
  native_test_run "$test_dir/test_distributed_state"
fi
native_test_complete=1
