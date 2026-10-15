#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
torch_dir=${LIBTORCH_DIR:-"$repo_root/external/wheels/torch"}
lean_dir=${LEAN_HOME:-$(lean --print-prefix)}
native_dir=${TYR_NATIVE_LIB_DIR:-"$repo_root/cc/build"}
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/tyr-attention-effects.XXXXXX")
source "$repo_root/scripts/native_tensor_test_common.sh"
trap native_test_cleanup EXIT
native_test_exception_smoke
"${CXX:-c++}" "${native_cxx_flags[@]}" \
  -I"$repo_root/cc/src" -I"$lean_dir/include" \
  -I"$torch_dir/include" -I"$torch_dir/include/torch/csrc/api/include" \
  "$repo_root/cc/tools/test_attention_effects.cpp" \
  "${native_link_flags[@]}" -o "$test_dir/test_attention_effects"
native_test_run "$test_dir/test_attention_effects"
native_test_complete=1
