#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
torch_dir=${LIBTORCH_DIR:-"$repo_root/external/wheels/torch"}
lean_dir=${LEAN_HOME:-$(lean --print-prefix)}
native_dir=${TYR_NATIVE_LIB_DIR:-"$repo_root/cc/build"}
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/tyr-native-ownership.XXXXXX")
source "$repo_root/scripts/native_tensor_test_common.sh"
trap native_test_cleanup EXIT

# Link the existing library; this regression never races a Lake native build.
if [[ ! -f "$native_dir/libTyrC.so" && ! -f "$native_dir/libTyrC.dylib" ]]; then
  echo "Build libTyrC with lake build before running native tensor ownership checks." >&2
  exit 1
fi

native_test_exception_smoke

"${CXX:-c++}" "${native_cxx_flags[@]}" \
  -I"$repo_root/cc/src" -I"$lean_dir/include" \
  "$repo_root/cc/tools/test_tensor_ownership.cpp" \
  "${native_link_flags[@]}" \
  -o "$test_dir/test_tensor_ownership"
native_test_run "$test_dir/test_tensor_ownership"
native_test_complete=1
