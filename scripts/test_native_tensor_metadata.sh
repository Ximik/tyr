#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
torch_dir=${LIBTORCH_DIR:-"$repo_root/external/wheels/torch"}
lean_dir=${LEAN_HOME:-$(lean --print-prefix)}
native_dir=${TYR_NATIVE_LIB_DIR:-"$repo_root/cc/build"}
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/tyr-native-metadata.XXXXXX")
trap 'rm -rf -- "$test_dir"' EXIT

# Use the existing library so this standalone regression never races a Lake
# build. Run `lake build` first when the native bindings have changed.
if [[ ! -f "$native_dir/libTyrC.so" && ! -f "$native_dir/libTyrC.dylib" ]]; then
  echo "Build libTyrC with lake build before running native tensor metadata checks." >&2
  exit 1
fi

cxx_flags=(-std=c++20 -O2)
config="$torch_dir/share/cmake/Torch/TorchConfig.cmake"
if [[ -f "$config" ]]; then
  abi=$(sed -n 's/.*-D_GLIBCXX_USE_CXX11_ABI=\([01]\).*/\1/p' "$config" | head -n 1)
  if [[ -n "$abi" ]]; then cxx_flags+=("-D_GLIBCXX_USE_CXX11_ABI=$abi"); fi
fi

"${CXX:-c++}" "${cxx_flags[@]}" \
  -I"$repo_root/cc/include" -I"$repo_root/cc/src" -I"$lean_dir/include" \
  -I"$torch_dir/include" -I"$torch_dir/include/torch/csrc/api/include" \
  "$repo_root/cc/tools/test_tensor_metadata.cpp" \
  -L"$lean_dir/lib/lean" -Wl,-rpath,"$lean_dir/lib/lean" -lleanshared \
  -L"$native_dir" -Wl,-rpath,"$native_dir" -lTyrC \
  -L"$torch_dir/lib" -Wl,-rpath,"$torch_dir/lib" -ltorch -ltorch_cpu -lc10 \
  -o "$test_dir/test_tensor_metadata"
"$test_dir/test_tensor_metadata" "$test_dir"
