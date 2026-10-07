#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
torch_dir=${LIBTORCH_DIR:-"$repo_root/external/wheels/torch"}
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/tyr-native-attention.XXXXXX")
trap 'rm -rf -- "$test_dir"' EXIT

# Match Linux distributions of LibTorch that use the pre-C++11 string ABI.
cxx_flags=(-std=c++20 -O2)
config="$torch_dir/share/cmake/Torch/TorchConfig.cmake"
if [[ -f "$config" ]]; then
  abi=$(sed -n 's/.*-D_GLIBCXX_USE_CXX11_ABI=\([01]\).*/\1/p' "$config" | head -n 1)
  if [[ -n "$abi" ]]; then cxx_flags+=("-D_GLIBCXX_USE_CXX11_ABI=$abi"); fi
fi

"${CXX:-c++}" "${cxx_flags[@]}" \
  -I"$repo_root/cc/include" -I"$torch_dir/include" \
  -I"$torch_dir/include/torch/csrc/api/include" \
  "$repo_root/cc/tools/test_attention.cpp" \
  -L"$torch_dir/lib" -Wl,-rpath,"$torch_dir/lib" -ltorch -ltorch_cpu -lc10 \
  -o "$test_dir/test_attention"
"$test_dir/test_attention" "${1:-cpu}"
