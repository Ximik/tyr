#!/usr/bin/env bash
set -euo pipefail
repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
torch_dir=${LIBTORCH_DIR:-"$repo_root/external/wheels/torch"}
test_dir=$(mktemp -d "${TMPDIR:-/tmp}/tyr-dtype-promotion.XXXXXX")
trap 'rm -rf -- "$test_dir"' EXIT

# This standalone native oracle links only the pinned libtorch, independently
# of libTyrC and Lean's dtype list. It never launches a shared project build.
cxx_flags=(-std=c++20 -O2)
config="$torch_dir/share/cmake/Torch/TorchConfig.cmake"
if [[ -f "$config" ]]; then
  abi=$(sed -n 's/.*-D_GLIBCXX_USE_CXX11_ABI=\([01]\).*/\1/p' "$config" | head -n 1)
  if [[ -n "$abi" ]]; then cxx_flags+=("-D_GLIBCXX_USE_CXX11_ABI=$abi"); fi
fi
"${CXX:-c++}" "${cxx_flags[@]}" \
  -I"$torch_dir/include" -I"$torch_dir/include/torch/csrc/api/include" \
  "$repo_root/cc/tools/test_dtype_promotion.cpp" \
  -L"$torch_dir/lib" -Wl,-rpath,"$torch_dir/lib" -ltorch -ltorch_cpu -lc10 \
  -o "$test_dir/native-promotion"
"$test_dir/native-promotion" > "$test_dir/native.tsv"
if [[ -n "${TYR_DTYPE_REPORT:-}" ]]; then cp "$test_dir/native.tsv" "$TYR_DTYPE_REPORT"; fi
cd "$repo_root"
lake env lean --run Tests/DTypePromotionProbe.lean "$test_dir/native.tsv"
