#!/usr/bin/env bash
# Shared support for standalone tests that load both Lean and libtorch.
# The caller supplies repo_root, torch_dir, lean_dir, native_dir and test_dir.

native_cxx_flags=(-std=c++20 -O2)
config="$torch_dir/share/cmake/Torch/TorchConfig.cmake"
if [[ -f "$config" ]]; then
  abi=$(sed -n 's/.*-D_GLIBCXX_USE_CXX11_ABI=\([01]\).*/\1/p' "$config" | head -n 1)
  if [[ -n "$abi" ]]; then native_cxx_flags+=("-D_GLIBCXX_USE_CXX11_ABI=$abi"); fi
fi

# Linux Lean ships its own C++ exception runtime. These harnesses are built by
# the system C++ compiler, like libtorch and libTyrC. Keep system libstdc++
# and its unwinder ahead of libleanshared in DT_NEEDED so exception symbols
# resolve consistently.
# --no-as-needed is scoped: do not depend on driver defaults or drop this entry
# just because the source's references could also resolve from libleanshared.
# macOS uses a shared system libc++ runtime and retains its existing link order.
native_link_flags=(-L"$lean_dir/lib/lean" -Wl,-rpath,"$lean_dir/lib/lean")
if [[ $(uname -s) == Linux ]]; then
  native_link_flags+=(-Wl,--push-state,--no-as-needed -lstdc++ -lgcc_s -Wl,--pop-state -ldl)
fi
native_link_flags+=(-lleanshared
  -L"$native_dir" -Wl,-rpath,"$native_dir" -lTyrC
  -L"$torch_dir/lib" -Wl,-rpath,"$torch_dir/lib" -ltorch -ltorch_cpu -lc10)

native_test_complete=0
native_test_cleanup() {
  local status=$?
  rm -rf -- "$test_dir"
  # Bash 3.2 can report status zero to EXIT after a nounset expansion failure.
  if [[ $native_test_complete != 1 && $status == 0 ]]; then status=1; fi
  exit "$status"
}

native_test_run() {
  local status diagnostics
  if "$@"; then return 0; else status=$?; fi
  diagnostics="$repo_root/output/ci/native-$(basename -- "$1")-failure.log"
  mkdir -p -- "$(dirname -- "$diagnostics")"
  {
    echo "Native test exited with status $status: $*"
    if [[ $(uname -s) == Linux ]]; then
      ldd "$1" || true
      if command -v readelf >/dev/null 2>&1; then readelf -d "$1" || true; fi
      if command -v gdb >/dev/null 2>&1; then
        timeout 60 gdb --batch -ex 'set pagination off' -ex 'run' \
          -ex 'thread apply all bt 30' --args "$@" || true
      else
        echo 'gdb unavailable; no native backtrace captured.'
      fi
    fi
  } >"$diagnostics" 2>&1
  cat "$diagnostics" >&2
  return "$status"
}

native_test_exception_smoke() {
  "${CXX:-c++}" "${native_cxx_flags[@]}" \
    -I"$lean_dir/include" -I"$torch_dir/include" \
    -I"$torch_dir/include/torch/csrc/api/include" \
    "$repo_root/cc/tools/test_native_exception_runtime.cpp" \
    "${native_link_flags[@]}" -o "$test_dir/exception-runtime"
  native_test_run "$test_dir/exception-runtime"
}
