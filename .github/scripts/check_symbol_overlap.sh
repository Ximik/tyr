#!/usr/bin/env bash
# Check that no C symbol is defined twice in a process and that libTyrC exports
# only `lean_*` and `tyr_ops::*`. Fails if:
#
# 1. For an executable in .lake/build/bin or a module library in
#    .lake/build/lib/lean, a C symbol (name not starting with `_Z`) is defined
#    by two of: the binary itself and the libraries it loads (`ldd`).
#    Not compared: glibc's libc, libm, libdl, librt and libpthread, and the
#    symbol `free_sized`.
#
# 2. cc/build/libTyrC.so exports a symbol other than `lean_*` or `tyr_ops::*`.
#
# Usage: .github/scripts/check_symbol_overlap.sh   (Linux only, from the repo root)
set -euo pipefail

binaries=()
for file in .lake/build/bin/*; do
  if [[ -x "${file}" ]]; then
    binaries+=("${file}")
  fi
done
binaries+=(.lake/build/lib/lean/*.so)

cache="$(mktemp -d)"
trap 'rm -rf "${cache}"' EXIT

# C symbols a file defines, cached because the same libraries repeat.
symbols() {
  local file key
  file="$(readlink -f "$1")"
  key="${cache}/$(echo "${file}" | md5sum | cut -d' ' -f1)"
  if [[ ! -f "${key}" ]]; then
    nm -D --defined-only "${file}" | awk '
      { sub(/@.*/, "", $3) }
      $3 !~ /^_Z/ && $3 !~ /^(__bss_start|_edata|_end|_init|_fini|free_sized)$/ { print $3 }
    ' | sort -u > "${key}"
  fi
  cat "${key}"
}

failed=0

# Check 1
for binary in "${binaries[@]}"; do
  libraries="$(ldd "${binary}" | awk '$3 ~ /^\// { print $3 }' \
    | grep -vE '/lib(c|m|dl|rt|pthread)\.so')"
  duplicates="$(
    for file in "${binary}" ${libraries}; do
      symbols "${file}" | awk -v file="$(basename "${file}")" '{ print $1, file }'
    done | awk '{ files[$1] = files[$1] " " $2; count[$1]++ }
                END { for (s in count) if (count[s] > 1) print "  " s ":" files[s] }'
  )"
  if [[ -n "${duplicates}" ]]; then
    echo "error: ${binary}: symbols defined more than once:"
    echo "${duplicates}"
    failed=1
  fi
done

# Check 2
unexpected="$(nm -D --defined-only --demangle --format=just-symbols cc/build/libTyrC.so \
  | grep -vE '^(lean_|tyr_ops::)' || true)"
if [[ -n "${unexpected}" ]]; then
  echo "error: cc/build/libTyrC.so exports symbols besides lean_* and tyr_ops:: (see cc/map/libTyrC.map):"
  echo "${unexpected}" | sed 's/^/  /'
  failed=1
fi

if [[ "${failed}" -eq 0 ]]; then
  echo "No duplicate C symbols in ${#binaries[@]} binaries or the libraries they load;"
  echo "libTyrC exports only lean_* and tyr_ops::."
fi
exit "${failed}"
