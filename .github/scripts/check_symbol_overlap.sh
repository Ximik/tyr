#!/usr/bin/env bash
# Fail when two files loaded into the same process define the same C symbol.
#
# The dynamic loader binds every call to the first definition it finds, so a
# duplicate silently replaces one library's copy with another's.
#
# Not checked:
#   - C++ symbols (`_Z...`): identical template/inline copies in every C++
#     library are normal.
#   - glibc's own libraries, which share some symbols among themselves.
#   - `free_sized`: the Lean runtime's fallback, which just calls glibc's `free`.
#
# Checks every built executable and precompiled module library under .lake/build.
#
# Usage: .github/scripts/check_symbol_overlap.sh   (Linux only, from the repo root)
set -euo pipefail

binaries=()

# Every built executable.
for file in .lake/build/bin/*; do
  if [[ -x "${file}" ]]; then
    binaries+=("${file}")
  fi
done

# Every precompiled module library.
binaries+=(.lake/build/lib/lean/*.so)

cache="$(mktemp -d)"
trap 'rm -rf "${cache}"' EXIT

# Plain C symbols a file defines (cached: the same libraries repeat).
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
for binary in "${binaries[@]}"; do
  libraries="$(ldd "${binary}" | awk '$3 ~ /^\// { print $3 }' \
    | grep -vE '/lib(c|m|dl|rt|pthread)\.so')"

  # "<symbol> <file>" for the binary and each library, then any symbol seen twice.
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

if [[ "${failed}" -eq 0 ]]; then
  echo "No duplicate C symbols in ${#binaries[@]} binaries or the libraries they load."
fi
exit "${failed}"
