#!/usr/bin/env bash
# Warn when the system loader would give a library a different libstdc++ than
# the compiler built it against.
#
# This happens with a compiler newer than the system's, e.g. a gcc loaded as a
# module on a cluster: the library then needs that compiler's libstdc++, which
# the loader only finds while the module is loaded (it sets LD_LIBRARY_PATH).
# Without it, the program fails at start with "GLIBCXX_... not found".
#
# Usage: check_libstdcxx.sh <compiler> <shared library>   (Linux only)
set -euo pipefail

compiler="$1"
library="$2"

if [[ "$(uname -s)" != Linux ]]; then
  exit 0
fi

# The libstdc++ the compiler links against. A compiler that does not know
# prints the bare name back instead of a path.
built_with="$("${compiler}" -print-file-name=libstdc++.so.6)"
if [[ "${built_with}" != /* ]]; then
  exit 0
fi
built_with="$(realpath "${built_with}")"

# The libstdc++ the loader picks by default (without LD_LIBRARY_PATH).
loaded="$(env -u LD_LIBRARY_PATH ldd "${library}" | awk '/libstdc\+\+/ { print $3 }')"
loaded="$(realpath -q "${loaded}" || true)"

if [[ "${loaded}" != "${built_with}" ]]; then
  echo "warning: ${library} is built against ${built_with}," >&2
  echo "  but the system loader would load ${loaded:-no libstdc++ at all}." >&2
  echo "  Keep the module of ${compiler} loaded when running Tyr." >&2
fi
