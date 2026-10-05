#!/usr/bin/env bash
# Source before building: `source ./env.sh` (also by absolute path from any directory).
# Native libraries resolve via rpaths into external/; see deps/fetch.sh.
if [[ "$(uname -s)" != Darwin ]]; then
  # Link with the system gcc/glibc; see scripts/lean_cc_wrapper.sh for why.
  # ${BASH_SOURCE[0]} in bash, $0 when sourced from zsh.
  export LEAN_CC="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/scripts/lean_cc_wrapper.sh"
fi
