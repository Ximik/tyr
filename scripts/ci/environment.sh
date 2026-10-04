#!/usr/bin/env bash
# Source from the repository root before builds, key generation and test runs.
# Native libraries resolve via rpaths into external/; see deps/fetch.sh.
if [[ "$(uname -s)" != Darwin ]]; then
  # Link with the system gcc/glibc; see scripts/lean_cc_wrapper.sh for why.
  export LEAN_CC="$PWD/scripts/lean_cc_wrapper.sh"
fi
