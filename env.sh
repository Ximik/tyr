#!/usr/bin/env bash
# Source before building: `source ./env.sh` (also by absolute path from any directory).

if [[ "$(uname -s)" != Darwin ]]; then
  export LEAN_CC="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/scripts/lean_cc_wrapper.sh"
elif [[ -z "${SDKROOT:-}" ]]; then
  export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
fi

if [[ -z "${TYR_MAKE_JOBS:-}" ]]; then
  export TYR_MAKE_JOBS="$(getconf _NPROCESSORS_ONLN)"
fi
