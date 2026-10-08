#!/usr/bin/env bash
# Source before building: `source ./env.sh` (also by absolute path from any directory).

if [[ "$(uname -s)" == Darwin ]]; then
  if [[ -z "${SDKROOT:-}" ]]; then
    export SDKROOT="$(xcrun --sdk macosx --show-sdk-path)"
  fi
else
  export LEAN_CC="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)/scripts/lean_cc_wrapper.sh"
  if [[ -z "${CUDA_HOME+x}" ]] && nvcc_path="$(readlink -f "$(command -v nvcc)")"; then
    export CUDA_HOME="${nvcc_path%/*/*}"
  fi
  if [[ -n "${CUDA_HOME:-}" && ! -x "${CUDA_HOME}/bin/nvcc" ]]; then
    echo "env.sh: CUDA_HOME=${CUDA_HOME} has no bin/nvcc" >&2
    return 1
  fi
fi

if [[ -z "${TYR_MAKE_JOBS:-}" ]]; then
  export TYR_MAKE_JOBS="$(getconf _NPROCESSORS_ONLN)"
fi
