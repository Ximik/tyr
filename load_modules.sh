#!/bin/bash
# Load the CUDA toolchain module for ThunderKittens development.
# Uses GCCcore/12.3.0 to match locally compiled Lean 4.

if ! type module >/dev/null 2>&1; then
  if [[ -f /etc/profile.d/modules.sh ]]; then
    # Initialize the module function in non-login automation shells.
    source /etc/profile.d/modules.sh >/dev/null 2>&1
  elif [[ -f /usr/share/lmod/lmod/init/bash ]]; then
    source /usr/share/lmod/lmod/init/bash >/dev/null 2>&1
  fi
fi

if ! type module >/dev/null 2>&1; then
  echo "Environment modules unavailable; assuming dependencies are already on PATH." >&2
  return 0 2>/dev/null || exit 0
fi

TYR_CUDA_MODULE="${TYR_CUDA_MODULE:-CUDA/12.9.1}"
if [[ -z "${TYR_NCCL_MODULE+x}" ]]; then
  TYR_NCCL_MODULE=""
fi

module purge
module load EB5
module load EB5Modules
module load EBModules
module load "${TYR_CUDA_MODULE}"
if [[ -n "${TYR_NCCL_MODULE}" ]]; then
  module load "${TYR_NCCL_MODULE}"
fi

# Keep scripted runs non-interactive. Set TYR_VERBOSE_MODULES=1 to print the stack.
if [[ "${TYR_VERBOSE_MODULES:-0}" == "1" ]]; then
  echo "Modules loaded:"
  module -t list
fi

if [[ -n "${EBROOTCUDA:-}" ]]; then
  export CUDA_HOME="${CUDA_HOME:-${EBROOTCUDA}}"
fi
