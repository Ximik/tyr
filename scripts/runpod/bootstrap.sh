#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/runpod/bootstrap.sh

Remote-only helper. Installs the Lean toolchain into the persistent workspace
cache and verifies the CUDA build prerequisites needed by tyr benchmarks.
EOF
}

if [[ "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

repo_root="$(cd "$(dirname "$0")/../.." && pwd)"
workspace_root="${TYR_RUNPOD_VOLUME_MOUNT_PATH:-/workspace}"
cache_root="${workspace_root%/}/cache"

export ELAN_HOME="${ELAN_HOME:-${cache_root}/elan}"
export PATH="${ELAN_HOME}/bin:${PATH}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-${cache_root}/pip}"

mkdir -p "${cache_root}" "${PIP_CACHE_DIR}"

maybe_install_apt_packages() {
  [[ "${TYR_RUNPOD_REMOTE_BOOTSTRAP_PACKAGES:-1}" == "1" ]] || return 0
  command -v apt-get >/dev/null 2>&1 || return 0
  [[ "$(id -u)" == "0" ]] || return 0

  DEBIAN_FRONTEND=noninteractive apt-get update -y >/dev/null
  DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    curl \
    git \
    rsync \
    pkg-config >/dev/null
}

install_elan() {
  if command -v elan >/dev/null 2>&1; then
    return 0
  fi
  curl -sSf https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh \
    | sh -s -- -y --no-modify-path --default-toolchain none >/dev/null
}

require_path() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[runpod-bootstrap] missing required command: $1" >&2
    exit 1
  }
}

maybe_install_apt_packages
install_elan

require_path python3
require_path git
require_path curl
require_path make
require_path c++
require_path nvcc
require_path elan

cd "${repo_root}"
toolchain="$(< lean-toolchain)"
elan toolchain install "${toolchain}" >/dev/null

if [[ ! -d "${repo_root}/external/python/torch/lib" ]]; then
  echo "[runpod-bootstrap] external/python/torch is missing under ${repo_root}. Run sync_repo.sh first." >&2
  exit 1
fi

if [[ ! -d "${repo_root}/external/soxr/src" ]]; then
  echo "[runpod-bootstrap] external/soxr is missing under ${repo_root}. Run sync_repo.sh first." >&2
  exit 1
fi

echo "[runpod-bootstrap] ready"
