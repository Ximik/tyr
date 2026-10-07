#!/usr/bin/env bash
# Download the wheels in deps/wheels.lock for this platform and variant, check
# their sha256, and unzip them into external/wheels.
#
# Lock lines are `<platform> <variant> <sha256> <url>`. A line applies when its
# platform matches this machine and its variant is `any` or the selected one:
# `cuda` when nvcc is on PATH (Linux), otherwise `cpu`.
#
# external/wheels/.stamp records which lock lines (and which version of this
# script) produced external/wheels; when it matches, the script does nothing.
#
# Usage: deps/fetch_wheels.sh [--dry-run]
# TYR_DEPS_VARIANT=cpu|cuda overrides the nvcc-based CPU/CUDA choice.
# TYR_DEPS_CACHE=<dir> keeps downloaded wheels in <dir> instead of
# external/.cache, e.g. on a CI runner whose checkout wipes the workspace.
set -euo pipefail

deps="$(cd "$(dirname "$0")" && pwd)"
root="$(dirname "${deps}")"
lock="${deps}/wheels.lock"
external="${root}/external"
cache="${TYR_DEPS_CACHE:-${external}/.cache}"
stamp="${external}/wheels/.stamp"

dry_run=0
case "${1:-}" in
  --dry-run) dry_run=1 ;;
  "") ;;
  *) echo "usage: $0 [--dry-run]" >&2; exit 2 ;;
esac

case "$(uname -s)-$(uname -m)" in
  Linux-x86_64) platform=linux-x86_64 ;;
  Linux-aarch64) platform=linux-aarch64 ;;
  Darwin-arm64) platform=macos-arm64 ;;
  *) echo "unsupported platform: $(uname -s) $(uname -m)" >&2; exit 1 ;;
esac

variant="${TYR_DEPS_VARIANT:-}"
if [[ -z "${variant}" ]]; then
  if [[ "${platform}" == linux-* ]] && command -v "${NVCC:-nvcc}" >/dev/null 2>&1; then
    variant=cuda
  else
    variant=cpu
  fi
fi
if [[ "${variant}" != cpu && "${variant}" != cuda ]]; then
  echo "TYR_DEPS_VARIANT must be cpu or cuda, got: ${variant}" >&2
  exit 1
fi

# `any` lines alone are not enough: the selected variant must exist for this
# platform (there is no cuda variant on macOS, for example).
if ! awk -v p="${platform}" -v v="${variant}" '$1 == p && $2 == v { found = 1 } END { exit !found }' "${lock}"; then
  echo "deps/wheels.lock has no ${variant} entries for ${platform}" >&2
  exit 1
fi
selected="$(awk -v p="${platform}" -v v="${variant}" \
  '$1 == p && ($2 == v || $2 == "any")' "${lock}")"

if command -v sha256sum >/dev/null 2>&1; then
  sha256() { sha256sum "$1" | cut -d' ' -f1; }
else
  sha256() { shasum -a 256 "$1" | cut -d' ' -f1; }
fi

echo "platform=${platform} variant=${variant}"
if [[ "${dry_run}" == 1 ]]; then
  echo "${selected}"
  exit 0
fi

# Hashing this script too means a change to how wheels are unpacked re-unpacks
# existing checkouts even when the lock is unchanged.
selection_id="$( { cat "${BASH_SOURCE[0]}"; printf '%s\n' "${selected}"; } | cksum | cut -d' ' -f1)"
if [[ -f "${stamp}" && "$(cat "${stamp}")" == "${selection_id}" ]]; then
  echo "external/wheels is up to date"
  exit 0
fi

mkdir -p "${cache}"
staging="${external}/.staging"
rm -rf "${staging}"
mkdir -p "${staging}"
trap 'rm -rf "${staging}"' EXIT

while read -r _ _ sha url; do
  wheel="${cache}/${sha}.whl"
  if [[ ! -f "${wheel}" ]]; then
    echo "downloading ${url##*/}"
    curl --fail --location --retry 5 --retry-all-errors --show-error --silent \
      -o "${wheel}.part" "${url}"
    mv "${wheel}.part" "${wheel}"
  fi
  if [[ "$(sha256 "${wheel}")" != "${sha}" ]]; then
    rm -f "${wheel}"
    echo "checksum mismatch for ${url##*/}; deleted the cached file" >&2
    exit 1
  fi
  unzip -q -o "${wheel}" -d "${staging}/wheels"
done <<< "${selected}"

# Replace external/wheels only after every wheel extracted successfully.
rm -rf "${external:?}/wheels"
mv "${staging}/wheels" "${external}/wheels"
echo "${selection_id}" > "${stamp}"
echo "installed $(printf '%s\n' "${selected}" | wc -l | tr -d ' ') wheels into external/wheels"
