#!/usr/bin/env bash
# Fetch the pinned native dependencies listed in dependencies.lock into external/:
#   external/python/torch    the torch wheel: libtorch (lib/, include/, share/)
#                            plus its Python package
#   external/python/nvidia   CUDA runtime wheels (only when nvcc is on PATH)
#   external/python/pyarrow  the pyarrow wheel: Arrow/Parquet headers (include/)
#                            and libraries, plus its Python package
#
# Every wheel is unpacked whole, so external/python is a complete site-packages
# directory (with *.dist-info): a Python 3.12 venv can import the same torch and
# pyarrow the C++ build links against by adding it with a .pth file.
#
# Usage: ./fetch_dependencies.sh [--dry-run]
# TYR_DEPS_VARIANT=cpu|cuda overrides the nvcc-based CPU/CUDA choice.
# TYR_DEPS_CACHE=<dir> keeps downloaded wheels outside external/.cache.
set -euo pipefail

root="$(cd "$(dirname "$0")" && pwd)"
lock="${root}/dependencies.lock"
external="${root}/external"
cache="${TYR_DEPS_CACHE:-${external}/.cache}"
stamp="${external}/.deps-stamp"

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

selected="$(awk -v p="${platform}" -v v="${variant}" \
  '$1 == p && ($2 == v || $2 == "any")' "${lock}")"
if [[ -z "${selected}" ]]; then
  echo "dependencies.lock has no entries for ${platform} ${variant}" >&2
  exit 1
fi

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

# Bump layout when the unpacking below changes, so existing checkouts re-unpack.
layout=2
selection_id="$(printf 'layout=%s\n%s\n' "${layout}" "${selected}" | cksum | cut -d' ' -f1)"
if [[ -f "${stamp}" && "$(cat "${stamp}")" == "${selection_id}" ]]; then
  echo "external/ is up to date"
  exit 0
fi

mkdir -p "${cache}"
staging="${external}/.staging"
rm -rf "${staging}"
mkdir -p "${staging}"
trap 'rm -rf "${staging}"' EXIT

while read -r _ _ target sha url; do
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

  case "${target}" in
    torch | nvidia | arrow)
      unzip -q -o "${wheel}" -d "${staging}/python"
      ;;
    *)
      echo "unknown target in dependencies.lock: ${target}" >&2
      exit 1
      ;;
  esac
done <<< "${selected}"

# Swap in the new trees only after every wheel extracted successfully.
# arrow, libtorch and nvidia are the pre-external/python layout; remove them.
for dir in python arrow libtorch nvidia; do
  rm -rf "${external:?}/${dir}"
  if [[ -d "${staging}/${dir}" ]]; then
    mv "${staging}/${dir}" "${external}/${dir}"
  fi
done
echo "${selection_id}" > "${stamp}"
echo "installed into external/: $(cd "${external}" && ls -d python/torch python/nvidia python/pyarrow 2>/dev/null | tr '\n' ' ')"
