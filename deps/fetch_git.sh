#!/usr/bin/env bash
# Check out the git dependencies pinned in deps/git.lock into external/git/<name>.
#
# Each repo is fetched as a single commit (`git fetch --depth 1 <url> <commit>`),
# so no history is downloaded, and git verifies the content against the commit
# hash. A repo already at its pinned commit is left alone.
#
# Usage: deps/fetch_git.sh
set -euo pipefail

deps="$(cd "$(dirname "$0")" && pwd)"
git_dir="$(dirname "${deps}")/external/git"
mkdir -p "${git_dir}"

# Fetch into <name>.tmp so an interrupted or failed run never leaves a
# half-populated checkout behind; remove the temporary directory on failure.
tmp=""
trap 'if [[ -n "${tmp}" ]]; then rm -rf "${tmp}"; fi' EXIT

while read -r name commit url; do
  if [[ -z "${name}" || "${name}" == \#* ]]; then
    continue
  fi
  dest="${git_dir}/${name}"
  if [[ -d "${dest}/.git" && "$(git -C "${dest}" rev-parse HEAD 2>/dev/null)" == "${commit}" ]]; then
    echo "${name} is at ${commit:0:12}"
    continue
  fi

  echo "fetching ${name} at ${commit:0:12}"
  tmp="${dest}.tmp"
  rm -rf "${tmp}"
  git -c init.defaultBranch=main init -q "${tmp}"
  git -C "${tmp}" fetch -q --depth 1 "${url}" "${commit}"
  git -C "${tmp}" -c advice.detachedHead=false checkout -q FETCH_HEAD
  if [[ "$(git -C "${tmp}" rev-parse HEAD)" != "${commit}" ]]; then
    echo "${name}: fetched $(git -C "${tmp}" rev-parse HEAD), expected ${commit}" >&2
    exit 1
  fi
  rm -rf "${dest}"
  mv "${tmp}" "${dest}"
  tmp=""
done < "${deps}/git.lock"
