#!/usr/bin/env bash
# Fetch every pinned dependency into external/:
#   deps/git.lock     git repos   -> external/git,    via deps/fetch_git.sh
#   deps/wheels.lock  wheels      -> external/wheels, via deps/fetch_wheels.sh
#
# Usage: deps/fetch.sh
# See deps/fetch_wheels.sh for TYR_DEPS_VARIANT and TYR_DEPS_CACHE.
set -euo pipefail

deps="$(cd "$(dirname "$0")" && pwd)"
source "${deps}/../env.sh"
"${deps}/fetch_git.sh"
"${deps}/fetch_wheels.sh"
