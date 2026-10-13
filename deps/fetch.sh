#!/usr/bin/env bash
# Fetch every pinned dependency into external/:
#   deps/git.lock     git repos   -> external/git,    via deps/fetch_git.sh
#   deps/wheels.lock  wheels      -> external/wheels, via deps/fetch_wheels.sh
#
# Usage: deps/fetch.sh cpu|cuda
# The variant must match the Lake configuration: plain `lake -R` for cpu,
# `lake -R -Kcuda=<toolkit>` for cuda. The build checks that they agree.
# See deps/fetch_wheels.sh for TYR_DEPS_CACHE.
set -euo pipefail

deps="$(cd "$(dirname "$0")" && pwd)"
"${deps}/fetch_wheels.sh" "$@"
"${deps}/fetch_git.sh"
