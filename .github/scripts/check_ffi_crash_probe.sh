#!/usr/bin/env bash
# Check that an uncaught libtorch error at the Lean FFI boundary aborts with a
# readable message instead of a bare crash.
#
# `ffi_crash_probe` triggers such an error on purpose. It must:
#   - abort with SIGABRT (exit code 134), and
#   - print the libtorch error through Tyr's terminate handler (cc/src/tyr.cpp).
#
# Usage: .github/scripts/check_ffi_crash_probe.sh   (Linux only, from the repo root)
set -uo pipefail

log=output/ci/ffi-crash-probe.log
mkdir -p "$(dirname "${log}")"

./.lake/build/bin/ffi_crash_probe > "${log}" 2>&1
code=$?

head -n 3 "${log}"

if [[ "${code}" -ne 134 ]]; then
  echo "::error::expected SIGABRT (134), got exit code ${code}"
  exit 1
fi

if ! grep -qF "[tyr] fatal: uncaught libtorch error" "${log}"; then
  echo "::error::probe aborted without Tyr's libtorch error report"
  exit 1
fi
