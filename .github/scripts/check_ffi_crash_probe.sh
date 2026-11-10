#!/usr/bin/env bash
# Check Tyr's fatal failure paths with `ffi_crash_probe`:
#   - libtorch: an uncaught libtorch error at the Lean FFI boundary aborts
#     (SIGABRT, exit code 134) with Tyr's libtorch error report;
#   - admission: `Tensor.ofRaw!` given a handle that contradicts its spec
#     aborts with the native tensor contract report;
#   - admission-ok: a matching claim is admitted and the probe exits 0.
#
# Usage: .github/scripts/check_ffi_crash_probe.sh   (from the repo root)
set -uo pipefail

mkdir -p output/ci
status=0

expect() {
  local mode=$1 code=$2 message=$3
  local log="output/ci/ffi-crash-probe-${mode}.log"
  ./.lake/build/bin/ffi_crash_probe "${mode}" > "${log}" 2>&1
  local actual=$?
  head -n 3 "${log}"
  if [[ "${actual}" -ne "${code}" ]]; then
    echo "::error::${mode}: expected exit code ${code}, got ${actual}"
    status=1
  elif ! grep -qF "${message}" "${log}"; then
    echo "::error::${mode}: output lacks '${message}'"
    status=1
  fi
}

expect libtorch 134 "[tyr] fatal: uncaught libtorch error"
expect admission 134 "[tyr] native tensor contract violation: Tests.FfiCrashProbe.wrongShapeBinding: Expected shape #[3], got #[2, 2]"
expect admission-ok 0 "admitted: #[1.000000, 1.000000, 1.000000, 1.000000]"
exit "${status}"
