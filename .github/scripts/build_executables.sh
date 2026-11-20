#!/usr/bin/env bash
# Check that the Examples library and every `lean_exe` in lakefile.lean build
# with --wfail. Only builds; nothing is run. Off macOS, the demos that use the
# macOS audio input are not built: they do not link elsewhere.
#
# Usage: .github/scripts/build_executables.sh   (from the repo root)
set -euo pipefail

macos_only=(Qwen3ASRLiveMic Qwen3ASRLiveMicTrueStream WhisperVoiceMode)

targets=(Examples)
while read -r name; do
  if [[ "$(uname -s)" != Darwin && " ${macos_only[*]} " == *" ${name} "* ]]; then
    continue
  fi
  targets+=("${name}")
done < <(sed -n 's/^lean_exe \([A-Za-z0-9_]*\) where$/\1/p' lakefile.lean)

lake build --wfail "${targets[@]}"
