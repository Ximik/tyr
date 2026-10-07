#!/usr/bin/env bash
# Validate one commit subject: check-commit-message.sh "<subject>"
set -euo pipefail

subject="${1:?usage: check-commit-message.sh \"<subject>\"}"

# Allow git-generated merge/revert commits.
if [[ "${subject}" =~ ^(Merge|Revert)[[:space:]] ]]; then
  exit 0
fi

pattern='^(feat|fix|chore|ci|build|docs|refactor|test|perf)\([a-z0-9][a-z0-9._/-]*\): .+'
if [[ "${subject}" =~ ${pattern} ]]; then
  exit 0
fi

cat >&2 <<EOF
Invalid commit subject: ${subject}
Expected format:
  type(scope): summary
Allowed types:
  feat, fix, chore, ci, build, docs, refactor, test, perf
Examples:
  feat(qwen35): add multimodal streaming decode path
  fix(torch-ffi): guard null tensor in scatter op
EOF
exit 1
