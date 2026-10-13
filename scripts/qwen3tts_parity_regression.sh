#!/usr/bin/env bash
set -euo pipefail

SKIP_EXIT_CODE=2
if [[ "${TYR_QUALIFICATION_STRICT:-0}" == 1 ]]; then SKIP_EXIT_CODE=1; fi
CHECK_ONLY=false

usage() {
  cat <<'USAGE'
Usage: scripts/qwen3tts_parity_regression.sh [--check-prereqs]

  --check-prereqs  Validate prerequisites and exit.
                   Exit codes: 0=ready, 2=skip.
USAGE
}

if [[ "${1:-}" == "--check-prereqs" ]]; then
  CHECK_ONLY=true
  shift
fi
if [[ "$#" -ne 0 ]]; then
  usage
  exit 64
fi

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_ENV=(uv run)
PYTHON_BIN=python
BRIDGE_PYTHON=uv
if [[ -n "${TYR_QUALIFICATION_PYTHON:-}" ]]; then
  RUN_ENV=(env)
  PYTHON_BIN="$TYR_QUALIFICATION_PYTHON"
  BRIDGE_PYTHON="$TYR_QUALIFICATION_PYTHON"
fi

MODEL_DIR="${QWEN3_TTS_MODEL_DIR:-weights/qwen3-tts-0.6b-base}"
if [[ -n "${QWEN3_TTS_PARITY_AUDIO:-}" ]]; then
  AUDIO_PATH="${QWEN3_TTS_PARITY_AUDIO}"
elif [[ -f output/mlk_10s.wav ]]; then
  AUDIO_PATH="output/mlk_10s.wav"
else
  AUDIO_PATH="MLKDream.wav"
fi
QWEN_REPO="${QWEN3_TTS_REPO:-../Qwen3-TTS}"
OUT_DIR="${QWEN3_TTS_PARITY_OUT_DIR:-output/parity_regression}"
LEAN_CODES="$OUT_DIR/lean.codes"
PY_CODES="$OUT_DIR/python.codes"
LEAN_LOG="$OUT_DIR/lean-encode.log"
PY_LOG="$OUT_DIR/python-encode.log"
TYR_DEVICE="${TYR_DEVICE:-mps}"
DEVICE_MAP="${QWEN3_TTS_DEVICE_MAP:-mps}"

check_prereqs() {
  local missing=0

  if [[ ! -d "$MODEL_DIR" ]]; then
    echo "[qwen3tts-parity] SKIP: model dir not found: $MODEL_DIR"
    missing=1
  fi
  if [[ ! -f "$AUDIO_PATH" ]]; then
    echo "[qwen3tts-parity] SKIP: audio file not found: $AUDIO_PATH"
    missing=1
  fi
  if [[ ! -d "$QWEN_REPO" ]]; then
    echo "[qwen3tts-parity] SKIP: qwen repo not found: $QWEN_REPO"
    missing=1
  fi
  if [[ ! -f scripts/qwen3tts_encode_audio.py ]]; then
    echo "[qwen3tts-parity] SKIP: missing scripts/qwen3tts_encode_audio.py"
    missing=1
  fi

  if [[ "$missing" -ne 0 ]]; then
    return 1
  fi
  return 0
}

if ! check_prereqs; then
  exit "$SKIP_EXIT_CODE"
fi

if [[ "$CHECK_ONLY" == "true" ]]; then
  echo "[qwen3tts-parity] prerequisite check: ready"
  exit 0
fi

mkdir -p "$OUT_DIR"
# Only codec files produced by this invocation may satisfy the comparison.
rm -f "$LEAN_CODES" "$PY_CODES"

echo "[qwen3tts-parity] building Lean executable"
if [[ "${TYR_SKIP_QUALIFICATION_BUILD:-0}" != 1 ]]; then
  "${RUN_ENV[@]}" lake build Qwen3TTSEndToEnd >/dev/null
fi

echo "[qwen3tts-parity] Lean encode"
TYR_DEVICE="$TYR_DEVICE" "${RUN_ENV[@]}" lake env ./.lake/build/bin/Qwen3TTSEndToEnd \
  --model-dir "$MODEL_DIR" \
  --seed "${TYR_QUALIFICATION_SEED:-0}" \
  --encode-audio-path "$AUDIO_PATH" \
  --encode-out-codes-path "$LEAN_CODES" \
  --python "$BRIDGE_PYTHON" \
  --qwen-repo "$QWEN_REPO" \
  --encode-only 2>&1 | tee "$LEAN_LOG"

if [[ "${TYR_QUALIFICATION_STRICT:-0}" == 1 ]] && \
    ! grep -Eq '^Target device: torch\.Device\.CUDA [0-9]+$' "$LEAN_LOG"; then
  echo "[qwen3tts-parity] FAIL: missing Lean CUDA execution evidence in $LEAN_LOG"
  exit 1
fi

echo "[qwen3tts-parity] Python reference encode"
"${RUN_ENV[@]}" "$PYTHON_BIN" scripts/qwen3tts_encode_audio.py \
  --speech-tokenizer-dir "$MODEL_DIR/speech_tokenizer" \
  --audio "$AUDIO_PATH" \
  --output-codes "$PY_CODES" \
  --device-map "$DEVICE_MAP" \
  --qwen3-tts-repo "$QWEN_REPO" 2>&1 | tee "$PY_LOG"

PREFIX_ROWS="${QWEN3_TTS_PARITY_PREFIX_ROWS:-125}"
PREFIX_TOKEN_MIN="${QWEN3_TTS_PARITY_PREFIX_TOKEN_MIN:-0.99}"
PREFIX_ROW_MIN="${QWEN3_TTS_PARITY_PREFIX_ROW_MIN:-0.99}"
FULL_TOKEN_MIN="${QWEN3_TTS_PARITY_FULL_TOKEN_MIN:-0.10}"
NONZERO_MIN="${QWEN3_TTS_PARITY_NONZERO_MIN:-0.90}"

"${RUN_ENV[@]}" "$PYTHON_BIN" - "$LEAN_CODES" "$PY_CODES" \
  "$PREFIX_ROWS" "$PREFIX_TOKEN_MIN" "$PREFIX_ROW_MIN" "$FULL_TOKEN_MIN" "$NONZERO_MIN" <<'PY'
import sys
from pathlib import Path
import numpy as np

lean_path, py_path, prefix_rows_s, p_tok_s, p_row_s, full_tok_s, nonzero_s = sys.argv[1:]
prefix_rows = int(prefix_rows_s)
prefix_tok_min = float(p_tok_s)
prefix_row_min = float(p_row_s)
full_tok_min = float(full_tok_s)
nonzero_min = float(nonzero_s)
if prefix_rows <= 0 or not all(np.isfinite(x) and 0.0 <= x <= 1.0
                              for x in (prefix_tok_min, prefix_row_min, full_tok_min, nonzero_min)):
    sys.exit("[qwen3tts-parity] FAIL: require positive prefix rows and finite thresholds in [0, 1]")

def read_codes(path: str) -> np.ndarray:
    try:
        lines = [ln.strip() for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
        rows = [[int(x) for x in ln.split()] for ln in lines]
        codes = np.asarray(rows, dtype=np.int64)
    except (OSError, ValueError, OverflowError) as error:
        sys.exit(f"[qwen3tts-parity] FAIL: invalid codec matrix {path}: {error}")
    if codes.ndim != 2 or codes.size == 0 or not np.isfinite(codes).all():
        sys.exit(f"[qwen3tts-parity] FAIL: expected a nonempty finite 2D codec matrix at {path}; shape={codes.shape}")
    return codes

lean = read_codes(lean_path)
py = read_codes(py_path)
if lean.shape != py.shape:
    print(f"[qwen3tts-parity] shape mismatch lean={lean.shape} py={py.shape}")
    sys.exit(1)

match = (lean == py)
full_tok = float(match.mean())
full_row = float(match.all(axis=1).mean())
nonzero_ratio = float((lean != 0).mean())

pref_n = min(prefix_rows, lean.shape[0])
pref = match[:pref_n]
pref_tok = float(pref.mean())
pref_row = float(pref.all(axis=1).mean())

print(f"[qwen3tts-parity] shape={lean.shape}")
print(f"[qwen3tts-parity] prefix_rows={pref_n} prefix_token_match={pref_tok:.6f} prefix_row_exact={pref_row:.6f}")
print(f"[qwen3tts-parity] full_token_match={full_tok:.6f} full_row_exact={full_row:.6f} nonzero_ratio={nonzero_ratio:.6f}")

ok = True
if pref_tok < prefix_tok_min:
    print(f"[qwen3tts-parity] FAIL: prefix token match {pref_tok:.6f} < {prefix_tok_min:.6f}")
    ok = False
if pref_row < prefix_row_min:
    print(f"[qwen3tts-parity] FAIL: prefix row exact {pref_row:.6f} < {prefix_row_min:.6f}")
    ok = False
if full_tok < full_tok_min:
    print(f"[qwen3tts-parity] FAIL: full token match {full_tok:.6f} < {full_tok_min:.6f}")
    ok = False
if nonzero_ratio < nonzero_min:
    print(f"[qwen3tts-parity] FAIL: nonzero ratio {nonzero_ratio:.6f} < {nonzero_min:.6f}")
    ok = False

if not ok:
    sys.exit(1)
PY

echo "[qwen3tts-parity] PASS"
