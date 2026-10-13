#!/usr/bin/env bash
set -euo pipefail

SKIP_EXIT_CODE=2
if [[ "${TYR_QUALIFICATION_STRICT:-0}" == 1 ]]; then SKIP_EXIT_CODE=1; fi
CHECK_ONLY=false

usage() {
  cat <<'USAGE'
Usage: scripts/qwen3tts_asr_regression.sh [--check-prereqs]

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

TTS_MODEL_DIR="${QWEN3_TTS_MODEL_DIR:-weights/qwen3-tts-0.6b-base}"
ASR_MODEL_DIR="${QWEN3_ASR_MODEL_DIR:-weights/qwen3-asr-0.6b}"
REF_AUDIO_PATH="${QWEN3_TTS_REF_AUDIO:-MLKDream.wav}"
QWEN_REPO="${QWEN3_TTS_REPO:-../Qwen3-TTS}"
OUT_DIR="${QWEN3_TTS_ASR_REGRESSION_OUT_DIR:-output/asr_regression}"
WAV_PATH="$OUT_DIR/tts.wav"
CODES_PATH="$OUT_DIR/tts.codes"
ASR_OUT="$OUT_DIR/asr.txt"
TTS_LOG="$OUT_DIR/tts.log"
TYR_DEVICE="${TYR_DEVICE:-mps}"

check_prereqs() {
  local missing=0

  if [[ ! -d "$TTS_MODEL_DIR" ]]; then
    echo "[qwen3tts-asr] SKIP: missing TTS model dir: $TTS_MODEL_DIR"
    missing=1
  fi
  if [[ ! -d "$ASR_MODEL_DIR" ]]; then
    echo "[qwen3tts-asr] SKIP: missing ASR model dir: $ASR_MODEL_DIR"
    missing=1
  fi
  if [[ ! -f "$REF_AUDIO_PATH" ]]; then
    echo "[qwen3tts-asr] SKIP: missing reference audio: $REF_AUDIO_PATH"
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
  echo "[qwen3tts-asr] prerequisite check: ready"
  exit 0
fi

mkdir -p "$OUT_DIR"

echo "[qwen3tts-asr] building executables"
if [[ "${TYR_SKIP_QUALIFICATION_BUILD:-0}" != 1 ]]; then
  "${RUN_ENV[@]}" lake build Qwen3TTSEndToEnd Qwen3ASRTranscribe >/dev/null
fi

echo "[qwen3tts-asr] generating audio"
TYR_DEVICE="$TYR_DEVICE" "${RUN_ENV[@]}" lake env ./.lake/build/bin/Qwen3TTSEndToEnd \
  --model-dir "$TTS_MODEL_DIR" \
  --seed "${TYR_QUALIFICATION_SEED:-0}" \
  --text "Regression audio validation" \
  --max-frames 40 \
  --ref-audio-path "$REF_AUDIO_PATH" \
  --python "$BRIDGE_PYTHON" \
  --qwen-repo "$QWEN_REPO" \
  --codes-path "$CODES_PATH" \
  --wav-path "$WAV_PATH" 2>&1 | tee "$TTS_LOG"

if [[ "${TYR_QUALIFICATION_STRICT:-0}" == 1 ]]; then
  if ! grep -Eq '^Target device: torch\.Device\.CUDA [0-9]+$' "$TTS_LOG"; then
    echo "[qwen3tts-asr] FAIL: missing TTS CUDA execution evidence in $TTS_LOG"
    exit 1
  fi
  if grep -Fq 'Python decode bridge' "$TTS_LOG"; then
    echo "[qwen3tts-asr] FAIL: strict qualification requires Lean decoding; Python fallback recorded in $TTS_LOG"
    exit 1
  fi
  if ! grep -Fxq "Saved waveform to $WAV_PATH (Lean decoder)" "$TTS_LOG"; then
    echo "[qwen3tts-asr] FAIL: missing Lean decode completion marker in $TTS_LOG"
    exit 1
  fi
fi

if [[ ! -f "$WAV_PATH" ]]; then
  echo "[qwen3tts-asr] FAIL: wav not generated"
  exit 1
fi

"${RUN_ENV[@]}" "$PYTHON_BIN" - "$WAV_PATH" <<'PY'
import sys, wave, struct, math
path = sys.argv[1]
with wave.open(path, 'rb') as w:
    n = w.getnframes()
    ch = w.getnchannels()
    sw = w.getsampwidth()
    sr = w.getframerate()
    data = w.readframes(n)
if sw != 2:
    print(f"[qwen3tts-asr] FAIL: expected 16-bit PCM, got sampwidth={sw}")
    sys.exit(1)
vals = struct.unpack('<' + 'h' * (len(data) // 2), data)
if ch > 1:
    vals = vals[::ch]
rms = math.sqrt(sum(v * v for v in vals) / max(1, len(vals)))
print(f"[qwen3tts-asr] wav stats: format=PCM16 sr={sr} channels={ch} frames={n} duration={n / sr:.6f}s samples={len(vals)} rms={rms:.2f}")
if rms < 20.0:
    print("[qwen3tts-asr] FAIL: waveform energy too low")
    sys.exit(1)
PY

echo "[qwen3tts-asr] transcribing generated audio"
TYR_DEVICE="$TYR_DEVICE" "${RUN_ENV[@]}" lake env ./.lake/build/bin/Qwen3ASRTranscribe \
  --model-dir "$ASR_MODEL_DIR" \
  --wav-path "$WAV_PATH" \
  --max-new-tokens 64 2>&1 | tee "$ASR_OUT"

if [[ "${TYR_QUALIFICATION_STRICT:-0}" == 1 ]] && \
    ! grep -Eq '^Qwen3-ASR target device: torch\.Device\.CUDA [0-9]+$' "$ASR_OUT"; then
  echo "[qwen3tts-asr] FAIL: missing ASR CUDA execution evidence in $ASR_OUT"
  exit 1
fi

TRANSCRIPT="$(awk '/^TEXT_BEGIN$/{flag=1;next}/^TEXT_END$/{flag=0}flag{print}' "$ASR_OUT" | tr '\n' ' ' | sed 's/[[:space:]]\+/ /g; s/^ //; s/ $//')"
if [[ -z "$TRANSCRIPT" ]]; then
  echo "[qwen3tts-asr] FAIL: empty transcript"
  cat "$ASR_OUT"
  exit 1
fi

echo "[qwen3tts-asr] transcript: $TRANSCRIPT"
echo "[qwen3tts-asr] PASS"
