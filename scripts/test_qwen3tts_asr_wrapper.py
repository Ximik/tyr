#!/usr/bin/env python3
"""Model-free checks that strict ASR qualification cannot hide decoder fallback."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
MOCK_LAKE = r'''#!/usr/bin/env python3
import json, os, pathlib, struct, sys, wave
args = sys.argv[1:]
if len(args) < 2 or args[0] != "env":
    raise SystemExit("Unexpected Lake command: " + repr(args))
binary = pathlib.Path(args[1]).name
root = pathlib.Path(os.environ["TYR_WRAPPER_TEST_ROOT"])
if binary == "Qwen3TTSEndToEnd":
    (root / "tts-args.json").write_text(json.dumps(args[2:]))
    behavior = os.environ["TYR_WRAPPER_TEST_BEHAVIOR"]
    print("synthetic TTS stderr retained", file=sys.stderr)
    device = os.environ["TYR_WRAPPER_TEST_TTS_DEVICE"]
    if device:
        print("Target device: torch.Device." + device)
    if behavior == "failure":
        raise SystemExit(7)
    wav_path = args[args.index("--wav-path") + 1]
    with wave.open(wav_path, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(1 if behavior == "wrong_width" else 2)
        wav.setframerate(16000)
        samples = [0] * 160 if behavior == "silent_audio" else [100, -100] * 80
        wav.writeframes(struct.pack("<" + "h" * 160, *samples))
    if behavior in ("fallback", "fallback_then_lean"):
        print("Lean speech-tokenizer decode unavailable (synthetic failure); falling back to Python decode bridge.")
        print(f"Saved waveform to {wav_path} (Python decode bridge)")
    if behavior in ("lean", "fallback_then_lean", "silent_audio", "wrong_width"):
        print(f"Saved waveform to {wav_path} (Lean decoder)")
elif binary == "Qwen3ASRTranscribe":
    (root / "asr-ran").touch()
    device = os.environ["TYR_WRAPPER_TEST_ASR_DEVICE"]
    if device:
        print("Qwen3-ASR target device: torch.Device." + device)
    print("TEXT_BEGIN\nRegression audio validation\nTEXT_END")
else:
    raise SystemExit("Unexpected executable: " + binary)
'''


class AsrWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tyr asr wrapper ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        (self.bin / "lake").write_text(MOCK_LAKE)
        (self.bin / "lake").chmod(0o755)
        # WAV metadata/validation must not depend on an optional system ffprobe.
        (self.bin / "ffprobe").write_text("#!/bin/sh\necho broken-ffprobe >&2\nexit 127\n")
        (self.bin / "ffprobe").chmod(0o755)
        self.output = self.root / "output"
        self.output.mkdir()
        self.reference = self.root / "pinned reference"
        self.reference.mkdir()
        self.audio = self.root / "reference.wav"
        self.audio.touch()

    def run_wrapper(self, behavior, strict=True, tts_device="CUDA 0", asr_device="CUDA 0"):
        env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
                   TYR_SKIP_QUALIFICATION_BUILD="1", TYR_QUALIFICATION_STRICT="1" if strict else "0",
                   TYR_QUALIFICATION_PYTHON=sys.executable,
                   QWEN3_TTS_MODEL_DIR=str(self.root), QWEN3_ASR_MODEL_DIR=str(self.root),
                   QWEN3_TTS_REF_AUDIO=str(self.audio), QWEN3_TTS_REPO=str(self.reference),
                   QWEN3_TTS_ASR_REGRESSION_OUT_DIR=str(self.output),
                   TYR_WRAPPER_TEST_ROOT=str(self.root), TYR_WRAPPER_TEST_BEHAVIOR=behavior,
                   TYR_WRAPPER_TEST_TTS_DEVICE=tts_device, TYR_WRAPPER_TEST_ASR_DEVICE=asr_device)
        return subprocess.run(["bash", str(REPO / "scripts/qwen3tts_asr_regression.sh")],
                              cwd=REPO, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=30)

    def test_strict_accepts_lean_decode_and_passes_pinned_bridge_arguments(self):
        result = self.run_wrapper("lean")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("[qwen3tts-asr] PASS", result.stdout)
        self.assertNotIn("broken-ffprobe", result.stdout)
        self.assertIn("format=PCM16 sr=16000 channels=1 frames=160 duration=0.010000s", result.stdout)
        self.assertTrue((self.root / "asr-ran").exists())
        args = json.loads((self.root / "tts-args.json").read_text())
        self.assertEqual(args[args.index("--python") + 1], sys.executable)
        self.assertEqual(args[args.index("--qwen-repo") + 1], str(self.reference))
        self.assertIn("synthetic TTS stderr retained", (self.output / "tts.log").read_text())

    def test_strict_rejects_fallback_before_asr_even_with_lean_marker(self):
        for behavior in ("fallback", "fallback_then_lean"):
            with self.subTest(behavior=behavior):
                result = self.run_wrapper(behavior)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("strict qualification requires Lean decoding", result.stdout)
                self.assertNotIn("[qwen3tts-asr] PASS", result.stdout)
                self.assertFalse((self.root / "asr-ran").exists())

    def test_strict_rejects_missing_completion_despite_stale_success_log(self):
        (self.output / "tts.log").write_text(
            f"Saved waveform to {self.output / 'tts.wav'} (Lean decoder)\n")
        result = self.run_wrapper("missing")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("missing Lean decode completion marker", result.stdout)
        self.assertFalse((self.root / "asr-ran").exists())

    def test_nonstrict_preserves_python_decode_fallback(self):
        result = self.run_wrapper("fallback", strict=False, tts_device="CPU", asr_device="MPS")
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("[qwen3tts-asr] PASS", result.stdout)
        self.assertTrue((self.root / "asr-ran").exists())

    def test_strict_rejects_missing_or_non_cuda_tts_evidence_before_asr(self):
        for device in ("", "CPU", "MPS"):
            with self.subTest(device=device):
                result = self.run_wrapper("lean", tts_device=device)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("missing TTS CUDA execution evidence", result.stdout)
                self.assertFalse((self.root / "asr-ran").exists())

    def test_strict_rejects_missing_or_non_cuda_asr_evidence(self):
        for device in ("", "CPU", "MPS"):
            with self.subTest(device=device):
                result = self.run_wrapper("lean", asr_device=device)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("missing ASR CUDA execution evidence", result.stdout)
                self.assertNotIn("[qwen3tts-asr] PASS", result.stdout)

    def test_tts_failure_is_preserved_through_log_capture(self):
        result = self.run_wrapper("failure")
        self.assertEqual(result.returncode, 7, result.stdout)
        self.assertFalse((self.root / "asr-ran").exists())
        self.assertIn("synthetic TTS stderr retained", (self.output / "tts.log").read_text())

    def test_wave_validation_still_gates_asr_without_ffprobe(self):
        for behavior, message in [("silent_audio", "waveform energy too low"),
                                  ("wrong_width", "expected 16-bit PCM")]:
            with self.subTest(behavior=behavior):
                result = self.run_wrapper(behavior)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn(message, result.stdout)
                self.assertNotIn("broken-ffprobe", result.stdout)
                self.assertFalse((self.root / "asr-ran").exists())


if __name__ == "__main__":
    unittest.main()
