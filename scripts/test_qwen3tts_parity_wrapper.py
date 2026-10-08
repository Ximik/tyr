#!/usr/bin/env python3
"""Exercise the real parity comparator with stub encoders; requires NumPy only."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]
CODES = "1 2\n3 4\n"
MOCK = r'''
import json, os, pathlib, sys
args = sys.argv[1:]
root = pathlib.Path(os.environ["TYR_PARITY_TEST_ROOT"])
if pathlib.Path(sys.argv[0]).name == "lake":
    if len(args) < 2 or args[:2] != ["env", "./.lake/build/bin/Qwen3TTSEndToEnd"]:
        raise SystemExit("Unexpected Lake command: " + repr(args))
    side, flag = "lean", "--encode-out-codes-path"
    device = os.environ["TYR_PARITY_TEST_DEVICE"]
    if device:
        print("Target device: torch.Device." + device)
elif args and args[0] == "scripts/qwen3tts_encode_audio.py":
    side, flag = "python", "--output-codes"
else:
    os.execv(sys.executable, [sys.executable, *args])
(root / (side + "-args.json")).write_text(json.dumps(args))
print(side + " encode stdout retained")
print(side + " encode stderr retained", file=sys.stderr)
if os.environ.get("TYR_PARITY_TEST_FAIL") == side:
    raise SystemExit(7)
if os.environ.get("TYR_PARITY_TEST_NO_WRITE") != side:
    pathlib.Path(args[args.index(flag) + 1]).write_text(os.environ["TYR_PARITY_TEST_" + side.upper()])
'''


class ParityWrapperTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="tyr parity wrapper ")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        for name in ("lake", "pinned-python"):
            command = self.bin / name
            command.write_text(f"#!{sys.executable}\n" + MOCK)
            command.chmod(0o755)
        self.output = self.root / "output"
        self.output.mkdir()
        self.audio = self.root / "reference.wav"
        self.audio.touch()
        self.reference = self.root / "pinned reference"
        self.reference.mkdir()

    def run_wrapper(self, lean=CODES, python=CODES, device="CUDA 0", strict=True, **overrides):
        env = dict(os.environ, PATH=str(self.bin) + os.pathsep + os.environ.get("PATH", ""),
                   TYR_SKIP_QUALIFICATION_BUILD="1", TYR_QUALIFICATION_STRICT="1" if strict else "0",
                   TYR_QUALIFICATION_PYTHON=str(self.bin / "pinned-python"),
                   QWEN3_TTS_MODEL_DIR=str(self.root), QWEN3_TTS_REPO=str(self.reference),
                   QWEN3_TTS_PARITY_AUDIO=str(self.audio), QWEN3_TTS_PARITY_OUT_DIR=str(self.output),
                   QWEN3_TTS_DEVICE_MAP="cuda:0", QWEN3_TTS_PARITY_PREFIX_ROWS="125",
                   QWEN3_TTS_PARITY_PREFIX_TOKEN_MIN="0.99", QWEN3_TTS_PARITY_PREFIX_ROW_MIN="0.99",
                   QWEN3_TTS_PARITY_FULL_TOKEN_MIN="0.10", QWEN3_TTS_PARITY_NONZERO_MIN="0.90",
                   TYR_PARITY_TEST_ROOT=str(self.root), TYR_PARITY_TEST_DEVICE=device,
                   TYR_PARITY_TEST_LEAN=lean, TYR_PARITY_TEST_PYTHON=python,
                   TYR_PARITY_TEST_FAIL="", TYR_PARITY_TEST_NO_WRITE="")
        env.update(overrides)
        return subprocess.run(["bash", str(REPO / "scripts/qwen3tts_parity_regression.sh")],
                              cwd=REPO, env=env, text=True, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, timeout=30)

    def test_valid_cuda_flow_retains_logs_and_pinned_arguments(self):
        result = self.run_wrapper()
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertIn("[qwen3tts-parity] PASS", result.stdout)
        lean_args = json.loads((self.root / "lean-args.json").read_text())
        python_args = json.loads((self.root / "python-args.json").read_text())
        self.assertEqual(lean_args[lean_args.index("--python") + 1], str(self.bin / "pinned-python"))
        self.assertEqual(lean_args[lean_args.index("--qwen-repo") + 1], str(self.reference))
        self.assertEqual(python_args[python_args.index("--qwen3-tts-repo") + 1], str(self.reference))
        self.assertEqual(python_args[python_args.index("--device-map") + 1], "cuda:0")
        for side in ("lean", "python"):
            log = (self.output / (side + "-encode.log")).read_text()
            self.assertIn(side + " encode stdout retained", log)
            self.assertIn(side + " encode stderr retained", log)

    def test_empty_matrices_are_rejected_explicitly(self):
        result = self.run_wrapper(lean="", python="")
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertIn("expected a nonempty finite 2D codec matrix", result.stdout)

    def test_invalid_or_nonfinite_matrices_are_rejected(self):
        for content in ("1 2\n3\n", "nan 2\n", "inf 2\n", "999999999999999999999999 2\n"):
            with self.subTest(content=content):
                result = self.run_wrapper(lean=content, python=content)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("invalid codec matrix", result.stdout)

    def test_strict_rejects_cpu_or_missing_device_before_reference_runs(self):
        for device in ("CPU", "MPS", ""):
            with self.subTest(device=device):
                result = self.run_wrapper(device=device)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("missing Lean CUDA execution evidence", result.stdout)
                self.assertFalse((self.root / "python-args.json").exists())

    def test_nonstrict_cpu_flow_remains_supported(self):
        result = self.run_wrapper(device="CPU", strict=False)
        self.assertEqual(result.returncode, 0, result.stdout)

    def test_command_failures_survive_log_capture(self):
        for side in ("lean", "python"):
            with self.subTest(side=side):
                result = self.run_wrapper(TYR_PARITY_TEST_FAIL=side)
                self.assertEqual(result.returncode, 7, result.stdout)
                self.assertIn(side + " encode stderr retained",
                              (self.output / (side + "-encode.log")).read_text())

    def test_zero_prefix_or_nan_threshold_cannot_bypass_comparison(self):
        for overrides in ({"QWEN3_TTS_PARITY_PREFIX_ROWS": "0"},
                          {"QWEN3_TTS_PARITY_PREFIX_TOKEN_MIN": "nan"}):
            with self.subTest(overrides=overrides):
                result = self.run_wrapper(**overrides)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("positive prefix rows and finite thresholds", result.stdout)

    def test_stale_codec_files_cannot_replace_missing_encoder_output(self):
        for side in ("lean", "python"):
            with self.subTest(side=side):
                (self.output / "lean.codes").write_text(CODES)
                (self.output / "python.codes").write_text(CODES)
                result = self.run_wrapper(TYR_PARITY_TEST_NO_WRITE=side)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertIn("invalid codec matrix", result.stdout)

    def test_shape_or_token_mismatch_still_fails_parity(self):
        for content in ("1 2\n", "5 6\n7 8\n"):
            with self.subTest(content=content):
                result = self.run_wrapper(python=content)
                self.assertNotEqual(result.returncode, 0, result.stdout)
                self.assertNotIn("[qwen3tts-parity] PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()
