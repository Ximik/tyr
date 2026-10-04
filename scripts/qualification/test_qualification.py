#!/usr/bin/env python3
"""Hardware-free regressions for fixture integrity and strict execution gates."""
import hashlib
import json
import io
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import wave

from prepare import ensure_file, ensure_pcm16, safe_path, verify
from audio_fixture import TRANSFORM, pcm16_wav
from gpu_plan import configuration as gpu_configuration
from cuda_runtime import library_path, loader_libraries
from readiness import configuration
from run import validate_result, wait_for_idle_gpu
from summarize import summarize


class QualificationTests(unittest.TestCase):
    @staticmethod
    def float_wav(samples, channels=1, rate=24000, kind=3, bits=32):
        data = struct.pack("<" + "f" * len(samples), *samples)
        fmt = struct.pack("<HHIIHH", kind, channels, rate, rate * channels * 4, channels * 4, bits)
        # Include an odd-length unknown chunk to exercise RIFF padding.
        chunks = b"JUNK\x01\x00\x00\x00x\x00" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
        chunks += b"data" + struct.pack("<I", len(data)) + data
        return b"RIFF" + struct.pack("<I", len(chunks) + 4) + b"WAVE" + chunks

    @staticmethod
    def audio_spec(frames, channels=1, rate=24000):
        return dict(transformation=TRANSFORM, frames=frames, channels=channels,
                    sample_rate=rate, bits_per_sample=16)

    def test_pcm16_conversion_preserves_frames_and_rounds_saturates_exactly(self):
        values = [-2.0, -1.0, -0.5, -1.5 / 32768, -0.5 / 32768,
                  0.0, 0.5 / 32768, 1.5 / 32768, 0.5, 1.0, 2.0, 0.25]
        spec = self.audio_spec(6, channels=2, rate=16000)
        converted = pcm16_wav(self.float_wav(values, channels=2, rate=16000), spec)
        with wave.open(io.BytesIO(converted)) as audio:
            self.assertEqual((audio.getnchannels(), audio.getframerate(), audio.getnframes(),
                              audio.getsampwidth(), audio.getcomptype()), (2, 16000, 6, 2, "NONE"))
            self.assertEqual(struct.unpack("<12h", audio.readframes(6)),
                (-32768, -32768, -16384, -2, 0, 0, 0, 2, 16384, 32767, 32767, 8192))

    def test_pcm16_conversion_rejects_invalid_dtype_layout_and_values(self):
        good = self.float_wav([0.25])
        bad_layout = bytearray(good)
        struct.pack_into("<H", bad_layout, good.index(b"fmt ") + 8 + 12, 2)
        for source, spec in [
            (self.float_wav([float("nan")]), self.audio_spec(1)),
            (self.float_wav([float("inf")]), self.audio_spec(1)),
            (self.float_wav([0.25], kind=1), self.audio_spec(1)),
            (self.float_wav([0.25], bits=64), self.audio_spec(1)),
            (good, self.audio_spec(1, rate=48000)),
            (good, self.audio_spec(1, channels=2)),
            (good, self.audio_spec(2)),
            (good, dict(self.audio_spec(1), transformation="unknown")),
            (good[:-1], self.audio_spec(1)),
            (bad_layout, self.audio_spec(1)),
        ]:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                pcm16_wav(source, spec)

    def test_derived_audio_requires_its_pinned_checksum_and_preserves_original(self):
        with tempfile.TemporaryDirectory() as directory:
            source, target = Path(directory) / "float.wav", Path(directory) / "pcm.wav"
            original = self.float_wav([0.5, -0.5])
            source.write_bytes(original)
            spec = self.audio_spec(2)
            expected = pcm16_wav(original, spec)
            spec.update(size=len(expected), sha256=hashlib.sha256(expected).hexdigest())
            with self.assertRaises(RuntimeError):
                ensure_pcm16(source, target, spec, False)
            with self.assertRaisesRegex(RuntimeError, "checksum"):
                ensure_pcm16(source, target, dict(spec, sha256="0" * 64), True)
            self.assertFalse(target.exists())
            ensure_pcm16(source, target, spec, True)
            ensure_pcm16(source, target, spec, False)
            self.assertEqual(target.read_bytes(), expected)
            self.assertEqual(source.read_bytes(), original)
            target.write_bytes(b"corrupt")
            with self.assertRaises(RuntimeError):
                ensure_pcm16(source, target, spec, False)

    def test_wheel_cuda_dependencies_precede_host_toolkit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            torch = root / "site-packages/torch"
            vendor = root / "site-packages/nvidia/cu13/lib"
            toolkit = root / "cuda/lib64"
            for path in (torch / "lib", vendor, toolkit):
                path.mkdir(parents=True)
            (torch / "__init__.py").touch()
            paths = library_path(torch, root / "cuda", str(toolkit) + ":").split(":")
            self.assertEqual(paths, list(map(str, (torch / "lib", vendor, toolkit))))
            # A standalone LibTorch archive has no Python-wheel layout.
            (torch / "__init__.py").unlink()
            self.assertEqual(library_path(torch, root / "cuda").split(":"),
                             list(map(str, (torch / "lib", toolkit))))

    def test_loader_evidence_records_initialized_libraries_not_search_candidates(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "loader.123"
            log.write_text("123: trying file=/wrong/libcublasLt.so.13\n"
                           "123: calling init: /pinned/libcublasLt.so.13\n"
                           "124: calling init: /pinned/libcublas.so.13\n"
                           "124: calling init: /pinned/libcudart.so.13\n"
                           "124: calling init: /other/libc.so.6\n")
            self.assertEqual(loader_libraries([log]), ["/pinned/libcublas.so.13",
                "/pinned/libcublasLt.so.13", "/pinned/libcudart.so.13"])

    def test_new_gpu_workload_is_waited_for_without_being_interrupted(self):
        with patch("run.subprocess.check_output", side_effect=["12345\n", ""]), \
             patch("run.time.monotonic", side_effect=[0.0, 0.0, 1.25]), \
             patch("run.time.sleep") as sleep, patch("builtins.print"):
            self.assertEqual(wait_for_idle_gpu(), 1.25)
            sleep.assert_called_once_with(5.0)
        with patch("run.subprocess.check_output", return_value="12345\n"), \
             patch("run.time.monotonic", side_effect=[0.0, 600.0]), \
             patch("run.time.sleep") as sleep:
            with self.assertRaisesRegex(ValueError, "remains occupied"):
                wait_for_idle_gpu()
            sleep.assert_not_called()

    def test_decode_build_inputs_match_hardware_and_production_dispatch(self):
        hopper = gpu_configuration("H100")
        self.assertIn("Tyr.GPU.Kernels.MhaH100Decode", hopper["modules"])
        self.assertEqual(hopper["decode_route"], "hopper_custom_kernel_when_eligible")
        for gpu in ("GB10", "B200", "B300"):
            with self.subTest(gpu=gpu):
                plan = gpu_configuration(gpu)
                self.assertNotIn("Tyr.GPU.Kernels.MhaH100Decode", plan["modules"])
                self.assertNotIn("Tyr.GPU.Kernels.MhaH100", plan["modules"])
                self.assertEqual(plan["runner"], "TestGPUGB10E2E")
                self.assertEqual(plan["decode_route"], "sdpa_fallback")
        with self.assertRaises(ValueError):
            gpu_configuration("A100")

    def test_empty_skipped_and_failed_gpu_runs_cannot_qualify(self):
        for output, code in [
            ("", 0), ("all good", 0),
            ("[gpu-coverage] executed=0 skipped=0 failed=0 selected=0 strict=true", 0),
            ("[gpu-coverage] executed=4 skipped=1 failed=0 selected=5 strict=true", 0),
            ("[gpu-coverage] executed=4 skipped=0 failed=1 selected=4 strict=true", 0),
            ("[gpu-coverage] executed=4 skipped=0 failed=0 selected=4 strict=true", 1),
            ("[gpu-coverage] executed=4 skipped=0 failed=0 selected=4 strict=false", 0),
        ]:
            with self.subTest(output=output, code=code), self.assertRaises(ValueError):
                validate_result("gpu", output, code)
        self.assertEqual(validate_result("gpu",
            "[gpu-coverage] executed=10 skipped=0 failed=0 selected=10 strict=true", 0), 10)

    def test_model_and_cache_checks_require_actual_success_markers(self):
        for kind in ("parity", "asr", "decode", "laguna", "native"):
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                validate_result(kind, "", 0)
        with self.assertRaises(ValueError):
            validate_result("asr", "SKIP missing model\n[qwen3tts-asr] PASS", 0)
        with self.assertRaises(ValueError):
            validate_result("laguna", "CUDA not available; skipped CUDA cases.\nAll Laguna model tests passed.", 0)
        with self.assertRaises(ValueError):
            validate_result("laguna", "All Laguna model tests passed.", 0)
        self.assertEqual(validate_result("laguna", "CACHE_BENCH device=cuda:0 capacity=128\nAll Laguna model tests passed.", 0), 1)
        self.assertEqual(validate_result("laguna", "PASS: session rejects skipped positions without damaging state\nCACHE_BENCH device=cuda:0 capacity=128\nAll Laguna model tests passed.", 0), 1)

    def test_fixture_cache_rejects_corruption_missing_files_and_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "fixture.bin"
            content = b"pinned fixture"
            spec = {"size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            with self.assertRaises(RuntimeError):
                ensure_file(path, spec, "https://invalid.example/unused", False)
            path.write_bytes(content)
            self.assertTrue(verify(path, spec))
            blob = {"size": len(content), "git_blob_sha1": hashlib.sha1(b"blob 14\0" + content).hexdigest()}
            self.assertTrue(verify(path, blob))
            path.write_bytes(b"corrupt bytes!")
            self.assertFalse(verify(path, spec))
            with self.assertRaises(RuntimeError):
                ensure_file(path, spec, "https://invalid.example/unused", False)
            with self.assertRaises(ValueError):
                safe_path(root, "../escape")

    def test_runner_association_is_explicit(self):
        with self.assertRaises(ValueError):
            configuration("ranvier-labs/tyr", "")
        labels = configuration("cpehle/tyr", "")
        self.assertIn("gb10", json.loads(labels))
        self.assertIn("tyr-qualification", json.loads(configuration("ranvier-labs/tyr",
            '["self-hosted", "Linux", "ARM64", "tyr-qualification", "gb10"]')))

    def test_final_summary_fails_missing_runtime_only_or_skipped_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            self.assertEqual(summarize(path, ["gpu"])["status"], "failed")
            for status, executed, skipped in [("runtime_ready", 0, 0), ("passed", 0, 0), ("passed", 1, 1)]:
                (path / "gpu.json").write_text(json.dumps({"status": status, "executed": executed, "skipped": skipped, "failed": 0}))
                self.assertEqual(summarize(path, ["gpu"])["status"], "failed")
            (path / "gpu.json").write_text(json.dumps({"status": "passed", "executed": 10, "skipped": 0, "failed": 0}))
            self.assertEqual(summarize(path, ["gpu"])["status"], "passed")
            self.assertEqual(summarize(path, ["gpu", "models"])["status"], "failed")


if __name__ == "__main__":
    unittest.main()
