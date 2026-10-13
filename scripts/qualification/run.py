#!/usr/bin/env python3
"""Strict CUDA/real-model execution with machine-readable qualification evidence."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

from prepare import MANIFEST, prepare
from gpu_plan import configuration as gpu_configuration
from cuda_runtime import LIBRARIES, library_path, loader_libraries

REPO = Path(__file__).resolve().parents[2]
COVERAGE = re.compile(r"\[gpu-coverage\] executed=(\d+) skipped=(\d+) failed=(\d+).*strict=true")


def wait_for_idle_gpu(timeout=600.0, interval=5.0):
    """A noncooperating service may acquire CUDA during our CPU build."""
    start = time.monotonic()
    while True:
        active = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).strip()
        if not active:
            return time.monotonic() - start
        elapsed = time.monotonic() - start
        if elapsed >= timeout:
            raise ValueError(f"GPU remains occupied by unrelated processes after {timeout:g}s: {active}")
        print(f"GPU occupied by processes {active.splitlines()}; waiting without interrupting them.", flush=True)
        time.sleep(min(interval, timeout - elapsed))


def validate_result(kind, output, code):
    if code:
        raise ValueError(f"Process exited with {code}")
    if re.search(r"(?m)^\s*SKIP(?:\s|:)|\[skip\]|\bSKIP:|\bskipped CUDA\b|skipped due to", output, re.IGNORECASE):
        raise ValueError("Strict qualification reported a skipped case")
    if kind == "gpu":
        matches = COVERAGE.findall(output)
        if not matches:
            raise ValueError("GPU runner did not produce strict executed/skipped accounting")
        executed, skipped, failed = map(int, matches[-1])
        if executed == 0 or skipped or failed:
            raise ValueError(f"Insufficient GPU coverage: {executed=}, {skipped=}, {failed=}")
        return executed
    marker = {"decode": "cache_parity: ok=true", "native": "backend checks passed on cuda",
              "laguna": "All Laguna model tests passed.", "parity": "[qwen3tts-parity] PASS",
              "asr": "[qwen3tts-asr] PASS"}[kind]
    if marker not in output:
        raise ValueError(f"Missing successful execution marker: {marker}")
    if kind == "laguna" and "cache_bench device=cuda" not in output.lower():
        raise ValueError("Laguna did not execute the CUDA owned-cache benchmark")
    return 1


def describe_libraries(python, paths, env):
    if not paths:
        return []
    code = """
import json, sys
sys.path.insert(0, sys.argv[1])
from cuda_runtime import library_versions
print(json.dumps(library_versions(sys.argv[2:])))
"""
    # Keep diagnostic CDLL loads out of the coordinator that performs the
    # next GPU-idle check; a failing loader/version query is contained here.
    return json.loads(subprocess.check_output(
        [python, "-c", code, str(REPO / "scripts/qualification"), *paths], env=env, text=True))


def runtime(python, manifest, real_models, env=None):
    code = """
import importlib.metadata as m, json, platform, sys, torch
sys.path.insert(0, sys.argv[1])
from cuda_runtime import library_versions, process_libraries
packages = {}
for distribution in m.distributions():
    packages.setdefault(distribution.metadata['Name'], distribution.version)
print(json.dumps({'torch':torch.__version__, 'cuda':torch.version.cuda,
 'arch':platform.machine(),
 'cuda_available':torch.cuda.is_available(), 'torch_dir':str(__import__('pathlib').Path(torch.__file__).parent),
 'devices':[{'name':torch.cuda.get_device_name(i),'capability':torch.cuda.get_device_capability(i)}
            for i in range(torch.cuda.device_count())],
 'packages':packages, 'loaded_cuda_libraries':library_versions(process_libraries())}))
"""
    info = json.loads(subprocess.check_output([python, "-c", code, str(REPO / "scripts/qualification")], text=True, env=env))
    if not info["cuda_available"]:
        raise ValueError("CUDA is unavailable in the selected Python reference runtime")
    if info["torch"] != manifest["python_packages"]["torch"] or info["cuda"] != manifest["cuda_version"]:
        raise ValueError(f"Expected pinned Torch/CUDA versions; found {info['torch']}/{info['cuda']}")
    if (REPO / "external/wheels/torch").resolve() != Path(info["torch_dir"]).resolve():
        raise ValueError("Lean LibTorch and Python reference must use the same pinned Torch installation")
    config = (REPO / "external/wheels/torch/share/cmake/Torch/TorchConfigVersion.cmake").read_text()
    if f'set(PACKAGE_VERSION "{manifest["libtorch_version"]}")' not in config:
        raise ValueError("LibTorch version does not match qualification manifest")
    nvcc = subprocess.check_output([os.path.join(env["CUDA_HOME"], "bin", "nvcc"), "--version"], text=True, env=env)
    if f'release {manifest["cuda_version"]},' not in nvcc:
        raise ValueError("NVCC does not match the pinned CUDA version")
    info["nvcc"] = nvcc.strip()
    if real_models:
        packages = {name.lower().replace("_", "-"): version for name, version in info["packages"].items()}
        required = dict(manifest["python_packages"])
        by_arch = manifest.get("python_packages_by_arch", {})
        if info["arch"] not in by_arch:
            raise ValueError(f"No pinned model environment for architecture {info['arch']}")
        required.update(by_arch[info["arch"]])
        for name, expected in required.items():
            if packages.get(name) != expected:
                raise ValueError(f"Expected {name}=={expected}, found {packages.get(name)}")
    return info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("gpu", "models"), required=True)
    parser.add_argument("--python", default=os.environ.get("TYR_QUALIFICATION_PYTHON", sys.executable))
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--check-runtime", action="store_true")
    args = parser.parse_args()
    os.chdir(REPO)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    report = {"status": "blocked", "executed": 0, "skipped": 0, "failed": 0, "suites": [],
              "kind": args.kind, "fixture_manifest_sha256": hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
              "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
              "source_status": subprocess.check_output(
                  ["git", "status", "--porcelain", "--untracked-files=normal"], text=True).splitlines()}
    env = dict(os.environ, TYR_GPU_TEST_STRICT="1", TYR_QUALIFICATION_STRICT="1",
               TYR_QUALIFICATION_PYTHON=args.python, TYR_SKIP_QUALIFICATION_BUILD="1",
               TYR_DEVICE="cuda", QWEN3_TTS_DEVICE_MAP="cuda:0", PYTHONHASHSEED="0")
    env["LIBTORCH_DIR"] = str(REPO / "external/wheels/torch")
    env["TYR_LAGUNA_CACHE_BENCH"] = "1"
    env["PATH"] = str(Path(args.python).parent) + os.pathsep + env.get("PATH", "")
    env["LD_LIBRARY_PATH"] = library_path(REPO / "external/wheels/torch",
        Path(env["CUDA_HOME"]), env.get("LD_LIBRARY_PATH", ""))
    try:
        if report["source_status"]:
            raise ValueError("Strict qualification requires a clean committed candidate checkout")
        report["preflight_gpu_wait_seconds"] = wait_for_idle_gpu()
        report["runtime"] = runtime(args.python, manifest, args.kind == "models", env)
        if args.check_runtime:
            report["status"] = "runtime_ready"
            return 0
        if args.kind == "gpu":
            gpu = env.get("TYR_GPU_TARGET", env.get("GPU", "GB10"))
            plan = gpu_configuration(gpu)
            report["gpu_plan"] = plan
            if not any(gpu in device["name"] for device in report["runtime"]["devices"]):
                raise ValueError(f"Configured GPU {gpu} does not match the detected devices")
            suite = plan["runner"]
            arguments = ["--filter", "TorchParity", "--fail-fast"] if gpu == "H100" else ["--fail-fast"]
            commands = [("gpu", ["lake", "env", f"./.lake/build/bin/{suite}", *arguments]),
                        ("decode", ["lake", "env", "./.lake/build/bin/RunMhaH100Decode", "--regen"]),
                        ("native", ["bash", "scripts/test_native_attention.sh", "cuda"]),
                        ("laguna", ["lake", "env", "./.lake/build/bin/LagunaModelTest"])]
        else:
            if args.cache is None:
                raise ValueError("--cache is required for pinned model qualification")
            paths = prepare(args.cache, manifest)
            report["fixture_paths"] = paths
            audio = manifest["audio"]
            report["audio_fixture"] = {
                "source": dict(audio, path=paths["audio-source"]),
                "derived": dict(audio["derived_pcm16"], path=paths["audio"])}
            del report["audio_fixture"]["source"]["derived_pcm16"]
            env.update(QWEN3_TTS_MODEL_DIR=paths["qwen3-tts"], QWEN3_ASR_MODEL_DIR=paths["qwen3-asr"],
                       QWEN3_TTS_REPO=paths["qwen-reference"], QWEN3_TTS_PARITY_AUDIO=paths["audio"],
                       QWEN3_TTS_REF_AUDIO=paths["audio"], TYR_QUALIFICATION_SEED="0",
                       QWEN3_TTS_PARITY_OUT_DIR=str(args.report.parent / "parity"),
                       QWEN3_TTS_ASR_REGRESSION_OUT_DIR=str(args.report.parent / "asr"))
            env["PYTHONPATH"] = paths["qwen-reference"] + os.pathsep + env.get("PYTHONPATH", "")
            commands = [("parity", ["bash", "scripts/qwen3tts_parity_regression.sh"]),
                        ("asr", ["bash", "scripts/qwen3tts_asr_regression.sh"])]
        for name, command in commands:
            item = {"name": name, "status": "failed", "command": command}
            if name == "decode":
                item["production_route"] = report["gpu_plan"]["decode_route"]
            report["suites"].append(item)
            item["gpu_wait_seconds"] = wait_for_idle_gpu()
            start = time.monotonic()
            log_path = args.report.parent / (args.kind + "-" + name + ".log")
            loader_prefix = args.report.parent.resolve() / (args.kind + "-" + name + "-loader")
            for stale in loader_prefix.parent.glob(loader_prefix.name + ".*"):
                stale.unlink()
            command_env = dict(env, LD_DEBUG="libs", LD_DEBUG_OUTPUT=str(loader_prefix))
            with log_path.open("w") as log:
                with subprocess.Popen(command, env=command_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True) as process:
                    output = []
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        output.append(line)
                    code = process.wait()
            item.update(exit_code=code, seconds=time.monotonic() - start, log=str(log_path))
            paths = loader_libraries(loader_prefix.parent.glob(loader_prefix.name + ".*"))
            try:
                item["loaded_cuda_libraries"] = describe_libraries(args.python, paths, env)
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                item["library_evidence_error"] = str(error)
            output = "".join(output)
            counts = COVERAGE.findall(output) if name == "gpu" else []
            if counts:
                executed, skipped, failed = map(int, counts[-1])
                item.update(executed=executed, skipped=skipped, failed=failed)
                report["executed"] += executed
                report["skipped"] += skipped
                report["failed"] += failed
            executed = validate_result(name, output, code)
            if "library_evidence_error" in item:
                raise ValueError(f"CUDA library evidence failed for {name}: {item['library_evidence_error']}")
            if any(not any(Path(path).name.startswith(lib) for path in paths) for lib in LIBRARIES):
                raise ValueError(f"Incomplete CUDA loader evidence for {name}")
            item.update(status="passed", executed=executed, skipped=0)
            if not counts:
                report["executed"] += executed
        report["status"] = "passed"
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        report.update(status="failed", failed=max(1, report["failed"]), reason=str(error))
        print(f"Qualification failed: {error}", file=sys.stderr)
        return 1
    finally:
        args.report.write_text(json.dumps(report, indent=2) + "\n")
        message = f"Qualification {args.kind}: status={report['status']} executed={report['executed']} skipped={report['skipped']} failed={report['failed']}"
        print(message, flush=True)
        if env.get("GITHUB_STEP_SUMMARY"):
            with open(env["GITHUB_STEP_SUMMARY"], "a") as out:
                out.write(message + "\n\n")


if __name__ == "__main__":
    raise SystemExit(main())
