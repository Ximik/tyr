#!/usr/bin/env python3
"""Compatibility boundary for reusable Lean/native build outputs in hosted CI.

Lake and Make still validate their dependency graphs after every restore. This
key prevents sharing binaries across incompatible hosts, dependencies or flags.
Only explicit build environment variables are recorded; never dump the full env.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess

REPO = Path(__file__).resolve().parents[2]
BUILD_ENV = (
    "CC", "CXX", "CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "LEAN_CC",
    "LEAN_CC_GCC", "LEAN_CC_FAST", "LIBRARY_PATH", "SDKROOT", "MACOSX_DEPLOYMENT_TARGET",
    "TYR_MACOS_SDKROOT", "TYR_MACOS_DEPLOYMENT_TARGET", "GPU", "GPU_FAMILY",
    "GPU_COMPUTE", "GPU_CODE", "TYR_GPU_TARGET", "TYR_GPU_FAMILY",
    "TYR_GPU_COMPUTE", "TYR_GPU_CODE", "TYR_GPU_CODEGEN_MODULE",
    "TYR_SKIP_GPU_CODEGEN", "TYR_BUILD_TYRC_DYLIB",
)


def command(*args):
    return subprocess.check_output(args, cwd=REPO, text=True).strip()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def keys(identity, source_tree):
    compatibility = digest(identity)
    return {"compatibility": compatibility, "source_tree": source_tree,
            "dependencies": f"tyr-deps-v1-{compatibility}",
            "build_prefix": f"tyr-build-v1-{compatibility}-",
            "build": f"tyr-build-v1-{compatibility}-{source_tree}"}


def identity():
    # deps/git.lock and deps/wheels.lock pin every native dependency; the libtorch
    # files below additionally record which variant (cpu/cuda) was fetched.
    files = ["lean-toolchain", "lake-manifest.json", "deps/git.lock", "deps/wheels.lock",
             "scripts/lean_cc_wrapper.sh", "env.sh"]
    torch_files = ["share/cmake/Torch/TorchConfigVersion.cmake",
                   "share/cmake/Torch/TorchConfig.cmake",
                   "include/torch/csrc/api/include/torch/version.h"]
    files += ["external/wheels/torch/" + path for path in torch_files]
    result = {
        "schema": 2, "workspace": str(REPO.resolve()),
        "os": platform.system(), "arch": platform.machine(), "release": platform.release(),
        "runner_image": {key: os.environ.get(key, "") for key in ("ImageOS", "ImageVersion")},
        "compiler": command("c++", "--version"), "lean": command("lean", "--version"),
        "environment": {key: os.environ.get(key, "") for key in BUILD_ENV},
        "files": {path: hashlib.sha256((REPO / path).read_bytes()).hexdigest() for path in files},
    }
    if platform.system() == "Darwin":
        result["sdk"] = command("xcrun", "--show-sdk-version")
    else:
        result["gcc"] = command("gcc", "--version")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = identity()
    result = keys(data, command("git", "rev-parse", "HEAD^{tree}"))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps({"identity": data, "keys": result}, indent=2) + "\n")
    for key, value in result.items():
        print(f"{key}={value}")


if __name__ == "__main__":
    main()
