#!/usr/bin/env bash
# Prepare the Modal image from the repository root: the pinned native
# dependencies, the reference venv at /opt/ref-venv, a CUDA (H100) Lake
# configuration, and every parity probe built with it.
set -euo pipefail
cd "$(dirname "$0")/../.."

deps/fetch.sh cuda

python3 -m venv /opt/ref-venv
/opt/ref-venv/bin/pip install --no-cache-dir -r parity/requirements.txt

# Any target fixes the configuration; this one needs neither libTyrC nor CUDA.
lake -R -Kcuda=/usr/local/cuda -Kgpu=H100 build GenerateFfiHeader
/opt/ref-venv/bin/python parity/run.py --build-only
