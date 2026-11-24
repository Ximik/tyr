#!/usr/bin/env bash
# Build a checkout in the Modal build sandbox, from the repository root: the
# pinned native dependencies, every parity venv under /opt/parity-envs, a CUDA
# (H100) Lake configuration, and every parity probe built with it.
set -euo pipefail
cd "$(dirname "$0")/../.."

deps/fetch.sh cuda

python3 parity/run.py --envs /opt/parity-envs --envs-only

# Any target fixes the configuration; this one needs neither libTyrC nor CUDA.
lake -R -Kcuda=/usr/local/cuda -Kgpu=H100 build GenerateFfiHeader
python3 parity/run.py --envs /opt/parity-envs --build-only
