# CI reuse and hardware qualification

Required CPU CI continues to build and run the suite manifest on Linux and
macOS. Main pushes compile project, dependencies and native code without
restoring compiled outputs. Manual CI runs default to the same clean build;
clear the `clean_build` input to exercise incremental reuse. The dependency
wheels (`deps/fetch_wheels.sh`) and the pinned Lean toolchain can be cached
in either mode.

Pull requests restore compatible `.lake/packages`, `.lake/build`, `cc/build`
and generated CUDA sources. `.github/scripts/cache_key.py` separates caches by OS,
architecture, runner image, workspace path, compiler/SDK, Lean version,
dependency manifests (including `deps/git.lock` and `deps/wheels.lock`),
LibTorch configuration, and explicit native/GPU build flags. The exact build
key also includes the source tree. An older source tree can supply an
incremental base only inside that compatibility boundary. Restoring a cache
never bypasses Lake/Make dependency checks, example typechecking, native
attention tests, or either FFI ABI audit.

Successful build outputs are saved **before** CPU tests. Re-running a failed
test job can therefore reuse its completed build while still revalidating
dependencies. `output/ci/` artifacts retain cache identity, per-suite timings,
exit codes and logs. Timing improvements must be measured in CI; cache presence
alone is not evidence of a speedup or correctness. Caches contain build products,
not runtime weights, test fixtures or source files tracked by Git.

## Spark qualification

The `GPU` workflow retains labelled-PR and GPU-path push triggers and
adds a weekly Monday schedule. Scheduled runs require both GPU and real-model
qualification; manual runs default to both, with a `real_models` input for a
GPU-only diagnostic. Missing prerequisites fail the selected qualification.
Optional pretrained-model runners are a separate tier. Four existing ASR WAV
frontend test bodies still return early on macOS and Linux CI, although LeanTest
counts them as passed. Hosted test totals therefore do not establish WAV frontend
execution; the strict Spark model phase checks the real audio path separately.

Spark's existing `spark-e626-gb10` runner belongs to `cpehle/tyr`, with labels
`self-hosted,Linux,ARM64,gpu,gb10,aarch64`. A separate upstream runner can use
`tyr-qualification` instead of generic `gpu`, so it cannot unexpectedly consume
older queued workflows. Repository registration and runner labels must match;
the hosted `select-runner` job cannot substitute one repository's runner for another.
It reports `configured` or `blocked`, never an executed qualification.

Configure these repository variables for the runner that actually serves it:

| Variable | Spark setting |
| --- | --- |
| `TYR_QUALIFICATION_RUNNER_LABELS` | JSON label array; upstream dedicated runner: `["self-hosted","Linux","ARM64","tyr-qualification","gb10"]` |
| `TYR_GPU` | `GB10` |
| `TYR_QUALIFICATION_ROOT` | optional; defaults to `$HOME/tyr-qualification` |
| `TYR_QUALIFICATION_BOOTSTRAP_PYTHON` | `python3.12` (any Python 3.12 interpreter) |
| `TYR_CUDA_HOME` | required; CUDA toolkit root on the runner, e.g. `/usr/local/cuda` |

The runner needs elan, a C++ compiler, NVCC 13.0, `flock`, and a Python 3.12
interpreter (`TYR_QUALIFICATION_BOOTSTRAP_PYTHON`). The job fetches the CUDA
variant of the pinned dependencies (`TYR_DEPS_VARIANT=cuda
deps/fetch.sh`, wheels cached in `$TYR_QUALIFICATION_ROOT/wheel-cache`):
torch `2.10.0+cu130` and its NVIDIA wheels land in `external/wheels`. The
qualification venv imports that same directory through a `.pth` file, so Lean
links against and Python imports one torch installation; `run.py` verifies they
resolve to the same directory. It does not replace the existing checkout,
Python environment or global elan default. Numerical
reference dependencies are pinned in `scripts/qualification/requirements.txt`;
the report records all resolved package versions. TorchAudio is pinned by
architecture because its official ARM64 and x86_64 wheel version strings differ.
The runtime search path prefers LibTorch's bundled libraries and a Python
wheel's sibling NVIDIA libraries before the host CUDA toolkit. This keeps
cuBLAS and cuBLASLt from different installations from being mixed; standalone
LibTorch archives do not require a sibling NVIDIA directory. Reports record
the Python runtime's loaded CUDA library paths/versions and glibc loader traces
for the actual Lean/Python child processes in each qualification command.

Both repositories must use the same qualification root. The host-wide
`gpu.lock` covers environment setup, fixture downloads, builds and execution,
so one job cannot change another's Python dependencies or partial downloads.
The job also rejects an already-active unrelated CUDA process. Other services
that launch CUDA work should honor the same lock; the startup check alone
cannot prevent a noncooperating workload from starting later. Qualification
rechecks before CUDA runtime preflight and each test command, waiting up to ten
minutes for unrelated processes to finish and recording the wait separately
from execution time. It never interrupts such workloads. Other CUDA services
still need the shared lock to guarantee exclusion throughout a test command.

The GPU phase runs the strict family-specific LeanTest suite, regenerated decode
and cache parity, the standalone native attention value/gradient tests on CUDA,
and Laguna model/cache checks with `TYR_LAGUNA_CACHE_BENCH=1`. CUDA benchmark
markers are required; CPU-only success cannot qualify Laguna. Timing output
compares cache capacities 128 and 8192 without imposing a hardware-independent
speed threshold.

`gpu-plan.json` and the GPU report record the architecture-specific codegen
inputs and production decode route. `RunMhaH100Decode` runs on every configured
GPU through the production dispatcher: H100 uses its custom decode kernel for
eligible shapes; GB10, B200 and B300 currently exercise the SDPA fallback.
The Blackwell plans do not compile the unused Hopper decode module. In
particular, GB10 cannot compile its WGMMA/tcgen05 instructions. A successful
Blackwell decode gate therefore establishes fallback/cache parity, not custom
Hopper-kernel qualification.

The real-model phase uses immutable inputs from
`scripts/qualification/fixtures.json`:

| Input | Pinned revision |
| --- | --- |
| [Qwen3-TTS 12Hz 0.6B Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base/tree/5d83992436eae1d760afd27aff78a71d676296fc) | `5d83992436eae1d760afd27aff78a71d676296fc` |
| [Qwen3-ASR 0.6B](https://huggingface.co/Qwen/Qwen3-ASR-0.6B/tree/5eb144179a02acc5e5ba31e748d22b0cf3e303b0) | `5eb144179a02acc5e5ba31e748d22b0cf3e303b0` |
| [Qwen Python reference](https://github.com/QwenLM/Qwen3-TTS/tree/022e286b98fbec7e1e916cb940cdf532cd9f488e) | `022e286b98fbec7e1e916cb940cdf532cd9f488e` |

Every required model file has its upstream LFS SHA-256 or Git blob SHA-1 recorded
and verified, including cache hits. The official Qwen `clone_2.wav` example is
pinned to SHA-256 `480f55f41c71c3d79c2a9acc48f0bfb3c5a46222e6e9ebf3e2888e93501a6b5c`.
That download uses IEEE float32 WAV, while Lean's audio reader accepts PCM.
Preparation preserves the original and creates `clone_2.pcm16.wav` with the
versioned `ieee_float32_to_pcm16_rne_v1` transformation: scale each finite sample
by 32768, round to the nearest integer with ties to even, and saturate to signed
16-bit. It preserves mono, 24000 Hz and all 193920 frames. The derived file is
pinned to SHA-256 `1fcde36ed1a9519adb27dbafa4443156068cf3b4affed5dbc1dbfa37ba2d33f2`;
Lean and Python receive this same PCM16 input. Reports retain both paths,
checksums and the transformation specification.
Downloads total approximately 4.4 GB and live in the dedicated qualification
cache, outside the candidate checkout. The model gates check tokenizer codes
against the pinned Python reference, then generated-audio energy and a nonempty
ASR transcription. The latter is a model integration smoke test, not a speech
quality or transcription-accuracy benchmark. Generation uses sampling seed zero.
Generated PCM16 WAV metadata and energy are checked with Python's standard
library; qualification does not require an optional system `ffprobe` installation.
Both gates retain Lean and Python execution logs. Strict qualification requires
CUDA device evidence from Lean; the ASR gate additionally requires a completed
Lean waveform decode and rejects the optional Python decoder fallback. Tokenizer
comparison rejects empty or malformed code matrices and invalid thresholds.

For an isolated **clean committed candidate**, after setting the variables above
and running `TYR_DEPS_VARIANT=cuda deps/fetch.sh`:

```bash
export TYR_QUALIFY_MODELS=true
mkdir -p "$TYR_QUALIFICATION_ROOT"
flock --timeout 3600 "$TYR_QUALIFICATION_ROOT/gpu.lock" bash scripts/qualification/job.sh
```

`output/qualification/` contains the source commit, runtime/package identity,
fixture manifest hash, individual logs and executed/skipped/failed counts. The
summary fails for missing reports, runtime-only preflights, zero executed tests,
skips or failures. GPU counts are actual LeanTest cases; other entries count
completed qualification gates, not individual tensors or model tokens. A fixture
download or successful build is never reported as model/GPU qualification.
