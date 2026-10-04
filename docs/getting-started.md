# Getting started

## Purpose & when to use

Tyr is a dependently-typed deep learning framework for Lean 4 that tracks tensor
shapes in the type system and executes through a C++ FFI layer (`cc/`) on top of
libtorch. This guide covers the path from a fresh checkout to a working build:
installing native dependencies, compiling with Lake, setting runtime library
paths, and running the test suites, examples, GPU parity scripts, and
distributed NanoChat training. Read this before any of the component guides.

## Architecture & main abstractions

Everything is driven by `lakefile.lean` (Lake DSL). The pieces that matter on
day one:

- **Lean libraries** — `Tyr` (`@[default_target]`, precompiled; everything under
  `Tyr.*`), `TyrCodegen` (pure-Lean GPU codegen modules, no FFI), `Tests`,
  `TestsExperimental`, and `Examples` (`lakefile.lean:560-593`).
- **C++ bridge** — `extern_lib libtyr` (`lakefile.lean:416`) wraps the Makefile
  build: it runs the GPU kernel codegen executable and then
  `make -C cc lib dylib`, producing `cc/build/libTyrC.a` (and `libTyrC.so`
  unless `TYR_BUILD_TYRC_DYLIB=0`). Every executable statically links
  `libTyrC.a` plus libtorch, OpenMP, Arrow/Parquet, and soxr via
  `commonLinkArgs` (`lakefile.lean:238`).
- **Executables** — `lean_exe` targets rooted in `Tests.*` and `Examples.*`;
  binaries land in `.lake/build/bin/`.
- **Lake scripts** (`lakefile.lean:1164-1232`) — thin wrappers that compute the
  runtime library path (`runtimeLibPath`, `lakefile.lean:1039`) and launch a
  compiled binary with it: `lake run` (test_runner), `lake run train`
  (TrainGPT), `lake run runBuiltTarget -- <ExeName> [args]`,
  `lake run buildGpuTarget -- <KernelModule> <Target>...`.

The runtime types a first program touches:

```lean
-- Tyr/Basic.lean:101
inductive Device where
  | CUDA : UInt64 → Device
  | CPU
  | MPS

-- Tyr/Basic.lean:108  -- the shape-indexed tensor type
def T (_ : Shape) : Type := TSpec.type

-- Tyr/Torch.lean:152  -- MPS > CUDA > CPU
def getBestDevice : IO Device
```

## Dependencies

The Lean toolchain is pinned in `lean-toolchain` (`leanprover/lean4:v4.29.0`);
install [elan](https://github.com/leanprover/elan) and the right nightly is
fetched automatically on the first build:

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

Native dependencies are pinned and live in `external/`; nothing else is
installed system-wide. You need a C++20 compiler (GCC with C++ (`g++`) on
Linux, Xcode command line tools on macOS), `python3`, `curl`, `unzip`, `make`,
and optionally the CUDA toolkit on Linux. On Debian/Ubuntu:
`sudo apt install g++ make curl unzip git python3`. Supported platforms are
Linux x86_64, Linux aarch64 and macOS arm64.

```bash
deps/fetch.sh                     # external/{git,wheels}/...
source scripts/ci/environment.sh  # Linux: LEAN_CC=scripts/lean_cc_wrapper.sh
```

`deps/fetch.sh` runs two fetchers. `deps/fetch_git.sh` checks out the git
repos pinned by commit in `deps/git.lock` (soxr, ThunderKittens), fetching only
that commit. `deps/fetch_wheels.sh` unpacks the wheels pinned with sha256 in
`deps/wheels.lock` (torch 2.10.0, pyarrow 25.0.1) into `external/wheels`; it
picks the CUDA variant, adding `external/wheels/nvidia`, when `nvcc` is on
`PATH` (override with `TYR_DEPS_VARIANT=cpu|cuda`). `deps/lock_wheels.py`
regenerates the wheel lock after a version bump.

Lake-level requirements are declared in `lakefile.lean`: `LeanTest`,
`LeanBenchmark`, and `LeanUrdfTypeProvider`, all pinned git revisions that Lake
fetches automatically.

## Building

```bash
lake build                  # the Tyr library (default target) + cc/build/libTyrC
lake build test_runner      # specific executables are built on demand
lake build TrainGPT TrainDiffusion TrainNanoChat FluxDemo
```

GPU-related build knobs, read by `extern_lib libtyr` (`lakefile.lean:426-435`):

| Variable | Default | Effect |
|---|---|---|
| `TYR_GPU_CODEGEN_MODULE` | `Tyr.GPU.Kernels.MhaH100` | Kernel module(s) (space-separated) to emit CUDA for |
| `TYR_SKIP_GPU_CODEGEN` | unset | `1` skips the codegen step and reuses `cc/src/generated` |
| `TYR_BUILD_TYRC_DYLIB` | unset (on) | `0` skips building `cc/build/libTyrC.so` |
| `TYR_GPU_TARGET` / `TYR_GPU_FAMILY` | auto | Forwarded to `make -C cc` as `GPU=` / `GPU_FAMILY=` |
| `TYR_MACOS_SDKROOT` / `TYR_MACOS_DEPLOYMENT_TARGET` | auto / `14.0` | macOS SDK and deployment-target overrides |

## Runtime environment

Executables find libtorch and Arrow in `external/` through their embedded
library search paths (rpath), so no `LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH` setup
is needed. The Lake scripts also set the library path themselves
(`runtimeLibPath` in `lakefile.lean`):

```bash
lake run                                   # test_runner
lake run train                             # TrainGPT
lake run runBuiltTarget -- FluxDemo        # any compiled executable, with args
```

Runtime variables a user actually sets:

| Variable | Used by | Effect |
|---|---|---|
| `TYR_DEVICE` | `Examples/TrainGPT.lean:92`, NanoChat scripts | `auto` / `cpu` / `cuda` / `mps` device selection (default CPU in TrainGPT, `cuda` in the torchrun scripts) |
| `RANK` / `LOCAL_RANK` / `WORLD_SIZE` / `MASTER_ADDR` / `MASTER_PORT` | distributed trainers | set by `torchrun`; see `scripts/nanochat/ENV_INVENTORY.md` |
| `TYR_LEAN_BIN` | `scripts/gpu/*.sh` | path to the `lean` binary when elan is not on `PATH` |

## Running tests

`test_runner` (`Tests/RunTests.lean`) is the main LeanTest suite and is marked
`@[test_driver]` (`lakefile.lean:598`), so `lake test` also works:

```bash
lake build test_runner
.lake/build/bin/test_runner                 # with the library path set, or:
lake run                                    # same thing, env handled

.lake/build/bin/test_runner --filter GPT    # only tests matching a pattern
.lake/build/bin/test_runner --ignored       # include tests marked ignored
.lake/build/bin/test_runner --fail-fast     # stop at first failure
```

The experimental suite tracks in-progress modules and is expected to be less
stable:

```bash
lake build test_runner_experimental
.lake/build/bin/test_runner_experimental
```

Focused suites exist as separate executables (`lakefile.lean:761-825`):
`TestDataLoader`, `TestDiffusion`, `TestDiffEq`, `TestDiffEqAdjoint`,
`TestGPUDSL`, `TestGPUKernels`, `TestGPUE2E`, `TestGPUGB10E2E`,
`TestGPUTileIR`, plus `RunRiemannianNanoGPTTests`. Build and run them the same
way (`lake build <name>`, then run from `.lake/build/bin/` with the library
path, or via `lake run runBuiltTarget -- <name>`).

## Running examples

Per-example details live in `Examples/README.md`; the common pattern is
`lake build <Exe>` then run with the library path set. Highlights:

| Executable | What it does | Notes |
|---|---|---|
| `TrainGPT` | char-level GPT on Shakespeare | `lake run train`; reads `data/shakespeare_char/{train,val}.bin` (in repo), falls back to random tokens; checkpoints to `checkpoints/gpt` |
| `TrainDiffusion` | discrete masked diffusion on ASCII text | flags `--generate/-g`, `--checkpoint/-c`, `--prompt/-p`, `--blocks/-n`, `--temperature/-t` |
| `TrainNanoChat` | modded-nanogpt distributed training | flags `--data`, `--val`, `--checkpoint-dir`, `--resume`, `--debug` |
| `NanoChatPipeline` / `NanoChatChat` | multi-stage pipeline / checkpoint chat | configured via env, see `scripts/nanochat/ENV_INVENTORY.md` |
| `FluxDemo` | Flux Klein 4B image generation | needs weights under `weights/`; writes `output.ppm` |
| `BranchingFlowsMoleculeTrainGenerate` | dataset-backed molecule training + generation | |

## GPU parity scripts

`scripts/gpu/` validates the ThunderKittens-style kernels against PyTorch. The
entry point builds three end-to-end checks (`scripts/gpu/test_parity_suite.sh`):

```bash
./scripts/gpu/test_parity_suite.sh          # LeanTest GPU E2E + B200 BF16 GEMM + mha_h100 seq=768
RANDOMIZED_MHA_TRIALS=10 ./scripts/gpu/test_parity_suite.sh   # add N randomized MHA trials
```

Per-kernel scripts (`test_copy_e2e.sh`, `test_rotary_e2e.sh`,
`test_layernorm_e2e.sh`, `test_flashattn_e2e.sh`, `test_mha_h100_e2e.sh`,
`test_mha_h100_768_e2e.sh`, `test_b200_bf16_gemm_e2e.sh`, ...) all delegate to
`scripts/gpu/run_e2e_kernel.sh <KernelModule> <RunnerExe> <Label>`, which runs a
six-step flow: build the codegen executable and kernel module, emit CUDA into
`cc/src/generated`, rebuild `cc/build/libTyrC.a` with `make -C cc
GPU=$TYR_GPU_TARGET GPU_FAMILY=$TYR_GPU_FAMILY`, build the runner, regenerate
fixtures, and run the parity check. Useful knobs:

- `TYR_GPU_TARGET` / `TYR_GPU_FAMILY` — override the `nvidia-smi`-based
  detection (`H100`/`GB10`/`B200`/`B300`/`A100`; `HOPPER`/`BLACKWELL`/`AMPERE`).
- `E2E_TRIALS` — repeat the fixture-regenerate + check loop N times.
- `TYR_GPU_VENDORED_REF_RUNNER` — optional vendored ThunderKittens reference
  runner, invoked as `runner <suite-name> <fixture-dir>` after each suite;
  defaults to `scripts/gpu/run_vendored_reference.sh` when executable.

These scripts source `load_modules.sh` (EasyBuild module stack: CUDA;
overridable via `TYR_CUDA_MODULE`, `TYR_NCCL_MODULE`) and
expect a CUDA toolchain (`nvcc`). They are cluster scripts — on a plain macOS or
CPU-only Linux checkout, skip this section.

## Distributed NanoChat scripts

`TrainNanoChat` runs multi-GPU under `torchrun`. The wrappers in
`scripts/nanochat/` avoid hand-managing the module stack:

```bash
# smoke run: 2 processes, --debug --iterations 2 --data data/nanochat --val data/nanochat
./scripts/nanochat/run_train_torchrun.sh

# explicit 4-GPU run
NPROC_PER_NODE=4 ./scripts/nanochat/run_train_torchrun.sh \
  --debug --iterations 2 --data data/nanochat --val data/nanochat

# scaling check over 1/2/4 GPUs; prints a throughput table
./scripts/nanochat/bench_distributed.sh
```

Knobs (all verified in the scripts):

- `TORCHRUN_BIN` — torchrun path; the default
  (`/grid/it/data/elzar/easybuild/software/Anaconda3/2023.07-2/bin/torchrun`) is
  site-specific, so override it on any other host.
- `NPROC_PER_NODE` (default 2), `SKIP_BUILD=1` (skip the `lake build
  TrainNanoChat` step), `TYR_DEVICE` (default `cuda` in these wrappers).
- `bench_distributed.sh`: `SIZES="1 2 4"`, `RUN_ARGS`, `LOG_DIR` (default
  `/tmp`); it greps for `Training complete!` and fails otherwise.
- `run_pipeline_torchrun.sh` — the multi-stage pipeline variant
  (`NPROC_PER_NODE` default 4, `PIPELINE_EXE`, `QUICK_MODE_FLAG`,
  `ENABLE_RL_FLAG`, `NANOCHAT_DIR`).
- `test_distributed_resume.sh` — checkpoint/resume smoke test
  (`CHECKPOINT_DIR`, `FRESH_ITERS`, `RESUME_ITERS`, `CUDA_VISIBLE_DEVICES`).

The full environment surface of the pipeline and trainers is inventoried in
`scripts/nanochat/ENV_INVENTORY.md`. You can also bypass the wrappers entirely:

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=8 --no_python \
  .lake/build/bin/TrainNanoChat --data data/fineweb10B --val data/fineweb_val
```

## Usage example

Reconstructed example (from `Examples/TrainGPT.lean`) — the minimal
build-and-train loop a new user runs, first in shell:

```bash
deps/fetch.sh                               # once: fetch all pinned dependencies
lake build test_runner && lake run          # sanity: test suite passes
lake build TrainGPT && lake run train       # trains, then generates from "ROMEO:"
```

and the corresponding Lean-side flow, condensed from the real `main`:

```lean
import Tyr
import Examples.GPT.GPT
import Examples.GPT.Train

open torch
open torch.gpt
open torch.train

def main : IO Unit := do
  -- device: TYR_DEVICE=auto|cpu|cuda|mps, else CPU (TrainGPT.lean:92)
  let device ← getBestDevice

  let modelCfg := Config.nanogpt_cpu_shakespeare      -- GPT.lean:42
  let trainCfg : TrainConfig := {                      -- Train.lean:15
    maxIters := 5000
    learningRate := 1e-3
    batchSize := 12
    blockSize := modelCfg.block_size
    device := device
  }

  -- data/shakespeare_char/train.bin ships in the repo (nanoGPT u16 format)
  let nTrain ← data.binFileTokenCount "data/shakespeare_char/train.bin"
  let trainData ← data.loadU16Bin nTrain "data/shakespeare_char/train.bin"

  let params ← GPTParams.init modelCfg trainCfg.device
  let opt := Optim.adamw (lr := trainCfg.learningRate) -- Optim.lean:190
  let optState := opt.init params
  let _trained ← trainLoop trainCfg params optState trainData
```

Shape mismatches in this pipeline (e.g. a wrong `blockSize` against the model
config) are compile-time errors, not runtime asserts — that is the point of the
framework.

## Related guides

- [Core tensors](core/tensors.md) — `T s`, shapes, devices, and the ops you call next
- [Data loading](data.md) — batch iterators and file formats beyond raw `.bin`
- [FFI and build internals](ffi-and-build.md) — `cc/` reference counting and link layout
- [Examples and testing](examples-and-testing.md) — full example catalog and test conventions
- [GPU kernels](gpu/kernels.md) and [GPU DSL codegen](gpu/dsl-codegen.md) — the kernel side of the parity scripts
- [Distributed training](distributed.md) — the collectives behind `TrainNanoChat`
- [LLM models](models/llms.md) — GPT/NanoChat model definitions

Exhaustive symbol-level documentation is generated separately by doc-gen4; see
`docbuild/`.
