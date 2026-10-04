# Tyr

A deep learning framework for Lean 4 with a typed tensor facade over LibTorch.

## Overview

Use `torch.Tensor` for compile-time shape and dtype checks in new model code.
The raw `torch.T s` interface remains available for existing models and native
interop; its shape argument is an annotation and does not enforce shape equality.

```lean
import Tyr.Typed
open torch

def forward : DTensor #[32, 512] .Float32 :=
  let x := Tensor.ones #[32, 768]
  let w := Tensor.ones #[512, 768]
  x.linear w

-- Replacing w with Tensor.ones #[768, 512] is a type error.
-- Raw handles cross into this facade through Tensor.ofTensor, which checks
-- their actual shape and dtype. Tensor.assumeSpec explicitly bypasses checks.
```

## Dependencies

### Lean 4

Install [elan](https://github.com/leanprover/elan) (the Lean version manager):

```bash
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh
```

The correct Lean nightly is pinned in `lean-toolchain` and will be installed
automatically on first `lake build`.

### URDF type provider (automatic)

The URDF-backed event-skeleton example uses the `lean-urdf-typeprovider`
package, which `lakefile.lean` requires from a pinned GitHub revision
(`github.com/ranvier-labs/lean-urdf-typeprovider`). Like the other git
dependencies, Lake fetches it automatically — no extra setup is needed.

### LibTorch (required)

**macOS:**
```bash
cd external
LIBTORCH_VERSION=2.10.0
curl --fail --location --retry 5 --retry-all-errors --show-error \
  -o libtorch.zip "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-${LIBTORCH_VERSION}.zip"
unzip -tq libtorch.zip
unzip -q libtorch.zip && rm libtorch.zip
cd ..
```

Or run the helper script:
```bash
bash dependencies_macos.sh
```

**Linux (CPU):**
```bash
cd external
LIBTORCH_VERSION=2.10.0
curl --fail --location --retry 5 --retry-all-errors --show-error \
  -o libtorch.zip "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-${LIBTORCH_VERSION}%2Bcpu.zip"
unzip -tq libtorch.zip
unzip -q libtorch.zip && rm libtorch.zip
cd ..
```

**Linux (CUDA 12.6):**
```bash
cd external
curl -O https://download.pytorch.org/libtorch/nightly/cu126/libtorch-cxx11-abi-shared-with-deps-latest.zip
unzip libtorch-cxx11-abi-shared-with-deps-latest.zip && rm libtorch-cxx11-abi-shared-with-deps-latest.zip
cd ..
```

### OpenMP (required)

**macOS (Homebrew):**
```bash
brew install libomp
```

**Linux:**
OpenMP is typically included with GCC. Install if needed:
```bash
sudo apt install libomp-dev   # Debian/Ubuntu
```

### Apache Arrow & Parquet (required for data loading)

**macOS:**
```bash
brew install apache-arrow
```

**Linux:**
```bash
sudo apt install libarrow-dev libparquet-dev
```

### C++17 Compiler (required)

- macOS: Xcode command line tools (`xcode-select --install`)
- Linux: GCC 9+ or Clang 10+ (`sudo apt install build-essential`)

## Quick Start

### Building

```bash
# Build all targets with Lake
lake build

# Build specific executables
lake build test_runner
lake build TrainGPT
lake build TrainDiffusion
lake build TrainNanoChat
lake build FluxDemo
```

### Environment Setup

All executables need the native library paths set at runtime:

**macOS (Apple Silicon):**
```bash
export DYLD_LIBRARY_PATH=external/libtorch/lib:/opt/homebrew/opt/libomp/lib:/opt/homebrew/lib
```

**macOS (Intel):**
```bash
export DYLD_LIBRARY_PATH=external/libtorch/lib:/usr/local/opt/libomp/lib:/usr/local/lib
```

**Linux:**
```bash
export LD_LIBRARY_PATH=external/libtorch/lib:/usr/lib
```

Or use the Lake helper scripts which set these automatically:
```bash
lake run           # runs test_runner
lake run train     # runs TrainGPT
```

### Running Tests

```bash
lake build test_runner
.lake/build/bin/test_runner

# Or use the helper script
lake run

# Experimental/in-progress suites
lake build test_runner_experimental
.lake/build/bin/test_runner_experimental
```

## Documentation

Per-component guides live in [docs/](docs/README.md) — covering the core
tensor/FFI layer, training stack, model families, GPU kernel DSL, scientific
computing, and the C++/build internals. An exhaustive API reference can be
generated with doc-gen4 (see `docbuild/`).

## Examples

See [Examples/README.md](Examples/README.md) for detailed per-example documentation.

| Example | Description | Build target |
|---------|-------------|--------------|
| **TrainGPT** | Character-level GPT on Shakespeare | `lake build TrainGPT` |
| **TrainDiffusion** | Discrete masked diffusion on ASCII text | `lake build TrainDiffusion` |
| **TrainNanoChat** | Modded-nanogpt distributed training | `lake build TrainNanoChat` |
| **FluxDemo** | Flux Klein 4B image generation | `lake build FluxDemo` |
| **BranchingFlows** | Dataset-backed molecule branching-flow training and generation | `lake build BranchingFlowsMoleculeTrainGenerate` |
| **NanoProof** | Transformer theorem prover (model only) | Part of `Examples` lib |

### Distributed NanoChat (GPU Node)

Use the helper scripts to run `TrainNanoChat` under `torchrun` without pulling in a mismatched CUDA module stack:

```bash
# default: debug smoke run on 2 GPUs
./scripts/nanochat/run_train_torchrun.sh

# explicit 4-GPU run
NPROC_PER_NODE=4 ./scripts/nanochat/run_train_torchrun.sh \
  --debug --iterations 2 --data data/nanochat --val data/nanochat

# scaling check (1/2/4 GPUs by default)
./scripts/nanochat/bench_distributed.sh
```

Notes:
- `run_train_torchrun.sh` defaults `TORCHRUN_BIN` to `/grid/it/data/elzar/easybuild/software/Anaconda3/2023.07-2/bin/torchrun`.
- Override launcher path with `TORCHRUN_BIN=/path/to/torchrun`.
- Override process counts in the benchmark script with `SIZES="2 4"` (or any space-separated list).

### GPU Kernel Parity

The ThunderKittens-style GPU coverage now has a reusable end-to-end parity path
centered on seeded fixture generation plus hardware-backed validation:

```bash
# Run the current GPU parity suite
./scripts/gpu/test_parity_suite.sh

# Add randomized MHA trials on top of the deterministic suite
RANDOMIZED_MHA_TRIALS=10 ./scripts/gpu/test_parity_suite.sh
```

Notes:
- The suite runs three deterministic checks: the LeanTest GPU executable
  (`TestGPUE2E`) covering `copy`, `rotary`, `layernorm` (f32/bf16), `rmsnorm`
  (f32/bf16), `flashattn`, and `mha_h100`; the deterministic `mha_h100_768`
  end-to-end run; and the `b200_bf16_gemm` end-to-end run (Blackwell-only, it
  skips itself on other GPUs). `RANDOMIZED_MHA_TRIALS=N` adds N randomized
  `mha_h100` trials with freshly regenerated fixtures.
- PyTorch (via the libtorch FFI) is the default numerical oracle for fixture
  generation and parity.
- A vendored ThunderKittens reference runner is called as
  `runner <suite-name> <fixture-dir>` after each suite and should exit nonzero
  on mismatch. It defaults to `scripts/gpu/run_vendored_reference.sh` (needs a
  Python env; see `./scripts/gpu/setup_libtorch_uv.sh`) and can be overridden
  with `TYR_GPU_VENDORED_REF_RUNNER=/path/to/runner`.

## Key Concepts

### Typed and Raw Tensors

`Tensor σ` tracks shape, dtype, and a device policy in one static specification:

```lean
def project {m k n : UInt64}
    (x : DTensor #[m, k] .Float32) (w : DTensor #[k, n] .Float32) :
    DTensor #[m, n] .Float32 :=
  x.mm w
```

The underlying `T s` is a reducible alias for one opaque tensor handle type;
different shape annotations on raw tensors are definitionally equal. Use
`Tensor.ofTensor` at raw boundaries and `Tensor.validate` when auditing a
typed value. `Tensor.reshape` requires equal element counts and treats `#[]`
as a scalar. Legacy raw `reshape t #[]` retains its shape-erasure behavior;
`reshapeExact` provides real scalar reshaping at the raw level.

### TensorStruct Typeclass

Generic traversal over structures containing tensors:

```lean
class TensorStruct (α : Type) where
  map     : (∀ {s}, T s → T s) → α → α
  mapM    : (∀ {s}, T s → m (T s)) → α → m α
  zipWith : (∀ {s}, T s → T s → T s) → α → α → α
  fold    : (∀ {s}, T s → β → β) → β → α → β
```

Use `Vector n α` instead of `Array α` for type-safe `zipWith` operations.

### Data Loading

Two patterns for different use cases:

```lean
-- Fixed dimensions (raw tensor API):
let iter := SequentialBatchIterator.new loader 8 256
let (batch, iter') := iter.next  -- Returns T #[8, 256]

-- Dynamic dimensions:
let iter := BatchIterator.new shard 8 256
let (batch, iter') ← iter.next   -- Returns T #[] (erased)
```

### SafeTensors Type Provider

Tyr includes a Lean command-level type provider for SafeTensors schemas:

```lean
import Tyr.SafeTensors

open torch

-- Works with a single .safetensors file or a sharded directory.
-- If `model.safetensors.index.json` exists, introspection follows `weight_map`.
safetensors_type_provider "/path/to/model_dir_or_file" as ModelWeights

def inspectWeights : IO Unit := do
  IO.println s!"discovered tensors: {ModelWeights.tensorCount}"

  -- Per-tensor typed loader + typed schema metadata
  let tokEmbed ← ModelWeights.load_model_embed_tokens_weight
  IO.println s!"embed dtype: {ModelWeights.model_embed_tokens_weightSpec.dtype}"
  IO.println s!"embed shape: {tokEmbed.runtimeShape}"

  -- Hierarchical aggregate generated from tensor names
  let weights ← ModelWeights.loadAll
  let qProj := weights.model.layers[0]!.self_attn.q_proj.weight
  IO.println s!"q_proj shape: {qProj.runtimeShape}"

  -- Hierarchical subtree loaders are also generated
  let decoder ← ModelWeights.model.load
  IO.println s!"loaded subtree"
```

Notes:
- Tensor schema dtype metadata uses the core `torch.DType` type (not raw strings).
- For sharded checkpoints, unsafe index shard paths (absolute paths or `..` traversal) are rejected.

## FFI Reference Counting

The C++ bindings use careful reference counting. See `cc/src/tyr.cpp` header for details:

- `borrowTensor()`: Shared ownership, auto-cleanup
- `giveTensor()`: Transfer ownership to Lean
- `lean_dec()`: Required after extracting from `lean_obj_arg`, not for `b_lean_obj_arg`

Monitor tensor leaks via `get_live_tensors` which tracks outstanding C++ tensors.

## Development

### Adding New Tensor Operations

1. Add Lean declaration in `Tyr/Torch.lean` with `@[extern "lean_torch_xxx"]`
2. Implement in `cc/src/tyr.cpp` following reference counting conventions
3. Rebuild: `lake build`

### Project Structure

- `lakefile.lean` - Lake build configuration
- `lean-toolchain` - Lean version specification
- `cc/` - C++ FFI bindings (LibTorch wrapper)
- `Tyr/` - Core framework (tensors, modules, optimizers, distributed)
- `Examples/` - Training scripts and model implementations
- `Tests/` - Test suites

### Commit Template

This repo uses scoped conventional commit subjects:

```text
type(scope): summary
```

Enable the git hooks in `.githooks/` and the commit message template
`.gitmessage` for your clone:

```bash
git config core.hooksPath .githooks
git config commit.template .gitmessage
```

Included hooks:
- `pre-commit`: fails on staged whitespace errors and conflict markers
- `commit-msg`: enforces `type(scope): summary` (e.g. `feat(qwen35): add video stream patchify`)
- `pre-push`: validates the subjects of commits being pushed

All subject checks, including CI on pull requests, use one checker:
```bash
./.githooks/check-commit-message.sh "feat(qwen35): add video stream patchify"
```

## License

Licensed under the [Apache License, Version 2.0](LICENSE).
