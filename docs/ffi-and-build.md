# C++ FFI and build system

## Purpose & when to use

Tyr executes tensors through a hand-written C++ bridge in `cc/` that wraps
libtorch (`torch::Tensor`) in Lean external objects, and a `lakefile.lean` that
compiles that bridge into `cc/build/libTyrC.a` and links every Lean executable
against it plus libtorch. Read this chapter when you add a new `@[extern]`
binding, debug a link or runtime-library failure, hunt a tensor leak, or port
the build to a new platform or GPU target. Day-to-day usage of the resulting
API is covered by the tensor and autodiff guides; this chapter is about the
boundary itself.

## Architecture & main abstractions

### The tensor contract

On the Lean side a tensor is opaque and shape-indexed (`Tyr/Basic.lean:23`,
`Tyr/Basic.lean:101-108`):

```lean
abbrev Shape := Array UInt64

inductive Device where
  | CUDA : UInt64 → Device
  | CPU
  | MPS

opaque TSpec : NonemptyType
def T (_ : Shape) : Type := TSpec.type
```

At runtime, `T s` is a Lean external object whose payload is a raw
`c10::TensorImpl*`. The external class is registered with a finalizer that
decrements the intrusive pointer (`cc/src/tyr.cpp:149-161`). Three functions
govern ownership across the boundary — declared in the 15-line header
`cc/include/tyr_tensor.h`, defined in `cc/src/tyr.cpp:165-202`:

```cpp
// Borrow a tensor from Lean: incref + reclaim → scoped owning torch::Tensor.
torch::Tensor borrowTensor(b_lean_obj_arg o);       // tyr.cpp:165

// Transfer a new tensor to Lean: incref, wrap in external, count it live.
lean_object* giveTensor(torch::Tensor t);           // tyr.cpp:190

// Alias of giveTensor, kept for backward compatibility.
lean_object* fromTorchTensor(torch::Tensor t);      // tyr.cpp:200
```

The lifecycle is: C++ creates a tensor → `giveTensor` hands Lean one owning
intrusive ref; Lean passes it back → `borrowTensor` increfs and reclaims a
`torch::Tensor` that decrefs on scope exit; Lean's GC finalizes the external
object → `deleteTorchTensorFinalize` releases Lean's ref. A global
`std::atomic<int64_t> g_live_lean_tensors` (`cc/src/tyr.cpp:145`) counts
tensors currently handed to Lean and is the first thing to check when memory
grows unboundedly.

Lean-side ownership conventions, documented in the `cc/src/tyr.cpp:1-54` header
comment: `lean_obj_arg` parameters are owned — the callee must `lean_dec` them
(typical for shape arrays); `b_lean_obj_arg` parameters are borrowed — never
`lean_dec`, use `borrowTensor`. The bulk elementwise ops are generated from
macros that show the pattern (`cc/src/tyr.cpp:615-660`):

```cpp
#define BINOP_FUN(F) \
lean_object* lean_torch_tensor_##F(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) { \
  auto a_ = borrowTensor(a); \
  auto b_ = borrowTensor(b); \
  auto c_ = torch::F(a_, b_); \
  lean_dec(s); \
  return fromTorchTensor(c_); \
}
BINOP_FUN(add) BINOP_FUN(sub) BINOP_FUN(mul)
```

`getDevice` (`cc/src/tyr.cpp:212`) maps the `Device` constructors by tag —
0 = `CUDA idx`, 1 = `CPU`, 2 = `MPS` (with a `std::call_once` MPS warmup on
Apple; falls back to CPU off Apple).

### Error strategy

Pure ops are unguarded: shapes are checked by Lean's type system before
crossing the FFI, so a global `std::terminate` handler (`cc/src/tyr.cpp:267-301`)
prints the pending `c10::Error`'s user-facing text and aborts, instead of
dumping libtorch's backtrace. Set `TYR_VERBOSE_ERRORS=1` for the full report.
IO-flavored entry points (safetensors, parquet, process groups) catch and
return `lean_io_result_mk_error` values instead. Two sites carry workarounds
for a libc++abi/libstdc++ exception-ABI mismatch (Lean's toolchain vs.
libtorch's): `lean_torch_manual_seed` seeds only the CPU generator, and CUDA
synchronization calls `cudaDeviceSynchronize` via cudart directly
(`cc/src/tyr.cpp:311-352`). `Tests/FfiCrashProbe.lean` is a manual probe for
this failure mode (own `ffi_crash_probe` exe, not part of the test suite).

### Translation units in `cc/src`

| File | Exports | Bound from |
|---|---|---|
| `tyr.cpp` (5236 lines) | ~235 `lean_torch_*` entry points: creation, elementwise, reductions, shape ops, linalg, norms, autograd, safetensors, sampling, soxr resampling | mostly `Tyr/Torch.lean` (245 `@[extern]`) |
| `tyr_ops.cpp` | flash-attention router `tyr_ops::flash_attn_dispatch` + `lean_torch_tyr_flash_attn_4d` | `Tyr/Torch.lean:1197` |
| `tyr_distributed.cpp` | 15 `lean_torch_dist_*` c10d wrappers (TCPStore, NCCL/Gloo, async work handles) | `Tyr/Distributed.lean` |
| `tyr_polar.cpp` | `lean_torch_xxt`, `ba_plus_cAA`, `newton_schulz_step`, `polar_express`, `muon_orthogonalize`, `cautious_update` | manifold/optimizer modules |
| `tyr_parquet.cpp` | 6 `lean_parquet_*` Arrow/Parquet readers, `__has_include`-gated | `Tyr/Data/Pretraining.lean:48-73` |
| `tyr_execution.cpp` | `lean_exec_python_sandboxed`, `lean_exec_shell_sandboxed` (fork + rlimit sandbox) | `Examples/NanoChat/Eval/Execution.lean:82` |
| `tyr_qwen35.cpp` | `lean_torch_qwen35_{chunk,recurrent}_gated_delta_rule` linear attention | `Tyr/Model/Qwen35/Model.lean:50,60` |
| `float_buffer.cpp` | `lean_float_buffer_*` streaming-audio ring buffer | `Tyr/Audio/FloatBuffer.lean` |
| `apple_*.mm` (Darwin only) | MPS probing, VL image/video preprocessing, mic I/O | `Tyr/Audio/Apple*.lean` |
| `tyr_media_stub.cpp` | non-Apple error stubs for the media entry points | same |
| `tk_vendor_mha_h100.cu` / `tk_vendor_stubs.cpp` | vendored ThunderKittens Hopper MHA kernels, or throwing stubs | `tyr_ops.cpp` |
| `generated/tyr_gpu_kernel_stubs.cpp` | weak `lean_launch_Tyr_GPU_Kernels_*` launchers (generated) | GPU codegen |

In total the tree exports ~280 `extern "C"` entry points; the Lean side has
310 `@[extern]` declarations under `Tyr/`. Only `tyr_qwen35.cpp` includes
`cc/include/tyr_tensor.h`; the other units forward-declare the three helpers
themselves. `cc/lxla.cpp` (0 bytes) and `cc/plugin_init.cpp` are dead files,
referenced by neither the Makefile nor the lakefile.

### The GPU kernel linking trick

Generated CUDA kernels export `lean_launch_Tyr_GPU_Kernels_*` symbols, but the
generator itself is a Lean executable that must link `libTyrC.a` — a
chicken-and-egg problem. The Makefile breaks it by always archiving
`generated/tyr_gpu_kernel_stubs.o`, which defines every launcher as
`__attribute__((weak))` throwing stubs, **last** in the archive
(`cc/Makefile:290-297, 310-320`): strong definitions from real `.cu` files win
when present, and links succeed when they are not. The stubs are regenerated
by `cc/tools/generate_gpu_kernel_stubs.py`, which scans `.c.o.export` IR and
`@[gpu_kernel]` sources for launcher names. The inventory is refreshed before
Lake checks the native build trace and on every Make invocation that needs
stubs. It preserves the output timestamp when symbols are unchanged. Source
declarations take precedence over stale IR when a kernel is removed; nested
namespaces and comments are handled during source scanning.
`tyr_ops.cpp` calls launchers only
after a runtime Hopper check (`device_supports_tk_hopper`,
`cc/src/tyr_ops.cpp:39-48`), routing everything else to portable SDPA.

### Build orchestration in `lakefile.lean`

`extern_lib libtyr` in `lakefile.lean` is the hub. It:

1. Writes `.lake/build/libtyr_gpu_codegen.env` recording
   `TYR_GPU_CODEGEN_MODULE` / `TYR_SKIP_GPU_CODEGEN` / `TYR_BUILD_TYRC_DYLIB`,
   so changing any of them invalidates the native build.
2. Runs `make -s -C cc native-config gpu-stubs` to refresh content-stable
   native configuration and symbol inventories. `cc/build/native-build.json`
   records effective compilers, flags, GPU target/family/compute/code, library
   paths, and selected objects, including values discovered by Make.
3. Mixes Lake jobs over those manifests, `cc/Makefile`, `cc/src`,
   `cc/include`, `cc/tools`, and the active kernel module's `.c.o.export` IR.
   Compiler-generated `.d` files also feed `native-dependencies.txt`, whose
   existing paths are tracked by Lake, including vendor headers.
4. Runs `lake build GenerateGpuKernels` and executes it into
   `cc/src/generated/`, unless `TYR_SKIP_GPU_CODEGEN=1`, or it is unset and
   `native-build.json` reports `HAS_NVCC=0` (without `CUDA_HOME` the Makefile ignores
   generated `.cu` files and links the weak launcher stubs, so codegen would be
   wasted). `TYR_SKIP_GPU_CODEGEN=0` forces codegen.
5. Runs `make -jN -C cc lib [dylib]` (`N` = `TYR_MAKE_JOBS`, which `env.sh` sets to the CPU count; no `-j` if unset) with `gpuMakeEnv` forwarding
   `GPU`/`GPU_FAMILY`/`GPU_COMPUTE`/`GPU_CODE`.

Make uses compiler dependency files for C, C++, Objective-C++, and CUDA.
Header changes rebuild their consumers; native configuration changes rebuild
the selected objects and relink the archives/libraries. No-op manifest refreshes
leave existing object timestamps unchanged. The inexpensive regression command
`python3 scripts/test_native_build.py` exercises these rules with tiny sources
in a temporary checkout: header edits, architecture switches, renamed headers,
clean/incremental output agreement, and launcher additions/removals. It does
not compile LibTorch or CUDA and does not modify workspace build artifacts.

The Makefile takes every native dependency from `external/` — libtorch
(`external/wheels/torch`), soxr (built from the `external/git/soxr` checkout without
cmake), Arrow/Parquet (`external/wheels/pyarrow`, linked by exact file name via
`ARROW_SOVERSION`), and the CUDA runtime and NCCL (`external/wheels/nvidia`, CUDA
variant only); `deps/fetch.sh` populates them. It probes only Lean
(`lean --print-prefix`) and the CUDA toolkit (`nvcc`). GPU targets are a matrix
(`cc/Makefile:127-196`): `GPU` selects SASS, `GPU_FAMILY` selects the
ThunderKittens arch guards —

| `GPU` | family | compute / sm |
|---|---|---|
| `A100` | AMPERE | `compute_80a` / `sm_80a` |
| `H100` (default) | HOPPER | `compute_90a` / `sm_90a` |
| `B200` | BLACKWELL | `compute_100a` / `sm_100a` |
| `B300` | BLACKWELL | `compute_103a` / `sm_103a` |
| `GB10` | HOPPER | `compute_121` / `sm_121` |

`tk_vendor_mha_h100.cu` (which emits sm_90a-only `wgmma`) is compiled only when
`GPU=H100`; other targets get `tk_vendor_stubs.cpp` instead.

Link arguments are computed in Lean and probed at runtime, never assumed:
`packageLinkArgs` (`lakefile.lean:220`) is the package-wide `moreLinkArgs`, and
`commonLinkArgs` (`lakefile.lean:238`) is the per-executable version that
prepends the absolute path to `cc/build/libTyrC.a`. They assemble from
`linuxLinkTail`, `linuxCudaLinkArgs`, `linuxCudaDriverStubLinkArgs`,
`linuxGlibc234CompatLinkArgs` (defines `__libc_csu_init/fini=0` for glibc ≥
2.34), `linuxCompilerLibDirArgs` (the system gcc's libstdc++ dir), `arrowLinkArgs`, `macOSSDKLinkArgs`, `macOSDeploymentLinkArgs`,
`macOSFrameworkArgs`, and `soxrLinkArgs`. If the vendored libtorch has no CUDA,
`linuxCudaLinkArgs` returns `#[]` and a CPU-only checkout still links. The five
`lean_lib`s are `TyrCodegen` (pure-Lean GPU codegen, `precompileModules := false`
to avoid the `.so` cascade), `Tyr` (default target, precompiled), `Tests`,
`TestsExperimental`, `Examples` (`lakefile.lean:560-593`), plus 64 `lean_exe`
targets that each take `moreLinkArgs := commonLinkArgs`.

### Running executables

Executables land in `.lake/build/bin/` and find libtorch and Arrow through
rpaths relative to the binary (`$ORIGIN/...` on Linux, `@loader_path/...` on
macOS), so the checkout can move as long as `external/` moves with it.
The eight `lake run` scripts (`lakefile.lean:1166-1232`) all go through
`runBuiltExecutable` (`lakefile.lean:1076`): it assembles the path via
`runtimeLibPath` (`lakefile.lean:1039`), validates the binary with `file`,
checks staleness against `.c.o.export` IR, and — if the binary is broken or
stale — relinks it into `/tmp/tyr_relinked` by replaying the link command
extracted from Lake's `.trace` file (`relinkBuiltExecutableToTmp`,
`lakefile.lean:364-383`).

| Script | What it does |
|---|---|
| `lake run` | run `test_runner` |
| `lake run train` | run `TrainGPT` |
| `lake run runBuiltTarget -- <Exe> [args]` | run any compiled exe |
| `lake run buildGpuTarget -- <KernelModule> <Target>...` | build exe(s) with one GPU kernel module |
| `lake run buildMhaH100Examples` | build `RunMhaH100` + `RunMhaH100Seq768` |
| `lake run runMhaH100Exe` / `runMhaH100Seq768Exe` | run those with lib paths set |
| `lake run validateMhaH100Examples` | build both, then run back-to-back |

### Environment variables

Build behavior is controlled entirely through the environment:

| Variable | Effect |
|---|---|
| `TYR_GPU_CODEGEN_MODULE` | kernel module(s) to emit CUDA for (space-separated; default `Tyr.GPU.Kernels.MhaH100`) |
| `TYR_SKIP_GPU_CODEGEN` | `1` skips the generator step in `extern_lib libtyr`, `0` forces it; unset skips it only when `nvcc` is missing |
| `TYR_MAKE_JOBS` | parallel jobs for the native `make` build; `source ./env.sh` sets it to the CPU count if unset (unset: serial) |
| `TYR_BUILD_TYRC_DYLIB=0` | build only `libTyrC.a`, skip `libTyrC.so/.dylib` |
| `GPU` (or `TYR_GPU_TARGET`), `GPU_FAMILY`, `GPU_COMPUTE`, `GPU_CODE` | override the Makefile GPU matrix |
| `TYR_MACOS_SDKROOT`, `TYR_MACOS_DEPLOYMENT_TARGET` | macOS SDK/deployment overrides |
| `CUDA_HOME` | the only CUDA switch (set by `env.sh` from `nvcc`); must contain `bin/nvcc` |
| `NCCL_ROOT` | NCCL discovery hint |
| `LEAN_CC_FAST=1` | `-O0` for Lean-generated C (fast local iteration) |
| `LEAN_CC_GCC`, `LEAN_CC_LINKER` | compiler/linker selection in the wrapper |

`lean-cc` (repository root) is the `LEAN_CC` wrapper used on Linux (set by
`env.sh`). Lean's bundled clang links against the old glibc
inside the Lean toolchain, while `cc/` and libtorch are built with the system
gcc against the system glibc/libstdc++, so the link fails with undefined glibc
symbols. The wrapper runs the system gcc instead, maps Lean's `-lc++`,
`-lc++abi`, `-lgmp` and `-luv` to the toolchain's static archives, and links
with the toolchain's lld. Lake reads the compiler only from `LEAN_CC`; there is
no lakefile option for it.

### `scripts/` overview

- `scripts/gpu/` — ~20 shell harnesses for H100/GB10/B200 E2E parity and
  benchmarks (`test_*_e2e.sh`, `bench_flash_attn_matrix.sh`), plus
  `setup_python_venv.sh` and a vendored-PyTorch reference runner.
- `scripts/runpod/` — disposable-pod workflow (`create_or_resume.sh`,
  `sync_repo.sh`, `bootstrap.sh`, `run_bench.sh`); see its `README.md`. Pods
  are disposable, repo and network volume are the durable state, and
  `RUNPOD_API_KEY` never enters tracked files.
- `scripts/nanochat/` — `torchrun` launchers for distributed NanoChat training
  (`run_train_torchrun.sh`, `bench_distributed.sh`) and `ENV_INVENTORY.md`.
- Conventional-commit hooks and their checker live in `.githooks/`.
- Python converters — `kokoro_to_safetensors.py`,
  `qm9_{sdf,xyz}_to_branching_jsonl.py`, `qwen3tts_*.py`,
  `kittentts_reference_synthesize.py` (dataset prep and parity references;
  run them in the repo's `.venv`, not a global Python).

## Key APIs

C++ side, what you touch when adding a binding (`cc/src/tyr.cpp`):

| Function | Purpose |
|---|---|
| `borrowTensor(b_lean_obj_arg)` :165 | borrowed Lean tensor → owning `torch::Tensor` |
| `giveTensor(torch::Tensor)` :190 | new tensor → owned Lean external (counts live) |
| `fromTorchTensor` :200 | alias of `giveTensor` |
| `getShape(b_lean_obj_arg)` :204 | Lean `Shape` array → `std::vector<int64_t>` |
| `getDevice(b_lean_obj_arg)` :212 | Lean `Device` → `torch::Device` |
| `mkIoUserError` / `mkC10IoError` :234-244 | wrap errors as `IO.Error` results |
| `lean_torch_get_live_tensors` :307 | leak gauge, bound as `torch.get_live_tensors` |

Lean side (`Tyr/Torch.lean:1060-1065`):

```lean
@[extern "lean_torch_get_live_tensors"]
opaque get_live_tensors : IO UInt64

@[extern "lean_torch_manual_seed"]
opaque manual_seed (seed : UInt64) : IO Unit
```

Makefile targets: `all` (default; static and shared libraries), `lib`, `dylib`,
`native-config`, `gpu-stubs`,
`bench-flash-attn` (standalone C++ attention benchmark from
`cc/tools/bench_flash_attn.cpp`), `soxr`, `clean`. `check-deps` runs first and
fails early, pointing at `deps/fetch.sh`, when `external/git/soxr`,
`external/git/ThunderKittens`, `external/wheels/torch` or the Arrow libraries are
missing.

## Usage example

Reconstructed example (from `Tests/FfiCrashProbe.lean`, `Tyr/Torch.lean:44-51,
984-989`, `Tyr/Distributed.lean:61-119`):

```lean
import Tyr.Torch
import Tyr.Distributed

open torch

def main : IO Unit := do
  -- Pure ops on the Lean side; shapes are compile-time checked.
  let x : T #[2, 2] := torch.ones #[2, 2]
  let y := torch.add x x                -- → lean_torch_tensor_add
  let z ← torch.randn #[2, 2]           -- IO: randomness

  -- Device transfer; on a CUDA-less machine this aborts via the
  -- terminate handler with a trimmed c10::Error message.
  let zg := z.to (.CUDA 0)

  -- Leak gauge: count of tensors currently handed to Lean.
  IO.println s!"live tensors: {← torch.get_live_tensors}"

  -- Safetensors handle (external-class object with a per-file cache).
  let h ← safetensors.openHandle "model.safetensors"
  let w ← safetensors.loadFromHandle h "layer.weight" #[768, 768]

  -- Distributed, launched via torchrun (see scripts/nanochat/).
  torch.dist.initProcessGroup "nccl" "127.0.0.1" 29500 rank worldSize
  torch.dist.allReduce y .sum
  torch.dist.destroyProcessGroup
```

Build and run through Lake so the native library and runtime paths are right:

```bash
lake build                                  # extern_lib libtyr → codegen → make -C cc lib dylib
lake run runBuiltTarget -- TrainGPT         # sets DYLD/LD_LIBRARY_PATH for you
# GPU build for a specific kernel module and target:
TYR_GPU_CODEGEN_MODULE=Tyr.GPU.Kernels.MhaH100 GPU=H100 lake build RunMhaH100
# Manual probe of the FFI failure mode:
lake build ffi_crash_probe && lake run runBuiltTarget -- ffi_crash_probe
```

## Related guides

- [Getting started](getting-started.md) — the day-one build path this chapter explains
- [Core tensors](core/tensors.md) — the `T s` API surface the FFI implements
- [Autodiff](autodiff.md) — gradient bindings (`lean_torch_backward` and friends)
- [Serialization](serialization.md) — safetensors handles and sharded loading
- [Distributed training](distributed.md) — the `torch.dist` collectives over c10d
- [GPU DSL codegen](gpu/dsl-codegen.md) — where the generated `.cu` kernels come from
- [GPU kernels](gpu/kernels.md) — the `@[gpu_kernel]` side of the launcher stubs
- [Data loading](data.md) — parquet readers and other data-path bindings
- [Examples and testing](examples-and-testing.md) — the executables these link args serve

Exhaustive symbol-level documentation is generated separately by doc-gen4; see
`docbuild/`.
