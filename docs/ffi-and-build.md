# C++ FFI and build system

## Purpose & when to use

Tyr executes tensors through a hand-written C++ bridge in `cc/` that wraps
libtorch (`torch::Tensor`) in Lean external objects, and a `lakefile.lean` that
compiles that bridge into `cc/build/libTyrC.so` and links every Lean executable
against it plus libtorch. Read this chapter when you add a new `@[extern]`
binding, debug a link or runtime-library failure, hunt a tensor leak, or port
the build to a new platform or GPU target. Day-to-day usage of the resulting
API is covered by the tensor and autodiff guides; this chapter is about the
boundary itself.

## Architecture & main abstractions

### The tensor contract

The native carrier is opaque. Public tensors index it with a specification
(`Tyr.Basic.Types` and `Tyr.Basic`):

```lean
abbrev Shape := Array UInt64

inductive Device where
  | CUDA : UInt64 → Device
  | CPU
  | MPS

opaque TSpec : NonemptyType
def RawTensor : Type := TSpec.type
```

`Tensor σ` uses a private opaque carrier indexed by `StaticSpec`. It has no
public constructor or structural recursor, and different indices are not
definitionally interchangeable. `Tensor.raw` exposes the unindexed handle;
`Tensor.ofRaw` admits it only after checking native shape, dtype, and concrete
device. The reserved dtype `Unknown "dynamic"` weakens only the dtype claim;
every indexed tensor names its actual device. An arbitrary unknown dtype token
does not bypass the check. `zerosFor` accepts only a specification and supplies the native zero
factory used by `Inhabited`.

Runtime representation casts are private **unsafe** implementations behind
opaque public declarations. Their safety does not depend on name privacy:
safe Lean code cannot use them even if it discovers their private names.
`Tests/TestTensorConstruction.lean` checks attempted construction, record
updates, index conversion, tactic-based construction, and private implementation
access, alongside checked native admission. These are runtime metadata checks,
not kernel proofs about LibTorch's implementation.

At runtime, `RawTensor` is a Lean external object whose payload is a raw
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
tensors currently handed to Lean. This counter does not include leaked Lean
shape arrays, options, or other arguments; their ownership needs separate checks.

Lean-side ownership conventions, documented in the `cc/src/tyr.cpp:1-54` header
comment: `lean_obj_arg` parameters are owned — the callee must `lean_dec` them
(typical for shape arrays); `b_lean_obj_arg` parameters are borrowed — never
`lean_dec`, use `borrowTensor`. Release unused owned arguments before any
throwable work. For arguments that must be parsed, use scoped ownership so
exceptions and early returns release them too. The bulk elementwise macros
release their unused shape before invoking libtorch:

```cpp
#define BINOP_FUN(F) \
lean_object* lean_torch_tensor_##F(lean_obj_arg s, b_lean_obj_arg a, b_lean_obj_arg b) { \
  lean_dec(s); \
  auto a_ = borrowTensor(a); \
  auto b_ = borrowTensor(b); \
  auto c_ = torch::F(a_, b_); \
  return fromTorchTensor(c_); \
}
BINOP_FUN(add) BINOP_FUN(sub) BINOP_FUN(mul)
```

`getDevice` (`cc/src/tyr.cpp:212`) maps the `Device` constructors by tag —
0 = `CUDA idx`, 1 = `CPU`, 2 = `MPS` (with a `std::call_once` MPS warmup on
Apple). Unsupported MPS requests and CUDA indices outside the native range
are rejected.

`scripts/test_native_tensor_ownership.sh` checks reference counts across
successful calls and native exception paths. It also checks that borrowed
tensor handles remain valid. The ABI checker cannot distinguish owned and
borrowed arguments: both appear as `lean_object*` in generated C.

FloatBuffer has its own audited header, `cc/src/tyr_float_buffer_abi.h`.
Its `Nat` inputs are boxed Lean objects, its pure `Float` result is a native
`double`, and its IO results use Lean's result wrapper. CI compares every
recognized declaration in the generated FloatBuffer adapter with the header
using `scripts/check_ffi_abi.py --require-complete`.

### Consuming tensor updates

Value updates receive their destination as an owned `lean_obj_arg`, acquire a
native owning handle, and release the consumed Lean reference before inspecting
native ownership. A surviving Lean alias keeps its external wrapper and that
wrapper's native reference alive. Native uniqueness can therefore rule out
surviving Lean aliases after this handoff without synchronizing the two counters.
Borrowed sources continue to use `borrowTensor` and must not be decremented.

TensorImpl uniqueness alone does not establish exclusive storage ownership.
The reuse path also checks storage and version ownership and excludes views,
autograd metadata, weak observers, foreign storage, and accelerator storage.
Only ordinary CPU allocations satisfying these checks are reused. Other inputs
are copied before writing. Every value update returns its resulting tensor;
callers must retain that result. The typed adapter captures dtype and device
before the call and validates the result with `Tensor.checkedUpdate?`, without
keeping the original tensor as a fallback.

Gradient accumulation remains explicit mutable state: backward, clipping, and
`zeroGrad` are IO operations, and `gradOf` returns an independent snapshot in IO.
`setRequiresGrad` is also effectful. Pure constant factories create tensors
without gradient identity; use `Tensor.parameter` or
`TensorStruct.makeLeafParams` to create independent trainable leaves. Repeating
an equal pure allocation may share its result, so pure factories do not promise
fresh storage. `data.cloneInferenceIO` supplies an independent detached copy
when storage freshness is required.

Pure forward computations may also be shared. The public backward operations
retain their graph so evaluating backward does not invalidate a surviving
forward value. Graph storage remains alive while that value is reachable;
release unused outputs and use detached inputs for inference to avoid retaining
unneeded graphs. `Tests/TestAutogradIdentity.lean` covers independent parameter
initialization and repeated backward on the same forward value.

Pure attention exports always pass literal zero dropout to native SDPA and do
not accept a dropout probability. Their `?` adapters return validation and
kernel errors through `Except String`; convenience wrappers use the usual
panic fallback. Configurable attention dropout uses separate native IO exports
and typed `attentionIO`, `attentionGQAIO`, `attentionGQAMaskIO`,
`attentionGQAQKVIO`, `attentionWindowIO`, and `flashAttentionIO` adapters.
These take required dropout and training arguments after the tensor operands.
Evaluation uses zero dropout; training retains the configured probability.
The IO adapters validate the original probability as finite and in `[0, 1]`
before selecting either mode, so invalid configuration also fails in evaluation.
An IO caller must bind that operation rather than wrap a pure stochastic call
in `pure`. The sampled mask stays in the native autograd graph for backward.
The GPU wrappers and GPT example follow this split; GPT evaluation disables
attention dropout as well as its residual and embedding dropout.
Zero dropout removes this RNG dependency, without promising cross-device or
cross-kernel bitwise reproducibility.

Inference uses explicit values, without a thread-local gradient-mode scope:

```lean
let inferenceModel := TensorStruct.detach model
let inferenceInput := Tensor.detach input
let output := forward inferenceModel inferenceInput
```

Detach every floating input that can carry gradients, including cached
activations and auxiliary embeddings. Detaching only the output still constructs
the forward graph. Generic callbacks must receive detached model parameters
explicitly; a wrapper cannot detach parameters hidden in a closure. The former
public `no_grad` / `noGrad` scopes and ambient mode setters are removed because
Lean may move or share pure expressions across IO actions. Native guards remain
local to specific operations such as gradient snapshots and cache copies, where
the operation's semantics explicitly require detached results.

`scripts/test_native_tensor_values.sh` checks actual exported operations against
Lean aliases, native aliases, storage views, weak observers, and saved autograd
inputs. It also verifies unique CPU reuse and gradient snapshot stability.
The Lean regressions cover returned-value use, metadata checks, and effect order.

The BranchingFlows molecule training CUDA-graph option currently uses its eager
training path with a notice: replay requires stable mutable buffer addresses,
which returned-value updates do not promise. A future graph implementation needs
an explicit mutable-buffer API. GPU benchmark kernels still require separate
hardware qualification.

### Error strategy

Native result carriers must match the Lean declaration on every path.
`scripts/check_ffi_recovery.py` inventories recoverable declarations in
`Tyr/RawTensor.lean`, its submodules, `Tyr/Basic.lean`, `Tyr/Torch.lean`, and the Parquet FFI in
`Tyr/Data/Pretraining.lean`. It requires every native implementation, including
inactive platform branches, to enter the common exception guard before doing
C++ work. The CPU tooling job runs this check alongside ABI and runtime
regressions; reviewed fixed platform-unavailable stubs are checked separately.
Both Arrow-enabled and Arrow-disabled Parquet implementations use the guard,
including error-message construction. Other native subsystems remain outside
this source gate's reviewed scope.
`RawTensor.flashAttention` and `RawTensor.transpose` return
`Except String RawTensor`; their typed `Tensor.flashAttention?` and
`Tensor.transpose?` adapters preserve recoverable failures and check successful
results. The convenience variants use Lean's panic fallback on errors.
`Tensor.randint` reports invalid bounds and device failures through `IO.Error`.
The native `randn`/`rand` factories and dropout also catch failures at their IO boundary;
those random factories enable requested gradients after device transfer so their
results remain leaf parameters.
`scripts/test_native_tensor_errors.sh` exercises success and failure carriers,
including borrowed-reference preservation and cleanup on invalid input.

Core arithmetic and matrix adapters use checked native exports returning
`Except String RawTensor`. Typed `Tensor.Promoting` and `Tensor.Preserving`
arithmetic `?` APIs preserve those errors, including missing kernels for otherwise
valid promotion pairs. Fused `affine3d?` checks the actual dtype and device of
both its weight and bias before calling the checked native kernel. Scalar
spectral and nuclear norms also expose checked `?` variants for dtype, backend,
and factorization failures. The checked expansion, concatenation, and reduction
families follow the same result-carrier contract. Their pure convenience
wrappers use Lean's panic fallback after receiving an error.

Selection has separate static and runtime entry points. `argmax`, `maxLast3d`,
and `topKLast2d` require ordered dtypes and the appropriate axis, nonempty-axis,
or top-k bound proofs. Their `?` variants check actual metadata and retain native
errors; the `*Checked` conveniences perform those checks with a panic fallback.
`crossEntropy` and `crossEntropyNone` require real floating logits and Int64
targets. Their `?` variants use checked native exports that validate target
labels against the class range, allowing the default ignored label `-100`,
before dispatching the loss kernel. Dynamic logits can use the corresponding
`*Checked` conveniences. The four convolution adapters `conv1d?`, `conv2dBias?`,
`conv1dGroups?`, and `convTranspose1d?` validate actual dtype/device equality,
retain native argument and kernel errors, and check successful result metadata.
These contracts do not extend to every raw selection, loss, or convolution export.

Checked vector indexing validates index bounds before accelerator dispatch,
so invalid token IDs do not reach a device-side indexing assertion. Token-byte
totals use this boundary together with checked transfer, arithmetic, reduction,
and scalar extraction. Equal-rate audio resampling retains the borrowed input
array before returning it, preserving ownership of its boxed Float values.
The resampler computes its output length with checked integer arithmetic before
narrowing it or allocating output storage, and its native body returns C++ failures through
`IO.Error`. The runtime-service exports in `Tyr.Torch` use the same exception
guard and have a complete ABI header, `cc/src/tyr_torch_abi.h`, checked against
generated `Tyr/Torch.c`. `scripts/test_native_torch_recovery.sh` exercises the
resampler size failures, borrowed arrays, and runtime error carriers. This
exception guard does not establish CUDA graph lifecycle or stream correctness;
those GPU-specific contracts still require repair and qualification.

STFT preserves its IO error contract even for invalid native FFT parameters.
Apple image and video preprocessing catches C++ failures at each IO entry point;
temporary native media resources are released during unwinding.
IO initializers use checked `zerosIO`, `onesIO`, or `eyeForIO` allocation and
conversion, so invalid dimensions or placement do not escape through a pure
factory before parameter creation.

These adapters check operation preconditions before native execution. Equal
concrete device indices tie tensors to the same placement; checked FFI adapters
validate native metadata as well. Binary arithmetic
permits CPU rank-zero operands on another device; matrix multiplication and
concatenation require equal actual devices. Selection validates its separate
native scalar relocation rule. Metadata checks and static witnesses describe
valid inputs and result claims; they do not prove backend kernel availability
or the C++ implementation itself. The [typed guide](core/typed.md#device-compatibility)
describes these contracts and their tests.

Many pure ops are unguarded. Type indices do not validate native execution;
typed adapters check metadata at selected boundaries. A global
`std::terminate` handler (`cc/src/tyr.cpp:267-301`)
prints the pending `c10::Error`'s user-facing text and aborts, instead of
dumping libtorch's backtrace. Set `TYR_VERBOSE_ERRORS=1` for the full report.
IO entry points such as safetensors, parquet, and process groups catch and
return `lean_io_result_mk_error` values instead. A native exception must be
caught before crossing a generated Lean call frame: Lean's reference-count
cleanup does not run during C++ stack unwinding. Two sites carry workarounds
for a libc++abi/libstdc++ exception-ABI mismatch (Lean's toolchain vs.
libtorch's): `lean_torch_manual_seed` seeds only the CPU generator, and CUDA
synchronization calls `cudaDeviceSynchronize` via cudart directly
(`cc/src/tyr.cpp:311-352`). `Tests/FfiCrashProbe.lean` is a manual probe for
this failure mode (own `ffi_crash_probe` exe, not part of the test suite).

`TensorStruct.functionalGrad` invokes a pure Lean callback with temporary
autograd leaves and returns detached derivatives. Setup and gradient-engine
failures return the raw API's `Except.error`; the typed convenience API uses a
panic fallback. The callback must return normally. A native exception from an
unguarded pure operation inside a compiled callback terminates with an explicit
diagnostic, because returning after unwinding generated Lean frames would leave
their owned references unreleased. Interpreter callback exceptions are also
unrecoverable. This API does not make arbitrary pure native failures recoverable.
`scripts/test_functional_grad_callback.sh` checks normal derivatives, recoverable
setup/result errors without leaks, and fatal callback errors in separate processes.

### Translation units in `cc/src`

| File | Exports | Bound from |
|---|---|---|
| `tyr.cpp` (5236 lines) | ~235 `lean_torch_*` entry points: creation, elementwise, reductions, shape ops, linalg, norms, autograd, safetensors, sampling, soxr resampling | `Tyr.RawTensor` and its adapter submodules |
| `tyr_ops.cpp` | flash-attention router `tyr_ops::flash_attn_dispatch` + `lean_torch_tyr_flash_attn_4d` | `Tyr.RawTensor.Neural` |
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

### Kernel launcher stubs

Every Lean GPU kernel declares a launcher, `@[extern "lean_launch_…"]`, which
Lean code calls. Its real definition exists only when the kernel's generated
CUDA is compiled, i.e. in a CUDA build and for the `-Kkernels` modules.
Everywhere else, `libTyrC` provides a placeholder: the Lean build runs
`GenerateGpuKernels --stubs` over every module of `TyrKernels`, which writes
`cc/src/generated/tyr_gpu_kernel_stubs.cpp` with an `__attribute__((weak))`
stub per launcher that returns a "kernel unavailable" error. A compiled
kernel's strong definition replaces its stub. The launcher names come from the
kernels' `@[extern]` declarations, and the file is rewritten only when they
change. `tyr_ops.cpp` calls launchers only after a runtime Hopper check (`device_supports_tk_hopper`,
`cc/src/tyr_ops.cpp:39-48`), routing everything else to portable SDPA.

### Build orchestration in `lakefile.lean`

`target libtyr` in `lakefile.lean` is the hub. It:

0. Checks the configured CUDA choice against the fetched wheels. The choice is
   made once with `lake -R -Kcuda=<toolkit>` (CUDA) or plain `lake -R` (CPU);
   Lake keeps it until the next `lake -R`, and the lakefile reads it as
   `cudaHome?`. If it disagrees with `external/wheels/torch` (fetched by
   `deps/fetch.sh cpu|cuda`), or the toolkit has no `bin/nvcc`, the build stops
   with the command to run. `-Kgpu` is required with `-Kcuda` and rejected
   without it. It then passes `CUDA_HOME` and `GPU` to every `make` call
   (unset for CPU); the Makefile does no CUDA detection of its own and derives
   the GPU family and architecture flags from `GPU`.
1. Writes the kernel launcher stubs (all builds; see "Kernel launcher stubs"
   above), rerun when the generator or a `TyrKernels` module changes.
2. GPU codegen, CUDA builds only: runs the `GenerateGpuKernels` executable over
   the `-Kkernels` modules, writing `cc/src/generated/`. Both are ordinary
   Lake dependencies of the target: the generator links no `libTyrC`
   (`TyrCodegen`), and the kernel modules (`TyrKernels`) are not precompiled,
   so their `.olean` files build without it. Lake reruns the generator when
   it, a kernel module's `.olean` or `-Kkernels` changes. Without CUDA the
   Makefile ignores generated `.cu` files and links the weak launcher stubs.
3. Runs `make -jN -C cc dylib` on every build (`N` = `TYR_MAKE_JOBS`, or the
   CPU count); a no-op run takes well under a second. Make alone decides what
   to recompile. The target's trace is the content of `libTyrC`, so if it is
   unchanged (e.g. after a comment-only edit) nothing downstream relinks.

Make uses compiler dependency files for C, C++, Objective-C++, and CUDA, so
header changes rebuild their consumers. `cc/build/build-flags.txt` lists every
value that affects compiled output (compilers, flags, GPU, kernel set, selected
objects); objects depend on it, and it is rewritten only when its content
changes, so a changed flag rebuilds and an unchanged one does not. The inexpensive regression command
`python3 .github/scripts/test_native_build.py` exercises these rules with tiny sources
in a temporary checkout: header edits, architecture switches, renamed headers,
clean/incremental output agreement, and launcher additions/removals. It does
not compile LibTorch or CUDA and does not modify workspace build artifacts.

The Makefile takes every native dependency from `external/` — libtorch
(`external/wheels/torch`), soxr (built from the `external/git/soxr` checkout without
cmake), Gloo headers (`external/git/gloo`, pinned to the wheel's matching
PyTorch submodule revision), Arrow/Parquet (`external/wheels/pyarrow`, linked by the file names the
wheel provides, e.g. `libarrow.so.2500`), and the CUDA runtime and NCCL (`external/wheels/nvidia`, CUDA
variant only); `deps/fetch.sh` populates them. It probes only Lean
(`lean --print-prefix`) and the CUDA toolkit (`nvcc`). GPU targets are a table
in `cc/Makefile`: `GPU` (from `-Kgpu`) selects SASS and the family, which
selects the ThunderKittens arch guards —

| `GPU` | family | compute / sm |
|---|---|---|
| `A100` | AMPERE | `compute_80a` / `sm_80a` |
| `H100` (default) | HOPPER | `compute_90a` / `sm_90a` |
| `B200` | BLACKWELL | `compute_100a` / `sm_100a` |
| `B300` | BLACKWELL | `compute_103a` / `sm_103a` |
| `GB10` | HOPPER | `compute_121` / `sm_121` |

`tk_vendor_mha_h100.cu` (which emits sm_90a-only `wgmma`) is compiled only when
`GPU=H100`; other targets get `tk_vendor_stubs.cpp` instead.

Lean code is compiled and linked with Lake's bundled clang, and `cc/` is
built with the system compiler. The two meet at a shared library:
`make -C cc dylib` links `cc/build/libTyrC.so` (`.dylib` on macOS) against
libtorch, Arrow/Parquet, soxr and, for a CUDA build, the CUDA runtime, with
run paths relative to `cc/build`. It does not link the Lean runtime; its
`lean_*` references resolve from the executable or the `lean` process that
loads it (`-undefined dynamic_lookup` on macOS). Because the system glibc and
libstdc++ stay behind `libTyrC`, the Lean toolchain's older glibc never has to
resolve them. It exports only the functions Lean calls (`lean_torch_*`,
`lean_launch_*` for the GPU kernel launchers, and a few other `lean_<prefix>_*`
groups; a new FFI prefix must be added to both lists) and the `tyr_ops::`
dispatch API used by
`cc/tools/bench_flash_attn.cpp` (`cc/map/libTyrC.map`, `cc/map/libTyrC.exports` on
macOS). Its copies of C++ template code and of soxr stay private, so they never
stand in for libtorch's own.

The package lists `` `@/libtyr `` in `moreLinkLibs`, so Lake links it into
every executable and precompiled module library. `packageLinkArgs` adds only a
run path to `cc/build` (`tyrCRPathArgs`) and, on Linux,
`linuxRuntimeIsolationLinkArgs`. Lean links executables with `-rdynamic`, so
every library the Lean runtime links statically is exported and would override
libtorch's copy: `-l:libgcc_s.so.1` ahead of the bundled linker's `-lunwind`
keeps libgcc's exception unwinder (LLVM's crashes on libtorch exceptions), and
`-Wl,--exclude-libs,libuv.a` keeps Lean's libuv from replacing the one bundled
in `libtorch_cpu` (its TCPStore). The five
`lean_lib`s are `Tyr` (default target, precompiled, links `libTyrC`),
`TyrCodegen` (the GPU code generator: `Tyr.GPU.Codegen.*` and the pure modules
it needs, including `Tyr.Basic.Types`; it does not link `libTyrC`, so neither do
the generator executables), `Tests`, `TestsExperimental` and `Examples`, plus
the `lean_exe` targets. `TyrCodegen` is declared after `Tyr` because Lake gives
a module to the last declared library whose roots match it.

### Running executables

Executables land in `.lake/build/bin/` and find `libTyrC` through a run path
relative to the binary (`$ORIGIN/...` on Linux, `@loader_path/...` on macOS);
`libTyrC` finds libtorch and Arrow the same way, so the checkout can move as
long as `external/` moves with it. No `LD_LIBRARY_PATH` is needed, with one
exception: a compiler newer than the system's (e.g. a gcc loaded as a cluster
module) builds `libTyrC` against its own libstdc++, which the loader finds only
while that module is loaded. `cc/tools/check_libstdcxx.sh` warns about this
right after linking `libTyrC`.

Run executables with `lake exe <Exe> [args]` (builds it if needed) or
`lake env .lake/build/bin/<Exe> [args]` (no rebuild); both set Lean's module
search path, which executables using the interpreter need.

### Configuration

Build configuration is a set of Lake options (`get_config?` in the lakefile),
given together on `lake -R`. Lake keeps them until the next `lake -R`, which
replaces all of them:

| Option | Effect |
|---|---|
| `-Kcuda=<toolkit>` | CUDA build with this toolkit, which must contain `bin/nvcc`. Without it: CPU build (CUDA kernels replaced by stubs). Checked against the libtorch from `deps/fetch.sh` |
| `-Kgpu=<name>` | with `-Kcuda`, required: GPU to build kernels for, `H100`, `A100`, `B200`, `B300` or `GB10` |
| `-Kkernels="<module> ..."` | kernel module(s) to generate CUDA for, space-separated (default `Tyr.GPU.Kernels.MhaH100`) |

Environment variables for a single build:

| Variable | Effect |
|---|---|
| `TYR_MAKE_JOBS` | parallel jobs for the native `make` build (default: CPU count) |
| `NCCL_ROOT` | NCCL discovery hint (Makefile) |

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

Lean side (`Tyr.Torch`):

```lean
@[extern "lean_torch_get_live_tensors"]
opaque get_live_tensors : IO UInt64

@[extern "lean_torch_manual_seed"]
opaque manualSeed (seed : UInt64) : IO Unit
```

Makefile targets: `all` (default) and `dylib` (both build `libTyrC`),
`bench-flash-attn` (standalone C++ attention benchmark from
`cc/tools/bench_flash_attn.cpp`), `soxr`, `clean`. `check-deps` runs first and
fails early, pointing at `deps/fetch.sh`, when `external/git/soxr`,
`external/git/ThunderKittens`, `external/wheels/torch` or the Arrow libraries are
missing.

## Usage example

This example uses indexed operations; native handles remain inside the adapters.
The checkpoint function checks stored shape and dtype before returning.

```lean
import Tyr.Torch
import Tyr.Typed.Neural
import Tyr.SafeTensors.Schema
open torch

def nativeExample : IO Unit := do
  let x : Tensor { shape := #[2, 2], dtype := .Float32, device := .CPU } := Tensor.ones #[2, 2]
  let y := Tensor.add x x
  let z ← Tensor.randn #[2, 2]
  IO.println s!"values: {← y.toFloatArray}, random shape: {z.runtimeShape}"
  IO.println s!"live tensors: {← torch.get_live_tensors}"

def loadCheckedWeight (path : String) : IO (Tensor { shape := #[768, 768], dtype := .Float32, device := .CPU }) := do
  let handle ← safetensors.openHandle path
  safetensors.loadFromHandleWithSpec handle "layer.weight" ⟨#[768, 768], .Float32⟩
```

`Tests.FfiCrashProbe` is a separate intentional device-error probe. It is not a
normal unit test. Distributed examples require their own process-group setup;
see [distributed training](distributed.md).

Build and run through Lake:

```bash
lake build                                  # target libtyr → codegen → make -C cc dylib
lake exe TrainGPT                           # build if needed, then run
# GPU build for a specific kernel module and target:
lake -R -Kcuda=/usr/local/cuda -Kgpu=H100 -Kkernels=Tyr.GPU.Kernels.MhaH100 build RunMhaH100
# Manual probe of the FFI failure mode:
lake exe ffi_crash_probe
```

## Related guides

- [Getting started](getting-started.md) — the day-one build path this chapter explains
- [Core tensors](core/tensors.md) — the indexed tensor API and native adapter boundary
- [Autodiff](autodiff.md) — gradient bindings (`lean_torch_backward` and friends)
- [Serialization](serialization.md) — safetensors handles and sharded loading
- [Distributed training](distributed.md) — the `torch.dist` collectives over c10d
- [GPU DSL codegen](gpu/dsl-codegen.md) — where the generated `.cu` kernels come from
- [GPU kernels](gpu/kernels.md) — the `@[gpu_kernel]` side of the launcher stubs
- [Data loading](data.md) — parquet readers and other data-path bindings
- [Examples and testing](examples-and-testing.md) — the executables these link args serve

Exhaustive symbol-level documentation is generated separately by doc-gen4; see
`docbuild/`.
