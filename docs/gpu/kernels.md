# GPU kernels and ops

Tyr ships a catalog of GPU kernels written in its own Lean-embedded,
ThunderKittens-style DSL, plus a thin typed ops layer that dispatches between
those kernels and portable libtorch fallbacks. Use the ops layer
(`Tyr.GPU.Ops.*`) when you want attention or a fused solver that picks the best
available path at runtime; use the raw kernel launchers
(`Tyr.GPU.Kernels.*`) when you are developing, benchmarking, or validating a
specific kernel and want full control over grid, block, shared memory, and
stream. The DSL itself is covered in [GPU DSL and codegen](dsl-codegen.md);
this chapter covers the kernel catalog and the op bridge.

## Architecture

Three layers sit between a Lean `def` and a running CUDA kernel:

```text
@[gpu_kernel .SM90] def foo … : KernelM Unit        -- kernel declaration (DSL)
        │  elaboration (Tyr/GPU/Codegen/Attribute.lean)
        ▼
foo.kernel : Kernel            foo.launch : … → IO Unit   -- IR record + FFI launcher
        │  lake build: GenerateGpuKernels → cc/src/generated/*.cu, then make -C cc
        ▼
extern "C" lean_object* lean_launch_Tyr_GPU_Kernels_foo(…)   -- C++ launcher in libTyrC.a
        ▲
Tyr.GPU.Ops.*  (typed wrappers, AttentionProblem dispatch, portable fallback)
```

### Kernel declarations

A kernel is a `KernelM Unit` do-block over typed DSL values — `GPtr dtype`
(global pointer), `KVal τ` (runtime scalar), register/shared tiles and vectors
— annotated with `@[gpu_kernel arch]`. The smallest example
(`Tyr/GPU/Kernels/Copy.lean:15`):

```lean
@[gpu_kernel .SM90]
def copy64x64 (input : GPtr GpuFloat.Float32) (output : GPtr GpuFloat.Float32) : KernelM Unit := do
  let coord ← blockCoord2D
  let reg ← allocRT .Float32 64 64
  let smem ← allocST .Float32 64 64
  loadGlobal smem input coord
  load reg smem
  store smem reg
  storeGlobal output smem coord
  sync
```

Shared enums live in `Tyr/GPU/Types.lean`: `GpuFloat` (`Float32`, `Float16`,
`BFloat16`, `FP8E4M3`, `FP8E5M2`, `FP8E8M0`, `FP4E2M1X2`), `GpuArch`
(`.SM80`/`.SM90`/`.SM100`), `GpuFamily` (`.Ampere`/`.Hopper`/`.Blackwell`),
`TileLayout` (`.Row`/`.Col`). `GpuArch.toGuard` maps an arch to the C++
preprocessor guard (`SM90 → "KITTENS_HOPPER"`, `Tyr/GPU/Types.lean:104`).

### What `@[gpu_kernel]` generates

The attribute (`Tyr/GPU/Codegen/Attribute.lean:730`) elaborates each annotated
declaration into two companions:

- `<name>.kernel` — a `Kernel` record (name, arch, family, params, IR body,
  `sharedMemBytes`) registered for C++ emission
  (`generateKernelCompanion`, `Tyr/GPU/Codegen/Attribute.lean:297`).
- `<name>.launch` — the callable FFI entrypoint
  (`generateLaunchDecl`, `Tyr/GPU/Codegen/Attribute.lean:442`):

```lean
@[extern "lean_launch_<mangled_name>"]
opaque <name>.launch
    (ptrs… : @& Tensor) (scalars… : UInt64/…)
    (grid_x grid_y grid_z : UInt64) (block_x block_y block_z : UInt64)
    (sharedMem : UInt64) (stream : CudaStream) : IO Unit
```

Pointer parameters become untyped `@& Tensor` arguments (any `T s` is
accepted; shape checking is the caller's problem at this layer), `KVal`
scalars become Lean scalars, and `CudaStream` is `UInt64`
(`Tyr/GPU/Codegen/TileTypes.lean:212`) — `torch.cuda_current_stream : IO
UInt64` provides it (`Tyr/Torch.lean:121`). The generated C++ launcher wraps
the kernel in its family guard, maps `sharedMem == 0` to the kernel default,
sets `cudaFuncAttributeMaxDynamicSharedMemorySize` when needed, launches
`<<<grid, block, smem, stream>>>`, and turns `cudaGetLastError` into a Lean IO
error (`Tyr/GPU/Codegen/Attribute.lean:420-438`). On a build without the
matching family it still links, but raises "unavailable in this build" at
runtime.

The bare form `@[gpu_kernel]` (no arch) is for kernels polymorphic over
`Arch.ArchLevel`; it emits one companion per architecture plus a polymorphic
launcher (`Tyr/GPU/Codegen/Attribute.lean:748-760`). Blackwell-family twins of
Hopper kernels are separate declarations that call `setFamily .Blackwell`
before the shared body, e.g. `dopri5Combine64Blackwell`
(`Tyr/GPU/Kernels/RKCombine.lean:184`) and
`tkFusedRMSNormResidual1024Blackwell` (`Tyr/GPU/Kernels/FusedRMSNorm.lean:179`).

### Build integration

`extern_lib libtyr` in `lakefile.lean:416-548` runs the
`GenerateGpuKernels` executable over the configured kernel modules, emitting
CUDA into `cc/src/generated/` (e.g. `Tyr_GPU_Kernels_MhaH100.cu` plus
`tyr_gpu_kernel_stubs.cpp`), then rebuilds `cc/build/libTyrC.a` via
`make -C cc`. Which modules get emitted is controlled by environment
variables — `TYR_GPU_CODEGEN_MODULE` (space-separated module list, default
`Tyr.GPU.Kernels.MhaH100`) and `TYR_SKIP_GPU_CODEGEN=1` to reuse the checked-in
generated tree. On a fresh checkout the codegen step needs the kernel modules'
own `.oleans`, so CI does a two-phase build: first `TYR_SKIP_GPU_CODEGEN=1
lake -R build <kernel modules>`, then the real build
(`.github/workflows/cuda-smoke.yml:149-171`). The full variable table is in
[Getting started](../getting-started.md).

Only the modules passed to the generator get `.cu` files, so a `.launch`
symbol for an un-emitted kernel fails at link time, not at elaboration time.
The CUDA-smoke workflow compensates by listing exactly the modules its tests
call (`MhaGB10`, `FusedRMSNorm`, `FusedLayerNorm`, `MhaH100Decode`,
`RKCombine`, `BrownianSample`).

### Module layout

`Tyr/GPU/Kernels.lean` is the umbrella: it imports the six family entrypoints
(`Attention`, `StateSpace`, `Parallel`, `Gemm`, `Normalization`,
`Experimental`) plus `BrownianSample` and `RKCombine` directly. Notably absent
from the umbrella: `AttentionFactory`, `MhaGB10`, and `FlashAttnCausal64` —
import those leaf modules explicitly. `Tyr/GPU/Kernels/Examples.lean` (via
`Experimental`) holds small teaching kernels (`simpleGemm`, `flashAttnFwd`,
`layerNorm`, …). Family entrypoints re-export selected leaf kernels as
`abbrev tk…` in the `Tyr.GPU.Kernels` namespace.

`Tyr/GPU/Ops.lean` is the ops umbrella, but it covers only the attention
surface (`AttentionProblem`, `MhaH100`, `FlashAttn`); the fused-solver ops
(`RKFused`, `SDEFused`, `BrownianFused`) are imported from their own modules.

## Key APIs

### Kernel catalog (selection)

All of the following are `@[gpu_kernel]` declarations; call them through their
generated `.launch`.

| Module | Kernels | Notes |
|---|---|---|
| `Tyr/GPU/Kernels/Copy.lean` | `copy64x64` (`tkCopy`) | minimal 64×64 global→shared→register→global copy |
| `Tyr/GPU/Kernels/Rotary.lean` | `Rotary.rotaryFwd` | RoPE on one 64×64 tile; backward in `RotaryBwd.lean` |
| `Tyr/GPU/Kernels/FusedLayerNorm.lean` | `tkFusedLayerNormResidual1024`, `…Blackwell`, `…F32`, `…F32Blackwell` | fused residual + LayerNorm, `d_model=1024` |
| `Tyr/GPU/Kernels/FusedRMSNorm.lean` | `tkFusedRMSNormResidual1024`, `…Blackwell`, `…F32`, `…F32Blackwell` | fused residual + RMSNorm, `d_model=1024` |
| `Tyr/GPU/Kernels/MhaH100.lean` | `tkFlashAttnFwd{2,12}Block[Lse]`, `tkMhaH100Fwd{2,12}Block`, `tkMhaH100BwdPrep2Block`, `tkMhaH100Bwd{2,12}Block{Partials,Dq,KvSweep}` | 13 kernels; BF16, head_dim 64, `seq=128` (2 KV blocks) and `seq=768` (12 blocks); forward is currently non-causal (`MhaH100.lean:68-70`) |
| `Tyr/GPU/Kernels/MhaH100Decode.lean` | `tkMhaH100DecodeFwd{,64,256}`, `tkMhaH100DecodeFwdGqa{,64,256}` | decode (`qSeq=1`) for head_dim 128/64/256, MHA and GQA; runtime loop over `ceil(kvSeq/64)` blocks with tail masking |
| `Tyr/GPU/Kernels/MhaH100LCF.lean` | `tkMhaH100LCFFwd64`, `tkMhaH100LCFFwd128` | load-compute-finish forward variants |
| `Tyr/GPU/Kernels/MhaGB10.lean` | `tkFlashAttnGb10Fwd2Block[Lse]`, `tkMhaGb10Fwd2Block`, `tkMhaGb10BwdPrep2Block`, `tkMhaGb10Bwd2BlockPartials` | GB10 (Blackwell-consumer) 2-block MHA |
| `Tyr/GPU/Kernels/FlashAttn3.lean` | `flashAttn3Fwd`, `flashAttn3FwdGQA`, `flashAttn3FwdPersistent`, `flashAttn3BwdPrep`, `flashAttn3Bwd` | FA3-style warp-specialized producer/consumer |
| `Tyr/GPU/Kernels/AttentionFactory.lean` | `tkFlashAttnFwd2BlockFactory`, … (12 `…Factory` kernels) | parameterized `FAVariantConfig` template (`mkFlashAttnFwdBody`, `mkFlashAttnBwdPrep`, `mkFlashAttnBwdPartials`) mirroring the hand-written MhaH100 kernels |
| `Tyr/GPU/Kernels/RKCombine.lean` | `dopri5Stage{2..7}[Blackwell]`, `dopri5Combine64[Blackwell]` | tableau-driven generators `rkStageSumBody` / `rkCombineBody` instantiate any explicit RK tableau |
| `Tyr/GPU/Kernels/BrownianSample.lean` | `keyedNormal[Blackwell]`, `vbtIncrement[Blackwell]`, `emStepVbt[Blackwell]` | device PRNG replicating the CPU `PRNGKey` LCG + Box-Muller per element |
| `Tyr/GPU/Kernels/{Bf16Gemm,PrecisionGemm,NvFp4Gemm}.lean` | `tkH100Bf16GemmFwd`, `tkB200Bf16GemmFwd`, `tkH100Fp8E4M3GemmFwd`, `tkB200NvFp4GemmFwd`, … | re-exported via `Kernels/Gemm.lean` |
| `Tyr/GPU/Kernels/{Distributed,MOE}.lean` | `tkAllGatherFwd`, `tkAllReduceFwd`, `tkAgGemmFwd`, `tkGemmArFwd`, `tkMoeDispatchGemm`, … | re-exported via `Kernels/Parallel.lean` |
| `Tyr/GPU/Kernels/{RingAttn,UlyssesAttn,Based,LinearAttn}.lean` | `tkRingAttnPartial`, `tkUlyssesAllToAllFwd`, `tkBasedLinearAttnFwd`, … | re-exported via `Kernels/Attention.lean` |
| `Tyr/GPU/Kernels/Mamba2.lean` | `tkMamba2Fwd` | via `Kernels/StateSpace.lean` |

### Ops: typed FlashAttention (`Tyr.GPU.Ops.FlashAttn`)

The main user-facing entry (`Tyr/GPU/Ops/FlashAttn.lean:87`):

```lean
def flashAttn
    (query : Query batch nHead qSeq headDim)        -- T #[batch, nHead, qSeq, headDim]
    (key value : KeyValue batch nKvHead kvSeq headDim)
    (attnMask : Option (PaddingMask batch kvSeq) := none)
    (dropoutP : Float := 0.0) (isCausal : Bool := false)
    (scale : Option Float := none) (enableGqa : Bool := false)
    : Query batch nHead qSeq headDim
```

It forwards to `torch.nn.tyrFlashAttn4d` (`Tyr/Torch.lean:1198`), i.e. the
C++ operator `tyr::flash_attn` registered via `TORCH_LIBRARY(tyr, …)` with a
`CompositeImplicitAutograd` implementation (`cc/src/tyr_ops.cpp:818-824`), so
`torch.autograd.backward` works through it with no custom Lean-side rule. The
C++ `select_route` (`cc/src/tyr_ops.cpp:314`) mirrors the Lean-side selector
and calls the generated `lean_launch_*` launchers for covered shapes, PyTorch
SDPA otherwise. Companions in the same module:

- `scaledDotProductAttention` — PyTorch-named alias.
- `dispatchRoute`, `flashAttnWithRoute` — expose the `.tkKernel` vs `.portable`
  decision (`DispatchRoute`) for observability.
- `currentSpecialization`, `supportsTkMhaKernel`, `attentionProblem` — typed
  probes over the same selector.
- `flashAttnDyn` — shape-erased adapter: takes rank-4 `T #[]` tensors,
  validates runtime shapes, packs them into the typed API, and returns a
  `Sigma`-typed existential (`DynOut`).

### Ops: H100 MHA (`Tyr.GPU.Ops.MhaH100`)

Explicit two-call forward/backward over the `MhaH100` kernels. Shapes are
fixed: `BF16 seq := T #[1, 1, seq, 64]`, per-tile vector
`L kvBlocks := T #[1, 1, 1, kvBlocks * 64]`. Variant selection is a typeclass:

```lean
class Variant (seq kvBlocks : UInt64) where
  launchFwd : BF16 seq → BF16 seq → BF16 seq → BF16 seq → L kvBlocks → LaunchCfg → IO Unit
  launchBwd : BF16 seq → … → F32 seq → LaunchCfg → IO Unit
-- instances: Variant 128 2 and Variant 768 12
```

- `mhaFwd (q k v : BF16 seq) (stream) : IO (FwdCtx seq kvBlocks)` — allocates
  `out`/`lOut`, launches forward, returns the saved context
  (`Ops/MhaH100.lean:105`).
- `mhaBwd (q k v dO) (ctx) (stream) : IO (F32 seq × F32 seq × F32 seq)` —
  backward prep + `KvSweep`, returns FP32 `(dQ, dK, dV)`
  (`Ops/MhaH100.lean:115`).
- `mhaFwdPortable` / `mhaBwdPortable` — SDPA + autograd fallback for any
  `seq`, causal or not.
- `mhaFwdDispatch` / `mhaBwdDispatch` (aliases `mhaFwdMain` / `mhaBwdMain`) —
  pick kernel vs portable via `AttentionProblem.currentSpecialization` and
  carry the choice in `FwdCtxDispatch` (see the decode asymmetry note under
  Caveats below).
- `supportsKernelShape`, `recommendedKernelSeqs := [128, 768]` — coverage
  predicate and hint list.

### Ops: runtime dispatch (`Tyr.GPU.Ops.AttentionProblem`)

`AttentionProblem` (`Ops/AttentionProblem.lean:106`) is a runtime descriptor —
batch, head counts, `qSeq`/`kvSeq`, `headDim`, dtype, device, arch, mode,
mask kind, dropout, causality, scale, GQA, window — with classifiers
(`AttentionMode`, `HeadDimClass`, `GqaClass`, `ScaleClass`,
`AttentionRoutingMetadata`). Build one from typed tensors with
`AttentionProblem.ofQKV`, or shape-only with `AttentionProblem.selfAttention`.
The central decision is:

```lean
def AttentionProblem.currentSpecialization : AttentionProblem → AttentionSpecialization
-- .tkMhaH100Decode   decode qSeq=1, BF16, CUDA+SM90, headDim ∈ {64,128,256}, no mask/dropout/causal
-- .tkMhaH1002Block   dense non-causal prefill, batch=heads=1, seq=128, headDim=64, BF16
-- .tkMhaH10012Block  same with seq=768
-- .portable          everything else
```

Eligibility details live in `currentTkBaseEligible` / `currentTkDecodeEligible`
(`Ops/AttentionProblem.lean:289,316`). The decode path handles any positive
`kvSeq` via runtime tail masking; GQA requires `enableGqa := true`.

### Ops: fused simulation kernels

- `Tyr.GPU.Ops.RKFused` — fused fixed-step Dopri5 on tile states
  `Tile cols := T #[1, 1, 64, cols]` (`cols % 64 == 0`):
  `Workspace.make cols device : IO (Workspace cols)` (scratch allocated through
  `IO`, see the aliasing caveat below),
  `dopri5StepFused`, and
  `dopri5SolveFused (vf : Float → Tile cols → Tile cols) (t0 t1 : Float) (steps : Nat) (y0) (ws) (blackwell) (stream) : IO (Tile cols)`.
- `Tyr.GPU.Ops.SDEFused` — fused fixed-step Euler–Maruyama on flat states
  `T #[n]`: `emStepFused` (one launch per step; noise drawn inline by the VBT
  descent kernel) and `emSolveFused`, with `DeviceSchedule.make` precomputing
  the Brownian value-query schedules.
- `Tyr.GPU.Ops.BrownianFused` — host side of the device virtual Brownian
  tree: `buildValueSchedule` (mirrors the CPU `valueAux` descent; time tags
  must satisfy `t·10⁶ < 2²⁴` for exact Float32 representation),
  `uploadSchedule`, `tensorRootState`, and
  `vbtIncrementDevice` for raw `w(tB) − w(tA)` increments.

## Usage examples

Raw launcher — reconstructed example (from
`Examples/GPU/RunMhaH100.lean` and `Examples/GPU/RunCopy.lean`):

```lean
import Tyr.Torch
import Tyr.GPU.Kernels.MhaH100

open torch Tyr.GPU.Kernels

-- q k v dO : T #[1, 1, 128, 64]  (BF16, CUDA)
let stream ← torch.cuda_current_stream

-- forward: ptrs, then KVal scalars (seq, head_dim), grid, block, sharedMem, stream
let out  := torch.zeros_like q
let lOut : T #[2, 64] := torch.zeros #[2, 64] false (Device.CUDA 0)
tkMhaH100Fwd2Block.launch q k v out lOut 128 64  1 2 1  128 1 1  0 stream
let _ ← torch.cuda_synchronize

-- backward: prep computes the D vector, KvSweep accumulates dQ/dK/dV
let dVec : T #[2, 64] := torch.mul_scalar lOut 0.0
tkMhaH100BwdPrep2Block.launch dO out dVec 128 64 1 2 1 128 1 1 0 stream
let dQ := torch.zeros #[1, 1, 128, 64] false (Device.CUDA 0)
let dK := torch.zeros #[1, 1, 128, 64] false (Device.CUDA 0)
let dV := torch.zeros #[1, 1, 128, 64] false (Device.CUDA 0)
tkMhaH100Bwd2BlockKvSweep.launch q k v dO lOut dVec dQ dK dV 128 64 1 2 1 128 1 1 0 stream
let _ ← torch.cuda_synchronize
```

The convention is positional: kernel pointer/scalar arguments in declaration
order, then `grid_x grid_y grid_z`, `block_x block_y block_z`, `sharedMem`
(`0` = generated default), `stream`. For the 2-block MHA forward, `grid_y = 2`
is the KV-block count and the block is one 128-thread warpgroup.

Typed op with route inspection and autograd — reconstructed example (from
`Examples/GPU/RunFlashAttnOp.lean`):

```lean
import Tyr.Torch
import Tyr.GPU.Ops.FlashAttn

open torch Tyr.GPU.Ops.FlashAttn

-- q k v : T #[1, 1, 128, 64] BF16 CUDA leaves with requires_grad
let (route, out) := flashAttnWithRoute q k v   -- route = .tkKernel for seq=128 non-causal
torch.autograd.backward out dO                  -- grads flow through tyr::flash_attn
let dq := torch.toFloat' (torch.autograd.grad_of q)
```

Fused ODE solve — reconstructed example (from
`Examples/GPU/RunRKFusedSolve.lean`):

```lean
import Tyr.Torch
import Tyr.GPU.Ops.RKFused
import Examples.GPU.Parity

open torch Tyr.GPU.Ops.RKFused

-- dy/dt = -y on a [1, 1, 64, 64] CUDA state, 200 fixed steps
let blackwell ← isBlackwellFamily
let stream ← torch.cuda_current_stream
let ws ← Workspace.make 64 (Device.CUDA 0)
let y1 ← dopri5SolveFused (fun _t y => mul_scalar y (-1.0))
  0.0 1.0 200 y0 ws blackwell stream
```

More runner programs live under `Examples/GPU/`: `RunRotary`, `RunLayerNorm`,
`RunRMSNorm`, `RunFlashAttn`, `RunMhaH100Seq768`, `RunMhaH100Decode` (Llama-3
d=128 GQA-4, Qwen3 d=64, Qwen3.5/3.6 d=256 shapes, including a `kvSeq=2049`
tail case), `RunMhaGB10`, `RunFlashAttn3`, `RunB200Bf16Gemm`,
`RunBrownianSample`, `RunEulerMaruyamaFused`, `RunRKCombine`. See
[Examples and testing](../examples-and-testing.md) for how to build and run
them.

## Validation and parity

Numerical correctness is validated on GPU hardware, not in CPU CI. The moving
parts:

- **Fixtures.** Each `Examples/GPU/Run*.lean` harness declares a
  `FixtureSpec` pointing at `data/gpu_fixtures/<suite>/` (not checked in) —
  PyTorch reference tensors saved with `torch.data.saveTensor`, generated on
  first run or with `--regen`. Shared machinery:
  `Examples/GPU/FixtureRunner.lean` (`runWithFixtures`, `--trials`, `--seed`,
  `--gen-only`) and `Examples/GPU/Parity.lean` (`requireCuda`, `seedFixtures`,
  `reportTensorComparison`, allclose with rtol/atol).
- **Shell harnesses.** `scripts/gpu/test_<kernel>_e2e.sh` all delegate to
  `scripts/gpu/run_e2e_kernel.sh <KernelModule> <RunnerExe> <Label>`, which
  emits CUDA for the module, rebuilds `libTyrC.a`, regenerates fixtures, and
  runs the parity check; `scripts/gpu/test_parity_suite.sh` is the umbrella.
  See [Getting started — GPU parity scripts](../getting-started.md).
- **Vendored reference.** If `TYR_GPU_VENDORED_REF_RUNNER` is set (default:
  `scripts/gpu/run_vendored_reference.sh` when executable), each suite also
  diffs against the vendored ThunderKittens CUDA reference in
  `external/git/ThunderKittens`.
- **LeanTest suites.** `Tests/TestGPUE2E.lean` (executable `TestGPUE2E`)
  wraps the torch-parity runs (copy, rotary, layernorm f32/bf16, rmsnorm
  f32/bf16, flashattn, mha_h100) plus the vendored-parity checks, printing
  `[skip]` when CUDA is unavailable. `Tests/TestGPUGB10E2E.lean` covers the
  Blackwell-family kernels on GB10. `Tests/TestGPUKernels.lean` is the
  CPU-side complement: codegen invariants and
  `AttentionProblem.currentSpecialization` selection.
- **CI.** `.github/workflows/cuda-smoke.yml` runs on a self-hosted GPU
  runner (GB10 by default; opt a PR in with the `gpu-ci` label), does the
  two-phase codegen build, and runs `TestGPUGB10E2E --fail-fast`, plus
  `RunMhaH100Decode` on Hopper hardware.

## Caveats

- **Mutation hides behind pure types.** Kernels mutate `T s` tensors that
  look pure at the Lean level. Allocate every mutable scratch buffer through
  `IO` (like `Workspace.make` does): identical pure factory expressions get
  common-subexpression-eliminated into one shared tensor, and two kernel
  outputs then alias the same buffer (`Tyr/GPU/Ops/RKFused.lean:16-22`,
  `Examples/GPU/RunMhaH100.lean:137-138`). Fused solvers also double-buffer
  state for the same reason.
- **Coverage is deliberately narrow.** The MhaH100 2/12-block forward is
  non-causal (dynamic block-offset masking is not yet represented in the IR,
  `Tyr/GPU/Kernels/MhaH100.lean:68-70`); decode supports only head_dim
  {64, 128, 256} with no mask/dropout/causal; anything else falls back to
  SDPA. Check `AttentionProblem.currentSpecialization` (or
  `flashAttnWithRoute`) if you need to know which path you got.
- **Decode in the Lean dispatch layer.** `mhaFwdDispatch` maps
  `.tkMhaH100Decode` to portable SDPA (`Tyr/GPU/Ops/MhaH100.lean:176-177`);
  native decode is currently only wired through `tyr::flash_attn`.
- **Scale policy.** The effective attention scale policy is
  `AttentionProblem.scaleMatchesDefault` (`1 / sqrt(headDim)`).

## Related guides

- [GPU DSL and codegen](dsl-codegen.md) — the `KernelM` DSL, tile IR, and C++ emission these kernels are written in
- [Getting started](../getting-started.md) — build knobs (`TYR_GPU_CODEGEN_MODULE`, `TYR_SKIP_GPU_CODEGEN`) and the parity-script flow
- [Examples and testing](../examples-and-testing.md) — building and running the `Examples/GPU/Run*` harnesses and test suites
- [FFI and build](../ffi-and-build.md) — `cc/`, `libTyrC.a`, and how generated CUDA reaches the linker
- [Core tensors](../core/tensors.md) — `T s`, `Device`, the `torch.*` FFI used at the launch boundary
- [Autodiff](../autodiff.md) — `torch.autograd`, which is what makes `tyr::flash_attn` differentiable
- [diffeq.md](../diffeq.md) — the generic solvers the fused RK/SDE ops accelerate
- [ThunderKittens porting status](thunderkittens-porting-status.md) — source-to-Lean parity matrix for the kernel catalog

Exhaustive, per-symbol documentation for every kernel and op mentioned here is
generated by doc-gen4 (see `docbuild/`); this chapter is a guide, not a
reference dump.
