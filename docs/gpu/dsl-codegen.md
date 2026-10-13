# GPU DSL and code generation

Tyr's GPU DSL is a Lean-embedded language for authoring GPU kernels that are
compiled to ThunderKittens/CUDA C++ at build time. Tile shapes, dtypes, layouts,
and memory locations are tracked in Lean's type system, so shape and layout
errors fail at elaboration instead of producing broken CUDA. Use it when the
built-in kernel catalog (see [GPU kernels and ops](kernels.md)) does not cover
the kernel you need, or when you want a new fused kernel with the same
correctness guarantees.

## Architecture and main abstractions

The pipeline has four stages:

1. **Authoring** — a kernel is a function returning `KernelM Unit`
   (`abbrev KernelM := StateM KernelState`, `Tyr/GPU/Codegen/Monad.lean:44`).
   The `do` block manipulates typed handles and every primitive appends a
   `KStmt` constructor to an IR accumulator.
2. **IR** — `KStmt` (155 constructors, `Tyr/GPU/Codegen/IR.lean:74`) is the
   instruction type; a finished `Kernel` record bundles name, arch, family,
   params, body, and shared-memory usage (`Tyr/GPU/Codegen/IR.lean:470`).
3. **Registration** — the `@[gpu_kernel]` attribute
   (`Tyr/GPU/Codegen/Attribute.lean:730`) type-checks the declaration,
   extracts parameters from the Lean signature, and generates two companions:
   a `Kernel`-valued constant `<decl>.kernel` and an opaque FFI launcher
   `<decl>.launch`.
4. **Emission and build** — `lake exe GenerateGpuKernels <Module>…`
   (`Tyr/GPU/GenerateGpuKernels.lean`) imports kernel modules, `evalExpr`s
   the companion constants, lowers each `Kernel` to guarded ThunderKittens C++
   via `generateKernel` (`Tyr/GPU/Codegen/EmitNew.lean:1654`), and writes one
   `.cu` file per Lean module (default `cc/src/generated`). The C++ side is
   then compiled by `make -C cc` as part of the `target libtyr` build.

### Shared enums (`Tyr/GPU/Types.lean`)

The common vocabulary between Lean-side construction and emitted C++:

```lean
inductive GpuFloat where
  | Float32 | Float16 | BFloat16
  | FP8E4M3 | FP8E5M2 | FP8E8M0 | FP4E2M1X2

inductive TileLayout where | Row | Col
inductive GpuArch where | SM80 | SM90 | SM100
inductive GpuFamily where | Ampere | Hopper | Blackwell
```

Renderers used by the emitter: `GpuFloat.toCpp` (`"bf16"`, …),
`GpuArch.toGuard` (`"KITTENS_AMPERE" / "KITTENS_HOPPER" / "KITTENS_BLACKWELL"`),
`GpuArch.toNvccArch` (`"sm_80a" / "sm_90a" / "sm_100a"`), and
`GpuArch.toFamily`. `TileLoc` (`Register`/`Shared`/`Global`/`TensorCore`),
`SwizzleMode`, and `Scope` round out the module.

### Abstract tiles (`Tyr/GPU/Tile.lean`)

```lean
class Tile (α : Type) where
  dtype : GpuFloat
  rows : Nat
  cols : Nat
  location : TileLoc
  layout : TileLayout

class RegisterTile (α : Type) extends Tile α where ...
class SharedTile (α : Type) extends Tile α where ...
```

Helpers `Tile.elements`, `Tile.bytes`, `Tile.validForMMA` work over any type
with an instance; the codegen handles below carry the instances.

### Capabilities (`Tyr/GPU/Capabilities.lean`)

`GpuCapabilities arch` is a typeclass with per-arch instances for SM80, SM90,
and SM100 exposing `maxSharedMem`, `supportedTypes`, `hasTMA`, `hasWGMMA`,
`hasTMEM`, `hasFP8E8M0`, `hasFP4`, `hasAsyncCopy`, and `mmaTileSize`. Proof
gates `RequiresTMA`, `RequiresWGMMA`, and `RequiresTMEM` make unsupported
operations fail at instance-resolution time. Note the SM100 numbers are marked
"estimated" in the source (`Tyr/GPU/Capabilities.lean:70`).

### Typed handles (`Tyr/GPU/Codegen/TileTypes.lean`)

Each handle wraps a `VarId` with phantom type parameters, which is what makes
mismatched shapes/layouts a Lean type error:

```lean
structure RT (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row) where
  id : VarId

structure ST (dtype : GpuFloat) (rows cols : Nat) (layout : TileLayout := .Row) where
  id : VarId

structure RV (dtype : GpuFloat) (len : Nat) where id : VarId
structure GPtr (dtype : GpuFloat) where id : VarId; name : String
structure KVal (T : Type) where id : VarId; name : String
```

Also: `STArray`, `SV`, `STRowVec`/`STColVec` (TMA vector views),
`SemaphoreArray`, `TT`/`TMEMPool` (Blackwell tensor memory), `KShared`
(runtime-sized dynamic shared memory), complex pairs `CRT`/`CST`/`CRV`/`CSV`,
and FFI aliases `Tensor := torch.T #[]`, `CudaStream := UInt64`
(`Tyr/GPU/Codegen/TileTypes.lean:209-212`).

### The builder monad (`Tyr/GPU/Codegen/Monad.lean`)

```lean
structure KernelState where
  nextId : Nat := 0
  body : Array KStmt := #[]
  arch : GpuArch := .SM90
  family : GpuFamily := .Hopper
  sharedMemBytes : Nat := 0
  launchBounds : Option (Nat × Nat) := none

abbrev KernelM := StateM KernelState

def freshVar : KernelM VarId
def emit (stmt : KStmt) : KernelM Unit
def setArch (arch : GpuArch) : KernelM Unit
def setFamily (family : GpuFamily) : KernelM Unit
def buildKernelM (name : String) (arch : GpuArch := .SM90)
    (params : Array KParam := #[]) (m : KernelM Unit) : Kernel
```

`buildKernelM` seals the record and warns (via `dbg_trace`) when
`sharedMemBytes` exceeds the arch budget. Because `KernelM` is a plain
`StateM`, builder errors go through `KernelM.fail`, which aborts with `panic!`
(`Tyr/GPU/Codegen/Monad.lean:57`) — configuration mistakes are not recoverable
at elaboration time.

### Primitives and loops

`Tyr/GPU/Codegen/Primitives.lean` (~300 defs) is the main authoring surface:
allocators (`allocRT`, `allocST`, `zeroRT`, …), tile `load`/`store`, TMA and
async global I/O, tensor-core ops, reductions, broadcasts, masking, semaphores,
and scalar ops. Dimension legality is enforced by type indices plus
`by decide` side goals, e.g. the warp-level MMA at
`Tyr/GPU/Codegen/Primitives.lean:608`:

```lean
def mma {M K N : Nat} {inDtype accDtype : GpuFloat}
    (dst : RT accDtype M N .Row)
    (a : RT inDtype M K .Row)
    (b : RT inDtype K N .Col)
    (c : RT accDtype M N .Row)
    (hM : M % 16 = 0 := by decide)
    (hK : K % 16 = 0 := by decide)
    (hN : N % 16 = 0 := by decide)
    : KernelM Unit
```

Variants: `mm` (no accumulate), `mmaT` (B transposed), `mmaAtBt`, plus
Hopper WGMMA and Blackwell tcgen05 ops gated by the capability classes.

Global memory is addressed through coordinates from
`Tyr/GPU/Codegen/GlobalLayout.lean`: `blockCoord2D : KernelM RTileCoord` gives
the block's tile coordinate, adjusted with `RTileCoord.withRow`/`withCol`;
`loadGlobal`/`storeGlobal` move tiles between `GPtr` and `ST`
(`GlobalLayout.lean:277-291`), and `loadGlobalAsync` adds TMA semaphores.

Loops use Lean `for` syntax: `krange lo hi` builds a static `KRange` and a
`ForIn KernelM KRange KIdx` instance (`Tyr/GPU/Codegen/Loop.lean:122`) runs the
body once to capture IR, then emits a single `.forLoop` statement. `kvrange`
takes a runtime `KVal UInt32` bound; `kstride` iterates between kernel scalars.

### Sugar and patterns

- `Tyr/GPU/Codegen/Notation.lean` — expression-style wrappers:
  `matmul a b .Float32`, the scoped infix `a ⬝ b` / `a ⬝ᵀ b`
  (`Notation.lean:192-195`, Float32 accumulators via `matmulF32`), and scoped
  `HAdd`/`HSub`/`HMul`/`HDiv` instances on `RT`/`RV` returning
  `KernelM (RT …)`. Requires `open scoped Tyr.GPU.Codegen`.
- `Tyr/GPU/Codegen/Macros.lean` — kernel patterns: `DoubleBuffer`, warp
  specialization (`asProducer`/`asConsumer`), `SoftmaxState` with
  `onlineSoftmax`/`onlineSoftmaxLog2`, `attentionBlockIter`, and config
  structures `FlashAttnConfig`/`GQAConfig`.
- `Pipeline.lean`, `PersistentKernel.lean`, `KernelTemplate.lean`,
  `TileDispatch.lean`, `Launch.lean` — N-stage ring buffers, persistent work
  loops, phased templates, tile-variant dispatch, launch-config computation.

### Architecture polymorphism (`Tyr/GPU/Codegen/Arch/`)

A parallel, type-indexed layer for writing one kernel body across
generations:

```lean
inductive ArchLevel where | Ampere | Hopper | Blackwell   -- Arch/Level.lean:24
inductive ArchLe : ArchLevel → ArchLevel → Prop            -- with ArchLe.trans

structure ArchKernelM (minArch : ArchLevel) (α : Type)     -- Arch/Monad.lean:32
```

`smartMMA`/`smartLoadAsync` (`Arch/Ops.lean:207,223`) select warp-MMA vs WGMMA
and sync vs async loads by target; `PolyKernel.compileAll`
(`Arch/Polymorphic.lean:65`) produces one `Kernel` per level. A bare
`@[gpu_kernel]` (no arch argument) requires the declaration to take an
`ArchLevel` parameter and generates SM80/SM90/SM100 companions named
`<decl>.kernel<suffix>` plus per-arch launchers.

### The `@[gpu_kernel]` attribute

```lean
syntax (name := gpuKernelAttr) "gpu_kernel" (term)? : attr   -- Attribute.lean:258
```

For `@[gpu_kernel .SM90] def k (p : GPtr …) (n : KVal UInt64) : KernelM Unit`
the handler (`Attribute.lean:730`) generates:

- `k.kernel : Kernel` — `buildKernelM` applied to the extracted `KParam`s and
  the body, i.e. the full IR record the emitter consumes;
- `@[extern "lean_launch_<mangled>"] opaque k.launch (p : @& Tensor)
  (n : UInt64) (grid_x grid_y grid_z block_x block_y block_z sharedMem : UInt64)
  (stream : CudaStream) : IO Unit` — the Lean-callable launcher, implemented by
  the generated C++ wrapper (pointers become `@& Tensor`, scalars keep their
  Lean scalar type).

Registrations live in the `gpuKernelPendingExt` map-declaration extension and
the runtime `gpuKernelRegistry`. `#generate_gpu_kernels` lists the companion
constants found in the current environment; the actual file writing is done by
the executable below.

### CUDA emission and build integration

`generateKernel (k : Kernel) : String` (`EmitNew.lean:1654`) runs layout/RV
inference passes over the `KStmt` body, types global params for TMA
descriptors, synthesizes helper templates, and wraps the kernel in
`#if defined(KITTENS_HOPPER)` (or the family's guard) with
`#include <kittens.cuh>`. `Tyr/GPU/Codegen/FFI.lean` generates the C++
launcher wrappers (`borrowTensor` extraction, CUDA-error checks) and
`writeKernelCudaUnitsByModuleFrom` (`FFI.lean:282`) writes one
`<Module_Name>.cu` per Lean module with change-detection (unchanged files are
not rewritten; stale `.cu` files are removed unless `--no-clean`).

```bash
lake exe GenerateGpuKernels [--out-dir cc/src/generated] [--no-clean] <Module>…
```

The normal build does this for you: the `target libtyr` in `lakefile.lean`
builds the `GenerateGpuKernels` executable, runs it over the configured
modules, then invokes
`make -C cc`. `-Kkernels="<module> ..."` (Lake option, set with `lake -R`)
selects the kernel module(s) to emit CUDA for (default
`Tyr.GPU.Kernels.MhaH100`); see [Getting started](../getting-started.md).

### Kernel-level autodiff (`Tyr/GPU/AutoGrad.lean`)

A symbolic AD pass over the kernel IR, in namespace `Tyr.GPU.AD`:

```lean
inductive LinearInst where | id | add | sub | mul | … | mma | loop | custom  -- :13

abbrev TraceM := StateT TraceState ADM                                        -- :61

partial def linearizeStmt (s : KStmt) : TraceM Unit                           -- :80
partial def transposeTrace (trace : Array LinearInst) : ADM (Array KStmt)     -- :227
```

`linearizeStmt` copies the primal statements and records a `LinearInst` trace
(JVP data); `transposeTrace` walks the trace in reverse emitting VJP `KStmt`s
in accumulate-into-cotangent style. It covers binary/unary tile ops,
broadcasts, reductions (including masked max/min/prod), all four
`MMATranspose` modes, and loops (reversed in the VJP). Custom rules hook in
through `LinearInst.custom`, resolved against `gpuAdRegistry` via
`registerGpuVJPRule` (`Tyr/AutoGrad.lean:726`). Status: exercised by
`Tests/TestGPUAutoGrad.lean` only — no `@[gpu_grad]`-style attribute exists,
and nothing in the repository registers custom GPU rules today.

`Tyr/GPU/AD.lean` is a *different* system sharing the same namespace:
handwritten JVP/VJP rules for Lean-IR `torch.add`/`sub`/`mul`/`matmul` ops
registered by `Tyr.GPU.AD.init`. That `init` has no call sites in the repo;
treat the file as dormant.

### TileIR backend (`Tyr/GPU/Codegen/TileIR/`, experimental)

An independent backend targeting NVIDIA's TileIR (MLIR) toolchain rather than
ThunderKittens: its own `Type`/`Expr` AST, a `Builder` do-DSL and `Frontend`
custom elaborator, `Render` to MLIR text, `Passes`, and a `Toolchain` driver
that shells out to `cuda-tile-opt`, `cuda-tile-translate`, and `tileiras`
(found via `PATH` or the `CUDA_TILE_OPT`/`CUDA_TILE_TRANSLATE`/`TILEIRAS`
environment variables). Declarations whose result type is
`Tyr.GPU.Codegen.TileIR.Module` can be marked `@[tileir_kernel]`
(`TileIR/Attribute.lean:37`) and compiled with
`lake exe GenerateTileIRKernels`. This path does not
share the `KStmt` IR or the `KernelM` surface above.

## Key APIs

Authoring surface (all in `Tyr.GPU.Codegen`, re-exported by
`Tyr/GPU/Kernels/Prelude.lean`):

| API | Location | Purpose |
|---|---|---|
| `KernelM`, `emit`, `freshVar`, `comment` | `Codegen/Monad.lean` | Builder monad and raw IR emission |
| `buildKernelM` | `Codegen/Monad.lean:125` | Seal a `KernelM Unit` into a `Kernel` |
| `setArch`, `setFamily`, `setLaunchBounds` | `Codegen/Monad.lean` | Target/family guards, `__launch_bounds__` |
| `RT`/`ST`/`RV`/`SV`/`GPtr`/`KVal`/`TT` | `Codegen/TileTypes.lean` | Typed phantom handles |
| `allocRT`/`allocST`/`zeroRT`/`allocRV` | `Codegen/Primitives.lean` | Tile/vector allocation |
| `load`/`store`, `loadGlobal`/`storeGlobal` | `Primitives.lean`, `GlobalLayout.lean` | Shared↔register and global↔shared movement |
| `mma`/`mm`/`mmaT` | `Primitives.lean:608-644` | Tensor-core MMA with checked dims |
| `sync` | `Primitives.lean:1769` | Block barrier |
| `blockCoord2D`, `RTileCoord.withRow/withCol` | `GlobalLayout.lean:211,526` | Block tile coordinates |
| `krange`/`kvrange`/`kstride` | `Loop.lean:75-83` | `for`-loop ranges emitting device loops |
| `matmul`, `⬝`, `⬝ᵀ` | `Notation.lean:96,192` | Expression-style MMA sugar |
| `DoubleBuffer`, `onlineSoftmax`, `attentionBlockIter` | `Macros.lean` | Common kernel patterns |
| `GpuCapabilities`, `RequiresTMA/WGMMA/TMEM` | `GPU/Capabilities.lean` | Feature gates at instance resolution |

Codegen/build surface:

| API | Location | Purpose |
|---|---|---|
| `@[gpu_kernel arch]` / `@[gpu_kernel]` | `Codegen/Attribute.lean:258` | Register a kernel; generates `.kernel` + `.launch` |
| `<decl>.launch` | generated | FFI launcher: tensors, scalars, grid, block, sharedMem, stream |
| `#generate_gpu_kernels` | `Codegen/Attribute.lean:775` | List companion constants in the environment |
| `generateKernel` | `Codegen/EmitNew.lean:1654` | `Kernel → String` ThunderKittens CUDA |
| `writeKernelCudaUnitsByModuleFrom` | `Codegen/FFI.lean:282` | Per-module `.cu` emission with change detection |
| `lake exe GenerateGpuKernels` | `Tyr/GPU/GenerateGpuKernels.lean` | CLI driver used by the Lake build |
| `-Kkernels` | `lakefile.lean` (`gpuKernels`, `target libtyr`) | Build-time kernel module selection |

## Usage example

Reconstructed example (from `Tyr/GPU/Kernels/Examples.lean:57-65` and `Examples/GPU/RunCopy.lean:20-34`).

Kernel side — declare the body, let the attribute derive the companions:

```lean
import Tyr.GPU.Kernels.Prelude

namespace Tyr.GPU.Kernels.Examples
open Tyr.GPU Tyr.GPU.Codegen

@[gpu_kernel .SM90]
def simpleGemm
    (aPtr : GPtr GpuFloat.BFloat16)
    (bPtr : GPtr GpuFloat.BFloat16)
    (cPtr : GPtr GpuFloat.Float32)
    (m : KVal UInt64) (n : KVal UInt64) (k : KVal UInt64)
    : KernelM Unit := do
  let coord ← blockCoord2D
  let a  : RT GpuFloat.BFloat16 64 64      ← allocRT .BFloat16 64 64
  let b  : RT GpuFloat.BFloat16 64 64 .Col ← allocRT .BFloat16 64 64 .Col
  let c  : RT GpuFloat.Float32 64 64       ← zeroRT .Float32 64 64
  let aS : ST GpuFloat.BFloat16 64 64      ← allocST .BFloat16 64 64
  let bS : ST GpuFloat.BFloat16 64 64 .Col ← allocST .BFloat16 64 64 .Col
  let cS : ST GpuFloat.Float32 64 64       ← allocST .Float32 64 64
  for kBlk in krange 0 8 do
    loadGlobal aS aPtr (coord.withCol kBlk.id)
    loadGlobal bS bPtr (coord.withRow kBlk.id)
    sync
    load a aS; load b bS
    mma c a b c          -- dims/layouts checked by Lean; emits kittens::mma_AB
    sync
  store cS c
  storeGlobal cPtr cS coord

end Tyr.GPU.Kernels.Examples
```

Build side — emit CUDA for the module (or configure `-Kkernels` and let
`lake build` do it):

```bash
lake exe GenerateGpuKernels Tyr.GPU.Kernels.Examples
# writes cc/src/generated/Tyr_GPU_Kernels_Examples.cu
```

Host side — launch through the generated FFI wrapper. The pattern below is
exactly how `RunCopy` drives `copy64x64.launch`; grid/block/sharedMem/stream
are the flattened trailing arguments the attribute appends:

```lean
import Tyr.Torch
import Tyr.GPU.Kernels.Copy
open torch Tyr.GPU.Kernels

-- input : T #[1, 1, 64, 64] (CUDA); output := torch.zeros_like input
let stream ← torch.cuda_current_stream
copy64x64.launch input output 1 1 1  128 1 1  0 stream
--                            grid    block   smem stream
let _ ← torch.cuda_synchronize
```

Scratch/output buffers must be allocated through `IO` (as above); kernels
mutate the tensors you pass in, and pure-looking `let` bindings can alias
through CSE — see the caveat in `Tyr/GPU/Ops/RKFused.lean`.

## Related guides

- [GPU kernels and ops](kernels.md) — the kernel catalog built with this DSL and the typed ops layer above the raw launchers
- [ThunderKittens porting status](thunderkittens-porting-status.md) — source-to-Lean parity matrix and codegen build notes
- [Getting started](../getting-started.md) — build instructions, including `-Kkernels`
- [Autodiff](../autodiff.md) — the host-side `Tyr.AutoGrad` engine that `Tyr/GPU/AutoGrad.lean` plugs into
- [FFI and build](../ffi-and-build.md) — the `cc/` C++ layer the generated `.cu` files compile into
- [Examples and testing](../examples-and-testing.md) — `Examples/GPU/Run*.lean` harnesses and GPU test entry points

Exhaustive symbol documentation for every module mentioned here is generated
by doc-gen4 from the source docstrings (see `docbuild/`); this chapter is a
guide, not a reference.
