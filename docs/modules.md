# Neural network modules

## Purpose and when to use

`Tyr/Module/` is the layer library: shape-indexed structures (`Linear`, `LayerNorm`, `RMSNorm`) whose parameters live in the `TensorStruct` tree, plus the `deriving TensorStruct` / `deriving Model` handlers that make your own model structures traversable. `Tyr/Modular/` is an optimization-theory layer on top: it implements the modular-norm framework from "Scalable Optimization in the Modular Norm" (NeurIPS 2024, arXiv:2405.14813), manifold-constrained linear layers (Stiefel/Grassmann/Orthogonal/Hyperbolic), low-rank metric factors with Woodbury solves, and a sensitivity-to-learning-rate budget compiler. Reach for `Tyr/Module` whenever you define a model; reach for `Tyr/Modular` when you want width/depth-robust learning rates or manifold-native updates.

## Architecture and main abstractions

Everything in `Tyr/Module/` lives in the `torch` namespace; everything in `Tyr/Modular/` lives in `Tyr.Modular`.

### Tyr/Module: layers and the Module classes

`Tyr/Module/Core.lean:77` defines three Equinox-style typeclasses over the `TensorStruct` parameter-tree substrate:

```lean
class Module (M : Type) (In : Type) (Out : Type) where
  [toTensorStruct : TensorStruct M]
  forward : M → In → Out

class ModuleIO (M : Type) (In : Type) (Out : Type) where
  [toTensorStruct : TensorStruct M]
  forward : M → In → IO Out

class ModuleCtx (M : Type) (Ctx : Type) (In : Type) (Out : Type) where
  [toTensorStruct : TensorStruct M]
  forward : M → Ctx → In → IO Out
```

A blanket instance lifts any `Module` to `ModuleIO` (`Core.lean:103`). Application syntax is scoped infix: `m |> x` (pure) and `m |>! x` (IO), defined via `Module.apply` / `ModuleIO.apply` (`Core.lean:129-140`). `TrainingCtx` (`Core.lean:148`) bundles a `training : Bool` flag and `dropout_p : Float` with `train` / `eval` / `withDropout` constructors for context-passing modules.

The concrete layers are plain structures indexed by `UInt64` dimensions, deriving `TensorStruct` and `ToModuleDisplay` (widget rendering, see `Tyr/Widget.lean:805`):

```lean
structure Linear (in_dim out_dim : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  weight : Tensor ⟨#[out_dim, in_dim], dtype, device⟩
  bias : Option (Tensor ⟨#[out_dim], dtype, device⟩) := none
  deriving TensorStruct, ToModuleDisplay

structure LayerNorm (dim : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  weight : Tensor ⟨#[dim], dtype, device⟩
  bias : Tensor ⟨#[dim], dtype, device⟩
  eps : Static Float := ⟨1e-5⟩
  deriving TensorStruct, ToModuleDisplay

structure RMSNorm (dim : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  weight : Tensor ⟨#[dim], dtype, device⟩
  eps : Static Float := ⟨1e-6⟩
  deriving TensorStruct, ToModuleDisplay
```

`Linear.init` uses Kaiming/He initialization (`std = sqrt(2 / in_dim)`) and marks every parameter as a trainable autograd leaf (`Linear.lean:23-35`); `LayerNorm.init` / `RMSNorm.init` do the same with ones/zeros. Each layer exposes explicit `forwardNd` functions plus `Module` instances for the common arities (see the API table below).

`Affine.lean` provides an indexed affine map with `[input, output]` weights, `TensorStruct` and `Module` instances, and typed gradient extraction. Its forward operation is `input @ weight + bias`; `Linear` uses the transposed weight convention instead. `Conv2d.lean` has runtime hyperparameters with indexed weight and bias fields; initialization validates grouped-channel dimensions and retains dtype and device. It currently has no model call sites.

### Deriving parameter-tree instances

`Tyr/Module/Derive.lean` registers two deriving handlers (`Derive.lean:212-216`):

- `deriving TensorStruct` — generates field-wise `map` / `mapM` / `zipWith` / `fold` for a structure. Every field type must already have a `TensorStruct` instance; `Tyr/TensorStruct.lean` provides them for `Tensor σ`, `Option α`, `Array α`, `List α`, `Vector n α`, `Static α`, and `Frozen σ`; `Derive.lean` adds the pair instance `α × β`. Parametric structures (shape or type parameters) are supported.
- `deriving Model` — one-shot derivation of `TensorStruct` + `ToTensorStructSchema` + `TensorStructFlatten` (`Derive.lean:200-210`; `Model` is a marker class at `Tyr/TensorStruct.lean:243`, the two schema handlers come from `Tyr/AD/TensorStructSchema.lean`). Use it when the structure should also serialize/flatten, e.g. for checkpointing.

For hand-written instances, `mapTensor` / `mapFrozen` / `mapStatic` (`Derive.lean:64-71`) are small helpers for the common field kinds.

### Tyr/Modular: normed modules and budgets

The core abstraction is `NormedModule` (`Tyr/Modular/Norm.lean:47`):

```lean
class NormedModule (M : Type) extends TensorStruct M where
  norm : M → Float          -- modular norm on weights, ‖w‖_M
  dualNorm : M → Float      -- dual norm for gradients, ‖g‖_M*
  nu : M → Float            -- input sensitivity bound ‖∂f/∂x‖
  mu : M → Float            -- weight sensitivity bound ‖∂f/∂w‖
  normalize : M → M         -- scale to unit modular norm
  normalizeDual : M → M     -- normalize in the dual norm (for updates)
```

Derived helpers in the same namespace: `isWellNormed`, `lipschitzConstant` (`ν · μ`), `scale`, and `normalizedUpdate lr grad` — the core modular update step `-lr · normalizeDual grad` (`Norm.lean:72-89`).

Atomic instances (`Tyr/Modular/Atomic.lean`): `Linear` uses the spectral norm on the weight (dual: nuclear norm), `LayerNorm` uses joint ℓ₂ of `(γ, β)`, and a locally-defined `Embedding vocabSize dim` (`Atomic.lean:128`) uses the max row norm. Trivial instances exist for `Float` and `Static α` (`Norm.lean:97-120`).

Composition is recursive (`Tyr/Modular/Compose.lean`):

- `structure Sequential (M₁ M₂ : Type) where first second` (`Compose.lean:31`) marks M₂-after-M₁ composition. Its norm is `max(ν₂·‖w₁‖, ‖w₂‖)`, sensitivities multiply (`ν = ν₁·ν₂`), and `μ = max(μ₁·ν₂, μ₂)` — the paper's formulas (`Compose.lean:67-94`). Build one with `compose m₁ m₂` or the `⊛` infix (`Compose.lean:199-201`).
- The plain product `M₁ × M₂` is parallel composition with ℓ₂ norm combination and `max` sensitivity (`Compose.lean:109-144`).
- `Array M` and `torch.Vector n M` aggregate norms by ℓ₂ but multiply `nu` across elements (`Compose.lean:155-194`) — sequential sensitivity with parallel norm, which the file header itself flags as a mixture.

The budget compiler (`Tyr/Modular/Budget.lean`) turns sensitivities into per-layer learning-rate multipliers:

```lean
structure BudgetConfig where
  epsilon : Float := 1e-6
  minMultiplier : Float := 1e-3
  maxMultiplier : Float := 1e3
  globalScale : Float := 1.0

-- For layer i of a sequential stack: sensitivity = μᵢ · Π_{j>i} νⱼ,
-- multiplier = clamp(globalScale / sensitivity)
def sequentialDownstreamScales [NormedModule M]
    (cfg : BudgetConfig) (modules : Array M) : Array Float

def budgetedSequentialLRs [NormedModule M]
    (cfg : BudgetConfig) (baseLR : Float) (modules : Array M) : Array Float
```

`GroupBudget` (`Budget.lean:59`) holds per-family multipliers (`matrix`, `embedding`, `lmHead`, `scalar`) with `ofSensitivities` and `applyToBaseLR`.

### Manifold-constrained layers

`Tyr/Modular/Manifold.lean` lets a linear layer's weight *live on* a matrix manifold instead of being projected after the fact. The carrier interfaces are:

```lean
class MatrixManifoldCarrier (M : UInt64 → UInt64 → Type) where
  /-- Project an ambient matrix to the manifold, given a legality witness. -/
  project : {m n : UInt64} → MatrixShapeOK m n → AmbientMatrix m n → M m n
  /-- Convert manifold value to its ambient matrix representation. -/
  toMatrix : {m n : UInt64} → M m n → AmbientMatrix m n
  /-- Replace the ambient contents of an existing manifold value, keeping its
      shape witnesses. Does not re-project; the caller asserts validity. -/
  reAmbient : {m n : UInt64} → M m n → AmbientMatrix m n → M m n
  /-- Re-project onto the manifold, reusing an existing value's witnesses. -/
  projectLike : {m n : UInt64} → M m n → AmbientMatrix m n → M m n
  /-- Take one geometry-aware step and retract on-manifold. -/
  dualMapStep : {m n : UInt64} → M m n → AmbientMatrix m n → Float → M m n
```

`MatrixShapeOK m n` is the explicit proof that the ambient matrix has at least
as many rows as columns. Projection takes this proof directly; `projectLike`
reuses the witnesses of an existing value. Ambient matrices/vectors are indexed
Float32 tensors. Carrier instances support Stiefel, Grassmann, and orthogonal
matrices, with a separate vector-carrier interface for hyperbolic parameters.
The layer types:

```lean
structure ManifoldLinear (M : UInt64 → UInt64 → Type) [MatrixManifoldCarrier M]
    (in_dim out_dim : UInt64) where
  weight : M out_dim in_dim
  bias : Option (Tensor { shape := #[out_dim], dtype := .Float32 }) := none

structure ManifoldVectorParam (V : UInt64 → Type) [VectorManifoldCarrier V] (n : UInt64) where
  value : V n
```

`ManifoldLinear.init` projects random weights onto the manifold and re-wraps them as trainable leaves; `forward2d` runs through `toMatrix`; `applyDualMapUpdate` takes a geometry-aware `dualMapStep` on the weight and a plain Euclidean step on the bias (`Manifold.lean:131-169`). Aliases: `StiefelLinear`, `GrassmannLinear`, `OrthogonalLinear dim` (square), `HyperbolicVector n` (`Manifold.lean:303-312`). Both types have `TensorStruct` and `NormedModule` instances (spectral-norm based, like `Linear`).

`ManifoldUpdatable` (`Manifold.lean:324`) is a composable update interface — `Cotangent : P → Type` plus `applyUpdate : (p : P) → Cotangent p → Float → P` — with instances for both manifold types, pairs, and `Sequential`, so tree-shaped parameter structures get geometry-aware steps via `ManifoldUpdatable.step`. `applyMatrixBudgetFromModules` (`Manifold.lean:367`) compiles modular sensitivities of a module array into the `matrix` budget multiplier of a `torch.Optim.DualOptimizer.Config` and sets `useModularBudget := true`.

### Metric factors and the Riemannian recursion

`Tyr/Modular/MetricFactor.lean` provides the linear algebra for low-rank pullback metrics:

- `MetricFactor rank dim` (default Float32 and unrestricted device policy) wraps `matrix : Tensor { shape := #[rank, dim], dtype := .Float32 }`, a factor `K` of the PSD metric `KᵀK`.
- `DiagonalMass dim` with the same defaults wraps `diag : Tensor { shape := #[dim], dtype := .Float32 }` (`ones`, `ofScalar`, `toMatrix`, `invDiag`, `apply`, `applyInv`).
- Operations: `pullback` along a Jacobian, `gram` (`KᵀK`), `denseMetric D K = D + KᵀK`, `apply` / `applyTranspose` / `applyRegularized`, and `solveWoodbury D K g` solving `(D + KᵀK)x = g` via the Woodbury identity with inner matrix `I + K D⁻¹ Kᵀ` (`MetricFactor.lean:109-119`).

`Tyr/Modular/RiemannianModule.lean:32` turns a leaf module into a tape-based, locally-linearizable unit for recursive metric-factor propagation:

```lean
class RiemannianModule (M : Type) (inDim outDim : UInt64) where
  paramDim : UInt64
  Tape : Type
  forwardWithTape : M → Tensor { shape := #[inDim], dtype := .Float32 } → Tensor { shape := #[outDim], dtype := .Float32 } × Tape
  localLinearization : M → Tape → LocalLinearization inDim paramDim outDim
  paramMass : M → DiagonalMass paramDim
  applyAmbientUpdate : M → Tensor { shape := #[paramDim], dtype := .Float32 } → Float → M
```

`LocalLinearization` (`RiemannianModule.lean:13`) bundles operator-form Jacobians (`applyA`, `applyAT`, `applyB`, `applyBT`) plus dense `materializeA` / `materializeB` for tests and factor pullbacks via row-wise adjoints. Instances exist for `torch.Linear` (parameter dim `out·in + out`, tape = input vector) and `ManifoldLinear` (ambient update re-projects onto the manifold). `backwardLeafStep` performs one layer's metric solve + update + factor/cotangent propagation and returns `LayerStepDiagnostics` (gradient/update/residual norms, inner condition estimate); `sequentialForwardWithTape` / `sequentialBackwardMetricStep` handle two-layer `Sequential` compositions. These are the building blocks consumed by `Tyr.Optim.RiemannianSGD`.

## Key APIs

Built-in layers (all in namespace `torch`):

| Layer | Init | Forward functions | `Module` instances |
|---|---|---|---|
| `Linear in_dim out_dim` | `Linear.init (in_dim out_dim) (withBias := true) : IO (Linear in_dim out_dim)` | `forward2d : Tensor { shape := #[b,i], dtype := .Float32 } → Tensor { shape := #[b,o], dtype := .Float32 }`, `forward3d : Tensor { shape := #[b,s,i], dtype := .Float32 } → Tensor { shape := #[b,s,o], dtype := .Float32 }` | 2D and 3D |
| `LayerNorm dim` | `LayerNorm.init (dim) (eps := 1e-5)`, `LayerNorm.initNoAffine` | `forward3d : Tensor { shape := #[b,s,d], dtype := .Float32 } → Tensor { shape := #[b,s,d], dtype := .Float32 }` | 3D only |
| `RMSNorm dim` | `RMSNorm.init (dim) (eps := 1e-6)` | `forward2d`, `forward3d`, `forward4d`, `forward5d` | 2D and 3D |

Note: `RMSNorm.forward5d` (`RMSNorm.lean:48`) is misnamed — it is a 4D forward on the attention layout `[batch, n_head, seq, head_dim]`; `forward4d` uses `[batch, seq, n_head, head_dim]`. Neither 4D variant has a `Module` instance.

`NormedModule` instances and their norms (`Tyr/Modular/`):

| Instance | `norm` | `dualNorm` | `nu` |
|---|---|---|---|
| `Linear i o` | spectral ‖W‖σ (max with ‖b‖₂) | nuclear ‖W‖* + ‖b‖₂ | ‖W‖σ |
| `LayerNorm d` | ℓ₂ of (γ, β) | ℓ₂ (self-dual) | 1 |
| `Embedding v d` (in `Tyr.Modular`) | max row norm | Σ row norms | max row norm |
| `Sequential M₁ M₂` | `max(ν₂·n₁, n₂)` | `ν₂·d₁ + d₂` | `ν₁ · ν₂` |
| `M₁ × M₂` | `√(n₁² + n₂²)` | `√(d₁² + d₂²)` | `max(ν₁, ν₂)` |
| `Array M`, `Vector n M` | ℓ₂ over elements | ℓ₂ over elements | `Π νᵢ` |

Caveat: `normalizeDual` for `Linear` and `ManifoldLinear` uses Frobenius normalization as an approximation of the true nuclear-norm dual projection — the code says so only in a comment (`Atomic.lean:61-63`, `Manifold.lean:270-274`).

Budget and manifold entry points:

```lean
-- Tyr/Modular/Budget.lean
def multiplierFromSensitivity (cfg : BudgetConfig) (sensitivity : Float) : Float
def applyBudget (baseLR multiplier : Float) : Float
def GroupBudget.ofSensitivities (cfg : BudgetConfig)
    (matrix embedding lmHead scalar : Float) : GroupBudget

-- Tyr/Modular/Manifold.lean
def ManifoldLinear.init [MatrixManifoldCarrier M] (in_dim out_dim : UInt64)
    (withBias : Bool := true) (h : MatrixShapeOK out_dim in_dim := by decide)
    : IO (ManifoldLinear M in_dim out_dim)
def ManifoldLinear.applyDualMapUpdate [MatrixManifoldCarrier M] {in_dim out_dim : UInt64}
    (lin : ManifoldLinear M in_dim out_dim)
    (weightGrad : Tensor { shape := #[out_dim, in_dim], dtype := .Float32 }) (biasGrad? : Option (Tensor { shape := #[out_dim], dtype := .Float32 }))
    (lr : Float) : ManifoldLinear M in_dim out_dim
def ManifoldUpdatable.step [ManifoldUpdatable P]
    (params : P) (grad : ManifoldUpdatable.Cotangent params) (lr : Float) : P
def applyMatrixBudgetFromModules [NormedModule M]
    (optCfg : torch.Optim.DualOptimizer.Config)
    (budgetCfg : BudgetConfig := {}) (modules : Array M)
    : torch.Optim.DualOptimizer.Config
```

## Usage examples

### Model structure with derived traversal

Reconstructed example (from `Examples/AlphaGradPort/PolicyTrain.lean:113-134`):

```lean
import Tyr.Module

open torch

structure PolicyNet (obsDim hidden actionDim : UInt64) where
  fc1 : Linear obsDim hidden
  policyHead : Linear hidden actionDim
  valueHead : Linear hidden 1
  deriving TensorStruct

def PolicyNet.init (obsDim hidden actionDim : UInt64)
    : IO (PolicyNet obsDim hidden actionDim) := do
  let fc1 ← Linear.init obsDim hidden true
  let policyHead ← Linear.init hidden actionDim true
  let valueHead ← Linear.init hidden 1 true
  pure { fc1, policyHead, valueHead }

def PolicyNet.forward {batch obsDim hidden actionDim : UInt64}
    (net : PolicyNet obsDim hidden actionDim) (x : Tensor { shape := #[batch, obsDim], dtype := .Float32 })
    : Tensor { shape := #[batch, actionDim], dtype := .Float32 } × Tensor { shape := #[batch, 1], dtype := .Float32 } :=
  let h := Tensor.relu (Linear.forward2d net.fc1 x)
  (Linear.forward2d net.policyHead h, Linear.forward2d net.valueHead h)
```

With `deriving TensorStruct`, generic tree operations (`TensorStruct.map` for detach/scale, `mapM` for initialization, `zipWith` for gradient accumulation, `fold` for counting) work on `PolicyNet` without further boilerplate. Larger models nest freely — `Examples/GPT/GPT.lean:75-85` derives `TensorStruct` for a `GPTParams` structure containing an `Array` of block structures.

### Modular norms and LR budgets

Reconstructed example (from `Tests/TestModularNorm.lean:118-152` and `Tests/TestModularManifold.lean:72-90`):

```lean
import Tyr.Modular

open torch Tyr.Modular

-- Sequential composition: norms and sensitivities combine by the paper's formulas
let l1 ← Linear.init 10 20
let l2 ← Linear.init 20 5
let net := compose l1 l2                 -- Sequential (Linear 10 20) (Linear 20 5)
let n := NormedModule.norm net           -- max(ν₂·‖l1‖, ‖l2‖)
let ν := NormedModule.nu net             -- ν₁ · ν₂

-- Compile per-layer LR multipliers from modular sensitivities
let scales := sequentialDownstreamScales ({} : BudgetConfig) #[net]

-- Inject the budget into the dual-optimizer config
let baseCfg : torch.Optim.DualOptimizer.Config := {}
let cfg' := applyMatrixBudgetFromModules baseCfg {} #[net]
-- cfg'.useModularBudget == true, cfg'.budget.matrix is the compiled multiplier
```

### Manifold-constrained linear layer

Reconstructed example (from `Tests/TestModularManifold.lean:12-22`):

```lean
import Tyr.Modular
import Tyr.Manifolds

open torch Tyr.Modular

def manifoldStepExample : IO Unit := do
  let layer ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 8 16 true
  let gW ← Tensor.randn #[16, 8] false
  let gB ← Tensor.randn #[16] false
  let layer' := ManifoldLinear.applyDualMapUpdate layer gW (some gB) 0.02
  let w := MatrixManifoldCarrier.toMatrix (M := Tyr.AD.Stiefel) layer'.weight
  let wtW := Tensor.mm (Tensor.transpose2d w) w
  IO.println s!"Gram shape: {wtW.runtimeShape}" -- #[8, 8], approximately identity
```

## Caveats and rough edges

- The `Module` / `ModuleIO` / `ModuleCtx` classes are thinly adopted: instances exist for the three built-in layers and two audio encoders (`Tyr/Model/Qwen3ASR/AudioEncoder.lean:470`, `Tyr/Model/Qwen3TTS/SpeakerEncoder.lean:314`), and the `|>` / `|>!` notations and `TrainingCtx` have no production call sites. Most model code calls `Linear.forward2d`-style functions directly, as in the examples above.
- The `Tyr/Module.lean` umbrella re-exports `Core`, `Derive`, `Linear`, `LayerNorm`, `Affine`, and `Conv2d` — but **not** `RMSNorm`. Write `import Tyr.Module.RMSNorm` explicitly (this is what `Tyr/Model/Qwen/Model.lean:10` and others do).
- `Affine` has indexed parameters and typed module/gradient interfaces, currently exercised by tests. `Conv2d` has indexed parameters and checked initialization, but no `TensorStruct`/`Module` instances or model call sites.
- There is no shared `Embedding` in `Tyr/Module`; the embedding table used for modular norms is `Tyr.Modular.Embedding` and is currently only exercised by tests. Its indexed weights preserve dtype and device policy, and `Embedding.forward` requires Int64 token tensors. The underlying typed `Tensor.embedding` and `Tensor.embedding1d` operations are also used by GPT. There is no `NormedModule` instance for `RMSNorm`.

## Related guides

- [Tensors](core/tensors.md) — `Tensor σ`, shapes, and the tensor operations used by layer forwards
- [TensorStruct](core/tensorstruct.md) — the parameter-tree class that `deriving TensorStruct` generates
- [Autodiff](autodiff.md) — autograd leaves, `requires_grad`, and gradient extraction
- [Serialization](serialization.md) — checkpointing structures derived with `Model`
- [Optimization](optimization.md) — `DualOptimizer` budgets, Muon, and the Riemannian optimizers that consume `RiemannianModule`
- [LLM model families](models/llms.md) — Qwen and other models built from these layers

Exhaustive symbol-level documentation for everything in this chapter is generated by doc-gen4 (see `docbuild/`); this guide covers the concepts and the main entry points only.
