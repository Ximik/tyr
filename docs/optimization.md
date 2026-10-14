# Optimization and manifold geometry

## Purpose & when to use

`Tyr/Optim/` is Tyr's optimizer stack: an Optax-style core of composable
`GradientTransformation`s (`adamw`, `sgd`, `chain`, `scale_by_schedule`), learning-rate
schedules, the nanochat/modded-nanogpt-derived first-order optimizers (NorMuon with Polar
Express orthogonalization, DistAdam, DualOptimizer), and experimental Riemannian
optimizers (ManifoldMuon, RiemannianSGD, RiemannianTreeSGD). `Tyr/Manifolds/` provides the
differential-geometry layer those Riemannian pieces build on: a `DifferentiableManifold`
typeclass plus a dozen concrete manifold families over shape-typed tensors. Use the Optax
core for ordinary training over `TensorStruct` parameter trees; reach for the manifold
layer when parameters carry constraints (orthonormal columns, unit norm, positive
definiteness).

## Architecture & main abstractions

The optimizer files under `Tyr/Optim/` declare `namespace torch.Optim.*` (e.g.
`torch.Optim.NorMuon`), while the manifold files under `Tyr/Manifolds/` all declare
`namespace Tyr.AD`. `Tyr/Optim.lean` is the Optax-style core; `Tyr/Manifolds.lean` is an
umbrella import for the whole geometry layer.

### The Optax-style core (`Tyr/Optim.lean`, namespace `torch.Optim`)

An optimizer is an explicit state transition over a tensor structure
(`Tyr/Optim.lean:105`):

```lean
structure GradientTransformation (α : Type) (S : Type) where
  /-- Initialize optimizer state from model structure -/
  init : α → S
  /-- Transform gradients: (params, grads, state) → (updates, new_state) -/
  update : α → α → S → (α × S)
```

`update` returns *transformed gradients*, not new parameters — that is what makes
transformations composable via `chain` (`Tyr/Optim.lean:173`). State types (`EmptyState`,
`ScaleByAdamState α`, `TraceState α`, `ChainState S1 S2`, `ScaleByScheduleState`) all have
`TensorStruct` instances, so optimizer state can be mapped, moved between devices, and
checkpointed with the same machinery as model parameters (see
[tensorstruct.md](core/tensorstruct.md)).

Built-in pieces (`Tyr/Optim.lean:114-297`):

- Primitives: `scale`, `scale_by_adam`, `add_decayed_weights`, `trace`.
- Composed optimizers: `adamw` = `scale_by_adam → add_decayed_weights → scale(-lr)`
  (`:190`), `sgd`, `sgd_momentum`.
- Application: `apply_updates` (detach, add, re-enable `requires_grad`, `:213`) and
  `step opt params grads state : (α × S)` (`:223`).
- Schedule integration: `Schedule := Nat → Float` (`:235`), `scale_by_schedule`, and
  schedule-aware variants `adamw_schedule`, `sgd_schedule`, `sgd_momentum_schedule`.

Gradient clipping is not implemented in the core — the section is an explicit TODO
(`Tyr/Optim.lean:299`); only `RiemannianTreeSGD` carries a private `clipScale`, and
training loops clip inline (e.g. `clipGPTGrads` in `Examples/GPT/Train.lean`).

### Schedules (`Tyr/Optim/Scheduler.lean`, namespace `torch.Optim.Scheduler`)

Schedules are bare functions `Nat → Float`, so they compose without any framework support.
Constructors: `constant`, `cosine_decay`, `linear_decay`, `step_decay`, `one_cycle`,
`warmup_plateau_cosine`, `exponential_decay`, `polynomial_decay`, `warmup`
(`Scheduler.lean:28-112`). Combinators: `join`, `sequence`, `scale_schedule`, `multiply`,
`add`, `clamp` (`Scheduler.lean:121-145`), plus weight-decay schedules
`linear_weight_decay` / `cosine_weight_decay`. A legacy config-struct API
(`ScheduleConfig`, `getLr`, `*.toSchedule`, `Scheduler.lean:165-294`) duplicates every
modern constructor; prefer the function form.

### The manifold layer (`Tyr/Manifolds/Basic.lean`, namespace `Tyr.AD`)

The central abstraction is a point-dependent bundle typeclass
(`Tyr/Manifolds/Basic.lean:36`):

```lean
class DifferentiableManifold (M : Type) where
  Tangent : M → Type
  Cotangent : M → Type
  zeroTangent (x : M) : Tangent x
  zeroCotangent (x : M) : Cotangent x
  addTangent {x : M} : Tangent x → Tangent x → Tangent x
  addCotangent {x : M} : Cotangent x → Cotangent x → Cotangent x
  scaleTangent {x : M} (s : Float) : Tangent x → Tangent x
  sharp {x : M} : Cotangent x → Tangent x      -- metric ♯
  flat {x : M} : Tangent x → Cotangent x       -- metric ♭
  exp (x : M) : Tangent x → M
  retract (x : M) : Tangent x → M := exp x     -- cheaper exp approximation
```

Three ways in:

- `EuclideanSpace M` (`zero`/`add`/`smul`/`inner`, `Basic.lean:75`) gets a free
  `DifferentiableManifold` instance with `sharp`/`flat := id` and `exp x v := x + v`.
  Instances exist for `Float`, `Static α` (0-dimensional), and any
  `[TensorStruct α] [Inhabited α]` as a product manifold (`Basic.lean:101-129`).
- `DifferentiableManifold.gradientStep x grad lr` (`Basic.lean:134`) is the generic
  Riemannian gradient step: `retract x (-lr • sharp grad)`.
- Concrete manifold families are structures wrapping one shape-typed tensor with a
  matching tangent structure and `project`/`random` constructors (table below).
  The table shows default Float32/unrestricted-placement tensors; manifold
  carriers also retain explicit dtype and device-policy parameters.

`abbrev Differentiable := DifferentiableManifold` (`Basic.lean:143`) is a
backward-compatibility alias.

Concrete families (all in `Tyr.AD`, one file each under `Tyr/Manifolds/`):

| Type | Constraint | Retraction / exp |
|---|---|---|
| `Stiefel n p` | `XᵀX = Iₚ`, `matrix : DTensor #[n, p] .Float32` | QR of `X + Z` |
| `Orthogonal n` | `QᵀQ = I`, tangent stored as `SkewSymmetric n` | `Q · exp(S)` via `linalg.matrix_exp`, then QR |
| `Grassmann n p` | `XᵀX = Iₚ` modulo `O(p)`, tangent `XᵀZ = 0` | QR of `X + Z` |
| `Sphere n` | unit-norm `coords : DTensor #[n] .Float32` | geodesic `exp`; normalize for `retract` |
| `Oblique m n` | unit-norm columns | column normalization |
| `Positive m n` | elementwise-positive | second-order step `x + v + v²/(2x)`, then clamp; `sharp` scales by `x²` |
| `SymmetricPositiveDefinite n` | SPD | log-Euclidean: `project(matrix_log X + v)` |
| `PSDFixedRank n k` | `Y Yᵀ` factor, `Y : DTensor #[n, k] .Float32` | addition in factor space |
| `Elliptope n k` | row-wise unit norms | row normalization |
| `FixedRankEmbedded m n k` | rank ≤ `k` | slice-based low-rank projection |
| `PoincareBall n` | `‖x‖ < 1` | Euclidean add + ball projection (`sharp`/`flat` are `id` — not the Poincaré metric) |
| `Hyperbolic n` | hyperboloid `⟨x,x⟩_L = -1`, `coords : DTensor #[n+1] .Float32` | Minkowski normalize; `sharp`/`flat` apply the Minkowski metric |

Grassmann also ships geometry extras: `principalAngles`, `distance` (chordal, despite the
geodesic docstring), and a `log` map (`Grassmann.lean:62-136`).

### Steepest descent beyond the metric (`Tyr/Manifolds/Optimizer.lean`)

`DualMapGeometry` (`Optimizer.lean:43`) is the hook for Finsler-style steepest descent
where the update direction is *not* the Riemannian `sharp`:

```lean
class DualMapGeometry (M : Type) [DifferentiableManifold M] where
  tangentNorm : {x : M} → Tangent x → Float
  cotangentNorm : {x : M} → Cotangent x → Float
  dualMap : {x : M} → Cotangent x → Tangent x
```

User-facing entry points: `DualMapGeometry.dualMapStep x g lr : M` and
`dualMapStepWithDiagnostics` (`Optimizer.lean:59-81`). Instances: `Float` (sign descent),
`Stiefel`, `Orthogonal`, `Grassmann` (spectral/nuclear norms, tangent-projected and
Frobenius-normalized dual map), `Hyperbolic` (Minkowski metric + L2 normalization).

`EmbeddedManifold` (`Tyr/Manifolds/Embedded.lean:17`) exposes an ambient representation
(`Ambient`, `toAmbient`, `projectAmbientTangent`) plus `retractAmbientStep`; instances for
`Tensor σ`, `Stiefel`, `Orthogonal`, `Grassmann`, `Hyperbolic`. Downstream consumers build on
`DualMapGeometry` rather than this class: `DualOptimizer`'s generic manifold path and
`Tyr.Modular.Manifold`'s `MatrixManifoldCarrier` both call `DualMapGeometry.dualMapStep`.

### First-order optimizers: PolarExpress, NorMuon, DistAdam

`PolarExpress` (`Tyr/Optim/PolarExpress.lean`) is a thin wrapper over the `Tyr.Distributed`
FFI kernels: `apply G cfg` → `dist.polarExpress` (Newton–Schulz matrix-sign approximation)
and `muonOrthogonalize G (numIters := 5)`, which orthogonalizes along the short dimension.
Both run in `IO`.

`NorMuon` (`Tyr/Optim/NorMuon.lean`) is the Muon-family optimizer for weight matrices. The
per-parameter idiom, shared with the other first-order optimizers:

```lean
def stepSingle {σ : StaticSpec} (param : Tensor σ) (grad : Tensor σ) (state : ParamState σ)
    (cfg : Config) (lrMul wdMul : Float) : IO (Tensor σ × ParamState σ)   -- NorMuon.lean:221
```

`stepSingle` does: momentum-buffer update on the raw gradient, Nesterov blend,
`PolarExpress.muonOrthogonalize` (skipped on exact-zero gradients), aspect-ratio LR scaling
(`aspectRatioScale`, `sqrt(max(1, h/w))` for 2D shapes), then decoupled weight decay
(`p ← p - effectiveLr * weightDecay * wdMul * p`, AdamW-style as in `DistAdam`) and the
scaled subtract. Distributed variants: `stepDistributedOwner` (one rank updates,
then broadcast) and `stepDistributedGroup` (block-cyclic ownership, reduce-scatter +
all-gather over homogeneous `Array (Tensor σ)` groups). `ParamLabel` classifies parameters
(`attn`, `mlp`, `embed`, `lmHead`, `scalars`, …) with `shouldOrthogonalize`,
`defaultLrMul` (embeddings 75×, scalars 5×), `defaultWdMul`, and `getMomentum` implements
nanochat's 0.85→base momentum warmup.

`DistAdam` (`Tyr/Optim/DistAdam.lean`) is AdamW for embeddings, heads, and scalars:
`ParamState s` holds `expAvg`/`expAvgSq`/`step`, `stepSingle` is textbook bias-corrected
Adam with decoupled weight decay (`DistAdam.lean:107`), and `stepDistributed`
(`DistAdam.lean:175`) reduce-scatters gradients and all-gathers updated shards when the
first dimension shards evenly, otherwise falls back to all-reduce + local step. A sharded
API (`ShardedParamState`, `stepSharded`, `ShardedEmbeddingAdam`) layered on
`Tyr.Sharding.ShardedTensor` has no call sites outside its own file — treat as
experimental.

### DualOptimizer: the matrix/embedding split

`DualOptimizer` (`Tyr/Optim/DualOptimizer.lean`) wires the nanochat strategy: matrix
parameters go to a Muon-family backend, embeddings/LM head/scalars to AdamW. The
top-level `Config` (`DualOptimizer.lean:40`) carries per-group LRs (`matrixLr`,
`embeddingLr`, `lmHeadLr`, `scalarLr`), Muon hyperparameters, scheduled momentum/weight
decay (`getScheduledMomentum`, `getScheduledWeightDecay`), a `modelDim`/`refDim` LR scale
(`dimLrScale = (modelDim/refDim)^(-0.5)`), and backend selection:

```lean
inductive MatrixOptimizerKind where | norMuon | manifoldMuon        -- DualOptimizer.lean:27
inductive MatrixManifoldFamily where | stiefel | orthogonal | grassmann  -- :33
```

Matrix state is tagged by backend (`DualOptimizer.lean:294`):

```lean
inductive MatrixParamState (m n : UInt64) where
  | norMuon (state : NorMuon.ParamState #[m, n])
  | manifoldMuon (state : ManifoldMuon.ParamState m n)
  | genericManifold (state : GenericManifoldState m n)
```

`initMatrixParamState cfg param` builds the right constructor; `stepMatrixSingle`
(`DualOptimizer.lean:402`) dispatches: `.norMuon` → `NorMuon.stepSingle`;
`.manifoldMuon` + `.stiefel` (and not `preferGenericManifoldPath`) → the specialized
`ManifoldMuon.stepSingle`; otherwise the generic path `stepGenericManifoldSingle`, which
does Nesterov blending and then a `DualMapGeometry.dualMapStep` on the configured manifold
family (`DualOptimizer.lean:344-399`). A state/config constructor mismatch throws
`IO.userError` (it used to silently re-initialize the state, dropping momentum). Group variants `stepMatrixGroupLocal`/`stepMatrixGroupDistributed`
and the closure bundle `MatrixBackendOps m n` (`matrixBackendOps`,
`matrixBackendOpsForLabel`) cover batched and per-label use.

### ManifoldMuon: Stiefel-constrained Muon

`ManifoldMuon` (`Tyr/Optim/ManifoldMuon.lean`, marked experimental in its header) keeps
matrix parameters on the Stiefel manifold. `stepSingle` (`ManifoldMuon.lean:286`, with
`stepSingleWithDiagnostics` at `:228`): Nesterov blend, then solve for a
tangent-constrained matrix-sign direction — dual ascent on `Λ` or a damped fixed point,
chosen by `SolverKind := dualAscent | fixedPoint`, with `msign` via Polar Express — then
`tangentProject`, Frobenius-normalize, aspect-scale the LR, and QR-retract via
`Stiefel.project`. `ParamState m n` carries the momentum buffer and dual variable
`Λ : Tensor ⟨#[n, n], d, dev⟩`; `SolveDiagnostics` reports iterations, residual, dual delta/objective, convergence.

### Whole-model Riemannian steps

`RiemannianSGD` (`Tyr/Optim/RiemannianSGD.lean`) works over
`Tyr.Modular.RiemannianModule` instances with low-rank `MetricFactor` pullbacks:
`stepWithFactor` for an explicit output metric, `stepMSE` for half-squared-error with
identity output metric, and `stepSequentialMSE`/`stepSequentialWithFactor` for two-layer
`Sequential M₁ M₂` compositions. Each returns a `StepResult` with updated params,
prediction, input factor/cotangent, per-layer diagnostics, and loss.

`RiemannianTreeSGD` (`Tyr/Optim/RiemannianTreeSGD.lean`) lifts the same idea to arbitrary
`[TensorStruct Params]` models. `TreeMetricFactor Params rank` stores `rank` parameter-tree
rows (VJPs of output cotangents) with a size proof; the regularized metric system is solved
by Woodbury (`TreeMetricFactor.solveWoodbury`) and applied with `Optim.apply_updates`.
Entry points:

```lean
def stepCrossEntropy [TensorStruct Params]
    (params : Params) (forward : Params → IO (DTensor #[batch, seq, vocab] .Float32))
    (targets : DTensor #[batch, seq] .Int64) (lr : Float) (gradClip : Float := 0.0)
    : IO (StepResult Params #[batch, seq, vocab])                -- RiemannianTreeSGD.lean:290

def stepCrossEntropySampledFisher [TensorStruct Params]
    (params : Params) (forward : Params → IO (DTensor #[batch, seq, vocab] .Float32))
    (targets : DTensor #[batch, seq] .Int64) (probeCount : UInt64) (lr : Float) (gradClip : Float := 0.0)
    : IO (StepResult Params #[batch, seq, vocab])                -- :332
```

`stepCrossEntropy` builds an identity output factor of rank `batch*seq*vocab` — one VJP
per output element (`RiemannianTreeSGD.lean:301-306`), practical only for tiny shapes. The
sampled-Fisher variant sketches the factor with `probeCount` multinomial-sampled rows
(exact loss gradient, sketched metric) and is the usable one. `StepDiagnostics` reports
gradient/update/residual norms and an inner condition estimate.

## Key APIs

Optim core (`torch.Optim`, `Tyr/Optim.lean`):

| Name | Signature (abridged) |
|---|---|
| `adamw` | `(lr := 1e-3) (b1 := 0.9) (b2 := 0.999) (eps := 1e-8) (weight_decay := 0.01) : GradientTransformation α (AdamWState α)` |
| `adamw_schedule` | `(lr_schedule : Schedule) (b1 b2 eps weight_decay := …) : GradientTransformation α (AdamWScheduleState α)` |
| `sgd` / `sgd_momentum` | `(lr : Float)` / `(lr) (momentum := 0.9)` |
| `chain` | `GradientTransformation α S1 → GradientTransformation α S2 → GradientTransformation α (ChainState S1 S2)` |
| `scale_by_schedule` | `(schedule : Schedule) : GradientTransformation α ScaleByScheduleState` |
| `step` | `opt → params → grads → state → (α × S)` |
| `apply_updates` | `params → updates → params` (detach/add/re-enable grad) |

Schedules (`torch.Optim.Scheduler`): constructors and combinators as listed above; the legacy `ScheduleConfig` + `getLr` API mirrors them.

Manifolds (`Tyr.AD`): `DifferentiableManifold.gradientStep`, `DualMapGeometry.dualMapStep`,
`dualMapStepWithDiagnostics`, `EmbeddedManifold.retractAmbientStep`; per-family
`random`/`project` constructors (`Stiefel.random n p`, `Orthogonal.random n`,
`Grassmann.random n p`, `Sphere.random n`, `Hyperbolic.random n`, …) and tangent
projections (`StiefelTangent.project X Z`, `GrassmannTangent.project X V`,
`OrthogonalTangent.fromAmbient Q Z`, `HyperbolicTangent.project X V`).

First-order / split optimizers (`torch.Optim.*`):

| Name | Role |
|---|---|
| `NorMuon.Config` | `lr := 0.023`, `weightDecay := 1.2` (decoupled, scaled by `wdMul`), `momentum := 0.95`, `beta2 := 0.95`, `numIters := 5`, `distributed`, `worldSize` |
| `NorMuon.initParamState` / `stepSingle` / `stepDistributedOwner` / `stepDistributedGroup` | per-param and distributed Muon steps |
| `NorMuon.getMomentum step totalSteps (baseMomentum := 0.95) (warmupSteps := 300)` | momentum warmup schedule |
| `DistAdam.Config` | `lr := 0.008`, `beta1 := 0.65`, `beta2 := 0.95`, `eps := 1e-8`, `weightDecay := 0.005`, `distributed` |
| `DistAdam.initParamState` / `stepSingle` / `stepDistributed` | AdamW steps, optionally ZeRO-style sharded |
| `DualOptimizer.Config` | group LRs, Muon hyperparams, `matrixOptimizer`, `matrixManifold`, solver tolerances, `modelDim`/`refDim` |
| `DualOptimizer.initMatrixParamState cfg param` / `stepMatrixSingle … cfg step (lrMul := 1.0) (wdMul := 1.0)` | backend-dispatched matrix steps |
| `DualOptimizer.matrixBackendOps` / `matrixBackendOpsForLabel` | `MatrixBackendOps m n` closure bundles |
| `ManifoldMuon.Config` / `initParamState` / `stepSingle` / `stepSingleWithDiagnostics` / `stepDistributedGroup` | Stiefel-constrained Muon |
| `RiemannianSGD.stepMSE` / `stepWithFactor` / `stepSequentialMSE`; `RiemannianTreeSGD.stepCrossEntropy` / `stepCrossEntropySampledFisher` | whole-model Riemannian steps |

## Usage examples

### Indexed NorMuon state

```lean
import Tyr.Optim.NorMuon
open torch

def matrixOptimizerStep {σ : StaticSpec}
    (parameter gradient : Tensor σ) (state : Optim.NorMuon.ParamState σ)
    (config : Optim.NorMuon.Config) : IO (Tensor σ × Optim.NorMuon.ParamState σ) :=
  Optim.NorMuon.stepSingle parameter gradient state config 1.0 1.0
```

The parameter, gradient, returned tensor, and optimizer state share the complete
specification. Runtime numerical requirements still apply to the selected
optimizer; a specification-preserving signature does not make every dtype
suitable for matrix optimization.


### AdamW over a parameter tree

Reconstructed example (from `Examples/GPT/Train.lean:138-145`):

```lean
-- params : GPTParams modelCfg, optState : Optim.AdamWState (GPTParams modelCfg)
let lossT ← gpt.loss params x y true
Tensor.backwardLoss lossT
let lossVal := Tensor.item lossT
let grads := TensorStruct.grads params
let opt := Optim.adamw (lr := lr)
let (params', optState') := Optim.step opt params grads optState
```

`lr` is computed by the caller (that example keeps its own inline cosine warmup). The
schedule-native alternative `Optim.adamw_schedule (Scheduler.cosine_decay …)` reads the LR
from the optimizer's internal step counter instead.

### Dual split: DistAdam for embeddings, NorMuon for matrices

Reconstructed example (from `Examples/NanoChat/ModdedTrain.lean:864-911`):

```lean
let adamCfgBase : DistAdam.Config := {
  lr := 1.0, beta1 := hp.adamBeta1, beta2 := hp.adamBeta2
  eps := 1e-10, weightDecay := hp.adamWeightDecay, distributed := isDistributed }
let muonCfg : NorMuon.Config := {
  lr := lrMatrix, weightDecay := 0.0, momentum := muMomentum
  beta2 := 0.95, numIters := 5, distributed := isDistributed, worldSize := worldSize }

let (embed', embedSt') ← DistAdam.stepDistributed
  params.embed grads.embed optState.dualState.embed { adamCfgBase with lr := lrEmbed } 1.0 1.0
let (wQs', wQSts') ← NorMuon.stepDistributedGroup
  #[params.wQ] #[grads.wQ] #[optState.dualState.wQ] muonCfg   -- lrMul/wdMul default to 1.0
```

### Manifold matrix backend via DualOptimizer

Reconstructed example (from `Tests/TestDualOptimizerDispatch.lean:106-112`):

```lean
let cfg : DualOptimizer.Config := {
  matrixOptimizer := .manifoldMuon
  matrixManifold := .grassmann }                 -- generic DualMapGeometry path
let st0 := DualOptimizer.initMatrixParamState cfg W
let (W1, st1) ← DualOptimizer.stepMatrixSingle W gW st0 cfg 0 1.0 1.0
```

### Direct manifold steps

Reconstructed example (from `Tests/TestManifoldMuon.lean:21-47`):

```lean
open Tyr.AD
let mut q ← Orthogonal.random 16
for _ in [:numSteps] do
  let gRaw ← randn #[16, 16] false
  let g := OrthogonalTangent.fromAmbient q gRaw
  q := DualMapGeometry.dualMapStep q g 0.02      -- retracted steepest step on O(16)
```

### Whole-model Riemannian step

Reconstructed example (from `Examples/GPT/RiemannianNanoGPT.lean:124-141`):

```lean
let step ← torch.Optim.RiemannianTreeSGD.stepCrossEntropy
  params (fun p => gpt.forward p x true) y lr gradClip
-- or, for realistic vocab sizes:
let step ← torch.Optim.RiemannianTreeSGD.stepCrossEntropySampledFisher
  params (fun p => gpt.forward p x true) y probeCount lr gradClip
pure (step.params, step.loss, step.diagnostics)
```

## Caveats

- `NorMuon.stepSingle` applies decoupled weight decay as
  `p ← p - effectiveLr * weightDecay * wdMul * p` (the aspect-ratio-scaled LR is included);
  with the default `weightDecay := 1.2` this is a strong pull — set `weightDecay := 0.0`
  (or pass `wdMul := 0.0`) to disable it.
- `RiemannianTreeSGD.stepCrossEntropy` is exact but scales with `batch*seq*vocab` VJPs;
  use `stepCrossEntropySampledFisher` outside tiny test shapes.
- `PoincareBall`'s instance uses identity `sharp`/`flat` and a Euclidean-add retraction
  (`Tyr/Manifolds/PoincareBall.lean:48-59`) — the ball constraint, not Poincaré geometry.
- `DualOptimizer.stepMatrixSingle` throws `IO.userError` when the `MatrixParamState`
  constructor does not match the configured backend (e.g. a config change mid-run) instead
  of silently re-initializing; rebuild state explicitly with `initMatrixParamState cfg`.
- Distributed step functions fall back to the local path when `dist.isInitialized` is
  false or `worldSize ≤ 1`, so single-process tests exercise the fallback, not collectives.

## Related guides

- [TensorStruct](core/tensorstruct.md) — the tree traversal class every optimizer here is generic over
- [Tensors and the raw FFI](core/tensors.md) — `Tensor σ`, `Shape`, indexed autograd
- [Autodiff](autodiff.md) — gradient computation feeding `step`/`stepSingle`
- [Modules](modules.md) — parameter structures the optimizer state mirrors
- [Serialization](serialization.md) — checkpointing optimizer state trees
- [Distributed](distributed.md) — the collectives behind `stepDistributed*` and Polar Express
- [Getting started](getting-started.md) — a first training loop
- [Examples and testing](examples-and-testing.md) — where these optimizers are exercised

---

This is a guide, not an exhaustive reference. Full per-symbol documentation for
`torch.Optim.*` and `Tyr.AD.*` is generated by doc-gen4 (see `docbuild/`).
