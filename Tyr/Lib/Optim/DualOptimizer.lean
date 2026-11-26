/-
  Tyr/Lib/Optim/DualOptimizer.lean

  Dual Optimizer Setup following nanochat's approach:
  - Muon (with Polar Express) for weight matrices (attention, MLP)
  - AdamW for embeddings, unembeddings, and scalar parameters

  This strategy reportedly improves training efficiency by 2x+ over pure AdamW.
-/
import Tyr.Core.TensorStruct
import Tyr.Lib.Distributed
import Tyr.Lib.Optim.NorMuon
import Tyr.Lib.Optim.DistAdam
import Tyr.Lib.Optim.ManifoldMuon
import Tyr.Lib.Optim.Scheduler
import Tyr.Lib.Optim.Modular.Budget
import Tyr.Lib.Optim.Manifolds.Optimizer

namespace torch.Optim.DualOptimizer

open torch
open NorMuon (ParamLabel)
open Tyr.Lib.Optim.Modular

variable {d : DType} {dev : Device} [DType.RealFloating d]

/-- Matrix optimizer backend selection (experimental). -/
inductive MatrixOptimizerKind where
  | norMuon
  | manifoldMuon
  deriving Repr, BEq

/-- Matrix manifold family for constrained matrix updates. -/
inductive MatrixManifoldFamily where
  | stiefel
  | orthogonal
  | grassmann
  deriving Repr, BEq

/-- Configuration for dual optimizer setup -/
structure Config where
  /-- Base learning rate for matrix parameters (Muon) -/
  matrixLr : Float := 0.02
  /-- Learning rate for token embeddings (AdamW) -/
  embeddingLr : Float := 0.3
  /-- Learning rate for language model head (AdamW) -/
  lmHeadLr : Float := 0.3
  /-- Learning rate for scalar parameters (AdamW) -/
  scalarLr : Float := 0.02
  /-- Muon momentum coefficient -/
  muonMomentum : Float := 0.95
  /-- Muon weight decay -/
  muonWeightDecay : Float := 0.01
  /-- Newton-Schulz iterations for NorMuon orthogonalization. -/
  muonNumIters : Nat := 5
  /-- Whether matrix updates run distributed collectives for NorMuon path. -/
  muonDistributed : Bool := false
  /-- AdamW beta1 -/
  adamBeta1 : Float := 0.9
  /-- AdamW beta2 -/
  adamBeta2 : Float := 0.95
  /-- AdamW epsilon -/
  adamEps : Float := 1e-8
  /-- Total training steps (for scheduling) -/
  totalSteps : Nat := 10000
  /-- Warmup steps for momentum -/
  warmupSteps : Nat := 300
  /-- Matrix optimizer backend (default keeps NorMuon behavior). -/
  matrixOptimizer : MatrixOptimizerKind := .norMuon
  /-- Matrix manifold family used by manifold-backed matrix optimizers. -/
  matrixManifold : MatrixManifoldFamily := .stiefel
  /-- Force generic dual-map manifold path even when specialized kernels exist. -/
  preferGenericManifoldPath : Bool := false
  /-- Newton-Schulz iterations for specialized manifold-Muon sign approximation. -/
  manifoldNumIters : Nat := 5
  /-- Dual-ascent steps for specialized manifold-Muon. -/
  manifoldDualAscentSteps : Nat := 2
  /-- Dual-ascent learning rate for specialized manifold-Muon. -/
  manifoldDualAscentLr : Float := 0.1
  /-- Inner solver backend for specialized manifold-Muon. -/
  manifoldSolver : torch.Optim.ManifoldMuon.SolverKind := .dualAscent
  /-- Minimum inner iterations before solver convergence checks. -/
  manifoldMinSolveSteps : Nat := 1
  /-- Tangent residual tolerance for specialized manifold-Muon inner solve. -/
  manifoldSolveResidualTol : Float := 1e-4
  /-- Dual-delta tolerance for specialized manifold-Muon inner solve. -/
  manifoldSolveDualDeltaTol : Float := 1e-5
  /-- Damping for specialized manifold-Muon fixed-point inner solve. -/
  manifoldFixedPointDamping : Float := 0.5
  /-- Whether manifold matrix paths use distributed collectives. -/
  manifoldDistributed : Bool := false
  /-- Enable modular-budget multipliers on group learning rates. -/
  useModularBudget : Bool := false
  /-- Optional groupwise modular budget multipliers. -/
  budget : GroupBudget := {}
  deriving Repr

/-- Get Muon config from dual optimizer config -/
def toMuonConfig (cfg : Config) : NorMuon.Config := {
  lr := cfg.matrixLr
  weightDecay := cfg.muonWeightDecay
  momentum := cfg.muonMomentum
  beta2 := 0.95
  numIters := cfg.muonNumIters
  distributed := cfg.muonDistributed
  worldSize := 1
}

/--
Per-label matrix backend spec.

This allows retrofitting manifold/NorMuon selection below top-level by binding
backend configs at parameter-label granularity.
-/
structure MatrixBackendSpec where
  /-- Fallback config used when no label-specific override is provided. -/
  default : Config := {}
  /-- Attention-matrix override (`attn`, `attnGate`). -/
  attn : Option Config := none
  /-- MLP-matrix override. -/
  mlp : Option Config := none
  /-- Smear-gate matrix override. -/
  smearGate : Option Config := none
  deriving Repr

/-- Resolve per-label matrix backend configuration from a `MatrixBackendSpec`. -/
def MatrixBackendSpec.configForLabel (spec : MatrixBackendSpec) (label : ParamLabel) : Config :=
  match label with
  | .attn => spec.attn.getD spec.default
  | .attnGate => spec.attn.getD spec.default
  | .mlp => spec.mlp.getD spec.default
  | .smearGate => spec.smearGate.getD spec.default
  | _ => spec.default

/-- Determine whether current config uses manifold-Muon for matrix groups. -/
def usesManifoldMuon (cfg : Config) : Bool :=
  cfg.matrixOptimizer == .manifoldMuon

/-- True when manifold matrix backend should use specialized Stiefel-Muon kernels. -/
def usesSpecializedStiefelPath (cfg : Config) : Bool :=
  cfg.matrixOptimizer == .manifoldMuon &&
    cfg.matrixManifold == .stiefel &&
    !cfg.preferGenericManifoldPath

/-- Get scheduled momentum for a given step (with warmup/cooldown) -/
def getScheduledMomentum (cfg : Config) (step : Nat) : Float :=
  NorMuon.getMomentum step cfg.totalSteps cfg.muonMomentum cfg.warmupSteps 50

/-- Get scheduled weight decay for a given step (linear decay to zero) -/
def getScheduledWeightDecay (cfg : Config) (step : Nat) : Float :=
  Scheduler.linear_weight_decay cfg.muonWeightDecay cfg.totalSteps step

/-- Get current Muon config with scheduled values -/
def getMuonConfigAtStep (cfg : Config) (step : Nat) : NorMuon.Config :=
  let momentum := getScheduledMomentum cfg step
  let weightDecay := getScheduledWeightDecay cfg step
  { toMuonConfig cfg with momentum, weightDecay }

/-- Get current manifold-Muon config with scheduled momentum. -/
def getManifoldMuonConfigAtStep (cfg : Config) (step : Nat) : torch.Optim.ManifoldMuon.Config :=
  let momentum := getScheduledMomentum cfg step
  {
    lr := cfg.matrixLr
    momentum := momentum
    numIters := cfg.manifoldNumIters
    dualAscentSteps := cfg.manifoldDualAscentSteps
    dualAscentLr := cfg.manifoldDualAscentLr
    solver := cfg.manifoldSolver
    minSolveSteps := cfg.manifoldMinSolveSteps
    solveResidualTol := cfg.manifoldSolveResidualTol
    solveDualDeltaTol := cfg.manifoldSolveDualDeltaTol
    fixedPointDamping := cfg.manifoldFixedPointDamping
    distributed := cfg.manifoldDistributed
  }

/-- Generic dual-map manifold state for matrix parameters. -/
structure GenericManifoldState (m n : Nat) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  /-- Momentum buffer used to form Nesterov-like blended gradients. -/
  momentumBuffer : Option (Tensor ⟨#[m, n], dtype, device⟩) := none
  /-- Local step count for this parameter. -/
  step : Nat := 0
  deriving Repr

instance {m n : Nat} {d : DType} {dev : Device} :
    TensorStruct (GenericManifoldState m n d dev) where
  map f s := { s with momentumBuffer := s.momentumBuffer.map (TensorStruct.map f) }
  mapM f s := do
    let momentumBuffer ← match s.momentumBuffer with
      | some t => some <$> TensorStruct.mapM f t
      | none => pure none
    pure { s with momentumBuffer }
  zipWith f a b := {
    momentumBuffer := match a.momentumBuffer, b.momentumBuffer with
      | some t1, some t2 => some (TensorStruct.zipWith f t1 t2)
      | _, _ => none
    step := a.step
  }
  fold f init s := match s.momentumBuffer with
    | some t => TensorStruct.fold f init t
    | none => init

/-- Matrix-parameter optimizer state tagged by backend family. -/
inductive MatrixParamState (m n : Nat) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  | norMuon (state : NorMuon.ParamState ⟨#[m, n], dtype, device⟩)
  | manifoldMuon (state : torch.Optim.ManifoldMuon.ParamState m n dtype device)
  | genericManifold (state : GenericManifoldState m n dtype device)
  deriving Repr

instance {m n : Nat} {d : DType} {dev : Device} :
    TensorStruct (MatrixParamState m n d dev) where
  map f st :=
    match st with
    | .norMuon s => .norMuon (TensorStruct.map f s)
    | .manifoldMuon s => .manifoldMuon (TensorStruct.map f s)
    | .genericManifold s => .genericManifold (TensorStruct.map f s)
  mapM f st := do
    match st with
    | .norMuon s => pure <| .norMuon (← TensorStruct.mapM f s)
    | .manifoldMuon s => pure <| .manifoldMuon (← TensorStruct.mapM f s)
    | .genericManifold s => pure <| .genericManifold (← TensorStruct.mapM f s)
  zipWith f a b :=
    match a, b with
    | .norMuon s1, .norMuon s2 => .norMuon (TensorStruct.zipWith f s1 s2)
    | .manifoldMuon s1, .manifoldMuon s2 => .manifoldMuon (TensorStruct.zipWith f s1 s2)
    | .genericManifold s1, .genericManifold s2 => .genericManifold (TensorStruct.zipWith f s1 s2)
    | .norMuon s, _ => .norMuon s
    | .manifoldMuon s, _ => .manifoldMuon s
    | .genericManifold s, _ => .genericManifold s
  fold f init st :=
    match st with
    | .norMuon s => TensorStruct.fold f init s
    | .manifoldMuon s => TensorStruct.fold f init s
    | .genericManifold s => TensorStruct.fold f init s

/-- Initialize generic manifold state. -/
def initGenericManifoldState {m n : Nat} (_param : Tensor ⟨#[m, n], d, dev⟩) :
    GenericManifoldState m n d dev := {}

/-- Initialize matrix state according to the configured matrix backend. -/
def initMatrixParamState {m n : Nat} (cfg : Config) (param : Tensor ⟨#[m, n], d, dev⟩) :
    MatrixParamState m n d dev :=
  if cfg.matrixOptimizer == .norMuon then
    .norMuon (NorMuon.initParamState param)
  else if usesSpecializedStiefelPath cfg then
    .manifoldMuon (torch.Optim.ManifoldMuon.initParamState param)
  else
    .genericManifold (initGenericManifoldState param)

/-- Backend tag name for a matrix state (used in error messages). -/
def MatrixParamState.backendName {m n : Nat} {d : DType} {dev : Device}
    (st : MatrixParamState m n d dev) : String :=
  match st with
  | .norMuon _ => "norMuon"
  | .manifoldMuon _ => "manifoldMuon"
  | .genericManifold _ => "genericManifold"

private def stepOnManifoldFamily {m n : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (family : MatrixManifoldFamily)
    (param : Tensor ⟨#[m, n], d, dev⟩)
    (ambientGrad : Tensor ⟨#[m, n], d, dev⟩)
    (lr : Float)
    : IO (Tensor ⟨#[m, n], d, dev⟩) := do
  match family with
  | .stiefel =>
    if h : n ≤ m then
      let x : Tyr.Lib.Optim.Manifolds.Stiefel m n d dev := ⟨param, h⟩
      let g := Tyr.Lib.Optim.Manifolds.StiefelTangent.project x ambientGrad
      pure (Tyr.Lib.Optim.Manifolds.DualMapGeometry.dualMapStep x g lr).matrix
    else
      -- Wide matrices have no column-orthonormal frame; step the transposed
      -- (row-orthonormal) Stiefel representative instead, matching Muon's
      -- usual tall/wide handling.
      let h' : m ≤ n := Nat.le_of_lt (Nat.lt_of_not_le h)
      let x : Tyr.Lib.Optim.Manifolds.Stiefel n m d dev := ⟨Tensor.transpose2d param, h'⟩
      let g := Tyr.Lib.Optim.Manifolds.StiefelTangent.project x (Tensor.transpose2d ambientGrad)
      pure (Tensor.transpose2d (Tyr.Lib.Optim.Manifolds.DualMapGeometry.dualMapStep x g lr).matrix)
  | .orthogonal =>
    let support ← d.acquireConstantSupport dev
    let _ := support.down
    -- Orthogonal requires square matrices.
    if h : m = n then
      let pSquare : Tensor ⟨#[n, n], d, dev⟩ := by simpa [h] using param
      let gSquare : Tensor ⟨#[n, n], d, dev⟩ := by simpa [h] using ambientGrad
      let x : Tyr.Lib.Optim.Manifolds.Orthogonal n d dev := ⟨pSquare⟩
      let tg := Tyr.Lib.Optim.Manifolds.OrthogonalTangent.fromAmbient x gSquare
      let x' := Tyr.Lib.Optim.Manifolds.DualMapGeometry.dualMapStep x tg lr
      pure (by simpa [h] using x'.matrix : Tensor ⟨#[m, n], d, dev⟩)
    else
      throw <| IO.userError s!"MatrixManifoldFamily.orthogonal requires square matrices, got {m}x{n}"
  | .grassmann =>
    if h : n ≤ m then
      let x : Tyr.Lib.Optim.Manifolds.Grassmann m n d dev := ⟨param, h⟩
      let g := Tyr.Lib.Optim.Manifolds.GrassmannTangent.project x ambientGrad
      pure (Tyr.Lib.Optim.Manifolds.DualMapGeometry.dualMapStep x g lr).matrix
    else
      let h' : m ≤ n := Nat.le_of_lt (Nat.lt_of_not_le h)
      let x : Tyr.Lib.Optim.Manifolds.Grassmann n m d dev := ⟨Tensor.transpose2d param, h'⟩
      let g := Tyr.Lib.Optim.Manifolds.GrassmannTangent.project x (Tensor.transpose2d ambientGrad)
      pure (Tensor.transpose2d (Tyr.Lib.Optim.Manifolds.DualMapGeometry.dualMapStep x g lr).matrix)

/--
Single matrix step for the generic manifold path using `DualMapGeometry`.

This path is intentionally simple and composable:
- shared momentum blending from NorMuon,
- manifold-family-specific projection/retraction from `Tyr.Lib.Optim.Manifolds`,
- no specialized dual-ascent state.
-/
def stepGenericManifoldSingle {m n : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (param : Tensor ⟨#[m, n], d, dev⟩)
    (grad : Tensor ⟨#[m, n], d, dev⟩)
    (state : GenericManifoldState m n d dev)
    (cfg : Config)
    (step : Nat)
    (lrMul : Float := 1.0)
    : IO (Tensor ⟨#[m, n], d, dev⟩ × GenericManifoldState m n d dev) := do
  let gRaw := Tensor.detach grad
  let momentum := getScheduledMomentum cfg step
  let newMomentum := NorMuon.updateMomentum gRaw state.momentumBuffer momentum
  let nesterovGrad := Tensor.add (Tensor.mulScalar gRaw (1.0 - momentum))
    (Tensor.mulScalar newMomentum momentum)
  let aspectScale := NorMuon.aspectRatioScale #[m, n]
  let effectiveLr := cfg.matrixLr * lrMul * aspectScale
  let updated ← stepOnManifoldFamily cfg.matrixManifold param nesterovGrad effectiveLr
  let newParam ← Tensor.parameter (Tensor.detach updated)
  let newState : GenericManifoldState m n d dev := {
    momentumBuffer := some newMomentum
    step := state.step + 1
  }
  return (newParam, newState)

/-- Single matrix step that dispatches to the configured matrix backend.

    The state constructor must match the configured backend. A mismatch means the
    state was built under a different config (e.g. a mid-run backend change) and
    stepping with it would silently drop the accumulated momentum state, so this
    throws `IO.userError` instead of re-initializing. Fresh state for a first step
    should be built with `initMatrixParamState cfg`. -/
def stepMatrixSingle {m n : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (param : Tensor ⟨#[m, n], d, dev⟩)
    (grad : Tensor ⟨#[m, n], d, dev⟩)
    (state : MatrixParamState m n d dev)
    (cfg : Config)
    (step : Nat)
    (lrMul : Float := 1.0)
    (wdMul : Float := 1.0)
    : IO (Tensor ⟨#[m, n], d, dev⟩ × MatrixParamState m n d dev) := do
  match cfg.matrixOptimizer with
  | .norMuon =>
    let st ← match state with
      | .norMuon s => pure s
      | _ => throw <| IO.userError s!"stepMatrixSingle: state backend '{state.backendName}' does not match configured backend 'norMuon'; refusing to silently reset optimizer state"
    let muonCfg := getMuonConfigAtStep cfg step
    let (p', st') ← NorMuon.stepSingle param grad st muonCfg lrMul wdMul
    return (p', .norMuon st')
  | .manifoldMuon =>
    if usesSpecializedStiefelPath cfg then
      let st ← match state with
        | .manifoldMuon s => pure s
        | _ => throw <| IO.userError s!"stepMatrixSingle: state backend '{state.backendName}' does not match configured backend 'manifoldMuon' (specialized Stiefel path); refusing to silently reset optimizer state"
      let manifoldCfg := getManifoldMuonConfigAtStep cfg step
      let (p', st') ← torch.Optim.ManifoldMuon.stepSingle param grad st manifoldCfg lrMul
      return (p', .manifoldMuon st')
    else
      let st ← match state with
        | .genericManifold s => pure s
        | _ => throw <| IO.userError s!"stepMatrixSingle: state backend '{state.backendName}' does not match configured generic manifold path; refusing to silently reset optimizer state"
      let (p', st') ← stepGenericManifoldSingle param grad st cfg step lrMul
      return (p', .genericManifold st')

/-- Local matrix-group step that composes `stepMatrixSingle` over each parameter. -/
def stepMatrixGroupLocal {m n k : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (params grads : Vector (Tensor ⟨#[m, n], d, dev⟩) k)
    (states : Vector (MatrixParamState m n d dev) k)
    (cfg : Config)
    (step : Nat)
    (lrMul : Float := 1.0)
    (wdMul : Float := 1.0)
    : IO (Vector (Tensor ⟨#[m, n], d, dev⟩) k × Vector (MatrixParamState m n d dev) k) := do
  let steps ← (Vector.finRange k).mapM fun i =>
    stepMatrixSingle params[i] grads[i] states[i] cfg step lrMul wdMul
  return (steps.map (·.1), steps.map (·.2))

private def allNorMuonStates? {m n k : Nat} {d : DType} {dev : Device}
    (states : Vector (MatrixParamState m n d dev) k) :
    Option (Vector (NorMuon.ParamState ⟨#[m, n], d, dev⟩) k) :=
  states.mapM fun
    | .norMuon s => some s
    | _ => none

private def allManifoldMuonStates? {m n k : Nat} {d : DType} {dev : Device}
    (states : Vector (MatrixParamState m n d dev) k) :
    Option (Vector (torch.Optim.ManifoldMuon.ParamState m n d dev) k) :=
  states.mapM fun
    | .manifoldMuon s => some s
    | _ => none

/--
Distributed matrix-group step with backend-aware dispatch.

Fast-paths:
- NorMuon uses NorMuon's distributed owner update path.
- Specialized Stiefel manifold-Muon uses its distributed path.

Fallback:
- generic manifold path does gradient all-reduce then local per-parameter updates.
-/
def stepMatrixGroupDistributed {m n k : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (params grads : Vector (Tensor ⟨#[m, n], d, dev⟩) k)
    (states : Vector (MatrixParamState m n d dev) k)
    (cfg : Config)
    (step : Nat)
    (lrMul : Float := 1.0)
    (wdMul : Float := 1.0)
    : IO (Vector (Tensor ⟨#[m, n], d, dev⟩) k × Vector (MatrixParamState m n d dev) k) := do
  if k = 0 then
    return (params, states)

  match cfg.matrixOptimizer with
  | .norMuon =>
    match allNorMuonStates? states with
    | some st =>
      let muonCfg := getMuonConfigAtStep cfg step
      let (params', states') ← NorMuon.stepDistributedGroup params grads st muonCfg lrMul wdMul
      return (params', states'.map MatrixParamState.norMuon)
    | none =>
      stepMatrixGroupLocal params grads states cfg step lrMul wdMul
  | .manifoldMuon =>
    if usesSpecializedStiefelPath cfg then
      match allManifoldMuonStates? states with
      | some st =>
        let manifoldCfg := getManifoldMuonConfigAtStep cfg step
        let (params', states') ← torch.Optim.ManifoldMuon.stepDistributedGroup params grads st manifoldCfg lrMul
        return (params', states'.map MatrixParamState.manifoldMuon)
      | none =>
        stepMatrixGroupLocal params grads states cfg step lrMul wdMul
    else
      let isDist ← dist.isInitialized
      if !cfg.manifoldDistributed || !isDist then
        stepMatrixGroupLocal params grads states cfg step lrMul wdMul
      else
        let worldSize ← dist.getWorldSize
        if worldSize <= 1 then
          stepMatrixGroupLocal params grads states cfg step lrMul wdMul
        else
          let reducedGrads ← grads.mapM fun g => dist.allReduce (Tensor.detach g) .avg
          stepMatrixGroupLocal params reducedGrads states cfg step lrMul wdMul

/-- Composable matrix backend operations for a fixed shape. -/
structure MatrixBackendOps (m n : Nat) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  initState : Tensor ⟨#[m, n], dtype, device⟩ → MatrixParamState m n dtype device
  stepSingle : Tensor ⟨#[m, n], dtype, device⟩ → Tensor ⟨#[m, n], dtype, device⟩ →
    MatrixParamState m n dtype device →
    IO (Tensor ⟨#[m, n], dtype, device⟩ × MatrixParamState m n dtype device)
  stepGroupLocal : {k : Nat} → Vector (Tensor ⟨#[m, n], dtype, device⟩) k →
    Vector (Tensor ⟨#[m, n], dtype, device⟩) k → Vector (MatrixParamState m n dtype device) k →
    IO (Vector (Tensor ⟨#[m, n], dtype, device⟩) k × Vector (MatrixParamState m n dtype device) k)
  stepGroupDistributed : {k : Nat} → Vector (Tensor ⟨#[m, n], dtype, device⟩) k →
    Vector (Tensor ⟨#[m, n], dtype, device⟩) k → Vector (MatrixParamState m n dtype device) k →
    IO (Vector (Tensor ⟨#[m, n], dtype, device⟩) k × Vector (MatrixParamState m n dtype device) k)

/--
Build a backend operations bundle for a matrix parameter shape.

This keeps optimizer composition explicit: call-sites can choose backend ops once,
then reuse the same `init/step` closures regardless of matrix backend kind.
-/
def matrixBackendOps {m n : Nat} {d : DType} {dev : Device} [DType.RealFloating d]
    (cfg : Config)
    (step : Nat)
    (lrMul : Float := 1.0)
    (wdMul : Float := 1.0)
    : MatrixBackendOps m n d dev := {
  initState := initMatrixParamState cfg
  stepSingle := fun p g st => stepMatrixSingle p g st cfg step lrMul wdMul
  stepGroupLocal := fun ps gs ss => stepMatrixGroupLocal ps gs ss cfg step lrMul wdMul
  stepGroupDistributed := fun ps gs ss => stepMatrixGroupDistributed ps gs ss cfg step lrMul wdMul
}

end torch.Optim.DualOptimizer
