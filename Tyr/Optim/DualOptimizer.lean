/-
  Tyr/Optim/DualOptimizer.lean

  Dual Optimizer Setup following nanochat's approach:
  - Muon (with Polar Express) for weight matrices (attention, MLP)
  - AdamW for embeddings, unembeddings, and scalar parameters

  This strategy reportedly improves training efficiency by 2x+ over pure AdamW.
-/
import Tyr.TensorStruct
import Tyr.Distributed
import Tyr.Optim.NorMuon
import Tyr.Optim.DistAdam
import Tyr.Optim.ManifoldMuon
import Tyr.Optim.Scheduler
import Tyr.Modular.Budget
import Tyr.Manifolds.Optimizer

namespace torch.Optim.DualOptimizer

open torch
open NorMuon (ParamLabel)
open Tyr.Modular

variable {d : DType} {dev : DevicePolicy}

/-- Matrix optimizer backend selection (experimental). -/
inductive MatrixOptimizerKind where
  | norMuon
  | manifoldMuon
  deriving Repr, BEq, Inhabited

/-- Matrix manifold family for constrained matrix updates. -/
inductive MatrixManifoldFamily where
  | stiefel
  | orthogonal
  | grassmann
  deriving Repr, BEq, Inhabited

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
  muonNumIters : UInt64 := 5
  /-- Whether matrix updates run distributed collectives for NorMuon path. -/
  muonDistributed : Bool := false
  /-- AdamW beta1 -/
  adamBeta1 : Float := 0.9
  /-- AdamW beta2 -/
  adamBeta2 : Float := 0.95
  /-- AdamW epsilon -/
  adamEps : Float := 1e-8
  /-- Model dimension (for LR scaling) -/
  modelDim : UInt64 := 768
  /-- Reference dimension for LR scaling (typically 768 for GPT-2 base) -/
  refDim : UInt64 := 768
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
  manifoldNumIters : UInt64 := 5
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
  deriving Repr, Inhabited

/-- Compute dimension-based learning rate scaling.
    scale = (modelDim / refDim)^(-0.5)
    Larger models get smaller learning rates. -/
def dimLrScale (cfg : Config) : Float :=
  Float.pow (cfg.modelDim.toFloat / cfg.refDim.toFloat) (-0.5)

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

/-- Get AdamW config from dual optimizer config -/
def toAdamConfig (cfg : Config) : DistAdam.Config := {
  lr := cfg.embeddingLr * dimLrScale cfg
  beta1 := cfg.adamBeta1
  beta2 := cfg.adamBeta2
  eps := cfg.adamEps
  weightDecay := 0.0  -- AdamW typically has no weight decay in nanochat
}

/-- Parameter group classification for dual optimizer -/
inductive ParamGroup where
  | matrix     : ParamGroup  -- Use Muon (attention, MLP weights)
  | embedding  : ParamGroup  -- Use AdamW (token embeddings)
  | lmHead     : ParamGroup  -- Use AdamW (output layer)
  | scalar     : ParamGroup  -- Use AdamW (per-layer scalars)
  deriving Repr, BEq, Inhabited

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
  deriving Repr, Inhabited

/-- Map NorMuon's ParamLabel to ParamGroup -/
def labelToGroup (label : ParamLabel) : ParamGroup :=
  match label with
  | .attn => .matrix
  | .mlp => .matrix
  | .smearGate => .matrix
  | .attnGate => .matrix
  | .embed => .embedding
  | .valueEmbed => .embedding
  | .lmHead => .lmHead
  | .scalars => .scalar

/-- Resolve per-label matrix backend configuration from a `MatrixBackendSpec`. -/
def MatrixBackendSpec.configForLabel (spec : MatrixBackendSpec) (label : ParamLabel) : Config :=
  match label with
  | .attn => spec.attn.getD spec.default
  | .attnGate => spec.attn.getD spec.default
  | .mlp => spec.mlp.getD spec.default
  | .smearGate => spec.smearGate.getD spec.default
  | _ => spec.default

/-- Get learning rate multiplier for a parameter group -/
def baseGroupLrMul (cfg : Config) (group : ParamGroup) : Float :=
  let scale := dimLrScale cfg
  match group with
  | .matrix => 1.0  -- Base LR for Muon
  | .embedding => cfg.embeddingLr / cfg.matrixLr * scale
  | .lmHead => cfg.lmHeadLr / cfg.matrixLr * scale
  | .scalar => cfg.scalarLr / cfg.matrixLr

/-- Get learning rate multiplier for a parameter group (optionally budget-scaled). -/
def groupLrMul (cfg : Config) (group : ParamGroup) : Float :=
  let raw := baseGroupLrMul cfg group
  if !cfg.useModularBudget then
    raw
  else
    let budgetMul :=
      match group with
      | .matrix => cfg.budget.matrix
      | .embedding => cfg.budget.embedding
      | .lmHead => cfg.budget.lmHead
      | .scalar => cfg.budget.scalar
    raw * budgetMul

/-- Get weight decay multiplier for a parameter group -/
def groupWdMul (_cfg : Config) (group : ParamGroup) : Float :=
  match group with
  | .matrix => 1.0
  | .embedding => 0.0  -- No weight decay on embeddings
  | .lmHead => 0.0     -- No weight decay on lm_head
  | .scalar => 0.0     -- No weight decay on scalars

/-- Determine if a parameter group should use orthogonalized updates (Muon) -/
def groupUsesMuon (group : ParamGroup) : Bool :=
  match group with
  | .matrix => true
  | _ => false

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
  Scheduler.linearWeightDecay { baseWd := cfg.muonWeightDecay, totalSteps := cfg.totalSteps } step

/-- Simple training step tracker -/
structure TrainingStep where
  /-- Current step -/
  step : Nat := 0
  deriving Repr, Inhabited

/-- Initialize training step tracker -/
def TrainingStep.init : TrainingStep := { step := 0 }

/-- Increment step counter -/
def TrainingStep.next (ts : TrainingStep) : TrainingStep :=
  { ts with step := ts.step + 1 }

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

/-- Initialize Muon optimizer state -/
def initMuonState (cfg : Config) : IO NorMuon.State :=
  NorMuon.State.init (toMuonConfig cfg)

/-- Generic dual-map manifold state for matrix parameters. -/
structure GenericManifoldState (m n : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  /-- Momentum buffer used to form Nesterov-like blended gradients. -/
  momentumBuffer : Option (Tensor ⟨#[m, n], dtype, device⟩) := none
  /-- Local step count for this parameter. -/
  step : Nat := 0
  deriving Repr, Inhabited

instance {m n : UInt64} {d : DType} {dev : DevicePolicy} :
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
inductive MatrixParamState (m n : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  | norMuon (state : NorMuon.ParamState ⟨#[m, n], dtype, device⟩)
  | manifoldMuon (state : torch.Optim.ManifoldMuon.ParamState m n dtype device)
  | genericManifold (state : GenericManifoldState m n dtype device)
  deriving Repr

instance {m n : UInt64} {d : DType} {dev : DevicePolicy} :
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
def initGenericManifoldState {m n : UInt64} (_param : Tensor ⟨#[m, n], d, dev⟩) :
    GenericManifoldState m n d dev := {}

/-- Initialize matrix state according to the configured matrix backend. -/
def initMatrixParamState {m n : UInt64} (cfg : Config) (param : Tensor ⟨#[m, n], d, dev⟩) :
    MatrixParamState m n d dev :=
  if cfg.matrixOptimizer == .norMuon then
    .norMuon (NorMuon.initParamState param)
  else if usesSpecializedStiefelPath cfg then
    .manifoldMuon (torch.Optim.ManifoldMuon.initParamState param)
  else
    .genericManifold (initGenericManifoldState param)

/-- Extract per-parameter step count from matrix state. -/
def MatrixParamState.step {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (st : MatrixParamState m n d dev) : Nat :=
  match st with
  | .norMuon s => s.step
  | .manifoldMuon s => s.step
  | .genericManifold s => s.step

/-- Backend tag name for a matrix state (used in error messages). -/
def MatrixParamState.backendName {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (st : MatrixParamState m n d dev) : String :=
  match st with
  | .norMuon _ => "norMuon"
  | .manifoldMuon _ => "manifoldMuon"
  | .genericManifold _ => "genericManifold"

private def stepOnManifoldFamily {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (family : MatrixManifoldFamily)
    (param : Tensor ⟨#[m, n], d, dev⟩)
    (ambientGrad : Tensor ⟨#[m, n], d, dev⟩)
    (lr : Float)
    : IO (Tensor ⟨#[m, n], d, dev⟩) := do
  match family with
  | .stiefel =>
    if h : n.toNat ≤ m.toNat then
      let x : Tyr.AD.Stiefel m n d dev := ⟨param, h⟩
      let g := Tyr.AD.StiefelTangent.project x ambientGrad
      pure (Tyr.AD.DualMapGeometry.dualMapStep x g lr).matrix
    else
      -- Wide matrices have no column-orthonormal frame; step the transposed
      -- (row-orthonormal) Stiefel representative instead, matching Muon's
      -- usual tall/wide handling.
      let h' : m.toNat ≤ n.toNat := Nat.le_of_lt (Nat.lt_of_not_le h)
      let x : Tyr.AD.Stiefel n m d dev := ⟨Tensor.transpose2d param, h'⟩
      let g := Tyr.AD.StiefelTangent.project x (Tensor.transpose2d ambientGrad)
      pure (Tensor.transpose2d (Tyr.AD.DualMapGeometry.dualMapStep x g lr).matrix)
  | .orthogonal =>
    -- Orthogonal requires square matrices.
    if h : m = n then
      let pSquare : Tensor ⟨#[n, n], d, dev⟩ := by simpa [h] using param
      let gSquare : Tensor ⟨#[n, n], d, dev⟩ := by simpa [h] using ambientGrad
      let x : Tyr.AD.Orthogonal n d dev := ⟨pSquare⟩
      let tg := Tyr.AD.OrthogonalTangent.fromAmbient x gSquare
      let x' := Tyr.AD.DualMapGeometry.dualMapStep x tg lr
      pure (by simpa [h] using x'.matrix : Tensor ⟨#[m, n], d, dev⟩)
    else
      throw <| IO.userError s!"MatrixManifoldFamily.orthogonal requires square matrices, got {m}x{n}"
  | .grassmann =>
    if h : n.toNat ≤ m.toNat then
      let x : Tyr.AD.Grassmann m n d dev := ⟨param, h⟩
      let g := Tyr.AD.GrassmannTangent.project x ambientGrad
      pure (Tyr.AD.DualMapGeometry.dualMapStep x g lr).matrix
    else
      let h' : m.toNat ≤ n.toNat := Nat.le_of_lt (Nat.lt_of_not_le h)
      let x : Tyr.AD.Grassmann n m d dev := ⟨Tensor.transpose2d param, h'⟩
      let g := Tyr.AD.GrassmannTangent.project x (Tensor.transpose2d ambientGrad)
      pure (Tensor.transpose2d (Tyr.AD.DualMapGeometry.dualMapStep x g lr).matrix)

/--
Single matrix step for the generic manifold path using `DualMapGeometry`.

This path is intentionally simple and composable:
- shared momentum blending from NorMuon,
- manifold-family-specific projection/retraction from `Tyr.Manifolds`,
- no specialized dual-ascent state.
-/
def stepGenericManifoldSingle {m n : UInt64} {d : DType} {dev : DevicePolicy}
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
  let nesterovGrad := Tensor.Preserving.add (Tensor.Preserving.mulScalar gRaw (1.0 - momentum))
    (Tensor.Preserving.mulScalar newMomentum momentum)
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
def stepMatrixSingle {m n : UInt64} {d : DType} {dev : DevicePolicy}
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
def stepMatrixGroupLocal {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (params : Array (Tensor ⟨#[m, n], d, dev⟩))
    (grads : Array (Tensor ⟨#[m, n], d, dev⟩))
    (states : Array (MatrixParamState m n d dev))
    (cfg : Config)
    (step : Nat)
    (lrMul : Float := 1.0)
    (wdMul : Float := 1.0)
    : IO (Array (Tensor ⟨#[m, n], d, dev⟩) × Array (MatrixParamState m n d dev)) := do
  let mut outParams : Array (Tensor ⟨#[m, n], d, dev⟩) := #[]
  let mut outStates : Array (MatrixParamState m n d dev) := #[]
  for i in [:params.size] do
    let p := params[i]!
    let g := grads[i]?.getD (Tensor.zerosLike p)
    let st := match states[i]? with
      | some s => s
      | none => initMatrixParamState cfg p
    let (p', st') ← stepMatrixSingle p g st cfg step lrMul wdMul
    outParams := outParams.push p'
    outStates := outStates.push st'
  return (outParams, outStates)

private def allNorMuonStates? {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (states : Array (MatrixParamState m n d dev)) :
    Option (Array (NorMuon.ParamState ⟨#[m, n], d, dev⟩)) :=
  Id.run do
    let mut out : Array (NorMuon.ParamState ⟨#[m, n], d, dev⟩) := #[]
    for st in states do
      match st with
      | .norMuon s => out := out.push s
      | _ => return none
    return some out

private def allManifoldMuonStates? {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (states : Array (MatrixParamState m n d dev)) :
    Option (Array (torch.Optim.ManifoldMuon.ParamState m n d dev)) :=
  Id.run do
    let mut out : Array (torch.Optim.ManifoldMuon.ParamState m n d dev) := #[]
    for st in states do
      match st with
      | .manifoldMuon s => out := out.push s
      | _ => return none
    return some out

/--
Distributed matrix-group step with backend-aware dispatch.

Fast-paths:
- NorMuon uses NorMuon's distributed owner update path.
- Specialized Stiefel manifold-Muon uses its distributed path.

Fallback:
- generic manifold path does gradient all-reduce then local per-parameter updates.
-/
def stepMatrixGroupDistributed {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (params : Array (Tensor ⟨#[m, n], d, dev⟩))
    (grads : Array (Tensor ⟨#[m, n], d, dev⟩))
    (states : Array (MatrixParamState m n d dev))
    (cfg : Config)
    (step : Nat)
    (lrMul : Float := 1.0)
    (wdMul : Float := 1.0)
    : IO (Array (Tensor ⟨#[m, n], d, dev⟩) × Array (MatrixParamState m n d dev)) := do
  if params.isEmpty then
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
          let mut reducedGrads : Array (Tensor ⟨#[m, n], d, dev⟩) := #[]
          for i in [:params.size] do
            let p := params[i]!
            let g := Tensor.detach (grads[i]?.getD (Tensor.zerosLike p))
            let g ← dist.allReduce g .avg
            reducedGrads := reducedGrads.push g
          stepMatrixGroupLocal params reducedGrads states cfg step lrMul wdMul

/-- Composable matrix backend operations for a fixed shape. -/
structure MatrixBackendOps (m n : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  initState : Tensor ⟨#[m, n], dtype, device⟩ → MatrixParamState m n dtype device
  stepSingle : Tensor ⟨#[m, n], dtype, device⟩ → Tensor ⟨#[m, n], dtype, device⟩ →
    MatrixParamState m n dtype device →
    IO (Tensor ⟨#[m, n], dtype, device⟩ × MatrixParamState m n dtype device)
  stepGroupLocal : Array (Tensor ⟨#[m, n], dtype, device⟩) → Array (Tensor ⟨#[m, n], dtype, device⟩) →
    Array (MatrixParamState m n dtype device) →
    IO (Array (Tensor ⟨#[m, n], dtype, device⟩) × Array (MatrixParamState m n dtype device))
  stepGroupDistributed : Array (Tensor ⟨#[m, n], dtype, device⟩) → Array (Tensor ⟨#[m, n], dtype, device⟩) →
    Array (MatrixParamState m n dtype device) →
    IO (Array (Tensor ⟨#[m, n], dtype, device⟩) × Array (MatrixParamState m n dtype device))

/--
Build a backend operations bundle for a matrix parameter shape.

This keeps optimizer composition explicit: call-sites can choose backend ops once,
then reuse the same `init/step` closures regardless of matrix backend kind.
-/
def matrixBackendOps {m n : UInt64} {d : DType} {dev : DevicePolicy}
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

/-- Build matrix backend ops for a specific parameter label from a backend spec. -/
def matrixBackendOpsForLabel {m n : UInt64} {d : DType} {dev : DevicePolicy}
    (spec : MatrixBackendSpec)
    (label : ParamLabel)
    (step : Nat)
    (lrMul : Float := 1.0)
    (wdMul : Float := 1.0)
    : MatrixBackendOps m n d dev :=
  matrixBackendOps (cfg := spec.configForLabel label) (step := step) (lrMul := lrMul) (wdMul := wdMul)

end torch.Optim.DualOptimizer
