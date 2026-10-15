/-
  Tyr/Optim/ManifoldMuon.lean

  Experimental Stiefel manifold variant of Muon.

  Design goals:
  - Keep matrix parameters on/near Stiefel via tangent projection + QR retraction.
  - Support an optional lightweight dual-ascent loop for tangent-constrained
    spectral-style updates.
  - Keep API shape close to NorMuon for easy experimentation.
-/
import Tyr.TensorStruct
import Tyr.Distributed
import Tyr.Optim.PolarExpress
import Tyr.Optim.NorMuon
import Tyr.Manifolds.Optimizer

namespace torch.Optim.ManifoldMuon

open torch

variable {d : DType} {dev : Device}

/-- Dual solve backend used inside manifold-Muon. -/
inductive SolverKind where
  | dualAscent
  | fixedPoint
  deriving Repr, BEq, Inhabited

/-- Manifold-Muon configuration (experimental). -/
structure Config where
  /-- Base learning rate. -/
  lr : Float := 0.02
  /-- Momentum coefficient for Nesterov-style blending. -/
  momentum : Float := 0.95
  /-- Newton-Schulz iterations used by matrix-sign approximation. -/
  numIters : UInt64 := 5
  /-- Number of dual-ascent iterations for tangent-constrained solve. -/
  dualAscentSteps : Nat := 2
  /-- Step size for dual-ascent variable update. -/
  dualAscentLr : Float := 0.1
  /-- Solver backend for the tangent-constrained inner problem. -/
  solver : SolverKind := .dualAscent
  /-- Minimum inner iterations before checking stopping criteria. -/
  minSolveSteps : Nat := 1
  /-- Residual tolerance for tangent-constraint satisfaction. -/
  solveResidualTol : Float := 1e-4
  /-- Dual-variable delta tolerance for solver convergence. -/
  solveDualDeltaTol : Float := 1e-5
  /-- Damping used by fixed-point updates. -/
  fixedPointDamping : Float := 0.5
  /-- Whether to run distributed gradient averaging before local step. -/
  distributed : Bool := false
  deriving Repr, Inhabited

/-- Diagnostics for the inner tangent-constrained solve. -/
structure SolveDiagnostics where
  iterations : Nat := 0
  residual : Float := 0.0
  dualDelta : Float := 0.0
  dualObjective : Float := 0.0
  converged : Bool := false
  solver : SolverKind := .dualAscent
  deriving Repr, Inhabited

/-- Result of solving for a constrained manifold update direction. -/
structure SolveResult (m n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  direction : Tensor ⟨#[m, n], dtype, device⟩
  dualVar : Tensor ⟨#[n, n], dtype, device⟩
  diagnostics : SolveDiagnostics
  deriving Repr

/-- Per-parameter state for Stiefel Manifold-Muon. -/
structure ParamState (m n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  /-- Momentum buffer in ambient space. -/
  momentumBuffer : Option (Tensor ⟨#[m, n], dtype, device⟩) := none
  /-- Dual variable for tangent-constrained solve. -/
  dualVar : Option (Tensor ⟨#[n, n], dtype, device⟩) := none
  /-- Step counter. -/
  step : Nat := 0
  deriving Repr

/-- Initialize per-parameter state. -/
def initParamState {m n : UInt64} (_param : Tensor ⟨#[m, n], d, dev⟩) : ParamState m n d dev := {
  momentumBuffer := none
  dualVar := none
  step := 0
}

instance {m n : UInt64} {d : DType} {dev : Device} :
    TensorStruct (ParamState m n d dev) where
  map f s := {
    momentumBuffer := s.momentumBuffer.map (TensorStruct.map f)
    dualVar := s.dualVar.map (TensorStruct.map f)
    step := s.step
  }
  mapM f s := do
    let momentumBuffer ← match s.momentumBuffer with
      | some t => some <$> TensorStruct.mapM f t
      | none => pure none
    let dualVar ← match s.dualVar with
      | some t => some <$> TensorStruct.mapM f t
      | none => pure none
    pure { s with momentumBuffer, dualVar }
  zipWith f a b := {
    momentumBuffer := match a.momentumBuffer, b.momentumBuffer with
      | some t1, some t2 => some (TensorStruct.zipWith f t1 t2)
      | _, _ => none
    dualVar := match a.dualVar, b.dualVar with
      | some t1, some t2 => some (TensorStruct.zipWith f t1 t2)
      | _, _ => none
    step := a.step
  }
  fold f init s :=
    let acc := match s.momentumBuffer with
      | some t => TensorStruct.fold f init t
      | none => init
    match s.dualVar with
    | some t => TensorStruct.fold f acc t
    | none => acc

/-- Symmetric part of a square matrix. -/
def sym {n : UInt64} (A : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n, n], d, dev⟩ :=
  Tensor.Preserving.mulScalar (Tensor.Preserving.add A (Tensor.transpose2d A)) 0.5

/-- Project an ambient matrix to the Stiefel tangent space at W. -/
def tangentProject {m n : UInt64} (W : Tensor ⟨#[m, n], d, dev⟩) (G : Tensor ⟨#[m, n], d, dev⟩) :
    Tensor ⟨#[m, n], d, dev⟩ :=
  let XtG := Tensor.mm (Tensor.transpose2d W) G
  let correction := Tensor.mm W (sym XtG)
  Tensor.Preserving.sub G correction

/-- Tangent-constraint residual matrix: `WᵀA + AᵀW`. -/
def tangentResidual {m n : UInt64} (W : Tensor ⟨#[m, n], d, dev⟩) (A : Tensor ⟨#[m, n], d, dev⟩) :
    Tensor ⟨#[n, n], d, dev⟩ :=
  Tensor.Preserving.add (Tensor.mm (Tensor.transpose2d W) A) (Tensor.mm (Tensor.transpose2d A) W)

private def solveConverged (cfg : Config) (iter residual dualDelta : Float) : Bool :=
  iter >= cfg.minSolveSteps.toFloat &&
    residual <= cfg.solveResidualTol &&
    dualDelta <= cfg.solveDualDeltaTol

/-- Dual objective estimate used for diagnostics. -/
private def dualObjectiveEstimate {m n : UInt64} (W G : Tensor ⟨#[m, n], d, dev⟩)
    (lambda : Tensor ⟨#[n, n], d, dev⟩) : Float :=
  let lambdaPlus := Tensor.Preserving.add lambda (Tensor.transpose2d lambda)
  let adjusted := Tensor.Preserving.add G (Tensor.Preserving.mulScalar (Tensor.mm W lambdaPlus) 2.0)
  0.0 - Tensor.nuclearNorm adjusted

/--
Run lightweight dual ascent and return matrix-sign direction plus updated dual var.
-/
def solveDirectionDualAscent {m n : UInt64}
    (W : Tensor ⟨#[m, n], d, dev⟩) (G : Tensor ⟨#[m, n], d, dev⟩)
    (dual0 : Tensor ⟨#[n, n], d, dev⟩) (cfg : Config)
    : IO (SolveResult m n d dev) := do
  let mut lambda := dual0
  let mut signedDir := G
  let mut residual := 1e30
  let mut dualDelta := 1e30
  let mut iterations : Nat := 0
  let mut converged := false
  for i in [:cfg.dualAscentSteps] do
    if !converged then
      let lambdaPrev := lambda
      let lambdaPlus := Tensor.Preserving.add lambda (Tensor.transpose2d lambda)
      let adjusted := Tensor.Preserving.add G (Tensor.Preserving.mulScalar (Tensor.mm W lambdaPlus) 2.0)
      signedDir ← PolarExpress.apply adjusted { numIters := cfg.numIters }
      let h := tangentResidual W signedDir
      lambda := Tensor.Preserving.add lambda (Tensor.Preserving.mulScalar h cfg.dualAscentLr)
      residual := Tensor.frobeniusNorm h
      dualDelta := Tensor.frobeniusNorm (Tensor.Preserving.sub lambda lambdaPrev)
      iterations := i + 1
      converged := solveConverged cfg iterations.toFloat residual dualDelta
  let diag : SolveDiagnostics := {
    iterations := iterations
    residual := residual
    dualDelta := dualDelta
    dualObjective := dualObjectiveEstimate W G lambda
    converged := converged
    solver := .dualAscent
  }
  return { direction := signedDir, dualVar := lambda, diagnostics := diag }

/--
Fixed-point solver variant for `H(Λ)=0` style updates with damping.
-/
def solveDirectionFixedPoint {m n : UInt64}
    (W : Tensor ⟨#[m, n], d, dev⟩) (G : Tensor ⟨#[m, n], d, dev⟩)
    (dual0 : Tensor ⟨#[n, n], d, dev⟩) (cfg : Config)
    : IO (SolveResult m n d dev) := do
  let mut lambda := dual0
  let mut signedDir := G
  let mut residual := 1e30
  let mut dualDelta := 1e30
  let mut iterations : Nat := 0
  let mut converged := false
  let damp := if cfg.fixedPointDamping < 0.0 then 0.0
    else if cfg.fixedPointDamping > 1.0 then 1.0 else cfg.fixedPointDamping
  for i in [:cfg.dualAscentSteps] do
    if !converged then
      let lambdaPrev := lambda
      let lambdaPlus := Tensor.Preserving.add lambda (Tensor.transpose2d lambda)
      let adjusted := Tensor.Preserving.add G (Tensor.Preserving.mulScalar (Tensor.mm W lambdaPlus) 2.0)
      signedDir ← PolarExpress.apply adjusted { numIters := cfg.numIters }
      let h := tangentResidual W signedDir
      let candidate := Tensor.Preserving.sub lambda (Tensor.Preserving.mulScalar h cfg.dualAscentLr)
      lambda := Tensor.Preserving.add (Tensor.Preserving.mulScalar lambda (1.0 - damp))
        (Tensor.Preserving.mulScalar candidate damp)
      residual := Tensor.frobeniusNorm h
      dualDelta := Tensor.frobeniusNorm (Tensor.Preserving.sub lambda lambdaPrev)
      iterations := i + 1
      converged := solveConverged cfg iterations.toFloat residual dualDelta
  let diag : SolveDiagnostics := {
    iterations := iterations
    residual := residual
    dualDelta := dualDelta
    dualObjective := dualObjectiveEstimate W G lambda
    converged := converged
    solver := .fixedPoint
  }
  return { direction := signedDir, dualVar := lambda, diagnostics := diag }

/-- Solve for direction using the configured inner solver backend. -/
def solveDirection {m n : UInt64}
    (W : Tensor ⟨#[m, n], d, dev⟩) (G : Tensor ⟨#[m, n], d, dev⟩)
    (dual0 : Tensor ⟨#[n, n], d, dev⟩) (cfg : Config)
    : IO (SolveResult m n d dev) := do
  match cfg.solver with
  | .dualAscent => solveDirectionDualAscent W G dual0 cfg
  | .fixedPoint => solveDirectionFixedPoint W G dual0 cfg

/-- Retraction onto Stiefel using reduced QR. Tall matrices (`n ≤ m`) retract
    column-wise; wide matrices retract row-wise via the transposed projection,
    matching Muon's usual tall/wide handling. -/
def retractStiefel {m n : UInt64} (W : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[m, n], d, dev⟩ :=
  if h : n.toNat ≤ m.toNat then
    (Tyr.AD.Stiefel.project m n W h).matrix
  else
    let h' : m.toNat ≤ n.toNat := Nat.le_of_lt (Nat.lt_of_not_le h)
    Tensor.transpose2d (Tyr.AD.Stiefel.project n m (Tensor.transpose2d W) h').matrix

/-- Single update step with inner-solver diagnostics. -/
def stepSingleWithDiagnostics {m n : UInt64}
    (param : Tensor ⟨#[m, n], d, dev⟩)
    (grad : Tensor ⟨#[m, n], d, dev⟩)
    (state : ParamState m n d dev)
    (cfg : Config)
    (lrMul : Float := 1.0)
    : IO (Tensor ⟨#[m, n], d, dev⟩ × ParamState m n d dev × SolveDiagnostics) := do
  let gRaw := Tensor.detach grad

  -- Nesterov-style blend (aligned with NorMuon's momentum semantics).
  let newMomentum := NorMuon.updateMomentum gRaw state.momentumBuffer cfg.momentum
  let nesterovGrad := Tensor.Preserving.add (Tensor.Preserving.mulScalar gRaw (1.0 - cfg.momentum))
    (Tensor.Preserving.mulScalar newMomentum cfg.momentum)

  -- Optional dual-ascent solve in ambient space.
  let dual0 := state.dualVar.getD (Tensor.zerosFor ⟨#[n, n], d, dev⟩)
  let (signedDir, dualVar', solveDiag) ←
    if cfg.dualAscentSteps == 0 then
      let signed ← PolarExpress.apply nesterovGrad { numIters := cfg.numIters }
      let diag : SolveDiagnostics := {
        iterations := 0
        residual := Tensor.frobeniusNorm (tangentResidual param signed)
        dualDelta := 0.0
        dualObjective := dualObjectiveEstimate param nesterovGrad dual0
        converged := true
        solver := cfg.solver
      }
      pure (signed, dual0, diag)
    else
      let solveRes ← solveDirection param nesterovGrad dual0 cfg
      pure (solveRes.direction, solveRes.dualVar, solveRes.diagnostics)

  -- Enforce tangent constraint explicitly.
  let tangentDir := tangentProject param signedDir

  -- Keep step magnitude predictable.
  let fnorm := Tensor.frobeniusNorm tangentDir
  let tangentUnit :=
    if fnorm == 0.0 then
      tangentDir
    else
      Tensor.Preserving.mulScalar tangentDir (1.0 / fnorm)

  let aspectScale := NorMuon.aspectRatioScale #[m, n]
  let effectiveLr := cfg.lr * lrMul * aspectScale

  -- Ambient update + Stiefel retraction.
  let updated := Tensor.Preserving.sub param (Tensor.Preserving.mulScalar tangentUnit effectiveLr)
  let retracted := retractStiefel updated
  let newParam ← Tensor.parameter (Tensor.detach retracted)

  let newState : ParamState m n d dev := {
    momentumBuffer := some newMomentum
    dualVar := some dualVar'
    step := state.step + 1
  }
  return (newParam, newState, solveDiag)

/-- Single local update step for a Stiefel-constrained matrix parameter. -/
def stepSingle {m n : UInt64}
    (param : Tensor ⟨#[m, n], d, dev⟩)
    (grad : Tensor ⟨#[m, n], d, dev⟩)
    (state : ParamState m n d dev)
    (cfg : Config)
    (lrMul : Float := 1.0)
    : IO (Tensor ⟨#[m, n], d, dev⟩ × ParamState m n d dev) := do
  let (param', state', _) ← stepSingleWithDiagnostics param grad state cfg lrMul
  return (param', state')

/-- Local group update for homogeneous matrix shapes. -/
def stepGroupLocal {m n : UInt64}
    (params : Array (Tensor ⟨#[m, n], d, dev⟩))
    (grads : Array (Tensor ⟨#[m, n], d, dev⟩))
    (states : Array (ParamState m n d dev))
    (cfg : Config)
    (lrMul : Float := 1.0)
    : IO (Array (Tensor ⟨#[m, n], d, dev⟩) × Array (ParamState m n d dev)) := do
  let mut outParams : Array (Tensor ⟨#[m, n], d, dev⟩) := #[]
  let mut outStates : Array (ParamState m n d dev) := #[]
  for i in [:params.size] do
    let p := params[i]!
    let g := grads[i]?.getD (Tensor.zerosLike p)
    let st := states[i]?.getD (initParamState p)
    let (p', st') ← stepSingle p g st cfg lrMul
    outParams := outParams.push p'
    outStates := outStates.push st'
  return (outParams, outStates)

/--
Distributed group update (prototype): all-reduce gradients then local manifold step.
-/
def stepDistributedGroup {m n : UInt64}
    (params : Array (Tensor ⟨#[m, n], d, dev⟩))
    (grads : Array (Tensor ⟨#[m, n], d, dev⟩))
    (states : Array (ParamState m n d dev))
    (cfg : Config)
    (lrMul : Float := 1.0)
    : IO (Array (Tensor ⟨#[m, n], d, dev⟩) × Array (ParamState m n d dev)) := do
  if params.isEmpty then
    return (params, states)

  let isDist ← dist.isInitialized
  if !cfg.distributed || !isDist then
    return (← stepGroupLocal params grads states cfg lrMul)

  let worldSize ← dist.getWorldSize
  if worldSize <= 1 then
    return (← stepGroupLocal params grads states cfg lrMul)

  let mut reducedGrads : Array (Tensor ⟨#[m, n], d, dev⟩) := #[]
  for i in [:params.size] do
    let p := params[i]!
    let g := Tensor.detach (grads[i]?.getD (Tensor.zerosLike p))
    let g ← dist.allReduce g .avg
    reducedGrads := reducedGrads.push g

  stepGroupLocal params reducedGrads states cfg lrMul

/-- Measure average local step latency for one parameter (benchmark helper). -/
def benchmarkLocalStep {m n : UInt64}
    (numSteps : Nat := 5)
    (cfg : Config := {})
    : IO Float := do
  let p0 ← Tensor.randn #[m, n]
  let p0 ← Tensor.parameter p0
  let mut p := retractStiefel p0
  let mut st := initParamState p

  let t0 ← IO.monoNanosNow
  for _ in [:numSteps] do
    let g ← Tensor.randn #[m, n]
    let (p', st') ← stepSingle p g st cfg
    p := p'
    st := st'
  let t1 ← IO.monoNanosNow

  let elapsed := (t1 - t0).toFloat / 1000000.0
  if numSteps == 0 then
    return elapsed
  else
    return elapsed / numSteps.toFloat

end torch.Optim.ManifoldMuon
