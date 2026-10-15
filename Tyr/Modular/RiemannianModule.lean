import Tyr.Modular.MetricFactor
import Tyr.Modular.Compose
import Tyr.Modular.Manifold
import Tyr.Module.Linear
import Tyr.Manifolds.Embedded

namespace Tyr.Modular

open torch
open Tyr.AD

/-- Local linearization data for one module at one forward point. -/
structure LocalLinearization (inDim paramDim outDim : UInt64) where
  applyA : AmbientVector inDim → AmbientVector outDim
  applyAT : AmbientVector outDim → AmbientVector inDim
  applyB : AmbientVector paramDim → AmbientVector outDim
  applyBT : AmbientVector outDim → AmbientVector paramDim

/-- Basic diagnostics for one layerwise Riemannian update. -/
structure LayerStepDiagnostics where
  gradientNorm : Float := 0.0
  factorRank : UInt64 := 0
  innerConditionEstimate : Float := 0.0
  residualNorm : Float := 0.0
  updateNorm : Float := 0.0
  deriving Repr, Inhabited

/--
Leaf module that can expose the local Jacobians needed for recursive
metric-factor propagation.
-/
class RiemannianModule (M : Type) (inDim outDim : UInt64) where
  paramDim : UInt64
  Tape : Type
  forwardWithTape : M → AmbientVector inDim → AmbientVector outDim × Tape
  localLinearization : M → Tape → LocalLinearization inDim paramDim outDim
  paramMass : M → DiagonalMass paramDim
  applyAmbientUpdate : M → AmbientVector paramDim → Float → IO M

namespace RiemannianModule

private def vectorToColumn {n : UInt64} {d : DType} {dev : DevicePolicy}
    (v : Tensor ⟨#[n], d, dev⟩) : Tensor ⟨#[n, 1], d, dev⟩ :=
  Tensor.reshape v #[n, 1] (by simp [TensorSpec.numelOfShape])

private def columnToVector {n : UInt64} {d : DType} {dev : DevicePolicy}
    (v : Tensor ⟨#[n, 1], d, dev⟩) : Tensor ⟨#[n], d, dev⟩ :=
  Tensor.reshape v #[n] (by simp [TensorSpec.numelOfShape])

private def splitLinearUpdate {inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (update : Tensor ⟨#[outDim * inDim + outDim], d, dev⟩) :
    Tensor ⟨#[outDim, inDim], d, dev⟩ × Tensor ⟨#[outDim], d, dev⟩ :=
  let weightSize := outDim * inDim
  let dwVec : Tensor ⟨#[weightSize], d, dev⟩ := Tensor.sliceChecked update 0 0 weightSize #[weightSize]
  let dbVec : Tensor ⟨#[outDim], d, dev⟩ := Tensor.sliceChecked update 0 weightSize outDim #[outDim]
  (Tensor.reshapeChecked dwVec #[outDim, inDim], dbVec)

namespace LocalLinearization

private def rowAt {rows cols : UInt64} {d : DType} {dev : DevicePolicy}
    (matrix : Tensor ⟨#[rows, cols], d, dev⟩) (idx : UInt64) : Tensor ⟨#[cols], d, dev⟩ :=
  Tensor.reshapeChecked (Tensor.sliceChecked matrix 0 idx 1 #[1, cols]) #[cols]

private def mapRows {rows inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (matrix : Tensor ⟨#[rows, inDim], d, dev⟩)
    (f : Tensor ⟨#[inDim], d, dev⟩ → Tensor ⟨#[outDim], d, dev⟩) :
    Tensor ⟨#[rows, outDim], d, dev⟩ :=
  let mapped : Array (Tensor ⟨#[outDim], d, dev⟩) := Id.run do
    let mut acc : Array (Tensor ⟨#[outDim], d, dev⟩) := #[]
    for i in [:rows.toNat] do
      acc := acc.push (f (rowAt matrix (UInt64.ofNat i)))
    pure acc
  let stacked := Tensor.stack1d mapped
  Tensor.reshapeChecked stacked #[rows, outDim]

/-- Materialize the dense input Jacobian for debugging/tests. -/
def materializeA {inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim) : AmbientMatrix outDim inDim :=
  let basis : AmbientMatrix inDim inDim := Tensor.eye inDim
  Tensor.transpose2d (mapRows basis lin.applyA)

/-- Materialize the dense parameter Jacobian for debugging/tests. -/
def materializeB {inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim) : AmbientMatrix outDim paramDim :=
  let basis : AmbientMatrix paramDim paramDim := Tensor.eye paramDim
  Tensor.transpose2d (mapRows basis lin.applyB)

/-- Pull a metric factor back through the input Jacobian using rowwise adjoints. -/
def pullbackInputFactor {rank inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim)
    (L : MetricFactor rank outDim) : MetricFactor rank inDim :=
  { matrix := mapRows L.matrix lin.applyAT }

/-- Pull a metric factor back through the parameter Jacobian using rowwise adjoints. -/
def pullbackParamFactor {rank inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim)
    (L : MetricFactor rank outDim) : MetricFactor rank paramDim :=
  { matrix := mapRows L.matrix lin.applyBT }

end LocalLinearization

private def linearApplyInput {inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩) (x : Tensor ⟨#[inDim], d, dev⟩) :
    Tensor ⟨#[outDim], d, dev⟩ :=
  columnToVector (Tensor.mm weight (vectorToColumn x))

private def linearApplyInputAdjoint {inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩) (y : Tensor ⟨#[outDim], d, dev⟩) :
    Tensor ⟨#[inDim], d, dev⟩ :=
  columnToVector (Tensor.mm (Tensor.transpose2d weight) (vectorToColumn y))

private def linearApplyParam {inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[inDim], d, dev⟩)
    (useBias : Bool)
    (update : Tensor ⟨#[outDim * inDim + outDim], d, dev⟩) : Tensor ⟨#[outDim], d, dev⟩ :=
  let (dW, dB) := splitLinearUpdate (inDim := inDim) (outDim := outDim) update
  let weightPart := linearApplyInput dW x
  if useBias then
    Tensor.Preserving.add weightPart dB
  else
    weightPart

private def linearApplyParamAdjoint {inDim outDim : UInt64} {d : DType} {dev : DevicePolicy}
    (x : Tensor ⟨#[inDim], d, dev⟩)
    (useBias : Bool)
    (y : Tensor ⟨#[outDim], d, dev⟩) : Tensor ⟨#[outDim * inDim + outDim], d, dev⟩ :=
  let outer : Tensor ⟨#[outDim, inDim], d, dev⟩ :=
    Tensor.mm (vectorToColumn y) (Tensor.reshape x #[1, inDim] (by simp [TensorSpec.numelOfShape]))
  let weightPart : Tensor ⟨#[outDim * inDim], d, dev⟩ := Tensor.reshapeChecked outer #[outDim * inDim]
  let biasPart : Tensor ⟨#[outDim], d, dev⟩ := if useBias then y else Tensor.zerosFor _
  let joined := Tensor.cat weightPart biasPart 0
  Tensor.reshapeChecked joined #[outDim * inDim + outDim]

private def makeManifoldLeaf [MatrixManifoldCarrier M] {m n : UInt64}
    (w : M m n) : IO (M m n) := do
  let leaf ← Tensor.parameter (MatrixManifoldCarrier.toMatrix w)
  pure (MatrixManifoldCarrier.reAmbient w leaf)

/-- Single leaf backward step: metric solve, parameter update, and propagation. -/
def backwardLeafStep {M : Type} {inDim outDim : UInt64} [RiemannianModule M inDim outDim]
    {rank : UInt64}
    (m : M)
    (tape : RiemannianModule.Tape (M := M) (inDim := inDim) (outDim := outDim))
    (L : MetricFactor rank outDim)
    (lambda : AmbientVector outDim)
    (lr : Float)
    : IO (M × MetricFactor rank inDim × AmbientVector inDim × LayerStepDiagnostics) := do
  let lin := RiemannianModule.localLinearization (M := M) (inDim := inDim) (outDim := outDim) m tape
  let g := lin.applyBT lambda
  let K := LocalLinearization.pullbackParamFactor lin L
  let mass := RiemannianModule.paramMass (M := M) (inDim := inDim) (outDim := outDim) m
  let update := MetricFactor.solveWoodbury mass K g
  let m' ← RiemannianModule.applyAmbientUpdate (M := M) (inDim := inDim) (outDim := outDim) m update lr
  let LPrev := LocalLinearization.pullbackInputFactor lin L
  let lambdaPrev := lin.applyAT lambda
  let inner := MetricFactor.woodburyInner mass K
  let innerInv := Tensor.inv inner
  let residual := Sub.sub (MetricFactor.applyRegularized mass K update) g
  let stats : LayerStepDiagnostics := {
    gradientNorm := Tensor.l2Norm g
    factorRank := rank
    innerConditionEstimate := Tensor.spectralNorm inner * Tensor.spectralNorm innerInv
    residualNorm := Tensor.l2Norm residual
    updateNorm := Tensor.l2Norm update
  }
  pure (m', LPrev, lambdaPrev, stats)

instance linearRiemannianModule (inDim outDim : UInt64) :
    RiemannianModule (torch.Linear inDim outDim) inDim outDim where
  paramDim := outDim * inDim + outDim
  Tape := AmbientVector inDim
  forwardWithTape lin x :=
    let x2 : AmbientMatrix 1 inDim :=
      Tensor.reshape x #[1, inDim] (by simp [TensorSpec.numelOfShape])
    let y2 : AmbientMatrix 1 outDim :=
      match lin.bias with
      | some b => Add.add (Tensor.linear x2 lin.weight) (Tensor.expand b #[1, outDim])
      | none => Tensor.linear x2 lin.weight
    (Tensor.reshape y2 #[outDim] (by simp [TensorSpec.numelOfShape]), x)
  localLinearization lin x := {
    applyA := linearApplyInput lin.weight
    applyAT := linearApplyInputAdjoint lin.weight
    applyB := linearApplyParam x lin.bias.isSome
    applyBT := linearApplyParamAdjoint x lin.bias.isSome
  }
  paramMass _ := DiagonalMass.ones (outDim * inDim + outDim)
  applyAmbientUpdate lin update lr := do
    let (dW, dB) := splitLinearUpdate (inDim := inDim) (outDim := outDim) update
    let weight' ← Tensor.parameter (Sub.sub lin.weight (Tensor.Preserving.mulScalar dW lr))
    let bias' ←
      match lin.bias with
      | some b => do
        pure (some (← Tensor.parameter (Sub.sub b (Tensor.Preserving.mulScalar dB lr))))
      | none => pure none
    pure { lin with weight := weight', bias := bias' }

instance manifoldLinearRiemannianModule
    (M : UInt64 → UInt64 → Type)
    [MatrixManifoldCarrier M]
    {inDim outDim : UInt64}
    :
    RiemannianModule (ManifoldLinear M inDim outDim) inDim outDim where
  paramDim := outDim * inDim + outDim
  Tape := AmbientVector inDim
  forwardWithTape lin x :=
    let weight := MatrixManifoldCarrier.toMatrix lin.weight
    let x2 : AmbientMatrix 1 inDim :=
      Tensor.reshape x #[1, inDim] (by simp [TensorSpec.numelOfShape])
    let y2 : AmbientMatrix 1 outDim :=
      match lin.bias with
      | some b => Add.add (Tensor.linear x2 weight) (Tensor.expand b #[1, outDim])
      | none => Tensor.linear x2 weight
    (Tensor.reshape y2 #[outDim] (by simp [TensorSpec.numelOfShape]), x)
  localLinearization lin x := {
    applyA := linearApplyInput (MatrixManifoldCarrier.toMatrix lin.weight)
    applyAT := linearApplyInputAdjoint (MatrixManifoldCarrier.toMatrix lin.weight)
    applyB := linearApplyParam x lin.bias.isSome
    applyBT := linearApplyParamAdjoint x lin.bias.isSome
  }
  paramMass _ := DiagonalMass.ones (outDim * inDim + outDim)
  applyAmbientUpdate lin update lr := do
    let (dW, dB) := splitLinearUpdate (inDim := inDim) (outDim := outDim) update
    let updatedMatrix := Sub.sub (MatrixManifoldCarrier.toMatrix lin.weight)
      (Tensor.Preserving.mulScalar dW lr)
    let weight' := MatrixManifoldCarrier.projectLike lin.weight updatedMatrix
    let weight' ← makeManifoldLeaf weight'
    let bias' ←
      match lin.bias with
      | some b => do
        pure (some (← Tensor.parameter (Sub.sub b (Tensor.Preserving.mulScalar dB lr))))
      | none => pure none
    pure { lin with weight := weight', bias := bias' }

end RiemannianModule

namespace LocalLinearization

/-- Materialize the dense input Jacobian for debugging/tests. -/
def materializeA {inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim) : AmbientMatrix outDim inDim :=
  RiemannianModule.LocalLinearization.materializeA lin

/-- Materialize the dense parameter Jacobian for debugging/tests. -/
def materializeB {inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim) : AmbientMatrix outDim paramDim :=
  RiemannianModule.LocalLinearization.materializeB lin

/-- Pull a metric factor back through the input Jacobian using rowwise adjoints. -/
def pullbackInputFactor {rank inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim)
    (L : MetricFactor rank outDim) : MetricFactor rank inDim :=
  RiemannianModule.LocalLinearization.pullbackInputFactor lin L

/-- Pull a metric factor back through the parameter Jacobian using rowwise adjoints. -/
def pullbackParamFactor {rank inDim paramDim outDim : UInt64}
    (lin : LocalLinearization inDim paramDim outDim)
    (L : MetricFactor rank outDim) : MetricFactor rank paramDim :=
  RiemannianModule.LocalLinearization.pullbackParamFactor lin L

end LocalLinearization

/-- Forward tape for a two-layer sequential composition. -/
abbrev SequentialTape
    (M₁ : Type) (inDim midDim : UInt64)
    (M₂ : Type) (outDim : UInt64)
    [RiemannianModule M₁ inDim midDim]
    [RiemannianModule M₂ midDim outDim]
    :=
  RiemannianModule.Tape (M := M₁) (inDim := inDim) (outDim := midDim) ×
    RiemannianModule.Tape (M := M₂) (inDim := midDim) (outDim := outDim)

/-- Forward pass with tapes for a two-layer sequential composition. -/
def sequentialForwardWithTape
    {M₁ M₂ : Type}
    {inDim midDim outDim : UInt64}
    [RiemannianModule M₁ inDim midDim]
    [RiemannianModule M₂ midDim outDim]
    (net : Sequential M₁ M₂)
    (x : AmbientVector inDim)
    : AmbientVector outDim × SequentialTape M₁ inDim midDim M₂ outDim := by
  let (mid, tape₁) :=
    RiemannianModule.forwardWithTape (M := M₁) (inDim := inDim) (outDim := midDim) net.first x
  let (y, tape₂) :=
    RiemannianModule.forwardWithTape (M := M₂) (inDim := midDim) (outDim := outDim) net.second mid
  exact (y, (tape₁, tape₂))

/-- Backward recursion for a two-layer sequential composition. -/
def sequentialBackwardMetricStep
    {M₁ M₂ : Type}
    {rank inDim midDim outDim : UInt64}
    [RiemannianModule M₁ inDim midDim]
    [RiemannianModule M₂ midDim outDim]
    (net : Sequential M₁ M₂)
    (tape : SequentialTape M₁ inDim midDim M₂ outDim)
    (L : MetricFactor rank outDim)
    (lambda : AmbientVector outDim)
    (lr : Float)
    : IO (Sequential M₁ M₂ × MetricFactor rank inDim × AmbientVector inDim × Array LayerStepDiagnostics) := do
  let (tape₁, tape₂) := tape
  let (second', midFactor, midCotangent, stats₂) ←
    RiemannianModule.backwardLeafStep net.second tape₂ L lambda lr
  let (first', inFactor, inCotangent, stats₁) ←
    RiemannianModule.backwardLeafStep net.first tape₁ midFactor midCotangent lr
  pure ({ first := first', second := second' }, inFactor, inCotangent, #[stats₁, stats₂])

end Tyr.Modular
