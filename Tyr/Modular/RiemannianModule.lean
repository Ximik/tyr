import Tyr.Modular.MetricFactor
import Tyr.Modular.Compose
import Tyr.Modular.Manifold
import Tyr.Module.Linear
import Tyr.Manifolds.Embedded

namespace Tyr.Modular

open torch

variable {device : Device}
open Tyr.AD

/-- Local linearization data for one module at one forward point. -/
structure LocalLinearization (inDim paramDim outDim : Nat) where
  applyA : AmbientVector (device := device) inDim → AmbientVector (device := device) outDim
  applyAT : AmbientVector (device := device) outDim → AmbientVector (device := device) inDim
  applyB : AmbientVector (device := device) paramDim → AmbientVector (device := device) outDim
  applyBT : AmbientVector (device := device) outDim → AmbientVector (device := device) paramDim

/-- Basic diagnostics for one layerwise Riemannian update. -/
structure LayerStepDiagnostics where
  gradientNorm : Float := 0.0
  factorRank : Nat := 0
  innerConditionEstimate : Float := 0.0
  residualNorm : Float := 0.0
  updateNorm : Float := 0.0
  deriving Repr, Inhabited

/--
Leaf module that can expose the local Jacobians needed for recursive
metric-factor propagation.
-/
class RiemannianModule (M : Type) (inDim outDim : Nat)
    (device : outParam Device) where
  paramDim : Nat
  Tape : Type
  forwardWithTape : M → AmbientVector (device := device) inDim → AmbientVector (device := device) outDim × Tape
  localLinearization : M → Tape → LocalLinearization (device := device) inDim paramDim outDim
  paramMass : M → DiagonalMass paramDim .Float32 device
  applyAmbientUpdate : M → AmbientVector (device := device) paramDim → Float → IO M

namespace RiemannianModule

private def vectorToColumn {n : Nat} {d : DType} {dev : Device}
    (v : Tensor ⟨#[n], d, dev⟩) : Tensor ⟨#[n, 1], d, dev⟩ :=
  Tensor.reshape v #[n, 1] (by simp [TensorSpec.numelOfShape])

private def columnToVector {n : Nat} {d : DType} {dev : Device}
    (v : Tensor ⟨#[n, 1], d, dev⟩) : Tensor ⟨#[n], d, dev⟩ :=
  Tensor.reshape v #[n] (by simp [TensorSpec.numelOfShape])

private def splitLinearUpdate {inDim outDim : Nat} {d : DType} {dev : Device}
    (update : Tensor ⟨#[outDim * inDim + outDim], d, dev⟩) :
    Tensor ⟨#[outDim, inDim], d, dev⟩ × Tensor ⟨#[outDim], d, dev⟩ :=
  let weightSize := outDim * inDim
  let dwVec : Tensor ⟨#[weightSize], d, dev⟩ := Tensor.sliceChecked update 0 0 weightSize #[weightSize]
  let dbVec : Tensor ⟨#[outDim], d, dev⟩ := Tensor.sliceChecked update 0 weightSize outDim #[outDim]
  (Tensor.reshape dwVec #[outDim, inDim], dbVec)

namespace LocalLinearization

private def rowAt {rows cols : Nat} {d : DType} {dev : Device}
    (matrix : Tensor ⟨#[rows, cols], d, dev⟩) (idx : Nat) : Tensor ⟨#[cols], d, dev⟩ :=
  Tensor.reshape (Tensor.sliceChecked matrix 0 idx 1 #[1, cols]) #[cols]

private def mapRows {rows inDim outDim : Nat} {d : DType} {dev : Device}
    (matrix : Tensor ⟨#[rows, inDim], d, dev⟩)
    (f : Tensor ⟨#[inDim], d, dev⟩ → Tensor ⟨#[outDim], d, dev⟩) :
    Tensor ⟨#[rows, outDim], d, dev⟩ :=
  let mapped : Array (Tensor ⟨#[outDim], d, dev⟩) := Id.run do
    let mut acc : Array (Tensor ⟨#[outDim], d, dev⟩) := #[]
    for i in [:rows] do
      acc := acc.push (f (rowAt matrix i))
    pure acc
  let stacked := Tensor.stack1d mapped
  Tensor.reshape stacked #[rows, outDim]

/-- Materialize the dense input Jacobian for debugging/tests. -/
def materializeA {inDim paramDim outDim : Nat}
    (lin : LocalLinearization (device := device) inDim paramDim outDim) : AmbientMatrix (device := device) outDim inDim :=
  let basis : AmbientMatrix (device := device) inDim inDim := Tensor.eyeFor (n := inDim) .Float32 device
  Tensor.transpose2d (mapRows basis lin.applyA)

/-- Materialize the dense parameter Jacobian for debugging/tests. -/
def materializeB {inDim paramDim outDim : Nat}
    (lin : LocalLinearization (device := device) inDim paramDim outDim) : AmbientMatrix (device := device) outDim paramDim :=
  let basis : AmbientMatrix (device := device) paramDim paramDim := Tensor.eyeFor (n := paramDim) .Float32 device
  Tensor.transpose2d (mapRows basis lin.applyB)

/-- Pull a metric factor back through the input Jacobian using rowwise adjoints. -/
def pullbackInputFactor {rank inDim paramDim outDim : Nat}
    (lin : LocalLinearization (device := device) inDim paramDim outDim)
    (L : MetricFactor rank outDim .Float32 device) : MetricFactor rank inDim .Float32 device :=
  { matrix := mapRows L.matrix lin.applyAT }

/-- Pull a metric factor back through the parameter Jacobian using rowwise adjoints. -/
def pullbackParamFactor {rank inDim paramDim outDim : Nat}
    (lin : LocalLinearization (device := device) inDim paramDim outDim)
    (L : MetricFactor rank outDim .Float32 device) : MetricFactor rank paramDim .Float32 device :=
  { matrix := mapRows L.matrix lin.applyBT }

end LocalLinearization

private def linearApplyInput {inDim outDim : Nat} {d : DType} {dev : Device}
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩) (x : Tensor ⟨#[inDim], d, dev⟩) :
    Tensor ⟨#[outDim], d, dev⟩ :=
  columnToVector (Tensor.mm weight (vectorToColumn x))

private def linearApplyInputAdjoint {inDim outDim : Nat} {d : DType} {dev : Device}
    (weight : Tensor ⟨#[outDim, inDim], d, dev⟩) (y : Tensor ⟨#[outDim], d, dev⟩) :
    Tensor ⟨#[inDim], d, dev⟩ :=
  columnToVector (Tensor.mm (Tensor.transpose2d weight) (vectorToColumn y))

private def linearApplyParam {inDim outDim : Nat} {d : DType} {dev : Device}
    [DType.Promotes d d d]
    (x : Tensor ⟨#[inDim], d, dev⟩)
    (useBias : Bool)
    (update : Tensor ⟨#[outDim * inDim + outDim], d, dev⟩) : Tensor ⟨#[outDim], d, dev⟩ :=
  let (dW, dB) := splitLinearUpdate (inDim := inDim) (outDim := outDim) update
  let weightPart := linearApplyInput dW x
  if useBias then
    Tensor.add weightPart dB
  else
    weightPart

private def linearApplyParamAdjoint {inDim outDim : Nat} {d : DType} {dev : Device}
    (x : Tensor ⟨#[inDim], d, dev⟩)
    (useBias : Bool)
    (y : Tensor ⟨#[outDim], d, dev⟩) : Tensor ⟨#[outDim * inDim + outDim], d, dev⟩ :=
  let outer : Tensor ⟨#[outDim, inDim], d, dev⟩ :=
    Tensor.mm (vectorToColumn y) (Tensor.reshape x #[1, inDim] (by simp [TensorSpec.numelOfShape]))
  let weightPart : Tensor ⟨#[outDim * inDim], d, dev⟩ := Tensor.reshape outer #[outDim * inDim]
  let biasPart : Tensor ⟨#[outDim], d, dev⟩ := if useBias then y else Tensor.zerosLike y
  let joined := Tensor.cat weightPart biasPart 0
  Tensor.reshape joined #[outDim * inDim + outDim]

private def makeManifoldLeaf [MatrixManifoldCarrier M device] {m n : Nat}
    (w : M m n) : IO (M m n) := do
  let leaf ← Tensor.parameter (MatrixManifoldCarrier.toMatrix (device := device) w)
  pure (MatrixManifoldCarrier.reAmbient (device := device) w leaf)

/-- Single leaf backward step: metric solve, parameter update, and propagation. -/
def backwardLeafStep {M : Type} {inDim outDim : Nat} [RiemannianModule M inDim outDim device]
    {rank : Nat}
    (m : M)
    (tape : RiemannianModule.Tape (device := device) (M := M) (inDim := inDim) (outDim := outDim))
    (L : MetricFactor rank outDim .Float32 device)
    (lambda : AmbientVector (device := device) outDim)
    (lr : Float)
    : IO (M × MetricFactor rank inDim .Float32 device × AmbientVector (device := device) inDim × LayerStepDiagnostics) := do
  let lin := RiemannianModule.localLinearization (device := device) (M := M) (inDim := inDim) (outDim := outDim) m tape
  let g := lin.applyBT lambda
  let K := LocalLinearization.pullbackParamFactor lin L
  let mass := RiemannianModule.paramMass (device := device) (M := M) (inDim := inDim) (outDim := outDim) m
  let update := MetricFactor.solveWoodbury mass K g
  let m' ← RiemannianModule.applyAmbientUpdate (device := device) (M := M) (inDim := inDim) (outDim := outDim) m update lr
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

instance linearRiemannianModule (inDim outDim : Nat) :
    RiemannianModule (torch.Linear inDim outDim .Float32 device) inDim outDim device where
  paramDim := outDim * inDim + outDim
  Tape := AmbientVector (device := device) inDim
  forwardWithTape lin x :=
    let x2 : AmbientMatrix (device := device) 1 inDim :=
      Tensor.reshape x #[1, inDim] (by simp [TensorSpec.numelOfShape])
    let y2 : AmbientMatrix (device := device) 1 outDim :=
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
  paramMass _ := DiagonalMass.ones (device := device) (outDim * inDim + outDim)
  applyAmbientUpdate lin update lr := do
    let (dW, dB) := splitLinearUpdate (inDim := inDim) (outDim := outDim) update
    let weight' ← Tensor.parameter (Sub.sub lin.weight (Tensor.mulScalar dW lr))
    let bias' ←
      match lin.bias with
      | some b => do
        pure (some (← Tensor.parameter (Sub.sub b (Tensor.mulScalar dB lr))))
      | none => pure none
    pure { lin with weight := weight', bias := bias' }

instance manifoldLinearRiemannianModule
    (M : Nat → Nat → Type)
    [MatrixManifoldCarrier M device]
    {inDim outDim : Nat}
    :
    RiemannianModule (ManifoldLinear (device := device) M inDim outDim) inDim outDim device where
  paramDim := outDim * inDim + outDim
  Tape := AmbientVector (device := device) inDim
  forwardWithTape lin x :=
    let weight := MatrixManifoldCarrier.toMatrix (device := device) lin.weight
    let x2 : AmbientMatrix (device := device) 1 inDim :=
      Tensor.reshape x #[1, inDim] (by simp [TensorSpec.numelOfShape])
    let y2 : AmbientMatrix (device := device) 1 outDim :=
      match lin.bias with
      | some b => Add.add (Tensor.linear x2 weight) (Tensor.expand b #[1, outDim])
      | none => Tensor.linear x2 weight
    (Tensor.reshape y2 #[outDim] (by simp [TensorSpec.numelOfShape]), x)
  localLinearization lin x := {
    applyA := linearApplyInput (MatrixManifoldCarrier.toMatrix (device := device) lin.weight)
    applyAT := linearApplyInputAdjoint (MatrixManifoldCarrier.toMatrix (device := device) lin.weight)
    applyB := linearApplyParam x lin.bias.isSome
    applyBT := linearApplyParamAdjoint x lin.bias.isSome
  }
  paramMass _ := DiagonalMass.ones (device := device) (outDim * inDim + outDim)
  applyAmbientUpdate lin update lr := do
    let (dW, dB) := splitLinearUpdate (inDim := inDim) (outDim := outDim) update
    let updatedMatrix := Sub.sub (MatrixManifoldCarrier.toMatrix (device := device) lin.weight)
      (Tensor.mulScalar dW lr)
    let weight' := MatrixManifoldCarrier.projectLike (device := device) lin.weight updatedMatrix
    let weight' ← makeManifoldLeaf weight'
    let bias' ←
      match lin.bias with
      | some b => do
        pure (some (← Tensor.parameter (Sub.sub b (Tensor.mulScalar dB lr))))
      | none => pure none
    pure { lin with weight := weight', bias := bias' }

end RiemannianModule

namespace LocalLinearization

/-- Materialize the dense input Jacobian for debugging/tests. -/
def materializeA {inDim paramDim outDim : Nat}
    (lin : LocalLinearization (device := device) inDim paramDim outDim) : AmbientMatrix (device := device) outDim inDim :=
  RiemannianModule.LocalLinearization.materializeA lin

/-- Materialize the dense parameter Jacobian for debugging/tests. -/
def materializeB {inDim paramDim outDim : Nat}
    (lin : LocalLinearization (device := device) inDim paramDim outDim) : AmbientMatrix (device := device) outDim paramDim :=
  RiemannianModule.LocalLinearization.materializeB lin

/-- Pull a metric factor back through the input Jacobian using rowwise adjoints. -/
def pullbackInputFactor {rank inDim paramDim outDim : Nat}
    (lin : LocalLinearization (device := device) inDim paramDim outDim)
    (L : MetricFactor rank outDim .Float32 device) : MetricFactor rank inDim .Float32 device :=
  RiemannianModule.LocalLinearization.pullbackInputFactor lin L

/-- Pull a metric factor back through the parameter Jacobian using rowwise adjoints. -/
def pullbackParamFactor {rank inDim paramDim outDim : Nat}
    (lin : LocalLinearization (device := device) inDim paramDim outDim)
    (L : MetricFactor rank outDim .Float32 device) : MetricFactor rank paramDim .Float32 device :=
  RiemannianModule.LocalLinearization.pullbackParamFactor lin L

end LocalLinearization

/-- Forward tape for a two-layer sequential composition. -/
abbrev SequentialTape
    (M₁ : Type) (inDim midDim : Nat)
    (M₂ : Type) (outDim : Nat)
    [RiemannianModule M₁ inDim midDim device]
    [RiemannianModule M₂ midDim outDim device]
    :=
  RiemannianModule.Tape (device := device) (M := M₁) (inDim := inDim) (outDim := midDim) ×
    RiemannianModule.Tape (device := device) (M := M₂) (inDim := midDim) (outDim := outDim)

/-- Forward pass with tapes for a two-layer sequential composition. -/
def sequentialForwardWithTape
    {M₁ M₂ : Type}
    {inDim midDim outDim : Nat}
    [RiemannianModule M₁ inDim midDim device]
    [RiemannianModule M₂ midDim outDim device]
    (net : Sequential M₁ M₂)
    (x : AmbientVector (device := device) inDim)
    : AmbientVector (device := device) outDim × SequentialTape (device := device) M₁ inDim midDim M₂ outDim := by
  let (mid, tape₁) :=
    RiemannianModule.forwardWithTape (device := device) (M := M₁) (inDim := inDim) (outDim := midDim) net.first x
  let (y, tape₂) :=
    RiemannianModule.forwardWithTape (device := device) (M := M₂) (inDim := midDim) (outDim := outDim) net.second mid
  exact (y, (tape₁, tape₂))

/-- Backward recursion for a two-layer sequential composition. -/
def sequentialBackwardMetricStep
    {M₁ M₂ : Type}
    {rank inDim midDim outDim : Nat}
    [RiemannianModule M₁ inDim midDim device]
    [RiemannianModule M₂ midDim outDim device]
    (net : Sequential M₁ M₂)
    (tape : SequentialTape (device := device) M₁ inDim midDim M₂ outDim)
    (L : MetricFactor rank outDim .Float32 device)
    (lambda : AmbientVector (device := device) outDim)
    (lr : Float)
    : IO (Sequential M₁ M₂ × MetricFactor rank inDim .Float32 device × AmbientVector (device := device) inDim × Array LayerStepDiagnostics) := do
  let (tape₁, tape₂) := tape
  let (second', midFactor, midCotangent, stats₂) ←
    RiemannianModule.backwardLeafStep (device := device) net.second tape₂ L lambda lr
  let (first', inFactor, inCotangent, stats₁) ←
    RiemannianModule.backwardLeafStep (device := device) net.first tape₁ midFactor midCotangent lr
  pure ({ first := first', second := second' }, inFactor, inCotangent, #[stats₁, stats₂])

end Tyr.Modular
