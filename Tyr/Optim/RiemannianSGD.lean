import Tyr.Modular.RiemannianModule

namespace torch.Optim.RiemannianSGD

open torch

variable {device : Device}
open Tyr.Modular

/-- Result of one Riemannian optimizer step. -/
structure StepResult (M : Type) (rank inDim outDim : UInt64) where
  params : M
  prediction : AmbientVector (device := device) outDim
  inputFactor : MetricFactor rank inDim .Float32 device
  inputCotangent : AmbientVector (device := device) inDim
  diagnostics : Array LayerStepDiagnostics := #[]
  loss : Float := 0.0

/-- One step given an explicit output metric factor and output cotangent. -/
def stepWithFactor {M : Type} {rank inDim outDim : UInt64}
    [RiemannianModule M inDim outDim device]
    (params : M)
    (x : AmbientVector (device := device) inDim)
    (outputFactor : MetricFactor rank outDim .Float32 device)
    (outputCotangent : AmbientVector (device := device) outDim)
    (lr : Float)
    : IO (StepResult (device := device) M rank inDim outDim) := do
  let (prediction, tape) :=
    RiemannianModule.forwardWithTape (device := device) (M := M) (inDim := inDim) (outDim := outDim) params x
  let (params', inputFactor, inputCotangent, diagnostics) ←
    RiemannianModule.backwardLeafStep (device := device) params tape outputFactor outputCotangent lr
  pure {
    params := params'
    prediction := prediction
    inputFactor := inputFactor
    inputCotangent := inputCotangent
    diagnostics := #[diagnostics]
  }

/-- One step for half-squared-error loss with identity output metric. -/
def stepMSE {M : Type} {inDim outDim : UInt64}
    [RiemannianModule M inDim outDim device]
    (params : M)
    (x : AmbientVector (device := device) inDim)
    (target : AmbientVector (device := device) outDim)
    (lr : Float)
    : IO (StepResult (device := device) M outDim inDim outDim) := do
  let (prediction, tape) :=
    RiemannianModule.forwardWithTape (device := device) (M := M) (inDim := inDim) (outDim := outDim) params x
  let residual := Sub.sub prediction target
  let loss := 0.5 * Tensor.item (Tensor.sumAll (Mul.mul residual residual))
  let outputFactor := MetricFactor.identity (device := device) outDim
  let (params', inputFactor, inputCotangent, diagnostics) ←
    RiemannianModule.backwardLeafStep (device := device) params tape outputFactor residual lr
  pure {
    params := params'
    prediction := prediction
    inputFactor := inputFactor
    inputCotangent := inputCotangent
    diagnostics := #[diagnostics]
    loss := loss
  }

/-- One sequential two-layer step with an explicit output metric factor. -/
def stepSequentialWithFactor
    {M₁ M₂ : Type}
    {rank inDim midDim outDim : UInt64}
    [RiemannianModule M₁ inDim midDim device]
    [RiemannianModule M₂ midDim outDim device]
    (params : Sequential M₁ M₂)
    (x : AmbientVector (device := device) inDim)
    (outputFactor : MetricFactor rank outDim .Float32 device)
    (outputCotangent : AmbientVector (device := device) outDim)
    (lr : Float)
    : IO (StepResult (device := device) (Sequential M₁ M₂) rank inDim outDim) := do
  let (prediction, tape) :
      AmbientVector (device := device) outDim × SequentialTape (device := device) M₁ inDim midDim M₂ outDim :=
    sequentialForwardWithTape
      (M₁ := M₁) (M₂ := M₂) (inDim := inDim) (midDim := midDim) (outDim := outDim)
      params x
  let (params', inputFactor, inputCotangent, diagnostics) ←
    sequentialBackwardMetricStep
      (M₁ := M₁) (M₂ := M₂) (rank := rank) (inDim := inDim) (midDim := midDim) (outDim := outDim)
      params tape outputFactor outputCotangent lr
  pure {
    params := params'
    prediction := prediction
    inputFactor := inputFactor
    inputCotangent := inputCotangent
    diagnostics := diagnostics
  }

/-- One sequential two-layer step for half-squared-error loss. -/
def stepSequentialMSE
    {M₁ M₂ : Type}
    {inDim midDim outDim : UInt64}
    [RiemannianModule M₁ inDim midDim device]
    [RiemannianModule M₂ midDim outDim device]
    (params : Sequential M₁ M₂)
    (x : AmbientVector (device := device) inDim)
    (target : AmbientVector (device := device) outDim)
    (lr : Float)
    : IO (StepResult (device := device) (Sequential M₁ M₂) outDim inDim outDim) := do
  let (prediction, tape) :
      AmbientVector (device := device) outDim × SequentialTape (device := device) M₁ inDim midDim M₂ outDim :=
    sequentialForwardWithTape
      (M₁ := M₁) (M₂ := M₂) (inDim := inDim) (midDim := midDim) (outDim := outDim)
      params x
  let residual := Sub.sub prediction target
  let loss := 0.5 * Tensor.item (Tensor.sumAll (Mul.mul residual residual))
  let outputFactor := MetricFactor.identity (device := device) outDim
  let (params', inputFactor, inputCotangent, diagnostics) ←
    sequentialBackwardMetricStep
      (M₁ := M₁) (M₂ := M₂) (rank := outDim) (inDim := inDim) (midDim := midDim) (outDim := outDim)
      params tape outputFactor residual lr
  pure {
    params := params'
    prediction := prediction
    inputFactor := inputFactor
    inputCotangent := inputCotangent
    diagnostics := diagnostics
    loss := loss
  }

end torch.Optim.RiemannianSGD
