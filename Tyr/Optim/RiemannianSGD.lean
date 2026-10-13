import Tyr.Modular.RiemannianModule

namespace torch.Optim.RiemannianSGD

open torch
open Tyr.Modular

/-- Result of one Riemannian optimizer step. -/
structure StepResult (M : Type) (rank inDim outDim : UInt64) where
  params : M
  prediction : AmbientVector outDim
  inputFactor : MetricFactor rank inDim
  inputCotangent : AmbientVector inDim
  diagnostics : Array LayerStepDiagnostics := #[]
  loss : Float := 0.0

/-- One step given an explicit output metric factor and output cotangent. -/
def stepWithFactor {M : Type} {rank inDim outDim : UInt64}
    [RiemannianModule M inDim outDim]
    (params : M)
    (x : AmbientVector inDim)
    (outputFactor : MetricFactor rank outDim)
    (outputCotangent : AmbientVector outDim)
    (lr : Float)
    : StepResult M rank inDim outDim :=
  let (prediction, tape) :=
    RiemannianModule.forwardWithTape (M := M) (inDim := inDim) (outDim := outDim) params x
  let (params', inputFactor, inputCotangent, diagnostics) :=
    RiemannianModule.backwardLeafStep params tape outputFactor outputCotangent lr
  {
    params := params'
    prediction := prediction
    inputFactor := inputFactor
    inputCotangent := inputCotangent
    diagnostics := #[diagnostics]
  }

/-- One step for half-squared-error loss with identity output metric. -/
def stepMSE {M : Type} {inDim outDim : UInt64}
    [RiemannianModule M inDim outDim]
    (params : M)
    (x : AmbientVector inDim)
    (target : AmbientVector outDim)
    (lr : Float)
    : StepResult M outDim inDim outDim :=
  let (prediction, tape) :=
    RiemannianModule.forwardWithTape (M := M) (inDim := inDim) (outDim := outDim) params x
  let residual := Sub.sub prediction target
  let loss := 0.5 * Tensor.item (Tensor.sumAll (Mul.mul residual residual))
  let outputFactor := MetricFactor.identity outDim
  let (params', inputFactor, inputCotangent, diagnostics) :=
    RiemannianModule.backwardLeafStep params tape outputFactor residual lr
  {
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
    [RiemannianModule M₁ inDim midDim]
    [RiemannianModule M₂ midDim outDim]
    (params : Sequential M₁ M₂)
    (x : AmbientVector inDim)
    (outputFactor : MetricFactor rank outDim)
    (outputCotangent : AmbientVector outDim)
    (lr : Float)
    : StepResult (Sequential M₁ M₂) rank inDim outDim :=
  let (prediction, tape) :
      AmbientVector outDim × SequentialTape M₁ inDim midDim M₂ outDim :=
    sequentialForwardWithTape
      (M₁ := M₁) (M₂ := M₂) (inDim := inDim) (midDim := midDim) (outDim := outDim)
      params x
  let (params', inputFactor, inputCotangent, diagnostics) :=
    sequentialBackwardMetricStep
      (M₁ := M₁) (M₂ := M₂) (rank := rank) (inDim := inDim) (midDim := midDim) (outDim := outDim)
      params tape outputFactor outputCotangent lr
  {
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
    [RiemannianModule M₁ inDim midDim]
    [RiemannianModule M₂ midDim outDim]
    (params : Sequential M₁ M₂)
    (x : AmbientVector inDim)
    (target : AmbientVector outDim)
    (lr : Float)
    : StepResult (Sequential M₁ M₂) outDim inDim outDim :=
  let (prediction, tape) :
      AmbientVector outDim × SequentialTape M₁ inDim midDim M₂ outDim :=
    sequentialForwardWithTape
      (M₁ := M₁) (M₂ := M₂) (inDim := inDim) (midDim := midDim) (outDim := outDim)
      params x
  let residual := Sub.sub prediction target
  let loss := 0.5 * Tensor.item (Tensor.sumAll (Mul.mul residual residual))
  let outputFactor := MetricFactor.identity outDim
  let (params', inputFactor, inputCotangent, diagnostics) :=
    sequentialBackwardMetricStep
      (M₁ := M₁) (M₂ := M₂) (rank := outDim) (inDim := inDim) (midDim := midDim) (outDim := outDim)
      params tape outputFactor residual lr
  {
    params := params'
    prediction := prediction
    inputFactor := inputFactor
    inputCotangent := inputCotangent
    diagnostics := diagnostics
    loss := loss
  }

end torch.Optim.RiemannianSGD
