import Tyr.Optim
import Tyr.Modular.MetricFactor

namespace torch.Optim.RiemannianTreeSGD

open torch

variable {device : Device}
open Tyr.Modular

/-- Whole-model diagnostics for one tree-structured Riemannian step. -/
structure StepDiagnostics where
  gradientNorm : Float := 0.0
  factorRank : UInt64 := 0
  innerConditionEstimate : Float := 0.0
  residualNorm : Float := 0.0
  updateNorm : Float := 0.0
  deriving Repr, Inhabited

/-- Result of one tree-structured Riemannian update. -/
structure StepResult (Params : Type) (s : Shape) where
  params : Params
  prediction : Tensor { shape := s, dtype := .Float32, device := device }
  diagnostics : StepDiagnostics := {}
  loss : Float := 0.0

private def shapeNumel (s : Shape) : UInt64 :=
  s.foldl (fun acc d => acc * d) 1

private def treeDot [TensorStruct α] [FloatingTensorStruct α] (x y : α) : Float :=
  let prod := FloatingTensorStruct.zipWith Tensor.mul x y
  TensorStruct.fold (fun t acc => acc + Tensor.item (Tensor.sumAll t)) 0.0 prod

private def gradStruct [TensorStruct α] (loss : Tensor τ) (x : α) : α :=
  TensorStruct.map (fun t => Tensor.grad loss t (Tensor.onesLike loss)) x

private def clipScale [TensorStruct α] [FloatingTensorStruct α] (x : α) (maxNorm : Float) : α :=
  let sqNorm := treeDot x x
  let norm := Float.sqrt sqNorm
  if norm <= maxNorm || norm == 0.0 then
    x
  else
    TensorStruct.scale x (maxNorm / norm)

private def applyMass [TensorStruct α] [FloatingTensorStruct α] (mass : α) (x : α) : α :=
  FloatingTensorStruct.zipWith Tensor.mul mass x

private def applyInvMass [TensorStruct α] [FloatingTensorStruct α] (mass : α) (x : α) : α :=
  FloatingTensorStruct.zipWith (fun d v => by simpa using Tensor.div v d) mass x

private def weightedInvDot [TensorStruct α] [FloatingTensorStruct α] (mass : α) (x y : α) : Float :=
  treeDot (applyInvMass mass x) y

/-- Tree-valued low-rank factor on parameter space. -/
structure TreeMetricFactor (Params : Type) (rank : UInt64) where
  rows : Array Params
  rows_size : rows.size = rank.toNat

namespace TreeMetricFactor

private def rowAt {Params : Type} {rank : UInt64}
    (K : TreeMetricFactor Params rank) (i : Fin rank.toNat) : Params :=
  K.rows[i.val]'(by simp [K.rows_size])

def apply {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params] {rank : UInt64}
    (K : TreeMetricFactor Params rank)
    (x : Params) : Tensor { shape := #[rank], dtype := .Float32, device := .CPU } :=
  let coeffs :=
    ((List.finRange rank.toNat).toArray.map fun i =>
      let row := K.rowAt i
      treeDot row x)
  Tensor.reshapeChecked (Tensor.fromFloatArray coeffs) #[rank]

def applyTranspose {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params] {rank : UInt64}
    (K : TreeMetricFactor Params rank)
    (prototype : Params)
    (coeffs : Tensor { shape := #[rank], dtype := .Float32, device := .CPU }) : Params :=
  Id.run do
    let mut acc := TensorStruct.map Tensor.zerosLike prototype
    for i in List.finRange rank.toNat do
      let coeff1 := Tensor.sliceChecked coeffs 0 (UInt64.ofNat i.val) 1 #[1]
      let coeff := Tensor.item (Tensor.sumAll coeff1)
      let row := K.rowAt i
      acc := TensorStruct.add acc (TensorStruct.scale row coeff)
    pure acc

def woodburyInner {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params] {rank : UInt64}
    (mass : Params)
    (K : TreeMetricFactor Params rank) : Tensor { shape := #[rank, rank], dtype := .Float32, device := .CPU } :=
  let entries : Array Float := Id.run do
    let mut acc : Array Float := #[]
    for i in List.finRange rank.toNat do
      for j in List.finRange rank.toNat do
        let rowI := K.rowAt i
        let rowJ := K.rowAt j
        let entry := weightedInvDot mass rowJ rowI
        acc := acc.push (if i.val == j.val then entry + 1.0 else entry)
    pure acc
  Tensor.reshapeChecked (Tensor.fromFloatArray entries) #[rank, rank]

def applyRegularized {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params] {rank : UInt64}
    (mass : Params)
    (K : TreeMetricFactor Params rank)
    (x : Params) : Params :=
  TensorStruct.add (applyMass mass x) (K.applyTranspose x (K.apply x))

def solveWoodbury {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params] {rank : UInt64}
    (mass : Params)
    (K : TreeMetricFactor Params rank)
    (g : Params) : Params :=
  let inner := K.woodburyInner mass
  let innerInv := Tensor.inv inner
  let base := applyInvMass mass g
  let correctionArg := K.apply base
  let correctionInner : Tensor { shape := #[rank], dtype := .Float32, device := .CPU } :=
    Tensor.reshapeChecked (Tensor.mm innerInv (Tensor.reshapeChecked correctionArg #[rank, 1])) #[rank]
  let correction := applyInvMass mass (K.applyTranspose g correctionInner)
  TensorStruct.sub base correction

end TreeMetricFactor

private def paramVjpFromPrepared [TensorStruct Params] [FloatingTensorStruct Params] {s : Shape}
    (output : Tensor { shape := s, dtype := .Float32, device := device })
    (params : Params)
    (adjOutput : Tensor { shape := s, dtype := .Float32, device := device }) : Params :=
  let loss := Tensor.sumAll (Tensor.mul output (Tensor.detach adjOutput))
  TensorStruct.detach (gradStruct loss params)

private def buildTreeFactorFromOutputFactor [TensorStruct Params] [FloatingTensorStruct Params] {s : Shape} {rank : UInt64}
    (output : Tensor { shape := s, dtype := .Float32, device := device })
    (params : Params)
    (outputFactor : MetricFactor rank (shapeNumel s) .Float32 device) : TreeMetricFactor Params rank := by
  let rows :=
    ((List.range rank.toNat).toArray.map fun i =>
      let rowFlat : Tensor { shape := #[shapeNumel s], dtype := .Float32, device := device } :=
        Tensor.reshapeChecked (Tensor.sliceChecked outputFactor.matrix 0 (UInt64.ofNat i) 1 #[1, shapeNumel s])
          #[shapeNumel s]
      let rowCotangent : Tensor { shape := s, dtype := .Float32, device := device } := Tensor.reshapeChecked rowFlat s
      paramVjpFromPrepared output params rowCotangent)
  exact {
    rows := rows
    rows_size := by
      simp [rows]
  }

private def buildTreeFactorFromCotangentRows [TensorStruct Params] [FloatingTensorStruct Params] {s : Shape} {rank : UInt64}
    (output : Tensor { shape := s, dtype := .Float32, device := device })
    (params : Params)
    (rows : Array (Tensor { shape := s, dtype := .Float32, device := device }))
    (rows_size : rows.size = rank.toNat) : TreeMetricFactor Params rank := by
  let pulled :=
    rows.map (fun row => paramVjpFromPrepared output params row)
  exact {
    rows := pulled
    rows_size := by
      simp [pulled, rows_size]
  }

private def finalizePreparedStepWithTreeFactor
    {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    {s : Shape} {rank : UInt64}
    (params : Params)
    (predictionReq : Tensor { shape := s, dtype := .Float32, device := device })
    (mass : Params)
    (factor : TreeMetricFactor Params rank)
    (gradient : Params)
    (lr : Float)
    : IO (StepResult (device := device) Params s) := do
  let update := TreeMetricFactor.solveWoodbury mass factor gradient
  let inner := TreeMetricFactor.woodburyInner mass factor
  let innerInv := Tensor.inv inner
  let residual := TreeMetricFactor.applyRegularized mass factor update |> fun x => TensorStruct.sub x gradient
  let params' ← torch.Optim.apply_updates params (TensorStruct.scale update (-lr))
  pure {
    params := params'
    prediction := Tensor.detach predictionReq
    diagnostics := {
      gradientNorm := Float.sqrt (treeDot gradient gradient)
      factorRank := rank
      innerConditionEstimate := Tensor.spectralNorm inner * Tensor.spectralNorm innerInv
      residualNorm := Float.sqrt (treeDot residual residual)
      updateNorm := Float.sqrt (treeDot update update)
      }
  }

private def stepPreparedWithFactorAndMass
    {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    {s : Shape} {rank : UInt64}
    (params : Params)
    (paramsReq : Params)
    (predictionReq : Tensor { shape := s, dtype := .Float32, device := device })
    (mass : Params)
    (outputFactor : MetricFactor rank (shapeNumel s) .Float32 device)
    (outputCotangent : Tensor { shape := s, dtype := .Float32, device := device })
    (lr : Float)
    : IO (StepResult (device := device) Params s) := do
  let gradient := paramVjpFromPrepared predictionReq paramsReq outputCotangent
  let factor := buildTreeFactorFromOutputFactor predictionReq paramsReq outputFactor
  finalizePreparedStepWithTreeFactor params predictionReq mass factor gradient lr

/-- Whole-model tree step with explicit output cotangent and output-space factor. -/
def stepWithFactorAndMass
    {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    {s : Shape} {rank : UInt64}
    (params : Params)
    (forward : Params → IO (Tensor { shape := s, dtype := .Float32, device := device }))
    (mass : Params)
    (outputFactor : MetricFactor rank (shapeNumel s) .Float32 device)
    (outputCotangent : Tensor { shape := s, dtype := .Float32, device := device })
    (lr : Float)
    : IO (StepResult (device := device) Params s) := do
  let paramsReq ← TensorStruct.makeLeafParams params
  let predictionReq ← forward paramsReq
  stepPreparedWithFactorAndMass
    params paramsReq predictionReq mass outputFactor outputCotangent lr

/-- Whole-model tree step with unit diagonal mass. -/
def stepWithFactor
    {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    {s : Shape} {rank : UInt64}
    (params : Params)
    (forward : Params → IO (Tensor { shape := s, dtype := .Float32, device := device }))
    (outputFactor : MetricFactor rank (shapeNumel s) .Float32 device)
    (outputCotangent : Tensor { shape := s, dtype := .Float32, device := device })
    (lr : Float)
    : IO (StepResult (device := device) Params s) := do
  let mass := TensorStruct.map Tensor.onesLike params
  stepWithFactorAndMass params forward mass outputFactor outputCotangent lr

private def prepareCrossEntropyStep
    {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    {batch seq vocab : UInt64}
    (params : Params)
    (forward : Params → IO (Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := device }))
    (targets : Tensor { shape := #[batch, seq], dtype := .Int64, device := device })
    : IO (Params × Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := device } × Tensor { shape := #[], dtype := .Float32, device := device } ×
        Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := device }) := do
  let paramsReq ← TensorStruct.makeLeafParams params
  let predictionReq ← forward paramsReq
  let logitsFlat : Tensor { shape := #[batch * seq, vocab], dtype := .Float32, device := device } :=
    Tensor.reshapeChecked predictionReq #[batch * seq, vocab]
  let targetsFlat : Tensor { shape := #[batch * seq], dtype := .Int64, device := device } :=
    Tensor.reshapeChecked targets #[batch * seq]
  let lossT ← IO.ofExcept (Tensor.crossEntropy? logitsFlat targetsFlat)
  let outputCotangent := Tensor.detach (Tensor.grad lossT predictionReq (Tensor.onesLike lossT))
  pure (paramsReq, predictionReq, lossT, outputCotangent)

private def sampledFisherCotangentRows
    {batch seq vocab : UInt64}
    (predictionReq : Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := device })
    (probeCount : UInt64)
    : IO { rows : Array (Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := device }) // rows.size = probeCount.toNat } := do
  if probeCount == 0 then
    throw <| IO.userError "RiemannianTreeSGD.stepCrossEntropySampledFisher requires probeCount > 0"
  let tokenCount := batch * seq
  let probs3d := Tensor.detach (Tensor.softmax predictionReq (-1) rfl)
  let probs : Tensor { shape := #[tokenCount, vocab], dtype := .Float32, device := device } := Tensor.reshapeChecked probs3d #[tokenCount, vocab]
  let scale := 1.0 / Float.sqrt probeCount.toFloat
  let srcOnes0 : Tensor { shape := #[tokenCount, 1], dtype := .Float32, device := device } := Tensor.ones #[tokenCount, 1] false device
  let srcOnes := srcOnes0
  let rec sampleRows (n : Nat) : IO { rows : List (Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := device }) // rows.length = n } := do
    match n with
    | 0 => pure ⟨[], by simp⟩
    | n + 1 =>
      let sampledIdx ← Tensor.multinomial probs 1 false
      let base : Tensor { shape := #[tokenCount, vocab], dtype := .Float32, device := device } := Tensor.zerosLike probs
      let sampledOneHot := Tensor.scatter2d base 1 sampledIdx srcOnes
      let rowFlat := Tensor.mulScalar (Tensor.sub sampledOneHot probs) scale
      let row : Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := device } := Tensor.reshapeChecked rowFlat #[batch, seq, vocab]
      let ⟨rest, hRest⟩ ← sampleRows n
      pure ⟨row :: rest, by simp [hRest]⟩
  let ⟨rowsList, hRowsList⟩ ← sampleRows probeCount.toNat
  let rows := rowsList.toArray
  have hRows : rows.size = probeCount.toNat := by
    simp [rows, hRowsList]
  pure ⟨rows, hRows⟩

private def sampledFisherTreeFactor
    {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    {batch seq vocab : UInt64} {rank : UInt64}
    (predictionReq : Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := device })
    (paramsReq : Params)
    (rows : Array (Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := device }))
    (rows_size : rows.size = rank.toNat)
    : TreeMetricFactor Params rank :=
  buildTreeFactorFromCotangentRows predictionReq paramsReq rows rows_size

/-- Cross-entropy step for tensor-logit models with unit diagonal mass. -/
def stepCrossEntropy
    {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    {batch seq vocab : UInt64}
    (params : Params)
    (forward : Params → IO (Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := device }))
    (targets : Tensor { shape := #[batch, seq], dtype := .Int64, device := device })
    (lr : Float)
    (gradClip : Float := 0.0)
    : IO (StepResult (device := device) Params #[batch, seq, vocab]) := do
  let (paramsReq, predictionReq, lossT, outputCotangent) ←
    prepareCrossEntropyStep params forward targets
  let outDim := shapeNumel #[batch, seq, vocab]
  let outputFactor : MetricFactor outDim outDim .Float32 device :=
    { matrix := Tensor.eyeFor .Float32 device }
  let mass := TensorStruct.map Tensor.onesLike params
  if gradClip > 0.0 then
    let paramGrad := clipScale (paramVjpFromPrepared predictionReq paramsReq outputCotangent) gradClip
    let factor := buildTreeFactorFromOutputFactor predictionReq paramsReq outputFactor
    let update := TreeMetricFactor.solveWoodbury mass factor paramGrad
    let inner := TreeMetricFactor.woodburyInner mass factor
    let innerInv := Tensor.inv inner
    let residual := TreeMetricFactor.applyRegularized mass factor update |> fun x => TensorStruct.sub x paramGrad
    let params' ← torch.Optim.apply_updates params (TensorStruct.scale update (-lr))
    pure {
      params := params'
      prediction := Tensor.detach predictionReq
      diagnostics := {
        gradientNorm := Float.sqrt (treeDot paramGrad paramGrad)
        factorRank := outDim
        innerConditionEstimate := Tensor.spectralNorm inner * Tensor.spectralNorm innerInv
        residualNorm := Float.sqrt (treeDot residual residual)
        updateNorm := Float.sqrt (treeDot update update)
      }
      loss := Tensor.item lossT
    }
  else
    let result ← stepPreparedWithFactorAndMass
      (rank := outDim)
      params paramsReq predictionReq mass outputFactor outputCotangent lr
    pure { result with loss := Tensor.item lossT }

/-- Cross-entropy step using a sampled logits-Fisher pullback factor.
    The loss gradient remains exact; only the output metric factor is sketched. -/
def stepCrossEntropySampledFisher
    {Params : Type} [TensorStruct Params] [FloatingTensorStruct Params]
    {batch seq vocab : UInt64}
    (params : Params)
    (forward : Params → IO (Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := device }))
    (targets : Tensor { shape := #[batch, seq], dtype := .Int64, device := device })
    (probeCount : UInt64)
    (lr : Float)
    (gradClip : Float := 0.0)
    : IO (StepResult (device := device) Params #[batch, seq, vocab]) := do
  let (paramsReq, predictionReq, lossT, outputCotangent) ←
    prepareCrossEntropyStep params forward targets
  let mass := TensorStruct.map Tensor.onesLike params
  let gradient0 := paramVjpFromPrepared predictionReq paramsReq outputCotangent
  let gradient := if gradClip > 0.0 then clipScale gradient0 gradClip else gradient0
  let ⟨rows, hRows⟩ ← sampledFisherCotangentRows predictionReq probeCount
  let factor :=
    sampledFisherTreeFactor
      (rank := probeCount)
      predictionReq paramsReq rows hRows
  let result ←
    finalizePreparedStepWithTreeFactor
      (rank := probeCount)
      params predictionReq mass factor gradient lr
  pure { result with loss := Tensor.item lossT }

end torch.Optim.RiemannianTreeSGD
