import Tyr.Optim
import Tyr.Modular.MetricFactor

namespace torch.Optim.RiemannianTreeSGD

open torch
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
  prediction : T s
  diagnostics : StepDiagnostics := {}
  loss : Float := 0.0

private def shapeNumel (s : Shape) : UInt64 :=
  s.foldl (fun acc d => acc * d) 1

private def sumAllCompat {s : Shape} (t : T s) : T #[] :=
  let mean := nn.meanAll t
  let scale := (T.shape t).foldl (fun acc d => acc * d.toFloat) 1.0
  mul_scalar mean scale

private def treeDot [TensorStruct α] (x y : α) : T #[] :=
  let prod := TensorStruct.zipWith (fun a b => a * b) x y
  TensorStruct.fold (fun t acc => acc + sumAllCompat t) (zeros #[]) prod

private def gradStruct [TensorStruct α] (loss : T #[]) (x : α) : α :=
  let one := ones #[]
  TensorStruct.map (fun t => autograd.grad loss t one) x

private def clipScale [TensorStruct α] (x : α) (maxNorm : Float) : α :=
  let sqNorm := nn.item (treeDot x x)
  let norm := Float.sqrt sqNorm
  if norm <= maxNorm || norm == 0.0 then
    x
  else
    TensorStruct.scale x (maxNorm / norm)

private def applyMass [TensorStruct α] (mass : α) (x : α) : α :=
  TensorStruct.zipWith (fun d v => d * v) mass x

private def applyInvMass [TensorStruct α] (mass : α) (x : α) : α :=
  TensorStruct.zipWith (fun d v => nn.div v d) mass x

private def weightedInvDot [TensorStruct α] (mass : α) (x y : α) : T #[] :=
  treeDot (applyInvMass mass x) y

/-- Tree-valued low-rank factor on parameter space. -/
structure TreeMetricFactor (Params : Type) (rank : UInt64) where
  rows : Array Params
  rows_size : rows.size = rank.toNat

namespace TreeMetricFactor

private def rowAt {Params : Type} {rank : UInt64}
    (K : TreeMetricFactor Params rank) (i : Fin rank.toNat) : Params :=
  K.rows[i.val]'(by simp [K.rows_size])

def apply {Params : Type} [TensorStruct Params] {rank : UInt64}
    (K : TreeMetricFactor Params rank)
    (x : Params) : T #[rank] :=
  let coeffs :=
    ((List.finRange rank.toNat).toArray.map fun i =>
      let row := K.rowAt i
      nn.item (treeDot row x))
  reshape (data.fromFloatArray coeffs) #[rank]

def applyTranspose {Params : Type} [TensorStruct Params] {rank : UInt64}
    (K : TreeMetricFactor Params rank)
    (prototype : Params)
    (coeffs : T #[rank]) : Params :=
  Id.run do
    let mut acc := TensorStruct.map torch.zeros_like prototype
    for i in List.finRange rank.toNat do
      let coeff1 : T #[1] := data.slice1d coeffs (Int64.ofNat i.val) (Int64.ofNat (i.val + 1))
      let coeff := nn.item (reshape coeff1 #[])
      let row := K.rowAt i
      acc := TensorStruct.add acc (TensorStruct.scale row coeff)
    pure acc

def woodburyInner {Params : Type} [TensorStruct Params] {rank : UInt64}
    (mass : Params)
    (K : TreeMetricFactor Params rank) : T #[rank, rank] :=
  let entries : Array Float := Id.run do
    let mut acc : Array Float := #[]
    for i in List.finRange rank.toNat do
      for j in List.finRange rank.toNat do
        let rowI := K.rowAt i
        let rowJ := K.rowAt j
        let entry := nn.item (weightedInvDot mass rowJ rowI)
        acc := acc.push (if i.val == j.val then entry + 1.0 else entry)
    pure acc
  reshape (data.fromFloatArray entries) #[rank, rank]

def applyRegularized {Params : Type} [TensorStruct Params] {rank : UInt64}
    (mass : Params)
    (K : TreeMetricFactor Params rank)
    (x : Params) : Params :=
  TensorStruct.add (applyMass mass x) (K.applyTranspose x (K.apply x))

def solveWoodbury {Params : Type} [TensorStruct Params] {rank : UInt64}
    (mass : Params)
    (K : TreeMetricFactor Params rank)
    (g : Params) : Params :=
  let inner := K.woodburyInner mass
  let innerInv := linalg.inv inner
  let base := applyInvMass mass g
  let correctionArg := K.apply base
  let correctionInner : T #[rank] :=
    reshape (nn.mm innerInv (reshape correctionArg #[rank, 1])) #[rank]
  let correction := applyInvMass mass (K.applyTranspose g correctionInner)
  TensorStruct.sub base correction

end TreeMetricFactor

private def paramVjpFromPrepared [TensorStruct Params] {s : Shape}
    (output : T s)
    (params : Params)
    (adjOutput : T s) : Params :=
  let loss := sumAllCompat (output * autograd.detach adjOutput)
  TensorStruct.detach (gradStruct loss params)

private def buildTreeFactorFromOutputFactor [TensorStruct Params] {s : Shape} {rank : UInt64}
    (output : T s)
    (params : Params)
    (outputFactor : MetricFactor rank (shapeNumel s)) : TreeMetricFactor Params rank := by
  let rows :=
    ((List.range rank.toNat).toArray.map fun i =>
      let rowFlat : T #[shapeNumel s] :=
        reshape (data.slice2d outputFactor.matrix (UInt64.ofNat i) 1) #[shapeNumel s]
      let rowCotangent : T s := reshape rowFlat (T.shape output)
      paramVjpFromPrepared output params rowCotangent)
  exact {
    rows := rows
    rows_size := by
      simp [rows]
  }

private def buildTreeFactorFromCotangentRows [TensorStruct Params] {s : Shape} {rank : UInt64}
    (output : T s)
    (params : Params)
    (rows : Array (T s))
    (rows_size : rows.size = rank.toNat) : TreeMetricFactor Params rank := by
  let pulled :=
    rows.map (fun row => paramVjpFromPrepared output params row)
  exact {
    rows := pulled
    rows_size := by
      simp [pulled, rows_size]
  }

private def finalizePreparedStepWithTreeFactor
    {Params : Type} [TensorStruct Params]
    {s : Shape} {rank : UInt64}
    (params : Params)
    (predictionReq : T s)
    (mass : Params)
    (factor : TreeMetricFactor Params rank)
    (gradient : Params)
    (lr : Float)
    : StepResult Params s :=
  let update := TreeMetricFactor.solveWoodbury mass factor gradient
  let inner := TreeMetricFactor.woodburyInner mass factor
  let innerInv := linalg.inv inner
  let residual := TreeMetricFactor.applyRegularized mass factor update |> fun x => TensorStruct.sub x gradient
  let params' := torch.Optim.apply_updates params (TensorStruct.scale update (-lr))
  {
    params := params'
    prediction := autograd.detach predictionReq
    diagnostics := {
      gradientNorm := Float.sqrt (nn.item (treeDot gradient gradient))
      factorRank := rank
      innerConditionEstimate := linalg.spectralNorm inner * linalg.spectralNorm innerInv
      residualNorm := Float.sqrt (nn.item (treeDot residual residual))
      updateNorm := Float.sqrt (nn.item (treeDot update update))
      }
  }

private def stepPreparedWithFactorAndMass
    {Params : Type} [TensorStruct Params]
    {s : Shape} {rank : UInt64}
    (params : Params)
    (paramsReq : Params)
    (predictionReq : T s)
    (mass : Params)
    (outputFactor : MetricFactor rank (shapeNumel s))
    (outputCotangent : T s)
    (lr : Float)
    : StepResult Params s :=
  let gradient := paramVjpFromPrepared predictionReq paramsReq outputCotangent
  let factor := buildTreeFactorFromOutputFactor predictionReq paramsReq outputFactor
  finalizePreparedStepWithTreeFactor params predictionReq mass factor gradient lr

/-- Whole-model tree step with explicit output cotangent and output-space factor. -/
def stepWithFactorAndMass
    {Params : Type} [TensorStruct Params]
    {s : Shape} {rank : UInt64}
    (params : Params)
    (forward : Params → IO (T s))
    (mass : Params)
    (outputFactor : MetricFactor rank (shapeNumel s))
    (outputCotangent : T s)
    (lr : Float)
    : IO (StepResult Params s) := do
  let paramsReq := TensorStruct.makeLeafParams params
  let predictionReq ← forward paramsReq
  pure <| stepPreparedWithFactorAndMass
    params paramsReq predictionReq mass outputFactor outputCotangent lr

/-- Whole-model tree step with unit diagonal mass. -/
def stepWithFactor
    {Params : Type} [TensorStruct Params]
    {s : Shape} {rank : UInt64}
    (params : Params)
    (forward : Params → IO (T s))
    (outputFactor : MetricFactor rank (shapeNumel s))
    (outputCotangent : T s)
    (lr : Float)
    : IO (StepResult Params s) := do
  let mass := TensorStruct.map torch.ones_like params
  stepWithFactorAndMass params forward mass outputFactor outputCotangent lr

private def prepareCrossEntropyStep
    {Params : Type} [TensorStruct Params]
    {batch seq vocab : UInt64}
    (params : Params)
    (forward : Params → IO (T #[batch, seq, vocab]))
    (targets : T #[batch, seq])
    : IO (Params × T #[batch, seq, vocab] × T #[] × T #[batch, seq, vocab]) := do
  let paramsReq := TensorStruct.makeLeafParams params
  let predictionReq ← forward paramsReq
  let logitsFlat : T #[batch * seq, vocab] := reshape predictionReq #[batch * seq, vocab]
  let targetsFlat : T #[batch * seq] := reshape targets #[batch * seq]
  let lossT := nn.cross_entropy logitsFlat targetsFlat
  let outputCotangent := autograd.detach (autograd.grad lossT predictionReq (ones #[]))
  pure (paramsReq, predictionReq, lossT, outputCotangent)

private def sampledFisherCotangentRows
    {batch seq vocab : UInt64}
    (predictionReq : T #[batch, seq, vocab])
    (probeCount : UInt64)
    : IO { rows : Array (T #[batch, seq, vocab]) // rows.size = probeCount.toNat } := do
  if probeCount == 0 then
    throw <| IO.userError "RiemannianTreeSGD.stepCrossEntropySampledFisher requires probeCount > 0"
  let tokenCount := batch * seq
  let probs3d := autograd.detach (nn.softmax predictionReq (-1))
  let probs : T #[tokenCount, vocab] := reshape probs3d #[tokenCount, vocab]
  let scale := 1.0 / Float.sqrt probeCount.toFloat
  let srcOnes0 : T #[tokenCount, 1] := ones #[tokenCount, 1]
  let srcOnes :=
    if srcOnes0.device == probs.device then srcOnes0 else srcOnes0.to probs.device
  let rec sampleRows (n : Nat) : IO { rows : List (T #[batch, seq, vocab]) // rows.length = n } := do
    match n with
    | 0 => pure ⟨[], by simp⟩
    | n + 1 =>
      let sampledIdx : T #[tokenCount, 1] ← nn.multinomial probs 1 false
      let base : T #[tokenCount, vocab] := zeros_like probs
      let sampledOneHot := scatter_2d base 1 sampledIdx srcOnes
      let rowFlat := mul_scalar (sampledOneHot - probs) scale
      let row : T #[batch, seq, vocab] := reshape rowFlat #[batch, seq, vocab]
      let ⟨rest, hRest⟩ ← sampleRows n
      pure ⟨row :: rest, by simp [hRest]⟩
  let ⟨rowsList, hRowsList⟩ ← sampleRows probeCount.toNat
  let rows := rowsList.toArray
  have hRows : rows.size = probeCount.toNat := by
    simp [rows, hRowsList]
  pure ⟨rows, hRows⟩

private def sampledFisherTreeFactor
    {Params : Type} [TensorStruct Params]
    {batch seq vocab : UInt64} {rank : UInt64}
    (predictionReq : T #[batch, seq, vocab])
    (paramsReq : Params)
    (rows : Array (T #[batch, seq, vocab]))
    (rows_size : rows.size = rank.toNat)
    : TreeMetricFactor Params rank :=
  buildTreeFactorFromCotangentRows predictionReq paramsReq rows rows_size

/-- Cross-entropy step for tensor-logit models with unit diagonal mass. -/
def stepCrossEntropy
    {Params : Type} [TensorStruct Params]
    {batch seq vocab : UInt64}
    (params : Params)
    (forward : Params → IO (T #[batch, seq, vocab]))
    (targets : T #[batch, seq])
    (lr : Float)
    (gradClip : Float := 0.0)
    : IO (StepResult Params #[batch, seq, vocab]) := do
  let (paramsReq, predictionReq, lossT, outputCotangent) ←
    prepareCrossEntropyStep params forward targets
  let outDim := shapeNumel #[batch, seq, vocab]
  let outputFactor : MetricFactor outDim outDim := MetricFactor.identity outDim
  let mass := TensorStruct.map torch.ones_like params
  if gradClip > 0.0 then
    let paramGrad := clipScale (paramVjpFromPrepared predictionReq paramsReq outputCotangent) gradClip
    let factor := buildTreeFactorFromOutputFactor predictionReq paramsReq outputFactor
    let update := TreeMetricFactor.solveWoodbury mass factor paramGrad
    let inner := TreeMetricFactor.woodburyInner mass factor
    let innerInv := linalg.inv inner
    let residual := TreeMetricFactor.applyRegularized mass factor update |> fun x => TensorStruct.sub x paramGrad
    let params' := torch.Optim.apply_updates params (TensorStruct.scale update (-lr))
    pure {
      params := params'
      prediction := autograd.detach predictionReq
      diagnostics := {
        gradientNorm := Float.sqrt (nn.item (treeDot paramGrad paramGrad))
        factorRank := outDim
        innerConditionEstimate := linalg.spectralNorm inner * linalg.spectralNorm innerInv
        residualNorm := Float.sqrt (nn.item (treeDot residual residual))
        updateNorm := Float.sqrt (nn.item (treeDot update update))
      }
      loss := nn.item lossT
    }
  else
    let result := stepPreparedWithFactorAndMass
      (rank := outDim)
      params paramsReq predictionReq mass outputFactor outputCotangent lr
    pure { result with loss := nn.item lossT }

/-- Cross-entropy step using a sampled logits-Fisher pullback factor.
    The loss gradient remains exact; only the output metric factor is sketched. -/
def stepCrossEntropySampledFisher
    {Params : Type} [TensorStruct Params]
    {batch seq vocab : UInt64}
    (params : Params)
    (forward : Params → IO (T #[batch, seq, vocab]))
    (targets : T #[batch, seq])
    (probeCount : UInt64)
    (lr : Float)
    (gradClip : Float := 0.0)
    : IO (StepResult Params #[batch, seq, vocab]) := do
  let (paramsReq, predictionReq, lossT, outputCotangent) ←
    prepareCrossEntropyStep params forward targets
  let mass := TensorStruct.map torch.ones_like params
  let gradient0 := paramVjpFromPrepared predictionReq paramsReq outputCotangent
  let gradient := if gradClip > 0.0 then clipScale gradient0 gradClip else gradient0
  let ⟨rows, hRows⟩ ← sampledFisherCotangentRows predictionReq probeCount
  let factor :=
    sampledFisherTreeFactor
      (rank := probeCount)
      predictionReq paramsReq rows hRows
  let result :=
    finalizePreparedStepWithTreeFactor
      (rank := probeCount)
      params predictionReq mass factor gradient lr
  pure { result with loss := nn.item lossT }

end torch.Optim.RiemannianTreeSGD
