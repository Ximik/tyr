import Tyr.Typed
import Tyr.TensorStruct
import Tyr.Optim
import Tyr.DiffEq.Solver.Euler
import Tyr.DiffEq.Integrate

/-!
  Flow language model helpers.

  This module ports the core tensor contracts from the sibling FLM repository:
  continuous one-hot corruption, flow-matching loss, ODE-based sampling, and
  the PSD semigroup target used by FMLM. The transformer itself is supplied by
  the caller through `FlowDenoiser`, which keeps this module independent of any
  particular backbone implementation.
-/

namespace torch.flm

open torch

/-- Runtime knobs shared by FLM and FMLM losses. -/
structure FlowConfig where
  tMin : Float := 0.0
  tMax : Float := 1.0
  softcap : Float := 30.0
  eps : Float := 1.0e-5
  weightDecay : Float := 0.0
  gradClip : Float := 0.0
  deriving Repr

/-- Continuous/discrete time conversion.

  The Python FLM code uses LUTs for alpha/gamma conversion on discrete states.
  Tyr callers can plug that mapping in here; the identity map gives the standard
  linear flow interpolation used by the core FLM equations. -/
structure TimeMap where
  tauToT : {batch : UInt64} → DTensor #[batch] .Float32 → DTensor #[batch] .Float32
  tToTau : {batch : UInt64} → DTensor #[batch] .Float32 → DTensor #[batch] .Float32

def TimeMap.identity : TimeMap :=
  { tauToT := fun {_batch} tau => tau
    tToTau := fun {_batch} t => t }

/-- Backbone interface for continuous-token denoisers.

  `x` has shape `[batch, seq, vocab]`. `tau` is the source time, and
  `tauPrime` is present for flow-map/FMLM denoisers that condition on a target
  time. The returned tensor is raw logits; this module applies softcap and
  log-softmax before losses or semigroup targets consume it.

  `forwardPure` is required for ODE sampling, because DiffEq vector fields are
  pure functions. It should be the inference/no-grad equivalent of `forward`. -/
structure FlowDenoiser (seq vocab : UInt64) (Params : Type) where
  forward : {batch : UInt64} → Params → DTensor #[batch, seq, vocab] .Float32 → DTensor #[batch] .Float32
    → Option (DTensor #[batch] .Float32) → IO (DTensor #[batch, seq, vocab] .Float32)
  forwardPure : Option ({batch : UInt64} → Params → DTensor #[batch, seq, vocab] .Float32 → DTensor #[batch] .Float32
    → Option (DTensor #[batch] .Float32) → DTensor #[batch, seq, vocab] .Float32) := none
  lossWeight : Option ({batch : UInt64} → Params → DTensor #[batch] .Float32 → Option (DTensor #[batch] .Float32)
    → IO (DTensor #[batch] .Float32)) := none

structure LossReport where
  loss : Float
  nlls : Float
  numTokens : Float
  deriving Repr

structure SemigroupTimes (batch : UInt64) where
  tauS : DTensor #[batch] .Float32
  tauU : DTensor #[batch] .Float32
  tauT : DTensor #[batch] .Float32

private def expandBatch {batch seq : UInt64} (x : DTensor #[batch] .Float32) : DTensor #[batch, seq] .Float32 :=
  Tensor.expand (Tensor.unsqueeze x 1) #[batch, seq]

private def expandTime3 {batch seq vocab : UInt64} (x : DTensor #[batch] .Float32)
    : DTensor #[batch, seq, vocab] .Float32 :=
  Tensor.expand (Tensor.unsqueeze (Tensor.unsqueeze x 1) 2) #[batch, seq, vocab]

/-- Token ids to one-hot vectors using `eye` as the embedding table. -/
def oneHot {batch seq vocab : UInt64} (tokens : DTensor #[batch, seq] .Int64)
    : DTensor #[batch, seq, vocab] .Float32 :=
  Tensor.embedding tokens (Tensor.eye vocab)

/-- Sample uniformly from the configured training time interval. -/
def sampleTau (cfg : FlowConfig) (batch : UInt64) : IO (DTensor #[batch] .Float32) := do
  let eps ← Tensor.uniformLike (Tensor.zeros #[batch])
  pure (eps * (cfg.tMax - cfg.tMin) + cfg.tMin)

/-- Continuous FLM corruption: `x_t = (1 - t) noise + t one_hot(x_1)`. -/
def corruptContinuous {batch seq vocab : UInt64}
    (tokens : DTensor #[batch, seq] .Int64)
    (t : DTensor #[batch] .Float32)
    : IO (DTensor #[batch, seq, vocab] .Float32 × DTensor #[batch, seq, vocab] .Float32) := do
  let target := oneHot (vocab := vocab) tokens
  let noise ← Tensor.randn #[batch, seq, vocab]
  let t3 := expandTime3 (seq := seq) (vocab := vocab) t
  let one := Tensor.onesLike target
  pure (((one - t3) * noise + t3 * target), target)

private def softcap {s : Shape} (x : DTensor s .Float32) (cap : Float) : DTensor s .Float32 :=
  Tensor.tanh (x / cap) * cap

def cappedLogSoftmax {s : Shape} (cfg : FlowConfig) (logits : DTensor s .Float32) : DTensor s .Float32 :=
  Tensor.logSoftmax (softcap logits cfg.softcap) (-1)

def forwardLogProbs {seq vocab batch : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (x : DTensor #[batch, seq, vocab] .Float32)
    (tau : DTensor #[batch] .Float32)
    (tauPrime : Option (DTensor #[batch] .Float32) := none)
    : IO (DTensor #[batch, seq, vocab] .Float32) := do
  let logits ← model.forward params x tau tauPrime
  pure (cappedLogSoftmax cfg logits)

def forwardLogProbsPure {seq vocab batch : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (forwardPure : {batch : UInt64} → Params → DTensor #[batch, seq, vocab] .Float32 → DTensor #[batch] .Float32
      → Option (DTensor #[batch] .Float32) → DTensor #[batch, seq, vocab] .Float32)
    (params : Params)
    (x : DTensor #[batch, seq, vocab] .Float32)
    (tau : DTensor #[batch] .Float32)
    (tauPrime : Option (DTensor #[batch] .Float32) := none)
    : DTensor #[batch, seq, vocab] .Float32 :=
  cappedLogSoftmax cfg (forwardPure params x tau tauPrime)

/-- Per-token cross entropy against a dense one-hot/probability target. -/
def denseTargetNll {batch seq vocab : UInt64}
    (logProbs target : DTensor #[batch, seq, vocab] .Float32) : DTensor #[batch, seq] .Float32 :=
  (Tensor.mulScalar (Tensor.sumDim (target * logProbs) 2) (-1.0)).reshapeChecked #[batch, seq]

private def maybeApplyLossWeight {batch seq : UInt64} {Params : Type}
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (tau : DTensor #[batch] .Float32)
    (tauPrime : Option (DTensor #[batch] .Float32))
    (loss : DTensor #[batch, seq] .Float32)
    : IO (DTensor #[batch, seq] .Float32) := do
  match model.lossWeight with
  | none => pure loss
  | some weightFn =>
      let w ← weightFn params tau tauPrime
      let w2 := expandBatch (seq := seq) w
      pure (Tensor.exp (Tensor.mulScalar w2 (-1.0)) * loss + w2)

/-- FLM flow-matching loss at caller-provided times. -/
def flowLossGivenTau {batch seq vocab : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (tokens : DTensor #[batch, seq] .Int64)
    (tau : DTensor #[batch] .Float32)
    : IO (DTensor #[batch, seq] .Float32) := do
  let t := timeMap.tauToT tau
  let (xT, target) ← corruptContinuous (vocab := vocab) tokens t
  let logProbs ← forwardLogProbs cfg model params xT tau none
  let loss := denseTargetNll logProbs target
  maybeApplyLossWeight model params tau none loss

/-- FLM flow-matching loss with uniformly sampled training times. -/
def flowLoss {batch seq vocab : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (tokens : DTensor #[batch, seq] .Int64)
    : IO (DTensor #[batch, seq] .Float32) := do
  let tau ← sampleTau cfg batch
  flowLossGivenTau cfg timeMap model params tokens tau

def maskedMean {batch seq : UInt64} (loss validTokens : DTensor #[batch, seq] .Float32)
    (eps : Float := 1.0e-8) : DTensor #[] .Float32 :=
  let nlls := Tensor.sumAll (loss * validTokens)
  let denom := Tensor.sumAll validTokens
  Tensor.div nlls (denom + eps)

/-- Optimizer step for an FLM loss. `validTokens` is a float mask. -/
def trainStep {seq vocab batch : UInt64} {Params : Type} [TensorStruct Params]
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (optState : Optim.AdamWState Params)
    (tokens : DTensor #[batch, seq] .Int64)
    (validTokens : DTensor #[batch, seq] .Float32)
    (lr : Float)
    (clipGrads : Params → Float → IO Unit := fun _ _ => pure ())
    : IO (Params × Optim.AdamWState Params × LossReport) := do
  let params := TensorStruct.zeroGrads params
  let perToken ← flowLoss cfg timeMap model params tokens
  let lossT := maskedMean perToken validTokens
  Tensor.backwardLoss lossT
  if cfg.gradClip > 0 then
    clipGrads params cfg.gradClip
  let grads := TensorStruct.grads params
  let opt := Optim.adamw (lr := lr) (weight_decay := cfg.weightDecay)
  let (params', optState') := Optim.step opt params grads optState
  let nlls := Tensor.sumAll (perToken * validTokens)
  let ntok := Tensor.sumAll validTokens
  let report : LossReport :=
    { loss := Tensor.item lossT
      nlls := Tensor.item nlls
      numTokens := Tensor.item ntok }
  pure (params', optState', report)

private def pureForwardOrThrow {seq vocab : UInt64} {Params : Type}
    (model : FlowDenoiser seq vocab Params)
    : IO ({batch : UInt64} → Params → DTensor #[batch, seq, vocab] .Float32 → DTensor #[batch] .Float32
      → Option (DTensor #[batch] .Float32) → DTensor #[batch, seq, vocab] .Float32) := do
  match model.forwardPure with
  | some forwardPure => pure forwardPure
  | none =>
      throw (IO.userError
        "generateFLM requires FlowDenoiser.forwardPure because DiffEq ODE terms are pure")

private def scalarMappedTime {batch : UInt64} (timeMap : TimeMap) (tau : Float) : Float :=
  let tauT := Tensor.full #[batch] tau
  Tensor.item (Tensor.meanAll (timeMap.tauToT tauT))

private def flowVelocityAtT {batch seq vocab : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (forwardPure : {batch : UInt64} → Params → DTensor #[batch, seq, vocab] .Float32 → DTensor #[batch] .Float32
      → Option (DTensor #[batch] .Float32) → DTensor #[batch, seq, vocab] .Float32)
    (params : Params)
    (t : DiffEq.Time)
    (z : DTensor #[batch, seq, vocab] .Float32)
    : DTensor #[batch, seq, vocab] .Float32 :=
  let tTensor := Tensor.full #[batch] t
  let tau := timeMap.tToTau tTensor
  let logPred := forwardLogProbsPure cfg forwardPure params z tau none
  let dataPred := Tensor.exp logPred
  let t3 := expandTime3 (seq := seq) (vocab := vocab) tTensor
  let one := Tensor.onesLike z
  Tensor.div (dataPred - z) (one - t3 + cfg.eps)

private def finalFlowProjection {batch seq vocab : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (forwardPure : {batch : UInt64} → Params → DTensor #[batch, seq, vocab] .Float32 → DTensor #[batch] .Float32
      → Option (DTensor #[batch] .Float32) → DTensor #[batch, seq, vocab] .Float32)
    (params : Params)
    (t : DiffEq.Time)
    (z : DTensor #[batch, seq, vocab] .Float32)
    : DTensor #[batch, seq, vocab] .Float32 :=
  let tTensor := Tensor.full #[batch] t
  let tau := timeMap.tToTau tTensor
  Tensor.exp (forwardLogProbsPure cfg forwardPure params z tau none)

/-- Generate tokens with a caller-supplied ODE solver over the FLM flow field. -/
def generateFLMWithSolver {seq vocab batch : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (steps : Nat)
    (solver : DiffEq.AbstractSolver
      (DiffEq.ODETerm (DTensor #[batch, seq, vocab] .Float32) Unit)
      (DTensor #[batch, seq, vocab] .Float32)
      (DTensor #[batch, seq, vocab] .Float32)
      DiffEq.Time
      Unit)
    (maxSteps : Nat := 4096)
    : IO (DTensor #[batch, seq] .Int64) := do
  if steps == 0 then
    throw (IO.userError "generateFLM requires steps > 0")
  let forwardPure ← pureForwardOrThrow model
  let z0 ← Tensor.randn #[batch, seq, vocab]
  let t0 := scalarMappedTime (batch := batch) timeMap cfg.tMin
  let t1 := scalarMappedTime (batch := batch) timeMap cfg.tMax
  let dt := (t1 - t0) / steps.toFloat
  let tFinalModel := t0 + dt * (steps - 1).toFloat
  let zBeforeFinal ←
    if steps == 1 then
      pure z0
    else do
      let term : DiffEq.ODETerm (DTensor #[batch, seq, vocab] .Float32) Unit :=
        { vectorField := fun t z _ => flowVelocityAtT cfg timeMap forwardPure params t z }
      let sol :=
        DiffEq.diffeqsolve
          (Term := DiffEq.ODETerm (DTensor #[batch, seq, vocab] .Float32) Unit)
          (Y := DTensor #[batch, seq, vocab] .Float32)
          (VF := DTensor #[batch, seq, vocab] .Float32)
          (Control := DiffEq.Time)
          (Args := Unit)
          (Controller := DiffEq.ConstantStepSize)
          term solver t0 tFinalModel (some dt) z0 () (saveat := { t1 := true })
          (maxSteps := maxSteps)
      if sol.result.isFailure then
        throw (IO.userError s!"generateFLM solve failed: {sol.result.message}")
      match sol.ys with
      | some ys =>
          if ys.size > 0 then
            pure ys[ys.size - 1]!
          else
            throw (IO.userError "generateFLM solve did not save a terminal state")
      | none =>
          throw (IO.userError "generateFLM solve did not save a terminal state")
  let dataPred := finalFlowProjection cfg timeMap forwardPure params tFinalModel zBeforeFinal
  pure ((Tensor.argmax dataPred 2).reshapeChecked #[batch, seq])

/-- Generate tokens through DiffEq Euler from Gaussian noise to one-hot data. -/
def generateFLM {seq vocab batch : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (steps : Nat)
    : IO (DTensor #[batch, seq] .Int64) := do
  let solver :=
    DiffEq.Euler.solver
      (Term := DiffEq.ODETerm (DTensor #[batch, seq, vocab] .Float32) Unit)
      (Y := DTensor #[batch, seq, vocab] .Float32)
      (VF := DTensor #[batch, seq, vocab] .Float32)
      (Args := Unit)
  generateFLMWithSolver cfg timeMap model params steps solver

/-- Interpolate from a source state toward a predicted data point. -/
def flowInterpolate {batch seq vocab : UInt64}
    (cfg : FlowConfig)
    (xS dataPred : DTensor #[batch, seq, vocab] .Float32)
    (s t : DTensor #[batch] .Float32)
    : DTensor #[batch, seq, vocab] .Float32 :=
  let s3 := expandTime3 (seq := seq) (vocab := vocab) s
  let t3 := expandTime3 (seq := seq) (vocab := vocab) t
  let one := Tensor.onesLike xS
  let denom := one - s3 + cfg.eps
  Tensor.div (one - t3) denom * xS + Tensor.div (t3 - s3) denom * dataPred

/-- PSD semigroup mixing coefficient from the FMLM paper/code. -/
def psdLambda {batch seq vocab : UInt64}
    (cfg : FlowConfig)
    (s u t : DTensor #[batch] .Float32)
    : DTensor #[batch, seq, vocab] .Float32 :=
  let s3 := expandTime3 (seq := seq) (vocab := vocab) s
  let u3 := expandTime3 (seq := seq) (vocab := vocab) u
  let t3 := expandTime3 (seq := seq) (vocab := vocab) t
  let one := Tensor.onesLike s3
  let num := (one - t3) * (u3 - s3)
  let den := (one - u3) * (t3 - s3) + cfg.eps
  Tensor.div num den

/-- FMLM PSD target: `lambda D_su + (1-lambda) D_ut`. -/
def psdTarget {batch seq vocab : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (xS : DTensor #[batch, seq, vocab] .Float32)
    (tauS tauU tauT : DTensor #[batch] .Float32)
    : IO (DTensor #[batch, seq, vocab] .Float32) := do
  let s := timeMap.tauToT tauS
  let u := timeMap.tauToT tauU
  let t := timeMap.tauToT tauT
  let logDSU ← forwardLogProbs cfg model params xS tauS (some tauU)
  let dSU := Tensor.exp logDSU
  let xSU := flowInterpolate cfg xS dSU s u
  let logDUT ← forwardLogProbs cfg model params xSU tauU (some tauT)
  let dUT := Tensor.exp logDUT
  let lam := psdLambda (seq := seq) (vocab := vocab) cfg s u t
  let one := Tensor.onesLike lam
  pure (lam * dSU + (one - lam) * dUT)

/-- FMLM PSD per-token loss for caller-provided semigroup times. -/
def psdLossGivenTau {batch seq vocab : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (tokens : DTensor #[batch, seq] .Int64)
    (tauS tauU tauT : DTensor #[batch] .Float32)
    : IO (DTensor #[batch, seq] .Float32) := do
  let s := timeMap.tauToT tauS
  let (xS, _) ← corruptContinuous (vocab := vocab) tokens s
  let target ← psdTarget cfg timeMap model params xS tauS tauU tauT
  let logDST ← forwardLogProbs cfg model params xS tauS (some tauT)
  let loss := denseTargetNll logDST target
  maybeApplyLossWeight model params tauS (some tauT) loss

/-- Diagonal FMLM loss. If no teacher target is supplied, this reduces to FLM. -/
def diagonalLossGivenTau {batch seq vocab : UInt64} {Params : Type}
    (cfg : FlowConfig)
    (timeMap : TimeMap)
    (model : FlowDenoiser seq vocab Params)
    (params : Params)
    (tokens : DTensor #[batch, seq] .Int64)
    (tau : DTensor #[batch] .Float32)
    (teacherTarget : Option (DTensor #[batch, seq, vocab] .Float32) := none)
    : IO (DTensor #[batch, seq] .Float32) := do
  let t := timeMap.tauToT tau
  let (xT, oneHotTarget) ← corruptContinuous (vocab := vocab) tokens t
  let target := teacherTarget.getD oneHotTarget
  let logProbs ← forwardLogProbs cfg model params xT tau (some tau)
  let loss := denseTargetNll logProbs target
  maybeApplyLossWeight model params tau (some tau) loss

/-- Sample off-diagonal PSD times with midpoint `u`. -/
def samplePsdTimes (batch : UInt64) : IO (SemigroupTimes batch) := do
  let dTau ← Tensor.uniformLike (Tensor.zeros #[batch])
  let eps ← Tensor.uniformLike (Tensor.zeros #[batch])
  let one := Tensor.onesLike dTau
  let tauS := eps * (one - dTau)
  let tauT := tauS + dTau
  let tauU := (tauS + tauT) * (0.5 : Float)
  pure { tauS, tauU, tauT }

end torch.flm
