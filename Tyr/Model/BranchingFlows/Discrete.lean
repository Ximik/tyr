import Tyr.Model.BranchingFlows

/-!
  Discrete Flowfusion-style helpers for BranchingFlows.

  This ports the schedule and probability-velocity equations used by
  `Flowfusion.DistNoisyInterpolatingDiscreteFlow`: a convex path over
  endpoint target, uniform noise, and the source/mask token.
-/

namespace torch.branching

structure DistNoisyDiscreteConfig where
  vocabSize : Nat
  targetTime : TimeDist := TimeDist.betaTwoTwo
  noiseTime : TimeDist := TimeDist.betaTwoTwo
  omegaUniform : Float := 0.2
  maskToken? : Option Nat := none
  eps : Float := 1.0e-10
  /-- Labels are drawn from a nonempty vocabulary that holds the mask token; checked when the
      config is built. -/
  vocabPos : 0 < vocabSize := by decide
  maskInVocab : maskToken?.all (· < vocabSize) = true := by decide
  /-- Normalizers divide by `eps`-floored totals; checked when the config is built. -/
  epsPos : 0 < eps := by native_decide

namespace DistNoisyDiscreteConfig

def qm9 (vocabSize maskToken : Nat) (hVocab : 0 < vocabSize := by decide)
    (hMask : maskToken < vocabSize := by decide) : DistNoisyDiscreteConfig :=
  { vocabSize,
    targetTime := TimeDist.betaTwoTwo,
    noiseTime := TimeDist.betaTwoTwo,
    omegaUniform := 0.2,
    maskToken? := some maskToken
    vocabPos := hVocab
    maskInVocab := by simpa using hMask }

private def clampNonnegative (x : Float) : Float :=
  max x 0.0

def k1 (cfg : DistNoisyDiscreteConfig) (t : Float) : Float :=
  cfg.targetTime.cdf t

def dk1 (cfg : DistNoisyDiscreteConfig) (t : Float) : Float :=
  cfg.targetTime.pdf t

def k2Tilde (cfg : DistNoisyDiscreteConfig) (t : Float) : Float :=
  cfg.noiseTime.cdf t

def dk2Tilde (cfg : DistNoisyDiscreteConfig) (t : Float) : Float :=
  cfg.noiseTime.pdf t

def k2 (cfg : DistNoisyDiscreteConfig) (t : Float) : Float :=
  cfg.omegaUniform * (1.0 - cfg.k1 t) * cfg.k2Tilde t

def dk2 (cfg : DistNoisyDiscreteConfig) (t : Float) : Float :=
  let k1t := cfg.k1 t
  let dk1t := cfg.dk1 t
  let k2t := cfg.k2Tilde t
  let dk2t := cfg.dk2Tilde t
  cfg.omegaUniform * (-(dk1t * k2t) + (1.0 - k1t) * dk2t)

def k3 (cfg : DistNoisyDiscreteConfig) (t : Float) : Float :=
  1.0 - cfg.k1 t - cfg.k2 t

def dk3 (cfg : DistNoisyDiscreteConfig) (t : Float) : Float :=
  -(cfg.dk1 t + cfg.dk2 t)

private def normalize3 (a b c eps : Float) : Float × Float × Float :=
  let a := clampNonnegative a
  let b := clampNonnegative b
  let c := clampNonnegative c
  let s := max (a + b + c) eps
  (a / s, b / s, c / s)

/-- Mixture weights `(target, uniform, source)` at a single time. -/
def weights (cfg : DistNoisyDiscreteConfig) (t : Float) : Float × Float × Float :=
  normalize3 (cfg.k1 t) (cfg.k2 t) (cfg.k3 t) cfg.eps

/-- Conditional mixture weights from an existing state at `t0` to `t`. -/
def conditionalWeights (cfg : DistNoisyDiscreteConfig) (t0 t : Float) :
    Float × Float × Float :=
  let k10 := cfg.k1 t0
  let k20 := cfg.k2 t0
  let k30 := 1.0 - k10 - k20
  let k1t := cfg.k1 t
  let k2t := cfg.k2 t
  let k3t := 1.0 - k1t - k2t
  let denom := max k30 cfg.eps
  let w3 := k3t / denom
  let w1 := (k1t * denom - k10 * k3t) / denom
  let w2 := (k2t * denom - k20 * k3t) / denom
  normalize3 w1 w2 w3 cfg.eps

def oneHot (cfg : DistNoisyDiscreteConfig) (label : Nat) : Array Float :=
  (Array.range cfg.vocabSize).map (fun i => if i == label then 1.0 else 0.0)

def uniformDistribution (cfg : DistNoisyDiscreteConfig) : Array Float :=
  let p := 1.0 / cfg.vocabSize.toFloat
  Array.replicate cfg.vocabSize p

def mixtureDistribution
    (cfg : DistNoisyDiscreteConfig)
    (source target : Nat)
    (wTarget wUniform wSource : Float) :
    Array Float :=
  let sourceDist := cfg.oneHot source
  let targetDist := cfg.oneHot target
  let uniformDist := cfg.uniformDistribution
  (Array.range cfg.vocabSize).map (fun i =>
    wTarget * targetDist[i]! + wUniform * uniformDist[i]! + wSource * sourceDist[i]!)

def distribution (cfg : DistNoisyDiscreteConfig) (source target : Nat) (t : Float) :
    Array Float :=
  let (w1, w2, w3) := cfg.weights t
  cfg.mixtureDistribution source target w1 w2 w3

def conditionalDistribution
    (cfg : DistNoisyDiscreteConfig)
    (source target : Nat)
    (t0 t : Float) :
    Array Float :=
  let (w1, w2, w3) := cfg.conditionalWeights t0 t
  cfg.mixtureDistribution source target w1 w2 w3

def randUniformLabel (cfg : DistNoisyDiscreteConfig) (rng : Rng) : Nat × Rng :=
  randNat rng cfg.vocabSize

def bridgeWithWeights
    (cfg : DistNoisyDiscreteConfig)
    (source target : Nat)
    (weights : Float × Float × Float)
    (rng : Rng) :
    Nat × Rng :=
  let (w1, w2, _w3) := weights
  let (r, rng) := randFloat rng
  if r < w1 then
    (target, rng)
  else if r < w1 + w2 then
    cfg.randUniformLabel rng
  else
    (source, rng)

def bridge (cfg : DistNoisyDiscreteConfig) (source target : Nat) (t : Float)
    (rng : Rng) : Nat × Rng :=
  cfg.bridgeWithWeights source target (cfg.weights t) rng

def bridgeFrom (cfg : DistNoisyDiscreteConfig) (source target : Nat) (t0 t : Float)
    (rng : Rng) : Nat × Rng :=
  cfg.bridgeWithWeights source target (cfg.conditionalWeights t0 t) rng

/-- Deterministic representative for APIs that cannot consume randomness. -/
def modeBridgeFrom (cfg : DistNoisyDiscreteConfig) (source target : Nat) (t0 t : Float) : Nat :=
  let (w1, w2, w3) := cfg.conditionalWeights t0 t
  if w1 >= w2 && w1 >= w3 then
    target
  else
    source

private def softmaxLogits (cfg : DistNoisyDiscreteConfig) (logits : Array Float) :
    Array Float :=
  let vals := (Array.range cfg.vocabSize).map (fun i => logits.getD i 0.0)
  let m := vals.foldl max (vals[0]'(by simp [vals, cfg.vocabPos]))
  let exps := vals.map (fun x => Float.exp (x - m))
  let total := max (exps.foldl (fun acc x => acc + x) 0.0) cfg.eps
  exps.map (fun x => x / total)

private def normalizeDistribution (cfg : DistNoisyDiscreteConfig) (xs : Array Float) :
    Array Float :=
  let clamped := (Array.range cfg.vocabSize).map (fun i => clampNonnegative (xs.getD i 0.0))
  let total := clamped.foldl (fun acc x => acc + x) 0.0
  if total <= cfg.eps then
    cfg.uniformDistribution
  else
    clamped.map (fun x => x / total)

def stepDistribution
    (cfg : DistNoisyDiscreteConfig)
    (currentLabel : Nat)
    (targetLogits : Array Float)
    (t0 t1 : Float) :
    Array Float :=
  let dt := t1 - t0
  if dt <= 0.0 then
    cfg.oneHot currentLabel
  else
    let pred := cfg.softmaxLogits targetLogits
    let current := cfg.oneHot currentLabel
    let uniform := cfg.uniformDistribution
    let k1t := cfg.k1 t0
    let k2t := cfg.k2 t0
    let k3t := 1.0 - k1t - k2t
    let dk1t := cfg.dk1 t0
    let dk2t := cfg.dk2 t0
    let dk3t := -(dk1t + dk2t)
    let next :=
      match cfg.maskToken? with
      | some maskToken =>
          let r1 := dk1t / (cfg.eps + k1t)
          let r2 := dk2t / (cfg.eps + k2t)
          let r3 := dk3t / (cfg.eps + k3t)
          let bt := min r1 (min r2 r3)
          let a1 := dk1t - k1t * bt
          let a2 := dk2t - k2t * bt
          let a3 := dk3t - k3t * bt
          (Array.range cfg.vocabSize).map (fun i =>
            let maskVelocity := if i == maskToken then a3 else 0.0
            current[i]! + dt * (a1 * pred[i]! + a2 * uniform[i]! + bt * current[i]! + maskVelocity))
      | none =>
          let beta := dk3t / max k3t cfg.eps
          let a1 := dk1t - k1t * beta
          let a2 := dk2t - k2t * beta
          (Array.range cfg.vocabSize).map (fun i =>
            current[i]! + dt * (a1 * pred[i]! + a2 * uniform[i]! + beta * current[i]!))
    cfg.normalizeDistribution next

private def sampleCategorical (probs : Array Float) (rng : Rng) : Nat × Rng := Id.run do
  if probs.isEmpty then
    return (0, rng)
  let (r, rng) := randFloat rng
  let mut acc := 0.0
  let mut out := probs.size - 1
  for i in [:probs.size] do
    acc := acc + probs[i]!
    if r <= acc then
      out := i
      break
  return (out, rng)

def modeOfDistribution (_cfg : DistNoisyDiscreteConfig) (probs : Array Float) (fallback : Nat := 0) : Nat :=
  if probs.isEmpty then
    fallback
  else Id.run do
    let mut out := 0
    let mut best := probs[0]!
    for i in [:probs.size] do
      let p := probs[i]!
      if p > best then
        best := p
        out := i
    return out

def stepLabel
    (cfg : DistNoisyDiscreteConfig)
    (currentLabel : Nat)
    (targetLogits : Array Float)
    (t0 t1 : Float)
    (rng : Rng) :
    Nat × Rng :=
  sampleCategorical (cfg.stepDistribution currentLabel targetLogits t0 t1) rng

def stepLabelMode
    (cfg : DistNoisyDiscreteConfig)
    (currentLabel : Nat)
    (targetLogits : Array Float)
    (t0 t1 : Float) : Nat :=
  cfg.modeOfDistribution (cfg.stepDistribution currentLabel targetLogits t0 t1) currentLabel

def lossScale (_cfg : DistNoisyDiscreteConfig) (t : Float)
    (pow : Float := 2.0) (eps : Float := 0.05) : Float :=
  1.0 / Float.pow ((1.0 + eps) - t) pow

end DistNoisyDiscreteConfig

end torch.branching
