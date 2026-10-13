/-
  Examples/GPT/NanoGPTCopy.lean

  EXPERIMENTAL: tiny nanoGPT-style prototype wiring for fast smoke/benchmark checks.
  This intentionally keeps model/data/training settings minimal.
-/
import Examples.GPT.GPT
import Examples.GPT.Train

namespace Examples.GPT.NanoGPTCopy

open torch
open torch.gpt
open torch.train

/-- EXPERIMENTAL prototype config for a tiny nanoGPT-like run. -/
structure PrototypeConfig where
  modelConfig : torch.gpt.Config
  trainConfig : torch.train.TrainConfig
  /-- Number of tokens in the synthetic character stream. -/
  streamTokens : UInt64 := 512
  /-- Steps used by `runPrototype`. -/
  trainSteps : Nat := 3
  /-- Steps used by `runBenchmark`. -/
  benchmarkSteps : Nat := 5
  /-- Use `trainStepWithAccum` when true; otherwise use `trainStep`. -/
  useGradAccum : Bool := false
  deriving Repr

/-- Tiny default config for quick local smoke tests (CPU-friendly). -/
def PrototypeConfig.tiny : PrototypeConfig := {
  modelConfig := {
    vocab_size := 65
    block_size := 16
    n_embd := 32
    n_head := 2
    n_layer := 1
    dropout := 0.0
  }
  trainConfig := {
    maxIters := 8
    evalInterval := 8
    logInterval := 8
    learningRate := 1e-3
    minLr := 1e-4
    warmupIters := 0
    lrDecayIters := 8
    gradClip := 1.0
    batchSize := 2
    blockSize := 16
    gradAccumSteps := 2
    device := Device.CPU
  }
  streamTokens := 512
  trainSteps := 3
  benchmarkSteps := 5
  useGradAccum := false
}

/-- Runtime stats emitted by prototype and benchmark helpers. -/
structure RunStats where
  steps : Nat
  elapsedNs : Nat
  elapsedMs : Float
  avgLoss : Float
  finalLoss : Float
  stepsPerSec : Float
  deriving Repr, Inhabited

/-- Float finite check helper for callers/tests. -/
def isFiniteFloat (x : Float) : Bool :=
  !x.isNaN && !x.isInf

/-- Random character-token stream in Shakespeare-sized vocab (65) by default. -/
def mkShakespeareLikeStream (cfg : PrototypeConfig) : IO (Tokens #[cfg.streamTokens]) := do
  Tensor.randint 0 cfg.modelConfig.vocab_size.toInt64 #[cfg.streamTokens]

private def validateConfig (cfg : PrototypeConfig) : IO Unit := do
  let needed := cfg.trainConfig.batchSize * (cfg.trainConfig.blockSize + 1)
  if cfg.trainConfig.blockSize > cfg.modelConfig.block_size then
    throw <| IO.userError
      s!"NanoGPTCopy invalid config: train blockSize ({cfg.trainConfig.blockSize}) > model block_size ({cfg.modelConfig.block_size})"
  if cfg.streamTokens <= needed then
    throw <| IO.userError
      s!"NanoGPTCopy invalid config: streamTokens ({cfg.streamTokens}) must be > batchSize*(blockSize+1) ({needed})"
  if cfg.useGradAccum && cfg.trainConfig.gradAccumSteps == 0 then
    throw <| IO.userError "NanoGPTCopy invalid config: gradAccumSteps must be > 0 when useGradAccum=true"

private def runSteps (cfg : PrototypeConfig) (steps : Nat) : IO RunStats := do
  validateConfig cfg
  let stream ← mkShakespeareLikeStream cfg
  let initParams ← GPTParams.init cfg.modelConfig cfg.trainConfig.device
  let opt := Optim.adamw (lr := cfg.trainConfig.learningRate)
  let mut params := initParams
  let mut optState := opt.init initParams
  let mut totalLoss : Float := 0.0
  let mut finalLoss : Float := 0.0

  let t0 ← IO.monoNanosNow
  for step in [:steps] do
    let lr := torch.train.getLr cfg.trainConfig step
    if cfg.useGradAccum then
      let getBatchFn : IO (Tokens #[cfg.trainConfig.batchSize, cfg.trainConfig.blockSize] × Tokens #[cfg.trainConfig.batchSize, cfg.trainConfig.blockSize]) :=
        torch.train.getBatch stream cfg.trainConfig.batchSize cfg.trainConfig.blockSize cfg.trainConfig.device
      let (nextParams, nextOptState, lossVal) ←
        torch.train.trainStepWithAccum cfg.trainConfig params optState getBatchFn lr
      params := nextParams
      optState := nextOptState
      finalLoss := lossVal
      totalLoss := totalLoss + lossVal
    else
      let (x, y) ← torch.train.getBatch stream cfg.trainConfig.batchSize cfg.trainConfig.blockSize cfg.trainConfig.device
      let (nextParams, nextOptState, lossVal) ←
        torch.train.trainStep cfg.trainConfig params optState x y lr
      params := nextParams
      optState := nextOptState
      finalLoss := lossVal
      totalLoss := totalLoss + lossVal
  let t1 ← IO.monoNanosNow

  let elapsedNs := t1 - t0
  let elapsedMs := elapsedNs.toFloat / 1000000.0
  let avgLoss :=
    if steps == 0 then 0.0 else totalLoss / steps.toFloat
  let stepsPerSec :=
    if elapsedNs == 0 then 0.0
    else steps.toFloat * 1000000000.0 / elapsedNs.toFloat
  return {
    steps := steps
    elapsedNs := elapsedNs
    elapsedMs := elapsedMs
    avgLoss := avgLoss
    finalLoss := finalLoss
    stepsPerSec := stepsPerSec
  }

/-- Tiny experimental prototype run (few steps, synthetic Shakespeare-like stream). -/
def runPrototype (cfg : PrototypeConfig := PrototypeConfig.tiny) : IO RunStats :=
  runSteps cfg cfg.trainSteps

/-- Tiny benchmark helper that runs a few steps and returns timing + loss stats. -/
def runBenchmark (cfg : PrototypeConfig := PrototypeConfig.tiny) : IO RunStats :=
  runSteps cfg cfg.benchmarkSteps

end Examples.GPT.NanoGPTCopy
