/-
  Examples/GPT/RiemannianNanoGPT.lean

  Exact-VJP Riemannian training path for the real masked self-attention GPT model.
  This keeps the full `Examples.GPT.GPT` architecture and uses a tree-structured
  Woodbury solve over the full parameter tree rather than flattening parameters
  by hand.
-/
import Tyr.Checkpoint
import Examples.GPT.Train
import Tyr.Optim.RiemannianTreeSGD

namespace Examples.GPT.RiemannianNanoGPT

open torch

variable {device : Device}
open torch.gpt
open torch.train
open torch.checkpoint

inductive PullbackMetricMode where
  | exact
  | sampledFisher
  deriving Repr, Inhabited, BEq

structure PullbackMetricConfig where
  mode : PullbackMetricMode := .exact
  fisherProbeCount : Nat := 8
  deriving Repr, Inhabited

/-- Experimental config for the Riemannian GPT prototype. -/
structure PrototypeConfig where
  modelConfig : torch.gpt.Config := {
    vocab_size := 65
    block_size := 4
    n_embd := 16
    n_head := 1
    n_layer := 1
    dropout := 0.0
  }
  trainConfig : torch.train.TrainConfig := {
    maxIters := 4
    evalInterval := 4
    logInterval := 4
    learningRate := 5e-3
    minLr := 1e-3
    warmupIters := 0
    lrDecayIters := 4
    gradClip := 0.0
    batchSize := 1
    blockSize := 4
    device := Device.CPU
  }
  streamTokens : Nat := 256
  trainSteps : Nat := 1
  benchmarkSteps : Nat := 2
  pullbackMetric : PullbackMetricConfig := {}
  deriving Repr

/-- Tiny default config for quick local smoke tests. -/
def PrototypeConfig.tiny : PrototypeConfig := {}

/-- Runtime stats emitted by prototype and benchmark helpers. -/
structure RunStats where
  steps : Nat
  elapsedNs : Nat
  elapsedMs : Float
  avgLoss : Float
  finalLoss : Float
  stepsPerSec : Float
  finalDiagnostics : torch.Optim.RiemannianTreeSGD.StepDiagnostics := {}
  deriving Repr, Inhabited

/-- Shakespeare character vocabulary (65 chars).
    This must match the `TrainGPT` preparation encoding. -/
def shakespeareChars : String :=
  "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"

def charToInt (c : Char) : Int64 :=
  match shakespeareChars.toList.findIdx? (· == c) with
  | some idx => idx.toInt64
  | none => 0

def intToChar (i : Int64) : Char :=
  shakespeareChars.toList.getD i.toUInt64.toNat '\n'

def encode (s : String) : Array Int64 :=
  s.toList.toArray.map charToInt

def decode (tokens : Array Int64) : String :=
  String.ofList (tokens.toList.map intToChar)

/-- Float finite check helper for callers/tests. -/
def isFiniteFloat (x : Float) : Bool :=
  !x.isNaN && !x.isInf

/-- Synthetic Shakespeare-sized token stream. -/
def mkShakespeareLikeStream (cfg : PrototypeConfig) : IO (Tokens (device := .CPU) #[cfg.streamTokens]) := do
  Tensor.randint 0 cfg.modelConfig.vocab_size.toInt64 #[cfg.streamTokens]

private def validateConfig (cfg : PrototypeConfig) : IO Unit := do
  let needed := cfg.trainConfig.batchSize * (cfg.trainConfig.blockSize + 1)
  if cfg.trainConfig.blockSize > cfg.modelConfig.block_size then
    throw <| IO.userError
      s!"RiemannianNanoGPT invalid config: train blockSize ({cfg.trainConfig.blockSize}) > model block_size ({cfg.modelConfig.block_size})"
  if cfg.streamTokens <= needed then
    throw <| IO.userError
      s!"RiemannianNanoGPT invalid config: streamTokens ({cfg.streamTokens}) must be > batchSize*(blockSize+1) ({needed})"
  if cfg.pullbackMetric.mode == .sampledFisher && cfg.pullbackMetric.fisherProbeCount == 0 then
    throw <| IO.userError "RiemannianNanoGPT invalid config: sampled Fisher requires fisherProbeCount > 0"

private def pullbackMetricLabel (cfg : PullbackMetricConfig) : String :=
  match cfg.mode with
  | .exact => "exact"
  | .sampledFisher => s!"sampled_fisher({cfg.fisherProbeCount})"

private def prototypeTrainBatch (cfg : PrototypeConfig)
    (params : GPTParams (device := cfg.trainConfig.device) cfg.modelConfig)
    (stream : Tokens (device := .CPU) #[cfg.streamTokens])
    (lr : Float)
    : IO (GPTParams (device := cfg.trainConfig.device) cfg.modelConfig × Float × torch.Optim.RiemannianTreeSGD.StepDiagnostics) := do
  let (x, y) ←
    torch.train.getBatch
      stream cfg.trainConfig.batchSize cfg.trainConfig.blockSize cfg.trainConfig.device
  let step ←
    match cfg.pullbackMetric.mode with
    | .exact =>
      torch.Optim.RiemannianTreeSGD.stepCrossEntropy
        params
        (fun p => gpt.forward p x true)
        y
        lr
        cfg.trainConfig.gradClip
    | .sampledFisher =>
      torch.Optim.RiemannianTreeSGD.stepCrossEntropySampledFisher
        params
        (fun p => gpt.forward p x true)
        y
        cfg.pullbackMetric.fisherProbeCount
        lr
        cfg.trainConfig.gradClip
  pure (step.params, step.loss, step.diagnostics)

private def runSteps (cfg : PrototypeConfig) (steps : Nat) : IO RunStats := do
  validateConfig cfg
  let stream ← mkShakespeareLikeStream cfg
  let mut params ← GPTParams.init cfg.modelConfig cfg.trainConfig.device
  let mut totalLoss : Float := 0.0
  let mut finalLoss : Float := 0.0
  let mut finalDiagnostics : torch.Optim.RiemannianTreeSGD.StepDiagnostics := {}

  let t0 ← IO.monoNanosNow
  for step in [:steps] do
    let lr := torch.train.getLr cfg.trainConfig step
    let (nextParams, lossVal, diagnostics) ← prototypeTrainBatch cfg params stream lr
    params := nextParams
    totalLoss := totalLoss + lossVal
    finalLoss := lossVal
    finalDiagnostics := diagnostics
  let t1 ← IO.monoNanosNow

  let elapsedNs := t1 - t0
  let elapsedMs := elapsedNs.toFloat / 1000000.0
  let avgLoss := if steps == 0 then 0.0 else totalLoss / steps.toFloat
  let stepsPerSec :=
    if elapsedNs == 0 then 0.0
    else steps.toFloat * 1000000000.0 / elapsedNs.toFloat
  pure {
    steps := steps
    elapsedNs := elapsedNs
    elapsedMs := elapsedMs
    avgLoss := avgLoss
    finalLoss := finalLoss
    stepsPerSec := stepsPerSec
    finalDiagnostics := finalDiagnostics
  }

/-- Tiny experimental prototype run. -/
def runPrototype (cfg : PrototypeConfig := PrototypeConfig.tiny) : IO RunStats :=
  runSteps cfg cfg.trainSteps

/-- Tiny benchmark helper. -/
def runBenchmark (cfg : PrototypeConfig := PrototypeConfig.tiny) : IO RunStats :=
  runSteps cfg cfg.benchmarkSteps

private def deviceName (device : Device) : String :=
  match device.kind with
  | DeviceKind.MPS => "MPS (Apple Silicon)"
  | DeviceKind.CUDA n => s!"CUDA:{n}"
  | DeviceKind.CPU => "CPU"

private def resolveDeviceFromEnv : IO Device := do
  let requestedDevice? := (← IO.getEnv "TYR_DEVICE")
  match requestedDevice?.map String.toLower with
  | some "auto" => getBestDevice
  | some "cuda" => Device.cuda 0
  | some "mps" => Device.mps
  | some "cpu" => pure Device.CPU
  | some _ => pure Device.CPU
  | none => pure Device.CPU

private def defaultTrainConfig (device : Device) (blockSize : Nat) : TrainConfig := {
  maxIters := 5000
  evalInterval := 500
  logInterval := 50
  learningRate := 1e-3
  minLr := 1e-4
  warmupIters := 100
  lrDecayIters := 2000
  gradClip := 1.0
  batchSize := 12
  blockSize := blockSize
  device := device
}

private def envFlagEnabled (name : String) : IO Bool := do
  match (← IO.getEnv name).map String.toLower with
  | some "1" | some "true" | some "yes" | some "on" => pure true
  | _ => pure false

private def envUInt64Or (name : String) (fallback : Nat) : IO Nat := do
  match (← IO.getEnv name) with
  | some raw =>
    match raw.toNat? with
    | some n => pure n
    | none => pure fallback
  | none => pure fallback

private def envNatOr (name : String) (fallback : Nat) : IO Nat := do
  match (← IO.getEnv name) with
  | some raw =>
    match raw.toNat? with
    | some n => pure n
    | none => pure fallback
  | none => pure fallback

private def applyTrainOverridesFromEnv (fallback : TrainConfig) : IO TrainConfig := do
  let maxIters ← envNatOr "TYR_RIEMANNIAN_MAX_ITERS" fallback.maxIters
  let evalInterval ← envNatOr "TYR_RIEMANNIAN_EVAL_INTERVAL" fallback.evalInterval
  let logInterval ← envNatOr "TYR_RIEMANNIAN_LOG_INTERVAL" fallback.logInterval
  pure {
    fallback with
    maxIters := maxIters
    evalInterval := evalInterval
    logInterval := logInterval
  }

private def envStringOr (name : String) (fallback : String) : IO String := do
  match ← IO.getEnv name with
  | some raw => pure raw
  | none => pure fallback

private def resolvePullbackMetricFromEnv (fallback : PullbackMetricConfig) : IO PullbackMetricConfig := do
  let mode :=
    match (← IO.getEnv "TYR_RIEMANNIAN_PULLBACK").map String.toLower with
    | some "exact" => PullbackMetricMode.exact
    | some "sampled_fisher" => PullbackMetricMode.sampledFisher
    | some "sampled-fisher" => PullbackMetricMode.sampledFisher
    | _ => fallback.mode
  let fisherProbeCount ← envUInt64Or "TYR_RIEMANNIAN_PROBES" fallback.fisherProbeCount
  pure { mode := mode, fisherProbeCount := fisherProbeCount }

/-- Optional smoke-mode config for compile/runtime verification of the full script path. -/
private def mainConfig (device : Device) : IO (Config × TrainConfig × PullbackMetricConfig × Option String) := do
  if ← envFlagEnabled "TYR_RIEMANNIAN_SMOKE" then
    let modelCfg := Config.tiny_shakespeare
    let trainCfg : TrainConfig := {
      maxIters := 1
      evalInterval := 1
      logInterval := 1
      learningRate := 5e-3
      minLr := 1e-3
      warmupIters := 0
      lrDecayIters := 1
      gradClip := 0.0
      batchSize := 1
      blockSize := 4
      device := device
    }
    let trainCfg ← applyTrainOverridesFromEnv trainCfg
    let pullbackMetric ←
      resolvePullbackMetricFromEnv { mode := .sampledFisher, fisherProbeCount := 4 }
    let checkpointDir ← envStringOr "TYR_RIEMANNIAN_CHECKPOINT_DIR" "checkpoints/riemannian_gpt_smoke"
    pure (modelCfg, trainCfg, pullbackMetric, some checkpointDir)
  else
    let modelCfg := Config.nanogpt_cpu_shakespeare
    let trainCfg ← applyTrainOverridesFromEnv (defaultTrainConfig device modelCfg.block_size)
    let pullbackMetric ←
      resolvePullbackMetricFromEnv { mode := .sampledFisher, fisherProbeCount := 32 }
    let checkpointDir ← envStringOr "TYR_RIEMANNIAN_CHECKPOINT_DIR" "checkpoints/riemannian_gpt"
    pure (modelCfg, trainCfg, pullbackMetric, some checkpointDir)

private def outputFactorRank (cfg : Config) (trainCfg : TrainConfig) (pullbackMetric : PullbackMetricConfig) : Nat :=
  match pullbackMetric.mode with
  | .exact => trainCfg.batchSize * trainCfg.blockSize * cfg.vocab_size
  | .sampledFisher => pullbackMetric.fisherProbeCount

private def riemannianTrainStep {modelCfg : Config} {batch seq : Nat}
    (trainCfg : TrainConfig)
    (pullbackMetric : PullbackMetricConfig)
    (params : GPTParams (device := device) modelCfg)
    (x : Tokens (device := device) #[batch, seq])
    (y : Tokens (device := device) #[batch, seq])
    (lr : Float)
    : IO (GPTParams (device := device) modelCfg × torch.Optim.RiemannianTreeSGD.StepDiagnostics × Float) := do
  let step ←
    match pullbackMetric.mode with
    | .exact =>
      torch.Optim.RiemannianTreeSGD.stepCrossEntropy
        params
        (fun p => gpt.forward p x true)
        y
        lr
        trainCfg.gradClip
    | .sampledFisher =>
      torch.Optim.RiemannianTreeSGD.stepCrossEntropySampledFisher
        params
        (fun p => gpt.forward p x true)
        y
        pullbackMetric.fisherProbeCount
        lr
        trainCfg.gradClip
  pure (step.params, step.diagnostics, step.loss)

/-- Train with data of known size (legacy, no validation). -/
def runTraining {n : Nat} (modelCfg : Config) (trainCfg : TrainConfig)
    (pullbackMetric : PullbackMetricConfig := { mode := .sampledFisher, fisherProbeCount := 32 })
    (trainData : Tokens (device := .CPU) #[n]) : IO (GPTParams (device := trainCfg.device) modelCfg) := do
  IO.println ""
  IO.println "Initializing model..."
  let mut params ← GPTParams.init modelCfg trainCfg.device
  IO.println s!"Model initialized with {modelCfg.n_layer} layers"
  IO.println ""

  let mut totalLoss : Float := 0.0
  let mut lastDiagnostics : torch.Optim.RiemannianTreeSGD.StepDiagnostics := {}

  IO.println s!"Starting Riemannian training for {trainCfg.maxIters} iterations on {deviceName trainCfg.device}..."
  IO.println s!"  batch_size={trainCfg.batchSize}, block_size={trainCfg.blockSize}"
  IO.println s!"  lr={trainCfg.learningRate}, warmup={trainCfg.warmupIters}"
  IO.println s!"  grad_clip={trainCfg.gradClip}, pullback={pullbackMetricLabel pullbackMetric}, factor_rank={outputFactorRank modelCfg trainCfg pullbackMetric}"
  (← IO.getStdout).flush

  for iterNum in [:trainCfg.maxIters] do
    let lr := getLr trainCfg iterNum
    let (x, y) ← getBatch trainData trainCfg.batchSize trainCfg.blockSize trainCfg.device
    let (params', diagnostics, lossVal) ← riemannianTrainStep trainCfg pullbackMetric params x y lr
    params := params'
    totalLoss := totalLoss + lossVal
    lastDiagnostics := diagnostics

    if iterNum % trainCfg.logInterval == 0 && iterNum > 0 then
      let avgLoss := totalLoss / trainCfg.logInterval.toFloat
      let liveTensors ← get_live_tensors
      IO.println <|
        s!"iter {iterNum}: loss={lossVal}, avg_loss={avgLoss}, lr={lr}, " ++
        s!"factor_rank={diagnostics.factorRank}, grad_norm={diagnostics.gradientNorm}, " ++
        s!"update_norm={diagnostics.updateNorm}, live_tensors={liveTensors}"
      (← IO.getStdout).flush
      totalLoss := 0.0

  IO.println s!"Training complete! Final factor_rank={lastDiagnostics.factorRank}"
  return params

/-- Train with validation data and checkpointing. -/
def runTrainingWithVal {nTrain nVal : Nat} (modelCfg : Config) (trainCfg : TrainConfig)
    (pullbackMetric : PullbackMetricConfig := { mode := .sampledFisher, fisherProbeCount := 32 })
    (trainData : Tokens (device := .CPU) #[nTrain]) (valData : Tokens (device := .CPU) #[nVal])
    (checkpointDir : Option String := none) : IO (GPTParams (device := trainCfg.device) modelCfg) := do
  IO.println ""
  IO.println "Initializing model..."
  let mut params ← GPTParams.init modelCfg trainCfg.device
  let mut totalLoss : Float := 0.0
  let mut bestValLoss : Float := 1e10
  let mut lastDiagnostics : torch.Optim.RiemannianTreeSGD.StepDiagnostics := {}

  IO.println s!"Model initialized with {modelCfg.n_layer} layers"
  IO.println ""
  IO.println s!"Starting Riemannian training for {trainCfg.maxIters} iterations on {deviceName trainCfg.device}..."
  IO.println s!"  batch_size={trainCfg.batchSize}, block_size={trainCfg.blockSize}"
  IO.println s!"  lr={trainCfg.learningRate}, warmup={trainCfg.warmupIters}"
  IO.println s!"  grad_clip={trainCfg.gradClip}, eval_interval={trainCfg.evalInterval}"
  IO.println s!"  pullback={pullbackMetricLabel pullbackMetric}, factor_rank={outputFactorRank modelCfg trainCfg pullbackMetric}"
  (← IO.getStdout).flush

  for iterNum in [:trainCfg.maxIters] do
    let lr := getLr trainCfg iterNum
    let (x, y) ← getBatch trainData trainCfg.batchSize trainCfg.blockSize trainCfg.device
    let (params', diagnostics, lossVal) ← riemannianTrainStep trainCfg pullbackMetric params x y lr
    params := params'
    totalLoss := totalLoss + lossVal
    lastDiagnostics := diagnostics

    if iterNum % trainCfg.logInterval == 0 && iterNum > 0 then
      let avgLoss := totalLoss / trainCfg.logInterval.toFloat
      let liveTensors ← get_live_tensors
      IO.println <|
        s!"iter {iterNum}: loss={lossVal}, avg_loss={avgLoss}, lr={lr}, " ++
        s!"factor_rank={diagnostics.factorRank}, grad_norm={diagnostics.gradientNorm}, " ++
        s!"update_norm={diagnostics.updateNorm}, live_tensors={liveTensors}"
      (← IO.getStdout).flush
      totalLoss := 0.0

    if iterNum % trainCfg.evalInterval == 0 && iterNum > 0 then
      let valLoss ← evalLoss params valData trainCfg.batchSize trainCfg.blockSize 10
      let valPpl := perplexity valLoss
      IO.println s!"  val_loss={valLoss}, val_ppl={valPpl}"
      if valLoss < bestValLoss then
        bestValLoss := valLoss
        IO.println "  [new best val_loss!]"
      (← IO.getStdout).flush

  let finalValLoss ← evalLoss params valData trainCfg.batchSize trainCfg.blockSize 10
  let finalValPpl := perplexity finalValLoss
  if finalValLoss < bestValLoss then
    bestValLoss := finalValLoss
  IO.println s!"Training complete! Final val_loss={finalValLoss}, val_ppl={finalValPpl}"
  IO.println s!"Best val_loss={bestValLoss}"
  IO.println s!"Final factor_rank={lastDiagnostics.factorRank}"

  if let some dir := checkpointDir then
    saveCheckpoint params trainCfg.maxIters bestValLoss finalValLoss dir

  return params

/-- Script-style entrypoint mirroring `Examples.TrainGPT.main`, but using
    the exact-VJP Riemannian optimizer path. -/
def main : IO Unit := do
  IO.println "Starting..."
  IO.println "=== Tyr Riemannian GPT Training ==="
  IO.println ""
  (← IO.getStdout).flush

  let device ← resolveDeviceFromEnv
  IO.println s!"Using device: {deviceName device}"
  IO.println ""

  let (modelCfg, trainCfg, pullbackMetric, checkpointDir) ← mainConfig device
  if ← envFlagEnabled "TYR_RIEMANNIAN_SMOKE" then
    IO.println "Smoke mode enabled via TYR_RIEMANNIAN_SMOKE."
  let checkpointDir := checkpointDir.getD "checkpoints/riemannian_gpt"
  IO.println s!"Model config: vocab={modelCfg.vocab_size}, block={modelCfg.block_size}, embd={modelCfg.n_embd}, heads={modelCfg.n_head}, layers={modelCfg.n_layer}, dropout={modelCfg.dropout}"
  IO.println s!"Pullback metric: {pullbackMetricLabel pullbackMetric}"

  IO.println ""
  IO.println "Loading data..."
  (← IO.getStdout).flush

  let trainFileExists ← do
    try
      let _ ← data.binFileTokenCount "data/shakespeare_char/train.bin"
      pure true
    catch _ =>
      pure false

  let valFileExists ← do
    try
      let _ ← data.binFileTokenCount "data/shakespeare_char/val.bin"
      pure true
    catch _ =>
      pure false

  let trainedParams ← if trainFileExists then
    IO.println "Found Shakespeare data..."
    let nTrain := (← data.binFileTokenCount "data/shakespeare_char/train.bin").toNat
    IO.println s!"Training set: {nTrain} tokens"
    let trainData ← Tensor.loadU16Bin nTrain "data/shakespeare_char/train.bin"
    if valFileExists then
      let nVal := (← data.binFileTokenCount "data/shakespeare_char/val.bin").toNat
      IO.println s!"Validation set: {nVal} tokens"
      let valData ← Tensor.loadU16Bin nVal "data/shakespeare_char/val.bin"
      IO.println "Training with validation..."
      runTrainingWithVal modelCfg trainCfg pullbackMetric trainData valData (some checkpointDir)
    else
      IO.println "No validation data found, training without validation..."
      runTraining modelCfg trainCfg pullbackMetric trainData
  else
    IO.println "Shakespeare data not found, using random tokens for testing..."
    let numTokens : Nat := 10000
    let tokens ← Tensor.randint 0 modelCfg.vocab_size.toInt64 #[numTokens]
    runTraining modelCfg trainCfg pullbackMetric tokens

  IO.println ""
  IO.println "=== Generating Shakespeare ==="
  IO.println ""

  let prompt := "ROMEO:"
  let promptTokens := encode prompt
  IO.println s!"Prompt: \"{prompt}\""

  let generated ← generate trainedParams promptTokens 200 1.0
  let text := decode generated

  IO.println "Generated text:"
  IO.println "---"
  IO.println text
  IO.println "---"
  IO.println ""
  IO.println "Done!"

end Examples.GPT.RiemannianNanoGPT
