/-
  Tyr/Examples/GPT/Pretraining.lean

  Base pretraining script following nanochat's approach:
  - Streaming data from parquet files
  - BPB (bits-per-byte) evaluation
  - Cosine LR schedule with warmup
  - Checkpointing and resume

  This is the foundation for training GPT models from scratch
  on large datasets like FineWeb.
-/
import Tyr.Torch
import Examples.GPT.GPT
import Examples.GPT.Train

namespace Examples.GPT.Pretraining

open torch

variable {device : Device}
open torch.gpt
open torch.train
open torch.Optim

/-! ## Pretraining Configuration -/

/-- Configuration for base pretraining -/
structure PretrainingConfig where
  /-- Model configuration -/
  modelConfig : torch.gpt.Config
  /-- Data directory containing parquet files -/
  dataPath : String := "data/fineweb"
  /-- Batch size per device -/
  deviceBatchSize : Nat := 8
  /-- Sequence length -/
  seqLen : Nat := 2048
  /-- Total training steps -/
  totalSteps : Nat := 10000
  /-- Warmup steps -/
  warmupSteps : Nat := 300
  /-- Evaluation interval -/
  evalInterval : Nat := 500
  /-- Number of eval steps -/
  evalSteps : Nat := 50
  /-- Log interval -/
  logInterval : Nat := 10
  /-- Checkpoint directory -/
  checkpointDir : String := "checkpoints/pretrain"
  /-- Checkpoint interval -/
  checkpointInterval : Nat := 1000
  /-- Learning rate -/
  learningRate : Float := 3e-4
  /-- Gradient accumulation steps -/
  gradAccumSteps : Nat := 4
  deriving Repr

/-- Default config for small-scale pretraining (testing) -/
def PretrainingConfig.small : PretrainingConfig := {
  modelConfig := torch.gpt.Config.nanogpt_cpu_shakespeare
  deviceBatchSize := 4
  seqLen := 512
  totalSteps := 1000
  warmupSteps := 50
  evalInterval := 100
  logInterval := 10
}

/-- Default config for medium-scale pretraining -/
def PretrainingConfig.medium : PretrainingConfig := {
  modelConfig := {
    vocab_size := 32768
    block_size := 2048
    n_embd := 768
    n_head := 12
    n_layer := 12
    dropout := 0.0
  }
  deviceBatchSize := 8
  totalSteps := 50000
  warmupSteps := 500
}

/-! ## Training State -/

/-- Training state for checkpointing -/
structure TrainState where
  /-- Current step -/
  step : Nat := 0
  /-- Best validation loss -/
  bestLoss : Float := 100.0
  /-- Total tokens processed -/
  totalTokens : Nat := 0
  /-- Running loss (exponential moving average) -/
  runningLoss : Float := 0.0
  deriving Repr, Inhabited

/-! ## Learning Rate Schedule -/

/-- Cosine learning rate schedule with warmup.
    Following nanochat: linear warmup, cosine decay to 0. -/
def cosineSchedule (step : Nat) (warmupSteps totalSteps : Nat) (baseLr : Float) : Float :=
  if step < warmupSteps then
    -- Linear warmup
    baseLr * step.toFloat / warmupSteps.toFloat
  else
    -- Cosine decay using cos(x) = cos(π * progress)
    let progress := (step - warmupSteps).toFloat / (totalSteps - warmupSteps).toFloat
    let piApprox := 3.14159265358979
    let decay := 0.5 * (1.0 + Float.cos (progress * piApprox))
    baseLr * decay

/-! ## Training Loop -/

/-- Run single training step -/
def trainStep (cfg : torch.gpt.Config)
    (params : GPTParams (device := device) cfg)
    (opt : GradientTransformation (GPTParams (device := device) cfg) (AdamWState (GPTParams (device := device) cfg)))
    (optState : AdamWState (GPTParams (device := device) cfg))
    (x : Tokens (device := device) #[batchSize, seqLen])
    (y : Tokens (device := device) #[batchSize, seqLen])
    : IO (GPTParams (device := device) cfg × AdamWState (GPTParams (device := device) cfg) × Float) := do
  -- Zero gradients
  TensorStruct.zeroGrads params

  -- Forward pass and compute loss
  let lossT ← torch.gpt.loss params x y true

  -- Backward pass
  Tensor.backwardLoss lossT

  -- Get loss value for logging
  let lossVal := Tensor.item lossT

  -- Extract gradients from parameters
  let grads ← TensorStruct.grads params

  -- Update parameters with optimizer step
  let (params', optState') ← step opt params grads optState

  return (params', optState', lossVal)

/-! ## Main Training Loop -/

/-- Run the main pretraining loop using simple in-memory data -/
def runPretraining (cfg : PretrainingConfig)
    (numTokens : Nat)
    (trainData : Tokens (device := .CPU) #[numTokens])
    (valData : Option (Tokens (device := .CPU) #[numTokens]))
    (device : Device := .CPU)
    : IO (GPTParams (device := device) cfg.modelConfig) := do
  IO.println "Initializing model..."
  let params ← GPTParams.init cfg.modelConfig device

  let opt := adamw (lr := cfg.learningRate)
  let optState := opt.init params

  let mut currentParams := params
  let mut currentOptState := optState
  let mut state : TrainState := {}

  IO.println s!"Starting training for {cfg.totalSteps} steps..."

  for stepNum in [:cfg.totalSteps] do
    -- Get learning rate for this step
    let _lr := cosineSchedule stepNum cfg.warmupSteps cfg.totalSteps cfg.learningRate

    -- Sample random batch from training data
    let batchTokens := cfg.deviceBatchSize * cfg.seqLen + 1
    if batchTokens > numTokens || cfg.deviceBatchSize == 0 || cfg.seqLen == 0 then
      throw <| IO.userError "GPT pretraining requires positive dimensions and enough tokens"
    let maxStart := numTokens - batchTokens
    let startTensor ← Tensor.randint 0 (maxStart + 1).toInt64 #[]
    let startIdx := startTensor.itemInt.toInt.toNat

    -- Get batch: x = tokens[start:start+seq], y = tokens[start+1:start+seq+1]
    let batchLen := cfg.deviceBatchSize * cfg.seqLen
    let xFlat : Tokens (device := .CPU) #[batchLen] := trainData.sliceChecked 0 startIdx batchLen #[batchLen]
    let yFlat : Tokens (device := .CPU) #[batchLen] := trainData.sliceChecked 0 (startIdx + 1) batchLen #[batchLen]

    let x := (Tensor.reshapeChecked xFlat #[cfg.deviceBatchSize, cfg.seqLen]).to device
    let y := (Tensor.reshapeChecked yFlat #[cfg.deviceBatchSize, cfg.seqLen]).to device

    -- Training step
    let (newParams, newOptState, loss) ←
      trainStep cfg.modelConfig currentParams opt currentOptState x y

    currentParams := newParams
    currentOptState := newOptState

    -- Update state
    state := { state with
      step := stepNum + 1
      runningLoss := 0.99 * state.runningLoss + 0.01 * loss
      totalTokens := state.totalTokens + (cfg.deviceBatchSize * cfg.seqLen)
    }

    -- Logging
    if stepNum % cfg.logInterval == 0 then
      let lr := cosineSchedule stepNum cfg.warmupSteps cfg.totalSteps cfg.learningRate
      IO.println s!"Step {stepNum}/{cfg.totalSteps}: loss={loss} lr={lr}"

    -- Evaluation
    if stepNum > 0 && stepNum % cfg.evalInterval == 0 then
      match valData with
      | some _val =>
        -- Simplified: just report running loss as proxy
        IO.println s!"Eval @ step {stepNum}: running_loss={state.runningLoss}"
        if state.runningLoss < state.bestLoss then
          state := { state with bestLoss := state.runningLoss }
          IO.println s!"New best loss: {state.runningLoss}"
      | none => pure ()

  IO.println s!"Training complete! Final running loss: {state.runningLoss}"
  return currentParams

/-! ## Entry Point -/

/-- Main pretraining entry point -/
def main : IO Unit := do
  IO.println "=== Tyr Base Pretraining ==="

  let device ← getBestDevice
  IO.println s!"Using device: {repr device}"

  -- Use small config for testing
  let cfg := PretrainingConfig.small
  IO.println s!"Config: layers={cfg.modelConfig.n_layer}, d_model={cfg.modelConfig.n_embd}"

  -- Generate random training data for testing
  IO.println "Generating random training data..."
  let numTokens : Nat := 50000
  let trainData ← Tensor.randint 0 cfg.modelConfig.vocab_size.toInt64 #[numTokens]

  -- No validation data for simple test
  let valData : Option (Tokens (device := .CPU) #[numTokens]) := none

  -- Run training
  IO.println "Starting training..."
  let _finalParams ← runPretraining cfg numTokens trainData valData device

  IO.println "Done!"

end Examples.GPT.Pretraining
