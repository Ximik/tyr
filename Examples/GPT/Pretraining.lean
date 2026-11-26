/-
  Tyr/Examples/GPT/Pretraining.lean

  GPT pretraining on an in-memory token stream: random contiguous batches,
  AdamW with linear warmup and cosine decay, and the running loss reported at
  each evaluation interval when validation data is given.
-/
import Tyr.Core.Torch
import Tyr.Lib.Optim.Scheduler
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
  /-- Log interval -/
  logInterval : Nat := 10
  /-- Learning rate -/
  learningRate : Float := 3e-4
  deriving Repr

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
  deriving Repr

/-! ## Training Loop -/

/-- Run single training step -/
def trainStep (cfg : torch.gpt.Config)
    (params : GPTParams (device := device) cfg)
    (opt : GradientTransformation (GPTParams (device := device) cfg) (AdamWScheduleState (GPTParams (device := device) cfg)))
    (optState : AdamWScheduleState (GPTParams (device := device) cfg))
    (x : Tokens (device := device) #[batchSize, seqLen])
    (y : Tokens (device := device) #[batchSize, seqLen])
    : IO (GPTParams (device := device) cfg × AdamWScheduleState (GPTParams (device := device) cfg) × Float) := do
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

  let schedule := Scheduler.cosine_decay cfg.learningRate cfg.totalSteps
    (warmupSteps := cfg.warmupSteps)
  let opt := adamw_schedule schedule
  let optState := opt.init params

  let mut currentParams := params
  let mut currentOptState := optState
  let mut state : TrainState := {}

  IO.println s!"Starting training for {cfg.totalSteps} steps..."

  for stepNum in [:cfg.totalSteps] do
    -- Sample random batch from training data
    let batchTokens := cfg.deviceBatchSize * cfg.seqLen + 1
    if batchTokens > numTokens || cfg.deviceBatchSize == 0 || cfg.seqLen == 0 then
      throw <| IO.userError "GPT pretraining requires positive dimensions and enough tokens"
    let maxStart := numTokens - batchTokens
    let startTensor ← Tensor.randint 0 (maxStart + 1).toInt64 #[]
    let startIdx := startTensor.itemInt.toInt.toNat

    -- Get batch: x = tokens[start:start+seq], y = tokens[start+1:start+seq+1]
    let batchLen := cfg.deviceBatchSize * cfg.seqLen
    let ⟨hStart⟩ ← if h : startIdx + batchLen + 1 ≤ numTokens then pure (PLift.up h) else
      throw <| IO.userError s!"GPT pretraining batch start {startIdx} exceeds {maxStart}"
    let xFlat : Tokens (device := .CPU) #[batchLen] := trainData.slice 0 startIdx batchLen
    let yFlat : Tokens (device := .CPU) #[batchLen] := trainData.slice 0 (startIdx + 1) batchLen

    let x := (Tensor.reshape xFlat #[cfg.deviceBatchSize, cfg.seqLen]).to device
    let y := (Tensor.reshape yFlat #[cfg.deviceBatchSize, cfg.seqLen]).to device

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
      IO.println s!"Step {stepNum}/{cfg.totalSteps}: loss={loss} lr={schedule stepNum}"

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

end Examples.GPT.Pretraining
