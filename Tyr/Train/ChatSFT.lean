/-
  Tyr/Train/ChatSFT.lean

  Supervised Fine-Tuning for chat models.

  Based on nanochat's chat_sft.py:
  - Task mixture training with masked loss
  - Linear LR decay schedule
  - Evaluation of validation loss and chat metrics

  Divergence from nanochat: `trainLoop` currently consumes only
  `embeddingLr` (scaled by `initLrFrac` and the linear decay multiplier)
  via a single AdamW optimizer applied to all parameters. The `matrixLr`,
  `unembeddingLr`, `weightDecay`, and `gradClip` config fields are parsed
  but unused: there is no Muon/Adam dual-optimizer split, no configurable
  weight decay, and no gradient clipping yet.
-/
import Tyr.Torch
import Tyr.Typed
import Tyr.TensorStruct
import Tyr.Module
import Tyr.Optim
import Tyr.Distributed
import Tyr.Data.Task
import Tyr.Train.RunLedger

namespace torch.Train.ChatSFT

open torch

variable {device : Device}
open torch.Data.Task
open torch.dist

/-! ## Configuration -/

/-- Chat SFT training configuration.
    Based on nanochat's chat_sft.py CLI arguments. -/
structure SFTConfig where
  /-- Number of training epochs -/
  numEpochs : Nat := 1
  /-- Override number of iterations (-1 = use numEpochs) -/
  numIterations : Int := -1
  /-- Per-device batch size -/
  deviceBatchSize : Nat := 4
  /-- Target examples per optimization step (for gradient accumulation) -/
  targetExamplesPerStep : Nat := 32
  /-- Learning rate for embedding parameters (Adam) -/
  embeddingLr : Float := 0.2
  /-- Learning rate for unembedding/LM head parameters (Adam) -/
  unembeddingLr : Float := 0.004
  /-- Learning rate for matrix parameters (Muon) -/
  matrixLr : Float := 0.02
  /-- Weight decay for Adam parameters -/
  weightDecay : Float := 0.0
  /-- Initial LR as fraction of base LR (for warmup) -/
  initLrFrac : Float := 0.02
  /-- Maximum sequence length -/
  maxSeqLen : Nat := 2048
  /-- Evaluate validation loss every N steps -/
  evalEvery : Nat := 100
  /-- Number of batches for validation loss evaluation -/
  evalSteps : Nat := 100
  /-- Evaluate chat metrics every N steps -/
  evalMetricsEvery : Nat := 200
  /-- Device to train on -/
  device : Device := Device.CPU
  /-- Gradient clipping (0 = disabled) -/
  gradClip : Float := 1.0
  /-- Logging interval (steps) -/
  logInterval : Nat := 10
  deriving Repr, Inhabited

/-- Compute gradient accumulation steps from config -/
def SFTConfig.gradAccumSteps (cfg : SFTConfig) (worldSize : Nat := 1) : Nat :=
  let examplesPerStep := cfg.deviceBatchSize * worldSize
  cfg.targetExamplesPerStep / examplesPerStep

/-! ## Learning Rate Schedule -/

/-- Linear LR decay schedule (nanochat style).
    Decays from 1.0 to 0.0 over num_iterations. -/
def linearLRMultiplier (step numIterations : Nat) : Float :=
  if numIterations == 0 then 1.0
  else 1.0 - step.toFloat / numIterations.toFloat

/-! ## Training State -/

/-- SFT training state tracking -/
structure SFTState where
  /-- Current step number -/
  step : Nat := 0
  /-- Current epoch number -/
  epoch : Nat := 0
  /-- Best validation loss seen -/
  bestValLoss : Float := 1e10
  /-- Running training loss for logging -/
  runningLoss : Float := 0.0
  /-- Number of tokens trained on -/
  totalTokens : Nat := 0
  deriving Repr, Inhabited

/-! ## SFT Batch -/

/-- SFT batch with inputs, targets, and mask.

    The batch and sequence dimensions are runtime values carried as fields;
    the tensors are indexed by them, so `inputs`, `targets` and `mask` always
    agree on shape. -/
structure SFTBatch where
  /-- Batch size -/
  batchSize : Nat
  /-- Sequence length -/
  seqLen : Nat
  /-- Input token IDs: [batchSize, seqLen] as Int64 tensor -/
  inputs : Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := device }
  /-- Target token IDs: [batchSize, seqLen] as Int64 tensor -/
  targets : Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := device }
  /-- Training mask: [batchSize, seqLen] as Float32 tensor (1.0 = train, 0.0 = ignore) -/
  mask : Tensor { shape := #[batchSize, seqLen], dtype := .Float32, device := device }
  /-- Number of valid (non-masked) target tokens -/
  numValidTokens : Nat
  deriving Repr

/-- Create an empty SFT batch -/
def SFTBatch.empty : SFTBatch (device := .CPU) :=
  { batchSize := 0
  , seqLen := 0
  , inputs := Tensor.fullInt #[0, 0] 0
  , targets := Tensor.fullInt #[0, 0] 0
  , mask := Tensor.zeros #[0, 0]
  , numValidTokens := 0 }

/-- Move all tensor fields of an SFT batch to a target device. -/
private def moveBatchToDevice {source : Device} (batch : SFTBatch (device := source))
    (device : Device) : SFTBatch (device := device) :=
  { batch with
    inputs := batch.inputs.to device
    targets := batch.targets.to device
    mask := batch.mask.to device
  }

/-- Convert a collated conversation batch into the SFT batch format. -/
def SFTBatch.ofConversationBatch (batch : ConversationBatch) : SFTBatch (device := .CPU) :=
  let numValidFloat := Tensor.item (Tensor.sumAll batch.mask)
  let numValidTokens := if numValidFloat <= 0 then 0 else numValidFloat.toUInt64.toNat
  {
    batchSize := batch.batchSize
    seqLen := batch.seqLen
    inputs := batch.tokens
    targets := batch.tokens
    mask := batch.mask
    numValidTokens := numValidTokens
  }

/-! ## Masked Loss Computation -/

/-- Compute cross-entropy loss with mask.
    Only positions where mask > 0 contribute to the loss.

    logits: [batch, seq, vocab] - Model output logits
    targets: [batch, seq] - Target token IDs (Int64)
    mask: [batch, seq] - Training mask (1.0 = train, 0.0 = ignore)

    Returns scalar loss averaged over valid positions. The result dtype is the
    promoted sum-accumulation dtype of logits and mask (Float32 when both are
    Float32).
-/
def maskedCrossEntropy? {batch seq vocab : Nat} {d m result : DType} {dev : Device}
    [DType.Promotes d m result]
    (logits : Tensor ⟨#[batch, seq, vocab], d, dev⟩)
    (targets : Tensor ⟨#[batch, seq], .Int64, dev⟩)
    (mask : Tensor ⟨#[batch, seq], m, dev⟩)
    : Except String (Tensor ⟨#[], (DType.promote d m).sumResult, dev⟩) := do
  if !(mask.raw.dtype.isBool || mask.raw.dtype.isIntegral || mask.raw.dtype.isRealFloating) then
    throw "maskedCrossEntropy requires a real numeric or Bool mask"
  -- Flatten for cross-entropy
  let batchSeq := batch * seq
  let logitsFlat ← logits.reshapeChecked? #[batchSeq, vocab]
  let targetsFlat ← targets.reshapeChecked? #[batchSeq]
  -- Compute per-element loss
  let lossesFlat ← Tensor.crossEntropyNone? logitsFlat targetsFlat
  -- Apply mask and compute mean over valid positions
  let maskFlat ← mask.reshapeChecked? #[batchSeq]
  Tensor.checkPointwise lossesFlat maskFlat
  let maskedLosses : Tensor ⟨#[batchSeq], DType.promote d m, dev⟩ ←
    Tensor.ofRaw (← RawTensor.mulChecked lossesFlat.raw maskFlat.raw)
  -- Sum losses and divide by number of valid tokens
  let totalLoss ← Tensor.sumAll? maskedLosses
  let numValid ← Tensor.sumAll? maskFlat
  -- Divide by number of valid tokens to get mean loss
  -- Use scalar multiplication with the reciprocal to avoid tensor division
  let numValidFloat ← Tensor.item? numValid
  let epsFloat : Float := 1e-8
  let floating ← DType.realFloating? (DType.promote d m).sumResult
  let _ := floating.down
  Tensor.mulScalar? totalLoss (1.0 / (numValidFloat + epsFloat))

/-- Pure masked loss over real floating logits. Use `maskedCrossEntropy?` to
recover invalid target labels or unsupported native kernels. -/
def maskedCrossEntropy {batch seq vocab : Nat} {d m result : DType} {dev : Device}
    [DType.RealFloating d] [DType.Promotes d m result]
    (logits : Tensor ⟨#[batch, seq, vocab], d, dev⟩)
    (targets : Tensor ⟨#[batch, seq], .Int64, dev⟩)
    (mask : Tensor ⟨#[batch, seq], m, dev⟩)
    (_maskDomain : (m.isBool || m.isIntegral || m.isRealFloating) = true := by first | rfl | decide) :
    Tensor ⟨#[], (DType.promote d m).sumResult, dev⟩ :=
  match maskedCrossEntropy? logits targets mask with
  | .ok loss => loss
  | .error message => nativeInvariantFailure s!"maskedCrossEntropy: {message}"

/-! ## Data Preparation -/

/-- Build an Int64 token tensor from host data over the raw FFI boundary.
    `reshapeChecked` validates the result metadata (shape and Int64 dtype). -/
private def int64TensorFromArray (arr : Array Int64) : Tensor { shape := #[arr.size], dtype := .Int64, device := .CPU } :=
  Tensor.fromInt64Array arr

/-- Prepare SFT batch from tokenized conversations.
    Creates input/target pairs where targets are shifted and masked.

    The mask ensures we only train on assistant responses.
    - Input: tokens[:-1]
    - Target: tokens[1:]
    - Mask: mask[1:] (shifted to align with targets)
-/
def prepareSFTBatch (convs : Array TokenizedConversation) (maxLen : Nat) (padTokenId : UInt64)
    : IO (SFTBatch (device := .CPU)) := do
  let batchSize := convs.size
  if batchSize == 0 then
    return SFTBatch.empty

  -- Find max length (minus 1 for shift, capped at maxLen)
  let seqLen := convs.foldl (fun acc c => max acc (min (c.tokens.size - 1) maxLen)) 0

  if seqLen == 0 then
    return SFTBatch.empty

  -- Build input, target, and mask arrays
  let mut allInputs : Array Int64 := #[]
  let mut allTargets : Array Int64 := #[]
  let mut allMasks : Array Float := #[]
  let mut numValidTokens : Nat := 0

  for conv in convs do
    let n := conv.tokens.size
    -- Input: tokens[:-1], Target: tokens[1:], Mask: mask[1:]
    for i in [:seqLen] do
      if i < n - 1 then
        -- Valid position
        allInputs := allInputs.push conv.tokens[i]!.toInt64
        allTargets := allTargets.push conv.tokens[i + 1]!.toInt64
        -- Mask for target position (i+1)
        let maskVal : Float := if i + 1 < conv.mask.size && conv.mask[i + 1]! == 1 then 1.0 else 0.0
        allMasks := allMasks.push maskVal
        if maskVal > 0 then
          numValidTokens := numValidTokens + 1
      else
        -- Padding position
        allInputs := allInputs.push padTokenId.toInt64
        allTargets := allTargets.push padTokenId.toInt64  -- Target doesn't matter, mask is 0
        allMasks := allMasks.push 0.0  -- Don't train on padding

  -- Convert to tensors
  let inputsTensor := (int64TensorFromArray allInputs).reshapeChecked
    #[batchSize, seqLen]
  let targetsTensor := (int64TensorFromArray allTargets).reshapeChecked
    #[batchSize, seqLen]
  let maskTensor := (Tensor.fromFloatArray allMasks).reshapeChecked
    #[batchSize, seqLen]

  return {
    batchSize := batchSize
    seqLen := seqLen
    inputs := inputsTensor
    targets := targetsTensor
    mask := maskTensor
    numValidTokens
  }

/-! ## Training Functions -/

/-- Single forward pass for SFT with masked loss.
    Returns loss tensor for backprop.

    forwardFn should take inputs [batch, seq] and return logits [batch, seq, vocab]
-/
def sftForward {batch seq vocab : Nat} {d m result : DType} {dev : Device}
    [DType.Promotes d m result]
    (forwardFn : Tensor ⟨#[batch, seq], .Int64, dev⟩ → IO (Tensor ⟨#[batch, seq, vocab], d, dev⟩))
    (inputs : Tensor ⟨#[batch, seq], .Int64, dev⟩)
    (targets : Tensor ⟨#[batch, seq], .Int64, dev⟩)
    (mask : Tensor ⟨#[batch, seq], m, dev⟩)
    : IO (Tensor ⟨#[], (DType.promote d m).sumResult, dev⟩) := do
  -- Get logits from model
  let logits ← forwardFn inputs
  -- Compute masked loss
  IO.ofExcept (maskedCrossEntropy? logits targets mask)

/-- Evaluate validation loss over multiple batches.

    The loss of each batch is the masked mean per-token cross entropy; batches
    are weighted by their valid-token counts so the result is the global
    masked mean over all evaluated tokens. -/
def evalValidationLoss [TensorStruct Params]
    (params : Params)
    (dataFn : IO (SFTBatch (device := .CPU)))
    (forwardFn : Params → {b s : Nat} → Tensor { shape := #[b, s], dtype := .Int64, device := device } →
      IO (Σ v : Nat, Tensor { shape := #[b, s, v], dtype := .Float32, device := device }))
    (numSteps : Nat)
    (targetDevice : Device := device)
    : IO Float := do
  if targetDevice != device then
    throw (IO.userError "Validation device differs from the model forward device")
  let params := TensorStruct.detach params
  let mut totalLoss : Float := 0.0
  let mut totalValid : Nat := 0

  for _ in [:numSteps] do
    let batch ← dataFn
    let batch := moveBatchToDevice batch device
    if batch.numValidTokens > 0 then
      let ⟨_, logits⟩ ← forwardFn params batch.inputs
      let loss ← IO.ofExcept (maskedCrossEntropy? logits batch.targets batch.mask)
      totalLoss := totalLoss + Tensor.item loss * batch.numValidTokens.toFloat
      totalValid := totalValid + batch.numValidTokens

  if totalValid == 0 then
    return 0.0
  return totalLoss / totalValid.toFloat

/-- Logging callback type -/
def LogCallback := Nat → Float → Float → Nat → IO Unit

/-- Silent training-step callback. -/
def silentLogger : LogCallback := fun _ _ _ _ => pure ()

/-- Validation logging callback type. -/
def ValidationCallback := Nat → Float → IO Unit

/-- Silent validation callback. -/
def silentValidationLogger : ValidationCallback := fun _ _ => pure ()

/-- Render a human-readable training-step update. -/
def formatTrainUpdate (step : Nat) (trainLoss lrm : Float) (numTokens : Nat) : String :=
  s!"Step {step}: loss={trainLoss}, lrm={lrm}, tokens={numTokens}"

/-- Render a human-readable validation update. -/
def formatValidationUpdate (step : Nat) (valLoss : Float) : String :=
  s!"Step {step}: val_loss={valLoss}"

/-- Hook set for SFT training progress and evaluation. -/
structure Callbacks where
  onTrainStep : LogCallback := silentLogger
  onValidation : ValidationCallback := silentValidationLogger

/-- Compose two callback sets. -/
def Callbacks.combine (lhs rhs : Callbacks) : Callbacks := {
  onTrainStep := fun step trainLoss lrm numTokens => do
    lhs.onTrainStep step trainLoss lrm numTokens
    rhs.onTrainStep step trainLoss lrm numTokens
  onValidation := fun step valLoss => do
    lhs.onValidation step valLoss
    rhs.onValidation step valLoss
}

/-- Emit SFT metrics to the run ledger. -/
def artifactCallbacks
    (artifacts : RunLedger.RunArtifacts)
    (scopePrefix : String := "sft") : Callbacks := {
  onTrainStep := fun step trainLoss lrm numTokens => do
    RunLedger.appendMetricEvent artifacts {
      scope := s!"{scopePrefix}/train"
      step := some step
      metrics := [
        RunLedger.metricFloat "loss" trainLoss,
        RunLedger.metricFloat "lr_multiplier" lrm,
        RunLedger.metricNat "num_tokens" numTokens
      ]
    }
  onValidation := fun step valLoss => do
    RunLedger.appendMetricEvent artifacts {
      scope := s!"{scopePrefix}/eval"
      step := some step
      metrics := [RunLedger.metricFloat "val_loss" valLoss]
    }
}

/-! ## Main Training Loop -/

/-- Run SFT training loop.

    This implements the nanochat chat_sft.py training algorithm:
    1. Linear LR decay schedule
    2. Gradient accumulation
    3. Masked loss on assistant responses only
    4. Periodic validation evaluation

    Parameters:
    - cfg: Training configuration
    - params: Model parameters to train (must implement TensorStruct)
    - optState: Initial optimizer state
    - forwardFn: Forward pass function that takes params and Int64 inputs
      `[batch, seq]`, returns Float32 logits `[batch, seq, vocab]`
    - lossFn: Loss function that takes logits, targets, mask, returns scalar loss
    - trainDataFn: Function to get next training batch
    - valDataFn: Function to get next validation batch (optional)
    - numTrainExamples: Total training examples (for computing iterations)
    - logger: Logging callback
-/
def trainLoop [TensorStruct P] [FloatingTensorStruct P]
    (cfg : SFTConfig)
    (params : P)
    (optState : Optim.AdamWState P)
    (forwardFn : P → {b s : Nat} → Tensor { shape := #[b, s], dtype := .Int64, device := device } →
      IO (Σ v : Nat, Tensor { shape := #[b, s, v], dtype := .Float32, device := device }))  -- params → inputs → logits
    (lossFn : {b s v : Nat} → Tensor { shape := #[b, s, v], dtype := .Float32, device := device } → Tensor { shape := #[b, s], dtype := .Int64, device := device } →
      Tensor { shape := #[b, s], dtype := .Float32, device := device } → Tensor { shape := #[], dtype := .Float32, device := device })  -- logits → targets → mask → loss
    (trainDataFn : IO (SFTBatch (device := .CPU)))
    (valDataFn : Option (IO (SFTBatch (device := .CPU))) := none)
    (numTrainExamples : Nat := 10000)
    (callbacks : Callbacks := {})
    : IO (P × Optim.AdamWState P × SFTState) := do
  if cfg.device != device then
    throw (IO.userError "Training device differs from the model forward device")
  let isDistributed ← dist.isInitialized
  let worldSizeU64 ← if isDistributed then dist.getWorldSize else pure 1
  let worldSize := worldSizeU64.toNat

  -- Compute number of iterations
  let numIterations : Nat :=
    if cfg.numIterations >= 0 then
      cfg.numIterations.toNat
    else
      (numTrainExamples / cfg.targetExamplesPerStep) * cfg.numEpochs

  let gradAccumSteps := max 1 (cfg.gradAccumSteps worldSize)

  let mut currentParams ← TensorStruct.makeLeafParams params
  let mut currentOptState := optState
  let mut state : SFTState := {}

  for step in [:numIterations] do
    let isLastStep := step == numIterations - 1
    state := { state with step }

    -- Validation evaluation
    if let some valFn := valDataFn then
      if isLastStep || (step > 0 && step % cfg.evalEvery == 0) then
        let valLoss ← evalValidationLoss currentParams valFn forwardFn cfg.evalSteps cfg.device
        callbacks.onValidation step valLoss
        if valLoss < state.bestValLoss then
          state := { state with bestValLoss := valLoss }

    if isLastStep then
      break

    -- Zero gradients at start of accumulation
    let workingParams ← TensorStruct.makeLeafParams currentParams
    TensorStruct.zeroGrads workingParams

    -- Gradient accumulation loop
    let mut accumLoss : Float := 0.0
    let mut numTokens : Nat := 0

    for _ in [:gradAccumSteps] do
      let batch ← trainDataFn
      let batch := moveBatchToDevice batch device

      if batch.numValidTokens > 0 then
        -- Forward pass
        let ⟨_, logits⟩ ← forwardFn workingParams batch.inputs
        -- Compute masked loss
        let loss := lossFn logits batch.targets batch.mask
        -- Scale loss for accumulation
        let scaledLoss := Tensor.mulScalar loss (1.0 / gradAccumSteps.toFloat)
        -- Backward pass (gradients accumulate)
        Tensor.backwardLoss scaledLoss

        accumLoss := accumLoss + Tensor.item loss
        numTokens := numTokens + batch.numValidTokens

    -- LR schedule: linear decay with initial warmup fraction
    let lrm := linearLRMultiplier step numIterations
    let lr := cfg.embeddingLr * cfg.initLrFrac * lrm

    -- Synchronize gradients across ranks in distributed training.
    let grads ← TensorStruct.grads workingParams
    let grads ← if isDistributed then dist.allReduceGrads grads .avg else pure grads

    -- Update parameters from the synchronized gradient snapshots.
    let opt := Optim.adamw (lr := lr) (b1 := 0.9) (b2 := 0.999)
    let (newParams, newOptState) ← Optim.step opt workingParams grads currentOptState

    currentParams := newParams
    currentOptState := newOptState

    state := { state with
      totalTokens := state.totalTokens + numTokens
      runningLoss := state.runningLoss + accumLoss / gradAccumSteps.toFloat
    }

    -- Logging
    if step % cfg.logInterval == 0 then
      let avgLoss := if gradAccumSteps > 0
        then accumLoss / gradAccumSteps.toFloat
        else 0.0
      callbacks.onTrainStep step avgLoss lrm numTokens

  return (currentParams, currentOptState, state)

/-! ## Convenience Functions -/

/-- Create a training data generator from a TaskIterator.
    Yields SFTBatch on each call, handling epoch boundaries automatically. -/
def makeTaskDataGenerator
    (iter : TaskIterator)
    (_maxLen : Nat)
    (_padTokenId : UInt64)
    : IO (IO (SFTBatch (device := .CPU))) := do
  let iterRef ← IO.mkRef iter
  return do
    let mut currentIter ← iterRef.get

    -- Get batch of tokenized conversations
    let (maybeBatch, newIter) ← currentIter.nextBatch
    currentIter := newIter

    -- Handle epoch boundary
    let batch ← match maybeBatch with
      | none =>
        -- Epoch ended, get first batch of new epoch
        let (maybeBatch', newerIter) ← currentIter.nextBatch
        currentIter := newerIter
        match maybeBatch' with
        | none => pure SFTBatch.empty
        | some convBatch => pure (SFTBatch.ofConversationBatch convBatch)
      | some convBatch =>
        pure (SFTBatch.ofConversationBatch convBatch)

    iterRef.set currentIter
    return batch

/-- Simple SFT training helper for models with standard interface.
    Wraps trainLoop with common defaults.

    `forwardFn` takes Int64 token inputs `[batch, seq]` and returns Float32
    logits `[batch, seq, vocab]`; `lossFn` computes the scalar masked loss
    (`maskedCrossEntropy` fits the signature directly). -/
def trainSFT [TensorStruct P] [FloatingTensorStruct P]
    (cfg : SFTConfig)
    (params : P)
    (forwardFn : P → {b s : Nat} → Tensor { shape := #[b, s], dtype := .Int64, device := device } →
      IO (Σ v : Nat, Tensor { shape := #[b, s, v], dtype := .Float32, device := device }))
    (lossFn : {b s v : Nat} → Tensor { shape := #[b, s, v], dtype := .Float32, device := device } → Tensor { shape := #[b, s], dtype := .Int64, device := device } →
      Tensor { shape := #[b, s], dtype := .Float32, device := device } → Tensor { shape := #[], dtype := .Float32, device := device })
    (trainDataFn : IO (SFTBatch (device := .CPU)))
    (numTrainExamples : Nat)
    : IO (P × SFTState) := do
  -- Initialize optimizer state via the adamw optimizer
  let opt := Optim.adamw (lr := cfg.embeddingLr) (b1 := 0.9) (b2 := 0.999)
  let optState := opt.init params

  let (finalParams, _, finalState) ← trainLoop cfg params optState forwardFn lossFn trainDataFn none numTrainExamples

  return (finalParams, finalState)

end torch.Train.ChatSFT
