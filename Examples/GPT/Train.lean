/-
  Training Infrastructure for GPT
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Optim
import Examples.GPT.GPT

namespace torch.train

open torch

variable {device : Device}
open torch.gpt

/-- Training configuration -/
structure TrainConfig where
  maxIters : Nat := 1000
  evalInterval : Nat := 100
  logInterval : Nat := 10
  learningRate : Float := 1e-3
  minLr : Float := 1e-4
  warmupIters : Nat := 100
  lrDecayIters : Nat := 1000
  gradClip : Float := 1.0
  batchSize : UInt64 := 4
  blockSize : UInt64 := 64
  /-- Number of gradient accumulation steps (effective batch = batchSize * gradAccumSteps) -/
  gradAccumSteps : Nat := 1
  /-- Device to train on (CPU, CUDA, or MPS) -/
  device : Device := Device.CPU
  deriving Repr, Inhabited

/-- Cosine learning rate schedule with warmup -/
def getLr (cfg : TrainConfig) (iterNum : Nat) : Float :=
  if iterNum < cfg.warmupIters then
    -- Linear warmup
    cfg.learningRate * (iterNum.toFloat / cfg.warmupIters.toFloat)
  else if iterNum > cfg.lrDecayIters then
    cfg.minLr
  else
    -- Cosine decay
    let decayRatio := (iterNum - cfg.warmupIters).toFloat /
                      (cfg.lrDecayIters - cfg.warmupIters).toFloat
    let pi : Float := 3.14159265358979323846
    let coeff := 0.5 * (1.0 + Float.cos (pi * decayRatio))
    cfg.minLr + coeff * (cfg.learningRate - cfg.minLr)

/-- Get a random batch of data from token array
    Returns (x, y) where y = x shifted by 1 position
    Uses a random starting position with batchSize consecutive sequences -/
def getBatch (trainData : Tokens (device := .CPU) #[n]) (batchSize blockSize : UInt64) (device : Device := Device.CPU) : IO (Tokens (device := device) #[batchSize, blockSize] × Tokens (device := device) #[batchSize, blockSize]) := do
  let required := batchSize.toNat * (blockSize.toNat + 1)
  if batchSize == 0 || blockSize == 0 || required > n.toNat then
    throw <| IO.userError "GPT batch requires positive dimensions and enough tokens"
  let totalTokens := required.toUInt64
  let maxStart := n - totalTokens
  let startTensor ← Tensor.randint 0 (maxStart + 1).toInt64 #[]
  let start := startTensor.itemInt.toUInt64
  let block := trainData.sliceChecked 0 start totalTokens #[totalTokens]
  let reshaped := block.reshapeChecked #[batchSize, blockSize + 1]
  let x := reshaped.sliceChecked 1 0 blockSize #[batchSize, blockSize]
  let y := reshaped.sliceChecked 1 1 blockSize #[batchSize, blockSize]
  return (x.to device, y.to device)

/-- Clip gradients for a single block's parameters -/
def clipBlockGrads {n_embd : UInt64} (params : BlockParams (device := device) n_embd) (maxNorm : Float) : IO Unit := do
  let _ ← Tensor.clipGradNorm params.ln1_weight maxNorm
  let _ ← Tensor.clipGradNorm params.ln1_bias maxNorm
  let _ ← Tensor.clipGradNorm params.q_proj maxNorm
  let _ ← Tensor.clipGradNorm params.k_proj maxNorm
  let _ ← Tensor.clipGradNorm params.v_proj maxNorm
  let _ ← Tensor.clipGradNorm params.c_proj maxNorm
  let _ ← Tensor.clipGradNorm params.c_proj_bias maxNorm
  let _ ← Tensor.clipGradNorm params.ln2_weight maxNorm
  let _ ← Tensor.clipGradNorm params.ln2_bias maxNorm
  let _ ← Tensor.clipGradNorm params.mlp_fc maxNorm
  let _ ← Tensor.clipGradNorm params.mlp_fc_bias maxNorm
  let _ ← Tensor.clipGradNorm params.mlp_proj maxNorm
  let _ ← Tensor.clipGradNorm params.mlp_proj_bias maxNorm

/-- Clip gradients for all GPT parameters -/
def clipGPTGrads {cfg : Config} (params : GPTParams (device := device) cfg) (maxNorm : Float) : IO Unit := do
  let _ ← Tensor.clipGradNorm params.wte maxNorm
  let _ ← Tensor.clipGradNorm params.wpe maxNorm
  for block in params.blocks do
    clipBlockGrads block maxNorm
  let _ ← Tensor.clipGradNorm params.ln_f_weight maxNorm
  let _ ← Tensor.clipGradNorm params.ln_f_bias maxNorm

/-- Single training step using Optax-style optimizer -/
def trainStep {modelCfg : Config} {batch seq : UInt64}
    (trainCfg : TrainConfig)
    (params : GPTParams (device := device) modelCfg)
    (optState : Optim.AdamWState (GPTParams (device := device) modelCfg))
    (x : Tokens (device := device) #[batch, seq])
    (y : Tokens (device := device) #[batch, seq])
    (lr : Float)
    : IO (GPTParams (device := device) modelCfg × Optim.AdamWState (GPTParams (device := device) modelCfg) × Float) := do
  -- Zero gradients BEFORE forward/backward to prevent gradient accumulation
  TensorStruct.zeroGrads params

  -- Forward pass: compute loss (training=true enables dropout)
  let lossT ← gpt.loss params x y true

  -- Backward pass
  Tensor.backwardLoss lossT

  -- Gradient clipping (only if gradClip > 0)
  if trainCfg.gradClip > 0 then
    do
      let _ ← clipGPTGrads params trainCfg.gradClip
      pure ()
  else
    pure ()

  -- Get loss value for logging
  let lossVal := Tensor.item lossT

  -- Extract gradients from parameters
  let grads ← TensorStruct.grads params

  -- Update parameters with AdamW using Optax-style step
  let opt := Optim.adamw (lr := lr)
  let (params', optState') ← Optim.step opt params grads optState

  return (params', optState', lossVal)

/-- Training step with gradient accumulation support.
    Accumulates gradients over multiple micro-batches before updating parameters.
    This enables training with larger effective batch sizes on limited memory. -/
def trainStepWithAccum {modelCfg : Config} {batch seq : UInt64}
    (trainCfg : TrainConfig)
    (params : GPTParams (device := device) modelCfg)
    (optState : Optim.AdamWState (GPTParams (device := device) modelCfg))
    (getBatchFn : IO (Tokens (device := device) #[batch, seq] × Tokens (device := device) #[batch, seq]))
    (lr : Float)
    : IO (GPTParams (device := device) modelCfg × Optim.AdamWState (GPTParams (device := device) modelCfg) × Float) := do
  let accumSteps := trainCfg.gradAccumSteps
  let lossScale := 1.0 / accumSteps.toFloat

  -- Zero gradients at the start of accumulation window
  TensorStruct.zeroGrads params
  let mut totalLoss : Float := 0.0

  -- Accumulate gradients over multiple micro-batches
  for _ in [:accumSteps] do
    let (x, y) ← getBatchFn

    -- Forward pass
    let lossT ← gpt.loss params x y true

    -- Scale loss for averaging (backward will scale gradients accordingly)
    let scaledLoss := lossT * lossScale

    -- Backward pass (gradients accumulate since we didn't zero them)
    Tensor.backwardLoss scaledLoss

    totalLoss := totalLoss + Tensor.item lossT

  -- Gradient clipping after accumulation
  if trainCfg.gradClip > 0 then
    let _ ← clipGPTGrads params trainCfg.gradClip
    pure ()

  -- Extract accumulated gradients
  let grads ← TensorStruct.grads params

  -- Update parameters
  let opt := Optim.adamw (lr := lr)
  let (params', optState') ← Optim.step opt params grads optState

  -- Return average loss over accumulation steps
  return (params', optState', totalLoss / accumSteps.toFloat)

/-- Compute perplexity from cross-entropy loss -/
def perplexity (loss : Float) : Float := Float.exp loss

/-- Evaluate model on validation data
    Returns average loss over numBatches batches.
    Uses detached parameters to avoid building computation graphs. -/
def evalLoss {modelCfg : Config}
    (params : GPTParams (device := device) modelCfg)
    (valData : Tokens (device := .CPU) #[n])
    (batchSize blockSize : UInt64)
    (numBatches : Nat)
    : IO Float := do
  let params := TensorStruct.detach params
  let mut totalLoss : Float := 0.0
  for _ in [:numBatches] do
    let (x, y) ← getBatch valData batchSize blockSize device
    -- Forward pass with training=false (disables dropout)
    let lossT ← gpt.loss params x y false
    let lossVal := Tensor.item lossT
    totalLoss := totalLoss + lossVal
  return totalLoss / numBatches.toFloat

/-- Main training loop -/
def trainLoop {modelCfg : Config}
    (trainCfg : TrainConfig)
    (initParams : GPTParams (device := trainCfg.device) modelCfg)
    (initOptState : Optim.AdamWState (GPTParams (device := trainCfg.device) modelCfg))
    (trainData : Tokens (device := .CPU) #[n])
    : IO (GPTParams (device := trainCfg.device) modelCfg) := do
  let device := trainCfg.device
  -- Parameters and optimizer buffers already carry the configured device index.
  let mut params := initParams
  let mut optState := initOptState
  let mut totalLoss : Float := 0.0

  let deviceName := match device.kind with
    | DeviceKind.MPS => "MPS"
    | DeviceKind.CUDA n => s!"CUDA:{n}"
    | DeviceKind.CPU => "CPU"
  IO.println s!"Starting training for {trainCfg.maxIters} iterations on {deviceName}..."
  IO.println s!"  batch_size={trainCfg.batchSize}, block_size={trainCfg.blockSize}"
  IO.println s!"  lr={trainCfg.learningRate}, warmup={trainCfg.warmupIters}"
  (← IO.getStdout).flush

  for iterNum in [:trainCfg.maxIters] do
    -- Get learning rate with schedule
    let lr := getLr trainCfg iterNum

    -- Get batch on target device
    let (x, y) ← getBatch trainData trainCfg.batchSize trainCfg.blockSize device

    -- Training step
    let (params', optState', lossVal) ← trainStep trainCfg params optState x y lr

    params := params'
    optState := optState'
    totalLoss := totalLoss + lossVal

    -- Logging
    if iterNum % trainCfg.logInterval == 0 then
      let avgLoss := totalLoss / trainCfg.logInterval.toFloat
      let liveTensors ← get_live_tensors
      IO.println s!"iter {iterNum}: loss={lossVal}, avg_loss={avgLoss}, lr={lr}, live_tensors={liveTensors}"
      (← IO.getStdout).flush
      totalLoss := 0.0

  IO.println "Training complete!"
  return params

/-- Training loop with validation support
    Returns trained parameters and best validation loss -/
def trainLoopWithVal {modelCfg : Config}
    (trainCfg : TrainConfig)
    (initParams : GPTParams (device := trainCfg.device) modelCfg)
    (initOptState : Optim.AdamWState (GPTParams (device := trainCfg.device) modelCfg))
    (trainData : Tokens (device := .CPU) #[nTrain])
    (valData : Tokens (device := .CPU) #[nVal])
    (evalBatches : Nat := 10)
    : IO (GPTParams (device := trainCfg.device) modelCfg × Float) := do
  let device := trainCfg.device
  -- Parameters and optimizer buffers already carry the configured device index.
  let mut params := initParams
  let mut optState := initOptState
  let mut totalLoss : Float := 0.0
  let mut bestValLoss : Float := 1e10  -- Use large value instead of inf

  let deviceName := match device.kind with
    | DeviceKind.MPS => "MPS"
    | DeviceKind.CUDA n => s!"CUDA:{n}"
    | DeviceKind.CPU => "CPU"
  IO.println s!"Starting training for {trainCfg.maxIters} iterations on {deviceName}..."
  IO.println s!"  batch_size={trainCfg.batchSize}, block_size={trainCfg.blockSize}"
  IO.println s!"  lr={trainCfg.learningRate}, warmup={trainCfg.warmupIters}"
  IO.println s!"  grad_clip={trainCfg.gradClip}, eval_interval={trainCfg.evalInterval}"
  (← IO.getStdout).flush

  for iterNum in [:trainCfg.maxIters] do
    -- Get learning rate with schedule
    let lr := getLr trainCfg iterNum

    -- Get batch on target device
    let (x, y) ← getBatch trainData trainCfg.batchSize trainCfg.blockSize device

    -- Training step
    let (params', optState', lossVal) ← trainStep trainCfg params optState x y lr

    params := params'
    optState := optState'
    totalLoss := totalLoss + lossVal

    -- Logging
    if iterNum % trainCfg.logInterval == 0 && iterNum > 0 then
      let avgLoss := totalLoss / trainCfg.logInterval.toFloat
      let liveTensors ← get_live_tensors
      IO.println s!"iter {iterNum}: loss={lossVal}, avg_loss={avgLoss}, lr={lr}, live_tensors={liveTensors}"
      (← IO.getStdout).flush
      totalLoss := 0.0

    -- Validation at evalInterval
    if iterNum % trainCfg.evalInterval == 0 && iterNum > 0 then
      let valLoss ← evalLoss params valData trainCfg.batchSize trainCfg.blockSize evalBatches
      let valPpl := perplexity valLoss
      IO.println s!"  val_loss={valLoss}, val_ppl={valPpl}"
      if valLoss < bestValLoss then
        bestValLoss := valLoss
        IO.println s!"  [new best val_loss!]"
      (← IO.getStdout).flush

  -- Final validation
  let finalValLoss ← evalLoss params valData trainCfg.batchSize trainCfg.blockSize evalBatches
  let finalValPpl := perplexity finalValLoss
  IO.println s!"Training complete! Final val_loss={finalValLoss}, val_ppl={finalValPpl}"
  IO.println s!"Best val_loss={bestValLoss}"
  return (params, bestValLoss)

end torch.train
