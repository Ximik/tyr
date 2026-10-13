/-
  Training Infrastructure for GPT
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Optim
import Examples.GPT.GPT

namespace torch.train

open torch
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

/-- Move a tensor tree to a device, preserving leaf tensor status. -/
def toDevice [TensorStruct α] (x : α) (_device : Device) : α := x

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
def getBatch (trainData : T #[n]) (batchSize blockSize : UInt64) (device : Device := Device.CPU) : IO (T #[batchSize, blockSize] × T #[batchSize, blockSize]) := do
  -- Pick a random starting position
  -- We need batchSize consecutive sequences of length blockSize+1
  let totalTokens := batchSize * (blockSize + 1)
  let maxStart := n - totalTokens
  let startTensor ← randint 0 maxStart.toInt64 #[1]
  let start := nn.itemInt startTensor

  -- Slice contiguous block from data using slice1d with type annotation
  let block : T #[totalTokens] := torch.data.slice1d trainData start (start + totalTokens.toInt64)

  -- Reshape to [batchSize, blockSize+1]
  let reshaped := reshape block #[batchSize, blockSize + 1]

  -- x = reshaped[:, :-1], y = reshaped[:, 1:]
  -- Using slice on dim 1 (second dimension)
  let xRaw := reshaped.slice 1 0 blockSize.toInt64 1
  let yRaw := reshaped.slice 1 1 (blockSize + 1).toInt64 1

  -- Reshape to ensure correct types
  let x := reshape xRaw #[batchSize, blockSize]
  let y := reshape yRaw #[batchSize, blockSize]

  -- Move to target device
  match device with
  | Device.CPU => return (x, y)
  | _ =>
    let x_dev := autograd.clone (x.to device)
    let y_dev := autograd.clone (y.to device)
    return (x_dev, y_dev)

/-- Clip gradients for a single block's parameters -/
def clipBlockGrads {n_embd : UInt64} (params : BlockParams n_embd) (maxNorm : Float) : IO Unit := do
  let _ ← nn.clip_grad_norm_ (s := params.ln1_weight.shape) params.ln1_weight.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.ln1_bias.shape) params.ln1_bias.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.q_proj.shape) params.q_proj.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.k_proj.shape) params.k_proj.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.v_proj.shape) params.v_proj.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.c_proj.shape) params.c_proj.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.c_proj_bias.shape) params.c_proj_bias.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.ln2_weight.shape) params.ln2_weight.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.ln2_bias.shape) params.ln2_bias.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.mlp_fc.shape) params.mlp_fc.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.mlp_fc_bias.shape) params.mlp_fc_bias.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.mlp_proj.shape) params.mlp_proj.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.mlp_proj_bias.shape) params.mlp_proj_bias.raw maxNorm

/-- Clip gradients for all GPT parameters -/
def clipGPTGrads {cfg : Config} (params : GPTParams cfg) (maxNorm : Float) : IO Unit := do
  let _ ← nn.clip_grad_norm_ (s := params.wte.shape) params.wte.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.wpe.shape) params.wpe.raw maxNorm
  for block in params.blocks do
    clipBlockGrads block maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.ln_f_weight.shape) params.ln_f_weight.raw maxNorm
  let _ ← nn.clip_grad_norm_ (s := params.ln_f_bias.shape) params.ln_f_bias.raw maxNorm

/-- Single training step using Optax-style optimizer -/
def trainStep {modelCfg : Config} {batch seq : UInt64}
    (trainCfg : TrainConfig)
    (params : GPTParams modelCfg)
    (optState : Optim.AdamWState (GPTParams modelCfg))
    (x : T #[batch, seq])
    (y : T #[batch, seq])
    (lr : Float)
    : IO (GPTParams modelCfg × Optim.AdamWState (GPTParams modelCfg) × Float) := do
  -- Zero gradients BEFORE forward/backward to prevent gradient accumulation
  let params := TensorStruct.zeroGrads params

  -- Forward pass: compute loss (training=true enables dropout)
  let lossT ← gpt.lossRaw params x y true

  -- Backward pass
  autograd.backwardLoss lossT

  -- Gradient clipping (only if gradClip > 0)
  if trainCfg.gradClip > 0 then
    do
      let _ ← clipGPTGrads params trainCfg.gradClip
      pure ()
  else
    pure ()

  -- Get loss value for logging
  let lossVal := nn.item lossT

  -- Extract gradients from parameters
  let grads := TensorStruct.grads params

  -- Update parameters with AdamW using Optax-style step
  let opt := Optim.adamw (lr := lr)
  let (params', optState') := Optim.step opt params grads optState

  return (params', optState', lossVal)

/-- Training step with gradient accumulation support.
    Accumulates gradients over multiple micro-batches before updating parameters.
    This enables training with larger effective batch sizes on limited memory. -/
def trainStepWithAccum {modelCfg : Config} {batch seq : UInt64}
    (trainCfg : TrainConfig)
    (params : GPTParams modelCfg)
    (optState : Optim.AdamWState (GPTParams modelCfg))
    (getBatchFn : IO (T #[batch, seq] × T #[batch, seq]))
    (lr : Float)
    : IO (GPTParams modelCfg × Optim.AdamWState (GPTParams modelCfg) × Float) := do
  let accumSteps := trainCfg.gradAccumSteps
  let lossScale := 1.0 / accumSteps.toFloat

  -- Zero gradients at the start of accumulation window
  let params := TensorStruct.zeroGrads params
  let mut totalLoss : Float := 0.0

  -- Accumulate gradients over multiple micro-batches
  for _ in [:accumSteps] do
    let (x, y) ← getBatchFn

    -- Forward pass
    let lossT ← gpt.lossRaw params x y true

    -- Scale loss for averaging (backward will scale gradients accordingly)
    let scaledLoss := mul_scalar lossT lossScale

    -- Backward pass (gradients accumulate since we didn't zero them)
    autograd.backwardLoss scaledLoss

    totalLoss := totalLoss + nn.item lossT

  -- Gradient clipping after accumulation
  if trainCfg.gradClip > 0 then
    let _ ← clipGPTGrads params trainCfg.gradClip
    pure ()

  -- Extract accumulated gradients
  let grads := TensorStruct.grads params

  -- Update parameters
  let opt := Optim.adamw (lr := lr)
  let (params', optState') := Optim.step opt params grads optState

  -- Return average loss over accumulation steps
  return (params', optState', totalLoss / accumSteps.toFloat)

/-- Compute perplexity from cross-entropy loss -/
def perplexity (loss : Float) : Float := Float.exp loss

/-- Evaluate model on validation data
    Returns average loss over numBatches batches.
    Uses no_grad context to avoid building computation graphs. -/
def evalLoss {modelCfg : Config}
    (params : GPTParams modelCfg)
    (valData : T #[n])
    (batchSize blockSize : UInt64)
    (numBatches : Nat)
    (device : Device := Device.CPU)
    : IO Float := autograd.no_grad do
  let mut totalLoss : Float := 0.0
  for _ in [:numBatches] do
    let (x, y) ← getBatch valData batchSize blockSize device
    -- Forward pass with training=false (disables dropout)
    let lossT ← gpt.lossRaw params x y false
    let lossVal := nn.item lossT
    totalLoss := totalLoss + lossVal
  return totalLoss / numBatches.toFloat

/-- Main training loop -/
def trainLoop {modelCfg : Config}
    (trainCfg : TrainConfig)
    (initParams : GPTParams modelCfg)
    (initOptState : Optim.AdamWState (GPTParams modelCfg))
    (trainData : T #[n])
    : IO (GPTParams modelCfg) := do
  let device := trainCfg.device
  -- Move model and optimizer state to device
  let mut params := toDevice initParams device
  let mut optState := toDevice initOptState device
  let mut totalLoss : Float := 0.0

  let deviceName := match device with
    | Device.MPS => "MPS"
    | Device.CUDA n => s!"CUDA:{n}"
    | Device.CPU => "CPU"
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
    (initParams : GPTParams modelCfg)
    (initOptState : Optim.AdamWState (GPTParams modelCfg))
    (trainData : T #[nTrain])
    (valData : T #[nVal])
    (evalBatches : Nat := 10)
    : IO (GPTParams modelCfg × Float) := do
  let device := trainCfg.device
  -- Move model and optimizer state to device
  let mut params := toDevice initParams device
  let mut optState := toDevice initOptState device
  let mut totalLoss : Float := 0.0
  let mut bestValLoss : Float := 1e10  -- Use large value instead of inf

  let deviceName := match device with
    | Device.MPS => "MPS"
    | Device.CUDA n => s!"CUDA:{n}"
    | Device.CPU => "CPU"
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
      let valLoss ← evalLoss params valData trainCfg.batchSize trainCfg.blockSize evalBatches device
      let valPpl := perplexity valLoss
      IO.println s!"  val_loss={valLoss}, val_ppl={valPpl}"
      if valLoss < bestValLoss then
        bestValLoss := valLoss
        IO.println s!"  [new best val_loss!]"
      (← IO.getStdout).flush

  -- Final validation
  let finalValLoss ← evalLoss params valData trainCfg.batchSize trainCfg.blockSize evalBatches device
  let finalValPpl := perplexity finalValLoss
  IO.println s!"Training complete! Final val_loss={finalValLoss}, val_ppl={finalValPpl}"
  IO.println s!"Best val_loss={bestValLoss}"
  return (params, bestValLoss)

end torch.train
