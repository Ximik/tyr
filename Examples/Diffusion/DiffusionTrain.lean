/-
  Training Infrastructure for Discrete Diffusion

  Adapted from GPT training for masked discrete diffusion.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Optim
import Examples.Diffusion.Parameters
import Examples.Diffusion.Diffusion
import Examples.Diffusion.DiffusionSchedule
import Examples.Diffusion.DiffusionCheckpoint

namespace torch.diffusion.train

open torch
open torch.diffusion

/-- Diffusion training configuration -/
structure TrainConfig where
  maxIters : Nat := 10000
  evalInterval : Nat := 500
  logInterval : Nat := 50
  learningRate : Float := 3e-4
  minLr : Float := 1e-5
  warmupIters : Nat := 200
  lrDecayIters : Nat := 10000
  gradClip : Float := 1.0
  batchSize : Nat := 32
  weightDecay : Float := 0.01
  checkpointInterval : Nat := 1000  -- Save checkpoint every N iterations
  checkpointDir : Option String := none  -- Directory for checkpoints
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

/-- Get a random batch of data from token array.
    For diffusion, we just need contiguous sequences (no shifting).
    Returns tokens of shape [batchSize, seqLen] -/
def getBatch {n : Nat} (trainData : Tokens #[n]) (batchSize seqLen : Nat)
    : IO (Tokens #[batchSize, seqLen]) := do
  let required := batchSize * seqLen
  if batchSize == 0 || seqLen == 0 || required > n then
    throw <| IO.userError "Diffusion batch requires positive dimensions and enough tokens"
  let totalTokens := required
  let startTensor ← Tensor.randint 0 (n - totalTokens + 1).toInt64 #[]
  let start := startTensor.itemInt.toInt.toNat
  return (trainData.sliceChecked 0 start totalTokens #[totalTokens]).reshape #[batchSize, seqLen]

/-- Clip gradients for a diffusion transformer block. -/
def clipBlockGrads {n_embd n_head n_kv_head : Nat}
    (params : BlockParams n_embd n_head n_kv_head)
    (maxNorm : Float) : IO Unit := do
  let _ ← Tensor.clipGradNorm params.attn.c_q maxNorm
  let _ ← Tensor.clipGradNorm params.attn.c_k maxNorm
  let _ ← Tensor.clipGradNorm params.attn.c_v maxNorm
  let _ ← Tensor.clipGradNorm params.attn.c_proj maxNorm
  let _ ← Tensor.clipGradNorm params.mlp.c_fc maxNorm
  let _ ← Tensor.clipGradNorm params.mlp.c_proj maxNorm

/-- Clip gradients for all diffusion model parameters -/
def clipDiffusionGrads {cfg : Config} (params : DiffusionParams cfg) (maxNorm : Float) : IO Unit := do
  let _ ← Tensor.clipGradNorm params.token_emb maxNorm
  let _ ← Tensor.clipGradNorm params.time_emb maxNorm
  for block in params.blocks do
    clipBlockGrads block maxNorm
  let _ ← Tensor.clipGradNorm params.output_head maxNorm

/-- Single diffusion training step -/
def trainStep {modelCfg : Config} {batch seq num_timesteps : Nat}
    (trainCfg : TrainConfig)
    (params : DiffusionParams modelCfg)
    (optState : Optim.AdamWState (DiffusionParams modelCfg))
    (schedule : MaskedDiffusionSchedule num_timesteps)
    (x_0 : Tokens #[batch, seq])  -- Clean tokens
    (rotaryCache : RotaryCache rotaryLen modelCfg.headDim)
    (lr : Float)
    : IO (DiffusionParams modelCfg × Optim.AdamWState (DiffusionParams modelCfg) × Float) := do
  -- Zero gradients BEFORE forward/backward
  TensorStruct.zeroGrads params

  -- Sample random timesteps for each sample in batch
  let t ← Tensor.randint 0 num_timesteps.toInt64 #[batch]

  -- Add masks to get noisy input
  let x_t ← schedule.addMasks x_0 t

  -- Forward pass
  let logits := forward params x_t t rotaryCache

  -- Compute masked loss (only on masked positions)
  let lossT := maskedCrossEntropyLoss logits x_0 x_t schedule.mask_token_id

  -- Backward pass
  Tensor.backwardLoss lossT

  -- Gradient clipping
  if trainCfg.gradClip > 0 then
    clipDiffusionGrads params trainCfg.gradClip

  -- Get loss value for logging
  let lossVal := Tensor.item lossT

  -- Extract gradients from parameters
  let grads ← TensorStruct.grads params

  -- Update parameters with AdamW
  let opt := Optim.adamw (lr := lr) (weight_decay := trainCfg.weightDecay)
  let (params', optState') ← Optim.step opt params grads optState

  return (params', optState', lossVal)

/-- Evaluate model on validation data.
    Uses detached parameters to avoid building computation graphs. -/
def evalLoss {modelCfg : Config} {num_timesteps : Nat}
    (params : DiffusionParams modelCfg)
    (schedule : MaskedDiffusionSchedule num_timesteps)
    (valData : Tokens #[n])
    (batchSize seqLen : Nat)
    (rotaryCache : RotaryCache rotaryLen modelCfg.headDim)
    (numBatches : Nat)
    : IO Float := do
  let params := TensorStruct.detach params
  let rotaryCache := { rotaryCache with cos := rotaryCache.cos.detach, sin := rotaryCache.sin.detach }
  let mut totalLoss : Float := 0.0
  for _ in [:numBatches] do
    let x_0 ← getBatch valData batchSize seqLen
    let t ← Tensor.randint 0 num_timesteps.toInt64 #[batchSize]
    let x_t ← schedule.addMasks x_0 t
    let logits := forward params x_t t rotaryCache
    let lossT := maskedCrossEntropyLoss logits x_0 x_t schedule.mask_token_id
    let lossVal := Tensor.item lossT
    totalLoss := totalLoss + lossVal
  return totalLoss / numBatches.toFloat

/-- Main diffusion training loop with validation -/
def trainLoop {modelCfg : Config} {num_timesteps : Nat}
    (trainCfg : TrainConfig)
    (initParams : DiffusionParams modelCfg)
    (initOptState : Optim.AdamWState (DiffusionParams modelCfg))
    (schedule : MaskedDiffusionSchedule num_timesteps)
    (trainData : Tokens #[nTrain])
    (valData : Tokens #[nVal])
    (rotaryCache : RotaryCache rotaryLen modelCfg.headDim)
    (evalBatches : Nat := 10)
    : IO (DiffusionParams modelCfg × Float) := do
  let mut params := initParams
  let mut optState := initOptState
  let mut totalLoss : Float := 0.0
  let mut bestValLoss : Float := 1e10

  IO.println s!"Starting diffusion training for {trainCfg.maxIters} iterations..."
  IO.println s!"  batch_size={trainCfg.batchSize}, seq_len={modelCfg.seq_len}"
  IO.println s!"  lr={trainCfg.learningRate}, warmup={trainCfg.warmupIters}"
  IO.println s!"  diffusion_steps={num_timesteps}, context_len={schedule.context_len}"
  (← IO.getStdout).flush

  for iterNum in [:trainCfg.maxIters] do
    -- Get learning rate with schedule
    let lr := getLr trainCfg iterNum

    -- Get batch of clean tokens
    let x_0 ← getBatch trainData trainCfg.batchSize modelCfg.seq_len

    -- Training step
    let (params', optState', lossVal) ← trainStep trainCfg params optState schedule x_0 rotaryCache lr

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

    -- Validation
    if iterNum % trainCfg.evalInterval == 0 && iterNum > 0 then
      let valLoss ← evalLoss params schedule valData trainCfg.batchSize modelCfg.seq_len rotaryCache evalBatches
      IO.println s!"  val_loss={valLoss}"
      if valLoss < bestValLoss then
        bestValLoss := valLoss
        IO.println s!"  [new best val_loss!]"
        -- Save best checkpoint
        if let some dir := trainCfg.checkpointDir then
          checkpoint.saveFullCheckpoint params optState iterNum bestValLoss totalLoss (dir ++ "/best")
      (← IO.getStdout).flush

    -- Periodic checkpoint
    if let some dir := trainCfg.checkpointDir then
      if iterNum % trainCfg.checkpointInterval == 0 && iterNum > 0 then
        checkpoint.saveFullCheckpoint params optState iterNum bestValLoss totalLoss (dir ++ "/latest")

  -- Final validation
  let finalValLoss ← evalLoss params schedule valData trainCfg.batchSize modelCfg.seq_len rotaryCache evalBatches
  IO.println s!"Training complete! Final val_loss={finalValLoss}"
  IO.println s!"Best val_loss={bestValLoss}"

  -- Save final checkpoint
  if let some dir := trainCfg.checkpointDir then
    checkpoint.saveFullCheckpoint params optState trainCfg.maxIters finalValLoss 0.0 (dir ++ "/final")

  return (params, bestValLoss)

end torch.diffusion.train
