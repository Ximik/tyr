/-
  Tyr/ModdedTrain.lean

  Training infrastructure for modded-nanogpt style training.

  Key features:
  - Dynamic batch size and window size schedules
  - LR schedule with cosine cooldown
  - Muon momentum warmup/cooldown
  - Alternating optimizer steps (Muon only on even steps)
  - Validation with HellaSwag
  - Distributed training coordination

  Based on modded-nanogpt's training loop.
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Distributed
import Examples.NanoChat.ModdedGPT
import Tyr.DataLoader
import Tyr.Checkpoint
import Examples.GPT.GPTDataLoader
import Tyr.Optim
import Tyr.Optim.NorMuon
import Tyr.Optim.DistAdam

namespace torch.ModdedTrain

open torch

variable {device : Device}
open torch.moddedGpt
open torch.DataLoader
open torch.Optim

/-! ## Hyperparameters -/

/-- Training hyperparameters for nanochat-style loops. -/
structure Hyperparameters where
  /-- Total scheduled training iterations -/
  numIterations : UInt64 := 2050
  /-- Extension iterations beyond `numIterations` (usually 0). -/
  extensionIterations : UInt64 := 0
  /-- Per-rank micro-batch size. -/
  deviceBatchSize : UInt64 := 32
  /-- Target global batch size in tokens across all ranks/accumulation. -/
  totalBatchSizeTokens : UInt64 := 524288
  /-- Embedding learning rate (Adam). -/
  embeddingLr : Float := 0.3
  /-- Unembedding / lm-head learning rate (Adam). -/
  unembeddingLr : Float := 0.004
  /-- Matrix learning rate (Muon). -/
  matrixLr : Float := 0.02
  /-- Adam beta1 (nanochat default: 0.8). -/
  adamBeta1 : Float := 0.8
  /-- Adam beta2 (nanochat default: 0.95). -/
  adamBeta2 : Float := 0.95
  /-- Adam weight decay (nanochat default: 0.0). -/
  adamWeightDecay : Float := 0.0
  /-- Warmup fraction of total steps. -/
  warmupFrac : Float := 0.0
  /-- Final warmdown fraction of total steps. -/
  cooldownFrac : Float := 0.4
  /-- Final learning-rate multiplier at end of warmdown. -/
  finalLrFrac : Float := 0.0
  /-- Block size for windowed attention -/
  blockSize : UInt64 := 128
  /-- Maximum sequence length -/
  maxSeqLen : UInt64 := 2048
  /-- Validation interval (in iterations) -/
  valInterval : UInt64 := 50
  /-- Logging interval -/
  logInterval : UInt64 := 10
  /-- Checkpoint interval -/
  checkpointInterval : UInt64 := 100
  deriving Repr, Inhabited, Lean.ToJson, Lean.FromJson

/-- Default hyperparameters -/
def Hyperparameters.default : Hyperparameters := {}

/-! ## Batch Size Schedule -/

/-- Get batch size for a given iteration.

    modded-nanogpt schedule:
    - Iterations 0-199: batch_size = 8
    - Iterations 200-999: batch_size = 16
    - Iterations 1000+: batch_size = 24

    These correspond to tokens/iter:
    - 8 * 2048 * 8 = 131072
    - 16 * 2048 * 8 = 262144
    - 24 * 2048 * 8 = 393216
-/
def getBatchSize (step : UInt64) : UInt64 :=
  if step < 200 then 8
  else if step < 1000 then 16
  else 24

/-- Get tokens per batch for a given iteration -/
def getTokensPerBatch (step : UInt64) (seqLen gradAccum worldSize : UInt64) : UInt64 :=
  getBatchSize step * seqLen * gradAccum * worldSize

/-! ## Window Size Schedule -/

/-- Get window sizes (short, long) for a given iteration.

    modded-nanogpt schedule:
    - Iterations 0-199: (3, 3) blocks = 384 tokens
    - Iterations 200-999: (3, 7) blocks = 384, 896 tokens
    - Iterations 1000+: (3, 11) blocks = 384, 1408 tokens

    Short window is used for most attention heads,
    long window for a few "global" heads.
-/
def getWindowSizes (step : UInt64) (_blockSize : UInt64 := 128) : (UInt64 × UInt64) :=
  if step < 200 then (3, 3)
  else if step < 1000 then (3, 7)
  else (3, 11)

/-- Get sequence length for a given iteration -/
def getSeqLen (step : UInt64) (blockSize : UInt64 := 128) : UInt64 :=
  let (_, wsLong) := getWindowSizes step blockSize
  wsLong * blockSize

/-! ## Learning Rate Schedule -/

/-- Get learning rate for a given iteration.
    Matches nanochat base_train's warmup/plateau/linear-warmdown semantics. -/
def getLearningRate (step : UInt64) (hp : Hyperparameters)
    (baseLr : Float := 0.023) : Float :=
  let totalSteps := max 1 (hp.numIterations + hp.extensionIterations)
  let warmupSteps := (hp.warmupFrac * totalSteps.toFloat).toUInt64
  let warmdownSteps := (hp.cooldownFrac * totalSteps.toFloat).toUInt64
  let warmdownStart := totalSteps - warmdownSteps
  if warmupSteps > 0 && step < warmupSteps then
    baseLr * (step.toFloat + 1.0) / warmupSteps.toFloat
  else if warmdownSteps == 0 || step <= warmdownStart then
    baseLr
  else
    let progress := (totalSteps - step).toFloat / warmdownSteps.toFloat
    baseLr * (progress + (1.0 - progress) * hp.finalLrFrac)

/-- Batch-size LR scaling used by nanochat: sqrt(batch/reference_batch). -/
def batchLrScale (hp : Hyperparameters) (referenceBatch : Float := 524288.0) : Float :=
  if hp.totalBatchSizeTokens == 0 then
    1.0
  else
    Float.sqrt (hp.totalBatchSizeTokens.toFloat / referenceBatch)

/-- Scale Adam learning rates by model dimension: (d_model / 768)^(-0.5). -/
def dModelLrScale (cfg : moddedGpt.Config) : Float :=
  Float.pow (cfg.modelDim.toFloat / 768.0) (-0.5)

/-- Compute grad-accum steps from global token budget and world size.
    Expects divisibility to be validated up-front (nanochat parity). -/
def effectiveGradAccumSteps (hp : Hyperparameters) (worldSize : UInt64) : UInt64 :=
  let world := max 1 worldSize
  let perMicro := hp.deviceBatchSize * hp.maxSeqLen * world
  if perMicro == 0 then
    1
  else
    let target := max perMicro hp.totalBatchSizeTokens
    max 1 (target / perMicro)

/-- Enforce nanochat batch semantics:
    total_batch_size must be divisible by per-rank micro-batch tokens. -/
def validateGradAccumConfig (hp : Hyperparameters) (worldSize : UInt64) : IO Unit := do
  let world := max 1 worldSize
  let perMicro := hp.deviceBatchSize * hp.maxSeqLen * world
  if perMicro == 0 then
    throw <| IO.userError "Invalid grad-accum config: perMicro tokens is zero"
  if hp.totalBatchSizeTokens % perMicro != 0 then
    throw <| IO.userError s!"Invalid grad-accum config: total_batch_size={hp.totalBatchSizeTokens} must be divisible by device_batch_size*max_seq_len*world_size={perMicro}"

/-! ## Momentum Schedule -/

/-- Get Muon momentum for a given iteration (nanochat parity).

    Momentum schedule:
    - Warmup: 0.85 -> baseMomentum over 300 steps
    - Then constant at baseMomentum
-/
def getMuonMomentum (step : UInt64) (hp : Hyperparameters)
    (baseMomentum : Float := 0.95) (warmupSteps : UInt64 := 300)
    (cooldownSteps : UInt64 := 50) : Float :=
  let _ := hp
  let _ := cooldownSteps
  NorMuon.getMomentum step.toNat
    0
    baseMomentum warmupSteps.toNat 0

/-! ## Optimizer State -/

/-- Muon state for an attention module. -/
structure MuonAttnState (cfg : moddedGpt.Config) where
  wQ : NorMuon.ParamState ⟨#[cfg.nHead * cfg.headDim, cfg.modelDim], .Float32, device⟩
  wK : NorMuon.ParamState ⟨#[cfg.nHead * cfg.headDim, cfg.modelDim], .Float32, device⟩
  wV : NorMuon.ParamState ⟨#[cfg.nHead * cfg.headDim, cfg.modelDim], .Float32, device⟩
  wO : NorMuon.ParamState ⟨#[cfg.modelDim, cfg.nHead * cfg.headDim], .Float32, device⟩
  deriving Repr, TensorStruct

/-- Muon state for a transformer block. -/
structure MuonBlockState (cfg : moddedGpt.Config) where
  attn : Option (MuonAttnState (device := device) cfg)
  cFc : NorMuon.ParamState ⟨#[4 * cfg.modelDim, cfg.modelDim], .Float32, device⟩
  cProj : NorMuon.ParamState ⟨#[4 * cfg.modelDim, cfg.modelDim], .Float32, device⟩
  deriving Repr, TensorStruct

/-- Full dual-optimizer parameter state (DistAdam + Muon). -/
structure DualParamState (cfg : moddedGpt.Config) where
  embed : DistAdam.ParamState ⟨#[cfg.vocabSize, cfg.modelDim], .Float32, device⟩
  valueEmbeds : Array (DistAdam.ParamState ⟨#[cfg.vocabSize, cfg.modelDim], .Float32, device⟩)
  lmHead : DistAdam.ParamState ⟨#[cfg.vocabSize, cfg.modelDim], .Float32, device⟩
  scalars : DistAdam.ParamState ⟨#[cfg.nLayer * 4 + 8], .Float32, device⟩
  smearGate : NorMuon.ParamState ⟨#[1, 12], .Float32, device⟩
  blocks : Array (MuonBlockState (device := device) cfg)
  deriving Repr, TensorStruct

private def initDualParamState (cfg : moddedGpt.Config) (params : ModdedGPTParams (device := device) cfg) : DualParamState (device := device) cfg :=
  let attnState? (attn : Option (CausalSelfAttention (device := device) cfg.modelDim cfg.headDim cfg.nHead)) :
      Option (MuonAttnState (device := device) cfg) :=
    match attn with
    | none => none
    | some a =>
      some {
        wQ := NorMuon.initParamState a.wQ
        wK := NorMuon.initParamState a.wK
        wV := NorMuon.initParamState a.wV
        wO := NorMuon.initParamState a.wO
      }
  let blockStates : Array (MuonBlockState (device := device) cfg) := params.blocks.map fun b =>
    {
      attn := attnState? b.attn
      cFc := NorMuon.initParamState b.mlp.cFc
      cProj := NorMuon.initParamState b.mlp.cProj
    }
  {
    embed := DistAdam.initParamState params.embed
    valueEmbeds := params.valueEmbeds.map DistAdam.initParamState
    lmHead := DistAdam.initParamState params.lmHead.weight
    scalars := DistAdam.initParamState params.scalars.values
    smearGate := NorMuon.initParamState params.smearGate.weight
    blocks := blockStates
  }

/-- Combined optimizer state for training.
    Keeps legacy Adam state for compatibility, and dual parameter states for updates. -/
structure OptimizerState (cfg : moddedGpt.Config) where
  /-- Legacy AdamW optimizer state (kept for checkpoint compatibility). -/
  adamState : Optim.AdamWState (ModdedGPTParams (device := device) cfg)
  /-- DistAdam + Muon parameter states used by the training step. -/
  dualState : DualParamState (device := device) cfg
  /-- Current step -/
  step : UInt64
  /-- Base learning rate -/
  baseLr : Float := 0.023
  /-- Weight decay -/
  weightDecay : Float := 0.01

/-- Initialize optimizer state from model parameters -/
def OptimizerState.init (cfg : moddedGpt.Config) (params : ModdedGPTParams (device := device) cfg)
    (lr : Float := 0.023) (weightDecay : Float := 0.0) : OptimizerState (device := device) cfg :=
  let opt := Optim.adamw (lr := lr) (weight_decay := weightDecay)
  {
    adamState := opt.init params
    dualState := initDualParamState cfg params
    step := 0
    baseLr := lr
    weightDecay := weightDecay
  }

/-- Get current learning rate from schedule -/
def OptimizerState.currentLr (state : OptimizerState (device := device) cfg) (hp : Hyperparameters) : Float :=
  getLearningRate state.step hp state.baseLr

/-! ## Training Step -/

/-- Result of a single training step -/
structure StepResult where
  /-- Training loss -/
  loss : Float
  /-- Gradient norm (for monitoring) -/
  gradNorm : Float
  /-- Tokens processed -/
  tokensProcessed : UInt64
  /-- Time taken (ms) -/
  timeMs : Float
  deriving Repr

/-- Perform a single training step.

    Steps:
    1. Zero gradients
    2. Forward pass
    3. Backward pass
    4. Extract gradients
    5. Apply optimizer (AdamW)
    6. Return updated params and optimizer state
-/
def trainStep {cfg : moddedGpt.Config} {batch seq : UInt64}
    (params : ModdedGPTParams (device := device) cfg)
    (yarn : YarnRotary (device := device) cfg.headDim cfg.maxSeqLen)
    (input : Tensor { shape := #[batch, seq], dtype := .Int64, device := device })
    (target : Tensor { shape := #[batch, seq], dtype := .Int64, device := device })
    (optState : OptimizerState (device := device) cfg)
    (hp : Hyperparameters)
    (gradClip : Float := 0.0)
    : IO (ModdedGPTParams (device := device) cfg × OptimizerState (device := device) cfg × StepResult) := do
  let startTime ← IO.monoMsNow

  -- Ensure parameters are trainable leaves before backprop.
  let params ← TensorStruct.makeLeafParams params
  TensorStruct.zeroGrads params

  -- Forward pass (window pattern is in cfg)
  let lossT ← moddedGpt.loss params yarn input target true

  -- Backward pass
  Tensor.backwardLoss lossT

  -- Get loss value for logging
  let lossVal := Tensor.item lossT

  -- Gradient clipping (per-tensor norm clipping)
  if gradClip > 0 then
    let _ ← Tensor.clipGradNorm params.embed gradClip
    let _ ← Tensor.clipGradNorm params.smearGate.weight gradClip
    for ve in params.valueEmbeds do
      let _ ← Tensor.clipGradNorm ve gradClip
    for block in params.blocks do
      -- Attention layer (may be None for some layers)
      match block.attn with
      | some attn =>
        let _ ← Tensor.clipGradNorm attn.wQ gradClip
        let _ ← Tensor.clipGradNorm attn.wK gradClip
        let _ ← Tensor.clipGradNorm attn.wV gradClip
        let _ ← Tensor.clipGradNorm attn.wO gradClip
      | none => pure ()
      -- MLP
      let _ ← Tensor.clipGradNorm block.mlp.cFc gradClip
      let _ ← Tensor.clipGradNorm block.mlp.cProj gradClip
    let _ ← Tensor.clipGradNorm params.lmHead.weight gradClip
    let _ ← Tensor.clipGradNorm params.scalars.values gradClip

  -- Extract gradients from parameters
  let grads ← TensorStruct.grads params

  -- Get current learning rate from schedule
  let lr := getLearningRate optState.step hp optState.baseLr

  -- Create optimizer with current LR
  let opt := Optim.adamw (lr := lr) (weight_decay := optState.weightDecay)

  -- Apply optimizer step
  let (newParams, newAdamState) ← Optim.step opt params grads optState.adamState

  let endTime ← IO.monoMsNow
  let timeMs := (endTime - startTime).toFloat

  let result : StepResult := {
    loss := lossVal
    gradNorm := 0.0  -- Could compute from grads if needed
    tokensProcessed := batch * seq
    timeMs := timeMs
  }

  let newOptState : OptimizerState (device := device) cfg := {
    adamState := newAdamState
    dualState := optState.dualState
    step := optState.step + 1
    baseLr := optState.baseLr
    weightDecay := optState.weightDecay
  }

  return (newParams, newOptState, result)

/-! ## Gradient Accumulation -/

/-- Accumulate gradients over multiple micro-batches -/
def accumulateGradients {cfg : moddedGpt.Config} {batch seq : UInt64}
    (params : ModdedGPTParams (device := device) cfg)
    (yarn : YarnRotary (device := device) cfg.headDim cfg.maxSeqLen)
    (batches : Array (Tensor { shape := #[batch, seq], dtype := .Int64, device := device } × Tensor { shape := #[batch, seq], dtype := .Int64, device := device }))
    : IO Float := do
  let mut totalLoss := 0.0
  let numBatches := batches.size

  for (input, target) in batches do
    let lossT ← moddedGpt.loss params yarn input target true
    -- Scale loss for accumulation
    let scaledLoss := Tensor.divScalar lossT numBatches.toFloat
    Tensor.backwardLoss scaledLoss
    totalLoss := totalLoss + Tensor.item lossT

  return totalLoss / numBatches.toFloat

/-! ## Validation -/

/-- Validation result -/
structure ValidationResult where
  /-- Validation loss -/
  loss : Float
  /-- HellaSwag accuracy (if evaluated) -/
  hellaswagAcc : Option Float
  /-- Time taken (ms) -/
  timeMs : Float
  deriving Repr

/-- Run validation on held-out data -/
def validate {cfg : moddedGpt.Config}
    (params : ModdedGPTParams (device := device) cfg)
    (yarn : YarnRotary (device := device) cfg.headDim cfg.maxSeqLen)
    (valData : Σ n, DataShard n)
    (batchSize seqLen : UInt64)
    (numBatches : UInt64 := 10)
    : IO ValidationResult := do
  let startTime ← IO.monoMsNow
  let mut totalLoss := 0.0
  let mut numValid := 0

  -- Create iterator
  let mut iter := BatchIterator.new valData.2 batchSize seqLen

  for _ in [:numBatches.toNat] do
    let (maybeBatch, newIter) ← iter.nextGPT
    iter := newIter
    match maybeBatch with
    | some (inputDyn, targetDyn) =>
      -- Reshape dynamic tensors to expected shape
      let input := (inputDyn.reshapeChecked #[batchSize, seqLen]).to device
      let target := (targetDyn.reshapeChecked #[batchSize, seqLen]).to device
      let lossT ← moddedGpt.loss params yarn input target false
      totalLoss := totalLoss + Tensor.item lossT
      numValid := numValid + 1
    | none => break

  let avgLoss := if numValid > 0 then totalLoss / numValid.toFloat else 0.0

  let endTime ← IO.monoMsNow
  let timeMs := (endTime - startTime).toFloat

  return {
    loss := avgLoss
    hellaswagAcc := none  -- Would run HellaSwag eval
    timeMs := timeMs
  }

/-! ## Checkpointing -/

/-- Checkpoint state -/
structure Checkpoint (cfg : moddedGpt.Config) where
  /-- Model parameters -/
  params : ModdedGPTParams (device := device) cfg
  /-- Optimizer state -/
  optState : OptimizerState (device := device) cfg
  /-- Current step -/
  step : UInt64
  /-- Best validation loss -/
  bestValLoss : Float

/-- Serializable data-loader cursor for faithful resume. -/
structure DataCursor where
  /-- Index into resolved training shard paths. -/
  trainPathIdx : Nat
  /-- Number of batches consumed by the distributed generator. -/
  globalStep : UInt64
  /-- Iterator epoch counter. -/
  epoch : UInt64
  /-- Iterator batch counter within epoch. -/
  batchCount : UInt64
  /-- Current BOS-finder token offset inside the current shard. -/
  bosCurrentPos : UInt64
  /-- Versioned full-partition document-stream cursor. Legacy offsets cannot
      identify the new shuffled stream and are rejected on exact resume. -/
  streamCursor? : Option StreamCursor := none
  deriving Repr, Inhabited, Lean.ToJson, Lean.FromJson

/-- Training metadata stored alongside tensor checkpoints. -/
structure CheckpointMetadata where
  /-- Metadata schema version. -/
  version : UInt64 := 1
  /-- `repr cfg` captured at save time for compatibility checks. -/
  modelConfigRepr : String
  /-- Structured model config captured at save time for robust resume. -/
  modelConfig? : Option moddedGpt.Config := none
  /-- Saved training hyperparameters for resume parity. -/
  hyperparameters : Hyperparameters
  /-- Total tokens processed when this checkpoint was written. -/
  totalTokens : UInt64 := 0
  /-- Optional parquet-loader cursor state. -/
  dataCursor? : Option DataCursor := none
  deriving Repr, Inhabited, Lean.ToJson, Lean.FromJson

private def checkpointMetadataFile (path : String) : System.FilePath :=
  ⟨s!"{path}/training_meta.json"⟩

private def checkpointExistsAt (path : String) : IO Bool := do
  data.fileExists s!"{path}/step.pt"

/-- Resolve once, retaining whether strict snapshot metadata is required. -/
private def resolveCheckpointPath (path : String) : IO (String × Bool) := do
  let snapshot ← checkpoint.resolveSnapshot path
  let inSnapshotDir := ((System.FilePath.mk snapshot).parent >>= System.FilePath.fileName) == some ".snapshots"
  pure (snapshot, snapshot != path || inSnapshotDir)

/-- Check whether a published snapshot or legacy flat checkpoint exists. -/
def checkpointExists (path : String) : IO Bool := do
  let (snapshot, strict) ← resolveCheckpointPath path
  -- A published but damaged snapshot must enter the resume/error path, rather
  -- than being mistaken for a fresh training directory.
  if strict then pure true else checkpointExistsAt snapshot

private def saveScalarUInt64 (value : UInt64) (path : String) : IO Unit := do
  let t := Tensor.fromInt64Array #[value.toInt64]
  Tensor.saveTensor t path

private def loadScalarUInt64 (path : String) : IO UInt64 := do
  let t ← Tensor.loadTensorExact ⟨#[1], .Int64, .CPU⟩ path
  let vals ← Tensor.toUInt64Array t
  match vals[0]? with
  | some v => pure v
  | none => throw <| IO.userError s!"Missing scalar value in {path}"

private def saveScalarFloat (value : Float) (path : String) : IO Unit := do
  let t := Tensor.full #[] value
  Tensor.saveTensor t path

private def loadScalarFloat (path : String) : IO Float := do
  let t ← Tensor.loadTensorExact ⟨#[], .Float32, .CPU⟩ path
  pure (Tensor.item t)

private structure MuonStateLayout where
  step : Nat
  momentum : Bool
  secondMomentShape? : Option Shape
  deriving Lean.ToJson, Lean.FromJson

/-- TensorStruct does not serialize scalar counters or Option presence. Persist
them separately so loading can reconstruct the exact optimizer traversal. -/
private structure CheckpointStateMetadata where
  version : Nat := 1
  trainingMetadata : Bool
  step : UInt64
  optimStep : UInt64
  adamCount : Nat
  bestValLossBits : UInt64
  baseLrBits : UInt64
  weightDecayBits : UInt64
  adamSteps : Array Nat
  muonStates : Array MuonStateLayout
  deriving Lean.ToJson, Lean.FromJson

private def muonStateLayout {s : Shape} (state : NorMuon.ParamState ⟨s, .Float32, device⟩) : MuonStateLayout := {
  step := state.step
  momentum := state.momentumBuffer.isSome
  secondMomentShape? := state.secondMoment.map (·.runtimeShape)
}

private def checkpointStateMetadata {cfg : moddedGpt.Config} (ckpt : Checkpoint (device := device) cfg)
    (trainingMetadata : Bool) :
    CheckpointStateMetadata := Id.run do
  let state := ckpt.optState.dualState
  let adamSteps := #[state.embed.step] ++ state.valueEmbeds.map (·.step) ++
    #[state.lmHead.step, state.scalars.step]
  let mut muonStates := #[muonStateLayout state.smearGate]
  for block in state.blocks do
    if let some attn := block.attn then
      muonStates := muonStates ++ #[muonStateLayout attn.wQ, muonStateLayout attn.wK,
        muonStateLayout attn.wV, muonStateLayout attn.wO]
    muonStates := muonStates ++ #[muonStateLayout block.cFc, muonStateLayout block.cProj]
  return {
    trainingMetadata
    step := ckpt.step, optimStep := ckpt.optState.step, adamCount := ckpt.optState.adamState.fst.count
    bestValLossBits := ckpt.bestValLoss.toBits, baseLrBits := ckpt.optState.baseLr.toBits
    weightDecayBits := ckpt.optState.weightDecay.toBits, adamSteps, muonStates
  }

private def restoreMuonState {s : Shape} (layouts : Array MuonStateLayout) (index : Nat) :
    Except String (NorMuon.ParamState ⟨s, .Float32, device⟩) := do
  let some layout := layouts[index]?
    | throw "Missing checkpoint Muon state layout"
  if let some shape := layout.secondMomentShape? then
    if !shape.isEmpty then
      throw "Checkpoint Muon second moment must be scalar"
  pure {
    momentumBuffer := if layout.momentum then some (Tensor.zeros s (device := device)) else none
    secondMoment := layout.secondMomentShape?.map fun _ => Tensor.zeros #[] (device := device)
    step := layout.step
  }

private def restoreDualStateLayout {cfg : moddedGpt.Config} (state : DualParamState (device := device) cfg)
    (metadata : CheckpointStateMetadata) : Except String (DualParamState (device := device) cfg) := do
  if metadata.version != 1 then
    throw s!"Unsupported checkpoint state metadata version: {metadata.version}"
  let expectedMuonCount := state.blocks.foldl
    (fun count block => count + if block.attn.isSome then 6 else 2) 1
  if metadata.adamSteps.size != state.valueEmbeds.size + 3 ||
      metadata.muonStates.size != expectedMuonCount then
    throw "Checkpoint optimizer layout does not match the model configuration"
  let embed := { state.embed with step := metadata.adamSteps[0]! }
  let valueEmbeds := state.valueEmbeds.mapIdx fun i value =>
    { value with step := metadata.adamSteps[i + 1]! }
  let lmHead := { state.lmHead with step := metadata.adamSteps[state.valueEmbeds.size + 1]! }
  let scalars := { state.scalars with step := metadata.adamSteps[state.valueEmbeds.size + 2]! }
  let smearGate ← restoreMuonState (s := #[1, 12]) metadata.muonStates 0
  let mut index := 1
  let mut blocks := #[]
  for block in state.blocks do
    let attn ← match block.attn with
      | none => pure none
      | some _ => do
          let wQ ← restoreMuonState (s := #[cfg.nHead * cfg.headDim, cfg.modelDim]) metadata.muonStates index
          let wK ← restoreMuonState (s := #[cfg.nHead * cfg.headDim, cfg.modelDim]) metadata.muonStates (index + 1)
          let wV ← restoreMuonState (s := #[cfg.nHead * cfg.headDim, cfg.modelDim]) metadata.muonStates (index + 2)
          let wO ← restoreMuonState (s := #[cfg.modelDim, cfg.nHead * cfg.headDim]) metadata.muonStates (index + 3)
          pure (some ({ wQ, wK, wV, wO } : MuonAttnState (device := device) cfg))
    if block.attn.isSome then index := index + 4
    let cFc ← restoreMuonState (s := #[4 * cfg.modelDim, cfg.modelDim]) metadata.muonStates index
    let cProj ← restoreMuonState (s := #[4 * cfg.modelDim, cfg.modelDim]) metadata.muonStates (index + 1)
    index := index + 2
    blocks := blocks.push { attn, cFc, cProj }
  pure { embed, valueEmbeds, lmHead, scalars, smearGate, blocks }

private def loadCheckpointStateMetadata (path : String) (required : Bool) :
    IO (Option CheckpointStateMetadata) := do
  let statePath : System.FilePath := s!"{path}/state_meta.json"
  if !(← statePath.pathExists) then
    if required then throw <| IO.userError s!"Missing checkpoint state metadata: {statePath}"
    return none
  let content ← IO.FS.readFile statePath
  let parsed := Lean.Json.parse content >>= (Lean.fromJson? (α := CheckpointStateMetadata))
  match parsed with
  | .ok metadata =>
      if metadata.version != 1 then
        throw <| IO.userError s!"Unsupported checkpoint state metadata version: {metadata.version}"
      if metadata.trainingMetadata && !(← (checkpointMetadataFile path).pathExists) then
        throw <| IO.userError s!"Missing checkpoint training metadata: {path}"
      return some metadata
  | .error err => throw <| IO.userError s!"Invalid checkpoint state metadata: {err}"

private def saveCheckpointMetadata (path : String) (metadata : CheckpointMetadata) : IO Unit := do
  IO.FS.writeFile (checkpointMetadataFile path) (Lean.toJson metadata).pretty

private def loadCheckpointMetadataAt (path : String) (quiet : Bool := false) (strict : Bool := false)
    : IO (Option CheckpointMetadata) := do
  let metaPath := checkpointMetadataFile path
  if !(← metaPath.pathExists) then
    return none
  try
    let content ← IO.FS.readFile metaPath
    match Lean.Json.parse content with
    | .error err =>
      if strict then throw <| IO.userError s!"Invalid checkpoint metadata {metaPath}: {err}"
      if !quiet then
        IO.eprintln s!"Warning: failed to parse checkpoint metadata {metaPath}: {err}"
      return none
    | .ok json =>
      match (Lean.fromJson? json : Except String CheckpointMetadata) with
      | .error err =>
        if strict then throw <| IO.userError s!"Invalid checkpoint metadata {metaPath}: {err}"
        if !quiet then
          IO.eprintln s!"Warning: invalid checkpoint metadata in {metaPath}: {err}"
        return none
      | .ok metadata => return some metadata
  catch e =>
    if strict then throw e
    if !quiet then
      IO.eprintln s!"Warning: failed to read checkpoint metadata {metaPath}: {e}"
    return none

/-- Load optional training metadata from one committed snapshot or a legacy directory. -/
def loadCheckpointMetadata (path : String) (quiet : Bool := false) :
    IO (Option CheckpointMetadata) := do
  try
    let (snapshot, strict) ← resolveCheckpointPath path
    if strict then let _ ← loadCheckpointStateMetadata snapshot true
    loadCheckpointMetadataAt snapshot quiet strict
  catch e =>
    if !quiet then IO.eprintln s!"Failed to load checkpoint metadata from {path}: {e}"
    return none

private def captureDataCursor (gen : DistributedDataGenerator) : DataCursor := {
  trainPathIdx := gen.trainPathIdx
  globalStep := gen.globalStep
  epoch := gen.iterator.epoch
  batchCount := gen.iterator.batchCount
  bosCurrentPos := gen.iterator.shard.bosFinder.currentPos
  streamCursor? := some gen.cursor
}

private def restoreDataCursor (gen : DistributedDataGenerator) (cursor : DataCursor)
    : IO DistributedDataGenerator := do
  if gen.worldSize != 1 then
    throw <| IO.userError "Exact distributed data resume requires per-rank cursors; this checkpoint stores only rank 0"
  let some streamCursor := cursor.streamCursor?
    | throw <| IO.userError "Legacy data cursor used truncated, unshuffled shards; restart the data stream explicitly instead of exact resume"
  gen.restoreCursor streamCursor

private def mergedResumeHyperparameters (requested saved : Hyperparameters) : Hyperparameters :=
  { saved with
    -- Allow extending/changing horizon while keeping schedule semantics identical.
    numIterations := requested.numIterations
    extensionIterations := requested.extensionIterations
  }

private def mkCheckpointMetadata (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (totalTokens : UInt64) (dataCursor? : Option DataCursor) : CheckpointMetadata := {
  version := 1
  modelConfigRepr := s!"{repr cfg}"
  modelConfig? := some cfg
  hyperparameters := hp
  totalTokens := totalTokens
  dataCursor? := dataCursor?
}

/-- Publish parameters, optimizer tensors/layout, counters, and training metadata
as one immutable snapshot. Existing readers retain the previously published directory. -/
def saveCheckpoint {cfg : moddedGpt.Config} (ckpt : Checkpoint (device := device) cfg) (path : String)
    (metadata? : Option CheckpointMetadata := none)
    (writeExtra? : Option (String → IO Unit) := none)
    : IO Unit := do
  checkpoint.publishSnapshot path "CURRENT" fun snapshot => do
    checkpoint.saveParams ckpt.params snapshot "param"
    checkpoint.saveParams ckpt.optState.adamState snapshot "optim_adam"
    checkpoint.saveParams ckpt.optState.dualState snapshot "optim_dual"
    -- Keep the legacy scalar files for checkpoint inspection tools. The state
    -- sidecar restores exact Float bits and all optimizer-local counters.
    saveScalarUInt64 ckpt.step s!"{snapshot}/step.pt"
    saveScalarUInt64 ckpt.optState.step s!"{snapshot}/opt_step.pt"
    saveScalarUInt64 ckpt.optState.adamState.fst.count.toUInt64 s!"{snapshot}/adam_count.pt"
    saveScalarFloat ckpt.bestValLoss s!"{snapshot}/best_val_loss.pt"
    saveScalarFloat ckpt.optState.baseLr s!"{snapshot}/base_lr.pt"
    saveScalarFloat ckpt.optState.weightDecay s!"{snapshot}/weight_decay.pt"
    IO.FS.writeFile (s!"{snapshot}/state_meta.json" : System.FilePath)
      (Lean.toJson (checkpointStateMetadata ckpt metadata?.isSome)).pretty
    if let some metadata := metadata? then
      saveCheckpointMetadata snapshot metadata
    if let some writeExtra := writeExtra? then
      writeExtra snapshot

  IO.println s!"Saving checkpoint to {path} at step {ckpt.step}"

  let numParams := TensorStruct.fold (fun {s} _t acc => acc + s.shape.foldl (· * ·) 1) 0 ckpt.params
  let numTensors := TensorStruct.fold (fun {_s} _t acc => acc + 1) 0 ckpt.params

  IO.println s!"  Parameters: {numTensors} tensors, {numParams} elements"
  IO.println s!"  Best validation loss: {ckpt.bestValLoss}"

private def loadCheckpointAt (cfg : moddedGpt.Config) (path : String) (quiet : Bool := false)
    (strict : Bool := false) (device : Device := .CPU)
    : IO (Option (Checkpoint (device := device) cfg)) := do
  if !(← checkpointExistsAt path) then
    return none

  try
    -- Templates supply the static structure and tensor shapes for deserialization.
    let templateParams ← ModdedGPTParams.init cfg (device := device)
    let templateOpt := OptimizerState.init cfg templateParams
    let stateMetadata? ← loadCheckpointStateMetadata path strict

    let params ← checkpoint.loadParams templateParams path "param"
    let loadedAdamState ← checkpoint.loadParams templateOpt.adamState path "optim_adam"
      (asParameters := false)
    let loadedDualState ← match stateMetadata? with
      | some metadata => do
          let dualTemplate ← match restoreDualStateLayout templateOpt.dualState metadata with
            | .ok state => pure state
            | .error err => throw <| IO.userError err
          checkpoint.loadParams dualTemplate path "optim_dual" (asParameters := false)
      | none => do
          -- Only legacy flat checkpoints lack layout metadata. Their optional
          -- Muon buffers/counters were not recorded; retain the old best-effort fallback.
          try checkpoint.loadParams templateOpt.dualState path "optim_dual" (asParameters := false)
          catch _ => pure (initDualParamState cfg params)

    let (step, optStep, adamCount, bestValLoss, baseLr, weightDecay) ← match stateMetadata? with
      | some metadata => pure (metadata.step, metadata.optimStep, metadata.adamCount,
          Float.ofBits metadata.bestValLossBits, Float.ofBits metadata.baseLrBits,
          Float.ofBits metadata.weightDecayBits)
      | none => do
          let step ← loadScalarUInt64 s!"{path}/step.pt"
          let optStep ← loadScalarUInt64 s!"{path}/opt_step.pt"
          let adamCount := (← loadScalarUInt64 s!"{path}/adam_count.pt").toNat
          let bestValLoss ← loadScalarFloat s!"{path}/best_val_loss.pt"
          let baseLr ← loadScalarFloat s!"{path}/base_lr.pt"
          let weightDecay ← loadScalarFloat s!"{path}/weight_decay.pt"
          pure (step, optStep, adamCount, bestValLoss, baseLr, weightDecay)

    let adamState := {
      loadedAdamState with
      fst := { loadedAdamState.fst with count := adamCount }
    }
    let optState : OptimizerState (device := device) cfg := {
      adamState := adamState
      dualState := loadedDualState
      step := optStep
      baseLr := baseLr
      weightDecay := weightDecay
    }

    return some {
      params := params
      optState := optState
      step := step
      bestValLoss := bestValLoss
    }
  catch e =>
    if !quiet then
      IO.eprintln s!"Failed to load checkpoint from {path}: {e}"
    return none

/-- Load all checkpoint tensors and scalars from one resolved snapshot, or a
legacy flat directory when no CURRENT pointer exists. -/
def loadCheckpoint (cfg : moddedGpt.Config) (path : String) (quiet : Bool := false) (device : Device := .CPU) :
    IO (Option (Checkpoint (device := device) cfg)) := do
  try
    let (snapshot, strict) ← resolveCheckpointPath path
    loadCheckpointAt (device := device) cfg snapshot quiet strict
  catch e =>
    if !quiet then IO.eprintln s!"Failed to load checkpoint from {path}: {e}"
    return none

private def loadCheckpointForResumeAt (cfg : moddedGpt.Config) (path : String) (quiet : Bool)
    (strict : Bool) (device : Device := .CPU)
    : IO (Option (Checkpoint (device := device) cfg × Option CheckpointMetadata)) := do
  let rawMetadata? ← loadCheckpointMetadataAt path quiet strict
  let metadata? ←
    match rawMetadata? with
    | some metadata =>
      if metadata.version != 1 then
        if strict then throw <| IO.userError s!"Unsupported checkpoint metadata version: {metadata.version}"
        if !quiet then
          IO.eprintln s!"Warning: unsupported checkpoint metadata version {metadata.version} in {path}; ignoring metadata."
        pure none
      else
        pure (some metadata)
    | none => pure none
  match metadata? with
  | some metadata =>
    let savedModelCfgRepr :=
      match metadata.modelConfig? with
      | some savedCfg => s!"{repr savedCfg}"
      | none => metadata.modelConfigRepr
    if savedModelCfgRepr != s!"{repr cfg}" then
      if !quiet then
        IO.eprintln s!"Checkpoint config mismatch for {path}; refusing resume because saved model config differs from current config."
      return none
  | none => pure ()
  match ← loadCheckpointAt (device := device) cfg path quiet strict with
  | none => return none
  | some ckpt => return some (ckpt, metadata?)

/-- Pin the snapshot once before reading both training metadata and tensors. -/
def loadCheckpointForResume (cfg : moddedGpt.Config) (path : String) (quiet : Bool := false) (device : Device := .CPU) :
    IO (Option (Checkpoint (device := device) cfg × Option CheckpointMetadata)) := do
  try
    let (snapshot, strict) ← resolveCheckpointPath path
    loadCheckpointForResumeAt (device := device) cfg snapshot quiet strict
  catch e =>
    if !quiet then IO.eprintln s!"Failed to resume checkpoint from {path}: {e}"
    return none

/-! ## Training Loop -/

/-- Training state for the main loop -/
structure TrainState (cfg : moddedGpt.Config) where
  /-- Model parameters -/
  params : ModdedGPTParams (device := device) cfg
  /-- YaRN rotary embeddings -/
  yarn : YarnRotary (device := device) cfg.headDim cfg.maxSeqLen
  /-- Optimizer state -/
  optState : OptimizerState (device := device) cfg
  /-- Data generator -/
  dataGen : DistributedDataGenerator
  /-- Validation data -/
  valData : Option (Σ n, DataShard n)
  /-- Current step -/
  step : UInt64
  /-- Best validation loss -/
  bestValLoss : Float
  /-- Accumulated tokens -/
  totalTokens : UInt64
  /-- Training start time -/
  startTime : UInt64

/-- Initialize training state -/
def TrainState.init (cfg : moddedGpt.Config) (dataConfig : DataLoader.Config)
    (hp : Hyperparameters)
    (_distributed : Bool) (_worldSize : UInt64)
    (device : Device := Device.CPU)
    (lr : Float := 0.023) (weightDecay : Float := 0.0)
    : IO (TrainState (device := device) cfg) := do
  let params ← ModdedGPTParams.init cfg (device := device)
  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen cfg.ropeBase (device := device)

  -- Initialize optimizer state from moved parameters so moment buffers match device
  let optState := OptimizerState.init cfg params lr weightDecay

  let initialBatchSize := hp.deviceBatchSize
  let initialSeqLen := hp.maxSeqLen
  let dataGen ← DistributedDataGenerator.init dataConfig initialBatchSize initialSeqLen
  let valData ←
    match dataConfig.valPath with
    | none => pure none
    | some valPath =>
      try
        let shard ← loadValidationData valPath cfg.maxSeqLen dataConfig.bosToken
        pure (some shard)
      catch e =>
        IO.eprintln s!"Warning: failed to load validation data from {valPath}: {e}"
        pure none

  let startTime ← IO.monoMsNow

  return {
    params := params
    yarn := yarn
    optState := optState
    dataGen := dataGen
    valData := valData
    step := 0
    bestValLoss := 1e30  -- Very large number instead of inf
    totalTokens := 0
    startTime := startTime.toUInt64
  }

/-- Log training progress -/
def logProgress (state : TrainState (device := device) cfg) (result : StepResult)
    (hp : Hyperparameters) : IO Unit := do
  if state.step % hp.logInterval == 0 then
    let nowMs ← IO.monoMsNow
    let elapsed := nowMs - state.startTime.toNat
    let elapsedSec := elapsed.toFloat / 1000.0
    let tokensPerSec := state.totalTokens.toFloat / elapsedSec
    let batchSize := hp.deviceBatchSize
    let seqLen := hp.maxSeqLen
    let lr := getLearningRate state.step hp

    IO.println s!"Step {state.step}: loss={result.loss} lr={lr} batch={batchSize} seq={seqLen} tok/s={tokensPerSec} time={result.timeMs}ms"

/-- Run validation and log results -/
def runValidation (state : TrainState (device := device) cfg) (hp : Hyperparameters)
    : IO (TrainState (device := device) cfg) := do
  match state.valData with
  | none => return state
  | some valData =>
    let batchSize := hp.deviceBatchSize
    let seqLen := hp.maxSeqLen
    let valResult ← validate state.params state.yarn valData batchSize seqLen

    IO.println s!"Validation: loss={valResult.loss} time={valResult.timeMs}ms"

    let newBest := valResult.loss < state.bestValLoss
    if newBest then
      IO.println s!"  New best validation loss!"

    return { state with
      bestValLoss := if newBest then valResult.loss else state.bestValLoss
    }

/-- Distributed training step with gradient accumulation over micro-batches. -/
def trainStepDistributedAccum {cfg : moddedGpt.Config} {batch seq : UInt64}
    (params : ModdedGPTParams (device := device) cfg)
    (yarn : YarnRotary (device := device) cfg.headDim cfg.maxSeqLen)
    (microBatches : Array (Tensor { shape := #[batch, seq], dtype := .Int64, device := device } × Tensor { shape := #[batch, seq], dtype := .Int64, device := device }))
    (optState : OptimizerState (device := device) cfg)
    (hp : Hyperparameters)
    (gradClip : Float := 0.0)
    : IO (ModdedGPTParams (device := device) cfg × OptimizerState (device := device) cfg × StepResult) := do
  if microBatches.isEmpty then
    return (params, optState, { loss := 0.0, gradNorm := 0.0, tokensProcessed := 0, timeMs := 0.0 })
  let startTime ← IO.monoMsNow
  let isDistributed ← dist.isInitialized
  let worldSize ← if isDistributed then dist.getWorldSize else pure 1
  let params ← TensorStruct.makeLeafParams params
  TensorStruct.zeroGrads params
  let microCount := microBatches.size.toUInt64
  let mut totalLoss := 0.0
  for (input, target) in microBatches do
    let lossT ← moddedGpt.loss params yarn input target true
    let scaledLoss := Tensor.divScalar lossT microCount.toFloat
    Tensor.backwardLoss scaledLoss
    totalLoss := totalLoss + Tensor.item lossT
  if gradClip > 0 then
    let _ ← Tensor.clipGradNorm params.embed gradClip
    let _ ← Tensor.clipGradNorm params.smearGate.weight gradClip
    for ve in params.valueEmbeds do
      let _ ← Tensor.clipGradNorm ve gradClip
    for block in params.blocks do
      match block.attn with
      | some attn =>
        let _ ← Tensor.clipGradNorm attn.wQ gradClip
        let _ ← Tensor.clipGradNorm attn.wK gradClip
        let _ ← Tensor.clipGradNorm attn.wV gradClip
        let _ ← Tensor.clipGradNorm attn.wO gradClip
      | none => pure ()
      let _ ← Tensor.clipGradNorm block.mlp.cFc gradClip
      let _ ← Tensor.clipGradNorm block.mlp.cProj gradClip
    let _ ← Tensor.clipGradNorm params.lmHead.weight gradClip
    let _ ← Tensor.clipGradNorm params.scalars.values gradClip
  let grads ← TensorStruct.grads params
  let batchScale := batchLrScale hp
  let dScale := dModelLrScale cfg
  let lrEmbed := getLearningRate optState.step hp (hp.embeddingLr * batchScale * dScale)
  let lrUnembed := getLearningRate optState.step hp (hp.unembeddingLr * batchScale * dScale)
  let lrMatrix := getLearningRate optState.step hp (hp.matrixLr * batchScale)
  let muMomentum := getMuonMomentum optState.step hp

  let adamCfgBase : DistAdam.Config := {
    lr := 1.0
    beta1 := hp.adamBeta1
    beta2 := hp.adamBeta2
    eps := 1e-10
    weightDecay := hp.adamWeightDecay
    distributed := isDistributed
  }
  let muonCfg : NorMuon.Config := {
    lr := lrMatrix
    weightDecay := 0.0
    momentum := muMomentum
    beta2 := 0.95
    numIters := 5
    distributed := isDistributed
    worldSize := worldSize
  }

  let runAdamStep : {s : Shape} → (p : Tensor { shape := s, dtype := .Float32, device := device }) → (g : Tensor { shape := s, dtype := .Float32, device := device }) → DistAdam.ParamState ⟨s, .Float32, device⟩ → Float →
      IO (Tensor { shape := s, dtype := .Float32, device := device } × DistAdam.ParamState ⟨s, .Float32, device⟩) :=
    fun {_} p g st lr => do
      DistAdam.stepDistributed p g st { adamCfgBase with lr := lr } 1.0 1.0

  let (embed', embedState') ← runAdamStep
    params.embed grads.embed optState.dualState.embed lrEmbed

  let mut newValueEmbeds : Array (Tensor { shape := #[cfg.vocabSize, cfg.modelDim], dtype := .Float32, device := device }) := #[]
  let mut newValueEmbedStates : Array (DistAdam.ParamState ⟨#[cfg.vocabSize, cfg.modelDim], .Float32, device⟩) := #[]
  for i in [:params.valueEmbeds.size] do
    let ve := params.valueEmbeds[i]!
    let veGrad := grads.valueEmbeds[i]!
    let veState := optState.dualState.valueEmbeds[i]?.getD (DistAdam.initParamState ve)
    let (ve', veState') ← runAdamStep ve veGrad veState lrEmbed
    newValueEmbeds := newValueEmbeds.push ve'
    newValueEmbedStates := newValueEmbedStates.push veState'

  let (lmHeadW', lmHeadState') ← runAdamStep
    params.lmHead.weight grads.lmHead.weight optState.dualState.lmHead lrUnembed
  let (scalars', scalarState') ← runAdamStep
    params.scalars.values grads.scalars.values optState.dualState.scalars lrUnembed

  let (smearGateWs, smearGateStates) ← NorMuon.stepDistributedGroup
    #[params.smearGate.weight]
    #[grads.smearGate.weight]
    #[optState.dualState.smearGate]
    muonCfg
  let smearGateW' := smearGateWs[0]?.getD params.smearGate.weight
  let smearGateState' := smearGateStates[0]?.getD optState.dualState.smearGate

  let mut blockStatesIn : Array (MuonBlockState (device := device) cfg) := #[]
  let mut cFcParams : Array (Tensor { shape := #[4 * cfg.modelDim, cfg.modelDim], dtype := .Float32, device := device }) := #[]
  let mut cFcGrads : Array (Tensor { shape := #[4 * cfg.modelDim, cfg.modelDim], dtype := .Float32, device := device }) := #[]
  let mut cFcStates : Array (NorMuon.ParamState ⟨#[4 * cfg.modelDim, cfg.modelDim], .Float32, device⟩) := #[]
  let mut cProjParams : Array (Tensor { shape := #[4 * cfg.modelDim, cfg.modelDim], dtype := .Float32, device := device }) := #[]
  let mut cProjGrads : Array (Tensor { shape := #[4 * cfg.modelDim, cfg.modelDim], dtype := .Float32, device := device }) := #[]
  let mut cProjStates : Array (NorMuon.ParamState ⟨#[4 * cfg.modelDim, cfg.modelDim], .Float32, device⟩) := #[]

  let mut qParams : Array (Tensor { shape := #[cfg.nHead * cfg.headDim, cfg.modelDim], dtype := .Float32, device := device }) := #[]
  let mut qGrads : Array (Tensor { shape := #[cfg.nHead * cfg.headDim, cfg.modelDim], dtype := .Float32, device := device }) := #[]
  let mut qStates : Array (NorMuon.ParamState ⟨#[cfg.nHead * cfg.headDim, cfg.modelDim], .Float32, device⟩) := #[]
  let mut kParams : Array (Tensor { shape := #[cfg.nHead * cfg.headDim, cfg.modelDim], dtype := .Float32, device := device }) := #[]
  let mut kGrads : Array (Tensor { shape := #[cfg.nHead * cfg.headDim, cfg.modelDim], dtype := .Float32, device := device }) := #[]
  let mut kStates : Array (NorMuon.ParamState ⟨#[cfg.nHead * cfg.headDim, cfg.modelDim], .Float32, device⟩) := #[]
  let mut vParams : Array (Tensor { shape := #[cfg.nHead * cfg.headDim, cfg.modelDim], dtype := .Float32, device := device }) := #[]
  let mut vGrads : Array (Tensor { shape := #[cfg.nHead * cfg.headDim, cfg.modelDim], dtype := .Float32, device := device }) := #[]
  let mut vStates : Array (NorMuon.ParamState ⟨#[cfg.nHead * cfg.headDim, cfg.modelDim], .Float32, device⟩) := #[]
  let mut oParams : Array (Tensor { shape := #[cfg.modelDim, cfg.nHead * cfg.headDim], dtype := .Float32, device := device }) := #[]
  let mut oGrads : Array (Tensor { shape := #[cfg.modelDim, cfg.nHead * cfg.headDim], dtype := .Float32, device := device }) := #[]
  let mut oStates : Array (NorMuon.ParamState ⟨#[cfg.modelDim, cfg.nHead * cfg.headDim], .Float32, device⟩) := #[]

  for i in [:params.blocks.size] do
    let block := params.blocks[i]!
    let blockGrad := grads.blocks[i]!
    let defaultBlockState : MuonBlockState (device := device) cfg := {
      attn := block.attn.map fun attn =>
        {
          wQ := NorMuon.initParamState attn.wQ
          wK := NorMuon.initParamState attn.wK
          wV := NorMuon.initParamState attn.wV
          wO := NorMuon.initParamState attn.wO
        }
      cFc := NorMuon.initParamState block.mlp.cFc
      cProj := NorMuon.initParamState block.mlp.cProj
    }
    let blockState := optState.dualState.blocks[i]?.getD defaultBlockState
    blockStatesIn := blockStatesIn.push blockState

    cFcParams := cFcParams.push block.mlp.cFc
    cFcGrads := cFcGrads.push blockGrad.mlp.cFc
    cFcStates := cFcStates.push blockState.cFc
    cProjParams := cProjParams.push block.mlp.cProj
    cProjGrads := cProjGrads.push blockGrad.mlp.cProj
    cProjStates := cProjStates.push blockState.cProj

    match block.attn, blockGrad.attn with
    | some attn, some attnGrad =>
      let attnState := blockState.attn.getD {
        wQ := NorMuon.initParamState attn.wQ
        wK := NorMuon.initParamState attn.wK
        wV := NorMuon.initParamState attn.wV
        wO := NorMuon.initParamState attn.wO
      }
      qParams := qParams.push attn.wQ
      qGrads := qGrads.push attnGrad.wQ
      qStates := qStates.push attnState.wQ
      kParams := kParams.push attn.wK
      kGrads := kGrads.push attnGrad.wK
      kStates := kStates.push attnState.wK
      vParams := vParams.push attn.wV
      vGrads := vGrads.push attnGrad.wV
      vStates := vStates.push attnState.wV
      oParams := oParams.push attn.wO
      oGrads := oGrads.push attnGrad.wO
      oStates := oStates.push attnState.wO
    | _, _ => pure ()

  let (qParams', qStates') ← NorMuon.stepDistributedGroup qParams qGrads qStates muonCfg
  let (kParams', kStates') ← NorMuon.stepDistributedGroup kParams kGrads kStates muonCfg
  let (vParams', vStates') ← NorMuon.stepDistributedGroup vParams vGrads vStates muonCfg
  let (oParams', oStates') ← NorMuon.stepDistributedGroup oParams oGrads oStates muonCfg
  let (cFcParams', cFcStates') ← NorMuon.stepDistributedGroup cFcParams cFcGrads cFcStates muonCfg
  let (cProjParams', cProjStates') ← NorMuon.stepDistributedGroup cProjParams cProjGrads cProjStates muonCfg

  let mut newBlocks : Array (Block (device := device) cfg.modelDim cfg.headDim cfg.nHead) := #[]
  let mut newBlockStates : Array (MuonBlockState (device := device) cfg) := #[]
  let mut attnCursor : Nat := 0
  for i in [:params.blocks.size] do
    let block := params.blocks[i]!
    let blockGrad := grads.blocks[i]!
    let defaultBlockState : MuonBlockState (device := device) cfg := {
      attn := block.attn.map fun attn =>
        {
          wQ := NorMuon.initParamState attn.wQ
          wK := NorMuon.initParamState attn.wK
          wV := NorMuon.initParamState attn.wV
          wO := NorMuon.initParamState attn.wO
        }
      cFc := NorMuon.initParamState block.mlp.cFc
      cProj := NorMuon.initParamState block.mlp.cProj
    }
    let blockState := blockStatesIn[i]?.getD defaultBlockState
    let cFc' := cFcParams'[i]?.getD block.mlp.cFc
    let cProj' := cProjParams'[i]?.getD block.mlp.cProj
    let cFcState' := cFcStates'[i]?.getD (NorMuon.initParamState cFc')
    let cProjState' := cProjStates'[i]?.getD (NorMuon.initParamState cProj')
    let (attn', attnState', nextAttnCursor) :=
      match block.attn, blockGrad.attn with
      | some attn, some _ =>
        if attnCursor < qParams'.size then
          let wQ' := qParams'[attnCursor]?.getD attn.wQ
          let wK' := kParams'[attnCursor]?.getD attn.wK
          let wV' := vParams'[attnCursor]?.getD attn.wV
          let wO' := oParams'[attnCursor]?.getD attn.wO
          let wQState' := qStates'[attnCursor]?.getD (NorMuon.initParamState wQ')
          let wKState' := kStates'[attnCursor]?.getD (NorMuon.initParamState wK')
          let wVState' := vStates'[attnCursor]?.getD (NorMuon.initParamState wV')
          let wOState' := oStates'[attnCursor]?.getD (NorMuon.initParamState wO')
          let newAttn : CausalSelfAttention (device := device) cfg.modelDim cfg.headDim cfg.nHead := {
            wQ := wQ'
            wK := wK'
            wV := wV'
            wO := wO'
          }
          let newAttnState : MuonAttnState (device := device) cfg := {
            wQ := wQState'
            wK := wKState'
            wV := wVState'
            wO := wOState'
          }
          (some newAttn, some newAttnState, attnCursor + 1)
        else
          (block.attn, blockState.attn, attnCursor)
      | none, _ => (none, none, attnCursor)
      | _, none => (block.attn, blockState.attn, attnCursor)
    attnCursor := nextAttnCursor
    let newBlock : Block (device := device) cfg.modelDim cfg.headDim cfg.nHead := {
      attn := attn'
      mlp := { cFc := cFc', cProj := cProj' }
    }
    let newBlockState : MuonBlockState (device := device) cfg := {
      attn := attnState'
      cFc := cFcState'
      cProj := cProjState'
    }
    newBlocks := newBlocks.push newBlock
    newBlockStates := newBlockStates.push newBlockState

  let newParams : ModdedGPTParams (device := device) cfg := {
    params with
    embed := embed'
    valueEmbeds := newValueEmbeds
    lmHead := { params.lmHead with weight := lmHeadW' }
    scalars := { params.scalars with values := scalars' }
    smearGate := { params.smearGate with weight := smearGateW' }
    blocks := newBlocks
  }
  let legacyAdamState := {
    optState.adamState with
    fst := { optState.adamState.fst with count := optState.adamState.fst.count + 1 }
  }
  let newDualState : DualParamState (device := device) cfg := {
    embed := embedState'
    valueEmbeds := newValueEmbedStates
    lmHead := lmHeadState'
    scalars := scalarState'
    smearGate := smearGateState'
    blocks := newBlockStates
  }
  let newParams ← TensorStruct.makeLeafParams newParams
  let endTime ← IO.monoMsNow
  let timeMs := (endTime - startTime).toFloat
  let result : StepResult := {
    loss := totalLoss / microCount.toFloat
    gradNorm := 0.0
    tokensProcessed := batch * seq * worldSize * microCount
    timeMs := timeMs
  }
  let newOptState : OptimizerState (device := device) cfg := {
    adamState := legacyAdamState
    dualState := newDualState
    step := optState.step + 1
    baseLr := optState.baseLr
    weightDecay := optState.weightDecay
  }
  return (newParams, newOptState, result)

/-- Main training loop -/
def trainLoop (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (state : TrainState (device := device) cfg) (checkpointDir : String := "checkpoints/modded")
    : IO (TrainState (device := device) cfg) := do
  let totalSteps := hp.numIterations + hp.extensionIterations
  let mut state := state
  let startStep := state.optState.step

  IO.FS.createDirAll ⟨checkpointDir⟩

  if startStep >= totalSteps then
    IO.println s!"Training already complete at step {state.step} (target {totalSteps})"
    return state

  for step in [startStep.toNat:totalSteps.toNat] do
    let stepU := step.toUInt64
    let batchSize := hp.deviceBatchSize
    let seqLen := hp.maxSeqLen
    let gradAccum := effectiveGradAccumSteps hp 1
    let iterator ← state.dataGen.iterator.updateParams batchSize seqLen
    let dataGen := { state.dataGen with iterator }
    let mut microBatches : Array (Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := device } × Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := device }) := #[]
    let mut newDataGen := dataGen
    let device := state.params.embed.device
    for _ in [:gradAccum.toNat] do
      let (maybeBatch, newerDataGen) ← newDataGen.nextBatchGPT
      newDataGen := newerDataGen
      match maybeBatch with
      | none => pure ()
      | some (inputDyn, targetDyn) =>
        let input := (inputDyn.reshapeChecked #[batchSize, seqLen]).to device
        let target := (targetDyn.reshapeChecked #[batchSize, seqLen]).to device
        microBatches := microBatches.push (input, target)
    if microBatches.isEmpty then
      state := { state with dataGen := newDataGen }
      continue
    let (newParams, newOptState, result) ← trainStepDistributedAccum state.params state.yarn
      microBatches state.optState hp
    state := { state with
      params := newParams
      optState := newOptState
      dataGen := newDataGen
      step := stepU
      totalTokens := state.totalTokens + result.tokensProcessed
    }
    logProgress state result hp
    if stepU % hp.valInterval == 0 && stepU > 0 then
      state ← runValidation state hp
    if stepU % hp.checkpointInterval == 0 && stepU > 0 then
      let ckpt : Checkpoint (device := device) cfg := {
        params := state.params
        optState := state.optState
        step := stepU
        bestValLoss := state.bestValLoss
      }
      let metadata := mkCheckpointMetadata cfg hp state.totalTokens (some (captureDataCursor state.dataGen))
      let stepPath := s!"{checkpointDir}/step_{stepU}.ckpt"
      let latestPath := s!"{checkpointDir}/latest.ckpt"
      saveCheckpoint ckpt stepPath (some metadata)
      saveCheckpoint ckpt latestPath (some metadata)

  let finalCkpt : Checkpoint (device := device) cfg := {
    params := state.params
    optState := state.optState
    step := state.step
    bestValLoss := state.bestValLoss
  }
  let finalMetadata := mkCheckpointMetadata cfg hp state.totalTokens (some (captureDataCursor state.dataGen))
  saveCheckpoint finalCkpt s!"{checkpointDir}/latest.ckpt" (some finalMetadata)

  IO.println s!"Training complete! Total tokens: {state.totalTokens}"
  return state

/-! ## Distributed Training -/

/-- Synchronize model parameters across ranks -/
def syncParameters {cfg : moddedGpt.Config} (params : ModdedGPTParams (device := device) cfg) : IO (ModdedGPTParams (device := device) cfg) := do
  let isDistributed ← dist.isInitialized
  if isDistributed then
    let syncedParams ← dist.broadcastParams params
    dist.barrier
    TensorStruct.makeLeafParams syncedParams
  else
    TensorStruct.makeLeafParams params

/-- Read gradient snapshots and average them across initialized ranks.
    The returned tree is passed directly to clipping and the optimizer. -/
def syncGradients {cfg : moddedGpt.Config} (params : ModdedGPTParams (device := device) cfg) :
    IO (ModdedGPTParams (device := device) cfg) := do
  let grads ← TensorStruct.grads params
  let isDistributed ← dist.isInitialized
  if isDistributed then dist.allReduceGrads grads .avg else pure grads

/-- Clip a gradient value without modifying its parameter's gradient accumulator. -/
private def clipGradientValue {σ : StaticSpec} (gradient : Tensor σ) (maxNorm : Float) :
    Tensor σ :=
  let values := gradient.toFloat32
  let norm := Float.sqrt (Tensor.sumAll (values * values)).item
  if norm > maxNorm then
    Tensor.Preserving.mulScalar gradient (maxNorm / norm)
  else gradient

/-- Distributed training step with gradient synchronization.

    This is the key integration point for multi-GPU training:
    1. Each rank computes loss and gradients on its local batch
    2. Gradients are all-reduced (averaged) across ranks
    3. Each rank applies the same optimizer update
    4. All ranks end up with identical parameters

    This pattern is called Data Parallel (DP) training.
-/
def trainStepDistributed {cfg : moddedGpt.Config} {batch seq : UInt64}
    (params : ModdedGPTParams (device := device) cfg)
    (yarn : YarnRotary (device := device) cfg.headDim cfg.maxSeqLen)
    (input : Tensor { shape := #[batch, seq], dtype := .Int64, device := device })
    (target : Tensor { shape := #[batch, seq], dtype := .Int64, device := device })
    (optState : OptimizerState (device := device) cfg)
    (hp : Hyperparameters)
    (gradClip : Float := 0.0)
    : IO (ModdedGPTParams (device := device) cfg × OptimizerState (device := device) cfg × StepResult) := do
  let startTime ← IO.monoMsNow

  -- Check if distributed
  let isDistributed ← dist.isInitialized
  let worldSize ← if isDistributed then dist.getWorldSize else pure 1

  -- Ensure parameters are trainable leaves before backprop.
  let params ← TensorStruct.makeLeafParams params
  TensorStruct.zeroGrads params

  -- Forward pass (window pattern is in cfg)
  let lossT ← moddedGpt.loss params yarn input target true

  -- Backward pass
  Tensor.backwardLoss lossT

  -- Get loss value for logging (before sync to avoid extra comm)
  let lossVal := Tensor.item lossT

  -- Synchronize the snapshots once and clip those exact values before updating.
  let grads ← syncGradients params
  let grads := if gradClip > 0 then
    TensorStruct.map (fun gradient => clipGradientValue gradient gradClip) grads
  else grads

  -- Get current learning rate from schedule
  let lr := getLearningRate optState.step hp optState.baseLr

  -- Create optimizer with current LR
  let opt := Optim.adamw (lr := lr) (weight_decay := optState.weightDecay)

  -- Apply optimizer step
  let (newParams, newAdamState) ← Optim.step opt params grads optState.adamState

  let endTime ← IO.monoMsNow
  let timeMs := (endTime - startTime).toFloat

  -- Tokens processed = local batch * worldSize
  let result : StepResult := {
    loss := lossVal
    gradNorm := 0.0  -- Could compute from grads if needed
    tokensProcessed := batch * seq * worldSize
    timeMs := timeMs
  }

  let newOptState : OptimizerState (device := device) cfg := {
    adamState := newAdamState
    dualState := optState.dualState
    step := optState.step + 1
    baseLr := optState.baseLr
    weightDecay := optState.weightDecay
  }

  return (newParams, newOptState, result)

/-- Distributed training state with sampler -/
structure DistributedTrainState (cfg : moddedGpt.Config) extends TrainState (device := device) cfg where
  /-- Distributed sampler for data sharding -/
  sampler : Option dist.DistributedSampler := none
  /-- World size -/
  worldSize : UInt64 := 1
  /-- This rank -/
  rank : UInt64 := 0

/-- Initialize distributed training state -/
def DistributedTrainState.init (cfg : moddedGpt.Config) (dataConfig : DataLoader.Config)
    (hp : Hyperparameters)
    (device : Device := Device.CPU)
    (lr : Float := 0.023) (weightDecay : Float := 0.0)
    : IO (DistributedTrainState (device := device) cfg) := do
  -- Check distributed status
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then
      dist.getRankAndWorldSize
    else
      pure (0, 1)

  -- Initialize base training state
  let params ← ModdedGPTParams.init cfg (device := device)
  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen cfg.ropeBase (device := device)

  -- Initialize optimizer state from moved parameters so moment buffers match device
  let optState := OptimizerState.init cfg params lr weightDecay

  let initialBatchSize := hp.deviceBatchSize
  let initialSeqLen := hp.maxSeqLen
  let dataGen ← DistributedDataGenerator.init dataConfig initialBatchSize initialSeqLen
  let valData ←
    match dataConfig.valPath with
    | none => pure none
    | some valPath =>
      try
        let shard ← loadValidationData valPath cfg.maxSeqLen dataConfig.bosToken
        pure (some shard)
      catch e =>
        IO.eprintln s!"Warning: failed to load validation data from {valPath}: {e}"
        pure none

  -- Create distributed sampler if in distributed mode
  let sampler := if isDistributed then
      some (dist.DistributedSampler.create {
        datasetSize := 1000000  -- Will be updated based on actual data
        rank := rank
        worldSize := worldSize
        seed := 42
        shuffle := true
      })
    else
      none

  let startTime ← IO.monoMsNow

  return {
    params := params
    yarn := yarn
    optState := optState
    dataGen := dataGen
    valData := valData
    step := 0
    bestValLoss := 1e30
    totalTokens := 0
    startTime := startTime.toUInt64
    sampler := sampler
    worldSize := worldSize
    rank := rank
  }

/-- Distributed training loop with proper gradient sync -/
def trainLoopDistributed (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (state : DistributedTrainState (device := device) cfg) (checkpointDir : String := "checkpoints/modded")
    : IO (DistributedTrainState (device := device) cfg) := do
  let totalSteps := hp.numIterations + hp.extensionIterations
  let mut state := state
  let isMaster := state.rank == 0
  let startStep := state.optState.step

  if isMaster then
    IO.FS.createDirAll ⟨checkpointDir⟩

  if startStep >= totalSteps then
    if isMaster then
      IO.println s!"Training already complete at step {state.step} (target {totalSteps})"
    return state

  for step in [startStep.toNat:totalSteps.toNat] do
    let stepU := step.toUInt64
    let batchSize := hp.deviceBatchSize
    let seqLen := hp.maxSeqLen
    let gradAccum := effectiveGradAccumSteps hp state.worldSize
    let iterator ← state.dataGen.iterator.updateParams batchSize seqLen
    let dataGen := { state.dataGen with iterator }
    let mut microBatches : Array (Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := device } × Tensor { shape := #[batchSize, seqLen], dtype := .Int64, device := device }) := #[]
    let mut newDataGen := dataGen
    let device := state.params.embed.device
    for _ in [:gradAccum.toNat] do
      let (maybeBatch, newerDataGen) ← newDataGen.nextBatchGPT
      newDataGen := newerDataGen
      match maybeBatch with
      | none => pure ()
      | some (inputDyn, targetDyn) =>
        let input := (inputDyn.reshapeChecked #[batchSize, seqLen]).to device
        let target := (targetDyn.reshapeChecked #[batchSize, seqLen]).to device
        microBatches := microBatches.push (input, target)
    if microBatches.isEmpty then
      state := { state with dataGen := newDataGen }
      continue
    let (newParams, newOptState, result) ← trainStepDistributedAccum state.params state.yarn
      microBatches state.optState hp
    state := { state with
      params := newParams
      optState := newOptState
      dataGen := newDataGen
      step := stepU
      totalTokens := state.totalTokens + result.tokensProcessed
    }
    if isMaster then
      logProgress state.toTrainState result hp
    if stepU % hp.valInterval == 0 && stepU > 0 && isMaster then
      let baseState ← runValidation state.toTrainState hp
      state := { state with
        bestValLoss := baseState.bestValLoss
        valData := baseState.valData
      }
    if stepU % hp.checkpointInterval == 0 && stepU > 0 && isMaster then
      let ckpt : Checkpoint (device := device) cfg := {
        params := state.params
        optState := state.optState
        step := stepU
        bestValLoss := state.bestValLoss
      }
      let metadata := mkCheckpointMetadata cfg hp state.totalTokens (some (captureDataCursor state.dataGen))
      let stepPath := s!"{checkpointDir}/step_{stepU}.ckpt"
      let latestPath := s!"{checkpointDir}/latest.ckpt"
      saveCheckpoint ckpt stepPath (some metadata)
      saveCheckpoint ckpt latestPath (some metadata)

  if isMaster then
    let finalCkpt : Checkpoint (device := device) cfg := {
      params := state.params
      optState := state.optState
      step := state.step
      bestValLoss := state.bestValLoss
    }
    let finalMetadata := mkCheckpointMetadata cfg hp state.totalTokens (some (captureDataCursor state.dataGen))
    saveCheckpoint finalCkpt s!"{checkpointDir}/latest.ckpt" (some finalMetadata)
    IO.println s!"Training complete! Total tokens: {state.totalTokens}"
  return state

/-- Distributed training loop wrapper -/
def trainDistributed (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (dataConfig : DataLoader.Config) (device : Device)
    (checkpointDir : String := "checkpoints/modded")
    (resume : Option String := none)
    : IO (DistributedTrainState (device := device) cfg) := do
  -- Check if distributed
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then
      dist.getRankAndWorldSize
    else
      pure (0, 1)

  let isMaster := rank == 0

  if isMaster then
    IO.println s!"Starting training with {worldSize} GPUs"
    IO.println s!"Config: {repr cfg}"
    IO.println s!"Hyperparameters: {repr hp}"

  -- Initialize distributed training state
  let initState ← DistributedTrainState.init cfg dataConfig hp device

  -- Resolve resume path (explicit --resume takes precedence over latest in checkpointDir).
  let latestPath := s!"{checkpointDir}/latest.ckpt"
  let resumePath? ←
    match resume with
    | some path => pure (some path)
    | none =>
      if ← checkpointExists latestPath then
        pure (some latestPath)
      else
        pure none

  let (state, effectiveHp) ←
    match resumePath? with
    | none => pure (initState, hp)
    | some resumePath =>
      if isMaster then
        IO.println s!"Resuming from checkpoint: {resumePath}"
      match ← loadCheckpointForResume (device := device) cfg resumePath (!isMaster) with
      | some (ckpt, metadata?) =>
        let resumedParams ← TensorStruct.makeLeafParams ckpt.params
        let resumedOptState : OptimizerState (device := device) cfg := {
          ckpt.optState with
          adamState := ckpt.optState.adamState
          dualState := ckpt.optState.dualState
        }
        let baseState : DistributedTrainState (device := device) cfg := {
          initState with
          params := resumedParams
          optState := resumedOptState
          step := ckpt.step
          bestValLoss := ckpt.bestValLoss
        }
        let (resumedState, restoredHp) ←
          match metadata? with
          | none =>
            if isMaster then
              IO.println "Warning: legacy checkpoint without training metadata; only model/optimizer state restored."
            pure (baseState, hp)
          | some metadata =>
            let hp' := mergedResumeHyperparameters hp metadata.hyperparameters
            let withTokens : DistributedTrainState (device := device) cfg := {
              baseState with
              totalTokens := metadata.totalTokens
            }
            let withCursor ←
              match metadata.dataCursor? with
              | none =>
                if isMaster then
                  IO.println "Warning: checkpoint metadata missing dataloader cursor; continuing with fresh loader position."
                pure withTokens
              | some cursor =>
                let restoredGen ← restoreDataCursor withTokens.dataGen cursor
                if isMaster then
                  IO.println s!"Restored dataloader cursor: shard={cursor.trainPathIdx} epoch={cursor.epoch} batch={cursor.batchCount} pos={cursor.bosCurrentPos}"
                pure { withTokens with dataGen := restoredGen }
            if isMaster then
              IO.println s!"Restored checkpoint hyperparameters (using current iteration horizon: numIterations={hp'.numIterations}, extensionIterations={hp'.extensionIterations})."
            pure (withCursor, hp')
        pure (resumedState, restoredHp)
      | none =>
        if resume.isSome then
          throw <| IO.userError s!"Resume checkpoint not found or invalid: {resumePath}"
        else
          throw <| IO.userError s!"Failed to load auto-resume checkpoint {resumePath}; refusing to start fresh automatically. Remove the checkpoint or fix compatibility to proceed."

  -- Refresh trainable parameter identity before NCCL collectives.
  -- Loading and initialization already place the complete state on this device.
  let paramsOnDevice ← TensorStruct.makeLeafParams state.params
  let adamOnDevice := state.optState.adamState
  let dualOnDevice := state.optState.dualState
  let state := {
    state with
    params := paramsOnDevice
    optState := {
      state.optState with
      adamState := adamOnDevice
      dualState := dualOnDevice
    }
  }

  -- Synchronize parameters from rank 0
  let syncedParams ← syncParameters state.params
  let state := {
    state with
    params := syncedParams
  }

  -- Match nanochat's strict grad-accum divisibility contract.
  validateGradAccumConfig effectiveHp worldSize

  -- Barrier to ensure all ranks are ready
  if isDistributed then
    dist.barrier

  -- Run distributed training loop
  let finalState ← trainLoopDistributed cfg effectiveHp state checkpointDir

  if isMaster then
    IO.println s!"Training finished!"
    IO.println s!"Best validation loss: {finalState.bestValLoss}"

  return finalState

/-- Dynamically-shaped GPT micro-batch `(inputs, targets)`. -/
structure DynamicGPTBatch where
  batch : UInt64
  seq : UInt64
  input : Tensor { shape := #[batch, seq], dtype := .Int64, device := .CPU }
  target : Tensor { shape := #[batch, seq], dtype := .Int64, device := .CPU }

/-- Provider callback for GPT micro-batches. -/
abbrev DynamicGPTBatchProvider := IO (Option DynamicGPTBatch)

/-- Distributed streaming training state (no shard-backed data generator). -/
structure StreamTrainState (cfg : moddedGpt.Config) where
  params : ModdedGPTParams (device := device) cfg
  yarn : YarnRotary (device := device) cfg.headDim cfg.maxSeqLen
  optState : OptimizerState (device := device) cfg
  step : UInt64 := 0
  bestValLoss : Float := 1e30
  totalTokens : UInt64 := 0
  startTime : UInt64 := 0
  rank : UInt64 := 0
  worldSize : UInt64 := 1

private def logStreamProgress (state : StreamTrainState (device := device) cfg) (result : StepResult)
    (hp : Hyperparameters) : IO Unit := do
  if state.step % hp.logInterval == 0 then
    let nowMs ← IO.monoMsNow
    let elapsed := nowMs - state.startTime.toNat
    let elapsedSec := elapsed.toFloat / 1000.0
    let tokensPerSec :=
      if elapsedSec <= 0.0 then 0.0 else state.totalTokens.toFloat / elapsedSec
    let lr := getLearningRate state.step hp
    IO.println s!"Step {state.step}: loss={result.loss} lr={lr} batch={hp.deviceBatchSize} seq={hp.maxSeqLen} tok/s={tokensPerSec} time={result.timeMs}ms"

private def initStreamTrainState (cfg : moddedGpt.Config) (device : Device)
    (lr : Float := 0.023) (weightDecay : Float := 0.0)
    : IO (StreamTrainState (device := device) cfg) := do
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then
      dist.getRankAndWorldSize
    else
      pure (0, 1)

  let params ← ModdedGPTParams.init cfg (device := device)
  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen cfg.ropeBase (device := device)
  let optState := OptimizerState.init cfg params lr weightDecay
  let startTime ← IO.monoMsNow

  return {
    params := params
    yarn := yarn
    optState := optState
    startTime := startTime.toUInt64
    rank := rank
    worldSize := worldSize
  }

private def collectMicroBatches {batch seq : UInt64}
    (provider : DynamicGPTBatchProvider)
    (gradAccum : Nat)
    (device : Device)
    : IO (Array (Tensor { shape := #[batch, seq], dtype := .Int64, device := device } × Tensor { shape := #[batch, seq], dtype := .Int64, device := device })) := do
  let mut microBatches : Array (Tensor { shape := #[batch, seq], dtype := .Int64, device := device } × Tensor { shape := #[batch, seq], dtype := .Int64, device := device }) := #[]
  for _ in [:gradAccum] do
    match ← provider with
    | none => pure ()
    | some batchData =>
      let input := (batchData.input.reshapeChecked #[batch, seq]).to device
      let target := (batchData.target.reshapeChecked #[batch, seq]).to device
      microBatches := microBatches.push (input, target)
  return microBatches

private def validateWithProvider {cfg : moddedGpt.Config} {batch seq : UInt64}
    (params : ModdedGPTParams (device := device) cfg)
    (yarn : YarnRotary (device := device) cfg.headDim cfg.maxSeqLen)
    (provider : DynamicGPTBatchProvider)
    (numBatches : Nat)
    : IO (Option Float) := do
  if numBatches == 0 then
    return none

  let mut totalLoss := 0.0
  let mut seen := 0

  for _ in [:numBatches] do
    match ← provider with
    | none => pure ()
    | some batchData =>
      let input := (batchData.input.reshapeChecked #[batch, seq]).to device
      let target := (batchData.target.reshapeChecked #[batch, seq]).to device
      let lossT ← moddedGpt.loss params yarn input target false
      totalLoss := totalLoss + Tensor.item lossT
      seen := seen + 1

  if seen == 0 then
    return none
  else
    return some (totalLoss / seen.toFloat)

/-- All ranks finish their cursor files before the master can publish a model
snapshot. The staging directory is never used by checkpoint readers. -/
private def checkpointPhase (label : String) (worldSize : UInt64) (device : Device)
    (action : IO Unit) : IO Unit := do
  let failure? ← try action; pure none catch e => pure (some s!"{e}")
  let failed : Bool ← if worldSize > 1 then do
      let flag := (Tensor.full #[1] (if failure?.isSome then 1.0 else 0.0)).to device
      let flag ← dist.allReduce flag .max
      pure (decide (Tensor.item (flag.reshapeChecked #[]) > 0.0))
    else pure failure?.isSome
  match failed with
  | true =>
      throw <| IO.userError s!"Checkpoint {label} failed: {failure?.getD "another rank reported an error"}"
  | false => pure ()

private def stageStreamCursors (checkpointDir : String) (step rank worldSize : UInt64)
    (device : Device)
    (saveCursor? : Option (String → IO Unit)) : IO (Option String) := do
  match saveCursor? with
  | none => pure none
  | some saveCursor => do
      let staging : System.FilePath := s!"{checkpointDir}/.stream-cursors-{step}"
      checkpointPhase "cursor staging preparation" worldSize device do
        if rank == 0 then
          if ← staging.pathExists then IO.FS.removeDirAll staging
          IO.FS.createDirAll staging
      checkpointPhase "rank cursor write" worldSize device (saveCursor staging.toString)
      return some staging.toString

private def streamCursorWriter (staging? : Option String) (worldSize : UInt64) :
    Option (String → IO Unit) :=
  staging?.map fun staging snapshot => do
    for rank in [:worldSize.toNat] do
      let name := s!"stream_cursor_rank{rank}.json"
      let contents ← IO.FS.readFile (s!"{staging}/{name}" : System.FilePath)
      IO.FS.writeFile (s!"{snapshot}/{name}" : System.FilePath) contents
    IO.FS.writeFile (s!"{snapshot}/stream_cursor_world_size.txt" : System.FilePath) (toString worldSize)

private def removeStreamCursorStaging (staging? : Option String) (rank : UInt64) : IO Unit := do
  if rank == 0 then
    if let some staging := staging? then
      try IO.FS.removeDirAll staging catch _ => pure ()

private def trainLoopDistributedStream (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (state : StreamTrainState (device := device) cfg)
    (trainProvider : DynamicGPTBatchProvider)
    (valProvider? : Option DynamicGPTBatchProvider)
    (valBatches : Nat)
    (checkpointDir : String)
    (saveStreamCursor? : Option (String → IO Unit) := none)
    : IO (StreamTrainState (device := device) cfg) := do
  let totalSteps := hp.numIterations + hp.extensionIterations
  let mut state := state
  let isMaster := state.rank == 0
  let latestPath := s!"{checkpointDir}/latest.ckpt"
  let startStep := state.optState.step
  if isMaster then
    IO.FS.createDirAll ⟨checkpointDir⟩

  if startStep >= totalSteps then
    if isMaster then
      IO.println s!"Training already complete at step {state.step} (target {totalSteps})"
    return state

  for step in [startStep.toNat:totalSteps.toNat] do
    let stepU := step.toUInt64
    let batchSize := hp.deviceBatchSize
    let seqLen := hp.maxSeqLen
    let gradAccum := effectiveGradAccumSteps hp state.worldSize
    let device := state.params.embed.device

    let microBatches ← collectMicroBatches
      (batch := batchSize) (seq := seqLen)
      trainProvider gradAccum.toNat device

    if microBatches.isEmpty then
      if isMaster then
        IO.println "No micro-batches available; ending stream training early."
      break

    let (newParams, newOptState, result) ← trainStepDistributedAccum state.params state.yarn
      microBatches state.optState hp

    state := { state with
      params := newParams
      optState := newOptState
      step := stepU
      totalTokens := state.totalTokens + result.tokensProcessed
    }

    if isMaster then
      logStreamProgress state result hp

    if stepU % hp.valInterval == 0 && stepU > 0 && isMaster then
      match valProvider? with
      | none => pure ()
      | some valProvider =>
        match ← validateWithProvider
            (cfg := cfg) (batch := batchSize) (seq := seqLen)
            state.params state.yarn valProvider valBatches with
        | none => pure ()
        | some valLoss =>
          IO.println s!"Validation: loss={valLoss}"
          if valLoss < state.bestValLoss then
            IO.println "  New best validation loss!"
            state := { state with bestValLoss := valLoss }

    if stepU % hp.checkpointInterval == 0 && stepU > 0 then
      let staging? ← stageStreamCursors checkpointDir state.optState.step state.rank state.worldSize
        state.params.embed.device saveStreamCursor?
      checkpointPhase "snapshot publication" state.worldSize state.params.embed.device do
        if isMaster then
          let ckpt : Checkpoint (device := device) cfg := {
            params := state.params
            optState := state.optState
            step := stepU
            bestValLoss := state.bestValLoss
          }
          let metadata := mkCheckpointMetadata cfg hp state.totalTokens none
          let stepPath := s!"{checkpointDir}/step_{stepU}.ckpt"
          saveCheckpoint ckpt stepPath (some metadata) (streamCursorWriter staging? state.worldSize)
          saveCheckpoint ckpt latestPath (some metadata) (streamCursorWriter staging? state.worldSize)
      removeStreamCursorStaging staging? state.rank

  let staging? ← stageStreamCursors checkpointDir state.optState.step state.rank state.worldSize
    state.params.embed.device saveStreamCursor?
  checkpointPhase "snapshot publication" state.worldSize state.params.embed.device do
    if isMaster then
      let finalCkpt : Checkpoint (device := device) cfg := {
        params := state.params
        optState := state.optState
        step := state.step
        bestValLoss := state.bestValLoss
      }
      let finalMetadata := mkCheckpointMetadata cfg hp state.totalTokens none
      saveCheckpoint finalCkpt latestPath (some finalMetadata) (streamCursorWriter staging? state.worldSize)
  removeStreamCursorStaging staging? state.rank
  if isMaster then
    IO.println s!"Training complete! Total tokens: {state.totalTokens}"

  return state

/-- Train from streaming GPT batch providers instead of parquet shards.
    This keeps the distributed optimizer/checkpoint path unchanged while
    allowing task-mixture token-buffer data feeding. -/
def trainDistributedWithBatchProvider (cfg : moddedGpt.Config) (hp : Hyperparameters)
    (device : Device)
    (trainProvider : DynamicGPTBatchProvider)
    (valProvider? : Option DynamicGPTBatchProvider := none)
    (valBatches : Nat := 0)
    (checkpointDir : String := "checkpoints/modded")
    (resume : Option String := none)
    (saveStreamCursor? : Option (String → IO Unit) := none)
    (restoreStreamCursor? : Option (String → IO Unit) := none)
    : IO (StreamTrainState (device := device) cfg) := do
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then
      dist.getRankAndWorldSize
    else
      pure (0, 1)

  let isMaster := rank == 0

  if isMaster then
    IO.println s!"Starting streaming training with {worldSize} GPUs"
    IO.println s!"Config: {repr cfg}"
    IO.println s!"Hyperparameters: {repr hp}"

  let initState ← initStreamTrainState cfg device

  let latestPath := s!"{checkpointDir}/latest.ckpt"
  let resumePath? ←
    match resume with
    | some path => pure (some path)
    | none =>
      if ← checkpointExists latestPath then
        pure (some latestPath)
      else
        pure none

  let (state, effectiveHp) ←
    match resumePath? with
    | none => pure (initState, hp)
    | some resumePath =>
      if isMaster then
        IO.println s!"Resuming from checkpoint: {resumePath}"
      let (snapshot, strict) ← resolveCheckpointPath resumePath
      match ← loadCheckpointForResumeAt (device := device) cfg snapshot (!isMaster) strict with
      | some (ckpt, metadata?) =>
        let resumedParams ← TensorStruct.makeLeafParams ckpt.params
        let resumedOptState : OptimizerState (device := device) cfg := {
          ckpt.optState with
          adamState := ckpt.optState.adamState
          dualState := ckpt.optState.dualState
        }
        let baseState : StreamTrainState (device := device) cfg := {
          initState with
          params := resumedParams
          optState := resumedOptState
          step := ckpt.step
          bestValLoss := ckpt.bestValLoss
        }
        let (resumedState, restoredHp) ←
          match metadata? with
          | none =>
            if isMaster then
              IO.println "Warning: legacy checkpoint without training metadata; only model/optimizer state restored."
            pure (baseState, hp)
          | some metadata =>
            let hp' := mergedResumeHyperparameters hp metadata.hyperparameters
            if isMaster then
              IO.println s!"Restored checkpoint hyperparameters (using current iteration horizon: numIterations={hp'.numIterations}, extensionIterations={hp'.extensionIterations})."
            pure ({ baseState with totalTokens := metadata.totalTokens }, hp')
        if let some restoreStreamCursor := restoreStreamCursor? then
          if strict then
            let cursorWorldSize ← IO.FS.readFile (s!"{snapshot}/stream_cursor_world_size.txt" : System.FilePath)
            if cursorWorldSize.trimAscii.toString.toNat? != some worldSize.toNat then
              throw <| IO.userError "Checkpoint stream cursor world size does not match this training run"
          restoreStreamCursor snapshot
          if isMaster then
            IO.println s!"Restored stream cursor state from {snapshot}"
        pure (resumedState, restoredHp)
      | none =>
        if resume.isSome then
          throw <| IO.userError s!"Resume checkpoint not found or invalid: {resumePath}"
        else
          throw <| IO.userError s!"Failed to load auto-resume checkpoint {resumePath}; refusing to start fresh automatically. Remove the checkpoint or fix compatibility to proceed."

  -- Refresh trainable parameter identity before NCCL collectives.
  -- Loading and initialization already place the complete state on this device.
  let paramsOnDevice ← TensorStruct.makeLeafParams state.params
  let adamOnDevice := state.optState.adamState
  let dualOnDevice := state.optState.dualState
  let state := {
    state with
    params := paramsOnDevice
    optState := {
      state.optState with
      adamState := adamOnDevice
      dualState := dualOnDevice
    }
  }

  let syncedParams ← syncParameters state.params
  let state := {
    state with
    params := syncedParams
  }

  -- Match nanochat's strict grad-accum divisibility contract.
  validateGradAccumConfig effectiveHp worldSize

  if isDistributed then
    dist.barrier

  let finalState ← trainLoopDistributedStream
    cfg effectiveHp state trainProvider valProvider? valBatches checkpointDir saveStreamCursor?

  if isMaster then
    IO.println "Streaming training finished!"
    IO.println s!"Best validation loss: {finalState.bestValLoss}"

  return finalState

end torch.ModdedTrain
