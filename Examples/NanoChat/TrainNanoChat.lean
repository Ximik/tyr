/-
  ModdedTrainGPT.lean

  Main training script for modded-nanogpt style GPT training.

  This is the entry point that ties together:
  - Model architecture (ModdedGPT)
  - NorMuon optimizer (for weight matrices)
  - DistAdam optimizer (for embeddings)
  - Distributed training
  - Document-aware data loading

  Usage:
    lake build TrainNanoChat
    .lake/build/bin/TrainNanoChat

  For distributed training (multi-GPU):
    scripts/nanochat/run_train_torchrun.sh
-/
import Tyr.Torch
import Tyr.Distributed
import Examples.NanoChat.ModdedGPT
import Tyr.DataLoader
import Examples.NanoChat.ModdedTrain
import Tyr.Optim.NorMuon
import Tyr.Optim.DistAdam

open torch
open torch.moddedGpt
open torch.DataLoader
open torch.ModdedTrain
open torch.dist

/-- Runtime environment state for standalone training. -/
structure TrainEnvState where
  localRank? : Option UInt64 := none
  worldSize? : Option UInt64 := none
  rank? : Option UInt64 := none
  masterAddr? : Option String := none
  masterPort? : Option UInt64 := none
  requestedDevice? : Option String := none
  deriving Repr, Inhabited

/-- Resolve distributed/device env vars once into a typed structure. -/
def loadTrainEnvState : IO TrainEnvState := do
  let localRank? := ((← IO.getEnv "LOCAL_RANK").bind (·.toNat?)).map (·.toUInt64)
  let worldSize? := ((← IO.getEnv "WORLD_SIZE").bind (·.toNat?)).map (·.toUInt64)
  let rank? := ((← IO.getEnv "RANK").bind (·.toNat?)).map (·.toUInt64)
  let masterAddr? := (← IO.getEnv "MASTER_ADDR")
  let masterPort? := ((← IO.getEnv "MASTER_PORT").bind (·.toNat?)).map (·.toUInt64)
  let requestedDevice? := (← IO.getEnv "TYR_DEVICE").map String.toLower
  pure {
    localRank? := localRank?
    worldSize? := worldSize?
    rank? := rank?
    masterAddr? := masterAddr?
    masterPort? := masterPort?
    requestedDevice? := requestedDevice?
  }

/-- Parse command line arguments -/
structure Args where
  /-- Path to training data -/
  dataPath : String := "data/fineweb10B"
  /-- Path to validation data -/
  valPath : String := "data/fineweb_val"
  /-- Checkpoint directory -/
  checkpointDir : String := "checkpoints/modded"
  /-- Resume from checkpoint -/
  resume : Option String := none
  /-- Override number of iterations -/
  numIterations : Option UInt64 := none
  /-- Local rank for distributed training (from env) -/
  localRank : UInt64 := 0
  /-- Run in debug mode with small model -/
  debug : Bool := false
  deriving Repr, Inhabited

/-- Parse args from command line -/
def parseArgs (args : List String) : Args :=
  let rec go : List String → Args → Args
    | [], acc => acc
    | "--data" :: path :: rest, acc => go rest { acc with dataPath := path }
    | "--val" :: path :: rest, acc => go rest { acc with valPath := path }
    | "--checkpoint-dir" :: dir :: rest, acc => go rest { acc with checkpointDir := dir }
    | "--resume" :: path :: rest, acc => go rest { acc with resume := some path }
    | "--iterations" :: n :: rest, acc =>
        go rest { acc with numIterations := n.toNat?.map (·.toUInt64) }
    | "--local-rank" :: n :: rest, acc =>
        go rest { acc with localRank := (n.toNat?.map (·.toUInt64)).getD acc.localRank }
    | "--local_rank" :: n :: rest, acc =>
        go rest { acc with localRank := (n.toNat?.map (·.toUInt64)).getD acc.localRank }
    | "--debug" :: rest, acc => go rest { acc with debug := true }
    | arg :: rest, acc =>
        if arg.startsWith "--local-rank=" then
          let v := arg.drop "--local-rank=".length
          go rest { acc with localRank := (v.toNat?.map (·.toUInt64)).getD acc.localRank }
        else if arg.startsWith "--local_rank=" then
          let v := arg.drop "--local_rank=".length
          go rest { acc with localRank := (v.toNat?.map (·.toUInt64)).getD acc.localRank }
        else if arg.startsWith "--local-rank:" then
          let v := arg.drop "--local-rank:".length
          go rest { acc with localRank := (v.toNat?.map (·.toUInt64)).getD acc.localRank }
        else
          go rest acc
  go args default

/-- Get local rank from typed env state (for torchrun). -/
def getLocalRankFromEnv (env : TrainEnvState) : UInt64 :=
  env.localRank?.getD 0

/-- Get world size from typed env state. -/
def getWorldSizeFromEnv (env : TrainEnvState) : UInt64 :=
  env.worldSize?.getD 1

/-- Get global rank from typed env state (for distributed training). -/
def getRankFromEnv (env : TrainEnvState) : UInt64 :=
  env.rank?.getD (getLocalRankFromEnv env)

/-- Render a device for logging. -/
def deviceToString : Device → String
  | Device.MPS => "MPS"
  | Device.CPU => "CPU"
  | Device.CUDA n => s!"CUDA:{n}"

/-- Resolve training device from TYR_DEVICE.
    In distributed mode, CUDA ordinal follows LOCAL_RANK (per-node index). -/
def resolveTrainingDevice (_rank localRank : UInt64) (isDistributed : Bool) (env : TrainEnvState) : IO Device := do
  let cudaOrdinal := if isDistributed then localRank else 0
  let requested? := env.requestedDevice?
  match requested? with
  | some "cpu" => pure Device.CPU
  | some "mps" => pure Device.MPS
  | some "cuda" => pure (Device.CUDA cudaOrdinal)
  | some "auto" | none =>
    if isDistributed then
      if ← cuda_is_available then pure (Device.CUDA cudaOrdinal) else pure Device.CPU
    else
      getBestDevice
  | some _ =>
    if isDistributed then
      if ← cuda_is_available then pure (Device.CUDA cudaOrdinal) else pure Device.CPU
    else
      getBestDevice

/-- Initialize distributed training if needed -/
def initDistributedIfNeeded (env : TrainEnvState) : IO Bool := do
  let worldSize := getWorldSizeFromEnv env
  if worldSize > 1 then
    let rank := getRankFromEnv env
    let localRank := getLocalRankFromEnv env
    let masterAddr := env.masterAddr?.getD "localhost"
    let masterPort := env.masterPort?.getD 29500
    IO.println s!"Initializing distributed: rank {rank} (local {localRank})/{worldSize} master={masterAddr}:{masterPort}"
    if ← cuda_is_available then
      dist.setCudaDevice localRank
    dist.initProcessGroup "nccl" masterAddr masterPort rank worldSize
    return true
  else
    return false

/-- Main training function -/
def runTraining (args : Args) (env : TrainEnvState) : IO Unit := do
  IO.println "=== Modded NanoGPT Training (Tyr Port) ==="
  IO.println ""

  -- Initialize distributed if needed
  let isDistributed ← initDistributedIfNeeded env
  let (rank, worldSize) ← if isDistributed then
      getRankAndWorldSize
    else
      pure (0, 1)
  let localRank := env.localRank?.getD args.localRank
  let trainDevice ← resolveTrainingDevice rank localRank isDistributed env
  let seed := 42 + rank
  manualSeed seed
  IO.println s!"Rank {rank}/{worldSize} local={localRank} device={deviceToString trainDevice}"

  let isMaster := rank == 0

  if isMaster then
    IO.println s!"Configuration:"
    IO.println s!"  Data path: {args.dataPath}"
    IO.println s!"  Validation path: {args.valPath}"
    IO.println s!"  Checkpoint dir: {args.checkpointDir}"
    IO.println s!"  Distributed: {isDistributed} (world size: {worldSize})"
    IO.println s!"  Ranks: global={rank}, local={localRank}"
    IO.println s!"  Device: {deviceToString trainDevice}"
    IO.println s!"  Seed: {seed}"
    IO.println s!"  Debug mode: {args.debug}"
    IO.println ""

  -- Model config
  let cfg := if args.debug then
      -- Smaller model for debugging
      { Config.default with
        nLayer := 4
        nHead := 4
        headDim := 64
        modelDim := 256
        maxSeqLen := 512
        blockSize := 64
      }
    else
      Config.default

  if isMaster then
    IO.println s!"Model config:"
    IO.println s!"  Vocab size: {cfg.vocabSize}"
    IO.println s!"  Layers: {cfg.nLayer}"
    IO.println s!"  Heads: {cfg.nHead}"
    IO.println s!"  Head dim: {cfg.headDim}"
    IO.println s!"  Model dim: {cfg.modelDim}"
    IO.println ""

  -- Hyperparameters
  let hpBase : Hyperparameters := if args.debug then
    -- Keep debug runs lightweight enough for congested multi-GPU nodes.
    { Hyperparameters.default with
      deviceBatchSize := 1
      totalBatchSizeTokens := 2048
      maxSeqLen := cfg.maxSeqLen
      valInterval := 1
      logInterval := 1
      checkpointInterval := 10
    }
  else
    { Hyperparameters.default with
      maxSeqLen := cfg.maxSeqLen
    }
  let hp : Hyperparameters := {
    hpBase with
    numIterations := args.numIterations.getD hpBase.numIterations
  }

  -- Data config
  let dataConfig : DataLoader.Config := {
    dataPath := args.dataPath
    valPath := some args.valPath
    seqLen := cfg.maxSeqLen
    bosToken := 50256  -- GPT-2 BOS
    numWorkers := 4
    bufferSize := 8
    seed := 42 + rank  -- Different seed per rank
  }

  if isMaster then
    IO.println s!"Training for {hp.numIterations + hp.extensionIterations} iterations"
    IO.println s!"  Cooldown fraction: {hp.cooldownFrac}"
    IO.println s!"  Gradient accumulation: {effectiveGradAccumSteps hp worldSize}"
    IO.println s!"  Validation interval: {hp.valInterval}"
    IO.println ""

  if isMaster then
    IO.println "Initializing model..."
    IO.println ""
    IO.println "Starting training..."
    IO.println "=============================================================="

  let finalState ← trainDistributed cfg hp dataConfig trainDevice args.checkpointDir args.resume

  if isMaster then
    IO.println ""
    IO.println "=============================================================="
    IO.println "Training complete!"
    IO.println s!"  Total steps: {finalState.step}"
    IO.println s!"  Total tokens: {finalState.totalTokens}"
    IO.println s!"  Best validation loss: {finalState.bestValLoss}"

  -- Cleanup distributed
  if isDistributed then
    dist.barrier
    dist.destroyProcessGroup

  if isMaster then
    IO.println ""
    IO.println "Done!"

/-- Main entry point -/
def main (args : List String) : IO Unit := do
  -- Parse arguments
  let parsedArgs := parseArgs args
  let envState ← loadTrainEnvState

  -- Run training with error handling
  try
    runTraining parsedArgs envState
  catch e =>
    IO.eprintln s!"Error: {e}"
    -- Cleanup if distributed
    let isInit ← dist.isInitialized
    if isInit then
      dist.destroyProcessGroup
    IO.Process.exit 1
