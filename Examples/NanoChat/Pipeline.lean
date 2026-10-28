/-
  Examples/NanoChat/Pipeline.lean

  NanoChat training pipeline orchestration.

  This implements the full nanochat training pipeline:
  1. Tokenizer training and evaluation
  2. Base model pretraining
  3. Midtraining (conversation tokens, tool use)
  4. Supervised fine-tuning
  5. Reinforcement learning (optional)

  Based on nanochat's speedrun.sh orchestration.
-/
import Tyr.Pipeline
import Tyr.Distributed
import Tyr.DataLoader
import Tyr.Data.Pretraining
import Tyr.Data.Task
import Tyr.Data.TaskClass
import Tyr.Train.ChatSFT
import Tyr.Train.RunLedger
import Tyr.RL.GRPO
import Tyr.Data.Download
import Tyr.Data.HuggingFace
import Tyr.Tokenizer
import Examples.NanoChat.ModdedGPT
import Examples.NanoChat.ModdedTrain
import Examples.NanoChat.Eval.CORE
import Examples.NanoChat.Eval.COREData
import Examples.NanoChat.Tasks.LLM

namespace torch.NanoChat.Pipeline

open torch
open torch.Pipeline
open torch.ModdedTrain
open torch.moddedGpt
open torch.Data.Pretraining
open torch.Data.Task
open torch.Data.TaskClass
open torch.Data.Download
open torch.Data.HuggingFace
open torch.Train.ChatSFT
open torch.Train.RunLedger
open torch.RL.GRPO
open torch.Eval.CORE
open torch.Eval.COREData
open torch.Tasks.LLM
open tokenizer

/-! ## Pipeline Configuration -/

private def consolePipelineLogHandlers : LogHandlers := {
  onInfo := fun msg => IO.println msg
  onError := fun msg => IO.eprintln msg
}

private def consoleSFTCallbacks : torch.Train.ChatSFT.Callbacks := {
  onTrainStep := fun step trainLoss lrm numTokens => do
    IO.println <| torch.Train.ChatSFT.formatTrainUpdate step trainLoss lrm numTokens
    (← IO.getStdout).flush
  onValidation := fun step valLoss => do
    IO.println <| torch.Train.ChatSFT.formatValidationUpdate step valLoss
    (← IO.getStdout).flush
}

private def consoleGRPOCallbacks : torch.RL.GRPO.Callbacks := {
  onTrainStep := fun result state => do
    IO.println <| torch.RL.GRPO.formatTrainUpdate result state
    (← IO.getStdout).flush
  onEval := fun passK step => do
    IO.println <| torch.RL.GRPO.formatEvalUpdate passK step
    (← IO.getStdout).flush
}

/-- Model-related environment state. -/
structure ModelEnvState where
  depth : UInt64 := 20
  aspectRatio : UInt64 := 64
  targetHeadDim : UInt64 := 128
  maxSeqLen : Nat := 2048
  ropeBase : Float := 500000.0
  paramDataRatio : UInt64 := 20
  vocabSize : UInt64 := 65536
  tokenizerMaxChars : UInt64 := 2000000000
  tokenizerDocCap : UInt64 := 10000
  deriving Repr, Inhabited

/-- Pipeline-wide environment state. -/
structure PipelineEnvState where
  baseDir : String := "~/.cache/nanochat"
  numGpus : UInt64 := 1
  dataPath : String := "base_data"
  numDataShards : UInt64 := 240
  initialDataShards : UInt64 := 8
  quickMode : Bool := false
  enableRL : Bool := false
  wandbRun : String := "dummy"
  wandbEnabled : Bool := false
  deriving Repr, Inhabited

/-- Pretraining environment state. -/
structure PretrainEnvState where
  dataPathOverride : Option String := none
  numIterations : UInt64 := 21400
  extensionIterations : UInt64 := 0
  valInterval : UInt64 := 250
  logInterval : UInt64 := 10
  deviceBatchSize : Nat := 32
  totalBatchSizeTokens : Nat := 524288
  checkpointInterval : UInt64 := 21400
  textColumn : String := "text"
  tokenizerBatchSize : Nat := 128
  evalTokens : Nat := 10485760
  deriving Repr, Inhabited

/-- Midtraining environment state. -/
structure MidtrainEnvState where
  numIterations : UInt64 := 811
  extensionIterations : UInt64 := 0
  valInterval : UInt64 := 150
  logInterval : UInt64 := 10
  deviceBatchSize : Nat := 32
  totalBatchSizeTokens : Nat := 524288
  checkpointInterval : UInt64 := 811
  evalTokens : Nat := 10485760
  deriving Repr, Inhabited

/-- SFT environment state. -/
structure SFTEnvState where
  numEpochs : UInt64 := 1
  deviceBatchSize : Nat := 4
  targetExamplesPerStep : UInt64 := 32
  maxExamples : Option UInt64 := none
  deriving Repr, Inhabited

/-- RL/GRPO environment state. -/
structure RLEnvState where
  numSamples : UInt64 := 16
  maxNewTokens : Nat := 256
  examplesPerStepGlobal : UInt64 := 16
  evalEvery : UInt64 := 60
  logEvery : UInt64 := 10
  epochs : UInt64 := 1
  maxPrompts : Option UInt64 := none
  deriving Repr, Inhabited

/-- Evaluation/task-data environment state. -/
structure EvalEnvState where
  coreMaxExamples : Option UInt64 := none
  identityConversationsPath : Option String := none
  deriving Repr, Inhabited

/-- Canonical environment snapshot for a NanoChat pipeline run. -/
structure NanoChatEnvState where
  pipeline : PipelineEnvState := {}
  model : ModelEnvState := {}
  pretrain : PretrainEnvState := {}
  midtrain : MidtrainEnvState := {}
  sft : SFTEnvState := {}
  rl : RLEnvState := {}
  eval : EvalEnvState := {}
  deriving Repr, Inhabited

/-- NanoChat-specific pipeline configuration -/
structure NanoChatConfig extends PipelineConfig where
  /-- Model depth (number of transformer layers) -/
  modelDepth : Nat := 20
  /-- Aspect ratio used to derive model dimension: modelDim = depth * aspectRatio. -/
  modelAspectRatio : Nat := 64
  /-- Target head dimension used when deriving number of heads. -/
  targetHeadDim : Nat := 128
  /-- Maximum sequence length for training/inference. -/
  maxSeqLen : Nat := 2048
  /-- RoPE base frequency. -/
  ropeBase : Float := 500000.0
  /-- Target parameter-to-data ratio (Chinchilla scaling) -/
  paramDataRatio : Nat := 20
  /-- Tokenizer vocabulary size -/
  vocabSize : Nat := 65536
  /-- Maximum characters for tokenizer training -/
  tokenizerMaxChars : Nat := 2000000000
  /-- Maximum characters per document for tokenizer training (nanochat tok_train default). -/
  tokenizerDocCap : Nat := 10000
  /-- Number of data shards for pretraining -/
  numDataShards : Nat := 240
  /-- Number of data shards for initial tokenizer training -/
  initialDataShards : Nat := 8
  /-- Path to data directory -/
  dataPath : String := "base_data"
  /-- HuggingFace dataset URL for FineWeb-Edu -/
  dataUrl : String := "https://huggingface.co/datasets/karpathy/fineweb-edu-100b-shuffle/resolve/main"
  /-- Maximum download retry attempts -/
  maxDownloadRetries : Nat := 5
  /-- SFT learning rate for embeddings -/
  sftEmbeddingLr : Float := 0.2
  /-- SFT learning rate for matrix weights -/
  sftMatrixLr : Float := 0.02
  /-- Number of SFT epochs -/
  sftEpochs : Nat := 1
  /-- Number of GRPO samples per prompt -/
  grpoNumSamples : Nat := 16
  /-- GRPO max new tokens -/
  grpoMaxNewTokens : Nat := 256
  /-- Enable reinforcement-learning stages in full pipeline runs. -/
  enableRL : Bool := false
  /-- Fully resolved model/training/pipeline environment state for this run. -/
  envState : NanoChatEnvState := {}
  deriving Repr, Inhabited

/-- Pick a head count that divides `modelDim`, closest to `targetHeadDim`. -/
private def findNumHeads (modelDim targetHeadDim : Nat) : Nat := Id.run do
  if modelDim == 0 then
    return 1
  let target := max 1 targetHeadDim
  let ideal := max 1 ((modelDim + target / 2) / target)
  for off in [:modelDim + 1] do
    let up := ideal + off
    if up > 0 && modelDim % up == 0 then
      return up
    if ideal > off then
      let down := ideal - off
      if down > 0 && modelDim % down == 0 then
        return down
  return 1

/-- Convert to model config -/
def NanoChatConfig.toModelConfig (cfg : NanoChatConfig) : moddedGpt.Config :=
  let modelDimNat := max 1 (cfg.modelDepth * cfg.modelAspectRatio)
  let nHeadNat := findNumHeads modelDimNat cfg.targetHeadDim
  let headDimNat := max 1 (modelDimNat / nHeadNat)
  {
    vocabSize := cfg.vocabSize
    nLayer := cfg.modelDepth
    nHead := nHeadNat
    headDim := headDimNat
    modelDim := modelDimNat
    maxSeqLen := cfg.maxSeqLen
    blockSize := 128
    ropeBase := cfg.ropeBase
  }

/-- Get full path for data directory -/
def NanoChatConfig.getDataDir (cfg : NanoChatConfig) : IO String := do
  let baseDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
  return s!"{baseDir}/{cfg.dataPath}"

/-- Get tokenizer path -/
def NanoChatConfig.getTokenizerPath (cfg : NanoChatConfig) : IO String := do
  let baseDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
  return s!"{baseDir}/tokenizer"

/-- Get checkpoint directory for a stage -/
def NanoChatConfig.getCheckpointDir (cfg : NanoChatConfig) (stage : String) : IO String := do
  let baseDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
  return s!"{baseDir}/checkpoints/{stage}"

private def resolveDataDirWithOverride (cfg : NanoChatConfig) (overridePath? : Option String) : IO String := do
  match overridePath? with
  | some overridePath =>
    if overridePath.startsWith "/" then
      pure overridePath
    else
      let baseDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
      pure s!"{baseDir}/{overridePath}"
  | none =>
    cfg.getDataDir

private def envNat (name : String) : IO (Option Nat) := do
  pure <| (← IO.getEnv name).bind String.toNat?

private def envUInt64 (name : String) : IO (Option UInt64) := do
  match ← envNat name with
  | none => pure none
  | some n =>
    if n < UInt64.size then pure (some n.toUInt64)
    else throw <| IO.userError s!"{name}={n} exceeds {UInt64.size - 1}"

private def parseBoolString? (raw : String) : Option Bool :=
  let s := raw.trimAscii.toString.toLower
  if s == "1" || s == "true" || s == "yes" || s == "on" then
    some true
  else if s == "0" || s == "false" || s == "no" || s == "off" then
    some false
  else
    none

private def envBool (name : String) : IO (Option Bool) := do
  pure <| (← IO.getEnv name).bind parseBoolString?

/-- Resolve all model/training/pipeline env vars once into a typed structure. -/
private def loadNanoChatEnvState : IO NanoChatEnvState := do
  let baseDir := (← IO.getEnv "NANOCHAT_DIR").getD "~/.cache/nanochat"
  let numGpus := (← envUInt64 "WORLD_SIZE").getD 1
  let modelDepth := (← envUInt64 "MODEL_DEPTH").getD 20
  let modelAspectRatio := (← envUInt64 "MODEL_ASPECT_RATIO").getD 64
  let targetHeadDim := (← envUInt64 "MODEL_HEAD_DIM").getD 128
  let maxSeqLen := (← envNat "MAX_SEQ_LEN").getD 2048
  let ropeBase := (← IO.getEnv "ROPE_BASE").bind (·.toNat?) |>.map Float.ofNat |>.getD 500000.0
  let paramDataRatio := (← envUInt64 "PARAM_DATA_RATIO").getD 20
  let vocabSize := (← envUInt64 "VOCAB_SIZE").getD 65536
  let tokenizerMaxChars := (← envUInt64 "TOKENIZER_MAX_CHARS").getD 2000000000
  let tokenizerDocCap := (← envUInt64 "TOKENIZER_DOC_CAP").getD 10000
  let dataPath := (← IO.getEnv "DATA_PATH").getD "base_data"
  let numShards := (← envUInt64 "NUM_SHARDS").getD 240
  let initialDataShards := (← envUInt64 "INITIAL_DATA_SHARDS").getD 8
  let quickMode := (← envBool "QUICK_MODE").getD ((← IO.getEnv "QUICK_MODE").isSome)
  let enableRL := (← envBool "ENABLE_RL").getD ((← IO.getEnv "ENABLE_RL").isSome)
  let wandbRun := (← IO.getEnv "WANDB_RUN").getD "dummy"
  let wandbEnabled := (← envBool "WANDB_ENABLED").getD (wandbRun != "dummy")

  let pretrainIters := (← envUInt64 "PRETRAIN_ITERS").getD 21400
  let pretrainExtIters := (← envUInt64 "PRETRAIN_EXTENSION_ITERS").getD 0
  let pretrainValInterval := (← envUInt64 "PRETRAIN_VAL_INTERVAL").getD 250
  let pretrainLogInterval := (← envUInt64 "PRETRAIN_LOG_INTERVAL").getD 10
  let pretrainDeviceBatchSize := (← envNat "PRETRAIN_DEVICE_BATCH_SIZE").getD 32
  let pretrainTotalBatchSize := (← envNat "PRETRAIN_TOTAL_BATCH_SIZE").getD 524288
  let pretrainCheckpointInterval := (← envUInt64 "PRETRAIN_CHECKPOINT_INTERVAL").getD pretrainIters
  let pretrainTextColumn := (← IO.getEnv "PRETRAIN_TEXT_COLUMN").getD "text"
  let pretrainTokenizerBatchSize := (← envNat "PRETRAIN_TOKENIZER_BATCH_SIZE").getD 128
  let pretrainEvalTokens := (← envNat "PRETRAIN_EVAL_TOKENS").getD 10485760
  let pretrainDataPathOverride := (← IO.getEnv "PRETRAIN_DATA_PATH")

  let midIters := (← envUInt64 "MIDTRAIN_ITERS").getD 811
  let midExtIters := (← envUInt64 "MIDTRAIN_EXTENSION_ITERS").getD 0
  let midValInterval := (← envUInt64 "MIDTRAIN_VAL_INTERVAL").getD 150
  let midLogInterval := (← envUInt64 "MIDTRAIN_LOG_INTERVAL").getD 10
  let midDeviceBatchSize := (← envNat "MIDTRAIN_DEVICE_BATCH_SIZE").getD 32
  let midTotalBatchSize := (← envNat "MIDTRAIN_TOTAL_BATCH_SIZE").getD 524288
  let midCheckpointInterval := (← envUInt64 "MIDTRAIN_CHECKPOINT_INTERVAL").getD midIters
  let midEvalTokens := (← envNat "MIDTRAIN_EVAL_TOKENS").getD 10485760

  let sftEpochs := (← envUInt64 "SFT_EPOCHS").getD 1
  let sftDeviceBatchSize := max 1 <| (← envNat "SFT_DEVICE_BATCH_SIZE").getD 4
  let sftTargetExamplesPerStep := max 1 <| (← envUInt64 "SFT_TARGET_EXAMPLES_PER_STEP").getD 32
  let sftMaxExamples := (← envUInt64 "SFT_MAX_EXAMPLES")

  let grpoNumSamples := (← envUInt64 "GRPO_NUM_SAMPLES").getD 16
  let grpoMaxNewTokens := (← envNat "GRPO_MAX_NEW_TOKENS").getD 256
  let grpoExamplesPerStepGlobal := max 1 <| (← envUInt64 "GRPO_EXAMPLES_PER_STEP").getD 16
  let grpoEvalEvery := max 1 <| (← envUInt64 "GRPO_EVAL_EVERY").getD 60
  let grpoLogEvery := max 1 <| (← envUInt64 "GRPO_LOG_EVERY").getD 10
  let grpoEpochs := max 1 <| (← envUInt64 "GRPO_EPOCHS").getD 1
  let grpoMaxPrompts := (← envUInt64 "GRPO_MAX_PROMPTS")

  let coreMaxExamples := (← envUInt64 "CORE_MAX_EXAMPLES")
  let identityConversationsPath := (← IO.getEnv "IDENTITY_CONVERSATIONS_PATH")

  pure {
    pipeline := {
      baseDir := baseDir
      numGpus := numGpus
      dataPath := dataPath
      numDataShards := numShards
      initialDataShards := initialDataShards
      quickMode := quickMode
      enableRL := enableRL
      wandbRun := wandbRun
      wandbEnabled := wandbEnabled
    }
    model := {
      depth := modelDepth
      aspectRatio := modelAspectRatio
      targetHeadDim := targetHeadDim
      maxSeqLen := maxSeqLen
      ropeBase := ropeBase
      paramDataRatio := paramDataRatio
      vocabSize := vocabSize
      tokenizerMaxChars := tokenizerMaxChars
      tokenizerDocCap := tokenizerDocCap
    }
    pretrain := {
      dataPathOverride := pretrainDataPathOverride
      numIterations := pretrainIters
      extensionIterations := pretrainExtIters
      valInterval := pretrainValInterval
      logInterval := pretrainLogInterval
      deviceBatchSize := pretrainDeviceBatchSize
      totalBatchSizeTokens := pretrainTotalBatchSize
      checkpointInterval := pretrainCheckpointInterval
      textColumn := pretrainTextColumn
      tokenizerBatchSize := pretrainTokenizerBatchSize
      evalTokens := pretrainEvalTokens
    }
    midtrain := {
      numIterations := midIters
      extensionIterations := midExtIters
      valInterval := midValInterval
      logInterval := midLogInterval
      deviceBatchSize := midDeviceBatchSize
      totalBatchSizeTokens := midTotalBatchSize
      checkpointInterval := midCheckpointInterval
      evalTokens := midEvalTokens
    }
    sft := {
      numEpochs := sftEpochs
      deviceBatchSize := sftDeviceBatchSize
      targetExamplesPerStep := sftTargetExamplesPerStep
      maxExamples := sftMaxExamples
    }
    rl := {
      numSamples := grpoNumSamples
      maxNewTokens := grpoMaxNewTokens
      examplesPerStepGlobal := grpoExamplesPerStepGlobal
      evalEvery := grpoEvalEvery
      logEvery := grpoLogEvery
      epochs := grpoEpochs
      maxPrompts := grpoMaxPrompts
    }
    eval := {
      coreMaxExamples := coreMaxExamples
      identityConversationsPath := identityConversationsPath
    }
  }

private def capArray {α : Type} (xs : Array α) (cap? : Option Nat) : Array α :=
  match cap? with
  | some n =>
    let m := min n xs.size
    xs.extract 0 m
  | none => xs

private def runOnMaster (action : PipelineM Unit) : PipelineM Unit := do
  let isDistributed ← dist.isInitialized
  let (rank, _) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  if rank == 0 then
    action
    if isDistributed then
      dist.barrier
  else
    if isDistributed then
      dist.barrier

private def runOnMasterIO (action : IO Unit) : IO Unit := do
  let isDistributed ← dist.isInitialized
  let (rank, _) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  if rank == 0 then
    action

/-! ## Device & Data Adapters -/

private def findSpecialId? (tok : BPETokenizer) (names : Array String) : Option UInt64 := Id.run do
  for name in names do
    match tok.specialTokens.get? name with
    | some id => return some id.toUInt64
    | none => pure ()
  none

private def requireSpecialId (tok : BPETokenizer) (label : String) (names : Array String) : IO UInt64 := do
  match findSpecialId? tok names with
  | some id => pure id
  | none =>
    throw <| IO.userError s!"Tokenizer is missing required {label} token. Tried: {repr names}"

private def loadTokenizerForPipeline (cfg : NanoChatConfig) : IO BPETokenizer := do
  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"
  tokenizer.load tokenizerFile

private def resolveBosTokenId (cfg : NanoChatConfig) : IO UInt64 := do
  let tok ← loadTokenizerForPipeline cfg
  requireSpecialId tok "BOS" #["<|bos|>", "<|endoftext|>"]

private def buildChatTokensFromTokenizer (tok : BPETokenizer) : IO ChatTokens := do
  let bos ← requireSpecialId tok "bos" #["<|bos|>", "<|endoftext|>"]
  let userStart ← requireSpecialId tok "user_start" #["<|user_start|>", "<|user|>"]
  let userEnd ← requireSpecialId tok "user_end" #["<|user_end|>", "<|eot|>"]
  let assistantStart ← requireSpecialId tok "assistant_start" #["<|assistant_start|>", "<|assistant|>"]
  let assistantEnd ← requireSpecialId tok "assistant_end" #["<|assistant_end|>", "<|eot|>"]
  let pythonStart ← requireSpecialId tok "python_start/tool_start" #["<|python_start|>", "<|tool_start|>"]
  let pythonEnd ← requireSpecialId tok "python_end/tool_end" #["<|python_end|>", "<|tool_end|>"]
  let outputStart ← requireSpecialId tok "output_start" #["<|output_start|>"]
  let outputEnd ← requireSpecialId tok "output_end" #["<|output_end|>"]
  let systemStart := (findSpecialId? tok #["<|system_start|>", "<|system|>"]).getD userStart
  let systemEnd := (findSpecialId? tok #["<|system_end|>", "<|eot|>"]).getD userEnd
  return {
    bos := bos
    userStart := userStart
    userEnd := userEnd
    assistantStart := assistantStart
    assistantEnd := assistantEnd
    pythonStart := pythonStart
    pythonEnd := pythonEnd
    systemStart := systemStart
    systemEnd := systemEnd
    toolStart := pythonStart
    toolEnd := pythonEnd
    outputStart := outputStart
    outputEnd := outputEnd
  }

/-- Load ARC task from HuggingFace parquet rows. -/
def loadARCFromHuggingFace (subset : String) (split : String)
    (cacheDir : String := "~/.cache/huggingface") : IO ARCTask := do
  let rows ← loadARC subset split cacheDir
  let examples := rows.filterMap ARCExample.fromJson?
  pure { subset, split, examples, config := {} }

/-- Load GSM8K task from HuggingFace parquet rows. -/
def loadGSM8KFromHuggingFace (subset : String := "main") (split : String := "train")
    (cacheDir : String := "~/.cache/huggingface") : IO GSM8KTask := do
  let rows ← loadGSM8K subset split cacheDir
  let examples := rows.filterMap GSM8KExample.fromJson?
  pure { subset, split, examples, config := {} }

/-- Load MMLU task from HuggingFace parquet rows. -/
def loadMMLUFromHuggingFace (subset : String) (split : String)
    (cacheDir : String := "~/.cache/huggingface") : IO MMLUTask := do
  let rows ← loadMMLU subset split cacheDir
  let examples := rows.filterMap MMLUExample.fromJson?
  pure { subset, split, examples, config := {} }

/-- Load SmolTalk task from HuggingFace parquet rows. -/
def loadSmolTalkFromHuggingFace (split : String)
    (cacheDir : String := "~/.cache/huggingface") : IO SmolTalkTask := do
  let rows ← loadSmolTalk split cacheDir
  let examples := rows.filterMap SmolTalkExample.fromJson?
  pure { split, examples, config := {} }

private def resolveIdentityConversationsPath (cacheDir : String) (overridePath? : Option String := none) : IO String := do
  match overridePath? with
  | some p =>
    let expanded ← expandHome p
    if ← System.FilePath.pathExists ⟨expanded⟩ then
      pure expanded
    else
      throw <| IO.userError s!"Identity conversations path does not exist: {expanded}"
  | none =>
    let fallback := s!"{cacheDir}/identity_conversations.jsonl"
    if ← System.FilePath.pathExists ⟨fallback⟩ then
      pure fallback
    else
      let _ ← downloadWithRetry
        "https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl"
        fallback
        3
      if ← System.FilePath.pathExists ⟨fallback⟩ then
        pure fallback
      else
        throw <| IO.userError s!"Missing identity conversations at {fallback}"

/-- Create spelling-bee task after ensuring the shared word list is available. -/
def createSpellingBeeTaskWithDownload (cacheDir : String) (size : Nat) (split : String) : IO SpellingBeeTask := do
  let wordListPath ← ensureWordList cacheDir
  createSpellingBeeTask ⟨wordListPath⟩ size split

/-- Create simple-spelling task after ensuring the shared word list is available. -/
def createSimpleSpellingTaskWithDownload (cacheDir : String) (size : Nat) (split : String) : IO SimpleSpellingTask := do
  let wordListPath ← ensureWordList cacheDir
  createSimpleSpellingTask ⟨wordListPath⟩ size split

/-! ## Data Download with Retry Logic -/

/-- Download a single file with exponential backoff retry -/
def downloadSingleFile (url : String) (filepath : String) (maxAttempts : Nat := 5) : IO Bool := do
  -- Check if file already exists
  if ← System.FilePath.pathExists ⟨filepath⟩ then
    IO.println s!"  Skipping (already exists): {filepath}"
    return true

  IO.println s!"  Downloading: {url}"

  for attempt in [1:maxAttempts+1] do
    let tempPath := s!"{filepath}.tmp"
    let result ← IO.Process.output {
      cmd := "curl"
      args := #["-fsSL", "-o", tempPath, url]
    }

    if result.exitCode == 0 then
      -- Move temp file to final location
      IO.FS.rename ⟨tempPath⟩ ⟨filepath⟩
      IO.println s!"  Downloaded successfully: {filepath}"
      return true
    else
      IO.println s!"  Attempt {attempt}/{maxAttempts} failed: {result.stderr}"
      -- Clean up temp file if it exists
      if ← System.FilePath.pathExists ⟨tempPath⟩ then
        IO.FS.removeFile ⟨tempPath⟩

      if attempt < maxAttempts then
        -- Exponential backoff: 2^attempt seconds
        let waitTime := (2 ^ attempt) * 1000
        IO.println s!"  Waiting {waitTime / 1000} seconds before retry..."
        IO.sleep waitTime.toUInt32

  IO.println s!"  Failed to download after {maxAttempts} attempts: {filepath}"
  return false

/-- Left-pad a string with zeros -/
def leftPadZeros (s : String) (width : Nat) : String :=
  let padding := String.ofList (List.replicate (width - s.length) '0')
  padding ++ s

/-- Download data shards with progress tracking -/
def downloadDataShards (cfg : NanoChatConfig) (numShards : Nat) : IO Unit := do
  let dataDir ← cfg.getDataDir
  IO.FS.createDirAll ⟨dataDir⟩

  IO.println s!"Downloading {numShards} shards to {dataDir}..."

  let mut successful := 0
  let mut failed := 0

  for i in [:numShards] do
    let filename := s!"shard_{leftPadZeros (toString i) 5}.parquet"
    let filepath := s!"{dataDir}/{filename}"
    let url := s!"{cfg.dataUrl}/{filename}"

    if ← downloadSingleFile url filepath cfg.maxDownloadRetries then
      successful := successful + 1
    else
      failed := failed + 1

    -- Progress update every 10 shards
    if (i + 1) % 10 == 0 then
      IO.println s!"Progress: {i + 1}/{numShards} shards processed"

  IO.println s!"Download complete: {successful} succeeded, {failed} failed"

  if failed > 0 then
    throw $ IO.userError s!"Failed to download {failed} shards"

/-! ## Tokenizer Training -/

/-- Load text from parquet files for tokenizer training. -/
def loadParquetTexts (dataDir : String) (maxChars : Nat) (docCap : Nat := 10000) : IO (Array String) := do
  let files ← listParquetFiles dataDir

  if files.isEmpty then
    throw $ IO.userError s!"No parquet files found in {dataDir}"

  IO.println s!"Found {files.size} parquet files for tokenizer training"

  let mut texts : Array String := #[]
  let mut totalChars := 0

  for file in files do
    if totalChars >= maxChars then break

    let metadata ← getParquetMetadata file
    IO.println s!"  Processing {file}: {metadata.numRowGroups} row groups"

    for rgIdx in [:metadata.numRowGroups] do
      if totalChars >= maxChars then break

      let rgData ← readRowGroup file rgIdx "text"
      for doc in rgData.documents do
        if totalChars >= maxChars then break
        let docCapped : String :=
          if docCap > 0 && doc.length > docCap then
            (doc.take docCap).toString
          else
            doc
        texts := texts.push docCapped
        totalChars := totalChars + docCapped.length

  IO.println s!"Loaded {texts.size} documents, {totalChars} characters"
  return texts

/-- Train tokenizer on parquet data -/
def trainTokenizer (cfg : NanoChatConfig) : PipelineM Unit := runOnMaster do
  log s!"Training tokenizer with vocab_size={cfg.vocabSize} on up to {cfg.tokenizerMaxChars} chars (doc_cap={cfg.tokenizerDocCap})..."

  let dataDir ← cfg.getDataDir
  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"

  -- Check if tokenizer already exists
  if ← System.FilePath.pathExists ⟨tokenizerFile⟩ then
    log "Tokenizer already exists, skipping training"
    recordMetrics [("status", "cached")]
    return

  -- Load text from parquet files
  let texts ← loadParquetTexts dataDir cfg.tokenizerMaxChars cfg.tokenizerDocCap

  -- Create training config
  let trainConfig : TrainConfig := {
    vocabSize := cfg.vocabSize
    maxChars := cfg.tokenizerMaxChars
    specialTokens := defaultChatSpecialTokens
  }

  -- Train the tokenizer
  let result ← tokenizer.trainBPE texts trainConfig

  -- Create tokenizer directory and save
  IO.FS.createDirAll ⟨tokenizerPath⟩
  log s!"Saving tokenizer to {tokenizerFile}"
  tokenizer.save result.tokenizer tokenizerFile

  recordMetrics [
    ("vocab_size", toString result.stats.finalVocabSize),
    ("total_chars", toString result.stats.totalChars),
    ("num_merges", toString result.stats.numMerges),
    ("compression_ratio", toString result.stats.compressionRatio)
  ]

/-- Evaluate tokenizer compression ratio -/
def evalTokenizer (cfg : NanoChatConfig) : PipelineM Unit := runOnMaster do
  log "Evaluating tokenizer compression ratio..."

  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"

  -- Load the trained tokenizer
  let tok ← tokenizer.load tokenizerFile

  let dataDir ← cfg.getDataDir
  let parquetFiles ← listParquetFiles dataDir

  -- Match nanochat tok_eval style: small fixed probes + tiny train/val excerpts.
  let mut sampleTexts : Array (String × String) := #[
    ("news", "Scientists report a new catalyst that lowers battery charging losses while preserving long cycle life."),
    ("code", "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"),
    ("math", "For n >= 1, sum_{k=1}^n k^3 = (n(n+1)/2)^2."),
    ("science", "Photosynthesis converts light into chemical energy via coupled electron transport and ATP synthesis.")
  ]

  if !parquetFiles.isEmpty then
    let trainRg ← readRowGroup parquetFiles[0]! 0 "text"
    let trainLimit := min 16 trainRg.documents.size
    let trainSample := String.intercalate "\n" ((trainRg.documents.extract 0 trainLimit).toList)
    if !trainSample.isEmpty then
      sampleTexts := sampleTexts.push ("fwe-train", trainSample)

    let valIdx := parquetFiles.size - 1
    if valIdx != 0 then
      let valRg ← readRowGroup parquetFiles[valIdx]! 0 "text"
      let valLimit := min 16 valRg.documents.size
      let valSample := String.intercalate "\n" ((valRg.documents.extract 0 valLimit).toList)
      if !valSample.isEmpty then
        sampleTexts := sampleTexts.push ("fwe-val", valSample)

  -- Count total chars and tokens across evaluation probes.
  let mut totalChars := 0
  let mut totalTokens := 0
  let mut metrics : List (String × String) := []

  for (name, text) in sampleTexts do
    totalChars := totalChars + text.length
    let tokenCount := tokenizer.countTokens tok text
    totalTokens := totalTokens + tokenCount
    let ratio := if tokenCount > 0 then text.length.toFloat / tokenCount.toFloat else 0.0
    metrics := metrics ++ [
      (s!"{name}_chars", toString text.length),
      (s!"{name}_tokens", toString tokenCount),
      (s!"{name}_ratio", toString ratio)
    ]

  let compressionRatio := if totalTokens > 0
    then totalChars.toFloat / totalTokens.toFloat
    else 0.0

  recordMetrics (metrics ++ [
    ("compression_ratio", toString compressionRatio),
    ("sample_chars", toString totalChars),
    ("sample_tokens", toString totalTokens),
    ("sample_count", toString sampleTexts.size),
    ("vocab_size", toString tok.vocabSize)
  ])

/-! ## Pretraining Stage -/

/-- Data split used by parquet pretraining streams. -/
private inductive PretrainParquetSplit where
  | train
  | val
  deriving Repr, BEq

/-- Streaming parquet token state for base pretraining.
    Mirrors nanochat's rank-strided row-group traversal and BOS-prepended tokenization. -/
private structure ParquetTokenStream where
  files : Array String
  tok : BPETokenizer
  bosToken : UInt64
  textColumn : String := "text"
  tokenizerBatchSize : Nat := 128
  rank : Nat := 0
  worldSize : Nat := 1
  fileIdx : Nat := 0
  rowGroupIdx : Nat := 0
  buffer : Array UInt64 := #[]
  consumed : Nat := 0
  epoch : Nat := 0

/-- Serializable cursor for pretraining parquet token streams. -/
private structure ParquetTokenStreamCursor where
  fileIdx : Nat := 0
  rowGroupIdx : Nat := 0
  buffer : Array UInt64 := #[]
  consumed : Nat := 0
  epoch : Nat := 0
  deriving Repr, Inhabited, Lean.ToJson, Lean.FromJson

/-- Rank-local pretraining stream cursor payload persisted alongside checkpoints. -/
private structure PretrainStreamCursors where
  train : ParquetTokenStreamCursor
  val : ParquetTokenStreamCursor
  deriving Repr, Inhabited, Lean.ToJson, Lean.FromJson

private def ParquetTokenStream.captureCursor (s : ParquetTokenStream) : ParquetTokenStreamCursor := {
  fileIdx := s.fileIdx
  rowGroupIdx := s.rowGroupIdx
  buffer := s.buffer
  consumed := s.consumed
  epoch := s.epoch
}

private def ParquetTokenStream.restoreCursor (s : ParquetTokenStream) (c : ParquetTokenStreamCursor)
    : ParquetTokenStream :=
  let fileIdx :=
    if s.files.isEmpty then
      0
    else
      c.fileIdx % s.files.size
  let consumed := min c.consumed c.buffer.size
  {
    s with
    fileIdx := fileIdx
    rowGroupIdx := c.rowGroupIdx
    buffer := c.buffer
    consumed := consumed
    epoch := c.epoch
  }

private def selectParquetFilesForSplit (files : Array String) (split : PretrainParquetSplit) : Array String :=
  match split with
  | .train =>
      if files.size <= 1 then #[] else files.extract 0 (files.size - 1)
  | .val =>
      if files.isEmpty then #[] else #[files[files.size - 1]!]

private def ParquetTokenStream.init
    (dataDir : String) (split : PretrainParquetSplit)
    (tok : BPETokenizer) (bosToken : UInt64)
    (rank worldSize : UInt64)
    (textColumn : String := "text")
    (tokenizerBatchSize : Nat := 128)
    : IO ParquetTokenStream := do
  let allFiles ← listParquetFiles dataDir
  let files := selectParquetFilesForSplit allFiles split
  pure {
    files := files
    tok := tok
    bosToken := bosToken
    textColumn := textColumn
    tokenizerBatchSize := tokenizerBatchSize
    rank := rank.toNat
    worldSize := max 1 worldSize.toNat
    fileIdx := 0
    rowGroupIdx := rank.toNat
  }

private def ParquetTokenStream.available (s : ParquetTokenStream) : Nat :=
  s.buffer.size - s.consumed

private def ParquetTokenStream.compact (s : ParquetTokenStream) : ParquetTokenStream :=
  if s.consumed == 0 then
    s
  else if s.consumed * 2 < s.buffer.size then
    s
  else
    { s with
      buffer := s.buffer.extract s.consumed s.buffer.size
      consumed := 0
    }

private def ParquetTokenStream.pushDocuments
    (stream : ParquetTokenStream) (documents : Array String) : ParquetTokenStream := Id.run do
  let mut st := stream
  let chunkSize := max 1 stream.tokenizerBatchSize
  let mut start := 0
  while start < documents.size do
    let stop := min (start + chunkSize) documents.size
    let chunk := documents.extract start stop
    let mut chunkTokens : Array UInt64 := #[]
    for doc in chunk do
      chunkTokens := chunkTokens.push stream.bosToken
      let ids := tokenizer.encodeWithSpecials stream.tok doc
      for id in ids do
        chunkTokens := chunkTokens.push id.toUInt64
    st := { st with buffer := st.buffer ++ chunkTokens }
    start := stop
  st

private def ParquetTokenStream.readNextRowGroup
    (stream : ParquetTokenStream) : IO (Option RowGroupData × ParquetTokenStream) := do
  if stream.files.isEmpty then
    return (none, stream)

  let mut st := stream
  let mut scanned := 0
  while scanned < st.files.size do
    if st.fileIdx >= st.files.size then
      st := { st with
        fileIdx := 0
        rowGroupIdx := st.rank
        epoch := st.epoch + 1
      }

    let filePath := st.files[st.fileIdx]!
    let metadata ← getParquetMetadata filePath
    if st.rowGroupIdx < metadata.numRowGroups then
      let rg ← readRowGroup filePath st.rowGroupIdx st.textColumn
      let st' := { st with rowGroupIdx := st.rowGroupIdx + st.worldSize }
      return (some rg, st')

    st := { st with
      fileIdx := st.fileIdx + 1
      rowGroupIdx := st.rank
    }
    scanned := scanned + 1

  return (none, st)

private def ParquetTokenStream.fillUntil
    (stream : ParquetTokenStream) (needed : Nat) : IO ParquetTokenStream := do
  let mut st := stream
  while st.available < needed do
    let (rg?, st') ← st.readNextRowGroup
    st := st'
    match rg? with
    | none => return st
    | some rg =>
      st := st.pushDocuments rg.documents
  return st

private def ParquetTokenStream.popWindow?
    (stream : ParquetTokenStream) (n : Nat)
    : Option (Array UInt64 × ParquetTokenStream) := do
  if stream.available < n then
    none
  else
    let start := stream.consumed
    let stop := start + n
    let window := stream.buffer.extract start stop
    let st := { stream with consumed := stop }
    some (window, st.compact)

/-- Emit one `(input,target)` batch for GPT next-token training from parquet text. -/
private def ParquetTokenStream.nextBatch
    (stream : ParquetTokenStream) (batchSize seqLen : Nat)
    : IO (Option DynamicGPTBatch × ParquetTokenStream) := do
  if batchSize == 0 || seqLen == 0 then
    return (none, stream)

  let needed := (batchSize * seqLen + 1)
  let st ← stream.fillUntil needed
  match st.popWindow? needed with
  | none =>
    return (none, st)
  | some (window, st') =>
    let inputIds := window.extract 0 (needed - 1)
    let targetIds := window.extract 1 needed
    let inputTensor ← IO.ofExcept <| Tensor.fromInt64ArrayAs? (inputIds.map (·.toInt64)) #[batchSize, seqLen]
    let targetTensor ← IO.ofExcept <| Tensor.fromInt64ArrayAs? (targetIds.map (·.toInt64)) #[batchSize, seqLen]
    pure (some { batch := batchSize, seq := seqLen, input := inputTensor, target := targetTensor }, st')

/-- Use model config captured in checkpoint metadata when available. -/
private def resolveModelConfigFromCheckpoint
    (fallback : moddedGpt.Config) (checkpointDir : String)
    (label : String := "resume") : PipelineM moddedGpt.Config := do
  let ckptPath := s!"{checkpointDir}/latest.ckpt"
  if !(← checkpointExists ckptPath) then
    return fallback
  match ← loadCheckpointMetadata ckptPath true with
  | some metadata =>
    match metadata.modelConfig? with
    | some savedCfg =>
      if s!"{repr savedCfg}" != s!"{repr fallback}" then
        log s!"{label}: using model config from checkpoint metadata at {ckptPath}"
      return savedCfg
    | none =>
      return fallback
  | none =>
    return fallback

/-- Pretrain base model using ModdedTrain infrastructure -/
def pretrainBaseModel (cfg : NanoChatConfig) : PipelineM Unit := do
  log s!"Pretraining d{cfg.modelDepth} model..."

  let checkpointDir ← cfg.getCheckpointDir "base"
  let modelCfg ← resolveModelConfigFromCheckpoint (cfg.toModelConfig) checkpointDir "pretrain"
  let pretrainEnv := cfg.envState.pretrain
  let defaultHp := Hyperparameters.default
  let numIterations := pretrainEnv.numIterations
  let extensionIterations := pretrainEnv.extensionIterations
  let valInterval := pretrainEnv.valInterval
  let logInterval := pretrainEnv.logInterval
  let deviceBatchSize := pretrainEnv.deviceBatchSize
  let totalBatchSizeTokens := pretrainEnv.totalBatchSizeTokens
  let checkpointInterval := pretrainEnv.checkpointInterval
  let hp : Hyperparameters := {
    defaultHp with
    numIterations := numIterations
    extensionIterations := extensionIterations
    deviceBatchSize := deviceBatchSize
    totalBatchSizeTokens := totalBatchSizeTokens
    embeddingLr := 0.3
    unembeddingLr := 0.004
    matrixLr := 0.02
    adamBeta1 := 0.8
    adamBeta2 := 0.95
    adamWeightDecay := 0.0
    warmupFrac := 0.0
    cooldownFrac := 0.4
    finalLrFrac := 0.0
    maxSeqLen := modelCfg.maxSeqLen
    valInterval := valInterval
    logInterval := logInterval
    checkpointInterval := checkpointInterval
  }
  let dataDir ← resolveDataDirWithOverride cfg pretrainEnv.dataPathOverride
  IO.FS.createDirAll ⟨checkpointDir⟩
  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"
  let tok ← tokenizer.load tokenizerFile
  let bosTokenId ← requireSpecialId tok "BOS" #["<|bos|>", "<|endoftext|>"]
  let pretrainTextColumn := pretrainEnv.textColumn
  let pretrainTokenizerBatchSize := max 1 pretrainEnv.tokenizerBatchSize

  -- Check for existing checkpoint to resume
  let ckptPath := s!"{checkpointDir}/latest.ckpt"
  let hasCheckpoint ← checkpointExists ckptPath

  -- Initialize or resume training state
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  let localRank ← getLocalRankFromEnv
  let trainDevice ← resolveTrainingDevice rank localRank isDistributed
  let isMaster := rank == 0

  -- Build parquet token streams (matches nanochat split convention:
  -- all-but-last shard for train, last shard for val).
  let trainStream0 ← ParquetTokenStream.init
    dataDir .train tok bosTokenId rank worldSize pretrainTextColumn pretrainTokenizerBatchSize
  if trainStream0.files.isEmpty then
    throw <| IO.userError s!"No parquet train shards found in {dataDir} (need at least 2 files: train + val)"
  let valStream0 ← ParquetTokenStream.init
    dataDir .val tok bosTokenId rank worldSize pretrainTextColumn pretrainTokenizerBatchSize

  let trainStreamRef ← IO.mkRef trainStream0
  let valStreamRef ← IO.mkRef valStream0

  let savePretrainStreamCursor : String → IO Unit := fun checkpointPath => do
    let rank ← currentDistributedRank
    let payload : PretrainStreamCursors := {
      train := (← trainStreamRef.get).captureCursor
      val := (← valStreamRef.get).captureCursor
    }
    writeJsonPayload (rankArtifactFile checkpointPath "stream_cursor" "json" rank) payload

  let restorePretrainStreamCursor : String → IO Unit := fun checkpointPath => do
    let rank ← currentDistributedRank
    let cursorPath := rankArtifactFile checkpointPath "stream_cursor" "json" rank
    if !(← cursorPath.pathExists) then
      throw <| IO.userError s!"Missing pretrain stream cursor file for rank {rank}: {cursorPath}"
    let payload : PretrainStreamCursors ← readJsonPayload cursorPath
    let trainStream ← trainStreamRef.get
    trainStreamRef.set (trainStream.restoreCursor payload.train)
    let valStream ← valStreamRef.get
    valStreamRef.set (valStream.restoreCursor payload.val)

  let trainProvider : DynamicGPTBatchProvider := do
    let stream ← trainStreamRef.get
    let (batch?, stream') ← stream.nextBatch hp.deviceBatchSize hp.maxSeqLen
    trainStreamRef.set stream'
    pure batch?

  let valProvider? : Option DynamicGPTBatchProvider :=
    if valStream0.files.isEmpty then
      none
    else
      some <| do
        let stream ← valStreamRef.get
        let (batch?, stream') ← stream.nextBatch hp.deviceBatchSize hp.maxSeqLen
        valStreamRef.set stream'
        pure batch?

  let evalTokens := pretrainEnv.evalTokens
  let tokensPerValBatch := max 1 (hp.deviceBatchSize * hp.maxSeqLen * worldSize.toNat)
  let valBatches : Nat :=
    if valProvider?.isSome then
      max 1 ((evalTokens / tokensPerValBatch))
    else
      0

  -- Run distributed training
  if isMaster then
    let gradAccum := effectiveGradAccumSteps hp worldSize.toNat
    log s!"Starting training with {worldSize} GPUs"
    log s!"Data path: {dataDir}"
    log s!"Pretrain schedule: iters={hp.numIterations}, ext={hp.extensionIterations}, batch={hp.deviceBatchSize}, seq={hp.maxSeqLen}, accum={gradAccum}, val_int={hp.valInterval}, ckpt_int={hp.checkpointInterval}"
    log s!"Pretrain parquet stream: train_files={trainStream0.files.size}, val_files={valStream0.files.size}, text_col={pretrainTextColumn}, tokenizer_batch={pretrainTokenizerBatchSize}, val_batches={valBatches}"
    if hasCheckpoint then
      log s!"Resuming base pretraining from {ckptPath}"

  let _finalState ← trainDistributedWithBatchProvider
    modelCfg hp trainDevice trainProvider valProvider? valBatches checkpointDir none
    (saveStreamCursor? := some savePretrainStreamCursor)
    (restoreStreamCursor? := some restorePretrainStreamCursor)

  -- Estimate parameters for logging
  let numParams := modelCfg.vocabSize * modelCfg.modelDim +  -- embeddings
                   modelCfg.nLayer * (4 * modelCfg.modelDim * modelCfg.modelDim)  -- approx
  let targetTokens := numParams * cfg.paramDataRatio

  recordMetrics [
    ("model_params", s!"{numParams / 1000000}M"),
    ("target_tokens", s!"{targetTokens / 1000000000}B"),
    ("num_layers", toString cfg.modelDepth),
    ("checkpoint_dir", checkpointDir)
  ]

/-- Evaluate base model on CORE benchmark -/
def evalBaseModel (cfg : NanoChatConfig) : PipelineM Unit := runOnMaster do
  log "Evaluating base model on CORE tasks..."

  let checkpointDir ← cfg.getCheckpointDir "base"
  let modelCfg ← resolveModelConfigFromCheckpoint (cfg.toModelConfig) checkpointDir "base-eval"
  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"

  -- Load tokenizer for evaluation
  let tok ← tokenizer.load tokenizerFile
  let bosTokenId ← requireSpecialId tok "bos" #["<|bos|>", "<|endoftext|>"]

  -- Load model checkpoint
  let ckptPath := s!"{checkpointDir}/latest.ckpt"
  let evalDevice ← resolveTrainingDevice 0 (← getLocalRankFromEnv) (← dist.isInitialized)
  let maybeCkpt ← loadCheckpoint modelCfg ckptPath (device := evalDevice)

  match maybeCkpt with
  | none =>
    log "Warning: No checkpoint found, skipping evaluation"
    recordMetrics [("CORE", "N/A"), ("checkpoint", "not found")]
  | some ckpt =>
    log s!"Loaded checkpoint from {ckptPath} at step {ckpt.step}"

    -- Create tokenization function
    let tokenizeFn := fun (text : String) =>
      (tokenizer.encodeWithSpecials tok text).map (·.toUInt64)

    -- Initialize YaRN rotary embeddings for evaluation
    let yarn ← YarnRotary.init modelCfg.headDim modelCfg.maxSeqLen modelCfg.ropeBase (device := evalDevice)

    -- Create model forward function for evaluation
    let runModel : ModelRunner (device := evalDevice) modelCfg.vocabSize := fun {batch seq} input => do
      let inputTyped := input.to evalDevice
      let logits ← moddedGpt.forward (batch := batch) (seq := seq) ckpt.params yarn inputTyped
      pure logits

    -- Evaluate on CORE tasks
    let evalConfig : EvalConfig := {
      bosToken := bosTokenId
      maxSeqLen := some modelCfg.maxSeqLen
      device := evalDevice
      parallelAcrossRanks := false
    }

    let mut taskResults : Array (String × Float) := #[]

    -- Load CORE eval data
    let cacheDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
    let evalBundleDir ← ensureCOREData cacheDir
    let coreConfigs ← loadCOREConfig evalBundleDir
    let evalCap := cfg.envState.eval.coreMaxExamples.map UInt64.toNat

    for taskMeta in coreTasks do
      log s!"  Evaluating {taskMeta.taskName}..."
      -- Load task-specific data from CORE bundle
      let examplesRaw ← match coreTaskMetaToConfig taskMeta coreConfigs with
        | some tc => loadCORETaskData evalBundleDir tc
        | none => pure #[]
      let examples := capArray examplesRaw evalCap
      let score ← evaluateTask examples taskMeta tokenizeFn runModel evalConfig
      taskResults := taskResults.push (taskMeta.taskName, score)

    let coreScore := computeCOREScore taskResults

    recordMetrics [
      ("CORE", toString coreScore),
      ("checkpoint", checkpointDir),
      ("step", toString ckpt.step)
    ]

/-! ## Midtraining Stage -/

/-- Create task mixture for midtraining -/
def createMidtrainTaskMixture (cfg : NanoChatConfig) : IO GenericTaskMixture := do
  -- Get cache directory
  let cacheDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
  let identityPath ← resolveIdentityConversationsPath cacheDir cfg.envState.eval.identityConversationsPath

  IO.println "Creating midtraining task mixture..."

  -- Match nanochat/scripts/mid_train.py defaults.
  let smolTalk ← loadSmolTalkFromHuggingFace "train" cacheDir
  let mmluAux ← loadMMLUFromHuggingFace "auxiliary_train" "train" cacheDir
  let gsm8k ← loadGSM8KFromHuggingFace "main" "train" cacheDir
  let identityTask ← createCustomJSONTask ⟨identityPath⟩

  let simpleSpelling ← createSimpleSpellingTaskWithDownload cacheDir 200000 "train"
  let spellingBee ← createSpellingBeeTaskWithDownload cacheDir 80000 "train"

  let entries : Array torch.Data.TaskClass.MixtureEntry := #[
    entry smolTalk 1,
    entry mmluAux 1,
    entry gsm8k 1,
    entry identityTask 2,
    entry simpleSpelling 1,
    entry spellingBee 1
  ]

  IO.println s!"Created mixture with {entries.size} tasks"
  return GenericTaskMixture.create entries

/-- Create validation task mixture for midtraining (matches nanochat/scripts/mid_train.py). -/
def createMidtrainValTaskMixture (cfg : NanoChatConfig) : IO GenericTaskMixture := do
  let cacheDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))

  IO.println "Creating midtraining validation task mixture..."

  let smolTalkTest ← loadSmolTalkFromHuggingFace "test" cacheDir
  let mmluAllTest ← loadMMLUFromHuggingFace "all" "test" cacheDir
  let gsm8kTest ← loadGSM8KFromHuggingFace "main" "test" cacheDir

  let mmluVal : MMLUTask := {
    mmluAllTest with
    config := { stop := some 5200 }
  }
  let gsm8kVal : GSM8KTask := {
    gsm8kTest with
    config := { stop := some 420 }
  }

  let entries : Array torch.Data.TaskClass.MixtureEntry := #[
    entry smolTalkTest 1,
    entry mmluVal 1,
    entry gsm8kVal 1
  ]

  IO.println s!"Created midtraining validation mixture with {entries.size} tasks"
  return GenericTaskMixture.create entries

/-- Serializable cursor for task-token streaming in midtraining. -/
private structure TaskTokenStreamCursor where
  cursor : Nat := 0
  buffer : Array UInt64 := #[]
  consumed : Nat := 0
  steps : Nat := 0
  lastWrapped : Bool := false
  lastProgress : Float := 0.0
  deriving Repr, Inhabited, Lean.ToJson, Lean.FromJson

/-- Rank-local midtraining stream cursor payload persisted alongside checkpoints. -/
private structure MidtrainStreamCursors where
  train : TaskTokenStreamCursor
  val : TaskTokenStreamCursor
  deriving Repr, Inhabited, Lean.ToJson, Lean.FromJson

private def captureTaskTokenStreamCursor (s : TaskTokenStream) : TaskTokenStreamCursor := {
  cursor := s.cursor
  buffer := s.buffer
  consumed := s.consumed
  steps := s.steps
  lastWrapped := s.lastWrapped
  lastProgress := s.lastProgress
}

private def restoreTaskTokenStreamCursor (s : TaskTokenStream) (c : TaskTokenStreamCursor)
    : TaskTokenStream :=
  let cursor :=
    if s.mixture.size == 0 then
      0
    else
      c.cursor % s.mixture.size
  let consumed := min c.consumed c.buffer.size
  {
    s with
    cursor := cursor
    buffer := c.buffer
    consumed := consumed
    steps := c.steps
    lastWrapped := c.lastWrapped
    lastProgress := c.lastProgress
  }

/-- Create task mixture for chat SFT (matches nanochat/scripts/chat_sft.py). -/
def createSFTTaskMixture (cfg : NanoChatConfig) : IO GenericTaskMixture := do
  let cacheDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
  let identityPath ← resolveIdentityConversationsPath cacheDir cfg.envState.eval.identityConversationsPath

  IO.println "Creating SFT task mixture..."

  let arcEasy ← loadARCFromHuggingFace "ARC-Easy" "train" cacheDir
  let arcChallenge ← loadARCFromHuggingFace "ARC-Challenge" "train" cacheDir
  let gsm8k ← loadGSM8KFromHuggingFace "main" "train" cacheDir
  let smolTalkAll ← loadSmolTalkFromHuggingFace "train" cacheDir
  let smolTalk : SmolTalkTask := {
    smolTalkAll with
    config := { stop := some 10000 }
  }
  let identityTask ← createCustomJSONTask ⟨identityPath⟩
  let simpleSpelling ← createSimpleSpellingTaskWithDownload cacheDir 300 "train"
  let spellingBee ← createSpellingBeeTaskWithDownload cacheDir 300 "train"

  let entries : Array torch.Data.TaskClass.MixtureEntry := #[
    entry arcEasy 1,
    entry arcChallenge 1,
    entry gsm8k 1,
    entry smolTalk 1,
    entry identityTask 1,
    entry simpleSpelling 1,
    entry spellingBee 1
  ]

  IO.println s!"Created SFT mixture with {entries.size} tasks"
  return GenericTaskMixture.create entries

/-- Run midtraining with conversation data and task mixture -/
def midtrain (cfg : NanoChatConfig) : PipelineM Unit := do
  log "Running midtraining (conversation tokens, tool use)..."
  let midEnv := cfg.envState.midtrain

  let baseCheckpointDir ← cfg.getCheckpointDir "base"
  let midCheckpointDir ← cfg.getCheckpointDir "mid"
  IO.FS.createDirAll ⟨midCheckpointDir⟩
  let baseCkptPath := s!"{baseCheckpointDir}/latest.ckpt"
  let midCkptPath := s!"{midCheckpointDir}/latest.ckpt"
  let hasMid ← checkpointExists midCkptPath
  let hasBase ← checkpointExists baseCkptPath

  if !hasMid && !hasBase then
    throw <| IO.userError s!"Base checkpoint not found at {baseCkptPath}"

  let modelCfg ←
    if hasMid then
      resolveModelConfigFromCheckpoint (cfg.toModelConfig) midCheckpointDir "midtrain"
    else if hasBase then
      resolveModelConfigFromCheckpoint (cfg.toModelConfig) baseCheckpointDir "midtrain"
    else
      pure (cfg.toModelConfig)

  -- Midtraining defaults aligned to nanochat speedrun/checkpoints.
  let midDefaultHp : Hyperparameters := {
    Hyperparameters.default with
    numIterations := 811
    extensionIterations := 0
    cooldownFrac := 0.2
    valInterval := 150
    logInterval := 10
  }
  let midIters := midEnv.numIterations
  let midExtIters := midEnv.extensionIterations
  let midValInterval := midEnv.valInterval
  let midLogInterval := midEnv.logInterval
  let midDeviceBatchSize := midEnv.deviceBatchSize
  let midTotalBatchSize := midEnv.totalBatchSizeTokens
  let midCheckpointInterval := midEnv.checkpointInterval
  let hp : Hyperparameters := {
    midDefaultHp with
    numIterations := midIters
    extensionIterations := midExtIters
    deviceBatchSize := midDeviceBatchSize
    totalBatchSizeTokens := midTotalBatchSize
    embeddingLr := 0.2
    unembeddingLr := 0.004
    matrixLr := 0.02
    adamBeta1 := 0.8
    adamBeta2 := 0.95
    adamWeightDecay := 0.0
    warmupFrac := 0.0
    finalLrFrac := 0.0
    maxSeqLen := modelCfg.maxSeqLen
    valInterval := midValInterval
    logInterval := midLogInterval
    checkpointInterval := midCheckpointInterval
  }

  -- Resume policy:
  -- 1) If mid checkpoint exists, continue from it.
  -- 2) Otherwise bootstrap midtraining from base/latest checkpoint.
  let resumePath : Option String :=
    if hasMid then none else some baseCkptPath

  -- Run training loop
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  let localRank ← getLocalRankFromEnv
  let trainDevice ← resolveTrainingDevice rank localRank isDistributed
  let isMaster := rank == 0

  -- Build task mixtures and streaming token providers.
  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"
  let tok ← tokenizer.load tokenizerFile
  let chatTokens ← buildChatTokensFromTokenizer tok
  let encode := fun (text : String) =>
    (tokenizer.encodeWithSpecials tok text).map (·.toUInt64)

  let trainTaskMixture := (← createMidtrainTaskMixture cfg).toConversationMixture
  let valTaskMixture := (← createMidtrainValTaskMixture cfg).toConversationMixture

  let trainStream0 := TaskTokenStream.new
    trainTaskMixture hp.deviceBatchSize hp.maxSeqLen
    chatTokens encode rank.toNat worldSize.toNat
  let valStream0 := TaskTokenStream.new
    valTaskMixture hp.deviceBatchSize hp.maxSeqLen
    chatTokens encode rank.toNat worldSize.toNat
  let trainStreamRef ← IO.mkRef trainStream0
  let valStreamRef ← IO.mkRef valStream0

  let saveMidtrainStreamCursor : String → IO Unit := fun checkpointPath => do
    let rank ← currentDistributedRank
    let payload : MidtrainStreamCursors := {
      train := captureTaskTokenStreamCursor (← trainStreamRef.get)
      val := captureTaskTokenStreamCursor (← valStreamRef.get)
    }
    writeJsonPayload (rankArtifactFile checkpointPath "stream_cursor" "json" rank) payload

  let restoreMidtrainStreamCursor : String → IO Unit := fun checkpointPath => do
    let rank ← currentDistributedRank
    let cursorPath := rankArtifactFile checkpointPath "stream_cursor" "json" rank
    if !(← cursorPath.pathExists) then
      throw <| IO.userError s!"Missing midtrain stream cursor file for rank {rank}: {cursorPath}"
    let payload : MidtrainStreamCursors ← readJsonPayload cursorPath
    let trainStream ← trainStreamRef.get
    trainStreamRef.set (restoreTaskTokenStreamCursor trainStream payload.train)
    let valStream ← valStreamRef.get
    valStreamRef.set (restoreTaskTokenStreamCursor valStream payload.val)

  let trainProvider : DynamicGPTBatchProvider := do
    let stream ← trainStreamRef.get
    let (batch?, stream') ← stream.nextGPTBatch
    trainStreamRef.set stream'
    pure (batch?.map fun (input, target) =>
      { batch := stream.batchSize, seq := stream.seqLen, input, target })

  let valProvider : DynamicGPTBatchProvider := do
    let stream ← valStreamRef.get
    let (batch?, stream') ← stream.nextGPTBatch
    valStreamRef.set stream'
    pure (batch?.map fun (input, target) =>
      { batch := stream.batchSize, seq := stream.seqLen, input, target })

  let evalTokens := midEnv.evalTokens
  let tokensPerValBatch := max 1 (hp.deviceBatchSize * hp.maxSeqLen * worldSize.toNat)
  let valBatches : Nat := max 1 ((evalTokens / tokensPerValBatch))

  if isMaster then
    let gradAccum := effectiveGradAccumSteps hp worldSize.toNat
    log s!"Starting midtraining with {worldSize} GPUs"
    log s!"Midtrain schedule: iters={hp.numIterations}, ext={hp.extensionIterations}, batch={hp.deviceBatchSize}, seq={hp.maxSeqLen}, accum={gradAccum}, val_int={hp.valInterval}, ckpt_int={hp.checkpointInterval}"
    log s!"Midtrain token-stream batches: train_rows={trainTaskMixture.size}, val_rows={valTaskMixture.size}, val_batches={valBatches}"
    if hasMid then
      log s!"Resuming midtraining from {midCkptPath}"
    else
      log s!"Bootstrapping midtraining from base checkpoint {baseCkptPath}"

  let finalState ← trainDistributedWithBatchProvider
    modelCfg hp trainDevice trainProvider (some valProvider) valBatches
    midCheckpointDir resumePath
    (saveStreamCursor? := some saveMidtrainStreamCursor)
    (restoreStreamCursor? := some restoreMidtrainStreamCursor)

  -- If no new step was run (e.g. resume already at horizon), ensure mid/latest exists.
  if isMaster && !(← checkpointExists midCkptPath) then
    let fallbackCkpt : Checkpoint (device := trainDevice) modelCfg := {
      params := finalState.params
      optState := finalState.optState
      step := finalState.step
      bestValLoss := finalState.bestValLoss
    }
    saveCheckpoint fallbackCkpt midCkptPath
    log s!"Materialized fallback mid checkpoint at {midCkptPath}"

  recordMetrics [
    ("steps", toString hp.numIterations),
    ("base_checkpoint", baseCheckpointDir),
    ("mid_checkpoint", midCheckpointDir)
  ]

/-- Evaluate chat model at a checkpoint -/
def evalChat (cfg : NanoChatConfig) (checkpoint : String) : PipelineM Unit := runOnMaster do
  log s!"Evaluating chat model ({checkpoint})..."

  let checkpointDir ← cfg.getCheckpointDir checkpoint
  let modelCfg ← resolveModelConfigFromCheckpoint (cfg.toModelConfig) checkpointDir s!"eval-{checkpoint}"
  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"

  -- Load tokenizer
  let tok ← tokenizer.load tokenizerFile
  let bosTokenId ← requireSpecialId tok "bos" #["<|bos|>", "<|endoftext|>"]

  -- Load checkpoint
  let ckptPath := s!"{checkpointDir}/latest.ckpt"
  let evalDevice ← resolveTrainingDevice 0 (← getLocalRankFromEnv) (← dist.isInitialized)
  let maybeCkpt ← loadCheckpoint modelCfg ckptPath (device := evalDevice)

  match maybeCkpt with
  | none =>
    log s!"Warning: No checkpoint found at {ckptPath}"
    recordMetrics [("status", "checkpoint not found")]
  | some ckpt =>
    log s!"Loaded checkpoint at step {ckpt.step}"

    -- Create tokenization function
    let tokenizeFn := fun (text : String) =>
      (tokenizer.encodeWithSpecials tok text).map (·.toUInt64)

    -- Initialize model inference components
    let yarn ← YarnRotary.init modelCfg.headDim modelCfg.maxSeqLen modelCfg.ropeBase (device := evalDevice)

    -- Create model forward function for evaluation
    let runModel : ModelRunner (device := evalDevice) modelCfg.vocabSize := fun {batch seq} input => do
      let inputTyped := input.to evalDevice
      let logits ← moddedGpt.forward (batch := batch) (seq := seq) ckpt.params yarn inputTyped
      pure logits

    -- Evaluate on multiple task categories
    let evalConfig : EvalConfig := {
      bosToken := bosTokenId
      maxSeqLen := some modelCfg.maxSeqLen
      device := evalDevice
      parallelAcrossRanks := false
    }

    let mut results : Array (String × Float) := #[]

    -- Define chat eval tasks
    let chatTasks : Array TaskMeta := #[
      { taskType := .multipleChoice, numFewshot := 0, taskName := "ARC-Easy" },
      { taskType := .multipleChoice, numFewshot := 0, taskName := "ARC-Challenge" },
      { taskType := .multipleChoice, numFewshot := 5, taskName := "MMLU" },
      { taskType := .languageModeling, numFewshot := 0, taskName := "GSM8K" },
      { taskType := .languageModeling, numFewshot := 0, taskName := "HumanEval" }
    ]

    -- Load CORE eval data for chat evaluation
    let cacheDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
    let evalBundleDir ← ensureCOREData cacheDir
    let coreConfigs ← loadCOREConfig evalBundleDir
    let evalCap := cfg.envState.eval.coreMaxExamples.map UInt64.toNat

    for taskMeta in chatTasks do
      log s!"  Evaluating {taskMeta.taskName}..."
      -- Load task-specific data from CORE bundle
      let examplesRaw ← match coreTaskMetaToConfig taskMeta coreConfigs with
        | some tc => loadCORETaskData evalBundleDir tc
        | none => pure #[]
      let examples := capArray examplesRaw evalCap
      let score ← evaluateTask examples taskMeta tokenizeFn runModel evalConfig
      results := results.push (taskMeta.taskName, score)

    -- Compute aggregate ChatCORE score
    let chatCoreScore := computeCOREScore results

    let resultsList := results.map (fun (name, score) => (name, toString score)) |>.toList
    recordMetrics (resultsList ++ [("ChatCORE", toString chatCoreScore), ("checkpoint", checkpointDir)])

/-! ## Supervised Fine-tuning Stage -/

/-- Run supervised fine-tuning using ChatSFT infrastructure -/
def supervisedFineTune (cfg : NanoChatConfig) : PipelineM Unit := do
  log "Running supervised fine-tuning..."
  let sftEnv := cfg.envState.sft

  let midCheckpointDir ← cfg.getCheckpointDir "mid"
  let modelCfg ← resolveModelConfigFromCheckpoint (cfg.toModelConfig) midCheckpointDir "sft"
  let sftCheckpointDir ← cfg.getCheckpointDir "sft"
  IO.FS.createDirAll ⟨sftCheckpointDir⟩
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  let localRank ← getLocalRankFromEnv
  let trainDevice ← resolveTrainingDevice rank localRank isDistributed
  let runBaseDir := (← get).config.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
  let runArtifacts := RunArtifacts.ofBaseDir runBaseDir
  let sftDeviceBatchSize := max 1 sftEnv.deviceBatchSize
  let sftTargetExamplesPerStep := max 1 sftEnv.targetExamplesPerStep.toNat

  -- SFT configuration
  let sftCfg : SFTConfig := {
    numEpochs := max 1 sftEnv.numEpochs.toNat
    deviceBatchSize := sftDeviceBatchSize
    targetExamplesPerStep := sftTargetExamplesPerStep
    embeddingLr := cfg.sftEmbeddingLr
    matrixLr := cfg.sftMatrixLr
    maxSeqLen := modelCfg.maxSeqLen
    evalEvery := 100
    logInterval := 10
    device := trainDevice
  }

  let deviceStr := match trainDevice.kind with
    | DeviceKind.CPU => "cpu"
    | DeviceKind.MPS => "mps"
    | DeviceKind.CUDA idx => s!"cuda:{idx}"
  log s!"SFT config: epochs={sftCfg.numEpochs}, batch={sftCfg.deviceBatchSize}"
  log s!"Learning rates: embed={sftCfg.embeddingLr}, matrix={sftCfg.matrixLr}"
  log s!"SFT device: {deviceStr}"

  -- Load mid checkpoint
  let ckptPath := s!"{midCheckpointDir}/latest.ckpt"
  log s!"Loading midtraining checkpoint from {ckptPath}"
  let maybeCkpt ← loadCheckpoint modelCfg ckptPath (device := trainDevice)

  match maybeCkpt with
  | none =>
    throw $ IO.userError s!"Midtraining checkpoint not found at {ckptPath}"
  | some midCkpt =>
    log s!"Loaded checkpoint at step {midCkpt.step}"
    let paramsOnDevice ← TensorStruct.makeLeafParams midCkpt.params

    -- Initialize YaRN rotary embeddings
    let yarn ← YarnRotary.init modelCfg.headDim modelCfg.maxSeqLen modelCfg.ropeBase (device := trainDevice)

    -- Keep batch and sequence indices across the SFT callback boundary.
    let forwardFn : ModdedGPTParams (device := trainDevice) modelCfg → {batch seq : Nat} →
        Tensor { shape := #[batch, seq], dtype := .Int64, device := trainDevice } → IO (Σ vocab : Nat, Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := trainDevice }) :=
      fun params {batch seq} input => do
        let logits ← moddedGpt.forward (batch := batch) (seq := seq) params yarn (input.to trainDevice)
        pure ⟨modelCfg.vocabSize, logits⟩

    let lossFn : {batch seq vocab : Nat} → Tensor { shape := #[batch, seq, vocab], dtype := .Float32, device := trainDevice } →
        Tensor { shape := #[batch, seq], dtype := .Int64, device := trainDevice } → Tensor { shape := #[batch, seq], dtype := .Float32, device := trainDevice } → Tensor { shape := #[], dtype := .Float32, device := trainDevice } :=
      fun logits targets mask => maskedCrossEntropy logits targets mask

    -- Initialize AdamW optimizer
    let opt := Optim.adamw (lr := sftCfg.embeddingLr) (b1 := 0.9) (b2 := 0.999)
    let optState := opt.init paramsOnDevice

    -- Load tokenizer
    let tokenizerPath ← cfg.getTokenizerPath
    let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"
    let tok ← tokenizer.load tokenizerFile

    -- Create task mixture from HuggingFace data
    let sftMixture ← createSFTTaskMixture cfg
    let taskMixture := sftMixture.toConversationMixtureSharded rank.toNat worldSize.toNat

    -- Create chat tokens from tokenizer special tokens
    let chatTokens ← buildChatTokensFromTokenizer tok

    -- Create encode function
    let encode := fun (text : String) => (tokenizer.encodeWithSpecials tok text).map (·.toUInt64)

    -- Create task iterator
    let taskIterator := TaskIterator.new taskMixture sftCfg.deviceBatchSize sftCfg.maxSeqLen chatTokens encode
    let trainDataFn ← makeTaskDataGenerator taskIterator sftCfg.maxSeqLen chatTokens.assistantEnd

    -- Get number of training examples from mixture (optionally capped for fast validation runs)
    let totalTrainExamples := taskMixture.size
    let sftMaxExamples := sftEnv.maxExamples.map UInt64.toNat
    let numTrainExamples :=
      match sftMaxExamples with
      | some n => max 1 (min n totalTrainExamples)
      | none => totalTrainExamples
    log s!"Training on {numTrainExamples}/{totalTrainExamples} local task examples (rank={rank}, world={worldSize})"

    -- Run SFT training loop
    log "Starting SFT training..."
    let sftCallbacks := consoleSFTCallbacks.combine (torch.Train.ChatSFT.artifactCallbacks runArtifacts "sft")
    let (finalParams, finalOptState, finalState) ← trainLoop sftCfg paramsOnDevice optState
      forwardFn lossFn trainDataFn none numTrainExamples sftCallbacks

    log s!"SFT complete: {finalState.totalTokens} tokens trained"

    -- Save SFT checkpoint
    if rank == 0 then
      let sftOptState : OptimizerState (device := trainDevice) modelCfg := {
        midCkpt.optState with
        adamState := finalOptState
        step := finalState.step.toUInt64
      }
      let sftCkpt : Checkpoint (device := trainDevice) modelCfg := {
        params := finalParams
        optState := sftOptState
        step := finalState.step.toUInt64
        bestValLoss := finalState.bestValLoss
      }
      saveCheckpoint sftCkpt s!"{sftCheckpointDir}/latest.ckpt"
      appendCheckpointEvent runArtifacts {
        name := "sft_latest"
        path := s!"{sftCheckpointDir}/latest.ckpt"
        kind := "model"
        step := some finalState.step
        metadata := [
          metricStr "stage" "sft",
          metricFloat "best_val_loss" finalState.bestValLoss
        ]
      }

    if rank == 0 then
      recordMetrics [
        ("epochs", toString sftCfg.numEpochs),
        ("batch_size", toString sftCfg.deviceBatchSize),
        ("embedding_lr", toString sftCfg.embeddingLr),
        ("matrix_lr", toString sftCfg.matrixLr),
        ("total_tokens", toString finalState.totalTokens),
        ("best_val_loss", toString finalState.bestValLoss),
        ("mid_checkpoint", midCheckpointDir),
        ("sft_checkpoint", sftCheckpointDir)
      ]
    if isDistributed then
      dist.barrier

/-! ## Reinforcement Learning Stage -/

/-- Run GRPO reinforcement learning on GSM8K -/
def reinforcementLearning (cfg : NanoChatConfig) : PipelineM Unit := do
  log "Running reinforcement learning (GRPO on GSM8K)..."
  let rlEnv := cfg.envState.rl

  let sftCheckpointDir ← cfg.getCheckpointDir "sft"
  let modelCfg ← resolveModelConfigFromCheckpoint (cfg.toModelConfig) sftCheckpointDir "grpo"
  let rlCheckpointDir ← cfg.getCheckpointDir "rl"
  let tokenizerPath ← cfg.getTokenizerPath
  let tokenizerFile := s!"{tokenizerPath}/tokenizer.bin"
  IO.FS.createDirAll ⟨rlCheckpointDir⟩
  let isDistributed ← dist.isInitialized
  let (rank, worldSize) ← if isDistributed then dist.getRankAndWorldSize else pure (0, 1)
  let runBaseDir := (← get).config.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
  let runArtifacts := RunArtifacts.ofBaseDir runBaseDir
  let localRank ← getLocalRankFromEnv
  let trainDevice ← resolveTrainingDevice rank localRank isDistributed

  let grpoExamplesPerStepGlobal := max 1 rlEnv.examplesPerStepGlobal.toNat
  let grpoExamplesPerStep :=
    if isDistributed && worldSize > 1 then
      max 1 (grpoExamplesPerStepGlobal / worldSize.toNat)
    else
      grpoExamplesPerStepGlobal
  let grpoEvalEvery := max 1 rlEnv.evalEvery.toNat
  let grpoLogEvery := max 1 rlEnv.logEvery.toNat
  let grpoEpochs := max 1 rlEnv.epochs.toNat
  let grpoMaxPrompts := rlEnv.maxPrompts.map UInt64.toNat

  -- GRPO configuration
  let grpoCfg : GRPOConfig := {
    numSamples := max 1 rlEnv.numSamples.toNat
    maxNewTokens := max 1 rlEnv.maxNewTokens
    temperature := 1.0
    topK := 50
    examplesPerStep := grpoExamplesPerStep
    evalEvery := grpoEvalEvery
    logEvery := grpoLogEvery
  }

  -- Load tokenizer
  let tok ← tokenizer.load tokenizerFile
  let bosTokenId ← requireSpecialId tok "bos" #["<|bos|>", "<|endoftext|>"]
  let assistantEndId ← requireSpecialId tok "assistant_end" #["<|assistant_end|>", "<|eot|>"]
  let grpoCfg : GRPOConfig := {
    grpoCfg with
    eosToken := bosTokenId.toNat
    padToken := assistantEndId.toNat
  }

  log s!"GRPO config: samples={grpoCfg.numSamples}, max_tokens={grpoCfg.maxNewTokens}, examples_per_step(local)={grpoCfg.examplesPerStep}, examples_per_step(global)={grpoExamplesPerStepGlobal}, eval_every={grpoCfg.evalEvery}, log_every={grpoCfg.logEvery}, epochs={grpoEpochs}, eos={grpoCfg.eosToken}, pad={grpoCfg.padToken}"

  -- Load SFT checkpoint
  let ckptPath := s!"{sftCheckpointDir}/latest.ckpt"
  log s!"Loading SFT checkpoint from {ckptPath}"
  let maybeCkpt ← loadCheckpoint modelCfg ckptPath (device := trainDevice)

  match maybeCkpt with
  | none =>
    throw $ IO.userError s!"SFT checkpoint not found at {ckptPath}"
  | some sftCkpt =>
    log s!"Loaded checkpoint at step {sftCkpt.step}"
    let paramsOnDevice ← TensorStruct.makeLeafParams sftCkpt.params
    let adamStateOnDevice := sftCkpt.optState.adamState

    -- Initialize model components
    let yarn ← YarnRotary.init modelCfg.headDim modelCfg.maxSeqLen modelCfg.ropeBase (device := trainDevice)

    -- Create decode function
    let decodeFn := fun (tokens : Array UInt64) =>
      tokenizer.decode tok (tokens.map (·.toUInt32))

    -- Create generation function (single token sampling)
    let generateOneFn := fun (params : ModdedGPTParams (device := trainDevice) modelCfg) (context : Array UInt64) => do
      let inputTensor := Tensor.fromInt64Array (context.map (·.toInt64))
      let inputReshaped := (inputTensor.reshape #[1, context.size]).to trainDevice
      let logits ← moddedGpt.forward params yarn inputReshaped
      -- Flatten logits and extract last position's vocab slice
      let logitsFlat := logits.reshape #[context.size * modelCfg.vocabSize]
      let startIdx := (context.size - 1) * modelCfg.vocabSize
      let lastLogits := logitsFlat.sliceChecked 0 startIdx modelCfg.vocabSize #[modelCfg.vocabSize]
      let rowLogits := lastLogits.reshape #[1, modelCfg.vocabSize]
      let scaled := if grpoCfg.temperature == 1.0 then rowLogits
                    else Tensor.mulScalar rowLogits (1.0 / grpoCfg.temperature)
      let filtered := if grpoCfg.topK == 0 then scaled
                      else Tensor.topKFilter scaled grpoCfg.topK
      let probs := Tensor.softmax filtered (-1)
      let sampled ← Tensor.multinomial probs 1
      let sampled := sampled.reshape #[]
      return (Tensor.itemInt sampled).toUInt64

    -- Create forward function for policy gradient
    -- Returns per-token NLL: (input, target) → nll_per_token
    let forwardFn := fun (params : ModdedGPTParams (device := trainDevice) modelCfg) (input : Tensor { shape := #[1, 256], dtype := .Int64, device := trainDevice }) (target : Tensor { shape := #[1, 256], dtype := .Int64, device := trainDevice }) => do
      let logits ← moddedGpt.forward params yarn (input.to trainDevice)
      -- Compute per-token cross-entropy (NLL)
      let logitsFlat := logits.reshape #[256, modelCfg.vocabSize]
      let targetFlat := (target.to trainDevice).reshape #[256]
      let nllFlat ← IO.ofExcept (Tensor.crossEntropyNone? logitsFlat targetFlat)
      return nllFlat.reshape #[1, 256]

    -- Load GSM8K prompts from HuggingFace
    let cacheDir := cfg.baseDir.replace "~" (← IO.getEnv "HOME" |>.map (·.getD ""))
    let gsm8kTask ← loadGSM8KFromHuggingFace "main" "train" cacheDir

    -- Convert GSM8K examples to (tokenized_prompt, question, expected_answer) tuples
    let gsm8kPromptsAll : Array (Array UInt64 × String × String) :=
      gsm8kTask.examples.filterMap fun ex => do
        let promptTokens := (tokenizer.encodeWithSpecials tok ex.question).map (·.toUInt64)
        -- Extract the numerical answer after "#### " marker
        let expectedAnswer ← extractHashAnswer ex.answer
        some (promptTokens, ex.question, expectedAnswer)

    let gsm8kPromptsPreShard :=
      match grpoMaxPrompts with
      | some n =>
        let cap := min n gsm8kPromptsAll.size
        gsm8kPromptsAll.extract 0 cap
      | none => gsm8kPromptsAll

    let gsm8kPrompts :=
      if isDistributed && worldSize > 1 then
        Id.run do
          let mut localPrompts : Array (Array UInt64 × String × String) := #[]
          for idx in [:gsm8kPromptsPreShard.size] do
            if idx % worldSize.toNat == rank.toNat then
              localPrompts := localPrompts.push gsm8kPromptsPreShard[idx]!
          localPrompts
      else
        gsm8kPromptsPreShard

    if gsm8kPrompts.isEmpty then
      throw <| IO.userError "No GSM8K prompts available for GRPO"

    log s!"Loaded {gsm8kPrompts.size} GSM8K prompts for GRPO"

    -- Math reward function (from GRPO module)
    let rewardFn := fun (expected : String) (response : String) =>
      (mathReward expected response).reward

    -- Run GRPO training
    log "Starting GRPO training..."
    let numEpochs := grpoEpochs  -- Multiple passes over GSM8K

    let initAdamState := adamStateOnDevice
    let grpoCallbacks := consoleGRPOCallbacks.combine (torch.RL.GRPO.artifactCallbacks runArtifacts "grpo")
    let (finalParams, finalAdamState, result) ← trainOnPromptsWithUpdates 1 256
      gsm8kPrompts
      generateOneFn
      forwardFn
      paramsOnDevice
      initAdamState
      decodeFn
      rewardFn
      grpoCfg
      numEpochs
      grpoCallbacks

    log s!"GRPO complete: best pass@1 = {result.bestPass1}"

    -- Save RL checkpoint
    if rank == 0 then
      let rlOptState : OptimizerState (device := trainDevice) modelCfg := {
        sftCkpt.optState with
        adamState := finalAdamState
        step := result.totalSteps.toUInt64
      }
      let rlCkpt : Checkpoint (device := trainDevice) modelCfg := {
        params := finalParams
        optState := rlOptState
        step := result.totalSteps.toUInt64
        bestValLoss := sftCkpt.bestValLoss
      }
      saveCheckpoint rlCkpt s!"{rlCheckpointDir}/latest.ckpt"
      appendCheckpointEvent runArtifacts {
        name := "rl_latest"
        path := s!"{rlCheckpointDir}/latest.ckpt"
        kind := "model"
        step := some result.totalSteps
        metadata := [
          metricStr "stage" "rl",
          metricFloat "best_pass_at_1" result.bestPass1
        ]
      }

    if rank == 0 then
      recordMetrics [
        ("num_samples", toString grpoCfg.numSamples),
        ("max_new_tokens", toString grpoCfg.maxNewTokens),
        ("temperature", toString grpoCfg.temperature),
        ("best_pass1", toString result.bestPass1),
        ("final_pass1", toString result.finalPass1),
        ("total_steps", toString result.totalSteps),
        ("sft_checkpoint", sftCheckpointDir),
        ("rl_checkpoint", rlCheckpointDir)
      ]
    if isDistributed then
      dist.barrier

/-! ## Utility Functions -/

/-- Print banner (master only) -/
def printBanner : PipelineM Unit := do
  log "                                                       █████                █████"
  log "                                                      ░░███                ░░███"
  log "     ████████    ██████   ████████    ██████   ██████  ░███████    ██████  ███████"
  log "    ░░███░░███  ░░░░░███ ░░███░░███  ███░░███ ███░░███ ░███░░███  ░░░░░███░░░███░"
  log "     ░███ ░███   ███████  ░███ ░███ ░███ ░███░███ ░░░  ░███ ░███   ███████  ░███"
  log "     ░███ ░███  ███░░███  ░███ ░███ ░███ ░███░███  ███ ░███ ░███  ███░░███  ░███ ███"
  log "     ████ █████░░████████ ████ █████░░██████ ░░██████  ████ █████░░███████  ░░█████"
  log "    ░░░░ ░░░░░  ░░░░░░░░ ░░░░ ░░░░░  ░░░░░░   ░░░░░░  ░░░░ ░░░░░  ░░░░░░░░   ░░░░░"

/-- Estimate training cost based on GPU type and duration -/
def estimateCost (gpuType : String) (numGpus : Nat) (hoursEstimate : Float) : PipelineM Unit := do
  let hourlyRate := match gpuType with
    | "H100" => 3.00
    | "A100" => 1.79
    | "V100" => 0.55
    | _ => 2.00
  let totalCost := hourlyRate * numGpus.toFloat * hoursEstimate
  log s!"Estimated cost: ${totalCost} ({numGpus}x {gpuType} @ ${hourlyRate}/GPU/hour)"

/-! ## Full Pipeline -/

/-- Run the complete NanoChat training pipeline -/
def runNanoChatPipeline (cfg : NanoChatConfig) : PipelineM Unit := do
  printBanner

  log ""
  log "Starting NanoChat training pipeline..."
  log s!"Model: d{cfg.modelDepth}, vocab: {cfg.vocabSize}"
  log s!"Base directory: {cfg.baseDir}"
  log ""

  -- Stage 1: Data Download
  -- Download initial data for tokenizer training
  stage "data-download-initial" do
    runOnMaster do
      downloadDataShards cfg cfg.initialDataShards

  -- Spawn background download of remaining shards while tokenizer trains
  let moreDataTask ← background "download-remaining-shards" do
    runOnMasterIO do
      downloadDataShards cfg cfg.numDataShards

  -- Stage 2: Tokenizer
  stage "tokenizer-training" do
    trainTokenizer cfg

  stage "tokenizer-eval" do
    evalTokenizer cfg

  -- Stage 3: Base model pretraining
  -- Wait for all data shards before pretraining
  stage "data-download-complete" do
    await moreDataTask
    let isDistributed ← dist.isInitialized
    if isDistributed then
      dist.barrier
    log s!"All {cfg.numDataShards} data shards ready"

  stage "pretrain" do
    pretrainBaseModel cfg

  stage "pretrain-eval" do
    evalBaseModel cfg

  -- Stage 4: Midtraining
  stage "midtrain" do
    midtrain cfg

  stage "midtrain-eval" do
    evalChat cfg "mid"

  -- Stage 5: Supervised Fine-tuning
  stage "sft" do
    supervisedFineTune cfg

  stage "sft-eval" do
    evalChat cfg "sft"

  -- Stage 6: Reinforcement Learning (optional)
  if cfg.enableRL then
    stage "rl" do
      reinforcementLearning cfg
    stage "rl-eval" do
      evalChat cfg "rl"
  else
    log "RL stage disabled (set ENABLE_RL=1 to enable)"

  log ""
  log "Pipeline complete!"

/-- Quick pipeline for testing (reduced settings) -/
def runQuickPipeline (cfg : NanoChatConfig) : PipelineM Unit := do
  log "Running QUICK pipeline (testing mode)..."
  let quickCfg : NanoChatConfig := {
    cfg with
    modelDepth := 4
    envState := {
      cfg.envState with
      model := { cfg.envState.model with depth := 4 }
    }
  }

  -- Skip data download, use existing data
  stage "tokenizer-training" do
    trainTokenizer quickCfg

  stage "pretrain" do
    pretrainBaseModel quickCfg

  stage "midtrain" do
    midtrain quickCfg

  stage "sft" do
    supervisedFineTune quickCfg

  log "Quick pipeline complete!"

/-- Main entry point for NanoChat pipeline -/
def main : IO Unit := do
  let envState ← loadNanoChatEnvState
  let pipelineEnv := envState.pipeline
  let modelEnv := envState.model

  let cfg : NanoChatConfig := {
    baseDir := pipelineEnv.baseDir
    wandbEnabled := pipelineEnv.wandbEnabled
    wandbRun := pipelineEnv.wandbRun
    numGpus := pipelineEnv.numGpus.toNat
    modelDepth := modelEnv.depth.toNat
    modelAspectRatio := modelEnv.aspectRatio.toNat
    targetHeadDim := modelEnv.targetHeadDim.toNat
    maxSeqLen := modelEnv.maxSeqLen
    ropeBase := modelEnv.ropeBase
    paramDataRatio := modelEnv.paramDataRatio.toNat
    vocabSize := modelEnv.vocabSize.toNat
    tokenizerMaxChars := modelEnv.tokenizerMaxChars.toNat
    tokenizerDocCap := modelEnv.tokenizerDocCap.toNat
    dataPath := pipelineEnv.dataPath
    numDataShards := pipelineEnv.numDataShards.toNat
    initialDataShards := pipelineEnv.initialDataShards.toNat
    sftEpochs := envState.sft.numEpochs.toNat
    grpoNumSamples := envState.rl.numSamples.toNat
    grpoMaxNewTokens := envState.rl.maxNewTokens
    resumeFromCheckpoint := true
    checkpointAfterStage := true
    enableRL := pipelineEnv.enableRL
    envState := envState
  }

  IO.println "╔════════════════════════════════════════╗"
  IO.println "║           NanoChat Pipeline            ║"
  IO.println "╚════════════════════════════════════════╝"
  let modelCfg := cfg.toModelConfig
  IO.println s!"Configuration:"
  IO.println s!"  Model depth: {cfg.modelDepth}"
  IO.println s!"  Model dim: {modelCfg.modelDim} (heads={modelCfg.nHead}, head_dim={modelCfg.headDim})"
  IO.println s!"  Max seq len: {modelCfg.maxSeqLen}, rope base: {cfg.ropeBase}"
  IO.println s!"  Vocab size: {cfg.vocabSize}"
  IO.println s!"  Param/data ratio: {cfg.paramDataRatio}"
  IO.println s!"  Data path: {cfg.dataPath}"
  IO.println s!"  Num GPUs: {cfg.numGpus}"
  IO.println s!"  Data shards: {cfg.numDataShards}"
  IO.println s!"  Initial shards: {cfg.initialDataShards}"
  IO.println s!"  Tokenizer max chars: {cfg.tokenizerMaxChars}"
  IO.println s!"  Tokenizer doc cap: {cfg.tokenizerDocCap}"
  IO.println s!"  SFT epochs: {cfg.sftEpochs}"
  IO.println s!"  GRPO samples: {cfg.grpoNumSamples}, max_new_tokens: {cfg.grpoMaxNewTokens}"
  IO.println s!"  Quick mode: {pipelineEnv.quickMode}"
  IO.println s!"  RL enabled: {cfg.enableRL}"
  IO.println ""

  withDistributedWithHandlers cfg.toPipelineConfig consolePipelineLogHandlers do
    if pipelineEnv.quickMode then
      runQuickPipeline cfg
    else
      runNanoChatPipeline cfg

end torch.NanoChat.Pipeline

/-- Executable entrypoint wrapper. -/
unsafe def main (_args : List String) : IO UInt32 := do
  torch.NanoChat.Pipeline.main
  pure 0
