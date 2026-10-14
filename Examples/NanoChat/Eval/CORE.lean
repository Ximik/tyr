/-
  Tyr/Eval/CORE.lean

  CORE metric evaluation framework based on nanochat's core_eval.py.

  The CORE metric (from DCLM paper: arxiv.org/abs/2406.11794) evaluates
  models on a standardized set of tasks using loss-based scoring.

  Supports three task types:
  - multiple_choice: Select option with lowest average continuation loss
  - schema: Context varies, continuation same (common suffix)
  - language_modeling: Predict continuation tokens exactly

  Key features:
  - Few-shot example rendering
  - Common prefix/suffix detection for efficient batching
  - Distributed evaluation support
-/
import Tyr.Typed.Neural
import Tyr.Distributed

namespace torch.Eval.CORE

open torch

/-- Evaluation models preserve the supplied batch and sequence dimensions and
expose their actual vocabulary size in the result type. -/
abbrev ModelRunner (vocab : UInt64) := {batch seq : UInt64} →
  Tensor { shape := #[batch, seq], dtype := .Int64 } → IO (Tensor { shape := #[batch, seq, vocab], dtype := .Float32 })

/-! ## Task Types -/

/-- Types of evaluation tasks -/
inductive TaskType where
  | multipleChoice   : TaskType
  | schema           : TaskType
  | languageModeling : TaskType
  deriving Repr, BEq, Inhabited

/-- Task metadata for evaluation -/
structure TaskMeta where
  /-- Task type (MC, schema, LM) -/
  taskType : TaskType
  /-- Number of few-shot examples -/
  numFewshot : Nat := 0
  /-- Delimiter between context and continuation -/
  continuationDelimiter : String := ""
  /-- Task name for logging -/
  taskName : String := "unknown"
  deriving Repr, Inhabited

/-! ## Data Items -/

/-- Multiple choice item -/
structure MCItem where
  /-- Query/question text -/
  query : String
  /-- List of choice texts -/
  choices : Array String
  /-- Index of correct answer (0-indexed) -/
  gold : Nat
  deriving Repr, Inhabited

/-- Schema item (varying context, same continuation) -/
structure SchemaItem where
  /-- List of context options -/
  contextOptions : Array String
  /-- Common continuation -/
  continuation : String
  /-- Index of correct context (0-indexed) -/
  gold : Nat
  deriving Repr, Inhabited

/-- Language modeling item -/
structure LMItem where
  /-- Context/prefix text -/
  context : String
  /-- Continuation to predict -/
  continuation : String
  deriving Repr, Inhabited

/-- Generic evaluation item (union type) -/
inductive EvalItem where
  | mc : MCItem → EvalItem
  | schema : SchemaItem → EvalItem
  | lm : LMItem → EvalItem
  deriving Repr, Inhabited

/-! ## Prompt Rendering -/

/-- Render few-shot examples for multiple choice -/
def renderFewshotMC (examples : Array MCItem) (delimiter : String) : String := Id.run do
  let mut result := ""
  for ex in examples do
    let answer := ex.choices[ex.gold]!
    result := result ++ ex.query ++ delimiter ++ answer ++ "\n\n"
  result

/-- Render prompts for multiple choice question.
    Returns one prompt per choice option. -/
def renderPromptsMC (item : MCItem) (delimiter : String) (fewshot : Array MCItem) : Array String := Id.run do
  let fewshotStr := renderFewshotMC fewshot delimiter
  let mut prompts := #[]
  for choice in item.choices do
    prompts := prompts.push (fewshotStr ++ item.query ++ delimiter ++ choice)
  prompts

/-- Render few-shot examples for schema tasks -/
def renderFewshotSchema (examples : Array SchemaItem) (delimiter : String) : String := Id.run do
  let mut result := ""
  for ex in examples do
    let context := ex.contextOptions[ex.gold]!
    result := result ++ context ++ delimiter ++ ex.continuation ++ "\n\n"
  result

/-- Render prompts for schema question.
    Returns one prompt per context option. -/
def renderPromptsSchema (item : SchemaItem) (delimiter : String) (fewshot : Array SchemaItem)
    : Array String := Id.run do
  let fewshotStr := renderFewshotSchema fewshot delimiter
  let mut prompts := #[]
  for ctx in item.contextOptions do
    prompts := prompts.push (fewshotStr ++ ctx ++ delimiter ++ item.continuation)
  prompts

/-- Render few-shot examples for LM tasks -/
def renderFewshotLM (examples : Array LMItem) (delimiter : String) : String := Id.run do
  let mut result := ""
  for ex in examples do
    result := result ++ ex.context.trimAscii.toString ++ delimiter ++ ex.continuation ++ "\n\n"
  result

/-- Render prompts for language modeling task.
    Returns two prompts: without continuation (for finding prefix length) and with. -/
def renderPromptsLM (item : LMItem) (delimiter : String) (fewshot : Array LMItem)
    : (String × String) :=
  let fewshotStr := renderFewshotLM fewshot delimiter
  let promptWithout := fewshotStr ++ item.context.trimAscii.toString ++ delimiter
  let promptWith := promptWithout ++ item.continuation
  (promptWithout.trimAscii.toString, promptWith)

/-! ## Sequence Processing -/

/-- Find length of common prefix across token sequences -/
def findCommonPrefixLength (sequences : Array (Array UInt64)) : Nat := Id.run do
  if sequences.isEmpty then return 0
  let minLen := sequences.foldl (fun acc seq => min acc seq.size) sequences[0]!.size
  for i in [:minLen] do
    let firstToken := sequences[0]![i]!
    for seq in sequences do
      if seq[i]! != firstToken then
        return i
  minLen

/-- Find length of common suffix across token sequences -/
def findCommonSuffixLength (sequences : Array (Array UInt64)) : Nat := Id.run do
  if sequences.isEmpty then return 0
  let minLen := sequences.foldl (fun acc seq => min acc seq.size) sequences[0]!.size
  for i in [:minLen] do
    let idx := sequences[0]!.size - 1 - i
    let firstToken := sequences[0]![idx]!
    for seq in sequences do
      let seqIdx := seq.size - 1 - i
      if seq[seqIdx]! != firstToken then
        return i
  minLen

/-- Pad sequences to same length -/
def padSequences (sequences : Array (Array UInt64)) (padToken : UInt64) : Array (Array UInt64) := Id.run do
  if sequences.isEmpty then return #[]
  let maxLen := sequences.foldl (fun acc seq => max acc seq.size) 0
  let mut result := #[]
  for seq in sequences do
    let padding := (List.replicate (maxLen - seq.size) padToken).toArray
    result := result.push (seq ++ padding)
  result

/-- Convert padded sequences to input tensor -/
def sequencesToTensor {batch seq : UInt64} (sequences : Array (Array UInt64))
    : Tensor { shape := #[batch, seq], dtype := .Int64 } :=
  -- Flatten and convert to int64 array
  let flat := sequences.foldl (fun acc seq => acc ++ seq.map (·.toInt64)) #[]
  let tensor := Tensor.fromInt64Array flat
  Tensor.reshapeChecked tensor #[batch, seq]

/-! ## Batching -/

/-- Batch info for evaluation -/
structure BatchInfo where
  /-- Token sequences (one per option/choice) -/
  tokens : Array (Array UInt64)
  /-- Start index of continuation for each sequence -/
  startIndices : Array Nat
  /-- End index of continuation for each sequence -/
  endIndices : Array Nat
  deriving Repr

/-- Batch sequences for multiple choice (common prefix) -/
def batchSequencesMC (tokenize : String → Array UInt64) (prompts : Array String) (bosToken : UInt64)
    : BatchInfo := Id.run do
  let mut tokens := #[]
  for prompt in prompts do
    tokens := tokens.push (#[bosToken] ++ tokenize prompt)

  let commonPrefix := findCommonPrefixLength tokens
  let startIndices := tokens.map (fun _ => commonPrefix)
  let endIndices := tokens.map (·.size)

  { tokens, startIndices, endIndices }

/-- Batch sequences for schema (common suffix) -/
def batchSequencesSchema (tokenize : String → Array UInt64) (prompts : Array String) (bosToken : UInt64)
    : BatchInfo := Id.run do
  let mut tokens := #[]
  for prompt in prompts do
    tokens := tokens.push (#[bosToken] ++ tokenize prompt)

  let commonSuffix := findCommonSuffixLength tokens
  let endIndices := tokens.map (·.size)
  let startIndices := endIndices.map (· - commonSuffix)

  { tokens, startIndices, endIndices }

/-- Batch sequences for language modeling -/
def batchSequencesLM (tokenize : String → Array UInt64) (promptWithout promptWith : String) (bosToken : UInt64)
    : BatchInfo :=
  let tokensWithout := #[bosToken] ++ tokenize promptWithout
  let tokensWith := #[bosToken] ++ tokenize promptWith

  { tokens := #[tokensWith]
    startIndices := #[tokensWithout.size]
    endIndices := #[tokensWith.size] }

/-! ## Evaluation -/

/-- Result of evaluating a single example -/
structure EvalResult where
  /-- Whether the prediction was correct -/
  correct : Bool
  /-- Loss values for each option (MC/schema) or continuation tokens (LM) -/
  losses : Array Float
  /-- Predicted index (MC/schema) or tokens (not used for LM) -/
  prediction : Nat
  deriving Repr

/-- Compute per-token losses from model output.
    Returns losses tensor where losses[i] is loss for predicting token i+1 from position i. -/
def computeLosses {batch seq vocab : UInt64} (logits : Tensor { shape := #[batch, seq, vocab], dtype := .Float32 })
    (targetIds : Tensor { shape := #[batch, seq], dtype := .Int64 }) : Tensor { shape := #[batch, seq], dtype := .Float32 } :=
  -- Compute cross-entropy at each position
  -- logits: [batch, seq, vocab], targets: [batch, seq]
  -- Result: [batch, seq] (loss at each position)
  let batchSeq := batch * seq
  let logitsFlat := Tensor.reshapeChecked logits #[batchSeq, vocab]
  let targetsFlat := Tensor.reshapeChecked targetIds #[batchSeq]
  let lossesFlat := Tensor.crossEntropyNone logitsFlat targetsFlat
  Tensor.reshapeChecked lossesFlat #[batch, seq]

/-- Evaluate multiple choice example.
    Returns index of choice with lowest average continuation loss.

    For each option:
    1. Get logits for that sequence in the batch
    2. Compute cross-entropy loss at each position
    3. Average the losses over the continuation region (startIdx to endIdx)
    4. Pick the option with lowest average loss -/
def evaluateMC {batch seq vocab : UInt64}
    (logits : Tensor { shape := #[batch, seq, vocab], dtype := .Float32 })
    (targetIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (batchInfo : BatchInfo)
    (goldIdx : Nat) : IO EvalResult := do
  -- Compute per-position losses: [batch, seq]
  let losses2d := computeLosses logits targetIds

  -- For each option, compute mean loss over continuation tokens
  let mut meanLosses : Array Float := #[]

  for i in [:batchInfo.tokens.size] do
    let si := batchInfo.startIndices[i]!
    let ei := batchInfo.endIndices[i]!
    let contLen := ei - si

    if contLen == 0 then
      -- No continuation tokens, assign very high loss
      meanLosses := meanLosses.push 1e30
    else
      -- Extract losses for this sequence: losses2d[i, si-1:ei-1]
      -- (loss at position j predicts token j+1)
      let rowIdx := i.toUInt64
      let startLossIdx := if si > 0 then si - 1 else 0
      let endLossIdx := if ei > 0 then ei - 1 else 0

      -- Compute mean via tensor operations
      let lossesFlat := Tensor.reshapeChecked losses2d #[batch * seq]
      let rowOffset := rowIdx * seq
      let startIdx := rowOffset + startLossIdx.toUInt64
      let numTokens := (endLossIdx - startLossIdx).toUInt64

      if numTokens > 0 then
        let segmentLosses := Tensor.sliceChecked lossesFlat 0 startIdx numTokens #[numTokens]
        let meanLoss := Tensor.item (Tensor.meanAll segmentLosses)
        meanLosses := meanLosses.push meanLoss
      else
        meanLosses := meanLosses.push 1e30

  -- Predict option with lowest loss
  let mut minVal : Float := 1e30
  let mut minIdx := 0
  for i in [:meanLosses.size] do
    if meanLosses[i]! < minVal then
      minVal := meanLosses[i]!
      minIdx := i

  return { correct := minIdx == goldIdx
           losses := meanLosses
           prediction := minIdx }

/-- Evaluate language modeling example.
    Returns whether all continuation tokens were predicted correctly.

    For language modeling:
    1. Get argmax predictions at each position
    2. Compare predictions[start-1:end-1] with target tokens[start:end]
    3. Return correct=true if all predictions match -/
def evaluateLM {seq vocab : UInt64}
    (logits : Tensor { shape := #[1, seq, vocab], dtype := .Float32 })
    (targetIds : Tensor { shape := #[1, seq], dtype := .Int64 })
    (batchInfo : BatchInfo) : IO EvalResult := do
  let startIdx := batchInfo.startIndices[0]!
  let endIdx := batchInfo.endIndices[0]!

  if startIdx >= endIdx then
    return { correct := false, losses := #[], prediction := 0 }

  -- Get argmax predictions at each position: [1, seq]
  -- predictions[i] is the predicted token for position i+1
  let predictions := Tensor.argmax logits 2

  -- Flatten to 1D for easier slicing
  let predsFlat := Tensor.reshapeChecked predictions #[seq]
  let targetFlat := Tensor.reshapeChecked targetIds #[seq]

  -- Compare predictions[start-1:end-1] with targets[start:end]
  -- This checks if the model correctly predicts each continuation token
  let predStartIdx := if startIdx > 0 then (startIdx - 1).toUInt64 else 0
  let predEndIdx := if endIdx > 0 then (endIdx - 1).toUInt64 else 0
  let numTokens := predEndIdx - predStartIdx

  if numTokens == 0 then
    return { correct := false, losses := #[], prediction := 0 }

  let predSlice := Tensor.sliceChecked predsFlat 0 predStartIdx numTokens #[numTokens]
  let targetSlice := Tensor.sliceChecked targetFlat 0 startIdx.toUInt64 numTokens #[numTokens]
  let allCorrect := (← predSlice.toUInt64Array) == (← targetSlice.toUInt64Array)

  return { correct := allCorrect
           losses := #[]
           prediction := if allCorrect then 1 else 0 }

/-! ## High-Level API -/

/-- Evaluation configuration -/
structure EvalConfig where
  /-- Model's maximum sequence length (for truncation) -/
  maxSeqLen : Option UInt64 := none
  /-- BOS token ID for padding -/
  bosToken : UInt64 := 0
  /-- Device to run on -/
  device : Device := Device.CPU
  /-- When true, shard evaluation examples across distributed ranks. -/
  parallelAcrossRanks : Bool := true
  deriving Repr, Inhabited

/-- Stack and pad token sequences to create input tensor -/
def stackAndPad (tokens : Array (Array UInt64)) (padToken : UInt64) : Array UInt64 := Id.run do
  if tokens.isEmpty then return #[]
  let maxLen := tokens.foldl (fun acc seq => max acc seq.size) 0
  let mut result : Array UInt64 := #[]
  for seq in tokens do
    let padded := seq ++ Array.mk (List.replicate (maxLen - seq.size) padToken)
    result := result ++ padded
  return result

/-- Evaluate a single example and return whether it's correct -/
def evaluateSingleExample {vocab : UInt64}
    (item : EvalItem)
    (taskMeta : TaskMeta)
    (tokenize : String → Array UInt64)
    (runModel : ModelRunner vocab)
    (config : EvalConfig)
    : IO (Option Bool) := do
  let bosToken := config.bosToken
  -- Render prompts and get batch info based on task type
  let maybeBatchAndGold := match taskMeta.taskType, item with
  | .multipleChoice, .mc mcItem =>
    let prompts := renderPromptsMC mcItem taskMeta.continuationDelimiter #[]
    some (batchSequencesMC tokenize prompts bosToken, mcItem.gold)
  | .schema, .schema schemaItem =>
    let prompts := renderPromptsSchema schemaItem taskMeta.continuationDelimiter #[]
    some (batchSequencesSchema tokenize prompts bosToken, schemaItem.gold)
  | .languageModeling, .lm lmItem =>
    let (ctx, full) := renderPromptsLM lmItem taskMeta.continuationDelimiter #[]
    some (batchSequencesLM tokenize ctx full bosToken, 0)
  | _, _ => none

  match maybeBatchAndGold with
  | none => return none
  | some (batchInfo, goldIdx) =>
    if batchInfo.tokens.isEmpty then return none
    -- Prepare input tensor
    let paddedTokens := stackAndPad batchInfo.tokens bosToken
    let batchSize := batchInfo.tokens.size.toUInt64
    let seqLen := if batchInfo.tokens.isEmpty then (1 : UInt64)
                  else
                    (batchInfo.tokens.foldl (fun acc (seq : Array UInt64) => max acc seq.size) 0).toUInt64

    -- Convert to tensor
    let inputTensor := Tensor.fromInt64Array (paddedTokens.map (·.toInt64))
    let inputReshaped := Tensor.reshapeChecked inputTensor #[batchSize, seqLen]

    -- Run model forward
    let logits ← runModel (inputReshaped.to config.device)

    -- Create target IDs tensor (same as input for loss computation)
    let targetTensor := Tensor.fromInt64Array (paddedTokens.map (·.toInt64))
    let targetReshaped := (Tensor.reshapeChecked targetTensor #[batchSize, seqLen]).to logits.device

    let result ← match taskMeta.taskType with
    | .multipleChoice | .schema => evaluateMC logits targetReshaped batchInfo goldIdx
    | .languageModeling =>
      if h : batchSize = 1 then
        evaluateLM (seq := seqLen) (by simpa only [h] using logits) (by simpa only [h] using targetReshaped) batchInfo
      else
        throw <| IO.userError "Language-model evaluation requires exactly one sequence"

    return some result.correct

/-- Evaluate a single task across many examples.
    Returns accuracy (fraction correct).

    For each example:
    1. Render prompts based on task type
    2. Tokenize and batch
    3. Run model forward
    4. Use evaluateMC/evaluateLM/evaluateSchema to determine correctness -/
def evaluateTask {vocab : UInt64}
    (examples : Array EvalItem)
    (taskMeta : TaskMeta)
    (tokenize : String → Array UInt64)
    (runModel : ModelRunner vocab)
    (config : EvalConfig)
    : IO Float := do
  if examples.isEmpty then return 0.0

  -- Get distributed info
  let isDistributed ← dist.isInitialized
  let useDistributed := config.parallelAcrossRanks && isDistributed
  let (rank, worldSize) ← if useDistributed
    then dist.getRankAndWorldSize
    else pure (0, 1)

  let mut numCorrect : Nat := 0
  let mut numTotal : Nat := 0

  -- Evaluate examples strided by rank
  for idx in [:examples.size] do
    -- Skip if not assigned to this rank
    if idx % worldSize.toNat == rank.toNat then
      let item := examples[idx]!
      let maybeCorrect ← evaluateSingleExample item taskMeta tokenize runModel config
      match maybeCorrect with
      | none => pure ()  -- Skip invalid examples
      | some isCorrect =>
        if isCorrect then
          numCorrect := numCorrect + 1
        numTotal := numTotal + 1

  -- Aggregate across ranks if distributed
  if useDistributed && worldSize > 1 then
    dist.barrier
    -- Read reduced counter values while preserving the local inputs.
    let correctTensor := Tensor.fromInt64Array #[numCorrect.toInt64]
    let totalTensor := Tensor.fromInt64Array #[numTotal.toInt64]
    let correctTensor ← dist.allReduce correctTensor .sum
    let totalTensor ← dist.allReduce totalTensor .sum
    numCorrect := (Tensor.itemInt (correctTensor.reshapeChecked #[])).toInt.toNat
    numTotal := (Tensor.itemInt (totalTensor.reshapeChecked #[])).toInt.toNat

  if numTotal == 0 then return 0.0
  return numCorrect.toFloat / numTotal.toFloat

/-- Predefined CORE tasks from DCLM benchmark -/
def coreTasks : Array TaskMeta := #[
  -- ARC-Easy and ARC-Challenge
  { taskType := .multipleChoice, numFewshot := 0, taskName := "arc_easy" },
  { taskType := .multipleChoice, numFewshot := 0, taskName := "arc_challenge" },
  -- HellaSwag
  { taskType := .multipleChoice, numFewshot := 0, taskName := "hellaswag" },
  -- Winogrande
  { taskType := .schema, numFewshot := 0, taskName := "winogrande" },
  -- PIQA
  { taskType := .multipleChoice, numFewshot := 0, taskName := "piqa" },
  -- MMLU
  { taskType := .multipleChoice, numFewshot := 0, taskName := "mmlu" },
  -- BoolQ
  { taskType := .multipleChoice, numFewshot := 0, taskName := "boolq" },
  -- LAMBADA
  { taskType := .languageModeling, numFewshot := 0, taskName := "lambada" }
]

/-- Aggregate CORE score across tasks (average accuracy) -/
def computeCOREScore (taskAccuracies : Array (String × Float)) : Float :=
  if taskAccuracies.isEmpty then 0.0
  else
    let sum := taskAccuracies.foldl (fun acc (_, acc') => acc + acc') 0.0
    sum / taskAccuracies.size.toFloat

end torch.Eval.CORE
