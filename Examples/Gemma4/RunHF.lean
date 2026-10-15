/-
  Examples/Gemma4/RunHF.lean

  Text and image-conditioned generation demo for Gemma 4 models resolved by
  local path or HuggingFace repo id.
-/
import Tyr.Hub
import Tyr.Model.Gemma4.Config
import Tyr.Model.Gemma4.ConfigIO
import Tyr.Model.Gemma4.Model
import Tyr.Model.Gemma4.Weights
import Tyr.Model.Gemma4.VLConfig
import Tyr.Model.Gemma4.VLConfigIO
import Tyr.Model.Gemma4.Media
import Tyr.Model.Gemma4.Multimodal
import Tyr.Model.Gemma4.VLWeights
import Tyr.Tokenizer.Gemma4
import Examples.ModelRunner

open torch

variable {device : Device} {dtype visionDType projectorDType : DType}
open torch.Model
open torch.gemma4
open Examples.ModelRunner

namespace Examples.Gemma4

structure Args where
  source : String := "google/gemma-4-E4B"
  revision : String := "main"
  cacheDir : String := Hub.defaultCacheDir
  device : String := "auto"
  prompt : String := "Give a concise definition of a dependent type."
  promptFile : Option String := none
  imagePaths : Array String := #[]
  batchSize : UInt64 := 1
  maxNewTokens : UInt64 := 32
  stream : Bool := false
  multimodal : Bool := false
  enableThinking : Bool := false
  debugIds : Bool := false
  showHelp : Bool := false
  deriving Inhabited

private partial def parseArgsLoop (xs : List String) (acc : Args) : IO Args := do
  match xs with
  | [] => pure acc
  | "--source" :: v :: rest => parseArgsLoop rest { acc with source := v }
  | "--revision" :: v :: rest => parseArgsLoop rest { acc with revision := v }
  | "--cache-dir" :: v :: rest => parseArgsLoop rest { acc with cacheDir := v }
  | "--device" :: v :: rest => parseArgsLoop rest { acc with device := v }
  | "--prompt" :: v :: rest => parseArgsLoop rest { acc with prompt := v }
  | "--prompt-file" :: v :: rest => parseArgsLoop rest { acc with promptFile := some v }
  | "--image" :: v :: rest => parseArgsLoop rest { acc with imagePaths := acc.imagePaths.push v }
  | "--batch-size" :: v :: rest => parseArgsLoop rest { acc with batchSize := (← parseNatArg "--batch-size" v) }
  | "--max-new-tokens" :: v :: rest => parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--stream" :: rest => parseArgsLoop rest { acc with stream := true }
  | "--multimodal" :: rest => parseArgsLoop rest { acc with multimodal := true }
  | "--enable-thinking" :: rest => parseArgsLoop rest { acc with enableThinking := true }
  | "--debug-ids" :: rest => parseArgsLoop rest { acc with debugIds := true }
  | "--help" :: _ => parseArgsLoop [] { acc with showHelp := true }
  | x :: _ => throw <| IO.userError s!"Unknown argument: {x}"

private def parseArgs (xs : List String) : IO Args :=
  parseArgsLoop xs {}

private def printHelp : IO Unit := do
  IO.println "Usage: lake exe Gemma4RunHF [options]"
  IO.println "  --source <path-or-repo>      Local model dir or HF repo id (default: google/gemma-4-E4B)"
  IO.println "  --revision <rev>             HF revision/branch/tag (default: main)"
  IO.println "  --cache-dir <path>           Local cache for downloaded files"
  IO.println "  --device <auto|cpu|mps|cuda[:n]>  Execution device (default: auto)"
  IO.println "  --prompt <text>              Prompt text"
  IO.println "  --prompt-file <path>         One prompt per non-empty line"
  IO.println "  --image <path>               Image file for multimodal generation (repeatable, Apple-only)"
  IO.println "  --batch-size <n>             Prompts per decode batch (default: 1)"
  IO.println "  --max-new-tokens <n>         Number of tokens to generate"
  IO.println "  --stream                     Stream generated tokens per decode step"
  IO.println "  --multimodal                 Force multimodal model load (auto-enabled by --image)"
  IO.println "  --enable-thinking            Use the Gemma 4 thinking-enabled chat template"
  IO.println "  --debug-ids                  Print generated token ids alongside decoded text"

private def encodePromptToIds
    (tok : tokenizer.gemma4.GemmaTokenizer)
    (enableThinking : Bool)
    (prompt : String)
    : Array UInt64 :=
  let text :=
    if enableThinking then tokenizer.gemma4.chatTemplateThinking prompt
    else tokenizer.gemma4.chatTemplate prompt
  (tokenizer.gemma4.encodeText tok text).map (fun t => t.toUInt64)

private def buildImageExpansion (softTokenCount : UInt64) : String :=
  Id.run do
    let mut out := "<|image>"
    for _ in [:softTokenCount.toNat] do
      out := out ++ "<|image|>"
    out := out ++ "<image|>"
    pure out

private def prefixImageExpansions (softTokenCounts : Array UInt64) : String :=
  Id.run do
    let mut out := ""
    for i in [:softTokenCounts.size] do
      if i > 0 then out := out ++ " "
      out := out ++ buildImageExpansion softTokenCounts[i]!
    pure out

private def injectImageExpansions (prompt : String) (softTokenCounts : Array UInt64) : IO String := do
  let placeholder := "<|image|>"
  let parts := (prompt.splitOn placeholder).toArray
  if parts.size <= 1 then
    let prefixText := prefixImageExpansions softTokenCounts
    if prefixText.isEmpty then pure prompt
    else if prompt.trimAscii.isEmpty then pure prefixText
    else pure s!"{prefixText}\n{prompt}"
  else
    let occurrences := parts.size - 1
    if occurrences != softTokenCounts.size then
      throw <| IO.userError
        s!"Prompt contains {occurrences} image placeholders but {softTokenCounts.size} image inputs were provided"
    let mut out := parts[0]!
    for i in [:occurrences] do
      out := out ++ buildImageExpansion softTokenCounts[i]! ++ parts[i + 1]!
    pure out

private def encodePromptToIdsMultimodal
    (tok : tokenizer.gemma4.GemmaTokenizer)
    (enableThinking : Bool)
    (softTokenCounts : Array UInt64)
    (prompt : String)
    : IO (Array UInt64) := do
  let prompt' ← injectImageExpansions prompt softTokenCounts
  pure (encodePromptToIds tok enableThinking prompt')

private def pushUnique (xs : Array UInt64) (x : UInt64) : Array UInt64 :=
  if xs.contains x then xs else xs.push x

private def eosStopTokenIds (tok : tokenizer.gemma4.GemmaTokenizer) (cfg : Config) : Array UInt64 :=
  Id.run do
    let mut out : Array UInt64 := #[]
    match cfg.eos_token_id with
    | some id => out := pushUnique out id
    | none =>
      match tok.eosToken with
      | some id => out := pushUnique out id.toUInt64
      | none => pure ()
    out

private def movePatchGridTo {cfg : VLConfig} {sourceDevice : Device} (device : Device) (x : ImagePatchGrid (device := sourceDevice) cfg) : ImagePatchGrid (device := device) cfg :=
  match x with
  | ⟨patchRows, ⟨patchCols, grid⟩⟩ => ⟨patchRows, ⟨patchCols, grid.to device⟩⟩

private def runTextBatches
    (tok : tokenizer.gemma4.GemmaTokenizer)
    (cfg : Config)
    (model : Gemma4ForCausalLM (device := device) cfg dtype)
    (args : Args)
    (prompts : Array String)
    : IO Unit := do
  let generationStopIds := eosStopTokenIds tok cfg
  let buildBatch chunk := do
    let ⟨batch, ⟨seq, (inputIds, promptLens)⟩⟩ ←
      buildBatchInputWithEncoder tok.padToken.toUInt64 chunk (fun p => pure (encodePromptToIds tok args.enableThinking p))
    pure ⟨batch, ⟨seq, (inputIds.to device, promptLens)⟩⟩
  let decodeBatch (promptLens : Array Nat) {b os : UInt64} (ids : Tensor { shape := #[b, os], dtype := .Int64, device := device }) : IO (Array String) :=
    decodeGeneratedBatch promptLens ids (fun xs => tokenizer.gemma4.decodeText tok xs)
  let runGen (b : UInt64) {s : UInt64} (ids : Tensor { shape := #[b, s], dtype := .Int64, device := device }) :=
    model.generate cfg ids args.maxNewTokens .greedy generationStopIds
  let runGenStream (b : UInt64) {s : UInt64} (ids : Tensor { shape := #[b, s], dtype := .Int64, device := device }) (cb : StreamCallback (device := device) b) :=
    model.generateStream cfg ids cb args.maxNewTokens .greedy generationStopIds
  runGenerationBatches prompts args.batchSize args.maxNewTokens args.stream args.debugIds
    (prompts.size == 1) buildBatch runGen runGenStream decodeBatch
    (makeStreamCallback (fun x => tokenizer.gemma4.decodeOne tok x))

private def runMultimodalBatches
    (tok : tokenizer.gemma4.GemmaTokenizer)
    (cfg : VLConfig)
    (model : Gemma4ForConditionalGeneration (device := device) cfg dtype visionDType projectorDType)
    (args : Args)
    (prompts : Array String)
    (imageSoftTokenCounts : Array UInt64)
    (imageFeatures : Option (ImageFeatures (device := device) cfg projectorDType))
    : IO Unit := do
  let generationStopIds := eosStopTokenIds tok cfg.text_config
  let buildBatch chunk := do
    let ⟨batch, ⟨seq, (inputIds, promptLens)⟩⟩ ←
      buildBatchInputWithEncoder tok.padToken.toUInt64 chunk
        (encodePromptToIdsMultimodal tok args.enableThinking imageSoftTokenCounts)
    pure ⟨batch, ⟨seq, (inputIds.to device, promptLens)⟩⟩
  let decodeBatch (promptLens : Array Nat) {b os : UInt64} (ids : Tensor { shape := #[b, os], dtype := .Int64, device := device }) : IO (Array String) :=
    decodeGeneratedBatch promptLens ids (fun xs => tokenizer.gemma4.decodeText tok xs)
  let runGen (b : UInt64) {s : UInt64} (ids : Tensor { shape := #[b, s], dtype := .Int64, device := device }) :=
    let imageFeaturesChunk :=
      match imageFeatures with
      | some ⟨_nTok, feats⟩ => some <| Gemma4ForConditionalGeneration.repeatFeaturesForBatch (batch := b) cfg feats
      | none => none
    model.generate cfg ids args.maxNewTokens .greedy generationStopIds imageFeaturesChunk
  let runGenStream (b : UInt64) {s : UInt64} (ids : Tensor { shape := #[b, s], dtype := .Int64, device := device }) (cb : StreamCallback (device := device) b) :=
    let imageFeaturesChunk :=
      match imageFeatures with
      | some ⟨_nTok, feats⟩ => some <| Gemma4ForConditionalGeneration.repeatFeaturesForBatch (batch := b) cfg feats
      | none => none
    model.generateStream cfg ids cb args.maxNewTokens .greedy generationStopIds imageFeaturesChunk
  runGenerationBatches prompts args.batchSize args.maxNewTokens args.stream args.debugIds
    (prompts.size == 1) buildBatch runGen runGenStream decodeBatch
    (makeStreamCallback (fun x => tokenizer.gemma4.decodeOne tok x))

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  if args.showHelp then
    printHelp
    return 0

  let (device, deviceWarning?) ← resolveDevice args.device
  let hasImages := !args.imagePaths.isEmpty
  let useMultimodal := args.multimodal || hasImages
  if hasImages && !args.multimodal then
    IO.println "Info: enabling multimodal mode because --image was provided."

  let modelDir ← Hub.resolvePretrainedDir args.source {
    revision := args.revision
    cacheDir := args.cacheDir
    includeTokenizer := true
  }
  IO.println s!"Model directory: {modelDir}"
  match deviceWarning? with
  | some msg => IO.println msg
  | none => pure ()
  IO.println s!"Using device: {deviceToString device}"

  let tok ← tokenizer.gemma4.loadTokenizer modelDir
  let prompts ← loadPrompts args.promptFile args.prompt

  if useMultimodal && hasImages then
    let cfg ← VLConfig.loadFromPretrainedDir modelDir {}
    let isSharded ← Hub.detectWeightLayout modelDir
    let ⟨_textDType, _visionDType, _projectorDType, model⟩ ←
      if isSharded then Gemma4ForConditionalGeneration.loadSharded modelDir cfg (device := device)
      else Gemma4ForConditionalGeneration.load s!"{modelDir}/model.safetensors" cfg (device := device)

    let mut imageGrids : Array (ImagePatchGrid (device := device) cfg) := #[]
    let mut imageSoftTokenCounts : Array UInt64 := #[]
    for p in args.imagePaths do
      IO.println s!"Loading image patches from {p}..."
      let grid := movePatchGridTo device (← media.loadImagePatchGrid cfg p)
      match grid with
      | ⟨patchRows, ⟨patchCols, _⟩⟩ =>
        imageSoftTokenCounts := imageSoftTokenCounts.push
          ((patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size))
      imageGrids := imageGrids.push grid

    let imageFeatures ← model.getImageFeaturesMany cfg imageGrids
    runMultimodalBatches tok cfg model args prompts imageSoftTokenCounts imageFeatures
  else
    if useMultimodal && !hasImages then
      IO.println "Warning: --multimodal requested without --image; running text-only path."
    let cfg ← Config.loadFromPretrainedDir modelDir Config.gemma4_E4B
    let isSharded ← Hub.detectWeightLayout modelDir
    let ⟨_dtype, model⟩ ←
      if isSharded then Gemma4ForCausalLM.loadSharded modelDir cfg (device := device)
      else Gemma4ForCausalLM.load s!"{modelDir}/model.safetensors" cfg (device := device)
    runTextBatches tok cfg model args prompts

  pure 0

end Examples.Gemma4

def main (argv : List String) : IO UInt32 :=
  Examples.Gemma4.runMain argv
