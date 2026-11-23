/-
  Examples/Qwen35/RunHF.lean

  Text generation demo for Qwen3.5/Qwen3.6 models resolved by local path or
  HuggingFace repo id.
-/
import Tyr.Hub
import Tyr.Model.Qwen35.Config
import Tyr.Model.Qwen35.ConfigIO
import Tyr.Model.Qwen35.Model
import Tyr.Model.Qwen35.Weights
import Tyr.Model.Qwen35.VLConfig
import Tyr.Model.Qwen35.VLConfigIO
import Tyr.Model.Qwen35.Multimodal
import Tyr.Model.Qwen35.Media
import Tyr.Model.Qwen35.VLWeights
import Tyr.TensorStruct
import Tyr.Tokenizer.Qwen35
import Examples.ModelRunner

open torch

variable {device : Device} {dtype visionDType : DType}
open torch.Model
open torch.qwen35
open Examples.ModelRunner

namespace Examples.Qwen35

structure Args where
  source : String := "tiny-random/qwen3.5"
  revision : String := "main"
  cacheDir : String := Hub.defaultCacheDir
  device : String := "auto"
  prompt : String := "Give a concise definition of a dependent type."
  promptFile : Option String := none
  imagePath : Option String := none
  videoPath : Option String := none
  batchSize : Nat := 1
  maxNewTokens : Nat := 32
  stream : Bool := false
  multimodal : Bool := false
  enableThinking : Bool := false
  debugIds : Bool := false
  showHelp : Bool := false

private partial def parseArgsLoop (xs : List String) (acc : Args) : IO Args := do
  match xs with
  | [] => pure acc
  | "--source" :: v :: rest => parseArgsLoop rest { acc with source := v }
  | "--revision" :: v :: rest => parseArgsLoop rest { acc with revision := v }
  | "--cache-dir" :: v :: rest => parseArgsLoop rest { acc with cacheDir := v }
  | "--device" :: v :: rest => parseArgsLoop rest { acc with device := v }
  | "--prompt" :: v :: rest => parseArgsLoop rest { acc with prompt := v }
  | "--prompt-file" :: v :: rest => parseArgsLoop rest { acc with promptFile := some v }
  | "--image" :: v :: rest => parseArgsLoop rest { acc with imagePath := some v }
  | "--video" :: v :: rest => parseArgsLoop rest { acc with videoPath := some v }
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
  IO.println "Usage: lake exe Qwen35RunHF [options]"
  IO.println "  --source <path-or-repo>      Local model dir or HF repo id (default: tiny-random/qwen3.5)"
  IO.println "  --revision <rev>             HF revision/branch/tag (default: main)"
  IO.println "  --cache-dir <path>           Local cache for downloaded files"
  IO.println "  --device <auto|cpu|mps|cuda[:n]>  Execution device (default: auto)"
  IO.println "  --prompt <text>              Prompt text"
  IO.println "  --prompt-file <path>         One prompt per non-empty line"
  IO.println "  --image <path>               Rejected: the Apple media loader gives no image grid"
  IO.println "  --video <path>               Rejected: the Apple media loader gives no video grid"
  IO.println "  --batch-size <n>             Prompts per decode batch (default: 1)"
  IO.println "  --max-new-tokens <n>         Number of tokens to generate"
  IO.println "  --stream                     Stream generated tokens per decode step"
  IO.println "  --multimodal                 Force multimodal model load (auto-enabled by --image/--video)"
  IO.println "  --enable-thinking            Use the Qwen3.5 thinking-enabled generation suffix"
  IO.println "  --debug-ids                  Print generated token ids alongside decoded text"
  IO.println "Examples:"
  IO.println "  lake exe Qwen35RunHF --source tiny-random/qwen3.5 --stream"
  IO.println "  lake exe Qwen35RunHF --source Qwen/Qwen3.5-0.8B --prompt \"Write one sentence about Lean.\""
  IO.println "  lake exe Qwen35RunHF --source tiny-random/qwen3.5 --prompt-file prompts.txt --batch-size 4"

private def encodePromptToIds
    (tok : tokenizer.qwen35.QwenTokenizer)
    (enableThinking : Bool)
    (prompt : String)
    : Array UInt64 :=
  let text :=
    if enableThinking then tokenizer.qwen35.chatTemplateThinking prompt
    else tokenizer.qwen35.chatTemplate prompt
  (tokenizer.qwen35.encodeText tok text).map (fun t => t.toUInt64)

private def visionPrefixIds (cfg : VLConfig) (imageTokenCount videoTokenCount : Nat) : Array UInt64 :=
  Id.run do
    let mut out : Array UInt64 := #[]
    if imageTokenCount > 0 then
      out := out.push cfg.vision_start_token_id.toUInt64
      for _ in [:imageTokenCount] do out := out.push cfg.image_token_id.toUInt64
      out := out.push cfg.vision_end_token_id.toUInt64
    if videoTokenCount > 0 then
      out := out.push cfg.vision_start_token_id.toUInt64
      for _ in [:videoTokenCount] do out := out.push cfg.video_token_id.toUInt64
      out := out.push cfg.vision_end_token_id.toUInt64
    out

private def encodePromptToIdsMultimodal
    (tok : tokenizer.qwen35.QwenTokenizer)
    (enableThinking : Bool)
    (cfg : VLConfig)
    (imageTokenCount videoTokenCount : Nat)
    (prompt : String)
    : Array UInt64 := Id.run do
  let userPrefixIds := (tokenizer.qwen35.encodeText tok tokenizer.qwen35.userPrefix).map (fun t => t.toUInt64)
  let promptIds := (tokenizer.qwen35.encodeText tok prompt).map (fun t => t.toUInt64)
  let suffixText :=
    if enableThinking then tokenizer.qwen35.assistantGenerationSuffixThinking
    else tokenizer.qwen35.assistantGenerationSuffix
  let suffix := (tokenizer.qwen35.encodeText tok suffixText).map (fun t => t.toUInt64)
  return userPrefixIds ++ (visionPrefixIds cfg imageTokenCount videoTokenCount) ++ promptIds ++ suffix

private def eosFromCfg (cfg : Config) : Array tokenizer.TokenId :=
  match cfg.eos_token_id with
  | some id => #[id]
  | none => #[]

private def runTextBatches [DType.RealFloating dtype]
    (tok : tokenizer.qwen35.QwenTokenizer)
    (cfg : Config)
    (model : Qwen35ForCausalLM (device := device) cfg dtype)
    (args : Args)
    (prompts : Array String)
    : IO Unit := do
  let eos := eosFromCfg cfg
  let buildBatch chunk := do
    let ⟨batch, ⟨seq, (inputIds, promptLens)⟩⟩ ←
      buildBatchInputWithEncoder tok.padToken.toUInt64 chunk (fun p => pure (encodePromptToIds tok args.enableThinking p))
    pure ⟨batch, ⟨seq, (inputIds.to device, promptLens)⟩⟩
  let decodeBatch (promptLens : Array Nat) {b os : Nat} (ids : Tensor { shape := #[b, os], dtype := .Int64, device := device }) : IO (Array String) :=
    decodeGeneratedBatch promptLens ids (fun xs => tokenizer.qwen35.decodeText tok xs)
  let runGen (b : Nat) {s : Nat} (ids : Tensor { shape := #[b, s], dtype := .Int64, device := device }) :=
    model.generate cfg ids args.maxNewTokens .greedy eos
  let runGenStream (b : Nat) {s : Nat} (ids : Tensor { shape := #[b, s], dtype := .Int64, device := device }) (cb : StreamCallback (device := device) b) :=
    model.generateStream cfg ids cb args.maxNewTokens .greedy eos
  runGenerationBatches prompts args.batchSize args.maxNewTokens args.stream args.debugIds
    (prompts.size == 1) buildBatch runGen runGenStream decodeBatch
    (makeStreamCallback (fun x => tokenizer.qwen35.decodeOne tok x))

private def runMultimodalBatches [DType.RealFloating dtype]
    (tok : tokenizer.qwen35.QwenTokenizer)
    (cfg : VLConfig)
    (model : Qwen35ForConditionalGeneration (device := device) cfg dtype visionDType)
    (args : Args)
    (prompts : Array String)
    (imageTokenCount : Nat)
    (videoTokenCount : Nat)
    (imageFeatures videoFeatures : Option (Qwen35ForConditionalGeneration.VisualFeatures (device := device) cfg visionDType))
    : IO Unit := do
  let eos := eosFromCfg cfg.text_config
  let buildBatch chunk := do
    let enc := encodePromptToIdsMultimodal tok args.enableThinking cfg imageTokenCount videoTokenCount
    let ⟨batch, ⟨seq, (inputIds, promptLens)⟩⟩ ←
      buildBatchInputWithEncoder tok.padToken.toUInt64 chunk (fun p => pure (enc p))
    pure ⟨batch, ⟨seq, (inputIds.to device, promptLens)⟩⟩
  let decodeBatch (promptLens : Array Nat) {b os : Nat} (ids : Tensor { shape := #[b, os], dtype := .Int64, device := device }) : IO (Array String) :=
    decodeGeneratedBatch promptLens ids (fun xs => tokenizer.qwen35.decodeText tok xs)
  let runGen (b : Nat) {s : Nat} (ids : Tensor { shape := #[b, s], dtype := .Int64, device := device }) :=
    model.generate cfg ids args.maxNewTokens .greedy eos imageFeatures videoFeatures
  let runGenStream (b : Nat) {s : Nat} (ids : Tensor { shape := #[b, s], dtype := .Int64, device := device }) (cb : StreamCallback (device := device) b) :=
    model.generateStream cfg ids cb args.maxNewTokens .greedy eos imageFeatures videoFeatures
  runGenerationBatches prompts args.batchSize args.maxNewTokens args.stream args.debugIds
    (prompts.size == 1) buildBatch runGen runGenStream decodeBatch
    (makeStreamCallback (fun x => tokenizer.qwen35.decodeOne tok x))

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  if args.showHelp then
    printHelp
    return 0

  let (device, deviceWarning?) ← resolveDevice args.device
  let hasMedia := args.imagePath.isSome || args.videoPath.isSome
  let useMultimodal := args.multimodal || hasMedia
  if hasMedia && !args.multimodal then
    IO.println "Info: enabling multimodal mode because --image/--video was provided."
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

  let tok ← tokenizer.qwen35.loadTokenizer modelDir
  let prompts ← loadPrompts args.promptFile args.prompt

  if useMultimodal then
    let cfg ← VLConfig.loadFromPretrainedDir modelDir {}
    let ⟨_textDType, visionDType, model⟩ ←
      Qwen35ForConditionalGeneration.load modelDir cfg device
    if hasMedia then
      throw <| IO.userError
        "--image/--video: the Apple media loader returns patches without their grids, which Qwen3.5's multimodal positions need"
    IO.println "Warning: --multimodal enabled but no --image/--video provided; running text-only path."

    runMultimodalBatches tok cfg model args prompts 0 0 none none
  else
    let cfg ← Config.loadFromPretrainedDir modelDir Config.qwen35_9B
    let ⟨_dtype, model⟩ ← Qwen35ForCausalLM.load modelDir cfg device
    runTextBatches tok cfg model args prompts

  pure 0

end Examples.Qwen35

def main (argv : List String) : IO UInt32 :=
  Examples.Qwen35.runMain argv
