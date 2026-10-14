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
  videoMaxFrames : UInt64 := 64
  videoFrameStride : UInt64 := 1
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
  | "--image" :: v :: rest => parseArgsLoop rest { acc with imagePath := some v }
  | "--video" :: v :: rest => parseArgsLoop rest { acc with videoPath := some v }
  | "--video-max-frames" :: v :: rest => parseArgsLoop rest { acc with videoMaxFrames := (← parseNatArg "--video-max-frames" v) }
  | "--video-frame-stride" :: v :: rest => parseArgsLoop rest { acc with videoFrameStride := (← parseNatArg "--video-frame-stride" v) }
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
  IO.println "  --image <path>               Image file for multimodal generation (Apple-only)"
  IO.println "  --video <path>               Video file for multimodal generation (Apple-only)"
  IO.println "  --video-max-frames <n>       Max decoded video frames (default: 64)"
  IO.println "  --video-frame-stride <n>     Keep every Nth decoded frame (default: 1)"
  IO.println "  --batch-size <n>             Prompts per decode batch (default: 1)"
  IO.println "  --max-new-tokens <n>         Number of tokens to generate"
  IO.println "  --stream                     Stream generated tokens per decode step"
  IO.println "  --multimodal                 Force multimodal model load (auto-enabled by --image/--video)"
  IO.println "  --enable-thinking            Use the Qwen3.5 thinking-enabled generation suffix"
  IO.println "  --debug-ids                  Print generated token ids alongside decoded text"
  IO.println "Examples:"
  IO.println "  lake exe Qwen35RunHF --source tiny-random/qwen3.5 --stream"
  IO.println "  lake exe Qwen35RunHF --source Qwen/Qwen3.5-0.8B --prompt \"Write one sentence about Lean.\""
  IO.println "  lake exe Qwen35RunHF --source Qwen/Qwen3.6-35B-A3B --multimodal --prompt \"Describe the image.\" --image cat.png"
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

private def visionPrefixIds (cfg : VLConfig) (imageTokenCount videoTokenCount : UInt64) : Array UInt64 :=
  Id.run do
    let mut out : Array UInt64 := #[]
    if imageTokenCount > 0 then
      out := out.push cfg.vision_start_token_id
      for _ in [:imageTokenCount.toNat] do out := out.push cfg.image_token_id
      out := out.push cfg.vision_end_token_id
    if videoTokenCount > 0 then
      out := out.push cfg.vision_start_token_id
      for _ in [:videoTokenCount.toNat] do out := out.push cfg.video_token_id
      out := out.push cfg.vision_end_token_id
    out

private def encodePromptToIdsMultimodal
    (tok : tokenizer.qwen35.QwenTokenizer)
    (enableThinking : Bool)
    (cfg : VLConfig)
    (imageTokenCount videoTokenCount : UInt64)
    (prompt : String)
    : Array UInt64 := Id.run do
  let userPrefixIds := (tokenizer.qwen35.encodeText tok tokenizer.qwen35.userPrefix).map (fun t => t.toUInt64)
  let promptIds := (tokenizer.qwen35.encodeText tok prompt).map (fun t => t.toUInt64)
  let suffixText :=
    if enableThinking then tokenizer.qwen35.assistantGenerationSuffixThinking
    else tokenizer.qwen35.assistantGenerationSuffix
  let suffix := (tokenizer.qwen35.encodeText tok suffixText).map (fun t => t.toUInt64)
  return userPrefixIds ++ (visionPrefixIds cfg imageTokenCount videoTokenCount) ++ promptIds ++ suffix

private def eosFromCfg (cfg : Config) : Array UInt64 :=
  match cfg.eos_token_id with
  | some id => #[id]
  | none => #[]

private def runTextBatches
    (tok : tokenizer.qwen35.QwenTokenizer)
    (cfg : Config)
    (model : Qwen35ForCausalLM cfg)
    (device : Device)
    (args : Args)
    (prompts : Array String)
    : IO Unit := do
  let eos := eosFromCfg cfg
  let buildBatch chunk := do
    let ⟨batch, ⟨seq, (inputIds, promptLens)⟩⟩ ←
      buildBatchInputWithEncoder tok.padToken.toUInt64 chunk (fun p => pure (encodePromptToIds tok args.enableThinking p))
    pure ⟨batch, ⟨seq, (inputIds.to device, promptLens)⟩⟩
  let decodeBatch (promptLens : Array Nat) {b os : UInt64} (ids : Tensor { shape := #[b, os], dtype := .Int64 }) : IO (Array String) :=
    decodeGeneratedBatch promptLens ids (fun xs => tokenizer.qwen35.decodeText tok xs)
  let runGen (b : UInt64) {s : UInt64} (ids : Tensor { shape := #[b, s], dtype := .Int64 }) :=
    model.generate cfg ids args.maxNewTokens .greedy eos
  let runGenStream (b : UInt64) {s : UInt64} (ids : Tensor { shape := #[b, s], dtype := .Int64 }) (cb : StreamCallback b) :=
    model.generateStream cfg ids cb args.maxNewTokens .greedy eos
  runGenerationBatches prompts args.batchSize args.maxNewTokens args.stream args.debugIds
    (prompts.size == 1) buildBatch runGen runGenStream decodeBatch
    (makeStreamCallback (fun x => tokenizer.qwen35.decodeOne tok x))

private def runMultimodalBatches
    (tok : tokenizer.qwen35.QwenTokenizer)
    (cfg : VLConfig)
    (model : Qwen35ForConditionalGeneration cfg)
    (device : Device)
    (args : Args)
    (prompts : Array String)
    (imageTokenCount : UInt64)
    (videoTokenCount : UInt64)
    (imageFeatures : Option (Sigma (fun n => Tensor { shape := #[n, cfg.vision_config.out_hidden_size] })))
    (videoFeatures : Option (Sigma (fun n => Tensor { shape := #[n, cfg.vision_config.out_hidden_size] })))
    : IO Unit := do
  let eos := eosFromCfg cfg.text_config
  let buildBatch chunk := do
    let enc := encodePromptToIdsMultimodal tok args.enableThinking cfg imageTokenCount videoTokenCount
    let ⟨batch, ⟨seq, (inputIds, promptLens)⟩⟩ ←
      buildBatchInputWithEncoder tok.padToken.toUInt64 chunk (fun p => pure (enc p))
    pure ⟨batch, ⟨seq, (inputIds.to device, promptLens)⟩⟩
  let decodeBatch (promptLens : Array Nat) {b os : UInt64} (ids : Tensor { shape := #[b, os], dtype := .Int64 }) : IO (Array String) :=
    decodeGeneratedBatch promptLens ids (fun xs => tokenizer.qwen35.decodeText tok xs)
  let runGen (b : UInt64) {s : UInt64} (ids : Tensor { shape := #[b, s], dtype := .Int64 }) :=
    model.generate cfg ids args.maxNewTokens .greedy eos imageFeatures videoFeatures
  let runGenStream (b : UInt64) {s : UInt64} (ids : Tensor { shape := #[b, s], dtype := .Int64 }) (cb : StreamCallback b) :=
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
    let isSharded ← Hub.detectWeightLayout modelDir
    let model ←
      if isSharded then Qwen35ForConditionalGeneration.loadSharded modelDir cfg device
      else Qwen35ForConditionalGeneration.load s!"{modelDir}/model.safetensors" cfg device
    let imagePatches? ←
      match args.imagePath with
      | some p =>
        IO.println s!"Loading image patches from {p}..."
        pure (some (moveSigmaTensorTo device (← media.loadImagePatches cfg p)))
      | none => pure none
    let videoPatches? ←
      match args.videoPath with
      | some p =>
        IO.println s!"Loading video patches from {p} (maxFrames={args.videoMaxFrames}, frameStride={args.videoFrameStride})..."
        pure (some (moveSigmaTensorTo device (← media.loadVideoPatches cfg p args.videoMaxFrames args.videoFrameStride)))
      | none => pure none

    let imageFeatures? ←
      match imagePatches? with
      | some ⟨nPatches, patches⟩ =>
        let feats ← model.getImageFeatures cfg patches.forgetDType
        let nTok := VisionConfig.mergedTokenCount cfg.vision_config nPatches
        pure (some ⟨nTok, feats⟩)
      | none => pure none
    let videoFeatures? ←
      match videoPatches? with
      | some ⟨nPatches, patches⟩ =>
        let feats ← model.getVideoFeatures cfg patches.forgetDType
        let nTok := VisionConfig.mergedTokenCount cfg.vision_config nPatches
        pure (some ⟨nTok, feats⟩)
      | none => pure none

    let imageTokenCount := match imageFeatures? with | some ⟨n, _⟩ => n | none => 0
    let videoTokenCount := match videoFeatures? with | some ⟨n, _⟩ => n | none => 0

    if imageTokenCount == 0 && videoTokenCount == 0 then
      IO.println "Warning: --multimodal enabled but no --image/--video provided; running text-only path."

    runMultimodalBatches
      tok cfg model device args prompts
      imageTokenCount videoTokenCount
      imageFeatures? videoFeatures?
  else
    let cfg ← Config.loadFromPretrainedDir modelDir Config.qwen35_9B
    let isSharded ← Hub.detectWeightLayout modelDir
    let model ←
      if isSharded then Qwen35ForCausalLM.loadSharded modelDir cfg device
      else Qwen35ForCausalLM.load s!"{modelDir}/model.safetensors" cfg device
    runTextBatches tok cfg model device args prompts

  pure 0

end Examples.Qwen35

def main (argv : List String) : IO UInt32 :=
  Examples.Qwen35.runMain argv
