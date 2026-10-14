/-
  Examples/Qwen25Omni/RunHF.lean

  Text generation demo for Qwen2.5-Omni thinker checkpoints (3B/7B),
  resolved by local path or HuggingFace repo id.
-/
import Tyr.Hub
import Tyr.Model.Qwen25Omni.Config
import Tyr.Model.Qwen25Omni.ConfigIO
import Tyr.Model.Qwen25Omni.Weights
import Tyr.TensorStruct
import Tyr.Tokenizer.Qwen3
import Examples.ModelRunner

open torch
open torch.qwen25omni
open Examples.ModelRunner

namespace Examples.Qwen25Omni

structure Args where
  source : String := "Qwen/Qwen2.5-Omni-3B"
  revision : String := "main"
  cacheDir : String := Hub.defaultCacheDir
  device : String := "auto"
  prompt : String := "Give a concise definition of a dependent type."
  promptFile : Option String := none
  batchSize : UInt64 := 1
  maxNewTokens : UInt64 := 64
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
  | "--batch-size" :: v :: rest => parseArgsLoop rest { acc with batchSize := (← parseNatArg "--batch-size" v) }
  | "--max-new-tokens" :: v :: rest => parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--help" :: _ => parseArgsLoop [] { acc with showHelp := true }
  | x :: _ => throw <| IO.userError s!"Unknown argument: {x}"

private def parseArgs (xs : List String) : IO Args :=
  parseArgsLoop xs {}

private def printHelp : IO Unit := do
  IO.println "Usage: lake exe Qwen25OmniRunHF [options]"
  IO.println "  --source <path-or-repo>      Local model dir or HF repo id (default: Qwen/Qwen2.5-Omni-3B)"
  IO.println "  --revision <rev>             HF revision/branch/tag (default: main)"
  IO.println "  --cache-dir <path>           Local cache for downloaded files"
  IO.println "  --device <auto|cpu|mps|cuda[:n]>  Execution device (default: auto)"
  IO.println "  --prompt <text>              Prompt text"
  IO.println "  --prompt-file <path>         One prompt per non-empty line"
  IO.println "  --batch-size <n>             Prompts per decode batch (default: 1)"
  IO.println "  --max-new-tokens <n>         Number of tokens to generate"
  IO.println "Examples:"
  IO.println "  lake exe Qwen25OmniRunHF --source Qwen/Qwen2.5-Omni-3B --prompt \"Hello\""
  IO.println "  lake exe Qwen25OmniRunHF --source Qwen/Qwen2.5-Omni-7B --prompt-file prompts.txt --batch-size 2"

private def encodePromptToIds
    (tok : tokenizer.qwen3.QwenTokenizer)
    (prompt : String)
    : Array UInt64 :=
  let text := tokenizer.qwen3.chatTemplate prompt
  (tokenizer.qwen3.encodeText tok text).map (fun t => t.toUInt64)

private def defaultConfigForSource (source : String) : Config :=
  if source.contains "-7B" then Config.qwen25omni_7B
  else Config.qwen25omni_3B

private def runBatches
    (tok : tokenizer.qwen3.QwenTokenizer)
    (model : Qwen25OmniForCausalLM cfg)
    (device : Device)
    (args : Args)
    (prompts : Array String)
    : IO Unit := do
  let eos : Array UInt64 := #[]
  let buildBatch chunk := do
    let ⟨batch, ⟨seq, (inputIds, promptLens)⟩⟩ ←
      buildBatchInput tok.padToken.toUInt64 chunk (encodePromptToIds tok)
    pure ⟨batch, ⟨seq, (inputIds.to device, promptLens)⟩⟩
  let decodeBatch (promptLens : Array Nat) {b os : UInt64} (ids : Tensor { shape := #[b, os], dtype := .Int64 }) : IO (Array String) :=
    decodeGeneratedBatch promptLens ids (fun xs => tokenizer.qwen3.decodeText tok xs)
  let runGen (b : UInt64) {s : UInt64} (ids : Tensor { shape := #[b, s], dtype := .Int64 }) :=
    model.generateGreedy ids args.maxNewTokens eos
  runGreedyBatches prompts args.batchSize args.maxNewTokens (prompts.size == 1)
    buildBatch runGen decodeBatch

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  if args.showHelp then
    printHelp
    return 0

  let (device, deviceWarning?) ← resolveDevice args.device
  let modelDir ← Hub.resolvePretrainedDir args.source {
    revision := args.revision
    cacheDir := args.cacheDir
    includeTokenizer := true
  }
  IO.println s!"Resolved model dir: {modelDir}"
  match deviceWarning? with
  | some msg => IO.println msg
  | none => pure ()
  IO.println s!"Using device: {deviceToString device}"

  let tok ← tokenizer.qwen3.loadTokenizer modelDir
  let prompts ← loadPrompts args.promptFile args.prompt

  let cfgDefaults := defaultConfigForSource args.source
  let cfg ← Config.loadFromPretrainedDir modelDir cfgDefaults
  let isSharded ← Hub.detectWeightLayout modelDir
  let modelCpu ←
    if isSharded then Qwen25OmniForCausalLM.loadSharded modelDir cfg
    else Qwen25OmniForCausalLM.load s!"{modelDir}/model.safetensors" cfg
  let model ← moveModelToDevice device modelCpu

  runBatches tok model device args prompts
  pure 0

end Examples.Qwen25Omni

def main (argv : List String) : IO UInt32 :=
  Examples.Qwen25Omni.runMain argv
