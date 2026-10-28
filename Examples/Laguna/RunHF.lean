/-
  Examples/Laguna/RunHF.lean

  Text generation demo for poolside Laguna models (Laguna-S-2.1) resolved by
  local path or HuggingFace repo id. Includes a --bench mode measuring
  prefill/decode throughput (tokens/sec).
-/
import Tyr.Hub
import Tyr.Model.Laguna.Config
import Tyr.Model.Laguna.ConfigIO
import Tyr.Model.Laguna.Model
import Tyr.Model.Laguna.Weights
import Tyr.TensorStruct
import Tyr.Tokenizer.Laguna
import Examples.ModelRunner

open torch

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
open torch.Model
open torch.laguna
open Examples.ModelRunner

namespace Examples.Laguna

structure Args where
  source : String := "poolside/Laguna-S-2.1-NVFP4"
  revision : String := "main"
  cacheDir : String := Hub.defaultCacheDir
  device : String := "auto"
  prompt : String := "Give a concise definition of a dependent type."
  promptFile : Option String := none
  batchSize : Nat := 1
  maxNewTokens : Nat := 32
  stream : Bool := false
  enableThinking : Bool := true
  debugIds : Bool := false
  bench : Bool := false
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
  | "--stream" :: rest => parseArgsLoop rest { acc with stream := true }
  | "--no-thinking" :: rest => parseArgsLoop rest { acc with enableThinking := false }
  | "--debug-ids" :: rest => parseArgsLoop rest { acc with debugIds := true }
  | "--bench" :: rest => parseArgsLoop rest { acc with bench := true }
  | "--help" :: _ => parseArgsLoop [] { acc with showHelp := true }
  | x :: _ => throw <| IO.userError s!"Unknown argument: {x}"

private def parseArgs (xs : List String) : IO Args :=
  parseArgsLoop xs {}

private def printHelp : IO Unit := do
  IO.println "Usage: lake exe LagunaRunHF [options]"
  IO.println "  --source <path-or-repo>      Local model dir or HF repo id (default: poolside/Laguna-S-2.1-NVFP4)"
  IO.println "  --revision <rev>             HF revision/branch/tag (default: main)"
  IO.println "  --cache-dir <path>           Local cache for downloaded files"
  IO.println "  --device <auto|cpu|mps|cuda[:n]>  Execution device (default: auto)"
  IO.println "  --prompt <text>              Prompt text"
  IO.println "  --prompt-file <path>         One prompt per non-empty line"
  IO.println "  --batch-size <n>             Prompts per decode batch (default: 1)"
  IO.println "  --max-new-tokens <n>         Number of tokens to generate"
  IO.println "  --stream                     Stream generated tokens per decode step"
  IO.println "  --no-thinking                Disable the thinking-enabled generation suffix"
  IO.println "  --debug-ids                  Print generated token ids alongside decoded text"
  IO.println "  --bench                      Measure prefill/decode throughput (tok/s)"
  IO.println "Examples:"
  IO.println "  lake exe LagunaRunHF --source poolside/Laguna-S-2.1-NVFP4 --device cuda --stream"
  IO.println "  lake exe LagunaRunHF --source poolside/Laguna-S-2.1-NVFP4 --device cuda --bench --max-new-tokens 128"

private def encodePromptToIds
    (tok : tokenizer.laguna.LagunaTokenizer)
    (enableThinking : Bool)
    (prompt : String)
    : Array UInt64 :=
  let text := tokenizer.laguna.chatTemplate prompt enableThinking
  (tokenizer.laguna.encodeText tok text).map (fun t => t.toUInt64)

/-- Timed single-prompt generation: reports prefill and decode throughput.
    Uses generateStream; the first callback marks the end of prefill. -/
private def runBench
    (cfg : Config)
    (model : LagunaForCausalLM (device := device) cfg dtype)
    {b seq : Nat}
    (inputIds : Tensor { shape := #[b, seq], dtype := .Int64, device := device })
    (promptLen : Nat)
    (maxNewTokens : Nat)
    (eos : Array UInt64)
    : IO Unit := do
  let firstMsRef ← IO.mkRef 0
  let tokenCount ← IO.mkRef 0
  let startMs ← IO.monoMsNow
  let cb : StreamCallback (device := device) b := fun _step _ids => do
    let n ← tokenCount.get
    tokenCount.set (n + 1)
    if n == 0 then
      let now ← IO.monoMsNow
      firstMsRef.set now
  let ⟨outSeq, _ids⟩ ← model.generateStream cfg inputIds cb maxNewTokens .greedy eos
  let endMs ← IO.monoMsNow
  let n ← tokenCount.get
  let firstMs ← firstMsRef.get
  let prefillMs := firstMs - startMs
  let decodeMs := endMs - firstMs
  let decodeToks := if n == 0 then 0 else n - 1
  let prefillTps := if prefillMs == 0 then 0.0 else promptLen.toFloat * 1000.0 / prefillMs.toFloat
  let decodeTps := if decodeMs == 0 then 0.0 else decodeToks.toFloat * 1000.0 / decodeMs.toFloat
  IO.println s!"BENCH prompt_tokens={promptLen} prefill_ms={prefillMs} prefill_tok_s={prefillTps}"
  IO.println s!"BENCH decode_tokens={decodeToks} decode_ms={decodeMs} decode_tok_s={decodeTps} out_seq={outSeq}"
  IO.println s!"BENCH total_ms={endMs - startMs}"

private def runTextBatches
    (tok : tokenizer.laguna.LagunaTokenizer)
    (cfg : Config)
    (model : LagunaForCausalLM (device := device) cfg dtype)
    (args : Args)
    (prompts : Array String)
    : IO Unit := do
  let eos := cfg.eos_token_ids
  if args.bench then
    -- Bench mode: single prompt, timed generation with throughput report.
    let prompt := prompts[0]?.getD args.prompt
    let ⟨batch, ⟨seq, (inputIds0, promptLens)⟩⟩ ←
      buildBatchInputWithEncoder tokenizer.laguna.padTokenId.toUInt64 #[prompt] (fun p => pure (encodePromptToIds tok args.enableThinking p))
    let inputIds : Tensor { shape := #[batch, seq], dtype := .Int64, device := device } := inputIds0.to device
    let promptLen := (promptLens[0]?.getD 0)
    -- warmup (also builds rope tables / caches), then measured run
    IO.println "Warming up..."
    let _ ← model.generate cfg inputIds 4 .greedy eos
    IO.println "Measuring..."
    runBench cfg model inputIds promptLen args.maxNewTokens eos
  else
    let buildBatch chunk := do
      let ⟨batch, ⟨seq, (inputIds, promptLens)⟩⟩ ←
        buildBatchInputWithEncoder tokenizer.laguna.padTokenId.toUInt64 chunk (fun p => pure (encodePromptToIds tok args.enableThinking p))
      pure ⟨batch, ⟨seq, (inputIds.to device, promptLens)⟩⟩
    let decodeBatch (promptLens : Array Nat) {b os : Nat} (ids : Tensor { shape := #[b, os], dtype := .Int64, device := device }) : IO (Array String) :=
      decodeGeneratedBatch promptLens ids (fun xs => tokenizer.laguna.decodeTokens tok xs)
    let runGen (b : Nat) {s : Nat} (ids : Tensor { shape := #[b, s], dtype := .Int64, device := device }) :=
      model.generate cfg ids args.maxNewTokens .greedy eos
    let runGenStream (b : Nat) {s : Nat} (ids : Tensor { shape := #[b, s], dtype := .Int64, device := device }) (cb : StreamCallback (device := device) b) :=
      model.generateStream cfg ids cb args.maxNewTokens .greedy eos
    runGenerationBatches prompts args.batchSize args.maxNewTokens args.stream args.debugIds
      (prompts.size == 1) buildBatch runGen runGenStream decodeBatch
      (makeStreamCallback (fun x => tokenizer.laguna.decodeOne tok x))

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
  IO.println s!"Model directory: {modelDir}"
  match deviceWarning? with
  | some msg => IO.println msg
  | none => pure ()
  IO.println s!"Using device: {deviceToString device}"

  let tok ← tokenizer.laguna.loadTokenizer modelDir
  let prompts ← loadPrompts args.promptFile args.prompt
  let cfg ← Config.loadFromPretrainedDir modelDir Config.laguna_s_2_1
  let isSharded ← Hub.detectWeightLayout modelDir
  let model ←
    if isSharded then LagunaForCausalLM.loadSharded modelDir cfg device
    else LagunaForCausalLM.load s!"{modelDir}/model.safetensors" cfg device
  runTextBatches tok cfg model.2 args prompts
  pure 0

end Examples.Laguna

def main (argv : List String) : IO UInt32 :=
  Examples.Laguna.runMain argv
