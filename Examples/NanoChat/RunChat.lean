/-
  Examples/NanoChat/RunChat.lean

  Checkpoint-backed NanoChat inference runner.

  This executable loads a trained NanoChat checkpoint + tokenizer and runs
  autoregressive decoding with the same per-token sampling path used in RL.
-/
import Tyr.Torch
import Tyr.Tokenizer
import Examples.NanoChat.ModdedGPT
import Examples.NanoChat.ModdedTrain

namespace torch.NanoChat.Chat

open torch

variable {device : Device}
open torch.moddedGpt
open torch.ModdedTrain
open tokenizer

/-- Runtime environment state for chat inference defaults. -/
private structure ChatEnvState where
  baseDir : String := "~/.cache/nanochat"
  modelDepth : Nat := 20
  vocabSize : Nat := 65536
  requestedDevice? : Option String := none
  deriving Repr, Inhabited

private def loadChatEnvState : IO ChatEnvState := do
  let baseDir := (← IO.getEnv "NANOCHAT_DIR").getD "~/.cache/nanochat"
  let modelDepth := ((← IO.getEnv "MODEL_DEPTH").bind String.toNat?) |>.getD 20
  let vocabSize := ((← IO.getEnv "VOCAB_SIZE").bind String.toNat?) |>.getD 65536
  let requestedDevice? := (← IO.getEnv "TYR_DEVICE").map String.toLower
  pure {
    baseDir := baseDir
    modelDepth := modelDepth
    vocabSize := vocabSize
    requestedDevice? := requestedDevice?
  }

structure Args where
  baseDir : String
  checkpoint : Option String := none
  stage : String := "rl"
  tokenizer : Option String := none
  prompt : Option String := none
  interactive : Bool := false
  useChatMarkers : Bool := false
  maxNewTokens : Nat := 128
  temperature : Float := 1.0
  topK : Nat := 50
  eosToken : Option UInt64 := none
  seed : UInt64 := 42
  modelDepth : Nat := 20
  vocabSize : Nat := 65536
  showTokens : Bool := false
  help : Bool := false
  deriving Repr, Inhabited

private def parseFloatLit? (s : String) : Option Float :=
  match s.splitOn "." with
  | [whole] =>
      whole.toNat?.map (·.toFloat)
  | [whole, frac] =>
      match whole.toNat?, frac.toNat? with
      | some w, some f =>
          let denom : Float := (Nat.pow 10 frac.length).toFloat
          some (w.toFloat + f.toFloat / denom)
      | _, _ => none
  | _ => none

private def withPrompt (acc : Args) (piece : String) : Args :=
  match acc.prompt with
  | none => { acc with prompt := some piece }
  | some p => { acc with prompt := some (p ++ " " ++ piece) }

private partial def parseArgsLoop : List String → Args → Args
  | [], acc => acc
  | "--base-dir" :: v :: rest, acc => parseArgsLoop rest { acc with baseDir := v }
  | "--checkpoint" :: v :: rest, acc => parseArgsLoop rest { acc with checkpoint := some v }
  | "--stage" :: v :: rest, acc => parseArgsLoop rest { acc with stage := v }
  | "--tokenizer" :: v :: rest, acc => parseArgsLoop rest { acc with tokenizer := some v }
  | "--prompt" :: v :: rest, acc => parseArgsLoop rest { acc with prompt := some v }
  | "--max-new-tokens" :: v :: rest, acc =>
      let n := v.toNat?.getD acc.maxNewTokens
      parseArgsLoop rest { acc with maxNewTokens := n }
  | "--temperature" :: v :: rest, acc =>
      let t := (parseFloatLit? v).getD acc.temperature
      parseArgsLoop rest { acc with temperature := t }
  | "--top-k" :: v :: rest, acc =>
      let k := v.toNat?.getD acc.topK
      parseArgsLoop rest { acc with topK := k }
  | "--eos-token" :: v :: rest, acc =>
      let eos? := (v.toNat?).map (·.toUInt64)
      parseArgsLoop rest { acc with eosToken := eos? }
  | "--seed" :: v :: rest, acc =>
      let seed := (v.toNat?).map (·.toUInt64) |>.getD acc.seed
      parseArgsLoop rest { acc with seed := seed }
  | "--model-depth" :: v :: rest, acc =>
      let d := (v.toNat?) |>.getD acc.modelDepth
      parseArgsLoop rest { acc with modelDepth := d }
  | "--vocab-size" :: v :: rest, acc =>
      let vSize := (v.toNat?) |>.getD acc.vocabSize
      parseArgsLoop rest { acc with vocabSize := vSize }
  | "--interactive" :: rest, acc => parseArgsLoop rest { acc with interactive := true }
  | "--chat-markers" :: rest, acc => parseArgsLoop rest { acc with useChatMarkers := true }
  | "--show-tokens" :: rest, acc => parseArgsLoop rest { acc with showTokens := true }
  | "--help" :: rest, acc => parseArgsLoop rest { acc with help := true }
  | arg :: rest, acc =>
      if arg.startsWith "--" then
        parseArgsLoop rest acc
      else
        parseArgsLoop rest (withPrompt acc arg)

private def mkDefaultArgs (env : ChatEnvState) : Args := {
  baseDir := env.baseDir
  modelDepth := env.modelDepth
  vocabSize := env.vocabSize
}

private def parseArgs (raw : List String) (defaults : Args) : Args :=
  parseArgsLoop raw defaults

private def printUsage : IO Unit := do
  IO.println "Usage: lake exe NanoChatChat [options] [prompt text]"
  IO.println ""
  IO.println "Options:"
  IO.println "  --base-dir <path>       Base nanochat directory (default: $NANOCHAT_DIR or ~/.cache/nanochat)"
  IO.println "  --checkpoint <path>     Explicit checkpoint path (overrides --stage)"
  IO.println "  --stage <name>          Checkpoint stage under base dir (default: rl)"
  IO.println "  --tokenizer <path>      Tokenizer file or tokenizer directory"
  IO.println "  --prompt <text>         One-shot prompt"
  IO.println "  --interactive           Interactive REPL mode"
  IO.println "  --chat-markers          Wrap prompts using tokenizer chat markers"
  IO.println "  --max-new-tokens <n>    Decode length cap (default: 128)"
  IO.println "  --temperature <f>       Sampling temperature (default: 1.0)"
  IO.println "  --top-k <n>             Top-k filter (default: 50, 0 disables)"
  IO.println "  --eos-token <id>        End token ID (default: bos, or assistant_end with --chat-markers)"
  IO.println "  --model-depth <n>       Model depth used to load checkpoint (default: $MODEL_DEPTH or 20)"
  IO.println "  --vocab-size <n>        Vocab size used to load checkpoint (default: $VOCAB_SIZE or 65536)"
  IO.println "  --show-tokens           Print generated token IDs"
  IO.println "  --help                  Show this help"

private def expandHome (path : String) : IO String := do
  if path == "~" then
    return (← IO.getEnv "HOME").getD path
  else if path.startsWith "~/" then
    return s!"{(← IO.getEnv "HOME").getD ""}/{path.drop 2}"
  else
    return path

private def resolveCheckpointPath (args : Args) : IO String := do
  match args.checkpoint with
  | some p => expandHome p
  | none =>
    let baseDir ← expandHome args.baseDir
    pure s!"{baseDir}/checkpoints/{args.stage}/latest.ckpt"

private def resolveTokenizerPath (args : Args) : IO String := do
  let defaultPath := s!"{(← expandHome args.baseDir)}/tokenizer/tokenizer.bin"
  match args.tokenizer with
  | none => pure defaultPath
  | some p =>
    let p ← expandHome p
    if p.endsWith ".bin" then
      pure p
    else
      pure s!"{p}/tokenizer.bin"

private def deviceToString (device : Device) : String :=
  match device.kind with
  | DeviceKind.MPS => "MPS"
  | DeviceKind.CPU => "CPU"
  | DeviceKind.CUDA n => s!"CUDA:{n}"

private def resolveDevice (env : ChatEnvState) : IO Device := do
  let requested? := env.requestedDevice?
  match requested? with
  | some "cpu" => pure Device.CPU
  | some "mps" => Device.mps
  | some "cuda" =>
    if ← cuda_is_available then Device.cuda 0 else pure Device.CPU
  | some "auto" | none => getBestDevice
  | some _ => getBestDevice


private def findSpecialId? (tok : BPETokenizer) (names : Array String) : Option UInt64 := Id.run do
  for name in names do
    match tok.specialTokens.get? name with
    | some id => return some id.toUInt64
    | none => pure ()
  none

private def requireSpecialId (tok : BPETokenizer) (label : String) (names : Array String) : IO UInt64 := do
  match findSpecialId? tok names with
  | some id => pure id
  | none => throw <| IO.userError s!"Tokenizer missing {label} token. Tried: {repr names}"

private structure ChatMarkerIds where
  bos : UInt64
  userStart : UInt64
  userEnd : UInt64
  assistantStart : UInt64
  assistantEnd : UInt64

private def resolveChatMarkers (tok : BPETokenizer) : IO ChatMarkerIds := do
  let bos ← requireSpecialId tok "bos" #["<|bos|>", "<|endoftext|>"]
  let userStart ← requireSpecialId tok "user_start" #["<|user_start|>", "<|user|>"]
  let userEnd ← requireSpecialId tok "user_end" #["<|user_end|>", "<|eot|>"]
  let assistantStart ← requireSpecialId tok "assistant_start" #["<|assistant_start|>", "<|assistant|>"]
  let assistantEnd ← requireSpecialId tok "assistant_end" #["<|assistant_end|>", "<|eot|>"]
  pure { bos, userStart, userEnd, assistantStart, assistantEnd }

private def buildModelConfig (args : Args) : Config := {
  vocabSize := args.vocabSize
  nLayer := args.modelDepth
  nHead := 16
  headDim := 64
  modelDim := 1024
  maxSeqLen := 2048
  blockSize := 128
  ropeBase := 10000.0
}

private def sampleNextToken {cfg : Config}
    (params : ModdedGPTParams (device := device) cfg)
    (yarn : YarnRotary (device := device) cfg.headDim cfg.maxSeqLen)
    (temperature : Float)
    (topK : Nat)
    (context : Array UInt64)
    : IO UInt64 := do
  if context.isEmpty then
    throw <| IO.userError "Cannot decode from an empty prompt."

  let inputTensor := Tensor.fromInt64Array (context.map (·.toInt64))
  let inputReshaped := (inputTensor.reshape #[1, context.size]).to device
  let logits ← moddedGpt.forward params yarn inputReshaped false

  -- Keep the final time step, then view its logits as one sampling row.
  let lastLogits := Tensor.sliceChecked logits 1 (context.size - 1) 1
    #[1, 1, cfg.vocabSize]
  let rowLogits := lastLogits.reshape #[1, cfg.vocabSize]
  let scaled :=
    if temperature == 1.0 then rowLogits
    else Tensor.mulScalar rowLogits (1.0 / temperature)
  let filtered :=
    if topK == 0 then scaled
    else Tensor.topKFilter scaled topK
  let probs := Tensor.softmax filtered (-1)
  let sampled ← Tensor.multinomial probs 1
  pure (Tensor.itemInt (sampled.reshape #[])).toUInt64

private def promptWithMarkers (markers : ChatMarkerIds) (base : Array UInt64) (prompt : Array UInt64)
    : Array UInt64 :=
  (if base.isEmpty then #[markers.bos] else base) ++
    #[markers.userStart] ++ prompt ++ #[markers.userEnd, markers.assistantStart]

private def generateTokens
    (promptTokens : Array UInt64)
    (nextTokenFn : Array UInt64 → IO UInt64)
    (maxNewTokens : Nat)
    (eosToken : UInt64)
    : IO (Array UInt64) := do
  let mut full := promptTokens
  let mut generated : Array UInt64 := #[]
  for _ in [:maxNewTokens] do
    let tok ← nextTokenFn full
    generated := generated.push tok
    full := full.push tok
    if tok == eosToken then
      break
  pure generated

unsafe def main (rawArgs : List String) : IO UInt32 := do
  let envState ← loadChatEnvState
  let defaults := mkDefaultArgs envState
  let args := parseArgs rawArgs defaults

  if args.help then
    printUsage
    return 0

  let checkpointPath ← resolveCheckpointPath args
  let tokenizerPath ← resolveTokenizerPath args
  let cfg := buildModelConfig args
  let device ← resolveDevice envState
  let tok ← tokenizer.load tokenizerPath
  let markers ← resolveChatMarkers tok
  let eosToken := args.eosToken.getD (if args.useChatMarkers then markers.assistantEnd else markers.bos)

  manualSeed args.seed

  IO.println s!"NanoChat inference:"
  IO.println s!"  checkpoint: {checkpointPath}"
  IO.println s!"  tokenizer:  {tokenizerPath}"
  IO.println s!"  device:     {deviceToString device}"
  IO.println s!"  model:      depth={cfg.nLayer} vocab={cfg.vocabSize}"
  IO.println s!"  decoding:   max_new={args.maxNewTokens} temp={args.temperature} top_k={args.topK} eos={eosToken}"
  IO.println ""

  let maybeCkpt ← loadCheckpoint cfg checkpointPath (device := device)
  let ckpt ← match maybeCkpt with
    | some ckpt => pure ckpt
    | none => throw <| IO.userError s!"Checkpoint not found or incompatible with current config: {checkpointPath}"

  let paramsOnDevice ← TensorStruct.makeLeafParams ckpt.params

  let yarn ← YarnRotary.init cfg.headDim cfg.maxSeqLen cfg.ropeBase (device := device)

  let encodeFn := fun (text : String) => (tokenizer.encodeWithSpecials tok text).map (·.toUInt64)
  let decodeFn := fun (tokens : Array UInt64) => tokenizer.decode tok (tokens.map (·.toUInt32))
  let nextTokenFn := sampleNextToken paramsOnDevice yarn args.temperature args.topK

  let runTurn := fun (baseCtx : Array UInt64) (prompt : String) => do
    let promptBody := encodeFn prompt
    let promptTokens :=
      if args.useChatMarkers then
        promptWithMarkers markers baseCtx promptBody
      else
        promptBody
    if promptTokens.isEmpty then
      throw <| IO.userError "Prompt encoded to zero tokens."
    let generated ← generateTokens promptTokens nextTokenFn args.maxNewTokens eosToken
    let response := decodeFn generated
    if args.showTokens then
      IO.println s!"generated_tokens={generated}"
    let generatedForCtx :=
      if args.useChatMarkers && (generated.isEmpty || generated.back! != markers.assistantEnd) then
        generated.push markers.assistantEnd
      else
        generated
    let nextCtx :=
      if args.useChatMarkers then
        promptTokens ++ generatedForCtx
      else
        #[]
    pure (response, nextCtx)

  if args.interactive then
    IO.println "Interactive mode. Type :quit to exit."
    let rec loop (ctx : Array UInt64) : IO UInt32 := do
      IO.print "user> "
      let line ← (← IO.getStdin).getLine
      let prompt := line.trimAscii.toString
      if prompt.isEmpty then
        loop ctx
      else if prompt == ":quit" || prompt == ":q" || prompt == ":exit" then
        pure 0
      else
        let (response, ctx') ← runTurn ctx prompt
        IO.println s!"assistant> {response}"
        IO.println ""
        loop ctx'
    loop #[]
  else
    let prompt := args.prompt.getD "What is 2+2?"
    let (response, _) ← runTurn #[] prompt
    IO.println s!"prompt: {prompt}"
    IO.println s!"completion: {response}"
    pure 0

end torch.NanoChat.Chat

unsafe def main : List String → IO UInt32 := torch.NanoChat.Chat.main
