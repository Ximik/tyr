/- 
  examples/Qwen3TTS/EndToEnd.lean

  End-to-end Qwen3-TTS demo in Lean:
  - load config + real checkpoint weights
  - tokenize text prompt
  - generate codec tokens with Lean talker
  - decode codec tokens to waveform with Lean speech-tokenizer decoder
-/
import Tyr.Torch
import Tyr.Model.Qwen3TTS
import Tyr.Tokenizer
import Tyr.Model.Qwen35.TypedOps

namespace Examples.Qwen3TTS

open torch

variable {device : Device}
open torch.qwen3tts
open tokenizer.qwen3

structure Args where
  modelDir : String := "weights/qwen3-tts"
  text : String := "Hello from Lean Qwen3-TTS."
  styleText : Option String := none
  maxTextLen : Nat := 512
  maxFrames : Nat := 256
  language : String := "Auto"
  thinkMode : String := "auto"
  seed : Option UInt64 := none
  temperature : Float := 0.9
  topK : Nat := 50
  topP : Float := 1.0
  repetitionPenalty : Float := 1.05
  suppressTail : Nat := 1024
  subtalkerTemperature : Float := 0.9
  subtalkerTopK : Nat := 50
  subtalkerTopP : Float := 1.0
  subtalkerTemperaturesByGroup : Option (Array Float) := none
  subtalkerTopKsByGroup : Option (Array Nat) := none
  subtalkerTopPsByGroup : Option (Array Float) := none
  encodeAudioPath : Option String := none
  encodeOutCodesPath : String := "output/qwen3tts_encoded_codes.txt"
  encodeOnly : Bool := false
  codesPath : String := "output/qwen3tts_codes.txt"
  wavPath : String := "output/qwen3tts.wav"
  skipDecode : Bool := false
  pythonExe : String := "uv"
  decodeScript : String := "scripts/qwen3tts_decode_codes.py"
  encodeScript : String := "scripts/qwen3tts_encode_audio.py"
  speakerMelScript : String := "scripts/qwen3tts_prepare_speaker_mel.py"
  deviceMap : Option String := none
  speakerEmbeddingPath : Option String := none
  refAudioPath : Option String := none
  speakerMelPath : String := ""
  speakerMelFramesPath : String := ""
  qwenRepo : String := "../Qwen3-TTS"
  speechTokenizerDir : Option String := none
  decodeViaPython : Bool := false
  streamingDecode : Bool := false
  trueStreaming : Bool := false
  streamChunkFrames : Nat := 4
  decodeChunkSize : Nat := 300
  decodeLeftContext : Nat := 25
  tokenIdsPath : String := "output/qwen3tts_text_ids.txt"
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

private def parseFloatCsv? (s : String) : Option (Array Float) := Id.run do
  let mut out : Array Float := #[]
  for part in s.splitOn "," do
    let tok := part.trimAscii.toString
    if tok.isEmpty then
      return none
    match parseFloatLit? tok with
    | some v => out := out.push v
    | none => return none
  if out.isEmpty then none else some out

private def parseNatCsv? (s : String) : Option (Array Nat) := Id.run do
  let mut out : Array Nat := #[]
  for part in s.splitOn "," do
    let tok := part.trimAscii.toString
    if tok.isEmpty then
      return none
    match tok.toNat? with
    | some v => out := out.push v
    | none => return none
  if out.isEmpty then none else some out

private inductive ThinkMode where
  | auto
  | think
  | noThink
  deriving Repr, Inhabited, BEq

private partial def parseArgsLoop : List String → Args → Args
  | [], acc => acc
  | "--model-dir" :: v :: rest, acc => parseArgsLoop rest { acc with modelDir := v }
  | "--text" :: v :: rest, acc => parseArgsLoop rest { acc with text := v }
  | "--style-text" :: v :: rest, acc => parseArgsLoop rest { acc with styleText := some v }
  | "--max-text-len" :: v :: rest, acc =>
      parseArgsLoop rest { acc with maxTextLen := v.toNat?.getD acc.maxTextLen }
  | "--max-frames" :: v :: rest, acc =>
      parseArgsLoop rest { acc with maxFrames := v.toNat?.getD acc.maxFrames }
  | "--language" :: v :: rest, acc => parseArgsLoop rest { acc with language := v }
  | "--think-mode" :: v :: rest, acc => parseArgsLoop rest { acc with thinkMode := v }
  | "--seed" :: v :: rest, acc =>
      parseArgsLoop rest { acc with seed := v.toNat?.map UInt64.ofNat }
  | "--temperature" :: v :: rest, acc =>
      parseArgsLoop rest { acc with temperature := (parseFloatLit? v).getD acc.temperature }
  | "--top-k" :: v :: rest, acc =>
      parseArgsLoop rest { acc with topK := v.toNat?.getD acc.topK }
  | "--top-p" :: v :: rest, acc =>
      parseArgsLoop rest { acc with topP := (parseFloatLit? v).getD acc.topP }
  | "--repetition-penalty" :: v :: rest, acc =>
      parseArgsLoop rest { acc with repetitionPenalty := (parseFloatLit? v).getD acc.repetitionPenalty }
  | "--suppress-tail" :: v :: rest, acc =>
      parseArgsLoop rest { acc with suppressTail := v.toNat?.getD acc.suppressTail }
  | "--subtalker-temperature" :: v :: rest, acc =>
      parseArgsLoop rest { acc with subtalkerTemperature := (parseFloatLit? v).getD acc.subtalkerTemperature }
  | "--subtalker-top-k" :: v :: rest, acc =>
      parseArgsLoop rest { acc with subtalkerTopK := v.toNat?.getD acc.subtalkerTopK }
  | "--subtalker-top-p" :: v :: rest, acc =>
      parseArgsLoop rest { acc with subtalkerTopP := (parseFloatLit? v).getD acc.subtalkerTopP }
  | "--subtalker-temp-by-group" :: v :: rest, acc =>
      parseArgsLoop rest { acc with subtalkerTemperaturesByGroup := parseFloatCsv? v }
  | "--subtalker-top-k-by-group" :: v :: rest, acc =>
      parseArgsLoop rest { acc with subtalkerTopKsByGroup := parseNatCsv? v }
  | "--subtalker-top-p-by-group" :: v :: rest, acc =>
      parseArgsLoop rest { acc with subtalkerTopPsByGroup := parseFloatCsv? v }
  | "--encode-audio-path" :: v :: rest, acc => parseArgsLoop rest { acc with encodeAudioPath := some v }
  | "--encode-out-codes-path" :: v :: rest, acc => parseArgsLoop rest { acc with encodeOutCodesPath := v }
  | "--encode-only" :: rest, acc => parseArgsLoop rest { acc with encodeOnly := true }
  | "--codes-path" :: v :: rest, acc => parseArgsLoop rest { acc with codesPath := v }
  | "--wav-path" :: v :: rest, acc => parseArgsLoop rest { acc with wavPath := v }
  | "--python" :: v :: rest, acc => parseArgsLoop rest { acc with pythonExe := v }
  | "--decode-script" :: v :: rest, acc => parseArgsLoop rest { acc with decodeScript := v }
  | "--encode-script" :: v :: rest, acc => parseArgsLoop rest { acc with encodeScript := v }
  | "--speaker-mel-script" :: v :: rest, acc => parseArgsLoop rest { acc with speakerMelScript := v }
  | "--device-map" :: v :: rest, acc => parseArgsLoop rest { acc with deviceMap := some v }
  | "--speaker-embedding-path" :: v :: rest, acc =>
      parseArgsLoop rest { acc with speakerEmbeddingPath := some v }
  | "--ref-audio-path" :: v :: rest, acc =>
      parseArgsLoop rest { acc with refAudioPath := some v }
  | "--speaker-mel-path" :: v :: rest, acc => parseArgsLoop rest { acc with speakerMelPath := v }
  | "--speaker-mel-frames-path" :: v :: rest, acc =>
      parseArgsLoop rest { acc with speakerMelFramesPath := v }
  | "--qwen-repo" :: v :: rest, acc => parseArgsLoop rest { acc with qwenRepo := v }
  | "--speech-tokenizer-dir" :: v :: rest, acc =>
      parseArgsLoop rest { acc with speechTokenizerDir := some v }
  | "--decode-via-python" :: rest, acc =>
      parseArgsLoop rest { acc with decodeViaPython := true }
  | "--streaming-decode" :: rest, acc =>
      parseArgsLoop rest { acc with streamingDecode := true }
  | "--true-streaming" :: rest, acc =>
      parseArgsLoop rest { acc with trueStreaming := true }
  | "--stream-chunk-frames" :: v :: rest, acc =>
      parseArgsLoop rest { acc with streamChunkFrames := v.toNat?.getD acc.streamChunkFrames }
  | "--decode-chunk-size" :: v :: rest, acc =>
      parseArgsLoop rest { acc with decodeChunkSize := v.toNat?.getD acc.decodeChunkSize }
  | "--decode-left-context" :: v :: rest, acc =>
      parseArgsLoop rest { acc with decodeLeftContext := v.toNat?.getD acc.decodeLeftContext }
  | "--token-ids-path" :: v :: rest, acc => parseArgsLoop rest { acc with tokenIdsPath := v }
  | "--skip-decode" :: rest, acc => parseArgsLoop rest { acc with skipDecode := true }
  | "--help" :: rest, acc => parseArgsLoop rest { acc with help := true }
  | _ :: rest, acc => parseArgsLoop rest acc

private def parseArgs (raw : List String) : Args :=
  parseArgsLoop raw {}

private def takesValueFlag (flag : String) : Bool :=
  #[
    "--model-dir", "--text", "--style-text", "--max-text-len", "--max-frames",
    "--language", "--think-mode", "--seed", "--temperature", "--top-k", "--top-p",
    "--repetition-penalty", "--suppress-tail", "--subtalker-temperature",
    "--subtalker-top-k", "--subtalker-top-p", "--subtalker-temp-by-group",
    "--subtalker-top-k-by-group", "--subtalker-top-p-by-group", "--encode-audio-path",
    "--encode-out-codes-path", "--codes-path", "--wav-path", "--python",
    "--decode-script", "--encode-script", "--speaker-mel-script", "--device-map",
    "--speaker-embedding-path", "--ref-audio-path", "--speaker-mel-path",
    "--speaker-mel-frames-path", "--qwen-repo", "--speech-tokenizer-dir",
    "--stream-chunk-frames", "--decode-chunk-size", "--decode-left-context",
    "--token-ids-path"
  ].contains flag

private def switchFlag (flag : String) : Bool :=
  #[
    "--encode-only", "--decode-via-python", "--streaming-decode",
    "--true-streaming", "--skip-decode", "--help"
  ].contains flag

private def validateFlagValue (flag : String) (value : String) : Except String Unit := do
  if #["--max-text-len", "--max-frames", "--top-k", "--suppress-tail", "--subtalker-top-k",
        "--stream-chunk-frames", "--decode-chunk-size", "--decode-left-context"].contains flag then
    if value.toNat?.isNone then
      throw s!"Invalid numeric value for {flag}: {value}"
  if #["--temperature", "--top-p", "--repetition-penalty", "--subtalker-temperature",
        "--subtalker-top-p"].contains flag then
    if (parseFloatLit? value).isNone then
      throw s!"Invalid float value for {flag}: {value}"
  if flag == "--seed" then
    if value.toNat?.isNone then
      throw s!"Invalid numeric value for {flag}: {value}"
  if #["--subtalker-temp-by-group", "--subtalker-top-p-by-group"].contains flag then
    if (parseFloatCsv? value).isNone then
      throw s!"Invalid CSV float value for {flag}: {value}"
  if flag == "--subtalker-top-k-by-group" then
    if (parseNatCsv? value).isNone then
      throw s!"Invalid CSV integer value for {flag}: {value}"

private partial def validateRawArgs : List String → Except String Unit
  | [] => .ok ()
  | flag :: rest =>
    if takesValueFlag flag then
      match rest with
      | [] => .error s!"Missing value for option: {flag}"
      | value :: tail =>
        match validateFlagValue flag value with
        | .ok _ => validateRawArgs tail
        | .error e => .error e
    else if switchFlag flag then
      validateRawArgs rest
    else if flag.startsWith "--" then
      .error s!"Unknown option: {flag}"
    else
      .error s!"Unexpected positional argument: {flag}"

private def printUsage : IO Unit := do
  IO.println "Usage: lake exe Qwen3TTSEndToEnd [options]"
  IO.println ""
  IO.println "Options:"
  IO.println "  --model-dir <path>            Qwen3-TTS model directory (contains config.json + sharded weights)"
  IO.println "  --text <prompt>               Text prompt"
  IO.println "  --style-text <prompt>         Optional style prefix text prepended to prompt"
  IO.println "  --max-text-len <n>            Max tokenized prompt length (default: 512)"
  IO.println "  --max-frames <n>              Max generated codec frames (default: 256)"
  IO.println "  --language <name|Auto>        Language conditioning tag (default: Auto)"
  IO.println "  --think-mode <auto|think|no-think>  Reasoning-mode codec prefill control (default: auto)"
  IO.println "  --seed <u64>                  Deterministic RNG seed for sampling"
  IO.println "  --temperature <f>             Sampling temperature (default: 0.9)"
  IO.println "  --top-k <n>                   Top-k sampling (default: 50)"
  IO.println "  --top-p <f>                   Top-p sampling (default: 1.0)"
  IO.println "  --repetition-penalty <f>      Repetition penalty for first-codebook sampling (default: 1.05)"
  IO.println "  --suppress-tail <n>           Suppress top tail vocab IDs except EOS (default: 1024)"
  IO.println "  --subtalker-temperature <f>   Subtalker sampling temperature (default: 0.9)"
  IO.println "  --subtalker-top-k <n>         Subtalker top-k sampling (default: 50)"
  IO.println "  --subtalker-top-p <f>         Subtalker top-p sampling (default: 1.0)"
  IO.println "  --subtalker-temp-by-group <csv>  Residual-group temperatures, e.g. 0.9,0.85,0.8"
  IO.println "  --subtalker-top-k-by-group <csv> Residual-group top-k values, e.g. 50,40,30"
  IO.println "  --subtalker-top-p-by-group <csv> Residual-group top-p values, e.g. 1.0,0.95,0.9"
  IO.println "  --encode-audio-path <path>    Encode audio to speech-tokenizer codec IDs"
  IO.println "  --encode-out-codes-path <p>   Output codec-ID file for --encode-audio-path"
  IO.println "  --encode-only                 Run audio->codes bridge and exit"
  IO.println "  --codes-path <path>           Output codec-token text file"
  IO.println "  --wav-path <path>             Output wav path"
  IO.println "  --skip-decode                 Skip waveform decode (codes only)"
  IO.println "  --python <exe>                Python launcher (default: uv; uses `uv run python`)"
  IO.println "  --decode-script <path>        Codec decode bridge script path"
  IO.println "  --encode-script <path>        Codec encode bridge script path"
  IO.println "  --speaker-mel-script <path>   Speaker mel extraction script path"
  IO.println "  --device-map <v>              Optional HF device_map for speech tokenizer bridge"
  IO.println "  --speaker-embedding-path <p>  Optional speaker embedding tensor path ([enc_dim])"
  IO.println "  --ref-audio-path <path>       Optional reference audio to derive speaker embedding"
  IO.println "  --speaker-mel-path <path>     Optional debug dump: intermediate speaker mel safetensors"
  IO.println "  --speaker-mel-frames-path <p> Optional debug dump: intermediate mel frame-count file"
  IO.println "  --token-ids-path <path>       Output token IDs file from Lean tokenizer"
  IO.println "  --qwen-repo <path>            Local Qwen3-TTS repo path (fallback import)"
  IO.println "  --speech-tokenizer-dir <path> Speech tokenizer dir (default: <model-dir>/speech_tokenizer)"
  IO.println "  --decode-via-python           Force Python decode bridge (auto-used for non-16 code groups)"
  IO.println "  --streaming-decode            Use chunked Lean decoder for long-form decode (12Hz/16-group only)"
  IO.println "  --true-streaming              Online frame-by-frame generation (Lean decode when compatible)"
  IO.println "  --stream-chunk-frames <n>     Frame chunk size emitted by true streaming decoder (default: 4)"
  IO.println "  --decode-chunk-size <n>       Codec-frame chunk size for --streaming-decode (default: 300)"
  IO.println "  --decode-left-context <n>     Left context codec frames for chunked decode (default: 25)"
  IO.println "  --help                        Show this help"

private def expandHome (path : String) : IO String := do
  if path == "~" then
    return (← IO.getEnv "HOME").getD path
  else if path.startsWith "~/" then
    return s!"{(← IO.getEnv "HOME").getD ""}/{path.drop 2}"
  else
    return path

private def ensureParentDir (path : String) : IO Unit := do
  match System.FilePath.parent ⟨path⟩ with
  | some parent =>
      if parent.toString != "" && parent.toString != "." then
        IO.FS.createDirAll parent
  | none => pure ()

private def formatCodesRows (flat : Array UInt64) (numCodeGroups : UInt64) : String := Id.run do
  let g := numCodeGroups.toNat
  if g == 0 then
    return ""
  let rows := flat.size / g
  let mut out := ""
  for r in [:rows] do
    for c in [:g] do
      if c > 0 then
        out := out ++ " "
      out := out ++ toString (flat[r * g + c]!)
    out := out ++ "\n"
  out

private def formatCodesMatrix (rows : Array (Array UInt64)) : String := Id.run do
  let mut out := ""
  for row in rows do
    out := out ++ String.intercalate " " (row.toList.map toString) ++ "\n"
  out

private def lowerAsciiString (s : String) : String :=
  String.ofList <| s.toList.map (fun c =>
    if c >= 'A' && c <= 'Z' then Char.ofNat (c.toNat + 32) else c)

private def parseThinkMode (s : String) : ThinkMode :=
  match lowerAsciiString s with
  | "think" => .think
  | "no-think" => .noThink
  | "nothink" => .noThink
  | "no_think" => .noThink
  | _ => .auto

private def lookupLanguageId? (cfg : Qwen3TTSConfig) (language : String) : Option UInt64 := Id.run do
  let q := lowerAsciiString language
  for (k, v) in cfg.talkerConfig.codecLanguageId do
    if lowerAsciiString k == q then
      return some v
  none

private def idsTensor1 (ids : Array UInt64) : Tensor { shape := #[1, ids.size.toUInt64], dtype := .Int64, device := device } :=
  (Tensor.reshapeChecked (Tensor.fromInt64Array (ids.map (Int64.ofNat ∘ UInt64.toNat))) #[1, ids.size.toUInt64]).to device

private def saveTokenIds (path : String) (ids : Array UInt64) : IO Unit := do
  ensureParentDir path
  let text := String.intercalate " " (ids.toList.map toString)
  IO.FS.writeFile path (text ++ "\n")

private def loadSpeakerEmbedding? (cfg : Qwen3TTSConfig) (path? : Option String)
    : IO (Option (Tensor { shape := #[1, 1, cfg.talkerConfig.hiddenSize], device := device })) := do
  match path? with
  | none => pure none
  | some p =>
      if cfg.speakerEncoderConfig.encDim != cfg.talkerConfig.hiddenSize then
        throw <| IO.userError
          s!"speaker embedding dim mismatch: speaker enc_dim={cfg.speakerEncoderConfig.encDim}, talker hidden={cfg.talkerConfig.hiddenSize}"
      let path ← expandHome p
      let emb ← Tensor.loadTensorExact { shape := #[cfg.speakerEncoderConfig.encDim], dtype := .Unknown "dynamic", device := .CPU } path
      pure (some (Tensor.reshapeChecked (emb.to device) #[1, 1, cfg.talkerConfig.hiddenSize]))

private def buildTalkerInputsEquivalent
    (cfg : Qwen3TTSConfig)
    (model : Qwen3TTSForConditionalGeneration (device := device) cfg)
    (inputIds : Array UInt64)
    (languageId : Option UInt64)
    (thinkMode : ThinkMode := .auto)
    (speakerEmbed : Option (Tensor { shape := #[1, 1, cfg.talkerConfig.hiddenSize], device := device }) := none)
    : IO ((Sigma fun seq => Tensor { shape := #[1, seq, cfg.talkerConfig.hiddenSize], device := device }) × Tensor { shape := #[1, 1, cfg.talkerConfig.hiddenSize], device := device }) := do
  if inputIds.size < 8 then
    throw <| IO.userError s!"Tokenized assistant input is too short ({inputIds.size}); expected at least 8 tokens."

  let roleIds := inputIds.extract 0 3
  if roleIds.size != 3 then
    throw <| IO.userError s!"Unexpected role-token size: {roleIds.size} (expected 3)."
  let textMainIds := inputIds.extract 3 (inputIds.size - 5)
  let textMainLen : UInt64 := textMainIds.size.toUInt64
  let hidden : UInt64 := cfg.talkerConfig.hiddenSize

  let roleTensor : Tensor { shape := #[1, 3], dtype := .Int64, device := device } := Tensor.reshapeChecked (idsTensor1 roleIds) #[1, 3]
  let roleEmbed : Tensor { shape := #[1, 3, hidden], device := device } := TalkerModel.embedText model.talker.model roleTensor

  let ttsSpecialIds : Tensor { shape := #[1, 3], dtype := .Int64, device := device } := Tensor.reshapeChecked ((Tensor.fromInt64Array #[
    Int64.ofNat cfg.ttsBosTokenId.toNat,
    Int64.ofNat cfg.ttsEosTokenId.toNat,
    Int64.ofNat cfg.ttsPadTokenId.toNat
  ]).to device) #[1, 3]
  let ttsSpecialEmb : Tensor { shape := #[1, 3, hidden], device := device } := TalkerModel.embedText model.talker.model ttsSpecialIds
  let ttsBosEmbed : Tensor { shape := #[1, 1, hidden], device := device } := qwen35.typedSlice ttsSpecialEmb 1 0 1
  let ttsEosEmbed : Tensor { shape := #[1, 1, hidden], device := device } := qwen35.typedSlice ttsSpecialEmb 1 1 1
  let ttsPadEmbed : Tensor { shape := #[1, 1, hidden], device := device } := qwen35.typedSlice ttsSpecialEmb 1 2 1

  let codecPrefillIds : Array UInt64 :=
    match thinkMode, languageId with
    | .noThink, _ =>
        #[
          cfg.talkerConfig.codecNoThinkId,
          cfg.talkerConfig.codecThinkBosId,
          cfg.talkerConfig.codecThinkEosId
        ]
    | .think, some langId =>
        #[
          cfg.talkerConfig.codecThinkId,
          cfg.talkerConfig.codecThinkBosId,
          langId,
          cfg.talkerConfig.codecThinkEosId
        ]
    | .think, none =>
        #[
          cfg.talkerConfig.codecThinkId,
          cfg.talkerConfig.codecThinkBosId,
          cfg.talkerConfig.codecThinkEosId
        ]
    | .auto, some langId =>
        #[
          cfg.talkerConfig.codecThinkId,
          cfg.talkerConfig.codecThinkBosId,
          langId,
          cfg.talkerConfig.codecThinkEosId
        ]
    | .auto, none =>
        #[
          cfg.talkerConfig.codecNoThinkId,
          cfg.talkerConfig.codecThinkBosId,
          cfg.talkerConfig.codecThinkEosId
        ]
  let codecPrefillLen : UInt64 := codecPrefillIds.size.toUInt64
  let codecPrefillTensor : Tensor { shape := #[1, codecPrefillLen], dtype := .Int64, device := device } := idsTensor1 codecPrefillIds
  let codecEmb0 : Tensor { shape := #[1, codecPrefillLen, hidden], device := device } := TalkerModel.embedCodec model.talker.model codecPrefillTensor

  let codecPadBosIds : Tensor { shape := #[1, 2], dtype := .Int64, device := device } := Tensor.reshapeChecked ((Tensor.fromInt64Array #[
    Int64.ofNat cfg.talkerConfig.codecPadId.toNat,
    Int64.ofNat cfg.talkerConfig.codecBosId.toNat
  ]).to device) #[1, 2]
  let codecEmb1 : Tensor { shape := #[1, 2, hidden], device := device } := TalkerModel.embedCodec model.talker.model codecPadBosIds

  let codecInputResult : (Sigma fun n => Tensor { shape := #[1, n, hidden], device := device }) :=
    match speakerEmbed with
    | some spk =>
        let codecWithSpeaker : Tensor { shape := #[1, codecPrefillLen + 1, hidden], device := device } := qwen35.typedCat codecEmb0 spk 1
        let codecInput : Tensor { shape := #[1, (codecPrefillLen + 1) + 2, hidden], device := device } := qwen35.typedCat codecWithSpeaker codecEmb1 1
        ⟨((codecPrefillLen + 1) + 2), codecInput⟩
    | none =>
        let codecInput : Tensor { shape := #[1, codecPrefillLen + 2, hidden], device := device } := qwen35.typedCat codecEmb0 codecEmb1 1
        ⟨(codecPrefillLen + 2), codecInput⟩

  let ⟨codecInputLen, codecInput⟩ := codecInputResult
  let codecPrefix : Tensor { shape := #[1, codecInputLen - 1, hidden], device := device } := qwen35.typedSlice codecInput 1 0 (codecInputLen - 1)
  let padScaffold : Tensor { shape := #[1, codecInputLen - 2, hidden], device := device } := Tensor.expand ttsPadEmbed #[1, codecInputLen - 2, hidden]
  let scaffold0 : Tensor { shape := #[1, codecInputLen - 1, hidden], device := device } :=
    qwen35.typedCat padScaffold ttsBosEmbed 1
  let scaffold : Tensor { shape := #[1, codecInputLen - 1, hidden], device := device } := Tensor.Promoting.add scaffold0 codecPrefix
  let partA : Tensor { shape := #[1, 3 + (codecInputLen - 1), hidden], device := device } := qwen35.typedCat roleEmbed scaffold 1

  let textMainTensor : Tensor { shape := #[1, textMainLen], dtype := .Int64, device := device } := idsTensor1 textMainIds
  let textMainEmbed : Tensor { shape := #[1, textMainLen, hidden], device := device } := TalkerModel.embedText model.talker.model textMainTensor
  let textPlusEos : Tensor { shape := #[1, textMainLen + 1, hidden], device := device } := qwen35.typedCat textMainEmbed ttsEosEmbed 1
  let codecPadIds : Tensor { shape := #[1, textMainLen + 1], dtype := .Int64, device := device } :=
    (Tensor.fullInt #[1, textMainLen + 1] (Int64.ofNat cfg.talkerConfig.codecPadId.toNat)).to device
  let codecPadEmbed : Tensor { shape := #[1, textMainLen + 1, hidden], device := device } := TalkerModel.embedCodec model.talker.model codecPadIds
  let partB : Tensor { shape := #[1, textMainLen + 1, hidden], device := device } := Tensor.Promoting.add textPlusEos codecPadEmbed

  let codecBosOnlyIds : Tensor { shape := #[1, 1], dtype := .Int64, device := device } := (Tensor.fullInt #[1, 1] (Int64.ofNat cfg.talkerConfig.codecBosId.toNat)).to device
  let codecBosEmbed : Tensor { shape := #[1, 1, hidden], device := device } := TalkerModel.embedCodec model.talker.model codecBosOnlyIds
  let partC : Tensor { shape := #[1, 1, hidden], device := device } := Tensor.Promoting.add ttsPadEmbed codecBosEmbed

  let partABLen : UInt64 := (3 + (codecInputLen - 1)) + (textMainLen + 1)
  let partAB : Tensor { shape := #[1, partABLen, hidden], device := device } := qwen35.typedCat partA partB 1
  let outLen : UInt64 := partABLen + 1
  let talkerInputs : Tensor { shape := #[1, outLen, hidden], device := device } := qwen35.typedCat partAB partC 1
  pure ((⟨outLen, talkerInputs⟩), ttsPadEmbed)

private def buildBridgeConfig (args : Args) : SpeechTokenizerBridgeConfig := {
  pythonExe := args.pythonExe
  qwenRepo := args.qwenRepo
  speechTokenizerDir := args.speechTokenizerDir
  decodeScript := args.decodeScript
  encodeScript := args.encodeScript
  speakerMelScript := args.speakerMelScript
  deviceMap := args.deviceMap
}

private def resolveRuntimeDevice : IO Device := do
  let requested := (← IO.getEnv "TYR_DEVICE").map String.toLower
  match requested with
  | some "cpu" => pure Device.CPU
  | some "cuda" => pure (Device.CUDA 0)
  | some "mps" => pure Device.MPS
  | some "auto" => getBestDevice
  | some _ => getBestDevice
  | none => getBestDevice

def runEndToEnd (args : Args) : IO Unit := do
  let modelDir ← expandHome args.modelDir
  let codesPath ← expandHome args.codesPath
  let encodeOutCodesPath ← expandHome args.encodeOutCodesPath
  let wavPath ← expandHome args.wavPath
  let tokenIdsPath ← expandHome args.tokenIdsPath
  let speakerMelPath ← expandHome args.speakerMelPath
  let speakerMelFramesPath ← expandHome args.speakerMelFramesPath
  let targetDevice ← resolveRuntimeDevice
  let bridgeCfgBase := buildBridgeConfig args
  let bridgeCfg : SpeechTokenizerBridgeConfig :=
    match bridgeCfgBase.deviceMap with
    | some _ => bridgeCfgBase
    | none =>
        match targetDevice with
        | Device.MPS => { bridgeCfgBase with deviceMap := some "mps" }
        | Device.CUDA _ => { bridgeCfgBase with deviceMap := some "cuda" }
        | _ => bridgeCfgBase

  IO.println "=== Qwen3-TTS End-to-End (Lean) ==="
  IO.println s!"Model dir: {modelDir}"
  IO.println s!"Prompt: {args.text}"
  IO.println s!"Language: {args.language}"
  IO.println s!"Target device: {repr targetDevice}"
  let thinkMode := parseThinkMode args.thinkMode
  let subtalkerTopKsByGroup : Option (Array UInt64) :=
    args.subtalkerTopKsByGroup.map (fun xs => xs.map Nat.toUInt64)
  match args.seed with
  | some seed =>
      torch.manualSeed seed
      IO.println s!"Sampling seed: {seed}"
  | none =>
      pure ()

  -- Load runtime config from HF config.json so tensor shapes match real checkpoints.
  let cfg ← Qwen3TTSConfig.loadFromPretrainedDir modelDir
  IO.println s!"Model type: {cfg.ttsModelType}, talker hidden={cfg.talkerConfig.hiddenSize}, code groups={cfg.talkerConfig.numCodeGroups}"
  let vocabTopKMax := cfg.talkerConfig.codePredictorConfig.vocabSize.toNat
  if args.topK > vocabTopKMax then
    throw <| IO.userError
      s!"--top-k ({args.topK}) exceeds talker vocab size ({vocabTopKMax})"
  if args.subtalkerTopK > vocabTopKMax then
    throw <| IO.userError
      s!"--subtalker-top-k ({args.subtalkerTopK}) exceeds talker vocab size ({vocabTopKMax})"
  match args.subtalkerTopKsByGroup with
  | some ks =>
      for i in [:ks.size] do
        let k := ks.getD i 0
        if k > vocabTopKMax then
          throw <| IO.userError
            s!"--subtalker-top-k-by-group[{i}] ({k}) exceeds talker vocab size ({vocabTopKMax})"
  | none =>
      pure ()

  match args.encodeAudioPath with
  | some audioPath =>
      let audioPath ← expandHome audioPath
      encodeAudioToCodes bridgeCfg modelDir audioPath encodeOutCodesPath targetDevice
      IO.println s!"Saved encoded audio codec IDs to {encodeOutCodesPath}"
      if args.encodeOnly then
        IO.println "Skipping TTS generation (--encode-only enabled)."
        return
  | none =>
      if args.encodeOnly then
        throw <| IO.userError "--encode-only requires --encode-audio-path"

  let model ← Qwen3TTSForConditionalGeneration.loadSharded modelDir cfg targetDevice

  let tok ← tokenizer.qwen3.loadTokenizer modelDir
  let baseText :=
    match args.styleText with
    | some style =>
        let style := style.trimAscii.toString
        if style.isEmpty then
          args.text
        else
          s!"{style}\n{args.text}"
    | none =>
        args.text
  let assistantText := tokenizer.qwen3.ttsAssistantText baseText
  let rawTokenIds := (tokenizer.qwen3.encodeText tok assistantText).map (fun t => t.toUInt64)
  let tokenIds :=
    if rawTokenIds.size > args.maxTextLen then
      rawTokenIds.extract 0 args.maxTextLen
    else
      rawTokenIds
  saveTokenIds tokenIdsPath tokenIds
  let textTokenCount := tokenIds.size.toUInt64
  if textTokenCount == 0 then
    throw <| IO.userError "Tokenized prompt is empty."
  IO.println s!"Tokenized prompt length: {textTokenCount}"
  IO.println s!"Saved token IDs to {tokenIdsPath}"

  let languageId ←
    if lowerAsciiString args.language == "auto" then
      pure none
    else
      match lookupLanguageId? cfg args.language with
      | some lid => pure (some lid)
      | none =>
          let supported := String.intercalate ", " (cfg.talkerConfig.codecLanguageId.toList.map (·.fst))
          throw <| IO.userError s!"Unsupported language '{args.language}'. Supported: {supported}"

  let speakerEmbed ←
    match args.speakerEmbeddingPath, args.refAudioPath with
    | some p, _ =>
        loadSpeakerEmbedding? cfg (some p)
    | none, some refAudioPath =>
        let refAudioPath ← expandHome refAudioPath
        let emb ← extractSpeakerEmbeddingFromAudio
          cfg model bridgeCfg refAudioPath speakerMelPath speakerMelFramesPath
        IO.println s!"Extracted speaker embedding from reference audio: {refAudioPath}"
        pure (some emb)
    | none, none =>
        pure none
  let ((⟨talkerSeq, talkerInputs⟩), ttsPadEmbed) ←
    buildTalkerInputsEquivalent cfg model tokenIds languageId thinkMode speakerEmbed
  IO.println s!"Built conditioned talker inputs, seq={talkerSeq}"

  let maxFrames : UInt64 := args.maxFrames.toUInt64
  if args.trueStreaming then
    let canUseLeanStreamingDecode : Bool :=
      !args.skipDecode && !args.decodeViaPython && cfg.talkerConfig.numCodeGroups == 16
    let mut usePythonDecodeAfterStream : Bool := !args.skipDecode && !canUseLeanStreamingDecode

    ensureParentDir codesPath
    let speechDecoder? ←
      if canUseLeanStreamingDecode then
        try
          let speechTokenizerDir ←
            match args.speechTokenizerDir with
            | some d => expandHome d
            | none => pure s!"{modelDir}/speech_tokenizer"
          IO.println s!"Loading Lean speech-tokenizer decoder from {speechTokenizerDir}..."
          let dec ← SpeechTokenizer12HzDecoder.loadFromDir speechTokenizerDir targetDevice
          ensureParentDir wavPath
          Tensor.wavBegin wavPath dec.outputSampleRate
          pure (some dec)
        catch e =>
          usePythonDecodeAfterStream := !args.skipDecode
          IO.println
            s!"Lean true-streaming decode unavailable ({e.toString}); falling back to Python decode bridge after generation."
          pure none
      else
        pure none

    if usePythonDecodeAfterStream then
      if args.decodeViaPython then
        IO.println "Using Python decode bridge for true streaming (--decode-via-python)."
      else if cfg.talkerConfig.numCodeGroups != 16 then
        IO.println
          s!"Lean true-streaming decode supports 16 code groups; got {cfg.talkerConfig.numCodeGroups}. Falling back to Python decode bridge after generation."
      else
        pure ()

    let callbacks : Qwen3TTSForConditionalGeneration.StreamingCallbacks (device := targetDevice) := {
      onAudioChunk := fun chunk => Tensor.wavAppend chunk.2 wavPath
    }
    let streamOpts : Qwen3TTSForConditionalGeneration.StreamingOptions (device := targetDevice) cfg := {
      maxFrames
      minNewTokens := 2
      temperature := args.temperature
      topK := args.topK.toUInt64
      topP := args.topP
      subtalkerTemperature := args.subtalkerTemperature
      subtalkerTopK := args.subtalkerTopK.toUInt64
      subtalkerTopP := args.subtalkerTopP
      repetitionPenalty := args.repetitionPenalty
      suppressTail := args.suppressTail.toUInt64
      trailingTextHidden := some ⟨1, ttsPadEmbed⟩
      ttsPadEmbed := some ttsPadEmbed
      subtalkerTemperaturesByGroup := args.subtalkerTemperaturesByGroup
      subtalkerTopKsByGroup := subtalkerTopKsByGroup
      subtalkerTopPsByGroup := args.subtalkerTopPsByGroup
      decodeChunkSize := args.streamChunkFrames.toUInt64
      decodeLeftContext := args.decodeLeftContext.toUInt64
      emitEosFrame := false
    }
    let streamOut ←
      model.streamFromTalkerInputs talkerInputs streamOpts speechDecoder? callbacks

    let codeLen := streamOut.codeRows.size.toUInt64
    let codesText := formatCodesMatrix streamOut.codeRows
    IO.FS.writeFile codesPath codesText
    IO.println s!"Generated {codeLen} codec frames."
    IO.println s!"Saved codec codes to {codesPath}"

    if speechDecoder?.isSome then
      Tensor.wavFinalize wavPath
      IO.println s!"Saved waveform to {wavPath} (Lean true streaming decode)"
    else if usePythonDecodeAfterStream then
      decodeCodesToWav bridgeCfg modelDir cfg.talkerConfig codesPath wavPath
      IO.println s!"Saved waveform to {wavPath} (Python decode bridge after true streaming)"
    else
      IO.println "Skipping waveform decode (--skip-decode enabled)."
  else
    let out ← TalkerForConditionalGeneration.generateCodesWithLengths
      cfg.talkerConfig model.talker talkerInputs maxFrames
      2
      args.temperature args.topK.toUInt64 args.topP
      args.subtalkerTemperature args.subtalkerTopK.toUInt64 args.subtalkerTopP
      args.repetitionPenalty args.suppressTail.toUInt64
      (some ⟨1, ttsPadEmbed⟩) (some ttsPadEmbed)
      args.subtalkerTemperaturesByGroup subtalkerTopKsByGroup args.subtalkerTopPsByGroup
    let codeLen := out.lengths.getD 0 maxFrames

    let codes3 : Tensor { shape := #[1, maxFrames, cfg.talkerConfig.numCodeGroups], dtype := .Int64, device := targetDevice } := qwen35.typedSlice out.codes 0 0 1
    let codes2 : Tensor { shape := #[maxFrames, cfg.talkerConfig.numCodeGroups], dtype := .Int64, device := targetDevice } :=
      Tensor.reshapeChecked codes3 #[maxFrames, cfg.talkerConfig.numCodeGroups]
    let trimmed : Tensor { shape := #[codeLen, cfg.talkerConfig.numCodeGroups], dtype := .Int64, device := targetDevice } := qwen35.typedSlice codes2 0 0 codeLen
    let flat : Tensor { shape := #[codeLen * cfg.talkerConfig.numCodeGroups], dtype := .Int64, device := targetDevice } :=
      Tensor.reshapeChecked trimmed #[codeLen * cfg.talkerConfig.numCodeGroups]
    let codeVals ← Tensor.toUInt64Array flat

    ensureParentDir codesPath
    let codesText := formatCodesRows codeVals cfg.talkerConfig.numCodeGroups
    IO.FS.writeFile codesPath codesText
    IO.println s!"Generated {codeLen} codec frames."
    IO.println s!"Saved codec codes to {codesPath}"

    if args.skipDecode then
      IO.println "Skipping waveform decode (--skip-decode enabled)."
    else
      ensureParentDir wavPath
      if args.decodeViaPython || cfg.talkerConfig.numCodeGroups != 16 then
        if !args.decodeViaPython && cfg.talkerConfig.numCodeGroups != 16 then
          IO.println
            s!"Lean speech-tokenizer decoder supports 16 code groups; got {cfg.talkerConfig.numCodeGroups}. Falling back to Python decode bridge."
        decodeCodesToWav bridgeCfg modelDir cfg.talkerConfig codesPath wavPath
        IO.println s!"Saved waveform to {wavPath} (Python decode bridge)"
      else
        try
          let speechTokenizerDir ←
            match args.speechTokenizerDir with
            | some d => expandHome d
            | none => pure s!"{modelDir}/speech_tokenizer"
          IO.println s!"Loading Lean speech-tokenizer decoder from {speechTokenizerDir}..."
          let speechDecoder ← SpeechTokenizer12HzDecoder.loadFromDir speechTokenizerDir targetDevice
          let trimmed16 : Tensor { shape := #[codeLen, 16], dtype := .Int64, device := targetDevice } := Tensor.reshapeChecked trimmed #[codeLen, 16]
          if args.streamingDecode then
            speechDecoder.decodeFrameMajorChunkedToWav
              trimmed16 wavPath args.decodeChunkSize.toUInt64 args.decodeLeftContext.toUInt64
            IO.println s!"Saved waveform to {wavPath} (Lean decoder, chunked streaming)"
          else
            speechDecoder.decodeFrameMajorToWav trimmed16 wavPath
            IO.println s!"Saved waveform to {wavPath} (Lean decoder)"
        catch e =>
          IO.println
            s!"Lean speech-tokenizer decode unavailable ({e.toString}); falling back to Python decode bridge."
          decodeCodesToWav bridgeCfg modelDir cfg.talkerConfig codesPath wavPath
          IO.println s!"Saved waveform to {wavPath} (Python decode bridge)"

def _root_.main (rawArgs : List String) : IO UInt32 := do
  match validateRawArgs rawArgs with
  | .error err =>
    throw <| IO.userError err
  | .ok _ =>
    pure ()
  let args := parseArgs rawArgs
  if args.help then
    printUsage
    return 0
  runEndToEnd args
  return 0

end Examples.Qwen3TTS
