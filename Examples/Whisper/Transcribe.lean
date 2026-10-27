import Tyr.Model.Whisper

open torch.whisper

namespace Examples.Whisper

structure Args where
  modelDir : String := "weights/whisper-base.en"
  wavPath : String := "MLKDream.wav"
  language : String := "en"
  maxNewTokens : UInt64 := 0
  noTimestamps : Bool := true
  beamSize : UInt64 := 1
  bestOf : UInt64 := 1
  temperature : Float := 0.0
  temperatureInc : Float := 0.2
  noFallback : Bool := true
  topK : UInt64 := 0
  topP : Float := 1.0
  logprobThreshold : Float := -1.0
  noSpeechThreshold : Float := 0.6
  compressionRatioThreshold : Float := 2.4
  noContext : Bool := false
  maxContextTokens : UInt64 := 0
  chunkOverlapSec : Float := 2.0
  deriving Inhabited

private def parseNatArg (name : String) (v : String) : IO UInt64 := do
  match v.toNat? with
  | some n => pure n.toUInt64
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

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

private def parseFloatArg (name : String) (v : String) : IO Float := do
  match parseFloatLit? v with
  | some x => pure x
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private def toDecodeOptions (args : Args) : WhisperDecodeOptions := {
  beamSize := args.beamSize
  bestOf := args.bestOf
  temperature := args.temperature
  temperatureInc := args.temperatureInc
  topK := args.topK
  topP := args.topP
  logprobThreshold := args.logprobThreshold
  noSpeechThreshold := args.noSpeechThreshold
  compressionRatioThreshold := args.compressionRatioThreshold
  conditionOnPreviousText := !args.noContext
  maxContextTokens := args.maxContextTokens
  chunkOverlapSeconds := args.chunkOverlapSec
  noFallback := args.noFallback
}

private partial def parseArgsLoop (xs : List String) (acc : Args) : IO Args := do
  match xs with
  | [] => pure acc
  | "--model-dir" :: v :: rest =>
      parseArgsLoop rest { acc with modelDir := v }
  | "--source" :: v :: rest =>
      parseArgsLoop rest { acc with modelDir := v }
  | "--wav-path" :: v :: rest =>
      parseArgsLoop rest { acc with wavPath := v }
  | "--language" :: v :: rest =>
      parseArgsLoop rest { acc with language := v }
  | "--max-new-tokens" :: v :: rest =>
      parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--beam-size" :: v :: rest =>
      parseArgsLoop rest { acc with beamSize := (← parseNatArg "--beam-size" v) }
  | "--best-of" :: v :: rest =>
      parseArgsLoop rest { acc with bestOf := (← parseNatArg "--best-of" v) }
  | "--temperature" :: v :: rest =>
      parseArgsLoop rest { acc with temperature := (← parseFloatArg "--temperature" v) }
  | "--temperature-inc" :: v :: rest =>
      parseArgsLoop rest { acc with temperatureInc := (← parseFloatArg "--temperature-inc" v) }
  | "--no-fallback" :: rest =>
      parseArgsLoop rest { acc with noFallback := true }
  | "--fallback" :: rest =>
      parseArgsLoop rest { acc with noFallback := false }
  | "--top-k" :: v :: rest =>
      parseArgsLoop rest { acc with topK := (← parseNatArg "--top-k" v) }
  | "--top-p" :: v :: rest =>
      parseArgsLoop rest { acc with topP := (← parseFloatArg "--top-p" v) }
  | "--logprob-thold" :: v :: rest =>
      parseArgsLoop rest { acc with logprobThreshold := (← parseFloatArg "--logprob-thold" v) }
  | "--no-speech-thold" :: v :: rest =>
      parseArgsLoop rest { acc with noSpeechThreshold := (← parseFloatArg "--no-speech-thold" v) }
  | "--compression-ratio-thold" :: v :: rest =>
      parseArgsLoop rest { acc with compressionRatioThreshold := (← parseFloatArg "--compression-ratio-thold" v) }
  | "--no-context" :: rest =>
      parseArgsLoop rest { acc with noContext := true }
  | "--max-context" :: v :: rest =>
      parseArgsLoop rest { acc with maxContextTokens := (← parseNatArg "--max-context" v) }
  | "--chunk-overlap-sec" :: v :: rest =>
      parseArgsLoop rest { acc with chunkOverlapSec := (← parseFloatArg "--chunk-overlap-sec" v) }
  | "--no-timestamps" :: rest =>
      parseArgsLoop rest { acc with noTimestamps := true }
  | "--timestamps" :: rest =>
      parseArgsLoop rest { acc with noTimestamps := false }
  | "--help" :: _ =>
      IO.println "Usage: lake exe WhisperTranscribe [options]"
      IO.println "  --model-dir <path>       Whisper model directory (HF layout with config/tokenizer/safetensors)"
      IO.println "  --source <path>          Alias for --model-dir"
      IO.println "  --wav-path <path>        WAV file path"
      IO.println "  --language <code>        Language code/name (default: en)"
      IO.println "  --max-new-tokens <n>     Max generated decoder tokens (default: 0 = model max)"
      IO.println "  --beam-size <n>          Beam size at temperature=0 (default: 1)"
      IO.println "  --best-of <n>            Best-of samples at temperature>0 (default: 1)"
      IO.println "  --temperature <x>        Initial decoding temperature (default: 0.0)"
      IO.println "  --temperature-inc <x>    Temperature fallback increment (default: 0.2)"
      IO.println "  --no-fallback            Disable temperature fallback retries (default: true)"
      IO.println "  --fallback               Enable temperature fallback retries"
      IO.println "  --top-k <n>              Top-k filter for stochastic passes (default: 0 = disabled)"
      IO.println "  --top-p <x>              Top-p filter for stochastic passes (default: 1.0)"
      IO.println "  --logprob-thold <x>      Avg logprob fallback threshold (default: -1.0)"
      IO.println "  --no-speech-thold <x>    No-speech probability threshold (default: 0.6)"
      IO.println "  --compression-ratio-thold <x>  Repetition fallback threshold (default: 2.4)"
      IO.println "  --no-context             Disable rolling prompt context carry across chunks"
      IO.println "  --max-context <n>        Max carried context tokens (default: auto)"
      IO.println "  --chunk-overlap-sec <x>  Overlap between adjacent audio chunks in seconds (default: 2.0)"
      IO.println "  --no-timestamps          Use <|notimestamps|> decoding prompt token (default: true)"
      IO.println "  --timestamps             Enable timestamp-token decoding (experimental)"
      pure acc
  | x :: _ =>
      throw <| IO.userError s!"Unknown argument: {x}"

private def parseArgs (xs : List String) : IO Args :=
  parseArgsLoop xs {}

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  if argv.contains "--help" then return 0
  let bundle ← loadFromPretrainedDir args.modelDir
  let out ←
    transcribeWav
      bundle.model
      bundle.tok
      bundle.preprocessor
      args.wavPath
      args.language
      args.maxNewTokens
      args.noTimestamps
      (toDecodeOptions args)
  IO.println s!"LANG={out.language}"
  IO.println "TEXT_BEGIN"
  IO.println out.text
  IO.println "TEXT_END"
  pure 0

end Examples.Whisper

def main (argv : List String) : IO UInt32 :=
  Examples.Whisper.runMain argv
