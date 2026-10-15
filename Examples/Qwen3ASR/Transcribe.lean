import Tyr.Hub
import Tyr.Model.Qwen3ASR
import Tyr.Model.Qwen3ASR.StreamModel
import Tyr.Tokenizer.Qwen3

open torch.qwen3asr

namespace Examples.Qwen3ASR

structure Args where
  source : String := "weights/qwen3-asr-0.6b"
  alignerSource : Option String := none
  revision : String := "main"
  cacheDir : String := torch.Hub.defaultCacheDir
  wavPath : String := "MLKDream.wav"
  language : Option String := none
  returnTimeStamps : Bool := false
  streamOutput : Bool := false
  streamChunkSec : Float := 2.0
  streamHopSec : Float := 0.5
  streamDecodeMode : StreamingDecodeMode := .rollingWindow
  maxInferenceBatchSize : Int := -1
  maxNewTokens : UInt64 := 1024
  deriving Inhabited

private def parseNatArg (name : String) (v : String) : IO UInt64 := do
  match v.toNat? with
  | some n => pure n.toUInt64
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private def parseIntArg (name : String) (v : String) : IO Int := do
  match v.toInt? with
  | some n => pure n
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

private def parseStreamDecodeModeArg (name : String) (v : String) : IO StreamingDecodeMode := do
  let mode := v.trimAscii.toString.toLower
  if mode == "rolling" || mode == "rolling-window" || mode == "rolling_window" then
    pure .rollingWindow
  else if mode == "full" || mode == "full-accumulation" || mode == "full_accumulation" then
    pure .fullAccumulation
  else
    throw <| IO.userError s!"Invalid {name}: {v} (expected rolling|full)"

private partial def parseArgsLoop (xs : List String) (acc : Args) : IO Args := do
  match xs with
  | [] => pure acc
  | "--source" :: v :: rest =>
      parseArgsLoop rest { acc with source := v }
  | "--aligner-source" :: v :: rest =>
      parseArgsLoop rest { acc with alignerSource := some v }
  | "--model-dir" :: v :: rest =>
      parseArgsLoop rest { acc with source := v }
  | "--revision" :: v :: rest =>
      parseArgsLoop rest { acc with revision := v }
  | "--cache-dir" :: v :: rest =>
      parseArgsLoop rest { acc with cacheDir := v }
  | "--wav-path" :: v :: rest =>
      parseArgsLoop rest { acc with wavPath := v }
  | "--language" :: v :: rest =>
      parseArgsLoop rest { acc with language := some v }
  | "--return-timestamps" :: rest =>
      parseArgsLoop rest { acc with returnTimeStamps := true }
  | "--stream-output" :: rest =>
      parseArgsLoop rest { acc with streamOutput := true }
  | "--stream-chunk-sec" :: v :: rest =>
      parseArgsLoop rest { acc with streamChunkSec := (← parseFloatArg "--stream-chunk-sec" v) }
  | "--stream-hop-sec" :: v :: rest =>
      parseArgsLoop rest { acc with streamHopSec := (← parseFloatArg "--stream-hop-sec" v) }
  | "--stream-decode-mode" :: v :: rest =>
      parseArgsLoop rest { acc with streamDecodeMode := (← parseStreamDecodeModeArg "--stream-decode-mode" v) }
  | "--max-inference-batch-size" :: v :: rest =>
      parseArgsLoop rest { acc with maxInferenceBatchSize := (← parseIntArg "--max-inference-batch-size" v) }
  | "--max-new-tokens" :: v :: rest =>
      parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--help" :: _ =>
      IO.println "Usage: lake exe Qwen3ASRTranscribe [options]"
      IO.println "  --source <path-or-repo>  Local model dir or HF repo id"
      IO.println "  --aligner-source <src>   Optional forced-aligner model dir/repo id"
      IO.println "  --model-dir <path>       Alias for --source (backward compatible)"
      IO.println "  --revision <rev>         HF revision/branch/tag (default: main)"
      IO.println "  --cache-dir <path>       Local cache for downloaded files"
      IO.println "  --wav-path <path>        WAV path to transcribe"
      IO.println "  --language <name>        Optional forced language (e.g. English)"
      IO.println "  --return-timestamps      Enable forced-alignment timestamps"
      IO.println "  --stream-output          Show whisper.cpp-style inline streaming updates while processing WAV"
      IO.println "  --stream-chunk-sec <f>   Streaming decode window seconds (default: 2.0)"
      IO.println "  --stream-hop-sec <f>     Streaming hop size seconds (default: 0.5)"
      IO.println "  --stream-decode-mode <m> Streaming decode mode: rolling|full (default: rolling)"
      IO.println "  --max-inference-batch-size <n>  Batch chunking control (-1 = unbounded)"
      IO.println "  --max-new-tokens <n>     Greedy decode max new tokens"
      IO.println "Example: lake exe Qwen3ASRTranscribe --source Qwen/Qwen3-ASR-1.7B --wav-path MLKDream.wav"
      throw <| IO.userError ""
  | x :: _ =>
      throw <| IO.userError s!"Unknown argument: {x}"

private def parseArgs (xs : List String) : IO Args :=
  parseArgsLoop xs {}

private def toSamples (sec : Float) : Nat :=
  let n := ((sec * 16000.0) + 0.5).toUInt64.toNat
  if n == 0 then 1 else n

private def uiPrintInline (s : String) : IO Unit := do
  IO.print "\x1b[2K\r"
  IO.print s
  (← IO.getStdout).flush

private def transcribeWavStreamed
    {device : torch.Device}
    {dtype : torch.ExactDType}
    (modelDir : String)
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype.val) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (pre : PreprocessorConfig)
    (wavPath : String)
    (language : Option String)
    (maxNewTokens : UInt64)
    (chunkSec : Float)
    (hopSec : Float)
    (decodeMode : StreamingDecodeMode)
    : IO ASRTranscription := do
  if chunkSec <= 0.0 || hopSec <= 0.0 then
    throw <| IO.userError "stream-chunk-sec and stream-hop-sec must be > 0"
  let wav16k ← normalizeAudioTo16kFromWav wavPath
  let maxSamples := PreprocessorConfig.expectedSampleCount pre
  let effectiveDecodeMode ←
    if decodeMode == .fullAccumulation && wav16k.size.toUInt64 > maxSamples then do
      let maxSec := maxSamples.toFloat / 16000.0
      IO.eprintln
        s!"stream-decode-mode=full exceeds model maxSeconds={maxSec} for this file; switching to rolling mode."
      pure .rollingWindow
    else
      pure decodeMode
  let sm : StreamModel := {
    device := device
    dtype := dtype
    cfg := cfg
    model := model
    tok := tok
    preprocessor := pre
    modelDir := modelDir
  }
  let mut ss ← newSession sm
    (chunkSec := chunkSec)
    (hopSec := hopSec)
    (decodeMode := effectiveDecodeMode)
    (language := language)
  let hopSamples := toSamples hopSec
  let mut off : Nat := 0
  let mut latestFull := ""
  let mut lastPrinted := ""
  let mut printedInline := false
  while off < wav16k.size do
    let hi := Nat.min wav16k.size (off + hopSamples)
    let chunk := wav16k.extract off hi
    let (ssNext, step) ← pushAudio sm ss chunk (maxNewTokens := maxNewTokens)
    ss := ssNext
    if step.didDecode then
      latestFull := step.fullText
      let view := step.fullText.trimAscii.toString
      if view != lastPrinted then
        uiPrintInline view
        lastPrinted := view
        printedInline := true
    off := hi
  let (ssFinal, finalStep) ← flush sm ss (maxNewTokens := maxNewTokens)
  ss := ssFinal
  if finalStep.didDecode then
    latestFull := finalStep.fullText
    let view := finalStep.fullText.trimAscii.toString
    if view != lastPrinted then
      uiPrintInline view
      lastPrinted := view
      printedInline := true

  if printedInline then
    IO.println ""

  let text :=
    let t := latestFull.trimAscii.toString
    if t.isEmpty then
      ss.asrState.text.trimAscii.toString
    else
      t
  pure {
    language := ss.asrState.language
    text := text
  }

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  let modelDir ← hub.resolvePretrainedDir args.source {
    revision := args.revision
    cacheDir := args.cacheDir
    includeTokenizer := true
    includePreprocessor := true
  }
  IO.println s!"Resolved model dir: {modelDir}"
  let cfg ← Qwen3ASRConfig.loadFromPretrainedDir modelDir
  let tok ← tokenizer.qwen3.loadTokenizer modelDir
  let pre ← PreprocessorConfig.loadFromPretrainedDir modelDir
  let ⟨_device, dtype, model⟩ ← Qwen3ASRForConditionalGeneration.loadSharded modelDir cfg
  IO.println s!"Qwen3-ASR target device: {repr model.thinker.lmHead.device}"
  let forcedAligner ←
    match args.alignerSource with
    | some src =>
      pure (some (← Qwen3ForcedAligner.loadFromPretrained src args.revision args.cacheDir))
    | none =>
      pure none
  let out ←
    if args.streamOutput && !args.returnTimeStamps && forcedAligner.isNone then
      transcribeWavStreamed
        (dtype := dtype)
        modelDir
        model
        tok
        pre
        args.wavPath
        args.language
        args.maxNewTokens
        args.streamChunkSec
        args.streamHopSec
        args.streamDecodeMode
    else
      if args.streamOutput && (args.returnTimeStamps || forcedAligner.isSome) then
        IO.eprintln
          "stream-output currently supports plain transcription only (no timestamps/aligner); falling back to offline mode."
      let outs ← model.transcribeWavs
        tok
        pre
        #[args.wavPath]
        (forcedAligner := forcedAligner)
        (contexts := #[""])
        (languages := #[args.language])
        (maxInferenceBatchSize := args.maxInferenceBatchSize)
        (returnTimeStamps := args.returnTimeStamps)
        (maxNewTokens := args.maxNewTokens)
      pure (outs.getD 0 default)
  IO.println s!"LANG={out.language}"
  IO.println "TEXT_BEGIN"
  IO.println out.text
  IO.println "TEXT_END"
  if args.returnTimeStamps then
    IO.println "TIMESTAMPS_BEGIN"
    match out.timeStamps with
    | some ts =>
      for it in ts.items do
        IO.println s!"{it.text}\t{it.startTime}\t{it.endTime}"
    | none =>
      IO.println ""
    IO.println "TIMESTAMPS_END"
  pure 0

end Examples.Qwen3ASR

def main (argv : List String) : IO UInt32 :=
  Examples.Qwen3ASR.runMain argv
