/-
  Examples/Whisper/VoiceMode.lean

  Interactive voice mode for Whisper ASR in Tyr.
  Mirrors the hermes-agent CLI voice mode:

  - Records from the microphone at 16kHz mono via Apple AudioQueue
  - Silence detection with configurable RMS threshold and dip tolerance
  - Audio feedback: beep on recording start, double-beep on transcription
  - Continuous mode: auto-restarts recording after each utterance;
    exits after N consecutive no-speech detections
  - Prints transcribed text to stdout
-/
import Tyr.Model.Whisper
import Tyr.Audio.AppleInput
import Tyr.Audio.AppleOutput
import Tyr.Audio.FloatBuffer

open torch.whisper

namespace Examples.Whisper.VoiceMode

-- ============================================================================
-- Configuration
-- ============================================================================

structure Args where
  modelDir : String := "weights/whisper-base.en"
  language : String := "en"
  maxNewTokens : UInt64 := 128
  /-- RMS threshold for speech detection (float amplitude, not int16). -/
  speechRmsThreshold : Float := 0.008
  /-- Seconds of sustained silence to end recording. -/
  silenceDuration : Float := 3.0
  /-- Tolerance for brief dips below threshold during speech (seconds). -/
  dipTolerance : Float := 0.3
  /-- Minimum speech duration to confirm speech started (seconds). -/
  minSpeechDuration : Float := 0.3
  /-- Maximum recording duration (seconds). -/
  maxRecordingSeconds : Float := 120.0
  /-- Maximum wait for initial speech before giving up (seconds). -/
  maxWaitNoSpeech : Float := 15.0
  /-- Enable continuous mode: auto-restart after each utterance. -/
  continuous : Bool := false
  /-- Max consecutive no-speech detections before exiting continuous mode. -/
  maxNoSpeechExits : Nat := 3
  /-- Play audio beep feedback. -/
  beeps : Bool := true
  /-- Whisper decode options. -/
  beamSize : UInt64 := 1
  noTimestamps : Bool := true
  noFallback : Bool := true
  deriving Inhabited

-- ============================================================================
-- Arg parsing
-- ============================================================================

private def parseFloatLit? (s : String) : Option Float :=
  match s.splitOn "." with
  | [whole] =>
      whole.toNat?.map (·.toFloat)
  | [whole, frac] =>
      match whole.toNat?, frac.toNat? with
      | some w, some f =>
          let denom : Float := (Nat.pow 10 frac.length).toFloat
          some (Nat.toFloat w + Nat.toFloat f / denom)
      | _, _ => none
  | _ => none

private def parseFloatArg (name : String) (v : String) : IO Float :=
  match parseFloatLit? v with
  | some x => pure x
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private def parseNatArg (name : String) (v : String) : IO UInt64 :=
  match v.toNat? with
  | some n => pure n.toUInt64
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private partial def parseArgsLoop (xs : List String) (acc : Args) : IO Args := do
  match xs with
  | [] => pure acc
  | "--model-dir" :: v :: rest =>
      parseArgsLoop rest { acc with modelDir := v }
  | "--source" :: v :: rest =>
      parseArgsLoop rest { acc with modelDir := v }
  | "--language" :: v :: rest =>
      parseArgsLoop rest { acc with language := v }
  | "--max-new-tokens" :: v :: rest =>
      parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--rms-threshold" :: v :: rest =>
      parseArgsLoop rest { acc with speechRmsThreshold := (← parseFloatArg "--rms-threshold" v) }
  | "--silence-duration" :: v :: rest =>
      parseArgsLoop rest { acc with silenceDuration := (← parseFloatArg "--silence-duration" v) }
  | "--dip-tolerance" :: v :: rest =>
      parseArgsLoop rest { acc with dipTolerance := (← parseFloatArg "--dip-tolerance" v) }
  | "--min-speech-duration" :: v :: rest =>
      parseArgsLoop rest { acc with minSpeechDuration := (← parseFloatArg "--min-speech-duration" v) }
  | "--max-recording" :: v :: rest =>
      parseArgsLoop rest { acc with maxRecordingSeconds := (← parseFloatArg "--max-recording" v) }
  | "--max-wait" :: v :: rest =>
      parseArgsLoop rest { acc with maxWaitNoSpeech := (← parseFloatArg "--max-wait" v) }
  | "--continuous" :: rest =>
      parseArgsLoop rest { acc with continuous := true }
  | "--no-beeps" :: rest =>
      parseArgsLoop rest { acc with beeps := false }
  | "--beam-size" :: v :: rest =>
      parseArgsLoop rest { acc with beamSize := (← parseNatArg "--beam-size" v) }
  | "--max-no-speech-exits" :: v :: rest =>
      parseArgsLoop rest { acc with maxNoSpeechExits := (← parseNatArg "--max-no-speech-exits" v).toNat }
  | "--help" :: _ => do
      IO.println "Usage: lake exe WhisperVoiceMode [options]"
      IO.println ""
      IO.println "Interactive voice mode for Whisper ASR."
      IO.println ""
      IO.println "  --model-dir <path>         Whisper model directory (HF layout)"
      IO.println "  --source <path>            Alias for --model-dir"
      IO.println "  --language <code>          Language code (default: en)"
      IO.println "  --max-new-tokens <n>       Max decoder tokens (default: 128)"
      IO.println "  --rms-threshold <f>        Speech RMS threshold (default: 0.008)"
      IO.println "  --silence-duration <f>     Seconds of silence to stop (default: 3.0)"
      IO.println "  --dip-tolerance <f>        Brief silence tolerance (default: 0.3)"
      IO.println "  --min-speech-duration <f>  Min speech to activate (default: 0.3)"
      IO.println "  --max-recording <f>        Max recording seconds (default: 120.0)"
      IO.println "  --max-wait <f>             Max wait for speech start (default: 15.0)"
      IO.println "  --continuous               Auto-restart after each utterance"
      IO.println "  --no-beeps                 Disable audio feedback beeps"
      IO.println "  --beam-size <n>            Beam search width (default: 1)"
      IO.println "  --max-no-speech-exits <n>  Exit continuous after N silent rounds (default: 3)"
      pure acc
  | x :: _ =>
      throw <| IO.userError s!"Unknown argument: {x}"

private def parseArgs (xs : List String) : IO Args :=
  parseArgsLoop xs {}

-- ============================================================================
-- Audio helpers
-- ============================================================================

private def sampleRate : Nat := 16000

/-- Convert seconds to sample count at 16kHz. -/
private def toSamples (sec : Float) : Nat :=
  let n := ((sec * sampleRate.toFloat) + 0.5).toUInt64.toNat
  if n == 0 then 1 else n

-- ============================================================================
-- Silence-detecting recorder
-- ============================================================================

/-- Recording state machine. -/
inductive RecordPhase where
  | waitingForSpeech
  | recording
  | done
  deriving BEq, Repr

/-- Record from the microphone until silence is detected.
    Returns the recorded waveform as a FloatBuffer (unboxed), or empty if no speech.
    Audio data flows entirely through unboxed FloatBuffers — no per-sample boxing. -/
private def recordUntilSilence (args : Args) : IO FloatBuffer := do
  let readChunkSamples : Nat := sampleRate / 10  -- 100ms chunks
  let silenceThresholdSamples := toSamples args.silenceDuration
  let dipToleranceSamples := toSamples args.dipTolerance
  let minSpeechSamples := toSamples args.minSpeechDuration
  let maxSamples := toSamples args.maxRecordingSeconds
  let maxWaitSamples := toSamples args.maxWaitNoSpeech

  -- Pre-roll: keep ~500ms of audio before speech onset so the beginning isn't lost.
  let preRollSamples := toSamples 0.5

  let mut audioBuf ← FloatBuffer.mkEmpty (sampleRate * 10) -- ~10s initial capacity
  let mut phase := RecordPhase.waitingForSpeech
  let mut speechSamples : Nat := 0
  let mut silenceSamples : Nat := 0
  let mut dipSamples : Nat := 0
  let mut totalSamples : Nat := 0

  while phase != RecordPhase.done do
    let pcm ← Tyr.Audio.AppleInput.readBuffer readChunkSamples 250
    let pcmSize := pcm.size
    if pcmSize == 0 then
      continue

    let chunkRms ← pcm.rms
    let isSpeech := chunkRms >= args.speechRmsThreshold
    totalSamples := totalSamples + pcmSize

    match phase with
    | .waitingForSpeech =>
        -- Always accumulate (including pre-speech audio for context).
        audioBuf ← audioBuf.append pcm
        if isSpeech then
          speechSamples := speechSamples + pcmSize
          dipSamples := 0
          if speechSamples >= minSpeechSamples then
            phase := .recording
        else
          if speechSamples > 0 then
            -- Brief dip during speech onset — tolerate it.
            dipSamples := dipSamples + pcmSize
            if dipSamples > dipToleranceSamples then
              -- False start — reset, but keep pre-roll.
              speechSamples := 0
              dipSamples := 0
              audioBuf ← audioBuf.keepLast preRollSamples
          else
            -- No speech yet — slide the pre-roll window forward.
            audioBuf ← audioBuf.keepLast preRollSamples
          -- Check if we've waited too long with no speech.
          if totalSamples >= maxWaitSamples then
            phase := .done
    | .recording =>
        audioBuf ← audioBuf.append pcm
        if isSpeech then
          silenceSamples := 0
          dipSamples := 0
        else
          dipSamples := dipSamples + pcmSize
          if dipSamples > dipToleranceSamples then
            silenceSamples := silenceSamples + pcmSize
            if silenceSamples >= silenceThresholdSamples then
              phase := .done
        -- Check max recording length.
        if audioBuf.size >= maxSamples then
          phase := .done
    | .done => pure ()

  pure audioBuf

-- ============================================================================
-- Main loop
-- ============================================================================

/-- Erase current line and print text in-place (for live partial updates). -/
private def printInline (s : String) : IO Unit := do
  IO.print s!"\x1b[2K\r  {s}"
  (← IO.getStdout).flush

/-- Record with interleaved transcription — shows partial results while speaking.
    Runs Whisper on the accumulated audio every `transcribeIntervalSec` seconds
    of new audio, updating the terminal in-place. -/
private def runOnce
    {dtype : torch.DType} [torch.DType.RealFloating dtype]
    {device : torch.Device}
    {cfg : WhisperConfig}
    (model : WhisperForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (pre : torch.qwen3asr.PreprocessorConfig)
    (hMel : pre.featureSize = cfg.numMelBins)
    (args : Args)
    (decodeOpts : WhisperDecodeOptions)
    : IO Bool := do
  IO.print "Listening... "
  (← IO.getStdout).flush
  if args.beeps then
    Tyr.Audio.AppleOutput.beepStart

  let readChunkSamples : Nat := sampleRate / 10  -- 100ms chunks
  let silenceThresholdSamples := toSamples args.silenceDuration
  let dipToleranceSamples := toSamples args.dipTolerance
  let minSpeechSamples := toSamples args.minSpeechDuration
  let maxSamples := toSamples args.maxRecordingSeconds
  let maxWaitSamples := toSamples args.maxWaitNoSpeech
  let preRollSamples := toSamples 0.5
  -- Transcribe every ~2s of new audio while recording.
  let transcribeInterval := toSamples 2.0

  let mut audioBuf ← FloatBuffer.mkEmpty (sampleRate * 10)
  let mut phase := RecordPhase.waitingForSpeech
  let mut speechSamples : Nat := 0
  let mut silenceSamples : Nat := 0
  let mut dipSamples : Nat := 0
  let mut totalSamples : Nat := 0
  let mut lastTranscribeSize : Nat := 0
  let mut lastText := ""

  while phase != RecordPhase.done do
    let pcm ← Tyr.Audio.AppleInput.readBuffer readChunkSamples 250
    let pcmSize := pcm.size
    if pcmSize == 0 then
      continue

    let chunkRms ← pcm.rms
    let isSpeech := chunkRms >= args.speechRmsThreshold
    totalSamples := totalSamples + pcmSize

    match phase with
    | .waitingForSpeech =>
        audioBuf ← audioBuf.append pcm
        if isSpeech then
          speechSamples := speechSamples + pcmSize
          dipSamples := 0
          if speechSamples >= minSpeechSamples then
            IO.print "\x1b[2K\r"
            phase := .recording
        else
          if speechSamples > 0 then
            dipSamples := dipSamples + pcmSize
            if dipSamples > dipToleranceSamples then
              speechSamples := 0
              dipSamples := 0
              audioBuf ← audioBuf.keepLast preRollSamples
          else
            audioBuf ← audioBuf.keepLast preRollSamples
          if totalSamples >= maxWaitSamples then
            phase := .done
    | .recording =>
        audioBuf ← audioBuf.append pcm
        if isSpeech then
          silenceSamples := 0
          dipSamples := 0
        else
          dipSamples := dipSamples + pcmSize
          if dipSamples > dipToleranceSamples then
            silenceSamples := silenceSamples + pcmSize
            if silenceSamples >= silenceThresholdSamples then
              phase := .done
        if audioBuf.size >= maxSamples then
          phase := .done
        -- Periodic transcription: run Whisper on accumulated audio every ~2s.
        let newSinceLastTranscribe := audioBuf.size - lastTranscribeSize
        if newSinceLastTranscribe >= transcribeInterval && audioBuf.size >= toSamples 0.5 then
          let wav ← audioBuf.toArray
          lastTranscribeSize := audioBuf.size
          let result ← transcribeWaveform16k model tok pre hMel wav args.language args.maxNewTokens true decodeOpts
          let text := result.text.trimAscii.toString
          if !text.isEmpty then
            lastText := text
            printInline text
    | .done => pure ()

  -- Final transcription on the complete audio.
  if audioBuf.size == 0 || audioBuf.size < toSamples 0.3 then
    IO.println "\x1b[2K\r(no speech detected)"
    if args.beeps then
      Tyr.Audio.AppleOutput.beepNoSpeech
    return false

  let wav ← audioBuf.toArray
  let result ← transcribeWaveform16k model tok pre hMel wav args.language args.maxNewTokens true decodeOpts
  let text := result.text.trimAscii.toString

  if text.isEmpty then
    IO.println "\x1b[2K\r(empty transcription)"
    if args.beeps then
      Tyr.Audio.AppleOutput.beepNoSpeech
    return false
  else
    IO.println s!"\x1b[2K\r  {text}\n"
    if args.beeps then
      Tyr.Audio.AppleOutput.beepDone
    return true

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  if argv.contains "--help" then return 0

  IO.println "Loading Whisper model..."
  let bundle ← loadFromPretrainedDir args.modelDir
  IO.println s!"Model loaded ({args.modelDir})"
  IO.println s!"Language: {args.language}, continuous: {args.continuous}"
  IO.println ""

  let decodeOpts : WhisperDecodeOptions := {
    beamSize := args.beamSize
    noFallback := args.noFallback
  }

  Tyr.Audio.AppleInput.start sampleRate 1 100

  try
    if args.continuous then
      IO.println "Continuous voice mode. Speak naturally, silence ends each utterance."
      IO.println "Press Ctrl+C to exit."
      IO.println ""
      let mut noSpeechCount : Nat := 0
      let mut running := true
      while running do
        let hadSpeech ← runOnce bundle.model bundle.tok bundle.preprocessor bundle.melBinsMatch args decodeOpts
        if hadSpeech then
          noSpeechCount := 0
        else
          noSpeechCount := noSpeechCount + 1
          if noSpeechCount >= args.maxNoSpeechExits then
            IO.println s!"No speech detected {noSpeechCount} times in a row, exiting."
            running := false
    else
      IO.println "Single-shot voice mode. Speak, then wait for silence."
      IO.println ""
      let _ ← runOnce bundle.model bundle.tok bundle.preprocessor bundle.melBinsMatch args decodeOpts

    Tyr.Audio.AppleInput.stop
  catch e =>
    Tyr.Audio.AppleInput.stop
    throw e

  pure 0

end Examples.Whisper.VoiceMode

def main (argv : List String) : IO UInt32 :=
  Examples.Whisper.VoiceMode.runMain argv
