/-
  Tyr/Model/SileroVAD/Utils.lean

  Lean4 port of the Silero-VAD utility layer:
  - speech timestamp extraction from chunk probabilities
  - streaming iterator boundaries
  - audio chunk collect/drop helpers
-/
import Tyr.Model.SileroVAD.Model
import Tyr.Model.Qwen3ASR.Frontend

namespace torch.silerovad

variable {device : Device}
structure SpeechTimestamp where
  start : Nat
  endPos : Nat
  deriving Repr, Inhabited, BEq

structure SpeechTimestampSeconds where
  start : Float
  endPos : Float
  deriving Repr, Inhabited

structure TimestampConfig where
  threshold : Float := 0.5
  samplingRate : UInt64 := 16000
  minSpeechDurationMs : UInt64 := 250
  maxSpeechDurationS : Float := 1.0e9
  minSilenceDurationMs : UInt64 := 100
  speechPadMs : UInt64 := 30
  negThreshold : Option Float := none
  minSilenceAtMaxSpeechMs : UInt64 := 98
  useMaxPossibleSilenceAtMaxSpeech : Bool := true
  deriving Repr, Inhabited

private def msToSamples (samplingRate ms : UInt64) : Nat :=
  (samplingRate.toNat * ms.toNat) / 1000

private def safeSub (a b : Nat) : Nat :=
  if a >= b then a - b else 0

private def fmax (a b : Float) : Float :=
  if a > b then a else b

private def roundToResolution (x : Float) (digits : Nat) : Float :=
  let scaleNat := Nat.pow 10 digits
  let scale := scaleNat.toFloat
  if scale <= 0.0 then
    x
  else
    (((x * scale) + 0.5).toUInt64.toFloat) / scale

/-- Port of Silero `get_speech_timestamps` core segmentation over chunk probabilities. -/
def timestampsFromProbabilities
    (speechProbs : Array Float)
    (audioLengthSamples : Nat)
    (cfg : TimestampConfig := {})
    : IO (Array SpeechTimestamp) := do
  if cfg.samplingRate != 8000 && cfg.samplingRate != 16000 then
    throw <| IO.userError
      s!"SileroVAD timestamp extraction supports 8000 or 16000 Hz, got {cfg.samplingRate}"

  let windowSize : Nat := if cfg.samplingRate == 16000 then 512 else 256
  let minSpeechSamples := msToSamples cfg.samplingRate cfg.minSpeechDurationMs
  let speechPadSamples := msToSamples cfg.samplingRate cfg.speechPadMs
  let minSilenceSamples := msToSamples cfg.samplingRate cfg.minSilenceDurationMs
  let minSilenceSamplesAtMaxSpeech := msToSamples cfg.samplingRate cfg.minSilenceAtMaxSpeechMs
  let maxSpeechSamplesFloat : Float :=
    cfg.samplingRate.toFloat * cfg.maxSpeechDurationS -
      windowSize.toFloat -
      2.0 * speechPadSamples.toFloat

  let negThreshold := cfg.negThreshold.getD (fmax (cfg.threshold - 0.15) 0.01)

  let mut triggered : Bool := false
  let mut speeches : Array SpeechTimestamp := #[]
  let mut currentStart : Nat := 0
  let mut tempEnd : Nat := 0
  let mut prevEnd : Nat := 0
  let mut nextStart : Nat := 0
  let mut possibleEnds : Array (Nat × Nat) := #[]

  for i in [:speechProbs.size] do
    let speechProb := speechProbs[i]!
    let curSample := windowSize * i

    if speechProb >= cfg.threshold && tempEnd != 0 then
      let silDur := curSample - tempEnd
      if silDur > minSilenceSamplesAtMaxSpeech then
        possibleEnds := possibleEnds.push (tempEnd, silDur)
      tempEnd := 0
      if nextStart < prevEnd then
        nextStart := curSample

    if speechProb >= cfg.threshold && !triggered then
      triggered := true
      currentStart := curSample
      continue

    if triggered && (curSample - currentStart).toFloat > maxSpeechSamplesFloat then
      if cfg.useMaxPossibleSilenceAtMaxSpeech && !possibleEnds.isEmpty then
        let best := possibleEnds.foldl
          (fun acc x => if x.2 > acc.2 then x else acc)
          (possibleEnds[0]!)
        prevEnd := best.1
        let dur := best.2
        speeches := speeches.push { start := currentStart, endPos := prevEnd }
        nextStart := prevEnd + dur
        if nextStart < prevEnd + curSample then
          currentStart := nextStart
        else
          triggered := false
        prevEnd := 0
        nextStart := 0
        tempEnd := 0
        possibleEnds := #[]
      else
        if prevEnd != 0 then
          speeches := speeches.push { start := currentStart, endPos := prevEnd }
          if nextStart < prevEnd then
            triggered := false
          else
            currentStart := nextStart
          prevEnd := 0
          nextStart := 0
          tempEnd := 0
          possibleEnds := #[]
        else
          speeches := speeches.push { start := currentStart, endPos := curSample }
          prevEnd := 0
          nextStart := 0
          tempEnd := 0
          triggered := false
          possibleEnds := #[]
          continue

    if speechProb < negThreshold && triggered then
      if tempEnd == 0 then
        tempEnd := curSample
      let silDurNow := curSample - tempEnd

      if !cfg.useMaxPossibleSilenceAtMaxSpeech && silDurNow > minSilenceSamplesAtMaxSpeech then
        prevEnd := tempEnd

      if silDurNow < minSilenceSamples then
        continue
      else
        let endPos := tempEnd
        if endPos - currentStart > minSpeechSamples then
          speeches := speeches.push { start := currentStart, endPos := endPos }
        prevEnd := 0
        nextStart := 0
        tempEnd := 0
        triggered := false
        possibleEnds := #[]
        continue

  if triggered && (audioLengthSamples - currentStart) > minSpeechSamples then
    speeches := speeches.push { start := currentStart, endPos := audioLengthSamples }

  let adjusted := Id.run do
    let mut out := speeches
    for i in [:out.size] do
      let cur := out[i]!
      let mut curStart := cur.start
      let mut curEnd := cur.endPos

      if i == 0 then
        curStart := safeSub curStart speechPadSamples

      if i + 1 < out.size then
        let nxt := out[i + 1]!
        let silenceDuration := safeSub nxt.start curEnd
        if silenceDuration < 2 * speechPadSamples then
          curEnd := curEnd + silenceDuration / 2
          let nxtStart := safeSub nxt.start (silenceDuration / 2)
          out := out.set! (i + 1) { nxt with start := nxtStart }
        else
          curEnd := Nat.min audioLengthSamples (curEnd + speechPadSamples)
          let nxtStart := safeSub nxt.start speechPadSamples
          out := out.set! (i + 1) { nxt with start := nxtStart }
        out := out.set! i { start := curStart, endPos := curEnd }
      else
        curEnd := Nat.min audioLengthSamples (curEnd + speechPadSamples)
        out := out.set! i { start := curStart, endPos := curEnd }
    out

  pure adjusted

def timestampsToSeconds
    (tss : Array SpeechTimestamp)
    (samplingRate : UInt64 := 16000)
    (timeResolution : Nat := 1)
    : Array SpeechTimestampSeconds :=
  let sr := if samplingRate == 0 then 1.0 else samplingRate.toFloat
  tss.map (fun ts => {
    start := roundToResolution (ts.start.toFloat / sr) timeResolution
    endPos := roundToResolution (ts.endPos.toFloat / sr) timeResolution
  })

/-- End-to-end Silero VAD timestamp extraction over raw waveform samples (`16kHz`). -/
def getSpeechTimestamps
    (audio : Array Float)
    (runtime : (SileroVADRuntime (device := device)))
    (cfg : TimestampConfig := {})
    : IO (Array SpeechTimestamp × (SileroVADRuntime (device := device))) := do
  if cfg.samplingRate != 16000 then
    throw <| IO.userError
      s!"SileroVAD runtime currently supports only 16000Hz inference, got {cfg.samplingRate}"
  let (probs, runtime') ← runtime.audioForward audio cfg.samplingRate
  let tss ← timestampsFromProbabilities probs audio.size cfg
  pure (tss, runtime')

def getSpeechTimestampsSeconds
    (audio : Array Float)
    (runtime : (SileroVADRuntime (device := device)))
    (cfg : TimestampConfig := {})
    (timeResolution : Nat := 1)
    : IO (Array SpeechTimestampSeconds × (SileroVADRuntime (device := device))) := do
  let (tss, runtime') ← getSpeechTimestamps audio runtime cfg
  pure (timestampsToSeconds tss cfg.samplingRate timeResolution, runtime')

inductive VADBoundary where
  | start (sample : Nat)
  | stop (sample : Nat)
  deriving Repr, Inhabited, BEq

/-- Streaming speech-boundary iterator mirroring Silero `VADIterator`. -/
structure VADIterator where
  runtime : SileroVADRuntime (device := device)
  threshold : Float := 0.5
  samplingRate : UInt64 := 16000
  minSilenceSamples : Nat := 1600
  speechPadSamples : Nat := 480
  triggered : Bool := false
  tempEnd : Nat := 0
  currentSample : Nat := 0

namespace VADIterator

def init
    (runtime : (SileroVADRuntime (device := device)))
    (threshold : Float := 0.5)
    (samplingRate : UInt64 := 16000)
    (minSilenceDurationMs : UInt64 := 100)
    (speechPadMs : UInt64 := 30) : IO (VADIterator (device := device)) := do
  if samplingRate != 16000 then
    throw <| IO.userError
      s!"SileroVAD iterator currently supports only 16000Hz, got {samplingRate}"
  pure {
    runtime := runtime.reset
    threshold
    samplingRate
    minSilenceSamples := msToSamples samplingRate minSilenceDurationMs
    speechPadSamples := msToSamples samplingRate speechPadMs
    triggered := false
    tempEnd := 0
    currentSample := 0
  }

def reset (it : (VADIterator (device := device))) : (VADIterator (device := device)) := {
  it with
  runtime := it.runtime.reset
  triggered := false
  tempEnd := 0
  currentSample := 0
}

def step (it : (VADIterator (device := device))) (chunk : Array Float) : IO (Option VADBoundary × (VADIterator (device := device))) := do
  let win := chunk.size
  let (speechProb, rt') ← it.runtime.step chunk it.samplingRate
  let curSample := it.currentSample + win
  let mut next := { it with runtime := rt', currentSample := curSample }

  if speechProb >= it.threshold && next.tempEnd != 0 then
    next := { next with tempEnd := 0 }

  if speechProb >= it.threshold && !next.triggered then
    let startPos := safeSub (safeSub curSample next.speechPadSamples) win
    next := { next with triggered := true }
    pure (some (.start startPos), next)
  else if speechProb < (it.threshold - 0.15) && next.triggered then
    if next.tempEnd == 0 then
      next := { next with tempEnd := curSample }
    if curSample - next.tempEnd < next.minSilenceSamples then
      pure (none, next)
    else
      let stopPos := safeSub (next.tempEnd + next.speechPadSamples) win
      next := { next with triggered := false, tempEnd := 0 }
      pure (some (.stop stopPos), next)
  else
    pure (none, next)

end VADIterator

def collectChunks (tss : Array SpeechTimestamp) (wav : Array Float) : Array Float := Id.run do
  let mut out : Array Float := #[]
  for ts in tss do
    let s := Nat.min ts.start wav.size
    let e := Nat.min ts.endPos wav.size
    if s < e then
      out := out ++ wav.extract s e
  out

def dropChunks (tss : Array SpeechTimestamp) (wav : Array Float) : Array Float := Id.run do
  let mut out : Array Float := #[]
  let mut curStart : Nat := 0
  for ts in tss do
    let s := Nat.min ts.start wav.size
    let e := Nat.min ts.endPos wav.size
    if curStart < s then
      out := out ++ wav.extract curStart s
    curStart := Nat.max curStart e
  if curStart < wav.size then
    out := out ++ wav.extract curStart wav.size
  out

/-- WAV-only audio loader (PCM16 WAV through existing Qwen3-ASR frontend parser). -/
def readAudio (path : String) (samplingRate : UInt64 := 16000) : IO (Array Float) := do
  let (srcRate, wav) ← qwen3asr.loadMonoPcm16Wav path
  if srcRate == samplingRate then
    pure wav
  else
    pure (qwen3asr.resampleLinear wav srcRate samplingRate)

end torch.silerovad
