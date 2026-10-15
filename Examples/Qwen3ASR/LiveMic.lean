import Tyr.Model.Qwen3ASR
import Tyr.Tokenizer.Qwen3
import Tyr.Audio.AppleInput

namespace Examples.Qwen3ASR

open torch.qwen3asr

inductive UseCase where
  | generic
  | tv
  | conversation
  deriving Inhabited, Repr, BEq

private def UseCase.toString : UseCase → String
  | .generic => "generic"
  | .tv => "tv"
  | .conversation => "conversation"

structure Args where
  source : String := "weights/qwen3-asr-0.6b"
  revision : String := "main"
  cacheDir : String := "~/.cache/huggingface/tyr-models"
  language : Option String := none
  context : String := ""
  maxNewTokens : UInt64 := 128
  chunkSec : Float := 2.0
  hopSec : Float := 0.5
  keepSec : Float := 0.2
  runSec : Float := 30.0
  commitEvery : Nat := 0
  promptTokens : Nat := 96
  useCase : UseCase := .generic
  rollbackChars : Nat := 12
  silenceSec : Float := 0.8
  minSpeechSec : Float := 0.35
  speechRmsThreshold : Float := 0.008
  emitPartials : Bool := false
  prettyUI : Bool := true
  simpleOutput : Bool := false
  deriving Inhabited

private def parseNatArg (name : String) (v : String) : IO UInt64 :=
  match v.toNat? with
  | some n => pure n.toUInt64
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private def trimWhitespace (s : String) : String :=
  let xs := s.toList
  let left := xs.dropWhile Char.isWhitespace
  let right := left.reverse.dropWhile Char.isWhitespace
  String.ofList right.reverse

private def parseUseCaseArg (v : String) : IO UseCase := do
  let s := (trimWhitespace v).toLower
  if s = "generic" then
    pure .generic
  else if s = "tv" || s = "broadcast" then
    pure .tv
  else if s = "conversation" || s = "voice" || s = "dialog" || s = "dialogue" then
    pure .conversation
  else
    throw <| IO.userError s!"Invalid --use-case: {v} (expected generic|tv|conversation)"

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

private def parseFloatArg (name : String) (v : String) : IO Float :=
  match parseFloatLit? v with
  | some x => pure x
  | none => throw <| IO.userError s!"Invalid {name}: {v}"

private partial def parseArgsLoop (xs : List String) (acc : Args) : IO Args := do
  match xs with
  | [] => pure acc
  | "--source" :: v :: rest => parseArgsLoop rest { acc with source := v }
  | "--model-dir" :: v :: rest => parseArgsLoop rest { acc with source := v }
  | "--revision" :: v :: rest => parseArgsLoop rest { acc with revision := v }
  | "--cache-dir" :: v :: rest => parseArgsLoop rest { acc with cacheDir := v }
  | "--language" :: v :: rest => parseArgsLoop rest { acc with language := some v }
  | "--context" :: v :: rest => parseArgsLoop rest { acc with context := v }
  | "--max-new-tokens" :: v :: rest =>
      parseArgsLoop rest { acc with maxNewTokens := (← parseNatArg "--max-new-tokens" v) }
  | "--chunk-sec" :: v :: rest =>
      parseArgsLoop rest { acc with chunkSec := (← parseFloatArg "--chunk-sec" v) }
  | "--hop-sec" :: v :: rest =>
      parseArgsLoop rest { acc with hopSec := (← parseFloatArg "--hop-sec" v) }
  | "--keep-sec" :: v :: rest =>
      parseArgsLoop rest { acc with keepSec := (← parseFloatArg "--keep-sec" v) }
  | "--run-sec" :: v :: rest =>
      parseArgsLoop rest { acc with runSec := (← parseFloatArg "--run-sec" v) }
  | "--commit-every" :: v :: rest =>
      parseArgsLoop rest { acc with commitEvery := (← parseNatArg "--commit-every" v).toNat }
  | "--prompt-tokens" :: v :: rest =>
      parseArgsLoop rest { acc with promptTokens := (← parseNatArg "--prompt-tokens" v).toNat }
  | "--use-case" :: v :: rest =>
      parseArgsLoop rest { acc with useCase := (← parseUseCaseArg v) }
  | "--rollback-chars" :: v :: rest =>
      parseArgsLoop rest { acc with rollbackChars := (← parseNatArg "--rollback-chars" v).toNat }
  | "--silence-sec" :: v :: rest =>
      parseArgsLoop rest { acc with silenceSec := (← parseFloatArg "--silence-sec" v) }
  | "--min-speech-sec" :: v :: rest =>
      parseArgsLoop rest { acc with minSpeechSec := (← parseFloatArg "--min-speech-sec" v) }
  | "--speech-rms-threshold" :: v :: rest =>
      parseArgsLoop rest { acc with speechRmsThreshold := (← parseFloatArg "--speech-rms-threshold" v) }
  | "--emit-partials" :: rest =>
      parseArgsLoop rest { acc with emitPartials := true }
  | "--plain-ui" :: rest =>
      parseArgsLoop rest { acc with prettyUI := false }
  | "--simple-output" :: rest =>
      parseArgsLoop rest { acc with simpleOutput := true }
  | "--help" :: _ =>
      IO.println "Usage: lake exe Qwen3ASRLiveMic [options]"
      IO.println "  --source <path-or-repo>  Local model dir or HF repo id"
      IO.println "  --model-dir <path>       Alias for --source (backward compatible)"
      IO.println "  --revision <rev>         HF revision/branch/tag (default: main)"
      IO.println "  --cache-dir <path>       Local cache for downloaded files"
      IO.println "  --language <name>        Optional forced language"
      IO.println "  --context <text>         Optional system context"
      IO.println "  --max-new-tokens <n>     Greedy decode max new tokens"
      IO.println "  --chunk-sec <f>          Decode window seconds (overlap window)"
      IO.println "  --hop-sec <f>            Step seconds between decodes"
      IO.println "  --keep-sec <f>           Legacy overlap knob (kept for compatibility)"
      IO.println "  --run-sec <f>            Total streaming duration"
      IO.println "  --commit-every <n>       Force-finalize unstable tail every n decodes (0=auto)"
      IO.println "  --prompt-tokens <n>      Streaming prompt token cap (bounded decode complexity)"
      IO.println "  --use-case <name>        generic | tv | conversation"
      IO.println "  --rollback-chars <n>     Stabilization rollback chars (default: 12)"
      IO.println "  --silence-sec <f>        Conversation endpoint silence seconds"
      IO.println "  --min-speech-sec <f>     Conversation min speech before activation"
      IO.println "  --speech-rms-threshold   Conversation speech RMS gate (default: 0.008)"
      IO.println "  --emit-partials          In conversation mode, emit PARTIAL lines"
      IO.println "  --plain-ui               Disable in-place terminal UI updates"
      IO.println "  --simple-output          Print single transcript updates (no dual-track lines)"
      throw <| IO.userError ""
  | x :: _ => throw <| IO.userError s!"Unknown argument: {x}"

private def parseArgs (xs : List String) : IO Args :=
  parseArgsLoop xs {}

private def toSamples (sec : Float) : Nat :=
  let n := ((sec * 16000.0) + 0.5).toUInt64.toNat
  if n == 0 then 1 else n

private def safeSubNat (a b : Nat) : Nat := if a >= b then a - b else 0

private structure AudioRing where
  data : Array Float
  head : Nat
  size : Nat
  cap : Nat
  deriving Inhabited

private def mkAudioRing (cap : Nat) : AudioRing :=
  if cap == 0 then
    { data := #[], head := 0, size := 0, cap := 0 }
  else
    { data := Array.replicate cap 0.0, head := 0, size := 0, cap := cap }

private def AudioRing.clear (r : AudioRing) : AudioRing :=
  { r with head := 0, size := 0 }

private def AudioRing.pushOne (r : AudioRing) (x : Float) : AudioRing :=
  if r.cap == 0 then
    r
  else
    let h := if r.head < r.cap then r.head else 0
    let data := r.data.set! h x
    let head := (h + 1) % r.cap
    let size := if r.size < r.cap then r.size + 1 else r.cap
    { r with data := data, head := head, size := size }

private def AudioRing.pushChunk (r : AudioRing) (xs : Array Float) : AudioRing :=
  Id.run do
    let mut out := r
    for x in xs do
      out := out.pushOne x
    out

private def AudioRing.tail (r : AudioRing) (n : Nat) : Array Float :=
  if r.size == 0 || n == 0 then
    #[]
  else
    let take := Nat.min n r.size
    let start :=
      if r.size < r.cap then
        r.size - take
      else
        (r.head + r.cap - take) % r.cap
    Id.run do
      let mut out : Array Float := Array.mkEmpty take
      for i in [:take] do
        let idx := (start + i) % r.cap
        out := out.push (r.data[idx]!)
      out

private def AudioRing.toArray (r : AudioRing) : Array Float :=
  r.tail r.size

private def audioRingFromTail (cap : Nat) (xs : Array Float) : AudioRing :=
  Id.run do
    let mut r := mkAudioRing cap
    let ys := if xs.size <= cap then xs else xs.extract (xs.size - cap) xs.size
    for x in ys do
      r := r.pushOne x
    r

private def rms (xs : Array Float) : Float :=
  if xs.isEmpty then
    0.0
  else
    let sumSq := xs.foldl (fun acc x => acc + x * x) 0.0
    Float.sqrt (sumSq / xs.size.toFloat)

private def toChars (s : String) : Array Char := s.toList.toArray
private def fromChars (xs : Array Char) : String := String.ofList xs.toList

private def tailChars (s : String) (n : Nat) : String :=
  let xs := toChars s
  if xs.size <= n then s else fromChars (xs.extract (xs.size - n) xs.size)

private def suffixPrefixOverlap (a b : String) : Nat :=
  let xa := toChars a
  let xb := toChars b
  let n := Nat.min xa.size xb.size
  Id.run do
    let mut best := 0
    for k in [1:n + 1] do
      let mut ok := true
      for i in [:k] do
        if ok && xa[(xa.size - k) + i]! != xb[i]! then
          ok := false
      if ok then
        best := k
    best

private def dedupeWithCommittedTail (committedTail candidate : String) : String :=
  let overlap := suffixPrefixOverlap committedTail candidate
  let xs := toChars candidate
  if overlap >= xs.size then "" else fromChars (xs.extract overlap xs.size)

private def previewTailText (s : String) (n : Nat := 120) : String :=
  let t := trimWhitespace s
  let xs := toChars t
  if xs.size <= n then
    t
  else
    "..." ++ fromChars (xs.extract (xs.size - n) xs.size)

private def uiPrintInline (s : String) : IO Unit := do
  IO.print "\x1b[2K\r"
  IO.print s
  (← IO.getStdout).flush

private def uiClearInline : IO Unit := do
  IO.print "\x1b[2K\r"
  (← IO.getStdout).flush

def runMain (argv : List String) : IO UInt32 := do
  let args ← parseArgs argv
  if args.chunkSec <= 0.0 || args.hopSec <= 0.0 || args.keepSec < 0.0 || args.runSec <= 0.0 then
    throw <| IO.userError "chunk-sec, hop-sec, run-sec must be > 0 and keep-sec must be >= 0"
  if args.keepSec > args.hopSec then
    throw <| IO.userError "keep-sec must be <= hop-sec"
  if args.rollbackChars == 0 then
    throw <| IO.userError "rollback-chars must be > 0"
  if args.useCase == .conversation &&
      (args.silenceSec <= 0.0 || args.minSpeechSec <= 0.0 || args.speechRmsThreshold < 0.0) then
    throw <| IO.userError "conversation mode requires silence-sec > 0, min-speech-sec > 0, speech-rms-threshold >= 0"

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
  let ⟨_device, model⟩ ← Qwen3ASRForConditionalGeneration.loadSharded modelDir cfg

  let hopSamples := toSamples args.hopSec
  let silenceSamples := toSamples args.silenceSec
  let minSpeechSamples := toSamples args.minSpeechSec
  let ratioSteps := Nat.max 1 (((args.chunkSec / args.hopSec) + 0.5).toUInt64.toNat)
  let autoCommitEvery := Nat.max 1 (safeSubNat ratioSteps 1)
  let commitEverySteps := if args.commitEvery == 0 then autoCommitEvery else args.commitEvery
  let maxNewTokensStream : UInt64 :=
    if args.useCase == .conversation then
      args.maxNewTokens
    else if args.maxNewTokens == 128 then
      96
    else
      args.maxNewTokens
  let steps := Nat.max 1 (((args.runSec / args.hopSec) + 0.5).toUInt64.toNat)

  IO.println s!"live mic: use_case={args.useCase.toString} chunk_sec={args.chunkSec} hop_sec={args.hopSec} keep_sec={args.keepSec} run_sec={args.runSec}"
  if args.useCase != .conversation then
    IO.println s!"stream commits: every={commitEverySteps} decode(s), prompt_tokens={args.promptTokens}, max_new_tokens={maxNewTokensStream}"
  if args.useCase == .conversation then
    IO.println s!"conversation gate: silence_sec={args.silenceSec} min_speech_sec={args.minSpeechSec} speech_rms_threshold={args.speechRmsThreshold}"
  if args.prettyUI then
    IO.println "ui: pretty (inline partial updates)"
  Tyr.Audio.AppleInput.start 16000 1 100

  try
    match args.useCase with
    | .conversation =>
        -- Accumulate full utterance audio (not a sliding window) so early
        -- words are never lost.  Re-transcribe the full buffer periodically.
        let preRollSamples := toSamples 0.5
        let transcribeInterval := toSamples 2.0
        let mut utteranceAudio : Array Float := #[]
        let mut prevSimple := ""
        let mut activeSpeechSamples := 0
        let mut activeSilenceSamples := 0
        let mut conversationActive := false
        let mut seenSpeech := false
        let mut lastTranscribeSize : Nat := 0
        for _ in [:steps] do
          let pcm ← Tyr.Audio.AppleInput.read hopSamples.toUInt64 1500
          if pcm.isEmpty then continue

          let frameRms := rms pcm
          -- Always accumulate audio.
          utteranceAudio := utteranceAudio ++ pcm

          if frameRms >= args.speechRmsThreshold then
            activeSpeechSamples := activeSpeechSamples + pcm.size
            activeSilenceSamples := 0
            if !conversationActive && activeSpeechSamples >= minSpeechSamples then
              conversationActive := true
              seenSpeech := true
              prevSimple := ""
          else
            if conversationActive then
              activeSilenceSamples := activeSilenceSamples + pcm.size
            else
              -- Not yet active — slide pre-roll window.
              if utteranceAudio.size > preRollSamples then
                utteranceAudio := utteranceAudio.extract (utteranceAudio.size - preRollSamples) utteranceAudio.size

          -- Periodic transcription on the full accumulated audio.
          let newSinceTranscribe := utteranceAudio.size - lastTranscribeSize
          if conversationActive && newSinceTranscribe >= transcribeInterval && utteranceAudio.size > 0 then
            lastTranscribeSize := utteranceAudio.size
            let out ← model.transcribeWaveform
              tok pre utteranceAudio
              (context := args.context) (language := args.language)
              (maxNewTokens := maxNewTokensStream)
            let text := trimWhitespace out.text
            if !text.isEmpty && text != prevSimple then
              prevSimple := text
              if args.simpleOutput then
                IO.println s!"TEXT: {text}"
              else if args.emitPartials then
                IO.println s!"PARTIAL= {text}"
              if args.prettyUI && !args.simpleOutput then
                uiPrintInline s!"LIVE: {previewTailText text}"

          -- Utterance endpoint: silence exceeded threshold.
          if conversationActive && activeSilenceSamples >= silenceSamples then
            -- Final transcription on the full utterance.
            let out ← model.transcribeWaveform
              tok pre utteranceAudio
              (context := args.context) (language := args.language)
              (maxNewTokens := maxNewTokensStream)
            let finalText := trimWhitespace out.text
            if !finalText.isEmpty then
              if args.prettyUI then
                uiClearInline
              IO.println s!"UTTERANCE: {finalText}"
            utteranceAudio := #[]
            prevSimple := ""
            activeSpeechSamples := 0
            activeSilenceSamples := 0
            conversationActive := false
            seenSpeech := false
            lastTranscribeSize := 0

        if seenSpeech || conversationActive then
          let out ← model.transcribeWaveform
            tok pre utteranceAudio
            (context := args.context) (language := args.language)
            (maxNewTokens := maxNewTokensStream)
          let finalText := trimWhitespace out.text
          if !finalText.isEmpty then
            if args.prettyUI then
              uiClearInline
            IO.println s!"UTTERANCE: {finalText}"
    | .generic | .tv =>
        let pendingCap := Nat.max 1 (4 * hopSamples)
        let mut pcmNew := mkAudioRing pendingCap
        let mut asrState ← model.initStreamingState
          (context := args.context)
          (language := args.language)
          (unfixedChunkNum := 2)
          (unfixedTokenNum := 5)
          (promptMaxTokens := args.promptTokens)
          (chunkSizeSec := args.chunkSec)
          (stepSizeSec := args.hopSec)
        let mut prevSimple := ""
        let mut lastPartial := ""
        let mut decodeStepsSinceCommit := 0
        let committedTailCap := 256
        let mut committedTail := ""
        let mut lockedLanguage : Option String := args.language
        let mut backpressureDrops : Nat := 0
        for _ in [:steps] do
          let pcm ← Tyr.Audio.AppleInput.read hopSamples.toUInt64 1500
          if !pcm.isEmpty then
            pcmNew := pcmNew.pushChunk pcm

          if pcmNew.size > 2 * hopSamples then
            backpressureDrops := backpressureDrops + 1
            IO.eprintln s!"live mic: warning: dropping buffered audio due to backpressure (event {backpressureDrops})"
            pcmNew := audioRingFromTail pendingCap (pcmNew.tail hopSamples)

          if pcmNew.size < hopSamples then
            continue

          let newAudio := pcmNew.toArray
          pcmNew := pcmNew.clear

          let chunkIdBefore := asrState.chunkId
          let asrNext ← model.streamingTranscribe
            tok
            pre
            newAudio
            asrState
            (maxNewTokens := maxNewTokensStream)
          asrState := asrNext
          let decodedSteps := asrState.chunkId - chunkIdBefore
          if decodedSteps == 0 then
            continue

          if lockedLanguage.isNone then
            let detected := trimWhitespace asrState.language
            if !detected.isEmpty then
              lockedLanguage := some detected
              asrState := { asrState with forceLanguage := some detected }
          let partialText := trimWhitespace asrState.text

          if args.simpleOutput then
            if partialText != prevSimple then
              IO.println s!"TEXT: {partialText}"
              prevSimple := partialText
          else
            if partialText != lastPartial then
              if args.prettyUI then
                uiPrintInline s!"LIVE[{backpressureDrops}d]: {previewTailText partialText}"
              else
                IO.println s!"UNSTABLE= {partialText}"
              lastPartial := partialText
            decodeStepsSinceCommit := decodeStepsSinceCommit + decodedSteps
            if decodeStepsSinceCommit >= commitEverySteps then
              let commitText := dedupeWithCommittedTail committedTail partialText
              if !commitText.isEmpty then
                if args.prettyUI then
                  uiClearInline
                IO.println s!"STABLE+= {commitText}"
                committedTail := tailChars (committedTail ++ commitText) committedTailCap
              if !lastPartial.isEmpty then
                if args.prettyUI then
                  uiClearInline
                else
                  IO.println "UNSTABLE= "
                lastPartial := ""
              decodeStepsSinceCommit := 0

        let chunkIdBeforeFlush := asrState.chunkId
        let asrFinal ← model.finishStreamingTranscribe
          tok
          pre
          asrState
          (maxNewTokens := maxNewTokensStream)
        asrState := asrFinal
        if asrState.chunkId > chunkIdBeforeFlush then
          let partialText := trimWhitespace asrState.text
          if args.simpleOutput then
            if partialText != prevSimple then
              IO.println s!"TEXT: {partialText}"
              prevSimple := partialText
          else if partialText != lastPartial then
            if args.prettyUI then
              uiPrintInline s!"LIVE[{backpressureDrops}d]: {previewTailText partialText}"
            else
              IO.println s!"UNSTABLE= {partialText}"
            lastPartial := partialText

        if !args.simpleOutput && !lastPartial.isEmpty then
          let commitText := dedupeWithCommittedTail committedTail lastPartial
          if !commitText.isEmpty then
            if args.prettyUI then
              uiClearInline
            IO.println s!"STABLE+= {commitText}"
          if args.prettyUI then
            uiClearInline
          else
            IO.println "UNSTABLE= "
    Tyr.Audio.AppleInput.stop
  catch e =>
    Tyr.Audio.AppleInput.stop
    throw e

  pure 0

end Examples.Qwen3ASR

def main (argv : List String) : IO UInt32 :=
  Examples.Qwen3ASR.runMain argv
