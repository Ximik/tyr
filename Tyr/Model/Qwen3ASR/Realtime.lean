import Tyr.Model.Qwen3ASR.Transcribe
import Tyr.Tokenizer.Qwen3

namespace torch.qwen3asr

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
/-- Coherent transcript assembly state for overlap-window streaming.
    `stableText` is append-only; `unstableText` may change each hop. -/
structure RealtimeTranscriptState where
  rollbackChars : Nat := 12
  committedChars : Nat := 0
  stableText : String := ""
  unstableText : String := ""
  prevHypothesis : String := ""
  deriving Repr, Inhabited

/-- Delta emitted per decode step. -/
structure TranscriptDelta where
  stableAppend : String := ""
  unstableText : String := ""
  fullText : String := ""
  deriving Repr, Inhabited

private def toChars (s : String) : Array Char := s.toList.toArray
private def fromChars (xs : Array Char) : String := String.ofList xs.toList

private def commonPrefixLen (a b : String) : Nat :=
  let xa := toChars a
  let xb := toChars b
  let n := Nat.min xa.size xb.size
  Id.run do
    let mut i := 0
    while i < n && xa[i]! == xb[i]! do
      i := i + 1
    i

private def prefixChars (s : String) (n : Nat) : String :=
  fromChars <| (toChars s).extract 0 (Nat.min n (toChars s).size)

private def suffixChars (s : String) (start : Nat) : String :=
  let xs := toChars s
  if start >= xs.size then "" else fromChars (xs.extract start xs.size)

private def safeSub (a b : Nat) : Nat := if a >= b then a - b else 0

private def charLen (s : String) : Nat := (toChars s).size

/-- Merge a fresh hypothesis into coherent `(stable, unstable)` form. -/
def updateTranscriptState
    (st : RealtimeTranscriptState)
    (hypothesis : String)
    : RealtimeTranscriptState × TranscriptDelta :=
  let oldStableLen := charLen st.stableText
  let lcpStable := commonPrefixLen st.stableText hypothesis
  let stableRegression := lcpStable < oldStableLen
  let lcpPrev := commonPrefixLen st.prevHypothesis hypothesis
  let commitCandidate := safeSub lcpPrev st.rollbackChars
  let commitCharsRaw := Nat.max st.committedChars commitCandidate
  let commitChars := Nat.max oldStableLen commitCharsRaw
  let stable :=
    if stableRegression then
      st.stableText
    else
      prefixChars hypothesis commitChars
  let unstable :=
    if stableRegression then
      st.unstableText
    else
      suffixChars hypothesis commitChars
  let newStableLen := charLen stable
  let stableAppend := if newStableLen > oldStableLen then suffixChars stable oldStableLen else ""
  let st' : RealtimeTranscriptState := {
    st with
    committedChars := if stableRegression then oldStableLen else commitChars
    stableText := stable
    unstableText := unstable
    prevHypothesis := hypothesis
  }
  let delta : TranscriptDelta := {
    stableAppend := stableAppend
    unstableText := unstable
    fullText := stable ++ unstable
  }
  (st', delta)

/-- Force unstable tail into stable text at a commit boundary. -/
def finalizeTranscriptState
    (st : RealtimeTranscriptState)
    : RealtimeTranscriptState × TranscriptDelta :=
  let oldStableLen := charLen st.stableText
  let stable := st.stableText ++ st.unstableText
  let newStableLen := charLen stable
  let stableAppend := if newStableLen > oldStableLen then suffixChars stable oldStableLen else ""
  let st' : RealtimeTranscriptState := {
    st with
    committedChars := newStableLen
    stableText := stable
    unstableText := ""
    prevHypothesis := stable
  }
  let delta : TranscriptDelta := {
    stableAppend := stableAppend
    unstableText := ""
    fullText := stable
  }
  (st', delta)

/-- Decode one overlap window and return coherent transcript deltas.
    This is complementary to existing streaming helpers and adds no overhead
    unless called by a realtime loop. -/
def decodeOverlapStep
    {cfg : Qwen3ASRConfig}
    (m : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (hMel : preprocessor.featureSize = cfg.thinkerConfig.audioConfig.numMelBins)
    (audio16kWindow : Array Float)
    (state : RealtimeTranscriptState)
    (context : String := "")
    (language : Option String := none)
    (maxNewTokens : Nat := 128)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (RealtimeTranscriptState × TranscriptDelta) := do
  let out ← m.transcribeWaveform
    tok
    preprocessor hMel
    audio16kWindow
    (context := context)
    (language := language)
    (returnTimeStamps := false)
    (maxNewTokens := maxNewTokens)
    (eosTokenIds := eosTokenIds)
  pure (updateTranscriptState state out.text)

end torch.qwen3asr
