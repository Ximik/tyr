/-
  Tyr/Model/Qwen3ASR/ForcedAligner.lean

  Lean-native forced-aligner helpers for Qwen3-ASR:
  - text tokenization/placeholder preparation
  - timestamp monotonic fixup
  - conversion from thinker logits/output IDs to structured spans
-/
import Tyr.Typed
import Tyr.Model.Qwen3ASR.Config

namespace torch.qwen3asr

structure ForcedAlignItem where
  text : String
  startTime : Float
  endTime : Float
  deriving Repr, Inhabited

structure ForcedAlignResult where
  items : Array ForcedAlignItem
  deriving Repr, Inhabited

namespace Qwen3ForceAlignProcessor

private def isKeptChar (ch : Char) : Bool :=
  ch == '\'' || ch.isAlphanum

private def cleanToken (token : String) : String :=
  String.ofList (token.toList.filter isKeptChar)

private def isCjkChar (ch : Char) : Bool :=
  let code := ch.toNat
  (0x4E00 <= code && code <= 0x9FFF) ||
  (0x3400 <= code && code <= 0x4DBF) ||
  (0x20000 <= code && code <= 0x2A6DF) ||
  (0x2A700 <= code && code <= 0x2B73F) ||
  (0x2B740 <= code && code <= 0x2B81F) ||
  (0x2B820 <= code && code <= 0x2CEAF) ||
  (0xF900 <= code && code <= 0xFAFF)

private def splitSegmentWithChinese (seg : String) : Array String :=
  Id.run do
    let mut out : Array String := #[]
    let mut buf : Array Char := #[]
    for ch in seg.toList do
      if isCjkChar ch then
        if !buf.isEmpty then
          out := out.push (String.ofList buf.toList)
          buf := #[]
        out := out.push (String.singleton ch)
      else
        buf := buf.push ch
    if !buf.isEmpty then
      out := out.push (String.ofList buf.toList)
    out

private def isJapaneseChar (ch : Char) : Bool :=
  let code := ch.toNat
  isCjkChar ch ||
  (0x3040 <= code && code <= 0x309F) ||  -- Hiragana
  (0x30A0 <= code && code <= 0x30FF) ||  -- Katakana
  (0x31F0 <= code && code <= 0x31FF) ||  -- Katakana Phonetic Extensions
  (0xFF66 <= code && code <= 0xFF9D)     -- Halfwidth Katakana

private def isKoreanChar (ch : Char) : Bool :=
  let code := ch.toNat
  (0xAC00 <= code && code <= 0xD7AF) ||  -- Hangul Syllables
  (0x1100 <= code && code <= 0x11FF) ||  -- Hangul Jamo
  (0x3130 <= code && code <= 0x318F) ||  -- Hangul Compatibility Jamo
  (0xA960 <= code && code <= 0xA97F) ||  -- Hangul Jamo Extended-A
  (0xD7B0 <= code && code <= 0xD7FF)     -- Hangul Jamo Extended-B

private def tokenizeScriptMixed (text : String) (isScriptChar : Char → Bool) : Array String :=
  Id.run do
    let mut out : Array String := #[]
    let mut buf : Array Char := #[]
    for ch in text.toList do
      if isScriptChar ch then
        if !buf.isEmpty then
          let tok := cleanToken (String.ofList buf.toList)
          if !tok.isEmpty then
            out := out.push tok
          buf := #[]
        out := out.push (String.singleton ch)
      else if isKeptChar ch then
        buf := buf.push ch
      else
        if !buf.isEmpty then
          let tok := cleanToken (String.ofList buf.toList)
          if !tok.isEmpty then
            out := out.push tok
          buf := #[]
    if !buf.isEmpty then
      let tok := cleanToken (String.ofList buf.toList)
      if !tok.isEmpty then
        out := out.push tok
    out

private def lowerAscii (c : Char) : Char :=
  if c >= 'A' && c <= 'Z' then
    Char.ofNat (c.toNat + 32)
  else
    c

private def toLowerAscii (s : String) : String :=
  String.ofList (s.toList.map lowerAscii)

def tokenizeSpaceLang (text : String) : Array String :=
  Id.run do
    let mut out : Array String := #[]
    for seg in text.splitOn " " do
      let cleaned := cleanToken seg
      if !cleaned.isEmpty then
        for tok in splitSegmentWithChinese cleaned do
          if !tok.isEmpty then
            out := out.push tok
    out

private def joinWith (xs : Array String) (sep : String) : String :=
  Id.run do
    if xs.isEmpty then
      ""
    else
      let mut out := xs[0]!
      for i in [1:xs.size] do
        out := out ++ sep ++ xs[i]!
      out

/-- Encode text into forced-aligner prompt tokens.
    Uses language-specific segmentation strategy for Japanese/Korean. -/
def encodeTimestampText (text : String) (language : String) : Array String × String :=
  let lang := toLowerAscii (language.trimAscii.toString)
  let wordList :=
    if lang == "japanese" then
      tokenizeScriptMixed text isJapaneseChar
    else if lang == "korean" then
      tokenizeScriptMixed text isKoreanChar
    else
      tokenizeSpaceLang text
  let inter := joinWith wordList "<timestamp><timestamp>"
  let body := if inter.isEmpty then "<timestamp><timestamp>" else inter ++ "<timestamp><timestamp>"
  let inputText := "<|audio_start|><|audio_pad|><|audio_end|>" ++ body
  (wordList, inputText)

/-- Port of reference monotonic timestamp repair (`fix_timestamp`). -/
def fixTimestamp (data : Array UInt64) : Array UInt64 :=
  Id.run do
    let n := data.size
    if n == 0 then
      return #[]

    let mut dp : Array Nat := Array.replicate n 1
    let mut parent : Array Int64 := Array.replicate n (-1)

    for i in [1:n] do
      for j in [:i] do
        if data.getD j 0 <= data.getD i 0 then
          let cand := dp.getD j 1 + 1
          if cand > dp.getD i 1 then
            dp := dp.set! i cand
            parent := parent.set! i (Int64.ofNat j)

    let mut maxLen : Nat := 0
    let mut maxIdx : Nat := 0
    for i in [:n] do
      let v := dp.getD i 0
      if v > maxLen then
        maxLen := v
        maxIdx := i

    let mut lis : Array Nat := #[]
    let mut idx : Int64 := Int64.ofNat maxIdx
    while idx != -1 do
      let k := idx.toUInt64.toNat
      lis := lis.push k
      idx := parent.getD k (-1)
    lis := lis.reverse

    let mut isNormal : Array Bool := Array.replicate n false
    for k in lis do
      isNormal := isNormal.set! k true

    let mut result := data
    let mut i : Nat := 0
    while i < n do
      if !(isNormal.getD i false) then
        let mut j := i
        while j < n && !(isNormal.getD j false) do
          j := j + 1
        let anomalyCount := j - i

        let mut leftVal : Option UInt64 := none
        if i > 0 then
          let mut k := i
          while k > 0 do
            let kk := k - 1
            if isNormal.getD kk false then
              leftVal := some (result.getD kk 0)
              k := 0
            else
              k := kk

        let mut rightVal : Option UInt64 := none
        let mut k2 := j
        while k2 < n do
          if isNormal.getD k2 false then
            rightVal := some (result.getD k2 0)
            k2 := n
          else
            k2 := k2 + 1

        if anomalyCount <= 2 then
          for k in [i:j] do
            let v :=
              match leftVal, rightVal with
              | none, some r => r
              | some l, none => l
              | some l, some r =>
                let dl := k - (i - 1)
                let dr := j - k
                if dl <= dr then l else r
              | none, none => result.getD k 0
            result := result.set! k v
        else
          match leftVal, rightVal with
          | some l, some r =>
            let steps := anomalyCount + 1
            for k in [i:j] do
              let numer := (r - l) * (k - i + 1).toUInt64
              let v := l + (numer / steps.toUInt64)
              result := result.set! k v
          | some l, none =>
            for k in [i:j] do
              result := result.set! k l
          | none, some r =>
            for k in [i:j] do
              result := result.set! k r
          | none, none =>
            pure ()
        i := j
      else
        i := i + 1

    result

def parseTimestamp (wordList : Array String) (timestampMs : Array UInt64) : ForcedAlignResult :=
  Id.run do
    let needed := wordList.size * 2
    let fixed0 :=
      if timestampMs.size >= needed then
        fixTimestamp (timestampMs[:needed].toArray)
      else
        fixTimestamp timestampMs
    let fixed :=
      if fixed0.size >= needed then
        fixed0
      else
        Id.run do
          let mut out := fixed0
          let padVal := out.getD (out.size - 1) 0
          for _ in [:needed - out.size] do
            out := out.push padVal
          out
    let mut items : Array ForcedAlignItem := #[]
    for i in [:wordList.size] do
      let s := fixed.getD (i * 2) 0
      let e := fixed.getD (i * 2 + 1) s
      items := items.push { text := wordList.getD i "", startTime := s.toFloat / 1000.0, endTime := e.toFloat / 1000.0 }
    pure { items }

end Qwen3ForceAlignProcessor

private def extractTimestampRows {batch seq : UInt64}
    (inputIds : DTensor #[batch, seq] .Int64)
    (outputIds : DTensor #[batch, seq] .Int64)
    (timestampTokenId : UInt64)
    (timestampSegmentTime : Float)
    : IO (Array (Array UInt64)) := do
  let inFlat : DTensor #[batch * seq] .Int64 := Tensor.reshapeChecked (Tensor.toInt64 inputIds) #[batch * seq]
  let outFlat : DTensor #[batch * seq] .Int64 := Tensor.reshapeChecked (Tensor.toInt64 outputIds) #[batch * seq]
  let inVals ← Tensor.toUInt64Array inFlat
  let outVals ← Tensor.toUInt64Array outFlat
  let mut rows : Array (Array UInt64) := Array.mkEmpty batch.toNat
  for b in [:batch.toNat] do
    let mut ts : Array UInt64 := #[]
    for t in [:seq.toNat] do
      let k := b * seq.toNat + t
      if inVals.getD k 0 == timestampTokenId then
        let bin := outVals.getD k 0
        let ms := (bin.toFloat * timestampSegmentTime).toUInt64
        ts := ts.push ms
    rows := rows.push ts
  pure rows

def alignFromOutputIds {batch seq : UInt64}
    (inputIds : DTensor #[batch, seq] .Int64)
    (outputIds : DTensor #[batch, seq] .Int64)
    (wordLists : Array (Array String))
    (timestampTokenId : UInt64)
    (timestampSegmentTime : Float := 1.0)
    : IO (Array ForcedAlignResult) := do
  let rows ← extractTimestampRows inputIds outputIds timestampTokenId timestampSegmentTime
  let mut out : Array ForcedAlignResult := Array.mkEmpty batch.toNat
  for b in [:batch.toNat] do
    let words := wordLists.getD b #[]
    let ts := rows.getD b #[]
    out := out.push (Qwen3ForceAlignProcessor.parseTimestamp words ts)
  pure out

def alignFromLogits {batch seq vocab : UInt64} {dtype : DType}
    (inputIds : DTensor #[batch, seq] .Int64)
    (logits : DTensor #[batch, seq, vocab] dtype)
    (wordLists : Array (Array String))
    (timestampTokenId : UInt64)
    (timestampSegmentTime : Float := 1.0)
    : IO (Array ForcedAlignResult) := do
  let outputIds := (Tensor.argmax logits 2).reshapeChecked #[batch, seq]
  alignFromOutputIds inputIds outputIds wordLists timestampTokenId timestampSegmentTime

end torch.qwen3asr
