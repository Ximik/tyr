/-
  Tyr/Model/Whisper/Transcribe.lean

  Offline Whisper transcription pipeline in Tyr.
  Includes a Whisper-CLI-style decoding path:
  - beam-first decode at temperature 0
  - temperature fallback with best-of sampling
  - no-speech gating and decode-failure heuristics
  - rolling prompt-context carry between chunks
-/
import Tyr.Model.Whisper.ConfigIO
import Tyr.Model.Whisper.Model
import Tyr.Model.Whisper.Weights
import Tyr.Model.Qwen3ASR.Frontend
import Tyr.Model.Qwen3ASR.PreprocessorConfig
import Tyr.Tokenizer.Qwen3

namespace torch.whisper

variable {device : Device}
open torch.qwen3asr

structure WhisperTranscription where
  language : String
  text : String
  tokenIds : Array UInt32 := #[]
  deriving Repr, Inhabited

structure WhisperBundle where
  cfg : WhisperConfig
  device : Device
  model : WhisperForConditionalGeneration (device := device) cfg
  tok : tokenizer.qwen3.QwenTokenizer
  preprocessor : PreprocessorConfig

structure WhisperDecodeOptions where
  beamSize : UInt64 := 1
  bestOf : UInt64 := 1
  temperature : Float := 0.0
  temperatureInc : Float := 0.2
  maxTemperature : Float := 1.0
  topK : UInt64 := 0
  topP : Float := 1.0
  logprobThreshold : Float := -1.0
  noSpeechThreshold : Float := 0.6
  compressionRatioThreshold : Float := 2.4
  conditionOnPreviousText : Bool := true
  maxContextTokens : UInt64 := 0
  chunkOverlapSeconds : Float := 2.0
  resetContextOnFallback : Bool := true
  noFallback : Bool := true
  deriving Repr, Inhabited

private def sampleRate16k : UInt64 := 16000

private structure DecodeState where
  allIds : Array UInt32
  generated : Array UInt32 := #[]
  sumLogprob : Float := 0.0
  tokenCount : Nat := 0
  noSpeechProb : Float := 0.0
  measuredNoSpeech : Bool := false
  deriving Inhabited

private structure BeamState (cfg : WhisperConfig) where
  allIds : Array UInt32
  generated : Array UInt32 := #[]
  sumLogprob : Float := 0.0
  tokenCount : Nat := 0
  noSpeechProb : Float := 0.0
  measuredNoSpeech : Bool := false
  finished : Bool := false
  caches : Array (WhisperForConditionalGeneration.LayerKVCache (device := device) cfg 1) := #[]
  lastLogits? : Option (Tensor { device := device, shape := #[1, cfg.vocabSize] }) := none
  nextPos : UInt64 := 0
  deriving Inhabited

private structure WhisperDecodeCandidate where
  generated : Array UInt32 := #[]
  text : String := ""
  avgLogprob : Float := -100.0
  noSpeechProb : Float := 0.0
  compressionRatio : Float := 0.0
  repetitionLoop : Bool := false
  temperature : Float := 0.0
  deriving Inhabited

private structure DecodeTokenRules where
  forbidden : Array UInt32 := #[]
  beginSuppress : Array UInt32 := #[]
  timestampBeginId : Option UInt32 := none
  maxInitialTimestampTokens : Nat := 0
  deriving Inhabited

private structure AudioChunk where
  start : Nat
  stop : Nat
  wav : Array Float
  deriving Inhabited

private def normalizeLanguageCode (language : String) : String :=
  let l := language.trimAscii.toString.toLower
  if l.isEmpty then
    "en"
  else if l == "english" then
    "en"
  else if l == "chinese" then
    "zh"
  else if l == "japanese" then
    "ja"
  else if l == "korean" then
    "ko"
  else
    l

private def languageTokenString (language : String) : String :=
  let code := normalizeLanguageCode language
  if code.startsWith "<|" && code.endsWith "|>" then
    code
  else
    s!"<|{code}|>"

private def tokenIdByText? (tok : tokenizer.qwen3.QwenTokenizer) (text : String) : Option UInt32 :=
  tok.specialTokens.get? text <|> tok.tokenToId.get? text

private def isTimestampSpecialToken (text : String) : Bool :=
  text.startsWith "<|" && text.endsWith "|>" && text.contains '.'

private def pushUniqueId (xs : Array UInt32) (id : UInt32) : Array UInt32 :=
  if xs.contains id then xs else xs.push id

private def buildDecodeTokenRules
    (tok : tokenizer.qwen3.QwenTokenizer)
    (cfg : WhisperConfig)
    (noTimestamps : Bool)
    : DecodeTokenRules :=
  Id.run do
    let mut forbidden : Array UInt32 := #[]
    for (id, text) in tok.idToSpecial.toList do
      let isEos := id.toUInt64 == cfg.eosTokenId
      let isTimestamp := isTimestampSpecialToken text
      let allowSpecial :=
        isEos || (!noTimestamps && isTimestamp)
      if !allowSpecial then
        forbidden := pushUniqueId forbidden id
    for id64 in cfg.suppressTokens do
      if id64 < cfg.vocabSize then
        forbidden := pushUniqueId forbidden id64.toUInt32
    let mut beginSuppress : Array UInt32 := #[]
    for id64 in cfg.beginSuppressTokens do
      if id64 < cfg.vocabSize then
        beginSuppress := pushUniqueId beginSuppress id64.toUInt32
    let timestampBeginId :=
      if noTimestamps then none else tokenIdByText? tok "<|0.00|>"
    let maxInitialTimestampTokens :=
      -- whisper.cpp default: max_initial_ts = 1.0s, 0.02s timestamp precision => 50 bins.
      if noTimestamps then 0 else 50
    {
      forbidden := forbidden
      beginSuppress := beginSuppress
      timestampBeginId := timestampBeginId
      maxInitialTimestampTokens := maxInitialTimestampTokens
    }

private def buildPromptIds
    (cfg : WhisperConfig)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (language : String)
    (noTimestamps : Bool := true)
    : Array UInt32 :=
  Id.run do
    let startId :=
      (tokenIdByText? tok "<|startoftranscript|>").getD cfg.decoderStartTokenId.toUInt32
    let mut out : Array UInt32 := #[startId]
    let langTok := languageTokenString language
    if let some langId := tokenIdByText? tok langTok then
      out := out.push langId
    if let some transcribeId := tokenIdByText? tok "<|transcribe|>" then
      out := out.push transcribeId
    if noTimestamps then
      if let some noTsId := tokenIdByText? tok "<|notimestamps|>" then
        out := out.push noTsId
    out

private def decodeGeneratedText
    (tok : tokenizer.qwen3.QwenTokenizer)
    (ids : Array UInt32)
    : String :=
  let filtered := ids.filter (fun id => !tok.idToSpecial.contains id)
  (tokenizer.qwen3.decodeText tok filtered).trimAscii.toString

private def takeLast (xs : Array α) (n : Nat) : Array α :=
  if n == 0 then
    #[]
  else if xs.size <= n then
    xs
  else
    xs.extract (xs.size - n) xs.size

private def mergeChunkTexts (parts : Array String) : String :=
  Id.run do
    let mut out := ""
    for p in parts do
      let t := p.trimAscii.toString
      if !t.isEmpty then
        if out.isEmpty then
          out := t
        else
          out := out ++ " " ++ t
    out

private def trimDanglingTailAfterPeriod (text : String) : String :=
  let t := text.trimAscii.toString
  let parts := t.splitOn "."
  match parts.reverse with
  | [] => t
  | suffixRaw :: restRev =>
      let suffix := suffixRaw.trimAscii.toString
      if suffix.isEmpty then
        t
      else
        let suffixWords :=
          (suffix.splitOn " ").foldl
            (fun acc w =>
              let ww := w.trimAscii.toString
              if ww.isEmpty then acc else acc + 1)
            0
        let isLongText := t.toList.length > 200
        if isLongText && suffixWords <= 8 then
          let body := (String.intercalate "." restRev.reverse).trimAscii.toString
          if body.isEmpty then
            t
          else
            body ++ "."
        else
          t

private def splitWaveformFixed
    (wav : Array Float)
    (chunkSamples : Nat)
    : Array (Array Float) :=
  if chunkSamples == 0 || wav.size <= chunkSamples then
    #[wav]
  else
    Id.run do
      let mut out : Array (Array Float) := #[]
      let mut start : Nat := 0
      while start < wav.size do
        let stop := Nat.min wav.size (start + chunkSamples)
        out := out.push (wav.extract start stop)
        start := stop
      out

private def splitWaveformWithOverlap
    (wav : Array Float)
    (chunkSamples : Nat)
    (overlapSamples : Nat)
    : Array AudioChunk :=
  if chunkSamples == 0 || wav.size <= chunkSamples then
    #[{ start := 0, stop := wav.size, wav := wav }]
  else
    Id.run do
      let maxOverlap := if chunkSamples <= 1 then 0 else chunkSamples - 1
      let overlap := Nat.min overlapSamples maxOverlap
      let stepRaw := chunkSamples - overlap
      let step := Nat.max 1 stepRaw
      let mut out : Array AudioChunk := #[]
      let mut start : Nat := 0
      while start < wav.size do
        let stop := Nat.min wav.size (start + chunkSamples)
        out := out.push { start := start, stop := stop, wav := wav.extract start stop }
        if stop == wav.size then
          start := wav.size
        else
          start := start + step
      out

private def isTimestampTokenId
    (tok : tokenizer.qwen3.QwenTokenizer)
    (id : UInt32)
    : Bool :=
  match tok.idToSpecial.get? id with
  | some text => isTimestampSpecialToken text
  | none => false

private def timestampAdvanceSamples
    (generated : Array UInt32)
    (timestampBeginId : Option UInt32)
    (samplesPerTimestamp : Nat)
    (chunkSize : Nat)
    : Nat :=
  match timestampBeginId with
  | none => 0
  | some tsBegin =>
      Id.run do
        let mut maxDelta : Nat := 0
        let tsBeginNat := tsBegin.toNat
        for id in generated do
          let idNat := id.toNat
          if idNat >= tsBeginNat then
            let delta := idNat - tsBeginNat
            if delta > maxDelta then
              maxDelta := delta
        if maxDelta == 0 then
          0
        else
          Nat.min chunkSize (maxDelta * samplesPerTimestamp)

private def whisperMaxChunkSeconds
    (cfg : WhisperConfig)
    (pre : PreprocessorConfig)
    : Float :=
  let sr := if pre.samplingRate == 0 then sampleRate16k else pre.samplingRate
  let hop := if pre.hopLength == 0 then 160 else pre.hopLength
  let maxFrames := cfg.maxSourcePositions * 2
  (maxFrames * hop).toFloat / sr.toFloat

private def decodeTemperatures (opts : WhisperDecodeOptions) : Array Float :=
  let t0 := if opts.temperature < 0.0 then 0.0 else opts.temperature
  if opts.noFallback || opts.temperatureInc <= 0.0 then
    #[t0]
  else
    Id.run do
      let maxT := if opts.maxTemperature < t0 then t0 else opts.maxTemperature
      let mut out : Array Float := #[t0]
      let mut t := t0
      let mut guard : Nat := 0
      while t + opts.temperatureInc <= maxT + (1e-6 : Float) && guard < 64 do
        t := t + opts.temperatureInc
        out := out.push t
        guard := guard + 1
      out

private def uniqueTokenCount (ids : Array UInt32) : Nat :=
  Id.run do
    let mut uniq : Array UInt32 := #[]
    for id in ids do
      if !uniq.contains id then
        uniq := uniq.push id
    uniq.size

private def approxCompressionRatio (ids : Array UInt32) : Float :=
  if ids.isEmpty then
    0.0
  else
    let uniq := uniqueTokenCount ids
    if uniq == 0 then
      0.0
    else
      ids.size.toFloat / uniq.toFloat

private def hasTailRepeatPattern (ids : Array UInt32) (patternLen repeats : Nat) : Bool :=
  if patternLen == 0 || repeats <= 1 then
    false
  else
    let need := patternLen * repeats
    if ids.size < need then
      false
    else
      let tail := ids.extract (ids.size - need) ids.size
      let refPat := tail.extract 0 patternLen
      Id.run do
        let mut ok := true
        for r in [:repeats] do
          if ok then
            let seg := tail.extract (r * patternLen) ((r + 1) * patternLen)
            if seg != refPat then
              ok := false
        ok

private def hasRepeatedSingleTail (ids : Array UInt32) (runLen : Nat) : Bool :=
  if runLen <= 1 || ids.size < runLen then
    false
  else
    let tail := ids.extract (ids.size - runLen) ids.size
    let first := tail[0]!
    Id.run do
      let mut ok := true
      for id in tail do
        if ok && id != first then
          ok := false
      ok

private def hasRepetitionLoop (ids : Array UInt32) : Bool :=
  hasRepeatedSingleTail ids 8 ||
  hasTailRepeatPattern ids 2 4 ||
  hasTailRepeatPattern ids 3 3

private def composePromptIds
    (basePrompt : Array UInt32)
    (rollingContext : Array UInt32)
    (promptMaxLen : Nat)
    (contextLimit : Nat)
    (conditionOnPreviousText : Bool)
    (prevTokenId : Option UInt32)
    : Array UInt32 :=
  if promptMaxLen <= basePrompt.size then
    basePrompt.extract 0 promptMaxLen
  else if !conditionOnPreviousText || rollingContext.isEmpty || contextLimit == 0 then
    basePrompt
  else
    let room := promptMaxLen - basePrompt.size
    let usePrev := prevTokenId.isSome && room > 0
    let ctxRoom := if usePrev then room - 1 else room
    let ctxBudget := Nat.min contextLimit ctxRoom
    let ctx := takeLast rollingContext ctxBudget
    Id.run do
      let mut out : Array UInt32 := #[]
      if usePrev then
        if let some prev := prevTokenId then
          out := out.push prev
      out := out ++ ctx
      out ++ basePrompt

private def tokenOverlapLength
    (leftIds : Array UInt32)
    (rightIds : Array UInt32)
    (maxSearch : Nat := 256)
    : Nat :=
  if leftIds.isEmpty || rightIds.isEmpty then
    0
  else
    let maxCand := Nat.min maxSearch (Nat.min leftIds.size rightIds.size)
    Id.run do
      let mut k := maxCand
      let mut best : Nat := 0
      while k > 0 && best == 0 do
        let left := leftIds.extract (leftIds.size - k) leftIds.size
        let right := rightIds.extract 0 k
        if left == right then
          best := k
        else
          k := k - 1
      best

private def appendWithTokenStitch
    (acc : Array UInt32)
    (next : Array UInt32)
    : Array UInt32 :=
  if acc.isEmpty then
    next
  else if next.isEmpty then
    acc
  else
    let overlap := tokenOverlapLength acc next
    acc ++ next.extract overlap next.size

private def candidateNeedsFallback (opts : WhisperDecodeOptions) (cand : WhisperDecodeCandidate) : Bool :=
  let badCompression :=
    opts.compressionRatioThreshold > 0.0 &&
      cand.compressionRatio > opts.compressionRatioThreshold
  let badLogprob :=
    cand.avgLogprob < opts.logprobThreshold &&
      cand.noSpeechProb < opts.noSpeechThreshold
  let likelyDecodeFailure := cand.generated.isEmpty && cand.noSpeechProb < opts.noSpeechThreshold
  cand.repetitionLoop || badCompression || badLogprob || likelyDecodeFailure

private def isNoSpeechChunk (opts : WhisperDecodeOptions) (cand : WhisperDecodeCandidate) : Bool :=
  cand.noSpeechProb > opts.noSpeechThreshold &&
    cand.avgLogprob < opts.logprobThreshold

private def buildSingleTokenOnDevice
    (id : UInt32)
    (device : Device)
    : Tensor { device := device, shape := #[1, 1], dtype := .Int64 } :=
  let idsVals : Array Int64 := #[Int64.ofNat id.toNat]
  let inputIdsCpu : Tensor { device := .CPU, shape := #[1, 1], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.fromInt64Array idsVals) #[1, 1]
  inputIdsCpu.to device

private def isForbiddenToken (rules : DecodeTokenRules) (id : UInt32) : Bool :=
  rules.forbidden.contains id

private def applyInitialTimestampCap
    {cfg : WhisperConfig}
    (rules : DecodeTokenRules)
    : DecodeTokenRules :=
  match rules.timestampBeginId with
  | none => rules
  | some ts =>
      if rules.maxInitialTimestampTokens == 0 then
        rules
      else
        let start := ts.toNat + rules.maxInitialTimestampTokens + 1
        if start >= cfg.vocabSize.toNat then
          rules
        else
          Id.run do
            let mut forb := rules.forbidden
            for i in [start:cfg.vocabSize.toNat] do
              forb := pushUniqueId forb i.toUInt32
            { rules with forbidden := forb }

private def applyTimestampPairMask
    {cfg : WhisperConfig}
    (rules : DecodeTokenRules)
    (generated : Array UInt32)
    (eosTokenId : UInt64)
    : DecodeTokenRules :=
  match rules.timestampBeginId with
  | none => rules
  | some tsBegin =>
      if generated.isEmpty then
        rules
      else
        let last := generated[generated.size - 1]!
        if last.toUInt64 < tsBegin.toUInt64 then
          rules
        else
          let penultimateWasTimestamp :=
            if generated.size < 2 then
              true
            else
              generated[generated.size - 2]!.toUInt64 >= tsBegin.toUInt64
          Id.run do
            let mut forb := rules.forbidden
            if penultimateWasTimestamp then
              -- Two consecutive timestamps are not allowed; force next token to text (or EOT).
              for i in [tsBegin.toNat:cfg.vocabSize.toNat] do
                forb := pushUniqueId forb i.toUInt32
            else
              -- After text->timestamp, next must be timestamp (or EOT).
              for i in [:tsBegin.toNat] do
                let tok := i.toUInt32
                if tok.toUInt64 != eosTokenId then
                  forb := pushUniqueId forb tok
            { rules with forbidden := forb }

private def logProbsFromLogitsArray (logits : Array Float) : Array Float :=
  if logits.isEmpty then
    #[]
  else
    let maxLogit := logits.foldl (fun acc x => if x > acc then x else acc) logits[0]!
    let sumExp := logits.foldl (fun acc x => acc + Float.exp (x - maxLogit)) 0.0
    let logDen :=
      if sumExp <= 0.0 then
        maxLogit
      else
        maxLogit + Float.log sumExp
    logits.map (fun x => x - logDen)

private def topKLogProbPairs (logProbs : Array Float) (k : Nat) : Array (Nat × Float) :=
  if k == 0 || logProbs.isEmpty then
    #[]
  else
    Id.run do
      let mut pairs : Array (Nat × Float) := #[]
      for i in [:logProbs.size] do
        pairs := pairs.push (i, logProbs[i]!)
      let sorted := pairs.qsort (fun a b => a.2 > b.2)
      sorted.extract 0 (Nat.min k sorted.size)

private def tokenLogProbFromLogits {vocab : UInt64}
    (logits : Tensor { device := device, shape := #[1, vocab] })
    (tokenId : UInt32)
    : IO Float := do
  let vals ← Tensor.toFloatArray logits
  let logProbs := logProbsFromLogitsArray vals
  pure (logProbs.getD tokenId.toNat (-100.0))

private def tokenProbFromLogits {vocab : UInt64}
    (logits : Tensor { device := device, shape := #[1, vocab] })
    (tokenId : UInt32)
    : IO Float := do
  let vals ← Tensor.toFloatArray logits
  let logProbs := logProbsFromLogitsArray vals
  pure (Float.exp (logProbs.getD tokenId.toNat (-100.0)))

private def clampTopK (topK vocab : UInt64) : UInt64 :=
  if topK == 0 then
    0
  else if topK > vocab then
    vocab
  else
    topK

private def filteredSamplingLogits {vocab : UInt64}
    (logits : Tensor { device := device, shape := #[1, vocab] })
    (temperature : Float)
    (topK : UInt64)
    (topP : Float)
    : Tensor { device := device, shape := #[1, vocab] } :=
  let scaled :=
    if temperature == 1.0 then
      logits
    else
      Tensor.Preserving.mulScalar logits (1.0 / temperature)
  let topK' := clampTopK topK vocab
  let filtered :=
    if topK' == 0 then
      scaled
    else
      Tensor.topKFilter scaled topK'
  let topP' :=
    if topP <= 0.0 then
      (1e-6 : Float)
    else if topP > 1.0 then
      1.0
    else
      topP
  if topP' >= 1.0 then
    filtered
  else
    Tensor.topPFilter filtered topP'

private def argmaxTokenFromLogits {vocab : UInt64}
    (logits : Tensor { device := device, shape := #[1, vocab] })
    (fallback : UInt64)
    (rules : DecodeTokenRules := {})
    : IO UInt32 := do
  let kNat := Nat.max 1 (Nat.min 64 vocab.toNat)
  let logProbs := logProbsFromLogitsArray (← Tensor.toFloatArray logits)
  let topPairs := topKLogProbPairs logProbs kNat
  for p in topPairs do
    let id := p.1.toUInt32
    if !(isForbiddenToken rules id) then
      return id
  let indices ← IO.ofExcept (Tensor.argmax? logits 1)
  let nextTokRaw : Tensor { device := device, shape := #[1], dtype := .Int64 } ← IO.ofExcept (indices.reshapeChecked? #[1])
  let nextVals ← Tensor.toUInt64Array nextTokRaw
  pure (nextVals.getD 0 fallback).toUInt32

private def sampleTokenFromLogits {vocab : UInt64}
    (logits : Tensor { device := device, shape := #[1, vocab] })
    (rules : DecodeTokenRules := {})
    (fallback : UInt64 := 0)
    : IO UInt32 := do
  let vals ← Tensor.toFloatArray logits
  let probsVals := (logProbsFromLogitsArray vals).map Float.exp
  let probs : Tensor { device := device, shape := #[1, vocab], dtype := .Float32 } := Tensor.reshapeChecked ((Tensor.fromFloatArray probsVals).to device) #[1, vocab]
  let mut tries : Nat := 0
  while tries < 8 do
    let sampled ← Tensor.multinomial probs 1 false
    let sampledFlat : Tensor { device := device, shape := #[1], dtype := .Int64 } := Tensor.reshapeChecked sampled #[1]
    let sampledVals ← Tensor.toUInt64Array sampledFlat
    let id := (sampledVals.getD 0 0).toUInt32
    if !isForbiddenToken rules id then
      return id
    tries := tries + 1
  argmaxTokenFromLogits logits fallback rules

private def prefillPromptCaches
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration (device := device) cfg)
    (crossCaches : Array (WhisperForConditionalGeneration.LayerCrossCache (device := device) cfg))
    (promptIds : Array UInt32)
    : IO (Tensor { device := device, shape := #[1, cfg.vocabSize] } × Array (WhisperForConditionalGeneration.LayerKVCache (device := device) cfg 1) × UInt64) := do
  if promptIds.isEmpty then
    throw <| IO.userError "Whisper cached decode prompt is empty"
  let mut caches : Array (WhisperForConditionalGeneration.LayerKVCache (device := device) cfg 1) :=
    model.initLayerKVCaches cfg.maxTargetPositions device
  let mut last? : Option (Tensor { device := device, shape := #[1, cfg.vocabSize] }) := none
  let mut pos : UInt64 := 0
  for i in [:promptIds.size] do
    let tok := buildSingleTokenOnDevice promptIds[i]! device
    let (logits, caches') ←
      model.decodeStepWithCache (encSeq := encSeq) tok i.toUInt64 crossCaches caches
    caches := caches'
    last? := some logits
    pos := i.toUInt64 + 1
  match last? with
  | some last => pure (last, caches, pos)
  | none => throw <| IO.userError "Whisper cached decode failed to prefill prompt"

private partial def decodeLoopCached
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration (device := device) cfg)
    (crossCaches : Array (WhisperForConditionalGeneration.LayerCrossCache (device := device) cfg))
    (noSpeechTokenId : Option UInt32)
    (rules : DecodeTokenRules)
    (eosTokenId : UInt64)
    (temperature : Float)
    (topK : UInt64)
    (topP : Float)
    (sample : Bool)
    (remaining : Nat)
    (state : DecodeState)
    (caches : Array (WhisperForConditionalGeneration.LayerKVCache (device := device) cfg 1))
    (lastLogits : Tensor { device := device, shape := #[1, cfg.vocabSize] })
    (nextPos : UInt64)
    : IO DecodeState := do
  if remaining == 0 then
    pure state
  else
    let noSpeechProb ←
      if state.measuredNoSpeech then
        pure state.noSpeechProb
      else
        match noSpeechTokenId with
        | none => pure 0.0
        | some tid => tokenProbFromLogits lastLogits tid
    let distLogits :=
      if sample then
        filteredSamplingLogits lastLogits temperature topK topP
      else
        lastLogits
    let stepRules :=
      if state.tokenCount == 0 then
        let forb := (rules.forbidden ++ rules.beginSuppress).push eosTokenId.toUInt32
        applyInitialTimestampCap (cfg := cfg) { rules with forbidden := forb, beginSuppress := #[] }
      else
        rules
    let stepRules := applyTimestampPairMask (cfg := cfg) stepRules state.generated eosTokenId
    let nextId ←
      if sample then
        sampleTokenFromLogits distLogits stepRules eosTokenId
      else
        argmaxTokenFromLogits lastLogits eosTokenId stepRules
    let tokLogprob ← tokenLogProbFromLogits distLogits nextId
    let nextIdU64 := nextId.toUInt64
    let generated' :=
      if nextIdU64 == eosTokenId then state.generated else state.generated.push nextId
    let tokenCount' :=
      if nextIdU64 == eosTokenId then state.tokenCount else state.tokenCount + 1
    let sumLogprob' :=
      if nextIdU64 == eosTokenId then state.sumLogprob else state.sumLogprob + tokLogprob
    let state' : DecodeState := {
      allIds := state.allIds.push nextId
      generated := generated'
      sumLogprob := sumLogprob'
      tokenCount := tokenCount'
      noSpeechProb := noSpeechProb
      measuredNoSpeech := true
    }
    if nextIdU64 == eosTokenId then
      pure state'
    else
      let tokenIds := buildSingleTokenOnDevice nextId lastLogits.device
      let (nextLogits, caches') ←
        model.decodeStepWithCache (encSeq := encSeq) tokenIds nextPos crossCaches caches
      decodeLoopCached
        (encSeq := encSeq)
        model
        crossCaches
        noSpeechTokenId
        rules
        eosTokenId
        temperature
        topK
        topP
        sample
        (remaining - 1)
        state'
        caches'
        nextLogits
        (nextPos + 1)

private def beamRankScore {cfg : WhisperConfig} (beam : BeamState (device := device) cfg) : Float :=
  if beam.tokenCount == 0 then
    -100.0
  else
    beam.sumLogprob / beam.tokenCount.toFloat

private def chooseBetterBeam {cfg : WhisperConfig} (a b : BeamState (device := device) cfg) : BeamState (device := device) cfg :=
  if beamRankScore b > beamRankScore a then b else a

private def expandBeamCached
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration (device := device) cfg)
    (crossCaches : Array (WhisperForConditionalGeneration.LayerCrossCache (device := device) cfg))
    (noSpeechTokenId : Option UInt32)
    (rules : DecodeTokenRules)
    (eosTokenId : UInt64)
    (beamWidth : UInt64)
    (beam : BeamState (device := device) cfg)
    : IO (Array (BeamState (device := device) cfg)) := do
  if beam.finished then
    pure #[beam]
  else
    match beam.lastLogits? with
    | none => pure #[{ beam with finished := true }]
    | some last =>
      let noSpeechProb ←
        if beam.measuredNoSpeech then
          pure beam.noSpeechProb
        else
          match noSpeechTokenId with
          | none => pure 0.0
          | some tid => tokenProbFromLogits last tid
      let kNat := Nat.max 1 (Nat.min beamWidth.toNat cfg.vocabSize.toNat)
      let logitsArr ← Tensor.toFloatArray last
      let logProbs := logProbsFromLogitsArray logitsArr
      let topPairs := topKLogProbPairs logProbs kNat
      let stepRules :=
        if beam.tokenCount == 0 then
          let forb := (rules.forbidden ++ rules.beginSuppress).push eosTokenId.toUInt32
          applyInitialTimestampCap (cfg := cfg) { rules with forbidden := forb, beginSuppress := #[] }
        else
          rules
      let stepRules := applyTimestampPairMask (cfg := cfg) stepRules beam.generated eosTokenId
      let mut out : Array (BeamState (device := device) cfg) := #[]
      for p in topPairs do
        let nextIdU64 := p.1.toUInt64
        let nextId := nextIdU64.toUInt32
        if isForbiddenToken stepRules nextId then
          continue
        let lp := p.2
        let generated' :=
          if nextIdU64 == eosTokenId then beam.generated else beam.generated.push nextId
        let tokenCount' :=
          if nextIdU64 == eosTokenId then beam.tokenCount else beam.tokenCount + 1
        let sumLogprob' :=
          if nextIdU64 == eosTokenId then beam.sumLogprob else beam.sumLogprob + lp
        if nextIdU64 == eosTokenId then
          out := out.push {
            allIds := beam.allIds.push nextId
            generated := generated'
            sumLogprob := sumLogprob'
            tokenCount := tokenCount'
            noSpeechProb := noSpeechProb
            measuredNoSpeech := true
            finished := true
            caches := beam.caches
            lastLogits? := none
            nextPos := beam.nextPos
          }
        else
          let tokenIds := buildSingleTokenOnDevice nextId last.device
          let (nextLogits, caches') ←
            model.decodeStepWithCache (encSeq := encSeq) tokenIds beam.nextPos crossCaches beam.caches
          out := out.push {
            allIds := beam.allIds.push nextId
            generated := generated'
            sumLogprob := sumLogprob'
            tokenCount := tokenCount'
            noSpeechProb := noSpeechProb
            measuredNoSpeech := true
            finished := false
            caches := caches'
            lastLogits? := some nextLogits
            nextPos := beam.nextPos + 1
          }
      pure out

private partial def beamDecodeLoopCached
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration (device := device) cfg)
    (crossCaches : Array (WhisperForConditionalGeneration.LayerCrossCache (device := device) cfg))
    (noSpeechTokenId : Option UInt32)
    (rules : DecodeTokenRules)
    (eosTokenId : UInt64)
    (beamWidth : UInt64)
    (remaining : Nat)
    (beams : Array (BeamState (device := device) cfg))
    : IO (Array (BeamState (device := device) cfg)) := do
  if remaining == 0 then
    pure beams
  else
    let mut allFinished := true
    for beam in beams do
      if !beam.finished then
        allFinished := false
    if allFinished then
      pure beams
    else
      let mut expanded : Array (BeamState (device := device) cfg) := #[]
      for beam in beams do
        if beam.finished then
          expanded := expanded.push beam
        else
          expanded := expanded ++
            (← expandBeamCached
              (encSeq := encSeq)
              model
              crossCaches
              noSpeechTokenId
              rules
              eosTokenId
              beamWidth
              beam)
      if expanded.isEmpty then
        pure beams
      else
        let sorted := expanded.qsort (fun a b => beamRankScore a > beamRankScore b)
        let keep := Nat.max 1 (Nat.min sorted.size beamWidth.toNat)
        beamDecodeLoopCached
          (encSeq := encSeq)
          model
          crossCaches
          noSpeechTokenId
          rules
          eosTokenId
          beamWidth
          (remaining - 1)
          (sorted.extract 0 keep)

private def decodeBeamCached
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration (device := device) cfg)
    (_encoderHidden : Tensor { device := device, shape := #[1, encSeq, cfg.dModel] })
    (crossCaches : Array (WhisperForConditionalGeneration.LayerCrossCache (device := device) cfg))
    (promptIds : Array UInt32)
    (noSpeechTokenId : Option UInt32)
    (rules : DecodeTokenRules)
    (eosTokenId : UInt64)
    (beamWidth : UInt64)
    (remaining : Nat)
    : IO DecodeState := do
  let (firstLogits, caches0, nextPos0) ←
    prefillPromptCaches (encSeq := encSeq) model crossCaches promptIds
  let initBeam : BeamState (device := device) cfg := {
    allIds := promptIds
    caches := caches0
    lastLogits? := some firstLogits
    nextPos := nextPos0
  }
  let beams ←
    beamDecodeLoopCached
      (encSeq := encSeq)
      model
      crossCaches
      noSpeechTokenId
      rules
      eosTokenId
      beamWidth
      remaining
      #[initBeam]
  if beams.isEmpty then
    pure { allIds := promptIds }
  else
    let mut best := beams[0]!
    for beam in beams do
      best := chooseBetterBeam best beam
    pure {
      allIds := best.allIds
      generated := best.generated
      sumLogprob := best.sumLogprob
      tokenCount := best.tokenCount
      noSpeechProb := best.noSpeechProb
      measuredNoSpeech := best.measuredNoSpeech
    }

private def toDecodeCandidate
    (tok : tokenizer.qwen3.QwenTokenizer)
    (st : DecodeState)
    (temperature : Float)
    : WhisperDecodeCandidate :=
  let avg :=
    if st.tokenCount == 0 then
      -100.0
    else
      st.sumLogprob / st.tokenCount.toFloat
  let text := decodeGeneratedText tok st.generated
  let ratio := approxCompressionRatio st.generated
  let repetition := hasRepetitionLoop st.generated
  {
    generated := st.generated
    text := text
    avgLogprob := avg
    noSpeechProb := st.noSpeechProb
    compressionRatio := ratio
    repetitionLoop := repetition
    temperature := temperature
  }

private def chooseBetterCandidate (a b : WhisperDecodeCandidate) : WhisperDecodeCandidate :=
  if b.avgLogprob > a.avgLogprob then b else a

private def decodeAtTemperature
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration (device := device) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (encoderHidden : Tensor { device := device, shape := #[1, encSeq, cfg.dModel] })
    (crossCaches : Array (WhisperForConditionalGeneration.LayerCrossCache (device := device) cfg))
    (promptIds : Array UInt32)
    (noSpeechTokenId : Option UInt32)
    (rules : DecodeTokenRules)
    (eosTokenId : UInt64)
    (maxNewTokens : Nat)
    (opts : WhisperDecodeOptions)
    (temperature : Float)
    : IO WhisperDecodeCandidate := do
  let (firstLogits, caches0, nextPos0) ←
    prefillPromptCaches (encSeq := encSeq) model crossCaches promptIds
  if temperature <= 0.0 then
    let st ←
      if opts.beamSize > 1 then
        decodeBeamCached
          model
          encoderHidden
          crossCaches
          promptIds
          noSpeechTokenId
          rules
          eosTokenId
          opts.beamSize
          maxNewTokens
      else
        decodeLoopCached
          (encSeq := encSeq)
          model
          crossCaches
          noSpeechTokenId
          rules
          eosTokenId
          0.0
          opts.topK
          opts.topP
          false
          maxNewTokens
          { allIds := promptIds }
          caches0
          firstLogits
          nextPos0
    pure (toDecodeCandidate tok st temperature)
  else
    let runs := Nat.max 1 opts.bestOf.toNat
    let mut best? : Option WhisperDecodeCandidate := none
    for _ in [:runs] do
      let st ←
        decodeLoopCached
          (encSeq := encSeq)
          model
          crossCaches
          noSpeechTokenId
          rules
          eosTokenId
          temperature
          opts.topK
          opts.topP
          true
          maxNewTokens
          { allIds := promptIds }
          caches0
          firstLogits
          nextPos0
      let cand := toDecodeCandidate tok st temperature
      best? :=
        match best? with
        | none => some cand
        | some cur => some (chooseBetterCandidate cur cand)
    pure ((best?).getD (default : WhisperDecodeCandidate))

private partial def decodeWithFallbackLoop
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration (device := device) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (encoderHidden : Tensor { device := device, shape := #[1, encSeq, cfg.dModel] })
    (crossCaches : Array (WhisperForConditionalGeneration.LayerCrossCache (device := device) cfg))
    (promptIds : Array UInt32)
    (promptIdsFallback : Array UInt32)
    (noSpeechTokenId : Option UInt32)
    (rules : DecodeTokenRules)
    (eosTokenId : UInt64)
    (maxNewTokens : Nat)
    (opts : WhisperDecodeOptions)
    (temperatures : Array Float)
    (idx : Nat)
    (usedFallback : Bool)
    : IO (WhisperDecodeCandidate × Bool) := do
  let t := temperatures.getD idx opts.temperature
  let cand ←
    decodeAtTemperature
      model
      tok
      encoderHidden
      crossCaches
      (if idx > 0 && opts.resetContextOnFallback then promptIdsFallback else promptIds)
      noSpeechTokenId
      rules
      eosTokenId
      maxNewTokens
      opts
      t
  let isLast := idx + 1 >= temperatures.size
  if isLast then
    pure (cand, usedFallback)
  else if candidateNeedsFallback opts cand then
    decodeWithFallbackLoop
      model
      tok
      encoderHidden
      crossCaches
      promptIds
      promptIdsFallback
      noSpeechTokenId
      rules
      eosTokenId
      maxNewTokens
      opts
      temperatures
      (idx + 1)
      true
  else
    pure (cand, usedFallback)

private def decodeWithFallback
    {cfg : WhisperConfig}
    {encSeq : UInt64}
    (model : WhisperForConditionalGeneration (device := device) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (encoderHidden : Tensor { device := device, shape := #[1, encSeq, cfg.dModel] })
    (promptIds : Array UInt32)
    (promptIdsFallback : Array UInt32)
    (noSpeechTokenId : Option UInt32)
    (rules : DecodeTokenRules)
    (eosTokenId : UInt64)
    (maxNewTokens : Nat)
    (opts : WhisperDecodeOptions)
    : IO (WhisperDecodeCandidate × Bool) := do
  let crossCaches ← model.precomputeCrossCaches encoderHidden
  let temperatures := decodeTemperatures opts
  decodeWithFallbackLoop
    model
    tok
    encoderHidden
    crossCaches
    promptIds
    promptIdsFallback
    noSpeechTokenId
    rules
    eosTokenId
    maxNewTokens
    opts
    temperatures
    0
    false

private def maybeAppendTailRescue
    {cfg : WhisperConfig}
    (model : WhisperForConditionalGeneration (device := device) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (pre : PreprocessorConfig)
    (wav16k : Array Float)
    (basePrompt : Array UInt32)
    (noSpeechTokenId : Option UInt32)
    (rules : DecodeTokenRules)
    (decodeOpts : WhisperDecodeOptions)
    (chunkSeconds : Float)
    (requestedMaxNew : Nat)
    (maxTarget : Nat)
    (currentText : String)
    : IO String := do
  if wav16k.isEmpty then
    pure currentText
  else
    let targetSamples := ((chunkSeconds * pre.samplingRate.toFloat) + 0.5).toUInt64.toNat
    let tailSpan := Nat.max 1 (Nat.min wav16k.size targetSamples)
    let tail := wav16k.extract (wav16k.size - tailSpan) wav16k.size
    let pack ←
      waveformToWhisperFeaturesDynamic
        pre
        tail
        (minSeconds := 0.05)
        (maxSeconds := some chunkSeconds)
        (device := model.model.decoderTokenEmbedding.device)
    match pack with
    | ⟨frames, out⟩ =>
      let inputFeatures : Tensor { device := device, shape := #[1, cfg.numMelBins, frames] } :=
        (Tensor.reshapeChecked out.inputFeatures #[1, cfg.numMelBins, frames]).forgetDType
      let encoderHidden : Tensor { device := device, shape := #[1, WhisperConfig.conv2OutputSeq frames, cfg.dModel] } ←
        model.encode (frames := frames) inputFeatures
      let maxNewCapPrompt :=
        if maxTarget <= basePrompt.size + 1 then
          1
        else
          maxTarget - basePrompt.size - 1
      let maxNewNat := Nat.max 1 (Nat.min requestedMaxNew maxNewCapPrompt)
      let rescueOpts : WhisperDecodeOptions := {
        decodeOpts with
          beamSize := if decodeOpts.beamSize < 5 then 5 else decodeOpts.beamSize
          bestOf := if decodeOpts.bestOf < 5 then 5 else decodeOpts.bestOf
          noFallback := false
          conditionOnPreviousText := false
          maxContextTokens := 0
      }
      let (cand, _) ←
        decodeWithFallback
          model
          tok
          encoderHidden
          basePrompt
          basePrompt
          noSpeechTokenId
          rules
          cfg.eosTokenId
          maxNewNat
          rescueOpts
      let tailText := cand.text.trimAscii.toString
      if tailText.isEmpty then
        pure currentText
      else if cand.avgLogprob < (-2.5) then
        pure currentText
      else
        let cur := currentText.trimAscii.toString
        let curLower := cur.toLower
        let tailLower := tailText.toLower
        if curLower.contains tailLower then
          pure cur
        else
          pure ((cur ++ " " ++ tailText).trimAscii.toString)

def loadFromPretrainedDir (modelDir : String) : IO WhisperBundle := do
  let cfg ← WhisperConfig.loadFromPretrainedDir modelDir {}
  let tok ← tokenizer.qwen3.loadTokenizer modelDir
  let preprocessor ←
    PreprocessorConfig.loadFromPretrainedDir modelDir {
      featureSize := cfg.numMelBins
      samplingRate := 16000
      hopLength := 160
      chunkLength := 30
      nFft := 400
      returnAttentionMask := true
      doNormalize := false
      dither := 0.0
    }
  let ⟨device, model⟩ ← WhisperForConditionalGeneration.loadSharded modelDir cfg
  pure { cfg, device, model, tok, preprocessor }

private def transcribeWavCore
    {cfg : WhisperConfig}
    (model : WhisperForConditionalGeneration (device := device) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (pre : PreprocessorConfig)
    (wav16k : Array Float)
    (language : String)
    (maxNewTokens : UInt64)
    (noTimestamps : Bool)
    (decodeOpts : WhisperDecodeOptions)
    : IO WhisperTranscription := do
  let frontendDevice := model.model.decoderTokenEmbedding.device
  let basePrompt := buildPromptIds cfg tok language noTimestamps
  let rules := buildDecodeTokenRules tok cfg noTimestamps
  let maxTarget := cfg.maxTargetPositions.toNat
  if maxTarget <= basePrompt.size + 1 then
    throw <| IO.userError
      s!"Whisper max_target_positions={cfg.maxTargetPositions} is too small for base prompt size={basePrompt.size}"
  let requestedMaxNew := if maxNewTokens == 0 then maxTarget else maxNewTokens.toNat
  let modelContextLimit := maxTarget / 2
  let userContextLimit :=
    if decodeOpts.maxContextTokens == 0 then
      modelContextLimit
    else
      Nat.min decodeOpts.maxContextTokens.toNat modelContextLimit
  let contextLimit := userContextLimit
  let promptExtraBudget := if decodeOpts.conditionOnPreviousText then contextLimit + 1 else 0
  let promptMaxLen := Nat.min maxTarget (basePrompt.size + promptExtraBudget)
  let noSpeechTokenId := tokenIdByText? tok "<|nospeech|>"
  let prevTokenId := tokenIdByText? tok "<|startofprev|>"
  let cfgCapSec := whisperMaxChunkSeconds cfg pre
  let preChunkSec :=
    if pre.chunkLength == 0 then
      cfgCapSec
    else
      pre.chunkLength.toFloat
  let chunkSeconds :=
    if preChunkSec <= 0.0 then
      cfgCapSec
    else
      if preChunkSec <= cfgCapSec then preChunkSec else cfgCapSec
  let sr := if pre.samplingRate == 0 then sampleRate16k else pre.samplingRate
  let chunkSamplesRaw := ((chunkSeconds * sr.toFloat) + 0.5).toUInt64.toNat
  let chunkSamples := if chunkSamplesRaw == 0 then wav16k.size else chunkSamplesRaw
  let overlapSec :=
    if decodeOpts.chunkOverlapSeconds <= 0.0 then
      0.0
    else
      decodeOpts.chunkOverlapSeconds
  let overlapSamplesRaw := ((overlapSec * sr.toFloat) + 0.5).toUInt64.toNat
  let overlapSamples :=
    if chunkSamples <= 1 then
      0
    else
      Nat.min overlapSamplesRaw (chunkSamples - 1)
  let defaultAdvance := Nat.max 1 (chunkSamples - overlapSamples)
  let timestampBeginId := if noTimestamps then none else tokenIdByText? tok "<|0.00|>"
  let samplesPerTimestamp := Nat.max 1 (((0.02 * sr.toFloat) + 0.5).toUInt64.toNat)

  let mut rollingContext : Array UInt32 := #[]
  let mut texts : Array String := #[]
  let mut allTokenIds : Array UInt32 := #[]
  let mut seek : Nat := 0
  let maxSeekSteps :=
    if noTimestamps then
      (wav16k.size / defaultAdvance) + 2
    else
      (wav16k.size / (Nat.max 1 (sr.toNat * 5))) + 4
  let mut seekStep : Nat := 0
  while seek < wav16k.size && seekStep < maxSeekSteps do
    seekStep := seekStep + 1
    let stop := Nat.min wav16k.size (seek + chunkSamples)
    let chunk := wav16k.extract seek stop
    let tailNewSamples :=
      if chunk.size > overlapSamples then
        chunk.size - overlapSamples
      else
        0
    let skipShortTail :=
      noTimestamps &&
      seek > 0 &&
      stop == wav16k.size &&
      -- Previous overlapped chunk already covered almost everything; decoding <=1s of novel tail
      -- often adds hallucinated suffixes.
      tailNewSamples <= sr.toNat
    if skipShortTail then
      seek := wav16k.size
    else
      if seek > 0 && stop == wav16k.size && chunk.size < (chunkSamples / 2) then
        -- Very short tail chunks tend to hallucinate with carried text context.
        rollingContext := #[]

      let pack ←
        waveformToWhisperFeaturesDynamic
          pre
          chunk
          (minSeconds := 0.05)
          (maxSeconds := some chunkSeconds)
          (device := frontendDevice)
      match pack with
      | ⟨frames, out⟩ =>
        let inputFeatures : Tensor { device := device, shape := #[1, cfg.numMelBins, frames] } :=
          (Tensor.reshapeChecked out.inputFeatures #[1, cfg.numMelBins, frames]).forgetDType
        let encoderHidden : Tensor { device := device, shape := #[1, WhisperConfig.conv2OutputSeq frames, cfg.dModel] } ←
          model.encode (frames := frames) inputFeatures
        let promptIds :=
          composePromptIds
            basePrompt
            rollingContext
            promptMaxLen
            contextLimit
            decodeOpts.conditionOnPreviousText
            prevTokenId
        let promptIdsFallback :=
          if decodeOpts.resetContextOnFallback then basePrompt else promptIds
        let maxNewCapPrompt :=
          if maxTarget <= promptIds.size + 1 then
            1
          else
            maxTarget - promptIds.size - 1
        let maxNewNatChunk := Nat.max 1 (Nat.min requestedMaxNew maxNewCapPrompt)
        let (cand, usedFallback) ←
          decodeWithFallback
            model
            tok
            encoderHidden
            promptIds
            promptIdsFallback
            noSpeechTokenId
            rules
            cfg.eosTokenId
            maxNewNatChunk
            decodeOpts
        let silent := isNoSpeechChunk decodeOpts cand
        if !silent then
          let text := cand.text.trimAscii.toString
          if !text.isEmpty then
            texts := texts.push text
          allTokenIds := appendWithTokenStitch allTokenIds cand.generated
        if decodeOpts.conditionOnPreviousText then
          if silent then
            pure ()
          else if decodeOpts.resetContextOnFallback && (usedFallback || cand.temperature > 0.5) then
            rollingContext := #[]
          else
            rollingContext := takeLast (rollingContext ++ cand.generated) contextLimit
        let mut advance := Nat.min chunk.size defaultAdvance
        if !noTimestamps then
          -- Keep runtime bounded while preserving overlap when explicit timestamp advance is absent.
          let minTimestampAdvance := Nat.max 1 (Nat.min chunk.size (sr.toNat * 5))
          let tsAdvance :=
            timestampAdvanceSamples
              cand.generated
              timestampBeginId
              samplesPerTimestamp
              chunk.size
          if tsAdvance > 0 then
            advance := Nat.max minTimestampAdvance tsAdvance
          else
            advance := Nat.max minTimestampAdvance (Nat.min chunk.size defaultAdvance)
          if cand.generated.size > 1 then
            let last := cand.generated[cand.generated.size - 1]!
            let prev := cand.generated[cand.generated.size - 2]!
            if isTimestampTokenId tok last && !(isTimestampTokenId tok prev) then
              advance := chunk.size
        if advance == 0 then
          seek := Nat.min wav16k.size (seek + 1)
        else
          seek := Nat.min wav16k.size (seek + advance)

  let stitchedText := decodeGeneratedText tok allTokenIds
  let mergedText := mergeChunkTexts texts
  let finalTextRaw := if stitchedText.isEmpty then mergedText else stitchedText
  let finalText0 := trimDanglingTailAfterPeriod finalTextRaw
  let finalText ←
    if noTimestamps then
      maybeAppendTailRescue
        model
        tok
        pre
        wav16k
        basePrompt
        noSpeechTokenId
        rules
        decodeOpts
        chunkSeconds
        requestedMaxNew
        maxTarget
        finalText0
    else
      pure finalText0
  pure {
    language := normalizeLanguageCode language
    text := finalText
    tokenIds := allTokenIds
  }

def transcribeWaveform16k
    {cfg : WhisperConfig}
    (model : WhisperForConditionalGeneration (device := device) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (pre : PreprocessorConfig)
    (wav16k : Array Float)
    (language : String := "en")
    (maxNewTokens : UInt64 := 128)
    (noTimestamps : Bool := true)
    (decodeOpts : WhisperDecodeOptions := {})
    : IO WhisperTranscription :=
  transcribeWavCore model tok pre wav16k language maxNewTokens noTimestamps decodeOpts

def transcribeWav
    {cfg : WhisperConfig}
    (model : WhisperForConditionalGeneration (device := device) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (pre : PreprocessorConfig)
    (wavPath : String)
    (language : String := "en")
    (maxNewTokens : UInt64 := 128)
    (noTimestamps : Bool := true)
    (decodeOpts : WhisperDecodeOptions := {})
    : IO WhisperTranscription := do
  let wav16k ← normalizeAudioTo16kFromWav wavPath
  transcribeWavCore model tok pre wav16k language maxNewTokens noTimestamps decodeOpts

end torch.whisper
