/-  Tyr/Model/Qwen3ASR/Transcribe.lean

  Lean-native offline ASR orchestration:
  - build prompt
  - run frontend + generation
  - parse ASR output
  - optional forced-aligner timestamps
-/
import Tyr.Model.Qwen3ASR.Config
import Tyr.Model.Qwen3ASR.Model
import Tyr.Model.Qwen3ASR.Frontend
import Tyr.Model.Qwen3ASR.Processor
import Tyr.Model.Qwen3ASR.ForcedAligner
import Tyr.Model.Qwen3ASR.Streaming
import Tyr.Model.Qwen3ASR.Pretrained
import Tyr.Tokenizer.Qwen3

namespace torch.qwen3asr

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
/-- One offline ASR transcription result. -/
structure ASRTranscription where
  language : String
  text : String
  timeStamps : Option ForcedAlignResult := none
  deriving Repr, Inhabited

/-- Standalone forced-aligner wrapper:
    independent model/tokenizer checkpoint that can be attached to ASR transcription. -/
structure Qwen3ForcedAligner where
  cfg : Qwen3ASRConfig
  device : Device
  dtype : RealFloatingDType
  model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype.val) cfg
  tok : tokenizer.qwen3.QwenTokenizer
  preprocessor : PreprocessorConfig

private def sampleRate16k : UInt64 := 16000
private def minAsrInputSeconds : Float := 0.5
private def maxAsrInputSeconds : Float := 1200.0
private def maxForceAlignInputSeconds : Float := 180.0

private def padRightWithZeros (xs : Array Float) (target : Nat) : Array Float :=
  if xs.size >= target then
    xs
  else
    Id.run do
      let mut out := xs
      let pad := target - xs.size
      for _ in [:pad] do
        out := out.push 0.0
      out

private def argminFloatArray (xs : Array Float) : Nat :=
  if xs.isEmpty then
    0
  else
    Id.run do
      let mut bestIdx : Nat := 0
      let mut bestVal := xs[0]!
      for i in [1:xs.size] do
        let v := xs[i]!
        if v < bestVal then
          bestVal := v
          bestIdx := i
      bestIdx

private def slidingWindowArgminAbs (xs : Array Float) (win : Nat) : Nat :=
  if xs.isEmpty || win == 0 || xs.size <= win then
    0
  else
    Id.run do
      let mut cur : Float := 0.0
      for i in [:win] do
        cur := cur + Float.abs (xs[i]!)
      let mut bestVal := cur
      let mut bestPos : Nat := 0
      let limit := xs.size - win + 1
      for pos in [1:limit] do
        let dropV := Float.abs (xs[pos - 1]!)
        let addV := Float.abs (xs[pos + win - 1]!)
        cur := cur - dropV + addV
        if cur < bestVal then
          bestVal := cur
          bestPos := pos
      bestPos

private def splitAudioIntoChunks
    (wav : Array Float)
    (sr : UInt64)
    (maxChunkSec : Float)
    (searchExpandSec : Float := 5.0)
    (minWindowMs : Float := 100.0)
    : Array (Array Float × Float) :=
  let wav := wav
  let srNat := if sr == 0 then sampleRate16k.toNat else sr.toNat
  if srNat == 0 then
    #[(wav, 0.0)]
  else
    let totalLen := wav.size
    let totalSec := totalLen.toFloat / srNat.toFloat
    if totalSec <= maxChunkSec then
      #[(wav, 0.0)]
    else
      Id.run do
        let maxLenRaw := ((maxChunkSec * srNat.toFloat) + 0.5).toUInt64.toNat
        let maxLen := if maxLenRaw == 0 then 1 else maxLenRaw
        let expandRaw := ((searchExpandSec * srNat.toFloat) + 0.5).toUInt64.toNat
        let expand := expandRaw
        let winRaw := (((minWindowMs / 1000.0) * srNat.toFloat) + 0.5).toUInt64.toNat
        let win := Nat.max 4 winRaw

        let mut chunks : Array (Array Float × Float) := #[]
        let mut start : Nat := 0
        let mut offsetSec : Float := 0.0

        while (totalLen - start) > maxLen do
          let cut := start + maxLen
          let left := if expand == 0 then cut else Nat.max start (cut - expand)
          let right := if expand == 0 then cut else Nat.min totalLen (cut + expand)
          let boundary0 :=
            if expand == 0 || right <= left || right - left <= win then
              cut
            else
              Id.run do
                let seg := wav.extract left right
                let minPos := slidingWindowArgminAbs seg win
                let wstart := minPos
                let wend := minPos + win
                let localAbs := (seg.extract wstart wend).map (fun x => Float.abs x)
                pure (left + wstart + argminFloatArray localAbs)

          let boundary1 := Nat.max boundary0 (start + 1)
          let boundary := Nat.min boundary1 totalLen
          let chunk := wav.extract start boundary
          chunks := chunks.push (chunk, offsetSec)
          offsetSec := offsetSec + (boundary - start).toFloat / srNat.toFloat
          start := boundary

        let tail := wav.extract start totalLen
        chunks := chunks.push (tail, offsetSec)

        let minLenRaw := ((minAsrInputSeconds * srNat.toFloat) + 0.5).toUInt64.toNat
        if minLenRaw == 0 then
          chunks
        else
          chunks.map (fun (c, off) =>
            if c.size < minLenRaw then
              (padRightWithZeros c minLenRaw, off)
            else
              (c, off))

private def offsetForcedAlignResult (r : ForcedAlignResult) (offsetSec : Float) : ForcedAlignResult :=
  { items := r.items.map (fun it =>
      { it with
        startTime := it.startTime + offsetSec
        endTime := it.endTime + offsetSec
      }) }

private def mergeChunkTexts (parts : Array String) : String := Id.run do
  let mut out := ""
  for p in parts do
    out := out ++ p
  out

/-- Preprocessor compatibility check for reusing computed frontend features
    between ASR decode and standalone forced-aligner timestamp pass. -/
private def isFrontendReuseCompatible
    (base : PreprocessorConfig)
    (align : PreprocessorConfig)
    : Bool :=
  base.featureSize == align.featureSize &&
  base.samplingRate == align.samplingRate &&
  base.hopLength == align.hopLength &&
  base.nFft == align.nFft &&
  base.doNormalize == align.doNormalize &&
  base.dither == align.dither

private def modelDevice {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg) : Device :=
  model.thinker.textModel.embed_tokens.device

private def validateForcedLanguage
    (supportedLanguages : Array String)
    (language : Option String)
    : IO (Option String) := do
  match language with
  | none => pure none
  | some l =>
    let s := l.trimAscii.toString
    if s.isEmpty then
      pure none
    else
      match normalizeLanguageName s with
      | .error e => throw <| IO.userError e
      | .ok ln =>
        match validateLanguage supportedLanguages ln with
        | .error e => throw <| IO.userError e
        | .ok _ => pure (some ln)

private def preparePromptInputIds
    {cfg : Qwen3ASRConfig}
    (tok : tokenizer.qwen3.QwenTokenizer)
    (prompt : String)
    (audioLen : UInt64)
    : IO (Sigma (fun seq => Tensor { device := .CPU, shape := #[1, seq], dtype := .Int64 })) := do
  let processor : Qwen3ASRProcessor := {}
  let promptIds ←
    match processor.encodeWithExpandedAudioTokenIds
      tok
      prompt
      #[audioLen]
      cfg.thinkerConfig.audioTokenId.toUInt32 with
    | .ok ids => pure ids
    | .error e => throw <| IO.userError e
  let idsVals : Array Int64 := promptIds.map (fun id => Int64.ofNat id.toNat)

  let seq : UInt64 := idsVals.size.toUInt64
  if seq == 0 then
    throw <| IO.userError "ASR prompt/input sequence is empty"
  let inputIds : Tensor { device := .CPU, shape := #[1, seq], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.fromInt64Array idsVals) #[1, seq]
  pure ⟨seq, inputIds⟩

private def decodeGeneratedText
    (tok : tokenizer.qwen3.QwenTokenizer)
    {seq : UInt64}
    (outSeq : UInt64)
    (outIds : Tensor { device := device, shape := #[1, outSeq], dtype := .Int64 })
    : IO String := do
  if outSeq <= seq then
    pure ""
  else
    let newSeq := outSeq - seq
    let newOnly : Tensor { device := device, shape := #[1, newSeq], dtype := .Int64 } := Tensor.sliceChecked outIds 1 seq newSeq #[1, newSeq]
    let flat : Tensor { device := device, shape := #[newSeq], dtype := .Int64 } := Tensor.reshapeChecked newOnly #[newSeq]
    let idsU64 ← Tensor.toUInt64Array flat
    let ids := idsU64.map (fun x => x.toUInt32)
    pure (tokenizer.qwen3.decodeText tok ids)

private def decodeGeneratedTextsBatch
    (tok : tokenizer.qwen3.QwenTokenizer)
    {batch seq : UInt64}
    (outSeq : UInt64)
    (outIds : Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })
    : IO (Array String) := do
  if batch == 0 then
    pure #[]
  else if outSeq <= seq then
    pure (Array.replicate batch.toNat "")
  else
    let newSeq := outSeq - seq
    let newOnly : Tensor { device := device, shape := #[batch, newSeq], dtype := .Int64 } := Tensor.sliceChecked outIds 1 seq newSeq #[batch, newSeq]
    let flat : Tensor { device := device, shape := #[batch * newSeq], dtype := .Int64 } := Tensor.reshapeChecked newOnly #[batch * newSeq]
    let idsU64 ← Tensor.toUInt64Array flat
    let rowWidth := newSeq.toNat
    let mut out : Array String := Array.mkEmpty batch.toNat
    for b in [:batch.toNat] do
      let start := b * rowWidth
      let stop := start + rowWidth
      let rowIds := (idsU64.extract start stop).map (fun x => x.toUInt32)
      out := out.push (tokenizer.qwen3.decodeText tok rowIds)
    pure out

private def extractAudioLenFromFeatureMask {cfg : Qwen3ASRConfig} {frames : UInt64}
    (featureAttentionMask : Tensor { device := device, shape := #[1, frames], dtype := .Int64 })
    : IO UInt64 := do
  let validFramesTensor := Tensor.sumAll featureAttentionMask
  let validFrames := validFramesTensor.itemInt.toInt.toNat.toUInt64
  let audioLenRaw := AudioEncoderConfig.featExtractOutputLength validFrames
  let audioLenCap := AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames
  pure (if audioLenRaw <= audioLenCap then audioLenRaw else audioLenCap)

private def alignSingleWithModel
    {alignCfg : Qwen3ASRConfig}
    (alignModel : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) alignCfg)
    (alignTok : tokenizer.qwen3.QwenTokenizer)
    (text : String)
    (language : String)
    {melBins frames : UInt64} {frontendDevice : Device}
    (frontendOut : WhisperFrontendOutput (device := frontendDevice) melBins frames)
    (audioLen : UInt64)
    : IO (Option ForcedAlignResult) := do
  let alignModel := TensorStruct.detach alignModel
  let (wordList, alignRaw) := Qwen3ForceAlignProcessor.encodeTimestampText text language
  let processor : Qwen3ASRProcessor := {}
  let alignIdsArr ←
    match processor.encodeWithExpandedAudioTokenIds
      alignTok
      alignRaw
      #[audioLen]
      alignCfg.thinkerConfig.audioTokenId.toUInt32 with
    | .ok ids => pure ids
    | .error e => throw <| IO.userError e
  let seqAlign : UInt64 := alignIdsArr.size.toUInt64
  if seqAlign == 0 then
    pure none
  else
    let dev := modelDevice alignModel
    let alignVals := alignIdsArr.map (fun id => Int64.ofNat id.toNat)
    let inputIdsAlignCpu : Tensor { device := .CPU, shape := #[1, seqAlign], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.fromInt64Array alignVals) #[1, seqAlign]
    let inputIdsAlign : Tensor { device := device, shape := #[1, seqAlign], dtype := .Int64 } := inputIdsAlignCpu.to dev
    let inputFeaturesDev : Tensor { device := device, shape := #[1, melBins, frames], dtype := .Float32 } := frontendOut.inputFeatures.detach.to dev
    let featureAttentionMaskDev : Tensor { device := device, shape := #[1, frames], dtype := .Int64 } := frontendOut.featureAttentionMask.to dev
    let results ←
      do
        alignModel.alignPrepared
          inputIdsAlign
          #[wordList]
          (inputFeatures := some (Tensor.reshapeChecked inputFeaturesDev #[1, alignCfg.thinkerConfig.audioConfig.numMelBins, frames]))
          (featureAttentionMask := some featureAttentionMaskDev)
          (attentionMask := (none : Option (Tensor { device := device, shape := #[1, seqAlign], dtype := .Int64 })))
          (timestampTokenId := alignCfg.timestampTokenId)
          (timestampSegmentTime := alignCfg.timestampSegmentTime)
    pure (results[0]?)

private def alignSingleWithForcedAligner
    (aligner : Qwen3ForcedAligner)
    (audio16k : Array Float)
    (text : String)
    (language : String)
    : IO (Option ForcedAlignResult) := do
  let preCfg : PreprocessorConfig := { aligner.preprocessor with returnAttentionMask := true }
  let dev := modelDevice aligner.model
  let ⟨frames, frontendOut⟩ ← waveformToWhisperFeaturesDynamic
    preCfg
    audio16k
    (minSeconds := minAsrInputSeconds)
    (maxSeconds := some maxForceAlignInputSeconds)
    (device := dev)
  let audioLen ← extractAudioLenFromFeatureMask (cfg := aligner.cfg) (frames := frames) frontendOut.featureAttentionMask
  alignSingleWithModel aligner.model aligner.tok text language frontendOut audioLen

private def maybeAlignTimestamps
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (text : String)
    (language : String)
    (audio16k : Array Float)
    {melBins frames : UInt64}
    (frontendOut : WhisperFrontendOutput (device := device) melBins frames)
    (audioLen : UInt64)
    (returnTimeStamps : Bool)
    : IO (Option ForcedAlignResult) := do
  if !returnTimeStamps then
    pure none
  else if (text.trimAscii.toString).isEmpty then
    pure none
  else
    match forcedAligner with
    | some aligner =>
        if isFrontendReuseCompatible preprocessor aligner.preprocessor then
          alignSingleWithModel aligner.model aligner.tok text language frontendOut audioLen
        else
          alignSingleWithForcedAligner aligner audio16k text language
    | none =>
      if !ThinkerConfig.isForcedAligner cfg.thinkerConfig then
        throw <| IO.userError
          "returnTimeStamps=true requires either a forced-aligner handle or a forced-aligner thinker checkpoint (model_type contains \"forced_aligner\")"
      else
        alignSingleWithModel model tok text language frontendOut audioLen

private def transcribeWithFrontend
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (audio16k : Array Float)
    {melBins frames : UInt64}
    (frontendOut : WhisperFrontendOutput (device := device) melBins frames)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription := do
  let model := TensorStruct.detach model
  let forceLanguage ← validateForcedLanguage model.supportLanguages language
  let prompt := buildTextPrompt context forceLanguage
  let audioLen ← extractAudioLenFromFeatureMask (cfg := cfg) frontendOut.featureAttentionMask

  let promptPack ← preparePromptInputIds (cfg := cfg) tok prompt audioLen
  let dev := modelDevice model
  let seq := promptPack.1
  let inputIds : Tensor { device := device, shape := #[1, seq], dtype := .Int64 } := promptPack.2.to dev
  let inputFeaturesDev : Tensor { device := device, shape := #[1, melBins, frames], dtype := .Float32 } := frontendOut.inputFeatures.detach.to dev
  let featureAttentionMaskDev : Tensor { device := device, shape := #[1, frames], dtype := .Int64 } := frontendOut.featureAttentionMask.to dev
  let generated ←
    do
      model.generateGreedy
        inputIds
        (inputFeatures := some (Tensor.reshapeChecked inputFeaturesDev #[1, cfg.thinkerConfig.audioConfig.numMelBins, frames]))
        (featureAttentionMask := some featureAttentionMaskDev)
        (maxNewTokens := maxNewTokens)
        (eosTokenIds := eosTokenIds)

  let outSeq := generated.1
  let outIds := generated.2
  let rawDecoded ← decodeGeneratedText tok (seq := seq) outSeq outIds
  let (lang, txt) := parseAsrOutput rawDecoded forceLanguage
  let timeStamps ← maybeAlignTimestamps
    model tok preprocessor forcedAligner txt lang audio16k frontendOut audioLen returnTimeStamps
  pure { language := lang, text := txt, timeStamps := timeStamps }

/-- Offline ASR for one in-memory 16k mono waveform. -/
def transcribeWaveform
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audio16k : Array Float)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription := do
  let dev := modelDevice model
  let maxChunkSec := if returnTimeStamps then maxForceAlignInputSeconds else maxAsrInputSeconds
  -- In non-timestamp mode, reduce boundary search window to lower CPU cost on long audios.
  let searchExpandSec := if returnTimeStamps then 5.0 else 1.0
  let chunks := splitAudioIntoChunks audio16k sampleRate16k maxChunkSec (searchExpandSec := searchExpandSec)

  let mut langs : Array String := #[]
  let mut texts : Array String := #[]
  let mut tsItems : Array ForcedAlignItem := #[]

  for (chunkWav, offsetSec) in chunks do
    let frontendPack ←
      waveformToWhisperFeaturesDynamic
        preprocessor
        chunkWav
        (minSeconds := minAsrInputSeconds)
        (maxSeconds := some maxChunkSec)
        (device := dev)
    let _frames := frontendPack.1
    let frontendOut := frontendPack.2
    let r ← transcribeWithFrontend
      model tok preprocessor forcedAligner chunkWav frontendOut context language returnTimeStamps maxNewTokens eosTokenIds
    langs := langs.push r.language
    texts := texts.push r.text
    if returnTimeStamps then
      match r.timeStamps with
      | some ts =>
        let shifted := offsetForcedAlignResult ts offsetSec
        tsItems := tsItems ++ shifted.items
      | none => pure ()

  pure {
    language := mergeLanguages langs
    text := mergeChunkTexts texts
    timeStamps :=
      if returnTimeStamps && !tsItems.isEmpty then
        some { items := tsItems }
      else
        none
  }

private def broadcastContexts (n : Nat) (contexts : Array String) : IO (Array String) := do
  if n == 0 then
    pure #[]
  else if contexts.isEmpty then
    pure (Array.replicate n "")
  else if contexts.size == 1 then
    pure (Array.replicate n (contexts[0]!))
  else if contexts.size == n then
    pure contexts
  else
    throw <| IO.userError s!"Batch size mismatch: audio={n}, context={contexts.size}"

private def broadcastLanguages (n : Nat) (languages : Array (Option String))
    : IO (Array (Option String)) := do
  if n == 0 then
    pure #[]
  else if languages.isEmpty then
    pure (Array.replicate n none)
  else if languages.size == 1 then
    pure (Array.replicate n (languages[0]!))
  else if languages.size == n then
    pure languages
  else
    throw <| IO.userError s!"Batch size mismatch: audio={n}, language={languages.size}"

private def effectiveBatchStep (n : Nat) (maxInferenceBatchSize : Int) : Nat :=
  if n == 0 then
    1
  else if maxInferenceBatchSize <= 0 then
    n
  else
    Nat.max 1 maxInferenceBatchSize.toNat

private structure BatchedDecodeRow where
  idx : Nat
  forceLanguage : Option String
  seq : UInt64
  frames : UInt64
  melBins : UInt64
  inputIdsRow : Tensor { device := device, shape := #[1, seq], dtype := .Int64 }
  inputFeaturesRow : Tensor { device := device, shape := #[1, melBins, frames], dtype := .Float32 }
  featureAttentionMaskRow : Tensor { device := device, shape := #[1, frames], dtype := .Int64 }
  deriving Inhabited

private structure BatchedDecodeBucket where
  frames : UInt64
  seq : UInt64
  rows : Array (BatchedDecodeRow (device := device)) := #[]
  deriving Inhabited

private def findBucketIdx
    (buckets : Array (BatchedDecodeBucket (device := device)))
    (frames seq : UInt64)
    : Option Nat :=
  Id.run do
    let mut found : Option Nat := none
    for i in [:buckets.size] do
      if found.isNone then
        let b := buckets[i]!
        if b.frames == frames && b.seq == seq then
          found := some i
    found

private def addRowToBuckets
    (buckets : Array (BatchedDecodeBucket (device := device)))
    (frames seq : UInt64)
    (row : BatchedDecodeRow (device := device))
    : Array (BatchedDecodeBucket (device := device)) :=
  match findBucketIdx buckets frames seq with
  | some i =>
      let b := buckets[i]!
      buckets.set! i { b with rows := b.rows.push row }
  | none =>
      buckets.push { frames := frames, seq := seq, rows := #[row] }

/-- Try native batched decode for one chunk of waveforms.
    Returns `none` when inputs are not batch-collatable (e.g. multi-chunk split). -/
private def tryTranscribeWaveformsBatchedChunk
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audios16k : Array (Array Float))
    (contexts : Array String)
    (languages : Array (Option String))
    (maxNewTokens : UInt64)
    (eosTokenIds : Array UInt64)
    : IO (Option (Array ASRTranscription)) := do
  let model := TensorStruct.detach model
  let dev := modelDevice model
  let n := audios16k.size
  if n == 0 then
    return some #[]
  if contexts.size != n || languages.size != n then
    throw <| IO.userError
      s!"Internal batch collation mismatch: audio={n}, context={contexts.size}, language={languages.size}"

  let maxChunkSec := maxAsrInputSeconds
  let mut buckets : Array (BatchedDecodeBucket (device := device)) := #[]
  for i in [:n] do
    let searchExpandSec : Float := 1.0
    let chunks :=
      splitAudioIntoChunks
        (audios16k[i]!)
        sampleRate16k
        maxChunkSec
        (searchExpandSec := searchExpandSec)
    if chunks.size != 1 then
      return none
    let chunkWav := (chunks[0]!).1
    let forceLanguage ← validateForcedLanguage model.supportLanguages (languages[i]!)
    let prompt := buildTextPrompt (contexts[i]!) forceLanguage
    let frontendPack ←
      waveformToWhisperFeaturesDynamic
        preprocessor
        chunkWav
        (minSeconds := minAsrInputSeconds)
        (maxSeconds := some maxChunkSec)
        (device := dev)
    let frames := frontendPack.1
    let frontendOut := frontendPack.2
    let audioLen ← extractAudioLenFromFeatureMask (cfg := cfg) (frames := frames) frontendOut.featureAttentionMask
    let promptPack ← preparePromptInputIds (cfg := cfg) tok prompt audioLen
    let seq := promptPack.1
    let inputIdsRowDyn : Tensor { device := device, shape := #[1, seq], dtype := .Int64 } := Tensor.reshapeChecked (promptPack.2.to device) #[1, seq]
    let inputFeaturesRowDyn : Tensor { device := device, shape := #[1, preprocessor.featureSize, frames], dtype := .Float32 } :=
      Tensor.reshapeChecked frontendOut.inputFeatures #[1, preprocessor.featureSize, frames]
    let featureAttentionMaskRowDyn : Tensor { device := device, shape := #[1, frames], dtype := .Int64 } := Tensor.reshapeChecked frontendOut.featureAttentionMask #[1, frames]
    buckets := addRowToBuckets buckets frames seq {
      seq := seq
      frames := frames
      melBins := preprocessor.featureSize
      idx := i
      forceLanguage := forceLanguage
      inputIdsRow := inputIdsRowDyn
      inputFeaturesRow := inputFeaturesRowDyn
      featureAttentionMaskRow := featureAttentionMaskRowDyn
    }

  let mut outByIdx : Array (Option ASRTranscription) := Array.replicate n none
  for bucket in buckets do
    if bucket.rows.isEmpty then
      pure ()
    else
      let batch := bucket.rows.size.toUInt64
      let seq := bucket.seq
      let frames := bucket.frames
      let inputRows := bucket.rows.map (fun r => r.inputIdsRow.packShape)
      let featRows := bucket.rows.map (fun r => r.inputFeaturesRow.packShape)
      let fmaskRows := bucket.rows.map (fun r => r.featureAttentionMaskRow.packShape)
      let inputIdsCpu : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 } := Tensor.catManyChecked inputRows 0 #[batch, seq]
      let inputFeaturesCpu : Tensor { device := device, shape := #[batch, preprocessor.featureSize, frames], dtype := .Float32 } :=
        Tensor.catManyChecked featRows 0 #[batch, preprocessor.featureSize, frames]
      let featureAttentionMaskCpu : Tensor { device := device, shape := #[batch, frames], dtype := .Int64 } :=
        Tensor.catManyChecked fmaskRows 0 #[batch, frames]
      let generated ←
        do
          model.generateGreedy
            (inputIdsCpu.to dev)
            (inputFeatures := some (Tensor.reshapeChecked (inputFeaturesCpu.to dev) #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames]))
            (featureAttentionMask := some (featureAttentionMaskCpu.to dev))
            (maxNewTokens := maxNewTokens)
            (eosTokenIds := eosTokenIds)
      let outSeq := generated.1
      let outIds := generated.2
      let raws ← decodeGeneratedTextsBatch tok (batch := batch) (seq := seq) outSeq outIds
      if raws.size != bucket.rows.size then
        throw <| IO.userError
          s!"Internal batch decode mismatch: rows={bucket.rows.size}, decoded={raws.size}"
      for i in [:bucket.rows.size] do
        let row := bucket.rows[i]!
        let raw := raws[i]!
        let (lang, txt) := parseAsrOutput raw row.forceLanguage
        outByIdx := outByIdx.set! row.idx (some {
          language := lang
          text := txt
          timeStamps := none
        })

  let mut out : Array ASRTranscription := Array.mkEmpty n
  for i in [:n] do
    match outByIdx[i]! with
    | some r => out := out.push r
    | none =>
        throw <| IO.userError s!"Internal batch decode missing output for row {i}"
  pure (some out)

/-- Offline ASR for a batch of in-memory 16k mono waveforms. -/
def transcribeWaveforms
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audios16k : Array (Array Float))
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (contexts : Array String := #[])
    (languages : Array (Option String) := #[])
    (maxInferenceBatchSize : Int := -1)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Array ASRTranscription) := do
  let n := audios16k.size
  let ctxs ← broadcastContexts n contexts
  let langs ← broadcastLanguages n languages
  let mut out : Array ASRTranscription := Array.mkEmpty n
  let step := effectiveBatchStep n maxInferenceBatchSize
  let mut off : Nat := 0
  while off < n do
    let hi := Nat.min n (off + step)
    let audChunk := audios16k.extract off hi
    let ctxChunk := ctxs.extract off hi
    let langChunk := langs.extract off hi
    let usedBatch ←
      if returnTimeStamps || forcedAligner.isSome then
        pure false
      else
        match (← tryTranscribeWaveformsBatchedChunk
          model
          tok
          preprocessor
          audChunk
          ctxChunk
          langChunk
          maxNewTokens
          eosTokenIds) with
        | some rs =>
            out := out ++ rs
            pure true
        | none =>
            pure false
    if !usedBatch then
      for i in [off:hi] do
        let r ← transcribeWaveform
          model tok preprocessor (audios16k[i]!)
          (forcedAligner := forcedAligner)
          (context := ctxs[i]!)
          (language := langs[i]!)
          (returnTimeStamps := returnTimeStamps)
          (maxNewTokens := maxNewTokens)
          (eosTokenIds := eosTokenIds)
        out := out.push r
    off := hi
  pure out

/-- Offline ASR from one WAV path. WAV is normalized/resampled to 16k first. -/
def transcribeWav
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (wavPath : String)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription := do
  let wav16k ← normalizeAudioTo16kFromWav wavPath
  transcribeWaveform
    model tok preprocessor wav16k forcedAligner context language returnTimeStamps maxNewTokens eosTokenIds

/-- Offline ASR from one unified audio input.
    Supports path/URL/base64/in-memory waveform+sample-rate forms. -/
def transcribeAudioInput
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audio : ASRAudioInput)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription := do
  let wav16k ← normalizeAudioInputTo16k audio
  transcribeWaveform
    model tok preprocessor wav16k forcedAligner context language returnTimeStamps maxNewTokens eosTokenIds

/-- Offline ASR from string source with auto-detection:
    local path / URL / base64 payload. -/
def transcribeAudioSource
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (source : String)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription :=
  transcribeAudioInput
    model
    tok
    preprocessor
    (ASRAudioInput.ofStringAuto source)
    forcedAligner
    context
    language
    returnTimeStamps
    maxNewTokens
    eosTokenIds

/-- Offline ASR from WAV paths. WAVs are normalized/resampled to 16k first. -/
def transcribeWavs
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (wavPaths : Array String)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (contexts : Array String := #[])
    (languages : Array (Option String) := #[])
    (maxInferenceBatchSize : Int := -1)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Array ASRTranscription) := do
  let mut audios16k : Array (Array Float) := Array.mkEmpty wavPaths.size
  for p in wavPaths do
    audios16k := audios16k.push (← normalizeAudioTo16kFromWav p)
  transcribeWaveforms
    model
    tok
    preprocessor
    audios16k
    forcedAligner
    contexts
    languages
    maxInferenceBatchSize
    returnTimeStamps
    maxNewTokens
    eosTokenIds

/-- Offline ASR from unified audio input batch.
    Inputs can mix path/URL/base64/in-memory waveform forms. -/
def transcribeAudioInputs
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audios : Array ASRAudioInput)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (contexts : Array String := #[])
    (languages : Array (Option String) := #[])
    (maxInferenceBatchSize : Int := -1)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Array ASRTranscription) := do
  let mut audios16k : Array (Array Float) := Array.mkEmpty audios.size
  for a in audios do
    audios16k := audios16k.push (← normalizeAudioInputTo16k a)
  transcribeWaveforms
    model
    tok
    preprocessor
    audios16k
    forcedAligner
    contexts
    languages
    maxInferenceBatchSize
    returnTimeStamps
    maxNewTokens
    eosTokenIds

/-- Offline ASR from string sources with auto-detection per source string. -/
def transcribeAudioSources
    {cfg : Qwen3ASRConfig}
    (model : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (sources : Array String)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (contexts : Array String := #[])
    (languages : Array (Option String) := #[])
    (maxInferenceBatchSize : Int := -1)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Array ASRTranscription) :=
  transcribeAudioInputs
    model
    tok
    preprocessor
    (sources.map ASRAudioInput.ofStringAuto)
    forcedAligner
    contexts
    languages
    maxInferenceBatchSize
    returnTimeStamps
    maxNewTokens
    eosTokenIds

namespace Qwen3ForcedAligner

def loadFromPretrained
    (source : String)
    (revision : String := "main")
    (cacheDir : String := Hub.defaultCacheDir) : IO Qwen3ForcedAligner := do
  let modelDir ← hub.resolvePretrainedDir source {
    revision := revision
    cacheDir := cacheDir
    includeTokenizer := true
    includePreprocessor := true
  }
  let cfg ← Qwen3ASRConfig.loadFromPretrainedDir modelDir
  if !ThinkerConfig.isForcedAligner cfg.thinkerConfig then
    throw <| IO.userError
      s!"Requested aligner source is not a forced-aligner checkpoint (model_type={cfg.thinkerConfig.modelType})"
  let tok ← tokenizer.qwen3.loadTokenizer modelDir
  let preprocessor ← PreprocessorConfig.loadFromPretrainedDir modelDir
  let ⟨device, dtype, model⟩ ← Qwen3ASRForConditionalGeneration.loadSharded modelDir cfg
  pure { cfg, device, dtype, model, tok, preprocessor }

private def validateAlignLanguage (aligner : Qwen3ForcedAligner) (language : String) : IO String := do
  let ln ←
    match normalizeLanguageName language with
    | .ok s => pure s
    | .error e => throw <| IO.userError e
  match validateLanguage aligner.cfg.supportLanguages ln with
  | .ok _ => pure ln
  | .error e => throw <| IO.userError e

private def broadcastTexts (n : Nat) (texts : Array String) : IO (Array String) := do
  if n == 0 then
    pure #[]
  else if texts.isEmpty then
    throw <| IO.userError "Forced aligner requires non-empty text input"
  else if texts.size == 1 then
    pure (Array.replicate n (texts[0]!))
  else if texts.size == n then
    pure texts
  else
    throw <| IO.userError s!"Batch size mismatch: audio={n}, text={texts.size}"

private def broadcastAlignLanguages (n : Nat) (languages : Array String) : IO (Array String) := do
  if n == 0 then
    pure #[]
  else if languages.isEmpty then
    throw <| IO.userError "Forced aligner requires non-empty language input"
  else if languages.size == 1 then
    pure (Array.replicate n (languages[0]!))
  else if languages.size == n then
    pure languages
  else
    throw <| IO.userError s!"Batch size mismatch: audio={n}, language={languages.size}"

/-- Align one 16k mono waveform against transcript text/language. -/
def alignWaveform
    (aligner : Qwen3ForcedAligner)
    (audio16k : Array Float)
    (text : String)
    (language : String)
    : IO ForcedAlignResult := do
  let dev := modelDevice aligner.model
  let ln ← validateAlignLanguage aligner language
  let trimmed := text.trimAscii.toString
  if trimmed.isEmpty then
    pure { items := #[] }
  else
    let preCfg : PreprocessorConfig := { aligner.preprocessor with returnAttentionMask := true }
    let ⟨frames, frontendOut⟩ ← waveformToWhisperFeaturesDynamic
      preCfg
      audio16k
      (minSeconds := minAsrInputSeconds)
      (maxSeconds := some maxForceAlignInputSeconds)
      (device := dev)
    let audioLen ← extractAudioLenFromFeatureMask (cfg := aligner.cfg) (frames := frames) frontendOut.featureAttentionMask
    match (← alignSingleWithModel aligner.model aligner.tok trimmed ln frontendOut audioLen) with
    | some r => pure r
    | none => pure { items := #[] }

/-- Align one unified audio input against transcript text/language. -/
def alignAudioInput
    (aligner : Qwen3ForcedAligner)
    (audio : ASRAudioInput)
    (text : String)
    (language : String)
    : IO ForcedAlignResult := do
  let wav16k ← normalizeAudioInputTo16k audio
  alignWaveform aligner wav16k text language

/-- Upstream-style align entrypoint alias on unified audio input. -/
def align
    (aligner : Qwen3ForcedAligner)
    (audio : ASRAudioInput)
    (text : String)
    (language : String)
    : IO ForcedAlignResult :=
  alignAudioInput aligner audio text language

/-- Align a batch of 16k mono waveforms with optional throughput chunking. -/
def alignWaveforms
    (aligner : Qwen3ForcedAligner)
    (audios16k : Array (Array Float))
    (texts : Array String)
    (languages : Array String)
    (maxInferenceBatchSize : Int := -1)
    : IO (Array ForcedAlignResult) := do
  let n := audios16k.size
  let ts ← broadcastTexts n texts
  let ls ← broadcastAlignLanguages n languages
  let step := effectiveBatchStep n maxInferenceBatchSize
  let mut out : Array ForcedAlignResult := Array.mkEmpty n
  let mut off : Nat := 0
  while off < n do
    let hi := Nat.min n (off + step)
    for i in [off:hi] do
      out := out.push (← alignWaveform aligner (audios16k[i]!) (ts[i]!) (ls[i]!))
    off := hi
  pure out

/-- Align a batch of unified audio inputs with optional throughput chunking. -/
def alignAudioInputs
    (aligner : Qwen3ForcedAligner)
    (audios : Array ASRAudioInput)
    (texts : Array String)
    (languages : Array String)
    (maxInferenceBatchSize : Int := -1)
    : IO (Array ForcedAlignResult) := do
  let n := audios.size
  let ts ← broadcastTexts n texts
  let ls ← broadcastAlignLanguages n languages
  let step := effectiveBatchStep n maxInferenceBatchSize
  let mut out : Array ForcedAlignResult := Array.mkEmpty n
  let mut off : Nat := 0
  while off < n do
    let hi := Nat.min n (off + step)
    for i in [off:hi] do
      out := out.push (← alignAudioInput aligner (audios[i]!) (ts[i]!) (ls[i]!))
    off := hi
  pure out

/-- Upstream-style batch align alias on unified audio input list. -/
def alignBatch
    (aligner : Qwen3ForcedAligner)
    (audios : Array ASRAudioInput)
    (texts : Array String)
    (languages : Array String)
    (maxInferenceBatchSize : Int := -1)
    : IO (Array ForcedAlignResult) :=
  alignAudioInputs aligner audios texts languages maxInferenceBatchSize

end Qwen3ForcedAligner

namespace Qwen3ASRForConditionalGeneration

/-- Method-style offline ASR for one waveform. -/
def transcribeWaveform
    (m : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audio16k : Array Float)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription :=
  torch.qwen3asr.transcribeWaveform
    m tok preprocessor audio16k forcedAligner context language returnTimeStamps maxNewTokens eosTokenIds

/-- Method-style offline ASR for multiple waveforms. -/
def transcribeWaveforms
    (m : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audios16k : Array (Array Float))
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (contexts : Array String := #[])
    (languages : Array (Option String) := #[])
    (maxInferenceBatchSize : Int := -1)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Array ASRTranscription) :=
  torch.qwen3asr.transcribeWaveforms
    m tok preprocessor audios16k forcedAligner contexts languages maxInferenceBatchSize returnTimeStamps maxNewTokens eosTokenIds

/-- Method-style offline ASR from one WAV path. -/
def transcribeWav
    (m : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (wavPath : String)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription :=
  torch.qwen3asr.transcribeWav
    m tok preprocessor wavPath forcedAligner context language returnTimeStamps maxNewTokens eosTokenIds

/-- Method-style offline ASR from one unified audio input. -/
def transcribeAudioInput
    (m : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audio : ASRAudioInput)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription :=
  torch.qwen3asr.transcribeAudioInput
    m tok preprocessor audio forcedAligner context language returnTimeStamps maxNewTokens eosTokenIds

/-- Method-style offline ASR from string source with auto-detection. -/
def transcribeAudioSource
    (m : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (source : String)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (context : String := "")
    (language : Option String := none)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ASRTranscription :=
  torch.qwen3asr.transcribeAudioSource
    m tok preprocessor source forcedAligner context language returnTimeStamps maxNewTokens eosTokenIds

/-- Method-style offline ASR from WAV paths. -/
def transcribeWavs
    (m : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (wavPaths : Array String)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (contexts : Array String := #[])
    (languages : Array (Option String) := #[])
    (maxInferenceBatchSize : Int := -1)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Array ASRTranscription) :=
  torch.qwen3asr.transcribeWavs
    m tok preprocessor wavPaths forcedAligner contexts languages maxInferenceBatchSize returnTimeStamps maxNewTokens eosTokenIds

/-- Method-style offline ASR from unified audio inputs. -/
def transcribeAudioInputs
    (m : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (audios : Array ASRAudioInput)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (contexts : Array String := #[])
    (languages : Array (Option String) := #[])
    (maxInferenceBatchSize : Int := -1)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Array ASRTranscription) :=
  torch.qwen3asr.transcribeAudioInputs
    m tok preprocessor audios forcedAligner contexts languages maxInferenceBatchSize returnTimeStamps maxNewTokens eosTokenIds

/-- Method-style offline ASR from string sources with auto-detection. -/
def transcribeAudioSources
    (m : Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tok : tokenizer.qwen3.QwenTokenizer)
    (preprocessor : PreprocessorConfig)
    (sources : Array String)
    (forcedAligner : Option Qwen3ForcedAligner := none)
    (contexts : Array String := #[])
    (languages : Array (Option String) := #[])
    (maxInferenceBatchSize : Int := -1)
    (returnTimeStamps : Bool := false)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Array ASRTranscription) :=
  torch.qwen3asr.transcribeAudioSources
    m tok preprocessor sources forcedAligner contexts languages maxInferenceBatchSize returnTimeStamps maxNewTokens eosTokenIds

end Qwen3ASRForConditionalGeneration

end torch.qwen3asr
