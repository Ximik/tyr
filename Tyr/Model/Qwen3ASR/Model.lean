/-
  Tyr/Model/Qwen3ASR/Model.lean

  Lean-native Qwen3-ASR thinker + top-level model:
  - audio tower
  - text decoder
  - LM head (ASR / forced-aligner shape aware)
  - rope-delta helpers + greedy generation utilities
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen.Model
import Tyr.Model.Qwen.Layer
import Tyr.Model.Qwen.RoPE
import Tyr.Model.Qwen3ASR.Config
import Tyr.Model.Qwen3ASR.AudioEncoder
import Tyr.Model.Qwen3ASR.ForcedAligner
import Tyr.Model.Utils

namespace torch.qwen3asr

variable {device : Device}
open torch.Model

abbrev TextQwenConfig (cfg : ThinkerConfig) : qwen.QwenConfig :=
  TextConfig.toQwenConfig cfg.textConfig

abbrev ThinkerLmVocabSize (cfg : ThinkerConfig) : UInt64 :=
  ThinkerConfig.lmHeadOutDim cfg

private def tokenInSetMask {batch : UInt64}
    (tokens : Tensor { device := device, shape := #[batch], dtype := .Int64 })
    (allow : Array UInt64)
    : Tensor { device := device, shape := #[batch], dtype := .Bool } :=
  Id.run do
    let mut out : Tensor { device := device, shape := #[batch], dtype := .Bool } := Tensor.eqScalar tokens (-1)
    for tok in allow do
      let isTok : Tensor { device := device, shape := #[batch], dtype := .Bool } := Tensor.eqScalar tokens (Int64.ofNat tok.toNat)
      out := Tensor.logicalOr out isTok
    out

private def eosVectorOnDevice {batch : UInt64}
    (eosTokenIds : Array UInt64)
    (device : Device)
    : Option (Tensor { device := device, shape := #[batch], dtype := .Int64 }) :=
  match eosTokenIds[0]? with
  | none => none
  | some eosTok =>
    let eosCpu : Tensor { device := .CPU, shape := #[batch], dtype := .Int64 } := Tensor.fullInt #[batch] (Int64.ofNat eosTok.toNat)
    let eosDev : Tensor { device := device, shape := #[batch], dtype := .Int64 } := eosCpu.to device
    some eosDev

private def buildLengthMask {batch maxLen : UInt64} (lengths : Array UInt64) : Tensor { device := device, shape := #[batch, maxLen], dtype := .Int64 } :=
  Id.run do
    let mut vals : Array Int64 := #[]
    for i in [:batch.toNat] do
      let len := lengths.getD i 0
      for j in [:maxLen.toNat] do
        let v : Int64 := if j.toUInt64 < len then 1 else 0
        vals := vals.push v
    Tensor.reshapeChecked ((Tensor.fromInt64Array vals).to device) #[batch, maxLen]

private def maskRowCounts {batch seq : UInt64} (attentionMask : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) : IO (Array UInt64) := do
  let flat : Tensor { device := device, shape := #[batch * seq], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.toInt64 attentionMask) #[batch * seq]
  let vals ← Tensor.toUInt64Array flat
  let mut counts : Array UInt64 := Array.mkEmpty batch.toNat
  for b in [:batch.toNat] do
    let mut c : UInt64 := 0
    for t in [:seq.toNat] do
      if vals.getD (b * seq.toNat + t) 0 != 0 then
        c := c + 1
    counts := counts.push c
  pure counts

private def clipTo (x hi : UInt64) : UInt64 :=
  if hi == 0 then 0 else if x < hi then x else hi - 1

/-- Derive per-batch position rows used by reference `get_rope_index`.
    Padded tokens map to position `1`; valid tokens are `0,1,2,...`. -/
private def positionRowsFromMask {batch seq : UInt64}
    (attentionMask : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : IO (Array (Array UInt64)) := do
  let flat : Tensor { device := device, shape := #[batch * seq], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.toInt64 attentionMask) #[batch * seq]
  let vals ← Tensor.toUInt64Array flat
  let mut rows : Array (Array UInt64) := Array.mkEmpty batch.toNat
  for b in [:batch.toNat] do
    let mut row : Array UInt64 := Array.mkEmpty seq.toNat
    let mut running : UInt64 := 0
    for t in [:seq.toNat] do
      let v := vals.getD (b * seq.toNat + t) 0
      if v == 0 then
        row := row.push 1
      else
        row := row.push running
        running := running + 1
    rows := rows.push row
  pure rows

/-- Gather `[seq, half]` rotary table rows using explicit per-token positions. -/
private def gatherRowsByPositions {seq half : UInt64}
    (table : Tensor { device := device, shape := #[seq, half], dtype := .Float32 })
    (positions : Array UInt64)
    : Tensor { device := device, shape := #[seq, half], dtype := .Float32 } :=
  let idxVals : Array Int64 :=
    Id.run do
      let mut out : Array Int64 := Array.mkEmpty seq.toNat
      for i in [:seq.toNat] do
        let p := clipTo (positions.getD i 0) seq
        out := out.push (Int64.ofNat p.toNat)
      out
  let idxCpu : Tensor { device := .CPU, shape := #[seq], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.fromInt64Array idxVals) #[seq]
  let idx : Tensor { device := device, shape := #[seq], dtype := .Int64 } := idxCpu.to table.device
  let idx2d : Tensor { device := device, shape := #[seq, half], dtype := .Int64 } := Tensor.expand (Tensor.reshapeChecked idx #[seq, 1]) #[seq, half]
  Tensor.reshapeChecked (Tensor.gather table 0 idx2d) #[seq, half]

/-- Output container mirroring `...CausalLMOutputWithPast` fields that are
    currently represented in the Lean port. -/
structure Qwen3ASRThinkerCausalLMOutput (cfg : ThinkerConfig) (batch seq : UInt64) where
  logits : Tensor { device := device, shape := #[batch, seq, ThinkerLmVocabSize cfg] }
  ropeDeltas : Option (Tensor { device := device, shape := #[batch, 1], dtype := .Int64 }) := none

/-- Small Lean equivalent of generation input preparation payload. -/
structure ThinkerGenerationInputs (cfg : ThinkerConfig) (batch seq frames : UInt64) where
  inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }
  inputFeatures : Option (Tensor { device := device, shape := #[batch, cfg.audioConfig.numMelBins, frames] }) := none
  featureAttentionMask : Option (Tensor { device := device, shape := #[batch, frames], dtype := .Int64 }) := none

/-- Thinker model: audio encoder + text decoder + LM head. -/
structure Qwen3ASRThinkerForConditionalGeneration (cfg : ThinkerConfig) where
  audioTower : AudioEncoder (device := device) cfg.audioConfig
  textModel : qwen.Qwen3Model (device := device) (TextQwenConfig cfg) (.Unknown "dynamic")
  audioProjectionWeight : Tensor { device := device, shape := #[cfg.textConfig.hiddenSize, cfg.audioConfig.outputDim] }
  audioProjectionBias : Tensor { device := device, shape := #[cfg.textConfig.hiddenSize] }
  lmHead : Tensor { device := device, shape := #[ThinkerLmVocabSize cfg, cfg.textConfig.hiddenSize] }
  deriving TensorStruct

namespace Qwen3ASRThinkerForConditionalGeneration

def init (cfg : ThinkerConfig) (device : Device := .CPU) : IO (Qwen3ASRThinkerForConditionalGeneration (device := device) cfg) := do
  let audioTower ← AudioEncoder.init (device := device) cfg.audioConfig
  let textModel ← qwen.Qwen3Model.init (device := device) (TextQwenConfig cfg) (.Unknown "dynamic")
  let audioProjectionWeight ←
    if h : cfg.audioConfig.outputDim = cfg.textConfig.hiddenSize then
      let eye : Tensor { device := device, shape := #[cfg.textConfig.hiddenSize, cfg.textConfig.hiddenSize] } ←
        Tensor.parameter ((← Tensor.eyeForIO (n := cfg.textConfig.hiddenSize) .Float32 device).forgetDType)
      let eye' : Tensor { device := device, shape := #[cfg.textConfig.hiddenSize, cfg.audioConfig.outputDim] } := by
        simpa [h] using eye
      pure eye'
    else
      throw <| IO.userError
        s!"Qwen3-ASR expects audio output dim == text hidden dim, got {cfg.audioConfig.outputDim} and {cfg.textConfig.hiddenSize}"
  let audioProjectionBias ← initBias (device := device) #[cfg.textConfig.hiddenSize] (.Unknown "dynamic")
  let lmHead ← initWeight (device := device) #[ThinkerLmVocabSize cfg, cfg.textConfig.hiddenSize] cfg.textConfig.hiddenSize (.Unknown "dynamic")
  pure { audioTower, textModel, audioProjectionWeight, audioProjectionBias, lmHead }

def embedText {batch textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (textIds : Tensor { device := device, shape := #[batch, textSeq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, textSeq, cfg.textConfig.hiddenSize] } :=
  Tensor.embedding textIds m.textModel.embed_tokens

def projectAudio {batch audioSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (audioFeatures : Tensor { device := device, shape := #[batch, audioSeq, cfg.audioConfig.outputDim] })
    : Tensor { device := device, shape := #[batch, audioSeq, cfg.textConfig.hiddenSize] } :=
  Tensor.affine3d audioFeatures m.audioProjectionWeight m.audioProjectionBias

def buildInputsFromText {batch textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (textIds : Tensor { device := device, shape := #[batch, textSeq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, textSeq, cfg.textConfig.hiddenSize] } :=
  embedText m textIds

def buildInputsFromAudioAndText {batch audioSeq textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (audioFeatures : Tensor { device := device, shape := #[batch, audioSeq, cfg.audioConfig.outputDim] })
    (textIds : Tensor { device := device, shape := #[batch, textSeq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, audioSeq + textSeq, cfg.textConfig.hiddenSize] } :=
  let audioEmb := projectAudio m audioFeatures
  let textEmb := embedText m textIds
  Tensor.cat audioEmb textEmb 1

def forwardEmbeds {batch seq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.textConfig.hiddenSize] })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, seq, ThinkerLmVocabSize cfg] } :=
  let (cos, sin) :=
    qwen.computeFreqsOnDevice
      seq
      cfg.textConfig.headDim
      cfg.textConfig.ropeTheta
      inputsEmbeds.device
  let hidden := match attnMask with
    | some mask =>
      m.textModel.layers.foldl
        (fun h layer => layer.forwardMasked h cos sin mask true)
        inputsEmbeds
    | none =>
      m.textModel.layers.foldl
        (fun h layer => layer.forward h cos sin true)
        inputsEmbeds
  let hidden := m.textModel.norm.forward3dPreserving hidden
  Tensor.linear3d hidden m.lmHead

private def forwardEmbedsMaskedWithPositionRows {batch seq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.textConfig.hiddenSize] })
    (attnMask : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : IO (Tensor { device := device, shape := #[batch, seq, ThinkerLmVocabSize cfg] }) := do
  let half := cfg.textConfig.headDim / 2
  let (baseCos, baseSin) :=
    qwen.computeFreqsOnDevice
      seq
      cfg.textConfig.headDim
      cfg.textConfig.ropeTheta
      inputsEmbeds.device
  let rows ← positionRowsFromMask attnMask

  if batch == 0 then
    pure ((Tensor.zeros (device := device) #[batch, seq, ThinkerLmVocabSize cfg]).forgetDType)
  else
    let mut perBatchHidden : Array (Sigma (fun shape => Tensor { device := device, shape := shape })) := #[]
    for b in [:batch.toNat] do
      let row := rows.getD b (Array.replicate seq.toNat 0)
      let cosB : Tensor { device := device, shape := #[seq, half], dtype := .Float32 } := gatherRowsByPositions baseCos row
      let sinB : Tensor { device := device, shape := #[seq, half], dtype := .Float32 } := gatherRowsByPositions baseSin row
      let embB : Tensor { device := device, shape := #[1, seq, cfg.textConfig.hiddenSize] } := Tensor.sliceChecked inputsEmbeds 0 b.toUInt64 1 #[1, seq, cfg.textConfig.hiddenSize]
      let maskB : Tensor { device := device, shape := #[1, seq], dtype := .Int64 } := Tensor.sliceChecked attnMask 0 b.toUInt64 1 #[1, seq]
      let hiddenB :=
        m.textModel.layers.foldl
          (fun h layer => layer.forwardMasked h cosB sinB maskB true)
          embB
      let hiddenB := m.textModel.norm.forward3dPreserving hiddenB
      perBatchHidden := perBatchHidden.push (Tensor.packShape hiddenB)

    let hidden := Tensor.catManyChecked perBatchHidden 0 #[batch, seq, cfg.textConfig.hiddenSize]
    pure (Tensor.linear3d hidden m.lmHead)

def forwardText {batch textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (textIds : Tensor { device := device, shape := #[batch, textSeq], dtype := .Int64 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, textSeq], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, textSeq, ThinkerLmVocabSize cfg] } :=
  forwardEmbeds m (buildInputsFromText m textIds) attnMask

def forwardAudioText {batch audioSeq textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (audioFeatures : Tensor { device := device, shape := #[batch, audioSeq, cfg.audioConfig.outputDim] })
    (textIds : Tensor { device := device, shape := #[batch, textSeq], dtype := .Int64 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, audioSeq + textSeq], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, audioSeq + textSeq, ThinkerLmVocabSize cfg] } :=
  forwardEmbeds m (buildInputsFromAudioAndText m audioFeatures textIds) attnMask

def encodeAudio {batch frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputFeatures : Tensor { device := device, shape := #[batch, cfg.audioConfig.numMelBins, frames] })
    (attnMask : Option (Tensor { device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames, cfg.audioConfig.outputDim] } :=
  m.audioTower.forward inputFeatures (attnMask.map Tensor.forgetDType)

def encodeAudioVarLen {batch frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputFeatures : Tensor { device := device, shape := #[batch, cfg.audioConfig.numMelBins, frames] })
    (featureLens : Array UInt64)
    : Tensor { device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames, cfg.audioConfig.outputDim] } :=
  m.audioTower.forwardVarLen inputFeatures featureLens

private def featureLengthsFromFeatureMask (cfg : ThinkerConfig) {batch featureSeq : UInt64}
    (featureAttentionMask : Tensor { device := device, shape := #[batch, featureSeq], dtype := .Int64 })
    : IO (Array UInt64) := do
  let _ := cfg
  let featureLensTensor : Tensor { device := device, shape := #[batch], dtype := .Int64 } := (Tensor.sumDim (Tensor.toInt64 featureAttentionMask) 1 false).reshapeChecked #[batch]
  let featureLensRaw ← Tensor.toUInt64Array featureLensTensor
  pure <| featureLensRaw.map (fun l => if l <= featureSeq then l else featureSeq)

private def getPlaceholderMask2d (cfg : ThinkerConfig) {batch seq : UInt64}
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq], dtype := .Bool } :=
  Tensor.eqScalar inputIds (Int64.ofNat cfg.audioTokenId.toNat)

private def getPlaceholderMask (cfg : ThinkerConfig) {batch seq : UInt64}
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq, cfg.textConfig.hiddenSize], dtype := .Bool } :=
  let mask2d := getPlaceholderMask2d cfg inputIds
  Tensor.expand (Tensor.reshapeChecked mask2d #[batch, seq, 1]) #[batch, seq, cfg.textConfig.hiddenSize]

private def audioValidMaskFromFeatureMask (cfg : ThinkerConfig) {batch featureSeq audioSeq : UInt64}
    (featureAttentionMask : Tensor { device := device, shape := #[batch, featureSeq], dtype := .Int64 })
    : IO (Tensor { device := device, shape := #[batch, audioSeq, cfg.textConfig.hiddenSize], dtype := .Bool }) := do
  let featureLens ← featureLengthsFromFeatureMask cfg featureAttentionMask
  let audioLens := AudioEncoderConfig.featExtractOutputLengths featureLens
  let valid2d : Tensor { device := device, shape := #[batch, audioSeq], dtype := .Int64 } := buildLengthMask audioLens
  let valid2dBool : Tensor { device := device, shape := #[batch, audioSeq], dtype := .Bool } := Tensor.eqScalar valid2d 1
  pure <| Tensor.expand (Tensor.reshapeChecked valid2dBool #[batch, audioSeq, 1]) #[batch, audioSeq, cfg.textConfig.hiddenSize]

private def audioAttentionMaskFromFeatureMask (cfg : ThinkerConfig) {batch featureSeq audioSeq : UInt64}
    (featureAttentionMask : Tensor { device := device, shape := #[batch, featureSeq], dtype := .Int64 })
    : IO (Tensor { device := device, shape := #[batch, audioSeq], dtype := .Int64 }) := do
  let featureLens ← featureLengthsFromFeatureMask cfg featureAttentionMask
  let audioLens := AudioEncoderConfig.featExtractOutputLengths featureLens
  pure (buildLengthMask audioLens)

private def buildInputsEmbeds {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputFeatures : Option (Tensor { device := device, shape := #[batch, cfg.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[batch, frames], dtype := .Int64 }) := none)
    : IO (Tensor { device := device, shape := #[batch, seq, cfg.textConfig.hiddenSize] }) := do
  let inputsEmbeds0 := embedText m inputIds
  match inputFeatures with
  | none => pure inputsEmbeds0
  | some feats =>
    let audioSeq := AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames
    let placeholderMask2d := getPlaceholderMask2d cfg inputIds
    let placeholderLensTensor : Tensor { device := device, shape := #[batch], dtype := .Int64 } := (Tensor.sumDim (Tensor.toInt64 placeholderMask2d) 1 false).reshapeChecked #[batch]
    let placeholderLens ← Tensor.toUInt64Array placeholderLensTensor
    let (audioLensOpt, _audioAttnMask, audioFeaturesRaw) ←
      match featureAttentionMask with
      | some fm =>
        let featureLens ← featureLengthsFromFeatureMask cfg fm
        let audioLens := AudioEncoderConfig.featExtractOutputLengths featureLens
        let m2d ← audioAttentionMaskFromFeatureMask cfg (audioSeq := audioSeq) fm
        let raw := encodeAudioVarLen m feats featureLens
        pure (some audioLens, some m2d, raw)
      | none =>
        let raw := encodeAudio m feats none
        pure (none, none, raw)

    let expectedAudioLens :=
      match audioLensOpt with
      | some xs => xs.map (fun l => if l <= audioSeq then l else audioSeq)
      | none => Array.replicate batch.toNat audioSeq
    let expectedRows := batch.toNat
    if expectedAudioLens.size != expectedRows then
      throw <| IO.userError
        s!"audio expected-length row mismatch: expected {expectedRows}, got {expectedAudioLens.size}"
    if placeholderLens.size != expectedRows then
      throw <| IO.userError
        s!"audio placeholder row mismatch: expected {expectedRows}, got {placeholderLens.size}"
    for i in [:expectedRows] do
      let got := placeholderLens[i]!
      let expected := expectedAudioLens[i]!
      if got = expected then
        pure ()
      else if got = 0 && expected > 0 then
        throw <| IO.userError
          s!"audio placeholder mismatch at batch {i}: expected {expected}, got 0 (silent audio-drop)"
      else if got < expected then
        throw <| IO.userError
          s!"audio placeholder mismatch at batch {i}: expected {expected}, got {got} (too few placeholders)"
      else
        throw <| IO.userError
          s!"audio placeholder mismatch at batch {i}: expected {expected}, got {got} (too many placeholders)"

    let audioFeatures := projectAudio m audioFeaturesRaw
    let source ←
      match featureAttentionMask with
      | some fm =>
        let validMask ← audioValidMaskFromFeatureMask cfg (audioSeq := audioSeq) fm
        let validMask :=
          validMask.to audioFeatures.device
        pure (Tensor.maskedSelect audioFeatures validMask)
      | none =>
        pure ⟨batch * audioSeq * cfg.textConfig.hiddenSize, Tensor.reshapeChecked audioFeatures #[batch * audioSeq * cfg.textConfig.hiddenSize]⟩
    let placeholderMask := getPlaceholderMask cfg inputIds
    pure (Tensor.maskedScatter inputsEmbeds0 placeholderMask source.2)

/-- Port of reference `get_rope_index` calculation from attention masks. -/
def getRopeIndex {batch seq : UInt64}
    (_m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (attentionMask : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : IO (Tensor { device := device, shape := #[3, batch, seq], dtype := .Int64 } × Tensor { device := device, shape := #[batch, 1], dtype := .Int64 }) := do
  let flat : Tensor { device := device, shape := #[batch * seq], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.toInt64 attentionMask) #[batch * seq]
  let vals ← Tensor.toUInt64Array flat

  let mut posVals : Array Int64 := Array.mkEmpty (3 * batch.toNat * seq.toNat)
  let mut deltaVals : Array Int64 := Array.mkEmpty batch.toNat

  for b in [:batch.toNat] do
    let mut row : Array UInt64 := Array.mkEmpty seq.toNat
    let mut running : UInt64 := 0
    let mut maxPos : UInt64 := 1
    let mut validCount : UInt64 := 0

    for t in [:seq.toNat] do
      let v := vals.getD (b * seq.toNat + t) 0
      if v == 0 then
        row := row.push 1
      else
        let p := running
        row := row.push p
        running := running + 1
        validCount := validCount + 1
        if p > maxPos then
          maxPos := p

    let delta := (maxPos + 1) - validCount
    deltaVals := deltaVals.push (Int64.ofNat delta.toNat)

    for _ in [:3] do
      for p in row do
        posVals := posVals.push (Int64.ofNat p.toNat)

  let positionIds : Tensor { device := device, shape := #[3, batch, seq], dtype := .Int64 } := Tensor.reshapeChecked ((Tensor.fromInt64Array posVals).to device) #[3, batch, seq]
  let ropeDeltas : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 } := Tensor.reshapeChecked ((Tensor.fromInt64Array deltaVals).to device) #[batch, 1]
  pure (positionIds, ropeDeltas)

abbrev LayerKVCache (cfg : ThinkerConfig) (batch : UInt64) :=
  qwen.QwenAttention.KVCache (device := device) batch cfg.textConfig.numKeyValueHeads cfg.textConfig.headDim (.Unknown "dynamic")

private def initLayerKVCaches {batch : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (maxLen : UInt64)
    (device : Device)
    : Array (LayerKVCache (device := device) cfg batch) :=
  m.textModel.layers.map (fun _ =>
    qwen.QwenAttention.initKVCache
      maxLen
      (batch := batch)
      (num_kv_heads := cfg.textConfig.numKeyValueHeads)
      (head_dim := cfg.textConfig.headDim)
      device (.Unknown "dynamic"))

private def precomputeDecodeRotary {maxLen : UInt64}
    (_m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (device : Device)
    : Tensor { device := device, shape := #[maxLen, cfg.textConfig.headDim / 2], dtype := .Float32 } × Tensor { device := device, shape := #[maxLen, cfg.textConfig.headDim / 2], dtype := .Float32 } :=
  qwen.computeFreqsOnDevice
    maxLen
    cfg.textConfig.headDim
    cfg.textConfig.ropeTheta
    device

private def decodeStepFromEmbedWithCache {batch maxLen : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (cosAll : Tensor { device := device, shape := #[maxLen, cfg.textConfig.headDim / 2], dtype := .Float32 })
    (sinAll : Tensor { device := device, shape := #[maxLen, cfg.textConfig.headDim / 2], dtype := .Float32 })
    (tokenEmbed : Tensor { device := device, shape := #[batch, 1, cfg.textConfig.hiddenSize] })
    (position : UInt64)
    (caches : Array (LayerKVCache (device := device) cfg batch))
    : IO (Tensor { device := device, shape := #[batch, ThinkerLmVocabSize cfg] } × Array (LayerKVCache (device := device) cfg batch)) := do
  let cos : Tensor { device := device, shape := #[1, cfg.textConfig.headDim / 2], dtype := .Float32 } := Tensor.sliceChecked cosAll 0 position 1 #[1, cfg.textConfig.headDim / 2]
  let sin : Tensor { device := device, shape := #[1, cfg.textConfig.headDim / 2], dtype := .Float32 } := Tensor.sliceChecked sinAll 0 position 1 #[1, cfg.textConfig.headDim / 2]

  let mut hidden : Tensor { device := device, shape := #[batch, 1, cfg.textConfig.hiddenSize] } := tokenEmbed
  let mut nextCaches := caches

  for i in [:m.textModel.layers.size] do
    let layer ←
      match m.textModel.layers[i]? with
      | some l => pure l
      | none => throw <| IO.userError s!"missing thinker text layer at index {i}"
    let cache ←
      match nextCaches[i]? with
      | some c => pure c
      | none => throw <| IO.userError s!"missing thinker KV cache at index {i}"
    let (hNext, cNext) := layer.forwardStep hidden cos sin cache
    hidden := hNext
    nextCaches := nextCaches.set! i cNext

  let hiddenNorm := m.textModel.norm.forward3dPreserving hidden
  let logits3 : Tensor { device := device, shape := #[batch, 1, ThinkerLmVocabSize cfg] } := Tensor.linear3d hiddenNorm m.lmHead
  let logits2 : Tensor { device := device, shape := #[batch, ThinkerLmVocabSize cfg] } := Tensor.reshapeChecked logits3 #[batch, ThinkerLmVocabSize cfg]
  pure (logits2, nextCaches)

private partial def prefillCachesFromEmbeds {batch seq maxLen : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (cosAll : Tensor { device := device, shape := #[maxLen, cfg.textConfig.headDim / 2], dtype := .Float32 })
    (sinAll : Tensor { device := device, shape := #[maxLen, cfg.textConfig.headDim / 2], dtype := .Float32 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.textConfig.hiddenSize] })
    (caches : Array (LayerKVCache (device := device) cfg batch))
    (position : Nat)
    (lastLogits : Tensor { device := device, shape := #[batch, ThinkerLmVocabSize cfg] })
    : IO (Tensor { device := device, shape := #[batch, ThinkerLmVocabSize cfg] } × Array (LayerKVCache (device := device) cfg batch)) := do
  if position >= seq.toNat then
    pure (lastLogits, caches)
  else
    let tok : Tensor { device := device, shape := #[batch, 1, cfg.textConfig.hiddenSize] } := Tensor.sliceChecked inputsEmbeds 1 position.toUInt64 1 #[batch, 1, cfg.textConfig.hiddenSize]
    let (logits, caches') ← decodeStepFromEmbedWithCache m cosAll sinAll tok position.toUInt64 caches
    prefillCachesFromEmbeds m cosAll sinAll inputsEmbeds caches' (position + 1) logits

private partial def greedyLoopCached {batch maxLen : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (cosAll : Tensor { device := device, shape := #[maxLen, cfg.textConfig.headDim / 2], dtype := .Float32 })
    (sinAll : Tensor { device := device, shape := #[maxLen, cfg.textConfig.headDim / 2], dtype := .Float32 })
    (eosTokenIds : Array UInt64)
    (eosVector : Option (Tensor { device := device, shape := #[batch], dtype := .Int64 }))
    (remaining : Nat)
    (caches : Array (LayerKVCache (device := device) cfg batch))
    (lastLogits : Tensor { device := device, shape := #[batch, ThinkerLmVocabSize cfg] })
    (finished : Option (Tensor { device := device, shape := #[batch], dtype := .Bool }))
    {curSeq : UInt64}
    (curIds : Tensor { device := device, shape := #[batch, curSeq], dtype := .Int64 })
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩

  let nextTokRaw : Tensor { device := device, shape := #[batch], dtype := .Int64 } := (Tensor.argmax lastLogits 1).reshapeChecked #[batch]
  let nextTok : Tensor { device := device, shape := #[batch], dtype := .Int64 } :=
    match eosVector, finished with
    | some eosTok, some doneMask =>
      let activeMask : Tensor { device := device, shape := #[batch], dtype := .Bool } := Tensor.logicalNot doneMask
      Tensor.whereSelect activeMask nextTokRaw eosTok
    | _, _ => nextTokRaw
  let nextCol : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 } := Tensor.reshapeChecked nextTok #[batch, 1]
  let appended : Tensor { device := device, shape := #[batch, curSeq + 1], dtype := .Int64 } := Tensor.cat curIds nextCol 1

  match eosVector with
  | none =>
    let nextEmb : Tensor { device := device, shape := #[batch, 1, cfg.textConfig.hiddenSize] } := m.embedText nextCol
    let (nextLogits, caches') ←
      decodeStepFromEmbedWithCache m cosAll sinAll nextEmb curSeq caches
    greedyLoopCached m cosAll sinAll eosTokenIds none (remaining - 1) caches' nextLogits none appended
  | some _ =>
    let reachedEos : Tensor { device := device, shape := #[batch], dtype := .Bool } := tokenInSetMask nextTok eosTokenIds
    let finished' : Tensor { device := device, shape := #[batch], dtype := .Bool } :=
      match finished with
      | some doneMask => Tensor.logicalOr doneMask reachedEos
      | none => reachedEos
    let hasActiveRows : Bool := Tensor.any (Tensor.logicalNot finished')
    if !hasActiveRows then
      return ⟨curSeq + 1, appended⟩
    else
      let nextEmb : Tensor { device := device, shape := #[batch, 1, cfg.textConfig.hiddenSize] } := m.embedText nextCol
      let (nextLogits, caches') ←
        decodeStepFromEmbedWithCache m cosAll sinAll nextEmb curSeq caches
      greedyLoopCached m cosAll sinAll eosTokenIds eosVector (remaining - 1) caches' nextLogits (some finished') appended

private def isUInt32Prefix (pref xs : Array UInt32) : Bool :=
  if pref.size > xs.size then
    false
  else
    Id.run do
      let mut ok := true
      for i in [:pref.size] do
        if ok && pref[i]! != xs[i]! then
          ok := false
      ok

/-- Reusable streaming prompt prefill state for batch=1 greedy decode.
    Carries text-model KV/cos/sin state for the current prompt prefix. -/
structure StreamingPromptCache (cfg : ThinkerConfig) where
  seq : UInt64
  maxLen : UInt64
  promptTokenIds : Array UInt32 := #[]
  promptEmbeds : Tensor { device := device, shape := #[1, seq, cfg.textConfig.hiddenSize] }
  cosAll : Tensor { device := device, shape := #[maxLen, cfg.textConfig.headDim / 2], dtype := .Float32 }
  sinAll : Tensor { device := device, shape := #[maxLen, cfg.textConfig.headDim / 2], dtype := .Float32 }
  kvCaches : Array (LayerKVCache (device := device) cfg 1)
  lastLogits : Tensor { device := device, shape := #[1, ThinkerLmVocabSize cfg] }

private def buildStreamingPromptCacheFromEmbeds {seq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputsEmbeds : Tensor { device := device, shape := #[1, seq, cfg.textConfig.hiddenSize] })
    (promptTokenIds : Array UInt32)
    (maxNewTokens : UInt64)
    : IO (StreamingPromptCache (device := device) cfg) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedyWithPromptCache requires non-empty prompt sequence"
  let reserve := if maxNewTokens == 0 then 1 else maxNewTokens
  let maxLen : UInt64 := seq + reserve
  let cacheDevice := inputsEmbeds.device
  let (cosAll, sinAll) := precomputeDecodeRotary (maxLen := maxLen) m cacheDevice
  let caches0 := initLayerKVCaches m maxLen cacheDevice
  let tok0 : Tensor { device := device, shape := #[1, 1, cfg.textConfig.hiddenSize] } := Tensor.sliceChecked inputsEmbeds 1 0 1 #[1, 1, cfg.textConfig.hiddenSize]
  let (logits0, caches1) ← decodeStepFromEmbedWithCache m cosAll sinAll tok0 0 caches0
  let (lastLogits, cachesPrefill) ←
    prefillCachesFromEmbeds m cosAll sinAll inputsEmbeds caches1 1 logits0
  pure {
    seq := seq
    maxLen := maxLen
    promptTokenIds := promptTokenIds
    promptEmbeds := inputsEmbeds
    cosAll := cosAll
    sinAll := sinAll
    kvCaches := cachesPrefill
    lastLogits := lastLogits
  }

private def canReuseStreamingPromptCache {seq : UInt64}
    (cache : StreamingPromptCache (device := device) cfg)
    (_inputsEmbeds : Tensor { device := device, shape := #[1, seq, cfg.textConfig.hiddenSize] })
    (promptTokenIds : Array UInt32)
    (maxNewTokens : UInt64)
    : IO Bool := do
  if cache.seq > seq then
    pure false
  else if !isUInt32Prefix cache.promptTokenIds promptTokenIds then
    pure false
  else
    let reserve := if maxNewTokens == 0 then 1 else maxNewTokens
    let neededMax := seq + reserve
    if cache.maxLen < neededMax then
      pure false
    else
      pure true

private def extendStreamingPromptCacheFromEmbeds {seq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (cache : StreamingPromptCache (device := device) cfg)
    (inputsEmbeds : Tensor { device := device, shape := #[1, seq, cfg.textConfig.hiddenSize] })
    (promptTokenIds : Array UInt32)
    : IO (StreamingPromptCache (device := device) cfg) := do
  if h : cache.seq = seq then
    pure {
      cache with
      promptTokenIds := promptTokenIds
      promptEmbeds := by simpa [h] using inputsEmbeds
    }
  else
    let (lastLogits, kvCaches) ←
      prefillCachesFromEmbeds
        m
        cache.cosAll
        cache.sinAll
        inputsEmbeds
        cache.kvCaches
        cache.seq.toNat
        cache.lastLogits
    pure {
      cache with
      seq := seq
      promptTokenIds := promptTokenIds
      promptEmbeds := inputsEmbeds
      kvCaches := kvCaches
      lastLogits := lastLogits
    }

/-- Lean analogue of `prepare_inputs_for_generation` behavior for audio features:
    only feed audio on the first generation step (`cachePosition == 0`). -/
def prepareInputsForGeneration {batch seq frames : UInt64}
    (_m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputFeatures : Option (Tensor { device := device, shape := #[batch, cfg.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[batch, frames], dtype := .Int64 }) := none)
    (cachePosition : UInt64 := 0)
    : ThinkerGenerationInputs (device := device) cfg batch seq frames :=
  if cachePosition == 0 then
    { inputIds, inputFeatures, featureAttentionMask }
  else
    { inputIds, inputFeatures := none, featureAttentionMask }

/-- Faithful thinker forward shape: text embeddings with in-place audio placeholder replacement. -/
def forwardWithAux {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputFeatures : Option (Tensor { device := device, shape := #[batch, cfg.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[batch, frames], dtype := .Int64 }) := none)
    (attentionMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Qwen3ASRThinkerCausalLMOutput (device := device) cfg batch seq) := do
  let inputsEmbeds ← buildInputsEmbeds m inputIds inputFeatures featureAttentionMask
  let logits ←
    match attentionMask with
    | none => pure (forwardEmbeds m inputsEmbeds none)
    | some mask => forwardEmbedsMaskedWithPositionRows m inputsEmbeds mask

  let ropeDeltas ←
    match attentionMask with
    | none => pure none
    | some mask =>
      let (_positionIds, ropeRaw) ← m.getRopeIndex mask
      let rawVals : Array UInt64 ←
        Tensor.toUInt64Array (Tensor.reshapeChecked (Tensor.toInt64 ropeRaw) #[batch])
      let validCounts ← maskRowCounts mask
      let mut out : Array Int64 := Array.mkEmpty batch.toNat
      for b in [:batch.toNat] do
        let raw := rawVals.getD b 0
        let valid := validCounts.getD b 0
        let pad := seq - valid
        let adj := if raw >= pad then raw - pad else 0
        out := out.push (Int64.ofNat adj.toNat)
      let deltas : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 } := Tensor.reshapeChecked ((Tensor.fromInt64Array out).to device) #[batch, 1]
      pure (some deltas)

  pure { logits, ropeDeltas }

def forward {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputFeatures : Option (Tensor { device := device, shape := #[batch, cfg.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[batch, frames], dtype := .Int64 }) := none)
    (attentionMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Tensor { device := device, shape := #[batch, seq, ThinkerLmVocabSize cfg] }) := do
  return (← m.forwardWithAux inputIds inputFeatures featureAttentionMask attentionMask).logits

private partial def greedyLoopUncached {batch : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    {promptSeq : UInt64}
    (promptEmbeds : Tensor { device := device, shape := #[batch, promptSeq, cfg.textConfig.hiddenSize] })
    (eosTokenIds : Array UInt64)
    (eosVector : Option (Tensor { device := device, shape := #[batch], dtype := .Int64 }))
    (remaining : Nat)
    (finished : Option (Tensor { device := device, shape := #[batch], dtype := .Bool }))
    {curSeq : UInt64}
    (curIds : Tensor { device := device, shape := #[batch, curSeq], dtype := .Int64 })
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩
  if curSeq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"
  if curSeq < promptSeq then
    throw <| IO.userError
      s!"generateGreedyUncached prompt boundary mismatch: promptSeq={promptSeq}, curSeq={curSeq}"

  let inputsEmbeds : Tensor { device := device, shape := #[batch, curSeq, cfg.textConfig.hiddenSize] } :=
    if h : curSeq = promptSeq then
      by simpa [h] using promptEmbeds
    else
      let suffixSeq := curSeq - promptSeq
      let suffixIds : Tensor { device := device, shape := #[batch, suffixSeq], dtype := .Int64 } := Tensor.sliceChecked curIds 1 promptSeq suffixSeq #[batch, suffixSeq]
      let suffixEmbeds : Tensor { device := device, shape := #[batch, suffixSeq, cfg.textConfig.hiddenSize] } := m.embedText suffixIds
      (Tensor.cat promptEmbeds suffixEmbeds 1).reshapeChecked #[batch, curSeq, cfg.textConfig.hiddenSize]
  let logits := m.forwardEmbeds inputsEmbeds none
  let lastPos := curSeq - 1
  let last3 : Tensor { device := device, shape := #[batch, 1, ThinkerLmVocabSize cfg] } :=
    Tensor.reshapeChecked (Tensor.sliceChecked logits 1 lastPos 1 #[batch, 1, ThinkerLmVocabSize cfg]) #[batch, 1, ThinkerLmVocabSize cfg]
  let last2 : Tensor { device := device, shape := #[batch, ThinkerLmVocabSize cfg] } := Tensor.reshapeChecked last3 #[batch, ThinkerLmVocabSize cfg]
  let nextTokRaw : Tensor { device := device, shape := #[batch], dtype := .Int64 } := (Tensor.argmax last2 1).reshapeChecked #[batch]
  let nextTok : Tensor { device := device, shape := #[batch], dtype := .Int64 } :=
    match eosVector, finished with
    | some eosTok, some doneMask =>
      let activeMask : Tensor { device := device, shape := #[batch], dtype := .Bool } := Tensor.logicalNot doneMask
      Tensor.whereSelect activeMask nextTokRaw eosTok
    | _, _ => nextTokRaw

  let nextCol : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 } := Tensor.reshapeChecked nextTok #[batch, 1]
  let appended : Tensor { device := device, shape := #[batch, curSeq + 1], dtype := .Int64 } := Tensor.cat curIds nextCol 1

  match eosVector with
  | none =>
    greedyLoopUncached
      m promptEmbeds eosTokenIds none (remaining - 1) none appended
  | some _ =>
    let reachedEos : Tensor { device := device, shape := #[batch], dtype := .Bool } := tokenInSetMask nextTok eosTokenIds
    let finished' : Tensor { device := device, shape := #[batch], dtype := .Bool } :=
      match finished with
      | some doneMask => Tensor.logicalOr doneMask reachedEos
      | none => reachedEos
    let hasActiveRows : Bool := Tensor.any (Tensor.logicalNot finished')
    if !hasActiveRows then
      return ⟨curSeq + 1, appended⟩
    else
      greedyLoopUncached
        m
        promptEmbeds
        eosTokenIds
        eosVector
        (remaining - 1)
        (some finished')
        appended

/-- Compatibility path mirroring the previous full re-forward greedy loop.
    Useful for parity checks against the cached generation path. -/
def generateGreedyUncached {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputFeatures : Option (Tensor { device := device, shape := #[batch, cfg.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[batch, frames], dtype := .Int64 }) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩
  let promptEmbeds ← buildInputsEmbeds m inputIds inputFeatures featureAttentionMask
  let eosVector := eosVectorOnDevice (batch := batch) eosTokenIds inputIds.device
  greedyLoopUncached
    m
    promptEmbeds
    eosTokenIds
    eosVector
    maxNewTokens.toNat
    none
    inputIds

/-- Greedy generation utility (Lean-only, no vLLM dependency).
    Returns generated full sequence `[prompt || new_tokens]` with dynamic output length. -/
def generateGreedy {batch seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputFeatures : Option (Tensor { device := device, shape := #[batch, cfg.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[batch, frames], dtype := .Int64 }) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedy requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let inputsEmbeds ← buildInputsEmbeds m inputIds inputFeatures featureAttentionMask
  let cacheDevice := inputsEmbeds.device
  let cacheMaxLen : UInt64 := seq + maxNewTokens
  let eosVector := eosVectorOnDevice (batch := batch) eosTokenIds cacheDevice
  let (cosAll, sinAll) := precomputeDecodeRotary (maxLen := cacheMaxLen) m cacheDevice
  let caches0 := initLayerKVCaches m cacheMaxLen cacheDevice
  let tok0 : Tensor { device := device, shape := #[batch, 1, cfg.textConfig.hiddenSize] } := Tensor.sliceChecked inputsEmbeds 1 0 1 #[batch, 1, cfg.textConfig.hiddenSize]
  let (logits0, caches1) ← decodeStepFromEmbedWithCache m cosAll sinAll tok0 0 caches0
  let (lastLogits, cachesPrefill) ←
    prefillCachesFromEmbeds m cosAll sinAll inputsEmbeds caches1 1 logits0
  greedyLoopCached m cosAll sinAll eosTokenIds eosVector maxNewTokens.toNat cachesPrefill lastLogits none inputIds

/-- Greedy generation with reusable batch=1 prompt-prefix cache.
    Intended for streaming decode where successive prompts typically extend
    previously accepted text prefixes. -/
def generateGreedyFromInputsEmbedsWithPromptCache {seq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[1, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[1, seq, cfg.textConfig.hiddenSize] })
    (promptTokenIds : Array UInt32)
    (prefixCache : Option (StreamingPromptCache (device := device) cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ((Sigma (fun outSeq => Tensor { device := device, shape := #[1, outSeq], dtype := .Int64 })) × StreamingPromptCache (device := device) cfg) := do
  if seq == 0 then
    throw <| IO.userError "generateGreedyWithPromptCache requires non-empty prompt sequence"
  if promptTokenIds.size != seq.toNat then
    throw <| IO.userError
      s!"generateGreedyWithPromptCache prompt-id mismatch: tensor_seq={seq}, prompt_ids={promptTokenIds.size}"
  if maxNewTokens == 0 then
    let cache ←
      match prefixCache with
      | some c =>
        if (← canReuseStreamingPromptCache c inputsEmbeds promptTokenIds maxNewTokens) then
          extendStreamingPromptCacheFromEmbeds m c inputsEmbeds promptTokenIds
        else
          buildStreamingPromptCacheFromEmbeds m inputsEmbeds promptTokenIds maxNewTokens
      | none =>
        buildStreamingPromptCacheFromEmbeds m inputsEmbeds promptTokenIds maxNewTokens
    pure (⟨seq, inputIds⟩, cache)
  else
    let cache ←
      match prefixCache with
      | some c =>
        if (← canReuseStreamingPromptCache c inputsEmbeds promptTokenIds maxNewTokens) then
          extendStreamingPromptCacheFromEmbeds m c inputsEmbeds promptTokenIds
        else
          buildStreamingPromptCacheFromEmbeds m inputsEmbeds promptTokenIds maxNewTokens
      | none =>
        buildStreamingPromptCacheFromEmbeds m inputsEmbeds promptTokenIds maxNewTokens
    let generated ←
      greedyLoopCached
        m
        cache.cosAll
        cache.sinAll
        eosTokenIds
        (eosVectorOnDevice (batch := 1) eosTokenIds inputIds.device)
        maxNewTokens.toNat
        cache.kvCaches
        cache.lastLogits
        none
        inputIds
    pure (generated, cache)

/-- Greedy generation with reusable batch=1 prompt-prefix cache.
    Intended for streaming decode where successive prompts typically extend
    previously accepted text prefixes. -/
def generateGreedyWithPromptCache {seq frames : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[1, seq], dtype := .Int64 })
    (inputFeatures : Option (Tensor { device := device, shape := #[1, cfg.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[1, frames], dtype := .Int64 }) := none)
    (promptTokenIds : Array UInt32)
    (prefixCache : Option (StreamingPromptCache (device := device) cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ((Sigma (fun outSeq => Tensor { device := device, shape := #[1, outSeq], dtype := .Int64 })) × StreamingPromptCache (device := device) cfg) := do
  let inputsEmbeds ← buildInputsEmbeds m inputIds inputFeatures featureAttentionMask
  m.generateGreedyFromInputsEmbedsWithPromptCache
    inputIds
    inputsEmbeds
    promptTokenIds
    prefixCache
    maxNewTokens
    eosTokenIds

def forwardFromMel {batch frames textSeq : UInt64}
    (m : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg)
    (inputFeatures : Tensor { device := device, shape := #[batch, cfg.audioConfig.numMelBins, frames] })
    (textIds : Tensor { device := device, shape := #[batch, textSeq], dtype := .Int64 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames + textSeq], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg.audioConfig frames + textSeq, ThinkerLmVocabSize cfg] } :=
  let audio := m.encodeAudio inputFeatures
  m.forwardAudioText audio textIds attnMask

end Qwen3ASRThinkerForConditionalGeneration

/-- Top-level Qwen3-ASR model wrapper (thinker-only inference in this Lean port). -/
structure Qwen3ASRForConditionalGeneration (cfg : Qwen3ASRConfig) where
  thinker : Qwen3ASRThinkerForConditionalGeneration (device := device) cfg.thinkerConfig
  supportLanguages : Array String := cfg.supportLanguages
  deriving TensorStruct

namespace Qwen3ASRForConditionalGeneration

def init (cfg : Qwen3ASRConfig) (device : Device := .CPU) : IO (Qwen3ASRForConditionalGeneration (device := device) cfg) := do
  let thinker ← Qwen3ASRThinkerForConditionalGeneration.init (device := device) cfg.thinkerConfig
  pure { thinker, supportLanguages := cfg.supportLanguages }

def getSupportedLanguages (m : Qwen3ASRForConditionalGeneration (device := device) cfg) : Array String :=
  m.supportLanguages

def forwardText {batch textSeq : UInt64}
    (m : Qwen3ASRForConditionalGeneration (device := device) cfg)
    (textIds : Tensor { device := device, shape := #[batch, textSeq], dtype := .Int64 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, textSeq], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, textSeq, ThinkerLmVocabSize cfg.thinkerConfig] } :=
  m.thinker.forwardText textIds attnMask

def forwardFromMel {batch frames textSeq : UInt64}
    (m : Qwen3ASRForConditionalGeneration (device := device) cfg)
    (inputFeatures : Tensor { device := device, shape := #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames] })
    (textIds : Tensor { device := device, shape := #[batch, textSeq], dtype := .Int64 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames + textSeq], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, AudioEncoderConfig.framesAfterConv3 cfg.thinkerConfig.audioConfig frames + textSeq, ThinkerLmVocabSize cfg.thinkerConfig] } :=
  m.thinker.forwardFromMel inputFeatures textIds attnMask

def forwardWithAux {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputFeatures : Option (Tensor { device := device, shape := #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[batch, frames], dtype := .Int64 }) := none)
    (attentionMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Qwen3ASRThinkerCausalLMOutput (device := device) cfg.thinkerConfig batch seq) :=
  m.thinker.forwardWithAux inputIds inputFeatures featureAttentionMask attentionMask

def forward {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputFeatures : Option (Tensor { device := device, shape := #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[batch, frames], dtype := .Int64 }) := none)
    (attentionMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Tensor { device := device, shape := #[batch, seq, ThinkerLmVocabSize cfg.thinkerConfig] }) :=
  m.thinker.forward inputIds inputFeatures featureAttentionMask attentionMask

def generateGreedy {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputFeatures : Option (Tensor { device := device, shape := #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[batch, frames], dtype := .Int64 }) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  m.thinker.generateGreedy inputIds inputFeatures featureAttentionMask maxNewTokens eosTokenIds

abbrev StreamingPromptCache (cfg : Qwen3ASRConfig) :=
  Qwen3ASRThinkerForConditionalGeneration.StreamingPromptCache (device := device) cfg.thinkerConfig

/-- Greedy generation with reusable batch=1 streaming prompt cache. -/
def generateGreedyWithPromptCache {seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[1, seq], dtype := .Int64 })
    (inputFeatures : Option (Tensor { device := device, shape := #[1, cfg.thinkerConfig.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[1, frames], dtype := .Int64 }) := none)
    (promptTokenIds : Array UInt32)
    (prefixCache : Option (StreamingPromptCache (device := device) cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ((Sigma (fun outSeq => Tensor { device := device, shape := #[1, outSeq], dtype := .Int64 })) × StreamingPromptCache (device := device) cfg) :=
  m.thinker.generateGreedyWithPromptCache
    inputIds
    inputFeatures
    featureAttentionMask
    promptTokenIds
    prefixCache
    maxNewTokens
    eosTokenIds

/-- Greedy generation from precomputed input embeddings with reusable
    batch=1 streaming prompt cache. -/
def generateGreedyFromInputsEmbedsWithPromptCache {seq : UInt64}
    (m : Qwen3ASRForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[1, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[1, seq, cfg.thinkerConfig.textConfig.hiddenSize] })
    (promptTokenIds : Array UInt32)
    (prefixCache : Option (StreamingPromptCache (device := device) cfg) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO ((Sigma (fun outSeq => Tensor { device := device, shape := #[1, outSeq], dtype := .Int64 })) × StreamingPromptCache (device := device) cfg) :=
  m.thinker.generateGreedyFromInputsEmbedsWithPromptCache
    inputIds
    inputsEmbeds
    promptTokenIds
    prefixCache
    maxNewTokens
    eosTokenIds

def generateGreedyUncached {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputFeatures : Option (Tensor { device := device, shape := #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[batch, frames], dtype := .Int64 }) := none)
    (maxNewTokens : UInt64 := 512)
    (eosTokenIds : Array UInt64 := defaultEosTokenIds)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  m.thinker.generateGreedyUncached inputIds inputFeatures featureAttentionMask maxNewTokens eosTokenIds

def alignFromOutputIds {batch seq : UInt64}
    (_m : Qwen3ASRForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (outputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (wordLists : Array (Array String))
    (timestampTokenId : UInt64 := cfg.timestampTokenId)
    (timestampSegmentTime : Float := cfg.timestampSegmentTime)
    : IO (Array ForcedAlignResult) :=
  torch.qwen3asr.alignFromOutputIds inputIds outputIds wordLists timestampTokenId timestampSegmentTime

def alignFromLogits {batch seq : UInt64}
    (_m : Qwen3ASRForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (logits : Tensor { device := device, shape := #[batch, seq, ThinkerLmVocabSize cfg.thinkerConfig] })
    (wordLists : Array (Array String))
    (timestampTokenId : UInt64 := cfg.timestampTokenId)
    (timestampSegmentTime : Float := cfg.timestampSegmentTime)
    : IO (Array ForcedAlignResult) :=
  torch.qwen3asr.alignFromLogits inputIds logits wordLists timestampTokenId timestampSegmentTime

def alignPrepared {batch seq frames : UInt64}
    (m : Qwen3ASRForConditionalGeneration (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (wordLists : Array (Array String))
    (inputFeatures : Option (Tensor { device := device, shape := #[batch, cfg.thinkerConfig.audioConfig.numMelBins, frames] }) := none)
    (featureAttentionMask : Option (Tensor { device := device, shape := #[batch, frames], dtype := .Int64 }) := none)
    (attentionMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    (timestampTokenId : UInt64 := cfg.timestampTokenId)
    (timestampSegmentTime : Float := cfg.timestampSegmentTime)
    : IO (Array ForcedAlignResult) := do
  let logits ← m.forward inputIds inputFeatures featureAttentionMask attentionMask
  m.alignFromLogits inputIds logits wordLists timestampTokenId timestampSegmentTime

end Qwen3ASRForConditionalGeneration

end torch.qwen3asr
