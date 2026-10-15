/-
  Tyr/Model/Qwen3TTS/Talker.lean

  Lean4 port of Qwen3-TTS talker components:
  - talker text/codec decoder
  - sub-talker (codec group predictor)
  - conditional generation heads

  Note: This is a Lean-native port that preserves the architecture shape
  and interfaces, while using Tyr's existing transformer primitives.
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.RMSNorm
import Tyr.Model.Qwen.Layer
import Tyr.Model.Qwen.RoPE
import Tyr.Model.Qwen3TTS.Config
import Tyr.Model.Utils

namespace torch.qwen3tts

variable {device : Device}

private def initWeight (shape : Shape) (fanIn : UInt64) : IO (Tensor { shape := shape, device := device }) :=
  Model.initWeight (device := device) shape fanIn (.Unknown "dynamic")

abbrev TalkerLayer (cfg : TalkerConfig) :=
  qwen.QwenLayer (device := device) cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim cfg.intermediateSize (.Unknown "dynamic")

abbrev CodePredictorLayer (cfg : TalkerCodePredictorConfig) :=
  qwen.QwenLayer (device := device) cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim cfg.intermediateSize (.Unknown "dynamic")

private def getOrFirst! [Inhabited α] (xs : Array α) (i : Nat) : α :=
  match xs[i]? with
  | some x => x
  | none =>
    match xs[0]? with
    | some x => x
    | none => panic! "empty array"

private def ensureAtLeastOne (n : Nat) : Nat :=
  if n == 0 then 1 else n

private def optAtOr (vals : Option (Array α)) (idx : Nat) (fallback : α) : α :=
  (vals >>= fun xs => xs[idx]?).getD fallback

/-- Main talker decoder (text + codec embeddings -> hidden states). -/
structure TalkerModel (cfg : TalkerConfig) where
  codecEmbedding : Tensor { shape := #[cfg.vocabSize, cfg.hiddenSize], device := device }
  textEmbedding : Tensor { shape := #[cfg.textVocabSize, cfg.textHiddenSize], device := device }
  textProjectionFc1 : Tensor { shape := #[cfg.textHiddenSize, cfg.textHiddenSize], device := device }
  textProjectionFc1Bias : Tensor { shape := #[cfg.textHiddenSize], device := device }
  textProjectionFc2 : Tensor { shape := #[cfg.hiddenSize, cfg.textHiddenSize], device := device }
  textProjectionFc2Bias : Tensor { shape := #[cfg.hiddenSize], device := device }
  layers : Array (TalkerLayer (device := device) cfg)
  norm : RMSNorm (device := device) cfg.hiddenSize (.Unknown "dynamic")
  deriving TensorStruct

namespace TalkerModel

def init (cfg : TalkerConfig) (device : Device := .CPU) : IO (TalkerModel (device := device) cfg) := do
  let codecEmbedding ← initWeight (device := device) #[cfg.vocabSize, cfg.hiddenSize] cfg.hiddenSize
  let textEmbedding ← initWeight (device := device) #[cfg.textVocabSize, cfg.textHiddenSize] cfg.textHiddenSize
  let textProjectionFc1 ← initWeight (device := device) #[cfg.textHiddenSize, cfg.textHiddenSize] cfg.textHiddenSize
  let textProjectionFc1Bias ← initWeight (device := device) #[cfg.textHiddenSize] cfg.textHiddenSize
  let textProjectionFc2 ← initWeight (device := device) #[cfg.hiddenSize, cfg.textHiddenSize] cfg.textHiddenSize
  let textProjectionFc2Bias ← initWeight (device := device) #[cfg.hiddenSize] cfg.textHiddenSize

  let mut layers := #[]
  for _ in [:cfg.numHiddenLayers.toNat] do
    let layer ← qwen.QwenLayer.init (device := device)
      cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads
      cfg.headDim cfg.intermediateSize cfg.rmsNormEps (.Unknown "dynamic")
    layers := layers.push layer

  let norm ← RMSNorm.init (device := device) cfg.hiddenSize cfg.rmsNormEps (.Unknown "dynamic")
  pure {
    codecEmbedding
    textEmbedding
    textProjectionFc1
    textProjectionFc1Bias
    textProjectionFc2
    textProjectionFc2Bias
    layers
    norm
  }

/-- Embed codec token ids with talker codec embedding table. -/
def embedCodec {batch seq : UInt64} (m : TalkerModel (device := device) cfg)
    (codecIds : Tensor { shape := #[batch, seq], dtype := .Int64, device := device }) : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device } :=
  let ids : Tensor { shape := #[batch, seq], dtype := .Int64, device := device } :=
    codecIds.to m.codecEmbedding.device
  Tensor.embedding ids m.codecEmbedding

/-- Embed text token ids and project to talker hidden size. -/
def embedText {batch seq : UInt64} (m : TalkerModel (device := device) cfg)
    (textIds : Tensor { shape := #[batch, seq], dtype := .Int64, device := device }) : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device } :=
  let ids : Tensor { shape := #[batch, seq], dtype := .Int64, device := device } :=
    textIds.to m.textEmbedding.device
  let t := Tensor.embedding ids m.textEmbedding
  let h := Tensor.affine3d t m.textProjectionFc1 m.textProjectionFc1Bias
  let h := Tensor.Preserving.silu h
  Tensor.affine3d h m.textProjectionFc2 m.textProjectionFc2Bias

/-- Prepend codec BOS embedding to a text-conditioning embedding sequence. -/
def prependCodecBos {batch seq : UInt64} (cfg : TalkerConfig) (m : TalkerModel (device := device) cfg)
    (textEmbeds : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device }) : Tensor { shape := #[batch, 1 + seq, cfg.hiddenSize], device := device } :=
  let bosIds0 : Tensor { shape := #[batch, 1], dtype := .Int64, device := .CPU } := Tensor.fullInt #[batch, 1] (Int64.ofNat cfg.codecBosId.toNat)
  let bosIds : Tensor { shape := #[batch, 1], dtype := .Int64, device := device } := bosIds0.to textEmbeds.device
  let bosEmb : Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device } := embedCodec m bosIds
  Tensor.cat bosEmb textEmbeds 1

/-- Build talker input embeddings from text token IDs. -/
def buildInputsFromText {batch textSeq : UInt64} (cfg : TalkerConfig) (m : TalkerModel (device := device) cfg)
    (textIds : Tensor { shape := #[batch, textSeq], dtype := .Int64, device := device }) : Tensor { shape := #[batch, 1 + textSeq, cfg.hiddenSize], device := device } :=
  prependCodecBos cfg m (embedText m textIds)

/-- Build talker input embeddings from instruct + text token IDs. -/
def buildInputsFromInstructText {batch instructSeq textSeq : UInt64} (cfg : TalkerConfig) (m : TalkerModel (device := device) cfg)
    (instructIds : Tensor { shape := #[batch, instructSeq], dtype := .Int64, device := device })
    (textIds : Tensor { shape := #[batch, textSeq], dtype := .Int64, device := device })
    : Tensor { shape := #[batch, 1 + (instructSeq + textSeq), cfg.hiddenSize], device := device } :=
  let instructEmb := embedText m instructIds
  let textEmb := embedText m textIds
  let merged : Tensor { shape := #[batch, instructSeq + textSeq, cfg.hiddenSize], device := device } := Tensor.cat instructEmb textEmb 1
  prependCodecBos cfg m merged

/-- Forward over already-built talker input embeddings. -/
def forwardEmbeds {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerModel (device := device) cfg)
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64, device := device }) := none)
    : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device } :=
  let (cos0, sin0) := Tensor.rotaryFreqs seq cfg.headDim cfg.ropeTheta
  let cos : Tensor { shape := #[seq, cfg.headDim / 2], dtype := .Float32, device := device } :=
    cos0.to inputsEmbeds.device
  let sin : Tensor { shape := #[seq, cfg.headDim / 2], dtype := .Float32, device := device } :=
    sin0.to inputsEmbeds.device
  let hidden := match attnMask with
    | some mask =>
      m.layers.foldl
        (fun h layer => layer.forwardMasked h cos sin mask true)
        inputsEmbeds
    | none =>
      m.layers.foldl
        (fun h layer => layer.forward h cos sin true)
        inputsEmbeds
  m.norm.forward3d hidden

/-- Convenience forward from codec token ids. -/
def forwardCodecIds {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerModel (device := device) cfg)
    (codecIds : Tensor { shape := #[batch, seq], dtype := .Int64, device := device })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64, device := device }) := none)
    : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device } :=
  forwardEmbeds cfg m (embedCodec m codecIds) attnMask

end TalkerModel

/-- Sub-talker model that predicts residual codec groups. -/
structure TalkerCodePredictor (cfg : TalkerConfig) where
  inputEmbeddings : Array (Tensor { shape := #[cfg.codePredictorConfig.vocabSize, cfg.hiddenSize], device := device })
  inputProjection : Tensor { shape := #[cfg.codePredictorConfig.hiddenSize, cfg.hiddenSize], device := device }
  inputProjectionBias : Tensor { shape := #[cfg.codePredictorConfig.hiddenSize], device := device }
  layers : Array (CodePredictorLayer (device := device) cfg.codePredictorConfig)
  norm : RMSNorm (device := device) cfg.codePredictorConfig.hiddenSize (.Unknown "dynamic")
  lmHeads : Array (Tensor { shape := #[cfg.codePredictorConfig.vocabSize, cfg.codePredictorConfig.hiddenSize], device := device })
  deriving TensorStruct

namespace TalkerCodePredictor

def init (cfg : TalkerConfig) (device : Device := .CPU) : IO (TalkerCodePredictor (device := device) cfg) := do
  let numCodebooks := ensureAtLeastOne (cfg.numCodeGroups.toNat - 1)

  let mut inputEmbeddings : Array (Tensor { shape := #[cfg.codePredictorConfig.vocabSize, cfg.hiddenSize], device := device }) := #[]
  for _ in [:numCodebooks] do
    let emb ← initWeight (device := device) #[cfg.codePredictorConfig.vocabSize, cfg.hiddenSize] cfg.hiddenSize
    inputEmbeddings := inputEmbeddings.push emb

  let inputProjection ← initWeight (device := device)
    #[cfg.codePredictorConfig.hiddenSize, cfg.hiddenSize]
    cfg.hiddenSize
  let inputProjectionBias ← initWeight (device := device) #[cfg.codePredictorConfig.hiddenSize] cfg.hiddenSize

  let mut layers := #[]
  for _ in [:cfg.codePredictorConfig.numHiddenLayers.toNat] do
    let layer ← qwen.QwenLayer.init (device := device)
      cfg.codePredictorConfig.hiddenSize
      cfg.codePredictorConfig.numAttentionHeads
      cfg.codePredictorConfig.numKeyValueHeads
      cfg.codePredictorConfig.headDim
      cfg.codePredictorConfig.intermediateSize
      cfg.codePredictorConfig.rmsNormEps (.Unknown "dynamic")
    layers := layers.push layer

  let norm ← RMSNorm.init (device := device) cfg.codePredictorConfig.hiddenSize cfg.codePredictorConfig.rmsNormEps (.Unknown "dynamic")

  let mut lmHeads : Array (Tensor { shape := #[cfg.codePredictorConfig.vocabSize, cfg.codePredictorConfig.hiddenSize], device := device }) := #[]
  for _ in [:numCodebooks] do
    let head ← initWeight (device := device)
      #[cfg.codePredictorConfig.vocabSize, cfg.codePredictorConfig.hiddenSize]
      cfg.codePredictorConfig.hiddenSize
    lmHeads := lmHeads.push head

  pure {
    inputEmbeddings
    inputProjection
    inputProjectionBias
    layers
    norm
    lmHeads
  }

/-- Embed tokens for a specific residual codebook group. -/
def embedGroupTokens {batch seq : UInt64} (m : TalkerCodePredictor (device := device) cfg)
    (groupIdx : Nat) (tokenIds : Tensor { shape := #[batch, seq], dtype := .Int64, device := device }) : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device } :=
  let emb := getOrFirst! m.inputEmbeddings groupIdx
  Tensor.embedding tokenIds emb

/-- Project talker hidden state to sub-talker hidden size. -/
def projectInputs {batch seq : UInt64} (m : TalkerCodePredictor (device := device) cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device })
    : Tensor { shape := #[batch, seq, cfg.codePredictorConfig.hiddenSize], device := device } :=
  Tensor.affine3d x m.inputProjection m.inputProjectionBias

/-- Forward hidden states for the sub-talker decoder stack. -/
def forwardHidden {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerCodePredictor (device := device) cfg)
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64, device := device }) := none)
    : Tensor { shape := #[batch, seq, cfg.codePredictorConfig.hiddenSize], device := device } :=
  let hidden0 := projectInputs m inputsEmbeds
  let cpCfg := cfg.codePredictorConfig
  let (cos0, sin0) := Tensor.rotaryFreqs seq cpCfg.headDim cpCfg.ropeTheta
  let cos : Tensor { shape := #[seq, cpCfg.headDim / 2], dtype := .Float32, device := device } :=
    cos0.to hidden0.device
  let sin : Tensor { shape := #[seq, cpCfg.headDim / 2], dtype := .Float32, device := device } :=
    sin0.to hidden0.device
  let hidden := match attnMask with
    | some mask =>
      m.layers.foldl
        (fun h layer => layer.forwardMasked h cos sin mask true)
        hidden0
    | none =>
      m.layers.foldl
        (fun h layer => layer.forward h cos sin true)
        hidden0
  m.norm.forward3d hidden

/-- Compute logits for one residual codebook group. -/
def logitsForGroup {batch seq : UInt64} (m : TalkerCodePredictor (device := device) cfg)
    (groupIdx : Nat)
    (hidden : Tensor { shape := #[batch, seq, cfg.codePredictorConfig.hiddenSize], device := device })
    : Tensor { shape := #[batch, seq, cfg.codePredictorConfig.vocabSize], device := device } :=
  let head := getOrFirst! m.lmHeads groupIdx
  Tensor.linear3d hidden head

end TalkerCodePredictor

/-- Talker + sub-talker generation heads. -/
structure TalkerForConditionalGeneration (cfg : TalkerConfig) where
  model : TalkerModel (device := device) cfg
  codePredictor : TalkerCodePredictor (device := device) cfg
  codecHead : Tensor { shape := #[cfg.vocabSize, cfg.hiddenSize], device := device }
  deriving TensorStruct

namespace TalkerForConditionalGeneration

structure CodeGenerationOutput (batch maxFrames numCodeGroups : UInt64) where
  codes : Tensor { shape := #[batch, maxFrames, numCodeGroups], dtype := .Int64, device := device }
  lengths : Array UInt64

structure FrameGenerationOutput (batch numCodeGroups hiddenSize : UInt64) where
  codes : Tensor { shape := #[batch, numCodeGroups], dtype := .Int64, device := device }
  summedEmbedding : Tensor { shape := #[batch, 1, hiddenSize], device := device }

structure CodePredictorRoPECache (cfg : TalkerConfig) where
  cos : Tensor { shape := #[cfg.numCodeGroups + 1, cfg.codePredictorConfig.headDim / 2], dtype := .Float32, device := device }
  sin : Tensor { shape := #[cfg.numCodeGroups + 1, cfg.codePredictorConfig.headDim / 2], dtype := .Float32, device := device }

def init (cfg : TalkerConfig) (device : Device := .CPU) : IO (TalkerForConditionalGeneration (device := device) cfg) := do
  let model ← TalkerModel.init (device := device) cfg
  let codePredictor ← TalkerCodePredictor.init (device := device) cfg
  let codecHead ← initWeight (device := device) #[cfg.vocabSize, cfg.hiddenSize] cfg.hiddenSize
  pure { model, codePredictor, codecHead }

/-- Forward talker decoder and return first-codebook logits. -/
def forward {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerForConditionalGeneration (device := device) cfg)
    (talkerInputs : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64, device := device }) := none)
    : Tensor { shape := #[batch, seq, cfg.vocabSize], device := device } :=
  let hidden := TalkerModel.forwardEmbeds cfg m.model talkerInputs attnMask
  Tensor.linear3d hidden m.codecHead

private def applySampling {batch vocab : UInt64}
    (logits : Tensor { shape := #[batch, vocab], device := device })
    (temperature : Float)
    (topK : UInt64)
    (topP : Float)
    : IO (Tensor { shape := #[batch], dtype := .Int64, device := device }) := do
  if temperature <= 0.0 then
    throw <| IO.userError s!"sampling requires temperature > 0, got {temperature}"
  let scaled :=
    if temperature == 1.0 then logits
    else Tensor.Preserving.mulScalar logits (1.0 / temperature)
  let filtered :=
    if topK == 0 then scaled
    else Tensor.topKFilter scaled topK
  let filtered :=
    if topP >= 1.0 then filtered
    else Tensor.topPFilter filtered topP
  let probs := Tensor.Preserving.softmax filtered (-1)
  let sampled ← Tensor.multinomial probs 1 false
  pure (Tensor.reshapeChecked sampled #[batch])

private def applySuppressTail {batch vocab : UInt64}
    (logits : Tensor { shape := #[batch, vocab], device := device })
    (suppressTail : UInt64)
    (eosTokenId : UInt64)
    : Tensor { shape := #[batch, vocab], device := device } :=
  if suppressTail == 0 then
    logits
  else
    let start : UInt64 := if suppressTail >= vocab then 0 else vocab - suppressTail
    let idx0 : Tensor { shape := #[vocab], dtype := .Int64, device := .CPU } := Tensor.arange vocab
    let idx : Tensor { shape := #[vocab], dtype := .Int64, device := device } := idx0.to logits.device
    let startVec0 : Tensor { shape := #[vocab], dtype := .Int64, device := .CPU } := Tensor.fullInt #[vocab] (Int64.ofNat start.toNat)
    let startVec : Tensor { shape := #[vocab], dtype := .Int64, device := device } := startVec0.to logits.device
    let endVec0 : Tensor { shape := #[vocab], dtype := .Int64, device := .CPU } := Tensor.fullInt #[vocab] (Int64.ofNat vocab.toNat)
    let endVec : Tensor { shape := #[vocab], dtype := .Int64, device := device } := endVec0.to logits.device
    let geMask : Tensor { shape := #[vocab], dtype := .Bool, device := device } := Tensor.ge idx startVec
    let ltMask : Tensor { shape := #[vocab], dtype := .Bool, device := device } := Tensor.lt idx endVec
    let tailMask : Tensor { shape := #[vocab], dtype := .Bool, device := device } := Tensor.logicalAnd geMask ltMask
    let notEos : Tensor { shape := #[vocab], dtype := .Bool, device := device } := Tensor.logicalNot (Tensor.eqScalar idx (Int64.ofNat eosTokenId.toNat))
    let suppress1d : Tensor { shape := #[vocab], dtype := .Bool, device := device } := Tensor.logicalAnd tailMask notEos
    let suppress2d : Tensor { shape := #[batch, vocab], dtype := .Bool, device := device } := Tensor.expand (Tensor.reshapeChecked suppress1d #[1, vocab]) #[batch, vocab]
    Tensor.maskedFill logits suppress2d (-1e9)

private def applySuppressEos {batch vocab : UInt64}
    (logits : Tensor { shape := #[batch, vocab], device := device })
    (eosTokenId : UInt64)
    : Tensor { shape := #[batch, vocab], device := device } :=
  let idx0 : Tensor { shape := #[vocab], dtype := .Int64, device := .CPU } := Tensor.arange vocab
  let idx : Tensor { shape := #[vocab], dtype := .Int64, device := device } := idx0.to logits.device
  let eosMask1d : Tensor { shape := #[vocab], dtype := .Bool, device := device } := Tensor.eqScalar idx (Int64.ofNat eosTokenId.toNat)
  let eosMask2d : Tensor { shape := #[batch, vocab], dtype := .Bool, device := device } := Tensor.expand (Tensor.reshapeChecked eosMask1d #[1, vocab]) #[batch, vocab]
  -- Strictly below the tail-suppression fill (-1e9): when `suppressTail`
  -- covers the whole vocab (tiny test configs), every non-EOS logit is
  -- already -1e9 and an equal EOS fill would make the distribution
  -- uniform — EOS must stay unsampleable relative to allowed tokens.
  Tensor.maskedFill logits eosMask2d (-1e30)

private def applyRepetitionPenalty {batch vocab histLen : UInt64}
    (logits : Tensor { shape := #[batch, vocab], device := device })
    (history : Tensor { shape := #[batch, histLen], dtype := .Int64, device := device })
    (penalty : Float)
    : Tensor { shape := #[batch, vocab], device := device } :=
  if histLen == 0 || penalty == 1.0 || penalty <= 0.0 then
    logits
  else
    let gathered : Tensor { shape := #[batch, histLen], device := device } := Tensor.gather logits (-1) history
    let negMask : Tensor { shape := #[batch, histLen], dtype := .Bool, device := device } := Tensor.ltScalar gathered 0.0
    let penalizedNeg := Tensor.Preserving.mulScalar gathered penalty
    let penalizedPos := Tensor.Preserving.divScalar gathered penalty
    let penalized : Tensor { shape := #[batch, histLen], device := device } := Tensor.Promoting.whereSelect negMask penalizedNeg penalizedPos
    Tensor.scatter2d logits (-1) history penalized

/-- Sample a full codec frame from the current talker hidden state `[batch,1,hidden]`. -/
private def generateFrameFromLastHidden {batch : UInt64} (cfg : TalkerConfig)
    (m : TalkerForConditionalGeneration (device := device) cfg)
    (lastHidden : Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device })
    (historyFirstCodes : Option (Sigma fun histLen => Tensor { shape := #[batch, histLen], dtype := .Int64, device := device }) := none)
    (allowEos : Bool := true)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    (repetitionPenalty : Float := 1.05)
    (suppressTail : UInt64 := 1024)
    (subtalkerTemperature : Float := 0.9)
    (subtalkerTopK : UInt64 := 50)
    (subtalkerTopP : Float := 1.0)
    (subtalkerTemperaturesByGroup : Option (Array Float) := none)
    (subtalkerTopKsByGroup : Option (Array UInt64) := none)
    (subtalkerTopPsByGroup : Option (Array Float) := none)
    (cpRoPECache : Option (CodePredictorRoPECache (device := device) cfg) := none)
    : IO (FrameGenerationOutput (device := device) batch cfg.numCodeGroups cfg.hiddenSize) := do
  let firstLogits3 := Tensor.linear3d lastHidden m.codecHead
  let firstLogits0 : Tensor { shape := #[batch, cfg.vocabSize], device := device } := Tensor.reshapeChecked firstLogits3 #[batch, cfg.vocabSize]
  let firstLogits1 := applySuppressTail firstLogits0 suppressTail cfg.codecEosTokenId
  let firstLogits2 :=
    if allowEos then firstLogits1 else applySuppressEos firstLogits1 cfg.codecEosTokenId
  let firstLogits :=
    match historyFirstCodes with
    | some ⟨_, hist⟩ => applyRepetitionPenalty firstLogits2 hist repetitionPenalty
    | none => firstLogits2
  let firstCode ← applySampling firstLogits temperature topK topP
  let firstCode2d : Tensor { shape := #[batch, 1], dtype := .Int64, device := device } := Tensor.reshapeChecked firstCode #[batch, 1]

  -- Collect per-codebook sampled tokens as `[batch, 1]` columns.
  let mut tokenCols : Array (Tensor { shape := #[batch, 1], dtype := .Int64, device := device }) := #[firstCode2d]

  -- Build autoregressive sub-talker context from (past hidden + codec embeddings).
  -- Use static preallocated KV cache and step decoding instead of re-running
  -- full-prefix forward for each residual group.
  let firstEmb : Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device } := Tensor.embedding firstCode2d m.model.codecEmbedding
  let mut summedEmb : Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device } := firstEmb
  let residualGroups := cfg.numCodeGroups.toNat - 1
  if residualGroups > 0 then
    let cpCfg := cfg.codePredictorConfig
    if m.codePredictor.layers.size != cpCfg.numHiddenLayers.toNat then
      throw <| IO.userError
        s!"Code predictor layer/cache mismatch: model has {m.codePredictor.layers.size} layers but cfg.numHiddenLayers={cpCfg.numHiddenLayers}."

    let cpDevice := lastHidden.device
    let cpMaxSeq : UInt64 := cfg.numCodeGroups + 1
    let (cpCosAll, cpSinAll) : (Tensor { shape := #[cpMaxSeq, cpCfg.headDim / 2], dtype := .Float32, device := device } × Tensor { shape := #[cpMaxSeq, cpCfg.headDim / 2], dtype := .Float32, device := device }) :=
      match cpRoPECache with
      | some rope =>
          let cos : Tensor { shape := #[cpMaxSeq, cpCfg.headDim / 2], dtype := .Float32, device := device } :=
            rope.cos.to cpDevice
          let sin : Tensor { shape := #[cpMaxSeq, cpCfg.headDim / 2], dtype := .Float32, device := device } :=
            rope.sin.to cpDevice
          (cos, sin)
      | none =>
          let (cpCosRaw, cpSinRaw) := Tensor.rotaryFreqs cpMaxSeq cpCfg.headDim cpCfg.ropeTheta
          let cpCosAll : Tensor { shape := #[cpMaxSeq, cpCfg.headDim / 2], dtype := .Float32, device := device } :=
            cpCosRaw.to cpDevice
          let cpSinAll : Tensor { shape := #[cpMaxSeq, cpCfg.headDim / 2], dtype := .Float32, device := device } :=
            cpSinRaw.to cpDevice
          (cpCosAll, cpSinAll)

    let mut cpCachesInit : Array (qwen.QwenAttention.KVCache batch cpCfg.numKeyValueHeads cpCfg.headDim (.Unknown "dynamic")) := #[]
    for _ in [:cpCfg.numHiddenLayers.toNat] do
      cpCachesInit := cpCachesInit.push
        (qwen.QwenAttention.initKVCache
          cpMaxSeq
          (batch := batch) (num_kv_heads := cpCfg.numKeyValueHeads) (head_dim := cpCfg.headDim)
          cpDevice (.Unknown "dynamic"))

    let cpStep :
        Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device } →
        UInt64 →
        Array (qwen.QwenAttention.KVCache batch cpCfg.numKeyValueHeads cpCfg.headDim (.Unknown "dynamic")) →
        IO (Tensor { shape := #[batch, 1, cpCfg.hiddenSize], device := device } ×
          Array (qwen.QwenAttention.KVCache batch cpCfg.numKeyValueHeads cpCfg.headDim (.Unknown "dynamic"))) :=
      fun tokenEmbed pos caches => do
        let xProj : Tensor { shape := #[batch, 1, cpCfg.hiddenSize], device := device } := TalkerCodePredictor.projectInputs m.codePredictor tokenEmbed
        let cos : Tensor { shape := #[1, cpCfg.headDim / 2], dtype := .Float32, device := device } := Tensor.sliceChecked cpCosAll 0 pos 1 #[1, cpCfg.headDim / 2]
        let sin : Tensor { shape := #[1, cpCfg.headDim / 2], dtype := .Float32, device := device } := Tensor.sliceChecked cpSinAll 0 pos 1 #[1, cpCfg.headDim / 2]
        let mut h : Tensor { shape := #[batch, 1, cpCfg.hiddenSize], device := device } := xProj
        let mut caches := caches
        for i in [:m.codePredictor.layers.size] do
          let layer : CodePredictorLayer (device := device) cpCfg ←
            match m.codePredictor.layers[i]? with
            | some layer => pure layer
            | none => throw <| IO.userError s!"missing code predictor layer at index {i}"
          let cache : qwen.QwenAttention.KVCache batch cpCfg.numKeyValueHeads cpCfg.headDim (.Unknown "dynamic") ←
            match caches[i]? with
            | some cache => pure cache
            | none => throw <| IO.userError s!"missing code predictor cache at index {i}"
          let (h', cache') := qwen.QwenLayer.forwardStep layer h cos sin cache
          h := h'
          caches := caches.set! i cache'
        pure (m.codePredictor.norm.forward3d h, caches)

    let (_, cpCaches0) ← cpStep lastHidden 0 cpCachesInit
    let (cpHidden0, cpCaches1) ← cpStep firstEmb 1 cpCaches0
    let mut cpHiddenCur := cpHidden0
    let mut cpCaches := cpCaches1
    let mut cpPos : UInt64 := 2

    for g in [:residualGroups] do
      let cpLogits3 := TalkerCodePredictor.logitsForGroup m.codePredictor g cpHiddenCur
      let cpLogits : Tensor { shape := #[batch, cpCfg.vocabSize], device := device } := Tensor.reshapeChecked cpLogits3 #[batch, cpCfg.vocabSize]

      let gTemp : Float := optAtOr subtalkerTemperaturesByGroup g subtalkerTemperature
      let gTopK : UInt64 := optAtOr subtalkerTopKsByGroup g subtalkerTopK
      let gTopP : Float := optAtOr subtalkerTopPsByGroup g subtalkerTopP
      let nextCode ← applySampling cpLogits gTemp gTopK gTopP
      let nextCode2d : Tensor { shape := #[batch, 1], dtype := .Int64, device := device } := Tensor.reshapeChecked nextCode #[batch, 1]
      tokenCols := tokenCols.push nextCode2d

      let nextEmb := TalkerCodePredictor.embedGroupTokens m.codePredictor g nextCode2d
      summedEmb := Tensor.Promoting.add summedEmb nextEmb

      if g + 1 < residualGroups then
        let (cpHiddenNext, cpCachesNext) ← cpStep nextEmb cpPos cpCaches
        cpHiddenCur := cpHiddenNext
        cpCaches := cpCachesNext
        cpPos := cpPos + 1

  let frameDyn := Tensor.catManyChecked (tokenCols.map Tensor.packShape) 1 #[batch, cfg.numCodeGroups]
  pure {
    codes := Tensor.reshapeChecked frameDyn #[batch, cfg.numCodeGroups]
    summedEmbedding := summedEmb
  }

/-- Generate one frame of codec tokens `[batch, numCodeGroups]` from talker context.
    This mirrors the two-stage talker/sub-talker generation in Qwen3-TTS. -/
def generateFrame {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerForConditionalGeneration (device := device) cfg)
    (talkerInputs : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64, device := device }) := none)
    (historyFirstCodes : Option (Sigma fun histLen => Tensor { shape := #[batch, histLen], dtype := .Int64, device := device }) := none)
    (allowEos : Bool := true)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    (repetitionPenalty : Float := 1.05)
    (suppressTail : UInt64 := 1024)
    (subtalkerTemperature : Float := 0.9)
    (subtalkerTopK : UInt64 := 50)
    (subtalkerTopP : Float := 1.0)
    (subtalkerTemperaturesByGroup : Option (Array Float) := none)
    (subtalkerTopKsByGroup : Option (Array UInt64) := none)
    (subtalkerTopPsByGroup : Option (Array Float) := none)
    : IO (FrameGenerationOutput (device := device) batch cfg.numCodeGroups cfg.hiddenSize) := do
  if cfg.numCodeGroups == 0 then
    return {
      codes := (Tensor.fullInt #[batch, cfg.numCodeGroups] 0).to talkerInputs.device
      summedEmbedding := Tensor.castLike talkerInputs (Tensor.zeros #[batch, 1, cfg.hiddenSize] false talkerInputs.device)
    }

  let hidden := TalkerModel.forwardEmbeds cfg m.model talkerInputs attnMask
  let lastHidden := Tensor.sliceChecked hidden 1 (seq - 1) 1 #[batch, 1, cfg.hiddenSize]
  generateFrameFromLastHidden cfg m lastHidden historyFirstCodes allowEos
    temperature topK topP repetitionPenalty suppressTail
    subtalkerTemperature subtalkerTopK subtalkerTopP
    subtalkerTemperaturesByGroup subtalkerTopKsByGroup subtalkerTopPsByGroup

mutual

/-- Autoregressively generate a fixed number of codec frames.
    Returns `[batch, maxFrames, numCodeGroups]`. -/
def generateCodesWithLengths {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerForConditionalGeneration (device := device) cfg)
    (talkerInputs : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device })
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    (subtalkerTemperature : Float := 0.9)
    (subtalkerTopK : UInt64 := 50)
    (subtalkerTopP : Float := 1.0)
    (repetitionPenalty : Float := 1.05)
    (suppressTail : UInt64 := 1024)
    (trailingTextHidden : Option (Sigma fun trailingSeq => Tensor { shape := #[batch, trailingSeq, cfg.hiddenSize], device := device }) := none)
    (ttsPadEmbed : Option (Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device }) := none)
    (subtalkerTemperaturesByGroup : Option (Array Float) := none)
    (subtalkerTopKsByGroup : Option (Array UInt64) := none)
    (subtalkerTopPsByGroup : Option (Array Float) := none)
    : IO (CodeGenerationOutput (device := device) batch maxFrames cfg.numCodeGroups) := do
  if maxFrames == 0 then
    return {
      codes := (Tensor.fullInt #[batch, maxFrames, cfg.numCodeGroups] 0).to talkerInputs.device
      lengths := Array.replicate batch.toNat 0
    }

  let padFrame : Tensor { shape := #[batch, 1, cfg.numCodeGroups], dtype := .Int64, device := device } :=
    let x0 : Tensor { shape := #[batch, 1, cfg.numCodeGroups], dtype := .Int64, device := .CPU } :=
      Tensor.fullInt #[batch, 1, cfg.numCodeGroups] (Int64.ofNat cfg.codecPadId.toNat)
    x0.to talkerInputs.device

  let frameDynRef ← IO.mkRef (#[] : Array (Tensor { shape := #[batch, 1, cfg.numCodeGroups], dtype := .Int64, device := device }))
  let onFrame : UInt64 → Tensor { shape := #[batch, cfg.numCodeGroups], dtype := .Int64, device := device } → IO Unit := fun _ frame => do
    let frame3 : Tensor { shape := #[batch, 1, cfg.numCodeGroups], dtype := .Int64, device := device } := Tensor.reshapeChecked frame #[batch, 1, cfg.numCodeGroups]
    frameDynRef.modify (fun xs => xs.push frame3)

  let lengths ← streamCodes cfg m talkerInputs onFrame maxFrames minNewTokens
    temperature topK topP
    subtalkerTemperature subtalkerTopK subtalkerTopP
    repetitionPenalty suppressTail
    trailingTextHidden ttsPadEmbed
    subtalkerTemperaturesByGroup subtalkerTopKsByGroup subtalkerTopPsByGroup

  let mut framesDyn ← frameDynRef.get
  for _ in [framesDyn.size:maxFrames.toNat] do
    framesDyn := framesDyn.push padFrame
  let outDyn := Tensor.catManyChecked (framesDyn.map Tensor.packShape) 1 #[batch, maxFrames, cfg.numCodeGroups]
  let codes : Tensor { shape := #[batch, maxFrames, cfg.numCodeGroups], dtype := .Int64, device := device } := Tensor.reshapeChecked outDyn #[batch, maxFrames, cfg.numCodeGroups]
  pure { codes, lengths }

/-- Stream codec frames one step at a time.
    `onFrame` is called for each generated frame in generation order.
    Returns per-sample EOS lengths (same convention as `generateCodesWithLengths`). -/
def streamCodes {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerForConditionalGeneration (device := device) cfg)
    (talkerInputs : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device })
    (onFrame : UInt64 → Tensor { shape := #[batch, cfg.numCodeGroups], dtype := .Int64, device := device } → IO Unit)
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    (subtalkerTemperature : Float := 0.9)
    (subtalkerTopK : UInt64 := 50)
    (subtalkerTopP : Float := 1.0)
    (repetitionPenalty : Float := 1.05)
    (suppressTail : UInt64 := 1024)
    (trailingTextHidden : Option (Sigma fun trailingSeq => Tensor { shape := #[batch, trailingSeq, cfg.hiddenSize], device := device }) := none)
    (ttsPadEmbed : Option (Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device }) := none)
    (subtalkerTemperaturesByGroup : Option (Array Float) := none)
    (subtalkerTopKsByGroup : Option (Array UInt64) := none)
    (subtalkerTopPsByGroup : Option (Array Float) := none)
    : IO (Array UInt64) :=
  do
    let m := TensorStruct.detach m
    let talkerInputs := talkerInputs.detach
    let trailingTextHidden : Option (Sigma fun trailingSeq => Tensor { shape := #[batch, trailingSeq, cfg.hiddenSize], device := device }) :=
      trailingTextHidden.map fun packed => ⟨packed.1, packed.2.detach⟩
    let ttsPadEmbed := ttsPadEmbed.map Tensor.detach
    if maxFrames == 0 then
      return Array.replicate batch.toNat 0

    if m.model.layers.size != cfg.numHiddenLayers.toNat then
      throw <| IO.userError
        s!"Talker layer/cache mismatch: model has {m.model.layers.size} layers but cfg.numHiddenLayers={cfg.numHiddenLayers}."

    let cacheDevice := talkerInputs.device
    let cacheMaxLen : UInt64 := seq + maxFrames + 1
    let mut caches : Array (qwen.QwenAttention.KVCache batch cfg.numKeyValueHeads cfg.headDim (.Unknown "dynamic")) := #[]
    for _ in [:cfg.numHiddenLayers.toNat] do
      caches := caches.push
        (qwen.QwenAttention.initKVCache
          cacheMaxLen
          (batch := batch) (num_kv_heads := cfg.numKeyValueHeads) (head_dim := cfg.headDim)
          cacheDevice (.Unknown "dynamic"))

    -- Prefill KV cache from prompt/context so per-frame generation only runs one token step.
    let (cosRaw, sinRaw) := Tensor.rotaryFreqs (seq + maxFrames + 1) cfg.headDim cfg.ropeTheta
    let cosAll : Tensor { shape := #[seq + maxFrames + 1, cfg.headDim / 2], dtype := .Float32, device := device } :=
      cosRaw.to cacheDevice
    let sinAll : Tensor { shape := #[seq + maxFrames + 1, cfg.headDim / 2], dtype := .Float32, device := device } :=
      sinRaw.to cacheDevice
    let mut lastHidden : Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device } :=
      Tensor.castLike talkerInputs (Tensor.zeros #[batch, 1, cfg.hiddenSize] false cacheDevice)
    let mut pos : UInt64 := 0
    while pos < seq do
      let xIn : Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device } := Tensor.sliceChecked talkerInputs 1 pos 1 #[batch, 1, cfg.hiddenSize]
      let cos : Tensor { shape := #[1, cfg.headDim / 2], dtype := .Float32, device := device } := Tensor.sliceChecked cosAll 0 pos 1 #[1, cfg.headDim / 2]
      let sin : Tensor { shape := #[1, cfg.headDim / 2], dtype := .Float32, device := device } := Tensor.sliceChecked sinAll 0 pos 1 #[1, cfg.headDim / 2]
      let mut h : Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device } := xIn
      for i in [:m.model.layers.size] do
        let layer : TalkerLayer (device := device) cfg ←
          match m.model.layers[i]? with
          | some layer => pure layer
          | none => throw <| IO.userError s!"missing talker layer at index {i}"
        let cache : qwen.QwenAttention.KVCache batch cfg.numKeyValueHeads cfg.headDim (.Unknown "dynamic") ←
          match caches[i]? with
          | some cache => pure cache
          | none => throw <| IO.userError s!"missing talker cache at index {i}"
        let (h', cache') := qwen.QwenLayer.forwardStep layer h cos sin cache
        h := h'
        caches := caches.set! i cache'
      lastHidden := m.model.norm.forward3d h
      pos := pos + 1

    let mut done : Array Bool := Array.replicate batch.toNat false
    let mut lengths : Array UInt64 := Array.replicate batch.toNat maxFrames
    let mut historyCols : Array (Tensor { shape := #[batch, 1], dtype := .Int64, device := device }) := #[]
    let defaultPadEmbed : Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device } :=
      Tensor.castLike talkerInputs (Tensor.zeros #[batch, 1, cfg.hiddenSize] false cacheDevice)
    let padEmbed : Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device } := ttsPadEmbed.getD defaultPadEmbed
    let cpMaxSeq : UInt64 := cfg.numCodeGroups + 1
    let (cpCosRaw, cpSinRaw) := Tensor.rotaryFreqs cpMaxSeq cfg.codePredictorConfig.headDim cfg.codePredictorConfig.ropeTheta
    let cpRoPE : CodePredictorRoPECache (device := device) cfg := {
      cos := cpCosRaw.to cacheDevice
      sin := cpSinRaw.to cacheDevice
    }

    for step in [:maxFrames.toNat] do
      let allDone := done.foldl (fun acc d => acc && d) true
      if allDone then
        break

      let historyOpt :=
        if historyCols.isEmpty then
          none
        else
          let histLen : UInt64 := historyCols.size.toUInt64
          let histDyn := Tensor.catManyChecked (historyCols.map Tensor.packShape) 1 #[batch, histLen]
          let hist : Tensor { shape := #[batch, histLen], dtype := .Int64, device := device } := Tensor.reshapeChecked histDyn #[batch, histLen]
          some ⟨histLen, hist⟩
      let allowEos := step.toUInt64 >= minNewTokens
      let frameOut ← generateFrameFromLastHidden cfg m lastHidden historyOpt
        allowEos temperature topK topP repetitionPenalty suppressTail
        subtalkerTemperature subtalkerTopK subtalkerTopP
        subtalkerTemperaturesByGroup subtalkerTopKsByGroup subtalkerTopPsByGroup
        (some cpRoPE)
      let frame := frameOut.codes
      onFrame step.toUInt64 frame

      let firstCode : Tensor { shape := #[batch, 1], dtype := .Int64, device := device } := Tensor.sliceChecked frame 1 0 1 #[batch, 1]
      historyCols := historyCols.push firstCode
      let firstCode1d : Tensor { shape := #[batch], dtype := .Int64, device := device } := Tensor.reshapeChecked firstCode #[batch]
      let firstCodes ← Tensor.toUInt64Array firstCode1d
      for i in [:batch.toNat] do
        let isDone := done.getD i false
        let tok := firstCodes.getD i cfg.codecEosTokenId
        if !isDone && allowEos && tok == cfg.codecEosTokenId then
          done := done.set! i true
          lengths := lengths.set! i step.toUInt64

      let trailingAdd : Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device } :=
        match trailingTextHidden with
        | some ⟨trailingSeq, trailing⟩ =>
            if step.toUInt64 < trailingSeq then
              Tensor.sliceChecked trailing 1 step.toUInt64 1 #[batch, 1, cfg.hiddenSize]
            else
              padEmbed
        | none => padEmbed
      let nextEmb : Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device } := Tensor.Promoting.add frameOut.summedEmbedding trailingAdd
      let nextPos : UInt64 := seq + step.toUInt64
      let cos : Tensor { shape := #[1, cfg.headDim / 2], dtype := .Float32, device := device } := Tensor.sliceChecked cosAll 0 nextPos 1 #[1, cfg.headDim / 2]
      let sin : Tensor { shape := #[1, cfg.headDim / 2], dtype := .Float32, device := device } := Tensor.sliceChecked sinAll 0 nextPos 1 #[1, cfg.headDim / 2]
      let mut h : Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device } := nextEmb
      for i in [:m.model.layers.size] do
        let layer : TalkerLayer (device := device) cfg ←
          match m.model.layers[i]? with
          | some layer => pure layer
          | none => throw <| IO.userError s!"missing talker layer at index {i}"
        let cache : qwen.QwenAttention.KVCache batch cfg.numKeyValueHeads cfg.headDim (.Unknown "dynamic") ←
          match caches[i]? with
          | some cache => pure cache
          | none => throw <| IO.userError s!"missing talker cache at index {i}"
        let (h', cache') := qwen.QwenLayer.forwardStep layer h cos sin cache
        h := h'
        caches := caches.set! i cache'
      lastHidden := m.model.norm.forward3d h

    pure lengths

end

/-- Autoregressively generate a fixed number of codec frames.
    Returns `[batch, maxFrames, numCodeGroups]`. -/
def generateCodes {batch seq : UInt64} (cfg : TalkerConfig)
    (m : TalkerForConditionalGeneration (device := device) cfg)
    (talkerInputs : Tensor { shape := #[batch, seq, cfg.hiddenSize], device := device })
    (maxFrames : UInt64 := 256)
    (minNewTokens : UInt64 := 2)
    (temperature : Float := 0.9)
    (topK : UInt64 := 50)
    (topP : Float := 1.0)
    (subtalkerTemperature : Float := 0.9)
    (subtalkerTopK : UInt64 := 50)
    (subtalkerTopP : Float := 1.0)
    (repetitionPenalty : Float := 1.05)
    (suppressTail : UInt64 := 1024)
    (trailingTextHidden : Option (Sigma fun trailingSeq => Tensor { shape := #[batch, trailingSeq, cfg.hiddenSize], device := device }) := none)
    (ttsPadEmbed : Option (Tensor { shape := #[batch, 1, cfg.hiddenSize], device := device }) := none)
    (subtalkerTemperaturesByGroup : Option (Array Float) := none)
    (subtalkerTopKsByGroup : Option (Array UInt64) := none)
    (subtalkerTopPsByGroup : Option (Array Float) := none)
    : IO (Tensor { shape := #[batch, maxFrames, cfg.numCodeGroups], dtype := .Int64, device := device }) := do
  let out ← generateCodesWithLengths cfg m talkerInputs maxFrames minNewTokens
    temperature topK topP
    subtalkerTemperature subtalkerTopK subtalkerTopP
    repetitionPenalty suppressTail
    trailingTextHidden ttsPadEmbed
    subtalkerTemperaturesByGroup subtalkerTopKsByGroup subtalkerTopPsByGroup
  pure out.codes

end TalkerForConditionalGeneration

end torch.qwen3tts
