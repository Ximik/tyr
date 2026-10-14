/-
  Tyr/Model/Whisper/Model.lean

  Native Whisper encoder-decoder model in Tyr.
-/
import Tyr.Model.Whisper.TypedOps
import Tyr.TensorStruct
import Tyr.Module.LayerNorm
import Tyr.Model.Whisper.Config
import Tyr.Model.Utils

namespace torch.whisper

open torch.Model

private def activate {s : Shape} (name : String) (x : Tensor { shape := s }) : Tensor { shape := s } :=
  if name == "relu" then
    Tensor.relu x
  else
    Tensor.Preserving.gelu x

structure WhisperAttention (dModel nHeads : UInt64) where
  qProjWeight : Tensor { shape := #[dModel, dModel] }
  qProjBias : Tensor { shape := #[dModel] }
  kProjWeight : Tensor { shape := #[dModel, dModel] }
  kProjBias : Tensor { shape := #[dModel] }
  vProjWeight : Tensor { shape := #[dModel, dModel] }
  vProjBias : Tensor { shape := #[dModel] }
  outProjWeight : Tensor { shape := #[dModel, dModel] }
  outProjBias : Tensor { shape := #[dModel] }
  deriving TensorStruct

namespace WhisperAttention

def init (dModel nHeads : UInt64) : IO (WhisperAttention dModel nHeads) := do
  if nHeads == 0 || dModel % nHeads != 0 then
    throw <| IO.userError s!"WhisperAttention requires d_model divisible by n_heads, got {dModel} and {nHeads}"
  let qProjWeight ← initWeightDynamic #[dModel, dModel] dModel
  let qProjBias ← initBiasDynamic #[dModel]
  let kProjWeight ← initWeightDynamic #[dModel, dModel] dModel
  let kProjBias ← initBiasDynamic #[dModel]
  let vProjWeight ← initWeightDynamic #[dModel, dModel] dModel
  let vProjBias ← initBiasDynamic #[dModel]
  let outProjWeight ← initWeightDynamic #[dModel, dModel] dModel
  let outProjBias ← initBiasDynamic #[dModel]
  pure { qProjWeight, qProjBias, kProjWeight, kProjBias, vProjWeight, vProjBias, outProjWeight, outProjBias }

/-- Incremental self-attention KV cache for one decoder layer. -/
structure KVCache (batch nHeads headDim : UInt64) where
  kStoreDyn : ShapePackedTensor (.Unknown "dynamic")
  vStoreDyn : ShapePackedTensor (.Unknown "dynamic")
  seq : UInt64 := 0
  maxLen : UInt64 := 0

/-- Initialize an empty static-capacity KV cache on `device`. -/
def initKVCache {batch nHeads headDim : UInt64}
    (maxLen : UInt64)
    (device : Device := Device.CPU) : KVCache batch nHeads headDim :=
  let k0 : Tensor { shape := #[batch, nHeads, maxLen, headDim] } :=
    zerosDynamic #[batch, nHeads, maxLen, headDim] false device
  let v0 : Tensor { shape := #[batch, nHeads, maxLen, headDim] } :=
    zerosDynamic #[batch, nHeads, maxLen, headDim] false device
  { kStoreDyn := Tensor.packShape k0, vStoreDyn := Tensor.packShape v0, seq := 0, maxLen := maxLen }

/-- Project external key/value states once into attention layout.
    Output layout is `[batch, nHeads, kvSeq, headDim]`. -/
def projectKV {batch kvSeq : UInt64}
    (m : WhisperAttention dModel nHeads)
    (keyValueStates : Tensor { shape := #[batch, kvSeq, dModel] })
    : Tensor { shape := #[batch, nHeads, kvSeq, dModel / nHeads] } × Tensor { shape := #[batch, nHeads, kvSeq, dModel / nHeads] } :=
  let headDim : UInt64 := dModel / nHeads
  let k : Tensor { shape := #[batch, kvSeq, dModel] } := Tensor.affine3d keyValueStates m.kProjWeight m.kProjBias
  let v : Tensor { shape := #[batch, kvSeq, dModel] } := Tensor.affine3d keyValueStates m.vProjWeight m.vProjBias
  let k : Tensor { shape := #[batch, kvSeq, nHeads, headDim] } := Tensor.reshapeChecked k #[batch, kvSeq, nHeads, headDim]
  let v : Tensor { shape := #[batch, kvSeq, nHeads, headDim] } := Tensor.reshapeChecked v #[batch, kvSeq, nHeads, headDim]
  (transposeAttention k, transposeAttention v)

/-- Run attention where K/V are already projected and transposed.
    `keyStates` / `valueStates` layout: `[batch, nHeads, kvSeq, headDim]`. -/
def forwardWithProjectedKV {batch qSeq kvSeq : UInt64}
    (m : WhisperAttention dModel nHeads)
    (queryStates : Tensor { shape := #[batch, qSeq, dModel] })
    (keyStates : Tensor { shape := #[batch, nHeads, kvSeq, dModel / nHeads] })
    (valueStates : Tensor { shape := #[batch, nHeads, kvSeq, dModel / nHeads] })
    (isCausal : Bool := false)
    : Tensor { shape := #[batch, qSeq, dModel] } :=
  let headDim : UInt64 := dModel / nHeads
  let q : Tensor { shape := #[batch, qSeq, dModel] } := Tensor.affine3d queryStates m.qProjWeight m.qProjBias
  let q : Tensor { shape := #[batch, qSeq, nHeads, headDim] } := Tensor.reshapeChecked q #[batch, qSeq, nHeads, headDim]
  let qh : Tensor { shape := #[batch, nHeads, qSeq, headDim] } := transposeAttention q
  let attn : Tensor { shape := #[batch, nHeads, qSeq, headDim] } :=
    Tensor.attentionGQAQKV
      (kvHeads := nHeads)
      qh
      keyStates
      valueStates
      0.0
      isCausal
      true
  let out : Tensor { shape := #[batch, qSeq, nHeads, headDim] } := fromAttention attn
  let out : Tensor { shape := #[batch, qSeq, dModel] } := Tensor.reshapeChecked out #[batch, qSeq, dModel]
  Tensor.affine3d out m.outProjWeight m.outProjBias

def forwardCross {batch qSeq kvSeq : UInt64}
    (m : WhisperAttention dModel nHeads)
    (queryStates : Tensor { shape := #[batch, qSeq, dModel] })
    (keyValueStates : Tensor { shape := #[batch, kvSeq, dModel] })
    (isCausal : Bool := false)
    : Tensor { shape := #[batch, qSeq, dModel] } :=
  let (kh, vh) := projectKV m keyValueStates
  forwardWithProjectedKV m queryStates kh vh isCausal

def forwardSelf {batch seq : UInt64}
    (m : WhisperAttention dModel nHeads)
    (x : Tensor { shape := #[batch, seq, dModel] })
    (isCausal : Bool := false)
    : Tensor { shape := #[batch, seq, dModel] } :=
  forwardCross m x x isCausal

/-- Incremental one-token self-attention step with KV cache update. -/
def forwardSelfStep {batch : UInt64}
    (m : WhisperAttention dModel nHeads)
    (x : Tensor { shape := #[batch, 1, dModel] })
    (cache : KVCache batch nHeads (dModel / nHeads))
    : Tensor { shape := #[batch, 1, dModel] } × KVCache batch nHeads (dModel / nHeads) :=
  let headDim : UInt64 := dModel / nHeads
  let q0 : Tensor { shape := #[batch, 1, dModel] } := Tensor.affine3d x m.qProjWeight m.qProjBias
  let k0 : Tensor { shape := #[batch, 1, dModel] } := Tensor.affine3d x m.kProjWeight m.kProjBias
  let v0 : Tensor { shape := #[batch, 1, dModel] } := Tensor.affine3d x m.vProjWeight m.vProjBias

  let q : Tensor { shape := #[batch, 1, nHeads, headDim] } := Tensor.reshapeChecked q0 #[batch, 1, nHeads, headDim]
  let k : Tensor { shape := #[batch, 1, nHeads, headDim] } := Tensor.reshapeChecked k0 #[batch, 1, nHeads, headDim]
  let v : Tensor { shape := #[batch, 1, nHeads, headDim] } := Tensor.reshapeChecked v0 #[batch, 1, nHeads, headDim]

  let qh : Tensor { shape := #[batch, nHeads, 1, headDim] } := transposeAttention q
  let kNew : Tensor { shape := #[batch, nHeads, 1, headDim] } := transposeAttention k
  let vNew : Tensor { shape := #[batch, nHeads, 1, headDim] } := transposeAttention v

  let kStore : Tensor { shape := #[batch, nHeads, cache.maxLen, headDim] } :=
    Tensor.castLike kNew (Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, nHeads, cache.maxLen, headDim])
  let vStore : Tensor { shape := #[batch, nHeads, cache.maxLen, headDim] } :=
    Tensor.castLike vNew (Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, nHeads, cache.maxLen, headDim])

  if cache.seq < cache.maxLen then
    let kStore' : Tensor { shape := #[batch, nHeads, cache.maxLen, headDim] } :=
      Tensor.sliceScatter kStore 2 cache.seq kNew
    let vStore' : Tensor { shape := #[batch, nHeads, cache.maxLen, headDim] } :=
      Tensor.sliceScatter vStore 2 cache.seq vNew
    let kvLen : UInt64 := cache.seq + 1
    let kAll : Tensor { shape := #[batch, nHeads, kvLen, headDim] } := sliceTyped kStore' 2 0 kvLen
    let vAll : Tensor { shape := #[batch, nHeads, kvLen, headDim] } := sliceTyped vStore' 2 0 kvLen
    let attn : Tensor { shape := #[batch, nHeads, 1, headDim] } :=
      Tensor.attentionGQAQKV
        (kvHeads := nHeads)
        qh
        kAll
        vAll
        0.0
        false
        true
    let out : Tensor { shape := #[batch, 1, nHeads, headDim] } := fromAttention attn
    let out : Tensor { shape := #[batch, 1, dModel] } := Tensor.reshapeChecked out #[batch, 1, dModel]
    let out : Tensor { shape := #[batch, 1, dModel] } := Tensor.affine3d out m.outProjWeight m.outProjBias
    let cache' : KVCache batch nHeads headDim := {
      kStoreDyn := Tensor.packShape kStore'
      vStoreDyn := Tensor.packShape vStore'
      seq := kvLen
      maxLen := cache.maxLen
    }
    (out, cache')
  else
    let writePos : UInt64 := if cache.maxLen == 0 then 0 else cache.maxLen - 1
    let kStore' : Tensor { shape := #[batch, nHeads, cache.maxLen, headDim] } :=
      Tensor.sliceScatter kStore 2 writePos kNew
    let vStore' : Tensor { shape := #[batch, nHeads, cache.maxLen, headDim] } :=
      Tensor.sliceScatter vStore 2 writePos vNew
    let kvLen : UInt64 := cache.maxLen
    let kAll : Tensor { shape := #[batch, nHeads, kvLen, headDim] } := sliceTyped kStore' 2 0 kvLen
    let vAll : Tensor { shape := #[batch, nHeads, kvLen, headDim] } := sliceTyped vStore' 2 0 kvLen
    let attn : Tensor { shape := #[batch, nHeads, 1, headDim] } :=
      Tensor.attentionGQAQKV
        (kvHeads := nHeads)
        qh
        kAll
        vAll
        0.0
        false
        true
    let out : Tensor { shape := #[batch, 1, nHeads, headDim] } := fromAttention attn
    let out : Tensor { shape := #[batch, 1, dModel] } := Tensor.reshapeChecked out #[batch, 1, dModel]
    let out : Tensor { shape := #[batch, 1, dModel] } := Tensor.affine3d out m.outProjWeight m.outProjBias
    let cache' : KVCache batch nHeads headDim := {
      kStoreDyn := Tensor.packShape kStore'
      vStoreDyn := Tensor.packShape vStore'
      seq := cache.maxLen
      maxLen := cache.maxLen
    }
    (out, cache')

end WhisperAttention

structure WhisperEncoderLayer (cfg : WhisperConfig) where
  selfAttn : WhisperAttention cfg.dModel cfg.encoderAttentionHeads
  selfAttnLayerNorm : LayerNorm cfg.dModel (.Unknown "dynamic")
  fc1Weight : Tensor { shape := #[cfg.encoderFfnDim, cfg.dModel] }
  fc1Bias : Tensor { shape := #[cfg.encoderFfnDim] }
  fc2Weight : Tensor { shape := #[cfg.dModel, cfg.encoderFfnDim] }
  fc2Bias : Tensor { shape := #[cfg.dModel] }
  finalLayerNorm : LayerNorm cfg.dModel (.Unknown "dynamic")
  deriving TensorStruct

namespace WhisperEncoderLayer

def init (cfg : WhisperConfig) : IO (WhisperEncoderLayer cfg) := do
  let selfAttn ← WhisperAttention.init cfg.dModel cfg.encoderAttentionHeads
  let selfAttnLayerNorm ← initLayerNorm cfg.dModel cfg.layerNormEps
  let fc1Weight ← initWeightDynamic #[cfg.encoderFfnDim, cfg.dModel] cfg.dModel
  let fc1Bias ← initBiasDynamic #[cfg.encoderFfnDim]
  let fc2Weight ← initWeightDynamic #[cfg.dModel, cfg.encoderFfnDim] cfg.encoderFfnDim
  let fc2Bias ← initBiasDynamic #[cfg.dModel]
  let finalLayerNorm ← initLayerNorm cfg.dModel cfg.layerNormEps
  pure { selfAttn, selfAttnLayerNorm, fc1Weight, fc1Bias, fc2Weight, fc2Bias, finalLayerNorm }

def forward {batch seq : UInt64}
    (m : WhisperEncoderLayer cfg)
    (x : Tensor { shape := #[batch, seq, cfg.dModel] })
    : Tensor { shape := #[batch, seq, cfg.dModel] } :=
  let h0 := m.selfAttnLayerNorm.forward3d x
  let h1 := m.selfAttn.forwardSelf h0 (isCausal := false)
  let x1 := x + h1
  let h2 := m.finalLayerNorm.forward3d x1
  let h3 : Tensor { shape := #[batch, seq, cfg.encoderFfnDim] } := Tensor.affine3d h2 m.fc1Weight m.fc1Bias
  let h4 := activate cfg.activationFunction h3
  let h5 : Tensor { shape := #[batch, seq, cfg.dModel] } := Tensor.affine3d h4 m.fc2Weight m.fc2Bias
  x1 + h5

end WhisperEncoderLayer

structure WhisperDecoderLayer (cfg : WhisperConfig) where
  selfAttn : WhisperAttention cfg.dModel cfg.decoderAttentionHeads
  selfAttnLayerNorm : LayerNorm cfg.dModel (.Unknown "dynamic")
  encoderAttn : WhisperAttention cfg.dModel cfg.decoderAttentionHeads
  encoderAttnLayerNorm : LayerNorm cfg.dModel (.Unknown "dynamic")
  fc1Weight : Tensor { shape := #[cfg.decoderFfnDim, cfg.dModel] }
  fc1Bias : Tensor { shape := #[cfg.decoderFfnDim] }
  fc2Weight : Tensor { shape := #[cfg.dModel, cfg.decoderFfnDim] }
  fc2Bias : Tensor { shape := #[cfg.dModel] }
  finalLayerNorm : LayerNorm cfg.dModel (.Unknown "dynamic")
  deriving TensorStruct

namespace WhisperDecoderLayer

def init (cfg : WhisperConfig) : IO (WhisperDecoderLayer cfg) := do
  let selfAttn ← WhisperAttention.init cfg.dModel cfg.decoderAttentionHeads
  let selfAttnLayerNorm ← initLayerNorm cfg.dModel cfg.layerNormEps
  let encoderAttn ← WhisperAttention.init cfg.dModel cfg.decoderAttentionHeads
  let encoderAttnLayerNorm ← initLayerNorm cfg.dModel cfg.layerNormEps
  let fc1Weight ← initWeightDynamic #[cfg.decoderFfnDim, cfg.dModel] cfg.dModel
  let fc1Bias ← initBiasDynamic #[cfg.decoderFfnDim]
  let fc2Weight ← initWeightDynamic #[cfg.dModel, cfg.decoderFfnDim] cfg.decoderFfnDim
  let fc2Bias ← initBiasDynamic #[cfg.dModel]
  let finalLayerNorm ← initLayerNorm cfg.dModel cfg.layerNormEps
  pure {
    selfAttn
    selfAttnLayerNorm
    encoderAttn
    encoderAttnLayerNorm
    fc1Weight
    fc1Bias
    fc2Weight
    fc2Bias
    finalLayerNorm
  }

def forward {batch seq encSeq : UInt64}
    (m : WhisperDecoderLayer cfg)
    (x : Tensor { shape := #[batch, seq, cfg.dModel] })
    (encoderHidden : Tensor { shape := #[batch, encSeq, cfg.dModel] })
    : Tensor { shape := #[batch, seq, cfg.dModel] } :=
  let h0 := m.selfAttnLayerNorm.forward3d x
  let h1 := m.selfAttn.forwardSelf h0 (isCausal := true)
  let x1 := x + h1
  let h2 := m.encoderAttnLayerNorm.forward3d x1
  let h3 := m.encoderAttn.forwardCross h2 encoderHidden (isCausal := false)
  let x2 := x1 + h3
  let h4 := m.finalLayerNorm.forward3d x2
  let h5 : Tensor { shape := #[batch, seq, cfg.decoderFfnDim] } := Tensor.affine3d h4 m.fc1Weight m.fc1Bias
  let h6 := activate cfg.activationFunction h5
  let h7 : Tensor { shape := #[batch, seq, cfg.dModel] } := Tensor.affine3d h6 m.fc2Weight m.fc2Bias
  x2 + h7

/-- One-token decoder layer step with incremental self KV cache and precomputed cross KV. -/
def forwardStepCached {batch encSeq : UInt64}
    (m : WhisperDecoderLayer cfg)
    (x : Tensor { shape := #[batch, 1, cfg.dModel] })
    (crossK : Tensor { shape := #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads] })
    (crossV : Tensor { shape := #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads] })
    (selfCache : WhisperAttention.KVCache batch cfg.decoderAttentionHeads (cfg.dModel / cfg.decoderAttentionHeads))
    : Tensor { shape := #[batch, 1, cfg.dModel] } × WhisperAttention.KVCache batch cfg.decoderAttentionHeads (cfg.dModel / cfg.decoderAttentionHeads) :=
  let h0 := m.selfAttnLayerNorm.forward3d x
  let (h1a, selfCache') := m.selfAttn.forwardSelfStep h0 selfCache
  let x1 := x + h1a
  let h2 := m.encoderAttnLayerNorm.forward3d x1
  let h3 := m.encoderAttn.forwardWithProjectedKV h2 crossK crossV (isCausal := false)
  let x2 := x1 + h3
  let h4 := m.finalLayerNorm.forward3d x2
  let h5 : Tensor { shape := #[batch, 1, cfg.decoderFfnDim] } := Tensor.affine3d h4 m.fc1Weight m.fc1Bias
  let h6 := activate cfg.activationFunction h5
  let h7 : Tensor { shape := #[batch, 1, cfg.dModel] } := Tensor.affine3d h6 m.fc2Weight m.fc2Bias
  (x2 + h7, selfCache')

end WhisperDecoderLayer

structure WhisperModel (cfg : WhisperConfig) where
  encoderConv1Weight : Tensor { shape := #[cfg.dModel, cfg.numMelBins, 3] }
  encoderConv1Bias : Tensor { shape := #[cfg.dModel] }
  encoderConv2Weight : Tensor { shape := #[cfg.dModel, cfg.dModel, 3] }
  encoderConv2Bias : Tensor { shape := #[cfg.dModel] }
  encoderPositionalEmbedding : Tensor { shape := #[cfg.maxSourcePositions, cfg.dModel] }
  encoderLayers : Array (WhisperEncoderLayer cfg)
  encoderLayerNorm : LayerNorm cfg.dModel (.Unknown "dynamic")

  decoderTokenEmbedding : Tensor { shape := #[cfg.vocabSize, cfg.dModel] }
  decoderPositionalEmbedding : Tensor { shape := #[cfg.maxTargetPositions, cfg.dModel] }
  decoderLayers : Array (WhisperDecoderLayer cfg)
  decoderLayerNorm : LayerNorm cfg.dModel (.Unknown "dynamic")
  deriving TensorStruct

namespace WhisperModel

def init (cfg : WhisperConfig) : IO (WhisperModel cfg) := do
  if !cfg.hasValidHeads then
    throw <| IO.userError
      s!"Invalid Whisper config: d_model={cfg.dModel}, encoder_heads={cfg.encoderAttentionHeads}, decoder_heads={cfg.decoderAttentionHeads}"
  let encoderConv1Weight ← initWeightDynamic #[cfg.dModel, cfg.numMelBins, 3] cfg.numMelBins
  let encoderConv1Bias ← initBiasDynamic #[cfg.dModel]
  let encoderConv2Weight ← initWeightDynamic #[cfg.dModel, cfg.dModel, 3] cfg.dModel
  let encoderConv2Bias ← initBiasDynamic #[cfg.dModel]
  let encoderPositionalEmbedding ← initWeightDynamic #[cfg.maxSourcePositions, cfg.dModel] cfg.dModel

  let mut encoderLayers : Array (WhisperEncoderLayer cfg) := #[]
  for _ in [:cfg.encoderLayers.toNat] do
    encoderLayers := encoderLayers.push (← WhisperEncoderLayer.init cfg)
  let encoderLayerNorm ← initLayerNorm cfg.dModel cfg.layerNormEps

  let decoderTokenEmbedding ← initWeightDynamic #[cfg.vocabSize, cfg.dModel] cfg.dModel
  let decoderPositionalEmbedding ← initWeightDynamic #[cfg.maxTargetPositions, cfg.dModel] cfg.dModel
  let mut decoderLayers : Array (WhisperDecoderLayer cfg) := #[]
  for _ in [:cfg.decoderLayers.toNat] do
    decoderLayers := decoderLayers.push (← WhisperDecoderLayer.init cfg)
  let decoderLayerNorm ← initLayerNorm cfg.dModel cfg.layerNormEps

  pure {
    encoderConv1Weight
    encoderConv1Bias
    encoderConv2Weight
    encoderConv2Bias
    encoderPositionalEmbedding
    encoderLayers
    encoderLayerNorm
    decoderTokenEmbedding
    decoderPositionalEmbedding
    decoderLayers
    decoderLayerNorm
  }

def encode {batch frames : UInt64}
    (m : WhisperModel cfg)
    (inputFeatures : Tensor { shape := #[batch, cfg.numMelBins, frames] })
    : IO (Tensor { shape := #[batch, WhisperConfig.conv2OutputSeq frames, cfg.dModel] }) := do
  if frames == 0 then
    throw <| IO.userError "Whisper encoder requires at least one input frame"
  let inputFeatures := Tensor.castLike m.encoderConv1Weight inputFeatures
  let x1 : Tensor { shape := #[batch, cfg.dModel, frames] } :=
    Tensor.reshapeChecked
      (Tensor.Preserving.gelu (Tensor.conv1dGroups inputFeatures m.encoderConv1Weight m.encoderConv1Bias 1 1 1 1))
      #[batch, cfg.dModel, frames]
  let outSeq := WhisperConfig.conv2OutputSeq frames
  if outSeq > cfg.maxSourcePositions then
    throw <| IO.userError
      s!"Whisper encoder sequence {outSeq} exceeds max_source_positions {cfg.maxSourcePositions}"
  let x2 : Tensor { shape := #[batch, cfg.dModel, outSeq] } :=
    Tensor.reshapeChecked
      (Tensor.Preserving.gelu (Tensor.conv1dGroups x1 m.encoderConv2Weight m.encoderConv2Bias 2 1 1 1))
      #[batch, cfg.dModel, outSeq]
  let x2t : Tensor { shape := #[batch, outSeq, cfg.dModel] } := (Tensor.permute x2 #[0, 2, 1]).reshapeChecked #[batch, outSeq, cfg.dModel]
  let pos : Tensor { shape := #[outSeq, cfg.dModel] } := sliceTyped m.encoderPositionalEmbedding 0 0 outSeq
  let posB : Tensor { shape := #[batch, outSeq, cfg.dModel] } :=
    Tensor.expand (Tensor.reshapeChecked pos #[1, outSeq, cfg.dModel]) #[batch, outSeq, cfg.dModel]
  let mut h : Tensor { shape := #[batch, outSeq, cfg.dModel] } := x2t + posB
  for layer in m.encoderLayers do
    h := layer.forward h
  pure (m.encoderLayerNorm.forward3d h)

def decode {batch seq encSeq : UInt64}
    (m : WhisperModel cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (encoderHidden : Tensor { shape := #[batch, encSeq, cfg.dModel] })
    : IO (Tensor { shape := #[batch, seq, cfg.dModel] }) := do
  if seq > cfg.maxTargetPositions then
    throw <| IO.userError
      s!"Whisper decoder sequence {seq} exceeds max_target_positions {cfg.maxTargetPositions}"
  let tokEmb : Tensor { shape := #[batch, seq, cfg.dModel] } := Tensor.embedding inputIds m.decoderTokenEmbedding
  let pos : Tensor { shape := #[seq, cfg.dModel] } := sliceTyped m.decoderPositionalEmbedding 0 0 seq
  let posB : Tensor { shape := #[batch, seq, cfg.dModel] } :=
    Tensor.expand (Tensor.reshapeChecked pos #[1, seq, cfg.dModel]) #[batch, seq, cfg.dModel]
  let mut h : Tensor { shape := #[batch, seq, cfg.dModel] } := tokEmb + posB
  for layer in m.decoderLayers do
    h := layer.forward h encoderHidden
  pure (m.decoderLayerNorm.forward3d h)

end WhisperModel

structure WhisperForConditionalGeneration (cfg : WhisperConfig) where
  model : WhisperModel cfg
  projOut : Tensor { shape := #[cfg.vocabSize, cfg.dModel] }
  deriving TensorStruct

namespace WhisperForConditionalGeneration

/-- Per-layer precomputed cross-attention K/V for one encoded audio chunk. -/
structure LayerCrossCache (cfg : WhisperConfig) where
  kDyn : ShapePackedTensor (.Unknown "dynamic")
  vDyn : ShapePackedTensor (.Unknown "dynamic")

/-- Per-layer self-attention KV cache for decoder incremental decode. -/
abbrev LayerKVCache (cfg : WhisperConfig) (batch : UInt64) :=
  WhisperAttention.KVCache batch cfg.decoderAttentionHeads (cfg.dModel / cfg.decoderAttentionHeads)

def init (cfg : WhisperConfig) : IO (WhisperForConditionalGeneration cfg) := do
  let model ← WhisperModel.init cfg
  let projOut ← initWeightDynamic #[cfg.vocabSize, cfg.dModel] cfg.dModel
  pure { model, projOut }

def encode {batch frames : UInt64}
    (m : WhisperForConditionalGeneration cfg)
    (inputFeatures : Tensor { shape := #[batch, cfg.numMelBins, frames] })
    : IO (Tensor { shape := #[batch, WhisperConfig.conv2OutputSeq frames, cfg.dModel] }) :=
  m.model.encode inputFeatures

def decode {batch seq encSeq : UInt64}
    (m : WhisperForConditionalGeneration cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (encoderHidden : Tensor { shape := #[batch, encSeq, cfg.dModel] })
    : IO (Tensor { shape := #[batch, seq, cfg.vocabSize] }) := do
  let hidden ← m.model.decode inputIds encoderHidden
  pure (Tensor.linear3d hidden m.projOut)

/-- Initialize one self-attention KV cache per decoder layer. -/
def initLayerKVCaches {batch : UInt64}
    (m : WhisperForConditionalGeneration cfg)
    (maxLen : UInt64)
    (device : Device)
    : Array (LayerKVCache cfg batch) :=
  m.model.decoderLayers.map (fun _ =>
    WhisperAttention.initKVCache
      maxLen
      (batch := batch)
      (nHeads := cfg.decoderAttentionHeads)
      (headDim := cfg.dModel / cfg.decoderAttentionHeads)
      device)

/-- Precompute cross-attention K/V once per decoder layer for an encoded chunk. -/
def precomputeCrossCaches {batch encSeq : UInt64}
    (m : WhisperForConditionalGeneration cfg)
    (encoderHidden : Tensor { shape := #[batch, encSeq, cfg.dModel] })
    : IO (Array (LayerCrossCache cfg)) := do
  let mut out : Array (LayerCrossCache cfg) := #[]
  for layer in m.model.decoderLayers do
    let (k, v) := layer.encoderAttn.projectKV encoderHidden
    out := out.push { kDyn := Tensor.packShape k, vDyn := Tensor.packShape v }
  pure out

/-- Decode one token step using static self KV caches and precomputed cross K/V. -/
def decodeStepWithCache {batch encSeq : UInt64}
    (m : WhisperForConditionalGeneration cfg)
    (tokenIds : Tensor { shape := #[batch, 1], dtype := .Int64 })
    (position : UInt64)
    (crossCaches : Array (LayerCrossCache cfg))
    (selfCaches : Array (LayerKVCache cfg batch))
    : IO (Tensor { shape := #[batch, cfg.vocabSize] } × Array (LayerKVCache cfg batch)) := do
  if position >= cfg.maxTargetPositions then
    throw <| IO.userError
      s!"Whisper decoder position {position} exceeds max_target_positions {cfg.maxTargetPositions}"
  let tokEmb : Tensor { shape := #[batch, 1, cfg.dModel] } := Tensor.embedding tokenIds m.model.decoderTokenEmbedding
  let pos : Tensor { shape := #[1, cfg.dModel] } := sliceTyped m.model.decoderPositionalEmbedding 0 position 1
  let posB : Tensor { shape := #[batch, 1, cfg.dModel] } :=
    Tensor.expand (Tensor.reshapeChecked pos #[1, 1, cfg.dModel]) #[batch, 1, cfg.dModel]
  let mut h : Tensor { shape := #[batch, 1, cfg.dModel] } := tokEmb + posB
  let mut nextCaches := selfCaches
  for i in [:m.model.decoderLayers.size] do
    let layer ←
      match m.model.decoderLayers[i]? with
      | some l => pure l
      | none => throw <| IO.userError s!"missing Whisper decoder layer at index {i}"
    let selfCache ←
      match nextCaches[i]? with
      | some c => pure c
      | none => throw <| IO.userError s!"missing Whisper self KV cache at index {i}"
    if selfCache.seq != position then
      throw <| IO.userError s!"Whisper cache position {selfCache.seq} does not match decode position {position}"
    if selfCache.seq >= selfCache.maxLen then
      throw <| IO.userError s!"Whisper self KV cache capacity {selfCache.maxLen} exhausted"
    let crossCache ←
      match crossCaches[i]? with
      | some c => pure c
      | none => throw <| IO.userError s!"missing Whisper cross KV cache at index {i}"
    let crossK : Tensor { shape := #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads] } :=
      Tensor.reshapeChecked crossCache.kDyn.2 #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads]
    let crossV : Tensor { shape := #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads] } :=
      Tensor.reshapeChecked crossCache.vDyn.2 #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads]
    let (hNext, cacheNext) := layer.forwardStepCached h crossK crossV selfCache
    h := hNext
    nextCaches := nextCaches.set! i cacheNext
  let hiddenNorm : Tensor { shape := #[batch, 1, cfg.dModel] } := m.model.decoderLayerNorm.forward3d h
  let logits3 : Tensor { shape := #[batch, 1, cfg.vocabSize] } := Tensor.linear3d hiddenNorm m.projOut
  let logits2 : Tensor { shape := #[batch, cfg.vocabSize] } := Tensor.reshapeChecked logits3 #[batch, cfg.vocabSize]
  pure (logits2, nextCaches)

end WhisperForConditionalGeneration

end torch.whisper
