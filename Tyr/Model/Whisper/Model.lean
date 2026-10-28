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

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
open torch.Model

/-- Proves a head split or merge from the `nHeads ∣ dModel` hypothesis in scope. -/
local macro "head_split" : tactic =>
  `(tactic| simp +zetaDelta [Nat.mul_assoc, Nat.mul_div_cancel' ‹_ ∣ _›])

private def activate {s : Shape} (name : String) (x : Tensor { dtype := dtype, device := device, shape := s }) : Tensor { dtype := dtype, device := device, shape := s } :=
  if name == "relu" then
    Tensor.relu x
  else
    Tensor.gelu x

structure WhisperAttention (dModel nHeads : Nat) (dtype : DType := .Float32) where
  qProjWeight : Tensor { dtype := dtype, device := device, shape := #[dModel, dModel] }
  qProjBias : Tensor { dtype := dtype, device := device, shape := #[dModel] }
  kProjWeight : Tensor { dtype := dtype, device := device, shape := #[dModel, dModel] }
  kProjBias : Tensor { dtype := dtype, device := device, shape := #[dModel] }
  vProjWeight : Tensor { dtype := dtype, device := device, shape := #[dModel, dModel] }
  vProjBias : Tensor { dtype := dtype, device := device, shape := #[dModel] }
  outProjWeight : Tensor { dtype := dtype, device := device, shape := #[dModel, dModel] }
  outProjBias : Tensor { dtype := dtype, device := device, shape := #[dModel] }
  deriving TensorStruct

namespace WhisperAttention

def init (dModel nHeads : Nat) (device : Device := .CPU) (dtype : DType := .Float32) : IO (WhisperAttention (device := device) (dtype := dtype) dModel nHeads) := do
  if nHeads == 0 || dModel % nHeads != 0 then
    throw <| IO.userError s!"WhisperAttention requires d_model divisible by n_heads, got {dModel} and {nHeads}"
  let qProjWeight ← initWeightFor (device := device) (dtype := dtype) #[dModel, dModel] dModel
  let qProjBias ← initBiasFor (device := device) (dtype := dtype) #[dModel]
  let kProjWeight ← initWeightFor (device := device) (dtype := dtype) #[dModel, dModel] dModel
  let kProjBias ← initBiasFor (device := device) (dtype := dtype) #[dModel]
  let vProjWeight ← initWeightFor (device := device) (dtype := dtype) #[dModel, dModel] dModel
  let vProjBias ← initBiasFor (device := device) (dtype := dtype) #[dModel]
  let outProjWeight ← initWeightFor (device := device) (dtype := dtype) #[dModel, dModel] dModel
  let outProjBias ← initBiasFor (device := device) (dtype := dtype) #[dModel]
  pure { qProjWeight, qProjBias, kProjWeight, kProjBias, vProjWeight, vProjBias, outProjWeight, outProjBias }

/-- Incremental self-attention KV cache for one decoder layer. -/
structure KVCache (batch nHeads headDim : Nat) (dtype : DType := .Float32) where
  maxLen : Nat := 0
  kStore : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, maxLen, headDim] }
  vStore : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, maxLen, headDim] }
  seq : Nat := 0

/-- Initialize an empty static-capacity KV cache on `device`. -/
def initKVCache {batch nHeads headDim : Nat}
    (maxLen : Nat)
    (device : Device := Device.CPU) (dtype : DType := .Float32)
    [DType.ConstantSupported dtype device] : KVCache (device := device) (dtype := dtype) batch nHeads headDim :=
  let k0 : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, maxLen, headDim] } :=
    zerosFor (dtype := dtype) #[batch, nHeads, maxLen, headDim] false device
  let v0 : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, maxLen, headDim] } :=
    zerosFor (dtype := dtype) #[batch, nHeads, maxLen, headDim] false device
  { maxLen := maxLen, kStore := k0, vStore := v0, seq := 0 }

/-- Project external key/value states once into attention layout.
    Output layout is `[batch, nHeads, kvSeq, headDim]`. -/
def projectKV {batch kvSeq : Nat}
    (m : WhisperAttention (device := device) (dtype := dtype) dModel nHeads) (hdiv : nHeads ∣ dModel)
    (keyValueStates : Tensor { dtype := dtype, device := device, shape := #[batch, kvSeq, dModel] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, kvSeq, dModel / nHeads] } × Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, kvSeq, dModel / nHeads] } :=
  let headDim : Nat := dModel / nHeads
  let k : Tensor { dtype := dtype, device := device, shape := #[batch, kvSeq, dModel] } := Tensor.affine3d keyValueStates m.kProjWeight m.kProjBias
  let v : Tensor { dtype := dtype, device := device, shape := #[batch, kvSeq, dModel] } := Tensor.affine3d keyValueStates m.vProjWeight m.vProjBias
  let k : Tensor { dtype := dtype, device := device, shape := #[batch, kvSeq, nHeads, headDim] } := Tensor.reshape k #[batch, kvSeq, nHeads, headDim] (by head_split)
  let v : Tensor { dtype := dtype, device := device, shape := #[batch, kvSeq, nHeads, headDim] } := Tensor.reshape v #[batch, kvSeq, nHeads, headDim] (by head_split)
  (transposeAttention k, transposeAttention v)

/-- Run attention where K/V are already projected and transposed.
    `keyStates` / `valueStates` layout: `[batch, nHeads, kvSeq, headDim]`. -/
def forwardWithProjectedKV {batch qSeq kvSeq : Nat}
    (m : WhisperAttention (device := device) (dtype := dtype) dModel nHeads) (hdiv : nHeads ∣ dModel)
    (queryStates : Tensor { dtype := dtype, device := device, shape := #[batch, qSeq, dModel] })
    (keyStates : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, kvSeq, dModel / nHeads] })
    (valueStates : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, kvSeq, dModel / nHeads] })
    (isCausal : Bool := false)
    : Tensor { dtype := dtype, device := device, shape := #[batch, qSeq, dModel] } :=
  let headDim : Nat := dModel / nHeads
  let q : Tensor { dtype := dtype, device := device, shape := #[batch, qSeq, dModel] } := Tensor.affine3d queryStates m.qProjWeight m.qProjBias
  let q : Tensor { dtype := dtype, device := device, shape := #[batch, qSeq, nHeads, headDim] } := Tensor.reshape q #[batch, qSeq, nHeads, headDim] (by head_split)
  let qh : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, qSeq, headDim] } := transposeAttention q
  let attn : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, qSeq, headDim] } :=
    Tensor.attentionGQAQKV
      (kvHeads := nHeads)
      qh
      keyStates
      valueStates
      isCausal
      true
  let out : Tensor { dtype := dtype, device := device, shape := #[batch, qSeq, nHeads, headDim] } := fromAttention attn
  let out : Tensor { dtype := dtype, device := device, shape := #[batch, qSeq, dModel] } := Tensor.reshape out #[batch, qSeq, dModel] (by head_split)
  Tensor.affine3d out m.outProjWeight m.outProjBias

def forwardCross {batch qSeq kvSeq : Nat}
    (m : WhisperAttention (device := device) (dtype := dtype) dModel nHeads) (hdiv : nHeads ∣ dModel)
    (queryStates : Tensor { dtype := dtype, device := device, shape := #[batch, qSeq, dModel] })
    (keyValueStates : Tensor { dtype := dtype, device := device, shape := #[batch, kvSeq, dModel] })
    (isCausal : Bool := false)
    : Tensor { dtype := dtype, device := device, shape := #[batch, qSeq, dModel] } :=
  let (kh, vh) := projectKV m hdiv keyValueStates
  forwardWithProjectedKV m hdiv queryStates kh vh isCausal

def forwardSelf {batch seq : Nat}
    (m : WhisperAttention (device := device) (dtype := dtype) dModel nHeads) (hdiv : nHeads ∣ dModel)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, dModel] })
    (isCausal : Bool := false)
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, dModel] } :=
  forwardCross m hdiv x x isCausal

/-- Incremental one-token self-attention step with KV cache update. -/
def forwardSelfStep {batch : Nat}
    (m : WhisperAttention (device := device) (dtype := dtype) dModel nHeads) (hdiv : nHeads ∣ dModel)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, 1, dModel] })
    (cache : KVCache (device := device) (dtype := dtype) batch nHeads (dModel / nHeads))
    : Tensor { dtype := dtype, device := device, shape := #[batch, 1, dModel] } × KVCache (device := device) (dtype := dtype) batch nHeads (dModel / nHeads) :=
  let headDim : Nat := dModel / nHeads
  let q0 : Tensor { dtype := dtype, device := device, shape := #[batch, 1, dModel] } := Tensor.affine3d x m.qProjWeight m.qProjBias
  let k0 : Tensor { dtype := dtype, device := device, shape := #[batch, 1, dModel] } := Tensor.affine3d x m.kProjWeight m.kProjBias
  let v0 : Tensor { dtype := dtype, device := device, shape := #[batch, 1, dModel] } := Tensor.affine3d x m.vProjWeight m.vProjBias

  let q : Tensor { dtype := dtype, device := device, shape := #[batch, 1, nHeads, headDim] } := Tensor.reshape q0 #[batch, 1, nHeads, headDim] (by head_split)
  let k : Tensor { dtype := dtype, device := device, shape := #[batch, 1, nHeads, headDim] } := Tensor.reshape k0 #[batch, 1, nHeads, headDim] (by head_split)
  let v : Tensor { dtype := dtype, device := device, shape := #[batch, 1, nHeads, headDim] } := Tensor.reshape v0 #[batch, 1, nHeads, headDim] (by head_split)

  let qh : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, 1, headDim] } := transposeAttention q
  let kNew : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, 1, headDim] } := transposeAttention k
  let vNew : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, 1, headDim] } := transposeAttention v

  let kStore : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, cache.maxLen, headDim] } :=
    Tensor.castLike kNew (cache.kStore)
  let vStore : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, cache.maxLen, headDim] } :=
    Tensor.castLike vNew (cache.vStore)

  if cache.seq < cache.maxLen then
    let kStore' : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, cache.maxLen, headDim] } :=
      Tensor.sliceScatter kStore 2 cache.seq kNew
    let vStore' : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, cache.maxLen, headDim] } :=
      Tensor.sliceScatter vStore 2 cache.seq vNew
    let kvLen : Nat := cache.seq + 1
    let kAll : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, kvLen, headDim] } := sliceTyped kStore' 2 0 kvLen
    let vAll : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, kvLen, headDim] } := sliceTyped vStore' 2 0 kvLen
    let attn : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, 1, headDim] } :=
      Tensor.attentionGQAQKV
        (kvHeads := nHeads)
        qh
        kAll
        vAll
        false
        true
    let out : Tensor { dtype := dtype, device := device, shape := #[batch, 1, nHeads, headDim] } := fromAttention attn
    let out : Tensor { dtype := dtype, device := device, shape := #[batch, 1, dModel] } := Tensor.reshape out #[batch, 1, dModel] (by head_split)
    let out : Tensor { dtype := dtype, device := device, shape := #[batch, 1, dModel] } := Tensor.affine3d out m.outProjWeight m.outProjBias
    let cache' : KVCache (device := device) (dtype := dtype) batch nHeads headDim := {
      kStore := kStore'
      vStore := vStore'
      seq := kvLen
      maxLen := cache.maxLen
    }
    (out, cache')
  else
    let writePos : Nat := if cache.maxLen == 0 then 0 else cache.maxLen - 1
    let kStore' : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, cache.maxLen, headDim] } :=
      Tensor.sliceScatter kStore 2 writePos kNew
    let vStore' : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, cache.maxLen, headDim] } :=
      Tensor.sliceScatter vStore 2 writePos vNew
    let kvLen : Nat := cache.maxLen
    let kAll : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, kvLen, headDim] } := sliceTyped kStore' 2 0 kvLen
    let vAll : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, kvLen, headDim] } := sliceTyped vStore' 2 0 kvLen
    let attn : Tensor { dtype := dtype, device := device, shape := #[batch, nHeads, 1, headDim] } :=
      Tensor.attentionGQAQKV
        (kvHeads := nHeads)
        qh
        kAll
        vAll
        false
        true
    let out : Tensor { dtype := dtype, device := device, shape := #[batch, 1, nHeads, headDim] } := fromAttention attn
    let out : Tensor { dtype := dtype, device := device, shape := #[batch, 1, dModel] } := Tensor.reshape out #[batch, 1, dModel] (by head_split)
    let out : Tensor { dtype := dtype, device := device, shape := #[batch, 1, dModel] } := Tensor.affine3d out m.outProjWeight m.outProjBias
    let cache' : KVCache (device := device) (dtype := dtype) batch nHeads headDim := {
      kStore := kStore'
      vStore := vStore'
      seq := cache.maxLen
      maxLen := cache.maxLen
    }
    (out, cache')

end WhisperAttention

structure WhisperEncoderLayer (cfg : WhisperConfig) (dtype : DType := .Float32) where
  selfAttn : WhisperAttention (device := device) (dtype := dtype) cfg.dModel cfg.encoderAttentionHeads
  selfAttnLayerNorm : LayerNorm (device := device) cfg.dModel dtype
  fc1Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.encoderFfnDim, cfg.dModel] }
  fc1Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.encoderFfnDim] }
  fc2Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel, cfg.encoderFfnDim] }
  fc2Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel] }
  finalLayerNorm : LayerNorm (device := device) cfg.dModel dtype
  deriving TensorStruct

namespace WhisperEncoderLayer

def init (cfg : WhisperConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (WhisperEncoderLayer (device := device) (dtype := dtype) cfg) := do
  let selfAttn ← WhisperAttention.init (device := device) (dtype := dtype) cfg.dModel cfg.encoderAttentionHeads
  let selfAttnLayerNorm ← initLayerNorm (device := device) (dtype := dtype) cfg.dModel cfg.layerNormEps
  let fc1Weight ← initWeightFor (device := device) (dtype := dtype) #[cfg.encoderFfnDim, cfg.dModel] cfg.dModel
  let fc1Bias ← initBiasFor (device := device) (dtype := dtype) #[cfg.encoderFfnDim]
  let fc2Weight ← initWeightFor (device := device) (dtype := dtype) #[cfg.dModel, cfg.encoderFfnDim] cfg.encoderFfnDim
  let fc2Bias ← initBiasFor (device := device) (dtype := dtype) #[cfg.dModel]
  let finalLayerNorm ← initLayerNorm (device := device) (dtype := dtype) cfg.dModel cfg.layerNormEps
  pure { selfAttn, selfAttnLayerNorm, fc1Weight, fc1Bias, fc2Weight, fc2Bias, finalLayerNorm }

def forward {batch seq : Nat}
    (m : WhisperEncoderLayer (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } :=
  let h0 := m.selfAttnLayerNorm.forward3d x
  let h1 := m.selfAttn.forwardSelf cfg.encoderHeadsDivide h0 (isCausal := false)
  let x1 := Tensor.add x h1
  let h2 := m.finalLayerNorm.forward3d x1
  let h3 : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.encoderFfnDim] } := Tensor.affine3d h2 m.fc1Weight m.fc1Bias
  let h4 := activate cfg.activationFunction h3
  let h5 : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } := Tensor.affine3d h4 m.fc2Weight m.fc2Bias
  Tensor.add x1 h5

end WhisperEncoderLayer

structure WhisperDecoderLayer (cfg : WhisperConfig) (dtype : DType := .Float32) where
  selfAttn : WhisperAttention (device := device) (dtype := dtype) cfg.dModel cfg.decoderAttentionHeads
  selfAttnLayerNorm : LayerNorm (device := device) cfg.dModel dtype
  encoderAttn : WhisperAttention (device := device) (dtype := dtype) cfg.dModel cfg.decoderAttentionHeads
  encoderAttnLayerNorm : LayerNorm (device := device) cfg.dModel dtype
  fc1Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.decoderFfnDim, cfg.dModel] }
  fc1Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.decoderFfnDim] }
  fc2Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel, cfg.decoderFfnDim] }
  fc2Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel] }
  finalLayerNorm : LayerNorm (device := device) cfg.dModel dtype
  deriving TensorStruct

namespace WhisperDecoderLayer

def init (cfg : WhisperConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (WhisperDecoderLayer (device := device) (dtype := dtype) cfg) := do
  let selfAttn ← WhisperAttention.init (device := device) (dtype := dtype) cfg.dModel cfg.decoderAttentionHeads
  let selfAttnLayerNorm ← initLayerNorm (device := device) (dtype := dtype) cfg.dModel cfg.layerNormEps
  let encoderAttn ← WhisperAttention.init (device := device) (dtype := dtype) cfg.dModel cfg.decoderAttentionHeads
  let encoderAttnLayerNorm ← initLayerNorm (device := device) (dtype := dtype) cfg.dModel cfg.layerNormEps
  let fc1Weight ← initWeightFor (device := device) (dtype := dtype) #[cfg.decoderFfnDim, cfg.dModel] cfg.dModel
  let fc1Bias ← initBiasFor (device := device) (dtype := dtype) #[cfg.decoderFfnDim]
  let fc2Weight ← initWeightFor (device := device) (dtype := dtype) #[cfg.dModel, cfg.decoderFfnDim] cfg.decoderFfnDim
  let fc2Bias ← initBiasFor (device := device) (dtype := dtype) #[cfg.dModel]
  let finalLayerNorm ← initLayerNorm (device := device) (dtype := dtype) cfg.dModel cfg.layerNormEps
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

def forward {batch seq encSeq : Nat}
    (m : WhisperDecoderLayer (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] })
    (encoderHidden : Tensor { dtype := dtype, device := device, shape := #[batch, encSeq, cfg.dModel] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } :=
  let h0 := m.selfAttnLayerNorm.forward3d x
  let h1 := m.selfAttn.forwardSelf cfg.decoderHeadsDivide h0 (isCausal := true)
  let x1 := Tensor.add x h1
  let h2 := m.encoderAttnLayerNorm.forward3d x1
  let h3 := m.encoderAttn.forwardCross cfg.decoderHeadsDivide h2 encoderHidden (isCausal := false)
  let x2 := Tensor.add x1 h3
  let h4 := m.finalLayerNorm.forward3d x2
  let h5 : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.decoderFfnDim] } := Tensor.affine3d h4 m.fc1Weight m.fc1Bias
  let h6 := activate cfg.activationFunction h5
  let h7 : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } := Tensor.affine3d h6 m.fc2Weight m.fc2Bias
  Tensor.add x2 h7

/-- One-token decoder layer step with incremental self KV cache and precomputed cross KV. -/
def forwardStepCached {batch encSeq : Nat}
    (m : WhisperDecoderLayer (device := device) (dtype := dtype) cfg)
    (x : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.dModel] })
    (crossK : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads] })
    (crossV : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads] })
    (selfCache : WhisperAttention.KVCache (device := device) (dtype := dtype) batch cfg.decoderAttentionHeads (cfg.dModel / cfg.decoderAttentionHeads))
    : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.dModel] } × WhisperAttention.KVCache (device := device) (dtype := dtype) batch cfg.decoderAttentionHeads (cfg.dModel / cfg.decoderAttentionHeads) :=
  let h0 := m.selfAttnLayerNorm.forward3d x
  let (h1a, selfCache') := m.selfAttn.forwardSelfStep cfg.decoderHeadsDivide h0 selfCache
  let x1 := Tensor.add x h1a
  let h2 := m.encoderAttnLayerNorm.forward3d x1
  let h3 := m.encoderAttn.forwardWithProjectedKV cfg.decoderHeadsDivide h2 crossK crossV (isCausal := false)
  let x2 := Tensor.add x1 h3
  let h4 := m.finalLayerNorm.forward3d x2
  let h5 : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.decoderFfnDim] } := Tensor.affine3d h4 m.fc1Weight m.fc1Bias
  let h6 := activate cfg.activationFunction h5
  let h7 : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.dModel] } := Tensor.affine3d h6 m.fc2Weight m.fc2Bias
  (Tensor.add x2 h7, selfCache')

end WhisperDecoderLayer

structure WhisperModel (cfg : WhisperConfig) (dtype : DType := .Float32) where
  encoderConv1Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel, cfg.numMelBins, 3] }
  encoderConv1Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel] }
  encoderConv2Weight : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel, cfg.dModel, 3] }
  encoderConv2Bias : Tensor { dtype := dtype, device := device, shape := #[cfg.dModel] }
  encoderPositionalEmbedding : Tensor { dtype := dtype, device := device, shape := #[cfg.maxSourcePositions, cfg.dModel] }
  encoderLayers : Array (WhisperEncoderLayer (device := device) (dtype := dtype) cfg)
  encoderLayerNorm : LayerNorm (device := device) cfg.dModel dtype

  decoderTokenEmbedding : Tensor { dtype := dtype, device := device, shape := #[cfg.vocabSize, cfg.dModel] }
  decoderPositionalEmbedding : Tensor { dtype := dtype, device := device, shape := #[cfg.maxTargetPositions, cfg.dModel] }
  decoderLayers : Array (WhisperDecoderLayer (device := device) (dtype := dtype) cfg)
  decoderLayerNorm : LayerNorm (device := device) cfg.dModel dtype
  deriving TensorStruct

namespace WhisperModel

def init (cfg : WhisperConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (WhisperModel (device := device) (dtype := dtype) cfg) := do
  if !cfg.hasValidHeads then
    throw <| IO.userError
      s!"Invalid Whisper config: d_model={cfg.dModel}, encoder_heads={cfg.encoderAttentionHeads}, decoder_heads={cfg.decoderAttentionHeads}"
  let encoderConv1Weight ← initWeightFor (device := device) (dtype := dtype) #[cfg.dModel, cfg.numMelBins, 3] cfg.numMelBins
  let encoderConv1Bias ← initBiasFor (device := device) (dtype := dtype) #[cfg.dModel]
  let encoderConv2Weight ← initWeightFor (device := device) (dtype := dtype) #[cfg.dModel, cfg.dModel, 3] cfg.dModel
  let encoderConv2Bias ← initBiasFor (device := device) (dtype := dtype) #[cfg.dModel]
  let encoderPositionalEmbedding ← initWeightFor (device := device) (dtype := dtype) #[cfg.maxSourcePositions, cfg.dModel] cfg.dModel

  let mut encoderLayers : Array (WhisperEncoderLayer (device := device) (dtype := dtype) cfg) := #[]
  for _ in [:cfg.encoderLayers] do
    encoderLayers := encoderLayers.push (← WhisperEncoderLayer.init (device := device) (dtype := dtype) cfg)
  let encoderLayerNorm ← initLayerNorm (device := device) (dtype := dtype) cfg.dModel cfg.layerNormEps

  let decoderTokenEmbedding ← initWeightFor (device := device) (dtype := dtype) #[cfg.vocabSize, cfg.dModel] cfg.dModel
  let decoderPositionalEmbedding ← initWeightFor (device := device) (dtype := dtype) #[cfg.maxTargetPositions, cfg.dModel] cfg.dModel
  let mut decoderLayers : Array (WhisperDecoderLayer (device := device) (dtype := dtype) cfg) := #[]
  for _ in [:cfg.decoderLayers] do
    decoderLayers := decoderLayers.push (← WhisperDecoderLayer.init (device := device) (dtype := dtype) cfg)
  let decoderLayerNorm ← initLayerNorm (device := device) (dtype := dtype) cfg.dModel cfg.layerNormEps

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

def encode {batch frames : Nat} {featureDType : DType}
    (m : WhisperModel (device := device) (dtype := dtype) cfg)
    (inputFeatures : Tensor { dtype := featureDType, device := device, shape := #[batch, cfg.numMelBins, frames] })
    (hFrames : 0 < frames)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, WhisperConfig.conv2OutputSeq frames, cfg.dModel] }) := do
  if frames == 0 then
    throw <| IO.userError "Whisper encoder requires at least one input frame"
  let inputFeatures := Tensor.castLike m.encoderConv1Weight inputFeatures
  let x1 : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.dModel, frames] } :=
    Tensor.reshape
      (Tensor.gelu (Tensor.conv1dGroups inputFeatures m.encoderConv1Weight m.encoderConv1Bias 1 1 1 1))
      #[batch, cfg.dModel, frames]
      (by have : Tensor.convExtent frames 3 1 1 1 = frames := by unfold Tensor.convExtent; omega
          simp [this])
  let outSeq := WhisperConfig.conv2OutputSeq frames
  if outSeq > cfg.maxSourcePositions then
    throw <| IO.userError
      s!"Whisper encoder sequence {outSeq} exceeds max_source_positions {cfg.maxSourcePositions}"
  let x2 : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.dModel, outSeq] } :=
    Tensor.reshape
      (Tensor.gelu (Tensor.conv1dGroups x1 m.encoderConv2Weight m.encoderConv2Bias 2 1 1 1))
      #[batch, cfg.dModel, outSeq]
      (by have : Tensor.convExtent frames 3 2 1 1 = WhisperConfig.conv2OutputSeq frames := by
            unfold Tensor.convExtent WhisperConfig.conv2OutputSeq; omega
          simp +zetaDelta [this])
  let x2t : Tensor { dtype := dtype, device := device, shape := #[batch, outSeq, cfg.dModel] } := (Tensor.permute x2 #[0, 2, 1]).reshape #[batch, outSeq, cfg.dModel]
  let pos : Tensor { dtype := dtype, device := device, shape := #[outSeq, cfg.dModel] } := sliceTyped m.encoderPositionalEmbedding 0 0 outSeq
  let posB : Tensor { dtype := dtype, device := device, shape := #[batch, outSeq, cfg.dModel] } :=
    Tensor.expand (Tensor.reshape pos #[1, outSeq, cfg.dModel]) #[batch, outSeq, cfg.dModel]
  let mut h : Tensor { dtype := dtype, device := device, shape := #[batch, outSeq, cfg.dModel] } := Tensor.add x2t posB
  for layer in m.encoderLayers do
    h := layer.forward h
  pure (m.encoderLayerNorm.forward3d h)

def decode {batch seq encSeq : Nat}
    (m : WhisperModel (device := device) (dtype := dtype) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (encoderHidden : Tensor { dtype := dtype, device := device, shape := #[batch, encSeq, cfg.dModel] })
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] }) := do
  if seq > cfg.maxTargetPositions then
    throw <| IO.userError
      s!"Whisper decoder sequence {seq} exceeds max_target_positions {cfg.maxTargetPositions}"
  let tokEmb : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } := Tensor.embedding inputIds m.decoderTokenEmbedding
  let pos : Tensor { dtype := dtype, device := device, shape := #[seq, cfg.dModel] } := sliceTyped m.decoderPositionalEmbedding 0 0 seq
  let posB : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } :=
    Tensor.expand (Tensor.reshape pos #[1, seq, cfg.dModel]) #[batch, seq, cfg.dModel]
  let mut h : Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.dModel] } := Tensor.add tokEmb posB
  for layer in m.decoderLayers do
    h := layer.forward h encoderHidden
  pure (m.decoderLayerNorm.forward3d h)

end WhisperModel

structure WhisperForConditionalGeneration (cfg : WhisperConfig) (dtype : DType := .Float32) where
  model : WhisperModel (device := device) (dtype := dtype) cfg
  projOut : Tensor { dtype := dtype, device := device, shape := #[cfg.vocabSize, cfg.dModel] }
  deriving TensorStruct

namespace WhisperForConditionalGeneration

/-- Per-layer precomputed cross-attention K/V for one encoded audio chunk. -/
structure LayerCrossCache (cfg : WhisperConfig) (batch encSeq : Nat) (dtype : DType := .Float32) where
  k : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads] }
  v : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.decoderAttentionHeads, encSeq, cfg.dModel / cfg.decoderAttentionHeads] }

/-- Per-layer self-attention KV cache for decoder incremental decode. -/
abbrev LayerKVCache (cfg : WhisperConfig) (batch : Nat) (dtype : DType := .Float32) :=
  WhisperAttention.KVCache (device := device) (dtype := dtype) batch cfg.decoderAttentionHeads (cfg.dModel / cfg.decoderAttentionHeads)

def init (cfg : WhisperConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (WhisperForConditionalGeneration (device := device) (dtype := dtype) cfg) := do
  let model ← WhisperModel.init (device := device) (dtype := dtype) cfg
  let projOut ← initWeightFor (device := device) (dtype := dtype) #[cfg.vocabSize, cfg.dModel] cfg.dModel
  pure { model, projOut }

def encode {batch frames : Nat} {featureDType : DType}
    (m : WhisperForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (inputFeatures : Tensor { dtype := featureDType, device := device, shape := #[batch, cfg.numMelBins, frames] })
    (hFrames : 0 < frames)
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, WhisperConfig.conv2OutputSeq frames, cfg.dModel] }) :=
  m.model.encode inputFeatures hFrames

def decode {batch seq encSeq : Nat}
    (m : WhisperForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (encoderHidden : Tensor { dtype := dtype, device := device, shape := #[batch, encSeq, cfg.dModel] })
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, seq, cfg.vocabSize] }) := do
  let hidden ← m.model.decode inputIds encoderHidden
  pure (Tensor.linear3d hidden m.projOut)

/-- Initialize one self-attention KV cache per decoder layer. -/
def initLayerKVCaches {batch : Nat}
    (m : WhisperForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (maxLen : Nat)
    (device : Device)
    [DType.ConstantSupported dtype device]
    : Array (LayerKVCache (device := device) (dtype := dtype) cfg batch) :=
  m.model.decoderLayers.map (fun _ =>
    WhisperAttention.initKVCache
      (dtype := dtype) maxLen
      (batch := batch)
      (nHeads := cfg.decoderAttentionHeads)
      (headDim := cfg.dModel / cfg.decoderAttentionHeads)
      device)

/-- Precompute cross-attention K/V once per decoder layer for an encoded chunk. -/
def precomputeCrossCaches {batch encSeq : Nat}
    (m : WhisperForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (encoderHidden : Tensor { dtype := dtype, device := device, shape := #[batch, encSeq, cfg.dModel] })
    : IO (Array (LayerCrossCache (device := device) (dtype := dtype) cfg batch encSeq)) := do
  let mut out : Array (LayerCrossCache (device := device) (dtype := dtype) cfg batch encSeq) := #[]
  for layer in m.model.decoderLayers do
    let (k, v) := layer.encoderAttn.projectKV cfg.decoderHeadsDivide encoderHidden
    out := out.push { k, v }
  pure out

/-- Decode one token step using static self KV caches and precomputed cross K/V. -/
def decodeStepWithCache {batch encSeq : Nat}
    (m : WhisperForConditionalGeneration (device := device) (dtype := dtype) cfg)
    (tokenIds : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 })
    (position : Nat)
    (crossCaches : Array (LayerCrossCache (device := device) (dtype := dtype) cfg batch encSeq))
    (selfCaches : Array (LayerKVCache (device := device) (dtype := dtype) cfg batch))
    : IO (Tensor { dtype := dtype, device := device, shape := #[batch, cfg.vocabSize] } × Array (LayerKVCache (device := device) (dtype := dtype) cfg batch)) := do
  if position >= cfg.maxTargetPositions then
    throw <| IO.userError
      s!"Whisper decoder position {position} exceeds max_target_positions {cfg.maxTargetPositions}"
  let tokEmb : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.dModel] } := Tensor.embedding tokenIds m.model.decoderTokenEmbedding
  let pos : Tensor { dtype := dtype, device := device, shape := #[1, cfg.dModel] } := sliceTyped m.model.decoderPositionalEmbedding 0 position 1
  let posB : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.dModel] } :=
    Tensor.expand (Tensor.reshape pos #[1, 1, cfg.dModel]) #[batch, 1, cfg.dModel]
  let mut h : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.dModel] } := Tensor.add tokEmb posB
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
    let (hNext, cacheNext) := layer.forwardStepCached h crossCache.k crossCache.v selfCache
    h := hNext
    nextCaches := nextCaches.set! i cacheNext
  let hiddenNorm : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.dModel] } := m.model.decoderLayerNorm.forward3d h
  let logits3 : Tensor { dtype := dtype, device := device, shape := #[batch, 1, cfg.vocabSize] } := Tensor.linear3d hiddenNorm m.projOut
  let logits2 : Tensor { dtype := dtype, device := device, shape := #[batch, cfg.vocabSize] } := Tensor.reshape logits3 #[batch, cfg.vocabSize]
  pure (logits2, nextCaches)

end WhisperForConditionalGeneration

end torch.whisper
