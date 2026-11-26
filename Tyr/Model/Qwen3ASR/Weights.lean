/-
  Tyr/Model/Qwen3ASR/Weights.lean

  Pretrained weight loading for Lean Qwen3-ASR port.
-/
import Tyr.Core.Typed.Neural
import Tyr.Core.TensorStruct
import Tyr.IO.Log
import Tyr.IO.SafeTensors.Source
import Tyr.Lib.Module.RMSNorm
import Tyr.Model.Qwen3ASR.Model

namespace torch.qwen3asr

variable {device : Device}
open torch.Log

/-- Admit a stored weight at the model dtype: equality only unless `cast`,
which converts it once here. -/
private def admitDType {shape : Shape} (cast : Bool) (dtype : DType) {device : Device}
    (tensor : SomeDTypeTensor shape device) : IO (Tensor { shape, dtype := dtype, device }) :=
  if cast then pure (tensor.2.castDType dtype) else IO.ofExcept (tensor.atDType? dtype)

private def loadWeight (dtype : DType) (cast : Bool) (modelDir name : String) (shape : Shape)
    (device : Device) : IO (Tensor { shape, dtype := dtype, device }) := do
  admitDType cast dtype (← (← safetensors.Source.open modelDir).load name shape device)

private def loadOptional (dtype : DType) (cast : Bool) (modelDir name : String) (shape : Shape)
    (device : Device) : IO (Option (Tensor { shape, dtype := dtype, device })) := do
  match ← (← safetensors.Source.open modelDir).load? name shape device with
  | none => pure none
  | some packed => pure (some (← admitDType cast dtype packed))

private def loadLayerNormSharded (modelDir : String) (name : String) (dim : Nat)
    (dtype : DType) (cast : Bool) (device : Device := .CPU) : IO (LayerNorm (device := device) dim dtype) := do
  let w ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{name}.weight" #[dim]
  let b ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{name}.bias" #[dim]
  pure { weight := Tensor.detach w, bias := Tensor.detach b, eps := ⟨1e-5⟩ }

private def loadAudioAttentionSharded (modelDir : String) (namePrefix : String) (cfg : AudioEncoderConfig)
    (dtype : DType) (cast : Bool) (device : Device := .CPU) : IO (AudioAttention (device := device) (dtype := dtype) cfg) := do
  let headDim := AudioEncoderConfig.headDim cfg
  let projDim := cfg.encoderAttentionHeads * headDim
  let qProjWeight ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.q_proj.weight" #[projDim, cfg.dModel]
  let qProjBias ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.q_proj.bias" #[projDim]
  let kProjWeight ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.k_proj.weight" #[projDim, cfg.dModel]
  let kProjBias ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.k_proj.bias" #[projDim]
  let vProjWeight ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.v_proj.weight" #[projDim, cfg.dModel]
  let vProjBias ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.v_proj.bias" #[projDim]
  let outProjWeight ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.out_proj.weight"
    #[cfg.dModel, projDim]
  let outProjBias ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.out_proj.bias" #[cfg.dModel]
  pure {
    qProjWeight := Tensor.detach qProjWeight
    qProjBias := Tensor.detach qProjBias
    kProjWeight := Tensor.detach kProjWeight
    kProjBias := Tensor.detach kProjBias
    vProjWeight := Tensor.detach vProjWeight
    vProjBias := Tensor.detach vProjBias
    outProjWeight := Tensor.detach outProjWeight
    outProjBias := Tensor.detach outProjBias
  }

private def loadAudioLayerSharded (modelDir : String) (layerIdx : Nat) (cfg : AudioEncoderConfig)
    (dtype : DType) (cast : Bool) (device : Device := .CPU) : IO (AudioEncoderLayer (device := device) (dtype := dtype) cfg) := do
  let p := s!"thinker.audio_tower.layers.{layerIdx}"
  let selfAttn ← loadAudioAttentionSharded (dtype := dtype) (cast := cast) (device := device) modelDir s!"{p}.self_attn" cfg
  let selfAttnLayerNorm ← loadLayerNormSharded (dtype := dtype) (cast := cast) (device := device) modelDir s!"{p}.self_attn_layer_norm" cfg.dModel
  let fc1Weight ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{p}.fc1.weight" #[cfg.encoderFfnDim, cfg.dModel]
  let fc1Bias ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{p}.fc1.bias" #[cfg.encoderFfnDim]
  let fc2Weight ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{p}.fc2.weight" #[cfg.dModel, cfg.encoderFfnDim]
  let fc2Bias ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{p}.fc2.bias" #[cfg.dModel]
  let finalLayerNorm ← loadLayerNormSharded (dtype := dtype) (cast := cast) (device := device) modelDir s!"{p}.final_layer_norm" cfg.dModel
  pure {
    selfAttn
    selfAttnLayerNorm
    fc1Weight := Tensor.detach fc1Weight
    fc1Bias := Tensor.detach fc1Bias
    fc2Weight := Tensor.detach fc2Weight
    fc2Bias := Tensor.detach fc2Bias
    finalLayerNorm
  }

private def loadAudioEncoderSharded (modelDir : String) (cfg : AudioEncoderConfig)
    (dtype : DType) (cast : Bool) (device : Device)
    (conv2d1Weight : Tensor { shape := #[cfg.downsampleHiddenSize, 1, 3, 3], dtype := dtype, device }) :
    IO (AudioEncoder (device := device) (dtype := dtype) cfg) := do
  let conv2d1Bias ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.audio_tower.conv2d1.bias"
    #[cfg.downsampleHiddenSize]
  let conv2d2Weight ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.audio_tower.conv2d2.weight"
    #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3]
  let conv2d2Bias ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.audio_tower.conv2d2.bias"
    #[cfg.downsampleHiddenSize]
  let conv2d3Weight ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.audio_tower.conv2d3.weight"
    #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3]
  let conv2d3Bias ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.audio_tower.conv2d3.bias"
    #[cfg.downsampleHiddenSize]

  let convOutWeight ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.audio_tower.conv_out.weight"
    #[cfg.dModel, AudioEncoderConfig.convOutInDim cfg]

  let layers ← Vector.ofFnM fun (i : Fin cfg.encoderLayers) =>
    loadAudioLayerSharded (dtype := dtype) (cast := cast) (device := device) modelDir i cfg

  let lnPost ← loadLayerNormSharded (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.audio_tower.ln_post" cfg.dModel

  let proj1Weight ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.audio_tower.proj1.weight"
    #[cfg.dModel, cfg.dModel]
  let proj1Bias ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.audio_tower.proj1.bias"
    #[cfg.dModel]
  let proj2Weight ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.audio_tower.proj2.weight"
    #[cfg.outputDim, cfg.dModel]
  let proj2Bias ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.audio_tower.proj2.bias"
    #[cfg.outputDim]

  pure {
    conv2d1Weight := Tensor.detach conv2d1Weight
    conv2d1Bias := Tensor.detach conv2d1Bias
    conv2d2Weight := Tensor.detach conv2d2Weight
    conv2d2Bias := Tensor.detach conv2d2Bias
    conv2d3Weight := Tensor.detach conv2d3Weight
    conv2d3Bias := Tensor.detach conv2d3Bias
    convOutWeight := Tensor.detach convOutWeight
    layers
    lnPost
    proj1Weight := Tensor.detach proj1Weight
    proj1Bias := Tensor.detach proj1Bias
    proj2Weight := Tensor.detach proj2Weight
    proj2Bias := Tensor.detach proj2Bias
  }

private def loadRMSNormSharded (modelDir : String) (name : String) (dim : Nat) (eps : Float)
    (dtype : DType) (cast : Bool) (device : Device := .CPU) : IO (RMSNorm (device := device) dim dtype) := do
  let w ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{name}.weight" #[dim]
  pure { weight := Tensor.detach w, eps := ⟨eps⟩ }

private def loadQwenAttentionSharded (modelDir : String) (namePrefix : String)
    (hiddenSize numHeads numKVHeads headDim : Nat)
    (dtype : DType) (cast : Bool) (device : Device := .CPU) : IO (qwen.QwenAttention (device := device) hiddenSize numHeads numKVHeads headDim dtype) := do
  let qProj ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.q_proj.weight" #[numHeads * headDim, hiddenSize]
  let kProj ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.k_proj.weight" #[numKVHeads * headDim, hiddenSize]
  let vProj ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.v_proj.weight" #[numKVHeads * headDim, hiddenSize]
  let oProj ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.o_proj.weight" #[hiddenSize, numHeads * headDim]
  let qNorm ← loadOptional (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.q_norm.weight" #[headDim]
  let kNorm ← loadOptional (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.k_norm.weight" #[headDim]
  pure {
    q_proj := Tensor.detach qProj
    k_proj := Tensor.detach kProj
    v_proj := Tensor.detach vProj
    o_proj := Tensor.detach oProj
    q_norm := qNorm.map (fun w => Tensor.detach w)
    k_norm := kNorm.map (fun w => Tensor.detach w)
  }

private def loadQwenMLPSharded (modelDir : String) (namePrefix : String)
    (hiddenSize intermediateSize : Nat)
    (dtype : DType) (cast : Bool) (device : Device := .CPU) : IO (qwen.QwenMLP (device := device) hiddenSize intermediateSize dtype) := do
  let gateProj ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.gate_proj.weight"
    #[intermediateSize, hiddenSize]
  let upProj ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.up_proj.weight"
    #[intermediateSize, hiddenSize]
  let downProj ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir s!"{namePrefix}.down_proj.weight"
    #[hiddenSize, intermediateSize]
  pure {
    gate_proj := Tensor.detach gateProj
    up_proj := Tensor.detach upProj
    down_proj := Tensor.detach downProj
  }

private def loadQwenLayerSharded (modelDir : String) (layerIdx : Nat) (cfg : TextConfig)
    (dtype : DType) (cast : Bool) (device : Device := .CPU) : IO (qwen.QwenLayer (device := device) cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim cfg.intermediateSize dtype) := do
  let p := s!"thinker.model.layers.{layerIdx}"
  let inputLayernorm ← loadRMSNormSharded (dtype := dtype) (cast := cast) (device := device) modelDir s!"{p}.input_layernorm" cfg.hiddenSize cfg.rmsNormEps
  let selfAttn ← loadQwenAttentionSharded (dtype := dtype) (cast := cast) (device := device) modelDir s!"{p}.self_attn"
    cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim
  let postAttentionLayernorm ← loadRMSNormSharded (dtype := dtype) (cast := cast) (device := device) modelDir s!"{p}.post_attention_layernorm" cfg.hiddenSize cfg.rmsNormEps
  let mlp ← loadQwenMLPSharded (dtype := dtype) (cast := cast) (device := device) modelDir s!"{p}.mlp" cfg.hiddenSize cfg.intermediateSize
  pure { input_layernorm := inputLayernorm, self_attn := selfAttn, post_attention_layernorm := postAttentionLayernorm, mlp }

private def loadTextModelSharded (modelDir : String) (cfg : TextConfig)
    (dtype : DType) (cast : Bool) (device : Device := .CPU) : IO (qwen.Qwen3Model (device := device) (TextConfig.toQwenConfig cfg) dtype) := do
  let embedTokens ← loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.model.embed_tokens.weight"
    #[cfg.vocabSize, cfg.hiddenSize]
  let layers ← Vector.ofFnM fun (i : Fin cfg.numHiddenLayers) =>
    loadQwenLayerSharded (dtype := dtype) (cast := cast) (device := device) modelDir i cfg
  let norm ← loadRMSNormSharded (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.model.norm" cfg.hiddenSize cfg.rmsNormEps
  pure {
    embed_tokens := Tensor.detach embedTokens
    layers
    norm
  }

private def loadThinkerSharded (modelDir : String) (cfg : ThinkerConfig)
    (dtype : DType) (cast : Bool) (device : Device)
    (conv2d1Weight : Tensor { shape := #[cfg.audioConfig.downsampleHiddenSize, 1, 3, 3], dtype := dtype, device }) :
    IO (Qwen3ASRThinkerForConditionalGeneration (device := device) (dtype := dtype) cfg) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let audioTower ← loadAudioEncoderSharded (dtype := dtype) (cast := cast) (device := device) modelDir cfg.audioConfig conv2d1Weight
  let textModel ← loadTextModelSharded (dtype := dtype) (cast := cast) (device := device) modelDir cfg.textConfig

  let audioProjectionWeight ← do
    have h := cfg.audioMatchesText
    let eye : Tensor { dtype := dtype, device := device, shape := #[cfg.textConfig.hiddenSize, cfg.textConfig.hiddenSize] } :=
      (Tensor.eyeFor (n := cfg.textConfig.hiddenSize) dtype device)
    let eye' : Tensor { dtype := dtype, device := device, shape := #[cfg.textConfig.hiddenSize, cfg.audioConfig.outputDim] } := by
      simpa [h] using eye
    pure eye'
  let audioProjectionBias :=
    (Tensor.zerosFor { shape := #[cfg.textConfig.hiddenSize], dtype := dtype, device })

  let lmHead ←
    if cfg.textConfig.tieWordEmbeddings then
      if h : ThinkerConfig.lmHeadOutDim cfg = cfg.textConfig.vocabSize then
        pure (h ▸ textModel.embed_tokens)
      else
        throw <| IO.userError "tie_word_embeddings needs an output head of vocab_size rows"
    else
      loadWeight (dtype := dtype) (cast := cast) (device := device) modelDir "thinker.lm_head.weight"
        #[ThinkerConfig.lmHeadOutDim cfg, cfg.textConfig.hiddenSize]

  pure {
    audioTower
    textModel
    audioProjectionWeight
    audioProjectionBias
    lmHead := Tensor.detach lmHead
  }

namespace Qwen3ASRForConditionalGeneration

/-- Load Qwen3-ASR model from HuggingFace sharded SafeTensors directory onto
`device`. The model dtype is `dtype?`, every weight cast to it once
here, or else the stored dtype of the first audio convolution, which every
weight must have. -/
def loadSharded (modelDir : String) (cfg : Qwen3ASRConfig) (device : Device)
    (log : Handlers := {}) (dtype? : Option RealFloatingDType := none) :
    IO (Σ dtype : RealFloatingDType, Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype.val) cfg) := do
  log.onInfo s!"Loading Qwen3-ASR weights from {modelDir}..."
  let conv2d1 ← (← safetensors.Source.open modelDir).load
    "thinker.audio_tower.conv2d1.weight" #[cfg.thinkerConfig.audioConfig.downsampleHiddenSize, 1, 3, 3] device
  let cast := dtype?.isSome
  let dtype := dtype?.map (·.val) |>.getD conv2d1.1
  let floating ← IO.ofExcept (DType.realFloating? dtype)
  let _ := floating.down
  let conv2d1Weight ← admitDType cast dtype conv2d1
  let thinker ← loadThinkerSharded (dtype := dtype) (cast := cast) (device := device) modelDir cfg.thinkerConfig conv2d1Weight
  log.onInfo s!"Qwen3-ASR target device: {repr device}"
  log.onInfo "Loaded Qwen3-ASR weights."
  pure ⟨RealFloatingDType.ofDType dtype, { thinker, supportLanguages := cfg.supportLanguages }⟩


end Qwen3ASRForConditionalGeneration

end torch.qwen3asr
