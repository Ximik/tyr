/-
  Tyr/Model/Qwen3ASR/Weights.lean

  Pretrained weight loading for Lean Qwen3-ASR port.
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Log
import Tyr.SafeTensors.Exact
import Tyr.Module.RMSNorm
import Tyr.Model.Utils
import Tyr.Model.Qwen3ASR.Model

namespace torch.qwen3asr

variable {device : Device}
open torch.Log
open torch.Model (reqGradFalse)

private def loadWeight (dtype : DType) (modelDir name : String) (shape : Shape)
    (device : Device) : IO (Tensor { shape, dtype := dtype, device }) := do
  let packed ← safetensors.loadTensorShardedExact modelDir name shape device
  IO.ofExcept (packed.atDType? dtype)

private def loadOptional (dtype : DType) (modelDir name : String) (shape : Shape)
    (device : Device) : IO (Option (Tensor { shape, dtype := dtype, device })) := do
  match ← safetensors.loadOptionalTensorShardedExact modelDir name shape device with
  | none => pure none
  | some packed => pure (some (← IO.ofExcept (packed.atDType? dtype)))

private def loadLayerNormSharded (modelDir : String) (name : String) (dim : Nat)
    (dtype : DType) (device : Device := .CPU) : IO (LayerNorm (device := device) dim dtype) := do
  let w ← loadWeight (dtype := dtype) (device := device) modelDir s!"{name}.weight" #[dim]
  let b ← loadWeight (dtype := dtype) (device := device) modelDir s!"{name}.bias" #[dim]
  pure { weight := reqGradFalse w, bias := reqGradFalse b, eps := ⟨1e-5⟩ }

private def loadAudioAttentionSharded (modelDir : String) (namePrefix : String) (cfg : AudioEncoderConfig)
    (dtype : DType) (device : Device := .CPU) : IO (AudioAttention (device := device) (dtype := dtype) cfg) := do
  let headDim := AudioEncoderConfig.headDim cfg
  let projDim := cfg.encoderAttentionHeads * headDim
  let qProjWeight ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.q_proj.weight" #[projDim, cfg.dModel]
  let qProjBias ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.q_proj.bias" #[projDim]
  let kProjWeight ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.k_proj.weight" #[projDim, cfg.dModel]
  let kProjBias ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.k_proj.bias" #[projDim]
  let vProjWeight ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.v_proj.weight" #[projDim, cfg.dModel]
  let vProjBias ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.v_proj.bias" #[projDim]
  let outProjWeight ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.out_proj.weight"
    #[cfg.dModel, projDim]
  let outProjBias ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.out_proj.bias" #[cfg.dModel]
  pure {
    qProjWeight := reqGradFalse qProjWeight
    qProjBias := reqGradFalse qProjBias
    kProjWeight := reqGradFalse kProjWeight
    kProjBias := reqGradFalse kProjBias
    vProjWeight := reqGradFalse vProjWeight
    vProjBias := reqGradFalse vProjBias
    outProjWeight := reqGradFalse outProjWeight
    outProjBias := reqGradFalse outProjBias
  }

private def loadAudioLayerSharded (modelDir : String) (layerIdx : Nat) (cfg : AudioEncoderConfig)
    (dtype : DType) (device : Device := .CPU) : IO (AudioEncoderLayer (device := device) (dtype := dtype) cfg) := do
  let p := s!"thinker.audio_tower.layers.{layerIdx}"
  let selfAttn ← loadAudioAttentionSharded (dtype := dtype) (device := device) modelDir s!"{p}.self_attn" cfg
  let selfAttnLayerNorm ← loadLayerNormSharded (dtype := dtype) (device := device) modelDir s!"{p}.self_attn_layer_norm" cfg.dModel
  let fc1Weight ← loadWeight (dtype := dtype) (device := device) modelDir s!"{p}.fc1.weight" #[cfg.encoderFfnDim, cfg.dModel]
  let fc1Bias ← loadWeight (dtype := dtype) (device := device) modelDir s!"{p}.fc1.bias" #[cfg.encoderFfnDim]
  let fc2Weight ← loadWeight (dtype := dtype) (device := device) modelDir s!"{p}.fc2.weight" #[cfg.dModel, cfg.encoderFfnDim]
  let fc2Bias ← loadWeight (dtype := dtype) (device := device) modelDir s!"{p}.fc2.bias" #[cfg.dModel]
  let finalLayerNorm ← loadLayerNormSharded (dtype := dtype) (device := device) modelDir s!"{p}.final_layer_norm" cfg.dModel
  pure {
    selfAttn
    selfAttnLayerNorm
    fc1Weight := reqGradFalse fc1Weight
    fc1Bias := reqGradFalse fc1Bias
    fc2Weight := reqGradFalse fc2Weight
    fc2Bias := reqGradFalse fc2Bias
    finalLayerNorm
  }

private def loadAudioEncoderSharded (modelDir : String) (cfg : AudioEncoderConfig)
    (dtype : DType) (device : Device)
    (conv2d1Weight : Tensor { shape := #[cfg.downsampleHiddenSize, 1, 3, 3], dtype := dtype, device }) :
    IO (AudioEncoder (device := device) (dtype := dtype) cfg) := do
  let conv2d1Bias ← loadWeight (dtype := dtype) (device := device) modelDir "thinker.audio_tower.conv2d1.bias"
    #[cfg.downsampleHiddenSize]
  let conv2d2Weight ← loadWeight (dtype := dtype) (device := device) modelDir "thinker.audio_tower.conv2d2.weight"
    #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3]
  let conv2d2Bias ← loadWeight (dtype := dtype) (device := device) modelDir "thinker.audio_tower.conv2d2.bias"
    #[cfg.downsampleHiddenSize]
  let conv2d3Weight ← loadWeight (dtype := dtype) (device := device) modelDir "thinker.audio_tower.conv2d3.weight"
    #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3]
  let conv2d3Bias ← loadWeight (dtype := dtype) (device := device) modelDir "thinker.audio_tower.conv2d3.bias"
    #[cfg.downsampleHiddenSize]

  let convOutWeight ← loadWeight (dtype := dtype) (device := device) modelDir "thinker.audio_tower.conv_out.weight"
    #[cfg.dModel, AudioEncoderConfig.convOutInDim cfg]

  let mut layers : Array (AudioEncoderLayer (device := device) (dtype := dtype) cfg) := #[]
  for i in [:cfg.encoderLayers] do
    let layer ← loadAudioLayerSharded (dtype := dtype) (device := device) modelDir i cfg
    layers := layers.push layer

  let lnPost ← loadLayerNormSharded (dtype := dtype) (device := device) modelDir "thinker.audio_tower.ln_post" cfg.dModel

  let proj1Weight ← loadWeight (dtype := dtype) (device := device) modelDir "thinker.audio_tower.proj1.weight"
    #[cfg.dModel, cfg.dModel]
  let proj1Bias ← loadWeight (dtype := dtype) (device := device) modelDir "thinker.audio_tower.proj1.bias"
    #[cfg.dModel]
  let proj2Weight ← loadWeight (dtype := dtype) (device := device) modelDir "thinker.audio_tower.proj2.weight"
    #[cfg.outputDim, cfg.dModel]
  let proj2Bias ← loadWeight (dtype := dtype) (device := device) modelDir "thinker.audio_tower.proj2.bias"
    #[cfg.outputDim]

  pure {
    conv2d1Weight := reqGradFalse conv2d1Weight
    conv2d1Bias := reqGradFalse conv2d1Bias
    conv2d2Weight := reqGradFalse conv2d2Weight
    conv2d2Bias := reqGradFalse conv2d2Bias
    conv2d3Weight := reqGradFalse conv2d3Weight
    conv2d3Bias := reqGradFalse conv2d3Bias
    convOutWeight := reqGradFalse convOutWeight
    layers
    lnPost
    proj1Weight := reqGradFalse proj1Weight
    proj1Bias := reqGradFalse proj1Bias
    proj2Weight := reqGradFalse proj2Weight
    proj2Bias := reqGradFalse proj2Bias
  }

private def loadRMSNormSharded (modelDir : String) (name : String) (dim : Nat) (eps : Float)
    (dtype : DType) (device : Device := .CPU) : IO (RMSNorm (device := device) dim dtype) := do
  let w ← loadWeight (dtype := dtype) (device := device) modelDir s!"{name}.weight" #[dim]
  pure { weight := reqGradFalse w, eps := ⟨eps⟩ }

private def loadQwenAttentionSharded (modelDir : String) (namePrefix : String)
    (hiddenSize numHeads numKVHeads headDim : Nat)
    (dtype : DType) (device : Device := .CPU) : IO (qwen.QwenAttention (device := device) hiddenSize numHeads numKVHeads headDim dtype) := do
  let qProj ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.q_proj.weight" #[numHeads * headDim, hiddenSize]
  let kProj ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.k_proj.weight" #[numKVHeads * headDim, hiddenSize]
  let vProj ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.v_proj.weight" #[numKVHeads * headDim, hiddenSize]
  let oProj ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.o_proj.weight" #[hiddenSize, numHeads * headDim]
  let qNorm ← loadOptional (dtype := dtype) (device := device) modelDir s!"{namePrefix}.q_norm.weight" #[headDim]
  let kNorm ← loadOptional (dtype := dtype) (device := device) modelDir s!"{namePrefix}.k_norm.weight" #[headDim]
  pure {
    q_proj := reqGradFalse qProj
    k_proj := reqGradFalse kProj
    v_proj := reqGradFalse vProj
    o_proj := reqGradFalse oProj
    q_norm := qNorm.map (fun w => reqGradFalse w)
    k_norm := kNorm.map (fun w => reqGradFalse w)
  }

private def loadQwenMLPSharded (modelDir : String) (namePrefix : String)
    (hiddenSize intermediateSize : Nat)
    (dtype : DType) (device : Device := .CPU) : IO (qwen.QwenMLP (device := device) hiddenSize intermediateSize dtype) := do
  let gateProj ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.gate_proj.weight"
    #[intermediateSize, hiddenSize]
  let upProj ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.up_proj.weight"
    #[intermediateSize, hiddenSize]
  let downProj ← loadWeight (dtype := dtype) (device := device) modelDir s!"{namePrefix}.down_proj.weight"
    #[hiddenSize, intermediateSize]
  pure {
    gate_proj := reqGradFalse gateProj
    up_proj := reqGradFalse upProj
    down_proj := reqGradFalse downProj
  }

private def loadQwenLayerSharded (modelDir : String) (layerIdx : Nat) (cfg : TextConfig)
    (dtype : DType) (device : Device := .CPU) : IO (qwen.QwenLayer (device := device) cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim cfg.intermediateSize dtype) := do
  let p := s!"thinker.model.layers.{layerIdx}"
  let inputLayernorm ← loadRMSNormSharded (dtype := dtype) (device := device) modelDir s!"{p}.input_layernorm" cfg.hiddenSize cfg.rmsNormEps
  let selfAttn ← loadQwenAttentionSharded (dtype := dtype) (device := device) modelDir s!"{p}.self_attn"
    cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim
  let postAttentionLayernorm ← loadRMSNormSharded (dtype := dtype) (device := device) modelDir s!"{p}.post_attention_layernorm" cfg.hiddenSize cfg.rmsNormEps
  let mlp ← loadQwenMLPSharded (dtype := dtype) (device := device) modelDir s!"{p}.mlp" cfg.hiddenSize cfg.intermediateSize
  pure { input_layernorm := inputLayernorm, self_attn := selfAttn, post_attention_layernorm := postAttentionLayernorm, mlp }

private def loadTextModelSharded (modelDir : String) (cfg : TextConfig)
    (dtype : DType) (device : Device := .CPU) : IO (qwen.Qwen3Model (device := device) (TextConfig.toQwenConfig cfg) dtype) := do
  let embedTokens ← loadWeight (dtype := dtype) (device := device) modelDir "thinker.model.embed_tokens.weight"
    #[cfg.vocabSize, cfg.hiddenSize]
  let mut layers : Array (qwen.QwenLayer (device := device) cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim cfg.intermediateSize dtype) := #[]
  for i in [:cfg.numHiddenLayers] do
    let layer ← loadQwenLayerSharded (dtype := dtype) (device := device) modelDir i cfg
    layers := layers.push layer
  let norm ← loadRMSNormSharded (dtype := dtype) (device := device) modelDir "thinker.model.norm" cfg.hiddenSize cfg.rmsNormEps
  pure {
    embed_tokens := reqGradFalse embedTokens
    layers
    norm
  }

private def loadThinkerSharded (modelDir : String) (cfg : ThinkerConfig)
    (dtype : DType) (device : Device)
    (conv2d1Weight : Tensor { shape := #[cfg.audioConfig.downsampleHiddenSize, 1, 3, 3], dtype := dtype, device }) :
    IO (Qwen3ASRThinkerForConditionalGeneration (device := device) (dtype := dtype) cfg) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let audioTower ← loadAudioEncoderSharded (dtype := dtype) (device := device) modelDir cfg.audioConfig conv2d1Weight
  let textModel ← loadTextModelSharded (dtype := dtype) (device := device) modelDir cfg.textConfig

  let audioProjectionWeight ←
    if h : cfg.audioConfig.outputDim = cfg.textConfig.hiddenSize then
      let eye : Tensor { dtype := dtype, device := device, shape := #[cfg.textConfig.hiddenSize, cfg.textConfig.hiddenSize] } :=
        (Tensor.eyeFor (n := cfg.textConfig.hiddenSize) dtype device)
      let eye' : Tensor { dtype := dtype, device := device, shape := #[cfg.textConfig.hiddenSize, cfg.audioConfig.outputDim] } := by
        simpa [h] using eye
      pure eye'
    else
      throw <| IO.userError
        s!"Qwen3-ASR checkpoint expects audio output dim == text hidden dim, got {cfg.audioConfig.outputDim} and {cfg.textConfig.hiddenSize}"
  let audioProjectionBias :=
    (Tensor.zerosFor { shape := #[cfg.textConfig.hiddenSize], dtype := dtype, device })

  let lmHead ← loadWeight (dtype := dtype) (device := device) modelDir "thinker.lm_head.weight"
    #[ThinkerConfig.lmHeadOutDim cfg, cfg.textConfig.hiddenSize]

  pure {
    audioTower
    textModel
    audioProjectionWeight
    audioProjectionBias
    lmHead := reqGradFalse lmHead
  }

namespace Qwen3ASRForConditionalGeneration

/-- Resolve runtime target device for ASR inference.
    Honors `TYR_DEVICE=cpu|cuda|mps|auto`; defaults to `auto`. -/
private def resolveAsrDevice (log : Handlers := {}) : IO Device := do
  let requested := (← IO.getEnv "TYR_DEVICE").map String.toLower
  match requested with
  | some "cpu" => pure Device.CPU
  | some "cuda" =>
    if ← cuda_is_available then
      Device.cuda 0
    else
      log.onWarn "TYR_DEVICE=cuda requested but CUDA is unavailable; falling back to auto device selection."
      getBestDevice
  | some "mps" => Device.mps
  | some "auto" => getBestDevice
  | some _ => getBestDevice
  | none => getBestDevice

/-- Load Qwen3-ASR model from HuggingFace sharded SafeTensors directory. -/
def loadShardedAt (modelDir : String) (cfg : Qwen3ASRConfig := {}) (log : Handlers := {})
    (device : Device) : IO (Σ dtype : RealFloatingDType, Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype.val) cfg) := do
  log.onInfo s!"Loading Qwen3-ASR weights from {modelDir}..."
  let ⟨dtype, conv2d1Weight⟩ ← safetensors.loadTensorShardedExact modelDir
    "thinker.audio_tower.conv2d1.weight" #[cfg.thinkerConfig.audioConfig.downsampleHiddenSize, 1, 3, 3] device
  let floating ← IO.ofExcept (DType.realFloating? dtype)
  let _ := floating.down
  let thinker ← loadThinkerSharded (dtype := dtype) (device := device) modelDir cfg.thinkerConfig conv2d1Weight
  log.onInfo s!"Qwen3-ASR target device: {repr device}"
  log.onInfo "Loaded Qwen3-ASR weights."
  pure ⟨RealFloatingDType.ofDType dtype, { thinker, supportLanguages := cfg.supportLanguages }⟩

/-- Load using the environment-selected device, retaining that index in the result. -/
def loadSharded (modelDir : String) (cfg : Qwen3ASRConfig := {}) (log : Handlers := {}) :
    IO (Σ device : Device, Σ dtype : RealFloatingDType, Qwen3ASRForConditionalGeneration (device := device) (dtype := dtype.val) cfg) := do
  let device ← resolveAsrDevice log
  let model ← loadShardedAt modelDir cfg log device
  pure ⟨device, model⟩

end Qwen3ASRForConditionalGeneration

end torch.qwen3asr
