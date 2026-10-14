/-
  Tyr/Model/Qwen3ASR/Weights.lean

  Pretrained weight loading for Lean Qwen3-ASR port.
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Log
import Tyr.SafeTensors.Load
import Tyr.Module.RMSNorm
import Tyr.Model.Utils
import Tyr.Model.Qwen3ASR.Model

namespace torch.qwen3asr

open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (tryLoadTensorSharded)

private def loadWeight (modelDir name : String) (shape : Shape) : IO (Tensor { shape := shape }) :=
  safetensors.loadTensorShardedCandidates modelDir #[name] shape


private def loadLayerNormSharded (modelDir : String) (name : String) (dim : UInt64)
    : IO (LayerNorm dim (.Unknown "dynamic")) := do
  let w ← loadWeight modelDir s!"{name}.weight" #[dim]
  let b ← loadWeight modelDir s!"{name}.bias" #[dim]
  pure { weight := reqGradFalse w, bias := reqGradFalse b, eps := ⟨1e-5⟩ }

private def loadAudioAttentionSharded (modelDir : String) (namePrefix : String) (cfg : AudioEncoderConfig)
    : IO (AudioAttention cfg) := do
  let headDim := AudioEncoderConfig.headDim cfg
  let projDim := cfg.encoderAttentionHeads * headDim
  let qProjWeight ← loadWeight modelDir s!"{namePrefix}.q_proj.weight" #[projDim, cfg.dModel]
  let qProjBias ← loadWeight modelDir s!"{namePrefix}.q_proj.bias" #[projDim]
  let kProjWeight ← loadWeight modelDir s!"{namePrefix}.k_proj.weight" #[projDim, cfg.dModel]
  let kProjBias ← loadWeight modelDir s!"{namePrefix}.k_proj.bias" #[projDim]
  let vProjWeight ← loadWeight modelDir s!"{namePrefix}.v_proj.weight" #[projDim, cfg.dModel]
  let vProjBias ← loadWeight modelDir s!"{namePrefix}.v_proj.bias" #[projDim]
  let outProjWeight ← loadWeight modelDir s!"{namePrefix}.out_proj.weight"
    #[cfg.dModel, projDim]
  let outProjBias ← loadWeight modelDir s!"{namePrefix}.out_proj.bias" #[cfg.dModel]
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

private def loadAudioLayerSharded (modelDir : String) (layerIdx : UInt64) (cfg : AudioEncoderConfig)
    : IO (AudioEncoderLayer cfg) := do
  let p := s!"thinker.audio_tower.layers.{layerIdx}"
  let selfAttn ← loadAudioAttentionSharded modelDir s!"{p}.self_attn" cfg
  let selfAttnLayerNorm ← loadLayerNormSharded modelDir s!"{p}.self_attn_layer_norm" cfg.dModel
  let fc1Weight ← loadWeight modelDir s!"{p}.fc1.weight" #[cfg.encoderFfnDim, cfg.dModel]
  let fc1Bias ← loadWeight modelDir s!"{p}.fc1.bias" #[cfg.encoderFfnDim]
  let fc2Weight ← loadWeight modelDir s!"{p}.fc2.weight" #[cfg.dModel, cfg.encoderFfnDim]
  let fc2Bias ← loadWeight modelDir s!"{p}.fc2.bias" #[cfg.dModel]
  let finalLayerNorm ← loadLayerNormSharded modelDir s!"{p}.final_layer_norm" cfg.dModel
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
    : IO (AudioEncoder cfg) := do
  let conv2d1Weight ← loadWeight modelDir "thinker.audio_tower.conv2d1.weight"
    #[cfg.downsampleHiddenSize, 1, 3, 3]
  let conv2d1Bias ← loadWeight modelDir "thinker.audio_tower.conv2d1.bias"
    #[cfg.downsampleHiddenSize]
  let conv2d2Weight ← loadWeight modelDir "thinker.audio_tower.conv2d2.weight"
    #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3]
  let conv2d2Bias ← loadWeight modelDir "thinker.audio_tower.conv2d2.bias"
    #[cfg.downsampleHiddenSize]
  let conv2d3Weight ← loadWeight modelDir "thinker.audio_tower.conv2d3.weight"
    #[cfg.downsampleHiddenSize, cfg.downsampleHiddenSize, 3, 3]
  let conv2d3Bias ← loadWeight modelDir "thinker.audio_tower.conv2d3.bias"
    #[cfg.downsampleHiddenSize]

  let convOutWeight ← loadWeight modelDir "thinker.audio_tower.conv_out.weight"
    #[cfg.dModel, AudioEncoderConfig.convOutInDim cfg]

  let mut layers : Array (AudioEncoderLayer cfg) := #[]
  for i in [:cfg.encoderLayers.toNat] do
    let layer ← loadAudioLayerSharded modelDir i.toUInt64 cfg
    layers := layers.push layer

  let lnPost ← loadLayerNormSharded modelDir "thinker.audio_tower.ln_post" cfg.dModel

  let proj1Weight ← loadWeight modelDir "thinker.audio_tower.proj1.weight"
    #[cfg.dModel, cfg.dModel]
  let proj1Bias ← loadWeight modelDir "thinker.audio_tower.proj1.bias"
    #[cfg.dModel]
  let proj2Weight ← loadWeight modelDir "thinker.audio_tower.proj2.weight"
    #[cfg.outputDim, cfg.dModel]
  let proj2Bias ← loadWeight modelDir "thinker.audio_tower.proj2.bias"
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

private def loadRMSNormSharded (modelDir : String) (name : String) (dim : UInt64) (eps : Float)
    : IO (RMSNorm dim (.Unknown "dynamic")) := do
  let w ← loadWeight modelDir s!"{name}.weight" #[dim]
  pure { weight := reqGradFalse w, eps := ⟨eps⟩ }

private def loadQwenAttentionSharded (modelDir : String) (namePrefix : String)
    (hiddenSize numHeads numKVHeads headDim : UInt64)
    : IO (qwen.QwenAttention hiddenSize numHeads numKVHeads headDim (.Unknown "dynamic")) := do
  let qProj ← loadWeight modelDir s!"{namePrefix}.q_proj.weight" #[numHeads * headDim, hiddenSize]
  let kProj ← loadWeight modelDir s!"{namePrefix}.k_proj.weight" #[numKVHeads * headDim, hiddenSize]
  let vProj ← loadWeight modelDir s!"{namePrefix}.v_proj.weight" #[numKVHeads * headDim, hiddenSize]
  let oProj ← loadWeight modelDir s!"{namePrefix}.o_proj.weight" #[hiddenSize, numHeads * headDim]
  let qNorm ← tryLoadTensorSharded modelDir s!"{namePrefix}.q_norm.weight" #[headDim]
  let kNorm ← tryLoadTensorSharded modelDir s!"{namePrefix}.k_norm.weight" #[headDim]
  pure {
    q_proj := reqGradFalse qProj
    k_proj := reqGradFalse kProj
    v_proj := reqGradFalse vProj
    o_proj := reqGradFalse oProj
    q_norm := qNorm.map (fun w => reqGradFalse w)
    k_norm := kNorm.map (fun w => reqGradFalse w)
  }

private def loadQwenMLPSharded (modelDir : String) (namePrefix : String)
    (hiddenSize intermediateSize : UInt64)
    : IO (qwen.QwenMLP hiddenSize intermediateSize (.Unknown "dynamic")) := do
  let gateProj ← loadWeight modelDir s!"{namePrefix}.gate_proj.weight"
    #[intermediateSize, hiddenSize]
  let upProj ← loadWeight modelDir s!"{namePrefix}.up_proj.weight"
    #[intermediateSize, hiddenSize]
  let downProj ← loadWeight modelDir s!"{namePrefix}.down_proj.weight"
    #[hiddenSize, intermediateSize]
  pure {
    gate_proj := reqGradFalse gateProj
    up_proj := reqGradFalse upProj
    down_proj := reqGradFalse downProj
  }

private def loadQwenLayerSharded (modelDir : String) (layerIdx : UInt64) (cfg : TextConfig)
    : IO (qwen.QwenLayer cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim cfg.intermediateSize (.Unknown "dynamic")) := do
  let p := s!"thinker.model.layers.{layerIdx}"
  let inputLayernorm ← loadRMSNormSharded modelDir s!"{p}.input_layernorm" cfg.hiddenSize cfg.rmsNormEps
  let selfAttn ← loadQwenAttentionSharded modelDir s!"{p}.self_attn"
    cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim
  let postAttentionLayernorm ← loadRMSNormSharded modelDir s!"{p}.post_attention_layernorm" cfg.hiddenSize cfg.rmsNormEps
  let mlp ← loadQwenMLPSharded modelDir s!"{p}.mlp" cfg.hiddenSize cfg.intermediateSize
  pure { input_layernorm := inputLayernorm, self_attn := selfAttn, post_attention_layernorm := postAttentionLayernorm, mlp }

private def loadTextModelSharded (modelDir : String) (cfg : TextConfig)
    : IO (qwen.Qwen3Model (TextConfig.toQwenConfig cfg) (.Unknown "dynamic")) := do
  let embedTokens ← loadWeight modelDir "thinker.model.embed_tokens.weight"
    #[cfg.vocabSize, cfg.hiddenSize]
  let mut layers : Array (qwen.QwenLayer cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim cfg.intermediateSize (.Unknown "dynamic")) := #[]
  for i in [:cfg.numHiddenLayers.toNat] do
    let layer ← loadQwenLayerSharded modelDir i.toUInt64 cfg
    layers := layers.push layer
  let norm ← loadRMSNormSharded modelDir "thinker.model.norm" cfg.hiddenSize cfg.rmsNormEps
  pure {
    embed_tokens := reqGradFalse embedTokens
    layers
    norm
  }

private def loadThinkerSharded (modelDir : String) (cfg : ThinkerConfig)
    : IO (Qwen3ASRThinkerForConditionalGeneration cfg) := do
  let audioTower ← loadAudioEncoderSharded modelDir cfg.audioConfig
  let textModel ← loadTextModelSharded modelDir cfg.textConfig

  let audioProjectionWeight ←
    if h : cfg.audioConfig.outputDim = cfg.textConfig.hiddenSize then
      let eye : Tensor { shape := #[cfg.textConfig.hiddenSize, cfg.textConfig.hiddenSize] } :=
        Tensor.castLike audioTower.proj2Weight (reqGradFalse (Tensor.eye cfg.textConfig.hiddenSize))
      let eye' : Tensor { shape := #[cfg.textConfig.hiddenSize, cfg.audioConfig.outputDim] } := by
        simpa [h] using eye
      pure eye'
    else
      throw <| IO.userError
        s!"Qwen3-ASR checkpoint expects audio output dim == text hidden dim, got {cfg.audioConfig.outputDim} and {cfg.textConfig.hiddenSize}"
  let audioProjectionBias :=
    Tensor.castLike audioTower.proj2Weight (reqGradFalse (Tensor.zeros #[cfg.textConfig.hiddenSize]))

  let lmHead ← loadWeight modelDir "thinker.lm_head.weight"
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
      pure (Device.CUDA 0)
    else
      log.onWarn "TYR_DEVICE=cuda requested but CUDA is unavailable; falling back to auto device selection."
      getBestDevice
  | some "mps" => pure Device.MPS
  | some "auto" => getBestDevice
  | some _ => getBestDevice
  | none => getBestDevice

/-- Load Qwen3-ASR model from HuggingFace sharded SafeTensors directory. -/
def loadSharded (modelDir : String) (cfg : Qwen3ASRConfig := {}) (log : Handlers := {})
    : IO (Qwen3ASRForConditionalGeneration cfg) := do
  log.onInfo s!"Loading Qwen3-ASR weights from {modelDir}..."
  let thinker ← loadThinkerSharded modelDir cfg.thinkerConfig
  let targetDevice ← resolveAsrDevice log
  let thinker :=
    TensorStruct.map (fun t => t.to targetDevice) thinker
  log.onInfo s!"Qwen3-ASR target device: {repr targetDevice}"
  log.onInfo "Loaded Qwen3-ASR weights."
  pure { thinker, supportLanguages := cfg.supportLanguages }

end Qwen3ASRForConditionalGeneration

end torch.qwen3asr
