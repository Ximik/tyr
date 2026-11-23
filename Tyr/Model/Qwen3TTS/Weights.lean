/-
  Tyr/Model/Qwen3TTS/Weights.lean

  Pretrained weight loading for the Lean Qwen3-TTS port.
  Loads talker and code-predictor weights from HuggingFace-style sharded
  SafeTensors directories so generation can use real checkpoint tensors.
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Log
import Tyr.SafeTensors.Source
import Tyr.Module.RMSNorm
import Tyr.Model.Utils
import Tyr.Model.Qwen3TTS.Model

namespace torch.qwen3tts

variable {device : Device} {dtype : RealFloatingDType}

open torch.Log

open torch.Model (reqGradFalse)


/-- Admit a stored tensor at the component dtype: equality only unless `cast`,
which converts it once here. -/
private def admitDType {s : Shape} (cast : Bool) (name : String) (packed : SomeDTypeTensor s device) :
    IO (Tensor { shape := s, dtype := dtype.val, device }) :=
  if cast then pure (packed.2.castDType dtype.val) else
  match packed.atDType? dtype.val with
  | .ok tensor => pure tensor
  | .error message => throw <| IO.userError s!"Qwen3TTS tensor '{name}': {message}"

/-- Load a stored tensor at the component dtype (see `admitDType`). -/
private def loadTensorShardedTarget (modelDir : String) (cast : Bool) (name : String) (s : Shape) :
    IO (Tensor { shape := s, dtype := dtype.val, device }) := do
  admitDType cast name (← (← safetensors.Source.open modelDir).load name s device)

/-- Only absent optional entries are suppressed; every present tensor is admitted. -/
private def loadOptionalTensorShardedTarget (modelDir : String) (cast : Bool) (name : String) (s : Shape) :
    IO (Option (Tensor { shape := s, dtype := dtype.val, device })) := do
  match ← (← safetensors.Source.open modelDir).load? name s device with
  | none => pure none
  | some packed => some <$> admitDType cast name packed

private def loadRMSNormSharded (modelDir : String) (cast : Bool) (name : String) (dim : Nat) (eps : Float)
    : IO (RMSNorm (device := device) dim dtype.val) := do
  let w ← loadTensorShardedTarget modelDir cast s!"{name}.weight" #[dim]
  pure { weight := reqGradFalse w, eps := ⟨eps⟩ }

private def loadTalkerLayerSharded (modelDir : String) (cast : Bool) (layerIdx : Nat) (cfg : TalkerConfig)
    : IO (TalkerLayer (device := device) (dtype := dtype) cfg) := do
  let p := s!"talker.model.layers.{layerIdx}"
  let sp := s!"{p}.self_attn"
  let mp := s!"{p}.mlp"

  let q_proj ← loadTensorShardedTarget modelDir cast s!"{sp}.q_proj.weight"
    #[cfg.numAttentionHeads * cfg.headDim, cfg.hiddenSize]
  let k_proj ← loadTensorShardedTarget modelDir cast s!"{sp}.k_proj.weight"
    #[cfg.numKeyValueHeads * cfg.headDim, cfg.hiddenSize]
  let v_proj ← loadTensorShardedTarget modelDir cast s!"{sp}.v_proj.weight"
    #[cfg.numKeyValueHeads * cfg.headDim, cfg.hiddenSize]
  let o_proj ← loadTensorShardedTarget modelDir cast s!"{sp}.o_proj.weight"
    #[cfg.hiddenSize, cfg.numAttentionHeads * cfg.headDim]

  let q_norm ← loadOptionalTensorShardedTarget modelDir cast s!"{sp}.q_norm.weight" #[cfg.headDim]
  let k_norm ← loadOptionalTensorShardedTarget modelDir cast s!"{sp}.k_norm.weight" #[cfg.headDim]

  let gate_proj ← loadTensorShardedTarget modelDir cast s!"{mp}.gate_proj.weight"
    #[cfg.intermediateSize, cfg.hiddenSize]
  let up_proj ← loadTensorShardedTarget modelDir cast s!"{mp}.up_proj.weight"
    #[cfg.intermediateSize, cfg.hiddenSize]
  let down_proj ← loadTensorShardedTarget modelDir cast s!"{mp}.down_proj.weight"
    #[cfg.hiddenSize, cfg.intermediateSize]

  let input_layernorm ← loadRMSNormSharded modelDir cast s!"{p}.input_layernorm" cfg.hiddenSize cfg.rmsNormEps
  let post_attention_layernorm ← loadRMSNormSharded modelDir cast s!"{p}.post_attention_layernorm" cfg.hiddenSize cfg.rmsNormEps

  let self_attn : qwen.QwenAttention (device := device) cfg.hiddenSize cfg.numAttentionHeads cfg.numKeyValueHeads cfg.headDim dtype.val := {
    q_proj := reqGradFalse q_proj
    k_proj := reqGradFalse k_proj
    v_proj := reqGradFalse v_proj
    o_proj := reqGradFalse o_proj
    q_norm := q_norm.map reqGradFalse
    k_norm := k_norm.map reqGradFalse
  }
  let mlp : qwen.QwenMLP (device := device) cfg.hiddenSize cfg.intermediateSize dtype.val := {
    gate_proj := reqGradFalse gate_proj
    up_proj := reqGradFalse up_proj
    down_proj := reqGradFalse down_proj
  }
  pure { input_layernorm, self_attn, post_attention_layernorm, mlp }

private def loadCodePredictorLayerSharded (modelDir : String) (cast : Bool) (layerIdx : Nat) (cfg : TalkerConfig)
    : IO (CodePredictorLayer (device := device) (dtype := dtype) cfg.codePredictorConfig) := do
  let cp := cfg.codePredictorConfig
  let p := s!"talker.code_predictor.model.layers.{layerIdx}"
  let sp := s!"{p}.self_attn"
  let mp := s!"{p}.mlp"

  let q_proj ← loadTensorShardedTarget modelDir cast s!"{sp}.q_proj.weight"
    #[cp.numAttentionHeads * cp.headDim, cp.hiddenSize]
  let k_proj ← loadTensorShardedTarget modelDir cast s!"{sp}.k_proj.weight"
    #[cp.numKeyValueHeads * cp.headDim, cp.hiddenSize]
  let v_proj ← loadTensorShardedTarget modelDir cast s!"{sp}.v_proj.weight"
    #[cp.numKeyValueHeads * cp.headDim, cp.hiddenSize]
  let o_proj ← loadTensorShardedTarget modelDir cast s!"{sp}.o_proj.weight"
    #[cp.hiddenSize, cp.numAttentionHeads * cp.headDim]

  let q_norm ← loadOptionalTensorShardedTarget modelDir cast s!"{sp}.q_norm.weight" #[cp.headDim]
  let k_norm ← loadOptionalTensorShardedTarget modelDir cast s!"{sp}.k_norm.weight" #[cp.headDim]

  let gate_proj ← loadTensorShardedTarget modelDir cast s!"{mp}.gate_proj.weight"
    #[cp.intermediateSize, cp.hiddenSize]
  let up_proj ← loadTensorShardedTarget modelDir cast s!"{mp}.up_proj.weight"
    #[cp.intermediateSize, cp.hiddenSize]
  let down_proj ← loadTensorShardedTarget modelDir cast s!"{mp}.down_proj.weight"
    #[cp.hiddenSize, cp.intermediateSize]

  let input_layernorm ← loadRMSNormSharded modelDir cast s!"{p}.input_layernorm" cp.hiddenSize cp.rmsNormEps
  let post_attention_layernorm ← loadRMSNormSharded modelDir cast s!"{p}.post_attention_layernorm" cp.hiddenSize cp.rmsNormEps

  let self_attn : qwen.QwenAttention (device := device) cp.hiddenSize cp.numAttentionHeads cp.numKeyValueHeads cp.headDim dtype.val := {
    q_proj := reqGradFalse q_proj
    k_proj := reqGradFalse k_proj
    v_proj := reqGradFalse v_proj
    o_proj := reqGradFalse o_proj
    q_norm := q_norm.map reqGradFalse
    k_norm := k_norm.map reqGradFalse
  }
  let mlp : qwen.QwenMLP (device := device) cp.hiddenSize cp.intermediateSize dtype.val := {
    gate_proj := reqGradFalse gate_proj
    up_proj := reqGradFalse up_proj
    down_proj := reqGradFalse down_proj
  }
  pure { input_layernorm, self_attn, post_attention_layernorm, mlp }

private def loadCodePredictorProjectionSharded (modelDir : String) (cast : Bool) (cfg : TalkerConfig)
    : IO (Tensor { shape := #[cfg.codePredictorConfig.hiddenSize, cfg.hiddenSize], dtype := dtype.val, device := device } × Tensor { shape := #[cfg.codePredictorConfig.hiddenSize], dtype := dtype.val, device := device }) := do
  let cp := cfg.codePredictorConfig
  let weight ← loadOptionalTensorShardedTarget (dtype := dtype) modelDir cast
    "talker.code_predictor.small_to_mtp_projection.weight" #[cp.hiddenSize, cfg.hiddenSize]
  let bias ← loadOptionalTensorShardedTarget (dtype := dtype) modelDir cast
    "talker.code_predictor.small_to_mtp_projection.bias" #[cp.hiddenSize]
  match weight, bias with
  | some w, some b => pure (reqGradFalse w, reqGradFalse b)
  | none, none =>
    if hSame : cp.hiddenSize = cfg.hiddenSize then
      let support ← dtype.val.acquireConstantSupport device
      let _ := support.down
      let w := Tensor.eyeFor (n := cp.hiddenSize) dtype.val device
      let b := Tensor.zerosFor { shape := #[cp.hiddenSize], dtype := dtype.val, device }
      pure (w.reshape #[cfg.codePredictorConfig.hiddenSize, cfg.hiddenSize] (by simp [cp, hSame]), b)
    else
      throw <| IO.userError
        "Missing talker.code_predictor.small_to_mtp_projection.* and hidden sizes differ; cannot build projection."
  | _, _ =>
    throw <| IO.userError
      "Incomplete talker.code_predictor.small_to_mtp_projection: weight and bias must both be present or both absent."

private def loadTalkerModelSharded (modelDir : String) (cast : Bool) (cfg : TalkerConfig)
    (codecEmbedding : Tensor { shape := #[cfg.vocabSize, cfg.hiddenSize], dtype := dtype.val, device })
    : IO (TalkerModel (device := device) (dtype := dtype) cfg) := do
  let textEmbedding ← loadTensorShardedTarget modelDir cast "talker.model.text_embedding.weight"
    #[cfg.textVocabSize, cfg.textHiddenSize]

  let textProjectionFc1 ← loadTensorShardedTarget modelDir cast "talker.text_projection.linear_fc1.weight"
    #[cfg.textHiddenSize, cfg.textHiddenSize]
  let textProjectionFc1Bias ← loadTensorShardedTarget modelDir cast "talker.text_projection.linear_fc1.bias"
    #[cfg.textHiddenSize]
  let textProjectionFc2 ← loadTensorShardedTarget modelDir cast "talker.text_projection.linear_fc2.weight"
    #[cfg.hiddenSize, cfg.textHiddenSize]
  let textProjectionFc2Bias ← loadTensorShardedTarget modelDir cast "talker.text_projection.linear_fc2.bias"
    #[cfg.hiddenSize]

  let mut layers : Array (TalkerLayer (device := device) (dtype := dtype) cfg) := #[]
  for i in [:cfg.numHiddenLayers] do
    let layer ← loadTalkerLayerSharded modelDir cast i cfg
    layers := layers.push layer

  let norm ← loadRMSNormSharded modelDir cast "talker.model.norm" cfg.hiddenSize cfg.rmsNormEps
  pure {
    codecEmbedding := reqGradFalse codecEmbedding
    textEmbedding := reqGradFalse textEmbedding
    textProjectionFc1 := reqGradFalse textProjectionFc1
    textProjectionFc1Bias := reqGradFalse textProjectionFc1Bias
    textProjectionFc2 := reqGradFalse textProjectionFc2
    textProjectionFc2Bias := reqGradFalse textProjectionFc2Bias
    layers
    norm
  }

private def loadCodePredictorSharded (modelDir : String) (cast : Bool) (cfg : TalkerConfig)
    : IO (TalkerCodePredictor (device := device) (dtype := dtype) cfg) := do
  let cp := cfg.codePredictorConfig
  let nCodebooks := cfg.numCodeGroups - 1

  let codebooks : Vector Nat nCodebooks := Vector.range nCodebooks
  let inputEmbeddings ← codebooks.mapM fun i => do
    let emb ← loadTensorShardedTarget modelDir cast
      s!"talker.code_predictor.model.codec_embedding.{i}.weight"
      #[cp.vocabSize, cfg.hiddenSize]
    pure (reqGradFalse emb)

  let (inputProjection, inputProjectionBias) ← loadCodePredictorProjectionSharded modelDir cast cfg

  let mut layers : Array (CodePredictorLayer (device := device) (dtype := dtype) cp) := #[]
  for i in [:cp.numHiddenLayers] do
    let layer ← loadCodePredictorLayerSharded modelDir cast i cfg
    layers := layers.push layer

  let norm ← loadRMSNormSharded modelDir cast "talker.code_predictor.model.norm" cp.hiddenSize cp.rmsNormEps

  let lmHeads ← codebooks.mapM fun i => do
    let head ← loadTensorShardedTarget modelDir cast
      s!"talker.code_predictor.lm_head.{i}.weight"
      #[cp.vocabSize, cp.hiddenSize]
    pure (reqGradFalse head)

  pure { inputEmbeddings, inputProjection, inputProjectionBias, layers, norm, lmHeads }

private def loadTalkerForConditionalGenerationSharded (modelDir : String) (cast : Bool) (cfg : TalkerConfig)
    (codecEmbedding : Tensor { shape := #[cfg.vocabSize, cfg.hiddenSize], dtype := dtype.val, device })
    : IO (TalkerForConditionalGeneration (device := device) (dtype := dtype) cfg) := do
  let model ← loadTalkerModelSharded modelDir cast cfg codecEmbedding
  let codePredictor ← loadCodePredictorSharded modelDir cast cfg
  let codecHead ← loadTensorShardedTarget modelDir cast "talker.codec_head.weight"
    #[cfg.vocabSize, cfg.hiddenSize]
  pure { model, codePredictor, codecHead := reqGradFalse codecHead }

private def loadTimeDelayNetBlockSharded (modelDir : String) (cast : Bool) (namePrefix : String)
    (inChannels outChannels kernelSize dilation : Nat)
    : IO (TimeDelayNetBlock (device := device) (dtype := dtype) inChannels outChannels kernelSize dilation) := do
  let weight ← loadTensorShardedTarget modelDir cast s!"{namePrefix}.conv.weight"
    #[outChannels, inChannels, kernelSize]
  let bias ← loadTensorShardedTarget modelDir cast s!"{namePrefix}.conv.bias"
    #[outChannels]
  pure {
    weight := reqGradFalse weight
    bias := reqGradFalse bias
  }

private def loadSqueezeExcitationBlockSharded (modelDir : String) (cast : Bool) (namePrefix : String)
    (channels seChannels : Nat)
    : IO (SqueezeExcitationBlock (device := device) (dtype := dtype) channels seChannels) := do
  let conv1Weight ← loadTensorShardedTarget modelDir cast s!"{namePrefix}.conv1.weight"
    #[seChannels, channels, 1]
  let conv1Bias ← loadTensorShardedTarget modelDir cast s!"{namePrefix}.conv1.bias"
    #[seChannels]
  let conv2Weight ← loadTensorShardedTarget modelDir cast s!"{namePrefix}.conv2.weight"
    #[channels, seChannels, 1]
  let conv2Bias ← loadTensorShardedTarget modelDir cast s!"{namePrefix}.conv2.bias"
    #[channels]
  pure {
    conv1Weight := reqGradFalse conv1Weight
    conv1Bias := reqGradFalse conv1Bias
    conv2Weight := reqGradFalse conv2Weight
    conv2Bias := reqGradFalse conv2Bias
  }

private def loadRes2NetBlockSharded (modelDir : String) (cast : Bool) (namePrefix : String)
    (channels scale kernelSize dilation : Nat)
    : IO (Res2NetBlock (device := device) (dtype := dtype) channels scale kernelSize dilation) := do
  let indices : Vector Nat (scale - 1) := Vector.range (scale - 1)
  let blocks ← indices.mapM fun i =>
    loadTimeDelayNetBlockSharded modelDir cast
      s!"{namePrefix}.blocks.{i}"
      (channels / scale) (channels / scale) kernelSize dilation
  pure { blocks }

private def loadSqueezeExcitationRes2NetBlockSharded (modelDir : String) (cast : Bool) (namePrefix : String)
    (inChannels outChannels res2netScale seChannels kernelSize dilation : Nat)
    : IO (SqueezeExcitationRes2NetBlock (device := device) (dtype := dtype) inChannels outChannels res2netScale seChannels kernelSize dilation) := do
  let tdnn1 ← loadTimeDelayNetBlockSharded modelDir cast
    s!"{namePrefix}.tdnn1" inChannels outChannels 1 1
  let res2net ← loadRes2NetBlockSharded modelDir cast
    s!"{namePrefix}.res2net_block" outChannels res2netScale kernelSize dilation
  let tdnn2 ← loadTimeDelayNetBlockSharded modelDir cast
    s!"{namePrefix}.tdnn2" outChannels outChannels 1 1
  let seBlock ← loadSqueezeExcitationBlockSharded modelDir cast
    s!"{namePrefix}.se_block" outChannels seChannels
  pure { tdnn1, res2net, tdnn2, seBlock }

private def loadAttentiveStatisticsPoolingSharded (modelDir : String) (cast : Bool) (namePrefix : String)
    (channels attentionChannels : Nat)
    : IO (AttentiveStatisticsPooling (device := device) (dtype := dtype) channels attentionChannels) := do
  let tdnn ← loadTimeDelayNetBlockSharded modelDir cast
    s!"{namePrefix}.tdnn" (channels * 3) attentionChannels 1 1
  let convWeight ← loadTensorShardedTarget modelDir cast s!"{namePrefix}.conv.weight"
    #[channels, attentionChannels, 1]
  let convBias ← loadTensorShardedTarget modelDir cast s!"{namePrefix}.conv.bias"
    #[channels]
  pure {
    tdnn
    convWeight := reqGradFalse convWeight
    convBias := reqGradFalse convBias
    eps := 1e-12
  }

private def loadSpeakerEncoderSharded (modelDir : String) (cast : Bool) (cfg : SpeakerEncoderConfig)
    : IO (SpeakerEncoder (device := device) (dtype := dtype) cfg) := do
  let c0 := cfg.encChannels.getD 0 512
  let c1 := cfg.encChannels.getD 1 512
  let c2 := cfg.encChannels.getD 2 512
  let c3 := cfg.encChannels.getD 3 512
  let c4 := cfg.encChannels.getD 4 1536
  let k0 := cfg.encKernelSizes.getD 0 5
  let k1 := cfg.encKernelSizes.getD 1 3
  let k2 := cfg.encKernelSizes.getD 2 3
  let k3 := cfg.encKernelSizes.getD 3 3
  let k4 := cfg.encKernelSizes.getD 4 1
  let d0 := cfg.encDilations.getD 0 1
  let d1 := cfg.encDilations.getD 1 2
  let d2 := cfg.encDilations.getD 2 3
  let d3 := cfg.encDilations.getD 3 4
  let d4 := cfg.encDilations.getD 4 1
  let mfaIn := c1 + c2 + c3


  let tdnn0 ← loadTimeDelayNetBlockSharded modelDir cast
    "speaker_encoder.blocks.0" cfg.melDim c0 k0 d0
  let block1 ← loadSqueezeExcitationRes2NetBlockSharded modelDir cast
    "speaker_encoder.blocks.1" c0 c1 cfg.encRes2NetScale cfg.encSeChannels k1 d1
  let block2 ← loadSqueezeExcitationRes2NetBlockSharded modelDir cast
    "speaker_encoder.blocks.2" c1 c2 cfg.encRes2NetScale cfg.encSeChannels k2 d2
  let block3 ← loadSqueezeExcitationRes2NetBlockSharded modelDir cast
    "speaker_encoder.blocks.3" c2 c3 cfg.encRes2NetScale cfg.encSeChannels k3 d3
  let mfa ← loadTimeDelayNetBlockSharded modelDir cast
    "speaker_encoder.mfa" mfaIn c4 k4 d4
  let asp ← loadAttentiveStatisticsPoolingSharded modelDir cast
    "speaker_encoder.asp" c4 cfg.encAttentionChannels
  let fcWeight ← loadTensorShardedTarget modelDir cast "speaker_encoder.fc.weight"
    #[cfg.encDim, c4 * 2, 1]
  let fcBias ← loadTensorShardedTarget modelDir cast "speaker_encoder.fc.bias"
    #[cfg.encDim]

  pure {
    tdnn0
    block1
    block2
    block3
    mfa
    asp
    fcWeight := reqGradFalse fcWeight
    fcBias := reqGradFalse fcBias
  }

namespace Qwen3TTSForConditionalGeneration

/-- Load Qwen3-TTS from a sharded SafeTensors model directory.
    This uses real checkpoint tensors for talker/sub-talker generation and,
    for base models, the ECAPA speaker encoder as well. Both components take
    `dtype?`, every weight cast to it once here, or else their stored dtype,
    which each of their weights must have. -/
def loadSharded
    (modelDir : String)
    (cfg : Qwen3TTSConfig := {})
    (targetDevice : Device := Device.CPU)
    (log : Handlers := {})
    (dtype? : Option RealFloatingDType := none)
    : IO (Σ dtypes : DTypeLayout, Qwen3TTSForConditionalGeneration (device := targetDevice) cfg dtypes) := do
  let cast := dtype?.isSome
  log.onInfo s!"Loading Qwen3TTS weights from {modelDir}..."
  log.onInfo s!"Qwen3-TTS target device: {repr targetDevice}"
  let codecEmbedding ← (← safetensors.Source.open modelDir).load
    "talker.model.codec_embedding.weight"
    #[cfg.talkerConfig.vocabSize, cfg.talkerConfig.hiddenSize] targetDevice
  let talkerDType ← match dtype? with
    | some dtype => pure dtype
    | none => do
      let floating ← IO.ofExcept (DType.realFloating? codecEmbedding.1)
      let _ := floating.down
      pure (RealFloatingDType.ofDType codecEmbedding.1)
  let codecEmbedding ← admitDType (dtype := talkerDType) cast "talker.model.codec_embedding.weight" codecEmbedding
  let talker ← loadTalkerForConditionalGenerationSharded
    (device := targetDevice) (dtype := talkerDType) modelDir cast cfg.talkerConfig codecEmbedding

  if cfg.ttsModelType == "base" then
    let scfg := cfg.speakerEncoderConfig
    let firstWeight ← (← safetensors.Source.open modelDir).load
      "speaker_encoder.blocks.0.conv.weight"
      #[scfg.encChannels.getD 0 512, scfg.melDim, scfg.encKernelSizes.getD 0 5] targetDevice
    let speakerDType ← match dtype? with
      | some dtype => pure dtype
      | none => do
        let floating ← IO.ofExcept (DType.realFloating? firstWeight.1)
        let _ := floating.down
        pure (RealFloatingDType.ofDType firstWeight.1)
    let speakerEncoder ← loadSpeakerEncoderSharded
      (device := targetDevice) (dtype := speakerDType) modelDir cast scfg
    log.onInfo "Loaded Qwen3TTS weights (real checkpoint tensors)."
    return ⟨{ talker := talkerDType, speaker := speakerDType },
      { talker, speakerEncoder := some speakerEncoder }⟩
  else
    log.onInfo "Loaded Qwen3TTS weights (real checkpoint tensors)."
    return ⟨{ talker := talkerDType, speaker := talkerDType },
      { talker, speakerEncoder := none }⟩

end Qwen3TTSForConditionalGeneration

end torch.qwen3tts
