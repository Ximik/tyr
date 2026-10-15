/-
  Tyr/Model/Qwen3TTS/Weights.lean

  Pretrained weight loading for the Lean Qwen3-TTS port.
  Loads talker and code-predictor weights from HuggingFace-style sharded
  SafeTensors directories so generation can use real checkpoint tensors.
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Log
import Tyr.SafeTensors.Exact
import Tyr.Module.RMSNorm
import Tyr.Model.Utils
import Tyr.Model.Qwen3TTS.Model

namespace torch.qwen3tts

variable {device : Device} {dtype : ExactDType}

open torch.Log

open torch.Model (reqGradFalse)


/-- Load a stored exact dtype and check the shared component index without casting. -/
private def loadTensorShardedTarget (modelDir : String) (name : String) (s : Shape) :
    IO (Tensor { shape := s, dtype := dtype.val, device }) := do
  let packed ← safetensors.loadTensorShardedExact modelDir name s device
  match packed.atDType? dtype with
  | .ok tensor => pure tensor
  | .error message => throw <| IO.userError s!"Qwen3TTS tensor '{name}': {message}"

/-- Only absent optional entries are suppressed; every present tensor must match. -/
private def loadOptionalTensorShardedTarget (modelDir : String) (name : String) (s : Shape) :
    IO (Option (Tensor { shape := s, dtype := dtype.val, device })) := do
  match ← safetensors.loadOptionalTensorShardedExact modelDir name s device with
  | none => pure none
  | some packed =>
    match packed.atDType? dtype with
    | .ok tensor => pure (some tensor)
    | .error message => throw <| IO.userError s!"Qwen3TTS tensor '{name}': {message}"

private def loadRMSNormSharded (modelDir : String) (name : String) (dim : UInt64) (eps : Float)
    : IO (RMSNorm (device := device) dim dtype.val) := do
  let w ← loadTensorShardedTarget modelDir s!"{name}.weight" #[dim]
  pure { weight := reqGradFalse w, eps := ⟨eps⟩ }

private def loadTalkerLayerSharded (modelDir : String) (layerIdx : UInt64) (cfg : TalkerConfig)
    : IO (TalkerLayer (device := device) (dtype := dtype) cfg) := do
  let p := s!"talker.model.layers.{layerIdx}"
  let sp := s!"{p}.self_attn"
  let mp := s!"{p}.mlp"

  let q_proj ← loadTensorShardedTarget modelDir s!"{sp}.q_proj.weight"
    #[cfg.numAttentionHeads * cfg.headDim, cfg.hiddenSize]
  let k_proj ← loadTensorShardedTarget modelDir s!"{sp}.k_proj.weight"
    #[cfg.numKeyValueHeads * cfg.headDim, cfg.hiddenSize]
  let v_proj ← loadTensorShardedTarget modelDir s!"{sp}.v_proj.weight"
    #[cfg.numKeyValueHeads * cfg.headDim, cfg.hiddenSize]
  let o_proj ← loadTensorShardedTarget modelDir s!"{sp}.o_proj.weight"
    #[cfg.hiddenSize, cfg.numAttentionHeads * cfg.headDim]

  let q_norm ← loadOptionalTensorShardedTarget modelDir s!"{sp}.q_norm.weight" #[cfg.headDim]
  let k_norm ← loadOptionalTensorShardedTarget modelDir s!"{sp}.k_norm.weight" #[cfg.headDim]

  let gate_proj ← loadTensorShardedTarget modelDir s!"{mp}.gate_proj.weight"
    #[cfg.intermediateSize, cfg.hiddenSize]
  let up_proj ← loadTensorShardedTarget modelDir s!"{mp}.up_proj.weight"
    #[cfg.intermediateSize, cfg.hiddenSize]
  let down_proj ← loadTensorShardedTarget modelDir s!"{mp}.down_proj.weight"
    #[cfg.hiddenSize, cfg.intermediateSize]

  let input_layernorm ← loadRMSNormSharded modelDir s!"{p}.input_layernorm" cfg.hiddenSize cfg.rmsNormEps
  let post_attention_layernorm ← loadRMSNormSharded modelDir s!"{p}.post_attention_layernorm" cfg.hiddenSize cfg.rmsNormEps

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

private def loadCodePredictorLayerSharded (modelDir : String) (layerIdx : UInt64) (cfg : TalkerConfig)
    : IO (CodePredictorLayer (device := device) (dtype := dtype) cfg.codePredictorConfig) := do
  let cp := cfg.codePredictorConfig
  let p := s!"talker.code_predictor.model.layers.{layerIdx}"
  let sp := s!"{p}.self_attn"
  let mp := s!"{p}.mlp"

  let q_proj ← loadTensorShardedTarget modelDir s!"{sp}.q_proj.weight"
    #[cp.numAttentionHeads * cp.headDim, cp.hiddenSize]
  let k_proj ← loadTensorShardedTarget modelDir s!"{sp}.k_proj.weight"
    #[cp.numKeyValueHeads * cp.headDim, cp.hiddenSize]
  let v_proj ← loadTensorShardedTarget modelDir s!"{sp}.v_proj.weight"
    #[cp.numKeyValueHeads * cp.headDim, cp.hiddenSize]
  let o_proj ← loadTensorShardedTarget modelDir s!"{sp}.o_proj.weight"
    #[cp.hiddenSize, cp.numAttentionHeads * cp.headDim]

  let q_norm ← loadOptionalTensorShardedTarget modelDir s!"{sp}.q_norm.weight" #[cp.headDim]
  let k_norm ← loadOptionalTensorShardedTarget modelDir s!"{sp}.k_norm.weight" #[cp.headDim]

  let gate_proj ← loadTensorShardedTarget modelDir s!"{mp}.gate_proj.weight"
    #[cp.intermediateSize, cp.hiddenSize]
  let up_proj ← loadTensorShardedTarget modelDir s!"{mp}.up_proj.weight"
    #[cp.intermediateSize, cp.hiddenSize]
  let down_proj ← loadTensorShardedTarget modelDir s!"{mp}.down_proj.weight"
    #[cp.hiddenSize, cp.intermediateSize]

  let input_layernorm ← loadRMSNormSharded modelDir s!"{p}.input_layernorm" cp.hiddenSize cp.rmsNormEps
  let post_attention_layernorm ← loadRMSNormSharded modelDir s!"{p}.post_attention_layernorm" cp.hiddenSize cp.rmsNormEps

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

private def loadCodePredictorProjectionSharded (modelDir : String) (cfg : TalkerConfig)
    : IO (Tensor { shape := #[cfg.codePredictorConfig.hiddenSize, cfg.hiddenSize], dtype := dtype.val, device := device } × Tensor { shape := #[cfg.codePredictorConfig.hiddenSize], dtype := dtype.val, device := device }) := do
  let cp := cfg.codePredictorConfig
  let weight ← loadOptionalTensorShardedTarget (dtype := dtype) modelDir
    "talker.code_predictor.small_to_mtp_projection.weight" #[cp.hiddenSize, cfg.hiddenSize]
  let bias ← loadOptionalTensorShardedTarget (dtype := dtype) modelDir
    "talker.code_predictor.small_to_mtp_projection.bias" #[cp.hiddenSize]
  match weight, bias with
  | some w, some b => pure (reqGradFalse w, reqGradFalse b)
  | none, none =>
    if cp.hiddenSize == cfg.hiddenSize then
      let w ← Tensor.eyeForIO (n := cp.hiddenSize) dtype.val device
      let b ← Tensor.zerosForIO { shape := #[cp.hiddenSize], dtype := dtype.val, device }
      pure (w.reshapeChecked #[cfg.codePredictorConfig.hiddenSize, cfg.hiddenSize], b)
    else
      throw <| IO.userError
        "Missing talker.code_predictor.small_to_mtp_projection.* and hidden sizes differ; cannot build projection."
  | _, _ =>
    throw <| IO.userError
      "Incomplete talker.code_predictor.small_to_mtp_projection: weight and bias must both be present or both absent."

private def loadTalkerModelSharded (modelDir : String) (cfg : TalkerConfig)
    (codecEmbedding : Tensor { shape := #[cfg.vocabSize, cfg.hiddenSize], dtype := dtype.val, device })
    : IO (TalkerModel (device := device) (dtype := dtype) cfg) := do
  let textEmbedding ← loadTensorShardedTarget modelDir "talker.model.text_embedding.weight"
    #[cfg.textVocabSize, cfg.textHiddenSize]

  let textProjectionFc1 ← loadTensorShardedTarget modelDir "talker.text_projection.linear_fc1.weight"
    #[cfg.textHiddenSize, cfg.textHiddenSize]
  let textProjectionFc1Bias ← loadTensorShardedTarget modelDir "talker.text_projection.linear_fc1.bias"
    #[cfg.textHiddenSize]
  let textProjectionFc2 ← loadTensorShardedTarget modelDir "talker.text_projection.linear_fc2.weight"
    #[cfg.hiddenSize, cfg.textHiddenSize]
  let textProjectionFc2Bias ← loadTensorShardedTarget modelDir "talker.text_projection.linear_fc2.bias"
    #[cfg.hiddenSize]

  let mut layers : Array (TalkerLayer (device := device) (dtype := dtype) cfg) := #[]
  for i in [:cfg.numHiddenLayers.toNat] do
    let layer ← loadTalkerLayerSharded modelDir i.toUInt64 cfg
    layers := layers.push layer

  let norm ← loadRMSNormSharded modelDir "talker.model.norm" cfg.hiddenSize cfg.rmsNormEps
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

private def loadCodePredictorSharded (modelDir : String) (cfg : TalkerConfig)
    : IO (TalkerCodePredictor (device := device) (dtype := dtype) cfg) := do
  let cp := cfg.codePredictorConfig
  let nCodebooks := if cfg.numCodeGroups.toNat == 0 then 0 else cfg.numCodeGroups.toNat - 1

  let mut inputEmbeddings : Array (Tensor { shape := #[cp.vocabSize, cfg.hiddenSize], dtype := dtype.val, device := device }) := #[]
  for i in [:nCodebooks] do
    let emb ← loadTensorShardedTarget modelDir
      s!"talker.code_predictor.model.codec_embedding.{i}.weight"
      #[cp.vocabSize, cfg.hiddenSize]
    inputEmbeddings := inputEmbeddings.push (reqGradFalse emb)

  let (inputProjection, inputProjectionBias) ← loadCodePredictorProjectionSharded modelDir cfg

  let mut layers : Array (CodePredictorLayer (device := device) (dtype := dtype) cp) := #[]
  for i in [:cp.numHiddenLayers.toNat] do
    let layer ← loadCodePredictorLayerSharded modelDir i.toUInt64 cfg
    layers := layers.push layer

  let norm ← loadRMSNormSharded modelDir "talker.code_predictor.model.norm" cp.hiddenSize cp.rmsNormEps

  let mut lmHeads : Array (Tensor { shape := #[cp.vocabSize, cp.hiddenSize], dtype := dtype.val, device := device }) := #[]
  for i in [:nCodebooks] do
    let head ← loadTensorShardedTarget modelDir
      s!"talker.code_predictor.lm_head.{i}.weight"
      #[cp.vocabSize, cp.hiddenSize]
    lmHeads := lmHeads.push (reqGradFalse head)

  pure { inputEmbeddings, inputProjection, inputProjectionBias, layers, norm, lmHeads }

private def loadTalkerForConditionalGenerationSharded (modelDir : String) (cfg : TalkerConfig)
    (codecEmbedding : Tensor { shape := #[cfg.vocabSize, cfg.hiddenSize], dtype := dtype.val, device })
    : IO (TalkerForConditionalGeneration (device := device) (dtype := dtype) cfg) := do
  let model ← loadTalkerModelSharded modelDir cfg codecEmbedding
  let codePredictor ← loadCodePredictorSharded modelDir cfg
  let codecHead ← loadTensorShardedTarget modelDir "talker.codec_head.weight"
    #[cfg.vocabSize, cfg.hiddenSize]
  pure { model, codePredictor, codecHead := reqGradFalse codecHead }

private def loadTimeDelayNetBlockSharded (modelDir : String) (namePrefix : String)
    (inChannels outChannels kernelSize dilation : UInt64)
    : IO (TimeDelayNetBlock (device := device) (dtype := dtype) inChannels outChannels kernelSize dilation) := do
  let weight ← loadTensorShardedTarget modelDir s!"{namePrefix}.conv.weight"
    #[outChannels, inChannels, kernelSize]
  let bias ← loadTensorShardedTarget modelDir s!"{namePrefix}.conv.bias"
    #[outChannels]
  pure {
    weight := reqGradFalse weight
    bias := reqGradFalse bias
  }

private def loadSqueezeExcitationBlockSharded (modelDir : String) (namePrefix : String)
    (channels seChannels : UInt64)
    : IO (SqueezeExcitationBlock (device := device) (dtype := dtype) channels seChannels) := do
  let conv1Weight ← loadTensorShardedTarget modelDir s!"{namePrefix}.conv1.weight"
    #[seChannels, channels, 1]
  let conv1Bias ← loadTensorShardedTarget modelDir s!"{namePrefix}.conv1.bias"
    #[seChannels]
  let conv2Weight ← loadTensorShardedTarget modelDir s!"{namePrefix}.conv2.weight"
    #[channels, seChannels, 1]
  let conv2Bias ← loadTensorShardedTarget modelDir s!"{namePrefix}.conv2.bias"
    #[channels]
  pure {
    conv1Weight := reqGradFalse conv1Weight
    conv1Bias := reqGradFalse conv1Bias
    conv2Weight := reqGradFalse conv2Weight
    conv2Bias := reqGradFalse conv2Bias
  }

private def loadRes2NetBlockSharded (modelDir : String) (namePrefix : String)
    (channels scale kernelSize dilation : UInt64)
    : IO (Res2NetBlock (device := device) (dtype := dtype) channels scale kernelSize dilation) := do
  let nBlocks := if scale.toNat <= 1 then 0 else scale.toNat - 1
  let mut blocks : Array (TimeDelayNetBlock (device := device) (dtype := dtype) (channels / scale) (channels / scale) kernelSize dilation) := #[]
  for i in [:nBlocks] do
    let block ← loadTimeDelayNetBlockSharded modelDir
      s!"{namePrefix}.blocks.{i}"
      (channels / scale) (channels / scale) kernelSize dilation
    blocks := blocks.push block
  pure { blocks }

private def loadSqueezeExcitationRes2NetBlockSharded (modelDir : String) (namePrefix : String)
    (inChannels outChannels res2netScale seChannels kernelSize dilation : UInt64)
    : IO (SqueezeExcitationRes2NetBlock (device := device) (dtype := dtype) inChannels outChannels res2netScale seChannels kernelSize dilation) := do
  let tdnn1 ← loadTimeDelayNetBlockSharded modelDir
    s!"{namePrefix}.tdnn1" inChannels outChannels 1 1
  let res2net ← loadRes2NetBlockSharded modelDir
    s!"{namePrefix}.res2net_block" outChannels res2netScale kernelSize dilation
  let tdnn2 ← loadTimeDelayNetBlockSharded modelDir
    s!"{namePrefix}.tdnn2" outChannels outChannels 1 1
  let seBlock ← loadSqueezeExcitationBlockSharded modelDir
    s!"{namePrefix}.se_block" outChannels seChannels
  pure { tdnn1, res2net, tdnn2, seBlock }

private def loadAttentiveStatisticsPoolingSharded (modelDir : String) (namePrefix : String)
    (channels attentionChannels : UInt64)
    : IO (AttentiveStatisticsPooling (device := device) (dtype := dtype) channels attentionChannels) := do
  let tdnn ← loadTimeDelayNetBlockSharded modelDir
    s!"{namePrefix}.tdnn" (channels * 3) attentionChannels 1 1
  let convWeight ← loadTensorShardedTarget modelDir s!"{namePrefix}.conv.weight"
    #[channels, attentionChannels, 1]
  let convBias ← loadTensorShardedTarget modelDir s!"{namePrefix}.conv.bias"
    #[channels]
  pure {
    tdnn
    convWeight := reqGradFalse convWeight
    convBias := reqGradFalse convBias
    eps := 1e-12
  }

private def loadSpeakerEncoderSharded (modelDir : String) (cfg : SpeakerEncoderConfig)
    : IO (SpeakerEncoder (device := device) (dtype := dtype) cfg) := do
  if cfg.encChannels.size < 5 then
    throw <| IO.userError "speaker encoder requires at least 5 enc_channels entries"
  if cfg.encKernelSizes.size < 5 then
    throw <| IO.userError "speaker encoder requires at least 5 enc_kernel_sizes entries"
  if cfg.encDilations.size < 5 then
    throw <| IO.userError "speaker encoder requires at least 5 enc_dilations entries"
  if cfg.encRes2NetScale < 2 then
    throw <| IO.userError "speaker encoder requires enc_res2net_scale >= 2"

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

  if c0 != c1 || c1 != c2 || c2 != c3 then
    throw <| IO.userError "speaker encoder expects enc_channels[0..3] to match for residual SE-Res2Net blocks"
  if c4 != mfaIn then
    throw <| IO.userError "speaker encoder expects enc_channels[4] = enc_channels[1] + enc_channels[2] + enc_channels[3]"

  let tdnn0 ← loadTimeDelayNetBlockSharded modelDir
    "speaker_encoder.blocks.0" cfg.melDim c0 k0 d0
  let block1 ← loadSqueezeExcitationRes2NetBlockSharded modelDir
    "speaker_encoder.blocks.1" c0 c1 cfg.encRes2NetScale cfg.encSeChannels k1 d1
  let block2 ← loadSqueezeExcitationRes2NetBlockSharded modelDir
    "speaker_encoder.blocks.2" c1 c2 cfg.encRes2NetScale cfg.encSeChannels k2 d2
  let block3 ← loadSqueezeExcitationRes2NetBlockSharded modelDir
    "speaker_encoder.blocks.3" c2 c3 cfg.encRes2NetScale cfg.encSeChannels k3 d3
  let mfa ← loadTimeDelayNetBlockSharded modelDir
    "speaker_encoder.mfa" mfaIn c4 k4 d4
  let asp ← loadAttentiveStatisticsPoolingSharded modelDir
    "speaker_encoder.asp" c4 cfg.encAttentionChannels
  let fcWeight ← loadTensorShardedTarget modelDir "speaker_encoder.fc.weight"
    #[cfg.encDim, c4 * 2, 1]
  let fcBias ← loadTensorShardedTarget modelDir "speaker_encoder.fc.bias"
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
    for base models, the ECAPA speaker encoder as well. -/
def loadSharded
    (modelDir : String)
    (cfg : Qwen3TTSConfig := {})
    (targetDevice : Device := Device.CPU)
    (log : Handlers := {})
    : IO (Σ dtypes : DTypeLayout, Qwen3TTSForConditionalGeneration (device := targetDevice) cfg dtypes) := do
  log.onInfo s!"Loading Qwen3TTS weights from {modelDir}..."
  log.onInfo s!"Qwen3-TTS target device: {repr targetDevice}"
  let codecEmbedding ← safetensors.loadTensorShardedExact modelDir
    "talker.model.codec_embedding.weight"
    #[cfg.talkerConfig.vocabSize, cfg.talkerConfig.hiddenSize] targetDevice
  let talker ← loadTalkerForConditionalGenerationSharded
    (device := targetDevice) (dtype := codecEmbedding.1) modelDir cfg.talkerConfig codecEmbedding.2

  if cfg.ttsModelType == "base" then
    let scfg := cfg.speakerEncoderConfig
    let firstWeight ← safetensors.loadTensorShardedExact modelDir
      "speaker_encoder.blocks.0.conv.weight"
      #[scfg.encChannels.getD 0 512, scfg.melDim, scfg.encKernelSizes.getD 0 5] targetDevice
    let speakerEncoder ← loadSpeakerEncoderSharded
      (device := targetDevice) (dtype := firstWeight.1) modelDir scfg
    log.onInfo "Loaded Qwen3TTS weights (real checkpoint tensors)."
    return ⟨{ talker := codecEmbedding.1, speaker := firstWeight.1 },
      { talker, speakerEncoder := some speakerEncoder }⟩
  else
    log.onInfo "Loaded Qwen3TTS weights (real checkpoint tensors)."
    return ⟨{ talker := codecEmbedding.1, speaker := codecEmbedding.1 },
      { talker, speakerEncoder := none }⟩

end Qwen3TTSForConditionalGeneration

end torch.qwen3tts
