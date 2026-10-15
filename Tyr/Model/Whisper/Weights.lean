/-
  Tyr/Model/Whisper/Weights.lean

  Pretrained weight loading for native Whisper model in Tyr.
-/
import Tyr.Model.Whisper.TypedOps
import Tyr.TensorStruct
import Tyr.Log
import Tyr.SafeTensors.Exact
import Tyr.Model.Utils
import Tyr.Model.Whisper.Model

namespace torch.whisper

variable {device : Device}
open torch.Log
open torch.Model (reqGradFalse)
private def loadWeight (dtype : ExactDType) (modelDir name : String) (shape : Shape)
    (device : Device) : IO (Tensor { shape, dtype := dtype.val, device }) := do
  let packed ← safetensors.loadTensorShardedExact modelDir name shape device
  IO.ofExcept (packed.atDType? dtype)

private def loadOptional (dtype : ExactDType) (modelDir name : String) (shape : Shape)
    (device : Device) : IO (Option (Tensor { shape, dtype := dtype.val, device })) := do
  match ← safetensors.loadOptionalTensorShardedExact modelDir name shape device with
  | none => pure none
  | some packed => pure (some (← IO.ofExcept (packed.atDType? dtype)))

private def loadLayerNormSharded (modelDir : String) (namePrefix : String) (dim : UInt64) (eps : Float)
    (dtype : ExactDType) (device : Device := .CPU) : IO (LayerNorm (device := device) dim dtype.val) := do
  let weight ← loadWeight (device := device) dtype modelDir s!"{namePrefix}.weight" #[dim]
  let bias ← loadWeight (device := device) dtype modelDir s!"{namePrefix}.bias" #[dim]
  pure { weight := reqGradFalse weight, bias := reqGradFalse bias, eps := ⟨eps⟩ }

private def loadAttentionSharded
    (modelDir : String)
    (namePrefix : String)
    (dModel nHeads : UInt64)
    (dtype : ExactDType) (device : Device := .CPU) : IO (WhisperAttention (device := device) (dtype := dtype.val) dModel nHeads) := do
  let support ← dtype.val.acquireConstantSupport device
  let _ := support.down
  let qProjWeight ← loadWeight (device := device) dtype modelDir s!"{namePrefix}.q_proj.weight" #[dModel, dModel]
  let kProjWeight ← loadWeight (device := device) dtype modelDir s!"{namePrefix}.k_proj.weight" #[dModel, dModel]
  let vProjWeight ← loadWeight (device := device) dtype modelDir s!"{namePrefix}.v_proj.weight" #[dModel, dModel]
  let outProjWeight ← loadWeight (device := device) dtype modelDir s!"{namePrefix}.out_proj.weight" #[dModel, dModel]
  let qProjBias ←
    match (← loadOptional (device := device) dtype modelDir s!"{namePrefix}.q_proj.bias" #[dModel]) with
    | some b => pure b
    | none => pure (Tensor.castLike qProjWeight (zerosFor (dtype := dtype.val) (device := device) #[dModel]))
  let kProjBias ←
    match (← loadOptional (device := device) dtype modelDir s!"{namePrefix}.k_proj.bias" #[dModel]) with
    | some b => pure b
    | none => pure (Tensor.castLike kProjWeight (zerosFor (dtype := dtype.val) (device := device) #[dModel]))
  let vProjBias ←
    match (← loadOptional (device := device) dtype modelDir s!"{namePrefix}.v_proj.bias" #[dModel]) with
    | some b => pure b
    | none => pure (Tensor.castLike vProjWeight (zerosFor (dtype := dtype.val) (device := device) #[dModel]))
  let outProjBias ←
    match (← loadOptional (device := device) dtype modelDir s!"{namePrefix}.out_proj.bias" #[dModel]) with
    | some b => pure b
    | none => pure (Tensor.castLike outProjWeight (zerosFor (dtype := dtype.val) (device := device) #[dModel]))
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

private def loadEncoderLayerSharded (modelDir : String) (layerIdx : UInt64) (cfg : WhisperConfig)
    (dtype : ExactDType) (device : Device := .CPU) : IO (WhisperEncoderLayer (device := device) (dtype := dtype.val) cfg) := do
  let p := s!"model.encoder.layers.{layerIdx}"
  let selfAttn ← loadAttentionSharded (dtype := dtype) (device := device) modelDir s!"{p}.self_attn" cfg.dModel cfg.encoderAttentionHeads
  let selfAttnLayerNorm ← loadLayerNormSharded (dtype := dtype) (device := device) modelDir s!"{p}.self_attn_layer_norm" cfg.dModel cfg.layerNormEps
  let fc1Weight ← loadWeight (device := device) dtype modelDir s!"{p}.fc1.weight" #[cfg.encoderFfnDim, cfg.dModel]
  let fc1Bias ← loadWeight (device := device) dtype modelDir s!"{p}.fc1.bias" #[cfg.encoderFfnDim]
  let fc2Weight ← loadWeight (device := device) dtype modelDir s!"{p}.fc2.weight" #[cfg.dModel, cfg.encoderFfnDim]
  let fc2Bias ← loadWeight (device := device) dtype modelDir s!"{p}.fc2.bias" #[cfg.dModel]
  let finalLayerNorm ← loadLayerNormSharded (dtype := dtype) (device := device) modelDir s!"{p}.final_layer_norm" cfg.dModel cfg.layerNormEps
  pure {
    selfAttn
    selfAttnLayerNorm
    fc1Weight := reqGradFalse fc1Weight
    fc1Bias := reqGradFalse fc1Bias
    fc2Weight := reqGradFalse fc2Weight
    fc2Bias := reqGradFalse fc2Bias
    finalLayerNorm
  }

private def loadDecoderLayerSharded (modelDir : String) (layerIdx : UInt64) (cfg : WhisperConfig)
    (dtype : ExactDType) (device : Device := .CPU) : IO (WhisperDecoderLayer (device := device) (dtype := dtype.val) cfg) := do
  let p := s!"model.decoder.layers.{layerIdx}"
  let selfAttn ← loadAttentionSharded (dtype := dtype) (device := device) modelDir s!"{p}.self_attn" cfg.dModel cfg.decoderAttentionHeads
  let selfAttnLayerNorm ← loadLayerNormSharded (dtype := dtype) (device := device) modelDir s!"{p}.self_attn_layer_norm" cfg.dModel cfg.layerNormEps
  let encoderAttn ← loadAttentionSharded (dtype := dtype) (device := device) modelDir s!"{p}.encoder_attn" cfg.dModel cfg.decoderAttentionHeads
  let encoderAttnLayerNorm ← loadLayerNormSharded (dtype := dtype) (device := device) modelDir s!"{p}.encoder_attn_layer_norm" cfg.dModel cfg.layerNormEps
  let fc1Weight ← loadWeight (device := device) dtype modelDir s!"{p}.fc1.weight" #[cfg.decoderFfnDim, cfg.dModel]
  let fc1Bias ← loadWeight (device := device) dtype modelDir s!"{p}.fc1.bias" #[cfg.decoderFfnDim]
  let fc2Weight ← loadWeight (device := device) dtype modelDir s!"{p}.fc2.weight" #[cfg.dModel, cfg.decoderFfnDim]
  let fc2Bias ← loadWeight (device := device) dtype modelDir s!"{p}.fc2.bias" #[cfg.dModel]
  let finalLayerNorm ← loadLayerNormSharded (dtype := dtype) (device := device) modelDir s!"{p}.final_layer_norm" cfg.dModel cfg.layerNormEps
  pure {
    selfAttn
    selfAttnLayerNorm
    encoderAttn
    encoderAttnLayerNorm
    fc1Weight := reqGradFalse fc1Weight
    fc1Bias := reqGradFalse fc1Bias
    fc2Weight := reqGradFalse fc2Weight
    fc2Bias := reqGradFalse fc2Bias
    finalLayerNorm
  }

private def loadModelSharded (modelDir : String) (cfg : WhisperConfig) (dtype : ExactDType) (device : Device)
    (encoderConv1Weight : Tensor { shape := #[cfg.dModel, cfg.numMelBins, 3], dtype := dtype.val, device }) :
    IO (WhisperModel (device := device) (dtype := dtype.val) cfg) := do
  let encoderConv1Bias ← loadWeight (device := device) dtype modelDir "model.encoder.conv1.bias" #[cfg.dModel]
  let encoderConv2Weight ← loadWeight (device := device) dtype modelDir "model.encoder.conv2.weight" #[cfg.dModel, cfg.dModel, 3]
  let encoderConv2Bias ← loadWeight (device := device) dtype modelDir "model.encoder.conv2.bias" #[cfg.dModel]
  let encoderPositionalEmbedding ←
    loadWeight (device := device) dtype modelDir "model.encoder.embed_positions.weight" #[cfg.maxSourcePositions, cfg.dModel]

  let mut encoderLayers : Array (WhisperEncoderLayer (device := device) (dtype := dtype.val) cfg) := #[]
  for i in [:cfg.encoderLayers.toNat] do
    encoderLayers := encoderLayers.push (← loadEncoderLayerSharded (dtype := dtype) (device := device) modelDir i.toUInt64 cfg)
  let encoderLayerNorm ← loadLayerNormSharded (dtype := dtype) (device := device) modelDir "model.encoder.layer_norm" cfg.dModel cfg.layerNormEps

  let decoderTokenEmbedding ←
    loadWeight (device := device) dtype modelDir "model.decoder.embed_tokens.weight" #[cfg.vocabSize, cfg.dModel]
  let decoderPositionalEmbedding ←
    loadWeight (device := device) dtype modelDir "model.decoder.embed_positions.weight" #[cfg.maxTargetPositions, cfg.dModel]

  let mut decoderLayers : Array (WhisperDecoderLayer (device := device) (dtype := dtype.val) cfg) := #[]
  for i in [:cfg.decoderLayers.toNat] do
    decoderLayers := decoderLayers.push (← loadDecoderLayerSharded (dtype := dtype) (device := device) modelDir i.toUInt64 cfg)
  let decoderLayerNorm ← loadLayerNormSharded (dtype := dtype) (device := device) modelDir "model.decoder.layer_norm" cfg.dModel cfg.layerNormEps

  pure {
    encoderConv1Weight := reqGradFalse encoderConv1Weight
    encoderConv1Bias := reqGradFalse encoderConv1Bias
    encoderConv2Weight := reqGradFalse encoderConv2Weight
    encoderConv2Bias := reqGradFalse encoderConv2Bias
    encoderPositionalEmbedding := reqGradFalse encoderPositionalEmbedding
    encoderLayers
    encoderLayerNorm
    decoderTokenEmbedding := reqGradFalse decoderTokenEmbedding
    decoderPositionalEmbedding := reqGradFalse decoderPositionalEmbedding
    decoderLayers
    decoderLayerNorm
  }

namespace WhisperForConditionalGeneration

private def resolveWhisperDevice (log : Handlers := {}) : IO Device := do
  let requested := (← IO.getEnv "TYR_DEVICE").map String.toLower
  match requested with
  | some "cpu" => pure Device.CPU
  | some "cuda" =>
    if ← cuda_is_available then
      Device.cuda 0
    else
      log.onWarn "TYR_DEVICE=cuda requested but CUDA is unavailable; falling back to auto."
      getBestDevice
  | some "mps" => Device.mps
  | some "auto" => getBestDevice
  | some _ => getBestDevice
  | none => getBestDevice

def loadShardedAt (modelDir : String) (cfg : WhisperConfig := {}) (log : Handlers := {})
    (device : Device) : IO (Σ dtype : RealFloatingDType, WhisperForConditionalGeneration (device := device) (dtype := dtype.val) cfg) := do
  log.onInfo s!"Loading Whisper weights from {modelDir}..."
  let ⟨dtype, encoderConv1Weight⟩ ← safetensors.loadTensorShardedExact modelDir
    "model.encoder.conv1.weight" #[cfg.dModel, cfg.numMelBins, 3] device
  let floating ← IO.ofExcept (DType.realFloating? dtype.val)
  let _ := floating.down
  let model ← loadModelSharded (dtype := dtype) (device := device) modelDir cfg encoderConv1Weight
  let projOut ←
    match (← loadOptional (device := device) dtype modelDir "proj_out.weight" #[cfg.vocabSize, cfg.dModel]) with
    | some w => pure (reqGradFalse w)
    | none =>
      log.onInfo "  proj_out.weight not found; tying output projection to decoder token embeddings."
      pure model.decoderTokenEmbedding
  log.onInfo s!"Whisper target device: {repr device}"
  pure ⟨RealFloatingDType.ofDType dtype.val, { model, projOut }⟩

/-- Load using the environment-selected device, retaining that index in the result. -/
def loadSharded (modelDir : String) (cfg : WhisperConfig := {}) (log : Handlers := {}) :
    IO (Σ device : Device, Σ dtype : RealFloatingDType, WhisperForConditionalGeneration (device := device) (dtype := dtype.val) cfg) := do
  let device ← resolveWhisperDevice log
  let model ← loadShardedAt modelDir cfg log device
  pure ⟨device, model⟩

end WhisperForConditionalGeneration

end torch.whisper
