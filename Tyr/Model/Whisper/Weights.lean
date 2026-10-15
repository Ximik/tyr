/-
  Tyr/Model/Whisper/Weights.lean

  Pretrained weight loading for native Whisper model in Tyr.
-/
import Tyr.Model.Whisper.TypedOps
import Tyr.TensorStruct
import Tyr.Log
import Tyr.SafeTensors.Load
import Tyr.Model.Utils
import Tyr.Model.Whisper.Model

namespace torch.whisper

variable {device : Device}
open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (tryLoadOptionalTensorSharded)

private def loadLayerNormSharded (modelDir : String) (namePrefix : String) (dim : UInt64) (eps : Float)
    (device : Device := .CPU) : IO (LayerNorm (device := device) dim (.Unknown "dynamic")) := do
  let weight ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{namePrefix}.weight"] #[dim]
  let bias ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{namePrefix}.bias"] #[dim]
  pure { weight := reqGradFalse weight, bias := reqGradFalse bias, eps := ⟨eps⟩ }

private def loadAttentionSharded
    (modelDir : String)
    (namePrefix : String)
    (dModel nHeads : UInt64)
    (device : Device := .CPU) : IO (WhisperAttention (device := device) dModel nHeads) := do
  let qProjWeight ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{namePrefix}.q_proj.weight"] #[dModel, dModel]
  let kProjWeight ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{namePrefix}.k_proj.weight"] #[dModel, dModel]
  let vProjWeight ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{namePrefix}.v_proj.weight"] #[dModel, dModel]
  let outProjWeight ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{namePrefix}.out_proj.weight"] #[dModel, dModel]
  let qProjBias ←
    match (← tryLoadOptionalTensorSharded (device := device) modelDir s!"{namePrefix}.q_proj.bias" #[dModel]) with
    | some b => pure b
    | none => pure (Tensor.castLike qProjWeight (zerosDynamic (device := device) #[dModel]))
  let kProjBias ←
    match (← tryLoadOptionalTensorSharded (device := device) modelDir s!"{namePrefix}.k_proj.bias" #[dModel]) with
    | some b => pure b
    | none => pure (Tensor.castLike kProjWeight (zerosDynamic (device := device) #[dModel]))
  let vProjBias ←
    match (← tryLoadOptionalTensorSharded (device := device) modelDir s!"{namePrefix}.v_proj.bias" #[dModel]) with
    | some b => pure b
    | none => pure (Tensor.castLike vProjWeight (zerosDynamic (device := device) #[dModel]))
  let outProjBias ←
    match (← tryLoadOptionalTensorSharded (device := device) modelDir s!"{namePrefix}.out_proj.bias" #[dModel]) with
    | some b => pure b
    | none => pure (Tensor.castLike outProjWeight (zerosDynamic (device := device) #[dModel]))
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
    (device : Device := .CPU) : IO (WhisperEncoderLayer (device := device) cfg) := do
  let p := s!"model.encoder.layers.{layerIdx}"
  let selfAttn ← loadAttentionSharded (device := device) modelDir s!"{p}.self_attn" cfg.dModel cfg.encoderAttentionHeads
  let selfAttnLayerNorm ← loadLayerNormSharded (device := device) modelDir s!"{p}.self_attn_layer_norm" cfg.dModel cfg.layerNormEps
  let fc1Weight ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{p}.fc1.weight"] #[cfg.encoderFfnDim, cfg.dModel]
  let fc1Bias ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{p}.fc1.bias"] #[cfg.encoderFfnDim]
  let fc2Weight ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{p}.fc2.weight"] #[cfg.dModel, cfg.encoderFfnDim]
  let fc2Bias ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{p}.fc2.bias"] #[cfg.dModel]
  let finalLayerNorm ← loadLayerNormSharded (device := device) modelDir s!"{p}.final_layer_norm" cfg.dModel cfg.layerNormEps
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
    (device : Device := .CPU) : IO (WhisperDecoderLayer (device := device) cfg) := do
  let p := s!"model.decoder.layers.{layerIdx}"
  let selfAttn ← loadAttentionSharded (device := device) modelDir s!"{p}.self_attn" cfg.dModel cfg.decoderAttentionHeads
  let selfAttnLayerNorm ← loadLayerNormSharded (device := device) modelDir s!"{p}.self_attn_layer_norm" cfg.dModel cfg.layerNormEps
  let encoderAttn ← loadAttentionSharded (device := device) modelDir s!"{p}.encoder_attn" cfg.dModel cfg.decoderAttentionHeads
  let encoderAttnLayerNorm ← loadLayerNormSharded (device := device) modelDir s!"{p}.encoder_attn_layer_norm" cfg.dModel cfg.layerNormEps
  let fc1Weight ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{p}.fc1.weight"] #[cfg.decoderFfnDim, cfg.dModel]
  let fc1Bias ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{p}.fc1.bias"] #[cfg.decoderFfnDim]
  let fc2Weight ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{p}.fc2.weight"] #[cfg.dModel, cfg.decoderFfnDim]
  let fc2Bias ← safetensors.loadTensorShardedCandidates (device := device) modelDir #[s!"{p}.fc2.bias"] #[cfg.dModel]
  let finalLayerNorm ← loadLayerNormSharded (device := device) modelDir s!"{p}.final_layer_norm" cfg.dModel cfg.layerNormEps
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

private def loadModelSharded (modelDir : String) (cfg : WhisperConfig) (device : Device := .CPU) : IO (WhisperModel (device := device) cfg) := do
  let encoderConv1Weight ← safetensors.loadTensorShardedCandidates (device := device) modelDir #["model.encoder.conv1.weight"] #[cfg.dModel, cfg.numMelBins, 3]
  let encoderConv1Bias ← safetensors.loadTensorShardedCandidates (device := device) modelDir #["model.encoder.conv1.bias"] #[cfg.dModel]
  let encoderConv2Weight ← safetensors.loadTensorShardedCandidates (device := device) modelDir #["model.encoder.conv2.weight"] #[cfg.dModel, cfg.dModel, 3]
  let encoderConv2Bias ← safetensors.loadTensorShardedCandidates (device := device) modelDir #["model.encoder.conv2.bias"] #[cfg.dModel]
  let encoderPositionalEmbedding ←
    safetensors.loadTensorShardedCandidates (device := device) modelDir #["model.encoder.embed_positions.weight"] #[cfg.maxSourcePositions, cfg.dModel]

  let mut encoderLayers : Array (WhisperEncoderLayer (device := device) cfg) := #[]
  for i in [:cfg.encoderLayers.toNat] do
    encoderLayers := encoderLayers.push (← loadEncoderLayerSharded (device := device) modelDir i.toUInt64 cfg)
  let encoderLayerNorm ← loadLayerNormSharded (device := device) modelDir "model.encoder.layer_norm" cfg.dModel cfg.layerNormEps

  let decoderTokenEmbedding ←
    safetensors.loadTensorShardedCandidates (device := device) modelDir #["model.decoder.embed_tokens.weight"] #[cfg.vocabSize, cfg.dModel]
  let decoderPositionalEmbedding ←
    safetensors.loadTensorShardedCandidates (device := device) modelDir #["model.decoder.embed_positions.weight"] #[cfg.maxTargetPositions, cfg.dModel]

  let mut decoderLayers : Array (WhisperDecoderLayer (device := device) cfg) := #[]
  for i in [:cfg.decoderLayers.toNat] do
    decoderLayers := decoderLayers.push (← loadDecoderLayerSharded (device := device) modelDir i.toUInt64 cfg)
  let decoderLayerNorm ← loadLayerNormSharded (device := device) modelDir "model.decoder.layer_norm" cfg.dModel cfg.layerNormEps

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
      pure (Device.CUDA 0)
    else
      log.onWarn "TYR_DEVICE=cuda requested but CUDA is unavailable; falling back to auto."
      getBestDevice
  | some "mps" => pure Device.MPS
  | some "auto" => getBestDevice
  | some _ => getBestDevice
  | none => getBestDevice

def loadShardedAt (modelDir : String) (cfg : WhisperConfig := {}) (log : Handlers := {})
    (device : Device) : IO (WhisperForConditionalGeneration (device := device) cfg) := do
  log.onInfo s!"Loading Whisper weights from {modelDir}..."
  let model ← loadModelSharded (device := device) modelDir cfg
  let projOut ←
    match (← tryLoadOptionalTensorSharded (device := device) modelDir "proj_out.weight" #[cfg.vocabSize, cfg.dModel]) with
    | some w => pure (reqGradFalse w)
    | none =>
      log.onInfo "  proj_out.weight not found; tying output projection to decoder token embeddings."
      pure model.decoderTokenEmbedding
  log.onInfo s!"Whisper target device: {repr device}"
  pure { model, projOut }

/-- Load using the environment-selected device, retaining that index in the result. -/
def loadSharded (modelDir : String) (cfg : WhisperConfig := {}) (log : Handlers := {}) :
    IO (Σ device : Device, WhisperForConditionalGeneration (device := device) cfg) := do
  let device ← resolveWhisperDevice log
  let model ← loadShardedAt modelDir cfg log device
  pure ⟨device, model⟩

end WhisperForConditionalGeneration

end torch.whisper
