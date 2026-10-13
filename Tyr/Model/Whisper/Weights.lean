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

open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (tryLoadOptionalTensorSharded)

private def loadLayerNormSharded (modelDir : String) (namePrefix : String) (dim : UInt64) (eps : Float)
    : IO (LayerNorm dim (.Unknown "dynamic")) := do
  let weight ← safetensors.loadTensorShardedCandidates modelDir #[s!"{namePrefix}.weight"] #[dim]
  let bias ← safetensors.loadTensorShardedCandidates modelDir #[s!"{namePrefix}.bias"] #[dim]
  pure { weight := reqGradFalse weight, bias := reqGradFalse bias, eps := ⟨eps⟩ }

private def loadAttentionSharded
    (modelDir : String)
    (namePrefix : String)
    (dModel nHeads : UInt64)
    : IO (WhisperAttention dModel nHeads) := do
  let qProjWeight ← safetensors.loadTensorShardedCandidates modelDir #[s!"{namePrefix}.q_proj.weight"] #[dModel, dModel]
  let kProjWeight ← safetensors.loadTensorShardedCandidates modelDir #[s!"{namePrefix}.k_proj.weight"] #[dModel, dModel]
  let vProjWeight ← safetensors.loadTensorShardedCandidates modelDir #[s!"{namePrefix}.v_proj.weight"] #[dModel, dModel]
  let outProjWeight ← safetensors.loadTensorShardedCandidates modelDir #[s!"{namePrefix}.out_proj.weight"] #[dModel, dModel]
  let qProjBias ←
    match (← tryLoadOptionalTensorSharded modelDir s!"{namePrefix}.q_proj.bias" #[dModel]) with
    | some b => pure b
    | none => pure (Tensor.castLike qProjWeight (zerosDynamic #[dModel]))
  let kProjBias ←
    match (← tryLoadOptionalTensorSharded modelDir s!"{namePrefix}.k_proj.bias" #[dModel]) with
    | some b => pure b
    | none => pure (Tensor.castLike kProjWeight (zerosDynamic #[dModel]))
  let vProjBias ←
    match (← tryLoadOptionalTensorSharded modelDir s!"{namePrefix}.v_proj.bias" #[dModel]) with
    | some b => pure b
    | none => pure (Tensor.castLike vProjWeight (zerosDynamic #[dModel]))
  let outProjBias ←
    match (← tryLoadOptionalTensorSharded modelDir s!"{namePrefix}.out_proj.bias" #[dModel]) with
    | some b => pure b
    | none => pure (Tensor.castLike outProjWeight (zerosDynamic #[dModel]))
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
    : IO (WhisperEncoderLayer cfg) := do
  let p := s!"model.encoder.layers.{layerIdx}"
  let selfAttn ← loadAttentionSharded modelDir s!"{p}.self_attn" cfg.dModel cfg.encoderAttentionHeads
  let selfAttnLayerNorm ← loadLayerNormSharded modelDir s!"{p}.self_attn_layer_norm" cfg.dModel cfg.layerNormEps
  let fc1Weight ← safetensors.loadTensorShardedCandidates modelDir #[s!"{p}.fc1.weight"] #[cfg.encoderFfnDim, cfg.dModel]
  let fc1Bias ← safetensors.loadTensorShardedCandidates modelDir #[s!"{p}.fc1.bias"] #[cfg.encoderFfnDim]
  let fc2Weight ← safetensors.loadTensorShardedCandidates modelDir #[s!"{p}.fc2.weight"] #[cfg.dModel, cfg.encoderFfnDim]
  let fc2Bias ← safetensors.loadTensorShardedCandidates modelDir #[s!"{p}.fc2.bias"] #[cfg.dModel]
  let finalLayerNorm ← loadLayerNormSharded modelDir s!"{p}.final_layer_norm" cfg.dModel cfg.layerNormEps
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
    : IO (WhisperDecoderLayer cfg) := do
  let p := s!"model.decoder.layers.{layerIdx}"
  let selfAttn ← loadAttentionSharded modelDir s!"{p}.self_attn" cfg.dModel cfg.decoderAttentionHeads
  let selfAttnLayerNorm ← loadLayerNormSharded modelDir s!"{p}.self_attn_layer_norm" cfg.dModel cfg.layerNormEps
  let encoderAttn ← loadAttentionSharded modelDir s!"{p}.encoder_attn" cfg.dModel cfg.decoderAttentionHeads
  let encoderAttnLayerNorm ← loadLayerNormSharded modelDir s!"{p}.encoder_attn_layer_norm" cfg.dModel cfg.layerNormEps
  let fc1Weight ← safetensors.loadTensorShardedCandidates modelDir #[s!"{p}.fc1.weight"] #[cfg.decoderFfnDim, cfg.dModel]
  let fc1Bias ← safetensors.loadTensorShardedCandidates modelDir #[s!"{p}.fc1.bias"] #[cfg.decoderFfnDim]
  let fc2Weight ← safetensors.loadTensorShardedCandidates modelDir #[s!"{p}.fc2.weight"] #[cfg.dModel, cfg.decoderFfnDim]
  let fc2Bias ← safetensors.loadTensorShardedCandidates modelDir #[s!"{p}.fc2.bias"] #[cfg.dModel]
  let finalLayerNorm ← loadLayerNormSharded modelDir s!"{p}.final_layer_norm" cfg.dModel cfg.layerNormEps
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

private def loadModelSharded (modelDir : String) (cfg : WhisperConfig) : IO (WhisperModel cfg) := do
  let encoderConv1Weight ← safetensors.loadTensorShardedCandidates modelDir #["model.encoder.conv1.weight"] #[cfg.dModel, cfg.numMelBins, 3]
  let encoderConv1Bias ← safetensors.loadTensorShardedCandidates modelDir #["model.encoder.conv1.bias"] #[cfg.dModel]
  let encoderConv2Weight ← safetensors.loadTensorShardedCandidates modelDir #["model.encoder.conv2.weight"] #[cfg.dModel, cfg.dModel, 3]
  let encoderConv2Bias ← safetensors.loadTensorShardedCandidates modelDir #["model.encoder.conv2.bias"] #[cfg.dModel]
  let encoderPositionalEmbedding ←
    safetensors.loadTensorShardedCandidates modelDir #["model.encoder.embed_positions.weight"] #[cfg.maxSourcePositions, cfg.dModel]

  let mut encoderLayers : Array (WhisperEncoderLayer cfg) := #[]
  for i in [:cfg.encoderLayers.toNat] do
    encoderLayers := encoderLayers.push (← loadEncoderLayerSharded modelDir i.toUInt64 cfg)
  let encoderLayerNorm ← loadLayerNormSharded modelDir "model.encoder.layer_norm" cfg.dModel cfg.layerNormEps

  let decoderTokenEmbedding ←
    safetensors.loadTensorShardedCandidates modelDir #["model.decoder.embed_tokens.weight"] #[cfg.vocabSize, cfg.dModel]
  let decoderPositionalEmbedding ←
    safetensors.loadTensorShardedCandidates modelDir #["model.decoder.embed_positions.weight"] #[cfg.maxTargetPositions, cfg.dModel]

  let mut decoderLayers : Array (WhisperDecoderLayer cfg) := #[]
  for i in [:cfg.decoderLayers.toNat] do
    decoderLayers := decoderLayers.push (← loadDecoderLayerSharded modelDir i.toUInt64 cfg)
  let decoderLayerNorm ← loadLayerNormSharded modelDir "model.decoder.layer_norm" cfg.dModel cfg.layerNormEps

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

def loadSharded (modelDir : String) (cfg : WhisperConfig := {}) (log : Handlers := {})
    : IO (WhisperForConditionalGeneration cfg) := do
  log.onInfo s!"Loading Whisper weights from {modelDir}..."
  let model ← loadModelSharded modelDir cfg
  let projOut ←
    match (← tryLoadOptionalTensorSharded modelDir "proj_out.weight" #[cfg.vocabSize, cfg.dModel]) with
    | some w => pure (reqGradFalse w)
    | none =>
      log.onInfo "  proj_out.weight not found; tying output projection to decoder token embeddings."
      pure model.decoderTokenEmbedding
  let targetDevice ← resolveWhisperDevice log
  let model := TensorStruct.map (fun t => t.to targetDevice) model
  let projOut := projOut.to targetDevice
  log.onInfo s!"Whisper target device: {repr targetDevice}"
  pure { model, projOut }

end WhisperForConditionalGeneration

end torch.whisper
