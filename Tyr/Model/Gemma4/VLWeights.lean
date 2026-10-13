/-
  Tyr/Model/Gemma4/VLWeights.lean

  Pretrained weight loading for the Gemma 4 multimodal wrapper.
-/
import Tyr.Log
import Tyr.SafeTensors.Load
import Tyr.Model.Utils
import Tyr.Model.Gemma4.Weights
import Tyr.Model.Gemma4.Multimodal

namespace torch.gemma4

open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (pushUnique
  loadTensorCandidates loadTensorShardedCandidates
  tryLoadTensorCandidates tryLoadTensorShardedCandidates)

private def visionTensorNameCandidates (name : String) : Array String :=
  Id.run do
    let mut out : Array String := #[]
    out := pushUnique out name
    if name.startsWith "model." then
      out := pushUnique out (name.drop 6).toString
    else
      out := pushUnique out s!"model.{name}"
    out

private def visionParameterNameCandidates (name : String) : Array String :=
  Id.run do
    let mut out := visionTensorNameCandidates name
    if !name.endsWith ".weight" then
      for suffix in [".weight", ".linear.weight"] do
        for cand in visionTensorNameCandidates s!"{name}{suffix}" do
          out := pushUnique out cand
    out

private def loadVisionParameterSharded {s : Shape}
    (modelDir : String)
    (name : String)
    : IO (STensor s) :=
  loadTensorShardedCandidates modelDir (visionParameterNameCandidates name) s

private def loadVisionParameter {s : Shape}
    (path : String)
    (name : String)
    : IO (STensor s) :=
  loadTensorCandidates path (visionParameterNameCandidates name) s

private def tryLoadVisionScalarSharded (modelDir : String) (name : String) : IO (Option Float) := do
  let t? ← tryLoadTensorShardedCandidates modelDir (visionTensorNameCandidates name) #[]
  match t? with
  | some t =>
    let vals ← Tensor.toFloatArray t
    pure (vals[0]?)
  | none =>
    pure none

private def tryLoadVisionScalar (path : String) (name : String) : IO (Option Float) := do
  let t? ← tryLoadTensorCandidates path (visionTensorNameCandidates name) #[]
  match t? with
  | some t =>
    let vals ← Tensor.toFloatArray t
    pure (vals[0]?)
  | none =>
    pure none

private def tryLoadVisionParameterSharded {s : Shape}
    (modelDir : String)
    (name : String)
    : IO (Option (STensor s)) :=
  tryLoadTensorShardedCandidates modelDir (visionParameterNameCandidates name) s

private def tryLoadVisionParameter {s : Shape}
    (path : String)
    (name : String)
    : IO (Option (STensor s)) :=
  tryLoadTensorCandidates path (visionParameterNameCandidates name) s

private def loadVisionLinearSharded (modelDir : String) (name : String) (outDim inDim : UInt64)
    : IO (Gemma4VisionLinear outDim inDim) := do
  let w ← loadVisionParameterSharded (s := #[outDim, inDim]) modelDir name
  let inputMin ← tryLoadVisionScalarSharded modelDir s!"{name}.input_min"
  let inputMax ← tryLoadVisionScalarSharded modelDir s!"{name}.input_max"
  let outputMin ← tryLoadVisionScalarSharded modelDir s!"{name}.output_min"
  let outputMax ← tryLoadVisionScalarSharded modelDir s!"{name}.output_max"
  pure {
    weight := reqGradFalse w
    input_min := inputMin
    input_max := inputMax
    output_min := outputMin
    output_max := outputMax
  }

private def loadVisionLinear (path : String) (name : String) (outDim inDim : UInt64)
    : IO (Gemma4VisionLinear outDim inDim) := do
  let w ← loadVisionParameter (s := #[outDim, inDim]) path name
  let inputMin ← tryLoadVisionScalar path s!"{name}.input_min"
  let inputMax ← tryLoadVisionScalar path s!"{name}.input_max"
  let outputMin ← tryLoadVisionScalar path s!"{name}.output_min"
  let outputMax ← tryLoadVisionScalar path s!"{name}.output_max"
  pure {
    weight := reqGradFalse w
    input_min := inputMin
    input_max := inputMax
    output_min := outputMin
    output_max := outputMax
  }

private def loadVisionRMSNormSharded (modelDir : String) (name : String) (dim : UInt64) (eps : Float)
    : IO (Gemma4RMSNorm dim) := do
  let w ← loadVisionParameterSharded (s := #[dim]) modelDir name
  pure (Gemma4RMSNorm.fromCheckpointWeight (reqGradFalse w) eps)

private def loadVisionRMSNorm (path : String) (name : String) (dim : UInt64) (eps : Float)
    : IO (Gemma4RMSNorm dim) := do
  let w ← loadVisionParameter (s := #[dim]) path name
  pure (Gemma4RMSNorm.fromCheckpointWeight (reqGradFalse w) eps)

private def loadVisionAttentionSharded (modelDir : String) (cfg : VisionConfig) (layerIdx : UInt64)
    : IO (Gemma4VisionAttention cfg) := do
  let p := s!"model.vision_tower.encoder.layers.{layerIdx}.self_attn"
  pure {
    q_proj := (← loadVisionLinearSharded modelDir s!"{p}.q_proj" cfg.hidden_size cfg.hidden_size)
    k_proj := (← loadVisionLinearSharded modelDir s!"{p}.k_proj" cfg.hidden_size cfg.hidden_size)
    v_proj := (← loadVisionLinearSharded modelDir s!"{p}.v_proj" cfg.hidden_size cfg.hidden_size)
    o_proj := (← loadVisionLinearSharded modelDir s!"{p}.o_proj" cfg.hidden_size cfg.hidden_size)
    q_norm := (← loadVisionRMSNormSharded modelDir s!"{p}.q_norm" cfg.head_dim cfg.rms_norm_eps)
    k_norm := (← loadVisionRMSNormSharded modelDir s!"{p}.k_norm" cfg.head_dim cfg.rms_norm_eps)
  }

private def loadVisionAttention (path : String) (cfg : VisionConfig) (layerIdx : UInt64)
    : IO (Gemma4VisionAttention cfg) := do
  let p := s!"model.vision_tower.encoder.layers.{layerIdx}.self_attn"
  pure {
    q_proj := (← loadVisionLinear path s!"{p}.q_proj" cfg.hidden_size cfg.hidden_size)
    k_proj := (← loadVisionLinear path s!"{p}.k_proj" cfg.hidden_size cfg.hidden_size)
    v_proj := (← loadVisionLinear path s!"{p}.v_proj" cfg.hidden_size cfg.hidden_size)
    o_proj := (← loadVisionLinear path s!"{p}.o_proj" cfg.hidden_size cfg.hidden_size)
    q_norm := (← loadVisionRMSNorm path s!"{p}.q_norm" cfg.head_dim cfg.rms_norm_eps)
    k_norm := (← loadVisionRMSNorm path s!"{p}.k_norm" cfg.head_dim cfg.rms_norm_eps)
  }

private def loadVisionMLPSharded (modelDir : String) (cfg : VisionConfig) (layerIdx : UInt64)
    : IO (Gemma4VisionMLP cfg) := do
  let p := s!"model.vision_tower.encoder.layers.{layerIdx}.mlp"
  pure {
    gate_proj := (← loadVisionLinearSharded modelDir s!"{p}.gate_proj" cfg.intermediate_size cfg.hidden_size)
    up_proj := (← loadVisionLinearSharded modelDir s!"{p}.up_proj" cfg.intermediate_size cfg.hidden_size)
    down_proj := (← loadVisionLinearSharded modelDir s!"{p}.down_proj" cfg.hidden_size cfg.intermediate_size)
  }

private def loadVisionMLP (path : String) (cfg : VisionConfig) (layerIdx : UInt64)
    : IO (Gemma4VisionMLP cfg) := do
  let p := s!"model.vision_tower.encoder.layers.{layerIdx}.mlp"
  pure {
    gate_proj := (← loadVisionLinear path s!"{p}.gate_proj" cfg.intermediate_size cfg.hidden_size)
    up_proj := (← loadVisionLinear path s!"{p}.up_proj" cfg.intermediate_size cfg.hidden_size)
    down_proj := (← loadVisionLinear path s!"{p}.down_proj" cfg.hidden_size cfg.intermediate_size)
  }

private def loadVisionBlockSharded (modelDir : String) (cfg : VisionConfig) (layerIdx : UInt64)
    : IO (Gemma4VisionBlock cfg) := do
  let p := s!"model.vision_tower.encoder.layers.{layerIdx}"
  pure {
    input_layernorm := (← loadVisionRMSNormSharded modelDir s!"{p}.input_layernorm" cfg.hidden_size cfg.rms_norm_eps)
    post_attention_layernorm := (← loadVisionRMSNormSharded modelDir s!"{p}.post_attention_layernorm" cfg.hidden_size cfg.rms_norm_eps)
    pre_feedforward_layernorm := (← loadVisionRMSNormSharded modelDir s!"{p}.pre_feedforward_layernorm" cfg.hidden_size cfg.rms_norm_eps)
    post_feedforward_layernorm := (← loadVisionRMSNormSharded modelDir s!"{p}.post_feedforward_layernorm" cfg.hidden_size cfg.rms_norm_eps)
    self_attn := (← loadVisionAttentionSharded modelDir cfg layerIdx)
    mlp := (← loadVisionMLPSharded modelDir cfg layerIdx)
  }

private def loadVisionBlock (path : String) (cfg : VisionConfig) (layerIdx : UInt64)
    : IO (Gemma4VisionBlock cfg) := do
  let p := s!"model.vision_tower.encoder.layers.{layerIdx}"
  pure {
    input_layernorm := (← loadVisionRMSNorm path s!"{p}.input_layernorm" cfg.hidden_size cfg.rms_norm_eps)
    post_attention_layernorm := (← loadVisionRMSNorm path s!"{p}.post_attention_layernorm" cfg.hidden_size cfg.rms_norm_eps)
    pre_feedforward_layernorm := (← loadVisionRMSNorm path s!"{p}.pre_feedforward_layernorm" cfg.hidden_size cfg.rms_norm_eps)
    post_feedforward_layernorm := (← loadVisionRMSNorm path s!"{p}.post_feedforward_layernorm" cfg.hidden_size cfg.rms_norm_eps)
    self_attn := (← loadVisionAttention path cfg layerIdx)
    mlp := (← loadVisionMLP path cfg layerIdx)
  }

private def loadVisionPatchEmbedderSharded (modelDir : String) (cfg : VLConfig)
    : IO (Gemma4VisionPatchEmbedder cfg) := do
  let inputProj ←
    loadVisionLinearSharded
      modelDir
      "model.vision_tower.patch_embedder.input_proj"
      cfg.vision_config.hidden_size
      (VisionConfig.patchDim cfg.vision_config)
  let posEmbed ←
    loadTensorShardedCandidates modelDir
      (visionTensorNameCandidates "model.vision_tower.patch_embedder.position_embedding_table")
      #[2, cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size]
  pure {
    input_proj := inputProj
    position_embedding_table := reqGradFalse posEmbed
  }

private def loadVisionPatchEmbedder (path : String) (cfg : VLConfig)
    : IO (Gemma4VisionPatchEmbedder cfg) := do
  let inputProj ←
    loadVisionLinear
      path
      "model.vision_tower.patch_embedder.input_proj"
      cfg.vision_config.hidden_size
      (VisionConfig.patchDim cfg.vision_config)
  let posEmbed ←
    loadTensorCandidates path
      (visionTensorNameCandidates "model.vision_tower.patch_embedder.position_embedding_table")
      #[2, cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size]
  pure {
    input_proj := inputProj
    position_embedding_table := reqGradFalse posEmbed
  }

private def loadVisionModelSharded (modelDir : String) (cfg : VLConfig) : IO (Gemma4VisionModel cfg) := do
  let patchEmbedder ← loadVisionPatchEmbedderSharded modelDir cfg
  let mut blocks : Array (Gemma4VisionBlock cfg.vision_config) := #[]
  for i in [:cfg.vision_config.num_hidden_layers.toNat] do
    blocks := blocks.push (← loadVisionBlockSharded modelDir cfg.vision_config i.toUInt64)
  let stdBias ← tryLoadVisionParameterSharded (s := #[cfg.vision_config.hidden_size]) modelDir "model.vision_tower.std_bias"
  let stdScale ← tryLoadVisionParameterSharded (s := #[cfg.vision_config.hidden_size]) modelDir "model.vision_tower.std_scale"
  pure {
    patch_embedder := patchEmbedder
    blocks := blocks
    std_bias := stdBias.map reqGradFalse
    std_scale := stdScale.map reqGradFalse
  }

private def loadVisionModel (path : String) (cfg : VLConfig) : IO (Gemma4VisionModel cfg) := do
  let patchEmbedder ← loadVisionPatchEmbedder path cfg
  let mut blocks : Array (Gemma4VisionBlock cfg.vision_config) := #[]
  for i in [:cfg.vision_config.num_hidden_layers.toNat] do
    blocks := blocks.push (← loadVisionBlock path cfg.vision_config i.toUInt64)
  let stdBias ← tryLoadVisionParameter (s := #[cfg.vision_config.hidden_size]) path "model.vision_tower.std_bias"
  let stdScale ← tryLoadVisionParameter (s := #[cfg.vision_config.hidden_size]) path "model.vision_tower.std_scale"
  pure {
    patch_embedder := patchEmbedder
    blocks := blocks
    std_bias := stdBias.map reqGradFalse
    std_scale := stdScale.map reqGradFalse
  }

private def loadVisionProjectorSharded (modelDir : String) (cfg : VLConfig)
    : IO (Gemma4MultimodalEmbedder cfg) := do
  let embedProj ←
    loadVisionParameterSharded
      (s := #[cfg.text_config.hidden_size, cfg.vision_config.hidden_size])
      modelDir
      "model.embed_vision.embedding_projection"
  pure { embedding_projection := reqGradFalse embedProj }

private def loadVisionProjector (path : String) (cfg : VLConfig)
    : IO (Gemma4MultimodalEmbedder cfg) := do
  let embedProj ←
    loadVisionParameter
      (s := #[cfg.text_config.hidden_size, cfg.vision_config.hidden_size])
      path
      "model.embed_vision.embedding_projection"
  pure { embedding_projection := reqGradFalse embedProj }

namespace Gemma4ForConditionalGeneration

/-- Load Gemma 4 multimodal model from a sharded HF SafeTensors directory. -/
def loadSharded (modelDir : String) (cfg : VLConfig := {})
    (log : Handlers := {})
    : IO (Gemma4ForConditionalGeneration cfg) := do
  log.onInfo s!"Loading Gemma4ForConditionalGeneration from {modelDir}..."
  let visionTower ← loadVisionModelSharded modelDir cfg
  let visionProjector ← loadVisionProjectorSharded modelDir cfg
  let languageModel ← Gemma4ForCausalLM.loadSharded modelDir cfg.text_config log
  log.onInfo "Loaded Gemma4ForConditionalGeneration weights."
  pure {
    vision_tower := visionTower
    embed_vision := visionProjector
    language_model := languageModel
  }

/-- Load Gemma 4 multimodal model from a single HF SafeTensors file. -/
def load (path : String) (cfg : VLConfig := {})
    (log : Handlers := {})
    : IO (Gemma4ForConditionalGeneration cfg) := do
  log.onInfo s!"Loading Gemma4ForConditionalGeneration from {path}..."
  let visionTower ← loadVisionModel path cfg
  let visionProjector ← loadVisionProjector path cfg
  let languageModel ← Gemma4ForCausalLM.load path cfg.text_config log
  log.onInfo "Loaded Gemma4ForConditionalGeneration weights."
  pure {
    vision_tower := visionTower
    embed_vision := visionProjector
    language_model := languageModel
  }

end Gemma4ForConditionalGeneration

end torch.gemma4
