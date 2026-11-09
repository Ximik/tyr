/-
  Tyr/Model/Gemma4/VLWeights.lean

  Pretrained weight loading for the Gemma 4 multimodal wrapper.
-/
import Tyr.Log
import Tyr.SafeTensors.Source
import Tyr.Model.Utils
import Tyr.Model.Gemma4.Weights
import Tyr.Model.Gemma4.Multimodal

namespace torch.gemma4

variable {device : Device} {dtype : DType}

open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (pushUnique Source)

private def freezeStored {shape : Shape} (tensor : SomeDTypeTensor shape device) :
    SomeDTypeTensor shape device := ⟨tensor.1, tensor.2.detach⟩

private def requireLinear {outDim inDim : Nat} (dtype : DType)
    (layer : Σ stored : DType, Gemma4VisionLinear (device := device) outDim inDim stored) :
    IO (Gemma4VisionLinear (device := device) outDim inDim dtype) :=
  if h : layer.1 = dtype then pure (h ▸ layer.2)
  else throw (IO.userError s!"Gemma4 vision linear expected {dtype}, got {layer.1}")

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

private def loadVisionParameter {s : Shape}
    (src : Source)
    (name : String)
    : IO (SomeDTypeTensor (s) device) := do
  src.load (← src.resolve (visionParameterNameCandidates name)) s device

private def tryLoadVisionScalar (src : Source) (name : String) : IO (Option Float) := do
  let t? ← ((src.find? (visionTensorNameCandidates name)).mapM (src.load · #[] .CPU))
  match t? with
  | some t =>
    let vals ← Tensor.toFloatArray t.2
    pure (vals[0]?)
  | none =>
    pure none

private def tryLoadVisionParameter {s : Shape}
    (src : Source)
    (name : String)
    : IO (Option (SomeDTypeTensor (s) device)) :=
  ((src.find? (visionParameterNameCandidates name)).mapM (src.load · s device))

private def loadVisionLinear (src : Source) (name : String) (outDim inDim : Nat)
    : IO (Σ dtype : DType, Gemma4VisionLinear (device := device) outDim inDim dtype) := do
  let w ← loadVisionParameter (s := #[outDim, inDim]) src name
  let inputMin ← tryLoadVisionScalar src s!"{name}.input_min"
  let inputMax ← tryLoadVisionScalar src s!"{name}.input_max"
  let outputMin ← tryLoadVisionScalar src s!"{name}.output_min"
  let outputMax ← tryLoadVisionScalar src s!"{name}.output_max"
  pure ⟨w.1, {
    weight := reqGradFalse w.2
    input_min := inputMin
    input_max := inputMax
    output_min := outputMin
    output_max := outputMax
  }⟩

private def loadVisionRMSNorm (src : Source) (name : String) (dim : Nat) (eps : Float)
    : IO (Gemma4RMSNorm (device := device) dim) := do
  let w ← loadVisionParameter (s := #[dim]) src name
  pure (Gemma4RMSNorm.fromCheckpointWeight (freezeStored w) eps)

private def loadVisionAttention (src : Source) (cfg : VisionConfig) (layerIdx : Nat)
    : IO (Gemma4VisionAttention (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.vision_tower.encoder.layers.{layerIdx}.self_attn"
  pure {
    q_proj := (← requireLinear dtype (← loadVisionLinear src s!"{p}.q_proj" cfg.hidden_size cfg.hidden_size))
    k_proj := (← requireLinear dtype (← loadVisionLinear src s!"{p}.k_proj" cfg.hidden_size cfg.hidden_size))
    v_proj := (← requireLinear dtype (← loadVisionLinear src s!"{p}.v_proj" cfg.hidden_size cfg.hidden_size))
    o_proj := (← requireLinear dtype (← loadVisionLinear src s!"{p}.o_proj" cfg.hidden_size cfg.hidden_size))
    q_norm := (← loadVisionRMSNorm src s!"{p}.q_norm" cfg.head_dim cfg.rms_norm_eps)
    k_norm := (← loadVisionRMSNorm src s!"{p}.k_norm" cfg.head_dim cfg.rms_norm_eps)
  }

private def loadVisionMLP (src : Source) (cfg : VisionConfig) (layerIdx : Nat)
    : IO (Gemma4VisionMLP (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.vision_tower.encoder.layers.{layerIdx}.mlp"
  pure {
    gate_proj := (← requireLinear dtype (← loadVisionLinear src s!"{p}.gate_proj" cfg.intermediate_size cfg.hidden_size))
    up_proj := (← requireLinear dtype (← loadVisionLinear src s!"{p}.up_proj" cfg.intermediate_size cfg.hidden_size))
    down_proj := (← requireLinear dtype (← loadVisionLinear src s!"{p}.down_proj" cfg.hidden_size cfg.intermediate_size))
  }

private def loadVisionBlock (src : Source) (cfg : VisionConfig) (layerIdx : Nat)
    : IO (Gemma4VisionBlock (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.vision_tower.encoder.layers.{layerIdx}"
  pure {
    input_layernorm := (← loadVisionRMSNorm src s!"{p}.input_layernorm" cfg.hidden_size cfg.rms_norm_eps)
    post_attention_layernorm := (← loadVisionRMSNorm src s!"{p}.post_attention_layernorm" cfg.hidden_size cfg.rms_norm_eps)
    pre_feedforward_layernorm := (← loadVisionRMSNorm src s!"{p}.pre_feedforward_layernorm" cfg.hidden_size cfg.rms_norm_eps)
    post_feedforward_layernorm := (← loadVisionRMSNorm src s!"{p}.post_feedforward_layernorm" cfg.hidden_size cfg.rms_norm_eps)
    self_attn := (← loadVisionAttention src cfg layerIdx)
    mlp := (← loadVisionMLP src cfg layerIdx)
  }

private def loadVisionPatchEmbedder (src : Source) (cfg : VLConfig)
    : IO (Σ dtype : DType, Gemma4VisionPatchEmbedder (device := device) cfg dtype) := do
  let inputProj ←
    loadVisionLinear
      src
      "model.vision_tower.patch_embedder.input_proj"
      cfg.vision_config.hidden_size
      (VisionConfig.patchDim cfg.vision_config)
  let posEmbed ←
    src.load (← src.resolve (visionTensorNameCandidates "model.vision_tower.patch_embedder.position_embedding_table")) #[2, cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size] device
  pure ⟨inputProj.1, {
    input_proj := inputProj.2
    position_embedding_table := reqGradFalse posEmbed.2.toFloat32
  }⟩

private def loadVisionModel (src : Source) (cfg : VLConfig) : IO (Σ dtype : RealFloatingDType, Gemma4VisionModel (device := device) cfg dtype.val) := do
  let patchEmbedder ← loadVisionPatchEmbedder src cfg
  let dtype := patchEmbedder.1
  let floating ← IO.ofExcept dtype.realFloating?
  let _ := floating.down
  let mut blocks : Array (Gemma4VisionBlock (dtype := dtype) (device := device) cfg.vision_config) := #[]
  for i in [:cfg.vision_config.num_hidden_layers] do
    blocks := blocks.push (← loadVisionBlock src cfg.vision_config i)
  let stdBias ← tryLoadVisionParameter (s := #[cfg.vision_config.hidden_size]) src "model.vision_tower.std_bias"
  let stdScale ← tryLoadVisionParameter (s := #[cfg.vision_config.hidden_size]) src "model.vision_tower.std_scale"
  pure ⟨RealFloatingDType.ofDType dtype, {
    patch_embedder := patchEmbedder.2
    blocks := blocks
    std_bias := stdBias.map (reqGradFalse ·.2.toFloat32)
    std_scale := stdScale.map (reqGradFalse ·.2.toFloat32)
  }⟩

private def loadVisionProjector (src : Source) (cfg : VLConfig)
    : IO (Σ dtype : RealFloatingDType, Gemma4MultimodalEmbedder (device := device) cfg dtype.val) := do
  let embedProj ←
    loadVisionParameter
      (s := #[cfg.text_config.hidden_size, cfg.vision_config.hidden_size])
      src
      "model.embed_vision.embedding_projection"
  let floating ← IO.ofExcept embedProj.1.realFloating?
  let _ := floating.down
  pure ⟨RealFloatingDType.ofDType embedProj.1, { embedding_projection := reqGradFalse embedProj.2 }⟩

namespace Gemma4ForConditionalGeneration

/-- Load Gemma 4 multimodal model from a SafeTensors file or a directory of shards. -/
def load (path : String) (cfg : VLConfig := {})
    (log : Handlers := {}) (device : Device := .CPU)
    : IO (Σ textDType : RealFloatingDType, Σ visionDType : RealFloatingDType, Σ projectorDType : RealFloatingDType,
      Gemma4ForConditionalGeneration (device := device) cfg textDType.val visionDType.val projectorDType.val) := do
  log.onInfo s!"Loading Gemma4ForConditionalGeneration (device := device) from {path}..."
  let src ← Source.open path
  let visionTower ← loadVisionModel src cfg
  let visionProjector ← loadVisionProjector src cfg
  let languageModel ← Gemma4ForCausalLM.load (device := device) path cfg.text_config log
  log.onInfo "Loaded Gemma4ForConditionalGeneration (device := device) weights."
  pure ⟨languageModel.1, visionTower.1, visionProjector.1, {
    vision_tower := visionTower.2
    embed_vision := visionProjector.2
    language_model := languageModel.2
  }⟩

end Gemma4ForConditionalGeneration

end torch.gemma4
