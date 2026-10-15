/-
  Tyr/Model/Qwen35/VLWeights.lean

  Pretrained weight loading for Qwen3.5 multimodal wrapper.
-/
import Tyr.Typed.Neural
import Tyr.Log
import Tyr.SafeTensors.Load
import Tyr.Model.Utils
import Tyr.Model.Qwen35.Weights
import Tyr.Model.Qwen35.Multimodal

namespace torch.qwen35

variable {device : Device}

open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (tryLoadOptionalTensor tryLoadOptionalTensorSharded
  loadTensorCandidates loadTensorShardedCandidates)

private def freezeStored {shape : Shape} (tensor : SomeDTypeTensor shape device) :
    SomeDTypeTensor shape device := ⟨tensor.1, tensor.2.detach⟩

private def requireDType {shape : Shape} (dtype : ExactDType)
    (tensor : SomeDTypeTensor shape device) : IO (Tensor ⟨shape, dtype.val, device⟩) :=
  IO.ofExcept (tensor.atDType? dtype)

private def visionNameCandidates (name : String) : Array String :=
  let out : Array String := #[name]
  let out :=
    if name.startsWith "model.visual." then
      out.push s!"visual.{name.drop 13}"
    else if name.startsWith "visual." then
      out.push s!"model.{name}"
    else
      out
  out

private def loadVisionLayerNormSharded (modelDir : String) (base : String) (dim : UInt64)
    (device : Device := Device.CPU)
    : IO (Qwen35VisionLayerNorm (device := device) dim) := do
  let w ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{base}.weight") #[dim] device
  let b ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{base}.bias") #[dim] device
  pure { weight := freezeStored w, bias := freezeStored b, eps := 1e-6 }

private def loadVisionLayerNorm (path : String) (base : String) (dim : UInt64)
    (device : Device := Device.CPU)
    : IO (Qwen35VisionLayerNorm (device := device) dim) := do
  let w ← loadTensorCandidates path (visionNameCandidates s!"{base}.weight") #[dim] device
  let b ← loadTensorCandidates path (visionNameCandidates s!"{base}.bias") #[dim] device
  pure { weight := freezeStored w, bias := freezeStored b, eps := 1e-6 }

private def loadVisionMLPSharded (modelDir : String) (cfg : VisionConfig) (layerIdx : UInt64)
    (device : Device := Device.CPU) (dtype : ExactDType)
    : IO (Qwen35VisionMLP (dtype := dtype.val) (device := device) cfg) := do
  let p := s!"model.visual.blocks.{layerIdx}.mlp"
  let w1 ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.linear_fc1.weight") #[cfg.intermediate_size, cfg.hidden_size] device
  let b1 ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.linear_fc1.bias") #[cfg.intermediate_size] device
  let w2 ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.linear_fc2.weight") #[cfg.hidden_size, cfg.intermediate_size] device
  let b2 ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.linear_fc2.bias") #[cfg.hidden_size] device
  pure {
    linear_fc1_weight := reqGradFalse (← requireDType dtype w1)
    linear_fc1_bias := reqGradFalse (← requireDType dtype b1)
    linear_fc2_weight := reqGradFalse (← requireDType dtype w2)
    linear_fc2_bias := reqGradFalse (← requireDType dtype b2)
  }

private def loadVisionMLP (path : String) (cfg : VisionConfig) (layerIdx : UInt64)
    (device : Device := Device.CPU) (dtype : ExactDType)
    : IO (Qwen35VisionMLP (dtype := dtype.val) (device := device) cfg) := do
  let p := s!"model.visual.blocks.{layerIdx}.mlp"
  let w1 ← loadTensorCandidates path (visionNameCandidates s!"{p}.linear_fc1.weight") #[cfg.intermediate_size, cfg.hidden_size] device
  let b1 ← loadTensorCandidates path (visionNameCandidates s!"{p}.linear_fc1.bias") #[cfg.intermediate_size] device
  let w2 ← loadTensorCandidates path (visionNameCandidates s!"{p}.linear_fc2.weight") #[cfg.hidden_size, cfg.intermediate_size] device
  let b2 ← loadTensorCandidates path (visionNameCandidates s!"{p}.linear_fc2.bias") #[cfg.hidden_size] device
  pure {
    linear_fc1_weight := reqGradFalse (← requireDType dtype w1)
    linear_fc1_bias := reqGradFalse (← requireDType dtype b1)
    linear_fc2_weight := reqGradFalse (← requireDType dtype w2)
    linear_fc2_bias := reqGradFalse (← requireDType dtype b2)
  }

private def loadVisionAttentionSharded (modelDir : String) (cfg : VisionConfig) (layerIdx : UInt64)
    (device : Device := Device.CPU) (dtype : ExactDType)
    : IO (Qwen35VisionAttention (dtype := dtype.val) (device := device) cfg) := do
  let p := s!"model.visual.blocks.{layerIdx}.attn"
  let qkvW ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.qkv.weight") #[cfg.hidden_size * 3, cfg.hidden_size] device
  let qkvB ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.qkv.bias") #[cfg.hidden_size * 3] device
  let projW ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.proj.weight") #[cfg.hidden_size, cfg.hidden_size] device
  let projB ← loadTensorShardedCandidates modelDir (visionNameCandidates s!"{p}.proj.bias") #[cfg.hidden_size] device
  pure {
    qkv_weight := reqGradFalse (← requireDType dtype qkvW)
    qkv_bias := reqGradFalse (← requireDType dtype qkvB)
    proj_weight := reqGradFalse (← requireDType dtype projW)
    proj_bias := reqGradFalse (← requireDType dtype projB)
  }

private def loadVisionAttention (path : String) (cfg : VisionConfig) (layerIdx : UInt64)
    (device : Device := Device.CPU) (dtype : ExactDType)
    : IO (Qwen35VisionAttention (dtype := dtype.val) (device := device) cfg) := do
  let p := s!"model.visual.blocks.{layerIdx}.attn"
  let qkvW ← loadTensorCandidates path (visionNameCandidates s!"{p}.qkv.weight") #[cfg.hidden_size * 3, cfg.hidden_size] device
  let qkvB ← loadTensorCandidates path (visionNameCandidates s!"{p}.qkv.bias") #[cfg.hidden_size * 3] device
  let projW ← loadTensorCandidates path (visionNameCandidates s!"{p}.proj.weight") #[cfg.hidden_size, cfg.hidden_size] device
  let projB ← loadTensorCandidates path (visionNameCandidates s!"{p}.proj.bias") #[cfg.hidden_size] device
  pure {
    qkv_weight := reqGradFalse (← requireDType dtype qkvW)
    qkv_bias := reqGradFalse (← requireDType dtype qkvB)
    proj_weight := reqGradFalse (← requireDType dtype projW)
    proj_bias := reqGradFalse (← requireDType dtype projB)
  }

private def loadVisionBlockSharded (modelDir : String) (cfg : VisionConfig) (layerIdx : UInt64)
    (device : Device := Device.CPU) (dtype : ExactDType)
    : IO (Qwen35VisionBlock (dtype := dtype.val) (device := device) cfg) := do
  let norm1 ← loadVisionLayerNormSharded modelDir s!"model.visual.blocks.{layerIdx}.norm1" cfg.hidden_size device
  let norm2 ← loadVisionLayerNormSharded modelDir s!"model.visual.blocks.{layerIdx}.norm2" cfg.hidden_size device
  let attn ← loadVisionAttentionSharded modelDir cfg layerIdx device dtype
  let mlp ← loadVisionMLPSharded modelDir cfg layerIdx device dtype
  pure { norm1 := norm1, norm2 := norm2, attn := attn, mlp := mlp }

private def loadVisionBlock (path : String) (cfg : VisionConfig) (layerIdx : UInt64)
    (device : Device := Device.CPU) (dtype : ExactDType)
    : IO (Qwen35VisionBlock (dtype := dtype.val) (device := device) cfg) := do
  let norm1 ← loadVisionLayerNorm path s!"model.visual.blocks.{layerIdx}.norm1" cfg.hidden_size device
  let norm2 ← loadVisionLayerNorm path s!"model.visual.blocks.{layerIdx}.norm2" cfg.hidden_size device
  let attn ← loadVisionAttention path cfg layerIdx device dtype
  let mlp ← loadVisionMLP path cfg layerIdx device dtype
  pure { norm1 := norm1, norm2 := norm2, attn := attn, mlp := mlp }

private def loadVisionModelSharded (modelDir : String) (cfg : VisionConfig)
    (device : Device := Device.CPU)
    : IO (Σ dtype : RealFloatingDType, Qwen35VisionModel (device := device) cfg dtype.val) := do
  let convW ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.patch_embed.proj.weight")
    #[cfg.hidden_size, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size]
    device
  let convB ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.patch_embed.proj.bias")
    #[cfg.hidden_size]
    device
  let dtype := convW.1
  let floating ← IO.ofExcept dtype.val.realFloating?
  let _ := floating.down
  let patchW : Tensor { device := device, shape := #[cfg.hidden_size, VisionConfig.patchDim cfg], dtype := dtype.val } :=
    Tensor.reshapeChecked convW.2 #[cfg.hidden_size, VisionConfig.patchDim cfg]
  let patchEmbed : Qwen35VisionPatchEmbed (dtype := dtype.val) (device := device) cfg := {
    weight := reqGradFalse patchW
    bias := reqGradFalse (← requireDType dtype convB)
  }

  let posEmbed ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.pos_embed.weight")
    #[cfg.num_position_embeddings, cfg.hidden_size]
    device

  let mut blocks : Array (Qwen35VisionBlock (dtype := dtype.val) (device := device) cfg) := #[]
  for i in [:cfg.depth.toNat] do
    blocks := blocks.push (← loadVisionBlockSharded modelDir cfg i.toUInt64 device dtype)

  let mergeUnit := VisionConfig.mergeUnit cfg
  let mergedHidden := cfg.hidden_size * mergeUnit
  let mergerNorm ← loadVisionLayerNormSharded modelDir "model.visual.merger.norm" cfg.hidden_size device
  let fc1W ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.merger.linear_fc1.weight")
    #[mergedHidden, mergedHidden]
    device
  let fc1B ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.merger.linear_fc1.bias")
    #[mergedHidden]
    device
  let fc2W ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.merger.linear_fc2.weight")
    #[cfg.out_hidden_size, mergedHidden]
    device
  let fc2B ← loadTensorShardedCandidates
    modelDir
    (visionNameCandidates "model.visual.merger.linear_fc2.bias")
    #[cfg.out_hidden_size]
    device

  let merger : Qwen35VisionPatchMerger (dtype := dtype.val) (device := device) cfg := {
    norm := mergerNorm
    linear_fc1_weight := reqGradFalse (← requireDType dtype fc1W)
    linear_fc1_bias := reqGradFalse (← requireDType dtype fc1B)
    linear_fc2_weight := reqGradFalse (← requireDType dtype fc2W)
    linear_fc2_bias := reqGradFalse (← requireDType dtype fc2B)
  }

  pure ⟨RealFloatingDType.ofDType dtype.val, {
    patch_embed := patchEmbed
    pos_embed := reqGradFalse (← requireDType dtype posEmbed)
    blocks := blocks
    merger := merger
  }⟩

private def loadVisionModel (path : String) (cfg : VisionConfig)
    (device : Device := Device.CPU)
    : IO (Σ dtype : RealFloatingDType, Qwen35VisionModel (device := device) cfg dtype.val) := do
  let convW ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.patch_embed.proj.weight")
    #[cfg.hidden_size, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size]
    device
  let convB ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.patch_embed.proj.bias")
    #[cfg.hidden_size]
    device
  let dtype := convW.1
  let floating ← IO.ofExcept dtype.val.realFloating?
  let _ := floating.down
  let patchW : Tensor { device := device, shape := #[cfg.hidden_size, VisionConfig.patchDim cfg], dtype := dtype.val } :=
    Tensor.reshapeChecked convW.2 #[cfg.hidden_size, VisionConfig.patchDim cfg]
  let patchEmbed : Qwen35VisionPatchEmbed (dtype := dtype.val) (device := device) cfg := {
    weight := reqGradFalse patchW
    bias := reqGradFalse (← requireDType dtype convB)
  }

  let posEmbed ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.pos_embed.weight")
    #[cfg.num_position_embeddings, cfg.hidden_size]
    device

  let mut blocks : Array (Qwen35VisionBlock (dtype := dtype.val) (device := device) cfg) := #[]
  for i in [:cfg.depth.toNat] do
    blocks := blocks.push (← loadVisionBlock path cfg i.toUInt64 device dtype)

  let mergeUnit := VisionConfig.mergeUnit cfg
  let mergedHidden := cfg.hidden_size * mergeUnit
  let mergerNorm ← loadVisionLayerNorm path "model.visual.merger.norm" cfg.hidden_size device
  let fc1W ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.merger.linear_fc1.weight")
    #[mergedHidden, mergedHidden]
    device
  let fc1B ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.merger.linear_fc1.bias")
    #[mergedHidden]
    device
  let fc2W ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.merger.linear_fc2.weight")
    #[cfg.out_hidden_size, mergedHidden]
    device
  let fc2B ← loadTensorCandidates
    path
    (visionNameCandidates "model.visual.merger.linear_fc2.bias")
    #[cfg.out_hidden_size]
    device

  let merger : Qwen35VisionPatchMerger (dtype := dtype.val) (device := device) cfg := {
    norm := mergerNorm
    linear_fc1_weight := reqGradFalse (← requireDType dtype fc1W)
    linear_fc1_bias := reqGradFalse (← requireDType dtype fc1B)
    linear_fc2_weight := reqGradFalse (← requireDType dtype fc2W)
    linear_fc2_bias := reqGradFalse (← requireDType dtype fc2B)
  }

  pure ⟨RealFloatingDType.ofDType dtype.val, {
    patch_embed := patchEmbed
    pos_embed := reqGradFalse (← requireDType dtype posEmbed)
    blocks := blocks
    merger := merger
  }⟩

namespace Qwen35ForConditionalGeneration

/-- Load Qwen3.5 multimodal model from sharded HF SafeTensors directory. -/
def loadSharded (modelDir : String) (cfg : VLConfig := {})
    (device : Device := Device.CPU)
    (log : Handlers := {})
    : IO (Σ dtype : RealFloatingDType, Σ visionDType : RealFloatingDType,
      Qwen35ForConditionalGeneration (device := device) cfg dtype.val visionDType.val) := do
  log.onInfo s!"Loading Qwen35ForConditionalGeneration (device := device) from {modelDir}..."
  if cfg.vision_config.out_hidden_size != cfg.text_config.hidden_size then
    throw <| IO.userError
      s!"vision out_hidden_size ({cfg.vision_config.out_hidden_size}) must match text hidden_size ({cfg.text_config.hidden_size})"

  let visual ← loadVisionModelSharded modelDir cfg.vision_config device
  let languageModel ← Qwen35ForCausalLM.loadSharded modelDir cfg.text_config device log
  log.onInfo "Loaded Qwen35ForConditionalGeneration (device := device) weights."
  pure ⟨languageModel.1, visual.1, { visual := visual.2, language_model := languageModel.2 }⟩

/-- Load Qwen3.5 multimodal model from a single HF SafeTensors file. -/
def load (path : String) (cfg : VLConfig := {})
    (device : Device := Device.CPU)
    (log : Handlers := {})
    : IO (Σ dtype : RealFloatingDType, Σ visionDType : RealFloatingDType,
      Qwen35ForConditionalGeneration (device := device) cfg dtype.val visionDType.val) := do
  log.onInfo s!"Loading Qwen35ForConditionalGeneration (device := device) from {path}..."
  if cfg.vision_config.out_hidden_size != cfg.text_config.hidden_size then
    throw <| IO.userError
      s!"vision out_hidden_size ({cfg.vision_config.out_hidden_size}) must match text hidden_size ({cfg.text_config.hidden_size})"

  let visual ← loadVisionModel path cfg.vision_config device
  let languageModel ← Qwen35ForCausalLM.load path cfg.text_config device log
  log.onInfo "Loaded Qwen35ForConditionalGeneration (device := device) weights."
  pure ⟨languageModel.1, visual.1, { visual := visual.2, language_model := languageModel.2 }⟩

end Qwen35ForConditionalGeneration

end torch.qwen35
