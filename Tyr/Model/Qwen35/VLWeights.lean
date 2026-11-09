/-
  Tyr/Model/Qwen35/VLWeights.lean

  Pretrained weight loading for Qwen3.5 multimodal wrapper.
-/
import Tyr.Typed.Neural
import Tyr.Log
import Tyr.SafeTensors.Source
import Tyr.Model.Utils
import Tyr.Model.Qwen35.Weights
import Tyr.Model.Qwen35.Multimodal

namespace torch.qwen35

variable {device : Device}

open torch.Log
open torch.Model (reqGradFalse)
open torch.safetensors (Source)

private def freezeStored {shape : Shape} (tensor : SomeDTypeTensor shape device) :
    SomeDTypeTensor shape device := ⟨tensor.1, tensor.2.detach⟩

private def requireDType {shape : Shape} (dtype : DType)
    (tensor : SomeDTypeTensor shape device) : IO (Tensor ⟨shape, dtype, device⟩) :=
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

private def loadVisionLayerNorm (src : Source) (base : String) (dim : Nat)
    (device : Device := Device.CPU)
    : IO (Qwen35VisionLayerNorm (device := device) dim) := do
  let w ← src.load (← src.resolve (visionNameCandidates s!"{base}.weight")) #[dim] device
  let b ← src.load (← src.resolve (visionNameCandidates s!"{base}.bias")) #[dim] device
  pure { weight := freezeStored w, bias := freezeStored b, eps := 1e-6 }

private def loadVisionMLP (src : Source) (cfg : VisionConfig) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35VisionMLP (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.visual.blocks.{layerIdx}.mlp"
  let w1 ← src.load (← src.resolve (visionNameCandidates s!"{p}.linear_fc1.weight")) #[cfg.intermediate_size, cfg.hidden_size] device
  let b1 ← src.load (← src.resolve (visionNameCandidates s!"{p}.linear_fc1.bias")) #[cfg.intermediate_size] device
  let w2 ← src.load (← src.resolve (visionNameCandidates s!"{p}.linear_fc2.weight")) #[cfg.hidden_size, cfg.intermediate_size] device
  let b2 ← src.load (← src.resolve (visionNameCandidates s!"{p}.linear_fc2.bias")) #[cfg.hidden_size] device
  pure {
    linear_fc1_weight := reqGradFalse (← requireDType dtype w1)
    linear_fc1_bias := reqGradFalse (← requireDType dtype b1)
    linear_fc2_weight := reqGradFalse (← requireDType dtype w2)
    linear_fc2_bias := reqGradFalse (← requireDType dtype b2)
  }

private def loadVisionAttention (src : Source) (cfg : VisionConfig) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35VisionAttention (dtype := dtype) (device := device) cfg) := do
  let p := s!"model.visual.blocks.{layerIdx}.attn"
  let qkvW ← src.load (← src.resolve (visionNameCandidates s!"{p}.qkv.weight")) #[cfg.hidden_size * 3, cfg.hidden_size] device
  let qkvB ← src.load (← src.resolve (visionNameCandidates s!"{p}.qkv.bias")) #[cfg.hidden_size * 3] device
  let projW ← src.load (← src.resolve (visionNameCandidates s!"{p}.proj.weight")) #[cfg.hidden_size, cfg.hidden_size] device
  let projB ← src.load (← src.resolve (visionNameCandidates s!"{p}.proj.bias")) #[cfg.hidden_size] device
  pure {
    qkv_weight := reqGradFalse (← requireDType dtype qkvW)
    qkv_bias := reqGradFalse (← requireDType dtype qkvB)
    proj_weight := reqGradFalse (← requireDType dtype projW)
    proj_bias := reqGradFalse (← requireDType dtype projB)
  }

private def loadVisionBlock (src : Source) (cfg : VisionConfig) (layerIdx : Nat)
    (device : Device := Device.CPU) (dtype : DType)
    : IO (Qwen35VisionBlock (dtype := dtype) (device := device) cfg) := do
  let norm1 ← loadVisionLayerNorm src s!"model.visual.blocks.{layerIdx}.norm1" cfg.hidden_size device
  let norm2 ← loadVisionLayerNorm src s!"model.visual.blocks.{layerIdx}.norm2" cfg.hidden_size device
  let attn ← loadVisionAttention src cfg layerIdx device dtype
  let mlp ← loadVisionMLP src cfg layerIdx device dtype
  pure { norm1 := norm1, norm2 := norm2, attn := attn, mlp := mlp }

private def loadVisionModel (src : Source) (cfg : VisionConfig)
    (device : Device := Device.CPU)
    : IO (Σ dtype : RealFloatingDType, Qwen35VisionModel (device := device) cfg dtype.val) := do
  let convW ← src.load (← src.resolve (visionNameCandidates "model.visual.patch_embed.proj.weight")) #[cfg.hidden_size, cfg.in_channels, cfg.temporal_patch_size, cfg.patch_size, cfg.patch_size] device
  let convB ← src.load (← src.resolve (visionNameCandidates "model.visual.patch_embed.proj.bias")) #[cfg.hidden_size] device
  let dtype := convW.1
  let floating ← IO.ofExcept dtype.realFloating?
  let _ := floating.down
  let patchW : Tensor { device := device, shape := #[cfg.hidden_size, VisionConfig.patchDim cfg], dtype := dtype } :=
    Tensor.reshape convW.2 #[cfg.hidden_size, VisionConfig.patchDim cfg]
  let patchEmbed : Qwen35VisionPatchEmbed (dtype := dtype) (device := device) cfg := {
    weight := reqGradFalse patchW
    bias := reqGradFalse (← requireDType dtype convB)
  }

  let posEmbed ← src.load (← src.resolve (visionNameCandidates "model.visual.pos_embed.weight")) #[cfg.num_position_embeddings, cfg.hidden_size] device

  let mut blocks : Array (Qwen35VisionBlock (dtype := dtype) (device := device) cfg) := #[]
  for i in [:cfg.depth] do
    blocks := blocks.push (← loadVisionBlock src cfg i device dtype)

  let mergeUnit := VisionConfig.mergeUnit cfg
  let mergedHidden := cfg.hidden_size * mergeUnit
  let mergerNorm ← loadVisionLayerNorm src "model.visual.merger.norm" cfg.hidden_size device
  let fc1W ← src.load (← src.resolve (visionNameCandidates "model.visual.merger.linear_fc1.weight")) #[mergedHidden, mergedHidden] device
  let fc1B ← src.load (← src.resolve (visionNameCandidates "model.visual.merger.linear_fc1.bias")) #[mergedHidden] device
  let fc2W ← src.load (← src.resolve (visionNameCandidates "model.visual.merger.linear_fc2.weight")) #[cfg.out_hidden_size, mergedHidden] device
  let fc2B ← src.load (← src.resolve (visionNameCandidates "model.visual.merger.linear_fc2.bias")) #[cfg.out_hidden_size] device

  let merger : Qwen35VisionPatchMerger (dtype := dtype) (device := device) cfg := {
    norm := mergerNorm
    linear_fc1_weight := reqGradFalse (← requireDType dtype fc1W)
    linear_fc1_bias := reqGradFalse (← requireDType dtype fc1B)
    linear_fc2_weight := reqGradFalse (← requireDType dtype fc2W)
    linear_fc2_bias := reqGradFalse (← requireDType dtype fc2B)
  }

  pure ⟨RealFloatingDType.ofDType dtype, {
    patch_embed := patchEmbed
    pos_embed := reqGradFalse (← requireDType dtype posEmbed)
    blocks := blocks
    merger := merger
  }⟩

namespace Qwen35ForConditionalGeneration

/-- Load Qwen3.5 multimodal model from a SafeTensors file or a directory of shards. -/
def load (path : String) (cfg : VLConfig := {})
    (device : Device := Device.CPU)
    (log : Handlers := {})
    : IO (Σ dtype : RealFloatingDType, Σ visionDType : RealFloatingDType,
      Qwen35ForConditionalGeneration (device := device) cfg dtype.val visionDType.val) := do
  log.onInfo s!"Loading Qwen35ForConditionalGeneration (device := device) from {path}..."
  let visual ← loadVisionModel (← Source.open path) cfg.vision_config device
  let languageModel ← Qwen35ForCausalLM.load path cfg.text_config device log
  log.onInfo "Loaded Qwen35ForConditionalGeneration (device := device) weights."
  pure ⟨languageModel.1, visual.1, { visual := visual.2, language_model := languageModel.2 }⟩

end Qwen35ForConditionalGeneration

end torch.qwen35
