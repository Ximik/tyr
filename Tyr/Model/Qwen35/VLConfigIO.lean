/-
  Tyr/Model/Qwen35/VLConfigIO.lean

  HuggingFace `config.json` loader for Qwen3.5 multimodal checkpoints.
-/
import Tyr.Model.Qwen35.ConfigIO
import Tyr.Model.Qwen35.VLConfig

namespace torch.qwen35

open Lean ConfigJson

private def parseVisionConfig (j : Json) (d : VisionConfig := {}) : Except String VisionConfig := do
  let hidden_size ← fieldD? j "hidden_size" d.hidden_size
  let num_heads ← fieldD? j "num_heads" d.num_heads
  let ⟨hHeads⟩ ← if h : 0 < num_heads ∧ num_heads ∣ hidden_size then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 vision config: num_heads ({num_heads}) must be positive and divide hidden_size ({hidden_size})"
  let ⟨hEven⟩ ← if h : 2 ∣ hidden_size / num_heads then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 vision config: head dim {hidden_size / num_heads} must be even for rotary embeddings"
  let spatial_merge_size ← fieldD? j "spatial_merge_size" d.spatial_merge_size
  let ⟨hMerge⟩ ← if h : 0 < spatial_merge_size then pure (PLift.up h) else
    throw "Invalid Qwen3.5 vision config: spatial_merge_size must be positive"
  pure {
    depth := ← fieldD? j "depth" d.depth
    hidden_size
    hidden_act := ← fieldD? j "hidden_act" d.hidden_act
    intermediate_size := ← fieldD? j "intermediate_size" d.intermediate_size
    num_heads
    in_channels := ← fieldD? j "in_channels" d.in_channels
    patch_size := ← fieldD? j "patch_size" d.patch_size
    spatial_merge_size
    temporal_patch_size := ← fieldD? j "temporal_patch_size" d.temporal_patch_size
    out_hidden_size := ← fieldD? j "out_hidden_size" d.out_hidden_size
    num_position_embeddings := ← fieldD? j "num_position_embeddings" d.num_position_embeddings
    half_head_dim := hidden_size / num_heads / 2
    headsSplit := by rw [Nat.mul_div_cancel' hEven, Nat.mul_div_cancel' hHeads.2]
    headsPos := hHeads.1
    mergePos := hMerge
  }

private def parseVLConfig (root : Json) (d : VLConfig := {}) : Except String VLConfig := do
  let textCfg ← Config.parseJson root d.text_config
  let visionCfg ← match ← objectField? root "vision_config" with
    | some v => parseVisionConfig v d.vision_config
    | none => pure d.vision_config
  let ⟨hMatch⟩ ← if h : visionCfg.out_hidden_size = textCfg.hidden_size then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 config: vision out_hidden_size ({visionCfg.out_hidden_size}) must match text hidden_size ({textCfg.hidden_size})"
  let image_token_id ← tokenIdFieldD? root "image_token_id" d.image_token_id
  let video_token_id ← tokenIdFieldD? root "video_token_id" d.video_token_id
  let vision_start_token_id ← tokenIdFieldD? root "vision_start_token_id" d.vision_start_token_id
  let vision_end_token_id ← tokenIdFieldD? root "vision_end_token_id" d.vision_end_token_id
  let vocab := textCfg.vocab_size
  let ⟨hIds⟩ ← if h : image_token_id.toNat < vocab ∧ video_token_id.toNat < vocab ∧
      vision_start_token_id.toNat < vocab ∧ vision_end_token_id.toNat < vocab then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 config: image, video and vision boundary token ids must be below vocab_size={vocab}"
  pure {
    text_config := textCfg
    vision_config := visionCfg
    image_token_id, video_token_id, vision_start_token_id, vision_end_token_id
    tie_word_embeddings := textCfg.tie_word_embeddings
    visionMatchesText := hMatch
    tokenIdsInVocab := hIds
  }

namespace VLConfig

/-- Load Qwen3.5 multimodal config from HuggingFace-style `config.json`. -/
def loadFromFile (path : String) (defaults : VLConfig := {}) : IO VLConfig := do
  IO.ofExcept (parseVLConfig (← parseFile path) defaults)

/-- Load Qwen3.5 multimodal config from a model directory containing `config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : VLConfig := {}) : IO VLConfig :=
  loadFromFile s!"{modelDir}/config.json" defaults

end VLConfig

end torch.qwen35
