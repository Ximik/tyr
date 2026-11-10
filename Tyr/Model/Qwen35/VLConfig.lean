/-
  Tyr/Model/Qwen35/VLConfig.lean

  Multimodal (vision + text) configuration for standalone Qwen3.5.
-/
import Tyr.Model.Qwen35.Config

namespace torch.qwen35

/-- Vision backbone config used by Qwen3.5 multimodal checkpoints. -/
structure Qwen35VisionConfig where
  depth : Nat := 27
  hidden_size : Nat := 1152
  hidden_act : String := "gelu_pytorch_tanh"
  intermediate_size : Nat := 4304
  num_heads : Nat := 16
  /-- Half the per-head channel count (`hidden_size / num_heads / 2`). -/
  half_head_dim : Nat := 36
  in_channels : Nat := 3
  patch_size : Nat := 16
  spatial_merge_size : Nat := 2
  temporal_patch_size : Nat := 2
  out_hidden_size : Nat := 3584
  num_position_embeddings : Nat := 2304
  /-- Attention splits `hidden_size` into a nonzero number of equal heads of an even
      size; checked when the config is built. -/
  headsSplit : num_heads * (2 * half_head_dim) = hidden_size := by decide
  headsPos : 0 < num_heads := by decide
  /-- Patches merge in nonempty `spatial_merge_size²` groups; checked when the config is built. -/
  mergePos : 0 < spatial_merge_size := by decide
  deriving Repr


namespace Qwen35VisionConfig

def qwen35_0_8B : Qwen35VisionConfig := {
  depth := 12
  hidden_size := 768
  hidden_act := "gelu_pytorch_tanh"
  intermediate_size := 3072
  num_heads := 12
  half_head_dim := 32
  in_channels := 3
  patch_size := 16
  spatial_merge_size := 2
  temporal_patch_size := 2
  out_hidden_size := 1024
  num_position_embeddings := 2304
}

/-- Default multimodal Qwen3.6-35B-A3B vision config from Hugging Face `config.json`. -/
def qwen36_35B_A3B : Qwen35VisionConfig := {
  depth := 27
  hidden_size := 1152
  hidden_act := "gelu_pytorch_tanh"
  intermediate_size := 4304
  num_heads := 16
  half_head_dim := 36
  in_channels := 3
  patch_size := 16
  spatial_merge_size := 2
  temporal_patch_size := 2
  out_hidden_size := 2048
  num_position_embeddings := 2304
}

@[simp] def patchDim (cfg : Qwen35VisionConfig) : Nat :=
  cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size

@[simp] def mergeUnit (cfg : Qwen35VisionConfig) : Nat :=
  cfg.spatial_merge_size * cfg.spatial_merge_size

def headDim (cfg : Qwen35VisionConfig) : Nat :=
  2 * cfg.half_head_dim

theorem heads_mul_headDim (cfg : Qwen35VisionConfig) : cfg.num_heads * cfg.headDim = cfg.hidden_size :=
  cfg.headsSplit

def mergedTokenCount (cfg : Qwen35VisionConfig) (nPatches : Nat) : Nat :=
  nPatches / mergeUnit cfg

end Qwen35VisionConfig

/-- Composite Qwen3.5 multimodal config (`text_config` + `vision_config`). -/
structure Qwen35VLConfig where
  text_config : Config := Config.qwen35_0_8B
  vision_config : Qwen35VisionConfig := Qwen35VisionConfig.qwen35_0_8B
  image_token_id : tokenizer.TokenId := 248056
  video_token_id : tokenizer.TokenId := 248057
  vision_start_token_id : tokenizer.TokenId := 248053
  vision_end_token_id : tokenizer.TokenId := 248054
  tie_word_embeddings : Bool := false
  /-- Vision features replace placeholder embeddings one-for-one; checked when the config is built. -/
  visionMatchesText : vision_config.out_hidden_size = text_config.hidden_size := by decide
  /-- Placeholder and boundary ids are text tokens; checked when the config is built. -/
  tokenIdsInVocab :
    image_token_id.toNat < text_config.vocab_size ∧ video_token_id.toNat < text_config.vocab_size ∧
    vision_start_token_id.toNat < text_config.vocab_size ∧
    vision_end_token_id.toNat < text_config.vocab_size := by decide
  deriving Repr

namespace Qwen35VLConfig

def qwen35_0_8B : Qwen35VLConfig := {
  text_config := Config.qwen35_0_8B
  vision_config := Qwen35VisionConfig.qwen35_0_8B
  image_token_id := 248056
  video_token_id := 248057
  vision_start_token_id := 248053
  vision_end_token_id := 248054
  tie_word_embeddings := true
}

/-- Default multimodal Qwen3.6-35B-A3B config from Hugging Face `config.json`. -/
def qwen36_35B_A3B : Qwen35VLConfig := {
  text_config := Config.qwen36_35B_A3B
  vision_config := Qwen35VisionConfig.qwen36_35B_A3B
  image_token_id := 248056
  video_token_id := 248057
  vision_start_token_id := 248053
  vision_end_token_id := 248054
  tie_word_embeddings := false
}

end Qwen35VLConfig

abbrev VisionConfig := Qwen35VisionConfig
abbrev VLConfig := Qwen35VLConfig

namespace VisionConfig

def qwen35_0_8B : VisionConfig := Qwen35VisionConfig.qwen35_0_8B

def qwen36_35B_A3B : VisionConfig := Qwen35VisionConfig.qwen36_35B_A3B

@[simp] def patchDim (cfg : VisionConfig) : Nat := Qwen35VisionConfig.patchDim cfg
@[simp] def mergeUnit (cfg : VisionConfig) : Nat := Qwen35VisionConfig.mergeUnit cfg
def headDim (cfg : VisionConfig) : Nat := Qwen35VisionConfig.headDim cfg

theorem heads_mul_headDim (cfg : VisionConfig) : cfg.num_heads * headDim cfg = cfg.hidden_size :=
  Qwen35VisionConfig.heads_mul_headDim cfg
def mergedTokenCount (cfg : VisionConfig) (nPatches : Nat) : Nat :=
  Qwen35VisionConfig.mergedTokenCount cfg nPatches

theorem mergedTokenCount_mul {cfg : VisionConfig} {nPatches : Nat}
    (h : mergeUnit cfg ∣ nPatches) (d : Nat) :
    mergedTokenCount cfg nPatches * (d * mergeUnit cfg) = nPatches * d := by
  unfold mergedTokenCount Qwen35VisionConfig.mergedTokenCount
  rw [Nat.mul_comm d, ← Nat.mul_assoc]
  change nPatches / mergeUnit cfg * mergeUnit cfg * d = nPatches * d
  rw [Nat.div_mul_cancel h]

end VisionConfig

namespace VLConfig

def qwen35_0_8B : VLConfig := Qwen35VLConfig.qwen35_0_8B

def qwen36_35B_A3B : VLConfig := Qwen35VLConfig.qwen36_35B_A3B

end VLConfig

end torch.qwen35
