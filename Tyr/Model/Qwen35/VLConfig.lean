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
  in_channels : Nat := 3
  patch_size : Nat := 16
  spatial_merge_size : Nat := 2
  temporal_patch_size : Nat := 2
  out_hidden_size : Nat := 3584
  num_position_embeddings : Nat := 2304
  initializer_range : Float := 0.02
  deriving Repr, Inhabited

namespace Qwen35VisionConfig

def qwen35_0_8B : Qwen35VisionConfig := {
  depth := 12
  hidden_size := 768
  hidden_act := "gelu_pytorch_tanh"
  intermediate_size := 3072
  num_heads := 12
  in_channels := 3
  patch_size := 16
  spatial_merge_size := 2
  temporal_patch_size := 2
  out_hidden_size := 1024
  num_position_embeddings := 2304
  initializer_range := 0.02
}

/-- Default multimodal Qwen3.6-35B-A3B vision config from Hugging Face `config.json`. -/
def qwen36_35B_A3B : Qwen35VisionConfig := {
  depth := 27
  hidden_size := 1152
  hidden_act := "gelu_pytorch_tanh"
  intermediate_size := 4304
  num_heads := 16
  in_channels := 3
  patch_size := 16
  spatial_merge_size := 2
  temporal_patch_size := 2
  out_hidden_size := 2048
  num_position_embeddings := 2304
  initializer_range := 0.02
}

def patchDim (cfg : Qwen35VisionConfig) : Nat :=
  cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size

def mergeUnit (cfg : Qwen35VisionConfig) : Nat :=
  cfg.spatial_merge_size * cfg.spatial_merge_size

def headDim (cfg : Qwen35VisionConfig) : Nat :=
  if cfg.num_heads == 0 then cfg.hidden_size else cfg.hidden_size / cfg.num_heads

def mergedTokenCount (cfg : Qwen35VisionConfig) (nPatches : Nat) : Nat :=
  let u := mergeUnit cfg
  if u == 0 then nPatches else nPatches / u

end Qwen35VisionConfig

/-- Composite Qwen3.5 multimodal config (`text_config` + `vision_config`). -/
structure Qwen35VLConfig where
  text_config : Config := Config.qwen35_9B
  vision_config : Qwen35VisionConfig := {}
  image_token_id : UInt64 := 248056
  video_token_id : UInt64 := 248057
  vision_start_token_id : UInt64 := 248053
  vision_end_token_id : UInt64 := 248054
  tie_word_embeddings : Bool := false
  deriving Repr, Inhabited

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

def normalize (cfg : Qwen35VLConfig) : Qwen35VLConfig :=
  { cfg with text_config := Config.normalize cfg.text_config }

end Qwen35VLConfig

abbrev VisionConfig := Qwen35VisionConfig
abbrev VLConfig := Qwen35VLConfig

namespace VisionConfig

def qwen35_0_8B : VisionConfig := Qwen35VisionConfig.qwen35_0_8B

def qwen36_35B_A3B : VisionConfig := Qwen35VisionConfig.qwen36_35B_A3B

def patchDim (cfg : VisionConfig) : Nat := Qwen35VisionConfig.patchDim cfg
def mergeUnit (cfg : VisionConfig) : Nat := Qwen35VisionConfig.mergeUnit cfg
def headDim (cfg : VisionConfig) : Nat := Qwen35VisionConfig.headDim cfg
def mergedTokenCount (cfg : VisionConfig) (nPatches : Nat) : Nat :=
  Qwen35VisionConfig.mergedTokenCount cfg nPatches

end VisionConfig

namespace VLConfig

def qwen35_0_8B : VLConfig := Qwen35VLConfig.qwen35_0_8B

def qwen36_35B_A3B : VLConfig := Qwen35VLConfig.qwen36_35B_A3B

def normalize (cfg : VLConfig) : VLConfig := Qwen35VLConfig.normalize cfg

end VLConfig

end torch.qwen35
