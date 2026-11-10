/-
  Tyr/Model/Gemma4/VLConfig.lean

  Multimodal (vision + text) configuration for Gemma 4.
-/
import Tyr.Model.Gemma4.Config

namespace torch.gemma4

/-- Vision backbone config published in Gemma 4 `config.json`. -/
structure Gemma4VisionConfig where
  hidden_size : Nat := 768
  intermediate_size : Nat := 3072
  num_hidden_layers : Nat := 16
  num_attention_heads : Nat := 12
  num_key_value_heads : Nat := 12
  /-- A quarter of the head dimension: 2-D RoPE rotates each head's two axis halves in
      channel pairs, so heads have `4 * quarter_head_dim` channels. -/
  quarter_head_dim : Nat := 16
  patch_size : Nat := 16
  pooling_kernel_size : Nat := 3
  position_embedding_size : Nat := 10240
  rms_norm_eps : Float := 1e-6
  rope_theta : Float := 100.0
  standardize : Bool := false
  use_clipped_linears : Bool := false
  /-- The `hidden_size` projections split into query and key/value heads; checked when the config is built. -/
  queryHeadsSplit : num_attention_heads * (4 * quarter_head_dim) = hidden_size := by decide
  keyValueHeadsSplit : num_key_value_heads * (4 * quarter_head_dim) = hidden_size := by decide
  /-- Query heads are whole multiples of K/V heads; checked when the config is built. -/
  kvHeadsDivide : num_key_value_heads ∣ num_attention_heads := by decide
  /-- Heads are nonempty; checked when the config is built. -/
  headDimPos : 0 < quarter_head_dim := by decide
  /-- Patches and pooling windows are nonempty; checked when the config is built. -/
  patchPoolPos : 0 < patch_size ∧ 0 < pooling_kernel_size := by decide
  /-- RoPE frequencies and RMSNorm divide by these; checked when the config is built. -/
  floatsPos : 0 < rms_norm_eps ∧ 0 < rope_theta := by native_decide
  deriving Repr


namespace Gemma4VisionConfig

/-- Channels per attention head. -/
def head_dim (cfg : Gemma4VisionConfig) : Nat := 4 * cfg.quarter_head_dim

@[simp] def in_channels (_cfg : Gemma4VisionConfig) : Nat := 3

@[simp] def patchDim (cfg : Gemma4VisionConfig) : Nat :=
  in_channels cfg * cfg.patch_size * cfg.patch_size

end Gemma4VisionConfig

/-- Image processor config published in Gemma 4 `processor_config.json`. -/
structure Gemma4ImageProcessorConfig where
  do_resize : Bool := true
  do_rescale : Bool := true
  do_normalize : Bool := false
  image_mean : Array Float := #[0.0, 0.0, 0.0]
  image_std : Array Float := #[1.0, 1.0, 1.0]
  image_seq_length : Nat := 280
  max_soft_tokens : Nat := 280
  patch_size : Nat := 16
  pooling_kernel_size : Nat := 3
  rescale_factor : Float := 1.0 / 255.0
  deriving Repr, Inhabited

namespace Gemma4ImageProcessorConfig

def softTokenCount (cfg : Gemma4ImageProcessorConfig) : Nat :=
  if cfg.max_soft_tokens > 0 then cfg.max_soft_tokens else cfg.image_seq_length

end Gemma4ImageProcessorConfig

/-- Composite Gemma 4 multimodal config. -/
structure Gemma4VLConfig where
  text_config : Config := Config.gemma4_E4B
  vision_config : Gemma4VisionConfig := {}
  image_processor : Gemma4ImageProcessorConfig := {}
  boi_token_id : tokenizer.TokenId := 255999
  image_token_id : tokenizer.TokenId := 258880
  eoi_token_id : tokenizer.TokenId := 258882
  video_token_id : tokenizer.TokenId := 258884
  audio_token_id : tokenizer.TokenId := 258881
  vision_soft_tokens_per_image : Nat := 280
  tie_word_embeddings : Bool := true
  /-- Media placeholder and boundary ids are text tokens; checked when the config is built. -/
  tokenIdsInVocab :
    boi_token_id.toNat < text_config.vocab_size ∧ image_token_id.toNat < text_config.vocab_size ∧
    eoi_token_id.toNat < text_config.vocab_size ∧ video_token_id.toNat < text_config.vocab_size ∧
    audio_token_id.toNat < text_config.vocab_size := by decide
  /-- Image placeholders are replaced by the pad token before text embedding; checked when the
      config is built. -/
  padPresent : text_config.pad_token_id.isSome = true := by decide
  /-- The processor patches and pools like the vision tower; checked when the config is built. -/
  processorMatchesVision : image_processor.patch_size = vision_config.patch_size ∧
    image_processor.pooling_kernel_size = vision_config.pooling_kernel_size := by decide
  deriving Repr

namespace Gemma4VLConfig

def imageSoftTokenCount (cfg : Gemma4VLConfig) : Nat :=
  let procBudget := cfg.image_processor.softTokenCount
  if procBudget > 0 then procBudget else cfg.vision_soft_tokens_per_image

end Gemma4VLConfig

abbrev VisionConfig := Gemma4VisionConfig
abbrev ImageProcessorConfig := Gemma4ImageProcessorConfig
abbrev VLConfig := Gemma4VLConfig

namespace VisionConfig

@[simp] def in_channels (cfg : VisionConfig) : Nat := Gemma4VisionConfig.in_channels cfg
@[simp] def patchDim (cfg : VisionConfig) : Nat := Gemma4VisionConfig.patchDim cfg

end VisionConfig

namespace ImageProcessorConfig

def softTokenCount (cfg : ImageProcessorConfig) : Nat := Gemma4ImageProcessorConfig.softTokenCount cfg

end ImageProcessorConfig

namespace VLConfig

def imageSoftTokenCount (cfg : VLConfig) : Nat := Gemma4VLConfig.imageSoftTokenCount cfg

end VLConfig

end torch.gemma4
