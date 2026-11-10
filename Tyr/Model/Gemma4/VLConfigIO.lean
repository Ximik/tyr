/-
  Tyr/Model/Gemma4/VLConfigIO.lean

  HuggingFace `config.json` + `processor_config.json` loader for Gemma 4
  multimodal checkpoints.
-/
import Tyr.Model.Gemma4.VLConfig
import Tyr.Model.Gemma4.ConfigIO
import Tyr.Model.ConfigJson

namespace torch.gemma4

open Lean ConfigJson

def VisionConfig.parseJson (j : Json) (d : VisionConfig := {}) : Except String VisionConfig := do
  let hidden_size ← fieldD? j "hidden_size" d.hidden_size
  let num_attention_heads ← fieldD? j "num_attention_heads" d.num_attention_heads
  let num_key_value_heads ← fieldD? j "num_key_value_heads" d.num_key_value_heads
  let head_dim ← fieldD? j "head_dim" d.head_dim
  let ⟨hQuery⟩ ← if h : num_attention_heads * head_dim = hidden_size then pure (PLift.up h) else
    throw s!"Invalid Gemma4 vision config: num_attention_heads ({num_attention_heads}) * head_dim ({head_dim}) must equal hidden_size ({hidden_size})"
  let ⟨hKV⟩ ← if h : num_key_value_heads * head_dim = hidden_size then pure (PLift.up h) else
    throw s!"Invalid Gemma4 vision config: num_key_value_heads ({num_key_value_heads}) * head_dim ({head_dim}) must equal hidden_size ({hidden_size})"
  let ⟨hDivide⟩ ← if h : num_key_value_heads ∣ num_attention_heads then pure (PLift.up h) else
    throw s!"Invalid Gemma4 vision config: num_attention_heads ({num_attention_heads}) must be divisible by num_key_value_heads ({num_key_value_heads})"
  let ⟨hHeadDim⟩ ← if h : 0 < head_dim ∧ 4 ∣ head_dim then pure (PLift.up h) else
    throw s!"Invalid Gemma4 vision config: head_dim={head_dim} must be a positive multiple of 4 for 2-D rotary embeddings"
  let patch_size ← fieldD? j "patch_size" d.patch_size
  let pooling_kernel_size ← fieldD? j "pooling_kernel_size" d.pooling_kernel_size
  let ⟨hPatch⟩ ← if h : 0 < patch_size ∧ 0 < pooling_kernel_size then pure (PLift.up h) else
    throw "Invalid Gemma4 vision config: patch_size and pooling_kernel_size must be positive"
  let rms_norm_eps ← fieldD? j "rms_norm_eps" d.rms_norm_eps
  let rope_theta ← match ← objectField? j "rope_parameters" with
    | some rope => fieldD? rope "rope_theta" d.rope_theta
    | none => pure d.rope_theta
  let ⟨hFloats⟩ ← if h : 0 < rms_norm_eps ∧ 0 < rope_theta then pure (PLift.up h) else
    throw s!"Invalid Gemma4 vision config: rms_norm_eps={rms_norm_eps} and rope_theta={rope_theta} must be positive"
  pure {
    hidden_size
    intermediate_size := ← fieldD? j "intermediate_size" d.intermediate_size
    num_hidden_layers := ← fieldD? j "num_hidden_layers" d.num_hidden_layers
    num_attention_heads
    num_key_value_heads
    quarter_head_dim := head_dim / 4
    patch_size
    pooling_kernel_size
    position_embedding_size := ← fieldD? j "position_embedding_size" d.position_embedding_size
    rms_norm_eps
    rope_theta
    standardize := ← fieldD? j "standardize" d.standardize
    use_clipped_linears := ← fieldD? j "use_clipped_linears" d.use_clipped_linears
    queryHeadsSplit := by rw [Nat.mul_div_cancel' hHeadDim.2]; exact hQuery
    keyValueHeadsSplit := by rw [Nat.mul_div_cancel' hHeadDim.2]; exact hKV
    kvHeadsDivide := hDivide
    headDimPos := by omega
    patchPoolPos := hPatch
    floatsPos := hFloats
  }

/-- Image processor config. A zero patch or pooling size defers to the vision tower,
and zero token budgets to `softTokens`. -/
def ImageProcessorConfig.parseJson (j : Json) (vision : VisionConfig) (softTokens : Nat)
    (d : ImageProcessorConfig := {}) : Except String ImageProcessorConfig := do
  let orElse (n fallback : Nat) := if n == 0 then fallback else n
  pure {
    do_resize := ← fieldD? j "do_resize" d.do_resize
    do_rescale := ← fieldD? j "do_rescale" d.do_rescale
    do_normalize := ← fieldD? j "do_normalize" d.do_normalize
    image_mean := ← fieldD? j "image_mean" d.image_mean
    image_std := ← fieldD? j "image_std" d.image_std
    image_seq_length := orElse (← fieldD? j "image_seq_length" d.image_seq_length) softTokens
    max_soft_tokens := orElse (← fieldD? j "max_soft_tokens" d.max_soft_tokens) softTokens
    patch_size := orElse (← fieldD? j "patch_size" d.patch_size) vision.patch_size
    pooling_kernel_size := orElse (← fieldD? j "pooling_kernel_size" d.pooling_kernel_size) vision.pooling_kernel_size
    rescale_factor := ← fieldD? j "rescale_factor" d.rescale_factor
  }

def VLConfig.parseJson
    (root : Json)
    (processorRoot? : Option Json := none)
    (d : VLConfig := {})
    : Except String VLConfig := do
  let vision_config ← match ← objectField? root "vision_config" with
    | some j => VisionConfig.parseJson j d.vision_config
    | none => pure d.vision_config
  let text_config ← Config.parseJson root d.text_config
  let softTokens0 ← fieldD? root "vision_soft_tokens_per_image" d.vision_soft_tokens_per_image
  let imageProcJson ← match processorRoot? with
    | some proc => objectField? proc "image_processor"
    | none => pure none
  let image_processor ← match imageProcJson with
    | some j => ImageProcessorConfig.parseJson j vision_config softTokens0 d.image_processor
    | none => pure d.image_processor
  let vision_soft_tokens_per_image := if softTokens0 == 0 then image_processor.softTokenCount else softTokens0
  let ⟨hProc⟩ ← if h : image_processor.patch_size = vision_config.patch_size ∧
      image_processor.pooling_kernel_size = vision_config.pooling_kernel_size then pure (PLift.up h) else
    throw s!"Invalid Gemma4 processor config: patch_size={image_processor.patch_size} and pooling_kernel_size={image_processor.pooling_kernel_size} must match the vision config ({vision_config.patch_size}, {vision_config.pooling_kernel_size})"
  let boi_token_id ← tokenIdFieldD? root "boi_token_id" d.boi_token_id
  let image_token_id ← tokenIdFieldD? root "image_token_id" d.image_token_id
  let eoi_token_id ← tokenIdFieldD? root "eoi_token_id" d.eoi_token_id
  let video_token_id ← tokenIdFieldD? root "video_token_id" d.video_token_id
  let audio_token_id ← tokenIdFieldD? root "audio_token_id" d.audio_token_id
  let vocab := text_config.vocab_size
  let ⟨hIds⟩ ← if h : boi_token_id.toNat < vocab ∧ image_token_id.toNat < vocab ∧
      eoi_token_id.toNat < vocab ∧ video_token_id.toNat < vocab ∧ audio_token_id.toNat < vocab then
      pure (PLift.up h)
    else
      throw s!"Invalid Gemma4 config: media token ids must be below vocab_size={vocab}"
  let ⟨hPad⟩ ← if h : text_config.pad_token_id.isSome = true then pure (PLift.up h) else
    throw "Invalid Gemma4 config: multimodal inputs need pad_token_id"
  pure {
    text_config
    vision_config
    image_processor
    boi_token_id, image_token_id, eoi_token_id, video_token_id, audio_token_id
    vision_soft_tokens_per_image
    tie_word_embeddings := ← fieldD? root "tie_word_embeddings" d.tie_word_embeddings
    tokenIdsInVocab := hIds
    padPresent := hPad
    processorMatchesVision := hProc
  }

namespace VLConfig

def loadFromFiles
    (configPath : String)
    (processorPath : Option String := none)
    (defaults : VLConfig := {})
    : IO VLConfig := do
  let root ← parseFile configPath
  let processorRoot? ←
    match processorPath with
    | some path => if ← System.FilePath.pathExists path then some <$> parseFile path else pure none
    | none => pure none
  IO.ofExcept (VLConfig.parseJson root processorRoot? defaults)

def loadFromPretrainedDir (modelDir : String) (defaults : VLConfig := {}) : IO VLConfig :=
  loadFromFiles
    s!"{modelDir}/config.json"
    (some s!"{modelDir}/processor_config.json")
    defaults

end VLConfig

end torch.gemma4
