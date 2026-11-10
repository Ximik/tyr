/-
  Tyr/Model/Gemma4/ConfigIO.lean

  HuggingFace `config.json` loader for Gemma 4 text checkpoints.
-/
import Tyr.Model.Gemma4.Config
import Tyr.Model.ConfigJson

namespace torch.gemma4

open Lean ConfigJson

private def layerType? (name : String) : Except String LayerType :=
  match LayerType.ofString? name with
  | some t => pure t
  | none => throw s!"unknown layer type \"{name}\"; expected sliding_attention or full_attention"

/-- `rope_parameters.<layerType>.<field>`, `default` when absent. -/
private def ropeFieldD? (j : Json) (layerType field : String) (default : Float) : Except String Float := do
  let some rope ← objectField? j "rope_parameters" | pure default
  let some params ← objectField? rope layerType | pure default
  fieldD? params field default

/-- A token id or a list of them at `key`; `default` when absent. -/
private def tokenIdsD? (j : Json) (key : String) (default : Array tokenizer.TokenId) :
    Except String (Array tokenizer.TokenId) := do
  match ← member? j key with
  | none | some .null => pure default
  | some (.arr _) => tokenIdArrayFieldD? j key default
  | some _ => return #[← tokenIdFieldD? j key 0]

/-- Parse the text decoder from a `config.json` node (`text_config`, or the root of a
text-only checkpoint). Behavior the decoder hard-codes is required of the file. -/
def Config.parseJson (j : Json) (d : Config := Config.gemma4_E4B) : Except String Config := do
  let textCfg := (← objectField? j "text_config").getD j
  requireEq textCfg "attention_bias" false
  let vocab ← fieldD? textCfg "vocab_size" d.vocab_size
  let hidden ← fieldD? textCfg "hidden_size" d.hidden_size
  let heads ← fieldD? textCfg "num_attention_heads" d.num_attention_heads
  let kvHeads ← fieldD? textCfg "num_key_value_heads" d.num_key_value_heads
  let globalKVHeads : FullSize ← match ← field? textCfg "num_global_key_value_heads" with
    | none => pure d.num_global_key_value_heads
    | some n => pure (if h : 0 < n then .distinct ⟨n, h⟩ else .same)
  let ⟨hHeads⟩ ← if h : 0 < heads ∧ 0 < kvHeads then pure (PLift.up h) else
    throw s!"Invalid Gemma4 config: num_attention_heads={heads} and num_key_value_heads={kvHeads} must be positive"
  let ⟨hSliding⟩ ← if h : kvHeads ∣ heads then pure (PLift.up h) else
    throw s!"Invalid Gemma4 config: num_attention_heads ({heads}) must be divisible by num_key_value_heads ({kvHeads})"
  let ⟨hFull⟩ ← if h : globalKVHeads.value kvHeads ∣ heads then pure (PLift.up h) else
    throw s!"Invalid Gemma4 config: num_attention_heads ({heads}) must be divisible by num_global_key_value_heads ({globalKVHeads.value kvHeads})"
  let headDim ← match ← field? textCfg "head_dim" with
    | some headDim => pure headDim
    | none =>
      if heads ∣ hidden then pure (hidden / heads)
      else throw s!"Invalid Gemma4 config: without head_dim, num_attention_heads={heads} must divide hidden_size={hidden}"
  let ⟨hHeadDims⟩ ← if h : 0 < headDim ∧ 2 ∣ headDim then pure (PLift.up h) else
    throw s!"Invalid Gemma4 config: head_dim={headDim} must be positive and even"
  let globalHalfHeadDim : FullSize ← match ← field? textCfg "global_head_dim" with
    | none => pure d.global_half_head_dim
    | some globalHeadDim =>
      if hPos : 0 < globalHeadDim then
        if hEven : globalHeadDim % 2 = 0 then pure (.distinct ⟨globalHeadDim / 2, by omega⟩)
        else throw s!"Invalid Gemma4 config: global_head_dim={globalHeadDim} must be even"
      else pure .same
  let fullHeadDim := 2 * globalHalfHeadDim.value (headDim / 2)
  let factor ← ropeFieldD? textCfg "full_attention" "partial_rotary_factor"
    ((Gemma4Config.fullRotaryDim d).toFloat / (Config.fullHeadDim d).toFloat)
  let fullRotaryDim := (fullHeadDim.toFloat * factor).round.toUInt64.toNat
  unless fullRotaryDim.toFloat == fullHeadDim.toFloat * factor do
    throw s!"Invalid Gemma4 config: partial_rotary_factor={factor} does not select a whole number of the {fullHeadDim} full-layer head channels"
  let ⟨hRotary⟩ ← if h : 0 < fullRotaryDim ∧ 2 ∣ fullRotaryDim ∧
      fullRotaryDim ≤ fullHeadDim then pure (PLift.up h) else
    throw s!"Invalid Gemma4 config: full-layer rotary dim {fullRotaryDim} must be positive, even and at most {fullHeadDim}"
  let slidingWindow ← fieldD? textCfg "sliding_window" d.sliding_window
  let ⟨hWindow⟩ ← if h : 0 < slidingWindow then pure (PLift.up h) else
    throw "Invalid Gemma4 config: sliding_window must be positive"
  let numLayers ← fieldD? textCfg "num_hidden_layers" d.num_hidden_layers
  let layerTypes ← match ← field? textCfg "layer_types" with
    | some (names : Array String) => names.mapM layerType?
    | none =>
      let interval : Nat ← fieldD? textCfg "sliding_window_pattern" 6
      unless 0 < interval do
        throw "Invalid Gemma4 config: sliding_window_pattern must be positive"
      pure (defaultLayerTypes numLayers interval)
  let ⟨hLayers⟩ ← if h : layerTypes.size = numLayers ∧ (numLayers = 0 ∨ layerTypes.back? = some .fullAttention) then
      pure (PLift.up h)
    else
      throw s!"Invalid Gemma4 config: layer_types needs one entry per layer ({numLayers}), ending in full_attention"
  let shared ← fieldD? textCfg "num_kv_shared_layers" d.num_kv_shared_layers
  let ⟨hSharing⟩ ← if h : kvSharingValid layerTypes numLayers shared = true then pure (PLift.up h) else
    throw s!"Invalid Gemma4 config: num_kv_shared_layers={shared} needs fewer shared than total layers ({numLayers}) and an earlier source layer of each shared kind"
  let enableMoe ← fieldD? textCfg "enable_moe_block" d.enable_moe_block
  let numExperts ← fieldD? textCfg "num_experts" d.num_experts
  let topK ← fieldD? textCfg "top_k_experts" d.top_k_experts
  let moeInter ← fieldD? textCfg "moe_intermediate_size" d.moe_intermediate_size
  let ⟨hTopK⟩ ← if h : topK ≤ numExperts then pure (PLift.up h) else
    throw s!"Invalid Gemma4 config: top_k_experts={topK} exceeds num_experts={numExperts}"
  let ⟨hMoe⟩ ← if h : enableMoe = true → 0 < topK ∧ 0 < moeInter then pure (PLift.up h) else
    throw "Invalid Gemma4 config: an enabled MoE block needs positive top_k_experts and moe_intermediate_size"
  let perLayerHidden ← fieldD? textCfg "hidden_size_per_layer_input" d.hidden_size_per_layer_input
  let perLayerVocab0 ← fieldD? textCfg "vocab_size_per_layer_input" d.vocab_size_per_layer_input
  let perLayerVocab := if 0 < perLayerHidden ∧ perLayerVocab0 = 0 then vocab else perLayerVocab0
  let ⟨hPerLayer⟩ ← if h : 0 < perLayerHidden → vocab ≤ perLayerVocab then pure (PLift.up h) else
    throw s!"Invalid Gemma4 config: vocab_size_per_layer_input={perLayerVocab} must cover vocab_size={vocab}"
  let bidirectional ← match ← member? textCfg "use_bidirectional_attention" with
    | none | some .null => pure d.use_bidirectional_attention
    | some _ => fieldD? textCfg "use_bidirectional_attention" d.use_bidirectional_attention
  let ⟨hBidirectional⟩ ← if h : bidirectional = "" ∨ bidirectional = "vision" then pure (PLift.up h) else
    throw s!"Invalid Gemma4 config: unsupported use_bidirectional_attention=\"{bidirectional}\"; this model implements vision"
  let slidingTheta ← ropeFieldD? textCfg "sliding_attention" "rope_theta" d.sliding_rope_theta
  let fullTheta ← ropeFieldD? textCfg "full_attention" "rope_theta" d.full_rope_theta
  let eps ← fieldD? textCfg "rms_norm_eps" d.rms_norm_eps
  let ⟨hFloats⟩ ← if h : 0 < slidingTheta ∧ 0 < fullTheta ∧ 0 < eps then pure (PLift.up h) else
    throw s!"Invalid Gemma4 config: rope thetas ({slidingTheta}, {fullTheta}) and rms_norm_eps={eps} must be positive"
  let softcap ← match ← member? textCfg "final_logit_softcapping" with
    | none => pure d.final_logit_softcapping
    | some .null => pure none
    | some _ => pure (some (← fieldD? textCfg "final_logit_softcapping" 0.0))
  let ⟨hSoftcap⟩ ← if h : softcap.all (0 < ·) = true then pure (PLift.up h) else
    throw "Invalid Gemma4 config: final_logit_softcapping must be positive or null"
  let pad ← match ← member? textCfg "pad_token_id" with
    | none => pure d.pad_token_id
    | some .null => pure none
    | some _ => pure (some (← tokenIdFieldD? textCfg "pad_token_id" 0))
  let eos ← tokenIdsD? textCfg "eos_token_id" d.eos_token_id
  let ⟨hIds⟩ ← if h : pad.all (·.toNat < vocab) = true ∧ eos.all (·.toNat < vocab) = true then pure (PLift.up h) else
    throw s!"Invalid Gemma4 config: pad and eos token ids must be below vocab_size={vocab}"
  let tie ← match ← field? j "tie_word_embeddings" with
    | some tie => pure tie
    | none => fieldD? textCfg "tie_word_embeddings" d.tie_word_embeddings
  pure {
    vocab_size := vocab
    hidden_size := hidden
    intermediate_size := ← fieldD? textCfg "intermediate_size" d.intermediate_size
    num_hidden_layers := numLayers
    num_attention_heads := heads
    num_key_value_heads := kvHeads
    num_global_key_value_heads := globalKVHeads
    half_head_dim := headDim / 2
    global_half_head_dim := globalHalfHeadDim
    sliding_window := slidingWindow
    sliding_rope_theta := slidingTheta
    full_rope_theta := fullTheta
    half_full_rotary_dim := fullRotaryDim / 2
    rms_norm_eps := eps
    hidden_activation := ← fieldD? textCfg "hidden_activation" d.hidden_activation
    attention_k_eq_v := ← fieldD? textCfg "attention_k_eq_v" d.attention_k_eq_v
    use_bidirectional_attention := bidirectional
    layer_types := layerTypes
    num_kv_shared_layers := shared
    use_double_wide_mlp := ← fieldD? textCfg "use_double_wide_mlp" d.use_double_wide_mlp
    hidden_size_per_layer_input := perLayerHidden
    vocab_size_per_layer_input := perLayerVocab
    enable_moe_block := enableMoe
    num_experts := numExperts
    top_k_experts := topK
    moe_intermediate_size := moeInter
    tie_word_embeddings := tie
    final_logit_softcapping := softcap
    pad_token_id := pad
    eos_token_id := eos
    slidingKVHeadsDivide := hSliding
    fullKVHeadsDivide := hFull
    headsPos := hHeads
    headDimPos := by omega
    fullRotaryValid := by
      obtain ⟨_, _, hLe⟩ := hRotary
      exact ⟨by omega, by omega⟩
    slidingWindowPos := hWindow
    layerTypesValid := hLayers
    kvSharing := hSharing
    topKValid := hTopK
    moeValid := hMoe
    perLayerVocab := hPerLayer
    bidirectionalSupported := hBidirectional
    floatsPos := hFloats
    softcapPos := hSoftcap
    tokenIdsInVocab := hIds
  }

namespace Config

def loadFromFile (path : String) (defaults : Config := Config.gemma4_E4B) : IO Config := do
  IO.ofExcept (Config.parseJson (← parseFile path) defaults)

def loadFromPretrainedDir (modelDir : String) (defaults : Config := Config.gemma4_E4B) : IO Config :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Config

end torch.gemma4
