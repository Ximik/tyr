/-
  Tyr/Model/Qwen35/ConfigIO.lean

  HuggingFace `config.json` loader for standalone Qwen3.5 causal-LM.
-/
import Tyr.Model.Qwen35.Config
import Tyr.Model.ConfigJson

namespace torch.qwen35

open Lean ConfigJson

private def layerType? (name : String) : Except String LayerType :=
  match LayerType.ofString? name with
  | some t => pure t
  | none => throw s!"unknown layer type \"{name}\"; expected linear_attention or full_attention"

/-- RoPE settings live in `rope_parameters` on newer checkpoints, else at the top level. -/
private def ropeField? {α : Type} [FromJson α] (j : Json) (key : String) : Except String (Option α) := do
  if let some params ← objectField? j "rope_parameters" then
    if let some x ← field? params key then return some x
  field? j key

/-- Parse the text decoder from a `config.json` node (`text_config`, or the root of a
text-only checkpoint). Behavior the decoder hard-codes is required of the file:
SiLU MLP, gated attention without bias, no MLP-only layers, float32 linear-attention
state, and plain or multimodal RoPE (identical for text positions). -/
def Config.parseJson (j : Json) (d : Config := Config.qwen35_9B) : Except String Config := do
  let textCfg := (← objectField? j "text_config").getD j
  requireOneOf textCfg "hidden_act" ["silu"]
  requireEq textCfg "attention_bias" false
  requireEq textCfg "attn_output_gate" true
  requireOneOf textCfg "mamba_ssm_dtype" ["float32"]
  requireEq textCfg "mlp_only_layers" (#[] : Array Nat)

  let vocab ← fieldD? textCfg "vocab_size" d.vocab_size
  let hidden ← fieldD? textCfg "hidden_size" d.hidden_size
  let heads ← fieldD? textCfg "num_attention_heads" d.num_attention_heads
  let kvHeads ← fieldD? textCfg "num_key_value_heads" d.num_key_value_heads
  let linearKeyHeads ← fieldD? textCfg "linear_num_key_heads" d.linear_num_key_heads
  let linearValueHeads ← fieldD? textCfg "linear_num_value_heads" d.linear_num_value_heads
  let ⟨hHeads⟩ ← if h : 0 < heads ∧ 0 < kvHeads ∧ 0 < linearKeyHeads then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 config: num_attention_heads={heads}, num_key_value_heads={kvHeads} and linear_num_key_heads={linearKeyHeads} must be positive"
  let ⟨hKV⟩ ← if h : kvHeads ∣ heads then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 config: num_attention_heads ({heads}) must be divisible by num_key_value_heads ({kvHeads})"
  let ⟨hLinear⟩ ← if h : linearKeyHeads ∣ linearValueHeads then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 config: linear_num_value_heads ({linearValueHeads}) must be divisible by linear_num_key_heads ({linearKeyHeads})"
  let headDim ← match ← field? textCfg "head_dim" with
    | some headDim => pure headDim
    | none =>
      if heads ∣ hidden then pure (hidden / heads)
      else throw s!"Invalid Qwen3.5 config: without head_dim, num_attention_heads={heads} must divide hidden_size={hidden}"

  let factor : Float := (← ropeField? textCfg "partial_rotary_factor").getD
    (d.rotaryDim.toFloat / d.head_dim.toFloat)
  let rotaryDim := (headDim.toFloat * factor).round.toUInt64.toNat
  unless rotaryDim.toFloat == headDim.toFloat * factor do
    throw s!"Invalid Qwen3.5 config: partial_rotary_factor={factor} does not select a whole number of the {headDim} head channels"
  let ⟨hRotary⟩ ← if h : 0 < rotaryDim ∧ 2 ∣ rotaryDim ∧ rotaryDim ≤ headDim then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 config: rotary dim {rotaryDim} (head_dim * partial_rotary_factor) must be positive, even and at most head_dim={headDim}"
  for key in ["rope_scaling", "rope_parameters"] do
    if let some (scaling : Json) ← field? textCfg key then
      if scaling != .null then
        let kind : Option String ← firstFieldD? scaling ["rope_type", "type"] none
        unless kind.all (fun k => k == "default" || k == "mrope") do
          throw s!"unsupported '{key}' = {scaling.compress}; this model implements plain RoPE"
  let ropeTheta := (← ropeField? textCfg "rope_theta").getD d.rope_theta
  let ⟨hTheta⟩ ← if h : 0 < ropeTheta then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 config: rope_theta={ropeTheta} must be positive"
  let eps ← fieldD? textCfg "rms_norm_eps" d.rms_norm_eps
  let ⟨hEps⟩ ← if h : 0 < eps then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 config: rms_norm_eps={eps} must be positive"

  let numLayers ← fieldD? textCfg "num_hidden_layers" d.num_hidden_layers
  let layerTypes ← match ← field? textCfg "layer_types" with
    | some (names : Array String) => names.mapM layerType?
    | none =>
      let interval : Nat ← fieldD? textCfg "full_attention_interval" 4
      unless 0 < interval do
        throw "Invalid Qwen3.5 config: full_attention_interval must be positive"
      pure (defaultLayerTypes numLayers interval)
  let ⟨hLayers⟩ ← if h : layerTypes.size = numLayers then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 config: layer_types has {layerTypes.size} entries for num_hidden_layers={numLayers}"
  let convKernel ← fieldD? textCfg "linear_conv_kernel_dim" d.linear_conv_kernel_dim
  let ⟨hConv⟩ ← if h : 0 < convKernel then pure (PLift.up h) else
    throw "Invalid Qwen3.5 config: linear_conv_kernel_dim must be positive"
  let keyHeadDim ← fieldD? textCfg "linear_key_head_dim" d.linear_key_head_dim
  let ⟨hKeyHeadDim⟩ ← if h : 0 < keyHeadDim then pure (PLift.up h) else
    throw "Invalid Qwen3.5 config: linear_key_head_dim must be positive"
  let numExperts ← fieldD? textCfg "num_experts" d.num_experts
  let perTok ← fieldD? textCfg "num_experts_per_tok" d.num_experts_per_tok
  let ⟨hExperts⟩ ← if h : 0 < numExperts → 0 < perTok ∧ perTok ≤ numExperts then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 config: num_experts_per_tok={perTok} must be between 1 and num_experts={numExperts}"
  let eos ← match ← tokenIdField? textCfg "eos_token_id" with
    | some id => pure (some id)
    | none => pure d.eos_token_id
  let ⟨hEos⟩ ← if h : eos.all (·.toNat < vocab) = true then pure (PLift.up h) else
    throw s!"Invalid Qwen3.5 config: eos_token_id must be below vocab_size={vocab}"
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
    head_dim := headDim
    rope_theta := ropeTheta
    half_rotary_dim := rotaryDim / 2
    rms_norm_eps := eps
    linear_conv_kernel_dim := convKernel
    linear_key_head_dim := keyHeadDim
    linear_value_head_dim := ← fieldD? textCfg "linear_value_head_dim" d.linear_value_head_dim
    linear_num_key_heads := linearKeyHeads
    linear_num_value_heads := linearValueHeads
    layer_types := layerTypes
    moe_intermediate_size := ← fieldD? textCfg "moe_intermediate_size" d.moe_intermediate_size
    shared_expert_intermediate_size :=
      ← fieldD? textCfg "shared_expert_intermediate_size" d.shared_expert_intermediate_size
    num_experts_per_tok := perTok
    num_experts := numExperts
    tie_word_embeddings := tie
    eos_token_id := eos
    linearKeyHeadsDivide := hLinear
    kvHeadsDivide := hKV
    headsPos := hHeads
    rotaryValid := by obtain ⟨_, _, _⟩ := hRotary; omega
    layerTypesSize := hLayers
    convKernelPos := hConv
    linearKeyHeadDimPos := hKeyHeadDim
    expertsValid := hExperts
    ropeThetaPos := hTheta
    rmsNormEpsPos := hEps
    eosInVocab := hEos
  }

namespace Config

/-- Load Qwen3.5 config from a HuggingFace-style `config.json` file. -/
def loadFromFile (path : String) (defaults : Config := Config.qwen35_9B) : IO Config := do
  IO.ofExcept (Config.parseJson (← parseFile path) defaults)

/-- Load Qwen3.5 config from a model directory containing `config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : Config := Config.qwen35_9B) : IO Config :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Config

end torch.qwen35
