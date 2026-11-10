/-
  Tyr/Model/Laguna/ConfigIO.lean

  HuggingFace `config.json` loader for poolside Laguna causal-LM.
-/
import Tyr.Model.Laguna.Config
import Tyr.Model.ConfigJson

namespace torch.laguna

open Lean ConfigJson

private def layerType? (name : String) : Except String LayerType :=
  match LayerType.ofString? name with
  | some t => pure t
  | none => throw s!"unknown layer type \"{name}\"; expected full_attention or sliding_attention"

/-- `rope_parameters.<section>.<field>`, `none` when absent. -/
private def ropeField? {α : Type} [FromJson α] (j : Json) (layerKind field : String) : Except String (Option α) := do
  let some rope ← objectField? j "rope_parameters" | pure none
  let some params ← objectField? rope layerKind | pure none
  ConfigJson.field? params field

/-- Rotary channels `headDim * factor`, which must be a positive even whole number of channels. -/
private def rotaryDim? (headDim : Nat) (factor : Float) (kind : String) : Except String Nat := do
  let r := (headDim.toFloat * factor).round.toUInt64.toNat
  unless r.toFloat == headDim.toFloat * factor do
    throw s!"Invalid Laguna config: {kind} partial_rotary_factor={factor} does not select a whole number of the {headDim} head channels"
  pure r

/-- Parse a Laguna `config.json` root. Behavior the model hard-codes (SiLU MLP, no
attention bias) is required of the file. -/
def Config.parseJson (j : Json) (d : Config := Config.laguna_s_2_1) : Except String Config := do
  requireOneOf j "hidden_act" ["silu"]
  requireEq j "attention_bias" false
  let numLayers ← fieldD? j "num_hidden_layers" d.num_hidden_layers
  let layerTypes ← match ← field? j "layer_types" with
    | some (names : Array String) => names.mapM layerType?
    | none => pure (defaultLayerTypes numLayers)
  let ⟨hLayers⟩ ← if h : layerTypes.size = numLayers then pure (PLift.up h) else
    throw s!"Invalid Laguna config: layer_types has {layerTypes.size} entries for num_hidden_layers={numLayers}"
  -- Per-layer head counts, which must agree within each attention kind.
  let mut numHeadsFull ← fieldD? j "num_attention_heads" d.num_attention_heads
  let mut numHeadsSliding := d.num_attention_heads_sliding
  if let some (perLayer : Array Nat) ← field? j "num_attention_heads_per_layer" then
    unless perLayer.size = numLayers do
      throw s!"Invalid Laguna config: num_attention_heads_per_layer has {perLayer.size} entries for {numLayers} layers"
    for kind in [LayerType.fullAttention, .slidingAttention] do
      let counts := ((layerTypes.zip perLayer).filter (·.1 == kind)).map (·.2)
      if let some first := counts[0]? then
        unless counts.all (· == first) do
          throw s!"Invalid Laguna config: num_attention_heads_per_layer varies within {kind.toString} layers"
        if kind == .fullAttention then numHeadsFull := first else numHeadsSliding := first
  let kvHeads ← fieldD? j "num_key_value_heads" d.num_key_value_heads
  let ⟨hHeads⟩ ← if h : 0 < numHeadsFull ∧ 0 < numHeadsSliding ∧ 0 < kvHeads then pure (PLift.up h) else
    throw s!"Invalid Laguna config: head counts ({numHeadsFull}, {numHeadsSliding}, kv {kvHeads}) must be positive"
  let ⟨hFull⟩ ← if h : kvHeads ∣ numHeadsFull then pure (PLift.up h) else
    throw s!"Invalid Laguna config: num_attention_heads ({numHeadsFull}) must be divisible by num_key_value_heads ({kvHeads})"
  let ⟨hSliding⟩ ← if h : kvHeads ∣ numHeadsSliding then pure (PLift.up h) else
    throw s!"Invalid Laguna config: sliding-layer heads ({numHeadsSliding}) must be divisible by num_key_value_heads ({kvHeads})"
  let slidingWindow ← fieldD? j "sliding_window" d.sliding_window
  let ⟨hWindow⟩ ← if h : 0 < slidingWindow then pure (PLift.up h) else
    throw "Invalid Laguna config: sliding_window must be positive"
  let headDim ← fieldD? j "head_dim" d.head_dim
  let fullFactor := (← ropeField? j "full_attention" "partial_rotary_factor").getD
    (d.rotaryDimFull.toFloat / d.head_dim.toFloat)
  let slidingFactor := (← ropeField? j "sliding_attention" "partial_rotary_factor").getD
    (d.rotaryDimSliding.toFloat / d.head_dim.toFloat)
  let rotaryFull ← rotaryDim? headDim fullFactor "full_attention"
  let rotarySliding ← rotaryDim? headDim slidingFactor "sliding_attention"
  let ⟨hRotary⟩ ← if h : 0 < rotaryFull ∧ 2 ∣ rotaryFull ∧ rotaryFull ≤ headDim ∧
      0 < rotarySliding ∧ 2 ∣ rotarySliding ∧ rotarySliding ≤ headDim then pure (PLift.up h) else
    throw s!"Invalid Laguna config: rotary dims ({rotaryFull}, {rotarySliding}) must be positive, even and at most head_dim={headDim}"
  let mlpOnly ← fieldD? j "mlp_only_layers" d.mlp_only_layers
  let ⟨hMlpOnly⟩ ← if h : ∀ i ∈ mlpOnly, i < numLayers then pure (PLift.up h) else
    throw s!"Invalid Laguna config: mlp_only_layers {mlpOnly} must name layers below {numLayers}"
  let numExperts ← fieldD? j "num_experts" d.num_experts
  let perTok ← fieldD? j "num_experts_per_tok" d.num_experts_per_tok
  let ⟨hExperts⟩ ← if h : 0 < numExperts → 0 < perTok ∧ perTok ≤ numExperts then pure (PLift.up h) else
    throw s!"Invalid Laguna config: num_experts_per_tok={perTok} must be between 1 and num_experts={numExperts}"
  let yarnContext := (← ropeField? j "full_attention" "original_max_position_embeddings").getD
    d.yarn_original_max_position_embeddings
  let ⟨hYarnContext⟩ ← if h : 0 < yarnContext then pure (PLift.up h) else
    throw "Invalid Laguna config: original_max_position_embeddings must be positive"
  let eps ← fieldD? j "rms_norm_eps" d.rms_norm_eps
  let thetaSliding := (← ropeField? j "sliding_attention" "rope_theta").getD d.rope_theta_sliding
  let thetaFull ← match ← ropeField? j "full_attention" "rope_theta" with
    | some theta => pure theta
    | none => fieldD? j "rope_theta" d.rope_theta_full
  let yarnFactor := (← ropeField? j "full_attention" "factor").getD d.yarn_factor
  let ⟨hFloats⟩ ← if h : 0 < eps ∧ 0 < thetaSliding ∧ 0 < thetaFull ∧ 0 < yarnFactor then pure (PLift.up h) else
    throw s!"Invalid Laguna config: rms_norm_eps={eps}, rope thetas ({thetaSliding}, {thetaFull}) and yarn factor={yarnFactor} must be positive"
  let vocab ← fieldD? j "vocab_size" d.vocab_size
  let eos ← match ← member? j "eos_token_id" with
    | none | some .null => pure d.eos_token_ids
    | some (.arr _) => tokenIdArrayFieldD? j "eos_token_id" d.eos_token_ids
    | some _ => pure #[← tokenIdFieldD? j "eos_token_id" 0]
  let ⟨hEos⟩ ← if h : eos.all (·.toNat < vocab) = true then pure (PLift.up h) else
    throw s!"Invalid Laguna config: eos_token_id must be below vocab_size={vocab}"
  pure {
    vocab_size := vocab
    hidden_size := ← fieldD? j "hidden_size" d.hidden_size
    intermediate_size := ← fieldD? j "intermediate_size" d.intermediate_size
    num_hidden_layers := numLayers
    num_attention_heads := numHeadsFull
    num_attention_heads_sliding := numHeadsSliding
    num_key_value_heads := kvHeads
    head_dim := headDim
    rms_norm_eps := eps
    sliding_window := slidingWindow
    rope_theta_sliding := thetaSliding
    half_rotary_dim_sliding := rotarySliding / 2
    rope_theta_full := thetaFull
    half_rotary_dim_full := rotaryFull / 2
    yarn_factor := yarnFactor
    yarn_original_max_position_embeddings := yarnContext
    yarn_beta_fast := (← ropeField? j "full_attention" "beta_fast").getD d.yarn_beta_fast
    yarn_beta_slow := (← ropeField? j "full_attention" "beta_slow").getD d.yarn_beta_slow
    yarn_attention_factor := (← ropeField? j "full_attention" "attention_factor").getD d.yarn_attention_factor
    num_experts := numExperts
    num_experts_per_tok := perTok
    moe_intermediate_size := ← fieldD? j "moe_intermediate_size" d.moe_intermediate_size
    shared_expert_intermediate_size := ← fieldD? j "shared_expert_intermediate_size" d.shared_expert_intermediate_size
    norm_topk_prob := ← fieldD? j "norm_topk_prob" d.norm_topk_prob
    moe_routed_scaling_factor := ← fieldD? j "moe_routed_scaling_factor" d.moe_routed_scaling_factor
    moe_router_logit_softcapping := ← fieldD? j "moe_router_logit_softcapping" d.moe_router_logit_softcapping
    mlp_only_layers := mlpOnly
    layer_types := layerTypes
    eos_token_ids := eos
    slidingWindowPos := hWindow
    kvHeadsDivideFull := hFull
    kvHeadsDivideSliding := hSliding
    headsPos := hHeads
    rotaryValid := by obtain ⟨_, _, _, _, _, _⟩ := hRotary; omega
    layerTypesSize := hLayers
    mlpOnlyLayersValid := hMlpOnly
    expertsValid := hExperts
    yarnContextPos := hYarnContext
    floatsPos := hFloats
    eosInVocab := hEos
  }

/-- Load a Laguna config from a JSON file path. -/
def Config.loadFromFile (path : String) (defaults : Config := Config.laguna_s_2_1) : IO Config := do
  IO.ofExcept (Config.parseJson (← parseFile path) defaults)

/-- Load a Laguna config from a pretrained model directory. -/
def Config.loadFromPretrainedDir (modelDir : String) (defaults : Config := Config.laguna_s_2_1) : IO Config :=
  Config.loadFromFile (modelDir ++ "/config.json") defaults

end torch.laguna
