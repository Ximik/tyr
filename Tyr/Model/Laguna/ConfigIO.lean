/-
  Tyr/Model/Laguna/ConfigIO.lean

  HuggingFace `config.json` loader for poolside Laguna causal-LM.
-/
import Tyr.Model.Laguna.Config
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.laguna

open Lean

private def parseJsonFile (path : String) : IO Json := do
  let contents ← IO.FS.readFile path
  match Json.parse contents with
  | .ok json => pure json
  | .error err => throw (IO.userError s!"Failed to parse JSON at {path}: {err}")

private def getObjVal? (j : Json) (key : String) : Option Json :=
  match j with
  | .obj kvs => Std.TreeMap.Raw.get? kvs key
  | _ => none

private def fromJson? {α} [FromJson α] (j : Json) : Option α :=
  match (FromJson.fromJson? j : Except String α) with
  | .ok v => some v
  | .error _ => none

private def getNat? (j : Json) : Option Nat :=
  fromJson? j

private def getBool? (j : Json) : Option Bool :=
  fromJson? j

private def getString? (j : Json) : Option String :=
  fromJson? j

private def getNatArray? (j : Json) : Option (Array Nat) :=
  match j with
  | .arr xs =>
    let mapped := xs.map getNat?
    if mapped.all Option.isSome then
      some <| mapped.map (fun x => x.getD 0)
    else
      none
  | _ => none

private def getTokenIdArray? (j : Json) : Option (Array UInt64) :=
  (getNatArray? j).map (·.map (·.toUInt64))

private def getFloat? (j : Json) : Option Float :=
  match (fromJson? (α := Float) j) with
  | some x => some x
  | none => (getNat? j).map (·.toFloat)

private def getNatFieldD (j : Json) (key : String) (d : Nat) : Nat :=
  match getObjVal? j key >>= getNat? with
  | some n => n
  | none => d

private def getBoolFieldD (j : Json) (key : String) (d : Bool) : Bool :=
  match getObjVal? j key >>= getBool? with
  | some b => b
  | none => d

private def getStringFieldD (j : Json) (key : String) (d : String) : String :=
  match getObjVal? j key >>= getString? with
  | some s => s
  | none => d

private def getFloatFieldD (j : Json) (key : String) (d : Float) : Float :=
  match getObjVal? j key >>= getFloat? with
  | some x => x
  | none => d

private def getOptTokenIdField (j : Json) (key : String) : Option UInt64 :=
  (getObjVal? j key >>= getNat?).map (·.toUInt64)

private def parseLayerTypes (j : Json) (key : String) : Option (Array LayerType) :=
  match getObjVal? j key with
  | some (.arr xs) =>
    let mapped := xs.map (fun x => (getString? x >>= LayerType.ofString?))
    if mapped.all Option.isSome then
      some <| mapped.map (fun x => x.getD .slidingAttention)
    else
      none
  | _ => none

/-- `rope_parameters.<sec>.<field>` nested float lookup. -/
private def ropeFloatField? (j : Json) (sec fieldKey : String) : Option Float :=
  getObjVal? j "rope_parameters"
    >>= fun rp => getObjVal? rp sec
    >>= fun nested => getObjVal? nested fieldKey >>= getFloat?

/-- `rope_parameters.<sec>.<field>` nested nat lookup. -/
private def ropeNatField? (j : Json) (sec fieldKey : String) : Option Nat :=
  getObjVal? j "rope_parameters"
    >>= fun rp => getObjVal? rp sec
    >>= fun nested => getObjVal? nested fieldKey >>= getNat?

def Config.parseJson (j : Json) (d : Config := Config.laguna_s_2_1) : Config :=
  let layerTypes :=
    match parseLayerTypes j "layer_types" with
    | some xs => xs
    | none => d.layer_types

  -- Per-layer head counts: take the count of the first full-attention layer and
  -- the first sliding-attention layer from `num_attention_heads_per_layer`.
  let perLayerHeads := getObjVal? j "num_attention_heads_per_layer" >>= getNatArray?
  let headsFor (pred : LayerType → Bool) (fallback : Nat) : Nat :=
    match perLayerHeads with
    | some hs =>
      let lts := if layerTypes.isEmpty then LagunaConfig.defaultLayerTypes hs.size else layerTypes
      let hits := (lts.zip hs).filter (fun (lt, _) => pred lt)
      hits[0]?.map (·.2) |>.getD fallback
    | none => fallback
  let numHeadsFull := headsFor (· == .fullAttention) d.num_attention_heads
  let numHeadsSliding := headsFor (· == .slidingAttention) d.num_attention_heads_sliding

  let ropeThetaFull :=
    match ropeFloatField? j "full_attention" "rope_theta" with
    | some x => x
    | none => getFloatFieldD j "rope_theta" d.rope_theta_full
  let partialFull :=
    match ropeFloatField? j "full_attention" "partial_rotary_factor" with
    | some x => x
    | none => d.partial_rotary_full
  let ropeThetaSliding :=
    match ropeFloatField? j "sliding_attention" "rope_theta" with
    | some x => x
    | none => d.rope_theta_sliding
  let partialSliding :=
    match ropeFloatField? j "sliding_attention" "partial_rotary_factor" with
    | some x => x
    | none => d.partial_rotary_sliding

  let eosIds :=
    match getObjVal? j "eos_token_id" with
    | some (.arr _) => (getTokenIdArray? (j.getObjValD "eos_token_id")).getD d.eos_token_ids
    | some _ => (getOptTokenIdField j "eos_token_id").map (fun e => #[e]) |>.getD d.eos_token_ids
    | none => d.eos_token_ids

  {
    vocab_size := getNatFieldD j "vocab_size" d.vocab_size
    hidden_size := getNatFieldD j "hidden_size" d.hidden_size
    intermediate_size := getNatFieldD j "intermediate_size" d.intermediate_size
    num_hidden_layers := getNatFieldD j "num_hidden_layers" d.num_hidden_layers
    num_attention_heads := getNatFieldD j "num_attention_heads" numHeadsFull
    num_attention_heads_sliding := numHeadsSliding
    num_key_value_heads := getNatFieldD j "num_key_value_heads" d.num_key_value_heads
    head_dim := getNatFieldD j "head_dim" d.head_dim

    rms_norm_eps := getFloatFieldD j "rms_norm_eps" d.rms_norm_eps
    max_position_embeddings := getNatFieldD j "max_position_embeddings" d.max_position_embeddings
    sliding_window := getNatFieldD j "sliding_window" d.sliding_window

    rope_theta_sliding := ropeThetaSliding
    partial_rotary_sliding := partialSliding
    rope_theta_full := ropeThetaFull
    partial_rotary_full := partialFull
    yarn_factor := ropeFloatField? j "full_attention" "factor" |>.getD d.yarn_factor
    yarn_original_max_position_embeddings :=
      ropeNatField? j "full_attention" "original_max_position_embeddings"
        |>.getD d.yarn_original_max_position_embeddings
    yarn_beta_fast := ropeFloatField? j "full_attention" "beta_fast" |>.getD d.yarn_beta_fast
    yarn_beta_slow := ropeFloatField? j "full_attention" "beta_slow" |>.getD d.yarn_beta_slow
    yarn_attention_factor :=
      ropeFloatField? j "full_attention" "attention_factor" |>.getD d.yarn_attention_factor

    num_experts := getNatFieldD j "num_experts" d.num_experts
    num_experts_per_tok := getNatFieldD j "num_experts_per_tok" d.num_experts_per_tok
    moe_intermediate_size := getNatFieldD j "moe_intermediate_size" d.moe_intermediate_size
    shared_expert_intermediate_size :=
      getNatFieldD j "shared_expert_intermediate_size" d.shared_expert_intermediate_size
    norm_topk_prob := getBoolFieldD j "norm_topk_prob" d.norm_topk_prob
    moe_routed_scaling_factor :=
      getFloatFieldD j "moe_routed_scaling_factor" d.moe_routed_scaling_factor
    moe_router_logit_softcapping :=
      getFloatFieldD j "moe_router_logit_softcapping" d.moe_router_logit_softcapping

    mlp_only_layers :=
      match getObjVal? j "mlp_only_layers" >>= getNatArray? with
      | some xs => xs
      | none => d.mlp_only_layers
    layer_types := layerTypes

    attention_bias := getBoolFieldD j "attention_bias" d.attention_bias
    hidden_act := getStringFieldD j "hidden_act" d.hidden_act
    use_cache := getBoolFieldD j "use_cache" d.use_cache
    tie_word_embeddings := getBoolFieldD j "tie_word_embeddings" d.tie_word_embeddings

    pad_token_id := getOptTokenIdField j "pad_token_id" <|> d.pad_token_id
    bos_token_id := getOptTokenIdField j "bos_token_id" <|> d.bos_token_id
    eos_token_ids := eosIds
  }

/-- Load and normalize a Laguna config from a JSON file path. -/
def Config.loadFromFile (path : String) (defaults : Config := Config.laguna_s_2_1) : IO Config := do
  let j ← parseJsonFile path
  pure (Config.normalize (Config.parseJson j defaults))

/-- Load and normalize a Laguna config from a pretrained model directory. -/
def Config.loadFromPretrainedDir (modelDir : String) (defaults : Config := Config.laguna_s_2_1) : IO Config :=
  Config.loadFromFile (modelDir ++ "/config.json") defaults

end torch.laguna
