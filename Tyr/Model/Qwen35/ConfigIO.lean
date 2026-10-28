/-
  Tyr/Model/Qwen35/ConfigIO.lean

  HuggingFace `config.json` loader for standalone Qwen3.5 text causal-LM.
-/
import Tyr.Model.Qwen35.Config
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.qwen35

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

private def getOptNatField (j : Json) (key : String) : Option Nat :=
  getObjVal? j key >>= getNat?

private def parseLayerTypes (j : Json) (key : String) : Option (Array LayerType) :=
  match getObjVal? j key with
  | some (.arr xs) =>
    let mapped := xs.map (fun x => (getString? x >>= LayerType.ofString?))
    if mapped.all Option.isSome then
      some <| mapped.map (fun x => x.getD .linearAttention)
    else
      none
  | _ => none

private def nestedFloatField? (j : Json) (objKey fieldKey : String) : Option Float :=
  getObjVal? j objKey >>= fun nested => getObjVal? nested fieldKey >>= getFloat?

def Config.parseJson (j : Json) (d : Config := Config.qwen35_9B) : Config :=
  let textCfg := match getObjVal? j "text_config" with | some t => t | none => j

  let hidden := getNatFieldD textCfg "hidden_size" d.hidden_size
  let heads := getNatFieldD textCfg "num_attention_heads" d.num_attention_heads
  let headDimDefault :=
    if heads == 0 then d.head_dim else hidden / heads
  let layerTypes :=
    match parseLayerTypes textCfg "layer_types" with
    | some xs => xs
    | none => d.layer_types

  let ropeTheta :=
    match nestedFloatField? textCfg "rope_parameters" "rope_theta" with
    | some x => x
    | none => getFloatFieldD textCfg "rope_theta" d.rope_theta

  let partialRotary :=
    match nestedFloatField? textCfg "rope_parameters" "partial_rotary_factor" with
    | some x => x
    | none => getFloatFieldD textCfg "partial_rotary_factor" d.partial_rotary_factor

  let tieWordEmbeddings :=
    match getObjVal? j "tie_word_embeddings" >>= getBool? with
    | some x => x
    | none => getBoolFieldD textCfg "tie_word_embeddings" d.tie_word_embeddings

  let mropeInterleaved :=
    match getObjVal? textCfg "rope_parameters" >>= (fun nested => getObjVal? nested "mrope_interleaved") >>= getBool? with
    | some x => x
    | none => getBoolFieldD textCfg "mrope_interleaved" d.mrope_interleaved

  let mropeSection :=
    match getObjVal? textCfg "rope_parameters" >>= (fun nested => getObjVal? nested "mrope_section") >>= getNatArray? with
    | some xs => xs
    | none =>
      match getObjVal? textCfg "mrope_section" >>= getNatArray? with
      | some xs => xs
      | none => d.mrope_section

  let mlpOnlyLayers :=
    match getObjVal? textCfg "mlp_only_layers" >>= getNatArray? with
    | some xs => xs
    | none => d.mlp_only_layers

  {
    vocab_size := getNatFieldD textCfg "vocab_size" d.vocab_size
    hidden_size := hidden
    intermediate_size := getNatFieldD textCfg "intermediate_size" d.intermediate_size
    num_hidden_layers := getNatFieldD textCfg "num_hidden_layers" d.num_hidden_layers
    num_attention_heads := heads
    num_key_value_heads := getNatFieldD textCfg "num_key_value_heads" d.num_key_value_heads
    head_dim := getNatFieldD textCfg "head_dim" headDimDefault

    rope_theta := ropeTheta
    partial_rotary_factor := partialRotary
    rms_norm_eps := getFloatFieldD textCfg "rms_norm_eps" d.rms_norm_eps
    max_position_embeddings := getNatFieldD textCfg "max_position_embeddings" d.max_position_embeddings

    attention_bias := getBoolFieldD textCfg "attention_bias" d.attention_bias
    attention_dropout := getFloatFieldD textCfg "attention_dropout" d.attention_dropout
    attn_output_gate := getBoolFieldD textCfg "attn_output_gate" d.attn_output_gate
    hidden_act := getStringFieldD textCfg "hidden_act" d.hidden_act

    linear_conv_kernel_dim := getNatFieldD textCfg "linear_conv_kernel_dim" d.linear_conv_kernel_dim
    linear_key_head_dim := getNatFieldD textCfg "linear_key_head_dim" d.linear_key_head_dim
    linear_value_head_dim := getNatFieldD textCfg "linear_value_head_dim" d.linear_value_head_dim
    linear_num_key_heads := getNatFieldD textCfg "linear_num_key_heads" d.linear_num_key_heads
    linear_num_value_heads := getNatFieldD textCfg "linear_num_value_heads" d.linear_num_value_heads
    mamba_ssm_dtype := getStringFieldD textCfg "mamba_ssm_dtype" d.mamba_ssm_dtype

    layer_types := layerTypes
    full_attention_interval := getNatFieldD textCfg "full_attention_interval" d.full_attention_interval
    mlp_only_layers := mlpOnlyLayers
    mrope_interleaved := mropeInterleaved
    mrope_section := mropeSection
    mtp_num_hidden_layers := getNatFieldD textCfg "mtp_num_hidden_layers" d.mtp_num_hidden_layers
    mtp_use_dedicated_embeddings :=
      getBoolFieldD textCfg "mtp_use_dedicated_embeddings" d.mtp_use_dedicated_embeddings

    moe_intermediate_size := getNatFieldD textCfg "moe_intermediate_size" d.moe_intermediate_size
    shared_expert_intermediate_size :=
      getNatFieldD textCfg "shared_expert_intermediate_size" d.shared_expert_intermediate_size
    num_experts_per_tok := getNatFieldD textCfg "num_experts_per_tok" d.num_experts_per_tok
    num_experts := getNatFieldD textCfg "num_experts" d.num_experts

    use_cache := getBoolFieldD textCfg "use_cache" d.use_cache
    tie_word_embeddings := tieWordEmbeddings

    pad_token_id :=
      match getOptNatField textCfg "pad_token_id" with
      | some x => some x.toUInt64
      | none => d.pad_token_id
    bos_token_id :=
      match getOptNatField textCfg "bos_token_id" with
      | some x => some x.toUInt64
      | none => d.bos_token_id
    eos_token_id :=
      match getOptNatField textCfg "eos_token_id" with
      | some x => some x.toUInt64
      | none => d.eos_token_id
  }

namespace Config

/-- Load Qwen3.5 config from a HuggingFace-style `config.json` file. -/
def loadFromFile (path : String) (defaults : Config := Config.qwen35_9B) : IO Config := do
  let root ← parseJsonFile path
  pure (Config.normalize (Config.parseJson root defaults))

/-- Load Qwen3.5 config from a model directory containing `config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : Config := Config.qwen35_9B) : IO Config :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Config

end torch.qwen35
