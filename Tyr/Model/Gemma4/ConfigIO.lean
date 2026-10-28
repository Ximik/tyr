/-
  Tyr/Model/Gemma4/ConfigIO.lean

  HuggingFace `config.json` loader for standalone Gemma 4 text causal-LM.
-/
import Tyr.Model.Gemma4.Config
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.gemma4

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

private def getNat? (j : Json) : Option Nat := fromJson? j
private def getBool? (j : Json) : Option Bool := fromJson? j
private def getString? (j : Json) : Option String := fromJson? j

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
    let mapped := xs.map (fun x => getString? x >>= LayerType.ofString?)
    if mapped.all Option.isSome then
      some (mapped.map (fun x => x.getD .slidingAttention))
    else
      none
  | _ => none

private def ropeParam? (j : Json) (layerType field : String) : Option Json := do
  let rope ← getObjVal? j "rope_parameters"
  let layerObj ← getObjVal? rope layerType
  getObjVal? layerObj field

private def ropeFloatFieldD (j : Json) (layerType field : String) (d : Float) : Float :=
  match ropeParam? j layerType field >>= getFloat? with
  | some x => x
  | none => d

def Config.parseJson (j : Json) (d : Config := Config.gemma4_E4B) : Config :=
  let textCfg := match getObjVal? j "text_config" with | some t => t | none => j
  let hidden := getNatFieldD textCfg "hidden_size" d.hidden_size
  let heads := getNatFieldD textCfg "num_attention_heads" d.num_attention_heads
  let headDimDefault :=
    if heads == 0 then d.head_dim else hidden / heads
  let layerTypes :=
    match parseLayerTypes textCfg "layer_types" with
    | some xs => xs
    | none => d.layer_types
  let tieWordEmbeddings :=
    match getObjVal? j "tie_word_embeddings" >>= getBool? with
    | some x => x
    | none => getBoolFieldD textCfg "tie_word_embeddings" d.tie_word_embeddings
  Config.normalize {
    vocab_size := getNatFieldD textCfg "vocab_size" d.vocab_size
    hidden_size := hidden
    intermediate_size := getNatFieldD textCfg "intermediate_size" d.intermediate_size
    num_hidden_layers := getNatFieldD textCfg "num_hidden_layers" d.num_hidden_layers
    num_attention_heads := heads
    num_key_value_heads := getNatFieldD textCfg "num_key_value_heads" d.num_key_value_heads
    num_global_key_value_heads := getNatFieldD textCfg "num_global_key_value_heads" d.num_global_key_value_heads
    head_dim := getNatFieldD textCfg "head_dim" headDimDefault
    global_head_dim := getNatFieldD textCfg "global_head_dim" d.global_head_dim

    sliding_window := getNatFieldD textCfg "sliding_window" d.sliding_window
    sliding_rope_theta := ropeFloatFieldD textCfg "sliding_attention" "rope_theta" d.sliding_rope_theta
    full_rope_theta := ropeFloatFieldD textCfg "full_attention" "rope_theta" d.full_rope_theta
    full_partial_rotary_factor :=
      ropeFloatFieldD textCfg "full_attention" "partial_rotary_factor" d.full_partial_rotary_factor
    rms_norm_eps := getFloatFieldD textCfg "rms_norm_eps" d.rms_norm_eps
    max_position_embeddings := getNatFieldD textCfg "max_position_embeddings" d.max_position_embeddings

    attention_bias := getBoolFieldD textCfg "attention_bias" d.attention_bias
    attention_dropout := getFloatFieldD textCfg "attention_dropout" d.attention_dropout
    hidden_activation := getStringFieldD textCfg "hidden_activation" d.hidden_activation
    attention_k_eq_v := getBoolFieldD textCfg "attention_k_eq_v" d.attention_k_eq_v
    use_bidirectional_attention :=
      getStringFieldD textCfg "use_bidirectional_attention" d.use_bidirectional_attention

    layer_types := layerTypes
    full_attention_interval := getNatFieldD textCfg "sliding_window_pattern" d.full_attention_interval
    num_kv_shared_layers := getNatFieldD textCfg "num_kv_shared_layers" d.num_kv_shared_layers
    use_double_wide_mlp := getBoolFieldD textCfg "use_double_wide_mlp" d.use_double_wide_mlp

    hidden_size_per_layer_input :=
      getNatFieldD textCfg "hidden_size_per_layer_input" d.hidden_size_per_layer_input
    vocab_size_per_layer_input :=
      getNatFieldD textCfg "vocab_size_per_layer_input" d.vocab_size_per_layer_input

    enable_moe_block := getBoolFieldD textCfg "enable_moe_block" d.enable_moe_block
    num_experts := getNatFieldD textCfg "num_experts" d.num_experts
    top_k_experts := getNatFieldD textCfg "top_k_experts" d.top_k_experts
    moe_intermediate_size := getNatFieldD textCfg "moe_intermediate_size" d.moe_intermediate_size

    use_cache := getBoolFieldD textCfg "use_cache" d.use_cache
    tie_word_embeddings := tieWordEmbeddings
    final_logit_softcapping :=
      getFloatFieldD textCfg "final_logit_softcapping" d.final_logit_softcapping

    pad_token_id :=
      match getOptTokenIdField textCfg "pad_token_id" with
      | some x => some x
      | none => d.pad_token_id
    bos_token_id :=
      match getOptTokenIdField textCfg "bos_token_id" with
      | some x => some x
      | none => d.bos_token_id
    eos_token_id :=
      match getOptTokenIdField textCfg "eos_token_id" with
      | some x => some x
      | none => d.eos_token_id
  }

namespace Config

def loadFromFile (path : String) (defaults : Config := Config.gemma4_E4B) : IO Config := do
  let root ← parseJsonFile path
  pure (Config.parseJson root defaults)

def loadFromPretrainedDir (modelDir : String) (defaults : Config := Config.gemma4_E4B) : IO Config :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Config

end torch.gemma4
