/-  
  Tyr/Model/Qwen25Omni/ConfigIO.lean

  HuggingFace `config.json` loader for Qwen2.5-Omni thinker text checkpoints.
-/
import Tyr.Model.Qwen25Omni.Config
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.qwen25omni

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

private def getFloat? (j : Json) : Option Float :=
  match (fromJson? (α := Float) j) with
  | some x => some x
  | none => (getNat? j).map (·.toFloat)

private def getNatFieldD (j : Json) (key : String) (d : Nat) : Nat :=
  match getObjVal? j key >>= getNat? with
  | some n => n
  | none => d

private def getFloatFieldD (j : Json) (key : String) (d : Float) : Float :=
  match getObjVal? j key >>= getFloat? with
  | some x => x
  | none => d

private def parseTextConfig (root : Json) (defaults : Config) : Config :=
  let textJson :=
    match getObjVal? root "thinker_config" with
    | some thinker =>
      match getObjVal? thinker "text_config" with
      | some t => t
      | none => thinker
    | none =>
      match getObjVal? root "text_config" with
      | some t => t
      | none => root

  let hidden := getNatFieldD textJson "hidden_size" defaults.hidden_size
  let heads := getNatFieldD textJson "num_attention_heads" defaults.num_attention_heads
  let headDimDefault :=
    if heads == 0 then defaults.head_dim else hidden / heads

  {
    vocab_size := getNatFieldD textJson "vocab_size" defaults.vocab_size
    hidden_size := hidden
    intermediate_size := getNatFieldD textJson "intermediate_size" defaults.intermediate_size
    num_hidden_layers := getNatFieldD textJson "num_hidden_layers" defaults.num_hidden_layers
    num_attention_heads := heads
    num_key_value_heads := getNatFieldD textJson "num_key_value_heads" defaults.num_key_value_heads
    head_dim := getNatFieldD textJson "head_dim" headDimDefault
    rope_theta := getFloatFieldD textJson "rope_theta" defaults.rope_theta
    rms_norm_eps := getFloatFieldD textJson "rms_norm_eps" defaults.rms_norm_eps
    max_position_embeddings := getNatFieldD textJson "max_position_embeddings" defaults.max_position_embeddings
  }

namespace Config

/-- Load Qwen2.5-Omni thinker text config from a HuggingFace-style `config.json`. -/
def loadFromFile (path : String) (defaults : Config := Config.qwen25omni_3B) : IO Config := do
  let root ← parseJsonFile path
  pure (parseTextConfig root defaults)

/-- Load Qwen2.5-Omni thinker text config from `modelDir/config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : Config := Config.qwen25omni_3B) : IO Config :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Config

end torch.qwen25omni

