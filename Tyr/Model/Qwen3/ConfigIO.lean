/-
  Tyr/Model/Qwen3/ConfigIO.lean

  HuggingFace `config.json` loader for standalone Qwen3 causal-LM.
-/
import Tyr.Model.Qwen3.Config
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.qwen3

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

private def parseConfig (j : Json) (d : Config := Config.qwen3_4B) : Config :=
  let hidden := getNatFieldD j "hidden_size" d.hidden_size
  let heads := getNatFieldD j "num_attention_heads" d.num_attention_heads
  let headDimDefault :=
    if heads == 0 then d.head_dim else hidden / heads
  {
    vocab_size := getNatFieldD j "vocab_size" d.vocab_size
    hidden_size := hidden
    intermediate_size := getNatFieldD j "intermediate_size" d.intermediate_size
    num_hidden_layers := getNatFieldD j "num_hidden_layers" d.num_hidden_layers
    num_attention_heads := heads
    num_key_value_heads := getNatFieldD j "num_key_value_heads" d.num_key_value_heads
    head_dim := getNatFieldD j "head_dim" headDimDefault
    rope_theta := getFloatFieldD j "rope_theta" d.rope_theta
    rms_norm_eps := getFloatFieldD j "rms_norm_eps" d.rms_norm_eps
    max_position_embeddings := getNatFieldD j "max_position_embeddings" d.max_position_embeddings
  }

private def validateConfig (cfg : Config) : IO Unit := do
  if cfg.vocab_size == 0 then
    throw <| IO.userError "Invalid Qwen3 config: vocab_size must be > 0"
  if cfg.hidden_size == 0 then
    throw <| IO.userError "Invalid Qwen3 config: hidden_size must be > 0"
  if cfg.intermediate_size == 0 then
    throw <| IO.userError "Invalid Qwen3 config: intermediate_size must be > 0"
  if cfg.num_hidden_layers == 0 then
    throw <| IO.userError "Invalid Qwen3 config: num_hidden_layers must be > 0"
  if cfg.num_attention_heads == 0 then
    throw <| IO.userError "Invalid Qwen3 config: num_attention_heads must be > 0"
  if cfg.num_key_value_heads == 0 then
    throw <| IO.userError "Invalid Qwen3 config: num_key_value_heads must be > 0"
  if cfg.hidden_size % cfg.num_attention_heads != 0 then
    throw <| IO.userError
      s!"Invalid Qwen3 config: hidden_size ({cfg.hidden_size}) must be divisible by num_attention_heads ({cfg.num_attention_heads})"
  if cfg.num_attention_heads % cfg.num_key_value_heads != 0 then
    throw <| IO.userError
      s!"Invalid Qwen3 config: num_attention_heads ({cfg.num_attention_heads}) must be divisible by num_key_value_heads ({cfg.num_key_value_heads})"
  if cfg.head_dim == 0 then
    throw <| IO.userError "Invalid Qwen3 config: head_dim must be > 0"
  if cfg.max_position_embeddings == 0 then
    throw <| IO.userError "Invalid Qwen3 config: max_position_embeddings must be > 0"

namespace Config

/-- Load Qwen3 config from a HuggingFace-style `config.json` file. -/
def loadFromFile (path : String) (defaults : Config := Config.qwen3_4B) : IO Config := do
  let root ← parseJsonFile path
  let cfg := parseConfig root defaults
  validateConfig cfg
  pure cfg

/-- Load Qwen3 config from a model directory containing `config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : Config := Config.qwen3_4B) : IO Config :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Config

end torch.qwen3
