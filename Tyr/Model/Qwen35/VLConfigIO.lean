/-
  Tyr/Model/Qwen35/VLConfigIO.lean

  HuggingFace `config.json` loader for Qwen3.5 multimodal checkpoints.
-/
import Tyr.Model.Qwen35.ConfigIO
import Tyr.Model.Qwen35.VLConfig
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

private def getTokenIdFieldD (j : Json) (key : String) (d : UInt64) : UInt64 :=
  match getObjVal? j key >>= getNat? with
  | some n => n.toUInt64
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

private def parseVisionConfig
    (j : Json)
    (d : VisionConfig := {})
    : Except String VisionConfig :=
  let hidden_size := getNatFieldD j "hidden_size" d.hidden_size
  let num_heads := getNatFieldD j "num_heads" d.num_heads
  if headsDivide : num_heads ∣ hidden_size then .ok {
    depth := getNatFieldD j "depth" d.depth
    hidden_size
    hidden_act := getStringFieldD j "hidden_act" d.hidden_act
    intermediate_size := getNatFieldD j "intermediate_size" d.intermediate_size
    num_heads
    in_channels := getNatFieldD j "in_channels" d.in_channels
    patch_size := getNatFieldD j "patch_size" d.patch_size
    spatial_merge_size := getNatFieldD j "spatial_merge_size" d.spatial_merge_size
    temporal_patch_size := getNatFieldD j "temporal_patch_size" d.temporal_patch_size
    out_hidden_size := getNatFieldD j "out_hidden_size" d.out_hidden_size
    num_position_embeddings := getNatFieldD j "num_position_embeddings" d.num_position_embeddings
    initializer_range := getFloatFieldD j "initializer_range" d.initializer_range
    headsDivide
  }
  else .error s!"Invalid Qwen3.5 vision config: hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"

private def parseVLConfig
    (root : Json)
    (d : VLConfig := {})
    : Except String VLConfig := do
  let textJson :=
    match getObjVal? root "text_config" with
    | some t => t
    | none => root
  let textCfg ←
    match getObjVal? root "text_config" with
    | some _ =>
      pure (Config.normalize (← Config.parseJson textJson d.text_config))
    | none =>
      -- Backward compatible: plain text-only config at root.
      pure (Config.normalize (← Config.parseJson root d.text_config))

  let visionCfg ←
    match getObjVal? root "vision_config" with
    | some v => parseVisionConfig v d.vision_config
    | none => pure d.vision_config

  let tieWordEmbeddings :=
    match getObjVal? root "tie_word_embeddings" >>= getBool? with
    | some x => x
    | none => textCfg.tie_word_embeddings

  pure <| VLConfig.normalize {
    text_config := textCfg
    vision_config := visionCfg
    image_token_id := getTokenIdFieldD root "image_token_id" d.image_token_id
    video_token_id := getTokenIdFieldD root "video_token_id" d.video_token_id
    vision_start_token_id := getTokenIdFieldD root "vision_start_token_id" d.vision_start_token_id
    vision_end_token_id := getTokenIdFieldD root "vision_end_token_id" d.vision_end_token_id
    tie_word_embeddings := tieWordEmbeddings
  }

namespace VLConfig

/-- Load Qwen3.5 multimodal config from HuggingFace-style `config.json`. -/
def loadFromFile (path : String) (defaults : VLConfig := {}) : IO VLConfig := do
  let root ← parseJsonFile path
  IO.ofExcept (parseVLConfig root defaults)

/-- Load Qwen3.5 multimodal config from a model directory containing `config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : VLConfig := {}) : IO VLConfig :=
  loadFromFile s!"{modelDir}/config.json" defaults

end VLConfig

end torch.qwen35
