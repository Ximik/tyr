/-
  Tyr/Model/Gemma4/VLConfigIO.lean

  HuggingFace `config.json` + `processor_config.json` loader for Gemma 4
  multimodal checkpoints.
-/
import Tyr.Model.Gemma4.VLConfig
import Tyr.Model.Gemma4.ConfigIO
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.gemma4

open Lean

private def parseJsonFile (path : String) : IO Json := do
  let contents ← IO.FS.readFile path
  match Json.parse contents with
  | .ok json => pure json
  | .error err => throw (IO.userError s!"Failed to parse JSON at {path}: {err}")

private def maybeParseJsonFile (path : String) : IO (Option Json) := do
  let p : System.FilePath := ⟨path⟩
  if !(← p.pathExists) then
    pure none
  else
    some <$> parseJsonFile path

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

private def getFloatFieldD (j : Json) (key : String) (d : Float) : Float :=
  match getObjVal? j key >>= getFloat? with
  | some x => x
  | none => d

private def getFloatArrayFieldD (j : Json) (key : String) (d : Array Float) : Array Float :=
  match getObjVal? j key with
  | some (.arr xs) =>
    let ys := xs.map getFloat?
    if ys.all Option.isSome then
      ys.map (fun x => x.getD 0.0)
    else
      d
  | _ => d

private def ropeParam? (j : Json) (field : String) : Option Json := do
  let rope ← getObjVal? j "rope_parameters"
  getObjVal? rope field

private def ropeFloatFieldD (j : Json) (field : String) (d : Float) : Float :=
  match ropeParam? j field >>= getFloat? with
  | some x => x
  | none => d

def VisionConfig.parseJson (j : Json) (d : VisionConfig := {}) : VisionConfig :=
  {
    hidden_size := getNatFieldD j "hidden_size" d.hidden_size
    intermediate_size := getNatFieldD j "intermediate_size" d.intermediate_size
    num_hidden_layers := getNatFieldD j "num_hidden_layers" d.num_hidden_layers
    num_attention_heads := getNatFieldD j "num_attention_heads" d.num_attention_heads
    num_key_value_heads := getNatFieldD j "num_key_value_heads" d.num_key_value_heads
    head_dim := getNatFieldD j "head_dim" d.head_dim
    patch_size := getNatFieldD j "patch_size" d.patch_size
    pooling_kernel_size := getNatFieldD j "pooling_kernel_size" d.pooling_kernel_size
    position_embedding_size := getNatFieldD j "position_embedding_size" d.position_embedding_size
    rms_norm_eps := getFloatFieldD j "rms_norm_eps" d.rms_norm_eps
    rope_theta := ropeFloatFieldD j "rope_theta" d.rope_theta
    standardize := getBoolFieldD j "standardize" d.standardize
    use_clipped_linears := getBoolFieldD j "use_clipped_linears" d.use_clipped_linears
  }

def ImageProcessorConfig.parseJson (j : Json) (d : ImageProcessorConfig := {}) : ImageProcessorConfig :=
  {
    do_resize := getBoolFieldD j "do_resize" d.do_resize
    do_rescale := getBoolFieldD j "do_rescale" d.do_rescale
    do_normalize := getBoolFieldD j "do_normalize" d.do_normalize
    image_mean := getFloatArrayFieldD j "image_mean" d.image_mean
    image_std := getFloatArrayFieldD j "image_std" d.image_std
    image_seq_length := getNatFieldD j "image_seq_length" d.image_seq_length
    max_soft_tokens := getNatFieldD j "max_soft_tokens" d.max_soft_tokens
    patch_size := getNatFieldD j "patch_size" d.patch_size
    pooling_kernel_size := getNatFieldD j "pooling_kernel_size" d.pooling_kernel_size
    rescale_factor := getFloatFieldD j "rescale_factor" d.rescale_factor
  }

def VLConfig.parseJson
    (root : Json)
    (processorRoot? : Option Json := none)
    (d : VLConfig := {})
    : VLConfig :=
  let visionJson := match getObjVal? root "vision_config" with | some j => j | none => .null
  let imageProcJson :=
    match processorRoot? with
    | some proc =>
      match getObjVal? proc "image_processor" with
      | some j => j
      | none => .null
    | none => .null
  VLConfig.normalize {
    text_config := Config.parseJson root d.text_config
    vision_config :=
      match visionJson with
      | .null => d.vision_config
      | j => VisionConfig.parseJson j d.vision_config
    image_processor :=
      match imageProcJson with
      | .null => d.image_processor
      | j => ImageProcessorConfig.parseJson j d.image_processor
    boi_token_id := getTokenIdFieldD root "boi_token_id" d.boi_token_id
    image_token_id := getTokenIdFieldD root "image_token_id" d.image_token_id
    eoi_token_id := getTokenIdFieldD root "eoi_token_id" d.eoi_token_id
    video_token_id := getTokenIdFieldD root "video_token_id" d.video_token_id
    audio_token_id := getTokenIdFieldD root "audio_token_id" d.audio_token_id
    vision_soft_tokens_per_image :=
      getNatFieldD root "vision_soft_tokens_per_image" d.vision_soft_tokens_per_image
    tie_word_embeddings := getBoolFieldD root "tie_word_embeddings" d.tie_word_embeddings
  }

namespace VLConfig

def loadFromFiles
    (configPath : String)
    (processorPath : Option String := none)
    (defaults : VLConfig := {})
    : IO VLConfig := do
  let root ← parseJsonFile configPath
  let processorRoot? ←
    match processorPath with
    | some path => maybeParseJsonFile path
    | none => pure none
  pure (VLConfig.parseJson root processorRoot? defaults)

def loadFromPretrainedDir (modelDir : String) (defaults : VLConfig := {}) : IO VLConfig :=
  loadFromFiles
    s!"{modelDir}/config.json"
    (some s!"{modelDir}/processor_config.json")
    defaults

end VLConfig

end torch.gemma4
