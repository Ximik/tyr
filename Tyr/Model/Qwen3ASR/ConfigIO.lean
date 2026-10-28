/-
  Tyr/Model/Qwen3ASR/ConfigIO.lean

  JSON config loader for Qwen3-ASR HuggingFace `config.json`.
-/
import Tyr.Model.Qwen3ASR.Config
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.qwen3asr

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

private def getStr? (j : Json) : Option String :=
  match j with
  | .str s => some s
  | _ => none

private def getBool? (j : Json) : Option Bool :=
  match j with
  | .bool b => some b
  | _ => none

private def getArr? (j : Json) : Option (Array Json) :=
  match j with
  | .arr a => some a
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

private def getTokenIdFieldD (j : Json) (key : String) (d : UInt64) : UInt64 :=
  match getObjVal? j key >>= getNat? with
  | some n => n.toUInt64
  | none => d

private def getFloatFieldD (j : Json) (key : String) (d : Float) : Float :=
  match getObjVal? j key >>= getFloat? with
  | some x => x
  | none => d

private def getBoolFieldD (j : Json) (key : String) (d : Bool) : Bool :=
  match getObjVal? j key >>= getBool? with
  | some x => x
  | none => d

private def getStrFieldD (j : Json) (key : String) (d : String) : String :=
  match getObjVal? j key >>= getStr? with
  | some x => x
  | none => d

private def getStringArrayFieldD (j : Json) (key : String) (d : Array String) : Array String :=
  match getObjVal? j key >>= getArr? with
  | some arr =>
    Id.run do
      let mut out : Array String := #[]
      for item in arr do
        match getStr? item with
        | some s => out := out.push s
        | none => pure ()
      if out.isEmpty then d else out
  | none => d

private def parseAudioConfig (j : Json) (d : AudioEncoderConfig := {}) : AudioEncoderConfig := {
  numMelBins := getNatFieldD j "num_mel_bins" d.numMelBins
  encoderLayers := getNatFieldD j "encoder_layers" d.encoderLayers
  encoderAttentionHeads := getNatFieldD j "encoder_attention_heads" d.encoderAttentionHeads
  encoderFfnDim := getNatFieldD j "encoder_ffn_dim" d.encoderFfnDim
  dModel := getNatFieldD j "d_model" d.dModel
  dropout := getFloatFieldD j "dropout" d.dropout
  attentionDropout := getFloatFieldD j "attention_dropout" d.attentionDropout
  activationFunction := getStrFieldD j "activation_function" d.activationFunction
  activationDropout := getFloatFieldD j "activation_dropout" d.activationDropout
  scaleEmbedding := getBoolFieldD j "scale_embedding" d.scaleEmbedding
  initializerRange := getFloatFieldD j "initializer_range" d.initializerRange
  maxSourcePositions := getNatFieldD j "max_source_positions" d.maxSourcePositions
  nWindow := getNatFieldD j "n_window" d.nWindow
  outputDim := getNatFieldD j "output_dim" d.outputDim
  nWindowInfer := getNatFieldD j "n_window_infer" d.nWindowInfer
  convChunkSize := getNatFieldD j "conv_chunksize" d.convChunkSize
  downsampleHiddenSize := getNatFieldD j "downsample_hidden_size" d.downsampleHiddenSize
}

private def parseTextConfig (j : Json) (d : TextConfig := {}) : TextConfig := {
  vocabSize := getNatFieldD j "vocab_size" d.vocabSize
  hiddenSize := getNatFieldD j "hidden_size" d.hiddenSize
  intermediateSize := getNatFieldD j "intermediate_size" d.intermediateSize
  numHiddenLayers := getNatFieldD j "num_hidden_layers" d.numHiddenLayers
  numAttentionHeads := getNatFieldD j "num_attention_heads" d.numAttentionHeads
  numKeyValueHeads := getNatFieldD j "num_key_value_heads" d.numKeyValueHeads
  headDim := getNatFieldD j "head_dim" d.headDim
  hiddenAct := getStrFieldD j "hidden_act" d.hiddenAct
  maxPositionEmbeddings := getNatFieldD j "max_position_embeddings" d.maxPositionEmbeddings
  initializerRange := getFloatFieldD j "initializer_range" d.initializerRange
  rmsNormEps := getFloatFieldD j "rms_norm_eps" d.rmsNormEps
  useCache := getBoolFieldD j "use_cache" d.useCache
  tieWordEmbeddings := getBoolFieldD j "tie_word_embeddings" d.tieWordEmbeddings
  ropeTheta := getFloatFieldD j "rope_theta" d.ropeTheta
  attentionBias := getBoolFieldD j "attention_bias" d.attentionBias
  attentionDropout := getFloatFieldD j "attention_dropout" d.attentionDropout
}

private def parseThinkerConfig (j : Json) (d : ThinkerConfig := {}) : ThinkerConfig :=
  let audioConfig :=
    match getObjVal? j "audio_config" with
    | some a => parseAudioConfig a d.audioConfig
    | none => d.audioConfig
  let textConfig :=
    match getObjVal? j "text_config" with
    | some t => parseTextConfig t d.textConfig
    | none => d.textConfig
  {
    audioConfig
    textConfig
    audioTokenId := getTokenIdFieldD j "audio_token_id" d.audioTokenId
    audioStartTokenId := getTokenIdFieldD j "audio_start_token_id" d.audioStartTokenId
    userTokenId := getTokenIdFieldD j "user_token_id" d.userTokenId
    modelType := getStrFieldD j "model_type" d.modelType
    classifyNum := getNatFieldD j "classify_num" d.classifyNum
    initializerRange := getFloatFieldD j "initializer_range" d.initializerRange
  }

namespace Qwen3ASRConfig

/-- Load Qwen3-ASR config from HuggingFace `config.json`. -/
def loadFromFile (path : String) (defaults : Qwen3ASRConfig := {}) : IO Qwen3ASRConfig := do
  let root ← parseJsonFile path
  let thinkerConfig :=
    match getObjVal? root "thinker_config" with
    | some t => parseThinkerConfig t defaults.thinkerConfig
    | none => defaults.thinkerConfig
  let supportLanguages := getStringArrayFieldD root "support_languages" defaults.supportLanguages
  let timestampTokenId := getTokenIdFieldD root "timestamp_token_id" defaults.timestampTokenId
  let timestampSegmentTime := getFloatFieldD root "timestamp_segment_time" defaults.timestampSegmentTime
  pure { thinkerConfig, supportLanguages, timestampTokenId, timestampSegmentTime }

/-- Load Qwen3-ASR config from a model directory containing `config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : Qwen3ASRConfig := {}) : IO Qwen3ASRConfig :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Qwen3ASRConfig

end torch.qwen3asr
