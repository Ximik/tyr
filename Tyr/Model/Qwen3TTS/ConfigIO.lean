/-
  Tyr/Model/Qwen3TTS/ConfigIO.lean

  JSON config loader for Qwen3-TTS HuggingFace `config.json`.
  This lets runtime examples load real checkpoint shapes/token IDs directly
  from a model directory.
-/
import Tyr.Model.Qwen3TTS.Config
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.qwen3tts

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

private def getUInt64ArrayFieldD (j : Json) (key : String) (d : Array Nat) : Array Nat :=
  match getObjVal? j key >>= getArr? with
  | some arr =>
    Id.run do
      let mut out : Array Nat := #[]
      for item in arr do
        match getNat? item with
        | some n => out := out.push n
        | none => pure ()
      if out.isEmpty then d else out
  | none => d

private def parseStringUInt64Map (j : Json) : Array (String × UInt64) :=
  match j with
  | .obj kvs =>
    Id.run do
      let mut out : Array (String × UInt64) := #[]
      for (k, v) in kvs do
        match getNat? v with
        | some n => out := out.push (k, n.toUInt64)
        | none => pure ()
      out
  | _ => #[]

private def parseStringStringMap (j : Json) : Array (String × String) :=
  match j with
  | .obj kvs =>
    Id.run do
      let mut out : Array (String × String) := #[]
      for (k, v) in kvs do
        match v with
        | .str s => out := out.push (k, s)
        | .bool b => out := out.push (k, if b then "true" else "false")
        | _ => pure ()
      out
  | _ => #[]

private def parseCodePredictorConfig (j : Json) (d : TalkerCodePredictorConfig := {})
    : TalkerCodePredictorConfig := {
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
  ropeTheta := getFloatFieldD j "rope_theta" d.ropeTheta
  attentionBias := getBoolFieldD j "attention_bias" d.attentionBias
  useSlidingWindow := getBoolFieldD j "use_sliding_window" d.useSlidingWindow
  slidingWindow := getNatFieldD j "sliding_window" d.slidingWindow
  maxWindowLayers := getNatFieldD j "max_window_layers" d.maxWindowLayers
  attentionDropout := getFloatFieldD j "attention_dropout" d.attentionDropout
  numCodeGroups := getNatFieldD j "num_code_groups" d.numCodeGroups
}

private def parseTalkerConfig (j : Json) (d : TalkerConfig := {}) : TalkerConfig :=
  let codePredictorConfig :=
    match getObjVal? j "code_predictor_config" with
    | some cp => parseCodePredictorConfig cp d.codePredictorConfig
    | none => d.codePredictorConfig
  let spkId :=
    match getObjVal? j "spk_id" with
    | some o => parseStringUInt64Map o
    | none => d.spkId
  let spkIsDialect :=
    match getObjVal? j "spk_is_dialect" with
    | some o => parseStringStringMap o
    | none => d.spkIsDialect
  let codecLanguageId :=
    match getObjVal? j "codec_language_id" with
    | some o => parseStringUInt64Map o
    | none => d.codecLanguageId
  {
    codePredictorConfig
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
    ropeTheta := getFloatFieldD j "rope_theta" d.ropeTheta
    attentionBias := getBoolFieldD j "attention_bias" d.attentionBias
    useSlidingWindow := getBoolFieldD j "use_sliding_window" d.useSlidingWindow
    slidingWindow := getNatFieldD j "sliding_window" d.slidingWindow
    attentionDropout := getFloatFieldD j "attention_dropout" d.attentionDropout
    numCodeGroups := getNatFieldD j "num_code_groups" d.numCodeGroups
    textHiddenSize := getNatFieldD j "text_hidden_size" d.textHiddenSize
    textVocabSize := getNatFieldD j "text_vocab_size" d.textVocabSize
    codecEosTokenId := getTokenIdFieldD j "codec_eos_token_id" d.codecEosTokenId
    codecThinkId := getTokenIdFieldD j "codec_think_id" d.codecThinkId
    codecNoThinkId := getTokenIdFieldD j "codec_nothink_id" d.codecNoThinkId
    codecThinkBosId := getTokenIdFieldD j "codec_think_bos_id" d.codecThinkBosId
    codecThinkEosId := getTokenIdFieldD j "codec_think_eos_id" d.codecThinkEosId
    codecPadId := getTokenIdFieldD j "codec_pad_id" d.codecPadId
    codecBosId := getTokenIdFieldD j "codec_bos_id" d.codecBosId
    spkId
    spkIsDialect
    codecLanguageId
  }

private def parseSpeakerEncoderConfig (j : Json) (d : SpeakerEncoderConfig := {})
    : SpeakerEncoderConfig := {
  melDim := getNatFieldD j "mel_dim" d.melDim
  encDim := getNatFieldD j "enc_dim" d.encDim
  encChannels := getUInt64ArrayFieldD j "enc_channels" d.encChannels
  encKernelSizes := getUInt64ArrayFieldD j "enc_kernel_sizes" d.encKernelSizes
  encDilations := getUInt64ArrayFieldD j "enc_dilations" d.encDilations
  encAttentionChannels := getNatFieldD j "enc_attention_channels" d.encAttentionChannels
  encRes2NetScale := getNatFieldD j "enc_res2net_scale" d.encRes2NetScale
  encSeChannels := getNatFieldD j "enc_se_channels" d.encSeChannels
  sampleRate := getNatFieldD j "sample_rate" d.sampleRate
}

namespace Qwen3TTSConfig

/-- Load Qwen3-TTS config from a HuggingFace `config.json` file. -/
def loadFromFile (path : String) (defaults : Qwen3TTSConfig := {}) : IO Qwen3TTSConfig := do
  let root ← parseJsonFile path
  let talkerConfig :=
    match getObjVal? root "talker_config" with
    | some j => parseTalkerConfig j defaults.talkerConfig
    | none => defaults.talkerConfig
  let speakerEncoderConfig :=
    match getObjVal? root "speaker_encoder_config" with
    | some j => parseSpeakerEncoderConfig j defaults.speakerEncoderConfig
    | none => defaults.speakerEncoderConfig
  pure {
    talkerConfig
    speakerEncoderConfig
    tokenizerType := getStrFieldD root "tokenizer_type" defaults.tokenizerType
    ttsModelSize := getStrFieldD root "tts_model_size" defaults.ttsModelSize
    ttsModelType := getStrFieldD root "tts_model_type" defaults.ttsModelType
    imStartTokenId := getTokenIdFieldD root "im_start_token_id" defaults.imStartTokenId
    imEndTokenId := getTokenIdFieldD root "im_end_token_id" defaults.imEndTokenId
    ttsPadTokenId := getTokenIdFieldD root "tts_pad_token_id" defaults.ttsPadTokenId
    ttsBosTokenId := getTokenIdFieldD root "tts_bos_token_id" defaults.ttsBosTokenId
    ttsEosTokenId := getTokenIdFieldD root "tts_eos_token_id" defaults.ttsEosTokenId
  }

/-- Load Qwen3-TTS config from a model directory containing `config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : Qwen3TTSConfig := {}) : IO Qwen3TTSConfig :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Qwen3TTSConfig

end torch.qwen3tts
