/-
  Tyr/Model/KittenTTS/ConfigIO.lean

  Kokoro / KittenTTS `config.json` loader.
-/
import Tyr.Model.KittenTTS.Config
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.kittentts

open Lean

abbrev VocabMap := Std.HashMap Char UInt64

private def parseJsonFile (path : String) : IO Json := do
  let contents ← IO.FS.readFile path
  match Json.parse contents with
  | .ok json => pure json
  | .error err => throw <| IO.userError s!"Failed to parse JSON at {path}: {err}"

private def getObjVal? (j : Json) (key : String) : Option Json :=
  match j with
  | .obj kvs => Std.TreeMap.Raw.get? kvs key
  | _ => none

private def getArr? (j : Json) : Option (Array Json) :=
  match j with
  | .arr a => some a
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

private def getBool? (j : Json) : Option Bool := fromJson? j

private def getNatFieldD (j : Json) (key : String) (d : Nat) : Nat :=
  match getObjVal? j key >>= getNat? with
  | some n => n
  | none => d

private def getFloatFieldD (j : Json) (key : String) (d : Float) : Float :=
  match getObjVal? j key >>= getFloat? with
  | some x => x
  | none => d

private def getBoolFieldD (j : Json) (key : String) (d : Bool) : Bool :=
  match getObjVal? j key >>= getBool? with
  | some x => x
  | none => d

private def getNatArrayFieldD (j : Json) (key : String) (d : Array Nat) : Array Nat :=
  match getObjVal? j key >>= getArr? with
  | some arr =>
    Id.run do
      let mut out : Array Nat := #[]
      for it in arr do
        match getNat? it with
        | some n => out := out.push n
        | none => pure ()
      if out.isEmpty then d else out
  | none => d

private def getNatMatrixFieldD
    (j : Json)
    (key : String)
    (d : Array (Array Nat))
    : Array (Array Nat) :=
  match getObjVal? j key >>= getArr? with
  | some rows =>
    Id.run do
      let mut out : Array (Array Nat) := #[]
      for row in rows do
        match getArr? row with
        | some items =>
          let mut inner : Array Nat := #[]
          for item in items do
            match getNat? item with
            | some n => inner := inner.push n
            | none => pure ()
          out := out.push inner
        | none => pure ()
      if out.isEmpty then d else out
  | none => d

private def stringToSingleChar? (s : String) : Option Char :=
  match s.toList with
  | [c] => some c
  | _ => none

namespace AlbertConfig

def loadFromJson (root : Json) (defaults : AlbertConfig := {}) : AlbertConfig := {
  numHiddenLayers := getNatFieldD root "num_hidden_layers" defaults.numHiddenLayers
  numAttentionHeads := getNatFieldD root "num_attention_heads" defaults.numAttentionHeads
  hiddenSize := getNatFieldD root "hidden_size" defaults.hiddenSize
  intermediateSize := getNatFieldD root "intermediate_size" defaults.intermediateSize
  maxPositionEmbeddings := getNatFieldD root "max_position_embeddings" defaults.maxPositionEmbeddings
  embeddingSize := getNatFieldD root "embedding_size" defaults.embeddingSize
  innerGroupNum := getNatFieldD root "inner_group_num" defaults.innerGroupNum
  numHiddenGroups := getNatFieldD root "num_hidden_groups" defaults.numHiddenGroups
  typeVocabSize := getNatFieldD root "type_vocab_size" defaults.typeVocabSize
  layerNormEps := getFloatFieldD root "layer_norm_eps" defaults.layerNormEps
  hiddenDropoutProb :=
    getFloatFieldD root "hidden_dropout_prob" <|
      getFloatFieldD root "dropout" defaults.hiddenDropoutProb
  attentionDropoutProb :=
    getFloatFieldD root "attention_probs_dropout_prob" <|
      getFloatFieldD root "dropout" defaults.attentionDropoutProb
}

end AlbertConfig

namespace GeneratorConfig

def loadFromJson (root : Json) (defaults : GeneratorConfig := {}) : GeneratorConfig := {
  resblockKernelSizes :=
    getNatArrayFieldD root "resblock_kernel_sizes" defaults.resblockKernelSizes
  upsampleRates :=
    getNatArrayFieldD root "upsample_rates" defaults.upsampleRates
  upsampleInitialChannel :=
    getNatFieldD root "upsample_initial_channel" defaults.upsampleInitialChannel
  resblockDilationSizes :=
    getNatMatrixFieldD root "resblock_dilation_sizes" defaults.resblockDilationSizes
  upsampleKernelSizes :=
    getNatArrayFieldD root "upsample_kernel_sizes" defaults.upsampleKernelSizes
  genIstftNFft :=
    getNatFieldD root "gen_istft_n_fft" defaults.genIstftNFft
  genIstftHopSize :=
    getNatFieldD root "gen_istft_hop_size" defaults.genIstftHopSize
  harmonicCount := getNatFieldD root "harmonic_num" defaults.harmonicCount
}

end GeneratorConfig

namespace KittenTTSConfig

def loadFromFile (path : String) (defaults : KittenTTSConfig := {}) : IO KittenTTSConfig := do
  let root ← parseJsonFile path
  let plbert :=
    match getObjVal? root "plbert" with
    | some j => AlbertConfig.loadFromJson j defaults.plbert
    | none => defaults.plbert
  let generator :=
    match getObjVal? root "istftnet" with
    | some j => GeneratorConfig.loadFromJson j defaults.generator
    | none => defaults.generator
  let cfg : KittenTTSConfig := {
    hiddenDim := getNatFieldD root "hidden_dim" defaults.hiddenDim
    maxConvDim := getNatFieldD root "max_conv_dim" defaults.maxConvDim
    maxDur := getNatFieldD root "max_dur" defaults.maxDur
    nLayer := getNatFieldD root "n_layer" defaults.nLayer
    nMels := getNatFieldD root "n_mels" defaults.nMels
    nToken := getNatFieldD root "n_token" defaults.nToken
    styleDim := getNatFieldD root "style_dim" defaults.styleDim
    textEncoderKernelSize :=
      getNatFieldD root "text_encoder_kernel_size" defaults.textEncoderKernelSize
    asrResDim := getNatFieldD root "dim_in" defaults.asrResDim
    decoderOutDim :=
      getNatFieldD root "decoder_out_dim" defaults.decoderOutDim
    sampleRate := getNatFieldD root "sample_rate" defaults.sampleRate
    plbert := plbert
    generator := generator
  }
  if cfg.hiddenDim % 2 != 0 then
    throw <| IO.userError s!"Invalid KittenTTS config: hidden_dim={cfg.hiddenDim} must be divisible by 2"
  if cfg.plbert.numAttentionHeads == 0 || cfg.plbert.hiddenSize % cfg.plbert.numAttentionHeads != 0 then
    throw <| IO.userError
      s!"Invalid KittenTTS PLBERT config: hidden_size={cfg.plbert.hiddenSize}, heads={cfg.plbert.numAttentionHeads}"
  pure cfg

def loadFromPretrainedDir (modelDir : String) (defaults : KittenTTSConfig := {}) : IO KittenTTSConfig :=
  loadFromFile s!"{modelDir}/config.json" defaults

end KittenTTSConfig

def loadVocabFromFile (path : String) : IO VocabMap := do
  let root ← parseJsonFile path
  match getObjVal? root "vocab" with
  | some (.obj kvs) =>
    pure <| Id.run do
      let mut vocab : VocabMap := {}
      for (key, value) in kvs do
        match stringToSingleChar? key, getNat? value with
        | some c, some n => vocab := vocab.insert c n.toUInt64
        | _, _ => pure ()
      vocab
  | _ =>
    throw <| IO.userError s!"Missing or invalid vocab object in {path}"

def loadVocabFromPretrainedDir (modelDir : String) : IO VocabMap :=
  loadVocabFromFile s!"{modelDir}/config.json"

end torch.kittentts
