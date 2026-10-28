/-
  Tyr/Model/Qwen3ASR/PreprocessorConfig.lean

  JSON loader for HuggingFace `preprocessor_config.json` used by
  `WhisperFeatureExtractor` in Qwen3-ASR.
-/
import Tyr.Basic
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.qwen3asr

open Lean

/-- Whisper feature-extractor config used by Qwen3-ASR processor. -/
structure PreprocessorConfig where
  featureExtractorType : String := "WhisperFeatureExtractor"
  featureSize : Nat := 128
  samplingRate : Nat := 16000
  hopLength : Nat := 160
  chunkLength : Nat := 30
  nFft : Nat := 400
  nSamples : Nat := 0
  nbMaxFrames : Nat := 0
  paddingSide : String := "right"
  paddingValue : Float := 0.0
  returnAttentionMask : Bool := true
  doNormalize : Bool := false
  dither : Float := 0.0
  deriving Repr, Inhabited

namespace PreprocessorConfig

def expectedSampleCount (cfg : PreprocessorConfig) : Nat :=
  if cfg.nSamples > 0 then
    cfg.nSamples
  else
    cfg.chunkLength * cfg.samplingRate

def expectedFrames (cfg : PreprocessorConfig) : Nat :=
  let hop := if cfg.hopLength == 0 then 1 else cfg.hopLength
  if cfg.nbMaxFrames > 0 then
    cfg.nbMaxFrames
  else
    expectedSampleCount cfg / hop

private def normalizeDerived (cfg : PreprocessorConfig) : PreprocessorConfig :=
  let nSamples :=
    if cfg.nSamples > 0 then cfg.nSamples else cfg.chunkLength * cfg.samplingRate
  let hop := if cfg.hopLength == 0 then 1 else cfg.hopLength
  let nbMaxFrames :=
    if cfg.nbMaxFrames > 0 then cfg.nbMaxFrames else nSamples / hop
  { cfg with nSamples, nbMaxFrames, hopLength := hop }

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

private def getUInt64FieldFromFloatD (j : Json) (key : String) (d : UInt64) : UInt64 :=
  match getObjVal? j key >>= getFloat? with
  | some x => x.toUInt64
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

/-- Load Whisper/Qwen-ASR preprocessor config from file. -/
def loadFromFile (path : String) (defaults : PreprocessorConfig := {}) : IO PreprocessorConfig := do
  let root ← parseJsonFile path
  let chunkLength :=
    match getObjVal? root "chunk_length" >>= getFloat? with
    | some x => x.toUInt64.toNat
    | none =>
      match getObjVal? root "chunk_length_s" >>= getFloat? with
      | some x => x.toUInt64.toNat
      | none => defaults.chunkLength
  let cfg : PreprocessorConfig := {
    featureExtractorType := getStrFieldD root "feature_extractor_type" defaults.featureExtractorType
    featureSize := getNatFieldD root "feature_size" defaults.featureSize
    samplingRate := getNatFieldD root "sampling_rate" defaults.samplingRate
    hopLength := getNatFieldD root "hop_length" defaults.hopLength
    chunkLength := chunkLength
    nFft := getNatFieldD root "n_fft" defaults.nFft
    nSamples := getNatFieldD root "n_samples" defaults.nSamples
    nbMaxFrames := getNatFieldD root "nb_max_frames" defaults.nbMaxFrames
    paddingSide := getStrFieldD root "padding_side" defaults.paddingSide
    paddingValue := getFloatFieldD root "padding_value" defaults.paddingValue
    returnAttentionMask := getBoolFieldD root "return_attention_mask" defaults.returnAttentionMask
    doNormalize := getBoolFieldD root "do_normalize" defaults.doNormalize
    dither := getFloatFieldD root "dither" defaults.dither
  }
  pure (normalizeDerived cfg)

/-- Load from a model directory containing `preprocessor_config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : PreprocessorConfig := {}) : IO PreprocessorConfig :=
  loadFromFile s!"{modelDir}/preprocessor_config.json" defaults

end PreprocessorConfig

end torch.qwen3asr
