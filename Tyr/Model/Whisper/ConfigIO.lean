/-
  Tyr/Model/Whisper/ConfigIO.lean

  HuggingFace `config.json` loader for Whisper.
-/
import Tyr.Model.Whisper.Config
import Lean.Data.Json
import Lean.Data.Json.FromToJson.Basic

namespace torch.whisper

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

private def getStringFieldD (j : Json) (key : String) (d : String) : String :=
  match getObjVal? j key with
  | some (.str s) => s
  | _ => d

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

namespace WhisperConfig

def loadFromFile (path : String) (defaults : WhisperConfig := {}) : IO WhisperConfig := do
  let root ← parseJsonFile path
  let cfg : WhisperConfig := {
    numMelBins := getNatFieldD root "num_mel_bins" defaults.numMelBins
    vocabSize := getNatFieldD root "vocab_size" defaults.vocabSize
    dModel := getNatFieldD root "d_model" defaults.dModel
    encoderLayers := getNatFieldD root "encoder_layers" defaults.encoderLayers
    encoderAttentionHeads := getNatFieldD root "encoder_attention_heads" defaults.encoderAttentionHeads
    encoderFfnDim := getNatFieldD root "encoder_ffn_dim" defaults.encoderFfnDim
    decoderLayers := getNatFieldD root "decoder_layers" defaults.decoderLayers
    decoderAttentionHeads := getNatFieldD root "decoder_attention_heads" defaults.decoderAttentionHeads
    decoderFfnDim := getNatFieldD root "decoder_ffn_dim" defaults.decoderFfnDim
    maxSourcePositions := getNatFieldD root "max_source_positions" defaults.maxSourcePositions
    maxTargetPositions := getNatFieldD root "max_target_positions" defaults.maxTargetPositions
    activationFunction := getStringFieldD root "activation_function" defaults.activationFunction
    layerNormEps := getFloatFieldD root "layer_norm_eps" defaults.layerNormEps
    padTokenId := getTokenIdFieldD root "pad_token_id" defaults.padTokenId
    bosTokenId := getTokenIdFieldD root "bos_token_id" defaults.bosTokenId
    eosTokenId := getTokenIdFieldD root "eos_token_id" defaults.eosTokenId
    decoderStartTokenId := getTokenIdFieldD root "decoder_start_token_id" defaults.decoderStartTokenId
    suppressTokens := getNatArrayFieldD root "suppress_tokens" defaults.suppressTokens
    beginSuppressTokens := getNatArrayFieldD root "begin_suppress_tokens" defaults.beginSuppressTokens
  }
  if !cfg.hasValidHeads then
    throw <| IO.userError
      s!"Invalid Whisper config: d_model={cfg.dModel}, encoder_heads={cfg.encoderAttentionHeads}, decoder_heads={cfg.decoderAttentionHeads}"
  pure cfg

def loadFromPretrainedDir (modelDir : String) (defaults : WhisperConfig := {}) : IO WhisperConfig :=
  loadFromFile s!"{modelDir}/config.json" defaults

end WhisperConfig

end torch.whisper
