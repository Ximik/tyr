/-
  Tyr/Model/Whisper/ConfigIO.lean

  HuggingFace `config.json` loader for Whisper.
-/
import Tyr.Model.Whisper.Config
import Tyr.Model.ConfigJson

namespace torch.whisper

open Lean ConfigJson

namespace WhisperConfig

def loadFromJson (root : Json) (defaults : WhisperConfig := {}) : Except String WhisperConfig := do
  let numMelBins ← fieldD? root "num_mel_bins" defaults.numMelBins
  let ⟨hMel⟩ ← if h : 0 < numMelBins then pure (PLift.up h) else
    throw "Invalid Whisper config: num_mel_bins must be positive"
  let dModel ← fieldD? root "d_model" defaults.dModel
  let encoderAttentionHeads ← fieldD? root "encoder_attention_heads" defaults.encoderAttentionHeads
  let decoderAttentionHeads ← fieldD? root "decoder_attention_heads" defaults.decoderAttentionHeads
  let ⟨hEnc⟩ ← if h : 0 < encoderAttentionHeads ∧ encoderAttentionHeads ∣ dModel then pure (PLift.up h) else
    throw s!"Invalid Whisper config: encoder_attention_heads={encoderAttentionHeads} must be positive and divide d_model={dModel}"
  let ⟨hDec⟩ ← if h : 0 < decoderAttentionHeads ∧ decoderAttentionHeads ∣ dModel then pure (PLift.up h) else
    throw s!"Invalid Whisper config: decoder_attention_heads={decoderAttentionHeads} must be positive and divide d_model={dModel}"
  let maxTargetPositions ← fieldD? root "max_target_positions" defaults.maxTargetPositions
  let ⟨hTarget⟩ ← if h : 0 < maxTargetPositions then pure (PLift.up h) else
    throw "Invalid Whisper config: max_target_positions must be positive"
  let vocabSize ← fieldD? root "vocab_size" defaults.vocabSize
  let ⟨hVocab⟩ ← if h : 0 < vocabSize then pure (PLift.up h) else
    throw "Invalid Whisper config: vocab_size must be positive"
  let layerNormEps ← fieldD? root "layer_norm_eps" defaults.layerNormEps
  let ⟨hEps⟩ ← if h : 0 < layerNormEps then pure (PLift.up h) else
    throw s!"Invalid Whisper config: layer_norm_eps={layerNormEps} must be positive"
  let eosTokenId ← tokenIdFieldD? root "eos_token_id" defaults.eosTokenId
  let ⟨hEos⟩ ← if h : eosTokenId.toNat < vocabSize then pure (PLift.up h) else
    throw s!"Invalid Whisper config: eos_token_id={eosTokenId} must be below vocab_size={vocabSize}"
  let decoderStartTokenId ← tokenIdFieldD? root "decoder_start_token_id" defaults.decoderStartTokenId
  let ⟨hStart⟩ ← if h : decoderStartTokenId.toNat < vocabSize then pure (PLift.up h) else
    throw s!"Invalid Whisper config: decoder_start_token_id={decoderStartTokenId} must be below vocab_size={vocabSize}"
  let suppressTokens ← tokenIdArrayFieldD? root "suppress_tokens" defaults.suppressTokens
  let ⟨hSuppress⟩ ← if h : ∀ t ∈ suppressTokens, t.toNat < vocabSize then pure (PLift.up h) else
    throw s!"Invalid Whisper config: suppress_tokens must be below vocab_size={vocabSize}"
  let beginSuppressTokens ← tokenIdArrayFieldD? root "begin_suppress_tokens" defaults.beginSuppressTokens
  let ⟨hBegin⟩ ← if h : ∀ t ∈ beginSuppressTokens, t.toNat < vocabSize then pure (PLift.up h) else
    throw s!"Invalid Whisper config: begin_suppress_tokens must be below vocab_size={vocabSize}"
  let activationFunction ← fieldD? root "activation_function" defaults.activationFunction
  let ⟨hActivation⟩ ← if h : activationFunction = "gelu" ∨ activationFunction = "relu" then pure (PLift.up h) else
    throw s!"Invalid Whisper config: unsupported activation_function=\"{activationFunction}\"; this model implements gelu and relu"
  pure {
    numMelBins
    vocabSize
    dModel
    encoderLayers := ← fieldD? root "encoder_layers" defaults.encoderLayers
    encoderAttentionHeads
    encoderFfnDim := ← fieldD? root "encoder_ffn_dim" defaults.encoderFfnDim
    decoderLayers := ← fieldD? root "decoder_layers" defaults.decoderLayers
    decoderAttentionHeads
    decoderFfnDim := ← fieldD? root "decoder_ffn_dim" defaults.decoderFfnDim
    maxSourcePositions := ← fieldD? root "max_source_positions" defaults.maxSourcePositions
    maxTargetPositions
    activationFunction
    layerNormEps
    eosTokenId
    decoderStartTokenId
    suppressTokens
    beginSuppressTokens
    numMelBinsPos := hMel
    encoderHeadsDivide := hEnc.2
    decoderHeadsDivide := hDec.2
    encoderHeadsPos := hEnc.1
    decoderHeadsPos := hDec.1
    maxTargetPositionsPos := hTarget
    vocabSizePos := hVocab
    layerNormEpsPos := hEps
    eosInVocab := hEos
    startInVocab := hStart
    suppressInVocab := hSuppress
    beginSuppressInVocab := hBegin
    activationSupported := hActivation
  }

def loadFromFile (path : String) (defaults : WhisperConfig := {}) : IO WhisperConfig := do
  IO.ofExcept (loadFromJson (← parseFile path) defaults)

def loadFromPretrainedDir (modelDir : String) (defaults : WhisperConfig := {}) : IO WhisperConfig :=
  loadFromFile s!"{modelDir}/config.json" defaults

end WhisperConfig

end torch.whisper
