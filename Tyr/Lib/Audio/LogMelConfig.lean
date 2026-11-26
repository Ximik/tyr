/-
  Tyr/Lib/Audio/LogMelConfig.lean

  Log-mel feature parameters, loaded from a HuggingFace `preprocessor_config.json`
  whose `feature_extractor_type` is `WhisperFeatureExtractor`.
-/
import Tyr.Core.FFI.Basic
import Tyr.IO.ConfigJson

namespace torch.audio

open Lean ConfigJson

structure LogMelConfig where
  featureSize : Nat := 128
  samplingRate : Nat := 16000
  hopLength : Nat := 160
  chunkLength : Nat := 30
  nFft : Nat := 400
  nSamples : Nat := 0
  nbMaxFrames : Nat := 0
  paddingValue : Float := 0.0
  returnAttentionMask : Bool := true
  doNormalize : Bool := false
  dither : Float := 0.0
  /-- Mel bins, STFT window and hop are nonzero; checked when the config is built. -/
  featureSizePos : 0 < featureSize := by decide
  samplingRatePos : 0 < samplingRate := by decide
  hopPos : 0 < hopLength := by decide
  nFftPos : 0 < nFft := by decide
  deriving Repr

namespace LogMelConfig

def expectedSampleCount (cfg : LogMelConfig) : Nat :=
  if cfg.nSamples > 0 then
    cfg.nSamples
  else
    cfg.chunkLength * cfg.samplingRate

def expectedFrames (cfg : LogMelConfig) : Nat :=
  if cfg.nbMaxFrames > 0 then
    cfg.nbMaxFrames
  else
    expectedSampleCount cfg / cfg.hopLength

/-- Load from a `preprocessor_config.json` node.
Callers that resample input to a fixed rate pass it as `requiredSamplingRate`. -/
def loadFromJson (root : Json) (defaults : LogMelConfig := {})
    (requiredSamplingRate : Option Nat := none) : Except String LogMelConfig := do
  requireOneOf root "feature_extractor_type" ["WhisperFeatureExtractor"]
  requireOneOf root "padding_side" ["right"]
  let samplingRate ← fieldD? root "sampling_rate" defaults.samplingRate
  let ⟨hRate⟩ ← if h : 0 < samplingRate then pure (PLift.up h) else
    throw "Invalid preprocessor config: sampling_rate must be positive"
  if let some required := requiredSamplingRate then
    unless samplingRate = required do
      throw s!"Invalid preprocessor config: sampling_rate={samplingRate}; this model resamples input to {required}"
  let featureSize ← fieldD? root "feature_size" defaults.featureSize
  let ⟨hFeature⟩ ← if h : 0 < featureSize then pure (PLift.up h) else
    throw "Invalid preprocessor config: feature_size must be positive"
  let hopLength ← fieldD? root "hop_length" defaults.hopLength
  let ⟨hHop⟩ ← if h : 0 < hopLength then pure (PLift.up h) else
    throw "Invalid preprocessor config: hop_length must be positive"
  let nFft ← fieldD? root "n_fft" defaults.nFft
  let ⟨hFft⟩ ← if h : 0 < nFft then pure (PLift.up h) else
    throw "Invalid preprocessor config: n_fft must be positive"
  let chunkLength ← firstFieldD? root ["chunk_length", "chunk_length_s"] defaults.chunkLength
  let nSamples ← fieldD? root "n_samples" defaults.nSamples
  let nSamples := if nSamples > 0 then nSamples else chunkLength * samplingRate
  let nbMaxFrames ← fieldD? root "nb_max_frames" defaults.nbMaxFrames
  let nbMaxFrames := if nbMaxFrames > 0 then nbMaxFrames else nSamples / hopLength
  pure {
    featureSize
    samplingRate
    hopLength
    chunkLength
    nFft
    nSamples
    nbMaxFrames
    paddingValue := ← fieldD? root "padding_value" defaults.paddingValue
    returnAttentionMask := ← fieldD? root "return_attention_mask" defaults.returnAttentionMask
    doNormalize := ← fieldD? root "do_normalize" defaults.doNormalize
    dither := ← fieldD? root "dither" defaults.dither
    featureSizePos := hFeature
    samplingRatePos := hRate
    hopPos := hHop
    nFftPos := hFft
  }

/-- Load from a `preprocessor_config.json` file. -/
def loadFromFile (path : String) (defaults : LogMelConfig := {})
    (requiredSamplingRate : Option Nat := none) : IO LogMelConfig := do
  IO.ofExcept (loadFromJson (← parseFile path) defaults requiredSamplingRate)

/-- Load from a model directory containing `preprocessor_config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : LogMelConfig := {})
    (requiredSamplingRate : Option Nat := none) : IO LogMelConfig :=
  loadFromFile s!"{modelDir}/preprocessor_config.json" defaults requiredSamplingRate

end LogMelConfig

end torch.audio
