/-
  Tyr/Model/Qwen3ASR/ConfigIO.lean

  JSON config loader for Qwen3-ASR HuggingFace `config.json`.
-/
import Tyr.Model.Qwen3ASR.Config
import Tyr.Model.Qwen.ConfigIO

namespace torch.qwen3asr

open Lean ConfigJson

private def parseAudioConfig (j : Json) (d : AudioEncoderConfig := {}) : Except String AudioEncoderConfig := do
  requireEq j "scale_embedding" false
  let numMelBins ← fieldD? j "num_mel_bins" d.numMelBins
  let ⟨hMel⟩ ← if h : 0 < numMelBins then pure (PLift.up h) else
    throw "Invalid Qwen3-ASR audio config: num_mel_bins must be positive"
  let dModel ← fieldD? j "d_model" d.dModel
  let heads ← fieldD? j "encoder_attention_heads" d.encoderAttentionHeads
  let ⟨hHeads⟩ ← if h : 0 < heads ∧ heads ∣ dModel then pure (PLift.up h) else
    throw s!"Invalid Qwen3-ASR audio config: encoder_attention_heads={heads} must be positive and divide d_model={dModel}"
  let ⟨hEven⟩ ← if h : 2 ∣ dModel then pure (PLift.up h) else
    throw s!"Invalid Qwen3-ASR audio config: d_model={dModel} must be even"
  let nWindow ← fieldD? j "n_window" d.nWindow
  let nWindowInfer ← fieldD? j "n_window_infer" d.nWindowInfer
  let ⟨hWindow⟩ ← if h : 0 < nWindow ∧ 0 < nWindowInfer ∧ 2 * nWindow ∣ nWindowInfer then pure (PLift.up h) else
    throw s!"Invalid Qwen3-ASR audio config: n_window_infer={nWindowInfer} must be a positive multiple of 2 * n_window (n_window={nWindow})"
  let activationFunction ← fieldD? j "activation_function" d.activationFunction
  let ⟨hAct⟩ ← if h : activationFunction = "gelu" ∨ activationFunction = "relu" ∨ activationFunction = "silu" then
      pure (PLift.up h)
    else
      throw s!"Invalid Qwen3-ASR audio config: unsupported activation_function=\"{activationFunction}\"; this model implements gelu, relu and silu"
  pure {
    numMelBins
    encoderLayers := ← fieldD? j "encoder_layers" d.encoderLayers
    encoderAttentionHeads := heads
    encoderFfnDim := ← fieldD? j "encoder_ffn_dim" d.encoderFfnDim
    dModel
    attentionDropout := ← fieldD? j "attention_dropout" d.attentionDropout
    activationFunction
    nWindow
    outputDim := ← fieldD? j "output_dim" d.outputDim
    nWindowInfer
    downsampleHiddenSize := ← fieldD? j "downsample_hidden_size" d.downsampleHiddenSize
    numMelBinsPos := hMel
    headsDivide := hHeads
    dModelEven := hEven
    windowValid := hWindow
    activationSupported := hAct
  }

/-- Text decoder through the shared Qwen parser, which checks the same facts and
rejects behavior the decoder hard-codes; the decoder always loads `lm_head`. -/
private def parseTextConfig (j : Json) (d : TextConfig := {}) : Except String TextConfig := do
  ConfigJson.requireEq j "tie_word_embeddings" false
  let q ← qwen.QwenConfig.parseJson j d.toQwenConfig (deriveHeadDim := false)
  pure {
    vocabSize := q.vocab_size
    hiddenSize := q.hidden_size
    intermediateSize := q.intermediate_size
    numHiddenLayers := q.num_hidden_layers
    numAttentionHeads := q.num_attention_heads
    numKeyValueHeads := q.num_key_value_heads
    halfHeadDim := q.half_head_dim
    rmsNormEps := q.rms_norm_eps
    ropeTheta := q.rope_theta
    headsValid := q.headsValid
    halfHeadDimPos := q.halfHeadDimPos
    vocabPos := q.vocabPos
    ropeThetaPos := q.ropeThetaPos
    rmsNormEpsPos := q.rmsNormEpsPos
  }

private def parseThinkerConfig (j : Json) (d : ThinkerConfig := {}) : Except String ThinkerConfig := do
  let audioConfig ← match ← objectField? j "audio_config" with
    | some a => parseAudioConfig a d.audioConfig
    | none => pure d.audioConfig
  let textConfig ← match ← objectField? j "text_config" with
    | some t => parseTextConfig t d.textConfig
    | none => pure d.textConfig
  let ⟨hMatch⟩ ← if h : audioConfig.outputDim = textConfig.hiddenSize then pure (PLift.up h) else
    throw s!"Invalid Qwen3-ASR config: audio output_dim={audioConfig.outputDim} must equal text hidden_size={textConfig.hiddenSize}"
  let audioTokenId ← tokenIdFieldD? j "audio_token_id" d.audioTokenId
  let ⟨hAudio⟩ ← if h : audioTokenId.toNat < textConfig.vocabSize then pure (PLift.up h) else
    throw s!"Invalid Qwen3-ASR config: audio_token_id={audioTokenId} must be below vocab_size={textConfig.vocabSize}"
  let modelType ← fieldD? j "model_type" d.modelType
  let classifyNum ← fieldD? j "classify_num" d.classifyNum
  let ⟨hClasses⟩ ← if h : isForcedAlignerType modelType = true → 0 < classifyNum then pure (PLift.up h) else
    throw s!"Invalid Qwen3-ASR config: forced aligner {modelType} needs a positive classify_num"
  pure {
    audioConfig
    textConfig
    audioTokenId
    modelType
    classifyNum
    audioMatchesText := hMatch
    audioTokenInVocab := hAudio
    alignerClasses := hClasses
  }

namespace Qwen3ASRConfig

def loadFromJson (root : Json) (defaults : Qwen3ASRConfig := {}) : Except String Qwen3ASRConfig := do
  let thinkerConfig ← match ← objectField? root "thinker_config" with
    | some t => parseThinkerConfig t defaults.thinkerConfig
    | none => pure defaults.thinkerConfig
  let supportLanguages ← fieldD? root "support_languages" defaults.supportLanguages
  let timestampTokenId ← match ← tokenIdField? root "timestamp_token_id" with
    | some id => pure id
    | none =>
      if thinkerConfig.isForcedAligner then
        throw "Invalid Qwen3-ASR config: a forced aligner needs timestamp_token_id"
      else pure defaults.timestampTokenId
  let vocab := thinkerConfig.textConfig.vocabSize
  let ⟨hTimestamp⟩ ← if h : timestampTokenId.toNat < vocab then pure (PLift.up h) else
    throw s!"Invalid Qwen3-ASR config: timestamp_token_id={timestampTokenId} must be below vocab_size={vocab}"
  let timestampSegmentTime ← fieldD? root "timestamp_segment_time" defaults.timestampSegmentTime
  let ⟨hSegment⟩ ← if h : 0 < timestampSegmentTime then pure (PLift.up h) else
    throw s!"Invalid Qwen3-ASR config: timestamp_segment_time={timestampSegmentTime} must be positive"
  pure {
    thinkerConfig, supportLanguages, timestampTokenId, timestampSegmentTime
    timestampTokenInVocab := hTimestamp
    timestampSegmentPos := hSegment
  }

/-- Load Qwen3-ASR config from HuggingFace `config.json`. -/
def loadFromFile (path : String) (defaults : Qwen3ASRConfig := {}) : IO Qwen3ASRConfig := do
  IO.ofExcept (loadFromJson (← parseFile path) defaults)

/-- Load Qwen3-ASR config from a model directory containing `config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : Qwen3ASRConfig := {}) : IO Qwen3ASRConfig :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Qwen3ASRConfig

end torch.qwen3asr
