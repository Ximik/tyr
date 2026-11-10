/-
  Tyr/Model/Qwen3TTS/ConfigIO.lean

  JSON config loader for Qwen3-TTS HuggingFace `config.json`.
-/
import Tyr.Model.Qwen3TTS.Config
import Tyr.Model.ConfigJson

namespace torch.qwen3tts

open Lean ConfigJson

/-- A `{name: token id}` object at `key`, `default` when absent. -/
private def tokenIdMapD? (j : Json) (key : String) (default : Array (String × tokenizer.TokenId)) :
    Except String (Array (String × tokenizer.TokenId)) := do
  let some obj ← objectField? j key | pure default
  let some (entries : Std.TreeMap.Raw String Json) := (match obj with | .obj kvs => some kvs | _ => none)
    | pure default
  entries.foldlM (init := #[]) fun out name _ => do
    pure (out.push (name, ← tokenIdFieldD? obj name 0))

/-- A `{name: string | bool}` object at `key`, `default` when absent. -/
private def stringMapD? (j : Json) (key : String) (default : Array (String × String)) :
    Except String (Array (String × String)) := do
  let some obj ← objectField? j key | pure default
  let some (entries : Std.TreeMap.Raw String Json) := (match obj with | .obj kvs => some kvs | _ => none)
    | pure default
  entries.foldlM (init := #[]) fun out name value => do
    match value with
    | .str s => pure (out.push (name, s))
    | .bool b => pure (out.push (name, if b then "true" else "false"))
    | _ => throw s!"invalid '{key}.{name}' = {value.compress}: expected a string or boolean"

/-- Behavior both decoders hard-code: SiLU MLP, no attention bias, no sliding window. -/
private def requireQwenLayer (j : Json) : Except String Unit := do
  requireOneOf j "hidden_act" ["silu"]
  requireEq j "attention_bias" false
  requireEq j "use_sliding_window" false

/-- Plain RoPE; `default` scaling with an mrope section equals plain RoPE for the
talker's identical position rows. -/
private def requirePlainRope (j : Json) : Except String Unit := do
  match ← member? j "rope_scaling" with
  | none | some .null => pure ()
  | some scaling =>
    let kind : Option String ← firstFieldD? scaling ["rope_type", "type"] none
    unless kind == some "default" do
      throw s!"unsupported 'rope_scaling' = {scaling.compress}; this model implements plain RoPE"

private def parseCodePredictorConfig (j : Json) (d : TalkerCodePredictorConfig := {}) :
    Except String TalkerCodePredictorConfig := do
  requireQwenLayer j
  requirePlainRope j
  let heads ← fieldD? j "num_attention_heads" d.numAttentionHeads
  let kvHeads ← fieldD? j "num_key_value_heads" d.numKeyValueHeads
  let ⟨hHeads⟩ ← if h : 0 < heads ∧ 0 < kvHeads ∧ kvHeads ∣ heads then pure (PLift.up h) else
    throw s!"Invalid Qwen3-TTS code predictor config: num_key_value_heads={kvHeads} must be positive and divide num_attention_heads={heads}"
  let headDim ← fieldD? j "head_dim" d.headDim
  let ⟨hHeadDim⟩ ← if h : 0 < headDim ∧ 2 ∣ headDim then pure (PLift.up h) else
    throw s!"Invalid Qwen3-TTS code predictor config: head_dim={headDim} must be positive and even"
  let ropeTheta ← fieldD? j "rope_theta" d.ropeTheta
  let ⟨hTheta⟩ ← if h : 0 < ropeTheta then pure (PLift.up h) else
    throw s!"Invalid Qwen3-TTS code predictor config: rope_theta={ropeTheta} must be positive"
  let rmsNormEps ← fieldD? j "rms_norm_eps" d.rmsNormEps
  let ⟨hEps⟩ ← if h : 0 < rmsNormEps then pure (PLift.up h) else
    throw s!"Invalid Qwen3-TTS code predictor config: rms_norm_eps={rmsNormEps} must be positive"
  pure {
    vocabSize := ← fieldD? j "vocab_size" d.vocabSize
    hiddenSize := ← fieldD? j "hidden_size" d.hiddenSize
    intermediateSize := ← fieldD? j "intermediate_size" d.intermediateSize
    numHiddenLayers := ← fieldD? j "num_hidden_layers" d.numHiddenLayers
    numAttentionHeads := heads
    numKeyValueHeads := kvHeads
    halfHeadDim := headDim / 2
    rmsNormEps
    ropeTheta
    headsValid := hHeads
    halfHeadDimPos := by omega
    ropeThetaPos := hTheta
    rmsNormEpsPos := hEps
  }

private def parseTalkerConfig (j : Json) (d : TalkerConfig := {}) : Except String TalkerConfig := do
  requireQwenLayer j
  requirePlainRope j
  let codePredictorConfig ← match ← objectField? j "code_predictor_config" with
    | some cp => parseCodePredictorConfig cp d.codePredictorConfig
    | none => pure d.codePredictorConfig
  let numCodeGroups ← fieldD? j "num_code_groups" d.numCodeGroups
  let ⟨hGroups⟩ ← if h : 1 < numCodeGroups then pure (PLift.up h) else
    throw s!"Invalid Qwen3-TTS talker config: num_code_groups={numCodeGroups} must exceed 1"
  let heads ← fieldD? j "num_attention_heads" d.numAttentionHeads
  let kvHeads ← fieldD? j "num_key_value_heads" d.numKeyValueHeads
  let ⟨hHeads⟩ ← if h : 0 < heads ∧ 0 < kvHeads ∧ kvHeads ∣ heads then pure (PLift.up h) else
    throw s!"Invalid Qwen3-TTS talker config: num_key_value_heads={kvHeads} must be positive and divide num_attention_heads={heads}"
  let headDim ← fieldD? j "head_dim" d.headDim
  let ⟨hHeadDim⟩ ← if h : 0 < headDim ∧ 2 ∣ headDim then pure (PLift.up h) else
    throw s!"Invalid Qwen3-TTS talker config: head_dim={headDim} must be positive and even"
  let ropeTheta ← fieldD? j "rope_theta" d.ropeTheta
  let ⟨hTheta⟩ ← if h : 0 < ropeTheta then pure (PLift.up h) else
    throw s!"Invalid Qwen3-TTS talker config: rope_theta={ropeTheta} must be positive"
  let rmsNormEps ← fieldD? j "rms_norm_eps" d.rmsNormEps
  let ⟨hEps⟩ ← if h : 0 < rmsNormEps then pure (PLift.up h) else
    throw s!"Invalid Qwen3-TTS talker config: rms_norm_eps={rmsNormEps} must be positive"
  let vocabSize ← fieldD? j "vocab_size" d.vocabSize
  let codecEosTokenId ← tokenIdFieldD? j "codec_eos_token_id" d.codecEosTokenId
  let codecThinkId ← tokenIdFieldD? j "codec_think_id" d.codecThinkId
  let codecNoThinkId ← tokenIdFieldD? j "codec_nothink_id" d.codecNoThinkId
  let codecThinkBosId ← tokenIdFieldD? j "codec_think_bos_id" d.codecThinkBosId
  let codecThinkEosId ← tokenIdFieldD? j "codec_think_eos_id" d.codecThinkEosId
  let codecPadId ← tokenIdFieldD? j "codec_pad_id" d.codecPadId
  let codecBosId ← tokenIdFieldD? j "codec_bos_id" d.codecBosId
  let ⟨hCodec⟩ ← if h : codecEosTokenId.toNat < vocabSize ∧ codecThinkId.toNat < vocabSize ∧
      codecNoThinkId.toNat < vocabSize ∧ codecThinkBosId.toNat < vocabSize ∧
      codecThinkEosId.toNat < vocabSize ∧ codecPadId.toNat < vocabSize ∧ codecBosId.toNat < vocabSize then
      pure (PLift.up h)
    else
      throw s!"Invalid Qwen3-TTS talker config: codec special ids must be below vocab_size={vocabSize}"
  let spkId ← tokenIdMapD? j "spk_id" d.spkId
  let ⟨hSpk⟩ ← if h : ∀ e ∈ spkId, e.2.toNat < vocabSize then pure (PLift.up h) else
    throw s!"Invalid Qwen3-TTS talker config: spk_id values must be below vocab_size={vocabSize}"
  let codecLanguageId ← tokenIdMapD? j "codec_language_id" d.codecLanguageId
  let ⟨hLang⟩ ← if h : ∀ e ∈ codecLanguageId, e.2.toNat < vocabSize then pure (PLift.up h) else
    throw s!"Invalid Qwen3-TTS talker config: codec_language_id values must be below vocab_size={vocabSize}"
  pure {
    codePredictorConfig
    vocabSize
    hiddenSize := ← fieldD? j "hidden_size" d.hiddenSize
    intermediateSize := ← fieldD? j "intermediate_size" d.intermediateSize
    numHiddenLayers := ← fieldD? j "num_hidden_layers" d.numHiddenLayers
    numAttentionHeads := heads
    numKeyValueHeads := kvHeads
    halfHeadDim := headDim / 2
    rmsNormEps
    ropeTheta
    numCodeGroups
    textHiddenSize := ← fieldD? j "text_hidden_size" d.textHiddenSize
    textVocabSize := ← fieldD? j "text_vocab_size" d.textVocabSize
    codecEosTokenId, codecThinkId, codecNoThinkId, codecThinkBosId, codecThinkEosId, codecPadId, codecBosId
    spkId
    spkIsDialect := ← stringMapD? j "spk_is_dialect" d.spkIsDialect
    codecLanguageId
    residualCodeGroups := hGroups
    headsValid := hHeads
    halfHeadDimPos := by omega
    ropeThetaPos := hTheta
    rmsNormEpsPos := hEps
    codecIdsInVocab := hCodec
    speakerIdsInVocab := hSpk
    languageIdsInVocab := hLang
  }

private def parseSpeakerEncoderConfig (j : Json) (d : SpeakerEncoderConfig := {}) :
    Except String SpeakerEncoderConfig := do
  let encChannels ← fieldD? j "enc_channels" d.encChannels
  let encKernelSizes ← fieldD? j "enc_kernel_sizes" d.encKernelSizes
  let encDilations ← fieldD? j "enc_dilations" d.encDilations
  let ⟨hArrays⟩ ← if h : encChannels.size = 5 ∧ encKernelSizes.size = 5 ∧ encDilations.size = 5 then
      pure (PLift.up h)
    else
      throw s!"Invalid speaker encoder config: enc_channels, enc_kernel_sizes and enc_dilations need one entry per TDNN block (5), got {encChannels}, {encKernelSizes} and {encDilations}"
  let ⟨hPadding⟩ ← if h :
      SameConv (encKernelSizes.getD 0 5) (encDilations.getD 0 1) ∧
      SameConv (encKernelSizes.getD 1 3) (encDilations.getD 1 2) ∧
      SameConv (encKernelSizes.getD 2 3) (encDilations.getD 2 3) ∧
      SameConv (encKernelSizes.getD 3 3) (encDilations.getD 3 4) ∧
      SameConv (encKernelSizes.getD 4 1) (encDilations.getD 4 1) then pure (PLift.up h) else
    throw s!"Invalid speaker encoder config: every TDNN block needs a positive kernel and dilation with dilation * (kernel - 1) even, got kernels {encKernelSizes} and dilations {encDilations}"
  let ⟨hResidual⟩ ← if h :
      encChannels.getD 0 512 = encChannels.getD 1 512 ∧
      encChannels.getD 1 512 = encChannels.getD 2 512 ∧
      encChannels.getD 2 512 = encChannels.getD 3 512 then pure (PLift.up h) else
    throw s!"Invalid speaker encoder config: enc_channels {encChannels} must be equal across the SE-Res2Net blocks"
  let ⟨hMfa⟩ ← if h : encChannels.getD 4 1536 =
      encChannels.getD 1 512 + encChannels.getD 2 512 + encChannels.getD 3 512 then pure (PLift.up h) else
    throw s!"Invalid speaker encoder config: enc_channels[4] must equal the sum of enc_channels[1..3], got {encChannels}"
  let encRes2NetScale ← fieldD? j "enc_res2net_scale" d.encRes2NetScale
  let ⟨hScale⟩ ← if h : 2 ≤ encRes2NetScale ∧ encRes2NetScale ∣ encChannels.getD 0 512 then pure (PLift.up h) else
    throw s!"Invalid speaker encoder config: enc_res2net_scale={encRes2NetScale} must be at least 2 and divide enc_channels[0]"
  let sampleRate ← fieldD? j "sample_rate" d.sampleRate
  let ⟨hRate⟩ ← if h : 0 < sampleRate then pure (PLift.up h) else
    throw "Invalid speaker encoder config: sample_rate must be positive"
  pure {
    melDim := ← fieldD? j "mel_dim" d.melDim
    encDim := ← fieldD? j "enc_dim" d.encDim
    encChannels
    encKernelSizes
    encDilations
    encAttentionChannels := ← fieldD? j "enc_attention_channels" d.encAttentionChannels
    encRes2NetScale
    encSeChannels := ← fieldD? j "enc_se_channels" d.encSeChannels
    sampleRate
    sameConvs := hPadding
    residualChannels := hResidual
    blockArrays := hArrays
    res2NetScale := hScale
    mfaChannels := hMfa
    sampleRatePos := hRate
  }

namespace Qwen3TTSConfig

def loadFromJson (root : Json) (defaults : Qwen3TTSConfig := {}) : Except String Qwen3TTSConfig := do
  let talkerConfig ← match ← objectField? root "talker_config" with
    | some j => parseTalkerConfig j defaults.talkerConfig
    | none => pure defaults.talkerConfig
  let speakerEncoderConfig ← match ← objectField? root "speaker_encoder_config" with
    | some j => parseSpeakerEncoderConfig j defaults.speakerEncoderConfig
    | none => pure defaults.speakerEncoderConfig
  let ttsModelType ← fieldD? root "tts_model_type" defaults.ttsModelType
  let ⟨hSpeaker⟩ ← if h : ttsModelType = "base" → speakerEncoderConfig.encDim = talkerConfig.hiddenSize then
      pure (PLift.up h)
    else
      throw s!"Invalid Qwen3-TTS config: speaker enc_dim={speakerEncoderConfig.encDim} must equal talker hidden_size={talkerConfig.hiddenSize}"
  let imStartTokenId ← tokenIdFieldD? root "im_start_token_id" defaults.imStartTokenId
  let imEndTokenId ← tokenIdFieldD? root "im_end_token_id" defaults.imEndTokenId
  let ttsPadTokenId ← tokenIdFieldD? root "tts_pad_token_id" defaults.ttsPadTokenId
  let ttsBosTokenId ← tokenIdFieldD? root "tts_bos_token_id" defaults.ttsBosTokenId
  let ttsEosTokenId ← tokenIdFieldD? root "tts_eos_token_id" defaults.ttsEosTokenId
  let textVocab := talkerConfig.textVocabSize
  let ⟨hText⟩ ← if h : imStartTokenId.toNat < textVocab ∧ imEndTokenId.toNat < textVocab ∧
      ttsPadTokenId.toNat < textVocab ∧ ttsBosTokenId.toNat < textVocab ∧ ttsEosTokenId.toNat < textVocab then
      pure (PLift.up h)
    else
      throw s!"Invalid Qwen3-TTS config: prompt text ids must be below text_vocab_size={textVocab}"
  pure {
    talkerConfig, speakerEncoderConfig, ttsModelType
    imStartTokenId, imEndTokenId, ttsPadTokenId, ttsBosTokenId, ttsEosTokenId
    textIdsInVocab := hText
    speakerMatchesTalker := hSpeaker
  }

/-- Load Qwen3-TTS config from a HuggingFace `config.json` file. -/
def loadFromFile (path : String) (defaults : Qwen3TTSConfig := {}) : IO Qwen3TTSConfig := do
  IO.ofExcept (loadFromJson (← parseFile path) defaults)

/-- Load Qwen3-TTS config from a model directory containing `config.json`. -/
def loadFromPretrainedDir (modelDir : String) (defaults : Qwen3TTSConfig := {}) : IO Qwen3TTSConfig :=
  loadFromFile s!"{modelDir}/config.json" defaults

end Qwen3TTSConfig

end torch.qwen3tts
