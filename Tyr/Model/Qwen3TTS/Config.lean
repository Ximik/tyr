/-
  Tyr/Model/Qwen3TTS/Config.lean

  Configuration for Qwen3-TTS components:
  - speaker encoder
  - talker
  - talker code predictor
  - top-level TTS model
-/
import Tyr.Basic
import Tyr.Tokenizer.Types

namespace torch.qwen3tts

/-- A "same"-padded convolution: a positive kernel and dilation, with
    `dilation * (kernel - 1)` even so that half of it pads each side. -/
def SameConv (kernel dilation : Nat) : Prop :=
  0 < kernel ∧ 0 < dilation ∧ 2 ∣ dilation * (kernel - 1)

instance (kernel dilation : Nat) : Decidable (SameConv kernel dilation) := by
  unfold SameConv; infer_instance

/-- Speaker encoder config (ECAPA-TDNN style). -/
structure SpeakerEncoderConfig where
  melDim : Nat := 128
  encDim : Nat := 1024
  encChannels : Array Nat := #[512, 512, 512, 512, 1536]
  encKernelSizes : Array Nat := #[5, 3, 3, 3, 1]
  encDilations : Array Nat := #[1, 2, 3, 4, 1]
  encAttentionChannels : Nat := 128
  encRes2NetScale : Nat := 8
  encSeChannels : Nat := 128
  sampleRate : Nat := 24000
  /-- Each TDNN conv is "same"-padded, so it keeps its length; the index defaults match the
      encoder's own lookups. Checked when the config is built. -/
  sameConvs :
      SameConv (encKernelSizes.getD 0 5) (encDilations.getD 0 1) ∧
      SameConv (encKernelSizes.getD 1 3) (encDilations.getD 1 2) ∧
      SameConv (encKernelSizes.getD 2 3) (encDilations.getD 2 3) ∧
      SameConv (encKernelSizes.getD 3 3) (encDilations.getD 3 4) ∧
      SameConv (encKernelSizes.getD 4 1) (encDilations.getD 4 1) := by decide
  /-- The SE-Res2Net blocks add their input as the residual, so their channels must match. -/
  residualChannels :
      encChannels.getD 0 512 = encChannels.getD 1 512 ∧
      encChannels.getD 1 512 = encChannels.getD 2 512 ∧
      encChannels.getD 2 512 = encChannels.getD 3 512 := by decide
  /-- Five TDNN blocks each read their channels, kernel and dilation; checked when the config is built. -/
  blockArrays : encChannels.size = 5 ∧ encKernelSizes.size = 5 ∧ encDilations.size = 5 := by decide
  /-- Res2Net splits the block channels into `encRes2NetScale ≥ 2` equal parts; checked when the
      config is built. -/
  res2NetScale : 2 ≤ encRes2NetScale ∧ encRes2NetScale ∣ encChannels.getD 0 512 := by decide
  /-- Multi-layer feature aggregation concatenates the three SE-Res2Net outputs; checked when the
      config is built. -/
  mfaChannels : encChannels.getD 4 1536 =
      encChannels.getD 1 512 + encChannels.getD 2 512 + encChannels.getD 3 512 := by decide
  sampleRatePos : 0 < sampleRate := by decide
  deriving Repr


/-- Codec sub-talker config (predicts codebooks 1..N from codebook 0 context). -/
structure TalkerCodePredictorConfig where
  vocabSize : Nat := 2048
  hiddenSize : Nat := 1024
  intermediateSize : Nat := 3072
  numHiddenLayers : Nat := 5
  numAttentionHeads : Nat := 16
  numKeyValueHeads : Nat := 8
  /-- Half the head dimension; each head has `2 * halfHeadDim` channels. -/
  halfHeadDim : Nat := 64
  rmsNormEps : Float := 1e-6
  ropeTheta : Float := 10000.0
  /-- Query heads form whole groups over a nonzero number of K/V heads; checked when the config is built. -/
  headsValid : 0 < numAttentionHeads ∧ 0 < numKeyValueHeads ∧ numKeyValueHeads ∣ numAttentionHeads := by decide
  /-- Heads are nonempty; checked when the config is built. -/
  halfHeadDimPos : 0 < halfHeadDim := by decide
  /-- RoPE frequencies and RMSNorm divide by these; checked when the config is built. -/
  ropeThetaPos : 0 < ropeTheta := by native_decide
  rmsNormEpsPos : 0 < rmsNormEps := by native_decide
  deriving Repr

/-- Channels per attention head. -/
def TalkerCodePredictorConfig.headDim (cfg : TalkerCodePredictorConfig) : Nat := 2 * cfg.halfHeadDim

/-- Main talker config (text-conditioned codec generator). -/
structure TalkerConfig where
  codePredictorConfig : TalkerCodePredictorConfig := {}
  vocabSize : Nat := 3072
  hiddenSize : Nat := 1024
  intermediateSize : Nat := 2048
  numHiddenLayers : Nat := 20
  numAttentionHeads : Nat := 16
  numKeyValueHeads : Nat := 2
  /-- Half the head dimension; each head has `2 * halfHeadDim` channels. -/
  halfHeadDim : Nat := 32
  rmsNormEps : Float := 1e-6
  ropeTheta : Float := 10000.0
  numCodeGroups : Nat := 32
  textHiddenSize : Nat := 2048
  textVocabSize : Nat := 151936
  codecEosTokenId : tokenizer.TokenId := 2150
  codecThinkId : tokenizer.TokenId := 2154
  codecNoThinkId : tokenizer.TokenId := 2155
  codecThinkBosId : tokenizer.TokenId := 2156
  codecThinkEosId : tokenizer.TokenId := 2157
  codecPadId : tokenizer.TokenId := 2148
  codecBosId : tokenizer.TokenId := 2149
  spkId : Array (String × tokenizer.TokenId) := #[]
  spkIsDialect : Array (String × String) := #[]
  codecLanguageId : Array (String × tokenizer.TokenId) := #[]
  /-- Every codec frame has codebook 0 plus at least one residual codebook the code predictor
      generates; checked when the config is loaded. -/
  residualCodeGroups : 1 < numCodeGroups := by decide
  /-- Query heads form whole groups over a nonzero number of K/V heads; checked when the config is built. -/
  headsValid : 0 < numAttentionHeads ∧ 0 < numKeyValueHeads ∧ numKeyValueHeads ∣ numAttentionHeads := by decide
  /-- Heads are nonempty; checked when the config is built. -/
  halfHeadDimPos : 0 < halfHeadDim := by decide
  /-- RoPE frequencies and RMSNorm divide by these; checked when the config is built. -/
  ropeThetaPos : 0 < ropeTheta := by native_decide
  rmsNormEpsPos : 0 < rmsNormEps := by native_decide
  /-- Codec special, speaker and language ids index the codec embedding; checked when the config is built. -/
  codecIdsInVocab :
    codecEosTokenId.toNat < vocabSize ∧ codecThinkId.toNat < vocabSize ∧ codecNoThinkId.toNat < vocabSize ∧
    codecThinkBosId.toNat < vocabSize ∧ codecThinkEosId.toNat < vocabSize ∧ codecPadId.toNat < vocabSize ∧
    codecBosId.toNat < vocabSize := by decide
  speakerIdsInVocab : ∀ e ∈ spkId, e.2.toNat < vocabSize := by decide
  languageIdsInVocab : ∀ e ∈ codecLanguageId, e.2.toNat < vocabSize := by decide
  deriving Repr

/-- Channels per attention head. -/
def TalkerConfig.headDim (cfg : TalkerConfig) : Nat := 2 * cfg.halfHeadDim


/-- Top-level Qwen3-TTS config. -/
structure Qwen3TTSConfig where
  talkerConfig : TalkerConfig := {}
  speakerEncoderConfig : SpeakerEncoderConfig := {}
  ttsModelType : String := "base"  -- base / custom_voice / voice_design
  imStartTokenId : tokenizer.TokenId := 151644
  imEndTokenId : tokenizer.TokenId := 151645
  ttsPadTokenId : tokenizer.TokenId := 151671
  ttsBosTokenId : tokenizer.TokenId := 151672
  ttsEosTokenId : tokenizer.TokenId := 151673
  /-- Prompt text ids index the talker text embedding; checked when the config is built. -/
  textIdsInVocab :
    imStartTokenId.toNat < talkerConfig.textVocabSize ∧ imEndTokenId.toNat < talkerConfig.textVocabSize ∧
    ttsPadTokenId.toNat < talkerConfig.textVocabSize ∧ ttsBosTokenId.toNat < talkerConfig.textVocabSize ∧
    ttsEosTokenId.toNat < talkerConfig.textVocabSize := by decide
  /-- Base models condition on a speaker embedding in the talker's hidden space; checked when
      the config is built. -/
  speakerMatchesTalker :
    ttsModelType = "base" → speakerEncoderConfig.encDim = talkerConfig.hiddenSize := by decide
  deriving Repr

namespace Qwen3TTSConfig

/-- Supported language names inferred from language-id map (+ Auto). -/
def supportedLanguages (cfg : Qwen3TTSConfig) : Array String :=
  Id.run do
    let mut langs := #["auto"]
    for (k, _) in cfg.talkerConfig.codecLanguageId do
      if !k.contains "dialect" then
        langs := langs.push k
    langs

/-- Supported speaker names inferred from speaker-id map. -/
def supportedSpeakers (cfg : Qwen3TTSConfig) : Array String :=
  cfg.talkerConfig.spkId.map Prod.fst

end Qwen3TTSConfig

end torch.qwen3tts
