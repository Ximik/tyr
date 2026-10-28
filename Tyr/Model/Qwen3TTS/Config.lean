/-
  Tyr/Model/Qwen3TTS/Config.lean

  Configuration for Qwen3-TTS components:
  - speaker encoder
  - talker
  - talker code predictor
  - top-level TTS model
-/
import Tyr.Basic

namespace torch.qwen3tts

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
  deriving Repr, Inhabited

/-- Codec sub-talker config (predicts codebooks 1..N from codebook 0 context). -/
structure TalkerCodePredictorConfig where
  vocabSize : Nat := 2048
  hiddenSize : Nat := 1024
  intermediateSize : Nat := 3072
  numHiddenLayers : Nat := 5
  numAttentionHeads : Nat := 16
  numKeyValueHeads : Nat := 8
  headDim : Nat := 128
  hiddenAct : String := "silu"
  maxPositionEmbeddings : Nat := 32768
  initializerRange : Float := 0.02
  rmsNormEps : Float := 1e-6
  useCache : Bool := true
  ropeTheta : Float := 10000.0
  attentionBias : Bool := false
  useSlidingWindow : Bool := false
  slidingWindow : Nat := 4096
  maxWindowLayers : Nat := 28
  attentionDropout : Float := 0.0
  numCodeGroups : Nat := 32
  deriving Repr, Inhabited

namespace TalkerCodePredictorConfig

/-- Build default per-layer attention pattern:
    first `maxWindowLayers` full attention, then sliding attention. -/
def defaultLayerTypes (cfg : TalkerCodePredictorConfig) : Array String :=
  Id.run do
    let mut out : Array String := #[]
    for i in [:cfg.numHiddenLayers] do
      if cfg.useSlidingWindow && i >= cfg.maxWindowLayers then
        out := out.push "sliding_attention"
      else
        out := out.push "full_attention"
    out

end TalkerCodePredictorConfig

/-- Main talker config (text-conditioned codec generator). -/
structure TalkerConfig where
  codePredictorConfig : TalkerCodePredictorConfig := {}
  vocabSize : Nat := 3072
  hiddenSize : Nat := 1024
  intermediateSize : Nat := 2048
  numHiddenLayers : Nat := 20
  numAttentionHeads : Nat := 16
  numKeyValueHeads : Nat := 2
  headDim : Nat := 64
  hiddenAct : String := "silu"
  maxPositionEmbeddings : Nat := 32768
  initializerRange : Float := 0.02
  rmsNormEps : Float := 1e-6
  useCache : Bool := true
  ropeTheta : Float := 10000.0
  attentionBias : Bool := false
  useSlidingWindow : Bool := false
  slidingWindow : Nat := 4096
  attentionDropout : Float := 0.0
  numCodeGroups : Nat := 32
  textHiddenSize : Nat := 2048
  textVocabSize : Nat := 151936
  codecEosTokenId : UInt64 := 4198
  codecThinkId : UInt64 := 4202
  codecNoThinkId : UInt64 := 4203
  codecThinkBosId : UInt64 := 4204
  codecThinkEosId : UInt64 := 4205
  codecPadId : UInt64 := 4196
  codecBosId : UInt64 := 4197
  spkId : Array (String × UInt64) := #[]
  spkIsDialect : Array (String × String) := #[]
  codecLanguageId : Array (String × UInt64) := #[]
  deriving Repr, Inhabited

/-- Top-level Qwen3-TTS config. -/
structure Qwen3TTSConfig where
  talkerConfig : TalkerConfig := {}
  speakerEncoderConfig : SpeakerEncoderConfig := {}
  tokenizerType : String := "qwen3_tts_tokenizer_12hz"
  ttsModelSize : String := "4b"
  ttsModelType : String := "base"  -- base / custom_voice / voice_design
  imStartTokenId : UInt64 := 151644
  imEndTokenId : UInt64 := 151645
  ttsPadTokenId : UInt64 := 151671
  ttsBosTokenId : UInt64 := 151672
  ttsEosTokenId : UInt64 := 151673
  deriving Repr, Inhabited

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
