/-
  Tyr/Model/Qwen3ASR/Config.lean

  Configuration types for the Lean4 Qwen3-ASR port.
-/
import Tyr.Basic
import Tyr.Model.Qwen.Config
import Tyr.Tokenizer.Types

namespace torch.qwen3asr

/-- Qwen3-ASR audio encoder config. Mirrors `Qwen3ASRAudioEncoderConfig`. -/
structure AudioEncoderConfig where
  numMelBins : Nat := 128
  encoderLayers : Nat := 32
  encoderAttentionHeads : Nat := 20
  encoderFfnDim : Nat := 5120
  dModel : Nat := 1280
  attentionDropout : Float := 0.0
  /-- `"gelu"`, `"relu"` or `"silu"`; checked when the config is built. -/
  activationFunction : String := "gelu"
  nWindow : Nat := 100
  outputDim : Nat := 4096
  nWindowInfer : Nat := 400
  downsampleHiddenSize : Nat := 480
  /-- The strided convolutions downsample the mel axis; checked when the config is built. -/
  numMelBinsPos : 0 < numMelBins := by decide
  /-- Attention splits `dModel` into a nonzero number of equal heads; checked when the config is built. -/
  headsDivide : 0 < encoderAttentionHeads ∧ encoderAttentionHeads ∣ dModel := by decide
  /-- Sinusoidal positions pair `dModel / 2` sines with cosines; checked when the config is built. -/
  dModelEven : 2 ∣ dModel := by decide
  /-- Features are chunked by `2 * nWindow`, and inference attends over a whole number of
      chunks; checked when the config is built. -/
  windowValid : 0 < nWindow ∧ 0 < nWindowInfer ∧ 2 * nWindow ∣ nWindowInfer := by decide
  activationSupported :
    activationFunction = "gelu" ∨ activationFunction = "relu" ∨ activationFunction = "silu" := by decide
  deriving Repr


namespace AudioEncoderConfig

def downsampleOnce (n : Nat) : Nat :=
  if n == 0 then 0 else ((n - 1) / 2) + 1

def downsampleTwice (n : Nat) : Nat :=
  downsampleOnce (downsampleOnce n)

def downsampleThrice (n : Nat) : Nat :=
  downsampleOnce (downsampleTwice n)

def melAfterConv1 (cfg : AudioEncoderConfig) : Nat :=
  downsampleOnce cfg.numMelBins

def melAfterConv2 (cfg : AudioEncoderConfig) : Nat :=
  downsampleTwice cfg.numMelBins

def melAfterConv3 (cfg : AudioEncoderConfig) : Nat :=
  downsampleThrice cfg.numMelBins

def framesAfterConv3 (_cfg : AudioEncoderConfig) (frames : Nat) : Nat :=
  downsampleThrice frames

@[simp] def convOutInDim (cfg : AudioEncoderConfig) : Nat :=
  cfg.downsampleHiddenSize * melAfterConv3 cfg

def headDim (cfg : AudioEncoderConfig) : Nat :=
  cfg.dModel / cfg.encoderAttentionHeads

/-- Reference `_get_feat_extract_output_lengths` port from Qwen3-ASR processor/modeling. -/
def featExtractOutputLength (inputLen : Nat) : Nat :=
  let leave := inputLen % 100
  let feat :=
    if leave == 0 then 0
    else ((leave - 1) / 2) + 1
  let outLeave :=
    if feat == 0 then 0
    else ((((feat - 1) / 2) + 1 - 1) / 2) + 1
  let out := outLeave + (inputLen / 100) * 13
  out

def featExtractOutputLengths (inputLens : Array Nat) : Array Nat :=
  inputLens.map featExtractOutputLength

end AudioEncoderConfig

/-- Qwen3-ASR text decoder config. Mirrors `Qwen3ASRTextConfig`. -/
structure TextConfig where
  vocabSize : Nat := 151936
  hiddenSize : Nat := 4096
  intermediateSize : Nat := 22016
  numHiddenLayers : Nat := 32
  numAttentionHeads : Nat := 32
  numKeyValueHeads : Nat := 32
  /-- Half the head dimension; each head has `2 * halfHeadDim` channels. -/
  halfHeadDim : Nat := 64
  rmsNormEps : Float := 1e-6
  ropeTheta : Float := 5000000.0
  /-- Query heads form whole groups over a nonzero number of K/V heads; checked when the config is built. -/
  headsValid : 0 < numAttentionHeads ∧ 0 < numKeyValueHeads ∧ numKeyValueHeads ∣ numAttentionHeads := by decide
  /-- Heads are nonempty; checked when the config is built. -/
  halfHeadDimPos : 0 < halfHeadDim := by decide
  /-- Greedy decoding picks among `vocabSize` logits; checked when the config is built. -/
  vocabPos : 0 < vocabSize := by decide
  /-- RoPE frequencies and RMSNorm divide by these; checked when the config is built. -/
  ropeThetaPos : 0 < ropeTheta := by native_decide
  rmsNormEpsPos : 0 < rmsNormEps := by native_decide
  deriving Repr

namespace TextConfig

/-- Channels per attention head. -/
def headDim (cfg : TextConfig) : Nat := 2 * cfg.halfHeadDim

def toQwenConfig (cfg : TextConfig) : qwen.QwenConfig := {
  vocab_size := cfg.vocabSize
  hidden_size := cfg.hiddenSize
  intermediate_size := cfg.intermediateSize
  num_hidden_layers := cfg.numHiddenLayers
  num_attention_heads := cfg.numAttentionHeads
  num_key_value_heads := cfg.numKeyValueHeads
  half_head_dim := cfg.halfHeadDim
  rope_theta := cfg.ropeTheta
  rms_norm_eps := cfg.rmsNormEps
  headsValid := cfg.headsValid
  halfHeadDimPos := cfg.halfHeadDimPos
  vocabPos := cfg.vocabPos
  ropeThetaPos := cfg.ropeThetaPos
  rmsNormEpsPos := cfg.rmsNormEpsPos
}

end TextConfig

/-- Composite thinker config: audio encoder + text decoder. -/
private def containsSubstr (s pat : String) : Bool :=
  if pat.isEmpty then
    false
  else
    (s.splitOn pat).length > 1

/-- Forced-aligner checkpoints expose `model_type` containing `"forced_aligner"`. -/
def isForcedAlignerType (modelType : String) : Bool :=
  containsSubstr modelType "forced_aligner"

structure ThinkerConfig where
  audioConfig : AudioEncoderConfig := {}
  textConfig : TextConfig := {}
  audioTokenId : tokenizer.TokenId := 151646
  modelType : String := "qwen3_asr_thinker"
  classifyNum : Nat := 0
  /-- Audio features replace text embeddings one-for-one; checked when the config is built. -/
  audioMatchesText : audioConfig.outputDim = textConfig.hiddenSize := by decide
  /-- The audio placeholder is embedded as a text token; checked when the config is built. -/
  audioTokenInVocab : audioTokenId.toNat < textConfig.vocabSize := by decide
  /-- A forced aligner classifies into `classifyNum` timestamp bins; checked when the config is built. -/
  alignerClasses : isForcedAlignerType modelType = true → 0 < classifyNum := by native_decide
  deriving Repr

namespace ThinkerConfig

def isForcedAligner (cfg : ThinkerConfig) : Bool :=
  isForcedAlignerType cfg.modelType

/-- LM head output dim:
    - normal ASR thinker: text vocab size
    - forced aligner thinker: `classify_num` from config -/
def lmHeadOutDim (cfg : ThinkerConfig) : Nat :=
  if isForcedAligner cfg then cfg.classifyNum else cfg.textConfig.vocabSize

theorem lmHeadOutDim_pos (cfg : ThinkerConfig) : 0 < lmHeadOutDim cfg := by
  unfold lmHeadOutDim isForcedAligner
  split
  · exact cfg.alignerClasses ‹_›
  · exact cfg.textConfig.vocabPos

end ThinkerConfig

def defaultSupportedLanguages : Array String :=
  #[
    "Chinese", "English", "Cantonese", "Arabic", "German", "French", "Spanish",
    "Portuguese", "Indonesian", "Italian", "Korean", "Russian", "Thai", "Vietnamese",
    "Japanese", "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish", "Finnish",
    "Polish", "Czech", "Filipino", "Persian", "Greek", "Romanian", "Hungarian", "Macedonian"
  ]

/-- Top-level Qwen3-ASR config. -/
structure Qwen3ASRConfig where
  thinkerConfig : ThinkerConfig := {}
  supportLanguages : Array String := defaultSupportedLanguages
  /-- Token ID used by forced-aligner prompt to mark timestamp slots. -/
  timestampTokenId : tokenizer.TokenId := 0
  /-- Milliseconds represented by one forced-aligner timestamp bin. -/
  timestampSegmentTime : Float := 1.0
  /-- Timestamp slots are text tokens; checked when the config is built. -/
  timestampTokenInVocab : timestampTokenId.toNat < thinkerConfig.textConfig.vocabSize := by decide
  /-- Timestamp bins span positive time; checked when the config is built. -/
  timestampSegmentPos : 0 < timestampSegmentTime := by native_decide
  deriving Repr

def defaultEosTokenIds : Array tokenizer.TokenId := #[151645, 151643]

namespace Qwen3ASRConfig

def getTextConfig (cfg : Qwen3ASRConfig) : TextConfig :=
  cfg.thinkerConfig.textConfig

def getAudioConfig (cfg : Qwen3ASRConfig) : AudioEncoderConfig :=
  cfg.thinkerConfig.audioConfig

end Qwen3ASRConfig

end torch.qwen3asr
