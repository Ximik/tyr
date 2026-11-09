/-
  Tyr/Model/Whisper/Config.lean

  Configuration for Whisper encoder-decoder ASR in Tyr.
-/
import Tyr.Basic
import Tyr.Tokenizer.Types

namespace torch.whisper

structure WhisperConfig where
  numMelBins : Nat := 80
  vocabSize : Nat := 51865
  dModel : Nat := 384
  encoderLayers : Nat := 4
  encoderAttentionHeads : Nat := 6
  encoderFfnDim : Nat := 1536
  decoderLayers : Nat := 4
  decoderAttentionHeads : Nat := 6
  decoderFfnDim : Nat := 1536
  maxSourcePositions : Nat := 1500
  maxTargetPositions : Nat := 448
  /-- `"gelu"` or `"relu"`; checked when the config is built. -/
  activationFunction : String := "gelu"
  layerNormEps : Float := 1e-5
  eosTokenId : tokenizer.TokenId := 50257
  decoderStartTokenId : tokenizer.TokenId := 50258
  suppressTokens : Array tokenizer.TokenId := #[]
  beginSuppressTokens : Array tokenizer.TokenId := #[]
  /-- The encoder convolves over a nonzero number of mel bins; checked when the config is built. -/
  numMelBinsPos : 0 < numMelBins := by decide
  /-- Attention splits `dModel` into equal heads; checked when the config is built. -/
  encoderHeadsDivide : encoderAttentionHeads ∣ dModel := by decide
  decoderHeadsDivide : decoderAttentionHeads ∣ dModel := by decide
  encoderHeadsPos : 0 < encoderAttentionHeads := by decide
  decoderHeadsPos : 0 < decoderAttentionHeads := by decide
  /-- The decoder KV cache holds `maxTargetPositions` tokens; checked when the config is built. -/
  maxTargetPositionsPos : 0 < maxTargetPositions := by decide
  /-- Beam and top-k candidate counts are clamped to the vocabulary; checked when the config is built. -/
  vocabSizePos : 0 < vocabSize := by decide
  /-- Layer norms divide by `sqrt (var + eps)`; checked when the config is built. -/
  layerNormEpsPos : 0 < layerNormEps := by native_decide
  /-- Special and suppressed tokens index the vocabulary; checked when the config is built. -/
  eosInVocab : eosTokenId.toNat < vocabSize := by decide
  startInVocab : decoderStartTokenId.toNat < vocabSize := by decide
  suppressInVocab : ∀ t ∈ suppressTokens, t.toNat < vocabSize := by decide
  beginSuppressInVocab : ∀ t ∈ beginSuppressTokens, t.toNat < vocabSize := by decide
  activationSupported : activationFunction = "gelu" ∨ activationFunction = "relu" := by decide
  deriving Repr


namespace WhisperConfig

def encoderHeadDim (cfg : WhisperConfig) : Nat :=
  cfg.dModel / cfg.encoderAttentionHeads

def decoderHeadDim (cfg : WhisperConfig) : Nat :=
  cfg.dModel / cfg.decoderAttentionHeads

def conv2OutputSeq (frames : Nat) : Nat :=
  (frames + 2 - (3 - 1) - 1) / 2 + 1

end WhisperConfig

end torch.whisper
