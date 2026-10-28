/-
  Tyr/Model/Whisper/Config.lean

  Configuration for Whisper encoder-decoder ASR in Tyr.
-/
import Tyr.Basic

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
  activationFunction : String := "gelu"
  layerNormEps : Float := 1e-5
  padTokenId : UInt64 := 50257
  bosTokenId : UInt64 := 50257
  eosTokenId : UInt64 := 50257
  decoderStartTokenId : UInt64 := 50258
  suppressTokens : Array Nat := #[]
  beginSuppressTokens : Array Nat := #[]
  deriving Repr, Inhabited

namespace WhisperConfig

def encoderHeadDim (cfg : WhisperConfig) : Nat :=
  cfg.dModel / cfg.encoderAttentionHeads

def decoderHeadDim (cfg : WhisperConfig) : Nat :=
  cfg.dModel / cfg.decoderAttentionHeads

def conv2OutputSeq (frames : Nat) : Nat :=
  (frames + 2 - (3 - 1) - 1) / 2 + 1

def hasValidHeads (cfg : WhisperConfig) : Bool :=
  cfg.encoderAttentionHeads > 0 &&
  cfg.decoderAttentionHeads > 0 &&
  cfg.dModel % cfg.encoderAttentionHeads == 0 &&
  cfg.dModel % cfg.decoderAttentionHeads == 0

end WhisperConfig

end torch.whisper
