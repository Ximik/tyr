/-
  Tyr/Model/KittenTTS/Config.lean

  Configuration for a Lean4 KittenTTS port.
  This mirrors the upstream model at the level needed for inference:
  - ALBERT-style phoneme encoder
  - duration / prosody predictor
  - AdaIN decoder
  - lightweight generator configuration
 -/
import Tyr.Basic

namespace torch.kittentts

structure AlbertConfig where
  numHiddenLayers : Nat := 12
  numAttentionHeads : Nat := 12
  hiddenSize : Nat := 768
  intermediateSize : Nat := 3072
  maxPositionEmbeddings : Nat := 512
  embeddingSize : Nat := 128
  innerGroupNum : Nat := 1
  numHiddenGroups : Nat := 1
  typeVocabSize : Nat := 2
  layerNormEps : Float := 1e-12
  hiddenDropoutProb : Float := 0.0
  attentionDropoutProb : Float := 0.0
  deriving Repr, Inhabited

namespace AlbertConfig

def headDim (cfg : AlbertConfig) : Nat :=
  if cfg.numAttentionHeads == 0 then 0 else cfg.hiddenSize / cfg.numAttentionHeads

def allHeadSize (cfg : AlbertConfig) : Nat :=
  cfg.numAttentionHeads * cfg.headDim

def layersPerGroup (cfg : AlbertConfig) : Nat :=
  if cfg.numHiddenGroups == 0 then 0 else cfg.numHiddenLayers / cfg.numHiddenGroups

end AlbertConfig

structure GeneratorConfig where
  resblockKernelSizes : Array Nat := #[3, 7, 11]
  upsampleRates : Array Nat := #[5, 5, 4, 3]
  upsampleInitialChannel : Nat := 512
  resblockDilationSizes : Array (Array Nat) := #[#[1, 3, 5], #[1, 3, 5], #[1, 3, 5]]
  upsampleKernelSizes : Array Nat := #[10, 10, 8, 6]
  genIstftNFft : Nat := 20
  genIstftHopSize : Nat := 5
  harmonicCount : Nat := 8
  deriving Repr, Inhabited

namespace GeneratorConfig

private def pow2 : Nat → Nat
  | 0 => 1
  | n + 1 => 2 * pow2 n

def totalUpsample (cfg : GeneratorConfig) : Nat :=
  cfg.upsampleRates.foldl (fun acc x => acc * x) 1

def sourceUpsample (cfg : GeneratorConfig) : Nat :=
  totalUpsample cfg * cfg.genIstftHopSize

def freqBins (cfg : GeneratorConfig) : Nat :=
  cfg.genIstftNFft / 2 + 1

def stftFeatureChannels (cfg : GeneratorConfig) : Nat :=
  2 * freqBins cfg

def finalChannels (cfg : GeneratorConfig) : Nat :=
  if cfg.upsampleRates.isEmpty then
    cfg.upsampleInitialChannel
  else
    cfg.upsampleInitialChannel / pow2 cfg.upsampleRates.size

private def prodPrefix (xs : Array Nat) (count : Nat) : Nat :=
  Id.run do
    let mut acc := 1
    for i in [:min count xs.size] do
      acc := acc * xs.getD i 1
    acc

def prefixUpsample (cfg : GeneratorConfig) (count : Nat) : Nat :=
  prodPrefix cfg.upsampleRates count

def suffixUpsample (cfg : GeneratorConfig) (start : Nat) : Nat :=
  Id.run do
    let mut acc := 1
    for i in [start:cfg.upsampleRates.size] do
      acc := acc * cfg.upsampleRates.getD i 1
    acc

def stageInChannels (cfg : GeneratorConfig) (idx : Nat) : Nat :=
  cfg.upsampleInitialChannel / pow2 idx

def stageOutChannels (cfg : GeneratorConfig) (idx : Nat) : Nat :=
  cfg.upsampleInitialChannel / pow2 (idx + 1)

def stageInFrames (cfg : GeneratorConfig) (baseFrames : Nat) (idx : Nat) : Nat :=
  baseFrames * prefixUpsample cfg idx

def stageUpsampledFrames (cfg : GeneratorConfig) (baseFrames : Nat) (idx : Nat) : Nat :=
  baseFrames * prefixUpsample cfg (idx + 1)

def harFrames (cfg : GeneratorConfig) (baseFrames : Nat) : Nat :=
  totalUpsample cfg * baseFrames + 1

def stageOutFrames (cfg : GeneratorConfig) (baseFrames : Nat) (idx : Nat) : Nat :=
  if idx + 1 < cfg.upsampleRates.size then
    stageUpsampledFrames cfg baseFrames idx
  else
    harFrames cfg baseFrames

def waveSamples (cfg : GeneratorConfig) (baseFrames : Nat) : Nat :=
  baseFrames * sourceUpsample cfg

def noiseStride (cfg : GeneratorConfig) (idx : Nat) : Nat :=
  if idx + 1 < cfg.upsampleRates.size then
    suffixUpsample cfg (idx + 1)
  else
    1

def noiseKernel (cfg : GeneratorConfig) (idx : Nat) : Nat :=
  if idx + 1 < cfg.upsampleRates.size then
    2 * noiseStride cfg idx
  else
    1

def noisePadding (cfg : GeneratorConfig) (idx : Nat) : Nat :=
  if idx + 1 < cfg.upsampleRates.size then
    (noiseStride cfg idx + 1) / 2
  else
    0

end GeneratorConfig

structure KittenTTSConfig where
  hiddenDim : Nat := 512
  maxConvDim : Nat := 512
  maxDur : Nat := 50
  nLayer : Nat := 3
  nMels : Nat := 80
  nToken : Nat := 178
  styleDim : Nat := 128
  textEncoderKernelSize : Nat := 5
  asrResDim : Nat := 64
  decoderOutDim : Nat := 0
  sampleRate : Nat := 24000
  plbert : AlbertConfig := {}
  generator : GeneratorConfig := {}
  deriving Repr, Inhabited

namespace KittenTTSConfig

def fullStyleDim (cfg : KittenTTSConfig) : Nat :=
  2 * cfg.styleDim

def decoderChannels (cfg : KittenTTSConfig) : Nat :=
  if cfg.decoderOutDim == 0 then cfg.generator.upsampleInitialChannel else cfg.decoderOutDim

end KittenTTSConfig

end torch.kittentts
