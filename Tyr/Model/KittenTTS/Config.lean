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
  /-- Layers split evenly across a nonzero number of groups; checked when the config is built. -/
  hiddenGroupsValid : 0 < numHiddenGroups ∧ numHiddenGroups ∣ numHiddenLayers := by decide
  /-- The hidden size splits evenly across a nonzero number of heads; checked when the config is built. -/
  headsDivide : 0 < numAttentionHeads ∧ numAttentionHeads ∣ hiddenSize := by decide
  /-- Token type 0 has an embedding row; checked when the config is built. -/
  typeVocabPos : 0 < typeVocabSize := by decide
  /-- Layer norms divide by `sqrt (var + eps)`; checked when the config is built. -/
  layerNormEpsPos : 0 < layerNormEps := by native_decide
  deriving Repr

namespace AlbertConfig

def headDim (cfg : AlbertConfig) : Nat :=
  cfg.hiddenSize / cfg.numAttentionHeads

@[simp] def allHeadSize (cfg : AlbertConfig) : Nat :=
  cfg.numAttentionHeads * cfg.headDim

def layersPerGroup (cfg : AlbertConfig) : Nat :=
  cfg.numHiddenLayers / cfg.numHiddenGroups

end AlbertConfig

private theorem foldl_mul_acc (l : List Nat) (acc : Nat) :
    l.foldl (· * ·) acc = acc * l.foldl (· * ·) 1 := by
  induction l generalizing acc with
  | nil => simp
  | cons x xs ih => simp only [List.foldl]; rw [ih, ih (1 * x)]; simp [Nat.mul_assoc]

private theorem foldl_mul_append (a b : List Nat) :
    (a ++ b).foldl (· * ·) 1 = a.foldl (· * ·) 1 * b.foldl (· * ·) 1 := by
  rw [List.foldl_append, foldl_mul_acc b]

private theorem foldl_mul_pos (l : List Nat) (h : ∀ x ∈ l, 0 < x) : 0 < l.foldl (· * ·) 1 := by
  induction l with
  | nil => simp
  | cons x xs ih =>
    rw [List.foldl, foldl_mul_acc]
    exact Nat.mul_pos (by simpa using h x (by simp)) (ih fun y hy => h y (by simp [hy]))

structure GeneratorConfig where
  resblockKernelSizes : Array Nat := #[3, 7, 11]
  upsampleRates : Array Nat := #[10, 6]
  upsampleInitialChannel : Nat := 512
  resblockDilationSizes : Array (Array Nat) := #[#[1, 3, 5], #[1, 3, 5], #[1, 3, 5]]
  upsampleKernelSizes : Array Nat := #[20, 12]
  genIstftNFft : Nat := 20
  genIstftHopSize : Nat := 5
  harmonicCount : Nat := 8
  /-- Each upsample stage maps `L` frames to `L * rate` (kernel at least the rate, with an even
      difference split as padding) and every rate exceeds one; checked when the config is built. -/
  stagesValid : 0 < upsampleRates.size ∧ ∀ i, i < upsampleRates.size →
      1 < upsampleRates.getD i 1 ∧ upsampleRates.getD i 1 ≤ upsampleKernelSizes.getD i 0 ∧
      2 ∣ upsampleKernelSizes.getD i 0 - upsampleRates.getD i 1 := by decide
  /-- Residual blocks use odd kernels so their padded convolutions keep the frame count;
      checked when the config is built. -/
  resblocksValid : 0 < resblockKernelSizes.size ∧
      ∀ j, j < resblockKernelSizes.size → resblockKernelSizes.getD j 0 % 2 = 1 := by decide
  /-- One upsample kernel per rate; checked when the config is built. -/
  kernelsMatchRates : upsampleKernelSizes.size = upsampleRates.size := by decide
  /-- Each residual block has three positive dilations, one per unit; checked when the config is built. -/
  dilationsValid : resblockDilationSizes.size = resblockKernelSizes.size ∧
      ∀ j, j < resblockDilationSizes.size → (resblockDilationSizes.getD j #[]).size = 3 ∧
        ∀ m, m < 3 → 0 < (resblockDilationSizes.getD j #[]).getD m 0 := by decide
  /-- Every upsample stage halves the channels and keeps at least one; checked when the config is built. -/
  channelsValid : 2 ^ upsampleRates.size ≤ upsampleInitialChannel := by decide
  /-- The iSTFT has a positive hop and a positive even FFT size, so its `nFft + 2` channels split
      into two `nFft / 2 + 1` halves; checked when the config is built. -/
  istftValid : 0 < genIstftHopSize ∧ 0 < genIstftNFft ∧ 2 ∣ genIstftNFft := by decide
  deriving Repr


namespace GeneratorConfig

def rate (cfg : GeneratorConfig) (i : Nat) : Nat :=
  cfg.upsampleRates.getD i 1

def upKernel (cfg : GeneratorConfig) (i : Nat) : Nat :=
  cfg.upsampleKernelSizes.getD i 0

def resKernel (cfg : GeneratorConfig) (j : Nat) : Nat :=
  cfg.resblockKernelSizes.getD j 0

def resDilation (cfg : GeneratorConfig) (j m : Nat) : Nat :=
  (cfg.resblockDilationSizes.getD j #[]).getD m 0

def totalUpsample (cfg : GeneratorConfig) : Nat :=
  cfg.upsampleRates.toList.foldl (· * ·) 1

def sourceUpsample (cfg : GeneratorConfig) : Nat :=
  totalUpsample cfg * cfg.genIstftHopSize

def freqBins (cfg : GeneratorConfig) : Nat :=
  cfg.genIstftNFft / 2 + 1

@[simp] def stftFeatureChannels (cfg : GeneratorConfig) : Nat :=
  2 * freqBins cfg

def finalChannels (cfg : GeneratorConfig) : Nat :=
  cfg.upsampleInitialChannel / 2 ^ cfg.upsampleRates.size

def prefixUpsample (cfg : GeneratorConfig) (count : Nat) : Nat :=
  (cfg.upsampleRates.toList.take count).foldl (· * ·) 1

def suffixUpsample (cfg : GeneratorConfig) (start : Nat) : Nat :=
  (cfg.upsampleRates.toList.drop start).foldl (· * ·) 1

theorem prefix_mul_suffix (cfg : GeneratorConfig) (c : Nat) :
    cfg.prefixUpsample c * cfg.suffixUpsample c = cfg.totalUpsample := by
  unfold prefixUpsample suffixUpsample totalUpsample
  rw [← foldl_mul_append, List.take_append_drop]

private theorem rates_pos (cfg : GeneratorConfig) : ∀ x ∈ cfg.upsampleRates.toList, 0 < x := by
  intro x hx
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hx
  have := (cfg.stagesValid.2 i (by simpa using hi)).1
  simp [Array.getD] at this
  simp at hi
  simp [hi] at this
  simp; omega

theorem prefix_pos (cfg : GeneratorConfig) (c : Nat) : 0 < cfg.prefixUpsample c :=
  foldl_mul_pos _ fun x hx => cfg.rates_pos x (List.mem_of_mem_take hx)

theorem suffix_pos (cfg : GeneratorConfig) (s : Nat) : 0 < cfg.suffixUpsample s :=
  foldl_mul_pos _ fun x hx => cfg.rates_pos x (List.mem_of_mem_drop hx)

theorem prefix_succ (cfg : GeneratorConfig) {i : Nat} (h : i < cfg.upsampleRates.size) :
    cfg.prefixUpsample (i + 1) = cfg.prefixUpsample i * cfg.rate i := by
  unfold prefixUpsample rate
  rw [List.take_add_one, foldl_mul_append]
  simp [h]

theorem suffix_eq (cfg : GeneratorConfig) {s : Nat} (h : s < cfg.upsampleRates.size) :
    cfg.suffixUpsample s = cfg.rate s * cfg.suffixUpsample (s + 1) := by
  unfold suffixUpsample rate
  rw [List.drop_eq_getElem_cons (by simpa using h), List.foldl, foldl_mul_acc]
  simp [h]

theorem prefix_size (cfg : GeneratorConfig) :
    cfg.prefixUpsample cfg.upsampleRates.size = cfg.totalUpsample := by
  unfold prefixUpsample totalUpsample
  rw [show cfg.upsampleRates.size = cfg.upsampleRates.toList.length by simp, List.take_length]

theorem rate_gt_one (cfg : GeneratorConfig) {i : Nat} (h : i < cfg.upsampleRates.size) : 1 < cfg.rate i :=
  (cfg.stagesValid.2 i h).1

theorem suffix_ge_two (cfg : GeneratorConfig) {s : Nat} (h : s < cfg.upsampleRates.size) :
    2 ≤ cfg.suffixUpsample s := by
  rw [cfg.suffix_eq h]
  calc 2 ≤ cfg.rate s := cfg.rate_gt_one h
    _ ≤ _ := Nat.le_mul_of_pos_right _ (cfg.suffix_pos _)

def stageInChannels (cfg : GeneratorConfig) (idx : Nat) : Nat :=
  cfg.upsampleInitialChannel / 2 ^ idx

def stageOutChannels (cfg : GeneratorConfig) (idx : Nat) : Nat :=
  cfg.upsampleInitialChannel / 2 ^ (idx + 1)

theorem stageInChannels_zero (cfg : GeneratorConfig) :
    cfg.stageInChannels 0 = cfg.upsampleInitialChannel := by
  simp [stageInChannels]

theorem finalChannels_last (cfg : GeneratorConfig) {i : Nat} (h : i + 1 = cfg.upsampleRates.size) :
    cfg.finalChannels = cfg.stageOutChannels i := by
  simp [finalChannels, stageOutChannels, ← h]

def stageInFrames (cfg : GeneratorConfig) (baseFrames : Nat) (idx : Nat) : Nat :=
  baseFrames * prefixUpsample cfg idx

def stageUpsampledFrames (cfg : GeneratorConfig) (baseFrames : Nat) (idx : Nat) : Nat :=
  baseFrames * prefixUpsample cfg (idx + 1)

@[simp] def harFrames (cfg : GeneratorConfig) (baseFrames : Nat) : Nat :=
  totalUpsample cfg * baseFrames + 1

def stageOutFrames (cfg : GeneratorConfig) (baseFrames : Nat) (idx : Nat) : Nat :=
  if idx + 1 < cfg.upsampleRates.size then
    stageUpsampledFrames cfg baseFrames idx
  else
    harFrames cfg baseFrames

def waveSamples (cfg : GeneratorConfig) (baseFrames : Nat) : Nat :=
  baseFrames * sourceUpsample cfg

theorem waveSamples_pos (cfg : GeneratorConfig) {baseFrames : Nat} (hb : 0 < baseFrames) :
    0 < cfg.waveSamples baseFrames :=
  Nat.mul_pos hb (Nat.mul_pos (cfg.prefix_size ▸ cfg.prefix_pos _) cfg.istftValid.1)

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
  nToken : Nat := 178
  styleDim : Nat := 128
  textEncoderKernelSize : Nat := 5
  asrResDim : Nat := 64
  decoderOutDim : Nat := 0
  sampleRate : Nat := 24000
  plbert : AlbertConfig := {}
  generator : GeneratorConfig := {}
  /-- The bidirectional LSTMs emit `2 * (hiddenDim / 2)` channels; checked when the config is built. -/
  hiddenDimEven : 2 ∣ hiddenDim := by decide
  /-- Text-encoder convolutions pad by half an odd kernel to keep the token count; checked when
      the config is built. -/
  textKernelOdd : textEncoderKernelSize % 2 = 1 := by decide
  /-- Waveform phases divide by the sample rate; checked when the config is built. -/
  sampleRatePos : 0 < sampleRate := by decide
  /-- Durations are predicted over `maxDur` bins; checked when the config is built. -/
  maxDurPos : 0 < maxDur := by decide
  /-- The decoder feeds the generator's first stage; checked when the config is built. -/
  decoderMatchesGenerator :
    (if decoderOutDim == 0 then generator.upsampleInitialChannel else decoderOutDim) =
      generator.upsampleInitialChannel := by decide
  deriving Repr


namespace KittenTTSConfig

def fullStyleDim (cfg : KittenTTSConfig) : Nat :=
  2 * cfg.styleDim

def decoderChannels (cfg : KittenTTSConfig) : Nat :=
  if cfg.decoderOutDim == 0 then cfg.generator.upsampleInitialChannel else cfg.decoderOutDim

end KittenTTSConfig

end torch.kittentts
