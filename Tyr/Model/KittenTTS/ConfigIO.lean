/-
  Tyr/Model/KittenTTS/ConfigIO.lean

  Kokoro / KittenTTS `config.json` loader.
-/
import Tyr.Model.KittenTTS.Config
import Tyr.Model.ConfigJson

namespace torch.kittentts

open Lean ConfigJson

abbrev VocabMap := Std.HashMap Char UInt64

namespace AlbertConfig

def loadFromJson (root : Json) (defaults : AlbertConfig := {}) : Except String AlbertConfig := do
  let numHiddenLayers ← fieldD? root "num_hidden_layers" defaults.numHiddenLayers
  let numHiddenGroups ← fieldD? root "num_hidden_groups" defaults.numHiddenGroups
  let ⟨hGroups⟩ ← if h : 0 < numHiddenGroups ∧ numHiddenGroups ∣ numHiddenLayers then
      pure (PLift.up h)
    else
      throw s!"Invalid KittenTTS plbert config: num_hidden_groups={numHiddenGroups} must be positive and divide num_hidden_layers={numHiddenLayers}"
  let numAttentionHeads ← fieldD? root "num_attention_heads" defaults.numAttentionHeads
  let hiddenSize ← fieldD? root "hidden_size" defaults.hiddenSize
  let ⟨hHeads⟩ ← if h : 0 < numAttentionHeads ∧ numAttentionHeads ∣ hiddenSize then
      pure (PLift.up h)
    else
      throw s!"Invalid KittenTTS plbert config: num_attention_heads={numAttentionHeads} must be positive and divide hidden_size={hiddenSize}"
  let layerNormEps ← fieldD? root "layer_norm_eps" defaults.layerNormEps
  let ⟨hEps⟩ ← if h : 0 < layerNormEps then pure (PLift.up h) else
    throw s!"Invalid KittenTTS plbert config: layer_norm_eps={layerNormEps} must be positive"
  let typeVocabSize ← fieldD? root "type_vocab_size" defaults.typeVocabSize
  let ⟨hTypes⟩ ← if h : 0 < typeVocabSize then pure (PLift.up h) else
    throw s!"Invalid KittenTTS plbert config: type_vocab_size={typeVocabSize} must hold token type 0"
  requireOneOf root "hidden_act" ["gelu_new", "gelu_pytorch_tanh"]
  pure {
    numHiddenLayers
    numAttentionHeads
    hiddenSize
    intermediateSize := ← fieldD? root "intermediate_size" defaults.intermediateSize
    maxPositionEmbeddings := ← fieldD? root "max_position_embeddings" defaults.maxPositionEmbeddings
    embeddingSize := ← fieldD? root "embedding_size" defaults.embeddingSize
    innerGroupNum := ← fieldD? root "inner_group_num" defaults.innerGroupNum
    numHiddenGroups
    typeVocabSize
    layerNormEps
    hiddenGroupsValid := hGroups
    headsDivide := hHeads
    typeVocabPos := hTypes
    layerNormEpsPos := hEps
  }

end AlbertConfig

namespace GeneratorConfig

def loadFromJson (root : Json) (defaults : GeneratorConfig := {}) : Except String GeneratorConfig := do
  let resblockKernelSizes ← fieldD? root "resblock_kernel_sizes" defaults.resblockKernelSizes
  let upsampleRates ← fieldD? root "upsample_rates" defaults.upsampleRates
  let upsampleKernelSizes ← fieldD? root "upsample_kernel_sizes" defaults.upsampleKernelSizes
  let upsampleInitialChannel ← fieldD? root "upsample_initial_channel" defaults.upsampleInitialChannel
  let resblockDilationSizes ← fieldD? root "resblock_dilation_sizes" defaults.resblockDilationSizes
  let genIstftNFft ← fieldD? root "gen_istft_n_fft" defaults.genIstftNFft
  let genIstftHopSize ← fieldD? root "gen_istft_hop_size" defaults.genIstftHopSize
  let ⟨hStages⟩ ← if h : 0 < upsampleRates.size ∧ ∀ i, i < upsampleRates.size →
      1 < upsampleRates.getD i 1 ∧ upsampleRates.getD i 1 ≤ upsampleKernelSizes.getD i 0 ∧
      2 ∣ upsampleKernelSizes.getD i 0 - upsampleRates.getD i 1 then pure (PLift.up h) else
    throw s!"Invalid KittenTTS generator config: upsample_rates={upsampleRates} must be nonempty, each above 1, with upsample_kernel_sizes={upsampleKernelSizes} at least the rate and differing from it by an even amount"
  let ⟨hKernels⟩ ← if h : upsampleKernelSizes.size = upsampleRates.size then pure (PLift.up h) else
    throw s!"Invalid KittenTTS generator config: upsample_kernel_sizes={upsampleKernelSizes} must have one entry per upsample rate {upsampleRates}"
  let ⟨hRes⟩ ← if h : 0 < resblockKernelSizes.size ∧
      ∀ j, j < resblockKernelSizes.size → resblockKernelSizes.getD j 0 % 2 = 1 then pure (PLift.up h) else
    throw s!"Invalid KittenTTS generator config: resblock_kernel_sizes={resblockKernelSizes} must be nonempty and odd"
  let ⟨hDilations⟩ ← if h : resblockDilationSizes.size = resblockKernelSizes.size ∧
      ∀ j, j < resblockDilationSizes.size → (resblockDilationSizes.getD j #[]).size = 3 ∧
        ∀ m, m < 3 → 0 < (resblockDilationSizes.getD j #[]).getD m 0 then pure (PLift.up h) else
    throw s!"Invalid KittenTTS generator config: resblock_dilation_sizes={resblockDilationSizes} must hold three positive dilations per resblock kernel {resblockKernelSizes}"
  let ⟨hChannels⟩ ← if h : 2 ^ upsampleRates.size ≤ upsampleInitialChannel then pure (PLift.up h) else
    throw s!"Invalid KittenTTS generator config: upsample_initial_channel={upsampleInitialChannel} must keep a channel after halving once per upsample stage ({upsampleRates.size})"
  let ⟨hIstft⟩ ← if h : 0 < genIstftHopSize ∧ 0 < genIstftNFft ∧ 2 ∣ genIstftNFft then pure (PLift.up h) else
    throw s!"Invalid KittenTTS generator config: gen_istft_hop_size={genIstftHopSize} must be positive and gen_istft_n_fft={genIstftNFft} positive and even"
  pure {
    resblockKernelSizes
    upsampleRates
    upsampleInitialChannel
    resblockDilationSizes
    upsampleKernelSizes
    genIstftNFft
    genIstftHopSize
    harmonicCount := ← fieldD? root "harmonic_num" defaults.harmonicCount
    stagesValid := hStages
    resblocksValid := hRes
    kernelsMatchRates := hKernels
    dilationsValid := hDilations
    channelsValid := hChannels
    istftValid := hIstft
  }

end GeneratorConfig

namespace KittenTTSConfig

def loadFromJson (root : Json) (defaults : KittenTTSConfig := {}) : Except String KittenTTSConfig := do
  let plbert ← match ← objectField? root "plbert" with
    | some j => AlbertConfig.loadFromJson j defaults.plbert
    | none => pure defaults.plbert
  let generator ← match ← objectField? root "istftnet" with
    | some j => GeneratorConfig.loadFromJson j defaults.generator
    | none => pure defaults.generator
  let hiddenDim ← fieldD? root "hidden_dim" defaults.hiddenDim
  let maxDur ← fieldD? root "max_dur" defaults.maxDur
  let ⟨hDur⟩ ← if h : 0 < maxDur then pure (PLift.up h) else
    throw s!"Invalid KittenTTS config: max_dur={maxDur} must be positive"
  let textEncoderKernelSize ← fieldD? root "text_encoder_kernel_size" defaults.textEncoderKernelSize
  let decoderOutDim ← fieldD? root "decoder_out_dim" defaults.decoderOutDim
  let sampleRate ← fieldD? root "sample_rate" defaults.sampleRate
  let ⟨hRate⟩ ← if h : 0 < sampleRate then pure (PLift.up h) else
    throw s!"Invalid KittenTTS config: sample_rate={sampleRate} must be positive"
  let ⟨hHidden⟩ ← if h : 2 ∣ hiddenDim then pure (PLift.up h) else
    throw s!"Invalid KittenTTS config: hidden_dim={hiddenDim} must be divisible by 2"
  let ⟨hKernel⟩ ← if h : textEncoderKernelSize % 2 = 1 then pure (PLift.up h) else
    throw s!"Invalid KittenTTS config: text_encoder_kernel_size={textEncoderKernelSize} must be odd"
  let ⟨hDecoder⟩ ← if h : (if decoderOutDim == 0 then generator.upsampleInitialChannel else decoderOutDim) =
      generator.upsampleInitialChannel then pure (PLift.up h) else
    throw s!"Invalid KittenTTS config: decoder_out_dim={decoderOutDim} must match upsample_initial_channel={generator.upsampleInitialChannel}"
  pure {
    hiddenDim
    maxConvDim := ← fieldD? root "max_conv_dim" defaults.maxConvDim
    maxDur
    nLayer := ← fieldD? root "n_layer" defaults.nLayer
    nToken := ← fieldD? root "n_token" defaults.nToken
    styleDim := ← fieldD? root "style_dim" defaults.styleDim
    textEncoderKernelSize
    asrResDim := ← fieldD? root "dim_in" defaults.asrResDim
    decoderOutDim
    sampleRate
    plbert
    generator
    hiddenDimEven := hHidden
    textKernelOdd := hKernel
    sampleRatePos := hRate
    maxDurPos := hDur
    decoderMatchesGenerator := hDecoder
  }

def loadFromFile (path : String) (defaults : KittenTTSConfig := {}) : IO KittenTTSConfig := do
  IO.ofExcept (loadFromJson (← parseFile path) defaults)

def loadFromPretrainedDir (modelDir : String) (defaults : KittenTTSConfig := {}) : IO KittenTTSConfig :=
  loadFromFile s!"{modelDir}/config.json" defaults

end KittenTTSConfig

/-- Phoneme vocabulary from `config.json`: single-character keys mapped to ids below `nToken`. -/
def loadVocabFromFile (path : String) (nToken : Nat) : IO VocabMap := do
  let root ← parseFile path
  let entries ← IO.ofExcept do
    let some (.obj kvs) ← member? root "vocab"
      | throw s!"Missing or invalid vocab object in {path}"
    kvs.foldlM (init := ({} : VocabMap)) fun vocab key value => do
      let [c] := key.toList
        | throw s!"Invalid vocab key \"{key}\" in {path}: expected one character"
      let id ← match (fromJson? value : Except String Nat) with
        | .ok id => pure id
        | .error err => throw s!"Invalid vocab id for \"{key}\" in {path}: {err}"
      unless id < nToken do
        throw s!"Vocab id {id} for \"{key}\" in {path} is outside the embedding table (n_token={nToken})"
      pure (vocab.insert c id.toUInt64)
  pure entries

def loadVocabFromPretrainedDir (modelDir : String) (nToken : Nat) : IO VocabMap :=
  loadVocabFromFile s!"{modelDir}/config.json" nToken

end torch.kittentts
