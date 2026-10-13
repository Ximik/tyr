/-
  Tyr/Model/Qwen35/Multimodal.lean

  Qwen3.5 multimodal path for Tyr:
  - Vision backbone (patch embed + ViT blocks + patch merger)
  - Placeholder-based fusion into text embeddings
  - Conditional generation wrapper over existing Qwen35 text model
-/
import Tyr.Torch
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen35.Model
import Tyr.Model.Qwen35.VLConfig
import Tyr.Model.Utils

namespace torch.qwen35

open torch
open torch.Model

private def arangeOn (start stop : UInt64) (step : UInt64 := 1) (device : Device) :
    T #[(stop - start) / step] :=
  (torch.arange start stop step).to device

private def addBias2d {n d : UInt64}
    (x : T #[n, d])
    (b : T #[d])
    : T #[n, d] :=
  x + nn.expand (reshape b #[1, d]) #[n, d]

private def linear2d {tokens in_dim out_dim : UInt64}
    (x : T #[tokens, in_dim])
    (w : T #[out_dim, in_dim])
    (b : T #[out_dim])
    : T #[tokens, out_dim] :=
  let yDyn : T #[] := torch.einsum2 "oh,th->to" w x
  let y : T #[tokens, out_dim] := reshape yDyn #[tokens, out_dim]
  addBias2d y b

private def allInSet (xs : Array UInt64) (allow : Array UInt64) : Bool :=
  xs.all (fun x => allow.contains x)

private def countMask2d {batch seq : UInt64} (mask : T #[batch, seq]) : IO UInt64 := do
  let row : T #[batch] := reshape (nn.sumDim (data.toLong mask) 1 false) #[batch]
  let counts ← data.tensorToUInt64Array row
  pure (counts.foldl (· + ·) 0)

/-- LayerNorm for vision stack. -/
structure Qwen35VisionLayerNorm (dim : UInt64) where
  weight : T #[dim]
  bias : T #[dim]
  eps : Float := 1e-6
  deriving TensorStruct

namespace Qwen35VisionLayerNorm

def init (dim : UInt64) (eps : Float := 1e-6) : Qwen35VisionLayerNorm dim :=
  {
    weight := autograd.set_requires_grad (torch.ones #[dim]) true
    bias := autograd.set_requires_grad (torch.zeros #[dim]) true
    eps := eps
  }

def forward2d {tokens dim : UInt64}
    (m : Qwen35VisionLayerNorm dim)
    (x : T #[tokens, dim])
    : T #[tokens, dim] :=
  let x3 : T #[1, tokens, dim] := reshape x #[1, tokens, dim]
  let y3 : T #[1, tokens, dim] := nn.layer_norm x3 m.weight m.bias m.eps
  reshape y3 #[tokens, dim]

end Qwen35VisionLayerNorm

/-- Vision MLP block. -/
structure Qwen35VisionMLP (cfg : VisionConfig) where
  linear_fc1_weight : T #[cfg.intermediate_size, cfg.hidden_size]
  linear_fc1_bias : T #[cfg.intermediate_size]
  linear_fc2_weight : T #[cfg.hidden_size, cfg.intermediate_size]
  linear_fc2_bias : T #[cfg.hidden_size]
  deriving TensorStruct

namespace Qwen35VisionMLP

def init (cfg : VisionConfig) : IO (Qwen35VisionMLP cfg) := do
  let w1 ← initWeight #[cfg.intermediate_size, cfg.hidden_size] cfg.hidden_size
  let b1 := initBias #[cfg.intermediate_size]
  let w2 ← initWeight #[cfg.hidden_size, cfg.intermediate_size] cfg.intermediate_size
  let b2 := initBias #[cfg.hidden_size]
  pure {
    linear_fc1_weight := w1
    linear_fc1_bias := b1
    linear_fc2_weight := w2
    linear_fc2_bias := b2
  }

def forward {tokens : UInt64}
    (cfg : VisionConfig)
    (m : Qwen35VisionMLP cfg)
    (x : T #[tokens, cfg.hidden_size])
    : T #[tokens, cfg.hidden_size] :=
  let h : T #[tokens, cfg.intermediate_size] :=
    linear2d x m.linear_fc1_weight m.linear_fc1_bias
  linear2d (nn.gelu h) m.linear_fc2_weight m.linear_fc2_bias

end Qwen35VisionMLP

/-- Vision patch embed implemented as linear projection over flattened patches.
    Input is expected to already be patchified to
    `[num_patches, in_channels * temporal_patch_size * patch_size * patch_size]`. -/
structure Qwen35VisionPatchEmbed (cfg : VisionConfig) where
  weight : T #[cfg.hidden_size, VisionConfig.patchDim cfg]
  bias : T #[cfg.hidden_size]
  deriving TensorStruct

namespace Qwen35VisionPatchEmbed

def init (cfg : VisionConfig) : IO (Qwen35VisionPatchEmbed cfg) := do
  let w ← initWeight #[cfg.hidden_size, VisionConfig.patchDim cfg] (VisionConfig.patchDim cfg)
  let b := initBias #[cfg.hidden_size]
  pure { weight := w, bias := b }

def forward {nPatches : UInt64}
    (cfg : VisionConfig)
    (m : Qwen35VisionPatchEmbed cfg)
    (x : T #[nPatches, VisionConfig.patchDim cfg])
    : T #[nPatches, cfg.hidden_size] :=
  linear2d x m.weight m.bias

end Qwen35VisionPatchEmbed

/-- Vision multi-head self-attention. -/
structure Qwen35VisionAttention (cfg : VisionConfig) where
  qkv_weight : T #[cfg.hidden_size * 3, cfg.hidden_size]
  qkv_bias : T #[cfg.hidden_size * 3]
  proj_weight : T #[cfg.hidden_size, cfg.hidden_size]
  proj_bias : T #[cfg.hidden_size]
  deriving TensorStruct

namespace Qwen35VisionAttention

def init (cfg : VisionConfig) : IO (Qwen35VisionAttention cfg) := do
  let qkvW ← initWeight #[cfg.hidden_size * 3, cfg.hidden_size] cfg.hidden_size
  let qkvB := initBias #[cfg.hidden_size * 3]
  let projW ← initWeight #[cfg.hidden_size, cfg.hidden_size] cfg.hidden_size
  let projB := initBias #[cfg.hidden_size]
  pure {
    qkv_weight := qkvW
    qkv_bias := qkvB
    proj_weight := projW
    proj_bias := projB
  }

def forward {tokens : UInt64}
    (cfg : VisionConfig)
    (m : Qwen35VisionAttention cfg)
    (x : T #[tokens, cfg.hidden_size])
    : T #[tokens, cfg.hidden_size] :=
  let qkv : T #[tokens, cfg.hidden_size * 3] := linear2d x m.qkv_weight m.qkv_bias
  let q : T #[tokens, cfg.hidden_size] := data.slice qkv 1 0 cfg.hidden_size
  let k : T #[tokens, cfg.hidden_size] := data.slice qkv 1 cfg.hidden_size cfg.hidden_size
  let v : T #[tokens, cfg.hidden_size] := data.slice qkv 1 (cfg.hidden_size * 2) cfg.hidden_size

  let q : T #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg] :=
    reshape q #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg]
  let k : T #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg] :=
    reshape k #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg]
  let v : T #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg] :=
    reshape v #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg]

  -- Qwen3.5-VL uses 2D rotary; this keeps a deterministic 1D rotary signal for now.
  let (cos, sin) := rotary.computeFreqsOnDevicePure tokens (VisionConfig.headDim cfg) 10000.0 x.device
  let q : T #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg] := rotary.applyRotaryEmb q cos sin
  let k : T #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg] := rotary.applyRotaryEmb k cos sin

  let qh : T #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg] := nn.transpose_for_attention q
  let kh : T #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg] := nn.transpose_for_attention k
  let vh : T #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg] := nn.transpose_for_attention v

  let attn : T #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg] :=
    nn.scaledDotProductAttentionGQA qh kh vh 0.0 false true
  let out : T #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg] := nn.transpose_from_attention attn
  let outFlat : T #[tokens, cfg.hidden_size] := reshape out #[tokens, cfg.hidden_size]
  linear2d outFlat m.proj_weight m.proj_bias

end Qwen35VisionAttention

/-- One vision transformer block. -/
structure Qwen35VisionBlock (cfg : VisionConfig) where
  norm1 : Qwen35VisionLayerNorm cfg.hidden_size
  norm2 : Qwen35VisionLayerNorm cfg.hidden_size
  attn : Qwen35VisionAttention cfg
  mlp : Qwen35VisionMLP cfg
  deriving TensorStruct

namespace Qwen35VisionBlock

def init (cfg : VisionConfig) : IO (Qwen35VisionBlock cfg) := do
  let attn ← Qwen35VisionAttention.init cfg
  let mlp ← Qwen35VisionMLP.init cfg
  pure {
    norm1 := Qwen35VisionLayerNorm.init cfg.hidden_size 1e-6
    norm2 := Qwen35VisionLayerNorm.init cfg.hidden_size 1e-6
    attn := attn
    mlp := mlp
  }

def forward {tokens : UInt64}
    (cfg : VisionConfig)
    (m : Qwen35VisionBlock cfg)
    (x : T #[tokens, cfg.hidden_size])
    : T #[tokens, cfg.hidden_size] :=
  let h1 := x + m.attn.forward cfg (m.norm1.forward2d x)
  h1 + m.mlp.forward cfg (m.norm2.forward2d h1)

end Qwen35VisionBlock

/-- Vision patch merger from patch tokens to LLM-width tokens. -/
structure Qwen35VisionPatchMerger (cfg : VisionConfig) where
  norm : Qwen35VisionLayerNorm cfg.hidden_size
  linear_fc1_weight : T #[cfg.hidden_size * VisionConfig.mergeUnit cfg, cfg.hidden_size * VisionConfig.mergeUnit cfg]
  linear_fc1_bias : T #[cfg.hidden_size * VisionConfig.mergeUnit cfg]
  linear_fc2_weight : T #[cfg.out_hidden_size, cfg.hidden_size * VisionConfig.mergeUnit cfg]
  linear_fc2_bias : T #[cfg.out_hidden_size]
  deriving TensorStruct

namespace Qwen35VisionPatchMerger

def init (cfg : VisionConfig) : IO (Qwen35VisionPatchMerger cfg) := do
  let mergedHidden := cfg.hidden_size * VisionConfig.mergeUnit cfg
  let fc1W ← initWeight #[mergedHidden, mergedHidden] mergedHidden
  let fc1B := initBias #[mergedHidden]
  let fc2W ← initWeight #[cfg.out_hidden_size, mergedHidden] mergedHidden
  let fc2B := initBias #[cfg.out_hidden_size]
  pure {
    norm := Qwen35VisionLayerNorm.init cfg.hidden_size 1e-6
    linear_fc1_weight := fc1W
    linear_fc1_bias := fc1B
    linear_fc2_weight := fc2W
    linear_fc2_bias := fc2B
  }

def forward {tokens : UInt64}
    (cfg : VisionConfig)
    (m : Qwen35VisionPatchMerger cfg)
    (x : T #[tokens, cfg.hidden_size])
    : IO (T #[VisionConfig.mergedTokenCount cfg tokens, cfg.out_hidden_size]) := do
  let mergeUnit := VisionConfig.mergeUnit cfg
  if mergeUnit == 0 then
    throw <| IO.userError "vision spatial_merge_size must be > 0"
  if tokens % mergeUnit != 0 then
    throw <| IO.userError
      s!"vision token count ({tokens}) must be divisible by merge unit ({mergeUnit})"

  let mergedTokens := VisionConfig.mergedTokenCount cfg tokens
  let xNorm : T #[tokens, cfg.hidden_size] := m.norm.forward2d x
  let grouped : T #[mergedTokens, cfg.hidden_size * mergeUnit] :=
    reshape xNorm #[mergedTokens, cfg.hidden_size * mergeUnit]
  let h : T #[mergedTokens, cfg.hidden_size * mergeUnit] :=
    linear2d grouped m.linear_fc1_weight m.linear_fc1_bias
  pure <|
    linear2d (nn.gelu h) m.linear_fc2_weight m.linear_fc2_bias

end Qwen35VisionPatchMerger

/-- Qwen3.5 vision backbone (without DeepStack branches). -/
structure Qwen35VisionModel (cfg : VisionConfig) where
  patch_embed : Qwen35VisionPatchEmbed cfg
  pos_embed : T #[cfg.num_position_embeddings, cfg.hidden_size]
  blocks : Array (Qwen35VisionBlock cfg)
  merger : Qwen35VisionPatchMerger cfg
  deriving TensorStruct

namespace Qwen35VisionModel

def init (cfg : VisionConfig) : IO (Qwen35VisionModel cfg) := do
  let patchEmbed ← Qwen35VisionPatchEmbed.init cfg
  let posRaw ← torch.randn #[cfg.num_position_embeddings, cfg.hidden_size]
  let posEmbed := autograd.set_requires_grad (mul_scalar posRaw 0.02) true

  let mut blocks : Array (Qwen35VisionBlock cfg) := #[]
  for _ in [:cfg.depth.toNat] do
    blocks := blocks.push (← Qwen35VisionBlock.init cfg)

  let merger ← Qwen35VisionPatchMerger.init cfg
  pure {
    patch_embed := patchEmbed
    pos_embed := posEmbed
    blocks := blocks
    merger := merger
  }

def forward {nPatches : UInt64}
    (cfg : VisionConfig)
    (m : Qwen35VisionModel cfg)
    (patchifiedPixels : T #[nPatches, VisionConfig.patchDim cfg])
    : IO (T #[VisionConfig.mergedTokenCount cfg nPatches, cfg.out_hidden_size]) := do
  if nPatches > cfg.num_position_embeddings then
    throw <| IO.userError
      s!"vision patch count ({nPatches}) exceeds num_position_embeddings ({cfg.num_position_embeddings})"

  let x0 : T #[nPatches, cfg.hidden_size] := m.patch_embed.forward cfg patchifiedPixels
  let posIds : T #[nPatches] := arangeOn 0 nPatches 1 patchifiedPixels.device
  let pos : T #[nPatches, cfg.hidden_size] := nn.embedding1d posIds m.pos_embed
  let mut x : T #[nPatches, cfg.hidden_size] := x0 + pos

  for i in [:m.blocks.size] do
    match m.blocks[i]? with
    | some blk =>
      x := blk.forward cfg x
    | none =>
      throw <| IO.userError s!"missing Qwen35 vision block at index {i}"

  m.merger.forward cfg x

end Qwen35VisionModel

/-- Full Qwen3.5 multimodal wrapper (vision + text causal LM). -/
structure Qwen35ForConditionalGeneration (cfg : VLConfig) where
  visual : Qwen35VisionModel cfg.vision_config
  language_model : Qwen35ForCausalLM cfg.text_config
  deriving TensorStruct

namespace Qwen35ForConditionalGeneration

def init (cfg : VLConfig) : IO (Qwen35ForConditionalGeneration cfg) := do
  if cfg.vision_config.out_hidden_size != cfg.text_config.hidden_size then
    throw <| IO.userError
      s!"vision out_hidden_size ({cfg.vision_config.out_hidden_size}) must match text hidden_size ({cfg.text_config.hidden_size})"
  let visual ← Qwen35VisionModel.init cfg.vision_config
  let lm ← Qwen35ForCausalLM.init cfg.text_config cfg.tie_word_embeddings
  pure { visual := visual, language_model := lm }

def getImageFeatures {nPatches : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (pixelValues : T #[nPatches, VisionConfig.patchDim cfg.vision_config])
    : IO (T #[VisionConfig.mergedTokenCount cfg.vision_config nPatches, cfg.vision_config.out_hidden_size]) :=
  m.visual.forward cfg.vision_config pixelValues

def getVideoFeatures {nPatches : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (pixelValuesVideos : T #[nPatches, VisionConfig.patchDim cfg.vision_config])
    : IO (T #[VisionConfig.mergedTokenCount cfg.vision_config nPatches, cfg.vision_config.out_hidden_size]) :=
  -- Qwen3.5 reference uses the same path for video/image features.
  getImageFeatures cfg m pixelValuesVideos

private def scatterFeaturesIntoToken {batch seq hidden featTokens featDim : UInt64}
    (inputIds : T #[batch, seq])
    (inputsEmbeds : T #[batch, seq, hidden])
    (tokenId : UInt64)
    (features : T #[featTokens, featDim])
    : IO (T #[batch, seq, hidden]) := do
  if featDim != hidden then
    throw <| IO.userError
      s!"feature hidden size ({featDim}) does not match text hidden size ({hidden})"

  let tokenMask2d : T #[batch, seq] := eq_scalar inputIds (Int64.ofNat tokenId.toNat)
  let nTokens ← countMask2d tokenMask2d
  if nTokens != featTokens then
    throw <| IO.userError
      s!"placeholder token count mismatch for token={tokenId}: ids={nTokens} features={featTokens}"

  let tokenMask : T #[batch, seq, hidden] :=
    nn.expand (reshape tokenMask2d #[batch, seq, 1]) #[batch, seq, hidden]
  let src : T #[featTokens * featDim] := reshape features #[featTokens * featDim]
  pure (nn.masked_scatter inputsEmbeds tokenMask src)

def forwardText {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (attnMask : Option (T #[batch, seq]) := none)
    : T #[batch, seq, cfg.text_config.vocab_size] :=
  m.language_model.forward cfg.text_config inputIds attnMask

def forwardWithImageFeatures {batch seq featTokens : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (imageFeatures : T #[featTokens, cfg.vision_config.out_hidden_size])
    (attnMask : Option (T #[batch, seq]) := none)
    : IO (T #[batch, seq, cfg.text_config.vocab_size]) := do
  let embeds0 : T #[batch, seq, cfg.text_config.hidden_size] := m.language_model.embedTokens inputIds
  let embeds ← scatterFeaturesIntoToken
    (inputIds := inputIds)
    (inputsEmbeds := embeds0)
    cfg.image_token_id
    imageFeatures
  pure (m.language_model.forwardEmbeds cfg.text_config embeds attnMask)

def forwardWithImageAndVideoFeatures {batch seq imageTokens videoTokens : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (imageFeatures : T #[imageTokens, cfg.vision_config.out_hidden_size])
    (videoFeatures : T #[videoTokens, cfg.vision_config.out_hidden_size])
    (attnMask : Option (T #[batch, seq]) := none)
    : IO (T #[batch, seq, cfg.text_config.vocab_size]) := do
  let embeds0 : T #[batch, seq, cfg.text_config.hidden_size] := m.language_model.embedTokens inputIds
  let embeds1 ← scatterFeaturesIntoToken
    (inputIds := inputIds)
    (inputsEmbeds := embeds0)
    cfg.image_token_id
    imageFeatures
  let embeds2 ← scatterFeaturesIntoToken
    (inputIds := inputIds)
    (inputsEmbeds := embeds1)
    cfg.video_token_id
    videoFeatures
  pure (m.language_model.forwardEmbeds cfg.text_config embeds2 attnMask)

def forwardWithImagePatches {batch seq nPatches : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (pixelValues : T #[nPatches, VisionConfig.patchDim cfg.vision_config])
    (attnMask : Option (T #[batch, seq]) := none)
    : IO (T #[batch, seq, cfg.text_config.vocab_size]) := do
  let imageFeatures ← m.getImageFeatures cfg pixelValues
  m.forwardWithImageFeatures cfg inputIds imageFeatures attnMask

def forwardWithImageAndVideoPatches {batch seq nImagePatches nVideoPatches : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (pixelValues : T #[nImagePatches, VisionConfig.patchDim cfg.vision_config])
    (pixelValuesVideos : T #[nVideoPatches, VisionConfig.patchDim cfg.vision_config])
    (attnMask : Option (T #[batch, seq]) := none)
    : IO (T #[batch, seq, cfg.text_config.vocab_size]) := do
  let imageFeatures ← m.getImageFeatures cfg pixelValues
  let videoFeatures ← m.getVideoFeatures cfg pixelValuesVideos
  m.forwardWithImageAndVideoFeatures cfg inputIds imageFeatures videoFeatures attnMask

private def buildInputEmbedsWithFeatures {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (imageFeatures : Option (Sigma (fun n => T #[n, cfg.vision_config.out_hidden_size])) := none)
    (videoFeatures : Option (Sigma (fun n => T #[n, cfg.vision_config.out_hidden_size])) := none)
    : IO (T #[batch, seq, cfg.text_config.hidden_size]) := do
  let embeds0 : T #[batch, seq, cfg.text_config.hidden_size] := m.language_model.embedTokens inputIds
  let embeds1 ←
    match imageFeatures with
    | some ⟨_ni, ifeat⟩ =>
      scatterFeaturesIntoToken
        (inputIds := inputIds)
        (inputsEmbeds := embeds0)
        cfg.image_token_id
        ifeat
    | none => pure embeds0
  let embeds2 ←
    match videoFeatures with
    | some ⟨_nv, vfeat⟩ =>
      scatterFeaturesIntoToken
        (inputIds := inputIds)
        (inputsEmbeds := embeds1)
        cfg.video_token_id
        vfeat
    | none => pure embeds1
  pure embeds2

/-- Cached multimodal generation by embedding-feature injection at prefill time. -/
def generate {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (Sigma (fun n => T #[n, cfg.vision_config.out_hidden_size])) := none)
    (videoFeatures : Option (Sigma (fun n => T #[n, cfg.vision_config.out_hidden_size])) := none)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  let inputsEmbeds ← buildInputEmbedsWithFeatures cfg m inputIds imageFeatures videoFeatures
  m.language_model.generateFromEmbeds
    cfg.text_config
    inputIds
    inputsEmbeds
    maxNewTokens
    strategy
    eosTokenIds

/-- Cached multimodal generation with per-step token callback (streaming). -/
def generateStream {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (Sigma (fun n => T #[n, cfg.vision_config.out_hidden_size])) := none)
    (videoFeatures : Option (Sigma (fun n => T #[n, cfg.vision_config.out_hidden_size])) := none)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  let inputsEmbeds ← buildInputEmbedsWithFeatures cfg m inputIds imageFeatures videoFeatures
  m.language_model.generateFromEmbedsStream
    cfg.text_config
    inputIds
    inputsEmbeds
    onStep
    maxNewTokens
    strategy
    eosTokenIds

/-- Uncached multimodal generation (greedy/multinomial) by repeated full forward.
    This keeps multimodal context support without introducing multimodal KV-cache plumbing. -/
partial def generateUncached {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : T #[batch, seq])
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (Sigma (fun n => T #[n, cfg.vision_config.out_hidden_size])) := none)
    (videoFeatures : Option (Sigma (fun n => T #[n, cfg.vision_config.out_hidden_size])) := none)
    : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
  let rec loop
      (remaining : Nat)
      {curSeq : UInt64}
      (cur : T #[batch, curSeq])
      : IO (Sigma (fun outSeq => T #[batch, outSeq])) := do
    if remaining == 0 then
      return ⟨curSeq, cur⟩

    let logits ←
      match imageFeatures, videoFeatures with
      | some ⟨_ni, ifeat⟩, some ⟨_nv, vfeat⟩ =>
        m.forwardWithImageAndVideoFeatures cfg cur ifeat vfeat none
      | some ⟨_ni, ifeat⟩, none =>
        m.forwardWithImageFeatures cfg cur ifeat none
      | none, some ⟨_nv, vfeat⟩ =>
        -- Reuse image path helper for single-modality scatter by token id.
        let embeds0 : T #[batch, curSeq, cfg.text_config.hidden_size] := m.language_model.embedTokens cur
        let embeds ← scatterFeaturesIntoToken
          (inputIds := cur)
          (inputsEmbeds := embeds0)
          cfg.video_token_id
          vfeat
        pure (m.language_model.forwardEmbeds cfg.text_config embeds none)
      | none, none =>
        pure (m.language_model.forward cfg.text_config cur none)

    if curSeq == 0 then
      throw <| IO.userError "generateUncached requires non-empty prompt sequence"

    let lastPos := curSeq - 1
    let last3 : T #[batch, 1, cfg.text_config.vocab_size] :=
      reshape (data.slice logits 1 lastPos 1) #[batch, 1, cfg.text_config.vocab_size]
    let last2 : T #[batch, cfg.text_config.vocab_size] := reshape last3 #[batch, cfg.text_config.vocab_size]
    let nextTok ←
      match strategy with
      | .greedy => pure (reshape (nn.argmax last2 1) #[batch])
      | .multinomial temperature topK topP =>
        let scaled :=
          if temperature == 1.0 then last2 else mul_scalar last2 (1.0 / temperature)
        let filtered := if topK == 0 then scaled else nn.topKFilter scaled topK
        let filtered := if topP >= 1.0 then filtered else nn.topPFilter filtered topP
        let probs := nn.softmax filtered (-1)
        let sampled ← nn.multinomial probs 1 false
        pure (reshape (nn.squeezeDim sampled (-1)) #[batch])

    let nextVals ← data.tensorToUInt64Array nextTok
    let nextCol : T #[batch, 1] := reshape nextTok #[batch, 1]
    let appended : T #[batch, curSeq + 1] := nn.cat cur nextCol 1
    let stop := eosTokenIds.size > 0 && allInSet nextVals eosTokenIds
    if stop then
      pure ⟨curSeq + 1, appended⟩
    else
      loop (remaining - 1) appended

  loop maxNewTokens.toNat inputIds

end Qwen35ForConditionalGeneration

end torch.qwen35
