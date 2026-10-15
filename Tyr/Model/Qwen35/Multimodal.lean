/-
  Tyr/Model/Qwen35/Multimodal.lean

  Qwen3.5 multimodal path for Tyr:
  - Vision backbone (patch embed + ViT blocks + patch merger)
  - Placeholder-based fusion into text embeddings
  - Conditional generation wrapper over existing Qwen35 text model
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen35.Model
import Tyr.Model.Qwen35.VLConfig
import Tyr.Model.Utils

namespace torch.qwen35

open torch
open torch.Model

private def addBias2d {n d : UInt64}
    (x : Tensor { shape := #[n, d] })
    (b : Tensor { shape := #[d] })
    : Tensor { shape := #[n, d] } :=
  Tensor.Promoting.add x (Tensor.expand (Tensor.reshapeChecked b #[1, d]) #[n, d])

private def linear2d {tokens in_dim out_dim : UInt64}
    (x : Tensor { shape := #[tokens, in_dim] })
    (w : Tensor { shape := #[out_dim, in_dim] })
    (b : Tensor { shape := #[out_dim] })
    : Tensor { shape := #[tokens, out_dim] } :=
  let yDyn : Tensor { shape := #[tokens, out_dim] } := einsumChecked "oh,th->to" w x
  let y : Tensor { shape := #[tokens, out_dim] } := Tensor.reshapeChecked yDyn #[tokens, out_dim]
  addBias2d y b

private def allInSet (xs : Array UInt64) (allow : Array UInt64) : Bool :=
  xs.all (fun x => allow.contains x)

private def countMask2d {batch seq : UInt64} (mask : Tensor { shape := #[batch, seq], dtype := .Bool }) : IO UInt64 := do
  let row : Tensor { shape := #[batch], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.sumDim (mask.toInt64) 1 false) #[batch]
  let counts ← Tensor.toUInt64Array row
  pure (counts.foldl (· + ·) 0)

/-- LayerNorm for vision stack. -/
structure Qwen35VisionLayerNorm (dim : UInt64) where
  weight : Tensor { shape := #[dim] }
  bias : Tensor { shape := #[dim] }
  eps : Float := 1e-6
  deriving TensorStruct

namespace Qwen35VisionLayerNorm

def init (dim : UInt64) (eps : Float := 1e-6) : IO (Qwen35VisionLayerNorm dim) := do
  pure {
    weight := (← Tensor.parameter (onesDynamic #[dim]))
    bias := (← Tensor.parameter (zerosDynamic #[dim]))
    eps := eps
  }

def forward2d {tokens dim : UInt64}
    (m : Qwen35VisionLayerNorm dim)
    (x : Tensor { shape := #[tokens, dim] })
    : Tensor { shape := #[tokens, dim] } :=
  let x3 : Tensor { shape := #[1, tokens, dim] } := Tensor.reshapeChecked x #[1, tokens, dim]
  let y3 : Tensor { shape := #[1, tokens, dim] } := Tensor.layerNorm x3 m.weight m.bias m.eps
  Tensor.reshapeChecked y3 #[tokens, dim]

end Qwen35VisionLayerNorm

/-- Vision MLP block. -/
structure Qwen35VisionMLP (cfg : VisionConfig) where
  linear_fc1_weight : Tensor { shape := #[cfg.intermediate_size, cfg.hidden_size] }
  linear_fc1_bias : Tensor { shape := #[cfg.intermediate_size] }
  linear_fc2_weight : Tensor { shape := #[cfg.hidden_size, cfg.intermediate_size] }
  linear_fc2_bias : Tensor { shape := #[cfg.hidden_size] }
  deriving TensorStruct

namespace Qwen35VisionMLP

def init (cfg : VisionConfig) : IO (Qwen35VisionMLP cfg) := do
  let w1 ← initDynamicWeight #[cfg.intermediate_size, cfg.hidden_size] cfg.hidden_size
  let b1 ← initDynamicBias #[cfg.intermediate_size]
  let w2 ← initDynamicWeight #[cfg.hidden_size, cfg.intermediate_size] cfg.intermediate_size
  let b2 ← initDynamicBias #[cfg.hidden_size]
  pure {
    linear_fc1_weight := w1
    linear_fc1_bias := b1
    linear_fc2_weight := w2
    linear_fc2_bias := b2
  }

def forward {tokens : UInt64}
    (cfg : VisionConfig)
    (m : Qwen35VisionMLP cfg)
    (x : Tensor { shape := #[tokens, cfg.hidden_size] })
    : Tensor { shape := #[tokens, cfg.hidden_size] } :=
  let h : Tensor { shape := #[tokens, cfg.intermediate_size] } :=
    linear2d x m.linear_fc1_weight m.linear_fc1_bias
  linear2d (Tensor.Preserving.gelu h) m.linear_fc2_weight m.linear_fc2_bias

end Qwen35VisionMLP

/-- Vision patch embed implemented as linear projection over flattened patches.
    Input is expected to already be patchified to
    `[num_patches, in_channels * temporal_patch_size * patch_size * patch_size]`. -/
structure Qwen35VisionPatchEmbed (cfg : VisionConfig) where
  weight : Tensor { shape := #[cfg.hidden_size, VisionConfig.patchDim cfg] }
  bias : Tensor { shape := #[cfg.hidden_size] }
  deriving TensorStruct

namespace Qwen35VisionPatchEmbed

def init (cfg : VisionConfig) : IO (Qwen35VisionPatchEmbed cfg) := do
  let w ← initDynamicWeight #[cfg.hidden_size, VisionConfig.patchDim cfg] (VisionConfig.patchDim cfg)
  let b ← initDynamicBias #[cfg.hidden_size]
  pure { weight := w, bias := b }

def forward {nPatches : UInt64}
    (cfg : VisionConfig)
    (m : Qwen35VisionPatchEmbed cfg)
    (x : Tensor { shape := #[nPatches, VisionConfig.patchDim cfg] })
    : Tensor { shape := #[nPatches, cfg.hidden_size] } :=
  linear2d x m.weight m.bias

end Qwen35VisionPatchEmbed

/-- Vision multi-head self-attention. -/
structure Qwen35VisionAttention (cfg : VisionConfig) where
  qkv_weight : Tensor { shape := #[cfg.hidden_size * 3, cfg.hidden_size] }
  qkv_bias : Tensor { shape := #[cfg.hidden_size * 3] }
  proj_weight : Tensor { shape := #[cfg.hidden_size, cfg.hidden_size] }
  proj_bias : Tensor { shape := #[cfg.hidden_size] }
  deriving TensorStruct

namespace Qwen35VisionAttention

def init (cfg : VisionConfig) : IO (Qwen35VisionAttention cfg) := do
  let qkvW ← initDynamicWeight #[cfg.hidden_size * 3, cfg.hidden_size] cfg.hidden_size
  let qkvB ← initDynamicBias #[cfg.hidden_size * 3]
  let projW ← initDynamicWeight #[cfg.hidden_size, cfg.hidden_size] cfg.hidden_size
  let projB ← initDynamicBias #[cfg.hidden_size]
  pure {
    qkv_weight := qkvW
    qkv_bias := qkvB
    proj_weight := projW
    proj_bias := projB
  }

def forward {tokens : UInt64}
    (cfg : VisionConfig)
    (m : Qwen35VisionAttention cfg)
    (x : Tensor { shape := #[tokens, cfg.hidden_size] })
    : Tensor { shape := #[tokens, cfg.hidden_size] } :=
  let qkv : Tensor { shape := #[tokens, cfg.hidden_size * 3] } := linear2d x m.qkv_weight m.qkv_bias
  let q : Tensor { shape := #[tokens, cfg.hidden_size] } := typedSlice qkv 1 0 cfg.hidden_size
  let k : Tensor { shape := #[tokens, cfg.hidden_size] } := typedSlice qkv 1 cfg.hidden_size cfg.hidden_size
  let v : Tensor { shape := #[tokens, cfg.hidden_size] } := typedSlice qkv 1 (cfg.hidden_size * 2) cfg.hidden_size

  let q : Tensor { shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg] } :=
    Tensor.reshapeChecked q #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg]
  let k : Tensor { shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg] } :=
    Tensor.reshapeChecked k #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg]
  let v : Tensor { shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg] } :=
    Tensor.reshapeChecked v #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg]

  -- Qwen3.5-VL uses 2D rotary; this keeps a deterministic 1D rotary signal for now.
  let (cos, sin) := Tensor.rotaryFreqs tokens (VisionConfig.headDim cfg) 10000.0 x.device
  let q : Tensor { shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg] } := Tensor.applyRotary q cos sin
  let k : Tensor { shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg] } := Tensor.applyRotary k cos sin

  let qh : Tensor { shape := #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg] } := Tensor.splitHeads q
  let kh : Tensor { shape := #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg] } := Tensor.splitHeads k
  let vh : Tensor { shape := #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg] } := Tensor.splitHeads v

  let attn : Tensor { shape := #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg] } :=
    Tensor.attentionGQA qh kh vh 0.0 false true
  let out : Tensor { shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg] } := Tensor.joinHeads attn
  let outFlat : Tensor { shape := #[tokens, cfg.hidden_size] } := Tensor.reshapeChecked out #[tokens, cfg.hidden_size]
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
    norm1 := (← Qwen35VisionLayerNorm.init cfg.hidden_size 1e-6)
    norm2 := (← Qwen35VisionLayerNorm.init cfg.hidden_size 1e-6)
    attn := attn
    mlp := mlp
  }

def forward {tokens : UInt64}
    (cfg : VisionConfig)
    (m : Qwen35VisionBlock cfg)
    (x : Tensor { shape := #[tokens, cfg.hidden_size] })
    : Tensor { shape := #[tokens, cfg.hidden_size] } :=
  let h1 := Tensor.Promoting.add x (m.attn.forward cfg (m.norm1.forward2d x))
  Tensor.Promoting.add h1 (m.mlp.forward cfg (m.norm2.forward2d h1))

end Qwen35VisionBlock

/-- Vision patch merger from patch tokens to LLM-width tokens. -/
structure Qwen35VisionPatchMerger (cfg : VisionConfig) where
  norm : Qwen35VisionLayerNorm cfg.hidden_size
  linear_fc1_weight : Tensor { shape := #[cfg.hidden_size * VisionConfig.mergeUnit cfg, cfg.hidden_size * VisionConfig.mergeUnit cfg] }
  linear_fc1_bias : Tensor { shape := #[cfg.hidden_size * VisionConfig.mergeUnit cfg] }
  linear_fc2_weight : Tensor { shape := #[cfg.out_hidden_size, cfg.hidden_size * VisionConfig.mergeUnit cfg] }
  linear_fc2_bias : Tensor { shape := #[cfg.out_hidden_size] }
  deriving TensorStruct

namespace Qwen35VisionPatchMerger

def init (cfg : VisionConfig) : IO (Qwen35VisionPatchMerger cfg) := do
  let mergedHidden := cfg.hidden_size * VisionConfig.mergeUnit cfg
  let fc1W ← initDynamicWeight #[mergedHidden, mergedHidden] mergedHidden
  let fc1B ← initDynamicBias #[mergedHidden]
  let fc2W ← initDynamicWeight #[cfg.out_hidden_size, mergedHidden] mergedHidden
  let fc2B ← initDynamicBias #[cfg.out_hidden_size]
  pure {
    norm := (← Qwen35VisionLayerNorm.init cfg.hidden_size 1e-6)
    linear_fc1_weight := fc1W
    linear_fc1_bias := fc1B
    linear_fc2_weight := fc2W
    linear_fc2_bias := fc2B
  }

def forward {tokens : UInt64}
    (cfg : VisionConfig)
    (m : Qwen35VisionPatchMerger cfg)
    (x : Tensor { shape := #[tokens, cfg.hidden_size] })
    : IO (Tensor { shape := #[VisionConfig.mergedTokenCount cfg tokens, cfg.out_hidden_size] }) := do
  let mergeUnit := VisionConfig.mergeUnit cfg
  if mergeUnit == 0 then
    throw <| IO.userError "vision spatial_merge_size must be > 0"
  if tokens % mergeUnit != 0 then
    throw <| IO.userError
      s!"vision token count ({tokens}) must be divisible by merge unit ({mergeUnit})"

  let mergedTokens := VisionConfig.mergedTokenCount cfg tokens
  let xNorm : Tensor { shape := #[tokens, cfg.hidden_size] } := m.norm.forward2d x
  let grouped : Tensor { shape := #[mergedTokens, cfg.hidden_size * mergeUnit] } :=
    Tensor.reshapeChecked xNorm #[mergedTokens, cfg.hidden_size * mergeUnit]
  let h : Tensor { shape := #[mergedTokens, cfg.hidden_size * mergeUnit] } :=
    linear2d grouped m.linear_fc1_weight m.linear_fc1_bias
  pure <|
    linear2d (Tensor.Preserving.gelu h) m.linear_fc2_weight m.linear_fc2_bias

end Qwen35VisionPatchMerger

/-- Qwen3.5 vision backbone (without DeepStack branches). -/
structure Qwen35VisionModel (cfg : VisionConfig) where
  patch_embed : Qwen35VisionPatchEmbed cfg
  pos_embed : Tensor { shape := #[cfg.num_position_embeddings, cfg.hidden_size] }
  blocks : Array (Qwen35VisionBlock cfg)
  merger : Qwen35VisionPatchMerger cfg
  deriving TensorStruct

namespace Qwen35VisionModel

def init (cfg : VisionConfig) : IO (Qwen35VisionModel cfg) := do
  let patchEmbed ← Qwen35VisionPatchEmbed.init cfg
  let posRaw := (← Tensor.randn #[cfg.num_position_embeddings, cfg.hidden_size]).forgetDType
  let posEmbed ← Tensor.parameter (Tensor.Preserving.mulScalar posRaw 0.02)

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
    (patchifiedPixels : Tensor { shape := #[nPatches, VisionConfig.patchDim cfg] })
    : IO (Tensor { shape := #[VisionConfig.mergedTokenCount cfg nPatches, cfg.out_hidden_size] }) := do
  if nPatches > cfg.num_position_embeddings then
    throw <| IO.userError
      s!"vision patch count ({nPatches}) exceeds num_position_embeddings ({cfg.num_position_embeddings})"

  let x0 : Tensor { shape := #[nPatches, cfg.hidden_size] } := m.patch_embed.forward cfg patchifiedPixels
  let posIds : Tensor { shape := #[nPatches], dtype := .Int64 } := (Tensor.arange nPatches).to patchifiedPixels.device
  let pos : Tensor { shape := #[nPatches, cfg.hidden_size] } := Tensor.embedding1d posIds m.pos_embed
  let mut x : Tensor { shape := #[nPatches, cfg.hidden_size] } := Tensor.Promoting.add x0 pos

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
    (pixelValues : Tensor { shape := #[nPatches, VisionConfig.patchDim cfg.vision_config] })
    : IO (Tensor { shape := #[VisionConfig.mergedTokenCount cfg.vision_config nPatches, cfg.vision_config.out_hidden_size] }) :=
  m.visual.forward cfg.vision_config pixelValues

def getVideoFeatures {nPatches : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (pixelValuesVideos : Tensor { shape := #[nPatches, VisionConfig.patchDim cfg.vision_config] })
    : IO (Tensor { shape := #[VisionConfig.mergedTokenCount cfg.vision_config nPatches, cfg.vision_config.out_hidden_size] }) :=
  -- Qwen3.5 reference uses the same path for video/image features.
  getImageFeatures cfg m pixelValuesVideos

private def scatterFeaturesIntoToken {batch seq hidden featTokens featDim : UInt64}
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, hidden] })
    (tokenId : UInt64)
    (features : Tensor { shape := #[featTokens, featDim] })
    : IO (Tensor { shape := #[batch, seq, hidden] }) := do
  if featDim != hidden then
    throw <| IO.userError
      s!"feature hidden size ({featDim}) does not match text hidden size ({hidden})"

  let tokenMask2d : Tensor { shape := #[batch, seq], dtype := .Bool } := Tensor.eqScalar inputIds (Int64.ofNat tokenId.toNat)
  let nTokens ← countMask2d tokenMask2d
  if nTokens != featTokens then
    throw <| IO.userError
      s!"placeholder token count mismatch for token={tokenId}: ids={nTokens} features={featTokens}"

  let tokenMask : Tensor { shape := #[batch, seq, hidden], dtype := .Bool } :=
    Tensor.expand (Tensor.reshapeChecked tokenMask2d #[batch, seq, 1]) #[batch, seq, hidden]
  let src : Tensor { shape := #[featTokens * featDim] } := Tensor.reshapeChecked features #[featTokens * featDim]
  pure (Tensor.maskedScatter inputsEmbeds tokenMask src)

def forwardText {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64 }) := none)
    : Tensor { shape := #[batch, seq, cfg.text_config.vocab_size] } :=
  m.language_model.forward cfg.text_config inputIds attnMask

def forwardWithImageFeatures {batch seq featTokens : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (imageFeatures : Tensor { shape := #[featTokens, cfg.vision_config.out_hidden_size] })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Tensor { shape := #[batch, seq, cfg.text_config.vocab_size] }) := do
  let embeds0 : Tensor { shape := #[batch, seq, cfg.text_config.hidden_size] } := m.language_model.embedTokens inputIds
  let embeds ← scatterFeaturesIntoToken
    (inputIds := inputIds)
    (inputsEmbeds := embeds0)
    cfg.image_token_id
    imageFeatures
  pure (m.language_model.forwardEmbeds cfg.text_config embeds attnMask)

def forwardWithImageAndVideoFeatures {batch seq imageTokens videoTokens : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (imageFeatures : Tensor { shape := #[imageTokens, cfg.vision_config.out_hidden_size] })
    (videoFeatures : Tensor { shape := #[videoTokens, cfg.vision_config.out_hidden_size] })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Tensor { shape := #[batch, seq, cfg.text_config.vocab_size] }) := do
  let embeds0 : Tensor { shape := #[batch, seq, cfg.text_config.hidden_size] } := m.language_model.embedTokens inputIds
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
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (pixelValues : Tensor { shape := #[nPatches, VisionConfig.patchDim cfg.vision_config] })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Tensor { shape := #[batch, seq, cfg.text_config.vocab_size] }) := do
  let imageFeatures ← m.getImageFeatures cfg pixelValues
  m.forwardWithImageFeatures cfg inputIds imageFeatures attnMask

def forwardWithImageAndVideoPatches {batch seq nImagePatches nVideoPatches : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (pixelValues : Tensor { shape := #[nImagePatches, VisionConfig.patchDim cfg.vision_config] })
    (pixelValuesVideos : Tensor { shape := #[nVideoPatches, VisionConfig.patchDim cfg.vision_config] })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Tensor { shape := #[batch, seq, cfg.text_config.vocab_size] }) := do
  let imageFeatures ← m.getImageFeatures cfg pixelValues
  let videoFeatures ← m.getVideoFeatures cfg pixelValuesVideos
  m.forwardWithImageAndVideoFeatures cfg inputIds imageFeatures videoFeatures attnMask

private def buildInputEmbedsWithFeatures {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (imageFeatures : Option (Sigma (fun n => Tensor { shape := #[n, cfg.vision_config.out_hidden_size] })) := none)
    (videoFeatures : Option (Sigma (fun n => Tensor { shape := #[n, cfg.vision_config.out_hidden_size] })) := none)
    : IO (Tensor { shape := #[batch, seq, cfg.text_config.hidden_size] }) := do
  let embeds0 : Tensor { shape := #[batch, seq, cfg.text_config.hidden_size] } := m.language_model.embedTokens inputIds
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
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (Sigma (fun n => Tensor { shape := #[n, cfg.vision_config.out_hidden_size] })) := none)
    (videoFeatures : Option (Sigma (fun n => Tensor { shape := #[n, cfg.vision_config.out_hidden_size] })) := none)
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
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
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (Sigma (fun n => Tensor { shape := #[n, cfg.vision_config.out_hidden_size] })) := none)
    (videoFeatures : Option (Sigma (fun n => Tensor { shape := #[n, cfg.vision_config.out_hidden_size] })) := none)
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
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
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (Sigma (fun n => Tensor { shape := #[n, cfg.vision_config.out_hidden_size] })) := none)
    (videoFeatures : Option (Sigma (fun n => Tensor { shape := #[n, cfg.vision_config.out_hidden_size] })) := none)
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
  let rec loop
      (remaining : Nat)
      {curSeq : UInt64}
      (cur : Tensor { shape := #[batch, curSeq], dtype := .Int64 })
      : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
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
        let embeds0 : Tensor { shape := #[batch, curSeq, cfg.text_config.hidden_size] } := m.language_model.embedTokens cur
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
    let last3 : Tensor { shape := #[batch, 1, cfg.text_config.vocab_size] } :=
      typedSlice logits 1 lastPos 1
    let last2 : Tensor { shape := #[batch, cfg.text_config.vocab_size] } := Tensor.reshapeChecked last3 #[batch, cfg.text_config.vocab_size]
    let nextTok ← sampleFromLogits last2.toFloat32 strategy

    let nextVals ← Tensor.toUInt64Array nextTok
    let nextCol : Tensor { shape := #[batch, 1], dtype := .Int64 } := Tensor.reshapeChecked nextTok #[batch, 1]
    let appended : Tensor { shape := #[batch, curSeq + 1], dtype := .Int64 } := typedCat cur nextCol 1
    let stop := eosTokenIds.size > 0 && allInSet nextVals eosTokenIds
    if stop then
      pure ⟨curSeq + 1, appended⟩
    else
      loop (remaining - 1) appended

  loop maxNewTokens.toNat inputIds

end Qwen35ForConditionalGeneration

end torch.qwen35
