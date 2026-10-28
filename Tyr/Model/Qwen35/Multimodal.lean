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

variable {device : Device} {dtype visionDType : DType}
variable [DType.RealFloating dtype] [DType.RealFloating visionDType]
variable [DType.ConstantSupported dtype device]

open torch
open torch.Model

private def addBias2d {n d : Nat}
    (x : Tensor { device := device, shape := #[n, d], dtype := dtype })
    (b : Tensor { device := device, shape := #[d], dtype := dtype })
    : Tensor { device := device, shape := #[n, d], dtype := dtype } :=
  Tensor.add x (Tensor.expand (Tensor.reshape b #[1, d]) #[n, d])

private def linear2d {tokens in_dim out_dim : Nat}
    (x : Tensor { device := device, shape := #[tokens, in_dim], dtype := dtype })
    (w : Tensor { device := device, shape := #[out_dim, in_dim], dtype := dtype })
    (b : Tensor { device := device, shape := #[out_dim], dtype := dtype })
    : Tensor { device := device, shape := #[tokens, out_dim], dtype := dtype } :=
  let yDyn : Tensor { device := device, shape := #[tokens, out_dim], dtype := dtype } := einsumChecked "oh,th->to" w x
  let y : Tensor { device := device, shape := #[tokens, out_dim], dtype := dtype } := Tensor.reshape yDyn #[tokens, out_dim]
  addBias2d y b

private def allInSet (xs : Array UInt64) (allow : Array UInt64) : Bool :=
  xs.all (fun x => allow.contains x)

private def countMask2d {batch seq : Nat} (mask : Tensor { device := device, shape := #[batch, seq], dtype := .Bool }) : IO Nat := do
  let row : Tensor { device := device, shape := #[batch], dtype := .Int64 } := Tensor.reshape (Tensor.sumDim (mask.toInt64) 1 false) #[batch]
  let counts ← Tensor.toUInt64Array row
  pure (counts.foldl (· + ·.toNat) 0)

/-- LayerNorm for vision stack. -/
structure Qwen35VisionLayerNorm (dim : Nat) where
  weight : SomeDTypeTensor #[dim] device
  bias : SomeDTypeTensor #[dim] device
  eps : Float := 1e-6
  deriving TensorStruct

namespace Qwen35VisionLayerNorm

def init (dim : Nat) (eps : Float := 1e-6) (device : Device := .CPU) : IO (Qwen35VisionLayerNorm (device := device) dim) := do
  IO.ofExcept (Shape.validateAllocation #[dim])
  pure {
    weight := ⟨.Float32, (← Tensor.parameter (Tensor.ones #[dim] (device := device)))⟩
    bias := ⟨.Float32, (← Tensor.parameter (Tensor.zeros #[dim] (device := device)))⟩
    eps := eps
  }

def forward2d {tokens dim : Nat}
    (m : Qwen35VisionLayerNorm (device := device) dim)
    (x : Tensor { device := device, shape := #[tokens, dim], dtype := dtype })
    : Tensor { device := device, shape := #[tokens, dim], dtype := dtype } :=
  let x3 : Tensor { device := device, shape := #[1, tokens, dim], dtype := dtype } := Tensor.reshape x #[1, tokens, dim]
  let y3 : Tensor { device := device, shape := #[1, tokens, dim], dtype := dtype } := x3.checkedReplacement (RawTensor.layerNorm x3.raw m.weight.2.raw m.bias.2.raw m.eps)
  Tensor.reshape y3 #[tokens, dim]

end Qwen35VisionLayerNorm

/-- Vision MLP block. -/
structure Qwen35VisionMLP (cfg : VisionConfig) (dtype : DType := .Float32) where
  linear_fc1_weight : Tensor { device := device, shape := #[cfg.intermediate_size, cfg.hidden_size], dtype := dtype }
  linear_fc1_bias : Tensor { device := device, shape := #[cfg.intermediate_size], dtype := dtype }
  linear_fc2_weight : Tensor { device := device, shape := #[cfg.hidden_size, cfg.intermediate_size], dtype := dtype }
  linear_fc2_bias : Tensor { device := device, shape := #[cfg.hidden_size], dtype := dtype }
  deriving TensorStruct

namespace Qwen35VisionMLP

def init (cfg : VisionConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35VisionMLP (dtype := dtype) (device := device) cfg) := do
  let w1 ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.intermediate_size, cfg.hidden_size] cfg.hidden_size
  let b1 ← initDynamicBias (dtype := dtype) (device := device) #[cfg.intermediate_size]
  let w2 ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.hidden_size, cfg.intermediate_size] cfg.intermediate_size
  let b2 ← initDynamicBias (dtype := dtype) (device := device) #[cfg.hidden_size]
  pure {
    linear_fc1_weight := w1
    linear_fc1_bias := b1
    linear_fc2_weight := w2
    linear_fc2_bias := b2
  }

def forward {tokens : Nat}
    (cfg : VisionConfig)
    (m : Qwen35VisionMLP (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
  let h : Tensor { device := device, shape := #[tokens, cfg.intermediate_size], dtype := dtype } :=
    linear2d x m.linear_fc1_weight m.linear_fc1_bias
  linear2d (Tensor.gelu h) m.linear_fc2_weight m.linear_fc2_bias

end Qwen35VisionMLP

/-- Vision patch embed implemented as linear projection over flattened patches.
    Input is expected to already be patchified to
    `[num_patches, in_channels * temporal_patch_size * patch_size * patch_size]`. -/
structure Qwen35VisionPatchEmbed (cfg : VisionConfig) (dtype : DType := .Float32) where
  weight : Tensor { device := device, shape := #[cfg.hidden_size, VisionConfig.patchDim cfg], dtype := dtype }
  bias : Tensor { device := device, shape := #[cfg.hidden_size], dtype := dtype }
  deriving TensorStruct

namespace Qwen35VisionPatchEmbed

def init (cfg : VisionConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35VisionPatchEmbed (dtype := dtype) (device := device) cfg) := do
  let w ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.hidden_size, VisionConfig.patchDim cfg] (VisionConfig.patchDim cfg)
  let b ← initDynamicBias (dtype := dtype) (device := device) #[cfg.hidden_size]
  pure { weight := w, bias := b }

def forward {nPatches : Nat}
    (cfg : VisionConfig)
    (m : Qwen35VisionPatchEmbed (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[nPatches, VisionConfig.patchDim cfg], dtype := dtype })
    : Tensor { device := device, shape := #[nPatches, cfg.hidden_size], dtype := dtype } :=
  linear2d x m.weight m.bias

end Qwen35VisionPatchEmbed

/-- Vision multi-head self-attention. -/
structure Qwen35VisionAttention (cfg : VisionConfig) (dtype : DType := .Float32) where
  qkv_weight : Tensor { device := device, shape := #[cfg.hidden_size * 3, cfg.hidden_size], dtype := dtype }
  qkv_bias : Tensor { device := device, shape := #[cfg.hidden_size * 3], dtype := dtype }
  proj_weight : Tensor { device := device, shape := #[cfg.hidden_size, cfg.hidden_size], dtype := dtype }
  proj_bias : Tensor { device := device, shape := #[cfg.hidden_size], dtype := dtype }
  deriving TensorStruct

namespace Qwen35VisionAttention

def init (cfg : VisionConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35VisionAttention (dtype := dtype) (device := device) cfg) := do
  let qkvW ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.hidden_size * 3, cfg.hidden_size] cfg.hidden_size
  let qkvB ← initDynamicBias (dtype := dtype) (device := device) #[cfg.hidden_size * 3]
  let projW ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.hidden_size, cfg.hidden_size] cfg.hidden_size
  let projB ← initDynamicBias (dtype := dtype) (device := device) #[cfg.hidden_size]
  pure {
    qkv_weight := qkvW
    qkv_bias := qkvB
    proj_weight := projW
    proj_bias := projB
  }

def forward {tokens : Nat}
    (cfg : VisionConfig)
    (m : Qwen35VisionAttention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
  let qkv : Tensor { device := device, shape := #[tokens, cfg.hidden_size * 3], dtype := dtype } := linear2d x m.qkv_weight m.qkv_bias
  let q : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := typedSlice qkv 1 0 cfg.hidden_size
  let k : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := typedSlice qkv 1 cfg.hidden_size cfg.hidden_size
  let v : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := typedSlice qkv 1 (cfg.hidden_size * 2) cfg.hidden_size

  let q : Tensor { device := device, shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked q #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg]
  let k : Tensor { device := device, shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked k #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg]
  let v : Tensor { device := device, shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked v #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg]

  -- Qwen3.5-VL uses 2D rotary; this keeps a deterministic 1D rotary signal for now.
  let (cos, sin) := Tensor.rotaryFreqs tokens (VisionConfig.headDim cfg) 10000.0 x.device
  let q : Tensor { device := device, shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg], dtype := dtype } := Tensor.applyRotary q cos sin
  let k : Tensor { device := device, shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg], dtype := dtype } := Tensor.applyRotary k cos sin

  let qh : Tensor { device := device, shape := #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg], dtype := dtype } := Tensor.splitHeads q
  let kh : Tensor { device := device, shape := #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg], dtype := dtype } := Tensor.splitHeads k
  let vh : Tensor { device := device, shape := #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg], dtype := dtype } := Tensor.splitHeads v

  let attn : Tensor { device := device, shape := #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg], dtype := dtype } :=
    Tensor.attentionGQA qh kh vh false true
  let out : Tensor { device := device, shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg], dtype := dtype } := Tensor.joinHeads attn
  let outFlat : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.reshapeChecked out #[tokens, cfg.hidden_size]
  linear2d outFlat m.proj_weight m.proj_bias

end Qwen35VisionAttention

/-- One vision transformer block. -/
structure Qwen35VisionBlock (cfg : VisionConfig) (dtype : DType := .Float32) where
  norm1 : Qwen35VisionLayerNorm (device := device) cfg.hidden_size
  norm2 : Qwen35VisionLayerNorm (device := device) cfg.hidden_size
  attn : Qwen35VisionAttention (dtype := dtype) (device := device) cfg
  mlp : Qwen35VisionMLP (dtype := dtype) (device := device) cfg
  deriving TensorStruct

namespace Qwen35VisionBlock

def init (cfg : VisionConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35VisionBlock (dtype := dtype) (device := device) cfg) := do
  let attn ← Qwen35VisionAttention.init (dtype := dtype) (device := device) cfg
  let mlp ← Qwen35VisionMLP.init (dtype := dtype) (device := device) cfg
  pure {
    norm1 := (← Qwen35VisionLayerNorm.init (device := device) cfg.hidden_size 1e-6)
    norm2 := (← Qwen35VisionLayerNorm.init (device := device) cfg.hidden_size 1e-6)
    attn := attn
    mlp := mlp
  }

def forward {tokens : Nat}
    (cfg : VisionConfig)
    (m : Qwen35VisionBlock (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
  let h1 := Tensor.add x (m.attn.forward cfg (m.norm1.forward2d x))
  Tensor.add h1 (m.mlp.forward cfg (m.norm2.forward2d h1))

end Qwen35VisionBlock

/-- Vision patch merger from patch tokens to LLM-width tokens. -/
structure Qwen35VisionPatchMerger (cfg : VisionConfig) (dtype : DType := .Float32) where
  norm : Qwen35VisionLayerNorm (device := device) cfg.hidden_size
  linear_fc1_weight : Tensor { device := device, shape := #[cfg.hidden_size * VisionConfig.mergeUnit cfg, cfg.hidden_size * VisionConfig.mergeUnit cfg], dtype := dtype }
  linear_fc1_bias : Tensor { device := device, shape := #[cfg.hidden_size * VisionConfig.mergeUnit cfg], dtype := dtype }
  linear_fc2_weight : Tensor { device := device, shape := #[cfg.out_hidden_size, cfg.hidden_size * VisionConfig.mergeUnit cfg], dtype := dtype }
  linear_fc2_bias : Tensor { device := device, shape := #[cfg.out_hidden_size], dtype := dtype }
  deriving TensorStruct

namespace Qwen35VisionPatchMerger

def init (cfg : VisionConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35VisionPatchMerger (dtype := dtype) (device := device) cfg) := do
  let mergedHidden := cfg.hidden_size * VisionConfig.mergeUnit cfg
  let fc1W ← initDynamicWeight (dtype := dtype) (device := device) #[mergedHidden, mergedHidden] mergedHidden
  let fc1B ← initDynamicBias (dtype := dtype) (device := device) #[mergedHidden]
  let fc2W ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.out_hidden_size, mergedHidden] mergedHidden
  let fc2B ← initDynamicBias (dtype := dtype) (device := device) #[cfg.out_hidden_size]
  pure {
    norm := (← Qwen35VisionLayerNorm.init (device := device) cfg.hidden_size 1e-6)
    linear_fc1_weight := fc1W
    linear_fc1_bias := fc1B
    linear_fc2_weight := fc2W
    linear_fc2_bias := fc2B
  }

def forward {tokens : Nat}
    (cfg : VisionConfig)
    (m : Qwen35VisionPatchMerger (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype })
    : IO (Tensor { device := device, shape := #[VisionConfig.mergedTokenCount cfg tokens, cfg.out_hidden_size], dtype := dtype }) := do
  let mergeUnit := VisionConfig.mergeUnit cfg
  if mergeUnit == 0 then
    throw <| IO.userError "vision spatial_merge_size must be > 0"
  if tokens % mergeUnit != 0 then
    throw <| IO.userError
      s!"vision token count ({tokens}) must be divisible by merge unit ({mergeUnit})"

  let mergedTokens := VisionConfig.mergedTokenCount cfg tokens
  let xNorm : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := m.norm.forward2d x
  let grouped : Tensor { device := device, shape := #[mergedTokens, cfg.hidden_size * mergeUnit], dtype := dtype } :=
    Tensor.reshapeChecked xNorm #[mergedTokens, cfg.hidden_size * mergeUnit]
  let h : Tensor { device := device, shape := #[mergedTokens, cfg.hidden_size * mergeUnit], dtype := dtype } :=
    linear2d grouped m.linear_fc1_weight m.linear_fc1_bias
  pure <|
    linear2d (Tensor.gelu h) m.linear_fc2_weight m.linear_fc2_bias

end Qwen35VisionPatchMerger

/-- Qwen3.5 vision backbone (without DeepStack branches). -/
structure Qwen35VisionModel (cfg : VisionConfig) (dtype : DType := .Float32) where
  patch_embed : Qwen35VisionPatchEmbed (dtype := dtype) (device := device) cfg
  pos_embed : Tensor { device := device, shape := #[cfg.num_position_embeddings, cfg.hidden_size], dtype := dtype }
  blocks : Array (Qwen35VisionBlock (dtype := dtype) (device := device) cfg)
  merger : Qwen35VisionPatchMerger (dtype := dtype) (device := device) cfg
  deriving TensorStruct

namespace Qwen35VisionModel

def init (cfg : VisionConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35VisionModel (dtype := dtype) (device := device) cfg) := do
  let patchEmbed ← Qwen35VisionPatchEmbed.init (dtype := dtype) (device := device) cfg
  let posRaw ← Tensor.randn (device := device) #[cfg.num_position_embeddings, cfg.hidden_size]
  let posEmbed ← Tensor.parameter (← IO.ofExcept ((Tensor.mulScalar posRaw 0.02).castDType? dtype))

  let mut blocks : Array (Qwen35VisionBlock (dtype := dtype) (device := device) cfg) := #[]
  for _ in [:cfg.depth] do
    blocks := blocks.push (← Qwen35VisionBlock.init (dtype := dtype) (device := device) cfg)

  let merger ← Qwen35VisionPatchMerger.init (dtype := dtype) (device := device) cfg
  pure {
    patch_embed := patchEmbed
    pos_embed := posEmbed
    blocks := blocks
    merger := merger
  }

def forward {nPatches : Nat}
    (cfg : VisionConfig)
    (m : Qwen35VisionModel (dtype := dtype) (device := device) cfg)
    (patchifiedPixels : Tensor { device := device, shape := #[nPatches, VisionConfig.patchDim cfg], dtype := dtype })
    : IO (Tensor { device := device, shape := #[VisionConfig.mergedTokenCount cfg nPatches, cfg.out_hidden_size], dtype := dtype }) := do
  if nPatches > cfg.num_position_embeddings then
    throw <| IO.userError
      s!"vision patch count ({nPatches}) exceeds num_position_embeddings ({cfg.num_position_embeddings})"

  let x0 : Tensor { device := device, shape := #[nPatches, cfg.hidden_size], dtype := dtype } := m.patch_embed.forward cfg patchifiedPixels
  let posIds : Tensor { device := device, shape := #[nPatches], dtype := .Int64 } := (Tensor.arange nPatches).to patchifiedPixels.device
  let pos : Tensor { device := device, shape := #[nPatches, cfg.hidden_size], dtype := dtype } := Tensor.embedding1d posIds m.pos_embed
  let mut x : Tensor { device := device, shape := #[nPatches, cfg.hidden_size], dtype := dtype } := Tensor.add x0 pos

  for i in [:m.blocks.size] do
    match m.blocks[i]? with
    | some blk =>
      x := blk.forward cfg x
    | none =>
      throw <| IO.userError s!"missing Qwen35 vision block at index {i}"

  m.merger.forward cfg x

end Qwen35VisionModel

/-- Full Qwen3.5 multimodal wrapper (vision + text causal LM). -/
structure Qwen35ForConditionalGeneration (cfg : VLConfig) (dtype : DType := .Float32)
    (visionDType : DType := .Float32) where
  visual : Qwen35VisionModel (dtype := visionDType) (device := device) cfg.vision_config
  language_model : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg.text_config
  deriving TensorStruct

namespace Qwen35ForConditionalGeneration

def init (cfg : VLConfig) (device : Device := .CPU) (dtype : DType := .Float32) (visionDType : DType := .Float32) : IO (Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg) := do
  if cfg.vision_config.out_hidden_size != cfg.text_config.hidden_size then
    throw <| IO.userError
      s!"vision out_hidden_size ({cfg.vision_config.out_hidden_size}) must match text hidden_size ({cfg.text_config.hidden_size})"
  let visual ← Qwen35VisionModel.init (dtype := visionDType) (device := device) cfg.vision_config
  let lm ← Qwen35ForCausalLM.init (dtype := dtype) (device := device) cfg.text_config cfg.tie_word_embeddings
  pure { visual := visual, language_model := lm }

def getImageFeatures {nPatches : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (pixelValues : Tensor { device := device, shape := #[nPatches, VisionConfig.patchDim cfg.vision_config], dtype := visionDType })
    : IO (Tensor { device := device, shape := #[VisionConfig.mergedTokenCount cfg.vision_config nPatches, cfg.vision_config.out_hidden_size], dtype := visionDType }) :=
  m.visual.forward cfg.vision_config pixelValues

def getVideoFeatures {nPatches : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (pixelValuesVideos : Tensor { device := device, shape := #[nPatches, VisionConfig.patchDim cfg.vision_config], dtype := visionDType })
    : IO (Tensor { device := device, shape := #[VisionConfig.mergedTokenCount cfg.vision_config nPatches, cfg.vision_config.out_hidden_size], dtype := visionDType }) :=
  -- Qwen3.5 reference uses the same path for video/image features.
  getImageFeatures cfg m pixelValuesVideos

private def scatterFeaturesIntoToken {batch seq hidden featTokens featDim : Nat} {featureDType : DType}
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, hidden], dtype := dtype })
    (tokenId : UInt64)
    (features : Tensor { device := device, shape := #[featTokens, featDim], dtype := featureDType })
    : IO (Tensor { device := device, shape := #[batch, seq, hidden], dtype := dtype }) := do
  if featDim != hidden then
    throw <| IO.userError
      s!"feature hidden size ({featDim}) does not match text hidden size ({hidden})"

  let tokenMask2d : Tensor { device := device, shape := #[batch, seq], dtype := .Bool } := Tensor.eqScalar inputIds (Int64.ofNat tokenId.toNat)
  let nTokens ← countMask2d tokenMask2d
  if nTokens != featTokens then
    throw <| IO.userError
      s!"placeholder token count mismatch for token={tokenId}: ids={nTokens} features={featTokens}"

  let tokenMask : Tensor { device := device, shape := #[batch, seq, hidden], dtype := .Bool } :=
    Tensor.expand (Tensor.reshape tokenMask2d #[batch, seq, 1]) #[batch, seq, hidden]
  let src ← IO.ofExcept (Tensor.ofRaw (σ := ⟨#[featTokens * featDim], dtype, device⟩)
    (Tensor.reshape features #[featTokens * featDim]).raw)
  pure (Tensor.maskedScatter inputsEmbeds tokenMask src)

def forwardText {batch seq : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.text_config.vocab_size], dtype := dtype } :=
  m.language_model.forward cfg.text_config inputIds attnMask

def forwardWithImageFeatures {batch seq featTokens : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (imageFeatures : Tensor { device := device, shape := #[featTokens, cfg.vision_config.out_hidden_size], dtype := visionDType })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Tensor { device := device, shape := #[batch, seq, cfg.text_config.vocab_size], dtype := dtype }) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let embeds0 : Tensor { device := device, shape := #[batch, seq, cfg.text_config.hidden_size], dtype := dtype } := m.language_model.embedTokens inputIds
  let embeds ← scatterFeaturesIntoToken
    (inputIds := inputIds)
    (inputsEmbeds := embeds0)
    cfg.image_token_id
    imageFeatures
  pure (m.language_model.forwardEmbeds cfg.text_config embeds attnMask)

def forwardWithImageAndVideoFeatures {batch seq imageTokens videoTokens : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (imageFeatures : Tensor { device := device, shape := #[imageTokens, cfg.vision_config.out_hidden_size], dtype := visionDType })
    (videoFeatures : Tensor { device := device, shape := #[videoTokens, cfg.vision_config.out_hidden_size], dtype := visionDType })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Tensor { device := device, shape := #[batch, seq, cfg.text_config.vocab_size], dtype := dtype }) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let embeds0 : Tensor { device := device, shape := #[batch, seq, cfg.text_config.hidden_size], dtype := dtype } := m.language_model.embedTokens inputIds
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

def forwardWithImagePatches {batch seq nPatches : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (pixelValues : Tensor { device := device, shape := #[nPatches, VisionConfig.patchDim cfg.vision_config], dtype := visionDType })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Tensor { device := device, shape := #[batch, seq, cfg.text_config.vocab_size], dtype := dtype }) := do
  let imageFeatures ← m.getImageFeatures cfg pixelValues
  m.forwardWithImageFeatures cfg inputIds imageFeatures attnMask

def forwardWithImageAndVideoPatches {batch seq nImagePatches nVideoPatches : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (pixelValues : Tensor { device := device, shape := #[nImagePatches, VisionConfig.patchDim cfg.vision_config], dtype := visionDType })
    (pixelValuesVideos : Tensor { device := device, shape := #[nVideoPatches, VisionConfig.patchDim cfg.vision_config], dtype := visionDType })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Tensor { device := device, shape := #[batch, seq, cfg.text_config.vocab_size], dtype := dtype }) := do
  let imageFeatures ← m.getImageFeatures cfg pixelValues
  let videoFeatures ← m.getVideoFeatures cfg pixelValuesVideos
  m.forwardWithImageAndVideoFeatures cfg inputIds imageFeatures videoFeatures attnMask

private def buildInputEmbedsWithFeatures {batch seq : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (imageFeatures : Option (Sigma (fun n => Tensor { device := device, shape := #[n, cfg.vision_config.out_hidden_size], dtype := visionDType })) := none)
    (videoFeatures : Option (Sigma (fun n => Tensor { device := device, shape := #[n, cfg.vision_config.out_hidden_size], dtype := visionDType })) := none)
    : IO (Tensor { device := device, shape := #[batch, seq, cfg.text_config.hidden_size], dtype := dtype }) := do
  let embeds0 : Tensor { device := device, shape := #[batch, seq, cfg.text_config.hidden_size], dtype := dtype } := m.language_model.embedTokens inputIds
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
def generate {batch seq : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (Sigma (fun n => Tensor { device := device, shape := #[n, cfg.vision_config.out_hidden_size], dtype := visionDType })) := none)
    (videoFeatures : Option (Sigma (fun n => Tensor { device := device, shape := #[n, cfg.vision_config.out_hidden_size], dtype := visionDType })) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let inputsEmbeds ← buildInputEmbedsWithFeatures cfg m inputIds imageFeatures videoFeatures
  m.language_model.generateFromEmbeds
    cfg.text_config
    inputIds
    inputsEmbeds
    maxNewTokens
    strategy
    eosTokenIds

/-- Cached multimodal generation with per-step token callback (streaming). -/
def generateStream {batch seq : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (onStep : StreamCallback (device := device) batch)
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (Sigma (fun n => Tensor { device := device, shape := #[n, cfg.vision_config.out_hidden_size], dtype := visionDType })) := none)
    (videoFeatures : Option (Sigma (fun n => Tensor { device := device, shape := #[n, cfg.vision_config.out_hidden_size], dtype := visionDType })) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
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
partial def generateUncached {batch seq : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (Sigma (fun n => Tensor { device := device, shape := #[n, cfg.vision_config.out_hidden_size], dtype := visionDType })) := none)
    (videoFeatures : Option (Sigma (fun n => Tensor { device := device, shape := #[n, cfg.vision_config.out_hidden_size], dtype := visionDType })) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let rec loop
      (remaining : Nat)
      {curSeq : Nat}
      (cur : Tensor { device := device, shape := #[batch, curSeq], dtype := .Int64 })
      : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
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
        let embeds0 : Tensor { device := device, shape := #[batch, curSeq, cfg.text_config.hidden_size], dtype := dtype } := m.language_model.embedTokens cur
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
    let last3 : Tensor { device := device, shape := #[batch, 1, cfg.text_config.vocab_size], dtype := dtype } :=
      typedSlice logits 1 lastPos 1
    let last2 : Tensor { device := device, shape := #[batch, cfg.text_config.vocab_size], dtype := dtype } := Tensor.reshape last3 #[batch, cfg.text_config.vocab_size]
    let nextTok ← sampleFromLogits last2.toFloat32 strategy

    let nextVals ← Tensor.toUInt64Array nextTok
    let nextCol : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 } := Tensor.reshape nextTok #[batch, 1]
    let appended : Tensor { device := device, shape := #[batch, curSeq + 1], dtype := .Int64 } := typedCat cur nextCol 1
    let stop := eosTokenIds.size > 0 && allInSet nextVals eosTokenIds
    if stop then
      pure ⟨curSeq + 1, appended⟩
    else
      loop (remaining - 1) appended

  loop maxNewTokens inputIds

end Qwen35ForConditionalGeneration

end torch.qwen35
