/-
  Tyr/Model/Qwen35/Multimodal.lean

  Qwen3.5 multimodal path for Tyr:
  - Vision backbone (patch embed + ViT blocks + patch merger)
  - Placeholder-based fusion into text embeddings
  - Conditional generation wrapper over existing Qwen35 text model
-/
import Tyr.Core.Typed.Neural
import Tyr.Core.TensorStruct
import Tyr.Lib.Module.Core
import Tyr.Core.TensorStruct.Derive
import Tyr.Model.Qwen35.Model
import Tyr.Model.Qwen35.VLConfig
import Tyr.Lib.Module.Init

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
  Tensor.add x (Tensor.broadcastTo (Tensor.reshape b #[1, d]) #[n, d])

private def linear2d {tokens in_dim out_dim : Nat}
    (x : Tensor { device := device, shape := #[tokens, in_dim], dtype := dtype })
    (w : Tensor { device := device, shape := #[out_dim, in_dim], dtype := dtype })
    (b : Tensor { device := device, shape := #[out_dim], dtype := dtype })
    : Tensor { device := device, shape := #[tokens, out_dim], dtype := dtype } :=
  addBias2d (Tensor.linear x w) b

private def countMask2d {batch seq : Nat} (mask : Tensor { device := device, shape := #[batch, seq], dtype := .Bool }) : IO Nat := do
  let row : Tensor { device := device, shape := #[batch], dtype := .Int64 } := Tensor.reshape (Tensor.sumDim (mask.toInt64) 1 false) #[batch]
  let counts ← Tensor.toUInt64Array row
  pure (counts.foldl (· + ·.toNat) 0)

/-- Row and column of each patch of an `h × w` grid, in `merge × merge` block order. -/
private def blockOrderRowsCols (h w merge : Nat) : Array (Nat × Nat) :=
  let blocksW := w / merge
  Array.ofFn (n := h * w) fun ⟨i, _⟩ =>
    let blockRow := i / (merge * merge * blocksW)
    let blockCol := (i / (merge * merge)) % blocksW
    (blockRow * merge + (i / merge) % merge, blockCol * merge + i % merge)

private def floatTable (values : Array Float32) (rows cols : Nat) (device : Device) :
    IO (Tensor { device := device, shape := #[rows, cols], dtype := .Float32 }) := do
  pure ((← IO.ofExcept (Tensor.fromFloatArrayAs? (values.map (·.toFloat)) #[rows, cols])).to device)

/-- Axial rotary cos and sin for patches at `rowsCols`: the first half of each row
    rotates by the patch's row, the second by its column. -/
private def visionRotary (cfg : VisionConfig) (rowsCols : Array (Nat × Nat)) (device : Device) :
    IO (Tensor { device := device, shape := #[rowsCols.size, cfg.half_head_dim], dtype := .Float32 } ×
      Tensor { device := device, shape := #[rowsCols.size, cfg.half_head_dim], dtype := .Float32 }) := do
  let quarter := cfg.half_head_dim / 2
  let theta := cfg.rope_theta.toFloat32
  let invFreq := (Array.range quarter).map fun k =>
    1.0 / Float32.pow theta ((2 * k).toFloat32 / cfg.half_head_dim.toFloat32)
  let args := rowsCols.flatMap fun (row, col) =>
    invFreq.map (row.toFloat32 * ·) ++ invFreq.map (col.toFloat32 * ·)
  pure (← floatTable (args.map Float32.cos) rowsCols.size cfg.half_head_dim device,
    ← floatTable (args.map Float32.sin) rowsCols.size cfg.half_head_dim device)

/-- Bilinear taps (`align_corners`) of target `index` on an axis of `size` into a
    `side`-long source axis, with their weights. -/
private def axisTaps (index size side : Nat) : (Nat × Nat) × (Float32 × Float32) :=
  let src := index.toFloat32 * (side - 1).toFloat32 / (max (size - 1) 1).toFloat32
  let lower := src.floor
  let frac := src - lower
  let tap := fun (offset : Nat) => min (lower.toUInt64.toNat + offset) (side - 1)
  ((tap 0, tap 1), (max 0 (1 - frac), max 0 (1 - (1 - frac))))

/-- LayerNorm for vision stack. -/
structure Qwen35VisionLayerNorm (dim : Nat) where
  weight : SomeDTypeTensor #[dim] device
  bias : SomeDTypeTensor #[dim] device
  eps : Float := 1e-6
  deriving TensorStruct

namespace Qwen35VisionLayerNorm

def init (dim : Nat) (eps : Float := 1e-6) (device : Device := .CPU) : IO (Qwen35VisionLayerNorm (device := device) dim) := do
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
  let weight := Tensor.castDType m.weight.2 dtype
  let bias := Tensor.castDType m.bias.2 dtype
  let y3 : Tensor { device := device, shape := #[1, tokens, dim], dtype := dtype } := Tensor.ofRaw! (RawTensor.layerNorm x3.raw weight.raw bias.raw m.eps)
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
  let w1 ← Model.initWeight (dtype := dtype) (device := device) #[cfg.intermediate_size, cfg.hidden_size] cfg.hidden_size
  let b1 ← Model.initBias (dtype := dtype) (device := device) #[cfg.intermediate_size]
  let w2 ← Model.initWeight (dtype := dtype) (device := device) #[cfg.hidden_size, cfg.intermediate_size] cfg.intermediate_size
  let b2 ← Model.initBias (dtype := dtype) (device := device) #[cfg.hidden_size]
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
  linear2d (Tensor.geluTanh h) m.linear_fc2_weight m.linear_fc2_bias

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
  let w ← Model.initWeight (dtype := dtype) (device := device) #[cfg.hidden_size, VisionConfig.patchDim cfg] (VisionConfig.patchDim cfg)
  let b ← Model.initBias (dtype := dtype) (device := device) #[cfg.hidden_size]
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
  let qkvW ← Model.initWeight (dtype := dtype) (device := device) #[cfg.hidden_size * 3, cfg.hidden_size] cfg.hidden_size
  let qkvB ← Model.initBias (dtype := dtype) (device := device) #[cfg.hidden_size * 3]
  let projW ← Model.initWeight (dtype := dtype) (device := device) #[cfg.hidden_size, cfg.hidden_size] cfg.hidden_size
  let projB ← Model.initBias (dtype := dtype) (device := device) #[cfg.hidden_size]
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
    (cos sin : Tensor { device := device, shape := #[tokens, cfg.half_head_dim], dtype := .Float32 })
    : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
  let qkv : Tensor { device := device, shape := #[tokens, cfg.hidden_size * 3], dtype := dtype } := linear2d x m.qkv_weight m.qkv_bias
  let q : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.slice qkv 1 0 cfg.hidden_size
  let k : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.slice qkv 1 cfg.hidden_size cfg.hidden_size
  let v : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.slice qkv 1 (cfg.hidden_size * 2) cfg.hidden_size

  let q : Tensor { device := device, shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg], dtype := dtype } :=
    Tensor.reshape q #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg]
      (by simp [Nat.mul_assoc, VisionConfig.heads_mul_headDim cfg])
  let k : Tensor { device := device, shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg], dtype := dtype } :=
    Tensor.reshape k #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg]
      (by simp [Nat.mul_assoc, VisionConfig.heads_mul_headDim cfg])
  let v : Tensor { device := device, shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg], dtype := dtype } :=
    Tensor.reshape v #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg]
      (by simp [Nat.mul_assoc, VisionConfig.heads_mul_headDim cfg])

  let q : Tensor { device := device, shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg], dtype := dtype } := Tensor.applyRotary q cos sin
  let k : Tensor { device := device, shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg], dtype := dtype } := Tensor.applyRotary k cos sin

  let qh : Tensor { device := device, shape := #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg], dtype := dtype } := Tensor.splitHeads q
  let kh : Tensor { device := device, shape := #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg], dtype := dtype } := Tensor.splitHeads k
  let vh : Tensor { device := device, shape := #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg], dtype := dtype } := Tensor.splitHeads v

  let attn : Tensor { device := device, shape := #[1, cfg.num_heads, tokens, VisionConfig.headDim cfg], dtype := dtype } :=
    Tensor.attentionGQA qh kh vh false true
  let out : Tensor { device := device, shape := #[1, tokens, cfg.num_heads, VisionConfig.headDim cfg], dtype := dtype } := Tensor.joinHeads attn
  let outFlat : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.reshape out #[tokens, cfg.hidden_size] (by simp [Nat.mul_assoc, VisionConfig.heads_mul_headDim cfg])
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
    (cos sin : Tensor { device := device, shape := #[tokens, cfg.half_head_dim], dtype := .Float32 })
    : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
  let h1 := Tensor.add x (m.attn.forward cfg (m.norm1.forward2d x) cos sin)
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
  let fc1W ← Model.initWeight (dtype := dtype) (device := device) #[mergedHidden, mergedHidden] mergedHidden
  let fc1B ← Model.initBias (dtype := dtype) (device := device) #[mergedHidden]
  let fc2W ← Model.initWeight (dtype := dtype) (device := device) #[cfg.out_hidden_size, mergedHidden] mergedHidden
  let fc2B ← Model.initBias (dtype := dtype) (device := device) #[cfg.out_hidden_size]
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
  let ⟨hDiv⟩ ← if hDiv : mergeUnit ∣ tokens then pure (PLift.up hDiv) else
    throw <| IO.userError
      s!"vision token count ({tokens}) must be divisible by merge unit ({mergeUnit})"

  let mergedTokens := VisionConfig.mergedTokenCount cfg tokens
  let xNorm : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := m.norm.forward2d x
  let grouped : Tensor { device := device, shape := #[mergedTokens, cfg.hidden_size * mergeUnit], dtype := dtype } :=
    Tensor.reshape xNorm #[mergedTokens, cfg.hidden_size * mergeUnit]
      (by simpa using (VisionConfig.mergedTokenCount_mul hDiv cfg.hidden_size).symm)
  let h : Tensor { device := device, shape := #[mergedTokens, cfg.hidden_size * mergeUnit], dtype := dtype } :=
    linear2d grouped m.linear_fc1_weight m.linear_fc1_bias
  pure <|
    linear2d (Tensor.gelu h) m.linear_fc2_weight m.linear_fc2_bias

end Qwen35VisionPatchMerger

/-- Qwen3.5 vision backbone (without DeepStack branches). -/
structure Qwen35VisionModel (cfg : VisionConfig) (dtype : DType := .Float32) where
  patch_embed : Qwen35VisionPatchEmbed (dtype := dtype) (device := device) cfg
  pos_embed : Tensor { device := device, shape := #[cfg.num_position_embeddings, cfg.hidden_size], dtype := dtype }
  blocks : Vector (Qwen35VisionBlock (dtype := dtype) (device := device) cfg) cfg.depth
  merger : Qwen35VisionPatchMerger (dtype := dtype) (device := device) cfg
  deriving TensorStruct

namespace Qwen35VisionModel

def init (cfg : VisionConfig) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35VisionModel (dtype := dtype) (device := device) cfg) := do
  let patchEmbed ← Qwen35VisionPatchEmbed.init (dtype := dtype) (device := device) cfg
  let posRaw ← Tensor.randn (device := device) #[cfg.num_position_embeddings, cfg.hidden_size]
  let posEmbed ← Tensor.parameter ((Tensor.mulScalar posRaw 0.02).castDType dtype)

  let blocks ← Vector.ofFnM fun (_ : Fin cfg.depth) =>
    Qwen35VisionBlock.init (dtype := dtype) (device := device) cfg

  let merger ← Qwen35VisionPatchMerger.init (dtype := dtype) (device := device) cfg
  pure {
    patch_embed := patchEmbed
    pos_embed := posEmbed
    blocks := blocks
    merger := merger
  }

/-- The learned position table resampled bilinearly to an `h × w` grid, in block order. -/
private def interpolatedPositions (cfg : VisionConfig)
    (table : Tensor { device := device, shape := #[cfg.num_position_embeddings, cfg.hidden_size], dtype := dtype })
    (h w : Nat) : IO (Tensor { device := device, shape := #[h * w, cfg.hidden_size], dtype := dtype }) := do
  let side := cfg.position_grid_side
  let taps := (blockOrderRowsCols h w cfg.spatial_merge_size).map fun (row, col) =>
    let ((r0, r1), (wr0, wr1)) := axisTaps row h side
    let ((c0, c1), (wc0, wc1)) := axisTaps col w side
    #[(r0 * side + c0, wr0 * wc0), (r0 * side + c1, wr0 * wc1), (r1 * side + c0, wr1 * wc0), (r1 * side + c1, wr1 * wc1)]
  let term (k : Nat) : IO (Tensor { device := device, shape := #[h * w, cfg.hidden_size], dtype := dtype }) := do
    let index : Tensor { device := device, shape := #[h * w], dtype := .Int64 } :=
      (← IO.ofExcept (Tensor.fromInt64ArrayAs? (taps.map fun t => (t[k]!.1 : Nat).toInt64) #[h * w])).to device
    let weight ← floatTable (taps.map fun t => t[k]!.2) (h * w) 1 device
    let weight : Tensor { device := device, shape := #[h * w, cfg.hidden_size], dtype := dtype } :=
      (Tensor.broadcastTo weight #[h * w, cfg.hidden_size]).castDType dtype
    pure (Tensor.mul (Tensor.embedding1d index table) weight)
  let mut out ← term 0
  for k in [1:4] do
    out := Tensor.add out (← term k)
  pure out

/-- Features of the patches of each frame in `frames` (`(h, w)` patch grids, in
    order, `frames` covering all `nPatches`); frames attend only within themselves. -/
def forward {nPatches : Nat}
    (cfg : VisionConfig)
    (m : Qwen35VisionModel (dtype := dtype) (device := device) cfg)
    (patchifiedPixels : Tensor { device := device, shape := #[nPatches, VisionConfig.patchDim cfg], dtype := dtype })
    (frames : Array (Nat × Nat))
    : IO (Tensor { device := device, shape := #[VisionConfig.mergedTokenCount cfg nPatches, cfg.out_hidden_size], dtype := dtype }) := do
  let merge := cfg.spatial_merge_size
  unless frames.foldl (fun n (h, w) => n + h * w) 0 == nPatches do
    throw <| IO.userError s!"frames {frames} do not cover the {nPatches} patches"
  let mut outs : Array (ShapePackedTensor dtype device) := #[]
  let mut offset := 0
  for (h, w) in frames do
    unless h % merge == 0 && w % merge == 0 do
      throw <| IO.userError s!"frame {h}x{w} is not a whole number of {merge}x{merge} merge blocks"
    let start := offset
    let ⟨hFits⟩ ← if hFits : start + h * w ≤ nPatches then pure (PLift.up hFits) else
      throw <| IO.userError s!"frames {frames} do not cover the {nPatches} patches"
    let pixels : Tensor { device := device, shape := #[h * w, VisionConfig.patchDim cfg], dtype := dtype } :=
      Tensor.slice patchifiedPixels 0 start (h * w) (_bounds := by simpa using hFits)
    let (cos, sin) ← visionRotary cfg (blockOrderRowsCols h w merge) device
    let ⟨hRows⟩ ← if hRows : (blockOrderRowsCols h w merge).size = h * w then pure (PLift.up hRows) else
      throw <| IO.userError "vision rotary table size"
    let cos : Tensor { device := device, shape := #[h * w, cfg.half_head_dim], dtype := .Float32 } := hRows ▸ cos
    let sin : Tensor { device := device, shape := #[h * w, cfg.half_head_dim], dtype := .Float32 } := hRows ▸ sin
    let mut x := Tensor.add (m.patch_embed.forward cfg pixels) (← interpolatedPositions cfg m.pos_embed h w)
    for blk in m.blocks do
      x := blk.forward cfg x cos sin
    outs := outs.push (← m.merger.forward cfg x).packShape
    offset := offset + h * w
  IO.ofExcept (Tensor.catMany? outs 0 #[VisionConfig.mergedTokenCount cfg nPatches, cfg.out_hidden_size])

end Qwen35VisionModel

/-- Full Qwen3.5 multimodal wrapper (vision + text causal LM). -/
structure Qwen35ForConditionalGeneration (cfg : VLConfig) (dtype : DType := .Float32)
    (visionDType : DType := .Float32) where
  visual : Qwen35VisionModel (dtype := visionDType) (device := device) cfg.vision_config
  language_model : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg.text_config
  deriving TensorStruct

namespace Qwen35ForConditionalGeneration

def init (cfg : VLConfig) (device : Device := .CPU) (dtype : DType := .Float32) (visionDType : DType := .Float32) : IO (Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg) := do
  let visual ← Qwen35VisionModel.init (dtype := visionDType) (device := device) cfg.vision_config
  let lm ← Qwen35ForCausalLM.init (dtype := dtype) (device := device) cfg.text_config cfg.tie_word_embeddings
  pure { visual := visual, language_model := lm }

/-- Image or video-frame features, with the patch grid `(h, w)` of each image or
    video frame in prompt order. -/
structure VisualFeatures (cfg : VLConfig) (dtype : DType) where
  frames : Array (Nat × Nat)
  tokens : Nat
  features : Tensor { device := device, shape := #[tokens, cfg.vision_config.out_hidden_size], dtype := dtype }

def getImageFeatures {nPatches : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (pixelValues : Tensor { device := device, shape := #[nPatches, VisionConfig.patchDim cfg.vision_config], dtype := visionDType })
    (grids : Array (Nat × Nat))
    : IO (VisualFeatures (device := device) cfg visionDType) := do
  pure ⟨grids, _, ← m.visual.forward cfg.vision_config pixelValues grids⟩

/-- Features of videos whose patches are `pixelValuesVideos`, with each video's
    `(temporal patches, h, w)` grid in `grids`; each temporal patch is its own frame. -/
def getVideoFeatures {nPatches : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (pixelValuesVideos : Tensor { device := device, shape := #[nPatches, VisionConfig.patchDim cfg.vision_config], dtype := visionDType })
    (grids : Array (Nat × Nat × Nat))
    : IO (VisualFeatures (device := device) cfg visionDType) := do
  let frames := grids.flatMap fun (t, h, w) => Array.replicate t (h, w)
  pure ⟨frames, _, ← m.visual.forward cfg.vision_config pixelValuesVideos frames⟩

private def scatterFeaturesIntoToken {batch seq hidden featTokens featDim : Nat} {featureDType : DType}
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, hidden], dtype := dtype })
    (tokenId : UInt32)
    (features : Tensor { device := device, shape := #[featTokens, featDim], dtype := featureDType })
    (_hDim : featDim = hidden)
    : IO (Tensor { device := device, shape := #[batch, seq, hidden], dtype := dtype }) := do

  let tokenMask2d : Tensor { device := device, shape := #[batch, seq], dtype := .Bool } := Tensor.eqScalar inputIds (Int64.ofNat tokenId.toNat)
  let nTokens ← countMask2d tokenMask2d
  if nTokens != featTokens then
    throw <| IO.userError
      s!"placeholder token count mismatch for token={tokenId}: ids={nTokens} features={featTokens}"

  let tokenMask : Tensor { device := device, shape := #[batch, seq, hidden], dtype := .Bool } :=
    Tensor.broadcastTo (Tensor.reshape tokenMask2d #[batch, seq, 1]) #[batch, seq, hidden]
  let flat := Tensor.reshape features #[featTokens * featDim]
  if h : featureDType = dtype then
    pure (Tensor.maskedScatter inputsEmbeds tokenMask (h ▸ flat))
  else
    throw <| IO.userError s!"Qwen35 feature dtype {featureDType} differs from text dtype {dtype}"

def forwardText {batch seq : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.text_config.vocab_size], dtype := dtype } :=
  m.language_model.forward cfg.text_config inputIds attnMask

/-- The reference's `get_rope_index` for one prompt: the `(t, h, w)` position of
    each token and the offset of later positions from the token count. Runs of
    image and video tokens take the next frame of `images` and `videos`. -/
private def ropeIndex (cfg : VLConfig) (ids : Array Nat) (images videos : Array (Nat × Nat)) :
    Except String (Array (Nat × Nat × Nat) × Int) := do
  let merge := cfg.vision_config.spatial_merge_size
  let kind (id : Nat) : Nat :=
    if id == cfg.image_token_id.toNat then 1 else if id == cfg.video_token_id.toNat then 2 else 0
  let mut positions : Array (Nat × Nat × Nat) := Array.mkEmpty ids.size
  let mut current := 0
  let mut nextImage := 0
  let mut nextVideo := 0
  let mut i := 0
  while h : i < ids.size do
    let k := kind ids[i]
    let mut j := i + 1
    while hj : j < ids.size do
      if kind ids[j] != k then break
      j := j + 1
    if k == 0 then
      for n in [:j - i] do
        positions := positions.push (current + n, current + n, current + n)
      current := current + (j - i)
    else
      let some (fh, fw) := (if k == 1 then images[nextImage]? else videos[nextVideo]?)
        | throw s!"more runs of {if k == 1 then "image" else "video"} tokens than frames"
      if k == 1 then nextImage := nextImage + 1 else nextVideo := nextVideo + 1
      let gh := fh / merge
      let gw := fw / merge
      unless j - i == gh * gw do
        throw s!"a run of {j - i} {if k == 1 then "image" else "video"} tokens for a {fh}x{fw} frame of {gh * gw}"
      for a in [:gh] do
        for b in [:gw] do
          positions := positions.push (current, current + a, current + b)
      current := current + max gh gw
    i := j
  unless nextImage == images.size && nextVideo == videos.size do
    throw s!"{images.size} image and {videos.size} video frames for {nextImage} and {nextVideo} runs of tokens"
  let largest := positions.foldl (fun m (t, h, w) => max m (max t (max h w))) 0
  pure (positions, (largest + 1 : Int) - ids.size)

/-- Multimodal RoPE tables for `rows` positions: the prompt's from `positions`, later
    ones at `row + delta` on all three axes; axes per `Config.mrope_section`. -/
private def mropeTables (cfg : Config) (positions : Array (Nat × Nat × Nat)) (delta : Int) (rows : Nat)
    (device : Device) : IO (Qwen35ForCausalLM.RotaryTables (device := device) cfg rows) := do
  let half := Config.rotaryHalfDim cfg
  let theta := cfg.rope_theta.toFloat32
  let sections := cfg.mrope_section
  let invFreq := (Array.range half).map fun j =>
    1.0 / Float32.pow theta ((2 * j).toFloat32 / (2 * half).toFloat32)
  let mut args : Array Float32 := Array.mkEmpty (rows * half)
  for r in [:rows] do
    let (t, h, w) ← match positions[r]? with
      | some p => pure p
      | none =>
        let p := (r : Int) + delta
        if p < 0 then throw <| IO.userError s!"negative multimodal position {p}"
        pure (p.toNat, p.toNat, p.toNat)
    for j in [:half] do
      let position :=
        if j % 3 == 1 && j < 3 * sections.getD 1 0 then h
        else if j % 3 == 2 && j < 3 * sections.getD 2 0 then w
        else t
      args := args.push (position.toFloat32 * invFreq[j]!)
  let cos ← IO.ofExcept (Tensor.fromFloatArrayAs? (args.map (·.cos.toFloat)) #[rows, half])
  let sin ← IO.ofExcept (Tensor.fromFloatArrayAs? (args.map (·.sin.toFloat)) #[rows, half])
  pure (cos.to device, sin.to device)

/-- Rotary tables for `rows` positions of a prompt holding `images` and `videos`;
    `none` (plain RoPE) without them. A prompt with them must be a single row. -/
private def rotaryFor {batch seq : Nat}
    (cfg : VLConfig)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (images videos : Option (VisualFeatures (device := device) cfg visionDType)) (rows : Nat)
    : IO (Option (Qwen35ForCausalLM.RotaryTables (device := device) cfg.text_config rows)) := do
  if images.isNone && videos.isNone then return none
  unless batch == 1 do
    throw <| IO.userError s!"multimodal positions need a single prompt, got a batch of {batch}"
  let ids := (← Tensor.toUInt64Array inputIds).map (·.toNat)
  let (positions, delta) ← IO.ofExcept <|
    ropeIndex cfg ids ((images.map (·.frames)).getD #[]) ((videos.map (·.frames)).getD #[])
  some <$> mropeTables cfg.text_config positions delta rows inputIds.device

private def buildInputEmbedsWithFeatures {batch seq : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (images videos : Option (VisualFeatures (device := device) cfg visionDType))
    : IO (Tensor { device := device, shape := #[batch, seq, cfg.text_config.hidden_size], dtype := dtype }) := do
  let embeds0 : Tensor { device := device, shape := #[batch, seq, cfg.text_config.hidden_size], dtype := dtype } :=
    m.language_model.embedTokens inputIds
  let embeds1 ← match images with
    | some v =>
      let feats := v.features
      scatterFeaturesIntoToken (inputIds := inputIds) (inputsEmbeds := embeds0) cfg.image_token_id feats
        cfg.visionMatchesText
    | none => pure embeds0
  match videos with
  | some v =>
    let feats := v.features
    scatterFeaturesIntoToken (inputIds := inputIds) (inputsEmbeds := embeds1) cfg.video_token_id feats
      cfg.visionMatchesText
  | none => pure embeds1

def forwardWithFeatures {batch seq : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (images videos : Option (VisualFeatures (device := device) cfg visionDType) := none)
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Tensor { device := device, shape := #[batch, seq, cfg.text_config.vocab_size], dtype := dtype }) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let embeds ← buildInputEmbedsWithFeatures cfg m inputIds images videos
  pure (m.language_model.forwardEmbeds cfg.text_config embeds attnMask (← rotaryFor cfg inputIds images videos seq))

def forwardWithImagePatches {batch seq nPatches : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (pixelValues : Tensor { device := device, shape := #[nPatches, VisionConfig.patchDim cfg.vision_config], dtype := visionDType })
    (grids : Array (Nat × Nat))
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : IO (Tensor { device := device, shape := #[batch, seq, cfg.text_config.vocab_size], dtype := dtype }) := do
  m.forwardWithFeatures cfg inputIds (some (← m.getImageFeatures cfg pixelValues grids)) none attnMask

/-- Cached multimodal generation by embedding-feature injection at prefill time. -/
def generate {batch seq : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt32 := #[])
    (images videos : Option (VisualFeatures (device := device) cfg visionDType) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let inputsEmbeds ← buildInputEmbedsWithFeatures cfg m inputIds images videos
  m.language_model.generateFromEmbeds cfg.text_config inputIds inputsEmbeds maxNewTokens strategy eosTokenIds
    (← rotaryFor cfg inputIds images videos (seq + maxNewTokens))

/-- Logits of feeding `forced` through the KV cache after the prompt, with
    image and video features injected at their placeholder tokens. -/
def forcedLogits {batch seq steps : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (forced : Tensor { device := device, shape := #[batch, steps], dtype := .Int64 })
    (images videos : Option (VisualFeatures (device := device) cfg visionDType) := none)
    : IO (Tensor { device := device, shape := #[batch, steps, cfg.text_config.vocab_size], dtype := dtype }) := do
  let inputsEmbeds ← buildInputEmbedsWithFeatures cfg m inputIds images videos
  m.language_model.forcedLogitsFromEmbeds cfg.text_config inputsEmbeds forced
    (← rotaryFor cfg inputIds images videos (seq + steps))

/-- Cached multimodal generation with per-step token callback (streaming). -/
def generateStream {batch seq : Nat}
    (cfg : VLConfig)
    (m : Qwen35ForConditionalGeneration (visionDType := visionDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (onStep : StreamCallback (device := device) batch)
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt32 := #[])
    (images videos : Option (VisualFeatures (device := device) cfg visionDType) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let inputsEmbeds ← buildInputEmbedsWithFeatures cfg m inputIds images videos
  m.language_model.generateFromEmbedsStream cfg.text_config inputIds inputsEmbeds onStep maxNewTokens strategy
    eosTokenIds (← rotaryFor cfg inputIds images videos (seq + maxNewTokens))

end Qwen35ForConditionalGeneration

end torch.qwen35
