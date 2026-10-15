/-
  Tyr/Model/Gemma4/Multimodal.lean

  Gemma 4 multimodal path for Tyr:
  - Gemma 4 vision patch embedder + 2D RoPE encoder + spatial pooler
  - Gemma 4 multimodal projection into text hidden size
  - Placeholder-based fusion into prompt embeddings

  This follows the public Hugging Face Gemma 4 reference architecture for the
  image tower and projector. Tyr currently preprocesses images one at a time
  and concatenates their soft tokens, instead of batching padded patch tensors.
-/
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Gemma4.Model
import Tyr.Model.Gemma4.VLConfig

namespace torch.gemma4

variable {device : Device} {dtype visionDType projectorDType inputDType : DType}
variable [DType.ConstantSupported dtype device]

open torch
open torch.Model torch.qwen35

abbrev ImagePatchGrid (cfg : VLConfig) (dtype : DType := .Float32) :=
  Sigma (fun patchRows =>
    Sigma (fun patchCols =>
      Tensor { device := device, shape := #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config], dtype := dtype }))

abbrev ImageFeatures (cfg : VLConfig) (dtype : DType := .Float32) :=
  Sigma (fun n => Tensor { device := device, shape := #[n, cfg.text_config.hidden_size], dtype := dtype })

abbrev PerLayerInputs (cfg : VLConfig) (batch seq : UInt64) (dtype : DType := .Float32) :=
  Option (Tensor { device := device, shape := #[batch, seq, cfg.text_config.num_hidden_layers, cfg.text_config.hidden_size_per_layer_input], dtype := dtype })

private def reqGradFalse {s : Shape} (t : Tensor { device := device, shape := s, dtype := dtype }) : Tensor { device := device, shape := s, dtype := dtype } :=
  Tensor.detach t

private def clampMin {s : Shape} (x : Tensor { device := device, shape := s, dtype := dtype }) (lo : Float) : Tensor { device := device, shape := s, dtype := dtype } :=
  Tensor.Preserving.clampFloat x lo (Float.ofBits 0x7ff0000000000000)

private def clampMax {s : Shape} (x : Tensor { device := device, shape := s, dtype := dtype }) (hi : Float) : Tensor { device := device, shape := s, dtype := dtype } :=
  Tensor.Preserving.clampFloat x (Float.ofBits 0xfff0000000000000) hi

private def clampRange {s : Shape}
    (x : Tensor { device := device, shape := s, dtype := dtype })
    (lo? : Option Float := none)
    (hi? : Option Float := none)
    : Tensor { device := device, shape := s, dtype := dtype } :=
  let x1 :=
    match lo? with
    | some lo => clampMin x lo
    | none => x
  match hi? with
  | some hi => clampMax x1 hi
  | none => x1

private def linear2d {tokens inDim outDim : UInt64}
    (x : Tensor { device := device, shape := #[tokens, inDim], dtype := dtype })
    (w : Tensor { device := device, shape := #[outDim, inDim], dtype := dtype })
    : Tensor { device := device, shape := #[tokens, outDim], dtype := dtype } :=
  let yDyn : Tensor { device := device, shape := #[tokens, outDim], dtype := dtype } := einsumChecked "oh,th->to" w x
  Tensor.reshapeChecked yDyn #[tokens, outDim]

private def countMask2d {batch seq : UInt64} (mask : Tensor { device := device, shape := #[batch, seq], dtype := .Bool }) : IO UInt64 := do
  let row : Tensor { device := device, shape := #[batch], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.Preserving.sumDim (Tensor.toInt64 mask) 1 false) #[batch]
  let counts ← Tensor.toUInt64Array row
  pure (counts.foldl (· + ·) 0)

private def positionIds (patchRows patchCols : UInt64) :
    Tensor { device := .CPU, shape := #[patchRows * patchCols], dtype := .Int64 } × Tensor { device := .CPU, shape := #[patchRows * patchCols], dtype := .Int64 } :=
  Id.run do
    let mut xIds : Array Int64 := #[]
    let mut yIds : Array Int64 := #[]
    for y in [:patchRows.toNat] do
      for x in [:patchCols.toNat] do
        xIds := xIds.push x.toUInt64.toInt64
        yIds := yIds.push y.toUInt64.toInt64
    pure
      ( Tensor.reshapeChecked (Tensor.fromInt64Array xIds) #[patchRows * patchCols]
      , Tensor.reshapeChecked (Tensor.fromInt64Array yIds) #[patchRows * patchCols] )

private def rmsNormNoScale2d {n dim : UInt64}
    (x : Tensor { device := device, shape := #[n, dim], dtype := dtype })
    (eps : Float := 1e-6)
    : Tensor { device := device, shape := #[n, dim], dtype := dtype } :=
  let xf : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := floatDynamic x
  let var : Tensor { device := device, shape := #[n, 1], dtype := .Float32 } := Tensor.Preserving.meanDim (Tensor.Preserving.mul xf xf) 1 true
  let inv : Tensor { device := device, shape := #[n, 1], dtype := .Float32 } := rsqrtDynamic (Tensor.Preserving.addScalar var eps)
  let inv : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.expand inv #[n, dim]
  Tensor.castLike x (Tensor.Preserving.mul xf inv)

private def rmsNormNoScale4d {a b c dim : UInt64}
    (x : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype })
    (eps : Float := 1e-6)
    : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype } :=
  let flat : Tensor { device := device, shape := #[a * b * c, dim], dtype := dtype } := Tensor.reshapeChecked x #[a * b * c, dim]
  Tensor.reshapeChecked (rmsNormNoScale2d flat eps) #[a, b, c, dim]

private def apply2dRotaryEmb {tokens nHead headDim : UInt64}
    (x : Tensor { device := device, shape := #[1, tokens, nHead, headDim], dtype := dtype })
    (patchRows patchCols : UInt64)
    (ropeTheta : Float)
    : Tensor { device := device, shape := #[1, tokens, nHead, headDim], dtype := dtype } :=
  let perDim := headDim / 2
  let xPart : Tensor { device := device, shape := #[1, tokens, nHead, perDim], dtype := dtype } := sliceTyped x 3 0 perDim
  let yPart : Tensor { device := device, shape := #[1, tokens, nHead, perDim], dtype := dtype } := sliceTyped x 3 perDim perDim
  let (xIdsCpu, yIdsCpu) := positionIds patchRows patchCols
  let (xCosTable, xSinTable) := Tensor.rotaryFreqs patchCols perDim ropeTheta x.device
  let (yCosTable, ySinTable) := Tensor.rotaryFreqs patchRows perDim ropeTheta x.device
  let xIds : Tensor { device := device, shape := #[tokens], dtype := .Int64 } := (xIdsCpu.to x.device).reshapeChecked #[tokens]
  let yIds : Tensor { device := device, shape := #[tokens], dtype := .Int64 } := (yIdsCpu.to x.device).reshapeChecked #[tokens]
  let xCos : Tensor { device := device, shape := #[tokens, perDim / 2], dtype := .Float32 } := Tensor.embedding1d xIds xCosTable
  let xSin : Tensor { device := device, shape := #[tokens, perDim / 2], dtype := .Float32 } := Tensor.embedding1d xIds xSinTable
  let yCos : Tensor { device := device, shape := #[tokens, perDim / 2], dtype := .Float32 } := Tensor.embedding1d yIds yCosTable
  let ySin : Tensor { device := device, shape := #[tokens, perDim / 2], dtype := .Float32 } := Tensor.embedding1d yIds ySinTable
  let xRot : Tensor { device := device, shape := #[1, tokens, nHead, perDim], dtype := dtype } := Tensor.applyRotary xPart xCos xSin
  let yRot : Tensor { device := device, shape := #[1, tokens, nHead, perDim], dtype := dtype } := Tensor.applyRotary yPart yCos ySin
  typedCat xRot yRot 3

private def replaceTokenId {batch seq : UInt64}
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (fromId toId : UInt64)
    : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 } :=
  let mask : Tensor { device := device, shape := #[batch, seq], dtype := .Bool } := Tensor.eqScalar inputIds (Int64.ofNat fromId.toNat)
  let repl : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 } := (Tensor.fullInt #[batch, seq] (Int64.ofNat toId.toNat)).to inputIds.device
  Tensor.whereSelect mask repl inputIds

/-- Optional clipped linear used by small Gemma 4 vision towers. -/
structure Gemma4VisionLinear (outDim inDim : UInt64) (dtype : DType := .Float32) where
  weight : Tensor { device := device, shape := #[outDim, inDim], dtype := dtype }
  input_min : Option Float := none
  input_max : Option Float := none
  output_min : Option Float := none
  output_max : Option Float := none
  deriving TensorStruct

namespace Gemma4VisionLinear

def forward2d {tokens outDim inDim : UInt64}
    (m : Gemma4VisionLinear (dtype := dtype) (device := device) outDim inDim)
    (x : Tensor { device := device, shape := #[tokens, inDim], dtype := inputDType })
    : Tensor { device := device, shape := #[tokens, outDim], dtype := dtype } :=
  let x' := clampRange (Tensor.castLike m.weight x) m.input_min m.input_max
  let y := linear2d x' m.weight
  clampRange y m.output_min m.output_max

end Gemma4VisionLinear

structure Gemma4VisionPatchEmbedder (cfg : VLConfig) (dtype : DType := .Float32) where
  input_proj : Gemma4VisionLinear (dtype := dtype) (device := device) cfg.vision_config.hidden_size (VisionConfig.patchDim cfg.vision_config)
  position_embedding_table : SomeDTypeTensor #[2, cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size] device
  deriving TensorStruct

namespace Gemma4VisionPatchEmbedder

def forward {patchRows patchCols : UInt64}
    (cfg : VLConfig)
    (m : Gemma4VisionPatchEmbedder (dtype := dtype) (device := device) cfg)
    (patchGrid : Tensor { device := device, shape := #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config], dtype := inputDType })
    : SomeDTypeTensor #[patchRows * patchCols, cfg.vision_config.hidden_size] device :=
  let tokens := patchRows * patchCols
  let flat : Tensor { device := device, shape := #[tokens, VisionConfig.patchDim cfg.vision_config], dtype := inputDType } :=
    Tensor.reshapeChecked patchGrid #[tokens, VisionConfig.patchDim cfg.vision_config]
  let flat := Tensor.Preserving.mulScalar (Tensor.Preserving.addScalar flat (-0.5)) 2.0
  let hidden : Tensor { device := device, shape := #[tokens, cfg.vision_config.hidden_size], dtype := dtype } := m.input_proj.forward2d flat
  let xTable : Tensor { device := device, shape := #[cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size], dtype := m.position_embedding_table.1.val } :=
    Tensor.reshapeChecked (sliceTyped m.position_embedding_table.2 0 0 1)
      #[cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size]
  let yTable : Tensor { device := device, shape := #[cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size], dtype := m.position_embedding_table.1.val } :=
    Tensor.reshapeChecked (sliceTyped m.position_embedding_table.2 0 1 1)
      #[cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size]
  let (xIdsCpu, yIdsCpu) := positionIds patchRows patchCols
  let xIds : Tensor { device := device, shape := #[tokens], dtype := .Int64 } := xIdsCpu.to hidden.device
  let yIds : Tensor { device := device, shape := #[tokens], dtype := .Int64 } := yIdsCpu.to hidden.device
  let xPos : Tensor { device := device, shape := #[tokens, cfg.vision_config.hidden_size], dtype := m.position_embedding_table.1.val } := Tensor.embedding1d xIds xTable
  let yPos : Tensor { device := device, shape := #[tokens, cfg.vision_config.hidden_size], dtype := m.position_embedding_table.1.val } := Tensor.embedding1d yIds yTable
  promotedAdd (promotedAdd hidden xPos).2 yPos

end Gemma4VisionPatchEmbedder

structure Gemma4VisionMLP (cfg : VisionConfig) (dtype : DType := .Float32) where
  gate_proj : Gemma4VisionLinear (dtype := dtype) (device := device) cfg.intermediate_size cfg.hidden_size
  up_proj : Gemma4VisionLinear (dtype := dtype) (device := device) cfg.intermediate_size cfg.hidden_size
  down_proj : Gemma4VisionLinear (dtype := dtype) (device := device) cfg.hidden_size cfg.intermediate_size
  deriving TensorStruct

namespace Gemma4VisionMLP

def forward {tokens : UInt64}
    (cfg : VisionConfig)
    (m : Gemma4VisionMLP (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := inputDType })
    : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
  let gate : Tensor { device := device, shape := #[tokens, cfg.intermediate_size], dtype := dtype } := m.gate_proj.forward2d x
  let up : Tensor { device := device, shape := #[tokens, cfg.intermediate_size], dtype := dtype } := m.up_proj.forward2d x
  m.down_proj.forward2d (Tensor.Preserving.mul (Tensor.Preserving.gelu gate) up)

end Gemma4VisionMLP

structure Gemma4VisionAttention (cfg : VisionConfig) (dtype : DType := .Float32) where
  q_proj : Gemma4VisionLinear (dtype := dtype) (device := device) cfg.hidden_size cfg.hidden_size
  k_proj : Gemma4VisionLinear (dtype := dtype) (device := device) cfg.hidden_size cfg.hidden_size
  v_proj : Gemma4VisionLinear (dtype := dtype) (device := device) cfg.hidden_size cfg.hidden_size
  o_proj : Gemma4VisionLinear (dtype := dtype) (device := device) cfg.hidden_size cfg.hidden_size
  q_norm : Gemma4RMSNorm (device := device) cfg.head_dim
  k_norm : Gemma4RMSNorm (device := device) cfg.head_dim
  deriving TensorStruct

namespace Gemma4VisionAttention

def forward {patchRows patchCols : UInt64}
    (cfg : VisionConfig)
    (m : Gemma4VisionAttention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[patchRows * patchCols, cfg.hidden_size], dtype := inputDType })
    : Tensor { device := device, shape := #[patchRows * patchCols, cfg.hidden_size], dtype := dtype } :=
  let tokens := patchRows * patchCols
  let q0 : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := m.q_proj.forward2d x
  let k0 : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := m.k_proj.forward2d x
  let v0 : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := m.v_proj.forward2d x

  let q1 : Tensor { device := device, shape := #[1, tokens, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } :=
    Tensor.reshapeChecked q0 #[1, tokens, cfg.num_attention_heads, cfg.head_dim]
  let k1 : Tensor { device := device, shape := #[1, tokens, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
    Tensor.reshapeChecked k0 #[1, tokens, cfg.num_key_value_heads, cfg.head_dim]
  let v1 : Tensor { device := device, shape := #[1, tokens, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
    Tensor.reshapeChecked v0 #[1, tokens, cfg.num_key_value_heads, cfg.head_dim]

  let q2 := m.q_norm.forward4d q1
  let k2 := m.k_norm.forward4d k1
  let v2 := rmsNormNoScale4d v1 cfg.rms_norm_eps

  let q3 : Tensor { device := device, shape := #[1, tokens, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } :=
    apply2dRotaryEmb q2 patchRows patchCols cfg.rope_theta
  let k3 : Tensor { device := device, shape := #[1, tokens, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
    apply2dRotaryEmb k2 patchRows patchCols cfg.rope_theta

  let qh : Tensor { device := device, shape := #[1, cfg.num_attention_heads, tokens, cfg.head_dim], dtype := dtype } :=
    Tensor.Preserving.mulScalar (transposeAttention q3) (Float.sqrt cfg.head_dim.toFloat)
  let kh : Tensor { device := device, shape := #[1, cfg.num_key_value_heads, tokens, cfg.head_dim], dtype := dtype } := transposeAttention k3
  let vh : Tensor { device := device, shape := #[1, cfg.num_key_value_heads, tokens, cfg.head_dim], dtype := dtype } := transposeAttention v2
  let attn : Tensor { device := device, shape := #[1, cfg.num_attention_heads, tokens, cfg.head_dim], dtype := dtype } :=
    Tensor.attentionGQA qh kh vh false true
  let out : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
    Tensor.reshapeChecked (transposeAttention attn) #[tokens, cfg.hidden_size]
  m.o_proj.forward2d out

end Gemma4VisionAttention

structure Gemma4VisionBlock (cfg : VisionConfig) (dtype : DType := .Float32) where
  input_layernorm : Gemma4RMSNorm (device := device) cfg.hidden_size
  post_attention_layernorm : Gemma4RMSNorm (device := device) cfg.hidden_size
  pre_feedforward_layernorm : Gemma4RMSNorm (device := device) cfg.hidden_size
  post_feedforward_layernorm : Gemma4RMSNorm (device := device) cfg.hidden_size
  self_attn : Gemma4VisionAttention (dtype := dtype) (device := device) cfg
  mlp : Gemma4VisionMLP (dtype := dtype) (device := device) cfg
  deriving TensorStruct

namespace Gemma4VisionBlock

def forward {patchRows patchCols : UInt64}
    (cfg : VisionConfig)
    (m : Gemma4VisionBlock (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[patchRows * patchCols, cfg.hidden_size], dtype := inputDType })
    : SomeDTypeTensor #[patchRows * patchCols, cfg.hidden_size] device :=
  let residual1 := x
  let h1 := m.input_layernorm.forward2d x
  let attn := m.self_attn.forward (patchRows := patchRows) (patchCols := patchCols) cfg h1
  let h2 := promotedAdd residual1 (m.post_attention_layernorm.forward2d attn)
  let residual2 := h2.2
  let h3 := m.pre_feedforward_layernorm.forward2d h2.2
  let mlp := m.mlp.forward cfg h3
  promotedAdd residual2 (m.post_feedforward_layernorm.forward2d mlp)

end Gemma4VisionBlock

structure Gemma4VisionModel (cfg : VLConfig) (dtype : DType := .Float32) where
  patch_embedder : Gemma4VisionPatchEmbedder (dtype := dtype) (device := device) cfg
  blocks : Array (Gemma4VisionBlock (dtype := dtype) (device := device) cfg.vision_config)
  std_bias : Option (SomeDTypeTensor #[cfg.vision_config.hidden_size] device) := none
  std_scale : Option (SomeDTypeTensor #[cfg.vision_config.hidden_size] device) := none
  deriving TensorStruct

namespace Gemma4VisionModel

private def poolFeatures {patchRows patchCols : UInt64}
    (cfg : VLConfig)
    (x : Tensor { device := device, shape := #[patchRows * patchCols, cfg.vision_config.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[(patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size),
        cfg.vision_config.hidden_size], dtype := dtype } :=
  let pool := cfg.vision_config.pooling_kernel_size
  let pooledRows := patchRows / pool
  let pooledCols := patchCols / pool
  let xGrid : Tensor { device := device, shape := #[1, patchRows, patchCols, cfg.vision_config.hidden_size], dtype := dtype } :=
    Tensor.reshapeChecked x #[1, patchRows, patchCols, cfg.vision_config.hidden_size]
  let xNchw : Tensor { device := device, shape := #[1, cfg.vision_config.hidden_size, patchRows, patchCols], dtype := dtype } := (Tensor.permute xGrid #[0, 3, 1, 2]).reshapeChecked #[1, cfg.vision_config.hidden_size, patchRows, patchCols]
  let pooled : Tensor { device := device, shape := #[1, cfg.vision_config.hidden_size, pooledRows, pooledCols], dtype := dtype } :=
    avgPool2d xNchw pool
  let pooledNhwc : Tensor { device := device, shape := #[1, pooledRows, pooledCols, cfg.vision_config.hidden_size], dtype := dtype } :=
    (Tensor.permute pooled #[0, 2, 3, 1]).reshapeChecked #[1, pooledRows, pooledCols, cfg.vision_config.hidden_size]
  Tensor.Preserving.mulScalar
    (Tensor.reshapeChecked pooledNhwc #[pooledRows * pooledCols, cfg.vision_config.hidden_size])
    (Float.sqrt cfg.vision_config.hidden_size.toFloat)

def forward {patchRows patchCols : UInt64}
    (cfg : VLConfig)
    (m : Gemma4VisionModel (dtype := dtype) (device := device) cfg)
    (patchGrid : Tensor { device := device, shape := #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config], dtype := inputDType })
    : SomeDTypeTensor #[(patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size),
        cfg.vision_config.hidden_size] device :=
  Id.run do
    let mut hidden := m.patch_embedder.forward cfg patchGrid
    for i in [:m.blocks.size] do
      match m.blocks[i]? with
      | some blk =>
        hidden := blk.forward (patchRows := patchRows) (patchCols := patchCols) cfg.vision_config hidden.2
      | none =>
        pure ()
    let pooled := poolFeatures cfg hidden.2
    match m.std_bias, m.std_scale with
    | some bias, some scale =>
      let nTok := (patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size)
      let bias : Tensor { device := device, shape := #[nTok, cfg.vision_config.hidden_size], dtype := bias.1.val } :=
        Tensor.expand (Tensor.reshapeChecked bias.2 #[1, cfg.vision_config.hidden_size]) #[nTok, cfg.vision_config.hidden_size]
      let scale : Tensor { device := device, shape := #[nTok, cfg.vision_config.hidden_size], dtype := scale.1.val } :=
        Tensor.expand (Tensor.reshapeChecked scale.2 #[1, cfg.vision_config.hidden_size]) #[nTok, cfg.vision_config.hidden_size]
      promotedMul (promotedSub pooled bias).2 scale
    | _, _ =>
      packExactTensor pooled

end Gemma4VisionModel

structure Gemma4MultimodalEmbedder (cfg : VLConfig) (dtype : DType := .Float32) where
  embedding_projection : Tensor { device := device, shape := #[cfg.text_config.hidden_size, cfg.vision_config.hidden_size], dtype := dtype }
  deriving TensorStruct

namespace Gemma4MultimodalEmbedder

def forward {tokens : UInt64}
    (cfg : VLConfig)
    (m : Gemma4MultimodalEmbedder (dtype := dtype) (device := device) cfg)
    (inputsEmbeds : Tensor { device := device, shape := #[tokens, cfg.vision_config.hidden_size], dtype := inputDType })
    : Tensor { device := device, shape := #[tokens, cfg.text_config.hidden_size], dtype := dtype } :=
  let normalized := rmsNormNoScale2d inputsEmbeds cfg.text_config.rms_norm_eps
  let admitted : Tensor ⟨#[tokens, cfg.vision_config.hidden_size], dtype, device⟩ :=
    match Tensor.ofRaw normalized.raw with
    | .ok tensor => tensor
    | .error message => nativeInvariantFailure s!"Gemma4 projector dtype: {message}"
  linear2d admitted m.embedding_projection

end Gemma4MultimodalEmbedder

structure Gemma4ForConditionalGeneration (cfg : VLConfig) (dtype : DType := .Float32)
    (visionDType : DType := .Float32) (projectorDType : DType := .Float32) where
  vision_tower : Gemma4VisionModel (dtype := visionDType) (device := device) cfg
  embed_vision : Gemma4MultimodalEmbedder (dtype := projectorDType) (device := device) cfg
  language_model : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg.text_config
  deriving TensorStruct

namespace Gemma4ForConditionalGeneration

def getImageFeatures {patchRows patchCols : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration (visionDType := visionDType) (projectorDType := projectorDType) (dtype := dtype) (device := device) cfg)
    (patchGrid : Tensor { device := device, shape := #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config], dtype := inputDType })
    : Tensor { device := device, shape := #[(patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size),
        cfg.text_config.hidden_size], dtype := projectorDType } :=
  let visionFeatures := m.vision_tower.forward cfg patchGrid
  m.embed_vision.forward cfg visionFeatures.2

def getImageFeaturesMany
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration (visionDType := visionDType) (projectorDType := projectorDType) (dtype := dtype) (device := device) cfg)
    (images : Array (ImagePatchGrid (dtype := inputDType) (device := device) cfg))
    : IO (Option (ImageFeatures (dtype := projectorDType) (device := device) cfg)) := do
  let mut acc? : Option (ImageFeatures (dtype := projectorDType) (device := device) cfg) := none
  for img in images do
    match img with
    | ⟨patchRows, ⟨patchCols, patchGrid⟩⟩ =>
      let nTok := (patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size)
      let feats : Tensor { device := device, shape := #[nTok, cfg.text_config.hidden_size], dtype := projectorDType } := getImageFeatures cfg m patchGrid
      acc? :=
        match acc? with
        | none =>
          some ⟨nTok, feats⟩
        | some ⟨nPrev, prev⟩ =>
          some ⟨nPrev + nTok, typedCat prev feats 0⟩
  pure acc?

def repeatFeaturesForBatch {batch tokens : UInt64}
    (cfg : VLConfig)
    (features : Tensor { device := device, shape := #[tokens, cfg.text_config.hidden_size], dtype := projectorDType })
    : ImageFeatures (dtype := projectorDType) (device := device) cfg :=
  Id.run do
    if batch <= 1 then
      return ⟨tokens, features⟩
    let repeated := Tensor.expand (features.reshapeChecked #[1, tokens, cfg.text_config.hidden_size])
      #[batch, tokens, cfg.text_config.hidden_size]
    return ⟨batch * tokens, repeated.reshapeChecked #[batch * tokens, cfg.text_config.hidden_size]⟩

private def scatterFeaturesIntoToken {batch seq hidden featTokens featDim : UInt64}
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, hidden], dtype := dtype })
    (tokenId : UInt64)
    (features : Tensor { device := device, shape := #[featTokens, featDim], dtype := projectorDType })
    : IO (Tensor { device := device, shape := #[batch, seq, hidden], dtype := dtype }) := do
  if featDim != hidden then
    throw <| IO.userError
      s!"Gemma4 feature hidden size ({featDim}) does not match text hidden size ({hidden})"
  let tokenMask2d : Tensor { device := device, shape := #[batch, seq], dtype := .Bool } := Tensor.eqScalar inputIds (Int64.ofNat tokenId.toNat)
  let nTokens ← countMask2d tokenMask2d
  if nTokens != featTokens then
    throw <| IO.userError
      s!"Gemma4 placeholder token count mismatch for token={tokenId}: ids={nTokens} features={featTokens}"
  let tokenMask : Tensor { device := device, shape := #[batch, seq, hidden], dtype := .Bool } :=
    Tensor.expand (Tensor.reshapeChecked tokenMask2d #[batch, seq, 1]) #[batch, seq, hidden]
  let src ← IO.ofExcept (Tensor.ofRaw (σ := ⟨#[featTokens * featDim], dtype, device⟩)
    (Tensor.reshapeChecked features #[featTokens * featDim]).raw)
  pure (Tensor.maskedScatter inputsEmbeds tokenMask src)

private def buildInputEmbedsWithFeatures {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration (visionDType := visionDType) (projectorDType := projectorDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (imageFeatures : Option (ImageFeatures (dtype := projectorDType) (device := device) cfg) := none)
    : IO (Tensor { device := device, shape := #[batch, seq, cfg.text_config.hidden_size], dtype := dtype } × PerLayerInputs (dtype := dtype) (device := device) cfg batch seq) := do
  let padId := cfg.text_config.pad_token_id.getD 0
  let llmInputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 } := replaceTokenId inputIds cfg.image_token_id padId
  let llmInputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.text_config.hidden_size], dtype := dtype } :=
    m.language_model.embedTokens cfg.text_config llmInputIds
  let perLayerInputs :=
    Gemma4Model.computePerLayerInputs cfg.text_config m.language_model.model llmInputIds llmInputsEmbeds
  match imageFeatures with
  | some ⟨_n, feats⟩ =>
    let embeds ←
      scatterFeaturesIntoToken
        (inputIds := inputIds)
        (inputsEmbeds := llmInputsEmbeds)
        cfg.image_token_id
        feats
    pure (embeds, perLayerInputs)
  | none =>
    pure (llmInputsEmbeds, perLayerInputs)

private def buildVisionBidirectionalPrefillMasks {batch seq : UInt64}
    (cfg : VLConfig)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : IO (Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 } × Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 }) := do
  let padId := cfg.text_config.pad_token_id.getD 0
  let slidingWindow := cfg.text_config.sliding_window.toNat
  let mut slidingFlat : Array Int64 := #[]
  let mut fullFlat : Array Int64 := #[]

  for b in [:batch.toNat] do
    let row2 : Tensor { device := device, shape := #[1, seq], dtype := .Int64 } := sliceTyped inputIds 0 b.toUInt64 1
    let row1 : Tensor { device := device, shape := #[seq], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.toInt64 row2) #[seq]
    let ids ← Tensor.toUInt64Array row1
    let mut groupIds : Array Int64 := Array.replicate ids.size (-1)
    let mut nextGroup : Int64 := 0

    for i in [:ids.size] do
      let tok := ids[i]!
      if tok == cfg.image_token_id then
        let prevIsImage :=
          i > 0 && ids[i - 1]! == cfg.image_token_id
        if prevIsImage then
          groupIds := groupIds.set! i (nextGroup - 1)
        else
          groupIds := groupIds.set! i nextGroup
          nextGroup := nextGroup + 1

    for q in [:ids.size] do
      let qTok := ids[q]!
      let qValid := qTok != padId
      let qGroup := groupIds[q]!
      for k in [:ids.size] do
        let kTok := ids[k]!
        let kValid := kTok != padId
        let baseFull := qValid && kValid && k <= q
        let baseSliding :=
          baseFull &&
          (slidingWindow == 0 || (q - k) < slidingWindow)
        let sameVisionGroup := qGroup >= 0 && qGroup == groupIds[k]!
        fullFlat := fullFlat.push (if baseFull || sameVisionGroup then 1 else 0)
        slidingFlat := slidingFlat.push (if baseSliding || sameVisionGroup then 1 else 0)

  let slidingMask : Tensor { device := .CPU, shape := #[batch, seq, seq], dtype := .Int64 } :=
    Tensor.reshapeChecked (Tensor.fromInt64Array slidingFlat) #[batch, seq, seq]
  let fullMask : Tensor { device := .CPU, shape := #[batch, seq, seq], dtype := .Int64 } :=
    Tensor.reshapeChecked (Tensor.fromInt64Array fullFlat) #[batch, seq, seq]
  pure (slidingMask.to inputIds.device, fullMask.to inputIds.device)

private def bidirectionalVisionPrefillMasks? {batch seq : UInt64}
    (cfg : VLConfig)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : IO (Option (Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 } × Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })) := do
  if cfg.text_config.use_bidirectional_attention == "vision" then
    some <$> buildVisionBidirectionalPrefillMasks cfg inputIds
  else
    pure none

def forwardText {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration (visionDType := visionDType) (projectorDType := projectorDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq, cfg.text_config.vocab_size], dtype := dtype } :=
  m.language_model.forward cfg.text_config inputIds

def forwardWithImageFeatures {batch seq featTokens : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration (visionDType := visionDType) (projectorDType := projectorDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (imageFeatures : Tensor { device := device, shape := #[featTokens, cfg.text_config.hidden_size], dtype := projectorDType })
    : IO (Tensor { device := device, shape := #[batch, seq, cfg.text_config.vocab_size], dtype := dtype }) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let (inputsEmbeds, perLayerInputs) ←
    buildInputEmbedsWithFeatures cfg m inputIds (some ⟨featTokens, imageFeatures⟩)
  match (← bidirectionalVisionPrefillMasks? cfg inputIds) with
  | some (slidingMask, fullMask) =>
    pure <|
      m.language_model.forwardEmbedsWithPerLayerInputsBidirectionalVision
        cfg.text_config
        inputIds
        inputsEmbeds
        perLayerInputs
        slidingMask
        fullMask
  | none =>
    pure <|
      m.language_model.forwardEmbedsWithPerLayerInputs
        cfg.text_config
        inputIds
        inputsEmbeds
        perLayerInputs

def generate {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration (visionDType := visionDType) (projectorDType := projectorDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (ImageFeatures (dtype := projectorDType) (device := device) cfg) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let (inputsEmbeds, perLayerInputs) ← buildInputEmbedsWithFeatures cfg m inputIds imageFeatures
  let prefillMasks ← bidirectionalVisionPrefillMasks? cfg inputIds
  m.language_model.generateFromEmbedsWithPerLayerInputs
    cfg.text_config
    inputIds
    inputsEmbeds
    perLayerInputs
    maxNewTokens
    strategy
    eosTokenIds
    prefillMasks

def generateStream {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration (visionDType := visionDType) (projectorDType := projectorDType) (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (onStep : StreamCallback (device := device) batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (ImageFeatures (dtype := projectorDType) (device := device) cfg) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let (inputsEmbeds, perLayerInputs) ← buildInputEmbedsWithFeatures cfg m inputIds imageFeatures
  let prefillMasks ← bidirectionalVisionPrefillMasks? cfg inputIds
  m.language_model.generateFromEmbedsStreamWithPerLayerInputs
    cfg.text_config
    inputIds
    inputsEmbeds
    perLayerInputs
    onStep
    maxNewTokens
    strategy
    eosTokenIds
    prefillMasks

end Gemma4ForConditionalGeneration

end torch.gemma4
