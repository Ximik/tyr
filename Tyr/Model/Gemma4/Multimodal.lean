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

open torch
open torch.Model torch.qwen35

abbrev ImagePatchGrid (cfg : VLConfig) :=
  Sigma (fun patchRows =>
    Sigma (fun patchCols =>
      Tensor { shape := #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config] }))

abbrev ImageFeatures (cfg : VLConfig) :=
  Sigma (fun n => Tensor { shape := #[n, cfg.text_config.hidden_size] })

abbrev PerLayerInputs (cfg : VLConfig) (batch seq : UInt64) :=
  Option (Tensor { shape := #[batch, seq, cfg.text_config.num_hidden_layers, cfg.text_config.hidden_size_per_layer_input] })

private def reqGradFalse {s : Shape} (t : Tensor { shape := s }) : Tensor { shape := s } :=
  Tensor.detach t

private def clampMin {s : Shape} (x : Tensor { shape := s }) (lo : Float) : Tensor { shape := s } :=
  Tensor.Preserving.clampFloat x lo (Float.ofBits 0x7ff0000000000000)

private def clampMax {s : Shape} (x : Tensor { shape := s }) (hi : Float) : Tensor { shape := s } :=
  Tensor.Preserving.clampFloat x (Float.ofBits 0xfff0000000000000) hi

private def clampRange {s : Shape}
    (x : Tensor { shape := s })
    (lo? : Option Float := none)
    (hi? : Option Float := none)
    : Tensor { shape := s } :=
  let x1 :=
    match lo? with
    | some lo => clampMin x lo
    | none => x
  match hi? with
  | some hi => clampMax x1 hi
  | none => x1

private def linear2d {tokens inDim outDim : UInt64}
    (x : Tensor { shape := #[tokens, inDim] })
    (w : Tensor { shape := #[outDim, inDim] })
    : Tensor { shape := #[tokens, outDim] } :=
  let yDyn : Tensor { shape := #[tokens, outDim] } := einsumChecked "oh,th->to" w x
  Tensor.reshapeChecked yDyn #[tokens, outDim]

private def countMask2d {batch seq : UInt64} (mask : Tensor { shape := #[batch, seq], dtype := .Bool }) : IO UInt64 := do
  let row : Tensor { shape := #[batch], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.Preserving.sumDim (Tensor.toInt64 mask) 1 false) #[batch]
  let counts ← Tensor.toUInt64Array row
  pure (counts.foldl (· + ·) 0)

private def positionIds (patchRows patchCols : UInt64) :
    Tensor { shape := #[patchRows * patchCols], dtype := .Int64 } × Tensor { shape := #[patchRows * patchCols], dtype := .Int64 } :=
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
    (x : Tensor { shape := #[n, dim] })
    (eps : Float := 1e-6)
    : Tensor { shape := #[n, dim] } :=
  let xf : Tensor { shape := #[n, dim] } := floatDynamic x
  let var : Tensor { shape := #[n, 1] } := Tensor.Preserving.meanDim (xf * xf) 1 true
  let inv : Tensor { shape := #[n, 1] } := rsqrtDynamic (var + eps)
  let inv : Tensor { shape := #[n, dim] } := Tensor.expand inv #[n, dim]
  Tensor.castLike x (xf * inv)

private def rmsNormNoScale4d {a b c dim : UInt64}
    (x : Tensor { shape := #[a, b, c, dim] })
    (eps : Float := 1e-6)
    : Tensor { shape := #[a, b, c, dim] } :=
  let flat : Tensor { shape := #[a * b * c, dim] } := Tensor.reshapeChecked x #[a * b * c, dim]
  Tensor.reshapeChecked (rmsNormNoScale2d flat eps) #[a, b, c, dim]

private def apply2dRotaryEmb {tokens nHead headDim : UInt64}
    (x : Tensor { shape := #[1, tokens, nHead, headDim] })
    (patchRows patchCols : UInt64)
    (ropeTheta : Float)
    : Tensor { shape := #[1, tokens, nHead, headDim] } :=
  let perDim := headDim / 2
  let xPart : Tensor { shape := #[1, tokens, nHead, perDim] } := sliceTyped x 3 0 perDim
  let yPart : Tensor { shape := #[1, tokens, nHead, perDim] } := sliceTyped x 3 perDim perDim
  let (xIdsCpu, yIdsCpu) := positionIds patchRows patchCols
  let (xCosTable, xSinTable) := Tensor.rotaryFreqs patchCols perDim ropeTheta x.device
  let (yCosTable, ySinTable) := Tensor.rotaryFreqs patchRows perDim ropeTheta x.device
  let xIds : Tensor { shape := #[tokens], dtype := .Int64 } := (xIdsCpu.to x.device).reshapeChecked #[tokens]
  let yIds : Tensor { shape := #[tokens], dtype := .Int64 } := (yIdsCpu.to x.device).reshapeChecked #[tokens]
  let xCos : Tensor { shape := #[tokens, perDim / 2], dtype := .Float32 } := Tensor.embedding1d xIds xCosTable
  let xSin : Tensor { shape := #[tokens, perDim / 2], dtype := .Float32 } := Tensor.embedding1d xIds xSinTable
  let yCos : Tensor { shape := #[tokens, perDim / 2], dtype := .Float32 } := Tensor.embedding1d yIds yCosTable
  let ySin : Tensor { shape := #[tokens, perDim / 2], dtype := .Float32 } := Tensor.embedding1d yIds ySinTable
  let xRot : Tensor { shape := #[1, tokens, nHead, perDim] } := Tensor.applyRotary xPart xCos xSin
  let yRot : Tensor { shape := #[1, tokens, nHead, perDim] } := Tensor.applyRotary yPart yCos ySin
  typedCat xRot yRot 3

private def replaceTokenId {batch seq : UInt64}
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (fromId toId : UInt64)
    : Tensor { shape := #[batch, seq], dtype := .Int64 } :=
  let mask : Tensor { shape := #[batch, seq], dtype := .Bool } := Tensor.eqScalar inputIds (Int64.ofNat fromId.toNat)
  let repl : Tensor { shape := #[batch, seq], dtype := .Int64 } := (Tensor.fullInt #[batch, seq] (Int64.ofNat toId.toNat)).to inputIds.device
  Tensor.whereSelect mask repl inputIds

/-- Optional clipped linear used by small Gemma 4 vision towers. -/
structure Gemma4VisionLinear (outDim inDim : UInt64) where
  weight : Tensor { shape := #[outDim, inDim] }
  input_min : Option Float := none
  input_max : Option Float := none
  output_min : Option Float := none
  output_max : Option Float := none
  deriving TensorStruct

namespace Gemma4VisionLinear

def forward2d {tokens outDim inDim : UInt64}
    (m : Gemma4VisionLinear outDim inDim)
    (x : Tensor { shape := #[tokens, inDim] })
    : Tensor { shape := #[tokens, outDim] } :=
  let x' := clampRange (Tensor.castLike m.weight x) m.input_min m.input_max
  let y := linear2d x' m.weight
  clampRange y m.output_min m.output_max

end Gemma4VisionLinear

structure Gemma4VisionPatchEmbedder (cfg : VLConfig) where
  input_proj : Gemma4VisionLinear cfg.vision_config.hidden_size (VisionConfig.patchDim cfg.vision_config)
  position_embedding_table : Tensor { shape := #[2, cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size] }
  deriving TensorStruct

namespace Gemma4VisionPatchEmbedder

def forward {patchRows patchCols : UInt64}
    (cfg : VLConfig)
    (m : Gemma4VisionPatchEmbedder cfg)
    (patchGrid : Tensor { shape := #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config] })
    : Tensor { shape := #[patchRows * patchCols, cfg.vision_config.hidden_size] } :=
  let tokens := patchRows * patchCols
  let flat : Tensor { shape := #[tokens, VisionConfig.patchDim cfg.vision_config] } :=
    Tensor.reshapeChecked patchGrid #[tokens, VisionConfig.patchDim cfg.vision_config]
  let flat := Tensor.Preserving.mulScalar (Tensor.Preserving.addScalar flat (-0.5)) 2.0
  let hidden : Tensor { shape := #[tokens, cfg.vision_config.hidden_size] } := m.input_proj.forward2d flat
  let xTable : Tensor { shape := #[cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size] } :=
    Tensor.reshapeChecked (sliceTyped m.position_embedding_table 0 0 1)
      #[cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size]
  let yTable : Tensor { shape := #[cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size] } :=
    Tensor.reshapeChecked (sliceTyped m.position_embedding_table 0 1 1)
      #[cfg.vision_config.position_embedding_size, cfg.vision_config.hidden_size]
  let (xIdsCpu, yIdsCpu) := positionIds patchRows patchCols
  let xIds : Tensor { shape := #[tokens], dtype := .Int64 } := xIdsCpu.to hidden.device
  let yIds : Tensor { shape := #[tokens], dtype := .Int64 } := yIdsCpu.to hidden.device
  let xPos : Tensor { shape := #[tokens, cfg.vision_config.hidden_size] } := Tensor.embedding1d xIds xTable
  let yPos : Tensor { shape := #[tokens, cfg.vision_config.hidden_size] } := Tensor.embedding1d yIds yTable
  hidden + xPos + yPos

end Gemma4VisionPatchEmbedder

structure Gemma4VisionMLP (cfg : VisionConfig) where
  gate_proj : Gemma4VisionLinear cfg.intermediate_size cfg.hidden_size
  up_proj : Gemma4VisionLinear cfg.intermediate_size cfg.hidden_size
  down_proj : Gemma4VisionLinear cfg.hidden_size cfg.intermediate_size
  deriving TensorStruct

namespace Gemma4VisionMLP

def forward {tokens : UInt64}
    (cfg : VisionConfig)
    (m : Gemma4VisionMLP cfg)
    (x : Tensor { shape := #[tokens, cfg.hidden_size] })
    : Tensor { shape := #[tokens, cfg.hidden_size] } :=
  let gate : Tensor { shape := #[tokens, cfg.intermediate_size] } := m.gate_proj.forward2d x
  let up : Tensor { shape := #[tokens, cfg.intermediate_size] } := m.up_proj.forward2d x
  m.down_proj.forward2d (Tensor.Preserving.gelu gate * up)

end Gemma4VisionMLP

structure Gemma4VisionAttention (cfg : VisionConfig) where
  q_proj : Gemma4VisionLinear cfg.hidden_size cfg.hidden_size
  k_proj : Gemma4VisionLinear cfg.hidden_size cfg.hidden_size
  v_proj : Gemma4VisionLinear cfg.hidden_size cfg.hidden_size
  o_proj : Gemma4VisionLinear cfg.hidden_size cfg.hidden_size
  q_norm : Gemma4RMSNorm cfg.head_dim
  k_norm : Gemma4RMSNorm cfg.head_dim
  deriving TensorStruct

namespace Gemma4VisionAttention

def forward {patchRows patchCols : UInt64}
    (cfg : VisionConfig)
    (m : Gemma4VisionAttention cfg)
    (x : Tensor { shape := #[patchRows * patchCols, cfg.hidden_size] })
    : Tensor { shape := #[patchRows * patchCols, cfg.hidden_size] } :=
  let tokens := patchRows * patchCols
  let q0 : Tensor { shape := #[tokens, cfg.hidden_size] } := m.q_proj.forward2d x
  let k0 : Tensor { shape := #[tokens, cfg.hidden_size] } := m.k_proj.forward2d x
  let v0 : Tensor { shape := #[tokens, cfg.hidden_size] } := m.v_proj.forward2d x

  let q1 : Tensor { shape := #[1, tokens, cfg.num_attention_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked q0 #[1, tokens, cfg.num_attention_heads, cfg.head_dim]
  let k1 : Tensor { shape := #[1, tokens, cfg.num_key_value_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked k0 #[1, tokens, cfg.num_key_value_heads, cfg.head_dim]
  let v1 : Tensor { shape := #[1, tokens, cfg.num_key_value_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked v0 #[1, tokens, cfg.num_key_value_heads, cfg.head_dim]

  let q2 := m.q_norm.forward4d q1
  let k2 := m.k_norm.forward4d k1
  let v2 := rmsNormNoScale4d v1 cfg.rms_norm_eps

  let q3 : Tensor { shape := #[1, tokens, cfg.num_attention_heads, cfg.head_dim] } :=
    apply2dRotaryEmb q2 patchRows patchCols cfg.rope_theta
  let k3 : Tensor { shape := #[1, tokens, cfg.num_key_value_heads, cfg.head_dim] } :=
    apply2dRotaryEmb k2 patchRows patchCols cfg.rope_theta

  let qh : Tensor { shape := #[1, cfg.num_attention_heads, tokens, cfg.head_dim] } :=
    Tensor.Preserving.mulScalar (transposeAttention q3) (Float.sqrt cfg.head_dim.toFloat)
  let kh : Tensor { shape := #[1, cfg.num_key_value_heads, tokens, cfg.head_dim] } := transposeAttention k3
  let vh : Tensor { shape := #[1, cfg.num_key_value_heads, tokens, cfg.head_dim] } := transposeAttention v2
  let attn : Tensor { shape := #[1, cfg.num_attention_heads, tokens, cfg.head_dim] } :=
    Tensor.attentionGQA qh kh vh 0.0 false true
  let out : Tensor { shape := #[tokens, cfg.hidden_size] } :=
    Tensor.reshapeChecked (transposeAttention attn) #[tokens, cfg.hidden_size]
  m.o_proj.forward2d out

end Gemma4VisionAttention

structure Gemma4VisionBlock (cfg : VisionConfig) where
  input_layernorm : Gemma4RMSNorm cfg.hidden_size
  post_attention_layernorm : Gemma4RMSNorm cfg.hidden_size
  pre_feedforward_layernorm : Gemma4RMSNorm cfg.hidden_size
  post_feedforward_layernorm : Gemma4RMSNorm cfg.hidden_size
  self_attn : Gemma4VisionAttention cfg
  mlp : Gemma4VisionMLP cfg
  deriving TensorStruct

namespace Gemma4VisionBlock

def forward {patchRows patchCols : UInt64}
    (cfg : VisionConfig)
    (m : Gemma4VisionBlock cfg)
    (x : Tensor { shape := #[patchRows * patchCols, cfg.hidden_size] })
    : Tensor { shape := #[patchRows * patchCols, cfg.hidden_size] } :=
  let residual1 := x
  let h1 := m.input_layernorm.forward2d x
  let attn := m.self_attn.forward (patchRows := patchRows) (patchCols := patchCols) cfg h1
  let h2 := residual1 + m.post_attention_layernorm.forward2d attn
  let residual2 := h2
  let h3 := m.pre_feedforward_layernorm.forward2d h2
  let mlp := m.mlp.forward cfg h3
  residual2 + m.post_feedforward_layernorm.forward2d mlp

end Gemma4VisionBlock

structure Gemma4VisionModel (cfg : VLConfig) where
  patch_embedder : Gemma4VisionPatchEmbedder cfg
  blocks : Array (Gemma4VisionBlock cfg.vision_config)
  std_bias : Option (Tensor { shape := #[cfg.vision_config.hidden_size] }) := none
  std_scale : Option (Tensor { shape := #[cfg.vision_config.hidden_size] }) := none
  deriving TensorStruct

namespace Gemma4VisionModel

private def poolFeatures {patchRows patchCols : UInt64}
    (cfg : VLConfig)
    (x : Tensor { shape := #[patchRows * patchCols, cfg.vision_config.hidden_size] })
    : Tensor { shape := #[(patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size),
        cfg.vision_config.hidden_size] } :=
  let pool := cfg.vision_config.pooling_kernel_size
  let pooledRows := patchRows / pool
  let pooledCols := patchCols / pool
  let xGrid : Tensor { shape := #[1, patchRows, patchCols, cfg.vision_config.hidden_size] } :=
    Tensor.reshapeChecked x #[1, patchRows, patchCols, cfg.vision_config.hidden_size]
  let xNchw : Tensor { shape := #[1, cfg.vision_config.hidden_size, patchRows, patchCols] } := (Tensor.permute xGrid #[0, 3, 1, 2]).reshapeChecked #[1, cfg.vision_config.hidden_size, patchRows, patchCols]
  let pooled : Tensor { shape := #[1, cfg.vision_config.hidden_size, pooledRows, pooledCols] } :=
    avgPool2d xNchw pool
  let pooledNhwc : Tensor { shape := #[1, pooledRows, pooledCols, cfg.vision_config.hidden_size] } :=
    (Tensor.permute pooled #[0, 2, 3, 1]).reshapeChecked #[1, pooledRows, pooledCols, cfg.vision_config.hidden_size]
  Tensor.Preserving.mulScalar
    (Tensor.reshapeChecked pooledNhwc #[pooledRows * pooledCols, cfg.vision_config.hidden_size])
    (Float.sqrt cfg.vision_config.hidden_size.toFloat)

def forward {patchRows patchCols : UInt64}
    (cfg : VLConfig)
    (m : Gemma4VisionModel cfg)
    (patchGrid : Tensor { shape := #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config] })
    : Tensor { shape := #[(patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size),
        cfg.vision_config.hidden_size] } :=
  Id.run do
    let mut hidden := m.patch_embedder.forward cfg patchGrid
    for i in [:m.blocks.size] do
      match m.blocks[i]? with
      | some blk =>
        hidden := blk.forward (patchRows := patchRows) (patchCols := patchCols) cfg.vision_config hidden
      | none =>
        pure ()
    let pooled := poolFeatures cfg hidden
    match m.std_bias, m.std_scale with
    | some bias, some scale =>
      let nTok := (patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size)
      let bias : Tensor { shape := #[nTok, cfg.vision_config.hidden_size] } :=
        Tensor.expand (Tensor.reshapeChecked bias #[1, cfg.vision_config.hidden_size]) #[nTok, cfg.vision_config.hidden_size]
      let scale : Tensor { shape := #[nTok, cfg.vision_config.hidden_size] } :=
        Tensor.expand (Tensor.reshapeChecked scale #[1, cfg.vision_config.hidden_size]) #[nTok, cfg.vision_config.hidden_size]
      (pooled - bias) * scale
    | _, _ =>
      pooled

end Gemma4VisionModel

structure Gemma4MultimodalEmbedder (cfg : VLConfig) where
  embedding_projection : Tensor { shape := #[cfg.text_config.hidden_size, cfg.vision_config.hidden_size] }
  deriving TensorStruct

namespace Gemma4MultimodalEmbedder

def forward {tokens : UInt64}
    (cfg : VLConfig)
    (m : Gemma4MultimodalEmbedder cfg)
    (inputsEmbeds : Tensor { shape := #[tokens, cfg.vision_config.hidden_size] })
    : Tensor { shape := #[tokens, cfg.text_config.hidden_size] } :=
  linear2d (rmsNormNoScale2d inputsEmbeds cfg.text_config.rms_norm_eps) m.embedding_projection

end Gemma4MultimodalEmbedder

structure Gemma4ForConditionalGeneration (cfg : VLConfig) where
  vision_tower : Gemma4VisionModel cfg
  embed_vision : Gemma4MultimodalEmbedder cfg
  language_model : Gemma4ForCausalLM cfg.text_config
  deriving TensorStruct

namespace Gemma4ForConditionalGeneration

def getImageFeatures {patchRows patchCols : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration cfg)
    (patchGrid : Tensor { shape := #[patchRows, patchCols, VisionConfig.patchDim cfg.vision_config] })
    : Tensor { shape := #[(patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size),
        cfg.text_config.hidden_size] } :=
  let visionFeatures := m.vision_tower.forward cfg patchGrid
  m.embed_vision.forward cfg visionFeatures

def getImageFeaturesMany
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration cfg)
    (images : Array (ImagePatchGrid cfg))
    : IO (Option (ImageFeatures cfg)) := do
  let mut acc? : Option (ImageFeatures cfg) := none
  for img in images do
    match img with
    | ⟨patchRows, ⟨patchCols, patchGrid⟩⟩ =>
      let nTok := (patchRows / cfg.vision_config.pooling_kernel_size) * (patchCols / cfg.vision_config.pooling_kernel_size)
      let feats : Tensor { shape := #[nTok, cfg.text_config.hidden_size] } := getImageFeatures cfg m patchGrid
      acc? :=
        match acc? with
        | none =>
          some ⟨nTok, feats⟩
        | some ⟨nPrev, prev⟩ =>
          some ⟨nPrev + nTok, typedCat prev feats 0⟩
  pure acc?

def repeatFeaturesForBatch {batch tokens : UInt64}
    (cfg : VLConfig)
    (features : Tensor { shape := #[tokens, cfg.text_config.hidden_size] })
    : ImageFeatures cfg :=
  Id.run do
    if batch <= 1 then
      return ⟨tokens, features⟩
    let repeated := Tensor.expand (features.reshapeChecked #[1, tokens, cfg.text_config.hidden_size])
      #[batch, tokens, cfg.text_config.hidden_size]
    return ⟨batch * tokens, repeated.reshapeChecked #[batch * tokens, cfg.text_config.hidden_size]⟩

private def scatterFeaturesIntoToken {batch seq hidden featTokens featDim : UInt64}
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, hidden] })
    (tokenId : UInt64)
    (features : Tensor { shape := #[featTokens, featDim] })
    : IO (Tensor { shape := #[batch, seq, hidden] }) := do
  if featDim != hidden then
    throw <| IO.userError
      s!"Gemma4 feature hidden size ({featDim}) does not match text hidden size ({hidden})"
  let tokenMask2d : Tensor { shape := #[batch, seq], dtype := .Bool } := Tensor.eqScalar inputIds (Int64.ofNat tokenId.toNat)
  let nTokens ← countMask2d tokenMask2d
  if nTokens != featTokens then
    throw <| IO.userError
      s!"Gemma4 placeholder token count mismatch for token={tokenId}: ids={nTokens} features={featTokens}"
  let tokenMask : Tensor { shape := #[batch, seq, hidden], dtype := .Bool } :=
    Tensor.expand (Tensor.reshapeChecked tokenMask2d #[batch, seq, 1]) #[batch, seq, hidden]
  let src : Tensor { shape := #[featTokens * featDim] } := Tensor.reshapeChecked features #[featTokens * featDim]
  pure (Tensor.maskedScatter inputsEmbeds tokenMask src)

private def buildInputEmbedsWithFeatures {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (imageFeatures : Option (ImageFeatures cfg) := none)
    : IO (Tensor { shape := #[batch, seq, cfg.text_config.hidden_size] } × PerLayerInputs cfg batch seq) := do
  let padId := cfg.text_config.pad_token_id.getD 0
  let llmInputIds : Tensor { shape := #[batch, seq], dtype := .Int64 } := replaceTokenId inputIds cfg.image_token_id padId
  let llmInputsEmbeds : Tensor { shape := #[batch, seq, cfg.text_config.hidden_size] } :=
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
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    : IO (Tensor { shape := #[batch, seq, seq], dtype := .Int64 } × Tensor { shape := #[batch, seq, seq], dtype := .Int64 }) := do
  let padId := cfg.text_config.pad_token_id.getD 0
  let slidingWindow := cfg.text_config.sliding_window.toNat
  let mut slidingFlat : Array Int64 := #[]
  let mut fullFlat : Array Int64 := #[]

  for b in [:batch.toNat] do
    let row2 : Tensor { shape := #[1, seq], dtype := .Int64 } := sliceTyped inputIds 0 b.toUInt64 1
    let row1 : Tensor { shape := #[seq], dtype := .Int64 } := Tensor.reshapeChecked (Tensor.toInt64 row2) #[seq]
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

  let slidingMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 } :=
    Tensor.reshapeChecked (Tensor.fromInt64Array slidingFlat) #[batch, seq, seq]
  let fullMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 } :=
    Tensor.reshapeChecked (Tensor.fromInt64Array fullFlat) #[batch, seq, seq]
  pure (slidingMask.to inputIds.device, fullMask.to inputIds.device)

private def bidirectionalVisionPrefillMasks? {batch seq : UInt64}
    (cfg : VLConfig)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    : IO (Option (Tensor { shape := #[batch, seq, seq], dtype := .Int64 } × Tensor { shape := #[batch, seq, seq], dtype := .Int64 })) := do
  if cfg.text_config.use_bidirectional_attention == "vision" then
    some <$> buildVisionBidirectionalPrefillMasks cfg inputIds
  else
    pure none

def forwardText {batch seq : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    : Tensor { shape := #[batch, seq, cfg.text_config.vocab_size] } :=
  m.language_model.forward cfg.text_config inputIds

def forwardWithImageFeatures {batch seq featTokens : UInt64}
    (cfg : VLConfig)
    (m : Gemma4ForConditionalGeneration cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (imageFeatures : Tensor { shape := #[featTokens, cfg.text_config.hidden_size] })
    : IO (Tensor { shape := #[batch, seq, cfg.text_config.vocab_size] }) := do
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
    (m : Gemma4ForConditionalGeneration cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (ImageFeatures cfg) := none)
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
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
    (m : Gemma4ForConditionalGeneration cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (imageFeatures : Option (ImageFeatures cfg) := none)
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
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
