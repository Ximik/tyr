/-
  Tyr/Model/Gemma4/Model.lean

  Standalone Gemma 4 text causal-LM for Tyr.
  Supports:
  - Sliding + full attention hybrid schedule
  - Small-model KV sharing (E2B/E4B)
  - Per-layer input embeddings/projections (E2B/E4B)
  - Dense and MoE feed-forward variants
-/
import Tyr.Model.Gemma4.TypedOps
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen.Attention
import Tyr.Model.Gemma4.Config
import Tyr.Model.Generation

namespace torch.gemma4

open torch
open torch.Model torch.qwen35

private def reqGradFalse {s : Shape} (t : Tensor { shape := s }) : Tensor { shape := s } :=
  Tensor.setRequiresGrad t false

/-- Gemma 4 RMSNorm with standard learned scale. -/
structure Gemma4RMSNorm (dim : UInt64) where
  weight : Tensor { shape := #[dim] }
  eps : Float := 1e-6
  deriving TensorStruct

namespace Gemma4RMSNorm

def fromCheckpointWeight {dim : UInt64}
    (weight : Tensor { shape := #[dim] })
    (eps : Float := 1e-6)
    : Gemma4RMSNorm dim :=
  { weight := weight, eps }

def forward2d {n dim : UInt64}
    (m : Gemma4RMSNorm dim)
    (x : Tensor { shape := #[n, dim] })
    : Tensor { shape := #[n, dim] } :=
  let xf : Tensor { shape := #[n, dim] } := floatDynamic x
  let var : Tensor { shape := #[n, 1] } := Tensor.Preserving.meanDim (xf * xf) 1 true
  let inv : Tensor { shape := #[n, 1] } := rsqrtDynamic (var + m.eps)
  let inv : Tensor { shape := #[n, dim] } := Tensor.expand inv #[n, dim]
  let scale : Tensor { shape := #[n, dim] } := Tensor.expand (Tensor.reshapeChecked (floatDynamic m.weight) #[1, dim]) #[n, dim]
  Tensor.castLike x (xf * inv * scale)

def forward3d {batch seq dim : UInt64}
    (m : Gemma4RMSNorm dim)
    (x : Tensor { shape := #[batch, seq, dim] })
    : Tensor { shape := #[batch, seq, dim] } :=
  let flat : Tensor { shape := #[batch * seq, dim] } := Tensor.reshapeChecked x #[batch * seq, dim]
  Tensor.reshapeChecked (forward2d m flat) #[batch, seq, dim]

def forward4d {a b c dim : UInt64}
    (m : Gemma4RMSNorm dim)
    (x : Tensor { shape := #[a, b, c, dim] })
    : Tensor { shape := #[a, b, c, dim] } :=
  let flat : Tensor { shape := #[a * b * c, dim] } := Tensor.reshapeChecked x #[a * b * c, dim]
  Tensor.reshapeChecked (forward2d m flat) #[a, b, c, dim]

end Gemma4RMSNorm

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

private def rmsNormWeighted4dSlice {a b c dim maxDim : UInt64}
    (weight : Tensor { shape := #[maxDim] })
    (x : Tensor { shape := #[a, b, c, dim] })
    (eps : Float := 1e-6)
    : Tensor { shape := #[a, b, c, dim] } :=
  let w : Tensor { shape := #[dim] } := sliceTyped weight 0 0 dim
  let xf : Tensor { shape := #[a * b * c, dim] } := Tensor.reshapeChecked x #[a * b * c, dim]
  let var : Tensor { shape := #[a * b * c, 1] } := Tensor.Preserving.meanDim (xf * xf) 1 true
  let inv : Tensor { shape := #[a * b * c, 1] } := rsqrtDynamic (var + eps)
  let inv : Tensor { shape := #[a * b * c, dim] } := Tensor.expand inv #[a * b * c, dim]
  let scale : Tensor { shape := #[a * b * c, dim] } := Tensor.expand (Tensor.reshapeChecked (floatDynamic w) #[1, dim]) #[a * b * c, dim]
  Tensor.castLike x (Tensor.reshapeChecked (xf * inv * scale) #[a, b, c, dim])

private def rmsNormNoScale4dSlice {a b c dim : UInt64}
    (x : Tensor { shape := #[a, b, c, dim] })
    (eps : Float := 1e-6)
    : Tensor { shape := #[a, b, c, dim] } :=
  rmsNormNoScale4d x eps

/-- Dense Gemma 4 MLP.
    Layers whose checkpoint width is smaller than `Config.maxIntermediateSize`
    are zero-padded during load. -/
structure Gemma4MLP (cfg : Config) where
  gate_proj : Tensor { shape := #[Config.maxIntermediateSize cfg, cfg.hidden_size] }
  up_proj : Tensor { shape := #[Config.maxIntermediateSize cfg, cfg.hidden_size] }
  down_proj : Tensor { shape := #[cfg.hidden_size, Config.maxIntermediateSize cfg] }
  deriving TensorStruct

namespace Gemma4MLP

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4MLP cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  Tensor.linear3d (Tensor.Preserving.gelu (Tensor.linear3d x m.gate_proj) * Tensor.linear3d x m.up_proj) m.down_proj

end Gemma4MLP

structure Gemma4TextRouter (cfg : Config) where
  proj : Tensor { shape := #[cfg.num_experts, cfg.hidden_size] }
  scale : Tensor { shape := #[cfg.hidden_size] }
  per_expert_scale : Tensor { shape := #[cfg.num_experts] }
  deriving TensorStruct

namespace Gemma4TextRouter

def forward {tokens : UInt64}
    (cfg : Config)
    (m : Gemma4TextRouter cfg)
    (hidden : Tensor { shape := #[tokens, cfg.hidden_size] })
    : Tensor { shape := #[tokens, cfg.num_experts] }
      × Tensor { shape := #[tokens, cfg.top_k_experts] }
      × Tensor { shape := #[tokens, cfg.top_k_experts], dtype := .Int64 } :=
  let normed : Tensor { shape := #[tokens, cfg.hidden_size] } := rmsNormNoScale2d hidden cfg.rms_norm_eps
  let scaleRoot := 1.0 / Float.sqrt cfg.hidden_size.toFloat
  let scale : Tensor { shape := #[tokens, cfg.hidden_size] } :=
    Tensor.expand (Tensor.reshapeChecked (floatDynamic m.scale) #[1, cfg.hidden_size]) #[tokens, cfg.hidden_size]
  let normed : Tensor { shape := #[tokens, cfg.hidden_size] } := Tensor.Preserving.mulScalar (normed * scale) scaleRoot
  let logitsDyn : Tensor { shape := #[tokens, cfg.num_experts] } := einsumChecked "eh,th->te" (floatDynamic m.proj) normed
  let logits : Tensor { shape := #[tokens, cfg.num_experts] } := Tensor.reshapeChecked logitsDyn #[tokens, cfg.num_experts]
  let probs : Tensor { shape := #[tokens, cfg.num_experts] } := Tensor.Preserving.softmax logits (-1)
  let (topValsRaw, topIdx) := topK2d probs cfg.top_k_experts 1
  let denom : Tensor { shape := #[tokens, 1] } := Tensor.Preserving.sumDim topValsRaw 1 true
  let denom : Tensor { shape := #[tokens, cfg.top_k_experts] } := Tensor.expand denom #[tokens, cfg.top_k_experts]
  let topVals : Tensor { shape := #[tokens, cfg.top_k_experts] } := Tensor.Preserving.div topValsRaw denom
  let flatIdx : Tensor { shape := #[tokens * cfg.top_k_experts], dtype := .Int64 } := Tensor.reshapeChecked topIdx #[tokens * cfg.top_k_experts]
  let scalesFlat : Tensor { shape := #[tokens * cfg.top_k_experts] } :=
    Tensor.indexSelect m.per_expert_scale flatIdx
  let scales : Tensor { shape := #[tokens, cfg.top_k_experts] } := Tensor.reshapeChecked scalesFlat #[tokens, cfg.top_k_experts]
  (probs, topVals * scales, topIdx)

end Gemma4TextRouter

structure Gemma4TextExperts (cfg : Config) where
  gate_up_proj : Tensor { shape := #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size] }
  down_proj : Tensor { shape := #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size] }
  deriving TensorStruct

namespace Gemma4TextExperts

def forward2d {tokens : UInt64}
    (cfg : Config)
    (m : Gemma4TextExperts cfg)
    (hidden : Tensor { shape := #[tokens, cfg.hidden_size] })
    (topVals : Tensor { shape := #[tokens, cfg.top_k_experts] })
    (topIdx : Tensor { shape := #[tokens, cfg.top_k_experts], dtype := .Int64 })
    : Tensor { shape := #[tokens, cfg.hidden_size] } :=
  routedTextExpertsForward hidden m.gate_up_proj m.down_proj topVals topIdx

end Gemma4TextExperts

structure Gemma4MoeBranch (cfg : Config) where
  router : Gemma4TextRouter cfg
  experts : Gemma4TextExperts cfg
  post_feedforward_layernorm_1 : Gemma4RMSNorm cfg.hidden_size
  post_feedforward_layernorm_2 : Gemma4RMSNorm cfg.hidden_size
  pre_feedforward_layernorm_2 : Gemma4RMSNorm cfg.hidden_size
  deriving TensorStruct

structure Gemma4PerLayerInputBlock (cfg : Config) where
  per_layer_input_gate : Tensor { shape := #[cfg.hidden_size_per_layer_input, cfg.hidden_size] }
  per_layer_projection : Tensor { shape := #[cfg.hidden_size, cfg.hidden_size_per_layer_input] }
  post_per_layer_input_norm : Gemma4RMSNorm cfg.hidden_size
  deriving TensorStruct

structure Gemma4Attention (cfg : Config) where
  q_proj : Tensor { shape := #[Config.maxAttentionDim cfg, cfg.hidden_size] }
  k_proj : Tensor { shape := #[Config.maxKVProjDim cfg, cfg.hidden_size] }
  v_proj : Tensor { shape := #[Config.maxKVProjDim cfg, cfg.hidden_size] }
  o_proj : Tensor { shape := #[cfg.hidden_size, Config.maxAttentionDim cfg] }
  q_norm : Gemma4RMSNorm (Config.maxHeadDim cfg)
  k_norm : Gemma4RMSNorm (Config.maxHeadDim cfg)
  deriving TensorStruct

namespace Gemma4Attention

abbrev KVCache (cfg : Config) (batch : UInt64) :=
  qwen.QwenAttention.KVCache batch (Config.maxKVHeads cfg) (Config.maxHeadDim cfg) (.Unknown "dynamic")

private def qProjSliding {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    : Tensor { shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim] } :=
  let qW : Tensor { shape := #[cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size] } :=
    sliceTyped m.q_proj 0 0 (cfg.num_attention_heads * cfg.head_dim)
  Tensor.reshapeChecked (Tensor.linear3d x qW) #[batch, seq, cfg.num_attention_heads, cfg.head_dim]

private def qProjFull {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    : Tensor { shape := #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg] } :=
  let dim := cfg.num_attention_heads * Config.fullHeadDim cfg
  let qW : Tensor { shape := #[dim, cfg.hidden_size] } := sliceTyped m.q_proj 0 0 dim
  Tensor.reshapeChecked (Tensor.linear3d x qW) #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg]

private def kProjSliding {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } :=
  let rows := cfg.num_key_value_heads * cfg.head_dim
  let kW : Tensor { shape := #[rows, cfg.hidden_size] } := sliceTyped m.k_proj 0 0 rows
  Tensor.reshapeChecked (Tensor.linear3d x kW) #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

private def kProjFull {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    : Tensor { shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] } :=
  let rows := Config.fullNumKVHeads cfg * Config.fullHeadDim cfg
  let kW : Tensor { shape := #[rows, cfg.hidden_size] } := sliceTyped m.k_proj 0 0 rows
  Tensor.reshapeChecked (Tensor.linear3d x kW) #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg]

private def vProjSliding {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } :=
  let rows := cfg.num_key_value_heads * cfg.head_dim
  let vW : Tensor { shape := #[rows, cfg.hidden_size] } := sliceTyped m.v_proj 0 0 rows
  Tensor.reshapeChecked (Tensor.linear3d x vW) #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

private def vProjFull {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    : Tensor { shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] } :=
  let rows := Config.fullNumKVHeads cfg * Config.fullHeadDim cfg
  let vW : Tensor { shape := #[rows, cfg.hidden_size] } := sliceTyped m.v_proj 0 0 rows
  Tensor.reshapeChecked (Tensor.linear3d x vW) #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg]

private def oProjSliding {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim] })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let cols := cfg.num_attention_heads * cfg.head_dim
  let oW : Tensor { shape := #[cfg.hidden_size, cols] } := sliceTyped m.o_proj 1 0 cols
  Tensor.linear3d x oW

private def oProjFull {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg] })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let cols := cfg.num_attention_heads * Config.fullHeadDim cfg
  let oW : Tensor { shape := #[cfg.hidden_size, cols] } := sliceTyped m.o_proj 1 0 cols
  Tensor.linear3d x oW

private def padSlidingKVForCache {batch seq : UInt64}
    (cfg : Config)
    (x : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] })
    : Tensor { shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] } :=
  let padDim : UInt64 := Config.maxHeadDim cfg - cfg.head_dim
  let xDim : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, Config.maxHeadDim cfg] } :=
    typedCat x (Tensor.castLike x (zerosDynamicOn x.device : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, padDim] })) 3
  let padHeads : UInt64 := Config.maxKVHeads cfg - cfg.num_key_value_heads
  typedCat xDim (Tensor.castLike x (zerosDynamicOn x.device : Tensor { shape := #[batch, padHeads, seq, Config.maxHeadDim cfg] })) 1

private def padFullKVForCache {batch seq : UInt64}
    (cfg : Config)
    (x : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] })
    : Tensor { shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] } :=
  let padDim : UInt64 := Config.maxHeadDim cfg - Config.fullHeadDim cfg
  let xDim : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.maxHeadDim cfg] } :=
    typedCat x (Tensor.castLike x (zerosDynamicOn x.device : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, seq, padDim] })) 3
  let padHeads : UInt64 := Config.maxKVHeads cfg - Config.fullNumKVHeads cfg
  typedCat xDim (Tensor.castLike x (zerosDynamicOn x.device : Tensor { shape := #[batch, padHeads, seq, Config.maxHeadDim cfg] })) 1

private def cacheSliceSliding {batch kvLen : UInt64}
    (cfg : Config)
    (cache : KVCache cfg batch)
    : Tensor { shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] }
      × Tensor { shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] } :=
  let kStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
    Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let vStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
    Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let k0 : Tensor { shape := #[batch, cfg.num_key_value_heads, kvLen, Config.maxHeadDim cfg] } :=
    sliceTyped (sliceTyped kStore 1 0 cfg.num_key_value_heads) 2 0 kvLen
  let v0 : Tensor { shape := #[batch, cfg.num_key_value_heads, kvLen, Config.maxHeadDim cfg] } :=
    sliceTyped (sliceTyped vStore 1 0 cfg.num_key_value_heads) 2 0 kvLen
  let k : Tensor { shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] } := sliceTyped k0 3 0 cfg.head_dim
  let v : Tensor { shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] } := sliceTyped v0 3 0 cfg.head_dim
  (k, v)

private def cacheSliceFull {batch kvLen : UInt64}
    (cfg : Config)
    (cache : KVCache cfg batch)
    : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg] }
      × Tensor { shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg] } :=
  let kStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
    Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let vStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
    Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let k0 : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.maxHeadDim cfg] } :=
    sliceTyped (sliceTyped kStore 1 0 (Config.fullNumKVHeads cfg)) 2 0 kvLen
  let v0 : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.maxHeadDim cfg] } :=
    sliceTyped (sliceTyped vStore 1 0 (Config.fullNumKVHeads cfg)) 2 0 kvLen
  let k : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg] } :=
    sliceTyped k0 3 0 (Config.fullHeadDim cfg)
  let v : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg] } :=
    sliceTyped v0 3 0 (Config.fullHeadDim cfg)
  (k, v)

private def attentionScale (d : UInt64) : Float :=
  Float.sqrt d.toFloat

private def attentionSlidingPrefillWithMask {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (attnMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 })
    (cache : KVCache cfg batch)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × KVCache cfg batch :=
  let q0 := qProjSliding cfg m x
  let kRaw := kProjSliding cfg m x
  let vRaw := vProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
  let q2 : Tensor { shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim] } := Tensor.applyRotary q1 cos sin
  let k2 : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } := Tensor.applyRotary k1 cos sin
  let v2 : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
  let qh : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } :=
    Tensor.Preserving.mulScalar (transposeAttention q2) (attentionScale cfg.head_dim)
  let kh : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := transposeAttention k2
  let vh : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := transposeAttention v2
  let attn : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } :=
    maskedAttention qh kh vh attnMask 0.0 true
  let out : Tensor { shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim] } :=
    Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
  let paddedK : Tensor { shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] } := padSlidingKVForCache cfg kh
  let paddedV : Tensor { shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] } := padSlidingKVForCache cfg vh
  let kStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
    Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let vStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
    Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let kStore' := Tensor.sliceScatter kStore 2 0 paddedK
  let vStore' := Tensor.sliceScatter vStore 2 0 paddedV
  let cache' : KVCache cfg batch := {
    kStoreDyn := kStore'.packShape
    vStoreDyn := vStore'.packShape
    seq := seq
    maxLen := cache.maxLen
  }
  (oProjSliding cfg m out, cache')

private def attentionFullPrefillWithMask {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (attnMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 })
    (cache : KVCache cfg batch)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × KVCache cfg batch :=
  let q0 := qProjFull cfg m x
  let kRaw := kProjFull cfg m x
  let vRaw :=
    if cfg.attention_k_eq_v then
      kRaw
    else
      vProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
  let q2 : Tensor { shape := #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg] } := Tensor.applyRotary q1 cos sin
  let k2 : Tensor { shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] } := Tensor.applyRotary k1 cos sin
  let v2 : Tensor { shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
  let qh : Tensor { shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] } :=
    Tensor.Preserving.mulScalar (transposeAttention q2) (attentionScale (Config.fullHeadDim cfg))
  let kh : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] } := transposeAttention k2
  let vh : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] } := transposeAttention v2
  let attn : Tensor { shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] } :=
    maskedAttention qh kh vh attnMask 0.0 true
  let out : Tensor { shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg] } :=
    Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
  let paddedK : Tensor { shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] } := padFullKVForCache cfg kh
  let paddedV : Tensor { shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] } := padFullKVForCache cfg vh
  let kStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
    Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let vStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
    Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let kStore' := Tensor.sliceScatter kStore 2 0 paddedK
  let vStore' := Tensor.sliceScatter vStore 2 0 paddedV
  let cache' : KVCache cfg batch := {
    kStoreDyn := kStore'.packShape
    vStoreDyn := vStore'.packShape
    seq := seq
    maxLen := cache.maxLen
  }
  (oProjFull cfg m out, cache')

private def attentionSlidingNoCache {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let q0 := qProjSliding cfg m x
  let k0 := kProjSliding cfg m x
  let v0 := vProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let k1 := rmsNormWeighted4dSlice m.k_norm.weight k0 m.k_norm.eps
  let q2 : Tensor { shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim] } := Tensor.applyRotary q1 cos sin
  let k2 : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } := Tensor.applyRotary k1 cos sin
  let v2 : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } := rmsNormNoScale4dSlice v0 cfg.rms_norm_eps
  let qh : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } :=
    Tensor.Preserving.mulScalar (transposeAttention q2) (attentionScale cfg.head_dim)
  let kh : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := transposeAttention k2
  let vh : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := transposeAttention v2
  let attn : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } :=
    Tensor.attentionWindow qh kh vh 0.0 true true cfg.sliding_window
  let out : Tensor { shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim] } :=
    Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
  oProjSliding cfg m out

private def attentionFullNoCache {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let q0 := qProjFull cfg m x
  let kRaw := kProjFull cfg m x
  let vRaw :=
    if cfg.attention_k_eq_v then
      kRaw
    else
      vProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
  let q2 : Tensor { shape := #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg] } := Tensor.applyRotary q1 cos sin
  let k2 : Tensor { shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] } := Tensor.applyRotary k1 cos sin
  let v2 : Tensor { shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
  let qh : Tensor { shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] } :=
    Tensor.Preserving.mulScalar (transposeAttention q2) (attentionScale (Config.fullHeadDim cfg))
  let kh : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] } := transposeAttention k2
  let vh : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] } := transposeAttention v2
  let attn : Tensor { shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] } :=
    Tensor.attentionGQA qh kh vh 0.0 true true
  let out : Tensor { shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg] } :=
    Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
  oProjFull cfg m out

private def attentionSlidingWithCache {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (cache : KVCache cfg batch)
    (sharedSource : Option (KVCache cfg batch) := none)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × KVCache cfg batch :=
  let q0 := qProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let q2 : Tensor { shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim] } := Tensor.applyRotary q1 cos sin
  let qh : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } :=
    Tensor.Preserving.mulScalar (transposeAttention q2) (attentionScale cfg.head_dim)
  match sharedSource with
  | some src =>
    let (kAll0, vAll0) := cacheSliceSliding (kvLen := seq) cfg src
    let attn : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } :=
      Tensor.attentionWindow qh kAll0 vAll0 0.0 true true cfg.sliding_window
    let out : Tensor { shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim] } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
    (oProjSliding cfg m out, cache)
  | none =>
    let kRaw := kProjSliding cfg m x
    let vRaw := vProjSliding cfg m x
    let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
    let k2 : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } := Tensor.applyRotary k1 cos sin
    let v2 : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
    let kh : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := transposeAttention k2
    let vh : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := transposeAttention v2
    let paddedK := padSlidingKVForCache cfg kh
    let paddedV := padSlidingKVForCache cfg vh
    let kStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
      Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let vStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
      Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let kStore' := Tensor.sliceScatter kStore 2 0 paddedK
    let vStore' := Tensor.sliceScatter vStore 2 0 paddedV
    let cache' : KVCache cfg batch := {
      kStoreDyn := kStore'.packShape
      vStoreDyn := vStore'.packShape
      seq := seq
      maxLen := cache.maxLen
    }
    let attn : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } :=
      Tensor.attentionWindow qh kh vh 0.0 true true cfg.sliding_window
    let out : Tensor { shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim] } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
    (oProjSliding cfg m out, cache')

private def attentionFullWithCache {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (cache : KVCache cfg batch)
    (sharedSource : Option (KVCache cfg batch) := none)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × KVCache cfg batch :=
  let q0 := qProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let q2 : Tensor { shape := #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg] } := Tensor.applyRotary q1 cos sin
  let qh : Tensor { shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] } :=
    Tensor.Preserving.mulScalar (transposeAttention q2) (attentionScale (Config.fullHeadDim cfg))
  match sharedSource with
  | some src =>
    let (kAll, vAll) := cacheSliceFull (kvLen := seq) cfg src
    let attn : Tensor { shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] } :=
      Tensor.attentionGQA qh kAll vAll 0.0 true true
    let out : Tensor { shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg] } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
    (oProjFull cfg m out, cache)
  | none =>
    let kRaw := kProjFull cfg m x
    let vRaw :=
      if cfg.attention_k_eq_v then
        kRaw
      else
        vProjFull cfg m x
    let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
    let k2 : Tensor { shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] } := Tensor.applyRotary k1 cos sin
    let v2 : Tensor { shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
    let kh : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] } := transposeAttention k2
    let vh : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg] } := transposeAttention v2
    let paddedK := padFullKVForCache cfg kh
    let paddedV := padFullKVForCache cfg vh
    let kStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
      Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let vStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
      Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let kStore' := Tensor.sliceScatter kStore 2 0 paddedK
    let vStore' := Tensor.sliceScatter vStore 2 0 paddedV
    let cache' : KVCache cfg batch := {
      kStoreDyn := kStore'.packShape
      vStoreDyn := vStore'.packShape
      seq := seq
      maxLen := cache.maxLen
    }
    let attn : Tensor { shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg] } :=
      Tensor.attentionGQA qh kh vh 0.0 true true
    let out : Tensor { shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg] } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
    (oProjFull cfg m out, cache')

private def attentionSlidingStep {batch : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, 1, cfg.hidden_size] })
    (cos : Tensor { shape := #[1, cfg.head_dim / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[1, cfg.head_dim / 2], dtype := .Float32 })
    (cache : KVCache cfg batch)
    (sharedSource : Option (KVCache cfg batch) := none)
    : Tensor { shape := #[batch, 1, cfg.hidden_size] } × KVCache cfg batch :=
  let q0 := qProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let q2 : Tensor { shape := #[batch, 1, cfg.num_attention_heads, cfg.head_dim] } := Tensor.applyRotary q1 cos sin
  let qh : Tensor { shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim] } :=
    Tensor.Preserving.mulScalar (transposeAttention q2) (attentionScale cfg.head_dim)
  match sharedSource with
  | some src =>
    let kvLen := src.seq
    let useLen := if kvLen > cfg.sliding_window then cfg.sliding_window else kvLen
    let start := kvLen - useLen
    let (kAll0, vAll0) := cacheSliceSliding (kvLen := kvLen) cfg src
    let kAll : Tensor { shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim] } := sliceTyped kAll0 2 start useLen
    let vAll : Tensor { shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim] } := sliceTyped vAll0 2 start useLen
    let attn : Tensor { shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim] } :=
      Tensor.attentionGQAQKV qh kAll vAll 0.0 false true
    let out : Tensor { shape := #[batch, 1, cfg.num_attention_heads * cfg.head_dim] } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, 1, cfg.num_attention_heads * cfg.head_dim]
    (oProjSliding cfg m out, cache)
  | none =>
    let kRaw := kProjSliding cfg m x
    let vRaw := vProjSliding cfg m x
    let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
    let k2 : Tensor { shape := #[batch, 1, cfg.num_key_value_heads, cfg.head_dim] } := Tensor.applyRotary k1 cos sin
    let v2 : Tensor { shape := #[batch, 1, cfg.num_key_value_heads, cfg.head_dim] } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
    let kNew : Tensor { shape := #[batch, cfg.num_key_value_heads, 1, cfg.head_dim] } := transposeAttention k2
    let vNew : Tensor { shape := #[batch, cfg.num_key_value_heads, 1, cfg.head_dim] } := transposeAttention v2
    let paddedK : Tensor { shape := #[batch, Config.maxKVHeads cfg, 1, Config.maxHeadDim cfg] } := padSlidingKVForCache cfg kNew
    let paddedV : Tensor { shape := #[batch, Config.maxKVHeads cfg, 1, Config.maxHeadDim cfg] } := padSlidingKVForCache cfg vNew
    let kStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
      Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let vStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
      Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let writePos := if cache.seq < cache.maxLen then cache.seq else cache.maxLen - 1
    let kStore' := Tensor.sliceScatter kStore 2 writePos paddedK
    let vStore' := Tensor.sliceScatter vStore 2 writePos paddedV
    let kvLen := if cache.seq < cache.maxLen then cache.seq + 1 else cache.maxLen
    let useLen := if kvLen > cfg.sliding_window then cfg.sliding_window else kvLen
    let start := kvLen - useLen
    let kAll0 : Tensor { shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] } :=
      (cacheSliceSliding (kvLen := kvLen) cfg { cache with kStoreDyn := kStore'.packShape, vStoreDyn := vStore'.packShape, seq := kvLen }) |>.1
    let vAll0 : Tensor { shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] } :=
      (cacheSliceSliding (kvLen := kvLen) cfg { cache with kStoreDyn := kStore'.packShape, vStoreDyn := vStore'.packShape, seq := kvLen }) |>.2
    let kAll : Tensor { shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim] } := sliceTyped kAll0 2 start useLen
    let vAll : Tensor { shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim] } := sliceTyped vAll0 2 start useLen
    let attn : Tensor { shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim] } :=
      Tensor.attentionGQAQKV qh kAll vAll 0.0 false true
    let out : Tensor { shape := #[batch, 1, cfg.num_attention_heads * cfg.head_dim] } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, 1, cfg.num_attention_heads * cfg.head_dim]
    let cache' : KVCache cfg batch := {
      kStoreDyn := kStore'.packShape
      vStoreDyn := vStore'.packShape
      seq := kvLen
      maxLen := cache.maxLen
    }
    (oProjSliding cfg m out, cache')

private def attentionFullStep {batch : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (x : Tensor { shape := #[batch, 1, cfg.hidden_size] })
    (cos : Tensor { shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (cache : KVCache cfg batch)
    (sharedSource : Option (KVCache cfg batch) := none)
    : Tensor { shape := #[batch, 1, cfg.hidden_size] } × KVCache cfg batch :=
  let q0 := qProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight q0 m.q_norm.eps
  let q2 : Tensor { shape := #[batch, 1, cfg.num_attention_heads, Config.fullHeadDim cfg] } := Tensor.applyRotary q1 cos sin
  let qh : Tensor { shape := #[batch, cfg.num_attention_heads, 1, Config.fullHeadDim cfg] } :=
    Tensor.Preserving.mulScalar (transposeAttention q2) (attentionScale (Config.fullHeadDim cfg))
  match sharedSource with
  | some src =>
    let kvLen := src.seq
    let (kAll, vAll) := cacheSliceFull (kvLen := kvLen) cfg src
    let attn : Tensor { shape := #[batch, cfg.num_attention_heads, 1, Config.fullHeadDim cfg] } :=
      Tensor.attentionGQAQKV qh kAll vAll 0.0 false true
    let out : Tensor { shape := #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg] } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg]
    (oProjFull cfg m out, cache)
  | none =>
    let kRaw := kProjFull cfg m x
    let vRaw :=
      if cfg.attention_k_eq_v then
        kRaw
      else
        vProjFull cfg m x
    let k1 := rmsNormWeighted4dSlice m.k_norm.weight kRaw m.k_norm.eps
    let k2 : Tensor { shape := #[batch, 1, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] } := Tensor.applyRotary k1 cos sin
    let v2 : Tensor { shape := #[batch, 1, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg] } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
    let kNew : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, 1, Config.fullHeadDim cfg] } := transposeAttention k2
    let vNew : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, 1, Config.fullHeadDim cfg] } := transposeAttention v2
    let paddedK := padFullKVForCache cfg kNew
    let paddedV := padFullKVForCache cfg vNew
    let kStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
      Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let vStore : Tensor { shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg] } :=
      Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let writePos := if cache.seq < cache.maxLen then cache.seq else cache.maxLen - 1
    let kStore' := Tensor.sliceScatter kStore 2 writePos paddedK
    let vStore' := Tensor.sliceScatter vStore 2 writePos paddedV
    let kvLen := if cache.seq < cache.maxLen then cache.seq + 1 else cache.maxLen
    let kAll : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg] } :=
      (cacheSliceFull (kvLen := kvLen) cfg { cache with kStoreDyn := kStore'.packShape, vStoreDyn := vStore'.packShape, seq := kvLen }) |>.1
    let vAll : Tensor { shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg] } :=
      (cacheSliceFull (kvLen := kvLen) cfg { cache with kStoreDyn := kStore'.packShape, vStoreDyn := vStore'.packShape, seq := kvLen }) |>.2
    let attn : Tensor { shape := #[batch, cfg.num_attention_heads, 1, Config.fullHeadDim cfg] } :=
      Tensor.attentionGQAQKV qh kAll vAll 0.0 false true
    let out : Tensor { shape := #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg] } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg]
    let cache' : KVCache cfg batch := {
      kStoreDyn := kStore'.packShape
      vStoreDyn := vStore'.packShape
      seq := kvLen
      maxLen := cache.maxLen
    }
    (oProjFull cfg m out, cache')

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (layerType : LayerType)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (slidingCos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  match layerType with
  | .slidingAttention => attentionSlidingNoCache cfg m x slidingCos slidingSin
  | .fullAttention => attentionFullNoCache cfg m x fullCos fullSin

def forwardWithCache {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (layerType : LayerType)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (slidingCos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (cache : KVCache cfg batch)
    (sharedSource : Option (KVCache cfg batch) := none)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × KVCache cfg batch :=
  match layerType with
  | .slidingAttention => attentionSlidingWithCache cfg m x slidingCos slidingSin cache sharedSource
  | .fullAttention => attentionFullWithCache cfg m x fullCos fullSin cache sharedSource

def forwardStep {batch : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (layerType : LayerType)
    (x : Tensor { shape := #[batch, 1, cfg.hidden_size] })
    (slidingCos : Tensor { shape := #[1, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { shape := #[1, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (cache : KVCache cfg batch)
    (sharedSource : Option (KVCache cfg batch) := none)
    : Tensor { shape := #[batch, 1, cfg.hidden_size] } × KVCache cfg batch :=
  match layerType with
  | .slidingAttention => attentionSlidingStep cfg m x slidingCos slidingSin cache sharedSource
  | .fullAttention => attentionFullStep cfg m x fullCos fullSin cache sharedSource

def forwardPrefillWithMask {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Attention cfg)
    (layerType : LayerType)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (slidingCos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (slidingAttnMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 })
    (cache : KVCache cfg batch)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × KVCache cfg batch :=
  match layerType with
  | .slidingAttention =>
    attentionSlidingPrefillWithMask cfg m x slidingCos slidingSin slidingAttnMask cache
  | .fullAttention =>
    attentionFullPrefillWithMask cfg m x fullCos fullSin fullAttnMask cache

end Gemma4Attention

structure Gemma4Layer (cfg : Config) where
  layerIdx : UInt64
  layerType : LayerType
  self_attn : Gemma4Attention cfg
  mlp : Gemma4MLP cfg
  input_layernorm : Gemma4RMSNorm cfg.hidden_size
  post_attention_layernorm : Gemma4RMSNorm cfg.hidden_size
  pre_feedforward_layernorm : Gemma4RMSNorm cfg.hidden_size
  post_feedforward_layernorm : Gemma4RMSNorm cfg.hidden_size
  per_layer_input : Option (Gemma4PerLayerInputBlock cfg) := none
  moe : Option (Gemma4MoeBranch cfg) := none
  layer_scalar : Tensor { shape := #[1] }
  deriving TensorStruct

structure Gemma4Cache (cfg : Config) (batch : UInt64) where
  attnCaches : Array (Gemma4Attention.KVCache cfg batch)

namespace Gemma4Layer

private def applyLayerScalar {batch seq : UInt64}
    (cfg : Config)
    (scale : Tensor { shape := #[1] })
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let s : Tensor { shape := #[batch, seq, cfg.hidden_size] } := Tensor.expand (Tensor.reshapeChecked scale #[1, 1, 1]) #[batch, seq, cfg.hidden_size]
  x * s

private def applyPerLayerInput {batch seq : UInt64}
    (cfg : Config)
    (block : Gemma4PerLayerInputBlock cfg)
    (hidden : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (perLayerInput : Tensor { shape := #[batch, seq, cfg.hidden_size_per_layer_input] })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let residual := hidden
  let gated : Tensor { shape := #[batch, seq, cfg.hidden_size_per_layer_input] } :=
    Tensor.Preserving.gelu (Tensor.linear3d hidden block.per_layer_input_gate)
  let mixed := gated * perLayerInput
  let projected : Tensor { shape := #[batch, seq, cfg.hidden_size] } := Tensor.linear3d mixed block.per_layer_projection
  let projected := block.post_per_layer_input_norm.forward3d projected
  residual + projected

def forward {batch seq : UInt64}
    (cfg : Config)
    (layer : Gemma4Layer cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (slidingCos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (perLayerInput : Option (Tensor { shape := #[batch, seq, cfg.hidden_size_per_layer_input] }) := none)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x
  let attn := layer.self_attn.forward cfg layer.layerType h1 slidingCos slidingSin fullCos fullSin
  let h2 := residual1 + layer.post_attention_layernorm.forward3d attn

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let tokens := batch * seq
      let routerInput : Tensor { shape := #[tokens, cfg.hidden_size] } := Tensor.reshapeChecked residual2 #[tokens, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : Tensor { shape := #[tokens, cfg.hidden_size] } :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals topIdx
      let expert3d : Tensor { shape := #[batch, seq, cfg.hidden_size] } := Tensor.reshapeChecked expert2d #[batch, seq, cfg.hidden_size]
      denseBranch + moe.post_feedforward_layernorm_2.forward3d expert3d
    | none =>
      mlpOut
  let h4 := residual2 + layer.post_feedforward_layernorm.forward3d ffCombined
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  applyLayerScalar cfg layer.layer_scalar h5

def forwardWithCache {batch seq : UInt64}
    (cfg : Config)
    (layer : Gemma4Layer cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (slidingCos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (layerCache : Gemma4Attention.KVCache cfg batch)
    (sharedSource : Option (Gemma4Attention.KVCache cfg batch) := none)
    (perLayerInput : Option (Tensor { shape := #[batch, seq, cfg.hidden_size_per_layer_input] }) := none)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × Gemma4Attention.KVCache cfg batch :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x
  let (attn, layerCache') :=
    layer.self_attn.forwardWithCache
      cfg
      layer.layerType
      h1
      slidingCos
      slidingSin
      fullCos
      fullSin
      layerCache
      sharedSource
  let h2 := residual1 + layer.post_attention_layernorm.forward3d attn

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let tokens := batch * seq
      let routerInput : Tensor { shape := #[tokens, cfg.hidden_size] } := Tensor.reshapeChecked residual2 #[tokens, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : Tensor { shape := #[tokens, cfg.hidden_size] } :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals topIdx
      let expert3d : Tensor { shape := #[batch, seq, cfg.hidden_size] } := Tensor.reshapeChecked expert2d #[batch, seq, cfg.hidden_size]
      denseBranch + moe.post_feedforward_layernorm_2.forward3d expert3d
    | none =>
      mlpOut
  let h4 := residual2 + layer.post_feedforward_layernorm.forward3d ffCombined
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  (applyLayerScalar cfg layer.layer_scalar h5, layerCache')

def forwardStep {batch : UInt64}
    (cfg : Config)
    (layer : Gemma4Layer cfg)
    (x : Tensor { shape := #[batch, 1, cfg.hidden_size] })
    (slidingCos : Tensor { shape := #[1, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { shape := #[1, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (layerCache : Gemma4Attention.KVCache cfg batch)
    (sharedSource : Option (Gemma4Attention.KVCache cfg batch) := none)
    (perLayerInput : Option (Tensor { shape := #[batch, 1, cfg.hidden_size_per_layer_input] }) := none)
    : Tensor { shape := #[batch, 1, cfg.hidden_size] } × Gemma4Attention.KVCache cfg batch :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x
  let (attn, layerCache') :=
    layer.self_attn.forwardStep
      cfg
      layer.layerType
      h1
      slidingCos
      slidingSin
      fullCos
      fullSin
      layerCache
      sharedSource
  let h2 := residual1 + layer.post_attention_layernorm.forward3d attn

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let routerInput : Tensor { shape := #[batch, cfg.hidden_size] } := Tensor.reshapeChecked residual2 #[batch, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : Tensor { shape := #[batch, cfg.hidden_size] } :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals topIdx
      let expert3d : Tensor { shape := #[batch, 1, cfg.hidden_size] } := Tensor.reshapeChecked expert2d #[batch, 1, cfg.hidden_size]
      denseBranch + moe.post_feedforward_layernorm_2.forward3d expert3d
    | none =>
      mlpOut
  let h4 := residual2 + layer.post_feedforward_layernorm.forward3d ffCombined
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  (applyLayerScalar cfg layer.layer_scalar h5, layerCache')

def forwardPrefillWithMask {batch seq : UInt64}
    (cfg : Config)
    (layer : Gemma4Layer cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (slidingCos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (slidingAttnMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 })
    (layerCache : Gemma4Attention.KVCache cfg batch)
    (perLayerInput : Option (Tensor { shape := #[batch, seq, cfg.hidden_size_per_layer_input] }) := none)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × Gemma4Attention.KVCache cfg batch :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x
  let (attn, layerCache') :=
    layer.self_attn.forwardPrefillWithMask
      cfg
      layer.layerType
      h1
      slidingCos
      slidingSin
      fullCos
      fullSin
      slidingAttnMask
      fullAttnMask
      layerCache
  let h2 := residual1 + layer.post_attention_layernorm.forward3d attn

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let tokens := batch * seq
      let routerInput : Tensor { shape := #[tokens, cfg.hidden_size] } := Tensor.reshapeChecked residual2 #[tokens, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : Tensor { shape := #[tokens, cfg.hidden_size] } :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals topIdx
      let expert3d : Tensor { shape := #[batch, seq, cfg.hidden_size] } := Tensor.reshapeChecked expert2d #[batch, seq, cfg.hidden_size]
      denseBranch + moe.post_feedforward_layernorm_2.forward3d expert3d
    | none =>
      mlpOut
  let h4 := residual2 + layer.post_feedforward_layernorm.forward3d ffCombined
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  (applyLayerScalar cfg layer.layer_scalar h5, layerCache')

end Gemma4Layer

structure Gemma4Model (cfg : Config) where
  embed_tokens : Tensor { shape := #[cfg.vocab_size, cfg.hidden_size] }
  embed_tokens_per_layer :
    Option (Tensor { shape := #[cfg.vocab_size_per_layer_input, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input] }) := none
  per_layer_model_projection :
    Option (Tensor { shape := #[cfg.num_hidden_layers * cfg.hidden_size_per_layer_input, cfg.hidden_size] }) := none
  per_layer_projection_norm : Option (Gemma4RMSNorm cfg.hidden_size_per_layer_input) := none
  layers : Array (Gemma4Layer cfg)
  norm : Gemma4RMSNorm cfg.hidden_size
  deriving TensorStruct

namespace Gemma4Model

private def extractPerLayerInput? {batch seq : UInt64}
    (cfg : Config)
    (perLayerInputs : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }))
    (layerIdx : UInt64)
    : Option (Tensor { shape := #[batch, seq, cfg.hidden_size_per_layer_input] }) :=
  match perLayerInputs with
  | some t =>
    let one : Tensor { shape := #[batch, seq, 1, cfg.hidden_size_per_layer_input] } := sliceTyped t 2 layerIdx 1
    some (Tensor.reshapeChecked one #[batch, seq, cfg.hidden_size_per_layer_input])
  | none => none

def embedTokens {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  Tensor.Preserving.mulScalar (Tensor.embedding inputIds m.embed_tokens) (Float.sqrt cfg.hidden_size.toFloat)

def computePerLayerInputs {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }) :=
  match m.embed_tokens_per_layer, m.per_layer_model_projection, m.per_layer_projection_norm with
  | some tokEmbTable, some projW, some projNorm =>
    let tokEmbFlat : Tensor { shape := #[batch, seq, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input] } :=
      Tensor.Preserving.mulScalar
        (Tensor.embedding inputIds tokEmbTable)
        (Float.sqrt cfg.hidden_size_per_layer_input.toFloat)
    let tokEmb : Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] } :=
      Tensor.reshapeChecked tokEmbFlat #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]
    let projScale := 1.0 / Float.sqrt cfg.hidden_size.toFloat
    let projFlat : Tensor { shape := #[batch, seq, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input] } :=
      Tensor.Preserving.mulScalar (Tensor.linear3d inputsEmbeds projW) projScale
    let proj4d : Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] } :=
      Tensor.reshapeChecked projFlat #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]
    let proj4d := projNorm.forward4d proj4d
    some (Tensor.Preserving.mulScalar (proj4d + tokEmb) (Float.sqrt 0.5))
  | _, _, _ =>
    none

private def precomputeSlidingRotary {seq : UInt64}
    (cfg : Config)
    (device : Device)
    : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } × Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } :=
  Tensor.rotaryFreqs seq cfg.head_dim cfg.sliding_rope_theta device

private def precomputeFullRotary {seq : UInt64}
    (cfg : Config)
    (device : Device)
    : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } × Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } :=
  let fullHalf := Config.fullHeadDim cfg / 2
  let rotHalf := Config.fullRotaryHalfDim cfg
  let (cos0, sin0) := Tensor.rotaryFreqs seq (Config.fullHeadDim cfg) cfg.full_rope_theta device
  let cosRot : Tensor { shape := #[seq, rotHalf], dtype := .Float32 } := sliceTyped cos0 1 0 rotHalf
  let sinRot : Tensor { shape := #[seq, rotHalf], dtype := .Float32 } := sliceTyped sin0 1 0 rotHalf
  let passHalf : UInt64 := fullHalf - rotHalf
  let cosPass : Tensor { shape := #[seq, passHalf], dtype := .Float32 } := onesFloatOn device
  let sinPass : Tensor { shape := #[seq, passHalf], dtype := .Float32 } := zerosFloatOn device
  (typedCat cosRot cosPass 1, typedCat sinRot sinPass 1)

def initCache {batch : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (maxLen : UInt64)
    (device : Device)
    : Gemma4Cache cfg batch :=
  Id.run do
    let mut caches : Array (Gemma4Attention.KVCache cfg batch) := #[]
    for _ in [:m.layers.size] do
      let kv := qwen.QwenAttention.initKVCache
        maxLen
        (batch := batch)
        (num_kv_heads := Config.maxKVHeads cfg)
        (head_dim := Config.maxHeadDim cfg)
        device (.Unknown "dynamic")
      let kv := { kv with
        kStoreDyn := (Tensor.castLike m.embed_tokens kv.kStoreDyn.2).packShape
        vStoreDyn := (Tensor.castLike m.embed_tokens kv.vStoreDyn.2).packShape }
      caches := caches.push kv
    return { attnCaches := caches }

/-- Full-sequence evaluation keeps temporary per-layer K/V when the configured
architecture shares them. This gives uncached evaluation the same layer inputs
as prefill and decode, while ordinary models avoid allocating cache storage. -/
private def forwardEmbedsCore {batch seq : UInt64}
    (cfg : Config) (m : Gemma4Model cfg)
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (perLayerInputs : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }))
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } := Id.run do
  let (slidingCos, slidingSin) := precomputeSlidingRotary cfg inputsEmbeds.device
  let (fullCos, fullSin) := precomputeFullRotary cfg inputsEmbeds.device
  let mut caches := if cfg.num_kv_shared_layers > 0 then
      (initCache cfg m seq inputsEmbeds.device).attnCaches else #[]
  let mut hidden := inputsEmbeds
  for i in [:m.layers.size] do
    match m.layers[i]? with
    | some layer =>
      let perLayer := extractPerLayerInput? cfg perLayerInputs i.toUInt64
      match caches[i]? with
      | some layerCache =>
        let sharedSource := (Config.sharedSourceLayer? cfg i.toUInt64).bind fun src => caches[src.toNat]?
        let (next, nextCache) := Gemma4Layer.forwardWithCache cfg layer hidden
          slidingCos slidingSin fullCos fullSin layerCache sharedSource perLayer
        hidden := next
        if sharedSource.isNone then caches := caches.set! i nextCache
      | none =>
        hidden := Gemma4Layer.forward cfg layer hidden slidingCos slidingSin fullCos fullSin perLayer
    | none => pure ()
  return m.norm.forward3d hidden

def forward {batch seq : UInt64}
    (cfg : Config) (m : Gemma4Model cfg) (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let inputsEmbeds := embedTokens cfg m inputIds
  forwardEmbedsCore cfg m inputsEmbeds (computePerLayerInputs cfg m inputIds inputsEmbeds)

def forwardEmbeds {batch seq : UInt64}
    (cfg : Config) (m : Gemma4Model cfg) (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  forwardEmbedsCore cfg m inputsEmbeds (computePerLayerInputs cfg m inputIds inputsEmbeds)

def forwardEmbedsWithPerLayerInputs {batch seq : UInt64}
    (cfg : Config) (m : Gemma4Model cfg) (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (perLayerInputs : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }))
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let _ := inputIds
  forwardEmbedsCore cfg m inputsEmbeds perLayerInputs

def forwardWithCache {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × Gemma4Cache cfg batch :=
  Id.run do
    let inputsEmbeds := embedTokens cfg m inputIds
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let slidingCos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingCosAll 0 0 seq
    let slidingSin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingSinAll 0 0 seq
    let fullCos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullCosAll 0 0 seq
    let fullSin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i.toUInt64
        let sharedSource :=
          match Config.sharedSourceLayer? cfg i.toUInt64 with
          | some src => cache'.attnCaches[src.toNat]?
          | none => none
        let (hiddenNext, layerCacheNext) :=
          Gemma4Layer.forwardWithCache
            cfg
            layer
            hidden
            slidingCos
            slidingSin
            fullCos
            fullSin
            layerCache
            sharedSource
            perLayer
        hidden := hiddenNext
        if sharedSource.isNone then
          cache' := { cache' with attnCaches := cache'.attnCaches.set! i layerCacheNext }
      | _, _ => pure ()
    return (m.norm.forward3d hidden, cache')

def forwardWithCacheEmbeds {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × Gemma4Cache cfg batch :=
  Id.run do
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let slidingCos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingCosAll 0 0 seq
    let slidingSin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingSinAll 0 0 seq
    let fullCos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullCosAll 0 0 seq
    let fullSin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i.toUInt64
        let sharedSource :=
          match Config.sharedSourceLayer? cfg i.toUInt64 with
          | some src => cache'.attnCaches[src.toNat]?
          | none => none
        let (hiddenNext, layerCacheNext) :=
          Gemma4Layer.forwardWithCache
            cfg
            layer
            hidden
            slidingCos
            slidingSin
            fullCos
            fullSin
            layerCache
            sharedSource
            perLayer
        hidden := hiddenNext
        if sharedSource.isNone then
          cache' := { cache' with attnCaches := cache'.attnCaches.set! i layerCacheNext }
      | _, _ => pure ()
    return (m.norm.forward3d hidden, cache')

def forwardWithCacheEmbedsWithPerLayerInputs {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (perLayerInputs : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }))
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × Gemma4Cache cfg batch :=
  Id.run do
    let _ := inputIds
    let slidingCos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingCosAll 0 0 seq
    let slidingSin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingSinAll 0 0 seq
    let fullCos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullCosAll 0 0 seq
    let fullSin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i.toUInt64
        let sharedSource :=
          match Config.sharedSourceLayer? cfg i.toUInt64 with
          | some src => cache'.attnCaches[src.toNat]?
          | none => none
        let (hiddenNext, layerCacheNext) :=
          Gemma4Layer.forwardWithCache
            cfg
            layer
            hidden
            slidingCos
            slidingSin
            fullCos
            fullSin
            layerCache
            sharedSource
            perLayer
        hidden := hiddenNext
        if sharedSource.isNone then
          cache' := { cache' with attnCaches := cache'.attnCaches.set! i layerCacheNext }
      | _, _ => pure ()
    return (m.norm.forward3d hidden, cache')

def forwardWithCacheEmbedsWithPerLayerInputsBidirectionalVision {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (perLayerInputs : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }))
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (slidingAttnMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × Gemma4Cache cfg batch :=
  Id.run do
    let _ := inputIds
    let slidingCos : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingCosAll 0 0 seq
    let slidingSin : Tensor { shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingSinAll 0 0 seq
    let fullCos : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullCosAll 0 0 seq
    let fullSin : Tensor { shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i.toUInt64
        let (hiddenNext, layerCacheNext) :=
          Gemma4Layer.forwardPrefillWithMask
            cfg
            layer
            hidden
            slidingCos
            slidingSin
            fullCos
            fullSin
            slidingAttnMask
            fullAttnMask
            layerCache
            perLayer
        hidden := hiddenNext
        cache' := { cache' with attnCaches := cache'.attnCaches.set! i layerCacheNext }
      | _, _ => pure ()
    return (m.norm.forward3d hidden, cache')

def forwardStep {batch : UInt64} {maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4Model cfg)
    (inputIds : Tensor { shape := #[batch, 1], dtype := .Int64 })
    (position : UInt64)
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : Tensor { shape := #[batch, 1, cfg.hidden_size] } × Gemma4Cache cfg batch :=
  Id.run do
    let inputsEmbeds := embedTokens cfg m inputIds
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let slidingCos : Tensor { shape := #[1, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingCosAll 0 position 1
    let slidingSin : Tensor { shape := #[1, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingSinAll 0 position 1
    let fullCos : Tensor { shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullCosAll 0 position 1
    let fullSin : Tensor { shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullSinAll 0 position 1
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer :=
          match perLayerInputs with
          | some p =>
            let one : Tensor { shape := #[batch, 1, 1, cfg.hidden_size_per_layer_input] } := sliceTyped p 2 i.toUInt64 1
            some (Tensor.reshapeChecked one #[batch, 1, cfg.hidden_size_per_layer_input])
          | none => none
        let sharedSource :=
          match Config.sharedSourceLayer? cfg i.toUInt64 with
          | some src => cache'.attnCaches[src.toNat]?
          | none => none
        let (hiddenNext, layerCacheNext) :=
          Gemma4Layer.forwardStep
            cfg
            layer
            hidden
            slidingCos
            slidingSin
            fullCos
            fullSin
            layerCache
            sharedSource
            perLayer
        hidden := hiddenNext
        if sharedSource.isNone then
          cache' := { cache' with attnCaches := cache'.attnCaches.set! i layerCacheNext }
      | _, _ => pure ()
    return (m.norm.forward3d hidden, cache')

end Gemma4Model

structure Gemma4ForCausalLM (cfg : Config) where
  model : Gemma4Model cfg
  lmHead : Tensor { shape := #[cfg.vocab_size, cfg.hidden_size] }
  tieWordEmbeddings : Bool := true
  deriving TensorStruct

namespace Gemma4ForCausalLM

def embedTokens {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  m.model.embedTokens cfg inputIds

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    : Tensor { shape := #[batch, seq, cfg.vocab_size] } :=
  let hidden := m.model.forward cfg inputIds
  softcap (Tensor.linear3d hidden m.lmHead) cfg.final_logit_softcapping

def forwardEmbeds {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    : Tensor { shape := #[batch, seq, cfg.vocab_size] } :=
  let hidden := m.model.forwardEmbeds cfg inputIds inputsEmbeds
  softcap (Tensor.linear3d hidden m.lmHead) cfg.final_logit_softcapping

def forwardEmbedsWithPerLayerInputs {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (perLayerInputs : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }))
    : Tensor { shape := #[batch, seq, cfg.vocab_size] } :=
  let hidden := m.model.forwardEmbedsWithPerLayerInputs cfg inputIds inputsEmbeds perLayerInputs
  softcap (Tensor.linear3d hidden m.lmHead) cfg.final_logit_softcapping

def forwardEmbedsWithPerLayerInputsBidirectionalVision {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (perLayerInputs : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }))
    (slidingAttnMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 })
    : Tensor { shape := #[batch, seq, cfg.vocab_size] } :=
  let cache := Gemma4Model.initCache cfg m.model seq inputsEmbeds.device
  let (slidingCosAll, slidingSinAll) := Gemma4Model.precomputeSlidingRotary cfg inputsEmbeds.device
  let (fullCosAll, fullSinAll) := Gemma4Model.precomputeFullRotary cfg inputsEmbeds.device
  let (hidden, _cache') :=
    m.model.forwardWithCacheEmbedsWithPerLayerInputsBidirectionalVision
      (maxLen := seq)
      cfg
      inputIds
      inputsEmbeds
      perLayerInputs
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
      slidingAttnMask
      fullAttnMask
  softcap (Tensor.linear3d hidden m.lmHead) cfg.final_logit_softcapping

private def precomputeDecodeRotary {maxLen : UInt64}
    (cfg : Config)
    (device : Device)
    : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 }
      × Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 }
      × Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 }
      × Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 } :=
  let (slCos, slSin) := Gemma4Model.precomputeSlidingRotary cfg device
  let (fullCos, fullSin) := Gemma4Model.precomputeFullRotary cfg device
  (slCos, slSin, fullCos, fullSin)

private def prefillCaches {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : IO (Tensor { shape := #[batch, cfg.vocab_size] } × Gemma4Cache cfg batch) := do
  let (hidden, cache') :=
    Gemma4Model.forwardWithCache
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let lastHidden : Tensor { shape := #[batch, 1, cfg.hidden_size] } := sliceTyped hidden 1 (seq - 1) 1
  let logits3 : Tensor { shape := #[batch, 1, cfg.vocab_size] } := softcap (Tensor.linear3d lastHidden m.lmHead) cfg.final_logit_softcapping
  pure (Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size], cache')

private def prefillCachesFromEmbeds {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : IO (Tensor { shape := #[batch, cfg.vocab_size] } × Gemma4Cache cfg batch) := do
  let (hidden, cache') :=
    Gemma4Model.forwardWithCacheEmbeds
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      inputsEmbeds
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let lastHidden : Tensor { shape := #[batch, 1, cfg.hidden_size] } := sliceTyped hidden 1 (seq - 1) 1
  let logits3 : Tensor { shape := #[batch, 1, cfg.vocab_size] } := softcap (Tensor.linear3d lastHidden m.lmHead) cfg.final_logit_softcapping
  pure (Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size], cache')

private def prefillCachesFromEmbedsWithPerLayerInputs {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (perLayerInputs : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }))
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : IO (Tensor { shape := #[batch, cfg.vocab_size] } × Gemma4Cache cfg batch) := do
  let (hidden, cache') :=
    Gemma4Model.forwardWithCacheEmbedsWithPerLayerInputs
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      inputsEmbeds
      perLayerInputs
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let lastHidden : Tensor { shape := #[batch, 1, cfg.hidden_size] } := sliceTyped hidden 1 (seq - 1) 1
  let logits3 : Tensor { shape := #[batch, 1, cfg.vocab_size] } := softcap (Tensor.linear3d lastHidden m.lmHead) cfg.final_logit_softcapping
  pure (Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size], cache')

private def prefillCachesFromEmbedsWithPerLayerInputsBidirectionalVision {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (perLayerInputs : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }))
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (slidingAttnMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { shape := #[batch, seq, seq], dtype := .Int64 })
    : IO (Tensor { shape := #[batch, cfg.vocab_size] } × Gemma4Cache cfg batch) := do
  let (hidden, cache') :=
    Gemma4Model.forwardWithCacheEmbedsWithPerLayerInputsBidirectionalVision
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      inputsEmbeds
      perLayerInputs
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
      slidingAttnMask
      fullAttnMask
  let lastHidden : Tensor { shape := #[batch, 1, cfg.hidden_size] } := sliceTyped hidden 1 (seq - 1) 1
  let logits3 : Tensor { shape := #[batch, 1, cfg.vocab_size] } := softcap (Tensor.linear3d lastHidden m.lmHead) cfg.final_logit_softcapping
  pure (Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size], cache')

private def decodeStep {batch maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, 1], dtype := .Int64 })
    (position : UInt64)
    (cache : Gemma4Cache cfg batch)
    (slidingCosAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : IO (Tensor { shape := #[batch, cfg.vocab_size] } × Gemma4Cache cfg batch) := do
  let (hidden, cache') :=
    Gemma4Model.forwardStep
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      position
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let logits3 : Tensor { shape := #[batch, 1, cfg.vocab_size] } := softcap (Tensor.linear3d hidden m.lmHead) cfg.final_logit_softcapping
  pure (Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size], cache')

private partial def decodeLoopCached {batch maxLen : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (finished : Tensor { shape := #[batch], dtype := .Bool })
    (remaining : Nat)
    (cache : Gemma4Cache cfg batch)
    (lastLogits : Tensor { shape := #[batch, cfg.vocab_size] })
    (slidingCosAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (onStep : Option (StreamCallback batch))
    (generatedSoFar : UInt64)
    {curSeq : UInt64}
    (curIds : Tensor { shape := #[batch, curSeq], dtype := .Int64 })
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩

  let nextTokRaw ← sampleFromLogits lastLogits.toFloat32 strategy
  let finished' : Tensor { shape := #[batch], dtype := .Bool } :=
    if eosTokenIds.isEmpty then
      finished
    else
      Tensor.logicalOr finished (tokenInSet nextTokRaw eosTokenIds)
  let nextTok : Tensor { shape := #[batch], dtype := .Int64 } :=
    if eosTokenIds.isEmpty then
      nextTokRaw
    else
      applyFinishedEos nextTokRaw finished (eosTokenIds.getD 0 0)

  match onStep with
  | some cb => cb generatedSoFar nextTok
  | none => pure ()

  let nextCol : Tensor { shape := #[batch, 1], dtype := .Int64 } := Tensor.reshapeChecked nextTok #[batch, 1]
  let appended : Tensor { shape := #[batch, curSeq + 1], dtype := .Int64 } := typedCat curIds nextCol 1

  let stop :=
    if eosTokenIds.isEmpty then
      false
    else
      !(Tensor.any (Tensor.logicalNot finished'))
  if stop then
    return ⟨curSeq + 1, appended⟩
  else
    let (nextLogits, cache') ←
      decodeStep
        cfg
        m
        nextCol
        curSeq
        cache
        slidingCosAll
        slidingSinAll
        fullCosAll
        fullSinAll
    decodeLoopCached
      cfg
      m
      strategy
      eosTokenIds
      finished'
      (remaining - 1)
      cache'
      nextLogits
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
      onStep
      (generatedSoFar + 1)
      appended

private def generateCore {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (onStep : Option (StreamCallback batch) := none)
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
  if seq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let maxLen := seq + maxNewTokens
  let (slidingCosAll, slidingSinAll, fullCosAll, fullSinAll) :=
    precomputeDecodeRotary (maxLen := maxLen) cfg inputIds.device
  let cache := Gemma4Model.initCache cfg m.model maxLen inputIds.device
  let (logits, cache') ←
    prefillCaches
      (maxLen := maxLen)
      cfg m inputIds cache
      slidingCosAll slidingSinAll fullCosAll fullSinAll
  let finished0 : Tensor { shape := #[batch], dtype := .Bool } := falseMaskTyped (n := batch) inputIds.device
  decodeLoopCached
    (maxLen := maxLen)
    cfg
    m
    strategy
    eosTokenIds
    finished0
    maxNewTokens.toNat
    cache'
    logits
    slidingCosAll
    slidingSinAll
    fullCosAll
    fullSinAll
    onStep
    0
    inputIds

private def generateFromEmbedsCore {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (onStep : Option (StreamCallback batch) := none)
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
  if seq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let maxLen := seq + maxNewTokens
  let (slidingCosAll, slidingSinAll, fullCosAll, fullSinAll) :=
    precomputeDecodeRotary (maxLen := maxLen) cfg inputsEmbeds.device
  let cache := Gemma4Model.initCache cfg m.model maxLen inputsEmbeds.device
  let (logits, cache') ←
    prefillCachesFromEmbeds
      (maxLen := maxLen)
      cfg
      m
      inputIds
      inputsEmbeds
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let finished0 : Tensor { shape := #[batch], dtype := .Bool } := falseMaskTyped (n := batch) inputIds.device
  decodeLoopCached
    (maxLen := maxLen)
    cfg
    m
    strategy
    eosTokenIds
    finished0
    maxNewTokens.toNat
    cache'
    logits
    slidingCosAll
    slidingSinAll
    fullCosAll
    fullSinAll
      onStep
      0
      inputIds

private def generateFromEmbedsCoreWithPerLayerInputs {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (perLayerInputs : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }))
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (onStep : Option (StreamCallback batch) := none)
    (prefillMasks : Option (Tensor { shape := #[batch, seq, seq], dtype := .Int64 } × Tensor { shape := #[batch, seq, seq], dtype := .Int64 }) := none)
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
  if seq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let maxLen := seq + maxNewTokens
  let (slidingCosAll, slidingSinAll, fullCosAll, fullSinAll) :=
    precomputeDecodeRotary (maxLen := maxLen) cfg inputsEmbeds.device
  let cache := Gemma4Model.initCache cfg m.model maxLen inputsEmbeds.device
  let (logits, cache') ←
    match prefillMasks with
    | some (slidingAttnMask, fullAttnMask) =>
      prefillCachesFromEmbedsWithPerLayerInputsBidirectionalVision
        (maxLen := maxLen)
        cfg
        m
        inputIds
        inputsEmbeds
        perLayerInputs
        cache
        slidingCosAll
        slidingSinAll
        fullCosAll
        fullSinAll
        slidingAttnMask
        fullAttnMask
    | none =>
      prefillCachesFromEmbedsWithPerLayerInputs
        (maxLen := maxLen)
        cfg
        m
        inputIds
        inputsEmbeds
        perLayerInputs
        cache
        slidingCosAll
        slidingSinAll
        fullCosAll
        fullSinAll
  let finished0 : Tensor { shape := #[batch], dtype := .Bool } := falseMaskTyped (n := batch) inputIds.device
  decodeLoopCached
    (maxLen := maxLen)
    cfg
    m
    strategy
    eosTokenIds
    finished0
    maxNewTokens.toNat
    cache'
    logits
    slidingCosAll
    slidingSinAll
    fullCosAll
    fullSinAll
    onStep
    0
    inputIds

def generate {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateCore cfg m inputIds maxNewTokens strategy eosTokenIds none

def generateFromEmbeds {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds none

def generateFromEmbedsWithPerLayerInputs {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (perLayerInputs : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }))
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (prefillMasks : Option (Tensor { shape := #[batch, seq, seq], dtype := .Int64 } × Tensor { shape := #[batch, seq, seq], dtype := .Int64 }) := none)
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCoreWithPerLayerInputs
    cfg m inputIds inputsEmbeds perLayerInputs maxNewTokens strategy eosTokenIds none prefillMasks

def generateStream {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateCore cfg m inputIds maxNewTokens strategy eosTokenIds (some onStep)

def generateFromEmbedsStream {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds (some onStep)

def generateFromEmbedsStreamWithPerLayerInputs {batch seq : UInt64}
    (cfg : Config)
    (m : Gemma4ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (perLayerInputs : Option (Tensor { shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input] }))
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (prefillMasks : Option (Tensor { shape := #[batch, seq, seq], dtype := .Int64 } × Tensor { shape := #[batch, seq, seq], dtype := .Int64 }) := none)
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCoreWithPerLayerInputs
    cfg m inputIds inputsEmbeds perLayerInputs maxNewTokens strategy eosTokenIds (some onStep) prefillMasks

end Gemma4ForCausalLM

end torch.gemma4
