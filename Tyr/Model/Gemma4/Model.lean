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

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
variable [DType.ConstantSupported dtype device]

open torch
open torch.Model torch.qwen35

private def reqGradFalse {s : Shape} (t : Tensor { device := device, shape := s, dtype := dtype }) : Tensor { device := device, shape := s, dtype := dtype } :=
  Tensor.detach t

/-- Gemma 4 RMSNorm with standard learned scale. -/
structure Gemma4RMSNorm (dim : Nat) where
  weight : SomeDTypeTensor #[dim] device
  eps : Float := 1e-6
  deriving TensorStruct

namespace Gemma4RMSNorm

def fromCheckpointWeight {dim : Nat}
    (weight : SomeDTypeTensor #[dim] device)
    (eps : Float := 1e-6)
    : Gemma4RMSNorm (device := device) dim :=
  { weight := weight, eps }

def forward2d {n dim : Nat}
    (m : Gemma4RMSNorm (device := device) dim)
    (x : Tensor { device := device, shape := #[n, dim], dtype := dtype })
    : Tensor { device := device, shape := #[n, dim], dtype := dtype } :=
  let xf : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := floatDynamic x
  let var : Tensor { device := device, shape := #[n, 1], dtype := .Float32 } := Tensor.meanDim (Tensor.mul xf xf) 1 true
  let inv : Tensor { device := device, shape := #[n, 1], dtype := .Float32 } := rsqrtDynamic (Tensor.addScalar var m.eps)
  let inv : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.expand inv #[n, dim]
  let scale : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.expand (Tensor.reshapeChecked (floatDynamic m.weight.2) #[1, dim]) #[n, dim]
  Tensor.castLike x (Tensor.mul (Tensor.mul xf inv) scale)

def forward3d {batch seq dim : Nat}
    (m : Gemma4RMSNorm (device := device) dim)
    (x : Tensor { device := device, shape := #[batch, seq, dim], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, dim], dtype := dtype } :=
  let flat : Tensor { device := device, shape := #[batch * seq, dim], dtype := dtype } := Tensor.reshapeChecked x #[batch * seq, dim]
  Tensor.reshapeChecked (forward2d m flat) #[batch, seq, dim]

def forward4d {a b c dim : Nat}
    (m : Gemma4RMSNorm (device := device) dim)
    (x : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype })
    : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype } :=
  let flat : Tensor { device := device, shape := #[a * b * c, dim], dtype := dtype } := Tensor.reshapeChecked x #[a * b * c, dim]
  Tensor.reshapeChecked (forward2d m flat) #[a, b, c, dim]

end Gemma4RMSNorm

private def rmsNormNoScale2d {n dim : Nat}
    (x : Tensor { device := device, shape := #[n, dim], dtype := dtype })
    (eps : Float := 1e-6)
    : Tensor { device := device, shape := #[n, dim], dtype := dtype } :=
  let xf : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := floatDynamic x
  let var : Tensor { device := device, shape := #[n, 1], dtype := .Float32 } := Tensor.meanDim (Tensor.mul xf xf) 1 true
  let inv : Tensor { device := device, shape := #[n, 1], dtype := .Float32 } := rsqrtDynamic (Tensor.addScalar var eps)
  let inv : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.expand inv #[n, dim]
  Tensor.castLike x (Tensor.mul xf inv)

private def rmsNormNoScale4d {a b c dim : Nat}
    (x : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype })
    (eps : Float := 1e-6)
    : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype } :=
  let flat : Tensor { device := device, shape := #[a * b * c, dim], dtype := dtype } := Tensor.reshapeChecked x #[a * b * c, dim]
  Tensor.reshapeChecked (rmsNormNoScale2d flat eps) #[a, b, c, dim]

private def rmsNormWeighted4dSlice {a b c dim maxDim : Nat} {weightDType : DType}
    (weight : Tensor { device := device, shape := #[maxDim], dtype := weightDType })
    (x : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype })
    (eps : Float := 1e-6)
    : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype } :=
  let w : Tensor { device := device, shape := #[dim], dtype := weightDType } := sliceTyped weight 0 0 dim
  let xf : Tensor { device := device, shape := #[a * b * c, dim], dtype := dtype } := Tensor.reshapeChecked x #[a * b * c, dim]
  let var : Tensor { device := device, shape := #[a * b * c, 1], dtype := dtype } := Tensor.meanDim (Tensor.mul xf xf) 1 true
  let inv : Tensor { device := device, shape := #[a * b * c, 1], dtype := dtype } := rsqrtDynamic (Tensor.addScalar var eps)
  let inv : Tensor { device := device, shape := #[a * b * c, dim], dtype := dtype } := Tensor.expand inv #[a * b * c, dim]
  let scale : Tensor { device := device, shape := #[a * b * c, dim], dtype := .Float32 } := Tensor.expand (Tensor.reshapeChecked (floatDynamic w) #[1, dim]) #[a * b * c, dim]
  Tensor.castLike x (Tensor.reshapeChecked ((promotedMul (Tensor.mul xf inv) scale).2) #[a, b, c, dim])

private def rmsNormNoScale4dSlice {a b c dim : Nat}
    (x : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype })
    (eps : Float := 1e-6)
    : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype } :=
  rmsNormNoScale4d x eps

/-- Dense Gemma 4 MLP.
    Layers whose checkpoint width is smaller than `Config.maxIntermediateSize`
    are zero-padded during load. -/
structure Gemma4MLP (cfg : Config) (dtype : DType := .Float32) where
  gate_proj : Tensor { device := device, shape := #[Config.maxIntermediateSize cfg, cfg.hidden_size], dtype := dtype }
  up_proj : Tensor { device := device, shape := #[Config.maxIntermediateSize cfg, cfg.hidden_size], dtype := dtype }
  down_proj : Tensor { device := device, shape := #[cfg.hidden_size, Config.maxIntermediateSize cfg], dtype := dtype }
  deriving TensorStruct

namespace Gemma4MLP

def forward {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4MLP (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  Tensor.linear3d (Tensor.mul (Tensor.gelu (Tensor.linear3d x m.gate_proj)) (Tensor.linear3d x m.up_proj)) m.down_proj

end Gemma4MLP

structure Gemma4TextRouter (cfg : Config) where
  proj : SomeDTypeTensor #[cfg.num_experts, cfg.hidden_size] device
  scale : SomeDTypeTensor #[cfg.hidden_size] device
  per_expert_scale : SomeDTypeTensor #[cfg.num_experts] device
  deriving TensorStruct

namespace Gemma4TextRouter

def forward {tokens : Nat}
    (cfg : Config)
    (m : Gemma4TextRouter (device := device) cfg)
    (hidden : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[tokens, cfg.num_experts], dtype := .Float32 }
      × SomeDTypeTensor #[tokens, cfg.top_k_experts] device
      × Tensor { device := device, shape := #[tokens, cfg.top_k_experts], dtype := .Int64 } :=
  let normed : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := rmsNormNoScale2d hidden cfg.rms_norm_eps
  let scaleRoot := 1.0 / Float.sqrt cfg.hidden_size.toFloat
  let scale : Tensor ⟨#[tokens, cfg.hidden_size], .Float32, device⟩ :=
    Tensor.expand (Tensor.reshapeChecked (floatDynamic m.scale.2) #[1, cfg.hidden_size]) #[tokens, cfg.hidden_size]
  let normed : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := .Float32 } :=
    match Tensor.ofRaw (σ := ⟨#[tokens, cfg.hidden_size], .Float32, device⟩)
        (promotedMul normed scale).2.raw with
    | .ok tensor => Tensor.mulScalar tensor scaleRoot
    | .error message => nativeInvariantFailure s!"Gemma4 router computation: {message}"
  let logitsDyn : Tensor { device := device, shape := #[tokens, cfg.num_experts], dtype := .Float32 } := einsumChecked "eh,th->te" (floatDynamic m.proj.2) normed
  let logits : Tensor { device := device, shape := #[tokens, cfg.num_experts], dtype := .Float32 } := Tensor.reshapeChecked logitsDyn #[tokens, cfg.num_experts]
  let probs : Tensor { device := device, shape := #[tokens, cfg.num_experts], dtype := .Float32 } := Tensor.softmax logits (-1)
  let (topValsRaw, topIdx) := topK2d probs cfg.top_k_experts 1
  let denom : Tensor { device := device, shape := #[tokens, 1], dtype := .Float32 } := Tensor.sumDim topValsRaw 1 true
  let denom : Tensor { device := device, shape := #[tokens, cfg.top_k_experts], dtype := .Float32 } := Tensor.expand denom #[tokens, cfg.top_k_experts]
  let topVals : Tensor { device := device, shape := #[tokens, cfg.top_k_experts], dtype := .Float32 } := Tensor.div topValsRaw denom
  let flatIdx : Tensor { device := device, shape := #[tokens * cfg.top_k_experts], dtype := .Int64 } := Tensor.reshapeChecked topIdx #[tokens * cfg.top_k_experts]
  let scalesFlat : Tensor { device := device, shape := #[tokens * cfg.top_k_experts], dtype := m.per_expert_scale.1 } :=
    Tensor.indexSelect m.per_expert_scale.2 flatIdx
  let scales : Tensor { device := device, shape := #[tokens, cfg.top_k_experts], dtype := m.per_expert_scale.1 } := Tensor.reshapeChecked scalesFlat #[tokens, cfg.top_k_experts]
  (probs, promotedMul topVals scales, topIdx)

end Gemma4TextRouter

structure Gemma4TextExperts (cfg : Config) (dtype : DType := .Float32) where
  gate_up_proj : Tensor { device := device, shape := #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size], dtype := dtype }
  down_proj : Tensor { device := device, shape := #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size], dtype := dtype }
  deriving TensorStruct

namespace Gemma4TextExperts

def forward2d {tokens : Nat} {routingDType : DType}
    (cfg : Config)
    (m : Gemma4TextExperts (dtype := dtype) (device := device) cfg)
    (hidden : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype })
    (topVals : Tensor { device := device, shape := #[tokens, cfg.top_k_experts], dtype := routingDType })
    (topIdx : Tensor { device := device, shape := #[tokens, cfg.top_k_experts], dtype := .Int64 })
    : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
  routedTextExpertsForward hidden m.gate_up_proj m.down_proj topVals topIdx

end Gemma4TextExperts

structure Gemma4MoeBranch (cfg : Config) (dtype : DType := .Float32) where
  router : Gemma4TextRouter (device := device) cfg
  experts : Gemma4TextExperts (dtype := dtype) (device := device) cfg
  post_feedforward_layernorm_1 : Gemma4RMSNorm (device := device) cfg.hidden_size
  post_feedforward_layernorm_2 : Gemma4RMSNorm (device := device) cfg.hidden_size
  pre_feedforward_layernorm_2 : Gemma4RMSNorm (device := device) cfg.hidden_size
  deriving TensorStruct

structure Gemma4PerLayerInputBlock (cfg : Config) (dtype : DType := .Float32) where
  per_layer_input_gate : Tensor { device := device, shape := #[cfg.hidden_size_per_layer_input, cfg.hidden_size], dtype := dtype }
  per_layer_projection : Tensor { device := device, shape := #[cfg.hidden_size, cfg.hidden_size_per_layer_input], dtype := dtype }
  post_per_layer_input_norm : Gemma4RMSNorm (device := device) cfg.hidden_size
  deriving TensorStruct

structure Gemma4Attention (cfg : Config) (dtype : DType := .Float32) where
  q_proj : Tensor { device := device, shape := #[Config.maxAttentionDim cfg, cfg.hidden_size], dtype := dtype }
  k_proj : Tensor { device := device, shape := #[Config.maxKVProjDim cfg, cfg.hidden_size], dtype := dtype }
  v_proj : Tensor { device := device, shape := #[Config.maxKVProjDim cfg, cfg.hidden_size], dtype := dtype }
  o_proj : Tensor { device := device, shape := #[cfg.hidden_size, Config.maxAttentionDim cfg], dtype := dtype }
  q_norm : Gemma4RMSNorm (device := device) (Config.maxHeadDim cfg)
  k_norm : Gemma4RMSNorm (device := device) (Config.maxHeadDim cfg)
  deriving TensorStruct

namespace Gemma4Attention

abbrev KVCache (cfg : Config) (batch : Nat) (dtype : DType := .Float32) :=
  qwen.QwenAttention.KVCache (device := device) batch (Config.maxKVHeads cfg) (Config.maxHeadDim cfg) dtype

private def qProjSliding {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } :=
  let qW : Tensor { device := device, shape := #[cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size], dtype := dtype } :=
    sliceTyped m.q_proj 0 0 (cfg.num_attention_heads * cfg.head_dim)
  Tensor.reshapeChecked (Tensor.linear3d x qW) #[batch, seq, cfg.num_attention_heads, cfg.head_dim]

private def qProjFull {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg], dtype := dtype } :=
  let dim := cfg.num_attention_heads * Config.fullHeadDim cfg
  let qW : Tensor { device := device, shape := #[dim, cfg.hidden_size], dtype := dtype } := sliceTyped m.q_proj 0 0 dim
  Tensor.reshapeChecked (Tensor.linear3d x qW) #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg]

private def kProjSliding {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
  let rows := cfg.num_key_value_heads * cfg.head_dim
  let kW : Tensor { device := device, shape := #[rows, cfg.hidden_size], dtype := dtype } := sliceTyped m.k_proj 0 0 rows
  Tensor.reshapeChecked (Tensor.linear3d x kW) #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

private def kProjFull {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg], dtype := dtype } :=
  let rows := Config.fullNumKVHeads cfg * Config.fullHeadDim cfg
  let kW : Tensor { device := device, shape := #[rows, cfg.hidden_size], dtype := dtype } := sliceTyped m.k_proj 0 0 rows
  Tensor.reshapeChecked (Tensor.linear3d x kW) #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg]

private def vProjSliding {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
  let rows := cfg.num_key_value_heads * cfg.head_dim
  let vW : Tensor { device := device, shape := #[rows, cfg.hidden_size], dtype := dtype } := sliceTyped m.v_proj 0 0 rows
  Tensor.reshapeChecked (Tensor.linear3d x vW) #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

private def vProjFull {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg], dtype := dtype } :=
  let rows := Config.fullNumKVHeads cfg * Config.fullHeadDim cfg
  let vW : Tensor { device := device, shape := #[rows, cfg.hidden_size], dtype := dtype } := sliceTyped m.v_proj 0 0 rows
  Tensor.reshapeChecked (Tensor.linear3d x vW) #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg]

private def oProjSliding {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let cols := cfg.num_attention_heads * cfg.head_dim
  let oW : Tensor { device := device, shape := #[cfg.hidden_size, cols], dtype := dtype } := sliceTyped m.o_proj 1 0 cols
  Tensor.linear3d x oW

private def oProjFull {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let cols := cfg.num_attention_heads * Config.fullHeadDim cfg
  let oW : Tensor { device := device, shape := #[cfg.hidden_size, cols], dtype := dtype } := sliceTyped m.o_proj 1 0 cols
  Tensor.linear3d x oW

private def padSlidingKVForCache {batch seq : Nat}
    (cfg : Config)
    (x : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype })
    : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } :=
  let padDim : Nat := Config.maxHeadDim cfg - cfg.head_dim
  let xDim : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, Config.maxHeadDim cfg], dtype := dtype } :=
    typedCat x (Tensor.castLike x (zerosDynamicOn x.device dtype : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, padDim], dtype := dtype })) 3
  let padHeads : Nat := Config.maxKVHeads cfg - cfg.num_key_value_heads
  typedCat xDim (Tensor.castLike x (zerosDynamicOn x.device dtype : Tensor { device := device, shape := #[batch, padHeads, seq, Config.maxHeadDim cfg], dtype := dtype })) 1

private def padFullKVForCache {batch seq : Nat}
    (cfg : Config)
    (x : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg], dtype := dtype })
    : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } :=
  let padDim : Nat := Config.maxHeadDim cfg - Config.fullHeadDim cfg
  let xDim : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } :=
    typedCat x (Tensor.castLike x (zerosDynamicOn x.device dtype : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, seq, padDim], dtype := dtype })) 3
  let padHeads : Nat := Config.maxKVHeads cfg - Config.fullNumKVHeads cfg
  typedCat xDim (Tensor.castLike x (zerosDynamicOn x.device dtype : Tensor { device := device, shape := #[batch, padHeads, seq, Config.maxHeadDim cfg], dtype := dtype })) 1

private def cacheSliceSliding {batch kvLen : Nat}
    (cfg : Config)
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype }
      × Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } :=
  let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let k0 : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, Config.maxHeadDim cfg], dtype := dtype } :=
    sliceTyped (sliceTyped kStore 1 0 cfg.num_key_value_heads) 2 0 kvLen
  let v0 : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, Config.maxHeadDim cfg], dtype := dtype } :=
    sliceTyped (sliceTyped vStore 1 0 cfg.num_key_value_heads) 2 0 kvLen
  let k : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } := sliceTyped k0 3 0 cfg.head_dim
  let v : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } := sliceTyped v0 3 0 cfg.head_dim
  (k, v)

private def cacheSliceFull {batch kvLen : Nat}
    (cfg : Config)
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg], dtype := dtype }
      × Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg], dtype := dtype } :=
  let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let k0 : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.maxHeadDim cfg], dtype := dtype } :=
    sliceTyped (sliceTyped kStore 1 0 (Config.fullNumKVHeads cfg)) 2 0 kvLen
  let v0 : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.maxHeadDim cfg], dtype := dtype } :=
    sliceTyped (sliceTyped vStore 1 0 (Config.fullNumKVHeads cfg)) 2 0 kvLen
  let k : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg], dtype := dtype } :=
    sliceTyped k0 3 0 (Config.fullHeadDim cfg)
  let v : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg], dtype := dtype } :=
    sliceTyped v0 3 0 (Config.fullHeadDim cfg)
  (k, v)

private def attentionScale (d : Nat) : Float :=
  Float.sqrt d.toFloat

private def attentionSlidingPrefillWithMask {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (attnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch :=
  let q0 := qProjSliding cfg m x
  let kRaw := kProjSliding cfg m x
  let vRaw := vProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight.2 q0 m.q_norm.eps
  let k1 := rmsNormWeighted4dSlice m.k_norm.weight.2 kRaw m.k_norm.eps
  let q2 : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } := Tensor.applyRotary q1 cos sin
  let k2 : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } := Tensor.applyRotary k1 cos sin
  let v2 : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
    Tensor.mulScalar (transposeAttention q2) (attentionScale cfg.head_dim)
  let kh : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype } := transposeAttention k2
  let vh : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype } := transposeAttention v2
  let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
    maskedAttention qh kh vh attnMask true
  let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
    Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
  let paddedK : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } := padSlidingKVForCache cfg kh
  let paddedV : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } := padSlidingKVForCache cfg vh
  let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let kStore' := Tensor.sliceScatter kStore 2 0 paddedK
  let vStore' := Tensor.sliceScatter vStore 2 0 paddedV
  let cache' : KVCache (dtype := dtype) (device := device) cfg batch := {
    kStoreDyn := kStore'.packShape
    vStoreDyn := vStore'.packShape
    seq := seq
    maxLen := cache.maxLen
  }
  (oProjSliding cfg m out, cache')

private def attentionFullPrefillWithMask {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (attnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch :=
  let q0 := qProjFull cfg m x
  let kRaw := kProjFull cfg m x
  let vRaw :=
    if cfg.attention_k_eq_v then
      kRaw
    else
      vProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight.2 q0 m.q_norm.eps
  let k1 := rmsNormWeighted4dSlice m.k_norm.weight.2 kRaw m.k_norm.eps
  let q2 : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg], dtype := dtype } := Tensor.applyRotary q1 cos sin
  let k2 : Tensor { device := device, shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg], dtype := dtype } := Tensor.applyRotary k1 cos sin
  let v2 : Tensor { device := device, shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg], dtype := dtype } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg], dtype := dtype } :=
    Tensor.mulScalar (transposeAttention q2) (attentionScale (Config.fullHeadDim cfg))
  let kh : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg], dtype := dtype } := transposeAttention k2
  let vh : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg], dtype := dtype } := transposeAttention v2
  let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg], dtype := dtype } :=
    maskedAttention qh kh vh attnMask true
  let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
  let paddedK : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } := padFullKVForCache cfg kh
  let paddedV : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } := padFullKVForCache cfg vh
  let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
  let kStore' := Tensor.sliceScatter kStore 2 0 paddedK
  let vStore' := Tensor.sliceScatter vStore 2 0 paddedV
  let cache' : KVCache (dtype := dtype) (device := device) cfg batch := {
    kStoreDyn := kStore'.packShape
    vStoreDyn := vStore'.packShape
    seq := seq
    maxLen := cache.maxLen
  }
  (oProjFull cfg m out, cache')

private def attentionSlidingNoCache {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let q0 := qProjSliding cfg m x
  let k0 := kProjSliding cfg m x
  let v0 := vProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight.2 q0 m.q_norm.eps
  let k1 := rmsNormWeighted4dSlice m.k_norm.weight.2 k0 m.k_norm.eps
  let q2 : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } := Tensor.applyRotary q1 cos sin
  let k2 : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } := Tensor.applyRotary k1 cos sin
  let v2 : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } := rmsNormNoScale4dSlice v0 cfg.rms_norm_eps
  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
    Tensor.mulScalar (transposeAttention q2) (attentionScale cfg.head_dim)
  let kh : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype } := transposeAttention k2
  let vh : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype } := transposeAttention v2
  let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
    Tensor.attentionWindow qh kh vh true true cfg.sliding_window
  let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
    Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
  oProjSliding cfg m out

private def attentionFullNoCache {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let q0 := qProjFull cfg m x
  let kRaw := kProjFull cfg m x
  let vRaw :=
    if cfg.attention_k_eq_v then
      kRaw
    else
      vProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight.2 q0 m.q_norm.eps
  let k1 := rmsNormWeighted4dSlice m.k_norm.weight.2 kRaw m.k_norm.eps
  let q2 : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg], dtype := dtype } := Tensor.applyRotary q1 cos sin
  let k2 : Tensor { device := device, shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg], dtype := dtype } := Tensor.applyRotary k1 cos sin
  let v2 : Tensor { device := device, shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg], dtype := dtype } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg], dtype := dtype } :=
    Tensor.mulScalar (transposeAttention q2) (attentionScale (Config.fullHeadDim cfg))
  let kh : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg], dtype := dtype } := transposeAttention k2
  let vh : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg], dtype := dtype } := transposeAttention v2
  let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg], dtype := dtype } :=
    Tensor.attentionGQA qh kh vh true true
  let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype } :=
    Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
  oProjFull cfg m out

private def attentionSlidingWithCache {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    (sharedSource : Option (KVCache (dtype := dtype) (device := device) cfg batch) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch :=
  let q0 := qProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight.2 q0 m.q_norm.eps
  let q2 : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } := Tensor.applyRotary q1 cos sin
  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
    Tensor.mulScalar (transposeAttention q2) (attentionScale cfg.head_dim)
  match sharedSource with
  | some src =>
    let (kAll0, vAll0) := cacheSliceSliding (kvLen := seq) cfg src
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
      Tensor.attentionWindow qh kAll0 vAll0 true true cfg.sliding_window
    let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
    (oProjSliding cfg m out, cache)
  | none =>
    let kRaw := kProjSliding cfg m x
    let vRaw := vProjSliding cfg m x
    let k1 := rmsNormWeighted4dSlice m.k_norm.weight.2 kRaw m.k_norm.eps
    let k2 : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } := Tensor.applyRotary k1 cos sin
    let v2 : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
    let kh : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype } := transposeAttention k2
    let vh : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype } := transposeAttention v2
    let paddedK := padSlidingKVForCache cfg kh
    let paddedV := padSlidingKVForCache cfg vh
    let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
      Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
      Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let kStore' := Tensor.sliceScatter kStore 2 0 paddedK
    let vStore' := Tensor.sliceScatter vStore 2 0 paddedV
    let cache' : KVCache (dtype := dtype) (device := device) cfg batch := {
      kStoreDyn := kStore'.packShape
      vStoreDyn := vStore'.packShape
      seq := seq
      maxLen := cache.maxLen
    }
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
      Tensor.attentionWindow qh kh vh true true cfg.sliding_window
    let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
    (oProjSliding cfg m out, cache')

private def attentionFullWithCache {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    (sharedSource : Option (KVCache (dtype := dtype) (device := device) cfg batch) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch :=
  let q0 := qProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight.2 q0 m.q_norm.eps
  let q2 : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg], dtype := dtype } := Tensor.applyRotary q1 cos sin
  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg], dtype := dtype } :=
    Tensor.mulScalar (transposeAttention q2) (attentionScale (Config.fullHeadDim cfg))
  match sharedSource with
  | some src =>
    let (kAll, vAll) := cacheSliceFull (kvLen := seq) cfg src
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.attentionGQA qh kAll vAll true true
    let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
    (oProjFull cfg m out, cache)
  | none =>
    let kRaw := kProjFull cfg m x
    let vRaw :=
      if cfg.attention_k_eq_v then
        kRaw
      else
        vProjFull cfg m x
    let k1 := rmsNormWeighted4dSlice m.k_norm.weight.2 kRaw m.k_norm.eps
    let k2 : Tensor { device := device, shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg], dtype := dtype } := Tensor.applyRotary k1 cos sin
    let v2 : Tensor { device := device, shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg], dtype := dtype } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
    let kh : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg], dtype := dtype } := transposeAttention k2
    let vh : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg], dtype := dtype } := transposeAttention v2
    let paddedK := padFullKVForCache cfg kh
    let paddedV := padFullKVForCache cfg vh
    let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
      Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
      Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let kStore' := Tensor.sliceScatter kStore 2 0 paddedK
    let vStore' := Tensor.sliceScatter vStore 2 0 paddedV
    let cache' : KVCache (dtype := dtype) (device := device) cfg batch := {
      kStoreDyn := kStore'.packShape
      vStoreDyn := vStore'.packShape
      seq := seq
      maxLen := cache.maxLen
    }
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.attentionGQA qh kh vh true true
    let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
    (oProjFull cfg m out, cache')

private def attentionSlidingStep {batch : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[1, cfg.head_dim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[1, cfg.head_dim / 2], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    (sharedSource : Option (KVCache (dtype := dtype) (device := device) cfg batch) := none)
    : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch :=
  let q0 := qProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight.2 q0 m.q_norm.eps
  let q2 : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } := Tensor.applyRotary q1 cos sin
  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim], dtype := dtype } :=
    Tensor.mulScalar (transposeAttention q2) (attentionScale cfg.head_dim)
  match sharedSource with
  | some src =>
    let kvLen := src.seq
    let useLen := if kvLen > cfg.sliding_window then cfg.sliding_window else kvLen
    let start := kvLen - useLen
    let (kAll0, vAll0) := cacheSliceSliding (kvLen := kvLen) cfg src
    let kAll : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim], dtype := dtype } := sliceTyped kAll0 2 start useLen
    let vAll : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim], dtype := dtype } := sliceTyped vAll0 2 start useLen
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim], dtype := dtype } :=
      Tensor.attentionGQAQKV qh kAll vAll false true
    let out : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, 1, cfg.num_attention_heads * cfg.head_dim]
    (oProjSliding cfg m out, cache)
  | none =>
    let kRaw := kProjSliding cfg m x
    let vRaw := vProjSliding cfg m x
    let k1 := rmsNormWeighted4dSlice m.k_norm.weight.2 kRaw m.k_norm.eps
    let k2 : Tensor { device := device, shape := #[batch, 1, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } := Tensor.applyRotary k1 cos sin
    let v2 : Tensor { device := device, shape := #[batch, 1, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
    let kNew : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, 1, cfg.head_dim], dtype := dtype } := transposeAttention k2
    let vNew : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, 1, cfg.head_dim], dtype := dtype } := transposeAttention v2
    let paddedK : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, 1, Config.maxHeadDim cfg], dtype := dtype } := padSlidingKVForCache cfg kNew
    let paddedV : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, 1, Config.maxHeadDim cfg], dtype := dtype } := padSlidingKVForCache cfg vNew
    let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
      Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
      Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let writePos := if cache.seq < cache.maxLen then cache.seq else cache.maxLen - 1
    let kStore' := Tensor.sliceScatter kStore 2 writePos paddedK
    let vStore' := Tensor.sliceScatter vStore 2 writePos paddedV
    let kvLen := if cache.seq < cache.maxLen then cache.seq + 1 else cache.maxLen
    let useLen := if kvLen > cfg.sliding_window then cfg.sliding_window else kvLen
    let start := kvLen - useLen
    let kAll0 : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } :=
      (cacheSliceSliding (kvLen := kvLen) cfg { cache with kStoreDyn := kStore'.packShape, vStoreDyn := vStore'.packShape, seq := kvLen }) |>.1
    let vAll0 : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } :=
      (cacheSliceSliding (kvLen := kvLen) cfg { cache with kStoreDyn := kStore'.packShape, vStoreDyn := vStore'.packShape, seq := kvLen }) |>.2
    let kAll : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim], dtype := dtype } := sliceTyped kAll0 2 start useLen
    let vAll : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim], dtype := dtype } := sliceTyped vAll0 2 start useLen
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim], dtype := dtype } :=
      Tensor.attentionGQAQKV qh kAll vAll false true
    let out : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, 1, cfg.num_attention_heads * cfg.head_dim]
    let cache' : KVCache (dtype := dtype) (device := device) cfg batch := {
      kStoreDyn := kStore'.packShape
      vStoreDyn := vStore'.packShape
      seq := kvLen
      maxLen := cache.maxLen
    }
    (oProjSliding cfg m out, cache')

private def attentionFullStep {batch : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    (sharedSource : Option (KVCache (dtype := dtype) (device := device) cfg batch) := none)
    : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch :=
  let q0 := qProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight.2 q0 m.q_norm.eps
  let q2 : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads, Config.fullHeadDim cfg], dtype := dtype } := Tensor.applyRotary q1 cos sin
  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, Config.fullHeadDim cfg], dtype := dtype } :=
    Tensor.mulScalar (transposeAttention q2) (attentionScale (Config.fullHeadDim cfg))
  match sharedSource with
  | some src =>
    let kvLen := src.seq
    let (kAll, vAll) := cacheSliceFull (kvLen := kvLen) cfg src
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.attentionGQAQKV qh kAll vAll false true
    let out : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg]
    (oProjFull cfg m out, cache)
  | none =>
    let kRaw := kProjFull cfg m x
    let vRaw :=
      if cfg.attention_k_eq_v then
        kRaw
      else
        vProjFull cfg m x
    let k1 := rmsNormWeighted4dSlice m.k_norm.weight.2 kRaw m.k_norm.eps
    let k2 : Tensor { device := device, shape := #[batch, 1, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg], dtype := dtype } := Tensor.applyRotary k1 cos sin
    let v2 : Tensor { device := device, shape := #[batch, 1, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg], dtype := dtype } := rmsNormNoScale4dSlice vRaw cfg.rms_norm_eps
    let kNew : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, 1, Config.fullHeadDim cfg], dtype := dtype } := transposeAttention k2
    let vNew : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, 1, Config.fullHeadDim cfg], dtype := dtype } := transposeAttention v2
    let paddedK := padFullKVForCache cfg kNew
    let paddedV := padFullKVForCache cfg vNew
    let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
      Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg], dtype := dtype } :=
      Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, Config.maxKVHeads cfg, cache.maxLen, Config.maxHeadDim cfg]
    let writePos := if cache.seq < cache.maxLen then cache.seq else cache.maxLen - 1
    let kStore' := Tensor.sliceScatter kStore 2 writePos paddedK
    let vStore' := Tensor.sliceScatter vStore 2 writePos paddedV
    let kvLen := if cache.seq < cache.maxLen then cache.seq + 1 else cache.maxLen
    let kAll : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg], dtype := dtype } :=
      (cacheSliceFull (kvLen := kvLen) cfg { cache with kStoreDyn := kStore'.packShape, vStoreDyn := vStore'.packShape, seq := kvLen }) |>.1
    let vAll : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg], dtype := dtype } :=
      (cacheSliceFull (kvLen := kvLen) cfg { cache with kStoreDyn := kStore'.packShape, vStoreDyn := vStore'.packShape, seq := kvLen }) |>.2
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.attentionGQAQKV qh kAll vAll false true
    let out : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.reshapeChecked (transposeAttention attn) #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg]
    let cache' : KVCache (dtype := dtype) (device := device) cfg batch := {
      kStoreDyn := kStore'.packShape
      vStoreDyn := vStore'.packShape
      seq := kvLen
      maxLen := cache.maxLen
    }
    (oProjFull cfg m out, cache')

def forward {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (layerType : LayerType)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  match layerType with
  | .slidingAttention => attentionSlidingNoCache cfg m x slidingCos slidingSin
  | .fullAttention => attentionFullNoCache cfg m x fullCos fullSin

def forwardWithCache {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (layerType : LayerType)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    (sharedSource : Option (KVCache (dtype := dtype) (device := device) cfg batch) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch :=
  match layerType with
  | .slidingAttention => attentionSlidingWithCache cfg m x slidingCos slidingSin cache sharedSource
  | .fullAttention => attentionFullWithCache cfg m x fullCos fullSin cache sharedSource

def forwardStep {batch : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (layerType : LayerType)
    (x : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[1, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[1, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    (sharedSource : Option (KVCache (dtype := dtype) (device := device) cfg batch) := none)
    : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch :=
  match layerType with
  | .slidingAttention => attentionSlidingStep cfg m x slidingCos slidingSin cache sharedSource
  | .fullAttention => attentionFullStep cfg m x fullCos fullSin cache sharedSource

def forwardPrefillWithMask {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (layerType : LayerType)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (slidingAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch :=
  match layerType with
  | .slidingAttention =>
    attentionSlidingPrefillWithMask cfg m x slidingCos slidingSin slidingAttnMask cache
  | .fullAttention =>
    attentionFullPrefillWithMask cfg m x fullCos fullSin fullAttnMask cache

end Gemma4Attention

structure Gemma4Layer (cfg : Config) (dtype : DType := .Float32) where
  layerIdx : Nat
  layerType : LayerType
  self_attn : Gemma4Attention (dtype := dtype) (device := device) cfg
  mlp : Gemma4MLP (dtype := dtype) (device := device) cfg
  input_layernorm : Gemma4RMSNorm (device := device) cfg.hidden_size
  post_attention_layernorm : Gemma4RMSNorm (device := device) cfg.hidden_size
  pre_feedforward_layernorm : Gemma4RMSNorm (device := device) cfg.hidden_size
  post_feedforward_layernorm : Gemma4RMSNorm (device := device) cfg.hidden_size
  per_layer_input : Option (Gemma4PerLayerInputBlock (dtype := dtype) (device := device) cfg) := none
  moe : Option (Gemma4MoeBranch (dtype := dtype) (device := device) cfg) := none
  layer_scalar : Tensor { device := device, shape := #[1], dtype := dtype }
  deriving TensorStruct

structure Gemma4Cache (cfg : Config) (batch : Nat) (dtype : DType := .Float32) where
  attnCaches : Array (Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch)

namespace Gemma4Layer

private def applyLayerScalar {batch seq : Nat}
    (cfg : Config)
    (scale : Tensor { device := device, shape := #[1], dtype := dtype })
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let s : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := Tensor.expand (Tensor.reshapeChecked scale #[1, 1, 1]) #[batch, seq, cfg.hidden_size]
  Tensor.mul x s

private def applyPerLayerInput {batch seq : Nat}
    (cfg : Config)
    (block : Gemma4PerLayerInputBlock (dtype := dtype) (device := device) cfg)
    (hidden : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInput : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size_per_layer_input], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let residual := hidden
  let gated : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size_per_layer_input], dtype := dtype } :=
    Tensor.gelu (Tensor.linear3d hidden block.per_layer_input_gate)
  let mixed := Tensor.mul gated perLayerInput
  let projected : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := Tensor.linear3d mixed block.per_layer_projection
  let projected := block.post_per_layer_input_norm.forward3d projected
  Tensor.add residual projected

def forward {batch seq : Nat}
    (cfg : Config)
    (layer : Gemma4Layer (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (perLayerInput : Option (Tensor { device := device, shape := #[batch, seq, cfg.hidden_size_per_layer_input], dtype := dtype }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x
  let attn := layer.self_attn.forward cfg layer.layerType h1 slidingCos slidingSin fullCos fullSin
  let h2 := Tensor.add residual1 (layer.post_attention_layernorm.forward3d attn)

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let tokens := batch * seq
      let routerInput : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.reshapeChecked residual2 #[tokens, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals.2 topIdx
      let expert3d : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := Tensor.reshapeChecked expert2d #[batch, seq, cfg.hidden_size]
      Tensor.add denseBranch (moe.post_feedforward_layernorm_2.forward3d expert3d)
    | none =>
      mlpOut
  let h4 := Tensor.add residual2 (layer.post_feedforward_layernorm.forward3d ffCombined)
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  applyLayerScalar cfg layer.layer_scalar h5

def forwardWithCache {batch seq : Nat}
    (cfg : Config)
    (layer : Gemma4Layer (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (layerCache : Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch)
    (sharedSource : Option (Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch) := none)
    (perLayerInput : Option (Tensor { device := device, shape := #[batch, seq, cfg.hidden_size_per_layer_input], dtype := dtype }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch :=
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
  let h2 := Tensor.add residual1 (layer.post_attention_layernorm.forward3d attn)

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let tokens := batch * seq
      let routerInput : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.reshapeChecked residual2 #[tokens, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals.2 topIdx
      let expert3d : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := Tensor.reshapeChecked expert2d #[batch, seq, cfg.hidden_size]
      Tensor.add denseBranch (moe.post_feedforward_layernorm_2.forward3d expert3d)
    | none =>
      mlpOut
  let h4 := Tensor.add residual2 (layer.post_feedforward_layernorm.forward3d ffCombined)
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  (applyLayerScalar cfg layer.layer_scalar h5, layerCache')

def forwardStep {batch : Nat}
    (cfg : Config)
    (layer : Gemma4Layer (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[1, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[1, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (layerCache : Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch)
    (sharedSource : Option (Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch) := none)
    (perLayerInput : Option (Tensor { device := device, shape := #[batch, 1, cfg.hidden_size_per_layer_input], dtype := dtype }) := none)
    : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } × Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch :=
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
  let h2 := Tensor.add residual1 (layer.post_attention_layernorm.forward3d attn)

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let routerInput : Tensor { device := device, shape := #[batch, cfg.hidden_size], dtype := dtype } := Tensor.reshapeChecked residual2 #[batch, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : Tensor { device := device, shape := #[batch, cfg.hidden_size], dtype := dtype } :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals.2 topIdx
      let expert3d : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := Tensor.reshapeChecked expert2d #[batch, 1, cfg.hidden_size]
      Tensor.add denseBranch (moe.post_feedforward_layernorm_2.forward3d expert3d)
    | none =>
      mlpOut
  let h4 := Tensor.add residual2 (layer.post_feedforward_layernorm.forward3d ffCombined)
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  (applyLayerScalar cfg layer.layer_scalar h5, layerCache')

def forwardPrefillWithMask {batch seq : Nat}
    (cfg : Config)
    (layer : Gemma4Layer (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (slidingAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (layerCache : Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch)
    (perLayerInput : Option (Tensor { device := device, shape := #[batch, seq, cfg.hidden_size_per_layer_input], dtype := dtype }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch :=
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
  let h2 := Tensor.add residual1 (layer.post_attention_layernorm.forward3d attn)

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let tokens := batch * seq
      let routerInput : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.reshapeChecked residual2 #[tokens, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals.2 topIdx
      let expert3d : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := Tensor.reshapeChecked expert2d #[batch, seq, cfg.hidden_size]
      Tensor.add denseBranch (moe.post_feedforward_layernorm_2.forward3d expert3d)
    | none =>
      mlpOut
  let h4 := Tensor.add residual2 (layer.post_feedforward_layernorm.forward3d ffCombined)
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  (applyLayerScalar cfg layer.layer_scalar h5, layerCache')

end Gemma4Layer

structure Gemma4Model (cfg : Config) (dtype : DType := .Float32) where
  embed_tokens : Tensor { device := device, shape := #[cfg.vocab_size, cfg.hidden_size], dtype := dtype }
  embed_tokens_per_layer :
    Option (Tensor { device := device, shape := #[cfg.vocab_size_per_layer_input, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input], dtype := dtype }) := none
  per_layer_model_projection :
    Option (Tensor { device := device, shape := #[cfg.num_hidden_layers * cfg.hidden_size_per_layer_input, cfg.hidden_size], dtype := dtype }) := none
  per_layer_projection_norm : Option (Gemma4RMSNorm (device := device) cfg.hidden_size_per_layer_input) := none
  layers : Array (Gemma4Layer (dtype := dtype) (device := device) cfg)
  norm : Gemma4RMSNorm (device := device) cfg.hidden_size
  deriving TensorStruct

namespace Gemma4Model

private def extractPerLayerInput? {batch seq : Nat}
    (cfg : Config)
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (layerIdx : Nat)
    : Option (Tensor { device := device, shape := #[batch, seq, cfg.hidden_size_per_layer_input], dtype := dtype }) :=
  match perLayerInputs with
  | some t =>
    let one : Tensor { device := device, shape := #[batch, seq, 1, cfg.hidden_size_per_layer_input], dtype := dtype } := sliceTyped t 2 layerIdx 1
    some (Tensor.reshapeChecked one #[batch, seq, cfg.hidden_size_per_layer_input])
  | none => none

def embedTokens {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  Tensor.mulScalar (Tensor.embedding inputIds m.embed_tokens) (Float.sqrt cfg.hidden_size.toFloat)

def computePerLayerInputs {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }) :=
  match m.embed_tokens_per_layer, m.per_layer_model_projection, m.per_layer_projection_norm with
  | some tokEmbTable, some projW, some projNorm =>
    let tokEmbFlat : Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input], dtype := dtype } :=
      Tensor.mulScalar
        (Tensor.embedding inputIds tokEmbTable)
        (Float.sqrt cfg.hidden_size_per_layer_input.toFloat)
    let tokEmb : Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype } :=
      Tensor.reshapeChecked tokEmbFlat #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]
    let projScale := 1.0 / Float.sqrt cfg.hidden_size.toFloat
    let projFlat : Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input], dtype := dtype } :=
      Tensor.mulScalar (Tensor.linear3d inputsEmbeds projW) projScale
    let proj4d : Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype } :=
      Tensor.reshapeChecked projFlat #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]
    let proj4d := projNorm.forward4d proj4d
    some (Tensor.mulScalar (Tensor.add proj4d tokEmb) (Float.sqrt 0.5))
  | _, _, _ =>
    none

private def precomputeSlidingRotary {seq : Nat}
    (cfg : Config)
    (device : Device)
    : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } × Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } :=
  Tensor.rotaryFreqs seq cfg.head_dim cfg.sliding_rope_theta device

private def precomputeFullRotary {seq : Nat}
    (cfg : Config)
    (device : Device)
    : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } × Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } :=
  let fullHalf := Config.fullHeadDim cfg / 2
  let rotHalf := Config.fullRotaryHalfDim cfg
  let (cos0, sin0) := Tensor.rotaryFreqs seq (Config.fullHeadDim cfg) cfg.full_rope_theta device
  let cosRot : Tensor { device := device, shape := #[seq, rotHalf], dtype := .Float32 } := sliceTyped cos0 1 0 rotHalf
  let sinRot : Tensor { device := device, shape := #[seq, rotHalf], dtype := .Float32 } := sliceTyped sin0 1 0 rotHalf
  let passHalf : Nat := fullHalf - rotHalf
  let cosPass : Tensor { device := device, shape := #[seq, passHalf], dtype := .Float32 } := onesFloatOn device
  let sinPass : Tensor { device := device, shape := #[seq, passHalf], dtype := .Float32 } := zerosFloatOn device
  (typedCat cosRot cosPass 1, typedCat sinRot sinPass 1)

def initCache {batch : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (maxLen : Nat)
    (cacheDevice : Device)
    [DType.ConstantSupported dtype cacheDevice]
    : Gemma4Cache (dtype := dtype) (device := cacheDevice) cfg batch :=
  Id.run do
    let mut caches : Array (Gemma4Attention.KVCache (dtype := dtype) (device := cacheDevice) cfg batch) := #[]
    for _ in [:m.layers.size] do
      let kv := qwen.QwenAttention.initKVCache
        maxLen
        (batch := batch)
        (num_kv_heads := Config.maxKVHeads cfg)
        (head_dim := Config.maxHeadDim cfg)
        cacheDevice dtype
      let kv := { kv with
        kStoreDyn := (Tensor.castLike m.embed_tokens kv.kStoreDyn.2).packShape
        vStoreDyn := (Tensor.castLike m.embed_tokens kv.vStoreDyn.2).packShape }
      caches := caches.push kv
    return { attnCaches := caches }

/-- Full-sequence evaluation keeps temporary per-layer K/V when the configured
architecture shares them. This gives uncached evaluation the same layer inputs
as prefill and decode, while ordinary models avoid allocating cache storage. -/
private def forwardEmbedsCore {batch seq : Nat}
    (cfg : Config) (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := Id.run do
  let (slidingCos, slidingSin) := precomputeSlidingRotary cfg inputsEmbeds.device
  let (fullCos, fullSin) := precomputeFullRotary cfg inputsEmbeds.device
  let mut caches := if cfg.num_kv_shared_layers > 0 then
      (initCache cfg m seq inputsEmbeds.device).attnCaches else #[]
  let mut hidden := inputsEmbeds
  for i in [:m.layers.size] do
    match m.layers[i]? with
    | some layer =>
      let perLayer := extractPerLayerInput? cfg perLayerInputs i
      match caches[i]? with
      | some layerCache =>
        let sharedSource := (Config.sharedSourceLayer? cfg i).bind fun src => caches[src]?
        let (next, nextCache) := Gemma4Layer.forwardWithCache cfg layer hidden
          slidingCos slidingSin fullCos fullSin layerCache sharedSource perLayer
        hidden := next
        if sharedSource.isNone then caches := caches.set! i nextCache
      | none =>
        hidden := Gemma4Layer.forward cfg layer hidden slidingCos slidingSin fullCos fullSin perLayer
    | none => pure ()
  return m.norm.forward3d hidden

def forward {batch seq : Nat}
    (cfg : Config) (m : Gemma4Model (dtype := dtype) (device := device) cfg) (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let inputsEmbeds := embedTokens cfg m inputIds
  forwardEmbedsCore cfg m inputsEmbeds (computePerLayerInputs cfg m inputIds inputsEmbeds)

def forwardEmbeds {batch seq : Nat}
    (cfg : Config) (m : Gemma4Model (dtype := dtype) (device := device) cfg) (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  forwardEmbedsCore cfg m inputsEmbeds (computePerLayerInputs cfg m inputIds inputsEmbeds)

def forwardEmbedsWithPerLayerInputs {batch seq : Nat}
    (cfg : Config) (m : Gemma4Model (dtype := dtype) (device := device) cfg) (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let _ := inputIds
  forwardEmbedsCore cfg m inputsEmbeds perLayerInputs

def forwardWithCache {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch :=
  Id.run do
    let inputsEmbeds := embedTokens cfg m inputIds
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let slidingCos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingCosAll 0 0 seq
    let slidingSin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingSinAll 0 0 seq
    let fullCos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullCosAll 0 0 seq
    let fullSin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i
        let sharedSource :=
          match Config.sharedSourceLayer? cfg i with
          | some src => cache'.attnCaches[src]?
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

def forwardWithCacheEmbeds {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch :=
  Id.run do
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let slidingCos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingCosAll 0 0 seq
    let slidingSin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingSinAll 0 0 seq
    let fullCos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullCosAll 0 0 seq
    let fullSin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i
        let sharedSource :=
          match Config.sharedSourceLayer? cfg i with
          | some src => cache'.attnCaches[src]?
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

def forwardWithCacheEmbedsWithPerLayerInputs {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch :=
  Id.run do
    let _ := inputIds
    let slidingCos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingCosAll 0 0 seq
    let slidingSin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingSinAll 0 0 seq
    let fullCos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullCosAll 0 0 seq
    let fullSin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i
        let sharedSource :=
          match Config.sharedSourceLayer? cfg i with
          | some src => cache'.attnCaches[src]?
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

def forwardWithCacheEmbedsWithPerLayerInputsBidirectionalVision {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (slidingAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch :=
  Id.run do
    let _ := inputIds
    let slidingCos : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingCosAll 0 0 seq
    let slidingSin : Tensor { device := device, shape := #[seq, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingSinAll 0 0 seq
    let fullCos : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullCosAll 0 0 seq
    let fullSin : Tensor { device := device, shape := #[seq, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer := extractPerLayerInput? cfg perLayerInputs i
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

def forwardStep {batch : Nat} {maxLen : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 })
    (position : Nat)
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch :=
  Id.run do
    let inputsEmbeds := embedTokens cfg m inputIds
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let slidingCos : Tensor { device := device, shape := #[1, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingCosAll 0 position 1
    let slidingSin : Tensor { device := device, shape := #[1, cfg.head_dim / 2], dtype := .Float32 } := sliceTyped slidingSinAll 0 position 1
    let fullCos : Tensor { device := device, shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullCosAll 0 position 1
    let fullSin : Tensor { device := device, shape := #[1, Config.fullHeadDim cfg / 2], dtype := .Float32 } := sliceTyped fullSinAll 0 position 1
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for i in [:m.layers.size] do
      match m.layers[i]?, cache'.attnCaches[i]? with
      | some layer, some layerCache =>
        let perLayer :=
          match perLayerInputs with
          | some p =>
            let one : Tensor { device := device, shape := #[batch, 1, 1, cfg.hidden_size_per_layer_input], dtype := dtype } := sliceTyped p 2 i 1
            some (Tensor.reshapeChecked one #[batch, 1, cfg.hidden_size_per_layer_input])
          | none => none
        let sharedSource :=
          match Config.sharedSourceLayer? cfg i with
          | some src => cache'.attnCaches[src]?
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

structure Gemma4ForCausalLM (cfg : Config) (dtype : DType := .Float32) where
  model : Gemma4Model (dtype := dtype) (device := device) cfg
  lmHead : Tensor { device := device, shape := #[cfg.vocab_size, cfg.hidden_size], dtype := dtype }
  tieWordEmbeddings : Bool := true
  deriving TensorStruct

namespace Gemma4ForCausalLM

def embedTokens {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  m.model.embedTokens cfg inputIds

def forward {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := dtype } :=
  let hidden := m.model.forward cfg inputIds
  softcap (Tensor.linear3d hidden m.lmHead) cfg.final_logit_softcapping

def forwardEmbeds {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := dtype } :=
  let hidden := m.model.forwardEmbeds cfg inputIds inputsEmbeds
  softcap (Tensor.linear3d hidden m.lmHead) cfg.final_logit_softcapping

def forwardEmbedsWithPerLayerInputs {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := dtype } :=
  let hidden := m.model.forwardEmbedsWithPerLayerInputs cfg inputIds inputsEmbeds perLayerInputs
  softcap (Tensor.linear3d hidden m.lmHead) cfg.final_logit_softcapping

def forwardEmbedsWithPerLayerInputsBidirectionalVision {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (slidingAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := dtype } :=
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

private def precomputeDecodeRotary {maxLen : Nat}
    (cfg : Config)
    (device : Device)
    : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 }
      × Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 }
      × Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 }
      × Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 } :=
  let (slCos, slSin) := Gemma4Model.precomputeSlidingRotary cfg device
  let (fullCos, fullSin) := Gemma4Model.precomputeFullRotary cfg device
  (slCos, slSin, fullCos, fullSin)

private def prefillCaches {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch) := do
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
  let lastHidden : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := sliceTyped hidden 1 (seq - 1) 1
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := softcap (Tensor.linear3d lastHidden m.lmHead) cfg.final_logit_softcapping
  pure (Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size], cache')

private def prefillCachesFromEmbeds {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch) := do
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
  let lastHidden : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := sliceTyped hidden 1 (seq - 1) 1
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := softcap (Tensor.linear3d lastHidden m.lmHead) cfg.final_logit_softcapping
  pure (Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size], cache')

private def prefillCachesFromEmbedsWithPerLayerInputs {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch) := do
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
  let lastHidden : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := sliceTyped hidden 1 (seq - 1) 1
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := softcap (Tensor.linear3d lastHidden m.lmHead) cfg.final_logit_softcapping
  pure (Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size], cache')

private def prefillCachesFromEmbedsWithPerLayerInputsBidirectionalVision {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (slidingAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch) := do
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
  let lastHidden : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := sliceTyped hidden 1 (seq - 1) 1
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := softcap (Tensor.linear3d lastHidden m.lmHead) cfg.final_logit_softcapping
  pure (Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size], cache')

private def decodeStep {batch maxLen : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 })
    (position : Nat)
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch) := do
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
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := softcap (Tensor.linear3d hidden m.lmHead) cfg.final_logit_softcapping
  pure (Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size], cache')

private partial def decodeLoopCached {batch maxLen : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (finished : Tensor { device := device, shape := #[batch], dtype := .Bool })
    (remaining : Nat)
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch)
    (lastLogits : Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype })
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.head_dim / 2], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHeadDim cfg / 2], dtype := .Float32 })
    (onStep : Option (StreamCallback (device := device) batch))
    (generatedSoFar : Nat)
    {curSeq : Nat}
    (curIds : Tensor { device := device, shape := #[batch, curSeq], dtype := .Int64 })
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩

  let nextTokRaw ← sampleFromLogits lastLogits.toFloat32 strategy
  let finished' : Tensor { device := device, shape := #[batch], dtype := .Bool } :=
    if eosTokenIds.isEmpty then
      finished
    else
      Tensor.logicalOr finished (tokenInSet nextTokRaw eosTokenIds)
  let nextTok : Tensor { device := device, shape := #[batch], dtype := .Int64 } :=
    if eosTokenIds.isEmpty then
      nextTokRaw
    else
      applyFinishedEos nextTokRaw finished (eosTokenIds.getD 0 0)

  match onStep with
  | some cb => cb generatedSoFar nextTok
  | none => pure ()

  let nextCol : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 } := Tensor.reshapeChecked nextTok #[batch, 1]
  let appended : Tensor { device := device, shape := #[batch, curSeq + 1], dtype := .Int64 } := typedCat curIds nextCol 1

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

private def generateCore {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (onStep : Option (StreamCallback (device := device) batch) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
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
  let finished0 : Tensor { device := device, shape := #[batch], dtype := .Bool } := falseMaskTyped (n := batch) inputIds.device
  decodeLoopCached
    (maxLen := maxLen)
    cfg
    m
    strategy
    eosTokenIds
    finished0
    maxNewTokens
    cache'
    logits
    slidingCosAll
    slidingSinAll
    fullCosAll
    fullSinAll
    onStep
    0
    inputIds

private def generateFromEmbedsCore {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (onStep : Option (StreamCallback (device := device) batch) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
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
  let finished0 : Tensor { device := device, shape := #[batch], dtype := .Bool } := falseMaskTyped (n := batch) inputIds.device
  decodeLoopCached
    (maxLen := maxLen)
    cfg
    m
    strategy
    eosTokenIds
    finished0
    maxNewTokens
    cache'
    logits
    slidingCosAll
    slidingSinAll
    fullCosAll
    fullSinAll
      onStep
      0
      inputIds

private def generateFromEmbedsCoreWithPerLayerInputs {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (onStep : Option (StreamCallback (device := device) batch) := none)
    (prefillMasks : Option (Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 } × Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 }) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
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
  let finished0 : Tensor { device := device, shape := #[batch], dtype := .Bool } := falseMaskTyped (n := batch) inputIds.device
  decodeLoopCached
    (maxLen := maxLen)
    cfg
    m
    strategy
    eosTokenIds
    finished0
    maxNewTokens
    cache'
    logits
    slidingCosAll
    slidingSinAll
    fullCosAll
    fullSinAll
    onStep
    0
    inputIds

def generate {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateCore cfg m inputIds maxNewTokens strategy eosTokenIds none

def generateFromEmbeds {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds none

def generateFromEmbedsWithPerLayerInputs {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (prefillMasks : Option (Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 } × Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 }) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCoreWithPerLayerInputs
    cfg m inputIds inputsEmbeds perLayerInputs maxNewTokens strategy eosTokenIds none prefillMasks

def generateStream {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (onStep : StreamCallback (device := device) batch)
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateCore cfg m inputIds maxNewTokens strategy eosTokenIds (some onStep)

def generateFromEmbedsStream {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (onStep : StreamCallback (device := device) batch)
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds (some onStep)

def generateFromEmbedsStreamWithPerLayerInputs {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (onStep : StreamCallback (device := device) batch)
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    (prefillMasks : Option (Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 } × Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 }) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCoreWithPerLayerInputs
    cfg m inputIds inputsEmbeds perLayerInputs maxNewTokens strategy eosTokenIds (some onStep) prefillMasks

end Gemma4ForCausalLM

end torch.gemma4
