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
  let inv : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.broadcastTo inv #[n, dim]
  let scale : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.broadcastTo (Tensor.reshape (floatDynamic m.weight.2) #[1, dim]) #[n, dim]
  Tensor.castLike x (Tensor.mul (Tensor.mul xf inv) scale)

def forward3d {batch seq dim : Nat}
    (m : Gemma4RMSNorm (device := device) dim)
    (x : Tensor { device := device, shape := #[batch, seq, dim], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, dim], dtype := dtype } :=
  let flat : Tensor { device := device, shape := #[batch * seq, dim], dtype := dtype } := Tensor.reshape x #[batch * seq, dim]
  Tensor.reshape (forward2d m flat) #[batch, seq, dim]

def forward4d {a b c dim : Nat}
    (m : Gemma4RMSNorm (device := device) dim)
    (x : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype })
    : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype } :=
  let flat : Tensor { device := device, shape := #[a * b * c, dim], dtype := dtype } := Tensor.reshape x #[a * b * c, dim]
  Tensor.reshape (forward2d m flat) #[a, b, c, dim]

end Gemma4RMSNorm

/-- Final logits, softcapped as `cap * tanh (logits / cap)` when the config sets a cap. -/
private def capLogits {σ : StaticSpec} (cfg : Config) (logits : Tensor σ) : Tensor σ :=
  match cfg.final_logit_softcapping with
  | some cap => softcap logits cap
  | none => logits

private def rmsNormNoScale2d {n dim : Nat}
    (x : Tensor { device := device, shape := #[n, dim], dtype := dtype })
    (eps : Float := 1e-6)
    : Tensor { device := device, shape := #[n, dim], dtype := dtype } :=
  let xf : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := floatDynamic x
  let var : Tensor { device := device, shape := #[n, 1], dtype := .Float32 } := Tensor.meanDim (Tensor.mul xf xf) 1 true
  let inv : Tensor { device := device, shape := #[n, 1], dtype := .Float32 } := rsqrtDynamic (Tensor.addScalar var eps)
  let inv : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.broadcastTo inv #[n, dim]
  Tensor.castLike x (Tensor.mul xf inv)

private def rmsNormNoScale4d {a b c dim : Nat}
    (x : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype })
    (eps : Float := 1e-6)
    : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype } :=
  let flat : Tensor { device := device, shape := #[a * b * c, dim], dtype := dtype } := Tensor.reshape x #[a * b * c, dim]
  Tensor.reshape (rmsNormNoScale2d flat eps) #[a, b, c, dim]

private def rmsNormWeighted4dSlice {a b c dim maxDim : Nat} {weightDType : DType}
    (weight : Tensor { device := device, shape := #[maxDim], dtype := weightDType })
    (x : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype })
    (eps : Float := 1e-6)
    (hDim : dim ≤ maxDim := by first | exact Config.headDim_le_max _ | exact Config.fullHeadDim_le_max _)
    : Tensor { device := device, shape := #[a, b, c, dim], dtype := dtype } :=
  let w : Tensor { device := device, shape := #[dim], dtype := weightDType } := Tensor.slice weight 0 0 dim
  let xf : Tensor { device := device, shape := #[a * b * c, dim], dtype := dtype } := Tensor.reshape x #[a * b * c, dim]
  let var : Tensor { device := device, shape := #[a * b * c, 1], dtype := dtype } := Tensor.meanDim (Tensor.mul xf xf) 1 true
  let inv : Tensor { device := device, shape := #[a * b * c, 1], dtype := dtype } := rsqrtDynamic (Tensor.addScalar var eps)
  let inv : Tensor { device := device, shape := #[a * b * c, dim], dtype := dtype } := Tensor.broadcastTo inv #[a * b * c, dim]
  let scale : Tensor { device := device, shape := #[a * b * c, dim], dtype := .Float32 } := Tensor.broadcastTo (Tensor.reshape (floatDynamic w) #[1, dim]) #[a * b * c, dim]
  Tensor.castLike x (Tensor.reshape (Tensor.mul (floatDynamic (Tensor.mul xf inv)) scale) #[a, b, c, dim])

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
  proj : Tensor { device := device, shape := #[cfg.num_experts, cfg.hidden_size], dtype := .Float32 }
  scale : Tensor { device := device, shape := #[cfg.hidden_size], dtype := .Float32 }
  per_expert_scale : Tensor { device := device, shape := #[cfg.num_experts], dtype := .Float32 }
  deriving TensorStruct

namespace Gemma4TextRouter

def forward {tokens : Nat}
    (cfg : Config)
    (m : Gemma4TextRouter (device := device) cfg)
    (hidden : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[tokens, cfg.num_experts], dtype := .Float32 }
      × Tensor { device := device, shape := #[tokens, cfg.top_k_experts], dtype := .Float32 }
      × Tensor { device := device, shape := #[tokens, cfg.top_k_experts], dtype := .Int64 } :=
  let normed : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := rmsNormNoScale2d hidden cfg.rms_norm_eps
  let scaleRoot := 1.0 / Float.sqrt cfg.hidden_size.toFloat
  let scale : Tensor ⟨#[tokens, cfg.hidden_size], .Float32, device⟩ :=
    Tensor.broadcastTo (Tensor.reshape m.scale #[1, cfg.hidden_size]) #[tokens, cfg.hidden_size]
  let normed : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := .Float32 } :=
    Tensor.mulScalar (Tensor.mul (floatDynamic normed) scale) scaleRoot
  let probs : Tensor { device := device, shape := #[tokens, cfg.num_experts], dtype := .Float32 } :=
    Tensor.softmax (Tensor.linear normed m.proj)
  let (topValsRaw, topIdx) := Tensor.topKLast2d probs cfg.top_k_experts cfg.topKValid
  let denom : Tensor { device := device, shape := #[tokens, 1], dtype := .Float32 } := Tensor.sumDim topValsRaw 1 true
  let denom : Tensor { device := device, shape := #[tokens, cfg.top_k_experts], dtype := .Float32 } := Tensor.broadcastTo denom #[tokens, cfg.top_k_experts]
  let topVals : Tensor { device := device, shape := #[tokens, cfg.top_k_experts], dtype := .Float32 } := Tensor.div topValsRaw denom
  let flatIdx : Tensor { device := device, shape := #[tokens * cfg.top_k_experts], dtype := .Int64 } := Tensor.reshape topIdx #[tokens * cfg.top_k_experts]
  let scalesFlat : Tensor { device := device, shape := #[tokens * cfg.top_k_experts], dtype := .Float32 } :=
    Tensor.indexSelect! m.per_expert_scale flatIdx
  let scales : Tensor { device := device, shape := #[tokens, cfg.top_k_experts], dtype := .Float32 } := Tensor.reshape scalesFlat #[tokens, cfg.top_k_experts]
  (probs, Tensor.mul topVals scales, topIdx)

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

abbrev KVCache (cfg : Config) (batch cap : Nat) (dtype : DType := .Float32) :=
  qwen.QwenAttention.KVCache (device := device) batch (Config.maxKVHeads cfg) (Config.maxHeadDim cfg) cap dtype

private def qProjSliding {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } :=
  let qW : Tensor { device := device, shape := #[cfg.num_attention_heads * cfg.head_dim, cfg.hidden_size], dtype := dtype } :=
    Tensor.slice m.q_proj 0 0 (cfg.num_attention_heads * cfg.head_dim) (_bounds := by simpa using cfg.qSliding_le)
  Tensor.reshape (Tensor.linear3d x qW) #[batch, seq, cfg.num_attention_heads, cfg.head_dim]

private def qProjFull {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg], dtype := dtype } :=
  let dim := cfg.num_attention_heads * Config.fullHeadDim cfg
  let qW : Tensor { device := device, shape := #[dim, cfg.hidden_size], dtype := dtype } := Tensor.slice m.q_proj 0 0 dim (_bounds := by simpa [dim] using cfg.qFull_le)
  Tensor.reshape (Tensor.linear3d x qW) #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg]

private def kProjSliding {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
  let rows := cfg.num_key_value_heads * cfg.head_dim
  let kW : Tensor { device := device, shape := #[rows, cfg.hidden_size], dtype := dtype } :=
    Tensor.slice m.k_proj 0 0 rows (_bounds := by simpa [rows] using cfg.kvSliding_le)
  Tensor.reshape (Tensor.linear3d x kW) #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

private def kProjFull {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg], dtype := dtype } :=
  let rows := Config.fullNumKVHeads cfg * Config.fullHeadDim cfg
  let kW : Tensor { device := device, shape := #[rows, cfg.hidden_size], dtype := dtype } :=
    Tensor.slice m.k_proj 0 0 rows (_bounds := by simpa [rows] using cfg.kvFull_le)
  Tensor.reshape (Tensor.linear3d x kW) #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg]

private def vProjSliding {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
  let rows := cfg.num_key_value_heads * cfg.head_dim
  let vW : Tensor { device := device, shape := #[rows, cfg.hidden_size], dtype := dtype } :=
    Tensor.slice m.v_proj 0 0 rows (_bounds := by simpa [rows] using cfg.kvSliding_le)
  Tensor.reshape (Tensor.linear3d x vW) #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

private def vProjFull {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg], dtype := dtype } :=
  let rows := Config.fullNumKVHeads cfg * Config.fullHeadDim cfg
  let vW : Tensor { device := device, shape := #[rows, cfg.hidden_size], dtype := dtype } :=
    Tensor.slice m.v_proj 0 0 rows (_bounds := by simpa [rows] using cfg.kvFull_le)
  Tensor.reshape (Tensor.linear3d x vW) #[batch, seq, Config.fullNumKVHeads cfg, Config.fullHeadDim cfg]

private def oProjSliding {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let cols := cfg.num_attention_heads * cfg.head_dim
  let oW : Tensor { device := device, shape := #[cfg.hidden_size, cols], dtype := dtype } :=
    Tensor.slice m.o_proj 1 0 cols (_bounds := by simpa [cols] using cfg.qSliding_le)
  Tensor.linear3d x oW

private def oProjFull {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let cols := cfg.num_attention_heads * Config.fullHeadDim cfg
  let oW : Tensor { device := device, shape := #[cfg.hidden_size, cols], dtype := dtype } :=
    Tensor.slice m.o_proj 1 0 cols (_bounds := by simpa [cols] using cfg.qFull_le)
  Tensor.linear3d x oW

private def padSlidingKVForCache {batch seq : Nat}
    (cfg : Config)
    (x : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype })
    : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } :=
  have hDim : cfg.head_dim ≤ Config.maxHeadDim cfg := by
    simp only [Config.maxHeadDim, Gemma4Config.maxHeadDim]; split <;> omega
  have hHeads : cfg.num_key_value_heads ≤ Config.maxKVHeads cfg := by
    simp only [Config.maxKVHeads, Gemma4Config.maxKVHeads]; split <;> omega
  let padDim : Nat := Config.maxHeadDim cfg - cfg.head_dim
  let xDim : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, Config.maxHeadDim cfg], dtype := dtype } :=
    (Tensor.cat x (Tensor.castLike x (zerosDynamicOn x.device dtype : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, padDim], dtype := dtype })) 3).reshape
      #[batch, cfg.num_key_value_heads, seq, Config.maxHeadDim cfg] (by simp [padDim, Nat.add_sub_cancel' hDim])
  let padHeads : Nat := Config.maxKVHeads cfg - cfg.num_key_value_heads
  (Tensor.cat xDim (Tensor.castLike x (zerosDynamicOn x.device dtype : Tensor { device := device, shape := #[batch, padHeads, seq, Config.maxHeadDim cfg], dtype := dtype })) 1).reshape
    #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] (by simp [padHeads, Nat.add_sub_cancel' hHeads])

private def padFullKVForCache {batch seq : Nat}
    (cfg : Config)
    (x : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.fullHeadDim cfg], dtype := dtype })
    : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } :=
  have hDim : Config.fullHeadDim cfg ≤ Config.maxHeadDim cfg := by
    simp only [Config.maxHeadDim, Gemma4Config.maxHeadDim, Config.fullHeadDim]; split <;> omega
  have hHeads : Config.fullNumKVHeads cfg ≤ Config.maxKVHeads cfg := by
    simp only [Config.maxKVHeads, Gemma4Config.maxKVHeads, Config.fullNumKVHeads]; split <;> omega
  let padDim : Nat := Config.maxHeadDim cfg - Config.fullHeadDim cfg
  let xDim : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } :=
    (Tensor.cat x (Tensor.castLike x (zerosDynamicOn x.device dtype : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, seq, padDim], dtype := dtype })) 3).reshape
      #[batch, Config.fullNumKVHeads cfg, seq, Config.maxHeadDim cfg] (by simp [padDim, Nat.add_sub_cancel' hDim])
  let padHeads : Nat := Config.maxKVHeads cfg - Config.fullNumKVHeads cfg
  (Tensor.cat xDim (Tensor.castLike x (zerosDynamicOn x.device dtype : Tensor { device := device, shape := #[batch, padHeads, seq, Config.maxHeadDim cfg], dtype := dtype })) 1).reshape
    #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg] (by simp [padHeads, Nat.add_sub_cancel' hHeads])

private def cacheSliceSliding {batch kvLen cap : Nat}
    (cfg : Config)
    (cache : KVCache (dtype := dtype) (device := device) cfg batch cap)
    (h : kvLen ≤ cap)
    : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype }
      × Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } :=
  let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
    cache.kStore
  let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
    cache.vStore
  let k0 : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, Config.maxHeadDim cfg], dtype := dtype } :=
    (kStore.slice 1 0 cfg.num_key_value_heads (_bounds := by simpa using Config.kvHeads_le_max cfg)).slice 2 0 kvLen
  let v0 : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, Config.maxHeadDim cfg], dtype := dtype } :=
    (vStore.slice 1 0 cfg.num_key_value_heads (_bounds := by simpa using Config.kvHeads_le_max cfg)).slice 2 0 kvLen
  let k : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } := k0.slice 3 0 cfg.head_dim (_bounds := by simpa using Config.headDim_le_max cfg)
  let v : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } := v0.slice 3 0 cfg.head_dim (_bounds := by simpa using Config.headDim_le_max cfg)
  (k, v)

private def cacheSliceFull {batch kvLen cap : Nat}
    (cfg : Config)
    (cache : KVCache (dtype := dtype) (device := device) cfg batch cap)
    (h : kvLen ≤ cap)
    : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg], dtype := dtype }
      × Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg], dtype := dtype } :=
  let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
    cache.kStore
  let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
    cache.vStore
  let k0 : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.maxHeadDim cfg], dtype := dtype } :=
    (kStore.slice 1 0 (Config.fullNumKVHeads cfg) (_bounds := by simpa using Config.fullKVHeads_le_max cfg)).slice 2 0 kvLen
  let v0 : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.maxHeadDim cfg], dtype := dtype } :=
    (vStore.slice 1 0 (Config.fullNumKVHeads cfg) (_bounds := by simpa using Config.fullKVHeads_le_max cfg)).slice 2 0 kvLen
  let k : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg], dtype := dtype } :=
    k0.slice 3 0 (Config.fullHeadDim cfg) (_bounds := by simpa using Config.fullHeadDim_le_max cfg)
  let v : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg], dtype := dtype } :=
    v0.slice 3 0 (Config.fullHeadDim cfg) (_bounds := by simpa using Config.fullHeadDim_le_max cfg)
  (k, v)

private def attentionScale (d : Nat) : Float :=
  Float.sqrt d.toFloat

private def attentionSlidingPrefillWithMask {batch seq cap : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (attnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch cap)
    (hfit : seq ≤ cap)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch cap :=
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
    maskedAttention qh kh vh attnMask true (_heads := .ofDvd cfg.slidingKVHeadsDivide)
  let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
    Tensor.reshape (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
  let paddedK : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } := padSlidingKVForCache cfg kh
  let paddedV : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } := padSlidingKVForCache cfg vh
  let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
    cache.kStore
  let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
    cache.vStore
  let kStore' := Tensor.sliceScatter kStore 2 0 paddedK
  let vStore' := Tensor.sliceScatter vStore 2 0 paddedV
  let cache' : KVCache (dtype := dtype) (device := device) cfg batch cap := {
    cap_pos := cache.cap_pos
    kStore := kStore'
    vStore := vStore'
  }
  (oProjSliding cfg m out, cache')

private def attentionFullPrefillWithMask {batch seq cap : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (attnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch cap)
    (hfit : seq ≤ cap)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch cap :=
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
    maskedAttention qh kh vh attnMask true (_heads := .ofDvd cfg.fullKVHeadsDivide)
  let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype } :=
    Tensor.reshape (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
  let paddedK : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } := padFullKVForCache cfg kh
  let paddedV : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, seq, Config.maxHeadDim cfg], dtype := dtype } := padFullKVForCache cfg vh
  let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
    cache.kStore
  let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
    cache.vStore
  let kStore' := Tensor.sliceScatter kStore 2 0 paddedK
  let vStore' := Tensor.sliceScatter vStore 2 0 paddedV
  let cache' : KVCache (dtype := dtype) (device := device) cfg batch cap := {
    cap_pos := cache.cap_pos
    kStore := kStore'
    vStore := vStore'
  }
  (oProjFull cfg m out, cache')

private def attentionSlidingNoCache {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
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
    Tensor.attentionWindow qh kh vh true true cfg.sliding_window (_heads := .ofDvd cfg.slidingKVHeadsDivide)
  let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
    Tensor.reshape (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
  oProjSliding cfg m out

private def attentionFullNoCache {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
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
    Tensor.attentionGQA qh kh vh true true (_heads := .ofDvd cfg.fullKVHeadsDivide)
  let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype } :=
    Tensor.reshape (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
  oProjFull cfg m out

private def attentionSlidingWithCache {batch seq cap : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch cap)
    (hfit : seq ≤ cap)
    (sharedSource : Option (KVCache (dtype := dtype) (device := device) cfg batch cap) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch cap :=
  let q0 := qProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight.2 q0 m.q_norm.eps
  let q2 : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } := Tensor.applyRotary q1 cos sin
  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
    Tensor.mulScalar (transposeAttention q2) (attentionScale cfg.head_dim)
  match sharedSource with
  | some src =>
    let (kAll0, vAll0) := cacheSliceSliding (kvLen := seq) cfg src hfit
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
      Tensor.attentionWindow qh kAll0 vAll0 true true cfg.sliding_window (_heads := .ofDvd cfg.slidingKVHeadsDivide)
    let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
      Tensor.reshape (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
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
    let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
      cache.kStore
    let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
      cache.vStore
    let kStore' := Tensor.sliceScatter kStore 2 0 paddedK
    let vStore' := Tensor.sliceScatter vStore 2 0 paddedV
    let cache' : KVCache (dtype := dtype) (device := device) cfg batch cap := {
      cap_pos := cache.cap_pos
      kStore := kStore'
      vStore := vStore'
    }
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
      Tensor.attentionWindow qh kh vh true true cfg.sliding_window (_heads := .ofDvd cfg.slidingKVHeadsDivide)
    let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
      Tensor.reshape (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
    (oProjSliding cfg m out, cache')

private def attentionFullWithCache {batch seq cap : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch cap)
    (hfit : seq ≤ cap)
    (sharedSource : Option (KVCache (dtype := dtype) (device := device) cfg batch cap) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch cap :=
  let q0 := qProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight.2 q0 m.q_norm.eps
  let q2 : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, Config.fullHeadDim cfg], dtype := dtype } := Tensor.applyRotary q1 cos sin
  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg], dtype := dtype } :=
    Tensor.mulScalar (transposeAttention q2) (attentionScale (Config.fullHeadDim cfg))
  match sharedSource with
  | some src =>
    let (kAll, vAll) := cacheSliceFull (kvLen := seq) cfg src hfit
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.attentionGQA qh kAll vAll true true (_heads := .ofDvd cfg.fullKVHeadsDivide)
    let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.reshape (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
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
    let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
      cache.kStore
    let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
      cache.vStore
    let kStore' := Tensor.sliceScatter kStore 2 0 paddedK
    let vStore' := Tensor.sliceScatter vStore 2 0 paddedV
    let cache' : KVCache (dtype := dtype) (device := device) cfg batch cap := {
      cap_pos := cache.cap_pos
      kStore := kStore'
      vStore := vStore'
    }
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.attentionGQA qh kh vh true true (_heads := .ofDvd cfg.fullKVHeadsDivide)
    let out : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.reshape (transposeAttention attn) #[batch, seq, cfg.num_attention_heads * Config.fullHeadDim cfg]
    (oProjFull cfg m out, cache')

private def attentionSlidingStep {batch cap : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[1, cfg.half_head_dim], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[1, cfg.half_head_dim], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch cap)
    (pos : Nat) (hpos : pos < cap)
    (sharedSource : Option (KVCache (dtype := dtype) (device := device) cfg batch cap) := none)
    : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch cap :=
  let q0 := qProjSliding cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight.2 q0 m.q_norm.eps
  let q2 : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } := Tensor.applyRotary q1 cos sin
  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim], dtype := dtype } :=
    Tensor.mulScalar (transposeAttention q2) (attentionScale cfg.head_dim)
  match sharedSource with
  | some src =>
    -- The source layer is earlier in this step, so it already holds `pos + 1` entries.
    let kvLen := pos + 1
    let useLen := if kvLen > cfg.sliding_window then cfg.sliding_window else kvLen
    let start := kvLen - useLen
    have _hUse : useLen ≤ kvLen := by simp only [useLen]; split <;> omega
    let (kAll0, vAll0) := cacheSliceSliding (kvLen := kvLen) cfg src hpos
    let kAll : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim], dtype := dtype } := kAll0.slice 2 start useLen
    let vAll : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim], dtype := dtype } := vAll0.slice 2 start useLen
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim], dtype := dtype } :=
      Tensor.attentionGQAQKV qh kAll vAll false true (_heads := .ofDvd cfg.slidingKVHeadsDivide)
    let out : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
      Tensor.reshape (transposeAttention attn) #[batch, 1, cfg.num_attention_heads * cfg.head_dim]
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
    let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
      cache.kStore
    let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
      cache.vStore
    let kStore' := Tensor.sliceScatter kStore 2 pos paddedK
    let vStore' := Tensor.sliceScatter vStore 2 pos paddedV
    let kvLen := pos + 1
    let useLen := if kvLen > cfg.sliding_window then cfg.sliding_window else kvLen
    let start := kvLen - useLen
    have _hUse : useLen ≤ kvLen := by simp only [useLen]; split <;> omega
    let kAll0 : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } :=
      (cacheSliceSliding (kvLen := kvLen) cfg { cache with kStore := kStore', vStore := vStore' } hpos) |>.1
    let vAll0 : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } :=
      (cacheSliceSliding (kvLen := kvLen) cfg { cache with kStore := kStore', vStore := vStore' } hpos) |>.2
    let kAll : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim], dtype := dtype } := kAll0.slice 2 start useLen
    let vAll : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, useLen, cfg.head_dim], dtype := dtype } := vAll0.slice 2 start useLen
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim], dtype := dtype } :=
      Tensor.attentionGQAQKV qh kAll vAll false true (_heads := .ofDvd cfg.slidingKVHeadsDivide)
    let out : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
      Tensor.reshape (transposeAttention attn) #[batch, 1, cfg.num_attention_heads * cfg.head_dim]
    let cache' : KVCache (dtype := dtype) (device := device) cfg batch cap := {
      cap_pos := cache.cap_pos
      kStore := kStore'
      vStore := vStore'
    }
    (oProjSliding cfg m out, cache')

private def attentionFullStep {batch cap : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[1, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[1, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch cap)
    (pos : Nat) (hpos : pos < cap)
    (sharedSource : Option (KVCache (dtype := dtype) (device := device) cfg batch cap) := none)
    : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch cap :=
  let q0 := qProjFull cfg m x
  let q1 := rmsNormWeighted4dSlice m.q_norm.weight.2 q0 m.q_norm.eps
  let q2 : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads, Config.fullHeadDim cfg], dtype := dtype } := Tensor.applyRotary q1 cos sin
  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, Config.fullHeadDim cfg], dtype := dtype } :=
    Tensor.mulScalar (transposeAttention q2) (attentionScale (Config.fullHeadDim cfg))
  match sharedSource with
  | some src =>
    -- The source layer is earlier in this step, so it already holds `pos + 1` entries.
    let kvLen := pos + 1
    let (kAll, vAll) := cacheSliceFull (kvLen := kvLen) cfg src hpos
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.attentionGQAQKV qh kAll vAll false true (_heads := .ofDvd cfg.fullKVHeadsDivide)
    let out : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.reshape (transposeAttention attn) #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg]
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
    let kStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
      cache.kStore
    let vStore : Tensor { device := device, shape := #[batch, Config.maxKVHeads cfg, cap, Config.maxHeadDim cfg], dtype := dtype } :=
      cache.vStore
    let kStore' := Tensor.sliceScatter kStore 2 pos paddedK
    let vStore' := Tensor.sliceScatter vStore 2 pos paddedV
    let kvLen := pos + 1
    let kAll : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg], dtype := dtype } :=
      (cacheSliceFull (kvLen := kvLen) cfg { cache with kStore := kStore', vStore := vStore' } hpos) |>.1
    let vAll : Tensor { device := device, shape := #[batch, Config.fullNumKVHeads cfg, kvLen, Config.fullHeadDim cfg], dtype := dtype } :=
      (cacheSliceFull (kvLen := kvLen) cfg { cache with kStore := kStore', vStore := vStore' } hpos) |>.2
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.attentionGQAQKV qh kAll vAll false true (_heads := .ofDvd cfg.fullKVHeadsDivide)
    let out : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg], dtype := dtype } :=
      Tensor.reshape (transposeAttention attn) #[batch, 1, cfg.num_attention_heads * Config.fullHeadDim cfg]
    let cache' : KVCache (dtype := dtype) (device := device) cfg batch cap := {
      cap_pos := cache.cap_pos
      kStore := kStore'
      vStore := vStore'
    }
    (oProjFull cfg m out, cache')

def forward {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (layerType : LayerType)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  match layerType with
  | .slidingAttention => attentionSlidingNoCache cfg m x slidingCos slidingSin
  | .fullAttention => attentionFullNoCache cfg m x fullCos fullSin

def forwardWithCache {batch seq cap : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (layerType : LayerType)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch cap)
    (hfit : seq ≤ cap)
    (sharedSource : Option (KVCache (dtype := dtype) (device := device) cfg batch cap) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch cap :=
  match layerType with
  | .slidingAttention => attentionSlidingWithCache cfg m x slidingCos slidingSin cache hfit sharedSource
  | .fullAttention => attentionFullWithCache cfg m x fullCos fullSin cache hfit sharedSource

def forwardStep {batch cap : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (layerType : LayerType)
    (x : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[1, cfg.half_head_dim], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[1, cfg.half_head_dim], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[1, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[1, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch cap)
    (pos : Nat) (hpos : pos < cap)
    (sharedSource : Option (KVCache (dtype := dtype) (device := device) cfg batch cap) := none)
    : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch cap :=
  match layerType with
  | .slidingAttention => attentionSlidingStep cfg m x slidingCos slidingSin cache pos hpos sharedSource
  | .fullAttention => attentionFullStep cfg m x fullCos fullSin cache pos hpos sharedSource

def forwardPrefillWithMask {batch seq cap : Nat}
    (cfg : Config)
    (m : Gemma4Attention (dtype := dtype) (device := device) cfg)
    (layerType : LayerType)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (slidingAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch cap)
    (hfit : seq ≤ cap)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch cap :=
  match layerType with
  | .slidingAttention =>
    attentionSlidingPrefillWithMask cfg m x slidingCos slidingSin slidingAttnMask cache hfit
  | .fullAttention =>
    attentionFullPrefillWithMask cfg m x fullCos fullSin fullAttnMask cache hfit

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

structure Gemma4Cache (cfg : Config) (batch cap : Nat) (dtype : DType := .Float32) where
  attnCaches : Vector cfg.num_hidden_layers (Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch cap)

namespace Gemma4Layer

private def applyLayerScalar {batch seq : Nat}
    (cfg : Config)
    (scale : Tensor { device := device, shape := #[1], dtype := dtype })
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let s : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := Tensor.broadcastTo (Tensor.reshape scale #[1, 1, 1]) #[batch, seq, cfg.hidden_size]
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
    (slidingCos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
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
      let routerInput : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.reshape residual2 #[tokens, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals topIdx
      let expert3d : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := Tensor.reshape expert2d #[batch, seq, cfg.hidden_size]
      Tensor.add denseBranch (moe.post_feedforward_layernorm_2.forward3d expert3d)
    | none =>
      mlpOut
  let h4 := Tensor.add residual2 (layer.post_feedforward_layernorm.forward3d ffCombined)
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  applyLayerScalar cfg layer.layer_scalar h5

def forwardWithCache {batch seq cap : Nat}
    (cfg : Config)
    (layer : Gemma4Layer (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (layerCache : Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch cap)
    (hfit : seq ≤ cap)
    (sharedSource : Option (Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch cap) := none)
    (perLayerInput : Option (Tensor { device := device, shape := #[batch, seq, cfg.hidden_size_per_layer_input], dtype := dtype }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch cap :=
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
      hfit
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
      let routerInput : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.reshape residual2 #[tokens, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals topIdx
      let expert3d : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := Tensor.reshape expert2d #[batch, seq, cfg.hidden_size]
      Tensor.add denseBranch (moe.post_feedforward_layernorm_2.forward3d expert3d)
    | none =>
      mlpOut
  let h4 := Tensor.add residual2 (layer.post_feedforward_layernorm.forward3d ffCombined)
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  (applyLayerScalar cfg layer.layer_scalar h5, layerCache')

def forwardStep {batch cap : Nat}
    (cfg : Config)
    (layer : Gemma4Layer (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[1, cfg.half_head_dim], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[1, cfg.half_head_dim], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[1, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[1, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (layerCache : Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch cap)
    (pos : Nat) (hpos : pos < cap)
    (sharedSource : Option (Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch cap) := none)
    (perLayerInput : Option (Tensor { device := device, shape := #[batch, 1, cfg.hidden_size_per_layer_input], dtype := dtype }) := none)
    : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } × Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch cap :=
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
      pos
      hpos
      sharedSource
  let h2 := Tensor.add residual1 (layer.post_attention_layernorm.forward3d attn)

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let routerInput : Tensor { device := device, shape := #[batch, cfg.hidden_size], dtype := dtype } := Tensor.reshape residual2 #[batch, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : Tensor { device := device, shape := #[batch, cfg.hidden_size], dtype := dtype } :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals topIdx
      let expert3d : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := Tensor.reshape expert2d #[batch, 1, cfg.hidden_size]
      Tensor.add denseBranch (moe.post_feedforward_layernorm_2.forward3d expert3d)
    | none =>
      mlpOut
  let h4 := Tensor.add residual2 (layer.post_feedforward_layernorm.forward3d ffCombined)
  let h5 :=
    match layer.per_layer_input, perLayerInput with
    | some block, some p => applyPerLayerInput cfg block h4 p
    | _, _ => h4
  (applyLayerScalar cfg layer.layer_scalar h5, layerCache')

def forwardPrefillWithMask {batch seq cap : Nat}
    (cfg : Config)
    (layer : Gemma4Layer (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (slidingCos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (slidingSin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 })
    (fullCos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (slidingAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (layerCache : Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch cap)
    (hfit : seq ≤ cap)
    (perLayerInput : Option (Tensor { device := device, shape := #[batch, seq, cfg.hidden_size_per_layer_input], dtype := dtype }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Gemma4Attention.KVCache (dtype := dtype) (device := device) cfg batch cap :=
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
      hfit
  let h2 := Tensor.add residual1 (layer.post_attention_layernorm.forward3d attn)

  let residual2 := h2
  let h3 := layer.pre_feedforward_layernorm.forward3d h2
  let mlpOut := layer.mlp.forward cfg h3
  let ffCombined :=
    match layer.moe with
    | some moe =>
      let denseBranch := moe.post_feedforward_layernorm_1.forward3d mlpOut
      let tokens := batch * seq
      let routerInput : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.reshape residual2 #[tokens, cfg.hidden_size]
      let (_routerProbs, topVals, topIdx) := moe.router.forward cfg routerInput
      let expertInput2d : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
        moe.pre_feedforward_layernorm_2.forward2d routerInput
      let expert2d := moe.experts.forward2d cfg expertInput2d topVals topIdx
      let expert3d : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := Tensor.reshape expert2d #[batch, seq, cfg.hidden_size]
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
  layers : Vector cfg.num_hidden_layers (Gemma4Layer (dtype := dtype) (device := device) cfg)
  norm : Gemma4RMSNorm (device := device) cfg.hidden_size
  deriving TensorStruct

namespace Gemma4Model

private def extractPerLayerInput? {batch seq : Nat}
    (cfg : Config)
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (layerIdx : Fin cfg.num_hidden_layers)
    : Option (Tensor { device := device, shape := #[batch, seq, cfg.hidden_size_per_layer_input], dtype := dtype }) :=
  match perLayerInputs with
  | some t =>
    let one : Tensor { device := device, shape := #[batch, seq, 1, cfg.hidden_size_per_layer_input], dtype := dtype } := Tensor.slice t 2 layerIdx 1
    some (Tensor.reshape one #[batch, seq, cfg.hidden_size_per_layer_input])
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
      Tensor.reshape tokEmbFlat #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]
    let projScale := 1.0 / Float.sqrt cfg.hidden_size.toFloat
    let projFlat : Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers * cfg.hidden_size_per_layer_input], dtype := dtype } :=
      Tensor.mulScalar (Tensor.linear3d inputsEmbeds projW) projScale
    let proj4d : Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype } :=
      Tensor.reshape projFlat #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input]
    let proj4d := projNorm.forward4d proj4d
    some (Tensor.mulScalar (Tensor.add proj4d tokEmb) (Float.sqrt 0.5))
  | _, _, _ =>
    none

private def precomputeSlidingRotary {seq : Nat}
    (cfg : Config)
    (device : Device)
    : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 } × Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 } :=
  Tensor.rotaryFreqs seq cfg.half_head_dim cfg.sliding_rope_theta device

private def precomputeFullRotary {seq : Nat}
    (cfg : Config)
    (device : Device)
    : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 } × Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 } :=
  let fullHalf := Config.fullHalfHeadDim cfg
  let rotHalf := Config.fullRotaryHalfDim cfg
  have hRot : rotHalf ≤ fullHalf := cfg.fullRotaryValid.2
  let (cos0, sin0) := Tensor.rotaryFreqs seq (Config.fullHalfHeadDim cfg) cfg.full_rope_theta device
  let cosRot : Tensor { device := device, shape := #[seq, rotHalf], dtype := .Float32 } := Tensor.slice cos0 1 0 rotHalf
  let sinRot : Tensor { device := device, shape := #[seq, rotHalf], dtype := .Float32 } := Tensor.slice sin0 1 0 rotHalf
  let passHalf : Nat := fullHalf - rotHalf
  let cosPass : Tensor { device := device, shape := #[seq, passHalf], dtype := .Float32 } := onesFloatOn device
  let sinPass : Tensor { device := device, shape := #[seq, passHalf], dtype := .Float32 } := zerosFloatOn device
  ((Tensor.cat cosRot cosPass 1).reshape #[seq, fullHalf] (by simp [passHalf, Nat.add_sub_cancel' hRot]),
   (Tensor.cat sinRot sinPass 1).reshape #[seq, fullHalf] (by simp [passHalf, Nat.add_sub_cancel' hRot]))

def initCache {batch : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (maxLen : Nat)
    (cacheDevice : Device)
    [DType.ConstantSupported dtype cacheDevice]
    (h : 0 < maxLen := by omega)
    : Gemma4Cache (dtype := dtype) (device := cacheDevice) cfg batch maxLen :=
  let layers : Vector cfg.num_hidden_layers Nat := ⟨Array.range cfg.num_hidden_layers, Array.size_range⟩
  { attnCaches := layers.map fun _ =>
      let kv := qwen.QwenAttention.initKVCache
        maxLen
        (batch := batch)
        (num_kv_heads := Config.maxKVHeads cfg)
        (head_dim := Config.maxHeadDim cfg)
        cacheDevice dtype (h := h)
      { kv with
        kStore := Tensor.castLike m.embed_tokens kv.kStore
        vStore := Tensor.castLike m.embed_tokens kv.vStore } }

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
  -- An empty sequence has no K/V to share, and runs every layer uncached.
  let mut caches := if h : cfg.num_kv_shared_layers > 0 ∧ 0 < seq then
      some (initCache cfg m seq inputsEmbeds.device h.2).attnCaches else none
  let mut hidden := inputsEmbeds
  for hL : i in [:cfg.num_hidden_layers] do
    have hi : i < cfg.num_hidden_layers := by have := hL.upper; simpa using this
    let layer := m.layers.get ⟨i, hi⟩
    let perLayer := extractPerLayerInput? cfg perLayerInputs ⟨i, hi⟩
    match caches with
    | some layerCaches =>
      let sharedSource := (Config.sharedSourceLayer? cfg i).map layerCaches.get
      let (next, nextCache) := Gemma4Layer.forwardWithCache cfg layer hidden
        slidingCos slidingSin fullCos fullSin (layerCaches.get ⟨i, hi⟩) (Nat.le_refl _) sharedSource perLayer
      hidden := next
      if sharedSource.isNone then caches := some (layerCaches.set ⟨i, hi⟩ nextCache)
    | none =>
      hidden := Gemma4Layer.forward cfg layer hidden slidingCos slidingSin fullCos fullSin perLayer
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
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen)
    (hfit : seq ≤ maxLen)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen :=
  Id.run do
    let inputsEmbeds := embedTokens cfg m inputIds
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let slidingCos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 } := Tensor.slice slidingCosAll 0 0 seq
    let slidingSin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 } := Tensor.slice slidingSinAll 0 0 seq
    let fullCos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 } := Tensor.slice fullCosAll 0 0 seq
    let fullSin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 } := Tensor.slice fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for hL : i in [:cfg.num_hidden_layers] do
      have hi : i < cfg.num_hidden_layers := by have := hL.upper; simpa using this
      let layer := m.layers.get ⟨i, hi⟩
      let layerCache := cache'.attnCaches.get ⟨i, hi⟩
      let perLayer := extractPerLayerInput? cfg perLayerInputs ⟨i, hi⟩
      let sharedSource :=
        (Config.sharedSourceLayer? cfg i).map cache'.attnCaches.get
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
          hfit
          sharedSource
          perLayer
      hidden := hiddenNext
      if sharedSource.isNone then
        cache' := { cache' with attnCaches := cache'.attnCaches.set ⟨i, hi⟩ layerCacheNext }
    return (m.norm.forward3d hidden, cache')

def forwardWithCacheEmbeds {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen)
    (hfit : seq ≤ maxLen)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen :=
  Id.run do
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let slidingCos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 } := Tensor.slice slidingCosAll 0 0 seq
    let slidingSin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 } := Tensor.slice slidingSinAll 0 0 seq
    let fullCos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 } := Tensor.slice fullCosAll 0 0 seq
    let fullSin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 } := Tensor.slice fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for hL : i in [:cfg.num_hidden_layers] do
      have hi : i < cfg.num_hidden_layers := by have := hL.upper; simpa using this
      let layer := m.layers.get ⟨i, hi⟩
      let layerCache := cache'.attnCaches.get ⟨i, hi⟩
      let perLayer := extractPerLayerInput? cfg perLayerInputs ⟨i, hi⟩
      let sharedSource :=
        (Config.sharedSourceLayer? cfg i).map cache'.attnCaches.get
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
          hfit
          sharedSource
          perLayer
      hidden := hiddenNext
      if sharedSource.isNone then
        cache' := { cache' with attnCaches := cache'.attnCaches.set ⟨i, hi⟩ layerCacheNext }
    return (m.norm.forward3d hidden, cache')

def forwardWithCacheEmbedsWithPerLayerInputs {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen)
    (hfit : seq ≤ maxLen)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen :=
  Id.run do
    let _ := inputIds
    let slidingCos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 } := Tensor.slice slidingCosAll 0 0 seq
    let slidingSin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 } := Tensor.slice slidingSinAll 0 0 seq
    let fullCos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 } := Tensor.slice fullCosAll 0 0 seq
    let fullSin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 } := Tensor.slice fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for hL : i in [:cfg.num_hidden_layers] do
      have hi : i < cfg.num_hidden_layers := by have := hL.upper; simpa using this
      let layer := m.layers.get ⟨i, hi⟩
      let layerCache := cache'.attnCaches.get ⟨i, hi⟩
      let perLayer := extractPerLayerInput? cfg perLayerInputs ⟨i, hi⟩
      let sharedSource :=
        (Config.sharedSourceLayer? cfg i).map cache'.attnCaches.get
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
          hfit
          sharedSource
          perLayer
      hidden := hiddenNext
      if sharedSource.isNone then
        cache' := { cache' with attnCaches := cache'.attnCaches.set ⟨i, hi⟩ layerCacheNext }
    return (m.norm.forward3d hidden, cache')

def forwardWithCacheEmbedsWithPerLayerInputsBidirectionalVision {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen)
    (hfit : seq ≤ maxLen)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (slidingAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen :=
  Id.run do
    let _ := inputIds
    let slidingCos : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 } := Tensor.slice slidingCosAll 0 0 seq
    let slidingSin : Tensor { device := device, shape := #[seq, cfg.half_head_dim], dtype := .Float32 } := Tensor.slice slidingSinAll 0 0 seq
    let fullCos : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 } := Tensor.slice fullCosAll 0 0 seq
    let fullSin : Tensor { device := device, shape := #[seq, Config.fullHalfHeadDim cfg], dtype := .Float32 } := Tensor.slice fullSinAll 0 0 seq
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for hL : i in [:cfg.num_hidden_layers] do
      have hi : i < cfg.num_hidden_layers := by have := hL.upper; simpa using this
      let layer := m.layers.get ⟨i, hi⟩
      let layerCache := cache'.attnCaches.get ⟨i, hi⟩
      let perLayer := extractPerLayerInput? cfg perLayerInputs ⟨i, hi⟩
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
          hfit
          perLayer
      hidden := hiddenNext
      cache' := { cache' with attnCaches := cache'.attnCaches.set ⟨i, hi⟩ layerCacheNext }
    return (m.norm.forward3d hidden, cache')

def forwardStep {batch cap : Nat}
    (cfg : Config)
    (m : Gemma4Model (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 })
    (position : Nat) (hpos : position < cap)
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch cap)
    (slidingCosAll : Tensor { device := device, shape := #[cap, cfg.half_head_dim], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[cap, cfg.half_head_dim], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[cap, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[cap, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch cap :=
  Id.run do
    let inputsEmbeds := embedTokens cfg m inputIds
    let perLayerInputs := computePerLayerInputs cfg m inputIds inputsEmbeds
    let slidingCos : Tensor { device := device, shape := #[1, cfg.half_head_dim], dtype := .Float32 } := Tensor.slice slidingCosAll 0 position 1
    let slidingSin : Tensor { device := device, shape := #[1, cfg.half_head_dim], dtype := .Float32 } := Tensor.slice slidingSinAll 0 position 1
    let fullCos : Tensor { device := device, shape := #[1, Config.fullHalfHeadDim cfg], dtype := .Float32 } := Tensor.slice fullCosAll 0 position 1
    let fullSin : Tensor { device := device, shape := #[1, Config.fullHalfHeadDim cfg], dtype := .Float32 } := Tensor.slice fullSinAll 0 position 1
    let mut hidden := inputsEmbeds
    let mut cache' := cache
    for hL : i in [:cfg.num_hidden_layers] do
      have hi : i < cfg.num_hidden_layers := by have := hL.upper; simpa using this
      let layer := m.layers.get ⟨i, hi⟩
      let layerCache := cache'.attnCaches.get ⟨i, hi⟩
      let perLayer :=
        match perLayerInputs with
        | some p =>
          let one : Tensor { device := device, shape := #[batch, 1, 1, cfg.hidden_size_per_layer_input], dtype := dtype } := Tensor.slice p 2 i 1
          some (Tensor.reshape one #[batch, 1, cfg.hidden_size_per_layer_input])
        | none => none
      let sharedSource :=
        (Config.sharedSourceLayer? cfg i).map cache'.attnCaches.get
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
          position
          hpos
          sharedSource
          perLayer
      hidden := hiddenNext
      if sharedSource.isNone then
        cache' := { cache' with attnCaches := cache'.attnCaches.set ⟨i, hi⟩ layerCacheNext }
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
  capLogits cfg (Tensor.linear3d hidden m.lmHead)

def forwardEmbeds {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := dtype } :=
  let hidden := m.model.forwardEmbeds cfg inputIds inputsEmbeds
  capLogits cfg (Tensor.linear3d hidden m.lmHead)

def forwardEmbedsWithPerLayerInputs {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := dtype } :=
  let hidden := m.model.forwardEmbedsWithPerLayerInputs cfg inputIds inputsEmbeds perLayerInputs
  capLogits cfg (Tensor.linear3d hidden m.lmHead)

def forwardEmbedsWithPerLayerInputsBidirectionalVision {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (slidingAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := dtype } :=
  -- An empty sequence has no K/V to share, so the vision masks change nothing.
  if hSeq : seq = 0 then forwardEmbedsWithPerLayerInputs cfg m inputIds inputsEmbeds perLayerInputs else
  let cache := Gemma4Model.initCache cfg m.model seq inputsEmbeds.device (by omega)
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
      (Nat.le_refl _)
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
      slidingAttnMask
      fullAttnMask
  capLogits cfg (Tensor.linear3d hidden m.lmHead)

private def precomputeDecodeRotary {maxLen : Nat}
    (cfg : Config)
    (device : Device)
    : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 }
      × Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 }
      × Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 }
      × Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 } :=
  let (slCos, slSin) := Gemma4Model.precomputeSlidingRotary cfg device
  let (fullCos, fullSin) := Gemma4Model.precomputeFullRotary cfg device
  (slCos, slSin, fullCos, fullSin)

private def prefillCaches {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen)
    (hfit : seq ≤ maxLen)
    (hseq : 0 < seq)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen) := do
  let (hidden, cache') :=
    Gemma4Model.forwardWithCache
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      cache
      hfit
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let lastHidden : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := Tensor.slice hidden 1 (seq - 1) 1
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := capLogits cfg (Tensor.linear3d lastHidden m.lmHead)
  pure (Tensor.reshape logits3 #[batch, cfg.vocab_size], cache')

private def prefillCachesFromEmbeds {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen)
    (hfit : seq ≤ maxLen)
    (hseq : 0 < seq)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen) := do
  let (hidden, cache') :=
    Gemma4Model.forwardWithCacheEmbeds
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      inputsEmbeds
      cache
      hfit
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let lastHidden : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := Tensor.slice hidden 1 (seq - 1) 1
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := capLogits cfg (Tensor.linear3d lastHidden m.lmHead)
  pure (Tensor.reshape logits3 #[batch, cfg.vocab_size], cache')

private def prefillCachesFromEmbedsWithPerLayerInputs {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen)
    (hfit : seq ≤ maxLen)
    (hseq : 0 < seq)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen) := do
  let (hidden, cache') :=
    Gemma4Model.forwardWithCacheEmbedsWithPerLayerInputs
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      inputsEmbeds
      perLayerInputs
      cache
      hfit
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let lastHidden : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := Tensor.slice hidden 1 (seq - 1) 1
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := capLogits cfg (Tensor.linear3d lastHidden m.lmHead)
  pure (Tensor.reshape logits3 #[batch, cfg.vocab_size], cache')

private def prefillCachesFromEmbedsWithPerLayerInputsBidirectionalVision {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen)
    (hfit : seq ≤ maxLen)
    (hseq : 0 < seq)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (slidingAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    (fullAttnMask : Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 })
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen) := do
  let (hidden, cache') :=
    Gemma4Model.forwardWithCacheEmbedsWithPerLayerInputsBidirectionalVision
      (maxLen := maxLen)
      cfg
      m.model
      inputIds
      inputsEmbeds
      perLayerInputs
      cache
      hfit
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
      slidingAttnMask
      fullAttnMask
  let lastHidden : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := Tensor.slice hidden 1 (seq - 1) 1
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := capLogits cfg (Tensor.linear3d lastHidden m.lmHead)
  pure (Tensor.reshape logits3 #[batch, cfg.vocab_size], cache')

private def decodeStep {batch maxLen : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 })
    (position : Nat) (hpos : position < maxLen)
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen)
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } × Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen) := do
  let (hidden, cache') :=
    Gemma4Model.forwardStep
      cfg
      m.model
      inputIds
      position
      hpos
      cache
      slidingCosAll
      slidingSinAll
      fullCosAll
      fullSinAll
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := capLogits cfg (Tensor.linear3d hidden m.lmHead)
  pure (Tensor.reshape logits3 #[batch, cfg.vocab_size], cache')

private partial def decodeLoopCached {batch maxLen : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (strategy : SamplingStrategy)
    (eosTokenIds : Array tokenizer.TokenId)
    (finished : Tensor { device := device, shape := #[batch], dtype := .Bool })
    (remaining : Nat)
    (cache : Gemma4Cache (dtype := dtype) (device := device) cfg batch maxLen)
    (lastLogits : Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype })
    (slidingCosAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (slidingSinAll : Tensor { device := device, shape := #[maxLen, cfg.half_head_dim], dtype := .Float32 })
    (fullCosAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (fullSinAll : Tensor { device := device, shape := #[maxLen, Config.fullHalfHeadDim cfg], dtype := .Float32 })
    (onStep : Option (StreamCallback (device := device) batch))
    (generatedSoFar : Nat)
    {curSeq : Nat}
    (curIds : Tensor { device := device, shape := #[batch, curSeq], dtype := .Int64 })
    (hfit : curSeq + remaining ≤ maxLen)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let ⟨hr⟩ ← if h : remaining ≠ 0 then pure (PLift.up h) else
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

  let nextCol : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 } := Tensor.reshape nextTok #[batch, 1]
  let appended : Tensor { device := device, shape := #[batch, curSeq + 1], dtype := .Int64 } := Tensor.cat curIds nextCol 1

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
        (by omega)
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
      (by omega)

private def generateCore {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array tokenizer.TokenId := #[])
    (onStep : Option (StreamCallback (device := device) batch) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let ⟨hSeq⟩ ← if h : 0 < seq then pure (PLift.up h) else
    throw <| IO.userError "generate requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let maxLen := seq + maxNewTokens
  let (slidingCosAll, slidingSinAll, fullCosAll, fullSinAll) :=
    precomputeDecodeRotary (maxLen := maxLen) cfg inputIds.device
  let cache := Gemma4Model.initCache cfg m.model maxLen inputIds.device (h := Nat.lt_of_lt_of_le hSeq (Nat.le_add_right _ _))
  let (logits, cache') ←
    prefillCaches
      (maxLen := maxLen)
      cfg m inputIds cache (Nat.le_add_right _ _) hSeq
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
    (Nat.le_refl _)

private def generateFromEmbedsCore {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array tokenizer.TokenId := #[])
    (onStep : Option (StreamCallback (device := device) batch) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let ⟨hSeq⟩ ← if h : 0 < seq then pure (PLift.up h) else
    throw <| IO.userError "generate requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let maxLen := seq + maxNewTokens
  let (slidingCosAll, slidingSinAll, fullCosAll, fullSinAll) :=
    precomputeDecodeRotary (maxLen := maxLen) cfg inputsEmbeds.device
  let cache := Gemma4Model.initCache cfg m.model maxLen inputsEmbeds.device (h := Nat.lt_of_lt_of_le hSeq (Nat.le_add_right _ _))
  let (logits, cache') ←
    prefillCachesFromEmbeds
      (maxLen := maxLen)
      cfg
      m
      inputIds
      inputsEmbeds
      cache
      (Nat.le_add_right _ _) hSeq
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
      (Nat.le_refl _)

private def generateFromEmbedsCoreWithPerLayerInputs {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (perLayerInputs : Option (Tensor { device := device, shape := #[batch, seq, cfg.num_hidden_layers, cfg.hidden_size_per_layer_input], dtype := dtype }))
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array tokenizer.TokenId := #[])
    (onStep : Option (StreamCallback (device := device) batch) := none)
    (prefillMasks : Option (Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 } × Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 }) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let support ← dtype.acquireConstantSupport device
  let _ := support.down
  let ⟨hSeq⟩ ← if h : 0 < seq then pure (PLift.up h) else
    throw <| IO.userError "generate requires non-empty prompt sequence"
  if maxNewTokens == 0 then
    return ⟨seq, inputIds⟩

  let maxLen := seq + maxNewTokens
  let (slidingCosAll, slidingSinAll, fullCosAll, fullSinAll) :=
    precomputeDecodeRotary (maxLen := maxLen) cfg inputsEmbeds.device
  let cache := Gemma4Model.initCache cfg m.model maxLen inputsEmbeds.device (h := Nat.lt_of_lt_of_le hSeq (Nat.le_add_right _ _))
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
        (Nat.le_add_right _ _) hSeq
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
        (Nat.le_add_right _ _) hSeq
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
    (Nat.le_refl _)

def generate {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array tokenizer.TokenId := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateCore cfg m inputIds maxNewTokens strategy eosTokenIds none

def generateFromEmbeds {batch seq : Nat}
    (cfg : Config)
    (m : Gemma4ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array tokenizer.TokenId := #[])
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
    (eosTokenIds : Array tokenizer.TokenId := #[])
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
    (eosTokenIds : Array tokenizer.TokenId := #[])
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
    (eosTokenIds : Array tokenizer.TokenId := #[])
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
    (eosTokenIds : Array tokenizer.TokenId := #[])
    (prefillMasks : Option (Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 } × Tensor { device := device, shape := #[batch, seq, seq], dtype := .Int64 }) := none)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCoreWithPerLayerInputs
    cfg m inputIds inputsEmbeds perLayerInputs maxNewTokens strategy eosTokenIds (some onStep) prefillMasks

end Gemma4ForCausalLM

end torch.gemma4
