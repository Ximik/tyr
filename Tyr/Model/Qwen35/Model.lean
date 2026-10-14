/-
  Tyr/Model/Qwen35/Model.lean

  Standalone Qwen3.5 text causal-LM for Tyr.
  Features:
  - Hybrid layer schedule (`linear_attention` + `full_attention`)
  - Dense FFN and MoE FFN variants
  - Full-attention Q-gate + per-head Q/K RMSNorm
  - Linear GatedDeltaNet with depthwise causal conv and recurrent state cache
  - Cached and uncached generation
-/
import Tyr.Model.Qwen35.TypedOps
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Model.Qwen.Attention
import Tyr.Model.Qwen35.Config
import Tyr.Model.Utils
import Tyr.Model.Generation

namespace torch.qwen35

open torch
open torch.Model

private def l2NormLast4d {b s h d : UInt64}
    (x : Tensor { shape := #[b, s, h, d] })
    (eps : Float := 1e-6)
    : Tensor { shape := #[b, s, h, d] } :=
  let flat : Tensor { shape := #[b * s * h, d] } := Tensor.reshapeChecked x #[b * s * h, d]
  let n2 : Tensor { shape := #[b * s * h, 1] } := Tensor.Preserving.sumDim (flat * flat) 1 true
  let inv : Tensor { shape := #[b * s * h, 1] } := rsqrtDynamic (n2 + eps)
  let inv : Tensor { shape := #[b * s * h, d] } := Tensor.expand inv #[b * s * h, d]
  Tensor.reshapeChecked (flat * inv) #[b, s, h, d]

private def applyMaskToPaddingStates {batch seq hidden : UInt64}
    (x : Tensor { shape := #[batch, seq, hidden] })
    (attentionMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64 }))
    : Tensor { shape := #[batch, seq, hidden] } :=
  if batch <= 1 || seq <= 1 then
    x
  else
    match attentionMask with
    | none => x
    | some m =>
      let mf : Tensor { shape := #[batch, seq] } := floatDynamic m
      let me : Tensor { shape := #[batch, seq, hidden] } := Tensor.expand (Tensor.reshapeChecked mf #[batch, seq, 1]) #[batch, seq, hidden]
      x * me

/-- Qwen3.5 RMSNorm with `(1 + weight)` scaling. -/
structure Qwen35RMSNorm (dim : UInt64) where
  weight : Tensor { shape := #[dim] }
  eps : Float := 1e-6
  deriving TensorStruct

namespace Qwen35RMSNorm

def initZeroCentered (dim : UInt64) (eps : Float := 1e-6) : Qwen35RMSNorm dim :=
  { weight := Tensor.setRequiresGrad (zerosDynamic #[dim]) true, eps }

def initOneCentered (dim : UInt64) (eps : Float := 1e-6) : Qwen35RMSNorm dim :=
  { weight := Tensor.setRequiresGrad (onesDynamic #[dim]) true, eps }

/-- HuggingFace Qwen3.5 checkpoints already store the zero-centered RMSNorm
    parameter used by the `(1 + weight)` formulation, so the raw tensor should
    be preserved as-is. -/
def fromCheckpointWeight {dim : UInt64}
    (weight : Tensor { shape := #[dim] })
    (eps : Float := 1e-6)
    : Qwen35RMSNorm dim :=
  { weight := weight, eps }

def forward2d {n dim : UInt64}
    (m : Qwen35RMSNorm dim)
    (x : Tensor { shape := #[n, dim] })
    : Tensor { shape := #[n, dim] } :=
  let xf : Tensor { shape := #[n, dim] } := floatDynamic x
  let var : Tensor { shape := #[n, 1] } := Tensor.Preserving.meanDim (xf * xf) 1 true
  let inv : Tensor { shape := #[n, 1] } := rsqrtDynamic (var + m.eps)
  let inv : Tensor { shape := #[n, dim] } := Tensor.expand inv #[n, dim]
  let normed : Tensor { shape := #[n, dim] } := xf * inv
  let scale1 : Tensor { shape := #[dim] } := Tensor.Preserving.addScalar (floatDynamic m.weight) 1.0
  let scale : Tensor { shape := #[n, dim] } := Tensor.expand (Tensor.reshapeChecked scale1 #[1, dim]) #[n, dim]
  Tensor.castLike x (normed * scale)

def forward3d {batch seq dim : UInt64}
    (m : Qwen35RMSNorm dim)
    (x : Tensor { shape := #[batch, seq, dim] })
    : Tensor { shape := #[batch, seq, dim] } :=
  let flat : Tensor { shape := #[batch * seq, dim] } := Tensor.reshapeChecked x #[batch * seq, dim]
  Tensor.reshapeChecked (forward2d m flat) #[batch, seq, dim]

end Qwen35RMSNorm

/-- Gated RMSNorm used at linear-attention output. -/
structure Qwen35RMSNormGated (dim : UInt64) where
  weight : Tensor { shape := #[dim] }
  eps : Float := 1e-6
  deriving TensorStruct

namespace Qwen35RMSNormGated

def init (dim : UInt64) (eps : Float := 1e-6) : Qwen35RMSNormGated dim :=
  { weight := Tensor.setRequiresGrad (onesDynamic #[dim]) true, eps }

def forward2d {n dim : UInt64}
    (m : Qwen35RMSNormGated dim)
    (x : Tensor { shape := #[n, dim] })
    (gate : Tensor { shape := #[n, dim] })
    : Tensor { shape := #[n, dim] } :=
  let xf : Tensor { shape := #[n, dim] } := floatDynamic x
  let var : Tensor { shape := #[n, 1] } := Tensor.Preserving.meanDim (xf * xf) 1 true
  let inv : Tensor { shape := #[n, 1] } := rsqrtDynamic (var + m.eps)
  let inv : Tensor { shape := #[n, dim] } := Tensor.expand inv #[n, dim]
  let normed : Tensor { shape := #[n, dim] } := xf * inv
  let scale : Tensor { shape := #[n, dim] } := Tensor.expand (Tensor.reshapeChecked (floatDynamic m.weight) #[1, dim]) #[n, dim]
  let g : Tensor { shape := #[n, dim] } := Tensor.Preserving.silu (floatDynamic gate)
  Tensor.castLike x (normed * scale * g)

end Qwen35RMSNormGated

/-- SwiGLU MLP block (dense FFN). -/
structure Qwen35MLP (hidden_size intermediate_size : UInt64) where
  gate_proj : Tensor { shape := #[intermediate_size, hidden_size] }
  up_proj : Tensor { shape := #[intermediate_size, hidden_size] }
  down_proj : Tensor { shape := #[hidden_size, intermediate_size] }
  deriving TensorStruct

namespace Qwen35MLP

def init (hidden_size intermediate_size : UInt64) : IO (Qwen35MLP hidden_size intermediate_size) := do
  let gate ← initDynamicWeight #[intermediate_size, hidden_size] hidden_size
  let up ← initDynamicWeight #[intermediate_size, hidden_size] hidden_size
  let down ← initDynamicWeight #[hidden_size, intermediate_size] intermediate_size
  pure { gate_proj := gate, up_proj := up, down_proj := down }

def forward2d {tokens hidden inter : UInt64}
    (m : Qwen35MLP hidden inter)
    (x : Tensor { shape := #[tokens, hidden] })
    : Tensor { shape := #[tokens, hidden] } :=
  let gateDyn : Tensor { shape := #[tokens, inter] } := einsumChecked "oh,th->to" m.gate_proj x
  let upDyn : Tensor { shape := #[tokens, inter] } := einsumChecked "oh,th->to" m.up_proj x
  let gate : Tensor { shape := #[tokens, inter] } := Tensor.reshapeChecked gateDyn #[tokens, inter]
  let up : Tensor { shape := #[tokens, inter] } := Tensor.reshapeChecked upDyn #[tokens, inter]
  let hid : Tensor { shape := #[tokens, inter] } := Tensor.Preserving.silu gate * up
  let outDyn : Tensor { shape := #[tokens, hidden] } := einsumChecked "ho,to->th" m.down_proj hid
  Tensor.reshapeChecked outDyn #[tokens, hidden]

def forward {batch seq hidden inter : UInt64}
    (m : Qwen35MLP hidden inter)
    (x : Tensor { shape := #[batch, seq, hidden] })
    : Tensor { shape := #[batch, seq, hidden] } :=
  Tensor.linear3d (Tensor.Preserving.silu (Tensor.linear3d x m.gate_proj) * Tensor.linear3d x m.up_proj) m.down_proj

end Qwen35MLP

/-- Top-k router for Qwen3.5 MoE. -/
structure Qwen35TopKRouter (cfg : Config) where
  weight : Tensor { shape := #[cfg.num_experts, cfg.hidden_size] }
  deriving TensorStruct

namespace Qwen35TopKRouter

def init (cfg : Config) : IO (Qwen35TopKRouter cfg) := do
  let w ← initDynamicWeight #[cfg.num_experts, cfg.hidden_size] cfg.hidden_size
  pure { weight := w }

def forward {tokens : UInt64}
    (cfg : Config)
    (m : Qwen35TopKRouter cfg)
    (hidden : Tensor { shape := #[tokens, cfg.hidden_size] })
    : Tensor { shape := #[tokens, cfg.num_experts] } × Tensor { shape := #[tokens, cfg.num_experts_per_tok] } × Tensor { shape := #[tokens, cfg.num_experts_per_tok], dtype := .Int64 } :=
  let logitsDyn : Tensor { shape := #[tokens, cfg.num_experts] } := einsumChecked "eh,th->te" m.weight hidden
  let logits : Tensor { shape := #[tokens, cfg.num_experts] } := Tensor.reshapeChecked logitsDyn #[tokens, cfg.num_experts]
  let probs : Tensor { shape := #[tokens, cfg.num_experts] } := Tensor.Preserving.softmax logits (-1)
  let (topValsRaw, topIdx) := topK2d probs cfg.num_experts_per_tok 1
  let denom : Tensor { shape := #[tokens, 1] } := Tensor.Preserving.sumDim topValsRaw 1 true
  let denom : Tensor { shape := #[tokens, cfg.num_experts_per_tok] } := Tensor.expand denom #[tokens, cfg.num_experts_per_tok]
  let topVals : Tensor { shape := #[tokens, cfg.num_experts_per_tok] } := Tensor.Preserving.div topValsRaw denom
  (probs, topVals, topIdx)

end Qwen35TopKRouter

/-- Expert weight bank for Qwen3.5 MoE FFN. -/
structure Qwen35MoeExperts (cfg : Config) where
  gate_up_proj : Tensor { shape := #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size] }
  down_proj : Tensor { shape := #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size] }
  deriving TensorStruct

namespace Qwen35MoeExperts

def init (cfg : Config) : IO (Qwen35MoeExperts cfg) := do
  let gu ← initDynamicWeight #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size] cfg.hidden_size
  let down ← initDynamicWeight #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size] cfg.moe_intermediate_size
  pure { gate_up_proj := gu, down_proj := down }

def forward2d {tokens : UInt64}
    (cfg : Config)
    (m : Qwen35MoeExperts cfg)
    (hidden : Tensor { shape := #[tokens, cfg.hidden_size] })
    (topVals : Tensor { shape := #[tokens, cfg.num_experts_per_tok] })
    (topIdx : Tensor { shape := #[tokens, cfg.num_experts_per_tok], dtype := .Int64 })
    : Tensor { shape := #[tokens, cfg.hidden_size] } :=
  Id.run do
    let device := hidden.device
    let mut acc : Tensor { shape := #[tokens, cfg.hidden_size] } := zerosDynamicOn device

    for slot in [:cfg.num_experts_per_tok.toNat] do
      -- Gather ONLY the routed expert's weights per token (an embedding row
      -- lookup) instead of a one-hot contraction over all `num_experts`. This is
      -- numerically identical to the one-hot selection but ~num_experts x cheaper.
      let idx2d : Tensor { shape := #[tokens, 1], dtype := .Int64 } := typedSlice topIdx 1 slot.toUInt64 1
      let idxRow : Tensor { shape := #[1, tokens], dtype := .Int64 } := Tensor.reshapeChecked idx2d #[1, tokens]
      let gateUp2d : Tensor { shape := #[cfg.num_experts, 2 * cfg.moe_intermediate_size * cfg.hidden_size] } :=
        Tensor.reshapeChecked m.gate_up_proj #[cfg.num_experts, 2 * cfg.moe_intermediate_size * cfg.hidden_size]
      let tokGateUpE : Tensor { shape := #[1, tokens, 2 * cfg.moe_intermediate_size * cfg.hidden_size] } :=
        Tensor.embedding idxRow gateUp2d
      let tokGateUp : Tensor { shape := #[tokens, 2 * cfg.moe_intermediate_size, cfg.hidden_size] } :=
        Tensor.reshapeChecked tokGateUpE #[tokens, 2 * cfg.moe_intermediate_size, cfg.hidden_size]

      let guDyn : Tensor { shape := #[tokens, 2 * cfg.moe_intermediate_size] } := einsumChecked "tih,th->ti" tokGateUp hidden
      let gu : Tensor { shape := #[tokens, 2 * cfg.moe_intermediate_size] } := Tensor.reshapeChecked guDyn #[tokens, 2 * cfg.moe_intermediate_size]

      let gate : Tensor { shape := #[tokens, cfg.moe_intermediate_size] } := typedSlice gu 1 0 cfg.moe_intermediate_size
      let up : Tensor { shape := #[tokens, cfg.moe_intermediate_size] } :=
        typedSlice gu 1 cfg.moe_intermediate_size cfg.moe_intermediate_size
      let inter : Tensor { shape := #[tokens, cfg.moe_intermediate_size] } := Tensor.Preserving.silu gate * up

      let down2d : Tensor { shape := #[cfg.num_experts, cfg.hidden_size * cfg.moe_intermediate_size] } :=
        Tensor.reshapeChecked m.down_proj #[cfg.num_experts, cfg.hidden_size * cfg.moe_intermediate_size]
      let tokDownE : Tensor { shape := #[1, tokens, cfg.hidden_size * cfg.moe_intermediate_size] } :=
        Tensor.embedding idxRow down2d
      let tokDown : Tensor { shape := #[tokens, cfg.hidden_size, cfg.moe_intermediate_size] } :=
        Tensor.reshapeChecked tokDownE #[tokens, cfg.hidden_size, cfg.moe_intermediate_size]

      let outDyn : Tensor { shape := #[tokens, cfg.hidden_size] } := einsumChecked "thi,ti->th" tokDown inter
      let out : Tensor { shape := #[tokens, cfg.hidden_size] } := Tensor.reshapeChecked outDyn #[tokens, cfg.hidden_size]

      let w2d : Tensor { shape := #[tokens, 1] } := typedSlice topVals 1 slot.toUInt64 1
      let w : Tensor { shape := #[tokens, cfg.hidden_size] } := Tensor.expand w2d #[tokens, cfg.hidden_size]
      acc := acc + out * w

    -- Restore the model dtype so downstream layers see consistent dtypes.
    return Tensor.castLike hidden acc

end Qwen35MoeExperts

/-- Sparse MoE block (routed experts + shared expert). -/
structure Qwen35SparseMoeBlock (cfg : Config) where
  router : Qwen35TopKRouter cfg
  experts : Qwen35MoeExperts cfg
  shared_expert : Qwen35MLP cfg.hidden_size cfg.shared_expert_intermediate_size
  shared_expert_gate : Tensor { shape := #[1, cfg.hidden_size] }
  deriving TensorStruct

namespace Qwen35SparseMoeBlock

def init (cfg : Config) : IO (Qwen35SparseMoeBlock cfg) := do
  let router ← Qwen35TopKRouter.init cfg
  let experts ← Qwen35MoeExperts.init cfg
  let sharedExpert ← Qwen35MLP.init cfg.hidden_size cfg.shared_expert_intermediate_size
  let sharedGate ← initDynamicWeight #[1, cfg.hidden_size] cfg.hidden_size
  pure {
    router := router
    experts := experts
    shared_expert := sharedExpert
    shared_expert_gate := sharedGate
  }

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35SparseMoeBlock cfg)
    (hidden : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let tokens := batch * seq
  let hidden2d : Tensor { shape := #[tokens, cfg.hidden_size] } := Tensor.reshapeChecked hidden #[tokens, cfg.hidden_size]

  let (_routerProbs, topVals, topIdx) := m.router.forward cfg hidden2d
  let expertOut : Tensor { shape := #[tokens, cfg.hidden_size] } :=
    m.experts.forward2d cfg hidden2d topVals topIdx

  let sharedOut : Tensor { shape := #[tokens, cfg.hidden_size] } :=
    m.shared_expert.forward2d hidden2d

  let gateDyn : Tensor { shape := #[tokens, 1] } := einsumChecked "oh,th->to" m.shared_expert_gate hidden2d
  let gate1 : Tensor { shape := #[tokens, 1] } := Tensor.reshapeChecked gateDyn #[tokens, 1]
  let gate : Tensor { shape := #[tokens, cfg.hidden_size] } :=
    Tensor.expand (Tensor.Preserving.sigmoid gate1) #[tokens, cfg.hidden_size]

  Tensor.reshapeChecked (expertOut + sharedOut * gate) #[batch, seq, cfg.hidden_size]

end Qwen35SparseMoeBlock

/-- Full-attention token mixer for Qwen3.5 (Q-gate + Q/K RMSNorm + partial RoPE). -/
structure Qwen35Attention (cfg : Config) where
  q_proj : Tensor { shape := #[cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size] }
  k_proj : Tensor { shape := #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] }
  v_proj : Tensor { shape := #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] }
  o_proj : Tensor { shape := #[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim] }
  q_norm : Qwen35RMSNorm cfg.head_dim
  k_norm : Qwen35RMSNorm cfg.head_dim
  deriving TensorStruct

namespace Qwen35Attention

abbrev KVCache (cfg : Config) (batch : UInt64) :=
  qwen.QwenAttention.KVCache batch cfg.num_key_value_heads cfg.head_dim (.Unknown "dynamic")

def init (cfg : Config) : IO (Qwen35Attention cfg) := do
  let q ← initDynamicWeight #[cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size] cfg.hidden_size
  let k ← initDynamicWeight #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] cfg.hidden_size
  let v ← initDynamicWeight #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] cfg.hidden_size
  let o ← initDynamicWeight #[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim] (cfg.num_attention_heads * cfg.head_dim)
  pure {
    q_proj := q
    k_proj := k
    v_proj := v
    o_proj := o
    q_norm := Qwen35RMSNorm.initZeroCentered cfg.head_dim cfg.rms_norm_eps
    k_norm := Qwen35RMSNorm.initZeroCentered cfg.head_dim cfg.rms_norm_eps
  }

private def applyHeadNorm {batch seq n_heads head_dim : UInt64}
    (norm : Qwen35RMSNorm head_dim)
    (x : Tensor { shape := #[batch, seq, n_heads, head_dim] })
    : Tensor { shape := #[batch, seq, n_heads, head_dim] } :=
  let flat : Tensor { shape := #[batch * seq * n_heads, head_dim] } := Tensor.reshapeChecked x #[batch * seq * n_heads, head_dim]
  Tensor.reshapeChecked (norm.forward2d flat) #[batch, seq, n_heads, head_dim]

private def applyRotaryPartial {batch seq n_head head_dim rotary_dim : UInt64}
    (x : Tensor { shape := #[batch, seq, n_head, head_dim] })
    (cos : Tensor { shape := #[seq, rotary_dim / 2], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, rotary_dim / 2], dtype := .Float32 })
    : Tensor { shape := #[batch, seq, n_head, head_dim] } :=
  let xRot : Tensor { shape := #[batch, seq, n_head, rotary_dim] } := typedSlice x 3 0 rotary_dim
  let xRot : Tensor { shape := #[batch, seq, n_head, rotary_dim] } := Tensor.applyRotary xRot cos sin
  let xPassLen : UInt64 := head_dim - rotary_dim
  let xPass : Tensor { shape := #[batch, seq, n_head, xPassLen] } := typedSlice x 3 rotary_dim xPassLen
  typedCat xRot xPass 3

private def splitQProjAndGate {batch seq : UInt64}
    (cfg : Config)
    (qgFlat : Tensor { shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim * 2] })
    : Tensor { shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim] }
      × Tensor { shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim] } :=
  let qg : Tensor { shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim * 2] } :=
    Tensor.reshapeChecked qgFlat #[batch, seq, cfg.num_attention_heads, cfg.head_dim * 2]
  let q : Tensor { shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim] } :=
    typedSlice qg 3 0 cfg.head_dim
  let gate : Tensor { shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim] } :=
    typedSlice qg 3 cfg.head_dim cfg.head_dim
  let gateFlat : Tensor { shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim] } :=
    Tensor.reshapeChecked gate #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
  (q, gateFlat)

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (is_causal : Bool := true)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let qg : Tensor { shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim * 2] } := Tensor.linear3d x m.q_proj
  let qLen : UInt64 := cfg.num_attention_heads * cfg.head_dim
  let (q, gateFlat) := splitQProjAndGate cfg qg

  let kFlat : Tensor { shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim] } := Tensor.linear3d x m.k_proj
  let vFlat : Tensor { shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim] } := Tensor.linear3d x m.v_proj

  let k : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked kFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]
  let v : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked vFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

  let q := applyHeadNorm m.q_norm q
  let k := applyHeadNorm m.k_norm k

  let q := applyRotaryPartial q cos sin
  let k := applyRotaryPartial k cos sin

  let qh : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } := Tensor.splitHeads q
  let kh : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := Tensor.splitHeads k
  let vh : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := Tensor.splitHeads v

  let attn : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } :=
    Tensor.attentionGQA qh kh vh 0.0 is_causal true

  let attn : Tensor { shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim] } := Tensor.joinHeads attn
  let attnFlat : Tensor { shape := #[batch, seq, qLen] } := Tensor.reshapeChecked attn #[batch, seq, qLen]
  let gated : Tensor { shape := #[batch, seq, qLen] } := attnFlat * Tensor.Preserving.sigmoid gateFlat
  Tensor.linear3d gated m.o_proj

/-- Forward pass with optional KV-cache update (for batch prefill). -/
def forwardWithCache {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (cache : KVCache cfg batch)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × KVCache cfg batch :=
  let qg : Tensor { shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim * 2] } := Tensor.linear3d x m.q_proj
  let qLen : UInt64 := cfg.num_attention_heads * cfg.head_dim
  let (q, gateFlat) := splitQProjAndGate cfg qg

  let kFlat : Tensor { shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim] } := Tensor.linear3d x m.k_proj
  let vFlat : Tensor { shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim] } := Tensor.linear3d x m.v_proj

  let k : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked kFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]
  let v : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked vFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

  let q := applyHeadNorm m.q_norm q
  let k := applyHeadNorm m.k_norm k

  let q := applyRotaryPartial q cos sin
  let k := applyRotaryPartial k cos sin

  -- Update cache with full sequence
  let kh_new := Tensor.splitHeads k
  let vh_new := Tensor.splitHeads v
  let kStore : Tensor { shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
    Tensor.castLike kh_new (Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])
  let vStore : Tensor { shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
    Tensor.castLike vh_new (Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])

  let kStore' := Tensor.sliceScatter kStore 2 0 kh_new
  let vStore' := Tensor.sliceScatter vStore 2 0 vh_new

  let cache' : KVCache cfg batch := { cache with
    kStoreDyn := Tensor.packShape kStore'
    vStoreDyn := Tensor.packShape vStore'
    seq := seq
  }

  let qh : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } := Tensor.splitHeads q
  let kh : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := typedSlice kStore' 2 0 seq
  let vh : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := typedSlice vStore' 2 0 seq
  let attn : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } :=
    Tensor.attentionGQA qh kh vh 0.0 true true

  let attn : Tensor { shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim] } := Tensor.joinHeads attn
  let attnFlat : Tensor { shape := #[batch, seq, qLen] } := Tensor.reshapeChecked attn #[batch, seq, qLen]
  let gated : Tensor { shape := #[batch, seq, qLen] } := attnFlat * Tensor.Preserving.sigmoid gateFlat
  (Tensor.linear3d gated m.o_proj, cache')

def forwardMasked {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35Attention cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (attn_mask : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (is_causal : Bool := true)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let qg : Tensor { shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim * 2] } := Tensor.linear3d x m.q_proj
  let qLen : UInt64 := cfg.num_attention_heads * cfg.head_dim
  let (q, gateFlat) := splitQProjAndGate cfg qg

  let kFlat : Tensor { shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim] } := Tensor.linear3d x m.k_proj
  let vFlat : Tensor { shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim] } := Tensor.linear3d x m.v_proj

  let k : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked kFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]
  let v : Tensor { shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked vFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

  let q := applyHeadNorm m.q_norm q
  let k := applyHeadNorm m.k_norm k

  let q := applyRotaryPartial q cos sin
  let k := applyRotaryPartial k cos sin

  let qh : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } := Tensor.splitHeads q
  let kh : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := Tensor.splitHeads k
  let vh : Tensor { shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim] } := Tensor.splitHeads v

  let attn : Tensor { shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim] } :=
    Tensor.attentionGQAMask qh kh vh attn_mask 0.0 is_causal true

  let attn : Tensor { shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim] } := Tensor.joinHeads attn
  let attnFlat : Tensor { shape := #[batch, seq, qLen] } := Tensor.reshapeChecked attn #[batch, seq, qLen]
  let gated : Tensor { shape := #[batch, seq, qLen] } := attnFlat * Tensor.Preserving.sigmoid gateFlat
  Tensor.linear3d gated m.o_proj

def forwardStep {batch : UInt64}
    (cfg : Config)
    (m : Qwen35Attention cfg)
    (x : Tensor { shape := #[batch, 1, cfg.hidden_size] })
    (cos : Tensor { shape := #[1, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { shape := #[1, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (cache : KVCache cfg batch)
    : Tensor { shape := #[batch, 1, cfg.hidden_size] } × KVCache cfg batch :=
  let qg : Tensor { shape := #[batch, 1, cfg.num_attention_heads * cfg.head_dim * 2] } := Tensor.linear3d x m.q_proj
  let qLen : UInt64 := cfg.num_attention_heads * cfg.head_dim
  let (q, gateFlat) := splitQProjAndGate cfg qg

  let kFlat : Tensor { shape := #[batch, 1, cfg.num_key_value_heads * cfg.head_dim] } := Tensor.linear3d x m.k_proj
  let vFlat : Tensor { shape := #[batch, 1, cfg.num_key_value_heads * cfg.head_dim] } := Tensor.linear3d x m.v_proj

  let k : Tensor { shape := #[batch, 1, cfg.num_key_value_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked kFlat #[batch, 1, cfg.num_key_value_heads, cfg.head_dim]
  let v : Tensor { shape := #[batch, 1, cfg.num_key_value_heads, cfg.head_dim] } :=
    Tensor.reshapeChecked vFlat #[batch, 1, cfg.num_key_value_heads, cfg.head_dim]

  let q := applyHeadNorm m.q_norm q
  let k := applyHeadNorm m.k_norm k

  let q := applyRotaryPartial q cos sin
  let k := applyRotaryPartial k cos sin

  let qh : Tensor { shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim] } := Tensor.splitHeads q
  let kNew : Tensor { shape := #[batch, cfg.num_key_value_heads, 1, cfg.head_dim] } := Tensor.splitHeads k
  let vNew : Tensor { shape := #[batch, cfg.num_key_value_heads, 1, cfg.head_dim] } := Tensor.splitHeads v

  let kStore : Tensor { shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
    Tensor.castLike kNew (Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])
  let vStore : Tensor { shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
    Tensor.castLike vNew (Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])

  if cache.seq < cache.maxLen then
    let kStore' : Tensor { shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
      Tensor.sliceScatter kStore 2 cache.seq kNew
    let vStore' : Tensor { shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
      Tensor.sliceScatter vStore 2 cache.seq vNew
    let kvLen : UInt64 := cache.seq + 1
    let kAll : Tensor { shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] } := typedSlice kStore' 2 0 kvLen
    let vAll : Tensor { shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] } := typedSlice vStore' 2 0 kvLen
    let attn : Tensor { shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim] } :=
      Tensor.attentionGQAQKV qh kAll vAll 0.0 false true
    let attn : Tensor { shape := #[batch, 1, cfg.num_attention_heads, cfg.head_dim] } := Tensor.joinHeads attn
    let attnFlat : Tensor { shape := #[batch, 1, qLen] } := Tensor.reshapeChecked attn #[batch, 1, qLen]
    let gated : Tensor { shape := #[batch, 1, qLen] } := attnFlat * Tensor.Preserving.sigmoid gateFlat
    let out : Tensor { shape := #[batch, 1, cfg.hidden_size] } := Tensor.linear3d gated m.o_proj
    let cache' : KVCache cfg batch := {
      kStoreDyn := Tensor.packShape kStore'
      vStoreDyn := Tensor.packShape vStore'
      seq := kvLen
      maxLen := cache.maxLen
    }
    (out, cache')
  else
    let writePos : UInt64 := if cache.maxLen == 0 then 0 else cache.maxLen - 1
    let kStore' : Tensor { shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
      Tensor.sliceScatter kStore 2 writePos kNew
    let vStore' : Tensor { shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim] } :=
      Tensor.sliceScatter vStore 2 writePos vNew
    let kvLen : UInt64 := cache.maxLen
    let kAll : Tensor { shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] } := typedSlice kStore' 2 0 kvLen
    let vAll : Tensor { shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim] } := typedSlice vStore' 2 0 kvLen
    let attn : Tensor { shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim] } :=
      Tensor.attentionGQAQKV qh kAll vAll 0.0 false true
    let attn : Tensor { shape := #[batch, 1, cfg.num_attention_heads, cfg.head_dim] } := Tensor.joinHeads attn
    let attnFlat : Tensor { shape := #[batch, 1, qLen] } := Tensor.reshapeChecked attn #[batch, 1, qLen]
    let gated : Tensor { shape := #[batch, 1, qLen] } := attnFlat * Tensor.Preserving.sigmoid gateFlat
    let out : Tensor { shape := #[batch, 1, cfg.hidden_size] } := Tensor.linear3d gated m.o_proj
    let cache' : KVCache cfg batch := {
      kStoreDyn := Tensor.packShape kStore'
      vStoreDyn := Tensor.packShape vStore'
      seq := cache.maxLen
      maxLen := cache.maxLen
    }
    (out, cache')

end Qwen35Attention

/-- Linear token mixer for Qwen3.5 (`GatedDeltaNet`). -/
structure Qwen35GatedDeltaNet (cfg : Config) where
  in_proj_qkv : Tensor { shape := #[Config.linearConvDim cfg, cfg.hidden_size] }
  in_proj_z : Tensor { shape := #[Config.linearValueDim cfg, cfg.hidden_size] }
  in_proj_b : Tensor { shape := #[cfg.linear_num_value_heads, cfg.hidden_size] }
  in_proj_a : Tensor { shape := #[cfg.linear_num_value_heads, cfg.hidden_size] }

  conv1d_weight : Tensor { shape := #[Config.linearConvDim cfg, 1, cfg.linear_conv_kernel_dim] }
  conv1d_bias : Tensor { shape := #[Config.linearConvDim cfg] }

  dt_bias : Tensor { shape := #[cfg.linear_num_value_heads] }
  a_log : Tensor { shape := #[cfg.linear_num_value_heads] }

  norm : Qwen35RMSNormGated cfg.linear_value_head_dim
  out_proj : Tensor { shape := #[cfg.hidden_size, Config.linearValueDim cfg] }
  deriving TensorStruct

namespace Qwen35GatedDeltaNet

def init (cfg : Config) : IO (Qwen35GatedDeltaNet cfg) := do
  let inQKV ← initDynamicWeight #[Config.linearConvDim cfg, cfg.hidden_size] cfg.hidden_size
  let inZ ← initDynamicWeight #[Config.linearValueDim cfg, cfg.hidden_size] cfg.hidden_size
  let inB ← initDynamicWeight #[cfg.linear_num_value_heads, cfg.hidden_size] cfg.hidden_size
  let inA ← initDynamicWeight #[cfg.linear_num_value_heads, cfg.hidden_size] cfg.hidden_size

  let convW ← initDynamicWeight #[Config.linearConvDim cfg, 1, cfg.linear_conv_kernel_dim] cfg.linear_conv_kernel_dim
  let convB := initDynamicBias #[Config.linearConvDim cfg]

  let dtBias := Tensor.setRequiresGrad (onesDynamic #[cfg.linear_num_value_heads]) true

  let aRand ← randDynamic #[cfg.linear_num_value_heads]
  let aScaled : Tensor { shape := #[cfg.linear_num_value_heads] } := Tensor.Preserving.addScalar (Tensor.Preserving.mulScalar aRand 16.0) 1e-6
  let aLog := Tensor.setRequiresGrad (logDynamic aScaled) true

  let outProj ← initDynamicWeight #[cfg.hidden_size, Config.linearValueDim cfg] (Config.linearValueDim cfg)
  pure {
    in_proj_qkv := inQKV
    in_proj_z := inZ
    in_proj_b := inB
    in_proj_a := inA
    conv1d_weight := convW
    conv1d_bias := convB
    dt_bias := dtBias
    a_log := aLog
    norm := Qwen35RMSNormGated.init cfg.linear_value_head_dim cfg.rms_norm_eps
    out_proj := outProj
  }

private def repeatHeads {batch seq n_head head_dim rep : UInt64}
    (x : Tensor { shape := #[batch, seq, n_head, head_dim] })
    : Tensor { shape := #[batch, seq, n_head * rep, head_dim] } :=
  if rep == 1 then
    Tensor.reshapeChecked x #[batch, seq, n_head * rep, head_dim]
  else
    let x5 : Tensor { shape := #[batch, seq, n_head, 1, head_dim] } := x.reshapeChecked #[batch, seq, n_head, 1, head_dim]
    let x5 : Tensor { shape := #[batch, seq, n_head, rep, head_dim] } := Tensor.expand x5 #[batch, seq, n_head, rep, head_dim]
    Tensor.reshapeChecked x5 #[batch, seq, n_head * rep, head_dim]

/-- Run linear token mixer. If `cacheConv`/`cacheRec` are provided and
    `usePrecomputedState=true`, performs incremental one-step update. -/
def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35GatedDeltaNet cfg)
    (hidden : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (attentionMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64 }) := none)
    (cacheConv : Option (Tensor { shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim] }) := none)
    (cacheRec : Option (Tensor { shape := #[batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim], dtype := .Float32 }) := none)
    (usePrecomputedState : Bool := false)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] }
      × Option (Tensor { shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim] })
      × Option (Tensor { shape := #[batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim], dtype := .Float32 }) :=
  let hidden := applyMaskToPaddingStates hidden attentionMask
  let device := hidden.device

  let mixedQKV : Tensor { shape := #[batch, seq, Config.linearConvDim cfg] } := Tensor.linear3d hidden m.in_proj_qkv
  let mixedQKVc : Tensor { shape := #[batch, Config.linearConvDim cfg, seq] } := transposeSeq mixedQKV

  let zFlat : Tensor { shape := #[batch, seq, Config.linearValueDim cfg] } := Tensor.linear3d hidden m.in_proj_z
  let z : Tensor { shape := #[batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim] } :=
    Tensor.reshapeChecked zFlat #[batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim]

  let b : Tensor { shape := #[batch, seq, cfg.linear_num_value_heads] } := Tensor.linear3d hidden m.in_proj_b
  let a : Tensor { shape := #[batch, seq, cfg.linear_num_value_heads] } := Tensor.linear3d hidden m.in_proj_a

  let (convOut, nextConvState) :
      Tensor { shape := #[batch, Config.linearConvDim cfg, seq] }
        × Option (Tensor { shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim] }) :=
    if usePrecomputedState && seq == 1 then
      match cacheConv with
      | some st =>
        let catIn : Tensor { shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim + 1] } := typedCat st mixedQKVc 2
        let convIn : Tensor { shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim + 1] } :=
          Tensor.castLike m.conv1d_weight catIn
        let stNext : Tensor { shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim] } :=
          typedSlice catIn 2 1 cfg.linear_conv_kernel_dim
        let raw : Tensor { shape := #[batch, Config.linearConvDim cfg, 2] } :=
          convGroups convIn m.conv1d_weight m.conv1d_bias 1 0 1 (Config.linearConvDim cfg)
        let y : Tensor { shape := #[batch, Config.linearConvDim cfg, 1] } := typedSlice raw 2 1 1
        ((Tensor.Preserving.silu y).reshapeChecked #[batch, Config.linearConvDim cfg, seq], some stNext)
      | none =>
        let convIn : Tensor { shape := #[batch, Config.linearConvDim cfg, seq] } :=
          Tensor.castLike m.conv1d_weight mixedQKVc
        let raw : Tensor { shape := #[batch, Config.linearConvDim cfg, seq + cfg.linear_conv_kernel_dim - 1] } :=
          convGroups convIn m.conv1d_weight m.conv1d_bias 1 (cfg.linear_conv_kernel_dim - 1) 1 (Config.linearConvDim cfg)
        let y : Tensor { shape := #[batch, Config.linearConvDim cfg, seq] } := typedSlice raw 2 0 seq
        let nextState :=
          if seq >= cfg.linear_conv_kernel_dim then
            some (typedSlice mixedQKVc 2 (seq - cfg.linear_conv_kernel_dim) cfg.linear_conv_kernel_dim)
          else
            let padLen : UInt64 := cfg.linear_conv_kernel_dim - seq
            let z0 : Tensor { shape := #[batch, Config.linearConvDim cfg, padLen] } := zerosDynamicOn device
            some (typedCat z0 mixedQKVc 2)
        (Tensor.Preserving.silu y, nextState)
    else
      let convIn : Tensor { shape := #[batch, Config.linearConvDim cfg, seq] } :=
        Tensor.castLike m.conv1d_weight mixedQKVc
      let raw : Tensor { shape := #[batch, Config.linearConvDim cfg, seq + cfg.linear_conv_kernel_dim - 1] } :=
        convGroups convIn m.conv1d_weight m.conv1d_bias 1 (cfg.linear_conv_kernel_dim - 1) 1 (Config.linearConvDim cfg)
      let y : Tensor { shape := #[batch, Config.linearConvDim cfg, seq] } := typedSlice raw 2 0 seq
      let nextState :=
        if seq >= cfg.linear_conv_kernel_dim then
          some (typedSlice mixedQKVc 2 (seq - cfg.linear_conv_kernel_dim) cfg.linear_conv_kernel_dim)
        else
          let padLen : UInt64 := cfg.linear_conv_kernel_dim - seq
          let z0 : Tensor { shape := #[batch, Config.linearConvDim cfg, padLen] } := zerosDynamicOn device
          some (typedCat z0 mixedQKVc 2)
      (Tensor.Preserving.silu y, nextState)

  let mixed : Tensor { shape := #[batch, seq, Config.linearConvDim cfg] } := transposeSeq convOut

  let keyDim : UInt64 := Config.linearKeyDim cfg
  let valDim : UInt64 := Config.linearValueDim cfg

  let qRaw : Tensor { shape := #[batch, seq, keyDim] } := typedSlice mixed 2 0 keyDim
  let kRaw : Tensor { shape := #[batch, seq, keyDim] } := typedSlice mixed 2 keyDim keyDim
  let vRaw : Tensor { shape := #[batch, seq, valDim] } := typedSlice mixed 2 (keyDim * 2) valDim

  let q0 : Tensor { shape := #[batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim] } :=
    Tensor.reshapeChecked qRaw #[batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim]
  let k0 : Tensor { shape := #[batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim] } :=
    Tensor.reshapeChecked kRaw #[batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim]
  let v : Tensor { shape := #[batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim] } :=
    Tensor.reshapeChecked vRaw #[batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim]

  let beta : Tensor { shape := #[batch, seq, cfg.linear_num_value_heads] } := Tensor.Preserving.sigmoid b

  let dt : Tensor { shape := #[batch, seq, cfg.linear_num_value_heads] } :=
    Tensor.expand (Tensor.reshapeChecked m.dt_bias #[1, 1, cfg.linear_num_value_heads]) #[batch, seq, cfg.linear_num_value_heads]
  let aExp : Tensor { shape := #[batch, seq, cfg.linear_num_value_heads] } :=
    Tensor.expand (Tensor.reshapeChecked (Tensor.Preserving.exp (floatDynamic m.a_log)) #[1, 1, cfg.linear_num_value_heads]) #[batch, seq, cfg.linear_num_value_heads]
  let gPos : Tensor { shape := #[batch, seq, cfg.linear_num_value_heads] } := softplusDynamic (floatDynamic (a + dt))
  let g : Tensor { shape := #[batch, seq, cfg.linear_num_value_heads] } := Tensor.Preserving.mulScalar (gPos * aExp) (-1.0)

  let rep := Config.linearKVRepeat cfg
  let q : Tensor { shape := #[batch, seq, cfg.linear_num_value_heads, cfg.linear_key_head_dim] } :=
    (repeatHeads (rep := rep) q0).reshapeChecked #[batch, seq, cfg.linear_num_value_heads, cfg.linear_key_head_dim]
  let k : Tensor { shape := #[batch, seq, cfg.linear_num_value_heads, cfg.linear_key_head_dim] } :=
    (repeatHeads (rep := rep) k0).reshapeChecked #[batch, seq, cfg.linear_num_value_heads, cfg.linear_key_head_dim]

  let (core, finalState) :=
    if usePrecomputedState then
      match cacheRec with
      | some rec =>
        recurrentGatedDeltaRule q k v g beta rec
      | none =>
        chunkGatedDeltaRule q k v g beta
    else
      chunkGatedDeltaRule q k v g beta

  let core2d : Tensor { shape := #[batch * seq * cfg.linear_num_value_heads, cfg.linear_value_head_dim] } :=
    Tensor.reshapeChecked core #[batch * seq * cfg.linear_num_value_heads, cfg.linear_value_head_dim]
  let z2d : Tensor { shape := #[batch * seq * cfg.linear_num_value_heads, cfg.linear_value_head_dim] } :=
    Tensor.reshapeChecked z #[batch * seq * cfg.linear_num_value_heads, cfg.linear_value_head_dim]
  let core2d := m.norm.forward2d core2d z2d
  let core3d : Tensor { shape := #[batch, seq, valDim] } := Tensor.reshapeChecked core2d #[batch, seq, valDim]

  let out : Tensor { shape := #[batch, seq, cfg.hidden_size] } := Tensor.linear3d core3d m.out_proj
  (out, nextConvState, some finalState)

end Qwen35GatedDeltaNet

/-- Layer-local cache bundle for hybrid token mixers. -/
structure HybridCache (cfg : Config) (batch : UInt64) where
  attnCaches : Array (Option (Qwen35Attention.KVCache cfg batch))
  convStates : Array (Option (Tensor { shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim] }))
  recurrentStates : Array (Option (Tensor { shape := #[batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim], dtype := .Float32 }))
  lastLinearLayer : Option Nat := none

namespace HybridCache

def hasPreviousState (c : HybridCache cfg batch) : Bool :=
  match c.lastLinearLayer with
  | none => false
  | some idx =>
    match c.convStates[idx]? with
    | some (some _) => true
    | _ => false

end HybridCache

/-- One Qwen3.5 decoder layer (hybrid token mixer + dense/MoE FFN). -/
structure Qwen35Layer (cfg : Config) where
  layerType : LayerType
  input_layernorm : Qwen35RMSNorm cfg.hidden_size
  full_attn : Option (Qwen35Attention cfg) := none
  linear_attn : Option (Qwen35GatedDeltaNet cfg) := none
  post_attention_layernorm : Qwen35RMSNorm cfg.hidden_size
  dense_mlp : Option (Qwen35MLP cfg.hidden_size cfg.intermediate_size) := none
  sparse_moe : Option (Qwen35SparseMoeBlock cfg) := none
  deriving TensorStruct

namespace Qwen35Layer

def init (cfg : Config) (layerType : LayerType) : IO (Qwen35Layer cfg) := do
  let inputNorm := Qwen35RMSNorm.initZeroCentered cfg.hidden_size cfg.rms_norm_eps
  let postNorm := Qwen35RMSNorm.initZeroCentered cfg.hidden_size cfg.rms_norm_eps

  let fullAttn ←
    match layerType with
    | .fullAttention =>
      let m ← Qwen35Attention.init cfg
      pure (some m)
    | .linearAttention => pure none

  let linearAttn ←
    match layerType with
    | .linearAttention =>
      let m ← Qwen35GatedDeltaNet.init cfg
      pure (some m)
    | .fullAttention => pure none

  let denseMlp ←
    if Config.isMoE cfg then
      pure none
    else
      let m ← Qwen35MLP.init cfg.hidden_size cfg.intermediate_size
      pure (some m)

  let sparseMoe ←
    if Config.isMoE cfg then
      let m ← Qwen35SparseMoeBlock.init cfg
      pure (some m)
    else
      pure none

  pure {
    layerType := layerType
    input_layernorm := inputNorm
    full_attn := fullAttn
    linear_attn := linearAttn
    post_attention_layernorm := postNorm
    dense_mlp := denseMlp
    sparse_moe := sparseMoe
  }

def forward {batch seq : UInt64}
    (cfg : Config)
    (layer : Qwen35Layer cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64 }) := none)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x
  let mixed :=
    match layer.layerType, layer.full_attn, layer.linear_attn, attnMask with
    | .fullAttention, some a, _, some m => a.forwardMasked cfg h1 cos sin m true
    | .fullAttention, some a, _, none => a.forward cfg h1 cos sin true
    | .linearAttention, _, some l, _ =>
      let (y, _, _) := l.forward cfg h1 attnMask none none false
      y
    | _, _, _, _ => h1
  let h2 := residual1 + mixed

  let residual2 := h2
  let h3 := layer.post_attention_layernorm.forward3d h2
  let ffn :=
    match layer.dense_mlp, layer.sparse_moe with
    | some mlp, _ => mlp.forward h3
    | _, some moe => moe.forward cfg h3
    | _, _ => h3
  residual2 + ffn

/-- Forward pass with optional cache update (for batch prefill). -/
def forwardWithCache {batch seq : UInt64}
    (cfg : Config)
    (layer : Qwen35Layer cfg)
    (x : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (cache : HybridCache cfg batch)
    (layerIdx : Nat)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } × HybridCache cfg batch :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x

  let (mixed, cache') :=
    match layer.layerType with
    | .fullAttention =>
      match layer.full_attn, cache.attnCaches[layerIdx]? with
      | some a, some (some kv) =>
        let (out, kv') := a.forwardWithCache cfg h1 cos sin kv
        let c' := { cache with attnCaches := cache.attnCaches.modify layerIdx (fun _ => some kv') }
        (out, c')
      | _, _ => (h1, cache)
    | .linearAttention =>
      match layer.linear_attn with
      | some l =>
        let convOpt := (cache.convStates[layerIdx]?).bind id
        let recOpt := (cache.recurrentStates[layerIdx]?).bind id
        -- Batch prefill for linear attention just uses standard forward and updates state
        let (out, conv', rec') := l.forward cfg h1 none convOpt recOpt false
        let c' := { cache with
          convStates := cache.convStates.modify layerIdx (fun _ => conv')
          recurrentStates := cache.recurrentStates.modify layerIdx (fun _ => rec')
        }
        (out, c')
      | none => (h1, cache)

  let h2 := residual1 + mixed
  let residual2 := h2
  let h3 := layer.post_attention_layernorm.forward3d h2
  let ffn :=
    match layer.dense_mlp, layer.sparse_moe with
    | some mlp, _ => mlp.forward h3
    | _, some moe => moe.forward cfg h3
    | _, _ => h3
  (residual2 + ffn, cache')

def forwardStep {batch : UInt64}
    (cfg : Config)
    (layer : Qwen35Layer cfg)
    (x : Tensor { shape := #[batch, 1, cfg.hidden_size] })
    (cos : Tensor { shape := #[1, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { shape := #[1, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (cache : HybridCache cfg batch)
    (layerIdx : Nat)
    : Tensor { shape := #[batch, 1, cfg.hidden_size] } × HybridCache cfg batch :=
  let residual1 := x
  let h1 := layer.input_layernorm.forward3d x

  let (mixed, cache1) :=
    match layer.layerType with
    | .fullAttention =>
      match layer.full_attn, cache.attnCaches[layerIdx]? with
      | some a, some (some kv) =>
        let (out, kv') := a.forwardStep cfg h1 cos sin kv
        let c' := { cache with attnCaches := cache.attnCaches.set! layerIdx (some kv') }
        (out, c')
      | _, _ =>
        (h1, cache)
    | .linearAttention =>
      match layer.linear_attn with
      | some l =>
        let convOpt := cache.convStates[layerIdx]?.getD none
        let recOpt := cache.recurrentStates[layerIdx]?.getD none
        let usePrev := cache.hasPreviousState && convOpt.isSome && recOpt.isSome
        let (out, conv', rec') := l.forward cfg h1 none convOpt recOpt usePrev
        let c' := {
          cache with
          convStates := cache.convStates.set! layerIdx conv'
          recurrentStates := cache.recurrentStates.set! layerIdx rec'
        }
        (out, c')
      | none => (h1, cache)

  let h2 := residual1 + mixed
  let residual2 := h2
  let h3 := layer.post_attention_layernorm.forward3d h2
  let ffn :=
    match layer.dense_mlp, layer.sparse_moe with
    | some mlp, _ => mlp.forward h3
    | _, some moe => moe.forward cfg h3
    | _, _ => h3
  (residual2 + ffn, cache1)

end Qwen35Layer

/-- Text-only Qwen3.5 base model. -/
structure Qwen35Model (cfg : Config) where
  embed_tokens : Tensor { shape := #[cfg.vocab_size, cfg.hidden_size] }
  layers : Array (Qwen35Layer cfg)
  norm : Qwen35RMSNorm cfg.hidden_size
  deriving TensorStruct

namespace Qwen35Model

def init (cfg : Config) : IO (Qwen35Model cfg) := do
  let embRaw := (← Tensor.randn #[cfg.vocab_size, cfg.hidden_size]).forgetDType
  let embedTokens := Tensor.setRequiresGrad (Tensor.Preserving.mulScalar embRaw 0.02) true

  let layerTypes := Config.normalizedLayerTypes cfg
  let mut layers : Array (Qwen35Layer cfg) := #[]
  for i in [:cfg.num_hidden_layers.toNat] do
    let lt := layerTypes.getD i .linearAttention
    let l ← Qwen35Layer.init cfg lt
    layers := layers.push l

  let norm := Qwen35RMSNorm.initZeroCentered cfg.hidden_size cfg.rms_norm_eps
  pure { embed_tokens := embedTokens, layers := layers, norm := norm }

def initCache {batch : UInt64}
    (cfg : Config)
    (m : Qwen35Model cfg)
    (maxLen : UInt64)
    (device : Device)
    : HybridCache cfg batch :=
  Id.run do
    let layerTypes := Config.normalizedLayerTypes cfg
    let mut attn : Array (Option (Qwen35Attention.KVCache cfg batch)) := #[]
    let mut conv : Array (Option (Tensor { shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim] })) := #[]
    let mut recStates : Array (Option (Tensor { shape := #[batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim], dtype := .Float32 })) := #[]
    let mut lastLinear : Option Nat := none

    for i in [:m.layers.size] do
      let lt := layerTypes.getD i .linearAttention
      match lt with
      | .fullAttention =>
        let kv := qwen.QwenAttention.initKVCache
          maxLen
          (batch := batch)
          (num_kv_heads := cfg.num_key_value_heads)
          (head_dim := cfg.head_dim)
          device (.Unknown "dynamic")
        attn := attn.push (some kv)
        conv := conv.push none
        recStates := recStates.push none
      | .linearAttention =>
        attn := attn.push none
        conv := conv.push none
        recStates := recStates.push none
        lastLinear := some i

    return {
      attnCaches := attn
      convStates := conv
      recurrentStates := recStates
      lastLinearLayer := lastLinear
    }

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35Model cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64 }) := none)
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  let x0 : Tensor { shape := #[batch, seq, cfg.hidden_size] } := Tensor.embedding inputIds m.embed_tokens
  let (cos, sin) := Tensor.rotaryFreqs seq (Config.rotaryDim cfg) cfg.rope_theta x0.device
  let h :=
    m.layers.foldl
      (fun h layer => layer.forward cfg h cos sin attnMask)
      x0
  m.norm.forward3d h

def forwardWithCache {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35Model cfg)
    (x0 : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cos : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (cache : HybridCache cfg batch)
    : IO (Tensor { shape := #[batch, seq, cfg.hidden_size] } × HybridCache cfg batch) := do
  let mut h := x0
  let mut c := cache
  for i in [:m.layers.size] do
    let layer ← match m.layers[i]? with
      | some l => pure l
      | none => throw <| IO.userError s!"missing layer at index {i}"
    let (hNext, cNext) := layer.forwardWithCache cfg h cos sin c i
    h := hNext
    c := cNext
  return (m.norm.forward3d h, c)


end Qwen35Model

/-- Causal-LM output container. -/
structure CausalLMOutput (cfg : Config) (batch seq : UInt64) where
  logits : Tensor { shape := #[batch, seq, cfg.vocab_size] }

/-- Full standalone Qwen3.5 causal language model (dense or MoE depending on config). -/
structure Qwen35ForCausalLM (cfg : Config) where
  model : Qwen35Model cfg
  lmHead : Tensor { shape := #[cfg.vocab_size, cfg.hidden_size] }
  tieWordEmbeddings : Bool := true
  deriving TensorStruct

namespace Qwen35ForCausalLM

def init (cfg : Config) (tieWordEmbeddings : Bool := true) : IO (Qwen35ForCausalLM cfg) := do
  let model ← Qwen35Model.init cfg
  let lmHead ←
    if tieWordEmbeddings then
      pure model.embed_tokens
    else
      initDynamicWeight #[cfg.vocab_size, cfg.hidden_size] cfg.hidden_size
  pure { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }

def embedTokens {batch seq : UInt64}
    (m : Qwen35ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    : Tensor { shape := #[batch, seq, cfg.hidden_size] } :=
  Tensor.embedding inputIds m.model.embed_tokens

def forwardEmbeds {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64 }) := none)
    : Tensor { shape := #[batch, seq, cfg.vocab_size] } :=
  let (cos, sin) := Tensor.rotaryFreqs seq (Config.rotaryDim cfg) cfg.rope_theta inputsEmbeds.device
  let hidden := m.model.layers.foldl (fun h layer => layer.forward cfg h cos sin attnMask) inputsEmbeds
  let hidden := m.model.norm.forward3d hidden
  Tensor.linear3d hidden m.lmHead

def forward {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (attnMask : Option (Tensor { shape := #[batch, seq], dtype := .Int64 }) := none)
    : Tensor { shape := #[batch, seq, cfg.vocab_size] } :=
  let embeds := m.embedTokens inputIds
  m.forwardEmbeds cfg embeds attnMask

private def precomputeDecodeRotary {maxLen : UInt64}
    (cfg : Config)
    (device : Device)
    : Tensor { shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 } × Tensor { shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 } :=
  Tensor.rotaryFreqs
    maxLen
    (Config.rotaryDim cfg)
    cfg.rope_theta
    device

/-- Single-pass prompt prefill for KV-cache. -/
private def prefillCachesFromEmbeds {batch seq maxLen : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (cosAll : Tensor { shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sinAll : Tensor { shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (cache : HybridCache cfg batch)
    : IO (Tensor { shape := #[batch, cfg.vocab_size] } × HybridCache cfg batch) := do
  let cos : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 } := typedSlice cosAll 0 0 seq
  let sin : Tensor { shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 } := typedSlice sinAll 0 0 seq

  let (hidden, cache') ← Qwen35Model.forwardWithCache cfg m.model inputsEmbeds cos sin cache

  let lastHidden : Tensor { shape := #[batch, 1, cfg.hidden_size] } := typedSlice hidden 1 (seq - 1) 1
  let logits3 : Tensor { shape := #[batch, 1, cfg.vocab_size] } := Tensor.linear3d lastHidden m.lmHead
  let logits : Tensor { shape := #[batch, cfg.vocab_size] } := Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size]
  pure (logits, cache')

private def decodeStepFromEmbedWithCache {batch maxLen : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (cosAll : Tensor { shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sinAll : Tensor { shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (tokenEmbed : Tensor { shape := #[batch, 1, cfg.hidden_size] })
    (position : UInt64)
    (cache : HybridCache cfg batch)
    : IO (Tensor { shape := #[batch, cfg.vocab_size] } × HybridCache cfg batch) := do
  let cos : Tensor { shape := #[1, Config.rotaryHalfDim cfg], dtype := .Float32 } := typedSlice cosAll 0 position 1
  let sin : Tensor { shape := #[1, Config.rotaryHalfDim cfg], dtype := .Float32 } := typedSlice sinAll 0 position 1

  let mut hidden : Tensor { shape := #[batch, 1, cfg.hidden_size] } := tokenEmbed
  let mut cache' := cache

  for i in [:m.model.layers.size] do
    let layer ←
      match m.model.layers[i]? with
      | some l => pure l
      | none => throw <| IO.userError s!"missing Qwen35 layer at index {i}"
    let (hNext, cNext) := layer.forwardStep cfg hidden cos sin cache' i
    hidden := hNext
    cache' := cNext

  let hiddenNorm := m.model.norm.forward3d hidden
  let logits3 : Tensor { shape := #[batch, 1, cfg.vocab_size] } := Tensor.linear3d hiddenNorm m.lmHead
  let logits2 : Tensor { shape := #[batch, cfg.vocab_size] } := Tensor.reshapeChecked logits3 #[batch, cfg.vocab_size]
  pure (logits2, cache')

private partial def decodeLoopCached {batch maxLen : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (cosAll : Tensor { shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sinAll : Tensor { shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (finished : Tensor { shape := #[batch], dtype := .Bool })
    (remaining : Nat)
    (cache : HybridCache cfg batch)
    (lastLogits : Tensor { shape := #[batch, cfg.vocab_size] })
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
    let nextEmb : Tensor { shape := #[batch, 1, cfg.hidden_size] } := m.embedTokens nextCol
    let (nextLogits, cache') ← decodeStepFromEmbedWithCache cfg m cosAll sinAll nextEmb curSeq cache
    decodeLoopCached cfg m cosAll sinAll strategy eosTokenIds finished' (remaining - 1) cache' nextLogits onStep (generatedSoFar + 1) appended

private partial def decodeLoopUncached {batch : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (finished : Tensor { shape := #[batch], dtype := .Bool })
    (remaining : Nat)
    {curSeq : UInt64}
    (curIds : Tensor { shape := #[batch, curSeq], dtype := .Int64 })
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩
  if curSeq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"

  let logits := m.forward cfg curIds none
  let lastPos := curSeq - 1
  let last3 : Tensor { shape := #[batch, 1, cfg.vocab_size] } := typedSlice logits 1 lastPos 1
  let last2 : Tensor { shape := #[batch, cfg.vocab_size] } := Tensor.reshapeChecked last3 #[batch, cfg.vocab_size]

  let nextTokRaw ← sampleFromLogits last2.toFloat32 strategy
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
    decodeLoopUncached cfg m strategy eosTokenIds finished' (remaining - 1) appended

private partial def decodeLoopUncachedEmbeds {batch : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (finished : Tensor { shape := #[batch], dtype := .Bool })
    (remaining : Nat)
    {curSeq : UInt64}
    (curIds : Tensor { shape := #[batch, curSeq], dtype := .Int64 })
    (curEmbeds : Tensor { shape := #[batch, curSeq, cfg.hidden_size] })
    (onStep : Option (StreamCallback batch))
    (generatedSoFar : UInt64)
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩
  if curSeq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"

  let logits := m.forwardEmbeds cfg curEmbeds none
  let lastPos := curSeq - 1
  let last3 : Tensor { shape := #[batch, 1, cfg.vocab_size] } :=
    typedSlice logits 1 lastPos 1
  let last2 : Tensor { shape := #[batch, cfg.vocab_size] } := Tensor.reshapeChecked last3 #[batch, cfg.vocab_size]

  let nextTokRaw ← sampleFromLogits last2.toFloat32 strategy
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
  let appendedIds : Tensor { shape := #[batch, curSeq + 1], dtype := .Int64 } := typedCat curIds nextCol 1

  let stop :=
    if eosTokenIds.isEmpty then
      false
    else
      !(Tensor.any (Tensor.logicalNot finished'))
  if stop then
    return ⟨curSeq + 1, appendedIds⟩
  else
    let nextEmb : Tensor { shape := #[batch, 1, cfg.hidden_size] } := m.embedTokens nextCol
    let appendedEmbeds : Tensor { shape := #[batch, curSeq + 1, cfg.hidden_size] } := typedCat curEmbeds nextEmb 1
    decodeLoopUncachedEmbeds
      cfg
      m
      strategy
      eosTokenIds
      finished'
      (remaining - 1)
      appendedIds
      appendedEmbeds
      onStep
      (generatedSoFar + 1)

/-- Exact generation entry point using KV-caching. -/
private def generateFromEmbedsCore {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
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
  let (cosAll, sinAll) := Tensor.rotaryFreqs maxLen (Config.rotaryDim cfg) cfg.rope_theta inputsEmbeds.device
  let cache := Qwen35Model.initCache cfg m.model maxLen inputsEmbeds.device

  -- 1) Prefill caches and get last prompt logits
  let (logits, caches) ← prefillCachesFromEmbeds cfg m cosAll sinAll inputsEmbeds cache

  -- 2) Start incremental decode loop
  let finished0 : Tensor { shape := #[batch], dtype := .Bool } := falseMaskTyped (n := batch) inputIds.device
  let finalIds ← decodeLoopCached
    (maxLen := maxLen)
    cfg m cosAll sinAll strategy eosTokenIds finished0 maxNewTokens.toNat
    caches logits onStep 0 inputIds

  pure finalIds


/-- Exact generation from explicit input embeddings.
    Useful for multimodal wrappers that inject vision features into token embeddings. -/
def generateFromEmbeds {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds none

/-- Exact generation from explicit input embeddings with per-step token callback. -/
def generateFromEmbedsStream {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { shape := #[batch, seq, cfg.hidden_size] })
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds (some onStep)

/-- Exact generation entry point for token IDs. -/
def generate {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
  let inputsEmbeds := m.embedTokens inputIds
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds none

/-- Exact generation with per-step token callback (streaming). -/
def generateStream {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (onStep : StreamCallback batch)
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
  let inputsEmbeds := m.embedTokens inputIds
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds (some onStep)

/-- Reference generation path by full re-forward each decode step. -/
def generateUncached {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) := do
  let finished0 : Tensor { shape := #[batch], dtype := .Bool } := falseMaskTyped (n := batch) inputIds.device
  decodeLoopUncached cfg m strategy eosTokenIds finished0 maxNewTokens.toNat inputIds

/-- Convenience wrapper for greedy generation. -/
def generateGreedy {batch seq : UInt64}
    (cfg : Config)
    (m : Qwen35ForCausalLM cfg)
    (inputIds : Tensor { shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : UInt64 := 256)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { shape := #[batch, outSeq], dtype := .Int64 })) :=
  generate cfg m inputIds maxNewTokens .greedy eosTokenIds

end Qwen35ForCausalLM

end torch.qwen35
