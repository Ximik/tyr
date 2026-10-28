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

variable {device : Device} {dtype : DType}
variable [DType.RealFloating dtype]
variable [DType.ConstantSupported dtype device]

open torch
open torch.Model

private def l2NormLast4d {b s h d : Nat}
    (x : Tensor { device := device, shape := #[b, s, h, d], dtype := dtype })
    (eps : Float := 1e-6)
    : Tensor { device := device, shape := #[b, s, h, d], dtype := dtype } :=
  let flat : Tensor { device := device, shape := #[b * s * h, d], dtype := dtype } := Tensor.reshape x #[b * s * h, d]
  let n2 : Tensor { device := device, shape := #[b * s * h, 1], dtype := dtype } := by simpa using Tensor.sumDim (Tensor.mul flat flat) 1 true
  let inv : Tensor { device := device, shape := #[b * s * h, 1], dtype := dtype } := rsqrtDynamic (Tensor.addScalar n2 eps)
  let inv : Tensor { device := device, shape := #[b * s * h, d], dtype := dtype } := Tensor.expand inv #[b * s * h, d]
  Tensor.reshape (Tensor.mul flat inv) #[b, s, h, d]

private def applyMaskToPaddingStates {batch seq hidden : Nat}
    (x : Tensor { device := device, shape := #[batch, seq, hidden], dtype := dtype })
    (attentionMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }))
    : Tensor { device := device, shape := #[batch, seq, hidden], dtype := dtype } :=
  if batch <= 1 || seq <= 1 then
    x
  else
    match attentionMask with
    | none => x
    | some m =>
      -- Keep the integer mask: native promotion with floating activations
      -- retains their dtype instead of introducing an unnecessary Float32 value.
      let me : Tensor { device := device, shape := #[batch, seq, hidden], dtype := .Int64 } :=
        Tensor.expand (Tensor.reshape m #[batch, seq, 1]) #[batch, seq, hidden]
      Tensor.castLike x (promotedMul x me).2

/-- Qwen3.5 RMSNorm with `(1 + weight)` scaling. -/
structure Qwen35RMSNorm (dim : Nat) where
  weight : SomeDTypeTensor #[dim] device
  eps : Float := 1e-6
  deriving TensorStruct

namespace Qwen35RMSNorm

def initZeroCentered (dim : Nat) (eps : Float := 1e-6) (device : Device := .CPU) : IO (Qwen35RMSNorm (device := device) dim) := do
  IO.ofExcept (Shape.validateAllocation #[dim])
  pure { weight := ⟨.Float32, (← Tensor.parameter (Tensor.zeros #[dim] (device := device)))⟩, eps }

def initOneCentered (dim : Nat) (eps : Float := 1e-6) (device : Device := .CPU) : IO (Qwen35RMSNorm (device := device) dim) := do
  IO.ofExcept (Shape.validateAllocation #[dim])
  pure { weight := ⟨.Float32, (← Tensor.parameter (Tensor.ones #[dim] (device := device)))⟩, eps }

/-- HuggingFace Qwen3.5 checkpoints already store the zero-centered RMSNorm
    parameter used by the `(1 + weight)` formulation, so the raw tensor should
    be preserved as-is. -/
def fromCheckpointWeight {dim : Nat}
    (weight : SomeDTypeTensor #[dim] device)
    (eps : Float := 1e-6)
    : Qwen35RMSNorm (device := device) dim :=
  { weight := weight, eps }

def forward2d {n dim : Nat}
    (m : Qwen35RMSNorm (device := device) dim)
    (x : Tensor { device := device, shape := #[n, dim], dtype := dtype })
    : Tensor { device := device, shape := #[n, dim], dtype := dtype } :=
  let xf : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := floatDynamic x
  let var : Tensor { device := device, shape := #[n, 1], dtype := .Float32 } := Tensor.meanDim (Tensor.mul xf xf) 1 true
  let inv : Tensor { device := device, shape := #[n, 1], dtype := .Float32 } := rsqrtDynamic (Tensor.addScalar var m.eps)
  let inv : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.expand inv #[n, dim]
  let normed : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.mul xf inv
  let scale1 : Tensor { device := device, shape := #[dim], dtype := .Float32 } := Tensor.addScalar (floatDynamic m.weight.2) 1.0
  let scale : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.expand (Tensor.reshape scale1 #[1, dim]) #[n, dim]
  Tensor.castLike x (Tensor.mul normed scale)

def forward3d {batch seq dim : Nat}
    (m : Qwen35RMSNorm (device := device) dim)
    (x : Tensor { device := device, shape := #[batch, seq, dim], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, dim], dtype := dtype } :=
  let flat : Tensor { device := device, shape := #[batch * seq, dim], dtype := dtype } := Tensor.reshape x #[batch * seq, dim]
  Tensor.reshape (forward2d m flat) #[batch, seq, dim]

end Qwen35RMSNorm

/-- Gated RMSNorm used at linear-attention output. -/
structure Qwen35RMSNormGated (dim : Nat) where
  weight : SomeDTypeTensor #[dim] device
  eps : Float := 1e-6
  deriving TensorStruct

namespace Qwen35RMSNormGated

def init (dim : Nat) (eps : Float := 1e-6) (device : Device := .CPU) : IO (Qwen35RMSNormGated (device := device) dim) := do
  IO.ofExcept (Shape.validateAllocation #[dim])
  pure { weight := ⟨.Float32, (← Tensor.parameter (Tensor.ones #[dim] (device := device)))⟩, eps }

def forward2d {n dim : Nat}
    (m : Qwen35RMSNormGated (device := device) dim)
    (x : Tensor { device := device, shape := #[n, dim], dtype := dtype })
    (gate : Tensor { device := device, shape := #[n, dim], dtype := dtype })
    : Tensor { device := device, shape := #[n, dim], dtype := dtype } :=
  let xf : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := floatDynamic x
  let var : Tensor { device := device, shape := #[n, 1], dtype := .Float32 } := Tensor.meanDim (Tensor.mul xf xf) 1 true
  let inv : Tensor { device := device, shape := #[n, 1], dtype := .Float32 } := rsqrtDynamic (Tensor.addScalar var m.eps)
  let inv : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.expand inv #[n, dim]
  let normed : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.mul xf inv
  let scale : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.expand (Tensor.reshape (floatDynamic m.weight.2) #[1, dim]) #[n, dim]
  let g : Tensor { device := device, shape := #[n, dim], dtype := .Float32 } := Tensor.silu (floatDynamic gate)
  Tensor.castLike x (Tensor.mul (Tensor.mul normed scale) g)

end Qwen35RMSNormGated

/-- SwiGLU MLP block (dense FFN). -/
structure Qwen35MLP (hidden_size intermediate_size : Nat) (dtype : DType := .Float32) where
  gate_proj : Tensor { device := device, shape := #[intermediate_size, hidden_size], dtype := dtype }
  up_proj : Tensor { device := device, shape := #[intermediate_size, hidden_size], dtype := dtype }
  down_proj : Tensor { device := device, shape := #[hidden_size, intermediate_size], dtype := dtype }
  deriving TensorStruct

namespace Qwen35MLP

def init (hidden_size intermediate_size : Nat) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35MLP (dtype := dtype) (device := device) hidden_size intermediate_size) := do
  let gate ← initDynamicWeight (dtype := dtype) (device := device) #[intermediate_size, hidden_size] hidden_size
  let up ← initDynamicWeight (dtype := dtype) (device := device) #[intermediate_size, hidden_size] hidden_size
  let down ← initDynamicWeight (dtype := dtype) (device := device) #[hidden_size, intermediate_size] intermediate_size
  pure { gate_proj := gate, up_proj := up, down_proj := down }

def forward2d {tokens hidden inter : Nat}
    (m : Qwen35MLP (dtype := dtype) (device := device) hidden inter)
    (x : Tensor { device := device, shape := #[tokens, hidden], dtype := dtype })
    : Tensor { device := device, shape := #[tokens, hidden], dtype := dtype } :=
  let gateDyn : Tensor { device := device, shape := #[tokens, inter], dtype := dtype } := einsumChecked "oh,th->to" m.gate_proj x
  let upDyn : Tensor { device := device, shape := #[tokens, inter], dtype := dtype } := einsumChecked "oh,th->to" m.up_proj x
  let gate : Tensor { device := device, shape := #[tokens, inter], dtype := dtype } := Tensor.reshape gateDyn #[tokens, inter]
  let up : Tensor { device := device, shape := #[tokens, inter], dtype := dtype } := Tensor.reshape upDyn #[tokens, inter]
  let hid : Tensor { device := device, shape := #[tokens, inter], dtype := dtype } := Tensor.mul (Tensor.silu gate) up
  let outDyn : Tensor { device := device, shape := #[tokens, hidden], dtype := dtype } := einsumChecked "ho,to->th" m.down_proj hid
  Tensor.reshape outDyn #[tokens, hidden]

def forward {batch seq hidden inter : Nat}
    (m : Qwen35MLP (dtype := dtype) (device := device) hidden inter)
    (x : Tensor { device := device, shape := #[batch, seq, hidden], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, hidden], dtype := dtype } :=
  Tensor.linear3d (Tensor.mul (Tensor.silu (Tensor.linear3d x m.gate_proj)) (Tensor.linear3d x m.up_proj)) m.down_proj

end Qwen35MLP

/-- Top-k router for Qwen3.5 MoE. -/
structure Qwen35TopKRouter (cfg : Config) (dtype : DType := .Float32) where
  weight : Tensor { device := device, shape := #[cfg.num_experts, cfg.hidden_size], dtype := dtype }
  deriving TensorStruct

namespace Qwen35TopKRouter

def init (cfg : Config) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35TopKRouter (dtype := dtype) (device := device) cfg) := do
  let w ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.num_experts, cfg.hidden_size] cfg.hidden_size
  pure { weight := w }

def forward {tokens : Nat}
    (cfg : Config)
    (m : Qwen35TopKRouter (dtype := dtype) (device := device) cfg)
    (hidden : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[tokens, cfg.num_experts], dtype := dtype } × Tensor { device := device, shape := #[tokens, cfg.num_experts_per_tok], dtype := dtype } × Tensor { device := device, shape := #[tokens, cfg.num_experts_per_tok], dtype := .Int64 } :=
  let logitsDyn : Tensor { device := device, shape := #[tokens, cfg.num_experts], dtype := dtype } := einsumChecked "eh,th->te" m.weight hidden
  let logits : Tensor { device := device, shape := #[tokens, cfg.num_experts], dtype := dtype } := Tensor.reshape logitsDyn #[tokens, cfg.num_experts]
  let probs : Tensor { device := device, shape := #[tokens, cfg.num_experts], dtype := dtype } := Tensor.softmax logits (-1)
  let (topValsRaw, topIdx) := topK2d probs cfg.num_experts_per_tok 1
  let denom : Tensor { device := device, shape := #[tokens, 1], dtype := dtype } := by simpa using Tensor.sumDim topValsRaw 1 true
  let denom : Tensor { device := device, shape := #[tokens, cfg.num_experts_per_tok], dtype := dtype } := Tensor.expand denom #[tokens, cfg.num_experts_per_tok]
  let topVals : Tensor { device := device, shape := #[tokens, cfg.num_experts_per_tok], dtype := dtype } := by simpa using Tensor.div topValsRaw denom
  (probs, topVals, topIdx)

end Qwen35TopKRouter

/-- Expert weight bank for Qwen3.5 MoE FFN. -/
structure Qwen35MoeExperts (cfg : Config) (dtype : DType := .Float32) where
  gate_up_proj : Tensor { device := device, shape := #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size], dtype := dtype }
  down_proj : Tensor { device := device, shape := #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size], dtype := dtype }
  deriving TensorStruct

namespace Qwen35MoeExperts

def init (cfg : Config) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35MoeExperts (dtype := dtype) (device := device) cfg) := do
  let gu ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.num_experts, 2 * cfg.moe_intermediate_size, cfg.hidden_size] cfg.hidden_size
  let down ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.num_experts, cfg.hidden_size, cfg.moe_intermediate_size] cfg.moe_intermediate_size
  pure { gate_up_proj := gu, down_proj := down }

def forward2d {tokens : Nat} {routingDType : DType}
    (cfg : Config)
    (m : Qwen35MoeExperts (dtype := dtype) (device := device) cfg)
    (hidden : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype })
    (topVals : Tensor { device := device, shape := #[tokens, cfg.num_experts_per_tok], dtype := routingDType })
    (topIdx : Tensor { device := device, shape := #[tokens, cfg.num_experts_per_tok], dtype := .Int64 })
    : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
  Id.run do
    let device := hidden.device
    let mut acc : SomeDTypeTensor #[tokens, cfg.hidden_size] device :=
      ⟨.Float32, Tensor.zeros #[tokens, cfg.hidden_size] (device := device)⟩

    for slot in [:cfg.num_experts_per_tok] do
      -- Gather ONLY the routed expert's weights per token (an embedding row
      -- lookup) instead of a one-hot contraction over all `num_experts`. This is
      -- numerically identical to the one-hot selection but ~num_experts x cheaper.
      let idx2d : Tensor { device := device, shape := #[tokens, 1], dtype := .Int64 } := typedSlice topIdx 1 slot 1
      let idxRow : Tensor { device := device, shape := #[1, tokens], dtype := .Int64 } := Tensor.reshape idx2d #[1, tokens]
      let gateUp2d : Tensor { device := device, shape := #[cfg.num_experts, 2 * cfg.moe_intermediate_size * cfg.hidden_size], dtype := dtype } :=
        Tensor.reshape m.gate_up_proj #[cfg.num_experts, 2 * cfg.moe_intermediate_size * cfg.hidden_size]
      let tokGateUpE : Tensor { device := device, shape := #[1, tokens, 2 * cfg.moe_intermediate_size * cfg.hidden_size], dtype := dtype } :=
        Tensor.embedding idxRow gateUp2d
      let tokGateUp : Tensor { device := device, shape := #[tokens, 2 * cfg.moe_intermediate_size, cfg.hidden_size], dtype := dtype } :=
        Tensor.reshape tokGateUpE #[tokens, 2 * cfg.moe_intermediate_size, cfg.hidden_size]

      let guDyn : Tensor { device := device, shape := #[tokens, 2 * cfg.moe_intermediate_size], dtype := dtype } := einsumChecked "tih,th->ti" tokGateUp hidden
      let gu : Tensor { device := device, shape := #[tokens, 2 * cfg.moe_intermediate_size], dtype := dtype } := Tensor.reshape guDyn #[tokens, 2 * cfg.moe_intermediate_size]

      let gate : Tensor { device := device, shape := #[tokens, cfg.moe_intermediate_size], dtype := dtype } := typedSlice gu 1 0 cfg.moe_intermediate_size
      let up : Tensor { device := device, shape := #[tokens, cfg.moe_intermediate_size], dtype := dtype } :=
        typedSlice gu 1 cfg.moe_intermediate_size cfg.moe_intermediate_size
      let inter : Tensor { device := device, shape := #[tokens, cfg.moe_intermediate_size], dtype := dtype } := Tensor.mul (Tensor.silu gate) up

      let down2d : Tensor { device := device, shape := #[cfg.num_experts, cfg.hidden_size * cfg.moe_intermediate_size], dtype := dtype } :=
        Tensor.reshape m.down_proj #[cfg.num_experts, cfg.hidden_size * cfg.moe_intermediate_size]
      let tokDownE : Tensor { device := device, shape := #[1, tokens, cfg.hidden_size * cfg.moe_intermediate_size], dtype := dtype } :=
        Tensor.embedding idxRow down2d
      let tokDown : Tensor { device := device, shape := #[tokens, cfg.hidden_size, cfg.moe_intermediate_size], dtype := dtype } :=
        Tensor.reshape tokDownE #[tokens, cfg.hidden_size, cfg.moe_intermediate_size]

      let outDyn : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := einsumChecked "thi,ti->th" tokDown inter
      let out : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.reshape outDyn #[tokens, cfg.hidden_size]

      let w2d : Tensor { device := device, shape := #[tokens, 1], dtype := routingDType } := typedSlice topVals 1 slot 1
      let w : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := routingDType } := Tensor.expand w2d #[tokens, cfg.hidden_size]
      acc := promotedAdd acc.2 (promotedMul out w).2

    -- Restore the model dtype so downstream layers see consistent dtypes.
    return Tensor.castLike hidden acc.2

end Qwen35MoeExperts

/-- Sparse MoE block (routed experts + shared expert). -/
structure Qwen35SparseMoeBlock (cfg : Config) (dtype : DType := .Float32) where
  router : Qwen35TopKRouter (dtype := dtype) (device := device) cfg
  experts : Qwen35MoeExperts (dtype := dtype) (device := device) cfg
  shared_expert : Qwen35MLP (dtype := dtype) (device := device) cfg.hidden_size cfg.shared_expert_intermediate_size
  shared_expert_gate : Tensor { device := device, shape := #[1, cfg.hidden_size], dtype := dtype }
  deriving TensorStruct

namespace Qwen35SparseMoeBlock

def init (cfg : Config) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35SparseMoeBlock (dtype := dtype) (device := device) cfg) := do
  let router ← Qwen35TopKRouter.init (dtype := dtype) (device := device) cfg
  let experts ← Qwen35MoeExperts.init (dtype := dtype) (device := device) cfg
  let sharedExpert ← Qwen35MLP.init (dtype := dtype) (device := device) cfg.hidden_size cfg.shared_expert_intermediate_size
  let sharedGate ← initDynamicWeight (dtype := dtype) (device := device) #[1, cfg.hidden_size] cfg.hidden_size
  pure {
    router := router
    experts := experts
    shared_expert := sharedExpert
    shared_expert_gate := sharedGate
  }

def forward {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35SparseMoeBlock (dtype := dtype) (device := device) cfg)
    (hidden : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let tokens := batch * seq
  let hidden2d : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } := Tensor.reshapeChecked hidden #[tokens, cfg.hidden_size]

  let (_routerProbs, topVals, topIdx) := m.router.forward cfg hidden2d
  let expertOut : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
    m.experts.forward2d cfg hidden2d topVals topIdx

  let sharedOut : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
    m.shared_expert.forward2d hidden2d

  let gateDyn : Tensor { device := device, shape := #[tokens, 1], dtype := dtype } := einsumChecked "oh,th->to" m.shared_expert_gate hidden2d
  let gate1 : Tensor { device := device, shape := #[tokens, 1], dtype := dtype } := Tensor.reshape gateDyn #[tokens, 1]
  let gate : Tensor { device := device, shape := #[tokens, cfg.hidden_size], dtype := dtype } :=
    Tensor.expand (Tensor.sigmoid gate1) #[tokens, cfg.hidden_size]

  Tensor.reshapeChecked (Tensor.add expertOut (Tensor.mul sharedOut gate)) #[batch, seq, cfg.hidden_size]

end Qwen35SparseMoeBlock

/-- Full-attention token mixer for Qwen3.5 (Q-gate + Q/K RMSNorm + partial RoPE). -/
structure Qwen35Attention (cfg : Config) (dtype : DType := .Float32) where
  q_proj : Tensor { device := device, shape := #[cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size], dtype := dtype }
  k_proj : Tensor { device := device, shape := #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size], dtype := dtype }
  v_proj : Tensor { device := device, shape := #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size], dtype := dtype }
  o_proj : Tensor { device := device, shape := #[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim], dtype := dtype }
  q_norm : Qwen35RMSNorm (device := device) cfg.head_dim
  k_norm : Qwen35RMSNorm (device := device) cfg.head_dim
  deriving TensorStruct

namespace Qwen35Attention

abbrev KVCache (cfg : Config) (batch : Nat) (dtype : DType := .Float32) :=
  qwen.QwenAttention.KVCache (device := device) batch cfg.num_key_value_heads cfg.head_dim dtype

def init (cfg : Config) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35Attention (dtype := dtype) (device := device) cfg) := do
  let q ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.num_attention_heads * cfg.head_dim * 2, cfg.hidden_size] cfg.hidden_size
  let k ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] cfg.hidden_size
  let v ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.num_key_value_heads * cfg.head_dim, cfg.hidden_size] cfg.hidden_size
  let o ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.hidden_size, cfg.num_attention_heads * cfg.head_dim] (cfg.num_attention_heads * cfg.head_dim)
  pure {
    q_proj := q
    k_proj := k
    v_proj := v
    o_proj := o
    q_norm := (← Qwen35RMSNorm.initZeroCentered (device := device) cfg.head_dim cfg.rms_norm_eps)
    k_norm := (← Qwen35RMSNorm.initZeroCentered (device := device) cfg.head_dim cfg.rms_norm_eps)
  }

private def applyHeadNorm {batch seq n_heads head_dim : Nat}
    (norm : Qwen35RMSNorm (device := device) head_dim)
    (x : Tensor { device := device, shape := #[batch, seq, n_heads, head_dim], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, n_heads, head_dim], dtype := dtype } :=
  let flat : Tensor { device := device, shape := #[batch * seq * n_heads, head_dim], dtype := dtype } := Tensor.reshape x #[batch * seq * n_heads, head_dim]
  Tensor.reshape (norm.forward2d flat) #[batch, seq, n_heads, head_dim]

private def applyRotaryPartial {batch seq n_head head_dim rotary_dim : Nat}
    (x : Tensor { device := device, shape := #[batch, seq, n_head, head_dim], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, rotary_dim / 2], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, rotary_dim / 2], dtype := .Float32 })
    : Tensor { device := device, shape := #[batch, seq, n_head, head_dim], dtype := dtype } :=
  let xRot : Tensor { device := device, shape := #[batch, seq, n_head, rotary_dim], dtype := dtype } := typedSlice x 3 0 rotary_dim
  let xRot : Tensor { device := device, shape := #[batch, seq, n_head, rotary_dim], dtype := dtype } := Tensor.applyRotary xRot cos sin
  let xPassLen : Nat := head_dim - rotary_dim
  let xPass : Tensor { device := device, shape := #[batch, seq, n_head, xPassLen], dtype := dtype } := typedSlice x 3 rotary_dim xPassLen
  typedCat xRot xPass 3

private def splitQProjAndGate {batch seq : Nat}
    (cfg : Config)
    (qgFlat : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim * 2], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim], dtype := dtype }
      × Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
  let qg : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim * 2], dtype := dtype } :=
    Tensor.reshape qgFlat #[batch, seq, cfg.num_attention_heads, cfg.head_dim * 2]
  let q : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } :=
    typedSlice qg 3 0 cfg.head_dim
  let gate : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } :=
    typedSlice qg 3 cfg.head_dim cfg.head_dim
  let gateFlat : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim], dtype := dtype } :=
    Tensor.reshape gate #[batch, seq, cfg.num_attention_heads * cfg.head_dim]
  (q, gateFlat)

def forward {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (is_causal : Bool := true)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let qg : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim * 2], dtype := dtype } := Tensor.linear3d x m.q_proj
  let qLen : Nat := cfg.num_attention_heads * cfg.head_dim
  let (q, gateFlat) := splitQProjAndGate cfg qg

  let kFlat : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim], dtype := dtype } := Tensor.linear3d x m.k_proj
  let vFlat : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim], dtype := dtype } := Tensor.linear3d x m.v_proj

  let k : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
    Tensor.reshape kFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]
  let v : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
    Tensor.reshape vFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

  let q := applyHeadNorm m.q_norm q
  let k := applyHeadNorm m.k_norm k

  let q := applyRotaryPartial q cos sin
  let k := applyRotaryPartial k cos sin

  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } := Tensor.splitHeads q
  let kh : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype } := Tensor.splitHeads k
  let vh : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype } := Tensor.splitHeads v

  let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
    Tensor.attentionGQA qh kh vh is_causal true

  let attn : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } := Tensor.joinHeads attn
  let attnFlat : Tensor { device := device, shape := #[batch, seq, qLen], dtype := dtype } := Tensor.reshapeChecked attn #[batch, seq, qLen]
  let gated : Tensor { device := device, shape := #[batch, seq, qLen], dtype := dtype } := Tensor.mul attnFlat (Tensor.sigmoid gateFlat)
  Tensor.linear3d gated m.o_proj

/-- Forward pass with optional KV-cache update (for batch prefill). -/
def forwardWithCache {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch :=
  let qg : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim * 2], dtype := dtype } := Tensor.linear3d x m.q_proj
  let qLen : Nat := cfg.num_attention_heads * cfg.head_dim
  let (q, gateFlat) := splitQProjAndGate cfg qg

  let kFlat : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim], dtype := dtype } := Tensor.linear3d x m.k_proj
  let vFlat : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim], dtype := dtype } := Tensor.linear3d x m.v_proj

  let k : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
    Tensor.reshape kFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]
  let v : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
    Tensor.reshape vFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

  let q := applyHeadNorm m.q_norm q
  let k := applyHeadNorm m.k_norm k

  let q := applyRotaryPartial q cos sin
  let k := applyRotaryPartial k cos sin

  -- Update cache with full sequence
  let kh_new := Tensor.splitHeads k
  let vh_new := Tensor.splitHeads v
  let kStore : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim], dtype := dtype } :=
    Tensor.castLike kh_new (Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])
  let vStore : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim], dtype := dtype } :=
    Tensor.castLike vh_new (Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])

  let kStore' := Tensor.sliceScatter kStore 2 0 kh_new
  let vStore' := Tensor.sliceScatter vStore 2 0 vh_new

  let cache' : KVCache (dtype := dtype) (device := device) cfg batch := { cache with
    kStoreDyn := Tensor.packShape kStore'
    vStoreDyn := Tensor.packShape vStore'
    seq := seq
  }

  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } := Tensor.splitHeads q
  let kh : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype } := typedSlice kStore' 2 0 seq
  let vh : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype } := typedSlice vStore' 2 0 seq
  let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
    Tensor.attentionGQA qh kh vh true true

  let attn : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } := Tensor.joinHeads attn
  let attnFlat : Tensor { device := device, shape := #[batch, seq, qLen], dtype := dtype } := Tensor.reshapeChecked attn #[batch, seq, qLen]
  let gated : Tensor { device := device, shape := #[batch, seq, qLen], dtype := dtype } := Tensor.mul attnFlat (Tensor.sigmoid gateFlat)
  (Tensor.linear3d gated m.o_proj, cache')

def forwardMasked {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (attn_mask : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (is_causal : Bool := true)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let qg : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads * cfg.head_dim * 2], dtype := dtype } := Tensor.linear3d x m.q_proj
  let qLen : Nat := cfg.num_attention_heads * cfg.head_dim
  let (q, gateFlat) := splitQProjAndGate cfg qg

  let kFlat : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim], dtype := dtype } := Tensor.linear3d x m.k_proj
  let vFlat : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads * cfg.head_dim], dtype := dtype } := Tensor.linear3d x m.v_proj

  let k : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
    Tensor.reshape kFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]
  let v : Tensor { device := device, shape := #[batch, seq, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
    Tensor.reshape vFlat #[batch, seq, cfg.num_key_value_heads, cfg.head_dim]

  let q := applyHeadNorm m.q_norm q
  let k := applyHeadNorm m.k_norm k

  let q := applyRotaryPartial q cos sin
  let k := applyRotaryPartial k cos sin

  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } := Tensor.splitHeads q
  let kh : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype } := Tensor.splitHeads k
  let vh : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, seq, cfg.head_dim], dtype := dtype } := Tensor.splitHeads v

  let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, seq, cfg.head_dim], dtype := dtype } :=
    Tensor.attentionGQAMask qh kh vh attn_mask is_causal true

  let attn : Tensor { device := device, shape := #[batch, seq, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } := Tensor.joinHeads attn
  let attnFlat : Tensor { device := device, shape := #[batch, seq, qLen], dtype := dtype } := Tensor.reshapeChecked attn #[batch, seq, qLen]
  let gated : Tensor { device := device, shape := #[batch, seq, qLen], dtype := dtype } := Tensor.mul attnFlat (Tensor.sigmoid gateFlat)
  Tensor.linear3d gated m.o_proj

def forwardStep {batch : Nat}
    (cfg : Config)
    (m : Qwen35Attention (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[1, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[1, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (cache : KVCache (dtype := dtype) (device := device) cfg batch)
    : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } × KVCache (dtype := dtype) (device := device) cfg batch :=
  let qg : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads * cfg.head_dim * 2], dtype := dtype } := Tensor.linear3d x m.q_proj
  let qLen : Nat := cfg.num_attention_heads * cfg.head_dim
  let (q, gateFlat) := splitQProjAndGate cfg qg

  let kFlat : Tensor { device := device, shape := #[batch, 1, cfg.num_key_value_heads * cfg.head_dim], dtype := dtype } := Tensor.linear3d x m.k_proj
  let vFlat : Tensor { device := device, shape := #[batch, 1, cfg.num_key_value_heads * cfg.head_dim], dtype := dtype } := Tensor.linear3d x m.v_proj

  let k : Tensor { device := device, shape := #[batch, 1, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
    Tensor.reshape kFlat #[batch, 1, cfg.num_key_value_heads, cfg.head_dim]
  let v : Tensor { device := device, shape := #[batch, 1, cfg.num_key_value_heads, cfg.head_dim], dtype := dtype } :=
    Tensor.reshape vFlat #[batch, 1, cfg.num_key_value_heads, cfg.head_dim]

  let q := applyHeadNorm m.q_norm q
  let k := applyHeadNorm m.k_norm k

  let q := applyRotaryPartial q cos sin
  let k := applyRotaryPartial k cos sin

  let qh : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim], dtype := dtype } := Tensor.splitHeads q
  let kNew : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, 1, cfg.head_dim], dtype := dtype } := Tensor.splitHeads k
  let vNew : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, 1, cfg.head_dim], dtype := dtype } := Tensor.splitHeads v

  let kStore : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim], dtype := dtype } :=
    Tensor.castLike kNew (Tensor.reshapeChecked cache.kStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])
  let vStore : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim], dtype := dtype } :=
    Tensor.castLike vNew (Tensor.reshapeChecked cache.vStoreDyn.2 #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim])

  if cache.seq < cache.maxLen then
    let kStore' : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim], dtype := dtype } :=
      Tensor.sliceScatter kStore 2 cache.seq kNew
    let vStore' : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim], dtype := dtype } :=
      Tensor.sliceScatter vStore 2 cache.seq vNew
    let kvLen : Nat := cache.seq + 1
    let kAll : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } := typedSlice kStore' 2 0 kvLen
    let vAll : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } := typedSlice vStore' 2 0 kvLen
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim], dtype := dtype } :=
      Tensor.attentionGQAQKV qh kAll vAll false true
    let attn : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } := Tensor.joinHeads attn
    let attnFlat : Tensor { device := device, shape := #[batch, 1, qLen], dtype := dtype } := Tensor.reshapeChecked attn #[batch, 1, qLen]
    let gated : Tensor { device := device, shape := #[batch, 1, qLen], dtype := dtype } := Tensor.mul attnFlat (Tensor.sigmoid gateFlat)
    let out : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := Tensor.linear3d gated m.o_proj
    let cache' : KVCache (dtype := dtype) (device := device) cfg batch := {
      kStoreDyn := Tensor.packShape kStore'
      vStoreDyn := Tensor.packShape vStore'
      seq := kvLen
      maxLen := cache.maxLen
    }
    (out, cache')
  else
    let writePos : Nat := if cache.maxLen == 0 then 0 else cache.maxLen - 1
    let kStore' : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim], dtype := dtype } :=
      Tensor.sliceScatter kStore 2 writePos kNew
    let vStore' : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, cache.maxLen, cfg.head_dim], dtype := dtype } :=
      Tensor.sliceScatter vStore 2 writePos vNew
    let kvLen : Nat := cache.maxLen
    let kAll : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } := typedSlice kStore' 2 0 kvLen
    let vAll : Tensor { device := device, shape := #[batch, cfg.num_key_value_heads, kvLen, cfg.head_dim], dtype := dtype } := typedSlice vStore' 2 0 kvLen
    let attn : Tensor { device := device, shape := #[batch, cfg.num_attention_heads, 1, cfg.head_dim], dtype := dtype } :=
      Tensor.attentionGQAQKV qh kAll vAll false true
    let attn : Tensor { device := device, shape := #[batch, 1, cfg.num_attention_heads, cfg.head_dim], dtype := dtype } := Tensor.joinHeads attn
    let attnFlat : Tensor { device := device, shape := #[batch, 1, qLen], dtype := dtype } := Tensor.reshapeChecked attn #[batch, 1, qLen]
    let gated : Tensor { device := device, shape := #[batch, 1, qLen], dtype := dtype } := Tensor.mul attnFlat (Tensor.sigmoid gateFlat)
    let out : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := Tensor.linear3d gated m.o_proj
    let cache' : KVCache (dtype := dtype) (device := device) cfg batch := {
      kStoreDyn := Tensor.packShape kStore'
      vStoreDyn := Tensor.packShape vStore'
      seq := cache.maxLen
      maxLen := cache.maxLen
    }
    (out, cache')

end Qwen35Attention

/-- Linear token mixer for Qwen3.5 (`GatedDeltaNet`). -/
structure Qwen35GatedDeltaNet (cfg : Config) (dtype : DType := .Float32) where
  in_proj_qkv : Tensor { device := device, shape := #[Config.linearConvDim cfg, cfg.hidden_size], dtype := dtype }
  in_proj_z : Tensor { device := device, shape := #[Config.linearValueDim cfg, cfg.hidden_size], dtype := dtype }
  in_proj_b : Tensor { device := device, shape := #[cfg.linear_num_value_heads, cfg.hidden_size], dtype := dtype }
  in_proj_a : Tensor { device := device, shape := #[cfg.linear_num_value_heads, cfg.hidden_size], dtype := dtype }

  conv1d_weight : Tensor { device := device, shape := #[Config.linearConvDim cfg, 1, cfg.linear_conv_kernel_dim], dtype := dtype }
  conv1d_bias : Tensor { device := device, shape := #[Config.linearConvDim cfg], dtype := dtype }

  dt_bias : SomeDTypeTensor #[cfg.linear_num_value_heads] device
  a_log : SomeDTypeTensor #[cfg.linear_num_value_heads] device

  norm : Qwen35RMSNormGated (device := device) cfg.linear_value_head_dim
  out_proj : Tensor { device := device, shape := #[cfg.hidden_size, Config.linearValueDim cfg], dtype := dtype }
  deriving TensorStruct

namespace Qwen35GatedDeltaNet

def init (cfg : Config) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35GatedDeltaNet (dtype := dtype) (device := device) cfg) := do
  let inQKV ← initDynamicWeight (dtype := dtype) (device := device) #[Config.linearConvDim cfg, cfg.hidden_size] cfg.hidden_size
  let inZ ← initDynamicWeight (dtype := dtype) (device := device) #[Config.linearValueDim cfg, cfg.hidden_size] cfg.hidden_size
  let inB ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.linear_num_value_heads, cfg.hidden_size] cfg.hidden_size
  let inA ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.linear_num_value_heads, cfg.hidden_size] cfg.hidden_size

  let convW ← initDynamicWeight (dtype := dtype) (device := device) #[Config.linearConvDim cfg, 1, cfg.linear_conv_kernel_dim] cfg.linear_conv_kernel_dim
  let convB ← initDynamicBias (dtype := dtype) (device := device) #[Config.linearConvDim cfg]

  let dtBias ← Tensor.parameter (Tensor.ones #[cfg.linear_num_value_heads] (device := device))

  let aRand ← randDynamic (device := device) #[cfg.linear_num_value_heads]
  let aScaled : Tensor { device := device, shape := #[cfg.linear_num_value_heads], dtype := .Float32 } := Tensor.addScalar (Tensor.mulScalar aRand 16.0) 1e-6
  let aLog ← Tensor.parameter (logDynamic aScaled)

  let outProj ← initDynamicWeight (dtype := dtype) (device := device) #[cfg.hidden_size, Config.linearValueDim cfg] (Config.linearValueDim cfg)
  pure {
    in_proj_qkv := inQKV
    in_proj_z := inZ
    in_proj_b := inB
    in_proj_a := inA
    conv1d_weight := convW
    conv1d_bias := convB
    dt_bias := ⟨.Float32, dtBias⟩
    a_log := ⟨.Float32, aLog⟩
    norm := (← Qwen35RMSNormGated.init (device := device) cfg.linear_value_head_dim cfg.rms_norm_eps)
    out_proj := outProj
  }

private def repeatHeads {batch seq n_head head_dim rep : Nat}
    (x : Tensor { device := device, shape := #[batch, seq, n_head, head_dim], dtype := dtype })
    : Tensor { device := device, shape := #[batch, seq, n_head * rep, head_dim], dtype := dtype } :=
  if rep == 1 then
    Tensor.reshapeChecked x #[batch, seq, n_head * rep, head_dim]
  else
    let x5 : Tensor { device := device, shape := #[batch, seq, n_head, 1, head_dim], dtype := dtype } := x.reshape #[batch, seq, n_head, 1, head_dim]
    let x5 : Tensor { device := device, shape := #[batch, seq, n_head, rep, head_dim], dtype := dtype } := Tensor.expand x5 #[batch, seq, n_head, rep, head_dim]
    Tensor.reshape x5 #[batch, seq, n_head * rep, head_dim]

/-- Run linear token mixer. If `cacheConv`/`cacheRec` are provided and
    `usePrecomputedState=true`, performs incremental one-step update. -/
def forward {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35GatedDeltaNet (dtype := dtype) (device := device) cfg)
    (hidden : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (attentionMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    (cacheConv : Option (Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim], dtype := dtype }) := none)
    (cacheRec : Option (Tensor { device := device, shape := #[batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim], dtype := .Float32 }) := none)
    (usePrecomputedState : Bool := false)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype }
      × Option (Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim], dtype := dtype })
      × Option (Tensor { device := device, shape := #[batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim], dtype := .Float32 }) :=
  let hidden := applyMaskToPaddingStates hidden attentionMask
  let device := hidden.device

  let mixedQKV : Tensor { device := device, shape := #[batch, seq, Config.linearConvDim cfg], dtype := dtype } := Tensor.linear3d hidden m.in_proj_qkv
  let mixedQKVc : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, seq], dtype := dtype } := transposeSeq mixedQKV

  let zFlat : Tensor { device := device, shape := #[batch, seq, Config.linearValueDim cfg], dtype := dtype } := Tensor.linear3d hidden m.in_proj_z
  let z : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim], dtype := dtype } :=
    Tensor.reshapeChecked zFlat #[batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim]

  let b : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_value_heads], dtype := dtype } := Tensor.linear3d hidden m.in_proj_b
  let a : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_value_heads], dtype := dtype } := Tensor.linear3d hidden m.in_proj_a

  let (convOut, nextConvState) :
      Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, seq], dtype := dtype }
        × Option (Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim], dtype := dtype }) :=
    if usePrecomputedState && seq == 1 then
      match cacheConv with
      | some st =>
        let catIn : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim + 1], dtype := dtype } := typedCat st mixedQKVc 2
        let convIn : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim + 1], dtype := dtype } :=
          Tensor.castLike m.conv1d_weight catIn
        let stNext : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim], dtype := dtype } :=
          typedSlice catIn 2 1 cfg.linear_conv_kernel_dim
        let raw : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, 2], dtype := dtype } :=
          convGroups convIn m.conv1d_weight m.conv1d_bias 1 0 1 (Config.linearConvDim cfg)
        let y : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, 1], dtype := dtype } := typedSlice raw 2 1 1
        ((Tensor.silu y).reshapeChecked #[batch, Config.linearConvDim cfg, seq], some stNext)
      | none =>
        let convIn : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, seq], dtype := dtype } :=
          Tensor.castLike m.conv1d_weight mixedQKVc
        let raw : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, seq + cfg.linear_conv_kernel_dim - 1], dtype := dtype } :=
          convGroups convIn m.conv1d_weight m.conv1d_bias 1 (cfg.linear_conv_kernel_dim - 1) 1 (Config.linearConvDim cfg)
        let y : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, seq], dtype := dtype } := typedSlice raw 2 0 seq
        let nextState :=
          if seq >= cfg.linear_conv_kernel_dim then
            some (typedSlice mixedQKVc 2 (seq - cfg.linear_conv_kernel_dim) cfg.linear_conv_kernel_dim)
          else
            let padLen : Nat := cfg.linear_conv_kernel_dim - seq
            let z0 : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, padLen], dtype := dtype } := zerosDynamicOn device dtype
            some (typedCat z0 mixedQKVc 2)
        (Tensor.silu y, nextState)
    else
      let convIn : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, seq], dtype := dtype } :=
        Tensor.castLike m.conv1d_weight mixedQKVc
      let raw : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, seq + cfg.linear_conv_kernel_dim - 1], dtype := dtype } :=
        convGroups convIn m.conv1d_weight m.conv1d_bias 1 (cfg.linear_conv_kernel_dim - 1) 1 (Config.linearConvDim cfg)
      let y : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, seq], dtype := dtype } := typedSlice raw 2 0 seq
      let nextState :=
        if seq >= cfg.linear_conv_kernel_dim then
          some (typedSlice mixedQKVc 2 (seq - cfg.linear_conv_kernel_dim) cfg.linear_conv_kernel_dim)
        else
          let padLen : Nat := cfg.linear_conv_kernel_dim - seq
          let z0 : Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, padLen], dtype := dtype } := zerosDynamicOn device dtype
          some (typedCat z0 mixedQKVc 2)
      (Tensor.silu y, nextState)

  let mixed : Tensor { device := device, shape := #[batch, seq, Config.linearConvDim cfg], dtype := dtype } := transposeSeq convOut

  let keyDim : Nat := Config.linearKeyDim cfg
  let valDim : Nat := Config.linearValueDim cfg

  let qRaw : Tensor { device := device, shape := #[batch, seq, keyDim], dtype := dtype } := typedSlice mixed 2 0 keyDim
  let kRaw : Tensor { device := device, shape := #[batch, seq, keyDim], dtype := dtype } := typedSlice mixed 2 keyDim keyDim
  let vRaw : Tensor { device := device, shape := #[batch, seq, valDim], dtype := dtype } := typedSlice mixed 2 (keyDim * 2) valDim

  let q0 : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim], dtype := dtype } :=
    Tensor.reshapeChecked qRaw #[batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim]
  let k0 : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim], dtype := dtype } :=
    Tensor.reshapeChecked kRaw #[batch, seq, cfg.linear_num_key_heads, cfg.linear_key_head_dim]
  let v : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim], dtype := dtype } :=
    Tensor.reshapeChecked vRaw #[batch, seq, cfg.linear_num_value_heads, cfg.linear_value_head_dim]

  let beta : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_value_heads], dtype := dtype } := Tensor.sigmoid b

  let dt : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_value_heads], dtype := m.dt_bias.1 } :=
    Tensor.expand (Tensor.reshape m.dt_bias.2 #[1, 1, cfg.linear_num_value_heads]) #[batch, seq, cfg.linear_num_value_heads]
  let aExp : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_value_heads], dtype := .Float32 } :=
    Tensor.expand (Tensor.reshape (Tensor.exp (floatDynamic m.a_log.2)) #[1, 1, cfg.linear_num_value_heads]) #[batch, seq, cfg.linear_num_value_heads]
  let gPos : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_value_heads], dtype := .Float32 } := softplusDynamic (floatDynamic (promotedAdd a dt).2)
  let g : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_value_heads], dtype := .Float32 } := Tensor.mulScalar (Tensor.mul gPos aExp) (-1.0)

  let rep := Config.linearKVRepeat cfg
  let q : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_value_heads, cfg.linear_key_head_dim], dtype := dtype } :=
    (repeatHeads (rep := rep) q0).reshapeChecked #[batch, seq, cfg.linear_num_value_heads, cfg.linear_key_head_dim]
  let k : Tensor { device := device, shape := #[batch, seq, cfg.linear_num_value_heads, cfg.linear_key_head_dim], dtype := dtype } :=
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

  let core2d : Tensor { device := device, shape := #[batch * seq * cfg.linear_num_value_heads, cfg.linear_value_head_dim], dtype := dtype } :=
    Tensor.reshape core #[batch * seq * cfg.linear_num_value_heads, cfg.linear_value_head_dim]
  let z2d : Tensor { device := device, shape := #[batch * seq * cfg.linear_num_value_heads, cfg.linear_value_head_dim], dtype := dtype } :=
    Tensor.reshape z #[batch * seq * cfg.linear_num_value_heads, cfg.linear_value_head_dim]
  let core2d := m.norm.forward2d core2d z2d
  let core3d : Tensor { device := device, shape := #[batch, seq, valDim], dtype := dtype } := Tensor.reshapeChecked core2d #[batch, seq, valDim]

  let out : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := Tensor.linear3d core3d m.out_proj
  (out, nextConvState, some finalState)

end Qwen35GatedDeltaNet

/-- Layer-local cache bundle for hybrid token mixers. -/
structure HybridCache (cfg : Config) (batch : Nat) (dtype : DType := .Float32) where
  attnCaches : Array (Option (Qwen35Attention.KVCache (dtype := dtype) (device := device) cfg batch))
  convStates : Array (Option (Tensor { device := device, shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim], dtype := dtype }))
  recurrentStates : Array (Option (Tensor { device := device, shape := #[batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim], dtype := .Float32 }))
  lastLinearLayer : Option Nat := none

namespace HybridCache

def hasPreviousState (c : HybridCache (dtype := dtype) (device := device) cfg batch) : Bool :=
  match c.lastLinearLayer with
  | none => false
  | some idx =>
    match c.convStates[idx]? with
    | some (some _) => true
    | _ => false

end HybridCache

/-- One Qwen3.5 decoder layer (hybrid token mixer + dense/MoE FFN). -/
structure Qwen35Layer (cfg : Config) (dtype : DType := .Float32) where
  layerType : LayerType
  input_layernorm : Qwen35RMSNorm (device := device) cfg.hidden_size
  full_attn : Option (Qwen35Attention (dtype := dtype) (device := device) cfg) := none
  linear_attn : Option (Qwen35GatedDeltaNet (dtype := dtype) (device := device) cfg) := none
  post_attention_layernorm : Qwen35RMSNorm (device := device) cfg.hidden_size
  dense_mlp : Option (Qwen35MLP (dtype := dtype) (device := device) cfg.hidden_size cfg.intermediate_size) := none
  sparse_moe : Option (Qwen35SparseMoeBlock (dtype := dtype) (device := device) cfg) := none
  deriving TensorStruct

namespace Qwen35Layer

def init (cfg : Config) (layerType : LayerType) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35Layer (dtype := dtype) (device := device) cfg) := do
  let inputNorm ← Qwen35RMSNorm.initZeroCentered (device := device) cfg.hidden_size cfg.rms_norm_eps
  let postNorm ← Qwen35RMSNorm.initZeroCentered (device := device) cfg.hidden_size cfg.rms_norm_eps

  let fullAttn ←
    match layerType with
    | .fullAttention =>
      let m ← Qwen35Attention.init (dtype := dtype) (device := device) cfg
      pure (some m)
    | .linearAttention => pure none

  let linearAttn ←
    match layerType with
    | .linearAttention =>
      let m ← Qwen35GatedDeltaNet.init (dtype := dtype) (device := device) cfg
      pure (some m)
    | .fullAttention => pure none

  let denseMlp ←
    if Config.isMoE cfg then
      pure none
    else
      let m ← Qwen35MLP.init (dtype := dtype) (device := device) cfg.hidden_size cfg.intermediate_size
      pure (some m)

  let sparseMoe ←
    if Config.isMoE cfg then
      let m ← Qwen35SparseMoeBlock.init (dtype := dtype) (device := device) cfg
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

def forward {batch seq : Nat}
    (cfg : Config)
    (layer : Qwen35Layer (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
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
  let h2 := Tensor.add residual1 mixed

  let residual2 := h2
  let h3 := layer.post_attention_layernorm.forward3d h2
  let ffn :=
    match layer.dense_mlp, layer.sparse_moe with
    | some mlp, _ => mlp.forward h3
    | _, some moe => moe.forward cfg h3
    | _, _ => h3
  Tensor.add residual2 ffn

/-- Forward pass with optional cache update (for batch prefill). -/
def forwardWithCache {batch seq : Nat}
    (cfg : Config)
    (layer : Qwen35Layer (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (cache : HybridCache (dtype := dtype) (device := device) cfg batch)
    (layerIdx : Nat)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × HybridCache (dtype := dtype) (device := device) cfg batch :=
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

  let h2 := Tensor.add residual1 mixed
  let residual2 := h2
  let h3 := layer.post_attention_layernorm.forward3d h2
  let ffn :=
    match layer.dense_mlp, layer.sparse_moe with
    | some mlp, _ => mlp.forward h3
    | _, some moe => moe.forward cfg h3
    | _, _ => h3
  (Tensor.add residual2 ffn, cache')

def forwardStep {batch : Nat}
    (cfg : Config)
    (layer : Qwen35Layer (dtype := dtype) (device := device) cfg)
    (x : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[1, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[1, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (cache : HybridCache (dtype := dtype) (device := device) cfg batch)
    (layerIdx : Nat)
    : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } × HybridCache (dtype := dtype) (device := device) cfg batch :=
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

  let h2 := Tensor.add residual1 mixed
  let residual2 := h2
  let h3 := layer.post_attention_layernorm.forward3d h2
  let ffn :=
    match layer.dense_mlp, layer.sparse_moe with
    | some mlp, _ => mlp.forward h3
    | _, some moe => moe.forward cfg h3
    | _, _ => h3
  (Tensor.add residual2 ffn, cache1)

end Qwen35Layer

/-- Text-only Qwen3.5 base model. -/
structure Qwen35Model (cfg : Config) (dtype : DType := .Float32) where
  embed_tokens : Tensor { device := device, shape := #[cfg.vocab_size, cfg.hidden_size], dtype := dtype }
  layers : Array (Qwen35Layer (dtype := dtype) (device := device) cfg)
  norm : Qwen35RMSNorm (device := device) cfg.hidden_size
  deriving TensorStruct

namespace Qwen35Model

def init (cfg : Config) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35Model (dtype := dtype) (device := device) cfg) := do
  let embRaw ← Tensor.randn (device := device) #[cfg.vocab_size, cfg.hidden_size]
  let embedTokens ← Tensor.parameter (← IO.ofExcept ((Tensor.mulScalar embRaw 0.02).castDType? dtype))

  let layerTypes := Config.normalizedLayerTypes cfg
  let mut layers : Array (Qwen35Layer (dtype := dtype) (device := device) cfg) := #[]
  for i in [:cfg.num_hidden_layers] do
    let lt := layerTypes.getD i .linearAttention
    let l ← Qwen35Layer.init (dtype := dtype) (device := device) cfg lt
    layers := layers.push l

  let norm ← Qwen35RMSNorm.initZeroCentered (device := device) cfg.hidden_size cfg.rms_norm_eps
  pure { embed_tokens := embedTokens, layers := layers, norm := norm }

def initCache {batch : Nat}
    (cfg : Config)
    (m : Qwen35Model (dtype := dtype) (device := device) cfg)
    (maxLen : Nat)
    (cacheDevice : Device)
    [DType.ConstantSupported dtype cacheDevice]
    : HybridCache (dtype := dtype) (device := cacheDevice) cfg batch :=
  Id.run do
    let layerTypes := Config.normalizedLayerTypes cfg
    let mut attn : Array (Option (Qwen35Attention.KVCache (dtype := dtype) (device := cacheDevice) cfg batch)) := #[]
    let mut conv : Array (Option (Tensor { device := cacheDevice, shape := #[batch, Config.linearConvDim cfg, cfg.linear_conv_kernel_dim], dtype := dtype })) := #[]
    let mut recStates : Array (Option (Tensor { device := cacheDevice, shape := #[batch, cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim], dtype := .Float32 })) := #[]
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
          cacheDevice dtype
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

def forward {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35Model (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  let x0 : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } := Tensor.embedding inputIds m.embed_tokens
  let (cos, sin) := Tensor.rotaryFreqs seq (Config.rotaryDim cfg) cfg.rope_theta x0.device
  let h :=
    m.layers.foldl
      (fun h layer => layer.forward cfg h cos sin attnMask)
      x0
  m.norm.forward3d h

def forwardWithCache {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35Model (dtype := dtype) (device := device) cfg)
    (x0 : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cos : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sin : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (cache : HybridCache (dtype := dtype) (device := device) cfg batch)
    : IO (Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } × HybridCache (dtype := dtype) (device := device) cfg batch) := do
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
structure CausalLMOutput (cfg : Config) (batch seq : Nat) (dtype : DType := .Float32) where
  logits : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := dtype }

/-- Full standalone Qwen3.5 causal language model (dense or MoE depending on config). -/
structure Qwen35ForCausalLM (cfg : Config) (dtype : DType := .Float32) where
  model : Qwen35Model (dtype := dtype) (device := device) cfg
  lmHead : Tensor { device := device, shape := #[cfg.vocab_size, cfg.hidden_size], dtype := dtype }
  tieWordEmbeddings : Bool := true
  deriving TensorStruct

namespace Qwen35ForCausalLM

def init (cfg : Config) (tieWordEmbeddings : Bool := true) (device : Device := .CPU) (dtype : DType := .Float32) : IO (Qwen35ForCausalLM (dtype := dtype) (device := device) cfg) := do
  let model ← Qwen35Model.init (dtype := dtype) (device := device) cfg
  let lmHead ←
    if tieWordEmbeddings then
      pure model.embed_tokens
    else
      initDynamicWeight (dtype := dtype) (device := device) #[cfg.vocab_size, cfg.hidden_size] cfg.hidden_size
  pure { model := model, lmHead := lmHead, tieWordEmbeddings := tieWordEmbeddings }

def embedTokens {batch seq : Nat}
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype } :=
  Tensor.embedding inputIds m.model.embed_tokens

def forwardEmbeds {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := dtype } :=
  let (cos, sin) := Tensor.rotaryFreqs seq (Config.rotaryDim cfg) cfg.rope_theta inputsEmbeds.device
  let hidden := m.model.layers.foldl (fun h layer => layer.forward cfg h cos sin attnMask) inputsEmbeds
  let hidden := m.model.norm.forward3d hidden
  Tensor.linear3d hidden m.lmHead

def forward {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (attnMask : Option (Tensor { device := device, shape := #[batch, seq], dtype := .Int64 }) := none)
    : Tensor { device := device, shape := #[batch, seq, cfg.vocab_size], dtype := dtype } :=
  let embeds := m.embedTokens inputIds
  m.forwardEmbeds cfg embeds attnMask

private def precomputeDecodeRotary {maxLen : Nat}
    (cfg : Config)
    (device : Device)
    : Tensor { device := device, shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 } × Tensor { device := device, shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 } :=
  Tensor.rotaryFreqs
    maxLen
    (Config.rotaryDim cfg)
    cfg.rope_theta
    device

/-- Single-pass prompt prefill for KV-cache. -/
private def prefillCachesFromEmbeds {batch seq maxLen : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (cosAll : Tensor { device := device, shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sinAll : Tensor { device := device, shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (cache : HybridCache (dtype := dtype) (device := device) cfg batch)
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } × HybridCache (dtype := dtype) (device := device) cfg batch) := do
  let cos : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 } := typedSlice cosAll 0 0 seq
  let sin : Tensor { device := device, shape := #[seq, Config.rotaryHalfDim cfg], dtype := .Float32 } := typedSlice sinAll 0 0 seq

  let (hidden, cache') ← Qwen35Model.forwardWithCache cfg m.model inputsEmbeds cos sin cache

  let lastHidden : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := typedSlice hidden 1 (seq - 1) 1
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := Tensor.linear3d lastHidden m.lmHead
  let logits : Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } := Tensor.reshape logits3 #[batch, cfg.vocab_size]
  pure (logits, cache')

private def decodeStepFromEmbedWithCache {batch maxLen : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (cosAll : Tensor { device := device, shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sinAll : Tensor { device := device, shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (tokenEmbed : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype })
    (position : Nat)
    (cache : HybridCache (dtype := dtype) (device := device) cfg batch)
    : IO (Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } × HybridCache (dtype := dtype) (device := device) cfg batch) := do
  let cos : Tensor { device := device, shape := #[1, Config.rotaryHalfDim cfg], dtype := .Float32 } := typedSlice cosAll 0 position 1
  let sin : Tensor { device := device, shape := #[1, Config.rotaryHalfDim cfg], dtype := .Float32 } := typedSlice sinAll 0 position 1

  let mut hidden : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := tokenEmbed
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
  let logits3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := Tensor.linear3d hiddenNorm m.lmHead
  let logits2 : Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } := Tensor.reshape logits3 #[batch, cfg.vocab_size]
  pure (logits2, cache')

private partial def decodeLoopCached {batch maxLen : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (cosAll : Tensor { device := device, shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (sinAll : Tensor { device := device, shape := #[maxLen, Config.rotaryHalfDim cfg], dtype := .Float32 })
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (finished : Tensor { device := device, shape := #[batch], dtype := .Bool })
    (remaining : Nat)
    (cache : HybridCache (dtype := dtype) (device := device) cfg batch)
    (lastLogits : Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype })
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

  let nextCol : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 } := Tensor.reshape nextTok #[batch, 1]
  let appended : Tensor { device := device, shape := #[batch, curSeq + 1], dtype := .Int64 } := typedCat curIds nextCol 1

  let stop :=
    if eosTokenIds.isEmpty then
      false
    else
      !(Tensor.any (Tensor.logicalNot finished'))
  if stop then
    return ⟨curSeq + 1, appended⟩
  else
    let nextEmb : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := m.embedTokens nextCol
    let (nextLogits, cache') ← decodeStepFromEmbedWithCache cfg m cosAll sinAll nextEmb curSeq cache
    decodeLoopCached cfg m cosAll sinAll strategy eosTokenIds finished' (remaining - 1) cache' nextLogits onStep (generatedSoFar + 1) appended

private partial def decodeLoopUncached {batch : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (finished : Tensor { device := device, shape := #[batch], dtype := .Bool })
    (remaining : Nat)
    {curSeq : Nat}
    (curIds : Tensor { device := device, shape := #[batch, curSeq], dtype := .Int64 })
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩
  if curSeq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"

  let logits := m.forward cfg curIds none
  let lastPos := curSeq - 1
  let last3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } := typedSlice logits 1 lastPos 1
  let last2 : Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } := Tensor.reshape last3 #[batch, cfg.vocab_size]

  let nextTokRaw ← sampleFromLogits last2.toFloat32 strategy
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

  let nextCol : Tensor { device := device, shape := #[batch, 1], dtype := .Int64 } := Tensor.reshape nextTok #[batch, 1]
  let appended : Tensor { device := device, shape := #[batch, curSeq + 1], dtype := .Int64 } := typedCat curIds nextCol 1

  let stop :=
    if eosTokenIds.isEmpty then
      false
    else
      !(Tensor.any (Tensor.logicalNot finished'))
  if stop then
    return ⟨curSeq + 1, appended⟩
  else
    decodeLoopUncached cfg m strategy eosTokenIds finished' (remaining - 1) appended

private partial def decodeLoopUncachedEmbeds {batch : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (strategy : SamplingStrategy)
    (eosTokenIds : Array UInt64)
    (finished : Tensor { device := device, shape := #[batch], dtype := .Bool })
    (remaining : Nat)
    {curSeq : Nat}
    (curIds : Tensor { device := device, shape := #[batch, curSeq], dtype := .Int64 })
    (curEmbeds : Tensor { device := device, shape := #[batch, curSeq, cfg.hidden_size], dtype := dtype })
    (onStep : Option (StreamCallback (device := device) batch))
    (generatedSoFar : Nat)
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  if remaining == 0 then
    return ⟨curSeq, curIds⟩
  if curSeq == 0 then
    throw <| IO.userError "generate requires non-empty prompt sequence"

  let logits := m.forwardEmbeds cfg curEmbeds none
  let lastPos := curSeq - 1
  let last3 : Tensor { device := device, shape := #[batch, 1, cfg.vocab_size], dtype := dtype } :=
    typedSlice logits 1 lastPos 1
  let last2 : Tensor { device := device, shape := #[batch, cfg.vocab_size], dtype := dtype } := Tensor.reshape last3 #[batch, cfg.vocab_size]

  let nextTokRaw ← sampleFromLogits last2.toFloat32 strategy
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
  let appendedIds : Tensor { device := device, shape := #[batch, curSeq + 1], dtype := .Int64 } := typedCat curIds nextCol 1

  let stop :=
    if eosTokenIds.isEmpty then
      false
    else
      !(Tensor.any (Tensor.logicalNot finished'))
  if stop then
    return ⟨curSeq + 1, appendedIds⟩
  else
    let nextEmb : Tensor { device := device, shape := #[batch, 1, cfg.hidden_size], dtype := dtype } := m.embedTokens nextCol
    let appendedEmbeds : Tensor { device := device, shape := #[batch, curSeq + 1, cfg.hidden_size], dtype := dtype } := typedCat curEmbeds nextEmb 1
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
private def generateFromEmbedsCore {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
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
  let (cosAll, sinAll) := Tensor.rotaryFreqs maxLen (Config.rotaryDim cfg) cfg.rope_theta inputsEmbeds.device
  let cache := Qwen35Model.initCache cfg m.model maxLen inputsEmbeds.device

  -- 1) Prefill caches and get last prompt logits
  let (logits, caches) ← prefillCachesFromEmbeds cfg m cosAll sinAll inputsEmbeds cache

  -- 2) Start incremental decode loop
  let finished0 : Tensor { device := device, shape := #[batch], dtype := .Bool } := falseMaskTyped (n := batch) inputIds.device
  let finalIds ← decodeLoopCached
    (maxLen := maxLen)
    cfg m cosAll sinAll strategy eosTokenIds finished0 maxNewTokens
    caches logits onStep 0 inputIds

  pure finalIds


/-- Exact generation from explicit input embeddings.
    Useful for multimodal wrappers that inject vision features into token embeddings. -/
def generateFromEmbeds {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds none

/-- Exact generation from explicit input embeddings with per-step token callback. -/
def generateFromEmbedsStream {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (inputsEmbeds : Tensor { device := device, shape := #[batch, seq, cfg.hidden_size], dtype := dtype })
    (onStep : StreamCallback (device := device) batch)
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds (some onStep)

/-- Exact generation entry point for token IDs. -/
def generate {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let inputsEmbeds := m.embedTokens inputIds
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds none

/-- Exact generation with per-step token callback (streaming). -/
def generateStream {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (onStep : StreamCallback (device := device) batch)
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let inputsEmbeds := m.embedTokens inputIds
  generateFromEmbedsCore cfg m inputIds inputsEmbeds maxNewTokens strategy eosTokenIds (some onStep)

/-- Reference generation path by full re-forward each decode step. -/
def generateUncached {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : Nat := 256)
    (strategy : SamplingStrategy := .greedy)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) := do
  let finished0 : Tensor { device := device, shape := #[batch], dtype := .Bool } := falseMaskTyped (n := batch) inputIds.device
  decodeLoopUncached cfg m strategy eosTokenIds finished0 maxNewTokens inputIds

/-- Convenience wrapper for greedy generation. -/
def generateGreedy {batch seq : Nat}
    (cfg : Config)
    (m : Qwen35ForCausalLM (dtype := dtype) (device := device) cfg)
    (inputIds : Tensor { device := device, shape := #[batch, seq], dtype := .Int64 })
    (maxNewTokens : Nat := 256)
    (eosTokenIds : Array UInt64 := #[])
    : IO (Sigma (fun outSeq => Tensor { device := device, shape := #[batch, outSeq], dtype := .Int64 })) :=
  generate cfg m inputIds maxNewTokens .greedy eosTokenIds

end Qwen35ForCausalLM

end torch.qwen35
