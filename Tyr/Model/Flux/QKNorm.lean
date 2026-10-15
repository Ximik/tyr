/-
  Tyr/Model/Flux/QKNorm.lean

  Query-Key normalization for Flux attention.
  Applies RMSNorm to Q and K independently.
-/
import Tyr.Model.Flux.TypedOps
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Module.RMSNorm

namespace torch.flux

variable {device : Device}
/-- Query-Key normalization.
    Applies RMSNorm to queries and keys for stable attention. -/
structure QKNorm (head_dim : UInt64) where
  query_norm : RMSNorm (device := device) head_dim (.Unknown "dynamic")
  key_norm : RMSNorm (device := device) head_dim (.Unknown "dynamic")
  deriving TensorStruct

namespace QKNorm

/-- Initialize QK normalization -/
def init (head_dim : UInt64) (eps : Float := 1e-6) (device : Device := .CPU) : IO (QKNorm (device := device) head_dim) := do
  pure {
    query_norm := (← RMSNorm.init (device := device) head_dim eps (.Unknown "dynamic"))
    key_norm := (← RMSNorm.init (device := device) head_dim eps (.Unknown "dynamic")) }

/-- Forward pass: normalize Q and K.
    q, k: [batch, seq, n_head, head_dim] or [batch, n_head, seq, head_dim]
    Returns normalized (q, k) -/
def forward {batch seq n_head head_dim : UInt64}
    (qkn : QKNorm (device := device) head_dim)
    (q : Tensor { device := device, shape := #[batch, seq, n_head, head_dim] })
    (k : Tensor { device := device, shape := #[batch, seq, n_head, head_dim] })
    : Tensor { device := device, shape := #[batch, seq, n_head, head_dim] } × Tensor { device := device, shape := #[batch, seq, n_head, head_dim] } :=
  let q_norm := qkn.query_norm.forward4d q
  let k_norm := qkn.key_norm.forward4d k
  (q_norm, k_norm)

/-- Forward for attention-transposed format: [batch, n_head, seq, head_dim] -/
def forwardAttn {batch n_head seq head_dim : UInt64}
    (qkn : QKNorm (device := device) head_dim)
    (q : Tensor { device := device, shape := #[batch, n_head, seq, head_dim] })
    (k : Tensor { device := device, shape := #[batch, n_head, seq, head_dim] })
    : Tensor { device := device, shape := #[batch, n_head, seq, head_dim] } × Tensor { device := device, shape := #[batch, n_head, seq, head_dim] } :=
  let q_norm := qkn.query_norm.forward5d q
  let k_norm := qkn.key_norm.forward5d k
  (q_norm, k_norm)

end QKNorm

end torch.flux
