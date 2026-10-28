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

variable {device : Device} {dtype : DType}
/-- Query-Key normalization.
    Applies RMSNorm to queries and keys for stable attention. -/
structure QKNorm (head_dim : Nat) (dtype : DType := .Float32) where
  query_norm : RMSNorm (device := device) head_dim dtype
  key_norm : RMSNorm (device := device) head_dim dtype
  deriving TensorStruct


namespace QKNorm

/-- Initialize QK normalization -/
def init (head_dim : Nat) (eps : Float := 1e-6) (device : Device := .CPU) (dtype : DType := .Float32) : IO (QKNorm (device := device) (dtype := dtype) head_dim) := do
  pure {
    query_norm := (← RMSNorm.init (dtype := dtype) (device := device) head_dim eps)
    key_norm := (← RMSNorm.init (dtype := dtype) (device := device) head_dim eps) }

/-- Forward pass: normalize Q and K.
    q, k: [batch, seq, n_head, head_dim] or [batch, n_head, seq, head_dim]
    Returns normalized (q, k) -/
def forward {batch seq n_head head_dim : Nat}
    (qkn : QKNorm (device := device) (dtype := dtype) head_dim)
    (q : Tensor { dtype := dtype, device := device, shape := #[batch, seq, n_head, head_dim] })
    (k : Tensor { dtype := dtype, device := device, shape := #[batch, seq, n_head, head_dim] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, seq, n_head, head_dim] } × Tensor { dtype := dtype, device := device, shape := #[batch, seq, n_head, head_dim] } :=
  let q_norm := qkn.query_norm.forward4d q
  let k_norm := qkn.key_norm.forward4d k
  (q_norm, k_norm)

/-- Forward for attention-transposed format: [batch, n_head, seq, head_dim] -/
def forwardAttn {batch n_head seq head_dim : Nat}
    (qkn : QKNorm (device := device) (dtype := dtype) head_dim)
    (q : Tensor { dtype := dtype, device := device, shape := #[batch, n_head, seq, head_dim] })
    (k : Tensor { dtype := dtype, device := device, shape := #[batch, n_head, seq, head_dim] })
    : Tensor { dtype := dtype, device := device, shape := #[batch, n_head, seq, head_dim] } × Tensor { dtype := dtype, device := device, shape := #[batch, n_head, seq, head_dim] } :=
  let q_norm := qkn.query_norm.forward5d q
  let k_norm := qkn.key_norm.forward5d k
  (q_norm, k_norm)

end QKNorm

end torch.flux
