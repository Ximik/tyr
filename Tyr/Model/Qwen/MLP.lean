/-
  Tyr/Model/Qwen/MLP.lean

  SwiGLU MLP for Qwen3.
  SwiGLU: down(silu(gate(x)) * up(x))

  Shape-indexed activations and weights share a dtype parameter. Normalization
  accumulates in Float32 and restores the input dtype for BF16 checkpoints.
-/
import Tyr.Typed.Neural
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive

namespace torch.qwen

/-- Qwen MLP with SwiGLU activation.
    SwiGLU is more expressive than standard GELU MLP. -/
structure QwenMLP (hidden_size intermediate_size : UInt64) (dtype : DType := .Float32) where
  /-- Gate projection: [intermediate_size, hidden_size] -/
  gate_proj : Tensor { shape := #[intermediate_size, hidden_size], dtype := dtype }
  /-- Up projection: [intermediate_size, hidden_size] -/
  up_proj : Tensor { shape := #[intermediate_size, hidden_size], dtype := dtype }
  /-- Down projection: [hidden_size, intermediate_size] -/
  down_proj : Tensor { shape := #[hidden_size, intermediate_size], dtype := dtype }
  deriving TensorStruct

namespace QwenMLP

variable {dtype : DType}

/-- Initialize MLP layers with random weights -/
def init (hidden_size intermediate_size : UInt64) (dtype : DType := .Float32) : IO (QwenMLP hidden_size intermediate_size dtype) := do
  let std := Float.sqrt (2.0 / hidden_size.toFloat)

  let gate ← Tensor.randn #[intermediate_size, hidden_size]
  let gate := gate.castDType dtype
  let up ← Tensor.randn #[intermediate_size, hidden_size]
  let up := up.castDType dtype
  let down ← Tensor.randn #[hidden_size, intermediate_size]
  let down := down.castDType dtype

  pure {
    gate_proj := Tensor.setRequiresGrad (Tensor.Preserving.mulScalar gate std) true
    up_proj := Tensor.setRequiresGrad (Tensor.Preserving.mulScalar up std) true
    down_proj := Tensor.setRequiresGrad (Tensor.Preserving.mulScalar down std) true
  }

/-- Forward pass for MLP.
    Input: [batch, seq, hidden_size]
    Output: [batch, seq, hidden_size]

    SwiGLU: down(silu(gate(x)) * up(x)) -/
def forward {batch seq hidden_size intermediate_size : UInt64}
    (mlp : QwenMLP hidden_size intermediate_size dtype)
    (x : Tensor { shape := #[batch, seq, hidden_size], dtype := dtype })
    : Tensor { shape := #[batch, seq, hidden_size], dtype := dtype } :=
  -- Compute gate and up projections
  let gate := Tensor.linear3d x mlp.gate_proj  -- [batch, seq, intermediate_size]
  let up := Tensor.linear3d x mlp.up_proj      -- [batch, seq, intermediate_size]

  -- SwiGLU: silu(gate) * up
  let gate_silu := Tensor.Preserving.silu gate
  let hidden := Mul.mul gate_silu up

  -- Down projection
  Tensor.linear3d hidden mlp.down_proj

end QwenMLP

end torch.qwen
