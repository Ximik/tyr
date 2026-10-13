/-
  Tyr/Model/Qwen/MLP.lean

  SwiGLU MLP for Qwen3.
  SwiGLU: down(silu(gate(x)) * up(x))

  Float32-pinned (`DTensor s .Float32`): the Qwen forward math is
  Float32-only (RMSNorm upcasts to Float32 and `linear3d` requires
  matching dtypes).
-/
import Tyr.Typed
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive

namespace torch.qwen

/-- Qwen MLP with SwiGLU activation.
    SwiGLU is more expressive than standard GELU MLP. -/
structure QwenMLP (hidden_size intermediate_size : UInt64) where
  /-- Gate projection: [intermediate_size, hidden_size] -/
  gate_proj : DTensor #[intermediate_size, hidden_size] .Float32
  /-- Up projection: [intermediate_size, hidden_size] -/
  up_proj : DTensor #[intermediate_size, hidden_size] .Float32
  /-- Down projection: [hidden_size, intermediate_size] -/
  down_proj : DTensor #[hidden_size, intermediate_size] .Float32
  deriving TensorStruct

namespace QwenMLP

/-- Initialize MLP layers with random weights -/
def init (hidden_size intermediate_size : UInt64) : IO (QwenMLP hidden_size intermediate_size) := do
  let std := Float.sqrt (2.0 / hidden_size.toFloat)

  let gate ← Tensor.randn #[intermediate_size, hidden_size]
  let up ← Tensor.randn #[intermediate_size, hidden_size]
  let down ← Tensor.randn #[hidden_size, intermediate_size]

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
    (mlp : QwenMLP hidden_size intermediate_size)
    (x : DTensor #[batch, seq, hidden_size] .Float32)
    : DTensor #[batch, seq, hidden_size] .Float32 :=
  -- Compute gate and up projections
  let gate := Tensor.linear3d x mlp.gate_proj  -- [batch, seq, intermediate_size]
  let up := Tensor.linear3d x mlp.up_proj      -- [batch, seq, intermediate_size]

  -- SwiGLU: silu(gate) * up
  let gate_silu := Tensor.silu gate rfl
  let hidden := gate_silu * up

  -- Down projection
  Tensor.linear3d hidden mlp.down_proj

end QwenMLP

end torch.qwen
