/-
  Tyr/Module/LayerNorm.lean

  Layer Normalization module with TensorStruct and Module instances.
-/
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Widget

namespace torch

/-- Layer Normalization: normalizes over the last dimension.
    y = (x - mean) / sqrt(var + eps) * weight + bias -/
structure LayerNorm (dim : Nat) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  weight : Tensor ⟨#[dim], dtype, device⟩
  bias : Tensor ⟨#[dim], dtype, device⟩
  eps : Static Float := ⟨1e-5⟩
  deriving TensorStruct, ToModuleDisplay

namespace LayerNorm

/-- Initialize layer norm with ones for weight and zeros for bias -/
def init (dim : Nat) (eps : Float := 1e-5) (device : Device := .CPU) :
    IO (LayerNorm dim .Float32 device) := do
  IO.ofExcept (Shape.validateAllocation #[dim])
  let weight ← Tensor.parameter (Tensor.ones #[dim] (device := device))
  let bias ← Tensor.parameter (Tensor.zeros #[dim] (device := device))
  pure { weight, bias, eps := ⟨eps⟩ }

/-- Initialize layer norm without learnable affine parameters.
    Used when elementwise_affine=False in PyTorch.
    Weight is ones, bias is zeros, both non-trainable. -/
def initNoAffine (dim : Nat) (eps : Float := 1e-5) (device : Device := .CPU) :
    LayerNorm dim .Float32 device :=
  let weight := Tensor.ones #[dim] (device := device)
  let bias := Tensor.zeros #[dim] (device := device)
  { weight, bias, eps := ⟨eps⟩ }

/-- Forward pass for 3D input [batch, seq, dim] -/
def forward3d {dim batch seq : Nat} {d : DType} {dev : Device} (ln : LayerNorm dim d dev)
    (x : Tensor ⟨#[batch, seq, dim], d, dev⟩) : Tensor ⟨#[batch, seq, dim], d, dev⟩ :=
  Tensor.layerNorm x ln.weight ln.bias ln.eps.val

end LayerNorm

/-- Module instance for 3D forward pass -/
instance {dim batch seq : Nat} {d : DType} {dev : Device} :
    Module (LayerNorm dim d dev) (Tensor ⟨#[batch, seq, dim], d, dev⟩)
      (Tensor ⟨#[batch, seq, dim], d, dev⟩) where
  forward := LayerNorm.forward3d

end torch
