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
structure LayerNorm (dim : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  weight : Tensor ⟨#[dim], dtype, device⟩
  bias : Tensor ⟨#[dim], dtype, device⟩
  eps : Static Float := ⟨1e-5⟩
  deriving TensorStruct, ToModuleDisplay

namespace LayerNorm

/-- Initialize layer norm with ones for weight and zeros for bias -/
def init (dim : UInt64) (eps : Float := 1e-5) : LayerNorm dim :=
  let weight := Tensor.ones #[dim] true
  let bias := Tensor.zeros #[dim] true
  { weight, bias, eps := ⟨eps⟩ }

/-- Initialize layer norm without learnable affine parameters.
    Used when elementwise_affine=False in PyTorch.
    Weight is ones, bias is zeros, both non-trainable. -/
def initNoAffine (dim : UInt64) (eps : Float := 1e-5) : LayerNorm dim :=
  let weight := Tensor.ones #[dim]
  let bias := Tensor.zeros #[dim]
  { weight, bias, eps := ⟨eps⟩ }

/-- Forward pass for 3D input [batch, seq, dim] -/
def forward3d {dim batch seq : UInt64} {d : DType} {dev : DevicePolicy} (ln : LayerNorm dim d dev)
    (x : Tensor ⟨#[batch, seq, dim], d, dev⟩) : Tensor ⟨#[batch, seq, dim], d, dev⟩ :=
  Tensor.layerNorm x ln.weight ln.bias ln.eps.val

end LayerNorm

/-- Module instance for 3D forward pass -/
instance {dim batch seq : UInt64} {d : DType} {dev : DevicePolicy} :
    Module (LayerNorm dim d dev) (Tensor ⟨#[batch, seq, dim], d, dev⟩)
      (Tensor ⟨#[batch, seq, dim], d, dev⟩) where
  forward := LayerNorm.forward3d

end torch
