/-
  Tyr/Module/Linear.lean

  Linear layer module with TensorStruct and Module instances.
-/
import Tyr.TensorStruct
import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Widget

namespace torch

/-- Linear layer: y = xW^T (+ b optionally)
    Weight shape is [out_features, in_features] following PyTorch convention. -/
structure Linear (in_dim out_dim : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  weight : Tensor ⟨#[out_dim, in_dim], dtype, device⟩
  bias : Option (Tensor ⟨#[out_dim], dtype, device⟩) := none
  deriving TensorStruct, ToModuleDisplay

namespace Linear

/-- Initialize a linear layer with random weights -/
def init (in_dim out_dim : UInt64) (withBias : Bool := true) : IO (Linear in_dim out_dim) := do
  -- Kaiming/He initialization: std = sqrt(2 / in_dim)
  let std := Float.sqrt (2.0 / in_dim.toFloat)
  let w ← Tensor.randn #[out_dim, in_dim]
  let weight := Tensor.Preserving.mulScalar w std
  let weight ← Tensor.parameter weight
  let bias ← if withBias then do
    let b ← Tensor.parameter (Tensor.zeros #[out_dim])
    pure (some b)
  else
    pure none
  pure { weight, bias }

/-- Forward pass for 2D input [batch, in_dim] -> [batch, out_dim] -/
def forward2d {in_dim out_dim batch : UInt64} {d : DType} {dev : DevicePolicy}
    (lin : Linear in_dim out_dim d dev)
    (x : Tensor ⟨#[batch, in_dim], d, dev⟩) : Tensor ⟨#[batch, out_dim], d, dev⟩ :=
  let y := Tensor.linear x lin.weight
  match lin.bias with
  | some b => Tensor.Preserving.add y (Tensor.expand b #[batch, out_dim])
  | none => y

/-- Forward pass for 3D input [batch, seq, in_dim] -> [batch, seq, out_dim] -/
def forward3d {in_dim out_dim batch seq : UInt64} {d : DType} {dev : DevicePolicy}
    (lin : Linear in_dim out_dim d dev)
    (x : Tensor ⟨#[batch, seq, in_dim], d, dev⟩) : Tensor ⟨#[batch, seq, out_dim], d, dev⟩ :=
  match lin.bias with
  | some b => Tensor.affine3d x lin.weight b
  | none => Tensor.linear3d x lin.weight

end Linear

/-- Module instance for 2D forward pass -/
instance {in_dim out_dim batch : UInt64} {d : DType} {dev : DevicePolicy} :
    Module (Linear in_dim out_dim d dev) (Tensor ⟨#[batch, in_dim], d, dev⟩)
      (Tensor ⟨#[batch, out_dim], d, dev⟩) where
  forward := Linear.forward2d

/-- Module instance for 3D forward pass -/
instance {in_dim out_dim batch seq : UInt64} {d : DType} {dev : DevicePolicy} :
    Module (Linear in_dim out_dim d dev) (Tensor ⟨#[batch, seq, in_dim], d, dev⟩)
      (Tensor ⟨#[batch, seq, out_dim], d, dev⟩) where
  forward := Linear.forward3d

end torch
