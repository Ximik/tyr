import Tyr.Module.Core
import Tyr.Differentiable

namespace torch

/-- Affine map `x ↦ x w + b`. Unlike `Linear`, weights use `[input, output]` order. -/
structure Affine {n m : UInt64} (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  w : Tensor ⟨#[m, n], dtype, device⟩
  b : Tensor ⟨#[n], dtype, device⟩

instance {n m : UInt64} {d : DType} {dev : DevicePolicy} :
    TensorStruct (@Affine n m d dev) where
  map f a := ⟨f a.w, f a.b⟩
  mapM f a := do pure ⟨← f a.w, ← f a.b⟩
  zipWith f a b := ⟨f a.w b.w, f a.b b.b⟩
  fold f init a := f a.b (f a.w init)

instance {n m : UInt64} {d : DType} {dev : DevicePolicy} :
    differentiable (@Affine n m d dev) where
  cotangent_space := @Affine n m d dev
  grad a := do pure ⟨← Tensor.gradOf a.w, ← Tensor.gradOf a.b⟩

/-- Apply an affine map to a batch, retaining dtype and device policy. -/
def Affine.step {batch n m : UInt64} {d : DType} {dev : DevicePolicy}
    (a : @Affine n m d dev) (input : Tensor ⟨#[batch, m], d, dev⟩) :
    Tensor ⟨#[batch, n], d, dev⟩ :=
  Add.add (Tensor.mm input a.w) (Tensor.expand a.b #[batch, n])

instance {batch n m : UInt64} {d : DType} {dev : DevicePolicy} :
    Module (@Affine n m d dev) (Tensor ⟨#[batch, m], d, dev⟩)
      (Tensor ⟨#[batch, n], d, dev⟩) where
  forward := Affine.step

end torch
