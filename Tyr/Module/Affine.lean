import Tyr.Module.Core
import Tyr.Module.Derive
import Tyr.Differentiable

namespace torch

/-- Affine map `x ↦ x w + b`. Unlike `Linear`, weights use `[input, output]` order. -/
structure Affine {n m : Nat} (dtype : DType := .Float32)
    (device : Device := .CPU) where
  w : Tensor ⟨#[m, n], dtype, device⟩
  b : Tensor ⟨#[n], dtype, device⟩

derive_tensor_struct_for Affine

instance {n m : Nat} {d : DType} {dev : Device} :
    TensorStruct (@Affine n m d dev) where
  map f a := ⟨f a.w, f a.b⟩
  mapM f a := do pure ⟨← f a.w, ← f a.b⟩
  zipWith f a b := ⟨f a.w b.w, f a.b b.b⟩
  fold f init a := f a.b (f a.w init)

instance {n m : Nat} {d : DType} {dev : Device} :
    differentiable (@Affine n m d dev) where
  cotangent_space := @Affine n m d dev
  grad a := do pure ⟨← Tensor.gradOf a.w, ← Tensor.gradOf a.b⟩

/-- Apply an affine map to a batch, retaining dtype and device. -/
def Affine.step {batch n m : Nat} {d : DType} {dev : Device} [DType.Promotes d d d]
    (a : @Affine n m d dev) (input : Tensor ⟨#[batch, m], d, dev⟩) :
    Tensor ⟨#[batch, n], d, dev⟩ :=
  Tensor.add (Tensor.mm input a.w) (Tensor.expand a.b #[batch, n])

instance {batch n m : Nat} {d : DType} {dev : Device} [DType.Promotes d d d] :
    Module (@Affine n m d dev) (Tensor ⟨#[batch, m], d, dev⟩)
      (Tensor ⟨#[batch, n], d, dev⟩) where
  forward := Affine.step

end torch
