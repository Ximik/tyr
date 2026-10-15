import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : Device}

/-- Poincare ball model with ambient coordinates in `R^n`, `||x|| < 1`. -/
structure PoincareBall (n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  coords : Tensor ⟨#[n], dtype, device⟩

namespace PoincareBall

private def eps : Float := 1e-6

/-- Project ambient vector into the open unit ball (with epsilon margin). -/
def project (n : UInt64) (v : Tensor ⟨#[n], d, dev⟩) : PoincareBall n d dev :=
  let norm := Tensor.l2Norm v
  let maxNorm := 1.0 - eps
  if norm < maxNorm then
    ⟨v⟩
  else
    ⟨Tensor.Preserving.mulScalar v (maxNorm / (norm + 1e-12))⟩

/-- Random point in the unit ball. -/
def random (n : UInt64) : IO (PoincareBall n) := do
  let v ← Tensor.randn #[n]
  pure (project n (Tensor.Preserving.mulScalar v 0.5))

/-- Conformal factor lambda(x) = 2 / (1 - ||x||^2). -/
def conformalFactor (x : PoincareBall n d dev) : Float :=
  let n2 := Tensor.l2Norm x.coords
  2.0 / (1.0 - n2 * n2 + 1e-12)

end PoincareBall

structure PoincareBallTangent (n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  vec : Tensor ⟨#[n], dtype, device⟩

namespace PoincareBallTangent

def zero [DType.ConstantSupported d dev] (n : UInt64) : PoincareBallTangent n d dev := ⟨Tensor.zerosFor ⟨#[n], d, dev⟩⟩
def add (a b : PoincareBallTangent n d dev) : PoincareBallTangent n d dev := ⟨Tensor.Preserving.add a.vec b.vec⟩
def smul (s : Float) (v : PoincareBallTangent n d dev) : PoincareBallTangent n d dev := ⟨Tensor.Preserving.mulScalar v.vec s⟩

end PoincareBallTangent

instance poincareBallManifold (n : UInt64) : DifferentiableManifold (PoincareBall n d dev) where
  Tangent _ := PoincareBallTangent n d dev
  Cotangent _ := PoincareBallTangent n d dev
  zeroTangent x := ⟨Tensor.zerosLike x.coords⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.coords⟩
  addTangent a b := PoincareBallTangent.add a b
  addCotangent a b := PoincareBallTangent.add a b
  scaleTangent s v := PoincareBallTangent.smul s v
  sharp := id
  flat := id
  exp x v := PoincareBall.project n (Tensor.Preserving.add x.coords v.vec)
  retract x v := PoincareBall.project n (Tensor.Preserving.add x.coords v.vec)

end Tyr.AD
