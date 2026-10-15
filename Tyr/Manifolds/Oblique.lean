import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : Device}

/-- Oblique manifold OB(m,n): matrices with unit-norm columns. -/
structure Oblique (m n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  matrix : Tensor ⟨#[m, n], dtype, device⟩

namespace Oblique

private def normalizeColumns {m n : UInt64} (A : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[m, n], d, dev⟩ :=
  let AtA := Tensor.mm (Tensor.transpose2d A) A
  let colNorms := Tensor.Preserving.sqrt (Tensor.diag AtA)
  let safeNorms := Tensor.Preserving.addScalar colNorms 1e-12
  let rowNorms : Tensor ⟨#[1, n], d, dev⟩ := Tensor.reshape safeNorms #[1, n] (by simp [TensorSpec.numelOfShape])
  let expanded : Tensor ⟨#[m, n], d, dev⟩ := Tensor.expand rowNorms #[m, n]
  Tensor.Preserving.div A expanded

/-- Project ambient matrix to the oblique manifold by column normalization. -/
def project (m n : UInt64) (A : Tensor ⟨#[m, n], d, dev⟩) : Oblique m n d dev :=
  ⟨normalizeColumns A⟩

/-- Random point on the oblique manifold. -/
def random (m n : UInt64) : IO (Oblique m n) := do
  let A ← Tensor.randn #[m, n]
  pure (project m n A)

end Oblique

/-- Tangent vector on OB(m,n). -/
structure ObliqueTangent (m n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  vec : Tensor ⟨#[m, n], dtype, device⟩

namespace ObliqueTangent

/-- Project ambient direction to tangent space at `X`. -/
def project (X : Oblique m n d dev) (V : Tensor ⟨#[m, n], d, dev⟩) : ObliqueTangent m n d dev :=
  let colDotsRow : Tensor ⟨#[1, n], d, dev⟩ :=
    Tensor.Preserving.sumDim (Tensor.Preserving.mul X.matrix V) 0 true
  let colDots : Tensor ⟨#[m, n], d, dev⟩ := Tensor.expand colDotsRow #[m, n]
  ⟨Tensor.Preserving.sub V (Tensor.Preserving.mul X.matrix colDots)⟩

/-- Zero tangent vector. -/
def zero (m n : UInt64) : ObliqueTangent m n d dev := ⟨Tensor.zerosFor ⟨#[m, n], d, dev⟩⟩

/-- Add tangent vectors. -/
def add (a b : ObliqueTangent m n d dev) : ObliqueTangent m n d dev :=
  ⟨Tensor.Preserving.add a.vec b.vec⟩

/-- Scale tangent vector. -/
def smul (s : Float) (v : ObliqueTangent m n d dev) : ObliqueTangent m n d dev :=
  ⟨Tensor.Preserving.mulScalar v.vec s⟩

end ObliqueTangent

instance obliqueManifold (m n : UInt64) : DifferentiableManifold (Oblique m n d dev) where
  Tangent _ := ObliqueTangent m n d dev
  Cotangent _ := ObliqueTangent m n d dev
  zeroTangent x := ⟨Tensor.zerosLike x.matrix⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.matrix⟩
  addTangent a b := ObliqueTangent.add a b
  addCotangent a b := ObliqueTangent.add a b
  scaleTangent s v := ObliqueTangent.smul s v
  sharp := id
  flat := id
  exp x v :=
    Oblique.project m n (Tensor.Preserving.add x.matrix v.vec)
  retract x v :=
    Oblique.project m n (Tensor.Preserving.add x.matrix v.vec)

end Tyr.AD
