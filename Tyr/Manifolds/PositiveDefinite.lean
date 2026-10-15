import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : Device} [DType.RealFloating d]

/-- Symmetric positive definite manifold SPD(n). -/
structure SymmetricPositiveDefinite (n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  matrix : Tensor ⟨#[n, n], dtype, device⟩

namespace SymmetricPositiveDefinite

private def symm {n : UInt64} (A : Tensor ⟨#[n, n], d, dev⟩) : Tensor ⟨#[n, n], d, dev⟩ :=
  Tensor.mulScalar (Tensor.add A (Tensor.transpose2d A)) 0.5

/-- Project ambient matrix to SPD via `exp(sym(A))`. -/
def project (n : UInt64) (A : Tensor ⟨#[n, n], d, dev⟩) : SymmetricPositiveDefinite n d dev :=
  ⟨Tensor.matrixExp (symm A)⟩

/-- Random SPD point using exponential parameterization. -/
def random (n : UInt64) : IO (SymmetricPositiveDefinite n) := do
  let A ← Tensor.randn #[n, n]
  pure (project n A)

/-- Affine-invariant inspired distance proxy using `log(inv(A) B)`. -/
def distance [DType.RealFloating d] (A B : SymmetricPositiveDefinite n d dev) : Float :=
  let rel := Tensor.mm (Tensor.inv A.matrix) B.matrix
  let logRel := Tensor.matrixLog rel
  Tensor.frobeniusNorm logRel

end SymmetricPositiveDefinite

structure SPDTangent (n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  vec : Tensor ⟨#[n, n], dtype, device⟩

namespace SPDTangent

def zero [DType.ConstantSupported d dev] (n : UInt64) : SPDTangent n d dev := ⟨Tensor.zerosFor ⟨#[n, n], d, dev⟩⟩
def add (a b : SPDTangent n d dev) : SPDTangent n d dev := ⟨Tensor.add a.vec b.vec⟩
def smul (s : Float) (v : SPDTangent n d dev) : SPDTangent n d dev := ⟨Tensor.mulScalar v.vec s⟩

def project (x : SymmetricPositiveDefinite n d dev) (V : Tensor ⟨#[n, n], d, dev⟩) : SPDTangent n d dev :=
  let _ := x
  ⟨Tensor.mulScalar (Tensor.add V (Tensor.transpose2d V)) 0.5⟩

end SPDTangent

instance spdManifold (n : UInt64) [DType.RealFloating d] :
    DifferentiableManifold (SymmetricPositiveDefinite n d dev) where
  Tangent _ := SPDTangent n d dev
  Cotangent _ := SPDTangent n d dev
  zeroTangent x := ⟨Tensor.zerosLike x.matrix⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.matrix⟩
  addTangent a b := SPDTangent.add a b
  addCotangent a b := SPDTangent.add a b
  scaleTangent s v := SPDTangent.smul s v
  sharp := id
  flat := id
  exp x v :=
    let lx := Tensor.matrixLog x.matrix
    SymmetricPositiveDefinite.project n (Tensor.add lx v.vec)
  retract x v :=
    let lx := Tensor.matrixLog x.matrix
    SymmetricPositiveDefinite.project n (Tensor.add lx v.vec)

end Tyr.AD
