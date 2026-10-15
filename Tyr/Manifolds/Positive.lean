import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : Device}

/-- Positive manifold of elementwise-positive `m x n` matrices. -/
structure Positive (m n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  matrix : Tensor ⟨#[m, n], dtype, device⟩

namespace Positive

private def eps : Float := 1e-6

/-- Project to positive matrices via clamping. -/
def project (m n : UInt64) (A : Tensor ⟨#[m, n], d, dev⟩) : Positive m n d dev :=
  ⟨Tensor.Preserving.clampFloat A eps 1e12⟩

/-- Random positive point. -/
def random (m n : UInt64) : IO (Positive m n) := do
  let A ← Tensor.randn #[m, n]
  pure (project m n (Tensor.Preserving.addScalar (Tensor.abs A) eps))

end Positive

/-- Tangent vectors on positive matrices are unconstrained ambient matrices. -/
structure PositiveTangent (m n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  vec : Tensor ⟨#[m, n], dtype, device⟩

namespace PositiveTangent

def zero (m n : UInt64) : PositiveTangent m n d dev := ⟨Tensor.zerosFor ⟨#[m, n], d, dev⟩⟩
def add (a b : PositiveTangent m n d dev) : PositiveTangent m n d dev := ⟨Tensor.Preserving.add a.vec b.vec⟩
def smul (s : Float) (v : PositiveTangent m n d dev) : PositiveTangent m n d dev := ⟨Tensor.Preserving.mulScalar v.vec s⟩

end PositiveTangent

instance positiveManifold (m n : UInt64) : DifferentiableManifold (Positive m n d dev) where
  Tangent _ := PositiveTangent m n d dev
  Cotangent _ := PositiveTangent m n d dev
  zeroTangent x := ⟨Tensor.zerosLike x.matrix⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.matrix⟩
  addTangent a b := PositiveTangent.add a b
  addCotangent a b := PositiveTangent.add a b
  scaleTangent s v := PositiveTangent.smul s v
  sharp := by
    intro x g
    let sq := Tensor.Preserving.mul x.matrix x.matrix
    exact ⟨Tensor.Preserving.mul g.vec sq⟩
  flat := by
    intro x v
    let sq := Tensor.Preserving.mul x.matrix x.matrix
    let sqSafe := Tensor.Preserving.addScalar sq 1e-12
    exact ⟨Tensor.Preserving.div v.vec sqSafe⟩
  exp x v :=
    let quad := Tensor.Preserving.mul v.vec v.vec
    let denom := Tensor.Preserving.mulScalar x.matrix 2.0
    let second := Tensor.Preserving.div quad (Tensor.Preserving.addScalar denom 1e-12)
    Positive.project m n (Tensor.Preserving.add (Tensor.Preserving.add x.matrix v.vec) second)
  retract x v :=
    let quad := Tensor.Preserving.mul v.vec v.vec
    let denom := Tensor.Preserving.mulScalar x.matrix 2.0
    let second := Tensor.Preserving.div quad (Tensor.Preserving.addScalar denom 1e-12)
    Positive.project m n (Tensor.Preserving.add (Tensor.Preserving.add x.matrix v.vec) second)

end Tyr.AD
