import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : Device} [DType.RealFloating d]

/-- Positive manifold of elementwise-positive `m x n` matrices. -/
structure Positive (m n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  matrix : Tensor ⟨#[m, n], dtype, device⟩

namespace Positive

private def eps : Float := 1e-6

/-- Project to positive matrices via clamping. -/
def project (m n : UInt64) (A : Tensor ⟨#[m, n], d, dev⟩) : Positive m n d dev :=
  ⟨Tensor.clampFloat A eps 1e12⟩

/-- Random positive point. -/
def random (m n : UInt64) : IO (Positive m n) := do
  let A ← Tensor.randn #[m, n]
  pure (project m n (Tensor.addScalar (Tensor.abs A) eps))

end Positive

/-- Tangent vectors on positive matrices are unconstrained ambient matrices. -/
structure PositiveTangent (m n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  vec : Tensor ⟨#[m, n], dtype, device⟩

namespace PositiveTangent

def zero [DType.ConstantSupported d dev] (m n : UInt64) : PositiveTangent m n d dev := ⟨Tensor.zerosFor ⟨#[m, n], d, dev⟩⟩
def add (a b : PositiveTangent m n d dev) : PositiveTangent m n d dev := ⟨Tensor.add a.vec b.vec⟩
def smul (s : Float) (v : PositiveTangent m n d dev) : PositiveTangent m n d dev := ⟨Tensor.mulScalar v.vec s⟩

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
    let sq := Tensor.mul x.matrix x.matrix
    exact ⟨Tensor.mul g.vec sq⟩
  flat := by
    intro x v
    let sq := Tensor.mul x.matrix x.matrix
    let sqSafe := Tensor.addScalar sq 1e-12
    exact ⟨by simpa using Tensor.div v.vec sqSafe⟩
  exp x v :=
    let quad := Tensor.mul v.vec v.vec
    let denom := Tensor.mulScalar x.matrix 2.0
    let second : Tensor ⟨#[m, n], d, dev⟩ := by
      simpa using Tensor.div quad (Tensor.addScalar denom 1e-12)
    Positive.project m n (Tensor.add (Tensor.add x.matrix v.vec) second)
  retract x v :=
    let quad := Tensor.mul v.vec v.vec
    let denom := Tensor.mulScalar x.matrix 2.0
    let second : Tensor ⟨#[m, n], d, dev⟩ := by
      simpa using Tensor.div quad (Tensor.addScalar denom 1e-12)
    Positive.project m n (Tensor.add (Tensor.add x.matrix v.vec) second)

end Tyr.AD
