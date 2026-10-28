import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : Device} [DType.RealFloating d]

/-- Quotient-factor representation for fixed-rank PSD matrices via `Y in R^{n x k}`. -/
structure PSDFixedRank (n k : Nat) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  factor : Tensor ⟨#[n, k], dtype, device⟩

namespace PSDFixedRank

/-- Ambient matrix in the PSD cone represented by this factor. -/
def toPSD (x : PSDFixedRank n k d dev) : Tensor ⟨#[n, n], d, dev⟩ :=
  Tensor.mm x.factor (Tensor.transpose2d x.factor)

/-- Random point (factor matrix). -/
def random (n k : Nat) : IO (PSDFixedRank n k) := do
  let y ← Tensor.randn #[n, k]
  pure ⟨y⟩

end PSDFixedRank

structure PSDFixedRankTangent (n k : Nat) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  vec : Tensor ⟨#[n, k], dtype, device⟩

namespace PSDFixedRankTangent

def zero [DType.ConstantSupported d dev] (n k : Nat) : PSDFixedRankTangent n k d dev := ⟨Tensor.zerosFor ⟨#[n, k], d, dev⟩⟩
def add (a b : PSDFixedRankTangent n k d dev) : PSDFixedRankTangent n k d dev := ⟨Tensor.add a.vec b.vec⟩
def smul (s : Float) (v : PSDFixedRankTangent n k d dev) : PSDFixedRankTangent n k d dev := ⟨Tensor.mulScalar v.vec s⟩

/-- Horizontal projection proxy akin Stiefel projection in factor space. -/
def project (x : PSDFixedRank n k d dev) (V : Tensor ⟨#[n, k], d, dev⟩) : PSDFixedRankTangent n k d dev :=
  let YtV := Tensor.mm (Tensor.transpose2d x.factor) V
  let sym := Tensor.mulScalar (Tensor.add YtV (Tensor.transpose2d YtV)) 0.5
  ⟨Tensor.sub V (Tensor.mm x.factor sym)⟩

end PSDFixedRankTangent

instance psdFixedRankManifold (n k : Nat) : DifferentiableManifold (PSDFixedRank n k d dev) where
  Tangent _ := PSDFixedRankTangent n k d dev
  Cotangent _ := PSDFixedRankTangent n k d dev
  zeroTangent x := ⟨Tensor.zerosLike x.factor⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.factor⟩
  addTangent a b := PSDFixedRankTangent.add a b
  addCotangent a b := PSDFixedRankTangent.add a b
  scaleTangent s v := PSDFixedRankTangent.smul s v
  sharp := id
  flat := id
  exp x v := ⟨Tensor.add x.factor v.vec⟩
  retract x v := ⟨Tensor.add x.factor v.vec⟩

/-- Elliptope factor manifold: row-wise unit norms in `R^{n x k}`. -/
structure Elliptope (n k : Nat) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  factor : Tensor ⟨#[n, k], dtype, device⟩

namespace Elliptope

private def normalizeRows {n k : Nat} (Y : Tensor ⟨#[n, k], d, dev⟩) : Tensor ⟨#[n, k], d, dev⟩ :=
  let norms := Tensor.rowNorms Y
  let safeNorms := Tensor.addScalar norms 1e-12
  let normsCol : Tensor ⟨#[n, 1], d.realResult, dev⟩ := Tensor.reshape safeNorms #[n, 1] (by simp [TensorSpec.numelOfShape])
  let expanded : Tensor ⟨#[n, k], d, dev⟩ := by simpa using Tensor.expand normsCol #[n, k]
  by simpa using Tensor.div Y expanded

/-- Project ambient matrix to row-normalized elliptope factor. -/
def project (n k : Nat) (Y : Tensor ⟨#[n, k], d, dev⟩) : Elliptope n k d dev :=
  ⟨normalizeRows Y⟩

/-- Random row-normalized factor. -/
def random (n k : Nat) : IO (Elliptope n k) := do
  let y ← Tensor.randn #[n, k]
  pure (project n k y)

end Elliptope

structure ElliptopeTangent (n k : Nat) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  vec : Tensor ⟨#[n, k], dtype, device⟩

namespace ElliptopeTangent

def zero [DType.ConstantSupported d dev] (n k : Nat) : ElliptopeTangent n k d dev := ⟨Tensor.zerosFor ⟨#[n, k], d, dev⟩⟩
def add (a b : ElliptopeTangent n k d dev) : ElliptopeTangent n k d dev := ⟨Tensor.add a.vec b.vec⟩
def smul (s : Float) (v : ElliptopeTangent n k d dev) : ElliptopeTangent n k d dev := ⟨Tensor.mulScalar v.vec s⟩

/-- Project ambient direction to row-wise tangent space (`<y_i, v_i> = 0`). -/
def project (x : Elliptope n k d dev) (V : Tensor ⟨#[n, k], d, dev⟩) : ElliptopeTangent n k d dev :=
  let rowDotsCol : Tensor ⟨#[n, 1], d, dev⟩ :=
    by simpa using Tensor.sumDim (Tensor.mul x.factor V) 1 true
  let rowDots : Tensor ⟨#[n, k], d, dev⟩ := Tensor.expand rowDotsCol #[n, k]
  ⟨Tensor.sub V (Tensor.mul x.factor rowDots)⟩

end ElliptopeTangent

instance elliptopeManifold (n k : Nat) : DifferentiableManifold (Elliptope n k d dev) where
  Tangent _ := ElliptopeTangent n k d dev
  Cotangent _ := ElliptopeTangent n k d dev
  zeroTangent x := ⟨Tensor.zerosLike x.factor⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.factor⟩
  addTangent a b := ElliptopeTangent.add a b
  addCotangent a b := ElliptopeTangent.add a b
  scaleTangent s v := ElliptopeTangent.smul s v
  sharp := id
  flat := id
  exp x v := Elliptope.project n k (Tensor.add x.factor v.vec)
  retract x v := Elliptope.project n k (Tensor.add x.factor v.vec)

end Tyr.AD
