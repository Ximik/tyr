import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : DevicePolicy}

/-- Quotient-factor representation for fixed-rank PSD matrices via `Y in R^{n x k}`. -/
structure PSDFixedRank (n k : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  factor : Tensor ⟨#[n, k], dtype, device⟩

namespace PSDFixedRank

/-- Ambient matrix in the PSD cone represented by this factor. -/
def toPSD (x : PSDFixedRank n k d dev) : Tensor ⟨#[n, n], d, dev⟩ :=
  Tensor.mm x.factor (Tensor.transpose2d x.factor)

/-- Random point (factor matrix). -/
def random (n k : UInt64) : IO (PSDFixedRank n k) := do
  let y ← Tensor.randn #[n, k]
  pure ⟨y⟩

end PSDFixedRank

structure PSDFixedRankTangent (n k : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  vec : Tensor ⟨#[n, k], dtype, device⟩

namespace PSDFixedRankTangent

def zero (n k : UInt64) : PSDFixedRankTangent n k d dev := ⟨Tensor.zerosFor ⟨#[n, k], d, dev⟩⟩
def add (a b : PSDFixedRankTangent n k d dev) : PSDFixedRankTangent n k d dev := ⟨Add.add a.vec b.vec⟩
def smul (s : Float) (v : PSDFixedRankTangent n k d dev) : PSDFixedRankTangent n k d dev := ⟨Tensor.Preserving.mulScalar v.vec s⟩

/-- Horizontal projection proxy akin Stiefel projection in factor space. -/
def project (x : PSDFixedRank n k d dev) (V : Tensor ⟨#[n, k], d, dev⟩) : PSDFixedRankTangent n k d dev :=
  let YtV := Tensor.mm (Tensor.transpose2d x.factor) V
  let sym := Tensor.Preserving.mulScalar (Add.add YtV (Tensor.transpose2d YtV)) 0.5
  ⟨Sub.sub V (Tensor.mm x.factor sym)⟩

end PSDFixedRankTangent

instance psdFixedRankManifold (n k : UInt64) : DifferentiableManifold (PSDFixedRank n k d dev) where
  Tangent _ := PSDFixedRankTangent n k d dev
  Cotangent _ := PSDFixedRankTangent n k d dev
  zeroTangent x := ⟨Tensor.zerosLike x.factor⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.factor⟩
  addTangent a b := PSDFixedRankTangent.add a b
  addCotangent a b := PSDFixedRankTangent.add a b
  scaleTangent s v := PSDFixedRankTangent.smul s v
  sharp := id
  flat := id
  exp x v := ⟨Add.add x.factor v.vec⟩
  retract x v := ⟨Add.add x.factor v.vec⟩

/-- Elliptope factor manifold: row-wise unit norms in `R^{n x k}`. -/
structure Elliptope (n k : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  factor : Tensor ⟨#[n, k], dtype, device⟩

namespace Elliptope

private def normalizeRows {n k : UInt64} (Y : Tensor ⟨#[n, k], d, dev⟩) : Tensor ⟨#[n, k], d, dev⟩ :=
  let norms := Tensor.rowNorms Y
  let safeNorms := Tensor.Preserving.addScalar norms 1e-12
  let normsCol : Tensor ⟨#[n, 1], d, dev⟩ := Tensor.reshape safeNorms #[n, 1] (by simp [TensorSpec.numelOfShape])
  let expanded : Tensor ⟨#[n, k], d, dev⟩ := Tensor.expand normsCol #[n, k]
  Tensor.Preserving.div Y expanded

/-- Project ambient matrix to row-normalized elliptope factor. -/
def project (n k : UInt64) (Y : Tensor ⟨#[n, k], d, dev⟩) : Elliptope n k d dev :=
  ⟨normalizeRows Y⟩

/-- Random row-normalized factor. -/
def random (n k : UInt64) : IO (Elliptope n k) := do
  let y ← Tensor.randn #[n, k]
  pure (project n k y)

end Elliptope

structure ElliptopeTangent (n k : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  vec : Tensor ⟨#[n, k], dtype, device⟩

namespace ElliptopeTangent

def zero (n k : UInt64) : ElliptopeTangent n k d dev := ⟨Tensor.zerosFor ⟨#[n, k], d, dev⟩⟩
def add (a b : ElliptopeTangent n k d dev) : ElliptopeTangent n k d dev := ⟨Add.add a.vec b.vec⟩
def smul (s : Float) (v : ElliptopeTangent n k d dev) : ElliptopeTangent n k d dev := ⟨Tensor.Preserving.mulScalar v.vec s⟩

/-- Project ambient direction to row-wise tangent space (`<y_i, v_i> = 0`). -/
def project (x : Elliptope n k d dev) (V : Tensor ⟨#[n, k], d, dev⟩) : ElliptopeTangent n k d dev :=
  let rowDotsCol : Tensor ⟨#[n, 1], d, dev⟩ :=
    Tensor.Preserving.sumDim (Mul.mul x.factor V) 1 true
  let rowDots : Tensor ⟨#[n, k], d, dev⟩ := Tensor.expand rowDotsCol #[n, k]
  ⟨Sub.sub V (Mul.mul x.factor rowDots)⟩

end ElliptopeTangent

instance elliptopeManifold (n k : UInt64) : DifferentiableManifold (Elliptope n k d dev) where
  Tangent _ := ElliptopeTangent n k d dev
  Cotangent _ := ElliptopeTangent n k d dev
  zeroTangent x := ⟨Tensor.zerosLike x.factor⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.factor⟩
  addTangent a b := ElliptopeTangent.add a b
  addCotangent a b := ElliptopeTangent.add a b
  scaleTangent s v := ElliptopeTangent.smul s v
  sharp := id
  flat := id
  exp x v := Elliptope.project n k (Add.add x.factor v.vec)
  retract x v := Elliptope.project n k (Add.add x.factor v.vec)

end Tyr.AD
