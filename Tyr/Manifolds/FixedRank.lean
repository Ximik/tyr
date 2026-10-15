import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : DevicePolicy}

/-- Embedded manifold of `m x n` matrices with rank constrained to `k`. -/
structure FixedRankEmbedded (m n k : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  matrix : Tensor ⟨#[m, n], dtype, device⟩
  rankBound : k.toNat ≤ m.toNat ∧ k.toNat ≤ n.toNat

namespace FixedRankEmbedded

/-- Low-rank factor-product approximation with rank at most `k`. -/
def project (m n k : UInt64) (A : Tensor ⟨#[m, n], d, dev⟩)
    (h : k.toNat ≤ m.toNat ∧ k.toNat ≤ n.toNat := by decide) : FixedRankEmbedded m n k d dev :=
  -- Practical low-rank projection without relying on backend SVD stability.
  -- This yields rank at most k by construction: (m×k) @ (k×n).
  let left : Tensor ⟨#[m, k], d, dev⟩ := Tensor.slice A 1 0 k (by simp) (by simpa using h.2)
  let right : Tensor ⟨#[k, n], d, dev⟩ := Tensor.slice A 0 0 k (by simp) (by simpa using h.1)
  let scale := if k == 0 then 1.0 else 1.0 / k.toFloat
  ⟨Tensor.Preserving.mulScalar (Tensor.mm left right) scale, h⟩

/-- Random rank-constrained point. -/
def random (m n k : UInt64)
    (h : k.toNat ≤ m.toNat ∧ k.toNat ≤ n.toNat := by decide) : IO (FixedRankEmbedded m n k) := do
  let A ← Tensor.randn #[m, n]
  pure (project m n k A h)

end FixedRankEmbedded

structure FixedRankTangent (m n k : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  vec : Tensor ⟨#[m, n], dtype, device⟩

namespace FixedRankTangent

def zero (m n k : UInt64) : FixedRankTangent m n k d dev := ⟨Tensor.zerosFor ⟨#[m, n], d, dev⟩⟩
def add (a b : FixedRankTangent m n k d dev) : FixedRankTangent m n k d dev := ⟨Tensor.Preserving.add a.vec b.vec⟩
def smul (s : Float) (v : FixedRankTangent m n k d dev) : FixedRankTangent m n k d dev := ⟨Tensor.Preserving.mulScalar v.vec s⟩

def project (x : FixedRankEmbedded m n k d dev) (V : Tensor ⟨#[m, n], d, dev⟩) : FixedRankTangent m n k d dev :=
  let _ := x
  ⟨V⟩

end FixedRankTangent

instance fixedRankEmbeddedManifold (m n k : UInt64) : DifferentiableManifold (FixedRankEmbedded m n k d dev) where
  Tangent _ := FixedRankTangent m n k d dev
  Cotangent _ := FixedRankTangent m n k d dev
  zeroTangent x := ⟨Tensor.zerosLike x.matrix⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.matrix⟩
  addTangent a b := FixedRankTangent.add a b
  addCotangent a b := FixedRankTangent.add a b
  scaleTangent s v := FixedRankTangent.smul s v
  sharp := id
  flat := id
  exp x v := FixedRankEmbedded.project m n k (Tensor.Preserving.add x.matrix v.vec) x.rankBound
  retract x v := FixedRankEmbedded.project m n k (Tensor.Preserving.add x.matrix v.vec) x.rankBound

end Tyr.AD
