/-
  Tyr/Manifolds/Stiefel.lean

  Stiefel manifold St(n, p): n×p matrices with orthonormal columns.
-/
import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : Device} [DType.RealFloating d]

/-! ## Stiefel Manifold St(n, p)

The Stiefel manifold St(n, p) consists of n×p matrices with orthonormal columns:
  St(n, p) = { X ∈ ℝⁿˣᵖ : XᵀX = Iₚ }

Special cases:
- St(n, n) = O(n), the orthogonal group
- St(n, 1) = Sⁿ⁻¹, the unit sphere

The tangent space at X is the horizontal space:
  TₓSt(n,p) = { Z ∈ ℝⁿˣᵖ : XᵀZ + ZᵀX = 0 }
            = { Z : XᵀZ is skew-symmetric }
-/

/-- Stiefel manifold St(n, p): n×p matrices with orthonormal columns.
    Constraint: XᵀX = Iₚ (p×p identity).
    The `tall` witness records that orthonormal p-column frames need n ≥ p. -/
structure Stiefel (n p : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  /-- The underlying n×p matrix -/
  matrix : Tensor ⟨#[n, p], dtype, device⟩
  /-- Orthonormal p-column frames in ℝⁿ require p ≤ n. -/
  tall : p.toNat ≤ n.toNat

namespace Stiefel

/-- Create a Stiefel element from the first p columns of a random matrix via QR.
    Uses QR decomposition of a random n×p matrix. -/
def random (n p : UInt64) (h : p.toNat ≤ n.toNat := by decide) : IO (Stiefel n p) := do
  let mat ← Tensor.randn #[n, p]
  -- QR decomposition gives Q with orthonormal columns
  let (Q, _R) := Tensor.qrReduced mat h
  return ⟨Q, h⟩

/-- Identity element for St(n, n) = O(n) -/
def identity [DType.ConstantSupported d dev] (n : UInt64) : Stiefel n n d dev :=
  ⟨Tensor.eyeFor d dev, Nat.le_refl _⟩

/-- Project a general matrix to the Stiefel manifold using QR decomposition.
    The Q factor of QR has orthonormal columns. -/
def project (n p : UInt64) (mat : Tensor ⟨#[n, p], d, dev⟩)
    (h : p.toNat ≤ n.toNat := by decide) : Stiefel n p d dev :=
  let (Q, _R) := Tensor.qrReduced mat h
  ⟨Q, h⟩

end Stiefel

/-- Tangent vector on the Stiefel manifold.
    Z ∈ TₓSt(n,p) satisfies: XᵀZ + ZᵀX = 0 (XᵀZ is skew-symmetric) -/
structure StiefelTangent (n p : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  /-- The tangent vector as an n×p matrix -/
  vec : Tensor ⟨#[n, p], dtype, device⟩

namespace StiefelTangent

/-- Zero tangent vector -/
def zero [DType.ConstantSupported d dev] (n p : UInt64) : StiefelTangent n p d dev :=
  ⟨Tensor.zerosFor ⟨#[n, p], d, dev⟩⟩

/-- Add two tangent vectors -/
def add (v w : StiefelTangent n p d dev) : StiefelTangent n p d dev :=
  ⟨Tensor.add v.vec w.vec⟩

/-- Scale a tangent vector -/
def smul (s : Float) (v : StiefelTangent n p d dev) : StiefelTangent n p d dev :=
  ⟨Tensor.mulScalar v.vec s⟩

/-- Project an ambient space vector to the tangent space at X.
    Π_X(Z) = Z - X sym(XᵀZ)
    where sym(A) = (A + Aᵀ)/2 -/
def project (X : Stiefel n p d dev) (Z : Tensor ⟨#[n, p], d, dev⟩) : StiefelTangent n p d dev :=
  -- X^T @ Z : [p, p]
  let XtZ := Tensor.mm (Tensor.transpose2d X.matrix) Z
  -- sym(X^T Z) = (X^T Z + (X^T Z)^T) / 2
  let sym := Tensor.mulScalar (Tensor.add XtZ (Tensor.transpose2d XtZ)) 0.5
  -- Z - X @ sym(X^T Z)
  ⟨Tensor.sub Z (Tensor.mm X.matrix sym)⟩

end StiefelTangent

/-- Stiefel manifold as a DifferentiableManifold.
    Uses QR retraction: R(X, Z) = qr(X + Z).Q -/
instance stiefelManifold (n p : UInt64) : DifferentiableManifold (Stiefel n p d dev) where
  Tangent _ := StiefelTangent n p d dev
  Cotangent _ := StiefelTangent n p d dev
  zeroTangent x := ⟨Tensor.zerosLike x.matrix⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.matrix⟩
  addTangent v w := StiefelTangent.add v w
  addCotangent v w := StiefelTangent.add v w
  scaleTangent s v := StiefelTangent.smul s v
  sharp := id
  flat := id
  exp X Z :=
    -- QR retraction: R(X, Z) = qr(X + Z).Q
    let (Q, _R) := Tensor.qrReduced (Tensor.add X.matrix Z.vec) X.tall
    ⟨Q, X.tall⟩

end Tyr.AD
