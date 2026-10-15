/-
  Tyr/Manifolds/Grassmann.lean

  Grassmann manifold Gr(n, p): p-dimensional subspaces of ℝⁿ.
-/
import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : DevicePolicy}

/-! ## Grassmann Manifold Gr(n, p)

The Grassmann manifold Gr(n, p) is the space of p-dimensional subspaces of ℝⁿ.

Key properties:
- Elements are equivalence classes of n×p matrices with orthonormal columns
- Two matrices X, Y represent the same subspace iff X = Y @ Q for some Q ∈ O(p)
- dim(Gr(n, p)) = p(n - p)
- Gr(n, 1) = ℝPⁿ⁻¹ (real projective space)

The tangent space at X is the horizontal space (orthogonal to the fiber):
  T_X Gr(n,p) = { Z ∈ ℝⁿˣᵖ : XᵀZ = 0 }

This differs from Stiefel where XᵀZ is skew-symmetric. For Grassmann, XᵀZ = 0 strictly.
-/

/-- Grassmann manifold Gr(n, p): p-dimensional subspaces of ℝⁿ.
    Represented by n×p matrices with orthonormal columns.
    Constraint: XᵀX = Iₚ, and tangent vectors satisfy XᵀZ = 0.
    The `tall` witness records that orthonormal p-column frames need n ≥ p. -/
structure Grassmann (n p : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  /-- Representative matrix with orthonormal columns -/
  matrix : Tensor ⟨#[n, p], dtype, device⟩
  /-- Orthonormal p-column frames in ℝⁿ require p ≤ n. -/
  tall : p.toNat ≤ n.toNat

namespace Grassmann

/-- Create a random point on Gr(n, p) via QR decomposition -/
def random (n p : UInt64) (h : p.toNat ≤ n.toNat := by decide) : IO (Grassmann n p) := do
  let mat ← Tensor.randn #[n, p]
  let (Q, _R) := Tensor.qrReduced mat h
  return ⟨Q, h⟩

/-- Project a general n×p matrix to Gr(n, p) using QR decomposition -/
def project (n p : UInt64) (mat : Tensor ⟨#[n, p], d, dev⟩)
    (h : p.toNat ≤ n.toNat := by decide) : Grassmann n p d dev :=
  let (Q, _R) := Tensor.qrReduced mat h
  ⟨Q, h⟩

/-- Standard basis subspace: span{e₁, ..., eₚ}
    Created by projecting a simple matrix to get orthonormal columns. -/
def standard (n p : UInt64) (h : p.toNat ≤ n.toNat := by decide) : Grassmann n p :=
  -- QR of ones matrix gives a valid orthonormal basis
  project n p (Tensor.ones #[n, p]) h

/-- Compute the projection matrix P = X @ Xᵀ onto the subspace -/
def projectionMatrix (X : Grassmann n p d dev) : Tensor ⟨#[n, n], d, dev⟩ :=
  Tensor.mm X.matrix (Tensor.transpose2d X.matrix)

/-- Principal angles between two subspaces.
    Returns the cosines of the principal angles (singular values of X₁ᵀX₂).
    The singular values σᵢ = cos(θᵢ) where θᵢ are the principal angles. -/
def principalAngles (X₁ X₂ : Grassmann n p d dev) : Tensor ⟨#[p], d, dev⟩ :=
  -- σ = svdvals(X₁ᵀ @ X₂)
  Tensor.svdvals (Tensor.mm (Tensor.transpose2d X₁.matrix) X₂.matrix)

/-- Chordal distance between two subspaces.
    d(X₁, X₂) = ||P₁ - P₂||_F / √2 = (Σᵢ sin²θᵢ)ᐟ², where P₁, P₂ are the
    orthogonal projectors and θᵢ the principal angles.
    Note: this is not the geodesic distance ||θ||₂ = (Σᵢ θᵢ²)ᐟ². -/
def distance (X₁ X₂ : Grassmann n p d dev) : Float :=
  let P₁ := projectionMatrix X₁
  let P₂ := projectionMatrix X₂
  let diff := Tensor.Preserving.sub P₁ P₂
  -- ||P₁ - P₂||_F / √2 is the chordal (projection-Frobenius) distance
  let frobSq := Tensor.item (Tensor.sumAll (Tensor.Preserving.mul diff diff))
  Float.sqrt frobSq / Float.sqrt 2.0

end Grassmann

/-- Tangent vector on the Grassmann manifold.
    Z ∈ T_X Gr(n,p) satisfies: XᵀZ = 0 (Z is orthogonal to columns of X) -/
structure GrassmannTangent (n p : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  /-- The tangent vector as an n×p matrix orthogonal to base point -/
  vec : Tensor ⟨#[n, p], dtype, device⟩

namespace GrassmannTangent

/-- Zero tangent vector -/
def zero (n p : UInt64) : GrassmannTangent n p d dev :=
  ⟨Tensor.zerosFor ⟨#[n, p], d, dev⟩⟩

/-- Add two tangent vectors -/
def add (v w : GrassmannTangent n p d dev) : GrassmannTangent n p d dev :=
  ⟨Tensor.Preserving.add v.vec w.vec⟩

/-- Scale a tangent vector -/
def smul (s : Float) (v : GrassmannTangent n p d dev) : GrassmannTangent n p d dev :=
  ⟨Tensor.Preserving.mulScalar v.vec s⟩

/-- Project an ambient space vector to the horizontal tangent space at X.
    Π_X(V) = (I - X @ Xᵀ) @ V
    This ensures XᵀΠ_X(V) = 0 -/
def project (X : Grassmann n p d dev) (V : Tensor ⟨#[n, p], d, dev⟩) : GrassmannTangent n p d dev :=
  -- (I - X @ X^T) @ V = V - X @ (X^T @ V)
  let XtV := Tensor.mm (Tensor.transpose2d X.matrix) V  -- [p, p]
  let XXtV := Tensor.mm X.matrix XtV                    -- [n, p]
  ⟨Tensor.Preserving.sub V XXtV⟩

/-- Inner product of tangent vectors (induced by Frobenius norm) -/
def inner (v w : GrassmannTangent n p d dev) : Float :=
  Tensor.item (Tensor.sumAll (Tensor.Preserving.mul v.vec w.vec))

/-- Norm of tangent vector -/
def norm (v : GrassmannTangent n p d dev) : Float :=
  Float.sqrt (inner v v)

end GrassmannTangent

namespace Grassmann

/-- Log map on Gr(n,p).
    Uses the canonical formula based on principal angles. -/
def log (X Y : Grassmann n p d dev) : GrassmannTangent n p d dev :=
  let XtY := Tensor.mm (Tensor.transpose2d X.matrix) Y.matrix        -- [p, p]
  let invXtY := Tensor.inv XtY
  let proj := Tensor.Preserving.sub Y.matrix (Tensor.mm X.matrix XtY)              -- (I - X Xᵀ) Y
  let A := Tensor.mm proj invXtY                                     -- [n, p]
  let (U, S, Vh) := Tensor.svdReduced A X.tall                       -- U [n,p], S [p], Vh [p,p]
  let theta := Tensor.Preserving.atan S
  let Theta := Tensor.diagflat theta
  ⟨Tensor.mm (Tensor.mm U Theta) Vh⟩

end Grassmann

/-- Grassmann manifold as a DifferentiableManifold.
    Uses QR-based retraction: R(X, Z) = qr(X + Z).Q -/
instance grassmannManifold (n p : UInt64) : DifferentiableManifold (Grassmann n p d dev) where
  Tangent _ := GrassmannTangent n p d dev
  Cotangent _ := GrassmannTangent n p d dev
  zeroTangent x := ⟨Tensor.zerosLike x.matrix⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.matrix⟩
  addTangent v w := GrassmannTangent.add v w
  addCotangent v w := GrassmannTangent.add v w
  scaleTangent s v := GrassmannTangent.smul s v
  sharp := id
  flat := id
  exp X Z :=
    -- QR-based retraction: R(X, Z) = qr(X + Z).Q
    -- This is a first-order approximation to the geodesic
    let (Q, _R) := Tensor.qrReduced (Tensor.Preserving.add X.matrix Z.vec) X.tall
    ⟨Q, X.tall⟩

end Tyr.AD
