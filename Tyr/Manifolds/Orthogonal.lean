/-
  Tyr/Manifolds/Orthogonal.lean

  Orthogonal group O(n): n×n orthogonal matrices.
  Includes SkewSymmetric matrices for tangent space representation.
-/
import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : DevicePolicy}

/-! ## Orthogonal Group O(n)

The orthogonal group O(n) is the special case of St(n, n):
  O(n) = { Q ∈ ℝⁿˣⁿ : QᵀQ = QQᵀ = I }

The tangent space at Q consists of matrices Q·S where S is skew-symmetric:
  T_Q O(n) = { Q·S : Sᵀ = -S }

The exponential map is: exp_Q(Q·S) = Q·exp(S)
where exp(S) is the matrix exponential.
-/

/-- Orthogonal group O(n): n×n orthogonal matrices.
    Constraint: QᵀQ = QQᵀ = I -/
structure Orthogonal (n : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  /-- The underlying n×n orthogonal matrix -/
  matrix : Tensor ⟨#[n, n], dtype, device⟩

namespace Orthogonal

/-- Identity element -/
def identity (n : UInt64) : Orthogonal n d dev :=
  ⟨Tensor.eyeFor d dev⟩

/-- Create a random orthogonal matrix via QR decomposition. -/
def random (n : UInt64) : IO (Orthogonal n) := do
  let mat ← Tensor.randn #[n, n]
  let (Q, _R) := Tensor.qrComplete mat
  return ⟨Q⟩

/-- Project a matrix to O(n) using QR decomposition.
    The Q factor is orthogonal. -/
def project (n : UInt64) (mat : Tensor ⟨#[n, n], d, dev⟩) : Orthogonal n d dev :=
  let (Q, _R) := Tensor.qrComplete mat
  ⟨Q⟩

/-- Transpose (also inverse for orthogonal matrices). -/
def transpose (Q : Orthogonal n d dev) : Orthogonal n d dev :=
  ⟨Tensor.transpose2d Q.matrix⟩

/-- Matrix multiplication (group operation). -/
def mul (Q R : Orthogonal n d dev) : Orthogonal n d dev :=
  ⟨Tensor.mm Q.matrix R.matrix⟩

end Orthogonal

/-- Skew-symmetric matrix: Sᵀ = -S
    Used for tangent space representation of O(n) -/
structure SkewSymmetric (n : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  /-- The underlying matrix (should satisfy Sᵀ = -S) -/
  matrix : Tensor ⟨#[n, n], dtype, device⟩

namespace SkewSymmetric

/-- Zero skew-symmetric matrix -/
def zero (n : UInt64) : SkewSymmetric n d dev :=
  ⟨Tensor.zerosFor ⟨#[n, n], d, dev⟩⟩

/-- Create a skew-symmetric matrix from any matrix: (A - Aᵀ)/2 -/
def fromMatrix (mat : Tensor ⟨#[n, n], d, dev⟩) : SkewSymmetric n d dev :=
  ⟨Tensor.Preserving.mulScalar (Sub.sub mat (Tensor.transpose2d mat)) 0.5⟩

/-- Add two skew-symmetric matrices -/
def add (S T : SkewSymmetric n d dev) : SkewSymmetric n d dev :=
  ⟨Add.add S.matrix T.matrix⟩

/-- Scale a skew-symmetric matrix -/
def smul (s : Float) (S : SkewSymmetric n d dev) : SkewSymmetric n d dev :=
  ⟨Tensor.Preserving.mulScalar S.matrix s⟩

/-- Frobenius inner product -/
def inner (S T : SkewSymmetric n d dev) : Float :=
  Tensor.item (Tensor.sumAll (Mul.mul S.matrix T.matrix))

end SkewSymmetric

/-- Tangent vector on the orthogonal group.
    Represented as Q·S where S is skew-symmetric.
    We store S directly since Q is known from the base point. -/
structure OrthogonalTangent (n : UInt64) (dtype : DType := .Float32)
    (device : DevicePolicy := .any) where
  /-- The skew-symmetric component S such that tangent = Q·S -/
  skew : SkewSymmetric n dtype device

namespace OrthogonalTangent

/-- Zero tangent -/
def zero (n : UInt64) : OrthogonalTangent n d dev :=
  ⟨SkewSymmetric.zero n⟩

/-- Create tangent from skew-symmetric matrix -/
def fromSkew (S : SkewSymmetric n d dev) : OrthogonalTangent n d dev := ⟨S⟩

/-- Add tangent vectors -/
def add (v w : OrthogonalTangent n d dev) : OrthogonalTangent n d dev :=
  ⟨SkewSymmetric.add v.skew w.skew⟩

/-- Scale tangent vector -/
def smul (s : Float) (v : OrthogonalTangent n d dev) : OrthogonalTangent n d dev :=
  ⟨SkewSymmetric.smul s v.skew⟩

/-- Get the ambient representation Q·S -/
def toAmbient (Q : Orthogonal n d dev) (v : OrthogonalTangent n d dev) : Tensor ⟨#[n, n], d, dev⟩ :=
  Tensor.mm Q.matrix v.skew.matrix

/-- Project an ambient tangent to the horizontal space -/
def fromAmbient (_Q : Orthogonal n d dev) (Z : Tensor ⟨#[n, n], d, dev⟩) : OrthogonalTangent n d dev :=
  ⟨SkewSymmetric.fromMatrix Z⟩

end OrthogonalTangent

/-- Orthogonal group as a DifferentiableManifold.
    Uses the matrix exponential for the proper exponential map: exp_Q(Q·S) = Q·exp(S). -/
instance orthogonalManifold (n : UInt64) : DifferentiableManifold (Orthogonal n d dev) where
  Tangent _ := OrthogonalTangent n d dev
  Cotangent _ := OrthogonalTangent n d dev
  zeroTangent _ := OrthogonalTangent.zero n
  zeroCotangent _ := OrthogonalTangent.zero n
  addTangent v w := OrthogonalTangent.add v w
  addCotangent v w := OrthogonalTangent.add v w
  scaleTangent s v := OrthogonalTangent.smul s v
  sharp := id
  flat := id
  exp Q v :=
    -- Proper exponential map: exp_Q(Q·S) = Q · exp(S)
    -- where S is skew-symmetric, so exp(S) is orthogonal
    let expS := Tensor.matrixExp v.skew.matrix
    let newQ := Tensor.mm Q.matrix expS
    -- QR for numerical stability (exp(S) should be orthogonal, but floating point...)
    let (Q', _R) := Tensor.qrComplete newQ
    ⟨Q'⟩

end Tyr.AD
