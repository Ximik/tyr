/-
  Tyr/Manifolds/Basic.lean

  Core differentiable manifold infrastructure.
  Provides the DifferentiableManifold typeclass and EuclideanSpace instances.
-/
import Tyr.TensorStruct

namespace Tyr.AD

open torch (Static TensorStruct)

/-! ## DifferentiableManifold Typeclass

A mathematically rigorous design based on differential geometry:

1. **The space (manifold M)** - The type α itself
2. **Tangent space (TₓM)** - For JVP/forward-mode, infinitesimal directions at a point
3. **Cotangent space (T*ₓM)** - For VJP/reverse-mode, linear functionals on tangent vectors
4. **Musical isomorphism (♭/♯)** - Mapping between tangent and cotangent via a metric
5. **Exponential map (exp)** - Moving a point in the space using a tangent vector

This enables proper Riemannian optimization on curved manifolds.
-/

/-- A differentiable manifold with tangent and cotangent bundles.

    Mathematically:
    - M is the manifold (type α)
    - TₓM is the tangent space at point x (Tangent x)
    - T*ₓM is the cotangent space at point x (Cotangent x)
    - ♯ (sharp): T*M → TM raises indices using the metric
    - ♭ (flat): TM → T*M lowers indices using the metric
    - exp: M × TM → M is the exponential map (or retraction)
-/
class DifferentiableManifold (M : Type) where
  /-- Tangent space at a point. For Euclidean spaces, this is M itself. -/
  Tangent : M → Type
  /-- Cotangent space at a point. For Euclidean spaces, this is M itself. -/
  Cotangent : M → Type

  /-- Zero tangent vector at a point -/
  zeroTangent (x : M) : Tangent x
  /-- Zero cotangent at a point -/
  zeroCotangent (x : M) : Cotangent x

  /-- Add tangent vectors (TₓM is a vector space) -/
  addTangent {x : M} : Tangent x → Tangent x → Tangent x
  /-- Add cotangents (T*ₓM is a vector space) -/
  addCotangent {x : M} : Cotangent x → Cotangent x → Cotangent x

  /-- Scale a tangent vector -/
  scaleTangent {x : M} (s : Float) : Tangent x → Tangent x

  /-- Musical isomorphism ♯ (sharp): Cotangent → Tangent
      Uses the Riemannian metric to "raise indices" -/
  sharp {x : M} : Cotangent x → Tangent x

  /-- Musical isomorphism ♭ (flat): Tangent → Cotangent
      Uses the Riemannian metric to "lower indices" -/
  flat {x : M} : Tangent x → Cotangent x

  /-- Exponential map: Move from x in direction v
      For Euclidean spaces: exp(x, v) = x + v
      For curved manifolds: follows geodesic from x with initial velocity v -/
  exp (x : M) : Tangent x → M

  /-- Retraction (computationally cheaper approximation to exp) -/
  retract (x : M) : Tangent x → M := exp x

namespace DifferentiableManifold

/-- Simplified typeclass for Euclidean spaces where T = T* = M.
    These spaces have a flat metric where sharp/flat are identity. -/
class EuclideanSpace (M : Type) where
  /-- Zero element -/
  zero : M
  /-- Addition -/
  add : M → M → M
  /-- Scalar multiplication -/
  smul : Float → M → M
  /-- Inner product (defines the Euclidean metric) -/
  inner : M → M → Float

/-- Euclidean spaces are automatically DifferentiableManifolds -/
instance euclideanManifold [E : EuclideanSpace M] : DifferentiableManifold M where
  Tangent _ := M
  Cotangent _ := M
  zeroTangent _ := E.zero
  zeroCotangent _ := E.zero
  addTangent := E.add
  addCotangent := E.add
  scaleTangent s v := E.smul s v
  -- For Euclidean metric, sharp and flat are identity
  sharp := id
  flat := id
  -- exp(x, v) = x + v
  exp x v := E.add x v

/-- Float is a Euclidean space (1-dimensional) -/
instance floatEuclidean : EuclideanSpace Float where
  zero := 0.0
  add := (· + ·)
  smul s x := s * x
  inner a b := a * b

/-- Static types are 0-dimensional manifolds (no tangent directions) -/
instance staticManifold {α : Type} : DifferentiableManifold (Static α) where
  Tangent _ := Unit
  Cotangent _ := Unit
  zeroTangent _ := ()
  zeroCotangent _ := ()
  addTangent _ _ := ()
  addCotangent _ _ := ()
  scaleTangent _ _ := ()
  sharp := id
  flat := id
  exp x _ := x  -- Static values don't move

/-- TensorStruct types form Euclidean spaces (product manifold).
    tangent = cotangent = primal type, using TensorStruct.zipWith for operations. -/
instance tensorStructEuclidean [s : TensorStruct α] [Inhabited α] : EuclideanSpace α where
  zero := default
  add := TensorStruct.zipWith Add.add
  smul scalar m := TensorStruct.map (torch.Tensor.Preserving.mulScalar · scalar) m
  -- Inner product: sum of all element-wise products across tensors
  inner a b :=
    let product := TensorStruct.zipWith Mul.mul a b
    TensorStruct.fold (fun t acc => acc + torch.Tensor.item (torch.Tensor.sumAll t)) 0.0 product

/-- Gradient descent step using exponential map / retraction.
    For Euclidean spaces: x' = x - lr * grad
    For curved manifolds: x' = retract(x, -lr * sharp(grad)) -/
def gradientStep [DifferentiableManifold M]
    (x : M) (grad : Cotangent x) (lr : Float) : M :=
  let tangent := sharp grad
  let negTangent := scaleTangent (-lr) tangent
  retract x negTangent

end DifferentiableManifold

-- Backward compatibility alias
abbrev Differentiable := DifferentiableManifold

end Tyr.AD
