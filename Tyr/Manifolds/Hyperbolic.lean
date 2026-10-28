/-
  Tyr/Manifolds/Hyperbolic.lean

  Hyperbolic manifold H^n in the hyperboloid model.
-/
import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : Device} [DType.RealFloating d] [DType.ConstantSupported d dev]

/-! ## Hyperbolic Manifold H^n (hyperboloid model)

We model hyperbolic space as a two-sheeted hyperboloid in Minkowski space:

  H^n = { x in R^{n+1} : <x,x>_L = -1, x_{n+1} > 0 }

where <x,y>_L = sum_{i=1..n} x_i y_i - x_{n+1} y_{n+1}.
-/

/-- Hyperbolic space H^n represented as an (n+1)-vector on the hyperboloid. -/
structure Hyperbolic (n : Nat) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  /-- Ambient coordinates in R^{n+1} -/
  coords : Tensor ⟨#[n + 1], dtype, device⟩

namespace Hyperbolic

/-- Spatial mask [1, …, 1, 0] selecting the spatial coordinates. -/
def spatialMask (n : Nat) : Tensor ⟨#[n + 1], d, dev⟩ :=
  Tensor.cat (Tensor.onesFor ⟨#[n], d, dev⟩) (Tensor.zerosFor ⟨#[1], d, dev⟩) 0

/-- Minkowski signature vector [1, …, 1, -1] of the ambient space. -/
def signatureVec (n : Nat) : Tensor ⟨#[n + 1], d, dev⟩ :=
  Tensor.cat (Tensor.onesFor ⟨#[n], d, dev⟩)
    (Tensor.mulScalar (Tensor.onesFor ⟨#[1], d, dev⟩) (-1.0)) 0

/-- Minkowski inner product (signature (n,1)) as Float. -/
def minkowskiInner (x y : Tensor ⟨#[n + 1], d, dev⟩) : Float :=
  Tensor.item (Tensor.sumAll (Tensor.mul (Tensor.mul x y) (signatureVec (d := d) n)))

/-- Project an ambient vector to the hyperboloid (future sheet).
    Keeps the spatial part and recomputes the time coordinate. -/
def project (n : Nat) (v : Tensor ⟨#[n + 1], d, dev⟩) : Hyperbolic n d dev :=
  let spatial := Tensor.mul v (spatialMask (d := d) n)
  let spatialSq := Tensor.item (Tensor.sumAll (Tensor.mul spatial spatial))
  let t := Float.sqrt (spatialSq + 1.0)
  let tVec := Tensor.mulScalar (Tensor.onesFor ⟨#[1], d, dev⟩) t
  let timeVec := Tensor.cat (Tensor.zerosFor ⟨#[n], d, dev⟩) tVec 0
  ⟨Tensor.add spatial timeVec⟩

/-- Normalize a time-like vector to the hyperboloid using Minkowski norm. -/
def normalize (n : Nat) (v : Tensor ⟨#[n + 1], d, dev⟩) : Hyperbolic n d dev :=
  let inner := minkowskiInner v v
  let scale := 1.0 / Float.sqrt (-inner)
  ⟨Tensor.mulScalar v scale⟩

/-- Random point on H^n via random spatial coordinates and derived time coord. -/
def random (n : Nat) : IO (Hyperbolic n) := do
  let spatial ← Tensor.randn #[n]
  let spatialSq := Tensor.item (Tensor.sumAll (Mul.mul spatial spatial))
  let t := Float.sqrt (spatialSq + 1.0)
  let tVec := Tensor.mulScalar (Tensor.onesFor ⟨#[1], .Float32, .CPU⟩) t
  return ⟨Tensor.cat spatial tVec 0⟩

/-- Origin point (0, ..., 0, 1). -/
def origin (n : Nat) : Hyperbolic n d dev :=
  ⟨Tensor.cat (Tensor.zerosFor ⟨#[n], d, dev⟩) (Tensor.onesFor ⟨#[1], d, dev⟩) 0⟩

end Hyperbolic

/-- Tangent vector on H^n, represented in ambient coordinates. -/
structure HyperbolicTangent (n : Nat) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  /-- Tangent vector in R^{n+1} -/
  vec : Tensor ⟨#[n + 1], dtype, device⟩

namespace HyperbolicTangent

/-- Zero tangent vector. -/
def zero (n : Nat) : HyperbolicTangent n d dev :=
  ⟨Tensor.zerosFor ⟨#[n + 1], d, dev⟩⟩

/-- Add two tangent vectors. -/
def add (v w : HyperbolicTangent n d dev) : HyperbolicTangent n d dev :=
  ⟨Tensor.add v.vec w.vec⟩

/-- Scale a tangent vector. -/
def smul (s : Float) (v : HyperbolicTangent n d dev) : HyperbolicTangent n d dev :=
  ⟨Tensor.mulScalar v.vec s⟩

/-- Project an ambient vector to the tangent space at X (Minkowski-orthogonal). -/
def project (X : Hyperbolic n d dev) (V : Tensor ⟨#[n + 1], d, dev⟩) : HyperbolicTangent n d dev :=
  let alpha := -Hyperbolic.minkowskiInner X.coords V / Hyperbolic.minkowskiInner X.coords X.coords
  ⟨Tensor.add V (Tensor.mulScalar X.coords alpha)⟩

/-- Apply the Minkowski metric (signature (n,1)) to a vector: flips last coord. -/
def applyMetric (v : Tensor ⟨#[n + 1], d, dev⟩) : Tensor ⟨#[n + 1], d, dev⟩ :=
  Tensor.mul v (Hyperbolic.signatureVec (d := d) n)

end HyperbolicTangent

/-- Hyperbolic manifold as a DifferentiableManifold.
    Uses a simple projection retraction onto the hyperboloid. -/
instance hyperbolicManifold (n : Nat) : DifferentiableManifold (Hyperbolic n d dev) where
  Tangent _ := HyperbolicTangent n d dev
  Cotangent _ := HyperbolicTangent n d dev
  zeroTangent x := ⟨Tensor.zerosLike x.coords⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.coords⟩
  addTangent v w := HyperbolicTangent.add v w
  addCotangent v w := HyperbolicTangent.add v w
  scaleTangent s v := HyperbolicTangent.smul s v
  sharp v := ⟨HyperbolicTangent.applyMetric v.vec⟩
  flat v := ⟨HyperbolicTangent.applyMetric v.vec⟩
  exp X V :=
    Hyperbolic.normalize n (Tensor.add X.coords V.vec)
  retract X V :=
    Hyperbolic.normalize n (Tensor.add X.coords V.vec)

end Tyr.AD
