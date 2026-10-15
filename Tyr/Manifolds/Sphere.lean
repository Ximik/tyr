import Tyr.Manifolds.Basic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : Device}

/-- Real sphere S^{n-1} represented as unit-norm vectors in `R^n`. -/
structure Sphere (n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  coords : Tensor ⟨#[n], dtype, device⟩

namespace Sphere

private def dot {n : UInt64} (a b : Tensor ⟨#[n], d, dev⟩) : Float :=
  Tensor.item (Tensor.sumAll (Tensor.Preserving.mul a b))

/-- Project ambient vector to the unit sphere. -/
def project (n : UInt64) (v : Tensor ⟨#[n], d, dev⟩) : Sphere n d dev :=
  let norm := Tensor.l2Norm v
  let eps := 1e-12
  if norm <= eps then
    ⟨Tensor.Preserving.mulScalar (Tensor.onesLike v) (1.0 / Float.sqrt n.toFloat)⟩
  else
    ⟨Tensor.Preserving.mulScalar v (1.0 / norm)⟩

/-- Random point on the sphere. -/
def random (n : UInt64) : IO (Sphere n) := do
  let v ← Tensor.randn #[n]
  pure (project n v)

end Sphere

/-- Tangent vector at a sphere point. -/
structure SphereTangent (n : UInt64) (dtype : DType := .Float32)
    (device : Device := .CPU) where
  vec : Tensor ⟨#[n], dtype, device⟩

namespace SphereTangent

/-- Orthogonal projection to the tangent space at `x`. -/
def project (x : Sphere n d dev) (v : Tensor ⟨#[n], d, dev⟩) : SphereTangent n d dev :=
  let xDotV := Sphere.dot x.coords v
  let correction := Tensor.Preserving.mulScalar x.coords xDotV
  ⟨Tensor.Preserving.sub v correction⟩

/-- Zero tangent vector. -/
def zero (n : UInt64) : SphereTangent n d dev := ⟨Tensor.zerosFor ⟨#[n], d, dev⟩⟩

/-- Add tangent vectors. -/
def add (a b : SphereTangent n d dev) : SphereTangent n d dev :=
  ⟨Tensor.Preserving.add a.vec b.vec⟩

/-- Scale tangent vector. -/
def smul (s : Float) (v : SphereTangent n d dev) : SphereTangent n d dev :=
  ⟨Tensor.Preserving.mulScalar v.vec s⟩

end SphereTangent

instance sphereManifold (n : UInt64) : DifferentiableManifold (Sphere n d dev) where
  Tangent _ := SphereTangent n d dev
  Cotangent _ := SphereTangent n d dev
  zeroTangent x := ⟨Tensor.zerosLike x.coords⟩
  zeroCotangent x := ⟨Tensor.zerosLike x.coords⟩
  addTangent a b := SphereTangent.add a b
  addCotangent a b := SphereTangent.add a b
  scaleTangent s v := SphereTangent.smul s v
  sharp := id
  flat := id
  exp x v :=
    let nv := Tensor.l2Norm v.vec
    if nv <= 1e-12 then x else
      let dir := Tensor.Preserving.mulScalar v.vec (1.0 / nv)
      let c := Float.cos nv
      let s := Float.sin nv
      let y := Tensor.Preserving.add (Tensor.Preserving.mulScalar x.coords c) (Tensor.Preserving.mulScalar dir s)
      Sphere.project n y
  retract x v :=
    Sphere.project n (Tensor.Preserving.add x.coords v.vec)

end Tyr.AD
