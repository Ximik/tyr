/-
  Tyr/Manifolds/Optimizer.lean

  Generic manifold-optimizer primitives with explicit dual-map support.

  This file extends the manifold stack with a light-weight interface for
  non-Riemannian/Finsler-style optimization where the steepest direction need
  not be the metric sharp-map direction.
-/
import Tyr.Manifolds.Basic
import Tyr.Manifolds.Stiefel
import Tyr.Manifolds.Orthogonal
import Tyr.Manifolds.Grassmann
import Tyr.Manifolds.Hyperbolic

namespace Tyr.AD

open torch
open DifferentiableManifold

variable {d : DType} {dev : Device} [DType.RealFloating d]

private def normalizeMatrixFrobenius {m n : UInt64}
    (A : Tensor ⟨#[m, n], d, dev⟩) : Tensor ⟨#[m, n], d, dev⟩ :=
  let fnorm := Tensor.frobeniusNorm A
  if fnorm == 0.0 then
    A
  else
    Tensor.mulScalar A (1.0 / fnorm)

private def normalizeVectorL2 {n : UInt64}
    (v : Tensor ⟨#[n], d, dev⟩) : Tensor ⟨#[n], d, dev⟩ :=
  let vnorm := Tensor.l2Norm v
  if vnorm == 0.0 then
    v
  else
    Tensor.mulScalar v (1.0 / vnorm)

private def minkowskiAbsNorm [DType.ConstantSupported d dev] {n : UInt64} (v : Tensor ⟨#[n + 1], d, dev⟩) : Float :=
  Float.sqrt (Float.abs (Hyperbolic.minkowskiInner v v))

/--
A geometry interface for constrained steepest descent with an explicit dual map.

`dualMap` can differ from the Riemannian `sharp`; this is the hook used by
non-Riemannian/norm-constrained optimizers.
-/
class DualMapGeometry (M : Type) [DifferentiableManifold M] where
  /-- Tangent-space norm used to measure step size. -/
  tangentNorm : {x : M} → Tangent x → Float
  /-- Cotangent-space dual norm used to measure gradient size. -/
  cotangentNorm : {x : M} → Cotangent x → Float
  /-- Steepest-descent dual map from cotangent to tangent space. -/
  dualMap : {x : M} → Cotangent x → Tangent x

namespace DualMapGeometry

/-- Steepest direction induced by the geometry's dual map. -/
def steepestDirection [DifferentiableManifold M] [DualMapGeometry M]
    {x : M} (g : Cotangent x) : Tangent x :=
  DualMapGeometry.dualMap g

/-- Apply a dual-map step then retract back to the manifold. -/
def dualMapStep [DifferentiableManifold M] [DualMapGeometry M]
    (x : M) (g : Cotangent x) (lr : Float) : M :=
  let dir := steepestDirection (x := x) g
  let step := scaleTangent (-lr) dir
  retract x step

/-- Optional diagnostics for geometry-aware updates. -/
structure StepDiagnostics where
  tangentNorm : Float
  cotangentNorm : Float
  deriving Repr, Inhabited

/-- Apply a dual-map step and return basic norm diagnostics. -/
def dualMapStepWithDiagnostics [DifferentiableManifold M] [DualMapGeometry M]
    (x : M) (g : Cotangent x) (lr : Float) : M × StepDiagnostics :=
  let dir := steepestDirection (x := x) g
  let step := scaleTangent (-lr) dir
  let x' := retract x step
  let stats : StepDiagnostics := {
    tangentNorm := DualMapGeometry.tangentNorm (x := x) dir
    cotangentNorm := DualMapGeometry.cotangentNorm (x := x) g
  }
  (x', stats)

end DualMapGeometry

/--
Float dual-map geometry using sign descent.

This intentionally differs from Euclidean gradient descent and acts as a tiny
non-Riemannian example: the step direction saturates to {-1, 0, 1}.
-/
instance floatSignDualMap : DualMapGeometry Float where
  tangentNorm := by
    intro x v
    let _ := x
    simpa using (Float.abs v)
  cotangentNorm := by
    intro x g
    let _ := x
    simpa using (Float.abs g)
  dualMap := by
    intro x g
    let _ := x
    let g' : Float := g
    simpa using (if g' > 0.0 then (1.0 : Float) else if g' < 0.0 then (-1.0 : Float) else (0.0 : Float))

/--
Stiefel geometry adapter with spectral/nuclear norms and projected dual map.

The dual map is implemented as tangent projection followed by unit Frobenius
normalization as a practical approximation.
-/
instance stiefelDualMapGeometry (m n : UInt64) [DType.RealFloating d] :
    DualMapGeometry (Stiefel m n d dev) where
  tangentNorm := fun {x} (v : StiefelTangent m n d dev) =>
    let _ := x
    Tensor.spectralNorm v.vec
  cotangentNorm := fun {x} (g : StiefelTangent m n d dev) =>
    let _ := x
    Tensor.nuclearNorm g.vec
  dualMap := fun {x} (g : StiefelTangent m n d dev) =>
    let projected := StiefelTangent.project x g.vec
    { vec := normalizeMatrixFrobenius projected.vec }

/--
Orthogonal-group adapter for dual-map optimization.

Uses ambient spectral/nuclear norms and normalizes the skew parameterization.
-/
instance orthogonalDualMapGeometry [DType.ConstantSupported d dev] (n : UInt64) [DType.RealFloating d] :
    DualMapGeometry (Orthogonal n d dev) where
  tangentNorm := fun {x} (v : OrthogonalTangent n d dev) =>
    Tensor.spectralNorm (OrthogonalTangent.toAmbient x v)
  cotangentNorm := fun {x} (g : OrthogonalTangent n d dev) =>
    Tensor.nuclearNorm (OrthogonalTangent.toAmbient x g)
  dualMap := fun {x} (g : OrthogonalTangent n d dev) =>
    let ambient := OrthogonalTangent.toAmbient x g
    let projected := OrthogonalTangent.fromAmbient x ambient
    { skew := { matrix := normalizeMatrixFrobenius projected.skew.matrix } }

/--
Grassmann adapter for dual-map optimization.

Projects gradients to the horizontal tangent space and normalizes step size.
-/
instance grassmannDualMapGeometry (n p : UInt64) [DType.RealFloating d] :
    DualMapGeometry (Grassmann n p d dev) where
  tangentNorm := fun {x} (v : GrassmannTangent n p d dev) =>
    let _ := x
    Tensor.spectralNorm v.vec
  cotangentNorm := fun {x} (g : GrassmannTangent n p d dev) =>
    let _ := x
    Tensor.nuclearNorm g.vec
  dualMap := fun {x} (g : GrassmannTangent n p d dev) =>
    let projected := GrassmannTangent.project x g.vec
    { vec := normalizeMatrixFrobenius projected.vec }

/--
Hyperbolic adapter for dual-map optimization in the hyperboloid model.

Applies the Minkowski metric to map cotangent to ambient tangent direction,
projects to the tangent space at the current point, then applies L2
normalization for a bounded update direction.
-/
instance hyperbolicDualMapGeometry [DType.ConstantSupported d dev] (n : UInt64) : DualMapGeometry (Hyperbolic n d dev) where
  tangentNorm := fun {x} (v : HyperbolicTangent n d dev) =>
    let _ := x
    minkowskiAbsNorm v.vec
  cotangentNorm := fun {x} (g : HyperbolicTangent n d dev) =>
    let _ := x
    minkowskiAbsNorm g.vec
  dualMap := fun {x} (g : HyperbolicTangent n d dev) =>
    let metric := HyperbolicTangent.applyMetric g.vec
    let projected := HyperbolicTangent.project x metric
    { vec := normalizeVectorL2 projected.vec }

end Tyr.AD
