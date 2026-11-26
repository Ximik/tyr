import Tyr.Lib.Optim.Manifolds.Basic
import Tyr.Lib.Optim.Manifolds.Embedded
import Tyr.Lib.Optim.Manifolds.Optimizer
import Tyr.Lib.Optim.Manifolds.Sphere
import Tyr.Lib.Optim.Manifolds.Stiefel
import Tyr.Lib.Optim.Manifolds.Orthogonal
import Tyr.Lib.Optim.Manifolds.Grassmann
import Tyr.Lib.Optim.Manifolds.Oblique
import Tyr.Lib.Optim.Manifolds.Positive
import Tyr.Lib.Optim.Manifolds.PoincareBall
import Tyr.Lib.Optim.Manifolds.PositiveDefinite
import Tyr.Lib.Optim.Manifolds.PSD
import Tyr.Lib.Optim.Manifolds.FixedRank
import Tyr.Lib.Optim.Manifolds.Hyperbolic

/-!
# Tyr.Lib.Optim.Manifolds

`Tyr.Lib.Optim.Manifolds` is the umbrella import for manifold geometry support in Tyr.
It re-exports the core manifold interfaces plus every concrete manifold family
currently shipped under `Tyr/Lib/Optim/Manifolds`, so geometry-aware optimization and
constrained parameterization code can depend on a single import.

## Major Components

- [`Basic`](##Tyr.Lib.Optim.Manifolds.Basic): foundational manifold/typeclass definitions.
- [`Embedded`](##Tyr.Lib.Optim.Manifolds.Embedded): embedded-manifold interface for ambient/tangent projections.
- [`Optimizer`](##Tyr.Lib.Optim.Manifolds.Optimizer): geometry-aware optimizer interfaces, including dual-map/Finsler hooks.
- [`Sphere`](##Tyr.Lib.Optim.Manifolds.Sphere): unit-norm vectors in Euclidean space.
- [`Stiefel`](##Tyr.Lib.Optim.Manifolds.Stiefel): orthonormal-column matrix manifold geometry.
- [`Orthogonal`](##Tyr.Lib.Optim.Manifolds.Orthogonal): orthogonal-group manifold utilities.
- [`Grassmann`](##Tyr.Lib.Optim.Manifolds.Grassmann): subspace-valued manifold representation.
- [`Oblique`](##Tyr.Lib.Optim.Manifolds.Oblique): matrices with unit-norm columns.
- [`Positive`](##Tyr.Lib.Optim.Manifolds.Positive): elementwise-positive matrices.
- [`PositiveDefinite`](##Tyr.Lib.Optim.Manifolds.PositiveDefinite): symmetric positive definite matrix geometry.
- [`PSD`](##Tyr.Lib.Optim.Manifolds.PSD): fixed-rank PSD factor and elliptope manifolds.
- [`FixedRank`](##Tyr.Lib.Optim.Manifolds.FixedRank): embedded fixed-rank matrix manifold geometry.
- [`PoincareBall`](##Tyr.Lib.Optim.Manifolds.PoincareBall): hyperbolic geometry in the ball model.
- [`Hyperbolic`](##Tyr.Lib.Optim.Manifolds.Hyperbolic): hyperboloid-model hyperbolic geometry.

## Scope

Use this module when you want full manifold support through one import.
Geometry-specialized code can import individual manifold modules directly.
-/
