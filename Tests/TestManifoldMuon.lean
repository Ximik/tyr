/-
  Tests/TestManifoldMuon.lean

  Tests for the experimental manifold optimizer stack:
  - Dual-map geometry behavior
  - Stiefel manifold-Muon step invariants
  - Small benchmark smoke check
-/
import LeanTest
import Tyr.Manifolds
import Tyr.Optim.ManifoldMuon

namespace Tests.ManifoldMuon

open LeanTest
open Tyr.AD
open Tyr.AD.DifferentiableManifold
open torch
open torch.Optim.ManifoldMuon

private def benchmarkOrthogonalDualMapStep (numSteps : Nat := 5) : IO Float := do
  let mut q ← Orthogonal.random 16
  let t0 ← IO.monoNanosNow
  for _ in [:numSteps] do
    let gRaw ← Tensor.randn #[16, 16]
    let g := OrthogonalTangent.fromAmbient q gRaw
    q := DualMapGeometry.dualMapStep q g 0.02
  let t1 ← IO.monoNanosNow
  let elapsedMs := (t1.toFloat - t0.toFloat) / 1000000.0
  if numSteps == 0 then
    return elapsedMs
  else
    return elapsedMs / numSteps.toFloat

private def benchmarkGrassmannDualMapStep (numSteps : Nat := 5) : IO Float := do
  let mut x ← Grassmann.random 24 8
  let t0 ← IO.monoNanosNow
  for _ in [:numSteps] do
    let gRaw ← Tensor.randn #[24, 8]
    let g := GrassmannTangent.project x gRaw
    x := DualMapGeometry.dualMapStep x g 0.02
  let t1 ← IO.monoNanosNow
  let elapsedMs := (t1.toFloat - t0.toFloat) / 1000000.0
  if numSteps == 0 then
    return elapsedMs
  else
    return elapsedMs / numSteps.toFloat

private def benchmarkHyperbolicDualMapStep (numSteps : Nat := 5) : IO Float := do
  let mut x ← Hyperbolic.random 8
  let t0 ← IO.monoNanosNow
  for _ in [:numSteps] do
    let gRaw ← Tensor.randn #[9]
    let g := HyperbolicTangent.project x gRaw
    x := DualMapGeometry.dualMapStep x g 0.02
  let t1 ← IO.monoNanosNow
  let elapsedMs := (t1.toFloat - t0.toFloat) / 1000000.0
  if numSteps == 0 then
    return elapsedMs
  else
    return elapsedMs / numSteps.toFloat

@[test]
def testFloatDualMapDiffersFromRiemannian : IO Unit := do
  let x : Float := 0.0
  let g : Float := 3.0
  let lr : Float := 0.1

  let xRiemann := gradientStep x g lr
  let xDual := DualMapGeometry.dualMapStep x g lr

  -- For Float sign-dual-map: xDual should move by lr, not lr*|g|.
  LeanTest.assertTrue (Float.abs (xRiemann - (-0.3)) < 1e-6)
    s!"Expected Euclidean step -0.3, got {xRiemann}"
  LeanTest.assertTrue (Float.abs (xDual - (-0.1)) < 1e-6)
    s!"Expected sign-dual step -0.1, got {xDual}"

@[test]
def testTangentProjectionConstraint : IO Unit := do
  let W := (Stiefel.identity 4).matrix
  let G ← Tensor.randn #[4, 4]
  let Z := tangentProject W G

  let lhs := Tensor.mm (Tensor.transpose2d W) Z + Tensor.mm (Tensor.transpose2d Z) W
  let zero := Tensor.zerosLike lhs

  LeanTest.assertTrue (Tensor.allclose lhs zero (rtol := 1e-4) (atol := 1e-5))
    "Projected update should satisfy Stiefel tangent constraint"

@[test]
def testStiefelConstraintAfterStep : IO Unit := do
  let W0raw ← Tensor.randn #[8, 4]
  let W0 ← Tensor.setRequiresGrad (Tensor.detach (retractStiefel W0raw)) true
  let G ← Tensor.randn #[8, 4]

  let st := initParamState W0
  let cfg : Config := {
    lr := 0.02
    momentum := 0.95
    numIters := 3
    dualAscentSteps := 1
    dualAscentLr := 0.1
    distributed := false
  }
  let (W1, _st1) ← stepSingle W0 G st cfg

  let wtW := Tensor.mm (Tensor.transpose2d W1) W1
  let I := Tensor.eye 4

  LeanTest.assertTrue (Tensor.allclose wtW I (rtol := 1e-4) (atol := 1e-5))
    "ManifoldMuon step should retract back to Stiefel (W^T W ≈ I)"

@[test]
def testDualAscentDiagnosticsEarlyStop : IO Unit := do
  let W0raw ← Tensor.randn #[8, 4]
  let W0 ← Tensor.setRequiresGrad (Tensor.detach (retractStiefel W0raw)) true
  let G ← Tensor.randn #[8, 4]
  let st := initParamState W0
  let cfg : Config := {
    lr := 0.02
    momentum := 0.95
    numIters := 3
    dualAscentSteps := 6
    dualAscentLr := 0.1
    solver := .dualAscent
    minSolveSteps := 1
    solveResidualTol := 1e9
    solveDualDeltaTol := 1e9
    distributed := false
  }
  let (_W1, _st1, diag) ← stepSingleWithDiagnostics W0 G st cfg
  LeanTest.assertTrue diag.converged "Expected dual-ascent solve to converge with loose tolerances"
  LeanTest.assertEqual diag.iterations 1
  LeanTest.assertTrue (Float.isFinite diag.dualObjective)
    s!"Expected finite dual objective, got {diag.dualObjective}"

@[test]
def testFixedPointDiagnosticsPath : IO Unit := do
  let W0raw ← Tensor.randn #[8, 4]
  let W0 ← Tensor.setRequiresGrad (Tensor.detach (retractStiefel W0raw)) true
  let G ← Tensor.randn #[8, 4]
  let st := initParamState W0
  let cfg : Config := {
    lr := 0.02
    momentum := 0.95
    numIters := 3
    dualAscentSteps := 4
    dualAscentLr := 0.1
    solver := .fixedPoint
    minSolveSteps := 1
    solveResidualTol := 1e9
    solveDualDeltaTol := 1e9
    fixedPointDamping := 0.5
    distributed := false
  }
  let (_W1, _st1, diag) ← stepSingleWithDiagnostics W0 G st cfg
  LeanTest.assertTrue (diag.solver == .fixedPoint)
    "Expected fixed-point solver diagnostics tag"
  LeanTest.assertTrue diag.converged
    "Expected fixed-point solve to converge with loose tolerances"
  LeanTest.assertEqual diag.iterations 1

@[test]
def testStrictToleranceRunsFullIterations : IO Unit := do
  let W0raw ← Tensor.randn #[8, 4]
  let W0 ← Tensor.setRequiresGrad (Tensor.detach (retractStiefel W0raw)) true
  let G ← Tensor.randn #[8, 4]
  let st := initParamState W0
  let cfg : Config := {
    lr := 0.02
    momentum := 0.95
    numIters := 3
    dualAscentSteps := 3
    dualAscentLr := 0.1
    solver := .dualAscent
    minSolveSteps := 1
    solveResidualTol := 0.0
    solveDualDeltaTol := 0.0
    distributed := false
  }
  let (_W1, _st1, diag) ← stepSingleWithDiagnostics W0 G st cfg
  LeanTest.assertTrue (!diag.converged)
    "Expected solver to remain unconverged with impossible zero tolerances"
  LeanTest.assertEqual diag.iterations cfg.dualAscentSteps

@[test]
def testManifoldMuonBenchmarkPositive : IO Unit := do
  let avgMs ← benchmarkLocalStep (m := 32) (n := 16) 3
  LeanTest.assertTrue (avgMs > 0.0) s!"Expected positive average step time, got {avgMs}"

@[test]
def testOrthogonalDualMapStepPreservesConstraint : IO Unit := do
  let q0 ← Orthogonal.random 6
  let gRaw ← Tensor.randn #[6, 6]
  let g := OrthogonalTangent.fromAmbient q0 gRaw
  let q1 := DualMapGeometry.dualMapStep q0 g 0.05

  let qtq := Tensor.mm (Tensor.transpose2d q1.matrix) q1.matrix
  let I := Tensor.eye 6
  LeanTest.assertTrue (Tensor.allclose qtq I (rtol := 1e-4) (atol := 1e-5))
    "Orthogonal dual-map step should preserve Q^T Q ≈ I via retraction"

@[test]
def testGrassmannDualMapStepPreservesConstraint : IO Unit := do
  let x0 ← Grassmann.random 7 3
  let gRaw ← Tensor.randn #[7, 3]
  let g := GrassmannTangent.project x0 gRaw
  let x1 := DualMapGeometry.dualMapStep x0 g 0.05

  let xtx := Tensor.mm (Tensor.transpose2d x1.matrix) x1.matrix
  let I := Tensor.eye 3
  LeanTest.assertTrue (Tensor.allclose xtx I (rtol := 1e-4) (atol := 1e-5))
    "Grassmann dual-map step should preserve X^T X ≈ I via retraction"

@[test]
def testHyperbolicDualMapStepPreservesConstraint : IO Unit := do
  let x0 ← Hyperbolic.random 4
  let gRaw ← Tensor.randn #[5]
  let g := HyperbolicTangent.project x0 gRaw
  let x1 := DualMapGeometry.dualMapStep x0 g 0.05

  let inner := Hyperbolic.minkowskiInner x1.coords x1.coords
  LeanTest.assertTrue (Float.abs (inner + 1.0) < 1e-4)
    s!"Hyperbolic dual-map step should preserve <x,x>_L = -1, got {inner}"

@[test]
def testDualMapAdapterBenchmarksPositive : IO Unit := do
  let orthoMs ← benchmarkOrthogonalDualMapStep 3
  let grassMs ← benchmarkGrassmannDualMapStep 3
  let hyperMs ← benchmarkHyperbolicDualMapStep 3
  LeanTest.assertTrue (orthoMs > 0.0)
    s!"Expected positive orthogonal benchmark latency, got {orthoMs}"
  LeanTest.assertTrue (grassMs > 0.0)
    s!"Expected positive grassmann benchmark latency, got {grassMs}"
  LeanTest.assertTrue (hyperMs > 0.0)
    s!"Expected positive hyperbolic benchmark latency, got {hyperMs}"

end Tests.ManifoldMuon
