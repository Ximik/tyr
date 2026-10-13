/-
  Tests/TestDualOptimizerDispatch.lean

  Focused tests for DualOptimizer matrix backend dispatch and composable
  matrix backend operation bundles.
-/
import LeanTest
import Tyr.Optim.DualOptimizer

namespace Tests.DualOptimizerDispatch

open LeanTest
open torch
open torch.Optim.DualOptimizer

@[test]
def testMatrixStateInitializationDispatch : IO Unit := do
  let p ← Tensor.randn #[8, 4]

  let cfgNor : Config := { matrixOptimizer := .norMuon }
  let stNor := initMatrixParamState cfgNor p
  match stNor with
  | .norMuon _ => pure ()
  | _ => LeanTest.fail "Expected NorMuon state for norMuon backend"

  let cfgStiefel : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .stiefel
    preferGenericManifoldPath := false
  }
  LeanTest.assertTrue (usesSpecializedStiefelPath cfgStiefel)
    "Expected specialized Stiefel path to be enabled"
  let stStiefel := initMatrixParamState cfgStiefel p
  match stStiefel with
  | .manifoldMuon _ => pure ()
  | _ => LeanTest.fail "Expected specialized manifold-Muon state for Stiefel path"

  let cfgGrassmann : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .grassmann
  }
  let stGrass := initMatrixParamState cfgGrassmann p
  match stGrass with
  | .genericManifold _ => pure ()
  | _ => LeanTest.fail "Expected generic manifold state for non-Stiefel manifold path"

@[test]
def testMatrixBackendSpecPerLabelDispatch : IO Unit := do
  let spec : MatrixBackendSpec := {
    default := { matrixOptimizer := MatrixOptimizerKind.norMuon }
    attn := some {
      matrixOptimizer := MatrixOptimizerKind.manifoldMuon
      matrixManifold := MatrixManifoldFamily.stiefel
    }
    mlp := some {
      matrixOptimizer := MatrixOptimizerKind.manifoldMuon
      matrixManifold := MatrixManifoldFamily.grassmann
      preferGenericManifoldPath := true
    }
  }
  let attnCfg := spec.configForLabel torch.Optim.NorMuon.ParamLabel.attn
  let mlpCfg := spec.configForLabel torch.Optim.NorMuon.ParamLabel.mlp
  let smearCfg := spec.configForLabel torch.Optim.NorMuon.ParamLabel.smearGate

  LeanTest.assertTrue (usesManifoldMuon attnCfg)
    "Expected manifold backend for attention label"
  LeanTest.assertTrue (attnCfg.matrixManifold == MatrixManifoldFamily.stiefel)
    "Expected Stiefel manifold for attention label"
  LeanTest.assertTrue (usesManifoldMuon mlpCfg)
    "Expected manifold backend for MLP label"
  LeanTest.assertTrue (mlpCfg.preferGenericManifoldPath && mlpCfg.matrixManifold == MatrixManifoldFamily.grassmann)
    "Expected generic Grassmann path for MLP label"
  LeanTest.assertTrue (!(usesManifoldMuon smearCfg))
    "Expected default non-manifold backend for unspecified label"

@[test]
def testManifoldMuonConfigForwardsSolverKnobs : IO Unit := do
  let cfg : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .stiefel
    manifoldSolver := .fixedPoint
    manifoldMinSolveSteps := 3
    manifoldSolveResidualTol := 1e-3
    manifoldSolveDualDeltaTol := 2e-3
    manifoldFixedPointDamping := 0.25
  }
  let mcfg := getManifoldMuonConfigAtStep cfg 7
  LeanTest.assertTrue (mcfg.solver == .fixedPoint)
    "Expected solver kind to be forwarded to manifold-Muon config"
  LeanTest.assertEqual mcfg.minSolveSteps 3
  LeanTest.assertTrue (Float.abs (mcfg.solveResidualTol - 1e-3) < 1e-12)
    s!"Expected residual tolerance 1e-3, got {mcfg.solveResidualTol}"
  LeanTest.assertTrue (Float.abs (mcfg.solveDualDeltaTol - 2e-3) < 1e-12)
    s!"Expected dual-delta tolerance 2e-3, got {mcfg.solveDualDeltaTol}"
  LeanTest.assertTrue (Float.abs (mcfg.fixedPointDamping - 0.25) < 1e-12)
    s!"Expected fixed-point damping 0.25, got {mcfg.fixedPointDamping}"

@[test]
def testStepMatrixSingleStiefelConstraint : IO Unit := do
  let pRaw ← Tensor.randn #[8, 4]
  let p := Tensor.setRequiresGrad
    (Tensor.detach (torch.Optim.ManifoldMuon.retractStiefel pRaw)) true
  let g ← Tensor.randn #[8, 4]

  let cfg : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .stiefel
    preferGenericManifoldPath := false
    matrixLr := 0.02
  }
  let st0 := initMatrixParamState cfg p
  let (p1, st1) ← stepMatrixSingle p g st0 cfg 0 1.0 1.0

  let wtW := Tensor.mm (Tensor.transpose2d p1) p1
  let I := Tensor.eye 4
  LeanTest.assertTrue (Tensor.allclose wtW I (rtol := 1e-4) (atol := 1e-5))
    "Specialized Stiefel path should preserve W^T W ≈ I"
  match st1 with
  | .manifoldMuon _ => pure ()
  | _ => LeanTest.fail "Expected specialized manifold-Muon state after Stiefel step"

@[test]
def testStepMatrixSingleGrassmannConstraint : IO Unit := do
  let pRaw ← Tensor.randn #[9, 3]
  let p := Tensor.setRequiresGrad
    (Tensor.detach (Tyr.AD.Grassmann.project 9 3 pRaw).matrix) true
  let g ← Tensor.randn #[9, 3]

  let cfg : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .grassmann
    matrixLr := 0.02
  }
  let st0 := initMatrixParamState cfg p
  let (p1, st1) ← stepMatrixSingle p g st0 cfg 0 1.0 1.0

  let xtx := Tensor.mm (Tensor.transpose2d p1) p1
  let I := Tensor.eye 3
  LeanTest.assertTrue (Tensor.allclose xtx I (rtol := 1e-4) (atol := 1e-5))
    "Generic Grassmann path should preserve X^T X ≈ I"
  match st1 with
  | .genericManifold _ => pure ()
  | _ => LeanTest.fail "Expected generic manifold state after Grassmann step"

@[test]
def testStepMatrixSingleOrthogonalSquareConstraint : IO Unit := do
  let pRaw ← Tensor.randn #[6, 6]
  let p := Tensor.setRequiresGrad
    (Tensor.detach (Tyr.AD.Orthogonal.project 6 pRaw).matrix) true
  let g ← Tensor.randn #[6, 6]

  let cfg : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .orthogonal
    matrixLr := 0.02
  }
  let st0 := initMatrixParamState cfg p
  let (p1, st1) ← stepMatrixSingle p g st0 cfg 0 1.0 1.0

  let qtq := Tensor.mm (Tensor.transpose2d p1) p1
  let I := Tensor.eye 6
  LeanTest.assertTrue (Tensor.allclose qtq I (rtol := 1e-4) (atol := 1e-5))
    "Generic orthogonal path should preserve Q^T Q ≈ I for square matrices"
  match st1 with
  | .genericManifold _ => pure ()
  | _ => LeanTest.fail "Expected generic manifold state after orthogonal step"

@[test]
def testStepMatrixSingleOrthogonalNonSquareFails : IO Unit := do
  let pRaw ← Tensor.randn #[8, 4]
  let p := Tensor.setRequiresGrad
    (Tensor.detach (Tyr.AD.Stiefel.project 8 4 pRaw).matrix) true
  let g ← Tensor.randn #[8, 4]

  let cfg : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .orthogonal
    matrixLr := 0.02
  }
  let st0 := initMatrixParamState cfg p
  let threw ← try
      let _ ← stepMatrixSingle p g st0 cfg 0 1.0 1.0
      pure false
    catch _ =>
      pure true
  LeanTest.assertTrue threw
    "Expected non-square orthogonal manifold step to fail fast"

@[test]
def testMatrixBackendOpsComposable : IO Unit := do
  let cfg : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .grassmann
    matrixLr := 0.01
  }
  let ops := matrixBackendOps (m := 12) (n := 4) cfg 0 1.0 1.0

  let p0Raw ← Tensor.randn #[12, 4]
  let p1Raw ← Tensor.randn #[12, 4]
  let p0 := Tensor.setRequiresGrad
    (Tensor.detach (Tyr.AD.Grassmann.project 12 4 p0Raw).matrix) true
  let p1 := Tensor.setRequiresGrad
    (Tensor.detach (Tyr.AD.Grassmann.project 12 4 p1Raw).matrix) true
  let g0 ← Tensor.randn #[12, 4]
  let g1 ← Tensor.randn #[12, 4]

  let s0 := ops.initState p0
  let s1 := ops.initState p1
  let (ps', ss') ← ops.stepGroupLocal #[p0, p1] #[g0, g1] #[s0, s1]

  LeanTest.assertEqual ps'.size 2
  LeanTest.assertEqual ss'.size 2

@[test]
def testMatrixBackendOpsBenchmarkPositive : IO Unit := do
  let cfg : Config := {
    matrixOptimizer := .manifoldMuon
    matrixManifold := .grassmann
    matrixLr := 0.01
  }
  let mut p := (Tyr.AD.Grassmann.project 20 6 (← Tensor.randn #[20, 6])).matrix
  p := Tensor.setRequiresGrad (Tensor.detach p) true
  let mut st := initMatrixParamState cfg p

  let t0 ← IO.monoNanosNow
  for _ in [:3] do
    let g ← Tensor.randn #[20, 6]
    let (p', st') ← stepMatrixSingle p g st cfg 0 1.0 1.0
    p := p'
    st := st'
  let t1 ← IO.monoNanosNow

  let elapsedMs := (t1.toFloat - t0.toFloat) / 1000000.0
  LeanTest.assertTrue (elapsedMs > 0.0)
    s!"Expected positive benchmark elapsed time, got {elapsedMs}ms"

end Tests.DualOptimizerDispatch
