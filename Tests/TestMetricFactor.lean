import LeanTest
import Tyr.Modular.MetricFactor
import Tyr.Manifolds.Embedded

namespace Tests.MetricFactor

open LeanTest
open torch
open Tyr.Modular
open Tyr.AD

private def vectorToColumn {n : UInt64} (v : Tensor { shape := #[n], dtype := .Float32, device := .CPU }) : Tensor { shape := #[n, 1], dtype := .Float32, device := .CPU } :=
  Tensor.reshape v #[n, 1] (by simp [TensorSpec.numelOfShape])

private def columnToVector {n : UInt64} (v : Tensor { shape := #[n, 1], dtype := .Float32, device := .CPU }) : Tensor { shape := #[n], dtype := .Float32, device := .CPU } :=
  Tensor.reshape v #[n] (by simp [TensorSpec.numelOfShape])

@[test]
def testMetricFactorPullbackMatchesExplicitMatmul : IO Unit := do
  let lMat ← Tensor.randn #[3, 4]
  let a ← Tensor.randn #[4, 2]
  let factor : MetricFactor 3 4 := { matrix := lMat }
  let pulled := MetricFactor.pullback factor a
  let expected := Tensor.mm lMat a
  LeanTest.assertTrue (Tensor.allclose pulled.matrix expected (rtol := 1e-4) (atol := 1e-5))
    "MetricFactor.pullback should match explicit matrix multiplication"

@[test]
def testWoodburySolveMatchesDenseInverse : IO Unit := do
  let kMat ← Tensor.randn #[3, 5]
  let rawDiag ← Tensor.randn #[5]
  let diag := Tensor.addScalar (Tensor.abs rawDiag) 1.0
  let g ← Tensor.randn #[5]
  let factor : MetricFactor 3 5 := { matrix := kMat }
  let mass : DiagonalMass 5 := { diag := diag }

  let woodbury := MetricFactor.solveWoodbury mass factor g
  let denseInv := Tensor.inv (MetricFactor.denseMetric mass factor)
  let dense := columnToVector (Tensor.mm denseInv (vectorToColumn g))

  LeanTest.assertTrue (Tensor.allclose woodbury dense (rtol := 1e-4) (atol := 1e-5))
    "Woodbury solve should match dense inverse solve on small problems"

@[test]
def testEmbeddedEuclideanRetractAmbientStepMatchesEuclideanUpdate : IO Unit := do
  let x ← Tensor.randn #[4] false
  let g ← Tensor.randn #[4] false
  let lr : Float := 0.1
  let x' := EmbeddedManifold.retractAmbientStep x g lr
  let expected := x - Tensor.mulScalar g lr
  LeanTest.assertTrue (Tensor.l2Norm (x' - expected) < 1e-6)
    "EmbeddedManifold.retractAmbientStep should reduce to Euclidean GD in flat space"

@[test]
def testEmbeddedStiefelRetractAmbientStepPreservesConstraint : IO Unit := do
  let x ← Stiefel.random 6 3
  let g ← Tensor.randn #[6, 3]
  let x' := EmbeddedManifold.retractAmbientStep x g 0.05
  let wtW := Tensor.mm (Tensor.transpose2d x'.matrix) x'.matrix
  let I := Tensor.eye 3
  LeanTest.assertTrue (Tensor.allclose wtW I (rtol := 1e-4) (atol := 1e-5))
    "Embedded ambient-step retraction should preserve the Stiefel constraint"

example (_x : Tensor { shape := #[4], dtype := .Float32, device := .CPU }) (_g : Tensor { shape := #[5], dtype := .Float32, device := .CPU }) : True := by
  fail_if_success
    have _ := EmbeddedManifold.retractAmbientStep _x _g 0.1
  trivial

end Tests.MetricFactor
