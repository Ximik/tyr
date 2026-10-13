import LeanTest
import Tyr.Module.Linear
import Tyr.Modular.RiemannianModule
import Tyr.Optim.RiemannianSGD

namespace Tests.RiemannianSGD

open LeanTest
open torch
open Tyr.Modular

@[test]
def testLinearRiemannianMSEStepProducesFiniteDiagnostics : IO Unit := do
  let layer ← torch.Linear.init 4 3 true
  let x ← Tensor.randn #[4]
  let target ← Tensor.randn #[3]
  let result := torch.Optim.RiemannianSGD.stepMSE layer x target 0.02
  LeanTest.assertEqual result.diagnostics.size 1
  let stats := result.diagnostics[0]!
  LeanTest.assertTrue (Float.isFinite result.loss && result.loss >= 0.0)
    s!"Expected finite non-negative loss, got {result.loss}"
  LeanTest.assertTrue (Float.isFinite stats.gradientNorm && Float.isFinite stats.updateNorm)
    "Expected finite gradient/update norms in leaf diagnostics"

@[test]
def testManifoldLinearRiemannianStepPreservesStiefelConstraint : IO Unit := do
  let layer ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 4 6 false
  let x ← Tensor.randn #[4]
  let target ← Tensor.randn #[6]
  let result := torch.Optim.RiemannianSGD.stepMSE layer x target 0.02
  let w := result.params.weight.matrix
  let wtW := Tensor.mm (Tensor.transpose2d w) w
  let I := Tensor.eye 4
  LeanTest.assertTrue (Tensor.allclose wtW I (rtol := 1e-4) (atol := 1e-5))
    "RiemannianSGD manifold step should keep StiefelLinear weights on-manifold"

@[test]
def testSequentialFactorRecursionMatchesExplicitGlobalJacobian : IO Unit := do
  let l1 ← torch.Linear.init 4 3 false
  let l2 ← torch.Linear.init 3 2 false
  let net : Sequential (torch.Linear 4 3) (torch.Linear 3 2) := Sequential.mk' l1 l2
  let x ← Tensor.randn #[4]
  let (_y, (tape1, tape2)) :
      AmbientVector 2 × SequentialTape (torch.Linear 4 3) 4 3 (torch.Linear 3 2) 2 :=
    sequentialForwardWithTape
      (M₁ := torch.Linear 4 3) (M₂ := torch.Linear 3 2)
      (inDim := 4) (midDim := 3) (outDim := 2)
      net x
  let lin1 := RiemannianModule.localLinearization net.first tape1
  let lin2 := RiemannianModule.localLinearization net.second tape2

  let lOut := MetricFactor.identity 2
  let k1Rec := LocalLinearization.pullbackParamFactor lin1 (LocalLinearization.pullbackInputFactor lin2 lOut)
  let a2 := LocalLinearization.materializeA lin2
  let b1 := LocalLinearization.materializeB lin1
  let globalJ1 := Tensor.mm a2 b1
  let k1Exp := MetricFactor.pullback lOut globalJ1

  LeanTest.assertTrue (Tensor.allclose k1Rec.matrix k1Exp.matrix (rtol := 1e-4) (atol := 1e-5))
    "Recursive factor propagation should match the explicit global Jacobian for layer 1"

@[test]
def testSequentialRiemannianMSEStepReturnsLayerDiagnostics : IO Unit := do
  let l1 ← torch.Linear.init 4 3 false
  let l2 ← torch.Linear.init 3 2 false
  let net : Sequential (torch.Linear 4 3) (torch.Linear 3 2) := Sequential.mk' l1 l2
  let x ← Tensor.randn #[4]
  let target ← Tensor.randn #[2]
  let result :=
    torch.Optim.RiemannianSGD.stepSequentialMSE
      (M₁ := torch.Linear 4 3) (M₂ := torch.Linear 3 2)
      (inDim := 4) (midDim := 3) (outDim := 2)
      net x target 0.02
  LeanTest.assertEqual result.diagnostics.size 2
  LeanTest.assertTrue (Float.isFinite result.loss && result.loss >= 0.0)
    s!"Expected finite non-negative sequential loss, got {result.loss}"

end Tests.RiemannianSGD
