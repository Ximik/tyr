import LeanTest
import Tyr.Modular
import Tyr.Manifolds

namespace Tests.ModularManifold

open LeanTest
open torch
open Tyr.Modular

@[test]
def testStiefelLinearConstraintAfterUpdate : IO Unit := do
  let layer ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 8 16 true
  let gW ← Tensor.randn #[16, 8] false
  let gB ← Tensor.randn #[16] false
  let layer' ← ManifoldLinear.applyDualMapUpdate layer gW (some gB) 0.02

  let w := layer'.weight.matrix
  let wtW := Tensor.mm (Tensor.transpose2d w) w
  let I := Tensor.eye 8
  LeanTest.assertTrue (Tensor.allclose wtW I (rtol := 1e-4) (atol := 1e-5))
    "StiefelLinear weight should remain on Stiefel after dual-map update"

@[test]
def testStiefelLinearWeightIsTrainableLeaf : IO Unit := do
  let layer ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 8 16 true
  let w0 := layer.weight.matrix
  LeanTest.assertTrue (Tensor.requiresGrad w0)
    "Expected manifold-linear weight to require gradients after init"
  LeanTest.assertTrue (Tensor.isLeaf w0)
    "Expected manifold-linear weight to be a leaf tensor after init"

  let gW ← Tensor.randn #[16, 8] false
  let gB ← Tensor.randn #[16] false
  let layer' ← ManifoldLinear.applyDualMapUpdate layer gW (some gB) 0.02
  let w1 := layer'.weight.matrix
  LeanTest.assertTrue (Tensor.requiresGrad w1)
    "Expected manifold-linear weight to require gradients after update"
  LeanTest.assertTrue (Tensor.isLeaf w1)
    "Expected manifold-linear weight to remain a leaf tensor after update"

@[test]
def testGrassmannLinearConstraintAfterUpdate : IO Unit := do
  let layer ← ManifoldLinear.init (M := Tyr.AD.Grassmann) 6 12 true
  let gW ← Tensor.randn #[12, 6] false
  let gB ← Tensor.randn #[12] false
  let layer' ← ManifoldLinear.applyDualMapUpdate layer gW (some gB) 0.02

  let w := layer'.weight.matrix
  let wtW := Tensor.mm (Tensor.transpose2d w) w
  let I := Tensor.eye 6
  LeanTest.assertTrue (Tensor.allclose wtW I (rtol := 1e-4) (atol := 1e-5))
    "GrassmannLinear representative should preserve X^T X ≈ I"

@[test]
def testOrthogonalLinearConstraintAfterUpdate : IO Unit := do
  let layer : OrthogonalLinear 8 ← ManifoldLinear.init (M := OrthogonalMatrix) 8 8 true
  let gW ← Tensor.randn #[8, 8] false
  let gB ← Tensor.randn #[8] false
  let layer' ← ManifoldLinear.applyDualMapUpdate layer gW (some gB) 0.02

  let w := layer'.weight.matrix
  let wtW := Tensor.mm (Tensor.transpose2d w) w
  let wwT := Tensor.mm w (Tensor.transpose2d w)
  let I := Tensor.eye 8
  LeanTest.assertTrue (Tensor.allclose wtW I (rtol := 1e-4) (atol := 1e-5))
    "OrthogonalLinear weight should preserve W^T W ≈ I"
  LeanTest.assertTrue (Tensor.allclose wwT I (rtol := 1e-4) (atol := 1e-5))
    "OrthogonalLinear weight should preserve W W^T ≈ I"

@[test]
def testModularBudgetBridgeProducesFiniteMultiplier : IO Unit := do
  let l1 ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 8 16 false
  let l2 ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 16 16 false
  let modules : Array (StiefelLinear 8 16 × StiefelLinear 16 16) := #[(l1, l2)]

  let merged : Array (Sequential (StiefelLinear 8 16) (StiefelLinear 16 16)) :=
    modules.map (fun (a, b) => Sequential.mk' a b)
  let scales := sequentialDownstreamScales {} merged
  LeanTest.assertTrue (!scales.isEmpty) "Expected non-empty modular scale array"
  for s in scales do
    LeanTest.assertTrue (Float.isFinite s && s > 0.0)
      s!"Expected positive finite modular scale, got {s}"

  let baseCfg : torch.Optim.DualOptimizer.Config := {}
  let cfg' := applyMatrixBudgetFromModules baseCfg {} merged
  LeanTest.assertTrue cfg'.useModularBudget
    "applyMatrixBudgetFromModules should enable modular budgeting"
  LeanTest.assertTrue (Float.isFinite cfg'.budget.matrix && cfg'.budget.matrix > 0.0)
    s!"Expected positive finite matrix budget multiplier, got {cfg'.budget.matrix}"

@[test]
def testManifoldLinearBenchmarkPositive : IO Unit := do
  let mut layer ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 32 64 false
  let t0 ← IO.monoNanosNow
  for _ in [:3] do
    let gW ← Tensor.randn #[64, 32] false
    layer ← ManifoldLinear.applyDualMapUpdate layer gW none 0.01
  let t1 ← IO.monoNanosNow
  let elapsedMs := (t1.toFloat - t0.toFloat) / 1000000.0
  LeanTest.assertTrue (elapsedMs > 0.0)
    s!"Expected positive benchmark elapsed time, got {elapsedMs}"

@[test]
def testHyperbolicVectorConstraintAfterUpdate : IO Unit := do
  let p0 ← ManifoldVectorParam.init (V := Tyr.AD.Hyperbolic) 6
  let v0 := p0.value.coords
  LeanTest.assertTrue (Tensor.requiresGrad v0)
    "Expected hyperbolic manifold vector to require gradients after init"
  LeanTest.assertTrue (Tensor.isLeaf v0)
    "Expected hyperbolic manifold vector to be a leaf tensor after init"

  let g ← Tensor.randn #[7] false
  let p1 ← ManifoldVectorParam.applyDualMapUpdate p0 g 0.02
  let coords := p1.value.coords
  LeanTest.assertTrue (Tensor.requiresGrad coords)
    "Expected hyperbolic manifold vector to require gradients after update"
  LeanTest.assertTrue (Tensor.isLeaf coords)
    "Expected hyperbolic manifold vector to remain a leaf tensor after update"
  let inner := Tyr.AD.Hyperbolic.minkowskiInner coords coords
  LeanTest.assertTrue (Float.abs (inner + 1.0) < 1e-4)
    s!"HyperbolicVector should preserve <x,x>_L = -1 after update, got {inner}"

@[test]
def testHyperbolicVectorBenchmarkPositive : IO Unit := do
  let mut p ← ManifoldVectorParam.init (V := Tyr.AD.Hyperbolic) 8
  let t0 ← IO.monoNanosNow
  for _ in [:3] do
    let g ← Tensor.randn #[9] false
    p ← ManifoldVectorParam.applyDualMapUpdate p g 0.01
  let t1 ← IO.monoNanosNow
  let elapsedMs := (t1.toFloat - t0.toFloat) / 1000000.0
  LeanTest.assertTrue (elapsedMs > 0.0)
    s!"Expected positive hyperbolic-vector benchmark elapsed time, got {elapsedMs}"

@[test]
def testManifoldUpdatableSequentialComposes : IO Unit := do
  let l1 ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 8 16 true
  let l2 ← ManifoldLinear.init (M := Tyr.AD.Grassmann) 16 16 true
  let net : Sequential (StiefelLinear 8 16) (GrassmannLinear 16 16) := Sequential.mk' l1 l2

  let g1w ← Tensor.randn #[16, 8] false
  let g1b ← Tensor.randn #[16] false
  let g2w ← Tensor.randn #[16, 16] false
  let g2b ← Tensor.randn #[16] false
  let grad := ((g1w, some g1b), (g2w, some g2b))

  let net' ← ManifoldUpdatable.step net grad 0.02

  let w1 := net'.first.weight.matrix
  let w1tW1 := Tensor.mm (Tensor.transpose2d w1) w1
  LeanTest.assertTrue (Tensor.allclose w1tW1 (Tensor.eye 8) (rtol := 1e-4) (atol := 1e-5))
    "Sequential ManifoldUpdatable step should keep Stiefel first layer on manifold"

  let w2 := net'.second.weight.matrix
  let w2tW2 := Tensor.mm (Tensor.transpose2d w2) w2
  LeanTest.assertTrue (Tensor.allclose w2tW2 (Tensor.eye 16) (rtol := 1e-4) (atol := 1e-5))
    "Sequential ManifoldUpdatable step should keep Grassmann second layer representative orthonormal"

@[test]
def testManifoldUpdatablePairComposes : IO Unit := do
  let layer ← ManifoldLinear.init (M := Tyr.AD.Stiefel) 6 10 false
  let vec ← ManifoldVectorParam.init (V := Tyr.AD.Hyperbolic) 5
  let params : StiefelLinear 6 10 × HyperbolicVector 5 := (layer, vec)

  let gw ← Tensor.randn #[10, 6] false
  let gv ← Tensor.randn #[6] false
  let grad := ((gw, none), gv)

  let params' ← ManifoldUpdatable.step params grad 0.01

  let w := params'.1.weight.matrix
  let wtW := Tensor.mm (Tensor.transpose2d w) w
  LeanTest.assertTrue (Tensor.allclose wtW (Tensor.eye 6) (rtol := 1e-4) (atol := 1e-5))
    "Pair ManifoldUpdatable step should preserve Stiefel constraint for matrix component"

  let hv := params'.2.value.coords
  let inner := Tyr.AD.Hyperbolic.minkowskiInner hv hv
  LeanTest.assertTrue (Float.abs (inner + 1.0) < 1e-4)
    s!"Pair ManifoldUpdatable step should preserve hyperbolic constraint, got {inner}"

end Tests.ModularManifold
