import Tyr.Model.FLM
import LeanTest

open torch
open torch.flm

private def tinyTokens : Tensor { shape := #[(2 : UInt64), 3], dtype := .Int64 } :=
  Tensor.reshapeChecked (Tensor.fromInt64Array #[0, 1, 2, 3, 4, 0]) #[(2 : UInt64), 3]

private def identityDenoiser : FlowDenoiser 3 5 Unit :=
  { forward := fun {_batch} _ x _ _ => pure x
    forwardPure := some (fun {_batch} _ x _ _ => x) }

@[test]
def testFLMOneHotAndCorruptAtDataTime : IO Unit := do
  let tokens := tinyTokens
  let target := oneHot (vocab := 5) tokens
  let targetSum := Tensor.item (Tensor.sumAll target)
  LeanTest.assertTrue (Float.abs (targetSum - 6.0) < 1.0e-5)
    "oneHot should put one unit of mass on each token"

  let tau := Tensor.full #[(2 : UInt64)] 1.0
  let (xT, target') ← corruptContinuous (vocab := 5) tokens tau
  LeanTest.assertTrue (Tensor.allclose target target' 1.0e-6 1.0e-6)
    "corruptContinuous should return the same dense target"
  LeanTest.assertTrue (Tensor.allclose xT target 1.0e-6 1.0e-6)
    "at t=1 corruption should equal one-hot data"

@[test]
def testFLMLossFinite : IO Unit := do
  let cfg : FlowConfig := {}
  let tau := Tensor.full #[(2 : UInt64)] 0.5
  let loss ← flowLossGivenTau cfg TimeMap.identity identityDenoiser () tinyTokens tau
  let meanLoss := Tensor.item (Tensor.meanAll loss)
  LeanTest.assertTrue (meanLoss.isFinite) "FLM loss should be finite"
  LeanTest.assertTrue (meanLoss > 0.0) "FLM loss should be positive for an untrained denoiser"

@[test]
def testFMLMPsdLossFinite : IO Unit := do
  let cfg : FlowConfig := {}
  let tauS := Tensor.full #[(2 : UInt64)] 0.1
  let tauU := Tensor.full #[(2 : UInt64)] 0.5
  let tauT := Tensor.full #[(2 : UInt64)] 0.9
  let loss ← psdLossGivenTau cfg TimeMap.identity identityDenoiser () tinyTokens tauS tauU tauT
  let meanLoss := Tensor.item (Tensor.meanAll loss)
  LeanTest.assertTrue (meanLoss.isFinite) "PSD loss should be finite"
  LeanTest.assertTrue (meanLoss > 0.0) "PSD loss should be positive for an untrained denoiser"

@[test]
def testFLMGenerateFiniteTokens : IO Unit := do
  let cfg : FlowConfig := {}
  let out ← generateFLM (seq := 3) (vocab := 5) (batch := 2)
    cfg TimeMap.identity identityDenoiser () 2
  let total := Tensor.item (Tensor.sumAll (Tensor.toFloat32 out))
  LeanTest.assertTrue (total.isFinite) "generated token ids should be numeric"
