import LeanTest
import Tyr.Module.Derive
import Tyr.TensorStruct
import Tyr.Inference.OwnedKV

namespace Tests.TensorAutogradEffects

open torch

@[test]
def testGradientSnapshotsSurviveAccumulationAndZeroing : IO Unit := do
  let fresh ← data.cloneInferenceIO (Tensor.fromFloatArray #[2.0, 3.0])
  let parameter ← fresh.setRequiresGrad true
  let missing ← parameter.gradOf
  LeanTest.assertEqual (← missing.toFloatArray) #[0.0, 0.0]
  Tensor.backwardLoss (Tensor.sumAll (parameter * parameter))
  let first ← parameter.gradOf
  LeanTest.assertEqual (← first.toFloatArray) #[4.0, 6.0]
  LeanTest.assertTrue (!first.requiresGrad) "Gradient snapshots must be detached"
  Tensor.backwardLoss (Tensor.sumAll (Tensor.mulScalar parameter 3.0))
  let accumulated ← parameter.gradOf
  LeanTest.assertEqual (← accumulated.toFloatArray) #[7.0, 9.0]
  LeanTest.assertEqual (← first.toFloatArray) #[4.0, 6.0]
  LeanTest.assertEqual (← missing.toFloatArray) #[0.0, 0.0]
  parameter.zeroGrad
  let cleared ← parameter.gradOf
  LeanTest.assertEqual (← cleared.toFloatArray) #[0.0, 0.0]
    "An ignored Unit result must still execute gradient zeroing"
  LeanTest.assertEqual (← first.toFloatArray) #[4.0, 6.0]
  LeanTest.assertEqual (← accumulated.toFloatArray) #[7.0, 9.0]
  Tensor.backwardLoss (Tensor.sumAll (Tensor.mulScalar parameter 5.0))
  LeanTest.assertEqual (← (← parameter.gradOf).toFloatArray) #[5.0, 5.0]
  LeanTest.assertEqual (← cleared.toFloatArray) #[0.0, 0.0]

@[test]
def testGradientSnapshotsSurviveClipping : IO Unit := do
  let fresh ← data.cloneInferenceIO (Tensor.fromFloatArray #[3.0, 4.0])
  let parameter ← fresh.setRequiresGrad true
  Tensor.backwardLoss (Tensor.sumAll (parameter * parameter))
  let before ← parameter.gradOf
  let norm ← parameter.clipGradNorm 1.0
  LeanTest.assertEqual norm 10.0
  let clipped ← parameter.gradOf
  LeanTest.assertTrue ((Tensor.l2Norm clipped - 1.0).abs < 1e-6)
  LeanTest.assertEqual (← before.toFloatArray) #[6.0, 8.0]

@[test]
def testGradientTreeSnapshotsAndAliasedFlagUpdates : IO Unit := do
  let source ← data.cloneInferenceIO (Tensor.fromFloatArray #[2.0, 3.0])
  let parameter ← source.setRequiresGrad true
  LeanTest.assertTrue (!source.requiresGrad) "Flag updates must preserve the input alias"
  LeanTest.assertTrue (parameter.requiresGrad && parameter.isLeaf)
  let otherFresh ← data.cloneInferenceIO (Tensor.fromFloatArray #[5.0, 7.0])
  let other ← otherFresh.setRequiresGrad true
  let model := (parameter, other)
  Tensor.backwardLoss (Tensor.sumAll (parameter * parameter) + Tensor.sumAll other)
  let gradients ← TensorStruct.grads model
  TensorStruct.zeroGrads model
  let cleared ← TensorStruct.grads model
  LeanTest.assertEqual (← gradients.1.toFloatArray) #[4.0, 6.0]
  LeanTest.assertEqual (← gradients.2.toFloatArray) #[1.0, 1.0]
  LeanTest.assertEqual (← cleared.1.toFloatArray) #[0.0, 0.0]
  LeanTest.assertEqual (← cleared.2.toFloatArray) #[0.0, 0.0]

end Tests.TensorAutogradEffects
