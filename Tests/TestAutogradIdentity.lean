import LeanTest
import Tyr.Model.Utils
import Tyr.TensorStruct
import Tyr.Optim

namespace Tests.AutogradIdentity

open torch

@[test] def testIndependentParameterCreation : IO Unit := do
  let source := Tensor.fromFloatArray #[2.0, 3.0]
  let first ← source.parameter
  let second ← source.parameter
  Tensor.backwardLoss (Tensor.sumAll (first * first))
  LeanTest.assertEqual (← (← first.gradOf).toFloatArray) #[4.0, 6.0]
  LeanTest.assertEqual (← (← second.gradOf).toFloatArray) #[0.0, 0.0]
    "Separate parameter creations must have independent gradient state"

@[test] def testIndependentBiasInitialization : IO Unit := do
  let first ← Model.initBias #[2]
  let second ← Model.initBias #[2]
  Tensor.backwardLoss (Tensor.sumAll (Tensor.mulScalar first 3.0))
  LeanTest.assertEqual (← (← first.gradOf).toFloatArray) #[3.0, 3.0]
  LeanTest.assertEqual (← (← second.gradOf).toFloatArray) #[0.0, 0.0]
    "Equal constant parameter initializers must not share autograd identity"

@[test] def testBackwardPreservesReusablePureGraph : IO Unit := do
  let fresh ← Tensor.randn #[2]
  let expected ← (Tensor.Preserving.mulScalar fresh 2.0).toFloatArray
  let parameter ← fresh.parameter
  let loss := Tensor.sumAll (parameter * parameter)
  Tensor.backwardLoss loss
  parameter.zeroGrad
  Tensor.backwardLoss loss
  LeanTest.assertEqual (← (← parameter.gradOf).toFloatArray) expected
    "Backward must not invalidate an immutable forward value"

@[test] def testExplicitBackwardPreservesReusablePureGraph : IO Unit := do
  let fresh ← Tensor.randn #[2]
  let expected ← (Tensor.Preserving.mulScalar fresh 2.0).toFloatArray
  let parameter ← fresh.parameter
  let output := parameter * parameter
  Tensor.backward output (Tensor.onesLike output)
  parameter.zeroGrad
  Tensor.backward output (Tensor.onesLike output)
  LeanTest.assertEqual (← (← parameter.gradOf).toFloatArray) expected

@[test] def testFunctionalGradSeparatesEqualTreeLeaves : IO Unit := do
  let input := Tensor.fromFloatArray #[2.0, 3.0]
  let gradients := TensorStruct.functionalGrad
    (fun (left, right) => Tensor.sumAll (left * left + Tensor.Preserving.mulScalar right 3.0))
    (input, input)
  LeanTest.assertEqual (← gradients.1.toFloatArray) #[4.0, 6.0]
  LeanTest.assertEqual (← gradients.2.toFloatArray) #[3.0, 3.0]
  LeanTest.assertFalse gradients.1.requiresGrad
  LeanTest.assertFalse gradients.2.requiresGrad
  LeanTest.assertFalse input.requiresGrad

@[test] def testFunctionalGradPreservesCapturedGraph : IO Unit := do
  let parameter ← (Tensor.fromFloatArray #[2.0, 3.0]).parameter
  let captured := parameter * parameter
  let loss := Tensor.sumAll captured
  let gradients := TensorStruct.functionalGrad (fun (temporary : Tensor { shape := #[2], dtype := .Float32, device := .CPU }) =>
    Tensor.sumAll (Tensor.mul captured temporary)) parameter
  LeanTest.assertEqual (← gradients.toFloatArray) #[4.0, 9.0]
  LeanTest.assertEqual (← (← parameter.gradOf).toFloatArray) #[0.0, 0.0]
  Tensor.backwardLoss loss
  LeanTest.assertEqual (← (← parameter.gradOf).toFloatArray) #[4.0, 6.0]

@[test] def testOptimizerUpdatesCreateIndependentParameters : IO Unit := do
  let source := Tensor.fromFloatArray #[2.0, 3.0]
  let update := Tensor.onesLike source
  let first ← Optim.apply_updates source update
  let second ← Optim.apply_updates source update
  Tensor.backwardLoss (Tensor.sumAll (first * first))
  LeanTest.assertEqual (← (← first.gradOf).toFloatArray) #[6.0, 8.0]
  LeanTest.assertEqual (← (← second.gradOf).toFloatArray) #[0.0, 0.0]

@[test] def testParameterOfParameterHasIndependentGradientState : IO Unit := do
  let first ← (Tensor.fromFloatArray #[2.0, 3.0]).parameter
  let second ← first.parameter
  Tensor.backwardLoss (Tensor.sumAll (first * first))
  LeanTest.assertEqual (← (← first.gradOf).toFloatArray) #[4.0, 6.0]
  LeanTest.assertEqual (← (← second.gradOf).toFloatArray) #[0.0, 0.0]

@[test] def testIntegerParameterReportsIOError : IO Unit := do
  let rejected ← try
    let _ ← (Tensor.fullInt #[2] 1).parameter
    pure false
  catch _ => pure true
  LeanTest.assertTrue rejected "Integer tensors cannot be trainable parameters"

-- A pure factory cannot manufacture mutable parameter identity.
example : True := by
  fail_if_success have _ := Tensor.zeros #[2] true
  fail_if_success have _ := Tensor.ones #[2] true
  fail_if_success have _ := Tensor.full #[2] 1.0 true
  fail_if_success have _ := RawTensor.eye 2 true
  trivial

end Tests.AutogradIdentity
