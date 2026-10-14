import LeanTest
import Tyr.Typed.Neural
import Tyr.Inference.OwnedKV
import Tyr.Model.BranchingFlows.TypedOps

namespace Tests.TensorValueUpdates

open torch

@[test] def testCopyIntoPreservesDestinationAlias : IO Unit := do
  let original := Tensor.full #[2] 3
  let source := Tensor.full #[2] 7
  let updated ← Tensor.copyInto original source
  LeanTest.assertEqual (← original.toFloatArray) #[3, 3]
    "A returned-value copy must preserve retained destination aliases"
  LeanTest.assertEqual (← updated.toFloatArray) #[7, 7]
  LeanTest.assertEqual (← source.toFloatArray) #[7, 7]

@[test] def testCopyIntoPreservesStorageView : IO Unit := do
  let original := Tensor.full #[4] 3
  let view := original.sliceChecked 0 1 2 #[2]
  let updated ← Tensor.copyInto original (Tensor.full #[4] 7)
  LeanTest.assertEqual (← view.toFloatArray) #[3, 3]
    "A retained storage view must preserve its original values"
  LeanTest.assertEqual (← updated.toFloatArray) #[7, 7, 7, 7]

@[test] def testCopyIntoPreservesDetachedAlias : IO Unit := do
  let original := Tensor.full #[2] 3
  let updated ← Tensor.copyInto original.detach (Tensor.full #[2] 7)
  LeanTest.assertEqual (← original.toFloatArray) #[3, 3]
    "Detached tensors still alias storage and must be protected by COW"
  LeanTest.assertEqual (← updated.toFloatArray) #[7, 7]

@[test] def testSliceCopyPreservesDestinationAlias : IO Unit := do
  let original := Tensor.full #[4] 3
  let updated ← data.copySliceIO original 0 1 (Tensor.full #[2] 7)
  LeanTest.assertEqual (← original.toFloatArray) #[3, 3, 3, 3]
    "A returned-value slice copy must preserve the original tensor"
  LeanTest.assertEqual (← updated.toFloatArray) #[3, 7, 7, 3]

@[test] def testSliceScatterPreservesStorageView : IO Unit := do
  let original := Tensor.full #[4] 3
  let view := original.sliceChecked 0 1 2 #[2]
  let updated := Tensor.sliceScatter original 0 1 (Tensor.full #[2] 7)
  LeanTest.assertEqual (← view.toFloatArray) #[3, 3]
    "Functional slice updates must preserve retained storage views"
  LeanTest.assertEqual (← updated.toFloatArray) #[3, 7, 7, 3]

@[test] def testBranchingCopyPreservesDestinationAlias : IO Unit := do
  let original := Tensor.full #[2] 3
  let updated ← branching.copyIntoChecked? original (Tensor.full #[2] 7)
  LeanTest.assertEqual (← original.toFloatArray) #[3, 3]
    "Branching copy must preserve retained destination aliases"
  LeanTest.assertEqual (← updated.toFloatArray) #[7, 7]

@[test] def testDynamicCopyPreservesDTypeAndRejectsImplicitCast : IO Unit := do
  let original := (Tensor.fullInt #[2] 3).forgetDType
  let copied ← Tensor.copyInto original (Tensor.fullInt #[2] 7).forgetDType
  LeanTest.assertEqual copied.raw.dtype DType.Int64
  LeanTest.assertTrue (copied.raw.device == original.raw.device)
  LeanTest.assertEqual (← copied.toFloatArray) #[7, 7]
  let rejected ← try
    let _ ← Tensor.copyInto original (Tensor.full #[2] 1.75).forgetDType
    pure false
  catch _ => pure true
  LeanTest.assertTrue rejected "Dynamic dtype does not permit implicit converting copies"
  LeanTest.assertEqual (← original.toFloatArray) #[3, 3]

@[test] def testDynamicRequiresGradPreservesNativeMetadata : IO Unit := do
  let original := ((Tensor.full #[2] 3).castDType .Float64).forgetDType
  let trainable ← original.setRequiresGrad true
  LeanTest.assertEqual trainable.raw.dtype DType.Float64
  LeanTest.assertTrue (trainable.raw.device == original.raw.device)
  LeanTest.assertEqual (← trainable.toFloatArray) #[3, 3]
  LeanTest.assertTrue trainable.requiresGrad "The returned dynamic tensor must enable autograd"
  LeanTest.assertTrue (!original.requiresGrad) "Changing autograd flags must preserve aliases"

end Tests.TensorValueUpdates
