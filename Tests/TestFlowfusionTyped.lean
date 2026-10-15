import Tyr.Model.BranchingFlows
import LeanTest

open torch torch.flowfusion
namespace Tests.FlowfusionTyped

private def condition : Tensor { shape := #[2, 1], dtype := .Float32, device := .CPU } := Tensor.full #[2, 1] 0.25
private def lossMask : Tensor { shape := #[2, 3], dtype := .Bool, device := .CPU } := Tensor.ltScalar (Tensor.zeros #[2, 3]) 1.0

@[test] def testFlowfusionStaticMaskTraversal : IO Unit := do
  let wrapped := MaskedState.ofMasks (Tensor.full #[2] 2) condition lossMask
  LeanTest.assertEqual (TensorStruct.count wrapped) 1
  let scaled := TensorStruct.scale wrapped 3
  LeanTest.assertEqual (← scaled.state.toFloatArray) #[6, 6]
  LeanTest.assertEqual (← scaled.getCmask.toFloatArray) #[0.25, 0.25]
  LeanTest.assertEqual scaled.getLmask.actualSpec.dtype .Bool
  LeanTest.assertEqual scaled.getLmask.runtimeShape #[2, 3]
  let calls ← IO.mkRef (0 : Nat)
  let mapped ← TensorStruct.mapM (fun t => do
    calls.modify (· + 1)
    pure (Tensor.Preserving.mulScalar t 2)) wrapped
  LeanTest.assertEqual (← calls.get) 1
  LeanTest.assertEqual (← mapped.state.toFloatArray) #[4, 4]
  LeanTest.assertEqual (← mapped.getCmask.toFloatArray) #[0.25, 0.25]
  let added := TensorStruct.add wrapped scaled
  LeanTest.assertEqual (← added.state.toFloatArray) #[8, 8]
  LeanTest.assertEqual (← added.getCmask.toFloatArray) #[0.25, 0.25]

@[test] def testFlowfusionGuideOptionalMasks : IO Unit := do
  let guide : Guide (Tensor { shape := #[2], dtype := .Float32, device := .CPU }) { shape := #[2, 1], dtype := .Float32, device := .CPU }
      { shape := #[2, 3], dtype := .Bool, device := .CPU } := {
    value := Tensor.ones #[2], cmask := some ⟨condition⟩, lmask := some ⟨lossMask⟩ }
  let transformed := TensorStruct.scale guide 4
  LeanTest.assertEqual (TensorStruct.count transformed) 1
  LeanTest.assertEqual (← transformed.value.toFloatArray) #[4, 4]
  let some mask := transformed.getCmask | throw (IO.userError "missing condition mask")
  LeanTest.assertEqual (← mask.toFloatArray) #[0.25, 0.25]
  let absent : Guide (Tensor { shape := #[2], dtype := .Float32, device := .CPU }) { shape := #[2, 1], dtype := .Float32, device := .CPU }
      { shape := #[2, 3], dtype := .Bool, device := .CPU } := { guide with cmask := none, lmask := none }
  let absent := TensorStruct.add absent absent
  LeanTest.assertTrue absent.getCmask.isNone
  LeanTest.assertTrue absent.getLmask.isNone

@[test] def testFlowfusionAnchorMergeRetainsMasks : IO Unit := do
  let a := MaskedState.ofMasks (Tensor.full #[2] 2) condition lossMask
  let b := MaskedState.ofMasks (Tensor.full #[2] 6) (Tensor.full #[2, 1] 0.75) lossMask
  let merged := torch.branching.AnchorMerge.merge a b 1 3
  LeanTest.assertEqual (← (unmask merged).toFloatArray) #[5, 5]
  LeanTest.assertEqual (← merged.getCmask.toFloatArray) #[0.25, 0.25]
  let replaced := maskLike (Tensor.zeros #[2]) merged
  LeanTest.assertEqual (← replaced.state.toFloatArray) #[0, 0]
  LeanTest.assertEqual (← replaced.getCmask.toFloatArray) #[0.25, 0.25]

end Tests.FlowfusionTyped
