import LeanTest
import Tests.Devices
import Tyr.Typed.Neural

namespace Tests.ExactDType

open torch

-- This requirement also fails with the old reifyDType API: its bare DType
-- witness admits the reserved wildcard and supplies no exclusion evidence.
private def reifiedIndex {σ : StaticSpec} (tensor : Tensor σ) :
    Except String { dtype : DType // dtype ≠ .Unknown "dynamic" } := do
  pure (← tensor.reifyDType).1

example {σ : StaticSpec} (tensor : Tensor σ)
    (exact : σ.dtype ≠ .Unknown "dynamic") : SomeDTypeTensor σ.shape σ.device :=
  tensor.packDType exact

example (_tensor : Tensor ⟨#[2], .Unknown "dynamic", .CPU⟩) : True := by
  fail_if_success
    have _ : torch.ExactDType := ⟨.Unknown "dynamic", by decide⟩
  fail_if_success
    have _ : torch.ExactDType := { val := .Unknown "dynamic", property := by decide }
  fail_if_success
    have _ : SomeDTypeTensor #[2] .CPU := ⟨⟨.Unknown "dynamic", by decide⟩, _tensor⟩
  fail_if_success
    have _ : SomeDTypeTensor #[2] .CPU := by
      constructor
      · exact ⟨.Unknown "dynamic", by decide⟩
      · exact _tensor
  fail_if_success
    have _ : SomeDTypeTensor #[2] .CPU := _tensor
  fail_if_success
    have _ := _tensor.packDType (by decide)
  trivial

example {_otherDevice : Device} (_packed : SomeDTypeTensor #[2] .CPU) : True := by
  fail_if_success
    have _ : Tensor ⟨#[2], .Unknown "dynamic", .CPU⟩ := _packed
  fail_if_success
    have _ : Tensor ⟨#[2], .Unknown "dynamic", .CPU⟩ := _packed.2
  fail_if_success
    have _ : Tensor ⟨#[2], .Float32, .CPU⟩ := _packed.2
  fail_if_success
    have _ : SomeDTypeTensor #[2] _otherDevice := _packed
  trivial

private abbrev float32 : torch.ExactDType := ⟨.Float32, by decide⟩

@[test] def testExactDTypeReificationRetainsValuesAndMetadata : IO Unit := do
  for dtype in #[DType.Float32, .BFloat16, .Int64, .Complex64] do
    let source := ((Tensor.fromFloatArray #[2, -3]).castDType dtype)
    let packed ← IO.ofExcept source.reifyDType
    let index ← IO.ofExcept (reifiedIndex source)
    LeanTest.assertEqual packed.1.val dtype
    LeanTest.assertEqual index.val dtype
    LeanTest.assertEqual packed.2.dtype dtype
    LeanTest.assertEqual packed.2.raw.dtype dtype
    LeanTest.assertEqual packed.2.runtimeShape #[2]
    LeanTest.assertTrue (packed.2.runtimeDevice == .CPU)
    LeanTest.assertTrue packed.2.validate.toOption.isSome
    LeanTest.assertEqual (← packed.2.toFloatArray) #[2, -3]
    LeanTest.assertEqual source.raw.dtype dtype

@[test] def testExactDTypeRawAdmissionChecksFullSpec : IO Unit := do
  let raw := (Tensor.fromFloatArray #[2, -3]).raw
  let packed ← IO.ofExcept (SomeDTypeTensor.ofRaw (shape := #[2]) (device := .CPU) raw)
  LeanTest.assertEqual packed.1.val .Float32
  LeanTest.assertEqual (← packed.2.toFloatArray) #[2, -3]
  LeanTest.assertTrue
    (SomeDTypeTensor.ofRaw (shape := #[3]) (device := .CPU) raw).toOption.isNone
  for device in ← Tests.acceleratorDevices do
    LeanTest.assertTrue
      (SomeDTypeTensor.ofRaw (shape := #[2]) (device := device) raw).toOption.isNone
  LeanTest.assertTrue (torch.ExactDType.ofDType? (.Unknown "dynamic")).toOption.isNone
  -- Unknown metadata tokens remain exact, not permissions to reinterpret data.
  let future ← IO.ofExcept (torch.ExactDType.ofDType? (.Unknown "future_dtype"))
  LeanTest.assertEqual future.val (.Unknown "future_dtype")
  LeanTest.assertTrue
    (Tensor.ofRaw (σ := ⟨#[2], future.val, .CPU⟩) raw).toOption.isNone

private def alignDType {shape : Shape} {device : Device}
    (left right : SomeDTypeTensor shape device) :
    Except String (Tensor ⟨shape, left.1.val, device⟩) := do
  let ⟨same⟩ ← left.sameDType? right
  pure (same.symm ▸ right.2)

@[test] def testExactDTypeEqualityBeforeSharedDTypeUse : IO Unit := do
  let source := (Tensor.fromFloatArray #[1, 2, 3, 4]).reshapeChecked #[2, 2]
  let left ← IO.ofExcept source.reifyDType
  let right ← IO.ofExcept source.reifyDType
  let aligned ← IO.ofExcept (alignDType left right)
  LeanTest.assertEqual (← (Tensor.mm left.2 aligned).toFloatArray) #[7, 10, 15, 22]
  let different ← IO.ofExcept (source.castDType .BFloat16).reifyDType
  LeanTest.assertTrue (left.sameDType? different).toOption.isNone
  LeanTest.assertTrue (left.atDType? ⟨.Int64, by decide⟩).toOption.isNone
  let recovered ← IO.ofExcept (left.atDType? float32)
  LeanTest.assertEqual (← recovered.toFloatArray) #[1, 2, 3, 4]

@[test] def testExactDTypePackagingPreservesAutogradIdentity : IO Unit := do
  let parameter ← (Tensor.fromFloatArray #[2, 3]).parameter
  let staticallyPacked := parameter.packDType (by decide)
  let packed ← IO.ofExcept staticallyPacked.2.reifyDType
  let recovered ← IO.ofExcept (packed.atDType? float32)
  LeanTest.assertTrue recovered.requiresGrad
  LeanTest.assertFalse recovered.hasGradFn
  let intermediate := recovered * recovered
  Tensor.backwardLoss (Tensor.sumAll intermediate) (retainGraph := some true)
  LeanTest.assertEqual (← (← parameter.gradOf).toFloatArray) #[4, 6]
  LeanTest.assertEqual (← (← recovered.gradOf).toFloatArray) #[4, 6]
  parameter.zeroGrad
  let graph ← IO.ofExcept intermediate.reifyDType
  let recoveredGraph ← IO.ofExcept (graph.atDType? float32)
  LeanTest.assertTrue recoveredGraph.requiresGrad
  LeanTest.assertTrue recoveredGraph.hasGradFn
  Tensor.backwardLoss (Tensor.sumAll recoveredGraph)
  LeanTest.assertEqual (← (← parameter.gradOf).toFloatArray) #[4, 6]

@[test] def testExactDTypeReificationKeepsImaginaryValues : IO Unit := do
  let negative : Tensor ⟨#[1], .Complex64, .CPU⟩ :=
    (Tensor.full #[1] (-4)).castDType .Complex64
  let imaginary := Tensor.sqrt negative
  let packed ← IO.ofExcept imaginary.reifyDType
  let recovered ← IO.ofExcept (packed.atDType? ⟨.Complex64, by decide⟩)
  LeanTest.assertEqual (← (Tensor.abs recovered).toFloatArray) #[2]
  LeanTest.assertEqual (← (recovered * recovered).toFloatArray) #[-4]

end Tests.ExactDType
