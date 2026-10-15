import LeanTest
import Tyr.Typed
import Tyr.Typed.Neural
import Tyr.TensorStruct

/-! This module intentionally imports no compatibility or legacy runtime API. -/

namespace Tests.TypedStandalone

open torch

-- Removed aliases must not leak through Basic, RawTensor or the typed facade.
example : True := by
  fail_if_success
    have _ := torch.T
  fail_if_success
    have _ := torch.DTensor
  fail_if_success
    have _ := torch.STensor
  fail_if_success
    have _ := torch.TT
  fail_if_success
    have _ := torch.SomeDTensor
  fail_if_success
    have _ := @torch.Tensor.assumeDType
  fail_if_success
    have _ := @torch.Tensor.assumeSpec
  fail_if_success
    have _ := @torch.Tensor.mk
  trivial

example (_x : Tensor { shape := #[2, 3], dtype := .Float32, device := .CPU }) : True := by
  fail_if_success
    have _wrongShape : Tensor { shape := #[3, 2], dtype := .Float32, device := .CPU } := _x
  fail_if_success
    have _wrongDtype : Tensor { shape := #[2, 3], dtype := .Float64, device := .CPU } := _x
  trivial

-- Native handles require an explicit boundary crossing; neither record syntax
-- nor structural updates may replace the checked handle or change its index.
example (_x : Tensor { shape := #[2, 3], dtype := .Float32, device := .CPU }) (_raw : RawTensor) : True := by
  fail_if_success
    have _ : RawTensor := _x
  fail_if_success
    have _ : Tensor { shape := #[2, 3], dtype := .Float32, device := .CPU } := ⟨_raw⟩
  fail_if_success
    have _ : Tensor { shape := #[2, 3], dtype := .Float32, device := .CPU } := { _x with raw := _raw }
  fail_if_success
    have _ : Tensor { shape := #[6], dtype := .Float32, device := .CPU } := { raw := _x.raw }
  trivial

@[test]
def testRawAdmissionRejectsFalseClaims : IO Unit := do
  let raw := (Tensor.full #[2, 3] 7).raw
  let rejected : Array StaticSpec := #[
    { shape := #[6], dtype := .Float32, device := .CPU },
    { shape := #[2, 3], dtype := .Int64, device := .CPU },
    { shape := #[2, 3], dtype := .Float32, device := (.CUDA 0) },
    { shape := #[2, 3], dtype := .Unknown "Float32_typo", device := .CPU }]
  for spec in rejected do
    LeanTest.assertTrue ((Tensor.ofRaw (σ := spec) raw).toOption.isNone)
      s!"Raw admission must reject {reprStr spec}"
  match Tensor.ofRaw (σ := { shape := #[2, 3], device := .CPU }) raw with
  | .error message => LeanTest.fail message
  | .ok tensor =>
      LeanTest.assertEqual (← tensor.toFloatArray) #[7, 7, 7, 7, 7, 7]
  LeanTest.assertTrue ((Tensor.ofTensor (shape := #[6]) .Float32 raw).toOption.isNone)
    "Checked raw conversion must not reshape equal-size storage"
  LeanTest.assertTrue ((Tensor.ofTensor (shape := #[2, 3]) .Int64 raw).toOption.isNone)
    "Checked raw conversion must not cast the native dtype"

@[test]
def testErasedDTypeCopyRejectsBeforeMutation : IO Unit := do
  let destination := Tensor.fullInt #[2] 9
  let source := Tensor.fromFloatArray #[1.75, 2.25]
  let rejected ← try
    let _ ← Tensor.copyInto destination.forgetDType source.forgetDType
    pure false
  catch error =>
    LeanTest.assertTrue (error.toString.containsSubstr "dtype")
      "Copy rejection must explain the dtype mismatch"
    pure true
  LeanTest.assertTrue rejected "Dtype erasure must not enable an implicit converting copy"
  LeanTest.assertEqual (← destination.toFloatArray) #[9, 9]
    "Rejected copy must leave the destination unchanged"
  let destination ← Tensor.copyInto destination source.toInt64
  LeanTest.assertEqual (← destination.toFloatArray) #[1, 2]
    "Explicit native conversion remains available"

@[test]
def testPreservingReplacementChecksErasedDType : IO Unit := do
  let original := (Tensor.fullInt #[2] 9).forgetDType
  LeanTest.assertTrue ((original.checkedReplacement? (Tensor.ones #[2]).raw).toOption.isNone)
    "A preserving adapter must reject dtype drift even for a partial specification"
  match original.checkedReplacement? (Tensor.fullInt #[2] 4).raw with
  | .error message => LeanTest.fail message
  | .ok replacement => LeanTest.assertEqual (← replacement.toFloatArray) #[4, 4]
  let smallInts := ((Tensor.ones #[2, 2]).castDType .Int32).forgetDType
  LeanTest.assertTrue ((Tensor.Preserving.sumDim? smallInts 0).toOption.isNone)
    "Erased Int32 accumulation must not masquerade as a dtype-preserving reduction"
  let reduced := Tensor.sumDim smallInts 0
  LeanTest.assertEqual reduced.raw.dtype .Int64
    "The ordinary reduction still permits native integer accumulation"

@[test]
def testExplicitConversionsCheckNativeResults : IO Unit := do
  let original := (Tensor.fromFloatArray #[1.25, 2.75, 3.5, 4.5]).reshapeChecked #[2, 2]
  match original.castDType? .Int64 with
  | .error message => LeanTest.fail message
  | .ok converted =>
      LeanTest.assertEqual (← converted.toFloatArray) #[1, 2, 3, 4]
      LeanTest.assertEqual converted.runtimeShape #[2, 2]
      LeanTest.assertEqual converted.raw.dtype .Int64
  LeanTest.assertTrue ((original.castDType? (.Unknown "Float32_typo")).toOption.isNone)
    "Unrecognized dtype tokens must not weaken a concrete dtype claim"
  match original.reshapeChecked? #[4] with
  | .error message => LeanTest.fail message
  | .ok reshaped => LeanTest.assertEqual reshaped.runtimeShape #[4]
  LeanTest.assertTrue ((original.reshapeChecked? #[3]).toOption.isNone)
    "Runtime reshape must reject a different element count"
  LeanTest.assertTrue ((original.reshapeChecked? #[18446744073709551615]).toOption.isNone)
    "Wrapped negative extents must not act as native shape inference"
  let pinned := Tensor.onesFor { shape := #[2], dtype := .Float32, device := .CPU }
  let converted ← TensorStruct.castDType .BFloat16 pinned.forgetDType
  LeanTest.assertEqual converted.raw.dtype .BFloat16
  LeanTest.assertTrue (converted.staticSpec.device == .CPU)
    "Explicit tree conversion retains placement policy"
  LeanTest.assertEqual (← converted.toFloatArray) #[1, 1]
  let rejected ← try
    let _ ← TensorStruct.castDType .BFloat16 pinned
    pure false
  catch _ => pure true
  LeanTest.assertTrue rejected "A tree cast cannot contradict a leaf's concrete dtype index"

@[test]
def testFoundationTypedOperationsAndRawMetadata : IO Unit := do
  let a := Tensor.full #[2, 3] 2.0
  let b := Tensor.full #[3, 4] 3.0
  let product : Tensor { shape := #[2, 4], dtype := .Float32, device := .CPU } := Tensor.mm a b
  match product.validate with
  | .ok () => pure ()
  | .error message => LeanTest.fail message
  LeanTest.assertEqual (Tensor.item (Tensor.sumAll product)) 144.0
    "Standalone typed imports must execute native matrix operations"
  LeanTest.assertTrue (!(toString product.raw).isEmpty)
    "Raw metadata formatting must not depend on the legacy alias"

@[test]
def testExistentialTensorPreservesFullSpec : IO Unit := do
  let spec : StaticSpec :=
    { shape := #[2, 3], dtype := .Float64, device := .CPU }
  let packed : Sigma Tensor := (Tensor.onesFor spec).toSome
  LeanTest.assertTrue (packed.fst == spec)
    "Existential tensors must retain shape, dtype, and concrete device"
  match packed.snd.validate with
  | .ok () => pure ()
  | .error message => LeanTest.fail message
  LeanTest.assertEqual (← packed.snd.toFloatArray) #[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
  match (Tensor.onesFor spec).forgetDType.reifyDType with
  | .error message => LeanTest.fail message
  | .ok ⟨dtype, tensor⟩ =>
      LeanTest.assertEqual dtype .Float64
      LeanTest.assertEqual tensor.runtimeShape #[2, 3]
      LeanTest.assertTrue (tensor.staticSpec.device == .CPU)
        "Recovering the native dtype must retain the concrete device"

end Tests.TypedStandalone

namespace Tests.TypedConvolution
open torch

example (_input : Tensor { shape := #[1, 2, 5], dtype := .Float32, device := .CPU })
    (_wrongChannels : Tensor { shape := #[3, 4, 3], dtype := .Float32, device := .CPU })
    (_wrongDtype : Tensor { shape := #[3, 2, 3], dtype := .Float64, device := .CPU }) : True := by
  fail_if_success
    have _ := _input.conv1d _wrongChannels
  fail_if_success
    have _ := _input.conv1d _wrongDtype
  trivial

@[test]
def testTypedConv1dValuesAndMetadata : IO Unit := do
  let input := (Tensor.fromFloatArray #[1, 2, 3, 4, 5]).reshapeChecked #[1, 1, 5]
  let weight := Tensor.ones #[1, 1, 3]
  let plain : Tensor { shape := #[1, 1, 3], dtype := .Float32, device := .CPU } := input.conv1d weight
  LeanTest.assertEqual (← plain.toFloatArray) #[6.0, 9.0, 12.0]
  LeanTest.assertEqual plain.runtimeShape #[1, 1, 3]
  let strided : Tensor { shape := #[1, 1, 3], dtype := .Float32, device := .CPU } := input.conv1d weight 2 1
  LeanTest.assertEqual (← strided.toFloatArray) #[3.0, 9.0, 9.0]
  let dilated : Tensor { shape := #[1, 1, 1], dtype := .Float32, device := .CPU } := input.conv1d weight 1 0 2
  LeanTest.assertEqual (← dilated.toFloatArray) #[9.0]

end Tests.TypedConvolution
