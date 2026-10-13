import LeanTest
import Tyr.Typed

/-! This module intentionally imports no compatibility or legacy runtime API. -/

namespace Tests.TypedStandalone

open torch

-- The legacy alias must not leak through Basic, RawTensor or the typed facade.
example : True := by
  fail_if_success
    have _ := torch.T
  trivial

example (_x : DTensor #[2, 3] .Float32) : True := by
  fail_if_success
    have _wrongShape : DTensor #[3, 2] .Float32 := _x
  fail_if_success
    have _wrongDtype : DTensor #[2, 3] .Float64 := _x
  trivial

@[test]
def testFoundationTypedOperationsAndRawMetadata : IO Unit := do
  let a := Tensor.full #[2, 3] 2.0
  let b := Tensor.full #[3, 4] 3.0
  let product : DTensor #[2, 4] .Float32 := Tensor.mm a b
  match product.validate with
  | .ok () => pure ()
  | .error message => LeanTest.fail message
  LeanTest.assertEqual (Tensor.item (Tensor.sumAll product)) 144.0
    "Standalone typed imports must execute native matrix operations"
  LeanTest.assertTrue (!(toString product.raw).isEmpty)
    "Raw metadata formatting must not depend on the legacy alias"

end Tests.TypedStandalone

namespace Tests.TypedConvolution
open torch

example (_input : DTensor #[1, 2, 5] .Float32)
    (_wrongChannels : DTensor #[3, 4, 3] .Float32)
    (_wrongDtype : DTensor #[3, 2, 3] .Float64) : True := by
  fail_if_success
    have _ := _input.conv1d _wrongChannels
  fail_if_success
    have _ := _input.conv1d _wrongDtype
  trivial

@[test]
def testTypedConv1dValuesAndMetadata : IO Unit := do
  let input := (Tensor.fromFloatArray #[1, 2, 3, 4, 5]).reshapeChecked #[1, 1, 5]
  let weight := Tensor.ones #[1, 1, 3]
  let plain : DTensor #[1, 1, 3] .Float32 := input.conv1d weight
  LeanTest.assertEqual (← plain.toFloatArray) #[6.0, 9.0, 12.0]
  LeanTest.assertEqual plain.runtimeShape #[1, 1, 3]
  let strided : DTensor #[1, 1, 3] .Float32 := input.conv1d weight 2 1
  LeanTest.assertEqual (← strided.toFloatArray) #[3.0, 9.0, 9.0]
  let dilated : DTensor #[1, 1, 1] .Float32 := input.conv1d weight 1 0 2
  LeanTest.assertEqual (← dilated.toFloatArray) #[9.0]

end Tests.TypedConvolution
