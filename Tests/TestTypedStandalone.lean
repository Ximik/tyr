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
