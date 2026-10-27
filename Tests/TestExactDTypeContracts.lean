import Tests.Devices
import Tyr.Typed.Neural
import LeanTest
import Tests.Assertions

open torch

namespace Tests.ExactDTypeContracts

-- Missing dtype information must be represented explicitly, never by a default
-- tensor index that allows differently typed values to share the same type.
example : True := by
  fail_if_success
    have _ : StaticSpec := { shape := #[2], device := .CPU }
  trivial

@[test] def testRuntimePromotionRetainsItsResultIndex : IO Unit := do
  let left := Tensor.ones #[2]
  let right := (Tensor.ones #[2]).castDType .Float64
  let result ← IO.ofExcept (Tensor.Promoting.add? left right)
  LeanTest.assertEqual result.dtype .Float64
  LeanTest.assertEqual result.tensor.raw.dtype result.dtype
  LeanTest.assertEqual (← result.tensor.toFloatArray) #[2.0, 2.0]

@[test] def testFileLoadingRetainsItsDTypeWitness : IO Unit := do
  IO.FS.withTempDir fun directory => do
    let path := (directory / "weights.pt").toString
    let input := (Tensor.fromFloatArray #[1.25, 2.5]).toBFloat16
    input.saveTensor path
    let packed ← SomeDTypeTensor.loadTensorExact #[2] .CPU path
    LeanTest.assertEqual packed.1 .BFloat16
    LeanTest.assertEqual packed.2.raw.dtype packed.1
    LeanTest.assertEqual (← packed.2.toFloatArray) #[1.25, 2.5]
    Tests.assertThrows (SomeDTypeTensor.loadTensorExact #[1, 2] .CPU path)
    for device in ← Tests.acceleratorDevices do
      Tests.assertThrows (SomeDTypeTensor.loadTensorExact #[2] device path)

end Tests.ExactDTypeContracts
