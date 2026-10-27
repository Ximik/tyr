import Tests.Assertions
import Tyr.ConstantSupport
import Tyr.Typed
import Tyr.Torch

open torch

example (device : Device) : DType.ConstantSupported .Float32 device := inferInstance
example (device : Device) : DType.ConstantSupported .Int64 device := inferInstance
example : DType.ConstantSupported .Float64 .CPU := inferInstance
example : DType.ConstantSupported .BFloat16 .CPU := inferInstance
example : DType.ConstantSupported .Complex128 .CPU := inferInstance

-- Recognizing a dtype, or exposing the capability class constructor, cannot
-- establish that its constants are supported. Nor can `default` bypass this.
example : True := by
  fail_if_success
    have _ : DType.ConstantSupported .QInt8 .CPU := inferInstance
  fail_if_success
    have _ : DType.ConstantSupported .QInt8 .CPU := ⟨by trivial⟩
  fail_if_success
    have _ : DType.ConstantSupported .Float8E8M0FNU .CPU := inferInstance
  fail_if_success
    have _ : Inhabited (Tensor ⟨#[2], .QInt8, .CPU⟩) := inferInstance
  trivial

namespace Tests.TestConstantSupport

private def supportedCPU : Array DType := #[
  .UInt8, .Int8, .Int16, .Int32, .Int64, .Float16, .BFloat16, .Float32,
  .Float64, .Bool, .Float8E4M3FN, .Float8E5M2, .Complex32, .Complex64,
  .Complex128, .Float8E5M2FNUZ, .Float8E4M3FNUZ, .UInt16, .UInt32, .UInt64
]

@[test]
def testConstantSupportCoversExactCPUCapabilities : IO Unit := do
  for dtype in DType.all do
    if supportedCPU.contains dtype then
      let support ← dtype.acquireConstantSupport .CPU
      let _ := support.down
      let zeros := Tensor.zerosFor ⟨#[2], dtype, .CPU⟩
      let ones := Tensor.onesFor ⟨#[2], dtype, .CPU⟩
      let eye := Tensor.eyeFor (n := 2) dtype
      LeanTest.assertEqual zeros.raw.dtype dtype
      LeanTest.assertEqual ones.raw.dtype dtype
      LeanTest.assertEqual eye.raw.dtype dtype
      LeanTest.assertEqual (← zeros.toFloatArray) #[0.0, 0.0]
      LeanTest.assertEqual (← ones.toFloatArray) #[1.0, 1.0]
      LeanTest.assertEqual (← eye.toFloatArray) #[1.0, 0.0, 0.0, 1.0]
    else
      Tests.assertThrows (dtype.acquireConstantSupport .CPU)

@[test]
def testConstantSupportRejectsUnrepresentableZero : IO Unit := do
  -- E8M0 is recognized and native conversion succeeds, but it has no zero.
  Tests.assertThrows (DType.Float8E8M0FNU.acquireConstantSupport .CPU)
    (some "cannot represent exact zero and one")

@[test]
def testConstantSupportChecksAcquiredAcceleratorPlacement : IO Unit := do
  let check (device : Device) : IO Unit := do
    let support ← DType.Float32.acquireConstantSupport device
    let _ := support.down
    let tensor := Tensor.onesFor ⟨#[2], .Float32, device⟩
    LeanTest.assertTrue (tensor.raw.device == device)
    LeanTest.assertEqual (← tensor.toFloatArray) #[1.0, 1.0]
  if ← cuda_is_available then check (← Device.cuda 0)
  if ← mps_is_available_stable then
    let device ← Device.mps
    check device
    Tests.assertThrows (DType.Float64.acquireConstantSupport device)

end Tests.TestConstantSupport
