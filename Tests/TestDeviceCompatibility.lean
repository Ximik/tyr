import LeanTest
import Tyr.Core.Typed.Ops

open torch

namespace Tests.TestDeviceCompatibility

-- Unrelated acquired placements require equality evidence before tensor operations.
example {cuda : Device} (_cpu : Tensor ⟨#[2], .Float32, .CPU⟩)
    (_cuda : Tensor ⟨#[2], .Float32, cuda⟩) : True := by
  fail_if_success have _ := _cpu.addB _cuda
  fail_if_success have _ := _cuda.subB _cpu
  fail_if_success have _ := _cpu.mulB _cuda
  fail_if_success have _ := _cuda.divB _cpu
  trivial

example {cuda0 cuda1 mps : Device}
    (_cuda0 : Tensor ⟨#[2], .Float32, cuda0⟩)
    (_cuda1 : Tensor ⟨#[2], .Float32, cuda1⟩)
    (_mps : Tensor ⟨#[2], .Float32, mps⟩) : True := by
  fail_if_success have _ := _cuda0.addB _cuda1
  fail_if_success have _ := _cuda0.mulB _mps
  trivial

example {device : Device} (left right : Tensor ⟨#[2], .Float32, device⟩) :
    Tensor ⟨#[2], .Float32, device⟩ := left.addB right

example {_left _right : Device} : True := by
  fail_if_success
    have _ : Device.SameDeviceCompatible .CPU _left := inferInstance
  fail_if_success
    have _ : Device.SameDeviceCompatible _left _right := inferInstance
  trivial

private def expectDeviceKind (result : Except String DeviceKind) (expected : DeviceKind) : IO Unit := do
  match result with
  | .ok actual =>
      LeanTest.assertTrue (actual == expected)
        s!"Expected device {expected.render}, got {actual.render}"
  | .error message => LeanTest.fail message

private def expectMismatch (result : Except String DeviceKind) : IO Unit := do
  match result with
  | .ok device => LeanTest.fail s!"Unexpectedly accepted device {device.render}"
  | .error _ => pure ()

@[test] def testStrictDeviceKindMetadata : IO Unit := do
  for device in #[DeviceKind.CPU, .MPS, .CUDA 0, .CUDA 1] do
    expectDeviceKind (DeviceKind.sameResult? device device) device
  for (left, right) in #[(DeviceKind.CPU, DeviceKind.CUDA 0), (.CPU, .MPS),
      (.CUDA 0, .CUDA 1), (.CUDA 0, .MPS)] do
    expectMismatch (DeviceKind.sameResult? left right)
    expectMismatch (DeviceKind.sameResult? right left)

end Tests.TestDeviceCompatibility
