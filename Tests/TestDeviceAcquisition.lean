import Tests.Assertions
import Tyr.Torch

open torch

-- Accelerator descriptions are not capabilities. Even the subtype constructor
-- requires availability evidence that pure code cannot manufacture.
example : True := by
  fail_if_success
    have _ : Device := .CUDA 0
  fail_if_success
    have _ : Device := .MPS
  fail_if_success
    have _ : Device := ⟨.CUDA 0, by trivial⟩
  fail_if_success
    have _ : Device := ⟨.CUDA 128, by constructor⟩
  fail_if_success
    have _ : Device := ⟨.MPS, by decide⟩
  trivial

-- CPU placement and equality remain reducible in tensor type indices.
example : Device.CPU.kind = DeviceKind.CPU := rfl
example : (Device.CPU == Device.CPU) = true := rfl
example : decide (Device.CPU = Device.CPU) = true := rfl
example : (default : Device) = Device.CPU := rfl

namespace Tests.TestDeviceAcquisition

@[test]
def testDeviceAcquisitionRejectsInvalidOrdinals : IO Unit := do
  Tests.assertThrows (Device.cuda 128)
  Tests.assertThrows (Device.cuda 18446744073709551615)

@[test]
def testDeviceAcquisitionMatchesBackendAvailability : IO Unit := do
  if ← cuda_is_available then
    let first ← Device.cuda 0
    let second ← Device.cuda 0
    LeanTest.assertTrue (first.kind == .CUDA 0)
    LeanTest.assertTrue (first == second)
  else
    Tests.assertThrows (Device.cuda 0)
  if ← mps_is_available_stable then
    let first ← Device.mps
    let second ← Device.mps
    LeanTest.assertTrue (first.kind == .MPS)
    LeanTest.assertTrue (first == second)
  else
    Tests.assertThrows Device.mps

end Tests.TestDeviceAcquisition
