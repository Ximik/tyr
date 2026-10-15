import LeanTest
import Tyr.Typed.Ops

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

-- A CPU vector does not get the rank-zero exception.
example {cuda mps : Device} (_cpuVector : Tensor ⟨#[1], .Float32, .CPU⟩)
    (_cudaScalar : Tensor ⟨#[], .Float32, cuda⟩)
    (_mpsScalar : Tensor ⟨#[], .Float32, mps⟩) : True := by
  fail_if_success have _ := _cpuVector.addB _cudaScalar
  fail_if_success have _ := _mpsScalar.addB _cpuVector
  fail_if_success have _ := _cudaScalar.addB _mpsScalar
  trivial

example {device : Device} (cpu : Tensor ⟨#[], .Float32, .CPU⟩)
    (other : Tensor ⟨#[2], .Float32, device⟩) :
    Tensor ⟨#[2], .Float32, device⟩ := cpu.addB other

example {device : Device} (cpu : Tensor ⟨#[], .Float32, .CPU⟩)
    (other : Tensor ⟨#[2], .Float32, device⟩) :
    Tensor ⟨#[2], .Float32, device⟩ := other.mulB cpu

example {device : Device} (cpu : Tensor ⟨#[], .Float32, .CPU⟩)
    (other : Tensor ⟨#[], .Float32, device⟩) :
    Tensor ⟨#[], .Float32, device⟩ := by
  rcases device with ⟨kind, valid⟩
  cases kind <;> exact other.addB cpu

example {device : Device} (left right : Tensor ⟨#[], .Float32, device⟩) :
    Tensor ⟨#[], .Float32, device⟩ := by
  rcases device with ⟨kind, valid⟩
  cases kind <;> exact left.addB right

example {leftShape rightShape : Shape} {leftDType rightDType : DType}
    {device : Device} : Device.PointwiseCompatible
    ⟨leftShape, leftDType, device⟩ ⟨rightShape, rightDType, device⟩ := inferInstance
example {spec : StaticSpec} : Device.PointwiseCompatible spec spec := inferInstance

example {_left _right : Device} : True := by
  fail_if_success
    have _ : Device.SameDeviceCompatible .CPU _left := inferInstance
  fail_if_success
    have _ : Device.SameDeviceCompatible _left _right := inferInstance
  fail_if_success
    have _ : Device.PointwiseCompatible
      ⟨#[1], .Float32, .CPU⟩ ⟨#[], .Float32, _right⟩ := inferInstance
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

/-- These are metadata checks, not accelerator execution tests. No CUDA or MPS
tensor is allocated by this test. -/
@[test] def testPointwiseDeviceKindMetadata : IO Unit := do
  for device in #[DeviceKind.CPU, .MPS, .CUDA 0, .CUDA 1] do
    expectDeviceKind (DeviceKind.pointwiseResult? #[2, 1] device #[3] device) device
    expectDeviceKind (DeviceKind.pointwiseResult? #[] .CPU #[2] device) device
    expectDeviceKind (DeviceKind.pointwiseResult? #[2] device #[] .CPU) device
    expectDeviceKind (DeviceKind.pointwiseResult? #[] .CPU #[] device) device
    expectDeviceKind (DeviceKind.pointwiseResult? #[] device #[] .CPU) device
  for (left, right) in #[(DeviceKind.CPU, DeviceKind.CUDA 0), (.CPU, .MPS),
      (.CUDA 0, .CUDA 1), (.CUDA 0, .MPS)] do
    expectMismatch (DeviceKind.pointwiseResult? #[2] left #[2] right)
    expectMismatch (DeviceKind.pointwiseResult? #[2] right #[2] left)
  -- A size-one or empty CPU vector is still not a zero-dimensional scalar.
  expectMismatch (DeviceKind.pointwiseResult? #[1] .CPU #[2] (.CUDA 0))
  expectMismatch (DeviceKind.pointwiseResult? #[0] .CPU #[0] .MPS)
  expectMismatch (DeviceKind.pointwiseResult? #[2] .CPU #[] (.CUDA 0))
  expectMismatch (DeviceKind.pointwiseResult? #[] .MPS #[2] .CPU)
  expectMismatch (DeviceKind.pointwiseResult? #[] (.CUDA 0) #[] (.CUDA 1))
  expectMismatch (DeviceKind.pointwiseResult? #[] .MPS #[] (.CUDA 0))

@[test] def testStrictDeviceKindMetadata : IO Unit := do
  for device in #[DeviceKind.CPU, .MPS, .CUDA 0, .CUDA 1] do
    expectDeviceKind (DeviceKind.sameResult? device device) device
  for (left, right) in #[(DeviceKind.CPU, DeviceKind.CUDA 0), (.CPU, .MPS),
      (.CUDA 0, .CUDA 1), (.CUDA 0, .MPS)] do
    expectMismatch (DeviceKind.sameResult? left right)
    expectMismatch (DeviceKind.sameResult? right left)

/-- `where` has a separate rule: every scalar, including an accelerator scalar,
can move to the first non-CPU operand's device. The operand order matters. -/
@[test] def testSelectionDeviceKindMetadata : IO Unit := do
  expectDeviceKind (DeviceKind.selectionResult? #[] .CPU #[] .CPU #[] .CPU) .CPU
  for device in #[DeviceKind.MPS, .CUDA 0, .CUDA 1] do
    expectDeviceKind (DeviceKind.selectionResult? #[2] device #[2] device #[2] device) device
    expectDeviceKind (DeviceKind.selectionResult? #[] .CPU #[2] device #[2] device) device
    expectDeviceKind (DeviceKind.selectionResult? #[2] device #[] .CPU #[2] device) device
    expectDeviceKind (DeviceKind.selectionResult? #[2] device #[2] device #[] .CPU) device
    expectDeviceKind (DeviceKind.selectionResult? #[] device #[] .CPU #[] .CPU) device
    expectDeviceKind (DeviceKind.selectionResult? #[] .CPU #[] device #[] .CPU) device
    expectDeviceKind (DeviceKind.selectionResult? #[] .CPU #[] .CPU #[] device) device
  -- Different accelerator indices are allowed for scalar relocation by where,
  -- although the same pair is invalid for ordinary binary pointwise arithmetic.
  expectDeviceKind (DeviceKind.selectionResult? #[] (.CUDA 0) #[] (.CUDA 1) #[] .CPU) (.CUDA 0)
  expectDeviceKind (DeviceKind.selectionResult? #[] .CPU #[] (.CUDA 1) #[] (.CUDA 0)) (.CUDA 1)
  expectDeviceKind (DeviceKind.selectionResult? #[] .MPS #[] (.CUDA 0) #[] .CPU) .MPS
  expectMismatch (DeviceKind.pointwiseResult? #[] (.CUDA 0) #[] (.CUDA 1))
  -- Non-scalars cannot relocate, including a one-element vector.
  expectMismatch (DeviceKind.selectionResult? #[2] (.CUDA 0) #[2] (.CUDA 1) #[2] (.CUDA 0))
  expectMismatch (DeviceKind.selectionResult? #[2] .CPU #[2] (.CUDA 0) #[2] (.CUDA 0))
  expectMismatch (DeviceKind.selectionResult? #[2] .MPS #[1] .CPU #[2] .MPS)
  expectMismatch (DeviceKind.selectionResult? #[] (.CUDA 0) #[2] (.CUDA 1) #[] .CPU)

@[test] def testCPUScalarPointwiseValues : IO Unit := do
  let scalar := (Tensor.full #[] 2).to .CPU
  let vector := (Tensor.fromFloatArray #[3, 5]).to .CPU
  LeanTest.assertEqual (← (scalar.addB vector).toFloatArray) #[5, 7]
  LeanTest.assertEqual (← (vector.mulB scalar).toFloatArray) #[6, 10]
  -- Independently validate the native CPU metadata.
  let cpuTensor := Tensor.fromFloatArray #[7, 11]
  match Tensor.checkPointwise cpuTensor vector with
  | .ok () => pure ()
  | .error message => LeanTest.fail message
  match Tensor.checkSameDevice cpuTensor vector with
  | .ok () => pure ()
  | .error message => LeanTest.fail message
  let condition := Tensor.eqScalar cpuTensor 7
  match Tensor.checkSelection condition cpuTensor vector with
  | .ok () => pure ()
  | .error message => LeanTest.fail message

end Tests.TestDeviceCompatibility
