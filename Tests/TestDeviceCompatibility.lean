import LeanTest
import Tyr.Typed.Ops

open torch

namespace Tests.TestDeviceCompatibility

-- Known placement conflicts must fail before allocating a native tensor.
example (_cpu : Tensor ⟨#[2], .Float32, .CPU⟩)
    (_cuda : Tensor ⟨#[2], .Float32, (.CUDA 0)⟩) : True := by
  fail_if_success have _ := _cpu.addB _cuda
  fail_if_success have _ := _cuda.subB _cpu
  fail_if_success have _ := _cpu.mulB _cuda
  fail_if_success have _ := _cuda.divB _cpu
  trivial

example (_cuda0 : Tensor ⟨#[2], .Float32, (.CUDA 0)⟩)
    (_cuda1 : Tensor ⟨#[2], .Float32, (.CUDA 1)⟩)
    (_mps : Tensor ⟨#[2], .Float32, .MPS⟩) : True := by
  fail_if_success have _ := _cuda0.addB _cuda1
  fail_if_success have _ := _cuda0.mulB _mps
  trivial

-- Rank zero is significant: CPU vectors and accelerator scalars cannot use
-- the exception, nor can scalars from two different accelerator devices.
example (_cpuVector : Tensor ⟨#[1], .Float32, .CPU⟩)
    (_cudaScalar : Tensor ⟨#[], .Float32, (.CUDA 0)⟩)
    (_mpsScalar : Tensor ⟨#[], .Float32, .MPS⟩) : True := by
  fail_if_success have _ := _cpuVector.addB _cudaScalar
  fail_if_success have _ := _mpsScalar.addB _cpuVector
  fail_if_success have _ := _cudaScalar.addB _mpsScalar
  trivial

example (cpu : Tensor ⟨#[], .Float32, .CPU⟩)
    (cuda : Tensor ⟨#[2], .Float32, (.CUDA 0)⟩) :
    Tensor ⟨#[2], .Float32, (.CUDA 0)⟩ := cpu.addB cuda

example (cpu : Tensor ⟨#[], .Float32, .CPU⟩)
    (cuda : Tensor ⟨#[2], .Float32, (.CUDA 0)⟩) :
    Tensor ⟨#[2], .Float32, (.CUDA 0)⟩ := cuda.mulB cpu

example (cpu : Tensor ⟨#[], .Float32, .CPU⟩)
    (mps : Tensor ⟨#[], .Float32, .MPS⟩) :
    Tensor ⟨#[], .Float32, .MPS⟩ := mps.addB cpu

example {index : UInt64} (cpu : Tensor ⟨#[], .Float32, .CPU⟩)
    (cuda : Tensor ⟨#[], .Float32, (.CUDA index)⟩) :
    Tensor ⟨#[], .Float32, (.CUDA index)⟩ := cuda.addB cpu

example {index : UInt64} (left right : Tensor ⟨#[], .Float32, (.CUDA index)⟩) :
    Tensor ⟨#[], .Float32, (.CUDA index)⟩ := left.addB right

-- Shared placement is valid even when its Device value is selected at runtime.
example {leftShape rightShape : Shape} {leftDType rightDType : DType}
    {device : Device} : Device.PointwiseCompatible
    ⟨leftShape, leftDType, device⟩ ⟨rightShape, rightDType, device⟩ := inferInstance
example {spec : StaticSpec} : Device.PointwiseCompatible spec spec := inferInstance

example : True := by
  fail_if_success
    have _ : Device.SameDeviceCompatible
      (.CPU) ((.CUDA 0)) := inferInstance
  fail_if_success
    have _ : Device.SameDeviceCompatible
      ((.CUDA 0)) ((.CUDA 1)) := inferInstance
  fail_if_success
    have _ : Device.PointwiseCompatible
      ⟨#[1], .Float32, .CPU⟩ ⟨#[], .Float32, .MPS⟩ := inferInstance
  trivial

private def expectDevice (result : Except String Device) (expected : Device) : IO Unit := do
  match result with
  | .ok actual =>
      LeanTest.assertTrue (actual == expected)
        s!"Expected device {expected.render}, got {actual.render}"
  | .error message => LeanTest.fail message

private def expectMismatch (result : Except String Device) : IO Unit := do
  match result with
  | .ok device => LeanTest.fail s!"Unexpectedly accepted device {device.render}"
  | .error _ => pure ()

/-- These are metadata checks, not accelerator execution tests. No CUDA or MPS
tensor is allocated by this test. -/
@[test] def testPointwiseDeviceMetadata : IO Unit := do
  for device in #[Device.CPU, .MPS, .CUDA 0, .CUDA 1] do
    expectDevice (Device.pointwiseResult? #[2, 1] device #[3] device) device
    expectDevice (Device.pointwiseResult? #[] .CPU #[2] device) device
    expectDevice (Device.pointwiseResult? #[2] device #[] .CPU) device
    expectDevice (Device.pointwiseResult? #[] .CPU #[] device) device
    expectDevice (Device.pointwiseResult? #[] device #[] .CPU) device
  for (left, right) in #[(Device.CPU, Device.CUDA 0), (.CPU, .MPS),
      (.CUDA 0, .CUDA 1), (.CUDA 0, .MPS)] do
    expectMismatch (Device.pointwiseResult? #[2] left #[2] right)
    expectMismatch (Device.pointwiseResult? #[2] right #[2] left)
  -- A size-one or empty CPU vector is still not a zero-dimensional scalar.
  expectMismatch (Device.pointwiseResult? #[1] .CPU #[2] (.CUDA 0))
  expectMismatch (Device.pointwiseResult? #[0] .CPU #[0] .MPS)
  expectMismatch (Device.pointwiseResult? #[2] .CPU #[] (.CUDA 0))
  expectMismatch (Device.pointwiseResult? #[] .MPS #[2] .CPU)
  expectMismatch (Device.pointwiseResult? #[] (.CUDA 0) #[] (.CUDA 1))
  expectMismatch (Device.pointwiseResult? #[] .MPS #[] (.CUDA 0))

@[test] def testStrictDeviceMetadata : IO Unit := do
  for device in #[Device.CPU, .MPS, .CUDA 0, .CUDA 1] do
    expectDevice (Device.sameResult? device device) device
  for (left, right) in #[(Device.CPU, Device.CUDA 0), (.CPU, .MPS),
      (.CUDA 0, .CUDA 1), (.CUDA 0, .MPS)] do
    expectMismatch (Device.sameResult? left right)
    expectMismatch (Device.sameResult? right left)

/-- `where` has a separate rule: every scalar, including an accelerator scalar,
can move to the first non-CPU operand's device. The operand order matters. -/
@[test] def testSelectionDeviceMetadata : IO Unit := do
  expectDevice (Device.selectionResult? #[] .CPU #[] .CPU #[] .CPU) .CPU
  for device in #[Device.MPS, .CUDA 0, .CUDA 1] do
    expectDevice (Device.selectionResult? #[2] device #[2] device #[2] device) device
    expectDevice (Device.selectionResult? #[] .CPU #[2] device #[2] device) device
    expectDevice (Device.selectionResult? #[2] device #[] .CPU #[2] device) device
    expectDevice (Device.selectionResult? #[2] device #[2] device #[] .CPU) device
    expectDevice (Device.selectionResult? #[] device #[] .CPU #[] .CPU) device
    expectDevice (Device.selectionResult? #[] .CPU #[] device #[] .CPU) device
    expectDevice (Device.selectionResult? #[] .CPU #[] .CPU #[] device) device
  -- Different accelerator indices are allowed for scalar relocation by where,
  -- although the same pair is invalid for ordinary binary pointwise arithmetic.
  expectDevice (Device.selectionResult? #[] (.CUDA 0) #[] (.CUDA 1) #[] .CPU) (.CUDA 0)
  expectDevice (Device.selectionResult? #[] .CPU #[] (.CUDA 1) #[] (.CUDA 0)) (.CUDA 1)
  expectDevice (Device.selectionResult? #[] .MPS #[] (.CUDA 0) #[] .CPU) .MPS
  expectMismatch (Device.pointwiseResult? #[] (.CUDA 0) #[] (.CUDA 1))
  -- Non-scalars cannot relocate, including a one-element vector.
  expectMismatch (Device.selectionResult? #[2] (.CUDA 0) #[2] (.CUDA 1) #[2] (.CUDA 0))
  expectMismatch (Device.selectionResult? #[2] .CPU #[2] (.CUDA 0) #[2] (.CUDA 0))
  expectMismatch (Device.selectionResult? #[2] .MPS #[1] .CPU #[2] .MPS)
  expectMismatch (Device.selectionResult? #[] (.CUDA 0) #[2] (.CUDA 1) #[] .CPU)

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
